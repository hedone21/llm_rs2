mod forward;
mod forward_gen;

use crate::backend::cpu::CpuBackend;
use crate::core::backend::Backend;
use crate::core::buffer::Buffer;
use crate::core::buffer::DType;
use crate::core::kv_cache::{KVCache, KVCacheOps};
use crate::core::memory::Memory;
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use crate::memory::galloc::Galloc;
use anyhow::Result;
use rayon::prelude::*;
use std::sync::Arc;

use crate::buffer::shared_buffer::SharedBuffer;
use crate::layers::tensor_partition::PartitionContext;

// Re-export OpProfiler from its canonical location for backward compatibility.
pub use crate::profile::ops::OpProfiler;

/// Update KV cache with K/V tensors, handling GPU→CPU readback when needed.
///
/// GPU-only tensors (`as_ptr()` returns null, e.g. NVIDIA UnifiedBuffer) are
/// read back to CPU when the cache does not support direct GPU buffer access
/// (e.g. `KiviCache`). Regular `KVCache` handles GPU tensors internally via
/// `backend.copy_slice`, so no readback occurs for the common case.
pub(super) fn update_kv_cache<C: KVCacheOps>(
    kv_cache: &mut C,
    k: &Tensor,
    v: &Tensor,
    backend: &Arc<dyn Backend>,
) -> Result<()> {
    // Fast path: tensors have CPU pointers (ARM shared memory or CPU backend)
    if !k.as_ptr().is_null() {
        return kv_cache.update(k, v);
    }

    // GPU-only tensors: check if cache can handle them directly.
    // KVCache.get_buffers_mut() returns Some (GPU copy_slice works);
    // KiviCache returns None (CPU-only, needs readback).
    {
        let has_gpu_buffers = kv_cache.get_buffers_mut().is_some();
        if has_gpu_buffers {
            return kv_cache.update(k, v);
        }
    }

    // CPU-only cache (e.g. KiviCache): read GPU data back to CPU
    let k_cpu = gpu_readback(k, backend)?;
    let v_cpu = gpu_readback(v, backend)?;
    kv_cache.update(&k_cpu, &v_cpu)
}

/// Read a GPU tensor back to a CPU SharedBuffer tensor.
fn gpu_readback(tensor: &Tensor, backend: &Arc<dyn Backend>) -> Result<Tensor> {
    let byte_size = tensor.buffer().size();
    let cpu_buf = Arc::new(SharedBuffer::new(byte_size, tensor.dtype()));
    unsafe {
        let dst = std::slice::from_raw_parts_mut(cpu_buf.as_mut_ptr(), byte_size);
        backend.read_buffer(tensor, dst)?;
    }
    Ok(Tensor::new(
        tensor.shape().clone(),
        cpu_buf,
        Arc::new(CpuBackend::new()),
    ))
}

// --- x86_64 AVX2 SIMD helpers for attention ---

/// Dot product: sum(a[i] * b[i]) for i in 0..len, using AVX2+FMA.
/// head_dim=64 → 8 AVX2 iterations (4x unrolled = 2 outer iterations).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) unsafe fn dot_f32_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    unsafe {
        use std::arch::x86_64::*;
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;

        // 4x unrolled: 32 floats per iteration
        while i + 32 <= len {
            let a0 = _mm256_loadu_ps(a.add(i));
            let b0 = _mm256_loadu_ps(b.add(i));
            sum = _mm256_fmadd_ps(a0, b0, sum);

            let a1 = _mm256_loadu_ps(a.add(i + 8));
            let b1 = _mm256_loadu_ps(b.add(i + 8));
            sum = _mm256_fmadd_ps(a1, b1, sum);

            let a2 = _mm256_loadu_ps(a.add(i + 16));
            let b2 = _mm256_loadu_ps(b.add(i + 16));
            sum = _mm256_fmadd_ps(a2, b2, sum);

            let a3 = _mm256_loadu_ps(a.add(i + 24));
            let b3 = _mm256_loadu_ps(b.add(i + 24));
            sum = _mm256_fmadd_ps(a3, b3, sum);

            i += 32;
        }

        while i + 8 <= len {
            let a0 = _mm256_loadu_ps(a.add(i));
            let b0 = _mm256_loadu_ps(b.add(i));
            sum = _mm256_fmadd_ps(a0, b0, sum);
            i += 8;
        }

        // Horizontal sum: 256→128→scalar
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result128 = _mm_add_ss(sums, shuf2);
        let mut result = _mm_cvtss_f32(result128);

        // Scalar tail
        while i < len {
            result += *a.add(i) * *b.add(i);
            i += 1;
        }

        result
    }
}

/// Weighted accumulation: out[i] += weight * v[i] for i in 0..len, using AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) unsafe fn weighted_accum_f32_avx2(
    out: *mut f32,
    v: *const f32,
    weight: f32,
    len: usize,
) {
    unsafe {
        use std::arch::x86_64::*;
        let w = _mm256_set1_ps(weight);
        let mut i = 0;

        // 4x unrolled: 32 floats per iteration
        while i + 32 <= len {
            let o0 = _mm256_loadu_ps(out.add(i));
            let v0 = _mm256_loadu_ps(v.add(i));
            _mm256_storeu_ps(out.add(i), _mm256_fmadd_ps(w, v0, o0));

            let o1 = _mm256_loadu_ps(out.add(i + 8));
            let v1 = _mm256_loadu_ps(v.add(i + 8));
            _mm256_storeu_ps(out.add(i + 8), _mm256_fmadd_ps(w, v1, o1));

            let o2 = _mm256_loadu_ps(out.add(i + 16));
            let v2 = _mm256_loadu_ps(v.add(i + 16));
            _mm256_storeu_ps(out.add(i + 16), _mm256_fmadd_ps(w, v2, o2));

            let o3 = _mm256_loadu_ps(out.add(i + 24));
            let v3 = _mm256_loadu_ps(v.add(i + 24));
            _mm256_storeu_ps(out.add(i + 24), _mm256_fmadd_ps(w, v3, o3));

            i += 32;
        }

        while i + 8 <= len {
            let o0 = _mm256_loadu_ps(out.add(i));
            let v0 = _mm256_loadu_ps(v.add(i));
            _mm256_storeu_ps(out.add(i), _mm256_fmadd_ps(w, v0, o0));
            i += 8;
        }

        // Scalar tail
        while i < len {
            *out.add(i) += weight * *v.add(i);
            i += 1;
        }
    }
}

pub struct QkvBias {
    pub bq: Tensor,
    pub bk: Tensor,
    pub bv: Tensor,
}

pub struct TransformerLayer {
    // Attention
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,

    // MLP
    pub w_gate: Tensor, // silu_mul gate
    pub w_up: Tensor,
    pub w_down: Tensor,

    // Norms
    /// Pre-attention norm (all architectures).
    pub attention_norm: Tensor,
    /// Llama/Qwen2: pre-FFN norm. Gemma3: post-attention norm.
    pub ffn_norm: Tensor,

    // Optional QKV bias (e.g. for Qwen, Phi models)
    pub qkv_bias: Option<QkvBias>,

    // Gemma3-only optional fields
    /// QK-Norm: per-head RMSNorm weight for Q [n_heads_q * head_dim] (Gemma3).
    pub q_norm: Option<Tensor>,
    /// QK-Norm: per-head RMSNorm weight for K [n_heads_kv * head_dim] (Gemma3).
    pub k_norm: Option<Tensor>,
    /// Pre-FFN norm (Gemma3: pre_feedforward_layernorm). None for Llama/Qwen2.
    pub pre_ffn_norm: Option<Tensor>,
    /// Post-FFN norm (Gemma3: post_feedforward_layernorm). None for Llama/Qwen2.
    pub post_ffn_norm: Option<Tensor>,

    /// Tensor partition context for CPU-GPU cooperative FFN inference.
    /// When set, FFN gate/up matmuls are split between GPU and CPU backends.
    /// None when tensor partition is disabled (default).
    pub partition_ctx: Option<PartitionContext>,
}

impl TransformerLayer {
    pub fn forward<C: KVCacheOps>(&self, args: LayerForwardArgs<C>) -> Result<()> {
        let skip_attn = args.skip_attn;
        let skip_mlp = args.skip_mlp;

        // SWIFT: if both sub-layers are skipped, early return (identity)
        if skip_attn && skip_mlp {
            return Ok(());
        }

        let x = args.x;
        let kv_cache = args.kv_cache;
        let start_pos = args.start_pos;
        let backend = args.backend;
        let memory = args.memory;
        let rms_norm_eps = args.rms_norm_eps;
        let rope_theta = args.rope_theta;
        let workspace = args.workspace;

        let batch_size = x.shape().dims()[0];
        let seq_len = x.shape().dims()[1];
        let dim = x.shape().dims()[2];
        let head_dim = args.head_dim;

        let need_scores = args.need_scores;

        if seq_len == 1
            && let Some(ws) = workspace
        {
            return self.forward_gen(ForwardGenArgs {
                x,
                kv_cache,
                start_pos,
                backend,
                memory,
                ws,
                rms_norm_eps,
                rope_theta,
                need_scores,
                head_dim,
                profiler: args.profiler,
                skip_attn,
                skip_mlp,
                rms_norm_add_unit: args.rms_norm_add_unit,
                use_gelu_tanh: args.use_gelu_tanh,
                is_local_attn: args.is_local_attn,
                local_attn_window: args.local_attn_window,
                layer_idx: args.layer_id,
                is_last_layer: args.is_last_layer,
            });
        }

        self.forward_prefill(
            x,
            kv_cache,
            start_pos,
            backend,
            memory,
            rms_norm_eps,
            rope_theta,
            need_scores,
            head_dim,
            batch_size,
            seq_len,
            dim,
            skip_attn,
            skip_mlp,
            args.rms_norm_add_unit,
            args.use_gelu_tanh,
            args.is_local_attn,
            args.local_attn_window,
            None, // prefill_ws: passed separately via forward_into dispatch
            0,    // layer_idx: unknown in this path (no variance collector)
            None, // variance_collector: not available here
            args.profiler.map(|p| &mut p.prefill),
        )
    }

    fn alloc_temp(
        &self,
        shape: Vec<usize>,
        memory: &dyn Memory,
        backend: &Arc<dyn Backend>,
    ) -> Result<Tensor> {
        let size: usize = shape.iter().product();
        let buf = memory.alloc(size * 4, DType::F32)?;
        Ok(Tensor::new(Shape::new(shape), buf, backend.clone()))
    }
}

pub struct ForwardGenArgs<'a, C: KVCacheOps = KVCache> {
    pub x: &'a mut Tensor,
    pub kv_cache: &'a mut C,
    pub start_pos: usize,
    pub backend: &'a Arc<dyn Backend>,
    pub memory: &'a dyn Memory,
    pub ws: &'a mut crate::layers::workspace::LayerWorkspace,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// When true, compute attention scores into ws.scores even for non-F32 KV cache.
    /// Required for H2O/H2O+ score accumulation with Q4_0/F16 KV cache.
    pub need_scores: bool,
    pub head_dim: usize,
    /// Optional per-op profiler for timing breakdown.
    pub profiler: Option<&'a mut OpProfiler>,
    /// SWIFT: skip attention sub-layer (identity pass).
    pub skip_attn: bool,
    /// SWIFT: skip MLP/FFN sub-layer (identity pass).
    pub skip_mlp: bool,
    /// Gemma3: true → `x * (1 + w) / rms(x)`, false → `x * w / rms(x)` (Llama/Qwen2).
    pub rms_norm_add_unit: bool,
    /// Gemma3: true → GELU_tanh activation, false → SiLU (Llama/Qwen2).
    pub use_gelu_tanh: bool,
    /// Gemma3: whether this layer uses local (sliding window) attention.
    pub is_local_attn: Option<bool>,
    /// Gemma3: local attention window size (sliding_window value).
    pub local_attn_window: Option<usize>,
    /// 0-based layer index. Used by `LLMRS_PARTITION_FUSED_MERGE` to decide
    /// whether the layer should consume the previous layer's partition
    /// partial buffers via `fused_norm_merge` (layer_idx > 0).
    pub layer_idx: usize,
    /// True when this is the final transformer layer. Used by
    /// `LLMRS_PARTITION_FUSED_MERGE` to keep the legacy merge + residual
    /// add path so the final norm + lm_head see the fully accumulated `x`.
    pub is_last_layer: bool,
}

pub struct LayerForwardArgs<'a, C: KVCacheOps = KVCache> {
    pub x: &'a mut Tensor,
    pub kv_cache: &'a mut C,
    pub start_pos: usize,
    pub backend: &'a Arc<dyn Backend>,
    pub memory: &'a dyn Memory,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub workspace: Option<&'a mut crate::layers::workspace::LayerWorkspace>,
    pub need_scores: bool,
    pub head_dim: usize,
    /// Optional per-op profiler for timing breakdown.
    pub profiler: Option<&'a mut OpProfiler>,
    /// Layer index (0-based). Used for SWIFT layer skip.
    pub layer_id: usize,
    /// If true, skip the attention sub-layer (identity pass).
    pub skip_attn: bool,
    /// If true, skip the MLP/FFN sub-layer (identity pass).
    pub skip_mlp: bool,
    /// Gemma3: true → `x * (1 + w) / rms(x)`, false → `x * w / rms(x)` (Llama/Qwen2).
    pub rms_norm_add_unit: bool,
    /// Gemma3: true → GELU_tanh activation, false → SiLU (Llama/Qwen2).
    pub use_gelu_tanh: bool,
    /// Gemma3: whether this layer uses local (sliding window) attention.
    pub is_local_attn: Option<bool>,
    /// Gemma3: local attention window size (sliding_window value).
    pub local_attn_window: Option<usize>,
    /// True when this is the final transformer layer. Consumed by
    /// `forward_gen` under `LLMRS_PARTITION_FUSED_MERGE` to keep the legacy
    /// post-FFN residual add on the last layer so the final norm + lm_head
    /// see the fully accumulated `x`.
    pub is_last_layer: bool,
}

// OpProfiler has been moved to crate::profile::ops.
// Re-exported via `pub use crate::profile::ops::OpProfiler;` at the top of this file.

// ═══════════════════════════════════════════════════════════════════
// Phase 1: Verify attention scores are post-softmax probabilities
// ═══════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::shared_buffer::SharedBuffer;

    /// Replicate the softmax computation used in forward_gen (F32 inline path)
    /// to verify the mathematical property: sum(softmax(Q·K^T / sqrt(d))) ≈ 1.0.
    ///
    /// This tests the SAME algorithm used in lines 548-636 of forward_gen().
    #[test]
    fn test_inline_softmax_produces_valid_probabilities() {
        let head_dim = 64;
        let cache_seq_len = 8;
        let n_heads_q = 4;
        let n_heads_kv = 2;
        let n_rep = n_heads_q / n_heads_kv;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Create Q: [n_heads_q, head_dim] with random-ish values
        let mut q_data = vec![0.0f32; n_heads_q * head_dim];
        for (i, v) in q_data.iter_mut().enumerate() {
            *v = ((i as f32 * 0.1).sin()) * 0.5;
        }

        // Create K cache: [cache_seq_len, n_heads_kv, head_dim] (SeqMajor)
        let mut k_data = vec![0.0f32; cache_seq_len * n_heads_kv * head_dim];
        for (i, v) in k_data.iter_mut().enumerate() {
            *v = ((i as f32 * 0.07).cos()) * 0.5;
        }

        let stride = cache_seq_len; // minimal stride = cache_seq_len
        let mut scores = vec![0.0f32; n_heads_q * stride];

        // Replicate the forward_gen inline path (non-NEON scalar path)
        for h in 0..n_heads_q {
            let kv_h = h / n_rep;
            let q_off = h * head_dim;
            let scores_h = &mut scores[h * stride..(h + 1) * stride];

            // Step 1: Q * K^T * scale
            for t in 0..cache_seq_len {
                let k_off = (t * n_heads_kv + kv_h) * head_dim; // SeqMajor
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_data[q_off + d] * k_data[k_off + d];
                }
                scores_h[t] = dot * scale;
            }

            // Step 2: Softmax (same algorithm as forward_gen lines 616-636)
            let active = &mut scores_h[..cache_seq_len];
            let max_val = active.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0f32;
            for s in active.iter_mut() {
                *s = (*s - max_val).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0 / sum_exp;
            for s in active.iter_mut() {
                *s *= inv_sum;
            }
        }

        // VERIFY: each head's scores sum to ~1.0
        for h in 0..n_heads_q {
            let head_scores = &scores[h * stride..h * stride + cache_seq_len];
            let sum: f32 = head_scores.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Head {} score sum = {} (expected ~1.0)",
                h,
                sum
            );
            // All values in [0, 1]
            for (t, &s) in head_scores.iter().enumerate() {
                assert!(
                    s >= 0.0 && s <= 1.0,
                    "Head {} token {} score = {} (must be in [0,1])",
                    h,
                    t,
                    s
                );
            }
        }
    }

    /// Verify compute_attention_scores() for F16 KV cache produces post-softmax.
    #[test]
    fn test_compute_attention_scores_f16_post_softmax() {
        let backend = Arc::new(CpuBackend::new());
        let head_dim = 64;
        let n_heads_q = 4;
        let n_heads_kv = 2;
        let capacity = 16;
        let cache_seq_len = 8;

        // Q tensor: [1, 1, n_heads_q * head_dim] = [1, 1, 256]
        let q_buf = Arc::new(SharedBuffer::new(n_heads_q * head_dim * 4, DType::F32));
        let q = Tensor::new(
            Shape::new(vec![1, 1, n_heads_q * head_dim]),
            q_buf,
            backend.clone(),
        );
        // Fill Q with values
        unsafe {
            let q_slice =
                std::slice::from_raw_parts_mut(q.as_mut_ptr() as *mut f32, n_heads_q * head_dim);
            for (i, v) in q_slice.iter_mut().enumerate() {
                *v = ((i as f32 * 0.1).sin()) * 0.3;
            }
        }

        // K cache tensor: [1, capacity, n_heads_kv, head_dim] in F16
        let k_buf = Arc::new(SharedBuffer::new(
            capacity * n_heads_kv * head_dim * 2, // F16 = 2 bytes
            DType::F16,
        ));
        let k_cache = Tensor::new(
            Shape::new(vec![1, capacity, n_heads_kv, head_dim]),
            k_buf,
            backend.clone(),
        );
        // Fill K with F16 values
        unsafe {
            let k_slice = std::slice::from_raw_parts_mut(
                k_cache.as_mut_ptr() as *mut half::f16,
                capacity * n_heads_kv * head_dim,
            );
            for (i, v) in k_slice.iter_mut().enumerate() {
                *v = half::f16::from_f32(((i as f32 * 0.07).cos()) * 0.3);
            }
        }

        let mut scores = vec![0.0f32; n_heads_q * capacity];

        TransformerLayer::compute_attention_scores(
            &q,
            &k_cache,
            &mut scores,
            n_heads_q,
            n_heads_kv,
            head_dim,
            cache_seq_len,
            false, // SeqMajor
            capacity,
            &(backend as Arc<dyn Backend>),
        )
        .unwrap();

        // Verify post-softmax: sum ≈ 1.0 per head
        let stride = scores.len() / n_heads_q;
        for h in 0..n_heads_q {
            let head_scores = &scores[h * stride..h * stride + cache_seq_len];
            let sum: f32 = head_scores.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "F16 Head {} score sum = {} (expected ~1.0)",
                h,
                sum
            );
            for (t, &s) in head_scores.iter().enumerate() {
                assert!(
                    s >= 0.0 && s <= 1.0,
                    "F16 Head {} token {} score = {} (must be in [0,1])",
                    h,
                    t,
                    s
                );
            }
        }
    }

    /// Verify that scores fed to accumulate_layer() are genuine post-softmax.
    /// Simulate the full pipeline: compute scores → accumulate → verify.
    #[test]
    fn test_accumulator_receives_post_softmax_scores() {
        use crate::core::attention_scores::AttentionScoreAccumulator;

        let n_heads_q = 4;
        let cache_seq_len = 8;
        let stride = cache_seq_len;

        // Simulate post-softmax scores (manually constructed)
        let mut scores = vec![0.0f32; n_heads_q * stride];
        for h in 0..n_heads_q {
            let head_scores = &mut scores[h * stride..h * stride + cache_seq_len];
            // Create a valid probability distribution
            let raw: Vec<f32> = (0..cache_seq_len)
                .map(|t| ((t as f32 + h as f32) * 0.5).exp())
                .collect();
            let sum: f32 = raw.iter().sum();
            for (t, &r) in raw.iter().enumerate() {
                head_scores[t] = r / sum;
            }
        }

        // Verify input is valid softmax
        for h in 0..n_heads_q {
            let sum: f32 = scores[h * stride..h * stride + cache_seq_len].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        // Feed to accumulator
        let mut acc = AttentionScoreAccumulator::new(16, n_heads_q, 1, 0, 0.0);
        acc.set_active(true);
        acc.begin_step();
        acc.accumulate_layer(&scores, stride, cache_seq_len, n_heads_q, 0);
        acc.end_step();

        // Per-token importance = sum across heads of softmax probs
        // Each head sums to 1.0 → total per-token sum across all tokens = n_heads_q
        let imp = acc.importance_scores();
        let total: f32 = imp[..cache_seq_len].iter().sum();
        assert!(
            (total - n_heads_q as f32).abs() < 1e-4,
            "Total importance = {} (expected {} = n_heads_q * 1.0 per head)",
            total,
            n_heads_q
        );
    }

    /// GEMMA-2.1: effective_cache_len is clamped to window_size for local layers.
    ///
    /// When is_local_attn=Some(true) and local_attn_window=Some(ws), the decode path
    /// must use min(cache_seq_len, ws) as effective attention length.
    #[test]
    fn test_effective_cache_len_local() {
        // Verify the computation logic directly (mirrors forward_gen.rs lines)
        let test_cases = vec![
            // (cache_seq_len, is_local, window, expected_effective)
            (100usize, Some(true), Some(512usize), 100usize), // cache < window → full
            (600, Some(true), Some(512), 512),                // cache > window → clamped
            (512, Some(true), Some(512), 512),                // cache == window → exact
            (100, Some(false), Some(512), 100),               // global layer → full
            (600, None, Some(512), 600),                      // no local flag → full
            (600, Some(true), None, 600),                     // no window → full (usize::MAX)
        ];

        for (cache_seq_len, is_local_attn, local_attn_window, expected) in test_cases {
            let effective = if let Some(true) = is_local_attn {
                let window = local_attn_window.unwrap_or(usize::MAX);
                cache_seq_len.min(window)
            } else {
                cache_seq_len
            };
            let kv_start_pos = cache_seq_len - effective;

            assert_eq!(
                effective, expected,
                "cache={} is_local={:?} window={:?}: effective={} expected={}",
                cache_seq_len, is_local_attn, local_attn_window, effective, expected
            );
            // kv_start_pos must be consistent
            assert_eq!(
                kv_start_pos,
                cache_seq_len - expected,
                "kv_start_pos mismatch for cache={} effective={}",
                cache_seq_len,
                effective
            );
        }
    }

    /// GEMMA-2.2: is_local_attn=None (Llama/Qwen2) leaves effective_cache_len unchanged.
    #[test]
    fn test_non_local_attn_unchanged() {
        let cache_seq_len: usize = 42;
        let is_local_attn: Option<bool> = None;
        let local_attn_window: Option<usize> = None;

        let effective = if let Some(true) = is_local_attn {
            let window = local_attn_window.unwrap_or(usize::MAX);
            cache_seq_len.min(window)
        } else {
            cache_seq_len
        };

        assert_eq!(
            effective, cache_seq_len,
            "Non-local layer must use full cache_seq_len"
        );
        assert_eq!(cache_seq_len - effective, 0, "kv_start_pos must be 0");
    }
}
