//! `forward_gen_fmt` — `forward_gen` 의 KVCacheFormat trait-object fork (Phase α-K substep 3c).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §9.1 (3c) — decode fallback 의 KV write +
//! attention 을 `Arc<dyn KVCacheFormat>` 로 flip. ADR-0001 (갈래 2: Generic → trait object).
//!
//! **branch-by-abstraction, additive**: 기존 `forward_gen<C: KVCacheOps>` 를 1바이트도 안 건드린다.
//! 본 fork 는 **decode 라이브 경로**(partition off, fused off)만 정확히 재현하고, dead branch(partition
//! / fused QKV·FFN / kivi_native / F32 batch-scatter / CPU inline NEON attention)는 전부 생략한다
//! (census 확정 — 게이트에서 미진입). 두 지점만 위임:
//!   - KV update 블록(forward_gen.rs:332-386)  → `fmt.write_kv(&k_rope, &ws.v, backend)` (3a/3b 흡수)
//!   - attention dispatch(forward_gen.rs:463-1068) → `fmt.attention_into(...)` (Q4-GPU-fallback 포함)
//!
//! **bit-identical 범위 (중요)**: 공유 골격(norm/QKV matmul/RoPE/O-proj/FFN/residual)은 forward_gen 의
//! 라이브 arm 과 같은 backend 호출. attention 위임은 **F16 KV / Q4_0 KV / F32-device-only(null host ptr)
//! 에서만 bit-identical** — 이 셋은 forward_gen 도 `backend.attention_gen`(또는 Q4 fallback)로 가기
//! 때문. **⚠️ F32 KV + host-mapped 버퍼**(`--opencl-rpcmem` non-null / `--zero-copy` mapped / CPU
//! backend)는 forward_gen 이 inline-NEON attention(forward_gen.rs:554-1068, kv_start_pos 적용)을 타는
//! 반면 `attention_into` 는 무조건 `backend.attention_gen` 위임 → FP 누산 순서 상이 **NOT bit-identical**.
//! 따라서 (3c) device 게이트는 **F16/Q4_0 KV**(default=F16) 또는 F32-device-only 만 대상으로 한다.
//! 생략한 instrumentation(prof/op-trace/set_label)은 수치 무관(게이트는 `--profile`·env 미사용).
//! `set_attn_scores`/`needs_attn_scores` OR 항 생략은 StandardKVCache trait default(no-op/false)라 안전
//! (KIVI 전용). score_offset/effective_cache_len 은 forward_gen.rs:404 와 동일 식으로 재현. GPU score
//! acc layer_idx routing 도 동일 위치 보존.

use super::*;
use crate::format::{AttnDims, KVCacheFormat};

/// `forward_gen_fmt` 인자 — `ForwardGenArgs` 의 `kv_cache: &mut C` 만 `fmt: &Arc<dyn KVCacheFormat>`
/// 로 교체. profiler/memory/is_last_layer 는 fmt 경로에서 불요(instrumentation·partition_fused 전용)
/// 라 드롭.
pub(crate) struct ForwardGenFmtArgs<'a> {
    pub x: &'a mut Tensor,
    pub fmt: &'a Arc<dyn KVCacheFormat>,
    pub start_pos: usize,
    pub backend: &'a Arc<dyn Backend>,
    pub ws: &'a mut crate::layers::workspace::LayerWorkspace,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// H2O/D2O score 누적이 필요하면 attention_into 에 scores 버퍼 전달.
    pub need_scores: bool,
    pub head_dim: usize,
    pub skip_attn: bool,
    pub skip_mlp: bool,
    /// Gemma3: true → `x * (1 + w) / rms(x)`, false → Llama/Qwen2.
    pub rms_norm_add_unit: bool,
    /// Gemma3: true → GELU_tanh, false → SiLU.
    pub use_gelu_tanh: bool,
    /// Gemma3: 이 layer 가 local(SWA) attention 인가.
    pub is_local_attn: Option<bool>,
    /// Gemma3: local attention window.
    pub local_attn_window: Option<usize>,
    pub layer_idx: usize,
}

impl TransformerLayer {
    /// `forward_gen` 의 trait-object fork (decode, seq_len=1). KV write + attention 만 fmt 위임.
    pub(crate) fn forward_gen_fmt(&self, args: ForwardGenFmtArgs) -> Result<()> {
        // SWIFT: 두 sub-layer 모두 skip 이면 identity (forward_gen.rs:26 동치).
        if args.skip_attn && args.skip_mlp {
            return Ok(());
        }

        let x = args.x;
        let fmt = args.fmt;
        let start_pos = args.start_pos;
        let backend = args.backend;
        let ws = args.ws;
        let rms_norm_eps = args.rms_norm_eps;
        let rope_theta = args.rope_theta;
        let rms_norm_add_unit = args.rms_norm_add_unit;
        let use_gelu_tanh = args.use_gelu_tanh;
        let head_dim = args.head_dim;
        let layer_idx = args.layer_idx;
        let batch_size = x.shape().dims()[0];
        let is_gpu = backend.is_gpu();

        // 1. Attention Norm — out-of-place: ws.residual = norm(x), x 보존(skip connection).
        backend.rms_norm_oop(
            x,
            &mut ws.residual,
            &self.attention_norm,
            rms_norm_eps,
            rms_norm_add_unit,
        )?;

        // 2. QKV projections (decode GPU 라이브 = matmul_transposed×3; fused QKV 는 aarch64 dead).
        crate::thread_pool::get_pool().begin_batch();
        backend.matmul_transposed(&ws.residual, &self.wq, &mut ws.q)?;
        backend.matmul_transposed(&ws.residual, &self.wk, &mut ws.k)?;
        backend.matmul_transposed(&ws.residual, &self.wv, &mut ws.v)?;
        crate::thread_pool::get_pool().end_batch();
        if is_gpu && std::env::var_os("LLMRS_DISABLE_FLUSH_QKV").is_none() {
            backend.flush()?;
        }

        // QKV bias (Qwen2 등).
        if let Some(ref bias) = self.qkv_bias {
            backend.add_row_bias(&mut ws.q, &bias.bq)?;
            backend.add_row_bias(&mut ws.k, &bias.bk)?;
            backend.add_row_bias(&mut ws.v, &bias.bv)?;
        }

        // 3. RoPE.
        let q_dim = self.wq.shape().dims()[0];
        let k_dim = self.wk.shape().dims()[0];
        let n_heads_q = q_dim / head_dim;
        let n_heads_kv = k_dim / head_dim;

        // Gemma3 QK-Norm: per-head RMSNorm on Q/K before RoPE.
        if let Some(ref q_norm_w) = self.q_norm {
            let total_q_heads = batch_size * n_heads_q;
            let saved_shape = ws.q.shape().clone();
            ws.q.reshape(Shape::new(vec![total_q_heads, head_dim]));
            backend.rms_norm(&mut ws.q, q_norm_w, rms_norm_eps, true)?;
            ws.q.reshape(saved_shape);
        }
        if let Some(ref k_norm_w) = self.k_norm {
            let total_k_heads = batch_size * n_heads_kv;
            let saved_shape = ws.k.shape().clone();
            ws.k.reshape(Shape::new(vec![total_k_heads, head_dim]));
            backend.rms_norm(&mut ws.k, k_norm_w, rms_norm_eps, true)?;
            ws.k.reshape(saved_shape);
        }

        let mut q_rope = Tensor::new(
            Shape::new(vec![batch_size, 1, n_heads_q, head_dim]),
            ws.q.buffer().clone(),
            backend.clone(),
        );
        let mut k_rope = Tensor::new(
            Shape::new(vec![batch_size, 1, n_heads_kv, head_dim]),
            ws.k.buffer().clone(),
            backend.clone(),
        );
        backend.rope_inplace(&mut q_rope, start_pos, rope_theta)?;
        backend.rope_inplace(&mut k_rope, start_pos, rope_theta)?;

        // 4. KV cache update → fmt.write_kv (3a/3b 흡수: GPU F16/F32 scatter / 비-F32 cast / F32 update).
        fmt.write_kv(&k_rope, &ws.v, backend.as_ref())?;

        // 5. Attention → fmt.attention_into (Q4-GPU-fallback 포함, 2단계에서 흡수).
        let cache_seq_len = fmt.current_pos();
        // Sliding window 는 is_local_attn==Some(true) 일 때만 적용 (forward_gen.rs:397 게이팅 동치).
        let window = if matches!(args.is_local_attn, Some(true)) {
            args.local_attn_window
        } else {
            None
        };
        let effective_cache_len = match window {
            Some(w) => cache_seq_len.min(w),
            None => cache_seq_len,
        };
        // score accumulator 의 ws.scores[t] → cache pos 매핑용 offset (forward_gen.rs:404 동일 식).
        ws.score_offset = cache_seq_len - effective_cache_len;

        // StandardKVCache 는 needs_attn_scores()=false (trait default) 라 OR 항 불요.
        let need_scores = args.need_scores;

        // GPU score acc layer_idx routing (forward_gen.rs:505-509) — base trait 불변, backend 핸들 경유.
        if let Some(gpu_acc) = backend.gpu_score_acc_mut()
            && gpu_acc.is_active()
        {
            gpu_acc.set_current_layer_idx(layer_idx);
        }

        fmt.attention_into(
            &q_rope,
            backend.as_ref(),
            &mut ws.out_attn,
            AttnDims { n_heads_q, window },
            if need_scores {
                Some(&mut ws.scores)
            } else {
                None
            },
        )?;
        // set_attn_scores(forward_gen.rs:1071) 는 StandardKVCache no-op(KIVI AWQE 전용) → 생략.

        // 6. Output projection.
        backend.matmul_transposed(&ws.out_attn, &self.wo, &mut ws.attn_out)?;

        // 7+8. Post-attention residual + pre-FFN norm.
        if rms_norm_add_unit {
            // Gemma3: post-attn norm(ffn_norm) → fused add + pre_ffn_norm.
            backend.rms_norm(&mut ws.attn_out, &self.ffn_norm, rms_norm_eps, true)?;
            if let Some(ref pfn) = self.pre_ffn_norm {
                backend.add_rms_norm_oop(
                    x,
                    &ws.attn_out,
                    &mut ws.residual,
                    pfn,
                    rms_norm_eps,
                    true,
                )?;
            } else {
                backend.add_assign(x, &ws.attn_out)?;
                backend.copy_into(x, &mut ws.residual)?;
            }
        } else {
            // Llama/Qwen2: fused add + norm.
            backend.add_rms_norm_oop(
                x,
                &ws.attn_out,
                &mut ws.residual,
                &self.ffn_norm,
                rms_norm_eps,
                false,
            )?;
        }

        // 9. FFN gate + up (decode GPU 라이브 = else arm; fused NEON 은 dead).
        crate::thread_pool::get_pool().begin_batch();
        if !use_gelu_tanh {
            backend.matmul_ffn_gate_up_silu(
                &ws.residual,
                &self.w_gate,
                &self.w_up,
                &mut ws.gate,
                &mut ws.up,
            )?;
        } else {
            backend.matmul_transposed(&ws.residual, &self.w_gate, &mut ws.gate)?;
            backend.matmul_transposed(&ws.residual, &self.w_up, &mut ws.up)?;
        }
        crate::thread_pool::get_pool().end_batch();
        if is_gpu && std::env::var_os("LLMRS_DISABLE_FLUSH_FFN").is_none() {
            backend.flush()?;
        }

        // silu_mul(GELU 경로만 별도 activation) + down matmul. partition 미적용이라 항상 실행.
        if use_gelu_tanh {
            backend.gelu_tanh_mul(&mut ws.gate, &ws.up)?;
        }
        backend.matmul_transposed(&ws.gate, &self.w_down, &mut ws.down)?;

        // 10. Residual 2 — FFN 결과를 x 에 누적.
        if let Some(ref pfn) = self.post_ffn_norm {
            backend.rms_norm(&mut ws.down, pfn, rms_norm_eps, true)?;
        }
        backend.add_assign(x, &ws.down)?;

        Ok(())
    }
}
