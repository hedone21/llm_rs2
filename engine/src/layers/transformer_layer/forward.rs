use super::*;

use std::collections::HashSet;
use std::sync::Mutex;

/// Per-run deduplication set for GPU→CPU attention fallback warnings.
/// Same (dtype, head_dim, reason) tuple is logged to stderr only once per process.
static FALLBACK_WARNED: Mutex<Option<HashSet<(String, usize, &'static str)>>> = Mutex::new(None);

/// Emit a one-time stderr warning for a GPU prefill → CPU attention fallback.
/// Always emits the OpProfiler event (if profiler is Some) regardless of dedup state.
fn warn_gpu_fallback_once(
    kv_dtype: crate::core::buffer::DType,
    head_dim: usize,
    reason: &'static str,
    profiler: Option<&mut crate::profile::ops::PrefillOpProfiler>,
) {
    let dtype_str = format!("{:?}", kv_dtype);
    let key = (dtype_str.clone(), head_dim, reason);

    let mut guard = FALLBACK_WARNED.lock().unwrap_or_else(|e| e.into_inner());
    let set = guard.get_or_insert_with(HashSet::new);
    if set.insert(key) {
        eprintln!(
            "[GPU-fallback] prefill attn: dtype={} head_dim={} reason=\"{}\"",
            dtype_str, head_dim, reason
        );
    }
    drop(guard);

    if let Some(p) = profiler {
        p.cpu_fallback_count += 1;
    }
}

impl TransformerLayer {
    /// Standard forward path (Prefill or dynamic generation)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_prefill<C: KVCacheOps>(
        &self,
        x: &mut Tensor,
        kv_cache: &mut C,
        start_pos: usize,
        backend: &Arc<dyn Backend>,
        memory: &dyn Memory,
        rms_norm_eps: f32,
        rope_theta: f32,
        _need_scores: bool,
        head_dim: usize,
        batch_size: usize,
        seq_len: usize,
        dim: usize,
        _skip_attn: bool,
        _skip_mlp: bool,
        rms_norm_add_unit: bool,
        use_gelu_tanh: bool,
        is_local_attn: Option<bool>,
        local_attn_window: Option<usize>,
        prefill_ws: Option<&mut crate::layers::workspace::PrefillWorkspace>,
        layer_idx: usize,
        mut variance_collector: Option<
            &mut crate::core::pressure::d2o_layer_alloc::D2OVarianceCollector,
        >,
        mut profiler: Option<&mut crate::profile::ops::PrefillOpProfiler>,
    ) -> Result<()> {
        let q_dim = self.wq.shape().dims()[0];
        let k_dim = self.wk.shape().dims()[0];
        let v_dim = self.wv.shape().dims()[0];
        let n_heads_q = q_dim / head_dim;
        let n_heads_kv = k_dim / head_dim;

        // Split profiler borrow: we need it in multiple branches below.
        // Use a raw pointer to avoid borrow-checker conflict across branches.
        // Safety: `profiler` outlives this function and is never aliased concurrently.
        let profiler_ptr = profiler
            .as_mut()
            .map(|p| *p as *mut crate::profile::ops::PrefillOpProfiler);

        // SAFETY: single-threaded forward pass; pointer is valid for function lifetime.
        macro_rules! pf {
            () => {
                profiler_ptr.map(|p| unsafe { &mut *p })
            };
        }

        let is_gpu = backend.is_gpu();

        macro_rules! pf_start {
            () => {
                if profiler_ptr.is_some() {
                    if is_gpu {
                        backend.synchronize().ok();
                    }
                    std::time::Instant::now()
                } else {
                    std::time::Instant::now()
                }
            };
        }

        macro_rules! pf_record {
            ($t:expr, $field:ident) => {
                if let Some(p) = pf!() {
                    if is_gpu {
                        backend.synchronize().ok();
                    }
                    p.$field += $t.elapsed().as_micros() as u64;
                }
            };
        }

        // PrefillWorkspace path: early return to keep the fallback path's
        // stack frame identical to the original (pre-ae62391) code.
        if let Some(ws) = prefill_ws {
            // Reshape ALL workspace tensors to match actual seq_len (may be < max_seq_len)
            ws.residual
                .reshape(Shape::new(vec![batch_size, seq_len, dim]));
            ws.residual_ffn
                .reshape(Shape::new(vec![batch_size, seq_len, dim]));
            ws.q.reshape(Shape::new(vec![batch_size, seq_len, q_dim]));
            ws.k.reshape(Shape::new(vec![batch_size, seq_len, k_dim]));
            ws.v.reshape(Shape::new(vec![batch_size, seq_len, v_dim]));

            // Reuse pre-allocated workspace — zero GPU alloc/free per layer
            let n_elem = batch_size * seq_len * dim;
            backend.copy_slice(x, &mut ws.residual, 0, 0, n_elem)?;
            {
                let t = pf_start!();
                backend.rms_norm(x, &self.attention_norm, rms_norm_eps, rms_norm_add_unit)?;
                pf_record!(t, rms_norm);
            }

            // QKV: always GPU-only (attention partition removed — merge overhead exceeds benefit)
            {
                let t = pf_start!();
                backend.matmul_transposed(x, &self.wq, &mut ws.q)?;
                backend.matmul_transposed(x, &self.wk, &mut ws.k)?;
                backend.matmul_transposed(x, &self.wv, &mut ws.v)?;
                pf_record!(t, matmul_qkv);
            }

            if let Some(ref bias) = self.qkv_bias {
                backend.add_row_bias(&mut ws.q, &bias.bq)?;
                backend.add_row_bias(&mut ws.k, &bias.bk)?;
                backend.add_row_bias(&mut ws.v, &bias.bv)?;
            }

            // QK-norm (Gemma3)
            if let Some(ref q_norm_w) = self.q_norm {
                let total_q_heads = batch_size * seq_len * n_heads_q;
                let saved = ws.q.shape().clone();
                ws.q.reshape(Shape::new(vec![total_q_heads, head_dim]));
                backend.rms_norm(&mut ws.q, q_norm_w, rms_norm_eps, true)?;
                ws.q.reshape(saved);
            }
            if let Some(ref k_norm_w) = self.k_norm {
                let total_k_heads = batch_size * seq_len * n_heads_kv;
                let saved = ws.k.shape().clone();
                ws.k.reshape(Shape::new(vec![total_k_heads, head_dim]));
                backend.rms_norm(&mut ws.k, k_norm_w, rms_norm_eps, true)?;
                ws.k.reshape(saved);
            }

            let mut q_rope = Tensor::new(
                Shape::new(vec![batch_size, seq_len, n_heads_q, head_dim]),
                ws.q.buffer().clone(),
                backend.clone(),
            );
            let mut k_rope = Tensor::new(
                Shape::new(vec![batch_size, seq_len, n_heads_kv, head_dim]),
                ws.k.buffer().clone(),
                backend.clone(),
            );

            {
                let t = pf_start!();
                backend.rope_inplace(&mut q_rope, start_pos, rope_theta)?;
                backend.rope_inplace(&mut k_rope, start_pos, rope_theta)?;
                pf_record!(t, rope);
            }

            // KV cache update
            let kv_dtype = kv_cache.kv_dtype();
            {
                let t = pf_start!();
                use crate::core::kv_cache::KVLayout;
                if kv_dtype == DType::F16 && is_gpu && kv_cache.layout() == KVLayout::HeadMajor {
                    // GPU F16 HeadMajor: batch cast+scatter kernel (1 dispatch)
                    kv_cache.ensure_capacity(kv_cache.current_pos() + seq_len)?;
                    let pos = kv_cache.current_pos();
                    let cap = kv_cache.capacity();
                    if let Some((k_buf, v_buf)) = kv_cache.get_buffers_mut() {
                        backend.kv_scatter_f32_to_f16_batch(
                            &k_rope, &ws.v, k_buf, v_buf, n_heads_kv, head_dim, cap, pos, seq_len,
                        )?;
                    }
                    kv_cache.advance_pos(seq_len);
                } else if kv_dtype != DType::F32 {
                    let n_elem = seq_len * n_heads_kv * head_dim;
                    let buf_size = match kv_dtype {
                        DType::F16 => n_elem * 2,
                        DType::Q4_0 => {
                            (n_elem / crate::core::quant::QK4_0)
                                * std::mem::size_of::<crate::core::quant::BlockQ4_0>()
                        }
                        _ => n_elem * 4,
                    };
                    if ws.k_cast.is_none() {
                        let buf = memory.alloc(buf_size, kv_dtype)?;
                        ws.k_cast = Some(Tensor::new(k_rope.shape().clone(), buf, backend.clone()));
                    }
                    if ws.v_cast.is_none() {
                        let buf = memory.alloc(buf_size, kv_dtype)?;
                        ws.v_cast = Some(Tensor::new(ws.v.shape().clone(), buf, backend.clone()));
                    }
                    let k_c = ws.k_cast.as_mut().unwrap();
                    let v_c = ws.v_cast.as_mut().unwrap();
                    backend.cast(&k_rope, k_c)?;
                    backend.cast(&ws.v, v_c)?;
                    kv_cache.update(k_c, v_c)?;
                } else {
                    super::update_kv_cache(kv_cache, &k_rope, &ws.v, backend)?;
                }
                pf_record!(t, kv_write);
            }

            let cache_seq_len = kv_cache.current_pos();
            let kv_capacity = kv_cache.capacity();
            let kv_layout = kv_cache.layout();
            let (k_cache, v_cache) = kv_cache.get_view();

            ws.out_attn
                .reshape(Shape::new(vec![batch_size, seq_len, q_dim]));

            // GPU flash attention — only if KV buffers are actually GPU buffers.
            // CPU-only caches (e.g. KiviCache with SharedBuffer) skip to CPU fallback.
            let kv_is_gpu = k_cache.buffer().is_gpu_buffer();
            let gpu_dispatched = if is_gpu && kv_is_gpu {
                let t = pf_start!();
                let dispatched = backend.flash_attention_prefill(
                    &q_rope,
                    &k_cache,
                    &v_cache,
                    &mut ws.out_attn,
                    n_heads_q,
                    n_heads_kv,
                    seq_len,
                    cache_seq_len,
                    head_dim,
                    kv_capacity,
                    batch_size,
                    kv_layout == crate::core::kv_cache::KVLayout::HeadMajor,
                )?;
                if dispatched {
                    pf_record!(t, flash_prefill_gpu);
                }
                dispatched
            } else {
                false
            };

            // Detect and log GPU→CPU attention fallback.
            if !gpu_dispatched && is_gpu && kv_is_gpu {
                let reason: &'static str = if !matches!(head_dim, 64 | 128 | 256) {
                    "head_dim not in {64, 128, 256} (no flash_attn DK variant compiled)"
                } else if kv_dtype == DType::Q4_0 {
                    "kv dtype Q4_0 not supported by flash_attn kernel"
                } else {
                    "flash_attn kernel unavailable for this dtype/config"
                };
                warn_gpu_fallback_once(kv_dtype, head_dim, reason, pf!());
            }

            // Use read_buffer path only when buffers are not CPU-accessible (device-only).
            // UMA/pinned buffers (CUDA, OpenCL zero-copy) have valid as_ptr().
            let is_device_only = is_gpu && q_rope.as_ptr().is_null();
            if !gpu_dispatched {
                // CPU attention fallback
                let pf_attn_t = pf_start!();
                let mut out_vec = Vec::new();
                {
                    fn as_u8_mut(v: &mut [f32]) -> &mut [u8] {
                        unsafe {
                            std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4)
                        }
                    }

                    let mut q_vec = Vec::new();
                    let mut k_vec = Vec::new();
                    let mut v_vec = Vec::new();

                    let (q_data, k_data, v_data, out_ptr) = if is_device_only {
                        let read_to_f32 = |t: &Tensor, vec: &mut Vec<f32>| -> Result<()> {
                            if t.dtype() == DType::Q4_0 {
                                use crate::core::quant::{BlockQ4_0, QK4_0};
                                let numel = t.numel();
                                let n_blocks = numel / QK4_0;
                                let byte_size = n_blocks * std::mem::size_of::<BlockQ4_0>();
                                let mut byte_vec = vec![0u8; byte_size];
                                backend.read_buffer(t, &mut byte_vec)?;
                                vec.resize(numel, 0.0);
                                let blocks = unsafe {
                                    std::slice::from_raw_parts(
                                        byte_vec.as_ptr() as *const BlockQ4_0,
                                        n_blocks,
                                    )
                                };
                                for i in 0..n_blocks {
                                    let mut tmp = [0.0f32; QK4_0];
                                    blocks[i].dequantize(&mut tmp);
                                    vec[i * QK4_0..(i + 1) * QK4_0].copy_from_slice(&tmp);
                                }
                            } else if t.dtype() == DType::F16 {
                                let numel = t.numel();
                                let byte_size = numel * 2;
                                let mut byte_vec = vec![0u8; byte_size];
                                backend.read_buffer(t, &mut byte_vec)?;
                                vec.resize(numel, 0.0);
                                #[cfg(target_arch = "aarch64")]
                                unsafe {
                                    crate::backend::cpu::neon::CpuBackendNeon::bulk_f16_to_f32(
                                        byte_vec.as_ptr() as *const u16,
                                        vec.as_mut_ptr(),
                                        numel,
                                    );
                                }
                                #[cfg(not(target_arch = "aarch64"))]
                                {
                                    let f16_slice = unsafe {
                                        std::slice::from_raw_parts(
                                            byte_vec.as_ptr() as *const half::f16,
                                            numel,
                                        )
                                    };
                                    for i in 0..numel {
                                        vec[i] = f16_slice[i].to_f32();
                                    }
                                }
                            } else {
                                vec.resize(t.numel(), 0.0);
                                backend.read_buffer(t, as_u8_mut(vec))?;
                            }
                            Ok(())
                        };

                        read_to_f32(&q_rope, &mut q_vec)?;
                        read_to_f32(&k_cache, &mut k_vec)?;
                        read_to_f32(&v_cache, &mut v_vec)?;

                        out_vec.resize(ws.out_attn.numel(), 0.0);

                        (&q_vec[..], &k_vec[..], &v_vec[..], &mut out_vec[..])
                    } else if k_cache.dtype() == DType::Q4_0 {
                        use crate::core::quant::{BlockQ4_0, QK4_0};
                        let n_elems = if kv_layout == crate::core::kv_cache::KVLayout::HeadMajor {
                            n_heads_kv * kv_capacity * head_dim
                        } else {
                            cache_seq_len * n_heads_kv * head_dim
                        };
                        let n_blocks = n_elems / QK4_0;

                        let k_q4 = unsafe {
                            std::slice::from_raw_parts(
                                k_cache.as_ptr() as *const BlockQ4_0,
                                n_blocks,
                            )
                        };
                        let v_q4 = unsafe {
                            std::slice::from_raw_parts(
                                v_cache.as_ptr() as *const BlockQ4_0,
                                n_blocks,
                            )
                        };

                        k_vec.resize(n_elems, 0.0f32);
                        v_vec.resize(n_elems, 0.0f32);
                        for i in 0..n_blocks {
                            let mut tmp = [0.0f32; QK4_0];
                            k_q4[i].dequantize(&mut tmp);
                            k_vec[i * QK4_0..(i + 1) * QK4_0].copy_from_slice(&tmp);
                            v_q4[i].dequantize(&mut tmp);
                            v_vec[i * QK4_0..(i + 1) * QK4_0].copy_from_slice(&tmp);
                        }
                        (
                            q_rope.as_slice::<f32>(),
                            &k_vec[..],
                            &v_vec[..],
                            ws.out_attn.as_mut_slice::<f32>(),
                        )
                    } else if k_cache.dtype() == DType::F16 {
                        let n_elems = if kv_layout == crate::core::kv_cache::KVLayout::HeadMajor {
                            n_heads_kv * kv_capacity * head_dim
                        } else {
                            cache_seq_len * n_heads_kv * head_dim
                        };
                        let k_f16_ptr = k_cache.as_ptr() as *const u16;
                        let v_f16_ptr = v_cache.as_ptr() as *const u16;
                        k_vec.resize(n_elems, 0.0f32);
                        v_vec.resize(n_elems, 0.0f32);
                        #[cfg(target_arch = "aarch64")]
                        unsafe {
                            crate::backend::cpu::neon::CpuBackendNeon::bulk_f16_to_f32(
                                k_f16_ptr,
                                k_vec.as_mut_ptr(),
                                n_elems,
                            );
                            crate::backend::cpu::neon::CpuBackendNeon::bulk_f16_to_f32(
                                v_f16_ptr,
                                v_vec.as_mut_ptr(),
                                n_elems,
                            );
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            let k_f16 = unsafe {
                                std::slice::from_raw_parts(k_f16_ptr as *const half::f16, n_elems)
                            };
                            let v_f16 = unsafe {
                                std::slice::from_raw_parts(v_f16_ptr as *const half::f16, n_elems)
                            };
                            for i in 0..n_elems {
                                k_vec[i] = k_f16[i].to_f32();
                                v_vec[i] = v_f16[i].to_f32();
                            }
                        }
                        (
                            q_rope.as_slice::<f32>(),
                            &k_vec[..],
                            &v_vec[..],
                            ws.out_attn.as_mut_slice::<f32>(),
                        )
                    } else {
                        (
                            q_rope.as_slice::<f32>(),
                            k_cache.as_slice::<f32>(),
                            v_cache.as_slice::<f32>(),
                            ws.out_attn.as_mut_slice::<f32>(),
                        )
                    };

                    for x in out_ptr.iter_mut() {
                        *x = 0.0;
                    }

                    use crate::layers::attention::flash_attention_forward_strided;
                    let is_head_major_pf = kv_layout == crate::core::kv_cache::KVLayout::HeadMajor;
                    let chunk_q_stride = seq_len * n_heads_q * head_dim;
                    let chunk_out_stride = seq_len * n_heads_q * head_dim;
                    let chunk_k_stride = kv_capacity * n_heads_kv * head_dim;
                    let (k_pos_stride, kv_head_stride) = if is_head_major_pf {
                        (head_dim, kv_capacity * head_dim)
                    } else {
                        (n_heads_kv * head_dim, head_dim)
                    };

                    let window_size = if let Some(true) = is_local_attn {
                        local_attn_window
                    } else {
                        None
                    };

                    for (b, out_batch) in out_ptr.chunks_mut(chunk_out_stride).enumerate() {
                        let q_start = b * chunk_q_stride;
                        let k_start = b * chunk_k_stride;
                        let v_start = b * chunk_k_stride;
                        let q_slice = &q_data[q_start..q_start + chunk_q_stride];
                        let k_valid_len = if is_head_major_pf {
                            n_heads_kv * kv_capacity * head_dim
                        } else {
                            cache_seq_len * n_heads_kv * head_dim
                        };
                        let k_slice = &k_data[k_start..k_start + k_valid_len];
                        let v_slice = &v_data[v_start..v_start + k_valid_len];

                        // D2O: collect per-layer attention column-sums for layer-level allocation.
                        // Only collect for the first batch element (LLM inference always has batch_size=1).
                        if b == 0
                            && let Some(ref mut vc) = variance_collector
                        {
                            vc.collect_layer(
                                layer_idx,
                                q_slice,
                                k_slice,
                                seq_len,
                                cache_seq_len,
                                n_heads_q * head_dim,
                                k_pos_stride,
                                kv_head_stride,
                                start_pos,
                            );
                        }

                        flash_attention_forward_strided(
                            q_slice,
                            k_slice,
                            v_slice,
                            out_batch,
                            n_heads_q,
                            n_heads_kv,
                            seq_len,
                            cache_seq_len,
                            head_dim,
                            n_heads_q * head_dim,
                            k_pos_stride,
                            k_pos_stride,
                            n_heads_q * head_dim,
                            kv_head_stride,
                            start_pos,
                            32,
                            32,
                            window_size,
                        );
                    }
                }

                if is_device_only {
                    // Write CPU attention result back to workspace GPU buffer.
                    let out_bytes = unsafe {
                        std::slice::from_raw_parts(out_vec.as_ptr() as *const u8, out_vec.len() * 4)
                    };
                    let dst_ptr = ws.out_attn.as_mut_ptr();
                    if !dst_ptr.is_null() {
                        // UMA / pinned memory: direct memcpy is sufficient.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                out_bytes.as_ptr(),
                                dst_ptr,
                                out_bytes.len(),
                            );
                        }
                    }
                    #[cfg(feature = "opencl")]
                    {
                        // OpenCL device-only buffers need enqueue_write_buffer.
                        if dst_ptr.is_null()
                            && let Ok(dst_mem) =
                                crate::backend::opencl::get_cl_mem(ws.out_attn.buffer().as_ref())
                        {
                            unsafe {
                                ocl::core::enqueue_write_buffer(
                                    &backend
                                        .as_any()
                                        .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
                                        .unwrap()
                                        .queue,
                                    dst_mem,
                                    true,
                                    0,
                                    out_bytes,
                                    None::<&ocl::core::Event>,
                                    None::<&mut ocl::core::Event>,
                                )?;
                            }
                        }
                    }
                }
                pf_record!(pf_attn_t, flash_prefill_cpu);
            }

            // O-proj: always GPU-only (wo partition removed — merge overhead exceeds benefit)
            ws.attn_out_proj
                .reshape(Shape::new(vec![batch_size, seq_len, dim]));
            {
                let t = pf_start!();
                backend.matmul_transposed(&ws.out_attn, &self.wo, &mut ws.attn_out_proj)?;
                pf_record!(t, matmul_wo);
            }
            if rms_norm_add_unit {
                backend.rms_norm(&mut ws.attn_out_proj, &self.ffn_norm, rms_norm_eps, true)?;
            }
            {
                let t = pf_start!();
                backend.add_assign(&mut ws.attn_out_proj, &ws.residual)?;
                pf_record!(t, add_assign);
            }

            // Copy to x for FFN
            backend.copy_slice(&ws.attn_out_proj, x, 0, 0, n_elem)?;

            // FFN
            backend.copy_slice(x, &mut ws.residual_ffn, 0, 0, n_elem)?;
            {
                let t = pf_start!();
                if let Some(ref pfn) = self.pre_ffn_norm {
                    backend.rms_norm(x, pfn, rms_norm_eps, true)?;
                } else {
                    backend.rms_norm(x, &self.ffn_norm, rms_norm_eps, false)?;
                }
                pf_record!(t, rms_norm);
            }

            let ffn_hidden = self.w_up.shape().dims()[0];
            ws.gate
                .reshape(Shape::new(vec![batch_size, seq_len, ffn_hidden]));
            ws.up
                .reshape(Shape::new(vec![batch_size, seq_len, ffn_hidden]));
            ws.down.reshape(Shape::new(vec![batch_size, seq_len, dim]));

            if let Some(ref part) = self.partition_ctx {
                // ── Partitioned FFN gate/up (prefill): cooperative GPU + CPU ──
                let total_rows = batch_size * seq_len;

                // Copy normalized x to CPU
                let x_cpu_bytes = batch_size * seq_len * dim * 4;
                backend.synchronize()?;
                let x_cpu_buf = Galloc::new().alloc(x_cpu_bytes, DType::F32)?;
                let x_cpu = Tensor::new(x.shape().clone(), x_cpu_buf, part.cpu_backend.clone());
                unsafe {
                    let dst =
                        std::slice::from_raw_parts_mut(x_cpu.as_ptr() as *mut u8, x_cpu_bytes);
                    backend.read_buffer(x, dst)?;
                }

                // Gate partition
                {
                    let sr = part.gate.split_row;
                    let cr = ffn_hidden - sr;
                    let gpu_buf = memory.alloc(total_rows * sr * 4, DType::F32)?;
                    let mut gpu_partial = Tensor::new(
                        Shape::new(vec![batch_size, seq_len, sr]),
                        gpu_buf,
                        backend.clone(),
                    );
                    backend.matmul_transposed(x, &part.gate.gpu_slice, &mut gpu_partial)?;
                    backend.flush()?;

                    let cpu_buf = Galloc::new().alloc(total_rows * cr * 4, DType::F32)?;
                    let mut cpu_partial = Tensor::new(
                        Shape::new(vec![batch_size, seq_len, cr]),
                        cpu_buf,
                        part.cpu_backend.clone(),
                    );
                    part.cpu_backend.matmul_transposed(
                        &x_cpu,
                        &part.gate.cpu_slice,
                        &mut cpu_partial,
                    )?;

                    crate::layers::tensor_partition::merge_partials_2d(
                        backend.as_ref(),
                        &gpu_partial,
                        &cpu_partial,
                        &mut ws.gate,
                        total_rows,
                        sr,
                        cr,
                    )?;
                }

                // Up partition
                {
                    let sr = part.up.split_row;
                    let cr = ffn_hidden - sr;
                    let gpu_buf = memory.alloc(total_rows * sr * 4, DType::F32)?;
                    let mut gpu_partial = Tensor::new(
                        Shape::new(vec![batch_size, seq_len, sr]),
                        gpu_buf,
                        backend.clone(),
                    );
                    backend.matmul_transposed(x, &part.up.gpu_slice, &mut gpu_partial)?;
                    backend.flush()?;

                    let cpu_buf = Galloc::new().alloc(total_rows * cr * 4, DType::F32)?;
                    let mut cpu_partial = Tensor::new(
                        Shape::new(vec![batch_size, seq_len, cr]),
                        cpu_buf,
                        part.cpu_backend.clone(),
                    );
                    part.cpu_backend.matmul_transposed(
                        &x_cpu,
                        &part.up.cpu_slice,
                        &mut cpu_partial,
                    )?;

                    crate::layers::tensor_partition::merge_partials_2d(
                        backend.as_ref(),
                        &gpu_partial,
                        &cpu_partial,
                        &mut ws.up,
                        total_rows,
                        sr,
                        cr,
                    )?;
                }
            } else {
                {
                    let t = pf_start!();
                    backend.matmul_transposed(x, &self.w_gate, &mut ws.gate)?;
                    pf_record!(t, ffn_gate);
                }
                {
                    let t = pf_start!();
                    backend.matmul_transposed(x, &self.w_up, &mut ws.up)?;
                    pf_record!(t, ffn_up);
                }
            }

            {
                let t = pf_start!();
                if use_gelu_tanh {
                    backend.gelu_tanh_mul(&mut ws.gate, &ws.up)?;
                } else {
                    backend.silu_mul(&mut ws.gate, &ws.up)?;
                }
                pf_record!(t, silu_mul);
            }

            // Down: always GPU-only (down partition removed — merge overhead exceeds benefit)
            {
                let t = pf_start!();
                backend.matmul_transposed(&ws.gate, &self.w_down, &mut ws.down)?;
                pf_record!(t, ffn_down);
            }
            if let Some(ref pfn) = self.post_ffn_norm {
                backend.rms_norm(&mut ws.down, pfn, rms_norm_eps, true)?;
            }
            {
                let t = pf_start!();
                backend.add_assign(&mut ws.down, &ws.residual_ffn)?;
                pf_record!(t, add_assign);
            }

            // Record layer count.
            if let Some(p) = pf!() {
                p.layer_count += 1;
            }

            // Output x = down
            backend.copy_slice(&ws.down, x, 0, 0, n_elem)?;
            return Ok(());
        }

        // Fallback: allocate temp buffers per layer (original path, pre-ae62391)
        let residual = backend.copy_from(x)?;
        {
            let t = pf_start!();
            backend.rms_norm(x, &self.attention_norm, rms_norm_eps, rms_norm_add_unit)?;
            pf_record!(t, rms_norm);
        }

        let mut q = self.alloc_temp(vec![batch_size, seq_len, q_dim], memory, backend)?;
        let mut k = self.alloc_temp(vec![batch_size, seq_len, k_dim], memory, backend)?;
        let mut v = self.alloc_temp(vec![batch_size, seq_len, v_dim], memory, backend)?;

        {
            let t = pf_start!();
            backend.matmul_transposed(x, &self.wq, &mut q)?;
            backend.matmul_transposed(x, &self.wk, &mut k)?;
            backend.matmul_transposed(x, &self.wv, &mut v)?;
            pf_record!(t, matmul_qkv);
        }

        if let Some(ref bias) = self.qkv_bias {
            backend.add_row_bias(&mut q, &bias.bq)?;
            backend.add_row_bias(&mut k, &bias.bk)?;
            backend.add_row_bias(&mut v, &bias.bv)?;
        }

        if let Some(ref q_norm_w) = self.q_norm {
            let total_q_heads = batch_size * seq_len * n_heads_q;
            let saved_shape = q.shape().clone();
            q.reshape(Shape::new(vec![total_q_heads, head_dim]));
            backend.rms_norm(&mut q, q_norm_w, rms_norm_eps, true)?;
            q.reshape(saved_shape);
        }
        if let Some(ref k_norm_w) = self.k_norm {
            let total_k_heads = batch_size * seq_len * n_heads_kv;
            let saved_shape = k.shape().clone();
            k.reshape(Shape::new(vec![total_k_heads, head_dim]));
            backend.rms_norm(&mut k, k_norm_w, rms_norm_eps, true)?;
            k.reshape(saved_shape);
        }

        let mut q_rope = Tensor::new(
            Shape::new(vec![batch_size, seq_len, n_heads_q, head_dim]),
            q.buffer().clone(),
            backend.clone(),
        );
        let mut k_rope = Tensor::new(
            Shape::new(vec![batch_size, seq_len, n_heads_kv, head_dim]),
            k.buffer().clone(),
            backend.clone(),
        );

        {
            let t = pf_start!();
            backend.rope_inplace(&mut q_rope, start_pos, rope_theta)?;
            backend.rope_inplace(&mut k_rope, start_pos, rope_theta)?;
            pf_record!(t, rope);
        }

        // Cast to target dtype if KV cache is not F32
        let kv_dtype = kv_cache.kv_dtype();
        {
            let t = pf_start!();
            use crate::core::kv_cache::KVLayout;
            if kv_dtype == DType::F16 && is_gpu && kv_cache.layout() == KVLayout::HeadMajor {
                // GPU F16 HeadMajor: batch cast+scatter kernel (1 dispatch)
                kv_cache.ensure_capacity(kv_cache.current_pos() + seq_len)?;
                let pos = kv_cache.current_pos();
                let cap = kv_cache.capacity();
                if let Some((k_buf, v_buf)) = kv_cache.get_buffers_mut() {
                    backend.kv_scatter_f32_to_f16_batch(
                        &k_rope, &v, k_buf, v_buf, n_heads_kv, head_dim, cap, pos, seq_len,
                    )?;
                }
                kv_cache.advance_pos(seq_len);
            } else if kv_dtype != DType::F32 {
                let n_elem = seq_len * n_heads_kv * head_dim;
                let buf_size = match kv_dtype {
                    DType::F16 => n_elem * 2,
                    DType::Q4_0 => {
                        (n_elem / crate::core::quant::QK4_0)
                            * std::mem::size_of::<crate::core::quant::BlockQ4_0>()
                    }
                    _ => n_elem * 4,
                };
                let k_cast_buf = memory.alloc(buf_size, kv_dtype)?;
                let mut k_cast = Tensor::new(k_rope.shape().clone(), k_cast_buf, backend.clone());
                backend.cast(&k_rope, &mut k_cast)?;
                let v_cast_buf = memory.alloc(buf_size, kv_dtype)?;
                let mut v_cast = Tensor::new(v.shape().clone(), v_cast_buf, backend.clone());
                backend.cast(&v, &mut v_cast)?;
                kv_cache.update(&k_cast, &v_cast)?;
            } else {
                super::update_kv_cache(kv_cache, &k_rope, &v, backend)?;
            }
            pf_record!(t, kv_write);
        }

        let cache_seq_len = kv_cache.current_pos();
        let kv_capacity = kv_cache.capacity();
        let kv_layout = kv_cache.layout();
        let (k_cache, v_cache) = kv_cache.get_view();

        let mut out_attn = self.alloc_temp(vec![batch_size, seq_len, q_dim], memory, backend)?;

        // GPU flash attention — only if KV buffers are actually GPU buffers.
        // CPU-only caches (e.g. KiviCache with SharedBuffer) skip to CPU fallback.
        let kv_is_gpu = k_cache.buffer().is_gpu_buffer();
        let gpu_dispatched = if is_gpu && kv_is_gpu {
            let t = pf_start!();
            let dispatched = backend.flash_attention_prefill(
                &q_rope,
                &k_cache,
                &v_cache,
                &mut out_attn,
                n_heads_q,
                n_heads_kv,
                seq_len,
                cache_seq_len,
                head_dim,
                kv_capacity,
                batch_size,
                kv_layout == crate::core::kv_cache::KVLayout::HeadMajor,
            )?;
            if dispatched {
                pf_record!(t, flash_prefill_gpu);
            }
            dispatched
        } else {
            false
        };

        // Detect and log GPU→CPU attention fallback.
        if !gpu_dispatched && is_gpu && kv_is_gpu {
            let reason: &'static str = if !matches!(head_dim, 64 | 128 | 256) {
                "head_dim not in {64, 128, 256} (no flash_attn DK variant compiled)"
            } else if kv_dtype == DType::Q4_0 {
                "kv dtype Q4_0 not supported by flash_attn kernel"
            } else {
                "flash_attn kernel unavailable for this dtype/config"
            };
            warn_gpu_fallback_once(kv_dtype, head_dim, reason, pf!());
        }

        let is_device_only2 = is_gpu && q_rope.as_ptr().is_null();
        if !gpu_dispatched {
            // ---- CPU flash attention path (also used as GPU fallback) ----
            let pf_attn_t2 = pf_start!();
            let mut out_vec = Vec::new();

            {
                fn as_u8_mut(v: &mut [f32]) -> &mut [u8] {
                    unsafe {
                        std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4)
                    }
                }

                let mut q_vec = Vec::new();
                let mut k_vec = Vec::new();
                let mut v_vec = Vec::new();

                let (q_data, k_data, v_data, out_ptr) = if is_device_only2 {
                    let read_to_f32 = |t: &Tensor, vec: &mut Vec<f32>| -> Result<()> {
                        if t.dtype() == DType::Q4_0 {
                            use crate::core::quant::{BlockQ4_0, QK4_0};
                            let numel = t.numel();
                            let n_blocks = numel / QK4_0;
                            let byte_size = n_blocks * std::mem::size_of::<BlockQ4_0>();

                            let mut byte_vec = vec![0u8; byte_size];
                            backend.read_buffer(t, &mut byte_vec)?;

                            vec.resize(numel, 0.0);
                            let blocks = unsafe {
                                std::slice::from_raw_parts(
                                    byte_vec.as_ptr() as *const BlockQ4_0,
                                    n_blocks,
                                )
                            };

                            for i in 0..n_blocks {
                                let mut tmp = [0.0f32; QK4_0];
                                blocks[i].dequantize(&mut tmp);
                                vec[i * QK4_0..(i + 1) * QK4_0].copy_from_slice(&tmp);
                            }
                        } else if t.dtype() == DType::F16 {
                            let numel = t.numel();
                            let byte_size = numel * 2;
                            let mut byte_vec = vec![0u8; byte_size];
                            backend.read_buffer(t, &mut byte_vec)?;
                            vec.resize(numel, 0.0);
                            #[cfg(target_arch = "aarch64")]
                            unsafe {
                                crate::backend::cpu::neon::CpuBackendNeon::bulk_f16_to_f32(
                                    byte_vec.as_ptr() as *const u16,
                                    vec.as_mut_ptr(),
                                    numel,
                                );
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            {
                                let f16_slice = unsafe {
                                    std::slice::from_raw_parts(
                                        byte_vec.as_ptr() as *const half::f16,
                                        numel,
                                    )
                                };
                                for i in 0..numel {
                                    vec[i] = f16_slice[i].to_f32();
                                }
                            }
                        } else {
                            vec.resize(t.numel(), 0.0);
                            backend.read_buffer(t, as_u8_mut(vec))?;
                        }
                        Ok(())
                    };

                    read_to_f32(&q_rope, &mut q_vec)?;
                    read_to_f32(&k_cache, &mut k_vec)?;
                    read_to_f32(&v_cache, &mut v_vec)?;

                    out_vec.resize(out_attn.numel(), 0.0);

                    (&q_vec[..], &k_vec[..], &v_vec[..], &mut out_vec[..])
                } else if k_cache.dtype() == DType::Q4_0 {
                    // Q4_0: dequantize KV cache to F32 temp buffers
                    use crate::core::quant::{BlockQ4_0, QK4_0};
                    // HeadMajor: need full buffer (heads are non-contiguous)
                    let n_elems = if kv_layout == crate::core::kv_cache::KVLayout::HeadMajor {
                        n_heads_kv * kv_capacity * head_dim
                    } else {
                        cache_seq_len * n_heads_kv * head_dim
                    };
                    let n_blocks = n_elems / QK4_0;
                    let k_blocks = unsafe {
                        std::slice::from_raw_parts(k_cache.as_ptr() as *const BlockQ4_0, n_blocks)
                    };
                    let v_blocks = unsafe {
                        std::slice::from_raw_parts(v_cache.as_ptr() as *const BlockQ4_0, n_blocks)
                    };
                    k_vec.resize(n_elems, 0.0f32);
                    v_vec.resize(n_elems, 0.0f32);
                    for bi in 0..n_blocks {
                        let mut tmp = [0.0f32; QK4_0];
                        k_blocks[bi].dequantize(&mut tmp);
                        k_vec[bi * QK4_0..(bi + 1) * QK4_0].copy_from_slice(&tmp);
                        v_blocks[bi].dequantize(&mut tmp);
                        v_vec[bi * QK4_0..(bi + 1) * QK4_0].copy_from_slice(&tmp);
                    }
                    (
                        q_rope.as_slice::<f32>(),
                        &k_vec[..],
                        &v_vec[..],
                        out_attn.as_mut_slice::<f32>(),
                    )
                } else if k_cache.dtype() == DType::F16 {
                    // F16: convert KV cache to F32 temp buffers using NEON bulk conversion
                    // HeadMajor: need full buffer (heads are non-contiguous)
                    let n_elems = if kv_layout == crate::core::kv_cache::KVLayout::HeadMajor {
                        n_heads_kv * kv_capacity * head_dim
                    } else {
                        cache_seq_len * n_heads_kv * head_dim
                    };
                    let k_f16_ptr = k_cache.as_ptr() as *const u16;
                    let v_f16_ptr = v_cache.as_ptr() as *const u16;
                    k_vec.resize(n_elems, 0.0f32);
                    v_vec.resize(n_elems, 0.0f32);
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        crate::backend::cpu::neon::CpuBackendNeon::bulk_f16_to_f32(
                            k_f16_ptr,
                            k_vec.as_mut_ptr(),
                            n_elems,
                        );
                        crate::backend::cpu::neon::CpuBackendNeon::bulk_f16_to_f32(
                            v_f16_ptr,
                            v_vec.as_mut_ptr(),
                            n_elems,
                        );
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        let k_f16 = unsafe {
                            std::slice::from_raw_parts(k_f16_ptr as *const half::f16, n_elems)
                        };
                        let v_f16 = unsafe {
                            std::slice::from_raw_parts(v_f16_ptr as *const half::f16, n_elems)
                        };
                        for i in 0..n_elems {
                            k_vec[i] = k_f16[i].to_f32();
                            v_vec[i] = v_f16[i].to_f32();
                        }
                    }
                    (
                        q_rope.as_slice::<f32>(),
                        &k_vec[..],
                        &v_vec[..],
                        out_attn.as_mut_slice::<f32>(),
                    )
                } else {
                    (
                        q_rope.as_slice::<f32>(),
                        k_cache.as_slice::<f32>(),
                        v_cache.as_slice::<f32>(),
                        out_attn.as_mut_slice::<f32>(),
                    )
                };

                use crate::layers::attention::flash_attention_forward_strided;

                let is_head_major_pf = kv_layout == crate::core::kv_cache::KVLayout::HeadMajor;

                let chunk_q_stride = seq_len * n_heads_q * head_dim;
                let chunk_out_stride = seq_len * n_heads_q * head_dim;
                // KV Cache is strided by capacity (physical buffer size), not max_seq_len
                let chunk_k_stride = kv_capacity * n_heads_kv * head_dim;

                // Layout-dependent strides
                let (k_pos_stride, kv_head_stride) = if is_head_major_pf {
                    (head_dim, kv_capacity * head_dim)
                } else {
                    (n_heads_kv * head_dim, head_dim)
                };

                // Iterate over batch.
                // We use chunks_mut for out_ptr to satisfy borrow checker.
                for (b, out_batch) in out_ptr.chunks_mut(chunk_out_stride).enumerate() {
                    let q_start = b * chunk_q_stride;
                    let k_start = b * chunk_k_stride;
                    let v_start = b * chunk_k_stride; // V has same layout as K in cache

                    let q_slice = &q_data[q_start..q_start + chunk_q_stride];

                    // For HeadMajor, heads are non-contiguous so we need the full buffer per batch.
                    // For SeqMajor, only valid positions are needed (contiguous from start).
                    let k_valid_len = if is_head_major_pf {
                        n_heads_kv * kv_capacity * head_dim
                    } else {
                        cache_seq_len * n_heads_kv * head_dim
                    };
                    let k_slice = &k_data[k_start..k_start + k_valid_len];
                    let v_slice = &v_data[v_start..v_start + k_valid_len];

                    // Sliding window for local attention (Gemma3): None = full causal
                    let window_size = if let Some(true) = is_local_attn {
                        local_attn_window
                    } else {
                        None
                    };

                    flash_attention_forward_strided(
                        q_slice,
                        k_slice,
                        v_slice,
                        out_batch,
                        n_heads_q,
                        n_heads_kv,
                        seq_len,
                        cache_seq_len,
                        head_dim,
                        n_heads_q * head_dim, // q_stride
                        k_pos_stride,         // k_stride (position stride)
                        k_pos_stride,         // v_stride (same as k)
                        n_heads_q * head_dim, // out_stride
                        kv_head_stride,       // kv_head_stride
                        start_pos,            // q_start_pos for causal mask
                        32,                   // br
                        32,                   // bc
                        window_size,
                    );
                }
            }

            if is_device_only2 {
                // Create temp CPU tensor from result and copy back
                // Using Galloc directly
                let size_bytes = out_vec.len() * 4;
                let buf = crate::memory::galloc::Galloc::new().alloc(size_bytes, DType::F32)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        out_vec.as_ptr(),
                        buf.as_mut_ptr() as *mut f32,
                        out_vec.len(),
                    );
                }
                let cpu_out = Tensor::new(
                    out_attn.shape().clone(),
                    buf,
                    Arc::new(crate::backend::cpu::CpuBackend::new()),
                );
                out_attn = backend.copy_from(&cpu_out)?;
            }
            pf_record!(pf_attn_t2, flash_prefill_cpu);
        }

        let mut attn_out_projected =
            self.alloc_temp(vec![batch_size, seq_len, dim], memory, backend)?;
        {
            let t = pf_start!();
            backend.matmul_transposed(&out_attn, &self.wo, &mut attn_out_projected)?;
            pf_record!(t, matmul_wo);
        }

        // Gemma3: apply post-attention norm (ffn_norm) to O-proj output before residual add.
        // Llama/Qwen2: no post-attention norm here.
        if rms_norm_add_unit {
            backend.rms_norm(&mut attn_out_projected, &self.ffn_norm, rms_norm_eps, true)?;
        }
        {
            let t = pf_start!();
            backend.add_assign(&mut attn_out_projected, &residual)?;
            pf_record!(t, add_assign);
        }
        *x = attn_out_projected;

        let residual_ffn = backend.copy_from(x)?;
        // Gemma3: use dedicated pre_ffn_norm. Llama/Qwen2: use ffn_norm as pre-FFN norm.
        {
            let t = pf_start!();
            if let Some(ref pfn) = self.pre_ffn_norm {
                backend.rms_norm(x, pfn, rms_norm_eps, true)?;
            } else {
                backend.rms_norm(x, &self.ffn_norm, rms_norm_eps, false)?;
            }
            pf_record!(t, rms_norm);
        }

        let ffn_hidden = self.w_up.shape().dims()[0];
        let mut gate = self.alloc_temp(vec![batch_size, seq_len, ffn_hidden], memory, backend)?;
        let mut up = self.alloc_temp(vec![batch_size, seq_len, ffn_hidden], memory, backend)?;

        {
            let t = pf_start!();
            backend.matmul_transposed(x, &self.w_gate, &mut gate)?;
            pf_record!(t, ffn_gate);
        }
        {
            let t = pf_start!();
            backend.matmul_transposed(x, &self.w_up, &mut up)?;
            pf_record!(t, ffn_up);
        }

        {
            let t = pf_start!();
            if use_gelu_tanh {
                backend.gelu_tanh_mul(&mut gate, &up)?;
            } else {
                backend.silu_mul(&mut gate, &up)?;
            }
            pf_record!(t, silu_mul);
        }

        let mut down = self.alloc_temp(vec![batch_size, seq_len, dim], memory, backend)?;
        {
            let t = pf_start!();
            backend.matmul_transposed(&gate, &self.w_down, &mut down)?;
            pf_record!(t, ffn_down);
        }

        // Gemma3: apply post-FFN norm to FFN output before residual add.
        if let Some(ref pfn) = self.post_ffn_norm {
            backend.rms_norm(&mut down, pfn, rms_norm_eps, true)?;
        }
        {
            let t = pf_start!();
            backend.add_assign(&mut down, &residual_ffn)?;
            pf_record!(t, add_assign);
        }

        // Record layer count.
        if let Some(p) = pf!() {
            p.layer_count += 1;
        }

        *x = down;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::buffer::DType;
    use crate::profile::ops::PrefillOpProfiler;

    /// Verify that warn_gpu_fallback_once increments cpu_fallback_count every call
    /// (the dedupe is only for stderr, not for the profiler counter).
    #[test]
    fn test_fallback_profiler_count_increments_every_call() {
        let mut p = PrefillOpProfiler::new();
        assert_eq!(p.cpu_fallback_count, 0);

        warn_gpu_fallback_once(
            DType::Q4_0,
            64,
            "kv dtype Q4_0 not supported by flash_attn kernel",
            Some(&mut p),
        );
        assert_eq!(p.cpu_fallback_count, 1);

        warn_gpu_fallback_once(
            DType::Q4_0,
            64,
            "kv dtype Q4_0 not supported by flash_attn kernel",
            Some(&mut p),
        );
        assert_eq!(p.cpu_fallback_count, 2);
    }

    /// Verify that warn_gpu_fallback_once works without profiler (None path).
    #[test]
    fn test_fallback_no_profiler_does_not_panic() {
        // Should not panic even with no profiler attached.
        warn_gpu_fallback_once(
            DType::F32,
            128,
            "flash_attn kernel unavailable for this dtype/config",
            None,
        );
        warn_gpu_fallback_once(
            DType::F32,
            128,
            "flash_attn kernel unavailable for this dtype/config",
            None,
        );
    }

    /// Verify that the FALLBACK_WARNED set distinguishes different keys.
    #[test]
    fn test_fallback_different_keys_both_counted() {
        let mut p1 = PrefillOpProfiler::new();
        let mut p2 = PrefillOpProfiler::new();

        warn_gpu_fallback_once(
            DType::Q4_0,
            64,
            "kv dtype Q4_0 not supported by flash_attn kernel",
            Some(&mut p1),
        );
        warn_gpu_fallback_once(
            DType::F32,
            256,
            "head_dim not in {64, 128, 256} (no flash_attn DK variant compiled)",
            Some(&mut p2),
        );

        // Each profiler gets its own count incremented.
        assert_eq!(p1.cpu_fallback_count, 1);
        assert_eq!(p2.cpu_fallback_count, 1);
    }
}
