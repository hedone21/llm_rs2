//! Pre-bound kernel execution plan for GPU decode.
//!
//! Eliminates per-dispatch overhead by pre-binding all static kernel arguments
//! (weights, workspace buffers, dimensions) at plan build time. During execution,
//! only dynamic scalars (start_pos, cache_seq_len, write_pos) are updated.
//!
//! This mirrors llama.cpp's tight enqueue loop where kernel arguments rarely change
//! between tokens, achieving near-zero CPU overhead per dispatch.

use anyhow::{Context, Result};
use ocl::core::Kernel as CoreKernel;
use ocl::core::Mem;

/// Operation tag for profiling — maps to OpProfiler fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpTag {
    RmsNorm,
    MatmulQKV,
    Rope,
    KvScatter,
    Attention,
    MatmulWo,
    AddRmsNorm,
    AddAssign,
    MatmulGateUp,
    SiluMul,
    MatmulDown,
    FinalNorm,
    LmHead,
}

/// Dynamic argument that changes per token.
#[derive(Debug, Clone)]
pub enum DynamicArg {
    /// RoPE start position (i32)
    StartPos { arg_idx: u32 },
    /// Attention cache sequence length (i32)
    CacheSeqLen { arg_idx: u32 },
    /// KV scatter write position (i32)
    WritePos { arg_idx: u32 },
    /// KV cache capacity — changes on resize (i32)
    KvCapacity { arg_idx: u32 },
}

/// A single GPU kernel dispatch with pre-bound arguments.
pub struct KernelStep {
    /// Dedicated kernel object (cloned per step, args pre-bound)
    pub kernel: CoreKernel,
    /// Number of work dimensions (1, 2, or 3)
    pub ndim: u32,
    /// Global work size
    pub global_work_size: [usize; 3],
    /// Local work size (None = driver picks)
    pub local_work_size: Option<[usize; 3]>,
    /// Arguments that must be updated per token
    pub dynamic_args: Vec<DynamicArg>,
    /// Operation tag for profiling
    pub op_tag: OpTag,
}

// SAFETY: CoreKernel is a raw cl_kernel pointer. We guarantee single-threaded access
// during plan execution (same safety model as the UnsafeCell<KernelCache> in mod.rs).
unsafe impl Send for KernelStep {}
unsafe impl Sync for KernelStep {}

/// Execution plan for a single transformer layer.
pub struct LayerKernelPlan {
    /// Ordered kernel steps for this layer
    pub steps: Vec<KernelStep>,
    /// Whether to call clFlush after this layer's steps
    pub flush_after: bool,
}

/// Execution plan for the full model decode pass.
pub struct FullKernelPlan {
    /// Per-layer plans (indexed by layer number)
    pub layers: Vec<LayerKernelPlan>,
    /// Final RMSNorm step (model.norm)
    pub final_norm: KernelStep,
    /// lm_head matmul step
    pub lm_head: KernelStep,
    /// KV cache capacity at plan creation time (for invalidation check)
    pub kv_capacity: usize,
}

/// Error indicating the plan's pre-bound arguments are stale.
#[derive(Debug)]
pub struct PlanInvalidated;

impl std::fmt::Display for PlanInvalidated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Kernel plan invalidated (KV cache resized)")
    }
}

impl std::error::Error for PlanInvalidated {}

impl FullKernelPlan {
    /// Execute the full decode plan for one token.
    ///
    /// Updates only dynamic args (start_pos, cache_seq_len, write_pos),
    /// then enqueues all kernels in a tight loop with minimal CPU intervention.
    ///
    /// Returns `Err(PlanInvalidated)` if KV cache capacity changed.
    pub fn execute<C: crate::core::kv_cache::KVCacheOps>(
        &self,
        queue: &ocl::core::CommandQueue,
        start_pos: usize,
        kv_caches: &mut [C],
    ) -> std::result::Result<(), PlanInvalidated> {
        for (i, layer_plan) in self.layers.iter().enumerate() {
            let cache = &mut kv_caches[i];

            // Check for KV cache resize (plan invalidation)
            if cache.capacity() != self.kv_capacity {
                return Err(PlanInvalidated);
            }

            let cache_seq_len = cache.current_pos() as i32;
            let write_pos = cache.current_pos() as i32;
            let start_pos_i32 = start_pos as i32;

            for step in &layer_plan.steps {
                // Update dynamic args only (typically 1-3 per step)
                for dyn_arg in &step.dynamic_args {
                    unsafe {
                        match dyn_arg {
                            DynamicArg::StartPos { arg_idx } => {
                                ocl::core::set_kernel_arg(
                                    &step.kernel,
                                    *arg_idx,
                                    ocl::core::ArgVal::scalar(&start_pos_i32),
                                )
                                .ok();
                            }
                            DynamicArg::CacheSeqLen { arg_idx } => {
                                ocl::core::set_kernel_arg(
                                    &step.kernel,
                                    *arg_idx,
                                    ocl::core::ArgVal::scalar(&cache_seq_len),
                                )
                                .ok();
                            }
                            DynamicArg::WritePos { arg_idx } => {
                                ocl::core::set_kernel_arg(
                                    &step.kernel,
                                    *arg_idx,
                                    ocl::core::ArgVal::scalar(&write_pos),
                                )
                                .ok();
                            }
                            DynamicArg::KvCapacity { arg_idx } => {
                                let cap = cache.capacity() as i32;
                                ocl::core::set_kernel_arg(
                                    &step.kernel,
                                    *arg_idx,
                                    ocl::core::ArgVal::scalar(&cap),
                                )
                                .ok();
                            }
                        }
                    }
                }

                // Enqueue kernel — minimal CPU work
                unsafe {
                    ocl::core::enqueue_kernel(
                        queue,
                        &step.kernel,
                        step.ndim,
                        None,
                        &step.global_work_size,
                        step.local_work_size,
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )
                    .ok();
                }
            }

            cache.advance_pos(1);

            if layer_plan.flush_after {
                ocl::core::flush(queue).ok();
            }
        }

        // Final norm
        unsafe {
            ocl::core::enqueue_kernel(
                queue,
                &self.final_norm.kernel,
                self.final_norm.ndim,
                None,
                &self.final_norm.global_work_size,
                self.final_norm.local_work_size,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .ok();
        }

        // lm_head matmul
        unsafe {
            ocl::core::enqueue_kernel(
                queue,
                &self.lm_head.kernel,
                self.lm_head.ndim,
                None,
                &self.lm_head.global_work_size,
                self.lm_head.local_work_size,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .ok();
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Layer plan builder
// ---------------------------------------------------------------------------

/// All references needed to build a pre-bound kernel plan for one layer.
pub struct LayerPlanConfig<'a> {
    // Programs for kernel creation
    pub f16_program: &'a ocl::Program,
    pub simple_ops_program: &'a ocl::Program,
    // Buffer handles (cl_mem) — model weights
    pub x_buf: &'a Mem,
    pub wq_buf: &'a Mem,
    pub wk_buf: &'a Mem,
    pub wv_buf: &'a Mem,
    pub wo_buf: &'a Mem,
    pub w_gate_buf: &'a Mem,
    pub w_up_buf: &'a Mem,
    pub w_down_buf: &'a Mem,
    pub attn_norm_buf: &'a Mem,
    pub ffn_norm_buf: &'a Mem,
    // Workspace buffers
    pub q_buf: &'a Mem,
    pub k_buf: &'a Mem,
    pub v_buf: &'a Mem,
    pub out_attn_buf: &'a Mem,
    pub attn_out_buf: &'a Mem,
    pub gate_buf: &'a Mem,
    pub up_buf: &'a Mem,
    pub down_buf: &'a Mem,
    pub residual_buf: &'a Mem,
    // KV cache buffers
    pub k_cache_buf: &'a Mem,
    pub v_cache_buf: &'a Mem,
    // Dimensions
    pub dim: usize,
    pub n_heads_q: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_hidden: usize,
    pub n_q: usize,
    pub n_k: usize,
    pub n_v: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub kv_capacity: usize,
    // Attention layout info
    pub kv_pos_stride: i32,
    pub kv_head_stride: i32,
}

/// Helper: create a dedicated kernel and pre-bind F16 matmul arguments.
///
/// Matches the dispatch in `OpenClBackend::matmul_f16` for the decode case (m=1):
///   kernel_mul_mat_f16_f32(weight, 0, src, 0, dst, 0, ne00=k, ne01=n, ne02=1,
///                          ne10=k, ne12=1, ne0=n, ne1=1, r2=1, r3=1)
///   ndim=3, global=[ceil(n/4)*64, 1, 1], local=[64,1,1]
fn make_f16_matmul_step(
    program: &ocl::Program,
    src_buf: &Mem,
    weight_buf: &Mem,
    dst_buf: &Mem,
    n: usize,
    k: usize,
    op_tag: OpTag,
) -> Result<KernelStep> {
    let kernel = ocl::core::create_kernel(program, "kernel_mul_mat_f16_f32")
        .context("create kernel_mul_mat_f16_f32")?;

    let ne00 = k as i32;
    let ne01 = n as i32;
    let ne02 = 1i32;
    let ne10 = k as i32;
    let ne12 = 1i32;
    let ne0 = n as i32;
    let ne1 = 1i32; // m=1 for decode
    let r2 = 1i32;
    let r3 = 1i32;

    unsafe {
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(weight_buf))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::scalar(&0u64))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(src_buf))?;
        ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&0u64))?;
        ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::mem(dst_buf))?;
        ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&0u64))?;
        ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::scalar(&ne00))?;
        ocl::core::set_kernel_arg(&kernel, 7, ocl::core::ArgVal::scalar(&ne01))?;
        ocl::core::set_kernel_arg(&kernel, 8, ocl::core::ArgVal::scalar(&ne02))?;
        ocl::core::set_kernel_arg(&kernel, 9, ocl::core::ArgVal::scalar(&ne10))?;
        ocl::core::set_kernel_arg(&kernel, 10, ocl::core::ArgVal::scalar(&ne12))?;
        ocl::core::set_kernel_arg(&kernel, 11, ocl::core::ArgVal::scalar(&ne0))?;
        ocl::core::set_kernel_arg(&kernel, 12, ocl::core::ArgVal::scalar(&ne1))?;
        ocl::core::set_kernel_arg(&kernel, 13, ocl::core::ArgVal::scalar(&r2))?;
        ocl::core::set_kernel_arg(&kernel, 14, ocl::core::ArgVal::scalar(&r3))?;
    }

    // N_DST=4 rows per workgroup, 64 threads
    let group_size_0 = n.div_ceil(4);
    Ok(KernelStep {
        kernel,
        ndim: 3,
        global_work_size: [group_size_0 * 64, 1, 1],
        local_work_size: Some([64, 1, 1]),
        dynamic_args: vec![],
        op_tag,
    })
}

/// Build a pre-bound kernel execution plan for one transformer layer's decode
/// step (seq_len=1).
///
/// Creates dedicated kernel objects via `ocl::core::create_kernel` and pre-binds
/// all static arguments. Dynamic arguments (start_pos, cache_seq_len, write_pos)
/// are tagged with [`DynamicArg`] and set to initial value 0.
pub fn build_layer_plan(config: &LayerPlanConfig) -> Result<LayerKernelPlan> {
    let local_size = 64usize;
    let local_mem_bytes = local_size * std::mem::size_of::<f32>();
    let dim = config.dim;
    let k = dim; // matmul inner dim = hidden dim

    let mut steps = Vec::with_capacity(15);

    // -----------------------------------------------------------------------
    // 1. rms_norm_oop (x → residual)
    //    args: x, out, weight, dim(i32), eps(f32), local_mem
    // -----------------------------------------------------------------------
    {
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_rms_norm_oop")
            .context("create kernel_rms_norm_oop")?;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.x_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.residual_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(config.attn_norm_buf))?;
            ocl::core::set_kernel_arg(
                &kernel,
                3,
                ocl::core::ArgVal::scalar(&(dim as i32)),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                4,
                ocl::core::ArgVal::scalar(&config.rms_norm_eps),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                5,
                ocl::core::ArgVal::local::<f32>(&local_mem_bytes),
            )?;
        }
        steps.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [local_size, 1, 1], // rows=1 for decode
            local_work_size: Some([local_size, 1, 1]),
            dynamic_args: vec![],
            op_tag: OpTag::RmsNorm,
        });
    }

    // -----------------------------------------------------------------------
    // 2. matmul Q (residual → q)
    // -----------------------------------------------------------------------
    steps.push(make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.wq_buf,
        config.q_buf,
        config.n_q,
        k,
        OpTag::MatmulQKV,
    )?);

    // -----------------------------------------------------------------------
    // 3. matmul K (residual → k)
    // -----------------------------------------------------------------------
    steps.push(make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.wk_buf,
        config.k_buf,
        config.n_k,
        k,
        OpTag::MatmulQKV,
    )?);

    // -----------------------------------------------------------------------
    // 4. matmul V (residual → v)
    // -----------------------------------------------------------------------
    steps.push(make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.wv_buf,
        config.v_buf,
        config.n_v,
        k,
        OpTag::MatmulQKV,
    )?);

    // -----------------------------------------------------------------------
    // 5. rope Q (q inplace)
    //    args: buf, head_dim(i32), n_heads(i32), seq_len(i32), start_pos(i32), theta(f32)
    // -----------------------------------------------------------------------
    {
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_rope_simple")
            .context("create kernel_rope_simple (Q)")?;
        let seq_len_i32 = 1i32; // decode
        let start_pos_init = 0i32;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.q_buf))?;
            ocl::core::set_kernel_arg(
                &kernel,
                1,
                ocl::core::ArgVal::scalar(&(config.head_dim as i32)),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                2,
                ocl::core::ArgVal::scalar(&(config.n_heads_q as i32)),
            )?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&seq_len_i32))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&start_pos_init))?;
            ocl::core::set_kernel_arg(
                &kernel,
                5,
                ocl::core::ArgVal::scalar(&config.rope_theta),
            )?;
        }
        let work_size = config.n_heads_q * (config.head_dim / 2);
        steps.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [work_size, 1, 1],
            local_work_size: None,
            dynamic_args: vec![DynamicArg::StartPos { arg_idx: 4 }],
            op_tag: OpTag::Rope,
        });
    }

    // -----------------------------------------------------------------------
    // 6. rope K (k inplace)
    // -----------------------------------------------------------------------
    {
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_rope_simple")
            .context("create kernel_rope_simple (K)")?;
        let seq_len_i32 = 1i32;
        let start_pos_init = 0i32;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.k_buf))?;
            ocl::core::set_kernel_arg(
                &kernel,
                1,
                ocl::core::ArgVal::scalar(&(config.head_dim as i32)),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                2,
                ocl::core::ArgVal::scalar(&(config.n_kv_heads as i32)),
            )?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&seq_len_i32))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&start_pos_init))?;
            ocl::core::set_kernel_arg(
                &kernel,
                5,
                ocl::core::ArgVal::scalar(&config.rope_theta),
            )?;
        }
        let work_size = config.n_kv_heads * (config.head_dim / 2);
        steps.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [work_size, 1, 1],
            local_work_size: None,
            dynamic_args: vec![DynamicArg::StartPos { arg_idx: 4 }],
            op_tag: OpTag::Rope,
        });
    }

    // -----------------------------------------------------------------------
    // 7. kv_scatter (k,v → cache)
    //    args: k_src, v_src, k_dst, v_dst, head_dim(i32), capacity(i32), write_pos(i32)
    // -----------------------------------------------------------------------
    {
        let kernel =
            ocl::core::create_kernel(config.simple_ops_program, "kernel_kv_scatter_f32_to_f16")
                .context("create kernel_kv_scatter_f32_to_f16")?;
        let capacity_init = config.kv_capacity as i32;
        let write_pos_init = 0i32;
        let n_elems = config.n_kv_heads * config.head_dim;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.k_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.v_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(config.k_cache_buf))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::mem(config.v_cache_buf))?;
            ocl::core::set_kernel_arg(
                &kernel,
                4,
                ocl::core::ArgVal::scalar(&(config.head_dim as i32)),
            )?;
            ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&capacity_init))?;
            ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::scalar(&write_pos_init))?;
        }
        steps.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [n_elems.div_ceil(64) * 64, 1, 1],
            local_work_size: Some([64, 1, 1]),
            dynamic_args: vec![
                DynamicArg::KvCapacity { arg_idx: 5 },
                DynamicArg::WritePos { arg_idx: 6 },
            ],
            op_tag: OpTag::KvScatter,
        });
    }

    // -----------------------------------------------------------------------
    // 8. attention (q, kv_cache → out_attn)
    //    args: q, k_cache, v_cache, out, head_dim(i32), n_heads_q(i32),
    //          n_heads_kv(i32), cache_seq_len(i32), scale(f32),
    //          kv_pos_stride(i32), kv_head_stride(i32), local_mem
    // -----------------------------------------------------------------------
    {
        let kernel =
            ocl::core::create_kernel(config.simple_ops_program, "kernel_attn_gen_half")
                .context("create kernel_attn_gen_half")?;
        let scale = 1.0f32 / (config.head_dim as f32).sqrt();
        let cache_seq_len_init = 0i32;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.q_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.k_cache_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(config.v_cache_buf))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::mem(config.out_attn_buf))?;
            ocl::core::set_kernel_arg(
                &kernel,
                4,
                ocl::core::ArgVal::scalar(&(config.head_dim as i32)),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                5,
                ocl::core::ArgVal::scalar(&(config.n_heads_q as i32)),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                6,
                ocl::core::ArgVal::scalar(&(config.n_kv_heads as i32)),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                7,
                ocl::core::ArgVal::scalar(&cache_seq_len_init),
            )?;
            ocl::core::set_kernel_arg(&kernel, 8, ocl::core::ArgVal::scalar(&scale))?;
            ocl::core::set_kernel_arg(
                &kernel,
                9,
                ocl::core::ArgVal::scalar(&config.kv_pos_stride),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                10,
                ocl::core::ArgVal::scalar(&config.kv_head_stride),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                11,
                ocl::core::ArgVal::local::<f32>(&local_mem_bytes),
            )?;
        }
        steps.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [config.n_heads_q * local_size, 1, 1],
            local_work_size: Some([local_size, 1, 1]),
            dynamic_args: vec![DynamicArg::CacheSeqLen { arg_idx: 7 }],
            op_tag: OpTag::Attention,
        });
    }

    // -----------------------------------------------------------------------
    // 9. matmul Wo (out_attn → attn_out)
    // -----------------------------------------------------------------------
    steps.push(make_f16_matmul_step(
        config.f16_program,
        config.out_attn_buf,
        config.wo_buf,
        config.attn_out_buf,
        dim, // wo output dim = hidden dim
        dim, // wo inner dim = n_heads_q * head_dim = dim
        OpTag::MatmulWo,
    )?);

    // -----------------------------------------------------------------------
    // 10. add_rms_norm_oop (x += attn_out, then norm → residual)
    //     args: x, attn_out, residual, ffn_norm, dim(i32), eps(f32), local_mem
    // -----------------------------------------------------------------------
    {
        let kernel =
            ocl::core::create_kernel(config.simple_ops_program, "kernel_add_rms_norm_oop")
                .context("create kernel_add_rms_norm_oop")?;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.x_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.attn_out_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(config.residual_buf))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::mem(config.ffn_norm_buf))?;
            ocl::core::set_kernel_arg(
                &kernel,
                4,
                ocl::core::ArgVal::scalar(&(dim as i32)),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                5,
                ocl::core::ArgVal::scalar(&config.rms_norm_eps),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                6,
                ocl::core::ArgVal::local::<f32>(&local_mem_bytes),
            )?;
        }
        steps.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [local_size, 1, 1],
            local_work_size: Some([local_size, 1, 1]),
            dynamic_args: vec![],
            op_tag: OpTag::AddRmsNorm,
        });
    }

    // -----------------------------------------------------------------------
    // 11. matmul gate (residual → gate)
    // -----------------------------------------------------------------------
    steps.push(make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.w_gate_buf,
        config.gate_buf,
        config.ffn_hidden,
        k,
        OpTag::MatmulGateUp,
    )?);

    // -----------------------------------------------------------------------
    // 12. matmul up (residual → up)
    // -----------------------------------------------------------------------
    steps.push(make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.w_up_buf,
        config.up_buf,
        config.ffn_hidden,
        k,
        OpTag::MatmulGateUp,
    )?);

    // -----------------------------------------------------------------------
    // 13. silu_mul (gate = silu(gate) * up)
    //     args: gate, up, size4(i32)
    // -----------------------------------------------------------------------
    {
        let kernel =
            ocl::core::create_kernel(config.simple_ops_program, "kernel_silu_mul_simple")
                .context("create kernel_silu_mul_simple")?;
        let size4 = (config.ffn_hidden / 4) as i32;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.gate_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.up_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&size4))?;
        }
        steps.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [config.ffn_hidden / 4, 1, 1],
            local_work_size: None,
            dynamic_args: vec![],
            op_tag: OpTag::SiluMul,
        });
    }

    // -----------------------------------------------------------------------
    // 14. matmul down (gate → down)
    // -----------------------------------------------------------------------
    steps.push(make_f16_matmul_step(
        config.f16_program,
        config.gate_buf,
        config.w_down_buf,
        config.down_buf,
        dim,
        config.ffn_hidden,
        OpTag::MatmulDown,
    )?);

    // -----------------------------------------------------------------------
    // 15. add_assign (x += down)
    //     args: x, down, size4(i32)
    // -----------------------------------------------------------------------
    {
        let kernel =
            ocl::core::create_kernel(config.simple_ops_program, "kernel_add_assign_simple")
                .context("create kernel_add_assign_simple")?;
        let size4 = (dim / 4) as i32;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.x_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.down_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&size4))?;
        }
        steps.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [dim / 4, 1, 1],
            local_work_size: None,
            dynamic_args: vec![],
            op_tag: OpTag::AddAssign,
        });
    }

    Ok(LayerKernelPlan {
        steps,
        flush_after: true,
    })
}
