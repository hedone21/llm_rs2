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
    // KIVI-specific dynamic args
    /// Residual write position (i32) — kivi_gather_update res_pos arg
    ResPos { arg_idx: u32 },
    /// Number of quantized tokens (i32) — kivi attention q2_tokens arg
    Q2Tokens { arg_idx: u32 },
    /// Number of valid residual tokens (i32) — kivi attention res_tokens arg
    ResTokens { arg_idx: u32 },
    /// Tok base offset for scatter (i32) — q2_tokens passed to scatter
    TokBase { arg_idx: u32 },
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
    /// Buffers created during plan build that must be kept alive while the plan
    /// exists. Dropping them would invalidate the cl_mem handles pre-bound to
    /// the kernel, leading to SIGSEGV on dispatch.
    #[allow(dead_code)]
    pub retained_bufs: Vec<Mem>,
}

// SAFETY: CoreKernel is a raw cl_kernel pointer. We guarantee single-threaded access
// during plan execution (same safety model as the UnsafeCell<KernelCache> in mod.rs).
unsafe impl Send for KernelStep {}
unsafe impl Sync for KernelStep {}

/// KV update variant — Standard scatter or KIVI gather-to-residual.
pub enum KvUpdateVariant {
    /// Standard F16 scatter: k,v → kv_cache
    Standard(KernelStep),
    /// KIVI gather: k → res_k, v → res_v (two separate kernels)
    Kivi {
        gather_k: KernelStep,
        gather_v: KernelStep,
    },
}

/// Attention variant — Standard half-precision or KIVI fused attention.
#[allow(clippy::large_enum_variant)]
pub enum AttentionVariant {
    /// Standard attention: kernel_attn_gen_half on F16 KV cache
    Standard(KernelStep),
    /// KIVI assembled: scatter residual to F32 attn buffer, then standard F32 attention.
    /// Used when native KIVI attention kernel is not available.
    KiviAssembled {
        scatter_k: KernelStep,
        scatter_v: KernelStep,
        attn: KernelStep,
    },
    /// KIVI native: fused Q2/Q4/Q8 + residual attention in a single kernel.
    KiviNative(KernelStep),
}

/// Execution plan for a single transformer layer.
pub struct LayerKernelPlan {
    /// Steps 1-6: RMSNorm, QKV matmul, RoPE Q, RoPE K
    pub steps_pre_kv: Vec<KernelStep>,
    /// Step 7: KV update (Standard scatter or KIVI gather)
    pub kv_update: KvUpdateVariant,
    /// Step 8: Attention (Standard, KIVI assembled, or KIVI native)
    pub attention: AttentionVariant,
    /// Steps 9-15: Wo matmul, add+RMSNorm, FFN, add_assign
    pub steps_post_attn: Vec<KernelStep>,
    /// Whether to call clFlush after this layer's steps
    pub flush_after: bool,
}

/// Execution plan for the full model decode pass.
pub struct FullKernelPlan {
    /// Per-layer plans (indexed by layer number)
    pub layers: Vec<LayerKernelPlan>,
    /// Final RMSNorm step (model.norm)
    pub final_norm: KernelStep,
    /// lm_head matmul step.
    /// `None` when lm_head is kept on CPU (e.g. gemma3's 604 MB tied embedding).
    /// The caller must run CPU-side matmul when this is `None`.
    pub lm_head: Option<KernelStep>,
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
    /// Dispatch a single kernel step, updating its dynamic args.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn dispatch_step(
        queue: &ocl::core::CommandQueue,
        step: &KernelStep,
        start_pos: i32,
        cache_seq_len: i32,
        write_pos: i32,
        kv_capacity: i32,
        res_pos: i32,
        q2_tokens: i32,
        res_tokens: i32,
    ) {
        for dyn_arg in &step.dynamic_args {
            let (arg_idx, result) = unsafe {
                match dyn_arg {
                    DynamicArg::StartPos { arg_idx } => (
                        *arg_idx,
                        ocl::core::set_kernel_arg(
                            &step.kernel,
                            *arg_idx,
                            ocl::core::ArgVal::scalar(&start_pos),
                        ),
                    ),
                    DynamicArg::CacheSeqLen { arg_idx } => (
                        *arg_idx,
                        ocl::core::set_kernel_arg(
                            &step.kernel,
                            *arg_idx,
                            ocl::core::ArgVal::scalar(&cache_seq_len),
                        ),
                    ),
                    DynamicArg::WritePos { arg_idx } => (
                        *arg_idx,
                        ocl::core::set_kernel_arg(
                            &step.kernel,
                            *arg_idx,
                            ocl::core::ArgVal::scalar(&write_pos),
                        ),
                    ),
                    DynamicArg::KvCapacity { arg_idx } => (
                        *arg_idx,
                        ocl::core::set_kernel_arg(
                            &step.kernel,
                            *arg_idx,
                            ocl::core::ArgVal::scalar(&kv_capacity),
                        ),
                    ),
                    DynamicArg::ResPos { arg_idx } => (
                        *arg_idx,
                        ocl::core::set_kernel_arg(
                            &step.kernel,
                            *arg_idx,
                            ocl::core::ArgVal::scalar(&res_pos),
                        ),
                    ),
                    DynamicArg::Q2Tokens { arg_idx } => (
                        *arg_idx,
                        ocl::core::set_kernel_arg(
                            &step.kernel,
                            *arg_idx,
                            ocl::core::ArgVal::scalar(&q2_tokens),
                        ),
                    ),
                    DynamicArg::ResTokens { arg_idx } => (
                        *arg_idx,
                        ocl::core::set_kernel_arg(
                            &step.kernel,
                            *arg_idx,
                            ocl::core::ArgVal::scalar(&res_tokens),
                        ),
                    ),
                    DynamicArg::TokBase { arg_idx } => (
                        *arg_idx,
                        ocl::core::set_kernel_arg(
                            &step.kernel,
                            *arg_idx,
                            ocl::core::ArgVal::scalar(&q2_tokens),
                        ),
                    ),
                }
            };
            if let Err(e) = result {
                log::error!(
                    "Plan set_kernel_arg failed: op={:?} arg_idx={}: {}",
                    step.op_tag,
                    arg_idx,
                    e
                );
            }
        }
        unsafe {
            if let Err(e) = ocl::core::enqueue_kernel(
                queue,
                &step.kernel,
                step.ndim,
                None,
                &step.global_work_size,
                step.local_work_size,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            ) {
                log::error!(
                    "Plan enqueue_kernel failed: op={:?} gws={:?}: {}",
                    step.op_tag,
                    step.global_work_size,
                    e
                );
            }
        }
    }

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
        let debug_sync = std::env::var("PLAN_DEBUG").is_ok();

        for (i, layer_plan) in self.layers.iter().enumerate() {
            let cache = &mut kv_caches[i];

            // Check for KV cache resize or capacity overflow (plan invalidation)
            if cache.capacity() != self.kv_capacity || cache.current_pos() >= cache.capacity() {
                return Err(PlanInvalidated);
            }

            let cache_seq_len = cache.current_pos() as i32;
            let write_pos = cache.current_pos() as i32;
            let start_pos_i32 = start_pos as i32;
            let kv_cap = cache.capacity() as i32;
            let rp = cache.res_pos() as i32;
            let q2t = cache.q2_tokens() as i32;
            let rt = rp; // res_tokens = res_pos before advance

            // Attention sees the token we just scattered
            let attn_seq_len = cache_seq_len + 1;

            // Steps 1-6: pre-KV steps
            for (si, step) in layer_plan.steps_pre_kv.iter().enumerate() {
                Self::dispatch_step(
                    queue,
                    step,
                    start_pos_i32,
                    cache_seq_len,
                    write_pos,
                    kv_cap,
                    rp,
                    q2t,
                    rt,
                );
                if debug_sync {
                    ocl::core::finish(queue).ok();
                    eprintln!(
                        "[Plan] L{} pre_kv[{}] {:?} OK (pos={}, cap={})",
                        i, si, step.op_tag, start_pos, kv_cap
                    );
                }
            }
            // Step 7: KV update
            match &layer_plan.kv_update {
                KvUpdateVariant::Standard(step) => {
                    Self::dispatch_step(
                        queue,
                        step,
                        start_pos_i32,
                        cache_seq_len,
                        write_pos,
                        kv_cap,
                        rp,
                        q2t,
                        rt,
                    );
                    if debug_sync {
                        ocl::core::finish(queue).ok();
                        eprintln!(
                            "[Plan] L{} kv_scatter OK (write_pos={}, cap={})",
                            i, write_pos, kv_cap
                        );
                    }
                }
                KvUpdateVariant::Kivi { gather_k, gather_v } => {
                    Self::dispatch_step(
                        queue,
                        gather_k,
                        start_pos_i32,
                        cache_seq_len,
                        write_pos,
                        kv_cap,
                        rp,
                        q2t,
                        rt,
                    );
                    Self::dispatch_step(
                        queue,
                        gather_v,
                        start_pos_i32,
                        cache_seq_len,
                        write_pos,
                        kv_cap,
                        rp,
                        q2t,
                        rt,
                    );
                }
            }

            // After KV update, res_tokens is res_pos + 1 for attention
            let rt_after = rp + 1;

            // Step 8: Attention — uses attn_seq_len (includes just-scattered token)
            match &layer_plan.attention {
                AttentionVariant::Standard(step) => {
                    if debug_sync {
                        eprintln!(
                            "[Plan] L{} attention dispatch (attn_seq_len={}, gws={:?}, lws={:?})",
                            i, attn_seq_len, step.global_work_size, step.local_work_size
                        );
                    }
                    Self::dispatch_step(
                        queue,
                        step,
                        start_pos_i32,
                        attn_seq_len,
                        write_pos,
                        kv_cap,
                        rp,
                        q2t,
                        rt,
                    );
                    if debug_sync {
                        eprintln!("[Plan] L{} attention enqueued, calling finish...", i);
                        ocl::core::finish(queue).ok();
                        eprintln!("[Plan] L{} attention OK (attn_seq_len={})", i, attn_seq_len);
                    }
                }
                AttentionVariant::KiviAssembled {
                    scatter_k,
                    scatter_v,
                    attn,
                } => {
                    // Scatter residual to F32 attn buffer (with updated res_tokens)
                    Self::dispatch_step(
                        queue,
                        scatter_k,
                        start_pos_i32,
                        cache_seq_len,
                        write_pos,
                        kv_cap,
                        rp,
                        q2t,
                        rt_after,
                    );
                    Self::dispatch_step(
                        queue,
                        scatter_v,
                        start_pos_i32,
                        cache_seq_len,
                        write_pos,
                        kv_cap,
                        rp,
                        q2t,
                        rt_after,
                    );
                    // Attention over total = q2_tokens + res_tokens_after
                    let total = q2t + rt_after;
                    Self::dispatch_step(
                        queue,
                        attn,
                        start_pos_i32,
                        total,
                        write_pos,
                        kv_cap,
                        rp,
                        q2t,
                        rt_after,
                    );
                }
                AttentionVariant::KiviNative(step) => {
                    // Native KIVI attention: q2_tokens and res_tokens are dynamic
                    Self::dispatch_step(
                        queue,
                        step,
                        start_pos_i32,
                        cache_seq_len,
                        write_pos,
                        kv_cap,
                        rp,
                        q2t,
                        rt_after,
                    );
                }
            }

            // Steps 9-15: post-attention steps
            for (si, step) in layer_plan.steps_post_attn.iter().enumerate() {
                Self::dispatch_step(
                    queue,
                    step,
                    start_pos_i32,
                    cache_seq_len,
                    write_pos,
                    kv_cap,
                    rp,
                    q2t,
                    rt,
                );
                if debug_sync {
                    ocl::core::finish(queue).ok();
                    eprintln!("[Plan] L{} post_attn[{}] {:?} OK", i, si, step.op_tag);
                }
            }
            cache.advance_pos(1);

            if layer_plan.flush_after
                && let Err(e) = ocl::core::flush(queue)
            {
                log::error!("Plan flush failed: {}", e);
            }
        }

        // Final norm
        unsafe {
            if let Err(e) = ocl::core::enqueue_kernel(
                queue,
                &self.final_norm.kernel,
                self.final_norm.ndim,
                None,
                &self.final_norm.global_work_size,
                self.final_norm.local_work_size,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            ) {
                log::error!("Plan enqueue final_norm failed: {}", e);
            }
        }

        // lm_head matmul (skipped when lm_head is on CPU)
        if let Some(ref lm_head) = self.lm_head {
            unsafe {
                if let Err(e) = ocl::core::enqueue_kernel(
                    queue,
                    &lm_head.kernel,
                    lm_head.ndim,
                    None,
                    &lm_head.global_work_size,
                    lm_head.local_work_size,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                ) {
                    log::error!("Plan enqueue lm_head failed: {}", e);
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Layer plan builder
// ---------------------------------------------------------------------------

/// All references needed to build a pre-bound kernel plan for one layer.
pub struct LayerPlanConfig<'a> {
    // Context for buffer creation
    pub context: &'a ocl::Context,
    // Programs for kernel creation
    pub f16_program: &'a ocl::Program,
    pub f16_l4_program: Option<&'a ocl::Program>,
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
    /// Whether the device lacks subgroup support (nosub fallback path).
    pub is_nosub: bool,
}

/// N threshold for switching from N_DST=2 (128 rows/WG) to N_DST=4 (256 rows/WG) GEMV kernel.
/// At N > 4096, the L4 kernel halves WG count and reuses activation across 4 rows.
const LARGE_N_THRESHOLD: usize = 4096;

/// Helper: create a dedicated kernel and pre-bind F16 matmul arguments.
///
/// Must mirror `OpenClBackend::matmul_f16` dispatch exactly.
///
/// Three dispatch shapes depending on which kernel is loaded:
/// - 4-wave (default): local=[64,4,1], N_DST=2 → 128 rows/WG,
///   global=[ceil(n/128)*64, 4, 1] for m=1 decode.
/// - 4-wave L4 (large-N): local=[64,4,1], N_DST=4 → 256 rows/WG,
///   global=[ceil(n/256)*64, 4, 1]. Used when l4_program is provided and n > LARGE_N_THRESHOLD.
/// - Nosub fallback: local=[64,1,1], N_DST=4 → 4 rows/WG,
///   global=[ceil(n/4)*64, 1, 1].
#[allow(clippy::too_many_arguments)]
fn make_f16_matmul_step(
    program: &ocl::Program,
    src_buf: &Mem,
    weight_buf: &Mem,
    dst_buf: &Mem,
    n: usize,
    k: usize,
    op_tag: OpTag,
    l4_program: Option<&ocl::Program>,
    is_nosub: bool,
) -> Result<KernelStep> {
    // Prefer L4 (N_DST=4, 4 rows/WG) for large-N when both conditions hold:
    // - 4-wave path is active (nosub fallback has incompatible dispatch)
    // - l4_program compiled successfully
    // - n exceeds the threshold where halving WG count pays off
    let use_l4 = !is_nosub && l4_program.is_some() && n > LARGE_N_THRESHOLD;
    let kernel = if use_l4 {
        ocl::core::create_kernel(l4_program.unwrap(), "kernel_mul_mat_f16_f32_l4")
            .context("create kernel_mul_mat_f16_f32_l4")?
    } else {
        ocl::core::create_kernel(program, "kernel_mul_mat_f16_f32")
            .context("create kernel_mul_mat_f16_f32")?
    };

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

    let (global_work_size, local_work_size) = if is_nosub {
        // Nosub: single-subgroup WG, 4 rows/WG.
        const NOSUB_N_DST: usize = 4;
        let n_groups = n.div_ceil(NOSUB_N_DST);
        ([n_groups * 64, 1, 1], [64usize, 1, 1])
    } else if use_l4 {
        // 4-wave L4: 4 waves × 64 lanes cooperate per row-group, N_DST=4.
        const L4_N_DST: usize = 4;
        let n_groups = n.div_ceil(L4_N_DST);
        // m=1 for decode → dim1 = m*4 = 4
        ([n_groups * 64, 4, 1], [64usize, 4, 1])
    } else {
        // 4-wave: 4 waves × 64 lanes cooperate per row-group, N_DST=2.
        const WAVE4_N_DST: usize = 2;
        let n_groups = n.div_ceil(WAVE4_N_DST);
        ([n_groups * 64, 4, 1], [64usize, 4, 1])
    };

    Ok(KernelStep {
        kernel,
        ndim: 3,
        global_work_size,
        local_work_size: Some(local_work_size),
        dynamic_args: vec![],
        op_tag,
        retained_bufs: vec![],
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

    let mut steps_pre_kv = Vec::with_capacity(6);

    // -----------------------------------------------------------------------
    // 1. rms_norm_oop (x -> residual)
    // -----------------------------------------------------------------------
    {
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_rms_norm_oop")
            .context("create kernel_rms_norm_oop")?;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.x_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.residual_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(config.attn_norm_buf))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&config.rms_norm_eps))?;
            // add_unit = 0 (non-Gemma3; Plan does not support Gemma3 yet)
            ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&0i32))?;
            ocl::core::set_kernel_arg(
                &kernel,
                6,
                ocl::core::ArgVal::local::<f32>(&local_mem_bytes),
            )?;
        }
        steps_pre_kv.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [local_size, 1, 1],
            local_work_size: Some([local_size, 1, 1]),
            dynamic_args: vec![],
            op_tag: OpTag::RmsNorm,
            retained_bufs: vec![],
        });
    }

    // -----------------------------------------------------------------------
    // 2. matmul Q (residual -> q)
    // -----------------------------------------------------------------------
    steps_pre_kv.push(make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.wq_buf,
        config.q_buf,
        config.n_q,
        k,
        OpTag::MatmulQKV,
        None,
        config.is_nosub,
    )?);

    // -----------------------------------------------------------------------
    // 3. matmul K (residual -> k)
    // -----------------------------------------------------------------------
    steps_pre_kv.push(make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.wk_buf,
        config.k_buf,
        config.n_k,
        k,
        OpTag::MatmulQKV,
        None,
        config.is_nosub,
    )?);

    // -----------------------------------------------------------------------
    // 4. matmul V (residual -> v)
    // -----------------------------------------------------------------------
    steps_pre_kv.push(make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.wv_buf,
        config.v_buf,
        config.n_v,
        k,
        OpTag::MatmulQKV,
        None,
        config.is_nosub,
    )?);

    // -----------------------------------------------------------------------
    // 5. rope Q (q inplace)
    // -----------------------------------------------------------------------
    {
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_rope_simple")
            .context("create kernel_rope_simple (Q)")?;
        let seq_len_i32 = 1i32;
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
            ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&config.rope_theta))?;
        }
        let work_size = config.n_heads_q * (config.head_dim / 2);
        steps_pre_kv.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [work_size, 1, 1],
            local_work_size: None,
            dynamic_args: vec![DynamicArg::StartPos { arg_idx: 4 }],
            op_tag: OpTag::Rope,
            retained_bufs: vec![],
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
            ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&config.rope_theta))?;
        }
        let work_size = config.n_kv_heads * (config.head_dim / 2);
        steps_pre_kv.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [work_size, 1, 1],
            local_work_size: None,
            dynamic_args: vec![DynamicArg::StartPos { arg_idx: 4 }],
            op_tag: OpTag::Rope,
            retained_bufs: vec![],
        });
    }

    // -----------------------------------------------------------------------
    // 7. kv_scatter (k,v -> cache) — Standard variant
    // -----------------------------------------------------------------------
    let kv_update = {
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
        KvUpdateVariant::Standard(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [n_elems.div_ceil(64) * 64, 1, 1],
            local_work_size: Some([64, 1, 1]),
            dynamic_args: vec![
                DynamicArg::KvCapacity { arg_idx: 5 },
                DynamicArg::WritePos { arg_idx: 6 },
            ],
            op_tag: OpTag::KvScatter,
            retained_bufs: vec![],
        })
    };

    // -----------------------------------------------------------------------
    // 8. attention (q, kv_cache -> out_attn) — Standard variant
    // -----------------------------------------------------------------------
    let attention = {
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_attn_gen_half")
            .context("create kernel_attn_gen_half")?;
        let scale = 1.0f32 / (config.head_dim as f32).sqrt();
        let cache_seq_len_init = 0i32;
        let write_scores = 0i32;
        let score_stride = 0i32;
        let dummy_score_buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                config.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                1,
                None,
            )
        }
        .context("create dummy score buffer for plan")?;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.q_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.k_cache_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(config.v_cache_buf))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::mem(config.out_attn_buf))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::mem(&dummy_score_buf))?;
            ocl::core::set_kernel_arg(
                &kernel,
                5,
                ocl::core::ArgVal::scalar(&(config.head_dim as i32)),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                6,
                ocl::core::ArgVal::scalar(&(config.n_heads_q as i32)),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                7,
                ocl::core::ArgVal::scalar(&(config.n_kv_heads as i32)),
            )?;
            ocl::core::set_kernel_arg(&kernel, 8, ocl::core::ArgVal::scalar(&cache_seq_len_init))?;
            ocl::core::set_kernel_arg(&kernel, 9, ocl::core::ArgVal::scalar(&scale))?;
            ocl::core::set_kernel_arg(
                &kernel,
                10,
                ocl::core::ArgVal::scalar(&config.kv_pos_stride),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                11,
                ocl::core::ArgVal::scalar(&config.kv_head_stride),
            )?;
            ocl::core::set_kernel_arg(&kernel, 12, ocl::core::ArgVal::scalar(&write_scores))?;
            ocl::core::set_kernel_arg(&kernel, 13, ocl::core::ArgVal::scalar(&score_stride))?;
            ocl::core::set_kernel_arg(
                &kernel,
                14,
                ocl::core::ArgVal::local::<f32>(&local_mem_bytes),
            )?;
        }
        AttentionVariant::Standard(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [config.n_heads_q * local_size, 1, 1],
            local_work_size: Some([local_size, 1, 1]),
            dynamic_args: vec![DynamicArg::CacheSeqLen { arg_idx: 8 }],
            op_tag: OpTag::Attention,
            retained_bufs: vec![dummy_score_buf],
        })
    };

    // -----------------------------------------------------------------------
    // Steps 9-15: post-attention steps
    // -----------------------------------------------------------------------
    let mut steps_post_attn = Vec::with_capacity(7);

    // 9. matmul Wo (out_attn -> attn_out)
    steps_post_attn.push(make_f16_matmul_step(
        config.f16_program,
        config.out_attn_buf,
        config.wo_buf,
        config.attn_out_buf,
        dim,
        dim,
        OpTag::MatmulWo,
        None,
        config.is_nosub,
    )?);

    // 10. add_rms_norm_oop (x += attn_out, then norm -> residual)
    {
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_add_rms_norm_oop")
            .context("create kernel_add_rms_norm_oop")?;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.x_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.attn_out_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(config.residual_buf))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::mem(config.ffn_norm_buf))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&config.rms_norm_eps))?;
            // add_unit = 0 (non-Gemma3)
            ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::scalar(&0i32))?;
            ocl::core::set_kernel_arg(
                &kernel,
                7,
                ocl::core::ArgVal::local::<f32>(&local_mem_bytes),
            )?;
        }
        steps_post_attn.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [local_size, 1, 1],
            local_work_size: Some([local_size, 1, 1]),
            dynamic_args: vec![],
            op_tag: OpTag::AddRmsNorm,
            retained_bufs: vec![],
        });
    }

    // 11. matmul gate (residual -> gate)
    steps_post_attn.push(make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.w_gate_buf,
        config.gate_buf,
        config.ffn_hidden,
        k,
        OpTag::MatmulGateUp,
        config.f16_l4_program,
        config.is_nosub,
    )?);

    // 12. matmul up (residual -> up)
    steps_post_attn.push(make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.w_up_buf,
        config.up_buf,
        config.ffn_hidden,
        k,
        OpTag::MatmulGateUp,
        config.f16_l4_program,
        config.is_nosub,
    )?);

    // 13. silu_mul (gate = silu(gate) * up)
    {
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_silu_mul_simple")
            .context("create kernel_silu_mul_simple")?;
        let size4 = (config.ffn_hidden / 4) as i32;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.gate_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.up_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&size4))?;
        }
        steps_post_attn.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [config.ffn_hidden / 4, 1, 1],
            local_work_size: None,
            dynamic_args: vec![],
            op_tag: OpTag::SiluMul,
            retained_bufs: vec![],
        });
    }

    // 14. matmul down (gate -> down)
    steps_post_attn.push(make_f16_matmul_step(
        config.f16_program,
        config.gate_buf,
        config.w_down_buf,
        config.down_buf,
        dim,
        config.ffn_hidden,
        OpTag::MatmulDown,
        None,
        config.is_nosub,
    )?);

    // 15. add_assign (x += down)
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
        steps_post_attn.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [dim / 4, 1, 1],
            local_work_size: None,
            dynamic_args: vec![],
            op_tag: OpTag::AddAssign,
            retained_bufs: vec![],
        });
    }

    Ok(LayerKernelPlan {
        steps_pre_kv,
        kv_update,
        attention,
        steps_post_attn,
        flush_after: false,
    })
}

// ---------------------------------------------------------------------------
// Full model plan builder
// ---------------------------------------------------------------------------

/// Config for building the full model plan (all layers + final norm + lm_head).
pub struct FullPlanConfig<'a> {
    pub context: &'a ocl::Context,
    pub f16_program: &'a ocl::Program,
    pub f16_l4_program: Option<&'a ocl::Program>,
    pub simple_ops_program: &'a ocl::Program,
    // Per-layer weight buffers: Vec<(wq, wk, wv, wo, w_gate, w_up, w_down, attn_norm, ffn_norm)>
    pub layer_bufs: Vec<LayerBufs<'a>>,
    // Workspace buffers (shared across layers)
    pub x_buf: &'a Mem,
    pub q_buf: &'a Mem,
    pub k_buf: &'a Mem,
    pub v_buf: &'a Mem,
    pub out_attn_buf: &'a Mem,
    pub attn_out_buf: &'a Mem,
    pub gate_buf: &'a Mem,
    pub up_buf: &'a Mem,
    pub down_buf: &'a Mem,
    pub residual_buf: &'a Mem,
    // Per-layer KV cache buffers
    pub kv_bufs: Vec<KvBufs<'a>>,
    // Final norm + lm_head
    pub final_norm_buf: &'a Mem,
    /// `None` when lm_head is kept on CPU (large tied embedding).
    pub lm_head_buf: Option<&'a Mem>,
    pub logits_buf: &'a Mem,
    // Model config
    pub dim: usize,
    pub n_heads_q: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_hidden: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub kv_capacity: usize,
    pub kv_pos_stride: i32,
    pub kv_head_stride: i32,
    /// Whether the device lacks subgroup support (nosub fallback path).
    pub is_nosub: bool,
}

/// Per-layer weight buffer references.
pub struct LayerBufs<'a> {
    pub wq: &'a Mem,
    pub wk: &'a Mem,
    pub wv: &'a Mem,
    pub wo: &'a Mem,
    pub w_gate: &'a Mem,
    pub w_up: &'a Mem,
    pub w_down: &'a Mem,
    pub attn_norm: &'a Mem,
    pub ffn_norm: &'a Mem,
}

/// Per-layer KV cache buffer references.
pub struct KvBufs<'a> {
    pub k_cache: &'a Mem,
    pub v_cache: &'a Mem,
}

/// Build a pre-bound plan for the full model decode pass (all layers + head).
pub fn build_full_plan(config: &FullPlanConfig) -> Result<FullKernelPlan> {
    let n_q = config.n_heads_q * config.head_dim;
    let n_k = config.n_kv_heads * config.head_dim;
    let n_v = n_k;

    let mut layers = Vec::with_capacity(config.layer_bufs.len());
    for (i, (lb, kb)) in config
        .layer_bufs
        .iter()
        .zip(config.kv_bufs.iter())
        .enumerate()
    {
        let layer_config = LayerPlanConfig {
            context: config.context,
            f16_program: config.f16_program,
            f16_l4_program: config.f16_l4_program,
            simple_ops_program: config.simple_ops_program,
            x_buf: config.x_buf,
            wq_buf: lb.wq,
            wk_buf: lb.wk,
            wv_buf: lb.wv,
            wo_buf: lb.wo,
            w_gate_buf: lb.w_gate,
            w_up_buf: lb.w_up,
            w_down_buf: lb.w_down,
            attn_norm_buf: lb.attn_norm,
            ffn_norm_buf: lb.ffn_norm,
            q_buf: config.q_buf,
            k_buf: config.k_buf,
            v_buf: config.v_buf,
            out_attn_buf: config.out_attn_buf,
            attn_out_buf: config.attn_out_buf,
            gate_buf: config.gate_buf,
            up_buf: config.up_buf,
            down_buf: config.down_buf,
            residual_buf: config.residual_buf,
            k_cache_buf: kb.k_cache,
            v_cache_buf: kb.v_cache,
            dim: config.dim,
            n_heads_q: config.n_heads_q,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            ffn_hidden: config.ffn_hidden,
            n_q,
            n_k,
            n_v,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            kv_capacity: config.kv_capacity,
            kv_pos_stride: config.kv_pos_stride,
            kv_head_stride: config.kv_head_stride,
            is_nosub: config.is_nosub,
        };
        layers.push(
            build_layer_plan(&layer_config)
                .with_context(|| format!("build plan for layer {}", i))?,
        );
    }

    // Only the last layer needs clFlush before final norm + lm_head
    if let Some(last) = layers.last_mut() {
        last.flush_after = true;
    }

    // Final RMSNorm (in-place on x)
    let final_norm = {
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_rms_norm_opt")
            .context("create final kernel_rms_norm_opt")?;
        let local_size = 64usize;
        let local_mem_bytes = local_size * std::mem::size_of::<f32>();
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.x_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.final_norm_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&(config.dim as i32)))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&config.rms_norm_eps))?;
            // add_unit = 0 (non-Gemma3)
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&0i32))?;
            ocl::core::set_kernel_arg(
                &kernel,
                5,
                ocl::core::ArgVal::local::<f32>(&local_mem_bytes),
            )?;
        }
        KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [local_size, 1, 1], // rows=1 for decode
            local_work_size: Some([local_size, 1, 1]),
            dynamic_args: vec![],
            op_tag: OpTag::FinalNorm,
            retained_bufs: vec![],
        }
    };

    // lm_head matmul: x [1, dim] × lm_head [vocab, dim]^T → logits [1, vocab]
    // None when lm_head is on CPU (large tied embedding that exceeds GPU alloc limit).
    let lm_head = if let Some(lm_head_buf) = config.lm_head_buf {
        Some(
            make_f16_matmul_step(
                config.f16_program,
                config.x_buf,
                lm_head_buf,
                config.logits_buf,
                config.vocab_size,
                config.dim,
                OpTag::LmHead,
                config.f16_l4_program,
                config.is_nosub,
            )
            .context("build lm_head matmul step")?,
        )
    } else {
        None
    };

    Ok(FullKernelPlan {
        layers,
        final_norm,
        lm_head,
        kv_capacity: config.kv_capacity,
    })
}

// ---------------------------------------------------------------------------
// KIVI plan builder
// ---------------------------------------------------------------------------

/// Per-layer KIVI buffer references for plan building.
pub struct KiviKvBufs<'a> {
    /// GPU F32 residual K buffer: [kv_heads, res_cap, head_dim]
    pub res_k: &'a Mem,
    /// GPU F32 residual V buffer: [kv_heads, res_cap, head_dim]
    pub res_v: &'a Mem,
    /// GPU Q2 key blocks (packed bytes)
    pub q2k: &'a Mem,
    /// GPU Q2 value blocks (packed bytes)
    pub q2v: &'a Mem,
    /// GPU F32 attention K buffer: [max_seq, kv_heads, head_dim]
    pub attn_k: &'a Mem,
    /// GPU F32 attention V buffer: [max_seq, kv_heads, head_dim]
    pub attn_v: &'a Mem,
    /// Residual buffer capacity in tokens
    pub res_cap: usize,
}

/// Config for building the KIVI variant full model plan.
pub struct KiviFullPlanConfig<'a> {
    pub context: &'a ocl::Context,
    pub f16_program: &'a ocl::Program,
    pub f16_l4_program: Option<&'a ocl::Program>,
    pub simple_ops_program: &'a ocl::Program,
    pub kivi_q2_program: &'a ocl::Program,
    pub kivi_attn_program: Option<&'a ocl::Program>,
    pub layer_bufs: Vec<LayerBufs<'a>>,
    // Workspace buffers (shared across layers)
    pub x_buf: &'a Mem,
    pub q_buf: &'a Mem,
    pub k_buf: &'a Mem,
    pub v_buf: &'a Mem,
    pub out_attn_buf: &'a Mem,
    pub attn_out_buf: &'a Mem,
    pub gate_buf: &'a Mem,
    pub up_buf: &'a Mem,
    pub down_buf: &'a Mem,
    pub residual_buf: &'a Mem,
    // Per-layer KIVI KV buffers
    pub kivi_kv_bufs: Vec<KiviKvBufs<'a>>,
    // Final norm + lm_head
    pub final_norm_buf: &'a Mem,
    /// `None` when lm_head is kept on CPU (large tied embedding).
    pub lm_head_buf: Option<&'a Mem>,
    pub logits_buf: &'a Mem,
    // Model config
    pub dim: usize,
    pub n_heads_q: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_hidden: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    /// Quantization bits (2, 4, 8)
    pub bits: u8,
    /// Whether to use native KIVI attention (true on nosub devices)
    pub use_native_attn: bool,
    /// Whether the device lacks subgroup support (nosub fallback path).
    pub is_nosub: bool,
}

/// Build a KIVI gather_update kernel step for K or V.
///
/// kivi_gather_update args:
///   input(0), residual(1), kv_heads(2), res_cap(3), head_dim(4), seq_len(5), res_pos(6)
fn make_kivi_gather_step(
    kivi_q2_program: &ocl::Program,
    input_buf: &Mem,
    residual_buf: &Mem,
    kv_heads: usize,
    res_cap: usize,
    head_dim: usize,
) -> Result<KernelStep> {
    let kernel = ocl::core::create_kernel(kivi_q2_program, "kivi_gather_update")
        .context("create kivi_gather_update")?;
    let seq_len_i32 = 1i32; // decode: always 1 token
    let res_pos_init = 0i32;
    unsafe {
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(input_buf))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(residual_buf))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&(kv_heads as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&(res_cap as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&(head_dim as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&seq_len_i32))?;
        ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::scalar(&res_pos_init))?;
    }
    let total = kv_heads * head_dim; // seq_len=1 for decode
    Ok(KernelStep {
        kernel,
        ndim: 3,
        global_work_size: [total, 1, 1],
        local_work_size: None,
        dynamic_args: vec![DynamicArg::ResPos { arg_idx: 6 }],
        op_tag: OpTag::KvScatter, // reuse tag for profiling
        retained_bufs: vec![],
    })
}

/// Build a KIVI scatter_residual kernel step for K or V.
///
/// kivi_scatter_residual args:
///   residual(0), attn(1), kv_heads(2), res_cap(3), head_dim(4), res_pos(5), tok_base(6)
fn make_kivi_scatter_step(
    kivi_q2_program: &ocl::Program,
    residual_buf: &Mem,
    attn_buf: &Mem,
    kv_heads: usize,
    res_cap: usize,
    head_dim: usize,
) -> Result<KernelStep> {
    let kernel = ocl::core::create_kernel(kivi_q2_program, "kivi_scatter_residual")
        .context("create kivi_scatter_residual")?;
    let res_pos_init = 0i32;
    let tok_base_init = 0i32;
    unsafe {
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(residual_buf))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(attn_buf))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&(kv_heads as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&(res_cap as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&(head_dim as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&res_pos_init))?;
        ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::scalar(&tok_base_init))?;
    }
    // Global work size is dynamic (depends on res_pos), but we set max possible.
    // At dispatch time, we recompute. For now, set to max = kv_heads * res_cap * head_dim.
    let max_total = kv_heads * res_cap * head_dim;
    Ok(KernelStep {
        kernel,
        ndim: 3,
        global_work_size: [max_total, 1, 1],
        local_work_size: None,
        dynamic_args: vec![
            DynamicArg::ResTokens { arg_idx: 5 },
            DynamicArg::TokBase { arg_idx: 6 },
        ],
        op_tag: OpTag::KvScatter,
        retained_bufs: vec![],
    })
}

/// Build native KIVI attention kernel step.
///
/// kernel_attn_gen_kivi_q2 args:
///   Q(0), q2_k(1), q2_v(2), res_k(3), res_v(4), O(5), S(6),
///   num_heads_q(7), num_heads_kv(8), head_dim(9),
///   q2_tokens(10), res_tokens(11), res_cap(12),
///   scale(13), score_stride(14), has_scores(15), local_mem(16)
#[allow(clippy::too_many_arguments)]
fn make_kivi_native_attn_step(
    kivi_attn_program: &ocl::Program,
    context: &ocl::Context,
    q_buf: &Mem,
    q2k_buf: &Mem,
    q2v_buf: &Mem,
    res_k_buf: &Mem,
    res_v_buf: &Mem,
    out_buf: &Mem,
    n_heads_q: usize,
    n_kv_heads: usize,
    head_dim: usize,
    res_cap: usize,
    bits: u8,
) -> Result<KernelStep> {
    let kernel_name = match bits {
        2 => "kernel_attn_gen_kivi_q2",
        4 => "kernel_attn_gen_kivi_q4",
        8 => "kernel_attn_gen_kivi_q8",
        _ => return Err(anyhow::anyhow!("Unsupported KIVI bits: {}", bits)),
    };
    let kernel =
        ocl::core::create_kernel(kivi_attn_program, kernel_name).context("create kivi attn")?;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let q2_tokens_init = 0i32;
    let res_tokens_init = 0i32;
    let has_scores = 0i32; // Plan path does not collect scores
    let score_stride = 0i32;

    let dummy_score_buf = unsafe {
        ocl::core::create_buffer::<_, f32>(context.as_core(), ocl::core::MEM_READ_WRITE, 1, None)
    }
    .context("create dummy score buffer for kivi plan")?;

    let local_size = 64usize;
    let local_mem_bytes = local_size * std::mem::size_of::<f32>();

    unsafe {
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(q_buf))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(q2k_buf))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(q2v_buf))?;
        ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::mem(res_k_buf))?;
        ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::mem(res_v_buf))?;
        ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::mem(out_buf))?;
        ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::mem(&dummy_score_buf))?;
        ocl::core::set_kernel_arg(&kernel, 7, ocl::core::ArgVal::scalar(&(n_heads_q as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 8, ocl::core::ArgVal::scalar(&(n_kv_heads as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 9, ocl::core::ArgVal::scalar(&(head_dim as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 10, ocl::core::ArgVal::scalar(&q2_tokens_init))?;
        ocl::core::set_kernel_arg(&kernel, 11, ocl::core::ArgVal::scalar(&res_tokens_init))?;
        ocl::core::set_kernel_arg(&kernel, 12, ocl::core::ArgVal::scalar(&(res_cap as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 13, ocl::core::ArgVal::scalar(&scale))?;
        ocl::core::set_kernel_arg(&kernel, 14, ocl::core::ArgVal::scalar(&score_stride))?;
        ocl::core::set_kernel_arg(&kernel, 15, ocl::core::ArgVal::scalar(&has_scores))?;
        ocl::core::set_kernel_arg(
            &kernel,
            16,
            ocl::core::ArgVal::local::<f32>(&local_mem_bytes),
        )?;
    }

    Ok(KernelStep {
        kernel,
        ndim: 1,
        global_work_size: [n_heads_q * local_size, 1, 1],
        local_work_size: Some([local_size, 1, 1]),
        dynamic_args: vec![
            DynamicArg::Q2Tokens { arg_idx: 10 },
            DynamicArg::ResTokens { arg_idx: 11 },
        ],
        op_tag: OpTag::Attention,
        retained_bufs: vec![dummy_score_buf],
    })
}

/// Build KIVI assembled attention: scatter residual to attn buffer, then standard F32 attention.
///
/// kernel_attn_gen_half args (with F32 attn buffers, SeqMajor layout for KIVI scatter output):
///   Q(0), K_attn(1), V_attn(2), O(3), S(4),
///   head_dim(5), n_heads_q(6), n_heads_kv(7), cache_seq_len(8), scale(9),
///   kv_pos_stride(10), kv_head_stride(11), write_scores(12), score_stride(13), local_mem(14)
#[allow(clippy::too_many_arguments)]
fn make_kivi_assembled_attn_step(
    simple_ops_program: &ocl::Program,
    context: &ocl::Context,
    q_buf: &Mem,
    attn_k_buf: &Mem,
    attn_v_buf: &Mem,
    out_buf: &Mem,
    n_heads_q: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<KernelStep> {
    let kernel = ocl::core::create_kernel(simple_ops_program, "kernel_attn_gen_half")
        .context("create kernel_attn_gen_half for KIVI assembled")?;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let cache_seq_len_init = 0i32;
    let write_scores = 0i32;
    let score_stride = 0i32;
    // KIVI scatter output is SeqMajor: [total_tokens, kv_heads, head_dim]
    let kv_pos_stride = (n_kv_heads * head_dim) as i32;
    let kv_head_stride = head_dim as i32;

    let dummy_score_buf = unsafe {
        ocl::core::create_buffer::<_, f32>(context.as_core(), ocl::core::MEM_READ_WRITE, 1, None)
    }
    .context("create dummy score buffer for kivi assembled plan")?;

    let local_size = 64usize;
    let local_mem_bytes = local_size * std::mem::size_of::<f32>();

    unsafe {
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(q_buf))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(attn_k_buf))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(attn_v_buf))?;
        ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::mem(out_buf))?;
        ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::mem(&dummy_score_buf))?;
        ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&(head_dim as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::scalar(&(n_heads_q as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 7, ocl::core::ArgVal::scalar(&(n_kv_heads as i32)))?;
        ocl::core::set_kernel_arg(&kernel, 8, ocl::core::ArgVal::scalar(&cache_seq_len_init))?;
        ocl::core::set_kernel_arg(&kernel, 9, ocl::core::ArgVal::scalar(&scale))?;
        ocl::core::set_kernel_arg(&kernel, 10, ocl::core::ArgVal::scalar(&kv_pos_stride))?;
        ocl::core::set_kernel_arg(&kernel, 11, ocl::core::ArgVal::scalar(&kv_head_stride))?;
        ocl::core::set_kernel_arg(&kernel, 12, ocl::core::ArgVal::scalar(&write_scores))?;
        ocl::core::set_kernel_arg(&kernel, 13, ocl::core::ArgVal::scalar(&score_stride))?;
        ocl::core::set_kernel_arg(
            &kernel,
            14,
            ocl::core::ArgVal::local::<f32>(&local_mem_bytes),
        )?;
    }

    Ok(KernelStep {
        kernel,
        ndim: 1,
        global_work_size: [n_heads_q * local_size, 1, 1],
        local_work_size: Some([local_size, 1, 1]),
        // CacheSeqLen is reused: dispatch_step sets it to total = q2_tokens + res_tokens
        dynamic_args: vec![DynamicArg::CacheSeqLen { arg_idx: 8 }],
        op_tag: OpTag::Attention,
        retained_bufs: vec![dummy_score_buf],
    })
}

/// Build a KIVI layer plan: Steps 1-6 and 9-15 are standard, Step 7 = KIVI gather, Step 8 = KIVI attention.
#[allow(clippy::too_many_arguments)]
fn build_kivi_layer_plan(
    config: &LayerPlanConfig,
    kivi_kv: &KiviKvBufs,
    kivi_q2_program: &ocl::Program,
    kivi_attn_program: Option<&ocl::Program>,
    bits: u8,
    use_native_attn: bool,
) -> Result<LayerKernelPlan> {
    // Build the standard plan first (gets us steps_pre_kv and steps_post_attn)
    let mut plan = build_layer_plan(config)?;

    // Replace Step 7: KV update -> KIVI gather
    let gather_k = make_kivi_gather_step(
        kivi_q2_program,
        config.k_buf,
        kivi_kv.res_k,
        config.n_kv_heads,
        kivi_kv.res_cap,
        config.head_dim,
    )?;
    let gather_v = make_kivi_gather_step(
        kivi_q2_program,
        config.v_buf,
        kivi_kv.res_v,
        config.n_kv_heads,
        kivi_kv.res_cap,
        config.head_dim,
    )?;
    plan.kv_update = KvUpdateVariant::Kivi { gather_k, gather_v };

    // Replace Step 8: Attention -> KIVI variant
    if use_native_attn {
        if let Some(attn_prog) = kivi_attn_program {
            plan.attention = AttentionVariant::KiviNative(make_kivi_native_attn_step(
                attn_prog,
                config.context,
                config.q_buf,
                kivi_kv.q2k,
                kivi_kv.q2v,
                kivi_kv.res_k,
                kivi_kv.res_v,
                config.out_attn_buf,
                config.n_heads_q,
                config.n_kv_heads,
                config.head_dim,
                kivi_kv.res_cap,
                bits,
            )?);
        } else {
            return Err(anyhow::anyhow!(
                "Native KIVI attention requested but kivi_attn program not available"
            ));
        }
    } else {
        // Assembled: scatter residual to attn buffer, then standard attention on the F32 buffer
        let scatter_k = make_kivi_scatter_step(
            kivi_q2_program,
            kivi_kv.res_k,
            kivi_kv.attn_k,
            config.n_kv_heads,
            kivi_kv.res_cap,
            config.head_dim,
        )?;
        let scatter_v = make_kivi_scatter_step(
            kivi_q2_program,
            kivi_kv.res_v,
            kivi_kv.attn_v,
            config.n_kv_heads,
            kivi_kv.res_cap,
            config.head_dim,
        )?;
        let attn = make_kivi_assembled_attn_step(
            config.simple_ops_program,
            config.context,
            config.q_buf,
            kivi_kv.attn_k,
            kivi_kv.attn_v,
            config.out_attn_buf,
            config.n_heads_q,
            config.n_kv_heads,
            config.head_dim,
        )?;
        plan.attention = AttentionVariant::KiviAssembled {
            scatter_k,
            scatter_v,
            attn,
        };
    }

    Ok(plan)
}

/// Build a pre-bound KIVI plan for the full model decode pass.
pub fn build_kivi_full_plan(config: &KiviFullPlanConfig) -> Result<FullKernelPlan> {
    let n_q = config.n_heads_q * config.head_dim;
    let n_k = config.n_kv_heads * config.head_dim;
    let n_v = n_k;

    let mut layers = Vec::with_capacity(config.layer_bufs.len());
    for (i, (lb, kkv)) in config
        .layer_bufs
        .iter()
        .zip(config.kivi_kv_bufs.iter())
        .enumerate()
    {
        // We need a LayerPlanConfig with dummy k_cache/v_cache for the standard steps.
        // For KIVI, the k_cache_buf/v_cache_buf are not used in the final plan
        // (replaced by KIVI gather/scatter), but build_layer_plan still creates a
        // standard scatter step that we immediately replace.
        // Use the attn_k buffer as a placeholder — it won't be dispatched.
        let layer_config = LayerPlanConfig {
            context: config.context,
            f16_program: config.f16_program,
            f16_l4_program: config.f16_l4_program,
            simple_ops_program: config.simple_ops_program,
            x_buf: config.x_buf,
            wq_buf: lb.wq,
            wk_buf: lb.wk,
            wv_buf: lb.wv,
            wo_buf: lb.wo,
            w_gate_buf: lb.w_gate,
            w_up_buf: lb.w_up,
            w_down_buf: lb.w_down,
            attn_norm_buf: lb.attn_norm,
            ffn_norm_buf: lb.ffn_norm,
            q_buf: config.q_buf,
            k_buf: config.k_buf,
            v_buf: config.v_buf,
            out_attn_buf: config.out_attn_buf,
            attn_out_buf: config.attn_out_buf,
            gate_buf: config.gate_buf,
            up_buf: config.up_buf,
            down_buf: config.down_buf,
            residual_buf: config.residual_buf,
            k_cache_buf: kkv.attn_k, // placeholder
            v_cache_buf: kkv.attn_v, // placeholder
            dim: config.dim,
            n_heads_q: config.n_heads_q,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            ffn_hidden: config.ffn_hidden,
            n_q,
            n_k,
            n_v,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            kv_capacity: config.max_seq_len,
            kv_pos_stride: (config.n_kv_heads * config.head_dim) as i32,
            kv_head_stride: config.head_dim as i32,
            is_nosub: config.is_nosub,
        };
        layers.push(
            build_kivi_layer_plan(
                &layer_config,
                kkv,
                config.kivi_q2_program,
                config.kivi_attn_program.as_ref().map(|p| *p),
                config.bits,
                config.use_native_attn,
            )
            .with_context(|| format!("build KIVI plan for layer {}", i))?,
        );
    }

    if let Some(last) = layers.last_mut() {
        last.flush_after = true;
    }

    // Final RMSNorm
    let final_norm = {
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_rms_norm_opt")
            .context("create final kernel_rms_norm_opt for KIVI plan")?;
        let local_size = 64usize;
        let local_mem_bytes = local_size * std::mem::size_of::<f32>();
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.x_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.final_norm_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&(config.dim as i32)))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&config.rms_norm_eps))?;
            // add_unit = 0 (non-Gemma3)
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&0i32))?;
            ocl::core::set_kernel_arg(
                &kernel,
                5,
                ocl::core::ArgVal::local::<f32>(&local_mem_bytes),
            )?;
        }
        KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [local_size, 1, 1],
            local_work_size: Some([local_size, 1, 1]),
            dynamic_args: vec![],
            op_tag: OpTag::FinalNorm,
            retained_bufs: vec![],
        }
    };

    // lm_head (None when kept on CPU)
    let lm_head = if let Some(lm_head_buf) = config.lm_head_buf {
        Some(
            make_f16_matmul_step(
                config.f16_program,
                config.x_buf,
                lm_head_buf,
                config.logits_buf,
                config.vocab_size,
                config.dim,
                OpTag::LmHead,
                config.f16_l4_program,
                config.is_nosub,
            )
            .context("build lm_head matmul step for KIVI plan")?,
        )
    } else {
        None
    };

    Ok(FullKernelPlan {
        layers,
        final_norm,
        lm_head,
        kv_capacity: config.max_seq_len,
    })
}
