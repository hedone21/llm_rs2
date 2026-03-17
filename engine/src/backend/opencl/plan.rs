//! Pre-bound kernel execution plan for GPU decode.
//!
//! Eliminates per-dispatch overhead by pre-binding all static kernel arguments
//! (weights, workspace buffers, dimensions) at plan build time. During execution,
//! only dynamic scalars (start_pos, cache_seq_len, write_pos) are updated via a
//! compact inline slot table — no Vec, no enum match dispatch.
//!
//! Includes pre-bound embedding gather to avoid per-token dispatch overhead.

use anyhow::{Context, Result};
use ocl::core::Kernel as CoreKernel;
use ocl::core::Mem;

/// Operation tag for profiling — maps to OpProfiler fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpTag {
    EmbedGather,
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

/// Tag identifying which pre-computed dynamic value a slot references.
/// Repr(u8) enables direct indexing into a `[i32; 4]` values array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DynTag {
    /// RoPE start position
    StartPos = 0,
    /// Attention cache sequence length (= current_pos before advance)
    CacheSeqLen = 1,
    /// KV scatter write position (= current_pos before advance)
    WritePos = 2,
    /// KV cache capacity (constant per plan lifetime)
    KvCapacity = 3,
}

/// Compact dynamic argument slot — inline, no heap allocation.
#[derive(Debug, Clone, Copy)]
pub struct DynArgSlot {
    /// Kernel argument index to update.
    pub arg_idx: u32,
    /// Which pre-computed value to use (indexes into values array).
    pub tag: DynTag,
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
    /// Up to 2 dynamic arguments per step (inline, no Vec allocation).
    /// Most steps (10/15) have [None, None]. Scatter has 2.
    pub dyn_slots: [Option<DynArgSlot>; 2],
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
    /// Pre-bound embedding gather step
    pub embed_gather: KernelStep,
    /// Final RMSNorm step (model.norm)
    pub final_norm: KernelStep,
    /// lm_head matmul step
    pub lm_head: KernelStep,
    /// KV cache capacity at plan creation time (for invalidation check)
    pub kv_capacity: usize,
    /// Pre-allocated GPU buffer for 1 input token (4 bytes, owned by plan)
    pub input_token_buf: Mem,
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

/// Helper: set a single dynamic arg on a kernel step.
#[inline(always)]
unsafe fn set_dyn_arg(kernel: &CoreKernel, slot: &DynArgSlot, vals: &[i32; 4]) {
    unsafe {
        ocl::core::set_kernel_arg(
            kernel,
            slot.arg_idx,
            ocl::core::ArgVal::scalar(&vals[slot.tag as usize]),
        )
        .ok();
    }
}

/// Helper: enqueue a single kernel step.
#[inline(always)]
unsafe fn enqueue_step(queue: &ocl::core::CommandQueue, step: &KernelStep) {
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

impl FullKernelPlan {
    /// Execute the full decode plan for one token.
    ///
    /// Writes `token_id` to the pre-allocated input buffer, enqueues gather,
    /// then runs all layer kernels with minimal CPU overhead.
    ///
    /// All dynamic values are pre-computed once before the layer loop:
    /// - `start_pos`, `cache_seq_len`, `write_pos`, `kv_capacity` hoisted.
    /// - Capacity check runs once (all caches are homogeneous).
    ///
    /// Returns `Err(PlanInvalidated)` if KV cache capacity changed.
    pub fn execute<C: crate::core::kv_cache::KVCacheOps>(
        &self,
        queue: &ocl::core::CommandQueue,
        token_id: u32,
        start_pos: usize,
        kv_caches: &mut [C],
    ) -> std::result::Result<(), PlanInvalidated> {
        // Write token to pre-allocated input buffer + enqueue gather
        unsafe {
            let token_slice = std::slice::from_raw_parts(&token_id as *const u32, 1);
            ocl::core::enqueue_write_buffer(
                queue,
                &self.input_token_buf,
                false, // non-blocking (ordered with subsequent kernels)
                0,
                token_slice,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
            .ok();
            enqueue_step(queue, &self.embed_gather);
        }
        self.execute_layers(queue, start_pos, kv_caches)
    }

    /// Core execution: hoisted invariants + flat dynamic arg dispatch.
    fn execute_layers<C: crate::core::kv_cache::KVCacheOps>(
        &self,
        queue: &ocl::core::CommandQueue,
        start_pos: usize,
        kv_caches: &mut [C],
    ) -> std::result::Result<(), PlanInvalidated> {
        // ── Hoist all invariants ──
        // All KV caches are homogeneous (same capacity/pos), check once.
        let initial_pos = kv_caches[0].current_pos();
        if kv_caches[0].capacity() != self.kv_capacity || initial_pos >= self.kv_capacity {
            return Err(PlanInvalidated);
        }

        // Pre-compute all 4 dynamic values (indexed by DynTag repr).
        let dyn_vals: [i32; 4] = [
            start_pos as i32,        // DynTag::StartPos = 0
            initial_pos as i32,      // DynTag::CacheSeqLen = 1
            initial_pos as i32,      // DynTag::WritePos = 2
            self.kv_capacity as i32, // DynTag::KvCapacity = 3
        ];

        // ── Layer loop — tight enqueue with flat dynamic arg dispatch ──
        for (i, layer_plan) in self.layers.iter().enumerate() {
            for step in &layer_plan.steps {
                // Set dynamic args (0, 1, or 2 per step — unrolled, no Vec)
                if let Some(ref s) = step.dyn_slots[0] {
                    unsafe { set_dyn_arg(&step.kernel, s, &dyn_vals) };
                }
                if let Some(ref s) = step.dyn_slots[1] {
                    unsafe { set_dyn_arg(&step.kernel, s, &dyn_vals) };
                }

                unsafe { enqueue_step(queue, step) };
            }

            kv_caches[i].advance_pos(1);

            if layer_plan.flush_after {
                ocl::core::flush(queue).ok();
            }
        }

        // ── 3. Final norm + lm_head ──
        unsafe {
            enqueue_step(queue, &self.final_norm);
            enqueue_step(queue, &self.lm_head);
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

/// No dynamic args (10 of 15 steps per layer).
const NO_DYN: [Option<DynArgSlot>; 2] = [None, None];

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
        dyn_slots: NO_DYN,
        op_tag,
    })
}

/// Build a pre-bound kernel execution plan for one transformer layer's decode
/// step (seq_len=1).
///
/// Creates dedicated kernel objects via `ocl::core::create_kernel` and pre-binds
/// all static arguments. Dynamic arguments (start_pos, cache_seq_len, write_pos)
/// are tagged with [`DynArgSlot`] for flat dispatch during execution.
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
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&config.rms_norm_eps))?;
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
            dyn_slots: NO_DYN,
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
            ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&config.rope_theta))?;
        }
        let work_size = config.n_heads_q * (config.head_dim / 2);
        steps.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [work_size, 1, 1],
            local_work_size: None,
            dyn_slots: [
                Some(DynArgSlot {
                    arg_idx: 4,
                    tag: DynTag::StartPos,
                }),
                None,
            ],
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
            ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&config.rope_theta))?;
        }
        let work_size = config.n_kv_heads * (config.head_dim / 2);
        steps.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [work_size, 1, 1],
            local_work_size: None,
            dyn_slots: [
                Some(DynArgSlot {
                    arg_idx: 4,
                    tag: DynTag::StartPos,
                }),
                None,
            ],
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
            dyn_slots: [
                Some(DynArgSlot {
                    arg_idx: 5,
                    tag: DynTag::KvCapacity,
                }),
                Some(DynArgSlot {
                    arg_idx: 6,
                    tag: DynTag::WritePos,
                }),
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
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_attn_gen_half")
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
            ocl::core::set_kernel_arg(&kernel, 7, ocl::core::ArgVal::scalar(&cache_seq_len_init))?;
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
            dyn_slots: [
                Some(DynArgSlot {
                    arg_idx: 7,
                    tag: DynTag::CacheSeqLen,
                }),
                None,
            ],
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
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_add_rms_norm_oop")
            .context("create kernel_add_rms_norm_oop")?;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.x_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.attn_out_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(config.residual_buf))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::mem(config.ffn_norm_buf))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&(dim as i32)))?;
            ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&config.rms_norm_eps))?;
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
            dyn_slots: NO_DYN,
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
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_silu_mul_simple")
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
            dyn_slots: NO_DYN,
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
            dyn_slots: NO_DYN,
            op_tag: OpTag::AddAssign,
        });
    }

    Ok(LayerKernelPlan {
        steps,
        flush_after: true,
    })
}

// ---------------------------------------------------------------------------
// Full model plan builder
// ---------------------------------------------------------------------------

/// Config for building the full model plan (all layers + gather + final norm + lm_head).
pub struct FullPlanConfig<'a> {
    pub f16_program: &'a ocl::Program,
    pub simple_ops_program: &'a ocl::Program,
    pub get_rows_program: &'a ocl::Program,
    pub context: &'a ocl::core::Context,
    // Per-layer weight buffers
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
    // Embedding + final norm + lm_head
    pub embed_tokens_buf: &'a Mem,
    pub final_norm_buf: &'a Mem,
    pub lm_head_buf: &'a Mem,
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

/// Build a pre-bound plan for the full model decode pass (gather + all layers + head).
pub fn build_full_plan(config: &FullPlanConfig) -> Result<FullKernelPlan> {
    let n_q = config.n_heads_q * config.head_dim;
    let n_k = config.n_kv_heads * config.head_dim;
    let n_v = n_k;

    // ── Pre-allocate input token buffer (4 bytes for 1 u32) ──
    let input_token_buf = unsafe {
        ocl::core::create_buffer::<_, u32>(config.context, ocl::core::MEM_READ_ONLY, 1, None)
            .context("create input token buffer")?
    };

    // ── Build embed_gather step (kernel_get_rows_f32) ──
    let embed_gather = {
        let kernel = ocl::core::create_kernel(config.get_rows_program, "kernel_get_rows_f32")
            .context("create kernel_get_rows_f32 for plan")?;
        let k = config.dim;
        let ne00 = k as i32;
        let nb01 = (k * 4) as u64; // F32 row stride
        let nb02 = nb01 * config.vocab_size as u64;
        let nb03 = nb02;
        let ne10 = 1i32; // 1 index for decode
        let nb10 = 4u64;
        let nb11 = 4u64;
        let nb12 = 4u64;
        let nb1 = (k * 4) as u64;
        let nb2 = nb1;
        let nb3 = nb2;

        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.embed_tokens_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(&input_token_buf))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::mem(config.x_buf))?;
            ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&0u64))?;
            ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::scalar(&ne00))?;
            ocl::core::set_kernel_arg(&kernel, 7, ocl::core::ArgVal::scalar(&nb01))?;
            ocl::core::set_kernel_arg(&kernel, 8, ocl::core::ArgVal::scalar(&nb02))?;
            ocl::core::set_kernel_arg(&kernel, 9, ocl::core::ArgVal::scalar(&nb03))?;
            ocl::core::set_kernel_arg(&kernel, 10, ocl::core::ArgVal::scalar(&ne10))?;
            ocl::core::set_kernel_arg(&kernel, 11, ocl::core::ArgVal::scalar(&nb10))?;
            ocl::core::set_kernel_arg(&kernel, 12, ocl::core::ArgVal::scalar(&nb11))?;
            ocl::core::set_kernel_arg(&kernel, 13, ocl::core::ArgVal::scalar(&nb12))?;
            ocl::core::set_kernel_arg(&kernel, 14, ocl::core::ArgVal::scalar(&nb1))?;
            ocl::core::set_kernel_arg(&kernel, 15, ocl::core::ArgVal::scalar(&nb2))?;
            ocl::core::set_kernel_arg(&kernel, 16, ocl::core::ArgVal::scalar(&nb3))?;
        }

        KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [64, 1, 1], // 1 index × 64 threads
            local_work_size: Some([64, 1, 1]),
            dyn_slots: NO_DYN,
            op_tag: OpTag::EmbedGather,
        }
    };

    // ── Build per-layer plans ──
    let mut layers = Vec::with_capacity(config.layer_bufs.len());
    for (i, (lb, kb)) in config
        .layer_bufs
        .iter()
        .zip(config.kv_bufs.iter())
        .enumerate()
    {
        let layer_config = LayerPlanConfig {
            f16_program: config.f16_program,
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
        };
        layers.push(
            build_layer_plan(&layer_config)
                .with_context(|| format!("build plan for layer {}", i))?,
        );
    }

    // ── Final RMSNorm (in-place on x) ──
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
            ocl::core::set_kernel_arg(
                &kernel,
                4,
                ocl::core::ArgVal::local::<f32>(&local_mem_bytes),
            )?;
        }
        KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [local_size, 1, 1], // rows=1 for decode
            local_work_size: Some([local_size, 1, 1]),
            dyn_slots: NO_DYN,
            op_tag: OpTag::FinalNorm,
        }
    };

    // ── lm_head matmul: x [1, dim] × lm_head [vocab, dim]^T → logits [1, vocab] ──
    let lm_head = make_f16_matmul_step(
        config.f16_program,
        config.x_buf,
        config.lm_head_buf,
        config.logits_buf,
        config.vocab_size,
        config.dim,
        OpTag::LmHead,
    )
    .context("build lm_head matmul step")?;

    Ok(FullKernelPlan {
        layers,
        embed_gather,
        final_norm,
        lm_head,
        kv_capacity: config.kv_capacity,
        input_token_buf,
    })
}
