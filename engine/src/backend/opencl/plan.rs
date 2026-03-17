//! Pre-bound kernel execution plan for GPU decode.
//!
//! Eliminates per-dispatch overhead by pre-binding all static kernel arguments
//! (weights, workspace buffers, dimensions) at plan build time. During execution,
//! only dynamic scalars (start_pos, cache_seq_len, write_pos) are updated.
//!
//! This mirrors llama.cpp's tight enqueue loop where kernel arguments rarely change
//! between tokens, achieving near-zero CPU overhead per dispatch.

use anyhow::Result;
use ocl::core::Kernel as CoreKernel;

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
