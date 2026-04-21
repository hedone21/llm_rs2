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
use std::sync::Arc;

use crate::core::backend::Backend;
use crate::core::tensor::Tensor;
use crate::layers::tensor_partition::{
    PartitionContext, PartitionPath, partition_fused_merge_enabled, partition_plan_debug_enabled,
    partition_plan_enabled, partition_replicate_norm_enabled, partition_trace_enabled,
    record_partition_timing,
};
use crate::layers::workspace::PartitionWsCell;

thread_local! {
    /// LLMRS_OP_TRACE: per-token wall-clock accumulator (label -> microseconds).
    /// Activated by Plan::execute; dispatch_step reads/writes when Some(_).
    /// Labels are split into `{op}@enqueue` (CPU submission) and `{op}@gpu`
    /// (clFinish wait, approximates pure GPU execution) to isolate where slope
    /// against n_kv originates.
    static OP_TRACE_ACC: std::cell::RefCell<Option<std::collections::HashMap<String, u64>>> =
        const { std::cell::RefCell::new(None) };
}

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

impl OpTag {
    /// Static label used by the event-based profiler (`--profile-events`).
    ///
    /// These names match the keys produced by the non-plan path (forward_gen
    /// via `OpenCLBackend::set_op_label`), so the resulting aggregate is
    /// self-consistent whether plan execution or the generic dispatch path
    /// was used. Must stay in sync with the label matrix in the
    /// `.agent/research/2026-04-14_decode_microbench_plan.md` report.
    pub fn profile_label(&self) -> &'static str {
        match self {
            OpTag::RmsNorm => "rms_norm",
            OpTag::AddRmsNorm => "rms_norm",
            OpTag::FinalNorm => "rms_norm",
            OpTag::MatmulQKV => "matmul_qkv",
            OpTag::MatmulWo => "matmul_wo",
            OpTag::MatmulGateUp => "matmul_ffn",
            OpTag::MatmulDown => "matmul_ffn",
            OpTag::Rope => "rope",
            OpTag::KvScatter => "kv_update",
            OpTag::Attention => "attention",
            OpTag::AddAssign => "add_assign",
            OpTag::SiluMul => "silu_mul",
            OpTag::LmHead => "lm_head",
        }
    }
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
    /// Legacy kernel_attn_gen_half — supports score writes, any head_dim.
    Standard(KernelStep),
    /// Decode flash attention (flash_attn_f32_f16_q1). Single-pass online
    /// softmax, no score output. Selected at plan-build time when
    /// head_dim==64, F16 KV, HeadMajor, and no scores are needed.
    StandardFlash(KernelStep),
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
    /// Steps 9-10: Wo matmul, add+RMSNorm. FFN input (`residual`) is produced
    /// by the last step here.
    pub steps_post_attn_pre_ffn: Vec<KernelStep>,
    /// Step 11-14: FFN — GPU-only (gate/up/silu/down) or cooperative partition
    /// (GPU-slice chain + CPU slice + merge).
    pub ffn: FfnVariant,
    /// Step 15+: typically `add_assign(x += down)`. Skipped by the caller when
    /// a partition layer defers the residual add into the next layer's fused
    /// norm+merge.
    pub steps_post_ffn: Vec<KernelStep>,
    /// Whether to call clFlush after this layer's steps
    pub flush_after: bool,
}

/// FFN execution strategy for a layer.
///
/// `GpuOnly` is the historical plan path (4 KernelSteps dispatched inline).
/// `Partitioned` wraps the cooperative GPU+CPU FFN slice (see
/// `arch/plan_partition_integration.md`, §A.3.1). `Box`-ing the variant keeps
/// the enum compact even though `PartitionStep` carries CPU-side buffers.
#[allow(clippy::large_enum_variant)]
pub enum FfnVariant {
    /// Dense GPU FFN: gate + up + silu/gelu + down. `gate` and `up` are
    /// separate matmuls because the builder already runs them sequentially
    /// (no fused gate_up kernel on Adreno today).
    GpuOnly {
        gate: KernelStep,
        up: KernelStep,
        silu_mul: KernelStep,
        down: KernelStep,
    },
    /// Cooperative partition path: GPU runs its split-row FFN chain, CPU runs
    /// the complementary slice, and the merge lands the sum back in
    /// `ws.down`. Boxed so a GpuOnly layer stays small.
    Partitioned(Box<PartitionStep>),
}

// ---------------------------------------------------------------------------
// Partition step — cooperative GPU+CPU FFN dispatch inside a plan.
// ---------------------------------------------------------------------------

/// Single PartitionStep: wraps the 4 GPU FFN KernelSteps (gate, up, silu/gelu
/// * up, down) plus the CPU-side FFN slice invocation plus the merge substeps.
///
/// Safety model matches `KernelStep` (single-threaded dispatch — see the
/// `unsafe impl Send/Sync` blocks below for the invariant).
pub struct PartitionStep {
    /// GPU slice gate matmul: `residual @ part.gate.gpu_slice^T -> gate_gpu`.
    pub gpu_gate: KernelStep,
    /// GPU slice up matmul: `residual @ part.up.gpu_slice^T -> up_gpu`.
    pub gpu_up: KernelStep,
    /// GPU SiLU/GELU × up in place on `gate_gpu`.
    pub gpu_act_mul: KernelStep,
    /// GPU slice down matmul: `gate_gpu @ part.down.gpu_slice^T -> down_partial_gpu`.
    pub gpu_down: KernelStep,
    /// CPU-side context (backends + workspace + geometry).
    pub cpu_ctx: Arc<PartitionPlanContext>,
    /// Merge variant decided at build time (env-gated).
    pub merge: PartitionMerge,
    /// Whether this is the last transformer layer (fused_norm_merge can't
    /// defer past the final layer because there's no next norm to fold into).
    pub is_last_layer: bool,
}

// SAFETY: Same model as `KernelStep`. `PartitionStep` owns `CoreKernel` / `Mem`
// handles and an `Arc<PartitionWsCell>`; all are accessed
// only from the plan's single-threaded dispatch loop (`FullKernelPlan::execute`),
// and `PartitionStep::run` asserts the TLS thread id matches the build thread.
unsafe impl Send for PartitionStep {}
unsafe impl Sync for PartitionStep {}

/// Merge strategy — how the GPU partial and the CPU partial combine into
/// `ws.down`.
#[allow(clippy::large_enum_variant)]
pub enum PartitionMerge {
    /// Immediate merge in the current layer. Requires 3 sub-steps:
    /// 1. copy `down_partial_gpu[0..hidden] -> ws.down[0..hidden]`
    /// 2. write CPU partial → `cpu_merge_staging` GPU buffer
    ///    (backend `write_buffer`, not a KernelStep — handled directly in `run`)
    /// 3. `add_assign(ws.down, cpu_merge_staging)`
    Inline {
        copy_gpu_to_down: KernelStep,
        add_assign: KernelStep,
    },
    /// Defer the final merge (copy + add_assign) to the next layer's
    /// `fused_norm_merge` kernel. Only the CPU upload runs now, landing the
    /// CPU partial in `cpu_merge_staging`. The next layer picks it up via the
    /// `partition_prev_*` carry slots (see `LayerWorkspace`).
    ///
    /// NOTE: plan path currently builds `Inline` always; `Deferred` is kept
    /// for future fusion work. The builder picks based on
    /// `partition_fused_merge_enabled()` + `!is_last_layer`, but since the
    /// plan path has no fused_norm_merge kernel wiring yet, the merge defers
    /// as a no-op on the next layer (equivalent to Inline on the current
    /// layer plus an extra copy, not bit-exact with forward_gen Deferred).
    /// Treat this variant as feature-flagged until the next layer side
    /// lands in Phase 2-B.
    Deferred,
}

/// CPU/workspace context used by a `PartitionStep`.
///
/// All Arc-held resources must outlive the plan (INV-082).
pub struct PartitionPlanContext {
    /// CPU-capable backend (NEON on Android/Apple, scalar fallback on host
    /// CPUs without NEON). Used for gate/up/silu/down CPU-slice matmul.
    pub cpu_backend: Arc<dyn Backend>,
    /// CPU slice of gate weight. Shape `[ffn_hidden - split_row, dim]`.
    pub gate_cpu: Tensor,
    /// CPU slice of up weight. Shape `[ffn_hidden - split_row, dim]`.
    pub up_cpu: Tensor,
    /// CPU slice of down weight. Shape `[dim, ffn_hidden - split_row]`.
    pub down_cpu: Tensor,
    /// Shared partition workspace — gate_cpu/up_cpu/down_partial_cpu (NEON
    /// working tensors), residual_cpu (CPU mirror of ws.residual),
    /// cpu_merge_staging (GPU upload target). `UnsafeCell` since the plan
    /// only ever mutates it from the single dispatch thread.
    pub workspace: Arc<PartitionWsCell>,
    /// Kernel args in the 4 GPU sub-steps are pre-bound to these cl_mem
    /// handles; storing them here lets `run()` sanity-check stale buffers
    /// (plan invalidation on UMA remapping). Currently informational only.
    #[allow(dead_code)]
    pub residual_buf_handle: Mem,
    #[allow(dead_code)]
    pub x_buf_handle: Mem,
    #[allow(dead_code)]
    pub attn_out_buf_handle: Mem,
    /// Whether the model uses `gelu(tanh)` instead of SiLU (arch-dependent).
    /// The GPU kernel binding picks `kernel_silu_mul_simple` vs
    /// `kernel_gelu_tanh_mul_simple`; this flag mirrors the decision so CPU
    /// FFN on the same layer stays in sync.
    pub use_gelu_tanh: bool,
    /// Mirror of `LayerPlanConfig::rms_norm_eps` — unused today (the pre-FFN
    /// norm is a separate KernelStep), kept for Phase 2-B fused_norm_merge.
    #[allow(dead_code)]
    pub rms_norm_eps: f32,
    /// Whether the pre-FFN norm uses Gemma3's `add_unit=1` variant.
    /// Currently the plan path rejects Gemma3 outright, but this is kept
    /// so a future fused_norm_merge can route both paths.
    #[allow(dead_code)]
    pub rms_norm_add_unit: bool,
    /// Debugging: which layer this step belongs to.
    pub layer_idx: usize,
    /// Which residual transport path this plan was built against. Plan path
    /// only supports `SyncRead` today (no async DMA, no replicate_norm).
    pub partition_path: PartitionPath,
    /// INV-120 generation captured at build time. `PartitionStep::run` loads
    /// `PartitionContext.ratio_generation` with `Acquire` and compares; a
    /// miss returns `PlanInvalidated` to the executor.
    pub ratio_generation_at_build: u64,
    /// Shared generation counter. Held as an Arc clone of
    /// `PartitionContext.ratio_generation` — cheap clone, keeps the atomic
    /// alive even if the caller drops the PartitionContext while this plan
    /// is still cached (shouldn't happen in generate.rs today but defensive).
    pub ratio_generation_counter: Arc<std::sync::atomic::AtomicU64>,
    /// TLS thread id captured at plan build. `run()` asserts the dispatch
    /// thread matches — enforces the single-threaded safety model.
    pub build_thread_id: std::thread::ThreadId,
}

// SAFETY: Mirrors `KernelStep` — all interior references are accessed only
// from the dispatch thread (debug-asserted in `PartitionStep::run`). The
// `UnsafeCell<PartitionPlanWorkspace>` is never aliased outside the plan.
unsafe impl Send for PartitionPlanContext {}
unsafe impl Sync for PartitionPlanContext {}

/// Plan-side partition workspace — a direct reuse of the layer's
/// `PartitionWorkspace` buffers. We re-declare as a type alias to the real
/// `PartitionWorkspace` so callers can share allocations with `forward_gen`
/// (both paths read/write the same `gate_cpu`, `down_partial_gpu`, etc).
pub type PartitionPlanWorkspace = crate::layers::workspace::PartitionWorkspace;

impl PartitionStep {
    /// Execute this partition layer step.
    ///
    /// Sequence (mirrors `forward_gen` partition SyncRead path, see
    /// `arch/plan_partition_integration.md` §A.4.3):
    ///   1. INV-120 generation check (stale ratio → `PlanInvalidated`).
    ///   2. `backend.synchronize()` — ARM UMA cache barrier; ensures the
    ///      preceding `add_rms_norm_oop` result in `ws.residual` is visible
    ///      to the CPU read.
    ///   3. `backend.read_buffer(ws.residual → pw.residual_cpu)` — load the
    ///      FFN input into CPU-accessible memory for NEON.
    ///   4. Enqueue the 4 GPU FFN sub-steps + `flush()`.
    ///   5. CPU FFN chain (gate → up → silu/gelu_mul → down) against the
    ///      partition slice, using `cpu_ctx.cpu_backend`.
    ///   6. Merge — `Inline` runs copy_gpu_to_down + write_buffer(CPU→GPU
    ///      staging) + add_assign. `Deferred` uploads the CPU partial and
    ///      returns (next layer will absorb).
    ///
    /// Returns `Err(PlanInvalidated)` if the captured ratio generation no
    /// longer matches the live `PartitionContext`.
    pub fn run(
        &self,
        backend: &crate::backend::opencl::OpenCLBackend,
        layer_idx: usize,
    ) -> std::result::Result<(), PlanInvalidated> {
        use std::sync::atomic::Ordering;
        debug_assert_eq!(
            std::thread::current().id(),
            self.cpu_ctx.build_thread_id,
            "PartitionStep::run must run on the plan build thread"
        );

        // 1. INV-120 generation check.
        let live_gen = self
            .cpu_ctx
            .ratio_generation_counter
            .load(Ordering::Acquire);
        if live_gen != self.cpu_ctx.ratio_generation_at_build {
            log::warn!(
                "PartitionStep stale: layer={} build_gen={} live_gen={} → PlanInvalidated",
                layer_idx,
                self.cpu_ctx.ratio_generation_at_build,
                live_gen,
            );
            return Err(PlanInvalidated);
        }

        let debug = partition_plan_debug_enabled();
        let trace = partition_trace_enabled();
        let t0 = if trace {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 2. Sync + read residual into CPU workspace.
        //    The plan path only supports SyncRead today — zcopy/async/replicate
        //    are rejected at build time. `ocl::core::finish` drains the
        //    in-order queue (add_rms_norm_oop + prior ops) and issues the
        //    ARM UMA cache flush needed before the host read.
        let queue = backend.queue.as_core();
        if let Err(e) = ocl::core::finish(queue) {
            log::error!(
                "PartitionStep synchronize failed: layer={} err={}",
                layer_idx,
                e
            );
            return Err(PlanInvalidated);
        }
        let t_sync_done = if trace {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // SAFETY: `workspace` is owned by this plan and only accessed here.
        let pw: &mut PartitionPlanWorkspace = unsafe { &mut *self.cpu_ctx.workspace.get() };

        // Read residual into CPU-visible buffer. The pre-bound cl_mem is the
        // canonical source; a blocking `enqueue_read_buffer` guarantees the
        // bytes land in `pw.residual_cpu` before the CPU FFN chain starts.
        let residual_bytes = pw.residual_cpu.size();
        let residual_slice =
            unsafe { std::slice::from_raw_parts_mut(pw.residual_cpu.as_mut_ptr(), residual_bytes) };
        if let Err(e) = unsafe {
            ocl::core::enqueue_read_buffer(
                queue,
                &self.cpu_ctx.residual_buf_handle,
                true, // blocking
                0,
                residual_slice,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )
        } {
            log::error!(
                "PartitionStep read residual failed: layer={} err={}",
                layer_idx,
                e
            );
            return Err(PlanInvalidated);
        }
        let t_read_done = if trace {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 3. Enqueue the 4 GPU FFN sub-steps. `dispatch_step` dynamic args
        //    are all zero for the FFN chain (no StartPos/CacheSeqLen/etc).
        for step in [
            &self.gpu_gate,
            &self.gpu_up,
            &self.gpu_act_mul,
            &self.gpu_down,
        ] {
            FullKernelPlan::dispatch_step(backend, step, 0, 0, 0, 0, 0, 0, 0);
            if debug {
                ocl::core::finish(queue).ok();
                eprintln!(
                    "plan-partition: layer={} sub_step={:?}",
                    layer_idx, step.op_tag,
                );
            }
        }
        if let Err(e) = ocl::core::flush(queue) {
            log::error!("PartitionStep flush failed: layer={} err={}", layer_idx, e);
            return Err(PlanInvalidated);
        }

        // 4. CPU FFN chain. Uses the `PartitionPlanContext`'s saved Tensor
        //    slices — these are Arc-cloned from the PartitionContext at
        //    plan build time. Live PartitionContext re-slices bump
        //    `ratio_generation`, so the check above already rejected us if
        //    the weights have moved.
        let cpu = &self.cpu_ctx.cpu_backend;

        // Run CPU gate/up through the backend trait — for the plan path we
        // intentionally skip the host `fused_matmul_*` fast path; the GPU
        // plan chain dominates wall-clock and the saved 100-300us per layer
        // on fused_matmul would come at the cost of duplicating the
        // `residual_cpu_ptr` routing logic here. Treat that as a future
        // optimization gated on benchmark evidence.
        let cpu_ok = (|| -> Result<()> {
            cpu.matmul_transposed(&pw.residual_cpu, &self.cpu_ctx.gate_cpu, &mut pw.gate_cpu)?;
            cpu.matmul_transposed(&pw.residual_cpu, &self.cpu_ctx.up_cpu, &mut pw.up_cpu)?;
            if self.cpu_ctx.use_gelu_tanh {
                cpu.gelu_tanh_mul(&mut pw.gate_cpu, &pw.up_cpu)?;
            } else {
                cpu.silu_mul(&mut pw.gate_cpu, &pw.up_cpu)?;
            }
            cpu.matmul_transposed(
                &pw.gate_cpu,
                &self.cpu_ctx.down_cpu,
                &mut pw.down_partial_cpu,
            )?;
            Ok(())
        })();
        if let Err(e) = cpu_ok {
            log::error!(
                "PartitionStep CPU FFN failed: layer={} err={}",
                layer_idx,
                e
            );
            return Err(PlanInvalidated);
        }
        let t_cpu_done = if trace {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 5. GPU wait — drain before the CPU `write_buffer` enqueues to
        //    avoid interleaving with outstanding GPU reads from the same
        //    staging buffer.
        if let Err(e) = ocl::core::finish(queue) {
            log::error!(
                "PartitionStep GPU wait failed: layer={} err={}",
                layer_idx,
                e
            );
            return Err(PlanInvalidated);
        }
        let t_gpu_done = if trace {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // 6. Merge.
        let staging_mem: &Mem =
            match crate::backend::opencl::get_cl_mem(pw.cpu_merge_staging.buffer().as_ref()) {
                Ok(m) => m,
                Err(e) => {
                    log::error!(
                        "PartitionStep staging cl_mem lookup failed: layer={} err={}",
                        layer_idx,
                        e
                    );
                    return Err(PlanInvalidated);
                }
            };
        let staging_bytes = pw.cpu_merge_staging.size();
        let cpu_partial_slice = unsafe {
            std::slice::from_raw_parts(pw.down_partial_cpu.as_ptr(), pw.down_partial_cpu.size())
        };
        debug_assert!(
            staging_bytes >= cpu_partial_slice.len(),
            "cpu_merge_staging must be >= down_partial_cpu size",
        );

        match &self.merge {
            PartitionMerge::Inline {
                copy_gpu_to_down,
                add_assign,
            } => {
                // (a) copy_slice(down_partial_gpu → ws.down). Pre-bound.
                FullKernelPlan::dispatch_step(backend, copy_gpu_to_down, 0, 0, 0, 0, 0, 0, 0);
                // (b) upload CPU partial into `cpu_merge_staging` (GPU buffer).
                //     write_buffer is blocking on in-order queue; after this
                //     the staging buffer is bound to the add_assign step.
                if let Err(e) = unsafe {
                    ocl::core::enqueue_write_buffer(
                        queue,
                        staging_mem,
                        true, // blocking
                        0,
                        cpu_partial_slice,
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )
                } {
                    log::error!(
                        "PartitionStep CPU partial upload failed: layer={} err={}",
                        layer_idx,
                        e
                    );
                    return Err(PlanInvalidated);
                }
                // (c) add_assign(ws.down, cpu_merge_staging).
                FullKernelPlan::dispatch_step(backend, add_assign, 0, 0, 0, 0, 0, 0, 0);
            }
            PartitionMerge::Deferred => {
                // Upload CPU partial only; the next layer's fused kernel
                // will perform copy+add. (Phase 2-B placeholder.)
                if let Err(e) = unsafe {
                    ocl::core::enqueue_write_buffer(
                        queue,
                        staging_mem,
                        true,
                        0,
                        cpu_partial_slice,
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )
                } {
                    log::error!(
                        "PartitionStep CPU partial upload failed (deferred): layer={} err={}",
                        layer_idx,
                        e
                    );
                    return Err(PlanInvalidated);
                }
            }
        }

        // 7. Trace accounting. Mirrors forward_gen segment breakdown:
        //    sync_drain / dma_read / cpu_matmul / gpu_wait / merge.
        if let (Some(t0), Some(t_sync), Some(t_read), Some(t_cpu), Some(t_gpu)) =
            (t0, t_sync_done, t_read_done, t_cpu_done, t_gpu_done)
        {
            let t_merge = std::time::Instant::now();
            let sync_ns = t_sync.duration_since(t0).as_nanos() as u64;
            let dma_ns = t_read.duration_since(t_sync).as_nanos() as u64;
            let cpu_ns = t_cpu.duration_since(t_read).as_nanos() as u64;
            let gpu_wait_ns = t_gpu.duration_since(t_cpu).as_nanos() as u64;
            let merge_ns = t_merge.duration_since(t_gpu).as_nanos() as u64;
            record_partition_timing(
                sync_ns,
                dma_ns,
                cpu_ns,
                gpu_wait_ns,
                merge_ns,
                self.cpu_ctx.partition_path,
            );
        }

        Ok(())
    }
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
    /// True when the legacy attention step was bound to the backend's GPU
    /// score accumulator at build time. `execute()` uses this flag to drive
    /// `GpuScoreAccumulator::reduce_layer` per layer and `end_step` after
    /// the final layer, mirroring the non-plan path in `transformer.rs`.
    pub writes_gpu_scores: bool,
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
    ///
    /// When `backend.profile_events_enabled` is true, the dispatch goes
    /// through `backend.enqueue_kernel_labeled()` so a profiling event is
    /// captured for each kernel. Otherwise, falls back to the legacy raw
    /// `ocl::core::enqueue_kernel` path (zero overhead in the non-profiled
    /// build).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_step(
        backend: &crate::backend::opencl::OpenCLBackend,
        step: &KernelStep,
        start_pos: i32,
        cache_seq_len: i32,
        write_pos: i32,
        kv_capacity: i32,
        res_pos: i32,
        q2_tokens: i32,
        res_tokens: i32,
    ) {
        let queue = backend.queue.as_core();
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
        let traced = OP_TRACE_ACC.with(|c| c.borrow().is_some());
        if traced {
            ocl::core::finish(queue).ok();
        }
        let t_op_start = std::time::Instant::now();
        if backend.profile_events_enabled {
            if let Err(e) = backend.enqueue_kernel_labeled(
                &step.kernel,
                step.op_tag.profile_label(),
                step.ndim,
                &step.global_work_size,
                step.local_work_size,
            ) {
                log::error!(
                    "Plan enqueue_kernel_labeled failed: op={:?} gws={:?}: {}",
                    step.op_tag,
                    step.global_work_size,
                    e
                );
            }
        } else {
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
        if traced {
            let t_enqueue_end = std::time::Instant::now();
            ocl::core::finish(queue).ok();
            let t_gpu_end = std::time::Instant::now();
            let enqueue_us = t_enqueue_end.duration_since(t_op_start).as_nanos() as u64 / 1000;
            let gpu_us = t_gpu_end.duration_since(t_enqueue_end).as_nanos() as u64 / 1000;
            let label = step.op_tag.profile_label();
            OP_TRACE_ACC.with(|c| {
                if let Some(m) = c.borrow_mut().as_mut() {
                    *m.entry(format!("{}@enqueue", label)).or_insert(0) += enqueue_us;
                    *m.entry(format!("{}@gpu", label)).or_insert(0) += gpu_us;
                }
            });
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
        backend: &crate::backend::opencl::OpenCLBackend,
        start_pos: usize,
        kv_caches: &mut [C],
    ) -> std::result::Result<(), PlanInvalidated> {
        let debug_sync = std::env::var("PLAN_DEBUG").is_ok();
        let op_trace = std::env::var_os("LLMRS_OP_TRACE").is_some();
        let queue = backend.queue.as_core();

        if op_trace {
            OP_TRACE_ACC.with(|c| {
                *c.borrow_mut() = Some(std::collections::HashMap::new());
            });
        }
        let mut trace_n_kv: i32 = 0;

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
            if op_trace {
                trace_n_kv = attn_seq_len;
            }

            // Steps 1-6: pre-KV steps
            for (si, step) in layer_plan.steps_pre_kv.iter().enumerate() {
                Self::dispatch_step(
                    backend,
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
                        backend,
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
                        backend,
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
                        backend,
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
                        backend,
                        step,
                        start_pos_i32,
                        attn_seq_len,
                        write_pos,
                        kv_cap,
                        rp,
                        q2t,
                        rt,
                    );
                    // GPU score accumulator: per-layer scores now live in the
                    // layer's own slice of `score_buf` (offset pre-baked at
                    // plan-build time, see `LayerPlanConfig::gpu_score_layer_offset`).
                    // A single fused reduce kernel folds all layers into
                    // cumulative importance at `end_step()` after the final
                    // layer — no per-layer dispatch needed here.
                    if debug_sync {
                        eprintln!("[Plan] L{} attention enqueued, calling finish...", i);
                        ocl::core::finish(queue).ok();
                        eprintln!("[Plan] L{} attention OK (attn_seq_len={})", i, attn_seq_len);
                    }
                }
                AttentionVariant::StandardFlash(step) => {
                    if debug_sync {
                        eprintln!(
                            "[Plan] L{} flash attention dispatch (attn_seq_len={}, gws={:?}, lws={:?})",
                            i, attn_seq_len, step.global_work_size, step.local_work_size
                        );
                    }
                    let trace_q1 = std::env::var_os("LLMRS_TRACE_Q1").is_some();
                    if trace_q1 {
                        ocl::core::finish(queue).ok();
                    }
                    let q1_start = std::time::Instant::now();
                    Self::dispatch_step(
                        backend,
                        step,
                        start_pos_i32,
                        attn_seq_len,
                        write_pos,
                        kv_cap,
                        rp,
                        q2t,
                        rt,
                    );
                    if trace_q1 {
                        ocl::core::finish(queue).ok();
                        let us = q1_start.elapsed().as_nanos() as u64 / 1000;
                        eprintln!("[Q1_TRACE] layer={} n_kv={} us={}", i, attn_seq_len, us);
                    }
                    // LLMRS_Q1_REPEAT=N: re-dispatch the Q1 kernel (N-1) additional
                    // times against the same KV state, measuring each repetition in
                    // isolation. The first (production) iteration follows matmul_qkv
                    // /rope/kv_update and reads KV "cold" from the just-written slot;
                    // subsequent reps read it "warm". Comparing rep=0 vs rep>=1
                    // slope against n_kv separates kernel-intrinsic cost from
                    // context-dependent cache/coherency effects.
                    let q1_repeat: u32 = std::env::var("LLMRS_Q1_REPEAT")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    for rep in 1..q1_repeat {
                        ocl::core::finish(queue).ok();
                        let rep_start = std::time::Instant::now();
                        Self::dispatch_step(
                            backend,
                            step,
                            start_pos_i32,
                            attn_seq_len,
                            write_pos,
                            kv_cap,
                            rp,
                            q2t,
                            rt,
                        );
                        ocl::core::finish(queue).ok();
                        let rep_us = rep_start.elapsed().as_nanos() as u64 / 1000;
                        eprintln!(
                            "[Q1_REPEAT] layer={} n_kv={} rep={} us={}",
                            i, attn_seq_len, rep, rep_us
                        );
                    }
                    if debug_sync {
                        ocl::core::finish(queue).ok();
                        eprintln!(
                            "[Plan] L{} flash attention OK (attn_seq_len={})",
                            i, attn_seq_len
                        );
                    }
                }
                AttentionVariant::KiviAssembled {
                    scatter_k,
                    scatter_v,
                    attn,
                } => {
                    // Scatter residual to F32 attn buffer (with updated res_tokens)
                    Self::dispatch_step(
                        backend,
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
                        backend,
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
                        backend,
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
                        backend,
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

            // Steps 9-10: post-attention pre-FFN (Wo matmul, add_rms_norm).
            for (si, step) in layer_plan.steps_post_attn_pre_ffn.iter().enumerate() {
                Self::dispatch_step(
                    backend,
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
                        "[Plan] L{} post_attn_pre_ffn[{}] {:?} OK",
                        i, si, step.op_tag
                    );
                }
            }

            // Steps 11-14: FFN (GPU-only or cooperative partition).
            let skip_post_ffn = match &layer_plan.ffn {
                FfnVariant::GpuOnly {
                    gate,
                    up,
                    silu_mul,
                    down,
                } => {
                    for step in [gate, up, silu_mul, down] {
                        Self::dispatch_step(
                            backend,
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
                            eprintln!("[Plan] L{} ffn {:?} OK", i, step.op_tag);
                        }
                    }
                    false
                }
                FfnVariant::Partitioned(step) => {
                    step.run(backend, i)?;
                    if debug_sync {
                        ocl::core::finish(queue).ok();
                        eprintln!("[Plan] L{} partition FFN OK", i);
                    }
                    // `Deferred` merge defers both the merge and the final
                    // `add_assign` to the next layer's fused_norm_merge.
                    // Today the plan path has no fused_norm_merge wiring yet
                    // (see `PartitionMerge::Deferred` doc), so the add_assign
                    // still runs here and we effectively behave like Inline.
                    matches!(step.merge, PartitionMerge::Deferred) && !step.is_last_layer
                }
            };

            // Step 15: add_assign (x += down). Partition `Deferred` layers
            // fold this into the next layer's fused norm kernel.
            if !skip_post_ffn {
                for (si, step) in layer_plan.steps_post_ffn.iter().enumerate() {
                    Self::dispatch_step(
                        backend,
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
                        eprintln!("[Plan] L{} post_ffn[{}] {:?} OK", i, si, step.op_tag);
                    }
                }
            }
            cache.advance_pos(1);

            if layer_plan.flush_after
                && let Err(e) = ocl::core::flush(queue)
            {
                log::error!("Plan flush failed: {}", e);
            }
        }

        // GPU score accumulator: flush step-local scores into cumulative
        // importance and clear step buffers. Mirrors transformer.rs:979 for
        // the non-plan path. Uses the post-advance cache position (one past
        // the token just scattered) so `end_step` sees the same length the
        // runtime sees.
        if self.writes_gpu_scores
            && !kv_caches.is_empty()
            && let Some(gpu_acc) = backend.gpu_score_acc_mut()
            && gpu_acc.is_active()
        {
            let cache_seq_len = kv_caches[0].current_pos();
            if let Err(e) = gpu_acc.end_step(queue, cache_seq_len) {
                log::error!(
                    "Plan gpu_score end_step failed: n_kv={}: {}",
                    cache_seq_len,
                    e
                );
            }
        }

        // Final norm
        if backend.profile_events_enabled {
            if let Err(e) = backend.enqueue_kernel_labeled(
                &self.final_norm.kernel,
                self.final_norm.op_tag.profile_label(),
                self.final_norm.ndim,
                &self.final_norm.global_work_size,
                self.final_norm.local_work_size,
            ) {
                log::error!("Plan enqueue_kernel_labeled final_norm failed: {}", e);
            }
        } else {
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
        }

        // lm_head matmul (skipped when lm_head is on CPU)
        if let Some(ref lm_head) = self.lm_head {
            if backend.profile_events_enabled {
                if let Err(e) = backend.enqueue_kernel_labeled(
                    &lm_head.kernel,
                    lm_head.op_tag.profile_label(),
                    lm_head.ndim,
                    &lm_head.global_work_size,
                    lm_head.local_work_size,
                ) {
                    log::error!("Plan enqueue_kernel_labeled lm_head failed: {}", e);
                }
            } else {
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
        }

        if op_trace {
            OP_TRACE_ACC.with(|c| {
                if let Some(m) = c.borrow_mut().take() {
                    let mut entries: Vec<(String, u64)> = m.into_iter().collect();
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                    let parts: Vec<String> = entries
                        .iter()
                        .map(|(k, v)| format!("{}={}", k, v))
                        .collect();
                    eprintln!("[OP_TRACE] n_kv={} {}", trace_n_kv, parts.join(" "));
                }
            });
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
    /// AOS Q4_0 matmul program (`kernel_mul_mat_q4_0_f32`). Used for small-N
    /// matmuls where noshuffle GEMV under-utilizes the GPU (e.g. K/V projection).
    pub q4_0_program: &'a ocl::Program,
    // Buffer handles (cl_mem) — model weights
    pub x_buf: &'a Mem,
    pub wq_buf: &'a Mem,
    pub wk_buf: &'a Mem,
    pub wv_buf: &'a Mem,
    /// Optional QKV bias buffer for Q (F32). When `Some`, the plan
    /// builder appends a `kernel_add_row_bias` step after the Q matmul.
    pub bq_buf: Option<&'a Mem>,
    /// Optional QKV bias buffer for K (F32).
    pub bk_buf: Option<&'a Mem>,
    /// Optional QKV bias buffer for V (F32).
    pub bv_buf: Option<&'a Mem>,
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
    /// Flash attention F32-Q / F16-KV program handle for head_dim=64.
    /// Used to create `flash_attn_f32_f16_q1` at plan-build time when
    /// runtime preconditions hold and the layer's head_dim is 64.
    /// `None` forces the legacy path for head_dim=64 models.
    pub flash_attn_f32_f16_program_dk64: Option<&'a ocl::Program>,
    /// Flash attention F32-Q / F16-KV program handle for head_dim=128.
    /// Used for Qwen 2.5-1.5B and other head_dim=128 models.
    /// `None` forces the legacy path for head_dim=128 models.
    pub flash_attn_f32_f16_program_dk128: Option<&'a ocl::Program>,
    /// True if this decode plan must capture attention scores (H2O/H2O+ or
    /// an active GPU score accumulator). When true, the builder must use
    /// `AttentionVariant::Standard` because the flash kernel has no score
    /// output.
    pub needs_attention_scores: bool,
    /// Persistent GPU score buffer from the backend's `GpuScoreAccumulator`.
    /// When `Some` and `needs_attention_scores` is true, the legacy attention
    /// step is pre-bound to write softmax scores directly into this buffer
    /// (arg 4, `write_scores=1`, `score_stride` below). Per-layer reduction
    /// (`GpuScoreAccumulator::reduce_layer`) is then driven by `FullKernelPlan::execute`.
    pub gpu_score_buf: Option<&'a Mem>,
    /// Score stride (= `max_seq_len`) matching `gpu_score_buf` layout. Unused
    /// when `gpu_score_buf` is `None`.
    pub gpu_score_stride: i32,
    /// Base offset (in f32 elements) for this layer's slice of `gpu_score_buf`.
    /// The score buffer has layout `[n_layers, n_heads_q, score_stride]`, so
    /// layer `l`'s base offset is `l * n_heads_q * score_stride`. This is
    /// pre-baked into the attention kernel's `score_layer_offset` arg by the
    /// layer builder, avoiding per-token arg updates. Unused when
    /// `gpu_score_buf` is `None`.
    pub gpu_score_layer_offset: i32,
    // -- Q4_0 noshuffle matmul support --
    /// Pre-compiled noshuffle GEMV programs, keyed by ne01 (M dimension).
    /// When `Some`, matmul steps use Q4_0 noshuffle dispatch instead of F16.
    pub noshuffle_programs: Option<&'a std::collections::HashMap<usize, ocl::Program>>,
    /// Per-weight noshuffle SOA entries (q_img + d_buf + dimensions).
    /// When `noshuffle_programs` is `Some`, these must also be `Some`.
    pub wq_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub wk_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub wv_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub wo_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub w_gate_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub w_up_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub w_down_noshuffle: Option<NoshufflePlanEntry<'a>>,
}

/// Lightweight reference to a noshuffle SOA entry for plan building.
/// Avoids coupling plan.rs to NoshuffleSoaEntry's full layout.
#[derive(Clone, Copy)]
pub struct NoshufflePlanEntry<'a> {
    /// image1d_buffer_t wrapping SOA nibbles (R32UI)
    pub q_img: &'a Mem,
    /// SOA scales buffer (half2*)
    pub d_buf: &'a Mem,
    /// K dimension (elements per row)
    pub ne00: usize,
    /// M dimension (number of output rows)
    pub ne01: usize,
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

/// Helper: create a dedicated kernel and pre-bind Q4_0 noshuffle GEMV arguments.
///
/// Must mirror `OpenClBackend::matmul_q4_0_noshuffle` dispatch exactly.
///
/// The noshuffle GEMV kernel uses image1d_buffer_t for both weight nibbles (R32UI)
/// and activation (RGBA32F). The activation image wraps the source F32 buffer and
/// is retained in `retained_bufs` so its cl_mem stays valid for the plan's lifetime.
///
/// Kernel args:
///   0: weight nibbles image (image1d_buffer_t, R32UI)
///   1: weight scales (global half2*)
///   2: activation image (image1d_buffer_t, RGBA32F)
///   3: output buffer (global float*)
///   4: ne00 (K dimension, i32)
///   5: ne01 (M dimension, i32)
///
/// Dispatch: global=[ne01/2, N_SIMDGROUP=4, 1], local=[64, 4, 1], ndim=2.
#[allow(clippy::too_many_arguments)]
fn make_q4_0_noshuffle_matmul_step(
    program: &ocl::Program,
    context: &ocl::core::Context,
    q_img: &Mem,
    d_buf: &Mem,
    src_buf: &Mem,
    dst_buf: &Mem,
    ne00: usize,
    ne01: usize,
    op_tag: OpTag,
) -> Result<KernelStep> {
    let kernel = ocl::core::create_kernel(program, "kernel_gemv_noshuffle_q4_0")
        .context("create kernel_gemv_noshuffle_q4_0 for plan")?;

    // Create activation image1d_buffer_t (RGBA32F) wrapping the F32 source buffer.
    // Each texel = float4 (4 floats), so width = ne00 / 4.
    // SAFETY: src_buf has at least ne00 * sizeof(f32) bytes, and ne00 is always a
    // multiple of 4 for Q4_0 (QK4_0=32).
    let act_img = {
        use ocl::core::{
            ImageChannelDataType, ImageChannelOrder, ImageDescriptor, ImageFormat, MemObjectType,
        };
        let fmt = ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelDataType::Float);
        let desc = ImageDescriptor::new(
            MemObjectType::Image1dBuffer,
            ne00 / 4,
            0,
            0,
            0,
            0,
            0,
            Some(src_buf.clone()),
        );
        unsafe {
            ocl::core::create_image(
                context,
                ocl::core::MEM_READ_ONLY,
                &fmt,
                &desc,
                None::<&[f32]>,
                None,
            )?
        }
    };

    let ne00_i = ne00 as i32;
    let ne01_i = ne01 as i32;

    unsafe {
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(q_img))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(d_buf))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(&act_img))?;
        ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::mem(dst_buf))?;
        ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&ne00_i))?;
        ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&ne01_i))?;
    }

    let simdgroup_width: usize = 64;
    let n_simdgroup: usize = 4;
    let global_work_size = [ne01 / 2, n_simdgroup, 1];
    let local_work_size = [simdgroup_width, n_simdgroup, 1];

    Ok(KernelStep {
        kernel,
        ndim: 2,
        global_work_size,
        local_work_size: Some(local_work_size),
        dynamic_args: vec![],
        op_tag,
        retained_bufs: vec![act_img],
    })
}

/// Build an AOS Q4_0 matmul step using `kernel_mul_mat_q4_0_f32`.
/// Better for small-N projections (e.g. K/V where N=n_kv_heads*head_dim is small)
/// because it dispatches `ceil(N/4)*64` threads along N, giving better GPU utilization
/// than noshuffle GEMV (which creates only N/2 threads along rows).
///
/// Weight is in AOS BlockQ4_0 layout (18 bytes per block: 2 bytes half scale + 16 bytes nibbles).
#[allow(clippy::too_many_arguments)]
fn make_q4_0_aos_matmul_step(
    program: &ocl::Program,
    weight_buf: &Mem,
    src_buf: &Mem,
    dst_buf: &Mem,
    k: usize,
    n: usize,
    op_tag: OpTag,
) -> Result<KernelStep> {
    let kernel = ocl::core::create_kernel(program, "kernel_mul_mat_q4_0_f32")
        .context("create kernel_mul_mat_q4_0_f32 for plan")?;

    let m = 1i32;
    let ne00 = k as i32;
    let ne01 = n as i32;
    let ne02 = 1i32;
    let ne10 = k as i32;
    let ne12 = k as i32;
    let ne0 = n as i32;
    let ne1 = n as i32;
    let r2 = 1i32;
    let r3 = 1i32;
    let zero_u64 = 0u64;
    unsafe {
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(weight_buf))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(src_buf))?;
        ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::mem(dst_buf))?;
        ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&zero_u64))?;
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

    let global_work_size = [n.div_ceil(4) * 64, m as usize, 1];
    let local_work_size = [64, 1, 1];

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

/// N threshold below which AOS Q4_0 matmul is preferred over noshuffle GEMV.
/// At small N, noshuffle's gws=[N/2, 4, 1] creates too few workgroups to saturate
/// Adreno 830's 16+ compute units. AOS's gws=[ceil(N/4)*64, 1, 1] scales better.
/// Empirical crossover (microbench_ops on Qwen 2.5-1.5B Q4_0):
///   N=256 (K/V proj):  aos 10μs  vs noshuffle 21μs → aos wins
///   N=1536 (Q/O proj): aos 35μs  vs noshuffle 24μs → noshuffle wins
///   N=8960 (FFN):      aos 164μs vs noshuffle 128μs → noshuffle wins
const SMALL_N_AOS_THRESHOLD: usize = 512;

/// Build noshuffle GEMV programs, keyed by ne01 (M dimension).
///
/// Each unique ne01 requires different compile-time defines (LINE_STRIDE_A,
/// BLOCK_STRIDE_A). Returns a HashMap that `make_q4_0_noshuffle_matmul_step`
/// indexes into for kernel creation.
///
/// Tries vector sub_group_broadcast first (Adreno 830+), falls back to scalar.
pub fn build_noshuffle_programs(
    device: &ocl::Device,
    context: &ocl::Context,
    cl_opts: &str,
    ne01_set: &[usize],
) -> Result<std::collections::HashMap<usize, ocl::Program>> {
    let gemv_src = include_str!("../../../kernels/gemv_noshuffle_q4_0.cl");
    let mut programs = std::collections::HashMap::new();

    for &ne01 in ne01_set {
        if programs.contains_key(&ne01) {
            continue;
        }
        let line_stride_a = ne01 / 2;
        let block_stride_a = 4 * ne01;
        let simdgroup_width: usize = 64;
        let defines = format!(
            "{} -DLINE_STRIDE_A={} -DBLOCK_STRIDE_A={} -DSIMDGROUP_WIDTH={}",
            cl_opts, line_stride_a, block_stride_a, simdgroup_width
        );

        // Try vector sub_group_broadcast first (Adreno 830+ / driver v47+)
        let defines_vec = format!("{} -DVECTOR_SUB_GROUP_BROADCAT", defines);
        let program = match ocl::Program::builder()
            .devices(device)
            .src(gemv_src)
            .cmplr_opt(&defines_vec)
            .build(context)
        {
            Ok(p) => p,
            Err(_) => ocl::Program::builder()
                .devices(device)
                .src(gemv_src)
                .cmplr_opt(&defines)
                .build(context)
                .with_context(|| format!("build noshuffle program for ne01={}", ne01))?,
        };
        programs.insert(ne01, program);
    }

    Ok(programs)
}

/// Build a pre-bound `KernelStep` that dispatches `flash_attn_f32_f16_q1`.
///
/// Arg layout mirrors `OpenCLBackend::flash_attention_decode_gpu` — see that
/// method in `engine/src/backend/opencl/mod.rs` for the canonical layout.
/// All 40 args are static except `n_kv` at index 10, which is patched per
/// decode step via `DynamicArg::CacheSeqLen`.
fn build_flash_attention_step(config: &LayerPlanConfig) -> Result<AttentionVariant> {
    // Pick the program matching this layer's head_dim. The caller
    // (`use_flash` gate) guarantees the matching program is `Some`.
    let program = match config.head_dim {
        64 => config.flash_attn_f32_f16_program_dk64,
        128 => config.flash_attn_f32_f16_program_dk128,
        _ => None,
    }
    .expect("caller must verify flash program is Some for this head_dim");

    let kernel = ocl::core::create_kernel(program, "flash_attn_f32_f16_q1")
        .context("create flash_attn_f32_f16_q1 for plan")?;

    let n_heads_q = config.n_heads_q;
    let n_heads_kv = config.n_kv_heads;
    let head_dim = config.head_dim;
    let kv_capacity = config.kv_capacity;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // Q strides (F32 [batch=1, seq=1, n_heads_q, head_dim]), bytes
    let q_nb1 = (n_heads_q * head_dim * 4) as u64;
    let q_nb2 = (head_dim * 4) as u64;
    let q_nb3 = q_nb1;

    // KV strides (F16 HeadMajor [1, n_heads_kv, capacity, head_dim]), bytes
    let kv_elem_size: u64 = 2;
    let k_nb1 = (head_dim as u64) * kv_elem_size;
    let k_nb2 = (kv_capacity * head_dim) as u64 * kv_elem_size;
    let k_nb3 = (n_heads_kv as u64) * k_nb2;

    // O strides (F32 [batch=1, seq=1, n_heads_q, head_dim]), bytes
    let o_nb1 = (head_dim * 4) as u64;
    let o_nb2 = (n_heads_q * head_dim * 4) as u64;
    let o_nb3 = o_nb2;

    let n_q = 1i32;
    let initial_n_kv = 0i32;
    let is_causal = 0i32;
    let n_head = n_heads_q as i32;
    let n_head_kv_arg = n_heads_kv as i32;
    let max_bias = 0.0f32;
    let m0 = 0.0f32;
    let m1 = 0.0f32;
    let n_head_log2 = 0i32;
    let logit_softcap = 0.0f32;
    let zero_u64 = 0u64;
    let zero_i32 = 0i32;

    unsafe {
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.q_buf))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(config.k_cache_buf))?;
        ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::mem(config.v_cache_buf))?;
        ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::mem(config.out_attn_buf))?;
        ocl::core::set_kernel_arg(&kernel, 7, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 8, ocl::core::ArgVal::scalar(&scale))?;
        ocl::core::set_kernel_arg(&kernel, 9, ocl::core::ArgVal::scalar(&n_q))?;
        ocl::core::set_kernel_arg(&kernel, 10, ocl::core::ArgVal::scalar(&initial_n_kv))?;
        ocl::core::set_kernel_arg(&kernel, 11, ocl::core::ArgVal::scalar(&is_causal))?;
        ocl::core::set_kernel_arg(&kernel, 12, ocl::core::ArgVal::scalar(&n_head))?;
        ocl::core::set_kernel_arg(&kernel, 13, ocl::core::ArgVal::scalar(&q_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 14, ocl::core::ArgVal::scalar(&q_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 15, ocl::core::ArgVal::scalar(&q_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 16, ocl::core::ArgVal::scalar(&k_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 17, ocl::core::ArgVal::scalar(&k_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 18, ocl::core::ArgVal::scalar(&k_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 19, ocl::core::ArgVal::scalar(&k_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 20, ocl::core::ArgVal::scalar(&k_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 21, ocl::core::ArgVal::scalar(&k_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 22, ocl::core::ArgVal::scalar(&o_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 23, ocl::core::ArgVal::scalar(&o_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 24, ocl::core::ArgVal::scalar(&o_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 25, ocl::core::ArgVal::scalar(&max_bias))?;
        ocl::core::set_kernel_arg(&kernel, 26, ocl::core::ArgVal::scalar(&m0))?;
        ocl::core::set_kernel_arg(&kernel, 27, ocl::core::ArgVal::scalar(&m1))?;
        ocl::core::set_kernel_arg(&kernel, 28, ocl::core::ArgVal::scalar(&n_head_log2))?;
        ocl::core::set_kernel_arg(&kernel, 29, ocl::core::ArgVal::scalar(&logit_softcap))?;
        ocl::core::set_kernel_arg(&kernel, 30, ocl::core::ArgVal::scalar(&n_head_kv_arg))?;
        ocl::core::set_kernel_arg(&kernel, 31, ocl::core::ArgVal::mem_null())?;
        ocl::core::set_kernel_arg(&kernel, 32, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 33, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 34, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 35, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 36, ocl::core::ArgVal::scalar(&zero_i32))?;
        ocl::core::set_kernel_arg(&kernel, 37, ocl::core::ArgVal::scalar(&zero_i32))?;
        ocl::core::set_kernel_arg(&kernel, 38, ocl::core::ArgVal::mem_null())?;
        ocl::core::set_kernel_arg(&kernel, 39, ocl::core::ArgVal::scalar(&zero_u64))?;
    }

    // Post-softmax score output (args 40-43). When a GPU score buffer is
    // supplied via LayerPlanConfig, bind it here so the Q1 kernel can write
    // per-token weights directly — avoiding the legacy kernel_attn_gen_half
    // fallback that used to dominate decode wall-clock. Otherwise bind a
    // 1-element dummy buffer and disable writes.
    let (score_mem, score_stride_val, score_layer_offset_val, write_scores, retained_bufs): (
        Mem,
        i32,
        i32,
        i32,
        Vec<Mem>,
    ) = match config.gpu_score_buf {
        Some(buf) => (
            buf.clone(),
            config.gpu_score_stride,
            config.gpu_score_layer_offset,
            1i32,
            vec![],
        ),
        None => {
            let dummy = unsafe {
                ocl::core::create_buffer::<_, f32>(
                    config.context.as_core(),
                    ocl::core::MEM_READ_WRITE,
                    1,
                    None,
                )
            }
            .context("create dummy score buffer for flash plan")?;
            let d_clone = dummy.clone();
            (dummy, 0i32, 0i32, 0i32, vec![d_clone])
        }
    };
    unsafe {
        ocl::core::set_kernel_arg(&kernel, 40, ocl::core::ArgVal::mem(&score_mem))?;
        ocl::core::set_kernel_arg(
            &kernel,
            41,
            ocl::core::ArgVal::scalar(&score_layer_offset_val),
        )?;
        ocl::core::set_kernel_arg(&kernel, 42, ocl::core::ArgVal::scalar(&score_stride_val))?;
        ocl::core::set_kernel_arg(&kernel, 43, ocl::core::ArgVal::scalar(&write_scores))?;
    }

    const Q1_WG_SIZE: usize = 64;
    Ok(AttentionVariant::StandardFlash(KernelStep {
        kernel,
        ndim: 2,
        global_work_size: [Q1_WG_SIZE, n_heads_q, 1],
        local_work_size: Some([Q1_WG_SIZE, 1, 1]),
        dynamic_args: vec![DynamicArg::CacheSeqLen { arg_idx: 10 }],
        op_tag: OpTag::Attention,
        retained_bufs,
    }))
}

/// Build a pre-bound `kernel_add_row_bias` step that adds the given bias
/// buffer to the given `x` buffer in-place. Used after QKV matmul steps
/// for models with `has_qkv_bias=true` (Qwen2 etc.).
///
/// Kernel signature (from simple_ops.cl:487):
///   kernel_add_row_bias(float* x, const float* bias, int dim, int total)
///
/// Dispatch: 1D, global = total.div_ceil(64) * 64, no local size.
/// For decode seq_len=1 batch=1, `total == dim == n_heads * head_dim`.
fn build_add_row_bias_step(
    simple_ops_program: &ocl::Program,
    x_buf: &Mem,
    bias_buf: &Mem,
    dim: usize,
    op_tag: OpTag,
) -> Result<KernelStep> {
    let kernel = ocl::core::create_kernel(simple_ops_program, "kernel_add_row_bias")
        .context("create kernel_add_row_bias for plan")?;
    let dim_i32 = dim as i32;
    let total_i32 = dim as i32; // decode: 1 row × dim elements
    unsafe {
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(bias_buf))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&dim_i32))?;
        ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&total_i32))?;
    }
    let gws = dim.div_ceil(64) * 64;
    Ok(KernelStep {
        kernel,
        ndim: 1,
        global_work_size: [gws, 1, 1],
        local_work_size: None,
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

    let mut steps_pre_kv = Vec::with_capacity(9);

    // -----------------------------------------------------------------------
    // 1. rms_norm_oop (x -> residual)
    // -----------------------------------------------------------------------
    {
        // float4 path for dim divisible by 4 (all decoder arches).
        let kernel_name = if dim % 4 == 0 {
            "kernel_rms_norm_oop_f4"
        } else {
            "kernel_rms_norm_oop"
        };
        let kernel = ocl::core::create_kernel(config.simple_ops_program, kernel_name)
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
    if let (Some(ns), Some(progs)) = (&config.wq_noshuffle, config.noshuffle_programs) {
        let prog = progs
            .get(&ns.ne01)
            .context("noshuffle program for wq ne01")?;
        steps_pre_kv.push(make_q4_0_noshuffle_matmul_step(
            prog,
            config.context.as_core(),
            ns.q_img,
            ns.d_buf,
            config.residual_buf,
            config.q_buf,
            ns.ne00,
            ns.ne01,
            OpTag::MatmulQKV,
        )?);
    } else {
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
    }

    // Optional: add Q bias (Qwen2 etc.) — non-bias models have bq_buf = None.
    if let Some(bq) = config.bq_buf {
        steps_pre_kv.push(build_add_row_bias_step(
            config.simple_ops_program,
            config.q_buf,
            bq,
            config.n_q,
            OpTag::MatmulQKV,
        )?);
    }

    // -----------------------------------------------------------------------
    // 3. matmul K (residual -> k)
    // -----------------------------------------------------------------------
    // NOTE: Tried AOS path for small-N K proj (N <= SMALL_N_AOS_THRESHOLD) — in
    // isolated microbench 10μs vs noshuffle 21μs, but production pp4096 showed
    // +24ms/tok regression (cause unclear, possibly cache/pipe interaction with
    // surrounding ops). Reverted to noshuffle pending investigation.
    if false
        && let Some(ns) = &config.wk_noshuffle
        && ns.ne01 <= SMALL_N_AOS_THRESHOLD
    {
        steps_pre_kv.push(make_q4_0_aos_matmul_step(
            config.q4_0_program,
            config.wk_buf,
            config.residual_buf,
            config.k_buf,
            ns.ne00,
            ns.ne01,
            OpTag::MatmulQKV,
        )?);
    } else if let (Some(ns), Some(progs)) = (&config.wk_noshuffle, config.noshuffle_programs) {
        let prog = progs
            .get(&ns.ne01)
            .context("noshuffle program for wk ne01")?;
        steps_pre_kv.push(make_q4_0_noshuffle_matmul_step(
            prog,
            config.context.as_core(),
            ns.q_img,
            ns.d_buf,
            config.residual_buf,
            config.k_buf,
            ns.ne00,
            ns.ne01,
            OpTag::MatmulQKV,
        )?);
    } else {
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
    }

    // Optional: add K bias (Qwen2 etc.) — non-bias models have bk_buf = None.
    if let Some(bk) = config.bk_buf {
        steps_pre_kv.push(build_add_row_bias_step(
            config.simple_ops_program,
            config.k_buf,
            bk,
            config.n_k,
            OpTag::MatmulQKV,
        )?);
    }

    // -----------------------------------------------------------------------
    // 4. matmul V (residual -> v)
    // -----------------------------------------------------------------------
    if false
        && let Some(ns) = &config.wv_noshuffle
        && ns.ne01 <= SMALL_N_AOS_THRESHOLD
    {
        steps_pre_kv.push(make_q4_0_aos_matmul_step(
            config.q4_0_program,
            config.wv_buf,
            config.residual_buf,
            config.v_buf,
            ns.ne00,
            ns.ne01,
            OpTag::MatmulQKV,
        )?);
    } else if let (Some(ns), Some(progs)) = (&config.wv_noshuffle, config.noshuffle_programs) {
        let prog = progs
            .get(&ns.ne01)
            .context("noshuffle program for wv ne01")?;
        steps_pre_kv.push(make_q4_0_noshuffle_matmul_step(
            prog,
            config.context.as_core(),
            ns.q_img,
            ns.d_buf,
            config.residual_buf,
            config.v_buf,
            ns.ne00,
            ns.ne01,
            OpTag::MatmulQKV,
        )?);
    } else {
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
    }

    // Optional: add V bias (Qwen2 etc.) — non-bias models have bv_buf = None.
    if let Some(bv) = config.bv_buf {
        steps_pre_kv.push(build_add_row_bias_step(
            config.simple_ops_program,
            config.v_buf,
            bv,
            config.n_v,
            OpTag::MatmulQKV,
        )?);
    }

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
    // 8. attention (q, kv_cache -> out_attn)
    // -----------------------------------------------------------------------
    // Precondition gate: select flash attention when all conditions hold at
    // plan-build time. These are static for a given model + KV layout, so we
    // pre-bake the choice instead of runtime-gating per step.
    //
    // HeadMajor predicate matches the runtime check in `attention_gen` —
    // kv_pos_stride == head_dim and kv_head_stride == capacity * head_dim.
    //
    // Implicit invariants (safe today, must be revisited on refactor):
    //   1. KV dtype is F16 — plan.rs is only invoked for F16 KV caches;
    //      Q4_0 and KIVI use separate code paths. Add an explicit
    //      `config.kv_dtype == DType::F16` check if a new KV dtype gets
    //      plan-routed.
    //   2. `is_head_major` is reverse-inferred from stride values set by
    //      the caller. This assumes HeadMajor is the only layout that
    //      sets `kv_pos_stride = head_dim` and
    //      `kv_head_stride = capacity * head_dim` (see `attention_gen`
    //      in mod.rs for the canonical assignment).
    let is_head_major = config.kv_pos_stride == config.head_dim as i32
        && config.kv_head_stride == (config.kv_capacity * config.head_dim) as i32;
    // Flash attention is gated per head_dim because each DK variant is a
    // separate compiled program. Add a new arm here when adding DK=256
    // (Gemma3) etc. The Q1 kernel now emits post-softmax scores directly
    // when a GPU score buffer is supplied, so `needs_attention_scores`
    // alone no longer forces the legacy path — it does only when scores
    // must land in a CPU-readback buffer (`gpu_score_buf == None`).
    let flash_program_available = match config.head_dim {
        64 => config.flash_attn_f32_f16_program_dk64.is_some(),
        128 => config.flash_attn_f32_f16_program_dk128.is_some(),
        _ => false,
    };
    let scores_need_legacy_readback =
        config.needs_attention_scores && config.gpu_score_buf.is_none();
    let use_flash = is_head_major && flash_program_available && !scores_need_legacy_readback;

    let attention = if use_flash {
        build_flash_attention_step(config)?
    } else {
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_attn_gen_half")
            .context("create kernel_attn_gen_half")?;
        let scale = 1.0f32 / (config.head_dim as f32).sqrt();
        let cache_seq_len_init = 0i32;
        // When a GPU score accumulator buffer is supplied and scores are
        // required, bind it as arg 4 (`scores`) with `write_scores=1` and
        // the accumulator's stride. This mirrors the runtime path in
        // `OpenCLBackend::attention_gen` (mod.rs:~3858), allowing the plan
        // to accumulate per-step importance without round-tripping through
        // forward_gen. When no GPU buffer is supplied we retain the legacy
        // dummy binding.
        let (write_scores, score_stride) =
            if config.needs_attention_scores && config.gpu_score_buf.is_some() {
                (1i32, config.gpu_score_stride)
            } else {
                (0i32, 0i32)
            };
        let dummy_score_buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                config.context.as_core(),
                ocl::core::MEM_READ_WRITE,
                1,
                None,
            )
        }
        .context("create dummy score buffer for plan")?;
        let score_arg_buf: &Mem = if write_scores == 1 {
            config.gpu_score_buf.unwrap()
        } else {
            &dummy_score_buf
        };
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.q_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.k_cache_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(config.v_cache_buf))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::mem(config.out_attn_buf))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::mem(score_arg_buf))?;
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
                ocl::core::ArgVal::scalar(&config.gpu_score_layer_offset),
            )?;
            ocl::core::set_kernel_arg(
                &kernel,
                15,
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
    // Steps 9-10: post-attention pre-FFN (Wo + add_rms_norm)
    // -----------------------------------------------------------------------
    let mut steps_post_attn_pre_ffn = Vec::with_capacity(2);

    // 9. matmul Wo (out_attn -> attn_out)
    if let (Some(ns), Some(progs)) = (&config.wo_noshuffle, config.noshuffle_programs) {
        let prog = progs
            .get(&ns.ne01)
            .context("noshuffle program for wo ne01")?;
        steps_post_attn_pre_ffn.push(make_q4_0_noshuffle_matmul_step(
            prog,
            config.context.as_core(),
            ns.q_img,
            ns.d_buf,
            config.out_attn_buf,
            config.attn_out_buf,
            ns.ne00,
            ns.ne01,
            OpTag::MatmulWo,
        )?);
    } else {
        steps_post_attn_pre_ffn.push(make_f16_matmul_step(
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
    }

    // 10. add_rms_norm_oop (x += attn_out, then norm -> residual)
    {
        let kernel_name = if dim % 4 == 0 {
            "kernel_add_rms_norm_oop_f4"
        } else {
            "kernel_add_rms_norm_oop"
        };
        let kernel = ocl::core::create_kernel(config.simple_ops_program, kernel_name)
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
        steps_post_attn_pre_ffn.push(KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [local_size, 1, 1],
            local_work_size: Some([local_size, 1, 1]),
            dynamic_args: vec![],
            op_tag: OpTag::AddRmsNorm,
            retained_bufs: vec![],
        });
    }

    // -----------------------------------------------------------------------
    // Steps 11-14: FFN (GPU-only variant built here; partition variant routed
    // through `build_partitioned_layer_plan`).
    // -----------------------------------------------------------------------
    // 11. matmul gate (residual -> gate)
    let gate_step =
        if let (Some(ns), Some(progs)) = (&config.w_gate_noshuffle, config.noshuffle_programs) {
            let prog = progs
                .get(&ns.ne01)
                .context("noshuffle program for w_gate ne01")?;
            make_q4_0_noshuffle_matmul_step(
                prog,
                config.context.as_core(),
                ns.q_img,
                ns.d_buf,
                config.residual_buf,
                config.gate_buf,
                ns.ne00,
                ns.ne01,
                OpTag::MatmulGateUp,
            )?
        } else {
            make_f16_matmul_step(
                config.f16_program,
                config.residual_buf,
                config.w_gate_buf,
                config.gate_buf,
                config.ffn_hidden,
                k,
                OpTag::MatmulGateUp,
                config.f16_l4_program,
                config.is_nosub,
            )?
        };

    // 12. matmul up (residual -> up)
    let up_step =
        if let (Some(ns), Some(progs)) = (&config.w_up_noshuffle, config.noshuffle_programs) {
            let prog = progs
                .get(&ns.ne01)
                .context("noshuffle program for w_up ne01")?;
            make_q4_0_noshuffle_matmul_step(
                prog,
                config.context.as_core(),
                ns.q_img,
                ns.d_buf,
                config.residual_buf,
                config.up_buf,
                ns.ne00,
                ns.ne01,
                OpTag::MatmulGateUp,
            )?
        } else {
            make_f16_matmul_step(
                config.f16_program,
                config.residual_buf,
                config.w_up_buf,
                config.up_buf,
                config.ffn_hidden,
                k,
                OpTag::MatmulGateUp,
                config.f16_l4_program,
                config.is_nosub,
            )?
        };

    // 13. silu_mul (gate = silu(gate) * up)
    let silu_mul_step = {
        let kernel = ocl::core::create_kernel(config.simple_ops_program, "kernel_silu_mul_simple")
            .context("create kernel_silu_mul_simple")?;
        let size4 = (config.ffn_hidden / 4) as i32;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.gate_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.up_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&size4))?;
        }
        KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [config.ffn_hidden / 4, 1, 1],
            local_work_size: None,
            dynamic_args: vec![],
            op_tag: OpTag::SiluMul,
            retained_bufs: vec![],
        }
    };

    // 14. matmul down (gate -> down)
    let down_step =
        if let (Some(ns), Some(progs)) = (&config.w_down_noshuffle, config.noshuffle_programs) {
            let prog = progs
                .get(&ns.ne01)
                .context("noshuffle program for w_down ne01")?;
            make_q4_0_noshuffle_matmul_step(
                prog,
                config.context.as_core(),
                ns.q_img,
                ns.d_buf,
                config.gate_buf,
                config.down_buf,
                ns.ne00,
                ns.ne01,
                OpTag::MatmulDown,
            )?
        } else {
            make_f16_matmul_step(
                config.f16_program,
                config.gate_buf,
                config.w_down_buf,
                config.down_buf,
                dim,
                config.ffn_hidden,
                OpTag::MatmulDown,
                None,
                config.is_nosub,
            )?
        };

    let ffn = FfnVariant::GpuOnly {
        gate: gate_step,
        up: up_step,
        silu_mul: silu_mul_step,
        down: down_step,
    };

    // -----------------------------------------------------------------------
    // Step 15: post-FFN (residual add). Partition layers with `Deferred`
    // merge skip this step on non-last layers — the plan executor owns that
    // logic, so we always emit it here and let the caller suppress.
    // -----------------------------------------------------------------------
    let mut steps_post_ffn = Vec::with_capacity(1);
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
        steps_post_ffn.push(KernelStep {
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
        steps_post_attn_pre_ffn,
        ffn,
        steps_post_ffn,
        flush_after: false,
    })
}

/// Build a layer plan with a cooperative FFN partition (see arch A.3/A.7.1).
///
/// `config` is the same LayerPlanConfig used by `build_layer_plan` — shared
/// QKV/attention/Wo bindings. The FFN section is replaced with a
/// `FfnVariant::Partitioned` that wraps GPU slice dispatches + a
/// `PartitionPlanContext` carrying CPU-side handles.
///
/// Returns `Err` when the plan path is disabled (`LLMRS_PARTITION_PLAN=0`) or
/// when an unsupported mode is active (`LLMRS_PARTITION_REPLICATE_NORM=1`,
/// `LLMRS_PARTITION_SYNC_EVERY_N>1`). The caller is expected to fall back to
/// `forward_gen` in those cases.
#[allow(clippy::too_many_arguments)]
pub fn build_partitioned_layer_plan(
    config: &LayerPlanConfig,
    partition_ctx: &PartitionContext,
    workspace: &Arc<PartitionWsCell>,
    cpu_backend: &Arc<dyn Backend>,
    use_gelu_tanh: bool,
    layer_idx: usize,
    is_last_layer: bool,
) -> Result<LayerKernelPlan> {
    use std::sync::atomic::Ordering;

    if !partition_plan_enabled() {
        anyhow::bail!("LLMRS_PARTITION_PLAN=0 — partition plan path disabled");
    }
    if partition_replicate_norm_enabled() {
        anyhow::bail!("LLMRS_PARTITION_REPLICATE_NORM=1 not supported in plan path (arch A.8.2)");
    }
    if let Ok(v) = std::env::var("LLMRS_PARTITION_SYNC_EVERY_N")
        && v.parse::<u64>().ok().is_some_and(|n| n > 1)
    {
        anyhow::bail!("LLMRS_PARTITION_SYNC_EVERY_N>1 not supported in plan path (arch A.8.3)");
    }

    let dim = config.dim;
    let split_row = partition_ctx.gate.split_row; // GPU rows (gate/up out_dim slice)
    let cpu_rows = config.ffn_hidden - split_row;
    anyhow::ensure!(
        split_row > 0 && cpu_rows > 0,
        "partition split_row must be in (0, ffn_hidden), got split_row={} ffn_hidden={}",
        split_row,
        config.ffn_hidden,
    );
    anyhow::ensure!(
        split_row.is_multiple_of(4),
        "partition split_row must be a multiple of 4 for silu_mul f4 kernel, got {}",
        split_row,
    );

    // SAFETY: `workspace` is only accessed from the dispatch thread.
    let pw_ref: &PartitionPlanWorkspace = unsafe { &*workspace.get() };

    // ── Steps 1-10 (pre-FFN) are reused from the non-partition path ──
    let mut base_plan = build_layer_plan(config)?;

    // cl_mem handles for the partition-side bindings.
    let gate_gpu_mem = crate::backend::opencl::get_cl_mem(pw_ref.gate_gpu.buffer().as_ref())
        .context("partition gate_gpu cl_mem")?;
    let up_gpu_mem = crate::backend::opencl::get_cl_mem(pw_ref.up_gpu.buffer().as_ref())
        .context("partition up_gpu cl_mem")?;
    let down_partial_gpu_mem =
        crate::backend::opencl::get_cl_mem(pw_ref.down_partial_gpu.buffer().as_ref())
            .context("partition down_partial_gpu cl_mem")?;
    let cpu_merge_staging_mem =
        crate::backend::opencl::get_cl_mem(pw_ref.cpu_merge_staging.buffer().as_ref())
            .context("partition cpu_merge_staging cl_mem")?;

    // Partition weight slice cl_mem handles. `gpu_slice` tensors are produced
    // by `split_weight` / `split_weight_col` via `backend.copy_from`, so they
    // are regular (non-noshuffle) GPU buffers.
    let gate_slice_mem =
        crate::backend::opencl::get_cl_mem(partition_ctx.gate.gpu_slice.buffer().as_ref())
            .context("partition gate slice cl_mem")?;
    let up_slice_mem =
        crate::backend::opencl::get_cl_mem(partition_ctx.up.gpu_slice.buffer().as_ref())
            .context("partition up slice cl_mem")?;
    let down_slice_mem =
        crate::backend::opencl::get_cl_mem(partition_ctx.down.gpu_slice.buffer().as_ref())
            .context("partition down slice cl_mem")?;

    // Dtype dispatch: partition weights inherit the base model dtype. Q4_0
    // partition slices use the AOS path (no noshuffle SOA registered).
    let gate_dtype = partition_ctx.gate.gpu_slice.dtype();
    let use_q4_0 = gate_dtype == crate::core::buffer::DType::Q4_0;

    let build_matmul = |src: &Mem,
                        weight: &Mem,
                        dst: &Mem,
                        n: usize,
                        k: usize,
                        tag: OpTag|
     -> Result<KernelStep> {
        if use_q4_0 {
            make_q4_0_aos_matmul_step(config.q4_0_program, weight, src, dst, k, n, tag)
        } else {
            // F16 path (default for F16/BF16).
            make_f16_matmul_step(
                config.f16_program,
                src,
                weight,
                dst,
                n,
                k,
                tag,
                None,
                config.is_nosub,
            )
        }
    };

    // GPU FFN slice: gate/up on residual → gate_gpu / up_gpu (split_row cols)
    let gpu_gate = build_matmul(
        config.residual_buf,
        gate_slice_mem,
        gate_gpu_mem,
        split_row,
        dim,
        OpTag::MatmulGateUp,
    )?;
    let gpu_up = build_matmul(
        config.residual_buf,
        up_slice_mem,
        up_gpu_mem,
        split_row,
        dim,
        OpTag::MatmulGateUp,
    )?;

    // SiLU/GELU × up into gate_gpu (in place). split_row must be multiple of 4.
    let gpu_act_mul = {
        let kernel_name = if use_gelu_tanh {
            "kernel_gelu_tanh_mul_simple"
        } else {
            "kernel_silu_mul_simple"
        };
        let kernel = ocl::core::create_kernel(config.simple_ops_program, kernel_name)
            .with_context(|| format!("create {}", kernel_name))?;
        let size4 = (split_row / 4) as i32;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(gate_gpu_mem))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(up_gpu_mem))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&size4))?;
        }
        KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [split_row / 4, 1, 1],
            local_work_size: None,
            dynamic_args: vec![],
            op_tag: OpTag::SiluMul,
            retained_bufs: vec![],
        }
    };

    // down: gate_gpu [1, 1, split_row] @ down.gpu_slice [dim, split_row]^T
    //       → down_partial_gpu [1, 1, dim]
    let gpu_down = build_matmul(
        gate_gpu_mem,
        down_slice_mem,
        down_partial_gpu_mem,
        dim,
        split_row,
        OpTag::MatmulDown,
    )?;

    // ── Merge sub-steps ──
    let merge_mode_deferred = partition_fused_merge_enabled() && !is_last_layer;

    // (a) copy_slice(down_partial_gpu → ws.down), first `dim` floats.
    let copy_gpu_to_down = {
        let kernel =
            ocl::core::create_kernel(config.simple_ops_program, "kernel_copy_slice_simple")
                .context("create kernel_copy_slice_simple for partition merge")?;
        let size = dim as i32;
        let src_offset = 0i32;
        let dst_offset = 0i32;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(down_partial_gpu_mem))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(config.down_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&src_offset))?;
            ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&dst_offset))?;
            ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::scalar(&size))?;
        }
        KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [dim, 1, 1],
            local_work_size: None,
            dynamic_args: vec![],
            op_tag: OpTag::AddAssign, // reuse the post-FFN tag for tracing
            retained_bufs: vec![],
        }
    };

    // (c) add_assign(ws.down, cpu_merge_staging) — requires dim % 4 == 0.
    anyhow::ensure!(
        dim.is_multiple_of(4),
        "partition requires dim to be a multiple of 4 for add_assign_simple, got dim={}",
        dim
    );
    let add_assign_staging = {
        let kernel =
            ocl::core::create_kernel(config.simple_ops_program, "kernel_add_assign_simple")
                .context("create kernel_add_assign_simple for partition merge")?;
        let size4 = (dim / 4) as i32;
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.down_buf))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(cpu_merge_staging_mem))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&size4))?;
        }
        KernelStep {
            kernel,
            ndim: 1,
            global_work_size: [dim / 4, 1, 1],
            local_work_size: None,
            dynamic_args: vec![],
            op_tag: OpTag::AddAssign,
            retained_bufs: vec![],
        }
    };

    let merge = if merge_mode_deferred {
        PartitionMerge::Deferred
    } else {
        PartitionMerge::Inline {
            copy_gpu_to_down,
            add_assign: add_assign_staging,
        }
    };

    // Capture the generation counter. Plan becomes stale if the live counter
    // diverges (see INV-120).
    let gen_arc = partition_ctx.ratio_generation.clone();
    let ratio_generation_at_build = gen_arc.load(Ordering::Acquire);

    // Clone Tensor slices so the PartitionPlanContext owns references that
    // live for the plan's lifetime (INV-082). `Tensor::clone` is a shallow
    // Arc clone on the underlying Buffer.
    let cpu_ctx = Arc::new(PartitionPlanContext {
        cpu_backend: cpu_backend.clone(),
        gate_cpu: partition_ctx.gate.cpu_slice.clone(),
        up_cpu: partition_ctx.up.cpu_slice.clone(),
        down_cpu: partition_ctx.down.cpu_slice.clone(),
        workspace: workspace.clone(),
        residual_buf_handle: config.residual_buf.clone(),
        x_buf_handle: config.x_buf.clone(),
        attn_out_buf_handle: config.attn_out_buf.clone(),
        use_gelu_tanh,
        rms_norm_eps: config.rms_norm_eps,
        rms_norm_add_unit: false,
        layer_idx,
        partition_path: PartitionPath::SyncRead,
        ratio_generation_at_build,
        ratio_generation_counter: gen_arc,
        build_thread_id: std::thread::current().id(),
    });

    let partition_step = PartitionStep {
        gpu_gate,
        gpu_up,
        gpu_act_mul,
        gpu_down,
        cpu_ctx,
        merge,
        is_last_layer,
    };

    // Replace FFN with the partition variant; the GpuOnly gate/up/silu/down
    // kernels produced by `build_layer_plan` are discarded (they bound the
    // wrong weights for the partition path).
    base_plan.ffn = FfnVariant::Partitioned(Box::new(partition_step));
    Ok(base_plan)
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
    /// AOS Q4_0 matmul program (`kernel_mul_mat_q4_0_f32`). Used for small-N
    /// matmuls where noshuffle GEMV under-utilizes the GPU (e.g. K/V projection).
    pub q4_0_program: &'a ocl::Program,
    /// Flash attention program for head_dim=64. When `Some`, the layer
    /// builder may select `AttentionVariant::StandardFlash` for layers
    /// with head_dim=64.
    pub flash_attn_f32_f16_program_dk64: Option<&'a ocl::Program>,
    /// Flash attention program for head_dim=128. When `Some`, the layer
    /// builder may select `AttentionVariant::StandardFlash` for layers
    /// with head_dim=128 (e.g. Qwen 2.5-1.5B).
    pub flash_attn_f32_f16_program_dk128: Option<&'a ocl::Program>,
    /// True if this decode plan must capture attention scores (H2O / GPU
    /// score accumulator). Forces the legacy attention path because flash
    /// attention has no score output.
    pub needs_attention_scores: bool,
    /// Persistent GPU score buffer from `OpenCLBackend::gpu_score_acc()`.
    /// Propagated into each layer's `LayerPlanConfig::gpu_score_buf`.
    pub gpu_score_buf: Option<&'a Mem>,
    /// Score stride for the buffer above.
    pub gpu_score_stride: i32,
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
    // -- Q4_0 noshuffle matmul support --
    /// Pre-compiled noshuffle GEMV programs, keyed by ne01 (M dimension).
    /// When `Some`, layers with noshuffle entries use Q4_0 GEMV kernels.
    pub noshuffle_programs: Option<std::collections::HashMap<usize, ocl::Program>>,
    /// Per-layer noshuffle SOA entries for lm_head. `None` for F16 or CPU lm_head.
    pub lm_head_noshuffle: Option<NoshufflePlanEntry<'a>>,
    /// Optional per-layer tensor-partition context. When
    /// `partition_layers[i].is_some()`, the FFN segment of that layer is
    /// built via `build_partitioned_layer_plan` instead of the default
    /// `build_layer_plan`. Length must equal `layer_bufs.len()` when present.
    pub partition_layers: Option<Vec<Option<&'a PartitionContext>>>,
    /// Per-layer shared workspace for the partition FFN (only consulted for
    /// layers with a non-None `partition_layers[i]`). Plan execution mutates
    /// the workspace through `Arc<UnsafeCell<..>>` from the dispatch thread.
    pub partition_workspace: Option<Arc<PartitionWsCell>>,
    /// CPU-capable backend for NEON FFN slice dispatch. Required when
    /// `partition_layers` contains any `Some(..)` entry.
    pub partition_cpu_backend: Option<Arc<dyn Backend>>,
    /// Whether the model uses `gelu(tanh)` instead of SiLU in the FFN. Mirror
    /// of the layer config seen by `forward_gen`. Default `false` when
    /// `partition_layers` is not set.
    pub partition_use_gelu_tanh: bool,
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
    /// Optional QKV bias buffers (F32). Present for models with
    /// `has_qkv_bias=true` (e.g. Qwen2). When `None`, the plan builder
    /// skips the `kernel_add_row_bias` step after each QKV matmul.
    pub bq: Option<&'a Mem>,
    pub bk: Option<&'a Mem>,
    pub bv: Option<&'a Mem>,
    // -- Q4_0 noshuffle SOA entries (optional, None for F16 weights) --
    pub wq_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub wk_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub wv_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub wo_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub w_gate_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub w_up_noshuffle: Option<NoshufflePlanEntry<'a>>,
    pub w_down_noshuffle: Option<NoshufflePlanEntry<'a>>,
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
            q4_0_program: config.q4_0_program,
            x_buf: config.x_buf,
            wq_buf: lb.wq,
            wk_buf: lb.wk,
            wv_buf: lb.wv,
            bq_buf: lb.bq,
            bk_buf: lb.bk,
            bv_buf: lb.bv,
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
            flash_attn_f32_f16_program_dk64: config.flash_attn_f32_f16_program_dk64,
            flash_attn_f32_f16_program_dk128: config.flash_attn_f32_f16_program_dk128,
            needs_attention_scores: config.needs_attention_scores,
            gpu_score_buf: config.gpu_score_buf,
            gpu_score_stride: config.gpu_score_stride,
            // Per-layer offset into the [n_layers, n_heads_q, score_stride]
            // score buffer (in f32 elements). Pre-baked into the attention
            // kernel's `score_layer_offset` arg so each layer writes into its
            // own slice without per-token arg updates.
            gpu_score_layer_offset: (i as i32)
                * (config.n_heads_q as i32)
                * config.gpu_score_stride,
            noshuffle_programs: config.noshuffle_programs.as_ref(),
            wq_noshuffle: lb.wq_noshuffle,
            wk_noshuffle: lb.wk_noshuffle,
            wv_noshuffle: lb.wv_noshuffle,
            wo_noshuffle: lb.wo_noshuffle,
            w_gate_noshuffle: lb.w_gate_noshuffle,
            w_up_noshuffle: lb.w_up_noshuffle,
            w_down_noshuffle: lb.w_down_noshuffle,
        };
        // Route through the partition builder when this layer has a
        // `PartitionContext` attached AND the plan-path feature is enabled.
        let partition_ctx = config
            .partition_layers
            .as_ref()
            .and_then(|v| v.get(i))
            .copied()
            .flatten();
        let layer_plan =
            if let Some(p_ctx) = partition_ctx {
                let workspace = config.partition_workspace.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("partition workspace missing for layer {}", i)
                })?;
                let cpu_backend = config.partition_cpu_backend.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("partition cpu backend missing for layer {}", i)
                })?;
                let is_last = i + 1 == config.layer_bufs.len();
                build_partitioned_layer_plan(
                    &layer_config,
                    p_ctx,
                    workspace,
                    cpu_backend,
                    config.partition_use_gelu_tanh,
                    i,
                    is_last,
                )
                .with_context(|| format!("build partitioned plan for layer {}", i))?
            } else {
                build_layer_plan(&layer_config)
                    .with_context(|| format!("build plan for layer {}", i))?
            };
        layers.push(layer_plan);
    }

    // Only the last layer needs clFlush before final norm + lm_head
    if let Some(last) = layers.last_mut() {
        last.flush_after = true;
    }

    // Final RMSNorm (in-place on x)
    let final_norm = {
        let kernel_name = if config.dim % 4 == 0 {
            "kernel_rms_norm_opt_f4"
        } else {
            "kernel_rms_norm_opt"
        };
        let kernel = ocl::core::create_kernel(config.simple_ops_program, kernel_name)
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
        if let (Some(ns), Some(progs)) = (&config.lm_head_noshuffle, &config.noshuffle_programs) {
            let prog = progs
                .get(&ns.ne01)
                .context("noshuffle program for lm_head ne01")?;
            Some(
                make_q4_0_noshuffle_matmul_step(
                    prog,
                    config.context.as_core(),
                    ns.q_img,
                    ns.d_buf,
                    config.x_buf,
                    config.logits_buf,
                    ns.ne00,
                    ns.ne01,
                    OpTag::LmHead,
                )
                .context("build lm_head noshuffle matmul step")?,
            )
        } else {
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
        }
    } else {
        None
    };

    Ok(FullKernelPlan {
        layers,
        final_norm,
        lm_head,
        kv_capacity: config.kv_capacity,
        writes_gpu_scores: config.needs_attention_scores && config.gpu_score_buf.is_some(),
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
    /// AOS Q4_0 matmul program (`kernel_mul_mat_q4_0_f32`). Used for small-N
    /// matmuls where noshuffle GEMV under-utilizes the GPU (e.g. K/V projection).
    pub q4_0_program: &'a ocl::Program,
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

/// Build a KIVI scatter_residual_f16 kernel step for K or V.
/// Scatters F32 residual data into F16 attention buffer.
///
/// kivi_scatter_residual_f16 args:
///   residual(0), attn(1), kv_heads(2), res_cap(3), head_dim(4), res_pos(5), tok_base(6)
fn make_kivi_scatter_step(
    kivi_q2_program: &ocl::Program,
    residual_buf: &Mem,
    attn_buf: &Mem,
    kv_heads: usize,
    res_cap: usize,
    head_dim: usize,
) -> Result<KernelStep> {
    // F16 variant: residual is F32 input, attn output is F16 buffer
    let kernel = ocl::core::create_kernel(kivi_q2_program, "kivi_scatter_residual_f16")
        .context("create kivi_scatter_residual_f16")?;
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
        // KIVI assembled path does not collect scores, so score_layer_offset
        // is a no-op; pass 0.
        ocl::core::set_kernel_arg(&kernel, 14, ocl::core::ArgVal::scalar(&0i32))?;
        ocl::core::set_kernel_arg(
            &kernel,
            15,
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
        // Assembled: scatter F32 residual to F16 attn buffer, then F16 attention
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
            q4_0_program: config.q4_0_program,
            x_buf: config.x_buf,
            wq_buf: lb.wq,
            wk_buf: lb.wk,
            wv_buf: lb.wv,
            bq_buf: None,
            bk_buf: None,
            bv_buf: None,
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
            // KIVI replaces the standard attention step with its own variant,
            // so the flash gate never matters here. Force the legacy path
            // (which build_kivi_layer_plan immediately overrides anyway).
            flash_attn_f32_f16_program_dk64: None,
            flash_attn_f32_f16_program_dk128: None,
            needs_attention_scores: false,
            gpu_score_buf: None,
            gpu_score_stride: 0,
            gpu_score_layer_offset: 0,
            // KIVI currently only supports F16 weights — no noshuffle.
            noshuffle_programs: None,
            wq_noshuffle: None,
            wk_noshuffle: None,
            wv_noshuffle: None,
            wo_noshuffle: None,
            w_gate_noshuffle: None,
            w_up_noshuffle: None,
            w_down_noshuffle: None,
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
        // KIVI plans never bind a GPU score accumulator; scores are collected
        // on the CPU path when KIVI is active.
        writes_gpu_scores: false,
    })
}
