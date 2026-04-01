//! GPU-side attention score accumulator.
//!
//! Maintains persistent GPU buffers for importance scores and runs reduction
//! kernels entirely on the device, eliminating per-token GPU->CPU blocking
//! reads (~129ms/token). CPU readback occurs only at eviction time.
//!
//! # Workflow
//!
//! Per decode token:
//! 1. `attention_gen` writes post-softmax scores to `score_buf` (persistent GPU buffer)
//! 2. `reduce_layer()` runs `kernel_score_reduce` after each layer
//! 3. `end_step()` runs `kernel_score_end_step` + clears step buffers
//!
//! At eviction time:
//! 4. `sync_to_cpu()` does a single blocking readback of cumulative importance
//! 5. `reset()` clears cumulative buffers after eviction

use anyhow::Result;
use ocl::core::Kernel as CoreKernel;

/// GPU-resident score accumulator that avoids per-token GPU->CPU readback.
pub struct GpuScoreAccumulator {
    /// Persistent GPU buffer for attention scores output.
    /// Layout: `[n_heads_q * score_stride]` where score_stride >= max_seq_len.
    /// Reused by `attention_gen` kernel across all layers/tokens.
    score_buf: ocl::core::Mem,

    /// Step-local flat importance buffer: `[max_seq_len]`.
    /// Aggregates per-layer scores via MAX within a single decode step.
    step_flat: ocl::core::Mem,

    /// Step-local per-KV-head importance: `[n_kv_heads * max_seq_len]`.
    step_head: ocl::core::Mem,

    /// Cumulative flat importance: `[max_seq_len]`.
    /// Grows over decode steps with exponential decay.
    importance: ocl::core::Mem,

    /// Cumulative per-KV-head importance: `[n_kv_heads * max_seq_len]`.
    head_importance: ocl::core::Mem,

    // --- Cached kernels ---
    kernel_reduce: CoreKernel,
    kernel_end_step: CoreKernel,
    kernel_clear: CoreKernel,

    // --- Config ---
    n_heads_q: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
    /// score_stride for the score_buf layout. Equal to max_seq_len.
    score_stride: usize,
    /// Exponential decay factor: `1.0 - decay`. Applied to cumulative scores each step.
    decay_factor: f32,
    active: bool,
    steps_accumulated: usize,
}

// SAFETY: GpuScoreAccumulator is only accessed from the inference thread,
// same as KernelCache. Single-threaded access is guaranteed.
unsafe impl Send for GpuScoreAccumulator {}
unsafe impl Sync for GpuScoreAccumulator {}

impl GpuScoreAccumulator {
    /// Create a new GPU score accumulator.
    ///
    /// Compiles the score_reduce.cl kernels and allocates persistent GPU buffers.
    /// Returns an error if kernel compilation or buffer allocation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        queue: &ocl::core::CommandQueue,
        context: &ocl::core::Context,
        device: &ocl::core::DeviceId,
        cl_opts: &str,
        n_heads_q: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
        decay: f32,
    ) -> Result<Self> {
        let src = include_str!("../../../kernels/score_reduce.cl");
        let program = ocl::core::create_build_program(
            context,
            &[std::ffi::CString::new(src)?],
            Some(&[*device]),
            &std::ffi::CString::new(cl_opts)?,
        )
        .map_err(|e| anyhow::anyhow!("score_reduce.cl compilation failed: {}", e))?;

        let kernel_reduce = ocl::core::create_kernel(&program, "kernel_score_reduce")?;
        let kernel_end_step = ocl::core::create_kernel(&program, "kernel_score_end_step")?;
        let kernel_clear = ocl::core::create_kernel(&program, "kernel_score_clear")?;

        let score_stride = max_seq_len;
        let score_buf_size = n_heads_q * score_stride;
        let flat_size = max_seq_len;
        let head_size = n_kv_heads * max_seq_len;

        // Allocate GPU buffers
        let score_buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                context,
                ocl::core::MEM_READ_WRITE,
                score_buf_size,
                None,
            )?
        };

        let step_flat = unsafe {
            ocl::core::create_buffer::<_, f32>(context, ocl::core::MEM_READ_WRITE, flat_size, None)?
        };

        let step_head = unsafe {
            ocl::core::create_buffer::<_, f32>(context, ocl::core::MEM_READ_WRITE, head_size, None)?
        };

        let importance = unsafe {
            ocl::core::create_buffer::<_, f32>(context, ocl::core::MEM_READ_WRITE, flat_size, None)?
        };

        let head_importance = unsafe {
            ocl::core::create_buffer::<_, f32>(context, ocl::core::MEM_READ_WRITE, head_size, None)?
        };

        // Zero-initialize all buffers
        let zeros_flat = vec![0.0f32; flat_size];
        let zeros_head = vec![0.0f32; head_size];
        let zeros_score = vec![0.0f32; score_buf_size];
        unsafe {
            ocl::core::enqueue_write_buffer(
                queue,
                &step_flat,
                true,
                0,
                &zeros_flat,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
            ocl::core::enqueue_write_buffer(
                queue,
                &step_head,
                true,
                0,
                &zeros_head,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
            ocl::core::enqueue_write_buffer(
                queue,
                &importance,
                true,
                0,
                &zeros_flat,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
            ocl::core::enqueue_write_buffer(
                queue,
                &head_importance,
                true,
                0,
                &zeros_head,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
            ocl::core::enqueue_write_buffer(
                queue,
                &score_buf,
                true,
                0,
                &zeros_score,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }

        Ok(Self {
            score_buf,
            step_flat,
            step_head,
            importance,
            head_importance,
            kernel_reduce,
            kernel_end_step,
            kernel_clear,
            n_heads_q,
            n_kv_heads,
            max_seq_len,
            score_stride,
            decay_factor: (1.0 - decay).clamp(0.0, 1.0),
            active: false,
            steps_accumulated: 0,
        })
    }

    /// Get the persistent GPU score buffer for use by `attention_gen`.
    /// The attention kernel writes post-softmax scores here.
    #[inline]
    pub fn score_buf_mem(&self) -> &ocl::core::Mem {
        &self.score_buf
    }

    /// Score stride (= max_seq_len) for the score buffer layout.
    #[inline]
    pub fn score_stride(&self) -> usize {
        self.score_stride
    }

    /// Number of query heads.
    #[inline]
    pub fn n_heads_q(&self) -> usize {
        self.n_heads_q
    }

    /// Run the per-layer score reduction kernel.
    ///
    /// Called after each `attention_gen` call to aggregate scores from the
    /// persistent `score_buf` into step-local `step_flat` and `step_head`.
    pub fn reduce_layer(
        &self,
        queue: &ocl::core::CommandQueue,
        cache_seq_len: usize,
    ) -> Result<()> {
        if !self.active || cache_seq_len == 0 {
            return Ok(());
        }

        let kernel = &self.kernel_reduce;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(&self.score_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(&self.step_flat))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(&self.step_head))?;
            ocl::core::set_kernel_arg(
                kernel,
                3,
                ocl::core::ArgVal::scalar(&(self.n_heads_q as i32)),
            )?;
            ocl::core::set_kernel_arg(
                kernel,
                4,
                ocl::core::ArgVal::scalar(&(self.n_kv_heads as i32)),
            )?;
            ocl::core::set_kernel_arg(
                kernel,
                5,
                ocl::core::ArgVal::scalar(&(cache_seq_len as i32)),
            )?;
            ocl::core::set_kernel_arg(
                kernel,
                6,
                ocl::core::ArgVal::scalar(&(self.score_stride as i32)),
            )?;
            ocl::core::set_kernel_arg(
                kernel,
                7,
                ocl::core::ArgVal::scalar(&(self.max_seq_len as i32)),
            )?;

            let gws = [Self::round_up(cache_seq_len, 64), 1, 1];
            let lws = [64, 1, 1];

            ocl::core::enqueue_kernel(
                queue,
                kernel,
                1,
                None,
                &gws,
                Some(lws),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    /// End the current decode step.
    ///
    /// Flushes step-local importance into cumulative importance with decay,
    /// then clears step buffers for the next token.
    pub fn end_step(
        &mut self,
        queue: &ocl::core::CommandQueue,
        cache_seq_len: usize,
    ) -> Result<()> {
        if !self.active || cache_seq_len == 0 {
            return Ok(());
        }

        // 1. Flush step -> cumulative with decay
        let kernel = &self.kernel_end_step;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(&self.importance))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::mem(&self.step_flat))?;
            ocl::core::set_kernel_arg(kernel, 2, ocl::core::ArgVal::mem(&self.head_importance))?;
            ocl::core::set_kernel_arg(kernel, 3, ocl::core::ArgVal::mem(&self.step_head))?;
            ocl::core::set_kernel_arg(kernel, 4, ocl::core::ArgVal::scalar(&self.decay_factor))?;
            ocl::core::set_kernel_arg(
                kernel,
                5,
                ocl::core::ArgVal::scalar(&(self.n_kv_heads as i32)),
            )?;
            ocl::core::set_kernel_arg(
                kernel,
                6,
                ocl::core::ArgVal::scalar(&(cache_seq_len as i32)),
            )?;
            ocl::core::set_kernel_arg(
                kernel,
                7,
                ocl::core::ArgVal::scalar(&(self.max_seq_len as i32)),
            )?;

            let gws = [Self::round_up(cache_seq_len, 64), 1, 1];
            let lws = [64, 1, 1];

            ocl::core::enqueue_kernel(
                queue,
                kernel,
                1,
                None,
                &gws,
                Some(lws),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }

        // 2. Clear step buffers for next token
        self.clear_step_buffers(queue)?;

        self.steps_accumulated += 1;
        Ok(())
    }

    /// Read cumulative importance scores back to CPU.
    ///
    /// Returns `(flat_importance, head_importance)`.
    /// This is the only blocking GPU->CPU read and should be called only at eviction time.
    pub fn sync_to_cpu(&self, queue: &ocl::core::CommandQueue) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut flat = vec![0.0f32; self.max_seq_len];
        let mut head = vec![0.0f32; self.n_kv_heads * self.max_seq_len];

        unsafe {
            ocl::core::enqueue_read_buffer(
                queue,
                &self.importance,
                true,
                0,
                &mut flat,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
            ocl::core::enqueue_read_buffer(
                queue,
                &self.head_importance,
                true,
                0,
                &mut head,
                None::<ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }

        Ok((flat, head))
    }

    /// Reset cumulative importance (after eviction).
    pub fn reset(&mut self, queue: &ocl::core::CommandQueue) -> Result<()> {
        let flat_size = self.max_seq_len;
        let head_size = self.n_kv_heads * self.max_seq_len;

        self.clear_buffer(queue, &self.importance, flat_size)?;
        self.clear_buffer(queue, &self.head_importance, head_size)?;
        self.clear_step_buffers(queue)?;
        self.steps_accumulated = 0;
        Ok(())
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Number of steps accumulated since last reset.
    #[allow(dead_code)]
    pub fn steps_accumulated(&self) -> usize {
        self.steps_accumulated
    }

    // --- Internal helpers ---

    fn clear_step_buffers(&self, queue: &ocl::core::CommandQueue) -> Result<()> {
        let flat_size = self.max_seq_len;
        let head_size = self.n_kv_heads * self.max_seq_len;
        self.clear_buffer(queue, &self.step_flat, flat_size)?;
        self.clear_buffer(queue, &self.step_head, head_size)?;
        Ok(())
    }

    fn clear_buffer(
        &self,
        queue: &ocl::core::CommandQueue,
        buf: &ocl::core::Mem,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let kernel = &self.kernel_clear;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ocl::core::ArgVal::mem(buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ocl::core::ArgVal::scalar(&(n as i32)))?;

            let gws = [Self::round_up(n, 256), 1, 1];
            let lws = [256, 1, 1];

            ocl::core::enqueue_kernel(
                queue,
                kernel,
                1,
                None,
                &gws,
                Some(lws),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    #[inline]
    fn round_up(n: usize, multiple: usize) -> usize {
        n.div_ceil(multiple) * multiple
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_up() {
        assert_eq!(GpuScoreAccumulator::round_up(1, 64), 64);
        assert_eq!(GpuScoreAccumulator::round_up(64, 64), 64);
        assert_eq!(GpuScoreAccumulator::round_up(65, 64), 128);
        assert_eq!(GpuScoreAccumulator::round_up(0, 64), 0);
    }

    #[test]
    fn test_decay_factor_clamping() {
        // decay=0.0 -> factor=1.0 (no decay)
        assert!((1.0f32 - 0.0f32).clamp(0.0, 1.0) - 1.0 < f32::EPSILON);
        // decay=0.5 -> factor=0.5
        assert!((1.0f32 - 0.5f32).clamp(0.0, 1.0) - 0.5 < f32::EPSILON);
        // decay=1.0 -> factor=0.0 (full decay)
        assert!((1.0f32 - 1.0f32).clamp(0.0, 1.0) < f32::EPSILON);
    }
}
