use crate::buffer::slice_buffer::SliceBuffer;
use crate::core::backend::Backend;
use crate::core::buffer::DType;
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use anyhow::{Result, ensure};
use log::debug;
use std::sync::Arc;

/// Row alignment granularity for weight splits.
/// 128 rows ensures alignment with GPU workgroup sizes and Q4_0 block boundaries.
const ROW_ALIGNMENT: usize = 128;

/// GPU-only fast-path threshold for `gpu_ratio`.
///
/// When `gpu_ratio >= GPU_ONLY_THRESHOLD`, the partition activation path is
/// skipped entirely: `partition_ctx` stays `None`, and the forward pass uses
/// the dense full-weight matmul on GPU. This avoids the ratio-independent
/// constant overhead of the partition path (host staging `read_buffer`,
/// CPU matmul kick-off for a clamped 128-row slice, and GPU<->host merge).
///
/// The rationale: `split_weight` clamps `split_row` to `[128, out_dim - 128]`
/// (Q4_0 alignment requirement). A caller passing `ratio=0.999` still ends up
/// with 128 CPU rows per weight, forcing the partition dispatch path every
/// token even though the split is effectively GPU-only. This threshold makes
/// that corner case behave as the user intends.
///
/// 0.995 is chosen empirically:
///   - For Llama 3.2 1B `ffn_hidden=8192`, 0.995 → CPU gets at most 128 rows
///     (1.6% of output), which is already the clamp minimum. Anything above
///     0.995 is structurally equivalent to "clamp min CPU rows" for any sane
///     FFN size, so skipping the partition path is information-preserving.
///   - 1.0 is considered GPU-only by design (`generate.rs` CLI gate is
///     `tensor_partition > 0.0 && tensor_partition < 1.0`).
pub const GPU_ONLY_THRESHOLD: f32 = 0.995;

/// Returns true when the given ratio should take the GPU-only fast path
/// (no partition context installed, no per-token host staging).
pub fn is_gpu_only_ratio(gpu_ratio: f32) -> bool {
    gpu_ratio >= GPU_ONLY_THRESHOLD
}

/// A weight tensor split row-wise into GPU and CPU partitions.
///
/// For a weight `W [out_dim, in_dim]`, the split produces:
///   - `gpu_slice`: `W[0..split_row, :]`   (processed by GPU)
///   - `cpu_slice`: `W[split_row.., :]`     (processed by CPU)
///
/// Each slice owns an independent buffer (pre-copied from the original weight).
/// GPU slices have a valid `cl_mem` handle; CPU slices have valid host pointers.
pub struct PartitionedWeight {
    pub gpu_slice: Tensor,
    pub cpu_slice: Tensor,
    pub split_row: usize,
}

/// Per-layer partition context holding CPU backend and partitioned weights.
///
/// Only FFN gate/up projections are partitioned; attention (wq/wk/wv/wo) and
/// FFN down are always executed on the GPU to avoid merge overhead.
pub struct PartitionContext {
    pub gpu_ratio: f32,
    pub cpu_backend: Arc<dyn Backend>,
    // FFN gate/up projections (always large enough for partitioning)
    pub gate: PartitionedWeight,
    pub up: PartitionedWeight,
}

/// Compute the number of bytes per row for a given dtype and inner dimension.
///
/// For Q4_0: each row of `in_dim` elements = `in_dim / 32` blocks, each 18 bytes.
/// For Q4_1: each row of `in_dim` elements = `in_dim / 32` blocks, each 20 bytes.
/// For F16/BF16: each row = `in_dim * 2` bytes.
/// For F32: each row = `in_dim * 4` bytes.
fn bytes_per_row(dtype: DType, in_dim: usize) -> Result<usize> {
    match dtype {
        DType::Q4_0 => {
            ensure!(
                in_dim.is_multiple_of(32),
                "Q4_0 requires in_dim divisible by 32, got {}",
                in_dim
            );
            Ok(in_dim / 32 * 18)
        }
        DType::Q4_1 => {
            ensure!(
                in_dim.is_multiple_of(32),
                "Q4_1 requires in_dim divisible by 32, got {}",
                in_dim
            );
            Ok(in_dim / 32 * 20)
        }
        DType::Q8_0 => {
            ensure!(
                in_dim.is_multiple_of(32),
                "Q8_0 requires in_dim divisible by 32, got {}",
                in_dim
            );
            Ok(in_dim / 32 * 34)
        }
        DType::F16 | DType::BF16 => Ok(in_dim * 2),
        DType::F32 => Ok(in_dim * 4),
        DType::U8 => Ok(in_dim),
    }
}

/// Round `value` down to the nearest multiple of `align`.
fn align_down(value: usize, align: usize) -> usize {
    (value / align) * align
}

/// Split a weight tensor row-wise for CPU-GPU cooperative inference.
///
/// Given a weight `W [out_dim, in_dim]` and a `gpu_ratio` (0.0 to 1.0),
/// produces a `PartitionedWeight` with:
///   - `gpu_slice`: rows `[0, split_row)` — independent GPU buffer (copied from original)
///   - `cpu_slice`: rows `[split_row, out_dim)` — independent CPU buffer (copied)
///
/// `split_row` is aligned to `ROW_ALIGNMENT` (128), clamped to `[128, out_dim - 128]`.
///
/// Each slice owns its own buffer via `Backend::copy_from()`, ensuring that
/// GPU slices have a valid `cl_mem` handle and CPU slices have a valid host pointer.
/// This avoids `SliceBuffer`'s inability to provide `cl_mem` for sub-regions.
pub fn split_weight(
    weight: &Tensor,
    gpu_ratio: f32,
    cpu_backend: &Arc<dyn Backend>,
) -> Result<PartitionedWeight> {
    let shape = weight.shape();
    let dims = shape.dims();
    ensure!(
        dims.len() == 2,
        "split_weight expects 2D weight [out_dim, in_dim], got {:?}",
        dims
    );

    let out_dim = dims[0];
    let in_dim = dims[1];
    let dtype = weight.dtype();
    let bpr = bytes_per_row(dtype, in_dim)?;

    ensure!(
        out_dim >= ROW_ALIGNMENT * 2,
        "out_dim ({}) must be >= {} for partitioning",
        out_dim,
        ROW_ALIGNMENT * 2
    );

    // Compute split_row with alignment and clamping.
    let raw_split = (out_dim as f32 * gpu_ratio) as usize;
    let split_row = align_down(raw_split, ROW_ALIGNMENT)
        .max(ROW_ALIGNMENT)
        .min(out_dim - ROW_ALIGNMENT);

    let gpu_bytes = split_row * bpr;
    let cpu_bytes = (out_dim - split_row) * bpr;

    // Verify total matches weight buffer size.
    ensure!(
        gpu_bytes + cpu_bytes == weight.size(),
        "Byte split mismatch: gpu({}) + cpu({}) = {} != weight.size({})",
        gpu_bytes,
        cpu_bytes,
        gpu_bytes + cpu_bytes,
        weight.size()
    );

    let parent_buf = weight.buffer().clone();

    // GPU slice: zero-copy sub-buffer if parent has cl_mem, else fallback to copy.
    #[cfg(feature = "opencl")]
    let gpu_tensor = {
        use crate::buffer::cl_sub_buffer::ClSubBuffer;
        match ClSubBuffer::new(parent_buf.clone(), 0, gpu_bytes, dtype) {
            Ok(sub_buf) => Tensor::new(
                Shape::new(vec![split_row, in_dim]),
                Arc::new(sub_buf),
                weight.backend().clone(),
            ),
            Err(_) => {
                // Fallback: parent has no cl_mem (CPU-only backend). Copy as before.
                let tmp_gpu_buf =
                    Arc::new(SliceBuffer::new(parent_buf.clone(), 0, gpu_bytes, dtype)?);
                let tmp_gpu_tensor = Tensor::new(
                    Shape::new(vec![split_row, in_dim]),
                    tmp_gpu_buf,
                    weight.backend().clone(),
                );
                let copied = weight.backend().copy_from(&tmp_gpu_tensor)?;
                Tensor::new(
                    Shape::new(vec![split_row, in_dim]),
                    copied.buffer().clone(),
                    weight.backend().clone(),
                )
            }
        }
    };
    #[cfg(not(feature = "opencl"))]
    let gpu_tensor = {
        let tmp_gpu_buf = Arc::new(SliceBuffer::new(parent_buf.clone(), 0, gpu_bytes, dtype)?);
        let tmp_gpu_tensor = Tensor::new(
            Shape::new(vec![split_row, in_dim]),
            tmp_gpu_buf,
            weight.backend().clone(),
        );
        let copied = weight.backend().copy_from(&tmp_gpu_tensor)?;
        Tensor::new(
            Shape::new(vec![split_row, in_dim]),
            copied.buffer().clone(),
            weight.backend().clone(),
        )
    };

    // CPU slice: zero-copy SliceBuffer (cl_mem not needed for CPU matmul).
    let cpu_tensor = Tensor::new(
        Shape::new(vec![out_dim - split_row, in_dim]),
        Arc::new(SliceBuffer::new(parent_buf, gpu_bytes, cpu_bytes, dtype)?),
        cpu_backend.clone(),
    );

    debug!(
        "split_weight: [{}x{}] split_row={}, gpu_bytes={}, cpu_bytes={} (pre-copy)",
        out_dim, in_dim, split_row, gpu_bytes, cpu_bytes,
    );

    Ok(PartitionedWeight {
        gpu_slice: gpu_tensor,
        cpu_slice: cpu_tensor,
        split_row,
    })
}

/// Merge 2D partial results from GPU and CPU partitions (prefill path).
///
/// GPU partial: `[batch, seq_len, split_row]` (F32, on GPU)
/// CPU partial: `[batch, seq_len, cpu_rows]` (F32, on CPU)
/// Output:      `[batch, seq_len, out_dim]`  (F32, on GPU) where out_dim = split_row + cpu_rows
///
/// Strategy (approach C): read GPU partial to CPU temp, interleave rows, write to output.
/// Prefill is bandwidth-bound, so the extra memcpy overhead is acceptable.
pub fn merge_partials_2d(
    backend: &dyn Backend,
    gpu_partial: &Tensor,
    cpu_partial: &Tensor,
    output: &mut Tensor,
    total_rows: usize, // batch_size * seq_len
    split_row: usize,
    cpu_rows: usize,
) -> Result<()> {
    let out_dim = split_row + cpu_rows;

    // 1. Read GPU partial to CPU temp buffer
    let gpu_bytes = total_rows * split_row * 4;
    let mut gpu_temp = vec![0u8; gpu_bytes];
    backend.read_buffer(gpu_partial, &mut gpu_temp)?;

    // 2. Interleave: build merged [total_rows, out_dim] on CPU
    let out_bytes = total_rows * out_dim * 4;
    let mut merged = vec![0u8; out_bytes];

    // Safety: cpu_partial is a CPU tensor with valid host pointer.
    let cpu_data =
        unsafe { std::slice::from_raw_parts(cpu_partial.as_ptr(), total_rows * cpu_rows * 4) };

    for s in 0..total_rows {
        let gpu_row_start = s * split_row * 4;
        let cpu_row_start = s * cpu_rows * 4;
        let out_row_start = s * out_dim * 4;

        // GPU columns [0..split_row)
        merged[out_row_start..out_row_start + split_row * 4]
            .copy_from_slice(&gpu_temp[gpu_row_start..gpu_row_start + split_row * 4]);

        // CPU columns [split_row..out_dim)
        merged[out_row_start + split_row * 4..out_row_start + out_dim * 4]
            .copy_from_slice(&cpu_data[cpu_row_start..cpu_row_start + cpu_rows * 4]);
    }

    // 3. Write merged result to GPU output
    backend.write_buffer(output, &merged)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::core::memory::Memory;
    use crate::core::quant::{BlockQ4_0, QK4_0};
    use crate::memory::galloc::Galloc;

    /// Helper: create a CPU backend Arc.
    fn cpu_backend() -> Arc<dyn Backend> {
        Arc::new(CpuBackend::new())
    }

    /// Helper: allocate an F32 weight tensor [out_dim, in_dim] with sequential values.
    fn make_f32_weight(out_dim: usize, in_dim: usize) -> Tensor {
        let memory = Galloc::new();
        let size = out_dim * in_dim * 4;
        let buf = memory.alloc(size, DType::F32).unwrap();
        let mut tensor = Tensor::new(Shape::new(vec![out_dim, in_dim]), buf, cpu_backend());
        let data = tensor.as_mut_slice::<f32>();
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f32 * 0.001;
        }
        tensor
    }

    /// Helper: allocate a Q4_0 weight tensor [out_dim, in_dim].
    /// Fills with quantized data from sequential float values.
    fn make_q4_weight(out_dim: usize, in_dim: usize) -> Tensor {
        assert!(
            in_dim % QK4_0 == 0,
            "in_dim must be divisible by 32 for Q4_0"
        );
        let blocks_per_row = in_dim / QK4_0;
        let total_blocks = out_dim * blocks_per_row;
        let total_bytes = total_blocks * std::mem::size_of::<BlockQ4_0>();

        let memory = Galloc::new();
        let buf = memory.alloc(total_bytes, DType::Q4_0).unwrap();
        let mut tensor = Tensor::new(Shape::new(vec![out_dim, in_dim]), buf, cpu_backend());

        // Fill with quantized data
        let blocks = unsafe {
            std::slice::from_raw_parts_mut(tensor.as_mut_ptr() as *mut BlockQ4_0, total_blocks)
        };
        for (bi, block) in blocks.iter_mut().enumerate() {
            let mut src = [0.0f32; QK4_0];
            for (j, v) in src.iter_mut().enumerate() {
                *v = ((bi * QK4_0 + j) as f32) * 0.01;
            }
            *block = BlockQ4_0::quantize(&src);
        }
        tensor
    }

    // PA-T1-FASTPATH: GPU-only fast-path threshold classifier is correct.
    // Ratios at or above `GPU_ONLY_THRESHOLD` must take the fast path
    // (partition_ctx = None). Ratios strictly below must NOT (real split).
    #[test]
    fn test_gpu_only_fast_path_classifier() {
        // At threshold: fast path.
        assert!(is_gpu_only_ratio(GPU_ONLY_THRESHOLD));
        // Above: fast path.
        assert!(is_gpu_only_ratio(0.999));
        assert!(is_gpu_only_ratio(1.0));
        // Below: partition path.
        assert!(!is_gpu_only_ratio(0.99));
        assert!(!is_gpu_only_ratio(0.75));
        assert!(!is_gpu_only_ratio(0.5));
        assert!(!is_gpu_only_ratio(0.001));
        assert!(!is_gpu_only_ratio(0.0));
    }

    // PA-T1-06: Q4_0 alignment — split_row is a multiple of 128.
    #[test]
    fn test_q4_alignment() {
        let w = make_q4_weight(5504, 1536);
        let cpu = cpu_backend();
        let pw = split_weight(&w, 0.5, &cpu).unwrap();
        assert_eq!(pw.split_row % ROW_ALIGNMENT, 0);
        assert!(pw.split_row >= ROW_ALIGNMENT);
        assert!(pw.split_row <= 5504 - ROW_ALIGNMENT);
    }

    // PA-T1-07: F16 split total — gpu + cpu rows == out_dim.
    // We use F32 here because F16 allocation is identical in shape logic.
    #[test]
    fn test_split_total_rows() {
        let w = make_f32_weight(5504, 1536);
        let cpu = cpu_backend();
        let pw = split_weight(&w, 0.7, &cpu).unwrap();
        let gpu_rows = pw.gpu_slice.shape().dims()[0];
        let cpu_rows = pw.cpu_slice.shape().dims()[0];
        assert_eq!(gpu_rows + cpu_rows, 5504);
        assert_eq!(gpu_rows, pw.split_row);
    }

    // PA-T1-08: split -> concat roundtrip for Q4_0.
    // Verifies that gpu_slice bytes + cpu_slice bytes == original weight bytes.
    #[test]
    fn test_q4_split_concat_roundtrip() {
        let w = make_q4_weight(5504, 1536);
        let cpu = cpu_backend();
        let pw = split_weight(&w, 0.6, &cpu).unwrap();

        let gpu_size = pw.gpu_slice.size();
        let cpu_size = pw.cpu_slice.size();
        assert_eq!(gpu_size + cpu_size, w.size());

        // Verify data content matches original weight bytes (pre-copy creates independent buffers).
        let orig = unsafe { std::slice::from_raw_parts(w.as_ptr(), w.size()) };
        let gpu_data = unsafe { std::slice::from_raw_parts(pw.gpu_slice.as_ptr(), gpu_size) };
        let cpu_data = unsafe { std::slice::from_raw_parts(pw.cpu_slice.as_ptr(), cpu_size) };
        assert_eq!(gpu_data, &orig[..gpu_size]);
        assert_eq!(cpu_data, &orig[gpu_size..]);
    }

    // PA-T1-09: split -> concat roundtrip for F32.
    #[test]
    fn test_f32_split_concat_roundtrip() {
        let w = make_f32_weight(1024, 512);
        let cpu = cpu_backend();
        let pw = split_weight(&w, 0.5, &cpu).unwrap();

        assert_eq!(pw.gpu_slice.size() + pw.cpu_slice.size(), w.size());

        // Verify data integrity: read a value from the cpu_slice region
        let split_row = pw.split_row;
        let in_dim = 512;
        let expected_idx = split_row * in_dim; // first element of cpu region
        let expected_val = expected_idx as f32 * 0.001;
        let cpu_data = pw.cpu_slice.as_slice::<f32>();
        assert!(
            (cpu_data[0] - expected_val).abs() < 1e-6,
            "expected {}, got {}",
            expected_val,
            cpu_data[0]
        );
    }

    // PA-T1-10: partitioned matmul F32 accuracy (CPU-only).
    // Compare full matmul vs partitioned (gpu_part + cpu_part) concat.
    #[test]
    fn test_partitioned_matmul_f32() {
        let backend = cpu_backend();
        let memory = Galloc::new();

        let out_dim = 512;
        let in_dim = 256;

        // Weight [out_dim, in_dim]
        let w = make_f32_weight(out_dim, in_dim);

        // Input x [1, 1, in_dim]
        let x_buf = memory.alloc(in_dim * 4, DType::F32).unwrap();
        let mut x = Tensor::new(Shape::new(vec![1, 1, in_dim]), x_buf, backend.clone());
        let x_data = x.as_mut_slice::<f32>();
        for (i, v) in x_data.iter_mut().enumerate() {
            *v = (i as f32 + 1.0) * 0.01;
        }

        // Full matmul: out_full = x * W^T => [1, 1, out_dim]
        let out_buf = memory.alloc(out_dim * 4, DType::F32).unwrap();
        let mut out_full = Tensor::new(Shape::new(vec![1, 1, out_dim]), out_buf, backend.clone());
        backend.matmul_transposed(&x, &w, &mut out_full).unwrap();

        // Partitioned matmul
        let cpu2 = cpu_backend(); // second "cpu" acting as cpu_backend
        let pw = split_weight(&w, 0.5, &cpu2).unwrap();
        let split_row = pw.split_row;

        // GPU part: out_gpu = x * W_gpu^T => [1, 1, split_row]
        let gpu_buf = memory.alloc(split_row * 4, DType::F32).unwrap();
        let mut out_gpu = Tensor::new(Shape::new(vec![1, 1, split_row]), gpu_buf, backend.clone());
        backend
            .matmul_transposed(&x, &pw.gpu_slice, &mut out_gpu)
            .unwrap();

        // CPU part: out_cpu = x * W_cpu^T => [1, 1, out_dim - split_row]
        let cpu_rows = out_dim - split_row;
        let cpu_buf = memory.alloc(cpu_rows * 4, DType::F32).unwrap();
        let mut out_cpu = Tensor::new(Shape::new(vec![1, 1, cpu_rows]), cpu_buf, cpu2.clone());
        cpu2.matmul_transposed(&x, &pw.cpu_slice, &mut out_cpu)
            .unwrap();

        // Compare: concat(out_gpu, out_cpu) == out_full
        let full_data = out_full.as_slice::<f32>();
        let gpu_data = out_gpu.as_slice::<f32>();
        let cpu_data = out_cpu.as_slice::<f32>();

        // Tolerance: relative error < 1e-4 or absolute error < 0.01.
        // F32 matmul with parallel summation can produce different rounding order.
        for i in 0..split_row {
            let diff = (full_data[i] - gpu_data[i]).abs();
            let rel = if full_data[i].abs() > 1e-6 {
                diff / full_data[i].abs()
            } else {
                diff
            };
            assert!(
                rel < 1e-4 || diff < 0.01,
                "GPU mismatch at [{}]: full={}, gpu={}, diff={}, rel={}",
                i,
                full_data[i],
                gpu_data[i],
                diff,
                rel,
            );
        }
        for i in 0..cpu_rows {
            let diff = (full_data[split_row + i] - cpu_data[i]).abs();
            let rel = if full_data[split_row + i].abs() > 1e-6 {
                diff / full_data[split_row + i].abs()
            } else {
                diff
            };
            assert!(
                rel < 1e-4 || diff < 0.01,
                "CPU mismatch at [{}]: full={}, cpu={}, diff={}, rel={}",
                i,
                full_data[split_row + i],
                cpu_data[i],
                diff,
                rel,
            );
        }
    }

    // PA-T1-11: partitioned matmul Q4_0 accuracy (CPU-only).
    #[test]
    fn test_partitioned_matmul_q4_0() {
        let backend = cpu_backend();
        let memory = Galloc::new();

        let out_dim = 512;
        let in_dim = 256; // divisible by 32

        let w = make_q4_weight(out_dim, in_dim);

        // Input x [1, 1, in_dim] F32
        let x_buf = memory.alloc(in_dim * 4, DType::F32).unwrap();
        let mut x = Tensor::new(Shape::new(vec![1, 1, in_dim]), x_buf, backend.clone());
        let x_data = x.as_mut_slice::<f32>();
        for (i, v) in x_data.iter_mut().enumerate() {
            *v = (i as f32 + 1.0) * 0.01;
        }

        // Full matmul
        let out_buf = memory.alloc(out_dim * 4, DType::F32).unwrap();
        let mut out_full = Tensor::new(Shape::new(vec![1, 1, out_dim]), out_buf, backend.clone());
        backend.matmul_transposed(&x, &w, &mut out_full).unwrap();

        // Partitioned
        let cpu2 = cpu_backend();
        let pw = split_weight(&w, 0.5, &cpu2).unwrap();
        let split_row = pw.split_row;
        let cpu_rows = out_dim - split_row;

        let gpu_buf = memory.alloc(split_row * 4, DType::F32).unwrap();
        let mut out_gpu = Tensor::new(Shape::new(vec![1, 1, split_row]), gpu_buf, backend.clone());
        backend
            .matmul_transposed(&x, &pw.gpu_slice, &mut out_gpu)
            .unwrap();

        let cpu_buf = memory.alloc(cpu_rows * 4, DType::F32).unwrap();
        let mut out_cpu = Tensor::new(Shape::new(vec![1, 1, cpu_rows]), cpu_buf, cpu2.clone());
        cpu2.matmul_transposed(&x, &pw.cpu_slice, &mut out_cpu)
            .unwrap();

        // Compare
        let full_data = out_full.as_slice::<f32>();
        let gpu_data = out_gpu.as_slice::<f32>();
        let cpu_data = out_cpu.as_slice::<f32>();

        // Q4_0 accumulates more rounding error from quantized dot products.
        // Use relative tolerance of 1e-3 or absolute tolerance of 1.0.
        for i in 0..split_row {
            let diff = (full_data[i] - gpu_data[i]).abs();
            let rel = if full_data[i].abs() > 1e-6 {
                diff / full_data[i].abs()
            } else {
                diff
            };
            assert!(
                rel < 1e-3 || diff < 1.0,
                "Q4_0 GPU mismatch at [{}]: full={}, gpu={}, diff={}, rel={}",
                i,
                full_data[i],
                gpu_data[i],
                diff,
                rel,
            );
        }
        for i in 0..cpu_rows {
            let diff = (full_data[split_row + i] - cpu_data[i]).abs();
            let rel = if full_data[split_row + i].abs() > 1e-6 {
                diff / full_data[split_row + i].abs()
            } else {
                diff
            };
            assert!(
                rel < 1e-3 || diff < 1.0,
                "Q4_0 CPU mismatch at [{}]: full={}, cpu={}, diff={}, rel={}",
                i,
                full_data[split_row + i],
                cpu_data[i],
                diff,
                rel,
            );
        }
    }

    // PA-T1-12: workspace size calculation for partitioned output.
    #[test]
    fn test_workspace_sizes() {
        let w = make_f32_weight(5504, 1536);
        let cpu = cpu_backend();
        let pw = split_weight(&w, 0.6, &cpu).unwrap();

        let split_row = pw.split_row;
        let cpu_rows = 5504 - split_row;

        // GPU output workspace: split_row * sizeof(f32)
        let gpu_ws_bytes = split_row * 4;
        // CPU output workspace: cpu_rows * sizeof(f32)
        let cpu_ws_bytes = cpu_rows * 4;
        // Total should equal full output size
        assert_eq!(gpu_ws_bytes + cpu_ws_bytes, 5504 * 4);
    }

    // PA-T1-13: extreme ratios (very low and very high).
    #[test]
    fn test_extreme_ratios() {
        let w = make_f32_weight(1024, 256);
        let cpu = cpu_backend();

        // Very low ratio: should clamp split_row to ROW_ALIGNMENT (128)
        let pw_low = split_weight(&w, 0.01, &cpu).unwrap();
        assert_eq!(pw_low.split_row, ROW_ALIGNMENT);
        assert_eq!(
            pw_low.gpu_slice.shape().dims()[0] + pw_low.cpu_slice.shape().dims()[0],
            1024
        );

        // Very high ratio: should clamp split_row to out_dim - ROW_ALIGNMENT
        let pw_high = split_weight(&w, 0.99, &cpu).unwrap();
        assert_eq!(pw_high.split_row, 1024 - ROW_ALIGNMENT);
        assert_eq!(
            pw_high.gpu_slice.shape().dims()[0] + pw_high.cpu_slice.shape().dims()[0],
            1024
        );
    }

    // Edge case: out_dim exactly 256 (minimum viable).
    #[test]
    fn test_minimum_out_dim() {
        let w = make_f32_weight(256, 128);
        let cpu = cpu_backend();
        let pw = split_weight(&w, 0.5, &cpu).unwrap();
        assert_eq!(pw.split_row, 128);
        assert_eq!(pw.gpu_slice.shape().dims(), &[128, 128]);
        assert_eq!(pw.cpu_slice.shape().dims(), &[128, 128]);
    }

    // Error case: out_dim too small for partitioning.
    #[test]
    fn test_too_small_out_dim() {
        let memory = Galloc::new();
        let buf = memory.alloc(128 * 64 * 4, DType::F32).unwrap();
        let w = Tensor::new(Shape::new(vec![128, 64]), buf, cpu_backend());
        let cpu = cpu_backend();
        let result = split_weight(&w, 0.5, &cpu);
        assert!(result.is_err());
    }

    // Error case: 1D tensor.
    #[test]
    fn test_1d_tensor_error() {
        let memory = Galloc::new();
        let buf = memory.alloc(1024 * 4, DType::F32).unwrap();
        let w = Tensor::new(Shape::new(vec![1024]), buf, cpu_backend());
        let cpu = cpu_backend();
        let result = split_weight(&w, 0.5, &cpu);
        assert!(result.is_err());
    }

    // ── merge_partials_2d tests ──

    /// Verify that merge_partials_2d correctly interleaves GPU and CPU partials.
    #[test]
    fn test_merge_partials_2d_basic() {
        let backend = cpu_backend();
        let memory = Galloc::new();

        let seq_len = 4;
        let split_row = 3;
        let cpu_rows = 2;
        let out_dim = split_row + cpu_rows;

        // GPU partial: [seq_len, split_row], values 100+
        let gpu_buf = memory.alloc(seq_len * split_row * 4, DType::F32).unwrap();
        let mut gpu_partial = Tensor::new(
            Shape::new(vec![seq_len, split_row]),
            gpu_buf,
            backend.clone(),
        );
        let gpu_data = gpu_partial.as_mut_slice::<f32>();
        for (i, v) in gpu_data.iter_mut().enumerate() {
            *v = 100.0 + i as f32;
        }

        // CPU partial: [seq_len, cpu_rows], values 200+
        let cpu_buf = memory.alloc(seq_len * cpu_rows * 4, DType::F32).unwrap();
        let mut cpu_partial = Tensor::new(
            Shape::new(vec![seq_len, cpu_rows]),
            cpu_buf,
            backend.clone(),
        );
        let cpu_data_w = cpu_partial.as_mut_slice::<f32>();
        for (i, v) in cpu_data_w.iter_mut().enumerate() {
            *v = 200.0 + i as f32;
        }

        // Output: [seq_len, out_dim]
        let out_buf = memory.alloc(seq_len * out_dim * 4, DType::F32).unwrap();
        let mut output = Tensor::new(Shape::new(vec![seq_len, out_dim]), out_buf, backend.clone());

        super::merge_partials_2d(
            backend.as_ref(),
            &gpu_partial,
            &cpu_partial,
            &mut output,
            seq_len,
            split_row,
            cpu_rows,
        )
        .unwrap();

        let result = output.as_slice::<f32>();
        for s in 0..seq_len {
            // GPU part
            for c in 0..split_row {
                let expected = 100.0 + (s * split_row + c) as f32;
                let actual = result[s * out_dim + c];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "row {} col {}: expected {}, got {}",
                    s,
                    c,
                    expected,
                    actual,
                );
            }
            // CPU part
            for c in 0..cpu_rows {
                let expected = 200.0 + (s * cpu_rows + c) as f32;
                let actual = result[s * out_dim + split_row + c];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "row {} col {}: expected {}, got {}",
                    s,
                    split_row + c,
                    expected,
                    actual,
                );
            }
        }
    }

    /// Verify merge_partials_2d with single-row input (degenerate case matching decode).
    #[test]
    fn test_merge_partials_2d_single_row() {
        let backend = cpu_backend();
        let memory = Galloc::new();

        let split_row = 4;
        let cpu_rows = 3;
        let out_dim = split_row + cpu_rows;

        let gpu_buf = memory.alloc(split_row * 4, DType::F32).unwrap();
        let mut gpu_partial = Tensor::new(Shape::new(vec![1, split_row]), gpu_buf, backend.clone());
        gpu_partial
            .as_mut_slice::<f32>()
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let cpu_buf = memory.alloc(cpu_rows * 4, DType::F32).unwrap();
        let mut cpu_partial = Tensor::new(Shape::new(vec![1, cpu_rows]), cpu_buf, backend.clone());
        cpu_partial
            .as_mut_slice::<f32>()
            .copy_from_slice(&[5.0, 6.0, 7.0]);

        let out_buf = memory.alloc(out_dim * 4, DType::F32).unwrap();
        let mut output = Tensor::new(Shape::new(vec![1, out_dim]), out_buf, backend.clone());

        super::merge_partials_2d(
            backend.as_ref(),
            &gpu_partial,
            &cpu_partial,
            &mut output,
            1,
            split_row,
            cpu_rows,
        )
        .unwrap();

        let result = output.as_slice::<f32>();
        assert_eq!(result, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }
}
