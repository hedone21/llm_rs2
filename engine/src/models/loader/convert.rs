//! Dtype conversion helpers for model weight loading.
//!
//! These functions convert raw safetensors byte slices into the target dtype
//! (F16, F32, Q4_0). Extracted from `transformer.rs::load_with_dtype` so they
//! can be reused by any `TensorSource` implementation.

use crate::core::quant::{BlockQ4_0, QK4_0};
use half::f16;

/// Convert BF16 raw bytes to F16 values in a pre-allocated destination buffer.
///
/// # Safety
/// `src` must contain `num_elements * 2` bytes of valid BF16 data.
/// `dst` must have at least `num_elements` entries.
pub fn bf16_to_f16_buf(src: &[u8], dst: &mut [f16], num_elements: usize) {
    let src_u16 = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, num_elements) };
    for (i, &b) in src_u16.iter().enumerate() {
        dst[i] = f16::from_f32(half::bf16::from_bits(b).to_f32());
    }
}

/// Convert F32 raw bytes to F16 values in a pre-allocated destination buffer.
///
/// # Safety
/// `src` must contain `num_elements * 4` bytes of valid F32 data.
/// `dst` must have at least `num_elements` entries.
pub fn f32_to_f16_buf(src: &[u8], dst: &mut [f16], num_elements: usize) {
    let src_f32 = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, num_elements) };
    for (i, &v) in src_f32.iter().enumerate() {
        dst[i] = f16::from_f32(v);
    }
}

/// Convert BF16 raw bytes to F32 values in a pre-allocated destination buffer.
///
/// # Safety
/// `src` must contain `num_elements * 2` bytes of valid BF16 data.
/// `dst` must have at least `num_elements` entries.
pub fn bf16_to_f32(src: &[u8], dst: &mut [f32], num_elements: usize) {
    let src_u16 = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, num_elements) };
    for (i, &b) in src_u16.iter().enumerate() {
        dst[i] = half::bf16::from_bits(b).to_f32();
    }
}

/// Convert F16 raw bytes to F32 values in a pre-allocated destination buffer.
///
/// # Safety
/// `src` must contain `num_elements * 2` bytes of valid F16 data.
/// `dst` must have at least `num_elements` entries.
pub fn f16_to_f32(src: &[u8], dst: &mut [f32], num_elements: usize) {
    let src_u16 = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, num_elements) };
    for (i, &b) in src_u16.iter().enumerate() {
        dst[i] = f16::from_bits(b).to_f32();
    }
}

/// Quantize F32 data into Q4_0 blocks.
///
/// `rows` and `cols` define the 2D shape. `cols` must be a multiple of `QK4_0` (32).
/// Returns a Vec of `BlockQ4_0` with `rows * (cols / QK4_0)` entries.
pub fn quantize_q4_0(f32_data: &[f32], rows: usize, cols: usize) -> Vec<BlockQ4_0> {
    let nb_k = cols / QK4_0;
    let mut blocks = Vec::with_capacity(rows * nb_k);
    for j in 0..rows {
        for bi in 0..nb_k {
            let offset = j * cols + bi * QK4_0;
            let src = &f32_data[offset..offset + QK4_0];
            let mut block = BlockQ4_0 {
                d: f16::from_f32(0.0),
                qs: [0; 16],
            };
            let max_val = src.iter().map(|v| v.abs()).fold(0.0f32, |x, y| x.max(y));
            let d = max_val / 7.0;
            let id = if d == 0.0 { 0.0 } else { 1.0 / d };
            block.d = f16::from_f32(d);
            for z in 0..16 {
                let v0 = (src[z] * id).round().clamp(-8.0, 7.0) as i8;
                let v1 = (src[z + 16] * id).round().clamp(-8.0, 7.0) as i8;
                block.qs[z] = (v0 + 8) as u8 | (((v1 + 8) as u8) << 4);
            }
            blocks.push(block);
        }
    }
    blocks
}

/// Release mmap pages after tensor data has been converted.
///
/// Calls `MADV_DONTNEED` on the page-aligned region so the kernel can reclaim
/// the source (e.g. BF16) pages that are no longer needed.
#[cfg(unix)]
pub fn release_source_pages(data: &[u8]) {
    const PAGE_SIZE: usize = 4096;
    let start = data.as_ptr() as usize;
    let end = start + data.len();
    let aligned_start = (start + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    let aligned_end = end & !(PAGE_SIZE - 1);
    if aligned_end > aligned_start {
        unsafe {
            libc::madvise(
                aligned_start as *mut libc::c_void,
                aligned_end - aligned_start,
                libc::MADV_DONTNEED,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_to_f32_roundtrip() {
        // bf16(1.0) = 0x3F80
        let bf16_bytes: [u8; 4] = [0x80, 0x3F, 0x00, 0x40]; // 1.0, 2.0 in BF16
        let mut dst = [0.0f32; 2];
        bf16_to_f32(&bf16_bytes, &mut dst, 2);
        assert!((dst[0] - 1.0).abs() < 1e-2);
        assert!((dst[1] - 2.0).abs() < 1e-2);
    }

    #[test]
    fn test_f16_to_f32_roundtrip() {
        let val = f16::from_f32(3.14);
        let bytes = val.to_bits().to_le_bytes();
        let mut src = [0u8; 2];
        src.copy_from_slice(&bytes);
        let mut dst = [0.0f32; 1];
        f16_to_f32(&src, &mut dst, 1);
        assert!((dst[0] - 3.14).abs() < 0.01);
    }

    #[test]
    fn test_bf16_to_f16_roundtrip() {
        // bf16(1.0) = 0x3F80
        let bf16_bytes: [u8; 2] = [0x80, 0x3F];
        let mut dst = [f16::from_f32(0.0); 1];
        bf16_to_f16_buf(&bf16_bytes, &mut dst, 1);
        assert!((dst[0].to_f32() - 1.0).abs() < 1e-2);
    }

    #[test]
    fn test_f32_to_f16_roundtrip() {
        let val = 2.5f32;
        let bytes = val.to_le_bytes();
        let mut dst = [f16::from_f32(0.0); 1];
        f32_to_f16_buf(&bytes, &mut dst, 1);
        assert!((dst[0].to_f32() - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_quantize_q4_0_basic() {
        // 1 row, 32 cols (1 block)
        let mut data = vec![0.0f32; 32];
        data[0] = 7.0;
        data[16] = -7.0;
        let blocks = quantize_q4_0(&data, 1, 32);
        assert_eq!(blocks.len(), 1);
        // d should be ~1.0 (max_abs=7.0, d=7.0/7.0=1.0)
        assert!((blocks[0].d.to_f32() - 1.0).abs() < 0.01);
    }
}
