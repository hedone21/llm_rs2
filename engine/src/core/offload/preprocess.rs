//! Byte-shuffle preprocessing for floating-point KV cache data.
//!
//! Raw F16/F32 data is nearly incompressible by LZ4 (~1.0x ratio).
//! Byte-shuffle groups high-order bytes together, enabling LZ4 to find
//! patterns in the exponent/sign bytes, achieving ~2.0-2.3x for F16.
//!
//! F16 layout: [b0_hi, b0_lo, b1_hi, b1_lo, ...] → [b0_hi, b1_hi, ...][b0_lo, b1_lo, ...]
//! F32 layout: [b0_3, b0_2, b0_1, b0_0, ...] → [b0_3, b1_3, ...][b0_2, b1_2, ...][...]

/// Byte-shuffle F16 data (2-byte elements) for better LZ4 compression.
/// `src` and `dst` must have the same length and be a multiple of 2.
pub fn shuffle_f16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert!(src.len().is_multiple_of(2));
    let n = src.len() / 2;
    let (hi, lo) = dst.split_at_mut(n);
    for i in 0..n {
        hi[i] = src[i * 2 + 1]; // high byte (exponent + sign)
        lo[i] = src[i * 2]; // low byte (mantissa)
    }
}

/// Reverse byte-shuffle for F16 data.
pub fn unshuffle_f16(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert!(src.len().is_multiple_of(2));
    let n = src.len() / 2;
    let (hi, lo) = src.split_at(n);
    for i in 0..n {
        dst[i * 2 + 1] = hi[i];
        dst[i * 2] = lo[i];
    }
}

/// Byte-shuffle F32 data (4-byte elements) for better LZ4 compression.
/// `src` and `dst` must have the same length and be a multiple of 4.
pub fn shuffle_f32(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert!(src.len().is_multiple_of(4));
    let n = src.len() / 4;
    for i in 0..n {
        dst[i] = src[i * 4 + 3]; // byte 3 (MSB: sign+exponent)
        dst[n + i] = src[i * 4 + 2]; // byte 2
        dst[2 * n + i] = src[i * 4 + 1]; // byte 1
        dst[3 * n + i] = src[i * 4]; // byte 0 (LSB)
    }
}

/// Reverse byte-shuffle for F32 data.
pub fn unshuffle_f32(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert!(src.len().is_multiple_of(4));
    let n = src.len() / 4;
    for i in 0..n {
        dst[i * 4 + 3] = src[i]; // MSB
        dst[i * 4 + 2] = src[n + i];
        dst[i * 4 + 1] = src[2 * n + i];
        dst[i * 4] = src[3 * n + i]; // LSB
    }
}

/// Dispatch shuffle based on element size (2 = F16, 4 = F32).
pub fn shuffle(src: &[u8], dst: &mut [u8], elem_size: usize) {
    match elem_size {
        2 => shuffle_f16(src, dst),
        4 => shuffle_f32(src, dst),
        _ => dst.copy_from_slice(src), // no-op for other types
    }
}

/// Dispatch unshuffle based on element size.
pub fn unshuffle(src: &[u8], dst: &mut [u8], elem_size: usize) {
    match elem_size {
        2 => unshuffle_f16(src, dst),
        4 => unshuffle_f32(src, dst),
        _ => dst.copy_from_slice(src),
    }
}

// ── Bytedelta filter (Blosc2-inspired) ──────────────────────────────────

/// Bytedelta encode: apply delta encoding within each byte stream.
///
/// After byte-shuffle, data is organized as `n_streams` contiguous streams
/// of `stream_len` bytes each. Within each stream, consecutive bytes often
/// differ by small amounts. Delta encoding replaces each byte with
/// `current - previous`, producing many near-zero values that compress well.
///
/// This is an in-place operation on the shuffled data.
pub fn bytedelta_encode(data: &mut [u8], stream_len: usize, n_streams: usize) {
    debug_assert_eq!(data.len(), stream_len * n_streams);
    for s in 0..n_streams {
        let base = s * stream_len;
        let mut prev = 0u8;
        for i in 0..stream_len {
            let curr = data[base + i];
            data[base + i] = curr.wrapping_sub(prev);
            prev = curr;
        }
    }
}

/// Bytedelta decode: restore original bytes via prefix-sum.
///
/// Inverse of `bytedelta_encode`. This is an in-place operation.
pub fn bytedelta_decode(data: &mut [u8], stream_len: usize, n_streams: usize) {
    debug_assert_eq!(data.len(), stream_len * n_streams);
    for s in 0..n_streams {
        let base = s * stream_len;
        let mut prev = 0u8;
        for i in 0..stream_len {
            let val = data[base + i].wrapping_add(prev);
            data[base + i] = val;
            prev = val;
        }
    }
}

// ── Truncated precision filter (lossy) ──────────────────────────────────

/// Zero the lowest `zero_bits` mantissa bits of each F16 value.
///
/// F16 format: 1 sign + 5 exponent + 10 mantissa bits.
/// Zeroing low mantissa bits reduces effective precision but makes the
/// data much more compressible (more repeated byte patterns after shuffle).
///
/// `zero_bits` must be in 0..=10. With 0, data is unchanged (lossless).
pub fn trunc_prec_f16(data: &mut [u8], zero_bits: u32) {
    if zero_bits == 0 || zero_bits > 10 {
        return;
    }
    let mask = !((1u16 << zero_bits) - 1);
    for chunk in data.chunks_exact_mut(2) {
        let val = u16::from_le_bytes([chunk[0], chunk[1]]);
        let truncated = val & mask;
        let bytes = truncated.to_le_bytes();
        chunk[0] = bytes[0];
        chunk[1] = bytes[1];
    }
}

/// Zero the lowest `zero_bits` mantissa bits of each F32 value.
///
/// F32 format: 1 sign + 8 exponent + 23 mantissa bits.
/// `zero_bits` must be in 0..=23.
pub fn trunc_prec_f32(data: &mut [u8], zero_bits: u32) {
    if zero_bits == 0 || zero_bits > 23 {
        return;
    }
    let mask = !((1u32 << zero_bits) - 1);
    for chunk in data.chunks_exact_mut(4) {
        let val = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let truncated = val & mask;
        let bytes = truncated.to_le_bytes();
        chunk[0] = bytes[0];
        chunk[1] = bytes[1];
        chunk[2] = bytes[2];
        chunk[3] = bytes[3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffle_unshuffle_f16_roundtrip() {
        let original: Vec<u8> = (0..64).collect();
        let mut shuffled = vec![0u8; 64];
        let mut restored = vec![0u8; 64];

        shuffle_f16(&original, &mut shuffled);
        assert_ne!(&original, &shuffled, "shuffle should change data");

        unshuffle_f16(&shuffled, &mut restored);
        assert_eq!(&original, &restored, "unshuffle must restore original");
    }

    #[test]
    fn test_shuffle_unshuffle_f32_roundtrip() {
        let original: Vec<u8> = (0..128).collect();
        let mut shuffled = vec![0u8; 128];
        let mut restored = vec![0u8; 128];

        shuffle_f32(&original, &mut shuffled);
        assert_ne!(&original, &shuffled, "shuffle should change data");

        unshuffle_f32(&shuffled, &mut restored);
        assert_eq!(&original, &restored, "unshuffle must restore original");
    }

    #[test]
    fn test_shuffle_f16_layout() {
        // [0xAA, 0xBB, 0xCC, 0xDD] → hi=[0xBB, 0xDD] lo=[0xAA, 0xCC]
        let src = [0xAA, 0xBB, 0xCC, 0xDD];
        let mut dst = [0u8; 4];
        shuffle_f16(&src, &mut dst);
        assert_eq!(dst, [0xBB, 0xDD, 0xAA, 0xCC]);
    }

    #[test]
    fn test_shuffle_f32_layout() {
        // [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        // elem0: [01,02,03,04], elem1: [05,06,07,08]
        // → byte3: [04,08], byte2: [03,07], byte1: [02,06], byte0: [01,05]
        let src = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let mut dst = [0u8; 8];
        shuffle_f32(&src, &mut dst);
        assert_eq!(dst, [0x04, 0x08, 0x03, 0x07, 0x02, 0x06, 0x01, 0x05]);
    }

    #[test]
    fn test_shuffle_dispatch() {
        let original: Vec<u8> = (0..64).collect();
        let mut shuffled = vec![0u8; 64];
        let mut restored = vec![0u8; 64];

        // F16 dispatch
        shuffle(&original, &mut shuffled, 2);
        unshuffle(&shuffled, &mut restored, 2);
        assert_eq!(&original, &restored);

        // F32 dispatch
        shuffle(&original, &mut shuffled, 4);
        unshuffle(&shuffled, &mut restored, 4);
        assert_eq!(&original, &restored);

        // Unknown elem size: identity
        shuffle(&original, &mut shuffled, 1);
        assert_eq!(&original, &shuffled);
    }

    // ── Bytedelta tests ──

    #[test]
    fn test_bytedelta_roundtrip() {
        let original: Vec<u8> = (0..64).collect();
        let mut data = original.clone();
        // F16: 2 streams of 32 bytes each
        bytedelta_encode(&mut data, 32, 2);
        assert_ne!(&data, &original, "encode should change data");
        bytedelta_decode(&mut data, 32, 2);
        assert_eq!(&data, &original, "decode must restore original");
    }

    #[test]
    fn test_bytedelta_constant_stream() {
        // Constant bytes → all deltas should be 0 after the first
        let mut data = vec![0x3Cu8; 16];
        bytedelta_encode(&mut data, 16, 1);
        assert_eq!(data[0], 0x3C); // first element unchanged
        assert!(
            data[1..].iter().all(|&b| b == 0),
            "constant stream → zero deltas"
        );
    }

    #[test]
    fn test_bytedelta_with_shuffle_f16() {
        // Full pipeline: shuffle → bytedelta → bytedelta_decode → unshuffle
        let original: Vec<u8> = (0..128).collect();
        let n = original.len() / 2; // 64 elements

        let mut shuffled = vec![0u8; 128];
        shuffle_f16(&original, &mut shuffled);
        bytedelta_encode(&mut shuffled, n, 2);

        // Decode
        bytedelta_decode(&mut shuffled, n, 2);
        let mut restored = vec![0u8; 128];
        unshuffle_f16(&shuffled, &mut restored);
        assert_eq!(&restored, &original);
    }

    #[test]
    fn test_bytedelta_with_shuffle_f32() {
        let original: Vec<u8> = (0..128).collect();
        let n = original.len() / 4; // 32 elements

        let mut shuffled = vec![0u8; 128];
        shuffle_f32(&original, &mut shuffled);
        bytedelta_encode(&mut shuffled, n, 4);

        bytedelta_decode(&mut shuffled, n, 4);
        let mut restored = vec![0u8; 128];
        unshuffle_f32(&shuffled, &mut restored);
        assert_eq!(&restored, &original);
    }

    #[test]
    fn test_bytedelta_wrapping() {
        // Test wrapping arithmetic: 0x02 - 0xFF = 0x03 (wrapping)
        let mut data = vec![0xFF, 0x02];
        bytedelta_encode(&mut data, 2, 1);
        assert_eq!(data, vec![0xFF, 0x03]); // 0x02 - 0xFF = 0x03 (wrapping_sub)
        bytedelta_decode(&mut data, 2, 1);
        assert_eq!(data, vec![0xFF, 0x02]);
    }

    // ── Trunc prec tests ──

    #[test]
    fn test_trunc_prec_f16_zero_bits_noop() {
        let original = vec![0x12, 0x34, 0x56, 0x78];
        let mut data = original.clone();
        trunc_prec_f16(&mut data, 0);
        assert_eq!(&data, &original, "zero_bits=0 should be no-op");
    }

    #[test]
    fn test_trunc_prec_f16_masks_low_bits() {
        // F16 value 0x3C01 (1.0 + small mantissa)
        // zero_bits=4 → mask = !0x000F = 0xFFF0
        // 0x3C01 & 0xFFF0 = 0x3C00
        let mut data = vec![0x01, 0x3C]; // little-endian 0x3C01
        trunc_prec_f16(&mut data, 4);
        assert_eq!(data, vec![0x00, 0x3C]); // 0x3C00
    }

    #[test]
    fn test_trunc_prec_f16_max_bits() {
        // zero_bits=10 → all mantissa zeroed, only sign+exponent remain
        // 0x3FFF → 0x3C00 (exponent=15, mantissa=0)
        let mut data = vec![0xFF, 0x3F];
        trunc_prec_f16(&mut data, 10);
        assert_eq!(data, vec![0x00, 0x3C]); // sign=0, exp=15, mantissa=0
    }

    #[test]
    fn test_trunc_prec_f16_over_10_noop() {
        let original = vec![0xFF, 0x3F];
        let mut data = original.clone();
        trunc_prec_f16(&mut data, 11);
        assert_eq!(&data, &original, "zero_bits>10 should be no-op");
    }

    #[test]
    fn test_trunc_prec_f32_masks_low_bits() {
        // F32 value 0x3F800001 (1.0 + tiny mantissa)
        // zero_bits=8 → mask = !0xFF = 0xFFFFFF00
        // 0x3F800001 & 0xFFFFFF00 = 0x3F800000
        let mut data = vec![0x01, 0x00, 0x80, 0x3F]; // little-endian
        trunc_prec_f32(&mut data, 8);
        assert_eq!(data, vec![0x00, 0x00, 0x80, 0x3F]); // exact 1.0
    }

    #[test]
    fn test_trunc_then_shuffle_pipeline_f16() {
        // Full lossy pipeline: trunc → shuffle → bytedelta → reverse
        let mut data = vec![0u8; 64];
        for i in 0..32 {
            let val = 0x3C00u16 + (i as u16 * 17); // varying mantissa
            data[i * 2] = (val & 0xFF) as u8;
            data[i * 2 + 1] = (val >> 8) as u8;
        }

        // Apply trunc
        let mut truncated = data.clone();
        trunc_prec_f16(&mut truncated, 4);

        // Full pipeline
        let n = truncated.len() / 2;
        let mut shuffled = vec![0u8; 64];
        shuffle_f16(&truncated, &mut shuffled);
        bytedelta_encode(&mut shuffled, n, 2);

        // Reverse
        bytedelta_decode(&mut shuffled, n, 2);
        let mut restored = vec![0u8; 64];
        unshuffle_f16(&shuffled, &mut restored);

        // Should match truncated (not original!)
        assert_eq!(&restored, &truncated);
        // But NOT the original (lossy)
        assert_ne!(&restored, &data);
    }

    #[test]
    fn test_shuffle_realistic_f16_data() {
        // Simulate F16 KV cache data: many values with similar exponents
        // F16 format: [mantissa_lo, sign_exp_hi]
        // Similar values → high bytes are similar → good compression after shuffle
        let n_elements = 512;
        let mut data = vec![0u8; n_elements * 2];
        for i in 0..n_elements {
            // Simulated F16: exponent ~15 (0x3C00 range), varying mantissa
            let val = 0x3C00u16 + (i as u16 % 256);
            data[i * 2] = (val & 0xFF) as u8;
            data[i * 2 + 1] = (val >> 8) as u8;
        }

        let mut shuffled = vec![0u8; data.len()];
        shuffle_f16(&data, &mut shuffled);

        // After shuffle, first half should be high bytes (mostly 0x3C/0x3D)
        let hi_bytes = &shuffled[..n_elements];
        let unique_hi: std::collections::HashSet<u8> = hi_bytes.iter().copied().collect();
        // With similar exponents, unique high bytes should be very few
        assert!(
            unique_hi.len() <= 4,
            "expected few unique high bytes, got {}",
            unique_hi.len()
        );

        // Verify roundtrip
        let mut restored = vec![0u8; data.len()];
        unshuffle_f16(&shuffled, &mut restored);
        assert_eq!(&data, &restored);
    }
}
