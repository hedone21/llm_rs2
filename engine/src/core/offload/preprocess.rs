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

// ── Bitshuffle (bit-plane transposition) ─────────────────────────────────

/// Bitshuffle: transpose N elements of `elem_size` bytes into bit-planes.
///
/// For N F16 values (elem_size=2), the 16-bit values are rearranged so that
/// all bit-0 values are contiguous, all bit-1 values, etc. This groups
/// exponent bits (planes 10-14) and sign (plane 15) together — these have
/// low entropy and compress extremely well. Mantissa planes (0-9) remain
/// high-entropy but no worse than before.
///
/// Layout: `src` has N elements of `elem_size` bytes each.
/// `dst` receives `elem_size * 8` bit-planes, each of `N / 8` bytes
/// (N must be a multiple of 8).
pub fn bitshuffle(src: &[u8], dst: &mut [u8], elem_size: usize) {
    debug_assert_eq!(src.len(), dst.len());
    match elem_size {
        2 => bitshuffle_f16(src, dst),
        _ => bitshuffle_generic(src, dst, elem_size),
    }
}

/// Optimized bitshuffle for F16 (elem_size=2, 16 bit-planes).
fn bitshuffle_f16(src: &[u8], dst: &mut [u8]) {
    let n_elem = src.len() / 2;
    debug_assert_eq!(n_elem % 8, 0);
    let plane_bytes = n_elem / 8;

    for group in 0..plane_bytes {
        // Load 8 F16 values as u16
        let base = group * 8 * 2;
        let mut vals = [0u16; 8];
        for (j, val) in vals.iter_mut().enumerate() {
            *val = u16::from_le_bytes([src[base + j * 2], src[base + j * 2 + 1]]);
        }

        // For each of 16 bit-planes, gather bits from 8 elements
        for bit in 0..16u32 {
            let mask = 1u16 << bit;
            let mut out_byte = 0u8;
            for (j, val) in vals.iter().enumerate() {
                if val & mask != 0 {
                    out_byte |= 1 << j;
                }
            }
            dst[bit as usize * plane_bytes + group] = out_byte;
        }
    }
}

/// Generic bitshuffle for arbitrary elem_size.
fn bitshuffle_generic(src: &[u8], dst: &mut [u8], elem_size: usize) {
    let n_elem = src.len() / elem_size;
    debug_assert_eq!(n_elem % 8, 0);
    let bits_per_elem = elem_size * 8;
    let plane_bytes = n_elem / 8;

    for group in 0..plane_bytes {
        let elem_base = group * 8;
        for bit in 0..bits_per_elem {
            let byte_in_elem = bit / 8;
            let bit_in_byte = bit % 8;
            let mask = 1u8 << bit_in_byte;
            let mut out_byte = 0u8;
            for j in 0..8 {
                let src_byte = src[(elem_base + j) * elem_size + byte_in_elem];
                if src_byte & mask != 0 {
                    out_byte |= 1 << j;
                }
            }
            dst[bit * plane_bytes + group] = out_byte;
        }
    }
}

/// Reverse bitshuffle: reconstruct elements from bit-planes.
pub fn bitunshuffle(src: &[u8], dst: &mut [u8], elem_size: usize) {
    debug_assert_eq!(src.len(), dst.len());
    // Dispatch to specialized F16 path or generic path
    match elem_size {
        2 => bitunshuffle_f16(src, dst),
        _ => bitunshuffle_generic(src, dst, elem_size),
    }
}

/// Optimized bitunshuffle for F16 (elem_size=2, 16 bit-planes).
/// Processes one output byte per plane in the inner loop, avoiding per-bit branching.
fn bitunshuffle_f16(src: &[u8], dst: &mut [u8]) {
    let n_elem = dst.len() / 2;
    debug_assert_eq!(n_elem % 8, 0);
    let plane_bytes = n_elem / 8;

    // For each group of 8 F16 elements, reconstruct from 16 bit-planes
    for group in 0..plane_bytes {
        // Accumulate 8 elements as u16 values
        let mut vals = [0u16; 8];
        for bit in 0..16u32 {
            let in_byte = src[bit as usize * plane_bytes + group];
            // Scatter each bit from in_byte to the 8 element accumulators
            for (j, val) in vals.iter_mut().enumerate() {
                *val |= (((in_byte >> j) & 1) as u16) << bit;
            }
        }
        // Write out as little-endian bytes
        let base = group * 8 * 2;
        for (j, val) in vals.iter().enumerate() {
            let bytes = val.to_le_bytes();
            dst[base + j * 2] = bytes[0];
            dst[base + j * 2 + 1] = bytes[1];
        }
    }
}

/// Generic bitunshuffle for arbitrary elem_size.
fn bitunshuffle_generic(src: &[u8], dst: &mut [u8], elem_size: usize) {
    let n_elem = dst.len() / elem_size;
    debug_assert_eq!(n_elem % 8, 0);
    let bits_per_elem = elem_size * 8;
    let plane_bytes = n_elem / 8;

    for group in 0..plane_bytes {
        let elem_base = group * 8;
        for j in 0..8 {
            for b in 0..elem_size {
                dst[(elem_base + j) * elem_size + b] = 0;
            }
        }
        for bit in 0..bits_per_elem {
            let byte_in_elem = bit / 8;
            let bit_in_byte = bit % 8;
            let in_byte = src[bit * plane_bytes + group];
            for j in 0..8 {
                if in_byte & (1 << j) != 0 {
                    dst[(elem_base + j) * elem_size + byte_in_elem] |= 1 << bit_in_byte;
                }
            }
        }
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

    // ── Bitshuffle tests ──

    #[test]
    fn test_bitshuffle_roundtrip_f16() {
        // 16 F16 elements = 32 bytes, N must be multiple of 8
        let original: Vec<u8> = (0..32).collect();
        let mut shuffled = vec![0u8; 32];
        let mut restored = vec![0u8; 32];

        bitshuffle(&original, &mut shuffled, 2);
        assert_ne!(&shuffled, &original, "bitshuffle should change data");
        bitunshuffle(&shuffled, &mut restored, 2);
        assert_eq!(&restored, &original, "bitunshuffle must restore original");
    }

    #[test]
    fn test_bitshuffle_roundtrip_f32() {
        let original: Vec<u8> = (0..128).collect();
        let mut shuffled = vec![0u8; 128];
        let mut restored = vec![0u8; 128];

        bitshuffle(&original, &mut shuffled, 4);
        assert_ne!(&shuffled, &original);
        bitunshuffle(&shuffled, &mut restored, 4);
        assert_eq!(&restored, &original);
    }

    #[test]
    fn test_bitshuffle_plane_structure_f16() {
        // 8 identical F16 values: 0x3C00 = 1.0 in F16
        // LE bytes: [0x00, 0x3C] repeated 8 times
        let mut src = vec![0u8; 16];
        for i in 0..8 {
            src[i * 2] = 0x00; // low byte
            src[i * 2 + 1] = 0x3C; // high byte = 0011_1100
        }
        let mut dst = vec![0u8; 16];
        bitshuffle(&src, &mut dst, 2);

        // plane_bytes = 8/8 = 1 byte per plane
        // Planes 0-7 (low byte = 0x00): all bits are 0 → plane bytes = 0x00
        for plane in 0..8 {
            assert_eq!(
                dst[plane], 0x00,
                "mantissa low plane {plane} should be all-zero"
            );
        }
        // Planes 8-9 (high byte bits 0-1, = 0b00): plane bytes = 0x00
        assert_eq!(dst[8], 0x00);
        assert_eq!(dst[9], 0x00);
        // Planes 10-13 (high byte bits 2-5 of 0x3C = 0b0011_1100):
        //   bit2=1, bit3=1, bit4=1, bit5=1 → all 8 elements have this set
        assert_eq!(dst[10], 0xFF, "exponent plane 10");
        assert_eq!(dst[11], 0xFF, "exponent plane 11");
        assert_eq!(dst[12], 0xFF, "exponent plane 12");
        assert_eq!(dst[13], 0xFF, "exponent plane 13");
        // Planes 14-15 (bits 6-7 of 0x3C = 0b00): 0x00
        assert_eq!(dst[14], 0x00, "exponent plane 14");
        assert_eq!(dst[15], 0x00, "sign plane 15");
    }

    #[test]
    fn test_bitshuffle_large_roundtrip_f16() {
        // Realistic size: 512 F16 elements
        let n = 512;
        let mut original = vec![0u8; n * 2];
        for i in 0..n {
            let val = 0x3C00u16 + (i as u16 % 1024);
            original[i * 2] = (val & 0xFF) as u8;
            original[i * 2 + 1] = (val >> 8) as u8;
        }
        let mut shuffled = vec![0u8; n * 2];
        let mut restored = vec![0u8; n * 2];

        bitshuffle(&original, &mut shuffled, 2);
        bitunshuffle(&shuffled, &mut restored, 2);
        assert_eq!(&restored, &original);
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
