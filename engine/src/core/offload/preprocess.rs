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
