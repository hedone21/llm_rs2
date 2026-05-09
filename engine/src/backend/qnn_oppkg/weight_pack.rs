//! GGML Q4_0 AOS → SOA layout 변환 (M3.4).
//!
//! ## 배경
//! Production GGUF Q4_0 weight는 AOS layout으로 mmap된다 — 한 block은 18 bytes
//! (2B FP16 scale + 16B nibbles). qnn_oppkg의 `MatMulQ40F32` op (M2 frozen)는
//! SOA layout을 요구한다 — `q[N*K/2]` 모든 nibbles가 연속, `d[N*K/32]` 모든
//! scales가 연속. 본 모듈은 zero-copy 불가능한 layout transform 1회 (model load
//! 시점)를 책임진다.
//!
//! ## 시그니처 정합 (M2 microbench `pack_q40_soa`)
//! microbench는 f32 weights → quantize → SOA. 본 모듈은 GGUF에서 이미 quantize
//! 된 AOS bytes → SOA. 결과 byte layout은 동일해야 한다 (M2 op이 SOA만 인식).
//!
//! ## Spec
//! - INV-176 — qnn_oppkg는 Qwen2.5-1.5B Q4_0 weight를 AOS→SOA 변환 후 graph
//!   binding.

/// GGML Q4_0 block size (elements per block).
pub const QK4_0: usize = 32;
/// Bytes per Q4_0 block in GGML AOS layout: 2 (FP16 scale) + 16 (nibbles).
pub const Q4_0_BLOCK_BYTES: usize = 18;
/// Nibble bytes per block (16 bytes = 32 nibbles = 32 quants).
pub const QS_PER_BLOCK: usize = 16;

/// AOS Q4_0 bytes (`num_blocks × 18`) → SOA `(q_bytes, d_halves)`.
///
/// Input layout (GGUF native): for each block i:
///   `aos[i*18..i*18+2]` = FP16 scale (little-endian u16)
///   `aos[i*18+2..i*18+18]` = 16 nibble bytes
///
/// Output layout (SOA, M2 `pack_q40_soa` 호환):
///   `q[i*16..i*16+16]` = block i 의 16 nibble bytes (이미 동일 packing)
///   `d[i]` = block i 의 FP16 scale (u16)
///
/// `n` = rows, `k` = cols. `k % QK4_0 == 0` 강제.
pub fn aos_to_soa_q4_0(aos: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<u16>) {
    assert!(
        k.is_multiple_of(QK4_0),
        "k={k} must be a multiple of QK4_0={QK4_0}"
    );
    let num_blocks = n * k / QK4_0;
    assert_eq!(
        aos.len(),
        num_blocks * Q4_0_BLOCK_BYTES,
        "aos.len()={} mismatch with expected {}*18={}",
        aos.len(),
        num_blocks,
        num_blocks * Q4_0_BLOCK_BYTES
    );

    let mut q = vec![0u8; num_blocks * QS_PER_BLOCK];
    let mut d = vec![0u16; num_blocks];

    for (blk, d_slot) in d.iter_mut().enumerate().take(num_blocks) {
        let aos_off = blk * Q4_0_BLOCK_BYTES;
        let scale_lo = aos[aos_off];
        let scale_hi = aos[aos_off + 1];
        *d_slot = u16::from_le_bytes([scale_lo, scale_hi]);
        let nib_off = aos_off + 2;
        let q_off = blk * QS_PER_BLOCK;
        q[q_off..q_off + QS_PER_BLOCK].copy_from_slice(&aos[nib_off..nib_off + QS_PER_BLOCK]);
    }

    (q, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AOS → SOA round-trip: 단일 block에 대해 d/q split이 정확한지.
    #[test]
    fn split_single_block() {
        let mut aos = [0u8; 18];
        // scale = 0x1234 (little-endian)
        aos[0] = 0x34;
        aos[1] = 0x12;
        // nibbles = 0..15
        for i in 0..16 {
            aos[2 + i] = i as u8;
        }
        let (q, d) = aos_to_soa_q4_0(&aos, 1, 32);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0], 0x1234);
        assert_eq!(q.len(), 16);
        for i in 0..16 {
            assert_eq!(q[i], i as u8);
        }
    }

    /// 2 rows × 32 cols = 2 blocks. 각 block의 d/q가 순서대로 들어가는지.
    #[test]
    fn two_block_layout() {
        let mut aos = vec![0u8; 36];
        // block 0: d=0x1111, q[i]=i
        aos[0] = 0x11;
        aos[1] = 0x11;
        for i in 0..16 {
            aos[2 + i] = i as u8;
        }
        // block 1: d=0x2222, q[i]=0x80+i
        aos[18] = 0x22;
        aos[19] = 0x22;
        for i in 0..16 {
            aos[20 + i] = (0x80 + i) as u8;
        }
        let (q, d) = aos_to_soa_q4_0(&aos, 2, 32);
        assert_eq!(d, vec![0x1111, 0x2222]);
        assert_eq!(q.len(), 32);
        for i in 0..16 {
            assert_eq!(q[i], i as u8, "block 0 nibble {i}");
        }
        for i in 0..16 {
            assert_eq!(q[16 + i], (0x80 + i) as u8, "block 1 nibble {i}");
        }
    }

    #[test]
    #[should_panic(expected = "must be a multiple of")]
    fn k_not_multiple_of_32_panics() {
        let aos = vec![0u8; 18];
        let _ = aos_to_soa_q4_0(&aos, 1, 31);
    }
}
