//! HTP FastRPC backend — Q4_0 → q4x4x2 weight repack (INV-HTP-FRPC-002 인접).
//!
//! llama.cpp HTP backend 는 standard `block_q4_0` (32 elem × 18 B) 를 **q4x4x2
//! layout** 으로 repack 후 DSP 에 전달. DSP-side `vec_dot_q4x4x2_q8x4x2_*`
//! (matmul-ops.c:734+) 가 이 layout 을 expect.
//!
//! q4x4x2 group = 8 인접 Q4_0 block (256 elem, 144 B):
//!   - 128 B quants (각 elem 4-bit pair-packed, group 내 256 elem)
//!   - 16 B scales (8 × f16)
//!
//! 본 모듈은 row-major weight matrix `W[N, K]` 의 standard Q4_0 byte stream
//! 을 q4x4x2-packed byte stream 으로 변환. K 는 256 multiple 가정 (Qwen2.5
//! shape K∈{1536, 8960} 모두 manage).
//!
//! microbench `engine/microbench/htp_matmul.rs` 가 GREEN reference 였고 본
//! 모듈로 byte-for-byte 승격 (de-dup). 알고리즘 변경 없음.

#![cfg(feature = "htp_fastrpc")]

use crate::quant::BlockQ4_0;

/// row-major weight matrix `W[N, K]` 의 standard Q4_0 byte stream 을
/// q4x4x2-packed byte stream 으로 변환. 결과 byte length 는 입력과 동일
/// (`n_rows * (k/32) * 18`).
pub fn repack_q4_0_to_q4x4x2_matrix(src: &[BlockQ4_0], n_rows: usize, k: usize) -> Vec<u8> {
    const QK_Q4_0X4X2: usize = 256; // llama.cpp htp-msg.h: QK_Q4_0x4x2
    const QK4_0: usize = 32;
    assert!(
        k.is_multiple_of(QK_Q4_0X4X2),
        "K must be QK_Q4_0x4x2 multiple"
    );
    let blocks_per_row = k / QK4_0; // 18 B/block in standard layout
    let row_bytes = blocks_per_row * 18; // = (k/32)*18 = k * 9 / 16
    let mut out = vec![0u8; n_rows * row_bytes];
    for r in 0..n_rows {
        let src_row = &src[r * blocks_per_row..(r + 1) * blocks_per_row];
        let dst_row = &mut out[r * row_bytes..(r + 1) * row_bytes];
        repack_row_q4x4x2(dst_row, src_row, k);
    }
    out
}

fn repack_row_q4x4x2(y: &mut [u8], x: &[BlockQ4_0], k: usize) {
    use half::f16;
    const QK_Q4_0X4X2: usize = 256;
    const QK4_0: usize = 32;
    let nb = k.div_ceil(QK_Q4_0X4X2); // number of q4x4x2 groups

    let dblk_size = 8 * 2; // 8 × f16 = 16 B
    let qblk_size = QK_Q4_0X4X2 / 2; // 128 B (4-bit per elem, 256 elem)
    let qrow_size = k / 2; // K/2 B (int4 not padded to blocks)

    // y_q at offset 0 (quants), y_d at offset qrow_size (scales).
    // SAFETY: caller 가 y 가 충분히 큼 (row_bytes = qrow_size + nb*dblk_size)을 보장.
    //
    // standard block_q4_0 (`{d: f16, qs: [u8; 16]}`) 의 qs 는 nibble pair-pack
    // 된 32 elem. nibble unpack 시 lower nibble = elem [0..16], upper nibble =
    // elem [16..32].

    // Repack quants
    for i in 0..nb {
        // unpacked 256 elem buffer for this group (8 blocks)
        let mut qs_unpacked = [0u8; QK_Q4_0X4X2];
        for bi in 0..8 {
            let block_idx = i * 8 + bi;
            if block_idx >= x.len() {
                break;
            }
            let blk = &x[block_idx];
            // unpack_q4_0_quants (ggml-hexagon.cpp:381)
            for j in 0..(QK4_0 / 2) {
                let x0 = blk.qs[j] & 0x0F;
                let x1 = blk.qs[j] >> 4;
                qs_unpacked[bi * QK4_0 + j] = x0;
                qs_unpacked[bi * QK4_0 + j + QK4_0 / 2] = x1;
            }
        }
        // repack: `q[j] = (qs[j+128] << 4) | qs[j]` for j in [0..128)
        let q_off = i * qblk_size;
        for j in 0..(QK_Q4_0X4X2 / 2) {
            y[q_off + j] = (qs_unpacked[j + 128] << 4) | qs_unpacked[j];
        }
    }

    // Repack scales (8 × f16 per group)
    for i in 0..nb {
        let d_off = qrow_size + i * dblk_size;
        for bi in 0..8 {
            let block_idx = i * 8 + bi;
            if block_idx >= x.len() {
                break;
            }
            let d_bits: u16 = f16::to_bits(x[block_idx].d);
            y[d_off + bi * 2] = (d_bits & 0xFF) as u8;
            y[d_off + bi * 2 + 1] = ((d_bits >> 8) & 0xFF) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::BlockQ4_0;
    use half::f16;

    /// q4x4x2 repack 은 byte-length 를 보존한다 (입력 Q4_0 byte len 동일).
    #[test]
    fn repack_preserves_byte_length() {
        // 1 row, K=256 → 8 blocks × 18 B = 144 B.
        let k = 256;
        let n_rows = 1;
        let blocks_per_row = k / 32; // 8
        let blocks: Vec<BlockQ4_0> = (0..n_rows * blocks_per_row)
            .map(|i| BlockQ4_0 {
                d: f16::from_f32(0.5),
                qs: [(i as u8).wrapping_mul(3).wrapping_add(1); 16],
            })
            .collect();
        let input_bytes = blocks.len() * std::mem::size_of::<BlockQ4_0>();
        let out = repack_q4_0_to_q4x4x2_matrix(&blocks, n_rows, k);
        assert_eq!(out.len(), input_bytes, "repack must preserve byte length");
        assert_eq!(out.len(), 144);
    }

    /// 작은 알려진 입력으로 q4x4x2 layout 의 quants(offset 0)/scales(offset K/2)
    /// 분리가 손으로 계산한 기대 byte 와 일치함을 검증.
    ///
    /// K=256, 1 row, 8 blocks (= 1 q4x4x2 group). 각 block 의 d 는 distinct
    /// f16 sentinel, qs 는 block index 로 식별 가능한 nibble pattern.
    #[test]
    fn repack_q4x4x2_layout_known_input() {
        let k = 256;
        let n_rows = 1;
        let blocks_per_row = k / 32; // 8 (= one group)

        // block bi: qs[j] = (bi as nibble in both halves). lower nibble = bi,
        // upper nibble = bi → qs byte = bi*16 + bi = bi*17.
        // d = f16(bi as f32 + 1.0) → distinct per block.
        let blocks: Vec<BlockQ4_0> = (0..blocks_per_row)
            .map(|bi| BlockQ4_0 {
                d: f16::from_f32(bi as f32 + 1.0),
                qs: [(bi as u8) * 17; 16],
            })
            .collect();

        let out = repack_q4_0_to_q4x4x2_matrix(&blocks, n_rows, k);
        assert_eq!(out.len(), 144);

        // ── quants region: offset 0, 128 B ──────────────────────────────────
        //
        // unpack: blk.qs[j] = bi*17 → x0 = (bi*17)&0x0F = bi, x1 = (bi*17)>>4 = bi.
        // qs_unpacked[bi*32 + j]      = bi  (lower nibble, j in [0..16))
        // qs_unpacked[bi*32 + j + 16] = bi  (upper nibble)
        // → 즉 group 256 elem 중 block bi 가 차지하는 32 elem 은 전부 값 bi.
        //
        // repack quants: y[j] = (qs_unpacked[j+128] << 4) | qs_unpacked[j], j in [0..128).
        // qs_unpacked index e (0..256) 는 block e/32 에 속함 → 값 = e/32.
        // y[j] = ((((j+128)/32) as u8) << 4) | ((j/32) as u8).
        for j in 0..128usize {
            let lo = (j / 32) as u8; // block of low elem
            let hi = ((j + 128) / 32) as u8; // block of high elem
            let expected = (hi << 4) | lo;
            assert_eq!(out[j], expected, "quants byte {j} mismatch");
        }

        // ── scales region: offset K/2 = 128, 16 B (8 × f16) ─────────────────
        //
        // d_off = qrow_size(=128) + 0(group). block bi 의 d_bits (little-endian).
        let qrow_size = k / 2; // 128
        for bi in 0..8usize {
            let d_bits = f16::to_bits(f16::from_f32(bi as f32 + 1.0));
            let lo = (d_bits & 0xFF) as u8;
            let hi = ((d_bits >> 8) & 0xFF) as u8;
            assert_eq!(out[qrow_size + bi * 2], lo, "scale lo byte block {bi}");
            assert_eq!(out[qrow_size + bi * 2 + 1], hi, "scale hi byte block {bi}");
        }
    }
}
