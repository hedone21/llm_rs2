//! DType → `KVLayoutDesc` 브리지 + descriptor-구동 generic block-quant unpacker (ADR-0005 M-F3, D5).
//!
//! 설계 SSOT: `docs/adr/0005-format-backend-capability-plugin-unification.md` D5(generic floor).
//!
//! D5 의 generic floor 는 hot format(Q4_0 등) 특화 arm 밖의 dtype 를 **dequant→f32 matmul**
//! 로 처리한다. 그 dequant 는 per-dtype if-else 의 단순 relocation 이 아니라, descriptor 어휘
//! (`block_elems`/`bits`/`scale_layout`/`packing`)로 구동되는 **family-generic block-quant
//! unpacker** 다(D5). 단 **byte-exact** 가 최우선 — 결과는 `quant.rs` 의 per-block
//! `dequantize()` 와 bit-identical 해야 한다(아래 테스트로 강제).
//!
//! 어휘 밖(mxfp4 shared-exponent·codebook·sparse)은 floor 밖 escape — 여기 None 을 반환하고
//! 호출부(backend dispatch)가 loud-fail 한다.

use crate::buffer::DType;
use crate::memory::host::shared::SharedBuffer;
use crate::shape::Shape;
use crate::tensor::Tensor;
use anyhow::{Result, anyhow};
use half::f16;
use std::sync::Arc;
use technique_api::{KVLayoutDesc, Packing, ScaleLayout};

/// `DType` → block-quant family descriptor 도출(ADR-0005 D5 어휘).
///
/// block-quant family(q4_0/q4_1/q8_0) 와 raw(f32/f16/bf16)만 표현 가능.
/// matmul 부적합 dtype(U8 등)은 `None` — 호출부가 loud-fail 한다(floor 밖 escape).
pub fn dtype_to_layout_desc(d: DType) -> Option<KVLayoutDesc> {
    Some(match d {
        DType::Q8_0 => KVLayoutDesc {
            block_elems: 32,
            bits: 8,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Byte,
        },
        DType::Q4_0 => KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        },
        DType::Q4_1 => KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16WithMin,
            packing: Packing::Nibble,
        },
        DType::F16 => KVLayoutDesc {
            block_elems: 1,
            bits: 16,
            scale_layout: ScaleLayout::None,
            packing: Packing::Dense,
        },
        DType::F32 => KVLayoutDesc {
            block_elems: 1,
            bits: 32,
            scale_layout: ScaleLayout::None,
            packing: Packing::Dense,
        },
        DType::BF16 => KVLayoutDesc {
            block_elems: 1,
            bits: 16,
            scale_layout: ScaleLayout::None,
            packing: Packing::Dense,
        },
        // matmul 부적합 dtype → floor 밖 escape.
        DType::U8 => return None,
    })
}

/// 한 block-quant 블록을 descriptor 어휘로 구동해 f32 로 unpack(canonical layout, llama.cpp 호환).
///
/// canonical block layout = `[f16 scale][f16 min?][packed quants]`:
/// - `PerBlockF16`        → `[scale][quants]`           (q4_0/q8_0)
/// - `PerBlockF16WithMin` → `[scale][min][quants]`      (q4_1)
///
/// nibble(4-bit) interleave 규약(llama.cpp): low nibble → `out[i]`, high nibble → `out[i + n/2]`.
/// nibble signed/unsigned 는 scale_layout 로 갈린다:
/// - min 없음(`PerBlockF16`)        → signed, zero-point = 2^(bits-1)  (q4_0: nibble−8)
/// - min 있음(`PerBlockF16WithMin`) → unsigned, dequant = quant·scale + min  (q4_1)
///
/// byte(8-bit) packing 은 signed i8 직저장(q8_0).
///
/// `block` 은 정확히 한 블록의 raw 바이트(scale/min/quants 포함). `out` 길이 = `block_elems`.
fn unpack_block_via_descriptor(desc: &KVLayoutDesc, block: &[u8], out: &mut [f32]) {
    let n = desc.block_elems as usize;
    debug_assert_eq!(out.len(), n);

    // scale (f16, little-endian) at offset 0.
    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();

    let (min, quants) = match desc.scale_layout {
        ScaleLayout::PerBlockF16WithMin => {
            let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
            (m, &block[4..])
        }
        ScaleLayout::PerBlockF16 => (0.0f32, &block[2..]),
        // raw(Dense) 포맷은 이 함수 경로로 오지 않는다(아래 dequant_via_descriptor 가 직접 처리).
        ScaleLayout::None => (0.0f32, &block[2..]),
    };

    match desc.packing {
        Packing::Nibble => {
            // half = n/2 packed bytes; low nibble → out[i], high nibble → out[i + half].
            let half = n / 2;
            let signed = matches!(desc.scale_layout, ScaleLayout::PerBlockF16);
            for i in 0..half {
                let b = quants[i];
                let lo = (b & 0x0F) as i32;
                let hi = (b >> 4) as i32;
                if signed {
                    // q4_0: zero-point = 2^(bits-1) = 8.
                    let zp = 1i32 << (desc.bits - 1);
                    out[i] = (lo - zp) as f32 * scale;
                    out[i + half] = (hi - zp) as f32 * scale;
                } else {
                    // q4_1: unsigned quant · scale + min.
                    out[i] = lo as f32 * scale + min;
                    out[i + half] = hi as f32 * scale + min;
                }
            }
        }
        Packing::Byte => {
            // q8_0: signed i8 · scale.
            for (i, o) in out.iter_mut().enumerate().take(n) {
                let q = quants[i] as i8 as i32;
                *o = q as f32 * scale;
            }
        }
        // Dense 는 이 함수로 오지 않는다(raw 경로는 dequant_via_descriptor 가 직접 처리).
        Packing::Dense => {
            debug_assert!(
                false,
                "Dense packing must not reach unpack_block_via_descriptor"
            );
        }
    }
}

/// descriptor-구동 generic dequant: weight tensor `b` 를 f32 `Vec<f32>` 로 unpack(D5 floor).
///
/// `b.dtype()` 에서 descriptor 를 도출하고, block-quant family 면 canonical layout 으로
/// block 단위 unpack 한다(`unpack_block_via_descriptor`). raw(f32/f16/bf16)는 직접 변환.
/// descriptor 도출 불가(`None`)면 loud-fail `Err`(floor 밖 escape — 기존 "Unsupported" 메시지 보존).
///
/// 출력은 row-major f32 (b 와 동일 element 순서, 총 `b.numel()` 개).
pub fn dequant_via_descriptor(b: &Tensor) -> Result<Vec<f32>> {
    let dtype = b.dtype();
    let desc = dtype_to_layout_desc(dtype)
        .ok_or_else(|| anyhow!("Unsupported dtype for matmul (no layout descriptor): {dtype:?}"))?;

    let numel = b.numel();

    // raw(Dense) 포맷 — 직접 변환(블록 unpack 불필요).
    if matches!(desc.packing, Packing::Dense) {
        return Ok(match dtype {
            DType::F32 => b.as_slice::<f32>()[..numel].to_vec(),
            DType::F16 => b.as_slice::<f16>()[..numel]
                .iter()
                .map(|x| x.to_f32())
                .collect(),
            DType::BF16 => b.as_slice::<half::bf16>()[..numel]
                .iter()
                .map(|x| x.to_f32())
                .collect(),
            _ => unreachable!("Dense packing implies raw dtype"),
        });
    }

    // block-quant family — block 단위 unpack.
    let block_elems = desc.block_elems as usize;
    if !numel.is_multiple_of(block_elems) {
        return Err(anyhow!(
            "matmul weight numel {numel} not a multiple of block_elems {block_elems} for {dtype:?}"
        ));
    }
    let n_blocks = numel / block_elems;

    // 블록 raw 바이트 크기: scale(2) + min(2?) + quants.
    let quant_bytes = match desc.packing {
        Packing::Nibble => block_elems / 2,
        Packing::Byte => block_elems,
        Packing::Dense => unreachable!(),
    };
    let scale_bytes = match desc.scale_layout {
        ScaleLayout::PerBlockF16 => 2,
        ScaleLayout::PerBlockF16WithMin => 4,
        ScaleLayout::None => 0,
    };
    let block_bytes = scale_bytes + quant_bytes;

    // b 의 raw 바이트(packed block 연속).
    let raw = b.as_slice::<u8>();
    if raw.len() < n_blocks * block_bytes {
        return Err(anyhow!(
            "matmul weight raw bytes {} < expected {} ({} blocks × {} bytes) for {dtype:?}",
            raw.len(),
            n_blocks * block_bytes,
            n_blocks,
            block_bytes
        ));
    }

    let mut out = vec![0.0f32; numel];
    for bi in 0..n_blocks {
        let byte_off = bi * block_bytes;
        let elem_off = bi * block_elems;
        unpack_block_via_descriptor(
            &desc,
            &raw[byte_off..byte_off + block_bytes],
            &mut out[elem_off..elem_off + block_elems],
        );
    }
    Ok(out)
}

/// generic floor 의 임시 f32 weight tensor 생성(ADR-0005 D5).
///
/// `b` 를 [`dequant_via_descriptor`] 로 f32 로 푼 뒤, **b 와 동일 shape**·dtype=F32 인 임시
/// Tensor 를 만든다. backend 의 floor arm 이 이걸 `matmul_transposed_f32(a, &f32_b, out)` 에
/// 넘긴다 — dequant 무손실 + f32 matmul 동일이라 **exact**(느릴 뿐).
///
/// backend 는 인자로 받지 않고 CPU scalar backend 를 붙인다(matmul_transposed_f32 는 backend
/// 인스턴스를 쓰지 않고 raw f32 slice 만 읽으므로 무해).
pub fn dequant_to_f32_tensor(b: &Tensor) -> Result<Tensor> {
    let f32_data = dequant_via_descriptor(b)?;
    // Vec<f32> → Vec<u8> (canonical little-endian, host-native; matmul 은 같은 호스트에서 읽음).
    let mut bytes = Vec::<u8>::with_capacity(f32_data.len() * 4);
    for v in &f32_data {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    let buf = SharedBuffer::from_vec(bytes, DType::F32);
    Ok(Tensor::new(
        Shape::new(b.shape().dims().to_vec()),
        Arc::new(buf),
        b.backend().clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{BlockQ4_0, BlockQ8_0, QK4_0, QK8_0};

    // ── (a) generic unpacker bit-identical vs quant.rs dequantize ──

    /// generic unpacker 의 Q8_0 unpack == BlockQ8_0::dequantize(), 요소별 동일.
    #[test]
    fn test_generic_unpack_q8_0_bit_identical() {
        let desc = dtype_to_layout_desc(DType::Q8_0).unwrap();
        // 고정 + 의사난수 블록 여러 개.
        let cases: Vec<BlockQ8_0> = vec![
            BlockQ8_0 {
                d: f16::from_f32(0.0),
                qs: [0; QK8_0],
            },
            BlockQ8_0 {
                d: f16::from_f32(0.125),
                qs: std::array::from_fn(|i| (i as i32 - 16) as i8),
            },
            BlockQ8_0 {
                d: f16::from_f32(-3.5),
                qs: std::array::from_fn(|i| ((i as i32 * 7 + 3) % 256 - 128) as i8),
            },
            BlockQ8_0 {
                d: f16::from_f32(2.0e-3),
                qs: [127; QK8_0],
            },
            BlockQ8_0 {
                d: f16::from_f32(64.0),
                qs: [-128; QK8_0],
            },
        ];
        for blk in &cases {
            // reference
            let mut want = [0.0f32; QK8_0];
            blk.dequantize(&mut want);
            // generic
            let raw: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    (blk as *const BlockQ8_0) as *const u8,
                    std::mem::size_of::<BlockQ8_0>(),
                )
            };
            let mut got = [0.0f32; QK8_0];
            unpack_block_via_descriptor(&desc, raw, &mut got);
            assert_eq!(got, want, "Q8_0 generic unpack mismatch");
        }
    }

    /// generic unpacker 의 Q4_0 unpack == BlockQ4_0::dequantize(), 요소별 동일.
    #[test]
    fn test_generic_unpack_q4_0_bit_identical() {
        let desc = dtype_to_layout_desc(DType::Q4_0).unwrap();
        let cases: Vec<BlockQ4_0> = vec![
            BlockQ4_0 {
                d: f16::from_f32(0.0),
                qs: [0; QK4_0 / 2],
            },
            BlockQ4_0 {
                d: f16::from_f32(0.5),
                qs: [0x55; QK4_0 / 2],
            },
            BlockQ4_0 {
                d: f16::from_f32(-1.25),
                qs: std::array::from_fn(|i| ((i * 13 + 1) % 256) as u8),
            },
            BlockQ4_0 {
                d: f16::from_f32(7.0),
                qs: [0xFF; QK4_0 / 2],
            },
            BlockQ4_0 {
                d: f16::from_f32(0.03125),
                qs: std::array::from_fn(|i| (i * 17) as u8),
            },
        ];
        for blk in &cases {
            let mut want = [0.0f32; QK4_0];
            blk.dequantize(&mut want);
            let raw: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    (blk as *const BlockQ4_0) as *const u8,
                    std::mem::size_of::<BlockQ4_0>(),
                )
            };
            let mut got = [0.0f32; QK4_0];
            unpack_block_via_descriptor(&desc, raw, &mut got);
            assert_eq!(got, want, "Q4_0 generic unpack mismatch");
        }
    }

    // ── descriptor 어휘 sanity ──

    #[test]
    fn test_dtype_layout_desc_vocabulary() {
        assert_eq!(
            dtype_to_layout_desc(DType::Q8_0).unwrap(),
            KVLayoutDesc {
                block_elems: 32,
                bits: 8,
                scale_layout: ScaleLayout::PerBlockF16,
                packing: Packing::Byte,
            }
        );
        assert_eq!(
            dtype_to_layout_desc(DType::Q4_1).unwrap(),
            KVLayoutDesc {
                block_elems: 32,
                bits: 4,
                scale_layout: ScaleLayout::PerBlockF16WithMin,
                packing: Packing::Nibble,
            }
        );
        // matmul 부적합 dtype → floor 밖 escape.
        assert!(dtype_to_layout_desc(DType::U8).is_none());
    }
}
