//! `OpaqueKvFormat` — `DType`-우회 opaque KV format 의 `KVCacheFormat` impl (ADR-0007 G6).
//!
//! 설계 SSOT: `docs/adr/0007-opaque-dtype-kv-format-unlock.md`.
//!
//! 새 KV format(`.so` block-quant family)이 **새 `DType` variant 없이** 동작함을 증명하는
//! production 계열 impl. `StandardFormat`(KVCache 재사용)과 달리 **자체 최소 저장**([`OpaqueBuffer`]
//! 2개)을 들고, 세 책임을 직접 수행한다:
//! - **write(quant)** = `encode_via_descriptor`(G4, 코드-구동 encoder) + head-major block scatter,
//! - **attention(read)** = `dequant_to_f32_tensor`(G3, descriptor floor) → `backend.attention_gen`
//!   (기존 CPU F32 attention 재사용 — read 경로는 데이터-구동),
//! - **geometry** = idx/current_pos/capacity.
//!
//! 이 분담이 ADR-0007 D2(byte/read=데이터-구동 ⊥ write=코드-구동)와 D4(format-bound encoder)를
//! 구현한다. production KVCache.update/grow 의 dtype-binary 분기를 건드리지 않아 Q4_0 회귀 위험 0
//! (외과적). grow/eviction 통합은 production 단계로 deferred — 본 impl 은 capacity pre-size.

use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow, bail};
use technique_api::KVLayoutDesc;

use crate::backend::Backend;
use crate::buffer::opaque::OpaqueBuffer;
use crate::buffer::{Buffer, DType};
use crate::format::{AttnDims, KVCacheFormat, dequant_to_f32_tensor, encode_via_descriptor};
use crate::memory::host::shared::SharedBuffer;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// 내부 가변 상태 — K/V opaque 버퍼(packed block bytes) + 유효 토큰 수.
struct OpaqueInner {
    /// HeadMajor `[1, kv_heads, capacity, head_dim]` opaque 버퍼 (packed block bytes).
    k: Tensor,
    v: Tensor,
    current_pos: usize,
}

/// opaque(block-quant) KV format 의 `KVCacheFormat` impl (ADR-0007 G6).
pub struct OpaqueKvFormat {
    idx: usize,
    desc: KVLayoutDesc,
    kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    /// `head_dim / block_elems` — (head, pos) 당 블록 수.
    blocks_per_pos: usize,
    /// 한 블록의 raw 바이트.
    block_bytes: usize,
    inner: Mutex<OpaqueInner>,
}

impl OpaqueKvFormat {
    /// capacity pre-sized opaque KV format 을 생성. `head_dim` 은 `desc.block_elems` 배수여야 한다.
    /// `desc` 는 block-quant family(raw/Dense 불가).
    pub fn new(
        idx: usize,
        desc: KVLayoutDesc,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        let block_elems = desc.block_elems as usize;
        if block_elems == 0 || !head_dim.is_multiple_of(block_elems) {
            bail!(
                "OpaqueKvFormat: head_dim {head_dim} not a multiple of block_elems {block_elems}"
            );
        }
        let block_bytes = desc
            .block_bytes()
            .ok_or_else(|| anyhow!("OpaqueKvFormat: block-quant descriptor 필요(Dense 불가)"))?;
        let blocks_per_pos = head_dim / block_elems;

        let numel = kv_heads * capacity * head_dim;
        let nbytes = desc
            .bytes_for_elems(numel)
            .ok_or_else(|| anyhow!("OpaqueKvFormat: bytes_for_elems({numel}) 실패"))?;
        let shape = Shape::new(vec![1, kv_heads, capacity, head_dim]);
        let mk = || -> Tensor {
            let inner_buf: Arc<dyn Buffer> = Arc::new(SharedBuffer::new(nbytes, DType::U8));
            let op: Arc<dyn Buffer> = Arc::new(OpaqueBuffer::new(inner_buf, desc));
            Tensor::new(shape.clone(), op, backend.clone())
        };

        Ok(Self {
            idx,
            desc,
            kv_heads,
            head_dim,
            capacity,
            blocks_per_pos,
            block_bytes,
            inner: Mutex::new(OpaqueInner {
                k: mk(),
                v: mk(),
                current_pos: 0,
            }),
        })
    }

    /// 한 토큰(`kv_heads * head_dim` f32, head-major)을 `pos` 에 encode+scatter.
    /// byte offset = `(h*capacity + pos) * blocks_per_pos * block_bytes` (HeadMajor block layout).
    fn scatter_token(&self, buf: &Tensor, pos: usize, token_f32: &[f32]) -> Result<()> {
        let bytes_per_pos = self.blocks_per_pos * self.block_bytes;
        let total = buf.buffer().size();
        // SAFETY: buf 의 OpaqueBuffer 는 total 바이트 유효(self 수명 동안 Arc 보유). 각 (h,pos)
        // sub-slice 는 non-overlapping. interior-mut 는 엔진 Buffer 관용구(SharedBuffer::as_mut_ptr).
        let all: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(buf.buffer().as_mut_ptr(), total) };
        for h in 0..self.kv_heads {
            let src = &token_f32[h * self.head_dim..(h + 1) * self.head_dim];
            let off = (h * self.capacity + pos) * bytes_per_pos;
            encode_via_descriptor(&self.desc, src, &mut all[off..off + bytes_per_pos])?;
        }
        Ok(())
    }
}

impl KVCacheFormat for OpaqueKvFormat {
    fn idx(&self) -> usize {
        self.idx
    }

    fn current_pos(&self) -> usize {
        self.inner.lock().unwrap().current_pos
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn write_kv(&self, new_k: &Tensor, new_v: &Tensor, _backend: &dyn Backend) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        let pos = inner.current_pos;
        if pos >= self.capacity {
            bail!(
                "OpaqueKvFormat: capacity {} 초과 (GATE-B pre-size, grow deferred)",
                self.capacity
            );
        }
        self.scatter_token(&inner.k, pos, new_k.as_slice::<f32>())?;
        self.scatter_token(&inner.v, pos, new_v.as_slice::<f32>())?;
        inner.current_pos += 1;
        Ok(())
    }

    fn write_kv_batch(&self, new_k: &Tensor, new_v: &Tensor, _backend: &dyn Backend) -> Result<()> {
        let seq_len = new_k.shape().dims()[1];
        let stride = self.kv_heads * self.head_dim;
        let kf = new_k.as_slice::<f32>();
        let vf = new_v.as_slice::<f32>();
        let mut inner = self.inner.lock().unwrap();
        for t in 0..seq_len {
            let pos = inner.current_pos + t;
            if pos >= self.capacity {
                bail!(
                    "OpaqueKvFormat: capacity {} 초과 (GATE-B pre-size, grow deferred)",
                    self.capacity
                );
            }
            self.scatter_token(&inner.k, pos, &kf[t * stride..(t + 1) * stride])?;
            self.scatter_token(&inner.v, pos, &vf[t * stride..(t + 1) * stride])?;
        }
        inner.current_pos += seq_len;
        Ok(())
    }

    fn attention_into(
        &self,
        q: &Tensor,
        backend: &dyn Backend,
        out: &mut Tensor,
        dims: AttnDims,
        scores: Option<&mut [f32]>,
    ) -> Result<()> {
        let inner = self.inner.lock().unwrap();
        let cache_seq_len = inner.current_pos;
        let seq_len = q.shape().dims()[1];

        // read 경로 = 데이터-구동 floor: opaque 버퍼를 descriptor 로 f32 unpack(G3) 후 기존 CPU
        // F32 attention 재사용. 새 backend 메서드·dtype-binary 분기 0(ADR-0007 D2).
        let k_f32 = dequant_to_f32_tensor(&inner.k)?;
        let v_f32 = dequant_to_f32_tensor(&inner.v)?;

        if seq_len > 1 {
            // prefill: causal multi-token. StandardFormat 의 prefill_attention 재사용(DRY).
            let batch_size = q.shape().dims()[0];
            let q_start_pos = cache_seq_len - seq_len;
            let _ = scores; // prefill 은 score 누적 안 함(StandardFormat 동일).
            return crate::pressure::standard_format::prefill_attention(
                q,
                out,
                &k_f32,
                &v_f32,
                dims.n_heads_q,
                self.kv_heads,
                self.head_dim,
                seq_len,
                cache_seq_len,
                self.capacity,
                batch_size,
                crate::kv_cache_ops::KVLayout::HeadMajor,
                q_start_pos,
                dims.window,
                backend,
            );
        }

        // decode (seq_len==1): single-query attention (StandardFormat typed 경로 동형).
        let effective = match dims.window {
            Some(w) => cache_seq_len.min(w),
            None => cache_seq_len,
        };
        backend.attention_gen(
            q,
            &k_f32,
            &v_f32,
            out,
            dims.n_heads_q,
            self.kv_heads,
            self.head_dim,
            effective,
            scores,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::quant::{BlockQ4_0, QK4_0};
    use technique_api::{Packing, ScaleLayout};

    fn f32_tensor(dims: Vec<usize>, data: &[f32], backend: &Arc<dyn Backend>) -> Tensor {
        let buf: Arc<dyn Buffer> = Arc::new(SharedBuffer::new(data.len() * 4, DType::F32));
        let mut t = Tensor::new(Shape::new(dims), buf, backend.clone());
        t.as_mut_slice::<f32>().copy_from_slice(data);
        t
    }

    fn synth_q4_desc() -> KVLayoutDesc {
        KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        }
    }

    /// q4_0 round-trip(quantize→dequantize)을 HeadMajor f32 reference 로 만든다.
    fn q4_0_roundtrip_block(src: &[f32]) -> [f32; QK4_0] {
        let mut a = [0.0f32; QK4_0];
        a.copy_from_slice(src);
        let mut o = [0.0f32; QK4_0];
        BlockQ4_0::quantize(&a).dequantize(&mut o);
        o
    }

    /// ADR-0007 GATE-B: DType variant 없는 opaque format(synth_q4 layout)의 write_kv(encode+scatter)
    /// + attention_into(dequant floor → F32 attention)가 동일 데이터의 q4_0 round-trip baseline 과
    /// **bit-identical**. write_kv 의 scatter geometry 와 read floor 의 통합을 검증한다.
    #[test]
    fn opaque_kv_format_decode_bit_identical_to_q4_0_roundtrip() {
        let kv_heads = 2usize;
        let head_dim = 64usize; // 2 blocks/head (block_elems=32)
        let n_heads_q = 2usize; // GQA ratio 1
        let n_tokens = 5usize;
        let capacity = 8usize; // pre-sized > n_tokens
        let desc = synth_q4_desc();
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());

        let gen_val = |t: usize, h: usize, d: usize, salt: f32| -> f32 {
            (((t * 7 + h * 13 + d * 3) % 17) as f32 - 8.0) * 0.1 + salt
        };
        let mut k_tokens: Vec<Vec<f32>> = Vec::new();
        let mut v_tokens: Vec<Vec<f32>> = Vec::new();
        for t in 0..n_tokens {
            let mut k = vec![0.0f32; kv_heads * head_dim];
            let mut v = vec![0.0f32; kv_heads * head_dim];
            for h in 0..kv_heads {
                for d in 0..head_dim {
                    k[h * head_dim + d] = gen_val(t, h, d, 0.0);
                    v[h * head_dim + d] = gen_val(t, h, d, 0.5);
                }
            }
            k_tokens.push(k);
            v_tokens.push(v);
        }

        // ── opaque path: write_kv(encode+scatter) × n_tokens → attention_into(dequant floor) ──
        let fmt =
            OpaqueKvFormat::new(0, desc, kv_heads, head_dim, capacity, backend.clone()).unwrap();
        assert_eq!(fmt.idx(), 0);
        assert_eq!(fmt.capacity(), capacity);
        for t in 0..n_tokens {
            let kt = f32_tensor(vec![1, 1, kv_heads, head_dim], &k_tokens[t], &backend);
            let vt = f32_tensor(vec![1, 1, kv_heads, head_dim], &v_tokens[t], &backend);
            fmt.write_kv(&kt, &vt, backend.as_ref()).unwrap();
        }
        assert_eq!(fmt.current_pos(), n_tokens);

        let q_data: Vec<f32> = (0..n_heads_q * head_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.07)
            .collect();
        let q = f32_tensor(vec![1, 1, n_heads_q, head_dim], &q_data, &backend);
        let mut out_opaque = f32_tensor(
            vec![1, 1, n_heads_q, head_dim],
            &vec![0.0; n_heads_q * head_dim],
            &backend,
        );
        fmt.attention_into(
            &q,
            backend.as_ref(),
            &mut out_opaque,
            AttnDims {
                n_heads_q,
                window: None,
            },
            None,
        )
        .unwrap();

        // ── reference: q4_0 round-trip(quantize→dequantize) HeadMajor + F32 attention_gen ──
        let mut ref_k = vec![0.0f32; kv_heads * capacity * head_dim];
        let mut ref_v = vec![0.0f32; kv_heads * capacity * head_dim];
        for (t, (kt, vt)) in k_tokens.iter().zip(v_tokens.iter()).enumerate() {
            for h in 0..kv_heads {
                for blk in 0..(head_dim / QK4_0) {
                    let lo = h * head_dim + blk * QK4_0;
                    let kc = q4_0_roundtrip_block(&kt[lo..lo + QK4_0]);
                    let vc = q4_0_roundtrip_block(&vt[lo..lo + QK4_0]);
                    let base = (h * capacity + t) * head_dim + blk * QK4_0;
                    ref_k[base..base + QK4_0].copy_from_slice(&kc);
                    ref_v[base..base + QK4_0].copy_from_slice(&vc);
                }
            }
        }
        let ref_k_t = f32_tensor(vec![1, kv_heads, capacity, head_dim], &ref_k, &backend);
        let ref_v_t = f32_tensor(vec![1, kv_heads, capacity, head_dim], &ref_v, &backend);
        let mut out_ref = f32_tensor(
            vec![1, 1, n_heads_q, head_dim],
            &vec![0.0; n_heads_q * head_dim],
            &backend,
        );
        backend
            .attention_gen(
                &q,
                &ref_k_t,
                &ref_v_t,
                &mut out_ref,
                n_heads_q,
                kv_heads,
                head_dim,
                n_tokens,
                None,
            )
            .unwrap();

        assert_eq!(
            out_opaque.as_slice::<f32>(),
            out_ref.as_slice::<f32>(),
            "opaque KV attention != q4_0 round-trip baseline (bit-identical)"
        );
    }

    /// prefill(write_kv_batch + seq_len>1 attention_into) smoke — 유한 출력 + current_pos 일치.
    #[test]
    fn opaque_kv_format_prefill_smoke() {
        let kv_heads = 1usize;
        let head_dim = 32usize; // 1 block/head
        let n_heads_q = 1usize;
        let seq = 4usize;
        let capacity = 8usize;
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let fmt = OpaqueKvFormat::new(
            0,
            synth_q4_desc(),
            kv_heads,
            head_dim,
            capacity,
            backend.clone(),
        )
        .unwrap();

        let kb: Vec<f32> = (0..seq * kv_heads * head_dim)
            .map(|i| (i as f32 % 5.0) - 2.0)
            .collect();
        let vb: Vec<f32> = (0..seq * kv_heads * head_dim)
            .map(|i| (i as f32 % 3.0) - 1.0)
            .collect();
        let kt = f32_tensor(vec![1, seq, kv_heads, head_dim], &kb, &backend);
        let vt = f32_tensor(vec![1, seq, kv_heads, head_dim], &vb, &backend);
        fmt.write_kv_batch(&kt, &vt, backend.as_ref()).unwrap();
        assert_eq!(fmt.current_pos(), seq);

        let q = f32_tensor(
            vec![1, seq, n_heads_q, head_dim],
            &vec![0.3f32; seq * n_heads_q * head_dim],
            &backend,
        );
        let mut out = f32_tensor(
            vec![1, seq, n_heads_q * head_dim],
            &vec![0.0; seq * n_heads_q * head_dim],
            &backend,
        );
        fmt.attention_into(
            &q,
            backend.as_ref(),
            &mut out,
            AttnDims {
                n_heads_q,
                window: None,
            },
            None,
        )
        .unwrap();
        for &x in out.as_slice::<f32>() {
            assert!(x.is_finite(), "prefill attention 출력은 유한해야 한다");
        }
    }
}
