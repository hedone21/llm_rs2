//! `KIVIFormat` — `KVCacheFormat` impl wrapping a `KiviCache` (§4.1, Phase α-K).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §4.1 (R4 ④ KIVI creep 제거 + AWQE 자가 흡수).
//!
//! **purely additive, host-only, unwired** — 기존 `KiviCache` 와 `KVCacheOps` 경로를 1바이트도
//! 건드리지 않고, 신규 wrapper 로 공존한다. production 에서 `KIVIFormat` 를 생성하는 코드는 0
//! (unit test 에서만 생성). 내부 가변성 = `std::sync::Mutex`.
//!
//! `attention_into` 는 kivi-native(GPU fused dequant) 와 fallback(F32 view →
//! `backend.attention_gen`) 에 더해 AWQE 자가 흡수(scores `Some` 일 때 내부
//! `KiviCache.set_attn_scores` 로 자가 기록)를 수행한다. base trait 에 `needs_attn_scores`
//! 메서드를 만들지 않는다(§4.1 R4 ③).

use std::sync::Mutex;

use anyhow::Result;

use crate::backend::Backend;
use crate::format::{AttnDims, KVCacheFormat, Merge};
use crate::pressure::kivi_cache::KiviCache;
use crate::tensor::Tensor;

/// KIVI (Q2 + residual) KV cache 를 `KVCacheFormat` 으로 노출하는 wrapper.
///
/// 기존 `KiviCache` 를 `Mutex` 로 감싸 `&self` 메서드에서 내부 `&mut` 메서드에 위임한다.
/// `KiviCache` 자체는 무변.
pub struct KIVIFormat {
    idx: usize,
    inner: Mutex<KiviCache>,
}

impl KIVIFormat {
    /// `KiviCache` 를 layer 인덱스와 함께 wrapping. (현재 unit test 전용 — unwired.)
    pub fn new(idx: usize, inner: KiviCache) -> Self {
        Self {
            idx,
            inner: Mutex::new(inner),
        }
    }

    /// KV write 흡수 — `KiviCache` 는 CPU-only(`get_buffers_mut`==None) 라 GPU scatter fast-path
    /// 대상이 아니다. 구 `update_kv_cache`(transformer_layer.rs:31) 의 CPU-only 분기를 옮긴 것:
    /// producer tensor 가 host-mapped GPU 메모리(non-null ptr)면 device 커널 완료 전 stale read
    /// 방지를 위해 `synchronize` 후 `KiviCache::update`(Q2 quant + residual append 자체 수행) 호출.
    ///
    /// decode/prefill 동일 경로(`KiviCache::update` 가 seq_len 으로 분기). device-only producer
    /// (`as_ptr()` null)의 명시적 readback 은 후속 device substep 으로 연기(host 미발생).
    fn write_inner(&self, new_k: &Tensor, new_v: &Tensor, backend: &dyn Backend) -> Result<()> {
        let mut cache = self.inner.lock().unwrap();
        if !new_k.as_ptr().is_null() {
            backend.synchronize()?;
        }
        cache.update(new_k, new_v)
    }

    /// wrapping 을 해제하고 내부 `KiviCache` 를 반환 (Phase α-K ①-c eval transient-wrap round-trip).
    ///
    /// `StandardFormat::into_inner` 대칭. eval 이 forward 1회 동안만 `Vec<KiviCache>` →
    /// `Arc<KIVIFormat>` 로 wrap 후 `Arc::try_unwrap().into_inner()` 로 복귀시킨다. base trait 무변.
    pub(crate) fn into_inner(self) -> KiviCache {
        self.inner.into_inner().unwrap()
    }
}

impl KVCacheFormat for KIVIFormat {
    fn idx(&self) -> usize {
        self.idx
    }

    fn current_pos(&self) -> usize {
        self.inner.lock().unwrap().current_pos()
    }

    fn capacity(&self) -> usize {
        self.inner.lock().unwrap().capacity()
    }

    fn write_kv(&self, new_k: &Tensor, new_v: &Tensor, backend: &dyn Backend) -> Result<()> {
        self.write_inner(new_k, new_v, backend)
    }

    fn write_kv_batch(&self, new_k: &Tensor, new_v: &Tensor, backend: &dyn Backend) -> Result<()> {
        self.write_inner(new_k, new_v, backend)
    }

    fn compact(&self, _keep: &[usize], _merges: &[Merge]) -> Result<()> {
        // KIVI 는 eviction-by-compaction 을 지원하지 않는다(quantized Q2 블록은 토큰 단위 재배치가
        // 불가; 현 `KiviCache` 에 compact 경로 부재). position 은 q2_tokens + res_pos 로 파생되어
        // set_current_pos 도 no-op. 따라서 본 substep 에서 compact 는 no-op(KIVI 는 ratio-driven
        // eviction 대상이 아니라 quantization 압축 자체가 메모리 정책 — MEMORY.md 참조).
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
        let mut cache = self.inner.lock().unwrap();
        let n_heads_q = dims.n_heads_q;

        // ── prefill (seq_len>1): multi-token causal attention (Phase α-K ①-e) ──
        // KIVI 는 multi-token prefill native 커널 부재(attention_gen / attention_native 는 single-query
        // decode 전용 — causal-mask 없음)라, dequantized view(get_view) + StandardFormat 의
        // `prefill_attention`(free fn, pub(crate)) 재사용으로 처리한다(DRY). OLD generic
        // `forward_prefill<C>`(forward.rs:251-585)의 KIVI 경로(get_view → flash_attention_prefill /
        // flash_attention_forward_strided)와 bit-identical: KIVI CPU(SeqMajor F32) / GPU(bits=16
        // HeadMajor, bits 2/4/8 assembled) 모두 `kv_layout`/`kv_capacity` 인자로 분기된다.
        // `q_start_pos = cache_seq_len - seq_len`(= forward_prefill 의 start_pos, write 후 불변식).
        // prefill 은 score 누적 안 함(forward_prefill 의 `_need_scores` 동일) → `scores` 무시.
        let seq_len = q.shape().dims()[1];
        if seq_len > 1 {
            let n_heads_kv = cache.kv_heads();
            let head_dim = cache.head_dim();
            let kv_capacity = cache.capacity();
            let kv_layout = cache.layout();
            let cache_seq_len = cache.current_pos();
            let batch_size = q.shape().dims()[0];
            let q_start_pos = cache_seq_len - seq_len;
            let (k_cache, v_cache) = cache.get_view();
            let _ = scores;
            return crate::pressure::standard_format::prefill_attention(
                q,
                out,
                &k_cache,
                &v_cache,
                n_heads_q,
                n_heads_kv,
                head_dim,
                seq_len,
                cache_seq_len,
                kv_capacity,
                batch_size,
                kv_layout,
                q_start_pos,
                dims.window,
                backend,
            );
        }

        // kivi-native 경로 게이팅(host 미검증 — device 검증은 후속 substep). 게이팅 조건만 미리
        // 평가하고(borrow 분리), dispatch 는 별도 헬퍼에 위임해 scores ownership 을 단일 경로로 가둔다.
        // get_kivi_raw_buffers 가 Some + backend 가 KiviAttentionBackend + has_kivi_attn_kernel +
        // is_nosub_device(NVIDIA) + 토큰 존재 일 때만. Adreno(subgroup)는 F32 dequant 경로가 더
        // 빠르므로 native 미사용(forward_gen 의 기존 게이팅 보존).
        let use_native = backend.is_gpu()
            && backend
                .as_kivi_attention()
                .zip(cache.get_kivi_raw_buffers())
                .map(|(kivi_be, raw)| {
                    kivi_be.has_kivi_attn_kernel(raw.bits)
                        && kivi_be.is_nosub_device()
                        && (raw.q_tokens + raw.res_tokens) > 0
                })
                .unwrap_or(false);

        if use_native {
            return self.attention_native(q, backend, out, n_heads_q, scores, &mut cache);
        }

        // fallback: dequantized F32 view → backend.attention_gen (CPU-testable).
        let n_heads_kv = cache.kv_heads();
        let head_dim = cache.head_dim();
        let (k_cache, v_cache) = cache.get_view();
        let cache_seq_len = cache.current_pos();
        let effective_cache_len = match dims.window {
            Some(w) => cache_seq_len.min(w),
            None => cache_seq_len,
        };

        // attention_gen 에 caller scores 슬라이스를 직접 넘긴 뒤 AWQE 자가 흡수
        // (set_attn_scores 는 awqe_enabled=false 면 자체 no-op, §4.1 R4 ③).
        match scores {
            Some(s) => {
                backend.attention_gen(
                    q,
                    &k_cache,
                    &v_cache,
                    out,
                    n_heads_q,
                    n_heads_kv,
                    head_dim,
                    effective_cache_len,
                    Some(s),
                )?;
                let stride = if n_heads_q == 0 {
                    0
                } else {
                    s.len() / n_heads_q
                };
                cache.set_attn_scores(s, n_heads_q, stride, effective_cache_len);
            }
            None => {
                backend.attention_gen(
                    q,
                    &k_cache,
                    &v_cache,
                    out,
                    n_heads_q,
                    n_heads_kv,
                    head_dim,
                    effective_cache_len,
                    None,
                )?;
            }
        }
        Ok(())
    }
}

impl KIVIFormat {
    /// kivi-native GPU fused dequant+attention dispatch + AWQE 자가 흡수 (§4.1 R4 ④).
    ///
    /// host 미검증(컴파일만) — device 검증은 후속 wiring substep. `scores` 가 `Some` 이면 native
    /// 커널이 임시 버퍼에 쓴 raw post-softmax score 를 caller 슬라이스로 복사 + 내부
    /// `KiviCache::set_attn_scores`(awqe_enabled 게이트 자가 처리)로 흡수한다.
    fn attention_native(
        &self,
        q: &Tensor,
        backend: &dyn Backend,
        out: &mut Tensor,
        n_heads_q: usize,
        scores: Option<&mut [f32]>,
        cache: &mut KiviCache,
    ) -> Result<()> {
        let n_heads_kv = cache.kv_heads();
        let head_dim = cache.head_dim();
        let kivi_be = backend
            .as_kivi_attention()
            .expect("attention_native gated on as_kivi_attention().is_some()");
        let raw = cache
            .get_kivi_raw_buffers()
            .expect("attention_native gated on get_kivi_raw_buffers().is_some()");
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = raw.q_tokens + raw.res_tokens;

        // native 커널 score 임시 버퍼 (caller scores 유무로 게이팅).
        let mut tmp_scores: Vec<f32> = if scores.is_some() {
            vec![0.0; n_heads_q * total]
        } else {
            Vec::new()
        };
        let scores_for_kernel: Option<&mut [f32]> = if scores.is_some() {
            Some(&mut tmp_scores)
        } else {
            None
        };

        kivi_be.attention_gen_kivi(
            q,
            raw.qk_buf,
            raw.qv_buf,
            raw.res_k,
            raw.res_v,
            out,
            n_heads_q,
            n_heads_kv,
            head_dim,
            raw.q_tokens,
            raw.res_tokens,
            raw.res_cap,
            scale,
            scores_for_kernel,
            raw.bits,
        )?;
        // 이후 `raw`(cache immutable borrow) 미사용 → NLL 이 set_attn_scores(가변) 전에 borrow 종료.

        if let Some(dst) = scores {
            let n = tmp_scores.len().min(dst.len());
            dst[..n].copy_from_slice(&tmp_scores[..n]);
            let valid_len = if n_heads_q == 0 {
                0
            } else {
                tmp_scores.len() / n_heads_q
            };
            cache.set_attn_scores(&tmp_scores, n_heads_q, valid_len, valid_len);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    fn f32_tensor(dims: Vec<usize>, data: &[f32]) -> Tensor {
        use crate::buffer::DType;
        let buf = Arc::new(SharedBuffer::new(data.len() * 4, DType::F32));
        let mut t = Tensor::new(Shape::new(dims), buf, Arc::new(CpuBackend::new()));
        t.as_mut_slice::<f32>().copy_from_slice(data);
        t
    }

    // KiviCache 제약: residual_size 와 head_dim 모두 QKKV(=32) 의 배수여야 한다 (kivi_cache.rs:333).
    const HD: usize = 32; // head_dim
    const RES: usize = 32; // residual_size
    const MAXSEQ: usize = 256;

    #[test]
    fn test_geometry_delegates_to_kivicache() {
        // KiviCache CPU mode (bits=2 default).
        let cache = KiviCache::new(2, HD, MAXSEQ, RES);
        let fmt = KIVIFormat::new(5, cache);
        assert_eq!(fmt.idx(), 5);
        assert_eq!(fmt.current_pos(), 0);
        // CPU mode capacity == max_seq_len.
        assert_eq!(fmt.capacity(), MAXSEQ);
    }

    #[test]
    fn test_write_kv_advances_pos() {
        let kv_heads = 2;
        let fmt = KIVIFormat::new(0, KiviCache::new(kv_heads, HD, MAXSEQ, RES));

        let token = vec![1.0f32; kv_heads * HD];
        let k = f32_tensor(vec![1, 1, kv_heads, HD], &token);
        let v = f32_tensor(vec![1, 1, kv_heads, HD], &token);
        fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 1);

        let batch = vec![2.0f32; 2 * kv_heads * HD];
        let kb = f32_tensor(vec![1, 2, kv_heads, HD], &batch);
        let vb = f32_tensor(vec![1, 2, kv_heads, HD], &batch);
        fmt.write_kv_batch(&kb, &vb, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 3);
    }

    #[test]
    fn test_compact_is_noop() {
        let fmt = KIVIFormat::new(0, KiviCache::new(1, HD, MAXSEQ, RES));
        let token = vec![1.0f32; HD];
        let k = f32_tensor(vec![1, 1, 1, HD], &token);
        let v = f32_tensor(vec![1, 1, 1, HD], &token);
        fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 1);
        // compact is a no-op for KIVI; pos unchanged.
        fmt.compact(&[0], &[]).unwrap();
        assert_eq!(fmt.current_pos(), 1);
    }

    #[test]
    fn test_attention_into_cpu_fallback() {
        // CPU KiviCache (bits=2): write 2 tokens, run attention via F32 view fallback.
        // Q2 quantization introduces error, so we only assert the output is finite and
        // bounded by the (positive) V magnitude — the CPU-testable seam is "runs + produces
        // a sane attention output", not bit-exact dequant (that is KIVI's own concern).
        let kv_heads = 1;
        let n_heads_q = 1;
        let fmt = KIVIFormat::new(0, KiviCache::new(kv_heads, HD, MAXSEQ, RES));

        let v_row = vec![3.0f32; HD];
        let k_row = vec![0.0f32; HD]; // zero K → uniform softmax
        for _ in 0..2 {
            let k = f32_tensor(vec![1, 1, kv_heads, HD], &k_row);
            let v = f32_tensor(vec![1, 1, kv_heads, HD], &v_row);
            fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        }
        assert_eq!(fmt.current_pos(), 2);

        let q = f32_tensor(vec![1, 1, n_heads_q, HD], &[1.0; HD]);
        let mut out = f32_tensor(vec![1, 1, n_heads_q, HD], &[0.0; HD]);
        let backend = CpuBackend::new();
        let mut scores = vec![0.0f32; n_heads_q * 2];

        fmt.attention_into(
            &q,
            &backend,
            &mut out,
            AttnDims {
                n_heads_q,
                window: None,
            },
            Some(&mut scores),
        )
        .unwrap();

        // Output: uniform attention over (Q2-dequantized) identical V rows → finite, ≈ V.
        let o = out.as_slice::<f32>();
        for &x in o {
            assert!(x.is_finite(), "attention output must be finite, got {x}");
            assert!(
                (0.0..=6.0).contains(&x),
                "out should be bounded near V=3, got {x}"
            );
        }
        // post-softmax scores recorded into caller slice: 2 ~equal weights summing to ~1.
        let s: f32 = scores.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-3,
            "post-softmax weights sum to 1, got {s}"
        );
    }

    #[test]
    fn test_attention_into_prefill_causal_uniform() {
        // Phase α-K ①-e: multi-token prefill arm (seq_len>1). seq=4 < res_cap(=RES=32)라 Q2 flush
        // 미발생 → residual 이 raw F32 그대로 dequant(exact)되어 bit-exact 검증 가능. K=0 → 모든
        // score 0 → uniform softmax. V[pos]=pos(broadcast). causal mask 로 query row r 은 cache pos
        // 0..=r 만 attend → out[r] = mean(0..=r) = r/2. write_kv_batch(prefill write) +
        // attention_into(신규 prefill arm) 합동 검증 + causal-mask 확인(arm 부재 시 panic 회귀 가드).
        let kv_heads = 1;
        let n_heads_q = 1;
        let seq = 4;
        let fmt = KIVIFormat::new(0, KiviCache::new(kv_heads, HD, MAXSEQ, RES));
        let backend = CpuBackend::new();

        let k_data = vec![0.0f32; seq * kv_heads * HD];
        let mut v_data = vec![0.0f32; seq * kv_heads * HD];
        for p in 0..seq {
            for d in 0..HD {
                v_data[p * kv_heads * HD + d] = p as f32;
            }
        }
        let kb = f32_tensor(vec![1, seq, kv_heads, HD], &k_data);
        let vb = f32_tensor(vec![1, seq, kv_heads, HD], &v_data);
        fmt.write_kv_batch(&kb, &vb, &backend).unwrap();
        assert_eq!(fmt.current_pos(), seq);

        // q 값은 무관(K=0 → score 0). out = [1, seq, n_heads_q*head_dim].
        let q = f32_tensor(
            vec![1, seq, n_heads_q, HD],
            &vec![1.0; seq * n_heads_q * HD],
        );
        let mut out = f32_tensor(
            vec![1, seq, n_heads_q * HD],
            &vec![0.0; seq * n_heads_q * HD],
        );

        fmt.attention_into(
            &q,
            &backend,
            &mut out,
            AttnDims {
                n_heads_q,
                window: None,
            },
            None,
        )
        .unwrap();

        let o = out.as_slice::<f32>();
        for r in 0..seq {
            let expected = r as f32 / 2.0; // mean(0..=r), causal mask
            for d in 0..HD {
                let got = o[r * HD + d];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "row {r} d {d}: expected {expected} (causal mean 0..=r), got {got}"
                );
            }
        }
    }

    #[test]
    fn test_attention_into_no_scores() {
        // scores=None path must not panic and must produce output.
        let kv_heads = 1;
        let n_heads_q = 1;
        let fmt = KIVIFormat::new(0, KiviCache::new(kv_heads, HD, MAXSEQ, RES));
        let row = vec![1.0f32; HD];
        let k = f32_tensor(vec![1, 1, kv_heads, HD], &row);
        let v = f32_tensor(vec![1, 1, kv_heads, HD], &row);
        fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();

        let q = f32_tensor(vec![1, 1, n_heads_q, HD], &row);
        let mut out = f32_tensor(vec![1, 1, n_heads_q, HD], &[0.0; HD]);
        let backend = CpuBackend::new();
        fmt.attention_into(
            &q,
            &backend,
            &mut out,
            AttnDims {
                n_heads_q,
                window: None,
            },
            None,
        )
        .unwrap();
        let o = out.as_slice::<f32>();
        for &x in o {
            assert!(x.is_finite());
        }
    }
}
