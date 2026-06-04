//! `OffloadFormat` — `KVCacheFormat` impl wrapping an `OffloadKVCache` (Phase α-K Step 5-B).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §4.1 / Step 5 (`design_alpha_k_step5_*`).
//! `StandardFormat`(pressure/standard_format.rs) 의 offload 짝 — offload KV forward 경로를
//! generic `<C: KVCacheOps>` 에서 `KVCacheFormat` trait-object 로 이주하기 위한 wrapper.
//!
//! **purely additive, env-gated OFF default, unwired-on-OFF** — OLD `OffloadKVCache`/`KVCacheOps`
//! 경로(`offload.rs`)와 OLD `forward_into_offload`(transformer.rs)를 1바이트도 안 건드린다.
//! `LLMRS_OFFLOAD_FMT` 게이트 ON 일 때만 `OffloadForward` 가 transient wrap 한다. 내부 가변성 =
//! `std::sync::Mutex`(trait `Send+Sync` 요구로 `RefCell` 불가). offload 는 SeqMajor + eviction
//! 미지원이라 HeadMajor GPU scatter fast-path 와 compact 가 전부 부재(StandardFormat 과 갈림).

use std::sync::Arc;
use std::sync::Mutex;

use anyhow::Result;

use crate::backend::Backend;
use crate::buffer::DType;
use crate::format::{AttnDims, KVCacheFormat, Merge};
use crate::kv_cache_ops::KVCacheOps;
use crate::pressure::offload::OffloadKVCache;
use crate::tensor::Tensor;

/// 내부 가변 상태 — `OffloadKVCache` 와 비-F32 cast scratch 를 **단일 lock** 으로 묶는다.
///
/// `StandardFormatInner` 미러: scratch(`k_cast`/`v_cast`)는 비-F32 write 의 reusable buffer.
/// cache 와 한 `Mutex` 안에 두어 별도 lock 으로 인한 동시성 hazard 를 차단한다.
struct OffloadFormatInner {
    cache: OffloadKVCache,
    /// Lazy cast scratch (target dtype). 첫 비-F32 write 에서 `cache.cast_memory()`(GPU) 또는
    /// host `Galloc`(CPU) 로 할당.
    k_cast: Option<Tensor>,
    v_cast: Option<Tensor>,
}

/// `OffloadKVCache` 를 `KVCacheFormat` 으로 노출하는 wrapper (`StandardFormat` 의 offload 짝).
///
/// 기존 `OffloadKVCache` 를 `Mutex` 로 감싸 `&self` 메서드에서 내부 `&mut` 메서드에 위임한다.
/// `OffloadKVCache` 자체는 무변.
pub struct OffloadFormat {
    idx: usize,
    inner: Mutex<OffloadFormatInner>,
}

impl OffloadFormat {
    /// `OffloadKVCache` 를 layer 인덱스와 함께 wrapping.
    pub fn new(idx: usize, cache: OffloadKVCache) -> Self {
        Self {
            idx,
            inner: Mutex::new(OffloadFormatInner {
                cache,
                k_cast: None,
                v_cast: None,
            }),
        }
    }

    /// wrapping 을 해제하고 내부 `OffloadKVCache` 를 반환 (transient-wrap round-trip).
    ///
    /// `StandardFormat::into_inner` 미러 — cross-token 상태(retained attn_buf / gpu_k_buf /
    /// current_pos / store_behind)는 전부 `OffloadKVCache` 필드라 unwrap 을 가로질러 보존된다.
    /// cast scratch(`k_cast`/`v_cast`)는 transient 라 버린다(다음 wrap 에서 lazy 재할당).
    ///
    /// `pub`(StandardFormat::into_inner 의 `pub(crate)` 와 달리) — `legacy_generate` 가 별도 bin
    /// 크레이트라 offload device 게이트 매체(run_offload)에서 wrap/unwrap 에 접근해야 한다(5-A).
    pub fn into_inner(self) -> OffloadKVCache {
        self.inner.into_inner().unwrap().cache
    }

    /// preload pool background task seam — interior-mut(`&self`).
    ///
    /// `forward_into_offload` 의 `OffloadKVCache::preload(&mut)` 를 `&self` 로 미러. raw cast 가
    /// `*const OffloadFormat` 이라도 `Mutex` 가 aliasing 을 흡수한다(far_idx≠i 불변식은 retain/release
    /// 로직 보존용으로만 유지, soundness 는 Mutex 가 보장).
    pub(crate) fn preload_locked(&self) -> Result<()> {
        self.inner.lock().unwrap().cache.preload()
    }

    /// `OffloadKVCache::release_buffers(&mut)` 를 `&self` 로 미러.
    pub(crate) fn release_locked(&self) {
        self.inner.lock().unwrap().cache.release_buffers();
    }

    /// `OffloadKVCache::retain_preload(&mut)` 를 `&self` 로 미러.
    pub(crate) fn retain_locked(&self) {
        self.inner.lock().unwrap().cache.retain_preload();
    }

    /// `OffloadKVCache::reset_session(&mut)` 를 `&self` 로 미러.
    #[allow(dead_code)]
    pub(crate) fn reset_session_locked(&self) {
        self.inner.lock().unwrap().cache.reset_session();
    }

    /// KV write 흡수 — `forward_gen` 의 KV-update 분기(transformer_layer.rs:35 `update_kv_cache`
    /// 미러)를 format 표면으로 옮긴 것. offload 는 SeqMajor 라 HeadMajor GPU scatter fast-path 부재 —
    /// 비-F32 cast 후 `update` / F32 직접 `update` 두 경로만.
    ///
    /// 비-F32(F16) cast 경로: `StandardFormat::write_inner` 의 cast 분기 미러. scratch allocator 는
    /// `cache.cast_memory()`(GPU memory) 또는 host `Galloc`(CPU). host 게이트(F16+CPU)는 CPU alloc
    /// 충분, device(F16 GPU)는 GPU 버퍼. F32 경로: `OffloadKVCache::update` 는 GPU-buffer 미보유
    /// (default None)라 `get_buffers_mut()==None` → non-null 입력이면 `backend.synchronize()`(구
    /// `update_kv_cache` 의 has_gpu_buffers=false 분기) 후 `update`. CpuBackend.synchronize 는 no-op.
    fn write_inner(&self, new_k: &Tensor, new_v: &Tensor, backend: &dyn Backend) -> Result<()> {
        let mut guard = self.inner.lock().unwrap();
        let kv_dtype = guard.cache.kv_dtype();

        if kv_dtype != DType::F32 {
            // 비-F32 cast 경로 (F16). F32 입력을 cache dtype 으로 cast 후 update.
            // `OffloadKVCache::update` 는 cast 를 안 하고 입력이 이미 cache dtype 임을 전제(raw byte
            // copy)하므로, 반드시 여기서 cast 해야 한다(dtype 미일치 silent garbage 방지).
            let memory = guard
                .cache
                .cast_memory()
                .unwrap_or_else(|| Arc::new(crate::memory::galloc::Galloc::new()));
            let n_elem: usize = new_k.shape().dims().iter().product();
            let buf_size = match kv_dtype {
                DType::F16 => n_elem * 2,
                DType::Q4_0 => {
                    (n_elem / crate::quant::QK4_0) * std::mem::size_of::<crate::quant::BlockQ4_0>()
                }
                _ => n_elem * 4,
            };
            let k_stale = guard
                .k_cast
                .as_ref()
                .is_none_or(|t| t.shape().dims() != new_k.shape().dims());
            if k_stale {
                let buf = memory.alloc(buf_size, kv_dtype)?;
                guard.k_cast = Some(Tensor::new(
                    new_k.shape().clone(),
                    buf,
                    new_k.backend().clone(),
                ));
            }
            let v_stale = guard
                .v_cast
                .as_ref()
                .is_none_or(|t| t.shape().dims() != new_v.shape().dims());
            if v_stale {
                let buf = memory.alloc(buf_size, kv_dtype)?;
                guard.v_cast = Some(Tensor::new(
                    new_v.shape().clone(),
                    buf,
                    new_v.backend().clone(),
                ));
            }
            let OffloadFormatInner {
                cache,
                k_cast,
                v_cast,
            } = &mut *guard;
            let k_cast = k_cast.as_mut().unwrap();
            let v_cast = v_cast.as_mut().unwrap();
            backend.cast(new_k, k_cast)?;
            backend.cast(new_v, v_cast)?;
            return cache.update(k_cast, v_cast);
        }

        // F32 경로 (transformer_layer.rs:35 `update_kv_cache` 미러).
        // OffloadKVCache 는 GPU-buffer 미보유(get_buffers_mut 부재 → has_gpu_buffers=false)라
        // non-null 입력이면 sync 후 update. device-only null-ptr 입력은 `OffloadKVCache::update`
        // 내부 read_buffer 가 자체 처리(offload.rs:325-340). CpuBackend.synchronize 는 no-op.
        if !new_k.as_ptr().is_null() {
            backend.synchronize()?;
        }
        guard.cache.update(new_k, new_v)
    }
}

impl KVCacheFormat for OffloadFormat {
    fn idx(&self) -> usize {
        self.idx
    }

    fn current_pos(&self) -> usize {
        // path X: KVCacheOps trait 은 5-E 까지 생존 (StandardFormat:304 미러).
        KVCacheOps::current_pos(&self.inner.lock().unwrap().cache)
    }

    fn capacity(&self) -> usize {
        self.inner.lock().unwrap().cache.capacity()
    }

    fn write_kv(&self, new_k: &Tensor, new_v: &Tensor, backend: &dyn Backend) -> Result<()> {
        // decode (seq_len=1). offload 는 SeqMajor 라 GPU scatter fast-path 미진입.
        self.write_inner(new_k, new_v, backend)
    }

    fn write_kv_batch(&self, new_k: &Tensor, new_v: &Tensor, backend: &dyn Backend) -> Result<()> {
        // prefill (seq_len>1). write 로직은 decode 와 동일(SeqMajor batch store 는
        // `OffloadKVCache::update` 의 seq_len 분기가 자체 처리).
        self.write_inner(new_k, new_v, backend)
    }

    fn compact(&self, _keep: &[usize], _merges: &[Merge]) -> Result<()> {
        // offload 는 eviction 미지원 (OffloadForward::on_kv_prune no-op 일치).
        anyhow::bail!("offload: eviction 미지원")
    }

    fn attention_into(
        &self,
        q: &Tensor,
        backend: &dyn Backend,
        out: &mut Tensor,
        dims: AttnDims,
        scores: Option<&mut [f32]>,
    ) -> Result<()> {
        let seq_len = q.shape().dims()[1];

        let mut guard = self.inner.lock().unwrap();
        let cache = &mut guard.cache;
        let n_heads_kv = cache.kv_heads();
        let head_dim = cache.head_dim();
        let cache_seq_len = KVCacheOps::current_pos(&*cache);

        // ── prefill (seq_len>1): multi-token causal attention (StandardFormat:361-385 미러) ──
        // decode delegate 는 single-query + causal-mask 부재라 재사용 불가 → prefill_attention(free fn).
        if seq_len > 1 {
            let kv_capacity = cache.capacity();
            let kv_layout = cache.layout(); // = SeqMajor
            let batch_size = q.shape().dims()[0];
            let q_start_pos = cache_seq_len - seq_len;
            let (k_cache, v_cache) = KVCacheOps::get_view(&mut *cache);
            let _ = scores;
            return crate::pressure::standard_format::prefill_attention(
                q,
                out,
                &k_cache,
                &v_cache,
                dims.n_heads_q,
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

        // ── decode (seq_len==1): StandardFormat:387-444 미러 ──
        let effective_cache_len = match dims.window {
            Some(w) => cache_seq_len.min(w),
            None => cache_seq_len,
        };
        let kv_dtype = cache.kv_dtype();
        let layout = cache.layout();
        let capacity = cache.capacity();
        let (k_cache, v_cache) = KVCacheOps::get_view(&mut *cache);

        // Q4_0 + GPU: backend.attention_gen 은 GPU Q4_0 dequant-attention 커널 부재라 garbage →
        // attention_q4_gpu_fallback 재사용. CpuBackend(is_gpu=false)는 미진입 → 아래 attention_gen.
        if kv_dtype == DType::Q4_0 && backend.is_gpu() {
            let kv_start_pos = cache_seq_len - effective_cache_len;
            let need_scores = scores.is_some();
            let mut empty: [f32; 0] = [];
            let scores_buf: &mut [f32] = match scores {
                Some(s) => s,
                None => &mut empty,
            };
            return crate::layers::transformer_layer::TransformerLayer::attention_q4_gpu_fallback(
                q,
                &k_cache,
                &v_cache,
                out,
                scores_buf,
                dims.n_heads_q,
                n_heads_kv,
                head_dim,
                effective_cache_len,
                kv_start_pos,
                layout,
                capacity,
                need_scores,
                backend,
            );
        }

        // typed/F32 경로: backend.attention_gen 에 위임. CPU F32 inline-NEON fallback
        // (forward_gen.rs:554+)은 복제하지 않는다 (StandardFormat 과 동일 carve-out — F32+host-mapped/
        // CPU 는 NOT bit-identical, 게이트가 회피).
        backend.attention_gen(
            q,
            &k_cache,
            &v_cache,
            out,
            dims.n_heads_q,
            n_heads_kv,
            head_dim,
            effective_cache_len,
            scores,
        )
    }
}

/// Type-erased preload function for `OffloadFormat` background prefetch tasks.
///
/// `forward_into_offload_fmt` 의 preload pool 이 `submit_raw(Arc::as_ptr(&fmt) as *mut (), …)` 로
/// 제출한다. `OffloadFormat` 은 interior-mut(`&self`)라 raw cast 가 `*const` 이며 `Mutex` 가
/// aliasing 을 흡수한다.
///
/// # Safety
/// `ptr` 은 살아있는, 정렬된 `OffloadFormat` 을 가리켜야 한다.
pub unsafe fn preload_offload_fmt_erased(ptr: *mut ()) -> Result<()> {
    unsafe { (*(ptr as *const OffloadFormat)).preload_locked() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::memory::host::shared::SharedBuffer;
    use crate::pressure::offload::raw_store::RawStore;
    use crate::shape::Shape;

    fn f32_tensor(dims: Vec<usize>, data: &[f32]) -> Tensor {
        let buf = Arc::new(SharedBuffer::new(data.len() * 4, DType::F32));
        let mut t = Tensor::new(Shape::new(dims), buf, Arc::new(CpuBackend::new()));
        t.as_mut_slice::<f32>().copy_from_slice(data);
        t
    }

    fn make_f16_offload(kv_heads: usize, head_dim: usize, max_seq: usize) -> OffloadKVCache {
        let token_bytes = kv_heads * head_dim * DType::F16.size();
        let store = RawStore::new(token_bytes);
        OffloadKVCache::new(0, kv_heads, head_dim, DType::F16, max_seq, Box::new(store))
    }

    /// OffloadFormat::attention_into(decode) == OLD 방식(get_view + attention_gen)이 CPU bit-identical.
    /// F16 KV, CpuBackend. forward_gen 의 decode 경로(get_view → backend.attention_gen)를 직접 미러한
    /// reference 와 wrapper 출력이 일치하는지 검증.
    #[test]
    fn test_attention_into_decode_matches_raw_offload() {
        use crate::kv_cache_ops::KVCacheOps;

        let kv_heads = 2;
        let head_dim = 8; // F32 attention path (Q4_0 아님)
        let n_heads_q = 4; // GQA n_rep=2
        let max_seq = 16;
        let backend = CpuBackend::new();

        // 두 개의 동일한 OffloadFormat 을 만들어 동일 토큰을 쓴다(F16 cache, 동일 cast 경로).
        // reference 는 attention 만 OLD 방식(get_view + attention_gen)으로 돌린다.
        let fmt_ref = OffloadFormat::new(0, make_f16_offload(kv_heads, head_dim, max_seq));
        let fmt = OffloadFormat::new(0, make_f16_offload(kv_heads, head_dim, max_seq));

        // 3 토큰 write (F32 입력 → F16 cast, write_kv 가 동일 cast 적용).
        let row = kv_heads * head_dim;
        for p in 0..3 {
            let token: Vec<f32> = (0..row).map(|i| ((p * row + i) as f32) * 0.25).collect();
            let k = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
            let v = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
            fmt.write_kv(&k, &v, &backend).unwrap();
            fmt_ref.write_kv(&k, &v, &backend).unwrap();
        }
        assert_eq!(fmt.current_pos(), 3);
        assert_eq!(fmt_ref.current_pos(), 3);
        let mut raw = fmt_ref.into_inner();
        assert_eq!(KVCacheOps::current_pos(&raw), 3);

        // decode query.
        let q_data: Vec<f32> = (0..n_heads_q * head_dim)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let q = f32_tensor(vec![1, 1, n_heads_q, head_dim], &q_data);

        // wrapper attention_into.
        let mut out_fmt = f32_tensor(
            vec![1, 1, n_heads_q, head_dim],
            &vec![0.0; n_heads_q * head_dim],
        );
        fmt.attention_into(
            &q,
            &backend,
            &mut out_fmt,
            AttnDims {
                n_heads_q,
                window: None,
            },
            None,
        )
        .unwrap();

        // OLD reference: get_view + backend.attention_gen.
        let cache_seq_len = KVCacheOps::current_pos(&raw);
        let (k_view, v_view) = KVCacheOps::get_view(&mut raw);
        let mut out_ref = f32_tensor(
            vec![1, 1, n_heads_q, head_dim],
            &vec![0.0; n_heads_q * head_dim],
        );
        backend
            .attention_gen(
                &q,
                &k_view,
                &v_view,
                &mut out_ref,
                n_heads_q,
                kv_heads,
                head_dim,
                cache_seq_len,
                None,
            )
            .unwrap();

        assert_eq!(
            out_fmt.as_slice::<f32>(),
            out_ref.as_slice::<f32>(),
            "OffloadFormat::attention_into decode != raw get_view+attention_gen"
        );
    }

    /// OffloadFormat::attention_into(prefill, seq_len>1) 의 causal mask + SeqMajor 인덱싱 정확성.
    ///
    /// 인과성: 첫 query 위치(seq 0)는 cache pos 0 하나만 attend → softmax over 1 element = 1.0 →
    /// out[seq 0] == V[0] (query 값 무관, 다른 FP 연산 없음 = exact). KV 를 F16-exact 값(1.0/2.0/3.0)
    /// 으로 채워 roundtrip 오차 0. 이 테스트는 prefill arm(prefill_attention SeqMajor)을 **정상
    /// 채워진 캐시**(preload 미발화)에서 격리 검증한다 — forward_into_offload 루프의 사전 존재
    /// preload-on-empty 버그(get_view zero 반환)와 무관하게 format 의 prefill 계산 정확성을 확인.
    #[test]
    fn test_attention_into_prefill_causal_first_position() {
        let kv_heads = 2;
        let head_dim = 8;
        let n_heads_q = 2; // n_rep=1
        let max_seq = 16;
        let seq_len = 3;
        let backend = CpuBackend::new();
        let fmt = OffloadFormat::new(0, make_f16_offload(kv_heads, head_dim, max_seq));

        // 3 토큰: pos p 의 K=V 전부 (p+1.0) (F16-exact). write_kv(seq=1) ×3 으로 채운다.
        let row = kv_heads * head_dim;
        for p in 0..seq_len {
            let val = (p + 1) as f32; // 1.0, 2.0, 3.0
            let token = vec![val; row];
            let k = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
            let v = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
            fmt.write_kv(&k, &v, &backend).unwrap();
        }
        assert_eq!(fmt.current_pos(), seq_len);

        // multi-token query (seq_len=3) — 값은 임의(causal 로 seq 0 은 query 무관).
        let q_data: Vec<f32> = (0..seq_len * n_heads_q * head_dim)
            .map(|i| (i as f32) * 0.05)
            .collect();
        let q = f32_tensor(vec![1, seq_len, n_heads_q, head_dim], &q_data);
        let mut out = f32_tensor(
            vec![1, seq_len, n_heads_q, head_dim],
            &vec![0.0; seq_len * n_heads_q * head_dim],
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

        // seq 위치 0 (모든 head) == V[0] = 1.0 (causal: pos 0 → cache 0 only, softmax=1).
        let out_s = out.as_slice::<f32>();
        let pos0 = &out_s[0..n_heads_q * head_dim];
        for (i, &x) in pos0.iter().enumerate() {
            assert!(
                (x - 1.0).abs() < 1e-5,
                "prefill causal: out[seq0][{i}]={x}, expected 1.0 (attends only to V[0])"
            );
        }
        // 전체 출력 finite (NaN/Inf 없음 — 사전 존재 preload 버그와 달리 정상 채워진 캐시).
        assert!(
            out_s.iter().all(|x| x.is_finite()),
            "prefill out must be all-finite"
        );
    }

    /// compact() 는 offload 에서 eviction 미지원이라 Err 를 반환해야 한다.
    #[test]
    fn test_compact_errors() {
        let fmt = OffloadFormat::new(0, make_f16_offload(1, 4, 8));
        assert!(fmt.compact(&[0], &[]).is_err());
    }

    /// into_inner 가 cross-token 상태(current_pos)를 보존하는지.
    #[test]
    fn test_into_inner_preserves_pos() {
        use crate::kv_cache_ops::KVCacheOps;
        let backend = CpuBackend::new();
        let fmt = OffloadFormat::new(3, make_f16_offload(1, 4, 8));
        let token = vec![0.5f32; 4];
        let k = f32_tensor(vec![1, 1, 1, 4], &token);
        let v = f32_tensor(vec![1, 1, 1, 4], &token);
        fmt.write_kv(&k, &v, &backend).unwrap();
        assert_eq!(fmt.idx(), 3);
        let cache = fmt.into_inner();
        assert_eq!(KVCacheOps::current_pos(&cache), 1);
    }
}
