//! `StandardFormat` — `KVCacheFormat` impl wrapping a standard `KVCache` (§4.1, Phase α-K).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §4.1 / §2.1 (guard rail: format impl 은 `kv/`
//! (현 `pressure/`)에, base trait 은 `format/` 에).
//!
//! **purely additive, host-only, unwired** — 기존 `KVCache` 와 `KVCacheOps` 경로를 1바이트도
//! 건드리지 않고, 신규 wrapper 로 공존한다. production 에서 `StandardFormat` 를 생성하는 코드는
//! 0(unit test 에서만 생성). 내부 가변성 = `std::sync::Mutex`(trait `Send+Sync` 요구로 `RefCell`
//! 불가; §4.1 R4 상 cold-path 라 lock 비용 무관).

use std::sync::Arc;
use std::sync::Mutex;

use anyhow::Result;

use crate::backend::Backend;
use crate::buffer::DType;
use crate::format::{AttnDims, KVCacheFormat};
use crate::memory::host::shared::SharedBuffer;
use crate::pressure::kv_cache::KVCache;
use crate::shape::Shape;
use crate::tensor::Tensor;
use technique_api::WeightedMerge;

/// 내부 가변 상태 — `KVCache` 와 비-F32 cast scratch 를 **단일 lock** 으로 묶는다.
///
/// scratch(`k_cast`/`v_cast`)는 비-F32 write 경로의 reusable buffer 로, `forward_gen` 의
/// `ws.k_cast`/`ws.v_cast` 와 같은 역할(토큰마다 재할당 방지). cache 와 한 `Mutex` 안에 두어
/// 별도 lock 으로 인한 동시성 hazard 를 원천 차단한다(write 가 cache+scratch 를 항상 함께 만짐).
pub(crate) struct StandardFormatInner {
    pub(crate) cache: KVCache,
    /// Lazy cast scratch (target dtype). 첫 비-F32 write 에서 inner cache 의 allocator 로 할당.
    k_cast: Option<Tensor>,
    v_cast: Option<Tensor>,
}

/// Standard (F32/F16/Q4_0) KV cache 를 `KVCacheFormat` 으로 노출하는 wrapper.
///
/// 기존 `KVCache` 를 `Mutex` 로 감싸 `&self` 메서드에서 내부 `&mut` 메서드에 위임한다.
/// `KVCache` 자체는 무변.
pub struct StandardFormat {
    idx: usize,
    inner: Mutex<StandardFormatInner>,
}

impl StandardFormat {
    /// `KVCache` 를 layer 인덱스와 함께 wrapping. (현재 unit test 전용 — unwired.)
    pub fn new(idx: usize, inner: KVCache) -> Self {
        Self {
            idx,
            inner: Mutex::new(StandardFormatInner {
                cache: inner,
                k_cast: None,
                v_cast: None,
            }),
        }
    }

    /// 내부 `KVCache` 에 `&mut` 접근하여 `f` 실행 (substep 3c fmt-cache wiring).
    ///
    /// forward(write_kv/attention_into)는 base trait 으로 통과하지만, fmt 활성 시
    /// non-forward 연산(reset_kv 등)이 inner cache 에 도달할 seam 이 필요하다 — base trait 에
    /// method 를 추가하지 않고(`INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC`) concrete inherent 로 제공.
    /// lock guard 안에서 closure 를 실행하므로 호출 종료 시 lock 이 풀린다.
    pub(crate) fn with_cache_mut<R>(&self, f: impl FnOnce(&mut KVCache) -> R) -> R {
        let mut guard = self.inner.lock().unwrap();
        f(&mut guard.cache)
    }

    #[cfg(feature = "opencl")]
    /// plan hot-path geometry 스냅샷 (Phase α-K (3p) ④-a).
    ///
    /// **단일 lock** 으로 `current_pos`/`capacity` 를 묶어 [`PlanGeometry`] 로 반환한다 —
    /// `execute<C>` 가 레이어 진입부에서 호출하던 4개 `KVCacheOps` getter 를 1 lock 으로 통합.
    /// standard 는 residual/quantized partition 부재라 `res_pos`/`q2_tokens` = 0.
    pub(crate) fn plan_geometry(&self) -> crate::backend::opencl::plan::PlanGeometry {
        let g = self.inner.lock().unwrap();
        crate::backend::opencl::plan::PlanGeometry {
            current_pos: g.cache.current_pos(),
            capacity: g.cache.capacity(),
            res_pos: 0,
            q2_tokens: 0,
        }
    }

    #[cfg(feature = "opencl")]
    /// plan hot-path position advance (Phase α-K (3p) ④-a).
    ///
    /// `execute<C>` 의 레이어 끝 `cache.advance_pos(n)` 를 `&self` + interior-mut 로 미러.
    pub(crate) fn plan_advance(&self, n: usize) {
        self.with_cache_mut(|c| c.advance_pos(n));
    }

    #[cfg(feature = "opencl")]
    /// plan 빌드용 lock guard (Phase α-K (3p) ④-a `build_plan`).
    ///
    /// `build_plan` 는 모든 핸들의 guard 를 동시에 잡고 `&KVCache` 슬라이스를 만들어
    /// `build_plan` 본문(byte-identical)을 재사용한다. cl_mem 핸들은 `build_full_plan` 안에서
    /// `set_kernel_arg` 로 즉시 바인딩(클론)되므로 guard 가 그 호출 동안만 살아 있으면 충분하다.
    pub(crate) fn plan_lock(&self) -> std::sync::MutexGuard<'_, StandardFormatInner> {
        self.inner.lock().unwrap()
    }

    /// wrapping 을 해제하고 내부 `KVCache` 를 반환 (Phase α-K ①-c eval transient-wrap round-trip).
    ///
    /// eval 이 forward 1회 동안만 `Vec<KVCache>` → `Arc<StandardFormat>` 로 wrap 한 뒤
    /// `Arc::try_unwrap().into_inner()` 로 concrete cache 를 복귀시키는 seam. cast scratch
    /// (`k_cast`/`v_cast`)는 transient 라 버린다(다음 wrap 에서 lazy 재할당). base trait 무변
    /// (`INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC`).
    pub(crate) fn into_inner(self) -> KVCache {
        self.inner.into_inner().unwrap().cache
    }

    /// Unwrap-Evict-Rewrap (UER) seam (Phase α-K BC (3d)): inner `KVCache` 를 일시적으로 꺼낸다.
    ///
    /// chat 멀티턴 eviction 이 `CacheManager::force_evict(&mut [KVCache])`(연속 슬라이스 요구,
    /// D2O cross-layer 정확성)를 **OLD 경로 그대로** 재사용하도록, fmt_caches 의 inner cache 들을
    /// 연속 `Vec<KVCache>` 로 모으는 용도. `put_inner` 와 페어 호출(단일 lock 구간 sequential).
    /// cast scratch(`k_cast`/`v_cast`)는 guard 에 남아 보존된다(다음 write 재사용). Arc 는 보존
    /// (into_inner 의 try_unwrap 과 달리 self 미소비) — listener phase 무관.
    ///
    /// `KVCache: !Default` 이므로 `mem::take` 불가 → cache 자신의 backend 로 만든 0-size
    /// placeholder 로 `mem::replace`. placeholder 는 `put_inner` 까지 microsecond 만 잔존(eviction
    /// = turn 경계 cold path 라 per-layer 0-byte 할당 무시 가능).
    pub(crate) fn take_inner(&self) -> KVCache {
        let mut guard = self.inner.lock().unwrap();
        let backend = guard.cache.k_buffer.backend().clone();
        let buf = Arc::new(SharedBuffer::new(0, DType::F32));
        let ph_k = Tensor::new(Shape::new(vec![1, 0, 1, 1]), buf.clone(), backend.clone());
        let ph_v = Tensor::new(Shape::new(vec![1, 0, 1, 1]), buf, backend);
        std::mem::replace(&mut guard.cache, KVCache::new(ph_k, ph_v, 0))
    }

    /// `take_inner` 의 역연산 — evict 된 `KVCache` 를 다시 넣는다(placeholder 폐기).
    pub(crate) fn put_inner(&self, cache: KVCache) {
        self.inner.lock().unwrap().cache = cache;
    }

    /// KV write 흡수 — `forward_gen` 의 KV-update 분기(transformer_layer/forward_gen.rs:330-386)를
    /// format 표면으로 옮긴 것. `is_decode`(seq_len=1)면 GPU fused cast+scatter fast-path 게이팅.
    ///
    /// **host 경로 = correctness fallback** — `CpuBackend` 는 `is_gpu()==false`라 GPU scatter 분기를
    /// 밟지 않으므로 host build+test 가 F32/비-F32 cast 경로를 검증하고, GPU scatter 정확성은
    /// device round(substep (3c))에서 검증한다. **비-F32(F16/Q4_0) cast 경로**(forward_gen 의
    /// `memory.alloc` + `ws.k_cast` scratch)는 inner `KVCache` 의 allocator 로 scratch 를 lazy 할당해
    /// 흡수한다 — write_kv signature 에 `memory` 를 추가하지 않는다(format⊥hardware, KVCache 가 이미
    /// 동일 allocator 보유). 여전히 unwired 라 무회귀(production 호출처 0).
    fn write_inner(
        &self,
        new_k: &Tensor,
        new_v: &Tensor,
        backend: &dyn Backend,
        is_decode: bool,
    ) -> Result<()> {
        use crate::kv_cache_ops::KVLayout;

        let mut guard = self.inner.lock().unwrap();
        let kv_dtype = guard.cache.kv_dtype();

        // GPU F16 HeadMajor decode: fused cast+scatter (1 dispatch). host 미진입(is_gpu=false).
        if is_decode
            && backend.is_gpu()
            && kv_dtype == DType::F16
            && guard.cache.layout() == KVLayout::HeadMajor
        {
            let cache = &mut guard.cache;
            let pos = cache.current_pos();
            cache.ensure_capacity(pos + 1)?;
            let cap = cache.capacity();
            let head_dim = cache.head_dim();
            if let Some((k_buf, v_buf)) = cache.get_buffers_mut() {
                backend.kv_scatter_f32_to_f16(new_k, new_v, k_buf, v_buf, head_dim, cap, pos)?;
            }
            cache.advance_pos(1);
            return Ok(());
        }

        // GPU F32 HeadMajor decode: single batched scatter dispatch.
        if is_decode
            && backend.is_gpu()
            && kv_dtype == DType::F32
            && guard.cache.layout() == KVLayout::HeadMajor
            && backend.supports_kv_scatter_f32_batch()
        {
            let cache = &mut guard.cache;
            let pos = cache.current_pos();
            cache.ensure_capacity(pos + 1)?;
            let cap = cache.capacity();
            let n_heads_kv = cache.kv_heads();
            let head_dim = cache.head_dim();
            if let Some((k_buf, v_buf)) = cache.get_buffers_mut() {
                backend.kv_scatter_f32_to_f32_batch(
                    new_k, new_v, k_buf, v_buf, n_heads_kv, head_dim, cap, pos, 1,
                )?;
            }
            cache.advance_pos(1);
            return Ok(());
        }

        // ──────────────────────────────────────────────────────────────────────
        // C3 (§9.1-BC1-CONTRACT ⚠️⚠️ 2차 정정): GPU *prefill batch* scatter fast-path.
        // `write_kv_batch`(decode_fast_path=false, seq_len>1)이 위 decode fast-path 묶음을
        // batch(count=seq_len)로 미러링한다. 게이팅·dtype·position 회계는 decode 분기와 동일하되
        // count 인자만 `1`→실제 seq_len. **bit-identical to cast/update** (kv_scatter_*_batch 의
        // dst_off = h*cap*head_dim + (write_pos_start+s)*head_dim = KVCache::update 의 batch dst_off,
        // advance_pos(seq_len) = update 의 `current_pos += seq_len` + high_water 갱신과 동일).
        // host(CpuBackend, is_gpu=false)는 미진입 → 아래 cast/update 경로가 검증. GPU scatter
        // 정확성은 device round(S25/Jetson)에서 검증. Q4_0 은 GPU fast-path 부재(아래 :131 주석)라
        // 진입하지 않고 cast 경로 유지.
        let seq_len = new_k.shape().dims()[1];

        // GPU F16 HeadMajor batch: fused cast+scatter (1 dispatch over seq_len positions).
        // decode F16(single-pos `kv_scatter_f32_to_f16`)과 달리 batch 변형은 host-pointer
        // fallback 이 device-only 버퍼에서 segfault 하므로 `supports_kv_scatter_batch()` 게이트
        // 필수(미충족 시 아래 cast 경로로 자연 강하 — 동일 출력).
        if !is_decode
            && backend.is_gpu()
            && kv_dtype == DType::F16
            && guard.cache.layout() == KVLayout::HeadMajor
            && backend.supports_kv_scatter_batch()
        {
            let cache = &mut guard.cache;
            let pos = cache.current_pos();
            cache.ensure_capacity(pos + seq_len)?;
            let cap = cache.capacity();
            let n_heads_kv = cache.kv_heads();
            let head_dim = cache.head_dim();
            if let Some((k_buf, v_buf)) = cache.get_buffers_mut() {
                backend.kv_scatter_f32_to_f16_batch(
                    new_k, new_v, k_buf, v_buf, n_heads_kv, head_dim, cap, pos, seq_len,
                )?;
            }
            cache.advance_pos(seq_len);
            return Ok(());
        }

        // GPU F32 HeadMajor batch: single batched scatter dispatch over seq_len positions.
        if !is_decode
            && backend.is_gpu()
            && kv_dtype == DType::F32
            && guard.cache.layout() == KVLayout::HeadMajor
            && backend.supports_kv_scatter_f32_batch()
        {
            let cache = &mut guard.cache;
            let pos = cache.current_pos();
            cache.ensure_capacity(pos + seq_len)?;
            let cap = cache.capacity();
            let n_heads_kv = cache.kv_heads();
            let head_dim = cache.head_dim();
            if let Some((k_buf, v_buf)) = cache.get_buffers_mut() {
                backend.kv_scatter_f32_to_f32_batch(
                    new_k, new_v, k_buf, v_buf, n_heads_kv, head_dim, cap, pos, seq_len,
                )?;
            }
            cache.advance_pos(seq_len);
            return Ok(());
        }

        // 비-F32 cast 경로: F32 입력을 cache dtype(F16/Q4_0)으로 cast 후 update. (forward_gen 의
        // `kv_dtype != F32` 분기 흡수.) `KVCache::update` 는 cast 를 하지 않고 입력이 이미 cache dtype
        // 임을 전제하므로, scatter fast-path 에 안 잡힌 비-F32 write 는 반드시 여기서 cast 해야 한다
        // (Q4_0 은 GPU 에서도 fast-path 부재라 이 경로). dtype 미일치 silent garbage 방지.
        if kv_dtype != DType::F32 {
            let memory = guard.cache.memory().ok_or_else(|| {
                anyhow::anyhow!(
                    "StandardFormat: non-F32 cast write requires a dynamic KVCache (memory=Some); \
                     fully pre-allocated caches built via KVCache::new() cannot allocate cast scratch"
                )
            })?;
            // scratch lazy 할당 (target dtype). 동일 shape 면 재사용(decode 연속 토큰=seq1 고정),
            // shape 가 바뀌면 재할당한다 — write_kv(decode seq=1)와 write_kv_batch(prefill seq>1)가
            // 같은 format 에서 cast 분기를 공유하므로(K/V 는 KV 불변식상 동일 shape), 첫 write 의
            // 크기로 굳히면 batch↔decode 혼용 시 cast zip 절단·update 오동작. (forward_gen 은
            // decode-only 라 단일 크기였음.)
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
            // 필드별 독립 mutable borrow (cache + scratch 동시 접근).
            let StandardFormatInner {
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

        // Correctness/CPU F32 경로: `KVCache` 는 GPU-buffer 보유라 `update` 가 내부 backend 로 자체 처리
        // (구 `update_kv_cache` 의 has_gpu_buffers 분기).
        guard.cache.update(new_k, new_v)
    }
}

impl KVCacheFormat for StandardFormat {
    fn idx(&self) -> usize {
        self.idx
    }

    fn current_pos(&self) -> usize {
        self.inner.lock().unwrap().cache.current_pos()
    }

    fn capacity(&self) -> usize {
        self.inner.lock().unwrap().cache.capacity()
    }

    fn write_kv(&self, new_k: &Tensor, new_v: &Tensor, backend: &dyn Backend) -> Result<()> {
        // decode (seq_len=1) — GPU fused cast+scatter fast-path 게이팅 가능.
        self.write_inner(new_k, new_v, backend, true)
    }

    fn write_kv_batch(&self, new_k: &Tensor, new_v: &Tensor, backend: &dyn Backend) -> Result<()> {
        // prefill (seq_len>1) — C3(§9.1-BC1-CONTRACT): GPU prefill batch scatter fast-path 흡수 완료
        // (F32/F16 HeadMajor + supports gate). Q4_0 및 게이트 미충족·CPU 는 cast/update 폴백.
        self.write_inner(new_k, new_v, backend, false)
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
        let cache_seq_len = cache.current_pos();

        // ── prefill (seq_len>1): multi-token causal attention (C-1, §9.1-BC1 / ①-b) ──
        // decode delegate(attention_gen / attention_q4_gpu_fallback)는 single-query +
        // causal-mask 부재라 재사용 불가 → forward_prefill(forward.rs:259-585) attention 블록을
        // `prefill_attention` 으로 미러. effective_cache_len clamp 를 **우회**하고(전체 cache_seq_len
        // K + window 를 flash 내부 마스킹에 위임) q_start_pos = cache_seq_len - seq_len. prefill 은
        // score 누적 안 함(scores 무시 — forward_prefill 의 `_need_scores`/variance_collector 와 동일).
        if seq_len > 1 {
            let kv_capacity = cache.capacity();
            let kv_layout = cache.layout();
            let batch_size = q.shape().dims()[0];
            let q_start_pos = cache_seq_len - seq_len;
            let (k_cache, v_cache) = cache.view();
            let _ = scores;
            return prefill_attention(
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

        // ── decode (seq_len==1): 기존 경로 (byte-불변) ──
        // Sliding window: 최근 window 토큰으로 제한 (Gemma3 local). global 이면 전체.
        let effective_cache_len = match dims.window {
            Some(w) => cache_seq_len.min(w),
            None => cache_seq_len,
        };

        let (k_cache, v_cache) = cache.view();

        // Q4_0 + GPU: `backend.attention_gen` 은 GPU 에 Q4_0 dequant-attention 커널이 없어
        // BlockQ4_0 raw 바이트를 float 로 오독 → garbage. forward_gen 의 `attention_q4_gpu_fallback`
        // (GPU→CPU readback + dequant + attention + writeback)을 그대로 재사용해 흡수한다 (substep
        // 3c, DRY — 중복 0). `kv_start_pos` = forward_gen.rs:404 와 동일 식(window-clamp 시작 offset).
        // CpuBackend(is_gpu=false)에선 진입 안 함 → host 경로는 아래 attention_gen 유지(Q4_0 CPU arm).
        if cache.kv_dtype() == DType::Q4_0 && backend.is_gpu() {
            let kv_start_pos = cache_seq_len - effective_cache_len;
            let layout = cache.layout();
            let capacity = cache.capacity();
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

        // typed/F32 경로: backend.attention_gen 에 위임. CPU backend 는 F32/F16/Q4_0 을 dtype-aware
        // 하게 처리(default impl=F32/F16, CpuBackend override 가 Q4_0 등 흡수). GPU backend 는
        // 자기 커널로 dispatch(host 미검증 — device 검증은 substep 3c device round).
        //
        // NOTE: kivi-native(get_kivi_raw_buffers) 분기는 KIVIFormat 소관이라 여기 없음.
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

/// (M4-b) [`WeightedMerge`](가중치 baked) 를 `&mut KVCache` 에 in-place 적용한다.
///
/// d2o 의 `scatter_reduce_merge_layer_wide`(d2o_handler.rs)와 **bit-identical** 산술이다 — per
/// `WeightedMerge` per head `acc = into_weight·into[d] + Σ w·from[d]`(`into` 먼저, `from` 은 list
/// 순서). K 는 `k_buffer.dtype()`, V 는 `v_buffer.dtype()` 로 독립 디스패치(F32/F16/Q4_0). 위치는
/// compact 적용 직전(pre-compact) 논리 좌표. ADR-0004 §4(M4 정정) — Q4_0 merge 활성.
///
/// `from` 은 evicted(retain 아님), `into` 는 retained 라 서로/merge 간 겹치지 않아(evicted∉retained)
/// in-place 적용이 안전하다. 빈 `from` 은 skip.
pub(crate) fn apply_weighted_merges(cache: &mut KVCache, merges: &[WeightedMerge]) {
    if merges.is_empty() {
        return;
    }
    use crate::quant::{BlockQ4_0, QK4_0};
    use half::f16;

    let kv_heads = cache.kv_heads();
    let head_dim = cache.head_dim();
    let blocks_per_pos = head_dim / QK4_0; // Q4_0 분기에서만 사용

    for m in merges {
        if m.from.is_empty() {
            continue;
        }
        let into_w = m.into_weight;
        let from_pos: Vec<usize> = m.from.iter().map(|&(p, _)| p).collect();
        let from_w: Vec<f32> = m.from.iter().map(|&(_, w)| w).collect();

        for h in 0..kv_heads {
            // ── K (k_buffer.dtype() 디스패치) ──
            match cache.k_buffer.dtype() {
                DType::F32 => {
                    let into_off = cache.offset(m.into, h);
                    let from_offs: Vec<usize> =
                        from_pos.iter().map(|&p| cache.offset(p, h)).collect();
                    let k = cache.k_buffer.as_mut_slice::<f32>();
                    merge_row_weighted_f32(k, into_off, &from_offs, &from_w, into_w, head_dim);
                }
                DType::F16 => {
                    let into_off = cache.offset(m.into, h);
                    let from_offs: Vec<usize> =
                        from_pos.iter().map(|&p| cache.offset(p, h)).collect();
                    let k = cache.k_buffer.as_mut_slice::<f16>();
                    merge_row_weighted_f16(k, into_off, &from_offs, &from_w, into_w, head_dim);
                }
                DType::Q4_0 => {
                    let into_bo = cache.q4_block_offset(m.into, h, blocks_per_pos);
                    let from_bos: Vec<usize> = from_pos
                        .iter()
                        .map(|&p| cache.q4_block_offset(p, h, blocks_per_pos))
                        .collect();
                    let k = cache.k_buffer.as_mut_slice::<BlockQ4_0>();
                    merge_row_weighted_q4(k, into_bo, &from_bos, &from_w, into_w, blocks_per_pos);
                }
                _ => {}
            }

            // ── V (v_buffer.dtype() 독립 디스패치) ──
            match cache.v_buffer.dtype() {
                DType::F32 => {
                    let into_off = cache.offset(m.into, h);
                    let from_offs: Vec<usize> =
                        from_pos.iter().map(|&p| cache.offset(p, h)).collect();
                    let v = cache.v_buffer.as_mut_slice::<f32>();
                    merge_row_weighted_f32(v, into_off, &from_offs, &from_w, into_w, head_dim);
                }
                DType::F16 => {
                    let into_off = cache.offset(m.into, h);
                    let from_offs: Vec<usize> =
                        from_pos.iter().map(|&p| cache.offset(p, h)).collect();
                    let v = cache.v_buffer.as_mut_slice::<f16>();
                    merge_row_weighted_f16(v, into_off, &from_offs, &from_w, into_w, head_dim);
                }
                DType::Q4_0 => {
                    let into_bo = cache.q4_block_offset(m.into, h, blocks_per_pos);
                    let from_bos: Vec<usize> = from_pos
                        .iter()
                        .map(|&p| cache.q4_block_offset(p, h, blocks_per_pos))
                        .collect();
                    let v = cache.v_buffer.as_mut_slice::<BlockQ4_0>();
                    merge_row_weighted_q4(v, into_bo, &from_bos, &from_w, into_w, blocks_per_pos);
                }
                _ => {}
            }
        }
    }
}

#[inline]
fn merge_row_weighted_f32(
    buf: &mut [f32],
    into_off: usize,
    from_offs: &[usize],
    from_w: &[f32],
    into_w: f32,
    head_dim: usize,
) {
    for d in 0..head_dim {
        let mut acc = into_w * buf[into_off + d];
        for (idx, &fo) in from_offs.iter().enumerate() {
            acc += from_w[idx] * buf[fo + d];
        }
        buf[into_off + d] = acc;
    }
}

#[inline]
fn merge_row_weighted_f16(
    buf: &mut [half::f16],
    into_off: usize,
    from_offs: &[usize],
    from_w: &[f32],
    into_w: f32,
    head_dim: usize,
) {
    use half::f16;
    for d in 0..head_dim {
        let mut acc = into_w * buf[into_off + d].to_f32();
        for (idx, &fo) in from_offs.iter().enumerate() {
            acc += from_w[idx] * buf[fo + d].to_f32();
        }
        buf[into_off + d] = f16::from_f32(acc);
    }
}

/// Q4_0 가중 병합 — from 을 head_dim f32 로 dequant(블록 단위) 후, into 블록을 dequant→`*=into_w`→
/// `+= from_w·from` → `BlockQ4_0::quantize`. scatter_reduce_q4 와 동일.
#[inline]
fn merge_row_weighted_q4(
    blocks: &mut [crate::quant::BlockQ4_0],
    into_block_off: usize,
    from_block_offs: &[usize],
    from_w: &[f32],
    into_w: f32,
    blocks_per_pos: usize,
) {
    use crate::quant::{BlockQ4_0, QK4_0};
    // from 을 먼저 full dequant(immutable read) — into write 와 별개 버퍼.
    let from_deq: Vec<Vec<f32>> = from_block_offs
        .iter()
        .map(|&fbo| {
            let mut buf = vec![0.0f32; blocks_per_pos * QK4_0];
            for bi in 0..blocks_per_pos {
                let mut tmp = [0.0f32; QK4_0];
                blocks[fbo + bi].dequantize(&mut tmp);
                buf[bi * QK4_0..(bi + 1) * QK4_0].copy_from_slice(&tmp);
            }
            buf
        })
        .collect();

    for bi in 0..blocks_per_pos {
        let mut r = [0.0f32; QK4_0];
        blocks[into_block_off + bi].dequantize(&mut r);
        for v in r.iter_mut() {
            *v *= into_w;
        }
        let base = bi * QK4_0;
        for (idx, fbuf) in from_deq.iter().enumerate() {
            for i in 0..QK4_0 {
                r[i] += from_w[idx] * fbuf[base + i];
            }
        }
        blocks[into_block_off + bi] = BlockQ4_0::quantize(&r);
    }
}

/// prefill multi-token causal attention (C-1, §9.1-BC1 / ①-b).
///
/// `forward_prefill`(transformer_layer/forward.rs:259-585)의 attention 블록을 그대로 미러한다 —
/// **decode delegate(`attention_gen` / `attention_q4_gpu_fallback`)는 single-query + causal-mask
/// 부재라 multi-token prefill 에 재사용 불가**(bit-identical 검증 wfceex20u 정정 B). GPU
/// `flash_attention_prefill` 시도 → 미dispatch(Q4_0 / head_dim 미지원 / CPU)면 dtype별 dequant +
/// `flash_attention_forward_strided`(causal mask 는 `q_start_pos`). prefill 은 score 누적 안 함
/// (forward_prefill 의 `_need_scores` 동일) → scores 인자 없음. `window` 는 flash 내부 마스킹에
/// 위임(decode 진입부의 `effective_cache_len` clamp **우회** — 정정 C). variance_collector/profiler/
/// fallback warn 은 happy-path 미진입·수치-무관이라 생략. forward_prefill 무수정(additive fork) —
/// 중복은 host parity test 로 bit-identical 증명, Step 5(forward_prefill<C> 삭제)에서 자연 해소.
///
/// **Phase α-K ①-e**: `KIVIFormat::attention_into` 의 prefill arm 도 이 free fn 을 재사용한다
/// (`pub(crate)`). KIVI 는 multi-token prefill native 커널 부재라 dequantized view(`get_view`) +
/// 본 함수로 처리 — KIVI CPU(SeqMajor F32) / GPU(bits=16 HeadMajor, bits 2/4/8 assembled) 모두
/// `kv_layout`/`kv_capacity` 인자로 분기되므로 별도 경로 불요.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_attention(
    q: &Tensor,
    out: &mut Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    cache_seq_len: usize,
    kv_capacity: usize,
    batch_size: usize,
    kv_layout: crate::kv_cache_ops::KVLayout,
    q_start_pos: usize,
    window: Option<usize>,
    backend: &dyn Backend,
) -> Result<()> {
    use crate::kv_cache_ops::KVLayout;

    let is_gpu = backend.is_gpu();
    // GPU flash attention prefill — KV 버퍼가 실제 GPU 버퍼일 때만(CPU-only cache 는 fallback).
    let kv_is_gpu = k_cache.buffer().is_gpu_buffer();
    let gpu_dispatched = if is_gpu && kv_is_gpu {
        backend.flash_attention_prefill(
            q,
            k_cache,
            v_cache,
            out,
            n_heads_q,
            n_heads_kv,
            seq_len,
            cache_seq_len,
            head_dim,
            kv_capacity,
            batch_size,
            kv_layout == KVLayout::HeadMajor,
        )?
    } else {
        false
    };
    if gpu_dispatched {
        return Ok(());
    }

    // CPU attention fallback (GPU 미dispatch 포함).
    let is_device_only = is_gpu && q.as_ptr().is_null();
    let mut out_vec: Vec<f32> = Vec::new();
    {
        fn as_u8_mut(v: &mut [f32]) -> &mut [u8] {
            unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4) }
        }

        let mut q_vec = Vec::new();
        let mut k_vec = Vec::new();
        let mut v_vec = Vec::new();

        let (q_data, k_data, v_data, out_ptr) = if is_device_only {
            let read_to_f32 = |t: &Tensor, vec: &mut Vec<f32>| -> Result<()> {
                if t.dtype() == DType::Q4_0 {
                    use crate::quant::{BlockQ4_0, QK4_0};
                    let numel = t.numel();
                    let n_blocks = numel / QK4_0;
                    let byte_size = n_blocks * std::mem::size_of::<BlockQ4_0>();
                    let mut byte_vec = vec![0u8; byte_size];
                    backend.read_buffer(t, &mut byte_vec)?;
                    vec.resize(numel, 0.0);
                    let blocks = unsafe {
                        std::slice::from_raw_parts(byte_vec.as_ptr() as *const BlockQ4_0, n_blocks)
                    };
                    for i in 0..n_blocks {
                        let mut tmp = [0.0f32; QK4_0];
                        blocks[i].dequantize(&mut tmp);
                        vec[i * QK4_0..(i + 1) * QK4_0].copy_from_slice(&tmp);
                    }
                } else if t.dtype() == DType::F16 {
                    let numel = t.numel();
                    let byte_size = numel * 2;
                    let mut byte_vec = vec![0u8; byte_size];
                    backend.read_buffer(t, &mut byte_vec)?;
                    vec.resize(numel, 0.0);
                    unsafe {
                        crate::quant::f16_bulk::bulk_f16_to_f32(
                            byte_vec.as_ptr() as *const u16,
                            vec.as_mut_ptr(),
                            numel,
                        );
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        let f16_slice = unsafe {
                            std::slice::from_raw_parts(byte_vec.as_ptr() as *const half::f16, numel)
                        };
                        for i in 0..numel {
                            vec[i] = f16_slice[i].to_f32();
                        }
                    }
                } else {
                    vec.resize(t.numel(), 0.0);
                    backend.read_buffer(t, as_u8_mut(vec))?;
                }
                Ok(())
            };

            read_to_f32(q, &mut q_vec)?;
            read_to_f32(k_cache, &mut k_vec)?;
            read_to_f32(v_cache, &mut v_vec)?;

            out_vec.resize(out.numel(), 0.0);

            (&q_vec[..], &k_vec[..], &v_vec[..], &mut out_vec[..])
        } else if k_cache.dtype() == DType::Q4_0 {
            use crate::quant::{BlockQ4_0, QK4_0};
            let n_elems = if kv_layout == KVLayout::HeadMajor {
                n_heads_kv * kv_capacity * head_dim
            } else {
                cache_seq_len * n_heads_kv * head_dim
            };
            let n_blocks = n_elems / QK4_0;
            let k_q4 = unsafe {
                std::slice::from_raw_parts(k_cache.as_ptr() as *const BlockQ4_0, n_blocks)
            };
            let v_q4 = unsafe {
                std::slice::from_raw_parts(v_cache.as_ptr() as *const BlockQ4_0, n_blocks)
            };
            k_vec.resize(n_elems, 0.0f32);
            v_vec.resize(n_elems, 0.0f32);
            for i in 0..n_blocks {
                let mut tmp = [0.0f32; QK4_0];
                k_q4[i].dequantize(&mut tmp);
                k_vec[i * QK4_0..(i + 1) * QK4_0].copy_from_slice(&tmp);
                v_q4[i].dequantize(&mut tmp);
                v_vec[i * QK4_0..(i + 1) * QK4_0].copy_from_slice(&tmp);
            }
            (
                q.as_slice::<f32>(),
                &k_vec[..],
                &v_vec[..],
                out.as_mut_slice::<f32>(),
            )
        } else if k_cache.dtype() == DType::F16 {
            let n_elems = if kv_layout == KVLayout::HeadMajor {
                n_heads_kv * kv_capacity * head_dim
            } else {
                cache_seq_len * n_heads_kv * head_dim
            };
            let k_f16_ptr = k_cache.as_ptr() as *const u16;
            let v_f16_ptr = v_cache.as_ptr() as *const u16;
            k_vec.resize(n_elems, 0.0f32);
            v_vec.resize(n_elems, 0.0f32);
            unsafe {
                crate::quant::f16_bulk::bulk_f16_to_f32(k_f16_ptr, k_vec.as_mut_ptr(), n_elems);
                crate::quant::f16_bulk::bulk_f16_to_f32(v_f16_ptr, v_vec.as_mut_ptr(), n_elems);
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let k_f16 =
                    unsafe { std::slice::from_raw_parts(k_f16_ptr as *const half::f16, n_elems) };
                let v_f16 =
                    unsafe { std::slice::from_raw_parts(v_f16_ptr as *const half::f16, n_elems) };
                for i in 0..n_elems {
                    k_vec[i] = k_f16[i].to_f32();
                    v_vec[i] = v_f16[i].to_f32();
                }
            }
            (
                q.as_slice::<f32>(),
                &k_vec[..],
                &v_vec[..],
                out.as_mut_slice::<f32>(),
            )
        } else {
            (
                q.as_slice::<f32>(),
                k_cache.as_slice::<f32>(),
                v_cache.as_slice::<f32>(),
                out.as_mut_slice::<f32>(),
            )
        };

        for x in out_ptr.iter_mut() {
            *x = 0.0;
        }

        use crate::layers::attention::flash_attention_forward_strided;
        let is_head_major_pf = kv_layout == KVLayout::HeadMajor;
        let chunk_q_stride = seq_len * n_heads_q * head_dim;
        let chunk_out_stride = seq_len * n_heads_q * head_dim;
        let chunk_k_stride = kv_capacity * n_heads_kv * head_dim;
        let (k_pos_stride, kv_head_stride) = if is_head_major_pf {
            (head_dim, kv_capacity * head_dim)
        } else {
            (n_heads_kv * head_dim, head_dim)
        };

        for (b, out_batch) in out_ptr.chunks_mut(chunk_out_stride).enumerate() {
            let q_start = b * chunk_q_stride;
            let k_start = b * chunk_k_stride;
            let v_start = b * chunk_k_stride;
            let q_slice = &q_data[q_start..q_start + chunk_q_stride];
            let k_valid_len = if is_head_major_pf {
                n_heads_kv * kv_capacity * head_dim
            } else {
                cache_seq_len * n_heads_kv * head_dim
            };
            let k_slice = &k_data[k_start..k_start + k_valid_len];
            let v_slice = &v_data[v_start..v_start + k_valid_len];

            flash_attention_forward_strided(
                q_slice,
                k_slice,
                v_slice,
                out_batch,
                n_heads_q,
                n_heads_kv,
                seq_len,
                cache_seq_len,
                head_dim,
                n_heads_q * head_dim,
                k_pos_stride,
                k_pos_stride,
                n_heads_q * head_dim,
                kv_head_stride,
                q_start_pos,
                32,
                32,
                window,
            );
        }
    }

    if is_device_only {
        let out_bytes =
            unsafe { std::slice::from_raw_parts(out_vec.as_ptr() as *const u8, out_vec.len() * 4) };
        let dst_ptr = out.as_mut_ptr();
        if !dst_ptr.is_null() {
            // UMA / pinned memory: direct memcpy.
            unsafe {
                std::ptr::copy_nonoverlapping(out_bytes.as_ptr(), dst_ptr, out_bytes.len());
            }
        }
        #[cfg(feature = "opencl")]
        {
            // OpenCL device-only buffers need enqueue_write_buffer.
            if dst_ptr.is_null()
                && let Ok(dst_mem) = crate::backend::opencl::get_cl_mem(out.buffer().as_ref())
            {
                if let Some(ocl) = backend
                    .as_any()
                    .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
                {
                    unsafe {
                        ocl::core::enqueue_write_buffer(
                            &ocl.queue,
                            dst_mem,
                            true,
                            0,
                            out_bytes,
                            None::<&ocl::core::Event>,
                            None::<&mut ocl::core::Event>,
                        )?;
                    }
                } else {
                    anyhow::bail!("prefill flash_attn CPU fallback: backend not OpenCL");
                }
            }
        }
    }
    Ok(())
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
        let buf = Arc::new(SharedBuffer::new(data.len() * 4, DType::F32));
        let mut t = Tensor::new(Shape::new(dims), buf, Arc::new(CpuBackend::new()));
        t.as_mut_slice::<f32>().copy_from_slice(data);
        t
    }

    /// Build a SeqMajor F32 KVCache: [1, max_seq, kv_heads, head_dim].
    fn make_cache(max_seq: usize, kv_heads: usize, head_dim: usize) -> KVCache {
        let total = max_seq * kv_heads * head_dim;
        let k = f32_tensor(vec![1, max_seq, kv_heads, head_dim], &vec![0.0; total]);
        let v = f32_tensor(vec![1, max_seq, kv_heads, head_dim], &vec![0.0; total]);
        KVCache::new(k, v, max_seq)
    }

    fn f16_tensor(dims: Vec<usize>, data: &[f32]) -> Tensor {
        use half::f16;
        let buf = Arc::new(SharedBuffer::new(data.len() * 2, DType::F16));
        let mut t = Tensor::new(Shape::new(dims), buf, Arc::new(CpuBackend::new()));
        for (d, &s) in t.as_mut_slice::<f16>().iter_mut().zip(data.iter()) {
            *d = f16::from_f32(s);
        }
        t
    }

    /// Build a SeqMajor F16 *dynamic* KVCache with a real allocator (`memory=Some`),
    /// so the non-F32 cast scratch path can lazily allocate its scratch buffers.
    fn make_f16_dynamic_cache(max_seq: usize, kv_heads: usize, head_dim: usize) -> KVCache {
        use crate::memory::Memory;
        use crate::memory::galloc::Galloc;
        let total = max_seq * kv_heads * head_dim;
        let k = f16_tensor(vec![1, max_seq, kv_heads, head_dim], &vec![0.0f32; total]);
        let v = f16_tensor(vec![1, max_seq, kv_heads, head_dim], &vec![0.0f32; total]);
        let mem: Arc<dyn Memory> = Arc::new(Galloc::new());
        KVCache::new_dynamic(k, v, max_seq, max_seq, kv_heads, head_dim, mem)
    }

    #[test]
    fn test_geometry_delegates_to_kvcache() {
        let cache = make_cache(8, 2, 4);
        let fmt = StandardFormat::new(3, cache);
        assert_eq!(fmt.idx(), 3);
        assert_eq!(fmt.capacity(), 8);
        assert_eq!(fmt.current_pos(), 0);
    }

    #[test]
    fn test_take_put_inner_round_trip() {
        // Phase α-K BC (3d) S1: take_inner → put_inner 는 identity. 토큰 write 후 take 한 cache 가
        // 데이터·pos 를 보존하고, put 후 wrapper 가 다시 정상 접근 가능해야 한다(eviction UER seam).
        let kv_heads = 2;
        let head_dim = 4;
        let fmt = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));
        let token = vec![7.0f32; kv_heads * head_dim];
        let k = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
        let v = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
        fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 1);

        // take: 꺼낸 cache 가 데이터·pos 보존, wrapper 는 placeholder(pos=0) 보유.
        let taken = fmt.take_inner();
        assert_eq!(taken.current_pos, 1);
        assert_eq!(taken.k_buffer.as_slice::<f32>()[0], 7.0);
        assert_eq!(fmt.current_pos(), 0, "take 후 wrapper 는 placeholder");

        // put: 복귀하면 wrapper 가 원래 cache 를 다시 노출.
        fmt.put_inner(taken);
        assert_eq!(fmt.current_pos(), 1, "put 후 원래 cache 복귀");
        let guard = fmt.inner.lock().unwrap();
        assert_eq!(guard.cache.k_buffer.as_slice::<f32>()[0], 7.0);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_plan_geometry_delegates_and_zeroes_residual() {
        // (3p) ④-a: plan_geometry()가 inner KVCache current_pos/capacity 를 정확히 위임하고
        // standard 의 res_pos/q2_tokens 는 0 이어야 한다.
        let fmt = StandardFormat::new(0, make_cache(8, 2, 4));
        let g = fmt.plan_geometry();
        assert_eq!(g.capacity, 8);
        assert_eq!(g.current_pos, 0);
        assert_eq!(g.res_pos, 0);
        assert_eq!(g.q2_tokens, 0);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_plan_advance_bumps_current_pos() {
        // (3p) ④-a: plan_advance(n) 후 plan_geometry().current_pos 가 증가해야 한다
        // (execute 의 레이어 끝 advance 미러).
        let kv_heads = 2;
        let head_dim = 4;
        let fmt = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));

        // ensure_capacity is not required for advance_pos (pure position bump).
        fmt.plan_advance(1);
        assert_eq!(fmt.plan_geometry().current_pos, 1);
        assert_eq!(
            fmt.current_pos(),
            1,
            "plan_advance must mutate the same cache"
        );

        fmt.plan_advance(2);
        assert_eq!(fmt.plan_geometry().current_pos, 3);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_plan_lock_reads_buffer() {
        // (3p) ④-a: plan_lock() guard seam — build_plan 가 KV buffer(`k_buffer`)에
        // 도달하는 경로(guard 를 잡고 `&KVCache` 슬라이스를 만들어 byte-identical build_plan
        // 본문을 재사용).
        let kv_heads = 1;
        let head_dim = 2;
        let fmt = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));

        // write one token = [7, 7], then read it back through the guard seam.
        let t = vec![7.0f32; kv_heads * head_dim];
        let k = f32_tensor(vec![1, 1, kv_heads, head_dim], &t);
        let v = f32_tensor(vec![1, 1, kv_heads, head_dim], &t);
        fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();

        let guard = fmt.plan_lock();
        assert_eq!(guard.cache.capacity(), 8);
        assert_eq!(guard.cache.k_buffer.as_slice::<f32>()[0], 7.0);
    }

    #[test]
    fn test_write_kv_advances_pos() {
        let kv_heads = 2;
        let head_dim = 4;
        let fmt = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));

        // single-token write: [1, 1, kv_heads, head_dim]
        let token = vec![1.0f32; kv_heads * head_dim];
        let k = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
        let v = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
        fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 1);

        // batch write: 2 tokens
        let batch = vec![2.0f32; 2 * kv_heads * head_dim];
        let kb = f32_tensor(vec![1, 2, kv_heads, head_dim], &batch);
        let vb = f32_tensor(vec![1, 2, kv_heads, head_dim], &batch);
        fmt.write_kv_batch(&kb, &vb, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 3);
    }

    #[test]
    fn test_write_kv_f16_casts_f32_input() {
        use half::f16;
        // F16 cache + CpuBackend(is_gpu()==false) → 비-F32 cast 분기(GPU scatter fast-path 미진입).
        // F32 입력이 F16 으로 cast 되어 저장되는지 검증 — `KVCache::update` 는 cast 를 안 하므로
        // 이 흡수가 빠지면 dtype 미일치 silent garbage. (forward_gen 의 `kv_dtype != F32` 흡수.)
        let kv_heads = 2;
        let head_dim = 4;
        let row = kv_heads * head_dim;
        let fmt = StandardFormat::new(0, make_f16_dynamic_cache(8, kv_heads, head_dim));

        // F16 로 정확히 표현 가능한 값(0.5 배수).
        let token0: Vec<f32> = (0..row).map(|i| (i as f32) * 0.5).collect();
        let k0 = f32_tensor(vec![1, 1, kv_heads, head_dim], &token0);
        let v0 = f32_tensor(vec![1, 1, kv_heads, head_dim], &token0);
        fmt.write_kv(&k0, &v0, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 1);

        // 두 번째 토큰 — lazy scratch 재사용 경로(k_cast/v_cast 이미 Some).
        let token1: Vec<f32> = (0..row).map(|i| (i as f32) + 1.0).collect();
        let k1 = f32_tensor(vec![1, 1, kv_heads, head_dim], &token1);
        let v1 = f32_tensor(vec![1, 1, kv_heads, head_dim], &token1);
        fmt.write_kv(&k1, &v1, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 2);

        // F16 buffer 검증: SeqMajor 라 pos*row + idx.
        let guard = fmt.inner.lock().unwrap();
        let k16 = guard.cache.k_buffer.as_slice::<f16>();
        let v16 = guard.cache.v_buffer.as_slice::<f16>();
        for (i, &exp) in token0.iter().enumerate() {
            assert!(
                (k16[i].to_f32() - exp).abs() < 1e-3,
                "pos0 K[{i}] expected {exp}, got {}",
                k16[i].to_f32()
            );
            assert!((v16[i].to_f32() - exp).abs() < 1e-3);
        }
        for (i, &exp) in token1.iter().enumerate() {
            assert!(
                (k16[row + i].to_f32() - exp).abs() < 1e-3,
                "pos1 K[{i}] expected {exp}, got {}",
                k16[row + i].to_f32()
            );
            assert!((v16[row + i].to_f32() - exp).abs() < 1e-3);
        }
    }

    #[test]
    fn test_write_kv_f16_batch_then_decode_reallocs_scratch() {
        use half::f16;
        // write_kv_batch(seq=2) 가 cast scratch 를 seq=2 크기로 굳힌 뒤 write_kv(seq=1) 가 와도
        // scratch 가 shape 변화에 맞춰 재할당되어 둘 다 정확해야 한다(가드 부재 시 cast zip 절단).
        let kv_heads = 2;
        let head_dim = 4;
        let row = kv_heads * head_dim;
        let fmt = StandardFormat::new(0, make_f16_dynamic_cache(8, kv_heads, head_dim));

        // prefill batch: 2 tokens. token@pos p = 0.5*(p+1) 균일.
        let batch: Vec<f32> = (0..2 * row)
            .map(|i| if i < row { 0.5 } else { 1.0 })
            .collect();
        let kb = f32_tensor(vec![1, 2, kv_heads, head_dim], &batch);
        let vb = f32_tensor(vec![1, 2, kv_heads, head_dim], &batch);
        fmt.write_kv_batch(&kb, &vb, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 2);

        // decode single: shape [1,1,...] — scratch 재할당 트리거.
        let dec = vec![2.5f32; row];
        let kd = f32_tensor(vec![1, 1, kv_heads, head_dim], &dec);
        let vd = f32_tensor(vec![1, 1, kv_heads, head_dim], &dec);
        fmt.write_kv(&kd, &vd, &CpuBackend::new()).unwrap();
        assert_eq!(fmt.current_pos(), 3);

        let guard = fmt.inner.lock().unwrap();
        let k16 = guard.cache.k_buffer.as_slice::<f16>();
        // pos0 = 0.5, pos1 = 1.0 (batch), pos2 = 2.5 (decode).
        for i in 0..row {
            assert!(
                (k16[i].to_f32() - 0.5).abs() < 1e-3,
                "pos0[{i}]={}",
                k16[i].to_f32()
            );
            assert!(
                (k16[row + i].to_f32() - 1.0).abs() < 1e-3,
                "pos1[{i}]={}",
                k16[row + i].to_f32()
            );
            assert!(
                (k16[2 * row + i].to_f32() - 2.5).abs() < 1e-3,
                "pos2[{i}]={}",
                k16[2 * row + i].to_f32()
            );
        }
    }

    #[test]
    fn test_write_kv_f16_requires_dynamic_cache() {
        // 비-F32 cast 는 inner cache 의 allocator 가 필요. `KVCache::new()`(memory=None)로 만든
        // F16 cache 는 scratch 할당 불가 → 명시적 에러(silent 오동작 금지).
        let kv_heads = 1;
        let head_dim = 4;
        let total = 8 * kv_heads * head_dim;
        let buf_k = Arc::new(SharedBuffer::new(total * 2, DType::F16));
        let buf_v = Arc::new(SharedBuffer::new(total * 2, DType::F16));
        let k = Tensor::new(
            Shape::new(vec![1, 8, kv_heads, head_dim]),
            buf_k,
            Arc::new(CpuBackend::new()),
        );
        let v = Tensor::new(
            Shape::new(vec![1, 8, kv_heads, head_dim]),
            buf_v,
            Arc::new(CpuBackend::new()),
        );
        let fmt = StandardFormat::new(0, KVCache::new(k, v, 8)); // memory=None

        let token = vec![1.0f32; kv_heads * head_dim];
        let kt = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
        let vt = f32_tensor(vec![1, 1, kv_heads, head_dim], &token);
        let err = fmt.write_kv(&kt, &vt, &CpuBackend::new());
        assert!(
            err.is_err(),
            "F16 cast on pre-allocated (memory=None) cache must error"
        );
    }

    #[test]
    fn test_write_kv_batch_f32_matches_sequential_decode() {
        // C3 (§9.1-BC1-CONTRACT): multi-token write_kv_batch must produce a buffer
        // bit-identical to writing the same tokens one-by-one via write_kv (decode).
        // host(CpuBackend, is_gpu=false) → cast/update fallback covers correctness;
        // GPU scatter fast-path is device-verified.
        let kv_heads = 2;
        let head_dim = 4;
        let row = kv_heads * head_dim;
        let seq = 3;

        // distinct per-(token, elem) values, exactly F32-representable.
        let batch: Vec<f32> = (0..seq * row).map(|i| i as f32).collect();

        // (A) batch write.
        let fmt_batch = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));
        let kb = f32_tensor(vec![1, seq, kv_heads, head_dim], &batch);
        let vb = f32_tensor(vec![1, seq, kv_heads, head_dim], &batch);
        fmt_batch
            .write_kv_batch(&kb, &vb, &CpuBackend::new())
            .unwrap();
        assert_eq!(fmt_batch.current_pos(), seq);

        // (B) reference: same tokens written one at a time via write_kv (decode).
        let fmt_seq = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));
        for s in 0..seq {
            let tok = &batch[s * row..(s + 1) * row];
            let k = f32_tensor(vec![1, 1, kv_heads, head_dim], tok);
            let v = f32_tensor(vec![1, 1, kv_heads, head_dim], tok);
            fmt_seq.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        }
        assert_eq!(fmt_seq.current_pos(), seq);

        // K/V buffers must be byte-identical.
        let gb = fmt_batch.inner.lock().unwrap();
        let gs = fmt_seq.inner.lock().unwrap();
        let kb_buf = gb.cache.k_buffer.as_slice::<f32>();
        let ks_buf = gs.cache.k_buffer.as_slice::<f32>();
        let vb_buf = gb.cache.v_buffer.as_slice::<f32>();
        let vs_buf = gs.cache.v_buffer.as_slice::<f32>();
        assert_eq!(kb_buf, ks_buf, "K batch buffer != sequential-decode buffer");
        assert_eq!(vb_buf, vs_buf, "V batch buffer != sequential-decode buffer");
    }

    #[test]
    fn test_write_kv_batch_f16_matches_sequential_decode() {
        use half::f16;
        // F16 cache: batch write goes through the non-F32 cast path on CpuBackend
        // (supports_kv_scatter_batch()==false). Must equal sequential decode writes.
        let kv_heads = 2;
        let head_dim = 4;
        let row = kv_heads * head_dim;
        let seq = 3;

        // 0.5-multiples so values are exactly representable in F16.
        let batch: Vec<f32> = (0..seq * row).map(|i| (i as f32) * 0.5).collect();

        // (A) batch write.
        let fmt_batch = StandardFormat::new(0, make_f16_dynamic_cache(8, kv_heads, head_dim));
        let kb = f32_tensor(vec![1, seq, kv_heads, head_dim], &batch);
        let vb = f32_tensor(vec![1, seq, kv_heads, head_dim], &batch);
        fmt_batch
            .write_kv_batch(&kb, &vb, &CpuBackend::new())
            .unwrap();
        assert_eq!(fmt_batch.current_pos(), seq);

        // (B) reference: sequential decode writes.
        let fmt_seq = StandardFormat::new(0, make_f16_dynamic_cache(8, kv_heads, head_dim));
        for s in 0..seq {
            let tok = &batch[s * row..(s + 1) * row];
            let k = f32_tensor(vec![1, 1, kv_heads, head_dim], tok);
            let v = f32_tensor(vec![1, 1, kv_heads, head_dim], tok);
            fmt_seq.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        }
        assert_eq!(fmt_seq.current_pos(), seq);

        let gb = fmt_batch.inner.lock().unwrap();
        let gs = fmt_seq.inner.lock().unwrap();
        let kb_buf = gb.cache.k_buffer.as_slice::<f16>();
        let ks_buf = gs.cache.k_buffer.as_slice::<f16>();
        let vb_buf = gb.cache.v_buffer.as_slice::<f16>();
        let vs_buf = gs.cache.v_buffer.as_slice::<f16>();
        assert_eq!(
            kb_buf, ks_buf,
            "F16 K batch buffer != sequential-decode buffer"
        );
        assert_eq!(
            vb_buf, vs_buf,
            "F16 V batch buffer != sequential-decode buffer"
        );
    }

    #[test]
    fn test_attention_into_f32_uniform() {
        // current_pos==0 is illegal for softmax; write 2 identical tokens so
        // softmax is uniform and output = the (identical) V row.
        let kv_heads = 1;
        let head_dim = 4;
        let n_heads_q = 1;
        let fmt = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));

        let k_row = vec![0.0f32; head_dim]; // zero K → all scores equal → uniform softmax
        let v_row = vec![5.0f32; head_dim];
        for _ in 0..2 {
            let k = f32_tensor(vec![1, 1, kv_heads, head_dim], &k_row);
            let v = f32_tensor(vec![1, 1, kv_heads, head_dim], &v_row);
            fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        }
        assert_eq!(fmt.current_pos(), 2);

        let q = f32_tensor(vec![1, 1, n_heads_q, head_dim], &vec![1.0; head_dim]);
        let mut out = f32_tensor(vec![1, 1, n_heads_q, head_dim], &vec![0.0; head_dim]);
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

        // Uniform attention over identical V rows → out == V row.
        let o = out.as_slice::<f32>();
        for &x in o {
            assert!((x - 5.0).abs() < 1e-4, "expected 5.0, got {x}");
        }
        // post-softmax scores: 2 equal weights summing to 1.
        assert!((scores[0] - 0.5).abs() < 1e-4);
        assert!((scores[1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_prefill_attention_causal_uniform() {
        // C-1 (①-b): multi-token prefill attention via attention_into(seq_len>1).
        // K=0 → 모든 score 0 → uniform softmax. V[pos]=pos (broadcast). causal mask 로
        // query row r 은 cache pos 0..=r 만 attend → out[r] = mean(0..=r) = r/2.
        // write_kv_batch(prefill write) + attention_into(prefill arm) 합동 검증.
        let kv_heads = 1;
        let head_dim = 4;
        let n_heads_q = 1;
        let seq = 4;
        let fmt = StandardFormat::new(0, make_cache(16, kv_heads, head_dim));
        let backend = CpuBackend::new();

        let k_data = vec![0.0f32; seq * kv_heads * head_dim];
        let mut v_data = vec![0.0f32; seq * kv_heads * head_dim];
        for p in 0..seq {
            for d in 0..head_dim {
                v_data[p * kv_heads * head_dim + d] = p as f32;
            }
        }
        let kb = f32_tensor(vec![1, seq, kv_heads, head_dim], &k_data);
        let vb = f32_tensor(vec![1, seq, kv_heads, head_dim], &v_data);
        fmt.write_kv_batch(&kb, &vb, &backend).unwrap();
        assert_eq!(fmt.current_pos(), seq);

        // q 값은 무관(K=0 → score 0). out = [1, seq, n_heads_q*head_dim].
        let q = f32_tensor(
            vec![1, seq, n_heads_q, head_dim],
            &vec![1.0; seq * n_heads_q * head_dim],
        );
        let mut out = f32_tensor(
            vec![1, seq, n_heads_q * head_dim],
            &vec![0.0; seq * n_heads_q * head_dim],
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
            let expected = r as f32 / 2.0; // mean(0..=r)
            for d in 0..head_dim {
                let got = o[r * head_dim + d];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "row {r} d {d}: expected {expected}, got {got}"
                );
            }
        }
    }

    #[test]
    fn test_attention_into_window_clamps_len() {
        // window=1 must restrict effective_cache_len to 1 (only first token seen
        // by backend.attention_gen). Verify scores buffer reflects single token.
        let kv_heads = 1;
        let head_dim = 4;
        let n_heads_q = 1;
        let fmt = StandardFormat::new(0, make_cache(8, kv_heads, head_dim));

        for p in 0..3 {
            let t = vec![p as f32; head_dim];
            let k = f32_tensor(vec![1, 1, kv_heads, head_dim], &t);
            let v = f32_tensor(vec![1, 1, kv_heads, head_dim], &t);
            fmt.write_kv(&k, &v, &CpuBackend::new()).unwrap();
        }

        let q = f32_tensor(vec![1, 1, n_heads_q, head_dim], &vec![1.0; head_dim]);
        let mut out = f32_tensor(vec![1, 1, n_heads_q, head_dim], &vec![0.0; head_dim]);
        let backend = CpuBackend::new();
        let mut scores = vec![0.0f32; n_heads_q * 3];

        fmt.attention_into(
            &q,
            &backend,
            &mut out,
            AttnDims {
                n_heads_q,
                window: Some(1),
            },
            Some(&mut scores),
        )
        .unwrap();

        // window=1 → only 1 token attended → score[0]=1.0, output = token0 (zeros).
        assert!((scores[0] - 1.0).abs() < 1e-4);
        let o = out.as_slice::<f32>();
        for &x in o {
            assert!(x.abs() < 1e-4, "token0 is all zeros, got {x}");
        }
    }
}
