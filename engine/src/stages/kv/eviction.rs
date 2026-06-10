//! `EvictionStage` — v2 `PipelineStage` 입주자 1호 (Phase β-3 commit B).
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §5.2.1 (가)(나) + roadmap β-3 commit B.
//!
//! `PreEviction` phase 에서 발화하여, register 시점 보유한 `Vec<Arc<StandardFormat>>` 핸들의
//! inner `KVCache` 를 **CacheManager UER**(`take_inner` → `force_evict` → `put_inner`)로 prune
//! 한다. 적용 경로가 v1 `ModelForward::try_evict`(model_forward.rs:505-548, AB-1)의 inner op 와
//! **byte-identical** 이라 madvise/CacheEvent/min-floor 회계가 그대로 보존된다(자기 비교 금지 —
//! 등가 게이트의 anchor 는 v1 `force_evict` 직접 호출).
//!
//! **pos 환류는 driver 책임**(§5.2.1 (가)): `StageOutcome` 는 무변경이고, driver(decode_loop.rs)가
//! eviction phase dispatch 후 held handle 의 `current_pos` 를 비교해 loop `pos` 를 갱신하고
//! `forward.on_kv_prune` 을 호출한다. Stage 는 cache 상태만 바꾼다.
//!
//! **Persistent⊕OneShot 同코드**: lifecycle 은 필드로만 분기한다. command-driven 1차 cut(v1 AB-1
//! 동일 조건 = score-free force_evict)은 [`EvictionStage::one_shot`] 로 생성하고, pressure
//! band-driven Persistent 변형은 β-5 에서 `on_phase` 의 `Continue` 반환만 다르게 구성하면 된다
//! (동일 UER 본문 재사용 — 본 파일 [`EvictionStage::on_phase`] 의 `match self.lifecycle`).

use std::sync::{Arc, Mutex};

use crate::pipeline::{LifecyclePhase, PipelineStage, StageContext, StageLifecycle, StageOutcome};
use crate::pressure::cache_manager::CacheManager;
use crate::pressure::kv_cache::KVCache;
use crate::pressure::standard_format::StandardFormat;

/// `PreEviction` phase 에서 CacheManager UER 로 force-evict 하는 Stage.
///
/// `CacheManager::force_evict` 는 `&self` 이지만, `PipelineStage: Send + Sync` 를 충족하려면
/// 내부 가변 상태가 `Sync` 여야 한다 → `Mutex<CacheManager>` 로 소유한다(per-dispatch 1 lock,
/// eviction = cold path 라 비용 무관). CacheManager 의 모든 필드(`Box<dyn SystemMonitor>`/
/// `HashMap<_, Box<dyn EvictionPolicy>>`/`Arc<SwapHandler>`)가 `Send + Sync` 라 `Mutex<CacheManager>`
/// 는 `Send + Sync` 다.
pub struct EvictionStage {
    lifecycle: StageLifecycle,
    /// register 시점 보유 (INV-STAGE-LAYER-HANDLE). enumerate 순서 == layer idx (W1 — D2O
    /// cross-layer 정확성 전제, `wrap_kv_caches` 와 동일 불변식).
    handles: Vec<Arc<StandardFormat>>,
    /// 적용 경로 = CacheManager UER (v1 `try_evict` 와 산출 동일).
    ///
    /// β-4: `Arc<Mutex<CacheManager>>` 공유. CommandDispatcher 가 KvEvict* directive 마다 새
    /// OneShot `EvictionStage` 를 만들지만, 단일 `CacheManager`(CLI 정책·sticky eviction 상태)를
    /// 모든 stage 가 공유해야 method-drop 시맨틱(3부)과 정책 일관성이 유지된다.
    cache_manager: Arc<Mutex<CacheManager>>,
    /// score-free force_evict 의 target ratio.
    target_ratio: f32,
}

impl EvictionStage {
    /// command-driven OneShot eviction (v1 AB-1 동일 조건 — score-free `force_evict`).
    ///
    /// 1회 발화 후 `Consumed` 를 반환해 registry 가 GC 한다(sticky 재적용 방지 — v1 `evict_applied`
    /// 게이트 등가). Persistent band-driven 변형(β-5)은 동일 핸들·CacheManager 로 `Persistent`
    /// lifecycle 만 바꿔 생성하면 되며, `on_phase` 본문(UER)은 그대로다.
    ///
    /// β-4: `cache_manager` 는 `Arc<Mutex<CacheManager>>` — CommandDispatcher 가 보유한 단일 CM 을
    /// directive 마다 새 OneShot stage 에 clone 주입한다(공유).
    pub fn one_shot(
        handles: Vec<Arc<StandardFormat>>,
        cache_manager: Arc<Mutex<CacheManager>>,
        target_ratio: f32,
    ) -> Self {
        Self {
            lifecycle: StageLifecycle::OneShot,
            handles,
            cache_manager,
            target_ratio,
        }
    }
}

impl PipelineStage for EvictionStage {
    fn name(&self) -> &str {
        "kv.eviction"
    }

    fn lifecycle(&self) -> StageLifecycle {
        self.lifecycle
    }

    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        _ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        // self-filter (§5.3): eviction 외 phase 는 무시.
        if *phase != LifecyclePhase::PreEviction {
            return Ok(StageOutcome::Continue);
        }

        // UER (Unwrap-Evict-Rewrap) — v1 try_evict(model_forward.rs:524-541)의 inner op 그대로.
        // take_inner 로 inner cache 들을 연속 Vec 로 꺼내(W1 순서 보존) force_evict 후, Err/Ok
        // 무관하게 put_inner 로 되돌린다(placeholder 폐기 — `?` 전파를 rewrap 이후로 미룸).
        let mut temp: Vec<KVCache> = self.handles.iter().map(|f| f.take_inner()).collect();
        let result = self
            .cache_manager
            .lock()
            .expect("EvictionStage CacheManager Mutex poisoned")
            .force_evict(&mut temp, self.target_ratio);
        for (f, c) in self.handles.iter().zip(temp) {
            f.put_inner(c);
        }
        // Err → dispatcher 가 panic (fail-fast 계약, INV-DECODE-STAGE-004).
        result?;

        match self.lifecycle {
            StageLifecycle::OneShot => Ok(StageOutcome::Consumed),
            StageLifecycle::Persistent => Ok(StageOutcome::Continue),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::DType;
    use crate::format::KVCacheFormat;
    use crate::memory::host::shared::SharedBuffer;
    use crate::observability::profile::OpProfiler;
    use crate::pipeline::{Pressure, StepInfo};
    use crate::pressure::eviction::sliding_window::SlidingWindowPolicy;
    use crate::resilience::sys_monitor::NoOpMonitor;
    use crate::shape::Shape;
    use crate::tensor::Tensor;

    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 32;
    const MAX_SEQ: usize = 128;
    /// ratio=0.3 → target_len=36, tokens_to_remove=84 ≥ MIN_EVICT_TOKENS(64) → 실제 prune.
    const N_TOKENS: usize = 120;
    const TARGET_RATIO: f32 = 0.3;

    /// SeqMajor F32 KVCache, current_pos = n_tokens.
    fn make_cache(n_tokens: usize) -> KVCache {
        let total = MAX_SEQ * KV_HEADS * HEAD_DIM;
        let k_buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
        let v_buf = Arc::new(SharedBuffer::new(total * 4, DType::F32));
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let shape = Shape::new(vec![1, MAX_SEQ, KV_HEADS, HEAD_DIM]);
        let k = Tensor::new(shape.clone(), k_buf, backend.clone());
        let v = Tensor::new(shape, v_buf, backend);
        let mut c = KVCache::new(k, v, MAX_SEQ);
        c.current_pos = n_tokens;
        c
    }

    fn make_cache_manager() -> Arc<Mutex<CacheManager>> {
        // sliding window(window=10, prefix=4): current>keep 면 force_evict 가 prune.
        let policy = Box::new(SlidingWindowPolicy::new(10, 4));
        Arc::new(Mutex::new(CacheManager::new(
            policy,
            Box::new(NoOpMonitor),
            usize::MAX,
            TARGET_RATIO,
        )))
    }

    fn make_ctx(profiler: &mut OpProfiler) -> StageContext<'_> {
        StageContext {
            step: StepInfo {
                pos: 0,
                decode_step: 0,
                pressure: Pressure::new(0),
            },
            profiler,
        }
    }

    /// PreEviction 외 phase 는 no-op(Continue) + cache 불변.
    #[test]
    fn non_eviction_phase_is_noop() {
        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let stage =
            EvictionStage::one_shot(vec![handle.clone()], make_cache_manager(), TARGET_RATIO);

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage
            .on_phase(&LifecyclePhase::DecodeStart, &mut ctx)
            .unwrap();
        assert!(matches!(outcome, StageOutcome::Continue));
        // current_pos 불변 (evict 미진입).
        assert_eq!(handle.current_pos(), N_TOKENS);
    }

    /// OneShot: PreEviction 1회 발화 → Consumed.
    #[test]
    fn one_shot_returns_consumed_on_eviction() {
        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let stage =
            EvictionStage::one_shot(vec![handle.clone()], make_cache_manager(), TARGET_RATIO);

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let outcome = stage
            .on_phase(&LifecyclePhase::PreEviction, &mut ctx)
            .unwrap();
        assert!(
            matches!(outcome, StageOutcome::Consumed),
            "OneShot eviction 은 Consumed 반환"
        );
        // sliding(window=10, prefix=4) prune → current_pos 감소.
        assert!(
            handle.current_pos() < N_TOKENS,
            "eviction 후 current_pos 감소해야 함 (got {})",
            handle.current_pos()
        );
    }

    /// take_inner 후 put_inner 로 inner 가 복원된다(handle 경유 재접근 가능).
    #[test]
    fn put_inner_restores_after_dispatch() {
        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let stage =
            EvictionStage::one_shot(vec![handle.clone()], make_cache_manager(), TARGET_RATIO);

        let mut profiler = OpProfiler::new();
        let mut ctx = make_ctx(&mut profiler);
        let _ = stage
            .on_phase(&LifecyclePhase::PreEviction, &mut ctx)
            .unwrap();

        // dispatch 후 inner 가 placeholder(0-len)가 아니라 실물 cache 여야 한다 —
        // take_inner 로 꺼내도 capacity 가 보존됨.
        let inner = handle.take_inner();
        assert_eq!(inner.capacity(), MAX_SEQ, "put_inner 가 실물 cache 를 복원");
    }
}
