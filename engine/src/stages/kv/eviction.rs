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

use llm_shared::Level;

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
    /// β-5: Persistent band-driven 발화 임계. `Some(min)` 이면 `PreEviction` 에서
    /// `ctx.step.pressure.band() >= min` 일 때만 prune 한다(미달 시 `Continue`, no-op).
    /// `None`(OneShot) 이면 무조건 발화(v1 AB-1 = command-driven, 압력 무관).
    min_band: Option<Level>,
    /// β-5: Persistent **episode edge-trigger** 무장 상태. band 가 `min_band` 를 상향 돌파한
    /// 에지에서 1회 발화 후 disarm, band 가 `min_band` 미만으로 떨어지면 re-arm — 압력 에피소드당
    /// prune 1회. 가드 없이는 지속 고압에서 매 step `force_evict(ratio)` 가 재발화해 캐시가
    /// floor 까지 나선 축소(churn — madvise/CacheEvent per-token spam)하는 퇴행 시맨틱이 된다.
    /// 압력 지속 중 캐시 재성장 시의 재prune 은 friction-triggered 후속 (doc).
    armed: std::sync::atomic::AtomicBool,
}

impl EvictionStage {
    /// command-driven OneShot eviction (v1 AB-1 동일 조건 — score-free `force_evict`).
    ///
    /// 1회 발화 후 `Consumed` 를 반환해 registry 가 GC 한다(sticky 재적용 방지 — v1 `evict_applied`
    /// 게이트 등가). 압력 무관(`min_band=None`).
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
            min_band: None,
            armed: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// β-5: pressure-driven Persistent eviction. 세션 내내 상주하며, `PreEviction` 에서
    /// `ctx.step.pressure.band() >= min_band` 일 때만 prune 한다(미달 시 `Continue`, no-op).
    ///
    /// **OneShot 과 同코드**: prune 본문([`run_eviction`](Self::run_eviction))은 공유하고,
    /// lifecycle(GC 여부)·발화 조건(band 게이트)만 분기한다. 같은 입력 캐시에서 band 충족 시의
    /// 발화 산출은 OneShot 과 byte-identical (unit `persistent_band_met_matches_one_shot`).
    pub fn persistent(
        handles: Vec<Arc<StandardFormat>>,
        cache_manager: Arc<Mutex<CacheManager>>,
        target_ratio: f32,
        min_band: Level,
    ) -> Self {
        Self {
            lifecycle: StageLifecycle::Persistent,
            handles,
            cache_manager,
            target_ratio,
            min_band: Some(min_band),
            armed: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// prune 본문 (UER, Unwrap-Evict-Rewrap) — OneShot/Persistent 공유.
    ///
    /// v1 try_evict(model_forward.rs:524-541)의 inner op 그대로. take_inner 로 inner cache 들을
    /// 연속 Vec 로 꺼내(W1 순서 보존) force_evict 후, Err/Ok 무관하게 put_inner 로 되돌린다
    /// (placeholder 폐기 — `?` 전파를 rewrap 이후로 미룸).
    fn run_eviction(&self) -> anyhow::Result<()> {
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
        Ok(())
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
        ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome> {
        // self-filter (§5.3): eviction 외 phase 는 무시.
        if *phase != LifecyclePhase::PreEviction {
            return Ok(StageOutcome::Continue);
        }

        // β-5: Persistent band 게이트 — 압력 미달이면 prune 없이 Continue (no-op) + re-arm.
        // OneShot(min_band=None) 은 무조건 발화 (command-driven, 압력 무관).
        if let Some(min) = self.min_band {
            use std::sync::atomic::Ordering;
            if ctx.step.pressure.band() < min {
                // 압력 에피소드 종료 → re-arm (다음 상향 돌파에서 재발화 가능).
                self.armed.store(true, Ordering::Relaxed);
                return Ok(StageOutcome::Continue);
            }
            // episode edge-trigger: 이번 에피소드에서 이미 발화했으면 no-op
            // (지속 고압에서 매 step force_evict 재발화 = floor 나선 축소 차단).
            if !self.armed.swap(false, Ordering::Relaxed) {
                return Ok(StageOutcome::Continue);
            }
        }

        self.run_eviction()?;

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
        make_ctx_with_pressure(profiler, 0)
    }

    fn make_ctx_with_pressure(profiler: &mut OpProfiler, pressure: u8) -> StageContext<'_> {
        StageContext {
            step: StepInfo {
                pos: 0,
                decode_step: 0,
                pressure: Pressure::new(pressure),
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

    // ── β-5: Persistent band-driven 발화 ──

    /// Persistent: band 충족(Warning 이상) 시 발화 → Continue + cache prune.
    #[test]
    fn persistent_band_met_fires_and_continues() {
        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let stage = EvictionStage::persistent(
            vec![handle.clone()],
            make_cache_manager(),
            TARGET_RATIO,
            Level::Warning,
        );

        let mut profiler = OpProfiler::new();
        // pressure=50 → band()=Warning >= min(Warning) → 발화.
        let mut ctx = make_ctx_with_pressure(&mut profiler, 50);
        let outcome = stage
            .on_phase(&LifecyclePhase::PreEviction, &mut ctx)
            .unwrap();
        assert!(
            matches!(outcome, StageOutcome::Continue),
            "Persistent 은 발화해도 Continue(상주, GC 안 함)"
        );
        assert!(
            handle.current_pos() < N_TOKENS,
            "band 충족 시 prune (got {})",
            handle.current_pos()
        );
    }

    /// Persistent: band 미달(Normal) 시 무발화 → Continue + cache 불변.
    #[test]
    fn persistent_band_unmet_is_noop() {
        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let stage = EvictionStage::persistent(
            vec![handle.clone()],
            make_cache_manager(),
            TARGET_RATIO,
            Level::Warning,
        );

        let mut profiler = OpProfiler::new();
        // pressure=49 → band()=Normal < min(Warning) → 무발화.
        let mut ctx = make_ctx_with_pressure(&mut profiler, 49);
        let outcome = stage
            .on_phase(&LifecyclePhase::PreEviction, &mut ctx)
            .unwrap();
        assert!(matches!(outcome, StageOutcome::Continue));
        assert_eq!(
            handle.current_pos(),
            N_TOKENS,
            "band 미달 시 prune 없음 (cache 불변)"
        );
    }

    /// 同코드 증명: 같은 입력 캐시에서 OneShot 발화 산출 == Persistent(band 충족) 발화 산출.
    /// `new_pos`(prune 후 current_pos)가 byte-identical 이어야 한다 (run_eviction 공유).
    #[test]
    fn persistent_band_met_matches_one_shot() {
        // OneShot 경로.
        let h_one = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let one = EvictionStage::one_shot(vec![h_one.clone()], make_cache_manager(), TARGET_RATIO);
        let mut p1 = OpProfiler::new();
        let mut c1 = make_ctx_with_pressure(&mut p1, 50);
        one.on_phase(&LifecyclePhase::PreEviction, &mut c1).unwrap();
        let one_shot_pos = h_one.current_pos();

        // Persistent(band 충족) 경로 — 동일 입력 캐시·CM·ratio.
        let h_per = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let per = EvictionStage::persistent(
            vec![h_per.clone()],
            make_cache_manager(),
            TARGET_RATIO,
            Level::Warning,
        );
        let mut p2 = OpProfiler::new();
        let mut c2 = make_ctx_with_pressure(&mut p2, 50);
        per.on_phase(&LifecyclePhase::PreEviction, &mut c2).unwrap();
        let persistent_pos = h_per.current_pos();

        assert_eq!(
            one_shot_pos, persistent_pos,
            "OneShot 과 Persistent(band 충족) prune 산출이 동일해야 함 (run_eviction 공유)"
        );
        assert!(
            persistent_pos < N_TOKENS,
            "양쪽 모두 실제 prune (비-vacuous)"
        );
    }

    /// β-5 episode edge-trigger: 지속 고압에서 prune 은 에피소드당 1회 — 매 step 재발화로
    /// floor 까지 나선 축소되는 퇴행 차단. band 가 min 미만으로 떨어지면 re-arm 되어
    /// 다음 상향 돌파에서 재발화한다.
    #[test]
    fn persistent_edge_trigger_once_per_episode() {
        let handle = Arc::new(StandardFormat::new(0, make_cache(N_TOKENS)));
        let stage = EvictionStage::persistent(
            vec![handle.clone()],
            make_cache_manager(),
            TARGET_RATIO,
            Level::Warning,
        );
        let mut profiler = OpProfiler::new();

        // 에피소드 1: 상향 돌파 → 1회 발화.
        let mut c = make_ctx_with_pressure(&mut profiler, 50);
        stage
            .on_phase(&LifecyclePhase::PreEviction, &mut c)
            .unwrap();
        let pos_after_first = handle.current_pos();
        assert!(pos_after_first < N_TOKENS, "에피소드 1 발화");

        // 지속 고압: 같은 에피소드 내 재dispatch → 무발화 (pos 불변).
        let mut c = make_ctx_with_pressure(&mut profiler, 60);
        stage
            .on_phase(&LifecyclePhase::PreEviction, &mut c)
            .unwrap();
        assert_eq!(
            handle.current_pos(),
            pos_after_first,
            "지속 고압에서 재발화 금지 (episode edge-trigger)"
        );

        // band 하강 → re-arm (무발화).
        let mut c = make_ctx_with_pressure(&mut profiler, 10);
        stage
            .on_phase(&LifecyclePhase::PreEviction, &mut c)
            .unwrap();
        assert_eq!(handle.current_pos(), pos_after_first, "Normal 에선 무발화");

        // 에피소드 2: 캐시 재성장 후 재돌파 → 재발화.
        handle.with_cache_mut(|cache| cache.current_pos = N_TOKENS);
        let mut c = make_ctx_with_pressure(&mut profiler, 80);
        stage
            .on_phase(&LifecyclePhase::PreEviction, &mut c)
            .unwrap();
        assert!(
            handle.current_pos() < N_TOKENS,
            "re-arm 후 두 번째 에피소드에서 재발화"
        );
    }
}
