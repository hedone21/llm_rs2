//! L2 추론 파이프라인 확장점 — `PipelineStage` 패밀리.
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §5 (PipelineStage 모델).
//!
//! **Phase β-2 배선 상태**: `session/decode_loop.rs` 의 `prefill`/`run` 에 9 phase 가
//! 발화된다 — PrefillStart·PrefillEnd (prefill 2종), DecodeStart·PreForward·PostForward·
//! PreSample·PostSample·DecodeEnd (per-token 6종), Finalize. `run_until_stop` 은 β-6
//! 에서 통합 예정 (현재 미발화).
//!
//! **β-3 추가**: PreEviction·PostEviction 은 `run()` 의 (a.6)↔(b)↔(c) 슬롯에서 발화됨
//! (run()만 — `run_until_stop`/chat 은 β-6). `EvictionStage`(stages/kv/eviction.rs)가 PreEviction
//! 에서 UER 로 force-evict 하고, driver 가 held handle 의 `current_pos` 를 query 해 pos-환류
//! (§5.2.1 (가)).
//!
//! **미발화 orphan (§5.2.1 (라))**: TurnStart·TurnEnd·SessionStart·SessionEnd 는 β-6,
//! PreLayer·PostLayer·Fine 는 β 범위 밖(`INV-HOTPATH-DISPATCH` layer-tier dyn 금지) — enum
//! variant 로 정의만 존재.
//!
//! 거버넌스: §1.2 Mechanism over policy(순서·안전 = stage 작성자 책임, §5.3),
//! `INV-DECODE-STAGE-001/004/005/006/007`, `INV-STAGE-LAYER-HANDLE`, `INV-HOTPATH-DISPATCH`.

use llm_shared::Level;

use crate::observability::profile::OpProfiler;

/// system 압력 scalar (0–100).
///
/// 모든 system 압력(memory·thermal·energy)의 *magnitude* 를 `PressureSource` impl 이 단일
/// scalar 로 융합한 값이다 — 의도적 lossy 일원화 (§5.1 R1, `docs/adr/0002`). Stage 는
/// 출처를 구분하지 않는다. scalar 로 환원 불가한 *mode* 명령(switch/suspend)은 이 값이 아니라
/// 이산 채널(`CommandSource`, §5.4)로 흐른다 — `Pressure` 는 graded 입력만 담는다.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Pressure(u8);

impl Pressure {
    /// 0–100 으로 clamp 하여 생성.
    pub fn new(raw: u8) -> Self {
        Pressure(raw.min(100))
    }

    /// 원시 scalar (0–100).
    pub fn raw(self) -> u8 {
        self.0
    }

    /// 4-level 강등 (구 `PressureLevel`/`llm_shared::Level` 흡수처).
    ///
    /// Pressure scalar(0–100) → `Level` 매핑. 이 cutoff(0-49/50-74/75-89/90-100) 는
    /// canonical 이며, `Pressure` 를 생산하는 source(`LocalPressureSource` 등)는 이 band
    /// 경계에 정확히 떨어지도록 scalar 를 산출한다(`Pressure::from_mem_available` 참조).
    pub fn band(self) -> Level {
        match self.0 {
            0..=49 => Level::Normal,
            50..=74 => Level::Warning,
            75..=89 => Level::Critical,
            _ => Level::Emergency,
        }
    }

    /// `MemAvailable`(bytes) → `Pressure` — 구 `cache_manager::determine_pressure_level`
    /// 계단 산식의 정본 이식.
    ///
    /// 구 산식 (threshold = `t`):
    /// - `mem >= t` → Normal
    /// - `t/2 <= mem < t` → Warning
    /// - `t/4 <= mem < t/2` → Critical
    /// - `mem < t/4` → Emergency
    ///
    /// 각 구간을 `band()` cutoff 의 하한값(0/50/75/90)으로 사상해 4-level↔band 전사 매핑이
    /// 경계값(`mem == t`, `t/2`, `t/4`)에서 구 enum 과 정확히 일치하게 한다 — 구 산식과
    /// 동일한 비교 연산자(`>=`)를 그대로 사용한다. `LocalPressureSource` 와 `CacheManager`
    /// 가 이 단일 함수를 공유해 cutoff 거처를 일원화한다 (§5.1, β-5).
    pub fn from_mem_available(mem_available: usize, threshold_bytes: usize) -> Self {
        if mem_available >= threshold_bytes {
            Pressure(0) // Normal
        } else if mem_available >= threshold_bytes / 2 {
            Pressure(50) // Warning
        } else if mem_available >= threshold_bytes / 4 {
            Pressure(75) // Critical
        } else {
            Pressure(90) // Emergency
        }
    }
}

/// 연속(graded) system 압력 source — manager IPC 수신 / 엔진 자율 계산 / 3rd-party (§5.4).
///
/// 소비 Stage 는 어느 source 인지 구분하지 않는다. construction 시점 보유(`Arc<dyn PressureSource>`,
/// 교체 지점). 구체 impl(`ManagerPressureSource` / `LocalPressureSource`)는 Phase α-W-3 에서
/// `resilience/`·`session/` 에 정착한다.
pub trait PressureSource: Send + Sync {
    fn pressure(&self) -> Pressure;
}

/// per-step read-only 값 (Copy, borrow 0) — §5.1 G5-2.
///
/// v1 `StepCtx`(`session/traits.rs`) 5필드 중 `pos`/`decode_step` 만 승계. 드롭: `prev_token`
/// (샘플러 도메인 — `TokenSampler` 별도 생존), `kv_capacity`(register 시점 보유한 Format handle
/// 에서 query — god-ctx 회피, `INV-STAGE-LAYER-HANDLE`), `stop_requested`(`StageOutcome::Stop`
/// 반환으로 정지 → borrow 제거로 `Copy` 성립).
#[derive(Debug, Clone, Copy)]
pub struct StepInfo {
    /// 시퀀스 절대 위치 (구 `StepCtx.pos`) — eviction/RoPE/kv 소비.
    pub pos: usize,
    /// decode 반복 카운터 0-based (구 `StepCtx.decode_step`) — OneShot/주기 stage.
    pub decode_step: usize,
    /// system 압력 scalar (pluggable `PressureSource` 출력).
    pub pressure: Pressure,
    // 승격 trigger: observe Stage 가 토큰 스트림 소비 요구 시 `prev_token: u32` 추가
    //               (driver 가 이미 보유 — ripple 0, trait·소비자 무변경).
}

/// Stage 가 받는 mutation 권한 컨텍스트 — 2 field 슬림 (§5.1, `INV-DECODE-STAGE-006`).
///
/// `kv`/`weights` field 없음(god ctx 회피, `INV-STAGE-LAYER-HANDLE`) — Stage 는 자기 Format
/// handle 을 register 시점 보관한다(wiring 표준 §3.6). `step` 은 read-only 값, `profiler` 가
/// 유일 mutable 자원.
pub struct StageContext<'a> {
    /// read-only per-step 값.
    pub step: StepInfo,
    /// 유일 mutable 자원.
    pub profiler: &'a mut OpProfiler,
}

/// stage 수명 (§5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageLifecycle {
    /// 세션 내내 상주 (pressure-driven).
    Persistent,
    /// 1회 실행 후 GC (command-driven 명령, §5.4).
    OneShot,
}

/// decode loop 정지 사유.
///
/// (v1 `session::traits::StopReason` 와 별개 타입 — Phase β 에서 수렴한다.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EosToken,
    BudgetExhausted,
    StopConditionMet,
    CommandRequested,
}

/// stage 실행 결과 (§5.1, `INV-DECODE-STAGE-004`).
pub enum StageOutcome {
    /// 다음 stage 진행.
    Continue,
    /// OneShot stage 소비 완료 → GC (Persistent 는 반환 금지, `INV-DECODE-STAGE-007`).
    Consumed,
    /// decode loop 정지.
    Stop(StopReason),
}

/// 추론 lifecycle 단계 — canonical variant 목록 (SSOT).
///
/// spec/41 `INV-DECODE-STAGE-001` 은 본 목록을 참조한다. tier 주석은 `INV-HOTPATH-DISPATCH`
/// 의 dispatch tier 와 정합한다.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecyclePhase {
    // 경계 (boundary tier — 1×/generation)
    SessionStart,
    SessionEnd,
    TurnStart,
    TurnEnd,
    // prefill
    PrefillStart,
    PrefillChunkBoundary,
    PrefillEnd,
    // decode step (1×/token)
    DecodeStart,
    PreForward,
    PostForward,
    PreSample,
    PostSample,
    DecodeEnd,
    // cross-cutting stage hook (mutation 허용)
    PreEviction,
    PostEviction,
    PreSwap,
    PostSwapBefore,
    PostSwapAfter,
    // per-layer (N×/token) — mutation 금지 (INV-DECODE-STAGE-001)
    PreLayer,
    PostLayer,
    /// sub-step 식별자. 구체 shape(layer idx 등)는 Phase β 배선 시점 확정 (placeholder).
    Fine(&'static str),
    Finalize,
}

/// 확장점 trait — v1 5-hook(StepHook/PhaseHook/LayerBoundaryHook/DecodeObserver/StopCondition)
/// + 7-trait 일부를 단일 trait 으로 통합(§5).
///
/// 순서·안전은 통합자(stage 작성자) 책임이다 (§1.2 Mechanism over policy, §5.3). 프레임워크는
/// stage 등록·순회 메커니즘만 제공하고, "틀린 구성 금지" policy 도 "어떤 순서에서도 crash-safe"
/// 보장도 하지 않는다. Stage 는 자기 Format handle 을 register 시점 보관한다(§3.4 3종 handle).
pub trait PipelineStage: Send + Sync {
    fn name(&self) -> &str;

    fn lifecycle(&self) -> StageLifecycle {
        StageLifecycle::Persistent
    }

    fn on_phase(
        &self,
        phase: &LifecyclePhase,
        ctx: &mut StageContext<'_>,
    ) -> anyhow::Result<StageOutcome>;
}

/// stage 순회 dispatcher (L4 `PipelineRegistry` 가 impl — §5.2, `INV-LAYER-006`).
///
/// submit 순서로 순회, 각 stage 가 `on_phase` 안에서 자기 phase self-filter.
/// `Continue`→진행 / `Consumed`→OneShot GC / `Stop(r)`→break / `Err`→panic(fail-fast).
pub trait PipelineDispatcher: Send + Sync {
    fn dispatch(&self, phase: LifecyclePhase, ctx: &mut StageContext<'_>) -> Option<StopReason>;
}

#[cfg(test)]
mod pressure_band_tests {
    use super::*;

    /// 구 `cache_manager::determine_pressure_level` 산식의 정본 oracle (변경 전 코드 그대로).
    /// β-5 ripple 의 전사 등가 anchor — `Pressure::from_mem_available(..).band()` 와 대조한다.
    fn legacy_determine_level(mem_available: usize, threshold: usize) -> Level {
        if mem_available >= threshold {
            Level::Normal
        } else if mem_available >= threshold / 2 {
            Level::Warning
        } else if mem_available >= threshold / 4 {
            Level::Critical
        } else {
            Level::Emergency
        }
    }

    fn new_level(mem_available: usize, threshold: usize) -> Level {
        Pressure::from_mem_available(mem_available, threshold).band()
    }

    /// 게이트 1: 4-level × {t, t÷2, t÷4} 경계 ±1 전수 — 구 산식 == 신 band() 전사.
    /// eviction/swap trigger 임계 불변 증명 (경계값에서 구 enum 과 신 band 동일 level).
    #[test]
    fn level_to_band_boundary_parity() {
        // 다양한 threshold (짝수/홀수/큰 값 — integer division 경계 보존 확인).
        let thresholds = [4usize, 100, 1024, 1_000_003, 8 * 1024 * 1024 * 1024];
        for &t in &thresholds {
            // 경계점 후보: 0, t/4, t/2, t 와 각 ±1.
            let edges = [
                0usize,
                (t / 4).saturating_sub(1),
                t / 4,
                t / 4 + 1,
                (t / 2).saturating_sub(1),
                t / 2,
                t / 2 + 1,
                t.saturating_sub(1),
                t,
                t + 1,
                t * 2,
            ];
            for &mem in &edges {
                assert_eq!(
                    new_level(mem, t),
                    legacy_determine_level(mem, t),
                    "level mismatch: mem={}, threshold={}",
                    mem,
                    t,
                );
            }
        }
    }

    /// 경계값 정확 사상: mem == t → Normal, mem == t/2 → Warning, mem == t/4 → Critical,
    /// mem < t/4 → Emergency. (band cutoff 의 하한값 매핑이 4-level 과 1:1.)
    #[test]
    fn exact_boundary_maps_to_band_lower_bounds() {
        let t = 1024usize;
        assert_eq!(new_level(t, t), Level::Normal);
        assert_eq!(new_level(t / 2, t), Level::Warning);
        assert_eq!(new_level(t / 4, t), Level::Critical);
        assert_eq!(new_level(t / 4 - 1, t), Level::Emergency);
        // band() cutoff 하한값 확인.
        assert_eq!(Pressure::new(0).band(), Level::Normal);
        assert_eq!(Pressure::new(50).band(), Level::Warning);
        assert_eq!(Pressure::new(75).band(), Level::Critical);
        assert_eq!(Pressure::new(90).band(), Level::Emergency);
    }
}
