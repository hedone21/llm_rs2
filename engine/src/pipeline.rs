//! L2 추론 파이프라인 확장점 — `PipelineStage` 패밀리.
//!
//! 설계 SSOT: `arch/pipeline_stage_design_v2.md` §5 (PipelineStage 모델).
//!
//! 본 파일은 **Phase α-W 에서 신설된 추가적(additive) 타입 어휘**다 — 아직 live decode
//! loop(`session/decode_loop.rs`, v1 7-trait)에 배선되지 않았다. v1 7-trait → 단일
//! `PipelineStage` 전환(DecodeLoop 재작성)은 Phase β 다 (§9). 따라서 이 파일의 추가는
//! 기존 동작을 바꾸지 않는다(byte-identical).
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
    /// **canonical cutoff 는 `LocalPressureSource`(Phase α-W-3)가 소유**한다 — 여기 default 는
    /// 구 `cache_manager::determine_pressure_level` / `MemoryStrategy` 이산 임계의 day-1 carry
    /// placeholder 이며, 실측 튜닝은 코드 시점이다 (G4 게이트 해제 —
    /// `handoff_design_concretization_2026_06_02`). 현재 `band()` 은 어떤 live 경로에도
    /// 배선되어 있지 않으므로(α-W additive), 이 placeholder 는 동작에 영향이 없다.
    pub fn band(self) -> Level {
        match self.0 {
            0..=49 => Level::Normal,
            50..=74 => Level::Warning,
            75..=89 => Level::Critical,
            _ => Level::Emergency,
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
