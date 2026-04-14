//! Simulator: harness tick 루프, PolicyStrategy 호출, trajectory 기록.
//!
//! tick 순서 (Phase 3 리포트 기준):
//!   1. drain_until(now + tick)
//!   2. 이벤트 처리 (signal/heartbeat/ObservationDue/Injection)
//!   3. check_qcf_timeout
//!   4. physics::step()
//!   5. trajectory.record_state_snapshot()
//!   6. clock.advance(tick)

#![allow(dead_code)]

use std::time::Duration;

use llm_shared::{EngineCommand, EngineDirective};

use super::clock::{EventKind, VirtualClock};
use super::config::ScenarioConfig;
use super::expr::ExprContext;
use super::noise::NoiseRng;
use super::physics;
use super::signal;
use super::state::{EngineStateModel, PhysicalState};
use super::trajectory::Trajectory;
use llm_manager::pipeline::PolicyStrategy;

/// action 발동 후 관측 지연 (lua_policy.rs OBSERVATION_DELAY_SECS 복제).
/// lua_policy는 cfg(feature="lua") 조건부이므로, 여기서 상수를 독립 보유한다.
/// 값 변경 시 lua_policy.rs와 함께 갱신할 것.
const OBSERVATION_DELAY: Duration = Duration::from_secs(3);

// ─────────────────────────────────────────────────────────
// SimError
// ─────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum SimError {
    #[error("expression eval failed: {0}")]
    ExprEval(String),

    #[error("policy returned error during {phase}: {err}")]
    Policy { phase: String, err: String },

    #[error("physics step failed: {0}")]
    Physics(String),

    #[error("observation schedule error: {0}")]
    Observation(String),

    #[error("max duration exceeded in run_until")]
    MaxDurationExceeded,
}

impl From<physics::SimError> for SimError {
    fn from(e: physics::SimError) -> Self {
        SimError::Physics(e.to_string())
    }
}

// ─────────────────────────────────────────────────────────
// Simulator
// ─────────────────────────────────────────────────────────

/// 완전한 시뮬레이션 루프.
///
/// 생성 후 `run_for` 또는 `run_until`로 실행하고, `trajectory()`로 결과를 검사한다.
pub struct Simulator {
    pub clock: VirtualClock,
    pub state: PhysicalState,
    pub engine: EngineStateModel,
    pub cfg: ScenarioConfig,
    pub policy: Box<dyn PolicyStrategy>,
    pub trajectory: Trajectory,
    pub rng: Option<NoiseRng>,
    tick_dt: Duration,
    ctx: ExprContext,
}

impl Simulator {
    /// 표준 엔트리: config 로드 + policy 주입.
    pub fn new(cfg: ScenarioConfig, policy: Box<dyn PolicyStrategy>) -> Self {
        let state = PhysicalState::from_config(&cfg.initial_state);
        let engine = EngineStateModel::from_config(&cfg.initial_state);
        let rng = cfg.rng_seed.map(NoiseRng::new);
        let ctx = ExprContext::new();
        Self {
            clock: VirtualClock::new(),
            state,
            engine,
            cfg,
            policy,
            trajectory: Trajectory::new(),
            rng,
            tick_dt: Duration::from_millis(50),
            ctx,
        }
    }

    /// tick 크기 변경 (기본 50ms).
    pub fn with_tick_dt(mut self, dt: Duration) -> Self {
        self.tick_dt = dt;
        self
    }

    /// Trajectory 참조.
    pub fn trajectory(&self) -> &Trajectory {
        &self.trajectory
    }

    // ─────────────────────────────────────────────────────
    // 주기 이벤트 프리로드
    // ─────────────────────────────────────────────────────

    /// `until` 시각까지 주기 이벤트를 프리로드한다.
    fn preload_events(&mut self, until: Duration) {
        let now = self.clock.now();

        // 이미 스케줄된 이벤트의 최대 시각 계산 (중복 스케줄 방지)
        // 현재는 단순히 now ~ until 구간을 채운다.
        // 더 정밀한 재시작 처리(lazy re-inject)는 Phase 5에서.

        let hb_interval = Duration::from_secs_f64(self.cfg.observation.heartbeat.interval_s);
        schedule_periodic_from(
            &mut self.clock,
            || EventKind::Heartbeat,
            now,
            hb_interval,
            until,
        );

        let mem_period =
            Duration::from_secs_f64(self.cfg.observation.signals.memory.poll_interval_s);
        schedule_periodic_from(
            &mut self.clock,
            || EventKind::SignalMemory,
            now,
            mem_period,
            until,
        );

        let cpu_period =
            Duration::from_secs_f64(self.cfg.observation.signals.compute.poll_interval_s);
        schedule_periodic_from(
            &mut self.clock,
            || EventKind::SignalCompute,
            now,
            cpu_period,
            until,
        );

        let therm_period =
            Duration::from_secs_f64(self.cfg.observation.signals.thermal.poll_interval_s);
        schedule_periodic_from(
            &mut self.clock,
            || EventKind::SignalThermal,
            now,
            therm_period,
            until,
        );

        let energy_period =
            Duration::from_secs_f64(self.cfg.observation.signals.energy.poll_interval_s);
        schedule_periodic_from(
            &mut self.clock,
            || EventKind::SignalEnergy,
            now,
            energy_period,
            until,
        );

        // external injections
        for (idx, inj) in self.cfg.external_injections.iter().enumerate() {
            let t_start = Duration::from_secs_f64(inj.t_start);
            let t_end = t_start + Duration::from_secs_f64(inj.duration);

            if t_start > now && t_start <= until {
                self.clock
                    .schedule(EventKind::ExternalInjectionStart(idx), t_start);
            }
            if t_end > now && t_end <= until {
                self.clock
                    .schedule(EventKind::ExternalInjectionEnd(idx), t_end);
            }
        }
    }

    // ─────────────────────────────────────────────────────
    // tick
    // ─────────────────────────────────────────────────────

    /// 1회 tick 실행.
    pub fn tick(&mut self) -> Result<(), SimError> {
        let tick_end = self.clock.now() + self.tick_dt;
        let events = self.clock.drain_until(tick_end);

        for event in &events {
            let at = event.at;
            match &event.kind {
                EventKind::Heartbeat => {
                    let msg = signal::derive_heartbeat(
                        &self.state,
                        &self.engine,
                        &self.cfg,
                        &mut self.rng,
                    );
                    self.policy.update_engine_state(&msg);
                    self.trajectory.record_heartbeat(at, &msg);
                }

                EventKind::SignalMemory
                | EventKind::SignalCompute
                | EventKind::SignalThermal
                | EventKind::SignalEnergy => {
                    if let Some(sig) =
                        signal::derive_signal(&event.kind, &self.state, &self.cfg, &mut self.rng)
                    {
                        if let Some(dir) = self.policy.process_signal(&sig) {
                            self.apply_directive(&dir)?;
                            self.trajectory.record_directive(at, &sig, &dir);
                        }
                        self.trajectory.record_signal(at, &sig);
                    }
                }

                EventKind::ObservationDue {
                    action,
                    recorded_at,
                } => {
                    self.trajectory
                        .record_observation_due(at, action, *recorded_at);
                }

                EventKind::ExternalInjectionStart(idx) => {
                    self.trajectory.record_injection_event(at, *idx, true);
                }

                EventKind::ExternalInjectionEnd(idx) => {
                    self.trajectory.record_injection_event(at, *idx, false);
                }

                EventKind::Custom(name) => {
                    self.trajectory.record_custom(at, name);
                }
            }
        }

        // check_qcf_timeout — 매 tick 호출
        if let Some(dir) = self.policy.check_qcf_timeout() {
            // qcf timeout directive는 trigger signal 없이 발동된다.
            // trajectory에 Custom 이벤트로 기록하고 apply한다.
            self.apply_directive(&dir)?;
            self.trajectory
                .record_custom(tick_end, "qcf_timeout_directive");
        }

        // physics step (clock.advance 이전에 호출)
        physics::step(
            &mut self.state,
            &self.engine,
            &self.cfg,
            &self.clock,
            self.tick_dt,
            &mut self.ctx,
        )?;

        // state snapshot 기록
        self.trajectory
            .record_state_snapshot(tick_end, &self.state, &self.engine);

        // clock advance
        self.clock.advance(self.tick_dt);

        Ok(())
    }

    // ─────────────────────────────────────────────────────
    // run_for / run_until
    // ─────────────────────────────────────────────────────

    /// 지정 기간 실행한다.
    pub fn run_for(&mut self, duration: Duration) -> Result<(), SimError> {
        let end = self.clock.now() + duration;
        self.preload_events(end);

        while self.clock.now() < end {
            self.tick()?;
        }
        Ok(())
    }

    /// predicate가 true가 될 때까지 실행한다 (최대 max_duration).
    pub fn run_until<F: FnMut(&Simulator) -> bool>(
        &mut self,
        mut predicate: F,
        max: Duration,
    ) -> Result<(), SimError> {
        // run_until은 종료 시각을 모르므로 horizon을 60s로 설정한다.
        const HORIZON: Duration = Duration::from_secs(60);
        let until = self.clock.now() + max.min(HORIZON);
        self.preload_events(until);

        let end = self.clock.now() + max;
        loop {
            if predicate(self) {
                return Ok(());
            }
            if self.clock.now() >= end {
                return Err(SimError::MaxDurationExceeded);
            }
            self.tick()?;
        }
    }

    // ─────────────────────────────────────────────────────
    // apply_directive (내부)
    // ─────────────────────────────────────────────────────

    fn apply_directive(&mut self, dir: &EngineDirective) -> Result<(), SimError> {
        // EngineStateModel에 의도 상태 반영
        self.engine.apply_directive(dir, &mut self.state);

        // observable action이면 ObservationDue 이벤트 스케줄
        for cmd in &dir.commands {
            if let Some(action_name) = observable_action_name(cmd) {
                let recorded_at = self.clock.now();
                let due_at = recorded_at + OBSERVATION_DELAY;
                self.clock.schedule(
                    EventKind::ObservationDue {
                        action: action_name,
                        recorded_at,
                    },
                    due_at,
                );
            }
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────
// 헬퍼 함수
// ─────────────────────────────────────────────────────────

/// `now` 이후 처음으로 오는 `period` 배수 시각부터 `until`까지
/// 이벤트를 프리로드한다.
fn schedule_periodic_from(
    clock: &mut VirtualClock,
    kind_fn: impl Fn() -> EventKind,
    now: Duration,
    period: Duration,
    until: Duration,
) {
    if period.is_zero() || period.as_nanos() == 0 {
        return;
    }
    // now 이후 첫 번째 발화 시각: now를 period로 나눈 다음 배수
    let now_nanos = now.as_nanos();
    let period_nanos = period.as_nanos();
    let first_nanos = ((now_nanos / period_nanos) + 1) * period_nanos;
    let first = Duration::from_nanos(first_nanos as u64);

    clock.schedule_periodic(kind_fn, first, period, until);
}

/// EngineCommand가 관측 가능한 action이면 canonical 이름을 반환한다.
fn observable_action_name(cmd: &EngineCommand) -> Option<String> {
    match cmd {
        EngineCommand::KvEvictH2o { .. } => Some("kv_evict_h2o".to_string()),
        EngineCommand::KvEvictSliding { .. } => Some("kv_evict_sliding".to_string()),
        EngineCommand::KvMergeD2o { .. } => Some("kv_evict_d2o".to_string()),
        EngineCommand::Throttle { .. } => Some("throttle".to_string()),
        EngineCommand::SwitchHw { .. } => Some("switch_hw".to_string()),
        EngineCommand::SetPartitionRatio { .. } => Some("set_partition_ratio".to_string()),
        EngineCommand::LayerSkip { .. } => Some("layer_skip".to_string()),
        // RestoreDefaults, RequestQcf, SetTargetTbt 등은 관측 불필요
        _ => None,
    }
}
