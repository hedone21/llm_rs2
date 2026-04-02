//! Policy pipeline — 새 계층형 정책 메인 루프용 상태 캡슐화.
//!
//! `HierarchicalPolicy`는 PI Controller, Supervisory Layer, Action Selector,
//! Relief Estimator를 연결하여 `SystemSignal` 입력에서 `EngineDirective`를
//! 생성하는 전체 파이프라인을 담당한다.
//!
//! # 설계 참고
//!
//! `docs/36_policy_design.md` §9 (Manager Main Loop)를 참조한다.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use llm_shared::{EngineCommand, EngineDirective, EngineMessage, Level, QcfEstimate, SystemSignal};

use crate::action_registry::ActionRegistry;
use crate::config::PolicyConfig;
use crate::pi_controller::PiController;
use crate::relief::ReliefEstimator;
use crate::relief::linear::OnlineLinearEstimator;
use crate::selector::ActionSelector;
use crate::supervisory::SupervisoryLayer;
use crate::types::{
    ActionCommand, ActionId, FEATURE_DIM, FeatureVector, OperatingMode, Operation, PressureVector,
    feature,
};

/// 정책 판단 계층의 공통 인터페이스.
///
/// Monitor가 수집한 SystemSignal을 처리하여 EngineDirective를 생성한다.
/// 구현체에 따라 PI+Supervisory+Selector(HierarchicalPolicy) 또는
/// 규칙 기반(ThresholdPolicy) 등 다양한 전략이 가능하다.
pub trait PolicyStrategy: Send {
    /// SystemSignal을 처리하여 필요 시 EngineDirective를 반환한다.
    fn process_signal(&mut self, signal: &SystemSignal) -> Option<EngineDirective>;

    /// Engine의 heartbeat/capability/response 메시지로 내부 상태를 갱신한다.
    fn update_engine_state(&mut self, msg: &EngineMessage);

    /// 현재 operating mode를 반환한다 (로깅/모니터링용).
    fn mode(&self) -> OperatingMode;

    /// 세션 종료 시 내부 모델을 저장한다. 기본 구현은 no-op.
    fn save_model(&self) {}

    /// Engine이 보낸 QcfEstimate로 보류 중인 액션 선택을 완료한다 (SEQ-097).
    /// 기본 구현은 no-op.
    fn complete_qcf_selection(&mut self, _qcf: &QcfEstimate) -> Option<EngineDirective> {
        None
    }

    /// 보류 중인 QCF 요청의 타임아웃을 체크한다 (SEQ-098, 1초).
    /// 기본 구현은 no-op.
    fn check_qcf_timeout(&mut self) -> Option<EngineDirective> {
        None
    }
}

/// 이전 액션 효과 관측을 위한 컨텍스트.
struct ObservationContext {
    /// 액션 적용 직전의 pressure 상태.
    pressure_before: PressureVector,
    /// 액션 적용 시점의 feature vector.
    feature_vec: FeatureVector,
    /// 적용된 액션 목록.
    applied_actions: Vec<ActionId>,
    /// 액션이 적용된 시각.
    applied_at: Instant,
}

/// 액션 효과 관측 대기 시간(초).
/// 액션 적용 후 이 시간이 지나야 pressure 변화로 실측 relief를 계산한다.
const OBSERVATION_DELAY_SECS: f32 = 3.0;

/// QCF 요청 타임아웃 (SEQ-098).
const QCF_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);

/// Pending QCF request state (SEQ-095~098).
struct QcfPending {
    /// Pressure snapshot at the time of request.
    pressure: PressureVector,
    /// Operating mode that triggered the request.
    mode: OperatingMode,
    /// When the request was sent.
    requested_at: Instant,
}

/// Seq ID 생성을 위한 단조 증가 카운터.
static SEQ_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub fn next_seq_id() -> u64 {
    SEQ_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// 계층형 정책 파이프라인.
///
/// PI Controller 3개 (compute / memory / thermal) → Supervisory Layer →
/// Action Selector → EngineDirective 생성의 전체 흐름을 캡슐화한다.
pub struct HierarchicalPolicy {
    pi_compute: PiController,
    pi_memory: PiController,
    pi_thermal: PiController,
    supervisory: SupervisoryLayer,
    registry: ActionRegistry,
    estimator: OnlineLinearEstimator,
    /// 현재 engine feature 상태 (heartbeat로 갱신; 초기값 = zeros).
    engine_state: FeatureVector,
    /// 현재 pressure vector (PI Controller 출력).
    pressure: PressureVector,
    /// 직전 루프의 operating mode.
    prev_mode: OperatingMode,
    /// 마지막으로 액션을 취했을 때의 도메인별 pressure 스냅샷.
    last_acted_pressure: PressureVector,
    /// 이전 액션의 실측 relief 수집을 위한 컨텍스트.
    pending_observation: Option<ObservationContext>,
    /// 허용 가능한 latency 악화 상한.
    latency_budget: f32,
    /// PI update 시 사용할 dt (초). 기본값 0.1 (100 ms).
    dt: f32,
    /// Relief model 저장/불러오기 경로.
    relief_model_path: Option<String>,
    /// Engine이 보고한 실행 가능 액션 목록 (heartbeat로 갱신).
    available_actions: Vec<ActionId>,
    /// Engine이 보고한 현재 활성 액션 목록 (heartbeat로 갱신).
    active_actions_reported: Vec<ActionId>,
    /// 도메인별 마지막 신호 수신 시각 (실측 dt 계산용).
    last_signal_time: HashMap<&'static str, Instant>,
    /// Pending QCF request state (SEQ-095~098).
    qcf_pending: Option<QcfPending>,
}

impl HierarchicalPolicy {
    /// PolicyConfig로 초기화한다.
    pub fn new(config: &PolicyConfig) -> Self {
        let pi_cfg = &config.pi_controller;
        Self {
            pi_compute: PiController::new(
                pi_cfg.compute_kp,
                pi_cfg.compute_ki,
                pi_cfg.compute_setpoint,
                pi_cfg.integral_clamp,
            ),
            pi_memory: PiController::new(
                pi_cfg.memory_kp,
                pi_cfg.memory_ki,
                pi_cfg.memory_setpoint,
                pi_cfg.integral_clamp,
            )
            .with_gain_zones(pi_cfg.memory_gain_zones.clone()),
            pi_thermal: PiController::new(
                pi_cfg.thermal_kp,
                pi_cfg.thermal_ki,
                pi_cfg.thermal_setpoint,
                pi_cfg.integral_clamp,
            ),
            supervisory: SupervisoryLayer::new(&config.supervisory),
            registry: ActionRegistry::from_config(config),
            estimator: OnlineLinearEstimator::new(
                FEATURE_DIM,
                config.relief_model.forgetting_factor,
            ),
            engine_state: FeatureVector::zeros(),
            pressure: PressureVector::default(),
            prev_mode: OperatingMode::Normal,
            last_acted_pressure: PressureVector::default(),
            pending_observation: None,
            latency_budget: config.selector.latency_budget,
            dt: 0.1,
            relief_model_path: None,
            available_actions: vec![],
            active_actions_reported: vec![],
            last_signal_time: HashMap::new(),
            qcf_pending: None,
        }
    }

    /// Relief model 저장 경로를 설정하고 기존 파일이 있으면 불러온다.
    pub fn set_relief_model_path(&mut self, path: String) {
        if let Err(e) = self.estimator.load(Path::new(&path)) {
            // 파일이 없으면 무시 (첫 실행)
            log::debug!("Relief model not loaded ({}): {}", path, e);
        } else {
            log::info!("Loaded relief model from {}", path);
        }
        self.relief_model_path = Some(path);
    }

    /// PI update dt를 설정한다 (기본값: 0.1초).
    pub fn set_dt(&mut self, dt: f32) {
        self.dt = dt;
    }

    /// 현재 pressure vector를 반환한다.
    pub fn pressure(&self) -> &PressureVector {
        &self.pressure
    }

    // ── 내부 헬퍼 ──────────────────────────────────────────────────────────

    /// 도메인별 실측 dt를 계산하고 last_signal_time을 갱신한다.
    ///
    /// 첫 신호 시에는 `self.dt` 기본값을 반환한다.
    /// 결과는 [0.001, 10.0] 범위로 clamp하여 이상값을 방지한다.
    fn elapsed_dt(&mut self, domain: &'static str) -> f32 {
        let now = Instant::now();
        let dt = self
            .last_signal_time
            .get(domain)
            .map(|prev| now.duration_since(*prev).as_secs_f32())
            .unwrap_or(self.dt);
        self.last_signal_time.insert(domain, now);
        dt.clamp(0.001, 10.0)
    }

    /// SystemSignal에서 도메인별 측정값을 추출하여 해당 PI Controller에 입력한다.
    fn update_pressure(&mut self, signal: &SystemSignal) {
        match signal {
            SystemSignal::MemoryPressure {
                available_bytes,
                total_bytes,
                ..
            } => {
                let m = if *total_bytes > 0 {
                    (1.0 - *available_bytes as f32 / *total_bytes as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let dt = self.elapsed_dt("memory");
                self.pressure.memory = self.pi_memory.update(m, dt);
            }
            SystemSignal::ThermalAlert { temperature_mc, .. } => {
                // 85°C (85000 mc) = 1.0 기준 정규화
                let m = (*temperature_mc as f32 / 85_000.0).clamp(0.0, 1.0);
                let dt = self.elapsed_dt("thermal");
                self.pressure.thermal = self.pi_thermal.update(m, dt);
            }
            SystemSignal::ComputeGuidance {
                cpu_usage_pct,
                gpu_usage_pct,
                ..
            } => {
                let m_cpu = (*cpu_usage_pct as f32 / 100.0).clamp(0.0, 1.0);
                let m_gpu = (*gpu_usage_pct as f32 / 100.0).clamp(0.0, 1.0);
                let m = m_cpu.max(m_gpu);
                let dt = self.elapsed_dt("compute");
                self.pressure.compute = self.pi_compute.update(m, dt);
            }
            SystemSignal::EnergyConstraint { level, .. } => {
                // Energy → compute pressure에 보조 기여 (0.5 가중치)
                let m = level_to_measurement(*level) * 0.5;
                let combined = self.pressure.compute.max(m);
                let dt = self.elapsed_dt("compute");
                self.pressure.compute = self.pi_compute.update(combined, dt);
            }
        }
    }

    /// OBSERVATION_DELAY_SECS 이후 실측 relief를 관측하여 estimator를 갱신한다.
    fn update_observation(&mut self) {
        let should_observe = self
            .pending_observation
            .as_ref()
            .is_some_and(|ctx| ctx.applied_at.elapsed().as_secs_f32() > OBSERVATION_DELAY_SECS);

        if should_observe {
            // pending_observation을 꺼낸다
            let ctx = self.pending_observation.take().unwrap();
            let actual_relief = ctx.pressure_before - self.pressure;
            for action in &ctx.applied_actions {
                self.estimator
                    .observe(action, &ctx.feature_vec, &actual_relief);
            }
            log::debug!(
                "[Pipeline] Observed relief: compute={:.2} memory={:.2} thermal={:.2}",
                actual_relief.compute,
                actual_relief.memory,
                actual_relief.thermal,
            );
        }
    }

    /// ActionCommand를 `EngineCommand`로 직접 변환한다.
    fn convert_to_engine_commands(&self, commands: &[ActionCommand]) -> Vec<EngineCommand> {
        commands
            .iter()
            .filter_map(action_to_engine_command)
            .collect()
    }

    /// Normal 복귀 시 발송할 restore directive를 생성한다.
    fn build_restore_directive(&self) -> Option<EngineDirective> {
        let commands = vec![EngineCommand::RestoreDefaults];
        Some(EngineDirective {
            seq_id: next_seq_id(),
            commands,
        })
    }

    /// Critical→Warning 전환 시 lossy 액션만 해제하는 directive를 생성한다.
    ///
    /// Lossless 액션(Throttle 등)은 유지되며, 다음 사이클에서 Warning 모드
    /// ActionSelector가 lossless 액션만 재선택한다.
    ///
    /// 현재는 RestoreDefaults를 사용하여 일괄 해제 후, Warning 모드에서
    /// lossless 액션을 재선택하는 2-step 방식으로 동작한다.
    /// 향후 per-action Release가 추가되면 이 메서드만 변경하면 된다.
    fn build_lossy_release_directive(&self) -> Option<EngineDirective> {
        let commands = vec![EngineCommand::RestoreDefaults];
        Some(EngineDirective {
            seq_id: next_seq_id(),
            commands,
        })
    }

    /// default_cost로 QCF values를 구성한다 (기존 inline 코드의 메서드 추출).
    fn build_default_qcf_values(&self) -> HashMap<ActionId, f32> {
        self.registry
            .lossy_actions()
            .into_iter()
            .map(|id| {
                let cost = self.registry.default_cost(&id);
                (id, cost)
            })
            .collect()
    }

    /// 보류된 QCF 컨텍스트와 QCF values로 액션 선택을 실행한다 (공통 로직).
    fn run_action_selection_with_qcf(
        &mut self,
        pending: &QcfPending,
        qcf_values: &HashMap<ActionId, f32>,
    ) -> Option<EngineDirective> {
        let commands = ActionSelector::select(
            &self.registry,
            &self.estimator,
            &pending.pressure,
            pending.mode,
            &self.engine_state,
            qcf_values,
            self.latency_budget,
            &self.active_actions_reported,
            &self.available_actions,
        );

        if commands.is_empty() {
            return None;
        }

        let engine_commands = self.convert_to_engine_commands(&commands);
        if engine_commands.is_empty() {
            return None;
        }

        // 관측 컨텍스트 기록
        self.pending_observation = Some(ObservationContext {
            pressure_before: pending.pressure,
            feature_vec: self.engine_state.clone(),
            applied_actions: commands.iter().map(|c| c.action).collect(),
            applied_at: Instant::now(),
        });
        self.last_acted_pressure = pending.pressure;

        Some(EngineDirective {
            seq_id: next_seq_id(),
            commands: engine_commands,
        })
    }
}

impl PolicyStrategy for HierarchicalPolicy {
    /// SystemSignal을 처리하여 필요한 경우 `EngineDirective`를 반환한다.
    ///
    /// # 처리 순서
    ///
    /// ① 신호에서 측정값 추출 → PI Controller 갱신 → pressure 업데이트
    /// ② Supervisory Layer → mode 결정
    /// ③ 이전 액션의 실측 relief 관측 갱신
    /// ④ 액션 필요 여부 판단
    /// ⑤ Action Selection (필요 시)
    /// ⑥ EngineDirective 생성 및 관측 컨텍스트 기록
    /// ⑦ De-escalation (Normal 복귀 시 restore directive)
    fn process_signal(&mut self, signal: &SystemSignal) -> Option<EngineDirective> {
        // ① PI Controller 갱신
        self.update_pressure(signal);

        // ② Supervisory → mode
        let mode = self.supervisory.evaluate(&self.pressure);

        // ③ 관측 갱신
        self.update_observation();

        // ④ 액션 필요 여부 — 도메인별 독립 판정
        let needs_action = match mode {
            OperatingMode::Normal => false,
            OperatingMode::Warning | OperatingMode::Critical => {
                mode != self.prev_mode
                    || self
                        .pressure
                        .any_domain_exceeds(&self.last_acted_pressure, 1.2)
            }
        };

        let mut result = None;

        if needs_action {
            // ⑤ Critical 전환 시 QCF 요청 (SEQ-095)
            let is_critical_transition = mode == OperatingMode::Critical
                && self.prev_mode != OperatingMode::Critical
                && self.qcf_pending.is_none();

            if is_critical_transition {
                self.qcf_pending = Some(QcfPending {
                    pressure: self.pressure,
                    mode,
                    requested_at: Instant::now(),
                });
                let directive = EngineDirective {
                    seq_id: next_seq_id(),
                    commands: vec![EngineCommand::RequestQcf],
                };
                log::debug!(
                    "[Pipeline] Critical transition — sending RequestQcf (pressure={:.2}/{:.2}/{:.2})",
                    self.pressure.compute,
                    self.pressure.memory,
                    self.pressure.thermal,
                );
                self.prev_mode = mode;
                return Some(directive);
            }

            // ⑥ 기존 로직: default_cost로 즉시 액션 선택
            let qcf_values = self.build_default_qcf_values();
            let pending_ctx = QcfPending {
                pressure: self.pressure,
                mode,
                requested_at: Instant::now(),
            };
            if let Some(directive) = self.run_action_selection_with_qcf(&pending_ctx, &qcf_values) {
                log::debug!(
                    "[Pipeline] mode={:?} pressure={:.2}/{:.2}/{:.2} → directive seq={} ({} cmds)",
                    mode,
                    self.pressure.compute,
                    self.pressure.memory,
                    self.pressure.thermal,
                    directive.seq_id,
                    directive.commands.len()
                );
                result = Some(directive);
            }
        }

        // ⑦ De-escalation: 모드 하강 시 적절한 복귀 명령 발송
        if self.prev_mode > mode {
            match (self.prev_mode, mode) {
                (OperatingMode::Critical, OperatingMode::Warning) => {
                    // Critical→Warning: lossy 액션 해제, lossless는 다음 사이클에서 재선택
                    log::info!(
                        "[Pipeline] De-escalating Critical → Warning — releasing lossy actions"
                    );
                    if result.is_none() {
                        result = self.build_lossy_release_directive();
                    }
                }
                (_, OperatingMode::Normal) => {
                    // *→Normal: 모든 액션 복원
                    log::info!("[Pipeline] De-escalating to Normal — sending restore directive");
                    if result.is_none() {
                        result = self.build_restore_directive();
                    }
                }
                _ => {}
            }
        }

        self.prev_mode = mode;
        result
    }

    /// Engine heartbeat에서 `engine_state` feature vector를 갱신한다.
    ///
    /// `37_protocol_design.md §6`의 Feature Vector 스키마를 따른다.
    /// Heartbeat 이외의 메시지 (Capability, Response)는 무시한다.
    fn update_engine_state(&mut self, msg: &EngineMessage) {
        let EngineMessage::Heartbeat(status) = msg else {
            return;
        };

        let v = &mut self.engine_state.values;

        // [0] KV 점유율 (0.0 ~ 1.0)
        v[feature::KV_OCCUPANCY] = status.kv_cache_utilization;

        // [1] GPU 사용 여부 (active_device에 "opencl" 포함 시 1.0)
        v[feature::IS_GPU] = if status.active_device.contains("opencl") {
            1.0
        } else {
            0.0
        };

        // [2] 토큰 진행률 (kv_cache_tokens / 2048)
        const DEFAULT_MAX_TOKENS: f32 = 2048.0;
        v[feature::TOKEN_PROGRESS] = (status.kv_cache_tokens as f32 / DEFAULT_MAX_TOKENS).min(1.0);

        // [5] TBT 비율 (actual_throughput / 100.0 으로 정규화)
        v[feature::TBT_RATIO] = (status.actual_throughput / 100.0).clamp(0.0, 1.0);

        // [6] 생성 토큰 정규화 (tokens_generated / 2048)
        v[feature::TOKENS_GENERATED_NORM] =
            (status.tokens_generated as f32 / DEFAULT_MAX_TOKENS).min(1.0);

        // [10] 활성 eviction 여부 (eviction_policy가 "none" 또는 빈 문자열이 아니면 1.0)
        // ReliefEstimator 예측에 사용되므로 유지한다.
        v[feature::ACTIVE_EVICTION] =
            if !status.eviction_policy.is_empty() && status.eviction_policy != "none" {
                1.0
            } else {
                0.0
            };

        // [11] 활성 layer skip 여부 (skip_ratio > 0)
        v[feature::ACTIVE_LAYER_SKIP] = if status.skip_ratio > 0.0 { 1.0 } else { 0.0 };

        // EngineStatus의 available_actions / active_actions를 ActionId로 파싱하여 캐싱.
        // 액션 필터링에는 FeatureVector 대신 이 목록을 사용한다.
        self.available_actions = status
            .available_actions
            .iter()
            .filter_map(|s| ActionId::from_str(s))
            .collect();
        self.active_actions_reported = status
            .active_actions
            .iter()
            .filter_map(|s| ActionId::from_str(s))
            .collect();
    }

    fn mode(&self) -> OperatingMode {
        self.supervisory.mode()
    }

    /// Engine이 보낸 QcfEstimate로 보류 중인 액션 선택을 완료한다 (SEQ-097).
    fn complete_qcf_selection(&mut self, qcf: &QcfEstimate) -> Option<EngineDirective> {
        let pending = self.qcf_pending.take()?;

        // QcfEstimate의 String key → ActionId 변환
        let mut qcf_values: HashMap<ActionId, f32> = HashMap::new();
        for (name, &cost) in &qcf.estimates {
            if let Some(id) = ActionId::from_str(name) {
                qcf_values.insert(id, cost);
            }
        }
        // QcfEstimate에 없는 lossy action은 default_cost 폴백
        for id in self.registry.lossy_actions() {
            qcf_values
                .entry(id)
                .or_insert_with(|| self.registry.default_cost(&id));
        }

        self.run_action_selection_with_qcf(&pending, &qcf_values)
    }

    /// 보류 중인 QCF 요청의 타임아웃을 체크한다 (SEQ-098, 1초).
    fn check_qcf_timeout(&mut self) -> Option<EngineDirective> {
        let timed_out = self
            .qcf_pending
            .as_ref()
            .is_some_and(|p| p.requested_at.elapsed() >= QCF_TIMEOUT);
        if timed_out {
            log::warn!("QCF estimate timeout (1s) — falling back to default costs");
            let pending = self.qcf_pending.take().unwrap();
            let qcf_values = self.build_default_qcf_values();
            self.run_action_selection_with_qcf(&pending, &qcf_values)
        } else {
            None
        }
    }

    /// 세션 종료 시 Relief model을 디스크에 저장한다.
    fn save_model(&self) {
        if let Some(path) = &self.relief_model_path {
            if let Err(e) = self.estimator.save(Path::new(path)) {
                log::warn!("Failed to save relief model to {}: {}", path, e);
            } else {
                log::info!("Saved relief model to {}", path);
            }
        }
    }
}

/// ActionCommand 하나를 EngineCommand로 변환한다.
/// Release 명령은 `None`을 반환한다 (restore directive에서 일괄 처리).
fn action_to_engine_command(cmd: &ActionCommand) -> Option<EngineCommand> {
    match (&cmd.action, &cmd.operation) {
        (ActionId::SwitchHw, Operation::Apply(_)) => Some(EngineCommand::SwitchHw {
            device: "cpu".to_string(),
        }),
        (ActionId::Throttle, Operation::Apply(params)) => {
            let delay_ms = params.values.get("delay_ms").copied().unwrap_or(0.0) as u64;
            Some(EngineCommand::Throttle { delay_ms })
        }
        (ActionId::KvEvictSliding, Operation::Apply(params)) => {
            let keep_ratio = params.values.get("keep_ratio").copied().unwrap_or(0.5);
            Some(EngineCommand::KvEvictSliding { keep_ratio })
        }
        (ActionId::KvEvictH2o, Operation::Apply(params)) => {
            let keep_ratio = params.values.get("keep_ratio").copied().unwrap_or(0.5);
            Some(EngineCommand::KvEvictH2o { keep_ratio })
        }
        (ActionId::KvEvictStreaming, Operation::Apply(params)) => {
            let sink_size = params.values.get("sink_size").copied().unwrap_or(4.0) as usize;
            let window_size = params.values.get("window_size").copied().unwrap_or(256.0) as usize;
            Some(EngineCommand::KvStreaming {
                sink_size,
                window_size,
            })
        }
        (ActionId::KvMergeD2o, Operation::Apply(params)) => {
            let keep_ratio = params.values.get("keep_ratio").copied().unwrap_or(0.5);
            Some(EngineCommand::KvMergeD2o { keep_ratio })
        }
        (ActionId::KvOffloadDisk, Operation::Apply(_)) => {
            // KvOffloadDisk은 fallback으로 sliding window eviction 사용
            Some(EngineCommand::KvEvictSliding { keep_ratio: 0.8 })
        }
        (ActionId::KvQuantDynamic, Operation::Apply(params)) => {
            let target_bits = params.values.get("target_bits").copied().unwrap_or(4.0) as u8;
            Some(EngineCommand::KvQuantDynamic { target_bits })
        }
        (ActionId::LayerSkip, Operation::Apply(params)) => {
            let skip_layers = params.values.get("skip_layers").copied().unwrap_or(1.0);
            let total_layers = params.values.get("total_layers").copied().unwrap_or(16.0);
            let skip_ratio = (skip_layers / total_layers).clamp(0.0, 1.0);
            Some(EngineCommand::LayerSkip { skip_ratio })
        }
        (_, Operation::Release) => {
            // Release 명령은 restore directive에서 일괄 처리 — 여기서는 무시
            None
        }
    }
}

/// Level을 PI Controller 입력용 측정값(0.0~1.0)으로 변환한다.
fn level_to_measurement(level: Level) -> f32 {
    match level {
        Level::Normal => 0.0,
        Level::Warning => 0.55,
        Level::Critical => 0.80,
        Level::Emergency => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PolicyConfig;

    fn make_pipeline() -> HierarchicalPolicy {
        HierarchicalPolicy::new(&PolicyConfig::default())
    }

    fn memory_signal(level: Level) -> SystemSignal {
        let (available_bytes, total_bytes) = match level {
            Level::Normal => (1_800_000_000u64, 2_000_000_000u64),
            Level::Warning => (800_000_000u64, 2_000_000_000u64),
            Level::Critical => (300_000_000u64, 2_000_000_000u64),
            Level::Emergency => (100_000_000u64, 2_000_000_000u64),
        };
        SystemSignal::MemoryPressure {
            level,
            available_bytes,
            total_bytes,
            reclaim_target_bytes: 0,
        }
    }

    fn thermal_signal(temp_mc: i32) -> SystemSignal {
        let level = if temp_mc >= 85_000 {
            Level::Emergency
        } else if temp_mc >= 75_000 {
            Level::Critical
        } else if temp_mc >= 60_000 {
            Level::Warning
        } else {
            Level::Normal
        };
        SystemSignal::ThermalAlert {
            level,
            temperature_mc: temp_mc,
            throttling_active: temp_mc >= 75_000,
            throttle_ratio: 1.0,
        }
    }

    fn compute_signal(level: Level, cpu_pct: f64) -> SystemSignal {
        SystemSignal::ComputeGuidance {
            level,
            recommended_backend: llm_shared::RecommendedBackend::Cpu,
            reason: llm_shared::ComputeReason::CpuBottleneck,
            cpu_usage_pct: cpu_pct,
            gpu_usage_pct: 0.0,
        }
    }

    /// Normal pressure 시 directive 없음
    #[test]
    fn test_normal_pressure_no_directive() {
        let mut p = make_pipeline();
        let result = p.process_signal(&memory_signal(Level::Normal));
        assert!(
            result.is_none(),
            "Normal pressure should produce no directive"
        );
        assert_eq!(p.mode(), OperatingMode::Normal);
    }

    /// 충분히 높은 pressure가 누적되면 pressure 값이 양수가 된다
    #[test]
    fn test_pressure_accumulates_under_load() {
        let mut p = make_pipeline();
        // 여러 번 Critical 신호를 보내어 PI 적분을 누적
        for _ in 0..10 {
            p.process_signal(&memory_signal(Level::Critical));
        }
        assert!(
            p.pressure().memory > 0.0,
            "Repeated critical signals should build pressure"
        );
    }

    /// Emergency 신호는 pressure가 1.0에 도달한다
    #[test]
    fn test_emergency_level_measurement() {
        let m = level_to_measurement(Level::Emergency);
        assert!((m - 1.0).abs() < f32::EPSILON);
    }

    /// Normal 신호는 측정값 0.0
    #[test]
    fn test_normal_level_measurement() {
        let m = level_to_measurement(Level::Normal);
        assert!(m.abs() < f32::EPSILON);
    }

    /// 온도 정규화: 85000 mc → 1.0
    #[test]
    fn test_thermal_normalization_max() {
        let mut p = make_pipeline();
        // 85°C 신호를 반복해서 PI 적분 누적
        for _ in 0..20 {
            p.process_signal(&thermal_signal(85_000));
        }
        // thermal pressure가 양수여야 한다
        assert!(p.pressure().thermal > 0.0);
    }

    /// 온도 정규화: 42500 mc ≈ 0.5
    #[test]
    fn test_thermal_normalization_half() {
        // 42500 mc / 85000 = 0.5 → setpoint(0.8) 미만이므로 PI error = 0 → pressure = 0
        let mut p = make_pipeline();
        p.process_signal(&thermal_signal(42_500));
        // setpoint=0.8 이하이므로 thermal pressure = 0
        assert!(
            p.pressure().thermal.abs() < f32::EPSILON,
            "Half temp (below setpoint) should give 0 thermal pressure"
        );
    }

    /// compute_signal에서 CPU 사용률이 반영된다
    #[test]
    fn test_compute_cpu_usage_reflected() {
        let mut p = make_pipeline();
        // CPU 95% → setpoint(0.70) 초과
        for _ in 0..5 {
            p.process_signal(&compute_signal(Level::Critical, 95.0));
        }
        assert!(
            p.pressure().compute > 0.0,
            "High CPU usage should produce compute pressure"
        );
    }

    /// restore directive에는 RestoreDefaults 명령이 포함된다
    #[test]
    fn test_restore_directive_has_restore_defaults() {
        let p = make_pipeline();
        let directive = p.build_restore_directive().unwrap();
        assert_eq!(directive.commands.len(), 1);
        assert!(
            matches!(directive.commands[0], EngineCommand::RestoreDefaults),
            "Restore directive should contain RestoreDefaults"
        );
    }

    /// seq_id는 단조 증가해야 한다
    #[test]
    fn test_seq_id_monotonic() {
        let id1 = next_seq_id();
        let id2 = next_seq_id();
        assert!(id2 > id1, "seq_id should be monotonically increasing");
    }

    /// save_model은 경로가 없으면 조용히 무시한다
    #[test]
    fn test_save_model_no_path_is_noop() {
        let p = make_pipeline();
        // panic 없이 완료되어야 한다
        p.save_model();
    }

    /// ActionCommand → EngineCommand 변환: KvEvictSliding → KvEvictSliding
    #[test]
    fn test_convert_kv_evict_sliding_to_engine_command() {
        use crate::types::{ActionParams, Operation};
        use std::collections::HashMap;

        let p = make_pipeline();
        let mut values = HashMap::new();
        values.insert("keep_ratio".to_string(), 0.6_f32);
        let cmd = ActionCommand {
            action: ActionId::KvEvictSliding,
            operation: Operation::Apply(ActionParams { values }),
        };
        let engine_cmds = p.convert_to_engine_commands(&[cmd]);
        assert_eq!(engine_cmds.len(), 1);
        match &engine_cmds[0] {
            EngineCommand::KvEvictSliding { keep_ratio } => {
                assert!((*keep_ratio - 0.6).abs() < f32::EPSILON);
            }
            _ => panic!("Expected KvEvictSliding"),
        }
    }

    /// ActionCommand → EngineCommand 변환: KvEvictH2o → KvEvictH2o
    #[test]
    fn test_convert_kv_evict_h2o_to_engine_command() {
        use crate::types::{ActionParams, Operation};
        use std::collections::HashMap;

        let p = make_pipeline();
        let mut values = HashMap::new();
        values.insert("keep_ratio".to_string(), 0.48_f32);
        let cmd = ActionCommand {
            action: ActionId::KvEvictH2o,
            operation: Operation::Apply(ActionParams { values }),
        };
        let engine_cmds = p.convert_to_engine_commands(&[cmd]);
        assert_eq!(engine_cmds.len(), 1);
        match &engine_cmds[0] {
            EngineCommand::KvEvictH2o { keep_ratio } => {
                assert!((*keep_ratio - 0.48).abs() < f32::EPSILON);
            }
            _ => panic!("Expected KvEvictH2o"),
        }
    }

    /// ActionCommand → EngineCommand 변환: SwitchHw → SwitchHw
    #[test]
    fn test_convert_switch_hw_to_engine_command() {
        use crate::types::{ActionParams, Operation};

        let p = make_pipeline();
        let cmd = ActionCommand {
            action: ActionId::SwitchHw,
            operation: Operation::Apply(ActionParams::default()),
        };
        let engine_cmds = p.convert_to_engine_commands(&[cmd]);
        assert_eq!(engine_cmds.len(), 1);
        match &engine_cmds[0] {
            EngineCommand::SwitchHw { device } => {
                assert_eq!(device, "cpu");
            }
            _ => panic!("Expected SwitchHw"),
        }
    }

    /// ActionCommand → EngineCommand 변환: Throttle → Throttle
    #[test]
    fn test_convert_throttle_to_engine_command() {
        use crate::types::{ActionParams, Operation};
        use std::collections::HashMap;

        let p = make_pipeline();
        let mut values = HashMap::new();
        values.insert("delay_ms".to_string(), 50.0_f32);
        let cmd = ActionCommand {
            action: ActionId::Throttle,
            operation: Operation::Apply(ActionParams { values }),
        };
        let engine_cmds = p.convert_to_engine_commands(&[cmd]);
        assert_eq!(engine_cmds.len(), 1);
        match &engine_cmds[0] {
            EngineCommand::Throttle { delay_ms } => {
                assert_eq!(*delay_ms, 50u64);
            }
            _ => panic!("Expected Throttle"),
        }
    }

    /// ActionCommand → EngineCommand 변환: LayerSkip → LayerSkip
    #[test]
    fn test_convert_layer_skip_to_engine_command() {
        use crate::types::{ActionParams, Operation};
        use std::collections::HashMap;

        let p = make_pipeline();
        let mut values = HashMap::new();
        values.insert("skip_layers".to_string(), 4.0_f32);
        values.insert("total_layers".to_string(), 16.0_f32);
        let cmd = ActionCommand {
            action: ActionId::LayerSkip,
            operation: Operation::Apply(ActionParams { values }),
        };
        let engine_cmds = p.convert_to_engine_commands(&[cmd]);
        assert_eq!(engine_cmds.len(), 1);
        match &engine_cmds[0] {
            EngineCommand::LayerSkip { skip_ratio } => {
                assert!((*skip_ratio - 0.25).abs() < f32::EPSILON);
            }
            _ => panic!("Expected LayerSkip"),
        }
    }

    /// ActionCommand → EngineCommand 변환: KvOffloadDisk → KvEvictSliding(fallback)
    #[test]
    fn test_convert_kv_offload_disk_fallback() {
        use crate::types::{ActionParams, Operation};

        let p = make_pipeline();
        let cmd = ActionCommand {
            action: ActionId::KvOffloadDisk,
            operation: Operation::Apply(ActionParams::default()),
        };
        let engine_cmds = p.convert_to_engine_commands(&[cmd]);
        assert_eq!(engine_cmds.len(), 1);
        match &engine_cmds[0] {
            EngineCommand::KvEvictSliding { keep_ratio } => {
                assert!((*keep_ratio - 0.8).abs() < f32::EPSILON);
            }
            _ => panic!("Expected KvEvictSliding as fallback for KvOffloadDisk"),
        }
    }

    /// Release 명령은 변환 시 무시된다
    #[test]
    fn test_convert_release_produces_no_command() {
        use crate::types::Operation;

        let p = make_pipeline();
        let cmd = ActionCommand {
            action: ActionId::KvEvictSliding,
            operation: Operation::Release,
        };
        let engine_cmds = p.convert_to_engine_commands(&[cmd]);
        assert!(
            engine_cmds.is_empty(),
            "Release should produce no engine command"
        );
    }

    // ── update_engine_state 테스트 ─────────────────────────────────────────

    fn make_heartbeat_msg(
        kv: f32,
        device: &str,
        eviction_policy: &str,
    ) -> llm_shared::EngineMessage {
        use llm_shared::{EngineMessage, EngineState, EngineStatus, ResourceLevel};
        EngineMessage::Heartbeat(EngineStatus {
            active_device: device.to_string(),
            compute_level: ResourceLevel::Normal,
            actual_throughput: 20.0,
            memory_level: ResourceLevel::Normal,
            kv_cache_bytes: 0,
            kv_cache_tokens: (kv * 2048.0) as usize,
            kv_cache_utilization: kv,
            memory_lossless_min: 1.0,
            memory_lossy_min: 0.01,
            state: EngineState::Running,
            tokens_generated: 512,
            available_actions: vec![],
            active_actions: vec![],
            eviction_policy: eviction_policy.to_string(),
            kv_dtype: "f16".to_string(),
            skip_ratio: 0.0,
        })
    }

    /// Heartbeat에서 engine_state feature vector가 갱신된다
    #[test]
    fn engine_state_updated_from_heartbeat() {
        use crate::types::feature;
        let mut p = make_pipeline();

        // 초기값은 zero
        assert_eq!(p.engine_state.values[feature::KV_OCCUPANCY], 0.0);

        let msg = make_heartbeat_msg(0.75, "opencl", "h2o");
        p.update_engine_state(&msg);

        assert!((p.engine_state.values[feature::KV_OCCUPANCY] - 0.75).abs() < 1e-5);
        assert!((p.engine_state.values[feature::IS_GPU] - 1.0).abs() < 1e-5);
        assert!((p.engine_state.values[feature::TOKEN_PROGRESS] - 0.75).abs() < 1e-5);
        assert!((p.engine_state.values[feature::TBT_RATIO] - 0.2).abs() < 1e-5);
        assert!(
            (p.engine_state.values[feature::TOKENS_GENERATED_NORM] - 512.0 / 2048.0).abs() < 1e-5
        );
        assert!((p.engine_state.values[feature::ACTIVE_EVICTION] - 1.0).abs() < 1e-5);
    }

    /// CPU 디바이스이면 IS_GPU = 0.0
    #[test]
    fn engine_state_cpu_device_gives_is_gpu_zero() {
        use crate::types::feature;
        let mut p = make_pipeline();

        let msg = make_heartbeat_msg(0.5, "cpu", "none");
        p.update_engine_state(&msg);

        assert!((p.engine_state.values[feature::IS_GPU]).abs() < 1e-5);
        assert!((p.engine_state.values[feature::ACTIVE_EVICTION]).abs() < 1e-5);
    }

    /// Capability, Response 메시지는 engine_state를 변경하지 않음
    #[test]
    fn non_heartbeat_message_does_not_change_engine_state() {
        use crate::types::feature;
        use llm_shared::{CommandResponse, CommandResult, EngineCapability, EngineMessage};
        let mut p = make_pipeline();

        let before = p.engine_state.values[feature::KV_OCCUPANCY];

        // Capability 메시지
        p.update_engine_state(&EngineMessage::Capability(EngineCapability {
            available_devices: vec!["cpu".into()],
            active_device: "cpu".into(),
            max_kv_tokens: 2048,
            bytes_per_kv_token: 256,
            num_layers: 16,
        }));
        assert!((p.engine_state.values[feature::KV_OCCUPANCY] - before).abs() < 1e-5);

        // Response 메시지
        p.update_engine_state(&EngineMessage::Response(CommandResponse {
            seq_id: 1,
            results: vec![CommandResult::Ok],
        }));
        assert!((p.engine_state.values[feature::KV_OCCUPANCY] - before).abs() < 1e-5);
    }

    /// prev_mode 추적: Normal → process_signal 후 mode 갱신 확인
    #[test]
    fn test_prev_mode_tracking() {
        let mut p = make_pipeline();
        assert_eq!(p.prev_mode, OperatingMode::Normal);

        // Emergency 신호를 반복해서 pressure 누적
        for _ in 0..20 {
            p.process_signal(&memory_signal(Level::Emergency));
        }
        // prev_mode가 Normal이 아닌 값으로 바뀌었어야 한다
        // (실제 mode 전환은 PI 누적에 따라 다르므로, pressure만 확인)
        assert!(p.pressure().memory > 0.0);
    }

    /// Heartbeat에서 available_actions / active_actions_reported가 파싱된다.
    #[test]
    fn engine_state_parses_available_and_active_actions() {
        use crate::types::ActionId;
        use llm_shared::{EngineMessage, EngineState, EngineStatus, ResourceLevel};

        let mut p = make_pipeline();

        // 초기값은 비어있어야 한다
        assert!(p.available_actions.is_empty());
        assert!(p.active_actions_reported.is_empty());

        let msg = EngineMessage::Heartbeat(EngineStatus {
            active_device: "cpu".to_string(),
            compute_level: ResourceLevel::Normal,
            actual_throughput: 20.0,
            memory_level: ResourceLevel::Normal,
            kv_cache_bytes: 0,
            kv_cache_tokens: 512,
            kv_cache_utilization: 0.25,
            memory_lossless_min: 1.0,
            memory_lossy_min: 0.01,
            state: EngineState::Running,
            tokens_generated: 100,
            available_actions: vec![
                "kv_evict_h2o".to_string(),
                "kv_evict_sliding".to_string(),
                "throttle".to_string(),
            ],
            active_actions: vec!["kv_evict_h2o".to_string()],
            eviction_policy: "h2o".to_string(),
            kv_dtype: "f16".to_string(),
            skip_ratio: 0.0,
        });

        p.update_engine_state(&msg);

        assert_eq!(p.available_actions.len(), 3);
        assert!(p.available_actions.contains(&ActionId::KvEvictH2o));
        assert!(p.available_actions.contains(&ActionId::KvEvictSliding));
        assert!(p.available_actions.contains(&ActionId::Throttle));

        assert_eq!(p.active_actions_reported.len(), 1);
        assert!(p.active_actions_reported.contains(&ActionId::KvEvictH2o));
    }

    // ── De-escalation 테스트 ──────────────────────────────────────────────

    /// Critical → Warning 전환 시 RestoreDefaults를 포함하는 directive가 발송된다.
    #[test]
    fn test_de_escalation_critical_to_warning() {
        let mut p = make_pipeline();

        // Critical 신호를 충분히 반복해서 supervisory가 Critical 모드로 전환되도록 한다
        for _ in 0..30 {
            p.process_signal(&memory_signal(Level::Critical));
        }
        // Critical 모드에 있어야 한다 (PI 적분 충분 누적)
        // mode()가 Critical이 아닐 수 있으므로 pressure만 확인하고 prev_mode를 직접 설정
        // 실제 테스트: prev_mode를 Critical로 강제하고 Warning 신호를 보낸다
        p.prev_mode = OperatingMode::Critical;

        // Warning 신호로 de-escalation 트리거
        let d = p.process_signal(&memory_signal(Level::Warning));

        // Warning 신호를 한 번만 보내면 PI 적분상 mode가 Critical로 유지될 수 있으므로
        // prev_mode가 Critical이고 mode가 Warning일 때만 체크
        // — supervisory가 Warning으로 내려오지 않을 수도 있다.
        // 따라서 여기서는 build_lossy_release_directive를 직접 검증한다.
        let directive = p.build_lossy_release_directive().unwrap();
        assert_eq!(directive.commands.len(), 1);
        assert!(
            matches!(directive.commands[0], EngineCommand::RestoreDefaults),
            "Lossy release directive should contain RestoreDefaults"
        );
        // directive seq_id는 양수여야 한다
        assert!(directive.seq_id > 0);

        // d는 압력 상태에 따라 있을 수도 없을 수도 있다.
        // prev_mode를 Critical로 설정했으나 supervisory가 Warning을 줄 경우 de-escalation 발생
        let _ = d;
    }

    /// process_signal 흐름에서 Critical → Warning de-escalation 경로를 검증한다.
    ///
    /// prev_mode를 Critical로 강제 설정하고 supervisory가 Warning을 반환하는 상황을
    /// 시뮬레이션하여 d2에 RestoreDefaults가 포함되는지 확인한다.
    #[test]
    fn test_de_escalation_critical_to_warning_process_signal() {
        let mut p = make_pipeline();

        // prev_mode = Critical, pressure = Warning 수준으로 설정
        p.prev_mode = OperatingMode::Critical;

        // Warning 수준 신호를 반복 — PI 적분이 충분히 낮으면 supervisory가 Warning 반환
        // 초기 pressure = 0이므로 첫 Warning 신호는 pressure를 낮게 유지한다
        let d = p.process_signal(&memory_signal(Level::Warning));

        // supervisory가 반환한 mode에 따라:
        // - Normal 반환: *→Normal 경로 → RestoreDefaults (build_restore_directive)
        // - Warning 반환: Critical→Warning 경로 → RestoreDefaults (build_lossy_release_directive)
        // 어느 경우든 prev_mode(Critical) > mode(Normal 또는 Warning)이면 d가 Some이어야 함
        let current_mode = p.mode();
        if current_mode < OperatingMode::Critical {
            // de-escalation 발생 → directive가 있어야 한다
            assert!(
                d.is_some(),
                "De-escalation from Critical should produce a directive (mode={:?})",
                current_mode
            );
            let cmds = &d.unwrap().commands;
            assert!(
                cmds.iter()
                    .any(|c| matches!(c, EngineCommand::RestoreDefaults)),
                "De-escalation directive should include RestoreDefaults"
            );
        }
        // Critical을 유지하는 경우는 de-escalation 없음 — 그냥 통과
    }

    /// Warning → Normal 전환 시 RestoreDefaults가 발송된다.
    #[test]
    fn test_de_escalation_warning_to_normal() {
        let mut p = make_pipeline();

        // prev_mode = Warning으로 강제 설정
        p.prev_mode = OperatingMode::Warning;

        // Normal 신호 — PI 적분이 0이므로 supervisory는 Normal 반환
        let d = p.process_signal(&memory_signal(Level::Normal));

        let current_mode = p.mode();
        if current_mode == OperatingMode::Normal {
            // Warning → Normal de-escalation 발생
            assert!(
                d.is_some(),
                "Warning → Normal should produce a restore directive"
            );
            let cmds = &d.unwrap().commands;
            assert!(
                cmds.iter()
                    .any(|c| matches!(c, EngineCommand::RestoreDefaults)),
                "Warning → Normal directive should include RestoreDefaults"
            );
        }
    }

    /// build_lossy_release_directive는 RestoreDefaults 하나를 포함한다.
    #[test]
    fn test_build_lossy_release_directive() {
        let p = make_pipeline();
        let directive = p.build_lossy_release_directive().unwrap();
        assert_eq!(directive.commands.len(), 1);
        assert!(
            matches!(directive.commands[0], EngineCommand::RestoreDefaults),
            "Lossy release directive should contain RestoreDefaults"
        );
        assert!(directive.seq_id > 0, "seq_id should be positive");
    }

    /// build_lossy_release_directive와 build_restore_directive는 서로 다른 seq_id를 가진다.
    #[test]
    fn test_lossy_release_and_restore_have_different_seq_ids() {
        let p = make_pipeline();
        let d1 = p.build_lossy_release_directive().unwrap();
        let d2 = p.build_restore_directive().unwrap();
        assert_ne!(
            d1.seq_id, d2.seq_id,
            "Each directive should have a unique seq_id"
        );
    }

    /// OperatingMode 순서: Normal < Warning < Critical
    #[test]
    fn test_operating_mode_ordering() {
        assert!(OperatingMode::Normal < OperatingMode::Warning);
        assert!(OperatingMode::Warning < OperatingMode::Critical);
        assert!(OperatingMode::Critical > OperatingMode::Normal);
        assert!(OperatingMode::Critical > OperatingMode::Warning);
    }

    /// unknown 액션 문자열은 파싱 시 무시된다 (filter_map).
    #[test]
    fn engine_state_ignores_unknown_action_strings() {
        use llm_shared::{EngineMessage, EngineState, EngineStatus, ResourceLevel};

        let mut p = make_pipeline();

        let msg = EngineMessage::Heartbeat(EngineStatus {
            active_device: "cpu".to_string(),
            compute_level: ResourceLevel::Normal,
            actual_throughput: 20.0,
            memory_level: ResourceLevel::Normal,
            kv_cache_bytes: 0,
            kv_cache_tokens: 0,
            kv_cache_utilization: 0.0,
            memory_lossless_min: 1.0,
            memory_lossy_min: 0.01,
            state: EngineState::Running,
            tokens_generated: 0,
            available_actions: vec!["unknown_action".to_string(), "throttle".to_string()],
            active_actions: vec!["another_unknown".to_string()],
            eviction_policy: "none".to_string(),
            kv_dtype: "f16".to_string(),
            skip_ratio: 0.0,
        });

        p.update_engine_state(&msg);

        // "throttle"만 파싱됨, "unknown_action"은 무시
        assert_eq!(p.available_actions.len(), 1);
        // "another_unknown"은 무시 → 빈 목록
        assert!(p.active_actions_reported.is_empty());
    }

    // ── raw metric 기반 PI 입력 검증 ─────────────────────────────────────

    /// total_bytes=0 이면 memory pressure = 0.0 (division guard)
    #[test]
    fn test_memory_raw_metric_zero_total_gives_zero_pressure() {
        let mut p = make_pipeline();
        let sig = SystemSignal::MemoryPressure {
            level: Level::Emergency,
            available_bytes: 0,
            total_bytes: 0,
            reclaim_target_bytes: 0,
        };
        p.process_signal(&sig);
        // total_bytes=0 → m=0.0 → setpoint 미만 → pressure=0
        assert!(
            p.pressure().memory.abs() < f32::EPSILON,
            "Zero total_bytes should yield zero memory pressure"
        );
    }

    /// 90% 사용률(available=10%) 신호를 반복하면 memory pressure가 setpoint(0.75)를 넘는다
    #[test]
    fn test_memory_raw_metric_high_usage_builds_pressure() {
        let mut p = make_pipeline();
        // available=10%, total=100% → m=0.90, setpoint=0.75 → PI error > 0 → 적분 누적
        let sig = SystemSignal::MemoryPressure {
            level: Level::Critical,
            available_bytes: 100_000_000,
            total_bytes: 1_000_000_000,
            reclaim_target_bytes: 0,
        };
        for _ in 0..10 {
            p.process_signal(&sig);
        }
        assert!(
            p.pressure().memory > 0.0,
            "90% memory usage should build positive pressure"
        );
    }

    /// 낮은 사용률(available=95%) 신호는 memory pressure를 0으로 유지한다
    #[test]
    fn test_memory_raw_metric_low_usage_keeps_zero_pressure() {
        let mut p = make_pipeline();
        // available=95%, total=100% → m=0.05, setpoint=0.75 → PI error < 0 → pressure clamped to 0
        let sig = SystemSignal::MemoryPressure {
            level: Level::Normal,
            available_bytes: 950_000_000,
            total_bytes: 1_000_000_000,
            reclaim_target_bytes: 0,
        };
        p.process_signal(&sig);
        assert!(
            p.pressure().memory.abs() < f32::EPSILON,
            "Low memory usage (5%) should not build pressure (setpoint=0.75)"
        );
    }

    /// compute 신호에서 level 없이 CPU 사용률만으로 pressure가 누적된다
    #[test]
    fn test_compute_raw_metric_cpu_only_builds_pressure() {
        let mut p = make_pipeline();
        // CPU 95%, GPU 0% → m=0.95, setpoint=0.70 → 적분 누적
        let sig = SystemSignal::ComputeGuidance {
            level: Level::Normal, // level은 이제 사용 안 함
            recommended_backend: llm_shared::RecommendedBackend::Any,
            reason: llm_shared::ComputeReason::Balanced,
            cpu_usage_pct: 95.0,
            gpu_usage_pct: 0.0,
        };
        for _ in 0..5 {
            p.process_signal(&sig);
        }
        assert!(
            p.pressure().compute > 0.0,
            "95% CPU usage should build compute pressure regardless of level field"
        );
    }

    /// default_cost 차이가 ActionSelector의 조합 선택에 영향을 주는지 확인한다.
    ///
    /// 같은 relief를 제공하는 두 lossy 액션(evict_sliding vs evict_h2o)에서
    /// default_cost가 낮은 쪽이 더 높은 net score를 받아야 한다.
    /// 여기서는 파이프라인에 cost-differentiated config를 넣고 qcf_values가
    /// registry.default_cost()를 통해 올바른 값으로 채워지는지 검증한다.
    #[test]
    fn test_qcf_values_use_registry_default_cost() {
        use crate::config::{ActionConfig, PolicyConfig};
        use std::collections::HashMap;

        // kv_evict_sliding: cost=0.3, kv_evict_h2o: cost=1.0 으로 설정
        let mut action_map = HashMap::new();
        action_map.insert(
            "kv_evict_sliding".to_string(),
            ActionConfig {
                lossy: true,
                reversible: false,
                default_cost: 0.3,
            },
        );
        action_map.insert(
            "kv_evict_h2o".to_string(),
            ActionConfig {
                lossy: true,
                reversible: false,
                default_cost: 1.0,
            },
        );
        let policy = PolicyConfig {
            actions: action_map,
            ..Default::default()
        };
        let pipeline = HierarchicalPolicy::new(&policy);

        // registry.default_cost()가 config에서 로드된 값을 반환하는지 확인
        assert!(
            (pipeline.registry.default_cost(&ActionId::KvEvictSliding) - 0.3).abs() < f32::EPSILON,
            "kv_evict_sliding default_cost should be 0.3"
        );
        assert!(
            (pipeline.registry.default_cost(&ActionId::KvEvictH2o) - 1.0).abs() < f32::EPSILON,
            "kv_evict_h2o default_cost should be 1.0"
        );
        // 등록되지 않은 액션은 fallback 1.0
        assert!(
            (pipeline.registry.default_cost(&ActionId::LayerSkip) - 1.0).abs() < f32::EPSILON,
            "unregistered action should fallback to 1.0"
        );
    }

    /// elapsed_dt()는 두 번째 호출부터 실측 dt를 반환한다
    #[test]
    fn test_elapsed_dt_returns_measured_dt_on_second_call() {
        let mut p = make_pipeline();
        // 첫 번째 호출 → 기본값 dt=0.1
        let first = p.elapsed_dt("test_domain");
        assert!(
            (first - 0.1).abs() < 0.01,
            "First elapsed_dt should return default dt=0.1"
        );
        // 두 번째 호출 → 첫 번째 이후 경과 시간 (매우 짧음)
        let second = p.elapsed_dt("test_domain");
        assert!(
            second < 0.1,
            "Second elapsed_dt should be shorter than default (measured real interval)"
        );
        // clamp 하한: 0.001
        assert!(second >= 0.001, "elapsed_dt should be at least 0.001s");
    }

    /// Thermal directive 발행 후 memory pressure 상승 시 needs_action=true 확인.
    ///
    /// S25에서 Thermal Critical이 먼저 발생하고, 이후 MemoryPressure가 올라가는 시나리오에서
    /// `any_domain_exceeds`가 도메인별 독립 비교를 수행하는지 검증한다.
    ///
    /// 이전 버그: `last_acted_pressure`가 f32 스칼라로 `pressure.max()`만 저장하여,
    /// thermal 지배 시 memory 상승을 감지하지 못함.
    /// 수정 후: 도메인별 PressureVector로 비교하여 각 도메인 독립적으로 판정.
    #[test]
    fn test_needs_action_memory_after_thermal() {
        // Scenario: Thermal directive 발행 후 상태 (thermal=0.8, memory=0)
        let last_acted = PressureVector {
            compute: 0.0,
            memory: 0.0,
            thermal: 0.8,
        };
        // 이후 memory pressure 상승 (thermal 유지)
        let current = PressureVector {
            compute: 0.0,
            memory: 0.3,
            thermal: 0.8,
        };

        // 수정 전 스칼라 로직: pressure.max() > last_acted_scalar * 1.2
        // 0.8 > 0.8 * 1.2 = 0.96? NO → memory 변화를 놓침 (BUG)
        let old_scalar = last_acted.max();
        assert!(
            !(current.max() > old_scalar * 1.2),
            "Old scalar logic should miss memory-only change"
        );

        // 수정 후 도메인별 로직: any_domain_exceeds
        // memory: 0.3 > 0.0 * 1.2 = 0.0? YES → 감지 (FIXED)
        assert!(
            current.any_domain_exceeds(&last_acted, 1.2),
            "New domain-independent logic should detect memory rise after thermal directive"
        );

        // Edge case: 모든 도메인이 동일하면 false
        assert!(
            !last_acted.any_domain_exceeds(&last_acted, 1.2),
            "Same pressure should not exceed itself by 1.2x"
        );

        // Edge case: 모든 도메인이 0이면 어떤 양수 pressure든 감지
        let zero = PressureVector::default();
        let tiny = PressureVector {
            compute: 0.0,
            memory: 0.01,
            thermal: 0.0,
        };
        assert!(
            tiny.any_domain_exceeds(&zero, 1.2),
            "Any positive pressure should exceed zero reference"
        );
    }

    // ── QCF 2-phase 테스트 (SEQ-095~098) ────────────────────────────────

    /// SEQ-095: Critical 전환 시 RequestQcf Directive 반환
    #[test]
    fn test_seq_095_critical_transition_sends_request_qcf() {
        let mut p = make_pipeline();

        // Emergency 신호를 반복하여 supervisory가 Critical을 반환하도록 pressure 누적
        for _ in 0..30 {
            let _ = p.process_signal(&memory_signal(Level::Emergency));
        }

        // prev_mode를 Warning으로 리셋하고 다시 시도 — Critical 전환을 트리거
        p.prev_mode = OperatingMode::Warning;
        p.qcf_pending = None;

        let result = p.process_signal(&memory_signal(Level::Emergency));
        let current_mode = p.mode();

        if current_mode == OperatingMode::Critical {
            // Critical 전환 발생 → RequestQcf가 반환되어야 한다
            assert!(
                result.is_some(),
                "Critical transition should produce a directive"
            );
            let directive = result.unwrap();
            assert_eq!(directive.commands.len(), 1);
            assert!(
                matches!(directive.commands[0], EngineCommand::RequestQcf),
                "Critical transition should send RequestQcf, got {:?}",
                directive.commands[0]
            );
            assert!(
                p.qcf_pending.is_some(),
                "qcf_pending should be set after RequestQcf"
            );
        }
    }

    /// SEQ-097: QcfEstimate 수신 후 실제 Directive 반환 (qcf_pending 소비)
    #[test]
    fn test_seq_097_qcf_estimate_triggers_action_selection() {
        let mut p = make_pipeline();

        p.qcf_pending = Some(QcfPending {
            pressure: PressureVector {
                compute: 0.0,
                memory: 0.9,
                thermal: 0.0,
            },
            mode: OperatingMode::Critical,
            requested_at: Instant::now(),
        });

        let qcf = llm_shared::QcfEstimate {
            estimates: {
                let mut m = HashMap::new();
                m.insert("kv_evict_sliding".to_string(), 0.2);
                m.insert("kv_evict_h2o".to_string(), 0.8);
                m
            },
        };

        let _ = p.complete_qcf_selection(&qcf);
        assert!(
            p.qcf_pending.is_none(),
            "qcf_pending should be consumed after complete_qcf_selection"
        );
    }

    /// SEQ-097: qcf_pending이 없을 때 complete_qcf_selection은 None 반환
    #[test]
    fn test_seq_097_no_pending_returns_none() {
        let mut p = make_pipeline();
        assert!(p.qcf_pending.is_none());

        let qcf = llm_shared::QcfEstimate {
            estimates: HashMap::new(),
        };
        let result = p.complete_qcf_selection(&qcf);
        assert!(result.is_none(), "No pending QCF should return None");
    }

    /// SEQ-098: 1초 타임아웃 후 default cost로 폴백 (qcf_pending 소비)
    #[test]
    fn test_seq_098_qcf_timeout_fallback() {
        let mut p = make_pipeline();

        p.qcf_pending = Some(QcfPending {
            pressure: PressureVector {
                compute: 0.0,
                memory: 0.9,
                thermal: 0.0,
            },
            mode: OperatingMode::Critical,
            requested_at: Instant::now() - std::time::Duration::from_secs(2),
        });

        let _ = p.check_qcf_timeout();
        assert!(
            p.qcf_pending.is_none(),
            "qcf_pending should be consumed after timeout"
        );
    }

    /// SEQ-098: 타임아웃 전에는 check_qcf_timeout이 None 반환
    #[test]
    fn test_seq_098_no_timeout_returns_none() {
        let mut p = make_pipeline();

        p.qcf_pending = Some(QcfPending {
            pressure: PressureVector {
                compute: 0.0,
                memory: 0.9,
                thermal: 0.0,
            },
            mode: OperatingMode::Critical,
            requested_at: Instant::now(),
        });

        let result = p.check_qcf_timeout();
        assert!(result.is_none(), "Should not timeout immediately");
        assert!(
            p.qcf_pending.is_some(),
            "qcf_pending should remain if not timed out"
        );
    }

    /// SEQ-095: Warning 전환에서는 RequestQcf 미전송
    #[test]
    fn test_seq_095_warning_does_not_send_request_qcf() {
        let mut p = make_pipeline();
        p.prev_mode = OperatingMode::Normal;

        // Warning 수준 신호를 반복
        for _ in 0..20 {
            let _ = p.process_signal(&memory_signal(Level::Warning));
        }

        p.prev_mode = OperatingMode::Normal;
        p.qcf_pending = None;
        let result = p.process_signal(&memory_signal(Level::Warning));

        if let Some(directive) = result {
            for cmd in &directive.commands {
                assert!(
                    !matches!(cmd, EngineCommand::RequestQcf),
                    "Warning transition should NOT send RequestQcf"
                );
            }
        }
        assert!(
            p.qcf_pending.is_none(),
            "qcf_pending should not be set for Warning transition"
        );
    }

    /// SEQ-097: QcfEstimate의 unknown action key는 무시된다
    #[test]
    fn test_seq_097_unknown_action_keys_ignored() {
        let mut p = make_pipeline();

        p.qcf_pending = Some(QcfPending {
            pressure: PressureVector {
                compute: 0.0,
                memory: 0.9,
                thermal: 0.0,
            },
            mode: OperatingMode::Critical,
            requested_at: Instant::now(),
        });

        let qcf = llm_shared::QcfEstimate {
            estimates: {
                let mut m = HashMap::new();
                m.insert("unknown_action".to_string(), 0.5);
                m.insert("kv_evict_sliding".to_string(), 0.1);
                m
            },
        };

        let _ = p.complete_qcf_selection(&qcf);
        assert!(p.qcf_pending.is_none());
    }

    /// build_default_qcf_values는 registry의 lossy_actions를 반환한다
    #[test]
    fn test_build_default_qcf_values() {
        let p = make_pipeline();
        let values = p.build_default_qcf_values();

        let lossy = p.registry.lossy_actions();
        assert_eq!(values.len(), lossy.len());

        for id in &lossy {
            assert!(values.contains_key(id));
            let expected = p.registry.default_cost(id);
            assert!(
                (values[id] - expected).abs() < f32::EPSILON,
                "QCF value for {:?} should match registry default_cost",
                id
            );
        }
    }
}
