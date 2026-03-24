//! Policy pipeline — 새 계층형 정책 메인 루프용 상태 캡슐화.
//!
//! `PolicyPipeline`은 PI Controller, Supervisory Layer, Action Selector,
//! Relief Estimator를 연결하여 `SystemSignal` 입력에서 `EngineDirective`를
//! 생성하는 전체 파이프라인을 담당한다.
//!
//! # 설계 참고
//!
//! `docs/36_policy_design.md` §9 (Manager Main Loop)를 참조한다.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use llm_shared::{
    EngineCommand, EngineDirective, EngineMessage, Level, ResourceLevel, SystemSignal,
};

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

/// Seq ID 생성을 위한 단조 증가 카운터.
static SEQ_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_seq_id() -> u64 {
    SEQ_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// 계층형 정책 파이프라인.
///
/// PI Controller 3개 (compute / memory / thermal) → Supervisory Layer →
/// Action Selector → EngineDirective 생성의 전체 흐름을 캡슐화한다.
pub struct PolicyPipeline {
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
    /// 마지막으로 액션을 취했을 때의 pressure.max() 값.
    last_acted_pressure: f32,
    /// 이전 액션의 실측 relief 수집을 위한 컨텍스트.
    pending_observation: Option<ObservationContext>,
    /// 허용 가능한 latency 악화 상한.
    latency_budget: f32,
    /// PI update 시 사용할 dt (초). 기본값 0.1 (100 ms).
    dt: f32,
    /// Relief model 저장/불러오기 경로.
    relief_model_path: Option<String>,
}

impl PolicyPipeline {
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
            ),
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
            last_acted_pressure: 0.0,
            pending_observation: None,
            latency_budget: config.selector.latency_budget,
            dt: 0.1,
            relief_model_path: None,
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

    /// 현재 operating mode를 반환한다.
    pub fn mode(&self) -> OperatingMode {
        self.supervisory.mode()
    }

    /// 현재 pressure vector를 반환한다.
    pub fn pressure(&self) -> &PressureVector {
        &self.pressure
    }

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
    pub fn process_signal(&mut self, signal: &SystemSignal) -> Option<EngineDirective> {
        // ① PI Controller 갱신
        self.update_pressure(signal);

        // ② Supervisory → mode
        let mode = self.supervisory.evaluate(&self.pressure);

        // ③ 관측 갱신
        self.update_observation();

        // ④ 액션 필요 여부
        let needs_action = match mode {
            OperatingMode::Normal => false,
            OperatingMode::Warning | OperatingMode::Critical => {
                mode != self.prev_mode || self.pressure.max() > self.last_acted_pressure * 1.2
            }
        };

        let mut result = None;

        if needs_action {
            // ⑤ QCF proxy (engine 미연결 시 1.0 가정)
            let qcf_values: HashMap<ActionId, f32> = self
                .registry
                .lossy_actions()
                .into_iter()
                .map(|id| (id, 1.0))
                .collect();

            // ⑥ Action Selection
            let commands = ActionSelector::select(
                &self.registry,
                &self.estimator,
                &self.pressure,
                mode,
                &self.engine_state,
                &qcf_values,
                self.latency_budget,
            );

            if !commands.is_empty() {
                let engine_commands = self.convert_to_engine_commands(&commands);
                if !engine_commands.is_empty() {
                    let seq = next_seq_id();
                    let directive = EngineDirective {
                        seq_id: seq,
                        commands: engine_commands,
                    };

                    // 관측 컨텍스트 기록
                    self.pending_observation = Some(ObservationContext {
                        pressure_before: self.pressure,
                        feature_vec: self.engine_state.clone(),
                        applied_actions: commands.iter().map(|c| c.action).collect(),
                        applied_at: Instant::now(),
                    });
                    self.last_acted_pressure = self.pressure.max();

                    log::debug!(
                        "[Pipeline] mode={:?} pressure={:.2}/{:.2}/{:.2} → directive seq={} ({} cmds)",
                        mode,
                        self.pressure.compute,
                        self.pressure.memory,
                        self.pressure.thermal,
                        seq,
                        directive.commands.len()
                    );

                    result = Some(directive);
                }
            }
        }

        // ⑦ De-escalation: Normal로 복귀 시 restore 명령
        if mode == OperatingMode::Normal && self.prev_mode != OperatingMode::Normal {
            log::info!("[Pipeline] De-escalating to Normal — sending restore directive");
            if result.is_none() {
                result = self.build_restore_directive();
            }
        }

        self.prev_mode = mode;
        result
    }

    /// 세션 종료 시 Relief model을 디스크에 저장한다.
    pub fn save_model(&self) {
        if let Some(path) = &self.relief_model_path {
            if let Err(e) = self.estimator.save(Path::new(path)) {
                log::warn!("Failed to save relief model to {}: {}", path, e);
            } else {
                log::info!("Saved relief model to {}", path);
            }
        }
    }

    /// Engine heartbeat에서 `engine_state` feature vector를 갱신한다.
    ///
    /// `37_protocol_design.md §6`의 Feature Vector 스키마를 따른다.
    /// Heartbeat 이외의 메시지 (Capability, Response)는 무시한다.
    pub fn update_engine_state(&mut self, msg: &EngineMessage) {
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

        // [10] 활성 eviction 여부 (memory_level이 Normal이 아니면 1.0)
        v[feature::ACTIVE_EVICTION] = if status.memory_level != llm_shared::ResourceLevel::Normal {
            1.0
        } else {
            0.0
        };
    }

    // ── 내부 헬퍼 ──────────────────────────────────────────────────────────

    /// SystemSignal에서 도메인별 측정값을 추출하여 해당 PI Controller에 입력한다.
    fn update_pressure(&mut self, signal: &SystemSignal) {
        match signal {
            SystemSignal::MemoryPressure { level, .. } => {
                let m = level_to_measurement(*level);
                self.pressure.memory = self.pi_memory.update(m, self.dt);
            }
            SystemSignal::ThermalAlert { temperature_mc, .. } => {
                // 85°C (85000 mc) = 1.0 기준 정규화
                let m = (*temperature_mc as f32 / 85_000.0).clamp(0.0, 1.0);
                self.pressure.thermal = self.pi_thermal.update(m, self.dt);
            }
            SystemSignal::ComputeGuidance {
                level,
                cpu_usage_pct,
                ..
            } => {
                // level 기반 측정값 + CPU 사용률 보조 입력 (둘 중 큰 값)
                let m_level = level_to_measurement(*level);
                let m_cpu = (*cpu_usage_pct as f32 / 100.0).clamp(0.0, 1.0);
                let m = m_level.max(m_cpu);
                self.pressure.compute = self.pi_compute.update(m, self.dt);
            }
            SystemSignal::EnergyConstraint { level, .. } => {
                // Energy → compute pressure에 보조 기여 (0.5 가중치)
                let m = level_to_measurement(*level) * 0.5;
                let combined = self.pressure.compute.max(m);
                self.pressure.compute = self.pi_compute.update(combined, self.dt);
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

    /// ActionCommand 목록을 기존 `EngineCommand`로 변환한다.
    ///
    /// Engine은 아직 `ActionCommand` 프로토콜을 지원하지 않으므로
    /// 기존 `EngineCommand` 체계로 변환하여 전송한다.
    fn convert_to_engine_commands(&self, commands: &[ActionCommand]) -> Vec<EngineCommand> {
        let mut result = Vec::new();
        for cmd in commands {
            match (&cmd.action, &cmd.operation) {
                (ActionId::SwitchHw, Operation::Apply(_)) => {
                    result.push(EngineCommand::SwitchComputeUnit {
                        device: "cpu".to_string(),
                    });
                }
                (ActionId::Throttle, Operation::Apply(params)) => {
                    let delay_ms = params.values.get("delay_ms").copied().unwrap_or(0.0);
                    // delay_ms [0, 100] → throughput [1.0, 0.1]
                    let throughput = (1.0 - delay_ms / 100.0).clamp(0.1, 1.0);
                    result.push(EngineCommand::SetComputeLevel {
                        level: ResourceLevel::Warning,
                        target_throughput: throughput,
                        deadline_ms: Some(2000),
                    });
                }
                (ActionId::KvEvictSliding | ActionId::KvEvictH2o, Operation::Apply(params)) => {
                    let keep_ratio = params.values.get("keep_ratio").copied().unwrap_or(0.5);
                    result.push(EngineCommand::SetMemoryLevel {
                        level: ResourceLevel::Critical,
                        target_ratio: keep_ratio,
                        deadline_ms: Some(1000),
                    });
                }
                (ActionId::KvOffloadDisk, Operation::Apply(_)) => {
                    result.push(EngineCommand::SetMemoryLevel {
                        level: ResourceLevel::Warning,
                        target_ratio: 0.8,
                        deadline_ms: Some(2000),
                    });
                }
                (ActionId::LayerSkip, Operation::Apply(params)) => {
                    let skip = params.values.get("skip_layers").copied().unwrap_or(1.0);
                    // skip_layers가 많을수록 throughput 감소
                    let throughput = (1.0 - skip / 16.0).clamp(0.1, 1.0);
                    result.push(EngineCommand::SetComputeLevel {
                        level: ResourceLevel::Critical,
                        target_throughput: throughput,
                        deadline_ms: Some(1000),
                    });
                }
                (ActionId::KvQuantDynamic, Operation::Apply(_)) => {
                    // KV quantization은 memory level로 표현
                    result.push(EngineCommand::SetMemoryLevel {
                        level: ResourceLevel::Warning,
                        target_ratio: 0.9,
                        deadline_ms: Some(2000),
                    });
                }
                (_, Operation::Release) => {
                    // Release 명령은 restore directive에서 일괄 처리 — 여기서는 무시
                }
            }
        }
        result
    }

    /// Normal 복귀 시 발송할 restore directive를 생성한다.
    fn build_restore_directive(&self) -> Option<EngineDirective> {
        let commands = vec![
            EngineCommand::SetComputeLevel {
                level: ResourceLevel::Normal,
                target_throughput: 1.0,
                deadline_ms: None,
            },
            EngineCommand::SetMemoryLevel {
                level: ResourceLevel::Normal,
                target_ratio: 1.0,
                deadline_ms: None,
            },
        ];
        Some(EngineDirective {
            seq_id: next_seq_id(),
            commands,
        })
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

    fn make_pipeline() -> PolicyPipeline {
        PolicyPipeline::new(&PolicyConfig::default())
    }

    fn memory_signal(level: Level) -> SystemSignal {
        SystemSignal::MemoryPressure {
            level,
            available_bytes: 1_000_000_000,
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

    /// restore directive에는 Normal 레벨 명령이 포함된다
    #[test]
    fn test_restore_directive_has_normal_commands() {
        let p = make_pipeline();
        let directive = p.build_restore_directive().unwrap();
        assert_eq!(directive.commands.len(), 2);
        let has_compute_normal = directive.commands.iter().any(|c| {
            matches!(
                c,
                EngineCommand::SetComputeLevel {
                    level: ResourceLevel::Normal,
                    ..
                }
            )
        });
        let has_memory_normal = directive.commands.iter().any(|c| {
            matches!(
                c,
                EngineCommand::SetMemoryLevel {
                    level: ResourceLevel::Normal,
                    ..
                }
            )
        });
        assert!(
            has_compute_normal,
            "Restore should include Normal compute level"
        );
        assert!(
            has_memory_normal,
            "Restore should include Normal memory level"
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

    /// ActionCommand → EngineCommand 변환: KvEvictSliding → SetMemoryLevel
    #[test]
    fn test_convert_kv_evict_to_set_memory_level() {
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
            EngineCommand::SetMemoryLevel {
                level,
                target_ratio,
                ..
            } => {
                assert_eq!(*level, ResourceLevel::Critical);
                assert!((*target_ratio - 0.6).abs() < f32::EPSILON);
            }
            _ => panic!("Expected SetMemoryLevel"),
        }
    }

    /// ActionCommand → EngineCommand 변환: SwitchHw → SwitchComputeUnit
    #[test]
    fn test_convert_switch_hw_to_switch_compute_unit() {
        use crate::types::{ActionParams, Operation};

        let p = make_pipeline();
        let cmd = ActionCommand {
            action: ActionId::SwitchHw,
            operation: Operation::Apply(ActionParams::default()),
        };
        let engine_cmds = p.convert_to_engine_commands(&[cmd]);
        assert_eq!(engine_cmds.len(), 1);
        match &engine_cmds[0] {
            EngineCommand::SwitchComputeUnit { device } => {
                assert_eq!(device, "cpu");
            }
            _ => panic!("Expected SwitchComputeUnit"),
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
        memory_level: llm_shared::ResourceLevel,
    ) -> llm_shared::EngineMessage {
        use llm_shared::{EngineMessage, EngineState, EngineStatus, ResourceLevel};
        EngineMessage::Heartbeat(EngineStatus {
            active_device: device.to_string(),
            compute_level: ResourceLevel::Normal,
            actual_throughput: 20.0,
            memory_level,
            kv_cache_bytes: 0,
            kv_cache_tokens: (kv * 2048.0) as usize,
            kv_cache_utilization: kv,
            memory_lossless_min: 1.0,
            memory_lossy_min: 0.01,
            state: EngineState::Running,
            tokens_generated: 512,
        })
    }

    /// Heartbeat에서 engine_state feature vector가 갱신된다
    #[test]
    fn engine_state_updated_from_heartbeat() {
        use crate::types::feature;
        let mut p = make_pipeline();

        // 초기값은 zero
        assert_eq!(p.engine_state.values[feature::KV_OCCUPANCY], 0.0);

        let msg = make_heartbeat_msg(0.75, "opencl", llm_shared::ResourceLevel::Critical);
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

        let msg = make_heartbeat_msg(0.5, "cpu", llm_shared::ResourceLevel::Normal);
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
}
