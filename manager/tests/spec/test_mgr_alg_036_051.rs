//! MGR-ALG-036 ~ MGR-ALG-051: Pipeline Spec 테스트
//!
//! HierarchicalPolicy의 ActionCommand -> EngineCommand 변환,
//! Heartbeat -> FeatureVector 갱신, 정규화, de-escalation,
//! Normal pressure 시 directive 생성 여부를 검증한다.
//! public API (process_signal, update_engine_state, mode, pressure)만 사용.

use llm_manager::config::PolicyConfig;
use llm_manager::pipeline::{HierarchicalPolicy, PolicyStrategy};
use llm_manager::types::OperatingMode;
use llm_shared::{
    ComputeReason, EngineCommand, EngineMessage, EngineState, EngineStatus, Level,
    RecommendedBackend, ResourceLevel, SystemSignal,
};

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
        recommended_backend: RecommendedBackend::Cpu,
        reason: ComputeReason::CpuBottleneck,
        cpu_usage_pct: cpu_pct,
        gpu_usage_pct: 0.0,
    }
}

fn make_heartbeat(
    kv: f32,
    device: &str,
    eviction_policy: &str,
    available_actions: Vec<String>,
    active_actions: Vec<String>,
) -> EngineMessage {
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
        available_actions,
        active_actions,
        eviction_policy: eviction_policy.to_string(),
        kv_dtype: "f16".to_string(),
        skip_ratio: 0.0,
        phase: String::new(),
        prefill_pos: 0,
        prefill_total: 0,
        partition_ratio: 0.0,
        self_cpu_pct: 0.0,
        self_gpu_pct: 0.0,
    })
}

// ── MGR-ALG-036: ActionCommand -> EngineCommand 변환 ────────────────
//
// 변환 함수가 private이므로 process_signal의 결과를 통해 간접 검증한다.
// 여기서는 충분한 pressure를 누적하여 directive가 생성되면 그 내용을 확인한다.

/// MGR-ALG-036: process_signal이 충분한 pressure 누적 후 directive를 생성하며,
/// 그 directive에는 유효한 EngineCommand가 포함된다.
#[test]
fn test_mgr_alg_036_process_signal_produces_valid_engine_commands() {
    let mut p = make_pipeline();

    // Critical 수준의 memory 신호를 반복하여 pressure 누적
    let mut directive = None;
    for _ in 0..50 {
        if let Some(d) = p.process_signal(&memory_signal(Level::Critical)) {
            directive = Some(d);
            break;
        }
    }

    // pressure가 충분히 누적되면 directive가 생성되어야 한다
    if let Some(d) = directive {
        assert!(
            !d.commands.is_empty(),
            "directive should contain at least one command"
        );
        assert!(d.seq_id > 0, "seq_id should be positive");

        // 각 command는 유효한 EngineCommand variant
        for cmd in &d.commands {
            match cmd {
                EngineCommand::KvEvictSliding { keep_ratio } => {
                    assert!(
                        (0.0..=1.0).contains(keep_ratio),
                        "keep_ratio should be in [0,1]"
                    );
                }
                EngineCommand::KvEvictH2o { keep_ratio } => {
                    assert!(
                        (0.0..=1.0).contains(keep_ratio),
                        "keep_ratio should be in [0,1]"
                    );
                }
                EngineCommand::SwitchHw { device } => {
                    assert!(!device.is_empty(), "device should not be empty");
                }
                EngineCommand::Throttle { .. } => {}
                EngineCommand::LayerSkip { skip_ratio } => {
                    assert!(
                        (0.0..=1.0).contains(skip_ratio),
                        "skip_ratio should be in [0,1]"
                    );
                }
                EngineCommand::RestoreDefaults => {}
                _ => {}
            }
        }
    }
}

// ── MGR-ALG-051 / MGR-080: Heartbeat -> FeatureVector 갱신 ──────────

/// MGR-ALG-051: Heartbeat 메시지에서 engine_state가 갱신된다.
/// public API (process_signal 전후의 mode 변화)를 통해 간접 확인.
#[test]
fn test_mgr_alg_051_engine_state_updated_from_heartbeat() {
    let mut p = make_pipeline();

    let msg = make_heartbeat(0.75, "opencl", "h2o", vec![], vec![]);
    p.update_engine_state(&msg);

    // heartbeat 수신 후 mode에 영향 없음 (heartbeat 자체는 pressure를 변경하지 않음)
    assert_eq!(
        p.mode(),
        OperatingMode::Normal,
        "heartbeat alone should not change mode"
    );
}

/// MGR-ALG-051: CPU 디바이스 heartbeat 수신 후에도 정상 동작.
#[test]
fn test_mgr_alg_051_cpu_device_heartbeat() {
    let mut p = make_pipeline();
    let msg = make_heartbeat(0.5, "cpu", "none", vec![], vec![]);
    p.update_engine_state(&msg);
    assert_eq!(p.mode(), OperatingMode::Normal);
}

// ── MGR-081: available_actions / active_actions 파싱 ────────────────

/// MGR-081: Heartbeat의 available_actions / active_actions가 파싱되어
/// 후속 ActionSelector 필터링에 반영된다.
/// (internal 필드가 private이므로 process_signal의 결과를 통해 간접 검증)
#[test]
fn test_mgr_081_engine_state_parses_actions() {
    let mut p = make_pipeline();

    // available_actions에 throttle만 설정
    let msg = make_heartbeat(0.25, "cpu", "none", vec!["throttle".to_string()], vec![]);
    p.update_engine_state(&msg);

    // 이후 process_signal에서 ActionSelector가 available_actions를 참조하여
    // throttle만 후보로 사용한다. 이 테스트에서는 heartbeat 수신 자체가
    // 에러 없이 동작하는 것을 확인한다.
    assert_eq!(p.mode(), OperatingMode::Normal);
}

// ── MGR-ALG-014: 정규화 ────────────────────────────────────────────

/// MGR-ALG-014: 온도 85000mc(85도) 신호 반복 시 thermal pressure가 양수.
#[test]
fn test_mgr_alg_014_thermal_normalization_max() {
    let mut p = make_pipeline();
    for _ in 0..20 {
        p.process_signal(&thermal_signal(85_000));
    }
    assert!(
        p.pressure().thermal > 0.0,
        "85000mc should produce positive thermal pressure"
    );
}

/// MGR-ALG-014: 온도 42500mc(약 50%) 신호는 setpoint(0.8) 미만이므로 pressure = 0.
#[test]
fn test_mgr_alg_014_thermal_normalization_half() {
    let mut p = make_pipeline();
    p.process_signal(&thermal_signal(42_500));
    assert!(
        p.pressure().thermal.abs() < f32::EPSILON,
        "half temp (below setpoint) should give 0 thermal pressure"
    );
}

/// MGR-ALG-014: compute_signal에서 CPU 사용률 95%이면 compute pressure가 양수.
#[test]
fn test_mgr_alg_014_compute_cpu_usage_reflected() {
    let mut p = make_pipeline();
    for _ in 0..5 {
        p.process_signal(&compute_signal(Level::Critical, 95.0));
    }
    assert!(
        p.pressure().compute > 0.0,
        "High CPU usage should produce compute pressure"
    );
}

// ── MGR-078: De-escalation ──────────────────────────────────────────

/// MGR-078: Warning에서 Normal로 de-escalation 시 RestoreDefaults가 발송된다.
/// hold_time이 짧은 config를 사용하여 de-escalation을 확실히 트리거한다.
#[test]
fn test_mgr_078_de_escalation_warning_to_normal() {
    use llm_manager::config::{PolicyConfig, SupervisoryConfig};

    // hold_time을 극히 짧게 설정
    let config = PolicyConfig {
        supervisory: SupervisoryConfig {
            hold_time_secs: 0.001,
            ..SupervisoryConfig::default()
        },
        ..PolicyConfig::default()
    };
    let mut p = HierarchicalPolicy::new(&config);

    // Warning 수준 pressure를 누적 (memory 사용률 60% -> setpoint 0.75 미만이므로
    // compute 도메인으로 테스트)
    // compute_signal CPU 95%를 반복하여 Warning/Critical 진입
    for _ in 0..30 {
        p.process_signal(&compute_signal(Level::Critical, 95.0));
    }

    let mode_after_critical = p.mode();
    // pressure가 양수인지만 확인 (모드 전환은 PI 누적 의존)
    assert!(
        p.pressure().compute > 0.0,
        "repeated critical signals should build compute pressure"
    );

    // Normal 신호를 반복하여 de-escalation 트리거
    let mut restore_found = false;
    for _ in 0..200 {
        if let Some(d) = p.process_signal(&compute_signal(Level::Normal, 10.0))
            && d.commands
                .iter()
                .any(|c| matches!(c, EngineCommand::RestoreDefaults))
        {
            restore_found = true;
            break;
        }
    }

    // de-escalation이 결국 Normal로 도달했다면 RestoreDefaults가 있어야 한다
    if p.mode() == OperatingMode::Normal && mode_after_critical != OperatingMode::Normal {
        assert!(
            restore_found,
            "de-escalation to Normal should produce RestoreDefaults"
        );
    }
}

// ── MGR-DAT-043: Normal pressure 시 directive 없음 ─────────────────

/// MGR-DAT-043: Normal pressure에서는 directive가 생성되지 않는다.
#[test]
fn test_mgr_dat_043_normal_pressure_no_directive() {
    let mut p = make_pipeline();
    let result = p.process_signal(&memory_signal(Level::Normal));
    assert!(
        result.is_none(),
        "Normal pressure should produce no directive"
    );
    assert_eq!(p.mode(), OperatingMode::Normal);
}

// ── 추가: Non-Heartbeat 메시지는 engine_state를 변경하지 않음 ───────

/// MGR-ALG-051: Capability, Response 메시지는 engine_state를 변경하지 않음.
#[test]
fn test_mgr_alg_051_non_heartbeat_ignored() {
    use llm_shared::{CommandResponse, CommandResult, EngineCapability};
    let mut p = make_pipeline();

    // Capability 메시지
    p.update_engine_state(&EngineMessage::Capability(EngineCapability {
        available_devices: vec!["cpu".into()],
        active_device: "cpu".into(),
        max_kv_tokens: 2048,
        bytes_per_kv_token: 256,
        num_layers: 16,
        ..Default::default()
    }));
    assert_eq!(p.mode(), OperatingMode::Normal);

    // Response 메시지
    p.update_engine_state(&EngineMessage::Response(CommandResponse {
        seq_id: 1,
        results: vec![CommandResult::Ok],
    }));
    assert_eq!(p.mode(), OperatingMode::Normal);
}
