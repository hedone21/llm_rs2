//! Phase 5 시나리오 기반 spec 테스트 — insta 스냅샷 중심.
//!
//! 각 테스트는:
//!   1. 시나리오 YAML 로드 + LuaPolicy 또는 MockPolicy 주입
//!   2. Simulator.run_for(30s)
//!   3. TrajectorySummary + relief 스냅샷을 insta로 고정
//!
//! 초기 실행: `INSTA_UPDATE=always cargo test -p llm_manager --test sim`
//! 스냅샷 검토: `cargo insta review` 또는 자동 accept

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use crate::common::sim::{
    config::load_scenario, harness::Simulator, trajectory::TrajectorySummary,
};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("sim")
}

fn scenarios_dir() -> PathBuf {
    fixtures_dir().join("scenarios")
}

fn lua_dir() -> PathBuf {
    fixtures_dir().join("lua")
}

/// ReliefTable 스냅샷을 안정적으로 직렬화하기 위해
/// BTreeMap<String, Vec<f32(3자리 반올림)>> 형태로 변환한다.
fn format_relief(
    relief: &std::collections::HashMap<String, [f32; 6]>,
) -> BTreeMap<String, Vec<f32>> {
    relief
        .iter()
        .map(|(k, vals)| {
            let rounded: Vec<f32> = vals
                .iter()
                .map(|&v| (v * 1000.0).round() / 1000.0)
                .collect();
            (k.clone(), rounded)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────
// 시나리오 1: memory_pressure_steady
// ─────────────────────────────────────────────────────────

/// 시나리오: decode 중 메모리 사용량 선형 상승 → Warning/Critical → Evict directive.
#[cfg(feature = "lua")]
#[test]
fn scenario_memory_pressure_steady() {
    use llm_manager::config::AdaptationConfig;

    let scenario_path = scenarios_dir().join("memory_pressure_steady.yaml");
    let lua_path = lua_dir().join("memory_evict_graduated.lua");

    let cfg = load_scenario(&scenario_path).unwrap_or_else(|e| panic!("시나리오 로드 실패: {e}"));

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut sim = Simulator::with_lua_policy(cfg, &lua_path, AdaptationConfig::default())
        .expect("Simulator::with_lua_policy 생성 실패");
    sim.run_for(Duration::from_secs(30)).expect("30s 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
        insta::assert_yaml_snapshot!("memory_pressure_summary", summary);
    });

    let relief = sim.policy.relief_snapshot().unwrap_or_default();
    if !relief.is_empty() {
        let formatted = format_relief(&relief);
        insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
            insta::assert_yaml_snapshot!("memory_pressure_relief", formatted);
        });
    }

    // 기본 검증: 30초 실행, signal 존재
    assert!(
        sim.trajectory().signal_count_by_kind("memory_pressure") >= 1,
        "memory_pressure signal이 기록되어야 함"
    );
}

/// lua feature 없을 때 MockPolicy로 memory_pressure_steady 기본 동작 확인.
#[cfg(not(feature = "lua"))]
#[test]
fn scenario_memory_pressure_steady() {
    use llm_shared::{EngineCommand, EngineDirective, Level, SystemSignal};

    use crate::common::sim::mock_policy::MockPolicy;

    let scenario_path = scenarios_dir().join("memory_pressure_steady.yaml");
    let cfg = load_scenario(&scenario_path).unwrap_or_else(|e| panic!("시나리오 로드 실패: {e}"));

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut mock = MockPolicy::new();
    mock.directive_on_signal = Some(Box::new(|sig| {
        if let SystemSignal::MemoryPressure { level, .. } = sig {
            if *level >= Level::Warning {
                return Some(EngineDirective {
                    seq_id: 100,
                    commands: vec![EngineCommand::KvEvictSliding { keep_ratio: 0.8 }],
                });
            }
        }
        None
    }));

    let mut sim = Simulator::new(cfg, Box::new(mock));
    sim.run_for(Duration::from_secs(30)).expect("30s 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
        insta::assert_yaml_snapshot!("memory_pressure_summary_mock", summary);
    });

    assert!(
        sim.trajectory().signal_count_by_kind("memory_pressure") >= 1,
        "memory_pressure signal이 기록되어야 함"
    );
}

// ─────────────────────────────────────────────────────────
// 시나리오 2: thermal_ramp_with_decode
// ─────────────────────────────────────────────────────────

/// 시나리오: decode + GPU 과열 → ThermalAlert → SwitchHw/Throttle directive.
#[cfg(feature = "lua")]
#[test]
fn scenario_thermal_ramp_with_decode() {
    use llm_manager::config::AdaptationConfig;

    let scenario_path = scenarios_dir().join("thermal_ramp_with_decode.yaml");
    let lua_path = lua_dir().join("thermal_switch_backend.lua");

    let cfg = load_scenario(&scenario_path).unwrap_or_else(|e| panic!("시나리오 로드 실패: {e}"));

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut sim = Simulator::with_lua_policy(cfg, &lua_path, AdaptationConfig::default())
        .expect("Simulator::with_lua_policy 생성 실패");
    sim.run_for(Duration::from_secs(30)).expect("30s 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
        insta::assert_yaml_snapshot!("thermal_ramp_summary", summary);
    });

    let relief = sim.policy.relief_snapshot().unwrap_or_default();
    if !relief.is_empty() {
        let formatted = format_relief(&relief);
        insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
            insta::assert_yaml_snapshot!("thermal_ramp_relief", formatted);
        });
    }

    // thermal signal이 기록되어야 함
    assert!(
        sim.trajectory().signal_count_by_kind("thermal_alert") >= 1,
        "thermal_alert signal이 기록되어야 함"
    );
}

/// lua feature 없을 때 MockPolicy로 thermal_ramp 기본 동작 확인.
#[cfg(not(feature = "lua"))]
#[test]
fn scenario_thermal_ramp_with_decode() {
    use llm_shared::{EngineCommand, EngineDirective, Level, SystemSignal};

    use crate::common::sim::mock_policy::MockPolicy;

    let scenario_path = scenarios_dir().join("thermal_ramp_with_decode.yaml");
    let cfg = load_scenario(&scenario_path).unwrap_or_else(|e| panic!("시나리오 로드 실패: {e}"));

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut mock = MockPolicy::new();
    mock.directive_on_signal = Some(Box::new(|sig| {
        if let SystemSignal::ThermalAlert { level, .. } = sig {
            if *level >= Level::Warning {
                return Some(EngineDirective {
                    seq_id: 101,
                    commands: vec![EngineCommand::Throttle { delay_ms: 100 }],
                });
            }
        }
        None
    }));

    let mut sim = Simulator::new(cfg, Box::new(mock));
    sim.run_for(Duration::from_secs(30)).expect("30s 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
        insta::assert_yaml_snapshot!("thermal_ramp_summary_mock", summary);
    });

    assert!(
        sim.trajectory().signal_count_by_kind("thermal_alert") >= 1,
        "thermal_alert signal이 기록되어야 함"
    );
}

// ─────────────────────────────────────────────────────────
// 시나리오 3: partition_contention
// ─────────────────────────────────────────────────────────

/// 시나리오: partition_ratio=0.5 decode + BW 경합 → ComputeGuidance → SetPartitionRatio.
#[cfg(feature = "lua")]
#[test]
fn scenario_partition_contention() {
    use llm_manager::config::AdaptationConfig;

    let scenario_path = scenarios_dir().join("partition_contention.yaml");
    let lua_path = lua_dir().join("partition_adaptive.lua");

    let cfg = load_scenario(&scenario_path).unwrap_or_else(|e| panic!("시나리오 로드 실패: {e}"));

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut sim = Simulator::with_lua_policy(cfg, &lua_path, AdaptationConfig::default())
        .expect("Simulator::with_lua_policy 생성 실패");
    sim.run_for(Duration::from_secs(30)).expect("30s 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
        insta::assert_yaml_snapshot!("partition_contention_summary", summary);
    });

    let relief = sim.policy.relief_snapshot().unwrap_or_default();
    if !relief.is_empty() {
        let formatted = format_relief(&relief);
        insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
            insta::assert_yaml_snapshot!("partition_contention_relief", formatted);
        });
    }

    // compute signal이 기록되어야 함
    assert!(
        sim.trajectory().signal_count_by_kind("compute_guidance") >= 1,
        "compute_guidance signal이 기록되어야 함"
    );
}

/// lua feature 없을 때 MockPolicy로 partition_contention 기본 동작 확인.
#[cfg(not(feature = "lua"))]
#[test]
fn scenario_partition_contention() {
    use llm_shared::{EngineCommand, EngineDirective, Level, SystemSignal};

    use crate::common::sim::mock_policy::MockPolicy;

    let scenario_path = scenarios_dir().join("partition_contention.yaml");
    let cfg = load_scenario(&scenario_path).unwrap_or_else(|e| panic!("시나리오 로드 실패: {e}"));

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut mock = MockPolicy::new();
    mock.directive_on_signal = Some(Box::new(|sig| {
        if let SystemSignal::ComputeGuidance { level, .. } = sig {
            if *level >= Level::Warning {
                return Some(EngineDirective {
                    seq_id: 102,
                    commands: vec![EngineCommand::SetPartitionRatio { ratio: 0.0 }],
                });
            }
        }
        None
    }));

    let mut sim = Simulator::new(cfg, Box::new(mock));
    sim.run_for(Duration::from_secs(30)).expect("30s 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
        insta::assert_yaml_snapshot!("partition_contention_summary_mock", summary);
    });

    assert!(
        sim.trajectory().signal_count_by_kind("compute_guidance") >= 1,
        "compute_guidance signal이 기록되어야 함"
    );
}

// ─────────────────────────────────────────────────────────
// 시나리오 4: memory + thermal 복합 신호
// ─────────────────────────────────────────────────────────

/// 시나리오: memory + thermal 두 신호 동시 발생 → composition 처리 검증.
/// baseline 시나리오에 높은 초기값으로 복합 압력 유도.
#[cfg(feature = "lua")]
#[test]
fn scenario_memory_and_thermal_combined() {
    use llm_manager::config::AdaptationConfig;

    let baseline_path = fixtures_dir().join("baseline.yaml");
    let lua_path = lua_dir().join("memory_and_thermal_combined.lua");

    let mut cfg =
        load_scenario(&baseline_path).unwrap_or_else(|e| panic!("baseline 로드 실패: {e}"));

    // 복합 압력 초기값 설정
    cfg.initial_state.device_memory_used_mb = 7000;
    cfg.initial_state.gpu_cluster_thermal_c = 69.0;
    cfg.initial_state.phase = "decode".to_string();
    cfg.rng_seed = Some(42);

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut sim = Simulator::with_lua_policy(cfg, &lua_path, AdaptationConfig::default())
        .expect("Simulator::with_lua_policy 생성 실패");
    sim.run_for(Duration::from_secs(20)).expect("20s 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
        insta::assert_yaml_snapshot!("combined_signals_summary", summary);
    });

    let relief = sim.policy.relief_snapshot().unwrap_or_default();
    if !relief.is_empty() {
        let formatted = format_relief(&relief);
        insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
            insta::assert_yaml_snapshot!("combined_signals_relief", formatted);
        });
    }

    // 두 신호 종류가 모두 기록되어야 함
    assert!(
        sim.trajectory().signal_count_by_kind("memory_pressure") >= 1,
        "memory_pressure signal이 기록되어야 함"
    );
    assert!(
        sim.trajectory().signal_count_by_kind("thermal_alert") >= 1,
        "thermal_alert signal이 기록되어야 함"
    );
}

/// lua feature 없을 때 MockPolicy로 복합 신호 기본 동작 확인.
#[cfg(not(feature = "lua"))]
#[test]
fn scenario_memory_and_thermal_combined() {
    use llm_shared::{EngineCommand, EngineDirective, Level, SystemSignal};

    use crate::common::sim::mock_policy::MockPolicy;

    let baseline_path = fixtures_dir().join("baseline.yaml");
    let mut cfg =
        load_scenario(&baseline_path).unwrap_or_else(|e| panic!("baseline 로드 실패: {e}"));

    cfg.initial_state.device_memory_used_mb = 7000;
    cfg.initial_state.gpu_cluster_thermal_c = 69.0;
    cfg.initial_state.phase = "decode".to_string();
    cfg.rng_seed = Some(42);

    let cpu_max = cfg.initial_state.cpu_max_freq_mhz as f64;
    let gpu_max = cfg.initial_state.gpu_max_freq_mhz as f64;

    let mut mock = MockPolicy::new();
    mock.directive_on_signal = Some(Box::new(|sig| match sig {
        SystemSignal::MemoryPressure { level, .. } if *level >= Level::Warning => {
            Some(EngineDirective {
                seq_id: 103,
                commands: vec![EngineCommand::KvEvictSliding { keep_ratio: 0.7 }],
            })
        }
        SystemSignal::ThermalAlert { level, .. } if *level >= Level::Warning => {
            Some(EngineDirective {
                seq_id: 104,
                commands: vec![EngineCommand::Throttle { delay_ms: 100 }],
            })
        }
        _ => None,
    }));

    let mut sim = Simulator::new(cfg, Box::new(mock));
    sim.run_for(Duration::from_secs(20)).expect("20s 실행 실패");

    let summary = TrajectorySummary::from_trajectory(sim.trajectory(), cpu_max, gpu_max);

    insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
        insta::assert_yaml_snapshot!("combined_signals_summary_mock", summary);
    });

    // 두 신호 모두 기록
    assert!(
        sim.trajectory().signal_count_by_kind("memory_pressure") >= 1,
        "memory_pressure signal이 기록되어야 함"
    );
    assert!(
        sim.trajectory().signal_count_by_kind("thermal_alert") >= 1,
        "thermal_alert signal이 기록되어야 함"
    );
}

// ─────────────────────────────────────────────────────────
// relief_snapshot 비공허 검증 (PR 2+3 핵심 회귀 테스트)
// ─────────────────────────────────────────────────────────

/// VirtualClockHandle이 LuaPolicy에 주입되어 30s 시뮬에서 relief가 실제로 학습되는지 검증.
/// 이전에는 wall-clock 기반 observation이 ~100ms 안에 끝나는 harness에서 3s delay를
/// 충족하지 못해 relief_snapshot이 항상 공허했다.
#[cfg(feature = "lua")]
#[test]
fn scenario_partition_contention_produces_non_empty_relief() {
    use llm_manager::config::AdaptationConfig;

    let scenario_path = scenarios_dir().join("partition_contention.yaml");
    let lua_path = lua_dir().join("partition_adaptive.lua");

    let cfg = load_scenario(&scenario_path).unwrap_or_else(|e| panic!("시나리오 로드 실패: {e}"));

    let mut sim = Simulator::with_lua_policy(cfg, &lua_path, AdaptationConfig::default())
        .expect("Simulator::with_lua_policy 생성 실패");
    sim.run_for(Duration::from_secs(30)).expect("30s 실행 실패");

    let relief = sim
        .policy
        .relief_snapshot()
        .expect("LuaPolicy는 Some을 반환해야 함");
    assert!(
        !relief.is_empty(),
        "30s 시뮬 후 relief_snapshot이 비어있지 않아야 함 (VirtualClockHandle이 3s 관측 지연을 충족해야 함)"
    );
}
