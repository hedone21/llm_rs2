//! ENG-ST-052 ~ ENG-ST-060: Strategy + Conflict Resolution
//!
//! 4개 ResilienceStrategy (Memory, Thermal, Energy, Compute) 반응 검증 및
//! resolve_conflicts() 충돌 해소 규칙 검증.

use llm_rs2::resilience::strategy::{
    ComputeStrategy, EnergyStrategy, MemoryStrategy, ThermalStrategy,
};
use llm_rs2::resilience::{OperatingMode, ResilienceAction, ResilienceStrategy};
use llm_shared::{ComputeReason, EnergyReason, Level, RecommendedBackend, SystemSignal};

// ── 시그널 헬퍼 ──

fn mem_signal(level: Level) -> SystemSignal {
    SystemSignal::MemoryPressure {
        level,
        available_bytes: 1024 * 1024,
        total_bytes: 4 * 1024 * 1024,
        reclaim_target_bytes: 512 * 1024,
    }
}

fn thermal_signal(level: Level, throttle_ratio: f64) -> SystemSignal {
    SystemSignal::ThermalAlert {
        level,
        temperature_mc: 75000,
        throttling_active: throttle_ratio < 1.0,
        throttle_ratio,
    }
}

fn energy_signal(level: Level, reason: EnergyReason) -> SystemSignal {
    SystemSignal::EnergyConstraint {
        level,
        reason,
        power_budget_mw: 5000,
    }
}

fn compute_signal(level: Level, backend: RecommendedBackend) -> SystemSignal {
    SystemSignal::ComputeGuidance {
        level,
        recommended_backend: backend,
        reason: ComputeReason::Balanced,
        cpu_usage_pct: 50.0,
        gpu_usage_pct: 50.0,
    }
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-052: MemoryStrategy — 레벨별 반응
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_052_memory_normal_restores_defaults() {
    let mut strategy = MemoryStrategy::new();
    // Warning → Normal 전이를 거쳐야 RestoreDefaults가 반환됨
    let _ = strategy.react(&mem_signal(Level::Warning), OperatingMode::Degraded);
    let actions = strategy.react(&mem_signal(Level::Normal), OperatingMode::Normal);
    assert_eq!(actions.len(), 1);
    assert!(matches!(actions[0], ResilienceAction::RestoreDefaults));
}

#[test]
fn test_eng_st_052_memory_critical_triggers_eviction() {
    let mut strategy = MemoryStrategy::new();
    let actions = strategy.react(&mem_signal(Level::Critical), OperatingMode::Minimal);
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        ResilienceAction::Evict { target_ratio } => assert!(*target_ratio <= 0.50),
        _ => panic!("Expected Evict"),
    }
}

#[test]
fn test_eng_st_052_memory_emergency_evicts_and_rejects() {
    let mut strategy = MemoryStrategy::new();
    let actions = strategy.react(&mem_signal(Level::Emergency), OperatingMode::Suspended);
    assert_eq!(actions.len(), 2);
    assert!(matches!(actions[0], ResilienceAction::Evict { .. }));
    assert!(matches!(actions[1], ResilienceAction::RejectNew));
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-052: ThermalStrategy — 레벨별 반응
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_052_thermal_emergency_suspends() {
    let mut strategy = ThermalStrategy::new();
    let actions = strategy.react(
        &thermal_signal(Level::Emergency, 0.3),
        OperatingMode::Suspended,
    );
    assert_eq!(actions.len(), 1);
    assert!(matches!(actions[0], ResilienceAction::Suspend));
}

#[test]
fn test_eng_st_052_thermal_critical_throttles_proportionally() {
    let mut strategy = ThermalStrategy::new();
    // throttle_ratio=0.5 → delay = (1.0-0.5)*100 = 50ms
    let actions = strategy.react(
        &thermal_signal(Level::Critical, 0.5),
        OperatingMode::Minimal,
    );
    assert_eq!(actions.len(), 3);
    match &actions[1] {
        ResilienceAction::Throttle { delay_ms } => assert_eq!(*delay_ms, 50),
        _ => panic!("Expected Throttle"),
    }
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-052: EnergyStrategy — 레벨별 반응
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_052_energy_normal_restores() {
    let mut strategy = EnergyStrategy::new();
    let actions = strategy.react(
        &energy_signal(Level::Normal, EnergyReason::Charging),
        OperatingMode::Normal,
    );
    assert_eq!(actions.len(), 1);
    assert!(matches!(actions[0], ResilienceAction::RestoreDefaults));
}

#[test]
fn test_eng_st_052_energy_emergency_suspends_and_rejects() {
    let mut strategy = EnergyStrategy::new();
    let actions = strategy.react(
        &energy_signal(Level::Emergency, EnergyReason::BatteryCritical),
        OperatingMode::Suspended,
    );
    assert_eq!(actions.len(), 2);
    assert!(matches!(actions[0], ResilienceAction::Suspend));
    assert!(matches!(actions[1], ResilienceAction::RejectNew));
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-052: ComputeStrategy — 레벨별 반응
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_052_compute_critical_switches_backend() {
    let mut strategy = ComputeStrategy::new();
    let actions = strategy.react(
        &compute_signal(Level::Critical, RecommendedBackend::Cpu),
        OperatingMode::Minimal,
    );
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        ResilienceAction::SwitchBackend { to } => assert_eq!(*to, RecommendedBackend::Cpu),
        _ => panic!("Expected SwitchBackend"),
    }
}

#[test]
fn test_eng_st_052_compute_warning_does_not_switch() {
    let mut strategy = ComputeStrategy::new();
    let actions = strategy.react(
        &compute_signal(Level::Warning, RecommendedBackend::Cpu),
        OperatingMode::Degraded,
    );
    assert!(actions.is_empty());
}

// ══════════════════════════════════════════════════════════════
// ENG-ST-060: resolve_conflicts — 충돌 해소 규칙
// ══════════════════════════════════════════════════════════════

#[test]
fn test_eng_st_060_suspend_overrides_all() {
    let actions = vec![
        ResilienceAction::Evict { target_ratio: 0.50 },
        ResilienceAction::SwitchBackend {
            to: RecommendedBackend::Cpu,
        },
        ResilienceAction::Suspend,
        ResilienceAction::Throttle { delay_ms: 100 },
    ];
    let result = llm_rs2::resilience::strategy::resolve_conflicts(actions);
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0], ResilienceAction::Suspend));
}

#[test]
fn test_eng_st_060_restore_only_when_no_other_constraints() {
    let actions = vec![
        ResilienceAction::RestoreDefaults,
        ResilienceAction::Evict { target_ratio: 0.85 },
    ];
    let result = llm_rs2::resilience::strategy::resolve_conflicts(actions);
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0], ResilienceAction::Evict { .. }));
}

#[test]
fn test_eng_st_060_restore_alone_passes_through() {
    let actions = vec![ResilienceAction::RestoreDefaults];
    let result = llm_rs2::resilience::strategy::resolve_conflicts(actions);
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0], ResilienceAction::RestoreDefaults));
}

#[test]
fn test_eng_st_060_most_aggressive_eviction_wins() {
    let actions = vec![
        ResilienceAction::Evict { target_ratio: 0.85 },
        ResilienceAction::Evict { target_ratio: 0.50 },
        ResilienceAction::Evict { target_ratio: 0.75 },
    ];
    let result = llm_rs2::resilience::strategy::resolve_conflicts(actions);
    assert_eq!(result.len(), 1);
    match &result[0] {
        ResilienceAction::Evict { target_ratio } => {
            assert!((target_ratio - 0.50).abs() < f32::EPSILON);
        }
        _ => panic!("Expected Evict"),
    }
}

#[test]
fn test_eng_st_060_cpu_always_wins_over_gpu() {
    let actions = vec![
        ResilienceAction::SwitchBackend {
            to: RecommendedBackend::Gpu,
        },
        ResilienceAction::SwitchBackend {
            to: RecommendedBackend::Cpu,
        },
    ];
    let result = llm_rs2::resilience::strategy::resolve_conflicts(actions);
    assert_eq!(result.len(), 1);
    match &result[0] {
        ResilienceAction::SwitchBackend { to } => assert_eq!(*to, RecommendedBackend::Cpu),
        _ => panic!("Expected SwitchBackend"),
    }
}

#[test]
fn test_eng_st_060_largest_delay_wins() {
    let actions = vec![
        ResilienceAction::Throttle { delay_ms: 30 },
        ResilienceAction::Throttle { delay_ms: 100 },
        ResilienceAction::Throttle { delay_ms: 50 },
    ];
    let result = llm_rs2::resilience::strategy::resolve_conflicts(actions);
    assert_eq!(result.len(), 1);
    match &result[0] {
        ResilienceAction::Throttle { delay_ms } => assert_eq!(*delay_ms, 100),
        _ => panic!("Expected Throttle"),
    }
}
