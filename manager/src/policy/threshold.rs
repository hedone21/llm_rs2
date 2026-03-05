use crate::collector::{Reading, ReadingData};
use crate::config::Config;
use crate::policy::PolicyEngine;
use llm_shared::{ComputeReason, EnergyReason, Level, RecommendedBackend, SystemSignal};

/// Hysteresis-based threshold policy engine.
///
/// Evaluates readings against configurable thresholds with hysteresis
/// gaps to prevent level oscillation near boundaries.
///
/// Escalation (worsening) is immediate and can skip levels.
/// Recovery (improving) requires the value to cross the recovery
/// threshold (= escalation threshold +/- hysteresis gap).
pub struct ThresholdPolicy {
    config: Config,
    memory: LevelState,
    thermal: LevelState,
    compute: ComputeState,
    energy: LevelState,
    last_memory: Option<MemorySnapshot>,
    last_thermal: Option<ThermalSnapshot>,
    last_compute: Option<ComputeSnapshot>,
    last_energy: Option<EnergySnapshot>,
}

struct LevelState {
    level: Level,
}

struct ComputeState {
    level: Level,
    recommended: RecommendedBackend,
    reason: ComputeReason,
}

#[derive(Clone)]
struct MemorySnapshot {
    available_bytes: u64,
    total_bytes: u64,
}

#[derive(Clone)]
struct ThermalSnapshot {
    temperature_mc: i32,
    throttling_active: bool,
}

#[derive(Clone)]
struct ComputeSnapshot {
    cpu_usage_pct: f64,
    gpu_usage_pct: f64,
}

#[derive(Clone)]
struct EnergySnapshot {
    charging: bool,
}

// ---------------------------------------------------------------------------
// Hysteresis evaluation functions
// ---------------------------------------------------------------------------

/// Evaluate level for "higher is worse" metrics (temperature, CPU/GPU usage).
///
/// - Escalation: immediate jump to highest triggered level
/// - Recovery: requires crossing recovery threshold (= threshold - hysteresis)
fn level_ascending(
    value: f64,
    current: Level,
    warning_up: f64,
    critical_up: f64,
    emergency_up: f64,
    hysteresis: f64,
) -> Level {
    let warning_down = warning_up - hysteresis;
    let critical_down = critical_up - hysteresis;
    let emergency_down = emergency_up - hysteresis;

    // Escalation: jump to highest triggered level
    if value >= emergency_up && current < Level::Emergency {
        return Level::Emergency;
    }
    if value >= critical_up && current < Level::Critical {
        return Level::Critical;
    }
    if value >= warning_up && current < Level::Warning {
        return Level::Warning;
    }

    // Recovery: drop when value convincingly crosses recovery threshold
    match current {
        Level::Emergency if value < emergency_down => {
            if value < warning_down {
                Level::Normal
            } else if value < critical_down {
                Level::Warning
            } else {
                Level::Critical
            }
        }
        Level::Critical if value < critical_down => {
            if value < warning_down {
                Level::Normal
            } else {
                Level::Warning
            }
        }
        Level::Warning if value < warning_down => Level::Normal,
        _ => current,
    }
}

/// Evaluate level for "lower is worse" metrics (available memory, battery %).
///
/// - Escalation: value drops below threshold
/// - Recovery: value rises above threshold + hysteresis
fn level_descending(
    value: f64,
    current: Level,
    warning_below: f64,
    critical_below: f64,
    emergency_below: f64,
    hysteresis: f64,
) -> Level {
    let warning_up = warning_below + hysteresis;
    let critical_up = critical_below + hysteresis;
    let emergency_up = emergency_below + hysteresis;

    // Escalation: value drops below thresholds
    if value <= emergency_below && current < Level::Emergency {
        return Level::Emergency;
    }
    if value <= critical_below && current < Level::Critical {
        return Level::Critical;
    }
    if value <= warning_below && current < Level::Warning {
        return Level::Warning;
    }

    // Recovery: value rises above recovery thresholds
    match current {
        Level::Emergency if value > emergency_up => {
            if value > warning_up {
                Level::Normal
            } else if value > critical_up {
                Level::Warning
            } else {
                Level::Critical
            }
        }
        Level::Critical if value > critical_up => {
            if value > warning_up {
                Level::Normal
            } else {
                Level::Warning
            }
        }
        Level::Warning if value > warning_up => Level::Normal,
        _ => current,
    }
}

// ---------------------------------------------------------------------------
// Compute recommendation helper
// ---------------------------------------------------------------------------

fn compute_recommendation(
    cpu: f64,
    gpu: f64,
    warning_pct: f64,
) -> (RecommendedBackend, ComputeReason) {
    let cpu_hot = cpu >= warning_pct;
    let gpu_hot = gpu >= warning_pct;

    match (cpu_hot, gpu_hot) {
        (true, true) => (RecommendedBackend::Any, ComputeReason::BothLoaded),
        (true, false) => (RecommendedBackend::Gpu, ComputeReason::CpuBottleneck),
        (false, true) => (RecommendedBackend::Cpu, ComputeReason::GpuBottleneck),
        (false, false) => {
            if (cpu - gpu).abs() < 10.0 {
                (RecommendedBackend::Any, ComputeReason::Balanced)
            } else if cpu < gpu {
                (RecommendedBackend::Cpu, ComputeReason::CpuAvailable)
            } else {
                (RecommendedBackend::Gpu, ComputeReason::GpuAvailable)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ThresholdPolicy implementation
// ---------------------------------------------------------------------------

impl ThresholdPolicy {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            memory: LevelState {
                level: Level::Normal,
            },
            thermal: LevelState {
                level: Level::Normal,
            },
            compute: ComputeState {
                level: Level::Normal,
                recommended: RecommendedBackend::Any,
                reason: ComputeReason::Balanced,
            },
            energy: LevelState {
                level: Level::Normal,
            },
            last_memory: None,
            last_thermal: None,
            last_compute: None,
            last_energy: None,
        }
    }

    fn evaluate_memory(&mut self, available_bytes: u64, total_bytes: u64) -> Option<SystemSignal> {
        let pct = if total_bytes == 0 {
            100.0
        } else {
            (available_bytes as f64 / total_bytes as f64) * 100.0
        };
        let cfg = &self.config.memory;

        let new_level = level_descending(
            pct,
            self.memory.level,
            cfg.warning_available_pct,
            cfg.critical_available_pct,
            cfg.emergency_available_pct,
            cfg.hysteresis_pct,
        );

        self.last_memory = Some(MemorySnapshot {
            available_bytes,
            total_bytes,
        });

        if new_level != self.memory.level {
            self.memory.level = new_level;
            Some(self.build_memory_signal())
        } else {
            None
        }
    }

    fn evaluate_thermal(
        &mut self,
        temperature_mc: i32,
        throttling_active: bool,
    ) -> Option<SystemSignal> {
        let cfg = &self.config.thermal;

        let new_level = level_ascending(
            temperature_mc as f64,
            self.thermal.level,
            cfg.warning_temp_mc as f64,
            cfg.critical_temp_mc as f64,
            cfg.emergency_temp_mc as f64,
            cfg.hysteresis_mc as f64,
        );

        self.last_thermal = Some(ThermalSnapshot {
            temperature_mc,
            throttling_active,
        });

        if new_level != self.thermal.level {
            self.thermal.level = new_level;
            Some(self.build_thermal_signal())
        } else {
            None
        }
    }

    fn evaluate_compute(&mut self, cpu: f64, gpu: f64) -> Option<SystemSignal> {
        let cfg = &self.config.compute;
        let worst = cpu.max(gpu);

        // ComputeGuidance has no Emergency level — use f64::MAX as unreachable threshold
        let new_level = level_ascending(
            worst,
            self.compute.level,
            cfg.warning_usage_pct,
            cfg.critical_usage_pct,
            f64::MAX,
            cfg.hysteresis_pct,
        );

        let (new_recommended, new_reason) = compute_recommendation(cpu, gpu, cfg.warning_usage_pct);

        self.last_compute = Some(ComputeSnapshot {
            cpu_usage_pct: cpu,
            gpu_usage_pct: gpu,
        });

        let level_changed = new_level != self.compute.level;
        let rec_changed = new_recommended != self.compute.recommended;

        if level_changed || rec_changed {
            self.compute.level = new_level;
            self.compute.recommended = new_recommended;
            self.compute.reason = new_reason;
            Some(self.build_compute_signal())
        } else {
            None
        }
    }

    fn evaluate_energy(
        &mut self,
        battery_pct: Option<f64>,
        charging: bool,
    ) -> Option<SystemSignal> {
        let cfg = &self.config.energy;

        // If charging and configured to ignore, force Normal
        let new_level = if charging && cfg.ignore_when_charging {
            Level::Normal
        } else if let Some(pct) = battery_pct {
            level_descending(
                pct,
                self.energy.level,
                cfg.warning_battery_pct,
                cfg.critical_battery_pct,
                cfg.emergency_battery_pct,
                // Energy uses a fixed 2% hysteresis for battery
                2.0,
            )
        } else {
            // No battery info — stay Normal
            Level::Normal
        };

        self.last_energy = Some(EnergySnapshot { charging });

        if new_level != self.energy.level {
            self.energy.level = new_level;
            Some(self.build_energy_signal())
        } else {
            None
        }
    }

    // --- Signal builders ---

    fn build_memory_signal(&self) -> SystemSignal {
        let snap = self
            .last_memory
            .as_ref()
            .cloned()
            .unwrap_or(MemorySnapshot {
                available_bytes: 0,
                total_bytes: 0,
            });
        let reclaim = match self.memory.level {
            Level::Normal => 0,
            Level::Warning => (snap.total_bytes as f64 * 0.05) as u64,
            Level::Critical => (snap.total_bytes as f64 * 0.10) as u64,
            Level::Emergency => (snap.total_bytes as f64 * 0.20) as u64,
        };
        SystemSignal::MemoryPressure {
            level: self.memory.level,
            available_bytes: snap.available_bytes,
            reclaim_target_bytes: reclaim,
        }
    }

    fn build_thermal_signal(&self) -> SystemSignal {
        let snap = self
            .last_thermal
            .as_ref()
            .cloned()
            .unwrap_or(ThermalSnapshot {
                temperature_mc: 25000,
                throttling_active: false,
            });
        let throttle_ratio = match self.thermal.level {
            Level::Normal => 1.0,
            Level::Warning => 1.0,
            Level::Critical => 0.7,
            Level::Emergency => 0.3,
        };
        SystemSignal::ThermalAlert {
            level: self.thermal.level,
            temperature_mc: snap.temperature_mc,
            throttling_active: snap.throttling_active,
            throttle_ratio,
        }
    }

    fn build_compute_signal(&self) -> SystemSignal {
        let snap = self
            .last_compute
            .as_ref()
            .cloned()
            .unwrap_or(ComputeSnapshot {
                cpu_usage_pct: 0.0,
                gpu_usage_pct: 0.0,
            });
        SystemSignal::ComputeGuidance {
            level: self.compute.level,
            recommended_backend: self.compute.recommended,
            reason: self.compute.reason,
            cpu_usage_pct: snap.cpu_usage_pct,
            gpu_usage_pct: snap.gpu_usage_pct,
        }
    }

    fn build_energy_signal(&self) -> SystemSignal {
        let snap = self
            .last_energy
            .as_ref()
            .cloned()
            .unwrap_or(EnergySnapshot { charging: false });
        let cfg = &self.config.energy;
        let (reason, budget) = match self.energy.level {
            Level::Normal => {
                if snap.charging {
                    (EnergyReason::Charging, 0)
                } else {
                    (EnergyReason::None, 0)
                }
            }
            Level::Warning => (EnergyReason::BatteryLow, cfg.warning_power_budget_mw),
            Level::Critical => (EnergyReason::BatteryCritical, cfg.critical_power_budget_mw),
            Level::Emergency => (EnergyReason::BatteryCritical, cfg.emergency_power_budget_mw),
        };
        SystemSignal::EnergyConstraint {
            level: self.energy.level,
            reason,
            power_budget_mw: budget,
        }
    }
}

impl PolicyEngine for ThresholdPolicy {
    fn process(&mut self, reading: &Reading) -> Vec<SystemSignal> {
        let mut signals = Vec::new();

        match &reading.data {
            ReadingData::Memory {
                available_bytes,
                total_bytes,
                ..
            } => {
                if let Some(sig) = self.evaluate_memory(*available_bytes, *total_bytes) {
                    signals.push(sig);
                }
            }
            ReadingData::Thermal {
                temperature_mc,
                throttling_active,
            } => {
                if let Some(sig) = self.evaluate_thermal(*temperature_mc, *throttling_active) {
                    signals.push(sig);
                }
            }
            ReadingData::Compute {
                cpu_usage_pct,
                gpu_usage_pct,
            } => {
                if let Some(sig) = self.evaluate_compute(*cpu_usage_pct, *gpu_usage_pct) {
                    signals.push(sig);
                }
            }
            ReadingData::Energy {
                battery_pct,
                charging,
                ..
            } => {
                if let Some(sig) = self.evaluate_energy(*battery_pct, *charging) {
                    signals.push(sig);
                }
            }
        }

        signals
    }

    fn current_signals(&self) -> Vec<SystemSignal> {
        vec![
            self.build_memory_signal(),
            self.build_thermal_signal(),
            self.build_compute_signal(),
            self.build_energy_signal(),
        ]
    }

    fn name(&self) -> &str {
        "ThresholdPolicy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn default_policy() -> ThresholdPolicy {
        ThresholdPolicy::new(Config::default())
    }

    fn reading(data: ReadingData) -> Reading {
        Reading {
            timestamp: Instant::now(),
            data,
        }
    }

    fn mem_reading(available_bytes: u64, total_bytes: u64) -> Reading {
        reading(ReadingData::Memory {
            available_bytes,
            total_bytes,
            psi_some_avg10: None,
        })
    }

    fn thermal_reading(temp_mc: i32, throttling: bool) -> Reading {
        reading(ReadingData::Thermal {
            temperature_mc: temp_mc,
            throttling_active: throttling,
        })
    }

    fn compute_reading(cpu: f64, gpu: f64) -> Reading {
        reading(ReadingData::Compute {
            cpu_usage_pct: cpu,
            gpu_usage_pct: gpu,
        })
    }

    fn energy_reading(battery: Option<f64>, charging: bool) -> Reading {
        reading(ReadingData::Energy {
            battery_pct: battery,
            charging,
            power_draw_mw: None,
        })
    }

    // --- Memory tests ---

    #[test]
    fn memory_escalation_path() {
        let mut p = default_policy();
        // Default: warning=40%, critical=20%, emergency=10%, hysteresis=5%
        // Total 1GB, so thresholds in bytes:
        // warning: 400MB, critical: 200MB, emergency: 100MB
        let total = 1_000_000_000u64;

        // 50% available → Normal (no signal, already Normal)
        let sigs = p.process(&mem_reading(500_000_000, total));
        assert!(sigs.is_empty());

        // 35% → Warning
        let sigs = p.process(&mem_reading(350_000_000, total));
        assert_eq!(sigs.len(), 1);
        assert_eq!(sigs[0].level(), Level::Warning);

        // 15% → Critical
        let sigs = p.process(&mem_reading(150_000_000, total));
        assert_eq!(sigs.len(), 1);
        assert_eq!(sigs[0].level(), Level::Critical);

        // 5% → Emergency
        let sigs = p.process(&mem_reading(50_000_000, total));
        assert_eq!(sigs.len(), 1);
        assert_eq!(sigs[0].level(), Level::Emergency);
    }

    #[test]
    fn memory_skip_to_emergency() {
        let mut p = default_policy();
        let total = 1_000_000_000u64;

        // Jump directly from Normal to Emergency
        let sigs = p.process(&mem_reading(50_000_000, total));
        assert_eq!(sigs.len(), 1);
        assert_eq!(sigs[0].level(), Level::Emergency);
    }

    #[test]
    fn memory_hysteresis_prevents_oscillation() {
        let mut p = default_policy();
        // warning=40%, hysteresis=5%, so recovery requires > 45%
        let total = 1_000_000_000u64;

        // Drop to 35% → Warning
        p.process(&mem_reading(350_000_000, total));
        assert_eq!(p.memory.level, Level::Warning);

        // Rise to 42% — still in hysteresis zone (need > 45%)
        let sigs = p.process(&mem_reading(420_000_000, total));
        assert!(sigs.is_empty());
        assert_eq!(p.memory.level, Level::Warning);

        // Rise to 46% — above recovery threshold
        let sigs = p.process(&mem_reading(460_000_000, total));
        assert_eq!(sigs.len(), 1);
        assert_eq!(sigs[0].level(), Level::Normal);
    }

    #[test]
    fn memory_recovery_from_emergency() {
        let mut p = default_policy();
        // emergency=10%, critical=20%, warning=40%, hysteresis=5%
        // Recovery thresholds: emergency→ >15%, critical→ >25%, warning→ >45%
        let total = 1_000_000_000u64;

        // Go to Emergency
        p.process(&mem_reading(50_000_000, total));
        assert_eq!(p.memory.level, Level::Emergency);

        // 12% — still in Emergency (need > 15%)
        let sigs = p.process(&mem_reading(120_000_000, total));
        assert!(sigs.is_empty());

        // 16% — recovers from Emergency, lands at Critical (16% < 25%)
        let sigs = p.process(&mem_reading(160_000_000, total));
        assert_eq!(sigs[0].level(), Level::Critical);

        // 30% — recovers from Critical (> 25%), lands at Warning (30% < 45%)
        let sigs = p.process(&mem_reading(300_000_000, total));
        assert_eq!(sigs[0].level(), Level::Warning);

        // 50% — recovers from Warning (> 45%), lands at Normal
        let sigs = p.process(&mem_reading(500_000_000, total));
        assert_eq!(sigs[0].level(), Level::Normal);
    }

    #[test]
    fn memory_full_recovery_jump() {
        let mut p = default_policy();
        let total = 1_000_000_000u64;

        // Go to Emergency
        p.process(&mem_reading(50_000_000, total));
        assert_eq!(p.memory.level, Level::Emergency);

        // Jump to 80% — well above all recovery thresholds → Normal
        let sigs = p.process(&mem_reading(800_000_000, total));
        assert_eq!(sigs[0].level(), Level::Normal);
    }

    #[test]
    fn memory_reclaim_target_scales_with_level() {
        let mut p = default_policy();
        let total = 1_000_000_000u64;

        // Warning → 5% of total
        p.process(&mem_reading(350_000_000, total));
        if let SystemSignal::MemoryPressure {
            reclaim_target_bytes,
            ..
        } = p.build_memory_signal()
        {
            assert_eq!(reclaim_target_bytes, 50_000_000); // 5% of 1GB
        }
    }

    // --- Thermal tests ---

    #[test]
    fn thermal_escalation_path() {
        let mut p = default_policy();
        // warning=60000, critical=75000, emergency=85000, hysteresis=5000

        // 50°C → Normal (no change)
        let sigs = p.process(&thermal_reading(50000, false));
        assert!(sigs.is_empty());

        // 65°C → Warning
        let sigs = p.process(&thermal_reading(65000, false));
        assert_eq!(sigs[0].level(), Level::Warning);

        // 76°C → Critical
        let sigs = p.process(&thermal_reading(76000, true));
        assert_eq!(sigs[0].level(), Level::Critical);

        // 90°C → Emergency
        let sigs = p.process(&thermal_reading(90000, true));
        assert_eq!(sigs[0].level(), Level::Emergency);
    }

    #[test]
    fn thermal_hysteresis() {
        let mut p = default_policy();
        // warning=60000, hysteresis=5000, so recovery < 55000

        // 65°C → Warning
        p.process(&thermal_reading(65000, false));
        assert_eq!(p.thermal.level, Level::Warning);

        // 58°C — still Warning (need < 55000)
        let sigs = p.process(&thermal_reading(58000, false));
        assert!(sigs.is_empty());

        // 54°C → Normal
        let sigs = p.process(&thermal_reading(54000, false));
        assert_eq!(sigs[0].level(), Level::Normal);
    }

    #[test]
    fn thermal_throttle_ratio() {
        let mut p = default_policy();

        p.process(&thermal_reading(76000, true));
        if let SystemSignal::ThermalAlert { throttle_ratio, .. } = p.build_thermal_signal() {
            assert!((throttle_ratio - 0.7).abs() < 0.01);
        }
    }

    // --- Compute tests ---

    #[test]
    fn compute_cpu_bottleneck() {
        let mut p = default_policy();
        // warning=70%, critical=90%

        // CPU 85%, GPU 30% → Warning, recommend GPU
        let sigs = p.process(&compute_reading(85.0, 30.0));
        assert_eq!(sigs.len(), 1);
        assert_eq!(sigs[0].level(), Level::Warning);
        if let SystemSignal::ComputeGuidance {
            recommended_backend,
            reason,
            ..
        } = &sigs[0]
        {
            assert_eq!(*recommended_backend, RecommendedBackend::Gpu);
            assert_eq!(*reason, ComputeReason::CpuBottleneck);
        }
    }

    #[test]
    fn compute_both_loaded() {
        let mut p = default_policy();

        // Both at 92% → Critical, both loaded
        let sigs = p.process(&compute_reading(92.0, 92.0));
        assert_eq!(sigs[0].level(), Level::Critical);
        if let SystemSignal::ComputeGuidance {
            recommended_backend,
            reason,
            ..
        } = &sigs[0]
        {
            assert_eq!(*recommended_backend, RecommendedBackend::Any);
            assert_eq!(*reason, ComputeReason::BothLoaded);
        }
    }

    #[test]
    fn compute_no_emergency_level() {
        let mut p = default_policy();

        // Even at 99% — max is Critical, not Emergency
        let sigs = p.process(&compute_reading(99.0, 99.0));
        assert_eq!(sigs[0].level(), Level::Critical);
    }

    #[test]
    fn compute_recommendation_change_triggers_signal() {
        let mut p = default_policy();

        // CPU hot → recommend GPU
        p.process(&compute_reading(85.0, 30.0));
        assert_eq!(p.compute.recommended, RecommendedBackend::Gpu);

        // GPU also hot → both loaded (level stays Warning, recommendation changes)
        let sigs = p.process(&compute_reading(85.0, 80.0));
        assert_eq!(sigs.len(), 1);
        if let SystemSignal::ComputeGuidance {
            recommended_backend,
            ..
        } = &sigs[0]
        {
            assert_eq!(*recommended_backend, RecommendedBackend::Any);
        }
    }

    #[test]
    fn compute_balanced_state() {
        let mut p = default_policy();

        // First push to Warning so we can detect a signal back to Normal
        p.process(&compute_reading(85.0, 30.0));
        assert_eq!(p.compute.level, Level::Warning);

        // Both low and similar → Normal, balanced
        let sigs = p.process(&compute_reading(30.0, 32.0));
        assert_eq!(sigs[0].level(), Level::Normal);
        if let SystemSignal::ComputeGuidance {
            recommended_backend,
            reason,
            ..
        } = &sigs[0]
        {
            assert_eq!(*recommended_backend, RecommendedBackend::Any);
            assert_eq!(*reason, ComputeReason::Balanced);
        }
    }

    // --- Energy tests ---

    #[test]
    fn energy_battery_depletion() {
        let mut p = default_policy();
        // warning=30%, critical=15%, emergency=5%

        // 50% → Normal
        let sigs = p.process(&energy_reading(Some(50.0), false));
        assert!(sigs.is_empty());

        // 25% → Warning
        let sigs = p.process(&energy_reading(Some(25.0), false));
        assert_eq!(sigs[0].level(), Level::Warning);
        if let SystemSignal::EnergyConstraint { reason, .. } = &sigs[0] {
            assert_eq!(*reason, EnergyReason::BatteryLow);
        }

        // 10% → Critical
        let sigs = p.process(&energy_reading(Some(10.0), false));
        assert_eq!(sigs[0].level(), Level::Critical);

        // 3% → Emergency
        let sigs = p.process(&energy_reading(Some(3.0), false));
        assert_eq!(sigs[0].level(), Level::Emergency);
    }

    #[test]
    fn energy_charging_overrides() {
        let mut p = default_policy();

        // 10% but charging → Normal (ignore_when_charging = true)
        let sigs = p.process(&energy_reading(Some(10.0), true));
        assert!(sigs.is_empty()); // already Normal

        // Go to Warning first
        p.process(&energy_reading(Some(25.0), false));
        assert_eq!(p.energy.level, Level::Warning);

        // Start charging → Normal
        let sigs = p.process(&energy_reading(Some(25.0), true));
        assert_eq!(sigs[0].level(), Level::Normal);
        if let SystemSignal::EnergyConstraint { reason, .. } = &sigs[0] {
            assert_eq!(*reason, EnergyReason::Charging);
        }
    }

    #[test]
    fn energy_no_battery() {
        let mut p = default_policy();

        // No battery info → stays Normal
        let sigs = p.process(&energy_reading(None, false));
        assert!(sigs.is_empty());
    }

    // --- General tests ---

    #[test]
    fn current_signals_returns_all_four() {
        let p = default_policy();
        let sigs = p.current_signals();
        assert_eq!(sigs.len(), 4);

        // All should be Normal initially
        for sig in &sigs {
            assert_eq!(sig.level(), Level::Normal);
        }
    }

    #[test]
    fn no_signal_when_level_unchanged() {
        let mut p = default_policy();
        let total = 1_000_000_000u64;

        // Multiple readings at 50% → no signals (stays Normal)
        for _ in 0..5 {
            let sigs = p.process(&mem_reading(500_000_000, total));
            assert!(sigs.is_empty());
        }
    }

    #[test]
    fn name_returns_expected() {
        let p = default_policy();
        assert_eq!(p.name(), "ThresholdPolicy");
    }

    // --- Hysteresis function unit tests ---

    #[test]
    fn ascending_skip_levels_on_escalation() {
        // Jump from Normal to Emergency
        let level = level_ascending(90.0, Level::Normal, 60.0, 75.0, 85.0, 5.0);
        assert_eq!(level, Level::Emergency);
    }

    #[test]
    fn ascending_multi_level_recovery() {
        // From Emergency, drop well below Warning recovery (55)
        let level = level_ascending(40.0, Level::Emergency, 60.0, 75.0, 85.0, 5.0);
        assert_eq!(level, Level::Normal);
    }

    #[test]
    fn ascending_stay_in_hysteresis_zone() {
        // At Warning (threshold 60, recovery 55), value = 57 → stay Warning
        let level = level_ascending(57.0, Level::Warning, 60.0, 75.0, 85.0, 5.0);
        assert_eq!(level, Level::Warning);
    }

    #[test]
    fn descending_skip_levels_on_escalation() {
        // Jump from Normal to Emergency (value drops below emergency threshold)
        let level = level_descending(3.0, Level::Normal, 40.0, 20.0, 10.0, 5.0);
        assert_eq!(level, Level::Emergency);
    }

    #[test]
    fn descending_multi_level_recovery() {
        // From Emergency, value rises well above all recovery thresholds
        let level = level_descending(80.0, Level::Emergency, 40.0, 20.0, 10.0, 5.0);
        assert_eq!(level, Level::Normal);
    }

    #[test]
    fn descending_stay_in_hysteresis_zone() {
        // At Warning (threshold 40, recovery 45), value = 42 → stay Warning
        let level = level_descending(42.0, Level::Warning, 40.0, 20.0, 10.0, 5.0);
        assert_eq!(level, Level::Warning);
    }
}
