use llm_shared::{
    EngineCommand, EngineDirective, EngineStatus, Level, ResourceLevel, SystemSignal,
};
use std::time::Instant;

/// Snapshot of all monitor readings.
#[derive(Debug, Clone)]
pub struct MonitorSnapshot {
    pub memory_level: Level,
    pub compute_level: Level,
    pub thermal_level: Level,
    pub energy_level: Level,
    pub thermal_throttle_ratio: f64,
    pub recommended_device: Option<String>,
}

impl Default for MonitorSnapshot {
    fn default() -> Self {
        Self {
            memory_level: Level::Normal,
            compute_level: Level::Normal,
            thermal_level: Level::Normal,
            energy_level: Level::Normal,
            thermal_throttle_ratio: 1.0,
            recommended_device: None,
        }
    }
}

impl MonitorSnapshot {
    /// Update snapshot from a SystemSignal.
    pub fn update(&mut self, signal: &SystemSignal) {
        match signal {
            SystemSignal::MemoryPressure { level, .. } => {
                self.memory_level = *level;
            }
            SystemSignal::ComputeGuidance {
                level,
                recommended_backend,
                ..
            } => {
                self.compute_level = *level;
                self.recommended_device = Some(match recommended_backend {
                    llm_shared::RecommendedBackend::Cpu => "cpu".to_string(),
                    llm_shared::RecommendedBackend::Gpu => "gpu".to_string(),
                    llm_shared::RecommendedBackend::Any => "any".to_string(),
                });
            }
            SystemSignal::ThermalAlert {
                level,
                throttle_ratio,
                ..
            } => {
                self.thermal_level = *level;
                self.thermal_throttle_ratio = *throttle_ratio;
            }
            SystemSignal::EnergyConstraint { level, .. } => {
                self.energy_level = *level;
            }
        }
    }
}

/// PolicyEngine converts 4-monitor signals into 2-domain EngineDirectives.
///
/// Decision rules:
/// - Compute domain: worst of (compute, thermal, energy) levels
/// - Memory domain: worst of (memory, thermal at Emergency, energy at Critical+)
/// - 4-level → 3-level: Emergency → Suspend command
/// - Deduplication: skip if (compute_level, memory_level, device) unchanged
/// - Cooldown: minimum interval between directives
pub struct PolicyEngine {
    last_compute_level: ResourceLevel,
    last_memory_level: ResourceLevel,
    last_device: String,
    last_suspended: bool,
    next_seq_id: u64,
    cooldown_ms: u64,
    last_directive_time: Option<Instant>,
}

impl PolicyEngine {
    pub fn new(cooldown_ms: u64) -> Self {
        Self {
            last_compute_level: ResourceLevel::Normal,
            last_memory_level: ResourceLevel::Normal,
            last_device: String::new(),
            last_suspended: false,
            next_seq_id: 1,
            cooldown_ms,
            last_directive_time: None,
        }
    }

    /// Evaluate the current monitor snapshot and optionally produce a directive.
    pub fn evaluate(
        &mut self,
        snapshot: &MonitorSnapshot,
        _engine_status: Option<&EngineStatus>,
    ) -> Option<EngineDirective> {
        // Cooldown check
        if let Some(last) = self.last_directive_time
            && last.elapsed().as_millis() < self.cooldown_ms as u128
        {
            return None;
        }

        // Compute effective levels for each domain
        let effective_compute = self.compute_effective_level(snapshot);
        let effective_memory = self.memory_effective_level(snapshot);

        // Check for Emergency → Suspend
        let any_emergency = snapshot.memory_level == Level::Emergency
            || snapshot.compute_level == Level::Emergency
            || snapshot.thermal_level == Level::Emergency
            || snapshot.energy_level == Level::Emergency;

        if any_emergency {
            if self.last_suspended {
                return None; // Already suspended, don't re-send
            }
            let directive = EngineDirective {
                seq_id: self.next_seq(),
                commands: vec![EngineCommand::Suspend],
            };
            self.last_suspended = true;
            self.last_directive_time = Some(Instant::now());
            return Some(directive);
        }

        // If previously suspended but no longer emergency → Resume
        if self.last_suspended {
            let directive = EngineDirective {
                seq_id: self.next_seq(),
                commands: vec![EngineCommand::Resume],
            };
            self.last_suspended = false;
            self.last_directive_time = Some(Instant::now());
            return Some(directive);
        }

        // Determine target device
        let target_device = self.compute_target_device(snapshot, effective_compute);

        // Deduplication: skip if nothing changed
        if effective_compute == self.last_compute_level
            && effective_memory == self.last_memory_level
            && target_device == self.last_device
        {
            return None;
        }

        // Build commands
        let mut commands = Vec::new();

        // Compute domain commands
        if effective_compute != self.last_compute_level || target_device != self.last_device {
            let (delay_ms, _deadline) = Self::compute_params(effective_compute);
            if delay_ms > 0 {
                commands.push(EngineCommand::Throttle { delay_ms });
            } else {
                commands.push(EngineCommand::RestoreDefaults);
            }

            if target_device != self.last_device && !target_device.is_empty() {
                commands.push(EngineCommand::SwitchHw {
                    device: target_device.clone(),
                });
            }
        }

        // PrepareComputeUnit: thermal Warning + current GPU → prepare CPU
        if snapshot.thermal_level == Level::Warning && target_device != "cpu" {
            commands.push(EngineCommand::PrepareComputeUnit {
                device: "cpu".to_string(),
            });
        }

        // Memory domain commands
        if effective_memory != self.last_memory_level {
            let (keep_ratio, _deadline) = Self::memory_params(effective_memory);
            if keep_ratio < 1.0 {
                commands.push(EngineCommand::KvEvictSliding { keep_ratio });
            } else {
                commands.push(EngineCommand::RestoreDefaults);
            }
        }

        if commands.is_empty() {
            return None;
        }

        self.last_compute_level = effective_compute;
        self.last_memory_level = effective_memory;
        self.last_device = target_device;
        self.last_directive_time = Some(Instant::now());

        Some(EngineDirective {
            seq_id: self.next_seq(),
            commands,
        })
    }

    /// Compute effective level: worst of (compute, thermal, energy).
    fn compute_effective_level(&self, snap: &MonitorSnapshot) -> ResourceLevel {
        let levels = [
            level_to_resource(snap.compute_level),
            level_to_resource(snap.thermal_level),
            level_to_resource(snap.energy_level),
        ];
        levels.into_iter().max().unwrap_or(ResourceLevel::Normal)
    }

    /// Memory effective level: worst of (memory; thermal at Emergency adds Critical).
    fn memory_effective_level(&self, snap: &MonitorSnapshot) -> ResourceLevel {
        let mut worst = level_to_resource(snap.memory_level);

        // Thermal Emergency → additional memory pressure (Critical)
        if snap.thermal_level == Level::Emergency {
            worst = worst.max(ResourceLevel::Critical);
        }

        // Energy Critical+ → mild memory pressure (Warning)
        if snap.energy_level >= Level::Critical {
            worst = worst.max(ResourceLevel::Warning);
        }

        worst
    }

    /// Determine target compute device.
    fn compute_target_device(&self, snap: &MonitorSnapshot, effective: ResourceLevel) -> String {
        // Thermal Critical+ → force CPU
        if snap.thermal_level >= Level::Critical {
            return "cpu".to_string();
        }
        // Energy Warning+ → prefer CPU
        if snap.energy_level >= Level::Warning {
            return "cpu".to_string();
        }
        // Compute Critical → use recommended device
        if effective == ResourceLevel::Critical {
            if let Some(ref dev) = snap.recommended_device
                && dev != "any"
            {
                return dev.clone();
            }
            return "cpu".to_string();
        }
        // Normal/Warning → keep current (empty string means no change)
        String::new()
    }

    /// 반환: (delay_ms, deadline_ms)
    /// Normal → delay 없음(0), Warning → 30ms delay, Critical → 100ms delay
    fn compute_params(level: ResourceLevel) -> (u64, Option<u64>) {
        match level {
            ResourceLevel::Normal => (0, None),
            ResourceLevel::Warning => (30, None),
            ResourceLevel::Critical => (100, Some(1000)),
        }
    }

    /// 반환: (keep_ratio, deadline_ms)
    fn memory_params(level: ResourceLevel) -> (f32, Option<u64>) {
        match level {
            ResourceLevel::Normal => (1.0, None),
            ResourceLevel::Warning => (0.85, Some(2000)),
            ResourceLevel::Critical => (0.50, Some(1000)),
        }
    }

    fn next_seq(&mut self) -> u64 {
        let id = self.next_seq_id;
        self.next_seq_id += 1;
        id
    }
}

/// Convert internal 4-level to protocol 3-level (Emergency handled separately as Suspend).
fn level_to_resource(level: Level) -> ResourceLevel {
    match level {
        Level::Normal => ResourceLevel::Normal,
        Level::Warning => ResourceLevel::Warning,
        Level::Critical | Level::Emergency => ResourceLevel::Critical,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap_all_normal() -> MonitorSnapshot {
        MonitorSnapshot::default()
    }

    fn snap_with(memory: Level, compute: Level, thermal: Level, energy: Level) -> MonitorSnapshot {
        MonitorSnapshot {
            memory_level: memory,
            compute_level: compute,
            thermal_level: thermal,
            energy_level: energy,
            thermal_throttle_ratio: 1.0,
            recommended_device: None,
        }
    }

    #[test]
    fn test_policy_normal_all() {
        let mut policy = PolicyEngine::new(0);
        let result = policy.evaluate(&snap_all_normal(), None);
        assert!(result.is_none(), "All Normal → no directive");
    }

    #[test]
    fn test_policy_memory_warning() {
        let mut policy = PolicyEngine::new(0);
        let snap = snap_with(Level::Warning, Level::Normal, Level::Normal, Level::Normal);
        let directive = policy.evaluate(&snap, None).unwrap();
        assert!(!directive.commands.is_empty());

        let mem_cmd = directive
            .commands
            .iter()
            .find(|c| matches!(c, EngineCommand::KvEvictSliding { .. }));
        assert!(mem_cmd.is_some());
        match mem_cmd.unwrap() {
            EngineCommand::KvEvictSliding { keep_ratio } => {
                assert!((*keep_ratio - 0.85).abs() < f32::EPSILON);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_policy_memory_critical() {
        let mut policy = PolicyEngine::new(0);
        let snap = snap_with(Level::Critical, Level::Normal, Level::Normal, Level::Normal);
        let directive = policy.evaluate(&snap, None).unwrap();

        let mem_cmd = directive
            .commands
            .iter()
            .find(|c| matches!(c, EngineCommand::KvEvictSliding { .. }));
        assert!(mem_cmd.is_some(), "Expected KvEvictSliding command");
        match mem_cmd.unwrap() {
            EngineCommand::KvEvictSliding { keep_ratio } => {
                assert!((*keep_ratio - 0.50).abs() < f32::EPSILON);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_policy_thermal_critical() {
        let mut policy = PolicyEngine::new(0);
        let snap = snap_with(Level::Normal, Level::Normal, Level::Critical, Level::Normal);
        let directive = policy.evaluate(&snap, None).unwrap();

        // Should have Throttle (compute Critical) + SwitchHw("cpu")
        let throttle_cmd = directive
            .commands
            .iter()
            .find(|c| matches!(c, EngineCommand::Throttle { .. }));
        assert!(throttle_cmd.is_some(), "Expected Throttle command");
        match throttle_cmd.unwrap() {
            EngineCommand::Throttle { delay_ms } => {
                assert!(*delay_ms > 0, "Critical should have non-zero delay");
            }
            _ => unreachable!(),
        }

        let switch_cmd = directive
            .commands
            .iter()
            .find(|c| matches!(c, EngineCommand::SwitchHw { .. }));
        assert!(switch_cmd.is_some(), "Expected SwitchHw command");
        match switch_cmd.unwrap() {
            EngineCommand::SwitchHw { device } => {
                assert_eq!(device, "cpu");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_policy_energy_emergency() {
        let mut policy = PolicyEngine::new(0);
        let snap = snap_with(
            Level::Normal,
            Level::Normal,
            Level::Normal,
            Level::Emergency,
        );
        let directive = policy.evaluate(&snap, None).unwrap();
        assert_eq!(directive.commands.len(), 1);
        assert!(matches!(directive.commands[0], EngineCommand::Suspend));
    }

    #[test]
    fn test_policy_cross_domain() {
        let mut policy = PolicyEngine::new(0);
        // Thermal(Critical) affects compute + memory domains
        let snap = snap_with(
            Level::Warning,
            Level::Normal,
            Level::Critical,
            Level::Normal,
        );
        let directive = policy.evaluate(&snap, None).unwrap();

        // Should have both compute (Throttle) and memory (KvEvictSliding) commands
        let has_compute = directive.commands.iter().any(|c| {
            matches!(
                c,
                EngineCommand::Throttle { .. } | EngineCommand::RestoreDefaults
            )
        });
        let has_memory = directive.commands.iter().any(|c| {
            matches!(
                c,
                EngineCommand::KvEvictSliding { .. } | EngineCommand::RestoreDefaults
            )
        });
        assert!(has_compute, "Should have compute command");
        assert!(has_memory, "Should have memory command");
    }

    #[test]
    fn test_policy_deduplication() {
        let mut policy = PolicyEngine::new(0);
        let snap = snap_with(Level::Warning, Level::Normal, Level::Normal, Level::Normal);

        // First call produces directive
        let d1 = policy.evaluate(&snap, None);
        assert!(d1.is_some());

        // Second call with same snapshot → no directive
        let d2 = policy.evaluate(&snap, None);
        assert!(
            d2.is_none(),
            "Duplicate snapshot should produce no directive"
        );
    }

    #[test]
    fn test_policy_cooldown() {
        let mut policy = PolicyEngine::new(1000); // 1 second cooldown

        let snap1 = snap_with(Level::Warning, Level::Normal, Level::Normal, Level::Normal);
        let d1 = policy.evaluate(&snap1, None);
        assert!(d1.is_some());

        // Immediately change to Critical → should be blocked by cooldown
        let snap2 = snap_with(Level::Critical, Level::Normal, Level::Normal, Level::Normal);
        let d2 = policy.evaluate(&snap2, None);
        assert!(d2.is_none(), "Should be blocked by cooldown");
    }

    #[test]
    fn test_policy_prepare_warmup() {
        let mut policy = PolicyEngine::new(0);
        // Thermal Warning + some device → should PrepareComputeUnit("cpu")
        let snap = snap_with(Level::Normal, Level::Normal, Level::Warning, Level::Normal);
        let directive = policy.evaluate(&snap, None).unwrap();

        let prepare_cmd = directive
            .commands
            .iter()
            .find(|c| matches!(c, EngineCommand::PrepareComputeUnit { .. }));
        assert!(
            prepare_cmd.is_some(),
            "Thermal Warning should trigger PrepareComputeUnit"
        );
        match prepare_cmd.unwrap() {
            EngineCommand::PrepareComputeUnit { device } => {
                assert_eq!(device, "cpu");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_policy_emergency_to_suspend() {
        let mut policy = PolicyEngine::new(0);

        // Emergency → Suspend
        let snap = snap_with(
            Level::Normal,
            Level::Normal,
            Level::Emergency,
            Level::Normal,
        );
        let d = policy.evaluate(&snap, None).unwrap();
        assert!(matches!(d.commands[0], EngineCommand::Suspend));

        // Recovery → Resume
        let snap2 = snap_all_normal();
        let d2 = policy.evaluate(&snap2, None).unwrap();
        assert!(matches!(d2.commands[0], EngineCommand::Resume));
    }

    #[test]
    fn test_policy_seq_id_increments() {
        let mut policy = PolicyEngine::new(0);

        let snap1 = snap_with(Level::Warning, Level::Normal, Level::Normal, Level::Normal);
        let d1 = policy.evaluate(&snap1, None).unwrap();
        assert_eq!(d1.seq_id, 1);

        let snap2 = snap_with(Level::Critical, Level::Normal, Level::Normal, Level::Normal);
        // Need to bypass cooldown
        policy.last_directive_time = None;
        let d2 = policy.evaluate(&snap2, None).unwrap();
        assert_eq!(d2.seq_id, 2);
    }

    #[test]
    fn test_monitor_snapshot_update() {
        let mut snap = MonitorSnapshot::default();

        snap.update(&SystemSignal::MemoryPressure {
            level: Level::Warning,
            available_bytes: 100_000_000,
            reclaim_target_bytes: 50_000_000,
        });
        assert_eq!(snap.memory_level, Level::Warning);

        snap.update(&SystemSignal::ThermalAlert {
            level: Level::Critical,
            temperature_mc: 80000,
            throttling_active: true,
            throttle_ratio: 0.5,
        });
        assert_eq!(snap.thermal_level, Level::Critical);
        assert!((snap.thermal_throttle_ratio - 0.5).abs() < f64::EPSILON);
    }
}
