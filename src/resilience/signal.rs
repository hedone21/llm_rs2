/// Common severity level shared by all signals.
/// Ordered by severity: Normal < Warning < Critical < Emergency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    Normal,
    Warning,
    Critical,
    Emergency,
}

/// Recommended compute backend from Manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedBackend {
    Cpu,
    Gpu,
    Any,
}

/// Reason for compute guidance signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeReason {
    CpuBottleneck,
    GpuBottleneck,
    CpuAvailable,
    GpuAvailable,
    BothLoaded,
    Balanced,
}

/// Reason for energy constraint signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyReason {
    BatteryLow,
    BatteryCritical,
    PowerLimit,
    ThermalPower,
    Charging,
    None,
}

impl Level {
    /// Convert D-Bus string argument to Level.
    pub fn from_dbus_str(s: &str) -> Option<Self> {
        match s {
            "normal" => Some(Level::Normal),
            "warning" => Some(Level::Warning),
            "critical" => Some(Level::Critical),
            "emergency" => Some(Level::Emergency),
            _ => None,
        }
    }
}

impl RecommendedBackend {
    /// Convert D-Bus string argument to RecommendedBackend.
    pub fn from_dbus_str(s: &str) -> Option<Self> {
        match s {
            "cpu" => Some(RecommendedBackend::Cpu),
            "gpu" => Some(RecommendedBackend::Gpu),
            "any" => Some(RecommendedBackend::Any),
            _ => None,
        }
    }
}

impl ComputeReason {
    /// Convert D-Bus string argument to ComputeReason.
    pub fn from_dbus_str(s: &str) -> Option<Self> {
        match s {
            "cpu_bottleneck" => Some(ComputeReason::CpuBottleneck),
            "gpu_bottleneck" => Some(ComputeReason::GpuBottleneck),
            "cpu_available" => Some(ComputeReason::CpuAvailable),
            "gpu_available" => Some(ComputeReason::GpuAvailable),
            "both_loaded" => Some(ComputeReason::BothLoaded),
            "balanced" => Some(ComputeReason::Balanced),
            _ => None,
        }
    }
}

impl EnergyReason {
    /// Convert D-Bus string argument to EnergyReason.
    pub fn from_dbus_str(s: &str) -> Option<Self> {
        match s {
            "battery_low" => Some(EnergyReason::BatteryLow),
            "battery_critical" => Some(EnergyReason::BatteryCritical),
            "power_limit" => Some(EnergyReason::PowerLimit),
            "thermal_power" => Some(EnergyReason::ThermalPower),
            "charging" => Some(EnergyReason::Charging),
            "none" => Some(EnergyReason::None),
            _ => None,
        }
    }
}

/// System signal received from D-Bus Manager (`org.llm.Manager1`).
#[derive(Debug, Clone)]
pub enum SystemSignal {
    MemoryPressure {
        level: Level,
        available_bytes: u64,
        reclaim_target_bytes: u64,
    },
    ComputeGuidance {
        level: Level,
        recommended_backend: RecommendedBackend,
        reason: ComputeReason,
        cpu_usage_pct: f64,
        gpu_usage_pct: f64,
    },
    ThermalAlert {
        level: Level,
        temperature_mc: i32,
        throttling_active: bool,
        throttle_ratio: f64,
    },
    EnergyConstraint {
        level: Level,
        reason: EnergyReason,
        power_budget_mw: u32,
    },
}

impl SystemSignal {
    /// Extract the level from any signal variant.
    pub fn level(&self) -> Level {
        match self {
            SystemSignal::MemoryPressure { level, .. } => *level,
            SystemSignal::ComputeGuidance { level, .. } => *level,
            SystemSignal::ThermalAlert { level, .. } => *level,
            SystemSignal::EnergyConstraint { level, .. } => *level,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_ordering() {
        assert!(Level::Normal < Level::Warning);
        assert!(Level::Warning < Level::Critical);
        assert!(Level::Critical < Level::Emergency);
        assert!(Level::Normal < Level::Emergency);
    }

    #[test]
    fn test_level_max_returns_worst() {
        assert_eq!(Level::Normal.max(Level::Warning), Level::Warning);
        assert_eq!(Level::Critical.max(Level::Warning), Level::Critical);
        assert_eq!(Level::Normal.max(Level::Emergency), Level::Emergency);
        assert_eq!(
            Level::Warning
                .max(Level::Critical)
                .max(Level::Normal)
                .max(Level::Emergency),
            Level::Emergency
        );
    }

    #[test]
    fn test_level_from_dbus_str() {
        assert_eq!(Level::from_dbus_str("normal"), Some(Level::Normal));
        assert_eq!(Level::from_dbus_str("warning"), Some(Level::Warning));
        assert_eq!(Level::from_dbus_str("critical"), Some(Level::Critical));
        assert_eq!(Level::from_dbus_str("emergency"), Some(Level::Emergency));
        assert_eq!(Level::from_dbus_str("unknown"), None);
        assert_eq!(Level::from_dbus_str(""), None);
        assert_eq!(Level::from_dbus_str("Normal"), None); // case-sensitive
    }

    #[test]
    fn test_recommended_backend_from_dbus_str() {
        assert_eq!(
            RecommendedBackend::from_dbus_str("cpu"),
            Some(RecommendedBackend::Cpu)
        );
        assert_eq!(
            RecommendedBackend::from_dbus_str("gpu"),
            Some(RecommendedBackend::Gpu)
        );
        assert_eq!(
            RecommendedBackend::from_dbus_str("any"),
            Some(RecommendedBackend::Any)
        );
        assert_eq!(RecommendedBackend::from_dbus_str("tpu"), None);
    }

    #[test]
    fn test_compute_reason_from_dbus_str() {
        assert_eq!(
            ComputeReason::from_dbus_str("cpu_bottleneck"),
            Some(ComputeReason::CpuBottleneck)
        );
        assert_eq!(
            ComputeReason::from_dbus_str("gpu_bottleneck"),
            Some(ComputeReason::GpuBottleneck)
        );
        assert_eq!(
            ComputeReason::from_dbus_str("cpu_available"),
            Some(ComputeReason::CpuAvailable)
        );
        assert_eq!(
            ComputeReason::from_dbus_str("gpu_available"),
            Some(ComputeReason::GpuAvailable)
        );
        assert_eq!(
            ComputeReason::from_dbus_str("both_loaded"),
            Some(ComputeReason::BothLoaded)
        );
        assert_eq!(
            ComputeReason::from_dbus_str("balanced"),
            Some(ComputeReason::Balanced)
        );
        assert_eq!(ComputeReason::from_dbus_str("overloaded"), None);
    }

    #[test]
    fn test_energy_reason_from_dbus_str() {
        assert_eq!(
            EnergyReason::from_dbus_str("battery_low"),
            Some(EnergyReason::BatteryLow)
        );
        assert_eq!(
            EnergyReason::from_dbus_str("battery_critical"),
            Some(EnergyReason::BatteryCritical)
        );
        assert_eq!(
            EnergyReason::from_dbus_str("power_limit"),
            Some(EnergyReason::PowerLimit)
        );
        assert_eq!(
            EnergyReason::from_dbus_str("thermal_power"),
            Some(EnergyReason::ThermalPower)
        );
        assert_eq!(
            EnergyReason::from_dbus_str("charging"),
            Some(EnergyReason::Charging)
        );
        assert_eq!(
            EnergyReason::from_dbus_str("none"),
            Some(EnergyReason::None)
        );
        assert_eq!(EnergyReason::from_dbus_str("solar"), None);
    }

    #[test]
    fn test_signal_level_extraction() {
        let sig = SystemSignal::MemoryPressure {
            level: Level::Critical,
            available_bytes: 1024,
            reclaim_target_bytes: 512,
        };
        assert_eq!(sig.level(), Level::Critical);

        let sig = SystemSignal::ThermalAlert {
            level: Level::Emergency,
            temperature_mc: 85000,
            throttling_active: true,
            throttle_ratio: 0.5,
        };
        assert_eq!(sig.level(), Level::Emergency);
    }
}
