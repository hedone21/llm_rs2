use serde::{Deserialize, Serialize};

/// Common severity level shared by all signals.
/// Ordered by severity: Normal < Warning < Critical < Emergency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    Normal,
    Warning,
    Critical,
    Emergency,
}

/// Recommended compute backend from Manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecommendedBackend {
    Cpu,
    Gpu,
    Any,
}

/// Reason for compute guidance signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeReason {
    CpuBottleneck,
    GpuBottleneck,
    CpuAvailable,
    GpuAvailable,
    BothLoaded,
    Balanced,
}

/// Reason for energy constraint signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyReason {
    BatteryLow,
    BatteryCritical,
    PowerLimit,
    ThermalPower,
    Charging,
    #[serde(rename = "none")]
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

/// System signal received from the resource manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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
    fn test_system_signal_serde_roundtrip() {
        let sig = SystemSignal::MemoryPressure {
            level: Level::Critical,
            available_bytes: 1024,
            reclaim_target_bytes: 512,
        };
        let json = serde_json::to_string(&sig).unwrap();
        let back: SystemSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(back.level(), Level::Critical);
    }
}
