use serde::Deserialize;

/// Top-level Manager configuration, loadable from TOML.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub monitor: MonitorConfig,
    pub memory: MemoryThresholds,
    pub thermal: ThermalThresholds,
    pub compute: ComputeThresholds,
    pub energy: EnergyThresholds,
}

impl Config {
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MonitorConfig {
    /// Default polling interval in milliseconds.
    pub poll_interval_ms: u64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            poll_interval_ms: 1000,
        }
    }
}

/// Memory pressure thresholds (available memory percentage).
///
/// Level escalates when available memory drops BELOW the threshold.
/// Hysteresis: recovery requires rising ABOVE threshold + gap.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MemoryThresholds {
    pub warning_available_pct: f64,
    pub critical_available_pct: f64,
    pub emergency_available_pct: f64,
    /// Hysteresis gap in percentage points.
    pub hysteresis_pct: f64,
}

impl Default for MemoryThresholds {
    fn default() -> Self {
        Self {
            warning_available_pct: 40.0,
            critical_available_pct: 20.0,
            emergency_available_pct: 10.0,
            hysteresis_pct: 5.0,
        }
    }
}

/// Thermal thresholds (millidegrees Celsius).
///
/// Level escalates when temperature rises ABOVE the threshold.
/// Hysteresis: recovery requires dropping BELOW threshold - gap.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ThermalThresholds {
    pub warning_temp_mc: i32,
    pub critical_temp_mc: i32,
    pub emergency_temp_mc: i32,
    /// Hysteresis gap in millidegrees.
    pub hysteresis_mc: i32,
    /// Filter thermal zones by type (e.g., `["x86_pkg_temp", "TCPU"]`).
    /// Only matching zones are monitored. Empty = monitor all zones (default).
    /// Zone types are read from `/sys/class/thermal/thermal_zone*/type`.
    pub zone_types: Vec<String>,
}

impl Default for ThermalThresholds {
    fn default() -> Self {
        Self {
            warning_temp_mc: 60000,
            critical_temp_mc: 75000,
            emergency_temp_mc: 85000,
            hysteresis_mc: 5000,
            zone_types: Vec::new(),
        }
    }
}

/// Compute (CPU/GPU usage) thresholds.
///
/// ComputeGuidance has no Emergency level (max: Critical).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ComputeThresholds {
    pub warning_usage_pct: f64,
    pub critical_usage_pct: f64,
    /// Hysteresis gap in percentage points.
    pub hysteresis_pct: f64,
}

impl Default for ComputeThresholds {
    fn default() -> Self {
        Self {
            warning_usage_pct: 70.0,
            critical_usage_pct: 90.0,
            hysteresis_pct: 5.0,
        }
    }
}

/// Energy thresholds (battery percentage and power budget).
///
/// Level escalates when battery drops BELOW the threshold.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EnergyThresholds {
    pub warning_battery_pct: f64,
    pub critical_battery_pct: f64,
    pub emergency_battery_pct: f64,
    pub warning_power_budget_mw: u32,
    pub critical_power_budget_mw: u32,
    pub emergency_power_budget_mw: u32,
    /// Skip energy signals when charging.
    pub ignore_when_charging: bool,
}

impl Default for EnergyThresholds {
    fn default() -> Self {
        Self {
            warning_battery_pct: 30.0,
            critical_battery_pct: 15.0,
            emergency_battery_pct: 5.0,
            warning_power_budget_mw: 3000,
            critical_power_budget_mw: 1500,
            emergency_power_budget_mw: 500,
            ignore_when_charging: true,
        }
    }
}
