use serde::Deserialize;

/// Top-level Manager configuration, loadable from TOML.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub manager: ManagerConfig,
    pub memory: Option<MemoryMonitorConfig>,
    pub thermal: Option<ThermalMonitorConfig>,
    pub compute: Option<ComputeMonitorConfig>,
    pub energy: Option<EnergyMonitorConfig>,
    pub external: Option<ExternalMonitorConfig>,
}

impl Config {
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ManagerConfig {
    /// Default polling interval in milliseconds.
    pub poll_interval_ms: u64,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            poll_interval_ms: 1000,
        }
    }
}

/// Memory monitor configuration.
///
/// Thresholds are available memory percentage (descending: lower is worse).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MemoryMonitorConfig {
    pub enabled: bool,
    pub poll_interval_ms: Option<u64>,
    pub warning_pct: f64,
    pub critical_pct: f64,
    pub emergency_pct: f64,
    pub hysteresis_pct: f64,
}

impl Default for MemoryMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poll_interval_ms: None,
            warning_pct: 40.0,
            critical_pct: 20.0,
            emergency_pct: 10.0,
            hysteresis_pct: 5.0,
        }
    }
}

/// Thermal monitor configuration.
///
/// Thresholds are in millidegrees Celsius (ascending: higher is worse).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ThermalMonitorConfig {
    pub enabled: bool,
    pub poll_interval_ms: Option<u64>,
    pub zone_types: Vec<String>,
    pub warning_mc: i32,
    pub critical_mc: i32,
    pub emergency_mc: i32,
    pub hysteresis_mc: i32,
}

impl Default for ThermalMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poll_interval_ms: None,
            zone_types: Vec::new(),
            warning_mc: 60000,
            critical_mc: 75000,
            emergency_mc: 85000,
            hysteresis_mc: 5000,
        }
    }
}

/// Compute monitor configuration.
///
/// ComputeGuidance has no Emergency level (max: Critical).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ComputeMonitorConfig {
    pub enabled: bool,
    pub poll_interval_ms: Option<u64>,
    pub warning_pct: f64,
    pub critical_pct: f64,
    pub hysteresis_pct: f64,
}

impl Default for ComputeMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poll_interval_ms: None,
            warning_pct: 70.0,
            critical_pct: 90.0,
            hysteresis_pct: 5.0,
        }
    }
}

/// Energy monitor configuration.
///
/// Thresholds are battery percentage (descending: lower is worse).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EnergyMonitorConfig {
    pub enabled: bool,
    pub poll_interval_ms: Option<u64>,
    pub warning_pct: f64,
    pub critical_pct: f64,
    pub emergency_pct: f64,
    pub warning_power_budget_mw: u32,
    pub critical_power_budget_mw: u32,
    pub emergency_power_budget_mw: u32,
    pub ignore_when_charging: bool,
}

impl Default for EnergyMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poll_interval_ms: None,
            warning_pct: 30.0,
            critical_pct: 15.0,
            emergency_pct: 5.0,
            warning_power_budget_mw: 3000,
            critical_power_budget_mw: 1500,
            emergency_power_budget_mw: 500,
            ignore_when_charging: true,
        }
    }
}

/// External monitor configuration for research/testing signal injection.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ExternalMonitorConfig {
    pub enabled: bool,
    /// Transport: "stdin" or "unix:<socket_path>".
    pub transport: String,
}

impl Default for ExternalMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            transport: "stdin".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_all_monitors_enabled() {
        let config = Config::default();
        assert_eq!(config.manager.poll_interval_ms, 1000);
        // Optional monitors are None by default
        assert!(config.memory.is_none());
        assert!(config.external.is_none());
    }

    #[test]
    fn parse_minimal_toml() {
        let toml_str = r#"
[manager]
poll_interval_ms = 500

[memory]
enabled = true
warning_pct = 35.0
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.manager.poll_interval_ms, 500);
        let mem = config.memory.unwrap();
        assert!(mem.enabled);
        assert_eq!(mem.warning_pct, 35.0);
        assert_eq!(mem.critical_pct, 20.0); // default
    }

    #[test]
    fn parse_external_config() {
        let toml_str = r#"
[external]
enabled = true
transport = "unix:/tmp/test.sock"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        let ext = config.external.unwrap();
        assert!(ext.enabled);
        assert_eq!(ext.transport, "unix:/tmp/test.sock");
    }

    #[test]
    fn parse_full_config() {
        let toml_str = r#"
[manager]
poll_interval_ms = 2000

[memory]
enabled = true
warning_pct = 40.0
critical_pct = 20.0
emergency_pct = 10.0
hysteresis_pct = 5.0

[thermal]
enabled = true
zone_types = ["x86_pkg_temp"]
warning_mc = 60000
critical_mc = 75000
emergency_mc = 85000
hysteresis_mc = 5000

[compute]
enabled = true
warning_pct = 70.0
critical_pct = 90.0

[energy]
enabled = false
ignore_when_charging = true

[external]
enabled = true
transport = "stdin"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.manager.poll_interval_ms, 2000);
        assert!(config.memory.unwrap().enabled);
        assert_eq!(
            config.thermal.unwrap().zone_types,
            vec!["x86_pkg_temp".to_string()]
        );
        assert!(!config.energy.unwrap().enabled);
        assert!(config.external.unwrap().enabled);
    }
}
