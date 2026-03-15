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

// ── Command Protocol Types (Manager ↔ Engine) ─────────────────

/// Protocol-level severity (3-level, for Manager↔Engine communication).
/// Internal monitors use 4-level `Level`; PolicyEngine converts Emergency → Suspend command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceLevel {
    Normal,
    Warning,
    Critical,
}

/// Engine operational state reported to Manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineState {
    Idle,
    Running,
    Suspended,
}

/// Manager → Engine command.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineCommand {
    /// Set compute resource level with optional throughput target.
    SetComputeLevel {
        level: ResourceLevel,
        target_throughput: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        deadline_ms: Option<u64>,
    },
    /// Switch active compute unit (e.g., "cpu", "gpu", "opencl").
    SwitchComputeUnit { device: String },
    /// Pre-warm a compute unit for potential switch.
    PrepareComputeUnit { device: String },
    /// Set memory pressure level with eviction target.
    SetMemoryLevel {
        level: ResourceLevel,
        target_ratio: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        deadline_ms: Option<u64>,
    },
    /// Suspend inference immediately.
    Suspend,
    /// Resume from suspended state.
    Resume,
}

/// Batch of commands from Manager to Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineDirective {
    pub seq_id: u64,
    pub commands: Vec<EngineCommand>,
}

/// Top-level message from Manager to Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ManagerMessage {
    Directive(EngineDirective),
}

/// Engine capability report (sent once after connection).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCapability {
    pub available_devices: Vec<String>,
    pub active_device: String,
    pub max_kv_tokens: usize,
    pub bytes_per_kv_token: usize,
    pub num_layers: usize,
}

/// Engine status heartbeat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatus {
    pub active_device: String,
    pub compute_level: ResourceLevel,
    pub actual_throughput: f32,
    pub memory_level: ResourceLevel,
    pub kv_cache_bytes: u64,
    pub kv_cache_tokens: usize,
    pub kv_cache_utilization: f32,
    pub memory_lossless_min: f32,
    pub memory_lossy_min: f32,
    pub state: EngineState,
    pub tokens_generated: usize,
}

/// Result of executing a single command.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CommandResult {
    Ok,
    Partial { achieved: f32, reason: String },
    Rejected { reason: String },
}

/// Response to an EngineDirective (matches seq_id).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResponse {
    pub seq_id: u64,
    pub results: Vec<CommandResult>,
}

/// Top-level message from Engine to Manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineMessage {
    Capability(EngineCapability),
    Heartbeat(EngineStatus),
    Response(CommandResponse),
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

    // ── Protocol types tests ──────────────────────────────────

    #[test]
    fn test_resource_level_ordering() {
        assert!(ResourceLevel::Normal < ResourceLevel::Warning);
        assert!(ResourceLevel::Warning < ResourceLevel::Critical);
        assert!(ResourceLevel::Normal < ResourceLevel::Critical);
    }

    #[test]
    fn test_engine_command_serde_set_compute_level() {
        let cmd = EngineCommand::SetComputeLevel {
            level: ResourceLevel::Warning,
            target_throughput: 0.7,
            deadline_ms: None,
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("\"type\":\"set_compute_level\""));
        assert!(json.contains("\"level\":\"warning\""));
        let back: EngineCommand = serde_json::from_str(&json).unwrap();
        match back {
            EngineCommand::SetComputeLevel {
                level,
                target_throughput,
                deadline_ms,
            } => {
                assert_eq!(level, ResourceLevel::Warning);
                assert!((target_throughput - 0.7).abs() < f32::EPSILON);
                assert!(deadline_ms.is_none());
            }
            _ => panic!("Expected SetComputeLevel"),
        }
    }

    #[test]
    fn test_engine_command_serde_set_memory_level() {
        let cmd = EngineCommand::SetMemoryLevel {
            level: ResourceLevel::Critical,
            target_ratio: 0.5,
            deadline_ms: Some(1000),
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("\"type\":\"set_memory_level\""));
        assert!(json.contains("\"deadline_ms\":1000"));
        let back: EngineCommand = serde_json::from_str(&json).unwrap();
        match back {
            EngineCommand::SetMemoryLevel {
                level,
                target_ratio,
                deadline_ms,
            } => {
                assert_eq!(level, ResourceLevel::Critical);
                assert!((target_ratio - 0.5).abs() < f32::EPSILON);
                assert_eq!(deadline_ms, Some(1000));
            }
            _ => panic!("Expected SetMemoryLevel"),
        }
    }

    #[test]
    fn test_engine_command_serde_switch_compute_unit() {
        let cmd = EngineCommand::SwitchComputeUnit {
            device: "cpu".to_string(),
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let back: EngineCommand = serde_json::from_str(&json).unwrap();
        match back {
            EngineCommand::SwitchComputeUnit { device } => assert_eq!(device, "cpu"),
            _ => panic!("Expected SwitchComputeUnit"),
        }
    }

    #[test]
    fn test_engine_command_serde_suspend_resume() {
        let suspend = EngineCommand::Suspend;
        let json = serde_json::to_string(&suspend).unwrap();
        assert!(json.contains("\"type\":\"suspend\""));
        let back: EngineCommand = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, EngineCommand::Suspend));

        let resume = EngineCommand::Resume;
        let json = serde_json::to_string(&resume).unwrap();
        assert!(json.contains("\"type\":\"resume\""));
        let back: EngineCommand = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, EngineCommand::Resume));
    }

    #[test]
    fn test_engine_directive_serde_roundtrip() {
        let directive = EngineDirective {
            seq_id: 42,
            commands: vec![
                EngineCommand::SetComputeLevel {
                    level: ResourceLevel::Warning,
                    target_throughput: 0.7,
                    deadline_ms: None,
                },
                EngineCommand::SetMemoryLevel {
                    level: ResourceLevel::Critical,
                    target_ratio: 0.5,
                    deadline_ms: Some(1000),
                },
            ],
        };
        let json = serde_json::to_string(&directive).unwrap();
        let back: EngineDirective = serde_json::from_str(&json).unwrap();
        assert_eq!(back.seq_id, 42);
        assert_eq!(back.commands.len(), 2);
    }

    #[test]
    fn test_manager_message_serde_roundtrip() {
        let msg = ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::Suspend],
        });
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"directive\""));
        let back: ManagerMessage = serde_json::from_str(&json).unwrap();
        match back {
            ManagerMessage::Directive(d) => {
                assert_eq!(d.seq_id, 1);
                assert_eq!(d.commands.len(), 1);
                assert!(matches!(d.commands[0], EngineCommand::Suspend));
            }
        }
    }

    #[test]
    fn test_engine_capability_serde() {
        let cap = EngineCapability {
            available_devices: vec!["cpu".to_string(), "opencl".to_string()],
            active_device: "cpu".to_string(),
            max_kv_tokens: 2048,
            bytes_per_kv_token: 256,
            num_layers: 16,
        };
        let json = serde_json::to_string(&cap).unwrap();
        let back: EngineCapability = serde_json::from_str(&json).unwrap();
        assert_eq!(back.available_devices.len(), 2);
        assert_eq!(back.max_kv_tokens, 2048);
        assert_eq!(back.num_layers, 16);
    }

    #[test]
    fn test_engine_status_serde() {
        let status = EngineStatus {
            active_device: "cpu".to_string(),
            compute_level: ResourceLevel::Normal,
            actual_throughput: 15.0,
            memory_level: ResourceLevel::Warning,
            kv_cache_bytes: 1024 * 1024,
            kv_cache_tokens: 512,
            kv_cache_utilization: 0.25,
            memory_lossless_min: 1.0,
            memory_lossy_min: 0.01,
            state: EngineState::Running,
            tokens_generated: 100,
        };
        let json = serde_json::to_string(&status).unwrap();
        let back: EngineStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back.state, EngineState::Running);
        assert!((back.actual_throughput - 15.0).abs() < f32::EPSILON);
        assert_eq!(back.kv_cache_tokens, 512);
    }

    #[test]
    fn test_command_result_serde() {
        let ok = CommandResult::Ok;
        let json = serde_json::to_string(&ok).unwrap();
        assert!(json.contains("\"status\":\"ok\""));

        let partial = CommandResult::Partial {
            achieved: 0.7,
            reason: "throttled".to_string(),
        };
        let json = serde_json::to_string(&partial).unwrap();
        assert!(json.contains("\"status\":\"partial\""));

        let rejected = CommandResult::Rejected {
            reason: "single backend".to_string(),
        };
        let json = serde_json::to_string(&rejected).unwrap();
        assert!(json.contains("\"status\":\"rejected\""));
        let back: CommandResult = serde_json::from_str(&json).unwrap();
        match back {
            CommandResult::Rejected { reason } => assert_eq!(reason, "single backend"),
            _ => panic!("Expected Rejected"),
        }
    }

    #[test]
    fn test_command_response_serde() {
        let resp = CommandResponse {
            seq_id: 5,
            results: vec![
                CommandResult::Ok,
                CommandResult::Rejected {
                    reason: "n/a".into(),
                },
            ],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: CommandResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.seq_id, 5);
        assert_eq!(back.results.len(), 2);
    }

    #[test]
    fn test_engine_message_serde_variants() {
        // Capability
        let msg = EngineMessage::Capability(EngineCapability {
            available_devices: vec!["cpu".into()],
            active_device: "cpu".into(),
            max_kv_tokens: 1024,
            bytes_per_kv_token: 128,
            num_layers: 8,
        });
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"capability\""));
        let back: EngineMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, EngineMessage::Capability(_)));

        // Heartbeat
        let msg = EngineMessage::Heartbeat(EngineStatus {
            active_device: "cpu".into(),
            compute_level: ResourceLevel::Normal,
            actual_throughput: 10.0,
            memory_level: ResourceLevel::Normal,
            kv_cache_bytes: 0,
            kv_cache_tokens: 0,
            kv_cache_utilization: 0.0,
            memory_lossless_min: 1.0,
            memory_lossy_min: 0.01,
            state: EngineState::Idle,
            tokens_generated: 0,
        });
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"heartbeat\""));

        // Response
        let msg = EngineMessage::Response(CommandResponse {
            seq_id: 1,
            results: vec![CommandResult::Ok],
        });
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"response\""));
    }

    #[test]
    fn test_engine_state_serde() {
        assert_eq!(
            serde_json::to_string(&EngineState::Idle).unwrap(),
            "\"idle\""
        );
        assert_eq!(
            serde_json::to_string(&EngineState::Running).unwrap(),
            "\"running\""
        );
        assert_eq!(
            serde_json::to_string(&EngineState::Suspended).unwrap(),
            "\"suspended\""
        );
    }
}
