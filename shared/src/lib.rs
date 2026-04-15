use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
        total_bytes: u64,
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
/// Action-specific variants that preserve Manager's cross-domain selection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineCommand {
    // ── Compute domain ──
    /// Throttle token generation by inserting fixed delay between tokens.
    Throttle { delay_ms: u64 },
    /// Set target TBT (time between tokens) in milliseconds.
    /// Engine dynamically sleeps after each token to maintain target TBT.
    /// 0 = disable pacing. More precise than Throttle for QoS control.
    SetTargetTbt { target_ms: u64 },
    /// Skip transformer layers to reduce compute load.
    LayerSkip { skip_ratio: f32 },

    // ── Memory domain ──
    /// Evict KV cache entries using H2O (Heavy-Hitter Oracle) policy.
    KvEvictH2o { keep_ratio: f32 },
    /// Evict KV cache entries using sliding window policy.
    KvEvictSliding { keep_ratio: f32 },
    /// Evict KV cache using StreamingLLM (sink + window) policy.
    KvStreaming {
        sink_size: usize,
        window_size: usize,
    },
    /// Evict and merge KV cache entries using D2O (Dynamic Discriminative Operations) policy.
    KvMergeD2o { keep_ratio: f32 },
    /// Dynamically transition KV cache quantization bits.
    KvQuantDynamic { target_bits: u8 },

    // ── Query ──
    /// Request QCF cost estimates for lossy actions (MSG-036b).
    /// Engine responds with Ok, then sends separate QcfEstimate message.
    RequestQcf,

    // ── Lifecycle ──
    /// Restore all action-induced state to defaults (skip→None, throttle→0).
    RestoreDefaults,
    /// Switch active compute unit (e.g., "cpu", "opencl").
    SwitchHw { device: String },
    /// Pre-warm a compute unit for potential switch.
    PrepareComputeUnit { device: String },
    /// Suspend inference immediately.
    Suspend,
    /// Resume from suspended state.
    Resume,

    // ── Tensor partition domain ──
    /// Set GPU ratio for tensor partition (0.0~1.0).
    /// 0.0 or 1.0 = disable partition (GPU-only).
    /// Triggers weight re-split and workspace reallocation at next forward pass.
    SetPartitionRatio { ratio: f32 },

    // ── Prefill domain ──
    /// Adjust prefill execution policy for GPU contention management.
    /// All fields are Optional — only provided fields are updated;
    /// omitted fields retain their current values.
    SetPrefillPolicy {
        /// Prefill chunk size (tokens per forward pass).
        /// Smaller = shorter GPU occupancy per chunk = more game frames.
        /// Default: 256.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chunk_size: Option<usize>,
        /// Inter-layer yield delay in milliseconds.
        /// After each transformer layer, engine calls synchronize() + sleep(yield_ms).
        /// 0 = no yield. Default: 0.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        yield_ms: Option<u32>,
        /// CPU chunk size for GPU-CPU interleaving.
        /// 0 = interleave disabled (GPU only).
        /// Positive value = after each GPU chunk, CPU processes this many
        /// tokens while GPU is free for the game. Default: 0.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cpu_chunk_size: Option<usize>,
    },
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
    /// Actions the Engine can currently execute (evaluated per heartbeat).
    #[serde(default)]
    pub available_actions: Vec<String>,
    /// Actions currently applied (e.g., "kv_evict_h2o", "throttle").
    #[serde(default)]
    pub active_actions: Vec<String>,
    /// Current eviction policy name ("none", "h2o", "sliding", etc.).
    #[serde(default)]
    pub eviction_policy: String,
    /// Current KV cache dtype ("f16", "q8", "q4", "q2").
    #[serde(default)]
    pub kv_dtype: String,
    /// Current layer skip ratio (0.0 = no skip).
    #[serde(default)]
    pub skip_ratio: f32,
    /// Current inference phase: "idle", "prefill", or "decode".
    #[serde(default)]
    pub phase: String,
    /// Prefill progress: tokens processed so far. 0 if not prefilling.
    #[serde(default)]
    pub prefill_pos: usize,
    /// Prefill total: prompt token count. 0 if not prefilling.
    #[serde(default)]
    pub prefill_total: usize,
    /// Current tensor partition GPU ratio (0.0 = disabled).
    #[serde(default)]
    pub partition_ratio: f32,
    /// Engine process 자신의 CPU 사용률 (MSG-060 #17, MSG-067). `/proc/self/stat`의
    /// (utime+stime) delta / (CLK_TCK × num_cpus × elapsed). 측정 실패 시 0.0.
    /// 값 범위는 송출 직전 [0.0, 1.0]로 clamp 된다 (INV-091, INV-092).
    #[serde(default)]
    pub self_cpu_pct: f64,
    /// Engine process 자신의 GPU 사용률 (MSG-060 #18, MSG-068). Phase 1에서는 항상 0.0
    /// placeholder이며 Phase 2에서 OpenCL profiling 기반 실측으로 재정의된다.
    #[serde(default)]
    pub self_gpu_pct: f64,
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

/// QCF cost estimates from Engine (MSG-085).
/// Sent as a separate EngineMessage after CommandResponse for RequestQcf.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QcfEstimate {
    /// Per-lossy-action estimated quality cost.
    /// Keys: action identifier (e.g., "kv_evict_h2o"). Values: QCF cost >= 0.0 (MSG-087).
    /// Only actions the Engine can currently compute are included (MSG-086).
    pub estimates: HashMap<String, f32>,
}

/// Top-level message from Engine to Manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineMessage {
    Capability(EngineCapability),
    Heartbeat(EngineStatus),
    Response(CommandResponse),
    QcfEstimate(QcfEstimate),
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
            total_bytes: 4096,
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
    fn test_engine_command_serde_throttle() {
        let cmd = EngineCommand::Throttle { delay_ms: 50 };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("\"type\":\"throttle\""));
        let back: EngineCommand = serde_json::from_str(&json).unwrap();
        match back {
            EngineCommand::Throttle { delay_ms } => assert_eq!(delay_ms, 50),
            _ => panic!("Expected Throttle"),
        }
    }

    #[test]
    fn test_engine_command_serde_layer_skip() {
        let cmd = EngineCommand::LayerSkip { skip_ratio: 0.25 };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("\"type\":\"layer_skip\""));
        let back: EngineCommand = serde_json::from_str(&json).unwrap();
        match back {
            EngineCommand::LayerSkip { skip_ratio } => {
                assert!((skip_ratio - 0.25).abs() < f32::EPSILON);
            }
            _ => panic!("Expected LayerSkip"),
        }
    }

    #[test]
    fn test_engine_command_serde_kv_evict_h2o() {
        let cmd = EngineCommand::KvEvictH2o { keep_ratio: 0.48 };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("\"type\":\"kv_evict_h2o\""));
        let back: EngineCommand = serde_json::from_str(&json).unwrap();
        match back {
            EngineCommand::KvEvictH2o { keep_ratio } => {
                assert!((keep_ratio - 0.48).abs() < f32::EPSILON);
            }
            _ => panic!("Expected KvEvictH2o"),
        }
    }

    #[test]
    fn test_engine_command_serde_kv_evict_sliding() {
        let cmd = EngineCommand::KvEvictSliding { keep_ratio: 0.6 };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("\"type\":\"kv_evict_sliding\""));
        let back: EngineCommand = serde_json::from_str(&json).unwrap();
        match back {
            EngineCommand::KvEvictSliding { keep_ratio } => {
                assert!((keep_ratio - 0.6).abs() < f32::EPSILON);
            }
            _ => panic!("Expected KvEvictSliding"),
        }
    }

    #[test]
    fn test_engine_command_serde_kv_streaming() {
        let cmd = EngineCommand::KvStreaming {
            sink_size: 4,
            window_size: 256,
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("\"type\":\"kv_streaming\""));
        let back: EngineCommand = serde_json::from_str(&json).unwrap();
        match back {
            EngineCommand::KvStreaming {
                sink_size,
                window_size,
            } => {
                assert_eq!(sink_size, 4);
                assert_eq!(window_size, 256);
            }
            _ => panic!("Expected KvStreaming"),
        }
    }

    #[test]
    fn test_engine_command_serde_kv_merge_d2o() {
        let cmd = EngineCommand::KvMergeD2o { keep_ratio: 0.75 };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("\"type\":\"kv_merge_d2o\""));
        let back: EngineCommand = serde_json::from_str(&json).unwrap();
        match back {
            EngineCommand::KvMergeD2o { keep_ratio } => {
                assert!((keep_ratio - 0.75).abs() < f32::EPSILON);
            }
            _ => panic!("Expected KvMergeD2o"),
        }
    }

    #[test]
    fn test_engine_command_serde_kv_quant_dynamic() {
        let cmd = EngineCommand::KvQuantDynamic { target_bits: 4 };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("\"type\":\"kv_quant_dynamic\""));
        let back: EngineCommand = serde_json::from_str(&json).unwrap();
        match back {
            EngineCommand::KvQuantDynamic { target_bits } => assert_eq!(target_bits, 4),
            _ => panic!("Expected KvQuantDynamic"),
        }
    }

    #[test]
    fn test_engine_command_serde_request_qcf() {
        let cmd = EngineCommand::RequestQcf;
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("\"type\":\"request_qcf\""));
        let back: EngineCommand = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, EngineCommand::RequestQcf));
    }

    #[test]
    fn test_engine_command_serde_restore_defaults() {
        let cmd = EngineCommand::RestoreDefaults;
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("\"type\":\"restore_defaults\""));
        let back: EngineCommand = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, EngineCommand::RestoreDefaults));
    }

    #[test]
    fn test_engine_command_serde_switch_hw() {
        let cmd = EngineCommand::SwitchHw {
            device: "cpu".to_string(),
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("\"type\":\"switch_hw\""));
        let back: EngineCommand = serde_json::from_str(&json).unwrap();
        match back {
            EngineCommand::SwitchHw { device } => assert_eq!(device, "cpu"),
            _ => panic!("Expected SwitchHw"),
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
    fn test_engine_command_serde_set_partition_ratio() {
        let cmd = EngineCommand::SetPartitionRatio { ratio: 0.65 };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("\"type\":\"set_partition_ratio\""));
        assert!(json.contains("\"ratio\":0.65"));
        let back: EngineCommand = serde_json::from_str(&json).unwrap();
        match back {
            EngineCommand::SetPartitionRatio { ratio } => {
                assert!((ratio - 0.65).abs() < f32::EPSILON);
            }
            _ => panic!("Expected SetPartitionRatio"),
        }
    }

    #[test]
    fn test_engine_status_partition_ratio() {
        let mut status = make_test_status();
        status.partition_ratio = 0.75;
        let json = serde_json::to_string(&status).unwrap();
        let back: EngineStatus = serde_json::from_str(&json).unwrap();
        assert!((back.partition_ratio - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_engine_command_serde_set_prefill_policy_full() {
        let cmd = EngineCommand::SetPrefillPolicy {
            chunk_size: Some(48),
            yield_ms: Some(10),
            cpu_chunk_size: Some(16),
        };
        let json = serde_json::to_string(&cmd).unwrap();
        assert!(json.contains("\"type\":\"set_prefill_policy\""));
        assert!(json.contains("\"chunk_size\":48"));
        assert!(json.contains("\"yield_ms\":10"));
        assert!(json.contains("\"cpu_chunk_size\":16"));
        let back: EngineCommand = serde_json::from_str(&json).unwrap();
        match back {
            EngineCommand::SetPrefillPolicy {
                chunk_size,
                yield_ms,
                cpu_chunk_size,
            } => {
                assert_eq!(chunk_size, Some(48));
                assert_eq!(yield_ms, Some(10));
                assert_eq!(cpu_chunk_size, Some(16));
            }
            _ => panic!("Expected SetPrefillPolicy"),
        }
    }

    #[test]
    fn test_engine_command_serde_set_prefill_policy_partial() {
        // Only chunk_size provided — other fields should default to None
        let json = r#"{"type":"set_prefill_policy","chunk_size":64}"#;
        let back: EngineCommand = serde_json::from_str(json).unwrap();
        match back {
            EngineCommand::SetPrefillPolicy {
                chunk_size,
                yield_ms,
                cpu_chunk_size,
            } => {
                assert_eq!(chunk_size, Some(64));
                assert_eq!(yield_ms, None);
                assert_eq!(cpu_chunk_size, None);
            }
            _ => panic!("Expected SetPrefillPolicy"),
        }
    }

    #[test]
    fn test_engine_command_serde_set_prefill_policy_empty() {
        // No fields — all None (valid: "update nothing, keep current")
        let json = r#"{"type":"set_prefill_policy"}"#;
        let back: EngineCommand = serde_json::from_str(json).unwrap();
        match back {
            EngineCommand::SetPrefillPolicy {
                chunk_size,
                yield_ms,
                cpu_chunk_size,
            } => {
                assert_eq!(chunk_size, None);
                assert_eq!(yield_ms, None);
                assert_eq!(cpu_chunk_size, None);
            }
            _ => panic!("Expected SetPrefillPolicy"),
        }
    }

    #[test]
    fn test_engine_directive_serde_roundtrip() {
        let directive = EngineDirective {
            seq_id: 42,
            commands: vec![
                EngineCommand::KvEvictH2o { keep_ratio: 0.48 },
                EngineCommand::Throttle { delay_ms: 30 },
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

    fn make_test_status() -> EngineStatus {
        EngineStatus {
            active_device: "cpu".to_string(),
            compute_level: ResourceLevel::Normal,
            actual_throughput: 15.0,
            memory_level: ResourceLevel::Normal,
            kv_cache_bytes: 1024 * 1024,
            kv_cache_tokens: 512,
            kv_cache_utilization: 0.25,
            memory_lossless_min: 1.0,
            memory_lossy_min: 0.01,
            state: EngineState::Running,
            tokens_generated: 100,
            available_actions: vec!["throttle".into(), "kv_evict_h2o".into()],
            active_actions: vec!["throttle".into()],
            eviction_policy: "none".into(),
            kv_dtype: "f16".into(),
            skip_ratio: 0.0,
            phase: "decode".into(),
            prefill_pos: 0,
            prefill_total: 0,
            partition_ratio: 0.0,
            self_cpu_pct: 0.0,
            self_gpu_pct: 0.0,
        }
    }

    #[test]
    fn test_engine_status_serde() {
        let status = make_test_status();
        let json = serde_json::to_string(&status).unwrap();
        let back: EngineStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back.state, EngineState::Running);
        assert!((back.actual_throughput - 15.0).abs() < f32::EPSILON);
        assert_eq!(back.kv_cache_tokens, 512);
        assert_eq!(back.available_actions, vec!["throttle", "kv_evict_h2o"]);
        assert_eq!(back.active_actions, vec!["throttle"]);
        assert_eq!(back.eviction_policy, "none");
        assert_eq!(back.kv_dtype, "f16");
        assert!((back.skip_ratio - 0.0).abs() < f32::EPSILON);
        assert_eq!(back.phase, "decode");
        assert_eq!(back.prefill_pos, 0);
        assert_eq!(back.prefill_total, 0);
    }

    #[test]
    fn test_engine_status_prefill_phase() {
        let mut status = make_test_status();
        status.phase = "prefill".into();
        status.prefill_pos = 420;
        status.prefill_total = 1073;
        let json = serde_json::to_string(&status).unwrap();
        let back: EngineStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back.phase, "prefill");
        assert_eq!(back.prefill_pos, 420);
        assert_eq!(back.prefill_total, 1073);
    }

    #[test]
    fn test_engine_status_new_fields_default_on_missing() {
        // Backward compat: old JSON without new fields should deserialize with defaults
        let old_json = r#"{
            "active_device":"cpu","compute_level":"normal","actual_throughput":10.0,
            "memory_level":"normal","kv_cache_bytes":0,"kv_cache_tokens":0,
            "kv_cache_utilization":0.0,"memory_lossless_min":1.0,"memory_lossy_min":0.01,
            "state":"running","tokens_generated":0
        }"#;
        let back: EngineStatus = serde_json::from_str(old_json).unwrap();
        assert!(back.available_actions.is_empty());
        assert!(back.active_actions.is_empty());
        assert_eq!(back.eviction_policy, "");
        assert_eq!(back.kv_dtype, "");
        assert!((back.skip_ratio - 0.0).abs() < f32::EPSILON);
        // New prefill fields default to empty/zero on old JSON
        assert_eq!(back.phase, "");
        assert_eq!(back.prefill_pos, 0);
        assert_eq!(back.prefill_total, 0);
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
        let msg = EngineMessage::Heartbeat(make_test_status());
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"heartbeat\""));

        // Response
        let msg = EngineMessage::Response(CommandResponse {
            seq_id: 1,
            results: vec![CommandResult::Ok],
        });
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"response\""));

        // QcfEstimate
        let msg = EngineMessage::QcfEstimate(QcfEstimate {
            estimates: {
                let mut m = HashMap::new();
                m.insert("kv_evict_h2o".to_string(), 0.1);
                m
            },
        });
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"qcf_estimate\""));
        let back: EngineMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, EngineMessage::QcfEstimate(_)));
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
