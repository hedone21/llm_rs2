//! MSG-010 ~ MSG-100: Shared Protocol Types Serde + Ordering
//!
//! 커버: MSG-010, MSG-011, MSG-020, MSG-030 (EngineCommand 13종 serde)
//! ManagerMessage/EngineMessage serde, EngineDirective/EngineCommand 10종 serde,
//! EngineCapability, EngineStatus (backward compat), CommandResult/CommandResponse,
//! ResourceLevel ordering, EngineState serde, Level ordering,
//! RecommendedBackend from_dbus_str, SystemSignal serde.

use llm_shared::*;

// ── 헬퍼 ──

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

// ══════════════════════════════════════════════════════════════
// MSG-010/011: ManagerMessage + EngineMessage serde
// ══════════════════════════════════════════════════════════════

#[test]
fn test_msg_010_manager_message_serde_roundtrip() {
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
fn test_msg_011_engine_message_serde_variants() {
    // Capability
    let msg = EngineMessage::Capability(EngineCapability {
        available_devices: vec!["cpu".into()],
        active_device: "cpu".into(),
        max_kv_tokens: 1024,
        bytes_per_kv_token: 128,
        num_layers: 8,
        ..Default::default()
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
            let mut m = std::collections::HashMap::new();
            m.insert("kv_evict_h2o".to_string(), 0.1);
            m
        },
        layer_swap: None,
    });
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"qcf_estimate\""));
    let back: EngineMessage = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, EngineMessage::QcfEstimate(_)));
}

// ══════════════════════════════════════════════════════════════
// MSG-020/021: EngineDirective + EngineCommand serde (10종)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_msg_020_engine_directive_serde_roundtrip() {
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
fn test_msg_021_engine_command_serde_throttle() {
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
fn test_msg_021_engine_command_serde_layer_skip() {
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
fn test_msg_021_engine_command_serde_kv_evict_h2o() {
    let cmd = EngineCommand::KvEvictH2o { keep_ratio: 0.48 };
    let json = serde_json::to_string(&cmd).unwrap();
    assert!(json.contains("\"type\":\"kv_evict_h2o\""));
}

#[test]
fn test_msg_021_engine_command_serde_kv_evict_sliding() {
    let cmd = EngineCommand::KvEvictSliding { keep_ratio: 0.6 };
    let json = serde_json::to_string(&cmd).unwrap();
    assert!(json.contains("\"type\":\"kv_evict_sliding\""));
}

#[test]
fn test_msg_021_engine_command_serde_kv_streaming() {
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
fn test_msg_021_engine_command_serde_kv_merge_d2o() {
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
fn test_msg_021_engine_command_serde_kv_quant_dynamic() {
    let cmd = EngineCommand::KvQuantDynamic { target_bits: 4 };
    let json = serde_json::to_string(&cmd).unwrap();
    assert!(json.contains("\"type\":\"kv_quant_dynamic\""));
}

#[test]
fn test_msg_036b_engine_command_serde_request_qcf() {
    let cmd = EngineCommand::RequestQcf;
    let json = serde_json::to_string(&cmd).unwrap();
    assert!(json.contains("\"type\":\"request_qcf\""));
    let back: EngineCommand = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, EngineCommand::RequestQcf));
}

#[test]
fn test_msg_021_engine_command_serde_restore_defaults() {
    let cmd = EngineCommand::RestoreDefaults;
    let json = serde_json::to_string(&cmd).unwrap();
    assert!(json.contains("\"type\":\"restore_defaults\""));
    let back: EngineCommand = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, EngineCommand::RestoreDefaults));
}

#[test]
fn test_msg_021_engine_command_serde_switch_hw() {
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
fn test_msg_021_engine_command_serde_suspend_resume() {
    let suspend = EngineCommand::Suspend;
    let json = serde_json::to_string(&suspend).unwrap();
    assert!(json.contains("\"type\":\"suspend\""));

    let resume = EngineCommand::Resume;
    let json = serde_json::to_string(&resume).unwrap();
    assert!(json.contains("\"type\":\"resume\""));
    let back: EngineCommand = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, EngineCommand::Resume));
}

// ══════════════════════════════════════════════════════════════
// MSG-050: EngineCapability serde
// ══════════════════════════════════════════════════════════════

#[test]
fn test_msg_050_engine_capability_serde() {
    let cap = EngineCapability {
        available_devices: vec!["cpu".to_string(), "opencl".to_string()],
        active_device: "cpu".to_string(),
        max_kv_tokens: 2048,
        bytes_per_kv_token: 256,
        num_layers: 16,
        ..Default::default()
    };
    let json = serde_json::to_string(&cap).unwrap();
    let back: EngineCapability = serde_json::from_str(&json).unwrap();
    assert_eq!(back.available_devices.len(), 2);
    assert_eq!(back.max_kv_tokens, 2048);
    assert_eq!(back.num_layers, 16);
}

// ══════════════════════════════════════════════════════════════
// MSG-060/061: EngineStatus serde + backward compat
// ══════════════════════════════════════════════════════════════

#[test]
fn test_msg_060_engine_status_serde() {
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
}

#[test]
fn test_msg_061_engine_status_new_fields_default_on_missing() {
    // 이전 버전 JSON (새 필드 없음) → 기본값으로 역직렬화
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
}

// ══════════════════════════════════════════════════════════════
// MSG-080: CommandResult serde
// ══════════════════════════════════════════════════════════════

#[test]
fn test_msg_080_command_result_serde() {
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

// ══════════════════════════════════════════════════════════════
// MSG-070: CommandResponse serde
// ══════════════════════════════════════════════════════════════

#[test]
fn test_msg_070_command_response_serde() {
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

// ══════════════════════════════════════════════════════════════
// MSG-090: ResourceLevel ordering
// ══════════════════════════════════════════════════════════════

#[test]
fn test_msg_090_resource_level_ordering() {
    assert!(ResourceLevel::Normal < ResourceLevel::Warning);
    assert!(ResourceLevel::Warning < ResourceLevel::Critical);
    assert!(ResourceLevel::Normal < ResourceLevel::Critical);
}

// ══════════════════════════════════════════════════════════════
// MSG-091: EngineState serde
// ══════════════════════════════════════════════════════════════

#[test]
fn test_msg_091_engine_state_serde() {
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

// ══════════════════════════════════════════════════════════════
// MSG-092: Level ordering (4-level)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_msg_092_level_ordering() {
    assert!(Level::Normal < Level::Warning);
    assert!(Level::Warning < Level::Critical);
    assert!(Level::Critical < Level::Emergency);
    assert!(Level::Normal < Level::Emergency);
}

// ══════════════════════════════════════════════════════════════
// MSG-093: RecommendedBackend from_dbus_str
// ══════════════════════════════════════════════════════════════

#[test]
fn test_msg_093_recommended_backend_from_dbus_str() {
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

// ══════════════════════════════════════════════════════════════
// MSG-100: SystemSignal serde roundtrip
// ══════════════════════════════════════════════════════════════

#[test]
fn test_msg_100_system_signal_serde_roundtrip() {
    let sig = SystemSignal::MemoryPressure {
        level: Level::Critical,
        available_bytes: 1024,
        total_bytes: 4096,
        reclaim_target_bytes: 512,
    };
    let json = serde_json::to_string(&sig).unwrap();
    let back: SystemSignal = serde_json::from_str(&json).unwrap();
    assert_eq!(back.level(), Level::Critical);

    // 다른 variant도 확인
    let sig = SystemSignal::ThermalAlert {
        level: Level::Emergency,
        temperature_mc: 95000,
        throttling_active: true,
        throttle_ratio: 0.3,
    };
    let json = serde_json::to_string(&sig).unwrap();
    let back: SystemSignal = serde_json::from_str(&json).unwrap();
    assert_eq!(back.level(), Level::Emergency);
}

// ══════════════════════════════════════════════════════════════
// MSG-085/086/087: QcfEstimate serde + constraints
// ══════════════════════════════════════════════════════════════

#[test]
fn test_msg_085_qcf_estimate_serde_roundtrip() {
    use std::collections::HashMap;
    let mut estimates = HashMap::new();
    estimates.insert("kv_evict_h2o".to_string(), 0.12f32);
    estimates.insert("kv_evict_sliding".to_string(), 0.18f32);
    estimates.insert("layer_skip".to_string(), 0.35f32);

    let qcf = QcfEstimate {
        estimates,
        layer_swap: None,
    };
    let msg = EngineMessage::QcfEstimate(qcf);
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"type\":\"qcf_estimate\""));
    assert!(json.contains("\"estimates\""));

    let back: EngineMessage = serde_json::from_str(&json).unwrap();
    match back {
        EngineMessage::QcfEstimate(q) => {
            assert_eq!(q.estimates.len(), 3);
            assert!((q.estimates["kv_evict_h2o"] - 0.12).abs() < f32::EPSILON);
            assert!((q.estimates["kv_evict_sliding"] - 0.18).abs() < f32::EPSILON);
            assert!((q.estimates["layer_skip"] - 0.35).abs() < f32::EPSILON);
        }
        _ => panic!("Expected QcfEstimate"),
    }
}

#[test]
fn test_msg_086_qcf_estimate_empty_estimates() {
    // MSG-086: Engine이 계산 불가 시 빈 map 반환 가능
    use std::collections::HashMap;
    let qcf = QcfEstimate {
        estimates: HashMap::new(),
        layer_swap: None,
    };
    let msg = EngineMessage::QcfEstimate(qcf);
    let json = serde_json::to_string(&msg).unwrap();
    let back: EngineMessage = serde_json::from_str(&json).unwrap();
    match back {
        EngineMessage::QcfEstimate(q) => {
            assert!(q.estimates.is_empty());
        }
        _ => panic!("Expected QcfEstimate"),
    }
}

#[test]
fn test_msg_087_qcf_values_non_negative() {
    // MSG-087: QCF 값은 >= 0.0, 0.0 = 저하 없음
    use std::collections::HashMap;
    let mut estimates = HashMap::new();
    estimates.insert("kv_evict_h2o".to_string(), 0.0f32);
    estimates.insert("layer_skip".to_string(), 1.5f32);

    let qcf = QcfEstimate {
        estimates,
        layer_swap: None,
    };
    // 모든 값 검증
    for v in qcf.estimates.values() {
        assert!(*v >= 0.0, "QCF value must be >= 0.0, got {}", v);
    }
}
