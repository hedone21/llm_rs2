//! Integration tests for Command Executor — generate.rs integration logic.
//!
//! These tests verify the resilience checkpoint behavior that runs inside
//! the token generation loop: directive → poll → execution plan → state change.
//!
//! Run with: `cargo test --test test_resilience_integration`

use std::sync::mpsc;
use std::time::Duration;

use llm_rs2::resilience::{CommandExecutor, KVSnapshot, MessageLoop, MockTransport, ResourceLevel};
use llm_shared::{EngineCommand, EngineDirective, EngineMessage, EngineState, ManagerMessage};

// ── Helpers ───────────────────────────────────────────────

fn make_executor() -> (
    CommandExecutor,
    mpsc::Sender<ManagerMessage>,
    mpsc::Receiver<EngineMessage>,
) {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (resp_tx, resp_rx) = mpsc::channel();
    let executor = CommandExecutor::new(
        cmd_rx,
        resp_tx,
        "cpu".to_string(),
        Duration::from_secs(60), // Long interval to avoid heartbeat noise in tests
    );
    (executor, cmd_tx, resp_rx)
}

fn empty_snap() -> KVSnapshot {
    KVSnapshot::default()
}

fn snap(tokens: usize, capacity: usize, prefix: usize) -> KVSnapshot {
    KVSnapshot {
        total_bytes: (tokens * 256) as u64,
        total_tokens: tokens,
        capacity,
        protected_prefix: prefix,
    }
}

fn send_directive(tx: &mpsc::Sender<ManagerMessage>, seq_id: u64, commands: Vec<EngineCommand>) {
    tx.send(ManagerMessage::Directive(EngineDirective {
        seq_id,
        commands,
    }))
    .unwrap();
}

// ── Test: Eviction flow ──────────────────────────────────

#[test]
fn test_resilience_eviction_flow() {
    let (mut executor, tx, rx) = make_executor();

    // Send SetMemoryLevel(Critical) → should produce EvictPlan
    send_directive(
        &tx,
        1,
        vec![EngineCommand::SetMemoryLevel {
            level: ResourceLevel::Critical,
            target_ratio: 0.5,
            deadline_ms: Some(1000),
        }],
    );

    let plan = executor.poll(&snap(500, 2048, 4));
    assert!(
        plan.evict.is_some(),
        "Should produce evict plan for Critical memory"
    );

    let evict = plan.evict.unwrap();
    assert!((evict.target_ratio - 0.5).abs() < f32::EPSILON);
    assert_eq!(evict.level, ResourceLevel::Critical);

    // Simulate KV cache eviction (like generate.rs does)
    let current_pos: usize = 500;
    let target_len = (current_pos as f32 * evict.target_ratio) as usize;
    let remove = current_pos.saturating_sub(target_len);
    assert!(remove > 0, "Should remove some tokens");

    let new_pos = current_pos - remove;
    assert!(
        new_pos < current_pos,
        "Position should decrease after eviction"
    );

    // Verify response was sent
    let resp = rx.recv().unwrap();
    assert!(matches!(resp, EngineMessage::Response(_)));
}

// ── Test: Throttle flow ──────────────────────────────────

#[test]
fn test_resilience_throttle_flow() {
    let (mut executor, tx, _rx) = make_executor();

    // SetComputeLevel(Warning) → should produce throttle delay
    send_directive(
        &tx,
        1,
        vec![EngineCommand::SetComputeLevel {
            level: ResourceLevel::Warning,
            target_throughput: 0.7,
            deadline_ms: None,
        }],
    );

    let plan = executor.poll(&empty_snap());
    assert!(
        plan.throttle_delay_ms > 0,
        "Throttle delay should be set, got {}",
        plan.throttle_delay_ms
    );
    assert!(!plan.suspended, "Should not be suspended on Warning");
    assert_eq!(executor.compute_level(), ResourceLevel::Warning);
}

// ── Test: Suspend flow ───────────────────────────────────

#[test]
fn test_resilience_suspend_flow() {
    let (mut executor, tx, _rx) = make_executor();

    // Suspend command
    send_directive(&tx, 1, vec![EngineCommand::Suspend]);

    let plan = executor.poll(&empty_snap());
    assert!(plan.suspended, "Should be suspended after Suspend command");
    assert_eq!(executor.state(), EngineState::Suspended);

    // Suspend should override any other plan fields
    assert!(plan.evict.is_none());
    assert!(plan.switch_device.is_none());
}

// ── Test: Disabled resilience is noop ────────────────────

#[test]
fn test_resilience_disabled_noop() {
    // When command_executor is None, no poll happens.
    let command_executor: Option<CommandExecutor> = None;

    let mut throttle_delay_ms = 0u64;
    if let Some(_ex) = &command_executor {
        throttle_delay_ms = 999;
    }

    assert_eq!(
        throttle_delay_ms, 0,
        "Disabled resilience should not affect state"
    );
}

// ── Test: Resume clears constraints ──────────────────────

#[test]
fn test_resilience_resume_clears_constraints() {
    let (mut executor, tx, _rx) = make_executor();

    // First: set to critical
    send_directive(
        &tx,
        1,
        vec![EngineCommand::SetComputeLevel {
            level: ResourceLevel::Critical,
            target_throughput: 0.3,
            deadline_ms: None,
        }],
    );
    executor.poll(&empty_snap());
    assert!(executor.throttle_delay_ms() > 0, "Throttle should be set");

    // Then: suspend
    send_directive(&tx, 2, vec![EngineCommand::Suspend]);
    executor.poll(&empty_snap());
    assert_eq!(executor.state(), EngineState::Suspended);

    // Resume should restore Normal
    send_directive(&tx, 3, vec![EngineCommand::Resume]);
    let plan = executor.poll(&empty_snap());
    assert!(plan.resumed);
    assert_eq!(executor.state(), EngineState::Running);
    assert_eq!(executor.compute_level(), ResourceLevel::Normal);
    assert_eq!(executor.memory_level(), ResourceLevel::Normal);
    assert_eq!(executor.throttle_delay_ms(), 0);
}

// ── Test: Channel disconnect is graceful ─────────────────

#[test]
fn test_resilience_channel_disconnect_graceful() {
    let (mut executor, tx, _rx) = make_executor();

    // Send directive then drop sender (simulates transport crash)
    send_directive(
        &tx,
        1,
        vec![EngineCommand::SetMemoryLevel {
            level: ResourceLevel::Warning,
            target_ratio: 0.85,
            deadline_ms: None,
        }],
    );
    drop(tx);

    // First poll: processes buffered directive
    let plan = executor.poll(&empty_snap());
    assert!(plan.evict.is_some());
    assert_eq!(executor.memory_level(), ResourceLevel::Warning);

    // Second poll: channel dead, no panic, state preserved
    let plan = executor.poll(&empty_snap());
    assert!(plan.evict.is_none()); // No new commands
    assert_eq!(executor.memory_level(), ResourceLevel::Warning);
}

// ── Transport integration tests ──────────────────────────

#[test]
fn test_mock_transport_e2e() {
    let messages = vec![
        ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::SetMemoryLevel {
                level: ResourceLevel::Warning,
                target_ratio: 0.85,
                deadline_ms: None,
            }],
        }),
        ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::SetComputeLevel {
                level: ResourceLevel::Critical,
                target_throughput: 0.3,
                deadline_ms: Some(1000),
            }],
        }),
    ];
    let transport = MockTransport::from_messages(messages);
    let (cmd_rx, resp_tx, _handle) = MessageLoop::spawn(transport).unwrap();

    let mut executor =
        CommandExecutor::new(cmd_rx, resp_tx, "cpu".to_string(), Duration::from_secs(60));

    // Wait for MessageLoop to forward messages
    std::thread::sleep(Duration::from_millis(50));

    let plan = executor.poll(&snap(500, 2048, 4));
    assert!(
        plan.evict.is_some() || plan.throttle_delay_ms > 0,
        "Should have actions from mock transport pipeline"
    );
}

#[cfg(unix)]
#[test]
fn test_unix_socket_e2e() {
    use llm_rs2::resilience::{Transport, UnixSocketTransport};
    use std::io::Write;
    use std::os::unix::net::UnixListener;

    let path = std::env::temp_dir().join(format!(
        "llm_rs_e2e_{}.sock",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let listener = UnixListener::bind(&path).unwrap();

    let path2 = path.clone();
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (resp_tx, _resp_rx) = mpsc::channel();

    std::thread::spawn(move || {
        let mut transport = UnixSocketTransport::new(path2);
        if transport.connect().is_err() {
            return;
        }
        loop {
            match transport.recv() {
                Ok(msg) => {
                    if cmd_tx.send(msg).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    let (mut server_stream, _) = listener.accept().unwrap();
    let msg = ManagerMessage::Directive(EngineDirective {
        seq_id: 1,
        commands: vec![EngineCommand::SetMemoryLevel {
            level: ResourceLevel::Critical,
            target_ratio: 0.5,
            deadline_ms: None,
        }],
    });
    let json = serde_json::to_vec(&msg).unwrap();
    let len = (json.len() as u32).to_be_bytes();
    server_stream.write_all(&len).unwrap();
    server_stream.write_all(&json).unwrap();
    server_stream.flush().unwrap();
    drop(server_stream);

    let mut executor =
        CommandExecutor::new(cmd_rx, resp_tx, "cpu".to_string(), Duration::from_secs(60));

    std::thread::sleep(Duration::from_millis(100));

    let plan = executor.poll(&snap(500, 2048, 4));
    assert!(
        plan.evict.is_some(),
        "Should have evict plan from unix socket pipeline"
    );
    std::fs::remove_file(&path).ok();
}

// ── Test: Superseding directives ─────────────────────────

#[test]
fn test_resilience_superseding_directives() {
    let (mut executor, tx, _rx) = make_executor();

    // Memory Warning, then Memory Critical in quick succession
    send_directive(
        &tx,
        1,
        vec![EngineCommand::SetMemoryLevel {
            level: ResourceLevel::Warning,
            target_ratio: 0.85,
            deadline_ms: None,
        }],
    );
    send_directive(
        &tx,
        2,
        vec![EngineCommand::SetMemoryLevel {
            level: ResourceLevel::Critical,
            target_ratio: 0.5,
            deadline_ms: None,
        }],
    );

    let plan = executor.poll(&empty_snap());
    // The second (more severe) should supersede
    let evict = plan.evict.unwrap();
    assert!((evict.target_ratio - 0.5).abs() < f32::EPSILON);
    assert_eq!(evict.level, ResourceLevel::Critical);
}

// ── Test: Multi-domain directives ────────────────────────

#[test]
fn test_resilience_multi_domain_directive() {
    let (mut executor, tx, _rx) = make_executor();

    // Single directive with both compute and memory commands
    send_directive(
        &tx,
        1,
        vec![
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
    );

    let plan = executor.poll(&empty_snap());
    assert!(plan.evict.is_some(), "Should have evict plan");
    assert!(plan.throttle_delay_ms > 0, "Should have throttle");
}

// ── Test: Suspend overrides everything ───────────────────

#[test]
fn test_resilience_suspend_overrides_all() {
    let (mut executor, tx, _rx) = make_executor();

    // Memory + Compute + Suspend in one directive
    send_directive(
        &tx,
        1,
        vec![
            EngineCommand::SetMemoryLevel {
                level: ResourceLevel::Critical,
                target_ratio: 0.5,
                deadline_ms: None,
            },
            EngineCommand::SwitchComputeUnit {
                device: "gpu".to_string(),
            },
            EngineCommand::Suspend,
        ],
    );

    let plan = executor.poll(&empty_snap());
    assert!(plan.suspended, "Suspend should be set");
    assert!(plan.evict.is_none(), "Suspend should clear evict");
    assert!(plan.switch_device.is_none(), "Suspend should clear switch");
}

// ── Test: Rapid signal buffering ─────────────────────────

#[test]
fn test_resilience_signal_buffering() {
    let (mut executor, tx, _rx) = make_executor();

    // Send 50 directives rapidly
    for i in 0..50 {
        let level = match i % 3 {
            0 => ResourceLevel::Normal,
            1 => ResourceLevel::Warning,
            _ => ResourceLevel::Critical,
        };
        send_directive(
            &tx,
            i + 1,
            vec![EngineCommand::SetMemoryLevel {
                level,
                target_ratio: match level {
                    ResourceLevel::Normal => 1.0,
                    ResourceLevel::Warning => 0.85,
                    ResourceLevel::Critical => 0.5,
                },
                deadline_ms: None,
            }],
        );
    }

    // poll() should drain all without panic
    let plan = executor.poll(&empty_snap());
    // Last directive: i=49, 49%3=1 → Warning
    assert_eq!(executor.memory_level(), ResourceLevel::Warning);
    // Should have an evict plan from the last superseding command
    assert!(plan.evict.is_some());
}
