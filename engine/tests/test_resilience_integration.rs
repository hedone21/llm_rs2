//! Integration tests for Resilience Manager — generate.rs integration logic.
//!
//! These tests verify the resilience checkpoint behavior that runs inside
//! the token generation loop: signal → poll → action → state change.
//!
//! Run with: `cargo test --test test_resilience_integration`

use std::sync::mpsc;

use llm_rs2::resilience::signal::{EnergyReason, Level, SystemSignal};
use llm_rs2::resilience::state::OperatingMode;
use llm_rs2::resilience::{
    InferenceContext, MockTransport, ResilienceAction, ResilienceManager, SignalListener,
    execute_action,
};

// ── Helpers ───────────────────────────────────────────────

fn send(tx: &mpsc::Sender<SystemSignal>, signal: SystemSignal) {
    tx.send(signal).unwrap();
}

fn make_ctx<'a>(
    num_tokens: &'a mut usize,
    throttle: &'a mut u64,
    suspended: &'a mut bool,
    reject: &'a mut bool,
) -> InferenceContext<'a> {
    InferenceContext {
        max_tokens: num_tokens,
        throttle_delay_ms: throttle,
        suspended,
        reject_new: reject,
    }
}

// ── Test: Eviction flow ──────────────────────────────────

#[test]
fn test_resilience_eviction_flow() {
    let (tx, rx) = mpsc::channel();
    let mut mgr = ResilienceManager::new(rx);

    // Send MemoryPressure(Critical) → should produce Evict action
    send(
        &tx,
        SystemSignal::MemoryPressure {
            level: Level::Critical,
            available_bytes: 50 * 1024 * 1024,
            reclaim_target_bytes: 100 * 1024 * 1024,
        },
    );

    let actions = mgr.poll();
    assert!(
        !actions.is_empty(),
        "Should produce actions for Critical memory"
    );

    // Find the Evict action
    let evict = actions
        .iter()
        .find(|a| matches!(a, ResilienceAction::Evict { .. }));
    assert!(evict.is_some(), "Should contain an Evict action");

    if let Some(ResilienceAction::Evict { target_ratio }) = evict {
        assert!(
            *target_ratio > 0.0 && *target_ratio < 1.0,
            "target_ratio should be between 0 and 1, got {}",
            target_ratio
        );

        // Simulate KV cache eviction (like generate.rs does)
        let current_pos: usize = 500;
        let target_len = (current_pos as f32 * target_ratio) as usize;
        let remove = current_pos.saturating_sub(target_len);
        assert!(remove > 0, "Should remove some tokens");

        let new_pos = current_pos - remove;
        assert!(
            new_pos < current_pos,
            "Position should decrease after eviction"
        );
    }
}

// ── Test: Throttle flow ──────────────────────────────────

#[test]
fn test_resilience_throttle_flow() {
    let (tx, rx) = mpsc::channel();
    let mut mgr = ResilienceManager::new(rx);

    // ThermalAlert(Critical) → should produce Throttle
    send(
        &tx,
        SystemSignal::ThermalAlert {
            level: Level::Critical,
            temperature_mc: 80000,
            throttling_active: true,
            throttle_ratio: 0.5,
        },
    );

    let actions = mgr.poll();
    assert!(!actions.is_empty());

    let mut num_tokens = 128usize;
    let mut throttle_delay_ms = 0u64;
    let mut suspended = false;
    let mut reject_new = false;
    let mut ctx = make_ctx(
        &mut num_tokens,
        &mut throttle_delay_ms,
        &mut suspended,
        &mut reject_new,
    );

    for action in &actions {
        if !matches!(action, ResilienceAction::Evict { .. }) {
            execute_action(action, &mut ctx);
        }
    }

    assert!(
        throttle_delay_ms > 0,
        "Throttle delay should be set, got {}",
        throttle_delay_ms
    );
    assert!(!suspended, "Should not be suspended on Warning");
}

// ── Test: Suspend flow ───────────────────────────────────

#[test]
fn test_resilience_suspend_flow() {
    let (tx, rx) = mpsc::channel();
    let mut mgr = ResilienceManager::new(rx);

    // EnergyConstraint(Emergency) → should produce Suspend
    send(
        &tx,
        SystemSignal::EnergyConstraint {
            level: Level::Emergency,
            reason: EnergyReason::BatteryCritical,
            power_budget_mw: 0,
        },
    );

    let actions = mgr.poll();
    assert!(!actions.is_empty());
    assert_eq!(mgr.mode(), OperatingMode::Suspended);

    let mut num_tokens = 128usize;
    let mut throttle_delay_ms = 0u64;
    let mut suspended = false;
    let mut reject_new = false;
    let mut ctx = make_ctx(
        &mut num_tokens,
        &mut throttle_delay_ms,
        &mut suspended,
        &mut reject_new,
    );

    for action in &actions {
        execute_action(action, &mut ctx);
    }

    assert!(
        suspended,
        "Should be suspended after Emergency energy signal"
    );
}

// ── Test: Disabled resilience is noop ────────────────────

#[test]
fn test_resilience_disabled_noop() {
    // When resilience_manager is None, no poll happens.
    // This test verifies the Option<ResilienceManager> pattern.
    let resilience_manager: Option<ResilienceManager> = None;

    // Simulate the generate.rs pattern
    let mut throttle_delay_ms = 0u64;
    if let Some(_rm) = &resilience_manager {
        // This block should never execute
        throttle_delay_ms = 999;
    }

    assert_eq!(
        throttle_delay_ms, 0,
        "Disabled resilience should not affect state"
    );
}

// ── Test: RestoreDefaults clears constraints ─────────────

#[test]
fn test_resilience_restore_defaults_flow() {
    let (tx, rx) = mpsc::channel();
    let mut mgr = ResilienceManager::new(rx);

    // First: Critical → set throttle
    send(
        &tx,
        SystemSignal::ThermalAlert {
            level: Level::Critical,
            temperature_mc: 80000,
            throttling_active: true,
            throttle_ratio: 0.5,
        },
    );

    let actions = mgr.poll();
    let mut num_tokens = 128usize;
    let mut throttle_delay_ms = 0u64;
    let mut suspended = false;
    let mut reject_new = false;

    {
        let mut ctx = make_ctx(
            &mut num_tokens,
            &mut throttle_delay_ms,
            &mut suspended,
            &mut reject_new,
        );
        for action in &actions {
            if !matches!(action, ResilienceAction::Evict { .. }) {
                execute_action(action, &mut ctx);
            }
        }
    }
    assert!(throttle_delay_ms > 0, "Throttle should be set");

    // Then: all Normal → RestoreDefaults
    send(
        &tx,
        SystemSignal::ThermalAlert {
            level: Level::Normal,
            temperature_mc: 40000,
            throttling_active: false,
            throttle_ratio: 1.0,
        },
    );

    let actions = mgr.poll();
    {
        let mut ctx = make_ctx(
            &mut num_tokens,
            &mut throttle_delay_ms,
            &mut suspended,
            &mut reject_new,
        );
        for action in &actions {
            execute_action(action, &mut ctx);
        }
    }

    assert_eq!(
        throttle_delay_ms, 0,
        "RestoreDefaults should clear throttle"
    );
    assert!(!reject_new, "RestoreDefaults should clear reject_new");
}

// ── Test: Channel disconnect is graceful ─────────────────

#[test]
fn test_resilience_channel_disconnect_graceful() {
    let (tx, rx) = mpsc::channel();
    let mut mgr = ResilienceManager::new(rx);

    // Send signal then drop sender (simulates listener crash)
    send(
        &tx,
        SystemSignal::MemoryPressure {
            level: Level::Warning,
            available_bytes: 200 * 1024 * 1024,
            reclaim_target_bytes: 50 * 1024 * 1024,
        },
    );
    drop(tx);

    // First poll: processes buffered signal
    let actions = mgr.poll();
    assert!(!actions.is_empty());
    assert_eq!(mgr.mode(), OperatingMode::Degraded);

    // Second poll: channel dead, no panic, state preserved
    let actions = mgr.poll();
    assert!(actions.is_empty());
    assert_eq!(mgr.mode(), OperatingMode::Degraded);
}

// ── Test: LimitTokens takes minimum ──────────────────────

#[test]
fn test_resilience_limit_tokens_takes_minimum() {
    let mut num_tokens = 200usize;
    let mut throttle_delay_ms = 0u64;
    let mut suspended = false;
    let mut reject_new = false;

    {
        let mut ctx = make_ctx(
            &mut num_tokens,
            &mut throttle_delay_ms,
            &mut suspended,
            &mut reject_new,
        );
        execute_action(&ResilienceAction::LimitTokens { max_tokens: 100 }, &mut ctx);
    }
    assert_eq!(num_tokens, 100);

    // Applying a higher limit should not increase
    {
        let mut ctx = make_ctx(
            &mut num_tokens,
            &mut throttle_delay_ms,
            &mut suspended,
            &mut reject_new,
        );
        execute_action(&ResilienceAction::LimitTokens { max_tokens: 300 }, &mut ctx);
    }
    assert_eq!(num_tokens, 100, "Should keep the lower limit");
}

// ── Transport integration tests ──────────────────────────

#[test]
fn test_mock_transport_e2e() {
    let signals = vec![
        SystemSignal::MemoryPressure {
            level: Level::Warning,
            available_bytes: 200_000_000,
            reclaim_target_bytes: 50_000_000,
        },
        SystemSignal::ThermalAlert {
            level: Level::Critical,
            temperature_mc: 80000,
            throttling_active: true,
            throttle_ratio: 0.5,
        },
    ];
    let transport = MockTransport::from_signals(signals);
    let (tx, rx) = mpsc::channel();
    let handle = SignalListener::new(transport, tx).spawn();

    let mut mgr = ResilienceManager::new(rx);

    // Wait for listener thread to send signals
    handle.join().unwrap();

    let actions = mgr.poll();
    assert!(
        !actions.is_empty(),
        "Should have actions from mock transport pipeline"
    );
}

#[cfg(unix)]
#[test]
fn test_unix_socket_e2e() {
    use llm_rs2::resilience::UnixSocketTransport;
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
    let (tx, rx) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        let transport = UnixSocketTransport::new(path2);
        SignalListener::new(transport, tx).spawn().join().unwrap();
    });

    let (mut server_stream, _) = listener.accept().unwrap();
    let sig = SystemSignal::MemoryPressure {
        level: Level::Critical,
        available_bytes: 50_000_000,
        reclaim_target_bytes: 100_000_000,
    };
    // Write length-prefixed JSON
    let json = serde_json::to_vec(&sig).unwrap();
    use std::io::Write;
    let len = (json.len() as u32).to_be_bytes();
    server_stream.write_all(&len).unwrap();
    server_stream.write_all(&json).unwrap();
    server_stream.flush().unwrap();
    drop(server_stream);

    let mut mgr = ResilienceManager::new(rx);
    handle.join().unwrap();

    let actions = mgr.poll();
    assert!(
        !actions.is_empty(),
        "Should have actions from unix socket pipeline"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn test_transport_failure_graceful() {
    // Create a transport that fails to connect (non-existent socket path)
    // The listener should exit gracefully and the manager should return empty polls.
    let (tx, rx) = mpsc::channel();

    #[cfg(unix)]
    {
        use llm_rs2::resilience::UnixSocketTransport;
        let transport = UnixSocketTransport::new(std::path::PathBuf::from(
            "/tmp/nonexistent_llm_e2e_test.sock",
        ));
        let handle = SignalListener::new(transport, tx).spawn();
        handle.join().unwrap();
    }

    #[cfg(not(unix))]
    {
        drop(tx);
    }

    let mut mgr = ResilienceManager::new(rx);
    let actions = mgr.poll();
    assert!(
        actions.is_empty(),
        "Failed transport should produce no actions"
    );
}
