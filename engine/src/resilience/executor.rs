use std::sync::mpsc;
use std::time::{Duration, Instant};

use llm_shared::{
    CommandResponse, CommandResult, EngineCapability, EngineCommand, EngineMessage, EngineState,
    EngineStatus, ManagerMessage, ResourceLevel,
};

// ── Public types ────────────────────────────────────────────

/// Plan produced by CommandExecutor::poll() for the inference loop to execute.
#[derive(Debug, Default)]
pub struct ExecutionPlan {
    /// KV cache eviction plan, if any.
    pub evict: Option<EvictPlan>,
    /// Device to switch to (e.g., "cpu", "gpu").
    pub switch_device: Option<String>,
    /// Device to pre-warm.
    pub prepare_device: Option<String>,
    /// Throttle delay between tokens (ms). 0 = no throttle.
    pub throttle_delay_ms: u64,
    /// Whether inference should be suspended.
    pub suspended: bool,
    /// Whether inference should resume from suspension.
    pub resumed: bool,
}

/// Eviction plan parameters.
#[derive(Debug, Clone)]
pub struct EvictPlan {
    /// Target ratio of cache to keep (0.0–1.0).
    pub target_ratio: f32,
    /// Resource level: Warning = lossless only, Critical = lossy OK.
    pub level: ResourceLevel,
}

/// Snapshot of KV cache state for status reporting.
#[derive(Debug, Clone, Default)]
pub struct KVSnapshot {
    pub total_bytes: u64,
    pub total_tokens: usize,
    pub capacity: usize,
    pub protected_prefix: usize,
}

// ── CommandExecutor ─────────────────────────────────────────

/// Receives ManagerMessages, builds ExecutionPlans for the inference loop.
/// No strategy logic — just translates commands to plans.
pub struct CommandExecutor {
    cmd_rx: mpsc::Receiver<ManagerMessage>,
    resp_tx: mpsc::Sender<EngineMessage>,

    // Current state
    compute_level: ResourceLevel,
    memory_level: ResourceLevel,
    engine_state: EngineState,
    active_device: String,
    throttle_delay_ms: u64,

    // Throughput tracking
    throughput_ema: f32,
    last_token_time: Option<Instant>,
    tokens_generated: usize,

    // Heartbeat
    last_heartbeat: Instant,
    heartbeat_interval: Duration,
}

impl CommandExecutor {
    pub fn new(
        cmd_rx: mpsc::Receiver<ManagerMessage>,
        resp_tx: mpsc::Sender<EngineMessage>,
        active_device: String,
        heartbeat_interval: Duration,
    ) -> Self {
        Self {
            cmd_rx,
            resp_tx,
            compute_level: ResourceLevel::Normal,
            memory_level: ResourceLevel::Normal,
            engine_state: EngineState::Idle,
            active_device,
            throttle_delay_ms: 0,
            throughput_ema: 0.0,
            last_token_time: None,
            tokens_generated: 0,
            last_heartbeat: Instant::now(),
            heartbeat_interval,
        }
    }

    /// Send initial capability report to Manager.
    pub fn send_capability(&self, cap: EngineCapability) {
        let _ = self.resp_tx.send(EngineMessage::Capability(cap));
    }

    /// Notify executor that inference has started.
    pub fn set_running(&mut self) {
        self.engine_state = EngineState::Running;
    }

    /// Record a generated token for throughput tracking.
    pub fn on_token_generated(&mut self) {
        let now = Instant::now();
        self.tokens_generated += 1;

        if let Some(last) = self.last_token_time {
            let elapsed = now.duration_since(last).as_secs_f32();
            if elapsed > 0.0 {
                let instant_tps = 1.0 / elapsed;
                const ALPHA: f32 = 0.1;
                if self.throughput_ema == 0.0 {
                    self.throughput_ema = instant_tps;
                } else {
                    self.throughput_ema = ALPHA * instant_tps + (1.0 - ALPHA) * self.throughput_ema;
                }
            }
        }
        self.last_token_time = Some(now);
    }

    /// Poll for pending commands and build an ExecutionPlan.
    /// Called once per token in the inference loop.
    pub fn poll(&mut self, kv_snap: &KVSnapshot) -> ExecutionPlan {
        let mut plan = ExecutionPlan::default();

        // 1. Heartbeat check
        if self.last_heartbeat.elapsed() >= self.heartbeat_interval {
            self.send_heartbeat(kv_snap);
            self.last_heartbeat = Instant::now();
        }

        // 2. Drain all pending messages
        let mut directives = Vec::new();
        while let Ok(msg) = self.cmd_rx.try_recv() {
            match msg {
                ManagerMessage::Directive(d) => directives.push(d),
            }
        }

        if directives.is_empty() {
            // Maintain existing throttle
            plan.throttle_delay_ms = self.throttle_delay_ms;
            return plan;
        }

        // 3. Process each directive, building plan and responses
        for directive in directives {
            let seq_id = directive.seq_id;
            let mut results = Vec::with_capacity(directive.commands.len());

            for cmd in &directive.commands {
                let result = self.apply_command(cmd, &mut plan);
                results.push(result);
            }

            // Send response
            let _ = self
                .resp_tx
                .send(EngineMessage::Response(CommandResponse { seq_id, results }));
        }

        // 4. Suspend overrides everything
        if plan.suspended {
            plan.evict = None;
            plan.switch_device = None;
            plan.prepare_device = None;
            plan.throttle_delay_ms = 0;
            plan.resumed = false;
            self.engine_state = EngineState::Suspended;
        }

        // Update internal throttle state
        self.throttle_delay_ms = plan.throttle_delay_ms;

        plan
    }

    fn apply_command(&mut self, cmd: &EngineCommand, plan: &mut ExecutionPlan) -> CommandResult {
        match cmd {
            EngineCommand::SetComputeLevel {
                level,
                target_throughput,
                ..
            } => {
                self.compute_level = *level;
                // Calculate throttle delay from target_throughput
                // target_throughput is a ratio (0.0-1.0) of max throughput
                if *target_throughput < 1.0 && *target_throughput > 0.0 {
                    // Simple mapping: lower throughput → longer delay
                    // At 0.7 → ~43ms, at 0.3 → ~233ms
                    let delay = ((1.0 / target_throughput - 1.0) * 100.0) as u64;
                    plan.throttle_delay_ms = plan.throttle_delay_ms.max(delay);
                } else if *target_throughput >= 1.0 {
                    // No throttle needed at full throughput
                    // Only reset if no other command set a throttle
                    if plan.throttle_delay_ms == 0 {
                        plan.throttle_delay_ms = 0;
                    }
                }
                CommandResult::Ok
            }
            EngineCommand::SwitchComputeUnit { device } => {
                // Supersedes any previous switch in this batch
                plan.switch_device = Some(device.clone());
                CommandResult::Ok
            }
            EngineCommand::PrepareComputeUnit { device } => {
                plan.prepare_device = Some(device.clone());
                CommandResult::Ok
            }
            EngineCommand::SetMemoryLevel {
                level,
                target_ratio,
                ..
            } => {
                self.memory_level = *level;
                // Supersedes previous eviction in same batch
                plan.evict = Some(EvictPlan {
                    target_ratio: *target_ratio,
                    level: *level,
                });
                CommandResult::Ok
            }
            EngineCommand::Suspend => {
                plan.suspended = true;
                self.engine_state = EngineState::Suspended;
                CommandResult::Ok
            }
            EngineCommand::Resume => {
                plan.resumed = true;
                self.engine_state = EngineState::Running;
                // Reset levels to normal on resume
                self.compute_level = ResourceLevel::Normal;
                self.memory_level = ResourceLevel::Normal;
                self.throttle_delay_ms = 0;
                plan.throttle_delay_ms = 0;
                CommandResult::Ok
            }
        }
    }

    fn send_heartbeat(&self, kv_snap: &KVSnapshot) {
        let utilization = if kv_snap.capacity > 0 {
            kv_snap.total_tokens as f32 / kv_snap.capacity as f32
        } else {
            0.0
        };

        let memory_lossy_min = if kv_snap.total_tokens > 0 {
            (kv_snap.protected_prefix as f32 / kv_snap.total_tokens as f32).max(0.01)
        } else {
            0.01
        };

        let status = EngineStatus {
            active_device: self.active_device.clone(),
            compute_level: self.compute_level,
            actual_throughput: self.throughput_ema,
            memory_level: self.memory_level,
            kv_cache_bytes: kv_snap.total_bytes,
            kv_cache_tokens: kv_snap.total_tokens,
            kv_cache_utilization: utilization,
            memory_lossless_min: 1.0, // No lossless shrink available currently
            memory_lossy_min,
            state: self.engine_state,
            tokens_generated: self.tokens_generated,
        };

        let _ = self.resp_tx.send(EngineMessage::Heartbeat(status));
    }

    /// Current engine state.
    pub fn state(&self) -> EngineState {
        self.engine_state
    }

    /// Current compute level.
    pub fn compute_level(&self) -> ResourceLevel {
        self.compute_level
    }

    /// Current memory level.
    pub fn memory_level(&self) -> ResourceLevel {
        self.memory_level
    }

    /// Current throttle delay.
    pub fn throttle_delay_ms(&self) -> u64 {
        self.throttle_delay_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_shared::EngineDirective;

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
            Duration::from_secs(10), // Long interval to avoid heartbeat noise in tests
        );
        (executor, cmd_tx, resp_rx)
    }

    fn empty_snap() -> KVSnapshot {
        KVSnapshot::default()
    }

    fn snap_with_tokens(tokens: usize, capacity: usize, prefix: usize) -> KVSnapshot {
        KVSnapshot {
            total_bytes: (tokens * 256) as u64,
            total_tokens: tokens,
            capacity,
            protected_prefix: prefix,
        }
    }

    #[test]
    fn test_executor_poll_empty() {
        let (mut executor, _tx, _rx) = make_executor();
        let plan = executor.poll(&empty_snap());
        assert!(plan.evict.is_none());
        assert!(plan.switch_device.is_none());
        assert!(!plan.suspended);
        assert!(!plan.resumed);
        assert_eq!(plan.throttle_delay_ms, 0);
    }

    #[test]
    fn test_executor_set_memory_critical() {
        let (mut executor, tx, rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::SetMemoryLevel {
                level: ResourceLevel::Critical,
                target_ratio: 0.5,
                deadline_ms: Some(1000),
            }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert!(plan.evict.is_some());
        let evict = plan.evict.unwrap();
        assert!((evict.target_ratio - 0.5).abs() < f32::EPSILON);
        assert_eq!(evict.level, ResourceLevel::Critical);
        assert_eq!(executor.memory_level(), ResourceLevel::Critical);

        // Check response
        let resp = rx.recv().unwrap();
        match resp {
            EngineMessage::Response(r) => {
                assert_eq!(r.seq_id, 1);
                assert_eq!(r.results.len(), 1);
                assert!(matches!(r.results[0], CommandResult::Ok));
            }
            _ => panic!("Expected Response"),
        }
    }

    #[test]
    fn test_executor_set_compute_warning() {
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::SetComputeLevel {
                level: ResourceLevel::Warning,
                target_throughput: 0.7,
                deadline_ms: None,
            }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert!(plan.throttle_delay_ms > 0);
        assert_eq!(executor.compute_level(), ResourceLevel::Warning);
    }

    #[test]
    fn test_executor_suspend_override() {
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![
                EngineCommand::SetMemoryLevel {
                    level: ResourceLevel::Critical,
                    target_ratio: 0.5,
                    deadline_ms: None,
                },
                EngineCommand::SwitchComputeUnit {
                    device: "cpu".to_string(),
                },
                EngineCommand::Suspend,
            ],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert!(plan.suspended);
        // Suspend should clear other plan fields
        assert!(plan.evict.is_none());
        assert!(plan.switch_device.is_none());
        assert_eq!(executor.state(), EngineState::Suspended);
    }

    #[test]
    fn test_executor_superseding() {
        let (mut executor, tx, rx) = make_executor();

        // Two directives in queue — second should supersede first for same domain
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::SetMemoryLevel {
                level: ResourceLevel::Warning,
                target_ratio: 0.85,
                deadline_ms: None,
            }],
        }))
        .unwrap();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::SetMemoryLevel {
                level: ResourceLevel::Critical,
                target_ratio: 0.5,
                deadline_ms: None,
            }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        // The second (more severe) directive should win
        let evict = plan.evict.unwrap();
        assert!((evict.target_ratio - 0.5).abs() < f32::EPSILON);
        assert_eq!(evict.level, ResourceLevel::Critical);

        // Both responses should be sent
        let r1 = rx.recv().unwrap();
        let r2 = rx.recv().unwrap();
        match (r1, r2) {
            (EngineMessage::Response(r1), EngineMessage::Response(r2)) => {
                assert_eq!(r1.seq_id, 1);
                assert_eq!(r2.seq_id, 2);
            }
            _ => panic!("Expected two Responses"),
        }
    }

    #[test]
    fn test_executor_heartbeat() {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (resp_tx, resp_rx) = mpsc::channel();
        let mut executor = CommandExecutor::new(
            cmd_rx,
            resp_tx,
            "cpu".to_string(),
            Duration::from_millis(10), // Very short interval
        );
        executor.set_running();

        // Wait for heartbeat interval to pass
        std::thread::sleep(Duration::from_millis(20));

        let snap = snap_with_tokens(100, 2048, 4);
        let _plan = executor.poll(&snap);

        // Should have sent a heartbeat
        let msg = resp_rx.recv().unwrap();
        match msg {
            EngineMessage::Heartbeat(status) => {
                assert_eq!(status.active_device, "cpu");
                assert_eq!(status.kv_cache_tokens, 100);
                assert_eq!(status.state, EngineState::Running);
                assert!((status.kv_cache_utilization - 100.0 / 2048.0).abs() < 0.01);
                assert!((status.memory_lossy_min - 4.0 / 100.0).abs() < 0.01);
                assert_eq!(status.memory_lossless_min, 1.0);
            }
            _ => panic!("Expected Heartbeat"),
        }

        drop(cmd_tx); // suppress warning
    }

    #[test]
    fn test_executor_throughput_ema() {
        let (mut executor, _tx, _rx) = make_executor();

        // Simulate token generation at ~10 tok/s
        executor.on_token_generated();
        std::thread::sleep(Duration::from_millis(100));
        executor.on_token_generated();

        // EMA should be approximately 10 tok/s (within bounds)
        assert!(executor.throughput_ema > 5.0);
        assert!(executor.throughput_ema < 20.0);
    }

    #[test]
    fn test_executor_command_response() {
        let (mut executor, tx, rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 42,
            commands: vec![
                EngineCommand::SetComputeLevel {
                    level: ResourceLevel::Normal,
                    target_throughput: 1.0,
                    deadline_ms: None,
                },
                EngineCommand::PrepareComputeUnit {
                    device: "gpu".to_string(),
                },
            ],
        }))
        .unwrap();

        let _plan = executor.poll(&empty_snap());

        let resp = rx.recv().unwrap();
        match resp {
            EngineMessage::Response(r) => {
                assert_eq!(r.seq_id, 42);
                assert_eq!(r.results.len(), 2);
                assert!(matches!(r.results[0], CommandResult::Ok));
                assert!(matches!(r.results[1], CommandResult::Ok));
            }
            _ => panic!("Expected Response"),
        }
    }

    #[test]
    fn test_executor_resume_preserves_levels() {
        let (mut executor, tx, _rx) = make_executor();

        // First: set to critical
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::SetComputeLevel {
                level: ResourceLevel::Critical,
                target_throughput: 0.3,
                deadline_ms: None,
            }],
        }))
        .unwrap();
        executor.poll(&empty_snap());
        assert_eq!(executor.compute_level(), ResourceLevel::Critical);

        // Then: suspend
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::Suspend],
        }))
        .unwrap();
        executor.poll(&empty_snap());
        assert_eq!(executor.state(), EngineState::Suspended);

        // Resume should reset levels to Normal
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 3,
            commands: vec![EngineCommand::Resume],
        }))
        .unwrap();
        let plan = executor.poll(&empty_snap());
        assert!(plan.resumed);
        assert_eq!(executor.state(), EngineState::Running);
        assert_eq!(executor.compute_level(), ResourceLevel::Normal);
        assert_eq!(executor.memory_level(), ResourceLevel::Normal);
        assert_eq!(executor.throttle_delay_ms(), 0);
    }

    #[test]
    fn test_executor_switch_and_prepare() {
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![
                EngineCommand::PrepareComputeUnit {
                    device: "gpu".to_string(),
                },
                EngineCommand::SwitchComputeUnit {
                    device: "gpu".to_string(),
                },
            ],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert_eq!(plan.prepare_device.as_deref(), Some("gpu"));
        assert_eq!(plan.switch_device.as_deref(), Some("gpu"));
    }

    #[test]
    fn test_executor_multiple_compute_levels_max_throttle() {
        let (mut executor, tx, _rx) = make_executor();

        // Two compute commands in one directive — second supersedes
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![
                EngineCommand::SetComputeLevel {
                    level: ResourceLevel::Warning,
                    target_throughput: 0.7,
                    deadline_ms: None,
                },
                EngineCommand::SetComputeLevel {
                    level: ResourceLevel::Critical,
                    target_throughput: 0.3,
                    deadline_ms: None,
                },
            ],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        // The max throttle delay should reflect the most severe command
        assert!(plan.throttle_delay_ms > 0);
        assert_eq!(executor.compute_level(), ResourceLevel::Critical);
    }

    #[test]
    fn test_executor_send_capability() {
        let (_executor, _tx, rx) = make_executor();

        _executor.send_capability(EngineCapability {
            available_devices: vec!["cpu".into(), "opencl".into()],
            active_device: "cpu".into(),
            max_kv_tokens: 2048,
            bytes_per_kv_token: 256,
            num_layers: 16,
        });

        let msg = rx.recv().unwrap();
        match msg {
            EngineMessage::Capability(cap) => {
                assert_eq!(cap.available_devices.len(), 2);
                assert_eq!(cap.num_layers, 16);
            }
            _ => panic!("Expected Capability"),
        }
    }
}
