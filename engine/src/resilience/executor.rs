use std::sync::mpsc;
use std::time::{Duration, Instant};

use llm_shared::{
    CommandResponse, CommandResult, EngineCapability, EngineCommand, EngineMessage, EngineState,
    EngineStatus, ManagerMessage, QcfEstimate, ResourceLevel,
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
    /// Layer skip ratio: 0.0 = no skip, from LayerSkip command.
    pub layer_skip: Option<f32>,
    /// KV quantization bits, from KvQuantDynamic command.
    pub kv_quant_bits: Option<u8>,
    /// Whether to restore all action-induced state to defaults.
    pub restore_defaults: bool,
    /// Whether Engine should compute and send QCF estimates.
    pub request_qcf: bool,
}

/// Eviction method identifier (engine-internal, not in shared protocol).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvictMethod {
    H2o,
    Sliding,
    Streaming,
}

/// StreamingLLM eviction parameters (sink + window).
#[derive(Debug, Clone)]
pub struct StreamingParams {
    pub sink_size: usize,
    pub window_size: usize,
}

/// Eviction plan parameters.
#[derive(Debug, Clone)]
pub struct EvictPlan {
    /// Target ratio of cache to keep (0.0–1.0).
    pub target_ratio: f32,
    /// Resource level: Warning = lossless only, Critical = lossy OK.
    pub level: ResourceLevel,
    /// Which eviction algorithm to use.
    pub method: EvictMethod,
    /// StreamingLLM-specific parameters (only set when method == Streaming).
    pub streaming_params: Option<StreamingParams>,
}

/// Snapshot of KV cache state for status reporting.
#[derive(Debug, Clone, Default)]
pub struct KVSnapshot {
    pub total_bytes: u64,
    pub total_tokens: usize,
    pub capacity: usize,
    pub protected_prefix: usize,
    /// Current KV dtype name for heartbeat reporting ("f16", "q8", "q4").
    pub kv_dtype: String,
    /// Current eviction policy name for heartbeat reporting.
    pub eviction_policy: String,
    /// Current layer skip ratio for heartbeat reporting.
    pub skip_ratio: f32,
}

// ── CommandExecutor ─────────────────────────────────────────

/// Receives ManagerMessages, builds ExecutionPlans for the inference loop.
/// No strategy logic — just translates commands to plans.
pub struct CommandExecutor {
    cmd_rx: mpsc::Receiver<ManagerMessage>,
    resp_tx: mpsc::Sender<EngineMessage>,

    // Current state (deprecated fields kept for EngineStatus backward compat)
    compute_level: ResourceLevel,
    memory_level: ResourceLevel,
    engine_state: EngineState,
    active_device: String,
    throttle_delay_ms: u64,

    // Currently active action names (e.g. "kv_evict_h2o", "throttle")
    active_actions: Vec<String>,

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
            active_actions: Vec::new(),
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

    /// Send QCF estimate to Manager (SEQ-096).
    pub fn send_qcf_estimate(&self, qcf: QcfEstimate) {
        let _ = self.resp_tx.send(EngineMessage::QcfEstimate(qcf));
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
            EngineCommand::Throttle { delay_ms } => {
                // 직접 지정된 딜레이를 적용한다 (처리량 비율 변환 없이)
                plan.throttle_delay_ms = *delay_ms;
                if *delay_ms > 0 {
                    if !self.active_actions.contains(&"throttle".to_string()) {
                        self.active_actions.push("throttle".to_string());
                    }
                } else {
                    self.active_actions.retain(|a| a != "throttle");
                }
                CommandResult::Ok
            }
            EngineCommand::LayerSkip { skip_ratio } => {
                plan.layer_skip = Some(*skip_ratio);
                if !self.active_actions.contains(&"layer_skip".to_string()) {
                    self.active_actions.push("layer_skip".to_string());
                }
                CommandResult::Ok
            }
            EngineCommand::KvEvictH2o { keep_ratio } => {
                plan.evict = Some(EvictPlan {
                    target_ratio: *keep_ratio,
                    level: ResourceLevel::Critical,
                    method: EvictMethod::H2o,
                    streaming_params: None,
                });
                if !self.active_actions.contains(&"kv_evict_h2o".to_string()) {
                    self.active_actions.push("kv_evict_h2o".to_string());
                }
                CommandResult::Ok
            }
            EngineCommand::KvEvictSliding { keep_ratio } => {
                plan.evict = Some(EvictPlan {
                    target_ratio: *keep_ratio,
                    level: ResourceLevel::Critical,
                    method: EvictMethod::Sliding,
                    streaming_params: None,
                });
                if !self
                    .active_actions
                    .contains(&"kv_evict_sliding".to_string())
                {
                    self.active_actions.push("kv_evict_sliding".to_string());
                }
                CommandResult::Ok
            }
            EngineCommand::KvStreaming {
                sink_size,
                window_size,
            } => {
                plan.evict = Some(EvictPlan {
                    target_ratio: 0.0, // StreamingLLMPolicy는 target_len 무시
                    level: ResourceLevel::Critical,
                    method: EvictMethod::Streaming,
                    streaming_params: Some(StreamingParams {
                        sink_size: *sink_size,
                        window_size: *window_size,
                    }),
                });
                if !self
                    .active_actions
                    .contains(&"kv_evict_streaming".to_string())
                {
                    self.active_actions.push("kv_evict_streaming".to_string());
                }
                CommandResult::Ok
            }
            EngineCommand::KvQuantDynamic { target_bits } => {
                plan.kv_quant_bits = Some(*target_bits);
                if !self
                    .active_actions
                    .contains(&"kv_quant_dynamic".to_string())
                {
                    self.active_actions.push("kv_quant_dynamic".to_string());
                }
                CommandResult::Ok
            }
            EngineCommand::RestoreDefaults => {
                plan.restore_defaults = true;
                plan.throttle_delay_ms = 0;
                self.throttle_delay_ms = 0;
                self.compute_level = ResourceLevel::Normal;
                self.memory_level = ResourceLevel::Normal;
                self.active_actions.clear();
                CommandResult::Ok
            }
            EngineCommand::SwitchHw { device } => {
                // 배치 내 이전 스위치를 덮어쓴다
                plan.switch_device = Some(device.clone());
                CommandResult::Ok
            }
            EngineCommand::PrepareComputeUnit { device } => {
                plan.prepare_device = Some(device.clone());
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
                // Resume 시 레벨과 스로틀을 Normal로 초기화
                self.compute_level = ResourceLevel::Normal;
                self.memory_level = ResourceLevel::Normal;
                self.throttle_delay_ms = 0;
                plan.throttle_delay_ms = 0;
                CommandResult::Ok
            }
            EngineCommand::RequestQcf => {
                plan.request_qcf = true;
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

        let eviction_policy = if kv_snap.eviction_policy.is_empty() {
            "none".to_string()
        } else {
            kv_snap.eviction_policy.clone()
        };

        let kv_dtype = if kv_snap.kv_dtype.is_empty() {
            "f16".to_string()
        } else {
            kv_snap.kv_dtype.clone()
        };

        let status = EngineStatus {
            active_device: self.active_device.clone(),
            compute_level: self.compute_level,
            actual_throughput: self.throughput_ema,
            memory_level: self.memory_level,
            kv_cache_bytes: kv_snap.total_bytes,
            kv_cache_tokens: kv_snap.total_tokens,
            kv_cache_utilization: utilization,
            memory_lossless_min: 1.0, // 현재 무손실 축소 불가
            memory_lossy_min,
            state: self.engine_state,
            tokens_generated: self.tokens_generated,
            available_actions: Self::compute_available_actions(&eviction_policy, &kv_dtype),
            active_actions: self.active_actions.clone(),
            eviction_policy,
            kv_dtype,
            skip_ratio: kv_snap.skip_ratio,
        };

        let _ = self.resp_tx.send(EngineMessage::Heartbeat(status));
    }

    /// Compute available actions based on engine capabilities.
    fn compute_available_actions(eviction_policy: &str, kv_dtype: &str) -> Vec<String> {
        let mut actions = vec![
            "throttle".to_string(),
            "switch_hw".to_string(),
            "layer_skip".to_string(),
        ];
        // Eviction actions: only if an eviction policy is configured
        if eviction_policy != "none" {
            actions.push("kv_evict_h2o".to_string());
            actions.push("kv_evict_sliding".to_string());
            actions.push("kv_evict_streaming".to_string());
        }
        // KV quantization: only available with KIVI cache (q2/q4/q8)
        if kv_dtype.starts_with('q') {
            actions.push("kv_quant_dynamic".to_string());
        }
        actions
    }

    /// Current engine state.
    pub fn state(&self) -> EngineState {
        self.engine_state
    }

    /// Current compute level (deprecated, always Normal unless set by legacy path).
    pub fn compute_level(&self) -> ResourceLevel {
        self.compute_level
    }

    /// Current memory level (deprecated, always Normal unless set by legacy path).
    pub fn memory_level(&self) -> ResourceLevel {
        self.memory_level
    }

    /// Current throttle delay.
    pub fn throttle_delay_ms(&self) -> u64 {
        self.throttle_delay_ms
    }

    /// Currently active action names.
    pub fn active_actions(&self) -> &[String] {
        &self.active_actions
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
            Duration::from_secs(10), // 테스트에서 하트비트 노이즈 방지용 긴 간격
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
            kv_dtype: "f16".to_string(),
            eviction_policy: "none".to_string(),
            skip_ratio: 0.0,
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
    fn test_executor_throttle_direct() {
        let (mut executor, tx, rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::Throttle { delay_ms: 50 }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert_eq!(plan.throttle_delay_ms, 50);
        assert!(executor.active_actions().contains(&"throttle".to_string()));

        // 응답 확인
        let resp = rx.recv().unwrap();
        match resp {
            EngineMessage::Response(r) => {
                assert_eq!(r.seq_id, 1);
                assert!(matches!(r.results[0], CommandResult::Ok));
            }
            _ => panic!("Expected Response"),
        }
    }

    #[test]
    fn test_executor_kv_evict_h2o() {
        let (mut executor, tx, rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::KvEvictH2o { keep_ratio: 0.5 }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert!(plan.evict.is_some());
        let evict = plan.evict.unwrap();
        assert!((evict.target_ratio - 0.5).abs() < f32::EPSILON);
        assert_eq!(evict.level, ResourceLevel::Critical);
        assert_eq!(evict.method, EvictMethod::H2o);
        assert!(
            executor
                .active_actions()
                .contains(&"kv_evict_h2o".to_string())
        );

        let resp = rx.recv().unwrap();
        match resp {
            EngineMessage::Response(r) => {
                assert_eq!(r.seq_id, 1);
                assert!(matches!(r.results[0], CommandResult::Ok));
            }
            _ => panic!("Expected Response"),
        }
    }

    #[test]
    fn test_executor_kv_evict_sliding() {
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::KvEvictSliding { keep_ratio: 0.6 }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert!(plan.evict.is_some());
        let evict = plan.evict.unwrap();
        assert!((evict.target_ratio - 0.6).abs() < f32::EPSILON);
        assert_eq!(evict.method, EvictMethod::Sliding);
        assert!(
            executor
                .active_actions()
                .contains(&"kv_evict_sliding".to_string())
        );
    }

    #[test]
    fn test_executor_kv_streaming_ok() {
        let (mut executor, tx, rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::KvStreaming {
                sink_size: 4,
                window_size: 256,
            }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert!(
            plan.evict.is_some(),
            "KvStreaming은 evict plan을 생성해야 함"
        );
        let evict = plan.evict.unwrap();
        assert_eq!(evict.method, EvictMethod::Streaming);
        assert!((evict.target_ratio - 0.0).abs() < f32::EPSILON);
        assert_eq!(evict.level, ResourceLevel::Critical);
        let params = evict
            .streaming_params
            .expect("streaming_params가 있어야 함");
        assert_eq!(params.sink_size, 4);
        assert_eq!(params.window_size, 256);
        assert!(
            executor
                .active_actions()
                .contains(&"kv_evict_streaming".to_string())
        );

        let resp = rx.recv().unwrap();
        match resp {
            EngineMessage::Response(r) => {
                assert_eq!(r.seq_id, 1);
                assert!(matches!(r.results[0], CommandResult::Ok));
            }
            _ => panic!("Expected Response"),
        }
    }

    #[test]
    fn test_executor_kv_quant_dynamic() {
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::KvQuantDynamic { target_bits: 4 }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert_eq!(plan.kv_quant_bits, Some(4));
        assert!(
            executor
                .active_actions()
                .contains(&"kv_quant_dynamic".to_string())
        );
    }

    #[test]
    fn test_executor_layer_skip() {
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::LayerSkip { skip_ratio: 0.25 }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert_eq!(plan.layer_skip, Some(0.25));
        assert!(
            executor
                .active_actions()
                .contains(&"layer_skip".to_string())
        );
    }

    #[test]
    fn test_executor_restore_defaults() {
        let (mut executor, tx, _rx) = make_executor();

        // 먼저 스로틀과 액션 설정
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![
                EngineCommand::Throttle { delay_ms: 50 },
                EngineCommand::LayerSkip { skip_ratio: 0.3 },
            ],
        }))
        .unwrap();
        executor.poll(&empty_snap());
        assert!(!executor.active_actions().is_empty());

        // RestoreDefaults로 초기화
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::RestoreDefaults],
        }))
        .unwrap();
        let plan = executor.poll(&empty_snap());
        assert!(plan.restore_defaults);
        assert_eq!(plan.throttle_delay_ms, 0);
        assert_eq!(executor.throttle_delay_ms(), 0);
        assert!(executor.active_actions().is_empty());
    }

    #[test]
    fn test_executor_switch_hw() {
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::SwitchHw {
                device: "opencl".to_string(),
            }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert_eq!(plan.switch_device.as_deref(), Some("opencl"));
    }

    #[test]
    fn test_executor_suspend_override() {
        let (mut executor, tx, _rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![
                EngineCommand::KvEvictH2o { keep_ratio: 0.5 },
                EngineCommand::SwitchHw {
                    device: "cpu".to_string(),
                },
                EngineCommand::Suspend,
            ],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        assert!(plan.suspended);
        // Suspend는 다른 plan 필드를 초기화해야 함
        assert!(plan.evict.is_none());
        assert!(plan.switch_device.is_none());
        assert_eq!(executor.state(), EngineState::Suspended);
    }

    #[test]
    fn test_executor_superseding_evict() {
        let (mut executor, tx, rx) = make_executor();

        // 두 개의 evict 명령 — 두 번째가 첫 번째를 덮어씀
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::KvEvictH2o { keep_ratio: 0.8 }],
        }))
        .unwrap();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::KvEvictSliding { keep_ratio: 0.5 }],
        }))
        .unwrap();

        let plan = executor.poll(&empty_snap());
        // 두 번째 명령이 승리
        let evict = plan.evict.unwrap();
        assert!((evict.target_ratio - 0.5).abs() < f32::EPSILON);
        assert_eq!(evict.method, EvictMethod::Sliding);

        // 두 응답 모두 전송돼야 함
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
            Duration::from_millis(10), // 매우 짧은 간격으로 하트비트 유도
        );
        executor.set_running();

        // 하트비트 간격이 지나길 기다림
        std::thread::sleep(Duration::from_millis(20));

        let snap = snap_with_tokens(100, 2048, 4);
        let _plan = executor.poll(&snap);

        // 하트비트가 전송돼야 함
        let msg = resp_rx.recv().unwrap();
        match msg {
            EngineMessage::Heartbeat(status) => {
                assert_eq!(status.active_device, "cpu");
                assert_eq!(status.kv_cache_tokens, 100);
                assert_eq!(status.state, EngineState::Running);
                assert!((status.kv_cache_utilization - 100.0 / 2048.0).abs() < 0.01);
                assert!((status.memory_lossy_min - 4.0 / 100.0).abs() < 0.01);
                assert_eq!(status.memory_lossless_min, 1.0);
                assert_eq!(status.eviction_policy, "none");
                assert_eq!(status.kv_dtype, "f16");
            }
            _ => panic!("Expected Heartbeat"),
        }

        drop(cmd_tx); // 미사용 경고 억제
    }

    #[test]
    fn test_executor_heartbeat_active_actions() {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (resp_tx, resp_rx) = mpsc::channel();
        let mut executor = CommandExecutor::new(
            cmd_rx,
            resp_tx,
            "cpu".to_string(),
            Duration::from_millis(10),
        );
        executor.set_running();

        // 스로틀 액션 활성화
        cmd_tx
            .send(ManagerMessage::Directive(EngineDirective {
                seq_id: 1,
                commands: vec![EngineCommand::Throttle { delay_ms: 30 }],
            }))
            .unwrap();
        executor.poll(&empty_snap());

        std::thread::sleep(Duration::from_millis(20));
        let _plan = executor.poll(&empty_snap());

        // 하트비트에 active_actions가 포함돼야 함
        // Response와 Heartbeat가 순서대로 올 수 있음
        let mut found_heartbeat = false;
        for _ in 0..3 {
            if let Ok(msg) = resp_rx.try_recv() {
                if let EngineMessage::Heartbeat(status) = msg {
                    assert!(status.active_actions.contains(&"throttle".to_string()));
                    found_heartbeat = true;
                    break;
                }
            }
        }
        assert!(
            found_heartbeat,
            "Should have received heartbeat with active_actions"
        );
    }

    #[test]
    fn test_executor_throughput_ema() {
        let (mut executor, _tx, _rx) = make_executor();

        // ~10 tok/s 속도로 토큰 생성 시뮬레이션
        executor.on_token_generated();
        std::thread::sleep(Duration::from_millis(100));
        executor.on_token_generated();

        // EMA가 약 10 tok/s 범위에 있어야 함
        assert!(executor.throughput_ema > 5.0);
        assert!(executor.throughput_ema < 20.0);
    }

    #[test]
    fn test_executor_command_response() {
        let (mut executor, tx, rx) = make_executor();

        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 42,
            commands: vec![
                EngineCommand::Throttle { delay_ms: 0 },
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
    fn test_executor_resume_resets_state() {
        let (mut executor, tx, _rx) = make_executor();

        // 스로틀 설정
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![EngineCommand::Throttle { delay_ms: 100 }],
        }))
        .unwrap();
        executor.poll(&empty_snap());
        assert_eq!(executor.throttle_delay_ms(), 100);

        // 중단
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::Suspend],
        }))
        .unwrap();
        executor.poll(&empty_snap());
        assert_eq!(executor.state(), EngineState::Suspended);

        // Resume은 스로틀을 Normal로 초기화해야 함
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
                EngineCommand::SwitchHw {
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

    #[test]
    fn test_executor_restore_clears_active_actions() {
        let (mut executor, tx, _rx) = make_executor();

        // 여러 액션 활성화
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 1,
            commands: vec![
                EngineCommand::Throttle { delay_ms: 50 },
                EngineCommand::KvEvictH2o { keep_ratio: 0.5 },
                EngineCommand::LayerSkip { skip_ratio: 0.2 },
            ],
        }))
        .unwrap();
        executor.poll(&empty_snap());
        assert_eq!(executor.active_actions().len(), 3);

        // RestoreDefaults로 모두 초기화
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 2,
            commands: vec![EngineCommand::RestoreDefaults],
        }))
        .unwrap();
        executor.poll(&empty_snap());
        assert!(executor.active_actions().is_empty());
        assert_eq!(executor.throttle_delay_ms(), 0);
    }

    #[test]
    fn test_request_qcf_sets_plan_flag() {
        let (mut executor, tx, rx) = make_executor();
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 10,
            commands: vec![EngineCommand::RequestQcf],
        }))
        .unwrap();
        let plan = executor.poll(&empty_snap());
        assert!(plan.request_qcf);

        // Should receive Response with Ok
        let resp = rx.try_recv().unwrap();
        match resp {
            EngineMessage::Response(r) => {
                assert_eq!(r.seq_id, 10);
                assert_eq!(r.results.len(), 1);
                assert!(matches!(r.results[0], CommandResult::Ok));
            }
            _ => panic!("Expected Response"),
        }
    }

    #[test]
    fn test_request_qcf_with_other_commands() {
        let (mut executor, tx, rx) = make_executor();
        tx.send(ManagerMessage::Directive(EngineDirective {
            seq_id: 11,
            commands: vec![
                EngineCommand::Throttle { delay_ms: 30 },
                EngineCommand::RequestQcf,
            ],
        }))
        .unwrap();
        let plan = executor.poll(&empty_snap());
        assert!(plan.request_qcf);
        assert_eq!(plan.throttle_delay_ms, 30);

        let resp = rx.try_recv().unwrap();
        match resp {
            EngineMessage::Response(r) => {
                assert_eq!(r.results.len(), 2);
                assert!(matches!(r.results[0], CommandResult::Ok));
                assert!(matches!(r.results[1], CommandResult::Ok));
            }
            _ => panic!("Expected Response"),
        }
    }

    #[test]
    fn test_request_qcf_default_false() {
        let (mut executor, _tx, _rx) = make_executor();
        let plan = executor.poll(&empty_snap());
        assert!(!plan.request_qcf);
    }

    #[test]
    fn test_send_qcf_estimate() {
        let (executor, _tx, rx) = make_executor();
        let mut estimates = std::collections::HashMap::new();
        estimates.insert("kv_evict_sliding".to_string(), 0.5);
        estimates.insert("kv_evict_h2o".to_string(), 0.1);
        executor.send_qcf_estimate(QcfEstimate { estimates });

        let msg = rx.try_recv().unwrap();
        match msg {
            EngineMessage::QcfEstimate(qcf) => {
                assert_eq!(qcf.estimates.len(), 2);
                assert!((qcf.estimates["kv_evict_sliding"] - 0.5).abs() < f32::EPSILON);
                assert!((qcf.estimates["kv_evict_h2o"] - 0.1).abs() < f32::EPSILON);
            }
            _ => panic!("Expected QcfEstimate"),
        }
    }
}
