use clap::Parser;
use llm_manager::channel::EngineReceiver;
use llm_manager::channel::TcpChannel;
use llm_manager::channel::unix_socket::UnixSocketChannel;
use llm_manager::config::Config;
use llm_manager::emitter::Emitter;
use llm_manager::monitor::Monitor;
use llm_manager::monitor::compute::ComputeMonitor;
use llm_manager::monitor::energy::EnergyMonitor;
use llm_manager::monitor::external::ExternalMonitor;
use llm_manager::monitor::memory::MemoryMonitor;
use llm_manager::monitor::thermal::ThermalMonitor;
use llm_manager::pipeline::{DirectiveDeduplicator, PolicyStrategy};
use llm_shared::{EngineMessage, SystemSignal};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

#[cfg(feature = "hierarchical")]
use llm_manager::config::PolicyConfig;
#[cfg(feature = "hierarchical")]
use llm_manager::pipeline::HierarchicalPolicy;

/// Transport 핸들. unix socket / tcp는 양방향, dbus는 emit-only.
enum TransportHandle {
    /// Unix socket — Emitter + EngineReceiver 겸용.
    Unix(UnixSocketChannel),
    /// TCP socket — Emitter + EngineReceiver 겸용.
    Tcp(TcpChannel),
    /// D-Bus 또는 기타 단방향 emitter.
    EmitterOnly(Box<dyn Emitter>),
}

impl TransportHandle {
    fn emitter(&mut self) -> &mut dyn Emitter {
        match self {
            Self::Unix(ch) => ch,
            Self::Tcp(ch) => ch,
            Self::EmitterOnly(em) => em.as_mut(),
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::Unix(ch) => ch.name(),
            Self::Tcp(ch) => ch.name(),
            Self::EmitterOnly(em) => em.name(),
        }
    }

    /// Engine으로부터 메시지를 non-blocking으로 수신한다.
    /// Unix / TCP transport일 때만 실제 수신 시도. dbus는 항상 None.
    fn try_recv_engine_message(&mut self) -> Option<EngineMessage> {
        match self {
            Self::Unix(ch) => ch.try_recv().ok().flatten(),
            Self::Tcp(ch) => ch.try_recv().ok().flatten(),
            Self::EmitterOnly(_) => None,
        }
    }
}

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_signal(_: libc::c_int) {
    SHUTDOWN.store(true, Ordering::Relaxed);
}

#[derive(Parser)]
#[command(
    about = "LLM Resource Manager — monitors system resources and emits directives to LLM engine"
)]
struct Args {
    /// Path to TOML configuration file.
    #[arg(short, long, default_value = "/etc/llm-manager/config.toml")]
    config: std::path::PathBuf,

    /// Transport: "dbus" (Linux System Bus), "unix:<socket_path>", or "tcp:<host:port>".
    #[arg(short, long, default_value = "dbus")]
    transport: String,

    /// Timeout in seconds to wait for LLM client (unix socket only).
    #[arg(long, default_value_t = 60)]
    client_timeout: u64,

    /// Path to policy configuration TOML.
    /// When omitted, built-in defaults are used.
    #[arg(long)]
    policy_config: Option<std::path::PathBuf>,

    /// Path to a Lua policy script.
    /// When specified, the Lua script replaces the built-in HierarchicalPolicy.
    /// Requires the `lua` feature to be enabled at compile time.
    #[arg(long)]
    policy_script: Option<std::path::PathBuf>,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    unsafe {
        libc::signal(
            libc::SIGINT,
            handle_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            handle_signal as *const () as libc::sighandler_t,
        );
    }

    let config = if args.config.exists() {
        log::info!("Loading config from {}", args.config.display());
        Config::from_file(&args.config)?
    } else {
        log::info!("Config not found, using defaults");
        Config::default()
    };

    let default_poll = config.manager.poll_interval_ms;
    log::info!("LLM Manager starting (poll_interval={}ms)", default_poll);

    let shutdown = Arc::new(AtomicBool::new(false));

    // Create transport (unix: 양방향, dbus: 단방향)
    let mut transport = create_transport(&args, &shutdown)?;
    log::info!("Transport: {}", transport.name());

    // Build monitors
    let monitors = build_monitors(&config);
    log::info!("Monitors: {}", monitors.len());

    // Collect initial state from monitors
    let initial_signals: Vec<SystemSignal> =
        monitors.iter().filter_map(|m| m.initial_signal()).collect();

    // Spawn monitor threads
    let (tx, rx) = mpsc::channel::<SystemSignal>();
    let handles = spawn_monitors(monitors, tx, shutdown.clone());
    log::info!("Started {} monitor threads", handles.len());

    // ── Policy 초기화 ─────────────────────────────────────────────────────────

    let mut policy: Box<dyn PolicyStrategy> = create_policy(&args, &config)?;

    // Emit initial state
    for signal in &initial_signals {
        if let Some(directive) = policy.process_signal(signal) {
            log::info!(
                "Initial directive seq={}: {} commands",
                directive.seq_id,
                directive.commands.len()
            );
            transport.emitter().emit_directive(&directive)?;
        }
    }

    // ── Main loop ─────────────────────────────────────────────────────────────
    log::info!("Entering main loop");
    let mut dedup = DirectiveDeduplicator::new();
    loop {
        if SHUTDOWN.load(Ordering::Relaxed) {
            shutdown.store(true, Ordering::Relaxed);
            break;
        }

        // ── Engine message 수신 (unix transport일 때만 유효) ──────────────
        while let Some(msg) = transport.try_recv_engine_message() {
            match &msg {
                EngineMessage::Heartbeat(status) => {
                    policy.update_engine_state(&msg);
                    log::debug!(
                        "Engine heartbeat: kv={:.2} device={}",
                        status.kv_cache_utilization,
                        status.active_device
                    );
                }
                EngineMessage::Response(resp) => {
                    log::info!(
                        "Engine response seq={}: {} results",
                        resp.seq_id,
                        resp.results.len()
                    );
                }
                EngineMessage::Capability(cap) => {
                    log::info!("Engine capability: devices={:?}", cap.available_devices);
                }
                EngineMessage::QcfEstimate(qcf) => {
                    log::info!("Engine QCF estimate: {} actions", qcf.estimates.len());
                    if let Some(directive) = policy.complete_qcf_selection(qcf) {
                        log::info!(
                            "QCF-based directive seq={}: {} commands",
                            directive.seq_id,
                            directive.commands.len()
                        );
                        if let Err(e) = transport.emitter().emit_directive(&directive) {
                            log::error!("Emit QCF directive failed: {}", e);
                        }
                    }
                }
            }
        }

        // QCF timeout check (SEQ-098) — 매 tick(50ms)마다 체크
        if let Some(directive) = policy.check_qcf_timeout() {
            log::info!(
                "QCF timeout fallback directive seq={}: {} commands",
                directive.seq_id,
                directive.commands.len()
            );
            if let Err(e) = transport.emitter().emit_directive(&directive) {
                log::error!("Emit QCF timeout directive failed: {}", e);
            }
        }

        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(signal) => {
                log::info!("Signal: {:?}", signal);

                if let Some(directive) = policy.process_signal(&signal) {
                    if let Some(directive) = dedup.process(directive) {
                        log::info!(
                            "Directive seq={}: {} commands [mode={:?}]",
                            directive.seq_id,
                            directive.commands.len(),
                            policy.mode()
                        );
                        if let Err(e) = transport.emitter().emit_directive(&directive) {
                            log::error!("Emit directive failed: {}", e);
                        }
                    } else {
                        policy.cancel_last_observation();
                        log::debug!("Directive suppressed (duplicate), observation cancelled");
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                log::warn!("All monitors disconnected");
                break;
            }
        }
    }

    log::info!("Shutting down...");
    shutdown.store(true, Ordering::Relaxed);

    // Relief model 저장
    policy.save_model();

    for handle in handles {
        let _ = handle.join();
    }
    log::info!("LLM Manager stopped");

    Ok(())
}

/// `--policy-script`가 지정되면 LuaPolicy를, 아니면 HierarchicalPolicy를 생성한다.
fn create_policy(args: &Args, config: &Config) -> anyhow::Result<Box<dyn PolicyStrategy>> {
    // Lua policy script 지정 시
    if let Some(ref script_path) = args.policy_script {
        return create_lua_policy(script_path, config);
    }

    // hierarchical feature 없이 script도 없으면 에러
    #[cfg(not(feature = "hierarchical"))]
    {
        let _ = config;
        anyhow::bail!(
            "--policy-script is required (built without 'hierarchical' feature; \
             compile with: cargo build --features hierarchical)"
        );
    }

    // 기본: HierarchicalPolicy (hierarchical feature 활성 시)
    #[cfg(feature = "hierarchical")]
    {
        let policy_cfg = load_policy_config(args, config);
        let mut p = HierarchicalPolicy::new(&policy_cfg);
        let storage_dir = if policy_cfg.relief_model.storage_dir.starts_with('~') {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            policy_cfg.relief_model.storage_dir.replacen('~', &home, 1)
        } else {
            policy_cfg.relief_model.storage_dir.clone()
        };
        let model_path = format!("{}/default_relief.json", storage_dir);
        p.set_relief_model_path(model_path);
        log::info!("HierarchicalPolicy initialized");
        Ok(Box::new(p))
    }
}

#[cfg(feature = "lua")]
fn create_lua_policy(
    script_path: &std::path::Path,
    config: &Config,
) -> anyhow::Result<Box<dyn PolicyStrategy>> {
    let path_str = script_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid UTF-8 in policy script path"))?;
    let policy =
        llm_manager::lua_policy::LuaPolicy::with_system_clock(path_str, config.adaptation.clone())?;
    log::info!("LuaPolicy initialized from {}", path_str);
    Ok(Box::new(policy))
}

#[cfg(not(feature = "lua"))]
fn create_lua_policy(
    _script_path: &std::path::Path,
    _config: &Config,
) -> anyhow::Result<Box<dyn PolicyStrategy>> {
    anyhow::bail!(
        "--policy-script requires the 'lua' feature (compile with: cargo build --features lua)"
    )
}

/// `--policy-config` 인자 또는 메인 config의 `[policy]` 섹션에서 PolicyConfig를 로드한다.
/// 둘 다 없으면 기본값을 사용한다.
///
/// 우선순위:
/// 1. `--policy-config` CLI 플래그
/// 2. 메인 config의 `[policy]` 섹션
/// 3. 기본값 (`PolicyConfig::default()`)
#[cfg(feature = "hierarchical")]
fn load_policy_config(args: &Args, config: &Config) -> PolicyConfig {
    if let Some(path) = &args.policy_config {
        match std::fs::read_to_string(path) {
            Ok(content) => match toml::from_str::<PolicyConfig>(&content) {
                Ok(cfg) => {
                    log::info!("Loaded policy config from {}", path.display());
                    return cfg;
                }
                Err(e) => {
                    log::error!(
                        "Failed to parse policy config {}: {} — using defaults",
                        path.display(),
                        e
                    );
                }
            },
            Err(e) => {
                log::error!(
                    "Failed to read policy config {}: {} — using defaults",
                    path.display(),
                    e
                );
            }
        }
    }
    if let Some(ref policy) = config.policy {
        log::info!("Using policy config from main config file");
        return policy.clone();
    }
    PolicyConfig::default()
}

fn create_transport(args: &Args, shutdown: &Arc<AtomicBool>) -> anyhow::Result<TransportHandle> {
    if let Some(path) = args.transport.strip_prefix("unix:") {
        let mut channel = UnixSocketChannel::new(std::path::Path::new(path))?;
        log::info!(
            "Waiting for client on {} (timeout={}s)...",
            path,
            args.client_timeout
        );
        if !channel.wait_for_client(Duration::from_secs(args.client_timeout), shutdown) {
            if shutdown.load(Ordering::Relaxed) {
                anyhow::bail!("Shutdown during client wait");
            }
            log::warn!("No client connected within timeout, proceeding anyway");
        }
        Ok(TransportHandle::Unix(channel))
    } else if let Some(addr) = args.transport.strip_prefix("tcp:") {
        let mut channel = TcpChannel::new(addr)?;
        log::info!(
            "TCP transport: waiting for client on {} (timeout={}s)...",
            addr,
            args.client_timeout
        );
        if !channel.wait_for_client(Duration::from_secs(args.client_timeout), shutdown) {
            if shutdown.load(Ordering::Relaxed) {
                anyhow::bail!("Shutdown during client wait");
            }
            anyhow::bail!("TCP: no client connected within {}s", args.client_timeout);
        }
        Ok(TransportHandle::Tcp(channel))
    } else if args.transport == "dbus" {
        Ok(TransportHandle::EmitterOnly(create_dbus_emitter()?))
    } else {
        anyhow::bail!(
            "Unknown transport: {}. Use 'dbus', 'unix:<path>', or 'tcp:<host:port>'",
            args.transport
        );
    }
}

#[cfg(feature = "dbus")]
fn create_dbus_emitter() -> anyhow::Result<Box<dyn Emitter>> {
    let emitter = llm_manager::emitter::dbus::DbusEmitter::new()?;
    Ok(Box::new(emitter))
}

#[cfg(not(feature = "dbus"))]
fn create_dbus_emitter() -> anyhow::Result<Box<dyn Emitter>> {
    anyhow::bail!("Transport 'dbus' requires the 'dbus' feature (compiled without it)")
}

fn build_monitors(config: &Config) -> Vec<Box<dyn Monitor>> {
    let default_poll = config.manager.poll_interval_ms;
    let mut monitors: Vec<Box<dyn Monitor>> = Vec::new();

    if config.memory.as_ref().is_none_or(|c| c.enabled) {
        let c = config.memory.clone().unwrap_or_default();
        monitors.push(Box::new(MemoryMonitor::new(&c, default_poll)));
    }

    if config.thermal.as_ref().is_none_or(|c| c.enabled) {
        let c = config.thermal.clone().unwrap_or_default();
        monitors.push(Box::new(ThermalMonitor::new(&c, default_poll)));
    }

    if config.compute.as_ref().is_none_or(|c| c.enabled) {
        let c = config.compute.clone().unwrap_or_default();
        monitors.push(Box::new(ComputeMonitor::new(&c, default_poll)));
    }

    if config.energy.as_ref().is_none_or(|c| c.enabled) {
        let c = config.energy.clone().unwrap_or_default();
        monitors.push(Box::new(EnergyMonitor::new(&c, default_poll)));
    }

    if config.external.as_ref().is_some_and(|c| c.enabled) {
        let c = config.external.as_ref().unwrap();
        monitors.push(Box::new(ExternalMonitor::new(c)));
    }

    monitors
}

fn spawn_monitors(
    monitors: Vec<Box<dyn Monitor>>,
    tx: mpsc::Sender<SystemSignal>,
    shutdown: Arc<AtomicBool>,
) -> Vec<std::thread::JoinHandle<()>> {
    let mut handles = Vec::new();

    for mut monitor in monitors {
        let tx = tx.clone();
        let shutdown = shutdown.clone();
        let name = monitor.name().to_string();

        let handle = std::thread::Builder::new()
            .name(name.clone())
            .spawn(move || {
                if let Err(e) = monitor.run(tx, shutdown) {
                    log::error!("[{}] Monitor error: {}", name, e);
                }
            })
            .expect("Failed to spawn monitor thread");

        handles.push(handle);
    }

    drop(tx);
    handles
}
