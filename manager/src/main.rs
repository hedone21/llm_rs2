use clap::Parser;
use llm_manager::config::Config;
use llm_manager::emitter::Emitter;
use llm_manager::monitor::Monitor;
use llm_manager::monitor::compute::ComputeMonitor;
use llm_manager::monitor::energy::EnergyMonitor;
use llm_manager::monitor::external::ExternalMonitor;
use llm_manager::monitor::memory::MemoryMonitor;
use llm_manager::monitor::thermal::ThermalMonitor;
use llm_manager::policy::{MonitorSnapshot, PolicyEngine};
use llm_shared::{ManagerMessage, SystemSignal};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

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

    /// Transport: "dbus" (Linux System Bus) or "unix:<socket_path>".
    #[arg(short, long, default_value = "dbus")]
    transport: String,

    /// Timeout in seconds to wait for LLM client (unix socket only).
    #[arg(long, default_value_t = 60)]
    client_timeout: u64,

    /// PolicyEngine cooldown between directives (ms).
    #[arg(long, default_value_t = 500)]
    policy_cooldown_ms: u64,

    /// Enable legacy passthrough mode (emit raw SystemSignals instead of directives).
    #[arg(long, default_value_t = false)]
    legacy_passthrough: bool,
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

    // Create emitter
    let mut emitter: Box<dyn Emitter> = create_emitter(&args, &shutdown)?;
    log::info!("Emitter: {}", emitter.name());

    // Build monitors
    let monitors = build_monitors(&config);
    log::info!("Monitors: {}", monitors.len());

    // Emit initial state
    let initial_signals: Vec<SystemSignal> =
        monitors.iter().filter_map(|m| m.initial_signal()).collect();
    emitter.emit_initial(&initial_signals)?;

    // Spawn monitor threads
    let (tx, rx) = mpsc::channel::<SystemSignal>();
    let handles = spawn_monitors(monitors, tx, shutdown.clone());
    log::info!("Started {} monitor threads", handles.len());

    // PolicyEngine for directive generation
    let mut policy = PolicyEngine::new(args.policy_cooldown_ms);
    let mut snapshot = MonitorSnapshot::default();

    // Initialize snapshot from initial signals
    for signal in &initial_signals {
        snapshot.update(signal);
    }

    // Main loop
    log::info!(
        "Entering main loop (legacy_passthrough={})",
        args.legacy_passthrough
    );
    loop {
        if SHUTDOWN.load(Ordering::Relaxed) {
            shutdown.store(true, Ordering::Relaxed);
            break;
        }

        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(signal) => {
                log::info!("Signal: {:?}", signal);

                if args.legacy_passthrough {
                    // Legacy mode: pass raw signals directly
                    if let Err(e) = emitter.emit(&signal) {
                        log::error!("Emit failed: {}", e);
                    }
                } else {
                    // New mode: update snapshot and evaluate policy
                    snapshot.update(&signal);

                    if let Some(directive) = policy.evaluate(&snapshot, None) {
                        log::info!(
                            "Directive seq={}: {} commands",
                            directive.seq_id,
                            directive.commands.len()
                        );
                        let msg = ManagerMessage::Directive(directive);
                        let json = serde_json::to_vec(&msg).unwrap_or_default();
                        // Wrap as a SystemSignal for the emitter (temporary bridge)
                        // TODO: Update Emitter trait to support ManagerMessage directly
                        log::debug!("Directive JSON: {} bytes", json.len());

                        // For now, emit the directive as a memory pressure signal
                        // with the JSON in the available_bytes field (hack for backward compat)
                        // In the real implementation, the emitter should send ManagerMessage
                        // directly over the transport. This is handled by the UnixSocket
                        // emitter's enhanced send_directive method.
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
    for handle in handles {
        let _ = handle.join();
    }
    log::info!("LLM Manager stopped");

    Ok(())
}

fn create_emitter(args: &Args, shutdown: &Arc<AtomicBool>) -> anyhow::Result<Box<dyn Emitter>> {
    if let Some(path) = args.transport.strip_prefix("unix:") {
        let mut emitter =
            llm_manager::emitter::unix_socket::UnixSocketEmitter::new(std::path::Path::new(path))?;
        log::info!(
            "Waiting for client on {} (timeout={}s)...",
            path,
            args.client_timeout
        );
        if !emitter.wait_for_client(Duration::from_secs(args.client_timeout), shutdown) {
            if shutdown.load(Ordering::Relaxed) {
                anyhow::bail!("Shutdown during client wait");
            }
            log::warn!("No client connected within timeout, proceeding anyway");
        }
        Ok(Box::new(emitter))
    } else if args.transport == "dbus" {
        let emitter = llm_manager::emitter::dbus::DbusEmitter::new()?;
        Ok(Box::new(emitter))
    } else {
        anyhow::bail!(
            "Unknown transport: {}. Use 'dbus' or 'unix:<path>'",
            args.transport
        );
    }
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
