use clap::Parser;
use llm_manager::collector::compute::ComputeCollector;
use llm_manager::collector::energy::EnergyCollector;
use llm_manager::collector::memory::MemoryCollector;
use llm_manager::collector::thermal::ThermalCollector;
use llm_manager::collector::{Collector, Reading};
use llm_manager::config::Config;
use llm_manager::emitter::Emitter;
use llm_manager::policy::PolicyEngine;
use llm_manager::policy::threshold::ThresholdPolicy;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

// Global shutdown flag for signal handler
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_signal(_: libc::c_int) {
    SHUTDOWN.store(true, Ordering::Relaxed);
}

#[derive(Parser)]
#[command(
    about = "LLM Resource Manager — monitors system resources and emits signals to LLM engine"
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
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    // Install signal handlers
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

    let poll_ms = config.monitor.poll_interval_ms;
    log::info!("LLM Manager starting (poll_interval={}ms)", poll_ms);

    // Shutdown flag shared with collector threads
    let shutdown = Arc::new(AtomicBool::new(false));

    // Create emitter
    let mut emitter: Box<dyn Emitter> = create_emitter(&args.transport, &args, &shutdown)?;
    log::info!("Emitter: {}", emitter.name());

    // Create policy engine
    let mut policy = ThresholdPolicy::new(config.clone());

    // Emit initial state
    let initial_signals = policy.current_signals();
    emitter.emit_initial(&initial_signals)?;

    // Create central channel
    let (tx, rx) = mpsc::channel::<Reading>();

    // Spawn collector threads
    let collector_threads = spawn_collectors(poll_ms, tx, shutdown.clone());
    log::info!("Started {} collector threads", collector_threads.len());

    // Main loop
    log::info!("Entering main loop");
    loop {
        // Check global shutdown signal from SIGINT/SIGTERM
        if SHUTDOWN.load(Ordering::Relaxed) {
            shutdown.store(true, Ordering::Relaxed);
            break;
        }

        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(reading) => {
                let signals = policy.process(&reading);
                for signal in &signals {
                    log::info!("Signal: {:?}", signal);
                    if let Err(e) = emitter.emit(signal) {
                        log::error!("Emit failed: {}", e);
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                log::warn!("All collectors disconnected");
                break;
            }
        }
    }

    // Shutdown
    log::info!("Shutting down...");
    shutdown.store(true, Ordering::Relaxed);
    for handle in collector_threads {
        let _ = handle.join();
    }
    log::info!("LLM Manager stopped");

    Ok(())
}

fn create_emitter(
    transport: &str,
    args: &Args,
    shutdown: &Arc<AtomicBool>,
) -> anyhow::Result<Box<dyn Emitter>> {
    if let Some(path) = transport.strip_prefix("unix:") {
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
    } else if transport == "dbus" {
        let emitter = llm_manager::emitter::dbus::DbusEmitter::new()?;
        Ok(Box::new(emitter))
    } else {
        anyhow::bail!(
            "Unknown transport: {}. Use 'dbus' or 'unix:<path>'",
            transport
        );
    }
}

fn spawn_collectors(
    poll_ms: u64,
    tx: mpsc::Sender<Reading>,
    shutdown: Arc<AtomicBool>,
) -> Vec<std::thread::JoinHandle<()>> {
    let mut handles = Vec::new();

    let collectors: Vec<Box<dyn Collector>> = vec![
        Box::new(MemoryCollector::new(poll_ms)),
        Box::new(ThermalCollector::new(poll_ms)),
        Box::new(ComputeCollector::new(poll_ms)),
        Box::new(EnergyCollector::new(poll_ms)),
    ];

    for mut collector in collectors {
        let tx = tx.clone();
        let shutdown = shutdown.clone();
        let name = collector.name().to_string();

        let handle = std::thread::Builder::new()
            .name(name.clone())
            .spawn(move || {
                if let Err(e) = collector.run(tx, shutdown) {
                    log::error!("[{}] Collector error: {}", name, e);
                }
            })
            .expect("Failed to spawn collector thread");

        handles.push(handle);
    }

    // Drop original sender so rx detects disconnection when all collectors stop
    drop(tx);

    handles
}
