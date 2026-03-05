//! Signal injector for resilience stress testing.
//!
//! Listens on a Unix socket, waits for the generate binary to connect,
//! then sends SystemSignal messages according to a JSON schedule file.
//!
//! Schedule format:
//! ```json
//! [
//!   {"delay_sec": 10, "signal": {"memory_pressure": {"level": "critical", ...}}},
//!   {"delay_sec": 20, "signal": {"thermal_alert": {"level": "warning", ...}}}
//! ]
//! ```

use clap::Parser;
use llm_shared::SystemSignal;
use serde::Deserialize;
use std::io::Write;
use std::os::unix::net::UnixListener;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Inject resilience signals via Unix socket for stress testing")]
struct Args {
    /// Path to the Unix socket to listen on
    #[arg(short, long)]
    socket: PathBuf,

    /// Path to the JSON schedule file
    #[arg(short = 'f', long)]
    schedule_file: PathBuf,

    /// Timeout in seconds to wait for a client connection
    #[arg(long, default_value_t = 30)]
    connect_timeout: u64,
}

#[derive(Debug, Deserialize)]
struct ScheduleEntry {
    delay_sec: f64,
    signal: SystemSignal,
}

fn write_signal(
    stream: &mut std::os::unix::net::UnixStream,
    signal: &SystemSignal,
) -> std::io::Result<()> {
    let json = serde_json::to_vec(signal).expect("SystemSignal serialization should not fail");
    let len = (json.len() as u32).to_be_bytes();
    stream.write_all(&len)?;
    stream.write_all(&json)?;
    stream.flush()
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Read schedule
    let schedule_json = std::fs::read_to_string(&args.schedule_file)?;
    let schedule: Vec<ScheduleEntry> = serde_json::from_str(&schedule_json)?;
    eprintln!(
        "[Injector] Loaded {} signals from {}",
        schedule.len(),
        args.schedule_file.display()
    );

    // Remove stale socket file
    let _ = std::fs::remove_file(&args.socket);

    // Listen
    let listener = UnixListener::bind(&args.socket)?;
    listener.set_nonblocking(false)?;
    eprintln!(
        "[Injector] Listening on {} (timeout {}s)",
        args.socket.display(),
        args.connect_timeout
    );

    // Wait for connection with timeout
    listener.set_nonblocking(true).ok();
    let start = std::time::Instant::now();
    let mut stream = loop {
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if start.elapsed().as_secs() >= args.connect_timeout {
                    anyhow::bail!("Timed out waiting for client connection");
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(e) => return Err(e.into()),
        }
    };
    stream.set_nonblocking(false)?;
    eprintln!("[Injector] Client connected");

    // Send signals according to schedule
    for (i, entry) in schedule.iter().enumerate() {
        if entry.delay_sec > 0.0 {
            std::thread::sleep(std::time::Duration::from_secs_f64(entry.delay_sec));
        }

        match write_signal(&mut stream, &entry.signal) {
            Ok(()) => {
                eprintln!(
                    "[Injector] [{}/{}] Sent: {:?}",
                    i + 1,
                    schedule.len(),
                    entry.signal
                );
            }
            Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => {
                eprintln!("[Injector] Client disconnected (broken pipe)");
                break;
            }
            Err(e) => {
                eprintln!("[Injector] Send error: {}", e);
                break;
            }
        }
    }

    eprintln!("[Injector] Schedule complete, exiting");
    let _ = std::fs::remove_file(&args.socket);
    Ok(())
}
