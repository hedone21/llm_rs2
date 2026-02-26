//! Mock D-Bus Manager for T3 testing.
//!
//! Emits `org.llm.Manager1` signals on the System Bus for testing
//! the DbusListener. Supports single-signal CLI mode and scenario replay.
//!
//! # Usage
//!
//! ```bash
//! # Single signal
//! mock_manager --signal MemoryPressure --level critical \
//!     --available-bytes 50000000 --reclaim-target 100000000
//!
//! # Scenario replay
//! mock_manager --scenario scenarios/thermal_spike.json
//! ```

use clap::Parser;
use serde::Deserialize;
use std::path::PathBuf;

const MANAGER_NAME: &str = "org.llm.Manager1";
const MANAGER_PATH: &str = "/org/llm/Manager1";
const MANAGER_IFACE: &str = "org.llm.Manager1";

#[derive(Parser, Debug)]
#[command(
    name = "mock_manager",
    about = "Mock D-Bus Manager for resilience testing"
)]
struct Args {
    /// Signal to emit (MemoryPressure, ComputeGuidance, ThermalAlert, EnergyConstraint)
    #[arg(long)]
    signal: Option<String>,

    /// Scenario JSON file to replay
    #[arg(long)]
    scenario: Option<PathBuf>,

    // --- MemoryPressure args ---
    #[arg(long)]
    level: Option<String>,

    #[arg(long)]
    available_bytes: Option<u64>,

    #[arg(long)]
    reclaim_target: Option<u64>,

    // --- ComputeGuidance args ---
    #[arg(long)]
    recommended_backend: Option<String>,

    #[arg(long)]
    reason: Option<String>,

    #[arg(long)]
    cpu_usage: Option<f64>,

    #[arg(long)]
    gpu_usage: Option<f64>,

    // --- ThermalAlert args ---
    #[arg(long)]
    temperature_mc: Option<i32>,

    #[arg(long)]
    throttling_active: Option<bool>,

    #[arg(long)]
    throttle_ratio: Option<f64>,

    // --- EnergyConstraint args ---
    #[arg(long)]
    power_budget_mw: Option<u32>,
}

/// Scenario file format for replaying signal sequences.
#[derive(Debug, Deserialize)]
struct Scenario {
    name: String,
    #[serde(default)]
    description: Option<String>,
    signals: Vec<ScenarioSignal>,
}

/// A single signal entry in a scenario file.
#[derive(Debug, Deserialize)]
struct ScenarioSignal {
    delay_ms: u64,
    signal: String,
    level: String,
    // MemoryPressure
    #[serde(default)]
    available_bytes: Option<u64>,
    #[serde(default)]
    reclaim_target_bytes: Option<u64>,
    // ComputeGuidance
    #[serde(default)]
    recommended_backend: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    cpu_usage_pct: Option<f64>,
    #[serde(default)]
    gpu_usage_pct: Option<f64>,
    // ThermalAlert
    #[serde(default)]
    temperature_mc: Option<i32>,
    #[serde(default)]
    throttling_active: Option<bool>,
    #[serde(default)]
    throttle_ratio: Option<f64>,
    // EnergyConstraint
    #[serde(default)]
    power_budget_mw: Option<u32>,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    // Connect to System Bus and request well-known name
    let conn = zbus::blocking::Connection::system()?;
    conn.request_name(MANAGER_NAME)?;
    println!("Registered as {} on System Bus", MANAGER_NAME);

    if let Some(scenario_path) = &args.scenario {
        run_scenario(&conn, scenario_path)?;
    } else if let Some(signal_name) = &args.signal {
        emit_single(&conn, &args, signal_name)?;
    } else {
        anyhow::bail!("Either --signal or --scenario must be specified");
    }

    Ok(())
}

fn run_scenario(conn: &zbus::blocking::Connection, path: &PathBuf) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(path)?;
    let scenario: Scenario = serde_json::from_str(&content)?;

    println!(
        "Playing scenario: {} ({} signals)",
        scenario.name,
        scenario.signals.len()
    );
    if let Some(desc) = &scenario.description {
        println!("  {}", desc);
    }

    for (i, entry) in scenario.signals.iter().enumerate() {
        if entry.delay_ms > 0 {
            println!("  Waiting {}ms...", entry.delay_ms);
            std::thread::sleep(std::time::Duration::from_millis(entry.delay_ms));
        }

        emit_scenario_signal(conn, entry)?;
        println!(
            "  [{}/{}] Emitted {} (level={})",
            i + 1,
            scenario.signals.len(),
            entry.signal,
            entry.level
        );
    }

    println!("Scenario complete.");
    Ok(())
}

fn emit_scenario_signal(
    conn: &zbus::blocking::Connection,
    entry: &ScenarioSignal,
) -> anyhow::Result<()> {
    match entry.signal.as_str() {
        "MemoryPressure" => {
            let body = (
                &entry.level,
                entry.available_bytes.unwrap_or(0),
                entry.reclaim_target_bytes.unwrap_or(0),
            );
            conn.emit_signal(
                Option::<&str>::None,
                MANAGER_PATH,
                MANAGER_IFACE,
                "MemoryPressure",
                &body,
            )?;
        }
        "ComputeGuidance" => {
            let body = (
                &entry.level,
                entry.recommended_backend.as_deref().unwrap_or("any"),
                entry.reason.as_deref().unwrap_or("balanced"),
                entry.cpu_usage_pct.unwrap_or(0.0),
                entry.gpu_usage_pct.unwrap_or(0.0),
            );
            conn.emit_signal(
                Option::<&str>::None,
                MANAGER_PATH,
                MANAGER_IFACE,
                "ComputeGuidance",
                &body,
            )?;
        }
        "ThermalAlert" => {
            let body = (
                &entry.level,
                entry.temperature_mc.unwrap_or(25000),
                entry.throttling_active.unwrap_or(false),
                entry.throttle_ratio.unwrap_or(1.0),
            );
            conn.emit_signal(
                Option::<&str>::None,
                MANAGER_PATH,
                MANAGER_IFACE,
                "ThermalAlert",
                &body,
            )?;
        }
        "EnergyConstraint" => {
            let body = (
                &entry.level,
                entry.reason.as_deref().unwrap_or("none"),
                entry.power_budget_mw.unwrap_or(0),
            );
            conn.emit_signal(
                Option::<&str>::None,
                MANAGER_PATH,
                MANAGER_IFACE,
                "EnergyConstraint",
                &body,
            )?;
        }
        other => {
            anyhow::bail!("Unknown signal type: {}", other);
        }
    }
    Ok(())
}

fn emit_single(
    conn: &zbus::blocking::Connection,
    args: &Args,
    signal_name: &str,
) -> anyhow::Result<()> {
    let level = args
        .level
        .as_deref()
        .unwrap_or("normal");

    match signal_name {
        "MemoryPressure" => {
            let body = (
                level,
                args.available_bytes.unwrap_or(0),
                args.reclaim_target.unwrap_or(0),
            );
            conn.emit_signal(
                Option::<&str>::None,
                MANAGER_PATH,
                MANAGER_IFACE,
                "MemoryPressure",
                &body,
            )?;
        }
        "ComputeGuidance" => {
            let body = (
                level,
                args.recommended_backend.as_deref().unwrap_or("any"),
                args.reason.as_deref().unwrap_or("balanced"),
                args.cpu_usage.unwrap_or(0.0),
                args.gpu_usage.unwrap_or(0.0),
            );
            conn.emit_signal(
                Option::<&str>::None,
                MANAGER_PATH,
                MANAGER_IFACE,
                "ComputeGuidance",
                &body,
            )?;
        }
        "ThermalAlert" => {
            let body = (
                level,
                args.temperature_mc.unwrap_or(25000),
                args.throttling_active.unwrap_or(false),
                args.throttle_ratio.unwrap_or(1.0),
            );
            conn.emit_signal(
                Option::<&str>::None,
                MANAGER_PATH,
                MANAGER_IFACE,
                "ThermalAlert",
                &body,
            )?;
        }
        "EnergyConstraint" => {
            let body = (
                level,
                args.reason.as_deref().unwrap_or("none"),
                args.power_budget_mw.unwrap_or(0),
            );
            conn.emit_signal(
                Option::<&str>::None,
                MANAGER_PATH,
                MANAGER_IFACE,
                "EnergyConstraint",
                &body,
            )?;
        }
        other => {
            anyhow::bail!("Unknown signal: {}", other);
        }
    }

    println!("Emitted {} (level={})", signal_name, level);
    Ok(())
}
