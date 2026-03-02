use std::sync::mpsc;

use super::signal::{ComputeReason, EnergyReason, Level, RecommendedBackend, SystemSignal};

/// D-Bus well-known name for the LLM resource manager.
const MANAGER_DEST: &str = "org.llm.Manager1";
/// D-Bus object path for the LLM resource manager.
const MANAGER_PATH: &str = "/org/llm/Manager1";
/// D-Bus interface for the LLM resource manager.
const MANAGER_IFACE: &str = "org.llm.Manager1";

/// D-Bus listener running in a separate thread.
/// Connects to System Bus and receives signals from `org.llm.Manager1`.
pub struct DbusListener {
    tx: mpsc::Sender<SystemSignal>,
}

impl DbusListener {
    pub fn new(tx: mpsc::Sender<SystemSignal>) -> Self {
        Self { tx }
    }

    /// Start D-Bus signal reception loop in a separate thread.
    /// If Manager unavailable or D-Bus connection fails,
    /// logs a warning and exits gracefully — LLM continues without resilience.
    pub fn spawn(self) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            if let Err(e) = self.run() {
                log::warn!(
                    "D-Bus listener exited: {}. LLM continues without resilience.",
                    e
                );
            }
        })
    }

    fn run(&self) -> anyhow::Result<()> {
        // 1. Connect to System Bus (blocking)
        let conn = zbus::blocking::Connection::system()?;

        // 2. Create proxy for org.llm.Manager1
        let proxy = zbus::blocking::Proxy::new(&conn, MANAGER_DEST, MANAGER_PATH, MANAGER_IFACE)?;

        log::info!("D-Bus listener connected to {}", MANAGER_DEST);

        // 3. Receive all signals via blocking iterator
        let signals = proxy.receive_all_signals()?;

        // 4. Signal reception loop
        for msg in signals {
            let header = msg.header();
            let member = match header.member() {
                Some(m) => m.to_owned(),
                None => continue,
            };

            let result = match member.as_str() {
                "MemoryPressure" => parse_memory_pressure(&msg),
                "ComputeGuidance" => parse_compute_guidance(&msg),
                "ThermalAlert" => parse_thermal_alert(&msg),
                "EnergyConstraint" => parse_energy_constraint(&msg),
                _ => {
                    log::debug!("Unknown signal: {}", member);
                    continue;
                }
            };

            match result {
                Ok(sys_signal) => {
                    log::debug!("Received D-Bus signal: {:?}", sys_signal);
                    if self.tx.send(sys_signal).is_err() {
                        log::info!("Receiver dropped. Stopping D-Bus listener.");
                        break;
                    }
                }
                Err(e) => {
                    log::warn!("Failed to parse signal {}: {}", member, e);
                }
            }
        }

        log::info!("D-Bus listener stopped.");
        Ok(())
    }
}

/// Parse MemoryPressure signal: (level: s, available_bytes: t, reclaim_target_bytes: t)
fn parse_memory_pressure(msg: &zbus::Message) -> anyhow::Result<SystemSignal> {
    let body = msg.body();
    let (level_str, available_bytes, reclaim_target_bytes): (String, u64, u64) =
        body.deserialize()?;

    let level = Level::from_dbus_str(&level_str)
        .ok_or_else(|| anyhow::anyhow!("Invalid level: {}", level_str))?;

    Ok(SystemSignal::MemoryPressure {
        level,
        available_bytes,
        reclaim_target_bytes,
    })
}

/// Parse ComputeGuidance signal: (level: s, recommended_backend: s, reason: s, cpu_usage_pct: d, gpu_usage_pct: d)
fn parse_compute_guidance(msg: &zbus::Message) -> anyhow::Result<SystemSignal> {
    let body = msg.body();
    let (level_str, backend_str, reason_str, cpu_usage_pct, gpu_usage_pct): (
        String,
        String,
        String,
        f64,
        f64,
    ) = body.deserialize()?;

    let level = Level::from_dbus_str(&level_str)
        .ok_or_else(|| anyhow::anyhow!("Invalid level: {}", level_str))?;
    let recommended_backend = RecommendedBackend::from_dbus_str(&backend_str)
        .ok_or_else(|| anyhow::anyhow!("Invalid backend: {}", backend_str))?;
    let reason = ComputeReason::from_dbus_str(&reason_str)
        .ok_or_else(|| anyhow::anyhow!("Invalid reason: {}", reason_str))?;

    Ok(SystemSignal::ComputeGuidance {
        level,
        recommended_backend,
        reason,
        cpu_usage_pct,
        gpu_usage_pct,
    })
}

/// Parse ThermalAlert signal: (level: s, temperature_mc: i, throttling_active: b, throttle_ratio: d)
fn parse_thermal_alert(msg: &zbus::Message) -> anyhow::Result<SystemSignal> {
    let body = msg.body();
    let (level_str, temperature_mc, throttling_active, throttle_ratio): (String, i32, bool, f64) =
        body.deserialize()?;

    let level = Level::from_dbus_str(&level_str)
        .ok_or_else(|| anyhow::anyhow!("Invalid level: {}", level_str))?;

    Ok(SystemSignal::ThermalAlert {
        level,
        temperature_mc,
        throttling_active,
        throttle_ratio,
    })
}

/// Parse EnergyConstraint signal: (level: s, reason: s, power_budget_mw: u)
fn parse_energy_constraint(msg: &zbus::Message) -> anyhow::Result<SystemSignal> {
    let body = msg.body();
    let (level_str, reason_str, power_budget_mw): (String, String, u32) = body.deserialize()?;

    let level = Level::from_dbus_str(&level_str)
        .ok_or_else(|| anyhow::anyhow!("Invalid level: {}", level_str))?;
    let reason = EnergyReason::from_dbus_str(&reason_str)
        .ok_or_else(|| anyhow::anyhow!("Invalid reason: {}", reason_str))?;

    Ok(SystemSignal::EnergyConstraint {
        level,
        reason,
        power_budget_mw,
    })
}
