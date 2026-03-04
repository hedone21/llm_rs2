use super::signal::{ComputeReason, EnergyReason, Level, RecommendedBackend, SystemSignal};
use super::transport::{Transport, TransportError};

/// D-Bus well-known name for the LLM resource manager.
const MANAGER_DEST: &str = "org.llm.Manager1";
/// D-Bus object path for the LLM resource manager.
const MANAGER_PATH: &str = "/org/llm/Manager1";
/// D-Bus interface for the LLM resource manager.
const MANAGER_IFACE: &str = "org.llm.Manager1";

/// D-Bus transport that connects to `org.llm.Manager1` on the System Bus.
pub struct DbusTransport {
    conn: Option<zbus::blocking::Connection>,
    proxy: Option<zbus::blocking::Proxy<'static>>,
    signals: Option<Box<dyn Iterator<Item = zbus::Message> + Send>>,
}

impl DbusTransport {
    pub fn new() -> Self {
        Self {
            conn: None,
            proxy: None,
            signals: None,
        }
    }
}

impl Transport for DbusTransport {
    fn connect(&mut self) -> Result<(), TransportError> {
        let conn = zbus::blocking::Connection::system()
            .map_err(|e| TransportError::ConnectionFailed(format!("D-Bus system bus: {}", e)))?;

        let proxy = zbus::blocking::Proxy::new(&conn, MANAGER_DEST, MANAGER_PATH, MANAGER_IFACE)
            .map_err(|e| TransportError::ConnectionFailed(format!("D-Bus proxy: {}", e)))?;

        let signals = proxy.receive_all_signals().map_err(|e| {
            TransportError::ConnectionFailed(format!("D-Bus signal iterator: {}", e))
        })?;

        // Store connection and proxy to keep them alive.
        // Use 'static lifetime by leaking the connection (it lives for the process lifetime).
        self.conn = Some(conn);
        self.proxy = Some(proxy);
        self.signals = Some(Box::new(signals));

        log::info!("D-Bus transport connected to {}", MANAGER_DEST);
        Ok(())
    }

    fn recv(&mut self) -> Result<SystemSignal, TransportError> {
        let signals = self
            .signals
            .as_mut()
            .ok_or_else(|| TransportError::ConnectionFailed("not connected".into()))?;

        loop {
            let msg = match signals.next() {
                Some(msg) => msg,
                None => return Err(TransportError::Disconnected),
            };

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
                    log::debug!("Unknown D-Bus signal: {}", member);
                    continue;
                }
            };

            return result.map_err(|e| TransportError::ParseError(format!("{}: {}", member, e)));
        }
    }

    fn name(&self) -> &str {
        "D-Bus"
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
