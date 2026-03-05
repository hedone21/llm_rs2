use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::Instant;

/// Resource category for readings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    Memory,
    Thermal,
    Compute,
    Energy,
}

/// A single data reading from a collector.
#[derive(Debug, Clone)]
pub struct Reading {
    pub timestamp: Instant,
    pub data: ReadingData,
}

/// Resource-specific reading data.
#[derive(Debug, Clone)]
pub enum ReadingData {
    Memory {
        available_bytes: u64,
        total_bytes: u64,
        /// PSI "some" avg10 (microseconds per second). None if PSI unavailable.
        psi_some_avg10: Option<f64>,
    },
    Thermal {
        /// Temperature in millidegrees Celsius (e.g., 75000 = 75.0C).
        temperature_mc: i32,
        /// Whether hardware throttling is currently active.
        throttling_active: bool,
    },
    Compute {
        /// System-wide CPU usage (0.0-100.0).
        cpu_usage_pct: f64,
        /// GPU usage (0.0-100.0).
        gpu_usage_pct: f64,
    },
    Energy {
        /// Battery level (0.0-100.0). None if no battery.
        battery_pct: Option<f64>,
        /// Whether the device is charging.
        charging: bool,
        /// Current power draw in milliwatts. None if unavailable.
        power_draw_mw: Option<u32>,
    },
}

impl ReadingData {
    pub fn kind(&self) -> ResourceKind {
        match self {
            ReadingData::Memory { .. } => ResourceKind::Memory,
            ReadingData::Thermal { .. } => ResourceKind::Thermal,
            ReadingData::Compute { .. } => ResourceKind::Compute,
            ReadingData::Energy { .. } => ResourceKind::Energy,
        }
    }
}

/// Collector gathers system resource data from a specific source.
///
/// Each collector runs in a dedicated thread and sends [`Reading`]s
/// to the central channel. Implementations may be event-driven
/// (preferred, e.g., PSI events, D-Bus signals) or polling-based
/// (fallback for sources without event support).
pub trait Collector: Send + 'static {
    /// Start collecting data. Called in a dedicated thread.
    ///
    /// Should send readings to `tx` when new data is available
    /// and return `Ok(())` when `shutdown` becomes true.
    fn run(&mut self, tx: mpsc::Sender<Reading>, shutdown: Arc<AtomicBool>) -> anyhow::Result<()>;

    /// Human-readable name for logging.
    fn name(&self) -> &str;
}
