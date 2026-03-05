use super::{Collector, Reading, ReadingData};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub struct ThermalCollector {
    poll_interval: Duration,
    thermal_base: String,
}

impl ThermalCollector {
    pub fn new(poll_interval_ms: u64) -> Self {
        Self {
            poll_interval: Duration::from_millis(poll_interval_ms),
            thermal_base: "/sys/class/thermal".to_string(),
        }
    }

    #[cfg(test)]
    fn with_base(poll_interval_ms: u64, base: String) -> Self {
        Self {
            poll_interval: Duration::from_millis(poll_interval_ms),
            thermal_base: base,
        }
    }

    fn read_once(&self) -> anyhow::Result<Reading> {
        let (max_temp, throttling) = read_thermal_state(&self.thermal_base)?;
        Ok(Reading {
            timestamp: Instant::now(),
            data: ReadingData::Thermal {
                temperature_mc: max_temp,
                throttling_active: throttling,
            },
        })
    }
}

impl Collector for ThermalCollector {
    fn run(&mut self, tx: mpsc::Sender<Reading>, shutdown: Arc<AtomicBool>) -> anyhow::Result<()> {
        log::info!(
            "[ThermalCollector] Starting (interval={}ms)",
            self.poll_interval.as_millis()
        );

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            match self.read_once() {
                Ok(reading) => {
                    if tx.send(reading).is_err() {
                        break;
                    }
                }
                Err(e) => log::warn!("[ThermalCollector] Read failed: {}", e),
            }

            std::thread::sleep(self.poll_interval);
        }

        log::info!("[ThermalCollector] Stopped");
        Ok(())
    }

    fn name(&self) -> &str {
        "ThermalCollector"
    }
}

/// Read thermal zones and cooling devices. Returns (max_temperature_mc, throttling_active).
fn read_thermal_state(base: &str) -> anyhow::Result<(i32, bool)> {
    let base_path = std::path::Path::new(base);

    // Find hottest thermal zone
    let mut max_temp: Option<i32> = None;
    for i in 0..32 {
        let temp_path = base_path.join(format!("thermal_zone{}/temp", i));
        match std::fs::read_to_string(&temp_path) {
            Ok(content) => {
                if let Ok(temp) = content.trim().parse::<i32>() {
                    max_temp = Some(max_temp.map_or(temp, |cur: i32| cur.max(temp)));
                }
            }
            Err(_) => break, // No more zones
        }
    }

    let max_temp = max_temp.ok_or_else(|| anyhow::anyhow!("No thermal zones found"))?;

    // Check cooling devices for throttling
    let mut throttling = false;
    for i in 0..64 {
        let cur_path = base_path.join(format!("cooling_device{}/cur_state", i));
        match std::fs::read_to_string(&cur_path) {
            Ok(content) => {
                if let Ok(state) = content.trim().parse::<u32>()
                    && state > 0
                {
                    throttling = true;
                    break;
                }
            }
            Err(_) => break,
        }
    }

    Ok((max_temp, throttling))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_thermal_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();

        // Create 2 thermal zones
        let zone0 = dir.path().join("thermal_zone0");
        std::fs::create_dir(&zone0).unwrap();
        std::fs::write(zone0.join("temp"), "45000\n").unwrap();

        let zone1 = dir.path().join("thermal_zone1");
        std::fs::create_dir(&zone1).unwrap();
        std::fs::write(zone1.join("temp"), "62000\n").unwrap();

        // Create cooling device (not throttling)
        let cool0 = dir.path().join("cooling_device0");
        std::fs::create_dir(&cool0).unwrap();
        std::fs::write(cool0.join("cur_state"), "0\n").unwrap();

        dir
    }

    #[test]
    fn reads_hottest_zone() {
        let dir = setup_thermal_dir();
        let (temp, throttling) = read_thermal_state(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(temp, 62000);
        assert!(!throttling);
    }

    #[test]
    fn detects_throttling() {
        let dir = setup_thermal_dir();
        // Set cooling device to active
        std::fs::write(dir.path().join("cooling_device0/cur_state"), "5\n").unwrap();

        let (_, throttling) = read_thermal_state(dir.path().to_str().unwrap()).unwrap();
        assert!(throttling);
    }

    #[test]
    fn collector_produces_reading() {
        let dir = setup_thermal_dir();
        let collector = ThermalCollector::with_base(100, dir.path().to_str().unwrap().to_string());

        let reading = collector.read_once().unwrap();
        if let ReadingData::Thermal {
            temperature_mc,
            throttling_active,
        } = reading.data
        {
            assert_eq!(temperature_mc, 62000);
            assert!(!throttling_active);
        } else {
            panic!("Expected Thermal reading");
        }
    }
}
