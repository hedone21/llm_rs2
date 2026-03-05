use super::{Collector, Reading, ReadingData};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub struct EnergyCollector {
    poll_interval: Duration,
    battery_path: Option<PathBuf>,
}

impl EnergyCollector {
    pub fn new(poll_interval_ms: u64) -> Self {
        let battery_path = find_battery_path("/sys/class/power_supply");
        if let Some(ref path) = battery_path {
            log::info!("[EnergyCollector] Found battery at {}", path.display());
        } else {
            log::info!("[EnergyCollector] No battery found, will report defaults");
        }
        Self {
            poll_interval: Duration::from_millis(poll_interval_ms),
            battery_path,
        }
    }

    #[cfg(test)]
    fn with_path(poll_interval_ms: u64, battery_path: Option<PathBuf>) -> Self {
        Self {
            poll_interval: Duration::from_millis(poll_interval_ms),
            battery_path,
        }
    }

    fn read_once(&self) -> Reading {
        let (battery_pct, charging, power_mw) = match &self.battery_path {
            Some(path) => read_battery_state(path),
            None => (None, false, None),
        };

        Reading {
            timestamp: Instant::now(),
            data: ReadingData::Energy {
                battery_pct,
                charging,
                power_draw_mw: power_mw,
            },
        }
    }
}

impl Collector for EnergyCollector {
    fn run(&mut self, tx: mpsc::Sender<Reading>, shutdown: Arc<AtomicBool>) -> anyhow::Result<()> {
        log::info!(
            "[EnergyCollector] Starting (interval={}ms, battery={})",
            self.poll_interval.as_millis(),
            self.battery_path
                .as_ref()
                .map_or("none".to_string(), |p| p.display().to_string())
        );

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            let reading = self.read_once();
            if tx.send(reading).is_err() {
                break;
            }

            std::thread::sleep(self.poll_interval);
        }

        log::info!("[EnergyCollector] Stopped");
        Ok(())
    }

    fn name(&self) -> &str {
        "EnergyCollector"
    }
}

/// Find the first battery device in `/sys/class/power_supply/`.
fn find_battery_path(base: &str) -> Option<PathBuf> {
    let base_path = Path::new(base);
    let entries = std::fs::read_dir(base_path).ok()?;

    for entry in entries.flatten() {
        let path = entry.path();
        let type_path = path.join("type");
        if let Ok(device_type) = std::fs::read_to_string(&type_path)
            && device_type.trim().eq_ignore_ascii_case("battery")
            && path.join("capacity").exists()
        {
            return Some(path);
        }
    }
    None
}

/// Read battery state from sysfs. Returns (capacity%, is_charging, power_mw).
fn read_battery_state(path: &Path) -> (Option<f64>, bool, Option<u32>) {
    let capacity = std::fs::read_to_string(path.join("capacity"))
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok());

    let charging = std::fs::read_to_string(path.join("status"))
        .ok()
        .map(|s| {
            let status = s.trim().to_lowercase();
            status == "charging" || status == "full"
        })
        .unwrap_or(false);

    // power_now is in microwatts, convert to milliwatts
    let power_mw = std::fs::read_to_string(path.join("power_now"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|uw| (uw / 1000) as u32);

    (capacity, charging, power_mw)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_battery_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();

        let bat = dir.path().join("BAT0");
        std::fs::create_dir(&bat).unwrap();
        std::fs::write(bat.join("type"), "Battery\n").unwrap();
        std::fs::write(bat.join("capacity"), "72\n").unwrap();
        std::fs::write(bat.join("status"), "Discharging\n").unwrap();
        std::fs::write(bat.join("power_now"), "15000000\n").unwrap(); // 15W

        dir
    }

    #[test]
    fn find_battery() {
        let dir = setup_battery_dir();
        let path = find_battery_path(dir.path().to_str().unwrap());
        assert!(path.is_some());
        assert!(path.unwrap().ends_with("BAT0"));
    }

    #[test]
    fn read_battery_discharging() {
        let dir = setup_battery_dir();
        let bat_path = dir.path().join("BAT0");
        let (cap, charging, power) = read_battery_state(&bat_path);

        assert_eq!(cap, Some(72.0));
        assert!(!charging);
        assert_eq!(power, Some(15000)); // 15W = 15000mW
    }

    #[test]
    fn read_battery_charging() {
        let dir = setup_battery_dir();
        let bat_path = dir.path().join("BAT0");
        std::fs::write(bat_path.join("status"), "Charging\n").unwrap();

        let (_, charging, _) = read_battery_state(&bat_path);
        assert!(charging);
    }

    #[test]
    fn collector_no_battery() {
        let collector = EnergyCollector::with_path(100, None);
        let reading = collector.read_once();
        if let ReadingData::Energy {
            battery_pct,
            charging,
            power_draw_mw,
        } = reading.data
        {
            assert_eq!(battery_pct, None);
            assert!(!charging);
            assert_eq!(power_draw_mw, None);
        } else {
            panic!("Expected Energy reading");
        }
    }

    #[test]
    fn collector_with_battery() {
        let dir = setup_battery_dir();
        let collector = EnergyCollector::with_path(100, Some(dir.path().join("BAT0")));
        let reading = collector.read_once();
        if let ReadingData::Energy {
            battery_pct,
            charging,
            ..
        } = reading.data
        {
            assert_eq!(battery_pct, Some(72.0));
            assert!(!charging);
        } else {
            panic!("Expected Energy reading");
        }
    }
}
