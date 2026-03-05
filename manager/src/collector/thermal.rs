use super::{Collector, Reading, ReadingData};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub struct ThermalCollector {
    poll_interval: Duration,
    thermal_base: String,
    /// Cached zone indices to monitor. Empty = discovered at first read.
    zone_indices: Vec<u32>,
    /// Zone type filter. Empty = all zones.
    zone_type_filter: Vec<String>,
    zones_discovered: bool,
}

impl ThermalCollector {
    pub fn new(poll_interval_ms: u64) -> Self {
        Self {
            poll_interval: Duration::from_millis(poll_interval_ms),
            thermal_base: "/sys/class/thermal".to_string(),
            zone_indices: Vec::new(),
            zone_type_filter: Vec::new(),
            zones_discovered: false,
        }
    }

    pub fn with_zone_filter(poll_interval_ms: u64, zone_types: Vec<String>) -> Self {
        Self {
            poll_interval: Duration::from_millis(poll_interval_ms),
            thermal_base: "/sys/class/thermal".to_string(),
            zone_indices: Vec::new(),
            zone_type_filter: zone_types,
            zones_discovered: false,
        }
    }

    #[cfg(test)]
    fn with_base(poll_interval_ms: u64, base: String) -> Self {
        Self {
            poll_interval: Duration::from_millis(poll_interval_ms),
            thermal_base: base,
            zone_indices: Vec::new(),
            zone_type_filter: Vec::new(),
            zones_discovered: false,
        }
    }

    #[cfg(test)]
    fn with_base_and_filter(poll_interval_ms: u64, base: String, zone_types: Vec<String>) -> Self {
        Self {
            poll_interval: Duration::from_millis(poll_interval_ms),
            thermal_base: base,
            zone_indices: Vec::new(),
            zone_type_filter: zone_types,
            zones_discovered: false,
        }
    }

    /// Discover thermal zones and cache matching indices.
    fn discover_zones(&mut self) {
        let base_path = std::path::Path::new(&self.thermal_base);
        let mut all_zones = Vec::new();

        for i in 0..32 {
            let type_path = base_path.join(format!("thermal_zone{}/type", i));
            let temp_path = base_path.join(format!("thermal_zone{}/temp", i));

            // Zone must have a temp file to be valid
            if !temp_path.exists() {
                continue;
            }

            let zone_type = std::fs::read_to_string(&type_path)
                .map(|s| s.trim().to_string())
                .unwrap_or_default();

            all_zones.push((i as u32, zone_type));
        }

        if self.zone_type_filter.is_empty() {
            // No filter: use all valid zones
            self.zone_indices = all_zones.iter().map(|(idx, _)| *idx).collect();
            log::info!(
                "[ThermalCollector] Monitoring all {} zones (no filter)",
                self.zone_indices.len()
            );
        } else {
            // Filter by type
            self.zone_indices = all_zones
                .iter()
                .filter(|(_, zone_type)| self.zone_type_filter.iter().any(|f| zone_type == f))
                .map(|(idx, _)| *idx)
                .collect();

            let matched: Vec<String> = all_zones
                .iter()
                .filter(|(_, zone_type)| self.zone_type_filter.iter().any(|f| zone_type == f))
                .map(|(idx, zt)| format!("zone{}({})", idx, zt))
                .collect();

            let skipped: Vec<String> = all_zones
                .iter()
                .filter(|(_, zone_type)| !self.zone_type_filter.iter().any(|f| zone_type == f))
                .map(|(idx, zt)| format!("zone{}({})", idx, zt))
                .collect();

            log::info!(
                "[ThermalCollector] Zone filter {:?}: matched [{}], skipped [{}]",
                self.zone_type_filter,
                matched.join(", "),
                skipped.join(", ")
            );

            if self.zone_indices.is_empty() {
                log::warn!(
                    "[ThermalCollector] No zones matched filter {:?}, falling back to all zones",
                    self.zone_type_filter
                );
                self.zone_indices = all_zones.iter().map(|(idx, _)| *idx).collect();
            }
        }

        self.zones_discovered = true;
    }

    fn read_once(&mut self) -> anyhow::Result<Reading> {
        if !self.zones_discovered {
            self.discover_zones();
        }

        let (max_temp, throttling) = read_thermal_state(&self.thermal_base, &self.zone_indices)?;
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
/// Only reads zones whose indices are in `zone_indices`.
fn read_thermal_state(base: &str, zone_indices: &[u32]) -> anyhow::Result<(i32, bool)> {
    let base_path = std::path::Path::new(base);

    // Find hottest thermal zone among filtered indices
    let mut max_temp: Option<i32> = None;
    for &i in zone_indices {
        let temp_path = base_path.join(format!("thermal_zone{}/temp", i));
        match std::fs::read_to_string(&temp_path) {
            Ok(content) => {
                if let Ok(temp) = content.trim().parse::<i32>() {
                    max_temp = Some(max_temp.map_or(temp, |cur: i32| cur.max(temp)));
                }
            }
            Err(_) => continue,
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
        std::fs::write(zone0.join("type"), "acpitz\n").unwrap();

        let zone1 = dir.path().join("thermal_zone1");
        std::fs::create_dir(&zone1).unwrap();
        std::fs::write(zone1.join("temp"), "62000\n").unwrap();
        std::fs::write(zone1.join("type"), "x86_pkg_temp\n").unwrap();

        // Create cooling device (not throttling)
        let cool0 = dir.path().join("cooling_device0");
        std::fs::create_dir(&cool0).unwrap();
        std::fs::write(cool0.join("cur_state"), "0\n").unwrap();

        dir
    }

    #[test]
    fn reads_hottest_zone() {
        let dir = setup_thermal_dir();
        let (temp, throttling) = read_thermal_state(dir.path().to_str().unwrap(), &[0, 1]).unwrap();
        assert_eq!(temp, 62000);
        assert!(!throttling);
    }

    #[test]
    fn detects_throttling() {
        let dir = setup_thermal_dir();
        // Set cooling device to active
        std::fs::write(dir.path().join("cooling_device0/cur_state"), "5\n").unwrap();

        let (_, throttling) = read_thermal_state(dir.path().to_str().unwrap(), &[0, 1]).unwrap();
        assert!(throttling);
    }

    #[test]
    fn collector_no_filter_uses_all_zones() {
        let dir = setup_thermal_dir();
        let mut collector =
            ThermalCollector::with_base(100, dir.path().to_str().unwrap().to_string());

        let reading = collector.read_once().unwrap();
        if let ReadingData::Thermal {
            temperature_mc,
            throttling_active,
        } = reading.data
        {
            // Max of 45000 and 62000
            assert_eq!(temperature_mc, 62000);
            assert!(!throttling_active);
        } else {
            panic!("Expected Thermal reading");
        }
    }

    #[test]
    fn collector_filters_by_zone_type() {
        let dir = setup_thermal_dir();

        // Add a third zone: WiFi at 70°C (should be excluded)
        let zone2 = dir.path().join("thermal_zone2");
        std::fs::create_dir(&zone2).unwrap();
        std::fs::write(zone2.join("temp"), "70000\n").unwrap();
        std::fs::write(zone2.join("type"), "iwlwifi_1\n").unwrap();

        // Only monitor x86_pkg_temp
        let mut collector = ThermalCollector::with_base_and_filter(
            100,
            dir.path().to_str().unwrap().to_string(),
            vec!["x86_pkg_temp".to_string()],
        );

        let reading = collector.read_once().unwrap();
        if let ReadingData::Thermal { temperature_mc, .. } = reading.data {
            // Should only see zone1 (62000), NOT zone2 (70000 WiFi)
            assert_eq!(temperature_mc, 62000);
        } else {
            panic!("Expected Thermal reading");
        }
    }

    #[test]
    fn collector_filter_multiple_types() {
        let dir = setup_thermal_dir();

        // Filter for both acpitz and x86_pkg_temp
        let mut collector = ThermalCollector::with_base_and_filter(
            100,
            dir.path().to_str().unwrap().to_string(),
            vec!["acpitz".to_string(), "x86_pkg_temp".to_string()],
        );

        let reading = collector.read_once().unwrap();
        if let ReadingData::Thermal { temperature_mc, .. } = reading.data {
            // Max of zone0 (45000) and zone1 (62000)
            assert_eq!(temperature_mc, 62000);
        } else {
            panic!("Expected Thermal reading");
        }
    }

    #[test]
    fn collector_fallback_on_no_match() {
        let dir = setup_thermal_dir();

        // Filter for nonexistent type
        let mut collector = ThermalCollector::with_base_and_filter(
            100,
            dir.path().to_str().unwrap().to_string(),
            vec!["nonexistent_sensor".to_string()],
        );

        // Should fall back to all zones
        let reading = collector.read_once().unwrap();
        if let ReadingData::Thermal { temperature_mc, .. } = reading.data {
            assert_eq!(temperature_mc, 62000);
        } else {
            panic!("Expected Thermal reading");
        }
    }
}
