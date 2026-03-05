use super::{Collector, Reading, ReadingData};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub struct ComputeCollector {
    poll_interval: Duration,
    stat_path: String,
    prev_snapshot: Option<CpuSnapshot>,
}

#[derive(Debug, Clone)]
struct CpuSnapshot {
    total: u64,
    idle: u64,
}

impl ComputeCollector {
    pub fn new(poll_interval_ms: u64) -> Self {
        Self {
            poll_interval: Duration::from_millis(poll_interval_ms),
            stat_path: "/proc/stat".to_string(),
            prev_snapshot: None,
        }
    }

    #[cfg(test)]
    fn with_path(poll_interval_ms: u64, path: String) -> Self {
        Self {
            poll_interval: Duration::from_millis(poll_interval_ms),
            stat_path: path,
            prev_snapshot: None,
        }
    }

    fn read_once(&mut self) -> anyhow::Result<Option<Reading>> {
        let snap = read_cpu_snapshot(&self.stat_path)?;

        let result = if let Some(prev) = &self.prev_snapshot {
            let total_delta = snap.total.saturating_sub(prev.total);
            let idle_delta = snap.idle.saturating_sub(prev.idle);

            let cpu_pct = if total_delta > 0 {
                ((total_delta - idle_delta) as f64 / total_delta as f64) * 100.0
            } else {
                0.0
            };

            Some(Reading {
                timestamp: Instant::now(),
                data: ReadingData::Compute {
                    cpu_usage_pct: cpu_pct,
                    // GPU usage requires vendor-specific sysfs — not implemented yet
                    gpu_usage_pct: 0.0,
                },
            })
        } else {
            None // First read, need a second snapshot for delta
        };

        self.prev_snapshot = Some(snap);
        Ok(result)
    }
}

impl Collector for ComputeCollector {
    fn run(&mut self, tx: mpsc::Sender<Reading>, shutdown: Arc<AtomicBool>) -> anyhow::Result<()> {
        log::info!(
            "[ComputeCollector] Starting (interval={}ms)",
            self.poll_interval.as_millis()
        );

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            match self.read_once() {
                Ok(Some(reading)) => {
                    if tx.send(reading).is_err() {
                        break;
                    }
                }
                Ok(None) => {} // First snapshot, no delta yet
                Err(e) => log::warn!("[ComputeCollector] Read failed: {}", e),
            }

            std::thread::sleep(self.poll_interval);
        }

        log::info!("[ComputeCollector] Stopped");
        Ok(())
    }

    fn name(&self) -> &str {
        "ComputeCollector"
    }
}

/// Parse first line of `/proc/stat`:
/// `cpu  user nice system idle iowait irq softirq steal guest guest_nice`
fn read_cpu_snapshot(path: &str) -> anyhow::Result<CpuSnapshot> {
    let content = std::fs::read_to_string(path)?;
    let line = content
        .lines()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Empty /proc/stat"))?;

    let fields: Vec<u64> = line
        .split_whitespace()
        .skip(1) // skip "cpu"
        .filter_map(|s| s.parse().ok())
        .collect();

    if fields.len() < 4 {
        anyhow::bail!("Unexpected /proc/stat format: too few fields");
    }

    // fields: [user, nice, system, idle, iowait?, irq?, softirq?, steal?, ...]
    let total: u64 = fields.iter().sum();
    let idle = fields[3]; // idle is the 4th field

    Ok(CpuSnapshot { total, idle })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_stat(path: &std::path::Path, user: u64, nice: u64, sys: u64, idle: u64) {
        let content = format!(
            "cpu  {} {} {} {} 100 50 30 0 0 0\ncpu0 {} {} {} {} 100 50 30 0 0 0\n",
            user, nice, sys, idle, user, nice, sys, idle
        );
        std::fs::write(path, content).unwrap();
    }

    #[test]
    fn cpu_snapshot_parsing() {
        let f = tempfile::NamedTempFile::new().unwrap();
        write_stat(f.path(), 1000, 100, 500, 8000);

        let snap = read_cpu_snapshot(f.path().to_str().unwrap()).unwrap();
        // total = 1000+100+500+8000+100+50+30+0+0+0 = 9780
        // idle = 8000
        assert_eq!(snap.total, 9780);
        assert_eq!(snap.idle, 8000);
    }

    #[test]
    fn compute_delta_calculation() {
        let f = tempfile::NamedTempFile::new().unwrap();

        // First snapshot: low usage
        write_stat(f.path(), 1000, 0, 500, 8000);
        let mut collector =
            ComputeCollector::with_path(100, f.path().to_str().unwrap().to_string());

        // First read returns None (no delta yet)
        assert!(collector.read_once().unwrap().is_none());

        // Second snapshot: more work, less idle
        write_stat(f.path(), 1500, 0, 700, 8200);
        let reading = collector.read_once().unwrap().unwrap();

        if let ReadingData::Compute {
            cpu_usage_pct,
            gpu_usage_pct,
        } = reading.data
        {
            // Delta: total = (1500+700+8200+100+50+30) - (1000+500+8000+100+50+30) = 10580-9680 = 900
            // idle delta = 8200-8000 = 200
            // usage = (900-200)/900 = 77.8%
            assert!(
                cpu_usage_pct > 70.0 && cpu_usage_pct < 85.0,
                "got {}",
                cpu_usage_pct
            );
            assert_eq!(gpu_usage_pct, 0.0);
        } else {
            panic!("Expected Compute reading");
        }
    }
}
