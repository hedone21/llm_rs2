use super::{Collector, Reading, ReadingData};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub struct MemoryCollector {
    poll_interval: Duration,
    meminfo_path: &'static str,
    psi_path: &'static str,
}

impl MemoryCollector {
    pub fn new(poll_interval_ms: u64) -> Self {
        Self {
            poll_interval: Duration::from_millis(poll_interval_ms),
            meminfo_path: "/proc/meminfo",
            psi_path: "/proc/pressure/memory",
        }
    }

    #[cfg(test)]
    fn with_paths(
        poll_interval_ms: u64,
        meminfo_path: &'static str,
        psi_path: &'static str,
    ) -> Self {
        Self {
            poll_interval: Duration::from_millis(poll_interval_ms),
            meminfo_path,
            psi_path,
        }
    }

    fn read_once(&self) -> anyhow::Result<Reading> {
        let (total, available) = parse_meminfo(self.meminfo_path)?;
        let psi = parse_psi_memory(self.psi_path).ok();

        Ok(Reading {
            timestamp: Instant::now(),
            data: ReadingData::Memory {
                available_bytes: available,
                total_bytes: total,
                psi_some_avg10: psi,
            },
        })
    }
}

impl Collector for MemoryCollector {
    fn run(&mut self, tx: mpsc::Sender<Reading>, shutdown: Arc<AtomicBool>) -> anyhow::Result<()> {
        log::info!(
            "[MemoryCollector] Starting (interval={}ms)",
            self.poll_interval.as_millis()
        );

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            match self.read_once() {
                Ok(reading) => {
                    if tx.send(reading).is_err() {
                        break; // receiver dropped
                    }
                }
                Err(e) => log::warn!("[MemoryCollector] Read failed: {}", e),
            }

            std::thread::sleep(self.poll_interval);
        }

        log::info!("[MemoryCollector] Stopped");
        Ok(())
    }

    fn name(&self) -> &str {
        "MemoryCollector"
    }
}

/// Parse `/proc/meminfo` for MemTotal and MemAvailable (returns bytes).
fn parse_meminfo(path: &str) -> anyhow::Result<(u64, u64)> {
    let content = std::fs::read_to_string(path)?;
    let mut total_kb = None;
    let mut available_kb = None;

    for line in content.lines() {
        if let Some(val) = line.strip_prefix("MemTotal:") {
            total_kb = Some(parse_kb_value(val)?);
        } else if let Some(val) = line.strip_prefix("MemAvailable:") {
            available_kb = Some(parse_kb_value(val)?);
        }
        if total_kb.is_some() && available_kb.is_some() {
            break;
        }
    }

    let total = total_kb.ok_or_else(|| anyhow::anyhow!("MemTotal not found"))?;
    let available = available_kb.ok_or_else(|| anyhow::anyhow!("MemAvailable not found"))?;

    Ok((total * 1024, available * 1024))
}

/// Parse "   12345 kB" → 12345u64
fn parse_kb_value(s: &str) -> anyhow::Result<u64> {
    let s = s.trim().trim_end_matches("kB").trim();
    Ok(s.parse()?)
}

/// Parse `/proc/pressure/memory` for "some avg10" value.
fn parse_psi_memory(path: &str) -> anyhow::Result<f64> {
    let content = std::fs::read_to_string(path)?;
    // First line: "some avg10=0.00 avg60=0.00 avg300=0.00 total=3"
    let line = content
        .lines()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Empty PSI file"))?;

    for field in line.split_whitespace() {
        if let Some(val) = field.strip_prefix("avg10=") {
            return Ok(val.parse()?);
        }
    }

    anyhow::bail!("avg10 not found in PSI output")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parse_meminfo_valid() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(
            f,
            "MemTotal:       65280212 kB\nMemFree:        40000000 kB\nMemAvailable:   55553552 kB\n"
        )
        .unwrap();

        let (total, avail) = parse_meminfo(f.path().to_str().unwrap()).unwrap();
        assert_eq!(total, 65280212 * 1024);
        assert_eq!(avail, 55553552 * 1024);
    }

    #[test]
    fn parse_psi_valid() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(
            f,
            "some avg10=1.50 avg60=0.75 avg300=0.25 total=123456\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
        )
        .unwrap();

        let val = parse_psi_memory(f.path().to_str().unwrap()).unwrap();
        assert!((val - 1.50).abs() < 0.001);
    }

    #[test]
    fn collector_produces_reading() {
        let mut f_mem = tempfile::NamedTempFile::new().unwrap();
        write!(
            f_mem,
            "MemTotal:       1000000 kB\nMemAvailable:   500000 kB\n"
        )
        .unwrap();

        let mut f_psi = tempfile::NamedTempFile::new().unwrap();
        write!(f_psi, "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n").unwrap();

        let collector = MemoryCollector::with_paths(
            100,
            // SAFETY: NamedTempFile paths are valid for the test duration
            unsafe { &*(f_mem.path().to_str().unwrap() as *const str) },
            unsafe { &*(f_psi.path().to_str().unwrap() as *const str) },
        );

        let reading = collector.read_once().unwrap();
        if let ReadingData::Memory {
            total_bytes,
            available_bytes,
            psi_some_avg10,
        } = reading.data
        {
            assert_eq!(total_bytes, 1000000 * 1024);
            assert_eq!(available_bytes, 500000 * 1024);
            assert_eq!(psi_some_avg10, Some(0.0));
        } else {
            panic!("Expected Memory reading");
        }
    }
}
