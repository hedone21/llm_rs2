use anyhow::{Result, anyhow};
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    pub total: usize,
    pub available: usize,
    pub free: usize,
}

pub trait SystemMonitor: Send + Sync {
    fn mem_stats(&self) -> Result<MemoryStats>;
}

pub struct LinuxSystemMonitor;

impl LinuxSystemMonitor {
    pub fn new() -> Self {
        Self
    }

    fn parse_meminfo(content: &str) -> Result<MemoryStats> {
        let mut total = None;
        let mut available = None;
        let mut free = None;

        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            let value = parts[1].parse::<usize>()?; // KB
            // meminfo uses KB, we want Bytes? Or just keep KB. Let's keep KB for internal, but maybe convert to bytes?
            // "MemTotal:       32768480 kB"
            // Let's standardise on Bytes.
            let value_bytes = value * 1024;

            match parts[0] {
                "MemTotal:" => total = Some(value_bytes),
                "MemAvailable:" => available = Some(value_bytes),
                "MemFree:" => free = Some(value_bytes),
                _ => {}
            }
        }

        Ok(MemoryStats {
            total: total.ok_or(anyhow!("MemTotal not found"))?,
            available: available.ok_or(anyhow!("MemAvailable not found"))?,
            free: free.ok_or(anyhow!("MemFree not found"))?,
        })
    }
}

impl SystemMonitor for LinuxSystemMonitor {
    fn mem_stats(&self) -> Result<MemoryStats> {
        let file = File::open("/proc/meminfo")?;
        let reader = BufReader::new(file);

        // Read file into string for parsing (or stream parsing)
        // Since /proc/meminfo is small, reading to string is fine.
        let mut content = String::new();
        // We only need the first few lines essentially, but reading all is safe.
        // Actually BufReader is good.
        // Let's reuse the static parsing function to make it testable
        // But we need to read it first.

        let content = std::fs::read_to_string("/proc/meminfo")?;
        Self::parse_meminfo(&content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linux_monitor_parsing() {
        let mock_meminfo = r#"
MemTotal:       32768480 kB
MemFree:         4567890 kB
MemAvailable:   12345678 kB
Buffers:          123456 kB
"#;
        let stats = LinuxSystemMonitor::parse_meminfo(mock_meminfo).unwrap();

        assert_eq!(stats.total, 32768480 * 1024);
        assert_eq!(stats.available, 12345678 * 1024);
        assert_eq!(stats.free, 4567890 * 1024);
    }
}
