use super::Monitor;
use crate::config::ComputeMonitorConfig;
use crate::evaluator::{Direction, ThresholdEvaluator, Thresholds};
use llm_shared::{ComputeReason, Level, RecommendedBackend, SystemSignal};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

/// max(cpu%, gpu%) → compute [`Level`] (ascending, Emergency not used — max is Critical).
///
/// Pure, stateless threshold check using the default `ComputeMonitorConfig` thresholds
/// (warning=70%, critical=90%). Hysteresis is not applied.
///
/// Exposed so the simulator can delegate level decisions here instead of
/// duplicating the threshold constants.
pub fn compute_level_from_pcts(cpu_pct: f64, gpu_pct: f64) -> Level {
    let cfg = ComputeMonitorConfig::default();
    let worst = cpu_pct.max(gpu_pct);
    if worst >= cfg.critical_pct {
        Level::Critical
    } else if worst >= cfg.warning_pct {
        Level::Warning
    } else {
        Level::Normal
    }
}

pub struct ComputeMonitor {
    poll_interval: Duration,
    evaluator: ThresholdEvaluator,
    warning_usage_pct: f64,
    stat_path: String,
    prev_snapshot: Option<CpuSnapshot>,
    last_cpu: f64,
    last_gpu: f64,
    recommended: RecommendedBackend,
    reason: ComputeReason,
}

#[derive(Debug, Clone)]
struct CpuSnapshot {
    total: u64,
    idle: u64,
}

impl ComputeMonitor {
    pub fn new(config: &ComputeMonitorConfig, default_poll_ms: u64) -> Self {
        Self {
            poll_interval: Duration::from_millis(
                config.poll_interval_ms.unwrap_or(default_poll_ms),
            ),
            evaluator: ThresholdEvaluator::new(
                Direction::Ascending,
                Thresholds {
                    warning: config.warning_pct,
                    critical: config.critical_pct,
                    emergency: f64::MAX, // ComputeGuidance has no Emergency level
                    hysteresis: config.hysteresis_pct,
                },
            ),
            warning_usage_pct: config.warning_pct,
            stat_path: "/proc/stat".into(),
            prev_snapshot: None,
            last_cpu: 0.0,
            last_gpu: 0.0,
            recommended: RecommendedBackend::Any,
            reason: ComputeReason::Balanced,
        }
    }

    #[cfg(test)]
    fn with_path(config: &ComputeMonitorConfig, path: String) -> Self {
        let mut m = Self::new(config, 1000);
        m.stat_path = path;
        m
    }

    fn read_once(&mut self) -> anyhow::Result<bool> {
        let snap = read_cpu_snapshot(&self.stat_path)?;

        let has_delta = if let Some(prev) = &self.prev_snapshot {
            let total_delta = snap.total.saturating_sub(prev.total);
            let idle_delta = snap.idle.saturating_sub(prev.idle);

            self.last_cpu = if total_delta > 0 {
                ((total_delta - idle_delta) as f64 / total_delta as f64) * 100.0
            } else {
                0.0
            };
            // GPU usage requires vendor-specific sysfs — not implemented yet
            self.last_gpu = 0.0;
            true
        } else {
            false
        };

        self.prev_snapshot = Some(snap);
        Ok(has_delta)
    }

    fn build_signal(&self) -> SystemSignal {
        SystemSignal::ComputeGuidance {
            level: self.evaluator.level(),
            recommended_backend: self.recommended,
            reason: self.reason,
            cpu_usage_pct: self.last_cpu,
            gpu_usage_pct: self.last_gpu,
        }
    }
}

impl Monitor for ComputeMonitor {
    fn run(
        &mut self,
        tx: mpsc::Sender<SystemSignal>,
        shutdown: Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        log::info!(
            "[ComputeMonitor] Starting (interval={}ms)",
            self.poll_interval.as_millis()
        );

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            match self.read_once() {
                Ok(true) => {
                    let worst = self.last_cpu.max(self.last_gpu);
                    self.evaluator.evaluate(worst);

                    let (new_rec, new_reason) = compute_recommendation(
                        self.last_cpu,
                        self.last_gpu,
                        self.warning_usage_pct,
                    );
                    self.recommended = new_rec;
                    self.reason = new_reason;

                    if tx.send(self.build_signal()).is_err() {
                        break;
                    }
                }
                Ok(false) => {} // First snapshot, no delta yet
                Err(e) => log::warn!("[ComputeMonitor] Read failed: {}", e),
            }

            std::thread::sleep(self.poll_interval);
        }

        log::info!("[ComputeMonitor] Stopped");
        Ok(())
    }

    fn initial_signal(&self) -> Option<SystemSignal> {
        Some(self.build_signal())
    }

    fn name(&self) -> &str {
        "ComputeMonitor"
    }
}

/// Recommend a backend based on CPU/GPU usage percentages.
///
/// Exposed so the simulator can share this logic without duplication.
/// `warning_pct` should match `ComputeMonitorConfig::warning_pct` (default: 70.0).
pub fn compute_recommendation(
    cpu: f64,
    gpu: f64,
    warning_pct: f64,
) -> (RecommendedBackend, ComputeReason) {
    let cpu_hot = cpu >= warning_pct;
    let gpu_hot = gpu >= warning_pct;

    match (cpu_hot, gpu_hot) {
        (true, true) => (RecommendedBackend::Any, ComputeReason::BothLoaded),
        (true, false) => (RecommendedBackend::Gpu, ComputeReason::CpuBottleneck),
        (false, true) => (RecommendedBackend::Cpu, ComputeReason::GpuBottleneck),
        (false, false) => {
            if (cpu - gpu).abs() < 10.0 {
                (RecommendedBackend::Any, ComputeReason::Balanced)
            } else if cpu < gpu {
                (RecommendedBackend::Cpu, ComputeReason::CpuAvailable)
            } else {
                (RecommendedBackend::Gpu, ComputeReason::GpuAvailable)
            }
        }
    }
}

fn read_cpu_snapshot(path: &str) -> anyhow::Result<CpuSnapshot> {
    let content = std::fs::read_to_string(path)?;
    let line = content
        .lines()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Empty /proc/stat"))?;

    let fields: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();

    if fields.len() < 4 {
        anyhow::bail!("Unexpected /proc/stat format: too few fields");
    }

    let total: u64 = fields.iter().sum();
    let idle = fields[3];

    Ok(CpuSnapshot { total, idle })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> ComputeMonitorConfig {
        ComputeMonitorConfig::default()
    }

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
        assert_eq!(snap.total, 9780);
        assert_eq!(snap.idle, 8000);
    }

    #[test]
    fn compute_delta_calculation() {
        let f = tempfile::NamedTempFile::new().unwrap();
        write_stat(f.path(), 1000, 0, 500, 8000);

        let config = default_config();
        let mut monitor =
            ComputeMonitor::with_path(&config, f.path().to_str().unwrap().to_string());

        // First read → no delta
        assert!(!monitor.read_once().unwrap());

        // Second read with higher usage
        write_stat(f.path(), 1500, 0, 700, 8200);
        assert!(monitor.read_once().unwrap());
        assert!(
            monitor.last_cpu > 70.0 && monitor.last_cpu < 85.0,
            "got {}",
            monitor.last_cpu
        );
        assert_eq!(monitor.last_gpu, 0.0);
    }

    #[test]
    fn cpu_bottleneck_recommendation() {
        let (rec, reason) = compute_recommendation(85.0, 30.0, 70.0);
        assert_eq!(rec, RecommendedBackend::Gpu);
        assert_eq!(reason, ComputeReason::CpuBottleneck);
    }

    #[test]
    fn both_loaded_recommendation() {
        let (rec, reason) = compute_recommendation(92.0, 92.0, 70.0);
        assert_eq!(rec, RecommendedBackend::Any);
        assert_eq!(reason, ComputeReason::BothLoaded);
    }

    #[test]
    fn balanced_recommendation() {
        let (rec, reason) = compute_recommendation(30.0, 32.0, 70.0);
        assert_eq!(rec, RecommendedBackend::Any);
        assert_eq!(reason, ComputeReason::Balanced);
    }

    #[test]
    fn no_emergency_level() {
        let config = default_config(); // warning=70, critical=90
        let mut monitor = ComputeMonitor::with_path(&config, String::new());

        monitor.last_cpu = 99.0;
        let level = monitor.evaluator.evaluate(99.0);
        assert_eq!(level, Some(Level::Critical)); // Not Emergency
    }

    #[test]
    fn recommendation_change_without_level_change() {
        let config = default_config();
        let mut monitor = ComputeMonitor::with_path(&config, String::new());

        // CPU bottleneck at Warning
        monitor.last_cpu = 85.0;
        monitor.last_gpu = 30.0;
        monitor.evaluator.evaluate(85.0); // → Warning
        let (rec, _) = compute_recommendation(85.0, 30.0, config.warning_pct);
        monitor.recommended = rec;
        assert_eq!(monitor.recommended, RecommendedBackend::Gpu);

        // GPU also hot → recommendation changes (level stays Warning)
        let (new_rec, _) = compute_recommendation(85.0, 80.0, config.warning_pct);
        assert_ne!(new_rec, monitor.recommended); // Changed!
        assert_eq!(new_rec, RecommendedBackend::Any);
    }
}
