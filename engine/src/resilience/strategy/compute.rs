use super::{ResilienceAction, ResilienceStrategy};
use crate::resilience::signal::{Level, RecommendedBackend, SystemSignal};
use crate::resilience::state::OperatingMode;

/// Compute guidance response strategy.
/// Handles backend switching recommendations from Manager.
pub struct ComputeStrategy {
    current_backend: RecommendedBackend,
}

impl Default for ComputeStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeStrategy {
    pub fn new() -> Self {
        Self {
            current_backend: RecommendedBackend::Any,
        }
    }
}

impl ResilienceStrategy for ComputeStrategy {
    fn react(&mut self, signal: &SystemSignal, _mode: OperatingMode) -> Vec<ResilienceAction> {
        let SystemSignal::ComputeGuidance {
            level,
            recommended_backend,
            ..
        } = signal
        else {
            return vec![];
        };

        match level {
            Level::Normal => {
                self.current_backend = RecommendedBackend::Any;
                vec![ResilienceAction::RestoreDefaults]
            }
            Level::Warning => {
                // Prepare only. Don't switch yet at Warning level.
                self.current_backend = *recommended_backend;
                vec![]
            }
            Level::Critical => {
                if *recommended_backend != self.current_backend {
                    self.current_backend = *recommended_backend;
                    vec![ResilienceAction::SwitchBackend {
                        to: *recommended_backend,
                    }]
                } else {
                    // Already on recommended backend. Add throttle.
                    vec![ResilienceAction::Throttle { delay_ms: 50 }]
                }
            }
            Level::Emergency => {
                vec![
                    ResilienceAction::SwitchBackend {
                        to: RecommendedBackend::Cpu,
                    },
                    ResilienceAction::Throttle { delay_ms: 100 },
                ]
            }
        }
    }

    fn name(&self) -> &str {
        "compute"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resilience::signal::ComputeReason;

    fn compute_signal(level: Level, backend: RecommendedBackend) -> SystemSignal {
        SystemSignal::ComputeGuidance {
            level,
            recommended_backend: backend,
            reason: ComputeReason::Balanced,
            cpu_usage_pct: 50.0,
            gpu_usage_pct: 50.0,
        }
    }

    #[test]
    fn test_compute_critical_switches_backend() {
        let mut strategy = ComputeStrategy::new();
        let actions = strategy.react(
            &compute_signal(Level::Critical, RecommendedBackend::Cpu),
            OperatingMode::Minimal,
        );
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            ResilienceAction::SwitchBackend { to } => {
                assert_eq!(*to, RecommendedBackend::Cpu);
            }
            _ => panic!("Expected SwitchBackend"),
        }
    }

    #[test]
    fn test_compute_warning_does_not_switch() {
        let mut strategy = ComputeStrategy::new();
        let actions = strategy.react(
            &compute_signal(Level::Warning, RecommendedBackend::Cpu),
            OperatingMode::Degraded,
        );
        assert!(actions.is_empty());
    }
}
