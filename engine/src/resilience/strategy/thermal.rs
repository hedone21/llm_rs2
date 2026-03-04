use super::{ResilienceAction, ResilienceStrategy};
use crate::resilience::signal::{Level, RecommendedBackend, SystemSignal};
use crate::resilience::state::OperatingMode;

/// Thermal alert response strategy.
/// Reduces compute intensity proportional to thermal severity.
pub struct ThermalStrategy;

impl Default for ThermalStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ThermalStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl ResilienceStrategy for ThermalStrategy {
    fn react(&mut self, signal: &SystemSignal, _mode: OperatingMode) -> Vec<ResilienceAction> {
        let SystemSignal::ThermalAlert {
            level,
            throttle_ratio,
            ..
        } = signal
        else {
            return vec![];
        };

        match level {
            Level::Normal => vec![ResilienceAction::RestoreDefaults],
            Level::Warning => vec![ResilienceAction::SwitchBackend {
                to: RecommendedBackend::Cpu,
            }],
            Level::Critical => {
                let delay = ((1.0 - throttle_ratio) * 100.0) as u64;
                vec![
                    ResilienceAction::SwitchBackend {
                        to: RecommendedBackend::Cpu,
                    },
                    ResilienceAction::Throttle { delay_ms: delay },
                    ResilienceAction::LimitTokens { max_tokens: 64 },
                ]
            }
            Level::Emergency => vec![ResilienceAction::Suspend],
        }
    }

    fn name(&self) -> &str {
        "thermal"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thermal_signal(level: Level, throttle_ratio: f64) -> SystemSignal {
        SystemSignal::ThermalAlert {
            level,
            temperature_mc: 75000,
            throttling_active: throttle_ratio < 1.0,
            throttle_ratio,
        }
    }

    #[test]
    fn test_thermal_emergency_suspends() {
        let mut strategy = ThermalStrategy::new();
        let actions = strategy.react(
            &thermal_signal(Level::Emergency, 0.3),
            OperatingMode::Suspended,
        );
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], ResilienceAction::Suspend));
    }

    #[test]
    fn test_thermal_critical_throttles_proportionally() {
        let mut strategy = ThermalStrategy::new();
        // throttle_ratio 0.5 → delay = (1.0 - 0.5) * 100 = 50ms
        let actions = strategy.react(
            &thermal_signal(Level::Critical, 0.5),
            OperatingMode::Minimal,
        );
        assert_eq!(actions.len(), 3);
        match &actions[1] {
            ResilienceAction::Throttle { delay_ms } => assert_eq!(*delay_ms, 50),
            _ => panic!("Expected Throttle"),
        }
    }
}
