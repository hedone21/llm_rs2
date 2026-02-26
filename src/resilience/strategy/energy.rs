use super::{ResilienceAction, ResilienceStrategy};
use crate::resilience::signal::{Level, RecommendedBackend, SystemSignal};
use crate::resilience::state::OperatingMode;

/// Energy constraint response strategy.
/// Reduces power consumption proportional to severity.
pub struct EnergyStrategy;

impl Default for EnergyStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl EnergyStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl ResilienceStrategy for EnergyStrategy {
    fn react(&mut self, signal: &SystemSignal, _mode: OperatingMode) -> Vec<ResilienceAction> {
        let SystemSignal::EnergyConstraint { level, .. } = signal else {
            return vec![];
        };

        match level {
            Level::Normal => vec![ResilienceAction::RestoreDefaults],
            Level::Warning => vec![ResilienceAction::SwitchBackend {
                to: RecommendedBackend::Cpu,
            }],
            Level::Critical => vec![
                ResilienceAction::SwitchBackend {
                    to: RecommendedBackend::Cpu,
                },
                ResilienceAction::LimitTokens { max_tokens: 64 },
                ResilienceAction::Throttle { delay_ms: 30 },
            ],
            Level::Emergency => vec![ResilienceAction::Suspend, ResilienceAction::RejectNew],
        }
    }

    fn name(&self) -> &str {
        "energy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resilience::signal::EnergyReason;

    fn energy_signal(level: Level, reason: EnergyReason) -> SystemSignal {
        SystemSignal::EnergyConstraint {
            level,
            reason,
            power_budget_mw: 5000,
        }
    }

    #[test]
    fn test_energy_normal_restores() {
        let mut strategy = EnergyStrategy::new();
        let actions = strategy.react(
            &energy_signal(Level::Normal, EnergyReason::Charging),
            OperatingMode::Normal,
        );
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], ResilienceAction::RestoreDefaults));
    }

    #[test]
    fn test_energy_emergency_suspends_and_rejects() {
        let mut strategy = EnergyStrategy::new();
        let actions = strategy.react(
            &energy_signal(Level::Emergency, EnergyReason::BatteryCritical),
            OperatingMode::Suspended,
        );
        assert_eq!(actions.len(), 2);
        assert!(matches!(actions[0], ResilienceAction::Suspend));
        assert!(matches!(actions[1], ResilienceAction::RejectNew));
    }
}
