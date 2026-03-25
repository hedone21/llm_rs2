use super::{ResilienceAction, ResilienceStrategy};
use crate::resilience::signal::{Level, SystemSignal};
use crate::resilience::state::OperatingMode;

/// Memory pressure response strategy.
/// Triggers KV cache eviction proportional to severity.
pub struct MemoryStrategy {
    last_level: Level,
}

impl Default for MemoryStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStrategy {
    pub fn new() -> Self {
        Self {
            last_level: Level::Normal,
        }
    }
}

impl ResilienceStrategy for MemoryStrategy {
    fn react(&mut self, signal: &SystemSignal, _mode: OperatingMode) -> Vec<ResilienceAction> {
        let SystemSignal::MemoryPressure { level, .. } = signal else {
            return vec![];
        };

        // Skip if level unchanged and Normal
        if *level == self.last_level && *level == Level::Normal {
            return vec![];
        }
        self.last_level = *level;

        match level {
            Level::Normal => vec![ResilienceAction::RestoreDefaults],
            Level::Warning => vec![ResilienceAction::Evict { target_ratio: 0.85 }],
            Level::Critical => vec![ResilienceAction::Evict { target_ratio: 0.50 }],
            Level::Emergency => vec![
                ResilienceAction::Evict { target_ratio: 0.25 },
                ResilienceAction::RejectNew,
            ],
        }
    }

    fn name(&self) -> &str {
        "memory"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem_signal(level: Level) -> SystemSignal {
        SystemSignal::MemoryPressure {
            level,
            available_bytes: 1024 * 1024,
            total_bytes: 4 * 1024 * 1024,
            reclaim_target_bytes: 512 * 1024,
        }
    }

    #[test]
    fn test_memory_normal_restores_defaults() {
        let mut strategy = MemoryStrategy::new();
        // First transition away from Normal to make the return meaningful
        let _ = strategy.react(&mem_signal(Level::Warning), OperatingMode::Degraded);
        let actions = strategy.react(&mem_signal(Level::Normal), OperatingMode::Normal);
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], ResilienceAction::RestoreDefaults));
    }

    #[test]
    fn test_memory_critical_triggers_eviction() {
        let mut strategy = MemoryStrategy::new();
        let actions = strategy.react(&mem_signal(Level::Critical), OperatingMode::Minimal);
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            ResilienceAction::Evict { target_ratio } => {
                assert!(*target_ratio <= 0.50);
            }
            _ => panic!("Expected Evict action"),
        }
    }

    #[test]
    fn test_memory_emergency_evicts_and_rejects() {
        let mut strategy = MemoryStrategy::new();
        let actions = strategy.react(&mem_signal(Level::Emergency), OperatingMode::Suspended);
        assert_eq!(actions.len(), 2);
        assert!(matches!(actions[0], ResilienceAction::Evict { .. }));
        assert!(matches!(actions[1], ResilienceAction::RejectNew));
    }
}
