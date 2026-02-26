use super::signal::{RecommendedBackend, SystemSignal};
use super::state::OperatingMode;

pub mod memory;
pub mod compute;
pub mod thermal;
pub mod energy;

pub use memory::MemoryStrategy;
pub use compute::ComputeStrategy;
pub use thermal::ThermalStrategy;
pub use energy::EnergyStrategy;

/// Action to be executed by the inference loop in response to a signal.
#[derive(Debug, Clone)]
pub enum ResilienceAction {
    /// KV cache eviction. target_ratio: target ratio relative to current cache (0.0~1.0).
    Evict { target_ratio: f32 },
    /// Backend switch.
    SwitchBackend { to: RecommendedBackend },
    /// Limit generated tokens.
    LimitTokens { max_tokens: usize },
    /// Insert delay between token generation (ms).
    Throttle { delay_ms: u64 },
    /// Pause inference.
    Suspend,
    /// Reject new inference requests.
    RejectNew,
    /// Release previous constraints (return to normal).
    RestoreDefaults,
}

/// Signal reaction strategy interface.
/// Follows the same pattern as `EvictionPolicy` trait.
pub trait ResilienceStrategy: Send + Sync {
    /// Receive signal and return list of actions to execute.
    /// Returns empty Vec if no action needed.
    fn react(
        &mut self,
        signal: &SystemSignal,
        mode: OperatingMode,
    ) -> Vec<ResilienceAction>;

    /// Strategy name (for logging).
    fn name(&self) -> &str;
}

/// Merge actions from multiple strategies, resolving conflicts.
///
/// Rules:
/// - Suspend overrides everything
/// - CPU backend preferred over GPU (safety first)
/// - Most aggressive eviction ratio (min) wins
/// - Largest delay wins
/// - RestoreDefaults only when no other constraints exist
pub fn resolve_conflicts(actions: Vec<ResilienceAction>) -> Vec<ResilienceAction> {
    if actions.is_empty() {
        return vec![];
    }

    let mut min_evict_ratio = f32::MAX;
    let mut max_delay = 0u64;
    let mut min_tokens = usize::MAX;
    let mut target_backend: Option<RecommendedBackend> = None;
    let mut has_suspend = false;
    let mut has_reject = false;
    let mut has_restore = false;

    for action in &actions {
        match action {
            ResilienceAction::Evict { target_ratio } => {
                min_evict_ratio = min_evict_ratio.min(*target_ratio);
            }
            ResilienceAction::SwitchBackend { to } => {
                target_backend = Some(match (target_backend, to) {
                    (Some(RecommendedBackend::Cpu), _) => RecommendedBackend::Cpu,
                    (_, RecommendedBackend::Cpu) => RecommendedBackend::Cpu,
                    (_, other) => *other,
                });
            }
            ResilienceAction::LimitTokens { max_tokens } => {
                min_tokens = min_tokens.min(*max_tokens);
            }
            ResilienceAction::Throttle { delay_ms } => {
                max_delay = max_delay.max(*delay_ms);
            }
            ResilienceAction::Suspend => has_suspend = true,
            ResilienceAction::RejectNew => has_reject = true,
            ResilienceAction::RestoreDefaults => has_restore = true,
        }
    }

    // Suspend overrides everything
    if has_suspend {
        return vec![ResilienceAction::Suspend];
    }

    // RestoreDefaults only when no other constraints
    if has_restore
        && min_evict_ratio >= f32::MAX
        && max_delay == 0
        && min_tokens == usize::MAX
        && target_backend.is_none()
        && !has_reject
    {
        return vec![ResilienceAction::RestoreDefaults];
    }

    let mut resolved = Vec::new();

    if min_evict_ratio < f32::MAX {
        resolved.push(ResilienceAction::Evict {
            target_ratio: min_evict_ratio,
        });
    }
    if let Some(backend) = target_backend {
        resolved.push(ResilienceAction::SwitchBackend { to: backend });
    }
    if min_tokens < usize::MAX {
        resolved.push(ResilienceAction::LimitTokens {
            max_tokens: min_tokens,
        });
    }
    if max_delay > 0 {
        resolved.push(ResilienceAction::Throttle { delay_ms: max_delay });
    }
    if has_reject {
        resolved.push(ResilienceAction::RejectNew);
    }

    resolved
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input_returns_empty() {
        let result = resolve_conflicts(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_cpu_always_wins_over_gpu() {
        let actions = vec![
            ResilienceAction::SwitchBackend {
                to: RecommendedBackend::Gpu,
            },
            ResilienceAction::SwitchBackend {
                to: RecommendedBackend::Cpu,
            },
        ];
        let result = resolve_conflicts(actions);
        assert_eq!(result.len(), 1);
        match &result[0] {
            ResilienceAction::SwitchBackend { to } => {
                assert_eq!(*to, RecommendedBackend::Cpu);
            }
            _ => panic!("Expected SwitchBackend"),
        }
    }

    #[test]
    fn test_most_aggressive_eviction_wins() {
        let actions = vec![
            ResilienceAction::Evict { target_ratio: 0.85 },
            ResilienceAction::Evict { target_ratio: 0.50 },
            ResilienceAction::Evict { target_ratio: 0.75 },
        ];
        let result = resolve_conflicts(actions);
        assert_eq!(result.len(), 1);
        match &result[0] {
            ResilienceAction::Evict { target_ratio } => {
                assert!((target_ratio - 0.50).abs() < f32::EPSILON);
            }
            _ => panic!("Expected Evict"),
        }
    }

    #[test]
    fn test_largest_delay_wins() {
        let actions = vec![
            ResilienceAction::Throttle { delay_ms: 30 },
            ResilienceAction::Throttle { delay_ms: 100 },
            ResilienceAction::Throttle { delay_ms: 50 },
        ];
        let result = resolve_conflicts(actions);
        assert_eq!(result.len(), 1);
        match &result[0] {
            ResilienceAction::Throttle { delay_ms } => assert_eq!(*delay_ms, 100),
            _ => panic!("Expected Throttle"),
        }
    }

    #[test]
    fn test_suspend_overrides_all() {
        let actions = vec![
            ResilienceAction::Evict { target_ratio: 0.50 },
            ResilienceAction::SwitchBackend {
                to: RecommendedBackend::Cpu,
            },
            ResilienceAction::Suspend,
            ResilienceAction::Throttle { delay_ms: 100 },
        ];
        let result = resolve_conflicts(actions);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0], ResilienceAction::Suspend));
    }

    #[test]
    fn test_restore_only_when_no_other_constraints() {
        // RestoreDefaults + Evict → Evict only (RestoreDefaults suppressed)
        let actions = vec![
            ResilienceAction::RestoreDefaults,
            ResilienceAction::Evict { target_ratio: 0.85 },
        ];
        let result = resolve_conflicts(actions);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0], ResilienceAction::Evict { .. }));
    }

    #[test]
    fn test_restore_alone_passes_through() {
        let actions = vec![ResilienceAction::RestoreDefaults];
        let result = resolve_conflicts(actions);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0], ResilienceAction::RestoreDefaults));
    }
}
