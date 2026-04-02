use std::sync::mpsc;

use super::signal::{Level, SystemSignal};
use super::state::OperatingMode;
use super::strategy::{
    ComputeStrategy, EnergyStrategy, MemoryStrategy, ResilienceAction, ResilienceStrategy,
    ThermalStrategy, resolve_conflicts,
};

/// Cache of latest levels for each signal type.
struct SignalLevels {
    memory: Level,
    compute: Level,
    thermal: Level,
    energy: Level,
}

impl SignalLevels {
    fn new() -> Self {
        Self {
            memory: Level::Normal,
            compute: Level::Normal,
            thermal: Level::Normal,
            energy: Level::Normal,
        }
    }
}

/// Strategy implementations per signal type.
struct Strategies {
    memory: Box<dyn ResilienceStrategy>,
    compute: Box<dyn ResilienceStrategy>,
    thermal: Box<dyn ResilienceStrategy>,
    energy: Box<dyn ResilienceStrategy>,
}

impl Strategies {
    fn new() -> Self {
        Self {
            memory: Box::new(MemoryStrategy::new()),
            compute: Box::new(ComputeStrategy::new()),
            thermal: Box::new(ThermalStrategy::new()),
            energy: Box::new(EnergyStrategy::new()),
        }
    }
}

/// Central resilience orchestrator.
/// Receives signals via mpsc channel, delegates to strategies,
/// resolves conflicts, and returns actions for the inference loop.
pub struct ResilienceManager {
    rx: mpsc::Receiver<SystemSignal>,
    mode: OperatingMode,
    current_levels: SignalLevels,
    strategies: Strategies,
}

impl ResilienceManager {
    /// Create a new ResilienceManager with default strategies.
    pub fn new(rx: mpsc::Receiver<SystemSignal>) -> Self {
        Self {
            rx,
            mode: OperatingMode::Normal,
            current_levels: SignalLevels::new(),
            strategies: Strategies::new(),
        }
    }

    /// Non-blocking poll: drain all pending signals from channel,
    /// process them through strategies, resolve conflicts,
    /// and return the final list of actions.
    /// Called once per token in the inference loop.
    pub fn poll(&mut self) -> Vec<ResilienceAction> {
        let mut all_actions = Vec::new();

        while let Ok(signal) = self.rx.try_recv() {
            self.process_signal(&signal, &mut all_actions);
        }

        if all_actions.is_empty() {
            return vec![];
        }

        resolve_conflicts(all_actions)
    }

    /// Current operating mode.
    pub fn mode(&self) -> OperatingMode {
        self.mode
    }

    fn process_signal(&mut self, signal: &SystemSignal, actions: &mut Vec<ResilienceAction>) {
        // 1. Update level cache
        match signal {
            SystemSignal::MemoryPressure { level, .. } => {
                self.current_levels.memory = *level;
            }
            SystemSignal::ComputeGuidance { level, .. } => {
                self.current_levels.compute = *level;
            }
            SystemSignal::ThermalAlert { level, .. } => {
                self.current_levels.thermal = *level;
            }
            SystemSignal::EnergyConstraint { level, .. } => {
                self.current_levels.energy = *level;
            }
        }

        // 2. Recalculate operating mode
        self.mode = OperatingMode::from_levels(
            self.current_levels.memory,
            self.current_levels.compute,
            self.current_levels.thermal,
            self.current_levels.energy,
        );

        // 3. Delegate to corresponding strategy
        let strategy_actions = match signal {
            SystemSignal::MemoryPressure { .. } => self.strategies.memory.react(signal, self.mode),
            SystemSignal::ComputeGuidance { .. } => {
                self.strategies.compute.react(signal, self.mode)
            }
            SystemSignal::ThermalAlert { .. } => self.strategies.thermal.react(signal, self.mode),
            SystemSignal::EnergyConstraint { .. } => {
                self.strategies.energy.react(signal, self.mode)
            }
        };

        actions.extend(strategy_actions);
    }
}

/// Mutable inference loop state passed to action executor.
pub struct InferenceContext<'a> {
    pub max_tokens: &'a mut usize,
    pub throttle_delay_ms: &'a mut u64,
    pub suspended: &'a mut bool,
    pub reject_new: &'a mut bool,
}

/// Execute a single resilience action against inference loop state.
pub fn execute_action(action: &ResilienceAction, ctx: &mut InferenceContext) {
    match action {
        ResilienceAction::Evict { target_ratio } => {
            // Phase 3a: integrate with CacheManager
            log::info!(
                "[Resilience] Evict requested: target_ratio={}",
                target_ratio
            );
        }
        ResilienceAction::SwitchBackend { to } => {
            // Handled by generate binary via ExecutionPlan.switch_device
            log::info!("[Resilience] Backend switch requested: {:?}", to);
        }
        ResilienceAction::LimitTokens { max_tokens } => {
            *ctx.max_tokens = (*ctx.max_tokens).min(*max_tokens);
        }
        ResilienceAction::Throttle { delay_ms } => {
            *ctx.throttle_delay_ms = *delay_ms;
        }
        ResilienceAction::Suspend => {
            *ctx.suspended = true;
        }
        ResilienceAction::RejectNew => {
            *ctx.reject_new = true;
        }
        ResilienceAction::RestoreDefaults => {
            *ctx.throttle_delay_ms = 0;
            *ctx.reject_new = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resilience::signal::EnergyReason;

    fn send_signal(tx: &mpsc::Sender<SystemSignal>, signal: SystemSignal) {
        tx.send(signal).unwrap();
    }

    #[test]
    fn test_manager_poll_returns_empty_when_no_signals() {
        let (_, rx) = mpsc::channel();
        let mut mgr = ResilienceManager::new(rx);
        let actions = mgr.poll();
        assert!(actions.is_empty());
        assert_eq!(mgr.mode(), OperatingMode::Normal);
    }

    #[test]
    fn test_manager_processes_memory_signal() {
        let (tx, rx) = mpsc::channel();
        let mut mgr = ResilienceManager::new(rx);

        send_signal(
            &tx,
            SystemSignal::MemoryPressure {
                level: Level::Critical,
                available_bytes: 50 * 1024 * 1024,
                total_bytes: 4 * 1024 * 1024 * 1024,
                reclaim_target_bytes: 100 * 1024 * 1024,
            },
        );

        let actions = mgr.poll();
        assert!(!actions.is_empty());
        assert_eq!(mgr.mode(), OperatingMode::Minimal);
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, ResilienceAction::Evict { .. }))
        );
    }

    #[test]
    fn test_manager_handles_multiple_signals() {
        let (tx, rx) = mpsc::channel();
        let mut mgr = ResilienceManager::new(rx);

        // Send memory warning + thermal critical simultaneously
        send_signal(
            &tx,
            SystemSignal::MemoryPressure {
                level: Level::Warning,
                available_bytes: 200 * 1024 * 1024,
                total_bytes: 4 * 1024 * 1024 * 1024,
                reclaim_target_bytes: 50 * 1024 * 1024,
            },
        );
        send_signal(
            &tx,
            SystemSignal::ThermalAlert {
                level: Level::Critical,
                temperature_mc: 80000,
                throttling_active: true,
                throttle_ratio: 0.5,
            },
        );

        let actions = mgr.poll();
        // Should have conflict-resolved actions from both strategies
        assert!(!actions.is_empty());
        // Mode should be Minimal (Critical > Warning)
        assert_eq!(mgr.mode(), OperatingMode::Minimal);
    }

    #[test]
    fn test_manager_state_transitions() {
        let (tx, rx) = mpsc::channel();
        let mut mgr = ResilienceManager::new(rx);
        assert_eq!(mgr.mode(), OperatingMode::Normal);

        // Normal → Degraded
        send_signal(
            &tx,
            SystemSignal::MemoryPressure {
                level: Level::Warning,
                available_bytes: 200 * 1024 * 1024,
                total_bytes: 4 * 1024 * 1024 * 1024,
                reclaim_target_bytes: 50 * 1024 * 1024,
            },
        );
        mgr.poll();
        assert_eq!(mgr.mode(), OperatingMode::Degraded);

        // Degraded → Minimal
        send_signal(
            &tx,
            SystemSignal::ThermalAlert {
                level: Level::Critical,
                temperature_mc: 80000,
                throttling_active: true,
                throttle_ratio: 0.6,
            },
        );
        mgr.poll();
        assert_eq!(mgr.mode(), OperatingMode::Minimal);

        // Minimal → Suspended
        send_signal(
            &tx,
            SystemSignal::EnergyConstraint {
                level: Level::Emergency,
                reason: EnergyReason::BatteryCritical,
                power_budget_mw: 0,
            },
        );
        mgr.poll();
        assert_eq!(mgr.mode(), OperatingMode::Suspended);

        // Suspended → Normal (all signals recover)
        send_signal(
            &tx,
            SystemSignal::MemoryPressure {
                level: Level::Normal,
                available_bytes: 1024 * 1024 * 1024,
                total_bytes: 4 * 1024 * 1024 * 1024,
                reclaim_target_bytes: 0,
            },
        );
        send_signal(
            &tx,
            SystemSignal::ThermalAlert {
                level: Level::Normal,
                temperature_mc: 40000,
                throttling_active: false,
                throttle_ratio: 1.0,
            },
        );
        send_signal(
            &tx,
            SystemSignal::EnergyConstraint {
                level: Level::Normal,
                reason: EnergyReason::Charging,
                power_budget_mw: 0,
            },
        );
        mgr.poll();
        assert_eq!(mgr.mode(), OperatingMode::Normal);
    }

    #[test]
    fn test_manager_survives_channel_disconnect() {
        let (tx, rx) = mpsc::channel();
        let mut mgr = ResilienceManager::new(rx);

        // Send one signal, then drop sender
        send_signal(
            &tx,
            SystemSignal::MemoryPressure {
                level: Level::Warning,
                available_bytes: 200 * 1024 * 1024,
                total_bytes: 4 * 1024 * 1024 * 1024,
                reclaim_target_bytes: 50 * 1024 * 1024,
            },
        );
        drop(tx);

        // First poll processes the buffered signal
        let actions = mgr.poll();
        assert!(!actions.is_empty());
        assert_eq!(mgr.mode(), OperatingMode::Degraded);

        // Second poll: channel disconnected, no panic, state preserved
        let actions = mgr.poll();
        assert!(actions.is_empty());
        assert_eq!(mgr.mode(), OperatingMode::Degraded);
    }

    #[test]
    fn test_execute_suspend_sets_flag() {
        let mut max_tokens = 128;
        let mut throttle_delay_ms = 0u64;
        let mut suspended = false;
        let mut reject_new = false;

        let mut ctx = InferenceContext {
            max_tokens: &mut max_tokens,
            throttle_delay_ms: &mut throttle_delay_ms,
            suspended: &mut suspended,
            reject_new: &mut reject_new,
        };

        execute_action(&ResilienceAction::Suspend, &mut ctx);
        assert!(suspended);
    }

    #[test]
    fn test_execute_limit_tokens() {
        let mut max_tokens = 128;
        let mut throttle_delay_ms = 0u64;
        let mut suspended = false;
        let mut reject_new = false;

        {
            let mut ctx = InferenceContext {
                max_tokens: &mut max_tokens,
                throttle_delay_ms: &mut throttle_delay_ms,
                suspended: &mut suspended,
                reject_new: &mut reject_new,
            };
            execute_action(&ResilienceAction::LimitTokens { max_tokens: 64 }, &mut ctx);
        }
        assert_eq!(max_tokens, 64);

        // Should take minimum
        {
            let mut ctx = InferenceContext {
                max_tokens: &mut max_tokens,
                throttle_delay_ms: &mut throttle_delay_ms,
                suspended: &mut suspended,
                reject_new: &mut reject_new,
            };
            execute_action(&ResilienceAction::LimitTokens { max_tokens: 200 }, &mut ctx);
        }
        assert_eq!(max_tokens, 64);
    }

    #[test]
    fn test_execute_restore_clears_constraints() {
        let mut max_tokens = 64;
        let mut throttle_delay_ms = 100u64;
        let mut suspended = false;
        let mut reject_new = true;

        let mut ctx = InferenceContext {
            max_tokens: &mut max_tokens,
            throttle_delay_ms: &mut throttle_delay_ms,
            suspended: &mut suspended,
            reject_new: &mut reject_new,
        };

        execute_action(&ResilienceAction::RestoreDefaults, &mut ctx);
        assert_eq!(throttle_delay_ms, 0);
        assert!(!reject_new);
    }

    #[test]
    fn test_execute_throttle_sets_delay() {
        let mut max_tokens = 128;
        let mut throttle_delay_ms = 0u64;
        let mut suspended = false;
        let mut reject_new = false;

        let mut ctx = InferenceContext {
            max_tokens: &mut max_tokens,
            throttle_delay_ms: &mut throttle_delay_ms,
            suspended: &mut suspended,
            reject_new: &mut reject_new,
        };

        execute_action(&ResilienceAction::Throttle { delay_ms: 50 }, &mut ctx);
        assert_eq!(throttle_delay_ms, 50);
    }
}
