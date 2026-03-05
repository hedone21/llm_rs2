pub mod threshold;

use crate::collector::Reading;
use llm_shared::SystemSignal;

/// PolicyEngine evaluates system readings and decides which signals to emit.
///
/// This is the core extensibility point of the Manager service.
/// New evaluation strategies are added by implementing this trait,
/// following the Open-Closed Principle (OCP).
///
/// # Implementations
///
/// - [`ThresholdPolicy`](threshold::ThresholdPolicy) — Hysteresis-based
///   threshold evaluation (default). Uses configurable up/down thresholds
///   to prevent oscillation near level boundaries.
///
/// # Future extensions
///
/// - Trend-based policy (moving average, rate of change)
/// - ML-based prediction
/// - Composite policy (delegates to per-resource sub-policies)
pub trait PolicyEngine: Send {
    /// Process a new reading and return any signals that should be emitted.
    ///
    /// Returns an empty vec if no level change occurred.
    fn process(&mut self, reading: &Reading) -> Vec<SystemSignal>;

    /// Return the current state as signals for all tracked resources.
    ///
    /// Used to emit initial state when a new LLM client connects.
    fn current_signals(&self) -> Vec<SystemSignal>;

    /// Human-readable name for logging.
    fn name(&self) -> &str;
}
