// Re-export signal types from shared crate.
// This keeps all internal `use crate::resilience::signal::*` paths working.
pub use llm_shared::{ComputeReason, EnergyReason, Level, RecommendedBackend, SystemSignal};
