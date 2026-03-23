//! Unified evaluation framework for log-likelihood benchmarks.
//!
//! Provides a generic eval loop (`run_eval_ll_generic`) that works with any
//! `KVCacheOps` implementation via the `StepHook` trait. This eliminates
//! code duplication between `run_eval_ll` (eviction) and `run_kivi_eval_ll`
//! (KIVI quantization).
//!
//! # Module Structure
//!
//! - `hook`: `StepHook` and `CacheSnapshot` traits
//! - `output`: `EvalOutput`, `EvalConfig`, `EvalQuestion`
//! - `qcf_helpers`: shared QCF/OPR metric aggregation utilities
//!
//! # Design
//!
//! See `docs/38_eval_refactoring.md` for the full design document.

pub mod hook;
pub mod output;

pub use hook::{CacheSnapshot, MetricsSummary, PostStepResult, StepHook};
pub use output::{EvalConfig, EvalOutput, EvalQuestion};
