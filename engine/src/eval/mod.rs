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

pub mod eval_loop;
pub mod eviction_hook;
pub mod hook;
pub mod kivi_hook;
pub mod output;
pub mod qcf_helpers;

pub use eval_loop::run_eval_ll_generic;
pub use eviction_hook::EvictionHook;
pub use hook::{CacheSnapshot, MetricsSummary, PostStepResult, StepHook};
pub use kivi_hook::KiviHook;
pub use output::{EvalConfig, EvalOutput, EvalQuestion};
pub use qcf_helpers::{QcfSwapDumpContext, dump_qcf_swap_json};
