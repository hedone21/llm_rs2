//! Phase 4-A: prompt-batch (`--prompt-batch <path>`) 모드 추출.
//!
//! `bin/generate.rs::main()`의 batch 분기(l.2235~3094)를 외과적으로 이동.
//! Light-lift 패턴: BatchRunCtx struct로 외부 state를 packaging 후
//! `run_prompt_batch(ctx)` 단일 호출로 main()에서 dispatch.

pub mod args;
pub mod helpers;
pub mod runner;

pub use args::BatchRunCtx;
pub use runner::run_prompt_batch;
