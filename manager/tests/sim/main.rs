//! sim 테스트 바이너리 진입점.
//!
//! `cargo test -p llm_manager --test sim`으로 실행.

#[path = "../common.rs"]
mod common;

mod test_config;
mod test_harness;
mod test_physics;
mod test_scenarios;
mod test_signal;
