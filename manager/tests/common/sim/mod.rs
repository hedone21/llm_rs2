//! Simulation harness 공통 모듈.
//!
//! Phase 1: config + expr 로더.
//! Phase 2: state, physics, compose, derived 추가.
//! Phase 3+: clock, signal, noise, harness, trajectory.

pub mod compose;
pub mod config;
pub mod derived;
pub mod expr;
pub mod physics;
pub mod state;
