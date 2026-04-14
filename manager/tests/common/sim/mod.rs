//! Simulation harness 공통 모듈.
//!
//! Phase 1: config + expr 로더.
//! Phase 2: state, physics, compose, derived 추가.
//! Phase 3: clock, signal, noise 추가.
//! Phase 4+: harness, trajectory.

pub mod clock;
pub mod compose;
pub mod config;
pub mod derived;
pub mod expr;
pub mod noise;
pub mod physics;
pub mod signal;
pub mod state;

// Phase 3 주요 타입 re-export (harness에서 사용 예정)
#[allow(unused_imports)]
pub use clock::{EventKind, ScheduledEvent, VirtualClock};
#[allow(unused_imports)]
pub use noise::{NoiseRng, maybe_create as maybe_create_noise};
#[allow(unused_imports)]
pub use signal::{derive_heartbeat, derive_signal};
