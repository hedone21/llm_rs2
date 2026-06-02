//! KIVI fused attention capability (§3.3).
//!
//! Phase α-W-4 에서 `backend.rs` 의 `KiviAttentionBackend` trait 정의를 이리로 이동하고
//! per-call `as_kivi_attention()` lookup 을 `CapabilityRegistry` pull 로 대체한다. 현재는
//! `backend.rs` 거주분을 re-export 하는 shim 이라 모든 call site 가 그대로 동작한다(byte-identical).
pub use crate::backend::KiviAttentionBackend;
