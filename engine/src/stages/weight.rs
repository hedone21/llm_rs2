//! weight-mutate Stage 거주지 (§2.1).
//!
//! 예정 입주자(Phase α-K): `WeightSwapStage` — concrete-handle `Arc<LayerSlot>`(§4.2). 현
//! `weight/weight_swap_handler.rs` 의 trigger 부분. swap 오케스트레이션은 `weight/`로
//! 분리되어 있다(γ-1 이동 완료).
//!
//! 현재는 골격(입주자 0) — α-K 진입 시 채운다.
