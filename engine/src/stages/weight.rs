//! weight-mutate Stage 거주지 (§2.1).
//!
//! 입주자:
//! - `PartitionStage` (AB-4, `partition.rs`) — `SetPartitionRatio` runtime directive 의 OneShot
//!   re-slice. concrete-handle `Vec<Arc<LayerSlot>>` + `Arc<Hardware>`(§5.5).
//!
//! 예정 입주자(AB-6): `WeightSwapStage` — concrete-handle `Arc<LayerSlot>`(§4.2). 현
//! `weight/weight_swap_handler.rs` 의 trigger 부분. swap 오케스트레이션은 `weight/`로
//! 분리되어 있다(γ-1 이동 완료). PartitionStage 와 형제이나 join 표면이 0(weight slot dispatch
//! mode 변경 vs precision swap)이라 별도 파일.

pub mod partition;
