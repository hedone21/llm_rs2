//! Single-layer 14-node QNN graph wrapper (M3.1 stub).
//!
//! Spec: `spec/30-engine.md` 부록 C.4 (ENG-QNN-221~230, INV-176~177),
//! `arch/30-engine.md` §18.5~§18.6.
//!
//! M3.1 단계: struct 정의 + placeholder constructor만. M3.2에서 M2.H
//! `microbench_qnn_qwen_layer.rs`의 graph build 로직 (~1500 LOC)을 이식하여
//! `graph_handle` / `weight_handles` / `kv_handles` / `x_in/x_out_handles` /
//! `finalize_ms`를 채운다.
//!
//! INV-176 (LAYER_NODE_COUNT == 14)은 `qnn_oppkg::graph::layer` const를
//! re-export하여 동기화 강제 (M3.2 본격 도입).

/// 14-node single-layer QNN graph (M2.H 이식 대상). M3.1 placeholder.
///
/// 본 struct는 M3.2에서 다음 필드를 갖게 된다:
///   - `graph_handle: Qnn_GraphHandle_t`
///   - `weight_handles: Vec<u64>` (Q4_0 weight rpcmem mem handles)
///   - `kv_handles: (k, v)` (rpcmem-backed)
///   - `x_in_handle / x_out_handle`
///   - `mask_handle` (optional, ENG-QNN-228)
///   - `pos_scalar_arg` (i32)
///   - `finalize_ms: f64`
///   - `layer_idx: usize`
pub struct LayerGraph {
    /// Layer index (0..n_layers). M3.1 stub에서는 placeholder만 보유.
    pub layer_idx: usize,
    /// `graphFinalize` 측정값 (ms). M3.2 빌드 시점에 기록.
    pub finalize_ms: f64,
}

impl LayerGraph {
    /// Placeholder 생성 — M3.2에서 본격 build_layer_graph 호출 후 채워진다.
    ///
    /// 본 단계에서는 28개 slot을 미리 점유하기 위한 dummy 인스턴스 생성용.
    pub fn placeholder(layer_idx: usize) -> Self {
        Self {
            layer_idx,
            finalize_ms: 0.0,
        }
    }
}
