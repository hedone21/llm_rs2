//! QNN OpPackage M3 — Layer Graph Contract (ENG-QNN-221~230).
//!
//! Spec ref tags for coverage: inv_176, inv_177
//!
//! Spec: `spec/30-engine.md` 부록 C.4 (ENG-QNN-221 ~ ENG-QNN-230),
//! `spec/41-invariants.md` §3.24 (INV-176 LAYER_NODE_COUNT == 14,
//! INV-177 KV layout view transform zero-copy),
//! `arch/30-engine.md` §18.5~§18.6.
//!
//! 매핑:
//! - ENG-QNN-221 / INV-176: 14-node layer graph build-time const.
//! - ENG-QNN-222: RoPE는 kernel_rope_simple_oop 사용 (M2.B 검증).
//! - ENG-QNN-223: KvScatter multi-output (k_rot → KV_K, v → KV_V).
//! - ENG-QNN-224: SiluMul OutputTensorAliased (M2 INV-164).
//! - ENG-QNN-225 / INV-177: KV layout view transform zero-copy.
//! - ENG-QNN-226: layer 입출력 인터페이스는 M2 ENG-QNN-110 그대로.
//! - ENG-QNN-227: pos scalar arg는 매 forward call마다 갱신.
//! - ENG-QNN-228: mask buffer model load 시 1회 alloc, 매 token mask[pos] update.
//! - ENG-QNN-229: Q4_0 weight RawBytes(18, N/32) (M2 INV-165).
//! - ENG-QNN-230: layer graph builder = `crates/qnn_oppkg::graph::layer::build_layer_graph`.

#![cfg(feature = "qnn")]

use llm_rs2::backend::qnn_oppkg::layer_graph::{
    FINALIZE_BUDGET_MS, LAYER_INTERMEDIATE_COUNT, LAYER_NODE_COUNT, LayerConfig,
};

/// INV-176: 14-node layer graph build-time const sanity. Phase analyzer (M4.0)
/// 도입 시 본 const와 동기화 강제 검증 추가 예정.
#[test]
fn qnn_221_layer_node_count_is_14() {
    assert_eq!(
        LAYER_NODE_COUNT, 14,
        "ENG-QNN-221 / INV-176: layer graph는 정확히 14 노드여야 한다 \
         (RmsNorm pre + Q/K/V + RoPE Q/K + KvScatter + FlashAttn + O + Add#1 + \
          RmsNorm post + gate/up + SiluMul + down + Add#2 = 14)"
    );
}

/// 14 nodes 사이의 NATIVE intermediate tensor가 13개 (graph endpoint 제외).
#[test]
fn qnn_221_intermediate_count_is_13() {
    assert_eq!(
        LAYER_INTERMEDIATE_COUNT, 13,
        "y1, q, k, v, q_rot, k_rot, attn_out, o, x_attn, y2, gate, up, silu_out = 13 \
         (silu_out aliases gate via M2.G OutputTensorAliased)"
    );
}

/// ENG-QNN-225 / INV-177: LayerConfig (Qwen2.5-1.5B) 정합. dim=1536, n_head=12,
/// n_kv_heads=2, head_dim=128, ffn_dim=8960, kv_capacity=2048.
///
/// host re-export가 `crates/qnn_oppkg::graph::layer::LayerConfig::qwen2p5_1p5b`
/// 와 동일 값을 노출하는지 검증. M3.3 forward path가 본 메타데이터로 KV
/// layout view transform을 zero-cost로 수행.
#[test]
fn qnn_225_layer_config_qwen2p5_1p5b_defaults() {
    let cfg = LayerConfig::qwen2p5_1p5b();
    assert_eq!(cfg.dim, 1536, "ENG-QNN-225: hidden dim 1536");
    assert_eq!(cfg.n_head, 12, "ENG-QNN-225: query heads 12");
    assert_eq!(cfg.n_kv_heads, 2, "ENG-QNN-225: KV heads 2 (GQA 6:1)");
    assert_eq!(cfg.head_dim, 128, "ENG-QNN-225: per-head dim 128");
    assert_eq!(cfg.ffn_dim, 8960, "ENG-QNN-225: FFN inner dim 8960");
    assert_eq!(
        cfg.kv_capacity, 2048,
        "ENG-QNN-225: max context 2048 (M2.H 검증된 max-padded fixed shape)"
    );
}

/// ENG-QNN-225: Q proj out = n_head × head_dim = 12 × 128 = 1536 (Qwen2.5-1.5B
/// isotropic). KV proj out = n_kv_heads × head_dim = 2 × 128 = 256.
#[test]
fn qnn_225_q_kv_proj_dims_consistent() {
    let cfg = LayerConfig::qwen2p5_1p5b();
    assert_eq!(cfg.q_proj_out(), cfg.dim, "Qwen2.5-1.5B isotropic Q proj");
    assert_eq!(cfg.kv_proj_out(), 256, "GQA 6:1: 2*128 = 256");
}

/// ENG-QNN-229 / INV-165: Q4_0 weight RawBytes(18, N/32) layout sanity.
/// `q40_bytes(N, K)` 가 `(N*K/32) * 16` (q) + `(N*K/32) * 2` (d_halves) 를
/// 반환해야 한다 (per-block 32 elements, 16 nibbles + 2 byte FP16 scale).
#[test]
fn qnn_229_q40_bytes_layout() {
    // Qwen2.5-1.5B O proj: 1536x1536
    let (q, d) = LayerConfig::q40_bytes(1536, 1536);
    assert_eq!(q, 1_179_648, "Q4_0 q layout: 1536*1536/32*16");
    assert_eq!(
        d, 147_456,
        "Q4_0 d layout: 1536*1536/32*2 (FP16 per-block scale)"
    );

    // gate/up: 8960x1536
    let (q, d) = LayerConfig::q40_bytes(8960, 1536);
    assert_eq!(q, 8960 * 1536 / 32 * 16);
    assert_eq!(d, 8960 * 1536 / 32 * 2);

    // K assertion: must be multiple of 32. const fn에서 panic을 trigger하지 않는
    // 합리적 입력값만 검증 (런타임 panic test는 별도 경로).
}

/// ENG-QNN-209/INV-167: graphFinalize budget 200 ms/layer. 본 단계는 host
/// 빌드에서 build가 항상 Err이므로 budget 자체의 const 정합만 검증.
/// 디바이스 측정 (28× ≤ 200 ms)은 M3.4 메인 게이트 측정 시 동반.
#[test]
fn qnn_209_finalize_budget_ms_is_200() {
    assert_eq!(
        FINALIZE_BUDGET_MS, 200,
        "ENG-QNN-209 / INV-167: graphFinalize 1회 호출은 layer당 ≤ 200 ms"
    );
}
