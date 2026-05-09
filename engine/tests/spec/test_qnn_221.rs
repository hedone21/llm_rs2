//! QNN OpPackage M3 — Layer Graph Contract (ENG-QNN-221~230) stub.
//!
//! Spec ref tags for coverage: inv_176, inv_177
//!
//! Spec: `spec/30-engine.md` 부록 C.4 (ENG-QNN-221 ~ ENG-QNN-230),
//! `spec/41-invariants.md` §3.24 (INV-176 LAYER_NODE_COUNT == 14,
//! INV-177 KV layout view transform zero-copy),
//! `arch/30-engine.md` §18.5~§18.6.
//!
//! M3.0 단계: 본 stub 파일은 entry point만 마련. 본문은 M3.2 (Layer graph
//! builder 이식 + per-layer cache 빌드) 진입 시 Senior Implementer가 채운다.
//!
//! 매핑:
//! - ENG-QNN-221 / INV-176: 14-node layer graph (M2 13-node + RoPE OOP Q/K
//!   분리 + Add residual 2개 명시). build-time const LAYER_NODE_COUNT == 14.
//! - ENG-QNN-222: RoPE는 kernel_rope_simple_oop 사용 (M2.B 검증, simple_ops.cl
//!   에 추가됨).
//! - ENG-QNN-223: KvScatter multi-output (k_rot → KV_K, v → KV_V). graph
//!   external buffer (rpcmem-backed) read-after-write hazard SDK 자동 처리.
//! - ENG-QNN-224: SiluMul OutputTensorAliased 패턴 (M2 INV-164 보존).
//! - ENG-QNN-225 / INV-177: KV layout view transform zero-copy
//!   (HeadMajor stride == graph 입력 stride).
//! - ENG-QNN-226: layer 입출력 인터페이스는 M2 ENG-QNN-110 그대로.
//! - ENG-QNN-227: pos scalar arg는 매 forward call마다 graph executor 갱신.
//! - ENG-QNN-228: mask buffer model load 시 1회 alloc, 매 token mask[pos] update.
//! - ENG-QNN-229: Q4_0 weight RawBytes(18, N/32) (M2 INV-165 보존).
//! - ENG-QNN-230: layer graph builder = `crates/qnn_oppkg::graph::layer::build_layer_graph`
//!   직접 재사용 (수정 없음).

/// Placeholder — M3.2에서 LAYER_NODE_COUNT 검증 + view transform 측정 +
/// graph builder host crate dep edge sanity 본문 채움.
#[test]
#[ignore = "M3.2에서 구현 — layer graph builder 이식 + 28× per-layer cache 빌드"]
fn placeholder_qnn_layer_graph_contract_seam() {
    let spec_ids = [
        "ENG-QNN-221", // 14-node graph (INV-176)
        "ENG-QNN-222", // RoPE OOP
        "ENG-QNN-223", // KvScatter multi-output
        "ENG-QNN-224", // SiluMul alias
        "ENG-QNN-225", // KV layout view transform (INV-177)
        "ENG-QNN-226", // I/O interface
        "ENG-QNN-227", // pos scalar arg
        "ENG-QNN-228", // mask buffer
        "ENG-QNN-229", // Q4_0 RawBytes
        "ENG-QNN-230", // builder reuse
    ];
    assert_eq!(spec_ids.len(), 10, "ENG-QNN-221~230 = 10 entries");

    // INV-176: build-time const sanity (real check: M3.2에서 LAYER_NODE_COUNT
    // 상수가 14인지 + phase analyzer enumerate set과 동기화 — 본 placeholder는
    // 수치 14만 기록).
    const EXPECTED_LAYER_NODE_COUNT: usize = 14;
    assert_eq!(EXPECTED_LAYER_NODE_COUNT, 14);
}
