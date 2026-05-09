//! QNN OpPackage M3 — Backend trait 신규 method (ENG-QNN-211~220) stub.
//!
//! Spec ref tags for coverage: inv_168, inv_174, inv_175
//!
//! Spec: `spec/30-engine.md` 부록 C.3 (ENG-QNN-211 ~ ENG-QNN-220),
//! `spec/41-invariants.md` §3.24 (INV-168, INV-174, INV-175),
//! `arch/30-engine.md` §18.7~§18.9.
//!
//! M3.0 단계: 본 stub 파일은 entry point만 마련. 본문은 M3.3 (single-layer
//! forward via graph executor) 진입 시 Senior Implementer가 채운다.
//!
//! 매핑:
//! - ENG-QNN-211: Backend trait `supports_layer_graph()`, `execute_layer_graph(...)`
//!   default impl 추가 — 기존 backend (CPU/OpenCL/CUDA) 변경 없이 컴파일.
//! - ENG-QNN-212 / INV-174: supports_layer_graph idempotent (model load 후
//!   항상 true 또는 false).
//! - ENG-QNN-213/214: execute_layer_graph pre/post conditions
//!   (layer_idx < n_layers, kv_cache shape INV-168, pos < 2048, x/x_out shape).
//! - ENG-QNN-215: caller `&mut KVCache` lifetime ownership (M2 INV-114 정신).
//! - ENG-QNN-216 / INV-175: trait fallback instrumentation — fast path 발동 시
//!   matmul/rope/etc 호출 count == 0.
//! - ENG-QNN-217: enqueue_write_async/wait_event_blocking/supports_async_transfer
//!   M4 chunk dispatcher용 (M3에서는 OpenCL secondary 위임 또는 자체 큐).
//! - ENG-QNN-218: weight swap hook noop (M4 chunk swap 시 본격 활용).
//! - ENG-QNN-219: unknown backend `bail!` 보존 (INV-170 동등).
//! - ENG-QNN-220: --qnn-graph-cache-prebuild / --qnn-allow-fallback CLI args.

/// Placeholder — M3.3에서 trait method default impl + idempotent +
/// instrumentation test 본문 채움.
#[test]
#[ignore = "M3.3에서 구현 — Backend trait 신규 method + fast path + fallback count"]
fn placeholder_qnn_backend_trait_methods_seam() {
    let spec_ids = [
        "ENG-QNN-211", // trait default impl
        "ENG-QNN-212", // supports_layer_graph idempotent (INV-174)
        "ENG-QNN-213", // execute_layer_graph pre conditions
        "ENG-QNN-214", // execute_layer_graph post conditions
        "ENG-QNN-215", // KVCache lifetime
        "ENG-QNN-216", // fallback instrumentation (INV-175)
        "ENG-QNN-217", // async transfer methods (M4 hook)
        "ENG-QNN-218", // weight swap hook noop
        "ENG-QNN-219", // unknown backend bail (INV-170)
        "ENG-QNN-220", // CLI args
    ];
    assert_eq!(spec_ids.len(), 10, "ENG-QNN-211~220 = 10 entries");
}
