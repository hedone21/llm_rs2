//! QNN OpPackage M3 backend module — ENG-QNN-201~210 stub.
//!
//! Spec ref tags for coverage: inv_166, inv_167, inv_170, inv_171
//!
//! Spec: `spec/30-engine.md` 부록 C.2 (ENG-QNN-201 ~ ENG-QNN-210),
//! `spec/41-invariants.md` §3.24 (INV-166, INV-167, INV-170, INV-171),
//! `arch/30-engine.md` §18.1~§18.4.
//!
//! M3.0 단계: 본 stub 파일은 entry point만 마련. 본문은 M3.1 (Backend skeleton)
//! 진입 시 Senior Implementer가 채운다.
//!
//! 매핑:
//! - ENG-QNN-201 / INV-166: QnnOppkgBackend가 Backend trait 모든 필수 method를
//!   OpenCL과 동일 시그니처로 구현 — 컴파일 타임 trait bound 검증.
//! - ENG-QNN-202 / INV-170: --backend qnn_oppkg | qnngpu opt-in flag (default
//!   off). unknown backend는 bail!.
//! - ENG-QNN-203 / INV-167: Layer graph cache lifetime — model load 후
//!   graphFinalize == 28회, decode 동안 +0회.
//! - ENG-QNN-204 / INV-171: KV cache rpcmem(DMA-BUF heap)-backed + host_ptr
//!   expose. 디바이스 검증 (host stub은 placeholder).
//! - ENG-QNN-205~209: weight slot snapshot, OpenCL secondary, fast path
//!   trigger, prefill fallback, eager prebuild — 본문은 M3.1~M3.2에서 작성.
//!
//! Pass-gate (M3.1): cargo build --features qnn,opencl PASS, --backend
//! qnn_oppkg dispatch가 model load까지 진행 + forward는 unimplemented! 허용.

/// Placeholder — M3.1에서 trait bound + dispatch path test 본문 채움.
///
/// 현재는 spec ID 매핑(주석 inv_166/167/170/171) 기록과 harness 등록
/// 검증만 수행. 실제 검증은 Senior Implementer가 M3.1 단계에서 본
/// 함수 본문을 교체한다.
#[test]
#[ignore = "M3.1에서 구현 — qnn_oppkg backend skeleton + dispatcher 등록 후"]
fn placeholder_qnn_backend_module_seam() {
    // Architect가 M3.0에서 명세한 ID 매핑 sanity:
    let spec_ids = [
        "ENG-QNN-201", // QnnOppkgBackend trait impl
        "ENG-QNN-202", // --backend qnn_oppkg dispatch
        "ENG-QNN-203", // graph cache lifetime
        "ENG-QNN-204", // rpcmem KV
        "ENG-QNN-205", // weight slot snapshot
        "ENG-QNN-206", // OpenCL secondary
        "ENG-QNN-207", // fast path
        "ENG-QNN-208", // prefill fallback
        "ENG-QNN-209", // eager prebuild
        "ENG-QNN-210", // mask buffer
    ];
    assert_eq!(spec_ids.len(), 10, "ENG-QNN-201~210 = 10 entries");
}
