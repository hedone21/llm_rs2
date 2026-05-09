//! QNN OpPackage M3 — Pass Gate (ENG-QNN-231~240) stub.
//!
//! Spec ref tags for coverage: inv_169, inv_172, inv_173, inv_178, inv_179, inv_180
//!
//! Spec: `spec/30-engine.md` 부록 C.5 (ENG-QNN-231 ~ ENG-QNN-240),
//! `spec/41-invariants.md` §3.24 (INV-169, INV-172, INV-173, INV-178, INV-179,
//! INV-180), `arch/30-engine.md` §18.12 (test 위치 매핑).
//!
//! M3.0 단계: 본 stub 파일은 entry point만 마련. 본문은 M3.4 (메인 게이트:
//! 32-token sequence) 진입 시 Senior Implementer + Tester가 채운다.
//!
//! 매핑:
//! - ENG-QNN-231 / INV-172: Qwen2.5-1.5B Q4_0 32-token greedy decode
//!   token sequence 100% 일치 (vs --backend opencl).
//! - ENG-QNN-232: single-layer accuracy max_abs_err < 1e-2 (M3.3 단계에서 격리).
//! - ENG-QNN-233 / INV-179: TBT GREEN ≤ 1.10× / YELLOW 1.10~1.20× / RED > 1.20×
//!   (Galaxy S25, 5회 평균 + warm-up 3회 제외).
//! - ENG-QNN-234: VmRSS ≤ baseline × 1.10.
//! - ENG-QNN-235 / INV-178: VmRSS slope < 50 KB/token (token leak detector).
//! - ENG-QNN-236: graphFinalize ≤ 200 ms × 28 (M2 INV-163 보존).
//! - ENG-QNN-237 / INV-169: cargo test --workspace --features opencl 회귀 0건.
//! - ENG-QNN-238 / INV-169: --backend opencl decode TBT ≤ 1.05× baseline
//!   (Backend trait 신규 method overhead 검증).
//! - ENG-QNN-239 / INV-175: trait fallback method 호출 count == 0
//!   (`test_qnn_211.rs`에서도 매핑되며 M3.3 측정 + M3.4 대규모 검증).
//! - ENG-QNN-240 / INV-173: TBT 측정 wall-clock only (--profile-events 금지).
//! - INV-180: cdylib binary는 dlopen 산출물로 engine binary가 link하지
//!   않음 — `cargo tree` 정적 검사.
//!
//! 본 stub의 #[test] 본문은 모두 디바이스 측정 또는 CI script로 검증되며,
//! host unit test로 직접 실행 가능한 항목은 INV-180 cargo tree 정도.

/// Placeholder — M3.4 메인 게이트 측정 시 본문 채움. 디바이스 microbench
/// (`microbench_qnn_oppkg_decode32`)에서 32-token greedy decode 정확성 +
/// TBT band + VmRSS slope 측정 후 본 host stub은 ID 매핑 sanity로만 잔존.
#[test]
#[ignore = "M3.4에서 디바이스 측정 (Galaxy S25) — 32-token decode + TBT + VmRSS"]
fn placeholder_qnn_pass_gate_seam() {
    let spec_ids = [
        "ENG-QNN-231", // 32-token decode 일치 (INV-172)
        "ENG-QNN-232", // single-layer accuracy
        "ENG-QNN-233", // TBT band (INV-179)
        "ENG-QNN-234", // VmRSS ≤ × 1.10
        "ENG-QNN-235", // VmRSS slope (INV-178)
        "ENG-QNN-236", // graphFinalize ≤ 200 ms × 28
        "ENG-QNN-237", // cargo test 회귀 0 (INV-169)
        "ENG-QNN-238", // OpenCL TBT 무회귀 (INV-169)
        "ENG-QNN-239", // trait fallback count 0 (INV-175)
        "ENG-QNN-240", // wall-clock only (INV-173)
    ];
    assert_eq!(spec_ids.len(), 10, "ENG-QNN-231~240 = 10 entries");

    // TBT band 임계값 sanity (INV-179) — 디바이스 측정 결과를 본 임계값에
    // 매핑하므로 stub에서도 기록.
    const GREEN_MAX: f64 = 1.10;
    const YELLOW_MAX: f64 = 1.20;
    assert!(GREEN_MAX < YELLOW_MAX, "INV-179: GREEN < YELLOW band");
}
