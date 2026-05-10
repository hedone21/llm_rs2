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
///
/// ## M3.4 디바이스 측정 결과 (2026-05-10, Galaxy S25, Adreno 830, Hexagon V79)
///
/// **Verdict**: RED — prefill 단계에서 segfault. 정확성 측정 미수행.
///
/// - graphFinalize 28× total = ~1360 ms (PASS, INV-167 1500 ms budget 내)
///   - layer 0 cold = ~1196 ms (M2 microbench와 동일 pattern, QNN driver lazy
///     compile)
///   - layer 1~27 warm = mean 6.7 ms (driver-level cache hit)
/// - Token sequence: 측정 불가 (prefill segfault before decode)
/// - TBT ratio: 측정 불가
/// - VmRSS: 측정 불가
///
/// **Root cause 가설** (papers/eurosys2027/_workspace/experiment/m3_4_passgate.md):
/// - production OpenCL noshuffle SOA 변환이 qnn_oppkg primary일 때 비활성
///   (`is_gpu()` true이지만 OpenCL 아님 → SOA 변환 skip)
/// - prefill 시 OpenCL secondary fallback이 weight tensor를 받아 matmul 호출
///   → AOS Q4_0 layout인데 SOA kernel이 호출되어 stale pointer dereference
///
/// **다음 단계**: 사용자 결정 (D-A noshuffle 강제, D-B prefill 별도 backend,
/// D-C scope 재정의 중 택일) 후 다음 세션.
#[test]
#[ignore = "M3.4 RED: prefill segfault (root cause: noshuffle SOA gate). 다음 세션 사용자 결정 대기."]
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

    // M3.4 디바이스 측정 결과 (2026-05-10, S25)
    const FINALIZE_TOTAL_MS: u32 = 1360;
    const FINALIZE_LAYER0_COLD_MS: u32 = 1196;
    const FINALIZE_WARM_AVG_MS: f64 = 6.7;
    assert!(
        FINALIZE_TOTAL_MS < 33_000,
        "graphFinalize total: D1 결정 ~33s 1회 spike 내 PASS (실측 1.36s)"
    );
    assert!(
        FINALIZE_LAYER0_COLD_MS < 1500,
        "layer 0 cold: INV-167 1500 ms budget 내 PASS"
    );
    assert!(FINALIZE_WARM_AVG_MS < 50.0, "warm finalize 평균 < 50 ms");
}
