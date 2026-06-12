//! INV-LAYER-005: Engine L5 production binary(`L5_PRODUCTION_BINS` 상수 기준)는
//! L4 `session/`만 직접 import한다. test/microbench binary(`L5_SKIP_PATTERNS`)는
//! 본 규칙 밖.
//!
//! 검증 방식: `scripts/layer_lint.py --filter inv-layer-005 --baseline`으로 baseline
//! 대비 새로운 위반이 없음을 확인. `L5_PRODUCTION_BINS`에 등록된 바이너리
//! (`argus_cli.rs`, `argus_bench.rs`, `argus_eval.rs`)가 enforcement 대상이다.
//! (generate.rs는 d5ed71d2에서 폐기되어 목록에서 제외됨.)
//!
//! 베이스라인 정책 (spec/41-invariants.md §3.26):
//! - 현재 위반 건수는 `engine/tests/spec/inv_layer_baseline.json`에 고정.
//!
//! 참조: ARCHITECTURE.md §13.5

use std::path::PathBuf;
use std::process::Command;

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("engine 부모 경로")
        .to_path_buf()
}

#[test]
fn test_inv_layer_005_no_new_violations() {
    let root = project_root();
    let script = root.join("scripts").join("layer_lint.py");
    let baseline = root
        .join("engine")
        .join("tests")
        .join("spec")
        .join("inv_layer_baseline.json");

    if !script.exists() {
        panic!(
            "INV-LAYER-005: scripts/layer_lint.py 가 존재하지 않음: {:?}",
            script
        );
    }
    if !baseline.exists() {
        panic!(
            "INV-LAYER-005: baseline JSON이 존재하지 않음: {:?}",
            baseline
        );
    }

    let output = Command::new("python3")
        .arg(&script)
        .arg("--filter")
        .arg("inv-layer-005")
        .arg("--baseline")
        .arg(&baseline)
        .current_dir(&root)
        .output()
        .expect("python3 실행 실패 — python3가 설치되어 있어야 함");

    assert!(
        output.status.success(),
        "INV-LAYER-005: layer_lint.py 실행 실패\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let result: serde_json::Value =
        serde_json::from_str(&stdout).expect("layer_lint.py 출력이 유효한 JSON이어야 함");

    let violations = result
        .get("violations")
        .and_then(|v| v.as_array())
        .expect("violations 배열이 존재해야 함");

    assert!(
        violations.is_empty(),
        "INV-LAYER-005: baseline 이후 새로운 위반 {}건 발견:\n{}",
        violations.len(),
        serde_json::to_string_pretty(&violations).unwrap_or_default()
    );
}
