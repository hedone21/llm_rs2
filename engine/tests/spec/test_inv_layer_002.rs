//! INV-LAYER-002: Engine L2 `shared/`(buffer/, memory/, auf/)는 L3(`pressure/`,
//! `inference/`), L4(`session/`), L5(`bin/`) 어떤 모듈도 import 금지.
//! backend-specific buffer는 `buffer/`가 아닌 `backend/<be>/buffer/`에 위치해야 한다.
//!
//! 검증 방식: `scripts/layer_lint.py --filter inv-layer-002 --baseline`으로 baseline
//! 대비 새로운 L2→L3+/L1 위반이 없음을 확인.
//!
//! 베이스라인 정책 (spec/41-invariants.md §3.26):
//! - 현재 위반 건수는 `engine/tests/spec/inv_layer_baseline.json`에 고정.
//! - 마이그레이션 PR마다 baseline을 줄여가며, 최종적으로 0건이 목표.
//!
//! 참조: ARCHITECTURE.md §13.5 V-07~V-09

use std::path::PathBuf;
use std::process::Command;

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("engine 부모 경로")
        .to_path_buf()
}

#[test]
fn test_inv_layer_002_no_new_violations() {
    let root = project_root();
    let script = root.join("scripts").join("layer_lint.py");
    let baseline = root
        .join("engine")
        .join("tests")
        .join("spec")
        .join("inv_layer_baseline.json");

    if !script.exists() {
        panic!(
            "INV-LAYER-002: scripts/layer_lint.py 가 존재하지 않음: {:?}",
            script
        );
    }
    if !baseline.exists() {
        panic!(
            "INV-LAYER-002: baseline JSON이 존재하지 않음: {:?}",
            baseline
        );
    }

    let output = Command::new("python3")
        .arg(&script)
        .arg("--filter")
        .arg("inv-layer-002")
        .arg("--baseline")
        .arg(&baseline)
        .current_dir(&root)
        .output()
        .expect("python3 실행 실패 — python3가 설치되어 있어야 함");

    assert!(
        output.status.success(),
        "INV-LAYER-002: layer_lint.py 실행 실패\nstderr: {}",
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
        "INV-LAYER-002: baseline 이후 새로운 위반 {}건 발견:\n{}",
        violations.len(),
        serde_json::to_string_pretty(&violations).unwrap_or_default()
    );
}
