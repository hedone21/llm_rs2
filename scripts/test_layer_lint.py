#!/usr/bin/env python3
"""
scripts/test_layer_lint.py
B-5a-0b: §13.8-J dispatch orchestrator zone parser 단위 테스트.

검증 케이스:
  1. zone marker 있는 함수: 내부 L3 호출 → 위반 제외
  2. zone marker 없는 함수: 내부 L3 호출 → 위반 유지
  3. 블록 형태 zone: marker 다음 `{` 블록 내부 → 위반 제외
  4. zone 뒤 함수: marker 밖 호출 → 위반 유지
"""

import sys
import os
import tempfile

# scripts/ 디렉토리를 sys.path에 추가하여 layer_lint 임포트 가능하게 함
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from layer_lint import (
    _find_exempt_zone_ranges,
    extract_imports,
    analyze,
    classify_module,
)

# ────────────────────────────────────────────────────────────────────
# 헬퍼
# ────────────────────────────────────────────────────────────────────

def _make_tmp_rs(content: str) -> str:
    """임시 .rs 파일을 생성하고 경로를 반환한다."""
    fd, path = tempfile.mkstemp(suffix=".rs")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


def _zone_ranges(content: str) -> list:
    lines = content.splitlines()
    return _find_exempt_zone_ranges(lines)


# ────────────────────────────────────────────────────────────────────
# Fixture: zone marker가 있는 함수
# ────────────────────────────────────────────────────────────────────

FIXTURE_FN_WITH_ZONE = """\
use std::sync::Arc;

// LAYER-EXEMPT: dispatch_orchestrator
pub fn build_plan(x: usize) -> usize {
    let flag = crate::layers::tensor_partition::partition_poll_flag_enabled();
    let _ = crate::inference::sampling::SamplingConfig::default();
    x + 1
}
"""

# zone marker 없는 함수
FIXTURE_FN_NO_ZONE = """\
use std::sync::Arc;

pub fn other_fn(x: usize) -> usize {
    let flag = crate::layers::tensor_partition::partition_poll_flag_enabled();
    x + 1
}
"""

# 블록 형태 zone: `{` 다음 줄에 marker
FIXTURE_BLOCK_ZONE = """\
pub fn dispatcher(x: usize) -> usize {
    // LAYER-EXEMPT: dispatch_orchestrator
    let flag = crate::layers::tensor_partition::partition_poll_flag_enabled();
    x + 1
}
"""

# zone이 있는 함수 다음에 zone 없는 함수
FIXTURE_MIXED = """\
// LAYER-EXEMPT: dispatch_orchestrator
pub fn zone_fn(x: usize) -> usize {
    let a = crate::layers::tensor_partition::partition_poll_flag_enabled();
    a
}

pub fn no_zone_fn(x: usize) -> usize {
    let b = crate::layers::tensor_partition::partition_poll_flag_enabled();
    b
}
"""

# ────────────────────────────────────────────────────────────────────
# 테스트: zone range 파서
# ────────────────────────────────────────────────────────────────────

def test_fn_zone_range_detected():
    """함수 형태 zone에서 range가 감지되어야 한다."""
    ranges = _zone_ranges(FIXTURE_FN_WITH_ZONE)
    assert len(ranges) == 1, f"zone 1개 기대, got {ranges}"
    start, end = ranges[0]
    # 함수 본문 `{` 이후 ~ `}` 포함 줄
    assert start >= 4, f"zone start should be >= 4 (line of opening brace), got {start}"
    assert end >= 8, f"zone end should cover closing brace, got {end}"
    print(f"  [PASS] test_fn_zone_range_detected: range={ranges[0]}")


def test_no_zone_no_range():
    """zone marker 없는 파일에서 range 목록이 비어야 한다."""
    ranges = _zone_ranges(FIXTURE_FN_NO_ZONE)
    assert ranges == [], f"zone 없어야 함, got {ranges}"
    print("  [PASS] test_no_zone_no_range")


def test_block_zone_range_detected():
    """블록 형태 zone(함수 본문 첫 줄에 marker)에서 range가 감지되어야 한다."""
    ranges = _zone_ranges(FIXTURE_BLOCK_ZONE)
    assert len(ranges) >= 1, f"zone 1개 이상 기대, got {ranges}"
    start, end = ranges[0]
    # marker가 2번째 줄, start = 2
    assert start == 2, f"zone start=2 기대, got {start}"
    print(f"  [PASS] test_block_zone_range_detected: range={ranges[0]}")


def test_mixed_zone_only_first_fn():
    """mixed fixture에서 zone range가 첫 번째 함수 본문만 커버해야 한다."""
    ranges = _zone_ranges(FIXTURE_MIXED)
    # zone이 정확히 1개 감지되어야 함
    assert len(ranges) == 1, f"zone 1개 기대, got {ranges}"
    start, end = ranges[0]
    # 두 번째 함수(9번째 줄 이후)는 zone에 포함되지 않아야 함
    # FIXTURE_MIXED에서 no_zone_fn은 9번째 줄 이후
    assert end < 8, f"zone end가 두 번째 함수 포함하면 안 됨, end={end}"
    print(f"  [PASS] test_mixed_zone_only_first_fn: range={ranges[0]}")


# ────────────────────────────────────────────────────────────────────
# 테스트: extract_imports + zone 플래그
# ────────────────────────────────────────────────────────────────────

def test_zone_imports_flagged():
    """zone 안의 import는 in_exempt_zone=True여야 한다."""
    path = _make_tmp_rs(FIXTURE_FN_WITH_ZONE)
    try:
        imports = extract_imports(path)
        # crate::layers::... 또는 crate::inference::... 가 zone 안에 있어야 함
        zone_imports = [(lineno, imp, it, iz) for lineno, imp, it, iz in imports
                        if "layers::" in imp or "inference::" in imp]
        assert len(zone_imports) > 0, "zone 안 L3 import가 감지되지 않음"
        for lineno, imp, is_test, is_exempt in zone_imports:
            assert is_exempt, f"line {lineno}: {imp} 는 zone 안이어야 함, is_exempt={is_exempt}"
        print(f"  [PASS] test_zone_imports_flagged: {len(zone_imports)}건 in_exempt_zone=True")
    finally:
        os.unlink(path)


def test_no_zone_imports_not_flagged():
    """zone 없는 파일의 import는 in_exempt_zone=False여야 한다."""
    path = _make_tmp_rs(FIXTURE_FN_NO_ZONE)
    try:
        imports = extract_imports(path)
        for lineno, imp, is_test, is_exempt in imports:
            assert not is_exempt, f"line {lineno}: zone 없는데 is_exempt=True"
        print("  [PASS] test_no_zone_imports_not_flagged")
    finally:
        os.unlink(path)


# ────────────────────────────────────────────────────────────────────
# 테스트: analyze() — zone에 따른 위반 제외/유지
# ────────────────────────────────────────────────────────────────────

def test_analyze_zone_excludes_l3_violations():
    """
    zone marker 있는 함수에서 L3 호출은 INV-LAYER-001 위반으로 집계되지 않아야 한다.
    (파일이 backend/ 하위에 위치한다고 가정)
    """
    # backend/ 하위에 파일을 임시로 만들어 L1로 분류되게 함
    tmpdir = tempfile.mkdtemp()
    backend_dir = os.path.join(tmpdir, "backend", "opencl")
    os.makedirs(backend_dir, exist_ok=True)
    rs_path = os.path.join(backend_dir, "plan.rs")
    with open(rs_path, "w") as f:
        f.write(FIXTURE_FN_WITH_ZONE)

    try:
        violations = analyze(tmpdir, inv_filter=None)
        # zone 안 L3 호출이 위반으로 잡히면 안 됨
        l3_viols = [v for v in violations
                    if ("layers::" in v.get("import", "") or "inference::" in v.get("import", ""))
                    and v.get("rule") == "INV-LAYER-001"]
        assert len(l3_viols) == 0, (
            f"zone 안 L3 호출이 INV-LAYER-001 위반으로 잡힘: {l3_viols}"
        )
        print(f"  [PASS] test_analyze_zone_excludes_l3_violations: 위반 0건")
    finally:
        import shutil
        shutil.rmtree(tmpdir)


def test_analyze_no_zone_keeps_l3_violations():
    """
    zone marker 없는 함수에서 L3 호출은 INV-LAYER-001 위반으로 집계되어야 한다.
    """
    tmpdir = tempfile.mkdtemp()
    backend_dir = os.path.join(tmpdir, "backend", "opencl")
    os.makedirs(backend_dir, exist_ok=True)
    rs_path = os.path.join(backend_dir, "plan.rs")
    with open(rs_path, "w") as f:
        f.write(FIXTURE_FN_NO_ZONE)

    try:
        violations = analyze(tmpdir, inv_filter=None)
        l3_viols = [v for v in violations
                    if "layers::" in v.get("import", "")
                    and v.get("rule") == "INV-LAYER-001"]
        assert len(l3_viols) >= 1, (
            f"zone 없는 L3 호출이 INV-LAYER-001로 잡혀야 하는데 위반 0건"
        )
        print(f"  [PASS] test_analyze_no_zone_keeps_l3_violations: 위반 {len(l3_viols)}건")
    finally:
        import shutil
        shutil.rmtree(tmpdir)


def test_analyze_mixed_partial_exclusion():
    """
    mixed fixture: zone_fn 내부는 제외, no_zone_fn 내부는 위반 유지.
    """
    tmpdir = tempfile.mkdtemp()
    backend_dir = os.path.join(tmpdir, "backend", "opencl")
    os.makedirs(backend_dir, exist_ok=True)
    rs_path = os.path.join(backend_dir, "plan.rs")
    with open(rs_path, "w") as f:
        f.write(FIXTURE_MIXED)

    try:
        violations = analyze(tmpdir, inv_filter=None)
        l3_viols = [v for v in violations
                    if "layers::" in v.get("import", "")
                    and v.get("rule") == "INV-LAYER-001"]
        # zone 밖 no_zone_fn 호출은 위반으로 남아야 함
        assert len(l3_viols) >= 1, (
            f"zone 밖 L3 호출이 위반으로 잡혀야 함, got {l3_viols}"
        )
        # zone 안 zone_fn 호출은 위반에서 제외되어야 함 — 총 위반 1건 (no_zone_fn만)
        assert len(l3_viols) == 1, (
            f"정확히 1건(no_zone_fn)만 위반이어야 하는데 {len(l3_viols)}건: {l3_viols}"
        )
        print(f"  [PASS] test_analyze_mixed_partial_exclusion: 위반 {len(l3_viols)}건 (no_zone_fn만)")
    finally:
        import shutil
        shutil.rmtree(tmpdir)


# ────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────

TESTS = [
    test_fn_zone_range_detected,
    test_no_zone_no_range,
    test_block_zone_range_detected,
    test_mixed_zone_only_first_fn,
    test_zone_imports_flagged,
    test_no_zone_imports_not_flagged,
    test_analyze_zone_excludes_l3_violations,
    test_analyze_no_zone_keeps_l3_violations,
    test_analyze_mixed_partial_exclusion,
]


def main():
    passed = 0
    failed = 0
    for test_fn in TESTS:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {test_fn.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n결과: {passed} PASS / {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
