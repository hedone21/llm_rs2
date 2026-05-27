#!/usr/bin/env python3
"""
scripts/layer_lint.py
Engine 내부 레이어드 아키텍처 위반 검출 도구.

INV-LAYER-001~005 (spec/41-invariants.md §3.26, ARCHITECTURE.md §13.5) 위반을
engine/src/**/*.rs 파일의 `use crate::` / 인라인 `crate::` import 분석으로 검출.

사용법:
  python3 scripts/layer_lint.py
  python3 scripts/layer_lint.py --baseline engine/tests/spec/inv_layer_baseline.json
  python3 scripts/layer_lint.py --filter inv-layer-001
  python3 scripts/layer_lint.py --baseline ... --filter inv-layer-002
"""

import argparse
import json
import os
import re
import sys

# ────────────────────────────────────────────────────────────────────
# DATA_CONSUMER_PATTERNS — L1 backend의 L3 weight struct/enum import를
# data consumer 카테고리로 분류하여 INV-LAYER-001 baseline에서 자동 제외.
# spec/41-invariants.md INV-LAYER-001 비고 "Data consumer 카테고리" 참조.
# ────────────────────────────────────────────────────────────────────

DATA_CONSUMER_PATTERNS = [
    re.compile(r"crate::models::weights::[A-Z]"),                        # struct/enum (UpperCamelCase)
    re.compile(r"crate::models::weights::[a-z_]+::[A-Z]"),               # sub-module struct/enum (예: rpcmem_secondary::RpcmemLayerRegion)
    re.compile(r"crate::layers::transformer_layer::TransformerLayer"),
    re.compile(r"crate::pressure::kv_cache::KVCache$"),                  # struct only, KVCacheOps trait 제외
]


def is_data_consumer(import_path: str) -> bool:
    """import_path가 data consumer 카테고리에 해당하면 True."""
    return any(p.search(import_path) for p in DATA_CONSUMER_PATTERNS)

# ────────────────────────────────────────────────────────────────────
# Layer 매핑 규칙 (ARCHITECTURE.md §13.2, §13.4 기준)
# ────────────────────────────────────────────────────────────────────

# 각 모듈 경로 prefix → layer 레이블
# 우선순위: 더 구체적인 prefix가 먼저 매칭됨 (정렬 순서로 처리)
LAYER_RULES = [
    # L1: hardware backend impl
    ("backend/opencl",          "L1"),
    ("backend/cpu",             "L1"),
    ("backend/cuda_embedded",   "L1"),
    ("backend/cuda_pc",         "L1"),
    ("backend",                 "L1"),

    # L2: shared buffer/memory abstractions + generic types
    # buffer/ 안에 backend-specific 파일이 섞여 있지만 현재 코드 구조상 buffer/ 전체를 L2로 분류
    ("buffer",                  "L2"),
    ("memory",                  "L2"),
    ("auf",                     "L2"),   # §13.8-A 결정: shared/auf/ = L2 자산
    # Step 4-A: L2 abstractions promoted from core/ to engine/src/ top-level
    ("tensor",                  "L2"),
    ("shape",                   "L2"),
    ("quant",                   "L2"),
    ("thread_pool",             "L2"),
    # B-2a: §13.8-G shared identifier promotion — op span identifier used by
    # both observability (producer) and L3 inference (consumer).
    ("op_kind",                 "L2"),

    # L3-pressure: KV cache 관리, eviction, offload, swap handler
    # Step 4-D: core/{cache_manager,kv_cache,kivi_cache,kv_migrate,eviction,
    # pressure,offload} → engine/src/pressure/ (top-level grouping).
    ("pressure",                "L3-pressure"),

    # L3-inference: 추론 연산 도메인
    # Step 4-C: sampling/skip_config/speculative/attention_scores promoted to engine/src/inference/
    ("inference",               "L3-inference"),
    ("layers",                  "L3-inference"),
    ("models",                  "L3-inference"),  # models/weights/* 포함

    # L3-qcf: Quality Cost Function 측정 도메인 (S-3b-2, 2026-05-24)
    # QCF는 lossy action(eviction/quantization/skip)의 quality cost 측정 도메인.
    # pressure 도 inference 도 아닌 *측정/평가* 도메인이며, 양쪽 모두로부터 호출됨.
    # data identifier 는 §G로 L2(`engine/src/qcf_types.rs`)에 격상, 측정 로직은
    # 본 L3-qcf 도메인(`engine/src/qcf/`)에 유지.
    ("qcf",                     "L3-qcf"),

    # L4: orchestration (Step 5-A: chat_template promoted from core/ to session/)
    ("session",                 "L4"),

    # cross-cutting: observability (Step 5-D/5-E: events/rss_trace promoted from core/ to observability/)
    # Step 5b-A/B: profile/eval physically relocated to observability/, redundant rules removed.
    ("observability",           "observability"),

    # cross-cutting: resilience (Step 5-B/5-C: sys_monitor/gpu_yield promoted from core/ to resilience/)
    ("resilience",              "resilience"),

    # L5: binary entrypoints
    ("bin",                     "L5"),

    # experiment
    ("experiment",              "L4"),

    # bin_helpers
    ("bin_helpers",             "L5"),
]

# INV-LAYER-005 enforcement 대상 외 binary 파일 prefix 목록
L5_SKIP_PATTERNS = [
    "microbench_",
    "test_",
    "probe_",
    "stage",
    "signal_injector",
    "auf_tool",
    "micro_bench",
]


def classify_module(rel_path: str) -> str:
    """
    engine/src/ 기준 상대경로로 layer를 반환.
    예: "backend/opencl/mod.rs" → "L1"
        "backend.rs" → "L2" (Step 4-A: top-level trait file)
    """
    # 경로 구분자를 /로 통일
    norm = rel_path.replace(os.sep, "/")
    # Step 4-A: top-level abstraction *.rs files take precedence over
    # directory-prefix rules (Rust 2018+ pattern — trait lives next to impl dir).
    TOP_LEVEL_L2 = {"backend.rs", "buffer.rs", "memory.rs", "tensor.rs",
                    "shape.rs", "quant.rs", "thread_pool.rs", "op_kind.rs",
                    "partition_workspace.rs", "kv_cache_ops.rs",
                    "yield_policy.rs", "qcf_types.rs",
                    "qcf_computer.rs", "qcf_collector.rs",
                    "model_config.rs", "layer_boundary_hook.rs",
                    "runtime_resources_access.rs",
                    "action_diag_helper.rs"}
    if norm in TOP_LEVEL_L2:
        return "L2"
    for prefix, layer in LAYER_RULES:
        if norm.startswith(prefix + "/") or norm == prefix or norm.startswith(prefix + "."):
            return layer
    return "unknown"


def classify_import(import_path: str) -> str:
    """
    `use crate::foo::bar` 또는 인라인 `crate::foo::bar`에서 foo::bar 부분을
    layer로 분류.
    예: "core::pressure::..." → "L3-pressure"
         "backend::opencl::..." → "L1"
         "backend::Backend"    → "L2" (Step 4-A: top-level trait file)
    """
    # crate:: 제거
    p = import_path.strip()
    if p.startswith("crate::"):
        p = p[len("crate::"):]
    # Step 4-A: leaf identifier가 PascalCase면 module path 추출 (top-level
    # *.rs 파일에 정의된 trait/struct를 디렉토리 prefix와 분리)
    segs = [s for s in p.split("::") if s]  # trailing `::` 등 empty segment 제거
    mod_segs: list[str] = []
    for s in segs:
        if s and s[0].isupper():
            break
        mod_segs.append(s)
    if not mod_segs:
        mod_segs = segs
    mod_path = "/".join(mod_segs)
    # Top-level L2 abstraction files (engine/src/*.rs, Rust 2018+ pattern)
    if mod_path in ("backend", "buffer", "memory", "tensor", "shape",
                    "quant", "thread_pool", "op_kind", "partition_workspace",
                    "kv_cache_ops", "yield_policy",
                    "runtime_resources_access", "action_diag_helper"):
        return "L2"
    # 기존 경로 기반 매칭으로 fallback
    return classify_module(mod_path if mod_path else p.replace("::", "/"))


# ────────────────────────────────────────────────────────────────────
# 위반 판정 규칙 (INV-LAYER-001~005)
# ────────────────────────────────────────────────────────────────────

_L3_DOMAINS = ("L3-pressure", "L3-inference", "L3-qcf")


def check_violation(src_layer: str, dst_layer: str, src_rel: str) -> tuple[str | None, str | None, str | None]:
    """
    (src_layer, dst_layer) 쌍에 대해 위반하는 INV-LAYER-XXX와 kind 문자열을 반환.
    위반 없으면 (None, None, None).
    반환: (inv_id, kind, note)
    """
    # INV-LAYER-001: L1 backend → L2(shared/buffer/memory/auf) + cross-cutting 외 import 금지
    # 허용: L1→L2, L1→L1(동일 backend 내부), L1→observability, L1→resilience, L1→L3-core(Backend trait)
    # 금지: L1→L3-pressure, L1→L3-inference, L1→L3-qcf, L1→L4, L1→L5
    if src_layer == "L1":
        if dst_layer in (*_L3_DOMAINS, "L4", "L5"):
            # V-01: L1→cross-cutting concrete (resilience임에도 concrete 직접 import)
            # V-02: L1→L3-inference
            # V-03: L1→L3-inference (models/weights)
            kind = f"L1→{dst_layer} (역방향 import)"
            return ("INV-LAYER-001", kind, None)
        # V-04, V-05: L1↔L1 cross-backend — 동일 backend 내부는 허용, 다른 backend는 INV-LAYER-001 위반
        # (백엔드 간 cross-import는 architecture 위반이지만 허용 zone에 해당하므로 INV-LAYER-001 mild 위반으로 처리)
        if dst_layer == "L1":
            # 동일 backend 폴더 내부는 허용 (x86.rs → cpu/common.rs 등)
            # 다른 backend로의 import는 INV-LAYER-001 (교차 backend)
            # src_rel로 backend prefix 비교
            src_be = _extract_backend(src_rel)
            # dst는 import 경로에서 판단 — 여기선 호출자가 별도로 처리
            pass
        if dst_layer == "resilience":
            # V-01: opencl → gpu_self_meter (resilience의 concrete)
            kind = f"L1→resilience (cross-cutting concrete 직접 의존)"
            return ("INV-LAYER-001", kind, "V-01 패턴")

    # INV-LAYER-002: L2 → L3+/L4/L5 import 금지
    if src_layer == "L2":
        if dst_layer in (*_L3_DOMAINS, "L4", "L5"):
            kind = f"L2→{dst_layer} (상위 레이어 역방향 import)"
            return ("INV-LAYER-002", kind, None)
        if dst_layer == "L1":
            kind = f"L2→L1 (backend-specific impl 직접 의존)"
            return ("INV-LAYER-002", kind, "V-07 패턴")

    # INV-LAYER-003 (S-3b-2, 2026-05-24 일반화): 모든 L3 도메인 쌍 사이 cross-domain
    # concrete import 금지 (Q1 결정 — 예외는 사용자 확인 후 처리). trait 만 허용.
    if src_layer in _L3_DOMAINS and dst_layer in _L3_DOMAINS and src_layer != dst_layer:
        kind = f"{src_layer}→{dst_layer} (cross-domain concrete import)"
        return ("INV-LAYER-003", kind, None)
    # L3→L1 (backend concrete downcast)
    if src_layer in _L3_DOMAINS and dst_layer == "L1":
        kind = f"{src_layer}→L1 (backend impl 직접 의존)"
        return ("INV-LAYER-003", kind, "downcast 패턴")
    # L3→cross-cutting concrete (V-10, V-14, V-22, V-26 등)
    if src_layer in _L3_DOMAINS and dst_layer in ("observability", "resilience"):
        kind = f"{src_layer}→cross-cutting({dst_layer}) (concrete 직접 의존, trait inversion 필요)"
        return ("INV-LAYER-003", kind, None)

    # INV-LAYER-004: cross-cutting(observability/resilience) → L3 concrete import 금지
    if src_layer in ("observability", "resilience"):
        if dst_layer in ("L1", *_L3_DOMAINS):
            kind = f"cross-cutting({src_layer})→{dst_layer} (trait inversion 필요)"
            return ("INV-LAYER-004", kind, None)

    # INV-LAYER-005: L5 bin → L4/session 외 direct import 금지 (generate.rs 한정)
    # generate.rs가 아닌 bin/ 파일은 enforcement 대상 외
    if src_layer == "L5":
        basename = os.path.basename(src_rel)
        name_no_ext = basename.replace(".rs", "")
        # skip 대상 binary인지 확인
        skip = any(name_no_ext.startswith(p) for p in L5_SKIP_PATTERNS)
        if not skip and basename == "generate.rs":
            # generate.rs → L1/L2/L3/observability/resilience 직접 import는 모두 위반
            if dst_layer in ("L1", "L2", *_L3_DOMAINS, "L3-core",
                             "observability", "resilience"):
                kind = f"L5/generate.rs→{dst_layer} (L4 session/ 우회)"
                return ("INV-LAYER-005", kind, None)

    return (None, None, None)


def _extract_backend(rel_path: str) -> str:
    """backend/<be>/... 에서 <be> 부분 추출. 해당 없으면 ''."""
    norm = rel_path.replace(os.sep, "/")
    if norm.startswith("backend/"):
        parts = norm.split("/")
        if len(parts) >= 2:
            return parts[1]
    return ""


# ────────────────────────────────────────────────────────────────────
# Cross-backend 위반 (V-04, V-05) 별도 처리
# ────────────────────────────────────────────────────────────────────

# ARCHITECTURE.md §13.8-K — Sub-layer dependency 허용 화이트리스트.
# (source_backend, target_backend) 페어. source가 target의 런타임 substrate
# (메모리/context owner)에 해당하는 경우만 등록 가능 — 일반 fallback은 제외.
ALLOWED_BACKEND_CHAINS: set[tuple[str, str]] = set()  # qnn_oppkg→opencl chain은 Sprint 2b (2026-05-26) 에서 제거됨


def check_cross_backend(src_rel: str, import_path: str) -> tuple[str | None, str | None]:
    """
    L1↔L1 cross-backend import를 별도로 검사.
    (inv_id, kind) 반환. 위반 없으면 (None, None).
    """
    # crate:: 제거 후 backend/ 시작 여부 확인
    p = import_path.strip()
    if p.startswith("crate::"):
        p = p[len("crate::"):]
    norm_import = p.replace("::", "/")

    if not norm_import.startswith("backend/"):
        return (None, None)

    src_be = _extract_backend(src_rel)
    parts = norm_import.split("/")
    if len(parts) < 2:
        return (None, None)
    dst_be = parts[1]

    if not src_be or not dst_be:
        return (None, None)

    # Step 4-A: parts[1]이 PascalCase면 backend.rs (top-level L2 trait file)
    # 의 item을 가리킴 (예: `crate::backend::Backend`, `crate::backend::GpuEvent`).
    # 이는 cross-backend import가 아니라 L2 trait import이므로 허용.
    if dst_be and dst_be[0].isupper():
        return (None, None)

    # 동일 backend 내부 (예: backend/cpu/x86.rs → backend/cpu/common.rs) 는 허용
    if src_be == dst_be:
        return (None, None)

    # §13.8-K sub-layer dependency 화이트리스트 — 허용 chain은 위반 미반환
    if (src_be, dst_be) in ALLOWED_BACKEND_CHAINS:
        return (None, None)

    # 다른 backend로의 cross-import
    kind = f"L1({src_be})→L1({dst_be}) (cross-backend import)"
    return ("INV-LAYER-001", kind)


# ────────────────────────────────────────────────────────────────────
# import 추출
# ────────────────────────────────────────────────────────────────────

# use crate:: 문 패턴
RE_USE_CRATE = re.compile(r'^\s*(?:pub\s+)?use\s+(crate::\S+)\s*;', re.MULTILINE)
# use llm_rs2:: 문 패턴 (bin/ 파일은 llm_rs2:: 형태로 import)
RE_USE_LIB = re.compile(r'^\s*(?:pub\s+)?use\s+(llm_rs2::\S+)\s*;', re.MULTILINE)
# 인라인 crate:: 참조 (함수 본문 내)
RE_INLINE_CRATE = re.compile(r'(?<!\w)(crate::[a-zA-Z_][a-zA-Z0-9_:]*)')

def extract_imports(file_path: str) -> list[tuple[int, str, bool, bool, bool]]:
    """
    파일에서 (line_number, import_path, is_test_block, in_exempt_zone, in_l_zone) 목록 반환.
    is_test_block=True는 #[cfg(test)] 블록 내부.
    in_exempt_zone=True는 §13.8-J LAYER-EXEMPT: dispatch_orchestrator zone 내부.
    in_l_zone=True는 §13.8-L LAYER-EXEMPT: backend_concrete_downcast zone 내부
            (marker 또는 EXT-anchored auto chain).
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    results = []
    lines = content.splitlines()

    # #[cfg(test)] 블록 범위 감지 (간단한 brace 카운팅)
    test_block_ranges = _find_test_block_ranges(lines)
    # §13.8-J dispatch orchestrator zone 범위 감지
    exempt_zone_ranges = _find_exempt_zone_ranges(lines, _EXEMPT_MARKER)
    # §13.8-L backend concrete downcast zone 범위 감지 (marker + EXT-anchored auto)
    l_marker_ranges = _find_exempt_zone_ranges(lines, _EXEMPT_MARKER_L)
    l_auto_ranges = _find_ext_anchor_ranges(lines)
    # §13.8-N cross-cutting trait/enum usage zone (cross-cutting ↔ L3 trait/enum import)
    n_marker_ranges = _find_exempt_zone_ranges(lines, _EXEMPT_MARKER_N)
    # §13.8-O cross-L3 vocabulary zone (L3↔L3 domain vocabulary: type alias default, public API surface)
    o_marker_ranges = _find_exempt_zone_ranges(lines, _EXEMPT_MARKER_O)
    # §13.8-P cross_backend_bootstrap zone (L1↔L1: cpu_companion / placeholder dependency)
    p_marker_ranges = _find_exempt_zone_ranges(lines, _EXEMPT_MARKER_P)
    l_zone_ranges = (
        l_marker_ranges + l_auto_ranges + n_marker_ranges + o_marker_ranges + p_marker_ranges
    )

    def in_test_block(lineno: int) -> bool:
        return any(s <= lineno <= e for s, e in test_block_ranges)

    def in_exempt_zone(lineno: int) -> bool:
        return any(s <= lineno <= e for s, e in exempt_zone_ranges)

    def in_l_zone(lineno: int) -> bool:
        return any(s <= lineno <= e for s, e in l_zone_ranges)

    # use crate:: 문 추출
    for i, line in enumerate(lines, 1):
        m = RE_USE_CRATE.match(line)
        if m:
            results.append((i, m.group(1), in_test_block(i), in_exempt_zone(i), in_l_zone(i)))

    # use llm_rs2:: 문 추출 (bin/ 파일용) — llm_rs2::foo → crate::foo로 정규화
    for i, line in enumerate(lines, 1):
        m = RE_USE_LIB.match(line)
        if m:
            # llm_rs2:: → crate:: 로 정규화하여 레이어 분류에 사용
            normalized = m.group(1).replace("llm_rs2::", "crate::", 1)
            results.append((i, normalized, in_test_block(i), in_exempt_zone(i), in_l_zone(i)))

    # 인라인 crate:: 참조 추출 (use 문이 아닌 것)
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # use 문은 이미 처리했으므로 제외
        if RE_USE_CRATE.match(line):
            continue
        for m in RE_INLINE_CRATE.finditer(line):
            imp = m.group(1)
            # 최소 2단계 이상의 경로만 (crate::foo 이상)
            if imp.count("::") >= 1:
                results.append((i, imp, in_test_block(i), in_exempt_zone(i), in_l_zone(i)))

    return results


def _find_test_block_ranges(lines: list[str]) -> list[tuple[int, int]]:
    """
    #[cfg(test)] mod tests { ... } 블록의 (start_line, end_line) 목록 반환.
    간단한 brace 카운팅으로 감지.
    """
    ranges = []
    in_test = False
    brace_depth = 0
    entered_block = False
    start = 0

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        if not in_test:
            if "#[cfg(test)]" in stripped or re.search(r'#\[test\]', stripped):
                in_test = True
                start = i
                brace_depth = 0
                entered_block = False

        if in_test:
            brace_depth += stripped.count("{") - stripped.count("}")
            if brace_depth > 0:
                entered_block = True
            # block에 한 번도 진입하지 않았으면 아직 attribute 행렬을 통과 중
            # (e.g. `#[cfg(test)] #[allow(...)] mod tests {`).
            # entered_block True인 상태에서만 종료 판정.
            if entered_block and brace_depth <= 0 and i > start:
                ranges.append((start, i))
                in_test = False
                brace_depth = 0
                entered_block = False

    return ranges


# ────────────────────────────────────────────────────────────────────
# §13.8-J dispatch orchestrator zone parser
# ARCHITECTURE.md §13.8-J: // LAYER-EXEMPT: dispatch_orchestrator marker로
# 표시된 함수/블록 내의 L3 정책 query 호출을 INV-LAYER-001 baseline에서 제외.
# ────────────────────────────────────────────────────────────────────

_EXEMPT_MARKER = "// LAYER-EXEMPT: dispatch_orchestrator"
_EXEMPT_MARKER_L = "// LAYER-EXEMPT: backend_concrete_downcast"
_EXEMPT_MARKER_N = "// LAYER-EXEMPT: cross_cutting_trait_usage"
_EXEMPT_MARKER_O = "// LAYER-EXEMPT: cross_l3_vocabulary"
_EXEMPT_MARKER_P = "// LAYER-EXEMPT: cross_backend_bootstrap"
_EXEMPT_END_MARKER = "// LAYER-EXEMPT-END"

# fn 시그니처 패턴: `fn name(...) -> ... {` 또는 `fn name(...) {`
_RE_FN_START = re.compile(r'^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+\w+')

# §13.8-L EXT-anchored chain 자동 인식 패턴
_RE_EXT_ANCHOR = re.compile(r'\.get_extension\(crate::backend::EXT_\w+\)')


def _find_exempt_zone_ranges(lines: list[str], marker: str = _EXEMPT_MARKER) -> list[tuple[int, int]]:
    """
    `// LAYER-EXEMPT: <kind>` marker가 표시된 zone의
    (start_line, end_line) 목록을 반환 (1-based 라인 번호, 양 끝 포함).

    두 가지 부착 형태 지원:
      1. **함수 형태**: marker가 `fn ...` 시그니처 *바로 위 줄*에 위치.
         zone = 함수 본문 시작 `{` 다음 줄 ~ 함수 본문 종료 `}` 줄.
      2. **블록 형태**: marker가 임의 `{` 줄의 *다음 줄*에 위치.
         zone = marker 줄 ~ `// LAYER-EXEMPT-END` 또는 블록 종료 `}` 직전 줄.

    알고리즘:
    - 라인 순 스캔으로 marker 위치를 감지.
    - marker 다음 줄이 `fn` 시그니처이면 함수 형태로 처리.
    - 그 외 marker가 `{` 다음 줄에 위치하면 블록 형태로 처리.
    - brace stack 카운팅으로 zone 종료 판정 (`// LAYER-EXEMPT-END`도 인식).
    - 문자열 리터럴 내 brace는 단순 카운팅으로 인식하지 않음
      (주석/문자열 완전 제거 없이 근사 처리 — 실무 코드에서 충분).
    """
    ranges: list[tuple[int, int]] = []
    n = len(lines)
    i = 0  # 0-based index

    while i < n:
        stripped = lines[i].strip()

        if marker not in stripped:
            i += 1
            continue

        marker_lineno = i + 1  # 1-based

        # marker 다음의 attribute / doc comment / 빈 줄을 스킵하여 target 코드 라인 찾기.
        next_i = i + 1
        code_i = next_i
        while code_i < n:
            stripped_c = lines[code_i].strip()
            if (stripped_c.startswith("#[") or stripped_c.startswith("///")
                    or stripped_c.startswith("//!") or stripped_c == ""):
                code_i += 1
                continue
            break

        # use 문 형태: 1줄 zone
        if code_i < n and (RE_USE_CRATE.match(lines[code_i]) or RE_USE_LIB.match(lines[code_i])):
            ranges.append((code_i + 1, code_i + 1))
            i = code_i + 1
            continue

        # 다음 줄이 fn 시그니처인지 확인 (함수 형태)
        next_i = code_i  # fn form 검사 시점도 attribute skip 이후로
        if next_i < n and _RE_FN_START.match(lines[next_i]):
            # 함수 형태: fn 시그니처 줄부터 (시그니처 안의 crate:: 참조도 포함) zone 시작.
            zone_start = next_i + 1  # 1-based, fn 시그니처 라인
            brace_depth = 0
            entered_block = False
            j = next_i
            while j < n:
                seg = lines[j].strip()
                brace_depth += seg.count("{") - seg.count("}")
                if brace_depth > 0:
                    entered_block = True
                if entered_block and brace_depth <= 0:
                    ranges.append((zone_start, j + 1))
                    i = j + 1
                    break
                j += 1
            else:
                i = next_i
            continue

        # 블록 형태: 현재 줄(marker) 바로 앞 줄에 `{`가 있는 경우
        # marker 자신이 zone의 첫 줄이 되어 `// LAYER-EXEMPT-END` 또는
        # 이전 brace depth 복귀 지점까지 zone으로 처리.
        # (marker가 `{` 다음 줄에 위치 — 즉 i-1 줄에 `{` 포함)
        prev_has_brace = (i > 0) and ("{" in lines[i - 1])
        if prev_has_brace:
            zone_start = marker_lineno  # marker 줄 자체가 zone 시작
            # `{`가 열린 depth 를 찾기 위해 이전 줄부터 depth를 정산
            # 간단화: brace_depth=1 (이전 줄의 `{`가 열었으므로)에서 시작
            brace_depth = 1
            j = i  # marker 줄 (0-based)
            while j < n:
                seg = lines[j].strip()
                if _EXEMPT_END_MARKER in seg:
                    ranges.append((zone_start, j + 1))
                    i = j + 1
                    break
                brace_depth += seg.count("{") - seg.count("}")
                if brace_depth <= 0:
                    # 블록 닫힘 — zone은 `}` 직전 줄까지
                    end = j  # `}` 줄 (1-based = j+1) 직전 = j (1-based)
                    ranges.append((zone_start, end))
                    i = j + 1
                    break
                j += 1
            else:
                i = next_i
            continue

        # fall-through: marker 다음 코드 라인 1줄을 zone으로 등록
        # (struct field / 일반 식별자 등 form 분기 없는 경우)
        if code_i < n:
            ranges.append((code_i + 1, code_i + 1))
            i = code_i + 1
            continue

        # 더 이상 코드 라인 없음
        i += 1

    return ranges


def _find_ext_anchor_ranges(lines: list[str], window: int = 5) -> list[tuple[int, int]]:
    """
    §13.8-L (L-auto) — `.get_extension(crate::backend::EXT_*)` chain 라인 ±window 윈도우를
    자동 화이트리스트 zone으로 등록. anchor 라인의 다음 N줄까지의 backend concrete
    downcast / instance import를 baseline 제외 대상으로 표시.
    """
    ranges: list[tuple[int, int]] = []
    n = len(lines)
    for i, line in enumerate(lines, 1):
        if _RE_EXT_ANCHOR.search(line):
            ranges.append((max(1, i - 1), min(n, i + window)))
    return ranges


# ────────────────────────────────────────────────────────────────────
# 위반 분석 메인
# ────────────────────────────────────────────────────────────────────

# V-번호 할당 — ARCHITECTURE.md §13.5 기준 알려진 위반에 ID 부여
# (file_path 패턴, import 패턴) → V-XX
KNOWN_V_MAP = [
    # V-01: backend/opencl/ → resilience::gpu_self_meter (trait import)
    (r"backend/opencl/",                r"resilience::gpu_self_meter",   "V-01"),
    # V-02: backend/opencl/plan.rs → layers::tensor_partition, layers::workspace
    (r"backend/opencl/plan\.rs",         r"layers::",                    "V-02"),
    # V-03/V-04: backend/qnn_oppkg/ 관련 패턴은 Sprint 2b (2026-05-26) 에서 production 제거됨
    # V-05: backend/cuda_*/ → backend::cpu::CpuBackend (cpu_fallback)
    (r"backend/cuda_(embedded|pc)/mod\.rs", r"backend::cpu::CpuBackend", "V-05"),
    # V-06: backend/cpu/x86.rs, neon.rs → cpu/common (동일 backend 내부 — 허용)
    (r"backend/cpu/(x86|neon)\.rs",      r"backend::cpu::common",       "V-06"),
    # V-07: buffer/host_ptr_pool_buffer.rs → backend::opencl (L2→L1)
    (r"buffer/host_ptr_pool_buffer\.rs", r"backend::opencl",            "V-07"),
    # V-08: buffer/cl_*/cuda_*/rpcmem_* (backend-specific buffer가 L2에 위치)
    (r"buffer/(cl_|cuda_|rpcmem_)",      r"",                           "V-08"),
    # V-09: buffer/ → models::weights::SecondaryMmap (L2→L3 pressure state)
    (r"buffer/",                         r"models::weights::SecondaryMmap", "V-09"),
    # V-10: pressure/cache_manager.rs → resilience::EvictMethod (Step 4-D path)
    (r"pressure/cache_manager\.rs",      r"resilience::EvictMethod",    "V-10"),
    # V-11: core/chat_template.rs → models::config::ModelArch
    (r"core/chat_template\.rs",          r"models::config::ModelArch",  "V-11"),
    # V-12: core/events.rs → pressure (의도된 의존), qcf:: (Step 4-B 후)
    (r"core/events\.rs",                 r"pressure::",                 "V-12"),
    (r"core/events\.rs",                 r"qcf::",                      "V-12"),
    # V-13: pressure/kivi_cache.rs → backend::cpu/opencl (L3→L1), qcf:: (Step 4-B path)
    (r"pressure/kivi_cache\.rs",         r"backend::",                  "V-13"),
    (r"pressure/kivi_cache\.rs",         r"qcf::",                      "V-13"),
    # V-13(b): pressure/mod.rs → qcf:: (L3-pressure→L3-inference)
    (r"pressure/mod\.rs",                r"qcf::",                      "V-13"),
    # V-14: qcf/, pressure/kivi_cache, inference/sampling → profile:: (L3→observability concrete)
    (r"(qcf/(unified_qcf|layer_importance|qcf_kv)|pressure/kivi_cache|inference/sampling)", r"profile::", "V-14"),
    # V-15: pressure/cache_manager.rs, pressure/eviction/* (테스트 블록) → backend::cpu (grandfathered)
    (r"pressure/(cache_manager|eviction/)", r"backend::cpu::CpuBackend", "V-15"),
    # V-16: eval/eval_loop.rs → backend:: (cross-cutting→L1)
    (r"eval/eval_loop\.rs",              r"backend::",                  "V-16"),
    # V-17: layers/ → backend::cpu::neon, opencl (L3→L1 downcast/direct call)
    (r"layers/(transformer_layer|attention|workspace)", r"backend::",    "V-17"),
    # V-18: layers/transformer_layer/ → memory::galloc, profile:: (L3→cross-cutting)
    (r"layers/transformer_layer/",       r"(memory::galloc|profile::)", "V-18"),
    # V-19: layers/tensor_partition.rs → buffer::slice_buffer/cl_sub_buffer
    (r"layers/tensor_partition\.rs",     r"buffer::(slice_buffer|cl_sub_buffer)", "V-19"),
    # V-20: models/transformer.rs → backend::opencl (L3→L1)
    (r"models/transformer\.rs",          r"backend::(opencl|cuda)",     "V-20"),
    # V-21: models/transformer.rs → pressure::offload::preload_pool (Step 4-D path)
    (r"models/transformer\.rs",          r"pressure::offload::preload_pool", "V-21"),
    # V-22: models/transformer.rs, layers/ → profile:: (L3→observability)
    (r"models/transformer\.rs",          r"profile::",                  "V-22"),
    # V-23: models/transformer.rs, models/weights/ → auf:: (→shared/auf/ 이동 전 L3→cross-cutting)
    (r"models/(transformer|weights/)",   r"auf::",                      "V-23"),
    # V-24: pressure/weight_swap_handler.rs → models:: (Pressure→Inference cross, Step 4-D path)
    (r"pressure/weight_swap_handler\.rs", r"models::",             "V-24"),
    # V-24(b): pressure/weight_swap_handler.rs → backend::cpu::CpuBackend
    (r"pressure/weight_swap_handler\.rs", r"backend::cpu::CpuBackend", "V-24"),
    # V-24(c): pressure/weight_swap_handler.rs → memory::galloc
    (r"pressure/weight_swap_handler\.rs", r"memory::galloc",       "V-24"),
    # V-25: models/weights/swap_executor.rs → layers::transformer_layer (L3-pressure→L3-inference concrete)
    (r"models/weights/(swap_executor|intra_forward_swap|phase_aware_swap)", r"layers::", "V-25"),
    # V-25(b): models/weights/swap_executor.rs → models::transformer (self-domain monolith)
    (r"models/weights/swap_executor\.rs", r"models::transformer",       "V-25"),
    # V-25(c): models/weights/swap_executor.rs → backend::opencl::host_ptr_pool
    (r"models/weights/swap_executor\.rs", r"backend::opencl::host_ptr_pool", "V-25"),
    # V-25(d): models/weights/swap_executor.rs → profile::
    (r"models/weights/swap_executor\.rs", r"profile::",                 "V-25"),
    # V-26: models/weights/decider.rs → qcf::layer_importance (현재 구조 내 cross)
    (r"models/weights/decider\.rs",      r"qcf::layer_importance",      "V-26"),
    # V-26(b): models/weights/decider.rs → profile:: (L3→observability concrete)
    (r"models/weights/decider\.rs",      r"profile::",                  "V-26"),
    # V-27: models/weights/layer_object_pool.rs → buffer::cuda_buffer (L3→L2 backend-specific)
    (r"models/weights/layer_object_pool\.rs", r"buffer::cuda_buffer",   "V-27"),
    # V-27(b): models/weights/layer_object_pool.rs → layers::transformer_layer
    (r"models/weights/layer_object_pool\.rs", r"layers::transformer_layer", "V-27"),
    # V-27(c): models/weights/layer_object_pool.rs → backend::cuda_embedded (downcast)
    (r"models/weights/layer_object_pool\.rs", r"backend::cuda_embedded","V-27"),
    # V-28: eval/ → models::, pressure::, qcf::, inference:: (cross-cutting→L3 다수)
    (r"eval/(qcf_helpers|eval_loop|eviction_hook|kivi_hook|hook)", r"models::",        "V-28"),
    (r"eval/(eval_loop|eviction_hook|kivi_hook|hook)",  r"(pressure::(cache_manager|kv_cache|kivi_cache)|qcf::)", "V-28"),
    (r"eval/(qcf_helpers|kivi_hook)",    r"qcf::",                       "V-28"),
    # Step 4-C: eval/ → inference:: (sampling/skip_config/attention_scores 이동 후)
    (r"eval/(eval_loop|eviction_hook|kivi_hook|hook)", r"inference::",  "V-28"),
    # Step 4-D: eval/ → pressure:: (cache_manager/kv_cache/kivi_cache/offload promoted)
    (r"eval/(eval_loop|eviction_hook|kivi_hook|hook|qcf_helpers)", r"pressure::", "V-28"),
    # V-29: eval/eviction_hook.rs → backend::opencl::OpenCLBackend (cross-cutting→L1 downcast)
    (r"eval/eviction_hook\.rs",          r"backend::opencl::OpenCLBackend", "V-29"),
    # V-30: bin/generate.rs → 모든 레이어 직접 import (L5 monolith)
    (r"bin/generate\.rs",                r"",                           "V-30"),
    # V-31: (V-21, V-10 재기재 — 이미 V-21, V-10으로 처리됨)
]


def lookup_v_id(rel_path: str, import_path: str) -> str:
    """
    알려진 V-XX ID를 반환. 매칭 없으면 "V-??" 반환.
    """
    norm_rel = rel_path.replace(os.sep, "/")
    for file_pat, import_pat, v_id in KNOWN_V_MAP:
        if not re.search(file_pat, norm_rel):
            continue
        if import_pat == "" or re.search(import_pat, import_path):
            return v_id
    return "V-??"


def analyze(src_root: str, inv_filter: str | None) -> list[dict]:
    """
    engine/src/ 하위 모든 .rs 파일을 분석하여 위반 목록 반환.
    """
    violations = []
    seen = set()  # (file, line, import) 중복 제거

    for dirpath, _, filenames in os.walk(src_root):
        for fname in filenames:
            if not fname.endswith(".rs"):
                continue
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, src_root)
            rel_path = rel_path.replace(os.sep, "/")

            src_layer = classify_module(rel_path)
            imports = extract_imports(full_path)

            for lineno, imp, is_test, is_exempt, in_l in imports:
                # imp에서 정규화된 경로 추출
                imp_clean = imp.strip()
                if imp_clean.startswith("crate::"):
                    dst_raw = imp_clean[len("crate::"):]
                else:
                    dst_raw = imp_clean
                dst_layer = classify_import(imp_clean)

                # §13.8-J: dispatch_orchestrator zone 안의 L3 import는
                # INV-LAYER-001 위반에서 제외 (정책 query 함수 호출 한정).
                # zone 안이면 cross-backend 검사도 건너뜀 (L1→L1 는 zone 범위 밖).
                if is_exempt and dst_layer in _L3_DOMAINS:
                    continue

                # §13.8-L: backend_concrete_downcast zone (marker 또는 EXT-anchored
                # auto chain) 안의 import는 INV-LAYER-003 전체에서 제외.
                # 적용 범위: L3→L1 backend impl + cross-L3 default initialization
                # (예: pressure가 qcf concrete unit struct을 default field로 보유).
                # 본 정책은 marker 단위로 적용되므로 별 sprint(S-C3) trait inversion
                # 대상과 충돌하지 않음 (의도성 명시 위치만 제외).
                if in_l and src_layer in _L3_DOMAINS \
                        and (dst_layer == "L1" or dst_layer in _L3_DOMAINS):
                    continue

                # §13.8-N: cross_cutting_trait_usage zone — L3 → cross-cutting
                # (INV-LAYER-003) 및 cross-cutting → L3 (INV-LAYER-004) 양방향 trait/
                # enum usage 의도성 명시. trait inversion이 이미 적용된 곳 또는
                # §13.8-F enum-as-data identifier 예외에 해당하는 경우 사용.
                # (S-D2 확장) L1 backend impl 이 cross-cutting trait 을 impl-only 로
                # import 하는 경우(예: `GpuSelfMeter`)도 동일 marker 허용.
                if in_l and (
                    (src_layer in _L3_DOMAINS and dst_layer in ("observability", "resilience"))
                    or (src_layer in ("observability", "resilience") and dst_layer in _L3_DOMAINS)
                    or (src_layer == "L1" and dst_layer in ("observability", "resilience"))
                ):
                    continue

                # §13.8-P: cross_backend_bootstrap zone — L1 backend impl이 다른
                # L1 backend의 singleton/constructor를 cpu_companion 또는 placeholder
                # dependency용으로 import. INV-LAYER-001 cross-backend 위반 한정 우회.
                if in_l and src_layer == "L1" and dst_layer == "L1":
                    continue

                # cross-backend 검사
                cb_inv, cb_kind = check_cross_backend(rel_path, imp_clean)
                if cb_inv:
                    if inv_filter and cb_inv.lower() != inv_filter.lower():
                        pass
                    else:
                        key = (rel_path, lineno, imp_clean)
                        if key not in seen:
                            seen.add(key)
                            v_id = lookup_v_id(rel_path, imp_clean)
                            violations.append({
                                "id": v_id,
                                "file": rel_path,
                                "line": lineno,
                                "import": imp_clean,
                                "rule": cb_inv,
                                "kind": cb_kind,
                                "is_test_block": is_test,
                            })
                    continue

                # 일반 위반 검사
                inv_id, kind, note = check_violation(src_layer, dst_layer, rel_path)
                if not inv_id:
                    continue
                if inv_filter and inv_id.lower() != inv_filter.lower():
                    continue

                key = (rel_path, lineno, imp_clean)
                if key in seen:
                    continue
                seen.add(key)

                # Data consumer 카테고리: L1 backend가 L3 weight struct/enum을
                # 데이터 소비자로 import하는 경우 INV-LAYER-001 baseline에서 제외.
                # (spec/41-invariants.md INV-LAYER-001 비고 참조)
                if inv_id == "INV-LAYER-001" and is_data_consumer(imp_clean):
                    continue

                # §13.8-E (S-C2b, 2026-05-24): test block grandfathered exception.
                # `#[cfg(test)]` 또는 `#[test]` 블록 안의 backend/cross-domain import는
                # 자동 baseline 제외. INV-LAYER-001/002/003에 공통 적용.
                if is_test and inv_id in ("INV-LAYER-001", "INV-LAYER-002", "INV-LAYER-003"):
                    continue

                v_id = lookup_v_id(rel_path, imp_clean)
                entry = {
                    "id": v_id,
                    "file": rel_path,
                    "line": lineno,
                    "import": imp_clean,
                    "rule": inv_id,
                    "kind": kind,
                    "is_test_block": is_test,
                }
                if note:
                    entry["note"] = note
                violations.append(entry)

    # V-ID 순 정렬
    def sort_key(v):
        vid = v.get("id", "V-??")
        try:
            n = int(vid.split("-")[1])
        except Exception:
            n = 9999
        return (n, v["file"], v["line"])

    violations.sort(key=sort_key)
    return violations


# ────────────────────────────────────────────────────────────────────
# baseline diff
# ────────────────────────────────────────────────────────────────────

def load_baseline(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("violations", [])


def diff_violations(current: list[dict], baseline: list[dict]) -> list[dict]:
    """
    baseline에 없는 새 위반만 반환 (회귀 감지).
    매칭 기준: (file, import, rule) 튜플.
    """
    baseline_keys = set()
    for v in baseline:
        key = (v.get("file", ""), v.get("import", ""), v.get("rule", ""))
        baseline_keys.add(key)

    new_violations = []
    for v in current:
        key = (v.get("file", ""), v.get("import", ""), v.get("rule", ""))
        if key not in baseline_keys:
            new_violations.append(v)
    return new_violations


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Engine 레이어드 아키텍처 위반 검출")
    parser.add_argument(
        "--src",
        default=None,
        help="engine/src/ 경로 (기본: 스크립트 위치에서 자동 탐색)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="baseline JSON 경로. 지정 시 새로 발견된 위반만 출력",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="특정 INV만 필터링 (예: inv-layer-001)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="JSON 출력 (기본)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=False,
        help="요약 출력 (violations만 아닌 통계 포함)",
    )
    args = parser.parse_args()

    # src 경로 결정
    if args.src:
        src_root = args.src
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # scripts/ → 프로젝트 루트 → engine/src
        project_root = os.path.dirname(script_dir)
        src_root = os.path.join(project_root, "engine", "src")

    if not os.path.isdir(src_root):
        print(f"오류: engine/src 디렉토리를 찾을 수 없음: {src_root}", file=sys.stderr)
        sys.exit(1)

    violations = analyze(src_root, args.filter)

    if args.baseline:
        baseline = load_baseline(args.baseline)
        violations = diff_violations(violations, baseline)

    result = {"violations": violations}

    if args.summary:
        print(f"총 위반: {len(violations)}건", file=sys.stderr)
        by_rule = {}
        for v in violations:
            r = v.get("rule", "?")
            by_rule[r] = by_rule.get(r, 0) + 1
        for r, cnt in sorted(by_rule.items()):
            print(f"  {r}: {cnt}건", file=sys.stderr)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
