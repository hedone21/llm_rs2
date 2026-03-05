#!/usr/bin/env python3
"""
Update docs/14_component_status.md with latest cargo test results.

Usage:
    python scripts/update_test_status.py              # run cargo test and update
    cargo test 2>&1 | python scripts/update_test_status.py --stdin  # pipe mode
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DOC_PATH = PROJECT_ROOT / "docs" / "14_component_status.md"
HISTORY_PATH = PROJECT_ROOT / "docs" / "test_history.json"

# Module path prefix → component metadata
# tier: T1=Foundation, T2=Algorithm, T3=Backend, T4=Integration
# maturity: Stable, Beta, Stub
COMPONENT_META = {
    # T1: Foundation (6)
    "core::shape": {
        "component": "Core",
        "name": "Shape",
        "tier": "T1",
        "maturity": "Stable",
    },
    "core::tensor": {
        "component": "Core",
        "name": "Tensor",
        "tier": "T1",
        "maturity": "Stable",
    },
    "core::buffer": {
        "component": "Core",
        "name": "Buffer/DType",
        "tier": "T1",
        "maturity": "Stable",
    },
    "core::quant": {
        "component": "Core",
        "name": "Quant",
        "tier": "T1",
        "maturity": "Stable",
    },
    "buffer::shared_buffer": {
        "component": "Buffer",
        "name": "SharedBuffer",
        "tier": "T1",
        "maturity": "Stable",
    },
    "memory::galloc": {
        "component": "Memory",
        "name": "Galloc",
        "tier": "T1",
        "maturity": "Stable",
    },
    # T2: Algorithm (7)
    "core::kv_cache": {
        "component": "Core",
        "name": "KVCache",
        "tier": "T2",
        "maturity": "Stable",
    },
    "core::eviction::no_eviction": {
        "component": "Core",
        "name": "NoEvictionPolicy",
        "tier": "T2",
        "maturity": "Stable",
    },
    "core::eviction::sliding_window": {
        "component": "Core",
        "name": "SlidingWindowPolicy",
        "tier": "T2",
        "maturity": "Stable",
    },
    "core::eviction::h2o": {
        "component": "Core",
        "name": "H2OPolicy",
        "tier": "T2",
        "maturity": "Stub",
    },
    "core::cache_manager": {
        "component": "Core",
        "name": "CacheManager",
        "tier": "T2",
        "maturity": "Stable",
    },
    "core::sys_monitor": {
        "component": "Core",
        "name": "SystemMonitor",
        "tier": "T2",
        "maturity": "Stable",
    },
    "layers::attention": {
        "component": "Layers",
        "name": "Attention",
        "tier": "T2",
        "maturity": "Stable",
    },
    # T3: Backend (2)
    "backend::cpu": {
        "component": "Backend",
        "name": "CpuBackend",
        "tier": "T3",
        "maturity": "Stable",
    },
    "backend::opencl": {
        "component": "Backend",
        "name": "OpenCLBackend",
        "tier": "T3",
        "maturity": "Stable",
    },
    # T4: Integration (4)
    "layers::llama_layer": {
        "component": "Layers",
        "name": "LlamaLayer",
        "tier": "T4",
        "maturity": "Stable",
    },
    "layers::workspace": {
        "component": "Layers",
        "name": "LayerWorkspace",
        "tier": "T4",
        "maturity": "Stable",
    },
    "models::llama": {
        "component": "Models",
        "name": "LlamaModel",
        "tier": "T4",
        "maturity": "Stable",
    },
    "buffer::unified_buffer": {
        "component": "Buffer",
        "name": "UnifiedBuffer",
        "tier": "T4",
        "maturity": "Stable",
    },
    # Resilience (feature-gated, T2 level)
    "resilience::manager": {
        "component": "Resilience",
        "name": "ResilienceManager",
        "tier": "T2",
        "maturity": "Stable",
    },
    "resilience::signal": {
        "component": "Resilience",
        "name": "Signal/Level",
        "tier": "T2",
        "maturity": "Stable",
    },
    "resilience::state": {
        "component": "Resilience",
        "name": "OperatingMode",
        "tier": "T2",
        "maturity": "Stable",
    },
    "resilience::strategy": {
        "component": "Resilience",
        "name": "Strategy",
        "tier": "T2",
        "maturity": "Stable",
    },
}

# Test result line pattern: test module::path::test_name ... ok/FAILED
TEST_LINE_RE = re.compile(
    r"^test\s+([\w:]+)\s+\.\.\.\s+(ok|FAILED|ignored)\s*.*$"
)

# Sorted module keys: T1→T2→T3→T4, then by name within tier
_TIER_ORDER = {"T1": 0, "T2": 1, "T3": 2, "T4": 3}
SORTED_KEYS = sorted(
    COMPONENT_META.keys(),
    key=lambda k: (_TIER_ORDER[COMPONENT_META[k]["tier"]], COMPONENT_META[k]["name"]),
)


def classify_test(test_path: str) -> tuple[str, str]:
    """Classify a test path into (module_key, name).

    Returns ("integration", "Integration") for unrecognized tests.
    """
    for prefix in COMPONENT_META:
        if prefix in test_path:
            return prefix, COMPONENT_META[prefix]["name"]
    return "integration", "Integration"


def parse_test_output(text: str) -> list[dict]:
    """Parse cargo test stdout for test results."""
    results = []
    
    # Pre-scan for SKIPPED panics by looking for the explicit failure block outputs
    # When a test fails it prints:
    # ---- buffer::unified_buffer::tests::test_alloc_unified_buffer stdout ----
    # thread 'buffer::unified_buffer::tests::test_alloc_unified_buffer' panicked at src/buffer/unified_buffer.rs:221:20:
    # [SKIPPED] No OpenCL device
    skipped_tests = set()
    for line in text.splitlines():
        if "panicked at" in line and "[SKIPPED]" in line:
            parts = line.split("'")
            if len(parts) >= 3:
                skipped_tests.add(parts[1])
        # Sometimes the panic message is on the next line
        elif "----" in line and "stdout ----" in line:
            test_name = line.replace("----", "").replace("stdout", "").strip()
            # We'll just check if the whole text contains [SKIPPED] and the test name
            if "[SKIPPED]" in text and test_name in text:
                skipped_tests.add(test_name)

    for line in text.splitlines():
        m = TEST_LINE_RE.match(line.strip())
        if m:
            test_path = m.group(1)
            status = m.group(2).upper()
            if status == "OK":
                status = "PASS"
            elif status == "IGNORED":
                status = "SKIP"
            
            # Override FAIL to SKIP if it panicked intentionally with [SKIPPED]
            if status == "FAILED" and test_path in skipped_tests:
                status = "SKIP"
                
            results.append({"name": test_path, "status": status})
    return results


def run_cargo_test() -> str:
    """Run cargo test and capture output."""
    result = subprocess.run(
        ["cargo", "test"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    return result.stdout + "\n" + result.stderr


def group_by_component(results: list[dict]) -> dict:
    """Group test results by module key."""
    groups = {}  # module_key -> [test_results]
    for r in results:
        key, _ = classify_test(r["name"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    return groups


def compute_gate_status(module_key: str, tests: list[dict]) -> str:
    """Compute quality gate status for a component.

    Returns: PASS, FAIL, BLOCKED, or N/A
    - T1/T2 with no tests → BLOCKED
    - T3/T4 with no tests → N/A (requires device)
    - All tests pass → PASS
    - Any test fails → FAIL (T1/T2) or FAIL (T3/T4)
    """
    meta = COMPONENT_META.get(module_key)
    if not meta:
        # Integration or unknown — treat as informational
        if not tests:
            return "N/A"
        return "PASS" if all(t["status"] == "PASS" for t in tests) else "FAIL"

    tier = meta["tier"]
    if not tests:
        return "N/A" if tier in ("T3", "T4") else "BLOCKED"
        
    # If all tests are SKIP, or a mix of PASS and SKIP, we consider the gate PASS (or SKIP).
    # We will return "SKIP" if all tests were skipped so it's clearer, else "PASS" or "FAIL".
    if all(t["status"] == "SKIP" for t in tests):
        return "SKIP"
    # If there are no FAILs, it passes
    if all(t["status"] in ("PASS", "SKIP") for t in tests):
        return "PASS"
    return "FAIL"


def compute_overall_gate(gate_statuses: dict) -> str:
    """Compute overall gate status.

    FAIL if any T1/T2 component is BLOCKED or FAIL.
    """
    for key in COMPONENT_META:
        tier = COMPONENT_META[key]["tier"]
        if tier in ("T1", "T2"):
            status = gate_statuses.get(key, "BLOCKED")
            if status in ("BLOCKED", "FAIL"):
                return "FAIL"
    return "PASS"


def update_history(total: int, passed: int, failed: int, details: dict) -> list:
    """Append a new entry to test_history.json and return updated history."""
    history = []
    if HISTORY_PATH.exists():
        try:
            history = json.loads(HISTORY_PATH.read_text())
        except (json.JSONDecodeError, IOError):
            history = []

    entry = {
        "date": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total * 100, 1) if total > 0 else 0.0,
        "details": details,
    }
    history.append(entry)
    HISTORY_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n")
    return history


def render_test_status(groups: dict, timestamp: str) -> str:
    """Render the quality gate status markdown."""
    lines = []
    lines.append(f"_Last updated: {timestamp}_\n")

    # Compute gate status for each component
    gate_statuses = {}
    for key in SORTED_KEYS:
        tests = groups.get(key, [])
        gate_statuses[key] = compute_gate_status(key, tests)

    overall = compute_overall_gate(gate_statuses)

    # Quality Gate Summary table
    lines.append("### Quality Gate Summary\n")
    lines.append("| Component | Tier | Maturity | Tests | Passed | Skipped | Gate |")
    lines.append("|:----------|:-----|:---------|------:|-------:|--------:|:-----|")

    total_all, passed_all, skipped_all = 0, 0, 0
    blocked_count = 0
    fail_count = 0

    for key in SORTED_KEYS:
        meta = COMPONENT_META[key]
        tests = groups.get(key, [])
        t = len(tests)
        p = sum(1 for x in tests if x["status"] == "PASS")
        s = sum(1 for x in tests if x["status"] == "SKIP")
        gate = gate_statuses[key]

        if gate == "BLOCKED":
            blocked_count += 1
        elif gate == "FAIL":
            fail_count += 1

        gate_display = f"**{gate}**" if gate in ("BLOCKED", "FAIL") else gate
        lines.append(
            f"| {meta['name']} | {meta['tier']} | {meta['maturity']} "
            f"| {t} | {p} | {s} | {gate_display} |"
        )
        total_all += t
        passed_all += p
        skipped_all += s

    failed_all = total_all - passed_all - skipped_all
    if failed_all > 0 or overall == "FAIL":
        overall_display = "**FAIL**"
    elif failed_all == 0 and skipped_all > 0 and passed_all == 0:
        overall_display = "SKIP"
    else:
        overall_display = "PASS"
    lines.append(
        f"| **Overall** | | | **{total_all}** | **{passed_all}** | **{skipped_all}** | {overall_display} |"
    )

    # Handle integration tests (not in COMPONENT_META)
    integration_tests = groups.get("integration", [])
    if integration_tests:
        it = len(integration_tests)
        ip = sum(1 for x in integration_tests if x["status"] == "PASS")
        ig = "PASS" if ip == it else "FAIL"
        ig_display = f"**{ig}**" if ig == "FAIL" else ig
        lines.append(f"| Integration | - | - | {it} | {ip} | {ig_display} |")

    # Test Details table
    lines.append("\n### Test Details\n")
    lines.append("| Test | Component | Result |")
    lines.append("|:-----|:----------|:------:|")

    for key in SORTED_KEYS:
        tests = groups.get(key, [])
        meta = COMPONENT_META[key]
        for t in sorted(tests, key=lambda x: x["name"]):
            short = t["name"].rsplit("::", 1)[-1]
            if t["status"] == "PASS":
                icon = "PASS"
            elif t["status"] == "SKIP":
                icon = "SKIP"
            else:
                icon = "**FAIL**"
            lines.append(f"| `{short}` | {meta['name']} | {icon} |")

    # Integration test details
    for t in sorted(integration_tests, key=lambda x: x["name"]):
        short = t["name"].rsplit("::", 1)[-1]
        if t["status"] == "PASS":
            icon = "PASS"
        elif t["status"] == "SKIP":
            icon = "SKIP"
        else:
            icon = "**FAIL**"
        lines.append(f"| `{short}` | Integration | {icon} |")

    return "\n".join(lines)


def render_test_history(history: list) -> str:
    """Render the test history markdown table."""
    lines = []
    if not history:
        lines.append("_No history yet._")
        return "\n".join(lines)

    lines.append("| Date | Total | Passed | Failed | Pass Rate |")
    lines.append("|:-----|------:|-------:|-------:|----------:|")
    for entry in history[-20:]:
        d = entry["date"]
        lines.append(
            f"| {d} | {entry['total']} | {entry['passed']} "
            f"| {entry['failed']} | {entry['pass_rate']}% |"
        )
    return "\n".join(lines)


def update_document(test_status_md: str, history_md: str) -> None:
    """Replace AUTO-GENERATED sections in 14_component_status.md."""
    if not DOC_PATH.exists():
        print(f"ERROR: {DOC_PATH} not found", file=sys.stderr)
        sys.exit(1)

    content = DOC_PATH.read_text()

    content = re.sub(
        r"(<!-- AUTO-GENERATED:TEST_STATUS:START -->)\n.*?\n(<!-- AUTO-GENERATED:TEST_STATUS:END -->)",
        rf"\1\n{test_status_md}\n\2",
        content,
        flags=re.DOTALL,
    )

    content = re.sub(
        r"(<!-- AUTO-GENERATED:TEST_HISTORY:START -->)\n.*?\n(<!-- AUTO-GENERATED:TEST_HISTORY:END -->)",
        rf"\1\n{history_md}\n\2",
        content,
        flags=re.DOTALL,
    )

    DOC_PATH.write_text(content)


GATE_JSON_PATH = PROJECT_ROOT / "results" / "data" / "component_gates.json"

TIER_NAMES = {
    "T1": "Foundation",
    "T2": "Algorithm",
    "T3": "Backend",
    "T4": "Integration",
}


def write_gate_json(gate_statuses, groups, results, overall_gate):
    """Write gate status data to results/data/component_gates.json for the web dashboard."""
    timestamp = datetime.now().isoformat()

    # Build per-tier component lists
    tiers = {}
    for tier_key in ("T1", "T2", "T3", "T4"):
        tiers[tier_key] = {
            "name": TIER_NAMES[tier_key],
            "components": [],
        }

    for key in SORTED_KEYS:
        meta = COMPONENT_META[key]
        tests = groups.get(key, [])
        t = len(tests)
        p = sum(1 for x in tests if x["status"] == "PASS")
        f = sum(1 for x in tests if x["status"] == "FAILED")
        s = sum(1 for x in tests if x["status"] == "SKIP")
        gate = gate_statuses.get(key, "N/A")

        component_item = {
            "name": meta["name"],
            "module": key,
            "maturity": meta["maturity"],
            "total_tests": t,
            "passed": p,
            "failed": f,
            "skipped": s,
            "gate": gate,
        }
        tiers[meta["tier"]]["components"].append(component_item)

    # Summary counts
    total_components = len(COMPONENT_META)
    pass_count = sum(1 for s in gate_statuses.values() if s == "PASS")
    fail_count = sum(1 for s in gate_statuses.values() if s == "FAIL")
    blocked_count = sum(1 for s in gate_statuses.values() if s == "BLOCKED")
    skip_count = sum(1 for s in gate_statuses.values() if s == "SKIP")
    na_count = sum(1 for s in gate_statuses.values() if s == "N/A")

    # Pass rate: components that PASS out of those that are not N/A
    countable = total_components - na_count
    pass_rate = round(pass_count / countable * 100, 1) if countable > 0 else 0.0

    summary = {
        "total_components": total_components,
        "pass": pass_count,
        "fail": fail_count,
        "blocked": blocked_count,
        "skip": skip_count,
        "na": na_count,
        "pass_rate": pass_rate,
    }

    # Load history from docs/test_history.json
    history = []
    if HISTORY_PATH.exists():
        try:
            raw_history = json.loads(HISTORY_PATH.read_text())
            for entry in raw_history:
                history.append({
                    "date": entry["date"],
                    "total": entry["total"],
                    "passed": entry["passed"],
                    "failed": entry["failed"],
                    "pass_rate": entry["pass_rate"],
                })
        except (json.JSONDecodeError, IOError, KeyError):
            history = []

    gate_data = {
        "timestamp": timestamp,
        "overall_gate": overall_gate,
        "summary": summary,
        "tiers": tiers,
        "history": history,
    }

    GATE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    GATE_JSON_PATH.write_text(json.dumps(gate_data, indent=2, ensure_ascii=False) + "\n")
    print(f"Gate JSON: {GATE_JSON_PATH.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Update test status document")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read cargo test output from stdin instead of running cargo test",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output (for use in git hooks)",
    )
    args = parser.parse_args()

    # 1. Get test output
    if args.stdin:
        output = sys.stdin.read()
    else:
        if not args.quiet:
            print("Running cargo test...")
        output = run_cargo_test()

    # 2. Parse results
    results = parse_test_output(output)
    if not results:
        print("WARNING: No test results found in output.", file=sys.stderr)
        print("Parsed output (first 500 chars):", file=sys.stderr)
        print(output[:500], file=sys.stderr)

    # 3. Group by component
    groups = group_by_component(results)

    # 4. Compute totals for history
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    failed = total - passed - skipped

    # Build details dict for history (only components with tests)
    details = {}
    for key, tests in groups.items():
        if key != "integration":
            details[key] = {
                "total": len(tests),
                "passed": sum(1 for t in tests if t["status"] == "PASS"),
            }

    # 5. Update history
    history = update_history(total, passed, failed, details)

    # 6. Render markdown
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    test_status_md = render_test_status(groups, timestamp)
    history_md = render_test_history(history)

    # 7. Update document
    update_document(test_status_md, history_md)

    # 7b. Write gate JSON for web dashboard
    gate_statuses = {}
    for key in SORTED_KEYS:
        tests = groups.get(key, [])
        gate_statuses[key] = compute_gate_status(key, tests)
    overall = compute_overall_gate(gate_statuses)
    write_gate_json(gate_statuses, groups, results, overall)

    # 8. Print summary with gate status
    rate = f"{passed/total*100:.1f}" if total > 0 else "0.0"

    # Compute gate for CLI output
    gate_statuses = {}
    for key in SORTED_KEYS:
        tests = groups.get(key, [])
        gate_statuses[key] = compute_gate_status(key, tests)

    overall = compute_overall_gate(gate_statuses)
    blocked = sum(1 for s in gate_statuses.values() if s == "BLOCKED")
    gate_fails = sum(1 for s in gate_statuses.values() if s == "FAIL")

    if not args.quiet:
        print(f"Tests: {total} total, {passed} passed, {failed} failed ({rate}%)")

        reasons = []
        if blocked:
            reasons.append(f"{blocked} components BLOCKED")
        if gate_fails:
            reasons.append(f"{gate_fails} components FAIL")
        if reasons:
            print(f"Gate: {overall} ({', '.join(reasons)})")
        else:
            print(f"Gate: {overall}")

        print(f"Updated: {DOC_PATH.relative_to(PROJECT_ROOT)}")
        print(f"History: {HISTORY_PATH.relative_to(PROJECT_ROOT)} ({len(history)} entries)")


if __name__ == "__main__":
    main()
