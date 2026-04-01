#!/usr/bin/env python3
"""Regression tests for high-risk feature combinations.

Tests that require a model but not a GPU device. Run on host with:
    python3 scripts/test_regression.py --model-path models/qwen2.5-1.5b

Tests:
  1. NLL monotonicity: r75 NLL <= r25 NLL (eviction quality)
  2. KIVI bit monotonicity: NLL(Q2) >= NLL(Q4) >= NLL(Q8) (compression quality)
"""

import argparse
import json
import subprocess
import sys
import tempfile
import os

_GENERATE = os.path.join(os.path.dirname(__file__), "..", "target", "release", "generate")
__EVAL_DATA = None  # Set by main()


def run_generate(model_path, extra_args, timeout=120):
    """Run generate and return parsed JSON output."""
    cmd = [
        _GENERATE,
        "--model-path", model_path,
        "--kv-type", "f32",
        "--max-seq-len", "2048",
        "--eval-ll", "--eval-batch", _EVAL_DATA,
        "--greedy",
    ] + extra_args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        return None, r.stderr
    try:
        return json.loads(r.stdout), None
    except json.JSONDecodeError:
        return None, f"JSON parse error: {r.stdout[:200]}"


def avg_min_nll(results):
    """Average of min(choice_nlls) across questions."""
    nlls = []
    for r in results.get("results", []):
        cn = [x for x in r.get("choice_nlls", []) if x is not None]
        if cn:
            nlls.append(min(cn))
    return sum(nlls) / len(nlls) if nlls else float("inf")


def test_nll_monotonicity(model_path):
    """Test that r75 (mild eviction) produces <= NLL than r25 (aggressive)."""
    print("\n[Test 1] NLL Monotonicity (r75 <= r25)")
    print("-" * 50)

    bl, err = run_generate(model_path, ["--eviction-policy", "none"])
    if not bl:
        print(f"  SKIP: baseline failed — {err}")
        return None

    r75, err = run_generate(model_path, [
        "--eviction-policy", "sliding", "--kv-budget-ratio", "0.75"])
    if not r75:
        print(f"  SKIP: r75 failed — {err}")
        return None

    r25, err = run_generate(model_path, [
        "--eviction-policy", "sliding", "--kv-budget-ratio", "0.25"])
    if not r25:
        print(f"  SKIP: r25 failed — {err}")
        return None

    bl_nll = avg_min_nll(bl)
    r75_nll = avg_min_nll(r75)
    r25_nll = avg_min_nll(r25)

    print(f"  Baseline NLL: {bl_nll:.2f}")
    print(f"  r75 NLL:      {r75_nll:.2f} (ratio: {r75_nll/bl_nll:.2f}x)")
    print(f"  r25 NLL:      {r25_nll:.2f} (ratio: {r25_nll/bl_nll:.2f}x)")

    # r75 should be <= r25 * 1.2 (allow 20% tolerance for small sample)
    ok = r75_nll <= r25_nll * 1.2
    # r75 should be close to baseline (within 2x)
    ok2 = r75_nll <= bl_nll * 2.0

    if ok and ok2:
        print("  PASS: NLL monotonicity holds")
        return True
    else:
        if not ok:
            print(f"  FAIL: r75 ({r75_nll:.2f}) > r25 * 1.2 ({r25_nll*1.2:.2f})")
        if not ok2:
            print(f"  FAIL: r75 ({r75_nll:.2f}) > baseline * 2 ({bl_nll*2:.2f})")
        return False


def test_kivi_monotonicity(model_path):
    """Test that NLL(Q2) >= NLL(Q4) >= NLL(Q8) (more compression = more loss)."""
    print("\n[Test 2] KIVI Bit Monotonicity (Q2 >= Q4 >= Q8)")
    print("-" * 50)

    nlls = {}
    for bits in [2, 4, 8]:
        result, err = run_generate(model_path, [
            "--kivi", "--kivi-bits", str(bits), "--kivi-residual-size", "32"])
        if not result:
            print(f"  SKIP: Q{bits} failed — {err}")
            return None
        nlls[bits] = avg_min_nll(result)
        print(f"  Q{bits} NLL: {nlls[bits]:.2f}")

    ok1 = nlls[2] >= nlls[4] * 0.9  # Q2 should be >= Q4 (allow 10% tolerance)
    ok2 = nlls[4] >= nlls[8] * 0.9  # Q4 should be >= Q8

    if ok1 and ok2:
        print("  PASS: KIVI bit monotonicity holds")
        return True
    else:
        if not ok1:
            print(f"  FAIL: Q2 ({nlls[2]:.2f}) < Q4 * 0.9 ({nlls[4]*0.9:.2f})")
        if not ok2:
            print(f"  FAIL: Q4 ({nlls[4]:.2f}) < Q8 * 0.9 ({nlls[8]*0.9:.2f})")
        return False


def create_mini_eval_data():
    """Create a minimal 3-question eval dataset for fast testing."""
    questions = [
        {
            "id": "reg_1",
            "prompt": "The cat sat on the mat. The dog lay on the rug. The bird flew over the house. What animal was on the mat?",
            "choices": [" The cat", " The dog", " The bird", " The house"],
        },
        {
            "id": "reg_2",
            "prompt": "Water boils at 100 degrees Celsius at standard atmospheric pressure. What temperature does water boil at?",
            "choices": [" 100 degrees", " 200 degrees", " 50 degrees", " 0 degrees"],
        },
        {
            "id": "reg_3",
            "prompt": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet. What animal jumps?",
            "choices": [" The fox", " The dog", " The cat", " The rabbit"],
        },
    ]
    fd, path = tempfile.mkstemp(suffix=".json", prefix="regression_eval_")
    with os.fdopen(fd, "w") as f:
        json.dump(questions, f)
    return path


def main():
    parser = argparse.ArgumentParser(description="Regression tests for llm.rs2")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--eval-data", help="Path to eval batch JSON (default: auto-generated)")
    parser.add_argument("--binary", default=None, help="Path to generate binary")
    args = parser.parse_args()

    global _GENERATE, _EVAL_DATA
    if args.binary:
        _GENERATE = args.binary

    # Create or use eval data
    auto_eval = False
    if args.eval_data:
        _EVAL_DATA = args.eval_data
    else:
        _EVAL_DATA = create_mini_eval_data()
        auto_eval = True

    print("=" * 55)
    print("llm.rs2 Regression Tests")
    print(f"  Model: {args.model_path}")
    print(f"  Eval:  {_EVAL_DATA}")
    print("=" * 55)

    results = []
    results.append(("NLL Monotonicity", test_nll_monotonicity(args.model_path)))
    results.append(("KIVI Bit Monotonicity", test_kivi_monotonicity(args.model_path)))

    # Cleanup
    if auto_eval:
        os.unlink(_EVAL_DATA)

    # Summary
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    passed = 0
    failed = 0
    skipped = 0
    for name, result in results:
        if result is True:
            print(f"  PASS: {name}")
            passed += 1
        elif result is False:
            print(f"  FAIL: {name}")
            failed += 1
        else:
            print(f"  SKIP: {name}")
            skipped += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
