#!/usr/bin/env python3
"""Phase 1: PPL sweep with QCF collection.

Measures perplexity at various KV budgets for Sliding and H2O policies.
Collects QCF metrics at each eviction event.

Usage:
    python run_ppl.py [--policies sliding,h2o] [--budgets B1,B2,B3,B4,B5]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (BUDGETS, POLICIES, RESULTS_DIR, ensure_binary,
                     run_ppl, save_result, compute_qcf_summary, get_qcf_metrics)

EVAL_TEXT = Path(__file__).resolve().parents[2] / "proxy_validation" / "texts" / "eval_text.txt"


def main():
    parser = argparse.ArgumentParser(description="Phase 1: PPL sweep")
    parser.add_argument("--policies", default="sliding,h2o")
    parser.add_argument("--budgets", default="B1,B2,B3,B4,B5")
    args = parser.parse_args()

    policies = args.policies.split(",")
    budget_keys = args.budgets.split(",")

    assert EVAL_TEXT.exists(), f"Eval text not found: {EVAL_TEXT}"
    ensure_binary()

    all_results = []
    start = time.time()

    # 1. Baseline (no eviction)
    print("[Phase 1] Running baseline (no eviction)...", file=sys.stderr)
    baseline = run_ppl(str(EVAL_TEXT), "none", 2048)
    baseline["budget_key"] = "B0"
    baseline["budget"] = 2048
    baseline["policy"] = "none"
    baseline["qcf_summary"] = compute_qcf_summary(get_qcf_metrics(baseline))
    save_result("ppl", "baseline", baseline)
    baseline_ppl = baseline.get("ppl", float("inf"))
    print(f"  Baseline PPL: {baseline_ppl:.4f}", file=sys.stderr)
    all_results.append(baseline)

    # 2. Sweep each policy × budget
    for policy in policies:
        for bk in budget_keys:
            budget = BUDGETS[bk]
            label = f"{policy}_{bk}"
            print(f"[Phase 1] {label} (budget={budget})...", file=sys.stderr)

            result = run_ppl(str(EVAL_TEXT), policy, budget)
            result["budget_key"] = bk
            result["budget"] = budget
            result["policy"] = policy
            result["delta_ppl"] = result.get("ppl", 0) - baseline_ppl
            result["qcf_summary"] = compute_qcf_summary(get_qcf_metrics(result))
            save_result("ppl", label, result)

            ppl = result.get("ppl", float("inf"))
            qcf_avg = result["qcf_summary"]["avg"]
            print(f"  PPL={ppl:.4f} (Δ={result['delta_ppl']:.4f}), "
                  f"QCF avg={qcf_avg:.6f}, evictions={result['qcf_summary']['count']}",
                  file=sys.stderr)
            all_results.append(result)

    # 3. Save summary
    summary = {
        "phase": "ppl",
        "baseline_ppl": baseline_ppl,
        "results": [
            {
                "policy": r["policy"],
                "budget_key": r["budget_key"],
                "budget": r["budget"],
                "ppl": r.get("ppl"),
                "delta_ppl": r.get("delta_ppl", 0),
                "qcf_avg": r["qcf_summary"]["avg"],
                "qcf_total": r["qcf_summary"]["total"],
                "eviction_count": r["qcf_summary"]["count"],
            }
            for r in all_results
        ],
        "wall_time_s": time.time() - start,
    }
    save_result("ppl", "_summary", summary)

    elapsed = time.time() - start
    print(f"\n[Phase 1] Complete: {len(all_results)} experiments in {elapsed:.0f}s",
          file=sys.stderr)
    print(f"[Phase 1] Results: {RESULTS_DIR / 'ppl'}", file=sys.stderr)


if __name__ == "__main__":
    main()
