#!/usr/bin/env python3
"""Phase 2: NIAH (Needle-in-a-Haystack) evaluation with QCF.

Assembles NIAH prompts at various depths, runs generation with KV budget
eviction, checks if the model retrieves the needle, and collects QCF
via a parallel PPL run on the same prompt.

Usage:
    python run_niah.py [--budgets B0,B1,B2,B3,B4]
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (BUDGETS, ensure_binary, load_prompts,
                     run_generate, run_ppl, save_result, compute_qcf_summary,
                     get_qcf_metrics, contains_answer)

NEEDLES = ["N-PASS", "N-FACT", "N-NUM"]
DEPTHS = [0.1, 0.25, 0.5]
NUM_BLOCKS = 20  # ~1500 tokens (8 filler blocks cycled)
POLICY = "sliding"
GEN_TOKENS = 64


def assemble_niah_prompt(prompts_data: dict, needle_id: str,
                          depth: float, num_blocks: int) -> tuple[str, str]:
    """Assemble a NIAH prompt. Returns (full_prompt, expected_answer)."""
    niah = prompts_data["niah"]
    fillers = niah["filler_blocks"]
    needle = next(n for n in niah["needles"] if n["id"] == needle_id)

    # Distribute fillers around needle based on depth
    n_before = max(1, int(num_blocks * depth))
    n_after = num_blocks - n_before

    # Cycle fillers if needed
    def get_fillers(count):
        result = []
        for i in range(count):
            result.append(fillers[i % len(fillers)]["text"])
        return result

    before = get_fillers(n_before)
    after = get_fillers(n_after)

    parts = before + [needle["needle_text"]] + after + [needle["question"]]
    prompt = "\n\n".join(parts)
    return prompt, needle["expected_answer"]


def main():
    parser = argparse.ArgumentParser(description="Phase 2: NIAH evaluation")
    parser.add_argument("--budgets", default="B0,B1,B2,B3,B4")
    parser.add_argument("--policy", default=POLICY)
    args = parser.parse_args()

    budget_keys = args.budgets.split(",")
    ensure_binary()
    prompts_data = load_prompts()

    all_results = []
    start = time.time()
    total = len(NEEDLES) * len(DEPTHS) * len(budget_keys)
    done = 0

    for needle_id in NEEDLES:
        for depth in DEPTHS:
            prompt, expected = assemble_niah_prompt(
                prompts_data, needle_id, depth, NUM_BLOCKS
            )
            prompt_tokens_approx = len(prompt.split()) * 1.3  # rough estimate

            for bk in budget_keys:
                budget = BUDGETS[bk]
                done += 1
                label = f"{needle_id}_d{depth}_{bk}"
                print(f"[Phase 2] ({done}/{total}) {label}...", file=sys.stderr)

                # 1. Generate answer
                generated = run_generate(prompt, GEN_TOKENS, args.policy, budget)
                success = contains_answer(generated, expected)

                # 2. Get QCF via PPL on same prompt
                qcf_avg = 0.0
                qcf_total = 0.0
                eviction_count = 0
                if budget < 2048:
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".txt", delete=False
                    ) as f:
                        f.write(prompt)
                        tmp_path = f.name
                    try:
                        ppl_result = run_ppl(tmp_path, args.policy, budget)
                        qcf_s = compute_qcf_summary(get_qcf_metrics(ppl_result))
                        qcf_avg = qcf_s["avg"]
                        qcf_total = qcf_s["total"]
                        eviction_count = qcf_s["count"]
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)

                result = {
                    "needle": needle_id,
                    "depth": depth,
                    "budget_key": bk,
                    "budget": budget,
                    "policy": args.policy,
                    "success": success,
                    "expected": expected,
                    "generated_prefix": generated[:200],
                    "prompt_len_approx": int(prompt_tokens_approx),
                    "qcf_avg": qcf_avg,
                    "qcf_total": qcf_total,
                    "eviction_count": eviction_count,
                }
                save_result("niah", label, result)
                all_results.append(result)

                status = "PASS" if success else "FAIL"
                print(f"  {status} | QCF={qcf_avg:.6f} | evictions={eviction_count}",
                      file=sys.stderr)

    # Summary: accuracy per budget
    summary_rows = []
    for bk in budget_keys:
        subset = [r for r in all_results if r["budget_key"] == bk]
        n_pass = sum(1 for r in subset if r["success"])
        n_total = len(subset)
        accuracy = n_pass / n_total if n_total > 0 else 0.0
        avg_qcf = (sum(r["qcf_avg"] for r in subset) / n_total) if n_total else 0.0
        summary_rows.append({
            "budget_key": bk,
            "budget": BUDGETS[bk],
            "accuracy": accuracy,
            "pass": n_pass,
            "total": n_total,
            "qcf_avg": avg_qcf,
        })

    summary = {
        "phase": "niah",
        "policy": args.policy,
        "needles": NEEDLES,
        "depths": DEPTHS,
        "results_by_budget": summary_rows,
        "all_results": all_results,
        "wall_time_s": time.time() - start,
    }
    save_result("niah", "_summary", summary)

    elapsed = time.time() - start
    print(f"\n[Phase 2] Complete: {done} experiments in {elapsed:.0f}s", file=sys.stderr)
    print("\n[Phase 2] Accuracy by budget:", file=sys.stderr)
    for row in summary_rows:
        print(f"  {row['budget_key']} (budget={row['budget']}): "
              f"{row['accuracy']:.1%} ({row['pass']}/{row['total']}), "
              f"QCF={row['qcf_avg']:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
