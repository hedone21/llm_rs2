#!/usr/bin/env python3
"""Phase 3: QA evaluation with QCF.

Runs Single-doc QA, Summarization, and Multi-hop QA benchmarks.
Documents are padded with filler text to create ~1500 token contexts,
forcing mid-inference eviction at lower KV budgets.

Usage:
    python run_qa.py [--budgets B0,B1,B2,B3,B4]
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
                     get_qcf_metrics, compute_f1, contains_answer)

POLICY = "sliding"
GEN_TOKENS = 128
TARGET_PROMPT_TOKENS = 1500  # Pad prompts to this length


def pad_prompt(document: str, question: str, fillers: list[dict],
               target_tokens: int = TARGET_PROMPT_TOKENS) -> str:
    """Pad document+question with filler blocks to reach target length.

    Structure: [Filler padding] [Document] [Question]
    This ensures the document is near the end and fillers are evicted first.
    """
    doc_question = f"{document}\n\nQuestion: {question}\nAnswer:"
    doc_tokens_approx = len(doc_question.split()) * 1.3

    # Add filler blocks before the document
    padding_parts = []
    current_tokens = doc_tokens_approx
    filler_idx = 0
    while current_tokens < target_tokens and filler_idx < len(fillers) * 2:
        filler = fillers[filler_idx % len(fillers)]
        filler_text = filler["text"]
        filler_tokens = len(filler_text.split()) * 1.3
        padding_parts.append(filler_text)
        current_tokens += filler_tokens
        filler_idx += 1

    # Assemble: padding first, then document + question
    if padding_parts:
        padding = "\n\n".join(padding_parts)
        return f"{padding}\n\n{doc_question}"
    return doc_question


def main():
    parser = argparse.ArgumentParser(description="Phase 3: QA evaluation")
    parser.add_argument("--budgets", default="B0,B1,B2,B3,B4")
    parser.add_argument("--policy", default=POLICY)
    args = parser.parse_args()

    budget_keys = args.budgets.split(",")
    ensure_binary()
    prompts_data = load_prompts()
    qa_data = prompts_data["qa"]
    fillers = prompts_data["niah"]["filler_blocks"]

    # Collect all QA tasks
    tasks = []
    for category in ["single_doc_qa", "summarization", "multi_hop"]:
        for item in qa_data.get(category, []):
            tasks.append({
                "id": item["id"],
                "category": category,
                "document": item["document"],
                "question": item.get("question", ""),
                "expected": item["expected_answer"],
            })

    all_results = []
    start = time.time()
    total = len(tasks) * len(budget_keys)
    done = 0

    for task in tasks:
        prompt = pad_prompt(task["document"], task["question"], fillers)
        prompt_tokens_approx = len(prompt.split()) * 1.3

        for bk in budget_keys:
            budget = BUDGETS[bk]
            done += 1
            label = f"{task['id']}_{bk}"
            print(f"[Phase 3] ({done}/{total}) {label}...", file=sys.stderr)

            # 1. Generate answer
            generated = run_generate(prompt, GEN_TOKENS, args.policy, budget)

            # 2. Evaluate
            f1 = compute_f1(generated, task["expected"])
            em = contains_answer(generated, task["expected"])

            # 3. Get QCF via PPL on same prompt
            qcf_avg = 0.0
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
                    eviction_count = qcf_s["count"]
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

            result = {
                "task_id": task["id"],
                "category": task["category"],
                "budget_key": bk,
                "budget": budget,
                "policy": args.policy,
                "f1": f1,
                "exact_match": em,
                "expected": task["expected"],
                "generated_prefix": generated[:300],
                "prompt_len_approx": int(prompt_tokens_approx),
                "qcf_avg": qcf_avg,
                "eviction_count": eviction_count,
            }
            save_result("qa", label, result)
            all_results.append(result)
            print(f"  F1={f1:.3f} EM={'Y' if em else 'N'} | QCF={qcf_avg:.6f}",
                  file=sys.stderr)

    # Summary by budget
    summary_rows = []
    for bk in budget_keys:
        subset = [r for r in all_results if r["budget_key"] == bk]
        if not subset:
            continue
        avg_f1 = sum(r["f1"] for r in subset) / len(subset)
        em_rate = sum(1 for r in subset if r["exact_match"]) / len(subset)
        avg_qcf = sum(r["qcf_avg"] for r in subset) / len(subset)
        summary_rows.append({
            "budget_key": bk,
            "budget": BUDGETS[bk],
            "avg_f1": avg_f1,
            "em_rate": em_rate,
            "qcf_avg": avg_qcf,
            "n_tasks": len(subset),
        })

    # Summary by category
    category_rows = []
    for cat in ["single_doc_qa", "summarization", "multi_hop"]:
        for bk in budget_keys:
            subset = [r for r in all_results
                      if r["budget_key"] == bk and r["category"] == cat]
            if not subset:
                continue
            avg_f1 = sum(r["f1"] for r in subset) / len(subset)
            category_rows.append({
                "category": cat,
                "budget_key": bk,
                "avg_f1": avg_f1,
                "n": len(subset),
            })

    summary = {
        "phase": "qa",
        "policy": args.policy,
        "results_by_budget": summary_rows,
        "results_by_category": category_rows,
        "all_results": all_results,
        "wall_time_s": time.time() - start,
    }
    save_result("qa", "_summary", summary)

    elapsed = time.time() - start
    print(f"\n[Phase 3] Complete: {done} experiments in {elapsed:.0f}s", file=sys.stderr)
    print("\n[Phase 3] F1 by budget:", file=sys.stderr)
    for row in summary_rows:
        print(f"  {row['budget_key']} (budget={row['budget']}): "
              f"F1={row['avg_f1']:.3f}, EM={row['em_rate']:.1%}, "
              f"QCF={row['qcf_avg']:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
