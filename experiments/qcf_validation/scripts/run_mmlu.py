#!/usr/bin/env python3
"""Phase 4: MMLU evaluation with QCF.

Uses many-shot (20-shot) ICL format to fill ~1500 tokens of context.
KV budget eviction removes earlier examples, degrading ICL performance.
Leverages the existing eval-ll mode for NLL-based answer selection.

Usage:
    python run_mmlu.py [--budgets B0,B1,B2,B3,B4]
    python run_mmlu.py --download  # Download MMLU data first

Prerequisites:
    pip install datasets
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (BUDGETS, ensure_binary, run_eval_ll, run_ppl,
                     save_result, compute_qcf_summary, get_qcf_metrics)

MMLU_DIR = Path(__file__).resolve().parents[1] / "data" / "mmlu"
POLICY = "sliding"
N_SHOTS = 20
N_TEST_PER_SUBJECT = 20

# Subjects where 1B models tend to perform above random (~30%+)
SUBJECTS = [
    "marketing",
    "professional_psychology",
    "high_school_psychology",
    "us_foreign_policy",
    "human_sexuality",
]

CHOICES = ["A", "B", "C", "D"]


def download_mmlu():
    """Download MMLU dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    MMLU_DIR.mkdir(parents=True, exist_ok=True)
    for subject in SUBJECTS:
        out_path = MMLU_DIR / f"{subject}.json"
        if out_path.exists():
            print(f"  [skip] {subject} already exists", file=sys.stderr)
            continue
        print(f"  [download] {subject}...", file=sys.stderr)
        ds = load_dataset("cais/mmlu", subject)
        # Save test + dev splits
        data = {
            "subject": subject,
            "test": [dict(row) for row in ds["test"]],
            "dev": [dict(row) for row in ds.get("validation", ds.get("dev", []))],
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"    test={len(data['test'])}, dev={len(data.get('dev', []))}", file=sys.stderr)

    print(f"[download] Complete. Data in {MMLU_DIR}", file=sys.stderr)


def format_mmlu_question(question: str, choices: list, answer_idx: int = None) -> str:
    """Format a single MMLU question as text."""
    parts = [question]
    for i, choice in enumerate(choices):
        parts.append(f"{CHOICES[i]}. {choice}")
    text = "\n".join(parts)
    if answer_idx is not None:
        text += f"\nAnswer: {CHOICES[answer_idx]}"
    return text


def build_eval_batch(subject_data: dict, n_shots: int,
                      n_test: int) -> tuple[list[dict], list[int]]:
    """Build eval-ll batch JSON from MMLU subject data.

    Returns (batch_tasks, correct_answers).
    Each task: {"id": ..., "prompt": "20-shot prefix + question", "choices": ["A","B","C","D"]}
    """
    dev = subject_data.get("dev", [])
    test = subject_data["test"]

    # Build few-shot prefix from dev set (cycle if needed)
    shot_examples = []
    for i in range(n_shots):
        if i < len(dev):
            ex = dev[i]
        elif dev:
            ex = dev[i % len(dev)]
        else:
            # Fallback: use test examples (skip the ones we'll evaluate)
            ex = test[n_test + (i % max(1, len(test) - n_test))]
        shot_examples.append(format_mmlu_question(
            ex["question"], ex["choices"], ex["answer"]
        ))

    prefix = "The following are multiple choice questions. Select the correct answer.\n\n"
    prefix += "\n\n".join(shot_examples)
    prefix += "\n\n"

    # Build test questions
    batch = []
    correct = []
    for i in range(min(n_test, len(test))):
        q = test[i]
        question_text = format_mmlu_question(q["question"], q["choices"])
        prompt = prefix + question_text + "\nAnswer:"

        # Each choice is a single letter
        task = {
            "id": f"{subject_data['subject']}_{i}",
            "prompt": prompt,
            "choices": [f" {c}" for c in CHOICES],  # space prefix for tokenizer
        }
        batch.append(task)
        correct.append(q["answer"])

    return batch, correct


def main():
    parser = argparse.ArgumentParser(description="Phase 4: MMLU evaluation")
    parser.add_argument("--download", action="store_true", help="Download MMLU data")
    parser.add_argument("--budgets", default="B0,B1,B2,B3,B4")
    parser.add_argument("--policy", default=POLICY)
    parser.add_argument("--n-shots", type=int, default=N_SHOTS)
    parser.add_argument("--n-test", type=int, default=N_TEST_PER_SUBJECT)
    args = parser.parse_args()

    if args.download:
        download_mmlu()
        return

    budget_keys = args.budgets.split(",")
    ensure_binary()

    # Check data exists
    if not MMLU_DIR.exists():
        print("MMLU data not found. Run with --download first.", file=sys.stderr)
        sys.exit(1)

    all_results = []
    start = time.time()

    for subject in SUBJECTS:
        data_path = MMLU_DIR / f"{subject}.json"
        if not data_path.exists():
            print(f"[skip] {subject} data not found", file=sys.stderr)
            continue

        with open(data_path) as f:
            subject_data = json.load(f)

        batch, correct_answers = build_eval_batch(
            subject_data, args.n_shots, args.n_test
        )
        if not batch:
            print(f"[skip] {subject}: no test questions", file=sys.stderr)
            continue

        print(f"\n[Phase 4] Subject: {subject} ({len(batch)} questions, "
              f"{args.n_shots}-shot)", file=sys.stderr)

        for bk in budget_keys:
            budget = BUDGETS[bk]
            label = f"{subject}_{bk}"
            print(f"  [{label}] budget={budget}...", file=sys.stderr)

            # Write batch to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(batch, f)
                batch_path = f.name

            try:
                eval_result = run_eval_ll(batch_path, args.policy, budget)
            finally:
                Path(batch_path).unlink(missing_ok=True)

            # Score results
            results_list = eval_result.get("results", [])
            n_correct = 0
            for i, res in enumerate(results_list):
                predicted = res.get("predicted", -1)
                if i < len(correct_answers) and predicted == correct_answers[i]:
                    n_correct += 1

            accuracy = n_correct / len(batch) if batch else 0.0

            # Get QCF for this budget (use first question's prompt as representative)
            qcf_avg = 0.0
            if budget < 2048 and batch:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False
                ) as f:
                    # Use first question prompt for QCF measurement
                    f.write(batch[0]["prompt"])
                    tmp_path = f.name
                try:
                    ppl_result = run_ppl(tmp_path, args.policy, budget)
                    qcf_s = compute_qcf_summary(get_qcf_metrics(ppl_result))
                    qcf_avg = qcf_s["avg"]
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

            result = {
                "subject": subject,
                "budget_key": bk,
                "budget": budget,
                "policy": args.policy,
                "n_shots": args.n_shots,
                "accuracy": accuracy,
                "n_correct": n_correct,
                "n_total": len(batch),
                "qcf_avg": qcf_avg,
            }
            save_result("mmlu", label, result)
            all_results.append(result)

            print(f"    Accuracy: {accuracy:.1%} ({n_correct}/{len(batch)}), "
                  f"QCF={qcf_avg:.6f}", file=sys.stderr)

    # Summary by budget (aggregate across subjects)
    summary_rows = []
    for bk in budget_keys:
        subset = [r for r in all_results if r["budget_key"] == bk]
        if not subset:
            continue
        total_correct = sum(r["n_correct"] for r in subset)
        total_questions = sum(r["n_total"] for r in subset)
        accuracy = total_correct / total_questions if total_questions else 0
        avg_qcf = sum(r["qcf_avg"] for r in subset) / len(subset) if subset else 0
        summary_rows.append({
            "budget_key": bk,
            "budget": BUDGETS[bk],
            "accuracy": accuracy,
            "n_correct": total_correct,
            "n_total": total_questions,
            "qcf_avg": avg_qcf,
        })

    # Summary by subject
    subject_rows = []
    for subject in SUBJECTS:
        for bk in budget_keys:
            subset = [r for r in all_results
                      if r["budget_key"] == bk and r["subject"] == subject]
            if subset:
                r = subset[0]
                subject_rows.append({
                    "subject": subject,
                    "budget_key": bk,
                    "accuracy": r["accuracy"],
                })

    summary = {
        "phase": "mmlu",
        "policy": args.policy,
        "n_shots": args.n_shots,
        "subjects": SUBJECTS,
        "results_by_budget": summary_rows,
        "results_by_subject": subject_rows,
        "all_results": all_results,
        "wall_time_s": time.time() - start,
    }
    save_result("mmlu", "_summary", summary)

    elapsed = time.time() - start
    print(f"\n[Phase 4] Complete in {elapsed:.0f}s", file=sys.stderr)
    print("\n[Phase 4] Accuracy by budget:", file=sys.stderr)
    for row in summary_rows:
        print(f"  {row['budget_key']} (budget={row['budget']}): "
              f"{row['accuracy']:.1%} ({row['n_correct']}/{row['n_total']}), "
              f"QCF={row['qcf_avg']:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
