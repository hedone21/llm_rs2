#!/usr/bin/env python3
"""Run log-likelihood evaluation on benchmark datasets.

Calls `generate --eval-ll --eval-batch` for each task x policy combination,
then computes accuracy by comparing log-likelihoods across choices.

Usage:
    # H2O paper reproduction (ratio-based budget)
    python experiments/benchmarks/run_eval.py \
        --model models/llama3.2-1b \
        --tasks copa,piqa,winogrande,openbookqa,rte,mathqa \
        --policies none,sliding,h2o,h2o_plus \
        --kv-budget-ratio 0.2 \
        --kv-type f16

    # Absolute budget (legacy)
    python experiments/benchmarks/run_eval.py \
        --model models/llama3.2-1b \
        --tasks boolq --policies none --kv-budget 256
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
BINARY = os.path.join(PROJECT_ROOT, "target/release/generate")


def load_tasks(task_name, n_questions=None):
    """Load evaluation tasks from prepared JSON file."""
    path = os.path.join(DATA_DIR, f"{task_name}.json")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run prepare_datasets.py first.", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        tasks = json.load(f)

    if n_questions is not None:
        question_ids = []
        seen = set()
        for t in tasks:
            qid = t["question_id"]
            if qid not in seen:
                seen.add(qid)
                question_ids.append(qid)
        question_ids = question_ids[:n_questions]
        keep = set(question_ids)
        tasks = [t for t in tasks if t["question_id"] in keep]

    return tasks


def tasks_to_grouped(tasks):
    """Convert flat tasks to grouped format for the binary."""
    questions = defaultdict(list)
    for t in tasks:
        questions[t["question_id"]].append(t)

    grouped = []
    gold_map = {}
    for qid, choices in questions.items():
        choices.sort(key=lambda c: c["choice_idx"])
        grouped.append({
            "id": qid,
            "prompt": choices[0]["prompt"],
            "choices": [c["continuation"] for c in choices],
        })
        gold_map[qid] = choices[0]["gold"]

    return grouped, gold_map


def run_eval_batch(tasks, model, policy, kv_budget, kv_budget_ratio, max_seq_len, kv_type,
                    kivi=False, kivi_residual_size=32, skip_layers=None, skip_ratio=None):
    """Run generate --eval-ll --eval-batch and return results."""
    grouped, gold_map = tasks_to_grouped(tasks)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(grouped, f, ensure_ascii=False)
        batch_path = f.name

    try:
        cmd = [
            BINARY,
            "-m", model,
            "--eval-ll",
            "--eval-batch", batch_path,
            "--max-seq-len", str(max_seq_len),
            "--eviction-policy", policy,
            "--kv-type", kv_type,
            "--greedy",
        ]

        if kivi:
            cmd.extend(["--kivi", "--kivi-residual-size", str(kivi_residual_size)])

        if kv_budget_ratio > 0:
            cmd.extend(["--kv-budget-ratio", str(kv_budget_ratio)])
        elif kv_budget > 0:
            cmd.extend(["--kv-budget", str(kv_budget)])

        if policy in ("h2o", "h2o_plus"):
            cmd.extend(["--h2o-keep-ratio", "0.5", "--h2o-decay", "0.0"])

        if skip_layers:
            cmd.extend(["--skip-layers", skip_layers])
        elif skip_ratio is not None:
            cmd.extend(["--skip-ratio", str(skip_ratio)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)

        if result.returncode != 0:
            print(f"ERROR: generate failed:\n{result.stderr}", file=sys.stderr)
            return None, None

        output = result.stdout.strip()
        json_start = output.find("{")
        if json_start < 0:
            print(f"ERROR: no JSON in output:\n{output}", file=sys.stderr)
            return None, None

        return json.loads(output[json_start:]), gold_map

    finally:
        os.unlink(batch_path)


def compute_accuracy(eval_results, gold_map):
    """Compute acc_norm from grouped log-likelihood results."""
    if not eval_results:
        return {"accuracy": 0.0, "correct": 0, "total": 0, "details": []}

    correct = 0
    total = 0
    details = []
    layer_skip_qcf_sum = 0.0
    layer_skip_qcf_normalized_sum = 0.0
    layer_skip_qcf_count = 0

    for r in eval_results.get("results", []):
        qid = r["id"]
        gold = gold_map.get(qid)
        if gold is None:
            continue

        pred = r.get("predicted", 0)

        if pred == gold:
            correct += 1
        total += 1

        layer_skip_qcf = r.get("layer_skip_qcf")
        layer_skip_qcf_normalized = r.get("layer_skip_qcf_normalized")
        if layer_skip_qcf is not None:
            layer_skip_qcf_sum += layer_skip_qcf
            layer_skip_qcf_normalized_sum += layer_skip_qcf_normalized if layer_skip_qcf_normalized is not None else 0.0
            layer_skip_qcf_count += 1

        details.append({
            "question_id": qid,
            "gold": gold,
            "predicted": pred,
            "correct": pred == gold,
            "choice_nlls": r.get("choice_nlls", []),
            "n_prompt_tokens": r.get("n_prompt_tokens", 0),
            "effective_budget": r.get("effective_budget", 0),
            "eviction_count": r.get("eviction_count", 0),
            "layer_skip_qcf": layer_skip_qcf,
            "layer_skip_qcf_normalized": layer_skip_qcf_normalized,
        })

    acc = correct / total if total > 0 else 0.0
    result = {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "details": details,
    }
    if layer_skip_qcf_count > 0:
        result["layer_skip_qcf_mean"] = layer_skip_qcf_sum / layer_skip_qcf_count
        result["layer_skip_qcf_normalized_mean"] = layer_skip_qcf_normalized_sum / layer_skip_qcf_count
    return result


def main():
    parser = argparse.ArgumentParser(description="Run downstream task evaluation")
    parser.add_argument("--model", default="models/llama3.2-1b")
    parser.add_argument("--tasks", default="copa,piqa,winogrande,openbookqa,rte,mathqa",
                        help="Comma-separated task names")
    parser.add_argument("--policies", default="none",
                        help="Comma-separated eviction policies")
    parser.add_argument("--kv-budget", type=int, default=0,
                        help="KV cache budget in tokens (0=unlimited)")
    parser.add_argument("--kv-budget-ratio", type=float, default=0.0,
                        help="KV cache budget as ratio of prompt length (0.0-1.0)")
    parser.add_argument("--kv-type", default="f16",
                        help="KV cache data type (f16, f32, q4)")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--n-questions", type=int, default=None,
                        help="Limit number of questions per task")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    parser.add_argument("--kivi", action="store_true",
                        help="Enable KIVI Q2 KV cache compression")
    parser.add_argument("--kivi-residual-size", type=int, default=32,
                        help="KIVI residual buffer size (default: 32)")
    parser.add_argument("--skip-layers", default=None,
                        help="Comma-separated layer indices to skip (e.g. '1,3,5'). "
                             "Passed directly to generate binary. Takes precedence over --skip-ratio.")
    parser.add_argument("--skip-ratio", type=float, default=None,
                        help="Fraction of layers to skip (0.0-1.0). "
                             "Ignored when --skip-layers is specified.")
    args = parser.parse_args()

    task_names = [t.strip() for t in args.tasks.split(",")]
    policies = [p.strip() for p in args.policies.split(",")]
    model_name = os.path.basename(args.model)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}

    for task_name in task_names:
        tasks = load_tasks(task_name, n_questions=args.n_questions)
        n_questions = len(set(t["question_id"] for t in tasks))
        print(f"\n{'='*60}")
        print(f"  Task: {task_name} ({n_questions} questions)")
        print(f"{'='*60}")

        for policy in policies:
            label = f"{task_name}/{policy}"
            if args.kv_budget_ratio > 0:
                label += f"/ratio={args.kv_budget_ratio}"
            elif args.kv_budget > 0:
                label += f"/budget={args.kv_budget}"
            if args.skip_layers:
                label += f"/skip_layers={args.skip_layers}"
            elif args.skip_ratio is not None:
                label += f"/skip_ratio={args.skip_ratio}"

            print(f"\n  Running: {label}...")
            eval_output, gold_map = run_eval_batch(
                tasks, args.model, policy,
                args.kv_budget, args.kv_budget_ratio,
                args.max_seq_len, args.kv_type,
                kivi=args.kivi,
                kivi_residual_size=args.kivi_residual_size,
                skip_layers=args.skip_layers,
                skip_ratio=args.skip_ratio,
            )

            if eval_output is None:
                print(f"  FAILED: {label}")
                continue

            acc_result = compute_accuracy(eval_output, gold_map)
            acc = acc_result["accuracy"]
            extra = ""
            if "layer_skip_qcf_mean" in acc_result:
                extra = (f", layer_skip_qcf={acc_result['layer_skip_qcf_mean']:.4f}"
                         f", qcf_norm={acc_result['layer_skip_qcf_normalized_mean']:.4f}")
            print(f"  Result: acc_norm={acc:.1%} ({acc_result['correct']}/{acc_result['total']}){extra}")

            key = f"{task_name}_{policy}"
            if args.kv_budget_ratio > 0:
                key += f"_r{int(args.kv_budget_ratio*100)}"
            elif args.kv_budget > 0:
                key += f"_b{args.kv_budget}"
            if args.kivi:
                key += f"_kivi_r{args.kivi_residual_size}"
            if args.skip_layers:
                key += f"_skip_{args.skip_layers.replace(',', '-')}"
            elif args.skip_ratio is not None:
                key += f"_skipratio_{int(args.skip_ratio * 100)}"
            all_results[key] = {
                "task": task_name,
                "policy": policy if not args.kivi else f"kivi_r{args.kivi_residual_size}",
                "kv_budget": args.kv_budget,
                "kv_budget_ratio": args.kv_budget_ratio,
                "kv_type": args.kv_type if not args.kivi else "q2+f32",
                "model": args.model,
                "accuracy": acc,
                "correct": acc_result["correct"],
                "total": acc_result["total"],
                "wall_time_s": eval_output.get("wall_time_s", 0),
                "config": eval_output.get("config", {}),
                "skip_layers": args.skip_layers,
                "skip_ratio": args.skip_ratio,
                "layer_skip_qcf_mean": acc_result.get("layer_skip_qcf_mean"),
                "layer_skip_qcf_normalized_mean": acc_result.get("layer_skip_qcf_normalized_mean"),
            }

    # Print summary table
    has_skip = any(r.get("skip_layers") or r.get("skip_ratio") is not None for r in all_results.values())
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    if has_skip:
        print(f"  {'Task':<15} {'Policy':<12} {'Budget':<12} {'Skip':<15} {'Acc(norm)':>10} {'QCF(norm)':>10}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*15} {'-'*10} {'-'*10}")
    else:
        print(f"  {'Task':<15} {'Policy':<12} {'Budget':<12} {'Acc(norm)':>10}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10}")
    for key, r in all_results.items():
        bl = (f"r={r['kv_budget_ratio']}" if r["kv_budget_ratio"] > 0
              else f"b={r['kv_budget']}" if r["kv_budget"] > 0 else "full")
        if has_skip:
            skip_label = (r["skip_layers"] if r.get("skip_layers")
                          else f"ratio={r['skip_ratio']}" if r.get("skip_ratio") is not None
                          else "-")
            qcf_norm = r.get("layer_skip_qcf_normalized_mean")
            qcf_str = f"{qcf_norm:.4f}" if qcf_norm is not None else "-"
            print(f"  {r['task']:<15} {r['policy']:<12} {bl:<12} {skip_label:<15} {r['accuracy']:>9.1%} {qcf_str:>10}")
        else:
            print(f"  {r['task']:<15} {r['policy']:<12} {bl:<12} {r['accuracy']:>9.1%}")

    # Save results
    output_path = args.output
    if output_path is None:
        policies_str = "_".join(policies)
        if args.kv_budget_ratio > 0:
            budget_str = f"_r{int(args.kv_budget_ratio*100)}"
        elif args.kv_budget > 0:
            budget_str = f"_b{args.kv_budget}"
        else:
            budget_str = ""
        if args.kivi:
            budget_str += f"_kivi_r{args.kivi_residual_size}"
        if args.skip_layers:
            budget_str += f"_skip_{args.skip_layers.replace(',', '-')}"
        elif args.skip_ratio is not None:
            budget_str += f"_skipratio_{int(args.skip_ratio * 100)}"
        output_path = os.path.join(
            RESULTS_DIR,
            f"{model_name}_{policies_str}{budget_str}.json"
        )

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
