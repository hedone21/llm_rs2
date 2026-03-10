#!/usr/bin/env python3
"""Round 15: Compare Sliding vs H2O-Raw vs H2O-Normalized on real model output."""

import json
import sys
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent.parent / "results" / "round15"

def load_jsonl(path):
    """Load JSONL experiment results (skip summary record)."""
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "token_id" in r:  # skip summary record
                records.append(r)
    return records

def token_ids(records):
    return [r["token_id"] for r in records]

def top_k_ids(records, k=5):
    """Extract top-k token IDs per position."""
    result = []
    for r in records:
        ids = [t[0] for t in r.get("top_logits", [])[:k]]
        result.append(set(ids))
    return result

def compute_metrics(baseline, target, label):
    """Compute EMR, FDT, Top-K overlap between baseline and target."""
    base_tokens = token_ids(baseline)
    tgt_tokens = token_ids(target)

    min_len = min(len(base_tokens), len(tgt_tokens))

    # EMR: exact match ratio
    matches = sum(1 for i in range(min_len) if base_tokens[i] == tgt_tokens[i])
    emr = matches / min_len if min_len > 0 else 0.0

    # FDT: first divergent token
    fdt = min_len  # if all match
    for i in range(min_len):
        if base_tokens[i] != tgt_tokens[i]:
            fdt = i
            break

    # Suffix EMR: match ratio after first eviction point
    # Find first eviction in target
    first_eviction_pos = None
    for r in target:
        if r.get("actions") and len(r["actions"]) > 0:
            first_eviction_pos = r["pos"]
            break

    if first_eviction_pos is not None and first_eviction_pos < min_len:
        suffix_matches = sum(1 for i in range(first_eviction_pos, min_len)
                           if base_tokens[i] == tgt_tokens[i])
        suffix_len = min_len - first_eviction_pos
        suffix_emr = suffix_matches / suffix_len if suffix_len > 0 else 0.0
    else:
        suffix_emr = emr
        first_eviction_pos = "N/A"

    # Top-5 overlap
    base_top5 = top_k_ids(baseline, 5)
    tgt_top5 = top_k_ids(target, 5)
    overlaps = []
    for i in range(min_len):
        if base_top5[i] and tgt_top5[i]:
            overlap = len(base_top5[i] & tgt_top5[i]) / max(len(base_top5[i]), 1)
            overlaps.append(overlap)
    avg_top5 = sum(overlaps) / len(overlaps) if overlaps else 0.0

    # Post-eviction Top-5 overlap
    if isinstance(first_eviction_pos, int) and first_eviction_pos < min_len:
        post_overlaps = overlaps[first_eviction_pos:]
        post_top5 = sum(post_overlaps) / len(post_overlaps) if post_overlaps else 0.0
    else:
        post_top5 = avg_top5

    # Average TBT
    base_tbt = sum(r["tbt_ms"] for r in baseline) / len(baseline)
    tgt_tbt = sum(r["tbt_ms"] for r in target) / len(target)

    return {
        "label": label,
        "total_tokens": min_len,
        "emr": emr,
        "fdt": fdt,
        "suffix_emr": suffix_emr,
        "first_eviction_pos": first_eviction_pos,
        "avg_top5_overlap": avg_top5,
        "post_eviction_top5": post_top5,
        "base_avg_tbt": base_tbt,
        "tgt_avg_tbt": tgt_tbt,
    }

def analyze_scores_csv(path, prompt_len=50):
    """Analyze the score distribution from eviction-time CSV dump."""
    if not path.exists():
        return None

    import csv
    scores = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pos = int(row["position"])
            scores.append({
                "position": pos,
                "score": float(row["score"]),
                "type": "bos" if pos == 0 else ("prompt" if pos < prompt_len else "generated"),
            })

    if not scores:
        return None

    gen_scores = [s for s in scores if s["type"] == "generated"]
    prompt_scores = [s for s in scores if s["type"] == "prompt"]
    bos = [s for s in scores if s["type"] == "bos"]

    if len(gen_scores) < 2:
        return None

    # Correlation: position vs score for generated tokens
    positions = [s["position"] for s in gen_scores]
    score_vals = [s["score"] for s in gen_scores]

    n = len(positions)
    mean_pos = sum(positions) / n
    mean_score = sum(score_vals) / n

    cov = sum((p - mean_pos) * (s - mean_score) for p, s in zip(positions, score_vals)) / n
    std_pos = (sum((p - mean_pos)**2 for p in positions) / n) ** 0.5
    std_score = (sum((s - mean_score)**2 for s in score_vals) / n) ** 0.5

    pearson_r = cov / (std_pos * std_score) if std_pos > 0 and std_score > 0 else 0

    # Score stats by category
    bos_score = bos[0]["score"] if bos else 0
    prompt_mean = sum(s["score"] for s in prompt_scores) / len(prompt_scores) if prompt_scores else 0
    prompt_std = (sum((s["score"] - prompt_mean)**2 for s in prompt_scores) / len(prompt_scores))**0.5 if len(prompt_scores) > 1 else 0

    # Top-10 and bottom-10 generated scores
    gen_sorted = sorted(gen_scores, key=lambda s: s["score"], reverse=True)
    top10 = gen_sorted[:10]
    bottom10 = gen_sorted[-10:]

    return {
        "n_tokens": len(scores),
        "n_generated": len(gen_scores),
        "n_prompt": len(prompt_scores),
        "bos_score": bos_score,
        "prompt_mean": prompt_mean,
        "prompt_std": prompt_std,
        "gen_mean_score": mean_score,
        "gen_std_score": std_score,
        "gen_min_score": min(score_vals),
        "gen_max_score": max(score_vals),
        "pearson_r": pearson_r,
        "top10": top10,
        "bottom10": bottom10,
    }

def print_report(all_metrics, score_analyses):
    """Print formatted comparison report."""
    print("=" * 80)
    print("Round 15: Time-Normalized Scoring — On-Device Experiment Results")
    print("=" * 80)
    print()

    for prompt_label, metrics_list in all_metrics.items():
        print(f"### {prompt_label}")
        print()

        # Header
        print(f"{'Metric':<25} {'H2O-Raw vs Slide':<22} {'H2O-Norm vs Slide':<22}")
        print("-" * 69)

        raw = metrics_list.get("H2O-Raw")
        norm = metrics_list.get("H2O-Norm")

        if raw and norm:
            print(f"{'EMR':<25} {raw['emr']:>8.1%}               {norm['emr']:>8.1%}")
            print(f"{'FDT (pos)':<25} {raw['fdt']:>8d}               {norm['fdt']:>8d}")
            print(f"{'Suffix EMR':<25} {raw['suffix_emr']:>8.1%}               {norm['suffix_emr']:>8.1%}")
            print(f"{'Eviction pos':<25} {str(raw['first_eviction_pos']):>8s}               {str(norm['first_eviction_pos']):>8s}")
            print(f"{'Avg Top-5 Overlap':<25} {raw['avg_top5_overlap']:>8.1%}               {norm['avg_top5_overlap']:>8.1%}")
            print(f"{'Post-Evict Top-5':<25} {raw['post_eviction_top5']:>8.1%}               {norm['post_eviction_top5']:>8.1%}")
            print(f"{'Avg TBT (ms)':<25} {raw['tgt_avg_tbt']:>8.1f}               {norm['tgt_avg_tbt']:>8.1f}")
            print(f"{'Baseline TBT (ms)':<25} {raw['base_avg_tbt']:>8.1f}               {norm['base_avg_tbt']:>8.1f}")
        print()

    # Score analysis
    if score_analyses:
        print("### Score Distribution at Eviction Time (from .scores.csv)")
        print()
        print(f"{'Experiment':<25} {'N_gen':<8} {'BOS':<12} {'Prompt μ':<10} {'Gen μ':<10} {'Gen σ':<10} {'Pearson r':<12}")
        print("-" * 87)
        for label, sa in sorted(score_analyses.items()):
            if sa:
                print(f"{label:<25} {sa['n_generated']:<8d} {sa['bos_score']:<12.1f} {sa['prompt_mean']:<10.3f} {sa['gen_mean_score']:<10.3f} {sa['gen_std_score']:<10.3f} {sa['pearson_r']:<12.4f}")
        print()

        # Detailed: top-10 and bottom-10 for each
        for label, sa in sorted(score_analyses.items()):
            if sa and sa.get("top10"):
                print(f"  {label} — Top-5 generated scores:")
                for s in sa["top10"][:5]:
                    print(f"    pos={s['position']:<6d} score={s['score']:.4f}")
                print(f"  {label} — Bottom-5 generated scores:")
                for s in sa["bottom10"][:5]:
                    print(f"    pos={s['position']:<6d} score={s['score']:.4f}")
                print()

    # Verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    for prompt_label, metrics_list in all_metrics.items():
        raw = metrics_list.get("H2O-Raw")
        norm = metrics_list.get("H2O-Norm")
        if raw and norm:
            raw_emr = raw["suffix_emr"]
            norm_emr = norm["suffix_emr"]
            delta = norm_emr - raw_emr
            direction = "IMPROVED" if delta > 0.01 else ("DEGRADED" if delta < -0.01 else "NO CHANGE")
            print(f"\n{prompt_label}:")
            print(f"  H2O-Raw  Suffix EMR: {raw_emr:.1%}")
            print(f"  H2O-Norm Suffix EMR: {norm_emr:.1%}")
            print(f"  Delta: {delta:+.1%} → {direction}")

            # Compare both against sliding
            print(f"  (Sliding baseline = 100% by definition)")

def main():
    experiments = {
        "PPL-01": {
            "baseline": "SLIDE-PPL01.jsonl",
            "targets": {
                "H2O-Raw": "H2O-RAW-PPL01.jsonl",
                "H2O-Norm": "H2O-NORM-PPL01.jsonl",
            },
            "scores": {
                "H2O-RAW-PPL01": "H2O-RAW-PPL01.scores.csv",
                "H2O-NORM-PPL01": "H2O-NORM-PPL01.scores.csv",
            }
        },
        "PPL-03": {
            "baseline": "SLIDE-PPL03.jsonl",
            "targets": {
                "H2O-Raw": "H2O-RAW-PPL03.jsonl",
                "H2O-Norm": "H2O-NORM-PPL03.jsonl",
            },
            "scores": {
                "H2O-RAW-PPL03": "H2O-RAW-PPL03.scores.csv",
                "H2O-NORM-PPL03": "H2O-NORM-PPL03.scores.csv",
            }
        },
    }

    all_metrics = {}
    all_scores = {}

    for prompt_label, config in experiments.items():
        baseline_path = RESULTS_DIR / config["baseline"]
        if not baseline_path.exists():
            print(f"SKIP {prompt_label}: baseline not found at {baseline_path}")
            continue

        baseline = load_jsonl(baseline_path)
        metrics = {}

        for target_label, target_file in config["targets"].items():
            target_path = RESULTS_DIR / target_file
            if not target_path.exists():
                print(f"SKIP {target_label}: not found at {target_path}")
                continue

            target = load_jsonl(target_path)
            m = compute_metrics(baseline, target, f"{target_label} vs Slide")
            metrics[target_label] = m

        all_metrics[prompt_label] = metrics

        for score_label, score_file in config.get("scores", {}).items():
            score_path = RESULTS_DIR / score_file
            all_scores[score_label] = analyze_scores_csv(score_path)

    print_report(all_metrics, all_scores)

if __name__ == "__main__":
    main()
