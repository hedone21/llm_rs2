#!/usr/bin/env python3
"""Analyze proxy validation results and compute Spearman rank correlation."""
import json, glob, sys, os
from pathlib import Path

def analyze_model(model_dir: str, model_name: str):
    """Analyze results for one model size."""
    baseline_path = f"{model_dir}/baseline.json"
    if not os.path.exists(baseline_path):
        print(f"  [SKIP] No baseline found for {model_name}")
        return None

    baseline = json.load(open(baseline_path))
    baseline_ppl = baseline["ppl"]
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}  |  Baseline PPL: {baseline_ppl:.4f}")
    print(f"{'='*60}")

    results = {}
    for policy in ["sliding", "h2o"]:
        policy_results = []
        for f in sorted(glob.glob(f"{model_dir}/{policy}_*.json")):
            try:
                data = json.load(open(f))
            except json.JSONDecodeError:
                continue
            budget = data["config"]["kv_budget"]
            ppl = data["ppl"]
            ppl_increase = ppl - baseline_ppl
            n_evictions = data["eviction_count"]
            proxies = data["proxy_metrics"]
            avg_proxy = sum(m["raw_value"] for m in proxies) / max(len(proxies), 1)
            total_proxy = sum(m["raw_value"] for m in proxies)
            policy_results.append({
                "budget": budget,
                "ppl": ppl,
                "ppl_increase": ppl_increase,
                "evictions": n_evictions,
                "avg_proxy": avg_proxy,
                "total_proxy": total_proxy,
            })

        if not policy_results:
            continue

        # Sort by budget descending (least aggressive first)
        policy_results.sort(key=lambda x: -x["budget"])

        print(f"\n  --- {policy.upper()} ---")
        print(f"  {'Budget':>8} {'PPL':>10} {'ΔPPL':>10} {'Evictions':>10} {'Avg Proxy':>12} {'Total Proxy':>12}")
        print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
        for r in policy_results:
            print(f"  {r['budget']:>8} {r['ppl']:>10.4f} {r['ppl_increase']:>10.4f} {r['evictions']:>10} {r['avg_proxy']:>12.6f} {r['total_proxy']:>12.6f}")

        # Spearman rank correlation
        ppl_increases = [r["ppl_increase"] for r in policy_results]
        avg_proxies = [r["avg_proxy"] for r in policy_results]
        total_proxies = [r["total_proxy"] for r in policy_results]

        rho_avg = spearman_rho(avg_proxies, ppl_increases)
        rho_total = spearman_rho(total_proxies, ppl_increases)

        print(f"\n  Spearman ρ (avg_proxy vs ΔPPL):   {rho_avg:+.4f}")
        print(f"  Spearman ρ (total_proxy vs ΔPPL): {rho_total:+.4f}")

        verdict_avg = "EXCELLENT" if abs(rho_avg) >= 0.85 else ("GOOD" if abs(rho_avg) >= 0.70 else "INSUFFICIENT")
        verdict_total = "EXCELLENT" if abs(rho_total) >= 0.85 else ("GOOD" if abs(rho_total) >= 0.70 else "INSUFFICIENT")
        print(f"  Verdict (avg):   {verdict_avg}")
        print(f"  Verdict (total): {verdict_total}")

        results[policy] = {
            "data": policy_results,
            "rho_avg": rho_avg,
            "rho_total": rho_total,
        }

    return {"baseline_ppl": baseline_ppl, "policies": results}


def spearman_rho(x, y):
    """Compute Spearman rank correlation without scipy."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = rank(x)
    ry = rank(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def rank(data):
    """Assign ranks (1-based, average for ties)."""
    indexed = sorted(enumerate(data), key=lambda x: x[1])
    ranks = [0.0] * len(data)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def main():
    base_dir = "experiments/proxy_validation/results"
    all_results = {}

    for model in ["1b", "3b"]:
        model_dir = f"{base_dir}/{model}"
        if os.path.isdir(model_dir):
            result = analyze_model(model_dir, f"Llama 3.2 {model.upper()}")
            if result:
                all_results[model] = result

    if not all_results:
        print("No results found.")
        return

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for model, data in all_results.items():
        print(f"\n  {model.upper()} (baseline PPL={data['baseline_ppl']:.4f}):")
        for policy, pdata in data["policies"].items():
            print(f"    {policy}: ρ_avg={pdata['rho_avg']:+.4f}, ρ_total={pdata['rho_total']:+.4f}")


if __name__ == "__main__":
    main()
