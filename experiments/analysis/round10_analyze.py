#!/usr/bin/env python3
"""Round 10: Academic Benchmark Analysis — KV Cache Eviction 정확도 평가

3-Tier 분석:
  Tier 1: Perplexity — EMR, Top-K Overlap, ROUGE-L, BLEU-4, Entropy Ratio
  Tier 2: NIAH — Retrieval Accuracy, Retrieval Score
  Tier 3: QA — F1, Exact Match, ROUGE-L

Usage:
    python experiments/analysis/round10_analyze.py
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.analysis.quality_metrics import (
    compute_baseline_token_rank,
    compute_bleu4,
    compute_emr,
    compute_exact_match,
    compute_f1_score,
    compute_fdt,
    compute_logit_entropy,
    compute_rouge_l,
    compute_topk_overlap,
    evaluate_niah_retrieval,
    load_jsonl,
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "round10"
PROMPTS_FILE = Path(__file__).parent.parent / "prompts" / "benchmark_prompts.json"


def load_prompts():
    with open(PROMPTS_FILE) as f:
        return json.load(f)


def get_generated_text(token_records):
    """Extract generated text from token records."""
    return "".join(r.get("text", "") for r in token_records)


def analyze_ppl_pair(base_file, exp_file, exp_name):
    """Analyze a perplexity baseline vs experiment pair."""
    base_tokens, base_summary = load_jsonl(base_file)
    exp_tokens, exp_summary = load_jsonl(exp_file)

    base_text = get_generated_text(base_tokens)
    exp_text = get_generated_text(exp_tokens)

    emr = compute_emr(base_tokens, exp_tokens)
    fdt = compute_fdt(base_tokens, exp_tokens)
    rouge = compute_rouge_l(base_text, exp_text)
    bleu = compute_bleu4(base_text, exp_text)
    topk = compute_topk_overlap(base_tokens, exp_tokens)
    topk_avg = sum(topk) / len(topk) if topk else 0.0

    # Entropy analysis
    base_entropy = compute_logit_entropy(base_tokens)
    exp_entropy = compute_logit_entropy(exp_tokens)
    min_len = min(len(base_entropy), len(exp_entropy))
    if min_len > 0:
        entropy_ratios = [
            exp_entropy[i] / base_entropy[i] if base_entropy[i] > 0 else 1.0
            for i in range(min_len)
        ]
        avg_entropy_ratio = sum(entropy_ratios) / len(entropy_ratios)
    else:
        avg_entropy_ratio = 1.0

    # Baseline token rank
    ranks = compute_baseline_token_rank(base_tokens, exp_tokens)
    avg_rank = sum(ranks) / len(ranks) if ranks else 1.0
    rank1_pct = ranks.count(1) / len(ranks) * 100 if ranks else 100.0

    return {
        "name": exp_name,
        "tokens": len(exp_tokens),
        "emr": emr,
        "fdt": fdt,
        "rouge_l": rouge["f1"],
        "bleu4": bleu,
        "topk_avg": topk_avg,
        "entropy_ratio": avg_entropy_ratio,
        "avg_base_rank": avg_rank,
        "base_rank1_pct": rank1_pct,
        "evictions": exp_summary.get("eviction_count", 0),
        "evicted_total": exp_summary.get("evicted_tokens_total", 0),
        "avg_tbt": exp_summary.get("avg_tbt_ms", 0),
    }


def analyze_niah(base_file, exp_file, expected_answer, exp_name):
    """Analyze a NIAH baseline vs experiment pair."""
    result = {"name": exp_name, "expected": expected_answer}

    for label, filepath in [("base", base_file), ("exp", exp_file)]:
        if not filepath.exists():
            result[f"{label}_success"] = None
            result[f"{label}_text"] = "(missing)"
            continue

        tokens, summary = load_jsonl(filepath)
        gen_text = get_generated_text(tokens)
        niah = evaluate_niah_retrieval(gen_text, expected_answer)
        result[f"{label}_success"] = niah["success"]
        result[f"{label}_accuracy"] = niah["accuracy"]
        result[f"{label}_score"] = niah["retrieval_score"]
        result[f"{label}_text"] = gen_text[:200]

    return result


def analyze_qa(base_file, exp_file, expected_answer, exp_name, category):
    """Analyze a QA baseline vs experiment pair."""
    result = {"name": exp_name, "category": category, "expected": expected_answer}

    for label, filepath in [("base", base_file), ("exp", exp_file)]:
        if not filepath.exists():
            result[f"{label}_f1"] = None
            continue

        tokens, summary = load_jsonl(filepath)
        gen_text = get_generated_text(tokens)

        f1 = compute_f1_score(gen_text, expected_answer)
        em = compute_exact_match(gen_text, expected_answer)
        rouge = compute_rouge_l(gen_text, expected_answer)

        result[f"{label}_f1"] = f1["f1"]
        result[f"{label}_em"] = em
        result[f"{label}_rouge"] = rouge["f1"]
        result[f"{label}_text"] = gen_text[:200]

    return result


def print_ppl_table(results):
    """Print perplexity results table."""
    print("\n" + "=" * 120)
    print("  TIER 1: PERPLEXITY — Eviction이 토큰 예측 품질에 미치는 영향")
    print("=" * 120)
    print(
        f"  {'ID':<22} {'Tok':>4} {'Evict':>3} {'EMR':>6} {'FDT':>5} "
        f"{'ROUGE':>6} {'BLEU':>6} {'TopK':>6} {'Entropy':>8} "
        f"{'AvgRank':>7} {'Rank1%':>7}"
    )
    print("-" * 120)
    for r in results:
        print(
            f"  {r['name']:<22} {r['tokens']:>4} {r['evicted_total']:>3} "
            f"{r['emr']:>6.3f} {r['fdt']:>5} "
            f"{r['rouge_l']:>6.3f} {r['bleu4']:>6.3f} {r['topk_avg']:>6.3f} "
            f"{r['entropy_ratio']:>8.3f} "
            f"{r['avg_base_rank']:>7.2f} {r['base_rank1_pct']:>6.1f}%"
        )


def print_niah_table(results):
    """Print NIAH results table."""
    print("\n" + "=" * 120)
    print("  TIER 2: NIAH — Eviction이 정보 검색 정확도에 미치는 영향")
    print("=" * 120)
    print(
        f"  {'ID':<30} {'Expected':<20} "
        f"{'Base':>6} {'Exp':>6} {'BaseScore':>9} {'ExpScore':>8}"
    )
    print("-" * 120)
    for r in results:
        base_ok = "PASS" if r.get("base_success") else ("FAIL" if r.get("base_success") is not None else "N/A")
        exp_ok = "PASS" if r.get("exp_success") else ("FAIL" if r.get("exp_success") is not None else "N/A")
        base_score = r.get("base_score", 0)
        exp_score = r.get("exp_score", 0)
        print(
            f"  {r['name']:<30} {r['expected']:<20} "
            f"{base_ok:>6} {exp_ok:>6} "
            f"{base_score:>9.3f} {exp_score:>8.3f}"
        )


def print_qa_table(results):
    """Print QA results table."""
    print("\n" + "=" * 120)
    print("  TIER 3: QA — Eviction이 태스크 수행에 미치는 영향")
    print("=" * 120)
    print(
        f"  {'ID':<20} {'Category':<15} "
        f"{'Base F1':>8} {'Exp F1':>8} {'Delta':>7} "
        f"{'Base EM':>8} {'Exp EM':>8}"
    )
    print("-" * 120)
    for r in results:
        base_f1 = r.get("base_f1", 0) or 0
        exp_f1 = r.get("exp_f1", 0) or 0
        delta = exp_f1 - base_f1
        base_em = r.get("base_em", 0) or 0
        exp_em = r.get("exp_em", 0) or 0
        print(
            f"  {r['name']:<20} {r['category']:<15} "
            f"{base_f1:>8.3f} {exp_f1:>8.3f} {delta:>+7.3f} "
            f"{base_em:>8} {exp_em:>8}"
        )


def main():
    if not RESULTS_DIR.exists():
        print(f"Error: {RESULTS_DIR} does not exist. Run round10 experiments first.")
        sys.exit(1)

    prompts_data = load_prompts()

    # ── Tier 1: Perplexity ──
    ppl_results = []
    for tokens in [512, 1024]:
        inject_pos = tokens // 2
        for ppl_id in ["PPL01", "PPL02", "PPL03", "PPL04", "PPL05"]:
            base_file = RESULTS_DIR / f"{ppl_id}-{tokens}-base.jsonl"
            if not base_file.exists():
                continue

            for policy in ["sl", "h2o"]:
                exp_file = RESULTS_DIR / f"{ppl_id}-{tokens}-{policy}.jsonl"
                if not exp_file.exists():
                    continue

                try:
                    r = analyze_ppl_pair(
                        base_file, exp_file,
                        f"{ppl_id}-{tokens}-{policy}"
                    )
                    ppl_results.append(r)
                except Exception as e:
                    print(f"  ERROR analyzing {ppl_id}-{tokens}-{policy}: {e}")

    if ppl_results:
        print_ppl_table(ppl_results)

        # Summary stats
        sl_results = [r for r in ppl_results if r["name"].endswith("-sl")]
        h2o_results = [r for r in ppl_results if r["name"].endswith("-h2o")]

        if sl_results:
            avg_emr_sl = sum(r["emr"] for r in sl_results) / len(sl_results)
            avg_topk_sl = sum(r["topk_avg"] for r in sl_results) / len(sl_results)
            print(f"\n  Sliding 평균: EMR={avg_emr_sl:.3f}, TopK={avg_topk_sl:.3f}")

        if h2o_results:
            avg_emr_h2o = sum(r["emr"] for r in h2o_results) / len(h2o_results)
            avg_topk_h2o = sum(r["topk_avg"] for r in h2o_results) / len(h2o_results)
            print(f"  H2O 평균:    EMR={avg_emr_h2o:.3f}, TopK={avg_topk_h2o:.3f}")

    # ── Tier 2: NIAH ──
    niah_needles = {
        "PASS": "58291",
        "FACT": "Crescentport",
    }

    niah_results_sl = []
    niah_results_h2o = []
    for needle_key, expected in niah_needles.items():
        for depth_pct in [10, 25, 50, 75, 90]:
            for blocks in [4, 8]:
                niah_id = f"NIAH-{needle_key}-D{depth_pct}-B{blocks}"
                base_file = RESULTS_DIR / f"{niah_id}-base.jsonl"

                for policy in ["sl", "h2o"]:
                    exp_file = RESULTS_DIR / f"{niah_id}-{policy}.jsonl"
                    if not base_file.exists() or not exp_file.exists():
                        continue

                    try:
                        r = analyze_niah(
                            base_file, exp_file, expected,
                            f"{niah_id}-{policy}"
                        )
                        if policy == "sl":
                            niah_results_sl.append(r)
                        else:
                            niah_results_h2o.append(r)
                    except Exception as e:
                        print(f"  ERROR analyzing {niah_id}-{policy}: {e}")

    all_niah = []
    # Interleave sl and h2o for the same prompt
    for sl_r, h2o_r in zip(niah_results_sl, niah_results_h2o):
        all_niah.append(sl_r)
        all_niah.append(h2o_r)

    if all_niah:
        print_niah_table(all_niah)

        # Summary
        sl_pass = sum(1 for r in niah_results_sl if r.get("exp_success"))
        h2o_pass = sum(1 for r in niah_results_h2o if r.get("exp_success"))
        sl_total = len(niah_results_sl)
        h2o_total = len(niah_results_h2o)
        base_pass = sum(1 for r in niah_results_sl if r.get("base_success"))

        print(f"\n  Baseline 검색 성공: {base_pass}/{sl_total}")
        print(f"  Sliding 검색 성공:  {sl_pass}/{sl_total} ({sl_pass/sl_total*100:.0f}%)" if sl_total else "")
        print(f"  H2O 검색 성공:     {h2o_pass}/{h2o_total} ({h2o_pass/h2o_total*100:.0f}%)" if h2o_total else "")

    # ── Tier 3: QA ──
    qa_expected = {
        "QA-SD01": ("single_doc_qa", "The United States took over the project in 1904 and opened the canal on August 15, 1914."),
        "QA-SD02": ("single_doc_qa", "The light-dependent reactions take place in the thylakoid membranes of the chloroplast, producing oxygen, ATP, and NADPH."),
        "QA-SD03": ("single_doc_qa", "The Medici family funded artists in Florence. Three key figures were Leonardo da Vinci, Michelangelo, and Raphael."),
        "QA-SUM01": ("summarization", "Earth lost 28 trillion tonnes of ice from 1994 to 2017, with the annual rate increasing from 0.8 to 1.3 trillion tonnes. Antarctic and Greenland ice sheet losses have increased sixfold since the 1990s, matching worst-case climate predictions."),
        "QA-FS01": ("few_shot", "positive"),
        "QA-MH01": ("multi_hop", "The Nobel Prize is presented in Stockholm. The ceremony is held on December 10, the anniversary of Alfred Nobel's death in 1896."),
    }

    qa_results_sl = []
    qa_results_h2o = []
    for qa_id, (category, expected) in qa_expected.items():
        base_file = RESULTS_DIR / f"{qa_id}-base.jsonl"

        for policy in ["sl", "h2o"]:
            exp_file = RESULTS_DIR / f"{qa_id}-{policy}.jsonl"
            if not base_file.exists() or not exp_file.exists():
                continue

            try:
                r = analyze_qa(
                    base_file, exp_file, expected,
                    f"{qa_id}-{policy}", category
                )
                if policy == "sl":
                    qa_results_sl.append(r)
                else:
                    qa_results_h2o.append(r)
            except Exception as e:
                print(f"  ERROR analyzing {qa_id}-{policy}: {e}")

    all_qa = []
    for sl_r, h2o_r in zip(qa_results_sl, qa_results_h2o):
        all_qa.append(sl_r)
        all_qa.append(h2o_r)

    if all_qa:
        print_qa_table(all_qa)

        if qa_results_sl:
            avg_f1_sl = sum((r.get("exp_f1") or 0) for r in qa_results_sl) / len(qa_results_sl)
            avg_f1_base = sum((r.get("base_f1") or 0) for r in qa_results_sl) / len(qa_results_sl)
            print(f"\n  Baseline 평균 F1: {avg_f1_base:.3f}")
            print(f"  Sliding 평균 F1:  {avg_f1_sl:.3f} (Δ{avg_f1_sl - avg_f1_base:+.3f})")

        if qa_results_h2o:
            avg_f1_h2o = sum((r.get("exp_f1") or 0) for r in qa_results_h2o) / len(qa_results_h2o)
            print(f"  H2O 평균 F1:     {avg_f1_h2o:.3f} (Δ{avg_f1_h2o - avg_f1_base:+.3f})")

    # ── Overall Summary ──
    print("\n" + "=" * 120)
    print("  종합 요약")
    print("=" * 120)

    # Print generated text samples for inspection
    print("\n  ── 생성 텍스트 샘플 (QA Baseline) ──")
    for qa_id in ["QA-SD01", "QA-FS01", "QA-MH01"]:
        base_file = RESULTS_DIR / f"{qa_id}-base.jsonl"
        if base_file.exists():
            try:
                tokens, _ = load_jsonl(base_file)
                text = get_generated_text(tokens)
                print(f"\n  {qa_id}: {text[:300]}")
            except Exception:
                pass

    print("\n  ── 생성 텍스트 샘플 (NIAH) ──")
    for niah_id in ["NIAH-PASS-D25-B4", "NIAH-PASS-D50-B8"]:
        for policy in ["base", "sl", "h2o"]:
            f = RESULTS_DIR / f"{niah_id}-{policy}.jsonl"
            if f.exists():
                try:
                    tokens, _ = load_jsonl(f)
                    text = get_generated_text(tokens)
                    print(f"\n  {niah_id}-{policy}: {text[:200]}")
                except Exception:
                    pass


if __name__ == "__main__":
    main()
