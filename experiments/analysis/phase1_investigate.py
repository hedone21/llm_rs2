#!/usr/bin/env python3
"""Phase 1: Token-level divergence analysis for EMR non-linearity investigation.

Analyzes:
1. Token-level match/mismatch pattern (position-by-position)
2. Baseline content around FDT (what text is being generated)
3. H2O vs Sliding evicted token comparison (do they evict the same tokens?)
4. Post-eviction cache_pos analysis
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quality_metrics import load_jsonl, compute_fdt, compute_emr

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

EXPERIMENTS = [
    # (label, seq_len, inject_pos, baseline, h2o, sliding)
    ("2K", 2048, 1024, "B-2048", "L-2k-h2o", "L-2k-sl"),
    ("4K", 4096, 2048, "B-4096", "L-4k-h2o", "L-4k-sl"),
    ("8K", 8192, 4096, "B-8192", "L-8k-h2o", "L-8k-sl"),
]


def load(name):
    path = os.path.join(RESULTS_DIR, f"{name}.jsonl")
    return load_jsonl(path)


def section(title):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()


def analysis_1_token_match_pattern():
    """Token-level match/mismatch in sliding windows."""
    section("Analysis 1: Token Match Pattern (windowed)")

    for label, seq_len, inject_pos, base_name, h2o_name, sl_name in EXPERIMENTS:
        base_recs, _ = load(base_name)
        h2o_recs, _ = load(h2o_name)
        sl_recs, _ = load(sl_name)

        min_len = min(len(base_recs), len(h2o_recs), len(sl_recs))

        h2o_fdt = compute_fdt(base_recs, h2o_recs)
        sl_fdt = compute_fdt(base_recs, sl_recs)

        # Compute per-position match
        h2o_match = [
            base_recs[i]["token_id"] == h2o_recs[i]["token_id"]
            for i in range(min_len)
        ]
        sl_match = [
            base_recs[i]["token_id"] == sl_recs[i]["token_id"]
            for i in range(min_len)
        ]

        print(f"── {label} (seq={seq_len}, inject@{inject_pos}) ──")
        print(f"  H2O FDT={h2o_fdt}, Sliding FDT={sl_fdt}")
        print(f"  Total compared tokens: {min_len}")
        print()

        # Window analysis: 100-token windows
        window = 100
        print(f"  {'Window':>12} │ {'H2O match%':>10} {'SL match%':>10} │ {'Phase':>20}")
        print(f"  {'─' * 12}─┼─{'─' * 10}─{'─' * 10}─┼─{'─' * 20}")

        for start in range(0, min_len, window):
            end = min(start + window, min_len)
            h2o_pct = sum(h2o_match[start:end]) / (end - start) * 100
            sl_pct = sum(sl_match[start:end]) / (end - start) * 100

            # Phase label
            if end <= inject_pos:
                phase = "pre-eviction"
            elif start < inject_pos < end:
                phase = "EVICTION POINT"
            elif start < h2o_fdt < end:
                phase = "FDT (diverge)"
            elif start >= h2o_fdt:
                phase = "post-FDT"
            else:
                phase = "post-evict/pre-FDT"

            marker = ""
            if h2o_pct < 100 and start < h2o_fdt:
                marker = " ◀"

            print(
                f"  {start:>5}-{end:>5} │ {h2o_pct:>9.1f}% {sl_pct:>9.1f}% │ {phase:>20}{marker}"
            )

        print()

        # Post-FDT reconvergence check
        post_fdt_len = min_len - max(h2o_fdt, sl_fdt)
        if post_fdt_len > 0:
            fdt_max = max(h2o_fdt, sl_fdt)
            h2o_post = sum(h2o_match[fdt_max:]) / post_fdt_len * 100
            sl_post = sum(sl_match[fdt_max:]) / post_fdt_len * 100
            print(
                f"  Post-FDT reconvergence: H2O={h2o_post:.1f}%, SL={sl_post:.1f}% "
                f"(over {post_fdt_len} tokens)"
            )

        # Fine-grained around FDT
        fdt = min(h2o_fdt, sl_fdt)
        print(f"\n  Fine-grained around FDT ({fdt}):")
        for offset in [-20, -10, -5, 0, 5, 10, 20, 50, 100]:
            pos = fdt + offset
            if 0 <= pos < min_len:
                h = "✓" if h2o_match[pos] else "✗"
                s = "✓" if sl_match[pos] else "✗"
                print(f"    pos {pos:>6} (FDT{offset:+d}): H2O={h} SL={s}")
        print()


def analysis_2_baseline_content_around_fdt():
    """Show baseline text around FDT to understand content context."""
    section("Analysis 2: Baseline Content Around FDT")

    for label, seq_len, inject_pos, base_name, h2o_name, _ in EXPERIMENTS:
        base_recs, _ = load(base_name)
        h2o_recs, _ = load(h2o_name)

        fdt = compute_fdt(base_recs, h2o_recs)

        print(f"── {label} (inject@{inject_pos}, FDT={fdt}) ──")

        # Show text around eviction point
        print(f"\n  [Around eviction point @{inject_pos}]:")
        start = max(0, inject_pos - 10)
        end = min(len(base_recs), inject_pos + 10)
        text = "".join(base_recs[i].get("text", "") for i in range(start, end))
        print(f"  ...{text}...")

        # Show text around FDT
        print(f"\n  [Around FDT @{fdt}]:")
        start = max(0, fdt - 15)
        end = min(len(base_recs), fdt + 15)
        for i in range(start, end):
            marker = ">>>" if i == fdt else "   "
            base_text = base_recs[i].get("text", "")
            if i < len(h2o_recs):
                exp_text = h2o_recs[i].get("text", "")
                match = "✓" if base_recs[i]["token_id"] == h2o_recs[i]["token_id"] else "✗"
            else:
                exp_text = "N/A"
                match = "?"
            print(
                f"  {marker} pos {i:>6}: {match} "
                f"base={repr(base_text):>15} exp={repr(exp_text):>15}"
            )
        print()

        # Show full text around FDT (30 tokens before, 30 after)
        print(f"  [Full baseline text around FDT: {fdt-30}..{fdt+30}]")
        start = max(0, fdt - 30)
        end = min(len(base_recs), fdt + 30)
        text = "".join(base_recs[i].get("text", "") for i in range(start, end))
        print(f"  {text}")
        print()


def analysis_3_h2o_vs_sliding_evicted():
    """Compare which tokens H2O and Sliding keep after eviction."""
    section("Analysis 3: H2O vs Sliding — Evicted Token Comparison")

    for label, seq_len, inject_pos, base_name, h2o_name, sl_name in EXPERIMENTS:
        base_recs, _ = load(base_name)
        h2o_recs, h2o_sum = load(h2o_name)
        sl_recs, sl_sum = load(sl_name)

        print(f"── {label} (inject@{inject_pos}) ──")
        print(f"  H2O evicted: {h2o_sum.get('evicted_tokens_total', 0)}")
        print(f"  SL  evicted: {sl_sum.get('evicted_tokens_total', 0)}")

        # Find eviction point: look for cache_pos drop
        h2o_evict_idx = None
        sl_evict_idx = None

        for i in range(1, min(len(h2o_recs), len(sl_recs))):
            if h2o_evict_idx is None and h2o_recs[i].get("cache_pos", 0) < h2o_recs[i - 1].get("cache_pos", 0):
                h2o_evict_idx = i
            if sl_evict_idx is None and sl_recs[i].get("cache_pos", 0) < sl_recs[i - 1].get("cache_pos", 0):
                sl_evict_idx = i

        print(f"  H2O eviction at record idx: {h2o_evict_idx}")
        print(f"  SL  eviction at record idx: {sl_evict_idx}")

        if h2o_evict_idx:
            pre = h2o_recs[h2o_evict_idx - 1].get("cache_pos", 0)
            post = h2o_recs[h2o_evict_idx].get("cache_pos", 0)
            print(f"  H2O cache_pos: {pre} → {post} (Δ={post - pre})")
        if sl_evict_idx:
            pre = sl_recs[sl_evict_idx - 1].get("cache_pos", 0)
            post = sl_recs[sl_evict_idx].get("cache_pos", 0)
            print(f"  SL  cache_pos: {pre} → {post} (Δ={post - pre})")

        # Compare token sequences right after eviction
        # The tokens generated immediately after eviction should diverge
        # if different tokens were kept in cache
        if h2o_evict_idx and sl_evict_idx:
            print(f"\n  Post-eviction token comparison (H2O vs Sliding):")
            fdt_h2o_sl = None
            match_count = 0
            compare_len = min(100, len(h2o_recs) - h2o_evict_idx, len(sl_recs) - sl_evict_idx)
            for j in range(compare_len):
                hi = h2o_evict_idx + j
                si = sl_evict_idx + j
                if h2o_recs[hi]["token_id"] == sl_recs[si]["token_id"]:
                    match_count += 1
                elif fdt_h2o_sl is None:
                    fdt_h2o_sl = j

            total_compare = min(len(h2o_recs) - h2o_evict_idx, len(sl_recs) - sl_evict_idx)
            all_match = sum(
                1
                for j in range(total_compare)
                if h2o_recs[h2o_evict_idx + j]["token_id"]
                == sl_recs[sl_evict_idx + j]["token_id"]
            )
            print(f"  H2O vs SL first 100 tokens after eviction: {match_count}/{compare_len} match")
            print(f"  H2O vs SL all tokens after eviction: {all_match}/{total_compare} match ({all_match/total_compare*100:.1f}%)")
            if fdt_h2o_sl is not None:
                print(f"  H2O-SL first divergence: {fdt_h2o_sl} tokens after eviction")
            else:
                print(f"  H2O-SL: IDENTICAL for first {compare_len} tokens")
        print()


def analysis_4_cache_pos_trajectory():
    """Analyze cache_pos over time to understand eviction mechanics."""
    section("Analysis 4: Cache Position Trajectory")

    for label, seq_len, inject_pos, base_name, h2o_name, sl_name in EXPERIMENTS:
        h2o_recs, _ = load(h2o_name)
        sl_recs, _ = load(sl_name)

        print(f"── {label} (inject@{inject_pos}) ──")

        # Sample cache_pos at regular intervals
        print(f"  {'Pos':>6} │ {'H2O cache':>10} {'SL cache':>10} │ {'Δ(H2O-SL)':>10}")
        print(f"  {'─' * 6}─┼─{'─' * 10}─{'─' * 10}─┼─{'─' * 10}")

        sample_points = list(range(0, min(len(h2o_recs), len(sl_recs)), max(1, seq_len // 20)))
        for i in sample_points:
            h2o_cp = h2o_recs[i].get("cache_pos", 0)
            sl_cp = sl_recs[i].get("cache_pos", 0)
            delta = h2o_cp - sl_cp
            marker = " ◀" if delta != 0 else ""
            print(f"  {i:>6} │ {h2o_cp:>10} {sl_cp:>10} │ {delta:>+10}{marker}")
        print()


def analysis_5_topk_divergence():
    """Check if top-K logit distributions diverge before token mismatch."""
    section("Analysis 5: Top-K Logit Divergence Before FDT")

    for label, seq_len, inject_pos, base_name, h2o_name, _ in EXPERIMENTS:
        base_recs, _ = load(base_name)
        h2o_recs, _ = load(h2o_name)

        fdt = compute_fdt(base_recs, h2o_recs)
        min_len = min(len(base_recs), len(h2o_recs))

        print(f"── {label} (inject@{inject_pos}, FDT={fdt}) ──")

        # Check top-K overlap in the region between eviction and FDT
        start = inject_pos
        end = min(fdt + 20, min_len)

        # Sample points
        if fdt - inject_pos > 40:
            points = list(range(start, min(start + 10, end)))
            points += list(range(max(start + 10, fdt - 10), min(fdt + 10, end)))
        else:
            points = list(range(start, end))

        print(f"  {'Pos':>6} │ {'Top-K Overlap':>13} {'Token Match':>11} │ {'Top1 base':>15} {'Top1 exp':>15}")
        print(f"  {'─' * 6}─┼─{'─' * 13}─{'─' * 11}─┼─{'─' * 15}─{'─' * 15}")

        for i in points:
            if i >= min_len:
                break
            base_logits = base_recs[i].get("top_logits", [])
            exp_logits = h2o_recs[i].get("top_logits", [])

            base_ids = set(e[0] for e in base_logits[:10])
            exp_ids = set(e[0] for e in exp_logits[:10])

            if base_ids and exp_ids:
                overlap = len(base_ids & exp_ids) / 10
            else:
                overlap = 0.0

            match = "✓" if base_recs[i]["token_id"] == h2o_recs[i]["token_id"] else "✗"

            top1_base = f"{base_logits[0][1]:.2f}" if base_logits else "N/A"
            top1_exp = f"{exp_logits[0][1]:.2f}" if exp_logits else "N/A"

            print(f"  {i:>6} │ {overlap:>13.1%} {match:>11} │ {top1_base:>15} {exp_logits[0][1] if exp_logits else 0:>15.2f}")

        # Overall top-K overlap between eviction and FDT
        if fdt > inject_pos:
            total_overlap = 0
            count = 0
            for i in range(inject_pos, fdt):
                if i >= min_len:
                    break
                base_logits = base_recs[i].get("top_logits", [])
                exp_logits = h2o_recs[i].get("top_logits", [])
                base_ids = set(e[0] for e in base_logits[:10])
                exp_ids = set(e[0] for e in exp_logits[:10])
                if base_ids and exp_ids:
                    total_overlap += len(base_ids & exp_ids) / 10
                    count += 1
            if count > 0:
                print(f"\n  Avg top-K overlap (eviction→FDT): {total_overlap / count:.1%} over {count} tokens")
        print()


def analysis_6_emr_sensitivity():
    """Break down EMR by segment: pre-eviction, eviction-to-FDT, post-FDT."""
    section("Analysis 6: EMR by Segment")

    print(f"{'Label':>5} │ {'Inject':>7} {'FDT':>6} │ "
          f"{'Pre-evict':>10} {'Evict→FDT':>10} {'Post-FDT':>10} │ "
          f"{'Frac pre':>9} {'Frac mid':>9} {'Frac post':>9}")
    print("─" * 100)

    for label, seq_len, inject_pos, base_name, h2o_name, _ in EXPERIMENTS:
        base_recs, _ = load(base_name)
        h2o_recs, _ = load(h2o_name)

        min_len = min(len(base_recs), len(h2o_recs))
        fdt = compute_fdt(base_recs, h2o_recs)

        # Segment 1: pre-eviction (0..inject_pos)
        seg1_end = min(inject_pos, min_len)
        seg1_match = sum(
            1 for i in range(seg1_end)
            if base_recs[i]["token_id"] == h2o_recs[i]["token_id"]
        )
        seg1_emr = seg1_match / seg1_end if seg1_end > 0 else 1.0

        # Segment 2: eviction-to-FDT (inject_pos..fdt)
        seg2_start = seg1_end
        seg2_end = min(fdt, min_len)
        seg2_len = seg2_end - seg2_start
        seg2_match = sum(
            1 for i in range(seg2_start, seg2_end)
            if base_recs[i]["token_id"] == h2o_recs[i]["token_id"]
        )
        seg2_emr = seg2_match / seg2_len if seg2_len > 0 else 1.0

        # Segment 3: post-FDT (fdt..end)
        seg3_start = seg2_end
        seg3_len = min_len - seg3_start
        seg3_match = sum(
            1 for i in range(seg3_start, min_len)
            if base_recs[i]["token_id"] == h2o_recs[i]["token_id"]
        )
        seg3_emr = seg3_match / seg3_len if seg3_len > 0 else 1.0

        # Fractions
        frac_pre = seg1_end / min_len
        frac_mid = seg2_len / min_len
        frac_post = seg3_len / min_len

        print(
            f"{label:>5} │ {inject_pos:>7} {fdt:>6} │ "
            f"{seg1_emr:>9.1%} {seg2_emr:>9.1%} {seg3_emr:>9.1%} │ "
            f"{frac_pre:>9.1%} {frac_mid:>9.1%} {frac_post:>9.1%}"
        )

    print()
    print("  Key: Pre-evict should be 100% (no eviction yet).")
    print("       Evict→FDT should be 100% (matches by definition).")
    print("       Post-FDT is where divergence happens.")
    print("       Frac = fraction of total tokens in each segment.")


def analysis_7_gap_analysis():
    """Quantitative summary of the FDT gap mystery."""
    section("Analysis 7: FDT Gap Summary & Hypotheses")

    for label, seq_len, inject_pos, base_name, h2o_name, sl_name in EXPERIMENTS:
        base_recs, _ = load(base_name)
        h2o_recs, h2o_sum = load(h2o_name)
        sl_recs, _ = load(sl_name)

        min_len = min(len(base_recs), len(h2o_recs))
        h2o_fdt = compute_fdt(base_recs, h2o_recs)
        sl_fdt = compute_fdt(base_recs, sl_recs)

        evicted = h2o_sum.get("evicted_tokens_total", 0)
        cache_after = h2o_sum.get("final_cache_pos", 0) - (min_len - inject_pos)  # approximate
        remaining_tokens = min_len - inject_pos
        fdt_gap = h2o_fdt - inject_pos

        print(f"── {label} ──")
        print(f"  Sequence length: {min_len}")
        print(f"  Inject position: {inject_pos} ({inject_pos/min_len*100:.0f}%)")
        print(f"  Tokens evicted:  {evicted}")
        print(f"  Tokens remaining after eviction: {remaining_tokens}")
        print(f"  H2O FDT:         {h2o_fdt}")
        print(f"  SL  FDT:         {sl_fdt}")
        print(f"  FDT gap (H2O):   {fdt_gap} tokens ({fdt_gap/remaining_tokens*100:.1f}% of remaining)")
        print(f"  FDT gap (SL):    {sl_fdt - inject_pos} tokens")
        print(f"  FDT match:       {'YES — same FDT' if h2o_fdt == sl_fdt else 'NO — different FDT'}")
        print()

    # Key observation
    print("  KEY: If H2O and Sliding have the same FDT, they keep the same tokens")
    print("  → The non-linearity is about the MODEL'S sensitivity to context loss,")
    print("    not about the EVICTION STRATEGY.")


def main():
    print("╔" + "═" * 78 + "╗")
    print("║  Phase 1: Token-Level Divergence Investigation" + " " * 31 + "║")
    print("╚" + "═" * 78 + "╝")

    analysis_1_token_match_pattern()
    analysis_2_baseline_content_around_fdt()
    analysis_3_h2o_vs_sliding_evicted()
    analysis_4_cache_pos_trajectory()
    analysis_5_topk_divergence()
    analysis_6_emr_sensitivity()
    analysis_7_gap_analysis()


if __name__ == "__main__":
    main()
