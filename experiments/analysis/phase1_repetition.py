#!/usr/bin/env python3
"""Phase 1 supplementary: Detect repetition patterns in baseline generation.

Hypothesis: The model generates repetitive text in a cycle, and the FDT
occurs at slight variation points within the cycle. This would explain
the non-linear EMR pattern.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quality_metrics import load_jsonl, compute_fdt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load(name):
    path = os.path.join(RESULTS_DIR, f"{name}.jsonl")
    return load_jsonl(path)


def find_repetition_cycle(tokens, window=50, search_range=2000):
    """Find the repetition cycle length in a token sequence.

    Looks for the shortest period P where tokens[i] == tokens[i+P]
    for a significant stretch.
    """
    n = len(tokens)
    best_period = None
    best_match_rate = 0

    for period in range(100, min(search_range, n // 2)):
        matches = 0
        compared = 0
        for i in range(min(window * 3, n - period)):
            if tokens[i] == tokens[i + period]:
                matches += 1
            compared += 1

        if compared > 0:
            rate = matches / compared
            if rate > 0.9 and (best_period is None or rate > best_match_rate):
                best_period = period
                best_match_rate = rate
                # Found a good match, try nearby periods
                break

    # Fine-tune: check periods near the best
    if best_period:
        for delta in range(-20, 21):
            p = best_period + delta
            if p < 50 or p >= n // 2:
                continue
            matches = 0
            compared = 0
            for i in range(min(500, n - p)):
                if tokens[i] == tokens[i + p]:
                    matches += 1
                compared += 1
            rate = matches / compared if compared > 0 else 0
            if rate > best_match_rate:
                best_period = p
                best_match_rate = rate

    return best_period, best_match_rate


def analyze_repetition():
    """Detect and display repetition patterns in baseline sequences."""
    print("=" * 80)
    print("  Repetition Pattern Detection")
    print("=" * 80)
    print()

    baselines = [
        ("B-2048", 2048),
        ("B-4096", 4096),
        ("B-8192", 8192),
    ]

    for base_name, seq_len in baselines:
        recs, _ = load(base_name)
        tokens = [r["token_id"] for r in recs]
        texts = [r.get("text", "") for r in recs]

        print(f"── {base_name} ({len(tokens)} tokens) ──")

        # Find repetition cycle
        period, rate = find_repetition_cycle(tokens)
        if period:
            print(f"  Repetition period: {period} tokens (match rate: {rate:.1%})")

            # Show where the cycle starts repeating
            # Check alignment of each cycle
            cycles = len(tokens) // period
            print(f"  Number of full cycles: {cycles}")

            for c in range(min(cycles, 6)):
                start = c * period
                end = min(start + period, len(tokens))
                # Match rate with first cycle
                if c > 0:
                    matches = sum(
                        1
                        for i in range(min(period, len(tokens) - start))
                        if start + i < len(tokens)
                        and tokens[i] == tokens[start + i]
                    )
                    rate_c = matches / min(period, len(tokens) - start)
                    # Find first mismatch in this cycle
                    first_mm = None
                    for i in range(min(period, len(tokens) - start)):
                        if start + i < len(tokens) and tokens[i] != tokens[start + i]:
                            first_mm = i
                            break
                    mm_text = f", first mismatch at offset {first_mm}" if first_mm is not None else ""
                    print(f"    Cycle {c}: pos {start}-{end} vs cycle 0: {rate_c:.1%}{mm_text}")

            # Show text of one cycle
            cycle_text = "".join(texts[:min(period, 200)])
            if len(cycle_text) > 500:
                cycle_text = cycle_text[:500] + "..."
            print(f"\n  First cycle text ({min(period, 200)} tokens):")
            print(f"  {cycle_text}")
        else:
            print("  No clear repetition detected")

            # Try smaller windows
            print("  Trying n-gram repetition detection...")
            for n in [20, 30, 50]:
                # Find repeated n-grams
                ngram_positions = {}
                for i in range(len(tokens) - n):
                    key = tuple(tokens[i : i + n])
                    if key not in ngram_positions:
                        ngram_positions[key] = []
                    ngram_positions[key].append(i)

                repeated = {
                    k: v for k, v in ngram_positions.items() if len(v) > 1
                }
                if repeated:
                    # Find the most repeated n-gram
                    most_repeated = max(repeated.items(), key=lambda x: len(x[1]))
                    gaps = [
                        most_repeated[1][i + 1] - most_repeated[1][i]
                        for i in range(len(most_repeated[1]) - 1)
                    ]
                    print(
                        f"    {n}-gram: {len(repeated)} unique repeated, "
                        f"max repeats={len(most_repeated[1])}, gaps={gaps[:5]}"
                    )

        print()


def analyze_fdt_in_repetition():
    """Show where FDT falls within the repetition cycle."""
    print("=" * 80)
    print("  FDT Position Within Repetition Cycle")
    print("=" * 80)
    print()

    experiments = [
        ("2K", "B-2048", "L-2k-h2o", 1024),
        ("4K", "B-4096", "L-4k-h2o", 2048),
        ("8K", "B-8192", "L-8k-h2o", 4096),
    ]

    for label, base_name, exp_name, inject_pos in experiments:
        base_recs, _ = load(base_name)
        exp_recs, _ = load(exp_name)
        fdt = compute_fdt(base_recs, exp_recs)

        base_tokens = [r["token_id"] for r in base_recs]

        # Find the text pattern around FDT and search for it earlier
        fdt_context = base_tokens[max(0, fdt - 20) : fdt + 5]

        print(f"── {label}: FDT={fdt}, inject@{inject_pos} ──")
        print(f"  FDT context (20 tokens before + FDT):")
        fdt_text = "".join(
            base_recs[i].get("text", "")
            for i in range(max(0, fdt - 20), min(fdt + 5, len(base_recs)))
        )
        print(f"  \"{fdt_text}\"")

        # Search for this pattern appearing earlier
        search_pattern = base_tokens[max(0, fdt - 15) : fdt]
        occurrences = []
        for i in range(len(base_tokens) - len(search_pattern)):
            if base_tokens[i : i + len(search_pattern)] == search_pattern:
                occurrences.append(i)

        print(f"  Pattern appears at positions: {occurrences}")

        if len(occurrences) > 1:
            # Show what comes AFTER each occurrence
            for occ in occurrences:
                end_pos = occ + len(search_pattern)
                if end_pos + 5 <= len(base_recs):
                    next_tokens = [base_recs[end_pos + j]["token_id"] for j in range(5)]
                    next_text = "".join(
                        base_recs[end_pos + j].get("text", "") for j in range(5)
                    )
                    print(
                        f"    @{occ}: ...pattern... → {repr(next_text)} (ids: {next_tokens})"
                    )

            # Gap between occurrences = cycle length
            gaps = [occurrences[i + 1] - occurrences[i] for i in range(len(occurrences) - 1)]
            print(f"  Gaps between occurrences: {gaps}")
            if gaps:
                print(f"  → Estimated cycle length: ~{sum(gaps) // len(gaps)} tokens")

        # Show where FDT falls relative to inject point
        fdt_offset = fdt - inject_pos
        print(f"  FDT is {fdt_offset} tokens after eviction point")
        print()


def analyze_divergence_point_content():
    """Deep dive into what happens at the exact divergence token."""
    print("=" * 80)
    print("  Divergence Point Analysis")
    print("=" * 80)
    print()

    experiments = [
        ("2K", "B-2048", "L-2k-h2o", 1024),
        ("4K", "B-4096", "L-4k-h2o", 2048),
        ("8K", "B-8192", "L-8k-h2o", 4096),
    ]

    for label, base_name, exp_name, inject_pos in experiments:
        base_recs, _ = load(base_name)
        exp_recs, _ = load(exp_name)
        fdt = compute_fdt(base_recs, exp_recs)

        print(f"── {label}: FDT={fdt} ──")

        # Show top logits at FDT position for both base and exp
        if fdt < len(base_recs) and fdt < len(exp_recs):
            base_logits = base_recs[fdt].get("top_logits", [])
            exp_logits = exp_recs[fdt].get("top_logits", [])

            print(f"  Base chose: id={base_recs[fdt]['token_id']} ({repr(base_recs[fdt].get('text', ''))})")
            print(f"  Exp  chose: id={exp_recs[fdt]['token_id']} ({repr(exp_recs[fdt].get('text', ''))})")
            print()

            print(f"  Base top logits:")
            for tid, logit in base_logits[:5]:
                # Find if this token appears in exp top logits
                exp_rank = None
                for j, (etid, _) in enumerate(exp_logits):
                    if etid == tid:
                        exp_rank = j
                        break
                rank_str = f"(exp rank {exp_rank})" if exp_rank is not None else "(not in exp top)"
                print(f"    id={tid:>6} logit={logit:>8.3f} {rank_str}")

            print(f"\n  Exp top logits:")
            for tid, logit in exp_logits[:5]:
                base_rank = None
                for j, (btid, _) in enumerate(base_logits):
                    if btid == tid:
                        base_rank = j
                        break
                rank_str = f"(base rank {base_rank})" if base_rank is not None else "(not in base top)"
                print(f"    id={tid:>6} logit={logit:>8.3f} {rank_str}")

            # Logit difference for top-1
            if base_logits and exp_logits:
                base_top1_logit = base_logits[0][1]
                base_top2_logit = base_logits[1][1] if len(base_logits) > 1 else 0
                exp_top1_logit = exp_logits[0][1]
                exp_top2_logit = exp_logits[1][1] if len(exp_logits) > 1 else 0

                print(f"\n  Base: top1-top2 gap = {base_top1_logit - base_top2_logit:.3f}")
                print(f"  Exp:  top1-top2 gap = {exp_top1_logit - exp_top2_logit:.3f}")
                print(f"  → {'Wide gap (confident)' if base_top1_logit - base_top2_logit > 2 else 'Narrow gap (uncertain)'}")

        print()


def main():
    print("╔" + "═" * 78 + "╗")
    print("║  Phase 1 Supplementary: Repetition & Cycle Analysis" + " " * 25 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    analyze_repetition()
    analyze_fdt_in_repetition()
    analyze_divergence_point_content()


if __name__ == "__main__":
    main()
