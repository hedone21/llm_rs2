#!/usr/bin/env python3
"""Deep analysis: What tokens did H2O keep vs evict vs sliding window?"""

import json
import csv
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path(__file__).parent.parent / "results" / "round15"

def load_scores(path, prompt_len=50):
    """Load score CSV and return dict of position -> score."""
    scores = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pos = int(row["position"])
            scores[pos] = float(row["score"])
    return scores

def simulate_h2o_eviction(scores, total_positions, protected_prefix=4, target=512, keep_ratio=0.5):
    """Simulate H2O eviction to determine which tokens are kept."""
    n_keep = target
    n_protected = protected_prefix
    n_available = n_keep - n_protected
    n_hh = int(n_available * keep_ratio)
    n_recent = n_available - n_hh

    # Protected
    protected = set(range(n_protected))

    # Recent window
    recent_start = total_positions - n_recent
    recent = set(range(recent_start, total_positions))

    # HH: pick highest scores from non-protected, non-recent pool
    candidates = []
    for pos in range(n_protected, recent_start):
        if pos in scores:
            candidates.append((pos, scores[pos]))
    candidates.sort(key=lambda x: x[1], reverse=True)
    hh = set(c[0] for c in candidates[:n_hh])

    kept = protected | hh | recent
    evicted = set(range(total_positions)) - kept

    return {
        "protected": sorted(protected),
        "hh": sorted(hh),
        "recent": sorted(recent),
        "evicted": sorted(evicted),
        "n_protected": len(protected),
        "n_hh": len(hh),
        "n_recent": len(recent),
        "n_evicted": len(evicted),
        "hh_positions": sorted(hh),
        "recent_start": recent_start,
    }

def simulate_sliding_eviction(total_positions, protected_prefix=4, target=512):
    """Simulate sliding window: keep most recent tokens."""
    n_recent = target - protected_prefix
    protected = set(range(protected_prefix))
    recent_start = total_positions - n_recent
    recent = set(range(recent_start, total_positions))
    kept = protected | recent
    evicted = set(range(total_positions)) - kept
    return {
        "protected": sorted(protected),
        "recent": sorted(recent),
        "evicted": sorted(evicted),
        "recent_start": recent_start,
    }

def analyze_experiment(label, scores_path, total_positions=1074):
    """Analyze one experiment's eviction behavior."""
    scores = load_scores(scores_path)

    h2o = simulate_h2o_eviction(scores, total_positions)
    slide = simulate_sliding_eviction(total_positions)

    # Compare H2O kept vs sliding kept
    h2o_kept = set(h2o["protected"]) | set(h2o["hh"]) | set(h2o["recent"])
    slide_kept = set(slide["protected"]) | set(slide["recent"])

    only_h2o = h2o_kept - slide_kept  # H2O keeps but sliding doesn't
    only_slide = slide_kept - h2o_kept  # Sliding keeps but H2O doesn't
    both = h2o_kept & slide_kept

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Total positions: {total_positions}")
    print(f"  H2O: {h2o['n_protected']} protected + {h2o['n_hh']} HH + {h2o['n_recent']} recent = {h2o['n_protected']+h2o['n_hh']+h2o['n_recent']}")
    print(f"  Sliding: 4 protected + {len(slide['recent'])} recent = {4+len(slide['recent'])}")
    print(f"  H2O recent start: {h2o['recent_start']}")
    print(f"  Sliding recent start: {slide['recent_start']}")
    print()
    print(f"  Overlap analysis:")
    print(f"    Both keep:      {len(both)}")
    print(f"    Only H2O keeps: {len(only_h2o)} (HH tokens not in sliding window)")
    print(f"    Only Slide keeps: {len(only_slide)} (sliding has but H2O evicted)")
    print()

    # The key difference: what does H2O keep instead of what sliding keeps?
    # H2O uses half for HH → its recent window is half-size
    # So sliding keeps positions [recent_start_slide, total) = 508 tokens
    # H2O keeps positions [recent_start_h2o, total) = 254 tokens + 254 HH
    # The difference: sliding keeps [recent_start_slide, recent_start_h2o) contiguously
    # H2O keeps the same positions SCATTERED across [protected_prefix, recent_start_h2o)
    print(f"  Contiguity analysis:")
    h2o_hh_sorted = sorted(only_h2o)
    if h2o_hh_sorted:
        gaps = []
        for i in range(1, len(h2o_hh_sorted)):
            gap = h2o_hh_sorted[i] - h2o_hh_sorted[i-1]
            if gap > 1:
                gaps.append(gap)
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        max_gap = max(gaps) if gaps else 0
        contiguous_runs = 1
        for i in range(1, len(h2o_hh_sorted)):
            if h2o_hh_sorted[i] - h2o_hh_sorted[i-1] > 1:
                contiguous_runs += 1

        print(f"    H2O-only positions range: [{h2o_hh_sorted[0]}, {h2o_hh_sorted[-1]}]")
        print(f"    Number of contiguous runs: {contiguous_runs}")
        print(f"    Avg gap between runs: {avg_gap:.1f}")
        print(f"    Max gap: {max_gap}")

    if only_slide:
        slide_only_sorted = sorted(only_slide)
        print(f"    Slide-only positions range: [{slide_only_sorted[0]}, {slide_only_sorted[-1]}]")
        # Check if these are contiguous
        is_contiguous = all(slide_only_sorted[i+1] - slide_only_sorted[i] == 1 for i in range(len(slide_only_sorted)-1))
        print(f"    Contiguous: {is_contiguous}")

    # Score analysis of the tokens that differ
    if only_h2o:
        h2o_only_scores = [scores.get(p, 0) for p in only_h2o]
        print(f"\n  Score stats of H2O-only (HH) tokens:")
        print(f"    Mean: {sum(h2o_only_scores)/len(h2o_only_scores):.4f}")
        print(f"    Min:  {min(h2o_only_scores):.4f}")
        print(f"    Max:  {max(h2o_only_scores):.4f}")

    if only_slide:
        slide_only_scores = [scores.get(p, 0) for p in only_slide]
        print(f"  Score stats of Slide-only tokens (evicted by H2O):")
        print(f"    Mean: {sum(slide_only_scores)/len(slide_only_scores):.4f}")
        print(f"    Min:  {min(slide_only_scores):.4f}")
        print(f"    Max:  {max(slide_only_scores):.4f}")

def main():
    experiments = [
        ("H2O-RAW-PPL01", RESULTS_DIR / "H2O-RAW-PPL01.scores.csv", 1074),
        ("H2O-NORM-PPL01", RESULTS_DIR / "H2O-NORM-PPL01.scores.csv", 1074),
        ("H2O-RAW-PPL03", RESULTS_DIR / "H2O-RAW-PPL03.scores.csv", 1084),
        ("H2O-NORM-PPL03", RESULTS_DIR / "H2O-NORM-PPL03.scores.csv", 1084),
    ]

    for label, path, total in experiments:
        if path.exists():
            analyze_experiment(label, path, total)

if __name__ == "__main__":
    main()
