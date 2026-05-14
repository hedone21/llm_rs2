#!/usr/bin/env python3
"""TBT distribution analyzer for swap experiments.

Reads tbt.jsonl files and reports:
- tok[0] forward / TBT (separated)
- tokens 1..end forward distribution (mean, p50, p99, max)
- swap-active window vs swap-idle window forward (windowing by --swap-active)
- "Decode:" full average (tok[0] included)

Usage:
  python analyze_tbt.py <dir-or-glob>
  python analyze_tbt.py <dir> --tag-filter 'k32_*'
  python analyze_tbt.py <dir> --swap-active 2  # tokens 0..1 swap-active
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys


def percentile(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def parse_tbt(path):
    tokens = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens.append(json.loads(line))
    return tokens


def infer_swap_active_from_k(tag):
    m = re.match(r"k(\d+)_", tag)
    if not m:
        return None
    K = int(m.group(1))
    return (32 + K - 1) // K  # ceil(32/K) batches = swap-active tokens


def summarize_file(path, swap_active=None):
    tag = os.path.basename(path).replace(".tbt.jsonl", "")
    if swap_active is None:
        swap_active = infer_swap_active_from_k(tag) or 1

    tokens = parse_tbt(path)
    if not tokens:
        return None

    tok0_fwd = tokens[0]["forward_ms"]
    tok0_tbt = tokens[0]["tbt_ms"]
    rest_fwd = [t["forward_ms"] for t in tokens[1:]]
    rest_tbt = [t["tbt_ms"] for t in tokens[1:]]

    # Active window = tokens 0..swap_active-1
    # Idle window = tokens swap_active..end
    active_fwd = [t["forward_ms"] for t in tokens[:swap_active]]
    idle_fwd = [t["forward_ms"] for t in tokens[swap_active:]]

    # Tok[0] included forward avg ("Decode:" equivalent)
    all_fwd = [t["forward_ms"] for t in tokens]
    all_tbt = [t["tbt_ms"] for t in tokens]

    return dict(
        tag=tag,
        n=len(tokens),
        swap_active=swap_active,
        tok0_fwd=tok0_fwd,
        tok0_tbt=tok0_tbt,
        # tokens 1..end forward (matches "Decode (excl tok[0])")
        rest_fwd_mean=statistics.mean(rest_fwd) if rest_fwd else float("nan"),
        rest_fwd_p50=percentile(rest_fwd, 0.5),
        rest_fwd_p99=percentile(rest_fwd, 0.99),
        rest_fwd_max=max(rest_fwd) if rest_fwd else float("nan"),
        # tok[0]-included forward (matches "Decode:" line)
        all_fwd_mean=statistics.mean(all_fwd),
        # TBT
        rest_tbt_mean=statistics.mean(rest_tbt) if rest_tbt else float("nan"),
        rest_tbt_p50=percentile(rest_tbt, 0.5),
        rest_tbt_p99=percentile(rest_tbt, 0.99),
        rest_tbt_max=max(rest_tbt) if rest_tbt else float("nan"),
        all_tbt_mean=statistics.mean(all_tbt),
        # Active/idle window split
        active_fwd_mean=statistics.mean(active_fwd) if active_fwd else float("nan"),
        active_fwd_max=max(active_fwd) if active_fwd else float("nan"),
        idle_fwd_mean=statistics.mean(idle_fwd) if idle_fwd else float("nan"),
        idle_fwd_max=max(idle_fwd) if idle_fwd else float("nan"),
    )


def group_by_kmode(rows):
    """Group rows by (K, mode) by stripping _r{N} suffix from tag."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in rows:
        base = re.sub(r"_r\d+$", "", r["tag"])
        groups[base].append(r)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Directory containing *.tbt.jsonl or glob pattern")
    ap.add_argument("--tag-filter", default=None, help="Regex filter on tag")
    ap.add_argument("--swap-active", type=int, default=None,
                    help="Override swap-active token count (default: ceil(32/K))")
    ap.add_argument("--per-file", action="store_true",
                    help="Print per-file rows instead of aggregating")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.tbt.jsonl")))
    else:
        files = sorted(glob.glob(args.path))

    if args.tag_filter:
        pat = re.compile(args.tag_filter)
        files = [f for f in files
                 if pat.search(os.path.basename(f).replace(".tbt.jsonl", ""))]

    if not files:
        print(f"No files matched: {args.path}", file=sys.stderr)
        return 1

    rows = []
    for f in files:
        row = summarize_file(f, swap_active=args.swap_active)
        if row:
            rows.append(row)

    if args.per_file:
        print(f"{'tag':<24} {'tok0_fwd':>8} {'rest_avg':>8} {'rest_p50':>8} "
              f"{'rest_p99':>8} {'rest_max':>8} {'all_avg':>8} "
              f"{'act_avg':>8} {'idle_avg':>8} {'rest_tbt':>8}")
        for r in rows:
            print(f"{r['tag']:<24} {r['tok0_fwd']:>8.2f} {r['rest_fwd_mean']:>8.2f} "
                  f"{r['rest_fwd_p50']:>8.2f} {r['rest_fwd_p99']:>8.2f} "
                  f"{r['rest_fwd_max']:>8.2f} {r['all_fwd_mean']:>8.2f} "
                  f"{r['active_fwd_mean']:>8.2f} {r['idle_fwd_mean']:>8.2f} "
                  f"{r['rest_tbt_mean']:>8.2f}")
    else:
        groups = group_by_kmode(rows)
        print(f"{'group':<20} {'n':>2} {'tok0':>7} {'rest_avg':>8} {'rest_p99':>8} "
              f"{'rest_max':>8} {'all_avg':>8} {'act_avg':>8} {'idle_avg':>8} "
              f"{'rest_tbt':>8} {'all_tbt':>8}")
        for k in sorted(groups.keys()):
            rs = groups[k]
            def avg(field):
                vals = [r[field] for r in rs]
                return statistics.mean(vals)
            print(f"{k:<20} {len(rs):>2} "
                  f"{avg('tok0_fwd'):>7.1f} "
                  f"{avg('rest_fwd_mean'):>8.2f} "
                  f"{avg('rest_fwd_p99'):>8.2f} "
                  f"{avg('rest_fwd_max'):>8.2f} "
                  f"{avg('all_fwd_mean'):>8.2f} "
                  f"{avg('active_fwd_mean'):>8.2f} "
                  f"{avg('idle_fwd_mean'):>8.2f} "
                  f"{avg('rest_tbt_mean'):>8.2f} "
                  f"{avg('all_tbt_mean'):>8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
