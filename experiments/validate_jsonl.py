#!/usr/bin/env python3
"""Validate experiment JSONL output format and reproducibility."""

import json
import sys
import os

def validate_token_record(record, idx, filename):
    """Validate a per-token record has all required fields with correct types."""
    errors = []
    required = {
        "pos": int,
        "token_id": int,
        "text": str,
        "tbt_ms": (int, float),
        "forward_ms": (int, float),
        "actions": list,
        "cache_pos": int,
        "throttle_ms": int,
        "top_logits": list,
    }
    for field, expected_type in required.items():
        if field not in record:
            errors.append(f"  line {idx+1}: missing field '{field}'")
        elif not isinstance(record[field], expected_type):
            errors.append(f"  line {idx+1}: '{field}' expected {expected_type}, got {type(record[field])}")

    # Validate top_logits structure
    if "top_logits" in record and isinstance(record["top_logits"], list):
        for entry in record["top_logits"]:
            if not isinstance(entry, list) or len(entry) != 2:
                errors.append(f"  line {idx+1}: top_logits entry not [id, value]: {entry}")
                break

    # sys field: can be null or object
    if "sys" in record and record["sys"] is not None:
        sys_fields = {"rss_mb": (int, float), "cpu_mhz": list, "thermal_mc": list}
        for sf, st in sys_fields.items():
            if sf not in record["sys"]:
                errors.append(f"  line {idx+1}: sys missing '{sf}'")
            elif not isinstance(record["sys"][sf], st):
                errors.append(f"  line {idx+1}: sys.{sf} expected {st}, got {type(record['sys'][sf])}")

    return errors


def validate_summary(record, filename):
    """Validate the summary record."""
    errors = []
    required = {
        "_summary": bool,
        "total_tokens": int,
        "ttft_ms": (int, float),
        "avg_tbt_ms": (int, float),
        "avg_forward_ms": (int, float),
        "total_throttle_ms": int,
        "eviction_count": int,
        "evicted_tokens_total": int,
        "final_cache_pos": int,
        "max_seq_len": int,
        "prompt": str,
        "schedule_name": str,
        "eviction_policy": str,
        "backend": str,
        "sample_interval": int,
    }
    for field, expected_type in required.items():
        if field not in record:
            errors.append(f"  summary: missing field '{field}'")
        elif not isinstance(record[field], expected_type):
            errors.append(f"  summary: '{field}' expected {expected_type}, got {type(record[field])}")

    if not record.get("_summary"):
        errors.append("  summary: _summary is not true")

    return errors


def validate_file(filepath):
    """Validate a single JSONL file."""
    filename = os.path.basename(filepath)
    print(f"\n--- {filename} ---")

    with open(filepath) as f:
        lines = f.readlines()

    if not lines:
        print("  ERROR: empty file")
        return False

    all_errors = []
    token_records = []
    summary = None

    for idx, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            all_errors.append(f"  line {idx+1}: invalid JSON: {e}")
            continue

        if record.get("_summary"):
            summary = record
            all_errors.extend(validate_summary(record, filename))
        else:
            token_records.append(record)
            all_errors.extend(validate_token_record(record, idx, filename))

    # Check summary exists
    if summary is None:
        all_errors.append("  ERROR: no summary record found")

    # Check token count matches
    if summary and summary.get("total_tokens") != len(token_records):
        all_errors.append(
            f"  ERROR: summary.total_tokens={summary.get('total_tokens')} "
            f"but found {len(token_records)} token records"
        )

    # Check pos is sequential
    positions = [r["pos"] for r in token_records if "pos" in r]
    expected = list(range(len(positions)))
    if positions != expected:
        all_errors.append(f"  ERROR: positions not sequential 0..{len(positions)-1}")

    # Check sys sampling
    sys_count = sum(1 for r in token_records if r.get("sys") is not None)

    # Report
    if all_errors:
        for e in all_errors:
            print(e)
        return False
    else:
        avg_tbt = summary.get("avg_tbt_ms", 0) if summary else 0
        print(f"  OK: {len(token_records)} tokens, "
              f"sys_samples={sys_count}/{len(token_records)}, "
              f"avg_tbt={avg_tbt:.1f}ms, "
              f"evictions={summary.get('eviction_count', 0)}")
        return True


def check_reproducibility(file1, file2):
    """Check if two greedy runs produce identical token sequences."""
    print(f"\n=== Reproducibility: {os.path.basename(file1)} vs {os.path.basename(file2)} ===")

    tokens1, tokens2 = [], []
    for fpath, tokens in [(file1, tokens1), (file2, tokens2)]:
        with open(fpath) as f:
            for line in f:
                r = json.loads(line)
                if not r.get("_summary"):
                    tokens.append(r["token_id"])

    if tokens1 == tokens2:
        print(f"  PASS: {len(tokens1)} tokens identical")
        return True
    else:
        mismatches = sum(1 for a, b in zip(tokens1, tokens2) if a != b)
        first_diff = next((i for i, (a, b) in enumerate(zip(tokens1, tokens2)) if a != b), -1)
        print(f"  FAIL: {mismatches} mismatches, first at pos {first_diff}")
        return False


def main():
    results_dir = "experiments/results"
    files = sorted(f for f in os.listdir(results_dir) if f.endswith(".jsonl"))

    if not files:
        print("No JSONL files found in", results_dir)
        return 1

    print("=" * 50)
    print("  JSONL Validation Report")
    print("=" * 50)

    all_ok = True
    for f in files:
        ok = validate_file(os.path.join(results_dir, f))
        if not ok:
            all_ok = False

    # Reproducibility check
    b128 = os.path.join(results_dir, "B-128.jsonl")
    b128_check = os.path.join(results_dir, "B-128-check.jsonl")
    if os.path.exists(b128) and os.path.exists(b128_check):
        if not check_reproducibility(b128, b128_check):
            all_ok = False

    print("\n" + "=" * 50)
    if all_ok:
        print("  ALL VALIDATIONS PASSED")
    else:
        print("  SOME VALIDATIONS FAILED")
    print("=" * 50)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
