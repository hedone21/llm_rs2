"""Three-layer assertions: functional / performance / accuracy.

Each verify_* function returns {"pass": bool, "details": {...}} so that the
top-level orchestrator can aggregate verdicts uniformly.

Phase 3: heartbeat_checks now supports
    - contains / does_not_contain (for active_actions)
    - transitions=[...] (for state sequence subsequence match)
    - transitions_to=<value> (for active_device / state / kv_dtype final value)
    - decrease_from_peak (kv_cache_tokens; currently skipped as the field is
      absent — kv_util is used as a best-effort proxy)
kv_dtype is not a heartbeat field; the check falls back to an stderr pattern
for KIVI-Resilience transitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .log_parser import (
    DEFAULT_CRASH_DENY_PATTERNS,
    count_decoded_tokens,
    count_stderr_patterns,
    find_crash_signatures,
    find_sequence,
    heartbeat_transitions,
    state_transition_matches,
    stderr_has_pattern,
)


def _check_heartbeat_field(
    check: Dict[str, Any],
    heartbeat_list: List[Dict[str, Any]],
    stderr_path: Path,
) -> Dict[str, Any]:
    """Handle one heartbeat_checks entry. Returns a check-result dict."""
    field = check.get("field")
    required = bool(check.get("required", True))
    raw_ok = False
    result: Dict[str, Any] = {
        "type": f"heartbeat_{field}",
        "field": field,
        "required": required,
    }

    if field == "active_actions":
        if "contains" in check:
            needle = str(check["contains"]).lower()
            matched = [
                hb for hb in heartbeat_list
                if needle in [a.lower() for a in hb.get("active_actions", [])]
            ]
            raw_ok = len(matched) > 0
            result["needle"] = needle
            result["matching_count"] = len(matched)
            result["total_heartbeats"] = len(heartbeat_list)
        elif "does_not_contain" in check:
            needle = str(check["does_not_contain"]).lower()
            matched = [
                hb for hb in heartbeat_list
                if needle in [a.lower() for a in hb.get("active_actions", [])]
            ]
            raw_ok = len(matched) == 0
            result["needle"] = needle
            result["matching_count"] = len(matched)
        else:
            result["note"] = "unsupported active_actions check"
            raw_ok = True
    elif field == "state":
        if "transitions" in check:
            expected_seq = list(check["transitions"])
            raw_ok = state_transition_matches(heartbeat_list, expected_seq)
            result["expected_sequence"] = expected_seq
            result["observed_sequence"] = heartbeat_transitions(heartbeat_list)[
                "state_sequence"
            ]
        elif "transitions_to" in check:
            target = str(check["transitions_to"])
            seq = heartbeat_transitions(heartbeat_list)["state_sequence"]
            raw_ok = bool(seq) and seq[-1] == target
            result["target"] = target
            result["observed_sequence"] = seq
        else:
            result["note"] = "unsupported state check"
            raw_ok = True
    elif field == "active_device":
        if "transitions_to" in check:
            target = str(check["transitions_to"])
            seq = heartbeat_transitions(heartbeat_list)["device_sequence"]
            raw_ok = bool(seq) and seq[-1] == target
            result["target"] = target
            result["observed_sequence"] = seq
        else:
            result["note"] = "unsupported active_device check"
            raw_ok = True
    elif field == "kv_cache_tokens":
        # Heartbeat records lack an explicit kv_cache_tokens field in the current
        # wire format; we best-effort substitute kv_util (a monotonic proxy) but
        # treat the assertion as informational unless required=true.
        if check.get("decrease_from_peak"):
            trans = heartbeat_transitions(heartbeat_list)
            peak = trans.get("kv_util_peak", 0.0)
            final = trans.get("kv_util_final", 0.0)
            # A sane evict should drop kv_util below peak by at least 0.01
            raw_ok = peak > 0.0 and (peak - final) >= 0.01
            result["kv_util_peak"] = peak
            result["kv_util_final"] = final
            result["note"] = "kv_cache_tokens unavailable — using kv_util proxy"
        else:
            result["note"] = "unsupported kv_cache_tokens check"
            raw_ok = True
    elif field == "kv_dtype":
        # Fallback path: KIVI-Resilience stderr pattern indicates a dtype switch.
        pattern = r"\[KIVI-Resilience\]\s+Transitioned\s+KV\s+cache"
        raw_ok = stderr_has_pattern(stderr_path, pattern)
        result["pattern"] = pattern
        result["note"] = "heartbeat has no kv_dtype field; using stderr fallback"
    elif field == "partition_ratio":
        # No direct field on heartbeat; stderr log or informational.
        pattern = r"(partition|Partition).*[Rr]atio"
        raw_ok = stderr_has_pattern(stderr_path, pattern)
        result["pattern"] = pattern
        result["note"] = "informational"
    else:
        result["note"] = f"unknown field: {field}"
        raw_ok = True

    ok = raw_ok or (not required)
    result["raw_pass"] = raw_ok
    result["pass"] = ok
    return result


def verify_functional(
    expected: Dict[str, Any],
    stderr_path: Path,
    heartbeat_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check stderr patterns + heartbeat conditions from expected.functional."""
    details: Dict[str, Any] = {"checks": []}
    overall = True

    # stderr patterns: each pattern must match at least once (unless min=0)
    patterns = list(expected.get("stderr_patterns", []) or [])
    min_occurrences = int(expected.get("min_occurrences", 1))
    if patterns:
        counts = count_stderr_patterns(stderr_path, patterns)
        for pat, cnt in counts.items():
            ok = cnt >= min_occurrences
            details["checks"].append({
                "type": "stderr_pattern",
                "pattern": pat,
                "count": cnt,
                "required_min": min_occurrences,
                "pass": ok,
            })
            overall = overall and ok

    # heartbeat checks
    hb_checks = list(expected.get("heartbeat_checks", []) or [])
    for check in hb_checks:
        result = _check_heartbeat_field(check, heartbeat_list, stderr_path)
        details["checks"].append(result)
        overall = overall and bool(result["pass"])

    # stderr_sequence (v2 — ordered pattern assertion)
    seq_spec = expected.get("stderr_sequence") or []
    if seq_spec:
        seq_result = find_sequence(stderr_path, list(seq_spec))
        details["stderr_sequence"] = seq_result
        overall = overall and bool(seq_result.get("pass"))

    # commands presence (optional forward-compat)
    any_of = list(expected.get("engine_commands_any_of", []) or [])
    if any_of:
        details["engine_commands_any_of"] = any_of

    # Attach a compact heartbeat transition summary for report readers.
    details["heartbeat_transitions"] = heartbeat_transitions(heartbeat_list)

    return {"pass": overall, "details": details}


def verify_crash_and_progress(
    expected: Dict[str, Any],
    stderr_path: Path,
    jsonl_path: Path,
    requested_decode_tokens: int,
    action_returncode: Optional[int],
) -> Dict[str, Any]:
    """Silent-failure guard: scan stderr for crash patterns, verify decode
    actually produced a reasonable fraction of the requested tokens, and
    reject non-zero returncodes.

    expected shape:
      {
        "decode_tokens_min_ratio": 0.5,   # default 0.5
        "crash_deny_patterns": [...],     # optional override, else defaults
        "allow_nonzero_returncode": false,
      }

    Any single violation fails.
    """
    details: Dict[str, Any] = {}
    overall = True

    # 1. Crash deny-list
    deny = list(expected.get("crash_deny_patterns") or DEFAULT_CRASH_DENY_PATTERNS)
    hits = find_crash_signatures(stderr_path, deny)
    details["crash_hits"] = hits
    if hits:
        overall = False

    # 2. Decode token progress
    min_ratio = float(expected.get("decode_tokens_min_ratio", 0.5))
    actual_tokens = count_decoded_tokens(jsonl_path)
    required = int(round(requested_decode_tokens * min_ratio))
    details["requested_decode_tokens"] = requested_decode_tokens
    details["actual_decode_tokens"] = actual_tokens
    details["min_ratio"] = min_ratio
    details["required_min_tokens"] = required
    if requested_decode_tokens > 0 and actual_tokens < required:
        overall = False
        details["progress_failed"] = True
    else:
        details["progress_failed"] = False

    # 3. returncode
    allow_nz = bool(expected.get("allow_nonzero_returncode", False))
    details["action_returncode"] = action_returncode
    details["allow_nonzero_returncode"] = allow_nz
    if action_returncode is not None and action_returncode != 0 and not allow_nz:
        overall = False
        details["nonzero_returncode"] = True

    return {"pass": overall, "details": details}


def verify_performance(
    expected: Dict[str, Any],
    baseline_summary: Optional[Dict[str, Any]],
    action_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare a metric between baseline and action JSONL summaries.

    expected = {
        "metric": "avg_tbt_ms",
        "delta_vs_baseline": {"max_pct": +500.0, "tolerance_pct": 50.0},
    }
    Passes if delta_pct <= max_pct + tolerance_pct.
    """
    details: Dict[str, Any] = {}
    if not expected:
        return {"pass": True, "details": {"note": "no performance expectation"}}
    if baseline_summary is None or action_summary is None:
        return {
            "pass": False,
            "details": {"error": "missing summary record(s)",
                         "baseline_present": baseline_summary is not None,
                         "action_present": action_summary is not None},
        }

    metric = expected.get("metric", "avg_tbt_ms")
    delta_cfg = expected.get("delta_vs_baseline", {}) or {}
    has_max = "max_pct" in delta_cfg
    has_min = "min_pct" in delta_cfg
    max_pct = float(delta_cfg.get("max_pct", 999.0))
    min_pct = float(delta_cfg.get("min_pct", -999.0))
    tol_pct = float(delta_cfg.get("tolerance_pct", 0.0))
    upper = max_pct + tol_pct
    # For min_pct the tolerance relaxes the lower bound (subtract tol).
    lower = min_pct - tol_pct

    base_val = baseline_summary.get(metric)
    act_val = action_summary.get(metric)
    details["metric"] = metric
    details["baseline"] = base_val
    details["action"] = act_val
    if has_max:
        details["max_pct"] = max_pct
        details["upper_bound_pct"] = upper
    if has_min:
        details["min_pct"] = min_pct
        details["lower_bound_pct"] = lower
    details["tolerance_pct"] = tol_pct

    if base_val is None or act_val is None or base_val == 0:
        return {"pass": False, "details": {**details, "error": "metric missing or zero baseline"}}

    delta_pct = (float(act_val) - float(base_val)) / float(base_val) * 100.0
    details["delta_pct"] = delta_pct
    ok_upper = (delta_pct <= upper) if has_max else True
    ok_lower = (delta_pct >= lower) if has_min else True
    ok = ok_upper and ok_lower
    if not ok_upper:
        details["failed_bound"] = "max_pct"
    elif not ok_lower:
        details["failed_bound"] = "min_pct"
    return {"pass": ok, "details": details}


def verify_accuracy(
    expected: Dict[str, Any],
    baseline_text: str,
    action_text: str,
) -> Dict[str, Any]:
    """Compare detokenized texts using ROUGE-L / BLEU-4 / char-similarity."""
    from .text_accuracy import compare_texts

    details = compare_texts(baseline_text, action_text)
    min_rouge_l = float(expected.get("min_rouge_l", 0.0))
    min_bleu_4 = float(expected.get("min_bleu_4", 0.0))
    min_char = float(expected.get("min_char_similarity", 0.0))

    rouge_ok = details["rouge_l_f1"] >= min_rouge_l
    bleu_ok = details["bleu_4"] >= min_bleu_4
    char_ok = details["char_similarity"] >= min_char
    overall = rouge_ok and bleu_ok and char_ok

    details.update({
        "min_rouge_l": min_rouge_l,
        "min_bleu_4": min_bleu_4,
        "min_char_similarity": min_char,
        "rouge_ok": rouge_ok,
        "bleu_ok": bleu_ok,
        "char_ok": char_ok,
    })
    return {"pass": overall, "details": details}


def aggregate_verdict(
    functional: Dict[str, Any],
    performance: Dict[str, Any],
    accuracy: Dict[str, Any],
    pass_criteria: str = "all",
    crash_and_progress: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Combine result layers into an overall verdict.

    crash_and_progress (v2) is ALWAYS a hard gate regardless of pass_criteria:
    a crash or silent truncation must never be masked by functional_only.

    pass_criteria:
      - "all" (default): functional AND performance AND accuracy must pass.
      - "functional_only": only functional matters; perf/accuracy recorded
        for reference but don't block. Still gated by crash_and_progress.
    """
    crash_pass = True if crash_and_progress is None else bool(crash_and_progress.get("pass"))

    if pass_criteria == "functional_only":
        import logging
        logging.getLogger(__name__).warning(
            "pass_criteria=functional_only in use — justify this on the YAML "
            "(backends that legitimately diverge output such as SwitchBackend / "
            "KvQuantDynamic->q2). Crash+progress gate still applies."
        )
        overall = bool(functional.get("pass")) and crash_pass
    else:
        overall = (
            bool(functional.get("pass"))
            and bool(performance.get("pass"))
            and bool(accuracy.get("pass"))
            and crash_pass
        )
    result = {
        "functional": functional,
        "performance": performance,
        "accuracy": accuracy,
        "overall_pass": overall,
        "pass_criteria": pass_criteria,
    }
    if crash_and_progress is not None:
        result["crash_and_progress"] = crash_and_progress
    return result
