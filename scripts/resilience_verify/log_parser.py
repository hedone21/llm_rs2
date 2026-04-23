"""Log / JSONL / heartbeat parsers for resilience verify harness."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ── stderr pattern matching ──────────────────────────────


def count_stderr_patterns(stderr_path: Path, patterns: Iterable[str]) -> Dict[str, int]:
    """Count occurrences of each regex pattern in the stderr file.

    Returns a dict mapping pattern → match count.
    """
    counts: Dict[str, int] = {p: 0 for p in patterns}
    if not stderr_path.exists():
        return counts
    compiled = [(p, re.compile(p)) for p in counts.keys()]
    with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            for pat, rx in compiled:
                if rx.search(line):
                    counts[pat] += 1
    return counts


def stderr_has_pattern(stderr_path: Path, pattern: str) -> bool:
    """Convenience: return True iff pattern is found at least once in stderr."""
    if not stderr_path.exists():
        return False
    rx = re.compile(pattern)
    with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if rx.search(line):
                return True
    return False


# ── Experiment JSONL ─────────────────────────────────────


def load_summary(jsonl_path: Path) -> Optional[Dict[str, Any]]:
    """Load the _summary record (typically the last line) from an experiment JSONL."""
    if not jsonl_path.exists():
        return None
    summary: Optional[Dict[str, Any]] = None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("_summary") is True:
                summary = rec
    return summary


def load_token_records(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load every non-summary record (token records) from JSONL in order."""
    out: List[Dict[str, Any]] = []
    if not jsonl_path.exists():
        return out
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("_summary") is True:
                continue
            out.append(rec)
    return out


# ── Heartbeat parser (mock_manager stdout) ────────────────


# Full: [MockManager] Heartbeat #<N>: device=<dev>, kv_util=<float>, tokens=<int>, state=<State>, active_actions=[...]
# Short (post-directive): [MockManager] Heartbeat #<N>: device=<dev>, kv_util=<float>, active_actions=[...]
_HEARTBEAT_FULL_RX = re.compile(
    r"\[MockManager\]\s+Heartbeat\s+#(?P<n>\d+):\s+"
    r"device=(?P<device>[^,]+),\s+"
    r"kv_util=(?P<kv_util>[0-9]+(?:\.[0-9]+)?),\s+"
    r"tokens=(?P<tokens>\d+),\s+"
    r"state=(?P<state>[^,]+),\s+"
    r"active_actions=(?P<active_actions>\[.*\])\s*$"
)

_HEARTBEAT_SHORT_RX = re.compile(
    r"\[MockManager\]\s+Heartbeat\s+#(?P<n>\d+):\s+"
    r"device=(?P<device>[^,]+),\s+"
    r"kv_util=(?P<kv_util>[0-9]+(?:\.[0-9]+)?),\s+"
    r"active_actions=(?P<active_actions>\[.*\])\s*$"
)

# Compact form (scenario mode interleaved during wait): "  (heartbeat: kv_util=0.123, actions=[\"throttle\"])"
_HEARTBEAT_COMPACT_RX = re.compile(
    r"\(heartbeat:\s+"
    r"kv_util=(?P<kv_util>[0-9]+(?:\.[0-9]+)?),\s+"
    r"actions=(?P<active_actions>\[.*\])\)"
)


def _parse_actions_field(raw: str) -> List[str]:
    """Parse Rust Debug formatted Vec<String> like `["throttle", "evict"]` or `[]`.

    Lenient: returns lowercase strings, stripping quotes.
    """
    raw = raw.strip()
    if not raw or raw == "[]":
        return []
    inner = raw.strip("[]")
    if not inner.strip():
        return []
    parts = [p.strip() for p in inner.split(",")]
    out: List[str] = []
    for p in parts:
        p = p.strip().strip('"').strip("'")
        if p:
            out.append(p.lower())
    return out


def parse_heartbeats(stdout_path: Path) -> List[Dict[str, Any]]:
    """Parse mock_manager stdout file for heartbeat lines.

    Returns list of dicts with keys: n, device, kv_util, tokens, state,
    active_actions, phase (pre|post).
    """
    out: List[Dict[str, Any]] = []
    if not stdout_path.exists():
        return out
    phase = "pre"  # flip to "post" once directive send logged
    with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if "Observing post-directive heartbeats" in line:
                phase = "post"
                continue
            m = _HEARTBEAT_FULL_RX.search(line)
            if m:
                out.append({
                    "n": int(m.group("n")),
                    "device": m.group("device").strip(),
                    "kv_util": float(m.group("kv_util")),
                    "tokens": int(m.group("tokens")),
                    "state": m.group("state").strip(),
                    "active_actions": _parse_actions_field(m.group("active_actions")),
                    "phase": phase,
                })
                continue
            m = _HEARTBEAT_SHORT_RX.search(line)
            if m:
                out.append({
                    "n": int(m.group("n")),
                    "device": m.group("device").strip(),
                    "kv_util": float(m.group("kv_util")),
                    "tokens": None,
                    "state": None,
                    "active_actions": _parse_actions_field(m.group("active_actions")),
                    "phase": phase,
                })
                continue
            m = _HEARTBEAT_COMPACT_RX.search(line)
            if m:
                out.append({
                    "n": None,
                    "device": None,
                    "kv_util": float(m.group("kv_util")),
                    "tokens": None,
                    "state": None,
                    "active_actions": _parse_actions_field(m.group("active_actions")),
                    "phase": phase,
                })
    return out


def heartbeat_transitions(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise heartbeat sequence:
      - first_seen[action] = index of first record containing `action`
      - device_sequence: deduped consecutive device values
      - state_sequence: deduped consecutive state values (None skipped)
      - kv_util_peak / kv_util_final
      - action_sequence: deduped list of first-seen action names in order
    """
    first_seen: Dict[str, int] = {}
    device_seq: List[str] = []
    state_seq: List[str] = []
    action_seq: List[str] = []
    kv_peak = 0.0
    kv_final = 0.0

    for idx, rec in enumerate(records):
        for a in rec.get("active_actions", []):
            if a not in first_seen:
                first_seen[a] = idx
            if not action_seq or action_seq[-1] != a:
                action_seq.append(a)
        dev = rec.get("device")
        if dev and (not device_seq or device_seq[-1] != dev):
            device_seq.append(dev)
        st = rec.get("state")
        if st and (not state_seq or state_seq[-1] != st):
            state_seq.append(st)
        kv = rec.get("kv_util") or 0.0
        kv_peak = max(kv_peak, kv)
        kv_final = kv

    return {
        "first_seen": first_seen,
        "device_sequence": device_seq,
        "state_sequence": state_seq,
        "action_sequence": action_seq,
        "kv_util_peak": kv_peak,
        "kv_util_final": kv_final,
        "total_heartbeats": len(records),
    }


def _subsequence_matches(needle: List[str], haystack: List[str]) -> bool:
    """Return True iff `needle` appears as a contiguous subsequence in `haystack`."""
    if not needle:
        return True
    if len(needle) > len(haystack):
        return False
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return True
    return False


def state_transition_matches(records: List[Dict[str, Any]], expected: List[str]) -> bool:
    """Check whether the deduped state sequence contains the expected subsequence."""
    trans = heartbeat_transitions(records)
    return _subsequence_matches(list(expected), trans["state_sequence"])


# ── v2: crash + progress helpers ─────────────────────────


DEFAULT_CRASH_DENY_PATTERNS = [
    r"SIGSEGV",
    r"thread '.*' panicked",
    r"^Aborted$",
    r"Aborted \(core dumped\)",
    r"fatal runtime error",
    r"double free",
    r"stack overflow",
    r"Segmentation fault",
    r"Illegal instruction",
    r"SIGABRT",
    r"SIGILL",
    r"SIGBUS",
]


def find_crash_signatures(stderr_path: Path, deny_patterns: Iterable[str]) -> List[Dict[str, Any]]:
    """Scan stderr for crash/panic patterns. Returns list of hits:
    [{pattern, line_no, line}]. Empty list means clean.
    """
    hits: List[Dict[str, Any]] = []
    if not stderr_path.exists():
        return hits
    compiled = [(p, re.compile(p)) for p in deny_patterns]
    with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, start=1):
            for pat, rx in compiled:
                if rx.search(line):
                    hits.append({"pattern": pat, "line_no": idx, "line": line.rstrip()})
                    break
    return hits


def count_decoded_tokens(jsonl_path: Path) -> int:
    """Count non-summary token records in an experiment JSONL."""
    if not jsonl_path.exists():
        return 0
    n = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("_summary") is True:
                continue
            n += 1
    return n


def find_sequence(stderr_path: Path, sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ordered stderr pattern assertion.

    sequence items:
      {pattern: str, name?: str, after?: str, min_occurrences?: int=1}

    Each item must match at least `min_occurrences` times, and if `after` is
    set, the matches must occur *after* the first-match line of the named
    preceding item. Returns:
      {pass: bool, steps: [{name, pattern, found_line_nos, after, pass, ...}]}

    Lines are read once into memory (stderr files are <~10 MB).
    """
    result: Dict[str, Any] = {"pass": True, "steps": []}
    if not stderr_path.exists():
        result["pass"] = False
        result["error"] = f"stderr missing: {stderr_path}"
        return result

    with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    name_to_first_line: Dict[str, int] = {}
    for idx, step in enumerate(sequence):
        pattern = step.get("pattern")
        name = step.get("name") or f"step{idx}"
        after_name = step.get("after")
        min_occ = int(step.get("min_occurrences", 1))
        start_line = 0
        after_missing = False
        if after_name:
            if after_name not in name_to_first_line:
                after_missing = True
            else:
                start_line = name_to_first_line[after_name]

        rx = re.compile(pattern) if pattern else None
        matched_line_nos: List[int] = []
        if rx is not None and not after_missing:
            for i in range(start_line, len(lines)):
                if rx.search(lines[i]):
                    matched_line_nos.append(i + 1)
                    if len(matched_line_nos) >= min_occ and not step.get("collect_all"):
                        break
        ok = (not after_missing) and len(matched_line_nos) >= min_occ
        if ok and matched_line_nos:
            name_to_first_line[name] = matched_line_nos[0] - 1  # zero-based index

        step_result = {
            "name": name,
            "pattern": pattern,
            "after": after_name,
            "start_line": start_line + 1,
            "min_occurrences": min_occ,
            "matched_line_nos": matched_line_nos,
            "pass": ok,
        }
        if after_missing:
            step_result["error"] = f"predecessor '{after_name}' did not match"
        result["steps"].append(step_result)
        if not ok:
            result["pass"] = False
    return result
