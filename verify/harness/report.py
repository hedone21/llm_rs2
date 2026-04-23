"""Summary report writers (Markdown + JSONL)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _fmt(v: Any, fmt: str = "{:.3f}") -> str:
    try:
        if v is None:
            return "-"
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def write_summary_jsonl(results: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def write_summary_md(results: List[Dict[str, Any]], out_path: Path) -> None:
    """Simple | scenario | device | model | pass | rouge_l | tbt_delta | table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Resilience Verify Summary\n",
        "",
        "| Scenario | Device | Model | Run | Pass | Crash | Tokens | ROUGE-L | BLEU-4 | TBT Δ% |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        acc = (r.get("accuracy") or {}).get("details") or {}
        perf = (r.get("performance") or {}).get("details") or {}
        cp = (r.get("crash_and_progress") or {}).get("details") or {}
        crash_hits = cp.get("crash_hits") or []
        crash_cell = "CRASH" if crash_hits else "ok"
        actual = cp.get("actual_decode_tokens")
        req = cp.get("requested_decode_tokens")
        tok_cell = f"{actual}/{req}" if actual is not None and req is not None else "-"
        lines.append(
            "| {scen} | {dev} | {model} | {run} | {pass_} | {crash} | {tok} | {rouge} | {bleu} | {tbt} |".format(
                scen=r.get("scenario_id", "?"),
                dev=r.get("device", "?"),
                model=r.get("model", "?"),
                run=r.get("run_idx", 0),
                pass_="YES" if r.get("overall_pass") else "NO",
                crash=crash_cell,
                tok=tok_cell,
                rouge=_fmt(acc.get("rouge_l_f1")),
                bleu=_fmt(acc.get("bleu_4")),
                tbt=_fmt(perf.get("delta_pct"), "{:+.2f}"),
            )
        )
    lines.append("")
    total = len(results)
    passed = sum(1 for r in results if r.get("overall_pass"))
    lines.append(f"**Total:** {total}  •  **Passed:** {passed}  •  **Failed:** {total - passed}")
    lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def render_console_table(results: List[Dict[str, Any]]) -> str:
    """Human-readable table for stdout."""
    headers = ["Scenario", "Device", "Model", "Run", "Pass", "Crash", "Tokens", "ROUGE-L", "BLEU-4", "TBT Δ%"]
    rows: List[List[str]] = []
    for r in results:
        acc = (r.get("accuracy") or {}).get("details") or {}
        perf = (r.get("performance") or {}).get("details") or {}
        cp = (r.get("crash_and_progress") or {}).get("details") or {}
        crash_hits = cp.get("crash_hits") or []
        crash_cell = "CRASH" if crash_hits else "ok"
        actual = cp.get("actual_decode_tokens")
        req = cp.get("requested_decode_tokens")
        tok_cell = f"{actual}/{req}" if actual is not None and req is not None else "-"
        rows.append([
            str(r.get("scenario_id", "?")),
            str(r.get("device", "?")),
            str(r.get("model", "?")),
            str(r.get("run_idx", 0)),
            "PASS" if r.get("overall_pass") else "FAIL",
            crash_cell,
            tok_cell,
            _fmt(acc.get("rouge_l_f1")),
            _fmt(acc.get("bleu_4")),
            _fmt(perf.get("delta_pct"), "{:+.2f}"),
        ])
    widths = [max(len(h), *(len(row[i]) for row in rows)) if rows else len(h) for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)
    hline = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    body = "\n".join("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)) for row in rows)
    return "\n".join([hline, sep, body]) if rows else "(no results)"
