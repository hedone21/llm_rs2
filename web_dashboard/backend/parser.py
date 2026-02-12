"""
Flexible JSON Profile Parser.

Loads benchmark profile JSON files and normalizes them into a consistent
format regardless of schema version.  Unknown fields are preserved and
auto-described via the Schema Registry.
"""

import json
import glob
import os
from pathlib import Path

from .schema_registry import (
    TIMESERIES_FIELDS,
    BENCHMARK_RESULT_FIELDS,
    METADATA_FIELDS,
    apply_transform,
    get_field_descriptor,
)

# Path to the results data directory (relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "results" / "data"


def _safe_float(val, default=None):
    """Safely convert to float."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _extract_temp_range(timeseries):
    """Extract start, end, and max temperature from timeseries."""
    temps = [s.get("temp_c") for s in timeseries if s.get("temp_c") is not None]
    if not temps:
        return None, None, None
    return temps[0], temps[-1], max(temps)


def _detect_timeseries_fields(timeseries):
    """Discover all field keys present in timeseries data."""
    keys = set()
    for sample in timeseries:
        for k, v in sample.items():
            if k == "timestamp":
                continue
            keys.add(k)
    return sorted(keys)


def load_profile_summary(filepath):
    """
    Load a profile JSON and return a lightweight summary (no timeseries).
    Returns None if file is invalid.
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    meta = data.get("metadata", {})
    results = data.get("benchmark_results", {})
    baseline = data.get("baseline", {})
    timeseries = data.get("timeseries", [])

    start_temp, end_temp, max_temp = _extract_temp_range(timeseries)

    filename = os.path.basename(filepath)
    profile_id = os.path.splitext(filename)[0]

    return {
        "id": profile_id,
        "filename": filename,
        "version": data.get("version", 0),
        "metadata": {
            "date": meta.get("date"),
            "model": meta.get("model", "unknown"),
            "backend": meta.get("backend", "unknown"),
            "num_tokens": meta.get("num_tokens"),
            "prefill_type": meta.get("prefill_type"),
            "foreground_app": meta.get("foreground_app"),
        },
        "results": {
            "ttft_ms": _safe_float(results.get("ttft_ms")),
            "tbt_ms": _safe_float(results.get("tbt_ms")),
            "tokens_per_sec": _safe_float(results.get("tokens_per_sec")),
        },
        "thermal": {
            "start_temp": start_temp,
            "end_temp": end_temp,
            "max_temp": max_temp,
        },
        "baseline": {
            "avg_memory_used_mb": _safe_float(baseline.get("avg_memory_used_mb")),
            "avg_gpu_load_percent": _safe_float(baseline.get("avg_gpu_load_percent")),
            "avg_start_temp_c": _safe_float(baseline.get("avg_start_temp_c")),
        },
        "timeseries_count": len(timeseries),
    }


def load_profile_full(filepath):
    """
    Load a profile JSON and return the full normalized data with
    field descriptors for the timeseries.
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    meta = data.get("metadata", {})
    results = data.get("benchmark_results", {})
    baseline = data.get("baseline", {})
    timeseries = data.get("timeseries", [])
    events = data.get("events", [])

    # Discover which fields are in the timeseries
    ts_keys = _detect_timeseries_fields(timeseries)

    # Build field descriptors
    field_descriptors = [get_field_descriptor(k) for k in ts_keys]

    # Transform timeseries values where needed
    transformed_timeseries = []
    for sample in timeseries:
        transformed = {"timestamp": sample.get("timestamp")}
        for key in ts_keys:
            raw_val = sample.get(key)
            if raw_val is None:
                transformed[key] = None
                continue
            desc = TIMESERIES_FIELDS.get(key, {})
            transform = desc.get("transform")
            if transform:
                transformed[key] = apply_transform(transform, raw_val)
            else:
                transformed[key] = raw_val
        transformed_timeseries.append(transformed)

    filename = os.path.basename(filepath)
    profile_id = os.path.splitext(filename)[0]

    start_temp, end_temp, max_temp = _extract_temp_range(timeseries)

    return {
        "id": profile_id,
        "filename": filename,
        "version": data.get("version", 0),
        "metadata": meta,
        "results": results,
        "baseline": baseline,
        "events": events,
        "thermal": {
            "start_temp": start_temp,
            "end_temp": end_temp,
            "max_temp": max_temp,
        },
        "timeseries": transformed_timeseries,
        "field_descriptors": field_descriptors,
    }


def load_all_summaries():
    """Scan results/data/ and return summaries for all profiles."""
    pattern = str(DATA_DIR / "*.json")
    files = sorted(glob.glob(pattern))
    summaries = []
    for fp in files:
        s = load_profile_summary(fp)
        if s is not None:
            summaries.append(s)
    return summaries


def get_profile_path(profile_id):
    """Convert a profile ID to a file path, or None if not found."""
    fp = DATA_DIR / f"{profile_id}.json"
    if fp.exists():
        return str(fp)
    return None
