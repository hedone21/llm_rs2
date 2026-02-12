"""
Schema Registry — Field metadata for flexible rendering.

Each entry maps a JSON field name to rendering hints consumed by the frontend.
To support a new timeseries field, add ONE line here and the dashboard will
automatically render it.

Keys:
    label      — Human-readable display name
    unit       — Unit string shown on axis / tooltip
    color      — Default trace colour (hex)
    group      — Subplot grouping key (fields in the same group share a subplot)
    chart      — "line" | "multi_line" (one trace per array element)
    style      — Optional Plotly dash style ("dash", "dot", "dashdot")
    transform  — Optional numeric transform applied before rendering
    axis_label — Optional y-axis label override for the group
"""

TIMESERIES_FIELDS = {
    # ── Thermal ──────────────────────────────────────────────
    "temp_c": {
        "label": "Battery Temp",
        "unit": "°C",
        "color": "#ef4444",
        "group": "thermal",
        "chart": "line",
        "axis_label": "Temperature (°C)",
    },

    # ── CPU Frequency ────────────────────────────────────────
    "cpu_freqs_khz": {
        "label": "CPU Freq",
        "unit": "GHz",
        "group": "cpu_freq",
        "chart": "multi_line",
        "transform": "khz_to_ghz",
        "axis_label": "CPU Frequency (GHz)",
    },

    # ── GPU Frequency ────────────────────────────────────────
    "gpu_freq_hz": {
        "label": "GPU Freq",
        "unit": "MHz",
        "color": "#22c55e",
        "group": "gpu_freq",
        "chart": "line",
        "transform": "hz_to_mhz",
        "axis_label": "GPU Frequency (MHz)",
    },

    # ── Memory ───────────────────────────────────────────────
    "mem_used_mb": {
        "label": "System Memory",
        "unit": "MB",
        "color": "#3b82f6",
        "group": "memory",
        "chart": "line",
        "axis_label": "Memory (MB)",
    },
    "process_mem_mb": {
        "label": "Process Memory",
        "unit": "MB",
        "color": "#93c5fd",
        "group": "memory",
        "chart": "line",
        "style": "dash",
    },

    # ── CPU Load ─────────────────────────────────────────────
    "cpu_load_percent": {
        "label": "System CPU Load",
        "unit": "%",
        "color": "#8b5cf6",
        "group": "cpu_load",
        "chart": "line",
        "axis_label": "CPU Load (%)",
    },
    "process_cpu_percent": {
        "label": "Process CPU",
        "unit": "%",
        "color": "#c4b5fd",
        "group": "cpu_load",
        "chart": "line",
        "style": "dash",
    },

    # ── GPU Load ─────────────────────────────────────────────
    "gpu_load_percent": {
        "label": "GPU Load",
        "unit": "%",
        "color": "#06b6d4",
        "group": "gpu_load",
        "chart": "line",
        "axis_label": "GPU Load (%)",
    },
}


BENCHMARK_RESULT_FIELDS = {
    "ttft_ms":        {"label": "TTFT",       "unit": "ms",    "description": "Time To First Token"},
    "tbt_ms":         {"label": "Avg TBT",    "unit": "ms",    "description": "Average Time Between Tokens"},
    "tokens_per_sec": {"label": "Tokens/sec", "unit": "tok/s", "description": "Generation speed"},
}


METADATA_FIELDS = {
    "date":           {"label": "Date"},
    "model":          {"label": "Model"},
    "backend":        {"label": "Backend"},
    "num_tokens":     {"label": "Tokens"},
    "prefill_type":   {"label": "Prefill Type"},
    "command":        {"label": "Command"},
    "cpu_models":     {"label": "CPU Cores"},
    "foreground_app": {"label": "FG App"},
}


# ── Transform functions ─────────────────────────────────────

TRANSFORMS = {
    "khz_to_ghz": lambda v: round(v / 1_000_000, 3),
    "hz_to_mhz":  lambda v: round(v / 1_000_000, 1),
}


def apply_transform(transform_name, value):
    """Apply a named transform. Returns value unchanged if transform unknown."""
    fn = TRANSFORMS.get(transform_name)
    if fn is None:
        return value
    if isinstance(value, list):
        return [fn(v) for v in value]
    return fn(value)


def get_field_descriptor(field_key):
    """Return the descriptor for a known field, or auto-generate one."""
    if field_key in TIMESERIES_FIELDS:
        desc = dict(TIMESERIES_FIELDS[field_key])
        desc["key"] = field_key
        desc["known"] = True
        return desc

    # Auto-detect: unknown field
    return {
        "key": field_key,
        "label": field_key.replace("_", " ").title(),
        "unit": "",
        "group": f"auto_{field_key}",
        "chart": "line",
        "known": False,
    }
