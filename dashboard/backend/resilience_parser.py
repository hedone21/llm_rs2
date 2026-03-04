"""
Resilience system data provider for the dashboard.

Returns the resilience architecture reference: signal types, strategies,
action mappings, operating modes — useful for team to understand the system.
"""


def load_resilience_data():
    """Return resilience system reference data."""
    return {
        "summary": {
            "signals": 4,
            "strategies": 4,
            "actions": 7,
            "modes": 4,
            "phase": "Phase 3 — generate.rs integration",
            "status": "implemented",
        },
        "operating_modes": [
            {
                "mode": "Normal",
                "condition": "All signals Normal",
                "description": "No constraints applied",
                "color": "#22c55e",
            },
            {
                "mode": "Degraded",
                "condition": "Any signal >= Warning",
                "description": "Lightweight constraints (backend switch)",
                "color": "#f59e0b",
            },
            {
                "mode": "Minimal",
                "condition": "Any signal >= Critical",
                "description": "Aggressive constraints (evict + throttle + limit)",
                "color": "#ef4444",
            },
            {
                "mode": "Suspended",
                "condition": "Any signal = Emergency",
                "description": "Inference paused",
                "color": "#6b7280",
            },
        ],
        "signals": [
            {
                "name": "MemoryPressure",
                "dbus_member": "MemoryPressure",
                "params": "level, available_bytes, reclaim_target_bytes",
                "description": "System memory pressure from D-Bus",
            },
            {
                "name": "ThermalAlert",
                "dbus_member": "ThermalAlert",
                "params": "level, temperature_mc, throttling_active, throttle_ratio",
                "description": "Device thermal throttling status",
            },
            {
                "name": "ComputeGuidance",
                "dbus_member": "ComputeGuidance",
                "params": "level, backend, reason, cpu_pct, gpu_pct",
                "description": "CPU/GPU utilization guidance",
            },
            {
                "name": "EnergyConstraint",
                "dbus_member": "EnergyConstraint",
                "params": "level, reason, power_budget_mw",
                "description": "Battery / power budget constraint",
            },
        ],
        "strategies": _build_strategy_matrix(),
        "actions": [
            {"name": "Evict", "target": "KV Cache", "description": "prune_prefix() on all layers"},
            {"name": "SwitchBackend", "target": "Backend", "description": "Switch to CPU/GPU (log only in Phase 3)"},
            {"name": "LimitTokens", "target": "Generation", "description": "Cap max generated tokens"},
            {"name": "Throttle", "target": "Generation", "description": "Insert delay between tokens"},
            {"name": "Suspend", "target": "Generation", "description": "Break inference loop"},
            {"name": "RejectNew", "target": "Server", "description": "Reject new requests (server mode)"},
            {"name": "RestoreDefaults", "target": "All", "description": "Clear all constraints"},
        ],
        "conflict_rules": [
            "Suspend overrides all other actions",
            "Most aggressive eviction ratio (minimum) wins",
            "Largest throttle delay wins",
            "Smallest token limit wins",
            "CPU backend preferred over GPU (safety first)",
            "RestoreDefaults only when no other constraints exist",
        ],
    }


def _build_strategy_matrix():
    """Build the signal×level → action matrix."""
    return {
        "Memory": {
            "Normal": [{"action": "RestoreDefaults"}],
            "Warning": [{"action": "Evict", "detail": "target_ratio: 0.85"}],
            "Critical": [
                {"action": "Evict", "detail": "target_ratio: 0.50"},
                {"action": "LimitTokens", "detail": "max: 32"},
            ],
            "Emergency": [{"action": "Suspend"}],
        },
        "Thermal": {
            "Normal": [{"action": "RestoreDefaults"}],
            "Warning": [{"action": "SwitchBackend", "detail": "→ CPU"}],
            "Critical": [
                {"action": "SwitchBackend", "detail": "→ CPU"},
                {"action": "Throttle", "detail": "(1-ratio)*100 ms"},
                {"action": "LimitTokens", "detail": "max: 64"},
            ],
            "Emergency": [{"action": "Suspend"}],
        },
        "Energy": {
            "Normal": [{"action": "RestoreDefaults"}],
            "Warning": [{"action": "SwitchBackend", "detail": "→ CPU"}],
            "Critical": [
                {"action": "SwitchBackend", "detail": "→ CPU"},
                {"action": "Throttle", "detail": "100 ms"},
            ],
            "Emergency": [{"action": "Suspend"}],
        },
        "Compute": {
            "Normal": [{"action": "RestoreDefaults"}],
            "Warning": [{"action": "SwitchBackend", "detail": "→ recommended"}],
            "Critical": [
                {"action": "SwitchBackend", "detail": "→ CPU"},
                {"action": "Throttle", "detail": "50 ms"},
            ],
            "Emergency": [{"action": "Suspend"}],
        },
    }
