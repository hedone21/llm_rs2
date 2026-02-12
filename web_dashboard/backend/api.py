"""
REST API routes for the benchmark dashboard.
"""

from flask import Blueprint, jsonify, request

from . import parser
from .schema_registry import TIMESERIES_FIELDS, BENCHMARK_RESULT_FIELDS, METADATA_FIELDS
from .runner import BenchmarkRunner

api = Blueprint("api", __name__, url_prefix="/api")

# Singleton runner instance
_runner = BenchmarkRunner()


# ── Profile endpoints ────────────────────────────────────────

@api.route("/profiles")
def list_profiles():
    """Return lightweight summaries of all profiles."""
    summaries = parser.load_all_summaries()
    return jsonify({"profiles": summaries, "total": len(summaries)})


@api.route("/profiles/<profile_id>")
def get_profile(profile_id):
    """Return full profile data including timeseries and field descriptors."""
    fp = parser.get_profile_path(profile_id)
    if fp is None:
        return jsonify({"error": f"Profile '{profile_id}' not found"}), 404
    data = parser.load_profile_full(fp)
    if data is None:
        return jsonify({"error": "Failed to parse profile"}), 500
    return jsonify(data)


@api.route("/profiles/<profile_id>/timeseries")
def get_timeseries(profile_id):
    """Return only timeseries + field descriptors for a profile."""
    fp = parser.get_profile_path(profile_id)
    if fp is None:
        return jsonify({"error": f"Profile '{profile_id}' not found"}), 404
    data = parser.load_profile_full(fp)
    if data is None:
        return jsonify({"error": "Failed to parse profile"}), 500
    return jsonify({
        "id": data["id"],
        "timeseries": data["timeseries"],
        "field_descriptors": data["field_descriptors"],
        "events": data.get("events", []),
    })


@api.route("/compare")
def compare_profiles():
    """Return full data for multiple profiles."""
    ids_param = request.args.get("ids", "")
    if not ids_param:
        return jsonify({"error": "Missing 'ids' query parameter"}), 400
    ids = [i.strip() for i in ids_param.split(",") if i.strip()]
    if len(ids) < 2:
        return jsonify({"error": "Need at least 2 profile IDs"}), 400

    profiles = []
    for pid in ids:
        fp = parser.get_profile_path(pid)
        if fp is None:
            return jsonify({"error": f"Profile '{pid}' not found"}), 404
        data = parser.load_profile_full(fp)
        if data is None:
            return jsonify({"error": f"Failed to parse profile '{pid}'"}), 500
        profiles.append(data)

    return jsonify({"profiles": profiles})


# ── Schema endpoint ──────────────────────────────────────────

@api.route("/schema")
def get_schema():
    """Return the current Schema Registry."""
    return jsonify({
        "timeseries_fields": TIMESERIES_FIELDS,
        "benchmark_result_fields": BENCHMARK_RESULT_FIELDS,
        "metadata_fields": METADATA_FIELDS,
    })


# ── Benchmark runner endpoints ───────────────────────────────

@api.route("/benchmark/run", methods=["POST"])
def run_benchmark():
    """Trigger a benchmark execution."""
    params = request.get_json(silent=True) or {}
    backend = params.get("backend", "cpu")
    prefill_type = params.get("prefill_type", "short_len")
    num_tokens = params.get("num_tokens", 128)

    if _runner.is_running():
        return jsonify({"error": "A benchmark is already running"}), 409

    _runner.start(backend=backend, prefill_type=prefill_type, num_tokens=num_tokens)
    return jsonify({"status": "started", "params": {
        "backend": backend,
        "prefill_type": prefill_type,
        "num_tokens": num_tokens,
    }})


@api.route("/benchmark/status")
def benchmark_status():
    """Return current benchmark execution status."""
    return jsonify(_runner.get_status())
