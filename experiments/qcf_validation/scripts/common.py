"""Common utilities for QCF validation experiments."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BINARY = PROJECT_ROOT / "target" / "release" / "generate"
MODEL_PATH = PROJECT_ROOT / "models" / "llama3.2-1b"
PROMPTS_FILE = PROJECT_ROOT / "experiments" / "prompts" / "benchmark_prompts.json"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

BUDGETS = {
    "B0": 2048,  # baseline (no eviction)
    "B1": 1600,
    "B2": 1200,
    "B3": 900,
    "B4": 600,
    "B5": 350,
}

POLICIES = ["sliding", "h2o"]


def ensure_binary():
    """Build release binary if not present."""
    if not BINARY.exists():
        print("[build] Compiling release binary...", file=sys.stderr)
        subprocess.run(
            ["cargo", "build", "--release", "-p", "llm_rs2", "--bin", "generate"],
            cwd=PROJECT_ROOT,
            check=True,
        )
    assert BINARY.exists(), f"Binary not found: {BINARY}"


def load_prompts():
    """Load benchmark_prompts.json."""
    with open(PROMPTS_FILE) as f:
        return json.load(f)


def run_ppl(text_file: str, policy: str, budget: int, **extra_args) -> dict:
    """Run PPL evaluation and return parsed JSON result.

    Returns dict with keys: ppl, total_nll, token_count, qcf_metrics, ...
    """
    cmd = [
        str(BINARY),
        "--model-path", str(MODEL_PATH),
        "--ppl", text_file,
        "--backend", "cpu",
        "--kv-type", "f32",
        "--temperature", "0",
        "--kv-layout", "head",
        "--max-seq-len", "2048",
        "--eviction-policy", policy,
        "--protected-prefix", "4",
    ]
    if budget < 2048:
        cmd += ["--kv-budget", str(budget)]

    if policy in ("h2o", "h2o_plus"):
        cmd += ["--h2o-keep-ratio", str(extra_args.get("keep_ratio", 0.5))]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"[ERROR] PPL failed: {result.stderr[-500:]}", file=sys.stderr)
        return {"ppl": float("inf"), "qcf_metrics": [], "error": result.stderr[-200:]}

    # stdout contains JSON result
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        # Try to find JSON in output
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return {"ppl": float("inf"), "qcf_metrics": [], "error": "JSON parse failed"}


def run_generate(prompt: str, n_tokens: int, policy: str, budget: int,
                 **extra_args) -> str:
    """Run text generation and return generated text (after prompt)."""
    cmd = [
        str(BINARY),
        "--model-path", str(MODEL_PATH),
        "--prompt", prompt,
        "-n", str(n_tokens),
        "--backend", "cpu",
        "--kv-type", "f32",
        "--temperature", "0",
        "--kv-layout", "head",
        "--max-seq-len", "2048",
        "--eviction-policy", policy,
        "--protected-prefix", "4",
    ]
    if budget < 2048:
        cmd += ["--kv-budget", str(budget)]

    if policy in ("h2o", "h2o_plus"):
        cmd += ["--h2o-keep-ratio", str(extra_args.get("keep_ratio", 0.5))]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"[ERROR] Generate failed: {result.stderr[-300:]}", file=sys.stderr)
        return ""

    # stdout contains prompt + generated text (streaming format)
    output = result.stdout.strip()
    # The last line of stdout is the full text
    # Split by \r (carriage returns from streaming) and take last
    lines = output.split("\r")
    full_text = lines[-1] if lines else output

    # Remove the prompt prefix to get just generated text
    if full_text.startswith(prompt[:50]):
        # Find where prompt ends and generation begins
        # Use a rough heuristic: skip prompt length
        gen_start = len(prompt)
        if gen_start < len(full_text):
            return full_text[gen_start:]
    return full_text


def run_eval_ll(batch_json_path: str, policy: str, budget: int,
                **extra_args) -> dict:
    """Run eval-ll mode with a batch JSON file and return results."""
    cmd = [
        str(BINARY),
        "--model-path", str(MODEL_PATH),
        "--eval-ll",
        "--eval-batch", batch_json_path,
        "--backend", "cpu",
        "--kv-type", "f32",
        "--temperature", "0",
        "--kv-layout", "head",
        "--max-seq-len", "2048",
        "--eviction-policy", policy,
        "--protected-prefix", "4",
    ]
    if budget < 2048:
        cmd += ["--kv-budget", str(budget)]

    if policy in ("h2o", "h2o_plus"):
        cmd += ["--h2o-keep-ratio", str(extra_args.get("keep_ratio", 0.5))]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        print(f"[ERROR] Eval-LL failed: {result.stderr[-500:]}", file=sys.stderr)
        return {"results": [], "error": result.stderr[-200:]}

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return {"results": [], "error": "JSON parse failed"}


def save_result(phase: str, name: str, data: dict):
    """Save result JSON to results/<phase>/<name>.json."""
    out_dir = RESULTS_DIR / phase
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def get_qcf_metrics(result: dict) -> list:
    """Extract QCF metrics from result, supporting both old and new key names."""
    return result.get("qcf_metrics", result.get("proxy_metrics", []))


def compute_qcf_summary(qcf_metrics: list) -> dict:
    """Compute summary statistics from qcf_metrics list."""
    if not qcf_metrics:
        return {"count": 0, "avg": 0.0, "total": 0.0, "max": 0.0}
    values = [m["raw_value"] for m in qcf_metrics if "raw_value" in m]
    if not values:
        return {"count": 0, "avg": 0.0, "total": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "total": sum(values),
        "max": max(values),
    }


def compute_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score between prediction and reference strings."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_answer(generated: str, expected: str) -> bool:
    """Check if expected answer is contained in generated text (case-insensitive)."""
    return expected.lower() in generated.lower()
