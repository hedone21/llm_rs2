#!/usr/bin/env python3
"""Multi-model regression test for llm.rs inference correctness and performance.

Usage:
    python scripts/regression_test.py                    # Run all tests
    python scripts/regression_test.py --save-baseline    # Save current results as baseline
    python scripts/regression_test.py --models llama     # Test specific model
    python scripts/regression_test.py --backends cpu     # Test specific backend
    python scripts/regression_test.py --nvidia            # Include NVIDIA GPU tests
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BASELINE_PATH = PROJECT_ROOT / "results" / "data" / "regression_baseline.json"
BINARY = str(PROJECT_ROOT / "target" / "release" / "generate")

# Test matrix: model, backend, extra_args
MODELS = {
    "llama": {
        "path": "models/llama3.2-1b",
        "display": "Llama 3.2 1B",
    },
    "qwen": {
        "path": "models/qwen2.5-1.5b",
        "display": "Qwen 2.5 1.5B",
    },
    "gemma": {
        "path": "models/gemma3-1b",
        "display": "Gemma 3 1B",
    },
}

BACKENDS = {
    "cpu": {"env": {}, "args": "-b cpu"},
    "pocl": {"env": {"OCL_PLATFORM": "Portable", "OCL_DEVICE_TYPE": "cpu"}, "args": "-b opencl"},
    "nvidia": {"env": {"OCL_PLATFORM": "NVIDIA"}, "args": "-b opencl"},
}

PROMPTS = [
    {"text": "Hello, how are you?", "n": 32, "label": "greeting"},
    {"text": "The capital of France is", "n": 16, "label": "factual"},
    {"text": "Once upon a time", "n": 64, "label": "creative"},
]

# Thresholds
TOKEN_MATCH_THRESHOLD = 0.90  # 90% token match
PERF_REGRESSION_THRESHOLD = 0.05  # 5% performance regression


def build_binary():
    """Build the release binary."""
    print("Building release binary...")
    result = subprocess.run(
        ["cargo", "build", "--release", "-p", "llm_rs2", "--bin", "generate"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)
    print("Build OK")


def run_generate(model_path, prompt, n_tokens, backend_key, temperature=0.0):
    """Run generate binary and parse output."""
    backend = BACKENDS[backend_key]
    env = os.environ.copy()
    env.update(backend["env"])
    env["RUST_LOG"] = "error"

    cmd = [
        BINARY,
        "--model-path", str(PROJECT_ROOT / model_path),
        "--prompt", prompt,
        "-n", str(n_tokens),
        "--temperature", str(temperature),
    ] + backend["args"].split()

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "generated": "", "metrics": {}}

    output = result.stdout + result.stderr

    # Parse generated text
    generated = ""
    for line in output.split('\n'):
        if prompt in line and not line.startswith('['):
            generated = line.replace(prompt, "", 1).strip()
            break

    # Parse metrics
    metrics = {}
    m = re.search(r"Prefill:\s+([\d.]+)\s+ms\s+\((\d+)\s+tokens,\s+([\d.]+)\s+tok/s\)", output)
    if m:
        metrics["prefill_ms"] = float(m.group(1))
        metrics["prefill_toks"] = float(m.group(3))

    m = re.search(r"TTFT:\s+([\d.]+)\s+ms", output)
    if m:
        metrics["ttft_ms"] = float(m.group(1))

    m = re.search(r"Decode:\s+([\d.]+)\s+ms/tok\s+\(([\d.]+)\s+tok/s\)\s+\[(\d+)\s+tokens", output)
    if m:
        metrics["decode_ms_per_tok"] = float(m.group(1))
        metrics["decode_toks"] = float(m.group(2))
        metrics["decode_tokens"] = int(m.group(3))

    error = None
    if result.returncode != 0 and not generated:
        error = output[-200:] if len(output) > 200 else output

    return {"generated": generated, "metrics": metrics, "error": error}


def token_match_ratio(text_a, text_b):
    """Compute token overlap ratio (word-level)."""
    if not text_a or not text_b:
        return 0.0
    tokens_a = text_a.split()
    tokens_b = text_b.split()
    if not tokens_a:
        return 0.0
    matches = sum(1 for a, b in zip(tokens_a, tokens_b) if a == b)
    return matches / max(len(tokens_a), len(tokens_b))


def is_coherent(text):
    """Basic coherence check: contains real words, no excessive repetition."""
    if not text or len(text) < 5:
        return False
    words = text.split()
    if len(words) < 2:
        return False
    # Check for excessive repetition
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.2:
        return False
    # Check for garbage patterns
    if re.search(r'[<>]{3,}|unused\d+', text):
        return False
    return True


def run_tests(models_filter, backends_filter):
    """Run all test combinations and return results."""
    results = {}

    for model_key, model_info in MODELS.items():
        if models_filter and model_key not in models_filter:
            continue
        if not (PROJECT_ROOT / model_info["path"]).exists():
            print(f"  SKIP {model_info['display']}: model not found at {model_info['path']}")
            continue

        for backend_key in backends_filter:
            for prompt_info in PROMPTS:
                test_key = f"{model_key}/{backend_key}/{prompt_info['label']}"
                print(f"  {test_key}...", end=" ", flush=True)

                result = run_generate(
                    model_info["path"],
                    prompt_info["text"],
                    prompt_info["n"],
                    backend_key,
                )

                if result.get("error"):
                    print(f"ERROR: {result['error'][:80]}")
                else:
                    coherent = is_coherent(result["generated"])
                    toks = result["metrics"].get("decode_toks", 0)
                    print(f"{'OK' if coherent else 'FAIL'} | {toks:.1f} tok/s | {result['generated'][:60]}...")

                results[test_key] = {
                    "model": model_key,
                    "backend": backend_key,
                    "prompt": prompt_info["label"],
                    "generated": result["generated"],
                    "metrics": result["metrics"],
                    "error": result.get("error"),
                    "coherent": is_coherent(result["generated"]),
                }

    return results


def compare_with_baseline(results, baseline):
    """Compare current results with baseline. Return (pass, report)."""
    all_pass = True
    report_lines = []

    for test_key, current in results.items():
        if current.get("error"):
            report_lines.append(f"  FAIL {test_key}: error — {current['error'][:80]}")
            all_pass = False
            continue

        if not current["coherent"]:
            report_lines.append(f"  FAIL {test_key}: incoherent output")
            all_pass = False
            continue

        if test_key in baseline:
            base = baseline[test_key]

            # Token match
            match_ratio = token_match_ratio(current["generated"], base["generated"])
            if match_ratio < TOKEN_MATCH_THRESHOLD:
                report_lines.append(
                    f"  FAIL {test_key}: token match {match_ratio:.1%} < {TOKEN_MATCH_THRESHOLD:.0%}")
                all_pass = False
            else:
                report_lines.append(f"  PASS {test_key}: token match {match_ratio:.1%}")

            # Performance regression
            base_toks = base["metrics"].get("decode_toks", 0)
            curr_toks = current["metrics"].get("decode_toks", 0)
            if base_toks > 0 and curr_toks > 0:
                regression = (base_toks - curr_toks) / base_toks
                if regression > PERF_REGRESSION_THRESHOLD:
                    report_lines.append(
                        f"  WARN {test_key}: perf regression {regression:.1%} "
                        f"({base_toks:.1f} → {curr_toks:.1f} tok/s)")
        else:
            if current["coherent"]:
                report_lines.append(f"  PASS {test_key}: coherent (no baseline)")
            else:
                report_lines.append(f"  FAIL {test_key}: incoherent (no baseline)")
                all_pass = False

    return all_pass, report_lines


def main():
    parser = argparse.ArgumentParser(description="llm.rs multi-model regression test")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save current results as baseline")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        help="Test specific models")
    parser.add_argument("--backends", nargs="+", choices=list(BACKENDS.keys()),
                        help="Test specific backends")
    parser.add_argument("--nvidia", action="store_true",
                        help="Include NVIDIA GPU tests")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip cargo build")
    args = parser.parse_args()

    models_filter = args.models
    backends_filter = args.backends or ["cpu"]
    if args.nvidia and "nvidia" not in backends_filter:
        backends_filter.append("nvidia")

    if not args.skip_build:
        build_binary()

    print(f"\n=== llm.rs Regression Test ===")
    print(f"Models:   {models_filter or 'all'}")
    print(f"Backends: {backends_filter}")
    print()

    results = run_tests(models_filter, backends_filter)

    # Load baseline if exists
    baseline = {}
    if BASELINE_PATH.exists() and not args.save_baseline:
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)

    # Compare
    print(f"\n=== Results ===")
    if baseline:
        all_pass, report = compare_with_baseline(results, baseline)
        for line in report:
            print(line)
        print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    else:
        # No baseline — just check coherence
        all_pass = True
        for key, res in results.items():
            status = "PASS" if res["coherent"] else "FAIL"
            if not res["coherent"]:
                all_pass = False
            toks = res["metrics"].get("decode_toks", 0)
            print(f"  {status} {key}: {toks:.1f} tok/s")
        print(f"\nOverall: {'PASS' if all_pass else 'FAIL'} (no baseline — coherence check only)")

    # Save baseline
    if args.save_baseline:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_PATH, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nBaseline saved to {BASELINE_PATH}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
