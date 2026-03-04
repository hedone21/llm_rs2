#!/usr/bin/env python3
"""
Device Stress Test Suite for llm_rs2.

Comprehensive 4-phase stress test:
  Phase 1: Thermal Stability   — sustained OpenCL inference, throttling detection
  Phase 2: Performance Sustain — repeated short inferences, consistency measurement
  Phase 3: Memory Stability    — long sequences + KV eviction, RSS leak detection
  Phase 4: Backend Correctness — CPU vs OpenCL numerical accuracy under heat

Usage:
  python scripts/stress_test_device.py                    # full test
  python scripts/stress_test_device.py --skip-build       # reuse existing binaries
  python scripts/stress_test_device.py --phases 1,4       # specific phases only
  python scripts/stress_test_device.py --dry-run           # preview commands
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

# Add scripts directory to path for android_profile imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from android_profile import DeviceMonitor, parse_results, extract_metadata, run_adb_command

# ─── Constants ───────────────────────────────────────────────────────────────

ANDROID_TMP_DIR = "/data/local/tmp"
MODEL_PATH = f"{ANDROID_TMP_DIR}/models/llama3.2-1b"
GENERATE_BIN_REMOTE = f"{ANDROID_TMP_DIR}/generate"
TEST_BACKEND_BIN_REMOTE = f"{ANDROID_TMP_DIR}/test_backend"
EVAL_DIR = f"{ANDROID_TMP_DIR}/llm_rs2/eval"

GENERATE_BIN_LOCAL = "target/aarch64-linux-android/release/generate"
TEST_BACKEND_BIN_LOCAL = "target/aarch64-linux-android/release/test_backend"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def run_command(cmd, check=True, dry_run=False):
    if dry_run:
        print(f"  [DryRun] {cmd}")
        return
    print(f"  [Exec] {cmd}")
    subprocess.run(cmd, shell=True, check=check)


def run_device_inference(device_cmd, timeout=600):
    """Run generate on device via adb shell. Returns (exit_code, stdout_text, duration_sec)."""
    full_cmd = f'adb shell "LD_LIBRARY_PATH=/data/local/tmp {device_cmd}"'
    start = time.time()
    try:
        result = subprocess.run(
            full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout,
        )
        duration = time.time() - start
        return result.returncode, result.stdout, duration
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Command timed out after {timeout}s")
        return -1, "", timeout


def generate_prompt_file(filename, approx_tokens):
    """Generate a synthetic prompt file with ~approx_tokens tokens."""
    seed = "The quick brown fox jumps over the lazy dog. "
    chars_needed = approx_tokens * 4
    content = (seed * (chars_needed // len(seed) + 1))[:chars_needed]
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w") as f:
        f.write(content)


def coefficient_of_variation(values):
    """CV as percentage."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return (variance ** 0.5) / mean * 100.0


def verdict(value, pass_thr, warn_thr):
    """Return PASS/WARN/FAIL based on thresholds (lower is better)."""
    if value < pass_thr:
        return "PASS"
    elif value < warn_thr:
        return "WARN"
    return "FAIL"


def verdict_eq(value, pass_val):
    """PASS if value equals pass_val, else FAIL."""
    return "PASS" if value == pass_val else "FAIL"


def get_git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def get_max_temp(monitor):
    """Get current max CPU temperature from device."""
    temps = monitor.get_temperatures()
    cpu_temps = [v for k, v in temps.items() if "cpu" in k.lower()]
    return max(cpu_temps) if cpu_temps else 0.0


def wait_for_cooldown(monitor, target_temp=35.0, timeout=90):
    """Wait until device cools below target or timeout."""
    print(f"  [Cooldown] Waiting for device to cool below {target_temp}°C (timeout {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        temp = get_max_temp(monitor)
        if temp > 0 and temp < target_temp:
            print(f"  [Cooldown] Device cooled to {temp:.1f}°C")
            return True
        time.sleep(5)
    print(f"  [Cooldown] Timeout reached, proceeding anyway")
    return False


def get_process_rss_mb(pid):
    """Get RSS of a process in MB."""
    if not pid:
        return 0.0
    try:
        output = run_adb_command(f'shell "cat /proc/{pid}/statm"', check=False)
        if output:
            parts = output.split()
            if len(parts) >= 2:
                return (int(parts[1]) * 4) / 1024.0
    except Exception:
        pass
    return 0.0


# ─── Phase 0: Build & Push ──────────────────────────────────────────────────

def phase_build_push(args):
    print("\n" + "=" * 60)
    print("Phase 0: BUILD & PUSH")
    print("=" * 60)

    # Generate eval prompt files
    os.makedirs("eval", exist_ok=True)
    for tok in [512, 1024]:
        path = f"eval/prefill_{tok}.txt"
        if not os.path.exists(path):
            generate_prompt_file(path, tok)
            print(f"  [Gen] Created {path}")

    if not args.skip_build:
        print("\n  [Build] Building generate and test_backend...")
        run_command(
            "cargo build --target aarch64-linux-android --release --bin generate --bin test_backend",
            dry_run=args.dry_run,
        )

    if not args.skip_push:
        print("\n  [Push] Pushing binaries and eval files...")
        run_command(f"adb push {GENERATE_BIN_LOCAL} {GENERATE_BIN_REMOTE}", dry_run=args.dry_run)
        run_command(f"adb push {TEST_BACKEND_BIN_LOCAL} {TEST_BACKEND_BIN_REMOTE}", dry_run=args.dry_run)
        run_command(f"adb shell chmod +x {GENERATE_BIN_REMOTE} {TEST_BACKEND_BIN_REMOTE}", dry_run=args.dry_run)
        run_command(f"adb shell mkdir -p {EVAL_DIR}", dry_run=args.dry_run)
        run_command(f"adb push eval/ {EVAL_DIR}/", check=False, dry_run=args.dry_run)

    # Verify model exists
    model_check = run_adb_command(f'shell "ls {MODEL_PATH}/config.json 2>/dev/null"', check=False)
    if not model_check or "config.json" not in model_check:
        print(f"  [ERROR] Model not found at {MODEL_PATH}")
        return False
    print("  [OK] Model verified on device")
    return True


# ─── Phase 1: Thermal Stability ─────────────────────────────────────────────

def phase_thermal(monitor, args, timestamp):
    print("\n" + "=" * 60)
    print("Phase 1: THERMAL STABILITY")
    print("  OpenCL 2048-token generation × 3 runs with cooldown")
    print("=" * 60)

    runs = []
    device_cmd = (
        f"{GENERATE_BIN_REMOTE} --model-path {MODEL_PATH} "
        f'--prompt "Hello world" -n 2048 -b opencl --temperature 0'
    )

    for i in range(1, 4):
        print(f"\n  --- Thermal Run {i}/3 ---")
        if args.dry_run:
            print(f'  [DryRun] adb shell "{device_cmd}"')
            runs.append({"run": i, "dry_run": True})
            continue

        start_temp = get_max_temp(monitor)
        monitor.data_buffer.clear()
        monitor.target_pid = None
        monitor.last_proc_cpu_stats = None
        monitor.start()

        exit_code, stdout, duration = run_device_inference(device_cmd, timeout=600)

        monitor.stop()

        # Parse results
        results = parse_results(stdout) if exit_code == 0 else {}

        # Find max temp from timeseries
        max_temp = start_temp
        for snap in monitor.data_buffer:
            for k, v in snap.get("temps", {}).items():
                if "cpu" in k.lower():
                    max_temp = max(max_temp, v)

        run_data = {
            "run": i,
            "exit_code": exit_code,
            "duration_sec": round(duration, 1),
            "start_temp_c": round(start_temp, 1),
            "max_temp_c": round(max_temp, 1),
            "delta_temp_c": round(max_temp - start_temp, 1),
            "ttft_ms": results.get("ttft_ms", 0),
            "tbt_ms": results.get("tbt_ms", 0),
            "tokens_per_sec": results.get("tokens_per_sec", 0),
            "timeseries_samples": len(monitor.data_buffer),
        }
        runs.append(run_data)

        # Save per-run profile
        profile = {
            "version": 1,
            "metadata": {
                **extract_metadata(device_cmd),
                "test_type": "stress_thermal",
                "run_number": i,
                "date": datetime.now().isoformat(),
            },
            "baseline": {},
            "benchmark_results": results,
            "events": [],
            "timeseries": monitor.data_buffer[:],
        }
        fname = f"stress_thermal_opencl_run{i}_{timestamp}.json"
        save_profile(profile, args.output_dir, fname)

        print(f"  Run {i}: TBT={run_data['tbt_ms']:.1f}ms, MaxTemp={run_data['max_temp_c']:.1f}°C")

        if i < 3:
            wait_for_cooldown(monitor, target_temp=35.0, timeout=args.cooldown + 30)

    if args.dry_run:
        return {"verdict": "DRY_RUN", "runs": runs, "summary": {}}

    # Compute summary
    valid_runs = [r for r in runs if r["exit_code"] == 0 and r["tbt_ms"] > 0]
    if len(valid_runs) >= 2:
        throttle_ratio = valid_runs[-1]["tbt_ms"] / valid_runs[0]["tbt_ms"]
    else:
        throttle_ratio = 1.0

    worst_temp = max((r["max_temp_c"] for r in runs), default=0)
    tbt_values = [r["tbt_ms"] for r in valid_runs]
    tbt_var = coefficient_of_variation(tbt_values) if tbt_values else 0

    crashed = any(r.get("exit_code", 0) != 0 for r in runs)
    temp_verdict = verdict(worst_temp, args.thermal_limit, 48.0)
    throttle_verdict = verdict(throttle_ratio, 1.15, 1.30)
    overall = "FAIL" if crashed or temp_verdict == "FAIL" or throttle_verdict == "FAIL" else (
        "WARN" if temp_verdict == "WARN" or throttle_verdict == "WARN" else "PASS"
    )

    summary = {
        "worst_max_temp_c": round(worst_temp, 1),
        "throttle_ratio": round(throttle_ratio, 3),
        "tbt_cv_pct": round(tbt_var, 1),
        "temp_verdict": temp_verdict,
        "throttle_verdict": throttle_verdict,
    }

    print(f"\n  Phase 1 Result: {overall}")
    print(f"    Worst temp: {worst_temp:.1f}°C [{temp_verdict}]")
    print(f"    Throttle ratio: {throttle_ratio:.3f} [{throttle_verdict}]")

    return {"verdict": overall, "runs": runs, "summary": summary}


# ─── Phase 2: Performance Sustainability ─────────────────────────────────────

def phase_sustainability(monitor, args, timestamp):
    print("\n" + "=" * 60)
    print("Phase 2: PERFORMANCE SUSTAINABILITY")
    print("  128-token × 10 iterations per backend (OpenCL, CPU)")
    print("=" * 60)

    backends_result = {}

    for backend in ["opencl", "cpu"]:
        print(f"\n  --- Backend: {backend} ---")
        device_cmd = (
            f"{GENERATE_BIN_REMOTE} --model-path {MODEL_PATH} "
            f'--prompt "Hello world" -n 128 -b {backend} --temperature 0'
        )

        iterations = []
        for i in range(1, 11):
            if args.dry_run:
                print(f'  [DryRun] Iteration {i}: adb shell "{device_cmd}"')
                iterations.append({"iteration": i, "dry_run": True})
                continue

            exit_code, stdout, duration = run_device_inference(device_cmd, timeout=120)
            results = parse_results(stdout) if exit_code == 0 else {}

            iter_data = {
                "iteration": i,
                "exit_code": exit_code,
                "ttft_ms": results.get("ttft_ms", 0),
                "tbt_ms": results.get("tbt_ms", 0),
                "tokens_per_sec": results.get("tokens_per_sec", 0),
            }
            iterations.append(iter_data)
            status = f"TBT={iter_data['tbt_ms']:.1f}ms" if exit_code == 0 else "CRASHED"
            print(f"    Iter {i:2d}/10: {status}")

            if i < 10:
                time.sleep(5)

        if args.dry_run:
            backends_result[backend] = {"iterations": iterations, "verdict": "DRY_RUN"}
            continue

        # Compute stats
        valid = [it for it in iterations if it["exit_code"] == 0 and it["tbt_ms"] > 0]
        tbt_values = [it["tbt_ms"] for it in valid]
        ttft_values = [it["ttft_ms"] for it in valid]

        tbt_cv = coefficient_of_variation(tbt_values) if tbt_values else 0
        ttft_cv = coefficient_of_variation(ttft_values) if ttft_values else 0
        first_vs_last = (tbt_values[-1] / tbt_values[0]) if len(tbt_values) >= 2 else 1.0

        crashed = any(it.get("exit_code", 0) != 0 for it in iterations)
        cv_verdict = verdict(tbt_cv, 10.0, 20.0)
        ratio_verdict = verdict(first_vs_last, 1.10, 1.25)
        overall = "FAIL" if crashed or cv_verdict == "FAIL" or ratio_verdict == "FAIL" else (
            "WARN" if cv_verdict == "WARN" or ratio_verdict == "WARN" else "PASS"
        )

        backend_data = {
            "iterations": iterations,
            "tbt_cv_pct": round(tbt_cv, 1),
            "ttft_cv_pct": round(ttft_cv, 1),
            "first_vs_last_ratio": round(first_vs_last, 3),
            "avg_tbt_ms": round(sum(tbt_values) / len(tbt_values), 1) if tbt_values else 0,
            "avg_ttft_ms": round(sum(ttft_values) / len(ttft_values), 1) if ttft_values else 0,
            "verdict": overall,
        }
        backends_result[backend] = backend_data

        # Save profile
        profile = {
            "version": 1,
            "metadata": {
                **extract_metadata(device_cmd),
                "test_type": "stress_sustainability",
                "iterations": 10,
                "date": datetime.now().isoformat(),
            },
            "baseline": {},
            "benchmark_results": {
                "ttft_ms": backend_data["avg_ttft_ms"],
                "tbt_ms": backend_data["avg_tbt_ms"],
                "tokens_per_sec": round(1000.0 / backend_data["avg_tbt_ms"], 1) if backend_data["avg_tbt_ms"] > 0 else 0,
            },
            "events": [],
            "timeseries": [],
        }
        fname = f"stress_sustainability_{backend}_{timestamp}.json"
        save_profile(profile, args.output_dir, fname)

        print(f"\n  {backend}: TBT CV={tbt_cv:.1f}% [{cv_verdict}], "
              f"First/Last={first_vs_last:.3f} [{ratio_verdict}] → {overall}")

    overall = "FAIL" if any(b.get("verdict") == "FAIL" for b in backends_result.values()) else (
        "WARN" if any(b.get("verdict") == "WARN" for b in backends_result.values()) else "PASS"
    )
    return {"verdict": overall, "backends": backends_result}


# ─── Phase 3: Memory Stability ──────────────────────────────────────────────

def phase_memory(monitor, args, timestamp):
    print("\n" + "=" * 60)
    print("Phase 3: MEMORY STABILITY")
    print("  Long sequences + KV cache eviction + RSS tracking")
    print("=" * 60)

    tests = [
        {
            "name": "no_eviction_opencl",
            "backend": "opencl", "prefill_file": f"{EVAL_DIR}/prefill_512.txt",
            "decode": 1024, "eviction": "none", "window": 0,
            "rss_limit_pass": 20, "rss_limit_warn": 50,
        },
        {
            "name": "eviction_sliding_opencl",
            "backend": "opencl", "prefill_file": f"{EVAL_DIR}/prefill_512.txt",
            "decode": 2048, "eviction": "sliding", "window": 512,
            "rss_limit_pass": 30, "rss_limit_warn": 60,
        },
        {
            "name": "eviction_large_prefill_opencl",
            "backend": "opencl", "prefill_file": f"{EVAL_DIR}/prefill_1024.txt",
            "decode": 1024, "eviction": "sliding", "window": 768,
            "rss_limit_pass": 30, "rss_limit_warn": 60,
        },
        {
            "name": "no_eviction_cpu",
            "backend": "cpu", "prefill_file": f"{EVAL_DIR}/prefill_512.txt",
            "decode": 1024, "eviction": "none", "window": 0,
            "rss_limit_pass": 20, "rss_limit_warn": 50,
        },
    ]

    test_results = {}

    for t in tests:
        print(f"\n  --- {t['name']} ---")

        eviction_flags = ""
        if t["eviction"] != "none":
            eviction_flags = (
                f" --eviction-policy {t['eviction']}"
                f" --eviction-window {t['window']}"
                f" --memory-threshold-mb 256"
            )

        device_cmd = (
            f"{GENERATE_BIN_REMOTE} --model-path {MODEL_PATH} "
            f"--prompt-file {t['prefill_file']} "
            f"-n {t['decode']} -b {t['backend']} "
            f"--max-seq-len 2048 --temperature 0"
            f"{eviction_flags}"
        )

        if args.dry_run:
            print(f'  [DryRun] adb shell "{device_cmd}"')
            test_results[t["name"]] = {"dry_run": True}
            continue

        # Monitor RSS: start monitoring, find PID after launch
        monitor.data_buffer.clear()
        monitor.target_pid = None
        monitor.last_proc_cpu_stats = None
        monitor.start()

        exit_code, stdout, duration = run_device_inference(device_cmd, timeout=600)

        monitor.stop()

        results = parse_results(stdout) if exit_code == 0 else {}

        # Extract RSS from timeseries
        rss_values = [s.get("process_mem_mb", 0) for s in monitor.data_buffer if s.get("process_mem_mb", 0) > 0]
        rss_start = rss_values[0] if rss_values else 0
        rss_peak = max(rss_values) if rss_values else 0
        rss_end = rss_values[-1] if rss_values else 0
        rss_delta = rss_end - rss_start if rss_start > 0 else 0

        eviction_triggered = "evict" in stdout.lower() or "prune" in stdout.lower() if stdout else False

        td = {
            "exit_code": exit_code,
            "completed": exit_code == 0,
            "duration_sec": round(duration, 1),
            "rss_start_mb": round(rss_start, 1),
            "rss_peak_mb": round(rss_peak, 1),
            "rss_end_mb": round(rss_end, 1),
            "rss_delta_mb": round(rss_delta, 1),
            "eviction_triggered": eviction_triggered,
            "ttft_ms": results.get("ttft_ms", 0),
            "tbt_ms": results.get("tbt_ms", 0),
        }

        # Verdict
        if exit_code != 0:
            td["verdict"] = "FAIL"
        else:
            td["verdict"] = verdict(abs(rss_delta), t["rss_limit_pass"], t["rss_limit_warn"])

        test_results[t["name"]] = td

        # Save profile
        profile = {
            "version": 1,
            "metadata": {
                **extract_metadata(device_cmd),
                "test_type": "stress_memory",
                "test_name": t["name"],
                "date": datetime.now().isoformat(),
            },
            "baseline": {},
            "benchmark_results": results,
            "events": [],
            "timeseries": monitor.data_buffer[:],
        }
        fname = f"stress_memory_{t['name']}_{timestamp}.json"
        save_profile(profile, args.output_dir, fname)

        print(f"  {t['name']}: RSS delta={rss_delta:.1f}MB, "
              f"Peak={rss_peak:.1f}MB → {td['verdict']}")

        time.sleep(5)

    overall = "FAIL" if any(r.get("verdict") == "FAIL" for r in test_results.values()) else (
        "WARN" if any(r.get("verdict") == "WARN" for r in test_results.values()) else "PASS"
    )

    print(f"\n  Phase 3 Result: {overall}")
    return {"verdict": overall, "tests": test_results}


# ─── Phase 4: Backend Correctness ───────────────────────────────────────────

def phase_correctness(monitor, args, timestamp):
    print("\n" + "=" * 60)
    print("Phase 4: BACKEND CORRECTNESS UNDER HEAT")
    print("  test_backend: CPU vs OpenCL numerical accuracy")
    print("=" * 60)

    device_cmd = f"{TEST_BACKEND_BIN_REMOTE} --backends auto,opencl"

    if args.dry_run:
        print(f'  [DryRun] adb shell "{device_cmd}"')
        return {"verdict": "DRY_RUN"}

    device_temp = get_max_temp(monitor)
    print(f"  Device temperature at test: {device_temp:.1f}°C")

    exit_code, stdout, duration = run_device_inference(device_cmd, timeout=300)

    if exit_code != 0:
        print(f"  [ERROR] test_backend crashed (exit code {exit_code})")
        return {
            "verdict": "FAIL",
            "reason": f"exit_code={exit_code}",
            "device_temp_at_test_c": round(device_temp, 1),
        }

    # Parse PASS/FAIL from output
    pass_count = len(re.findall(r'\bPASS\b', stdout))
    fail_count = len(re.findall(r'\bFAIL\b', stdout))
    error_count = len(re.findall(r'\bERROR\b', stdout))

    # Extract max numerical error
    errors = re.findall(r'err=\s*([\d.eE+-]+)', stdout)
    max_error = max((float(e) for e in errors), default=0.0) if errors else 0.0

    result = {
        "exit_code": exit_code,
        "duration_sec": round(duration, 1),
        "total_tests": pass_count + fail_count + error_count,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "error_count": error_count,
        "max_error": max_error,
        "device_temp_at_test_c": round(device_temp, 1),
    }

    # Verdict
    if error_count > 0 or fail_count >= 3:
        result["verdict"] = "FAIL"
    elif fail_count > 0:
        result["verdict"] = "WARN"
    else:
        result["verdict"] = "PASS"

    # Save profile
    profile = {
        "version": 1,
        "metadata": {
            "test_type": "stress_correctness",
            "backend": "auto,opencl",
            "date": datetime.now().isoformat(),
        },
        "baseline": {},
        "benchmark_results": {
            "pass_count": pass_count,
            "fail_count": fail_count,
            "max_error": max_error,
        },
        "events": [],
        "timeseries": [],
    }
    fname = f"stress_correctness_{timestamp}.json"
    save_profile(profile, args.output_dir, fname)

    print(f"\n  Tests: {pass_count} PASS / {fail_count} FAIL / {error_count} ERROR")
    print(f"  Max error: {max_error:.6f}")
    print(f"  Phase 4 Result: {result['verdict']}")

    return result


# ─── Output ──────────────────────────────────────────────────────────────────

def save_profile(data, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [Saved] {path}")


def print_report(summary):
    """Print a human-readable final report."""
    print("\n" + "=" * 60)
    print("STRESS TEST REPORT")
    print("=" * 60)

    for phase_name, phase_data in summary.get("phases", {}).items():
        v = phase_data.get("verdict", "N/A")
        icon = {"PASS": "+", "WARN": "!", "FAIL": "X", "DRY_RUN": "~"}.get(v, "?")
        print(f"  [{icon}] {phase_name:25s} {v}")

        # Phase-specific details
        if phase_name == "thermal" and "summary" in phase_data:
            s = phase_data["summary"]
            print(f"       Max temp: {s.get('worst_max_temp_c', '?')}°C, "
                  f"Throttle: {s.get('throttle_ratio', '?')}")
        elif phase_name == "sustainability" and "backends" in phase_data:
            for bk, bd in phase_data["backends"].items():
                print(f"       {bk}: TBT CV={bd.get('tbt_cv_pct', '?')}%, "
                      f"Avg TBT={bd.get('avg_tbt_ms', '?')}ms")
        elif phase_name == "memory" and "tests" in phase_data:
            for tn, td in phase_data["tests"].items():
                if isinstance(td, dict) and "rss_delta_mb" in td:
                    print(f"       {tn}: RSS Δ={td['rss_delta_mb']}MB")
        elif phase_name == "correctness":
            print(f"       {phase_data.get('pass_count', '?')} PASS / "
                  f"{phase_data.get('fail_count', '?')} FAIL")

    overall = summary.get("overall_verdict", "N/A")
    print(f"\n  OVERALL: {overall}")
    print("=" * 60)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Device Stress Test Suite for llm_rs2")
    parser.add_argument("--skip-build", action="store_true", help="Skip cargo build step")
    parser.add_argument("--skip-push", action="store_true", help="Skip adb push step")
    parser.add_argument("--phases", default="1,2,3,4", help="Comma-separated phase numbers")
    parser.add_argument("--cooldown", type=int, default=60, help="Cooldown seconds between phases")
    parser.add_argument("--thermal-limit", type=float, default=45.0, help="Max acceptable temp (°C)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--output-dir", default="results/data", help="Output directory for JSON profiles")
    args = parser.parse_args()

    phases = [int(p) for p in args.phases.split(",")]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()

    print("=" * 60)
    print("llm_rs2 Device Stress Test Suite")
    print(f"  Phases: {phases}")
    print(f"  Timestamp: {timestamp}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 60)

    # Phase 0: Build & Push
    if not phase_build_push(args):
        if not args.dry_run:
            print("[ABORT] Build/push failed or model not found")
            sys.exit(1)

    # Init monitor
    if not args.dry_run:
        monitor = DeviceMonitor(interval=1.0)
    else:
        monitor = None

    results = {}

    # Phase 1: Thermal
    if 1 in phases:
        results["thermal"] = phase_thermal(monitor, args, timestamp)
        if not args.dry_run and 2 in phases:
            wait_for_cooldown(monitor, target_temp=35.0, timeout=args.cooldown)

    # Phase 2: Sustainability
    if 2 in phases:
        results["sustainability"] = phase_sustainability(monitor, args, timestamp)
        if not args.dry_run and 3 in phases:
            wait_for_cooldown(monitor, target_temp=35.0, timeout=args.cooldown)

    # Phase 3: Memory
    if 3 in phases:
        results["memory"] = phase_memory(monitor, args, timestamp)
        if not args.dry_run and 4 in phases:
            wait_for_cooldown(monitor, target_temp=35.0, timeout=args.cooldown)

    # Phase 4: Correctness
    if 4 in phases:
        results["correctness"] = phase_correctness(monitor, args, timestamp)

    # Overall verdict
    verdicts = [r.get("verdict", "N/A") for r in results.values()]
    if "FAIL" in verdicts:
        overall = "FAIL"
    elif "WARN" in verdicts:
        overall = "WARN"
    elif all(v == "PASS" for v in verdicts):
        overall = "PASS"
    else:
        overall = "N/A"

    total_duration = time.time() - start_time

    # Summary JSON
    summary = {
        "version": 1,
        "type": "stress_test_summary",
        "metadata": {
            "date": datetime.now().isoformat(),
            "device": run_adb_command('shell "getprop ro.product.model"', check=False) or "unknown",
            "total_duration_sec": round(total_duration, 1),
            "phases_run": phases,
            "git_commit": get_git_commit(),
        },
        "overall_verdict": overall,
        "phases": results,
    }

    if not args.dry_run:
        fname = f"stress_test_summary_{timestamp}.json"
        save_profile(summary, args.output_dir, fname)

    print_report(summary)
    print(f"\n  Total time: {total_duration / 60:.1f} min")


if __name__ == "__main__":
    main()
