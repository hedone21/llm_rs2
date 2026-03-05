#!/usr/bin/env python3
"""
Device Stress Test Suite for llm_rs2.

Comprehensive 6-phase stress test:
  Phase 1: Thermal Stability   — sustained OpenCL inference, throttling detection
  Phase 2: Performance Sustain — repeated short inferences, consistency measurement
  Phase 3: Memory Stability    — long sequences + KV eviction, RSS leak detection
  Phase 4: Backend Correctness — CPU vs OpenCL numerical accuracy under heat
  Phase 5: Output Quality      — eviction vs no-eviction text quality comparison
  Phase 6: Resilience Signals  — signal injection via Unix socket, adaptive response

Usage:
  python scripts/stress_test_device.py                    # full test
  python scripts/stress_test_device.py --skip-build       # reuse existing binaries
  python scripts/stress_test_device.py --phases 1,4       # specific phases only
  python scripts/stress_test_device.py --phases 3,5       # memory + quality
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

SIGNAL_INJECTOR_BIN_REMOTE = f"{ANDROID_TMP_DIR}/signal_injector"
RESILIENCE_SOCKET = f"{ANDROID_TMP_DIR}/resilience.sock"
SCHEDULES_DIR_REMOTE = f"{ANDROID_TMP_DIR}/schedules"

GENERATE_BIN_LOCAL = "target/aarch64-linux-android/release/generate"
TEST_BACKEND_BIN_LOCAL = "target/aarch64-linux-android/release/test_backend"
SIGNAL_INJECTOR_BIN_LOCAL = "target/aarch64-linux-android/release/signal_injector"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def run_command(cmd, check=True, dry_run=False):
    if dry_run:
        print(f"  [DryRun] {cmd}")
        return
    print(f"  [Exec] {cmd}")
    subprocess.run(cmd, shell=True, check=check)


def run_device_inference(device_cmd, timeout=600, env_vars=""):
    """Run generate on device via adb shell. Returns (exit_code, stdout_text, duration_sec).

    Args:
        env_vars: Extra environment variables to prepend (e.g. "RUST_LOG=info").
    """
    env_prefix = f"{env_vars} " if env_vars else ""
    adb_args = ["adb", "shell", f"{env_prefix}LD_LIBRARY_PATH=/data/local/tmp {device_cmd} 2>&1"]
    start = time.time()
    try:
        result = subprocess.run(
            adb_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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


def extract_generated_text(stdout):
    """Extract generated text from generate binary stdout.

    The binary output format:
      [Profile] Event: ModelLoadStart
      ... header lines ...
      [Profile] Event: DecodingStart
      <generated text streamed here>
      Done.
      [Profile] Event: End
      TTFT: ... ms
    """
    if not stdout:
        return ""
    lines = stdout.split("\n")
    text_start = None
    text_end = None
    for i, line in enumerate(lines):
        if "[Profile] Event: DecodingStart" in line:
            text_start = i + 1
        if line.strip() == "Done." and text_start is not None:
            text_end = i
            break
    if text_start is not None and text_end is not None:
        return "\n".join(lines[text_start:text_end])
    return ""


def count_eviction_events(stdout):
    """Count [CacheManager] eviction events from RUST_LOG=info output."""
    if not stdout:
        return 0
    return sum(1 for line in stdout.splitlines() if "[CacheManager] Evicting" in line)


def count_tokens_generated(stdout):
    """Estimate tokens generated by counting Avg TBT line or Stopped message."""
    if not stdout:
        return 0
    # Check if stopped early
    stopped = "[Stopped: Max context length reached]" in stdout
    # Parse from "Avg TBT: X.XX ms (Y.Y tokens/sec)" — total tokens ~= duration / tbt
    # Simpler: count tokens from TTFT/TBT if available
    results = parse_results(stdout)
    ttft = results.get("ttft_ms", 0)
    tbt = results.get("tbt_ms", 0)
    if tbt > 0 and ttft > 0:
        # Extract total generation text length as a proxy
        text = extract_generated_text(stdout)
        return len(text.split()), stopped
    return 0, stopped


def compute_text_divergence(text_a, text_b):
    """Compare two generated texts and find where they diverge.

    Returns dict with divergence analysis.
    """
    tokens_a = text_a.split()
    tokens_b = text_b.split()

    # Find first diverging token
    min_len = min(len(tokens_a), len(tokens_b))
    diverge_token = min_len
    for i in range(min_len):
        if tokens_a[i] != tokens_b[i]:
            diverge_token = i
            break

    # Find first diverging character
    min_chars = min(len(text_a), len(text_b))
    diverge_char = min_chars
    for i in range(min_chars):
        if text_a[i] != text_b[i]:
            diverge_char = i
            break

    return {
        "common_tokens": diverge_token,
        "total_tokens_a": len(tokens_a),
        "total_tokens_b": len(tokens_b),
        "common_chars": diverge_char,
        "total_chars_a": len(text_a),
        "total_chars_b": len(text_b),
        "common_ratio": round(diverge_token / max(min_len, 1), 4),
    }


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

    phases = [int(p) for p in args.phases.split(",")]

    if not args.skip_build:
        bins = "--bin generate --bin test_backend"
        if 6 in phases:
            bins += " --bin signal_injector"
        print(f"\n  [Build] Building {bins}...")
        run_command(
            f"cargo build --target aarch64-linux-android --release {bins}",
            dry_run=args.dry_run,
        )

    if not args.skip_push:
        print("\n  [Push] Pushing binaries and eval files...")
        run_command(f"adb push {GENERATE_BIN_LOCAL} {GENERATE_BIN_REMOTE}", dry_run=args.dry_run)
        run_command(f"adb push {TEST_BACKEND_BIN_LOCAL} {TEST_BACKEND_BIN_REMOTE}", dry_run=args.dry_run)
        run_command(f"adb shell chmod +x {GENERATE_BIN_REMOTE} {TEST_BACKEND_BIN_REMOTE}", dry_run=args.dry_run)
        run_command(f"adb shell mkdir -p {EVAL_DIR}", dry_run=args.dry_run)
        run_command(f"adb push eval/ {EVAL_DIR}/", check=False, dry_run=args.dry_run)

        if 6 in phases:
            run_command(f"adb push {SIGNAL_INJECTOR_BIN_LOCAL} {SIGNAL_INJECTOR_BIN_REMOTE}", dry_run=args.dry_run)
            run_command(f"adb shell chmod +x {SIGNAL_INJECTOR_BIN_REMOTE}", dry_run=args.dry_run)
            run_command(f"adb shell mkdir -p {SCHEDULES_DIR_REMOTE}", dry_run=args.dry_run)
            run_command(f"adb push scripts/schedules/ {SCHEDULES_DIR_REMOTE}/", check=False, dry_run=args.dry_run)

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
        f'--prompt Hello -n 2048 -b opencl --temperature 0'
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
            f'--prompt Hello -n 128 -b {backend} --temperature 0'
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
        # Capacity comparison: same decode target, eviction enables completion
        {
            "name": "capacity_no_eviction_opencl",
            "backend": "opencl", "prefill_file": f"{EVAL_DIR}/prefill_512.txt",
            "decode": 4096, "eviction": "none", "window": 0,
            "rss_limit_pass": 30, "rss_limit_warn": 60,
            "capacity_test": True,
        },
        {
            "name": "capacity_eviction_opencl",
            "backend": "opencl", "prefill_file": f"{EVAL_DIR}/prefill_512.txt",
            "decode": 4096, "eviction": "sliding", "window": 512,
            "rss_limit_pass": 30, "rss_limit_warn": 60,
            "capacity_test": True,
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

        is_capacity_test = t.get("capacity_test", False)
        env = "RUST_LOG=info" if is_capacity_test else ""
        exit_code, stdout, duration = run_device_inference(device_cmd, timeout=600, env_vars=env)

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

        # Extra fields for capacity tests
        if is_capacity_test:
            td["stopped_early"] = "[Stopped: Max context length reached]" in (stdout or "")
            td["eviction_count"] = count_eviction_events(stdout)
            gen_text = extract_generated_text(stdout)
            td["generated_text_len"] = len(gen_text)
            td["generated_word_count"] = len(gen_text.split())

        # Verdict: capacity tests pass if they complete without crash
        if exit_code != 0:
            td["verdict"] = "FAIL"
        elif is_capacity_test:
            td["verdict"] = "PASS"  # Capacity tests: just need exit 0
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

    # Parse results from the table format:
    # Each row has columns: Op | Shape | DType | Backend1 result | Backend2 result
    # Successful: "0.23ms  | 0.000000"   Failed: "FAIL"   Error: "ERROR"
    pass_count = 0
    fail_count = 0
    error_count = 0
    max_error = 0.0

    for line in stdout.splitlines():
        line_stripped = line.strip()
        # Count FAIL/ERROR columns in table rows (each row can have 2 backend results)
        if "|" in line_stripped and any(
            op in line_stripped for op in ["MatMul", "Softmax", "RMSNorm", "RoPE"]
        ):
            # Count occurrences in this row
            fail_count += line_stripped.count("FAIL")
            error_count += line_stripped.count("ERROR")
            # Count successful results (has "ms  |" pattern with error value)
            successes = re.findall(r'([\d.]+)ms\s+\|\s+([\d.]+)', line_stripped)
            pass_count += len(successes)
            for _, err_str in successes:
                err_val = float(err_str)
                max_error = max(max_error, err_val)

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


# ─── Phase 5: Output Quality ─────────────────────────────────────────────────

def phase_quality(monitor, args, timestamp):
    """Phase 5: Compare output quality with and without eviction."""
    print("\n" + "=" * 60)
    print("Phase 5: OUTPUT QUALITY")
    print("  Eviction vs no-eviction text quality comparison")
    print("=" * 60)

    tests = [
        {
            "name": "baseline_no_eviction",
            "eviction": "none", "window": 0, "prefix": None,
            "decode": 1024,
        },
        {
            "name": "mild_eviction",
            "eviction": "sliding", "window": 512, "prefix": None,
            "decode": 1024,
        },
        {
            "name": "aggressive_eviction",
            "eviction": "sliding", "window": 256, "prefix": 4,
            "decode": 1024,
        },
    ]

    prompt_file = f"{EVAL_DIR}/med_len.txt"
    captured_texts = {}
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
            if t["prefix"] is not None:
                eviction_flags += f" --protected-prefix {t['prefix']}"

        device_cmd = (
            f"{GENERATE_BIN_REMOTE} --model-path {MODEL_PATH} "
            f"--prompt-file {prompt_file} "
            f"-n {t['decode']} -b opencl "
            f"--max-seq-len 2048 --temperature 0"
            f"{eviction_flags}"
        )

        if args.dry_run:
            print(f'  [DryRun] adb shell "{device_cmd}"')
            test_results[t["name"]] = {"dry_run": True}
            continue

        exit_code, stdout, duration = run_device_inference(device_cmd, timeout=300)

        results = parse_results(stdout) if exit_code == 0 else {}
        gen_text = extract_generated_text(stdout)
        captured_texts[t["name"]] = gen_text

        td = {
            "exit_code": exit_code,
            "completed": exit_code == 0,
            "duration_sec": round(duration, 1),
            "ttft_ms": results.get("ttft_ms", 0),
            "tbt_ms": results.get("tbt_ms", 0),
            "generated_text_len": len(gen_text),
            "generated_word_count": len(gen_text.split()),
            "eviction_count": count_eviction_events(stdout),
        }
        test_results[t["name"]] = td

        status = f"words={td['generated_word_count']}, evictions={td['eviction_count']}"
        print(f"  {t['name']}: {status}")

        # Cooldown between quality tests to avoid thermal kills
        time.sleep(30)

    if args.dry_run:
        return {"verdict": "DRY_RUN", "tests": test_results}

    # Compute divergence comparisons
    comparisons = {}
    baseline_text = captured_texts.get("baseline_no_eviction", "")

    for name in ["mild_eviction", "aggressive_eviction"]:
        other_text = captured_texts.get(name, "")
        if baseline_text and other_text:
            div = compute_text_divergence(baseline_text, other_text)
            comparisons[f"baseline_vs_{name}"] = div
            print(f"\n  baseline vs {name}:")
            print(f"    Common tokens: {div['common_tokens']}/{div['total_tokens_a']}"
                  f" ({div['common_ratio']:.1%})")

    # Save texts and comparisons
    quality_data = {
        "version": 1,
        "type": "stress_quality",
        "metadata": {
            "date": datetime.now().isoformat(),
            "prompt_file": "eval/med_len.txt",
            "decode_tokens": 512,
            "temperature": 0,
        },
        "tests": test_results,
        "comparisons": comparisons,
        "texts": {k: v[:2000] for k, v in captured_texts.items()},  # truncate for JSON
    }
    fname = f"stress_quality_{timestamp}.json"
    save_profile(quality_data, args.output_dir, fname)

    # Verdict: all tests should complete
    all_completed = all(
        r.get("completed", False) for r in test_results.values() if not r.get("dry_run")
    )
    overall = "PASS" if all_completed else "FAIL"

    print(f"\n  Phase 5 Result: {overall}")
    return {"verdict": overall, "tests": test_results, "comparisons": comparisons}


# ─── Phase 6: Resilience Signal Injection ────────────────────────────────────

def count_resilience_events(stdout):
    """Count resilience-related events from stderr/stdout."""
    if not stdout:
        return {"evictions": 0, "throttles": 0, "suspends": 0, "restores": 0}
    lines = stdout.splitlines()
    return {
        "evictions": sum(1 for l in lines if "[Resilience] Evicted" in l or "[Hybrid/Resilience] Evicted" in l),
        "throttles": sum(1 for l in lines if "Throttle" in l and "Resilience" in l),
        "suspends": sum(1 for l in lines if "suspended" in l.lower() and "Resilience" in l),
        "restores": sum(1 for l in lines if "RestoreDefaults" in l),
    }


def run_with_signal_injection(schedule_name, generate_flags, timeout, args, monitor, timestamp):
    """Run generate + signal_injector pair and collect results.

    Starts signal_injector in background, then launches generate with
    --enable-resilience --resilience-transport unix:<socket>.
    Returns (test_data_dict, stdout).
    """
    schedule_remote = f"{SCHEDULES_DIR_REMOTE}/{schedule_name}.json"

    # Start signal injector in background
    injector_cmd = (
        f"{SIGNAL_INJECTOR_BIN_REMOTE} "
        f"--socket {RESILIENCE_SOCKET} "
        f"--schedule-file {schedule_remote} "
        f"--connect-timeout 30"
    )
    injector_adb = f'adb shell "LD_LIBRARY_PATH=/data/local/tmp nohup {injector_cmd} > /dev/null 2>&1 &"'

    if args.dry_run:
        print(f"  [DryRun] {injector_adb}")
        print(f"  [DryRun] generate with resilience")
        return {"dry_run": True}, ""

    # Clean up stale socket
    run_adb_command(f'shell "rm -f {RESILIENCE_SOCKET}"', check=False)

    # Start injector
    subprocess.Popen(
        ["adb", "shell",
         f"LD_LIBRARY_PATH=/data/local/tmp {injector_cmd} 2>/data/local/tmp/injector.log"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Give injector time to bind socket
    time.sleep(1)

    # Run generate with resilience
    resilience_flags = f" --enable-resilience --resilience-transport unix:{RESILIENCE_SOCKET}"
    device_cmd = (
        f"{GENERATE_BIN_REMOTE} --model-path {MODEL_PATH} "
        f"{generate_flags}{resilience_flags}"
    )

    if monitor:
        monitor.start()
    exit_code, stdout, duration = run_device_inference(device_cmd, timeout=timeout,
                                                        env_vars="RUST_LOG=info")
    timeseries = monitor.stop() if monitor else []

    # Read injector log
    injector_log = run_adb_command('shell "cat /data/local/tmp/injector.log 2>/dev/null"', check=False) or ""

    results = parse_results(stdout) if exit_code == 0 else {}
    events = count_resilience_events(stdout)

    # Collect RSS data
    rss_values = [s.get("process_mem_mb", 0) for s in timeseries if s.get("process_mem_mb", 0) > 0]
    rss_start = rss_values[0] if rss_values else 0
    rss_peak = max(rss_values) if rss_values else 0
    rss_end = rss_values[-1] if rss_values else 0

    td = {
        "schedule": schedule_name,
        "exit_code": exit_code,
        "completed": exit_code == 0 or events["suspends"] > 0,
        "duration_sec": round(duration, 1),
        "ttft_ms": results.get("ttft_ms", 0),
        "tbt_ms": results.get("tbt_ms", 0),
        "tokens_per_sec": results.get("tokens_per_sec", 0),
        "resilience_events": events,
        "rss_start_mb": round(rss_start, 1),
        "rss_peak_mb": round(rss_peak, 1),
        "rss_end_mb": round(rss_end, 1),
        "rss_delta_mb": round(rss_end - rss_start, 1),
        "injector_log_lines": len(injector_log.strip().splitlines()),
    }

    return td, stdout


def phase_resilience(monitor, args, timestamp):
    """Phase 6: Resilience signal injection tests.

    Tests:
      D3 — Memory eviction under MemoryPressure(Critical)
      D4 — Thermal throttling under ThermalAlert(Critical)
      D5 — Suspend under EnergyConstraint(Emergency)
      D7 — Recovery cycle (Critical → Normal → Critical → Normal)
    """
    print("\n" + "=" * 60)
    print("Phase 6: RESILIENCE SIGNAL INJECTION")
    print("  Test adaptive inference under system signal pressure")
    print("=" * 60)

    scenarios = [
        {
            "name": "memory_eviction",
            "schedule": "memory_eviction",
            "desc": "MemoryPressure(Critical) → evict KV cache",
            "generate_flags": (
                f"--prompt-file {EVAL_DIR}/prefill_512.txt "
                "-n 512 -b opencl --max-seq-len 2048 "
                "--eviction-policy h2o --h2o-recent-window 128 --h2o-keep-ratio 0.5"
            ),
            "timeout": 180,
            "expect_evictions": True,
            "expect_suspend": False,
        },
        {
            "name": "thermal_throttle",
            "schedule": "thermal_throttle",
            "desc": "ThermalAlert(Critical) → throttle token generation",
            "generate_flags": (
                f"--prompt-file {EVAL_DIR}/prefill_512.txt "
                "-n 256 -b opencl --max-seq-len 2048"
            ),
            "timeout": 180,
            "expect_evictions": False,
            "expect_suspend": False,
        },
        {
            "name": "energy_suspend",
            "schedule": "energy_suspend",
            "desc": "EnergyConstraint(Emergency) → suspend inference",
            "generate_flags": (
                f"--prompt-file {EVAL_DIR}/prefill_512.txt "
                "-n 1024 -b opencl --max-seq-len 2048"
            ),
            "timeout": 120,
            "expect_evictions": False,
            "expect_suspend": True,
        },
        {
            "name": "recovery_cycle",
            "schedule": "recovery_cycle",
            "desc": "Critical → Normal → Critical cycles, verify recovery",
            "generate_flags": (
                f"--prompt-file {EVAL_DIR}/prefill_512.txt "
                "-n 1024 -b opencl --max-seq-len 2048 "
                "--eviction-policy h2o --h2o-recent-window 128 --h2o-keep-ratio 0.5"
            ),
            "timeout": 300,
            "expect_evictions": True,
            "expect_suspend": False,
        },
    ]

    test_results = {}
    verdicts = []

    for scenario in scenarios:
        print(f"\n  --- {scenario['name']}: {scenario['desc']} ---")

        td, stdout = run_with_signal_injection(
            schedule_name=scenario["schedule"],
            generate_flags=scenario["generate_flags"],
            timeout=scenario["timeout"],
            args=args,
            monitor=monitor,
            timestamp=timestamp,
        )

        if td.get("dry_run"):
            test_results[scenario["name"]] = td
            continue

        # Validate expectations
        events = td.get("resilience_events", {})
        issues = []

        if scenario["expect_evictions"] and events.get("evictions", 0) == 0:
            issues.append("Expected eviction events but found none")
        if scenario["expect_suspend"] and events.get("suspends", 0) == 0:
            issues.append("Expected suspend event but found none")
        if not td["completed"]:
            issues.append(f"Process exited with code {td['exit_code']}")

        td["issues"] = issues
        td["verdict"] = "PASS" if not issues else "FAIL"
        test_results[scenario["name"]] = td
        verdicts.append(td["verdict"])

        status = (
            f"evict={events.get('evictions', 0)} throttle={events.get('throttles', 0)} "
            f"suspend={events.get('suspends', 0)} → {td['verdict']}"
        )
        print(f"  Result: {status}")
        if issues:
            for issue in issues:
                print(f"    [!] {issue}")

        # Save individual test data
        test_data = {
            "version": 1,
            "type": "stress_resilience",
            "metadata": {
                "date": datetime.now().isoformat(),
                "scenario": scenario["name"],
                "schedule": scenario["schedule"],
                "description": scenario["desc"],
            },
            "result": td,
        }
        fname = f"stress_resilience_{scenario['name']}_{timestamp}.json"
        save_profile(test_data, args.output_dir, fname)

        # Cooldown between resilience tests
        if monitor:
            time.sleep(20)

    if args.dry_run:
        return {"verdict": "DRY_RUN", "tests": test_results}

    # Overall verdict
    if "FAIL" in verdicts:
        overall = "FAIL"
    elif all(v == "PASS" for v in verdicts):
        overall = "PASS"
    else:
        overall = "WARN"

    print(f"\n  Phase 6 Result: {overall}")
    return {"verdict": overall, "tests": test_results}


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
        elif phase_name == "quality" and "comparisons" in phase_data:
            for cmp_name, cmp in phase_data["comparisons"].items():
                print(f"       {cmp_name}: {cmp.get('common_tokens', '?')}"
                      f"/{cmp.get('total_tokens_a', '?')} common"
                      f" ({cmp.get('common_ratio', 0):.1%})")
        elif phase_name == "resilience" and "tests" in phase_data:
            for tn, td in phase_data["tests"].items():
                if isinstance(td, dict) and "resilience_events" in td:
                    ev = td["resilience_events"]
                    print(f"       {tn}: evict={ev.get('evictions', 0)} "
                          f"throttle={ev.get('throttles', 0)} "
                          f"suspend={ev.get('suspends', 0)} → {td.get('verdict', '?')}")

    overall = summary.get("overall_verdict", "N/A")
    print(f"\n  OVERALL: {overall}")
    print("=" * 60)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Device Stress Test Suite for llm_rs2")
    parser.add_argument("--skip-build", action="store_true", help="Skip cargo build step")
    parser.add_argument("--skip-push", action="store_true", help="Skip adb push step")
    parser.add_argument("--phases", default="1,2,3,4,5,6", help="Comma-separated phase numbers")
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
        if not args.dry_run and 5 in phases:
            wait_for_cooldown(monitor, target_temp=35.0, timeout=args.cooldown)

    # Phase 5: Output Quality
    if 5 in phases:
        results["quality"] = phase_quality(monitor, args, timestamp)
        if not args.dry_run and 6 in phases:
            wait_for_cooldown(monitor, target_temp=35.0, timeout=args.cooldown)

    # Phase 6: Resilience
    if 6 in phases:
        results["resilience"] = phase_resilience(monitor, args, timestamp)

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
