#!/usr/bin/env python3
"""Resilience Mode A vs C experiment runner.

Runs baseline (Mode A) and resilience (Mode C) experiments on the target device,
then generates a comparison report using experiments/analysis/compare.py.

Usage:
    # Host (quick test)
    python scripts/run_resilience_experiment.py -d host --tokens 512

    # On-device (full experiment)
    python scripts/run_resilience_experiment.py -d pixel --tokens 1024

    # Custom configs
    python scripts/run_resilience_experiment.py -d pixel \
        --mode-a experiments/configs/resil_baseline.json \
        --mode-c experiments/configs/resil_mode_c_repeated.json

    # Skip build (reuse existing binary)
    python scripts/run_resilience_experiment.py -d pixel --skip-build --tokens 512
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add device_registry to path
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from device_registry.config import load_all_devices, DeviceConfig
from device_registry.connection import create_connection, Connection
from device_registry.builder import build_binary
from device_registry.deployer import deploy_binary, deploy_eval_files


DEFAULT_MODE_A = "experiments/configs/resil_baseline.json"
DEFAULT_MODE_C = "experiments/configs/resil_mode_c_repeated.json"
DEFAULT_PROMPT = "experiments/prompts/med_len.txt"


def run_experiment(
    conn: Connection,
    device: DeviceConfig,
    schedule_path: str,
    output_path: str,
    prompt_file: str,
    num_tokens: int,
    extra_args: list[str],
    dry_run: bool = False,
) -> dict | None:
    """Run a single experiment on the device and return the summary record."""
    binary = f"{device.paths.work_dir}/generate" if not device.is_local else \
             str(PROJECT_ROOT / device.build.binary_dir / "generate")

    model_path = device.paths.model_dir

    # Build command
    cmd_parts = [
        binary,
        "--model-path", model_path,
        "--prompt-file", prompt_file,
        "-n", str(num_tokens),
        "--ignore-eos",
        "--prefill-chunk-size", "512",
        "--experiment-schedule", schedule_path,
        "--experiment-output", output_path,
        "--experiment-sample-interval", "10",
        "-b", "cpu",
        "--temperature", "0",
    ] + extra_args

    cmd_str = " ".join(cmd_parts)

    if dry_run:
        print(f"  [dry-run] {cmd_str}")
        return None

    print(f"  Running: {Path(schedule_path).stem} → {output_path}")
    start = time.time()
    result = conn.execute(cmd_str, timeout=600)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  ERROR (exit={result.returncode}, {elapsed:.1f}s)")
        if result.stderr:
            print(f"  stderr: {result.stderr[-500:]}")
        return None

    print(f"  Completed in {elapsed:.1f}s")

    # Parse summary from JSONL (last line with _summary=true)
    # For remote devices, pull to temp file first
    if device.is_local:
        jsonl_path = output_path
    else:
        jsonl_path = f"/tmp/_resil_exp_{Path(output_path).name}"
        try:
            conn.pull(output_path, jsonl_path)
        except Exception as e:
            print(f"  Warning: could not pull {output_path}: {e}")
            return None

    try:
        summary = None
        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line)
                if record.get("_summary"):
                    summary = record
        return summary
    except Exception as e:
        print(f"  Warning: could not parse summary: {e}")
        return None


def deploy_configs(conn: Connection, device: DeviceConfig, config_paths: list[str]):
    """Deploy experiment config files to device."""
    if device.is_local:
        return  # configs are already accessible

    for path in config_paths:
        local = PROJECT_ROOT / path
        remote = f"{device.paths.work_dir}/{Path(path).name}"
        conn.push(str(local), remote)
        print(f"  Deployed config: {Path(path).name}")


def main():
    parser = argparse.ArgumentParser(description="Resilience Mode A vs C experiment")
    parser.add_argument("-d", "--device", default="host", help="Device ID from devices.toml")
    parser.add_argument("--mode-a", default=DEFAULT_MODE_A, help="Mode A (baseline) schedule JSON")
    parser.add_argument("--mode-c", default=DEFAULT_MODE_C, help="Mode C (resilience) schedule JSON")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt file path")
    parser.add_argument("--tokens", type=int, default=512, help="Number of decode tokens")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per mode")
    parser.add_argument("--eviction-policy", default="h2o", help="Eviction policy for Mode C")
    parser.add_argument("--kv-budget", type=int, default=2048, help="KV budget for eviction")
    parser.add_argument("--output-dir", default=None, help="Results output directory")
    parser.add_argument("--skip-build", action="store_true", help="Skip cargo build")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip binary/config deployment")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    # Load device config
    devices = load_all_devices(PROJECT_ROOT / "devices.toml")
    if args.device not in devices:
        print(f"Error: device '{args.device}' not found. Available: {list(devices.keys())}")
        return 1

    device = devices[args.device]
    conn = create_connection(device.connection)

    # Output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else \
                 PROJECT_ROOT / "experiments" / "results" / f"resil_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {output_dir}")

    # Build
    if not args.skip_build and not args.dry_run:
        print("\n[1/4] Building...")
        if not build_binary(device, "generate", PROJECT_ROOT):
            print("Build failed!")
            return 1

    # Deploy
    if not args.skip_deploy and not device.is_local and not args.dry_run:
        print("\n[2/4] Deploying...")
        local_bin = PROJECT_ROOT / device.build.binary_dir / "generate"
        deploy_binary(conn, device, local_bin, "generate")
        deploy_eval_files(conn, device, PROJECT_ROOT / "experiments" / "prompts")
        deploy_configs(conn, device, [args.mode_a, args.mode_c])

    # Determine paths (local vs remote)
    if device.is_local:
        prompt_file = str(PROJECT_ROOT / args.prompt)
        schedule_a = str(PROJECT_ROOT / args.mode_a)
        schedule_c = str(PROJECT_ROOT / args.mode_c)
        out_prefix = str(output_dir)
    else:
        prompt_file = f"{device.paths.eval_dir}/{Path(args.prompt).name}"
        schedule_a = f"{device.paths.work_dir}/{Path(args.mode_a).name}"
        schedule_c = f"{device.paths.work_dir}/{Path(args.mode_c).name}"
        out_prefix = device.paths.work_dir

    # Extra args for Mode C (eviction enabled)
    mode_c_extra = [
        "--eviction-policy", args.eviction_policy,
        "--kv-budget", str(args.kv_budget),
    ]

    # Run experiments
    print(f"\n[3/4] Running experiments (×{args.runs} each)...")
    summaries = {"mode_a": [], "mode_c": []}

    for run_idx in range(args.runs):
        tag = f"_r{run_idx}" if args.runs > 1 else ""

        for mode, schedule, extra, label in [
            ("mode_a", schedule_a, [], "A"),
            ("mode_c", schedule_c, mode_c_extra, "C"),
        ]:
            remote_out = f"{out_prefix}/{mode}{tag}.jsonl"
            local_out = str(output_dir / f"{mode}{tag}.jsonl")
            print(f"\n  --- Mode {label} (run {run_idx+1}/{args.runs}) ---")
            summary = run_experiment(
                conn, device, schedule, remote_out, prompt_file,
                args.tokens, extra, args.dry_run,
            )
            # Pull from device to local results dir
            if not args.dry_run and not device.is_local:
                try:
                    conn.pull(remote_out, local_out)
                except Exception as e:
                    print(f"  Warning: pull failed: {e}")
            if summary:
                summaries[mode].append(summary)

    if args.dry_run:
        return 0

    # Analysis — compare first run of each mode
    print(f"\n[4/4] Generating comparison report...")
    first_tag = "_r0" if args.runs > 1 else ""
    local_a = output_dir / f"mode_a{first_tag}.jsonl"
    local_c = output_dir / f"mode_c{first_tag}.jsonl"
    report_path = output_dir / "comparison.md"

    if local_a.exists() and local_c.exists():
        subprocess.run([
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "analysis" / "compare.py"),
            "--baseline", str(local_a),
            "--experiment", str(local_c),
            "--output", str(report_path),
        ])
        print(f"\nReport: {report_path}")
    else:
        missing = []
        if not local_a.exists():
            missing.append(str(local_a))
        if not local_c.exists():
            missing.append(str(local_c))
        print(f"  Skipping comparison — missing: {', '.join(missing)}")

    # Print summary table
    if summaries["mode_a"] and summaries["mode_c"]:
        print("\n" + "=" * 60)
        print("  SUMMARY: Mode A vs Mode C")
        print("=" * 60)
        for mode, label in [("mode_a", "A"), ("mode_c", "C")]:
            for i, s in enumerate(summaries[mode]):
                evictions = s.get("eviction_count", 0)
                evicted = s.get("evicted_tokens_total", 0)
                tbt = s.get("avg_tbt_ms", 0)
                cache = s.get("final_cache_pos", 0)
                print(f"  Mode {label} run {i}: "
                      f"TBT={tbt:.1f}ms, "
                      f"evictions={evictions} ({evicted} tokens), "
                      f"cache_pos={cache}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
