#!/usr/bin/env python3
"""mock_manager 전수 검증 — 53 test cases on Android device.

Usage:
    # Build + push + run all
    python scripts/test_mock_commands.py

    # Skip build (binaries already on device)
    python scripts/test_mock_commands.py --skip-build

    # Run specific phase only
    python scripts/test_mock_commands.py --phase 1

    # Run specific test
    python scripts/test_mock_commands.py --test 3-07
"""
import subprocess
import time
import json
import sys
import os
import re
import tempfile
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

# ─── Constants ──────────────────────────────────────────────────────────────
BIN = "/data/local/tmp"
MODEL = f"{BIN}/models/llama3.2-1b"
TCP_ADDR = "127.0.0.1:19999"
WAIT_SECS = 3
GEN_N = 500  # enough tokens for longest scenario (~20s)

# ─── ADB helpers ────────────────────────────────────────────────────────────
def adb_shell(cmd: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(["adb", "shell", cmd],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return ""

def adb_push(local: str, remote: str):
    subprocess.run(["adb", "push", local, remote],
                   capture_output=True, check=True, timeout=30)

def kill_procs():
    adb_shell("pkill -9 -f mock_manager; pkill -9 -f 'generate.*resilience'", timeout=5)
    time.sleep(0.5)

def push_scenario(scenario: dict) -> str:
    device_path = f"{BIN}/test_scenario.json"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(scenario, f)
        tmp = f.name
    adb_push(tmp, device_path)
    os.unlink(tmp)
    return device_path

# ─── Engine configs ─────────────────────────────────────────────────────────
def _base(extra: str = "") -> str:
    b = (f"--model-path {MODEL} --prompt Hello -n {GEN_N} --greedy --ignore-eos "
         f"--enable-resilience --resilience-transport tcp:{TCP_ADDR}")
    return f"{b} {extra}".strip()

CONFIGS = {
    "C1": _base(),                                          # CPU, resilience only
    "C2": _base("-b opencl --resilience-prealloc-switch"),   # GPU + prealloc (for SwitchHw)
    "C3": _base("-b opencl --resilience-prealloc-switch "
                "--tensor-partition 0.001"),                  # GPU + partition + prealloc
    "C4": _base("--kv-dynamic-quant"),                       # KIVI path
}

# ─── Test case definition ───────────────────────────────────────────────────
@dataclass
class Test:
    id: str
    desc: str
    config: str
    mm_args: str = ""
    scenario: Optional[dict] = None
    expect: List[str] = field(default_factory=lambda: [r"Response.*\[Ok\]"])
    allow_broken_pipe: bool = False  # Suspend causes engine exit → broken pipe OK

def scenario(name: str, commands: list) -> dict:
    return {"name": name, "commands": commands}

def sc(delay_ms: int, command: str, **kw) -> dict:
    d = {"delay_ms": delay_ms, "command": command}
    d.update(kw)
    return d

# ─── Test execution ─────────────────────────────────────────────────────────
def calc_timeout(t: Test) -> int:
    base = WAIT_SECS + 25  # startup + wait + post-observe + buffer
    if t.scenario:
        base += sum(c["delay_ms"] for c in t.scenario["commands"]) / 1000
    return int(base)

def run_test(t: Test) -> Tuple[bool, str]:
    kill_procs()
    env = f"LD_LIBRARY_PATH={BIN}"
    timeout = calc_timeout(t)

    # Build mock_manager command
    if t.scenario:
        sc_path = push_scenario(t.scenario)
        mm_cmd = (f"{env} {BIN}/mock_manager --tcp {TCP_ADDR} "
                  f"--scenario {sc_path} --wait-secs {WAIT_SECS}")
    else:
        mm_cmd = (f"{env} {BIN}/mock_manager --tcp {TCP_ADDR} "
                  f"{t.mm_args} --wait-secs {WAIT_SECS}")

    gen_cmd = f"{env} {BIN}/generate {CONFIGS[t.config]}"

    # Start mock_manager (server, waits for engine connection)
    mm_proc = subprocess.Popen(
        ["adb", "shell", f"{mm_cmd} 2>&1"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(0.8)

    # Start engine (client, connects to mock_manager)
    gen_proc = subprocess.Popen(
        ["adb", "shell", f"{gen_cmd} 2>&1"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Wait for mock_manager to complete
    try:
        mm_out, _ = mm_proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        mm_proc.kill()
        kill_procs()
        return False, "TIMEOUT: mock_manager did not exit"

    # Give engine a moment, then kill it
    time.sleep(0.5)
    adb_shell("pkill -9 -f 'generate.*resilience'", timeout=5)
    try:
        gen_out, _ = gen_proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        gen_proc.kill()
        gen_out = ""

    # ── Verify ──
    # Check engine didn't crash
    if "panic" in gen_out.lower() or "SIGABRT" in gen_out:
        return False, f"ENGINE PANIC: {gen_out[-300:]}"

    # Check mock_manager received any response at all
    if "Timed out waiting for message" in mm_out:
        return False, f"Engine never connected: {mm_out[-200:]}"

    # Broken pipe after Suspend is expected (engine exits after suspend)
    if "Broken pipe" in mm_out and not t.allow_broken_pipe:
        return False, f"Broken pipe (unexpected): {mm_out[-200:]}"

    # Check expected patterns
    for pattern in t.expect:
        if not re.search(pattern, mm_out):
            return False, f"Pattern not found: {pattern}"

    return True, "OK"

# ═══════════════════════════════════════════════════════════════════════════
#  TEST DEFINITIONS (53 cases)
# ═══════════════════════════════════════════════════════════════════════════

TESTS: List[Test] = []

# ── Phase 1: Individual commands (13 cases) ─────────────────────────────
TESTS += [
    Test("1-01", "KvEvictSliding(0.7)", "C1",
         mm_args="--command KvEvictSliding --keep-ratio 0.7"),
    Test("1-02", "KvEvictH2o(0.7)", "C1",
         mm_args="--command KvEvictH2o --keep-ratio 0.7"),
    Test("1-03", "KvStreaming(4,500)", "C1",
         mm_args="--command KvStreaming --sink-size 4 --window-size 500"),
    Test("1-04", "KvQuantDynamic(4bit)", "C4",
         mm_args="--command KvQuantDynamic --target-bits 4"),
    Test("1-05", "Throttle(100ms)", "C1",
         mm_args="--command Throttle --delay-ms 100"),
    Test("1-06", "SwitchHw(cpu)", "C2",
         mm_args="--command SwitchHw --device cpu"),
    Test("1-07", "SetPartitionRatio(0.5)", "C3",
         mm_args="--command SetPartitionRatio --ratio 0.5"),
    Test("1-08", "SetPrefillPolicy(48,10,16)", "C3",
         mm_args="--command SetPrefillPolicy --chunk-size 48 --yield-ms 10 --cpu-chunk-size 16"),
    Test("1-09", "Suspend", "C1",
         mm_args="--command Suspend"),
    Test("1-10", "Suspend→Resume", "C1",
         scenario=scenario("suspend-resume", [
             sc(3000, "Suspend"), sc(3000, "Resume")]),
         allow_broken_pipe=True),  # engine exits after Suspend
    Test("1-11", "RestoreDefaults", "C1",
         mm_args="--command RestoreDefaults"),
    Test("1-12", "SetTargetTbt(150ms)", "C1",
         mm_args="--command SetTargetTbt --target-ms 150"),
    Test("1-13", "RequestQcf", "C1",
         mm_args="--command RequestQcf",
         expect=[r"Response.*\[Ok\]", r"QcfEstimate"]),
]

# ── Phase 2: Intra-domain combinations (8 cases) ───────────────────────
TESTS += [
    Test("2-01", "KvEvictSliding→KvEvictH2o", "C1",
         scenario=scenario("evict-replace", [
             sc(3000, "KvEvictSliding", keep_ratio=0.7),
             sc(3000, "KvEvictH2o", keep_ratio=0.7)])),
    Test("2-02", "KvEvictH2o→KvStreaming", "C1",
         scenario=scenario("evict-to-streaming", [
             sc(3000, "KvEvictH2o", keep_ratio=0.7),
             sc(3000, "KvStreaming", sink_size=4, window_size=500)])),
    Test("2-03", "KvEvictH2o→KvQuantDynamic(4)", "C4",
         scenario=scenario("evict-quant", [
             sc(3000, "KvEvictH2o", keep_ratio=0.7),
             sc(3000, "KvQuantDynamic", target_bits=4)])),
    Test("2-04", "KvQuant 4→8→16", "C4",
         scenario=scenario("quant-cascade", [
             sc(3000, "KvQuantDynamic", target_bits=4),
             sc(3000, "KvQuantDynamic", target_bits=8),
             sc(3000, "KvQuantDynamic", target_bits=16)])),
    Test("2-05", "Suspend→Resume", "C1",
         scenario=scenario("lifecycle", [
             sc(3000, "Suspend"), sc(3000, "Resume")]),
         allow_broken_pipe=True),
    Test("2-06", "Suspend→Suspend (dup)", "C1",
         scenario=scenario("double-suspend", [
             sc(3000, "Suspend"), sc(2000, "Suspend")]),
         allow_broken_pipe=True),
    Test("2-07", "Resume→Resume (dup)", "C1",
         scenario=scenario("double-resume", [
             sc(3000, "Resume"), sc(2000, "Resume")])),
    Test("2-08", "Throttle→RestoreDefaults", "C1",
         scenario=scenario("throttle-restore", [
             sc(3000, "Throttle", delay_ms_param=100),
             sc(3000, "RestoreDefaults")])),
]

# ── Phase 3: Cross-domain combinations (20 cases) ─���────────────────────

# 3A: SwitchHw combinations
TESTS += [
    Test("3-01", "SwitchHw(cpu)→Throttle", "C2",
         scenario=scenario("switch-throttle", [
             sc(3000, "SwitchHw", device="cpu"),
             sc(3000, "Throttle", delay_ms_param=50)])),
    Test("3-02", "SwitchHw(cpu)→KvEvictSliding", "C2",
         scenario=scenario("switch-evict", [
             sc(3000, "SwitchHw", device="cpu"),
             sc(3000, "KvEvictSliding", keep_ratio=0.7)])),
    Test("3-03", "SwitchHw(cpu)→SetPartitionRatio", "C3",
         scenario=scenario("switch-partition", [
             sc(3000, "SwitchHw", device="cpu"),
             sc(3000, "SetPartitionRatio", ratio=0.5)])),
    Test("3-04", "SwitchHw(cpu→opencl) roundtrip", "C2",
         scenario=scenario("switch-roundtrip", [
             sc(3000, "SwitchHw", device="cpu"),
             sc(5000, "SwitchHw", device="opencl")])),
    Test("3-05", "SwitchHw(opencl)→SetPartitionRatio", "C3",
         scenario=scenario("gpu-then-partition", [
             sc(3000, "SwitchHw", device="opencl"),
             sc(3000, "SetPartitionRatio", ratio=0.5)])),
    Test("3-06", "SwitchHw(cpu)→Suspend→Resume→SwitchHw(opencl)", "C2",
         scenario=scenario("switch-lifecycle", [
             sc(3000, "SwitchHw", device="cpu"),
             sc(3000, "Suspend"),
             sc(3000, "Resume"),
             sc(3000, "SwitchHw", device="opencl")]),
         allow_broken_pipe=True),
]

# 3B: SetPartitionRatio combinations
TESTS += [
    Test("3-07", "SetPartitionRatio→Throttle", "C3",
         scenario=scenario("partition-throttle", [
             sc(3000, "SetPartitionRatio", ratio=0.5),
             sc(3000, "Throttle", delay_ms_param=50)])),
    Test("3-08", "SetPartitionRatio→KvEvictSliding", "C3",
         scenario=scenario("partition-evict", [
             sc(3000, "SetPartitionRatio", ratio=0.5),
             sc(3000, "KvEvictSliding", keep_ratio=0.7)])),
    Test("3-09", "SetPartitionRatio 0.5→0.0 toggle", "C3",
         scenario=scenario("partition-toggle", [
             sc(3000, "SetPartitionRatio", ratio=0.5),
             sc(3000, "SetPartitionRatio", ratio=0.0)])),
    Test("3-10", "SetPartitionRatio→RestoreDefaults", "C3",
         scenario=scenario("partition-restore", [
             sc(3000, "SetPartitionRatio", ratio=0.5),
             sc(3000, "RestoreDefaults")])),
]

# 3C: SetPrefillPolicy combinations
TESTS += [
    Test("3-11", "SetPrefillPolicy→SetPartitionRatio", "C3",
         scenario=scenario("prefill-partition", [
             sc(3000, "SetPrefillPolicy", chunk_size=48, yield_ms=10, cpu_chunk_size=16),
             sc(3000, "SetPartitionRatio", ratio=0.5)])),
    Test("3-12", "SetPrefillPolicy→SwitchHw(cpu)", "C3",
         scenario=scenario("prefill-switch", [
             sc(3000, "SetPrefillPolicy", chunk_size=48, yield_ms=10),
             sc(3000, "SwitchHw", device="cpu")])),
    Test("3-13", "SetPrefillPolicy→RestoreDefaults", "C3",
         scenario=scenario("prefill-restore", [
             sc(3000, "SetPrefillPolicy", chunk_size=48, yield_ms=10, cpu_chunk_size=16),
             sc(3000, "RestoreDefaults")])),
]

# 3D: Suspend/Resume state persistence
TESTS += [
    Test("3-14", "Throttle→Suspend→Resume", "C1",
         scenario=scenario("throttle-lifecycle", [
             sc(3000, "Throttle", delay_ms_param=100),
             sc(3000, "Suspend"),
             sc(3000, "Resume")]),
         allow_broken_pipe=True),
    Test("3-15", "KvEvictH2o→Suspend→Resume", "C1",
         scenario=scenario("evict-lifecycle", [
             sc(3000, "KvEvictH2o", keep_ratio=0.7),
             sc(3000, "Suspend"),
             sc(3000, "Resume")]),
         allow_broken_pipe=True),
    Test("3-16", "SetPartitionRatio→Suspend→Resume", "C3",
         scenario=scenario("partition-lifecycle", [
             sc(3000, "SetPartitionRatio", ratio=0.5),
             sc(3000, "Suspend"),
             sc(3000, "Resume")]),
         allow_broken_pipe=True),
]

# 3E: Multi-domain stacking
TESTS += [
    Test("3-17", "Throttle+KvEvictH2o", "C1",
         scenario=scenario("multi-2", [
             sc(3000, "Throttle", delay_ms_param=50),
             sc(2000, "KvEvictH2o", keep_ratio=0.7)])),
    Test("3-18", "SetPartitionRatio+SetPrefillPolicy+Throttle", "C3",
         scenario=scenario("multi-3", [
             sc(3000, "SetPartitionRatio", ratio=0.5),
             sc(2000, "SetPrefillPolicy", chunk_size=48, yield_ms=10),
             sc(2000, "Throttle", delay_ms_param=30)])),
    Test("3-19", "SwitchHw(cpu)+Throttle+KvEvictSliding", "C2",
         scenario=scenario("multi-switch-3", [
             sc(3000, "SwitchHw", device="cpu"),
             sc(3000, "Throttle", delay_ms_param=100),
             sc(2000, "KvEvictSliding", keep_ratio=0.7)])),
    Test("3-20", "3-19 → RestoreDefaults", "C2",
         scenario=scenario("multi-switch-restore", [
             sc(3000, "SwitchHw", device="cpu"),
             sc(3000, "Throttle", delay_ms_param=100),
             sc(2000, "KvEvictSliding", keep_ratio=0.7),
             sc(3000, "RestoreDefaults")])),
]

# ── Phase 4: Edge cases (8 cases) ──────────────────────────────────────
TESTS += [
    Test("4-01", "SetPartitionRatio w/o tensor-partition flag", "C1",
         mm_args="--command SetPartitionRatio --ratio 0.5"),
    Test("4-02", "SwitchHw w/o prealloc-switch flag", "C1",
         mm_args="--command SwitchHw --device cpu"),
    Test("4-03", "KvQuantDynamic w/o kv-dynamic-quant flag", "C1",
         mm_args="--command KvQuantDynamic --target-bits 4"),
    Test("4-04", "SetPrefillPolicy cpu_chunk w/o zero-copy", "C1",
         mm_args="--command SetPrefillPolicy --cpu-chunk-size 16"),
    Test("4-05", "SetPartitionRatio out-of-range (2.0)", "C3",
         mm_args="--command SetPartitionRatio --ratio 2.0"),
    Test("4-06", "KvEvictSliding extreme (keep_ratio=0.0)", "C1",
         mm_args="--command KvEvictSliding --keep-ratio 0.0"),
    Test("4-07", "Throttle(0) noop", "C1",
         mm_args="--command Throttle --delay-ms 0"),
    Test("4-08", "Rapid-fire 10 commands", "C1",
         scenario=scenario("rapid-fire", [
             sc(500, "Throttle", delay_ms_param=10),
             sc(500, "KvEvictSliding", keep_ratio=0.9),
             sc(500, "RestoreDefaults"),
             sc(500, "Throttle", delay_ms_param=20),
             sc(500, "KvEvictH2o", keep_ratio=0.8),
             sc(500, "RestoreDefaults"),
             sc(500, "Throttle", delay_ms_param=30),
             sc(500, "SetTargetTbt", delay_ms_param=150),
             sc(500, "RestoreDefaults"),
             sc(500, "RequestQcf")])),
]

# ── Phase 5: Scenario JSON replay (4 cases) ────────────────────────────
TESTS += [
    Test("5-01", "Scenario: lifecycle", "C1",
         scenario=scenario("sc-lifecycle", [
             sc(3000, "Throttle", delay_ms_param=100),
             sc(3000, "Suspend"),
             sc(3000, "Resume"),
             sc(3000, "RestoreDefaults")]),
         allow_broken_pipe=True),
    Test("5-02", "Scenario: KV cascade", "C1",
         scenario=scenario("sc-kv-cascade", [
             sc(3000, "KvEvictSliding", keep_ratio=0.7),
             sc(3000, "KvEvictH2o", keep_ratio=0.6),
             sc(3000, "KvStreaming", sink_size=4, window_size=500),
             sc(3000, "RestoreDefaults")])),
    Test("5-03", "Scenario: resource optimization", "C3",
         scenario=scenario("sc-resource-opt", [
             sc(3000, "SetPartitionRatio", ratio=0.5),
             sc(3000, "SetPrefillPolicy", chunk_size=48, yield_ms=10, cpu_chunk_size=16),
             sc(3000, "RestoreDefaults")])),
    Test("5-04", "Scenario: full cycle", "C2",
         scenario=scenario("sc-full-cycle", [
             sc(3000, "SwitchHw", device="cpu"),
             sc(3000, "Throttle", delay_ms_param=50),
             sc(3000, "KvEvictSliding", keep_ratio=0.7),
             sc(3000, "SwitchHw", device="opencl"),
             sc(3000, "RestoreDefaults")])),
]

# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

PHASE_LABELS = {
    1: "단일 Command",
    2: "도메인 내 조합",
    3: "크로스 도메인 조합",
    4: "엣지 케이스",
    5: "시나리오 재생",
}

def build_and_push():
    """Build both binaries for Android and push to device."""
    print("[Build] Building generate + mock_manager for Android...")
    r = subprocess.run(
        "source android.source && "
        "cargo build --release --target aarch64-linux-android "
        "--bin generate --bin mock_manager",
        shell=True, capture_output=True, text=True, timeout=300,
        executable="/bin/zsh")
    if r.returncode != 0:
        print(f"[Build] FAILED:\n{r.stderr[-500:]}")
        sys.exit(1)
    print("[Build] OK")

    target = "target/aarch64-linux-android/release"
    for binary in ["generate", "mock_manager"]:
        print(f"[Deploy] Pushing {binary}...")
        subprocess.run(["adb", "push", f"{target}/{binary}", f"{BIN}/{binary}"],
                       check=True, capture_output=True, timeout=60)
        adb_shell(f"chmod +x {BIN}/{binary}")
    print("[Deploy] OK")

def main():
    parser = argparse.ArgumentParser(description="mock_manager full verification")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--phase", type=int, help="Run specific phase only")
    parser.add_argument("--test", type=str, help="Run specific test ID (e.g. 3-07)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Filter tests
    tests = TESTS
    if args.phase:
        tests = [t for t in tests if int(t.id.split("-")[0]) == args.phase]
    if args.test:
        tests = [t for t in tests if t.id == args.test]

    if not tests:
        print("No tests match filter")
        sys.exit(1)

    print(f"=== mock_manager 전수 검증 ({len(tests)} cases) ===\n")

    # Check device
    devs = subprocess.run(["adb", "devices"], capture_output=True, text=True).stdout
    if devs.strip().count("\n") < 1 or "device" not in devs:
        print("ERROR: No device connected")
        sys.exit(1)

    if not args.skip_build:
        build_and_push()
    else:
        print("[Build] Skipped\n")

    # Run tests
    results: List[Tuple[str, str, bool, str]] = []
    current_phase = 0
    start_time = time.time()

    for t in tests:
        phase = int(t.id.split("-")[0])
        if phase != current_phase:
            current_phase = phase
            print(f"\n--- Phase {phase}: {PHASE_LABELS.get(phase, '')} ---")

        label = f"  [{t.id}] {t.desc:<45s} "
        sys.stdout.write(label)
        sys.stdout.flush()

        t_start = time.time()
        try:
            passed, detail = run_test(t)
        except Exception as e:
            passed, detail = False, f"EXCEPTION: {e}"
        elapsed = time.time() - t_start

        results.append((t.id, t.desc, passed, detail))
        status = f"PASS ({elapsed:.1f}s)" if passed else f"FAIL ({elapsed:.1f}s)"
        print(status)

        if not passed and args.verbose:
            print(f"         {detail[:300]}")

    total_time = time.time() - start_time
    passed_count = sum(1 for _, _, p, _ in results if p)
    total_count = len(results)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Result: {passed_count}/{total_count} passed  ({total_time:.0f}s)")
    print(f"{'=' * 60}")

    failed = [(tid, desc, detail) for tid, desc, p, detail in results if not p]
    if failed:
        print(f"\n  Failed ({len(failed)}):")
        for tid, desc, detail in failed:
            print(f"    [{tid}] {desc}")
            print(f"      → {detail[:200]}")

    kill_procs()
    sys.exit(0 if passed_count == total_count else 1)


if __name__ == "__main__":
    main()
