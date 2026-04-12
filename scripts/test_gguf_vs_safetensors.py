#!/usr/bin/env python3
"""Verify KIVI, Eviction, SwitchHW, TensorPartition on GGUF Q4 vs Safetensors F16."""
import subprocess, time, sys, re

BIN = "/data/local/tmp"
TCP = "127.0.0.1:19999"
ST_MODEL = f"{BIN}/models/llama3.2-1b"
GGUF_MODEL = f"{BIN}/models/llama3.2-1b-q4_0.gguf"
ENV = f"LD_LIBRARY_PATH={BIN}"
N = 128

def adb_shell(cmd, timeout=30):
    try:
        r = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"

def kill():
    adb_shell("pkill -9 -f mock_manager; pkill -9 -f 'generate.*model'", timeout=5)
    time.sleep(0.5)

def run_test(name, gen_flags, mock_cmd=None, timeout=60):
    """Run a test: start mock_manager (optional) + generate, return (pass, summary)."""
    kill()

    if mock_cmd:
        mm_cmd = f"{ENV} {BIN}/mock_manager --tcp {TCP} {mock_cmd} --wait-secs 3"
        mm = subprocess.Popen(["adb", "shell", f"{mm_cmd} 2>&1"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(0.8)

    gen_cmd = f"{ENV} {BIN}/generate {gen_flags} --prompt Hello -n {N} --greedy --ignore-eos"
    gen = subprocess.Popen(["adb", "shell", f"{gen_cmd} 2>&1"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if mock_cmd:
        try:
            mm_out, _ = mm.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            mm.kill()
            mm_out = "TIMEOUT"

    try:
        gen_out, _ = gen.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        gen.kill()
        gen_out = "TIMEOUT"

    kill()

    # Check results
    if "panic" in gen_out.lower() or "Aborted" in gen_out:
        return False, "CRASH"
    if "Error:" in gen_out and "UNSUPPORTED" not in gen_out:
        err = re.search(r'Error: (.+)', gen_out)
        return False, f"ERROR: {err.group(1)[:80]}" if err else "ERROR"

    # Extract metrics
    decode = re.search(r'Decode: ([\d.]+) ms/tok \(([\d.]+) tok/s\)', gen_out)
    kivi = re.search(r'\[KIVI\].*Compression: ([\d.]+)x', gen_out)
    tokens = re.search(r'\[(\d+) tokens', gen_out)

    summary = ""
    if decode:
        summary += f"{decode.group(2)} tok/s"
    if kivi:
        summary += f", {kivi.group(1)}x compress"
    if tokens:
        summary += f", {tokens.group(1)} tok"
    if mock_cmd and "Response" in (mm_out if mock_cmd else ""):
        summary += ", cmd OK"

    if not decode and not kivi:
        # Check if any output was generated
        if "Done." in gen_out or "Decode:" in gen_out:
            summary = "completed"
        else:
            return False, f"No output: {gen_out[-200:]}"

    return True, summary

# ── Test matrix ──
TESTS = []

# Safetensors F16 tests
st_base = f"--model-path {ST_MODEL} -b opencl"
TESTS += [
    ("ST-F16 | Baseline",       f"{st_base}",                                          None),
    ("ST-F16 | KIVI Q2",        f"{st_base} --kivi --kivi-bits 2",                     None),
    ("ST-F16 | SwitchHW→CPU",   f"{st_base} --resilience-prealloc-switch --enable-resilience --resilience-transport tcp:{TCP}",
                                 "--command SwitchHw --device cpu"),
    ("ST-F16 | TensorPart 0.5", f"{st_base} --tensor-partition 0.5",                   None),
]

# GGUF Q4 tests
gguf_base = f"--model-path {GGUF_MODEL} -b opencl"
TESTS += [
    ("GGUF-Q4 | Baseline",       f"{gguf_base}",                                        None),
    ("GGUF-Q4 | KIVI Q2",        f"{gguf_base} --kivi --kivi-bits 2",                   None),
    ("GGUF-Q4 | SwitchHW→CPU",   f"{gguf_base} --resilience-prealloc-switch --enable-resilience --resilience-transport tcp:{TCP}",
                                   "--command SwitchHw --device cpu"),
    ("GGUF-Q4 | TensorPart 0.5", f"{gguf_base} --tensor-partition 0.5",                 None),
]

# Eviction needs longer generation to fill KV cache
st_evict = f"--model-path {ST_MODEL} -b opencl --enable-resilience --resilience-transport tcp:{TCP}"
gguf_evict = f"--model-path {GGUF_MODEL} -b opencl --enable-resilience --resilience-transport tcp:{TCP}"
TESTS += [
    ("ST-F16 | Evict H2O",     st_evict,   "--command KvEvictH2o --keep-ratio 0.5"),
    ("GGUF-Q4 | Evict H2O",    gguf_evict, "--command KvEvictH2o --keep-ratio 0.5"),
]

def main():
    print(f"{'=' * 70}")
    print(f"  GGUF Q4 vs Safetensors F16 — Feature Verification ({len(TESTS)} cases)")
    print(f"{'=' * 70}\n")

    results = []
    for name, flags, mock in TESTS:
        sys.stdout.write(f"  {name:<30s} ")
        sys.stdout.flush()
        t0 = time.time()
        passed, summary = run_test(name, flags, mock, timeout=45)
        elapsed = time.time() - t0
        status = "PASS" if passed else "FAIL"
        print(f"{status:4s} ({elapsed:.0f}s) {summary}")
        results.append((name, passed, summary))

    passed = sum(1 for _, p, _ in results if p)
    print(f"\n{'=' * 70}")
    print(f"  Result: {passed}/{len(results)} passed")
    print(f"{'=' * 70}")

    failed = [(n, s) for n, p, s in results if not p]
    if failed:
        print("\n  Failed:")
        for n, s in failed:
            print(f"    {n}: {s}")

    kill()
    sys.exit(0 if passed == len(results) else 1)

if __name__ == "__main__":
    main()
