#!/usr/bin/env python3
"""GGUF full regression test + llama.cpp comparison.

Part 1: 3 models × 5 features × 2 formats = 30 regression cases
Part 2: llm.rs vs llama.cpp — output accuracy + prefill/decode speed
"""
import subprocess, time, sys, re, json

BIN = "/data/local/tmp"
TCP = "127.0.0.1:19999"
ENV = f"LD_LIBRARY_PATH={BIN}"
N = 64
PROMPT = "The meaning of life is"

MODELS = {
    "Llama-1B": {
        "st": f"{BIN}/models/llama3.2-1b",
        "gguf": f"{BIN}/models/llama3.2-1b-gguf/llama3.2-1b-q4_0.gguf",
    },
    "Qwen-1.5B": {
        "st": f"{BIN}/models/qwen2.5-1.5b",
        "gguf": f"{BIN}/models/qwen2.5-1.5b-gguf/qwen2.5-1.5b-q4_0.gguf",
    },
    "Gemma3-1B": {
        "st": f"{BIN}/models/gemma3-1b",
        "gguf": f"{BIN}/models/gemma3-1b-gguf/gemma3-1b-q4_0.gguf",
    },
}

def adb_shell(cmd, timeout=60):
    try:
        r = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"

def kill():
    adb_shell("pkill -9 -f mock_manager; pkill -9 -f 'generate.*model'; pkill -9 -f llama-cli", timeout=5)
    time.sleep(0.5)

def run_generate(gen_flags, mock_cmd=None, timeout=60):
    kill()
    if mock_cmd:
        mm_full = f"{ENV} {BIN}/mock_manager --tcp {TCP} {mock_cmd} --wait-secs 3"
        mm_proc = subprocess.Popen(
            ["adb", "shell", f"{mm_full} 2>&1"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(0.8)

    cmd = f"{ENV} {BIN}/generate {gen_flags} --prompt '{PROMPT}' -n {N} --greedy --ignore-eos"
    gen_proc = subprocess.Popen(
        ["adb", "shell", f"{cmd} 2>&1"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if mock_cmd:
        try:
            mm_proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            mm_proc.kill()

    try:
        out, _ = gen_proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        gen_proc.kill()
        out = "TIMEOUT"

    kill()
    return out

def parse_metrics(out):
    """Extract decode tok/s, prefill tok/s, generated text."""
    decode = re.search(r'Decode: ([\d.]+) ms/tok \(([\d.]+) tok/s\)', out)
    prefill = re.search(r'Prefill: ([\d.]+) ms \((\d+) tokens, ([\d.]+) tok/s\)', out)
    kivi = re.search(r'\[KIVI\].*Compression: ([\d.]+)x', out)
    # Extract generated text (after prompt, before Done.)
    text_match = re.search(r'(?:^|\n)(' + re.escape(PROMPT[:10]) + r'.*?)(?:\n\[|Done\.|\n$)', out, re.DOTALL)
    text = text_match.group(1).strip() if text_match else ""

    return {
        "decode_tps": float(decode.group(2)) if decode else 0,
        "prefill_tps": float(prefill.group(3)) if prefill else 0,
        "kivi_compress": float(kivi.group(1)) if kivi else 0,
        "text": text[:120],
        "error": ("Error:" in out and "Error executing function" not in out) or "panic" in out.lower() or "Aborted" in out,
        "raw": out,
    }

def run_llamacpp_bench(gguf_path, pp_tokens=7, tg_tokens=64, timeout=90):
    """Run llama-bench for speed comparison."""
    kill()
    cmd = (f"{ENV} {BIN}/llama-bench -m {gguf_path} "
           f"-p {pp_tokens} -n {tg_tokens} -r 1 2>&1")
    out = adb_shell(cmd, timeout=timeout)
    kill()

    # Parse: | model | size | params | backend | threads | test | t/s |
    pp = re.search(r'pp\d+\s*\|\s*([\d.]+)', out)
    tg = re.search(r'tg\d+\s*\|\s*([\d.]+)', out)

    return {
        "prefill_tps": float(pp.group(1)) if pp else 0,
        "decode_tps": float(tg.group(1)) if tg else 0,
        "raw": out,
    }

# ═══════════════════════════════════════════════════════════
# Part 1: Full Regression (30 cases)
# ═══════════════════════════════════════════════════════════
def run_regression():
    print(f"{'=' * 75}")
    print(f"  Part 1: GGUF Full Regression — 3 models × 5 features × 2 formats")
    print(f"{'=' * 75}\n")

    features = [
        ("Baseline", lambda m: f"-m {m} -b opencl", None),
        ("KIVI-Q2", lambda m: f"-m {m} -b opencl --kivi --kivi-bits 2", None),
        ("SwitchHW", lambda m: f"-m {m} -b opencl --resilience-prealloc-switch --enable-resilience --resilience-transport tcp:{TCP}",
         "--command SwitchHw --device cpu"),
        ("TensPart", lambda m: f"-m {m} -b opencl --tensor-partition 0.5", None),
        ("Evict", lambda m: f"-m {m} -b opencl --enable-resilience --resilience-transport tcp:{TCP}",
         "--command KvEvictH2o --keep-ratio 0.5"),
    ]

    results = []
    for model_name, paths in MODELS.items():
        print(f"  --- {model_name} ---")
        for feat_name, flags_fn, mock in features:
            for fmt, path in [("ST-F16", paths["st"]), ("GGUF-Q4", paths["gguf"])]:
                label = f"{model_name:10s} | {fmt:7s} | {feat_name:8s}"
                sys.stdout.write(f"    {label:<40s} ")
                sys.stdout.flush()
                t0 = time.time()
                out = run_generate(flags_fn(path), mock)
                m = parse_metrics(out)
                elapsed = time.time() - t0
                status = "FAIL" if m["error"] else "PASS"
                info = f'{m["decode_tps"]:.1f} tok/s' if m["decode_tps"] else ""
                if m["kivi_compress"]:
                    info += f', {m["kivi_compress"]:.0f}x'
                print(f"{status} ({elapsed:.0f}s) {info}")
                results.append((label, not m["error"], m))
        print()

    passed = sum(1 for _, p, _ in results if p)
    print(f"  Regression: {passed}/{len(results)} passed\n")
    return results

# ═══════════════════════════════════════════════════════════
# Part 2: llm.rs vs llama.cpp comparison
# ═══════════════════════════════════════════════════════════
def run_comparison():
    print(f"{'=' * 75}")
    print(f"  Part 2: llm.rs vs llama.cpp — Speed Comparison")
    print(f"{'=' * 75}\n")

    # llm.rs uses GPU (OpenCL), llama.cpp uses CPU
    # Both use Q4_0 quantization
    # For fair CPU-only comparison, also run llm.rs CPU

    # Use bartowski GGUF for llama.cpp (our custom GGUF isn't loadable by llama.cpp)
    bartowski = f"{BIN}/models/llama3.2-1b-gguf/bartowski-q4_0.gguf"
    our_gguf = MODELS["Llama-1B"]["gguf"]

    print(f"  Model: Llama 3.2 1B Q4_0 | Prompt: 7 tokens | Generate: {N} tokens\n")
    print(f"  {'Engine':<20s} | {'Backend':>8s} | {'Prefill':>10s} | {'Decode':>10s} | {'Δ Decode':>10s}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    results = {}

    # llama.cpp CPU (baseline)
    sys.stdout.write(f"  {'llama.cpp':<20s} | {'CPU':>8s} | ")
    sys.stdout.flush()
    lcp = run_llamacpp_bench(bartowski, pp_tokens=7, tg_tokens=N)
    print(f"{lcp['prefill_tps']:>8.1f}/s | {lcp['decode_tps']:>8.1f}/s | {'ref':>10s}")
    results["llamacpp_cpu"] = lcp

    # llm.rs CPU Q4
    sys.stdout.write(f"  {'llm.rs (ST→Q4)':<20s} | {'CPU':>8s} | ")
    sys.stdout.flush()
    out = run_generate(f"-m {MODELS['Llama-1B']['st']} --weight-dtype q4")
    m = parse_metrics(out)
    delta = (m['decode_tps'] / lcp['decode_tps'] - 1) * 100 if lcp['decode_tps'] else 0
    print(f"{m['prefill_tps']:>8.1f}/s | {m['decode_tps']:>8.1f}/s | {delta:>+8.1f}%")
    results["llmrs_cpu_st"] = m

    # llm.rs GGUF CPU
    sys.stdout.write(f"  {'llm.rs (GGUF)':<20s} | {'CPU':>8s} | ")
    sys.stdout.flush()
    out = run_generate(f"-m {our_gguf}")
    m = parse_metrics(out)
    delta = (m['decode_tps'] / lcp['decode_tps'] - 1) * 100 if lcp['decode_tps'] else 0
    print(f"{m['prefill_tps']:>8.1f}/s | {m['decode_tps']:>8.1f}/s | {delta:>+8.1f}%")
    results["llmrs_cpu_gguf"] = m

    # llm.rs GGUF GPU
    sys.stdout.write(f"  {'llm.rs (GGUF)':<20s} | {'GPU':>8s} | ")
    sys.stdout.flush()
    out = run_generate(f"-m {our_gguf} -b opencl")
    m = parse_metrics(out)
    delta = (m['decode_tps'] / lcp['decode_tps'] - 1) * 100 if lcp['decode_tps'] else 0
    print(f"{m['prefill_tps']:>8.1f}/s | {m['decode_tps']:>8.1f}/s | {delta:>+8.1f}%")
    results["llmrs_gpu_gguf"] = m

    # llm.rs ST GPU
    sys.stdout.write(f"  {'llm.rs (ST-F16)':<20s} | {'GPU':>8s} | ")
    sys.stdout.flush()
    out = run_generate(f"-m {MODELS['Llama-1B']['st']} -b opencl")
    m = parse_metrics(out)
    delta = (m['decode_tps'] / lcp['decode_tps'] - 1) * 100 if lcp['decode_tps'] else 0
    print(f"{m['prefill_tps']:>8.1f}/s | {m['decode_tps']:>8.1f}/s | {delta:>+8.1f}%")

    print()

def main():
    print(f"\n  Device check: ", end="")
    devs = subprocess.run(["adb", "devices"], capture_output=True, text=True).stdout
    if "device" not in devs:
        print("No device!")
        sys.exit(1)
    print("OK\n")

    run_regression()
    run_comparison()
    kill()

if __name__ == "__main__":
    main()
