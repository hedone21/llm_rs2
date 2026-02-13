import argparse
import subprocess
import os
import sys
import time

# Constants
ANDROID_TMP_DIR = "/data/local/tmp"
LLM_DIR = f"{ANDROID_TMP_DIR}/llm_rs2"
EVAL_DIR = f"{LLM_DIR}/eval"
MODEL_PATH = f"{LLM_DIR}/models/llama3.2-1b"
GENERATE_BIN_REMOTE = f"{ANDROID_TMP_DIR}/generate"
RESULTS_DIR = "results/data"

def run_command(cmd, check=True, dry_run=False):
    if dry_run:
        print(f"[DryRun] {cmd}")
        return
    print(f"[Exec] {cmd}")
    subprocess.run(cmd, shell=True, check=check)

def run_scenario(name, prompt_file, num_tokens, backend, dry_run=False):
    print(f"\n[Scenario] {name}: Backend={backend}, Tokens={num_tokens}")
    
    device_cmd = (
        f"{GENERATE_BIN_REMOTE} "
        f"--model-path {MODEL_PATH} "
        f"--prompt-file {EVAL_DIR}/{prompt_file} "
        f"--num-tokens {num_tokens} "
        f"-b {backend} "
        f"--temperature 0" # Deterministic
    )
    
    # Add a tag to the output filename for easy identification
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"profile_{backend}_{name}_{timestamp}.json"
    
    profile_cmd = (
        f"python3 scripts/android_profile.py "
        f"--cmd '{device_cmd}' "
        f"--output-dir {RESULTS_DIR} "
        f"--output-name {output_filename}"
    )
    
    run_command(profile_cmd, check=False, dry_run=dry_run)
    if not dry_run:
        time.sleep(5) # Cool down between runs

def main():
    parser = argparse.ArgumentParser(description="Run Comparison Benchmarks")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--backend", choices=["cpu", "opencl", "all"], default="all", help="Backend to run (default: all)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    backends = ["cpu", "opencl"] if args.backend == "all" else [args.backend]

    # 1. Baseline Latency (Responsiveness)
    if "cpu" in backends: run_scenario("baseline_128", "short_len.txt", 128, "cpu", args.dry_run)
    if "opencl" in backends: run_scenario("baseline_128", "short_len.txt", 128, "opencl", args.dry_run)

    # 2. Long Generation (Stability & Thermals)
    if "cpu" in backends: run_scenario("long_2048", "short_len.txt", 2048, "cpu", args.dry_run)
    if "opencl" in backends: run_scenario("long_2048", "short_len.txt", 2048, "opencl", args.dry_run)

    # 3. Prefill Performance (Prompt Processing)
    if "cpu" in backends: run_scenario("prefill_1024", "prefill_1024.txt", 128, "cpu", args.dry_run)
    if "opencl" in backends: run_scenario("prefill_1024", "prefill_1024.txt", 128, "opencl", args.dry_run)

if __name__ == "__main__":
    main()
