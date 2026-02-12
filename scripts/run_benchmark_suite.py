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
GENERATE_BIN_LOCAL = "target/aarch64-linux-android/release/generate"
GENERATE_BIN_REMOTE = f"{ANDROID_TMP_DIR}/generate"

def run_command(cmd, check=True, dry_run=False):
    if dry_run:
        print(f"[DryRun] {cmd}")
        return
    print(f"[Exec] {cmd}")
    subprocess.run(cmd, shell=True, check=check)

def generate_prompt_file(filename, approx_tokens):
    """
    Generates a prompt file with approximately `approx_tokens`.
    Assumption: 1 token ~= 4 characters on average for random English text.
    """
    # A simple seed text
    seed = "The quick brown fox jumps over the lazy dog. "
    # Calculate repeats needed
    chars_needed = approx_tokens * 4
    repeats = int(chars_needed / len(seed)) + 1
    content = seed * repeats
    
    # Trim to approx size to be safe (though tokenizer varies)
    content = content[:chars_needed]
    
    with open(filename, 'w') as f:
        f.write(content)
    print(f"[Gen] Created {filename} (~{approx_tokens} tokens)")

def build_project(dry_run=False):
    print("[Build] Building for Android aarch64...")
    run_command("cargo build --release --bin generate --target aarch64-linux-android", dry_run=dry_run)

def push_artifacts(dry_run=False):
    print("[Push] Pushing artifacts to device...")
    
    # Push binary
    run_command(f"adb push {GENERATE_BIN_LOCAL} {GENERATE_BIN_REMOTE}", dry_run=dry_run)
    run_command(f"adb shell chmod +x {GENERATE_BIN_REMOTE}", dry_run=dry_run)
    
    # Create remote eval dir
    run_command(f"adb shell mkdir -p {EVAL_DIR}", dry_run=dry_run)
    
    # Push all locally generated prompt files
    run_command(f"adb push eval/prefill_*.txt {EVAL_DIR}/", dry_run=dry_run)
    run_command(f"adb push eval/short_len.txt {EVAL_DIR}/", dry_run=dry_run)

def run_suite(args):
    # Configurations
    backends = ["cpu", "opencl"]
    # Synthetic prefill lengths
    synthetic_prefills = [128, 512, 1024]
    # Existing prompt files
    existing_prompts = ["short_len"] 
    
    decode_lengths = [128, 256]

    # Eviction policies to test
    if args.skip_eviction:
        eviction_policies = ["none"]
    else:
        eviction_policies = ["none", "sliding"]
    
    total_runs = len(backends) * (len(synthetic_prefills) + len(existing_prompts)) * len(decode_lengths) * len(eviction_policies)
    current_run = 0

    results_dir = "results/data"
    os.makedirs(results_dir, exist_ok=True)

    for backend in backends:
        # 1. Run Synthetic
        for prefill in synthetic_prefills:
            prompt_file_remote = f"{EVAL_DIR}/prefill_{prefill}.txt"
            
            for decode in decode_lengths:
              for eviction in eviction_policies:
                current_run += 1
                eviction_label = f", Eviction={eviction}" if eviction != "none" else ""
                print(f"\n[Suite] Run {current_run}/{total_runs}: Backend={backend}, Prefill={prefill}, Decode={decode}{eviction_label}")
                
                eviction_flags = ""
                if eviction != "none":
                    eviction_flags = (
                        f" --eviction-policy {eviction}"
                        f" --eviction-window {args.eviction_window}"
                        f" --memory-threshold-mb {args.memory_threshold}"
                    )
                    if args.protected_prefix > 0:
                        eviction_flags += f" --protected-prefix {args.protected_prefix}"

                device_cmd = (
                    f"{GENERATE_BIN_REMOTE} "
                    f"--model-path {MODEL_PATH} "
                    f"--prompt-file {prompt_file_remote} "
                    f"--num-tokens {decode} "
                    f"-b {backend}"
                    f"{eviction_flags}"
                )
                
                profile_cmd = (
                    f"python3 scripts/android_profile.py "
                    f"--cmd '{device_cmd}' "
                    f"--output-dir {results_dir} "
                )
                
                run_command(profile_cmd, check=False, dry_run=args.dry_run)
                if not args.dry_run:
                     time.sleep(2)

        # 2. Run Existing Prompts
        for prompt_name in existing_prompts:
            prompt_file_remote = f"{EVAL_DIR}/{prompt_name}.txt"
            
            for decode in decode_lengths:
              for eviction in eviction_policies:
                current_run += 1
                eviction_label = f", Eviction={eviction}" if eviction != "none" else ""
                print(f"\n[Suite] Run {current_run}/{total_runs}: Backend={backend}, Prefill={prompt_name}, Decode={decode}{eviction_label}")
                
                eviction_flags = ""
                if eviction != "none":
                    eviction_flags = (
                        f" --eviction-policy {eviction}"
                        f" --eviction-window {args.eviction_window}"
                        f" --memory-threshold-mb {args.memory_threshold}"
                    )
                    if args.protected_prefix > 0:
                        eviction_flags += f" --protected-prefix {args.protected_prefix}"

                device_cmd = (
                    f"{GENERATE_BIN_REMOTE} "
                    f"--model-path {MODEL_PATH} "
                    f"--prompt-file {prompt_file_remote} "
                    f"--num-tokens {decode} "
                    f"-b {backend}"
                    f"{eviction_flags}"
                )
                
                profile_cmd = (
                    f"python3 scripts/android_profile.py "
                    f"--cmd '{device_cmd}' "
                    f"--output-dir {results_dir} "
                )
                
                run_command(profile_cmd, check=False, dry_run=args.dry_run)
                if not args.dry_run:
                     time.sleep(2)

def main():
    parser = argparse.ArgumentParser(description="Run Benchmark Suite")
    parser.add_argument("--skip-build", action="store_true", help="Skip cargo build")
    parser.add_argument("--skip-push", action="store_true", help="Skip adb push")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--skip-eviction", action="store_true", help="Skip eviction policy runs (only test none)")
    parser.add_argument("--eviction-window", type=int, default=1024, help="Eviction window size (tokens)")
    parser.add_argument("--protected-prefix", type=int, default=0, help="Protected prefix tokens")
    parser.add_argument("--memory-threshold", type=int, default=256, help="Memory threshold (MB)")
    args = parser.parse_args()

    # 1. Local Setup
    if not os.path.exists("eval"): 
        os.makedirs("eval")
        
    # Generate prompt files locally
    generate_prompt_file("eval/prefill_128.txt", 128)
    generate_prompt_file("eval/prefill_512.txt", 512)
    generate_prompt_file("eval/prefill_1024.txt", 1024)

    # 2. Build
    if not args.skip_build:
        build_project(dry_run=args.dry_run)

    # 3. Push
    if not args.skip_push:
        push_artifacts(dry_run=args.dry_run)

    # 4. Run Suite
    run_suite(args)

if __name__ == "__main__":
    main()
