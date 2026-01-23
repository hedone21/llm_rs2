# 📉 Android Profiling & Benchmarking Guide

This guide outlines the workflow for profiling the `llm_rs2` inference engine on Android devices. It covers building, deploying, executing benchmarks with system profiling, and visualizing the results.

## 1. Prerequisites

- **Android Device**: Connected via USB with Developer Options & USB Debugging enabled.
- **ADB**: Installed and reachable in your `PATH`.
- **Rust Toolchain**: Configured for `aarch64-linux-android`.
- **Python 3**: With `matplotlib` installed for visualization.

## 2. Methodology Overview

The profiling pipeline consists of four main steps:
1.  **Build**: Cross-compile the `generate` binary for Android.
2.  **Deploy**: Push the binary and model/prompt files to the device.
3.  **Profile**: Run the inference command wrapped in `scripts/android_profile.py` to capture performance and system metrics.
4.  **Visualize**: Generate plots using `scripts/visualize_profile.py`.

## 3. Step-by-Step Instructions

### Step 1: Build for Android

Cross-compile the `generate` binary in release mode.

```bash
cargo build --release --bin generate --target aarch64-linux-android
```

### Step 2: Deploy Artifacts

Push the binary and necessary assets to `/data/local/tmp` on the device.

```bash
# Push binary
adb push target/aarch64-linux-android/release/generate /data/local/tmp/generate
adb shell "chmod +x /data/local/tmp/generate"

# Push model (if not already present)
# adb push models/llama3.2-1b /data/local/tmp/llm_rs2/models/

# Push evaluation prompts
adb push eval/ /data/local/tmp/llm_rs2/
```

### Step 3: Run Profiling

Use the `scripts/android_profile.py` script. This wrapper handles:
- Running the `adb` command.
- Monitoring system stats (Temp, CPU/GPU Freq, Load, Memory) in the background.
- Capturing application events (Load, Prefill, Decode) from stdout.
- Saving everything to a structured JSON file.

**Example: CPU Short Prefill (128 Tokens)**
```bash
python3 scripts/android_profile.py \
    --cmd "/data/local/tmp/generate \
        --model-path /data/local/tmp/llm_rs2/models/llama3.2-1b \
        --prompt-file /data/local/tmp/llm_rs2/eval/short_len.txt \
        --num-tokens 128 \
        -b cpu" \
    --output-dir benchmarks/data
```

**Example: GPU (OpenCL) Short Prefill**
```bash
python3 scripts/android_profile.py \
    --cmd "/data/local/tmp/generate \
        ... \
        -b opencl" \
    --output-dir benchmarks/data
```

### Step 4: Visualize Results

Generate a PNG plot from the profiling JSON. The plot includes:
- **Temperature**
- **CPU Frequencies** (with Core Type labeling: Gold/Big/Little)
- **GPU Frequency**
- **Memory Usage**
- **CPU & GPU Load**
- **Event Regions** (Gray=Load, Orange=Prefill, Green=Decode)

```bash
python3 scripts/visualize_profile.py \
    benchmarks/data/profile_name.json \
    --output benchmarks/plots/profile_name.png
```

### Step 5: Update Research Log

Update the consolidated `benchmarks/README.md` table with the new results.

```bash
python3 scripts/update_benchmark_summary.py
```

This script scans `benchmarks/data/` and `benchmarks/plots/` to auto-generate the summary table.

## 4. Output Data Structure

The JSON profiles contain:
- `metadata`: Command, device models (Core types), date.
- `baseline`: Avg temp/mem/gpu load before run.
- `benchmark_results`: TTFT, TBT, Tokens/Sec.
- `events`: Timestamps for distinct phases (LoadStart, PrefillStart, DecodingStart, End).
- `timeseries`: High-resolution samples of system metrics.
