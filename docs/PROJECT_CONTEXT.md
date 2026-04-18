# Project Context: llm_rs2

## 1. Project Overview & Structure
**llm_rs2** is a Rust-based LLM serving framework optimized for mobile (aarch64) and Generic Linux environments. It supports multiple backends (CPU, OpenCL, QNN) and quantization for efficient on-device inference.

### Key Directories
- **`src/bin/`**: Entry points for CLI tools.
    - `generate.rs`: Main text generation tool.
    - `micro_bench.rs`: Low-level performance benchmarks.
    - `test_backend.rs`: Hardware-specific backend verification.
- **`src/models/`**: Model architectures (currently `llama/`).
- **`src/backend/`**: Hardware backends (`cpu/`, `opencl/`).
- **`src/core/`**: Core abstractions (`Tensor`, `Buffer`, `Device`).
- **`kernels/`**: OpenCL / compute kernels.

### Model Weights
- **`models/`**: 모델 가중치 저장 디렉토리 (gitignored). 호스트 PC에서 테스트 시 사용.
    - `models/llama3.2-1b/`: Llama 3.2 1B 모델 (HuggingFace Safetensors 포맷)
    - 다운로드: `huggingface-cli download meta-llama/Llama-3.2-1B --local-dir models/llama3.2-1b`
- **Android 디바이스**: `/data/local/tmp/models/llama3.2-1b`

## 2. Target Environment
- **Primary Target**: Android (aarch64) on Mobile SoC (e.g., Snapdragon).
- **Secondary Target**: Generic Linux (x86_64 / aarch64).
- **Build Host**: Linux.
- **Languages**: Rust (Core), OpenCL C (Kernels).

## 3. Implementation Status
### Supported Models
- **Llama 3.2 1B**: Supported via `src/models/llama`.
    - Implements `safetensors` loading.
    - Supports KV Caching & RoPE.

### Compute Backends
| Backend | Status | Key Features |
| :--- | :--- | :--- |
| **CPU** | **Active** | Base implementation. Optimized Attention (Loop Interchange) for long contexts. |
| **OpenCL** | **Active** | Kernels for `MatMul` (F32, F16, Q4_0, Q8_0), `RoPE`, `Softmax`. |
| **QNN** | *Planned* | Feature-gated, currently inactive. |

### Quantization (DTypes)
- **Supported**: `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `U8`.

## 4. Testing & Verification Strategy

### Terminology (User Preference)
- **"Test" (테스트)**: Running the model inference via the provided scripts (e.g., `generate`), including generating the benchmark results/logs and making them visible on the Dashboard.
- **"Unit Test" (유닛테스트)**: Standard Rust unit tests or backend verification logic (e.g., `test_backend`).

### Testing Philosophy
1.  **Correctness First**: Ensure `test_backend` passes on target device before optimizing.
2.  **Performance Check**: Use `micro_bench` for hot-path analysis.
3.  **End-to-End**: Use `generate` for integration testing.

### Tooling
- **Unit Tests**: `cargo test` (Host-side logic).
- **On-Device Tests**:
    - `test_backend`: Validates generic tensor ops on GPU/CPU.
    - `micro_bench`: Measures raw kernel performance.

## 5. Development Cheat Sheet (Commands)

### Build
**Preferred**: use `run_device.py` which auto-injects NDK env from `hosts.toml`. Run `python scripts/device_registry.py bootstrap-host` once to generate `hosts.toml` (NDK auto-detected).
```bash
# Build + deploy + run
python scripts/run_device.py -d pixel generate

# Build + deploy only (no execute)
python scripts/run_device.py -d pixel --skip-exec generate --extra-bin llm_manager

# Direct cargo (legacy, not recommended): source android.source first
# cargo build --target aarch64-linux-android --release --bin generate
```

### Usage Guidelines
> [!IMPORTANT]
> **GPU Attention (`--gpu-attn` / `--attn-gen`) Policy**:
> Do NOT use the `--gpu-attn` (or `--attn-gen`) option when running with the GPU backend unless EXPLICITLY instructed by the user. Default to standard OpenCL execution.

### Deploy & Run (Android)
Use `adb` to push artifacts to `/data/local/tmp/`.

#### 1. Push Artifacts
```bash
adb push target/aarch64-linux-android/release/generate /data/local/tmp/
adb push target/aarch64-linux-android/release/test_backend /data/local/tmp/
```

#### 2. Run Generation
```bash
adb shell "chmod +x /data/local/tmp/generate && \
/data/local/tmp/generate \
  --model-path /data/local/tmp/llm_rs2/models/llama3.2-1b \
  --prompt 'tell me a long story' \
  -b opencl \
  -n 128 \
  --temperature 0"
  -b opencl \
  -n 128 \
  --temperature 0"

# Run with prompt file
adb shell "/data/local/tmp/generate \
  --model-path /data/local/tmp/llm_rs2/models/llama3.2-1b \
  --prompt-file /data/local/tmp/llm_rs2/experiments/prompts/short_len.txt \
  -n 128"
```

#### 3. Run Verification
```bash
# Backend Verification
adb shell "/data/local/tmp/test_backend"

# Micro Benchmarks
adb shell "/data/local/tmp/micro_bench"
```

## 6. Profiling & Visualization Workflow
For comprehensive performance analysis (CPU/GPU load, Thermal Throttling, Memory), use the dedicated Python scripts.

> [!TIP]
> **Detailed Instructions**: See [results/GUIDE.md](results/GUIDE.md) for the complete end-to-end workflow.
> **Research Log**: See [results/README.md](results/README.md) for historical results and plots.

### Quick Commands
```bash
# 1. Profile (CPU 128 Tokens)
python3 scripts/android_profile.py \
    --cmd "/data/local/tmp/generate --model-path ... -n 128 -b cpu" \
    --output-dir results/data

# 2. Visualize
python3 scripts/visualize_profile.py \
    results/data/profile_latest.json \
    --output results/plots/profile_latest.png

# 3. Update Log
python3 scripts/update_benchmark_summary.py
```
