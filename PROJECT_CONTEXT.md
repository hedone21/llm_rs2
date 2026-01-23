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
**Note**: Source `android.source` for NDK environment variables first.
```bash
# Release Build for Android
cargo build --target aarch64-linux-android --release

# Build specific binary
cargo build --target aarch64-linux-android --release --bin generate
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

# Run with prompt file (useful for eval/ files)
adb shell "/data/local/tmp/generate \
  --model-path /data/local/tmp/llm_rs2/models/llama3.2-1b \
  --prompt-file /data/local/tmp/llm_rs2/eval/short_len.txt \
  -n 128"
```

#### 3. Run Verification
```bash
# Backend Verification
adb shell "/data/local/tmp/test_backend"

# Micro Benchmarks
adb shell "/data/local/tmp/micro_bench"
```
