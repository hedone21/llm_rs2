# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Antigravity (llm_rs2) — a high-performance on-device LLM inference framework in Rust, targeting ARM64 Android/edge devices. Supports Llama 3.2 models in HuggingFace Safetensors format with Q4_0/Q8_0 quantization and OpenCL GPU acceleration.

## Build Commands

```bash
# Android cross-compilation (MUST source env first)
source android.source
cargo build --target aarch64-linux-android --release --bin generate

# Host build (CPU-only, for development)
cargo check          # syntax check
cargo test           # unit tests (platform-agnostic logic only)

# Code quality
./.agent/skills/developing/scripts/sanity_check.sh   # runs cargo fmt + cargo clippy
```

## Testing

3-tier strategy:

1. **Host unit tests**: `cargo test` — tests tokenizer, shape inference, platform-agnostic logic
2. **Backend verification (on-device)**: `./.agent/skills/testing/scripts/run_android.sh test_backend` — validates OpenCL/CPU kernel correctness
3. **E2E inference (on-device)**: `./.agent/skills/testing/scripts/run_android.sh generate --prompt "Hello" -n 128`

Unit tests go in `#[cfg(test)] mod tests` within the same file. Every feature/fix requires a test.

## Architecture

**Module structure** (`src/lib.rs`):
- `core/` — Traits and abstractions: `Backend` (15+ ops), `Buffer`, `Tensor`, `KVCache`, eviction policies
- `backend/cpu/` — CPU backend with ARM64 NEON (`neon.rs`) and x86 AVX2 (`x86.rs`) specializations
- `backend/opencl/` — OpenCL GPU backend; kernels live in `kernels/*.cl` (~78 files)
- `models/llama/` — Llama 3.2 model loading and forward pass
- `layers/` — Transformer layer, attention (naive + flash), pre-allocated workspace buffers
- `memory/` — Galloc shared allocator
- `buffer/` — SharedBuffer (zero-copy GPU↔CPU) and UnifiedBuffer

**Key binaries** (`src/bin/`):
- `generate` — Main inference binary (single backend, CPU or OpenCL)
- `generate_hybrid` — Dynamic CPU↔GPU switching based on sequence length
- `test_backend` — Backend correctness verification (compares CPU vs OpenCL)
- `micro_bench` — Individual operator benchmarks

**Inference flow**: Prefill (batch tokens) → Decode (token-by-token). Each layer: RMSNorm → QKV matmul → RoPE → KV cache update → Attention → FFN. The model has separate `forward()` (prefill) and `forward_gen()` (decode) paths.

**Zero-copy memory**: On ARM SoCs, `CL_MEM_ALLOC_HOST_PTR` maps GPU buffers to CPU pointers, eliminating memcpy between CPU and GPU.

**KV cache eviction**: `EvictionPolicy` trait with `SlidingWindowPolicy` (keep recent N tokens) and `NoEvictionPolicy`. RoPE position increments monotonically even after eviction; physical KV cache position can decrease via `prune_prefix()`.

## Important Constraints

- **Do NOT modify `.cl` kernel files** unless explicitly instructed. They are highly optimized and stable.
- **Do NOT use `--gpu-attn` flag** unless explicitly instructed.
- The `opencl` feature is enabled by default. Host builds without a GPU will still compile but GPU ops won't run.
- Release profile uses `lto = "fat"`, `codegen-units = 1`, `opt-level = 3`.
- Android target requires NEON+dotprod; x86 target enables AVX2+FMA (set in `.cargo/config.toml`).

## Commit Convention

Conventional Commits: `type(scope): subject` — imperative present tense. Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert.

## Profiling & Benchmarks

- `scripts/android_profile.py` — On-device profiling with JSON output
- `scripts/visualize_profile.py` — Generate performance graphs
- `web_dashboard/` — Flask dashboard for benchmark visualization (`cd web_dashboard && python app.py`)
- Results stored in `results/data/` (JSON) — **committed to repo as test data**, plots in `results/plots/` (gitignored)

## Key Documentation

- `ARCHITECTURE.md` — Detailed component design, trait interfaces, execution flow
- `PROJECT_CONTEXT.md` — Implementation status and development cheat sheet
- `docs/00_build_guide.md` — Step-by-step implementation guide (build order)
- `docs/01_design_rationale.md` — Why decisions were made (Rust, OpenCL, Q4_0, etc.)
- `docs/02_core_abstractions.md` — Tensor, Buffer, Shape, DType, KVCache details
- `docs/03_cpu_backend.md` — CPU scalar + NEON SIMD + AVX2 implementation
- `docs/04_model_loading.md` — Safetensors loading, HF name mapping, Q4_0 quantization
- `docs/05_tokenizer_and_sampling.md` — Tokenizer integration and sampling algorithm
- `docs/06_opencl_backend.md` — OpenCL backend struct, init, kernel dispatch
- `docs/07_kernel_implementation.md` — OpenCL kernel algorithms and Adreno optimizations
- `docs/08_memory_management.md` — Buffer types, zero-copy, transfer patterns
- `docs/09_attention_mechanism.md` — GPU attention kernel, GQA, performance
- `docs/10_model_inference.md` — Llama 3.2 config, forward pass, LayerWorkspace
- `docs/11_kv_cache_management.md` — KV cache eviction system design
- `docs/12_hybrid_inference.md` — CPU→GPU dynamic switching strategy
- `docs/13_testing_and_benchmarks.md` — Oracle testing, micro_bench, profiling
