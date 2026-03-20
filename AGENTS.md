# llm.rs Project Guide

프로젝트의 AI 에이전트 작업 가이드.

## System Instructions

- **Language**: 모든 응답, 리포트, 계획, 설명은 한국어로 작성한다. 기술 용어와 코드 식별자는 원문 유지.

## Project Overview

llm.rs (repo: llm_rs2) — a high-performance on-device LLM inference framework in Rust, targeting ARM64 Android/edge devices. Supports Llama 3.2 models in HuggingFace Safetensors format with Q4_0/Q8_0 quantization and OpenCL GPU acceleration.

## Build Commands

```bash
# Android cross-compilation (MUST source env first)
source android.source
cargo build --target aarch64-linux-android --release -p llm_rs2 --bin generate

# Host build (CPU-only, for development)
cargo check --workspace    # syntax check (all crates)
cargo test -p llm_rs2      # unit tests (engine)
cargo test -p llm_shared   # unit tests (shared types)

# Code quality
./.agent/skills/developing/scripts/sanity_check.sh   # runs cargo fmt + cargo clippy (workspace)
```

## Testing

3-tier strategy:

1. **Host unit tests**: `cargo test` — tests tokenizer, shape inference, platform-agnostic logic
2. **Backend verification (on-device)**: `./.agent/skills/testing/scripts/run_android.sh test_backend` — validates OpenCL/CPU kernel correctness
3. **E2E inference (on-device)**: `./.agent/skills/testing/scripts/run_android.sh generate --prompt "Hello" -n 128`

Unit tests go in `#[cfg(test)] mod tests` within the same file. Every feature/fix requires a test.

## Model Weights

`models/` 디렉토리는 `.gitignore`에 등록되어 git에 포함되지 않습니다. 호스트 PC에서 테스트할 때 모델 가중치를 이 경로에 저장합니다.

```bash
# HuggingFace에서 모델 다운로드
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir models/llama3.2-1b

# 호스트에서 추론 테스트
cargo run --release --bin generate -- --model-path models/llama3.2-1b --prompt "Hello" -n 128
```

| 경로 | 용도 |
|------|------|
| `models/llama3.2-1b/` | 호스트 PC 테스트용 (gitignored) |
| `/data/local/tmp/models/llama3.2-1b` | Android 디바이스용 |

## Architecture

**Cargo workspace** with 3 Rust crates:
- `engine/` — LLM inference engine (`llm_rs2` crate)
- `shared/` — Shared signal types (`llm_shared` crate)
- `manager/` — System resource manager service (`llm_manager` crate)

**Non-Rust components**:
- `web_dashboard/` — Web dashboard (Python/Flask)

**Engine module structure** (`engine/src/lib.rs`):
- `core/` — Traits and abstractions: `Backend` (17+ ops), `Buffer`, `Tensor`, `KVCache`, eviction policies
- `backend/cpu/` — CPU backend with ARM64 NEON (`neon.rs`) and x86 AVX2 (`x86.rs`) specializations
- `backend/opencl/` — OpenCL GPU backend; kernels live in `engine/kernels/*.cl` (~80 files)
- `models/llama/` — Llama 3.2 model loading and forward pass
- `layers/` — Transformer layer, attention (naive + flash), pre-allocated workspace buffers
- `memory/` — Galloc shared allocator
- `buffer/` — SharedBuffer (zero-copy GPU↔CPU) and UnifiedBuffer
- `resilience/` — Resilience manager (D-Bus/UnixSocket IPC, strategy patterns)

**Key binaries** (`engine/src/bin/`):
- `generate` — Main inference binary (single backend, CPU or OpenCL)
- `generate_hybrid` — Dynamic CPU↔GPU switching based on sequence length
- `test_backend` — Backend correctness verification (compares CPU vs OpenCL)
- `micro_bench` — Individual operator benchmarks
- `test_model` — Model loading verification
- `signal_injector` — Resilience signal injection for testing

**Inference flow**: Prefill (batch tokens) → Decode (token-by-token). Each layer: RMSNorm → QKV matmul → RoPE → KV cache update → Attention → FFN. Model uses `forward_into()` for unified forward pass; eviction is caller's responsibility via `CacheManager`. `LlamaLayer::forward()` internally dispatches to a private `forward_gen()` path when `seq_len == 1`.

**Zero-copy memory**: On ARM SoCs, `CL_MEM_ALLOC_HOST_PTR` maps GPU buffers to CPU pointers, eliminating memcpy between CPU and GPU.

**KV cache eviction**: `EvictionPolicy` trait with `NoEvictionPolicy`, `SlidingWindowPolicy` (keep recent N tokens), `H2OPolicy` (3-partition: prefix + heavy hitters + recent window), and `H2OPlusPolicy` (per-head GQA-aware variant). Also `D2OHandler` (merge compensation via `CachePressureHandler`). RoPE position increments monotonically even after eviction; physical KV cache position can decrease via `prune_prefix()`.

## Important Constraints

- **Do NOT modify `.cl` kernel files** unless explicitly instructed. They are highly optimized and stable.
- **Do NOT use `--gpu-attn` flag** unless explicitly instructed.
- The `opencl` feature is enabled by default. Host builds without a GPU will still compile but GPU ops won't run.
- Release profile uses `lto = "fat"`, `codegen-units = 1`, `opt-level = 3`.
- Android target requires NEON+dotprod; x86 target enables AVX2+FMA (set in `.cargo/config.toml`).

## Commit Convention

Conventional Commits: `type(scope): subject` — imperative present tense. Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert.

## Agent System

5개 특화 서브에이전트가 `.claude/agents/`에 정의되어 있다. 메인 세션이 오케스트레이터 역할을 하며 에이전트 간 결과를 전달한다.

| Agent | Role | Tools | Scope |
|-------|------|-------|-------|
| **PM** | 계획 수립, TODO 관리, 우선순위 조정, 작업 배분 제안 | Read, Glob, Grep, Edit | `.agent/todos/*.md`만 수정 |
| **Architect** | 코드 분석, SOLID 설계, 아키텍처 문서 작성 | Read, Glob, Grep, Edit | `docs/*.md`, `ARCHITECTURE.md`만 수정 |
| **Implementer** | 코드 구현, 유닛 테스트, 버그 수정, sanity check | Read, Edit, Write, Glob, Grep, Bash | `engine/`, `shared/`, `manager/` 소스 코드 |
| **Tester** | 호스트/디바이스 테스트 실행, 결과 분석, 품질 게이트 검증 | Read, Glob, Grep, Bash | 수정 불가, 실행만 |
| **Researcher** | 논문 분석, 기술 조사, 적용 가능성 평가 | Read, Glob, Grep, WebSearch, WebFetch | 수정 불가, 조사 결과 반환만 |

**Workflow**:
```
[PM] 계획/TODO → [Architect] 설계 → [Implementer] 구현+테스트 → [Tester] 검증
                                       ↑
                               [Researcher] 기법 조사
```

**제약**: 서브에이전트는 다른 서브에이전트를 호출할 수 없다 (최대 1단계). 작업 위임은 메인 세션이 담당.

## Skill System

`.claude/skills/`에 에이전트별 특화 스킬이 정의되어 있다. 스크립트는 `.agent/skills/*/scripts/`에 위치.

| Skill | 용도 | 주 사용 Agent |
|-------|------|--------------|
| **sanity-check** | cargo fmt + clippy + test | Implementer |
| **deploy-test** | Android 빌드→배포→테스트 | Tester |
| **profile** | 온디바이스 프로파일링 + 시각화 | Tester, Implementer |
| **dashboard** | 웹 대시보드 실행/관리 | (공용) |
| **design-review** | SOLID 원칙 기반 코드 구조 검토 | Architect |
| **research** | 논문/기술 조사 + 적용 가능성 평가 | Researcher |

## Workflow Rules

- **Auto-commit on completion**: Implementer가 작업을 완료하면 자동으로 커밋한다. 미커밋 작업을 남기지 않는다.
- **Desktop notification on completion**: 작업 완료 후 `notify-send "llm.rs" "<task summary>"`로 데스크톱 알림을 보낸다.

## Profiling & Benchmarks

- `scripts/android_profile.py` — On-device profiling with JSON output
- `scripts/visualize_profile.py` — Generate performance graphs
- `web_dashboard/` — Flask dashboard for benchmark visualization (`cd web_dashboard && python app.py`)
- Results stored in `results/data/` (JSON) — **committed to repo as test data**, plots in `results/plots/` (gitignored)

## Key Documentation

- `ARCHITECTURE.md` — Detailed component design, trait interfaces, execution flow
- `docs/PROJECT_CONTEXT.md` — Implementation status and development cheat sheet
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
- `docs/20_dbus_ipc_spec.md` — D-Bus IPC specification for Resilience Manager
- `docs/21_resilience_architecture.md` — Resilience system architecture and strategy patterns
- `docs/22_resilience_integration.md` — Phase 3 generate.rs integration design spec
- `docs/README.md` — Documentation index and reading order guide
- `docs/14_component_status.md` — Component quality gates and test status
- `docs/15_test_strategy.md` — Resilience test strategy (T1-T4 tiers)
- `docs/23_resilience_test_strategy.md` — Resilience integration test summary
- `docs/24_resilience_usage_guide.md` — Resilience system usage guide
- `docs/25_troubleshooting.md` — Troubleshooting guide
- `docs/26_api_reference.md` — Resilience API reference
- `docs/27_manager_architecture.md` — Manager service internal architecture (3-layer, OCP PolicyEngine)
- `docs/28_experiment_guide.md` — Experiment guide
- `docs/29_manager_monitor_redesign.md` — Manager monitor redesign
- `docs/30_evaluation_methodology.md` — KV Cache Eviction evaluation methodology (related work survey + benchmark design)
- `docs/35_experiment_runner_guide.md` — **실험 에이전트 인수인계 문서** (바이너리, 러너, 디바이스, CLI, 스키마, 트러블슈팅)
- `docs/31_memory_architecture.md` — Memory architecture overview (Buffer → KV Cache → Policy unified view)
- `docs/32_kv_offload.md` — KV cache offload (RawStore, PrefetchController, PreloadPool)
- `docs/34_profiling_framework_design.md` — Inference profiling framework design

## Experiment Benchmarks

Benchmark prompts for KV cache eviction evaluation in `experiments/prompts/`:

```bash
# Perplexity: 5 domain prompts (PPL-01 ~ PPL-05)
# NIAH: Parameterized needle-in-a-haystack prompts
python experiments/prompts/assemble_niah.py --needle N-PASS --depth 0.25 --blocks 4
python experiments/prompts/assemble_niah.py --all --output niah_all.json

# QA: LongBench-style single-doc QA, summarization, few-shot, multi-hop
# All prompts defined in experiments/prompts/benchmark_prompts.json
```

See `experiments/PLAN.md` Section 10 for experiment matrix (Round 10-12).

## Device Registry

TOML-based device configuration at `devices.toml` (project root). Manages build targets, connection info, and device paths for all scripts.

```bash
# CLI commands
python scripts/device_registry.py discover          # scan & register devices
python scripts/device_registry.py list               # show registered devices
python scripts/device_registry.py validate           # check TOML schema

# Unified runner (build -> deploy -> execute)
python scripts/run_device.py -d pixel generate --prompt "Hello" -n 128
python scripts/run_device.py -d host test_backend
python scripts/run_device.py -d pixel --skip-build generate -b opencl

# Existing scripts with --device option
python scripts/stress_test_device.py --device pixel --phases 1,4
python scripts/run_benchmark_suite.py --device pixel --dry-run
python scripts/run_comparison_benchmark.py --device pixel --dry-run
```

Package: `scripts/device_registry/` — config.py (TOML loader), connection.py (Connection ABC), builder.py (cargo build), deployer.py (binary push), discover.py (device scan).

## Web Dashboard

```bash
cd web_dashboard && .venv/bin/python app.py   # http://localhost:5000
```

Tabs: Overview, Table, Detail, Compare, Trends, Runner, Gates, Todos. API endpoints under `/api/`. Dashboard uses Flask + Plotly.js, venv at `web_dashboard/.venv/`.

## TODO System

`.agent/todos/`에서 역할별 작업 추적. 형식 및 워크플로우 규칙은 `.agent/todos/README.md` 참고.

| TODO 파일 | 담당 Agent |
|-----------|-----------|
| `backlog.md` | PM이 관리, 미배정 작업 |
| `architect.md` | Architect |
| `rust_developer.md` | Implementer |
| `tester.md` | Tester |
| `tech_writer.md` | Researcher |
| `frontend_developer.md` | (메인 세션 직접 처리) |

Dashboard에서 조회: Todos 탭 또는 `curl http://localhost:5000/api/todos`.
