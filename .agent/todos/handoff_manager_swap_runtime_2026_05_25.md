# Handoff: M-Sprint — EngineSwapRuntime Manager-driven swap path 통합 (backlog P3 잔여 a)

**작성**: 2026-05-25
**HEAD**: `9c2a9820 feat(session): M — EngineSwapRuntime Manager-driven swap path 통합 (backlog P3 잔여 a)`
**브랜치**: `worktree-b5_trait_extension`
**다음 세션 진입 문장**: "v1-3 swap 흡수 진행" (argus_cli reject 해제) 또는 "S-1+β 진행" (async_swap L256) 또는 "Manager swap E2E 측정 진행" (mock_manager로 EngineSwapRuntime path 실측)

---

## TL;DR

backlog P3 잔여 (a) 종결. CLI `--swap intra-forward` flag의 의도가 Manager 경로
에도 자연 전파되도록 `EngineSwapRuntime` 도입. Manager wire format (`SwapWeights
{ ratio, target_dtype }` 3 필드) 무변경 — engine 내부 default mode (CLI flag
normalize 결과) 로 4-way 분기. `arch/weight_swap.md §2.8.1` 신설로 mental model
(Manager=WHAT, Engine=HOW) 명시. S25 Adreno OpenCL 두 시나리오 (no secondary
NoOp + secondary force-swap) 회귀 0. 멈춤: M-Sprint 완료.

---

## 진행 상태

| Step | 작업 | 결과 |
|---|---|---|
| M1 | arch/weight_swap.md §2.8.1 — Manager WHAT vs Engine HOW 1단락 | 본 문서 |
| M2 | SwapCommitSlot enum + EngineSwapRuntime struct skeleton | engine/src/session/swap_runtime.rs (신설) |
| M3 | handle_swap_weights method + 4-way mode 분기 | 같은 파일 |
| M4 | legacy/generate.rs main() runtime 초기화 + line 2402 caller 갱신 | engine/legacy/generate.rs |
| M5 | 호스트 sanity (build + lib test + clippy + fmt) | PASS |
| M6 | S25 디바이스 게이트 — 회귀 0 + IntraForward 정상 | PASS |
| M7 | commit + handoff + push + notify | 본 문서 |

### 측정 게이트

- `cargo build --release -p llm_rs2 --lib --bins`: PASS.
- `cargo test -p llm_rs2 --lib --test-threads=1 --skip backend::opencl --skip backend::cuda --skip memory::opencl`: **1196 passed, 0 failed** (swap_runtime unit test 2 PASS).
- `cargo test -p llm_rs2 --test spec inv_layer`: **8 passed**.
- `cargo clippy -p llm_rs2 --lib --bin legacy_generate --bin argus_cli -- -D warnings`: clean.
- `cargo fmt --all`: applied.
- 호스트 CPU Qwen2.5-1.5b Q4_0 `--swap intra-forward` (no secondary): 정상 "Paris" 생성.
- 호스트 CPU `--swap intra-forward + secondary + --force-swap-ratio 0.5`: `weight_swap: intra-forward (LISWAP-4) mode — ratio=0.50, 14 target layers` 로그 + `[WeightSwap] PlanRetired: kind=IntraForward, qcf=n/a, token=0, elapsed=72.8ms` 정상.
- S25 Adreno OpenCL Qwen2.5-1.5b Q4_0 32 토큰 `--swap intra-forward` (no secondary):
  - generation: "The capital of France is Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into" — **A1 sprint과 동일**.
  - Decode 28.57 ms/tok / Avg TBT 30.28 ms (A1 28.64 / 30.39 → **회귀 0**, |Δ|≤0.3%).
- S25 Adreno OpenCL `--swap intra-forward + secondary + --force-swap-ratio 0.5`:
  - `weight_swap: intra-forward (LISWAP-4) mode — ratio=0.50, 14 target layers` + `PlanRetired (kind=IntraForward, token=0)` 정상.
  - generation: "The capital of France is Paris. The population of Paris is 2,140,87..." — 정확성 정상.
  - Decode(excl tok[0]) 36.12 ms/tok — IntraForward swap 발생 시 정상 성능.

### 핵심 변경

**arch/weight_swap.md §2.8.1 (신설)**:
- Manager → Engine swap protocol mental model: WHAT (Manager) vs HOW (Engine).
- wire format 3 필드 (`SwapWeights` variant + `target_dtype` + `ratio`) 는 mode 미포함 — 의도된 설계.
- CLI `--swap` flag = engine 내부 default mode 결정 + Manager 경로에도 그 mode 흡수.
- 한 줄 mental model: "Manager는 'Q4로 50% 바꿔라'만, Engine은 '어떤 시점에 어떤 mechanism으로 바꿀지' 자율 결정".

**engine/src/session/swap_runtime.rs (신설, 248 LOC)**:
- `SwapCommitSlot` enum: Idle / Incremental(IncrementalSwapPlan) / IntraForward(Arc<IntraForwardSwapHook>) / PhaseAware(Arc<PhaseAwareSwapDispatcher>). R-1 동시 활성화 가드 자연 해소 (단일 enum이므로 4 mode 동시 set 불가).
- `EngineSwapRuntime` struct 8 field: swap_backend / dispatcher / config / release_worker / event_sink / default_mode / phase_chunk_size_bytes / phase_max_chunks_per_token.
- `handle_swap_weights` method: validate (secondary present / ratio in (0,1] / dtype=Q4_0) + commit slot is_idle 가드 + WeightSwapDecider → selected_layers + compute_qcf_weight_swap + 4-way mode 분기 + manager_report_out set.
- unit test 2 PASS (default Idle, take resets to Idle).

**engine/src/session/cli/mod.rs::Args::resolved_swap_mode()**:
- `--swap` enum 우선, legacy 4 flag fallback, 모두 미지정 시 IntraForward (production winner).

**engine/legacy/generate.rs main()**:
- line 1153 직후 EngineSwapRuntime 초기화 — swap_backend + AsyncSwapDispatcher (manager용 전용 1회 생성, 매크로 dispatcher와 분리).
- line 2402 `dispatch_swap_weights()` → `swap_runtime.handle_swap_weights()` 치환. SwapCommitSlot ↔ 기존 3 Option 변수 (incremental_force_swap_plan / intra_forward_swap_hook / phase_aware_swap_dispatcher) 분해 어댑터 inline. decode loop의 기존 drain/retire 경로 무변경.

**unused import 정리**: `dispatch_swap_weights` from `qcf_runtime` use 제거. dispatch_swap_weights free fn은 PPL caller (session/ppl/runner.rs:829) 에서 그대로 사용 — (α) scope 외 변경 없음.

---

## 다음 작업 후보

### A. v1-3 swap 흡수 진행 (argus_cli reject 해제, 0.5~1일)

`engine/src/bin/argus_cli.rs:189-192 reject_unsupported_modes_v0` 가 swap 4 flag
차단 중. `--swap` normalize 후 `swap_intra_forward=true` 가 되어도 reject 함수가
차단 → argus_cli에서 swap 활성 불가. 본 sprint 의 EngineSwapRuntime은 argus_cli
binary에서는 사용 안 됨 (legacy_generate 전용). argus_cli 흡수 후 reject 제거 +
argus_cli/init.rs에 EngineSwapRuntime 초기화 추가.

### B. S-1+β 진행 (async_swap L256 + B/C 카테고리, 0.5~1일)

S-1+α handoff R6 §1. `engine/src/models/weights/async_swap.rs:256`
AsyncSwapDispatcher 생성자에 event_sink 인자 + caller 3곳 (intra_forward +
phase_aware + swap_dispatch) 갱신. eprintln B/C 카테고리 (SwapFailed /
BatchSummary 외) 도 event_sink emit으로 마이그.

### C. Manager swap E2E 측정 진행 (mock_manager + EngineSwapRuntime 실측, 1일)

본 sprint는 EngineSwapRuntime 코드 정합성만 검증 (호스트 build + S25 CLI force
경로). 진짜 Manager-driven 경로 (Manager → EngineCommand::SwapWeights → engine
dispatch) 는 mock_manager 또는 resilience signal로 trigger 해야 실측 가능.
manager/src/bin/mock_manager.rs 사용 + S25에서 SwapWeights signal 발사 →
IntraForward dispatch 확인 + generation 정확성 + TBT 측정.

### D. CLI 매크로 → EngineSwapRuntime 흡수 (β scope, 1.5~2일)

`engine/legacy/generate.rs:1340-1500 dispatch_force_swap!` 매크로 4-way 분기를
`EngineSwapRuntime` method 로 흡수. DRY 완전 + CLI/Manager 양쪽이 동일 method
호출. 본 sprint scope 외 (α) — 후속 sprint 가능.

---

## Landmines / 미해결

### 1. dispatch_swap_weights free fn 잔류 (PPL caller 의존)

`engine/src/session/qcf_runtime.rs:671-774 dispatch_swap_weights` 는 본 sprint
에서 제거 안 됨 — PPL caller (session/ppl/runner.rs:829) 가 사용. PPL은 mode
무관 컨텍스트 (teacher-forcing) 이므로 Incremental hardcoded 그대로 정상.
EngineSwapRuntime과 dispatch_swap_weights 양쪽이 같은 validate + decider 로직을
중복 보유 — DRY 위반이지만 (α) scope에서 의도된 트레이드오프. PPL 도
EngineSwapRuntime을 사용하도록 통합하면 DRY 달성 가능 (별 sprint).

### 2. Manager 경로 직접 실측 미수행

본 sprint 게이트는 (a) 호스트/S25 baseline (no secondary, --swap intra-forward
NoOp) 와 (b) CLI 강제 경로 (`--force-swap-ratio` + secondary + IntraForward) 의
회귀 0 검증만. Manager `EngineCommand::SwapWeights` 진입을 직접 trigger 하려면
mock_manager 또는 resilience signal 환경 필요 → 본 sprint scope 외. EngineSwapRuntime
코드 정합성은 build PASS + handle_swap_weights 4-way 분기 review + S25 CLI force
경로 회귀 0 로 간접 검증.

### 3. dispatcher 중복 생성 (매크로와 분리)

CLI 매크로 (`dispatch_force_swap!` line 1355) 와 EngineSwapRuntime (main line
1153 직후) 양쪽이 각자 `AsyncSwapDispatcher::new` 호출 → worker thread 2개
spawn 가능. 메모리/CPU 비용 미미하지만 후속 sprint D 진행 시 자연 통합. 현재는
정합성 우선.

### 4. AsyncSwapDispatcher unconditional 생성

EngineSwapRuntime은 args.swap_async_dispatch와 무관하게 main() 진입 시 항상
AsyncSwapDispatcher 1회 생성 — backend가 async_transfer 미지원 시에도 worker
spawn. 측정 noise 가능성 미미하지만 정확한 영향은 후속 microbench 필요. 현재
A1 sprint 직후 S25 게이트 회귀 0 확인 (Δ Decode ≤ 0.3%) — 가시 영향 없음.

### 5. resolved_swap_mode() default = IntraForward

`Args::resolved_swap_mode()` 가 모든 flag 미지정 시 IntraForward 반환. 이는
LISWAP-4 production winner 기반 결정. Manager 경로에서 사용자가 명시 안 한
경우에도 IntraForward로 분기 → 사용자가 mode 선택을 의식하지 않으면 자동으로
production winner 사용 (정합성 best). 단 secondary 없으면 `handle_swap_weights`
가 reject — fallback 없음. 의도된 동작.

### 6. tests/spec/*.rs `assert!(true, ...)` 7 clippy errors 잔재

본 sprint 와 무관. `cargo clippy --tests` 시 발생. lib + main bins 만으로 검증한
본 sprint 에는 영향 없음. 별 sprint 정리.

---

## 즉시 재현 명령

### 호스트 CPU (no secondary, EngineSwapRuntime NoOp)
```bash
./target/release/legacy_generate \
  --model-path /home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --tokenizer-path /home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/tokenizer.json \
  --prompt "The capital of France is" --num-tokens 8 --greedy \
  --backend cpu --kv-type f16 --no-resilience --swap intra-forward
# 기대: "The capital of France is Paris. It has a population of about"
```

### 호스트 CPU (secondary + force-swap-ratio + IntraForward 정상 dispatch)
```bash
./target/release/legacy_generate \
  --model-path /home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf \
  --tokenizer-path /home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/tokenizer.json \
  --secondary-gguf /home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --secondary-layout aos \
  --force-swap-ratio 0.5 \
  --prompt "The capital of France is" --num-tokens 8 --greedy \
  --backend cpu --kv-type f16 --no-resilience --swap intra-forward
# 기대: "weight_swap: intra-forward (LISWAP-4) mode" + "[WeightSwap] PlanRetired: kind=IntraForward"
```

### S25 Adreno OpenCL (회귀 0 게이트)
```bash
# Baseline (no secondary, --swap intra-forward NoOp)
python3 scripts/run_device.py -d galaxy_s25 legacy_generate \
  --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json \
  --prompt-file /data/local/tmp/prompts/capital.txt \
  --num-tokens 32 --greedy --backend opencl --kv-type f16 --no-resilience \
  --swap intra-forward
# 기대: "Paris. It has a population..." A1 sprint 동일, Decode ≈ 28.57 ms/tok

# Force-swap (secondary + IntraForward dispatch)
python3 scripts/run_device.py -d galaxy_s25 legacy_generate \
  --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf \
  --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json \
  --secondary-gguf /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --force-swap-ratio 0.5 \
  --prompt-file /data/local/tmp/prompts/capital.txt \
  --num-tokens 16 --greedy --backend opencl --kv-type f16 --no-resilience \
  --swap intra-forward
# 기대: "intra-forward (LISWAP-4) mode" + PlanRetired + "Paris. The population of Paris is 2,140,87..."
```

---

## 진입 명령 (다음 세션)

```
"v1-3 swap 흡수 진행"              # argus_cli reject 해제 + EngineSwapRuntime 흡수, 0.5~1일
"S-1+β 진행"                       # async_swap L256 + B/C 카테고리, 0.5~1일
"Manager swap E2E 측정 진행"        # mock_manager + EngineSwapRuntime path 실측, 1일
"CLI 매크로 → EngineSwapRuntime 흡수 진행"  # β scope, 1.5~2일
```
