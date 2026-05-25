# Handoff: A1 — `--swap [MODE]` shorthand + legacy 4 flag deprecation 종결

**작성**: 2026-05-25
**HEAD**: `6c6a8eeb feat(cli): A1 — --swap [MODE] shorthand + legacy 4 flag deprecation (backlog P3)`
**브랜치**: `worktree-b5_trait_extension`
**다음 세션 진입 문장**: "Manager swap IntraForward 진행" (잔여 a) 또는 "v1-3 swap 흡수 진행" (argus_cli reject) 또는 "S-1+β 진행" (async_swap L256)

---

## TL;DR

backlog P3 종결. 옵션 (b) CLI shorthand 선택. `--swap [intra-forward|
incremental|phase-aware|layer-immediate]` 도입 + flag 단독 시 default
IntraForward. 기존 4 flag (`--swap-intra-forward` 등)는 deprecated +
stderr 1회 경고 후 그대로 동작. `Args::normalize_swap_shorthand()`이
`Args::parse()` 직후 1회 호출되어 legacy field로 변환 — dispatch path는
무변경. S25 Adreno OpenCL Qwen2.5-1.5b Q4_0 32 토큰 bit-identical 검증
PASS. 잔여: Manager-driven swap path + argus_cli v0 reject 함수의 swap
차단 (별 sprint).

---

## 진행 상태

| Step | 작업 | 결과 |
|---|---|---|
| A1.1 | `SwapMode` enum + `parse_swap_mode` + `swap: Option<SwapMode>` field | engine/src/session/cli/mod.rs |
| A1.2 | `Args::normalize_swap_shorthand()` + caller 2곳 (argus_cli, legacy/generate) | argus_cli.rs:39 + legacy/generate.rs:97 |
| A1.3 | 호스트 빌드 + parser + deprecation + clippy + fmt | PASS |
| A1.4 | S25 디바이스 게이트 bit-identical | PASS |
| A1.5 | backlog P3 mark + handoff + commit + push + notify | 본 문서 |

### 측정 게이트

- `cargo build --release -p llm_rs2 --lib --bins`: PASS (1m 03s).
- `cargo test -p llm_rs2 --lib --test-threads=1 --skip backend::opencl --skip backend::cuda --skip memory::opencl`: **1194 passed, 0 failed**.
- `cargo test -p llm_rs2 --test spec inv_layer`: **8 passed**.
- `cargo clippy -p llm_rs2 --lib --bin argus_cli --bin legacy_generate -- -D warnings`: clean.
- `cargo fmt --all`: applied.
- `--help` 노출: `--swap [<SWAP>]` 정상 표시.
- parser: 4 모드 alias (`intra-forward`/`intra_forward`/`intraforward` 등) PASS, `garbage` rejection PASS.
- legacy `--swap-intra-forward` 직접 사용 → stderr `[deprecation]` 1회.
- S25 Adreno OpenCL Qwen2.5-1.5b Q4_0 32 tokens generation **bit-identical**:
  - baseline: "The capital of France is Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into" / Decode 30.72 ms/tok / Avg TBT 32.71 ms / first=12095 / final_pos=36
  - `--swap intra-forward`: **동일 생성 텍스트** / Decode 28.64 ms/tok / Avg TBT 30.39 ms (-7% — swap codepath "forward only" sync 동작 차이)

### 핵심 변경

`engine/src/session/cli/mod.rs`:
- `SwapMode` enum + `parse_swap_mode` value_parser (case-insensitive + alias `intra_forward`, `intraforward`, `phase_aware`, `phaseaware`, `layer_immediate`, `layerimmediate`).
- `swap: Option<SwapMode>` field, `#[arg(long, value_parser=parse_swap_mode, num_args=0..=1, default_missing_value="intra-forward")]`. `--swap` 단독 → IntraForward, `--swap <MODE>` → 명시 모드.
- `Args::normalize_swap_shorthand()` (impl Args 블록 진입부): Some(mode)→ legacy field set (Incremental + K=0 → K=2 default), None + legacy direct → `std::sync::Once` stderr 1회 경고.

`engine/src/bin/argus_cli.rs:39`, `engine/legacy/generate.rs:97`:
- `Args::parse()` 직후 `args.normalize_swap_shorthand()` 호출 추가.

`.agent/todos/backlog.md`:
- P3 entry → DONE 마킹 + 잔여 (a)/(c) Manager-driven path 별 sprint 명시.

---

## 다음 작업 후보

### A. Manager-driven swap path 자동 IntraForward 분기 (backlog 잔여)

`engine/src/session/qcf_runtime.rs:754` (sync path) + Manager-driven
`executor.execute_on_slots` caller가 IntraForward dispatcher로 분기하도록.
현재는 `incremental_force_swap_plan` 단일 진입 + `kind=IntraForward` fallback
(S-1+α landmine R6 §2). 0.5~1일 + Manager 정책 영향 분석 필요.

### B. argus_cli v1-3 — swap 흡수

argus_cli.rs:189-192 `reject_unsupported_modes_v0`에서 swap 4 fields 차단 중.
현재는 `--swap`이 normalize 후 reject됨 → argus_cli에서 swap 활성 불가.
legacy_generate만 swap 사용 가능. v1-3에서 argus_cli swap 흡수 → reject 함수
제거. 0.5~1일.

### C. S-1+β — `async_swap.rs:256` + B/C 카테고리

S-1+α handoff R6 §1. AsyncSwapDispatcher 생성자에 event_sink 인자 +
caller 3곳 (intra_forward + phase_aware + swap_dispatch) 갱신. 단독 0.5~1일.

---

## Landmines / 미해결

### 1. Decode TBT measurement 차이 (bit-identical과 무관)

S25 게이트에서 `--swap intra-forward` 사용 시 Decode 28.64 ms/tok (forward
only label), baseline 30.72 ms/tok. 7% 빠르게 측정되지만 **generation 텍스트는
완전 동일**. 원인: IntraForwardSwapHook 등록 시 `forward_into` 경로의 sync
동작이 약간 달라짐 (force_swap_ratio 없으니 실제 swap dispatch는 NoOp).
swap-on production 측정에서는 force_swap_ratio + secondary 모두 set한 환경
사용. paper measurement에 영향 없음.

### 2. argus_cli v0 reject 함수 (B 항목)

`engine/src/bin/argus_cli.rs:189-192`의 `reject_unsupported_modes_v0`이
swap 4 fields (`swap_incremental_per_tick > 0` 포함)를 차단. `--swap`
normalize 후 swap_intra_forward=true가 되어도 reject 함수가 차단 → argus_cli
swap 활성 불가. legacy_generate만 swap 사용 가능. v1-3 sprint에서 흡수
예정.

### 3. Manager-driven swap path 미통합 (A 항목)

`--swap` shorthand는 CLI flag 변환만 처리. Manager가 SwapWeights command를
보낼 때는 여전히 `incremental_force_swap_plan` (legacy default) 사용.
backlog P3 잔여 (a) 옵션. Manager 정책 layer 영향 큰 작업으로 별 sprint.

### 4. legacy 4 flag deprecation timing

deprecation 경고는 stderr 1회만 출력 (std::sync::Once). 자동 테스트 환경
(CI 등)에서 stderr 출력이 결과 파일에 섞여 들어가지 않도록 운영 가이드
업데이트 필요할 수 있음. 향후 4 flag 제거 시점은 미정 — 별 sprint.

### 5. tests/spec/*.rs `assert!(true, ...)` 7 clippy errors 잔재

본 변경과 무관. `cargo clippy --tests` 시 발생. lib + main bins 만으로
검증한 본 sprint에는 영향 없음. 별 sprint 정리.

---

## 즉시 재현 명령

### 호스트 (argus_cli, --help + parser 동작)
```bash
./target/release/argus_cli --help | grep -A6 "\-\-swap "
./target/release/argus_cli --model-path /nonexistent --tokenizer-path /nonexistent --swap=garbage
# 기대: error: invalid value 'garbage' for '--swap [<SWAP>]'
./target/release/argus_cli --model-path /nonexistent --tokenizer-path /nonexistent --swap-intra-forward
# 기대: [deprecation] ... 1줄
```

### 호스트 CPU (legacy_generate, --swap normalize 동작)
```bash
./target/release/legacy_generate \
  --model-path /home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --tokenizer-path /home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/tokenizer.json \
  --prompt "The capital of France is" --num-tokens 8 --greedy --backend cpu --kv-type f16 \
  --no-resilience --swap intra-forward
# 기대: 동일 생성 텍스트, deprecation 경고 없음
```

### S25 Adreno OpenCL (bit-identical 게이트)
```bash
# Baseline (no swap flag)
python3 scripts/run_device.py -d galaxy_s25 legacy_generate \
  --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json \
  --prompt-file /data/local/tmp/prompts/capital.txt \
  --num-tokens 32 --greedy --backend opencl --kv-type f16 --no-resilience
# 기대: "Paris. It has a population..." / first=12095 / final_pos=36

# --swap intra-forward (normalize NoOp without secondary)
python3 scripts/run_device.py -d galaxy_s25 legacy_generate \
  --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json \
  --prompt-file /data/local/tmp/prompts/capital.txt \
  --num-tokens 32 --greedy --backend opencl --kv-type f16 --no-resilience \
  --swap intra-forward
# 기대: 동일 생성 텍스트
```

---

## 진입 명령 (다음 세션)

```
"Manager swap IntraForward 진행"   # backlog 잔여 (a), 0.5~1일
"v1-3 swap 흡수 진행"              # argus_cli reject 해제, 0.5~1일
"S-1+β 진행"                       # async_swap L256 + B/C 카테고리, 0.5~1일
```
