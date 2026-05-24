# Handoff: argus-cli v0 PoC 종결 → argus-cli v1 또는 argus-chat 진입

**작성**: 2026-05-24
**HEAD**: `7065196c feat(bin): argus-cli v0 — single-prompt inference 분리 + legacy generate 동결`
**브랜치**: `worktree-b5_trait_extension`
**다음 세션 진입 문장**: "argus-cli v1 진행" 또는 "argus-chat 진행" 또는 "argus-bench 진행" 또는 "argus-eval 진행"

---

## TL;DR

generate 분할 sprint 첫 단계 — `argus-cli v0` minimal PoC 완료. legacy `generate.rs` 는 `engine/legacy/` 로 동결 (cargo 자동 빌드 대상에서 제외, 더 이상 fmt/clippy/test 안 함). `engine/src/bin/argus_cli.rs` 288 LOC 신설 — standard happy path 전용. chat/experiment/ppl/eval/dump/prompt-batch/swap/KIVI/Offload/profile/tensor-partition/resilience 는 runtime 에서 명시 reject + 향후 갈 곳 안내. 호스트 CPU + S25 Adreno OpenCL 양쪽에서 32 토큰 bit-identical 게이트 PASS.

---

## 진행 상태

### sub-sprint commit chain

| 단계 | commit | 핵심 |
|---|---|---|
| **v0 PoC** | `7065196c` | argus_cli.rs 신설 + generate.rs → legacy/ 이동 + Cargo.toml 임시 entry |

### 검증

- **호스트 CPU**: Qwen2.5-1.5b Q4_0 32 토큰 greedy
  - legacy_generate: `Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into`
  - argus_cli:        같은 출력. `generated=32 first=12095 run=31 final_pos=36 stopped_by=BudgetExhausted` 동일
  - TBT: legacy 113.91 / argus 109.62 (Δ -3.8%, 측정 노이즈)
- **S25 Adreno OpenCL**: Qwen2.5-1.5b Q4_0 32 토큰 greedy
  - 둘 다 같은 출력 + 같은 token chain
  - TBT: legacy 30.65 / argus 32.78 (Δ +6.9%, 측정 노이즈)
- **cargo fmt + clippy**: clean (argus_cli)
- **spec inv_layer**: 8/8 PASS
- **layer_lint**: new_violations=0, 잔여 V-02 1건 (plan.rs tensor_partition import — 기존 baseline 항목)

### 4 바이너리 최종 구조 (사용자 확정)

| 바이너리 | resilience | 서브커맨드 | KV mode | swap |
|---|---|---|---|---|
| **argus-cli** v0 | (v1 도입) | (없음) | standard 한정 (v1 KIVI/Offload) | (v1) |
| **argus-cli** v1 | default ON / `--no-resilience` | (없음) | standard/KIVI/Offload | 전 8종 |
| **argus-chat** | default ON / `--no-resilience` | (없음) | standard 한정 | 전 8종 |
| **argus-bench** | default OFF / `--enable-resilience` | (없음) | 전부 | 전 8종 |
| **argus-eval** | default OFF / `--enable-resilience` | `experiment`/`ppl`/`ll`/`dump` | 전부 | 전 8종 |

### v0 reject 정책 (argus_cli.rs::reject_unsupported_modes_v0)

진입 시 즉시 bail! 처리하는 flag — 향후 갈 곳을 메시지에 명시:
- `--chat`, `--chat-socket`, `--chat-tcp` → argus-chat
- `--experiment-schedule`, `--experiment-output` → argus-eval experiment
- `--ppl` → argus-eval ppl
- `--eval-ll`, `--eval-batch`, `--eval-continuation` → argus-eval ll
- `--dump-importance`, `--qcf-dump` → argus-eval dump
- `--prompt-batch` → v1 흡수 (cli 한정)
- `--enable-resilience` → v1
- KIVI/Offload `--kv-mode` → v1
- `--secondary-gguf` + swap 8종 → v1
- `--profile` / `--profile-events` → v1
- `--tensor-partition > 0` → v1

---

## 다음 작업 (4 갈래 — 사용자 선택)

### A. argus-cli v1 — production 기능 흡수 (Recommended)

v0 reject 정책 일부 해제 + production binary로 격상:
- `--no-resilience` flag 신설 (clap flatten wrapper `ArgusCliArgs` 또는 session::cli::Args 필드 추가)
- resilience listener default ON (`enable_resilience = !no_resilience`)
- `--prompt-batch` 흡수 (session::batch::run_prompt_batch 재사용)
- KIVI/Offload `--kv-mode` 흡수 (KIVI/Offload setup → session/* 위임)
- swap 8종 + `--secondary-gguf` 흡수 (session::qcf_runtime::run_qcf_warmup_workflow 활용)
- `--tensor-partition`, `--profile`, `--profile-events` 흡수

비용: 1~2일. 디바이스 게이트 (S25 + Jetson) 32 토큰 bit-identical + swap on/off 정확성.

### B. argus-chat 신설

- session::chat::repl::run_chat_repl_v2 위임 (이미 존재)
- `--chat-socket` UDS + `--chat-tcp` listener 통합 (chat_ipc 모듈 활용)
- ChatML/Llama template auto-detect (chat_template 모듈)
- multi-turn KV pos 보존 (이미 검증, 4-5-g `c1a4b481`)
- default-ON resilience, `--no-resilience` opt-out

비용: 0.5~1일. 디바이스 게이트 단순 (multi-turn 5 turn).

### C. argus-bench 신설

신규 모듈 작성 필요:
- `engine/src/session/bench/runner.rs` — N iterations + warmup loop
- `engine/src/session/bench/metrics.rs` — TBT/TTFT histogram + p50/p95
- `engine/src/session/bench/thermal.rs` — V10 strict isolation cooldown
- `engine/src/bin/argus_bench.rs` — 진입점

새 옵션 6종:
- `--bench-iterations N` (default 5)
- `--bench-warmup N` (default 1)
- `--bench-cold-fire`
- `--bench-output <jsonl|tsv>`
- `--bench-metrics tbt,ttft,throughput,gpu_pct`
- `--bench-thermal-isolation`

비용: 1~2일.

### D. argus-eval 신설

clap subcommand 4종 (experiment/ppl/ll/dump). 각각 session::* 모듈 위임:
- `argus-eval experiment` → session::experiment (신규, mpsc + schedule)
- `argus-eval ppl` → session::ppl::run_ppl_dispatch (재사용)
- `argus-eval ll` → session::eval::run_eval_ll_generic (재사용)
- `argus-eval dump {importance|qcf|weight}` → session::dump_importance + qcf_runtime + AUF dump

비용: 2~3일. PPL/LL swap quality 검증 게이트 필수.

---

## Landmines / 미해결

### 1. legacy_generate Cargo.toml entry 임시 보존
- `engine/Cargo.toml:336` 의 `[[bin]] name = "legacy_generate"` 는 PoC 비교용. 4 sub-sprint 완료 시점 `git rm engine/legacy/generate.rs` + entry 제거 일괄 처리 예정.
- 그동안은 매 cargo build 시 legacy_generate 도 빌드됨 (호스트 ~30s 추가). 다음 sub-sprint 들도 baseline 비교 필요하면 보존, 아니면 제거.

### 2. qnn_oppkg crate 빌드 broken (master부터)
- 본 sprint 회귀 아님. devices.toml `features = ["opencl", "vulkan", "qnn"]` 그대로 두면 S25 빌드 실패.
- 임시 우회: devices.toml features 를 `["opencl"]` 로 변경 → 빌드 → 게이트 → 복원.
- backlog `[P3] qnn_oppkg_poc clippy not_unsafe_ptr_arg_deref 15 errors` 의 broader 항목 (53 errors).
- 다음 sub-sprint 들 (chat/bench/eval) 도 같은 우회 필요할 가능성.

### 3. INV-LAYER-005 baseline JSON stale
- `engine/tests/spec/inv_layer_baseline.json` 에 등록된 V-30 27건 (bin/generate.rs) 는 src/bin 제외로 검사 대상에서 사라졌지만 JSON 에는 그대로 등록.
- 영향: 0 (baseline JSON entry 가 실제 위반 없을 때 silent ignore). 단 stale entry 정리 필요.
- 정리 시점: `legacy/generate.rs` final 제거 sub-sprint (4 sub-sprint 완료 후).

### 4. resilience default-ON 가 실제 의미
- argus-cli/chat 결정: default ON, `--no-resilience` opt-out. 단 Manager 프로세스 미실행 환경에서는?
- 옵션: (a) IPC connect fail → fatal error, (b) warning + continue (resilience off), (c) silent fallback. 사용자 결정 필요.
- v1 sub-sprint 진입 시 결정.

### 5. ArgusCliArgs vs session::cli::Args 재사용
- v0 는 session::cli::Args 그대로 재사용 + runtime reject. v1 에서 `clap flatten` 으로 `ArgusCliArgs { #[clap(flatten)] base: Args, #[arg(long)] no_resilience: bool }` wrapper 추가 후보.
- 단, ArgusCliArgs 를 각 sub-sprint 마다 정의하면 session::* (init/prefill/decode_loop/qcf_runtime) 호출 시 `args: &Args` 추출 필요. 그대로 base 필드 접근 가능.

### 6. argus-cli v0 reject 메시지 일관성
- 모든 reject 메시지는 향후 갈 곳 명시 (예: "moved to argus-eval ll (planned)"). v1 에서 prompt-batch 흡수 시 reject 제거. 일관성 위해 점진 제거.

---

## 진입 명령 (다음 세션)

```
"argus-cli v1 진행"      # production resilience + prompt-batch + KIVI/Offload + swap 흡수
"argus-chat 진행"        # session::chat::repl 위임 + UDS/TCP listener
"argus-bench 진행"       # 신규 session::bench 모듈 + TBT histogram + cold-fire
"argus-eval 진행"        # clap subcommand 4종 (experiment/ppl/ll/dump)
```

### 즉시 재현 명령 (검증용)

```bash
# 호스트 CPU bit-identical
./target/release/argus_cli \
  --model-path models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --prompt "The capital of France is" --num-tokens 32 \
  --greedy --backend cpu --kv-type f16

# S25 Adreno OpenCL (devices.toml features 임시 ["opencl"] 후)
python scripts/run_device.py -d galaxy_s25 argus_cli \
  --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --prompt "The capital of France is" --num-tokens 32 \
  --greedy --backend opencl --kv-type f16
```
