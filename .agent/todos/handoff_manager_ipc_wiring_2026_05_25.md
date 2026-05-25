# Handoff: Manager IPC wiring P1~P4 종결 → v1-2 또는 P5+

**작성**: 2026-05-25
**HEAD**: `e7929a9a fix(resilience): graceful fallback when Manager unreachable`
**브랜치**: `worktree-b5_trait_extension` (origin push 완료)
**다음 세션 진입 문장**: "argus-cli v1-2 prompt-batch 진행" 또는 "Manager IPC P5+ 진행" 또는 "PR 생성"

---

## TL;DR

argus-cli ↔ Manager IPC wiring sprint 4 commits 완료. trait 분리(P1~P2) →
ResilienceAdapter + CommandExecutor 이식(P3~P4) → graceful fallback fix.
호스트 + S25 Adreno OpenCL 두 모드(default-on / `--no-resilience`)
bit-identical PASS. mock_manager E2E PASS. 다음 갈래: v1-2 prompt-batch 흡수,
또는 P5+ (KVSnapshot dtype 실값/gpu_meter Arc/capability 점검), 또는 PR 정리.

---

## 진행 상태

### Commit chain (4건)

| commit | 단계 | 내용 |
|---|---|---|
| `59b9da40` | P1~P2 | `session/{traits,defaults,decode_loop,mod}.rs` 갱신. `EngineReport`/`TokenTickSink`/`ResilienceBundle` 신규, `CommandSource::poll` 시그니처 확장(`KVSnapshot` 추가 + `ExecutionPlan` 반환), `DecodeLoop::run/run_until_stop`에 `plan.suspended`/`plan.throttle_delay_ms` minimal consume + `tick_sink.on_token_generated` hook |
| `dfded1a6` | P3~P4 | `session/resilience_{adapter,init}.rs` 신규. `ResilienceAdapter`(3 trait impl) + `build_command_executor`(legacy:596~700 외과적 이식). `DecodeLoopBuilder::with_resilience(adapter)` 추가(Arc<Mutex> + newtype wrappers). `build_standard_loop`/`StandardHappyCtx`에 `resilience: Option<ResilienceAdapter>` 인자 추가. argus-cli main에서 `args.enable_resilience` 시 executor 생성+adapter wrapping |
| `e7929a9a` | 회귀 fix | `resilience_init.rs`. transport spawn 실패/unknown transport/feature off 시 `eprintln!` warn + `Ok(None)` 반환 (graceful fallback). default-on + default `dbus` transport + `feature=resilience` off 조합에서 모든 일반 추론이 깨지던 회귀 차단 |
| (이전) `c231b110` | arch docs | drift 정리 (§2 디렉토리/§3 Binaries/§4-§8 빌더 예시) |

### 게이트 결과

| 게이트 | 결과 |
|---|---|
| `cargo build --release --bin argus_cli` | PASS (41s) |
| `cargo fmt --all` | PASS |
| `cargo clippy -p llm_rs2 --bin argus_cli -- -D warnings` | PASS (0 warning) |
| `cargo test -p llm_rs2 --lib session` | PASS (100/100) |
| `cargo test -p llm_rs2 --test spec inv_layer` | PASS (8/8) |
| `layer_lint.py --baseline` new_violations | 0 |
| 호스트 CPU 32 토큰 default-on (graceful) | "Paris..." generated=32 first=12095 final_pos=36 |
| 호스트 CPU 32 토큰 `--no-resilience` | bit-identical |
| S25 Adreno OpenCL default-on | bit-identical TBT 30.74 ms (v1-1 baseline 32.85 대비 -6.4%) |
| S25 Adreno OpenCL `--no-resilience` | bit-identical TBT 32.96 ms (Δ +0.3%) |
| mock_manager E2E (P3~P4 sprint) | Capability 전달 + 8 토큰 정상 생성 |
| Manager 미연결 (default-on) | `[Resilience] Manager unreachable ..., running without resilience.` warn + 정상 추론 |

### master 대비 미머지 commit (11건)

```
e7929a9a fix(resilience): graceful fallback when Manager unreachable
dfded1a6 refactor(session): trait split P3~P4 — ResilienceAdapter + CommandExecutor 이식 + argus-cli wiring
c231b110 docs(arch): drift 정리 — §2 디렉토리 트리 + §3 Binaries 표 + inference_pipeline 빌더 예시
59b9da40 refactor(session): trait split P1~P2 — EngineReport/TokenTickSink + ResilienceBundle + ExecutionPlan minimal consume
83d7cb4a docs(handoff): argus-cli v1-1 종결 — resilience default-on
14b3de7a refactor(cli): argus-cli v1-1 resilience default-on
a7122009 style: cargo fmt
9ba98ea7 ci/build: feature-matrix 도입
532e65d8 build(qnn_oppkg): SDK 경로 3단계 fallback
e2f5feff fix(swap_executor): import path
3afcc47b refactor(weights): map_weights_for_host_access cfg-free wrapper
```

---

## 다음 작업 (3 갈래 — 우선순위 순)

### A. argus-cli v1-2 prompt-batch 흡수 (예상 0.5~1일)

- reject 1줄 제거: `engine/src/bin/argus_cli.rs::reject_unsupported_modes_v0` 의 `args.prompt_batch.is_some()` 분기.
- 분기: happy path 가드 이전에 `if args.prompt_batch.is_some() { return run_prompt_batch(BatchRunCtx { ... }) }`.
- `BatchRunCtx` 필드 조립(`cache_manager`, `score_accumulator`, `command_executor`, `skip_config`, `actual_protected_prefix`, `score_based_eviction`, `throttle_delay_ms`, `last_skip_ratio`, `hidden_size` 등)이 legacy generate.rs 596~1100 에 분산되어 있어 SessionInitCtx 확장 또는 main 직접 조립 둘 중 선택.
- `command_executor` 는 본 sprint에서 만든 `Option<CommandExecutor>` 를 그대로 BatchRunCtx로 전달 가능 (ResilienceAdapter::executor_mut() 노출이라 가능). 또는 happy path 처럼 ResilienceAdapter로 감싸지 않고 raw executor를 BatchRunCtx에 직접 주입 — 어느 쪽이 옳은지 별 점검.
- 게이트: 호스트 CPU + S25 OpenCL prompt-batch 파일(N=2~4 entries) 정상 출력.

### B. Manager IPC P5+ — KVSnapshot 실값 / gpu_meter / capability 점검 (예상 1~1.5일)

P3~P4의 미해결 4건:

1. **KVSnapshot.kv_dtype 실값 주입** — 현재 stub (`String::new()`). Manager 정책 엔진이 dtype 기반 정책 분기(KIVI quantization 등) 시 거짓 정보. `DecodeLoopBuilder` 가 `KVSnapshotMeta` (kv_dtype/protected_prefix/bytes_per_token) struct 받아 `DecodeLoop` 보유, `build_kv_snapshot()` 에서 진짜 값 조립.

2. **gpu_meter backend Arc 추가** — 현재 `build_command_executor` 가 backend Arc 미수신 → silent None. `build_command_executor(args, model, backend: &Arc<dyn Backend>)` 시그니처 확장 + OpenCL downcast 시도. `--heartbeat-gpu-profile` 효과 회복.

3. **capability advertise subset 검증** — Manager 정책 엔진이 `available_actions=[throttle/set_target_tbt/suspend/reject_new/limit_tokens/restore_defaults]` 만 advertise 했을 때 진짜로 swap/evict/switch_hw 등 위험 action을 발송 안 하는지 mock_manager + 실제 manager 양쪽에서 확인. 만약 무시한다면 engine side에 명시적 reject 또는 plan filter 추가.

4. **ResilienceBundle 통합 슬롯의 `Arc<Mutex>` 비용** — token당 3 lock (poll/report/tick). 무시 가능하다고 가정했으나 microbench로 확정. S25 Qwen2.5-1.5b Q4_0 Δ TBT 측정.

### C. PR 생성 + master 머지 (예상 1~2h)

- master 대비 11 commits 미머지. 단일 PR로 묶거나 분할 결정 필요.
- `gh auth login` 필요 (auth 미설정 상태).
- PR 분할 후보:
  - PR1: B-1~B-5 + fmt + cli v1-1 (5 commits, backend feature unification + cli v1-1)
  - PR2: arch drift + Manager IPC wiring (5 commits)
  - PR3: graceful fallback (1 commit)
- 또는 단일 PR (11 commits) — 리뷰 부담 큼.

---

## Landmines / 미해결

### 1. default-on의 silent fallback 우려 (사용자 인지 필요)

`e7929a9a` 의 graceful fallback 은 v1-1 의 default-on 회귀를 차단했지만, **Manager 활성을 의도한 사용자가 fallback 됐다는 사실을 놓칠 수 있음**.
- mitigation 후보: `--require-resilience` flag 도입 (있으면 fallback 안 함, fail-loud). 또는 stderr warn 색깔 강조.
- 현 정책: silent + warn 1줄. 사용자가 `[Resilience] Manager unreachable ...` 를 놓치면 NoOp 으로 진행됨.

### 2. KVSnapshot stub의 dtype/policy/protected_prefix 거짓 정보

P5에서 해결 예정. 현재 happy path는 Throttle/Suspend 만 발송하는 Manager 가정이라 거짓 dtype 무영향 — 단 Manager가 dtype 보고 다른 결정(예: KIVI suggest)을 내리는 코드가 있다면 회귀.

### 3. legacy generate path는 본 wiring 무관

`engine/legacy/generate.rs` 의 거대 main()은 본 sprint 변경 0. legacy path는 그대로 자체 CommandExecutor 생성/사용. argus-cli만 새 wiring 사용. master 머지 후 두 path 가 영구 공존(다수 바이너리 분할 방향, [[generate-split-binaries]]).

### 4. resilience_init.rs:36~43 의 gpu_meter 추출 dead code

P3~P4에서 `#[cfg(feature = "opencl")] if args.heartbeat_gpu_profile { let _ = args.heartbeat_gpu_profile; }` 형태 stub. backend Arc 미수신이라 추출 불가. P5에서 본격 fix 시 함께 cleanup 또는 dead code 명시 표시 추가.

### 5. mock_manager 가 unknown_actions 발송 시 engine 거동 미검증

P3~P4 의 mock_manager E2E sanity 는 capability 전달 + 정상 8 토큰만 확인. unsafe action (swap_weights/kv_evict_h2o 등) 을 manager가 정책 위반으로 발송했을 때 engine 이 어떻게 응답하는지 미확인. ExecutionPlan 의 해당 필드를 happy path 가 무시 (eviction/swap_weights/switch_device/layer_skip/partition_ratio 등) 하므로 silent ignore 추정. 명시적 reject 또는 log 는 P5 항목.

### 6. INV-LAYER-005 baseline JSON stale (이전 sprint 부터)

`engine/tests/spec/inv_layer_baseline.json` 의 V-30 27건 (bin/generate.rs) entry 는 검사 대상에서 사라졌지만 JSON 갱신 미적용. silent ignore. 본 sprint 무관, cleanup만 필요.

---

## 즉시 재현 명령

```bash
# 호스트 CPU
./target/release/argus_cli \
  --model-path /home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --tokenizer-path /home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/tokenizer.json \
  --prompt "The capital of France is" --num-tokens 32 --greedy --backend cpu --kv-type f16
# stderr: [Resilience] Manager unreachable ..., running without resilience.
# stdout: Paris. It has a population of about 2 million people ...

# S25 Adreno OpenCL
python scripts/run_device.py -d galaxy_s25 argus_cli \
  --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json \
  --prompt-file /data/local/tmp/prompts/capital.txt \
  --num-tokens 32 --greedy --backend opencl --kv-type f16

# mock_manager E2E
manager/target/release/mock_manager --tcp 127.0.0.1:55556 &
./target/release/argus_cli \
  --model-path .../qwen2.5-1.5b-q4_0.gguf --tokenizer-path .../tokenizer.json \
  --prompt "The capital of France is" --num-tokens 8 --greedy --backend cpu --kv-type f16 \
  --resilience-transport "tcp:127.0.0.1:55556"
```

### v1-2 prompt-batch 진입 시 1차 grep 목록

```bash
# BatchRunCtx 필드 출처 파악
grep -nE "let (cache_manager|command_executor|score_accumulator|skip_config|throttle_delay_ms|last_skip_ratio|score_based_eviction|actual_protected_prefix|hidden_size)" engine/legacy/generate.rs

# run_prompt_batch 시그니처
grep -nE "pub fn run_prompt_batch|pub struct BatchRunCtx" engine/src/session/batch/

# argus-cli reject 함수 위치
grep -nE "prompt_batch" engine/src/bin/argus_cli.rs
```
