# Manager SwapWeights Routing — 진행 중 작업 Handoff (2026-05-12)

## TL;DR

**옵션 (c) — manager SwapWeights 신호를 engine 내부 incremental K + dynamic-K + sub-batch pause path 로 routing** 작업이 **부분 완료** 상태로 중단됨. Commit `25130cb` 는 `dispatch_swap_weights` 를 IncrementalSwapPlan commit 으로 변경했으나, **prefill loop 의 `executor.poll` 이 SwapWeights directive 를 먼저 흡수해서 decode loop 에 도달 못 함** — 실제 swap 안 일어남. **`CommandExecutor` 가 `swap_weights` directive 를 sticky state 로 보관하는 후속 fix 필요**.

## 현재 상태

### Commit

```
25130cb feat(liswap6): manager SwapWeights routes through incremental + dynamic-K + sub-batch pause path
b501054 feat(liswap6): CLI default async+dynamic-K ON + production handoff
81a99c2 feat(liswap6): sub-batch reactive pause — burst truncate by release queue state
e58d31e feat(liswap6): dynamic-K controller — timing-based safe K 자동 결정
```

본 handoff 시점에 `25130cb` 는 origin 에 push 완료 가정 (handoff 커밋과 함께 push).

### 디바이스 측정 결과 (Galaxy S25, R3CY408S5SB)

**시나리오 A — Manager-triggered swap** (검증 목적, 본 작업의 핵심):
```
mock_manager --tcp 127.0.0.1:9999 --command SwapWeights --ratio 0.9 --target-dtype q4_0
```

| 지표 | 측정값 | 기대값 | 판정 |
|---|---|---|---|
| Directive 수신 | OK | OK | ✓ |
| `Response: [Ok]` | OK | OK (즉시 ack) | ✓ |
| `[WeightSwap] manager path: ...` log | **없음** | 출력 | ✗ |
| `[IncrementalSwap] tick=...` log | **없음** | 14 ticks | ✗ |
| `[SwapPeak] max_release_pending=0` | **없음** | 출력 | ✗ |
| `[DynamicK] calibrated safe_k=2` | **없음** | 출력 | ✗ |
| WeightSwapReport | **없음** | layers_swapped=25 | ✗ |

→ **manager directive 가 처리되긴 하나 incremental swap path 진입 못 함**. 실제 swap 0 layer.

**시나리오 B — CLI path 회귀** (commit `81a99c2` baseline 비교):
```
generate --force-swap-ratio 0.9 --swap-incremental-per-tick 2 ...
```

| 지표 | 측정값 | 기대값 | 판정 |
|---|---|---|---|
| `[DynamicK] calibrated safe_k=2` | OK | OK | ✓ |
| `[SwapPeak] max_release_pending=0` (14 tick) | OK | 0 | ✓ |
| `[SwapPeak] sub_batch_cutoff=0` | OK | 0 | ✓ |
| Decode quality | plausible | plausible | ✓ |
| Avg TBT | 33.25 ms | 31~33 ±2 ms | ✓ |

→ **CLI path 정상**. `25130cb` 변경이 CLI 회귀 만들지 않음.

## 근본 원인 (tester 분석)

`engine/src/resilience/executor.rs:541` 의 `apply_command` 가 `SwapWeights` directive 를 **transient** `plan.swap_weights` 필드에만 기록 (sticky state 아님).

`engine/src/bin/generate.rs` 의 `executor.poll` 호출 위치 5개:

| Line | 위치 | `plan.swap_weights` 처리? |
|---|---|---|
| 4052 | prefill loop | ✗ (consume 없이 plan drop) |
| 4912 | prefill loop | ✗ (consume 없이 plan drop) |
| 6524 | **decode loop** | ✓ (consume) |
| 9234 | (다른 path) | ✗ |
| 9792 | (다른 path) | ✗ |

`mock_manager --wait-secs 0` 으로 즉시 directive 보내면:
1. engine 의 prefill loop 가 `executor.poll` 호출
2. directive 가 `plan.swap_weights` 에 기록됨
3. prefill loop 가 plan 처리 후 drop → **swap_weights 소실**
4. decode loop 의 line 6524 `executor.poll` 시점에는 plan 없음 → swap 안 일어남

**race 없는 fix = `CommandExecutor.swap_weights` 를 sticky state 로 보관**.

## 다음 세션 첫 작업 — 두 가지 선택

### Option A: Revert `25130cb` (작업 백아웃)

```bash
git revert 25130cb
git push origin master
```

이유:
- 본 작업의 ROI 가 명확하지 않으면 revert
- Manager path 의 spike 위험은 코드 분석 시점에서 발견된 잠재 이슈 — 실제 manager 사용 시나리오가 production scope 안에 있는지 불확실
- CLI path 는 이미 spike-safe (commit `81a99c2`) → manager path 우선순위 낮을 수 있음

리스크: revert 시 사용자 hard constraint (`feedback_no_memory_spike.md`) 가 manager 신호 path 에서 보호 안 됨 — non-alias 환경 / 다른 모델에서 real spike 가능.

### Option B: Sticky state fix 진행 (작업 완료)

**구현 위치**: `engine/src/resilience/executor.rs`

```rust
pub struct CommandExecutor {
    ...
    // 신규: SwapWeights directive 를 sticky state 로 보관.
    // poll 이 directive 를 받으면 여기에 저장, decode loop 가 consume.
    pending_swap_weights: Option<(f32, DtypeTag)>,
}

impl CommandExecutor {
    pub fn take_pending_swap_weights(&mut self) -> Option<(f32, DtypeTag)> {
        self.pending_swap_weights.take()
    }
}

// apply_command 안:
EngineCommand::SwapWeights { ratio, target_dtype } => {
    self.pending_swap_weights = Some((ratio, target_dtype));
    // plan.swap_weights 는 backward compat 로 유지하거나 제거
}
```

**호출 site 변경**: `engine/src/bin/generate.rs` decode loop 6524 line 근처 — `plan.swap_weights` 대신 `executor.take_pending_swap_weights()` 호출.

**디바이스 재검증** (시나리오 A 재실행):
```bash
adb -s R3CY408S5SB shell "cd /data/local/tmp && \
  export LD_LIBRARY_PATH=... LLMRS_SUB_BATCH_PAUSE_DIAG=1 LLMRS_SWAP_DRAIN_DIAG=1 LLMRS_DYNAMIC_K_DIAG=1 && \
  ./generate -m qwen2.5-1.5b-f16.gguf --tokenizer-path tokenizer.json \
    --secondary-gguf qwen2.5-1.5b-q4_0.gguf --backend qnn_oppkg --temperature 0 \
    --manager-tcp 127.0.0.1:9999 -p 'The capital of France is' -n 30"

# 별도 셸:
adb -s R3CY408S5SB shell "cd /data/local/tmp && \
  ./mock_manager --tcp 127.0.0.1:9999 \
    --command SwapWeights --ratio 0.9 --target-dtype q4_0 --wait-secs 0"
```

기대 (시나리오 A FAIL → PASS 전환):
- `[DynamicK] calibrated safe_k=2` 출력
- `[IncrementalSwap] tick=...` 14 ticks 출력
- `[SwapPeak] max_release_pending=0`
- WeightSwapReport.layers_swapped=25

**예상 작업량**: 100~150 LoC + 디바이스 검증. 0.5d.

## 참조 코드 위치

### Commit `25130cb` 의 변경
- `engine/src/bin/generate.rs`
  - `dispatch_swap_weights` (line 8080~8195) — IncrementalSwapPlan commit 으로 재설계
  - `async_swap_dispatcher` 초기화 조건 완화 (line 2854~)
  - `dynamic_k_controller` 초기화 조건 완화 (line 2878~)
  - `manager_swap_report_pending` + `ready_weight_swap_report` 스테이징 (line 2832~)
  - 호출 site 업데이트 (line 6525)

### 미수정 — fix 필요한 부분
- `engine/src/resilience/executor.rs:541` `apply_command::SwapWeights` arm — transient plan 기록만, sticky state 없음
- `engine/src/bin/generate.rs` 의 `executor.poll` 5개 호출 위치 중 decode loop (6524) 만 swap_weights consume

## 디바이스 환경 재현 (다른 세션에서)

```bash
# 1. 모델 파일 디바이스 확인 (이미 배포되어 있음)
adb -s <serial> shell "ls /data/local/tmp/models/qwen2.5-1.5b-gguf/"

# 2. Android cross-compile
python scripts/run_device.py -d <device_key> build  # 또는
source android.source && cargo build --release --target aarch64-linux-android \
  -p llm_rs2 --features opencl,qnn

# mock_manager 별도 빌드 (run_device.py 가 default_features=false 라 manager lua resolve 실패 — workaround)
source android.source && cargo build --release --target aarch64-linux-android \
  --bin mock_manager  # 기본 features 사용

# 3. 디바이스 배포
adb -s <serial> push target/aarch64-linux-android/release/generate /data/local/tmp/
adb -s <serial> push target/aarch64-linux-android/release/mock_manager /data/local/tmp/
```

## 발견된 부수 이슈

1. **`run_device.py` build 실패** — `devices.toml` 이 `default_features=false` 로 강제 → `mock_manager` 의 `crate::lua_policy` resolve 실패 (`pipeline.rs:71`). workaround: `cargo build --bin mock_manager` 직접 호출 (default features 사용).
2. **종료 시 SIGSEGV** — generation 완료 후 cleanup phase. 사전 존재 이슈 (task #33). 측정 결과에 영향 없음.
3. **adb 연결 끊김** — thermal/USB 사유로 한 번 측정 후 끊김. 재연결 가능.

## 코드/측정/문서 산출물

### 본 작업 산출물
- commit `25130cb` (구현, 부분 완료 — 검증 FAIL)
- 측정 로그: 디바이스 `/data/local/tmp/_mock_mgr_out.log`, `/data/local/tmp/_engine_err.log` (다음 세션에서 새 측정 시 덮어쓰기 가능)
- 본 handoff

### 이전 작업 연계 산출물 (변경 없음)
- `docs/48_swap_dynamic_k_guide.md` — 3-layer safety net 사용 가이드
- `.agent/todos/handoff_dynamic_k_2026_05_12.md` — dynamic-K + sub-batch pause 종합 handoff
- `.agent/todos/handoff_swap_memory_spike_constraint_2026_05_11.md` — spike constraint 정책 (link 됨)
- 메모리 `feedback_no_memory_spike.md`, `feedback_swap_async_default.md`, `project_liswap6_alias_production.md`

## 권장

**Option B (sticky state fix) 진행 권장**. 이유:
- 옵션 (c) 의 의도 (manager 신호 → spike-safe path) 가 사용자 hard constraint 준수에 직결
- 변경 범위 작음 (100~150 LoC)
- 검증 방법 명확 (시나리오 A 재실행)
- CLI path 영향 없음 (회귀 가드)

다만 manager 사용 빈도가 낮으면 Option A revert 도 합리적 — 다음 세션 사용자 결정.

## 다음 세션 시작 명령

```bash
git log --oneline -6
# (이번 push 후) 26xxxxx docs(liswap6): manager swap routing handoff — 작업 중단 상태
#               25130cb feat(liswap6): manager SwapWeights routes through ...
# ...

# Option B 진행 시:
# 1. engine/src/resilience/executor.rs 의 apply_command 와 CommandExecutor 구조체 확인
# 2. pending_swap_weights: Option<(f32, DtypeTag)> 필드 추가
# 3. apply_command::SwapWeights arm 에서 self.pending_swap_weights = Some(...) 저장
# 4. take_pending_swap_weights() method 추가
# 5. generate.rs decode loop 6524 line 근처에서 plan.swap_weights 대신 executor.take_pending_swap_weights() 호출
# 6. 디바이스 시나리오 A 재검증
```
