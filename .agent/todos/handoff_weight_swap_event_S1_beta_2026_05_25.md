# Handoff: S-1+β — async_swap event_sink + ConfigWarning/SubBatchWait/SwapProfBreakdown 종결

**작성**: 2026-05-25
**HEAD**: `bd0725e9 refactor(observability): S-1+β — async_swap event_sink + ConfigWarning/SubBatchWait/SwapProfBreakdown variants`
**브랜치**: `worktree-b5_trait_extension`
**다음 세션 진입 문장**: "v1-3 swap 흡수 진행" (argus_cli reject 해제) 또는 "Manager swap E2E 측정 진행" (mock_manager + EngineSwapRuntime 실측) 또는 "CLI 매크로 흡수 진행" (β scope)

---

## TL;DR

S-1+α 후속 sprint. (1) A 카테고리 잔여 1건 (`async_swap.rs:256` wait_event_blocking
failed) → `SwapFailed` emit, AsyncSwapDispatcher::new 시그니처에 `event_sink:
Arc<dyn EventSink>` 추가 + worker thread Arc clone hold. caller 4건 (production)
+ 8건 (test) 갱신. (2) B 카테고리 3건 → 신설 `ConfigWarning { source, message }`
variant — force_every_tick startup warning + phase_aware unknown subname 2건.
(3) C 카테고리 5건 → 신설 `SubBatchWait { layer_idx, wait_ms }` + `SwapProfBreakdown
{...stage timings...}` 2 variant — profiling trace를 EventSink로 격리. 세 모듈
(async_swap / phase_aware_swap / swap_executor) eprintln 0 달성. 호스트 + S25
bit-identical 회귀 0.

---

## 진행 상태

| Step | 작업 | 결과 |
|---|---|---|
| β.1 | `ConfigWarning { source, message }` variant + StderrSink arm + unit test | events.rs |
| β.2 | `SubBatchWait { layer_idx, wait_ms }` + `SwapProfBreakdown {...12 fields, source}` 2 variant + tests | events.rs |
| β.3 | `AsyncSwapDispatcher::new(backend, event_sink)` 시그니처 + worker_loop / process_commit 인자 ripple + L256 SwapFailed emit | async_swap.rs |
| β.4 | production caller 4건 (legacy/generate.rs:1138/1167/1381/1431) + 내부 test caller 4건 + spec test 4건 갱신 | legacy/generate.rs + tests/spec/ |
| β.5 | B 카테고리 3건 ConfigWarning emit (force_every_tick OnceLock 별도 1회 gate + phase_aware install bool 반환 + dispatch_chunk emit) | swap_executor.rs + phase_aware_swap.rs |
| β.6 | C 카테고리 5건 SubBatchWait + SwapProfBreakdown emit | swap_executor.rs |
| β.7 | 호스트 빌드 + lib test + spec inv_layer + clippy + fmt | PASS |
| β.8 | S25 Adreno OpenCL 32 토큰 bit-identical | PASS |
| β.9 | commit + handoff + push + notify | 본 문서 |

### 측정 게이트

- `cargo test -p llm_rs2 --lib events`: **16/16 PASS** (S-1+α 13건 + 신규 3건 — `config_warning_emit_and_collect` / `sub_batch_wait_emit_and_collect` / `swap_prof_breakdown_emit_and_collect`).
- `cargo test -p llm_rs2 --test spec inv_layer`: **8/8 PASS** (V-31 async_swap.rs:41 baseline 추가, 3 → 4).
- `cargo clippy -p llm_rs2 --lib --bin legacy_generate --bin argus_cli -- -D warnings`: clean.
- `cargo fmt --all`: applied.
- 호스트 CPU 32 토큰 bit-identical: `first=12095, final_pos=36, Decode 112.65 ms/tok` (A1+α baseline 112.02 noise 이내).
- S25 Adreno OpenCL 32 토큰 bit-identical: `first=12095, final_pos=36, Decode 28.71 ms/tok, Avg TBT 30.69 ms` (S-1+α 31.07/30.28 / v1-1 33.05 모두 |Δ|≤0.5 ms noise 이내).

### 핵심 변경

`engine/src/observability/events.rs` (+171 LOC):
- **`ConfigWarning { source: &'static str, message: String }`**: dispatcher 와 무관 — `source` 가 origin 식별 (`"force_every_tick"` / `"phase_aware_subname"` / `"phase_aware_chunk_subname"`). `kind: WeightSwapKind` 미포함. StderrSink 출력: `[WeightSwap] ConfigWarning: source={}, {message}`.
- **`SubBatchWait { layer_idx, wait_ms }`**: per-layer sub-batch wait observation. `wait_ms > 0.1` 일 때만 emit. StderrSink 출력: `[WeightSwap] SubBatchWait: layer_idx={} wait_ms={:.2}`.
- **`SwapProfBreakdown { layer_idx, subname, is_weight, tensor_size, lookup_us, dim_us, bytes_us, permute_us, wrap_us, cpu_us, upload_us, total_us, source }`**: `LLMRS_SWAP_PROFILE_BREAKDOWN=1` env-gated trace. alias / cpu early-return / pool upload / final upload 4 path 모두 같은 variant 재사용. StderrSink 출력: 기존 `[swap-prof]` line 포맷 유지 (downstream grep 호환).
- 신규 tests 3: `test_config_warning_emit_and_collect` / `test_sub_batch_wait_emit_and_collect` / `test_swap_prof_breakdown_emit_and_collect`.

`engine/src/models/weights/async_swap.rs`:
- `AsyncSwapDispatcher::new(backend: Arc<dyn Backend>, event_sink: Arc<dyn EventSink>) -> Self`. event_sink가 worker thread spawn 시 `move` capture → worker_loop / process_commit 시그니처에 `event_sink: &Arc<dyn EventSink>` 추가.
- L256 `eprintln! "[AsyncSwap] wait_event_blocking failed..."` → `event_sink.emit(CacheEvent::WeightSwap(WeightSwapEvent::SwapFailed { kind: Subsystem, reason, layer: job.layer_idx, token: None }))`. reason은 기존 stderr 메시지 그대로 (downstream grep 호환).
- 내부 4 test caller `AsyncSwapDispatcher::new(be.clone())` → `+ noop_sink()`.
- LAYER-EXEMPT 마커 추가 (line 39 직전) + 본 sprint baseline에 등록 (multi-item use parser 한계).

`engine/src/models/weights/swap_executor.rs`:
- `force_every_tick_enabled()` 함수에서 eprintln 제거 — bool만 반환. 별도 함수 `force_every_tick_warning_consumed()` (OnceLock 1회 gate)로 caller (execute_on_slots) 가 `self.event_sink.emit(ConfigWarning {source: "force_every_tick", message})` 1회 발화.
- L751 `[SubBatchWait]` eprintln → `SubBatchWait` emit (us → ms 변환은 `as f32`).
- L2039/2093/2135/2177 `[swap-prof]` 4건 → `SwapProfBreakdown` emit (각 path 의 stage timings + `source` = `"rpcmem-alias"` / `""`).

`engine/src/models/weights/phase_aware_swap.rs`:
- `PartialLayer::install(&mut self, subname, tensor) -> bool` 시그니처 변경 — Ok=true / unknown=false. caller (`dispatch_chunk` inner L448) 에서 `if !installed` → `self.event_sink.emit(ConfigWarning {source: "phase_aware_subname"})`. test caller는 bool drop.
- L373 `dispatch_chunk` unknown chunk subname → `ConfigWarning { source: "phase_aware_chunk_subname" }` emit + return Ok(()) (기존 동작 보존).

`engine/legacy/generate.rs`:
- 4 production caller `AsyncSwapDispatcher::new(...)` 전부 `+ Arc::clone(&event_sink)` 또는 `std::sync::Arc::clone(&event_sink)` 추가 (L1138 main async dispatcher / L1167 EngineSwapRuntime / L1381 phase-aware mode 매크로 / L1431 intra-forward + layer-immediate mode 매크로).

`engine/tests/spec/`:
- 3 spec test 파일 (test_async_swap_executor / test_inv_149_wait_gate_ordering / test_inv_150_plan_run_to_completion) `use llm_rs2::observability::events::noop_sink;` + caller 갱신.
- `inv_layer_baseline.json` baseline_count 3 → 4 (V-31 async_swap.rs:41 추가).

---

## 다음 작업 후보 (우선순위 순)

### A. v1-3 swap 흡수 (argus_cli reject 해제, 0.5~1일) — 가장 자연스러운 후속

`engine/src/bin/argus_cli.rs:189-192 reject_unsupported_modes_v0` 가 swap 4 flag
차단 중. `--swap` normalize 후 `swap_intra_forward=true` 가 되어도 reject 함수가
차단 → argus_cli 에서 swap 활성 불가. EngineSwapRuntime을 argus_cli/init.rs 에도
도입 + reject 제거. M-Sprint 의 EngineSwapRuntime 흡수가 legacy_generate 에만
적용된 상태 — argus_cli 도 확장.

### B. Manager swap E2E 측정 (mock_manager + EngineSwapRuntime, 1일)

M-Sprint 게이트는 CLI force 경로 (`--force-swap-ratio`) 만. 진짜 Manager-driven
경로 (Manager → `EngineCommand::SwapWeights` → `EngineSwapRuntime.handle_swap_weights`)
는 mock_manager 또는 resilience signal로 trigger 해야 실측 가능. S25에서
`weight_swap: intra-forward (LISWAP-4) mode` 로그 + `PlanRetired` + 정확성 확인.

### C. CLI 매크로 → EngineSwapRuntime 흡수 (β scope, 1.5~2일)

`engine/legacy/generate.rs:1340-1500 dispatch_force_swap!` 매크로 4-way 분기를
`EngineSwapRuntime` method 로 흡수. DRY 완전 + CLI/Manager 양쪽이 동일 method
호출. 두 dispatcher 분리 (현재 main runtime 1개 + 매크로 자체 1개) 해소.

### D. PPL caller 흡수 (별 sprint)

`engine/src/session/ppl/runner.rs:829` 가 `dispatch_swap_weights` free fn 사용.
PPL 도 EngineSwapRuntime을 사용하도록 통합 → free fn 제거 가능. teacher-forcing
컨텍스트 (mode 무관) 이지만 통합 가치 있음.

---

## Landmines / 미해결

### 1. `--secondary-gguf` 없는 환경에서 emit 발화 0

본 sprint 호스트 + S25 게이트는 모두 swap-on 시나리오가 아니라 baseline NoOp.
A/B/C 카테고리 emit 실 발화 검증은 `--secondary-gguf` + `--force-swap-ratio`
가 있는 환경에서 별도 실측 필요. unit test (`test_config_warning_emit_and_collect`
등) 가 emit 코드 path는 커버 — production 발화 검증은 device sprint scope.

### 2. force_every_tick OnceLock 1회 gate의 process-wide scope

`force_every_tick_warning_consumed()` 가 process-wide static OnceLock — 여러
SwapExecutor 인스턴스가 같은 process 안에서 동작해도 ConfigWarning 1회만 발화.
실제로 SwapExecutor는 execute_on_slots 호출마다 새로 생성되지만 OnceLock은
process scope이므로 의도된 동작. 단 multi-process 테스트 (예: 한 binary 가
fork) 에서는 각 child가 자체 OnceLock — 의도된 격리.

### 3. PartialLayer.install() 반환값 변경 — caller side-effect

`install` 시그니처가 `bool` 반환으로 바뀌어 일부 test 가 `partial.install(...)`
형태로 호출 시 unused result 경고 가능성. clippy clean이지만 미래 변경 시
`#[must_use]` 추가 검토 후보. 본 sprint는 unused result 의도 — test는 unknown
subname을 전달하지 않으므로 항상 true.

### 4. SwapProfBreakdown `source: &'static str` vs `subname: String`

stage breakdown variant의 `subname` 은 runtime-bound (`"q"` / `"k"` / 등 chunk
subname 직접 전달) 이므로 `String` 보유 — clone 비용 발생. profiling trace는
env-gated 발화이므로 hot path 영향 없음. production에서 `LLMRS_SWAP_PROFILE_BREAKDOWN`
미설정 시 t_total = None → emit 안 됨 (`if let Some(t_tot) = t_total` 가드).

### 5. async_swap.rs:41 baseline 등록 (parser 한계 회피)

multi-item `use crate::observability::events::{CacheEvent, EventSink, ...};`
라인 직전 `LAYER-EXEMPT:` 마커가 layer_lint parser에서 block-form 으로 오해됨
(S-1+α handoff R6 §4 동일 이슈). 회피책: baseline JSON에 V-31 등록. 근본 fix
는 `scripts/layer_lint.py` `_find_exempt_zone_ranges` 에 `use ... {...};` 패턴
예외 추가 — 별 sprint.

### 6. PPL caller (qcf_runtime.rs:72 / dispatch_swap_weights) 잔존

PPL teacher-forcing path 의 SwapExecutor는 여전히 event_sink NoOp 디폴트로
생성. SwapFailed emit 안 됨. PPL은 mode 무관 컨텍스트이므로 본 sprint 의
변경 영향 없음. EngineSwapRuntime 통합 (옵션 D) 시 자연 해소.

---

## 즉시 재현 명령

### 호스트 CPU 32 토큰 (baseline NoOp — emit 발화 0)
```bash
./target/release/argus_cli \
  --model-path /home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --tokenizer-path /home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/tokenizer.json \
  --prompt "The capital of France is" --num-tokens 32 --greedy \
  --backend cpu --kv-type f16 --no-resilience
# 기대: "Paris. It has a population of about 2 million people..."
# first=12095, final_pos=36, Decode ~112 ms/tok
```

### S25 Adreno OpenCL 32 토큰 (baseline 회귀 게이트)
```bash
python3 scripts/run_device.py -d galaxy_s25 argus_cli \
  --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json \
  --prompt-file /data/local/tmp/prompts/capital.txt \
  --num-tokens 32 --greedy --backend opencl --kv-type f16 --no-resilience
# 기대: 동일 출력, Decode ~28 ms/tok, Avg TBT ~30 ms
```

### B 카테고리 emit 발화 확인 (force_every_tick)
```bash
LLMRS_SWAP_FORCE_EVERY_TICK=1 ./target/release/legacy_generate \
  --model-path <F16 GGUF> --tokenizer-path <PATH> --secondary-gguf <Q4_0 GGUF> \
  --prompt "X" --num-tokens 64 --greedy --backend opencl --kv-type f16 \
  --force-swap-ratio 0.5 --swap-incremental-per-tick 2 2>&1 | grep ConfigWarning
# 기대: [WeightSwap] ConfigWarning: source=force_every_tick, LLMRS_SWAP_FORCE_EVERY_TICK=1 enabled ...
```

### C 카테고리 emit 발화 확인 (SwapProfBreakdown)
```bash
LLMRS_SWAP_PROFILE_BREAKDOWN=1 ./target/release/legacy_generate \
  --model-path <F16> --secondary-gguf <Q4_0> --force-swap-ratio 0.5 \
  --prompt "X" --num-tokens 32 --backend opencl --kv-type f16 \
  --swap-incremental-per-tick 2 2>&1 | grep '\[swap-prof\]'
# 기대: [swap-prof] layer=N sub=q is_weight=1 size=... source=rpcmem-alias (rpcmem path)
#       또는 [swap-prof] layer=N sub=q ... (memcpy path, source 없음)
```

---

## 진입 명령 (다음 세션)

```
"v1-3 swap 흡수 진행"              # argus_cli reject 해제 + EngineSwapRuntime 흡수, 0.5~1일
"Manager swap E2E 측정 진행"        # mock_manager + EngineSwapRuntime path 실측, 1일
"CLI 매크로 흡수 진행"              # β scope, 1.5~2일
"PPL caller 흡수 진행"             # dispatch_swap_weights free fn 제거, 별 sprint
```
