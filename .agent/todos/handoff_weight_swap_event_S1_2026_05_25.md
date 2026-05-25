# Handoff: S-1 — WeightSwapEvent + EventSink 통합 완료

**작성**: 2026-05-25
**HEAD**: `13b63385 refactor(observability): S-1 — WeightSwapEvent enum + EventSink 통합`
**브랜치**: `worktree-b5_trait_extension`
**다음 세션 진입 문장**: "S-1+α 진행" (sub-module emit 13건) 또는 "swap default 변경 진행" (backlog P3) 또는 "PR 생성"

---

## TL;DR

KV/Weight 관리 구조 평가(Round 1/2 + Devil's Advocate)에서 격차 top 1(이벤트 인프라 47:1)으로 식별된 항목 해소. `CacheEvent::WeightSwap(WeightSwapEvent)` variant 추가 + `swap_dispatch.rs` A 11건 emit 치환. 호스트 CPU + S25 Adreno OpenCL bit-identical 회귀 0. 단 sub-module 13건(intra_forward / phase_aware / swap_executor / async_swap)은 생성자 갱신 필요로 별 sprint 보류. plan은 `.agent/todos/plan_weight_swap_event_S1_2026_05_25.md`.

---

## 진행 상태

| Step | 작업 | 결과 |
|---|---|---|
| S-1.0 | eprintln 60건 4 카테고리 분류 | A 24건 (게이트 ≥20 PASS) — 자료 `$CLAUDE_JOB_DIR/eprintln_classification.md` |
| S-1.1 | `WeightSwapEvent` 4 variant + `CacheEvent::WeightSwap` + StderrSink match arm 4개 + 3 tests | events.rs 11/11 PASS |
| S-1.2 | `SwapDispatchCtx.event_sink: &Arc<dyn EventSink>` 23번째 필드 + generate.rs L1086 단일 Arc 생성 | cargo check PASS |
| S-1.3 | swap_dispatch.rs A 11건 emit 치환 (J3=7 / J4=2 / J5=2) | 19→8 eprintln (잔존 B/C 카테고리) |
| S-1.4 | host CPU 32 토큰 sanity | bit-identical (first=12095, final_pos=36), Decode 112.01 ms/tok |
| S-1.5 | S25 Adreno OpenCL 32 토큰 게이트 | bit-identical, TBT 30.67 ms (v1-1 33.01 대비 -7.1%) |
| S-1.6 | 단일 commit `13b63385` (+685/-54) | 5 files changed |

### 핵심 변경

`engine/src/observability/events.rs`:
- `WeightSwapEvent` enum 4 variant: `PlanCommitted` / `ChunkDrained` / `PlanRetired` / `SwapFailed`. 모든 variant에 `kind: &'static str` 필드 (Incremental / IntraForward / PhaseAware / Subsystem).
- `CacheEvent::WeightSwap(WeightSwapEvent)` variant 추가.
- `StderrDiagnosticSink::emit` match arm 4개 추가, 모든 메시지 첫 컬럼에 `kind=` 노출.
- 3 신규 tests: noop accept / collecting 4 kind / clone+debug.

`engine/src/session/decode_fallback/swap_dispatch.rs`:
- 22 → 23 필드 (`event_sink: &'a Arc<dyn EventSink>` 추가).
- A 11건 eprintln → `ctx.event_sink.emit(CacheEvent::WeightSwap(...))`.

`engine/legacy/generate.rs`:
- L1086 부근에 `event_sink: Arc<dyn EventSink> = Arc::new(StderrDiagnosticSink)` 단일 인스턴스 생성.
- SwapDispatchCtx 생성 시 `event_sink: &event_sink`.

`.agent/todos/backlog.md`:
- [P3] "swap 기본 모드 `--swap-intra-forward`" 항목 등록.

---

## 다음 작업 후보 (우선순위 순)

### A. S-1+α — sub-module emit 13건 (예상 1~2일)

본 sprint scope 축소로 보류된 13건:
- `intra_forward_swap.rs` 2건 (L459, L504): IntraForwardSwapHook에 `event_sink` 필드 추가 + 생성자 인자 갱신.
- `phase_aware_swap.rs` 5건 (L387, L403, L464, L486, L528): PhaseAwareSwapDispatcher 동일 패턴.
- `swap_executor.rs` 6건 (L880, L890, L909, L972, L1000, L1285): SwapExecutor 동일 패턴 + `kind` 인자 전파 검토 (caller가 Incremental/IntraForward/PhaseAware 중 어디서 fire인지 추적).
- 게이트: 동일 호스트 + S25 bit-identical.

### B. swap 기본 모드 `--swap-intra-forward` 결정 (backlog P3)

3 옵션 중 사용자 결정 필요:
- (a) Manager-driven path 변경 (`executor.rs:549~`) — blast radius 큼
- (b) `--swap` shorthand 신규 flag
- (c) `--enable-swap` shorthand + IntraForward fallback

S-1 완료로 IntraForward path event 발행 정착 → (b)/(c) 진입 안전.

### C. PR 생성

master 대비 13 commits 미머지. gh CLI auth 필요. 분할 PR 후보:
- Manager IPC wiring 4 commits (이전 sprint)
- S-1 1 commit (본 sprint)

### D. KV/Weight 평가 Round 2 격차 잔여 (보류 항목 재고)

- DIP `SwapDispatchCtx` 분해 — 22 필드 16:1:1 J3 편중, G3 LOC 회귀 위험. 우선순위 낮음.
- WeightStrategy — Manager-side 부재 정당화 (ENG-ALG-214-ROUTE), engine-side `SwapAlgorithm` trait 보류.

---

## Landmines / 미해결

### 1. sub-module 13건은 본 sprint scope 외

`intra_forward_swap.rs` / `phase_aware_swap.rs` / `swap_executor.rs` 직접 emit은 SwapDispatchCtx 접근 불가 (hook/dispatcher 자체 구조체). event_sink를 hook 생성 시 inject 필요 → blast radius 증가. S-1+α 별 sprint.

### 2. host OpenCL backend 22 tests pre-existing fail

`cargo test --lib` 전체 실행 시 backend::opencl::{noshuffle,kv_scatter_batch,gpu_buffer_shift}_tests 22건 실패. master 동일 환경에서 같은 fail (검증 완료). NVIDIA OpenCL fallback 환경 known issue.

### 3. ChunkDrained 의미 모호 (LISWAP-2 drain)

L254 LISWAP-2 dispatcher drain은 chunk 단위 layer 아님. `ChunkDrained { layers_done: 0, stages: Some("liswap-2-drain") }`로 매핑했지만 운영자가 ChunkDrained 카운트 시 혼동 가능. 별 variant (`DispatcherDrained`) 도입 검토 — S-1+α에서 결정.

### 4. PlanRetired.elapsed_ms 0.0 fallback

J3 incremental retire 시 async dispatcher가 없으면 drain_ms = 0.0. plan 시작~종료 elapsed는 inc_plan에서 추출 가능하나 본 sprint에서는 단순화로 0.0 유지. S-1+α에서 보강.

### 5. SwapAlgorithm enum 미적용

PlanCommitted variant의 `algorithm: &'static str` 필드는 본 sprint에서 사용되지 않음 (swap_dispatch.rs A 11건에 PlanCommitted emit 없음 — incremental_plan 생성 시점은 generate.rs L1448에서 J3 시작 전). 이 emit은 S-1+α에서 추가.

### 6. Subsystem kind 미사용

`Subsystem` kind는 doc string에 정의되어 있으나 본 sprint에서 발행 없음. S-1+α의 swap_executor 6건에서 사용 예정.

---

## 즉시 재현 명령

### 호스트 CPU 32 토큰
```bash
./target/release/argus_cli \
  --model-path /home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --tokenizer-path /home/go/Workspace/llm_rs2/models/qwen2.5-1.5b/tokenizer.json \
  --prompt "The capital of France is" --num-tokens 32 --greedy --backend cpu --kv-type f16 \
  --no-resilience
# 기대: "Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into"
# first=12095, final_pos=36, Decode 112 ms/tok
```

### S25 Adreno OpenCL 32 토큰
```bash
python3 scripts/run_device.py -d galaxy_s25 argus_cli \
  --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json \
  --prompt-file /data/local/tmp/prompts/capital.txt \
  --num-tokens 32 --greedy --backend opencl --kv-type f16 --no-resilience
# 기대: 동일 출력, TBT ~30 ms/tok
```

### emit 발행 확인 (swap flag 켜기)
```bash
./target/release/legacy_generate \
  --model-path <PATH> --tokenizer-path <PATH> \
  --prompt "X" --num-tokens 64 --greedy --backend opencl --kv-type f16 \
  --force-swap-ratio 0.5 --swap-incremental-per-tick 2 2>&1 | grep '\[WeightSwap\]'
# 기대: [WeightSwap] ChunkDrained/PlanRetired 메시지 다수 출력
```

---

## 진입 명령 (다음 세션)

```
"S-1+α 진행"        # sub-module 13건 emit 처리
"swap default 진행"  # backlog P3
"PR 생성"            # gh CLI 필요
```
