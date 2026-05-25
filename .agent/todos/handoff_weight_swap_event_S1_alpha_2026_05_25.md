# Handoff: S-1+α — WeightSwapKind enum + sub-module emit 14건 완료

**작성**: 2026-05-25
**HEAD**: `8f476c4c refactor(observability): S-1+α — WeightSwapKind enum + BatchSummary + sub-module emit 14건`
**브랜치**: `worktree-b5_trait_extension`
**다음 세션 진입 문장**: "swap default 진행" (backlog P3) 또는 "PR 생성" 또는 "S-1+β 진행" (async_swap L256 + B/C 카테고리)

---

## TL;DR

S-1 후속 sprint. 4 variant kind 필드 `&'static str → WeightSwapKind` enum 전환 +
`BatchSummary` 5번째 variant 신설 + sub-module 14건 eprintln → emit 치환
(intra_forward 2 + phase_aware 6 + swap_executor 6). 호스트 CPU + S25 Adreno
OpenCL bit-identical 회귀 0. `async_swap.rs:256` 1건은 dispatcher 생성자
ripple ROI 차로 별 sprint 보류 (backlog P3). plan은 대화 상의 결정 사항
(Q1 enum / Q2 BatchSummary / Q3 builder default IntraForward) 기반.

---

## 진행 상태

| Step | 작업 | 결과 |
|---|---|---|
| α.1 | `WeightSwapKind` enum (4 variant, default IntraForward, Display, as_str) | events.rs |
| α.2 | 4 기존 variant kind 필드 enum 전환 + tests 갱신 | events.rs |
| α.3 | `BatchSummary { kind, mode, target_layers, max_release_pending, max_dispatcher_pending }` variant 신설 + test | events.rs |
| α.4 | swap_dispatch.rs 10건 string → enum 전환 | swap_dispatch.rs |
| α.5 | `SwapExecutor` kind/event_sink 필드 + `.with_kind` / `.with_event_sink` builder | swap_executor.rs |
| α.6 | IntraForwardSwapHook 2건 emit + 생성자 event_sink 인자 | intra_forward_swap.rs + generate.rs L1436 caller |
| α.7 | PhaseAwareSwapDispatcher 6건 emit + 생성자 event_sink 인자 | phase_aware_swap.rs + generate.rs L1372 caller |
| α.8 | SwapExecutor 6건 emit (5 SwapFailed + 1 BatchSummary) + caller chain .with_kind/.with_event_sink | swap_executor.rs |
| α.9 | 호스트 CPU + S25 Adreno OpenCL 32 토큰 bit-identical | both PASS |
| α.10 | commit + handoff + push + notify | 본 문서 |

### 측정 게이트 (검증 가능 형태)

- `cargo test -p llm_rs2 --lib events`: **13/13 PASS** (S-1 기존 10 + 신규 3 — BatchSummary emit + WeightSwapKind default + clone).
- `cargo test -p llm_rs2 --test spec inv_layer`: **8/8 PASS** (V-31 2건 baseline 등록 — multi-item `use {}` 라인 parser block-form 오해 우회).
- `cargo clippy -p llm_rs2 --lib -- -D warnings`: clean (Default derive로 derivable_impls 해소).
- 호스트 CPU 32 토큰 bit-identical: `first=12095, final_pos=36, Decode 111.93 ms/tok` (v1-1 baseline 112.02 noise 이내).
- S25 Adreno OpenCL 32 토큰 bit-identical: `first=12095, final_pos=36, Decode 31.07 ms/tok, Avg TBT 33.05 ms` (v1-1 baseline 33.01 -5.9% / S-1 30.67 noise 이내).

### 핵심 변경

`engine/src/observability/events.rs`:
- `WeightSwapKind` enum: 4 variant (`Incremental` / `IntraForward` / `PhaseAware` / `Subsystem`), `Default = IntraForward` (Q1 결정), `Display` + `as_str()` (S-1 stderr grep 호환).
- 기존 4 variant의 `kind: &'static str` → `kind: WeightSwapKind`.
- `BatchSummary { kind, mode, target_layers, max_release_pending, max_dispatcher_pending }` 5번째 variant (Q2 결정).
- StderrSink match arm 5개 (kind는 Display 포맷).
- 신규 tests 2: `test_batch_summary_emit_and_collect`, `test_weight_swap_kind_default_is_intra_forward`.

`engine/src/models/weights/swap_executor.rs`:
- 23 → 25 필드 (`kind: WeightSwapKind` + `event_sink: Arc<dyn EventSink>`).
- `.with_kind()` + `.with_event_sink()` builder (Q3 결정).
- BgFetch 3건 + AsyncSwap 2건 SwapFailed emit (kind = caller-injected, default IntraForward).
- L1342 `[SwapPeak]` → `BatchSummary` emit.
- LAYER-EXEMPT `cross_cutting_trait_usage` 마커.

`engine/src/models/weights/intra_forward_swap.rs`:
- `event_sink: Arc<dyn EventSink>` field 추가.
- `new(...)` 인자 +1 (production 생성자); `new_for_test`는 NoOpSink default.
- 2건 emit (build/dispatcher submit failed, kind=IntraForward).
- 내부 `SwapExecutor::new(...)` caller에 `.with_event_sink(Arc::clone(&self.event_sink))` chain.

`engine/src/models/weights/phase_aware_swap.rs`:
- `event_sink: Arc<dyn EventSink>` field 추가.
- `new(...)` 인자 +1.
- 6건 emit: wait_pending failed (L309) + build failed x2 (L399/L415) + assemble failed (L488) + dispatcher submit failed (L510) + finalize drain (L552). 모두 kind=PhaseAware.
- 내부 `SwapExecutor::new(...)` caller에 `.with_kind(WeightSwapKind::PhaseAware).with_event_sink(Arc::clone(&self.event_sink))` chain.

`engine/src/session/decode_fallback/swap_dispatch.rs`:
- 10건 `kind: "..."` 문자열 → `WeightSwapKind::*` enum.
- import 갱신.

`engine/legacy/generate.rs`:
- L1372 PhaseAwareSwapDispatcher::new + L1436 IntraForwardSwapHook::new 두 caller에 `Arc::clone(&event_sink)` 전달 (S-1 L1093 단일 Arc 재사용).

`engine/tests/spec/inv_layer_baseline.json`:
- V-31 2건 추가: swap_executor.rs:63 + phase_aware_swap.rs:35.
- 사유: multi-item `use crate::observability::events::{A, B, C};` 라인 직후의
  `// LAYER-EXEMPT:` 마커가 parser의 block-form 검사(`{` 존재 시)를 트립.
  의도 documentation은 §13.8-N 마커가 보존, 기술 등록은 baseline.

---

## 다음 작업 후보 (우선순위 순)

### A. swap 기본 모드 변경 (backlog P3) — 가장 자연스러운 후속

S-1+α 완료로 IntraForward path event 발행 정착 → backlog P3 진입 안전.
3 옵션 중 사용자 결정 필요:
- (a) Manager-driven path 변경 (`executor.rs:549~`) — blast radius 큼
- (b) `--swap` shorthand 신규 flag — 1~2h
- (c) `--enable-swap` shorthand + IntraForward fallback — 0.5~1일

### B. PR 생성

master 대비 14 commits 미머지 (S-1 13 + S-1+α 1). gh CLI auth + 분할 PR 권장:
- Manager IPC wiring 4 commits (이전 sprint)
- S-1 1 commit + handoff 1
- S-1+α 1 commit + handoff 1 (본)

### C. S-1+β — 잔여 emit 정리 (별 sprint)

- `async_swap.rs:256` wait_event_blocking failed — AsyncSwapDispatcher 생성자에 event_sink 인자 + caller 3곳 갱신 (intra_forward + phase_aware + swap_dispatch). 단독 0.5~1일.
- B/C 카테고리 (config-parse / startup warning류) — `phase_aware:106/372`, `swap_executor:38/751/2011/2066/2108/2150` 등. 별 sprint, ROI 차.

---

## Landmines / 미해결

### 1. `async_swap.rs:256` emit 미적용

`[AsyncSwap] wait_event_blocking failed: {e}; commit skipped` 라인은 stderr만 출력.
EventSink 기반 alert 구독 시 미커버. AsyncSwapDispatcher 생성자에 event_sink 인자
추가하려면 SwapCommitJob worker thread가 Arc clone hold 필요 + 3 caller 갱신.
backlog P3 등록 (S-1+α handoff R6 §1).

### 2. Manager-driven WeightSwapHandler executor.execute_on_slots 시 kind 디폴트

`engine/src/pressure/weight_swap_handler.rs:98` SwapExecutor::new caller는
`.with_kind()` chain 안 함 → kind=IntraForward fallback. Manager-driven swap의
SwapFailed emit이 IntraForward로 attribute됨 (실제로는 manager 트리거). 별
sprint에서 `.with_kind(Subsystem)` 또는 신규 `WeightSwapKind::Manager` variant
검토.

### 3. swap_executor의 SwapExecutor::new caller 미커버

`engine/src/session/qcf_runtime.rs:72` (run_layer_swap sync path) 도 event_sink
inject 안 함 → 동일 NoOp 디폴트. sync path의 SwapFailed는 stderr 누락 (단,
sync path는 swap-on 시 거의 안 쓰임 — async가 디폴트).

### 4. Multi-item `use {...};` 라인 parser 한계

`scripts/layer_lint.py`의 `_find_exempt_zone_ranges`가 marker 직전 라인에 `{`가
있으면 block-form으로 잘못 분류. multi-item use는 `{`를 포함하므로 마커가 의도대로
안 동작. 회피책: baseline 등록 (§13.8-N 의도는 마커 주석으로 보존).
근본 fix는 parser에 `prev_has_brace` 체크에서 `use ... { ... };` 패턴 제외 추가.

### 5. swap-on grep 검증 누락

`--force-swap-ratio` flag는 `--secondary-gguf` 필요 (호스트에 없음). 따라서 호스트
sanity는 emit 발화 0 (no-swap path), unit test (`test_batch_summary_emit_and_collect`
등)가 emit 코드 path 커버. swap-on production 검증은 secondary GGUF가 있는 device
sprint (예: S25 weight swap microbench)에서.

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
# first=12095, final_pos=36, Decode 111.93 ms/tok
```

### S25 Adreno OpenCL 32 토큰
```bash
python3 scripts/run_device.py -d galaxy_s25 argus_cli \
  --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
  --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json \
  --prompt-file /data/local/tmp/prompts/capital.txt \
  --num-tokens 32 --greedy --backend opencl --kv-type f16 --no-resilience
# 기대: 동일 출력, Decode ~31 ms/tok, Avg TBT ~33 ms
```

### emit 발행 확인 (secondary GGUF 있는 환경)
```bash
LLMRS_SWAP_DRAIN_DIAG=1 ./target/release/legacy_generate \
  --model-path <F16 GGUF> --tokenizer-path <PATH> --secondary-gguf <Q4_0 GGUF> \
  --prompt "X" --num-tokens 64 --greedy --backend opencl --kv-type f16 \
  --force-swap-ratio 0.5 --swap-incremental-per-tick 2 2>&1 | grep '\[WeightSwap\]'
# 기대: [WeightSwap] BatchSummary kind=IntraForward 등 + 실패 시 SwapFailed
```

---

## 진입 명령 (다음 세션)

```
"swap default 진행"   # backlog P3 (옵션 a/b/c 결정 후 1~2h or 0.5~1일)
"PR 생성"            # gh CLI 필요, 분할 PR 권장
"S-1+β 진행"         # async_swap L256 + B/C 카테고리 정리
```
