# Handoff — Phase 4-4-2 옵션 E1' 진입점 (2026-05-19)

**작성**: 2026-05-19 (Sprint 1 / Sprint 2 종결 + Phase 4-4-2 E1' 결정 직후)
**master HEAD**: `65ade7ea`
**진입 문장**: "Phase 4-4-2.1 진행" (prefill 추출부터)

---

## 1. 직전 sprint 종결 요약

### Sprint 1 — Phase 4-4 외과적 추출

| Commit | 추출 | LOC |
|---|---|---|
| `fcc1ea87` | dump-importance → `session::dump_importance` | -73 |
| `15fc0fee` | standard happy path → `session::standard_happy` | -78 |
| `1b5b93b6` | DVFS warmup → `session::warmup` | -77 |

generate.rs: 6,513 → 6,318 LOC (-3%)

### Sprint 2 — QCF rename

| Commit | 작업 |
|---|---|
| `65ade7ea` | `unified_qcf` → `qcf_kv` 패밀리 + `compute_qcf_swap` → `compute_qcf_weight_swap` + INV-LAYER baseline 동기화 + backlog [P2] RESOLVED |

---

## 2. Phase 4-4-2 옵션 평가 결과 (2026-05-19 결정)

원안 **옵션 E** (단일 sprint cut/paste 3,058 LOC + ctx 70+ 필드) 검토 후,
ctx 필드 70+ 개와 fallback 진입 직전 mut state 다수(`incremental_force_swap_plan`,
`manager_swap_report_pending`, `ready_weight_swap_report`, `tbt_log_writer` 등)로
인해 단일 sprint 위험이 과대.

**최종 채택**: **옵션 E1'** — 4 sub-sprint 분할.

### 옵션 E1' 분해

| Sub-sprint | 영역 | LOC | 모듈 |
|---|---|---|---|
| **4-4-2.1** | Prefill block (L1785~2375) | ~590 | `session::prefill::run_chunked_prefill` |
| 4-4-2.2 | Transition block (L2377~2570) | ~190 | `session::transition::run_prefill_to_decode_transition` (또는 기존 모듈에 흡수) |
| 4-4-2.3 | Decode main loop (L2570~4640) | ~2,070 | `session::decode_fallback::run_decode_fallback` |
| 4-4-2.4 | Post-process (L4640~4842) | ~270 | `session::post_process::run_fallback_post_process` |

각 sub-sprint = 단일 worktree + bit-identical 32 tok + avg_tbt n=5 ≤5% gate + ff-merge.

각 단계가 **즉시 commit/ff-merge** 후 다음 단계 진입 → main HEAD 항상 안정.

---

## 3. Sub-sprint 1: Phase 4-4-2.1 — Prefill block 추출

### 3.1 목표

`bin/generate.rs::main()` L1792~2375 (약 590 LOC) — fallback 진입 직후 prefill block을
`session::prefill::run_chunked_prefill(...)` 단일 함수로 외과적 이식.

### 3.2 영역 식별 (master `65ade7ea` 기준)

| 시작 | 끝 | 책임 |
|---|---|---|
| L1792 | L1809 | Inference profiler 생성 (`InferenceProfiler`) |
| L1811~1822 | | logits buffer pre-alloc, EOS id |
| L1822~1904 | | **이미 추출됨**: `session::warmup::run_warmup` 호출 1줄 |
| L1834~1855 | | `variance_collector` (D2O layer-alloc) 생성 |
| L1856~1864 | | weight swap state (`importance_table_for_swap`, `collector_armed`, `deferred_switch`) 초기화 |
| L1866~2370 | | **chunked prefill 본체** — auto-chunk size, logits buffer, score/skip/importance/variance collector 주입, ENG-ALG-218 last-chunk collector arm, `forward_into`, last_logits read, sampling, `tokens.push` |
| L2375 | | prefill 끝 (start_pos 증가) |

**진입 후 반환할 state** (decode/transition으로 전달):
- `profiler` (mut)
- `variance_collector` (mut)
- `importance_table_for_swap` / `collector_armed` / `deferred_switch` (mut)
- `tokens` (mut, push 후 prompt+1)
- `start_pos` (mut)
- `_ttft_ms` (mut)
- `_last_token_time` (mut)
- `kv_caches` (mut, prefill 후 KV 채워짐)
- `logits` / `logits_buf` (decode 재사용)
- prefill_forward_ms, prefill_pure_fwd_ms 등 timing

### 3.3 권장 시그니처 — 인자 통째 ctx 패키징 vs 함수 args

옵션 A: `PrefillCtx` struct 35+ 필드
옵션 B: 인자 35+ 개 함수 시그니처

**권장 A** — Sprint 1 패턴 (`DumpImportanceCtx`, `StandardHappyCtx`, etc.) 일관성.

```rust
pub struct PrefillCtx {
    pub args: Args,
    pub backend: Arc<dyn Backend>,
    pub memory: Arc<dyn Memory>,
    pub model: TransformerModel,
    pub kv_caches: Vec<KVCache>,
    pub tokens: Vec<u32>,
    pub sampling_config: SamplingConfig,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub start_pos: usize,
    pub score_accumulator: Option<AttentionScoreAccumulator>,
    pub skip_config: Option<SkipConfig>,
    pub start_time: std::time::Instant,
    pub qcf_warmup_importance: Option<crate::core::qcf::ImportanceTable>,
    pub qcf_swap_decision: Option<crate::models::weights::decider::SwapDecision>,
    pub command_executor: Option<CommandExecutor>,
    pub last_skip_ratio: Option<f32>,
    pub auto_eviction: bool,
    // ... +20개
}

pub struct PrefillOutput {
    pub kv_caches: Vec<KVCache>,
    pub tokens: Vec<u32>,
    pub start_pos: usize,
    pub last_logits: Vec<f32>,
    pub profiler: Option<InferenceProfiler>,
    pub variance_collector: Option<D2OVarianceCollector>,
    pub importance_table_for_swap: Option<ImportanceTable>,
    pub collector_armed: bool,
    pub deferred_switch: Option<String>,
    pub ttft_ms: f64,
    pub last_token_time: std::time::Instant,
    pub score_accumulator: Option<AttentionScoreAccumulator>,
    pub command_executor: Option<CommandExecutor>,
    pub last_skip_ratio: Option<f32>,
    // ... +5
}

pub fn run_chunked_prefill(ctx: PrefillCtx) -> anyhow::Result<PrefillOutput>;
```

### 3.4 단계별 절차

1. `EnterWorktree phase4_4_2_1_prefill`
2. `git rebase master`
3. `engine/src/session/prefill/` 모듈 신규 (mod.rs + args.rs + runner.rs)
   - `args.rs`: PrefillCtx + PrefillOutput
   - `runner.rs`: run_chunked_prefill(ctx) — main()의 prefill block 1:1 cut/paste
4. `engine/src/session/mod.rs`에 `pub mod prefill;` 추가
5. `bin/generate.rs::main()` 의 fallback 영역에서 prefill block을 dispatcher로 치환:
   ```rust
   let prefill_out = llm_rs2::session::prefill::run_chunked_prefill(PrefillCtx {
       args: args.clone(),
       backend: backend.clone(),
       // ... ctx fields
   })?;
   let mut kv_caches = prefill_out.kv_caches;
   let mut tokens = prefill_out.tokens;
   let mut start_pos = prefill_out.start_pos;
   let mut profiler = prefill_out.profiler;
   // ... unwrap
   ```
6. 빌드 PASS
7. `cargo test --lib session::` PASS (52 + α)
8. `cargo test --test spec` 회귀 0 (사전 device-required 3건 외)
9. **S25 디바이스 게이트**:
   - bit-identical 32 tok output
   - avg_tbt n=5 회귀 ≤5%
10. (가능 시) Jetson CUDA 동일 게이트
11. commit + ExitWorktree + ff-merge + branch delete + notify-send

### 3.5 위험 / 완화

| R | 항목 | 완화 |
|---|---|---|
| R1 | ctx 35+ 필드 + Output 15+ 필드 ownership transfer | `PrefillCtx { ... } = ctx;` destructure, prefill 끝에 PrefillOutput 빌드 |
| R2 | chunked prefill 안의 ENG-ALG-218 last-chunk collector arm 로직 변경 시 정확성 회귀 | 1:1 cut/paste, 로직 변경 0 |
| R3 | `deferred_switch` 등 mid-prefill checkpoint mutation | ctx 진입 시 None, output에서 final value 반환 |
| R4 | `backend`/`is_gpu` mid-prefill switch (deferred_switch는 transition에서 처리이므로 prefill은 영향 없음) | prefill 종료 시점에서는 backend 변경 0, transition으로 위임 |
| R5 | 빌드 에러 type mismatch | 컴파일러 에러로 type 식별 후 ctx 필드 수정 반복 |

### 3.6 예상 결과

- generate.rs: 6,318 → 5,800 LOC (~ -520, prefill 590 → dispatcher 70)
- 신규 `session/prefill/` 모듈: ~620 LOC
- prefill block 격리: 후속 sub-sprint (decode/transition/post)가 더 작아짐

---

## 4. Sub-sprint 2~4 (Phase 4-4-2.2 ~ 4-4-2.4) 개요

각 sub-sprint는 4-4-2.1 완료 후 동일 패턴으로 진행. PrefillOutput → TransitionCtx → DecodeFallbackCtx → PostProcessCtx 체인.

### Phase 4-4-2.2 — Transition (L2377~2570)

- 책임: deferred SwitchHw 실행 (kv_migrate GPU↔CPU), D2O per-layer budget, position_birth_step 초기화
- 신규 모듈: `session::transition` (또는 `session::decode_fallback`에 prologue로 흡수)
- 진입 문장: "Phase 4-4-2.2 진행"

### Phase 4-4-2.3 — Decode main loop (L2570~4640)

- 책임: 토큰 단위 forward + score eviction + swap dispatcher + experiment writer + dynamic throttle
- 신규 모듈: `session::decode_fallback`
- **가장 크고 위험. 가능하면 더 분해 (eviction trigger / swap dispatch / experiment writer 등)**
- 진입 문장: "Phase 4-4-2.3 진행"

### Phase 4-4-2.4 — Post-process (L4640~4842)

- 책임: token decode + final stats + experiment summary + qcf-dump JSON
- 신규 모듈: `session::post_process`
- 진입 문장: "Phase 4-4-2.4 진행"

---

## 5. 환경 / 규칙 (재확인)

- **언어**: 모든 응답 한국어, 기술 용어/코드 식별자 원문 유지
- **EnterWorktree**: 코드 변경 작업 시 worktree 격리 필수
- **테스트 기본 모델 포맷**: GGUF
- **Android 벤치 스레드**: Galaxy S25는 6T만
- **TBT metric**: 항상 avg_tbt (tok0 inclusive). rest_tbt 단독 비교 금지
- **성능 측정**: `--profile` 없이
- **신규 spec test**: `engine/tests/spec/`
- **완료 시 자동 commit + `notify-send`**
- **`.cl` 커널 수정**: Senior Implementer만, Adreno 실측 교훈 준수
- **Background job 임시 파일**: `$CLAUDE_JOB_DIR`

---

## 6. 미해결 (낮은 우선) — 별 sprint 필요

- **DegradationEstimator action key prefix 정규화** (kv./weight.) — Sprint 2 deferred
- **shared::QcfEstimate.estimates HashMap key prefix** (IPC, manager 동시 변경) — Sprint 2 deferred
- **[P0] WSWAP-6-A** Fused SOA convert kernel
- **[P0] WSWAP-6-C** Primary cl_mem release를 critical path에서 제거
- **[P0] M3.4 RED** pos baked architectural blocker
- **[P0] M2.A~M2.G** QNN OpPackage layer-level op wrap
- **[P0] Weight Swap Mixed Precision**
- **[P0] Long context CPU attention** llama.cpp 대비 35% 수준
- **[P1] Qwen2 default Instruct 교체**
- **[P1] chat_template Gemma3 구현**
- **[P2] clippy 회귀 cleanup**

---

## 7. 진입 명령 요약

| 순서 | 문장 | 작업 |
|---|---|---|
| 1 | **"Phase 4-4-2.1 진행"** | Prefill block 추출 (L1792~2375, ~590 LOC) |
| 2 | **"Phase 4-4-2.2 진행"** | Transition block 추출 (L2377~2570, ~190 LOC) |
| 3 | **"Phase 4-4-2.3 진행"** | Decode main loop 추출 (L2570~4640, ~2,070 LOC) — 추가 분해 권장 |
| 4 | **"Phase 4-4-2.4 진행"** | Post-process 추출 (L4640~4842, ~270 LOC) |

각 sprint = 단일 worktree + bit-identical 32 tok + avg_tbt n=5 ≤5% gate + ff-merge.

---

## 8. 참조 문서

- `arch/inference_pipeline.md` — Phase 4 6 trait 설계
- `ARCHITECTURE.md` §13 — Layered architecture INV-LAYER 정의
- `docs/qcf_taxonomy.md` — QCF kv/weight 분리 사양 (Sprint 2 갱신 완료)
- `.agent/todos/handoff_phase4_4_qcf_rename_entry_2026_05_19.md` — Sprint 1/2 진입 (완료)
- `.agent/todos/backlog.md` — 미해결 P0/P1 항목
