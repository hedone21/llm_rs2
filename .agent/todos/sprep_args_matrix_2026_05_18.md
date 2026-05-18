# S-prep — `cli.rs::Args` 사용 매트릭스 + Binary 분리 그룹화 (2026-05-18)

**Base HEAD**: `30320225` (S-prep 초안 commit, 본 doc 갱신은 그 위에 누적)
**선행 handoff**: `handoff_phase4_DABC_complete_2026_05_18.md`
**다음 sprint**: S-cleanup (불필요 field 삭제 + 측정 전용 field binary 분리) → S-1 (BaseArgs 추출)

---

## 갱신 이력

| 일자 | 갱신 사항 |
|---|---|
| 2026-05-18 (초안) | 17 sub-struct 그룹화, 130 field 매트릭스 |
| 2026-05-18 (정정) | grep 정규식 누락 12개 추가 → **142 field 확정**, 사용자 분류 동의 → 사양 확정, S-cleanup 진입점 추가 |

## 목적

Phase 4 종결 후 `bin/generate.rs::main()` dispatch가 5분기 + legacy로 정착됨. 다음 sprint(S-1~S-4)에서 chat/ppl/batch/eval-ll 별도 binary를 만들기 전, **`cli.rs::Args` 142 field가 어느 분기에서 실제 참조되는지** 데이터로 확정한다. 이 매트릭스가 BaseArgs 추출 라인 + sub-struct 그룹화의 근거다.

---

## 집계 방법

- 진입점 디렉토리/파일별로 `args\.<field>` 패턴 grep (`self.args.` / `ctx.args.` 정규화 후).
- 7개 source bucket:
  | bucket | 위치 | 역할 |
  |---|---|---|
  | `init` | `session/init.rs` | SessionInitCtx 빌더 — **모든 분기에서 호출되는 공통** |
  | `asm` | `session/assembly/` | standard happy-path (DecodeLoop+ModelForward) |
  | `chat` | `session/chat/` | chat REPL (Phase 4-5 산출) |
  | `ppl` | `session/ppl/` | PPL + run_kivi_ppl (Phase 4-C) |
  | `batch` | `session/batch/` | prompt batch (Phase 4-A) |
  | `eval` | `session/eval/` | eval-ll (Phase 4-B) |
  | `gen_bin` | `bin/generate.rs` | main dispatch + legacy fallback |
  | `qcf` | `session/qcf_runtime.rs` | shared swap/QCF helpers |

## 분기별 ref 카운트 (v2, 142 field 정정 후)

| bucket | unique field refs |
|---|---|
| init | 54 |
| assembly | 11 |
| chat | **42** (h2o/d2o 5+ 추가 발견) |
| ppl | **27** |
| batch | 18 |
| eval | **23** |
| generate_bin (legacy + dispatch) | **104** |
| qcf_runtime | **0** ✓ |

`qcf_runtime`이 args 의존성 0이라 lib boundary로 깨끗 (Phase 4-B-1/4-C-1 산출물).

---

## 핵심 관찰 (그룹화 근거)

### O1. `init.rs`가 진짜 공통 진입점

54 unique field가 모든 분기에서 호출되는 init.rs에 응집. 이건 그대로 **BaseArgs 후보의 1차 필터** 역할.

### O2. KIVI/Offload는 init level에서 KVCache 생성 후 분기는 unaware

`args.kivi` / `args.kv_offload` / `args.offload_path` / `args.kivi_bits`는 **chat/ppl/batch/eval lib 어디서도 직접 참조 안 함**. init.rs가 `KVCache::Kivi/Offload` enum variant로 빌드해서 `kv_caches: Vec<KVCache>` 형태로 분기에 넘김.

→ `KiviArgs` / `OffloadArgs` sub-struct를 만들되 **BaseArgs 안에 flatten**하면 분기별 binary는 args 추가 노출 없이 동작. R4 (KIVI가 chat/ppl 양쪽 사용) 자동 해결.

### O3. Swap은 init + ppl + eval 위주

| bucket | swap-related refs |
|---|---|
| chat | 0 |
| ppl | 3 |
| batch | 0 |
| eval | 6 |
| assembly | 4 |
| init | 18 |

chat/batch는 swap 의존성 0. **`SwapArgs`는 PPL + Eval binary만 flatten**, chat/batch는 노출하지 않음. assembly(4)는 standard happy-path에서 force_swap_ratio 1회 처리.

### O4. Eviction params는 모든 분기에서 직접 참조 빈번

| bucket | eviction-related refs |
|---|---|
| chat | 29 |
| ppl | 16 |
| batch | 6 |
| eval | 15 |
| assembly | 6 |
| init | 1 |

→ **`EvictionArgs` sub-struct는 BaseArgs 안에 flatten** (4개 binary 모두 사용). init은 1회만 보지만 분기별 eviction 처리는 lib 모듈에 있음.

### O5. QCF/ARGUS는 ppl/batch/eval 3분기 cross-cut

`qcf_dump` / `qcf_mode` / `qcf_sample_layers` / `qcf_warmup_tokens` / `enable_qcf_experimental` 등이 chat 제외 3분기에 등장. chat은 ARGUS 측정 대상 아님.

→ **`QcfArgs` sub-struct는 ppl/batch/eval만 flatten**.

### O6. legacy generate가 가장 큰 superset (96 fields)

main()의 dispatch는 5분기 + legacy fallback 4,700 LOC. legacy가 weight swap 일반 / KIVI 일반 / experiment 분기를 모두 들고 있음. legacy retain 결정에 따라 `generate` binary는 모든 sub-struct를 flatten한 통합 args 유지.

---

## Sub-struct 그룹화 제안

| Sub-struct | Field 수 | Flatten 대상 binary | 비고 |
|---|---|---|---|
| **`BaseArgs`** | ~30 | generate, chat, ppl, batch, eval-ll | 모델 로딩 + 백엔드 + sampling + 공통 KV |
| **`SamplingArgs`** | 8 | (BaseArgs 안에 flatten) | temperature/top_p/top_k/greedy/repetition_* |
| **`ProfileArgs`** | 11 | (BaseArgs 안에 flatten) | profile/profile_events/heartbeat_gpu_profile/profile_dir/profile_interval/profile_per_head/profile_probes/cuda_profile/tbt_log/target_tbt/throttle_delay_ms |
| **`KiviArgs`** | 4 | (BaseArgs 안에 flatten — init.rs 전용) | kivi/kivi_bits/kivi_residual_size/kv_dynamic_quant (awqe는 env `LLMRS_KIVI_AWQE`로 이전) |
| **`OffloadArgs`** | 3 | (BaseArgs 안에 flatten — init.rs 전용) | kv_offload/offload_path/max_prefetch_depth |
| **`EvictionArgs`** | 14 | (BaseArgs 안에 flatten) | eviction_*/h2o_*/d2o_*/sink_size/streaming_window/kv_budget/kv_budget_ratio/protected_prefix/min_kv_cache/initial_kv_capacity/memory_threshold_mb/skip_layers/skip_ratio |
| **`PrefillArgs`** | 5 | (BaseArgs 안에 flatten — init.rs 전용) | prefill_chunk_size/prefill_yield_ms/prefill_cpu_chunk_size/no_prefill_ws/no_gpu_plan |
| **`SwapArgs`** | 24 | ppl, eval-ll, generate | secondary_*/force_swap_ratio/swap_*/quantize_lm_head/eager_prefault_secondary/swap_dir |
| **`QcfArgs`** | 11 | ppl, batch, eval-ll, generate | qcf_dump/qcf_mode/qcf_sample_layers/qcf_warmup_tokens/qcf_trajectory/enable_qcf_experimental/decode_x_steps/importance_formula/**dump_importance**(ProfileArgs에서 이동, 2026-05-18) |
| **`ResilienceArgs`** | 3 | (BaseArgs 안에 flatten) | enable_resilience/resilience_prealloc_switch/resilience_transport |
| **`ExperimentArgs`** | 5 | (BaseArgs 안에 flatten — legacy 전용) | experiment_*/ignore_eos |
| **`ChatArgs`** | 4 | chat, generate | chat/system_prompt/chat_socket/chat_tcp |
| **`PplArgs`** | 10 | ppl, generate | ppl/ppl_*/dump_q4_* |
| **`BatchArgs`** | 3 | batch, generate | prompt_batch/prompt_batch_loop/max_iterations |
| **`EvalLlArgs`** | 3 | eval-ll, generate | eval_ll/eval_continuation/eval_batch |
| **`TensorPartitionArgs`** | 1 | (BaseArgs 안에 flatten) | tensor_partition |
| **`BackendArgs`** | 9 | (BaseArgs 안에 flatten) | backend/no_zero_copy/threads/cuda_sync_policy/cuda_weights_device/cuda_graph/qnn_graph_cache_prebuild/qnn_allow_fallback/gpu_priority |

**총합**: 약 142 field → 17개 sub-struct로 그룹화 (`BaseArgs` 자체가 9 sub-struct flatten 컨테이너).

**2026-05-18 확정**: 사용자 review 후 분류 동의. S-cleanup에서 측정용 / 미사용 field 삭제 또는 binary 분리 진행 (아래 §S-cleanup).

---

## Binary 정의 (구체화)

```rust
// session/cli/base.rs
#[derive(clap::Parser, Clone)]
pub struct BaseArgs {
    #[clap(flatten)] pub model: ModelLoadArgs,         // model_path/tokenizer_path/weight_dtype/quantize_lm_head
    #[clap(flatten)] pub backend: BackendArgs,         // backend/zero_copy/switch_threshold/threads/cuda_*/qnn_*/gpu_*/use_rayon
    #[clap(flatten)] pub sampling: SamplingArgs,
    #[clap(flatten)] pub profile: ProfileArgs,
    #[clap(flatten)] pub kivi: KiviArgs,
    #[clap(flatten)] pub offload: OffloadArgs,
    #[clap(flatten)] pub eviction: EvictionArgs,
    #[clap(flatten)] pub prefill: PrefillArgs,
    #[clap(flatten)] pub resilience: ResilienceArgs,
    #[clap(flatten)] pub partition: TensorPartitionArgs,
    pub max_seq_len: usize,
    pub num_tokens: usize,
    pub prompt: String,
    pub prompt_file: Option<String>,
}

// bin/chat.rs
#[derive(clap::Parser)]
struct ChatBinArgs { #[clap(flatten)] base: BaseArgs, #[clap(flatten)] chat: ChatArgs, }

// bin/ppl.rs
#[derive(clap::Parser)]
struct PplBinArgs { #[clap(flatten)] base: BaseArgs, #[clap(flatten)] ppl: PplArgs, #[clap(flatten)] swap: SwapArgs, #[clap(flatten)] qcf: QcfArgs, }

// bin/batch.rs
#[derive(clap::Parser)]
struct BatchBinArgs { #[clap(flatten)] base: BaseArgs, #[clap(flatten)] batch: BatchArgs, #[clap(flatten)] qcf: QcfArgs, }

// bin/eval_ll.rs
#[derive(clap::Parser)]
struct EvalBinArgs { #[clap(flatten)] base: BaseArgs, #[clap(flatten)] eval: EvalLlArgs, #[clap(flatten)] swap: SwapArgs, #[clap(flatten)] qcf: QcfArgs, }

// bin/generate.rs (legacy retain — 모든 sub-struct flatten)
#[derive(clap::Parser)]
struct GenerateArgs {
    #[clap(flatten)] base: BaseArgs,
    #[clap(flatten)] chat: ChatArgs,
    #[clap(flatten)] ppl: PplArgs,
    #[clap(flatten)] batch: BatchArgs,
    #[clap(flatten)] eval: EvalLlArgs,
    #[clap(flatten)] swap: SwapArgs,
    #[clap(flatten)] qcf: QcfArgs,
    #[clap(flatten)] experiment: ExperimentArgs,
}
```

신규 binary CLI 표면:
- `chat`: BaseArgs(~30) + ChatArgs(4) = **~34 flag**
- `ppl`: BaseArgs(~30) + PplArgs(10) + SwapArgs(24) + QcfArgs(13) = **~77 flag**
- `batch`: BaseArgs(~30) + BatchArgs(3) + QcfArgs(13) = **~46 flag**
- `eval-ll`: BaseArgs(~30) + EvalLlArgs(3) + SwapArgs(24) + QcfArgs(13) = **~70 flag**
- `generate` (legacy): 130 (변화 없음, 사용자 영향 0)

가장 큰 chat 감축률: 130 → 34 (**−74%**).

---

## RunCtx 매핑 변경 (S-1 작업)

현재 패턴 (예: `PplRunCtx`):
```rust
pub struct PplRunCtx {
    pub args: Args,  // 250+ field 통째로
    pub backend: Arc<dyn Backend>,
    ...
}
```

S-1 후:
```rust
pub struct PplRunCtx {
    pub base: BaseArgs,
    pub ppl: PplArgs,
    pub swap: SwapArgs,
    pub qcf: QcfArgs,
    pub backend: Arc<dyn Backend>,
    ...
}
```

`run_ppl_dispatch(ctx)` 호출자(bin/ppl.rs)는 `PplBinArgs`를 destructure해서 `PplRunCtx` 생성.

---

## 진입 명령 — S-1

다음 session에서 `"S-1 진행"`으로 시작. S-1 분해:

| Step | 작업 | 산출물 |
|---|---|---|
| S-1a | `session/cli/` 디렉토리 신설 + 17 sub-struct를 module 분할 (base/model/backend/sampling/profile/kivi/offload/eviction/prefill/swap/qcf/resilience/partition/experiment/chat/ppl/batch/eval) | `session/cli/{base,model,...}.rs` + `mod.rs` |
| S-1b | `BaseArgs` flatten 컨테이너 정의 + `session/cli.rs::Args` 호환 wrapper 유지 (legacy) | bin/generate.rs 변경 0건 |
| S-1c | `SessionInitCtx`가 `Args` 대신 `&BaseArgs` 받도록 시그니처 변경 + init.rs 내부 ref 갱신 | 5분기 호출자 모두 갱신 |
| S-1d | `PplRunCtx` / `BatchRunCtx` / `EvalLlRunCtx`가 sub-struct별 field 보유 + lib 내부 ref 갱신 (chat은 ChatSession 패턴이라 별도) | 4 lib 모듈 |
| S-1e | sanity (cargo fmt/clippy/test) + commit + handoff doc 업데이트 | — |

S-1 PR 단일, S-2~S-4 (binary 추가)는 별도 PR.

---

## Risk + 완화

| R | 항목 | 완화 |
|---|---|---|
| R1 | clap `#[clap(flatten)]` 중첩 시 long-help / short flag 충돌 | sub-struct별 unique prefix 또는 `rename_all` 정책. 호스트 cargo build에서 clap 충돌은 컴파일 에러 → 즉시 발견 |
| R2 | `Args::clone()` 호출처가 분기별 sub-struct로 split 시 메모리 cost 증가 | `BaseArgs`는 Clone derive, 분기별 sub-struct도 동일. 총 size는 변동 없음 |
| R3 | legacy `Args` struct에 의존하는 외부 코드 (테스트/benchmark) | `pub type Args = session::cli::legacy::LegacyArgs` re-export으로 type alias 유지. 외부 영향 0 |
| R4 | KIVI/Offload가 BaseArgs 안에 있는데 chat/ppl/batch/eval 모두에 노출 | OK — init.rs가 모든 분기에서 호출되므로 BaseArgs flatten이 자연스러움. KIVI 미사용 분기는 default value로 무영향 |
| R5 | SwapArgs/QcfArgs/ExperimentArgs가 chat에 노출 안 됨 → 일부 사용자 워크플로 단절 | chat에서 swap/qcf 측정 필요 시 `generate --chat` legacy retain으로 처리 |

---

## 데이터 아티팩트

- `.agent/data/sprep_args_matrix/args_fields_unique.txt` — **142** unique field (v2, 숫자 포함 정규식)
- `.agent/data/sprep_args_matrix/args_matrix.tsv` — 8 bucket × 142 field 매트릭스
- `.agent/data/sprep_args_matrix/refs_<bucket>.txt` — 분기별 refs
- `.agent/data/sprep_args_matrix/args_matrix.txt` — 사람이 읽기 좋은 padded 매트릭스

---

## §S-cleanup — 그룹별 field 정리 (S-1 진입 전)

사용자 결정 (2026-05-18): "필요없는 field 많이 삭제하자. 한 그룹씩 살펴보면서 진행. 기능 많이 차이나면 binary 구분으로 처리하자(e.g. eval-ll)."

### Cleanup 분류 정책

| 분류 | 처리 |
|---|---|
| **즉시 삭제** | 모든 bucket 0건 + cli.rs 정의 자체가 placeholder/dead |
| **측정 종결 시 삭제** | ARGUS/EuroSys'27 측정 끝난 ablation flag |
| **별도 binary로 분리** | production 분기와 기능적으로 무관한 측정/실험 모드 (e.g. experiment_*, swap ablation) |
| **production 유지** | secondary_gguf 등 production swap path |

### 즉시 삭제 후보 (확정 — group review 무관)

| field | 근거 |
|---|---|
| `no_prefill_ws` | 142 field 중 유일하게 **모든 bucket 0건**. cli.rs에 정의만 있고 사용처 0 |
| `qcf_betas` | placeholder ("currently fixed in dump"). 0건 |
| `qcf_topk_values` | placeholder. 0건 |
| `qcf_defensive_taus` | placeholder. 0건 |

총 4 field 확정 삭제 → 142 → **138**.

### 그룹별 review 순서 (실제 진행)

| 순서 | 그룹 | 상태 | 처리 |
|---|---|---|---|
| 1 | ModelLoadArgs (4) | ✓ done | cleanup 0 (모두 production) |
| 2 | BackendArgs (13→9) | ✓ done | -5 field (switch_threshold/cuda_defer_sync/gpu_yield_every_layer/gpu_yield_us/use_rayon) + zero_copy→no_zero_copy (default ON 전환). 커밋 `5e5d1743` |
| 3 | SamplingArgs (6) | ✓ done | cleanup 0. `repetition_window` = repeat-last-N(=64), `top_k`+`top_p` cascade는 표준 — 둘 다 보존 |
| 4 | ProfileArgs (12→10) | ✓ done | `profile_per_head` 보존(H2O+ 분석). `dump_importance` → QcfArgs 재분류 |
| 5 | KiviArgs (5→4) | ✓ done | `awqe` flag → env var `LLMRS_KIVI_AWQE` 이전. 4 production 보존 |
| 6 | OffloadArgs (3) | — | production 예상 |
| 7 | EvictionArgs (14) | — | `h2o_debug` gen-only 삭제 검토 |
| 8 | PrefillArgs (5→4) | ✓ partial done | `no_prefill_ws` 삭제 (batch 1) |
| 9 | ResilienceArgs (3) | — | production 예상 |
| 10 | TensorPartitionArgs (1) | — | production |
| 11 | SwapArgs (24) | — | **대대적 cleanup 대상** — production 5개 외 19개 ablation. 별도 binary(`swap_bench`) 분리 검토 |
| 12 | QcfArgs (13→9 + dump_importance) | ✓ partial done | placeholder 3 삭제(batch 1). `qcf_trajectory`/`decode_x_steps`/`importance_formula` 측정용 — 별도 binary(`argus_bench`) 분리 검토 |
| 13 | ExperimentArgs (5) | — | **전체 binary 분리** — `bin/experiment_runner` 신설 |
| 14 | ChatArgs (4) | — | `chat` flag bin/chat.rs에서 제거 |
| 15 | PplArgs (10) | — | `dump_q4_*` / `ppl_warmup_swap` / `ppl_measure_prefill_tokens` 측정 종결 시 삭제 |
| 16 | BatchArgs (3) | — | production |
| 17 | EvalLlArgs (3) | — | `eval_ll` flag bin/eval-ll.rs에서 제거 |

### Binary 분리 후보 (사용자 "기능 차이 크면 binary 구분" 지시 적용)

| 신규 binary | 흡수 field | 용도 |
|---|---|---|
| `bin/experiment_runner` (신설) | ExperimentArgs 5개 + 일부 BaseArgs flatten | experiment_schedule JSON 기반 측정 모드. 현재 main legacy fallback의 일부 |
| `bin/swap_bench` (신설, 선택) | SwapArgs 측정 ablation 15+ field | 모든 LISWAP-* / Probing-K / Phase-aware ablation. ARGUS/EuroSys'27 측정 종결 후 archive 또는 별도 brunch 유지 |
| `bin/argus_bench` (신설, 선택) | QcfArgs 측정 ablation 6+ field | qcf_trajectory / decode_x_steps / importance_formula |

분리 결정 정책:
- production 4 binary (chat/ppl/batch/eval-ll)는 **측정 옵션 노출 안 함**
- 측정 binary들은 ARGUS 종결 후 별도 retire 시점 결정 가능

### 진입 명령

다음 session에서 `"그룹 1 진행"` (ModelLoadArgs부터 cleanup) 또는 `"즉시 삭제 4개 진행"` (no_prefill_ws + qcf placeholder 3) 으로 시작.

---

## 게이트

본 sprint(S-prep) 문서 갱신은 **코드 변경 0**, cargo build / test 게이트 불필요.
S-cleanup 그룹별 진행 시:
- 호스트: `cargo build --workspace --bins` PASS
- 호스트: `cargo test --workspace` 회귀 ≤ master
- 각 그룹 cleanup PR 단위로 commit

S-1 (BaseArgs 추출) 진입 시 추가 게이트:
- 디바이스: Galaxy S25 generate 32 tok smoke (S-1 lift-only라 perf-neutral 예상)
