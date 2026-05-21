# Phase 4-D / 4-A / 4-B / 4-C 종결 + 다음 sprint 진입점 (2026-05-18)

**Master HEAD**: `5e5c5753`
**이전 handoff**: `handoff_phase4_5_complete_2026_05_18.md` (Phase 4-5 chat 추출 종결)

---

## TL;DR — 종결 결과

| Sprint | Commit | 작업 | main() LOC |
|---|---|---|---|
| 시작 | `75edb358` | — | 9,860 |
| 4-D microbench 분리 | `bc2a73da` | engine/src/bin/ 62 파일 → engine/microbench/ + Cargo.toml [[bin]] 명시 | 9,860 |
| 4-A batch 추출 | `f6c491a1` | --prompt-batch 모드 → session/batch/ (860 LOC) | 8,984 |
| 4-B-1 qcf shared | `645a91ed` | run_qcf_warmup_workflow + run_layer_swap → session/qcf_runtime (655 LOC) | 8,327 |
| 4-B-2 eval 추출 | `9db119c5` | --eval-ll 모드 → session/eval/ (350 LOC) | 7,902 |
| 4-C-1 swap shared | `ebaa7254` | dispatch_swap_weights + dump_layer_weights_to_dir → qcf_runtime (231 LOC) | 7,671 |
| 4-C-2 ppl 추출 | `5e5c5753` | --ppl 모드 + run_ppl + run_kivi_ppl → session/ppl/ (1,124 LOC) | **6,556** |

**총 감축**: 9,860 → 6,556 LOC (**-3,304, -33.5%**).

---

## 현재 bin/generate.rs::main() dispatch 구조

```
main():
  ├─ args.kivi && args.eval_ll          → session::ppl::run_kivi_ppl
  ├─ args.chat                          → session::chat::repl::run_chat_repl_v2  (Phase 4-5)
  ├─ args.eval_ll                       → session::eval::run_eval_ll             (Phase 4-B)
  ├─ args.ppl.is_some()                 → session::ppl::run_ppl_dispatch         (Phase 4-C)
  ├─ args.prompt_batch.is_some()        → session::batch::run_prompt_batch       (Phase 4-A)
  ├─ is_standard_happy_path(&args)      → session::assembly::build_standard_loop (Phase 4-4)
  └─ fallback                           → legacy generate (main 잔여 ~4,700 LOC)
```

모두 단일 `generate` binary, args flag로 분기. 각 분기 `return` 패턴.

---

## 신규 lib 모듈 (Phase 4-D~C로 신설)

- `session/qcf_runtime.rs` (884 LOC): 5 shared helpers + QcfWarmupResult struct
  - `run_qcf_warmup_workflow`, `run_layer_swap`, `dispatch_swap_weights`, `dump_layer_weights_to_dir`, `read_allow_boundary_env`, `QcfWarmupResult`
- `session/batch/{mod,args,helpers,runner}.rs` (~1,050 LOC): `run_prompt_batch(ctx)`
- `session/eval/{mod,args,helpers,runner}.rs` (~560 LOC): `run_eval_ll(ctx)`
- `session/ppl/{mod,args,runner}.rs` (~1,266 LOC): `run_ppl_dispatch(ctx)`, `run_ppl`, `run_kivi_ppl`, `PplResult`

bin/ 정리:
- `engine/src/bin/`: 6 production (generate, test_backend, auf_tool, signal_injector, test_model, test_q4_soa_byte_equal)
- `engine/microbench/`: 62 microbench/probe/stage (Phase 4-D, Cargo.toml [[bin]] 명시)

---

## 다음 Sprint — Binary 분리 + Args 정리

### 사용자 요구

> "generate는 legacy로 남기되, 다음 세션에서는 chat, ppl, batch, eval-ll 바이너리를 별도로 두고 인자도 정리하자. 지금 인자 너무너무 많아."

### 신규 binary 4개

| Binary | 진입 분기 | 호출 모듈 (이미 lib) |
|---|---|---|
| `chat` | chat REPL (3 KV mode) | `session::chat::repl::run_chat_repl_v2` |
| `ppl` | Perplexity eval | `session::ppl::run_ppl_dispatch` + `run_kivi_ppl` |
| `batch` | Prompt batch | `session::batch::run_prompt_batch` |
| `eval-ll` | Log-likelihood eval | `session::eval::run_eval_ll` |
| `generate` (legacy) | standard + 6 legacy 분기 | bin/generate.rs 잔여 4,700 LOC |

### Args 정리 권장 (옵션 B)

**`BaseArgs` + 분기별 sub-struct (clap `#[clap(flatten)]` 패턴)**

```rust
// session/cli/base.rs (신규)
#[derive(clap::Parser)]
pub struct BaseArgs {
    // 공통 25 field: --model-path / --backend / --max-seq-len /
    // --num-tokens / --temperature / --top-p / --top-k /
    // --greedy / --repetition-penalty / --kv-type / --tokenizer-path / ...
}

// bin/chat.rs (신규)
#[derive(clap::Parser)]
struct ChatArgs {
    #[clap(flatten)]
    base: BaseArgs,
    // chat 전용 30 field: --system-prompt / --chat-template /
    // --kivi / --kv-offload / --eviction-policy / ...
}
```

현재 `cli.rs::Args` 250+ field를:
- `BaseArgs` 25 (공통)
- `ChatArgs` 30~40 (chat 전용)
- `PplArgs` 40~50 (PPL/Swap 관련)
- `BatchArgs` 20~30
- `EvalLlArgs` 20~30
- `GenerateArgs` (legacy retain, 모든 field 포함)

로 분할.

### 작업 분해 (4~5 sprint)

| Sprint | 작업 | 산출물 |
|---|---|---|
| **S-prep** | `cli.rs::Args` field별 사용 매트릭스 분석 (어느 분기에서 사용되는지) | 표 / spec doc |
| **S-1** | `BaseArgs` 추출 + cli.rs 분할 (session/cli/base.rs + session/cli/{chat,ppl,batch,eval}.rs) | `BaseArgs` 25 field + 4 sub-struct |
| **S-2** | `chat` binary 신규 (`bin/chat.rs`, ~50 LOC shim) | `cargo build --bin chat` |
| **S-3** | `ppl` binary 신규 | `cargo build --bin ppl` |
| **S-4** | `batch` + `eval-ll` binary 신규 | 2 binary |
| **S-5** (선택) | generate legacy의 weight swap / KIVI 일반 / offload 추가 추출 | session/standard/ 등 |

### 호환성

- 기존 `generate --chat ...` 사용 패턴 유지 (legacy retain)
- 신규 binary는 추가 옵션, 기존 사용자 영향 없음
- run_device.py, hosts.toml, scripts 등은 binary name 추가만 (변경 zero)

### Risk

| R | 항목 | 완화 |
|---|---|---|
| R1 | `Args::clone()` 의존 분기별 별도 struct로 분할 시 ctx field 매핑 변경 | BaseArgs/sub-struct destructure 후 RunCtx 생성 패턴 유지 |
| R2 | 공통 field가 어느 sub-struct에 속하는지 모호한 case | sprint S-prep에서 사용 매트릭스로 명확화 |
| R3 | clap default value / env / help 메시지 일관성 | clap derive로 한 위치에서 관리 |
| R4 | KIVI / Offload 분기는 chat/ppl 양쪽 모두에서 사용 | KiviArgs / OffloadArgs sub-struct를 chat/ppl 양쪽이 flatten |

### 진입 명령

다음 session에서 `"S-prep 진행"` (사용 매트릭스 분석) 또는 `"S-1 진행"` (BaseArgs 추출 곧장)으로 시작.

---

## 남은 게이트 (별도 sprint, 본 sprint 미실행)

| 게이트 | 설명 | 디바이스 |
|---|---|---|
| **G2 batch** | 2-entry .jsonl baseline diff | Galaxy S25 OpenCL |
| **G2 eval-ll** | eval-ll log-likelihood baseline | Galaxy S25 OpenCL |
| **G2 ppl** | PPL NLL ε<1e-6 | Galaxy S25 OpenCL |
| **G7'** | TBT n=3 ≤5% (전 분기) | Galaxy S25 OpenCL |

lift-and-shift라 perf-neutral 예상이지만 회귀 0 확인 필요.

---

## 미해결 backlog

- **[P1]** Qwen2.5-1.5b chat garbage 출력 (Phase 4-5 baseline 회귀, 본 sprint 외)
- **[P2]** llm_rs2 lib clippy 회귀 — doc_lazy_continuation 29 + unsafe pointer 1 (Phase 4-D 발견)
- **[P2]** noshuffle SOA opt-in Path B (Phase 4-4.10에서 default invert 결정 후 잔여 작업)
- **[P0]** M3.4 RED — pos baked architectural blocker (사용자 결정 대기)
- **[P0]** Weight Swap Layer-Level Mixed Precision Phase A (Architect spec 시작)
- **[P0]** Long context CPU attention (Senior Implementer NEON)

---

## 환경 + 재현

```bash
# 호스트 sanity
cargo check --workspace --bins --release
cargo fmt --all
cargo test -p llm_rs2 --test spec  # ~650 pass, ~3-4 flaky fail (host GPU 미존재)

# Android 디바이스
python scripts/run_device.py -d galaxy_s25 generate -- \
    --backend opencl --model-path /data/local/tmp/qwen2.5-1.5b-q4_0.gguf \
    --prompt "The capital of France is" --num-tokens 32 --greedy
```

Master clippy는 사전 회귀 (29 doc_lazy + 1 unsafe) — `cargo clippy -- -D warnings` 통과 안 됨. 본 sprint는 `cargo check --bins`로 게이트 좁힘. [P2] backlog 해결 후 복원.
