# Handoff — Phase 4-4 + QCF rename 진입점 (2026-05-19)

**작성**: 2026-05-19 (S-subcmd sprint 종결 직후)
**master HEAD**: `2eb3765d`
**진입 문장 (Sprint 1)**: "Phase 4-4 진행"
**진입 문장 (Sprint 2)**: "QCF rename 진행"
**병렬 평가 결과**: **순차 권장** (병렬 시 ff-merge 충돌 거의 확실)

---

## 1. 직전 sprint 종결 요약 — S-subcmd 옵션 C

| Commit | 작업 |
|---|---|
| `ad658a61` | 설계 doc (EvictionCmd + KvMode) |
| `887037d1` ~ `ba7d2cff` | C1~C8: EvictionCmd subcommand + 175 call site shim + docs/scripts sed |
| `83c73dff` | C4-1~3 KvMode subcommand 신설 |
| `50792ec8` | C4-4a mode 분기 shim (10건) |
| `6a18c5b6` | C4-4b sub-args shim 통합 (13건) |
| `28055252` | C5/C6 옵션 B legacy hide + deprecation warning |
| **`2eb3765d`** | **옵션 C legacy 완전 제거 + docs/scripts 마이그레이션** |

**최종 결과**:
- 23 eviction field + 7 KIVI/offload field → `eviction <policy> <subcmd>` + `--kv-mode {standard,kivi,offload}` 단일 enum 구조
- legacy flag 6개 완전 제거 (`--kivi`, `--kivi-bits`, `--kivi-residual-size`, `--kv-offload`, `--offload-path`, `--max-prefetch-depth`)
- docs/scripts 7 파일 일괄 마이그레이션
- spec test 9/9 PASS, session::cli 52/52 PASS
- `generate --help` 깔끔 (legacy 0건 노출), `--kivi` 호출 시 clap parse error

---

## 2. Sprint 1: Phase 4-4 — main() eval/ppl/batch 추출

### 2.1 목표

`generate.rs` main()의 잔여 3 분기를 `session/{eval,ppl,batch}/` 모듈로 추출하여 main() LOC를 **9,860 → 목표 ≤400**으로 줄인다.

### 2.2 현 상태 (master `2eb3765d` 기준)

```
$ wc -l engine/src/bin/generate.rs  →  6513
$ wc -l engine/src/session/cli/mod.rs  →  1182
```

main() 진입점 + 분기 위치:

| Line | 분기 |
|---|---|
| `engine/src/bin/generate.rs:94` | `let mut args = Args::parse();` |
| `engine/src/bin/generate.rs:96` | `let ctx = SessionInitCtx::build(&args)?;` (Phase 4-1 추출 완료) |
| `engine/src/bin/generate.rs:379` | `let initial_kv_capacity = if args.eval_ll || args.ppl.is_some() { ... }` |
| `engine/src/bin/generate.rs:1658` | **`if args.eval_ll { ... }` — eval-ll 분기** |
| `engine/src/bin/generate.rs:1754` | **`if args.ppl.is_some() { ... }` — PPL 분기** |
| `engine/src/bin/generate.rs:1785` | **`if args.prompt_batch.is_some() { ... }` — prompt batch 분기** |

### 2.3 이미 존재하는 모듈 (Phase 4-5 sprint에서 일부 추출됨)

```
engine/src/session/
├── eval/      (이미 존재 — eval-ll runner)
├── ppl/       (이미 존재 — PPL runner)
├── batch/     (이미 존재 — batch runner)
├── chat/      (Phase 4-5 완료)
├── forward/   (Phase 4-3 ModelForward 완료)
├── init.rs    (Phase 4-1 SessionInitCtx 완료)
├── cli/       (S-subcmd sprint 완료)
├── decode_loop.rs (Phase 4-2 완료)
├── samplers/  (Phase 4-2 완료)
├── assembly/
└── qcf_runtime.rs
```

즉, `eval/`, `ppl/`, `batch/` 모듈은 이미 부분 존재. **잔여 작업**은:
- main()의 1658/1754/1785 분기 내부 로직을 해당 모듈로 마저 옮긴다
- main()에는 dispatch 케이스만 남긴다 (`session::eval::run(...)` 등)
- DecodeLoop + ModelForward + ChatSession 패턴 재사용

### 2.4 진입 절차

1. `EnterWorktree` 명 `phase4_4_eval_ppl_batch`
2. `git rebase master` (HEAD `2eb3765d`로 정렬)
3. **eval-ll 분기 먼저** — 가장 작고 격리하기 쉬움
4. **PPL 분기** — eval-ll와 유사한 구조
5. **batch 분기** — 가장 복잡 (multi-prompt)
6. 각 추출 후 게이트:
   - `cargo build --release -p llm_rs2` PASS
   - `cargo fmt --all -- --check`
   - `cargo test --release -p llm_rs2 --test spec` PASS
   - `cargo test --release -p llm_rs2 --lib session::` PASS
   - **S25 디바이스 게이트** (chat 추출과 동일 절차):
     - bit-identical 32 tok (eval-ll output)
     - avg_tbt n=5 회귀 ≤5%
7. 단일 commit OR 3 commit (eval/ppl/batch 분리) — 작업자 판단
8. ExitWorktree + ff-merge + branch delete + notify-send

### 2.5 위험 / 완화

| R | 항목 | 완화 |
|---|---|---|
| R1 | eval-ll의 ImportanceTable 누적 로직이 main() local state | `session/eval/state.rs`에 SOLID extract |
| R2 | PPL의 NLL 누적 + token-window slide | `session/ppl/runner.rs`에 이미 일부 있음 — 확장만 |
| R3 | batch가 chat과 prefill workspace 공유 | Phase 4-3 ModelForward로 통일 |
| R4 | QCF runtime 호출 (Sprint 2와 직접 충돌) | **Sprint 1 완료 후 Sprint 2 진입** (순차) |
| R5 | bit-identical 게이트 fail | Phase 4-5 patterns — eager workspace + score 보존 검증 |

---

## 3. Sprint 2: QCF rename — kv./weight. prefix 통일 (Sprint 1 완료 후)

### 3.1 배경

backlog `[P2] QCF 명명 컨벤션 정리`. 2026-04-27 결정 (`docs/qcf_taxonomy.md`):

- **QCF_kv**: KV cache → attention output `‖ΔO‖₂ / ‖O‖₂` 측정. sliding/H2O/streaming eviction, KIVI quant, D2O merge.
- **QCF_weight**: 모델 forward path 측정. weight swap (F16→Q4), layer skip (SWIFT).

코드 위치는 unified_qcf.rs / decider.rs::compute_qcf_swap / layer_importance.rs::compute_qcf 등 prefix 없이 혼재. rename으로 패밀리 구분 명시.

### 3.2 영향 범위 (현 master 기준)

```
engine/src/core/qcf/
├── mod.rs
├── unified_qcf.rs    → qcf_kv.rs 또는 qcf_kv/mod.rs
├── layer_importance.rs (compute_qcf → compute_qcf_weight)
├── estimator.rs (DegradationEstimator key 정규화)
├── entropy.rs
├── topk_retention.rs
├── quant_qcf.rs (KIVI 측정 — kv 계열)
├── skip_qcf.rs (skip 측정 — weight 계열)
└── layer_aggregation.rs

engine/src/models/weights/
├── decider.rs (compute_qcf_swap → compute_qcf_weight_swap)
└── mod.rs

engine/src/eval/{eviction_hook,qcf_helpers,eval_loop}.rs (호출처)
engine/src/session/{ppl/runner.rs, eval/runner.rs, qcf_runtime.rs}
engine/src/profile/quality_metrics.rs
engine/src/bin/generate.rs (CLI dump)
engine/src/session/cli/mod.rs (CLI arg names: --qcf-*)
shared/src/lib.rs (QcfEstimate IPC — estimates HashMap key prefix)
```

### 3.3 작업 단계

| Step | 작업 | 검증 |
|---|---|---|
| Q1 | unified_qcf.rs → qcf_kv.rs rename + mod.rs re-export | build PASS |
| Q2 | `layer_importance.rs::compute_qcf` → `compute_qcf_weight` | build PASS |
| Q3 | `decider.rs::compute_qcf_swap` → `compute_qcf_weight_swap` | build PASS |
| Q4 | DegradationEstimator key prefix 정규화 (`kv.sliding`, `weight.swap` 등) | unit test |
| Q5 | IPC `QcfEstimate.estimates` HashMap key 호환성 — manager 측 갱신 | shared test |
| Q6 | docs/qcf_taxonomy.md 갱신 (rename 적용 표) | doc |
| Q7 | spec test `engine/tests/spec/test_qcf_taxonomy.rs` 신규 (선택) | spec PASS |
| Q8 | backlog `[P2] QCF 명명 컨벤션 정리` → RESOLVED | doc |

### 3.4 진입 절차

1. `EnterWorktree` 명 `qcf_rename`
2. `git rebase master` (Sprint 1 완료 후 master HEAD)
3. Q1~Q8 단일 commit OR 분리 commit
4. 게이트: build + fmt + clippy 사전 존재 외 신규 0 + spec test PASS
5. ff-merge + 정리

### 3.5 위험 / 완화

| R | 항목 | 완화 |
|---|---|---|
| R1 | IPC HashMap key 변경 시 manager 호환성 | shared 크레이트와 동시 변경, e2e mock 테스트 |
| R2 | CLI `--qcf-*` flag 이름은 변경 안 함 (사용자 영향) | 메서드/타입 rename만 |
| R3 | docs/논문 참조 텍스트 다수 | grep + sed 일괄 |
| R4 | rename 회귀 (clippy unused import 등) | cargo fmt + clippy 정리 commit |

---

## 4. 병렬 평가 — 왜 순차인가

### 4.1 동시 수정 충돌 가능성

Phase 4-4 작업이 추출하는 `generate.rs` 1658~1900 영역 + `session/{eval,ppl}/*.rs` 내부에는 **QCF runtime 호출이 다수 포함**:

- `engine/src/bin/generate.rs:1658` (eval-ll) — `ImportanceTable` 생성/누적, `compute_qcf` 호출
- `engine/src/session/eval/runner.rs` — `QcfEstimate` IPC, `compute_qcf_swap` 호출
- `engine/src/session/ppl/runner.rs` — `compute_qcf`, `DegradationEstimator` 호출
- `engine/src/session/qcf_runtime.rs` — 모든 QCF 호출 hub

Phase 4-4가 이 라인들을 **추출 (이동)**하고, QCF rename이 같은 라인들의 **함수명을 변경**한다.

두 worktree에서 동일 라인 동시 변경 → ff-merge 시 3-way merge 충돌 거의 확실.

### 4.2 충돌 해결 비용

- 충돌 해결 시 한쪽 worktree에서 다른 쪽 변경을 수동 재적용 필요
- 게이트 (bit-identical 32 tok + avg_tbt) 가 충돌 후 회귀 가능
- 결국 사용자가 "리스크 있으면 순차로"라 명시했으므로 순차 진행

### 4.3 순차 진행 순서

1. **Phase 4-4 먼저** — main() LOC delta가 더 큰 작업 (메인 트랙)
2. **QCF rename 그 다음** — Phase 4-4가 추출한 깨끗한 session/{eval,ppl,batch}/* 모듈에서 함수명만 변경

이 순서로 진행 시 QCF rename의 영향 범위가 명확히 격리됨 (이미 모듈화된 코드만 수정).

---

## 5. 환경 / 규칙 (재확인)

- **언어**: 모든 응답 한국어, 기술 용어/코드 식별자 원문 유지
- **Background job 임시 파일**: `$CLAUDE_JOB_DIR = /home/go/.claude/jobs/...`
- **EnterWorktree**: 코드 변경 작업 시 worktree 격리 필수
- **테스트 기본 모델 포맷**: GGUF (`--model-path *.gguf`)
- **Android 벤치 스레드**: Galaxy S25는 6T만
- **TBT metric**: 항상 avg_tbt (tok0 inclusive). rest_tbt 단독 비교 금지
- **성능 측정**: `--profile` 없이 (Decode TBT는 `actual_throughput` 또는 `Decode: X ms/tok` 로그)
- **신규 spec test**: `engine/tests/spec/`
- **완료 시 자동 commit + `notify-send`**
- **`.cl` 커널 수정**: Senior Implementer만, Adreno 실측 교훈 준수 (DK=128 register limit, SLM tree-reduce > sub_group_reduce)

---

## 6. 진입 명령 (요약)

다음 세션 사용자 진입 문장:

| 순서 | 문장 |
|---|---|
| 1 | **"Phase 4-4 진행"** — main() eval/ppl/batch 추출 |
| 2 | **"QCF rename 진행"** — Sprint 1 완료 후 |

각 sprint는 단일 worktree + ff-merge + handoff 갱신 패턴 유지.

---

## 7. 참조 문서

- `arch/inference_pipeline.md` — Phase 4 6 trait 설계
- `ARCHITECTURE.md` §13 — Layered architecture INV-LAYER 정의
- `docs/qcf_taxonomy.md` — QCF kv/weight 분리 사양 (rename 대상)
- `.agent/todos/handoff_phase4_5_complete_2026_05_18.md` — Phase 4-5 결과 (직전 main() LOC delta)
- `.agent/todos/handoff_phase4_5_p0_gate_2026_05_19.md` — P0 게이트 + C4-1~3 ff-merge
- `.agent/todos/s_subcmd_design_2026_05_19.md` — S-subcmd 설계 doc
- `.agent/todos/backlog.md` — `[P2] QCF 명명 컨벤션 정리` 항목

---

## 8. 미해결 (낮은 우선)

- **Qwen2 default Instruct 교체** — scripts/run_device.py + docs (R-chat 분리 후속, UX)
- **chat_template Gemma3 구현** — `engine/src/core/chat_template.rs` Gemma3 분기 unimplemented
- **clippy 회귀 cleanup** — backlog [P2] doc_lazy_continuation 29건 + unsafe ptr 1건
- **Adreno noshuffle GEMV cross-run tuning** — backlog [P2] Phase 4-4.9/10 Path B
- **WSWAP-6-A/C** — backlog [P0] Fused SOA convert + Primary release bg worker
- **M3.4 RED pos baked** — backlog [P0] QNN OpPackage architectural blocker

이 항목들은 본 sprint 범위 외 — 진행 시 별도 handoff doc 필요.
