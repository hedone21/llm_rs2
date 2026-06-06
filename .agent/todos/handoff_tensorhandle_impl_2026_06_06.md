# Handoff: TensorHandle 범용 읽기 표면 + CAOTE → production 배선 / push

**작성**: 2026-06-06
**HEAD**: `a13ee802 fix(technique-api): TensorShape repr(C) + enum repr(u32) + CAOTE weight precompute (review)`
**브랜치**: master (origin 대비 **ahead 2** — 미푸시)
**작성자**: 메인 세션 (오케스트레이터)
**다음 세션 진입 문장**: **"TensorHandle 후속: CAOTE production 배선(eviction-hook threading + CLI + S25 device 게이트)"**
(단순 푸시만 원하면 → "TensorHandle 커밋 푸시")

---

## TL;DR

ADR-0004 §7 = **읽기 통합 인터페이스(`StageCtx::tensor(kind) -> Option<&dyn TensorHandle>`)** 와 **CAOTE
value-aware 기법**을 구현 완료, master 에 커밋(2개, **미푸시**). 전 게이트 GREEN + 적대적 리뷰(wf_b5d13ff1)
통과(확정 2건 수정 반영). value(V)를 plugin 이 직접 읽어 자체 metric 을 계산하는 "zero-compile plugin" 북극성
달성. **멈춘 이유**: M-A~M-G 마일스톤 전부 완료 — push 여부 + deferred 후속(production 배선) 결정 대기.

---

## 진행 상태 (검증된 수치)

| 항목 | 결과 / 커밋 |
|---|---|
| feat 커밋 | `06b902b7` — TensorHandle 표면 + CAOTE |
| fix 커밋 | `a13ee802` — repr(C/u32) + CAOTE weight precompute (리뷰 반영) |
| 무회귀 | `compact_parity`·`d2o_stage_eq_handler_*` **12/0** (완전 통합 = zero functional divergence) |
| lib test | llm_rs2 **1238/0** (page-release flake 격리통과) + technique-api 2 / caote 2 / example 2 / manager 223 |
| lint/fmt | clippy `--workspace -D warnings` clean, fmt clean |
| release | linkme fat-LTO 생존 **11/0** |
| bit-identity | `dequantize_v` ≡ `dequantize_k`(v_buffer) F32/F16/Q4_0 |
| PoC(perf) | handle vs additive dispatch host+ARM(S25 Oryon) **±0~1%** → perf 차별 없음 |

**구현 범위**: 읽기를 `tensor()` 단일 경로로 통일(`TensorKind`={Key,Value,AttnWeights,Scores}).
`dequant_k/v`·`head_score`·`attn_weight`·`has_*` = `tensor()` 위 default sugar. `importance()` 만 zero-copy
직접(D1 예외). 엔진 핸들 = `KeyHandle`/`ValueHandle`/`ScalarHandle`. CAOTE = `crates/techniques/caote/`
(technique-api 만 의존, dev-dep + force-link).

---

## 다음 작업 (검증 게이트 포함)

1. **push** (원하면): `git push origin master` → ahead 0 확인. (가드레일: 지시 전 금지였음 — 확인 필요)
2. **CAOTE production 배선** (ADR §7 deferred 핵심):
   - `EvictionPolicy::evict_with_head_scores` 에 `last_attn` 전달 경로 추가 → `StageBackedPolicy` 가
     `KVStageCtx::new(.., head_scores, last_attn)` 로 thread (현재 builtins 는 None 고정).
   - eviction-hook(`engine/src/session/eval/eviction_hook.rs`)이 `acc.head_importance_scores()` +
     `acc.last_step_head_attn()` 를 `ScoreContext`/policy 로 공급.
   - session match-arm("caote") 또는 generic `_ => find_stage(name)` fallback 으로 `--eviction-policy caote`
     선택 가능화 (closed match-arm 제거 = 진짜 OCP — 단 동작 변경이라 ADR 판단 필요).
   - **검증**: S25 `--backend opencl --opencl-rpcmem --eviction-policy caote` decode 무크래시 + import_gpu_scores
     proxy 경로(GPU 는 last_attn 비어 head_importance proxy) 확인.
3. **(선택) windowed RawAttn** 엔진 보존 → SnapKV/Scissorhands/Ada-KV 패밀리 해금(TensorKind variant 추가).

---

## Landmines / 미해결 / 안 가본 길

- **AttnWeights = `last_step_head_attn`** = **last layer · last decode step 근사**(windowed/per-layer 정확값
  아님). GPU 경로는 `import_gpu_scores`(attention_scores.rs:308-314)가 head_importance 를 proxy 로 채움 →
  CPU≠GPU 의미 차. CAOTE 는 `has_attn_weights()` false 시 `importance()` 폴백. **blindly 신뢰 금지.**
- **`importance()` 는 통합의 유일 예외** — zero-copy 직접 유지. flat scalar 를 `read_row`(per-element)로
  돌리면 H2O scalar 랭킹 경로만 순손해(PoC 대조군). 새로 통합하지 말 것.
- **PerHead executor 는 여전히 `bail!`** (`stage_registry.rs:execute_kv_plan`, 단계 ⑤ deferred). CAOTE v1 은
  `KeepSpec::LayerWide` 만(head reduce 는 plugin 내부). per-head CAOTE 는 ⑤ 와 함께.
- **production d2o 는 D2OHandler if-branch** 유지(D2OStage 등록돼 있으나 우회) — 본 작업 불변.
- **page-release 테스트 = 환경 flake**: `test_prune_prefix_calls_release_unused_pages`,
  `test_release_unused_pages_rss_reduction` (OS munmap 계정 의존). 병렬 실행 시 간헐 실패, **격리 9/9 통과**.
  회귀 아님 — 재실행/격리로 확인.
- **`engine/Cargo.toml` 외부 batch_dispatch 변경은 내 것 아님** — 건드리지 말 것(내 diff 는 caote dep 1줄만).
- **PoC 소스는 repo 밖**: `/tmp/llm_rs2_poc/stagectx_read_dispatch.rs` (throwaway, 미커밋). 디바이스 재현
  필요 시 `engine/microbench/` 정식 bin 으로 편입 가능(현재 미편입).
- **커밋 금지 untracked**: `.antigravitycli/`, `.claude/scheduled_tasks.lock`, `papers/**/microbench_*`,
  `.agent/todos/handoff_microbench_*.md` 등.

---

## 참조

- SSOT: `docs/adr/0004-kvcachestage-plan-returning-trait.md` §7 (M-A 개정안), 진행 원장
  `.agent/todos/tensorhandle_impl_progress.md`
- 표면 정본: `crates/technique-api/src/lib.rs` (TensorKind/Shape/Dtype/Handle, StageCtx)
- 엔진 impl: `engine/src/pressure/eviction/stage_registry.rs` (핸들 + KVStageCtx::tensor),
  `engine/src/pressure/d2o_handler.rs:556` (dequantize_v)
- CAOTE: `crates/techniques/caote/src/lib.rs`
- 설계/리뷰 workflow: `wf_1dda0f82`(판정단), `wf_b5d13ff1`(적대적 리뷰), `wf_da6a44e7`/`wf_2f0e8bcd`(사전 조사)
