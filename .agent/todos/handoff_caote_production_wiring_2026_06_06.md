# Handoff: CAOTE production 배선 MVP (Tier 1) → live E2E / Tier 2

**작성**: 2026-06-06
**HEAD**: (이 커밋) `feat(eviction): CAOTE production 배선 MVP — feature-gate 플러그인 install + value-aware`
**브랜치**: master
**작성자**: 메인 세션 (오케스트레이터)
**다음 세션 진입 문장**: **"CAOTE Tier 2 (last_attn threading)"** 또는 **"argus-chat 마이그레이션에 CAOTE value-aware E2E 배선"**
(단순 후속 조사면 → "CAOTE 배선 갭 검토")

---

## TL;DR

CAOTE 를 **feature-gate 플러그인(`--features caote`)으로 production 화** — `--eviction caote` 로 선택 가능한
value-aware(importance-weighted) eviction 정책. 선택 seam(chat·argus_bench)이 이미 `find_stage` OCP 라
**기존 로직 수정 0**, 변경은 6파일(Cargo feature + force-link cfg + score_based 1항 + CLI variant + ADR/docs).
전 게이트 GREEN. **멈춘 이유**: MVP 완주 — Tier 2(attn-weight) + live E2E 바이너리는 명시적 deferred/마이그레이션 종속.

---

## 진행 상태 (검증된 수치)

| 항목 | 결과 |
|---|---|
| lib test `--features caote` | **1238/0** (caote 통합 `caote_stage_visible_and_value_aware_executes` 포함) |
| lib test default(무회귀) | 1237/0 + `test_release_unused_pages_rss_reduction` **알려진 flake**(격리 1/1 통과, page-release 무관) |
| caote crate | **2/0** |
| CLI parse | feature ON `parses_caote_unit_subcommand` ok / OFF `rejects_caote_when_plugin_absent` ok |
| release fat-LTO(--gc-sections) | caote distributed_slice **생존**(`--release ... stage_registry` 11/0) |
| clippy | `--workspace` & `--features caote` 양쪽 `-D warnings` clean |
| fmt | clean (무관 htp_fastrpc 드리프트는 복원) |
| 적대적 리뷰 | wf_ed8fbbd0 (결과는 커밋 전 반영) |

**변경 파일(전부 7)**: `engine/Cargo.toml`, `engine/src/pressure/eviction/stage_registry.rs`,
`engine/src/session/chat/session.rs`, `engine/src/session/cli/eviction.rs`,
`engine/src/session/assembly/build_bench_loop.rs`(적대적 리뷰 반영 — 에러 메시지에 d2o/caote 안내),
`docs/adr/0004-kvcachestage-plan-returning-trait.md`(§8), `docs/50_adding_kvcache_stage.md`(§3-3).

**배선 구조**:
- 링크 = feature `caote`(optional dep + `[features]`) + module-level `#[cfg(feature="caote")] use caote as _;`.
- 선택 = chat `session.rs:build_chat_eviction_internal` + argus_bench `build_bench_loop:build_resilience_cache_manager`
  둘 다 `name => find_stage(name) → StageBackedPolicy` 와일드카드(수정 0). CLI = `EvictionCmd::Caote`(unit,
  feature-gate) + `policy_name()→"caote"`.
- value-aware = `session.rs` `score_based` 에 "caote" 추가 → `force_evict_with_scores`(importance) →
  `KVStageCtx(Some(importance))` → CAOTE `crit=importance·‖v−o_h‖`. V 는 `tensor(Value)`(cache)로 항상 Some.

---

## 다음 작업 (검증 게이트 포함)

1. **CAOTE Tier 2 (attn-weight, ADR-0004 §8)** — `use_aw=true` per-head attention-weight CAOTE:
   - `EvictionPolicy` trait 에 last_attn 슬롯 = `ScoreContext` 신규 variant(`PerHeadAttn{flat,last_attn,n_kv_heads}`)
     → `cache_manager`(force/maybe_evict_with_attn) → `try_evict`(model_forward, 3 호출부) → `StageBackedPolicy`
     override → `KVStageCtx::new(.., Some(head_scores), Some(last_attn))`.
   - **선결**: production decode 의 `last_step_head_attn` 은 현재 eval-ll probe 전용 → chat decode 에서 채우는
     probe 배선 필요(없으면 chat threading 만으론 attn_handle=None → importance fallback).
   - **검증**: host CPU 에서 `has_attn_weights()=true` 단위 테스트 + (가능 시) S25 GPU proxy(`import_gpu_scores`).
2. **live E2E value-aware 배선** — 현재 갭 해소:
   - (택1) argus_bench `build_resilience_cache_manager` 에 `AttentionScoreAccumulator` 장착 + `ScoreContext` 공급
     → `argus_bench --features caote ... eviction caote` 로 resilience-directive eviction E2E.
   - (택2) argus-chat 신규 bin 이 추출 `ChatSession`/`run_chat_repl_v2` 를 호출하도록 배선(현재 미존재).
   - **검증**: `--features caote` 빌드, caote eviction 실발동 + coherent 출력 + 무크래시.
3. (선택) eval-ll/ppl/batch 의 `score_based_eviction` 소스에 caote 포함(동형 1줄, 하네스 일관성).

---

## Landmines / 미해결 / 안 가본 길

- **value-aware CAOTE E2E 실행 shipping 바이너리 부재**(핵심 갭, ADR §8): chat session 추출본
  (`session/chat/`)은 **어떤 bin 도 호출 안 함**(`argus-chat` planned; `argus_cli`/`argus_bench` 는 `--chat`
  reject). argus_bench(live)는 eviction 지원하나 **score accumulator 미장착**(comment: "AB-1 eviction:
  score-free") → caote recency-degrade. legacy_generate 동결 + 자체 inline chat-eviction(추출본 미사용).
  → 본 MVP 는 **배선만** 완성. 이건 내 배선 결함이 아니라 argus-* 전환 진행 상태.
- **importance 없이 도는 caote = degenerate**: `weight()` importance=None → 0 → crit 전부 0 → keep 은 sort_unstable
  안정성에 의존(앞쪽 인덱스). 그래서 caote 는 반드시 score_based 경로여야 함. argus_bench(score-free)에서
  `--eviction caote` 는 이 degenerate 양상(recency-유사). 위 #2 로 해소.
- **`EvictionCmd::Caote` 는 feature-gate** — default 빌드엔 subcommand 부재(clap reject). policy_name() 의
  exhaustive match 도 cfg arm. **EvictionCmd 의 유일 exhaustive match 는 policy_name()** (타 메서드는 `_ =>`).
  새 변형 추가 시 컴파일러가 잡음.
- **caote 는 튜닝 파라미터 없음**(`make:|_params|Box::new(Caote)`) → unit variant. 향후 파라미터화 시 `Caote(CaoteArgs)`.
- **page-release flake**: `test_release_unused_pages_rss_reduction`(OS munmap 계정) 병렬 실패 간헐 — 격리 통과.
- **htp_fastrpc.rs fmt 드리프트**는 내 작업 무관 → `git checkout` 복원함(diff 6파일 유지).

---

## 참조

- SSOT: `docs/adr/0004-kvcachestage-plan-returning-trait.md` §8(production 배선 MVP), 진행 원장
  `.agent/todos/tensorhandle_impl_progress.md` M-H.
- 기여자 가이드: `docs/50_adding_kvcache_stage.md` §3-3(production 활성화 레시피).
- 선택 seam: `engine/src/session/chat/session.rs`(score_based), `engine/src/session/assembly/build_bench_loop.rs`.
- CLI: `engine/src/session/cli/eviction.rs`(EvictionCmd::Caote).
- CAOTE crate: `crates/techniques/caote/src/lib.rs`.
- 매핑/리뷰 workflow: `wf_8abfb17f`(4-서브시스템 매핑), `wf_ed8fbbd0`(적대적 리뷰).
- 메모리: `[[project-tensorhandle-interface]]`.
