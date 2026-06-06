# TensorHandle 인터페이스 구현 진행 원장

**진입 문장**: "TensorHandle 구현 M-? 진행"
**SSOT**: 이 파일 + docs/adr/0004 (M-A 개정안) + PoC `/tmp/llm_rs2_poc/stagectx_read_dispatch.rs`

## 잠긴 결정 (사용자 확정)
- **D-1 완전 통합**: `dequant_k`/`dequant_v`/`head_score`/`has_head_scores`/`attn_weight`/`has_attn_weights`를
  `tensor()` 위 default sugar 로. `importance()`는 zero-copy 직접 유지(예외 — flat scalar 를 per-element
  read_row 로 돌리면 H2O scalar 랭킹 경로만 순손해).
- **D-2 TensorKind 4종**: Key/Value/AttnWeights/Scores. AttnWeights = `last_step_head_attn`(last layer·last step,
  CPU overwrite / GPU=head_importance proxy). `has_attn_weights()` 게이트. CAOTE 선택 시 startup assert.
- **D-3 CAOTE crate 동봉 + host 실행 테스트**. 품질/PPL eval·CLI·device 는 별도 라운드.

## 설계 contract (동결)
- TensorKind { Key, Value, AttnWeights, Scores }  (Scores = per-head 누적 head_importance, per_head=true cols=1)
- TensorShape { rows, cols, per_head }  (POD, repr(C)-able) / TensorDtype { F32, F16, Q4_0 }
- trait TensorHandle { shape()->TensorShape; dtype()->TensorDtype; read_row(row,kv_head,out:&mut[f32]) }  (dyn-safe)
- StageCtx 필수: current_pos/target_len/layer_idx/n_kv_heads/head_dim/importance/**tensor(kind)**
- StageCtx sugar(default, tensor() 위임): dequant_k/dequant_v/head_score/has_head_scores/attn_weight/has_attn_weights
- StageCtx impl 3곳만 +tensor() 필요: 엔진 KVStageCtx + DummyCtx(lib.rs test) + TestCtx(stage_registry test).
  technique crate(6개 + example + caote)는 KVCacheStage 만 구현 → **무수정**(OCP).
- v1 CAOTE: KeepSpec::LayerWide (head reduce 는 plugin 내부; PerHead 는 ⑤ deferred).

## 코드 맵 (편집 대상)
- crates/technique-api/src/lib.rs — 표면 (M-B)
- engine/src/pressure/d2o_handler.rs:514 `dequantize_k` → 형제 `dequantize_v`(v_buffer, 동일 q4_block_offset) (M-C)
- engine/src/pressure/eviction/stage_registry.rs — KVStageCtx + 핸들 구조체 + last_attn 필드 (M-D)
- engine/src/pressure/cache_manager.rs:29 `ScoreContext`(+last_attn) / run_policy_eviction 언팩 (M-D)
- engine/src/session/eval/eviction_hook.rs + ppl/runner.rs — last_step_head_attn() thread (M-D)
- crates/techniques/caote/ (신규) + engine/Cargo.toml dev-dep + force-link (M-F)
- 원천 접근자 OK: AttentionScoreAccumulator::last_step_head_attn() (attention_scores.rs:366), head_importance_scores():348

## 마일스톤 게이트
- [x] M-A ADR-0004 §7 개정안 (TensorHandle 범용 읽기 표면 — interface 동결 기록)
- [x] M-B technique-api 표면 → build + lib test 2/0
- [x] M-C dequantize_v → bit-identity F32/F16/Q4_0 (`dequantize_v_equals_dequantize_k_bit_identical`)
- [x] M-D KVStageCtx::tensor() + KeyHandle/ValueHandle/ScalarHandle + head_scores/last_attn 필드 → 3 테스트
- [x] M-E 완전 통합(sugar) + d2o reader 정합 → compact_parity·d2o_stage_eq_handler_* 12/0 유지 + lib 1238/0 + clippy clean
- [x] M-F caote crate + 등록 + force-link + host value-aware 테스트 (caote 2/0, 엔진 cross-crate 2/0)
- [x] M-G 전체 sanity: workspace lib(technique-api 2 / caote 2 / example 2 / manager 223 / llm_rs2 1238, page-release flake 격리통과) + clippy --workspace clean + release linkme 11/0

**완료**: D-1 완전통합 + D-2 4종 TensorKind + D-3 CAOTE 동봉+host 테스트. 모든 게이트 GREEN.

## M-H: CAOTE production 배선 MVP (Tier 1, 2026-06-06) — ADR-0004 §8
사용자 결정: **범위=Tier 1 MVP(importance-weighted), 링크=feature-gate**.
- [x] H-1 feature install: `caote` dev-dep → optional `[dependencies]` + `[features] caote=["dep:caote"]`
- [x] H-2 production force-link: `stage_registry.rs` module-level `#[cfg(feature="caote")] use caote as _;`
      (test-only force-link 제거, caote_* 테스트 `#[cfg(feature="caote")]` 게이트)
- [x] H-3 선택 seam: chat(session.rs)·argus_bench(build_bench_loop) 둘 다 **이미** `find_stage(name)` generic
      fallback → match-arm 수정 0(OCP). CLI 만 `EvictionCmd::Caote` unit variant + `policy_name()→"caote"`
- [x] H-4 value-aware: session.rs `score_based` 집합에 "caote" 추가(importance 공급 → crit=importance·‖v−o_h‖)
- [x] H-5 게이트: lib 1238/0(`--features caote`, caote 통합 포함) + caote 2/0 + CLI parse 양쪽(ON parse/OFF
      reject) + clippy --workspace & --features caote clean + default 무회귀 + release fat-LTO caote 생존
- [x] H-6 적대적 리뷰(wf_ed8fbbd0, 6 findings/2 real MINOR): unknown-policy 에러 메시지에 caote(feature-gate)
      + d2o 안내 추가(session.rs + build_bench_loop.rs, cfg!(feature="caote") conditional)
- 문서: ADR-0004 §8 + docs/50 §3-3(production 활성화 레시피)

**Landmine(ADR §8)**: value-aware CAOTE E2E 실행 shipping 바이너리 **부재** — chat session 추출본 미배선
(argus-chat planned), argus_bench score-free, legacy 동결. 배선은 완성, live E2E 는 argus-* 전환 종속.

## deferred(후속, ADR §7 Scope + §8)
- **Tier 2**: last_attn threading (trait ScoreContext variant → cache_manager → try_evict 3호출부 →
  StageBackedPolicy override). + production decode 의 last_step_head_attn probe 배선(현재 eval-ll 전용) +
  S25 GPU proxy. `use_aw=true` per-head attention-weight CAOTE.
- **타 경로 value-aware**: argus_bench 에 AttentionScoreAccumulator 장착(AB-task) / eval-ll·ppl·batch 의
  score_based_eviction 소스에 caote 포함(동형 1줄).
- windowed RawAttn(SnapKV 류) 엔진 보존 / query_state(Quest) / TensorKind→TensorHandle fold 임계.

## iter 로그(M-H)
- (map) 4-서브시스템 워크플로우(wf_8abfb17f)로 eviction 경로 매핑 — trait엔 last_attn 슬롯 부재, eviction
  트리거 3곳(model_forward/eviction_hook/ppl) 전부 flat importance만 전달. 선택 seam 2곳(session/bench)이
  이미 find_stage OCP화 완료(M2-B②). chat session 추출본은 미배선 바이너리(argus-chat planned) 발견.
- (impl) 6파일 외과적 변경: Cargo feature + force-link cfg + score_based + CLI variant + ADR/docs.

## iter 로그
- (init) 결정 잠금 + 코드 맵 확보. last_step_head_attn 원천 ready 확인(forward 신규작업 불요, thread만).
- (M-B~M-G) 1세션 완주. technique-api sugar 전환(기존 6기법+example+caote 무수정, StageCtx impl 3곳만 +tensor).
  dequantize_v = dequant_k 의 v_buffer 미러(bit-identical). CAOTE = a_i·‖v_i−o_h‖, V 를 tensor(Value)로 직접
  읽어 metric 자체 계산(plugin=작성자) 증명. PoC(host+ARM) 로 perf 차별 없음 확인 후 TensorHandle 채택.
