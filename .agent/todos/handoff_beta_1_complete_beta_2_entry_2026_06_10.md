# Handoff: β-1 완료 → β-2 진입 (driver phase 배선)

**작성**: 2026-06-10
**HEAD**: `b8b58465 feat(microbench): score_buf D2H readback 측정 bin + S25 실측` (origin/master 동기 — push 완료)
**브랜치**: master
**작성자**: 메인 세션
**다음 세션 진입 문장**: **"β-2 진행 — driver phase 배선. SSOT = `.agent/todos/roadmap_beta_decode_loop_rewrite_2026_06_10.md` §β-2 (계약은 §β-1 완료 기록 + `arch/pipeline_stage_design_v2.md` §5.2.1)."**

---

## TL;DR

- **끝남**: ① **β roadmap 확정** — 워크플로우 21-agent 설계(census→3안→judge→합성→적대검증) + 사용자 grill 7건(G1~G7) 해소, 7 substep 확정. ② **β-1 "계약·어휘 마감" 완료** — PipelineRegistry 신설(unwired) + INV-DECODE-STAGE-004~007 spec test 11건 + INV-LAYER-006 개정으로 **master RED 해소** + SSOT 정오표·normative 3건. ③ **score_readback microbench 완결** — 4세션 묵은 미커밋 아티팩트를 S25 실측 후 커밋(D2H 2.7% TBT, plugin 실용 가능 / rpcmem readback 부적합).
- **다음 = β-2**: decode_loop run()/prefill()에 empty-registry phase dispatch 배선(거동-0). 멈춘 이유 = β-1 게이트 GREEN 후 자연 경계(사용자 세션 종료).
- **β-2는 device 게이트 substep** — S25(`R3CY408S5SB`, devices.toml key=`galaxy_s25`) 오늘 연결 확인됨.

---

## 진행 상태 (전부 origin push 완료)

| commit | 내용 | 게이트 |
|---|---|---|
| `97eea420` | β roadmap 확정 (7 substep + G1~G7) | — |
| `c5e0fdae` | docs: spec/41 + SSOT §5.2.1 신설·정오표·normative 3건 | — |
| `9ec108c0` | feat: pipeline_registry.rs + spec test 11건 + master RED 해소 | lib 1261→**1267**(+6, 회귀 0) · spec 11/11 · `test_inv_layer_006` RED→GREEN |
| `653f56e5` | fix: BANNED `CacheManager` 재추가(금지 복원 구조 — Tester P1) | test GREEN 유지 |
| `ba0b343f` | style: fmt-dirty 10파일(gate-C 잔여, stash 왕복으로 순수 fmt 검증) | fmt --check clean |
| `4b52cf21` | β-1 완료 roadmap 기록 | — |
| `b8b58465` | score_readback microbench + S25 실측 | Qwen 구성 csl2048: B=0.877ms(2.7% TBT), D=3.73ms |

Tester 독립 검증: full suite GREEN(GPU 부재 21건 환경 실패 제외 — `backend::opencl` 모듈, 회귀 아님).

---

## 다음 작업 (β-2 — roadmap §β-2가 정본, 여기는 요약)

1. **구현**: `decode_loop.rs` run(:124-346)/prefill(:79-105)에 `Arc<PipelineRegistry>` 슬롯(builder `with_pipeline`, default empty) + 9 phase 발화(Prefill 2종 + per-token 6종 + Finalize). **run_until_stop(:356-497) 미접촉**. StepInfo pressure는 `Pressure::default()`. v1 trait 호출 (a)~(h) 전부 보존.
2. **검증(host)**: full suite + chat spec 2종 → **검증(device)**: `argus_cli --no-resilience` frozen baseline 3-dtype sig md5(f16 `304f4ada..`/f32 `684d01d9..`/q4 `1cfba273..`, `frozen_baseline_alpha_k_5f_2026_06_05.md`) + avg_tbt n=5 median **Δ≤+3% (α-K frozen 절대값 54.22/54.04/53.79 ms/tok 기준)**.
3. 완료 시 roadmap §β-2 완료 기록 + 커밋.

---

## Landmines / 미해결

- **★ tbt 게이트는 항상 α-K frozen 절대값 기준** — per-substep rolling 재기준화 금지(미세 회귀 누적 차단). β 전 구간 공통.
- **★ per-token 6 phase 발화 오버헤드**가 β-2 유일 실위험 — β-1의 `len==0` atomic fast-path(무lock)가 방어선. 회귀 시 emission 지점 축소로 substep 내 후퇴(roadmap 기재).
- **chat도 prefill() 공유**(chat/session.rs:100) — Prefill 2종 phase가 chat에 닿음. empty registry라 거동-0이지만 chat spec 2종이 게이트에 포함된 이유.
- **`Pressure(0)` 리터럴 불가** — tuple 필드 private. `Pressure::default()` 사용(pipeline.rs:24).
- **INV-LAYER-006 allowlist 인계**: `cache_manager` 필드는 과도기 허용(`ALLOWLISTED_TRANSITIONAL`, test_inv_layer_006.rs:44) — **β-3에서 필드+allowlist 동시 제거**(BANNED 복원은 기성립, `653f56e5`).
- **host GPU 부재**: `backend::opencl` 테스트 21건은 host에서 환경 실패가 정상 — full suite 판정 시 제외하고 볼 것.
- **AB- 트랙 동결(G1)** — argus-bench AB-2/4/6 작업 요청이 와도 β-4 완료 전엔 착수하지 않는다(2회 이전 비용).
- **백로그(비차단)**: `scripts/check_spec_coverage.sh` line 90 octal 버그(`printf '%03d' 008`) + INV-DECODE-STAGE 시리즈 추출 로직 부재 — 별도 chore.
- **워크플로우 설계 산출물**: 합성 계획 원본은 세션 한정(`/tmp/beta_plan.json`) — roadmap에 보정 반영 완료라 손실 없음.

## 참조
- roadmap(정본): `.agent/todos/roadmap_beta_decode_loop_rewrite_2026_06_10.md` — grill 결정 G1~G7 + substep 상세 + 게이트.
- 계약: `arch/pipeline_stage_design_v2.md` §5.2.1(driver↔Stage 4계약: pos-환류/PreEviction·PostEviction/StopReason 매핑/미발화 처분).
- 코드: `engine/src/session/pipeline_registry.rs`(신설, unwired) · `engine/src/pipeline.rs`(L2 어휘) · `engine/tests/spec/test_inv_decode_stage_004_007.rs`.
- 메모리: [[score-readback-microbench]](오늘 실측) · [[project_pipeline_alpha_w]] · [[project-pipeline-alpha-k]].
