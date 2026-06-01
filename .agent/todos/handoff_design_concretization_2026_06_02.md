# Handoff — 설계 구체화 (pressure/ 해체 reconciliation 닫힘 → G5 / Phase α 코드)

**작성**: 2026-06-02 (pressure grill 종료 + arch doc 반영·커밋·푸쉬 후)
**HEAD**: `2baf581b docs(arch): pressure/ 해체 → kv/ + weight/ 도메인 rename (G3-reconcile §2.1)` (= origin/master, **pushed**)
**브랜치**: `master` (ahead 0, clean)
**작성자**: 메인 세션 (Claude)
**다음 세션 진입 문장**: **"G5 진행"** (`DeviceTarget` variant + `StepInfo` 필드 확정)

> 설계 SSOT = `arch/pipeline_stage_design_v2.md`. v1(동결) = `arch/pipeline_stage_design.md`. **코드 아님 — 설계 문서 구체화 트랙.** Phase α-W/α-K = 코드 진입(아직 아님).

---

## TL;DR

pressure grill (Q1~Q4) 완료 → arch doc §0.4/§2.1/連歴/산재참조 반영·커밋(`2baf581b`)·푸쉬. pressure/ god-디렉토리를 **kv/(KV-cache 도메인) + weight/(swap 오케스트레이션)** 로 해체 확정. 직전 세션의 "두 Q1-Q6 충돌"(pressure/ 유지 vs 삭제)이 **kv/+weight/ 합성**(제3안)으로 종결. 설계 구조 grill은 G1/Hardware/G3/G3-reconcile/ResilienceStrategy 까지 사실상 다 닫힘. 남은 건 **detail-fill 게이트(G5) + 코드(Phase α)**. 멈춘 이유 = 세션 결론 반영 완료, 다음은 G5(또는 사용자가 code 진입 선택).

---

## pressure/ 해체 — 닫힌 결정 (doc 반영·커밋 완료, **재grill 금지**)

| # | 결정 |
|---|---|
| Q1 | `pressure/` → **`kv/`** (KV-cache 도메인: format **flat** + `eviction/` policy + `offload/` tier + `d2o/` algo) |
| Q2 | `pressure/weights/` → **`weight/`** 신설 (runtime swap 오케스트레이션. §13.8-O `RuntimeResourcesAccess` trait 경계 유지, `models/weights/` load-time artifact 와 분리) |
| Q3 | handler split = **함수 단위 cut** (트리거 `handle()`→`stages/`, 알고리즘→도메인 dir; d2o merge ~440 LOC·`offload_one`/`recall_one` 추출). G3 "file 단위" 정밀화 |
| Q4 | format = `kv/` **1차 타입(flat)**, `kv/format/` subdir 철회. format=**축**(L2 trait `KVCacheFormat`/`WeightFormat` + 공유수학 `quant/`), kv 종속 아님. per-layer 동적 precision = Stage+format mutation primitive |

**근거 핵심 (재확인 불요)**: redesign 후 pressure/ 내용물 전부 KV-cache 데이터 → "pressure" 는 역사적 사고. `weights/` 는 KVCache 0 import 오배치 입주자. format impl 은 데이터 내재(KIVI KV cache ↔ Q4 weight block 공유 코드 0, `quant/` 만 공유). cache_manager(1529) = 전부 orchestration → `session/` PipelineRegistry(L4), kv/ 잔여 0.

---

## 다음 작업 (순서)

1. **G5** (진입 문장) — `DeviceTarget` variant 열거 + `StepInfo` 필드 확정. doc §2.1 line 178(`hardware.rs`)·179(`pipeline.rs`)에 **G5** 마킹된 미결 2건. 영향: `Hardware::resolve(target)` 대상 + per-step pressure/step carrier.
2. **G4** (수치/임계) — **R3/R4 사용자 리스크 트랙 결과 먼저 확인** (user-owned, 재실행 금지). `Pressure` `band()` cutoff 등.
3. 이후 코드 = **Phase α-W**(`hardware.rs`/`capability/`/`pipeline.rs` 신설 + `stages/` 골격 + `CommandSource`/`CommandDispatcher`/`LoopControl` + `ResilienceAction`/`MemoryStrategy` 삭제) → **Phase α-K**(`pressure/`→`kv/` + `pressure/weights/`→`weight/` rename **189 ref/53 file** + handler 함수 cut + `KVCacheFormat`/`WeightFormat` trait 확립).

---

## Landmines / 미해결

- **kv//weight/ rename = Phase α-K 코드 작업, 아직 미적용** — doc만 반영됨. 코드 디렉토리는 여전히 `engine/src/pressure/`(+ `pressure/weights/`). rename blast radius **189 ref/53 file**(기계적). d2o_handler 2273 LOC 함수 cut(트리거/알고리즘/테스트 분리) 동행.
- **`kv_migrate.rs`** = §4.1 연혁 storage-slot 트랙(🟡) 미해소. kv/ 로 가지만 최종 거처는 KV storage-slot 분리(weight `Arc<LayerSlot>` 대칭) 후 재검토.
- **리스크 트랙 = 사용자 소유** — R1~R6 재실행 금지. G4 전 R3/R4 결과 확인.
- **커밋 금지 대상**(working tree untracked): `.antigravitycli/`, `.claude/scheduled_tasks.lock`, `papers/.../microbench_full_matrix_2026_05_28/*`, `.agent/todos/handoff_microbench_full_matrix_p4_sweep_wait_2026_05_29.md`. 커밋 시 **파일명 지정**(`git add -A` 금지).
- **review 스킬 트리거 = 한국어 '리뷰' 만**. "확인/논의/검토/비교"는 일반 grill/분석 모드.

## 선행 문서
① 본 handoff ② SSOT `arch/pipeline_stage_design_v2.md`(§0.4 front-door·§2.1 type→file + G3-reconcile 連歴·§5.4 resilience 2-source·§3.5 Hardware·§3.6 wiring) ③ v1 동결 `arch/pipeline_stage_design.md` ④ 용어 `/CONTEXT.md`(3축: stage⊥format⊥hardware) ⑤ 코드 근거: `engine/src/pressure/`(현 위치, Phase α-K rename 대상), `engine/src/quant/`(공유 format 수학), `shared/src/lib.rs:189`(`EngineCommand`), `engine/src/pressure/cache_manager.rs`(→ L4 PipelineRegistry).

## 자기점검
- 진입 문장으로 첫 명령 가능: ✓ "G5 진행"
- 왜 멈췄나: ✓ pressure reconciliation 반영 완료, 다음은 detail-fill(G5)
- 최대 landmine: ✓ kv//weight/ rename 은 doc만 — 코드(Phase α-K) 미적용
- 검증: 설계 구체화라 게이트 = doc 자기완결 (코드 게이트 아님)
