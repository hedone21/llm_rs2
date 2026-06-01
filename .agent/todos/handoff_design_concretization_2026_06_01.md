# Handoff — 설계 구체화 (구현 결정성 공백 G1~G7 메우기)

**작성**: 2026-06-01
**HEAD**: `0911f714 docs(skills): grill-me — 코드 변경 질문 시 구체 산출물 포맷 가이드 추가` (= origin/master, 동기화됨)
**브랜치**: `master`
**작성자**: 메인 세션 (Claude)
**다음 세션 진입 문장**: **"설계 구체화 진입 — G1(wiring 표준 패턴)부터"**

> 이 문서는 `handoff_device_3axis_regrill_2026_05_31.md`(진입 문장 "Phase α-W 진입")를 **supersede**한다. 그 진입은 시기상조였음 — 아래 "이번 세션" 참조. **다음 세션은 코드가 아니라 설계 문서 구체화다.**

---

## TL;DR

3축 설계 grill은 종결됐고(`287b8668`), 설계 문서를 두 렌즈로 검토했다: (1) **리스크**(SOLID/레이어드/성능-확장성/외부기여 + 교차 R1~R6) → **사용자가 직접 검토 중**(이 트랙 건드리지 말 것). (2) **구현 결정성**("누가 구현해도 같은 구조·동작이 나오나") → 공백 **G1~G7 추출 완료**(아래). 다음 작업 = **G1~G7을 설계 문서에 구체화**(코드 아님), G1(construction wiring 표준 패턴)부터. 멈춘 이유: 사용자가 리스크를 병렬 검토하는 동안 결정성 보완을 별 세션으로 분리.

---

## 이번 세션에 일어난 일 (중요 — 코드 상태 오해 방지)

- "Phase α-W 진입"으로 시작 → 내가 **코드 작업으로 오해**하고 W1-step1(CapabilityRegistry 커밋 `6ee69122`) + W1-step2(KIVI 이전, sub-agent, 미커밋)를 진행.
- 사용자: **"코드 작업 아직 시작하지마. 변경사항 다 되돌려"** → **전부 revert 완료**. step2 working 폐기 + step1 커밋 되돌림(HEAD `287b8668` 복귀) + `capability.rs` 삭제.
- 이후 `grill-me/SKILL.md`만 커밋·푸쉬(`0911f714`).
- **현재 engine/ 코드 변경 0. 아직 구현 단계 아님.** α-W(코드)는 설계 구체화가 끝난 뒤.

---

## 다음 작업 (R5) — 설계 구체화: G1~G7 공백 메우기

설계 문서(`arch/pipeline_stage_design_v2.md`)는 **개념·축·계약 차원은 deterministic이나 구현 세부는 미명세**다. 이 프로젝트 결정성은 본래 ①doc(개념)+②spec/INV 테스트(계약)+③bit-identical 게이트(동작) 3중인데, **②가 §8에서 "Phase α 구현 시점"으로 연기**돼 현재 비어 있음 → doc 단독으론 세션마다 구조가 갈림.

게이트도 spec도 안 잡아주는 **진짜 공백(우선)**: **G1·G3·G4·G5**. 나머지(G2·G6·G7)는 "spec/41 + tests/spec/ 동행 등록"이라는 연기된 ②를 실행하면 자연 해소.

| | 미명세 | 갈림 | 구조/동작 | 보완 |
|---|---|---|---|---|
| **G1** ★최우선 | construction wiring/threading 토폴로지 ("construction-held"만 있고 *어떻게 도달*하는지 전무) | KIVI handle = model필드+args / ctx-registry+decode resolve / layer-held / args관통 — 다 충족하나 struct·소유그래프 상이 (이번 세션 실제 겪음) | 구조 | registry 소유자 + 소비자군(model/layer/cache/stage)별 handle 전달 **표준 패턴 1개** 박기 |
| **G2** | trait 메서드 시그니처 (`attention_into` "정확한 시그니처=impl 단계", `write_kv`/`attention_gen_kivi`/`GpuScoreAccess` `/* ... */`) | 메서드 집합은 고정, param 순서·타입은 세션마다 | 구조 | 7 KVCacheFormat + apply_dispatch + capability trait param 리스트 확정 |
| **G3** | 타입 배치(파일) — CapabilityRegistry/Hardware/StepInfo/Pressure 위치, stages/{kv,weight,system}/ 매핑 | 같은 타입 다른 파일 → 모듈그래프 갈림 | 구조 | type→file 배치표 |
| **G4** | 수치/임계 — `Pressure(0-100)→band()` 컷오프, pressure→action 매핑 | 컷오프 다르면 **다른 압력에서 eviction 발동 = 동작 상이** | 동작 | band() 컷오프 + pressure→MemoryStrategy 매핑 고정 |
| **G5** | struct 필드 — `StepInfo`(pressure만 언급), `DeviceTarget` enum variant 미정의(SliceSpec.hardware 참조), Backend/MemoryRegistry 내부 | DeviceTarget variant 다르면 partition·resolve 시그니처 갈림 | 구조 | StepInfo 필드 + DeviceTarget variant 확정 |
| **G6** | Backend "required floor (~4)" + compute op "..." 근사 | required vs auto-default 판단 상이 | 구조 | required/auto-default/memory-sync 계약 전수 목록 |
| **G7** | PipelineRegistry.dispatch 세부 (OneShot GC 시점, Err→panic 범위, self-filter 기전) | 동작 엣지 갈림 | 동작 | dispatch 루프 의미 명시 |

**진행 형식**: 산출물은 `arch/pipeline_stage_design_v2.md` 보완(+ 필요시 spec/41 INV/시그니처 동행). 코드 0줄. G1을 grill로 먼저 닫고 doc에 박은 뒤 G3/G4/G5 순.

---

## Landmines / 미해결 (R6)

- **리스크 트랙은 사용자 소유** — SOLID/레이어드/성능-확장성/외부기여 4기준 검토 + 교차 리스크 **R1(성능게이트↔외부기여 충돌) / R2(미션 "최소변경"↔물리 M×N 커널) / R3(safety-over-policy가 correctness 미보증→silent garbage) / R4(INV-HOTPATH-DISPATCH enforce 미정) / R5(TransformerModel 필드 누적) / R6(paired-kernel panic + fallback silent slow)** 를 이번 세션에 제시했고 **사용자가 직접 검토 중**. 다음 세션은 이 트랙을 재실행하지 말 것(중복). 단 G4/G7 구체화가 R3/R4와 닿으면 사용자 검토 결과를 먼저 확인.
- **코드 진입 금지** — 설계 구체화가 먼저. α-W(코드)는 그 다음. "Phase α-W 진입"을 구현 신호로 받지 말 것.
- **seam (c) storage-slot은 잠정**(§4.1 item1 연혁) — α-K에서 코드로 닫힘. G1 wiring과 별개.
- **3축·item1·item2·CapabilityRegistry §3.3·INV-HOTPATH 경계(coarse capability 허용)는 확정** — 재grill 대상 아님.
- **G2/G6 = 연기된 spec/INV ②장치 실행으로 해소** — 새 grill보다 spec 작성 작업에 가까움.

## 선행 문서
① 본 handoff ② 설계 SSOT `arch/pipeline_stage_design_v2.md`(3축 갱신본) ③ 용어 `/CONTEXT.md`(Flagged ambiguities=정리된 구분) ④ grill 이력 `arch/pipeline_stage_design.md`(v1 동결).

## 자기점검
- 진입 문장으로 첫 명령 가능: ✓ "설계 구체화 진입 — G1부터"
- 왜 멈췄나: ✓ 사용자 리스크 병렬 검토 + 결정성 보완 별 세션 분리
- 최대 landmine: ✓ 코드 진입 금지 + 리스크 트랙 사용자 소유(중복 금지)
- 검증: 설계 구체화라 게이트는 "doc 자기완결 + (해당 시) spec test" — 코드 게이트 아님
