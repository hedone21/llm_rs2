# Handoff — 설계 구체화 (G1 완료 → G3 다음)

**작성**: 2026-06-01 (G1 grill+문서화 후 갱신)
**HEAD**: `3b12e490 docs(arch): §3.6 wiring 표준 신설 — capability handle 도달 경로 (G1 공백 해소)` (= origin/master, push 완료)
**브랜치**: `master`
**작성자**: 메인 세션 (Claude)
**다음 세션 진입 문장**: **"설계 구체화 진입 — G3(타입 배치)부터"**

> 설계 SSOT = `arch/pipeline_stage_design_v2.md`. **코드 아님 — 설계 문서 구체화 트랙.** "Phase α-W 진입"을 구현 신호로 받지 말 것.

---

## TL;DR

3축 설계 grill 종결(`287b8668`) 후 설계 문서를 두 렌즈로 검토: (1) **리스크**(R1~R6) → **사용자 직접 검토 중**(이 트랙 건드리지 말 것). (2) **구현 결정성**(공백 G1~G7). **G1(construction wiring) 이번 세션 완료** — grill로 4결정(Q1~Q4) 닫고 `arch/pipeline_stage_design_v2.md §3.6 Wiring`에 박음(`3b12e490`, push 완료). 다음 = **G3(타입 배치)부터 grill→문서화**. G4는 사용자 risk 트랙(R3/R4) 결과 확인 후. 멈춘 이유: 단계(G1) 완료 커밋 + 인계.

---

## G1 완료 (2026-06-01) — 재grill 금지

`§3.6 Wiring — capability handle 도달 경로` 신설로 닫힌 4결정 (상세 = 문서 §3.6 + 연혁):

- **Q1 획득** = locator-at-owner: 도메인 소유자가 *자기 생성 시점*에 registry에서 *자기 것만* pull(조립자 god-wiring 회피). hot per-layer는 resolved handle을 forward-args 관통 + `Option` 직접분기(per-forward lookup 0). 플러그인 천장 = **tier-2**(새 파일+등록≤1줄, data-driven). tier-1 자동발견 미도입(§5.3 "순서=사용자 책임" 보존, 미래 escalation만).
- **Q2 수명** = session-lived: register 후 `Arc<CapabilityRegistry>` freeze(read-only). build-time + 런타임 IPC 양쪽 pull(런타임 stage 생성 지원). `Hardware`와 대칭.
- **Q3 forward home** = capability 성격이 거처 결정: **format 관심사 → format 객체 보유**(KIVI→`KIVIFormat`, forward `attention_into` + pressure `update_gpu`가 동일 객체 공유 → cross-domain 단일화). **format-agnostic → execution 주인(`DecodeLoop`)** args 관통. **model은 capability 보유 0**(capability=execution 관심사, model=data). ※초기 "model-필드+args" 추천을 사용자 push back으로 역전 — KIVI는 §4.1 `attention_into` 캡슐화로 대안 A/B 둘 다 증발.
- **Q4 토폴로지**: freeze ≺ 모든 pull. model ⊥ caps. `SessionInitCtx` 출생 → `DecodeLoop` Arc 보유.
- **산출**: §3.6 + Mermaid 2(조립 시퀀스/cross-domain) + §8 `INV-CAP-WIRING`(spec/test는 Phase α 연기) + §0.5/§3.3/§5.1 교차참조.
- **코드 적용 시점**: Phase α-W(`CapabilityRegistry` 신설 + `SessionInitCtx`/`Hardware` 정리 + `KiviCache.gpu_backend`→`Arc<dyn KiviAttentionBackend>` 좁히기) / α-K(`KIVIFormat` 확립). **아직 engine/ 코드 변경 0.**

---

## 다음 작업 — G3·G4·G5 (게이트도 spec도 안 잡는 진짜 공백)

**진행 형식**: 각 공백을 grill로 닫고 `arch/pipeline_stage_design_v2.md` 보완(역할/의도 먼저 + `> 연혁` 끝). 코드 0줄. G3 → G4 → G5 순.

| | 미명세 | 갈림 | 보완 |
|---|---|---|---|
| **G3** ★다음 | 타입 배치(파일) — `CapabilityRegistry`/`Hardware`/`StepInfo`/`Pressure` 위치, `stages/{kv,weight,system}/` 매핑 | 같은 타입 다른 파일 → 모듈그래프 갈림 | type→file 배치표 (§2 레이어 표 확장) |
| **G4** | 수치/임계 — `Pressure(0–100)→band()` 컷오프, pressure→action 매핑 | 컷오프 다르면 **다른 압력에서 eviction 발동 = 동작 상이** | band() 컷오프 + pressure→MemoryStrategy 고정. **R3/R4 사용자 검토 결과 먼저 확인** |
| **G5** | struct 필드 — `StepInfo`(pressure만 언급), `DeviceTarget` enum variant 미정의(`SliceSpec.hardware` 참조) | variant 다르면 partition·resolve 시그니처 갈림 | `StepInfo` 필드 + `DeviceTarget` variant 확정 |

G2(trait 시그니처)·G6(Backend required floor)·G7(`PipelineRegistry.dispatch`)는 연기된 spec/41 INV+test ② 실행으로 자연 해소 — 새 grill보다 spec 작성 작업.

---

## Landmines / 미해결

- **리스크 트랙 = 사용자 소유** — R1(성능게이트↔외부기여) / R2(최소변경↔M×N 커널) / R3(safety-over-policy↔correctness 미보증) / R4(INV-HOTPATH-DISPATCH enforce 미정) / R5(TransformerModel 필드 누적) / R6(paired-kernel panic + fallback silent slow). **재실행 금지(중복)**. 단 **G4 구체화 전 R3/R4 결과 확인**.
- **G1 재grill 금지** — §3.6 확정. Q1~Q4 위 기록.
- **코드 진입 금지** — 설계 구체화가 먼저. α-W(코드)는 그 다음.
- **확정/잠정 — 재grill 대상 아님**: 3축·item1·item2·`CapabilityRegistry` §3.3·INV-HOTPATH 경계(coarse capability 허용)는 확정. seam (c) storage-slot(§4.1 item1 연혁)은 α-K에서 코드로 닫히는 잠정.
- **working tree에 `grill-me`/`grill-with-docs` 스킬 파일 deletion(`D`) 표시** — 이번 세션 작업 아님, 커밋 안 함(커밋 금지 대상). 의도 불명 — 다음 세션/사용자 확인 필요.

## 선행 문서
① 본 handoff ② 설계 SSOT `arch/pipeline_stage_design_v2.md`(§3.6 신규) ③ 용어 `/CONTEXT.md` ④ grill 이력 `arch/pipeline_stage_design.md`(v1 동결).

## 자기점검
- 진입 문장으로 첫 명령 가능: ✓ "설계 구체화 진입 — G3(타입 배치)부터"
- 왜 멈췄나: ✓ G1 완료 커밋+인계, G4는 사용자 risk 트랙(R3/R4) 대기
- 최대 landmine: ✓ 코드 진입 금지 + 리스크 트랙 사용자 소유 + G1 재grill 금지
- 검증: 설계 구체화라 게이트 = doc 자기완결 + (해당 시) spec test — 코드 게이트 아님
