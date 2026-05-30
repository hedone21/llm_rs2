# Handoff: Format 용어 정리 사이클 종결 → KV/Weight grill → Phase α-W 스레드 재개

**작성**: 2026-05-30
**HEAD**: `dc485383 docs: Format 용어 전파 — spec INV 본문 + backend_conformance + adr/0001`
**브랜치**: `master` (origin 동기화 완료, ahead 0)
**작성자**: 메인 세션 (Claude)
**다음 세션 진입 문장**: **"Phase α-W 진입 전 finalize — #12 KVCacheFormat/WeightFormat impl 시그니처부터"** (또는 곧장 "Phase α-W 진입")

---

## TL;DR

`/grill-with-docs` 로 시작한 **Format/Stage 용어 정리 사이클이 완전히 종결**됐다. 저장 형태(noun)=`Format`, 관리 동작(verb)=`Stage` 두 직교 축 확정 + 전 문서 전파 완료 (`CONTEXT.md`·v2 = 지난 세션 `771e1830`, spec INV 본문·backend_conformance·adr/0001 = 이번 세션 `dc485383`). **이 side-quest는 KV/Weight grill 스레드의 ④ 산물**이었고, 이제 그 스레드의 **본류 = Phase α-W 진입**으로 복귀한다. 멈춘 이유: 용어 정리는 끝났고 다음은 별도 결정/구현 단위(Phase α-W 진입 전 finalize grill #11/#12/#3)라 세션 분리.

---

## 진행 상태 (검증 가능)

| 항목 | 상태 | 근거 |
|---|---|---|
| `CONTEXT.md` + `pipeline_stage_design_v2.md` Format 통일 | 완료 | `771e1830` (지난 세션) |
| `spec/41` 현행 normative INV 본문 rename | 완료 | `dc485383`. INV ID 17개 보존(grep 확인), changelog/v1-pointer 동결 |
| `arch/backend_conformance_harness.md` | 완료 | `dc485383`. 잔여 old name 0 (grep 확인) |
| `docs/adr/0001` 타입명 + storage-paradigm | 완료 | `dc485383`. `KVCacheOps` 18곳·dispatch-paradigm·파일명 보존 |
| `inference_pipeline.md` / `README.md` | **편집 불요** | occurrence 전부 `>` 연혁 changelog → diff 0 |
| backlog [P2] Format 명명 통일 | **문서 prose 전부 완료** | 잔여 = 코드 `KVCacheOps` rename만 (Phase α-K 동행) |

**검증 명령** (재확인용): `grep -rn "KVCacheLayer\b" spec/41-invariants.md` → frozen changelog(532/541/542/547/579/580) + v1-pointer(585/586 원본열)만 남으면 정상.

---

## 다음 작업 (R5)

용어 정리 종결 → **KV/Weight grill 본류 = Phase α-W 진입**. 설계 단일 진실원본 = `arch/pipeline_stage_design_v2.md`. 상세 plan은 이미 존재 — 아래 두 handoff 가 0→1 진입점:
- **부모**: `.agent/todos/handoff_kv_weight_grill_2026_05_28.md` (R0 진입순서 + R3 Phase α-W Next actions + R5 미해결 Q)
- **④ 산물**: `.agent/todos/handoff_kvcachelayer_attention_into_2026_05_30.md` (④-a `attention_into` 확정, ④-b는 Phase α-K)

**Phase α-W 진입 전 필수 finalize grill** (부모 handoff R0 순서):
1. **#12 — `KVCacheFormat`/`WeightFormat` impl 시그니처 detail finalize** ← 추천 시작점. `attention_into` 정확 시그니처(`dims: AttnDims` + `scores: Option<&mut [f32]>` + `&self` vs `&mut self` interior mutability) 확정. 검증: v2 §4.1 ↔ spec INV-KVCACHELAYER-PAIRED-KERNEL 정합.
2. **#11 — Layer impl backend ref 보유 패턴** (결정 #14 `BackendExtensions` 폐기 후속).
3. **#3 — SecondaryStore** sub-grill (Phase α-W 와 동행, `arch/weights_pressure_split.md §7.5`).

그 후 **Phase α-W** (2-3주, Weight + PipelineStage 인프라 + KvBundle/WeightBundle 폐기). 종료 게이트: (a) S25 bit-identical, (b) avg_tbt Δ ≤ +3%, (c) Weight swap 정확성 회귀 0, (d) 신규 INV PASS.

**대안 트랙** (Phase α-W 안 가면): 1차 ⑤ PipelineStage 순서-안전 property test(INV-STAGE-ORDER-SAFETY 실체화) / 무관 [P2] (generate 바이너리 분할, SecondaryStore trait inversion, QCF rename).

---

## Landmines / 미해결 (R6)

- **이번 rename은 코드 0줄** — `KVCacheLayer`/`StandardLayer` 등은 *설계 명칭*이고 코드엔 grep 0건. 코드 trait명은 여전히 `KVCacheOps`. 코드 rename(`KVCacheOps→KVCacheFormat`)은 **Phase α-K(Generic→dyn) 동행** — 선행 금지(불필요).
- **changelog 동결 정책**: spec §3.28 변경요약·폐기 로그 + arch `>` 연혁 + v1(`pipeline_stage_design.md`)은 *당시 용어 보존*. 다음 세션이 grep으로 `KVCacheLayer` 발견해도 그게 frozen 영역이면 **고치지 말 것** (v1 = 결정-이력 아카이브).
- **ADR "paradigm" 두 의미**: "dispatch paradigm"(Generic↔dyn 접근법, 제목·§5·파일명) ≠ "storage paradigm"(저장형태=Format). §6 게이트 "5 KV 구성"은 Format(KIVI)+Stage(Sliding/H2O/D2O) 혼재라 'Format'이 틀려서 중립어 '구성'. 추후 누가 일괄치환하면 안 됨 — adr/0001:10 amendment 노트 참조.
- **INV ID는 opaque 안정 키**: `INV-KVCACHELAYER-*`/`INV-STAGE-LAYER-HANDLE`는 타입이 Format으로 바뀌어도 ID 유지(2026-05-30 결정, 사용자 승인). `tests/spec/` 영향 0 (해당 INV는 source-grep 검증, 전용 테스트 파일 부재).
- **미응답 thread**: ②′ Backend 80-method long-tail → capability 분리의 ADR-0002 작성 여부 (부모 handoff R0, friction-triggered 보류 중).

---

## 자기점검

| 항목 | 확인 |
|---|---|
| 진입 문장으로 첫 명령 가능 | ✓ "Phase α-W 진입 전 finalize — #12 …부터" |
| 왜 멈췄는가 | ✓ 용어 정리 종결, 다음은 별 결정/구현 단위라 세션 분리 |
| 최대 landmine | ✓ 코드 0줄(설계명만) + changelog 동결 + ADR paradigm 두 의미 |
| 검증 게이트 = 명령/수치 | ✓ grep 재확인 명령 + Phase α-W 4 게이트 수치 |
| 본문 길이 | ✓ 상세 plan은 부모 handoff 링크로 위임 (중복 회피) |
