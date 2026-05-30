# Handoff — device 3축 분리 기각 → 2축 + 실행 바탕(Fabric) 확정 + 문서 반영

**작성**: 2026-05-30
**HEAD**: `0eca0720 docs(handoff): #12 attention_into 시그니처 확정 + device-placement grill 진입`
**브랜치**: `master` (미커밋 변경 2: `CONTEXT.md` + `arch/pipeline_stage_design_v2.md`)
**작성자**: 메인 세션 (Claude)
**다음 세션 진입 문장**: **"device 2축 후속 grill 재개 — 남은 미결(Fabric 이름 / pressure_level / 성능 불변식)부터"**

---

## TL;DR

직전 handoff(`handoff_kvformat_sig_device_placement_grill_2026_05_30.md`)의 SW-1("device-placement를 3번째 축으로?")을 grill(SW-1~SW-8)한 결과 **3축을 기각하고 2축 + 실행 바탕으로 확정**했다. device(switch+partition)는 Format/Stage와 곱해지는 축이 아니라 두 축이 *그 위에서 실행되는 바탕*이다 (device 바뀌어도 Format은 따라감 = 곱 아닌 위치, deletion test 통과). switch=device 제어 Stage, partition=WeightFormat dispatch 모드, device 자원은 `Fabric`(신설)이 묶음. 확정 1~7을 `CONTEXT.md` + v2에 반영 완료. 멈춘 이유: 확정 매듭(인자 모델) + 문서 반영이 끝났고, 남은 미결 5개는 새 grill 라운드라 분리.

---

## 확정 (검증 가능) — 이번 grill 1~7

| # | 결정 | 반영 위치 |
|---|---|---|
| 1 | **2축** (Format + Stage). device = 실행 바탕(축 아님) | CONTEXT.md 도입+「실행 바탕」섹션, v2 §3.5 연혁 |
| 2 | 철학: 논리 모델(고정) ⊥ 물리 실행(=Format+Stage). device는 무대 | CONTEXT.md 「두 축」 |
| 3 | **switch = device 제어 Stage** (별도 LifecycleStage 폐기) | CONTEXT.md Stage 정의, v2 §5.1 + §3.5 연혁 |
| 4 | **partition = WeightFormat dispatch 모드 × companion backend** 곱 | CONTEXT.md 「실행 바탕」, v2 §4.2 |
| 5 | **Fabric**(신설) = BackendRegistry ⊥ MemoryRegistry + `resolve()` (UMA/discrete 캡슐화). 흩어진 4 변수(init.rs) 흡수 | v2 §3.5 + §0.3 다이어그램 |
| 6 | **인자 모델(옵션 A)**: KV handle + device(Fabric) 둘 다 register 시점 Arc 보관 + interior mutability. `StageContext` 2 field 유지 | v2 §5.1 |
| 7 | god-ctx 정리: `HandlerContext` 9필드 → handle보관/self-derive/score accessor(#17)/register config/Fabric query/profiler 분산 | v2 §5.1 (StageContext 2 field 근거) |

**핵심 통찰** (재현용): device를 ctx로 넘길 뻔했으나(switch로 변함→stale 우려), KV handle처럼 `Arc + interior mutability`면 switch가 내부만 mutate→같은 Arc 든 Stage 자동 최신→register 보관 유지(god ctx 회피). KV와 device 대칭.

---

## 다음 작업 (R5) — 남은 미결 grill

1. **Fabric 이름 확정** — Substrate/Placement/ExecutionContext 모두 사용자 기각. v2/CONTEXT.md에 "이름 미확정 — Fabric 잠정"으로 박아둠. 확정 시 두 문서 일괄 치환.
2. **시스템 상태(pressure_level) 처리** — `StageContext` 2 field 확정했으니, pressure_level(매 step 변하는 신호)을 register Arc 보관(device 대칭) vs `StepInfo` 값 중 결정.
3. **성능 불변식 명문화** — "layer=정적(Option+직접분기) / step=Box\<dyn\> / 경계=자유, **hot path에 trait object 금지**"를 spec INV로 (SW-6 2번). "성능 저하 없이" north star의 검증 게이트.
4. **switch의 KV migrate 메커니즘** — switch Stage가 보관 handle(`Arc<dyn KVCacheFormat>`) 내부를 migrate하는 method = **#12 KVCacheFormat mutation에 추가될지** (§4.1 mutation 3 외).
5. **partition wiring** — `LayerDispatch::Partition` × `Fabric.resolve(companion)` 구체 연결 (현재 layer가 `Option<PartitionContext>` 정적 소유, forward_gen.rs:1145).

---

## Landmines / 미해결 (R6)

- **코드 0줄** — 전부 설계. `Fabric`/`StageContext`/switch Stage 모두 미구현. 구현은 Phase α-W/α-K.
- **Fabric 이름이 문서에 가칭으로 박힘** — v2 §3.5 코드블록 주석 + CONTEXT.md에 "이름 미확정". 다음 세션이 grep `Fabric` 하면 잠정명임을 인지할 것.
- **미커밋 2 파일** — `CONTEXT.md` + `arch/pipeline_stage_design_v2.md`. 커밋은 사용자 명시 요청 시. 무관 untracked(`.antigravitycli/`·`.claude/scheduled_tasks.lock`·microbench 산출물)는 **절대 커밋 금지**.
- **§8 불변식 요약 / Front-door 표(§0.4) 미갱신** — 성능 불변식(미결 3)이 확정되면 §8에 INV 추가 필요. 지금은 미확정이라 의도적으로 안 건드림.
- **#12 attention_into 시그니처는 확정 유지** — 직전 handoff의 `attention_into(&self, q, out, dims)` (backend/scores 제거)는 이번 2축 결정과 무관하게 그대로. switch가 same-device 아닌 유일 예외지만 Stage라 attention_into 경로 밖.
- **선행 문서**: 직전 `handoff_kvformat_sig_device_placement_grill_2026_05_30.md`(SW-1 진입) + 부모 `handoff_kv_weight_grill_2026_05_28.md`. 설계 단일원본 = `arch/pipeline_stage_design_v2.md`, 용어 = `/CONTEXT.md`.

---

## 자기점검

| 항목 | 확인 |
|---|---|
| 진입 문장으로 첫 명령 가능 | ✓ "device 2축 후속 grill 재개 — 남은 미결부터" |
| 왜 멈췄는가 | ✓ 확정 매듭+문서 반영 종료, 남은 미결은 새 라운드 |
| 최대 landmine | ✓ Fabric 이름 가칭 + 코드 0줄 + 미커밋 2파일(무관 untracked 격리) |
| 검증 게이트 | ✓ 미결 3(성능 불변식)이 "성능 저하 없이"의 spec test 게이트 |
| 본문 길이 | ✓ 상세는 CONTEXT.md/v2 §3.5 링크 위임 |
