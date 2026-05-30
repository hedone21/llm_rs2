# Handoff — device 2축 후속 grill 종결 (Hardware 명명 + pressure + 성능 불변식) + 문서 반영

**작성**: 2026-05-31
**HEAD**: `40a94873 docs(arch): device 3축 분리 기각 → 2축 + 실행 바탕(Fabric) 확정` (본 작업 커밋 전)
**브랜치**: `master`
**작성자**: 메인 세션 (Claude)
**다음 세션 진입 문장**: **"device 2축 후속 — 남은 미결(switch KV migrate / partition wiring)부터, 또는 Phase α-W 진입"**

---

## TL;DR

직전 handoff(`handoff_device_2axis_fabric_2026_05_30.md`)의 남은 미결 5개 중 **3개(Fabric 이름 / pressure_level / 성능 불변식)를 grill로 종결**하고 `CONTEXT.md` + v2 에 반영했다. 멈춘 이유: 이 3개가 한 grill 라운드의 자연 종결점이고, 남은 2개(switch KV migrate / partition wiring)는 #12 KVCacheFormat mutation·`LayerDispatch` 구현과 엮인 별도 라운드라 분리.

---

## 확정 (검증 가능) — 이번 grill

| # | 결정 | 반영 위치 |
|---|---|---|
| 1 | **Fabric → `Hardware`** 명명 확정 (Substrate/Placement/ExecutionContext/Fabric 검토 후). `RuntimeResources`(pressure-owned, `pressure/weights/setup.rs`)와 별개 | v2 §3.5 + §0.3 mermaid + §4.2 + §5.1, CONTEXT.md 「실행 바탕」 |
| 2 | **pressure = `Pressure(0-100)` 단일 scalar** + pluggable `PressureSource`(Manager 기본/Local/3rd-party). source=construction 보유, value=`StepInfo` read-only 값. **anymap·multi-field SystemState 기각** | v2 §5.1 pressure 단락 + §0.4 front-door, CONTEXT.md 「Pressure/PressureSource」 |
| 3 | **`INV-HOTPATH-DISPATCH`** (신규) — §1.1 operationalize. 3-tier(layer 정적/step `Box<dyn>`/boundary 자유) + static grep + avg_tbt Δ≤+3% | v2 §1.1 + §8 |

**핵심 통찰** (재현용):
- **pressure가 device와 달리 StepInfo 값인 이유 = mutation 축**. device는 switch(Stage)가 mutate→Arc 보관. pressure는 *어떤 Stage도 mutate 안 함*(드라이버 read-only 스냅샷)→StepInfo. 대칭이 깨지는 정확한 지점.
- **anymap 기각 = deletion test**. 사용자 "확장 가능"의 진짜 축은 *신호 종류*가 아니라 **값의 source**(manager/local/3rd-party). source는 trait 하나로 끝나고 carrier는 단일 scalar로 안정 — 인터페이스가 숫자 하나라 더 못 안정적이라 carrier를 열 이유 없음(deep module).
- **성능 불변식의 진짜 비용 = vtable(~ns) 아니라 인라인·Format 특화·벡터화 장벽**. layer tier dispatch를 construction-held concrete handle(§3.3/§4.1)로 흡수해 특화 보존.

---

## 다음 작업 (R5) — 남은 미결 2개 (직전 handoff 4·5)

1. **switch의 KV migrate 메커니즘** — switch Stage가 보관 handle(`Arc<dyn KVCacheFormat>`) 내부를 migrate 하는 method 가 **#12 KVCacheFormat mutation 3종**(write_kv/write_kv_batch/compact)에 추가될지. 현 구현 = `pressure/kv_migrate.rs::migrate_kv_caches`(버퍼 재생성, UMA zero-copy tag swap). 검증: §4.1 mutation set ↔ switch 경로 정합.
2. **partition wiring** — `LayerDispatch::Partition`(§4.2, **코드 미존재**) × `Hardware.resolve(companion)` 구체 연결. 현 layer = `Option<PartitionContext>` 정적 소유(`transformer_layer/mod.rs:228`, `forward_gen.rs:1145` 직접분기). `LayerDispatch` enum 신설 시 이 정적 소유와 어떻게 합류할지.
3. **그 후 Phase α-W 진입** — Weight + PipelineStage 인프라 + Bundle 폐기 + CapabilityRegistry (v2 §9). 종료 게이트: S25 bit-identical + avg_tbt Δ≤+3% + Weight swap 정확성 회귀 0 + 신규 INV PASS.

---

## Landmines / 미해결 (R6)

- **코드 0줄** — 전부 설계. `Hardware`/`PressureSource`/`Pressure`/switch Stage/`LayerDispatch` 모두 미구현. 구현은 Phase α-W/α-K.
- **`INV-HOTPATH-DISPATCH` spec/41 normative + tests/spec/ 는 의도적 연기** — Phase α 구현 시점. 코드 0줄 + spec ID 작업엔 test 필수 규칙(`feedback_spec_tests_required`)이라 testless spec INV를 지금 안 박음. 현재는 **v2 §8(설계 SSOT)에만** 존재.
- **코드 rename(`Fabric`→`Hardware` 자체가 코드엔 없음)** — 흡수 대상 = `session/init.rs:35-46` 의 4 변수(`cpu_backend_arc`/`gpu_backend_arc`/`cpu_memory_arc`/`gpu_memory_arc`)+`is_gpu`. Phase α-W에서 `Hardware`로 통합.
- **잔여 "Fabric" 문자열은 정상** — v2 §3.5 연혁 + CONTEXT.md "구 Fabric" 포인터뿐(grep 확인). 운영 참조 0.
- **pressure ripple 미적용** — `PressureLevel` enum(`pressure/mod.rs`)→`Pressure(0-100)`+`band()` 전환은 설계만. 영향 사이트 `swap_handler.rs:371`/`cache_manager.rs:153`/`MemoryStrategy`/`d2o_handler`.
- **wire format 단순화 후보** — manager 연동 시 닫힌 `SystemSignal` enum(`shared/src/lib.rs:125`, 4 variant) 대신 통합 0-100 하나만 보내도 됨. manager 쪽 리팩토링이라 별도.
- **선행 문서**: 직전 `handoff_device_2axis_fabric_2026_05_30.md` + 부모 `handoff_kvformat_sig_device_placement_grill_2026_05_30.md`. 설계 SSOT = `arch/pipeline_stage_design_v2.md`, 용어 = `/CONTEXT.md`.

---

## 자기점검

| 항목 | 확인 |
|---|---|
| 진입 문장으로 첫 명령 가능 | ✓ "남은 미결(switch KV migrate / partition wiring)부터" |
| 왜 멈췄는가 | ✓ 3 미결 grill 종결 + 문서 반영 끝, 남은 2개는 구현-엮인 별 라운드 |
| 최대 landmine | ✓ 코드 0줄 + INV spec/test 연기 + pressure ripple 미적용 |
| 검증 게이트 | ✓ INV-HOTPATH-DISPATCH = avg_tbt Δ≤+3% + S25 bit-identical |
| 본문 길이 | ✓ 상세는 v2 §3.5/§5.1/§8 링크 위임 |
