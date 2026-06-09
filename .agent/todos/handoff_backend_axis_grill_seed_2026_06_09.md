# Handoff(seed): Backend 축(3번째 axis) GATE-C v3 grill — KIVI 추동

**작성**: 2026-06-09
**HEAD**: `15fec965 docs(handoff): opaque floor 최적화 (A: descriptor 인식) 결과 + Backend v3 연결`
**브랜치**: master (미푸시)
**작성자**: 메인 세션 (다음-세션 grill seed)

**다음 세션 진입 문장**: **"Backend 축(3번째 axis) GATE-C v3 grill 진행 — KIVI(`KiviAttentionBackend`, 이미 존재·OpenCL-only 정적)를 dlopen-pluggable capability 로 만드는 설계. 시작점 = 기존 `capability/CapabilityRegistry` + sub-trait 2종(blank slate 아님)."**

> 이 문서는 **grill seed** — 다음 세션이 grill-me/grill-with-docs 로 Backend 축 설계를 walk 할 때의 진입점·분기 트리·앵커. 이번 세션은 grill 미시작(사용자: 다음 세션 진행).

---

## TL;DR

북극성 3축(Stage ⊥ Format ⊥ Backend) 중 **Stage·Format dlopen 은 완주**(GATE-C v1/v2 + opaque + device + descriptor-recognition opt). 남은 **Backend 축**을 다음 세션에 grill. **추동 use-case = KIVI**(사용자 지정): 2-bit asymmetric KV quant(ICML 2024)는 그 가치가 **custom fused 커널**(per-channel K ⊥ per-token V + residual + flush dequant+attention)이라 Format 축(순수 descriptor + 엔진-소유 floor, ADR-0005 D3/D4)으로 **표현 불가** → Backend-capability 가 정확한 자리. **★정정**: Backend 축은 blank slate 아님 — `CapabilityRegistry`(typed anymap, α-W-4) + `KiviAttentionBackend`/`GpuScoreAccess` sub-trait 이미 존재. grill = "trait 0에서 설계"가 아니라 **"기존 capability sub-trait 을 dlopen-pluggable 로"**.

---

## 시작 상태 (코드 실측 — blank slate 아님)

| 요소 | 위치 | 상태 |
|---|---|---|
| `CapabilityRegistry`(backend→capability handle, typed anymap, construction-time lookup, hot-path 0) | `engine/src/capability.rs` | ✅ α-W-4 신설 |
| `KiviAttentionBackend` capability sub-trait | `engine/src/capability/kivi_attention.rs` | ✅ `has_kivi_attn_kernel(bits)` · `is_nosub_device()` · `attention_gen_kivi(q, qk_buf, qv_buf, res_k, res_v, out, …)` 2-bit fused dequant+attention |
| `GpuScoreAccess` capability sub-trait (두번째 vehicle) | `engine/src/capability/gpu_score.rs` | ✅ |
| 현재 dispatch | `backend.rs:1305 as_kivi_attention()`(legacy downcast) + `caps.get::<dyn KiviAttentionBackend>()`(registry) | KIVI 커널 = **OpenCL backend 정적 보유** |
| 설계 SSOT | `arch/pipeline_stage_design_v2.md` §3.3 | "새 backend 능력(fused kernel) = capability sub-trait + CapabilityRegistry 등록, in `backend/<hw>/`" |

**즉 "BackendCapability = name() 1개 미확정"(구 핸드오프)은 부정확**: `name()` 은 base `Backend` trait. capability 축은 `CapabilityRegistry` + 위 sub-trait 들로 이미 실현돼 있고 rich method 보유. Backend v3 = **이들을 `.so` 로 plugin 화**(Stage/Format 처럼).

---

## grill 분기 트리 (다음 세션이 walk 할 질문들)

1. **KIVI 축 귀속 — Backend 단독 vs Format+Backend 번들?** KIVI 표현(2-bit per-channel K ⊥ per-token V + FP32 residual + flush)은 `KVLayoutDesc` 어휘(block_elems/bits/scale_layout/packing) **밖**(K/V 그룹축 상이 + residual + 동적 flush). → 표현을 capability(KiviCache 가 storage 소유)가 갖고 Backend 단독? 아니면 Format 확장 동반? **cross-axis 번들이면 ADR-0010 multi-vtable bundle ABI 가 이미 지원**(한 `.so` 가 Format vtable + Capability vtable).
2. **capability ABI 형태 — 가장 어려운 marshalling.** `attention_gen_kivi` 는 live `&Tensor` 6+개(GPU `Mem` 핸들 래핑) 인자. Stage=`PlanAbi`(POD plan), Format=POD descriptor(콜백 0) 였으나 **Backend=live backend 자원**(GPU 버퍼/queue) C-ABI 통과 → 3축 중 최난도. fn-ptr vtable + opaque handle 어떻게?
3. **GPU 커널 delivery via `.so`.** plugin 의 가치가 곧 `.cl` 커널. host 의 OpenCL context/queue 에 대해 어떻게 컴파일? (source string 전달? plugin 이 libOpenCL 직접 링크해 host context 로 빌드? precompiled binary?)
4. **backend×capability 바인딩.** capability 는 특정 backend(OpenCL) 대상. plugin 이 host backend 의 device/context/queue raw 핸들을 얻어야 함 → Stage/Format 보다 깊은 결합. ABI 가 backend 내부를 노출?
5. **GATE-C 봉투 재사용.** capability vtable = ADR-0010 envelope 의 3번째 vtable type(`register_kv_capabilities_v2`)? + dlopen 시 `CapabilityRegistry` 등록 배선.
6. **device-only.** 전부 GPU(OpenCL/Adreno) → host 검증 불가, S25 device gate 가 acceptance.

---

## Landmines / 주의

- **Backend 축 ABI 는 3축 중 최난도**: Stage(plan POD)·Format(descriptor POD, 콜백 0) 와 달리 live GPU 자원(Tensor=Mem, queue, context)을 `.so` 경계로 넘긴다. workspace `panic="abort"`(ADR-0009 D7) + fat-pointer trait object 문제(ADR-0009 D2) 그대로 + GPU 자원 수명/소유권 추가.
- **KIVI 표현 ≠ Format-expressible**: `encode_via_descriptor`/`KVLayoutDesc` 로 KIVI 2-bit 못 적음 → "Format 확장 먼저"의 유혹 주의. KIVI storage 는 `KiviCache`(자체 타입)가 이미 소유 — capability 는 **커널만** 공급할 수도(번들 불요). grill 1번이 이걸 가른다.
- **opaque floor-fast 와의 관계**: 직전 세션 결론 — floor-fast(novel family) 동기는 약함(runnable novel family 부재). KIVI 는 그것과 **다른** Backend-axis 가치(custom 커널 공급, floor 무관). 즉 Backend v3 의 정당성은 floor-fast 아니라 **KIVI 같은 custom-compute capability 의 plugin 화**.
- **encode 확장(Q8_0/q4_1)은 추가 안 함**(직전 세션 결정): 기존에도 미지원 + 수요 부재 + 추측. KIVI grill 과 무관.
- **engine/Cargo.toml drift**(score_readback)는 계속 미커밋 유지.

---

## 참조
- 코드 앵커: `engine/src/capability.rs`(Registry) · `capability/kivi_attention.rs`(KiviAttentionBackend) · `capability/gpu_score.rs` · `pressure/kivi_cache.rs`(KiviCache storage, 2-bit/residual/flush) · `pressure/kivi_format.rs`(KIVIFormat) · `session/forward/kivi_forward.rs` · `backend.rs:1305`(as_kivi_attention dispatch) · `session/init.rs`(CapabilityRegistry build).
- 설계 SSOT: `arch/pipeline_stage_design_v2.md` §3.3(Capability over god-trait, CapabilityRegistry) · §1.3.
- 선행: ADR-0010(multi-vtable bundle ABI — 봉투/dispatcher 재사용 가능) · ADR-0009(D2 fn-ptr ABI·D6 device landmine·D7 panic=abort) · ADR-0005(D6 3축 평행 registry, 단일 병합 금지). Stage/Format 완주 = `handoff_format_axis_production_e2e_2026_06_09.md` + `handoff_gate_c_v3_multivtable_2026_06_09.md`.
