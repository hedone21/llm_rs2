# Handoff: Format · Backend Capability plugin 통합 설계 확정(ADR-0005) → crate 단계 구현

**작성**: 2026-06-06
**HEAD**: (이 커밋) `docs(adr): ADR-0005 Format·Backend capability plugin 통합`
**브랜치**: master
**작성자**: 메인 세션 (grill-me 세션)

**다음 세션 진입 문장**: **"ADR-0005 crate 단계 구현 — KV_FORMATS + BACKEND_CAPABILITIES registry 신설부터"**

---

## TL;DR

grill-me 6문답으로 **Format·Backend capability 를 plugin 화하는 방법·인터페이스를 확정**(→ ADR-0005 Accepted). 핵심: 세 축을 **병합하지 않고**(직교 불변) 같은 plugin 관용구로 확장 — Format = `repr(C)` descriptor(데이터) + manage trait, Backend = 커널 소유(이미 dtype-dispatch), generic floor + 특화 opt-in, 3축 평행 linkme registry. **crate 완전 분리 먼저 → `.so` 나중**(crate 경계가 "엔진 타입 0" 컴파일러 강제 = de-risk). **멈춘 이유**: 설계·인터페이스 확정 완료 — 구현은 별도 세션(코드 변경 0인 ADR 단계 종료).

---

## 진행 상태

### 확정 결정 (ADR-0005, grill 6문답 락)

| # | 결정 |
|---|---|
| D1 | "통합" = 축 병합(a) **기각** / (b) 같은 패턴·패키징 + (c) 커널 seam 채택 |
| D2 | tier-driven hybrid ABI; **layer-tier read = descriptor not code** (repr(C)) |
| D3 | Format plugin = `KVLayoutDesc`(데이터) + `KVFormat`(manage); compute dissolve |
| D4 | Format=데이터, **Backend 가 커널 소유**(이미 `match dtype`, cpu/common.rs:76), 인터페이스만 직교 |
| D5 | **generic floor + 특화 opt-in**; descriptor 어휘=block-quant family; mxfp4/codebook=escape |
| D6 | 등록 = **3축 평행 linkme registry**(KV_CACHE_STAGES ✅ / KV_FORMATS·BACKEND_CAPABILITIES 신설); 단일 registry 병합 금지(§3.3.1) |
| D7 | **crate 분리 먼저 → `.so` 나중**; crate 단계 게이트 **L1**(layer-tier 타입 repr(C) now) + **L2**(hot call api 경계 안 넘기 — LTO 가 숨김, `.so` 가 회귀) |

### 측정 / 검증 (이 세션, 이미 완료 — 재실행 불요)

| 항목 | 값 |
|---|---|
| observe-hook PoC 등가성 (x86+ARM) | bit-identical (4 config, max\|Δ\|=0) |
| observe-hook dispatch 비용 | x86: RefCell +407%(벡터화 차단, asm 확증) / `&mut self` ~0% / ARM: 전 변종 ~1.1% TBT(scalar) |
| OpenCL readback (S25 Adreno 830) | custom plugin copy ~1.1ms(=3.5% TBT@full ctx); **rpcmem zero-copy 기각**(map 3-6× 느림, 코히런시 깨짐) |
| ADR-0005 | Accepted, README 인덱스 갱신 |

---

## 다음 작업 (crate 단계 구현, 순서대로)

1. **`KV_FORMATS` + `BACKEND_CAPABILITIES` registry 신설** (linkme, `stage_registry.rs` 동형) → 검증: `find_*`/`registered_names` 단위 테스트 + release fat-LTO linkme 생존(ADR-0003 §4 self-test)
2. **`KVLayoutDesc`(repr(C)) + `KVFormat` trait** 를 `format/kv_cache_format.rs` 에 (L1 게이트: repr(C) 준수) → 검증: `cargo build` + clippy clean
3. **generic floor** — backend `_ => dequant_via_descriptor → matmul_transposed_f32` (공유 엔진 헬퍼) + 기존 특화 arm 보존 → 검증: 새 descriptor(예: q5 mock)가 전 backend 동작 + 기존 q4_0 bit-identical 무회귀
4. **`KVCacheFormat` 해체** — compute(`write_kv`/`attention_into`)를 backend dispatch 로, manage(`compact`)만 `KVFormat` 잔류 → 검증: **L2 게이트** = plan-path bit-identical + S25 TBT Δ≤+3%(§8)
5. (선택) GpuFold = backend capability 첫 instance — observe-hook PoC(`/tmp/llm_rs2_poc_hook/`) → `FoldRunner<dyn GpuFold>` (`GpuScoreAccumulator` 일반화)

### 위임 prompt (선택)

> **에이전트**: `architect`(설계 정착 spec/arch) → `senior-implementer`(registry/trait/floor 구현)
> **권한**: `crates/`, `engine/src/format/`, `engine/src/backend/`, `engine/src/pressure/eviction/stage_registry.rs`
> **첫 명령**: "ADR-0005 D6 — KV_FORMATS + BACKEND_CAPABILITIES registry 를 KV_CACHE_STAGES 동형으로 신설. L1(repr(C)) 준수."

---

## Landmines / 미해결 / 안 가본 길

- **L2(최대 함정)**: fat-LTO 가 crate 단계에서 cross-crate hot 경계 call 을 **인라인해 비용을 숨김** → `.so` 가 인라인 제거 → perf 회귀. **"LTO 켜니 빠르다" 신뢰 금지** — §8 게이트(plan-path bit-identical + TBT Δ≤+3%)로 crate 단계 검증. layer-tier call 0 across api 경계.
- **§3.3.1 단일 registry 금지** — "4 메커니즘 1 registry 수렴"은 폐기된 오해(`as_any` capability lookup 0건). 3축 분리 유지.
- **descriptor 어휘 한계** — block-quant family(block_elems/bits/scale_layout/packing)만. mxfp4 shared-exponent·codebook·sparse 는 **escape**(특화 opt-in, floor 안 거침). 어휘 넓히지 말 것.
- **forward_gen_fmt(cold) ↔ plan path(hot) 분기** — production GPU hot = `backend/opencl/plan.rs::execute`(plan path). `forward_gen_fmt` 의 `Arc<dyn KVCacheFormat>` 는 cold/fallback(§8 규정 안, §9.1). `KVCacheFormat` 해체 시 이 분기 회귀 주의.
- **`.so` 전환은 전부 deferred** — manifest/매크로 dual-wiring, C-ABI entry, opaque handle shim, Vec→ptr+len, catch_unwind, allocator. crate 단계엔 안 만듦(L1·L2만 지키면 기계적).
- **미커밋 실험 아티팩트**: `engine/microbench/score_readback.rs` + Cargo.toml bin 엔트리(OpenCL readback 측정, A/B/C/D 4-way), `/tmp/llm_rs2_poc_hook/`(observe-hook CPU PoC). 보존/폐기 미결정.

---

## 참조

- **SSOT**: `docs/adr/0005-format-backend-capability-plugin-unification.md` (본 grill 6결정 정본)
- 선행: ADR-0003(확장 메커니즘=정적 crate+linkme), ADR-0004(KVCacheStage plan-returning), CONTEXT.md(3축 직교), `arch/pipeline_stage_design_v2.md` §8(INV-HOTPATH-DISPATCH)/§3.3(CapabilityRegistry)/§3.3.1/§2.1(format.rs 배치)
- 현 코드 앵커: `format/kv_cache_format.rs:57`(KVCacheFormat), `cpu/common.rs:76`(dtype dispatch), `pressure/eviction/stage_registry.rs`(KV_CACHE_STAGES 동형 모델), `backend/opencl/gpu_score.rs:137`(런타임 .cl JIT 선례)
