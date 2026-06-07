# ADR-0005: Format · Backend Capability 확장 — plugin 패턴 통일 (축 병합 아님; descriptor + backend-owned kernel + 3축 평행 registry)

> **Status**: Accepted
> **Date**: 2026-06-06
> **Decision-makers**: 사용자 + 메인 세션 (grill-me 세션 "Format·Backend Capability 통합", observe-hook/GpuFold 조사 후속)
> **Selected**: Stage·Format·Backend capability 를 **같은 plugin 관용구**(thin trait + linkme/loader 등록 + engine harness)로 확장하되, **세 축을 병합하지 않는다**. Format = `repr(C)` layout descriptor(데이터) + step-tier manage trait, Backend = 커널 소유(이미 dtype-dispatch), 둘의 결합점(M×N 커널)은 backend 내부에 격리. 등록은 3축 평행 linkme registry. **crate 완전 분리 먼저 → `.so` 나중**, crate 단계 게이트 L1·L2.
> **Related**: ADR-0003(확장 메커니즘 = 정적 crate + linkme + 런타임 `.so` 보류), ADR-0004(`KVCacheStage` plan-returning trait), `/CONTEXT.md`(3축 직교 stage ⊥ format ⊥ hardware), `arch/pipeline_stage_design_v2.md` §8(INV-HOTPATH-DISPATCH) / §3.3(CapabilityRegistry) / §3.3.1(단일 registry 병합 기각)

---

## 1. Context

궁극 목표: **Stage·Format·Backend(capability) 를 모두 `.so` plugin 으로**. 달성 어려우면 중간 단계로 **별도 crate 분리**(진행 중 — Stage 는 ADR-0003/0004 + `crates/{technique-api,techniques}` 로 완료). 본 ADR 은 **Format 과 Backend Capability** 를 같은 궤도에 올리는 방법과 인터페이스를 확정한다.

**발단** (이 세션 observe-hook 조사): KV-eviction 커스텀 fold 의 OpenCL readback 비용을 없애는 길로 **GpuFold**(plugin 이 `.cl` 소스를 싣고 엔진이 런타임 JIT — `gpu_score.rs:137` 이 자기 `score_reduce.cl` 에 이미 쓰는 메커니즘)를 설계했다. 이것이 "plugin 이 backend capability 를 확장"의 canonical instance 임이 드러났고, "Format·Backend 도 같은 방식이냐"로 확장됐다.

**충돌하는 불변·제약**:
- **3축 직교**(CONTEXT.md): stage ⊥ format ⊥ hardware, 인터페이스 비용 M+N+K(곱 아님). `(format × hardware)` 커널은 환원 불가 M×N 이나 **`backend.matmul(q4_tensor)` 내부 dispatch 에 격리**(CONTEXT.md §"M×N").
- **§8 INV-HOTPATH-DISPATCH**: layer-tier(N_layers×/token, production hot = plan path) = 정적만, **`dyn` 금지**(비용은 vtable 가 아니라 인라인·Format 특화·벡터화 장벽). step(1×/token) = `Box<dyn>` OK. boundary = 자유.
- **§3.3.1**: "4 메커니즘을 1 registry 로 수렴" 은 폐기된 오해(`as_any` capability lookup 0건). 단일 통합 registry = 과설계.
- **`.so` C-ABI**: Rust trait/Vec/`Box<dyn>` 는 ABI-불안정 → 경계엔 `repr(C)`/opaque handle/fn-table 필요.

---

## 2. Decision

### D1 — "통합" = 축 병합 아님. (b) 같은 패턴·패키징 + (c) 커널 seam.
Format 과 Backend 은 **병합하지 않는다**(직교 위반 + §3.3.1). "통합"의 실체:
- **(b)** 세 축이 같은 plugin 관용구(thin trait + 등록 + engine harness)를 쓰고, **한 plugin 아티팩트가 stage/format/capability 아무 조합이나** 싣는다. registry 는 **축별 분리**.
- **(c)** 둘의 결합점 = `(format × hardware)` 커널 행렬. 진짜 설계 난제이나 D4 에서 backend 내부로 흡수.

### D2 — tier-driven hybrid ABI. layer-tier read = descriptor not code.
확장점 인터페이스의 ABI 형태는 **호출 tier 로 결정**한다:
- **layer-tier 경계 데이터** → `#[repr(C)]` / zero-copy handle (FFI-stable now). 선례 `TensorKind #[repr(u32)]`/`TensorShape #[repr(C)]` (technique-api lib.rs:25,42).
- **step/boundary-tier** → Rust-native(`Box<dyn>`/`Vec`); `.so` 전환 시 **기계적 C-ABI shim**(저빈도 표면에만, locality).

따름 정리: **layer-tier read 는 코드가 아니라 descriptor** — Format plugin 이 hot-path 에서 코드를 돌리는 대신 layout descriptor 만 싣고 엔진이 monomorphized read 를 돌린다 → `.so` 호출이 layer 마다 안 일어나 §8 원천 회피(KVCachePlan 의 "data-not-code" 를 Format read 에 적용).

### D3 — Format plugin = `KVLayoutDesc`(데이터) + `KVFormat`(manage). compute 는 (c) 로 dissolve.
현 `KVCacheFormat`(`format/kv_cache_format.rs:57`)은 4 tier 를 혼재(state getter / `write_kv`·`attention_into` = layer-tier COMPUTE, **backend 인자로 위임** / `compact` = step-tier MANAGE). plugin 화하며 분해:
```rust
#[repr(C)]                          // layer-tier 경계 POD — format plugin 의 실제 기여
pub struct KVLayoutDesc { pub block_elems: u32, pub bits: u8, pub scale_layout: ScaleLayout, pub packing: Packing }
pub trait KVFormat: Send + Sync {   // step/boundary-tier → Box<dyn> OK
    fn layout(&self) -> KVLayoutDesc;            // 엔진 generic reader 가 hot-path read 수행
    // compact 는 여기 없음 — ADR-0004 D1 으로 superseded (아래 갱신 NOTE)
    // write_kv / attention_into 도 없음 — (c) 커널 seam 으로 이동
}
```
`write_kv`/`attention_into` 은 이미 backend 위임 glue → Format 축이 아니라 hardware 축의 M×N 셀. attention 커널은 **사라지지 않고** descriptor 가 그 커널에 layout 을 공급(여전히 M×N 셀).

> **갱신 NOTE (S4-2, 2026-06-07): compact 은 `KVFormat` 에 추가하지 않는다 — ADR-0004 D1 으로 superseded.** 위 스케치는 `compact(keep, merges)` 를 `KVFormat`(api) 에 두려 했으나, 그 사이 ADR-0004(plan-returning `KVCacheStage`)가 동일 의미를 이미 구현했다: keep/merge 결정은 **stage 축**(`KVCacheStage::plan → KVCachePlan{KeepSpec, Vec<WeightedMerge>}`)이 독점하고, 변형은 엔진 executor `execute_kv_plan`(`pressure/eviction/stage_registry.rs`)이 독점한다(D1: 결정=plugin, 변형=엔진). 따라서 (1) `KVFormat`(api) 에 `compact` 표면 **미추가**(2-method 유지: `name`+`layout`), (2) `compact` 결정-소유권은 stage 축(format 축 아님 — stage⊥format 직교 보존), (3) `Merge`(engine, 균등)는 api 로 누출하지 않고 `WeightedMerge`(api, lib.rs)가 정본. 부산물: 구 engine `KVCacheFormat::compact` trait 메서드 + `StandardFormat::compact`/`apply_merges`(균등)는 **폐기**(production 미호출 — 전부 execute_kv_plan 경유), `KVCacheFormat` 은 6-method(geometry 3 + write_kv/write_kv_batch/attention_into). 게이트: `compact_parity` 9/9 bit-identical(4 정책 × 3 dtype). engine `Merge` 타입은 `EvictionPolicy::plan_keep` 가 계속 쓰므로 보존(별도 dedup 여지).

### D4 — Format = 데이터, Backend 가 커널 소유, 인터페이스만 직교. (R2)
`(format × hardware)` 커널은 **이미 backend 안에서 통합**돼 있다 — `match b.dtype() { Q4_0 => matmul_transposed_q4_0, ... }`(`cpu/common.rs:76`), OpenCL `kernel_mul_mat_q4_0_f32` 등. 그래서:
- Format plugin 은 **커널을 싣지 않는다** — 순수 descriptor + manage.
- 커널은 **backend 소유**(descriptor 로 키잉된 내부 dispatch). 새 backend(행) = backend plugin. 새 format(열) = descriptor + 각 backend 의 커널 arm(= backend 관심사).
- D3 의 "format 이 커널 column 을 양방향 seam 에 inject" 는 **단순화 폐기** — 커널은 backend 가 소유, seam = backend 내부 dtype→kernel 테이블 그 자체.
- 직교는 **인터페이스 층위에서만**(Format descriptor ⊥ Backend trait). 커널은 인정된 M×N 으로 backend 에 격리.

### D5 — generic floor + 특화 opt-in. descriptor 어휘 = block-quant family.
현 backend 는 strict specialized(`_ => Err("Unsupported dtype")`, `cpu/common.rs:81`) → 새 descriptor 는 모든 backend 에서 튕김. 해소:
```rust
match b.dtype() {
    DType::Q4_0 => self.matmul_transposed_q4_0(...),   // hot format: 특화 (opt-in, backend 소유)
    _ => { let f32 = dequant_via_descriptor(b, b.layout_desc()); self.matmul_transposed_f32(a, &f32, out) }  // floor: exact
}
```
- floor 는 **exact**(dequant 무손실 + f32 matmul 동일), 느릴 뿐. 검증된 GpuFold "menu + escape" 와 동형.
- per-backend 비용 최소 — `_ =>` arm = 공유 `dequant_via_descriptor`(엔진 헬퍼, descriptor 가 unpack 구동) + 기존 f32 path. attention 동형(dequant KV→f32→f32 attention).
- **descriptor 어휘 = block-quant family**(`block_elems`/`bits`/`scale_layout`/`packing`) → q4_0/q4_1/q8_0/q5 등 전부 generic dequant 가능. **mxfp4 shared-exponent·codebook·sparse 는 floor 밖 → 특화 opt-in escape**(backend 작업, floor 안 거침).

### D6 — 등록 = 3축 평행 linkme registry. 단일 registry 병합 금지.
```
KV_CACHE_STAGES (✅ 존재, stage_registry.rs)
KV_FORMATS      (🔵 신설, KVCacheStageReg 동형: KVFormatReg{ name, make })
BACKEND_CAPABILITIES (🔵 신설, BackendCapReg{ name, make })
```
각각 `find_*(name)`/`registered_names()` 동형 표면. linkme `#[distributed_slice]` 정적 수집 보존(§3.3 Android-safe). **단일 registry 로 병합 금지**(§3.3.1). 저작 통합용 `PluginManifest` + `register_plugin!` 매크로(rlib→linkme / cdylib→C-ABI entry dual-wiring)는 골격만, **`.so` 배선은 deferred**.

### D7 — phasing: 완전 crate 분리 먼저 → `.so` 나중. crate 단계 게이트 L1·L2.
crate 경계(engine → api crate)가 **"데이터 계약에 엔진 타입 0"을 컴파일러로 강제**(circular dep = 컴파일 에러) — `.so` 의 가장 어려운 전제를 plumbing 전에 증명한다. 증명체: `technique-api` deps = linkme 만, 엔진 타입 import 0. `.so` 전환의 나머지(linkme→C-ABI entry, `Box<dyn>`→opaque shim, `Vec`→ptr+len, panic catch_unwind, allocator)는 **기계적·저빈도 표면 국한** → deferrable.

**그러나 컴파일러가 *안* 잡는 crate 단계 게이트 2개**:
- **L1 — layer-tier 경계 타입은 지금 `repr(C)`** (D2 를 미루지 말 것). 지금 Rust enum/`Vec` 로 만들면 `.so` 때 reshape 강제. step/boundary 타입(`KVCachePlan`)은 Rust-native OK.
- **L2 — hot(layer-tier) CALL 이 api 경계를 넘지 않게(descriptor/데이터만).** fat-LTO 가 crate 단계에선 cross-crate `Box<dyn>`/call 을 인라인 → hot 경계 call 이 "공짜처럼" 보인다. `.so` 는 cross-crate 인라인을 제거 → 그 call 이 인라인 불가 C-ABI call 로 → **성능 회귀**. ∴ crate 단계에서 "layer-tier call 이 경계를 안 넘는다"를 **§8 게이트(plan-path bit-identical + TBT Δ≤+3%)로 검증**. "LTO 켜니 빠르다"를 신뢰 금지.

---

## 3. Consequences

- **Format plugin 이 매우 얇아짐** — 순수 데이터(`KVLayoutDesc`) + step-tier manage. 새 block-quant format = descriptor 1개로 generic floor 통해 **전 backend 즉시 동작, backend 수정 0**. hot format 만 backend 가 특화 opt-in.
- **Backend plugin = hardware + 커널 테이블**(descriptor 로 키잉). 커널 M×N 은 backend 내부에 갇힘(직교 인터페이스 보존).
- **`.so` 전환이 기계적** — L1·L2 만 crate 단계에서 지키면 나머지는 저빈도 표면 shim.
- 확장 비용 = 폴더 + descriptor/trait + linkme 1줄(stage 와 동일), 기존 로직 수정 0(OCP).

## 4. Alternatives considered

- **(a) 단일 통합 추상**(Format 을 Backend god-trait 에 흡수) — **기각**: 직교 위반(M+N+K→M×N 인터페이스 폭발), format 비-커널 로직이 N backend 에 중복, §3.3.1 과 모순.
- **(c-양방향 seam)** Format plugin 이 자기 커널 column 을 별도 seam 에 inject — **단순화 폐기**(D4): 커널은 이미 backend 소유, 양방향 레지스트리 불필요.
- **status quo** specialized-only(`match dtype { _ => Err }`) — 새 format = 전 backend 수정, plugin 성 0.
- **`.so`-now**(crate 단계 건너뛰기) — **기각**: 데이터 계약 검증(엔진 타입 0) 없이 plumbing → 계약 leak 발견 시 전면 재작성. crate-first 가 이를 컴파일러로 선검증.

## 5. Risks / Landmines

| 위험 | 심각도 | 완화 |
|---|---|---|
| **L2**: fat-LTO 가 crate 단계에서 hot 경계 call 을 인라인해 숨김 → `.so` 때 perf 회귀 | 高 | §8 게이트(plan-path bit-identical + TBT Δ≤+3%) 를 **crate 단계**에서 검증. layer-tier call 0 across api 경계 |
| descriptor 어휘가 미래 format(mxfp4 등) 못 담음 | 中 | escape hatch — 특화 opt-in(floor 안 거침). 어휘는 block-quant family 로 고정 |
| generic floor 가 너무 느려 실용성 의심 | 中 | hot format 특화 opt-in(floor 는 correctness/availability floor) |
| `KVCacheFormat` 해체 시 forward_gen_fmt(cold) ↔ plan path(hot) 분기 회귀 | 中 | §8/§9.1 — plan path 가 production hot, forward_gen_fmt 는 cold. α-K substep 게이트 |

## 6. Deferred / 안 가본 길

- **`.so` dual-wiring**: `PluginManifest`/`register_plugin!` 매크로, C-ABI register entry, `Box<dyn>`→opaque handle shim, `Vec`→ptr+len 마샬링, catch_unwind, allocator 규율.
- **GpuFold = Backend capability 의 첫 구체 instance** production 배선 — observe-hook PoC(`/tmp/llm_rs2_poc_hook/`, host+ARM 검증완료) → `FoldRunner<dyn GpuFold>`(`GpuScoreAccumulator` 일반화). OpenCL readback 레버 결론: #1 layer 축소(정확도 trade) / #2 rpcmem zero-copy(**기각**, copy 보다 3-6× 느림) / #3 curated GPU fold(채택).
- **WeightFormat 평행 처리** — 본 ADR 은 KV(`KVCacheFormat`)에 집중. weight 축은 동형(`format/weight_format.rs`, arch §2.1 C2).
