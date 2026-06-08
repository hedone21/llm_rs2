# ADR-0007: opaque-dtype — descriptor-운반 KV format 의 `DType`-우회 해금 (zero-compile `.so` 북극성)

> **Status**: Accepted
> **Date**: 2026-06-08
> **Decision-makers**: 사용자 + 메인 세션 (DType 해금 grill 세션; 적대적 검증 워크플로 12-agent 대조 후속 — `wf_817dbf35`)
> **Selected**: 새 KV format 은 **새 `DType` enum variant 가 아니라** 새 `KVCacheFormat` impl(`OpaqueKvFormat`) + 새 `Buffer` impl(`OpaqueBuffer`{raw bytes + `KVLayoutDesc` sidecar, `DType::U8` tag})로 표현한다. **READ·byte-회계·alloc 은 데이터-구동**(`KVLayoutDesc` floor, panic-free, `#[repr(C)]`) — `WRITE(quant)만 format-bound encoder fn`(코드-구동)으로 분리한다. `DType` enum 은 **닫힌 채 유지**, `.so` 는 variant 를 늘리지 않고 descriptor 로 다양성을 표현한다. 이번 단계 종착 = **GATE-B**(read+write opaque format host end-to-end). `.so` cdylib 승격(GATE-C)은 device-gated 다음 단계.
> **Related**: ADR-0003(정적 crate + linkme + force-link), ADR-0004(`KVCacheStage` plan-returning, **D1 변형=엔진 독점**), ADR-0005(D3 format=데이터-only descriptor, **D5 generic floor**, D6 3축 평행 registry, §6 deferred=opaque-buffer 골격), `arch/pipeline_stage_design_v2.md` §4.1(`KVCacheFormat` base trait), `/CONTEXT.md`(format 축 = 연산 표현/precision)

---

## 1. Context

궁극 목표(북극성): **zero-compile `.so` 설치로 새 KV format(block-quant family) 추가** — 새 `DType` enum variant·engine/backend 수정 없이. `.so` 가 제공하는 것은 `name` + `KVLayoutDesc`(`#[repr(C)]`: `block_elems`/`bits`/`scale_layout`/`packing`) + 암묵적 raw bytes 뿐이다 (Rust generic·새 `DType` variant 금지).

**발단 (S4-3 설계 패스가 무효화한 전제, 2026-06-08)**: ADR-0005 step4 의 다음 작업으로 지목됐던 **S4-3(attention floor)** — `CpuBackend::attention_gen`(`cpu/common.rs:886`)의 `_ => Err` 를 matmul floor 와 동형(`_ => dequant_via_descriptor → f32 attention`)으로 교체 — 을 설계해보니 **production dead arm** 으로 확정됐다:
- KV dtype CLI 가 `{f32,f16,q4}` 로 hard-bail(`bin_setup.rs:69`) + KV write quantize 가 Q4_0-only(`backend.cast` (F32,Q4_0)만, `kv_cache.rs` type_size `_=>0`) → **Q8_0/Q4_1/BF16 KV 가 채워질 경로 0**. floor arm 은 unit-test fixture 로만 도달.
- "M-F3 matmul Q8_0 floor 도 dead-arm 선례" 주장은 **틀림** — `--primary-dtype q8_0`(`session/cli.rs:158`) 지원 → Q8_0 weight 가 matmul `_=>` floor 에 실제 도달(**reachable**). 따라서 attention floor 는 선례 엄호 없는 순수 dead code.
- 결정적으로 attention/matmul floor 어느 것도 **북극성을 전진시키지 않는다** — `.so` 의 진짜 blocker 는 **닫힌 `DType` enum** 이고, floor 는 in-enum dtype 의 OCP 만 푼다.

**진짜 blocker (확정)**: `DType`(`engine/src/buffer.rs:25`, 7-variant, `#[derive(Copy,Eq)]`)는 닫힌 Rust enum. 버퍼 byte-회계(`DType::size()`)·typed 접근(`as_slice::<T>()`)·backend dispatch(`match dtype`)·KV alloc 이 전부 `DType` 에 키잉 → `.so` 가 variant 를 추가할 수 없다.

**이미 깔린 토대**: `KVLayoutDesc`(`#[repr(C)]`, technique-api) + `dtype_to_layout_desc`(DType→desc) + `dequant_via_descriptor`(Tensor→f32, **read floor**, M-F3) + `KV_FORMATS` linkme registry(S4-1, production 소비자 0=unwired) + `KVFormat` trait(name+layout). `KVCacheFormat` base trait(`kv_cache_format.rs:58`)은 이미 `as_any()` 부재(downcast 차단)·`dtype()` 부재(Stage 가 Format 을 모름)·`write_kv`/`attention_into` 가 backend 핸들 받아 내부 dispatch 하도록 설계 — 주석이 "새 Format = 새 impl, base trait·forward 변경 0"을 명시(line 56-57).

**충돌 불변·제약**:
- 엔진은 opaque `.so` 데이터를 **raw bytes 로만** 보고 descriptor floor(dequant)로만 읽는다. typed `as_slice::<KnownT>()` 금지(데이터 의미 모름).
- KV write 는 **매 토큰** 발생 → opaque format 도 write(quant) 필수.
- L1: opaque 경계 타입 `#[repr(C)]` now. L2: hot layer-tier call 이 api crate 경계 미초과(fat-LTO 가 숨기므로 device 게이트로 검증). 어휘 = block-quant family, codebook/sparse/mxfp4 = escape(ADR-0005 D5).
- crate-phase 먼저 → `.so` 나중(C-ABI 기계적). 호스트 GPU 부재(GPU=device-gated). CLAUDE.md: 외과적·단순함·추측성 금지·테스트 약화 금지·신규 모듈 no-`mod.rs`.

---

## 2. Decision

### D1 — opaque format = 새 `OpaqueKvFormat` impl + `OpaqueBuffer`, **`DType` variant 추가 금지**.
새 KV format 은 `KVCacheFormat` 의 의도된 확장점(§4.1)을 그대로 쓴다 — 새 impl(`OpaqueKvFormat`) + 새 `Buffer` impl(`OpaqueBuffer`{`raw: Vec<u8>`, `desc: KVLayoutDesc`, `encoder`}). forward 가 이미 `Arc<dyn KVCacheFormat>` 로 디스패치하므로 base trait·forward 변경 0. `DType` enum 은 닫힌 채 유지하고, `.so` 는 variant 를 늘리는 대신 `KVLayoutDesc` 로 다양성을 표현한다.

### D2 — `byte_accounting` ⊥ `write_encoder` 본질적 비대칭. READ=데이터, WRITE=코드.
- **READ(attention)·byte-회계·alloc = 데이터-구동** (panic-free, `#[repr(C)]` 가 `.so` 경계 그대로):
  - attention: `OpaqueKvFormat::attention_into` 가 KV row 를 `dequant_via_descriptor` per-row floor 로 f32 unpack 후 기존 CPU F32 attention 재사용 (KV 는 매 토큰 grow 하는 대용량이라 통째 dequant 부적합 → row-단위 lazy floor).
  - byte-회계: `dtype_layout.rs:178-189` 의 인라인 block_bytes 공식(`quant_bytes`=Nibble→`block_elems/2`·Byte→`block_elems`, `scale_bytes`=PerBlockF16→2·WithMin→4·None→0)을 `KVLayoutDesc::bytes_for_elems(numel)` 로 추출(단일원천). alloc/grow/shrink 가 호출.
- **WRITE(quant) = 코드-구동 필수**. ← 이 작업 전체 실현성의 분기점, `.so` 의 가장 어려운 단계.

  **결론: `KVLayoutDesc` 는 quantize 정책을 canonical 하게 함의하지 않는다. layout-only 로 write 는 닫히지 않는다.** 증거: (1) 역방향 descriptor-driven encoder 가 코드베이스 전역 0개(`encode/quantize/pack_via_descriptor` grep 0건; 존재하는 건 `unpack_block_via_descriptor`/`dequant_via_descriptor` read 뿐). (2) 매 토큰 quant 산술은 하드코딩(`BlockQ4_0::quantize`: `d=max_abs/7.0` symmetric, `round().clamp(-8,7)`, zero-point +8). (3) `KVLayoutDesc` 4필드는 전부 "바이트가 어디 놓이나"(layout)이고 "값을 어떻게 정하나"(policy: scale=max/qmax vs absmax, zero-point 유무, round vs truncate, clamp 경계, q4_1 min 산출식)는 자유도라 layout 밖. dequant 가 desc 만으로 byte-exact 였던 건 역연산(scale·min 읽어 선형식)이 **결정적**이기 때문 — quantize 는 **손실**이고 정책이 자유도라 닫히지 않는다.

### D3 — opaque tag = `DType::U8` + sidecar `KVLayoutDesc` (`buffer.rs` enum 무변).
`Buffer::dtype()` 가 `DType` 반환을 강제하나 opaque 데이터는 closed `DType` 중 무엇도 아니다. `OpaqueBuffer::dtype()` 는 **`DType::U8`(raw bytes)** 를 반환하고 `descriptor() -> KVLayoutDesc` sidecar 로 의미를 운반한다. `buffer.rs:25` enum 완전 무변(외과적 최상). U8 arm 은 backend match 전반에서 이미 floor/Err 로 빠지므로 typed 재해석이 자연 차단된다(`as_slice::<KnownT>()` 미도달). `OpaqueBuffer` 는 `raw_bytes()`/`descriptor()` 만 노출하고 typed accessor 를 두지 않는다.
- **승격 경로**: "진짜 U8 weight 와 opaque KV 가 둘 다 U8 tag" 의 의미 충돌이 드러나면(`memory_usage`/`copy_slice` 에서 desc 유무 분기가 지저분해지면) **(B) `DType::Opaque` sentinel variant 1개 추가**로 승격(7→8, exhaustive match 전수 점검은 컴파일러가 강제). A→측정→필요시 B 의 점진 경로.

### D4 — write encoder = `OpaqueKvFormat` 보유 Rust 메서드(host) → `.so` 는 `#[repr(C)]` vtable 승격.
D2 의 "write=코드 필수" 결론을 **format 이 보유한 encoder fn** 에 둔다(ggml `from_float` 검증된 패턴). crate-phase 는 순수 Rust 메서드/closure(panic-safe, 검증 용이). `.so` phase 는 `#[repr(C)]` vtable + `extern "C"` fn(`catch_unwind`/abort 정책 동반)으로 승격.
- **encoder 시그니처를 처음부터 `n_blocks` 배치 + 에러코드 반환(panic-free)으로 설계** — `.so` vtable 의 매-토큰 indirect call 빈도를 row/block 단위로 낮춰 L2(call 빈도) 부채를 선제 차단하고 GATE-C 의 기계성을 확보.
- 게이트: `encode → unpack` round-trip 이 `quant.rs` 하드코딩 encoder 와 **요소별 byte-exact**(read 쪽 `dtype_layout.rs:248/292` 거울).
- 기각: (B) generic `encode_block_via_descriptor`(데이터-구동)는 q4_0/q4_1/q8_0 family canonical 2종(symmetric/affine)만 닫고 q2_0 asymmetric 등은 `scale_layout` 어휘 밖 → 어휘 폐포 한계 재현. (C) backend 축 per-family quantize 커널 셀은 GPU 해금 시 read floor 대칭으로 자연 채택하되 **device 단계**로 미룬다.

### D5 — 어휘 = block-quant family. 'open dtype' 범위는 descriptor 폐포 안.
`KVLayoutDesc`(ScaleLayout 3 × Packing 3)는 block-quant family + raw 만 표현한다. q4_0/q4_1/q8_0/q5 류는 generic. **mxfp4 shared-exponent·codebook·sparse·q2_0 asymmetric 은 floor/어휘 밖 → backend 특화 escape**(ADR-0005 D5 정합). 'open dtype' 의 범위가 어휘 폐포 안으로 한정됨을 정직히 명시한다.

### D6 — phasing/게이트: GATE-B(host) → GATE-C(`.so`, device).
- **GATE-B (이번 단계 종착)**: `DType` variant 없는 합성 format `synth_q4`(`OpaqueBuffer`+`KVLayoutDesc`, q4_0 와 동일 layout)을 CpuBackend forward(prefill `write_kv_batch` + decode `write_kv`[encode] + `attention_into`[per-row dequant floor])에 돌려, 동일 데이터의 Q4_0(알려진 dtype) baseline 과 logits/토큰 ID **bit-identical**. 호스트 CPU 만으로 완결. 통과 = 형식 축이 alloc+write+attention 에서 `.so`-ready 첫 실증.
- **GATE-C (다음 단계, device-gated)**: cdylib 승격 + dlopen/`register_plugin` + `extern "C"` vtable. PluginManifest/libloading 인프라 0% 착수 → 별도 단계.
- **L1**: opaque 경계 타입 `#[repr(C)]` now(`KVLayoutDesc` 이미 충족). **L2**: hot layer-tier call 이 api 경계 미초과 — host 정적 증명 불가, S25/Jetson Decode TBT Δ≤+3% device 실측으로만 검증.

---

## 3. Consequences

- **새 block-quant KV format = `OpaqueKvFormat` impl + `KVLayoutDesc` + encoder fn + linkme 1줄** — `DType` enum·backend dispatch·forward 수정 0(read 경로), encoder 1개(write). OCP.
- **`.so` 전환이 비대칭으로 기계적** — read/byte-회계/alloc 은 `#[repr(C)]` 데이터라 C-ABI 그대로; write encoder 만 vtable+`extern "C"` 신규 표면.
- **`KV_FORMATS` registry 의 unwired(소비자 0) 상태 해소** — GATE-B 가 `ensure_*_registered` force-link 를 검증 bin startup 에 배선.
- **정직한 한계**: read-only opaque format 은 데이터만으로 `.so`-ready 이나, **write(production KV) opaque format 은 encoder fn 어휘가 추가돼야 닫힌다**. over-claim 금지.

## 4. Alternatives considered

- **(α) `DType::Opaque(KVLayoutDesc)` 데이터-운반 variant** — `KVLayoutDesc: Copy` 면 `DType: Copy` 유지 가능하나 enum 오염 + 모든 exhaustive match 가 데이터 추출 분기 강제. **D3 의 sentinel(데이터 없는) 승격안으로 축소**(필요시).
- **(γ) `Buffer`/`Tensor` 에 `Option<KVLayoutDesc>` sidecar 필드** — D3 가 채택한 형태에 근접하나 `Tensor` 전역 표면 변경. `OpaqueBuffer`(별도 impl)가 typed-access 차단을 더 자연스럽게 강제 → **별도 buffer impl 채택**.
- **(δ) registry-id key(`FormatId(u32)`) + fn-pointer type_traits(ggml 모사)** — write encoder fn-pointer 아이디어는 **D4 로 접목**. 단 read 까지 fn-pointer 로 가면 ADR-0005 D3(데이터-only) 이탈 폭이 커져 read 는 데이터 유지.
- **`KVLayoutDesc` 에 `QuantPolicy` 축 데이터 추가** — **기각**: `#[repr(C)]` struct 변경이 `.so` forward-compat(L1) 약속을 깨고, round mode/clamp/min 산출 조합을 닫힌 enum 으로 못 담음(어휘 폭발).
- **status quo**(closed `DType` 유지, opaque 미지원) — `.so` 북극성 불가.

## 5. Risks / Landmines

| 위험 | 심각도 | 완화 |
|---|---|---|
| **write encoder 가 read 처럼 공짜가 아님** — 역방향 descriptor encoder 부재(grep 0). | 高 | D4: format-bound Rust encoder + `encode→unpack` round-trip bit-identical 게이트(+경계값/fuzz). `.so` 는 vtable 승격. |
| **GPU silent-garbage** — opencl `attention_gen`(`opencl.rs:6098` `_=>kernel_attn_gen` F32 가정) / `matmul_transposed`(`:5577` `_=>self.matmul`) / cuda_pc `copy_slice`(`:1635` `_=>1`)가 비-F16/opaque KV 를 조용히 오독. | 高 | **device-gated**(host cargo test 검출 불가). opaque 진입 전 loud-fail guard 또는 readback→CPU floor 를 S25/Jetson 실측. backend 간 동작 비대칭을 known-limitation 등재. cuda copy_slice `_=>1` 은 DType 해금과 무관한 **기존 잠재 버그** → 선결. |
| **typed-access 차단이 컴파일타임 보증 아님** — `Tensor::as_slice::<T>()`(`tensor.rs:56`)가 public layout-blind. opaque 에 `as_slice::<f32>()` 오호출 시 silent garbage. | 中 | `OpaqueBuffer` 가 typed accessor 미노출 + `OpaqueKvFormat::as_any` 부재(downcast 차단). 잔존 직접 호출은 audit/규율; 필요시 `OpaqueTensor` wrapper(다음 단계). |
| **`is_q4` 이진분기 일반화 회귀**(`kv_cache.rs:400`) — Q4_0 하드코딩 offset 을 `desc.block_elems` 로 치환 시 element-stride/block-stride 혼선. | 中 | Q4_0 기존 동작 bit-identical 회귀 테스트 선행(update/scatter/shift/prune 전 경로). |
| **L2** fat-LTO 가 crate-phase hot 경계 call 숨김 → `.so` perf 회귀. | 中 | encoder n_blocks 배치 시그니처로 call 빈도↓ + S25 TBT Δ≤+3% device 게이트. |

## 6. Deferred / 안 가본 길

- **GATE-C**: `.so` cdylib 승격 — `register_plugin!` 매크로(rlib→linkme / cdylib→C-ABI entry dual-wiring), `libloading`/dlopen, `extern "C"` vtable forward-compat(abi_stable prefix-type vs fixed `#[repr(C)]`), panic catch_unwind/abort, allocator 경계. 인프라 0% 착수.
- **GPU backend 축 셀**: opaque format 의 GPU attention/quantize 커널(read floor readback-guard, fused cast+scatter quantize kernel) — device 작업(ADR-0005 D4 (format×hardware) 셀).
- **D3 (B) sentinel 승격**: U8/opaque 의미 충돌이 측정으로 드러날 때.
- **rope_inplace/rms_norm dtype-guard**(`common.rs:167/208`) — activation 항상 F32 불변이 호출처에만 의존; debug-assert 추가 검토.
- **write encoder 의 임의 family 확장**(q2_0 asymmetric 등) — D4 (B) 데이터-구동 한계 밖, vtable encoder 로만.
