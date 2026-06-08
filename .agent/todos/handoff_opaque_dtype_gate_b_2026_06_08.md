# Handoff: ADR-0007 opaque-dtype — GATE-B 완주 (DType-우회 KV format 해금, .so 북극성 첫 실증)

**작성**: 2026-06-08
**HEAD**: `f864cbf0 feat(pressure): ADR-0007 G6/G7 — OpaqueKvFormat + GATE-B bit-identical 통과`
**브랜치**: master (미푸시)
**작성자**: 메인 세션

**다음 세션 진입 문장**: **"ADR-0007 다음 단계 결정부터 — GATE-C(.so cdylib 승격) vs production KVCache 통합(grow/eviction) vs 다른 축(stage/backend .so). 北극성 우선순위 grill 후 착수."**

---

## TL;DR

3-axis plugin 북극성(zero-compile `.so` 로 새 KV format)을 **직접 전진**. 직전 세션 S4-3(attention floor) 설계 패스가 "production dead arm + 북극성 무전진"으로 무효화되자, 사용자가 **진짜 blocker=닫힌 `DType` enum** 으로 pivot. **ADR-0007 opaque-dtype** 설계 → grill → **GATE-B(host) 완주**(7 커밋, 전부 bit-identical 게이트). `DType` variant **없이** 합성 format `synth_q4` 가 CPU 에서 alloc+write+attention end-to-end 동작 = 형식 축 `.so`-ready 첫 실증. **멈춘 이유**: GATE-B 가 이번 단계 종착(사용자 확정). 다음(GATE-C/.so cdylib 등)은 새 결정 + 일부 device-gated.

---

## 진행 상태 (이번 세션, 8 커밋 — 명시 파일만, 무회귀)

| | 커밋 | 내용 | 게이트 |
|---|---|---|---|
| ADR | `a7f859a5` | ADR-0007 설계 SSOT (D1~D6) | 문서 |
| G1 | `9d749056` | `KVLayoutDesc::bytes_for_elems`/`block_bytes` 단일원천 + dtype_layout DRY | technique-api parity + engine size_of cross-check + read floor 무회귀 |
| G2 | `70484076` | `OpaqueBuffer`(raw + KVLayoutDesc sidecar, U8 tag, typed accessor 부재) | tag/descriptor/downcast |
| G3 | `aba2c0d7` | `dequant_via_descriptor` opaque sidecar 인식 | opaque dequant == Q4_0 dequant bit-identical |
| G4 | `da5da032` | `encode_via_descriptor`(write encoder, batch) | == BlockQ4_0::quantize **byte-exact** + round-trip |
| G5 | `ed7ad09e` | `synth-q4-format` 외부 crate(KV_FORMATS 등록, engine 의존 0) | find_kv_format round-trip |
| G6/G7 | `f864cbf0` | `OpaqueKvFormat` impl + **GATE-B** | synth_q4 == q4_0 round-trip **bit-identical** + prefill smoke |

전체 게이트: `cargo test --workspace --lib`(skip opencl/memory) = **llm_rs2 1256/0** + technique_api 10 + synth_q4_format 1 + 나머지 전부 0 failed. `cargo clippy --workspace -- -D warnings` clean. fmt clean(drift 2파일 미변경).

**핵심 설계(ADR-0007)**: 새 format = 새 `DType` 아님 → `OpaqueKvFormat` impl + `OpaqueBuffer`. **D2 비대칭**: READ/byte/alloc=데이터-구동 floor ⊥ **WRITE=코드-구동 encoder 필수**(layout-only descriptor 가 quantize 정책 못 담음). 사용자 grill 확정: GATE-B / Rust encoder→.so vtable / `DType::U8`+sidecar.

---

## 다음 작업 (전부 새 결정 필요 — grill 먼저)

1. **GATE-C = `.so` cdylib 승격** (북극성 최종). `register_plugin!`(rlib→linkme / cdylib→C-ABI dual), `libloading`/dlopen, write encoder `#[repr(C)]` vtable + `extern "C"` + `catch_unwind`. **인프라 0% 착수** + device-gated. 가장 크고 어려움.
2. **production KVCache 통합** — opaque 를 실제 추론에 쓰려면 KVCache grow/shrink/eviction 일반화(`is_q4`→block-generic, byte-accounting descriptor-first). GATE-B 는 capacity pre-size 자체 저장이라 미통합. Q4_0 bit-identical 회귀 필수.
3. **GPU opaque arm** — opencl `attention_gen`(`opencl.rs:6098` `_=>kernel_attn_gen` F32 가정=silent garbage) / `matmul`(`opencl.rs:5577`) dtype-guard + readback floor. **device-gated**(host 미검증).
4. **write encoder family 확장** — `encode_via_descriptor` 현재 PerBlockF16/Nibble(q4_0)만. q8_0(Byte)/q4_1(WithMin)/q2_0(asymmetric)는 범위 한정 Err → 필요 시 확장(q2_0 은 D4 vtable encoder 로만).

---

## Landmines / 핵심 발견 (다음 세션 필독)

- **S4-3(attention floor) = dead arm, 재추진 금지** — KV dtype CLI `{f32,f16,q4}` hard-bail(`bin_setup.rs:69`) + KV write Q4_0-only 라 Q8_0/Q4_1 KV 채울 경로 0. "M-F3 matmul floor 도 dead-arm 선례"는 **틀림**(`--primary-dtype q8_0` 지원 → matmul floor reachable). attention floor 는 북극성 무전진.
- **`DType` 는 닫힌 Copy enum**(`buffer.rs:25`, 7-variant). `.so` 는 variant 못 늘림 → opaque 우회가 유일. `DType::size()` 는 block-quant 에 1 반환("handled separately") — byte-회계는 `KVLayoutDesc::bytes_for_elems`(G1) 가 단일원천.
- **engine/Cargo.toml 건드리지 말 것** — pre-existing 미커밋 M(microbench_score_readback). 그래서 `synth-q4-format`(외부 crate)을 engine 에 force-link 못 함 → GATE-B engine 게이트는 inline descriptor 사용(외부 crate 는 자체 self-test 로 등록 증명). GATE-C 의 engine→플러그인 링크는 이 제약 해소 필요.
- **opaque typed-access 차단은 컴파일 강제 아님** — `Tensor::as_slice::<T>()`(`tensor.rs:56`)는 public layout-blind. `OpaqueBuffer` 가 typed accessor 미노출 + `as_any` downcast 로만 식별하나, opaque tensor 에 `as_slice::<f32>()` 직접 호출은 audit/규율로만 막음(`OpaqueTensor` wrapper 는 다음 단계).
- **설계 정련 기록**: synth(12-agent) 원안은 OpaqueKvFormat→production KVCache.update/grow 재사용(is_q4 일반화)이었으나, Q4_0 회귀 위험 + 외과성 위배로 **기각** → 자체 최소 저장 채택. production 통합은 별 단계(위 #2).
- **RSS flaky**: `pressure::kv_cache::tests::test_*release_unused_pages*` — full lib 집계 시 간헐 1-fail, 단독·재실행 PASS = 환경성(회귀 아님). 이번 세션은 미발생(1256/0).

---

## 참조

- SSOT: `docs/adr/0007-opaque-dtype-kv-format-unlock.md`(D1~D6 / §5 Risks / §6 Deferred=GATE-C), `docs/adr/0005-...md`(D3 format=데이터-only / D5 floor / §6), `CONTEXT.md`(format 축).
- 코드 앵커: `engine/src/buffer/opaque.rs`(OpaqueBuffer) / `engine/src/pressure/opaque_format.rs`(OpaqueKvFormat + GATE-B 테스트) / `engine/src/format/dtype_layout.rs`(bytes_for_elems / dequant_via_descriptor opaque / encode_via_descriptor) / `crates/technique-api/src/lib.rs`(KVLayoutDesc::bytes_for_elems) / `crates/techniques/synth-q4-format/`(외부 등록).
- 설계 워크플로우 산물: S4-3 dead-arm `wf_4110e3e2` / DType 해금 `wf_817dbf35`(map 6 + 4안 + 심사 + 종합).
- 선행 handoff: `handoff_adr0005_step4_format_axis_2026_06_07.md`(S4-1/S4-2 + S4-3 진입, 본 세션이 S4-3 무효화).
