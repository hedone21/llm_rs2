# Handoff: GATE-C 멀티-vtable bundle ABI (ADR-0010) V1~V5 호스트 완주

**작성**: 2026-06-09
**HEAD**: `ffea6dec test(engine): GATE-C V5 — v2 멀티-vtable 재증명 게이트 통합` (+ 원장 후속 커밋)
**브랜치**: master (미푸시)
**작성자**: 메인 세션 (/goal 루프 5 iter)

**다음 세션 진입 문장**: **"GATE-C 멀티-vtable bundle ABI(ADR-0010) 호스트 완주 — 다음은 (a) production e2e(실모델 argus_cli --load-plugin 번들 .so + --kv-format 동적 해석, CPU smoke), (b) device(Android dlopen, SELinux/W^X), 또는 (c) Backend v3 축 grill(BackendCapability 메서드 미확정) 중 택1."**

---

## TL;DR

북극성("zero-compile `.so` plugin")의 동적 엔트리 ABI 를 **"vtable 1개 → vtable 리스트(봉투)"로 교체**(ABI_VERSION 1→2). 한 `.so` 가 한 축에 **다수** capability(quant 패밀리) + **양축 번들**(stage+format)을 싣고, host 가 `.so` 1회 dlopen 으로 흡수(open-once cross-axis dispatcher, registry 병합 없음). ADR-0009 가 defer 한 production 배선(`--load-plugin`→format + `--kv-format` make_format)도 완주. **멈춘 이유**: ADR-0010 V1~V5 전 마일스톤 + 완료 6게이트 green → 호스트 스코프 완주. SSOT=`docs/adr/0010-gate-c-multi-vtable-bundle-abi.md`.

---

## 진행 상태 (V1~V5, 무회귀)

| 마일스톤 | 커밋 | 핵심 |
|---|---|---|
| ADR | `48d77243` | ADR-0010 (적대적 리뷰 4렌즈 반영, ADR-0009 D2/D6/#118 부분 supersede) |
| **V1** technique-api | `90b04aa7` | 봉투(Stage/FormatExportAbi) + PLUGIN_*_VTABLES 슬라이스 + register_kv_*! 재작성(const-block 격리) + export_plugin! + ABI 1→2 |
| **V2** engine 로더 | `8846d0b3` | try_register_*(count, 2-pass 원자, vtables.add(i)) + register_dynamic_plugins(open-once) + batch strict 래퍼 + rename + vtable abi_version 제거 |
| **V3** 배선 | `31c8405b` | bin_setup W1(register_dynamic_plugins)+W2(make_format)+ensure_builtin startup |
| **V4** vehicle | `62097107` | export_plugin! 3종 + example-multi-format/bundle/no-export 신설 |
| **V5** 게이트 | `ffea6dec` | gate_c_plugin_bundle.rs(11 단언) 통합 재작성 + bundle_perhead + example-rollback + obsolete v1 테스트 2 삭제 |

**완료 6게이트(전부 green)**: technique-api **19/0**(OFF)·**20/0**(plugin-cdylib) · lib **1260/0** · `gate_c_plugin_bundle` **1/1**(11 단언) · `clippy --workspace` clean · `nm`(export 6종 v2 entry 존재, no-export 부재) · release fat-LTO **54.17s green**.

---

## 다음 작업 (택1)

1. **production e2e** (v2 defer 와 동형, 가장 자연스러운 연속): 실모델 GGUF 로 `argus_cli --load-plugin <번들 .so> --kv-format <동적 format>` CPU smoke → opaque storage 가 동적 descriptor 로 실제 구동되는지 토큰 생성 확인. 모델 의존이라 커밋 테스트 아닌 수동 검증·수치 보고. (gate_c_plugin_bundle 은 격리 테스트로 메커니즘 증명; v1 의 eviction-firing e2e defer 와 동형.)
2. **device** (host-only scope 밖): Android dlopen — SELinux/W^X, `--load-plugin` 경로 제약. ADR-0009 D6 landmine.
3. **Backend v3 축** (최후순위): `BackendCapability` trait 메서드 미확정(`name()` 1개) → **설계 grill 선행**. V1~V2 인프라(봉투/슬라이스/dispatcher/export_plugin! 프레임) 재사용하나 BackendVTableAbi+dual-wiring 매크로+DynBackendCap 선행 필요(ADR-0010 §3 C2). GPU device-only.

---

## Landmines / 미해결

- **구 통합 테스트 삭제됨**: `gate_c_dlopen_equivalence.rs`(stage v1) + `gate_c_format_dlopen_equivalence.rs`(format v1)는 v1 단일-vtable ABI 제거로 컴파일 불가 → `gate_c_plugin_bundle.rs` 가 더 강하게 통합 대체(약화 아님, CF4 7요소 1:1 보존 매핑 = 그 파일 헤더 참조).
- **`register_dynamic_stages`/`register_dynamic_formats`**(strict batch 래퍼)는 이제 gate 테스트·축-격리 진단용 — production 진입은 `register_dynamic_plugins`(dispatcher) 단일. 두 래퍼 보존(pub).
- **PerHead bail**: dlopen plugin 이 PerHead keep 산출 시 host `planabi_to_plan`(`stage_registry.rs:632`)가 bail → DynStage.plan None(silent garbage 방지). v2 봉투 교체와 무관(D5 마샬링 불변). bundle_perhead vehicle 로 검증.
- **encode_via_descriptor = PerBlockF16+Nibble(q4_0 canonical)만**(`dtype_layout.rs:271`). multi-format 의 mf_q8(Byte/8)은 descriptor-identity 만, floor round-trip 은 mf_q4(q4_0)만. 다른 family write 는 ADR-0007 D4 per-format 확장 시.
- **engine/Cargo.toml drift**(score_readback microbench `[[bin]]`)는 5 마일스톤 내내 **미커밋 유지**(working tree ` M`). V1~V5 는 이 파일 무수정. 커밋 금지.
- **technique-api plugin-cdylib feature**: V1 에서 추가(테스트가 동적 arm 누적 검증용). plugin crate 의 plugin-cdylib 와 독립(매크로 #[cfg]는 전개 crate feature 로 평가).

---

## 참조
- SSOT: `docs/adr/0010-gate-c-multi-vtable-bundle-abi.md`(E1~E7). 선행 `0009`(D2/D6/#118 부분 supersede).
- 코드 앵커: `crates/technique-api/src/lib.rs`(봉투/슬라이스/register_kv_*!/export_plugin!) · `engine/src/pressure/eviction/stage_registry.rs`+`engine/src/format/dynamic_format_registry.rs`(try_register_*) · `engine/src/session/plugin_dispatch.rs`(dispatcher) · `engine/src/session/bin_setup.rs`(W1/W2) · `engine/tests/gate_c_plugin_bundle.rs`(게이트) · `crates/techniques/example-{multi-format,bundle,no-export,rollback}`(vehicle).
- 진행 원장: `gate_c_v3_multivtable_progress_2026_06_09.md`.
