# 진행 원장: GATE-C v2 — Format 축 `.so` dlopen plugin (호스트)

**시작**: 2026-06-09 · **SSOT**: `docs/adr/0009-gate-c-stage-dlopen-plugin.md`(D4 Format 축) · **거울**: v1 C1–C5(`851e4d03`~`5a6a44bc`)

## ▶ 재개 진입점
"GATE-C v2 Format 축 — 다음 미완 CF 단계 1개 착수(원장 체크리스트 확인)."

## 핵심 통찰 (v1 대비 차이)
- `KVFormat` = `name()` + `layout() -> KVLayoutDesc`(콜백 0, ctx 0, plan 0). v1의 StageCtxAbi/PlanAbi/PlanArena/AbiStageCtx **전부 불필요** — vtable = abi_version + name + make + layout(POD by-value) + drop.
- `KVLayoutDesc` 는 이미 `#[repr(C)]` POD(block_elems:u32 / bits:u8 / scale_layout:ScaleLayout(repr u32) / packing:Packing(repr u32)) → extern "C" fn-ptr 로 값 전달 가능. CF1 마샬링 = 사실상 POD 복사.
- **force-link 비대칭(중요)**: `synth_q4` 는 엔진 lib 이 `use synth_q4_format as _;`(builtin_kv_formats.rs)로 **무조건 정적 등록** → 통합 테스트에서 `find_kv_format("synth_q4")==Some` → `register_dynamic_formats([synth_q4.so])` 는 **builtin-collision reject**(v1 example_keep_recent 은 force-link 안 됐던 것과 다름). 따라서:
  - synth_q4 → descriptor-identity(**직접 dlopen** vtable vs 정적) + builtin-collision reject + 심볼부재 reject 담당.
  - **신규 `example-kv-format`**(비충돌 이름, 엔진 force-link 안 함) → registry-merge 성공 + make_format 동적 fallback + dup reject 담당. (stage 축 example-keep-recent 의 format 축 짝 — 영구 템플릿 가치)
- 인터페이스는 v1 거울로 100% 결정 → grill 불요. 위 second-vehicle 은 테스트 인프라 결정(문서화 후 진행).

## 스코프 경계 (목표 task 목록 = 4 + 게이트)
- **포함**: CF1 technique-api / CF2 engine 동적 레지스트리 / CF3 vehicle(synth_q4 cdylib + example-kv-format) / CF4 gate test.
- **defer(목표 게이트 밖, v1 의 eviction-firing e2e 에 해당)**: bin_setup `--load-plugin`→format 배선 + `--kv-format` 동적 해석(make_format) 프로덕션 배선. dlopen 메커니즘은 CF4 격리 테스트로 증명.

## 체크리스트 — ✅ 전부 완료 (2026-06-09)
- [x] **CF1** `e307ed9f` technique-api: `KV_FORMAT_ABI_VERSION` + `FormatVTableAbi`(repr C) + `unsafe impl Sync` + `register_kv_format!` 매크로 + POD round-trip 테스트. 게이트=technique-api 16/16.
- [x] **CF2** `a8efc8f9` engine `format/dynamic_format_registry.rs`: `DYN_FORMAT_REGISTRY` + `register_dynamic_formats` + `dynamic_registered_format_names` + `make_format` + `DynFormat`. 게이트=빌드 green + clippy clean.
- [x] **CF3** `f2e16602` vehicle: synth-q4-format cdylib+register_kv_format! / 신규 example-kv-format. 게이트=각 정적 1/1 + `nm register_kv_format_v1` T(ON)/부재(OFF).
- [x] **CF4** `c7734a94` `engine/tests/gate_c_format_dlopen_equivalence.rs`: 7요소. 게이트=`gate_c_format_dlopen_equivalence` 1 passed.
- [x] **완료**: technique-api 16/16 · lib 1260/0(flaky RSS 격리 통과, format 무관) · clippy --workspace clean · release fat-LTO 51.87s green · handoff/메모리 갱신.

## 최종 게이트 결과
- technique-api **16/16** · lib **1259+1 flaky**(`test_release_unused_pages_rss_reduction` 격리 ok, pressure/kv_cache 변경 0 → 회귀 아님) = 실질 1260/0
- clippy `--workspace -D warnings` clean · release `-p llm_rs2` fat-LTO **51.87s green**(GATE-C v2 코드 LTO 생존)
- `nm register_kv_format_v1`: synth_q4.so/example_kv_format.so feature ON=`T`, OFF=부재

## iter 로그
- (iter 0, 2026-06-09) 지형 매핑: v1 템플릿 전문 확보, KVFormat=descriptor-only 확인, force-link 비대칭 발견, example-kv-format second-vehicle 결정.
- (iter 1) CF1 `e307ed9f` — technique-api Format C-ABI + 매크로 + POD round-trip. technique-api 16/16.
- (iter 2) CF2 `a8efc8f9` — engine 동적 레지스트리/make_format/DynFormat. 빌드+clippy green.
- (iter 3) CF3 `f2e16602` — synth-q4-format cdylib + example-kv-format 신설. nm T.
- (iter 4) CF4 `c7734a94` — descriptor-identity 게이트 1/1. 전 완료 게이트 green → 종료.
