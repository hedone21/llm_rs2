# 진행 원장: GATE-C 멀티-vtable bundle ABI (V1–V5)

**시작**: 2026-06-09 · **SSOT**: `docs/adr/0010-gate-c-multi-vtable-bundle-abi.md` (E1–E7) · **선행**: ADR-0009(GATE-C 단일-vtable, D2/D6-defer 부분 supersede)

## ▶ 재개 진입점
"GATE-C v3 멀티-vtable — 원장 체크리스트의 다음 미완 V단계 1개 착수."

## 컴파일-green 순서 결정 (커밋 위생)
ABI 교체는 technique-api(V1)↔engine(V2)가 결합돼 있다. 깨진 commit 최소화 규칙:
- **V1**: technique-api — 봉투/슬라이스/`export_plugin!` 추가 + `register_kv_*!` 재작성(const-block 격리 + 슬라이스 기여 + **no_mangle `register_kv_*_v1` entry 제거** ← multi-call 위해 필수). **`abi_version` vtable 필드는 V1 에서 유지**(엔진이 아직 읽음 → 컴파일 green), V2 에서 봉투로 이전+제거. `ABI_VERSION 1→2`.
  - V1 후: `cargo build -p llm_rs2` green(필드 유지). 단 **구 통합 게이트(`gate_c_*.rs`)는 런타임 실패**(register_kv_*_v1 심볼 부재) → V5 에서 재작성. **중간 게이트는 `--lib` 만**(통합 테스트 제외).
- **V2**: engine 로더 — `try_register_*`(봉투 읽기, `vtables.add(i)`, 2-pass) + `register_dynamic_plugins` + batch strict 래퍼 + `dynamic_registered_names→_stage_names` rename + **vtable `abi_version` 제거(struct+macro+engine read 동시)**.
- **V3**: bin_setup W1+W2 + `ensure_builtin_kv_formats_registered` startup 배선.
- **V4**: synth_q4/example 에 `export_plugin!()` + multi-format(서로 다른 desc ≥2) crate + no-export vehicle.
- **V5**: `gate_c_*.rs` 재작성(v2 의미론, CF4 7요소 보존 매핑) + 전체 sanity.

## 체크리스트
- [x] **V1** `90b04aa7` technique-api: 봉투(`StageExportAbi`/`FormatExportAbi`) + `PLUGIN_*_VTABLES` 슬라이스(1곳 선언) + `register_kv_*!` 재작성(const-block 격리 + 슬라이스 기여 + no_mangle v1 entry 제거) + `export_plugin!()` + `ABI_VERSION 1→2` + plugin-cdylib feature. abi_version vtable 필드 V1 잔존(엔진 green). 게이트: technique-api **19/0**(OFF) + **20/0**(plugin-cdylib, 동적 다회 누적) + 봉투 sret round-trip 양축 + 정적/동적 다회 2건 + `cargo build -p llm_rs2` green + clippy clean.
- [x] **V2** `8846d0b3` engine 로더: `try_register_stage`/`try_register_format`(count, 2-pass 원자, `vtables.add(i)`) + `session/plugin_dispatch::register_dynamic_plugins`(open-once) + batch strict 래퍼 + `dynamic_registered_names→_stage_names` rename + vtable abi_version 제거(봉투 단일 게이트). 게이트: `cargo build -p llm_rs2` green + `cargo test -p llm_rs2 --lib` **1260/0**(회귀 0) + lib clippy clean. ⚠ 구 통합 테스트(gate_c_*.rs) v1 ABI 제거로 컴파일 깨짐 → V5 재작성.
- [x] **V3** `31c8405b` 배선: bin_setup W1(register_dynamic_plugins) + W2(make_format, opaque arm) + ensure_builtin_kv_formats_registered startup(C3). make_format 변경 불요(동적 arm abi_version 안 읽음). 게이트: build green(lib+bins) + `--lib` **1260/0** + clippy clean.
- [x] **V4** `62097107` vehicle: synth_q4/example-kv-format/example-keep-recent 에 export_plugin! + **example-multi-format**(mf_q4 Nibble/4 + mf_q8 Byte/8, 서로 다른 desc) + **example-bundle**(stage+format) + **example-no-export**(register_kv_format! 만, capability-0) 신설. 게이트: nm — export 4종 formats_v2+stages_v2 존재, no-export 둘 다 부재. 전 crate test green + 엔진 force-link green.
- [ ] **V5** 게이트: `gate_c_*.rs` 재작성(v2 의미론, CF4 7요소 매핑 + 번들/multi-format distinct/capability-0/wrong-type/원자성/collision) + 전체 sanity(완료 6게이트).
  - **vehicle 준비완료(.so 이름)**: lib{example_multi_format,example_bundle,example_no_export,synth_q4_format,example_keep_recent,example_kv_format}.so (debug, `--features plugin-cdylib`). 게이트 테스트가 `cargo build --message-format=json` 으로 빌드+경로추출(기존 gate_c_format_dlopen_equivalence.rs:48 helper 패턴 재사용).

## iter 로그
- (iter 0, 2026-06-09) ADR-0010 커밋 `48d77243`. 적대적 리뷰 4렌즈 반영 완료. technique-api 매크로/vtable 전모 확보. 컴파일-green 순서 결정. V1 착수.
- (iter 1, 2026-06-09) **V1 완료** `90b04aa7`. 봉투+슬라이스+매크로 재작성+export_plugin!. technique-api 19/0(OFF)·20/0(ON). 엔진/plugin crate build green(abi_version 잔존). 다음=V2(engine 로더).
- (iter 2, 2026-06-09) **V2 완료** `8846d0b3`. try_register_*(2-pass)+register_dynamic_plugins+rename+vtable abi_version 제거. engine lib 1260/0, build green, lib clippy clean. 다음=V3(bin_setup W1/W2 + ensure_builtin 배선).
- (iter 3, 2026-06-09) **V3 완료** `31c8405b`. bin_setup W1(register_dynamic_plugins)+W2(make_format)+ensure_builtin startup. build green(lib+bins), lib 1260/0, clippy clean. 다음=V4(export_plugin! 적용 + multi-format crate + no-export vehicle).
- (iter 4, 2026-06-09) **V4 완료** `62097107`. export_plugin! 3종 + example-multi-format/bundle/no-export 신설. nm: export 4종 v2 entry 존재, no-export 부재. 전 crate test green. 다음=V5(gate_c_*.rs 재작성 + 완료 6게이트).
