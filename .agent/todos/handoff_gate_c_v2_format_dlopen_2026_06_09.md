# Handoff: GATE-C v2 — Format 축 `.so` dlopen plugin 호스트 완주

**작성**: 2026-06-09
**HEAD**: `c7734a94 test(engine): GATE-C v2 CF4 — format dlopen descriptor-identity 재증명 게이트` (+ 후속 docs 커밋)
**브랜치**: master (미푸시)
**작성자**: 메인 세션

**다음 세션 진입 문장**: **"GATE-C v2(Format 축 .so) 호스트 완주 — 다음은 GATE-C v3(Backend capability 축) 설계 grill(BackendCapability trait 메서드 미확정), 또는 format 축 production 배선(--load-plugin→format + --kv-format 동적 해석) e2e."**

---

## TL;DR

북극성("zero-compile `.so` plugin")의 **Format 축**을 정적 linkme 등록에서 런타임 dlopen 으로 승격(호스트). v1(Stage 축) C1–C5 패턴을 거울로 CF1–CF4 구현. `KVFormat`=콜백 0(순수 descriptor)이라 v1 의 StageCtxAbi/PlanAbi/PlanArena **전부 불필요** — `KVLayoutDesc`(이미 `#[repr(C)]` POD)를 `layout` fn-ptr 로 값 전달. 정적 `KV_FORMATS` 경로 보존(가산). **완료 게이트 = descriptor-identity**(동적 vtable 경로 descriptor == 정적 == 알려진 q4_0-like) 통과. **멈춘 이유**: ADR-0009 D4/D7 게이트 충족 + 전 sanity green → v2 호스트 마일스톤 완주. 다음 = Backend 축(v3) 또는 format production e2e.

---

## 진행 상태 (4 커밋 + docs, 무회귀)

| 커밋 | 단계 | 내용 | 게이트 |
|---|---|---|---|
| `e307ed9f` | **CF1** technique-api | `KV_FORMAT_ABI_VERSION` + `FormatVTableAbi`(`#[repr(C)]`: abi_version/name/make/layout/drop) + `unsafe impl Sync` + `register_kv_format!` dual-wiring 매크로 | technique-api **16/16**(+POD round-trip) |
| `a8efc8f9` | **CF2** engine loader | `format/dynamic_format_registry.rs`(신설): `DYN_FORMAT_REGISTRY` OnceLock + `register_dynamic_formats`(dlopen+abi+충돌/dup fail-fast) + `dynamic_registered_format_names` + `make_format`(정적 우선→동적 fallback) + `DynFormat`(vtable layout/drop) | 빌드 green · clippy clean |
| `f2e16602` | **CF3** vehicle | synth-q4-format `crate-type=["cdylib","rlib"]`+plugin-cdylib+`register_kv_format!` / **신규 `crates/techniques/example-kv-format`**(format 축 템플릿) | 정적 각 1/1 · `nm register_kv_format_v1` T(ON)/부재(OFF) |
| `c7734a94` | **CF4** 재증명 게이트 | `engine/tests/gate_c_format_dlopen_equivalence.rs` 단일 #[test] 7요소 | 통합 테스트 **1/1** |

**최종 게이트(전부 green)**: technique-api 16/16 · lib **1260/0**(`test_release_unused_pages_rss_reduction` 1건은 RSS OS-noise flaky — 격리 재실행 ok, pressure/kv_cache 변경 0 → 회귀 아님) · clippy `--workspace -D warnings` clean · `gate_c_format_dlopen_equivalence` 1/1 · release fat-LTO **51.87s green**(GATE-C v2 코드 LTO 생존) · `nm` synth_q4.so/example_kv_format.so feature ON=`register_kv_format_v1 T`, OFF=부재.

---

## 다음 작업 (택1, grill 권장)

1. **GATE-C v3 = Backend capability 축** (ADR-0009 D4 순서 최후). `BackendCapability` trait 이 `name()` **1개뿐**(`lib.rs:988` "GpuFold 등 첫 instance 가 메서드 확정") → **trait 인터페이스 자체가 미설계**. 설계 grill 선행 필수(메서드 확정 + GPU 결합). GPU device-only(host-implementable 아님). CF1 인프라(vtable 패턴/OnceLock 병합/dlopen 로더) 재사용.
2. **format 축 production 배선 e2e** (v2 게이트 밖, defer 됨 — v1 eviction-firing e2e 와 동형). `bin_setup` `build_inference_ctx` 가 `register_dynamic_formats(&args.load_plugin)` 도 호출 + `--kv-format` 해석을 `find_kv_format`→`make_format`(동적 fallback)로 전환. **선행 고려**: `--load-plugin` 한 리스트에 stage `.so` + format `.so` 가 섞이면 `register_dynamic_stages`(심볼부재 batch bail)와 `register_dynamic_formats`(동) 가 상호 reject → per-`.so` 심볼 프로빙 dispatcher(`register_dynamic_plugins`) 필요. + opaque 저장 경로(OpaqueBuffer)로 동적 format descriptor 흐름 확인.
3. (v1 잔여) Stage 축 eviction-firing e2e(argus_bench --load-plugin + signal_injector, directive→policy 매핑 선행).

---

## Landmines / 핵심 발견

- **force-link 비대칭(v1 과 결정적 차이)**: `synth_q4` 는 엔진 lib 이 `use synth_q4_format as _;`(`format/builtin_kv_formats.rs`)로 **무조건 정적 등록** → 임의 바이너리에서 `find_kv_format("synth_q4")==Some` → `register_dynamic_formats([synth_q4.so])` 는 **builtin-collision reject**(v1 의 example_keep_recent 은 force-link 안 됐음). 그래서 동적 성공 경로(merge/make_format/dup)를 증명하려면 **비force-link 이름**이 필요 → `example-kv-format` 신설(동일 q4_0-like layout). synth_q4 는 builtin-collision reject 만 담당. **production 배선 시 이 비대칭이 dispatcher 설계를 강제**(다음작업 #2 선행).
- **`KVFormat`=콜백 0**: ctx/plan 없음 → vtable=make/layout/drop 뿐. `KVLayoutDesc`/`Packing`/`ScaleLayout` 가 이미 `#[repr(C)]`/`#[repr(u32)]` 라 reshape 0(값 전달). v1 의 PlanArena cross-allocator 마샬링이 format 엔 불필요(descriptor 는 stack POD).
- **device perf 게이트는 v2 등가 게이트가 아니다**: dlopen 은 encode 경로를 안 바꾼다(엔진 floor `encode_via_descriptor` 소유). descriptor-floor 의 per-token TBT(ADR-0007 L2 opaque 비용)는 **정적 synth_q4 에도 이미 존재** = dlopen 무관. v2 등가 = descriptor-identity(host, GPU 불요). (목표가 device 게이트를 범위 밖으로 둔 근거.)
- **`register_kv_format!` 매크로 = linkme 직접 dep 필수**: proc-macro 가 `linkme::` 경로 하드코딩(v1 와 동일). plugin crate 는 technique-api + linkme 2 deps. 정적 force-link 빌드는 `plugin-cdylib` OFF → `#[unsafe(no_mangle)] register_kv_format_v1` 미emit(충돌 차단).
- **defer = production 배선**: `register_dynamic_formats`/`make_format` 는 pub 이나 production 호출부 0(unwired). lib crate pub API 라 dead_code 경고 없음. dlopen 메커니즘은 CF4 격리 테스트가 증명(v1 milestone 과 동형 — C5 도 격리 증명, e2e defer).
- **engine/Cargo.toml drift 유지**: `microbench_score_readback` `[[bin]]`(untracked `microbench/score_readback.rs`) 미커밋. v2 는 engine/Cargo.toml 을 **건드리지 않음**(format registry 는 신규 파일 + format.rs 1줄). 커밋 금지.

---

## 참조

- SSOT: `docs/adr/0009-gate-c-stage-dlopen-plugin.md`(Status 헤더에 v2 CF1–CF4 완주 추가). D4=Format 축 순서, D7=재증명 게이트(plan→descriptor-identity 동형).
- 코드 앵커: `crates/technique-api/src/lib.rs`(FormatVTableAbi/register_kv_format! 매크로, "GATE-C v2 Format 축" 섹션 ~line 982) / `engine/src/format/dynamic_format_registry.rs`(DYN_FORMAT_REGISTRY/register_dynamic_formats/make_format/DynFormat) / `crates/techniques/{synth-q4-format,example-kv-format}/`(cdylib vehicle) / `engine/tests/gate_c_format_dlopen_equivalence.rs`(CF4 게이트).
- 빌드: `.so` = `cargo build -p {synth-q4-format,example-kv-format} --features plugin-cdylib` → `target/debug/lib*.so`. 게이트 = `cargo test -p llm_rs2 --test gate_c_format_dlopen_equivalence`.
- 선행 handoff: `handoff_gate_c_v1_stage_dlopen_2026_06_09.md`(Stage 축 v1 — GATE-C 전 단계). 진행 원장: `gate_c_v2_format_progress_2026_06_09.md`.
