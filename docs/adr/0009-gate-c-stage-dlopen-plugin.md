# ADR-0009: GATE-C — Stage 축 `.so` cdylib dlopen plugin 승격 (북극성 zero-compile install, ADR-0007 D6 해결)

> **Status**: Accepted · **v1(Stage 축) 구현 완주 2026-06-09** (C1 `851e4d03` / C2 `c1e4b986` / C3 `88926e0c` / C4 `a5ae6fd3` / C5 `5a6a44bc`). 게이트: technique-api 15/15 · lib **1260/0** · clippy `--workspace` clean · GATE-C dlopen 통합 테스트 1/1(plan-identity + merge + reject 2종) · release fat-LTO 빌드 green · production `argus_cli --load-plugin <release .so>` startup dlopen 등록 + coherent 추론 실증. (설계 grill: Q1–Q7 잠금 2026-06-09)
> **v2(Format 축, D4) 호스트 구현 완주 2026-06-09** (CF1 `e307ed9f` `FormatVTableAbi`+`register_kv_format!` / CF2 `a8efc8f9` `register_dynamic_formats`+`make_format`+`DynFormat` / CF3 `f2e16602` synth-q4-format cdylib + **example-kv-format 신설**(format 축 템플릿) / CF4 `c7734a94` descriptor-identity 게이트). `KVFormat`=콜백 0(순수 descriptor)이라 v1 의 StageCtxAbi/PlanAbi/PlanArena 불필요 — `KVLayoutDesc`(이미 `#[repr(C)]` POD)를 `layout` fn-ptr 로 값 전달. 게이트: technique-api 16/16 · lib 1260/0 · clippy clean · `gate_c_format_dlopen_equivalence` 1/1(descriptor-identity + floor byte-identical + reject 3종) · release fat-LTO green · nm `register_kv_format_v1` T(feature ON)/부재(OFF). **force-link 비대칭 발견**: `synth_q4` 는 엔진 force-link 정적 → builtin-collision reject 담당, `example_kv_format`(비force-link, 동일 q4_0-like layout) 이 동적 성공 경로 vehicle. **defer(v2 게이트 밖)**: production `--load-plugin`→format 배선 + `--kv-format` 동적 해석(make_format) — dlopen 메커니즘은 격리 테스트로 증명(v1 의 eviction-firing e2e defer 와 동형). **다음 = Backend 축 v3**(`BackendCapability` trait 메서드 미확정 → 설계 grill 선행, GPU device-only).
> **Superseded-in-part by [ADR-0010](0010-gate-c-multi-vtable-bundle-abi.md) (2026-06-09)**: 본 ADR 의 **D2**(단일 `register_kv_*_v1() -> *const VTableAbi` 엔트리)는 ADR-0010 의 **리스트 봉투 엔트리**(`register_kv_*s_v2() -> *ExportAbi`, 한 `.so` 다수 vtable)로 대체, **D6 의 production 배선 defer** 는 ADR-0010 이 완주(cross-axis `register_dynamic_plugins` dispatcher + `--kv-format` `make_format` 배선), **§3 landmine #118 의 심볼명**(`register_kv_*_v1` 단수)은 ADR-0010 의 복수·`_v2` 로 갱신(분리-심볼 불변식은 보존). D1/D3/D4/D5/D7 은 그대로 유효.
> **Date**: 2026-06-09
> **Decision-makers**: 사용자 + 메인 세션 (GATE-C 설계 grill 세션 — landscape map 워크플로 `wf_2e3f5e9d` 6영역 fan-out + 직접 코드 검증 후 Q1–Q7 1문항씩 grill)
> **Selected**: 기존 정적 linkme `KV_CACHE_STAGES` 등록을 **런타임 `.so`(cdylib) dlopen 으로 승격**하되 정적 슬라이스는 **보존(가산, replace 아님)** 한다. plugin 은 단일 `register_kv_stage_v1() -> *const PluginVTableAbi` C-ABI 심볼을 export 하고, `&dyn StageCtx`(C-ABI 불안정 trait object)는 `#[repr(C)]` fn-ptr 테이블(`StageCtxAbi`)로 평탄화한다. `panic` 은 **호스트 abort 를 상속**(정적 stage 와 parity, catch_unwind 격리는 untrusted plugin 요구 시 defer). `KVCachePlan` 출력은 **plugin-arena → host-copy → plan_free**("각자 자기 것 free")로 마샬링. v1 범위 = **Stage 축만**(가장 어려운 fn-ptr ABI 를 먼저 증명 → Format v2 / Backend v3 재사용). 완료 게이트 = **plan-identity**(정적-링크 stage 와 dlopen stage 가 동일 ctx 에 동일 `KVCachePlan`, token-identity 금지). **host-implementable**(CPU only, GPU 불필요).
> **Related**: ADR-0007 D6(GATE-C stub — 본 ADR 이 그 open question 을 **해결**: abi_stable-vs-`#[repr(C)]`, catch_unwind/abort, allocator 경계, `register_plugin!` dual-wiring), ADR-0003(정적 crate + linkme + force-link, §4 fat-LTO self-test, §6 panic=abort 근거), ADR-0004(`KVCacheStage` plan-returning, **D1 변형=엔진 독점**, D5 `StageCtx` 읽기 추상), ADR-0005(D6 3축 평행 registry — 단일 병합 금지), ADR-0008(opaque KV — token-identity 가 틀린 게이트임을 확정), `arch/pipeline_stage_design_v2.md`, `/CONTEXT.md`(stage 축 = 메모리 데이터 제어).

---

## 1. Context

북극성: **zero-compile `.so` plugin 설치로 기능 확장**(Stage ⊥ Format ⊥ Backend-capability 3축). ADR-0003 은 중간 단계로 **정적 crate + linkme `#[distributed_slice]` + force-link** 를 깔았고(확장 = 폴더 + dep 1줄 + force-link 1줄, 기존 로직 수정 0), ADR-0007/0008 은 format 축 opaque KV 를 production 까지 끌어왔다. GATE-C 는 그 정적 메커니즘을 **런타임 dlopen** 으로 승격하는 단계다.

**grill 이 확정한 핵심 — 인프라는 이미 GATE-C 를 위해 forward-design 되어 있다**:
- `technique-api/src/lib.rs` 첫 docstring: *"정적 단계엔 borrow, 미래 `.so` C-ABI 단계엔 동일 추상이 C accessor/flat 스냅샷으로 교체 — forward-compatible."*
- `TensorKind`/`TensorDtype`/`DeviceTarget`/`LayerMetricKind` = `#[repr(u32)]`(discriminant 직접 전달), `TensorShape` = `#[repr(C)]`(POD), `TensorHandle::read_row` = `out: &mut [f32]` out-param(슬라이스 반환 금지 — FFI 평탄화 대비).
- `KVCacheStage` trait = **dyn-safe**(제네릭/Self-by-value/연관타입/`impl Trait` 인자 없음). `plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan>` 단일 메서드(plan-returning, 변형은 엔진 `execute_kv_plan` 독점).
- `libloading = "0.8"` 이 이미 engine 무조건 의존성(`engine/Cargo.toml:85`). `htp_fastrpc/host.rs` 에 `libloading::Library::new` + `dlsym!` 매크로 + 17 `extern "C"` typedef 완성 템플릿 존재.
- 4개 평행 정적 registry 이미 존재: `KV_CACHE_STAGES`(257) / `WEIGHT_STAGES`(403) / `KV_FORMATS`(534) / `BACKEND_CAPABILITIES`(567). Stage 의 C-ABI 설계가 나머지 3축의 템플릿.

**빠진 것(GATE-C 가 신설)**: `register_plugin` C-ABI entry(`#[no_mangle]` 전무), 런타임 registry 병합, trait-object/`Vec` 마샬링, **panic 정책**(미결정).

**중심 긴장(2 root)**: (1) 워크스페이스 release `panic = "abort"`(`Cargo.toml:17`, 근거 = `technique-api/Cargo.toml:8` linkme 생성자 0 / Android 안전)가 `.so` 경계를 넘는 `catch_unwind` 를 원천 봉쇄. (2) `&dyn StageCtx`·`Box<dyn KVCacheStage>` trait object 가 C-ABI 불안정(fat pointer) → vtable/opaque-handle 설계 필요. 나머지 결정(병합·마샬링·축 순서·게이트)은 모두 이 둘에 의존.

---

## 2. Decisions (D1–D7 = grill Q1–Q7)

### D1 — panic 정책: **호스트 abort 상속 (parity)** [Q1]

dlopen plugin 의 panic 은 **정적 builtin stage(h2o/sliding)의 panic 과 동일하게 프로세스 abort** 한다. C-ABI thunk 는 *예상된* 결과만 i32 로 보고(0=plan, 1=NoOp/`None`, 음수=깨끗한 논리 오류)하고 panic 은 잡지 않는다. parity 가 핵심 — GATE-C v1 의 목표는 "같은 코드, 다른 링크 경로의 등가"이므로 plugin 을 정적 stage 보다 높은 crash-safety 기준에 묶지 않는다.

**기각 (A 합성 권고: plugin = `panic="unwind"` + thunk `catch_unwind`)**: (i) Cargo 는 `panic` 을 워크스페이스 **멤버별로 override 못 한다**(`panic`/`lto` 는 root `[profile]` 전용; `[profile.release.package.*]` 는 `opt-level`/`codegen-units` 만) → plugin 이 멤버인 한 abort 상속, unwind 빌드는 별도 워크스페이스/RUSTFLAGS 필요(무거움). (ii) parity 논거상 v1 엔 불필요. **catch_unwind 격리는 untrusted 3rd-party `.so` 로드가 실제 요구가 될 때**(plugin 을 별도 unwind 패키지로 빌드 + thunk 가 plugin 측에서 catch) defer. **기각 (B 전역 unwind)**: `technique-api/Cargo.toml:8` abort/Android 근거 파기 + production 코드크기/성능 손실. **기각 (C panic-free 정적 보장)**: plugin 저자 부담 비현실(Vec 인덱싱·alloc 도 panic), 강제 약함.

### D2 — C-ABI 경계: **opaque handle + 단일 vtable 심볼 + fn-ptr ctx 테이블** [Q2]

plugin 은 **단일 `#[no_mangle] register_kv_stage_v1() -> *const PluginVTableAbi`** 만 export. plugin 인스턴스 = `*mut c_void` opaque handle. `StageCtx` 는 `#[repr(C)]` 함수 포인터 테이블(`StageCtxAbi`)로 평탄화(host 가 concrete ctx 위로 shim 채움). technique-api 가 `StageCtxAbi` 위에 `StageCtx` 를 다시 구현하는 어댑터 `AbiStageCtx` 를 제공 → **plugin 저자는 정적/동적 무관하게 동일한 `impl KVCacheStage` 코드**를 쓴다(write-once, link-either-way). `register_kv_stage!(Ty, "name")` 매크로가 정적(rlib→`#[distributed_slice]`)·동적(cdylib→C export + thunk)을 **동시 생성**(ADR-0007 D6 dual-wiring 구체화). fixed `#[repr(C)]` 로 시작.

```rust
#[repr(C)] pub struct StageCtxAbi {
    ctx: *const c_void,
    current_pos / target_len / layer_idx / n_kv_heads / head_dim: extern "C" fn(*const c_void) -> usize,
    importance:      extern "C" fn(*const c_void, *mut *const f32, *mut usize) -> bool,
    tensor_read_row: extern "C" fn(*const c_void, kind: u32, row: usize, kv_head: usize, out: *mut f32, out_len: usize) -> bool,
    tensor_shape:    extern "C" fn(*const c_void, kind: u32, out: *mut TensorShape) -> bool,  // TensorShape 기존 #[repr(C)]
}
#[repr(C)] pub struct PluginVTableAbi {
    abi_version: u32, name: *const c_char,
    make:      extern "C" fn(*const StageParams) -> *mut c_void,   // StageParams → #[repr(C)]화 (POD 5필드)
    plan:      extern "C" fn(*mut c_void, *const StageCtxAbi, *mut PlanAbi) -> i32,  // 0=plan/1=NoOp/<0=err
    plan_free: extern "C" fn(owner: *mut c_void),
    drop:      extern "C" fn(*mut c_void),
}
#[no_mangle] pub extern "C" fn register_kv_stage_v1() -> *const PluginVTableAbi;
```

**기각 (B abi_stable)**: prefix-type vtable 의 ABI 진화는 매력이나 무거운 의존성 + plugin 저자에게 abi_stable 매크로 강제 → "technique-api + linkme 만 의존" hermetic 계약 파기. v1 과설계 — 2번째 외부 plugin 이 ABI 진화를 실제 요구할 때 deletion-test 로 도입. **기각 (C eager flat 스냅샷)**: `StageCtx` 전체를 값으로 미리 materialize → K/V per-(row,head) lazy read 를 전부 eager dequant copy 해야 해 큰 KV 에서 메모리/시간 폭발, dyn-safe out-param 설계 철학 역행.

**v1 단순화**: 한 `.so` = 한 stage(`register_kv_stage_v1` 단일 vtable). multi-stage per `.so`(`*const VTableArray` 반환)는 future.

### D3 — registry 병합: **가산(additive), 정적 경로 0 변경** [Q3]

`technique-api`(ABI 계약 + 정적 registry, libloading 의존 없음)는 **무변** — `find_stage`/`ensure_builtin_stages_registered`(ADR-0003 §4 load-bearing self-test)/6 테스트 그대로. `engine`(`pressure/eviction/stage_registry.rs`)이 런타임 loader 소유: 별도 `static DYN_REGISTRY: OnceLock<Vec<RuntimeStageReg>>` 에 dlopen 결과 append(`Arc<Library>` 영구 보관 — init-once leak-and-keep, 정당). **단일 진입 `make_stage(name, &StageParams) -> Option<Box<dyn KVCacheStage>>`** 신설 → production 호출부 2곳만 교체. 동적 arm = host `DynStage` 어댑터(`{handle, vtable, _lib: Arc<Library>}`)가 `impl KVCacheStage` 로 vtable 마샬링 → 호출부는 source-agnostic `Box<dyn KVCacheStage>` 수령.

이름 충돌 = **register 시점 eager fail-fast(builtin 항상 우선)**: 동적 name 이 `registered_names()`(정적)에 있으면 `bail!`. silent override 차단(Known Bug #1/#2 류 silent fallback 재발 방지).

**기각 (B 단일 병합 registry)**: `OnceLock<MergedRegistry>` 가 `find_stage` 의 source 를 교체(정적 복사 + 동적 append). 단일 진실원으로 깔끔하나 `find_stage` 반환 타입(`Option<&'static KVCacheStageReg>`) 변경 → `ensure_builtin` + 6 테스트 ripple, 검증된 정적 경로의 blast radius 큼. **기각 (C linkme 슬라이스 런타임 mutate)**: `distributed_slice` 는 링크타임 고정 — 불가.

### D4 — 축 순서: **Stage v1 → Format v2 → Backend v3** [Q4]

GATE-C v1 = **Stage 축만**. 근거 = "가장 어려운 ABI 를 먼저 증명": Stage 는 `StageCtxAbi` fn-ptr 콜백 테이블(가장 어려운 표면)을 강제하므로, 이를 증명하면 Format(콜백 0, `KVFormat` = 2-method descriptor)·Backend 는 **검증된 더 쉬운 템플릿 재사용**. 또한 Stage 는 (i) plan-only → 변형 engine 독점(버퍼가 경계 안 넘음, 가장 안전), (ii) **host-implementable**(CPU only, GPU 불필요), (iii) vehicle 존재(`example_keep_recent`).

**Format(v2)**: `KVFormat` trait 은 `name`+`layout` descriptor 뿐(콜백 0, `KVLayoutDesc` POD)이라 ABI 표면은 더 단순하나, WRITE encoder 가 per-token hot-path → LTO 단절 + (ADR-0007 L2) S25 TBT Δ≤+3% **device 게이트** 필요라 2순위. **Backend(v3)**: `BackendCapability` trait 이 `name()` 1개뿐인 골격(lib.rs:550 "GpuFold 등 첫 instance 가 메서드 확정") + GPU 결합 device-only → 최후순위.

> **ADR-0007 D6 정정**: 거기서 GATE-C 를 일괄 "device-gated"로 표기했으나, **Stage 축 GATE-C 는 host-implementable**(plan-identity 게이트는 GPU 불필요). device-gated 는 Format 축 encoder perf(v2)에 한정된다.

### D5 — plan 마샬링: **plugin-arena + flat(ptr+len) + host-copy + plan_free** [Q5]

`KVCachePlan`(heap Vec 3중첩: `KeepSpec(LayerWide(Vec<usize>)|PerHead(Vec<Vec<usize>>))`, `merges: Vec<WeightedMerge{into, into_weight, from: Vec<(usize,f32)>}>`)을 평탄화: plugin 이 자기 arena 에 plan 을 쓰고 `#[repr(C)] PlanAbi` 로 (ptr+len) 노출 → host 가 즉시 host `KVCachePlan` 으로 **복사**(host allocator) → plugin 이 `plan_free(owner)` fn-ptr 로 자기 arena 회수. **"각자 자기 것 free"** → cross-allocator UB 없음. plan 은 per-eviction 저빈도라 복사 무해.

```rust
#[repr(C)] pub struct PlanAbi { keep_kind: u32, keep_ptr: *const usize, keep_len: usize,
    keep_outer_lens: *const usize, keep_outer_count: usize,
    merges_ptr: *const MergeAbi, merges_len: usize, owner: *mut c_void }
#[repr(C)] pub struct MergeAbi { into: usize, into_weight: f32, from_ptr: *const FromPairAbi, from_len: usize }
#[repr(C)] pub struct FromPairAbi { pos: usize, weight: f32 }
```

`code<0`(plugin 의 깨끗한 논리 오류) → host 가 log + `None`(skip, D1 abort 아님). **PerHead 예약+차단**: `KeepSpec::PerHead` 는 promotion-trigger 전까지 엔진 `bail!`(ADR-0004 line 37). v1 PlanAbi 는 discriminant 만 예약하고 `keep_kind==1` 이면 **host 에서 명시적 bail** → silent garbage 방지. v1 vehicle(`example_keep_recent`)은 LayerWide·`merges` 빈 Vec.

**기각 (B host out-buffer 2-pass)**: plugin 이 keep 길이를 점수 계산 전 모를 수 있어 count-then-fill 2-pass 강제, 가변 merges 부적합. **기각 (C opaque plan handle + accessor fn-ptr)**: 복사 회피하나 apply 루프 per-element FFI → keep 수천이면 호출 폭증, 과설계.

### D6 — 로드 trigger: **`--load-plugin <path.so>` CLI flag (반복 가능)** [Q6]

`make_stage` 를 가진 bin(`argus_bench`/`argus_cli`)에 `--load-plugin <PathBuf>`(`Vec`) 추가 → `ensure_builtin` 직후 `register_dynamic_stages(&paths)`. 결정적·테스트 친화(re-proof 하네스가 정적 vs 동적 *동일* `.so` 지목). `libloading::Library::new` 기본 RTLD_NOW → 누락 심볼 즉시 fail-fast. `dlsym("register_kv_stage_v1")` 실패 / `abi_version` mismatch → `bail!`.

**기각 (B env glob `GATE_PLUGINS=/path/*.so`)**: 배포 친화이나 glob 순서 비결정 + Android SELinux/W^X dlopen 제약 + 임의 `.so` 로드 보안 → device 단계 defer. **기각 (C 고정 search-path + manifest)**: production install 경험이나 인프라 0%에서 최중량 + manifest 포맷 설계 추가, 시기상조.

### D7 — re-proof 게이트: **plan-identity** [Q7]

dlopen 이 바꾸는 유일한 것 = `KVCacheStage` 인스턴스가 정적 슬라이스에서 오느냐 vtable 에서 오느냐. 따라서 게이트 = 동일 입력 ctx 에 대해 정적-링크 stage 와 dlopen stage 가 **동일 `KVCachePlan`**(이미 `#[derive(PartialEq)]`). 이는 `PlanAbi` 마샬링 무손실(round-trip)을 직접 검증한다. 추가로 ① registry 등가(`registered_names() ⊇ 정적 ∪ 동적`) ② 충돌 reject("h2o" `.so` → bail) + `abi_version` mismatch reject ③ fat-LTO 생존 release smoke.

**기각 (B execution-identity)**: 동일 plan 을 `execute_kv_plan` 적용해 byte-identical — `execute_kv_plan` 은 engine-only(dlopen 무관)라 dlopen 이 안 바꾸는 경로를 테스트, plan-identity 통과 시 자명 → 중복. **기각 (C token-identity)**: ADR-0008 이 명시 기각(opaque floor 발산과 dlopen-vs-static 발산 혼동, false-positive). **사용 금지**. F32 round-trip/encoder bit-identity 는 Format 축(v2) 게이트이지 Stage 축 게이트가 아니다(Stage 는 버퍼 안 만짐).

**게이트 함정**: `example_keep_recent` 가 정적 linkme 로도 등록(stage_registry.rs:473)되면 같은 이름 cdylib dlopen 시 D3 충돌-reject 에 걸린다. 등가 테스트는 **dlopen-only 경로 + rlib dev-dep 로 reference 계산**으로 충돌을 피하고, 충돌-reject 는 별도 assertion("h2o" 같은 builtin 이름 `.so` → bail)으로 검증.

---

## 3. Consequences / Landmines

- **abort parity 의 함정**: panic=unwind plugin 을 (미래에) 도입하면 catch 없이 panic 이 abort host frame 으로 unwind 시 UB. catch_unwind 는 반드시 unwind 로 컴파일된 plugin 측 thunk 에서 수행돼야 한다. v1 은 abort 상속이라 무관(정적 stage 와 동일).
- **allocator 경계 cross-free**: plugin 이 alloc 한 `PlanAbi` ptr·plugin 인스턴스(`*mut c_void`)는 **반드시 plugin 의 `plan_free`/`drop` fn-ptr 로만** 해제. host 가 free 하면 cross-allocator UB.
- **fat-LTO 단절**: ADR-0003 §6 의 정적 선택 근거(LTO 인라인이 `.so` 경계에서 끊김). Stage 의 `plan()` 은 per-eviction 저빈도라 안전 — 이 추론을 Format 축(per-token encoder)에 그대로 적용 금지(L2 hot-path 직격).
- **`make` fn 캐스팅**: 정적 `make: fn(StageParams)->Box<dyn>` 는 Rust ABI. `extern "C" fn` 으로 그대로 캐스팅하면 ABI mismatch UB → `register_kv_stage!` 매크로가 정적/동적 두 경로를 **분리 생성**(같은 fn 양쪽 사용 금지).
- **out_len 계약**: `tensor_read_row(out: *mut f32, out_len)` 에서 `out_len == shape().cols` 검증 책임을 host 가 명확히 — 안 하면 plugin 이 head_dim 초과 OOB write 가능.
- **3축 register 통합 유혹 금지**: 단일 `register_plugin` 이 stage/format/backend 를 다 처리하면 ADR-0005 D6(3 평행 registry, 단일 병합 금지) 위반 + 직교 축 결합. 반드시 `register_kv_stage_v1` / `register_kv_format_v1` / `register_capability_v1` 분리.
- **Android SELinux/W^X**: device 에서 임의 경로 dlopen 차단 가능 → GATE-C v1 은 **host-only scope** 명시. device 배포는 후속.

---

## 4. Phasing / Gates

| 단계 | 산출 | 게이트 |
|---|---|---|
| **C1** technique-api | ABI 타입(`StageCtxAbi`/`PlanAbi`/`MergeAbi`/`FromPairAbi`/`PluginVTableAbi`) + `StageParams` `#[repr(C)]` + `AbiStageCtx` 어댑터 + `register_kv_stage!` 매크로 | technique-api cargo test + 정적 등록 무회귀 |
| **C2** engine loader | `DYN_REGISTRY` OnceLock + `register_dynamic_stages`(dlopen+abi_version+충돌 fail-fast) + `make_stage` + `DynStage`(vtable 마샬링, PlanAbi↔KVCachePlan, plan_free/drop, PerHead bail) | engine cargo test |
| **C3** 배선 | 호출부 2곳(`build_bench_loop:101`/`session:636`) `find_stage`→`make_stage` + `--load-plugin` flag | 빌드 + 정적 stage e2e 무회귀 |
| **C4** vehicle | `example-keep-recent` `crate-type=["cdylib","rlib"]` + `register_kv_stage!` | cargo build cdylib 산출 + `nm` 심볼 확인 |
| **C5** re-proof | `test_dlopen_stage_equivalence`(plan-identity + registry 등가 + 충돌/버전 reject) + fat-LTO release smoke | 신규 테스트 통과 + lib 무회귀(≥1260/0) + release smoke green |

**GATE-C v1 완료 = D7 plan-identity 통과 + 전체 sanity green + 정적 경로 무회귀.** 이후 v2(Format 축, encoder device 게이트) / v3(Backend 축, capability trait 메서드 확정 후)가 C1–C2 인프라(vtable/병합/panic/마샬링) 재사용.
