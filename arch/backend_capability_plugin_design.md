# Backend 축(3번째 axis) capability plugin 설계 — GATE-C v3

**상태**: 🔶 구현 중 (D1–D8 확정; **D8(2026-06-10)이 D6의 two-trait 가정을 single-trait로 대체**). 완성 시 **ADR-0011로 승격** 예정.
**시작**: 2026-06-10
**SSOT(선행)**: `arch/pipeline_stage_design_v2.md` §1.3/§3.3 · ADR-0005(3축 평행 registry) · ADR-0009(D2 fn-ptr ABI·D7 panic=abort) · ADR-0010(multi-vtable bundle envelope)
**추동 use-case**: KIVI(2-bit asymmetric KV quant, fused GPU attention 커널) — `engine/src/capability/kivi_attention.rs::KiviAttentionBackend`
**grill seed**: `.agent/todos/handoff_backend_axis_grill_seed_2026_06_09.md`

> 북극성 3축(Stage ⊥ Format ⊥ Backend-capability) 중 Stage·Format dlopen 은 완주. 이 문서는 마지막 **Backend 축**을 `.so` plugin 으로 만드는 설계의 grill 결정 원장이다.

---

## 핵심 프레임 — Backend 축은 왜 다른가

Stage·Format 축은 엔진이 플러그인을 **일반적으로(generically) 소비**한다(eviction 파이프라인이 임의 `KVCacheStage` 실행, alloc 경로가 임의 `KVFormat` descriptor 수용). 그래서 새 *종류* 추가도 엔진 재컴파일 0.

**Backend capability 는 소비자가 trait-specific 이다.** `KiviCache` 가 `attention_gen_kivi` 를 부르는 코드는 `KiviAttentionBackend` 라는 **구체 trait 을 이름으로 알아야** 한다. 엔진이 "임의의 capability" 를 generic 하게 부를 방법이 없다(그러려면 layer-tier dyn 디스패치 → §8 INV-HOTPATH-DISPATCH 위반). **capability 는 그것을 부르는 호출부가 곧 정의다.** 이 비대칭이 plugin 단위·ABI 모양을 전부 규정한다.

---

## 확정 결정 (D1–D5)

### D1 — plugin 가능 단위 = "host-정의 capability 카테고리의 새 *구현*", 바인딩은 데이터 선언 (Q1)

- **카테고리(=호출 시그니처)는 host 정의**. capability trait(예: `KiviAttentionBackend`)은 host 소유 → **shared `technique-api` crate 로 이동**해야 plugin 이 동일 정의에 컴파일된다(현재 `engine/src/capability/kivi_attention.rs` 에 위치).
- **구현(커널)은 plugin `.so`**.
- **바인딩(어느 소비자가 어느 impl 을 쓰는지)은 데이터로 선언** — 이름 기반, construction 시 1회 resolve(예: `--kivi-impl <name>`). hot-path 룩업 0.
- 근거: ADR-0005 D4(커널=backend 소유)·deletion-test(소비자 없는 capability 금지)·Stage `DynStage` 패턴과 동형.

### D2 — 표준 메타데이터 함수+구조체 + host 어댑터 (Q2)

Stage/Format 의 `register_*_v2() -> *ExportAbi { vtables: [*VTableAbi{name, fn-ptrs}] }` 패턴을 **그대로 복사** + Backend 전용 `category` 필드 1개 추가:

```rust
// plugin 자기소개 함수 (export_plugin! 에 "한 줄" 추가)
#[no_mangle] pub extern "C" fn register_backend_caps_v2() -> BackendExportAbi;

#[repr(C)] struct BackendExportAbi { abi_version: u32, count: usize, vtables: *const BackendCapVTableAbi }

// 얇은 태그드 포인터 — make/drop/work-fn 은 카테고리별 테이블 안 (D7)
#[repr(C)] struct BackendCapVTableAbi {
    name:     *const c_char,   // "kivi_attn_adreno_v2"   ← 메타데이터(이름)
    category: u32,             // ATTENTION | FOLD | SCORE ← 메타데이터(종류, Backend 에만 추가)
    vtable:   *const c_void,   // category 별 #[repr(C)] 테이블 (host 가 category 로 캐스팅, D7)
}
```

- host 는 `category` 를 읽어 **얇은 어댑터**(`DynKiviAttentionBackend`)로 감싼다 — C 함수 포인터를 엔진 Rust trait 으로 *재구현*(Stage `DynStage` 와 동형). → 소비자(`KiviCache`) **0 수정**.
- host-side **"카테고리 다리" `match category`**: 알려진 카테고리 1개당 arm 1개. 알려진 카테고리의 새 impl = zero-compile / 새 카테고리 = arm 추가 = 재컴파일(D1 결과와 일치).
- **두 레지스트리는 분리 유지**(ADR-0005 D6): plugin `BACKEND_CAPABILITIES`(name 키) ⊥ in-process `CapabilityRegistry`(TypeId 키). 둘은 **공통 산출 타입 `Arc<dyn KiviAttentionBackend>`** 에서 만난다.

### D3 — ABI 형태 = A: host 가 raw bare-C GPU 핸들 빌려주고, plugin 이 직접 OpenCL 실행 (Q3)

- `Tensor` 는 경계 통과 불가(`Arc<dyn Buffer>`+`Arc<dyn Backend>` = host 내부 fat ptr). **대신 그 안의 raw `cl_mem`(`get_cl_mem(tensor)`)·`cl_context`·`cl_command_queue` 를 `*mut c_void` C 핸들로 넘긴다.**
- plugin 이 빌린 핸들로 **자기 커널을 빌드·enqueue**(B안 = host generic 디스패처는 KIVI fused 커널엔 거대·누수 API 라 기각).
- **bare-C 핸들만** 주고받음 — `ocl::core::Mem` 등 Rust 래퍼 금지. plugin 은 순수 C OpenCL API 사용(ocl crate 버전 비결합).

### D4 — 커널 생애주기: make 에서 1회 빌드, host 가 옵션 제공 (Q4)

```
make(cl_ctx, device, build_opts, params) -> handle   // 느린 clBuildProgram 은 여기서 딱 1회, cl_program/kernel 을 handle 에 저장
<call>(handle, cl_queue, buffers..., scalars...) -> i32  // 이미-빌드된 커널 재사용, args+enqueue (매 토큰, 빠름)
drop(handle)                                          // cl_program/kernel 해제
```

- **cl_context 는 make 시점**에 넘긴다(per-call 아님). per-call 은 queue + 버퍼만.
- plugin 은 자기 `.cl` 소스를 `.so` 에 `include_str!`(엔진과 동일) → **컴파일 옵션은 host 의 `build_cl_opts(device)` 결과를 받아 사용**(Adreno 일관성).

### D5 — 버퍼 소유권: 엔진 버퍼는 host 할당, plugin 은 빌려 쓰고 retain 금지 (Q5)

- **입력(q/kv) + 출력(out) 전부 host 가 미리 할당** → plugin 엔 raw `cl_mem` 빌려주기만. plugin 은 **새 GPU 버퍼를 만들어 반환하지 않음**(host `get_cl_mem` allow-list 가 닫혀 있어 plugin 버퍼를 못 읽음).
- 빌린 `cl_mem` 은 **호출 동안만 유효, plugin 은 절대 retain(`clRetainMemObject`) 금지.**
- **plugin 내부 scratch** 만 예외: plugin 이 빌린 cl_context 로 자체 할당·관리(make 에서 잡아 handle 저장, drop 해제), host 로 안 넘어감.

### D6 — plugin-facing trait + `#[repr(C)]` struct-by-ptr 인자 (Q8) ⚠️ **D8에서 single-trait로 대체**

> D6은 "엔진 `&Tensor` trait 무수정 + technique-api 별도 `KiviAttentionPlugin`"의 two-trait 안이었다. **D8(2026-06-10)이 이를 single-trait로 대체** — 엔진 trait을 technique-api로 옮겨 1벌화(C11 해소). 아래 D6 본문은 ABI struct(`KiviAttnArgs` 등) 정의 부분만 유효하고, "엔진 trait 무수정"·"별도 plugin trait" 부분은 D8이 뒤집는다.

- 엔진 소비자가 부르는 trait(`&Tensor`)과 plugin 이 구현하는 표면(raw `cl_mem`)은 **시그니처가 갈린다**(GPU 버퍼는 Stage 의 `read_row` 처럼 CPU 행 평탄화 불가). → **plugin-facing trait 을 별도로** 둔다(technique-api):
  ```rust
  pub trait KiviAttentionPlugin: Send + Sync {
      fn attention_gen_kivi(&self, a: &KiviAttnArgs) -> i32;
      fn kivi_gather_update(&self, a: &KiviGatherArgs) -> i32;
  }
  ```
- **인자는 나열 금지, `#[repr(C)]` 구조체 포인터로 전달**(기존 `make: fn(*const StageParams)`·`StageCtxAbi` by-ptr 관례와 동일). 15개 위치 인자 순서 실수 제거 + 버저닝(필드 끝에 추가) 유리:
  ```rust
  #[repr(C)] pub struct KiviAttnArgs {
      // GPU 자원 (borrow-for-call, C5 retain 금지)
      cl_queue, q_mem, qk_mem, qv_mem, res_k_mem, res_v_mem, out_mem: *mut c_void,
      scores_out: *mut f32, scores_len: usize,   // CPU optional (null=없음)
      num_heads_q, num_heads_kv, head_dim, q_tokens, res_tokens, res_cap: usize,
      scale: f32, bits: u8,
  }
  ```
- C vtable: `attention_gen_kivi: unsafe extern "C" fn(*mut c_void /*handle*/, *const KiviAttnArgs) -> i32`.
- host 어댑터 `DynKiviAttentionBackend` 가 엔진 trait(`&Tensor`) → `get_cl_mem` 으로 cl_mem 추출 → `KiviAttnArgs` 패킹 → C 함수 호출. **&Tensor ↔ cl_mem 다리**.
- **`make` 도 struct 통일**: `make(*const KiviMakeArgs) -> *mut c_void`, `KiviMakeArgs{ cl_ctx, device, build_opts: *const c_char, params.. }`.
- **버저닝 = 봉투 top-level `abi_version`**(축당 1개, 로드 시 체크)에 의존. struct 변경 시 bump → 옛 plugin reject. per-struct `struct_size` 자기서술(Win32 `cbSize`)은 미채택(무거움).
- **엔진-내부 trait `KiviAttentionBackend(&Tensor)` 무수정**(FFI 안 넘음 + 정적 OpenCL impl·`KiviCache` 의존 → 외과적 스코프 밖. 15-arg 정리는 별건 refactor).

### D7 — 한 봉투, N 카테고리: 공통 헤더 + 카테고리별 vtable 포인터 (Q7)

카테고리마다 함수 집합이 다르므로(attention 은 `attention_gen_kivi`+`kivi_gather_update`, fold/score 는 상이) `#[repr(C)]` 엔트리에 카테고리-가변 필드를 직접 못 둔다. → 엔트리는 **얇은 태그드 포인터**, 실제 함수는 카테고리별 테이블:

```rust
#[repr(C)] struct BackendCapVTableAbi { name: *const c_char, category: u32, vtable: *const c_void }

#[repr(C)] struct KiviAttnVTable {   // category == ATTENTION 일 때 vtable 이 가리키는 것 (make/drop 포함)
    make:               unsafe extern "C" fn(*const KiviMakeArgs) -> *mut c_void,
    attention_gen_kivi: unsafe extern "C" fn(*mut c_void, *const KiviAttnArgs) -> i32,
    kivi_gather_update: unsafe extern "C" fn(*mut c_void, *const KiviGatherArgs) -> i32,
    drop:               unsafe extern "C" fn(*mut c_void),
}
```

- **make/drop 도 카테고리 테이블 안** — make 인자가 카테고리별(`KiviMakeArgs` vs `FoldMakeArgs`)이라 공통 헤더에 못 둠.
- host 카테고리 다리(D2): `match category { ATTENTION => { let vt = vtable as *const KiviAttnVTable; vt 로 make → DynKiviAttentionBackend 로 감쌈 } … }`.
- ✅ 봉투·entry 심볼 **1개**(`register_backend_caps_v2`) 유지, 카테고리 **N개**. 카테고리 비결합(fold 추가가 attention 테이블 무관). 포인터 간접 1회 = **로드 타임**, hot-path 무관.
- (b) 카테고리별 심볼(심볼 폭발) / (c) fat union(카테고리 결합 → abi_version 동반 bump) 기각.

---

### D8 — single-trait 채택 (D6의 two-trait 대체, 2026-06-10 사용자 결정)

**결정**: 카테고리당 trait 2벌(엔진 `&Tensor` + plugin `cl_mem`, D6/C11)을 **1벌**로 통합. canonical `KiviAttentionBackend`를 **technique-api로 이동**하고, 시그니처의 `&Tensor`를 **ABI struct `&KiviAttnArgs`/`&KiviGatherArgs`(cl_mem `*mut c_void`)로 재작성**한다. 엔진의 정적 OpenCL impl·live 소비자(`KiviCache`/`kivi_format`)가 모두 이 단일 trait로 이전한다.

- **근거(사용자)**: 플러그인 독립성은 two-trait도 달성하지만, single-trait가 **단일 원천**(C11 dual-maintenance 제거)이라 장기적으로 옳은 방향. 비용(live 경로 재작성·device 회귀검증)을 수용.
- **canonical trait (technique-api)** — `&Tensor` 제거, ABI struct-by-ptr + `i32` 반환(C3 panic=abort, 0=OK·음수=err):
  ```rust
  pub trait KiviAttentionBackend: Send + Sync {
      fn has_kivi_attn_kernel(&self, bits: u8) -> bool;
      fn is_nosub_device(&self) -> bool;
      fn attention_gen_kivi(&self, args: &KiviAttnArgs) -> i32;
      fn kivi_gather_update(&self, args: &KiviGatherArgs) -> i32;
  }
  ```
  엔진은 `pub use technique_api::KiviAttentionBackend`로 재수출 → 모든 소비자 path(`crate::backend::KiviAttentionBackend`) + `CapabilityRegistry` TypeId 키 안정(re-export는 동일 타입 = 동일 TypeId).
- **누가 이 trait를 구현하나** (Stage/Format의 `KVCacheStage` 단일 trait 패턴과 동형):
  1. 엔진 정적 `OpenCLBackend` — 내부에서 cl_mem→커널 dispatch (기존 로직 유지, 시그니처만 `&Tensor`→`&KiviAttnArgs`).
  2. host 어댑터 `DynKiviAttentionBackend` — dlopen plugin의 C `KiviAttnVTable`를 호출.
  3. plugin `.so` — `register_kivi_attention_plugin!` 매크로가 `impl KiviAttentionBackend`를 C thunk로 래핑.
- **live 소비자 재작성 (device-only 회귀면, C12)**: `kivi_format::attention_native`·`kivi_cache::update_gpu`가 `&Tensor`에서 `get_cl_mem`로 cl_mem 추출 → `KiviAttnArgs`/`KiviGatherArgs` 패킹 → 단일 trait 호출. 이들은 **이미 OpenCL-only 경로**라 `get_cl_mem` 사용이 새 backend 누수 아님.
- **정적-생존(static) registry 재사용**: 기존 `BackendCapability{name()}` + `BACKEND_CAPABILITIES`(name 키 linkme) **무수정 유지** — `register_kivi_attention_plugin!`가 **name-only stub**(GPU 불요)을 자동 생성해 이 슬라이스에 기여 → fat-LTO 생존 smoke(`registered_backend_capability_names()`)가 GPU 없이 통과. 실제 KIVI 구현은 동적 vtable + 엔진 정적 impl이 담당(static 슬라이스는 이름 생존만 증명).
- **host 검증 범위**: dlopen ABI round-trip(register→category 다리→어댑터→make/dispatch, 더미 핸들)만 host 게이트로 증명. `&Tensor↔cl_mem` 다리 + live KIVI 수학 무회귀는 **S25 device 재증명(C12)**.

---

## ⚠️ 제약사항 (반드시 준수 — 설계의 의식적 trade-off)

| ID | 제약 | 출처 | 비고 |
|---|---|---|---|
| **C1** | **새 capability *종류*는 host 재컴파일 필요.** zero-compile 은 "알려진 카테고리의 새 impl" 에만 적용. | D1 | capability 의 본질(호출부=정의). 우회 불가. |
| **C2** | **POD-only 불변식 깨짐.** Backend 는 live GPU 자원(cl_mem/ctx/queue)이 `.so` 경계를 넘는 **첫 축**. | D3 | Stage(plan POD)·Format(descriptor POD)이 지킨 불변식을 의식적으로 포기. |
| **C3** | **crash 격리 없음.** `panic=abort`(workspace-wide) → plugin GPU/드라이버 크래시가 **추론 프로세스 전체를 죽임.** Backend plugin = **신뢰 코드**로 취급. | ADR-0009 D7 | 모든 vtable fn-ptr 는 panic 금지, `i32` 코드 반환(0=OK, 음수=에러). |
| **C4** | 경계는 **bare-C 핸들만**(`*mut c_void`). `ocl` crate 타입 금지. | D3 | plugin 이 host 와 같은 ocl 버전에 안 묶임. plugin 은 raw OpenCL C API. |
| **C5** | 빌린 `cl_mem` 은 **호출 동안만 유효, retain 금지.** | D5 | retain 하면 refcount 소유권이 경계를 넘어 위험. |
| **C6** | **host 가 출력 크기를 미리 못 정하는 capability 미지원.** host 가 출력 버퍼를 사전 할당해야 함. | D5 | KIVI=고정 shape 라 OK. 동적 출력 필요 시 모델 확장(2-phase size-query 또는 host-alloc 콜백). |
| **C7** | plugin 은 **host 빌드 옵션을 받아 써야** 함. 독자 도출 시 Adreno 기능 불일치 위험. | D4 | CLAUDE.md Adreno 교훈(`-cl-std`/nosub/fast-math). |
| **C8** | **TypeId 는 `.so` 경계에서 불안정** → cross-boundary 룩업을 TypeId 로 키잉 금지. name/category 사용. | D2 | rustc/crate-version 마다 TypeId 다름 → silent None. |
| **C9** | **KIVI 저장형(`KiviCache`)은 in-engine 유지** — `KVLayoutDesc{block_elems,bits,scale_layout,packing}` 로 표현 불가(per-channel-K ⊥ per-token-V + 비대칭 d+m 2-bit + 잔차 윈도우). **커널만** 경계를 넘는다. | Q1 map | Format 축 plugin 아님. Format 어휘 확장은 직전 세션이 거부(YAGNI). |
| **C10** | hot-path(per-layer/token) 디스패치 + fat-LTO inlining 배리어 — §8 게이트(bit-identical + TBT Δ≤+3%) 재증명 필요. | ADR-0005 L2·§8 | **Q6 미해결**(아래). |
| ~~**C11**~~ | ~~capability 카테고리당 trait 2벌 수동 동기.~~ **해소(D8 single-trait)** — canonical trait 1벌(technique-api). | D6→D8 | — |
| **C12** | live KIVI device 경로(`opencl.rs` impl·`kivi_cache`·`kivi_format`·`init`) 재작성 무회귀. | D8 | **✅ device 부분검증 GREEN(2026-06-10 S25 Adreno)**: `test_backend --backends opencl` KIVI 오라클 Q2/Q4/Q8 attention **L2=0.000000**(GPU `attention_gen_kivi` ≡ CPU `get_view` bit-exact, threshold 0.15/0.08/0.05) + `kivi_gather_update` fill(total=128/q96/res32) 정상 → CB0 커널 마샬링(`borrow_cl_mem` ManuallyDrop, C5) **비트 정확**. L2=0이라 pre/post-CB0 비교 불요(회귀 여지 0). **잔여**: production 소비자 e2e(`kivi_format::attention_native`/`kivi_cache::update_gpu`)는 KIVI CLI gate 부재(argus reject·legacy 삭제 → AB-2 영역)라 미실행 — compile + 오라클 패턴 공유로 저위험. |

---

## 미해결 / 다음 grill 분기

- **Q6 — hot-path §8 재증명**: `.so` 경계를 per-attention(per-layer·per-token)으로 넘어도 §8(bit-identical + TBT Δ≤+3%) 통과하나? 가설 = FFI 호출 비용은 GPU 커널 dispatch 에 비해 무시할 수준이라 OK. **정적(builtin) 경로는 FFI 경계 없음 → C12 device 검증으로 bit-exact 확정(L2=0)**. Q6 은 **dlopen plugin 경로 한정**(실제 KIVI `.so` + production `resolve_kivi_capability` 바인딩 + on-device TBT 측정 필요 = 미착수, synthetic plugin 은 GPU 수학 0). device-only 측정.
- **Q7 — dual-wiring 매크로 + host 어댑터 + dyn 레지스트리**: ✅ **구현 완료(CB1–CB5, host-검증, 2026-06-10)**. technique-api `register_kivi_attention_plugin!`(static `KIVI_ATTENTION_REGS` + dynamic `PLUGIN_BACKEND_CAP_VTABLES` 양 경로) + `export_plugin!` 3rd entry(`register_backend_caps_v2`) + ABI(`BackendCapExportAbi`/`BackendCapVTableAbi` 태그드/`KiviAttnVTable`/`Kivi{Make,Attn,Gather}Args`). engine `capability/dynamic_backend_registry.rs`(`DYN_BACKEND_REGISTRY` + `try_register_backend_cap` + `DynKiviAttentionBackend` 어댑터 + category 다리 `match` + `resolve_kivi_capability`) + `plugin_dispatch` 3rd 축. trait relocation = **D8 single-trait**(`KiviAttentionBackend` `&Tensor`→`&KiviAttnArgs`, technique-api 소유, engine 재수출). 게이트 = `engine/tests/gate_c_backend_cap_dlopen.rs`(dlopen→register→category 다리→어댑터→make/dispatch round-trip + sentinel 마샬링) GREEN · lib 1261/0 회귀 0 · release fat-LTO 빌드+게이트 · clippy --workspace clean. **device(C12: KIVI GPU 실행·§8 TBT)는 미검증** — S25 후속.
- **Q8 — Rust trait ↔ C vtable 발산**: ✅ **해소 → D6** (plugin-facing trait `KiviAttentionPlugin` + struct-by-ptr 인자 + host 어댑터가 &Tensor↔cl_mem 다리).
- **Q9 — 재증명 게이트(device-only)**: "dlopen capability == 정적 capability" 를 무엇으로/어느 device 에서 검증? GPU device-only 라 host CI 등가 없음.
- **cl_program/kernel 수명** = `Arc<Library>` 에 묶여 프로세스 수명까지 유지(never dlclose), Stage/Format 과 동일.

---

## 코드 앵커

- in-process: `engine/src/capability.rs`(CapabilityRegistry) · `capability/kivi_attention.rs`(KiviAttentionBackend) · `pressure/kivi_cache.rs`(KiviCache storage) · `backend/opencl.rs:120`(get_cl_mem)·`:7013`(attention_gen_kivi impl) · `init.rs:337`(현 등록)
- plugin ABI: `crates/technique-api/src/lib.rs`(BackendCapability stub :1178, BACKEND_CAPABILITIES :1193, export_plugin! :1144 — backend "한 줄" 미추가) · `engine/src/session/plugin_dispatch.rs`(register_dynamic_plugins)
- 마샬링 원시: `ocl::core::{Context,CommandQueue,Mem}` = raw C 핸들 1-필드 newtype + `as_ptr()`/`from_raw_*`
