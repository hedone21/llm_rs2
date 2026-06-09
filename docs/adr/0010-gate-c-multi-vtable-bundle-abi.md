# ADR-0010: GATE-C 멀티-vtable bundle ABI — 한 `.so` 다수 capability + cross-axis open-once dispatch (ADR-0009 D2 supersede + production 배선 해결)

> **Status**: Proposed · 설계 grill 잠금 2026-06-09 (사용자 + 메인 세션). 구현 미착수 — 본 ADR 이 V1–V5 구현의 SSOT.
> **Date**: 2026-06-09
> **Decision-makers**: 사용자 + 메인 세션 (Format 축 production 배선 논의 중 "한 `.so` 에 여러 기능" 의문 제기 → 멀티-vtable bundle 로 확장. landscape 매핑 워크플로 `wf_315389a6` 3영역 fan-out + linkme-in-cdylib 실측 2건 후 Q1–Q2 grill).
> **Selected**: GATE-C 의 동적(dlopen) 엔트리 ABI 를 **"vtable 1개 반환"에서 "vtable 리스트(봉투) 반환"으로 교체**한다(`ABI_VERSION 1→2`, 미푸시라 공존 아닌 **대체**). 한 plugin `.so` 가 한 축에 **다수** capability 를 실을 수 있다(예: quant 패밀리 format 3종). 작성자는 기존 per-item `register_kv_stage!` / `register_kv_format!` 를 그대로 쓰되(이제 한 `.so` 에 여러 번 호출 가능 — linkme `distributed_slice` 누적), `.so` 당 **`export_plugin!()` 1회**로 per-axis 엔트리 심볼을 emit 한다. 엔진 로더는 `.so` 를 **한 번만 dlopen** 하고 present 한 모든 per-axis 엔트리를 호출해 각 축 registry 에 등록한다(`register_dynamic_plugins` cross-axis dispatcher, **registry 병합 없음** — ADR-0005 D6 준수). 동시에 ADR-0009 가 defer 한 **production 배선**(`--load-plugin`→format + `--kv-format` 동적 해석)을 해결한다.
> **Supersedes**: ADR-0009 **D2**(단일 `register_kv_*_v1() -> *const VTableAbi` 엔트리 → 본 ADR 의 리스트 봉투 엔트리), ADR-0009 **D6 의 defer**(production 배선 미완 → 본 ADR 이 완주), ADR-0009 **§3 landmine #118 의 심볼명**(`register_kv_stage_v1`/`register_kv_format_v1`/`register_capability_v1` 단수·`_v1` → 본 ADR 의 복수·`_v2` `register_kv_stages_v2`/`register_kv_formats_v2`/`register_backend_caps_v2`). **단 #118 의 불변식(3축 분리 심볼·통합 registry 금지)은 그대로 유효** — 심볼명(문자)만 갱신, 분리 원칙(정신)은 보존. ADR-0009 의 D1(panic=abort parity) / D3(가산, 정적 0 변경) / D4(축 순서) / D5(`StageCtxAbi`/`PlanAbi`/`PlanArena` plan 마샬링) / D7(plan-identity·descriptor-identity 재증명 게이트) 는 **그대로 유효**(엔트리 모양만 바뀌고 ctx/plan 마샬링·panic·가산성은 불변).
> **Related**: ADR-0009(GATE-C 단일-vtable dlopen — 본 ADR 이 그 엔트리를 진화), ADR-0005(3축 평행 registry, **단일 병합 금지** — dispatcher 가 준수), ADR-0003(정적 linkme + force-link, §4 fat-LTO self-test — 정적 경로 보존), ADR-0007/0008(opaque KV — format descriptor 가 production storage 구동), `arch/pipeline_stage_design_v2.md`, `/CONTEXT.md`.

---

## 1. Context

북극성: **zero-compile `.so` plugin 설치로 기능 확장**(Stage ⊥ Format ⊥ Backend-capability 3축). ADR-0009 가 Stage 축(v1)·Format 축(호스트)을 단일-vtable dlopen 으로 승격했고, 메커니즘은 격리 테스트로 증명됐으나 **production 배선은 defer** 됐다(`--load-plugin` 은 stage 만, `--kv-format` 은 정적 `find_kv_format` 만).

production 배선을 시작하며 사용자가 근본 의문을 제기했다: **"왜 stage `.so` 와 format `.so` 를 나누나? 한 `.so` 에 둘 다 실을 수 있지 않나?"**

매크로 원문 검토로 확인된 사실:
- `register_kv_stage!` 는 `register_kv_stage_v1`, `register_kv_format!` 는 `register_kv_format_v1` 을 emit — **서로 다른 심볼**. 한 crate 에서 둘 다 호출해도 충돌 없음 → **한 `.so` 가 stage 1 + format 1 을 동시에 export 할 수 있다.** "축별 `.so` 분리"는 불필요한 전제였다.
- **단, 같은 매크로를 한 `.so` 에서 두 번 부르면** `register_kv_*_v1` no_mangle 심볼이 중복 → 링크 에러. 즉 단일-vtable ABI(v1)는 **축당 최대 1개**로 묶인다.

**multi-per-axis 의 실수요**: 양자화 스킴은 **패밀리로** 배포된다(NF4 + AWQ-int4 + GPTQ-int4 한 라이브러리). vendor backend SDK 도 op 를 묶음으로 낸다(matmul + flash-attn accel). 정적(force-link) 경로는 모듈 분리로 이미 다수 등록이 되지만, **dlopen 단일심볼 경로만** 막혀 있다 — 그게 북극성("zero-compile `.so` 배포")의 정중앙이다.

**기회**: GATE-C v1·v2 는 **전부 미푸시·내부 한정**(외부 plugin 소비자 0). 따라서 단일-vtable ABI 를 영구 공존 부채 없이 **깔끔하게 리스트-vtable 로 갈아끼울** 수 있다.

**실측 선결(추측 금지)**: 리스트 누적을 `linkme distributed_slice` 로 한다면 그것이 **cdylib 내부에서**(선언+적재+판독 같은 `.so`) fat-LTO release 에 살아남는지 모름 → `/tmp` 최소 probe 로 검증:
- ① 모듈 간 기여 2건 → dlopen 후 `count==2`, `sum==42`, `__start/__stop_linkme_*` 심볼 해소. **PASS**
- ② const-block-scoped `static __VT`(매크로 패턴, invocation 마다 동일 이름) 3회 → `count==3`, `sum==25`. **PASS**

→ 작성자 ergonomics 보존(per-item 매크로) + `.so` 당 다수 vtable 이 실현 가능함을 확정.

---

## 2. Decisions (E1–E7)

### E1 — 엔트리: **단일 vtable → 리스트(봉투) 반환, v1 ABI 대체** [grill Q1]

동적 엔트리 심볼이 vtable 하나가 아니라 **봉투(개수 + 배열 포인터)**를 반환한다. `ABI_VERSION 1→2`. 단일 format `.so` = `count==1`(특수 케이스 아님). **v1 완전 제거**(미푸시 → 공존 cruft 0). `abi_version` 은 vtable 마다 두지 않고 **봉투에 1회**(.so 는 ABI 하나).

```rust
// abi_version 은 vtable 에서 제거(봉투로 이전).
#[repr(C)] pub struct FormatVTableAbi { name: *const c_char, make, layout, drop }              // 4 fn-ptr
#[repr(C)] pub struct PluginVTableAbi { name: *const c_char, make, plan, plan_free, drop }       // stage (D5 마샬링 불변)

// 봉투 — .so 가 한 축의 vtable 들을 한 번에 신고. by-value 반환(E3).
#[repr(C)] pub struct FormatExportAbi { abi_version: u32, count: usize, vtables: *const FormatVTableAbi }
#[repr(C)] pub struct StageExportAbi  { abi_version: u32, count: usize, vtables: *const PluginVTableAbi }
```

- **봉투 abi_version 1회 검사로 충분**: 한 `.so` 는 ABI 하나(봉투 1개)라 로더가 봉투 abi_version 만 검사하면 그 `.so` 의 모든 vtable 이 같은 ABI 임이 보장된다. 개별 vtable 에 abi_version 부재로 인한 손실 0.
- **`unsafe impl Sync` 유지 필수**: abi_version 을 빼도 `FormatVTableAbi`/`PluginVTableAbi` 는 `name: *const c_char` raw ptr 를 보유 → auto-`Sync` 안 됨. `register_kv_*!` 가 `static __VT: FormatVTableAbi` 를 distributed_slice 에 넣으려면(static element 요건) 두 타입의 `unsafe impl Sync`(현 `lib.rs:1018`/`413`)를 **그대로 유지**해야 한다(필드 삭제는 Sync 불변식과 무관 — SAFETY 주석 'vtable 불변' 유효).

**기각 (B v2 를 v1 옆에 추가, 로더 v2→v1 fallback)**: 외부 소비자 0 인데 두 ABI 경로를 영구 유지 = 순부채. 곧 외부 배포가 있다면 의미 있으나 현재 아님. **기각 (C 축-분리 플래그 `--load-stage-plugin`/`--load-format-plugin`)**: 번들 `.so` 를 두 플래그에 중복 지정해야 함 → 번들 모델 파괴, 사용자 의문의 정반대.

### E2 — 작성자 모델: **per-item `register_kv_*!` 유지(슬라이스 누적) + `export_plugin!()` 1회** [grill Q2]

작성자는 기존 호출법을 그대로 쓰되 한 `.so` 에 여러 번 부를 수 있다. `.so` 당 `export_plugin!()` 1회로 per-axis 엔트리를 emit.

```rust
register_kv_format!("nf4",  || Box::new(Nf4));
register_kv_format!("awq4", || Box::new(Awq4));   // ← 한 .so 에 여러 format
register_kv_stage!("myevict", make_evict);
export_plugin!();   // .so 당 1회 — register_kv_formats_v2 + register_kv_stages_v2 emit
```

- `register_kv_*!`: **정적** `KV_FORMATS`/`KV_CACHE_STAGES` 기여 + **동적** `PLUGIN_*_VTABLES` 기여(plugin-cdylib 게이트). 기존 v1 의 `#[no_mangle] register_kv_*_v1` entry emit 은 **제거**. **`__make`/`__layout`/`__plan`/`__plan_free`/`__drop` thunk 본문은 무변경**(ADR-0009 #116 준수 — `$make`(Rust-ABI fn)는 thunk *내부* `let make_fn: fn(..)->__Handle = $make;` 호출 전용, `extern "C"` 직접 캐스팅 금지). 즉 v2 재작성 = **entry 교체 + 슬라이스 기여만**.
- **⚠ 다회 호출 전제 = 모든 기여 static 을 invocation 별 익명 `const _: () = {...}` 스코프 안에**: linkme **static element 는 ident 를 rename 하지 않으므로**(fn element 만 `_LINKME_ELEMENT_*` rename — `linkme-impl element.rs`) 심볼 유일성이 오직 enclosing scope 에서 나온다. **현 정적 기여 `__REGISTER_KV_FORMAT_REG` 는 매크로 top-level(const-block 밖, `lib.rs:1036`)에 있어 한 crate 1회로 제약**된다(2회 → `__REGISTER_KV_FORMAT_REG` E0428 중복정의). 따라서 **"정적 기여 무변경"은 거짓** — 다회(nf4+awq4 한 `.so`)를 지원하려면 정적·동적 양 기여 static 을 **둘 다 const-block 안으로 이동**해야 한다(V1 매크로 본체 수정 대상).
- `export_plugin!()`(plugin-cdylib 게이트 — force-link 정적 빌드에선 미emit, 안 그러면 다수 force-link plugin 의 `register_kv_*s_v2` 가 엔진 바이너리에서 충돌): per-axis `register_kv_formats_v2()`/`register_kv_stages_v2()` 를 emit, 각자 자기 `PLUGIN_*_VTABLES` 슬라이스를 봉투로 by-value 반환. **기여 0 인 축은 `count==0`**(빈 distributed_slice — ELF/Linux·Android `__start==__stop`, `from_raw_parts(ptr,0)` 안전).
- **슬라이스 선언은 반드시 technique-api 1곳**(무조건): linkme **section 명이 선언 static 이름으로 결정**되므로(`linker.rs` `format!("linkme_{}", ident)`) plugin 측 선언은 cross-crate 기여를 깬다. 모든 plugin 의 기여는 technique-api 가 노출한 동일 path 를 참조. 정적 빌드에선 빈 채 무해(엔진은 안 읽음).

```rust
// technique-api 에 1곳만 선언.
#[distributed_slice] pub static PLUGIN_KV_FORMAT_VTABLES: [FormatVTableAbi] = [..];
#[distributed_slice] pub static PLUGIN_KV_STAGE_VTABLES:  [PluginVTableAbi] = [..];
```

**기각 (B 리스트 매크로 한 방 `register_kv_formats!{ "a"=>ma, "b"=>mb }`)**: linkme-in-cdylib 실측 통과 후 기각 — per-item/모듈-분산 ergonomics 상실 + 기존 호출부(synth_q4/example) 전부 재작성. **기각 (C multi ABI 정의만, count=1 영구)**: 사용자가 multi 를 지금 정의하기로 명시 선택(향후 quant 패밀리 실수요).

> **landmine #ADR0009-118 준수 확인**: `export_plugin!()` 는 **단일 통합 심볼이 아니다** — per-axis 분리 심볼(`register_kv_formats_v2` ⊥ `register_kv_stages_v2`)을 emit 하고, 각자 **분리된** `PLUGIN_*_VTABLES` 슬라이스 → 분리된 DYN registry 로 간다(병합 0). 매크로는 작성자 편의(2회→1회)일 뿐 ADR-0005 D6 의 "3 평행 registry, 단일 병합 금지"를 위반하지 않는다.

### E3 — 봉투 **by-value 반환** + abi_version 봉투 이전 [grill Q2 leaf]

`register_kv_formats_v2() -> FormatExportAbi`(값 반환). `count`/`vtables` 는 linkme `Deref`(`static_slice()`) 경유 **런타임 평가**(`.len()`/`.as_ptr()` 는 const-context 불가) → 봉투를 `const static` 으로 못 둠 → 포인터 반환 시 봉투 자체의 수명 문제. **값 반환**이 이를 회피(봉투는 호출자 스택, `vtables` 는 `.so` static 슬라이스의 base(`&PLUGIN_*[0]`)를 가리켜 `.so` 수명 동안 유효).

- **FFI 건전성(sret)**: `sizeof(FormatExportAbi) = 24B`(u32 4 + pad 4 + usize 8 + ptr 8) > 16B → x86-64 SysV 는 隐 sret 포인터(rdi) 경로를 쓴다. **양측이 동일 `#[repr(C)]` 레이아웃에 합의**하는 한 건전 — 호출측(libloading)은 `unsafe extern "C" fn() -> FormatExportAbi` 로 선언하면 컴파일러가 동일 sret 규약을 생성한다. aarch64 AAPCS 도 동형(>16B → indirect). (기존 `layout: extern "C" fn(..) -> KVLayoutDesc` 는 8B POD 라 register 반환 — 봉투는 sret 경로라 V1 게이트에서 별도 round-trip 검증.)

**기각 (포인터 반환 `-> *const FormatExportAbi`)**: 봉투를 어디에 둘지(런타임 값이라 static 불가) 문제 발생. 값 반환이 단순·안전.

### E4 — **양축 대칭 + backend v3 동일 합류**

리스트화는 stage·format **양축 동시** 적용(ABI_VERSION 양쪽 2). 비대칭(stage 단일/format 리스트)은 north-star(3축 균일 plugin) 위반. v3 Backend 축은 `register_backend_caps_v2` + `PLUGIN_BACKEND_VTABLES` 를 `export_plugin!()` 에 **한 줄 추가**로 합류(ADR-0009 D4 축 순서 유지).

### E5 — 로더: **count 기반 `try_register_*` + open-once cross-axis dispatcher** (병합 없음)

```rust
// 각 축 registry 모듈에 per-.so 코어 추출 — 반환 = 등록한 개수
fn try_register_format(lib: &Arc<Library>, path) -> Result<usize>   // dlsym register_kv_formats_v2; 없으면 Ok(0);
                                                                    // 있으면 봉투 abi 검사 → 2-pass(아래) → 등록 수
fn try_register_stage (lib: &Arc<Library>, path) -> Result<usize>   // register_kv_stages_v2 대칭

// production 단일 진입 — .so 당 dlopen 1회, present 한 모든 축 등록, Arc 공유
pub fn register_dynamic_plugins(paths: &[PathBuf]) -> Result<()> {
    for path in paths {
        let lib = Arc::new(unsafe { Library::new(path) }?);          // 1회 dlopen
        let n = try_register_stage(&lib, path)? + try_register_format(&lib, path)?;
        if n == 0 { bail!("plugin {}: 등록된 capability 0 (export_plugin! 누락 또는 빈 plugin)", path); }
    }
    Ok(())
}
```

- **봉투 → vtable 포인터 도출(F3)**: 로더는 봉투를 by-value 로 받은 직후 `vtables`(= `.so` static 슬라이스 base)를 읽고, `i in 0..count` 에 대해 `vtable_ptr_i = envelope.vtables.add(i)` 를 `RuntimeReg.vtable` 에 저장한다. **봉투 자체(스택)의 주소를 잡지 않는다**(스택은 즉시 폐기; vtable 은 `.so` static 이라 Arc 수명 동안 유효 — distributed_slice 는 read-only 섹션 고정 배열이라 원소 주소 안정, Vec 같은 이동 없음). `RuntimeReg.vtable` SAFETY 주석을 단일→배열원소(`.so` static, `add(i)` 안정)로 갱신.
- **per-name 충돌 검사 + per-`.so` 원자성(2-pass, G3)**: 충돌 검사(`find_kv_format` builtin + 동적 registry 중복 + 봉투 내부 중복)는 봉투 `count` 순회의 **각 vtable 이름마다** 수행. 부분 등록 방지 = **2-pass**: ① 봉투의 모든 이름을 먼저 검사(하나라도 충돌 → 즉시 bail, push 0), ② 전부 통과 시 write-lock 1회로 일괄 push. 이로써 `[synth_q4(충돌), awq4(정상)]` 봉투가 awq4 부분등록을 남기지 않는다(다른 `.so` 의 awq4 위양성 dup 방지 — 게이트가 in-process 라 leftover 가 후속 단언 오염).
- **wrong-type 은 더 이상 reject 아님(G2, 의도된 변화)**: v1 은 분리 심볼이라 stage-only `.so` 를 format 로더에 넣으면 "심볼 부재" reject 됐다. v2 는 한 `.so` 를 1회 dlopen 하고 양축 `try_register_*` 를 둘 다 호출하므로, **단일축 `.so`(stage 1 + format 0)는 번들의 부분집합으로 정당** → `n>=1` 성공 수용. 어느 축에도 기여 0 인 `.so` 만 capability-0 bail. (게이트가 이 시나리오를 명시 단언 — E7.)
- 번들 `.so` 의 stage-reg 와 format-reg 가 **같은 `Arc<Library>` 를 공유**(정직한 단일 핸들).
- 기존 batch `register_dynamic_stages`/`register_dynamic_formats`(gate 테스트·축-격리 진단용)는 `try_register_*` 위의 얇은 strict 래퍼로 보존(0 등록 시 "심볼 부재" bail — 기존 계약·메시지 유지). `dynamic_registered_format_names()` 가시화 불변(`register_dynamic_plugins` 도 동일 `DYN_FORMAT_REGISTRY` 적재).
- **registry 병합 없음**: DYN_REGISTRY(stage) ⊥ DYN_FORMAT_REGISTRY(format) 분리 유지(ADR-0005 D6).

**기각 (A2 partition + 축별 재-dlopen)**: 번들 `.so` 를 stage 로더·format 로더가 각각 따로 dlopen → 같은 `.so` 3회 open + 핸들 2개 분리 보유. 기능은 맞지만 번들을 "별개 등록 2건"인 척하고 v3 에 깔끔히 확장 안 됨(세 번째 partition+로더 추가). open-once-shared-Arc 가 정직·확장적.

### E6 — production 배선 (ADR-0009 D6 defer 해결)

- **W1**: `bin_setup::build_inference_ctx` 의 `register_dynamic_stages(&load_plugin)` → `register_dynamic_plugins(&load_plugin)`(stage+format 혼합 허용).
- **W2**: `--kv-format` 해석을 정적 `find_kv_format` → `make_format`(정적 우선 → 동적 fallback)으로 전환. **`make_format` 은 이미 존재**(`dynamic_format_registry.rs:127`, CF2 `a8efc8f9` — 기술한 정적-우선·동적-fallback 동작을 그대로 구현). W2 = 신설이 아니라 (i) `bin_setup.rs:98` 의 `find_kv_format`/`bin_setup.rs:116` inline `(reg.make)().layout()` 를 `make_format(name)` 호출로 교체 + (ii) V2 의 봉투 ABI 도입에 맞춰 `make_format` 의 **동적 arm 만 봉투-aware**(`DynFormat` 가 `register_kv_formats_v2` 봉투 경로로 적재된 vtable 을 읽도록)로 재작성. 내장 typed(f32/f16/q4_0/q8_0)은 `builtin_format_dtype` 이 분기(이름 기반 — 동적 format 은 자동 opaque). opaque arm 에서 `make_format(name).layout()` → `alloc_opaque_kv_caches(desc)`. **format descriptor 는 production storage(update_opaque/encode_via_descriptor/D2O merge/grow)를 실제 구동**(매핑으로 확인 — 메타데이터 아님).

### E7 — 게이트 재작성 (v2 의미론)

ADR-0009 의 plan-identity(stage)·descriptor-identity(format) **재증명 정신은 유지**하되, 게이트를 v2 의미론으로 재작성한다(약화 아님 — ABI 교체에 따른 재작성). **CF4 7요소 → V5 1:1 보존 매핑**(감사가능성):

| CF4 요소(v1) | V5 보존(v2) |
|---|---|
| ① builtin-collision reject | per-name `find_kv_format` 검사(봉투 순회) + 별개로 capability-0 reject |
| ② 동적 등록 성공 | `try_register_format` usize≥1 |
| ③ registry merge 가시화 | `dynamic_registered_format_names()` 불변 가시화 |
| ④ descriptor-identity | **multi 확장**(아래 — 서로 다른 desc N종) |
| ⑤ floor byte-identity | per-format round-trip |
| ⑥ 동적 중복 reject | per-name dup bail |
| ⑦ 심볼부재 reject | **capability-0 reject** 로 대체(아래) |

추가/강화 항목:
- **번들 `.so`(stage 1 + format 1)** → 한 번 dlopen 으로 양축 등록(stage·format 둘 다 가시화).
- **multi-format `.so`(패밀리 N≥2종) → N 전부 등록 + 각자 고유 descriptor-identity (G4)**: 예제 crate 를 **서로 다른 descriptor**(예: `bits` 4/8 또는 `block_elems` 16/32/64 상이)로 설계해야 한다. 동일 desc N개는 인덱스-swap·이름↔vtable 미스바인딩(awq4 vtable 이 nf4 이름에 붙음)을 못 잡는 위양성. 단언 = (i) 이름 카운트 ⊇ {nf4,awq4,…}, (ii) `make_format("nf4").layout()==nf4_고유 && make_format("awq4").layout()==awq4_고유 && nf4_desc != awq4_desc`, (iii) per-format floor round-trip byte-identity.
- **capability-0 reject (G1)**: vehicle = **plugin-cdylib ON + `register_kv_format!` 호출하되 `export_plugin!()` 누락** `.so`(슬라이스 기여는 있으나 `register_kv_*s_v2` entry 심볼 부재) → `try_register_*` 양축 dlsym 실패 → `n==0` → dispatcher bail("capability 0 … export_plugin! 누락?"). (feature OFF `.so` 도 v2 심볼 부재라 동일 경로 — 둘 다 단언.) 이게 v1 "심볼부재 reject" 의 등가 대체.
- **wrong-type 은 reject 아님(G2)**: stage-only `.so` 를 `register_dynamic_plugins` 에 → stage 1 + format 0 + **전체 Ok(no bail)** 단언(단일축 정당). 의도된 의미 변화의 명시 게이트.
- **per-`.so` 원자성(G3)**: `[synth_q4(builtin-collision), other(정상)]` 봉투 → bail + **other 미등록(롤백)** 단언(이후 다른 `.so` 의 other 가 dup 위양성 안 나는지).
- **force-link 비대칭**: `synth_q4`(엔진 force-link 정적) 가 bundle(`export_plugin!`)로 빌드돼 dispatcher 에 들어가면 per-name builtin-collision reject 단언.
- **stage 축 multi-vtable (G5)**: multi-stage `.so`(stage 2종) → 각 vtable `plan` fn-ptr 가 올바른 이름에 바인딩됐는지 `make_stage(name).plan(ctx)==알려진 정답`(LayerWide keep + merges + no-op(None) + **PerHead host bail** D5 보존)으로 multi-format 과 대칭 검증.

---

## 3. Consequences / Landmines

- **`export_plugin!()` 명시 1회 필수**: 선언적 매크로 + per-axis 단일 심볼 제약상 "자동 emit" 불가(반복 호출 충돌). 누락 시 dispatcher 가 "capability 0" 으로 fail-fast(명확한 실패). 기여 가이드에 작성자 의무로 명시.
- **3축 분리 심볼 불변(ADR-0009 #118)**: `register_kv_stages_v2` ⊥ `register_kv_formats_v2`(⊥ 미래 `register_backend_caps_v2`)는 **분리 유지**. `export_plugin!()` 가 한 번에 emit 하나 통합 심볼/통합 registry 아님.
- **v2 entry 는 panic-free + cdylib panic=abort 필수(D1 parity 전제)**: `register_kv_*s_v2` entry 는 `.so` static 봉투(슬라이스 base + count)를 순수 by-value 반환할 뿐 **사용자 코드 0** → panic 원천 부재. 사용자 코드 panic 은 `make`/`plan`/`layout`/`drop` thunk 에서만 발생하며 ADR-0009 D1(panic=abort)이 cross-FFI unwind 를 차단(parity 유지). **단 plugin cdylib 이 panic=abort 프로파일로 빌드돼야 parity 성립**(작성자 가이드 경고 — 봉투/thunk 무관하게 abort 빌드가 D1 선결조건). entry 본문에 향후 사용자 코드(동적 vtable 빌드 등)를 넣으면 unwind-across-FFI UB 회귀 — 금지.
- **빈 distributed_slice 안전(ELF/Linux·Android 한정)**: 기여 0 축의 봉투 `count==0`, `vtables = __start(== __stop)` → `from_raw_parts(ptr, 0)` 빈 슬라이스, 0회 순회. linkme `linker.rs::linux` 가 빈 섹션도 경계 심볼 생성 → 정상(실측). (Windows/UEFI 는 boundary element non-ZST + dupcheck 분기라 다르나 본 프로젝트 무관.)
- **allocator 경계 불변(ADR-0009)**: plugin 인스턴스(`make` 산출 `*mut c_void`)는 plugin `drop` fn-ptr 로만 해제. 봉투/`PLUGIN_*_VTABLES` 슬라이스는 `.so` static(해제 안 함). vtable 포인터 dangling 방지 = registry 의 `Arc<Library>`.
- **번들 `.so` 의 Arc 공유**: 한 `.so` 의 여러 등록(stage+format+다수 format)이 동일 `Arc<Library>` 공유 → 마지막 등록/인스턴스 drop 시 unload(프로세스 수명). A2(핸들 분리)와 달리 정직.
- **`export_plugin!()` 네이밍**: 현 per-item 매크로는 `register_kv_*`(kv 접두). backend capability 는 "kv" 아님 → 통합 export 매크로는 축-무관 `export_plugin!()` 채택. (대안 `export_kv_plugin!` 은 v3 합류 시 오칭 — 회피.)
- **multi-per-axis 정적 경로**: dlopen 단일심볼 경로만 막혀 있었으나, 정적 경로도 **한 crate 내 같은 매크로 2회 호출은 현재 불가**(top-level `__REGISTER_*_REG` 심볼 충돌 — `lib.rs:627` docstring "crate 당 1회"). "이미 다수 가능" 은 **서로 다른 모듈/crate 가 각자 1회씩** 호출하는 경우에 한함. 한 crate(=한 `.so`) 내 다회(nf4+awq4)는 본 ADR 의 매크로 재작성(정적·동적 기여 static 을 const-block 격리)으로 **비로소 가능**. 본 ADR 후 양 경로(정적 linkme ⊥ 동적 PLUGIN) 대칭으로 다수.
- **ABI_VERSION bump 의 의미**: v1(단일) `.so` 는 v2 로더에서 `register_kv_*s_v2` 심볼 부재 → capability-0 reject. 미푸시라 기존 `.so` 자산 없음(synth_q4/example 재빌드). 외부 배포 시작 전이므로 안전.
- **명명 비대칭 정리(C4)**: stage 진단 함수 `dynamic_registered_names()`(`stage_registry.rs:487`, "format" 없는 일반명) vs format `dynamic_registered_format_names()`(`dynamic_format_registry.rs:112`). 정적 짝도 `registered_names()` vs `registered_kv_format_names()` 비대칭. V2 cross-axis dispatcher 진단 표면 혼동 방지 위해 stage 를 `dynamic_registered_stage_names()` 로 rename(내부 진단/self-test 전용 — breaking 아님).
- **`ensure_builtin_kv_formats_registered` 는 현재 unwired(C3)**: ADR-0003 §4 fat-LTO self-test 코드는 존재하나 **production 호출부 0**(`builtin_kv_formats.rs:103`, 테스트만). `register_kv_format!` 정적 기여(`KV_FORMATS`)는 V1 재작성에서 보존되므로 `find_kv_format` 전제는 불변이나, self-test 가 **실제 게이트로 작동하려면 V5 가 production startup 에 배선**(또는 release smoke 에서 실제 호출 검증)해야 한다 — "유지"가 "작동 중"을 뜻하지 않음.
- **fat-LTO 생존**: 정적 `KV_FORMATS` 기여 보존(self-test 전제). 동적 `PLUGIN_*_VTABLES` 슬라이스의 cdylib·fat-LTO 생존은 실측(probe ①②) + V5 게이트가 담당.
- **v3 Backend "한 줄" 은 선행 전부 후의 마지막 라인(C2)**: `export_plugin!()` 에 `PLUGIN_BACKEND_VTABLES` 한 줄을 더하는 게 "한 줄" 이 되려면 ① `BackendCapability` trait 메서드 확정(현재 `name()` 1개 골격, `lib.rs:1088`) ② `BackendVTableAbi` `#[repr(C)]` ③ `register_backend_capability!` dual-wiring 매크로 ④ host `DynBackendCap` + `DYN_BACKEND_REGISTRY` 가 **모두 선행**돼야 한다(ADR-0009 D4: capability trait 메서드 확정 후). V1–V2 인프라(봉투/슬라이스/dispatcher/`export_plugin!` 프레임)는 재사용되나 backend 의 vtable 표면은 전무 — "한 줄" 은 그 선행 후의 합류 라인.

---

## 4. Phasing / Gates

| 단계 | 산출 | 게이트 |
|---|---|---|
| **V1** technique-api | 봉투(`FormatExportAbi`/`StageExportAbi`) + `PLUGIN_*_VTABLES` 슬라이스(technique-api 1곳 선언) + vtable 에서 `abi_version` 제거(`unsafe impl Sync` 유지) + `register_kv_*!` 재작성(no_mangle entry 제거 → 정적·동적 기여 static 을 **const-block 격리**, thunk 본문 불변) + `export_plugin!()`(plugin-cdylib 게이트) 신설. `ABI_VERSION 1→2`(양축) | technique-api cargo test + 봉투 **by-value sret round-trip**(count/ptr 보존) + **한 crate 2회 `register_kv_format!` → 정적·동적 양쪽 2건 등록** + thunk 시그니처=`unsafe extern C`·`$make` 직접 캐스팅 0 + 정적 등록 무회귀 |
| **V2** engine 로더 | `try_register_*`(count 반환, **2-pass 원자 + per-name 충돌검사** + 봉투 `vtables.add(i)`) + `register_dynamic_plugins`(open-once dispatcher) + batch 로더를 strict 래퍼로 재배치 + `dynamic_registered_names`→`dynamic_registered_stage_names` rename | engine cargo test + clippy clean |
| **V3** 배선 | bin_setup W1(`register_dynamic_plugins`) + W2(`make_format` 호출부 교체 + 동적 arm 봉투-aware) + (C3) `ensure_builtin_kv_formats_registered` startup 배선 | 빌드 + 정적 stage/format e2e 무회귀 |
| **V4** vehicle | synth_q4/example crate 에 `export_plugin!()` 적용 + **multi-format 예제 crate(서로 다른 descriptor ≥2종)** 신설 + **no-export vehicle**(`register_kv_format!` 있고 `export_plugin!` 없음 — capability-0 유발) | 각 cdylib `nm register_kv_formats_v2`/`register_kv_stages_v2`(no-export 는 부재) 확인 |
| **V5** 게이트 | 번들 `.so` 양축 등록 + multi-format N등록 **서로 다른 descriptor-identity** + capability-0 reject(no-export & feature-OFF) + wrong-type=graceful 흡수 + per-`.so` 원자성 롤백 + builtin-collision(synth_q4 bundle)/동적중복 + plan-identity(stage, multi-stage 인덱스 바인딩 + PerHead bail) + descriptor-floor byte-identity(per-format) + fat-LTO release smoke | 신규/재작성 테스트 통과 + lib 무회귀(≥1260/0) + release smoke green |

**완료 = V5 게이트 통과 + 전체 sanity green + 정적 경로 무회귀.** v3 Backend 축은 V1–V2 인프라(봉투/슬라이스/dispatcher/`export_plugin!` 프레임)를 재사용하되, `BackendCapability` 메서드 확정 + `BackendVTableAbi` + `register_backend_capability!` dual-wiring + `DynBackendCap` 어댑터가 **선행**돼야 `export_plugin!` 의 `register_backend_caps_v2` 한 줄로 합류(§3 C2).
