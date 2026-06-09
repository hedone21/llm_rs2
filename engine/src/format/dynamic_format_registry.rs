//! GATE-C v2 — 런타임 `.so` dlopen format 레지스트리 (ADR-0009 D4 / CF2).
//!
//! 설계 SSOT: `docs/adr/0009-gate-c-stage-dlopen-plugin.md`(D4 Format 축). 거울 =
//! `pressure/eviction/stage_registry.rs`(GATE-C v1 stage 축: `DYN_REGISTRY`/`register_dynamic_stages`/
//! `make_stage`/`DynStage`).
//!
//! 정적 `KV_FORMATS`(linkme, `builtin_kv_formats.rs` + 외부 crate)는 그대로 두고(D3 가산), dlopen 된
//! format plugin 을 별도 [`struct@DYN_FORMAT_REGISTRY`] 에 모은다. [`make_format`] 이 정적 우선 →
//! 동적 fallback 으로 source-agnostic `Box<dyn KVFormat>` 를 돌려준다. Stage 와 달리 `KVFormat` 은
//! 콜백 0(순수 descriptor)이라 host 측 어댑터([`DynFormat`])는 `layout` fn-ptr 호출 + `drop` 뿐이다.
//!
//! **purely additive·unwired (CF2)** — 등록/조회 표면만 신설하고 production 소비자(`--load-plugin`→
//! format 배선, `--kv-format` 동적 해석)는 후속(ADR-0009 v2 e2e). dlopen 메커니즘은
//! `engine/tests/gate_c_format_dlopen_equivalence.rs`(CF4) 격리 테스트가 descriptor-identity 로 증명.

use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};

use anyhow::{Context, Result};
use core::ffi::c_void;
use technique_api::{FormatVTableAbi, KV_FORMAT_ABI_VERSION, KVFormat, KVLayoutDesc};

/// dlopen 된 한 format plugin 의 등록 항목. vtable 은 plugin `.so` 의 immutable static 을 가리킨다.
struct RuntimeFormatReg {
    name: String,
    vtable: *const FormatVTableAbi,
    /// `.so` 를 프로세스 수명 동안 유지(vtable/handle dangling 방지). drop 안 함.
    _lib: Arc<libloading::Library>,
}

// SAFETY: vtable 은 `.so` 의 immutable static 을 가리키고 `_lib`(Arc) 가 `.so` 를 살려 둔다.
// 읽기 전용 공유이므로 스레드 간 안전 — `DYN_FORMAT_REGISTRY`(static) 에 담기 위해 필요.
unsafe impl Send for RuntimeFormatReg {}
unsafe impl Sync for RuntimeFormatReg {}

/// 동적 format 등록 레지스트리 — init 시 append, construction 시 read. 정적 슬라이스와 **병합하지 않는다**(D3).
static DYN_FORMAT_REGISTRY: OnceLock<RwLock<Vec<RuntimeFormatReg>>> = OnceLock::new();

/// `--load-plugin` 으로 지정된 `.so` 들을 dlopen 해 [`struct@DYN_FORMAT_REGISTRY`] 에 등록한다(D4).
///
/// 각 `.so`: `Library::new`(RTLD_NOW) → `register_kv_format_v1` dlsym → `abi_version` 검사 →
/// 이름 충돌 fail-fast(빌트인 우선 + 동적 중복 금지). [`register_dynamic_stages`](crate::pressure::eviction::stage_registry::register_dynamic_stages)
/// 의 format 축 짝(동형 strict 로더 — 심볼 부재 시 거부).
pub fn register_dynamic_formats(paths: &[PathBuf]) -> Result<()> {
    if paths.is_empty() {
        return Ok(());
    }
    let registry = DYN_FORMAT_REGISTRY.get_or_init(|| RwLock::new(Vec::new()));
    for path in paths {
        // SAFETY: dlopen — 신뢰된 plugin 경로(사용자 명시 --load-plugin). RTLD_NOW 즉시 바인딩.
        let lib = unsafe { libloading::Library::new(path) }
            .with_context(|| format!("plugin dlopen 실패: {}", path.display()))?;
        // SAFETY: register_kv_format_v1 심볼 dlsym + 호출. vtable_ptr 는 `.so` 의 static 을 가리킨다.
        let vtable_ptr: *const FormatVTableAbi = unsafe {
            let reg_fn: libloading::Symbol<unsafe extern "C" fn() -> *const FormatVTableAbi> = lib
                .get(b"register_kv_format_v1\0")
                .with_context(|| {
                    format!("plugin {}: register_kv_format_v1 심볼 부재", path.display())
                })?;
            reg_fn()
        };
        if vtable_ptr.is_null() {
            anyhow::bail!(
                "plugin {}: register_kv_format_v1 가 null vtable 반환",
                path.display()
            );
        }
        // SAFETY: 위에서 non-null 확인. vtable 은 `.so` static.
        let vtable = unsafe { &*vtable_ptr };
        if vtable.abi_version != KV_FORMAT_ABI_VERSION {
            anyhow::bail!(
                "plugin {}: abi_version {} != 기대 {} (재빌드 필요)",
                path.display(),
                vtable.abi_version,
                KV_FORMAT_ABI_VERSION
            );
        }
        // SAFETY: name 은 plugin 의 null-종단 'static str.
        let name = unsafe { CStr::from_ptr(vtable.name) }
            .to_str()
            .with_context(|| format!("plugin {}: name 이 유효 UTF-8 아님", path.display()))?
            .to_owned();
        // 충돌 fail-fast (빌트인 우선 — silent override 차단). 빌트인엔 외부 force-link 정적 format
        // (예: synth_q4)도 포함 → 같은 이름 `.so` dlopen 은 거부된다.
        if technique_api::find_kv_format(&name).is_some() {
            anyhow::bail!(
                "plugin {}: format 이름 '{}' 이 빌트인과 충돌 (빌트인 우선, 동적 등록 거부)",
                path.display(),
                name
            );
        }
        let lib = Arc::new(lib);
        let mut w = registry.write().expect("DYN_FORMAT_REGISTRY RwLock poisoned");
        if w.iter().any(|r| r.name == name) {
            anyhow::bail!(
                "plugin {}: format 이름 '{}' 이 이미 동적 등록됨 (중복)",
                path.display(),
                name
            );
        }
        w.push(RuntimeFormatReg {
            name,
            vtable: vtable_ptr,
            _lib: lib,
        });
    }
    Ok(())
}

/// 동적으로 등록된 format 이름들(self-test / 진단용 — 정적 `registered_kv_format_names()` 의 동적 짝).
pub fn dynamic_registered_format_names() -> Vec<String> {
    DYN_FORMAT_REGISTRY
        .get()
        .map(|r| {
            r.read()
                .expect("DYN_FORMAT_REGISTRY RwLock poisoned")
                .iter()
                .map(|reg| reg.name.clone())
                .collect()
        })
        .unwrap_or_default()
}

/// 이름으로 format 인스턴스를 만든다 — **정적 우선 → 동적 fallback**(D3). 호출부는 source 를 모른다.
/// 정적/동적 모두 miss 면 `None`(graceful unknown). `find_kv_format` 의 source-agnostic 짝.
pub fn make_format(name: &str) -> Option<Box<dyn KVFormat>> {
    // 1) 정적(linkme) 우선.
    if let Some(reg) = technique_api::find_kv_format(name) {
        return Some((reg.make)());
    }
    // 2) 동적(dlopen) fallback.
    let registry = DYN_FORMAT_REGISTRY.get()?;
    let (vtable, lib) = {
        let guard = registry.read().expect("DYN_FORMAT_REGISTRY RwLock poisoned");
        let reg = guard.iter().find(|r| r.name == name)?;
        (reg.vtable, Arc::clone(&reg._lib))
    };
    // SAFETY: vtable 는 `.so` static (lib 가 살려 둠). make 가 opaque plugin 핸들 반환.
    let handle = unsafe { ((*vtable).make)() };
    if handle.is_null() {
        eprintln!("[make_format] plugin '{name}' make 가 null 핸들 반환");
        return None;
    }
    Some(Box::new(DynFormat {
        handle,
        vtable,
        _lib: lib,
    }))
}

/// 동적 plugin format 의 host 측 어댑터 — vtable 마샬링으로 [`KVFormat`] 을 구현(D4). Stage 의
/// `DynStage` 거울이나 콜백 0이라 `layout`(POD 반환) + `drop` 뿐.
struct DynFormat {
    handle: *mut c_void,
    vtable: *const FormatVTableAbi,
    _lib: Arc<libloading::Library>,
}

// SAFETY: 핸들은 plugin 의 `KVFormat`(trait 계약상 Send+Sync) 인스턴스, vtable 불변, lib Arc 유지.
unsafe impl Send for DynFormat {}
unsafe impl Sync for DynFormat {}

impl Drop for DynFormat {
    fn drop(&mut self) {
        // SAFETY: handle 은 make 가 만든 plugin 인스턴스, 정확히 1회 해제.
        unsafe { ((*self.vtable).drop)(self.handle) };
    }
}

impl KVFormat for DynFormat {
    fn name(&self) -> &str {
        // SAFETY: vtable.name 은 plugin `.so` 의 'static null-종단 str (lib 가 살려 둠).
        unsafe { CStr::from_ptr((*self.vtable).name) }
            .to_str()
            .unwrap_or("<plugin>")
    }

    fn layout(&self) -> KVLayoutDesc {
        // SAFETY: handle = make 가 만든 인스턴스. layout 은 KVLayoutDesc POD 를 값으로 반환(arena 불요).
        unsafe { ((*self.vtable).layout)(self.handle) }
    }
}
