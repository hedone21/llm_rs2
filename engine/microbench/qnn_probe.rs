//! microbench_qnn_probe — LISWAP-5 Phase A: vendor QNN runtime feasibility
//!
//! 목적: stock Galaxy S25에서 user-space app이 vendor QNN runtime을 dlopen +
//! getProviders 호출 가능한지 확인. selinux Enforcing 환경에서 shell uid 2000이
//! /vendor/lib64/snap/libQnn*.so를 사용 가능한가의 fail-fast 검증.
//!
//! 측정/출력:
//! - libQnnSystem.so dlopen + QnnSystemInterface_getProviders 호출 결과
//! - libQnnHtp.so dlopen + QnnInterface_getProviders 호출 결과
//! - num_providers 출력 (HTP backend 검출 여부)
//! - selinux 차단 시 dlopen 단계에서 fail
//!
//! Build: `cargo build --release --features qnn --target aarch64-linux-android --bin microbench_qnn_probe`
//! Run:   `LD_LIBRARY_PATH=/vendor/lib64/snap:/vendor/lib64 adb shell /data/local/tmp/microbench_qnn_probe`

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_qnn_probe requires --features qnn");
    std::process::exit(2);
}

#[cfg(feature = "qnn")]
fn main() -> anyhow::Result<()> {
    use libloading::{Library, Symbol};
    use std::os::raw::{c_uint, c_void};

    // QNN public API:
    //   QnnSystemInterface_getProviders(
    //       const QnnSystemInterface_t*** providers,
    //       uint32_t* num_providers
    //   ) -> Qnn_ErrorHandle_t (uint64_t, 0 == QNN_SUCCESS)
    //
    //   QnnInterface_getProviders(
    //       const QnnInterface_t*** providers,
    //       uint32_t* num_providers
    //   ) -> Qnn_ErrorHandle_t
    //
    // Provider struct는 SDK 헤더에서만 정의되지만 num_providers > 0이면
    // backend가 메모리에 존재함을 의미. dlopen + getProviders 호출 자체가
    // selinux 차단 여부 판단에 충분.
    type ProvidersFn = unsafe extern "C" fn(*mut *const *const c_void, *mut c_uint) -> u64;

    println!("=== microbench_qnn_probe (LISWAP-5 Phase A) ===\n");

    // ── libQnnSystem.so ──
    println!("[1/2] libQnnSystem.so");
    let sys_paths = [
        "/vendor/lib64/snap/libQnnSystem.so",
        "/data/local/tmp/libQnnSystem.so",
        "libQnnSystem.so",
    ];
    let mut sys_lib: Option<Library> = None;
    for p in sys_paths {
        match unsafe { Library::new(p) } {
            Ok(lib) => {
                println!("  dlopen OK: {}", p);
                sys_lib = Some(lib);
                break;
            }
            Err(e) => println!("  dlopen FAIL: {} ({})", p, e),
        }
    }
    let sys_lib =
        sys_lib.ok_or_else(|| anyhow::anyhow!("libQnnSystem.so not loadable from any path"))?;
    let sys_fn: Symbol<ProvidersFn> = unsafe {
        sys_lib
            .get(b"QnnSystemInterface_getProviders\0")
            .map_err(|e| anyhow::anyhow!("dlsym QnnSystemInterface_getProviders: {}", e))?
    };
    println!("  dlsym QnnSystemInterface_getProviders: OK");

    let mut sys_providers: *const *const c_void = std::ptr::null();
    let mut sys_num: c_uint = 0;
    let sys_err = unsafe { sys_fn(&mut sys_providers, &mut sys_num) };
    println!(
        "  call: err=0x{:x}, num_providers={}, providers_ptr={:p}",
        sys_err, sys_num, sys_providers
    );
    if sys_err != 0 {
        anyhow::bail!("QnnSystemInterface_getProviders returned non-zero error");
    }

    // ── libQnnHtp.so ──
    println!("\n[2/2] libQnnHtp.so (Hexagon V79 HTP backend)");
    let htp_paths = [
        "/vendor/lib64/snap/libQnnHtp.so",
        "/data/local/tmp/libQnnHtp.so",
        "libQnnHtp.so",
    ];
    let mut htp_lib: Option<Library> = None;
    for p in htp_paths {
        match unsafe { Library::new(p) } {
            Ok(lib) => {
                println!("  dlopen OK: {}", p);
                htp_lib = Some(lib);
                break;
            }
            Err(e) => println!("  dlopen FAIL: {} ({})", p, e),
        }
    }
    let htp_lib =
        htp_lib.ok_or_else(|| anyhow::anyhow!("libQnnHtp.so not loadable from any path"))?;
    let htp_fn: Symbol<ProvidersFn> = unsafe {
        htp_lib
            .get(b"QnnInterface_getProviders\0")
            .map_err(|e| anyhow::anyhow!("dlsym QnnInterface_getProviders: {}", e))?
    };
    println!("  dlsym QnnInterface_getProviders: OK");

    let mut htp_providers: *const *const c_void = std::ptr::null();
    let mut htp_num: c_uint = 0;
    let htp_err = unsafe { htp_fn(&mut htp_providers, &mut htp_num) };
    println!(
        "  call: err=0x{:x}, num_providers={}, providers_ptr={:p}",
        htp_err, htp_num, htp_providers
    );
    if htp_err != 0 {
        anyhow::bail!("QnnInterface_getProviders (HTP) returned non-zero error");
    }

    // ── Phase A verdict ──
    println!("\n=== Phase A summary ===");
    if sys_num > 0 && htp_num > 0 {
        println!("✓ PASS: dlopen + getProviders 호출 모두 성공");
        println!("  - System providers: {}", sys_num);
        println!("  - HTP providers:    {}", htp_num);
        println!("  - selinux가 vendor QNN runtime 사용을 차단하지 않음");
        println!("\nNext: Phase B (Q1 correctness) — backend create + simple Add op execute");
        Ok(())
    } else if sys_num > 0 || htp_num > 0 {
        println!("⚠ PARTIAL: 일부 provider 검출 안 됨");
        println!("  - dlopen은 성공했지만 backend discovery에서 차단 가능성");
        println!("  - logcat에서 selinux denial 확인 필요");
        Ok(())
    } else {
        anyhow::bail!("dlopen 성공했지만 num_providers=0 — backend invisible")
    }
}
