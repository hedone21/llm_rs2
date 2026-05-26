//! microbench_htp_fastrpc_dryrun — Q-2.2 dry-run (B scope)
//!
//! 목적: stock Galaxy S25 (non-rooted) 에서 libcdsprpc.so FastRPC layer 가 살아있는지
//! 직접 증명. Q-2.1 dry-run 의 `adsprpc remote.c:40 Control stubbed routine` 이
//! QNN backend 의 vendor-extension control 만 차단된 것인지, FastRPC layer 전체가
//! 차단된 것인지 가설 분리.
//!
//! Stop point: DSPRPC_CONTROL_UNSIGNED_MODULE 호출까지. handle_open / dspqueue 는
//! 자체 빌드 skel (libggml-htp-vNN.so) 이 필요해 dry-run scope 초과.
//!
//! PASS criteria (가설 정밀화 = QNN domain_init 만 차단):
//!   - dlopen libcdsprpc.so OK
//!   - dlsym remote_session_control OK
//!   - FASTRPC_RESERVE_NEW_SESSION return 0 + effective_domain_id valid (>0 또는 known CDSP id)
//!   - DSPRPC_CONTROL_UNSIGNED_MODULE return 0
//!
//! FAIL → 가설 (5) 확장 (FastRPC 전체 OS-level ACL).
//!
//! 매크로/struct 정의 출처: github.com/qualcomm/fastrpc/inc/remote.h (open source).
//!
//! Build: cargo build --release --features qnn --target aarch64-linux-android \
//!        --bin microbench_htp_fastrpc_dryrun
//! Run:   adb shell /data/local/tmp/microbench_htp_fastrpc_dryrun

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_htp_fastrpc_dryrun requires --features qnn");
    std::process::exit(2);
}

#[cfg(feature = "qnn")]
#[allow(
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code
)]
mod fastrpc {
    // session_control_req_id (remote.h enum, 0-indexed)
    pub const FASTRPC_RESERVE_NEW_SESSION: u32 = 13;
    pub const FASTRPC_GET_URI: u32 = 15;
    pub const DSPRPC_CONTROL_UNSIGNED_MODULE: u32 = 2;

    // handle_control_req_id (별도 enum)
    pub const DSPRPC_CONTROL_LATENCY: u32 = 1;

    // domain.h
    pub const CDSP_DOMAIN_NAME: &str = "cdsp";

    // remote.h struct remote_rpc_reserve_new_session
    // 주의: char* + uint32_t 교차. ARM64 에서 padding 발생 가능 → repr(C) + 필드 순서 정확히.
    #[repr(C)]
    #[derive(Debug)]
    pub struct RemoteRpcReserveNewSession {
        pub domain_name: *mut u8, // char*
        pub domain_name_len: u32,
        pub session_name: *mut u8,
        pub session_name_len: u32,
        pub effective_domain_id: u32, // [out]
        pub session_id: u32,          // [out]
    }

    // struct remote_rpc_control_unsigned_module
    #[repr(C)]
    pub struct RemoteRpcControlUnsignedModule {
        pub domain: i32,
        pub enable: i32,
    }
}

#[cfg(feature = "qnn")]
fn main() -> anyhow::Result<()> {
    use libloading::Symbol;
    use std::os::raw::{c_int, c_void};

    use fastrpc::*;

    println!("=== microbench_htp_fastrpc_dryrun (Q-2.2 dry-run B) ===\n");

    // ── Step 1: dlopen libcdsprpc.so ──
    // stock Android 경로: /vendor/lib64/libcdsprpc.so. fallback 으로 push path 도 시도.
    let candidates = [
        "/vendor/lib64/libcdsprpc.so",
        "/data/local/tmp/libcdsprpc.so",
        "libcdsprpc.so",
    ];
    let mut loaded: Option<(libloading::Library, &'static str)> = None;
    for &path in &candidates {
        match unsafe { libloading::Library::new(path) } {
            Ok(lib) => {
                println!("[1/4] dlopen OK: {}", path);
                let p: &'static str = Box::leak(path.to_string().into_boxed_str());
                loaded = Some((lib, p));
                break;
            }
            Err(e) => {
                println!("[1/4] dlopen skip: {} ({})", path, e);
            }
        }
    }
    let (lib, lib_path) =
        loaded.ok_or_else(|| anyhow::anyhow!("libcdsprpc.so not found in any path"))?;
    println!("[1/4] using: {}\n", lib_path);

    // ── Step 2: dlsym remote_session_control ──
    type RemoteSessionControlFn =
        unsafe extern "C" fn(req: u32, data: *mut c_void, datalen: u32) -> c_int;
    let remote_session_control: Symbol<RemoteSessionControlFn> = unsafe {
        lib.get(b"remote_session_control\0")
            .map_err(|e| anyhow::anyhow!("dlsym remote_session_control fail: {}", e))?
    };
    println!("[2/4] dlsym remote_session_control OK\n");

    // ── Step 3: FASTRPC_RESERVE_NEW_SESSION ──
    // 가설: 이게 PASS → FastRPC channel 자체는 stock S25 에서 open 가능.
    //       FAIL → 가설 (5) 를 FastRPC 전체로 확장.
    let mut domain_name = CDSP_DOMAIN_NAME.to_string().into_bytes();
    domain_name.push(0); // null-terminate
    let mut session_name = b"llmrs_dryrun\0".to_vec();

    let mut req = RemoteRpcReserveNewSession {
        domain_name: domain_name.as_mut_ptr(),
        domain_name_len: (domain_name.len() - 1) as u32, // exclude null
        session_name: session_name.as_mut_ptr(),
        session_name_len: (session_name.len() - 1) as u32,
        effective_domain_id: 0,
        session_id: 0,
    };
    let req_size = std::mem::size_of::<RemoteRpcReserveNewSession>() as u32;
    println!(
        "[3/4] FASTRPC_RESERVE_NEW_SESSION (req_id={}, struct size={})",
        FASTRPC_RESERVE_NEW_SESSION, req_size
    );
    println!(
        "      domain=\"{}\" (len={}), session=\"llmrs_dryrun\" (len={})",
        CDSP_DOMAIN_NAME, req.domain_name_len, req.session_name_len
    );

    let err = unsafe {
        remote_session_control(
            FASTRPC_RESERVE_NEW_SESSION,
            &mut req as *mut _ as *mut c_void,
            req_size,
        )
    };
    if err != 0 {
        println!("[3/4] ★ FAIL — return {} (0x{:x})", err, err as u32);
        println!("      ⇒ 가설 (5) FastRPC 전체 ACL 차단 가능성");
        println!("      ⇒ logcat 에서 adsprpc tag 확인 필요");
        return Err(anyhow::anyhow!("FASTRPC_RESERVE_NEW_SESSION fail: {}", err));
    }
    println!(
        "[3/4] ✓ PASS — effective_domain_id={}, session_id={}",
        req.effective_domain_id, req.session_id
    );
    println!("      ⇒ FastRPC channel open 성공 (Q-2.1 QNN domain_init 차단과 대비)\n");

    // ── Step 4: DSPRPC_CONTROL_UNSIGNED_MODULE ──
    // stock S25 의 가장 중요한 게이트. 이게 PASS 면 unsigned PD 활성화 가능 = 자체 빌드 skel 로드 가능.
    let mut unsigned_req = RemoteRpcControlUnsignedModule {
        domain: req.effective_domain_id as i32,
        enable: 1,
    };
    let u_size = std::mem::size_of::<RemoteRpcControlUnsignedModule>() as u32;
    println!(
        "[4/4] DSPRPC_CONTROL_UNSIGNED_MODULE (req_id={}, struct size={})",
        DSPRPC_CONTROL_UNSIGNED_MODULE, u_size
    );
    println!("      domain={}, enable=1", unsigned_req.domain);

    let err = unsafe {
        remote_session_control(
            DSPRPC_CONTROL_UNSIGNED_MODULE,
            &mut unsigned_req as *mut _ as *mut c_void,
            u_size,
        )
    };
    if err != 0 {
        println!("[4/4] ★ FAIL — return {} (0x{:x})", err, err as u32);
        println!("      ⇒ unsigned PD 차단 (stock device permission denied)");
        println!("      ⇒ 자체 빌드 skel 로드 불가 → llama.cpp path 도 동작 불가 가능성");
        return Err(anyhow::anyhow!(
            "DSPRPC_CONTROL_UNSIGNED_MODULE fail: {}",
            err
        ));
    }
    println!("[4/4] ✓ PASS — unsigned PD 활성화 성공\n");

    // ── Summary ──
    println!("=== ALL PASS — Q-2.2 진입 GREEN ===");
    println!("  • FastRPC channel: open OK");
    println!("  • Unsigned PD: enabled OK");
    println!("  • 가설 정밀화: Q-2.1 차단은 QNN backend 의 domain_init vendor control 에 한정.");
    println!("                  FastRPC layer 자체는 stock S25 에서 살아있음.");
    println!("  • 다음: Q-2.2-α Architect spec (FastRPC IDL backend 신설)");

    Ok(())
}
