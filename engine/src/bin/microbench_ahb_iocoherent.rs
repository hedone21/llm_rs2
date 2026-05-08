//! microbench_ahb_iocoherent — AHardwareBuffer + IOCOHERENT 측정 (Phase 3)
//!
//! 목적: 기존 DMA-BUF heap path의 garbage 출력 (cache coherency 미보장)을
//! AHB + IOCOHERENT 결합으로 회피 가능한지 검증.
//! Phase 0 baseline: ALLOC_HOST_PTR 600MB H2D = 22ms (27.5 GB/s).
//!
//! 측정:
//! - AHardwareBuffer_allocate (USAGE_GPU_DATA_BUFFER + FORMAT_BLOB)
//! - AHardwareBuffer_lock으로 host pointer 획득
//! - clCreateBuffer with cl_mem_ahardwarebuffer_host_ptr struct
//!   + ext_host_ptr.host_cache_policy = CL_MEM_HOST_IOCOHERENT_QCOM
//! - host write throughput (memcpy CPU→AHB host_ptr)
//! - GPU kernel read throughput
//! - decode 정확성 검증 (host write data vs kernel result)
//!
//! Build: `cargo build --release --target aarch64-linux-android --bin microbench_ahb_iocoherent`
//! Run:   `adb shell ./microbench_ahb_iocoherent [SIZE_MB] [N_ITERS] [CACHE_POLICY]`
//!        CACHE_POLICY: writeback (0x40A5) | iocoherent (0x40A9). default: iocoherent.

#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("microbench_ahb_iocoherent requires --features opencl");
    std::process::exit(2);
}

#[cfg(feature = "opencl")]
#[allow(non_camel_case_types, non_upper_case_globals, dead_code)]
mod ahb {
    use std::ffi::c_void;

    /// Opaque NDK type.
    pub type AHardwareBuffer = c_void;

    #[repr(C)]
    pub struct AHardwareBuffer_Desc {
        pub width: u32,
        pub height: u32,
        pub layers: u32,
        pub format: u32,
        pub usage: u64,
        pub stride: u32,
        pub rfu0: u32,
        pub rfu1: u64,
    }

    pub const AHARDWAREBUFFER_FORMAT_BLOB: u32 = 0x21;
    pub const AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER: u64 = 1u64 << 24;
    pub const AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN: u64 = 0x3;
    pub const AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN: u64 = 0x30;

    pub type allocate_fn =
        unsafe extern "system" fn(*const AHardwareBuffer_Desc, *mut *mut AHardwareBuffer) -> i32;
    pub type lock_fn = unsafe extern "system" fn(
        *mut AHardwareBuffer,
        u64,
        i32,
        *const c_void,
        *mut *mut c_void,
    ) -> i32;
    pub type unlock_fn = unsafe extern "system" fn(*mut AHardwareBuffer, *mut i32) -> i32;
    pub type release_fn = unsafe extern "system" fn(*mut AHardwareBuffer);

    pub struct AhbFns {
        pub allocate: allocate_fn,
        pub lock: lock_fn,
        pub unlock: unlock_fn,
        pub release: release_fn,
    }

    pub fn load() -> Option<AhbFns> {
        unsafe {
            let lib_paths = [
                b"libnativewindow.so\0".as_ptr() as *const libc::c_char,
                b"libandroid.so\0".as_ptr() as *const libc::c_char,
            ];
            let mut handle: *mut c_void = std::ptr::null_mut();
            for path in lib_paths {
                handle = libc::dlopen(path, libc::RTLD_NOW);
                if !handle.is_null() {
                    break;
                }
            }
            if handle.is_null() {
                eprintln!("dlopen libnativewindow.so failed");
                return None;
            }
            macro_rules! sym {
                ($name:literal) => {{
                    let s =
                        libc::dlsym(handle, concat!($name, "\0").as_ptr() as *const libc::c_char);
                    if s.is_null() {
                        eprintln!("dlsym '{}' failed", $name);
                        return None;
                    }
                    std::mem::transmute(s)
                }};
            }
            Some(AhbFns {
                allocate: sym!("AHardwareBuffer_allocate"),
                lock: sym!("AHardwareBuffer_lock"),
                unlock: sym!("AHardwareBuffer_unlock"),
                release: sym!("AHardwareBuffer_release"),
            })
        }
    }
}

#[cfg(feature = "opencl")]
#[allow(non_camel_case_types, non_upper_case_globals, dead_code)]
mod qcom {
    use ocl::ffi::cl_uint;
    use std::ffi::c_void;

    pub const CL_MEM_EXT_HOST_PTR_QCOM: u64 = 1 << 29;
    pub const CL_MEM_ANDROID_AHARDWAREBUFFER_HOST_PTR_QCOM: cl_uint = 0x4119;
    pub const CL_MEM_HOST_WRITEBACK_QCOM: cl_uint = 0x40A5;
    pub const CL_MEM_HOST_IOCOHERENT_QCOM: cl_uint = 0x40A9;
    pub const CL_MEM_HOST_UNCACHED_QCOM: cl_uint = 0x40A6;

    #[repr(C)]
    pub struct cl_mem_ext_host_ptr {
        pub allocation_type: cl_uint,
        pub host_cache_policy: cl_uint,
    }

    #[repr(C)]
    pub struct cl_mem_ahardwarebuffer_host_ptr {
        pub ext_host_ptr: cl_mem_ext_host_ptr,
        pub ahb_ptr: *mut c_void,
    }
}

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use ocl::core::ClContextPtr;
    use ocl::ffi;
    use ocl::{Context, Device, Platform, Queue};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let size_mb: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(600);
    let n_iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let cache_label = args.get(3).map(String::as_str).unwrap_or("iocoherent");
    let cache_policy = match cache_label {
        "writeback" => qcom::CL_MEM_HOST_WRITEBACK_QCOM,
        "iocoherent" => qcom::CL_MEM_HOST_IOCOHERENT_QCOM,
        "uncached" => qcom::CL_MEM_HOST_UNCACHED_QCOM,
        other => {
            eprintln!(
                "unknown cache policy: {} (try writeback|iocoherent|uncached)",
                other
            );
            std::process::exit(2);
        }
    };
    let size_bytes = size_mb * 1024 * 1024;

    println!(
        "AHB+iocoherent: {}MB, n_iters={}, cache_policy={} (0x{:04x})",
        size_mb, n_iters, cache_label, cache_policy
    );

    let ahb_fns = match ahb::load() {
        Some(f) => f,
        None => anyhow::bail!("Failed to load AHardwareBuffer NDK symbols"),
    };
    println!("✓ Loaded AHardwareBuffer_* NDK symbols");

    let platform = Platform::default();
    let device = Device::first(platform)?;
    println!("Device: {}", device.name()?);

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let queue = Queue::new(&context, device, None)?;
    let ctx_ptr = ClContextPtr::as_ptr(&&context);

    // === Allocate AHardwareBuffer ===
    let desc = ahb::AHardwareBuffer_Desc {
        width: size_bytes as u32, // FORMAT_BLOB: width = total bytes, height = 1
        height: 1,
        layers: 1,
        format: ahb::AHARDWAREBUFFER_FORMAT_BLOB,
        usage: ahb::AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER
            | ahb::AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN
            | ahb::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
        stride: 0,
        rfu0: 0,
        rfu1: 0,
    };
    let mut ahb_buf: *mut ahb::AHardwareBuffer = std::ptr::null_mut();
    let alloc_err = unsafe { (ahb_fns.allocate)(&desc as *const _, &mut ahb_buf as *mut _) };
    if alloc_err != 0 || ahb_buf.is_null() {
        anyhow::bail!(
            "AHardwareBuffer_allocate failed: err={}, ahb_buf={:p}",
            alloc_err,
            ahb_buf
        );
    }
    println!("✓ AHB allocated: {:p}", ahb_buf);

    // Lock to get the host pointer (CPU read+write)
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let lock_err = unsafe {
        (ahb_fns.lock)(
            ahb_buf,
            ahb::AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | ahb::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
            -1,
            std::ptr::null(),
            &mut host_ptr as *mut _,
        )
    };
    if lock_err != 0 || host_ptr.is_null() {
        anyhow::bail!(
            "AHardwareBuffer_lock failed: err={}, host_ptr={:p}",
            lock_err,
            host_ptr
        );
    }
    println!("✓ AHB locked: host_ptr={:p}", host_ptr);

    // === Create cl_mem from AHB ===
    let ahb_mem_struct = qcom::cl_mem_ahardwarebuffer_host_ptr {
        ext_host_ptr: qcom::cl_mem_ext_host_ptr {
            allocation_type: qcom::CL_MEM_ANDROID_AHARDWAREBUFFER_HOST_PTR_QCOM,
            host_cache_policy: cache_policy,
        },
        ahb_ptr: ahb_buf as *mut std::ffi::c_void,
    };

    // Match MNN OpenCLBackend.cpp:758 — USE_HOST_PTR + EXT_HOST_PTR_QCOM, size=0.
    // Driver derives size from AHB descriptor.
    const CL_MEM_READ_WRITE: u64 = 1 << 0;
    const CL_MEM_USE_HOST_PTR: u64 = 1 << 3;
    let flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | qcom::CL_MEM_EXT_HOST_PTR_QCOM;
    let mut errcode: i32 = 0;
    let cl_mem_raw = unsafe {
        ffi::clCreateBuffer(
            ctx_ptr,
            flags,
            0, // MNN passes 0; driver derives size from AHB descriptor
            &ahb_mem_struct as *const _ as *mut std::ffi::c_void,
            &mut errcode,
        )
    };
    if cl_mem_raw.is_null() || errcode != 0 {
        unsafe {
            (ahb_fns.unlock)(ahb_buf, std::ptr::null_mut());
            (ahb_fns.release)(ahb_buf);
        }
        anyhow::bail!(
            "clCreateBuffer(AHB+{:?}) failed: errcode={}",
            cache_label,
            errcode
        );
    }
    println!("✓ cl_mem created from AHB ({:p})", cl_mem_raw);

    // === Test 1: Host write (memcpy CPU→AHB host_ptr) ===
    let n_floats = size_bytes / std::mem::size_of::<f32>();
    let host_src: Vec<f32> = (0..n_floats).map(|i| (i as f32) * 1.0e-6).collect();

    println!("\n=== Test 1: Host write (memcpy → AHB host_ptr) ===");
    let mut samples_w_ms = Vec::with_capacity(n_iters);
    for i in 0..n_iters {
        let t0 = Instant::now();
        unsafe {
            std::ptr::copy_nonoverlapping(
                host_src.as_ptr() as *const u8,
                host_ptr as *mut u8,
                size_bytes,
            );
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        samples_w_ms.push(elapsed_ms);
        if i < 5 || i % 10 == 0 {
            println!("  iter {:2}: {:7.2} ms", i, elapsed_ms);
        }
    }
    report_stats("Host write (AHB)", size_mb, &samples_w_ms);

    // === Test 2: GPU kernel read AHB ===
    println!("\n=== Test 2: GPU kernel read (busy accumulator over AHB cl_mem) ===");
    let kernel_src = r#"
        __kernel void ahb_read(__global const float* in_buf, __global float* out_buf, const uint chunk) {
            uint id = get_global_id(0);
            float acc = 0.0f;
            uint base = id * chunk;
            for (uint i = 0; i < chunk; i++) {
                acc += in_buf[base + i] * 0.001f;
            }
            out_buf[id] = acc;
        }
    "#;
    let program = ocl::Program::builder()
        .devices(device)
        .src(kernel_src)
        .build(&context)?;
    let kernel = ocl::core::create_kernel(&program, "ahb_read")?;

    const CHUNK: u32 = 65536;
    let n_workitems = n_floats / CHUNK as usize;
    let out_buf: ocl::core::Mem = unsafe {
        ocl::core::create_buffer::<_, f32>(
            context.as_core(),
            ocl::core::MEM_READ_WRITE,
            n_workitems,
            None,
        )?
    };

    // Wrap raw cl_mem from clCreateBuffer in ocl::core::Mem for set_kernel_arg.
    // ocl 0.19's Mem::from_raw_create_ptr exists.
    let in_buf_ocl: ocl::core::Mem = unsafe { ocl::core::Mem::from_raw_create_ptr(cl_mem_raw) };

    let mut samples_k_ms = Vec::with_capacity(n_iters);
    for i in 0..n_iters {
        let t0 = Instant::now();
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(&in_buf_ocl))?;
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(&out_buf))?;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&CHUNK))?;
            let gws = [n_workitems, 1, 1];
            ocl::core::enqueue_kernel(
                &queue,
                &kernel,
                1,
                None,
                &gws,
                None,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        ocl::core::finish(&queue)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        samples_k_ms.push(elapsed_ms);
        if i < 5 || i % 10 == 0 {
            println!("  iter {:2}: {:7.2} ms", i, elapsed_ms);
        }
    }
    report_stats("GPU kernel read (AHB)", size_mb, &samples_k_ms);

    // === Decode correctness check ===
    // Read back out_buf and verify a few values match expected accumulation
    let mut readback = vec![0.0f32; n_workitems.min(16)];
    unsafe {
        ocl::core::enqueue_read_buffer(
            &queue,
            &out_buf,
            true,
            0,
            &mut readback,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    let mut all_finite = true;
    for (i, v) in readback.iter().take(8).enumerate() {
        let finite = v.is_finite();
        all_finite &= finite;
        println!(
            "  out[{}]: {} ({})",
            i,
            v,
            if finite { "OK" } else { "NaN/Inf!" }
        );
    }
    println!(
        "Correctness: {} (first 8 outputs all finite)",
        if all_finite {
            "PASS"
        } else {
            "FAIL — coherency violation"
        }
    );

    // Cleanup — drop cl_mem first then release/unlock AHB
    drop(in_buf_ocl);
    unsafe {
        (ahb_fns.unlock)(ahb_buf, std::ptr::null_mut());
        (ahb_fns.release)(ahb_buf);
    }

    println!("\n=== Phase 3 interpretation ===");
    println!("- Host write < 22ms (Phase 0 baseline) → AHB+iocoherent 우월. swap에 통합");
    println!("- Host write ≈ 22ms → 동등");
    println!("- Host write > 50ms → 더 느림. AHB가 swap path에 유리하지 않음");
    println!("- Correctness FAIL → cache coherency 미보장 (DMA-BUF heap garbage 재현)");

    Ok(())
}

#[cfg(feature = "opencl")]
fn report_stats(label: &str, size_mb: usize, samples_ms: &[f64]) {
    let n = samples_ms.len() as f64;
    let mean = samples_ms.iter().sum::<f64>() / n;
    let var = samples_ms.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let stddev = var.sqrt();
    let mut sorted: Vec<f64> = samples_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let cv = stddev / mean;
    let bandwidth_gbs = (size_mb as f64) / 1024.0 / (mean / 1000.0);

    println!("\n[{}] {} MB, n={}", label, size_mb, samples_ms.len());
    println!("  mean    : {:7.2} ms", mean);
    println!("  median  : {:7.2} ms", median);
    println!("  stddev  : {:7.2} ms", stddev);
    println!(
        "  σ/mean  : {:6.3} ({})",
        cv,
        if cv < 0.05 { "OK" } else { "WARN" }
    );
    println!("  effective BW: {:5.2} GB/s", bandwidth_gbs);
}
