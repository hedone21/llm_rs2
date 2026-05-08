//! microbench_svm_throughput — Adreno SVM fine-grain buffer throughput 측정
//!
//! 목적 (Phase 1): SVM fine-grain이 ALLOC_HOST_PTR Map/Unmap를 우회해 weight swap
//! 비용을 줄일 수 있는지 정량 측정.
//! Phase 0 baseline: ALLOC_HOST_PTR 600MB H2D = 22ms (27.5 GB/s).
//! 이론: SVM fine-grain은 host write 직후 GPU가 sniff → write 자체가 H2D
//! 대체. 만약 host write가 ~22ms 안쪽이면 동등. 더 빠르면 winner.
//!
//! 측정:
//! - SVM fine-grain alloc 600MB
//! - host write (memcpy) wall-clock
//! - GPU kernel read (busy loop accumulator) wall-clock — driver가 page를
//!   GPU 측에 mirror할 때의 first-touch latency 검출
//! - 비교: ALLOC_HOST_PTR baseline (microbench_h2d_baseline) vs SVM
//!
//! Build: `cargo build --release --target aarch64-linux-android --bin microbench_svm_throughput`
//! Run:   `adb shell ./microbench_svm_throughput [SIZE_MB] [N_ITERS]`

#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("microbench_svm_throughput requires --features opencl");
    std::process::exit(2);
}

#[cfg(feature = "opencl")]
#[allow(non_camel_case_types, non_upper_case_globals, dead_code)]
mod svm_ffi {
    use ocl::ffi::{cl_context, cl_int, cl_kernel, cl_uint};
    use std::ffi::c_void;

    pub type cl_svm_mem_flags = u64;
    pub const CL_MEM_READ_WRITE: cl_svm_mem_flags = 1 << 0;
    pub const CL_MEM_SVM_FINE_GRAIN_BUFFER: cl_svm_mem_flags = 1 << 10;
    pub const CL_MEM_SVM_ATOMICS: cl_svm_mem_flags = 1 << 11;

    unsafe extern "system" {
        pub fn clSVMAlloc(
            context: cl_context,
            flags: cl_svm_mem_flags,
            size: usize,
            alignment: cl_uint,
        ) -> *mut c_void;

        pub fn clSVMFree(context: cl_context, svm_pointer: *mut c_void);

        pub fn clSetKernelArgSVMPointer(
            kernel: cl_kernel,
            arg_index: cl_uint,
            arg_value: *const c_void,
        ) -> cl_int;
    }
}

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use ocl::core::ClContextPtr;
    use ocl::{Context, Device, Platform, Program, Queue};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let size_mb: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(600);
    let n_iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);

    let size_bytes = size_mb * 1024 * 1024;
    let n_floats = size_bytes / std::mem::size_of::<f32>();

    let platform = Platform::default();
    let device = Device::first(platform)?;
    println!("Platform: {}", platform.name()?);
    println!("Device:   {}", device.name()?);
    println!(
        "Buffer:   {} MB ({} floats), n_iters: {}",
        size_mb, n_floats, n_iters
    );

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let queue = Queue::new(&context, device, None)?;
    let ctx_raw: ocl::ffi::cl_context = ClContextPtr::as_ptr(&&context);

    // SVM fine-grain alloc
    println!("\nclSVMAlloc(FINE_GRAIN_BUFFER | ATOMICS, {} bytes, align=64)...", size_bytes);
    let flags = svm_ffi::CL_MEM_READ_WRITE
        | svm_ffi::CL_MEM_SVM_FINE_GRAIN_BUFFER
        | svm_ffi::CL_MEM_SVM_ATOMICS;
    let svm_ptr =
        unsafe { svm_ffi::clSVMAlloc(ctx_raw, flags, size_bytes, 64) };
    if svm_ptr.is_null() {
        anyhow::bail!(
            "clSVMAlloc returned NULL ({} MB request). Driver may have refused the size.",
            size_mb
        );
    }
    println!("  svm_ptr = {:p}", svm_ptr);

    // Source data (host-side reference)
    let host_src: Vec<f32> = (0..n_floats).map(|i| (i as f32) * 1.0e-6).collect();

    // === Test 1: Host write throughput (no GPU read) ===
    println!("\n=== Test 1: SVM host write (memcpy CPU→SVM) ===");
    println!("(SVM fine-grain: no Map/Unmap needed; CPU write is GPU-visible immediately)");
    let mut samples_write_ms = Vec::with_capacity(n_iters);
    for i in 0..n_iters {
        let t0 = Instant::now();
        unsafe {
            std::ptr::copy_nonoverlapping(
                host_src.as_ptr() as *const u8,
                svm_ptr as *mut u8,
                size_bytes,
            );
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        samples_write_ms.push(elapsed_ms);
        println!("  iter {:2}: {:7.2} ms", i, elapsed_ms);
    }
    report_stats("SVM host write", size_mb, &samples_write_ms);

    // === Test 2: GPU kernel read (first-touch + accumulator) ===
    println!("\n=== Test 2: GPU kernel read (busy accumulator over SVM) ===");

    // Kernel: read SVM buffer, accumulate to small output. Use floats but
    // one work-item per CHUNK to avoid massive GPU dispatch.
    const CHUNK: usize = 65536; // each WI processes 64K floats
    let n_workitems = n_floats / CHUNK;
    let src = format!(
        r#"
        __kernel void svm_read(__global const float* in_buf, __global float* out_buf, const uint chunk) {{
            uint id = get_global_id(0);
            float acc = 0.0f;
            uint base = id * chunk;
            for (uint i = 0; i < chunk; i++) {{
                acc += in_buf[base + i] * 0.001f;
            }}
            out_buf[id] = acc;
        }}
        "#
    );
    let program = Program::builder()
        .devices(device)
        .src(src)
        .build(&context)?;
    let kernel = ocl::core::create_kernel(&program, "svm_read")?;

    let out_buf: ocl::core::Mem = unsafe {
        ocl::core::create_buffer::<_, f32>(
            context.as_core(),
            ocl::core::MEM_READ_WRITE,
            n_workitems,
            None,
        )?
    };

    let mut samples_kernel_ms = Vec::with_capacity(n_iters);
    for i in 0..n_iters {
        // Optional: refresh SVM data so GPU sees latest CPU writes (atomics
        // guarantees coherence on ATOMICS-supporting fine-grain devices).
        // For the fine-grain path we skip explicit Map/Unmap.

        let t0 = Instant::now();
        unsafe {
            // arg 0: SVM pointer
            let err =
                svm_ffi::clSetKernelArgSVMPointer(kernel.as_ptr(), 0, svm_ptr as *const _);
            if err != 0 {
                anyhow::bail!("clSetKernelArgSVMPointer failed: err={}", err);
            }
            // arg 1: regular cl_mem (output)
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(&out_buf))?;
            // arg 2: chunk uint
            let chunk_u32 = CHUNK as u32;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&chunk_u32))?;

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
        samples_kernel_ms.push(elapsed_ms);
        println!("  iter {:2}: {:7.2} ms", i, elapsed_ms);
    }
    report_stats("GPU kernel read SVM", size_mb, &samples_kernel_ms);

    // === Test 3: combined (host write + GPU read) — production swap simulator ===
    println!("\n=== Test 3: Combined (host write + GPU kernel read) ===");
    let mut samples_combined_ms = Vec::with_capacity(n_iters);
    for i in 0..n_iters {
        let t0 = Instant::now();
        unsafe {
            std::ptr::copy_nonoverlapping(
                host_src.as_ptr() as *const u8,
                svm_ptr as *mut u8,
                size_bytes,
            );
            let err =
                svm_ffi::clSetKernelArgSVMPointer(kernel.as_ptr(), 0, svm_ptr as *const _);
            if err != 0 {
                anyhow::bail!("clSetKernelArgSVMPointer failed: err={}", err);
            }
            ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(&out_buf))?;
            let chunk_u32 = CHUNK as u32;
            ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&chunk_u32))?;
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
        samples_combined_ms.push(elapsed_ms);
        println!("  iter {:2}: {:7.2} ms", i, elapsed_ms);
    }
    report_stats("Host write + kernel read", size_mb, &samples_combined_ms);

    // Cleanup
    unsafe { svm_ffi::clSVMFree(ctx_raw, svm_ptr) };

    println!("\n=== Phase 1 interpretation ===");
    println!("- SVM host write < 22 ms (Phase 0 baseline) → SVM 우월. swap_executor에 통합");
    println!("- SVM host write ≈ 22 ms                    → 동등. complexity 늘리지 않으면 채택 안함");
    println!("- SVM host write > 30 ms                    → driver coercion. 'compliance wrapper' finding");
    println!("- Kernel read >> kernel read on host_ptr     → first-touch overhead 잔존");

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
    let p99 = sorted[(sorted.len() as f64 * 0.99) as usize].min(*sorted.last().unwrap());
    let cv = stddev / mean;
    let bandwidth_gbs = (size_mb as f64) / 1024.0 / (mean / 1000.0);

    println!("\n[{}] {} MB, n={}", label, size_mb, samples_ms.len());
    println!("  mean    : {:7.2} ms", mean);
    println!("  median  : {:7.2} ms", median);
    println!("  p99     : {:7.2} ms", p99);
    println!("  stddev  : {:7.2} ms", stddev);
    println!("  σ/mean  : {:6.3} ({})", cv, if cv < 0.05 { "OK" } else { "WARN" });
    println!("  effective BW: {:5.2} GB/s", bandwidth_gbs);
}
