//! microbench_recordable_queues — cl_qcom_recordable_queues 측정
//!
//! 목적 (Phase 2): NDRange enqueue overhead가 swap stall 290ms 중 얼마인지 분리.
//! Phase 0 baseline: 600MB H2D = 22ms. 나머지 ~268ms는 prep + dispatch + commit.
//! Recordable queues는 dispatch overhead를 0에 수렴시키므로,
//! 해당 부분이 290ms 중 얼마를 차지하는지 정량 측정.
//!
//! API: dlsym으로 Adreno libOpenCL의 clNewRecordingQCOM 등 동적 로드.
//! Spec: MNN/3rd_party/OpenCLHeaders/CL/cl_ext_qcom.h
//!
//! 측정:
//! - Method A: 100 NDRange enqueue with kernel arg update (per layer simulation)
//! - Method B: 1 record + 100 replay with arg mutation (cl_array_arg_qcom)
//!
//! Build: `cargo build --release --target aarch64-linux-android --bin microbench_recordable_queues`
//! Run:   `adb shell ./microbench_recordable_queues [N_DISPATCHES]`

#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("microbench_recordable_queues requires --features opencl");
    std::process::exit(2);
}

#[cfg(feature = "opencl")]
#[allow(non_camel_case_types, non_upper_case_globals, dead_code)]
mod qcom_ffi {
    use ocl::ffi::{cl_command_queue, cl_event, cl_int, cl_uint};
    use std::ffi::c_void;

    pub type cl_recording_qcom = *mut c_void;

    #[repr(C)]
    pub struct cl_array_arg_qcom {
        pub dispatch_index: cl_uint,
        pub arg_index: cl_uint,
        pub arg_size: usize,
        pub arg_value: *const c_void,
    }

    #[repr(C)]
    pub struct cl_workgroup_qcom {
        pub dispatch_index: cl_uint,
        pub workgroup_size: *const usize,
    }

    #[repr(C)]
    pub struct cl_offset_qcom {
        pub dispatch_index: cl_uint,
        pub offsets: [usize; 3],
    }

    pub type clNewRecordingQCOM_fn =
        unsafe extern "system" fn(cl_command_queue, *mut cl_int) -> cl_recording_qcom;
    pub type clEndRecordingQCOM_fn = unsafe extern "system" fn(cl_recording_qcom) -> cl_int;
    pub type clReleaseRecordingQCOM_fn = unsafe extern "system" fn(cl_recording_qcom) -> cl_int;
    pub type clEnqueueRecordingQCOM_fn = unsafe extern "system" fn(
        cl_command_queue,
        cl_recording_qcom,
        usize,
        *const cl_array_arg_qcom,
        usize,
        *const cl_offset_qcom,
        usize,
        *const cl_workgroup_qcom,
        usize,
        *const cl_workgroup_qcom,
        cl_uint,
        *const cl_event,
        *mut cl_event,
    ) -> cl_int;

    pub struct QcomFns {
        pub new_recording: clNewRecordingQCOM_fn,
        pub end_recording: clEndRecordingQCOM_fn,
        pub release_recording: clReleaseRecordingQCOM_fn,
        pub enqueue_recording: clEnqueueRecordingQCOM_fn,
    }

    #[allow(clippy::missing_transmute_annotations)]
    pub fn load() -> Option<QcomFns> {
        unsafe {
            // libOpenCL.so on Android = /vendor/lib64/libOpenCL.so or /system/vendor/lib64/libOpenCL.so
            let lib_paths = [
                c"/vendor/lib64/libOpenCL.so".as_ptr(),
                c"libOpenCL.so".as_ptr(),
            ];
            let mut handle: *mut c_void = std::ptr::null_mut();
            for path in lib_paths {
                handle = libc::dlopen(path, libc::RTLD_NOW);
                if !handle.is_null() {
                    break;
                }
            }
            if handle.is_null() {
                eprintln!("dlopen libOpenCL.so failed");
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
            Some(QcomFns {
                new_recording: sym!("clNewRecordingQCOM"),
                end_recording: sym!("clEndRecordingQCOM"),
                release_recording: sym!("clReleaseRecordingQCOM"),
                enqueue_recording: sym!("clEnqueueRecordingQCOM"),
            })
        }
    }
}

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use ocl::core::Mem;
    use ocl::{Context, Device, Platform, Program, Queue};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let n_dispatches: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let n_iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);

    println!(
        "n_dispatches per iter: {}, n_iters: {}",
        n_dispatches, n_iters
    );

    let qcom = match qcom_ffi::load() {
        Some(q) => q,
        None => anyhow::bail!("Failed to load cl_qcom_recordable_queues symbols"),
    };
    println!("✓ Loaded clNewRecordingQCOM / clEndRecordingQCOM / clEnqueueRecordingQCOM");

    let platform = Platform::default();
    let device = Device::first(platform)?;
    println!("Device: {}", device.name()?);

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    // Standard queue for preparation (write_buffer, prefill).
    let prep_queue = Queue::new(&context, device, None)?;

    // CL_QUEUE_RECORDABLE_QCOM = (1u << 30). Recordable queues reject write_buffer
    // and other non-NDRange commands, so we keep them separate.
    const CL_QUEUE_RECORDABLE_QCOM: u64 = 1u64 << 30;
    let recordable_props =
        unsafe { ocl::core::CommandQueueProperties::from_bits_unchecked(CL_QUEUE_RECORDABLE_QCOM) };
    let queue = Queue::new(&context, device, Some(recordable_props))?;
    println!("✓ Created standard prep_queue + recordable queue (CL_QUEUE_RECORDABLE_QCOM)");

    // Compile a kernel with two args: input + output. Small compute (~5us per dispatch).
    let src = r#"
        __kernel void layer_op(__global const float* in_buf, __global float* out_buf, const int dummy) {
            int id = get_global_id(0);
            float v = in_buf[id];
            for (int i = 0; i < 32; i++) {
                v = v * 1.0001f + 0.5f;
                v -= 0.5f;
            }
            out_buf[id] = v;
        }
    "#;
    let program = Program::builder()
        .devices(device)
        .src(src)
        .build(&context)?;
    let kernel = ocl::core::create_kernel(&program, "layer_op")?;

    const NF: usize = 4096;
    let in_buf: Mem = unsafe {
        ocl::core::create_buffer::<_, f32>(context.as_core(), ocl::core::MEM_READ_ONLY, NF, None)?
    };
    let out_buf: Mem = unsafe {
        ocl::core::create_buffer::<_, f32>(context.as_core(), ocl::core::MEM_READ_WRITE, NF, None)?
    };

    // Pre-fill input via prep_queue (recordable queue rejects write_buffer).
    let host_data: Vec<f32> = (0..NF).map(|i| i as f32 * 0.001).collect();
    unsafe {
        ocl::core::enqueue_write_buffer(
            &prep_queue,
            &in_buf,
            true,
            0,
            &host_data,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(&prep_queue)?;

    let dummy_arg: i32 = 0;
    let gws = [NF, 1, 1];

    // === Method A: standard NDRange enqueue (n_dispatches per iter) ===
    println!("\n=== Method A: standard clEnqueueNDRangeKernel × N (prep_queue) ===");

    // Warmup A on prep_queue
    unsafe {
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(&in_buf))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(&out_buf))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&dummy_arg))?;
        for _ in 0..10 {
            ocl::core::enqueue_kernel(
                &prep_queue,
                &kernel,
                1,
                None,
                &gws,
                None,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        ocl::core::finish(&prep_queue)?;
    }

    let mut samples_a_ms = Vec::with_capacity(n_iters);
    for it in 0..n_iters {
        let t0 = Instant::now();
        unsafe {
            for d in 0..n_dispatches {
                let dummy_d = d as i32;
                ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(&in_buf))?;
                ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(&out_buf))?;
                ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&dummy_d))?;
                ocl::core::enqueue_kernel(
                    &prep_queue,
                    &kernel,
                    1,
                    None,
                    &gws,
                    None,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            ocl::core::finish(&prep_queue)?;
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        samples_a_ms.push(elapsed_ms);
        println!(
            "  iter {:2}: {:7.2} ms ({:5.1} us/dispatch)",
            it,
            elapsed_ms,
            elapsed_ms * 1000.0 / n_dispatches as f64
        );
    }
    report_stats(
        "Method A: clEnqueueNDRangeKernel × N (prep_queue)",
        &samples_a_ms,
        n_dispatches,
    );

    // === Method B: record once + replay × N ===
    println!("\n=== Method B: clNewRecording → 1× NDRange → clEnd → clEnqueueRecording × N ===");
    // Set up a recording. CRITICAL: ocl 0.19 has unsafe impl ClContextPtr for &Queue,
    // so queue.as_ptr() resolves to cl_context. Use explicit Deref to CommandQueue.
    let q_ref: &ocl::core::CommandQueue = &queue;
    let q_ptr: ocl::ffi::cl_command_queue = q_ref.as_ptr();

    let mut errcode: i32 = 0;
    let recording = unsafe { (qcom.new_recording)(q_ptr, &mut errcode) };
    if recording.is_null() || errcode != 0 {
        eprintln!(
            "clNewRecordingQCOM failed: errcode={}, recording={:p}",
            errcode, recording
        );
        anyhow::bail!("Cannot proceed without recording handle");
    }
    println!("  recording handle: {:p}", recording);

    // Record exactly 1 NDRange dispatch.
    unsafe {
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(&in_buf))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(&out_buf))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&dummy_arg))?;
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
        let end_err = (qcom.end_recording)(recording);
        if end_err != 0 {
            anyhow::bail!("clEndRecordingQCOM failed: err={}", end_err);
        }
    }
    println!("  recording finalized (1 NDRange dispatch)");

    // Warmup replay
    let arg_updates: [qcom_ffi::cl_array_arg_qcom; 0] = [];
    for _ in 0..10 {
        let err = unsafe {
            (qcom.enqueue_recording)(
                q_ptr,
                recording,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        if err != 0 {
            anyhow::bail!("clEnqueueRecordingQCOM warmup failed: err={}", err);
        }
    }
    ocl::core::finish(&queue)?;

    let mut samples_b_ms = Vec::with_capacity(n_iters);
    for it in 0..n_iters {
        let t0 = Instant::now();
        for _d in 0..n_dispatches {
            let err = unsafe {
                (qcom.enqueue_recording)(
                    q_ptr,
                    recording,
                    0,
                    std::ptr::null(), // arg updates
                    0,
                    std::ptr::null(), // offset updates
                    0,
                    std::ptr::null(), // global wg updates
                    0,
                    std::ptr::null(), // local wg updates
                    0,
                    std::ptr::null(),     // event wait
                    std::ptr::null_mut(), // event out
                )
            };
            if err != 0 {
                anyhow::bail!("clEnqueueRecordingQCOM failed: err={}", err);
            }
        }
        ocl::core::finish(&queue)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        samples_b_ms.push(elapsed_ms);
        println!(
            "  iter {:2}: {:7.2} ms ({:5.1} us/replay)",
            it,
            elapsed_ms,
            elapsed_ms * 1000.0 / n_dispatches as f64
        );
    }
    report_stats(
        "Method B: replay × N (no arg update)",
        &samples_b_ms,
        n_dispatches,
    );
    let _ = arg_updates;

    // Cleanup
    unsafe { (qcom.release_recording)(recording) };

    // Summary
    let mean_a = samples_a_ms.iter().sum::<f64>() / samples_a_ms.len() as f64;
    let mean_b = samples_b_ms.iter().sum::<f64>() / samples_b_ms.len() as f64;
    let speedup = mean_a / mean_b;
    println!("\n=== Phase 2 summary ===");
    println!(
        "  Method A mean: {:7.2} ms ({:5.1} us/dispatch)",
        mean_a,
        mean_a * 1000.0 / n_dispatches as f64
    );
    println!(
        "  Method B mean: {:7.2} ms ({:5.1} us/replay)",
        mean_b,
        mean_b * 1000.0 / n_dispatches as f64
    );
    println!("  Speedup B/A:  {:.2}x", speedup);
    println!("\n=== Phase 2 interpretation ===");
    println!("  speedup ≥ 5x → recordable_queues delivers significant dispatch latency reduction");
    println!("  speedup ~ 1x → recordable_queues is no-op or driver bypasses recording");
    println!("\nNOTE: For weight swap, this only helps if dispatch overhead dominates 290ms.");
    println!(
        "      Phase 0 showed H2D = 22ms; remaining ~268ms is mostly Q4 conv + ArcSwap + cl_mem alloc,"
    );
    println!(
        "      none of which are NDRange dispatches. Recordable queues primarily helps inference, not swap."
    );

    Ok(())
}

#[cfg(feature = "opencl")]
fn report_stats(label: &str, samples_ms: &[f64], n_dispatches: usize) {
    let n = samples_ms.len() as f64;
    let mean = samples_ms.iter().sum::<f64>() / n;
    let var = samples_ms.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let stddev = var.sqrt();
    let mut sorted: Vec<f64> = samples_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let cv = stddev / mean;

    println!(
        "\n[{}] n_iters={}, n_dispatches={}",
        label,
        samples_ms.len(),
        n_dispatches
    );
    println!(
        "  mean    : {:7.2} ms ({:6.2} us/dispatch)",
        mean,
        mean * 1000.0 / n_dispatches as f64
    );
    println!("  median  : {:7.2} ms", median);
    println!("  stddev  : {:7.2} ms", stddev);
    println!(
        "  σ/mean  : {:6.3} ({})",
        cv,
        if cv < 0.05 { "OK" } else { "WARN" }
    );
}
