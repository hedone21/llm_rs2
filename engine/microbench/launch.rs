//! microbench_launch — OpenCL per-kernel-launch overhead 측정
//!
//! 목적: Rust ocl::core API를 통해 noop 커널을 10000회 launch 하여
//! per-launch 오버헤드(μs)를 실측한다. 이 값이 Decode 갭(30.6 ms/tok,
//! 364 launches/token)을 설명하는지 검증한다.
//!
//! 측정 원칙:
//! - 기존 `OpenCLBackend` 사용 금지 (오버헤드 배제)
//! - `CL_QUEUE_PROFILING_ENABLE` OFF (production dispatch 경로와 동일)
//! - set_kernel_arg + enqueue_kernel만 루프 (profile_events 없음)
//! - wall-clock `Instant`로 전체 구간 측정

#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("microbench_launch requires --features opencl");
    std::process::exit(2);
}

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use ocl::core::{ArgVal, Event, Mem};
    use ocl::{Context, Device, Platform, Program, Queue};
    use std::time::Instant;

    // === 1. Platform / Device / Context / Queue ===
    let platform = Platform::default();
    let device = Device::first(platform)?;
    let platform_name = platform.name().unwrap_or_else(|_| "unknown".into());
    let device_name = device.name().unwrap_or_else(|_| "unknown".into());
    println!("Rust ocl::core microbench");
    println!("Platform: {}", platform_name);
    println!("Device:   {}", device_name);

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    // production 경로와 동일: profiling flag OFF
    let queue = Queue::new(&context, device, None)?;

    // === 2. Compile noop kernel ===
    let src = r#"
        __kernel void noop(__global int *dummy) {
            // arg 바인딩만 검증; 실제로 쓰지 않음
            (void)dummy;
        }
    "#;
    let program = Program::builder()
        .devices(device)
        .src(src)
        .build(&context)?;

    let kernel = ocl::core::create_kernel(&program, "noop")?;

    // === 3. Dummy buffer (cl_int[1]) ===
    let dummy_buf: Mem = unsafe {
        ocl::core::create_buffer::<_, i32>(context.as_core(), ocl::core::MEM_READ_WRITE, 1, None)?
    };

    // === 4. Warmup: 1000 launches + finish ===
    let gws = [1usize, 1, 1];
    let lws: Option<[usize; 3]> = None; // driver picks local size
    for _ in 0..1000 {
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(&dummy_buf))?;
            ocl::core::enqueue_kernel(
                &queue,
                &kernel,
                1,
                None,
                &gws,
                lws,
                None::<&Event>,
                None::<&mut Event>,
            )?;
        }
    }
    ocl::core::finish(&queue)?;

    // === 5. Measured region: 10000 launches ===
    const N: usize = 10_000;
    let t_start = Instant::now();
    for _ in 0..N {
        unsafe {
            ocl::core::set_kernel_arg(&kernel, 0, ArgVal::mem(&dummy_buf))?;
            ocl::core::enqueue_kernel(
                &queue,
                &kernel,
                1,
                None,
                &gws,
                lws,
                None::<&Event>,
                None::<&mut Event>,
            )?;
        }
    }
    ocl::core::finish(&queue)?;
    let elapsed = t_start.elapsed();

    let total_us = elapsed.as_secs_f64() * 1_000_000.0;
    let per_launch_us = total_us / N as f64;

    println!("N = {}", N);
    println!("total = {:.2} ms", total_us / 1000.0);
    println!(
        "per-launch = {:.3} us (includes set_kernel_arg + enqueue_kernel + avg finish/N)",
        per_launch_us
    );
    println!("(kernel GPU time per call estimate: negligible, noop)");

    Ok(())
}
