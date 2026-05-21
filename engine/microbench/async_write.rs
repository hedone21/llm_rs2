//! microbench_async_write — Adreno OpenCL driver의 async write_buffer + kernel overlap 측정
//!
//! 목적: `clEnqueueWriteBuffer(blocking=CL_FALSE)` + 후속
//! `clEnqueueNDRangeKernel` 시퀀스에서 실제 DMA/compute overlap이 발생하는지
//! 확인한다. partition fused norm-merge 설계의 핵심 전제 (Step 0 gate).
//!
//! 판정:
//! - blocking_write 평균 - non_blocking_write 평균 >= 0.1 ms → OVERLAP_OK
//! - 그 외 → NO_OVERLAP
//!
//! 측정 원칙:
//! - 기존 `OpenCLBackend` 사용 금지 (오버헤드 배제)
//! - `CL_QUEUE_PROFILING_ENABLE` OFF (production과 동일, CLAUDE.md 제약)
//! - wall-clock `Instant`로 write+kernel+clFinish 구간 측정
//! - 같은 context/queue/buffer 재사용

#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("microbench_async_write requires --features opencl");
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
    println!("Platform: {}", platform_name);
    println!("Device:   {}", device_name);

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    // Profiling OFF (production과 동일)
    let queue = Queue::new(&context, device, None)?;

    // === 2. Compile busy kernel ===
    // Dummy compute-heavy kernel: ~1 ms 목표
    let src = r#"
        __kernel void busy(__global float* out, const int iters) {
            int id = get_global_id(0);
            float v = (float)id;
            for (int i = 0; i < iters; i++) {
                v = v * 1.00001f + 0.5f;
                v = v - 0.5f;
            }
            out[id] = v;
        }
    "#;
    let program = Program::builder()
        .devices(device)
        .src(src)
        .build(&context)?;

    let kernel = ocl::core::create_kernel(&program, "busy")?;

    // === 3. Buffers ===
    // Write target: 6 KB (6 * 1024 bytes = 1536 float) — partition cpu_staging 크기
    const WRITE_BYTES: usize = 6 * 1024;
    const WRITE_FLOATS: usize = WRITE_BYTES / std::mem::size_of::<f32>();

    let write_buf: Mem = unsafe {
        ocl::core::create_buffer::<_, f32>(
            context.as_core(),
            ocl::core::MEM_READ_WRITE,
            WRITE_FLOATS,
            None,
        )?
    };

    // Kernel output buffer: global size 1024 floats
    const GSIZE: usize = 1024;
    let out_buf: Mem = unsafe {
        ocl::core::create_buffer::<_, f32>(
            context.as_core(),
            ocl::core::MEM_READ_WRITE,
            GSIZE,
            None,
        )?
    };

    // Host staging data (pinned-ish vec)
    let host_data: Vec<f32> = (0..WRITE_FLOATS).map(|i| i as f32 * 0.001).collect();

    // === 4. Tune iters to ~1 ms kernel duration ===
    fn run_kernel(
        queue: &Queue,
        kernel: &ocl::core::Kernel,
        out_buf: &Mem,
        gsize: usize,
        iters: i32,
    ) -> anyhow::Result<()> {
        let gws = [gsize, 1, 1];
        let lws: Option<[usize; 3]> = None;
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(out_buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::scalar(&iters))?;
            ocl::core::enqueue_kernel(
                queue,
                kernel,
                1,
                None,
                &gws,
                lws,
                None::<&Event>,
                None::<&mut Event>,
            )?;
        }
        Ok(())
    }

    // Initial tune: start at 100_000, binary-search adjust to hit ~1 ms
    let mut iters: i32 = 100_000;
    let target_us = 1000.0_f64;
    let mut kernel_measured_ms = 0.0_f64;

    // Warm up kernel cache
    for _ in 0..5 {
        run_kernel(&queue, &kernel, &out_buf, GSIZE, iters)?;
    }
    ocl::core::finish(&queue)?;

    // Tune loop (max 6 iterations)
    for _tune_step in 0..6 {
        let t0 = Instant::now();
        for _ in 0..10 {
            run_kernel(&queue, &kernel, &out_buf, GSIZE, iters)?;
        }
        ocl::core::finish(&queue)?;
        let avg_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / 10.0;
        kernel_measured_ms = avg_us / 1000.0;

        // Within 30% of target -> accept
        if (avg_us - target_us).abs() / target_us < 0.30 {
            break;
        }
        // Rescale iters proportionally, clamp
        let scale = target_us / avg_us.max(1.0);
        let next = ((iters as f64) * scale) as i64;
        iters = next.clamp(1_000, 10_000_000) as i32;
    }

    // === 5. Scenarios (교차 실행으로 thermal/driver 편향 제거) ===
    // 샘플 수를 대폭 늘리고 (a), (b)를 번갈아 실행해
    // 순서에 따른 편향을 평균으로 상쇄한다.
    const TOTAL_EACH: usize = 100; // 시나리오당 100회
    const WARMUP: usize = 10;

    let host_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            host_data.as_ptr() as *const u8,
            WRITE_FLOATS * std::mem::size_of::<f32>(),
        )
    };

    // Warmup both paths
    for _ in 0..WARMUP {
        unsafe {
            ocl::core::enqueue_write_buffer(
                &queue,
                &write_buf,
                true,
                0,
                host_bytes,
                None::<&Event>,
                None::<&mut Event>,
            )?;
        }
        run_kernel(&queue, &kernel, &out_buf, GSIZE, iters)?;
        ocl::core::finish(&queue)?;

        unsafe {
            ocl::core::enqueue_write_buffer(
                &queue,
                &write_buf,
                false,
                0,
                host_bytes,
                None::<&Event>,
                None::<&mut Event>,
            )?;
        }
        run_kernel(&queue, &kernel, &out_buf, GSIZE, iters)?;
        ocl::core::finish(&queue)?;
    }

    let mut samples_a: Vec<f64> = Vec::with_capacity(TOTAL_EACH);
    let mut samples_b: Vec<f64> = Vec::with_capacity(TOTAL_EACH);

    // 교차 실행: ABABAB...
    for _ in 0..TOTAL_EACH {
        // (a) blocking
        let t0 = Instant::now();
        unsafe {
            ocl::core::enqueue_write_buffer(
                &queue,
                &write_buf,
                true,
                0,
                host_bytes,
                None::<&Event>,
                None::<&mut Event>,
            )?;
        }
        run_kernel(&queue, &kernel, &out_buf, GSIZE, iters)?;
        ocl::core::finish(&queue)?;
        samples_a.push(t0.elapsed().as_secs_f64() * 1_000_000.0);

        // (b) non-blocking
        let t0 = Instant::now();
        unsafe {
            ocl::core::enqueue_write_buffer(
                &queue,
                &write_buf,
                false,
                0,
                host_bytes,
                None::<&Event>,
                None::<&mut Event>,
            )?;
        }
        run_kernel(&queue, &kernel, &out_buf, GSIZE, iters)?;
        ocl::core::finish(&queue)?;
        samples_b.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }

    fn stats(v: &[f64]) -> (f64, f64, f64, f64) {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let mut sorted = v.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let min = sorted[0];
        (mean, var.sqrt(), median, min)
    }

    let (a_mean, a_std, a_med, a_min) = stats(&samples_a);
    let (b_mean, b_std, b_med, b_min) = stats(&samples_b);
    let delta_mean = a_mean - b_mean;
    let delta_med = a_med - b_med;
    let delta_min = a_min - b_min;

    // === 6. Report (매우 엄격한 파싱 포맷) ===
    println!(
        "kernel_target_ms: 1.0 (tuned iters={}, measured={:.2} ms)",
        iters, kernel_measured_ms
    );
    println!("samples_per_scenario: {}", samples_a.len());
    println!(
        "[a] blocking_write:     avg={:.2} ms  stdev={:.2}  median={:.2}  min={:.2}",
        a_mean / 1000.0,
        a_std / 1000.0,
        a_med / 1000.0,
        a_min / 1000.0
    );
    println!(
        "[b] non_blocking_write: avg={:.2} ms  stdev={:.2}  median={:.2}  min={:.2}",
        b_mean / 1000.0,
        b_std / 1000.0,
        b_med / 1000.0,
        b_min / 1000.0
    );
    println!(
        "delta: a - b = {:.2} ms (= {:.2} us overlap_gain)",
        delta_mean / 1000.0,
        delta_mean
    );
    println!(
        "delta_median: {:.2} us   delta_min: {:.2} us",
        delta_med, delta_min
    );
    // 판정은 mean 기준 (요구 포맷).
    // 단, median/min도 함께 볼 것 — 불안정 시 경고
    if delta_mean > 100.0 {
        println!("verdict: OVERLAP_OK");
    } else {
        println!("verdict: NO_OVERLAP");
    }

    Ok(())
}
