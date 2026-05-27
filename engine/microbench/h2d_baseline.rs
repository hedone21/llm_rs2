//! microbench_h2d_baseline — 600MB single-shot H2D wall-clock 측정
//!
//! 목적 (Phase 0): 290ms swap stall 중 (a) bytes-bound H2D vs (b) driver overhead 분리.
//! Adreno LPDDR5X 84.8 GB/s 이론 lower bound = 600MB / 85GB/s ≈ 7ms.
//! 실측 H2D가 ~7ms이면 290ms 중 ~283ms는 driver overhead → SVM/command_buffer 우회 가능성 높음.
//! 실측 H2D가 100ms+이면 H2D bytes 자체가 병목 → SVM 효과 상한 작음.
//!
//! 측정:
//! - profile_events {OFF, ON} × n=20
//! - clEnqueueWriteBuffer (blocking) wall-clock
//! - 보고: mean, median, p99, σ, σ/mean
//!
//! Build: `cargo build --release --target aarch64-linux-android --bin microbench_h2d_baseline`
//! Run:   `adb shell ./microbench_h2d_baseline [SIZE_MB] [N_ITERS]`

#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("microbench_h2d_baseline requires --features opencl");
    std::process::exit(2);
}

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use ocl::core::Mem;
    use ocl::ffi;
    use ocl::{Context, Device, Platform, Queue};
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

    // Host source data (cold init, deterministic)
    println!("\nAllocating host source ({} MB)...", size_mb);
    let host_src: Vec<f32> = (0..n_floats).map(|i| (i as f32) * 1.0e-6).collect();

    // Two configs: profile_events OFF (production), ON (instrumentation)
    const QUEUE_PROFILING_ENABLE: u64 = 1 << 0;
    let configs: &[(&str, Option<u64>)] = &[
        ("profile_events=OFF", None),
        ("profile_events=ON", Some(QUEUE_PROFILING_ENABLE)),
    ];

    for (label, props) in configs {
        println!("\n=== {} ===", label);

        let queue_props = props.map(ocl::core::CommandQueueProperties::from_bits_truncate);
        let queue = Queue::new(&context, device, queue_props)?;

        // Allocate ALLOC_HOST_PTR buffer (production swap path와 동일)
        let buf: Mem = unsafe {
            ocl::core::create_buffer::<_, f32>(
                context.as_core(),
                ocl::core::MEM_READ_ONLY | ocl::core::MEM_ALLOC_HOST_PTR,
                n_floats,
                None,
            )?
        };

        // Warmup (1회) — driver page pinning, JIT 등 lazy 비용 제거
        unsafe {
            ocl::core::enqueue_write_buffer(
                &queue,
                &buf,
                true,
                0,
                &host_src,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        ocl::core::finish(&queue)?;

        // Measure
        let mut samples_ms = Vec::with_capacity(n_iters);
        for i in 0..n_iters {
            let t0 = Instant::now();
            unsafe {
                ocl::core::enqueue_write_buffer(
                    &queue,
                    &buf,
                    true, // blocking
                    0,
                    &host_src,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            // blocking=true 이미 wait 포함; 안전하게 finish
            ocl::core::finish(&queue)?;
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            samples_ms.push(elapsed_ms);
            println!("  iter {:2}: {:7.2} ms", i, elapsed_ms);
        }

        report_stats(label, size_mb, &samples_ms);

        // Buffer drop은 scope 종료 시 자동
        drop(buf);
        // ocl::ffi 명시적 release (안전을 위해)
        let _ = unsafe { ffi::clFinish(queue.as_core().as_ptr()) };
    }

    println!("\n=== Phase 0 interpretation ===");
    println!(
        " 7~50 ms  : driver overhead 우위, SVM/command_buffer 우회 가능성 높음 → Phase 1/2 PROCEED"
    );
    println!("100~200 ms: 중간, mixed bottleneck → 모든 phase 측정 필요");
    println!(">200 ms   : H2D bytes-bound, SVM 효과 상한 작음 → Phase 3/4 우선");

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
    println!(
        "  σ/mean  : {:6.3} ({})",
        cv,
        if cv < 0.05 { "OK" } else { "WARN >5%" }
    );
    println!("  effective BW: {:5.2} GB/s", bandwidth_gbs);
}
