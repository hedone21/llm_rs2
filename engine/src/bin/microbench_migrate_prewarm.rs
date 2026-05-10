//! microbench_migrate_prewarm — clEnqueueMigrateMemObjects(CONTENT_UNDEFINED) prewarm 측정
//!
//! 목적 (Phase 5): page pinning을 사전 수행하여 first-touch latency 분산.
//! Phase 0 baseline write iter 0 = 166.33ms (cold), iter 1+ = ~22-29ms (warm)
//! → migrate prewarm 효과는 cold first-touch 단축 가능성

#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("requires --features opencl");
    std::process::exit(2);
}

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use ocl::ffi;
    use ocl::{Context, Device, Platform, Queue};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let size_mb: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(25);
    let n_runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let size_bytes = size_mb * 1024 * 1024;
    let n_floats = size_bytes / std::mem::size_of::<f32>();

    let platform = Platform::default();
    let device = Device::first(platform)?;
    println!("Device: {}", device.name()?);
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let queue = Queue::new(&context, device, None)?;
    let host_src: Vec<f32> = (0..n_floats).map(|i| i as f32 * 1.0e-6).collect();

    const CL_MIGRATE_MEM_OBJECT_HOST: u64 = 1 << 0;
    const CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED: u64 = 1 << 1;

    let q_ref: &ocl::core::CommandQueue = &queue;
    let q_ptr: ffi::cl_command_queue = q_ref.as_ptr();

    // Two test paths: (A) cold first-write, (B) prewarm + first-write
    let mut cold_first: Vec<f64> = Vec::new();
    let mut warm_first: Vec<f64> = Vec::new();

    println!(
        "\n=== Phase 5: cold vs prewarm first-write, {} MB, n={} ===",
        size_mb, n_runs
    );
    for run in 0..n_runs {
        // Each run gets a fresh buffer
        let buf = unsafe {
            ocl::core::create_buffer::<_, f32>(
                context.as_core(),
                ocl::core::MEM_READ_WRITE | ocl::core::MEM_ALLOC_HOST_PTR,
                n_floats,
                None,
            )?
        };

        if run % 2 == 0 {
            // Cold path
            let t0 = Instant::now();
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
            cold_first.push(t0.elapsed().as_secs_f64() * 1000.0);
        } else {
            // Prewarm path
            let mem_objs: [ffi::cl_mem; 1] = [buf.as_ptr()];
            let migrate_err = unsafe {
                ffi::clEnqueueMigrateMemObjects(
                    q_ptr,
                    1,
                    mem_objs.as_ptr(),
                    CL_MIGRATE_MEM_OBJECT_HOST | CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
                    0,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                )
            };
            if migrate_err != 0 {
                println!("  run {}: migrate failed err={} → SKIP", run, migrate_err);
                continue;
            }
            ocl::core::finish(&queue)?;
            // Now time first write after prewarm
            let t0 = Instant::now();
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
            warm_first.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        // buf goes out of scope, releases. Next iter fresh.
    }

    fn report(label: &str, samples: &[f64]) {
        if samples.is_empty() {
            println!("[{}] no samples", label);
            return;
        }
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let stddev = var.sqrt();
        let cv = stddev / mean;
        println!(
            "[{}] n={}, mean={:.2} median={:.2} σ={:.2} σ/mean={:.3}",
            label,
            samples.len(),
            mean,
            median,
            stddev,
            cv
        );
    }

    println!();
    report("Cold first-write", &cold_first);
    report("Prewarm + first-write", &warm_first);

    let cold_mean: f64 = cold_first.iter().sum::<f64>() / cold_first.len().max(1) as f64;
    let warm_mean: f64 = warm_first.iter().sum::<f64>() / warm_first.len().max(1) as f64;
    let pct_change = (warm_mean - cold_mean) / cold_mean * 100.0;

    println!(
        "\nDifference: warm vs cold = {:+.1}% ({:.2}ms vs {:.2}ms)",
        pct_change, warm_mean, cold_mean
    );
    println!("\nInterpretation:");
    println!("  warm < cold by ≥30% → prewarm effective. Combine with LISWAP-1.");
    println!("  warm ≈ cold        → migrate hint is no-op. SKIP.");
    println!("  warm > cold        → prewarm cost dominates. SKIP.");
    Ok(())
}
