//! microbench_two_queue_concurrent — Adreno multi-queue HW serialize 직접 측정 (Phase 6)
//!
//! 목적: 9개 SW 트랙 negative의 공통 가설("Adreno HW serialize")을 직접 검증.
//!
//! 측정: 두 cl_command_queue가 disjoint cl_mem에 1ms compute kernel을 동시 launch.
//! - parallel ~1ms → HW serialize 부재 → multi-queue 트랙 재해석 필요
//! - serial ~2ms → HW serialize 확정 → 본 paper 핵심 evidence
//!
//! Configurations:
//! - same-context vs multi-context
//! - in-order vs out-of-order queue
//! - profile_events ON vs OFF
//!
//! Build: `cargo build --release --target aarch64-linux-android --bin microbench_two_queue_concurrent`
//! Run:   `adb shell ./microbench_two_queue_concurrent [N_ITERS]`

#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("microbench_two_queue_concurrent requires --features opencl");
    std::process::exit(2);
}

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use ocl::core::{ArgVal, CommandQueueProperties, Mem};
    use ocl::{Context, Device, Platform, Program, Queue};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let n_iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);

    let platform = Platform::default();
    let device = Device::first(platform)?;
    println!("Device: {}", device.name()?);
    println!("n_iters per config: {}", n_iters);

    // Two contexts (multi-context test)
    let ctx_a = Context::builder().platform(platform).devices(device).build()?;
    let ctx_b = Context::builder().platform(platform).devices(device).build()?;

    // === Compile busy kernel (1ms target) on both contexts ===
    let src = r#"
        __kernel void busy(__global float* out, const int iters) {
            int id = get_global_id(0);
            float v = (float)id;
            for (int i = 0; i < iters; i++) {
                v = v * 1.00001f + 0.5f;
                v -= 0.5f;
            }
            out[id] = v;
        }
    "#;
    let program_a = Program::builder().devices(device).src(src).build(&ctx_a)?;
    let program_b = Program::builder().devices(device).src(src).build(&ctx_b)?;
    let kernel_a = ocl::core::create_kernel(&program_a, "busy")?;
    let kernel_b = ocl::core::create_kernel(&program_b, "busy")?;

    // Disjoint output buffers per queue
    const GSIZE: usize = 1024;
    // Same-context: both buffers in ctx_a
    let buf_a_in_a: Mem = unsafe {
        ocl::core::create_buffer::<_, f32>(
            ctx_a.as_core(), ocl::core::MEM_READ_WRITE, GSIZE, None,
        )?
    };
    let buf_a2_in_a: Mem = unsafe {
        ocl::core::create_buffer::<_, f32>(
            ctx_a.as_core(), ocl::core::MEM_READ_WRITE, GSIZE, None,
        )?
    };
    // Multi-context: buffer in ctx_b
    let buf_b_in_b: Mem = unsafe {
        ocl::core::create_buffer::<_, f32>(
            ctx_b.as_core(), ocl::core::MEM_READ_WRITE, GSIZE, None,
        )?
    };

    // Tune iters to ~1ms duration
    let mut iters: i32 = 100_000;
    println!("\nTuning kernel iterations to ~1ms...");
    for _ in 0..6 {
        let q = Queue::new(&ctx_a, device, None)?;
        let t0 = Instant::now();
        unsafe {
            ocl::core::set_kernel_arg(&kernel_a, 0, ArgVal::mem(&buf_a_in_a))?;
            ocl::core::set_kernel_arg(&kernel_a, 1, ArgVal::scalar(&iters))?;
            ocl::core::enqueue_kernel(
                &q, &kernel_a, 1, None, &[GSIZE, 1, 1], None,
                None::<&ocl::core::Event>, None::<&mut ocl::core::Event>,
            )?;
        }
        ocl::core::finish(&q)?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("  iters={}: {:.2} ms", iters, ms);
        if ms < 0.7 {
            iters *= 2;
        } else if ms > 1.5 {
            iters = (iters as f64 / (ms / 1.0)) as i32;
        } else {
            break;
        }
    }
    println!("  Final iters={} (~1ms target)", iters);

    fn run_kernel(
        queue: &Queue,
        kernel: &ocl::core::Kernel,
        buf: &Mem,
        iters: i32,
    ) -> anyhow::Result<()> {
        unsafe {
            ocl::core::set_kernel_arg(kernel, 0, ArgVal::mem(buf))?;
            ocl::core::set_kernel_arg(kernel, 1, ArgVal::scalar(&iters))?;
            ocl::core::enqueue_kernel(
                queue, kernel, 1, None, &[GSIZE, 1, 1], None,
                None::<&ocl::core::Event>, None::<&mut ocl::core::Event>,
            )?;
        }
        Ok(())
    }

    println!("\n=== Two-queue concurrent kernel measurement ===");
    println!("Hypothesis: if HW serializes, two-queue wall-clock ≈ 2× single-queue.");
    println!("            if HW parallelizes, two-queue wall-clock ≈ 1× single-queue.\n");

    enum Config {
        Single,
        SameCtxInOrder,
        SameCtxOoO,
        MultiCtxInOrder,
        MultiCtxProf,
    }
    let configs: Vec<(&str, Config)> = vec![
        ("Single-queue baseline (1 kernel)", Config::Single),
        ("Same-context, in-order × 2", Config::SameCtxInOrder),
        ("Same-context, out-of-order × 2", Config::SameCtxOoO),
        ("Multi-context, in-order × 2", Config::MultiCtxInOrder),
        ("Multi-context + profile_events × 2", Config::MultiCtxProf),
    ];

    let mut all_results: Vec<(String, f64, f64)> = Vec::new();
    for (label, cfg) in configs {
        println!("--- {} ---", label);
        let (q1, q2_opt, kernel_for_q2, buf_for_q2): (
            Queue,
            Option<Queue>,
            &ocl::core::Kernel,
            &Mem,
        ) = match cfg {
            Config::Single => (Queue::new(&ctx_a, device, None)?, None, &kernel_a, &buf_a_in_a),
            Config::SameCtxInOrder => (
                Queue::new(&ctx_a, device, None)?,
                Some(Queue::new(&ctx_a, device, None)?),
                &kernel_a,
                &buf_a2_in_a,
            ),
            Config::SameCtxOoO => {
                let p = CommandQueueProperties::OUT_OF_ORDER_EXEC_MODE_ENABLE;
                (
                    Queue::new(&ctx_a, device, Some(p))?,
                    Some(Queue::new(&ctx_a, device, Some(p))?),
                    &kernel_a,
                    &buf_a2_in_a,
                )
            }
            Config::MultiCtxInOrder => (
                Queue::new(&ctx_a, device, None)?,
                Some(Queue::new(&ctx_b, device, None)?),
                &kernel_b,
                &buf_b_in_b,
            ),
            Config::MultiCtxProf => {
                let p = CommandQueueProperties::PROFILING_ENABLE;
                (
                    Queue::new(&ctx_a, device, Some(p))?,
                    Some(Queue::new(&ctx_b, device, Some(p))?),
                    &kernel_b,
                    &buf_b_in_b,
                )
            }
        };

        // Warmup
        for _ in 0..5 {
            run_kernel(&q1, &kernel_a, &buf_a_in_a, iters)?;
            if let Some(q2) = q2_opt.as_ref() {
                run_kernel(q2, kernel_for_q2, buf_for_q2, iters)?;
            }
            ocl::core::finish(&q1)?;
            if let Some(q2) = q2_opt.as_ref() {
                ocl::core::finish(q2)?;
            }
        }

        let mut samples = Vec::with_capacity(n_iters);
        for _it in 0..n_iters {
            let t0 = Instant::now();
            // Issue both kernels in rapid succession (same thread, no sleep)
            run_kernel(&q1, &kernel_a, &buf_a_in_a, iters)?;
            if let Some(q2) = q2_opt.as_ref() {
                run_kernel(q2, kernel_for_q2, buf_for_q2, iters)?;
            }
            // Wait for both
            ocl::core::finish(&q1)?;
            if let Some(q2) = q2_opt.as_ref() {
                ocl::core::finish(q2)?;
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            samples.push(ms);
        }

        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let stddev = var.sqrt();
        let cv = stddev / mean;
        println!(
            "  mean={:.2} median={:.2} σ={:.2} σ/mean={:.3}",
            mean, median, stddev, cv
        );
        all_results.push((label.to_string(), mean, median));
    }

    // === Summary ===
    println!("\n=== Phase 6 summary ===");
    let baseline_mean = all_results[0].1;
    println!("\n{:<45} {:>10} {:>10} {:>12}", "Config", "mean", "median", "ratio_to_1q");
    println!("{}", "-".repeat(80));
    for (label, mean, median) in &all_results {
        let ratio = mean / baseline_mean;
        println!(
            "{:<45} {:>8.2}ms {:>8.2}ms {:>11.2}x",
            label, mean, median, ratio
        );
    }

    println!("\n=== Phase 6 interpretation ===");
    println!("- ratio ≈ 1.0x → two queues run parallel → HW NO_SERIALIZE");
    println!("- ratio ≈ 2.0x → two queues run serial   → HW SERIALIZE confirmed");
    println!("- ratio ≈ 1.5x → partial overlap         → HW partial serialize");
    println!("\nConclusion (paper main evidence):");
    println!("  Adreno 830 OpenCL command processor schedules — measured directly above.");

    Ok(())
}
