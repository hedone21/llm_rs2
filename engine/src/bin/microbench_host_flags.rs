//! microbench_host_flags — CL_MEM_HOST_* + ALLOC_HOST_PTR 조합 측정 (Phase 4)
//!
//! 목적: HOST_WRITE_ONLY / HOST_NO_ACCESS hint가 driver 최적화 trigger하는지 측정.
//! Phase 0 baseline: ALLOC_HOST_PTR + READ_WRITE 25MB H2D (within 600MB scaled): ~0.92ms
//!
//! Build: `cargo build --release --target aarch64-linux-android --bin microbench_host_flags`
//! Run:   `adb shell ./microbench_host_flags [SIZE_MB] [N_ITERS]`

#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("microbench_host_flags requires --features opencl");
    std::process::exit(2);
}

#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use ocl::{Context, Device, Platform, Queue};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let size_mb: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(25);
    let n_iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let size_bytes = size_mb * 1024 * 1024;
    let n_floats = size_bytes / std::mem::size_of::<f32>();

    let platform = Platform::default();
    let device = Device::first(platform)?;
    println!("Device: {}", device.name()?);
    println!("Buffer: {} MB, n_iters: {}", size_mb, n_iters);

    let context = Context::builder().platform(platform).devices(device).build()?;
    let queue = Queue::new(&context, device, None)?;
    let host_src: Vec<f32> = (0..n_floats).map(|i| i as f32 * 1.0e-6).collect();

    // CL_MEM flag combinations to measure
    const READ_WRITE: u64 = 1 << 0;
    const WRITE_ONLY: u64 = 1 << 1;
    const READ_ONLY: u64 = 1 << 2;
    const ALLOC_HOST_PTR: u64 = 1 << 4;
    const HOST_WRITE_ONLY: u64 = 1 << 7;
    const HOST_READ_ONLY: u64 = 1 << 8;
    #[allow(dead_code)]
    const HOST_NO_ACCESS: u64 = 1 << 9;

    let combos: &[(&str, u64)] = &[
        ("READ_WRITE | ALLOC_HOST_PTR (baseline)", READ_WRITE | ALLOC_HOST_PTR),
        ("READ_ONLY | ALLOC_HOST_PTR", READ_ONLY | ALLOC_HOST_PTR),
        ("WRITE_ONLY | ALLOC_HOST_PTR", WRITE_ONLY | ALLOC_HOST_PTR),
        ("READ_WRITE | ALLOC_HOST_PTR | HOST_WRITE_ONLY", READ_WRITE | ALLOC_HOST_PTR | HOST_WRITE_ONLY),
        ("READ_ONLY | ALLOC_HOST_PTR | HOST_WRITE_ONLY", READ_ONLY | ALLOC_HOST_PTR | HOST_WRITE_ONLY),
        ("READ_WRITE | ALLOC_HOST_PTR | HOST_READ_ONLY", READ_WRITE | ALLOC_HOST_PTR | HOST_READ_ONLY),
        // HOST_NO_ACCESS conflicts with HOST_WRITE_ONLY/READ_ONLY semantically; test separately
        ("WRITE_ONLY | ALLOC_HOST_PTR (read-once GPU)", WRITE_ONLY | ALLOC_HOST_PTR),
    ];

    println!("\n=== Phase 4 CL_MEM flag combination microbench ===\n");
    let mut results: Vec<(String, f64)> = Vec::new();
    for (label, flags) in combos {
        let buf: ocl::core::Mem = match unsafe {
            ocl::core::create_buffer::<_, f32>(
                context.as_core(),
                ocl::core::MemFlags::from_bits_truncate(*flags),
                n_floats,
                None,
            )
        } {
            Ok(b) => b,
            Err(e) => {
                println!("  [{}] alloc FAILED: {}", label, e);
                continue;
            }
        };
        // Warmup
        unsafe {
            ocl::core::enqueue_write_buffer(
                &queue, &buf, true, 0, &host_src, None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        ocl::core::finish(&queue)?;

        let mut samples = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let t0 = Instant::now();
            unsafe {
                ocl::core::enqueue_write_buffer(
                    &queue, &buf, true, 0, &host_src, None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            ocl::core::finish(&queue)?;
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
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
            "[{}]\n  mean={:.2} median={:.2} σ={:.2} σ/mean={:.3}\n",
            label, mean, median, stddev, cv
        );
        results.push((label.to_string(), mean));
    }

    println!("\n=== Phase 4 summary ===");
    if let Some(baseline) = results.first().map(|(_, m)| *m) {
        println!("\n{:<60} {:>8} {:>8}", "Config", "mean(ms)", "vs base");
        println!("{}", "-".repeat(80));
        for (label, mean) in &results {
            let pct = (mean - baseline) / baseline * 100.0;
            println!("{:<60} {:>8.2} {:>+7.1}%", label, mean, pct);
        }
    }
    println!("\nInterpretation: differences ≤ ±2% are likely noise.");
    println!("Phase 0 baseline showed σ/mean = 5.9% for ALLOC_HOST_PTR alone.");

    Ok(())
}
