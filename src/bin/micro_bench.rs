use std::time::{Instant, Duration};
use llm_rs2::core::quant::{BlockQ4_0, BlockQ8_0, QK4_0, QK8_0};
use rand::Rng;

#[cfg(target_arch = "aarch64")]
use llm_rs2::backend::cpu::neon::CpuBackendNeon;
#[cfg(target_arch = "x86_64")]
use llm_rs2::backend::cpu::x86::CpuBackendAVX2;
use llm_rs2::backend::cpu::common::CpuBackendCommon;

// Helper to init random data
fn init_f32(n: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..n).map(|_| rng.random::<f32>() - 0.5).collect()
}

fn init_q4_0(n: usize) -> Vec<BlockQ4_0> {
    let nb = n / QK4_0;
    vec![BlockQ4_0 { d: half::f16::from_f32(1.0), qs: [0; 16] }; nb]
}

fn init_q8_0(n: usize) -> Vec<BlockQ8_0> {
    let nb = n / QK8_0;
    vec![BlockQ8_0 { d: half::f16::from_f32(1.0), qs: [0; 32] }; nb]
}

// Return avg time in ms
fn benchmark<F>(iter: usize, mut f: F) -> f64
where F: FnMut() 
{
    // Warmup
    for _ in 0..iter/10 { f(); }
    
    let start = Instant::now();
    for _ in 0..iter {
        f();
    }
    let dur = start.elapsed();
    (dur.as_secs_f64() * 1000.0) / iter as f64
}

struct BenchResult {
    name: String,
    scalar_ms: f64,
    simd_ms: f64,
}

impl BenchResult {
    fn speedup(&self) -> f64 {
        if self.simd_ms > 0.0 {
            self.scalar_ms / self.simd_ms
        } else {
            0.0
        }
    }
}

fn main() {
    let k = 4096;
    let iter = 10000;
    
    println!("Running micro-benchmarks (K={}, Iter={})", k, iter);
    println!("{:<30} | {:<10} | {:<10} | {:<10}", "Benchmark", "Scalar (ms)", "SIMD (ms)", "Speedup");
    println!("{:-<30}-+-{:-<10}-+-{:-<10}-+-{:-<10}", "", "", "", "");

    let mut results = Vec::new();

    // 1. Quantize Row Q8_0
    results.push(bench_quantize_row_q8_0(k, iter));

    // 2. Dot Product
    results.push(bench_vec_dot_q4_0_q8_0(k, iter));

    for r in results {
        println!("{:<30} | {:<10.4} | {:<10.4} | {:<10.2}x", r.name, r.scalar_ms, r.simd_ms, r.speedup());
    }
}

fn bench_quantize_row_q8_0(k: usize, iter: usize) -> BenchResult {
    let src = init_f32(k);
    let mut dst = init_q8_0(k);
    let backend_scalar = CpuBackendCommon::new();
    
    let scalar_ms = benchmark(iter, || {
        backend_scalar.quantize_row_q8_0(&src, &mut dst, k);
    });

    #[cfg(target_arch = "aarch64")]
    let simd_ms = {
        let backend = CpuBackendNeon::new();
        benchmark(iter, || unsafe { backend.quantize_row_q8_0(&src, &mut dst, k) })
    };

    #[cfg(target_arch = "x86_64")]
    let simd_ms = {
        if is_x86_feature_detected!("avx2") {
            let backend = CpuBackendAVX2::new();
            benchmark(iter, || unsafe { backend.quantize_row_q8_0(&src, &mut dst, k) })
        } else {
            0.0
        }
    };

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let simd_ms = 0.0;

    BenchResult {
        name: "quantize_row_q8_0".to_string(),
        scalar_ms,
        simd_ms,
    }
}

fn bench_vec_dot_q4_0_q8_0(k: usize, iter: usize) -> BenchResult {
    let q4 = init_q4_0(k);
    let q8 = init_q8_0(k);
    let mut s = 0.0;
    let backend_scalar = CpuBackendCommon::new();

    let scalar_ms = benchmark(iter, || {
        unsafe {
            backend_scalar.vec_dot_q4_0_q8_0(k, &mut s, q4.as_ptr(), q8.as_ptr());
        }
        std::hint::black_box(s);
    });

    #[cfg(target_arch = "aarch64")]
    let simd_ms = {
        let backend = CpuBackendNeon::new();
        benchmark(iter, || {
            unsafe { 
                 backend.vec_dot_q4_0_q8_0(k, &mut s, q4.as_ptr(), q8.as_ptr());
            }
            std::hint::black_box(s);
        })
    };

    #[cfg(target_arch = "x86_64")]
    let simd_ms = {
        if is_x86_feature_detected!("avx2") {
            let backend = CpuBackendAVX2::new();
            benchmark(iter, || {
                 unsafe { 
                     backend.vec_dot_q4_0_q8_0(k, &mut s, q4.as_ptr(), q8.as_ptr());
                }
                std::hint::black_box(s);
            })
        } else {
            0.0
        }
    };

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let simd_ms = 0.0;

    BenchResult {
        name: "vec_dot_q4_0_q8_0".to_string(),
        scalar_ms,
        simd_ms,
    }
}
