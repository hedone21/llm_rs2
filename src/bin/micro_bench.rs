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

fn benchmark<F>(name: &str, iter: usize, mut f: F) 
where F: FnMut() 
{
    // Warmup
    for _ in 0..iter/10 { f(); }
    
    let start = Instant::now();
    for _ in 0..iter {
        f();
    }
    let dur = start.elapsed();
    let avg = dur.as_secs_f64() / iter as f64;
    println!("{:<30}: Total {:.4}s | Avg {:.4}ms", name, dur.as_secs_f64(), avg * 1000.0);
}

fn main() {
    let k = 4096;
    let iter = 10000;
    
    println!("Running micro-benchmarks (K={}, Iter={})", k, iter);
    
    bench_scalar(k, iter);

    #[cfg(target_arch = "aarch64")]
    bench_neon(k, iter);

    #[cfg(target_arch = "x86_64")]
    bench_avx2(k, iter);
}

fn bench_scalar(k: usize, iter: usize) {
    let backend = CpuBackendCommon::new();
    println!("\n--- Scalar Benchmarks ---");

    // 1. Quantize Row Q8_0
    let src = init_f32(k);
    let mut dst = init_q8_0(k);
    
    benchmark("quantize_row_q8_0", iter, || {
        backend.quantize_row_q8_0(&src, &mut dst, k);
    });

    // 2. Dot Product
    let q4 = init_q4_0(k);
    let q8 = init_q8_0(k);
    let mut s = 0.0;
    
    benchmark("vec_dot_q4_0_q8_0", iter, || {
        unsafe {
            backend.vec_dot_q4_0_q8_0(k, &mut s, q4.as_ptr(), q8.as_ptr());
        }
        std::hint::black_box(s);
    });
}

#[cfg(target_arch = "aarch64")]
fn bench_neon(k: usize, iter: usize) {
    let backend = CpuBackendNeon::new();
    println!("\n--- NEON Benchmarks ---");

    // 1. Quantize Row Q8_0
    let src = init_f32(k);
    let mut dst = init_q8_0(k);
    
    benchmark("quantize_row_q8_0", iter, || {
        unsafe { backend.quantize_row_q8_0(&src, &mut dst, k); }
    });

    // 2. Dot Product
    let q4 = init_q4_0(k);
    let q8 = init_q8_0(k);
    let mut s = 0.0;
    
    benchmark("vec_dot_q4_0_q8_0", iter, || {
        unsafe { 
             backend.vec_dot_q4_0_q8_0(k, &mut s, q4.as_ptr(), q8.as_ptr());
        }
        std::hint::black_box(s);
    });
}

#[cfg(target_arch = "x86_64")]
fn bench_avx2(k: usize, iter: usize) {
    if !is_x86_feature_detected!("avx2") {
        println!("AVX2 not supported on this machine.");
        return;
    }
    
    let backend = CpuBackendAVX2::new();
    println!("\n--- AVX2 Benchmarks ---");

    // 1. Quantize Row Q8_0
    let src = init_f32(k);
    let mut dst = init_q8_0(k);
    
    benchmark("quantize_row_q8_0", iter, || {
        unsafe { backend.quantize_row_q8_0(&src, &mut dst, k); }
    });

    // 2. Dot Product
    let q4 = init_q4_0(k);
    let q8 = init_q8_0(k);
    let mut s = 0.0;
    
    benchmark("vec_dot_q4_0_q8_0", iter, || {
        unsafe { 
             backend.vec_dot_q4_0_q8_0(k, &mut s, q4.as_ptr(), q8.as_ptr());
        }
        std::hint::black_box(s);
    });
}
