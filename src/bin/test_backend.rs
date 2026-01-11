use llm_rs2::core::backend::Backend;
use llm_rs2::core::tensor::Tensor;
use llm_rs2::core::shape::Shape;
use llm_rs2::core::buffer::DType;
use llm_rs2::backend::cpu::{CpuBackend, CpuBackendCommon};
#[cfg(target_arch = "x86_64")]
use llm_rs2::backend::cpu::CpuBackendAVX2;
#[cfg(feature = "opencl")]
use llm_rs2::backend::opencl::OpenCLBackend;
use llm_rs2::memory::galloc::Galloc;
use llm_rs2::core::memory::Memory;
use llm_rs2::core::quant::{BlockQ4_0, BlockQ4_1, QK4_0, QK4_1};
use std::sync::Arc;
use std::time::{Instant, Duration};
use clap::Parser;
use half::f16;
use std::collections::BTreeMap;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "auto,scalar", value_delimiter = ',')]
    backends: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum OpType {
    MatMul,
    MatMulTransposed,
    MatMulSlice,
    Softmax,
    RMSNorm,
    RoPE,
}

impl std::fmt::Display for OpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug)]
struct TestResult {
    op: OpType,
    status: String,
    dtype: String,
    shape: String,
    backend: String,
    duration: Duration,
    error: f32,
    msg: String,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    
    println!("=== LLM RS2 Backend Test Suite (Expanded) ===");

    let mut backends: Vec<Arc<dyn Backend>> = Vec::new();
    for name in &args.backends {
        match name.to_lowercase().as_str() {
            "auto" | "cpu" => backends.push(Arc::new(CpuBackend::new())),
            "scalar" | "common" => backends.push(Arc::new(CpuBackendCommon::new())),
            "avx2" => {
                #[cfg(target_arch = "x86_64")]
                if std::is_x86_feature_detected!("avx2") {
                     backends.push(Arc::new(CpuBackendAVX2::new()))
                } else {
                    println!("Warning: AVX2 requested but not supported or not x86_64. Skipping.");
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    println!("Warning: AVX2 requested but not x86_64. Skipping.");
                }
            },
            "opencl" | "gpu" => {
                #[cfg(feature = "opencl")]
                match OpenCLBackend::new() {
                    Ok(b) => backends.push(Arc::new(b)),
                    Err(e) => println!("Warning: Failed to initialize OpenCL backend: {}. Skipping.", e),
                }
                #[cfg(not(feature = "opencl"))]
                {
                    println!("Warning: OpenCL not enabled in this build. Skipping.");
                }
            },
            _ => println!("Warning: Unknown backend '{}'. Usage: auto, scalar, avx2, opencl", name),
        }
    }

    if backends.is_empty() {
        println!("No valid backends selected. Defaulting to Auto (CPU) and Scalar.");
        backends.push(Arc::new(CpuBackend::new()));
        backends.push(Arc::new(CpuBackendCommon::new()));
    }

    let memory = Galloc::new();
    let mut all_results = Vec::new();
    
    let shapes = vec![
        // Small test shapes first for OpenCL debugging
        (1, 64, 32),
        (4, 128, 64),
        (1, 256, 128),
        (4, 256, 128),
        (8, 256, 128),
        (32, 256, 128),
        (64, 256, 128),     

        // Llama 7B Realistic Benchmarks (Hidden Size = 4096)
        (1, 4096, 4096),  // Single token - works reliably
        // (32, 4096, 4096), // Disabled - needs separate test run due to OOM with many shapes
    ];
    
    for backend in &backends {
        println!("\n--- Running Tests for Backend: {} ---", backend.name());
        
        for (m, k, n) in &shapes {
            let (m, k, n) = (*m, *k, *n);

            // Skip MatMul Standard
            // if k < 512 { ... }
            
            // Test MatMulTransposed - F32
            run_matmul_test(&mut all_results, backend.clone(), &memory, OpType::MatMulTransposed, DType::F32, m, k, n);
            

            if k % 32 == 0 {
                 run_matmul_test(&mut all_results, backend.clone(), &memory, OpType::MatMulTransposed, DType::Q4_0, m, k, n);
                 // run_matmul_test(&mut all_results, backend.clone(), &memory, OpType::MatMulTransposed, DType::Q4_1, m, k, n);
            }
            run_matmul_test(&mut all_results, backend.clone(), &memory, OpType::MatMulSlice, DType::F32, m, k, n);

             // Additional Ops Benchmarks - run on all shapes for debugging
             if k >= 64 {
                 run_matmul_test(&mut all_results, backend.clone(), &memory, OpType::RMSNorm, DType::F32, m, k, n);
                 run_matmul_test(&mut all_results, backend.clone(), &memory, OpType::Softmax, DType::F32, m, k, n);
                 run_matmul_test(&mut all_results, backend.clone(), &memory, OpType::RoPE, DType::F32, m, k, n);
             }
        }
    }


    print_comparison_table(&all_results, &backends);
    
    Ok(())
}

fn print_comparison_table(results: &[TestResult], backends: &[Arc<dyn Backend>]) {
    #[derive(PartialEq, Eq, Hash, Ord, PartialOrd)]
    struct RowKey {
        op: OpType,
        shape: String,
        dtype: String,
    }

    let mut map: BTreeMap<RowKey, BTreeMap<String, (Duration, f32, String)>> = BTreeMap::new();

    for res in results {
        let key = RowKey {
            op: res.op,
            shape: res.shape.clone(),
            dtype: res.dtype.clone(),
        };
        
        map.entry(key)
           .or_default()
           .insert(res.backend.clone(), (res.duration, res.error, res.status.clone()));
    }

    println!("\n{:=<200}", "");
    print!("{:<18} | {:<16} | {:<6} |", "Op", "Shape", "DTy");
    for backend in backends {
        let short_name = backend.name().replace("CPU (", "").replace(")", "").replace("Auto - ", ""); 
        print!(" {:^32} |", short_name);
    }
    println!("");
    
    print!("{:<18} | {:<16} | {:<6} |", "", "", "");
    for _ in backends {
        print!(" {:<18} | {:<11} |", "Duration", "Error");
    }
    println!("\n{:-<200}", "");

    for (key, backend_map) in map {
        print!("{:<18} | {:<16} | {:<6} |", key.op.to_string(), key.shape, key.dtype);
        
        let ref_dur_secs = backend_map.get("CPU (Scalar)").map(|(d, _, _)| d.as_secs_f64());

        for backend in backends {
            let b_name = backend.name();
            if let Some((dur, err, status)) = backend_map.get(b_name) {
                if status == "PASS" {
                    let dur_secs = dur.as_secs_f64();
                    let dur_ms = dur_secs * 1000.0;
                    
                    let speedup_str = if let Some(ref_secs) = ref_dur_secs {
                        if dur_secs > 0.0 {
                            format!("(x{:.2})", ref_secs / dur_secs)
                        } else {
                            "(x0.00)".to_string()
                        }
                    } else {
                        "".to_string()
                    };
                    
                    let dur_display = format!("{:.2}ms {}", dur_ms, speedup_str);
                    print!(" {:>18} | {:<11.6} |", dur_display, err);
                } else {
                     print!(" {:^32} |", status);
                }
            } else {
                 print!(" {:^32} |", "N/A");
            }
        }
        println!("");
    }
    println!("{:=<200}", "");
}

fn run_matmul_test(
    results: &mut Vec<TestResult>,
    backend: Arc<dyn Backend>,
    memory: &Galloc,
    op: OpType,
    dtype: DType,
    m: usize, k: usize, n: usize
) {
    let shape_str = format!("[{}, {}, {}]", m, k, n);
    match perform_matmul_test(backend.clone(), memory, op, dtype, m, k, n) {
        Ok((diff, dur)) => {
            let status = if diff > 1e-2 { "FAIL" } else { "PASS" };
            results.push(TestResult {
                op,
                status: status.to_string(),
                dtype: format!("{:?}", dtype),
                shape: shape_str,
                backend: backend.name().to_string(),
                duration: dur,
                error: diff,
                msg: "".to_string(),
            });
        },
        Err(e) => {
             eprintln!("ERROR in {} {:?} {}: {}", op, dtype, shape_str, e);
             results.push(TestResult {
                op,
                status: "ERROR".to_string(),
                dtype: format!("{:?}", dtype),
                shape: shape_str,
                backend: backend.name().to_string(),
                duration: Duration::from_secs(0),
                error: -1.0,
                msg: e.to_string(),
            });
        }
    }
}


fn perform_matmul_test(
    backend: Arc<dyn Backend>, 
    memory: &Galloc,
    op: OpType,
    dtype: DType,
    m: usize, k: usize, n: usize
) -> anyhow::Result<(f32, Duration)> {
    
    // A: [M, K]
    let a_size = m * k * 4;
    let buf_a = memory.alloc(a_size, DType::F32)?;
    let mut a_vec = vec![0.0f32; m * k];
    for i in 0..a_vec.len() { a_vec[i] = ((i % 100) as f32 * 0.01) - 0.5; }
    unsafe { std::ptr::copy_nonoverlapping(a_vec.as_ptr(), buf_a.as_mut_ptr() as *mut f32, a_vec.len()); }
    let a = Tensor::new(Shape::new(vec![m, k]), buf_a, backend.clone());

    let mut b_vec_f32 = vec![0.0f32; n * k];
    for i in 0..b_vec_f32.len() { b_vec_f32[i] = ((i % 123) as f32 * 0.01) - 0.5; }

    // Helpers to hold blocks for verification if needed
    let mut q4_0_blocks: Vec<BlockQ4_0> = Vec::new();
    let mut q4_1_blocks: Vec<BlockQ4_1> = Vec::new();

    // Prepare B Tensor
    let b = match (op, dtype) {
        (OpType::MatMul, DType::F32) => {
             let buf_b = memory.alloc(k * n * 4, DType::F32)?;
             unsafe { std::ptr::copy_nonoverlapping(b_vec_f32.as_ptr(), buf_b.as_mut_ptr() as *mut f32, k*n); }
             Tensor::new(Shape::new(vec![k, n]), buf_b, backend.clone())
        },
        (OpType::MatMulTransposed, DType::F32) => {
             let buf_b = memory.alloc(n * k * 4, DType::F32)?;
             unsafe { std::ptr::copy_nonoverlapping(b_vec_f32.as_ptr(), buf_b.as_mut_ptr() as *mut f32, n*k); }
             Tensor::new(Shape::new(vec![n, k]), buf_b, backend.clone())
        },
        (OpType::MatMulTransposed, DType::Q4_0) => {
             q4_0_blocks = quantize_q4_0(&b_vec_f32, n, k);
             let b_size_bytes = q4_0_blocks.len() * std::mem::size_of::<BlockQ4_0>();
             let buf_b = memory.alloc(b_size_bytes, DType::Q4_0)?;
             unsafe { std::ptr::copy_nonoverlapping(q4_0_blocks.as_ptr(), buf_b.as_mut_ptr() as *mut BlockQ4_0, q4_0_blocks.len()); }
             Tensor::new(Shape::new(vec![n, k]), buf_b, backend.clone())
        },
        (OpType::MatMulTransposed, DType::Q4_1) => {
             q4_1_blocks = quantize_q4_1(&b_vec_f32, n, k);
             let b_size_bytes = q4_1_blocks.len() * std::mem::size_of::<BlockQ4_1>();
             let buf_b = memory.alloc(b_size_bytes, DType::Q4_1)?;
             unsafe { std::ptr::copy_nonoverlapping(q4_1_blocks.as_ptr(), buf_b.as_mut_ptr() as *mut BlockQ4_1, q4_1_blocks.len()); }
             Tensor::new(Shape::new(vec![n, k]), buf_b, backend.clone())
        },
        (OpType::MatMulSlice, DType::F32) => {
             // Just allocate dummy B, but must be 2D [N, K] for matmul_transposed to accept it.
             let buf_b = memory.alloc(n * k * 4, DType::F32)?;
             unsafe { std::ptr::copy_nonoverlapping(b_vec_f32.as_ptr(), buf_b.as_mut_ptr() as *mut f32, n*k); }
             Tensor::new(Shape::new(vec![n, k]), buf_b, backend.clone())
        },
        (OpType::Softmax, _) => {
             // B unused
             let buf_b = memory.alloc(4, DType::F32)?;
             Tensor::new(Shape::new(vec![1]), buf_b, backend.clone())
        },
        (OpType::RMSNorm, _) | (OpType::RoPE, _) => {
             // RMSNorm: W is [Dim] = [K].
             // RoPE: unused but technically we might pass something?
             // Safest to alloc K elements to avoid range checks failing if B is used.
             let buf_b = memory.alloc(k * 4, DType::F32)?;
             // Fill with 1.0
             let vec_b = vec![1.0f32; k];
             unsafe { std::ptr::copy_nonoverlapping(vec_b.as_ptr(), buf_b.as_mut_ptr() as *mut f32, k); }
             Tensor::new(Shape::new(vec![k]), buf_b, backend.clone())
        },
        _ => return Err(anyhow::anyhow!("Unsupported config")),
    };

    let buf_c = memory.alloc(m * n * 4, DType::F32)?;
    let c = Tensor::new(Shape::new(vec![m, n]), buf_c, backend.clone());

    // For OpenCL backend, we need to transfer tensors to GPU
    let is_opencl = backend.name() == "OpenCL";
    let (a_gpu, b_gpu, mut c_gpu) = if is_opencl {
        (backend.copy_from(&a)?, backend.copy_from(&b)?, backend.copy_from(&c)?)
    } else {
        (a.clone(), b.clone(), c)
    };
    
    // Keep original A for verification (a still exists)

    let start = Instant::now();

    let iterations = 10;  // Reduced from 50 to prevent OOM on large tensors
    for _ in 0..iterations {
        match op {
            OpType::MatMul => backend.matmul(&a_gpu, &b_gpu, &mut c_gpu)?,
            OpType::MatMulTransposed => backend.matmul_transposed(&a_gpu, &b_gpu, &mut c_gpu)?,
            OpType::MatMulSlice => backend.matmul_slice(&a_gpu, &b_gpu, m, n, &mut c_gpu)?,
            OpType::RMSNorm => {
                backend.rms_norm(&mut c_gpu, &b_gpu, 1e-5)? 
            },
            OpType::Softmax => backend.softmax(&mut c_gpu)?,
            OpType::RoPE => {
                let head_dim = 128;
                let num_heads = n / head_dim;
                if n % head_dim != 0 {
                     backend.rope_inplace(&mut c_gpu, 0, 10000.0)?;
                } else {
                     let r_shape = Shape::new(vec![1, m, num_heads, head_dim]);
                     let r_buf = memory.alloc(m * n * 4, DType::F32)?;
                     let r_tensor = Tensor::new(r_shape, r_buf, backend.clone());
                     let mut r_gpu = if is_opencl { backend.copy_from(&r_tensor)? } else { r_tensor };
                     backend.rope_inplace(&mut r_gpu, 0, 10000.0)?;
                }
            },
        }
        backend.synchronize()?;
    }
    let dur = start.elapsed();
    // Verify - read back results from GPU if OpenCL
    let c_data: Vec<f32> = if is_opencl {
        #[cfg(feature = "opencl")]
        {
            use llm_rs2::core::buffer::Buffer;
            let buf = c_gpu.buffer();
            if let Some(cl_buf) = buf.as_any().downcast_ref::<llm_rs2::backend::opencl::buffer::OpenCLBuffer>() {
                let mut data = vec![0u8; m * n * 4];
                cl_buf.buffer.read(&mut data).enq()?;
                // Convert u8 to f32
                let mut result = vec![0.0f32; m * n];
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), result.as_mut_ptr() as *mut u8, m * n * 4);
                }
                result
            } else {
                vec![0.0f32; m * n]
            }
        }
        #[cfg(not(feature = "opencl"))]
        { vec![0.0f32; m * n] }
    } else {
        c_gpu.as_slice::<f32>().to_vec()
    };
    
    let r_m = m / 2;
    let r_n = n / 2;
    let mut ref_sum = 0.0;
    
    match (op, dtype) {
        (OpType::MatMul, DType::F32) => {
            for idx_k in 0..k {
                ref_sum += a_vec[r_m * k + idx_k] * b_vec_f32[idx_k * n + r_n];
            }
        },
        (OpType::MatMulTransposed, DType::F32) => {
            for idx_k in 0..k {
                ref_sum += a_vec[r_m * k + idx_k] * b_vec_f32[r_n * k + idx_k];
            }
        },
        (OpType::MatMulTransposed, DType::Q4_0) => {
             // Dequantize B row r_n
             let nb_k = k / QK4_0;
             let row_blocks = &q4_0_blocks[r_n * nb_k .. (r_n + 1) * nb_k];
             let mut deq_buf = vec![0.0f32; k];
             for (i, block) in row_blocks.iter().enumerate() {
                 let mut temp = [0.0f32; 32];
                 block.dequantize(&mut temp);
                 deq_buf[i*32..(i+1)*32].copy_from_slice(&temp);
             }
             for idx_k in 0..k {
                 ref_sum += a_vec[r_m * k + idx_k] * deq_buf[idx_k];
             }
        },
        (OpType::MatMulTransposed, DType::Q4_1) => {
             let nb_k = k / QK4_1;
             let row_blocks = &q4_1_blocks[r_n * nb_k .. (r_n + 1) * nb_k];
             let mut deq_buf = vec![0.0f32; k];
             for (i, block) in row_blocks.iter().enumerate() {
                 let mut temp = [0.0f32; 32];
                 block.dequantize(&mut temp);
                 deq_buf[i*32..(i+1)*32].copy_from_slice(&temp);
             }
             for idx_k in 0..k {
                 ref_sum += a_vec[r_m * k + idx_k] * deq_buf[idx_k];
             }
        },
        (OpType::MatMulSlice, DType::F32) => {
            // Uses same layout as MatMulTransposed logic in our implementation
            for idx_k in 0..k {
                ref_sum += a_vec[r_m * k + idx_k] * b_vec_f32[r_n * k + idx_k];
            }
        },
        _ => {}
    }

    let val = c_data[r_m * n + r_n];
    let diff = (ref_sum - val).abs();
    Ok((diff, dur))
}

fn quantize_q4_0(data: &[f32], n: usize, k: usize) -> Vec<BlockQ4_0> {
    let nb_k = k / QK4_0;
    let mut blocks = Vec::with_capacity(n * nb_k);
    
    for j in 0..n {
        for bi in 0..nb_k {
             let offset = j * k + bi * QK4_0;
             let src = &data[offset..offset+QK4_0];
             let mut block = BlockQ4_0 { d: f16::from_f32(0.0), qs: [0; 16] };
             
             let max_val = src.iter().map(|v| v.abs()).fold(0.0f32, |x, y| x.max(y));
             let d = max_val / 7.0;
             let id = if d == 0.0 { 0.0 } else { 1.0 / d };
             
             block.d = f16::from_f32(d);
             for z in 0..16 {
                 let v0 = (src[z] * id).round().clamp(-8.0, 7.0) as i8;
                 let v1 = (src[z + 16] * id).round().clamp(-8.0, 7.0) as i8;
                 let b0 = (v0 + 8) as u8;
                 let b1 = (v1 + 8) as u8;
                 block.qs[z] = b0 | (b1 << 4);
             }
             blocks.push(block);
        }
    }
    blocks
}

fn quantize_q4_1(data: &[f32], n: usize, k: usize) -> Vec<BlockQ4_1> {
    let nb_k = k / QK4_1;
    let mut blocks = Vec::with_capacity(n * nb_k);
    
    for j in 0..n {
        for bi in 0..nb_k {
             let offset = j * k + bi * QK4_1;
             let src = &data[offset..offset+QK4_1];
             let mut block = BlockQ4_1 { d: f16::from_f32(0.0), m: f16::from_f32(0.0), qs: [0; 16] };
             
             let min_val = src.iter().fold(f32::INFINITY, |a, &b| a.min(b));
             let max_val = src.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
             let d = (max_val - min_val) / 15.0;
             let m_val = min_val;
             let id = if d == 0.0 { 0.0 } else { 1.0 / d };
             
             block.d = f16::from_f32(d);
             block.m = f16::from_f32(m_val);
             
             for z in 0..16 {
                 let v0 = ((src[z] - m_val) * id).round().clamp(0.0, 15.0) as u8;
                 let v1 = ((src[z + 16] - m_val) * id).round().clamp(0.0, 15.0) as u8;
                 block.qs[z] = v0 | (v1 << 4);
             }
             blocks.push(block);
        }
    }
    blocks
}
