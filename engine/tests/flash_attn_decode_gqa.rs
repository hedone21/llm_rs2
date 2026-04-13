//! Host unit tests for the GQA-aware flash decode kernel
//! (`flash_attn_f32_f16_q1_gqa`).
//!
//! Verifies:
//!   1. Eligibility heuristic (pure Rust, always runs)
//!   2. Kernel correctness vs. CPU brute-force reference (GPU-gated)
//!   3. Self-consistency across repeated dispatches (GPU-gated)
//!
//! On hosts without an OpenCL driver or where the program fails to compile
//! (e.g. NVIDIA fallback paths), GPU-gated tests skip cleanly.

#![cfg(feature = "opencl")]

use llm_rs2::backend::opencl::OpenCLBackend;
use llm_rs2::backend::opencl::memory::OpenCLMemory;
use llm_rs2::core::backend::Backend;
use llm_rs2::core::buffer::DType;
use llm_rs2::core::memory::Memory;
use llm_rs2::core::shape::Shape;
use llm_rs2::core::tensor::Tensor;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Eligibility heuristic unit tests — always run, no GPU needed.
// ---------------------------------------------------------------------------

#[test]
fn gqa_eligible_llama_standard_decode() {
    // Llama 3.2: 32 Q heads, 8 KV heads, head_dim=64, long context.
    assert!(OpenCLBackend::gqa_decode_eligible_pub(32, 8, 64, 512));
    assert!(OpenCLBackend::gqa_decode_eligible_pub(32, 8, 64, 2048));
    // 4K context still eligible.
    assert!(OpenCLBackend::gqa_decode_eligible_pub(32, 8, 64, 4096));
}

#[test]
fn gqa_eligible_qwen_decode() {
    // Qwen 2.5-1.5B: 12 Q heads, 2 KV heads, head_dim=128.
    assert!(OpenCLBackend::gqa_decode_eligible_pub(12, 2, 128, 256));
    assert!(OpenCLBackend::gqa_decode_eligible_pub(12, 2, 128, 2048));
}

#[test]
fn gqa_rejects_mha() {
    // MHA (ratio = 1): no KV reuse, gqa kernel not selected.
    assert!(!OpenCLBackend::gqa_decode_eligible_pub(32, 32, 64, 2048));
    assert!(!OpenCLBackend::gqa_decode_eligible_pub(8, 8, 128, 512));
}

#[test]
fn gqa_rejects_short_ctx() {
    // Below GQA_MIN_CTX threshold (128), fall back to q1.
    assert!(!OpenCLBackend::gqa_decode_eligible_pub(32, 8, 64, 16));
    assert!(!OpenCLBackend::gqa_decode_eligible_pub(32, 8, 64, 127));
    // Boundary: exactly at threshold → eligible.
    assert!(OpenCLBackend::gqa_decode_eligible_pub(32, 8, 64, 128));
}

#[test]
fn gqa_rejects_unsupported_head_dim() {
    // head_dim != {64, 128} has no compiled GQA kernel.
    assert!(!OpenCLBackend::gqa_decode_eligible_pub(32, 8, 96, 2048));
    assert!(!OpenCLBackend::gqa_decode_eligible_pub(32, 8, 256, 2048));
}

#[test]
fn gqa_rejects_irregular_ratio() {
    // n_heads_q not divisible by n_heads_kv.
    assert!(!OpenCLBackend::gqa_decode_eligible_pub(7, 3, 64, 2048));
    // Ratio > GQA_RATIO_MAX (8).
    assert!(!OpenCLBackend::gqa_decode_eligible_pub(72, 8, 64, 2048));
    // Zero n_heads_kv guard.
    assert!(!OpenCLBackend::gqa_decode_eligible_pub(32, 0, 64, 2048));
}

// ---------------------------------------------------------------------------
// GPU correctness tests. Skipped gracefully when no GPU / kernel unavailable.
// ---------------------------------------------------------------------------

struct GqaShape {
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    cache_seq_len: usize,
    capacity: usize,
}

/// Reference attention on CPU: brute-force softmax over all positions,
/// per Q-head. Uses F32 throughout (match kernel's F32 accumulator).
fn cpu_attention_reference(
    q: &[f32], // [n_heads_q * head_dim]
    k: &[u16], // HeadMajor [n_heads_kv * capacity * head_dim] (F16 bits)
    v: &[u16], // HeadMajor [n_heads_kv * capacity * head_dim] (F16 bits)
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    cache_seq_len: usize,
    capacity: usize,
) -> Vec<f32> {
    let gqa_ratio = n_heads_q / n_heads_kv;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; n_heads_q * head_dim];

    for h in 0..n_heads_q {
        let hk = h / gqa_ratio;
        let q_base = h * head_dim;

        // Compute all scores.
        let mut scores = vec![f32::NEG_INFINITY; cache_seq_len];
        for p in 0..cache_seq_len {
            let k_base = hk * capacity * head_dim + p * head_dim;
            let mut s = 0.0f32;
            for d in 0..head_dim {
                let kf = half::f16::from_bits(k[k_base + d]).to_f32();
                s += q[q_base + d] * kf;
            }
            scores[p] = s * scale;
        }
        // Softmax.
        let m_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - m_max).exp();
            denom += *s;
        }
        // Weighted sum of V.
        let out_base = h * head_dim;
        if denom > 0.0 {
            let inv = 1.0 / denom;
            for p in 0..cache_seq_len {
                let w = scores[p] * inv;
                let v_base = hk * capacity * head_dim + p * head_dim;
                for d in 0..head_dim {
                    let vf = half::f16::from_bits(v[v_base + d]).to_f32();
                    out[out_base + d] += w * vf;
                }
            }
        }
    }
    out
}

fn upload_f32(
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    data: &[f32],
    shape: Vec<usize>,
) -> Tensor {
    let buf = memory.alloc(data.len() * 4, DType::F32).unwrap();
    let mut t = Tensor::new(Shape::new(shape), buf, backend.clone());
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    backend.write_buffer(&mut t, bytes).unwrap();
    t
}

fn upload_f16(
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    data: &[u16],
    shape: Vec<usize>,
) -> Tensor {
    let buf = memory.alloc(data.len() * 2, DType::F16).unwrap();
    let mut t = Tensor::new(Shape::new(shape), buf, backend.clone());
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
    backend.write_buffer(&mut t, bytes).unwrap();
    t
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

fn run_gqa_vs_cpu_case(shape: GqaShape, seed_scale: f32) {
    let backend = match OpenCLBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping: OpenCLBackend init failed: {e}");
            return;
        }
    };

    // Skip early if the GQA kernel isn't compiled (host NVIDIA fallback, etc.)
    if !backend.has_flash_decode_gqa_kernel(shape.head_dim) {
        eprintln!(
            "Skipping: GQA kernel not available for head_dim={} on this host",
            shape.head_dim
        );
        return;
    }

    let GqaShape {
        n_heads_q,
        n_heads_kv,
        head_dim,
        cache_seq_len,
        capacity,
    } = shape;

    let ocl_arc: Arc<dyn Backend> = Arc::new(backend);
    let ocl = ocl_arc
        .as_any()
        .downcast_ref::<OpenCLBackend>()
        .expect("OpenCLBackend downcast");

    let memory: Arc<dyn Memory> = Arc::new(OpenCLMemory::new(
        ocl.context.clone(),
        ocl.queue.clone(),
        true,
    ));

    // Deterministic inputs — distinct per (h, p, d) to catch head mixups.
    let q_data: Vec<f32> = (0..n_heads_q * head_dim)
        .map(|i| ((i as f32) * seed_scale * 0.0131).sin() * 0.3)
        .collect();

    let kv_total = n_heads_kv * capacity * head_dim;
    let mut k_data = vec![0u16; kv_total];
    let mut v_data = vec![0u16; kv_total];
    for h in 0..n_heads_kv {
        for p in 0..cache_seq_len {
            for d in 0..head_dim {
                let idx = h * capacity * head_dim + p * head_dim + d;
                let seed = (h as f32) * 31.7 + (p as f32) * 0.17 + (d as f32) * 0.29;
                k_data[idx] =
                    half::f16::from_f32((seed * seed_scale * 0.011).cos() * 0.4).to_bits();
                v_data[idx] =
                    half::f16::from_f32((seed * seed_scale * 0.013).sin() * 0.5).to_bits();
            }
        }
    }

    let q = upload_f32(&ocl_arc, &*memory, &q_data, vec![1, 1, n_heads_q, head_dim]);
    let k = upload_f16(
        &ocl_arc,
        &*memory,
        &k_data,
        vec![1, n_heads_kv, capacity, head_dim],
    );
    let v = upload_f16(
        &ocl_arc,
        &*memory,
        &v_data,
        vec![1, n_heads_kv, capacity, head_dim],
    );
    let zero = vec![0.0f32; n_heads_q * head_dim];
    let mut out = upload_f32(&ocl_arc, &*memory, &zero, vec![1, 1, n_heads_q, head_dim]);

    let ok = ocl
        .flash_attention_decode_gpu(
            &q,
            &k,
            &v,
            &mut out,
            n_heads_q,
            n_heads_kv,
            head_dim,
            cache_seq_len,
        )
        .expect("flash dispatch");
    if !ok {
        eprintln!("Skipping: flash_attention_decode_gpu returned false (kernel absent)");
        return;
    }
    ocl_arc.synchronize().unwrap();

    let out_len_bytes = n_heads_q * head_dim * std::mem::size_of::<f32>();
    let mut raw = vec![0u8; out_len_bytes];
    ocl_arc.read_buffer(&out, &mut raw).unwrap();
    let gpu_out: &[f32] =
        unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, n_heads_q * head_dim) };

    let cpu_out = cpu_attention_reference(
        &q_data,
        &k_data,
        &v_data,
        n_heads_q,
        n_heads_kv,
        head_dim,
        cache_seq_len,
        capacity,
    );

    let cos = cosine_similarity(gpu_out, &cpu_out);
    assert!(
        cos > 0.999,
        "GPU vs CPU cosine similarity too low (n_q={n_heads_q}, n_kv={n_heads_kv}, dk={head_dim}, \
         ctx={cache_seq_len}): cos={cos:.6}"
    );
    // Output must not be trivially all-zero.
    assert!(
        gpu_out.iter().any(|&v| v.abs() > 1e-5),
        "GPU output all zeros — likely a dispatch no-op"
    );
}

#[test]
fn gqa_llama_kv_sweep() {
    // Llama 3.2 shape: 32:8:64, GQA ratio = 4.
    for &ctx in &[128usize, 512, 1024, 2048] {
        run_gqa_vs_cpu_case(
            GqaShape {
                n_heads_q: 32,
                n_heads_kv: 8,
                head_dim: 64,
                cache_seq_len: ctx,
                capacity: ctx.max(2048),
            },
            1.0,
        );
    }
}

#[test]
fn gqa_llama_4k() {
    // Larger context isolated — cheaper to skip on slow hosts.
    run_gqa_vs_cpu_case(
        GqaShape {
            n_heads_q: 32,
            n_heads_kv: 8,
            head_dim: 64,
            cache_seq_len: 4096,
            capacity: 4096,
        },
        1.0,
    );
}

#[test]
fn gqa_qwen_kv_sweep() {
    // Qwen 2.5-1.5B shape: 12:2:128, GQA ratio = 6.
    for &ctx in &[128usize, 512, 1024, 2048] {
        run_gqa_vs_cpu_case(
            GqaShape {
                n_heads_q: 12,
                n_heads_kv: 2,
                head_dim: 128,
                cache_seq_len: ctx,
                capacity: ctx.max(2048),
            },
            0.9,
        );
    }
}

#[test]
fn gqa_self_consistent() {
    // Two identical dispatches must produce bit-identical output.
    let backend = match OpenCLBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping: OpenCLBackend init failed: {e}");
            return;
        }
    };
    if !backend.has_flash_decode_gqa_kernel(64) {
        eprintln!("Skipping: GQA DK=64 kernel unavailable");
        return;
    }

    let ocl_arc: Arc<dyn Backend> = Arc::new(backend);
    let ocl = ocl_arc
        .as_any()
        .downcast_ref::<OpenCLBackend>()
        .expect("downcast");

    let n_heads_q = 32usize;
    let n_heads_kv = 8usize;
    let head_dim = 64usize;
    let cache_seq_len = 512usize;
    let capacity = 1024usize;

    let memory: Arc<dyn Memory> = Arc::new(OpenCLMemory::new(
        ocl.context.clone(),
        ocl.queue.clone(),
        true,
    ));

    let q_data: Vec<f32> = (0..n_heads_q * head_dim)
        .map(|i| ((i as f32) * 0.02).sin())
        .collect();
    let kv_total = n_heads_kv * capacity * head_dim;
    let mut k_data = vec![0u16; kv_total];
    let mut v_data = vec![0u16; kv_total];
    for h in 0..n_heads_kv {
        for p in 0..cache_seq_len {
            for d in 0..head_dim {
                let idx = h * capacity * head_dim + p * head_dim + d;
                let seed = (h * 11 + p * 5 + d * 3) as f32;
                k_data[idx] = half::f16::from_f32((seed * 0.004).cos()).to_bits();
                v_data[idx] = half::f16::from_f32((seed * 0.006).sin()).to_bits();
            }
        }
    }

    let q = upload_f32(&ocl_arc, &*memory, &q_data, vec![1, 1, n_heads_q, head_dim]);
    let k = upload_f16(
        &ocl_arc,
        &*memory,
        &k_data,
        vec![1, n_heads_kv, capacity, head_dim],
    );
    let v = upload_f16(
        &ocl_arc,
        &*memory,
        &v_data,
        vec![1, n_heads_kv, capacity, head_dim],
    );
    let zero = vec![0.0f32; n_heads_q * head_dim];
    let mut out_a = upload_f32(&ocl_arc, &*memory, &zero, vec![1, 1, n_heads_q, head_dim]);
    let mut out_b = upload_f32(&ocl_arc, &*memory, &zero, vec![1, 1, n_heads_q, head_dim]);

    for out in [&mut out_a, &mut out_b] {
        let ok = ocl
            .flash_attention_decode_gpu(
                &q,
                &k,
                &v,
                out,
                n_heads_q,
                n_heads_kv,
                head_dim,
                cache_seq_len,
            )
            .expect("flash dispatch");
        assert!(ok, "GQA dispatch must succeed");
        ocl_arc.synchronize().unwrap();
    }

    let bytes = n_heads_q * head_dim * std::mem::size_of::<f32>();
    let mut ra = vec![0u8; bytes];
    let mut rb = vec![0u8; bytes];
    ocl_arc.read_buffer(&out_a, &mut ra).unwrap();
    ocl_arc.read_buffer(&out_b, &mut rb).unwrap();
    let a: &[f32] =
        unsafe { std::slice::from_raw_parts(ra.as_ptr() as *const f32, n_heads_q * head_dim) };
    let b: &[f32] =
        unsafe { std::slice::from_raw_parts(rb.as_ptr() as *const f32, n_heads_q * head_dim) };
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-6,
            "nondeterministic GQA output at [{i}]: {x} vs {y}"
        );
    }
}
