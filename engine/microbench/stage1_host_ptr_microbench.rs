//! Stage 1 — Direction A `CL_MEM_USE_HOST_PTR` microbench (kill-switch).
//!
//! Compares three weight-buffer construction paths for one AUF layer
//! (7 Q4_0 tensors: Q/K/V/O + FFN gate/up/down) and runs the same
//! `kernel_mul_mat_q4_0_f32` GEMV against each path's `cl_mem`.
//!
//! Paths:
//! 1. **`staging`** — `clCreateBuffer(MEM_READ_ONLY) + clEnqueueWriteBuffer`
//!    (current `copy_weight_from` flow). 16 ms staging on Galaxy S25.
//! 2. **`host_ptr_no_flush`** — `clCreateBuffer(USE_HOST_PTR | READ_ONLY)`
//!    against the AUF mmap pointer + offset. No cache flush. Behaviour is
//!    not guaranteed (UB possible) but we measure it for sanity.
//! 3. **`host_ptr_with_flush`** — same as (2) plus `clEnqueueMapBuffer(READ)`
//!    + `clEnqueueUnmapMemObject` to force a driver-side cache flush.
//!
//! Per-tensor we record alloc + write + flush wall-clock and run the
//! standard Q4_0 GEMV against each weight, reading the F32 output back to
//! host. Outputs from path 1/2/3 are byte-compared (the kernel is
//! deterministic, so a successful zero-copy path is bit-equal).
//!
//! Usage:
//! ```text
//! stage1_host_ptr_microbench --auf <path> [--layer 0] [--dump-mismatches N]
//! ```
//!
//! All measurements use `std::time::Instant` (wall-clock). Profiling events
//! are deliberately disabled — driver-specific patches (see
//! `feedback_opencl_profile_events_cross_engine.md`).

use anyhow::{Result, anyhow};
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the AUF file (CPU AOS variant required for the standard
    /// `kernel_mul_mat_q4_0_f32` AOS layout).
    #[arg(long)]
    auf: PathBuf,

    /// Layer index whose 7 Q4_0 weight tensors are loaded.
    #[arg(long, default_value_t = 0)]
    layer: u32,

    /// Print first N byte-mismatches per tensor (when path 2/3 differ from path 1).
    #[arg(long, default_value_t = 0)]
    dump_mismatches: usize,

    /// Backend tag for AUF open. `cpu` (default) — AOS layout matches the
    /// AOS Q4_0 GEMV kernel. `adreno` — SOA layout, which would not be
    /// readable by the standard kernel. Keep `cpu` for byte-equal compares.
    #[arg(long, default_value = "cpu")]
    tag: String,

    /// Comma-separated path subset: 0=staging, 1=host_ptr_no_flush,
    /// 2=host_ptr_with_flush. Default: all three. Useful to isolate a
    /// crashing path on driver-fragile devices (Adreno).
    #[arg(long, default_value = "0,1,2")]
    paths: String,
}

#[cfg(not(feature = "opencl"))]
fn main() -> Result<()> {
    anyhow::bail!("stage1_host_ptr_microbench requires the `opencl` feature")
}

// Shared Direction A helpers (no duplicate definitions; see plan ref below).
#[cfg(feature = "opencl")]
#[path = "../src/bin_helpers/stage_host_ptr_helpers.rs"]
mod helpers;

#[cfg(feature = "opencl")]
fn main() -> Result<()> {
    use helpers::{
        build_buffer_alloc_host_ptr_empty, build_buffer_staging, build_q4_0_kernel,
        fill_alloc_host_ptr_via_map, run_matmul_q4_0,
    };
    use llm_shared::auf::BackendTag;
    use llm_shared::auf::reader::open;
    use llm_shared::auf::tensor_index::{TensorDType, TensorKind};

    let args = Args::parse();

    let tag = match args.tag.to_lowercase().as_str() {
        "cpu" | "cpu_aos" => BackendTag::CpuAos,
        "adreno" | "adreno_soa" => BackendTag::AdrenoSoa,
        "cuda" | "cuda_aos" => BackendTag::CudaAos,
        other => anyhow::bail!("Unknown tag '{}': use cpu/adreno/cuda", other),
    };

    println!("=== Stage 1 microbench: HOST_PTR vs staging ===");
    println!("File: {}", args.auf.display());
    println!("Backend tag: {:?}", tag);
    println!("Layer: {}", args.layer);
    println!();

    let view = open(&args.auf, tag)?;
    let weights = view
        .weights_bytes()
        .ok_or_else(|| anyhow!("WEIGHTS section absent (BackendTag::Any?)"))?;
    let weights_start = view
        .weights_range
        .ok_or_else(|| anyhow!("weights_range missing"))?
        .0;
    println!(
        "WEIGHTS section: file_offset=0x{:x}, size={} bytes",
        weights_start,
        weights.len()
    );

    // mmap prefault — touch every page so the kernel's HOST_PTR path
    // doesn't pay the first-touch fault latency at measurement time.
    {
        let mut acc: u64 = 0;
        for chunk in weights.chunks(4096) {
            acc = acc.wrapping_add(chunk[0] as u64);
        }
        std::hint::black_box(acc);
    }

    // Collect 7 Q4_0 tensors for the requested layer.
    let kinds = [
        ("attn_q", TensorKind::AttnQ),
        ("attn_k", TensorKind::AttnK),
        ("attn_v", TensorKind::AttnV),
        ("attn_o", TensorKind::AttnO),
        ("ffn_gate", TensorKind::FfnGate),
        ("ffn_up", TensorKind::FfnUp),
        ("ffn_down", TensorKind::FfnDown),
    ];

    let variant_idx = view
        .tensor_index
        .variant_index_for_tag(tag.weights_section_tag().unwrap())
        .ok_or_else(|| anyhow!("variant tag not present in TENSOR_INDEX"))?;

    struct TensorInfo {
        name: &'static str,
        offset_in_section: u64,
        size: usize,
        // Q4_0 shape: [out_dim (n), in_dim (k)].
        n: usize,
        k: usize,
    }

    let mut tensors: Vec<TensorInfo> = Vec::new();
    for (name, kind) in &kinds {
        let entry = view.lookup_tensor(args.layer, kind.as_u32(), Some(TensorDType::Q4_0))?;
        if entry.shape.len() != 2 {
            anyhow::bail!("{}: expected 2D shape, got {:?}", name, entry.shape);
        }
        let n = entry.shape[0] as usize;
        let k = entry.shape[1] as usize;
        let voff = entry
            .variant_offsets
            .get(variant_idx)
            .copied()
            .unwrap_or(u64::MAX);
        let vsize = entry.variant_sizes.get(variant_idx).copied().unwrap_or(0);
        if voff == u64::MAX || vsize == 0 {
            anyhow::bail!("{}: variant payload missing for tag {:?}", name, tag);
        }
        tensors.push(TensorInfo {
            name,
            offset_in_section: voff,
            size: vsize as usize,
            n,
            k,
        });
    }

    for t in &tensors {
        println!(
            "  {:>8}: section_off=0x{:08x} size={:>10} bytes  shape=[n={}, k={}]",
            t.name, t.offset_in_section, t.size, t.n, t.k
        );
    }
    println!();

    let k_max = tensors.iter().map(|t| t.k).max().unwrap();
    let n_max = tensors.iter().map(|t| t.n).max().unwrap();

    // Build deterministic input activation.
    let mut input_f32: Vec<f32> = Vec::with_capacity(k_max);
    for i in 0..k_max {
        input_f32.push(((i as f32) * 0.001).sin() * 0.1);
    }

    let backend = llm_rs2::backend::opencl::OpenCLBackend::new()?;
    println!(
        "OpenCL device: {}",
        backend.device.name().unwrap_or_default()
    );
    println!("use_zero_copy = {}", backend.use_zero_copy);
    println!();

    // Build the standard Q4_0 GEMV kernel standalone (avoid touching the
    // backend's cached SOA registry / kernel cache state).
    let dbg = std::env::var("LLMRS_STAGE1_DEBUG").is_ok();
    macro_rules! dprint {
        ($($arg:tt)*) => {
            if dbg { eprintln!($($arg)*); }
        }
    }
    dprint!("[debug] building Q4_0 kernel");
    let kernel = build_q4_0_kernel(&backend)?;
    dprint!("[debug] Q4_0 kernel built");

    let weights_ptr_base = weights.as_ptr();
    dprint!("[debug] weights base ptr = {:p}", weights_ptr_base);

    // Persistent input cl_mem (one-time alloc, not measured).
    dprint!("[debug] alloc input_mem ({} f32)", k_max);
    let input_mem = unsafe {
        ocl::core::create_buffer::<_, f32>(
            backend.context.as_core(),
            ocl::core::MEM_READ_WRITE,
            k_max,
            None,
        )?
    };
    dprint!("[debug] write input_mem");
    unsafe {
        ocl::core::enqueue_write_buffer(
            &backend.queue,
            &input_mem,
            true,
            0,
            &input_f32,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    dprint!("[debug] input_mem ready");

    // Persistent output cl_mem.
    dprint!("[debug] alloc output_mem ({} f32)", n_max);
    let output_mem = unsafe {
        ocl::core::create_buffer::<_, f32>(
            backend.context.as_core(),
            ocl::core::MEM_READ_WRITE,
            n_max,
            None,
        )?
    };
    dprint!("[debug] output_mem ready, entering warmup");

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    enum Path {
        Staging,
        HostPtrNoFlush,
        HostPtrWithFlush,
        AllocHostPtrPerSwap,
        AllocHostPtrPool,
    }
    let all_paths = [
        Path::Staging,
        Path::HostPtrNoFlush,
        Path::HostPtrWithFlush,
        Path::AllocHostPtrPerSwap,
        Path::AllocHostPtrPool,
    ];
    let mut paths: Vec<Path> = Vec::new();
    for tok in args.paths.split(',') {
        match tok.trim() {
            "0" => paths.push(Path::Staging),
            "1" => paths.push(Path::HostPtrNoFlush),
            "2" => paths.push(Path::HostPtrWithFlush),
            "3" => paths.push(Path::AllocHostPtrPerSwap),
            "4" => paths.push(Path::AllocHostPtrPool),
            "" => {}
            other => anyhow::bail!("Unknown path token '{}': use 0/1/2/3/4", other),
        }
    }
    if paths.is_empty() {
        paths = all_paths.to_vec();
    }
    dprint!("[debug] selected paths: {:?}", paths);

    let mut all_outputs: Vec<Vec<Vec<f32>>> = vec![Vec::new(); paths.len()];
    let mut all_build_us: Vec<Vec<f64>> = vec![Vec::new(); paths.len()];
    let mut all_matmul_us: Vec<Vec<f64>> = vec![Vec::new(); paths.len()];

    // Warmup: run path 1 once (kernel program JIT, driver lazy init).
    {
        let t = &tensors[0];
        dprint!("[debug] warmup: build_buffer_staging({})", t.name);
        let weight_mem =
            build_buffer_staging(&backend, weights, t.offset_in_section as usize, t.size)?;
        dprint!("[debug] warmup: run_matmul_q4_0");
        run_matmul_q4_0(
            &backend,
            &kernel,
            &input_mem,
            &weight_mem,
            &output_mem,
            t.n,
            t.k,
        )?;
        dprint!("[debug] warmup: queue.finish");
        backend.queue.finish()?;
        dprint!("[debug] warmup: read_buffer");
        let mut tmp = vec![0f32; t.n];
        let dst_bytes =
            unsafe { std::slice::from_raw_parts_mut(tmp.as_mut_ptr() as *mut u8, t.n * 4) };
        unsafe {
            ocl::core::enqueue_read_buffer(
                &backend.queue,
                &output_mem,
                true,
                0,
                dst_bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        let _ = tmp;
        dprint!("[debug] warmup done");
    }

    for (pi, path) in paths.iter().enumerate() {
        dprint!("[debug] entering path {} / {:?}", pi, path);

        // Pre-allocate pool for AllocHostPtrPool path (alloc cost amortized).
        let pool_mems: Option<Vec<ocl::core::Mem>> = if *path == Path::AllocHostPtrPool {
            dprint!(
                "[debug] preallocating ALLOC_HOST_PTR pool ({} tensors)",
                tensors.len()
            );
            let mut v = Vec::with_capacity(tensors.len());
            for t in &tensors {
                v.push(build_buffer_alloc_host_ptr_empty(&backend, t.size)?);
            }
            backend.queue.finish()?;
            Some(v)
        } else {
            None
        };

        for (ti, t) in tensors.iter().enumerate() {
            dprint!("[debug]   tensor={} (path {:?})", t.name, path);
            // ── (a) Build cl_mem ──────────────────────────────────
            let t0 = Instant::now();
            let weight_mem = match path {
                Path::Staging => {
                    build_buffer_staging(&backend, weights, t.offset_in_section as usize, t.size)?
                }
                Path::HostPtrNoFlush => unsafe {
                    build_buffer_host_ptr(
                        &backend,
                        weights_ptr_base.add(t.offset_in_section as usize) as *mut std::ffi::c_void,
                        t.size,
                    )?
                },
                Path::HostPtrWithFlush => unsafe {
                    let mem = build_buffer_host_ptr(
                        &backend,
                        weights_ptr_base.add(t.offset_in_section as usize) as *mut std::ffi::c_void,
                        t.size,
                    )?;
                    flush_via_map_unmap(&backend, &mem, t.size)?;
                    mem
                },
                Path::AllocHostPtrPerSwap => {
                    // alloc + map + memcpy + unmap (each iteration)
                    let mem = build_buffer_alloc_host_ptr_empty(&backend, t.size)?;
                    let src_ptr = unsafe { weights_ptr_base.add(t.offset_in_section as usize) };
                    unsafe { fill_alloc_host_ptr_via_map(&backend, &mem, src_ptr, t.size)? };
                    mem
                }
                Path::AllocHostPtrPool => {
                    // Reuse pre-allocated cl_mem. Just map + memcpy + unmap.
                    let mem = pool_mems.as_ref().unwrap()[ti].clone();
                    let src_ptr = unsafe { weights_ptr_base.add(t.offset_in_section as usize) };
                    unsafe { fill_alloc_host_ptr_via_map(&backend, &mem, src_ptr, t.size)? };
                    mem
                }
            };
            backend.queue.finish()?;
            let build_us = t0.elapsed().as_secs_f64() * 1e6;

            // ── (b) Matmul + read back ────────────────────────────
            let t1 = Instant::now();
            run_matmul_q4_0(
                &backend,
                &kernel,
                &input_mem,
                &weight_mem,
                &output_mem,
                t.n,
                t.k,
            )?;
            backend.queue.finish()?;
            let mut out_vec = vec![0f32; t.n];
            let dst_bytes =
                unsafe { std::slice::from_raw_parts_mut(out_vec.as_mut_ptr() as *mut u8, t.n * 4) };
            unsafe {
                ocl::core::enqueue_read_buffer(
                    &backend.queue,
                    &output_mem,
                    true,
                    0,
                    dst_bytes,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
            let matmul_us = t1.elapsed().as_secs_f64() * 1e6;

            all_outputs[pi].push(out_vec);
            all_build_us[pi].push(build_us);
            all_matmul_us[pi].push(matmul_us);
            // weight_mem dropped → cl_mem released.
        }
    }

    // ── Report ────────────────────────────────────────────────────────
    let path_name = |p: Path| match p {
        Path::Staging => "staging",
        Path::HostPtrNoFlush => "host_ptr_no_flush",
        Path::HostPtrWithFlush => "host_ptr_with_flush",
        Path::AllocHostPtrPerSwap => "alloc_host_ptr_per_swap",
        Path::AllocHostPtrPool => "alloc_host_ptr_pool",
    };
    for (pi, path) in paths.iter().enumerate() {
        println!("[{}]", path_name(*path));
        let builds = &all_build_us[pi];
        let mms = &all_matmul_us[pi];
        let build_str: Vec<String> = builds
            .iter()
            .map(|v| format!("{:.2}", v / 1000.0))
            .collect();
        let mm_str: Vec<String> = mms.iter().map(|v| format!("{:.3}", v / 1000.0)).collect();
        let build_mean = builds.iter().sum::<f64>() / builds.len() as f64;
        let mm_mean = mms.iter().sum::<f64>() / mms.len() as f64;
        let build_total: f64 = builds.iter().sum();
        let mm_total: f64 = mms.iter().sum();
        println!(
            "  per-tensor build (ms):  [{}] mean={:.2} ms",
            build_str.join(" / "),
            build_mean / 1000.0
        );
        println!(
            "  per-tensor matmul (ms): [{}] mean={:.3} ms",
            mm_str.join(" / "),
            mm_mean / 1000.0
        );
        println!(
            "  total: {:.2} ms (build {:.2} + matmul {:.2})",
            (build_total + mm_total) / 1000.0,
            build_total / 1000.0,
            mm_total / 1000.0
        );

        let baseline_idx = paths.iter().position(|p| *p == Path::Staging);
        if let Some(bidx) = baseline_idx
            && pi != bidx
        {
            let mut max_diff = 0.0f32;
            let mut total_mismatch = 0usize;
            let mut bit_equal_total = true;
            for (ti, t) in tensors.iter().enumerate() {
                let baseline = &all_outputs[bidx][ti];
                let candidate = &all_outputs[pi][ti];
                if baseline.len() != candidate.len() {
                    bit_equal_total = false;
                    total_mismatch += baseline.len();
                    continue;
                }
                let mut local_mismatch = 0usize;
                for (idx, (&a, &b)) in baseline.iter().zip(candidate.iter()).enumerate() {
                    let diff = (a - b).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    if a.to_bits() != b.to_bits() {
                        local_mismatch += 1;
                        if args.dump_mismatches > 0 && local_mismatch <= args.dump_mismatches {
                            eprintln!(
                                "  [{}][{}] idx={:>5} baseline={:.6e} candidate={:.6e} diff={:.3e}",
                                path_name(*path),
                                t.name,
                                idx,
                                a,
                                b,
                                diff
                            );
                        }
                    }
                }
                if local_mismatch != 0 {
                    bit_equal_total = false;
                }
                total_mismatch += local_mismatch;
            }
            println!(
                "  byte-equal vs staging: {} (max diff: {:.3e}, mismatches: {} / {})",
                if bit_equal_total { "PASS" } else { "FAIL" },
                max_diff,
                total_mismatch,
                tensors.iter().map(|t| t.n).sum::<usize>()
            );
        }
        println!();
    }

    // ── Verdict ───────────────────────────────────────────────────────
    let staging_idx = paths.iter().position(|p| *p == Path::Staging);
    let with_flush_idx = paths.iter().position(|p| *p == Path::HostPtrWithFlush);
    let no_flush_idx = paths.iter().position(|p| *p == Path::HostPtrNoFlush);
    let alloc_per_swap_idx = paths.iter().position(|p| *p == Path::AllocHostPtrPerSwap);
    let alloc_pool_idx = paths.iter().position(|p| *p == Path::AllocHostPtrPool);

    let path_total = |idx: usize| -> f64 {
        (all_build_us[idx].iter().sum::<f64>() + all_matmul_us[idx].iter().sum::<f64>()) / 1000.0
    };

    println!("=== Verdict ===");
    if let Some(s) = staging_idx {
        println!("- staging total          : {:.2} ms", path_total(s));
    }
    if let Some(n) = no_flush_idx {
        let total = path_total(n);
        if let Some(s) = staging_idx {
            let saving = 100.0 * (1.0 - total / path_total(s));
            println!(
                "- host_ptr_no_flush      : {:.2} ms ({:+.1}% vs staging)",
                total, -saving
            );
        } else {
            println!(
                "- host_ptr_no_flush      : {:.2} ms (no staging baseline)",
                total
            );
        }
    }
    if let Some(w) = with_flush_idx {
        let total = path_total(w);
        if let Some(s) = staging_idx {
            let saving = 100.0 * (1.0 - total / path_total(s));
            println!(
                "- host_ptr_with_flush    : {:.2} ms ({:+.1}% vs staging)",
                total, -saving
            );
        } else {
            println!(
                "- host_ptr_with_flush    : {:.2} ms (no staging baseline)",
                total
            );
        }
    }
    if let Some(a) = alloc_per_swap_idx {
        let total = path_total(a);
        if let Some(s) = staging_idx {
            let saving = 100.0 * (1.0 - total / path_total(s));
            println!(
                "- alloc_host_ptr_per_swap: {:.2} ms ({:+.1}% vs staging)",
                total, -saving
            );
        } else {
            println!(
                "- alloc_host_ptr_per_swap: {:.2} ms (no staging baseline)",
                total
            );
        }
    }
    if let Some(a) = alloc_pool_idx {
        let total = path_total(a);
        if let Some(s) = staging_idx {
            let saving = 100.0 * (1.0 - total / path_total(s));
            println!(
                "- alloc_host_ptr_pool    : {:.2} ms ({:+.1}% vs staging)",
                total, -saving
            );
        } else {
            println!(
                "- alloc_host_ptr_pool    : {:.2} ms (no staging baseline)",
                total
            );
        }
    }

    let bit_equal = |idx_a: Option<usize>, idx_b: Option<usize>| -> bool {
        let (Some(a_idx), Some(b_idx)) = (idx_a, idx_b) else {
            return false;
        };
        for (a, b) in all_outputs[a_idx]
            .iter()
            .zip(all_outputs[b_idx].iter())
            .take(tensors.len())
        {
            if a.len() != b.len() {
                return false;
            }
            for (&x, &y) in a.iter().zip(b.iter()) {
                if x.to_bits() != y.to_bits() {
                    return false;
                }
            }
        }
        true
    };

    let bit_equal_no_flush = bit_equal(staging_idx, no_flush_idx);
    let bit_equal_with_flush = bit_equal(staging_idx, with_flush_idx);
    let bit_equal_alloc_per_swap = bit_equal(staging_idx, alloc_per_swap_idx);
    let bit_equal_alloc_pool = bit_equal(staging_idx, alloc_pool_idx);
    if no_flush_idx.is_some() {
        println!(
            "- byte-equal (no_flush)  : {}",
            if bit_equal_no_flush {
                "PASS"
            } else {
                "FAIL/SKIP"
            }
        );
    }
    if with_flush_idx.is_some() {
        println!(
            "- byte-equal (with_flush): {}",
            if bit_equal_with_flush {
                "PASS"
            } else {
                "FAIL/SKIP"
            }
        );
    }
    if alloc_per_swap_idx.is_some() {
        println!(
            "- byte-equal (alloc_per_swap): {}",
            if bit_equal_alloc_per_swap {
                "PASS"
            } else {
                "FAIL/SKIP"
            }
        );
    }
    if alloc_pool_idx.is_some() {
        println!(
            "- byte-equal (alloc_pool): {}",
            if bit_equal_alloc_pool {
                "PASS"
            } else {
                "FAIL/SKIP"
            }
        );
    }

    let stage2_recommended = match (staging_idx, alloc_pool_idx, with_flush_idx) {
        (Some(s), Some(a), _) => bit_equal_alloc_pool && path_total(a) < 5.0 && path_total(s) > 1.0,
        (Some(s), None, Some(w)) => {
            bit_equal_with_flush && path_total(w) < 5.0 && path_total(s) > 1.0
        }
        _ => false,
    };
    println!(
        "- Stage 2 entry recommended: {}",
        if stage2_recommended { "YES" } else { "NO" }
    );

    Ok(())
}

// ─── stage1-only helpers (cfg-gated to opencl feature) ───────────────────
// Shared helpers (build_q4_0_kernel / build_buffer_staging /
// build_buffer_alloc_host_ptr_empty / fill_alloc_host_ptr_via_map /
// run_matmul_q4_0) are imported from `bin_helpers/stage_host_ptr_helpers.rs`
// at the top of this file. Below are stage1-only paths (USE_HOST_PTR + the
// MAP_READ flush pattern).

/// `clCreateBuffer(USE_HOST_PTR | READ_ONLY)` against the AUF mmap pointer.
///
/// # Safety
/// Caller must ensure `host_ptr` is alive (mmap not unmapped) for the
/// lifetime of the returned `Mem`. `size` bytes starting at `host_ptr` must
/// be readable.
#[cfg(feature = "opencl")]
unsafe fn build_buffer_host_ptr(
    backend: &llm_rs2::backend::opencl::OpenCLBackend,
    host_ptr: *mut std::ffi::c_void,
    size: usize,
) -> Result<ocl::core::Mem> {
    use ocl::core::ClContextPtr;

    const CL_MEM_READ_ONLY: u64 = 1 << 2;
    const CL_MEM_USE_HOST_PTR: u64 = 1 << 3;

    let flags = (CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR) as ocl::ffi::cl_bitfield;
    let ctx_ptr = <&ocl::Context as ClContextPtr>::as_ptr(&&backend.context);
    let mut errcode: ocl::ffi::cl_int = 0;
    let raw = unsafe { ocl::ffi::clCreateBuffer(ctx_ptr, flags, size, host_ptr, &mut errcode) };
    if errcode != 0 || raw.is_null() {
        anyhow::bail!("clCreateBuffer(USE_HOST_PTR) failed: errcode={}", errcode);
    }
    let mem = unsafe { ocl::core::Mem::from_raw_create_ptr(raw) };
    Ok(mem)
}

/// Force a driver-side cache flush via `clEnqueueMapBuffer(READ)` +
/// `clEnqueueUnmapMemObject`. Canonical OpenCL pattern when host-side
/// data was written outside the runtime's view.
#[cfg(feature = "opencl")]
fn flush_via_map_unmap(
    backend: &llm_rs2::backend::opencl::OpenCLBackend,
    mem: &ocl::core::Mem,
    size: usize,
) -> Result<()> {
    use ocl::ffi;
    const CL_TRUE: ffi::cl_bool = 1;
    const CL_MAP_READ: ffi::cl_map_flags = 1;

    let q_ref: &ocl::core::CommandQueue = &backend.queue;
    let q_ptr: ffi::cl_command_queue = q_ref.as_ptr();

    let mut errcode: ffi::cl_int = 0;
    let mapped_ptr = unsafe {
        ffi::clEnqueueMapBuffer(
            q_ptr,
            mem.as_ptr(),
            CL_TRUE,
            CL_MAP_READ,
            0,
            size,
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
            &mut errcode,
        )
    };
    if errcode != 0 || mapped_ptr.is_null() {
        anyhow::bail!(
            "clEnqueueMapBuffer failed: errcode={} ptr_null={}",
            errcode,
            mapped_ptr.is_null()
        );
    }

    let rc = unsafe {
        ffi::clEnqueueUnmapMemObject(
            q_ptr,
            mem.as_ptr(),
            mapped_ptr,
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };
    if rc != 0 {
        anyhow::bail!("clEnqueueUnmapMemObject failed: rc={}", rc);
    }
    backend.queue.finish()?;
    Ok(())
}
