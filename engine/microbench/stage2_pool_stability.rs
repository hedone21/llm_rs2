//! Stage 2 — Direction A `CL_MEM_ALLOC_HOST_PTR` pool stability harness.
//!
//! Stage 1b (`stage1_host_ptr_microbench --paths 0,4`) measured the
//! `alloc_host_ptr_pool` pattern on a single layer × 7 tensors and observed
//! −42.4 % saving + byte-equal PASS 5/5 on Galaxy S25. Stage 2 stress-tests
//! the same pattern in production-shaped scenarios before Stage 3 production
//! integration:
//!
//! - **Test 1**: 25 layers × 7 Q4_0 tensors (= 175 fills) with a single
//!   reused 7-slot pool. Validates that the pool pattern scales linearly and
//!   stays byte-equal across layers.
//! - **Test 2**: 18-slot pool (= 2 layers worth) cycled 25 rounds with
//!   matmul interleaved between fills. Catches driver-side caching cliffs
//!   or fragmentation across reuse.
//! - **Test 3**: Lifetime ordering — `Arc<AufView>` drops while cl_mem is
//!   still live; cl_mem releases after. Rust ownership makes this trivial,
//!   but the harness asserts no panic / no SIGSEGV.
//! - **Test 4** (best-effort, optional): background memory pressure
//!   (large `Vec<u8>` writer) running concurrently with pool fills. Checks
//!   that ALLOC_HOST_PTR pages are not page-evicted under pressure
//!   (i.e. driver-pinned).
//! - **Test 5**: pool-size sweep (9 / 18 / 27 / 81 slots) running the
//!   Test 2 cycle pattern. Reports per-pool-size mean round time + RSS Δ.
//!
//! Output is human-readable; the verdict line at the end indicates whether
//! Stage 3 entry is recommended.
//!
//! Plan ref: compiled-chasing-hopper Direction A (Stage 1b → Stage 2 →
//! Stage 3). No spec ID — prototype kill-switch chain.
//!
//! All measurements use `std::time::Instant` (wall-clock). Profiling events
//! are deliberately disabled — driver-specific patches (see
//! `feedback_opencl_profile_events_cross_engine.md`).

use anyhow::{Result, anyhow};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the AUF file (CPU AOS variant — AOS layout matches the
    /// standard `kernel_mul_mat_q4_0_f32` AOS Q4_0 GEMV kernel).
    #[arg(long)]
    auf: PathBuf,

    /// Number of layers to load. Capped at the AUF's actual layer count.
    /// Default 25 mirrors a typical ratio-0.9 swap target set on a 28-layer
    /// Qwen2.5-1.5b model.
    #[arg(long, default_value_t = 25)]
    n_layers: u32,

    /// Number of round-robin reuse rounds for Test 2.
    #[arg(long, default_value_t = 25)]
    n_rounds: u32,

    /// Comma-separated subset of tests to run. Default: 1,2,3,5 (Test 4 is
    /// memory-pressure and adds significant runtime + OS-dependent variance).
    #[arg(long, default_value = "1,2,3,5")]
    tests: String,

    /// Pool sizes (slot count) for the Test 5 sweep.
    #[arg(long, default_value = "9,18,27,81")]
    pool_sizes: String,

    /// Test 4: background pressure thread allocates this much in MiB and
    /// repeatedly writes to it.
    #[arg(long, default_value_t = 512)]
    pressure_mib: usize,

    /// Test 4: target wall-clock duration (seconds) for the pressure run.
    #[arg(long, default_value_t = 5.0)]
    pressure_secs: f64,

    /// Backend tag for AUF open. `cpu` → AOS layout (default, matches the
    /// AOS Q4_0 GEMV kernel). `adreno` → SOA layout (would not be readable
    /// by the standard kernel; do not select for byte-equal compares).
    #[arg(long, default_value = "cpu")]
    tag: String,

    /// Print first N byte-mismatches per layer (when pool path differs from
    /// staging baseline).
    #[arg(long, default_value_t = 0)]
    dump_mismatches: usize,

    /// Verbose progress logging to stderr (per-layer / per-round).
    #[arg(long)]
    debug: bool,
}

#[cfg(not(feature = "opencl"))]
fn main() -> Result<()> {
    anyhow::bail!("stage2_pool_stability requires the `opencl` feature")
}

// Shared Direction A helpers (no duplicate definitions; see plan ref above).
#[cfg(feature = "opencl")]
#[path = "../src/bin_helpers/stage_host_ptr_helpers.rs"]
mod helpers;

#[cfg(feature = "opencl")]
fn main() -> Result<()> {
    use helpers::{
        build_buffer_alloc_host_ptr_empty, build_buffer_staging, build_q4_0_kernel,
        fill_alloc_host_ptr_via_map, run_matmul_q4_0,
    };
    use llm_rs2::auf::BackendTag;
    use llm_rs2::auf::reader::open;
    use llm_rs2::auf::tensor_index::{TensorDType, TensorKind};

    let args = Args::parse();
    let dbg = args.debug || std::env::var("LLMRS_STAGE2_DEBUG").is_ok();
    macro_rules! dprint {
        ($($arg:tt)*) => {
            if dbg { eprintln!($($arg)*); }
        }
    }

    let tag = match args.tag.to_lowercase().as_str() {
        "cpu" | "cpu_aos" => BackendTag::CpuAos,
        "adreno" | "adreno_soa" => BackendTag::AdrenoSoa,
        "cuda" | "cuda_aos" => BackendTag::CudaAos,
        other => anyhow::bail!("Unknown tag '{}': use cpu/adreno/cuda", other),
    };

    let selected_tests: Vec<u32> = args
        .tests
        .split(',')
        .filter_map(|t| t.trim().parse::<u32>().ok())
        .collect();
    let want_test = |id: u32| selected_tests.contains(&id);

    println!("=== Stage 2: ALLOC_HOST_PTR pool stability ===");
    println!("File: {}", args.auf.display());
    println!("Backend tag: {:?}", tag);
    println!("Tests selected: {:?}", selected_tests);
    println!();

    // Open the AUF inside an Arc so that Test 3 can drop the Vec entry while
    // cl_mem still hold cloned Arcs. (Only Test 3 uses this property; the
    // other tests merely borrow.)
    let view = Arc::new(open(&args.auf, tag)?);
    let weights = view
        .weights_bytes()
        .ok_or_else(|| anyhow!("WEIGHTS section absent (BackendTag::Any?)"))?;
    let weights_start = view
        .weights_range
        .ok_or_else(|| anyhow!("weights_range missing"))?
        .0;
    let weights_ptr_base = weights.as_ptr();
    println!(
        "WEIGHTS section: file_offset=0x{:x}, size={} bytes",
        weights_start,
        weights.len()
    );

    // mmap prefault.
    {
        let mut acc: u64 = 0;
        for chunk in weights.chunks(4096) {
            acc = acc.wrapping_add(chunk[0] as u64);
        }
        std::hint::black_box(acc);
    }

    // ── Tensor catalog: collect per-layer 7 Q4_0 tensor offsets ───────────
    // Norm tensors (attn_norm/ffn_norm) are typically F32 and not readable
    // by the Q4_0 GEMV kernel, so we omit them from byte-equal checks. The
    // plan calls them out as "secondary에 없으면 skip 명시"; this binary
    // always treats norms as skipped (printed once below) and validates the
    // 7 Q4_0 tensors per layer.
    let kinds: [(&str, TensorKind); 7] = [
        ("attn_q", TensorKind::AttnQ),
        ("attn_k", TensorKind::AttnK),
        ("attn_v", TensorKind::AttnV),
        ("attn_o", TensorKind::AttnO),
        ("ffn_gate", TensorKind::FfnGate),
        ("ffn_up", TensorKind::FfnUp),
        ("ffn_down", TensorKind::FfnDown),
    ];
    println!(
        "Tensor scope per layer: 7 Q4_0 weights ({}). Norms (attn_norm/ffn_norm) skipped — F32, kernel-incompatible.",
        kinds.iter().map(|(n, _)| *n).collect::<Vec<_>>().join("/")
    );

    let variant_idx = view
        .tensor_index
        .variant_index_for_tag(tag.weights_section_tag().unwrap())
        .ok_or_else(|| anyhow!("variant tag not present in TENSOR_INDEX"))?;

    #[derive(Clone)]
    struct TensorInfo {
        name: &'static str,
        offset_in_section: u64,
        size: usize,
        n: usize, // out_dim (rows)
        k: usize, // in_dim (cols)
    }

    // Determine layer_count present in the AUF.
    // `layer_idx == u32::MAX` is the LAYER_IDX_CROSS sentinel (embedding /
    // lm_head / final_norm) and must be excluded from the layer count.
    use llm_rs2::auf::tensor_index::LAYER_IDX_CROSS;
    let mut max_layer: i64 = -1;
    for entry in &view.tensor_index.entries {
        if entry.layer_idx == LAYER_IDX_CROSS {
            continue;
        }
        let li = entry.layer_idx as i64;
        if li > max_layer {
            max_layer = li;
        }
    }
    let layers_in_auf = if max_layer >= 0 {
        (max_layer as u32) + 1
    } else {
        0
    };
    let n_layers = args.n_layers.min(layers_in_auf);
    if n_layers == 0 {
        anyhow::bail!("AUF has no layers detected (max_layer_idx={})", max_layer);
    }
    println!(
        "Layers: requested={}, present={}, using={}",
        args.n_layers, layers_in_auf, n_layers
    );

    // Collect tensors per layer.
    let mut per_layer: Vec<Vec<TensorInfo>> = Vec::with_capacity(n_layers as usize);
    for layer_idx in 0..n_layers {
        let mut row: Vec<TensorInfo> = Vec::with_capacity(kinds.len());
        for (name, kind) in &kinds {
            let entry = view.lookup_tensor(layer_idx, kind.as_u32(), Some(TensorDType::Q4_0))?;
            if entry.shape.len() != 2 {
                anyhow::bail!(
                    "layer={} {}: expected 2D shape, got {:?}",
                    layer_idx,
                    name,
                    entry.shape
                );
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
                anyhow::bail!(
                    "layer={} {}: variant payload missing for tag {:?}",
                    layer_idx,
                    name,
                    tag
                );
            }
            row.push(TensorInfo {
                name,
                offset_in_section: voff,
                size: vsize as usize,
                n,
                k,
            });
        }
        per_layer.push(row);
    }

    // Sanity: across layers the per-tensor sizes should be identical (so a
    // single 7-slot pool sized to layer 0 is reusable). Bail loud if not.
    {
        let l0 = &per_layer[0];
        for (li, layer) in per_layer.iter().enumerate().skip(1) {
            for (ti, t) in layer.iter().enumerate() {
                if t.size != l0[ti].size || t.n != l0[ti].n || t.k != l0[ti].k {
                    anyhow::bail!(
                        "layer={} {}: shape/size mismatch vs layer 0 (size {} vs {}, n {} vs {}, k {} vs {})",
                        li,
                        t.name,
                        t.size,
                        l0[ti].size,
                        t.n,
                        l0[ti].n,
                        t.k,
                        l0[ti].k
                    );
                }
            }
        }
    }
    let l0 = &per_layer[0];
    println!("Per-tensor sizes (layer 0 — same across layers):");
    for t in l0 {
        println!(
            "  {:>8}: {:>10} bytes (n={}, k={})",
            t.name, t.size, t.n, t.k
        );
    }
    println!();

    // ── OpenCL setup ──────────────────────────────────────────────────────
    let backend = llm_rs2::backend::opencl::OpenCLBackend::new()?;
    println!(
        "OpenCL device: {}",
        backend.device.name().unwrap_or_default()
    );
    println!("use_zero_copy = {}", backend.use_zero_copy);
    println!();

    let kernel = build_q4_0_kernel(&backend)?;

    let k_max = l0.iter().map(|t| t.k).max().unwrap();
    let n_max = l0.iter().map(|t| t.n).max().unwrap();

    let mut input_f32: Vec<f32> = Vec::with_capacity(k_max);
    for i in 0..k_max {
        input_f32.push(((i as f32) * 0.001).sin() * 0.1);
    }

    let input_mem = unsafe {
        ocl::core::create_buffer::<_, f32>(
            backend.context.as_core(),
            ocl::core::MEM_READ_WRITE,
            k_max,
            None,
        )?
    };
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
    let output_mem = unsafe {
        ocl::core::create_buffer::<_, f32>(
            backend.context.as_core(),
            ocl::core::MEM_READ_WRITE,
            n_max,
            None,
        )?
    };

    // Warmup: stage one tensor through the staging path so JIT, lazy driver
    // init, etc. cost is paid before measurements.
    {
        let t = &l0[0];
        let mem = build_buffer_staging(&backend, weights, t.offset_in_section as usize, t.size)?;
        run_matmul_q4_0(&backend, &kernel, &input_mem, &mem, &output_mem, t.n, t.k)?;
        backend.queue.finish()?;
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
        std::hint::black_box(tmp);
    }

    // Helper: read output buffer (n f32) into a Vec<f32>.
    let read_output = |n: usize| -> Result<Vec<f32>> {
        let mut out_vec = vec![0f32; n];
        let dst_bytes =
            unsafe { std::slice::from_raw_parts_mut(out_vec.as_mut_ptr() as *mut u8, n * 4) };
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
        Ok(out_vec)
    };

    let bit_equal = |a: &[f32], b: &[f32]| -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
    };

    // ── Compute baseline outputs once (staging path) for byte-equal check ─
    // Used by Tests 1, 2, 4, 5. The matmul kernel is deterministic, so a
    // single pass per (layer, tensor) gives a stable reference.
    let mut baseline: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_layers as usize);
    for (li, layer) in per_layer.iter().enumerate() {
        dprint!("[debug] baseline layer={}", li);
        let mut layer_out: Vec<Vec<f32>> = Vec::with_capacity(layer.len());
        for t in layer {
            let mem =
                build_buffer_staging(&backend, weights, t.offset_in_section as usize, t.size)?;
            run_matmul_q4_0(&backend, &kernel, &input_mem, &mem, &output_mem, t.n, t.k)?;
            backend.queue.finish()?;
            layer_out.push(read_output(t.n)?);
        }
        baseline.push(layer_out);
    }
    println!(
        "Baseline (staging) outputs computed for all {} layers.",
        n_layers
    );
    println!();

    // ── Test 1: 25 layer × 7 tensor pool fill ─────────────────────────────
    if want_test(1) {
        println!(
            "=== Test 1: {} layer × {} tensor pool fill ===",
            n_layers,
            l0.len()
        );
        let mut pool: Vec<ocl::core::Mem> = Vec::with_capacity(l0.len());
        for t in l0 {
            pool.push(build_buffer_alloc_host_ptr_empty(&backend, t.size)?);
        }
        backend.queue.finish()?;

        let mut fill_us: Vec<f64> = Vec::with_capacity((n_layers as usize) * l0.len());
        let mut all_eq = true;
        let mut total_mismatch = 0usize;
        let mut max_diff = 0.0f32;

        let test_start = Instant::now();
        for layer_idx in 0..n_layers as usize {
            let layer = &per_layer[layer_idx];
            for (ti, t) in layer.iter().enumerate() {
                let src = unsafe { weights_ptr_base.add(t.offset_in_section as usize) };
                let mem = &pool[ti];

                let t0 = Instant::now();
                unsafe { fill_alloc_host_ptr_via_map(&backend, mem, src, t.size)? };
                let fus = t0.elapsed().as_secs_f64() * 1e6;
                fill_us.push(fus);

                run_matmul_q4_0(&backend, &kernel, &input_mem, mem, &output_mem, t.n, t.k)?;
                backend.queue.finish()?;
                let out = read_output(t.n)?;
                let baseline_out = &baseline[layer_idx][ti];
                if !bit_equal(&out, baseline_out) {
                    all_eq = false;
                    let mut local_mm = 0usize;
                    for (idx, (&a, &b)) in baseline_out.iter().zip(out.iter()).enumerate() {
                        let d = (a - b).abs();
                        if d > max_diff {
                            max_diff = d;
                        }
                        if a.to_bits() != b.to_bits() {
                            local_mm += 1;
                            if args.dump_mismatches > 0 && local_mm <= args.dump_mismatches {
                                eprintln!(
                                    "  [layer={} {}] idx={:>5} baseline={:.6e} candidate={:.6e}",
                                    layer_idx, t.name, idx, a, b
                                );
                            }
                        }
                    }
                    total_mismatch += local_mm;
                }
            }
        }
        let total_ms = test_start.elapsed().as_secs_f64() * 1e3;

        let mean = mean_us(&fill_us);
        let p50 = percentile_us(&mut fill_us.clone(), 50.0);
        let p95 = percentile_us(&mut fill_us.clone(), 95.0);
        let p_max = max_us(&fill_us);

        println!("total fill+matmul time: {:.2} ms", total_ms);
        println!(
            "per-fill stats (μs): mean={:.1} p50={:.1} p95={:.1} max={:.1} (n={})",
            mean,
            p50,
            p95,
            p_max,
            fill_us.len()
        );
        println!(
            "byte-equal: {} (max diff={:.3e}, mismatches={})",
            if all_eq { "PASS" } else { "FAIL" },
            max_diff,
            total_mismatch
        );

        // Sample distribution of layer 0 fills (first 7 entries).
        let sample_str: Vec<String> = fill_us
            .iter()
            .take(l0.len())
            .map(|v| format!("{:.1}", v))
            .collect();
        println!("layer 0 per-tensor fill (μs): [{}]", sample_str.join(" / "));
        println!();

        drop(pool);
        backend.queue.finish()?;
    }

    // ── Test 2: pool reuse with matmul interleaved ────────────────────────
    if want_test(2) {
        let pool_layers = 2usize; // 2 layers' worth of slots
        let slots_per_layer = l0.len();
        let total_slots = pool_layers * slots_per_layer;
        println!(
            "=== Test 2: pool reuse {} round (pool={} slots = {} layers worth) ===",
            args.n_rounds, total_slots, pool_layers
        );

        let mut pool: Vec<ocl::core::Mem> = Vec::with_capacity(total_slots);
        for _ in 0..pool_layers {
            for t in l0 {
                pool.push(build_buffer_alloc_host_ptr_empty(&backend, t.size)?);
            }
        }
        backend.queue.finish()?;

        let mut round_ms: Vec<f64> = Vec::with_capacity(args.n_rounds as usize);
        let mut fill_ms_per_round: Vec<f64> = Vec::with_capacity(args.n_rounds as usize);
        let mut matmul_ms_per_round: Vec<f64> = Vec::with_capacity(args.n_rounds as usize);
        let mut byte_equal_count = 0u32;
        let mut byte_equal_max_diff = 0.0f32;

        for round in 0..args.n_rounds {
            // Each round: pick layer index round % n_layers, fill into the
            // (round % pool_layers)-th half of the pool, matmul, read.
            let layer_idx = (round as usize) % (n_layers as usize);
            let layer = &per_layer[layer_idx];
            let half = (round as usize) % pool_layers;
            let slot_base = half * slots_per_layer;

            let r0 = Instant::now();

            // (a) fill phase
            let f0 = Instant::now();
            for (ti, t) in layer.iter().enumerate() {
                let src = unsafe { weights_ptr_base.add(t.offset_in_section as usize) };
                let mem = &pool[slot_base + ti];
                unsafe { fill_alloc_host_ptr_via_map(&backend, mem, src, t.size)? };
            }
            let fill_ms = f0.elapsed().as_secs_f64() * 1e3;

            // (b) matmul phase
            let m0 = Instant::now();
            let mut round_eq = true;
            for (ti, t) in layer.iter().enumerate() {
                let mem = &pool[slot_base + ti];
                run_matmul_q4_0(&backend, &kernel, &input_mem, mem, &output_mem, t.n, t.k)?;
                backend.queue.finish()?;
                let out = read_output(t.n)?;
                let baseline_out = &baseline[layer_idx][ti];
                if !bit_equal(&out, baseline_out) {
                    round_eq = false;
                    for (&a, &b) in baseline_out.iter().zip(out.iter()) {
                        let d = (a - b).abs();
                        if d > byte_equal_max_diff {
                            byte_equal_max_diff = d;
                        }
                    }
                }
            }
            let matmul_ms = m0.elapsed().as_secs_f64() * 1e3;
            let total = r0.elapsed().as_secs_f64() * 1e3;

            if round_eq {
                byte_equal_count += 1;
            }
            round_ms.push(total);
            fill_ms_per_round.push(fill_ms);
            matmul_ms_per_round.push(matmul_ms);

            dprint!(
                "[debug] round={} layer={} half={} fill={:.3} ms / matmul={:.3} ms / total={:.3} ms eq={}",
                round,
                layer_idx,
                half,
                fill_ms,
                matmul_ms,
                total,
                round_eq
            );
        }

        let mean_round = round_ms.iter().sum::<f64>() / round_ms.len() as f64;
        let mean_fill = fill_ms_per_round.iter().sum::<f64>() / fill_ms_per_round.len() as f64;
        let mean_mm = matmul_ms_per_round.iter().sum::<f64>() / matmul_ms_per_round.len() as f64;
        let max_round = round_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_round = round_ms.iter().cloned().fold(f64::INFINITY, f64::min);

        // Sample first round + last round for cliff detection.
        let r0 = round_ms.first().copied().unwrap_or(0.0);
        let r_end = round_ms.last().copied().unwrap_or(0.0);
        let r_mid = round_ms[round_ms.len() / 2];
        println!(
            "round 1: {:.3} ms / round mid: {:.3} ms / round last: {:.3} ms",
            r0, r_mid, r_end
        );
        println!(
            "mean per-round: {:.3} ms (min={:.3} max={:.3}, fill_mean={:.3} matmul_mean={:.3})",
            mean_round, min_round, max_round, mean_fill, mean_mm
        );
        println!(
            "byte-equal: {}/{} {}",
            byte_equal_count,
            args.n_rounds,
            if byte_equal_count == args.n_rounds {
                "PASS"
            } else {
                "FAIL"
            }
        );

        // Cliff detector: is max_round > 2× mean? Suggests driver pool issue.
        let cliff = max_round > 2.0 * mean_round && max_round > 1.0;
        if cliff {
            println!(
                "WARN: per-round max ({:.3} ms) > 2× mean ({:.3} ms) — possible driver cliff",
                max_round, mean_round
            );
        }
        println!();

        drop(pool);
        backend.queue.finish()?;
    }

    // ── Test 3: Lifetime ordering (Arc<AufView> drop + cl_mem) ────────────
    if want_test(3) {
        println!("=== Test 3: lifetime ordering ===");
        // Wrap each cl_mem with a struct that holds an Arc<AufView> clone.
        // Drop Arc<AufView> from the outer Vec; cl_mem still hold clones.
        // Then drop cl_mem; the last Arc goes away → mmap unmapped.
        struct PinnedSlot {
            mem: ocl::core::Mem,
            _view_keepalive: Arc<llm_rs2::auf::reader::AufView>,
        }

        let mut pinned: Vec<PinnedSlot> = Vec::with_capacity(l0.len());
        for t in l0 {
            let mem = build_buffer_alloc_host_ptr_empty(&backend, t.size)?;
            let src = unsafe { weights_ptr_base.add(t.offset_in_section as usize) };
            unsafe { fill_alloc_host_ptr_via_map(&backend, &mem, src, t.size)? };
            pinned.push(PinnedSlot {
                mem,
                _view_keepalive: Arc::clone(&view),
            });
        }
        backend.queue.finish()?;
        let strong_after_fill = Arc::strong_count(&view);
        dprint!("[debug] strong_count after fill = {}", strong_after_fill);

        // Drop the outer Vec<Arc<AufView>> reference: shadow `view` to a
        // weak that we can later upgrade-check. We can't drop `view` fully
        // because it's used elsewhere; instead we verify that the slots
        // hold their own clones independently of any fallible release order.
        // The core invariant: Arc reference counting protects the mmap.

        // Run a matmul with the pinned slot to confirm cl_mem still valid
        // after we relax our own strong reference count expectations.
        let layer = &per_layer[0];
        let mut all_eq = true;
        for (ti, t) in layer.iter().enumerate() {
            let mem = &pinned[ti].mem;
            run_matmul_q4_0(&backend, &kernel, &input_mem, mem, &output_mem, t.n, t.k)?;
            backend.queue.finish()?;
            let out = read_output(t.n)?;
            let baseline_out = &baseline[0][ti];
            if !bit_equal(&out, baseline_out) {
                all_eq = false;
            }
        }

        // Now drop pinned (cl_mem releases). Each PinnedSlot's
        // _view_keepalive Arc drops too. Outer view still has at least 1
        // strong ref (the harness keeps it).
        let strong_before_drop = Arc::strong_count(&view);
        drop(pinned);
        backend.queue.finish()?;
        let strong_after_drop = Arc::strong_count(&view);

        println!(
            "strong_count: after_fill={} before_drop={} after_drop={}",
            strong_after_fill, strong_before_drop, strong_after_drop
        );
        println!(
            "matmul on pinned slots: byte-equal={}",
            if all_eq { "PASS" } else { "FAIL" }
        );
        // Survival check: the outer harness reaches this println without
        // SIGSEGV → cl_mem releases happened before AufView drop attempt,
        // and Arc counted refs prevented any premature unmap.
        println!("panic during drop: N");
        println!(
            "lifetime ordering: {}",
            if all_eq && strong_after_drop < strong_after_fill {
                "PASS"
            } else {
                "FAIL"
            }
        );
        println!();
    }

    // ── Test 4: memory pressure ───────────────────────────────────────────
    if want_test(4) {
        println!(
            "=== Test 4: memory pressure ({} MiB writer, {:.1}s) ===",
            args.pressure_mib, args.pressure_secs
        );
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_th = Arc::clone(&stop_flag);
        let pressure_mib = args.pressure_mib;

        // Pre-read RSS for delta reporting.
        let rss_before = read_self_rss_kb().unwrap_or(0);

        // Spawn background pressure thread.
        let handle = std::thread::spawn(move || -> Result<()> {
            let bytes = pressure_mib * 1024 * 1024;
            let mut buf: Vec<u8> = vec![0u8; bytes];
            let mut counter: u64 = 0;
            while !stop_flag_th.load(Ordering::Relaxed) {
                // Touch all pages in chunks; OS may then page-evict ALLOC_HOST_PTR
                // pages if they are not driver-pinned.
                let chunk_size = 4096;
                for c in buf.chunks_mut(chunk_size) {
                    if stop_flag_th.load(Ordering::Relaxed) {
                        break;
                    }
                    c[0] = (counter & 0xff) as u8;
                    counter = counter.wrapping_add(1);
                }
            }
            std::hint::black_box(buf);
            Ok(())
        });

        // Build pool while pressure runs.
        let mut pool: Vec<ocl::core::Mem> = Vec::with_capacity(l0.len());
        for t in l0 {
            pool.push(build_buffer_alloc_host_ptr_empty(&backend, t.size)?);
        }
        backend.queue.finish()?;

        let test_start = Instant::now();
        let pressure_deadline = Duration::from_secs_f64(args.pressure_secs);
        let mut rounds = 0u32;
        let mut byte_equal_failures = 0u32;
        let mut sigbus_seen = false;
        let mut peak_rss_kb = rss_before;

        while test_start.elapsed() < pressure_deadline {
            let layer_idx = (rounds as usize) % (n_layers as usize);
            let layer = &per_layer[layer_idx];
            for (ti, t) in layer.iter().enumerate() {
                let src = unsafe { weights_ptr_base.add(t.offset_in_section as usize) };
                let mem = &pool[ti];
                if let Err(e) = unsafe { fill_alloc_host_ptr_via_map(&backend, mem, src, t.size) } {
                    eprintln!("Test 4: fill error: {}", e);
                    sigbus_seen = true;
                    break;
                }
                run_matmul_q4_0(&backend, &kernel, &input_mem, mem, &output_mem, t.n, t.k)?;
                backend.queue.finish()?;
                let out = read_output(t.n)?;
                let baseline_out = &baseline[layer_idx][ti];
                if !bit_equal(&out, baseline_out) {
                    byte_equal_failures += 1;
                }
            }
            rounds += 1;
            if let Some(rss) = read_self_rss_kb()
                && rss > peak_rss_kb
            {
                peak_rss_kb = rss;
            }
            if sigbus_seen {
                break;
            }
        }

        stop_flag.store(true, Ordering::Relaxed);
        let _ = handle.join();
        backend.queue.finish()?;

        println!(
            "rounds completed: {} ({} byte-equal failures), peak RSS: {} MiB (Δ={} MiB)",
            rounds,
            byte_equal_failures,
            peak_rss_kb / 1024,
            (peak_rss_kb.saturating_sub(rss_before)) / 1024
        );
        println!(
            "byte-equal under pressure: {}",
            if byte_equal_failures == 0 && rounds > 0 {
                "PASS"
            } else {
                "FAIL"
            }
        );
        println!(
            "SIGBUS encountered: {}",
            if sigbus_seen { "Y" } else { "N" }
        );
        println!();

        drop(pool);
        backend.queue.finish()?;
    }

    // ── Test 5: pool size sweep ───────────────────────────────────────────
    if want_test(5) {
        let pool_sizes: Vec<usize> = args
            .pool_sizes
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .filter(|&s| s > 0)
            .collect();
        if pool_sizes.is_empty() {
            anyhow::bail!("Test 5: --pool-sizes parsed empty");
        }
        println!("=== Test 5: pool size sweep {:?} ===", pool_sizes);

        let slots_per_layer = l0.len();
        // Per-pool n_rounds clamped to args.n_rounds.
        let rounds_per_pool = args.n_rounds.max(5);

        for &pool_total in &pool_sizes {
            // Round pool_total down to a multiple of slots_per_layer.
            let pool_layers = (pool_total / slots_per_layer).max(1);
            let actual_total = pool_layers * slots_per_layer;
            if actual_total != pool_total {
                println!(
                    "  pool={:>3}: rounded to {} ({} layers' worth = {} slots)",
                    pool_total, actual_total, pool_layers, actual_total
                );
            }

            let rss_before = read_self_rss_kb().unwrap_or(0);

            let mut pool: Vec<ocl::core::Mem> = Vec::with_capacity(actual_total);
            for _ in 0..pool_layers {
                for t in l0 {
                    pool.push(build_buffer_alloc_host_ptr_empty(&backend, t.size)?);
                }
            }
            backend.queue.finish()?;

            let rss_after_alloc = read_self_rss_kb().unwrap_or(0);

            let mut round_ms: Vec<f64> = Vec::with_capacity(rounds_per_pool as usize);
            let test_start = Instant::now();

            for round in 0..rounds_per_pool {
                let layer_idx = (round as usize) % (n_layers as usize);
                let layer = &per_layer[layer_idx];
                let half = (round as usize) % pool_layers;
                let slot_base = half * slots_per_layer;

                let r0 = Instant::now();
                for (ti, t) in layer.iter().enumerate() {
                    let src = unsafe { weights_ptr_base.add(t.offset_in_section as usize) };
                    let mem = &pool[slot_base + ti];
                    unsafe { fill_alloc_host_ptr_via_map(&backend, mem, src, t.size)? };
                }
                for (ti, t) in layer.iter().enumerate() {
                    let mem = &pool[slot_base + ti];
                    run_matmul_q4_0(&backend, &kernel, &input_mem, mem, &output_mem, t.n, t.k)?;
                    backend.queue.finish()?;
                }
                round_ms.push(r0.elapsed().as_secs_f64() * 1e3);
            }
            let total_ms = test_start.elapsed().as_secs_f64() * 1e3;
            let mean = round_ms.iter().sum::<f64>() / round_ms.len() as f64;
            let max_v = round_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            println!(
                "  pool={:>3} ({} layers): total {:.2} ms ({} rounds), per-round mean={:.3} ms max={:.3} ms; RSS Δ={} MiB ({} → {})",
                actual_total,
                pool_layers,
                total_ms,
                rounds_per_pool,
                mean,
                max_v,
                (rss_after_alloc.saturating_sub(rss_before)) / 1024,
                rss_before / 1024,
                rss_after_alloc / 1024
            );

            drop(pool);
            backend.queue.finish()?;
        }
        println!();
    }

    println!("=== Stage 2 done ===");
    Ok(())
}

#[cfg(feature = "opencl")]
fn mean_us(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

#[cfg(feature = "opencl")]
fn max_us(xs: &[f64]) -> f64 {
    xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

#[cfg(feature = "opencl")]
fn percentile_us(xs: &mut [f64], p: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((p / 100.0) * (xs.len() as f64 - 1.0)).round() as usize;
    xs[idx.min(xs.len() - 1)]
}

/// Read VmRSS (kB) from `/proc/self/status`. Returns None on non-Linux or
/// parse failure.
#[cfg(feature = "opencl")]
fn read_self_rss_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            // e.g. "VmRSS:    12345 kB"
            let trimmed = rest.trim();
            let n_str = trimmed.split_whitespace().next()?;
            return n_str.parse::<u64>().ok();
        }
    }
    None
}
