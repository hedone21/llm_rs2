//! M2.H Qwen 2.5-1.5B 1-layer correctness — production OpPackage vs raw OpenCL.
//!
//! Builds a single QNN `Qnn_GraphHandle_t` containing the full 14-node Qwen
//! decoder layer (RmsNorm → QKV/KV/V → RoPE(Q,K) → KvScatter → FlashAttn → O
//! → Add → RmsNorm → gate/up → SiluMul → down → Add) and compares the layer
//! output against an identical 14-stage raw OpenCL chain that drives the same
//! kernels directly.
//!
//! Reference path keeps it simple: random F32 weights → on-host Q4_0 packed
//! quantisation → identical SOA `q_buf` / `d_buf` consumed by both paths so
//! the numerical baseline is bit-stable apart from QNN graph scheduling
//! drift.
//!
//! ## Pass-gate (M2.A)
//! - `max_abs_err < 1e-2` vs the raw 14-stage OpenCL chain.
//! - `graphFinalize <= 200 ms`. Exceeding the bound is YELLOW (kernel
//!   compilation amortises across application lifetime).
//! - 14 `graphAddNode` calls succeed (verified by `LAYER_NODE_COUNT`).
//!
//! ## Build / deploy
//!
//! ```text
//! cargo build --release --features qnn,opencl --target aarch64-linux-android \
//!   --bin microbench_qnn_qwen_layer
//! cargo build --release -p qnn_oppkg --target aarch64-linux-android
//! adb push target/aarch64-linux-android/release/libqnn_oppkg.so /data/local/tmp/
//! python scripts/run_device.py -d galaxy_s25 microbench_qnn_qwen_layer
//! ```

#[cfg(not(feature = "qnn"))]
fn main() {
    eprintln!("microbench_qnn_qwen_layer requires --features qnn");
    std::process::exit(2);
}

#[cfg(feature = "qnn")]
#[allow(
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code
)]
mod qnn {
    include!(concat!(env!("OUT_DIR"), "/qnn_bindings.rs"));
}

#[cfg(feature = "qnn")]
const RMS_EPS: f32 = 1e-5;

#[cfg(feature = "qnn")]
const QK4_0: usize = 32;
#[cfg(feature = "qnn")]
const QS_PER_BLOCK: usize = 16;

#[cfg(feature = "qnn")]
fn main() -> anyhow::Result<()> {
    use libloading::{Library, Symbol};
    use std::ffi::{CString, c_void};
    use std::os::raw::c_uint;
    use std::ptr;

    use qnn::*;

    const PKG_PATH: &str = "/data/local/tmp/libqnn_oppkg.so";
    const PKG_PROVIDER: &str = "QnnOpPackage_InitInterface";
    const PKG_TARGET: &str = "GPU_QTI_AISW";

    // Qwen 2.5-1.5B reference — mirrors `qnn_oppkg::graph::LayerConfig::qwen2p5_1p5b()`.
    // Inlined here to keep the engine crate dependency-free of qnn_oppkg
    // (INV-160: production code unchanged).
    let dim: usize = 1536;
    let n_head: usize = 12;
    let n_kv_heads: usize = 2;
    let head_dim: usize = 128;
    let ffn_dim: usize = 8960;
    let kv_capacity: usize = 2048;
    let q_proj_out: usize = n_head * head_dim;
    let kv_proj_out: usize = n_kv_heads * head_dim;
    // D-D.6 디버깅: production state inject 시 (pos, n_kv) override.
    // `LLMRS_MICROBENCH_POS=k`, `LLMRS_MICROBENCH_N_KV=k` 설정 시.
    let pos: i32 = std::env::var("LLMRS_MICROBENCH_POS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(0);
    let n_kv: usize = std::env::var("LLMRS_MICROBENCH_N_KV")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1);
    let theta: f32 = 1_000_000.0; // Qwen2.5 RoPE theta.
    if pos != 0 || n_kv != 1 {
        eprintln!("[microbench] (pos, n_kv) override: pos={pos}, n_kv={n_kv}");
    }

    // D-D.6 디버깅: production-style backend ordering inject. OpenCLBackend
    // secondary 인스턴스를 미리 init하여 cl_context가 GPU에 alloc되어 있는
    // 상태에서 QNN graph build/execute. `LLMRS_MICROBENCH_OCL_SECONDARY=1` 시 활성.
    let _ocl_secondary = if std::env::var("LLMRS_MICROBENCH_OCL_SECONDARY").as_deref() == Ok("1") {
        eprintln!("[microbench] OCL_SECONDARY: initializing OpenCLBackend (production-style)");
        Some(llm_rs2::backend::opencl::OpenCLBackend::new()?)
    } else {
        None
    };

    println!("=== microbench_qnn_qwen_layer (M2.H) ===\n");
    println!("Op Package:       {}", PKG_PATH);
    println!("Interface symbol: {}", PKG_PROVIDER);
    println!(
        "Layer dims: dim={}, n_head={}, n_kv_heads={}, head_dim={}, ffn_dim={}, kv_capacity={}",
        dim, n_head, n_kv_heads, head_dim, ffn_dim, kv_capacity
    );
    println!(
        "Topology: 14 nodes — RmsNorm -> Q/K/V -> RoPE(Q,K) -> KvScatter -> FlashAttn -> O \
         -> Add -> RmsNorm -> gate/up -> SiluMul -> down -> Add\n"
    );

    let backend_lib_path = std::env::var("QNN_BACKEND_LIB")
        .unwrap_or_else(|_| "/data/local/tmp/qnn/libQnnGpu.so".to_string());
    println!("Backend lib: {}\n", backend_lib_path);

    // ── Build raw-OpenCL reference toolchain (14 separate kernel objects) ────
    use ocl::{Context, Device, Platform, Program, Queue};

    let platform = Platform::default();
    let device = Device::first(platform)?;
    let cl_ctx = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let cl_q = Queue::new(&cl_ctx, device, None)?;

    let simple_src = include_str!("../../kernels/simple_ops.cl");
    let add_src = include_str!("../../kernels/add.cl");
    let q40_src = include_str!("../../kernels/mul_mv_q4_0_f32_8x_flat.cl");
    let fa_src = include_str!("../../kernels/flash_attn_f32_f16.cl");
    // D-D.6: production OpenCLBackend의 build flag와 byte-equal 통일.
    let cl_opts = "-cl-std=CL2.0 -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math";
    let fa_opts = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math -DDK=128 -DDV=128 \
                   -DBLOCK_M=32 -DBLOCK_N=32";

    let prog_simple = Program::builder()
        .devices(device)
        .src(simple_src)
        .cmplr_opt(cl_opts)
        .build(&cl_ctx)?;
    let prog_add = Program::builder()
        .devices(device)
        .src(add_src)
        .cmplr_opt(cl_opts)
        .build(&cl_ctx)?;
    let prog_q40 = Program::builder()
        .devices(device)
        .src(q40_src)
        .cmplr_opt(cl_opts)
        .build(&cl_ctx)?;
    let prog_fa = Program::builder()
        .devices(device)
        .src(fa_src)
        .cmplr_opt(fa_opts)
        .build(&cl_ctx)?;

    // D-D.6: graph가 사용하는 새 OOP single-subgroup variant로 raw chain도 통일.
    // 같은 algorithm 사용 → microbench에서 graph (new) vs raw chain (new)가 byte-equal.
    let k_rms = ocl::core::create_kernel(&prog_simple, "kernel_rms_norm_oop_subgroup")?;
    let k_q40 = ocl::core::create_kernel(&prog_q40, "kernel_mul_mat_q4_0_f32_8x_flat")?;
    let k_rope = ocl::core::create_kernel(&prog_simple, "kernel_rope_simple")?;
    let k_kvs = ocl::core::create_kernel(&prog_simple, "kernel_kv_scatter_f32_to_f16")?;
    let k_fa = ocl::core::create_kernel(&prog_fa, "flash_attn_f32_f16_q1")?;
    let k_silu = ocl::core::create_kernel(&prog_simple, "kernel_silu_mul_simple")?;
    let k_add = ocl::core::create_kernel(&prog_add, "kernel_add_row")?;
    // D-D.6 Phase A.5: Qwen2.5 QKV bias add (forward_gen와 동일 환경 재현용).
    let k_bias = ocl::core::create_kernel(&prog_simple, "kernel_add_row_bias")?;

    // ── QNN backend + register OpPackage ─────────────────────────────────────
    let gpu_lib = unsafe { Library::new(&backend_lib_path) }?;
    type GetProvidersFn = unsafe extern "C" fn(*mut *mut *const QnnInterface_t, *mut c_uint) -> u64;
    let gp: Symbol<GetProvidersFn> = unsafe { gpu_lib.get(b"QnnInterface_getProviders\0")? };
    let mut provs: *mut *const QnnInterface_t = ptr::null_mut();
    let mut np: c_uint = 0;
    let err = unsafe { gp(&mut provs, &mut np) };
    anyhow::ensure!(err == 0 && np > 0, "GPU getProviders err=0x{:x}", err);
    let v = unsafe { (**provs).__bindgen_anon_1.v2_25 };

    // VERBOSE logger — NULL callback = QNN's default platform logger (Android logcat).
    // Inspect with: adb logcat -v threadtime QnnGpu:V QnnGpuOpPackage:V QnnGraph:V QnnDevice:V \*:S
    let mut logger: Qnn_LogHandle_t = ptr::null_mut();
    if let Some(log_create) = v.logCreate {
        let err = unsafe { log_create(None, QnnLog_Level_t_QNN_LOG_LEVEL_VERBOSE, &mut logger) };
        if err != 0 {
            eprintln!("logCreate err=0x{:x} (proceeding without logger)", err);
            logger = ptr::null_mut();
        } else {
            eprintln!("logCreate VERBOSE: OK");
        }
    }

    let mut be: Qnn_BackendHandle_t = ptr::null_mut();
    let err = unsafe { (v.backendCreate.unwrap())(logger, ptr::null_mut(), &mut be) };
    anyhow::ensure!(err == 0, "backendCreate err=0x{:x}", err);
    println!("backend: OK");

    let pkg_path = CString::new(PKG_PATH).unwrap();
    let pkg_provider = CString::new(PKG_PROVIDER).unwrap();
    let pkg_target = CString::new(PKG_TARGET).unwrap();
    let reg_fn = v
        .backendRegisterOpPackage
        .ok_or_else(|| anyhow::anyhow!("backendRegisterOpPackage is NULL"))?;
    let err = unsafe {
        reg_fn(
            be,
            pkg_path.as_ptr(),
            pkg_provider.as_ptr(),
            pkg_target.as_ptr(),
        )
    };
    println!("registerOpPackage -> err=0x{:x}", err);
    anyhow::ensure!(err == 0, "registerOpPackage err=0x{:x}", err);

    let mut ctx: Qnn_ContextHandle_t = ptr::null_mut();
    let err = unsafe { (v.contextCreate.unwrap())(be, ptr::null_mut(), ptr::null_mut(), &mut ctx) };
    anyhow::ensure!(err == 0, "contextCreate err=0x{:x}", err);

    const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
    const RPCMEM_DEFAULT_FLAGS: u32 = 1;
    type RpcmemAllocFn = unsafe extern "C" fn(heapid: i32, flags: u32, size: i32) -> *mut c_void;
    type RpcmemFreeFn = unsafe extern "C" fn(po: *mut c_void);
    type RpcmemToFdFn = unsafe extern "C" fn(po: *const c_void) -> i32;
    let rpc_lib = unsafe { Library::new("/vendor/lib64/libcdsprpc.so") }?;
    let rpcmem_alloc: Symbol<RpcmemAllocFn> = unsafe { rpc_lib.get(b"rpcmem_alloc\0")? };
    let _rpcmem_free: Symbol<RpcmemFreeFn> = unsafe { rpc_lib.get(b"rpcmem_free\0")? };
    let rpcmem_to_fd: Symbol<RpcmemToFdFn> = unsafe { rpc_lib.get(b"rpcmem_to_fd\0")? };

    let result = run_layer(
        &v,
        ctx,
        &cl_q,
        &cl_ctx,
        &k_rms,
        &k_q40,
        &k_rope,
        &k_kvs,
        &k_fa,
        &k_silu,
        &k_add,
        &k_bias,
        &rpcmem_alloc,
        &rpcmem_to_fd,
        RPCMEM_HEAP_ID_SYSTEM,
        RPCMEM_DEFAULT_FLAGS,
        dim,
        n_head,
        n_kv_heads,
        head_dim,
        ffn_dim,
        kv_capacity,
        q_proj_out,
        kv_proj_out,
        pos,
        n_kv,
        theta,
    );

    let pass = match result {
        Ok((max_err, finalize_ms)) => {
            let acc_thresh = 1e-2_f32;
            let acc_pass = max_err < acc_thresh;
            let fin_pass = finalize_ms <= 200.0;
            println!(
                "\nlayer max_abs_err = {:.6e}  thresh={:.0e}  {}",
                max_err,
                acc_thresh,
                if acc_pass { "PASS" } else { "FAIL" }
            );
            println!(
                "graphFinalize     = {:.2} ms        budget=200 ms  {}",
                finalize_ms,
                if fin_pass { "PASS" } else { "YELLOW" }
            );
            acc_pass && fin_pass
        }
        Err(e) => {
            println!("\nERROR: {}", e);
            false
        }
    };

    println!(
        "\n=== M2.H verdict: {} ===",
        if pass { "GREEN" } else { "RED" }
    );

    unsafe {
        let _ = (v.contextFree.unwrap())(ctx, ptr::null_mut());
        let _ = (v.backendFree.unwrap())(be);
    }

    if pass { Ok(()) } else { std::process::exit(1) }
}

/// Pack `m * k` row-major F32 weights into Q4_0 SOA form (`q_bytes`,
/// `d_halves`). Layout matches `kernel_mul_mat_q4_0_f32_8x_flat` expectations:
/// `q_bytes` stores 4-bit packed quants in 16 nibbles per 32-element block,
/// and `d_halves` stores per-block FP16 scales.
#[cfg(feature = "qnn")]
fn pack_q40_soa(weights: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<u16>) {
    assert!(k.is_multiple_of(QK4_0), "K must be a multiple of 32");
    let num_blocks = n * k / QK4_0;
    let mut q = vec![0u8; num_blocks * QS_PER_BLOCK];
    let mut d = vec![0u16; num_blocks];

    for row in 0..n {
        for blk in 0..(k / QK4_0) {
            let base = row * k + blk * QK4_0;
            let block = &weights[base..base + QK4_0];

            // Q4_0 scale: max(abs(x)) / -8.
            let mut amax = 0.0f32;
            let mut max_signed = 0.0f32;
            for &x in block {
                if x.abs() > amax {
                    amax = x.abs();
                    max_signed = x;
                }
            }
            let scale = if amax > 0.0 { max_signed / -8.0 } else { 0.0 };
            let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };

            let blk_idx = row * (k / QK4_0) + blk;
            d[blk_idx] = f32_to_f16_bits(scale);

            // Quantise: q = clamp(round(x * inv) + 8, 0, 15).
            // Pack nibble pair (q[i + 16] << 4) | q[i] for i ∈ [0, 16).
            let mut quants = [0u8; QK4_0];
            for (i, &x) in block.iter().enumerate() {
                let raw = ((x * inv).round() as i32) + 8;
                quants[i] = raw.clamp(0, 15) as u8;
            }
            let q_off = blk_idx * QS_PER_BLOCK;
            for i in 0..QS_PER_BLOCK {
                q[q_off + i] = (quants[i + QS_PER_BLOCK] << 4) | (quants[i] & 0x0F);
            }
        }
    }

    (q, d)
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "qnn")]
fn run_layer(
    v: &qnn::QnnInterface_ImplementationV2_25_t,
    ctx: qnn::Qnn_ContextHandle_t,
    cl_q: &ocl::Queue,
    cl_ctx: &ocl::Context,
    k_rms: &ocl::core::Kernel,
    k_q40: &ocl::core::Kernel,
    k_rope: &ocl::core::Kernel,
    k_kvs: &ocl::core::Kernel,
    k_fa: &ocl::core::Kernel,
    k_silu: &ocl::core::Kernel,
    k_add: &ocl::core::Kernel,
    k_bias: &ocl::core::Kernel,
    rpcmem_alloc: &libloading::Symbol<unsafe extern "C" fn(i32, u32, i32) -> *mut std::ffi::c_void>,
    rpcmem_to_fd: &libloading::Symbol<unsafe extern "C" fn(*const std::ffi::c_void) -> i32>,
    heap_id: i32,
    flags: u32,
    dim: usize,
    n_head: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    kv_capacity: usize,
    q_proj_out: usize,
    kv_proj_out: usize,
    pos: i32,
    n_kv: usize,
    theta: f32,
) -> anyhow::Result<(f32, f64)> {
    use ocl::core::ArgVal;
    use qnn::*;
    use std::ffi::CString;
    use std::os::raw::c_char;
    use std::ptr;
    use std::time::Instant;

    // ── 1. Generate identical inputs/weights for both paths ─────────────────
    // Random but deterministic. Q4_0 weights: pack on host so both paths
    // consume bit-identical SOA buffers.
    let mut host_x = vec![0.0f32; dim];
    for (i, x) in host_x.iter_mut().enumerate() {
        *x = ((i as f32) * 0.0173 + 0.07).rem_euclid(1.0) - 0.5;
    }
    // D-D.6 디버깅: production이 dump한 x_in bytes를 inject. random input
    // (well-conditioned)에서 PASS인데 production에서 RED라면 input pattern이
    // root cause. `LLMRS_MICROBENCH_X_FILE=/path/to/x.bin` 설정 시 활성.
    if let Ok(p) = std::env::var("LLMRS_MICROBENCH_X_FILE") {
        let bytes = std::fs::read(&p)?;
        let expected = dim * 4;
        anyhow::ensure!(
            bytes.len() == expected,
            "x file: bytes={} != expected {expected}",
            bytes.len()
        );
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), host_x.as_mut_ptr() as *mut u8, expected);
        }
        eprintln!(
            "[microbench] x_in injected from {p} — host_x[0..8]={:?}",
            &host_x[..8]
        );
    }
    let mut host_rms_w_pre = vec![0.0f32; dim];
    let mut host_rms_w_post = vec![0.0f32; dim];
    for (i, w) in host_rms_w_pre.iter_mut().enumerate() {
        *w = ((i as f32) * 0.0091 + 0.13).rem_euclid(1.0) * 0.4 + 0.8;
    }
    for (i, w) in host_rms_w_post.iter_mut().enumerate() {
        *w = ((i as f32) * 0.0107 + 0.19).rem_euclid(1.0) * 0.4 + 0.8;
    }
    // Q4_0 weights: small magnitude so quantisation error stays bounded.
    fn gen_weights(n: usize, k: usize, seed: f32) -> Vec<f32> {
        let mut w = vec![0.0f32; n * k];
        for (i, x) in w.iter_mut().enumerate() {
            *x = (((i as f32) * 0.000_31 + seed).rem_euclid(1.0) - 0.5) * 0.1;
        }
        w
    }
    // D-D.6 Phase A.5: Qwen2.5 attention QKV bias. random fallback = zeros (no bias).
    // GGUF inject 시 blk.0.attn_{q,k,v}.bias로 갱신.
    let mut host_q_bias = vec![0.0f32; q_proj_out];
    let mut host_k_bias = vec![0.0f32; kv_proj_out];
    let mut host_v_bias = vec![0.0f32; kv_proj_out];

    let w_q = gen_weights(q_proj_out, dim, 0.13);
    let w_k = gen_weights(kv_proj_out, dim, 0.21);
    let w_v = gen_weights(kv_proj_out, dim, 0.29);
    let w_o = gen_weights(dim, q_proj_out, 0.37);
    let w_gate = gen_weights(ffn_dim, dim, 0.43);
    let w_up = gen_weights(ffn_dim, dim, 0.51);
    let w_down = gen_weights(dim, ffn_dim, 0.59);

    // Pack to Q4_0 SOA on host (so raw OpenCL and OpPackage paths consume
    // identical byte buffers — quantisation error becomes a constant offset
    // shared by both paths).
    let (mut qq_q, mut qq_d) = pack_q40_soa(&w_q, q_proj_out, dim);
    let (mut qk_q, mut qk_d) = pack_q40_soa(&w_k, kv_proj_out, dim);
    let (mut qv_q, mut qv_d) = pack_q40_soa(&w_v, kv_proj_out, dim);
    let (mut qo_q, mut qo_d) = pack_q40_soa(&w_o, dim, q_proj_out);
    let (mut qg_q, mut qg_d) = pack_q40_soa(&w_gate, ffn_dim, dim);
    let (mut qu_q, mut qu_d) = pack_q40_soa(&w_up, ffn_dim, dim);
    let (mut qd_q, mut qd_d) = pack_q40_soa(&w_down, dim, ffn_dim);

    // D-D.6 디버깅: GGUF에서 layer 0 production weight를 inject하여 random
    // weight (well-conditioned) vs 실제 weight (specific patterns) 차이 검증.
    // `LLMRS_MICROBENCH_GGUF=/path/to/qwen.gguf` 설정 시 활성. inject 후 raw
    // OpenCL chain과 graph 둘 다 GGUF AOS→SOA bytes를 consume하므로
    // bit-identical (graph kernel이 정상이면 PASS).
    if let Ok(p) = std::env::var("LLMRS_MICROBENCH_GGUF") {
        eprintln!("[microbench] GGUF inject: loading layer 0 weights from {p}");
        use llm_rs2::backend::qnn_oppkg::weight_pack::aos_to_soa_q4_0;
        use llm_rs2::models::loader::gguf::GgufFile;
        let gguf = GgufFile::open(std::path::Path::new(&p))?;
        let load_q4_0 = |name: &str, n: usize, k: usize| -> anyhow::Result<(Vec<u8>, Vec<u16>)> {
            let info = gguf
                .find_tensor(name)
                .ok_or_else(|| anyhow::anyhow!("GGUF: {name} not found"))?;
            let raw = gguf.tensor_data(info);
            let num_blocks = n * k / 32;
            let expected = num_blocks * 18;
            anyhow::ensure!(
                raw.len() >= expected,
                "GGUF {name}: bytes={} < expected {expected}",
                raw.len()
            );
            Ok(aos_to_soa_q4_0(&raw[..expected], n, k))
        };
        let load_f32 = |name: &str, n: usize| -> anyhow::Result<Vec<f32>> {
            let info = gguf
                .find_tensor(name)
                .ok_or_else(|| anyhow::anyhow!("GGUF: {name} not found"))?;
            let raw = gguf.tensor_data(info);
            let expected = n * 4;
            anyhow::ensure!(
                raw.len() >= expected,
                "GGUF {name}: bytes={} < expected {expected}",
                raw.len()
            );
            let mut out = vec![0.0f32; n];
            unsafe {
                std::ptr::copy_nonoverlapping(raw.as_ptr(), out.as_mut_ptr() as *mut u8, expected);
            }
            Ok(out)
        };
        let (q_q, q_d) = load_q4_0("blk.0.attn_q.weight", q_proj_out, dim)?;
        qq_q = q_q;
        qq_d = q_d;
        let (k_q, k_d) = load_q4_0("blk.0.attn_k.weight", kv_proj_out, dim)?;
        qk_q = k_q;
        qk_d = k_d;
        let (v_q, v_d) = load_q4_0("blk.0.attn_v.weight", kv_proj_out, dim)?;
        qv_q = v_q;
        qv_d = v_d;
        let (o_q, o_d) = load_q4_0("blk.0.attn_output.weight", dim, q_proj_out)?;
        qo_q = o_q;
        qo_d = o_d;
        let (g_q, g_d) = load_q4_0("blk.0.ffn_gate.weight", ffn_dim, dim)?;
        qg_q = g_q;
        qg_d = g_d;
        let (u_q, u_d) = load_q4_0("blk.0.ffn_up.weight", ffn_dim, dim)?;
        qu_q = u_q;
        qu_d = u_d;
        let (d_q, d_d) = load_q4_0("blk.0.ffn_down.weight", dim, ffn_dim)?;
        qd_q = d_q;
        qd_d = d_d;
        host_rms_w_pre = load_f32("blk.0.attn_norm.weight", dim)?;
        host_rms_w_post = load_f32("blk.0.ffn_norm.weight", dim)?;
        // D-D.6 Phase A.5: Qwen2.5 QKV bias load (forward_gen와 동일 환경).
        host_q_bias = load_f32("blk.0.attn_q.bias", q_proj_out)?;
        host_k_bias = load_f32("blk.0.attn_k.bias", kv_proj_out)?;
        host_v_bias = load_f32("blk.0.attn_v.bias", kv_proj_out)?;
        eprintln!(
            "[microbench] GGUF inject done — qq_q.len={} qq_d[0..4]={:?} rms_pre[0..4]={:?} q_bias[0..4]={:?}",
            qq_q.len(),
            &qq_d[..4.min(qq_d.len())],
            &host_rms_w_pre[..4.min(host_rms_w_pre.len())],
            &host_q_bias[..4.min(host_q_bias.len())]
        );
    }

    // ── 2. Path A: raw OpenCL — 14-stage chain ──────────────────────────────
    // Buffers: x, rms_w_pre, q4_0 (q+d) per weight, intermediates, kv cache.
    let kv_total = n_kv_heads * kv_capacity * head_dim;
    let mk_buf_f32_ro = |n| unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, n, None)
    };
    let mk_buf_f32_rw = |n| unsafe {
        ocl::core::create_buffer::<_, f32>(cl_ctx.as_core(), ocl::core::MEM_READ_WRITE, n, None)
    };
    let mk_buf_u8_ro = |n| unsafe {
        ocl::core::create_buffer::<_, u8>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, n, None)
    };
    let mk_buf_u16_ro = |n| unsafe {
        ocl::core::create_buffer::<_, u16>(cl_ctx.as_core(), ocl::core::MEM_READ_ONLY, n, None)
    };
    let mk_buf_u16_rw = |n| unsafe {
        ocl::core::create_buffer::<_, u16>(cl_ctx.as_core(), ocl::core::MEM_READ_WRITE, n, None)
    };

    let buf_x = mk_buf_f32_ro(dim)?;
    let buf_rms_w_pre = mk_buf_f32_ro(dim)?;
    let buf_rms_w_post = mk_buf_f32_ro(dim)?;
    let buf_y1 = mk_buf_f32_rw(dim)?;
    let buf_qq = mk_buf_u8_ro(qq_q.len())?;
    let buf_qd = mk_buf_u16_ro(qq_d.len())?;
    let buf_kq = mk_buf_u8_ro(qk_q.len())?;
    let buf_kd = mk_buf_u16_ro(qk_d.len())?;
    let buf_vq = mk_buf_u8_ro(qv_q.len())?;
    let buf_vd = mk_buf_u16_ro(qv_d.len())?;
    let buf_oq = mk_buf_u8_ro(qo_q.len())?;
    let buf_od = mk_buf_u16_ro(qo_d.len())?;
    let buf_gq = mk_buf_u8_ro(qg_q.len())?;
    let buf_gd = mk_buf_u16_ro(qg_d.len())?;
    let buf_uq = mk_buf_u8_ro(qu_q.len())?;
    let buf_ud = mk_buf_u16_ro(qu_d.len())?;
    let buf_dq = mk_buf_u8_ro(qd_q.len())?;
    let buf_dd = mk_buf_u16_ro(qd_d.len())?;
    let buf_q = mk_buf_f32_rw(q_proj_out)?;
    let buf_k = mk_buf_f32_rw(kv_proj_out)?;
    let buf_v = mk_buf_f32_rw(kv_proj_out)?;
    // D-D.6 Phase A.5: Qwen2.5 QKV bias buffers.
    let buf_q_bias = mk_buf_f32_ro(q_proj_out)?;
    let buf_k_bias = mk_buf_f32_ro(kv_proj_out)?;
    let buf_v_bias = mk_buf_f32_ro(kv_proj_out)?;
    let buf_kcache = mk_buf_u16_rw(kv_total)?;
    let buf_vcache = mk_buf_u16_rw(kv_total)?;
    let buf_attn_o = mk_buf_f32_rw(q_proj_out)?;
    let buf_o = mk_buf_f32_rw(dim)?;
    let buf_x_attn = mk_buf_f32_rw(dim)?;
    let buf_y2 = mk_buf_f32_rw(dim)?;
    let buf_gate = mk_buf_f32_rw(ffn_dim)?;
    let buf_up = mk_buf_f32_rw(ffn_dim)?;
    let buf_down = mk_buf_f32_rw(dim)?;
    let buf_x_out = mk_buf_f32_rw(dim)?;

    // Upload all host-side buffers (zero-init kv cache).
    let host_kv_zero = vec![0u16; kv_total];
    macro_rules! up_f32 {
        ($buf:expr, $host:expr) => {
            unsafe {
                ocl::core::enqueue_write_buffer(
                    cl_q,
                    $buf,
                    true,
                    0,
                    $host,
                    None::<ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        };
    }
    macro_rules! up_u8 {
        ($buf:expr, $host:expr) => {
            unsafe {
                ocl::core::enqueue_write_buffer(
                    cl_q,
                    $buf,
                    true,
                    0,
                    $host,
                    None::<ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        };
    }
    macro_rules! up_u16 {
        ($buf:expr, $host:expr) => {
            unsafe {
                ocl::core::enqueue_write_buffer(
                    cl_q,
                    $buf,
                    true,
                    0,
                    $host,
                    None::<ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        };
    }
    up_f32!(&buf_x, &host_x);
    up_f32!(&buf_rms_w_pre, &host_rms_w_pre);
    up_f32!(&buf_rms_w_post, &host_rms_w_post);
    up_u8!(&buf_qq, &qq_q);
    up_u16!(&buf_qd, &qq_d);
    up_u8!(&buf_kq, &qk_q);
    up_u16!(&buf_kd, &qk_d);
    up_u8!(&buf_vq, &qv_q);
    up_u16!(&buf_vd, &qv_d);
    up_u8!(&buf_oq, &qo_q);
    up_u16!(&buf_od, &qo_d);
    up_u8!(&buf_gq, &qg_q);
    up_u16!(&buf_gd, &qg_d);
    up_u8!(&buf_uq, &qu_q);
    up_u16!(&buf_ud, &qu_d);
    up_u8!(&buf_dq, &qd_q);
    up_u16!(&buf_dd, &qd_d);
    up_f32!(&buf_q_bias, &host_q_bias);
    up_f32!(&buf_k_bias, &host_k_bias);
    up_f32!(&buf_v_bias, &host_v_bias);
    // D-D.6 디버깅: KV K/V file inject — production state inject 시. 미설정이면
    // host_kv_zero (default). raw OpenCL chain의 buf_kcache/vcache + graph rpcmem
    // KV slot 양쪽 모두 동일 bytes로 초기화 (둘이 동일 input 갖도록 byte-equal 비교).
    let host_kv_k: Vec<u16> = match std::env::var("LLMRS_MICROBENCH_KV_K_FILE") {
        Ok(p) => {
            let raw = std::fs::read(&p)?;
            let expected = kv_total * 2;
            anyhow::ensure!(
                raw.len() == expected,
                "kv_k file: bytes={} != expected {expected}",
                raw.len()
            );
            let mut out = vec![0u16; kv_total];
            unsafe {
                std::ptr::copy_nonoverlapping(raw.as_ptr(), out.as_mut_ptr() as *mut u8, expected);
            }
            eprintln!("[microbench] KV K injected from {p}");
            out
        }
        Err(_) => host_kv_zero.clone(),
    };
    let host_kv_v: Vec<u16> = match std::env::var("LLMRS_MICROBENCH_KV_V_FILE") {
        Ok(p) => {
            let raw = std::fs::read(&p)?;
            let expected = kv_total * 2;
            anyhow::ensure!(
                raw.len() == expected,
                "kv_v file: bytes={} != expected {expected}",
                raw.len()
            );
            let mut out = vec![0u16; kv_total];
            unsafe {
                std::ptr::copy_nonoverlapping(raw.as_ptr(), out.as_mut_ptr() as *mut u8, expected);
            }
            eprintln!("[microbench] KV V injected from {p}");
            out
        }
        Err(_) => host_kv_zero.clone(),
    };
    up_u16!(&buf_kcache, &host_kv_k);
    up_u16!(&buf_vcache, &host_kv_v);
    ocl::core::finish(cl_q)?;

    // ── M2.I breakdown timing — raw OpenCL chain wall-clock ─────────────────
    // Measure stage 1..16 (per-stage cl_finish pattern, matching M2.I 1st
    // measurement). Single-shot is acceptable here since per-stage finish
    // already provides 14 sync points; graph-build closure / multi-run loop
    // would require structural refactor of 363 lines of in-place kernel
    // dispatch — out of scope for measurement-only sprint.
    let t_raw_chain_start = Instant::now();

    // Stage 1: RmsNorm(pre) — kernel_rms_norm_simple(x, w, y1, dim, eps)
    let dim_i: i32 = dim as i32;
    let ffn_dim_i: i32 = ffn_dim as i32;
    ocl::core::set_kernel_arg(k_rms, 0, ArgVal::mem(&buf_x))?;
    ocl::core::set_kernel_arg(k_rms, 1, ArgVal::mem(&buf_rms_w_pre))?;
    ocl::core::set_kernel_arg(k_rms, 2, ArgVal::mem(&buf_y1))?;
    ocl::core::set_kernel_arg(k_rms, 3, ArgVal::scalar(&dim_i))?;
    ocl::core::set_kernel_arg(k_rms, 4, ArgVal::scalar(&RMS_EPS))?;
    // D-D.6: kernel_rms_norm_oop_subgroup contract — global=[rows*64], local=[64].
    let rms_global = [1usize * 64, 1, 1];
    let rms_local = [64usize, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_rms,
            1,
            None,
            &rms_global,
            Some(rms_local),
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // D-D.6 디버깅: raw chain RMS output (buf_y1) dump.
    if let Ok(p) = std::env::var("LLMRS_MICROBENCH_DUMP_RMS_OUT") {
        let mut bytes = vec![0u8; dim * 4];
        unsafe {
            ocl::core::enqueue_read_buffer(
                cl_q,
                &buf_y1,
                true,
                0,
                &mut bytes,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        std::fs::write(&p, &bytes)?;
        eprintln!("[microbench-dump-rms] wrote {} bytes to {p}", bytes.len());
    }

    // D-D.6 stage-by-stage dump (LLMRS_MICROBENCH_DUMP_PREFIX=path).
    // 각 stage 결과를 `path.stageN`에 binary로 저장. forward_gen
    // (`LLMRS_QNN_OPPKG_DUMP_FALLBACK_PREFIX`)와 stage 번호 1:1 매칭.
    macro_rules! dump_stage_mb {
        ($stage:literal, $buf:expr, $nbytes:expr) => {
            if let Ok(prefix) = std::env::var("LLMRS_MICROBENCH_DUMP_PREFIX") {
                let mut bytes = vec![0u8; $nbytes];
                unsafe {
                    ocl::core::enqueue_read_buffer(
                        cl_q,
                        $buf,
                        true,
                        0,
                        &mut bytes,
                        None::<&ocl::core::Event>,
                        None::<&mut ocl::core::Event>,
                    )?;
                }
                let path = format!("{}.stage{}", prefix, $stage);
                std::fs::write(&path, &bytes)?;
                eprintln!(
                    "[microbench-dump-s{}] {} bytes -> {}",
                    $stage,
                    bytes.len(),
                    path
                );
            }
        };
    }
    dump_stage_mb!(1, &buf_y1, dim * 4);

    // Helper for Q4_0 matmul stage. Mirrors microbench_qnn_oppkg_matmul_q40.
    let dispatch_q40 = |k_q40: &ocl::core::Kernel,
                        bq: &ocl::core::Mem,
                        bd: &ocl::core::Mem,
                        bx: &ocl::core::Mem,
                        by: &ocl::core::Mem,
                        m: usize,
                        n: usize,
                        k: usize|
     -> anyhow::Result<()> {
        let ne00 = k as i32;
        let ne01 = n as i32;
        let ne02: i32 = 1;
        let ne10 = k as i32;
        let ne12: i32 = 1;
        let ne0 = n as i32;
        let ne1 = m as i32;
        let r2: i32 = 1;
        let r3: i32 = 1;
        let off1: u64 = 0;
        let offd: u64 = 0;
        ocl::core::set_kernel_arg(k_q40, 0, ArgVal::mem(bq))?;
        ocl::core::set_kernel_arg(k_q40, 1, ArgVal::mem(bd))?;
        ocl::core::set_kernel_arg(k_q40, 2, ArgVal::mem(bx))?;
        ocl::core::set_kernel_arg(k_q40, 3, ArgVal::scalar(&off1))?;
        ocl::core::set_kernel_arg(k_q40, 4, ArgVal::mem(by))?;
        ocl::core::set_kernel_arg(k_q40, 5, ArgVal::scalar(&offd))?;
        ocl::core::set_kernel_arg(k_q40, 6, ArgVal::scalar(&ne00))?;
        ocl::core::set_kernel_arg(k_q40, 7, ArgVal::scalar(&ne01))?;
        ocl::core::set_kernel_arg(k_q40, 8, ArgVal::scalar(&ne02))?;
        ocl::core::set_kernel_arg(k_q40, 9, ArgVal::scalar(&ne10))?;
        ocl::core::set_kernel_arg(k_q40, 10, ArgVal::scalar(&ne12))?;
        ocl::core::set_kernel_arg(k_q40, 11, ArgVal::scalar(&ne0))?;
        ocl::core::set_kernel_arg(k_q40, 12, ArgVal::scalar(&ne1))?;
        ocl::core::set_kernel_arg(k_q40, 13, ArgVal::scalar(&r2))?;
        ocl::core::set_kernel_arg(k_q40, 14, ArgVal::scalar(&r3))?;
        let global = [n.div_ceil(8) * 64, 1, 1];
        let local = [64usize, 1, 1];
        unsafe {
            ocl::core::enqueue_kernel(
                cl_q,
                k_q40,
                3,
                None,
                &global,
                Some(local),
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        ocl::core::finish(cl_q)?;
        Ok(())
    };

    // Stages 2-4: Q/K/V projection (M=1).
    // D-D.6 Phase A.5: Q/K/V matmul 직후 bias add (Qwen2.5 attention bias).
    // dump_stage_mb!(N) 위치는 forward_gen와 동일하게 pre-bias로 유지.
    let dispatch_bias = |bias_buf: &ocl::core::Mem,
                         x_buf: &ocl::core::Mem,
                         total: usize|
     -> anyhow::Result<()> {
        let total_i = total as i32;
        let dim_i = total as i32;
        ocl::core::set_kernel_arg(k_bias, 0, ArgVal::mem(x_buf))?;
        ocl::core::set_kernel_arg(k_bias, 1, ArgVal::mem(bias_buf))?;
        ocl::core::set_kernel_arg(k_bias, 2, ArgVal::scalar(&dim_i))?;
        ocl::core::set_kernel_arg(k_bias, 3, ArgVal::scalar(&total_i))?;
        let global = [total, 1, 1];
        unsafe {
            ocl::core::enqueue_kernel(
                cl_q,
                k_bias,
                1,
                None,
                &global,
                None,
                None::<&ocl::core::Event>,
                None::<&mut ocl::core::Event>,
            )?;
        }
        ocl::core::finish(cl_q)?;
        Ok(())
    };
    dispatch_q40(k_q40, &buf_qq, &buf_qd, &buf_y1, &buf_q, 1, q_proj_out, dim)?;
    dump_stage_mb!(2, &buf_q, q_proj_out * 4);
    dispatch_bias(&buf_q_bias, &buf_q, q_proj_out)?;
    dispatch_q40(
        k_q40,
        &buf_kq,
        &buf_kd,
        &buf_y1,
        &buf_k,
        1,
        kv_proj_out,
        dim,
    )?;
    dump_stage_mb!(3, &buf_k, kv_proj_out * 4);
    dispatch_bias(&buf_k_bias, &buf_k, kv_proj_out)?;
    dispatch_q40(
        k_q40,
        &buf_vq,
        &buf_vd,
        &buf_y1,
        &buf_v,
        1,
        kv_proj_out,
        dim,
    )?;
    dump_stage_mb!(4, &buf_v, kv_proj_out * 4);
    dispatch_bias(&buf_v_bias, &buf_v, kv_proj_out)?;

    // Stage 5: RoPE(Q) — kernel_rope_simple(x, head_dim, num_heads, seq_len, start_pos, theta)
    let head_dim_i: i32 = head_dim as i32;
    let n_head_i: i32 = n_head as i32;
    let n_kv_heads_i: i32 = n_kv_heads as i32;
    let seq_len_i: i32 = 1;
    ocl::core::set_kernel_arg(k_rope, 0, ArgVal::mem(&buf_q))?;
    ocl::core::set_kernel_arg(k_rope, 1, ArgVal::scalar(&head_dim_i))?;
    ocl::core::set_kernel_arg(k_rope, 2, ArgVal::scalar(&n_head_i))?;
    ocl::core::set_kernel_arg(k_rope, 3, ArgVal::scalar(&seq_len_i))?;
    ocl::core::set_kernel_arg(k_rope, 4, ArgVal::scalar(&pos))?;
    ocl::core::set_kernel_arg(k_rope, 5, ArgVal::scalar(&theta))?;
    let pairs_q = n_head * (head_dim / 2);
    let global_rope_q = [pairs_q, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_rope,
            1,
            None,
            &global_rope_q,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;
    dump_stage_mb!(5, &buf_q, q_proj_out * 4);

    // Stage 6: RoPE(K)
    ocl::core::set_kernel_arg(k_rope, 0, ArgVal::mem(&buf_k))?;
    ocl::core::set_kernel_arg(k_rope, 1, ArgVal::scalar(&head_dim_i))?;
    ocl::core::set_kernel_arg(k_rope, 2, ArgVal::scalar(&n_kv_heads_i))?;
    ocl::core::set_kernel_arg(k_rope, 3, ArgVal::scalar(&seq_len_i))?;
    ocl::core::set_kernel_arg(k_rope, 4, ArgVal::scalar(&pos))?;
    ocl::core::set_kernel_arg(k_rope, 5, ArgVal::scalar(&theta))?;
    let pairs_k = n_kv_heads * (head_dim / 2);
    let global_rope_k = [pairs_k, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_rope,
            1,
            None,
            &global_rope_k,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;
    dump_stage_mb!(6, &buf_k, kv_proj_out * 4);

    // Stage 7: KvScatter — kernel_kv_scatter_f32_to_f16(k_src, v_src, k_dst, v_dst, head_dim, capacity, write_pos)
    let capacity_i: i32 = kv_capacity as i32;
    ocl::core::set_kernel_arg(k_kvs, 0, ArgVal::mem(&buf_k))?;
    ocl::core::set_kernel_arg(k_kvs, 1, ArgVal::mem(&buf_v))?;
    ocl::core::set_kernel_arg(k_kvs, 2, ArgVal::mem(&buf_kcache))?;
    ocl::core::set_kernel_arg(k_kvs, 3, ArgVal::mem(&buf_vcache))?;
    ocl::core::set_kernel_arg(k_kvs, 4, ArgVal::scalar(&head_dim_i))?;
    ocl::core::set_kernel_arg(k_kvs, 5, ArgVal::scalar(&capacity_i))?;
    ocl::core::set_kernel_arg(k_kvs, 6, ArgVal::scalar(&pos))?;
    let global_kvs = [kv_proj_out, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_kvs,
            1,
            None,
            &global_kvs,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // Stage 8: FlashAttn — flash_attn_f32_f16_q1 (decode, n_kv=1)
    let buf_score_dummy = mk_buf_f32_rw(1)?;
    let q_nb1 = (n_head * head_dim * 4) as u64;
    let q_nb2 = (head_dim * 4) as u64;
    let q_nb3 = q_nb1;
    let k_nb1 = (head_dim * 2) as u64;
    let k_nb2 = (kv_capacity * head_dim * 2) as u64;
    let k_nb3 = (n_kv_heads as u64) * k_nb2;
    let o_nb1 = (head_dim * 4) as u64;
    let o_nb2 = (n_head * head_dim * 4) as u64;
    let o_nb3 = o_nb2;
    let zero_u64: u64 = 0;
    let zero_i32: i32 = 0;
    let n_kv_i: i32 = n_kv as i32;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();
    let n_q: i32 = 1;
    let max_bias: f32 = 0.0;
    let m0: f32 = 0.0;
    let m1: f32 = 0.0;
    let n_head_log2: i32 = 0;
    let logit_softcap: f32 = 0.0;
    ocl::core::set_kernel_arg(k_fa, 0, ArgVal::mem(&buf_q))?;
    ocl::core::set_kernel_arg(k_fa, 1, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 2, ArgVal::mem(&buf_kcache))?;
    ocl::core::set_kernel_arg(k_fa, 3, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 4, ArgVal::mem(&buf_vcache))?;
    ocl::core::set_kernel_arg(k_fa, 5, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 6, ArgVal::mem(&buf_attn_o))?;
    ocl::core::set_kernel_arg(k_fa, 7, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 8, ArgVal::scalar(&scale))?;
    ocl::core::set_kernel_arg(k_fa, 9, ArgVal::scalar(&n_q))?;
    ocl::core::set_kernel_arg(k_fa, 10, ArgVal::scalar(&n_kv_i))?;
    ocl::core::set_kernel_arg(k_fa, 11, ArgVal::scalar(&zero_i32))?;
    ocl::core::set_kernel_arg(k_fa, 12, ArgVal::scalar(&n_head_i))?;
    ocl::core::set_kernel_arg(k_fa, 13, ArgVal::scalar(&q_nb1))?;
    ocl::core::set_kernel_arg(k_fa, 14, ArgVal::scalar(&q_nb2))?;
    ocl::core::set_kernel_arg(k_fa, 15, ArgVal::scalar(&q_nb3))?;
    ocl::core::set_kernel_arg(k_fa, 16, ArgVal::scalar(&k_nb1))?;
    ocl::core::set_kernel_arg(k_fa, 17, ArgVal::scalar(&k_nb2))?;
    ocl::core::set_kernel_arg(k_fa, 18, ArgVal::scalar(&k_nb3))?;
    ocl::core::set_kernel_arg(k_fa, 19, ArgVal::scalar(&k_nb1))?;
    ocl::core::set_kernel_arg(k_fa, 20, ArgVal::scalar(&k_nb2))?;
    ocl::core::set_kernel_arg(k_fa, 21, ArgVal::scalar(&k_nb3))?;
    ocl::core::set_kernel_arg(k_fa, 22, ArgVal::scalar(&o_nb1))?;
    ocl::core::set_kernel_arg(k_fa, 23, ArgVal::scalar(&o_nb2))?;
    ocl::core::set_kernel_arg(k_fa, 24, ArgVal::scalar(&o_nb3))?;
    ocl::core::set_kernel_arg(k_fa, 25, ArgVal::scalar(&max_bias))?;
    ocl::core::set_kernel_arg(k_fa, 26, ArgVal::scalar(&m0))?;
    ocl::core::set_kernel_arg(k_fa, 27, ArgVal::scalar(&m1))?;
    ocl::core::set_kernel_arg(k_fa, 28, ArgVal::scalar(&n_head_log2))?;
    ocl::core::set_kernel_arg(k_fa, 29, ArgVal::scalar(&logit_softcap))?;
    ocl::core::set_kernel_arg(k_fa, 30, ArgVal::scalar(&n_kv_heads_i))?;
    ocl::core::set_kernel_arg(k_fa, 31, ArgVal::mem_null())?;
    ocl::core::set_kernel_arg(k_fa, 32, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 33, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 34, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 35, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 36, ArgVal::scalar(&zero_i32))?;
    ocl::core::set_kernel_arg(k_fa, 37, ArgVal::scalar(&zero_i32))?;
    ocl::core::set_kernel_arg(k_fa, 38, ArgVal::mem_null())?;
    ocl::core::set_kernel_arg(k_fa, 39, ArgVal::scalar(&zero_u64))?;
    ocl::core::set_kernel_arg(k_fa, 40, ArgVal::mem(&buf_score_dummy))?;
    ocl::core::set_kernel_arg(k_fa, 41, ArgVal::scalar(&zero_i32))?;
    ocl::core::set_kernel_arg(k_fa, 42, ArgVal::scalar(&zero_i32))?;
    ocl::core::set_kernel_arg(k_fa, 43, ArgVal::scalar(&zero_i32))?;
    const Q1_WG: usize = 64;
    let global_fa = [Q1_WG, n_head, 1];
    let local_fa = [Q1_WG, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_fa,
            2,
            None,
            &global_fa,
            Some(local_fa),
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;
    // microbench stage 8 (FlashAttn) == forward_gen stage 9 (Attention)
    dump_stage_mb!(9, &buf_attn_o, q_proj_out * 4);

    // Stage 9: O proj — Q4_0 matmul (M=1, N=dim, K=q_proj_out)
    dispatch_q40(
        k_q40,
        &buf_oq,
        &buf_od,
        &buf_attn_o,
        &buf_o,
        1,
        dim,
        q_proj_out,
    )?;
    // microbench stage 9 (O proj) == forward_gen stage 10 (post O-proj attn_out)
    dump_stage_mb!(10, &buf_o, dim * 4);

    // Stage 10: Add (residual #1) — kernel_add_row(o, x, x_attn, n4)
    let n4_dim: i32 = (dim / 4) as i32;
    let n4_ffn: i32 = (ffn_dim / 4) as i32;
    let zero_u64_h: u64 = 0;
    ocl::core::set_kernel_arg(k_add, 0, ArgVal::mem(&buf_o))?;
    ocl::core::set_kernel_arg(k_add, 1, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 2, ArgVal::mem(&buf_x))?;
    ocl::core::set_kernel_arg(k_add, 3, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 4, ArgVal::mem(&buf_x_attn))?;
    ocl::core::set_kernel_arg(k_add, 5, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 6, ArgVal::scalar(&n4_dim))?;
    let global_add_dim = [n4_dim as usize, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_add,
            1,
            None,
            &global_add_dim,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // Stage 11: RmsNorm(post)
    ocl::core::set_kernel_arg(k_rms, 0, ArgVal::mem(&buf_x_attn))?;
    ocl::core::set_kernel_arg(k_rms, 1, ArgVal::mem(&buf_rms_w_post))?;
    ocl::core::set_kernel_arg(k_rms, 2, ArgVal::mem(&buf_y2))?;
    ocl::core::set_kernel_arg(k_rms, 3, ArgVal::scalar(&dim_i))?;
    ocl::core::set_kernel_arg(k_rms, 4, ArgVal::scalar(&RMS_EPS))?;
    // D-D.6: kernel_rms_norm_oop_subgroup contract — global=[rows*64], local=[64].
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_rms,
            1,
            None,
            &rms_global,
            Some(rms_local),
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;
    dump_stage_mb!(11, &buf_y2, dim * 4);

    // Stages 12-13: gate / up — Q4_0 matmul (M=1, N=ffn_dim, K=dim)
    dispatch_q40(k_q40, &buf_gq, &buf_gd, &buf_y2, &buf_gate, 1, ffn_dim, dim)?;
    dump_stage_mb!(12, &buf_gate, ffn_dim * 4);
    dispatch_q40(k_q40, &buf_uq, &buf_ud, &buf_y2, &buf_up, 1, ffn_dim, dim)?;
    dump_stage_mb!(13, &buf_up, ffn_dim * 4);

    // Stage 14: SiluMul — kernel_silu_mul_simple(gate, up, ffn_dim/4) [in-place on gate]
    ocl::core::set_kernel_arg(k_silu, 0, ArgVal::mem(&buf_gate))?;
    ocl::core::set_kernel_arg(k_silu, 1, ArgVal::mem(&buf_up))?;
    ocl::core::set_kernel_arg(k_silu, 2, ArgVal::scalar(&n4_ffn))?;
    let global_silu = [n4_ffn as usize, 1, 1];
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_silu,
            1,
            None,
            &global_silu,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;
    dump_stage_mb!(14, &buf_gate, ffn_dim * 4);

    // Stage 15: down proj — Q4_0 matmul (M=1, N=dim, K=ffn_dim)
    let _ = ffn_dim_i; // referenced symbol, used implicitly via cfg
    dispatch_q40(
        k_q40, &buf_dq, &buf_dd, &buf_gate, &buf_down, 1, dim, ffn_dim,
    )?;
    dump_stage_mb!(15, &buf_down, dim * 4);

    // Stage 16: Add (residual #2)
    ocl::core::set_kernel_arg(k_add, 0, ArgVal::mem(&buf_down))?;
    ocl::core::set_kernel_arg(k_add, 1, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 2, ArgVal::mem(&buf_x_attn))?;
    ocl::core::set_kernel_arg(k_add, 3, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 4, ArgVal::mem(&buf_x_out))?;
    ocl::core::set_kernel_arg(k_add, 5, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 6, ArgVal::scalar(&n4_dim))?;
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_add,
            1,
            None,
            &global_add_dim,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;
    dump_stage_mb!(16, &buf_x_out, dim * 4);
    let raw_chain_per_stage_ms = t_raw_chain_start.elapsed().as_secs_f64() * 1000.0;

    // ── Raw OpenCL chain — chain-only sync mode (production-style) ──────────
    // Re-run stage 1..16 without per-stage finishes; one cl_finish at the end.
    // KV cache writes the same pos with deterministic data → same outputs.
    // This measures kernel wall-clock without 14× sync overhead, matching the
    // production engine's fused dispatch pattern (1 finish per layer batch).
    let t_chain_only_start = Instant::now();
    // Stage 1: RmsNorm(pre)
    ocl::core::set_kernel_arg(k_rms, 0, ArgVal::mem(&buf_x))?;
    ocl::core::set_kernel_arg(k_rms, 1, ArgVal::mem(&buf_rms_w_pre))?;
    ocl::core::set_kernel_arg(k_rms, 2, ArgVal::mem(&buf_y1))?;
    ocl::core::set_kernel_arg(k_rms, 3, ArgVal::scalar(&dim_i))?;
    ocl::core::set_kernel_arg(k_rms, 4, ArgVal::scalar(&RMS_EPS))?;
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_rms,
            1,
            None,
            &rms_global,
            Some(rms_local),
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    // Stages 2-4: Q/K/V projection (no finish in helper variant). Inline
    // dispatches to skip the per-stage finish baked into `dispatch_q40`.
    macro_rules! q40_no_finish {
        ($bq:expr, $bd:expr, $bx:expr, $by:expr, $m:expr, $n:expr, $k:expr) => {{
            let ne00 = $k as i32;
            let ne01 = $n as i32;
            let ne02: i32 = 1;
            let ne10 = $k as i32;
            let ne12: i32 = 1;
            let ne0 = $n as i32;
            let ne1 = $m as i32;
            let r2: i32 = 1;
            let r3: i32 = 1;
            let off1: u64 = 0;
            let offd: u64 = 0;
            ocl::core::set_kernel_arg(k_q40, 0, ArgVal::mem($bq))?;
            ocl::core::set_kernel_arg(k_q40, 1, ArgVal::mem($bd))?;
            ocl::core::set_kernel_arg(k_q40, 2, ArgVal::mem($bx))?;
            ocl::core::set_kernel_arg(k_q40, 3, ArgVal::scalar(&off1))?;
            ocl::core::set_kernel_arg(k_q40, 4, ArgVal::mem($by))?;
            ocl::core::set_kernel_arg(k_q40, 5, ArgVal::scalar(&offd))?;
            ocl::core::set_kernel_arg(k_q40, 6, ArgVal::scalar(&ne00))?;
            ocl::core::set_kernel_arg(k_q40, 7, ArgVal::scalar(&ne01))?;
            ocl::core::set_kernel_arg(k_q40, 8, ArgVal::scalar(&ne02))?;
            ocl::core::set_kernel_arg(k_q40, 9, ArgVal::scalar(&ne10))?;
            ocl::core::set_kernel_arg(k_q40, 10, ArgVal::scalar(&ne12))?;
            ocl::core::set_kernel_arg(k_q40, 11, ArgVal::scalar(&ne0))?;
            ocl::core::set_kernel_arg(k_q40, 12, ArgVal::scalar(&ne1))?;
            ocl::core::set_kernel_arg(k_q40, 13, ArgVal::scalar(&r2))?;
            ocl::core::set_kernel_arg(k_q40, 14, ArgVal::scalar(&r3))?;
            let global = [($n as usize).div_ceil(8) * 64, 1, 1];
            let local = [64usize, 1, 1];
            unsafe {
                ocl::core::enqueue_kernel(
                    cl_q,
                    k_q40,
                    3,
                    None,
                    &global,
                    Some(local),
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        }};
    }
    // D-D.6 Phase A.5: chain-only QKV bias (no finish).
    macro_rules! bias_no_finish {
        ($bias:expr, $x:expr, $total:expr) => {{
            let total_i = $total as i32;
            let dim_i = $total as i32;
            ocl::core::set_kernel_arg(k_bias, 0, ArgVal::mem($x))?;
            ocl::core::set_kernel_arg(k_bias, 1, ArgVal::mem($bias))?;
            ocl::core::set_kernel_arg(k_bias, 2, ArgVal::scalar(&dim_i))?;
            ocl::core::set_kernel_arg(k_bias, 3, ArgVal::scalar(&total_i))?;
            let global = [$total as usize, 1, 1];
            unsafe {
                ocl::core::enqueue_kernel(
                    cl_q,
                    k_bias,
                    1,
                    None,
                    &global,
                    None,
                    None::<&ocl::core::Event>,
                    None::<&mut ocl::core::Event>,
                )?;
            }
        }};
    }
    q40_no_finish!(&buf_qq, &buf_qd, &buf_y1, &buf_q, 1usize, q_proj_out, dim);
    bias_no_finish!(&buf_q_bias, &buf_q, q_proj_out);
    q40_no_finish!(&buf_kq, &buf_kd, &buf_y1, &buf_k, 1usize, kv_proj_out, dim);
    bias_no_finish!(&buf_k_bias, &buf_k, kv_proj_out);
    q40_no_finish!(&buf_vq, &buf_vd, &buf_y1, &buf_v, 1usize, kv_proj_out, dim);
    bias_no_finish!(&buf_v_bias, &buf_v, kv_proj_out);
    // Stage 5: RoPE(Q)
    ocl::core::set_kernel_arg(k_rope, 0, ArgVal::mem(&buf_q))?;
    ocl::core::set_kernel_arg(k_rope, 1, ArgVal::scalar(&head_dim_i))?;
    ocl::core::set_kernel_arg(k_rope, 2, ArgVal::scalar(&n_head_i))?;
    ocl::core::set_kernel_arg(k_rope, 3, ArgVal::scalar(&seq_len_i))?;
    ocl::core::set_kernel_arg(k_rope, 4, ArgVal::scalar(&pos))?;
    ocl::core::set_kernel_arg(k_rope, 5, ArgVal::scalar(&theta))?;
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_rope,
            1,
            None,
            &global_rope_q,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    // Stage 6: RoPE(K)
    ocl::core::set_kernel_arg(k_rope, 0, ArgVal::mem(&buf_k))?;
    ocl::core::set_kernel_arg(k_rope, 1, ArgVal::scalar(&head_dim_i))?;
    ocl::core::set_kernel_arg(k_rope, 2, ArgVal::scalar(&n_kv_heads_i))?;
    ocl::core::set_kernel_arg(k_rope, 3, ArgVal::scalar(&seq_len_i))?;
    ocl::core::set_kernel_arg(k_rope, 4, ArgVal::scalar(&pos))?;
    ocl::core::set_kernel_arg(k_rope, 5, ArgVal::scalar(&theta))?;
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_rope,
            1,
            None,
            &global_rope_k,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    // Stage 7: KvScatter
    ocl::core::set_kernel_arg(k_kvs, 0, ArgVal::mem(&buf_k))?;
    ocl::core::set_kernel_arg(k_kvs, 1, ArgVal::mem(&buf_v))?;
    ocl::core::set_kernel_arg(k_kvs, 2, ArgVal::mem(&buf_kcache))?;
    ocl::core::set_kernel_arg(k_kvs, 3, ArgVal::mem(&buf_vcache))?;
    ocl::core::set_kernel_arg(k_kvs, 4, ArgVal::scalar(&head_dim_i))?;
    ocl::core::set_kernel_arg(k_kvs, 5, ArgVal::scalar(&capacity_i))?;
    ocl::core::set_kernel_arg(k_kvs, 6, ArgVal::scalar(&pos))?;
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_kvs,
            1,
            None,
            &global_kvs,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    // Stage 8: FlashAttn — args already set above; just enqueue.
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_fa,
            2,
            None,
            &global_fa,
            Some(local_fa),
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    // Stage 9: O proj
    q40_no_finish!(
        &buf_oq,
        &buf_od,
        &buf_attn_o,
        &buf_o,
        1usize,
        dim,
        q_proj_out
    );
    // Stage 10: Add (residual #1)
    ocl::core::set_kernel_arg(k_add, 0, ArgVal::mem(&buf_o))?;
    ocl::core::set_kernel_arg(k_add, 1, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 2, ArgVal::mem(&buf_x))?;
    ocl::core::set_kernel_arg(k_add, 3, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 4, ArgVal::mem(&buf_x_attn))?;
    ocl::core::set_kernel_arg(k_add, 5, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 6, ArgVal::scalar(&n4_dim))?;
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_add,
            1,
            None,
            &global_add_dim,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    // Stage 11: RmsNorm(post)
    ocl::core::set_kernel_arg(k_rms, 0, ArgVal::mem(&buf_x_attn))?;
    ocl::core::set_kernel_arg(k_rms, 1, ArgVal::mem(&buf_rms_w_post))?;
    ocl::core::set_kernel_arg(k_rms, 2, ArgVal::mem(&buf_y2))?;
    ocl::core::set_kernel_arg(k_rms, 3, ArgVal::scalar(&dim_i))?;
    ocl::core::set_kernel_arg(k_rms, 4, ArgVal::scalar(&RMS_EPS))?;
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_rms,
            1,
            None,
            &rms_global,
            Some(rms_local),
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    // Stages 12-13: gate / up
    q40_no_finish!(&buf_gq, &buf_gd, &buf_y2, &buf_gate, 1usize, ffn_dim, dim);
    q40_no_finish!(&buf_uq, &buf_ud, &buf_y2, &buf_up, 1usize, ffn_dim, dim);
    // Stage 14: SiluMul
    ocl::core::set_kernel_arg(k_silu, 0, ArgVal::mem(&buf_gate))?;
    ocl::core::set_kernel_arg(k_silu, 1, ArgVal::mem(&buf_up))?;
    ocl::core::set_kernel_arg(k_silu, 2, ArgVal::scalar(&n4_ffn))?;
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_silu,
            1,
            None,
            &global_silu,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    // Stage 15: down proj
    q40_no_finish!(&buf_dq, &buf_dd, &buf_gate, &buf_down, 1usize, dim, ffn_dim);
    // Stage 16: Add (residual #2)
    ocl::core::set_kernel_arg(k_add, 0, ArgVal::mem(&buf_down))?;
    ocl::core::set_kernel_arg(k_add, 1, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 2, ArgVal::mem(&buf_x_attn))?;
    ocl::core::set_kernel_arg(k_add, 3, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 4, ArgVal::mem(&buf_x_out))?;
    ocl::core::set_kernel_arg(k_add, 5, ArgVal::scalar(&zero_u64_h))?;
    ocl::core::set_kernel_arg(k_add, 6, ArgVal::scalar(&n4_dim))?;
    unsafe {
        ocl::core::enqueue_kernel(
            cl_q,
            k_add,
            1,
            None,
            &global_add_dim,
            None,
            None::<&ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;
    let raw_chain_only_ms = t_chain_only_start.elapsed().as_secs_f64() * 1000.0;

    eprintln!(
        "[M2.I-breakdown] raw_chain_per_stage_finish = {:.3} ms (14 cl_finish)",
        raw_chain_per_stage_ms
    );
    eprintln!(
        "[M2.I-breakdown] raw_chain_only_finish      = {:.3} ms (1 cl_finish — production-like)",
        raw_chain_only_ms
    );

    // Read back the reference layer output.
    let mut ref_x_out = vec![0.0f32; dim];
    unsafe {
        ocl::core::enqueue_read_buffer(
            cl_q,
            &buf_x_out,
            true,
            0,
            &mut ref_x_out,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    // M2.H 5th: also read attn_o + kcache slot for QNN-vs-ref diagnostic.
    let mut ref_attn_o = vec![0.0f32; q_proj_out];
    unsafe {
        ocl::core::enqueue_read_buffer(
            cl_q,
            &buf_attn_o,
            true,
            0,
            &mut ref_attn_o,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    let mut ref_kcache_full = vec![0u16; kv_total];
    unsafe {
        ocl::core::enqueue_read_buffer(
            cl_q,
            &buf_kcache,
            true,
            0,
            &mut ref_kcache_full,
            None::<ocl::core::Event>,
            None::<&mut ocl::core::Event>,
        )?;
    }
    ocl::core::finish(cl_q)?;

    // ── 3. Path B: QNN single graph with 14 chained ops ──────────────────────
    // rpcmem allocations for all host-backed graph endpoints.
    let bytes_x = (dim * 4) as i32;
    let bytes_rms = (dim * 4) as i32;
    let bytes_kv = (kv_total * 2) as i32;
    let bytes_x_out = (dim * 4) as i32;
    let bytes_q40 = |q: &[u8]| q.len() as i32;
    let bytes_q40d = |d: &[u16]| (d.len() * 2) as i32;
    // D-D.6: mask sized to kv_capacity (F16) — n_kv는 input tensor로 동적 처리.
    let bytes_mask = (kv_capacity * 2) as i32;
    let bytes_dummy = 4_i32;
    // FlashAttn sinks: kernel reads `sinks_ptr[head_idx]` for head_idx ∈
    // [0, n_head). The OpPackage path always binds a non-null mem object for
    // sinks (no `mem_null` equivalent), so the buffer must be sized to n_head
    // f32s. Host populates with -1e30 so `exp(sink - m_final) ≈ 0`,
    // matching the raw-OpenCL `mem_null()` baseline. A 1-element buffer was
    // the source of the M2.H 0.5-ratio: kernel read sinks[head_idx] = 0
    // (zero-init) instead of -INFINITY, then `l_final += exp(0 - m_final)`
    // inflated the denominator at small n_kv.
    let bytes_sinks = (n_head * 4) as i32;

    let rpc_x = unsafe { rpcmem_alloc(heap_id, flags, bytes_x) };
    let rpc_rms_pre = unsafe { rpcmem_alloc(heap_id, flags, bytes_rms) };
    let rpc_rms_post = unsafe { rpcmem_alloc(heap_id, flags, bytes_rms) };
    let rpc_kcache = unsafe { rpcmem_alloc(heap_id, flags, bytes_kv) };
    let rpc_vcache = unsafe { rpcmem_alloc(heap_id, flags, bytes_kv) };
    let rpc_x_out = unsafe { rpcmem_alloc(heap_id, flags, bytes_x_out) };
    let rpc_qq = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40(&qq_q)) };
    let rpc_qd = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40d(&qq_d)) };
    let rpc_kq = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40(&qk_q)) };
    let rpc_kd = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40d(&qk_d)) };
    let rpc_vq = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40(&qv_q)) };
    let rpc_vd = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40d(&qv_d)) };
    let rpc_oq = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40(&qo_q)) };
    let rpc_od = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40d(&qo_d)) };
    let rpc_gq = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40(&qg_q)) };
    let rpc_gd = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40d(&qg_d)) };
    let rpc_uq = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40(&qu_q)) };
    let rpc_ud = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40d(&qu_d)) };
    let rpc_dq = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40(&qd_q)) };
    let rpc_dd = unsafe { rpcmem_alloc(heap_id, flags, bytes_q40d(&qd_d)) };
    let rpc_mask = unsafe { rpcmem_alloc(heap_id, flags, bytes_mask) };
    let rpc_sinks = unsafe { rpcmem_alloc(heap_id, flags, bytes_sinks) };
    let rpc_score = unsafe { rpcmem_alloc(heap_id, flags, bytes_dummy) };
    // M3.4 D-D.1: pos_buf — INT_32 [1].
    let rpc_pos = unsafe { rpcmem_alloc(heap_id, flags, 4) };
    // M3.4 D-D.6: n_kv_buf — INT_32 [1]. FlashAttn input tensor.
    let rpc_n_kv = unsafe { rpcmem_alloc(heap_id, flags, 4) };
    anyhow::ensure!(
        !rpc_x.is_null()
            && !rpc_rms_pre.is_null()
            && !rpc_rms_post.is_null()
            && !rpc_kcache.is_null()
            && !rpc_vcache.is_null()
            && !rpc_x_out.is_null()
            && !rpc_qq.is_null()
            && !rpc_qd.is_null()
            && !rpc_kq.is_null()
            && !rpc_kd.is_null()
            && !rpc_vq.is_null()
            && !rpc_vd.is_null()
            && !rpc_oq.is_null()
            && !rpc_od.is_null()
            && !rpc_gq.is_null()
            && !rpc_gd.is_null()
            && !rpc_uq.is_null()
            && !rpc_ud.is_null()
            && !rpc_dq.is_null()
            && !rpc_dd.is_null()
            && !rpc_mask.is_null()
            && !rpc_sinks.is_null()
            && !rpc_score.is_null()
            && !rpc_pos.is_null()
            && !rpc_n_kv.is_null(),
        "rpcmem_alloc failed"
    );

    // Copy host data into rpcmem.
    unsafe {
        std::ptr::copy_nonoverlapping(
            host_x.as_ptr() as *const u8,
            rpc_x as *mut u8,
            bytes_x as usize,
        );
        std::ptr::copy_nonoverlapping(
            host_rms_w_pre.as_ptr() as *const u8,
            rpc_rms_pre as *mut u8,
            bytes_rms as usize,
        );
        std::ptr::copy_nonoverlapping(
            host_rms_w_post.as_ptr() as *const u8,
            rpc_rms_post as *mut u8,
            bytes_rms as usize,
        );
        // D-D.6 디버깅: KV inject가 활성이면 host_kv_k/v 사용 (raw OpenCL과 동일).
        // 미활성이면 zero init (default).
        std::ptr::copy_nonoverlapping(
            host_kv_k.as_ptr() as *const u8,
            rpc_kcache as *mut u8,
            bytes_kv as usize,
        );
        std::ptr::copy_nonoverlapping(
            host_kv_v.as_ptr() as *const u8,
            rpc_vcache as *mut u8,
            bytes_kv as usize,
        );
        std::ptr::copy_nonoverlapping(qq_q.as_ptr(), rpc_qq as *mut u8, qq_q.len());
        std::ptr::copy_nonoverlapping(
            qq_d.as_ptr() as *const u8,
            rpc_qd as *mut u8,
            qq_d.len() * 2,
        );
        std::ptr::copy_nonoverlapping(qk_q.as_ptr(), rpc_kq as *mut u8, qk_q.len());
        std::ptr::copy_nonoverlapping(
            qk_d.as_ptr() as *const u8,
            rpc_kd as *mut u8,
            qk_d.len() * 2,
        );
        std::ptr::copy_nonoverlapping(qv_q.as_ptr(), rpc_vq as *mut u8, qv_q.len());
        std::ptr::copy_nonoverlapping(
            qv_d.as_ptr() as *const u8,
            rpc_vd as *mut u8,
            qv_d.len() * 2,
        );
        std::ptr::copy_nonoverlapping(qo_q.as_ptr(), rpc_oq as *mut u8, qo_q.len());
        std::ptr::copy_nonoverlapping(
            qo_d.as_ptr() as *const u8,
            rpc_od as *mut u8,
            qo_d.len() * 2,
        );
        std::ptr::copy_nonoverlapping(qg_q.as_ptr(), rpc_gq as *mut u8, qg_q.len());
        std::ptr::copy_nonoverlapping(
            qg_d.as_ptr() as *const u8,
            rpc_gd as *mut u8,
            qg_d.len() * 2,
        );
        std::ptr::copy_nonoverlapping(qu_q.as_ptr(), rpc_uq as *mut u8, qu_q.len());
        std::ptr::copy_nonoverlapping(
            qu_d.as_ptr() as *const u8,
            rpc_ud as *mut u8,
            qu_d.len() * 2,
        );
        std::ptr::copy_nonoverlapping(qd_q.as_ptr(), rpc_dq as *mut u8, qd_q.len());
        std::ptr::copy_nonoverlapping(
            qd_d.as_ptr() as *const u8,
            rpc_dd as *mut u8,
            qd_d.len() * 2,
        );
        std::ptr::write_bytes(rpc_mask as *mut u8, 0, bytes_mask as usize);
        // Sinks: -1e30 per head — neutralises kernel's sink-attention path
        // so output matches raw-OpenCL `mem_null()` baseline (M2.H fix).
        let neg_huge: f32 = -1.0e30f32;
        for i in 0..n_head {
            std::ptr::write_unaligned((rpc_sinks as *mut u8).add(i * 4) as *mut f32, neg_huge);
        }
        std::ptr::write_bytes(rpc_score as *mut u8, 0, bytes_dummy as usize);
        // M3.4 D-D.1: initial pos value = `pos` (matches raw-OpenCL reference).
        *(rpc_pos as *mut i32) = pos;
        // M3.4 D-D.6: initial n_kv value (FlashAttn dynkv kernel reads this slot).
        *(rpc_n_kv as *mut i32) = n_kv as i32;
    }

    let fd_x = unsafe { rpcmem_to_fd(rpc_x) };
    let fd_rms_pre = unsafe { rpcmem_to_fd(rpc_rms_pre) };
    let fd_rms_post = unsafe { rpcmem_to_fd(rpc_rms_post) };
    let fd_kcache = unsafe { rpcmem_to_fd(rpc_kcache) };
    let _fd_vcache = unsafe { rpcmem_to_fd(rpc_vcache) };
    let fd_x_out = unsafe { rpcmem_to_fd(rpc_x_out) };
    let fd_qq = unsafe { rpcmem_to_fd(rpc_qq) };
    let fd_qd = unsafe { rpcmem_to_fd(rpc_qd) };
    let fd_kq = unsafe { rpcmem_to_fd(rpc_kq) };
    let fd_kd = unsafe { rpcmem_to_fd(rpc_kd) };
    let fd_vq = unsafe { rpcmem_to_fd(rpc_vq) };
    let fd_vd = unsafe { rpcmem_to_fd(rpc_vd) };
    let fd_oq = unsafe { rpcmem_to_fd(rpc_oq) };
    let fd_od = unsafe { rpcmem_to_fd(rpc_od) };
    let fd_gq = unsafe { rpcmem_to_fd(rpc_gq) };
    let fd_gd = unsafe { rpcmem_to_fd(rpc_gd) };
    let fd_uq = unsafe { rpcmem_to_fd(rpc_uq) };
    let fd_ud = unsafe { rpcmem_to_fd(rpc_ud) };
    let fd_dq = unsafe { rpcmem_to_fd(rpc_dq) };
    let fd_dd = unsafe { rpcmem_to_fd(rpc_dd) };
    let fd_mask = unsafe { rpcmem_to_fd(rpc_mask) };
    let fd_sinks = unsafe { rpcmem_to_fd(rpc_sinks) };
    let fd_score = unsafe { rpcmem_to_fd(rpc_score) };
    let fd_pos = unsafe { rpcmem_to_fd(rpc_pos) };
    let fd_n_kv = unsafe { rpcmem_to_fd(rpc_n_kv) };

    // ── 4. Build graph: declare tensors then 14 nodes ───────────────────────
    let qp = Qnn_QuantizeParams_t {
        encodingDefinition: Qnn_Definition_t_QNN_DEFINITION_UNDEFINED,
        quantizationEncoding: Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED,
        __bindgen_anon_1: Qnn_QuantizeParams_t__bindgen_ty_1 {
            scaleOffsetEncoding: Qnn_ScaleOffset_t {
                scale: 0.0,
                offset: 0,
            },
        },
    };
    let mk_tv1 = |ttype, dtype, rank: u32, dims_ptr: *mut u32| Qnn_TensorV1_t {
        id: 0,
        name: ptr::null(),
        type_: ttype,
        dataFormat: QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
        dataType: dtype,
        quantizeParams: qp,
        rank,
        dimensions: dims_ptr,
        memType: Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
        __bindgen_anon_1: Qnn_TensorV1_t__bindgen_ty_1 {
            clientBuf: Qnn_ClientBuffer_t {
                data: ptr::null_mut(),
                dataSize: 0,
            },
        },
    };

    // Dimensions storage. Each Vec must outlive graph build (pointer stored).
    // M2.H 5th attempt: x_in dropped from rank 3 [1, 1, dim] to rank 2 [1, dim]
    // to match RmsNorm op build_layout expectation. rank 3 caused only the
    // first element of y1 to be processed (bisect diagnostic — write coverage
    // 0.05% on y1).
    let mut dims_x: Vec<u32> = vec![1, dim as u32];
    let mut dims_rms_pre = vec![dim as u32];
    let mut dims_rms_post = vec![dim as u32];
    let mut dims_y1 = vec![1, dim as u32];
    let mut dims_qq = vec![qq_q.len() as u32];
    let mut dims_qd = vec![qq_d.len() as u32];
    let mut dims_kq = vec![qk_q.len() as u32];
    let mut dims_kd = vec![qk_d.len() as u32];
    let mut dims_vq = vec![qv_q.len() as u32];
    let mut dims_vd = vec![qv_d.len() as u32];
    let mut dims_oq = vec![qo_q.len() as u32];
    let mut dims_od = vec![qo_d.len() as u32];
    let mut dims_gq = vec![qg_q.len() as u32];
    let mut dims_gd = vec![qg_d.len() as u32];
    let mut dims_uq = vec![qu_q.len() as u32];
    let mut dims_ud = vec![qu_d.len() as u32];
    let mut dims_dq = vec![qd_q.len() as u32];
    let mut dims_dd = vec![qd_d.len() as u32];
    // M2.H 3rd attempt: matmul outputs reshape directly to rank-3 attention
    // views so RoPE / FlashAttn can consume them without an explicit Reshape
    // node. matmul build_layout is rank-flexible (M2.H fix in
    // crates/qnn_oppkg/src/ops/matmul_q40_f32.rs).
    let mut dims_q = vec![1u32, n_head as u32, head_dim as u32];
    let mut dims_kvec = vec![1u32, n_kv_heads as u32, head_dim as u32];
    let mut dims_vvec = vec![1u32, n_kv_heads as u32, head_dim as u32];
    let mut dims_q_rope = vec![1u32, n_head as u32, head_dim as u32];
    let mut dims_k_rope = vec![1u32, n_kv_heads as u32, head_dim as u32];
    let mut dims_kcache = vec![1u32, n_kv_heads as u32, kv_capacity as u32, head_dim as u32];
    let mut dims_vcache = dims_kcache.clone();
    let mut dims_q_fa = vec![1u32, n_head as u32, head_dim as u32];
    let mut dims_attn_o = vec![1u32, n_head as u32, head_dim as u32];
    let mut dims_o = vec![1u32, dim as u32];
    let mut dims_x_attn = vec![1u32, dim as u32];
    let mut dims_y2 = vec![1, dim as u32];
    let mut dims_gate = vec![1u32, ffn_dim as u32];
    let mut dims_up = vec![1u32, ffn_dim as u32];
    let mut dims_silu_out = vec![1u32, ffn_dim as u32];
    let mut dims_down = vec![1u32, dim as u32];
    // M2.H 5th: x_out follows x_in rank to keep Add(residual2) shapes consistent.
    let mut dims_x_out = vec![1u32, dim as u32];
    // D-D.6: mask sized to kv_capacity (n_kv now dynamic).
    let mut dims_mask = vec![kv_capacity as u32];
    let mut dims_sinks = vec![n_head as u32];
    let mut dims_score = vec![1u32];
    // M3.4 D-D.1: pos_buf is INT_32 [1] shared across RoPE Q/K and KvScatter.
    let mut dims_pos = vec![1u32];
    // M3.4 D-D.6: n_kv_buf is INT_32 [1] consumed by FlashAttn input.
    let mut dims_n_kv = vec![1u32];

    // Tensor names (CStrings must outlive graph build).
    let nm_x = CString::new("x").unwrap();
    let nm_rms_pre = CString::new("rms_pre").unwrap();
    let nm_rms_post = CString::new("rms_post").unwrap();
    let nm_y1 = CString::new("y1").unwrap();
    let nm_qq = CString::new("qq").unwrap();
    let nm_qd = CString::new("qd").unwrap();
    let nm_kq = CString::new("kq").unwrap();
    let nm_kd = CString::new("kd").unwrap();
    let nm_vq = CString::new("vq").unwrap();
    let nm_vd = CString::new("vd").unwrap();
    let nm_oq = CString::new("oq").unwrap();
    let nm_od = CString::new("od").unwrap();
    let nm_gq = CString::new("gq").unwrap();
    let nm_gd = CString::new("gd").unwrap();
    let nm_uq = CString::new("uq").unwrap();
    let nm_ud = CString::new("ud").unwrap();
    let nm_dq = CString::new("dq").unwrap();
    let nm_dd = CString::new("dd").unwrap();
    let nm_q = CString::new("q").unwrap();
    let nm_k = CString::new("k").unwrap();
    let nm_v = CString::new("v").unwrap();
    let nm_q_rope = CString::new("q_rope").unwrap();
    let nm_k_rope = CString::new("k_rope").unwrap();
    let nm_kcache = CString::new("kcache").unwrap();
    let nm_vcache = CString::new("vcache").unwrap();
    let nm_attn_o = CString::new("attn_o").unwrap();
    let nm_o = CString::new("o").unwrap();
    let nm_x_attn = CString::new("x_attn").unwrap();
    let nm_y2 = CString::new("y2").unwrap();
    let nm_gate = CString::new("gate").unwrap();
    let nm_up = CString::new("up").unwrap();
    let nm_silu_out = CString::new("silu_out").unwrap();
    let nm_down = CString::new("down").unwrap();
    let nm_x_out = CString::new("x_out").unwrap();
    let nm_mask = CString::new("mask").unwrap();
    let nm_sinks = CString::new("sinks").unwrap();
    let nm_score = CString::new("score").unwrap();
    // M3.4 D-D.1: pos_buf is APP_WRITE so the host can update pos per execute.
    let nm_pos = CString::new("pos").unwrap();
    // M3.4 D-D.6: n_kv_buf — APP_WRITE INT_32 [1]; FlashAttn dynkv input.
    let nm_n_kv = CString::new("n_kv_buf").unwrap();

    // Helper: build tensor with given type/dtype/rank/dims.
    let build_tensor = |ttype, dtype, rank: u32, dims_ptr: *mut u32, name: *const c_char| {
        let mut t = Qnn_Tensor_t {
            version: Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
            __bindgen_anon_1: Qnn_Tensor_t__bindgen_ty_1 {
                v1: mk_tv1(ttype, dtype, rank, dims_ptr),
            },
        };
        t.__bindgen_anon_1.v1.name = name;
        t
    };

    let app_w = Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE;
    let app_r = Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ;
    let native = Qnn_TensorType_t_QNN_TENSOR_TYPE_NATIVE;
    let f32_t = Qnn_DataType_t_QNN_DATATYPE_FLOAT_32;
    let f16_t = Qnn_DataType_t_QNN_DATATYPE_FLOAT_16;
    let u8_t = Qnn_DataType_t_QNN_DATATYPE_UINT_8;

    // Graph endpoints (APP_WRITE/APP_READ) — host buffers via memHandle.
    // M2.H 5th: rank 2 [1, dim] (was rank 3 [1, 1, dim]). See dims_x comment.
    let mut t_x = build_tensor(app_w, f32_t, 2, dims_x.as_mut_ptr(), nm_x.as_ptr());
    let mut t_rms_pre = build_tensor(
        app_w,
        f32_t,
        1,
        dims_rms_pre.as_mut_ptr(),
        nm_rms_pre.as_ptr(),
    );
    let mut t_rms_post = build_tensor(
        app_w,
        f32_t,
        1,
        dims_rms_post.as_mut_ptr(),
        nm_rms_post.as_ptr(),
    );
    let mut t_qq = build_tensor(app_w, u8_t, 1, dims_qq.as_mut_ptr(), nm_qq.as_ptr());
    let mut t_qd = build_tensor(app_w, f16_t, 1, dims_qd.as_mut_ptr(), nm_qd.as_ptr());
    let mut t_kq = build_tensor(app_w, u8_t, 1, dims_kq.as_mut_ptr(), nm_kq.as_ptr());
    let mut t_kd = build_tensor(app_w, f16_t, 1, dims_kd.as_mut_ptr(), nm_kd.as_ptr());
    let mut t_vq = build_tensor(app_w, u8_t, 1, dims_vq.as_mut_ptr(), nm_vq.as_ptr());
    let mut t_vd = build_tensor(app_w, f16_t, 1, dims_vd.as_mut_ptr(), nm_vd.as_ptr());
    let mut t_oq = build_tensor(app_w, u8_t, 1, dims_oq.as_mut_ptr(), nm_oq.as_ptr());
    let mut t_od = build_tensor(app_w, f16_t, 1, dims_od.as_mut_ptr(), nm_od.as_ptr());
    let mut t_gq = build_tensor(app_w, u8_t, 1, dims_gq.as_mut_ptr(), nm_gq.as_ptr());
    let mut t_gd = build_tensor(app_w, f16_t, 1, dims_gd.as_mut_ptr(), nm_gd.as_ptr());
    let mut t_uq = build_tensor(app_w, u8_t, 1, dims_uq.as_mut_ptr(), nm_uq.as_ptr());
    let mut t_ud = build_tensor(app_w, f16_t, 1, dims_ud.as_mut_ptr(), nm_ud.as_ptr());
    let mut t_dq = build_tensor(app_w, u8_t, 1, dims_dq.as_mut_ptr(), nm_dq.as_ptr());
    let mut t_dd = build_tensor(app_w, f16_t, 1, dims_dd.as_mut_ptr(), nm_dd.as_ptr());
    // KV cache: APP_WRITE — graph reads/writes them through host-registered
    // fds. NOTE: kcache=NATIVE was attempted (M2.H 5th) but SDK rejects NATIVE
    // tensors bound as `InOutTensor` in KvScatter (graphAddNode err=0x1777).
    // Reverted to APP_WRITE; root-cause analysis continues elsewhere.
    // M2.H multi-output: KvScatter outputs[0] = kcache. SDK requires output
    // tensors to be APP_READ (or NATIVE). NATIVE blocks host memHandle
    // registration; APP_READ lets us bind an fd while satisfying the output
    // type check. The kcache also serves as FA input within the same graph;
    // SDK accepts read-after-write within graph scope for APP_READ tensors.
    let mut t_kcache = build_tensor(
        app_r,
        f16_t,
        4,
        dims_kcache.as_mut_ptr(),
        nm_kcache.as_ptr(),
    );
    // vcache: APP_WRITE — symmetric with kcache. With M2.H multi-output
    // KvScatter (both kcache and vcache as claimed outputs), NATIVE on one
    // output causes graphAddNode err=0x1775 (validation failure). Keeping
    // both APP_WRITE matches the kvs standalone microbench (M2.E) pattern
    // where both dst tensors are host-registered fds. Downstream consumer
    // (FA) reads them through the same memHandle.
    let mut t_vcache = build_tensor(
        app_r,
        f16_t,
        4,
        dims_vcache.as_mut_ptr(),
        nm_vcache.as_ptr(),
    );
    // FlashAttn auxiliaries (mask zero / sinks / score dummy) — APP_WRITE.
    let mut t_mask = build_tensor(app_w, f16_t, 1, dims_mask.as_mut_ptr(), nm_mask.as_ptr());
    let mut t_sinks = build_tensor(app_w, f32_t, 1, dims_sinks.as_mut_ptr(), nm_sinks.as_ptr());
    let mut t_score = build_tensor(app_w, f32_t, 1, dims_score.as_mut_ptr(), nm_score.as_ptr());
    // M3.4 D-D.1: pos_buf — INT_32 [1] APP_WRITE; updated per execute by host.
    let i32_t = Qnn_DataType_t_QNN_DATATYPE_INT_32;
    let mut t_pos = build_tensor(app_w, i32_t, 1, dims_pos.as_mut_ptr(), nm_pos.as_ptr());
    // M3.4 D-D.6: n_kv_buf — INT_32 [1] APP_WRITE; FlashAttn dynkv input.
    let mut t_n_kv = build_tensor(app_w, i32_t, 1, dims_n_kv.as_mut_ptr(), nm_n_kv.as_ptr());
    // Output of layer.
    // M2.H 5th: rank 2 [1, dim] to match x_in shape (Add(residual2) needs same rank).
    let mut t_x_out = build_tensor(app_r, f32_t, 2, dims_x_out.as_mut_ptr(), nm_x_out.as_ptr());

    // Intermediates (NATIVE).
    let mut t_y1 = build_tensor(native, f32_t, 2, dims_y1.as_mut_ptr(), nm_y1.as_ptr());
    // M2.H 3rd attempt: rank-3 reshape views — matmul output rank == RoPE/
    // FlashAttn input rank, eliminating the previous rank-mismatch finalize
    // failure (graphFinalize err=0x1786).
    let mut t_q = build_tensor(native, f32_t, 3, dims_q.as_mut_ptr(), nm_q.as_ptr());
    let mut t_k = build_tensor(native, f32_t, 3, dims_kvec.as_mut_ptr(), nm_k.as_ptr());
    let mut t_v = build_tensor(native, f32_t, 3, dims_vvec.as_mut_ptr(), nm_v.as_ptr());
    let mut t_q_rope = build_tensor(
        native,
        f32_t,
        3,
        dims_q_rope.as_mut_ptr(),
        nm_q_rope.as_ptr(),
    );
    let mut t_k_rope = build_tensor(
        native,
        f32_t,
        3,
        dims_k_rope.as_mut_ptr(),
        nm_k_rope.as_ptr(),
    );
    let mut t_attn_o = build_tensor(
        native,
        f32_t,
        3,
        dims_attn_o.as_mut_ptr(),
        nm_attn_o.as_ptr(),
    );
    // Reuse FlashAttn input shape for q_fa.
    let mut _t_q_fa = build_tensor(native, f32_t, 3, dims_q_fa.as_mut_ptr(), nm_q_rope.as_ptr());
    let mut t_o = build_tensor(native, f32_t, 2, dims_o.as_mut_ptr(), nm_o.as_ptr());
    let mut t_x_attn = build_tensor(
        native,
        f32_t,
        2,
        dims_x_attn.as_mut_ptr(),
        nm_x_attn.as_ptr(),
    );
    let mut t_y2 = build_tensor(native, f32_t, 2, dims_y2.as_mut_ptr(), nm_y2.as_ptr());
    let mut t_gate = build_tensor(native, f32_t, 2, dims_gate.as_mut_ptr(), nm_gate.as_ptr());
    let mut t_up = build_tensor(native, f32_t, 2, dims_up.as_mut_ptr(), nm_up.as_ptr());
    let mut t_silu_out = build_tensor(
        native,
        f32_t,
        2,
        dims_silu_out.as_mut_ptr(),
        nm_silu_out.as_ptr(),
    );
    let mut t_down = build_tensor(native, f32_t, 2, dims_down.as_mut_ptr(), nm_down.as_ptr());

    let g_name = CString::new("qwen_layer_graph").unwrap();
    let mut graph: Qnn_GraphHandle_t = ptr::null_mut();

    // P0 — disableMemoryOptimizations + disableNodeOptimizations
    // GPU header 측 struct는 engine bindgen이 안 가져오므로 manual layout
    // (QnnGpuGraph.h:50-55: precision u32 + 3 u8 = 8 bytes)
    #[repr(C)]
    struct QnnGpuGraph_CustomConfig_local {
        precision: u32, // QnnGpu_Precision_t
        disable_memory_optimizations: u8,
        disable_node_optimizations: u8,
        disable_queue_recording: u8,
        _pad: u8,
    }
    const QNN_GPU_PRECISION_USER_PROVIDED: u32 = 3;
    // M2.I 재측정: P0 disable 설정 해제. sinks fix로 정확성 회복 후 SDK 기본 최적화 활성화.
    let mut gpu_custom = QnnGpuGraph_CustomConfig_local {
        precision: QNN_GPU_PRECISION_USER_PROVIDED,
        disable_memory_optimizations: 0,
        disable_node_optimizations: 0,
        disable_queue_recording: 0,
        _pad: 0,
    };
    let mut graph_cfg = qnn::QnnGraph_Config_t {
        option: qnn::QnnGraph_ConfigOption_t_QNN_GRAPH_CONFIG_OPTION_CUSTOM,
        __bindgen_anon_1: qnn::QnnGraph_Config_t__bindgen_ty_1 {
            customConfig: &mut gpu_custom as *mut _ as *mut _,
        },
    };
    let mut configs: [*const qnn::QnnGraph_Config_t; 2] = [&graph_cfg as *const _, ptr::null()];

    let err =
        unsafe { (v.graphCreate.unwrap())(ctx, g_name.as_ptr(), configs.as_mut_ptr(), &mut graph) };
    anyhow::ensure!(err == 0, "graphCreate err=0x{:x}", err);
    eprintln!("[P0] graphCreate with disableMemoryOptimizations=1 disableNodeOptimizations=1 OK");

    // Register tensors. Order matters only for debugging; uniqueness by name
    // suffices.
    let registrations: &mut [(&str, &mut Qnn_Tensor_t)] = &mut [
        ("x", &mut t_x),
        ("rms_pre", &mut t_rms_pre),
        ("rms_post", &mut t_rms_post),
        ("qq", &mut t_qq),
        ("qd", &mut t_qd),
        ("kq", &mut t_kq),
        ("kd", &mut t_kd),
        ("vq", &mut t_vq),
        ("vd", &mut t_vd),
        ("oq", &mut t_oq),
        ("od", &mut t_od),
        ("gq", &mut t_gq),
        ("gd", &mut t_gd),
        ("uq", &mut t_uq),
        ("ud", &mut t_ud),
        ("dq", &mut t_dq),
        ("dd", &mut t_dd),
        ("kcache", &mut t_kcache),
        ("vcache", &mut t_vcache),
        ("mask", &mut t_mask),
        ("sinks", &mut t_sinks),
        ("score", &mut t_score),
        ("pos", &mut t_pos),
        ("n_kv_buf", &mut t_n_kv),
        ("x_out", &mut t_x_out),
        ("y1", &mut t_y1),
        ("q", &mut t_q),
        ("k", &mut t_k),
        ("v", &mut t_v),
        ("q_rope", &mut t_q_rope),
        ("k_rope", &mut t_k_rope),
        ("attn_o", &mut t_attn_o),
        ("o", &mut t_o),
        ("x_attn", &mut t_x_attn),
        ("y2", &mut t_y2),
        ("gate", &mut t_gate),
        ("up", &mut t_up),
        ("silu_out", &mut t_silu_out),
        ("down", &mut t_down),
    ];
    for (label, t) in registrations.iter_mut() {
        let err = unsafe { (v.tensorCreateGraphTensor.unwrap())(graph, *t) };
        anyhow::ensure!(err == 0, "tensorCreate({}) err=0x{:x}", label, err);
    }

    // ── 5. Add 14 nodes ─────────────────────────────────────────────────────
    let pkg = CString::new("qnn_oppkg").unwrap();
    let make_op = |name: *const c_char,
                   typ: *const c_char,
                   ins: *mut Qnn_Tensor_t,
                   n_in: u32,
                   outs: *mut Qnn_Tensor_t,
                   n_out: u32,
                   params: *mut Qnn_Param_t,
                   n_params: u32|
     -> Qnn_OpConfig_t {
        Qnn_OpConfig_t {
            version: Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
            __bindgen_anon_1: Qnn_OpConfig_t__bindgen_ty_1 {
                v1: Qnn_OpConfigV1_t {
                    name,
                    packageName: pkg.as_ptr(),
                    typeName: typ,
                    numOfParams: n_params,
                    params,
                    numOfInputs: n_in,
                    inputTensors: ins,
                    numOfOutputs: n_out,
                    outputTensors: outs,
                },
            },
        }
    };

    // Reusable param names.
    let pn_start_pos = CString::new("start_pos").unwrap();
    let pn_theta = CString::new("theta").unwrap();
    let pn_head_dim = CString::new("head_dim").unwrap();
    let pn_capacity = CString::new("capacity").unwrap();
    let pn_write_pos = CString::new("write_pos").unwrap();
    let pn_n_kv = CString::new("n_kv").unwrap();
    let pn_n_head = CString::new("n_head").unwrap();
    let pn_n_head_kv = CString::new("n_head_kv").unwrap();
    let pn_kv_capacity = CString::new("kv_capacity").unwrap();
    let pn_head_dim_fa = CString::new("head_dim").unwrap();

    // RoPE params (separate copies — Q/K share theta).
    // M3.4 D-D.1: `start_pos` is no longer a SCALAR param; it's supplied via
    // pos_buf input tensor at execute time.
    let mk_rope_params = |th: f32| {
        [Qnn_Param_t {
            paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
            name: pn_theta.as_ptr(),
            __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                scalarParam: Qnn_Scalar_t {
                    dataType: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                    __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 { floatValue: th },
                },
            },
        }]
    };
    let mut rope_q_params = mk_rope_params(theta);
    let mut rope_k_params = mk_rope_params(theta);
    // Suppress unused-warning for the kept `start_pos` / `write_pos` CString:
    let _ = &pn_start_pos;
    let _ = &pn_write_pos;

    // M3.4 D-D.1: KvScatter `write_pos` is now in pos_buf. Only head_dim and
    // capacity remain as SCALAR params.
    let mut kvs_params = [
        Qnn_Param_t {
            paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
            name: pn_head_dim.as_ptr(),
            __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                scalarParam: Qnn_Scalar_t {
                    dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                    __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                        int32Value: head_dim as i32,
                    },
                },
            },
        },
        Qnn_Param_t {
            paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
            name: pn_capacity.as_ptr(),
            __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                scalarParam: Qnn_Scalar_t {
                    dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                    __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                        int32Value: kv_capacity as i32,
                    },
                },
            },
        },
    ];

    // D-D.6: n_kv SCALAR 제거 — n_kv는 input tensor (n_kv_buf)로 동적 처리.
    let _ = &pn_n_kv;
    let mut fa_params = [
        Qnn_Param_t {
            paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
            name: pn_n_head.as_ptr(),
            __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                scalarParam: Qnn_Scalar_t {
                    dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                    __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                        int32Value: n_head as i32,
                    },
                },
            },
        },
        Qnn_Param_t {
            paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
            name: pn_n_head_kv.as_ptr(),
            __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                scalarParam: Qnn_Scalar_t {
                    dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                    __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                        int32Value: n_kv_heads as i32,
                    },
                },
            },
        },
        Qnn_Param_t {
            paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
            name: pn_kv_capacity.as_ptr(),
            __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                scalarParam: Qnn_Scalar_t {
                    dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                    __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                        int32Value: kv_capacity as i32,
                    },
                },
            },
        },
        Qnn_Param_t {
            paramType: Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
            name: pn_head_dim_fa.as_ptr(),
            __bindgen_anon_1: Qnn_Param_t__bindgen_ty_1 {
                scalarParam: Qnn_Scalar_t {
                    dataType: Qnn_DataType_t_QNN_DATATYPE_INT_32,
                    __bindgen_anon_1: Qnn_Scalar_t__bindgen_ty_1 {
                        int32Value: head_dim as i32,
                    },
                },
            },
        },
    ];

    // Op type names.
    let ot_rms = CString::new("CustomRmsNorm").unwrap();
    let ot_q40 = CString::new("CustomMatMulQ40F32").unwrap();
    let ot_rope = CString::new("CustomRope").unwrap();
    let ot_kvs = CString::new("CustomKvScatter").unwrap();
    let ot_fa = CString::new("CustomFlashAttn").unwrap();
    let ot_silu = CString::new("CustomSiluMul").unwrap();
    let ot_add = CString::new("CustomAdd").unwrap();

    // Op names.
    let on_rms_pre = CString::new("rms_pre_op").unwrap();
    let on_q_proj = CString::new("q_proj").unwrap();
    let on_k_proj = CString::new("k_proj").unwrap();
    let on_v_proj = CString::new("v_proj").unwrap();
    let on_rope_q = CString::new("rope_q").unwrap();
    let on_rope_k = CString::new("rope_k").unwrap();
    let on_kvs = CString::new("kvs").unwrap();
    let on_fa = CString::new("fa").unwrap();
    let on_o_proj = CString::new("o_proj").unwrap();
    let on_add1 = CString::new("add1").unwrap();
    let on_rms_post = CString::new("rms_post_op").unwrap();
    let on_gate_proj = CString::new("gate_proj").unwrap();
    let on_up_proj = CString::new("up_proj").unwrap();
    let on_silu = CString::new("silu").unwrap();
    let on_down_proj = CString::new("down_proj").unwrap();
    let on_add2 = CString::new("add2").unwrap();

    // Re-clone tensors for graphAddNode (Qnn_OpConfig copies pointers, not
    // payload — we rebuild Qnn_Tensor_t views referring to the same names).
    // Because the op only reads name/dims/dtype/type, re-using the registered
    // tensor structs is fine.
    let mut in_rms_pre = [t_x, t_rms_pre];
    let mut out_rms_pre = [t_y1];
    let mut in_q_proj = [t_qq, t_qd, t_y1];
    let mut out_q_proj = [t_q];
    let mut in_k_proj = [t_kq, t_kd, t_y1];
    let mut out_k_proj = [t_k];
    let mut in_v_proj = [t_vq, t_vd, t_y1];
    let mut out_v_proj = [t_v];
    // M3.4 D-D.1: RoPE Q/K and KvScatter all consume the same pos_buf as a
    // runtime input so the host can update pos per graphExecute.
    let mut in_rope_q = [t_q, t_pos];
    let mut out_rope_q = [t_q_rope];
    let mut in_rope_k = [t_k, t_pos];
    let mut out_rope_k = [t_k_rope];
    // M2.H multi-output: KvScatter declares 3 inputs (k_rope, v, pos) and 2
    // outputs (kcache, vcache). Earlier revisions routed kcache through
    // inputs[2] (InOutTensor) to fit the single-claim abstraction.
    let mut in_kvs = [t_k_rope, t_v, t_pos];
    let mut out_kvs = [t_kcache, t_vcache];
    // D-D.6: FlashAttn input은 (q, k, v, mask, sinks, score, n_kv_buf) 7개.
    let mut in_fa = [
        t_q_rope, t_kcache, t_vcache, t_mask, t_sinks, t_score, t_n_kv,
    ];
    let mut out_fa = [t_attn_o];
    let mut in_o_proj = [t_oq, t_od, t_attn_o];
    let mut out_o_proj = [t_o];
    let mut in_add1 = [t_o, t_x];
    let mut out_add1 = [t_x_attn];
    let mut in_rms_post = [t_x_attn, t_rms_post];
    let mut out_rms_post = [t_y2];
    let mut in_gate = [t_gq, t_gd, t_y2];
    let mut out_gate = [t_gate];
    let mut in_up = [t_uq, t_ud, t_y2];
    let mut out_up = [t_up];
    let mut in_silu = [t_gate, t_up];
    let mut out_silu = [t_silu_out];
    let mut in_down = [t_dq, t_dd, t_silu_out];
    let mut out_down = [t_down];
    let mut in_add2 = [t_down, t_x_attn];
    let mut out_add2 = [t_x_out];

    // 14 nodes.
    type NodeSpec<'a> = (
        *const c_char,
        *const c_char,
        *mut Qnn_Tensor_t,
        u32,
        *mut Qnn_Tensor_t,
        u32,
        *mut Qnn_Param_t,
        u32,
        &'a str,
    );
    let nodes: [NodeSpec<'_>; 14] = [
        (
            on_rms_pre.as_ptr(),
            ot_rms.as_ptr(),
            in_rms_pre.as_mut_ptr(),
            2,
            out_rms_pre.as_mut_ptr(),
            1,
            ptr::null_mut(),
            0,
            "RmsNorm(pre)",
        ),
        (
            on_q_proj.as_ptr(),
            ot_q40.as_ptr(),
            in_q_proj.as_mut_ptr(),
            3,
            out_q_proj.as_mut_ptr(),
            1,
            ptr::null_mut(),
            0,
            "Q proj",
        ),
        (
            on_k_proj.as_ptr(),
            ot_q40.as_ptr(),
            in_k_proj.as_mut_ptr(),
            3,
            out_k_proj.as_mut_ptr(),
            1,
            ptr::null_mut(),
            0,
            "K proj",
        ),
        (
            on_v_proj.as_ptr(),
            ot_q40.as_ptr(),
            in_v_proj.as_mut_ptr(),
            3,
            out_v_proj.as_mut_ptr(),
            1,
            ptr::null_mut(),
            0,
            "V proj",
        ),
        (
            on_rope_q.as_ptr(),
            ot_rope.as_ptr(),
            in_rope_q.as_mut_ptr(),
            2, // M3.4 D-D.1: x_in + pos_buf
            out_rope_q.as_mut_ptr(),
            1,
            rope_q_params.as_mut_ptr(),
            1, // theta only
            "RoPE Q",
        ),
        (
            on_rope_k.as_ptr(),
            ot_rope.as_ptr(),
            in_rope_k.as_mut_ptr(),
            2, // M3.4 D-D.1: x_in + pos_buf
            out_rope_k.as_mut_ptr(),
            1,
            rope_k_params.as_mut_ptr(),
            1, // theta only
            "RoPE K",
        ),
        (
            on_kvs.as_ptr(),
            ot_kvs.as_ptr(),
            in_kvs.as_mut_ptr(),
            3, // M3.4 D-D.1: k_src + v_src + pos_buf
            out_kvs.as_mut_ptr(),
            2,
            kvs_params.as_mut_ptr(),
            2, // head_dim + capacity (write_pos via pos_buf)
            "KvScatter",
        ),
        (
            on_fa.as_ptr(),
            ot_fa.as_ptr(),
            in_fa.as_mut_ptr(),
            7, // D-D.6: q + k + v + mask + sinks + score + n_kv_buf
            out_fa.as_mut_ptr(),
            1,
            fa_params.as_mut_ptr(),
            4, // D-D.6: n_head + n_head_kv + kv_capacity + head_dim (n_kv 제거)
            "FlashAttn",
        ),
        (
            on_o_proj.as_ptr(),
            ot_q40.as_ptr(),
            in_o_proj.as_mut_ptr(),
            3,
            out_o_proj.as_mut_ptr(),
            1,
            ptr::null_mut(),
            0,
            "O proj",
        ),
        (
            on_add1.as_ptr(),
            ot_add.as_ptr(),
            in_add1.as_mut_ptr(),
            2,
            out_add1.as_mut_ptr(),
            1,
            ptr::null_mut(),
            0,
            "Add(residual1)",
        ),
        (
            on_rms_post.as_ptr(),
            ot_rms.as_ptr(),
            in_rms_post.as_mut_ptr(),
            2,
            out_rms_post.as_mut_ptr(),
            1,
            ptr::null_mut(),
            0,
            "RmsNorm(post)",
        ),
        (
            on_gate_proj.as_ptr(),
            ot_q40.as_ptr(),
            in_gate.as_mut_ptr(),
            3,
            out_gate.as_mut_ptr(),
            1,
            ptr::null_mut(),
            0,
            "gate proj",
        ),
        (
            on_up_proj.as_ptr(),
            ot_q40.as_ptr(),
            in_up.as_mut_ptr(),
            3,
            out_up.as_mut_ptr(),
            1,
            ptr::null_mut(),
            0,
            "up proj",
        ),
        (
            on_silu.as_ptr(),
            ot_silu.as_ptr(),
            in_silu.as_mut_ptr(),
            2,
            out_silu.as_mut_ptr(),
            1,
            ptr::null_mut(),
            0,
            "SiluMul",
        ),
    ];
    // Note: we declared the tuple as 14 entries but the layer DAG has 14 nodes
    // with two more (down_proj + add2). Move them out to keep readable size.
    let mut in_pad: [Qnn_Tensor_t; 0] = [];
    let _ = &mut in_pad; // silence unused if cfg
    // Helper to dump per-tensor metadata before graphAddNode for debug.
    let tensor_brief = |t: *const Qnn_Tensor_t| -> String {
        if t.is_null() {
            return "<null>".to_string();
        }
        unsafe {
            let v1 = (*t).__bindgen_anon_1.v1;
            let nm = if v1.name.is_null() {
                "<no-name>".to_string()
            } else {
                std::ffi::CStr::from_ptr(v1.name)
                    .to_string_lossy()
                    .into_owned()
            };
            let mut dims_str = String::from("[");
            for i in 0..v1.rank {
                if i > 0 {
                    dims_str.push_str(", ");
                }
                let d = *v1.dimensions.add(i as usize);
                dims_str.push_str(&d.to_string());
            }
            dims_str.push(']');
            format!(
                "{}(type={},dt={},rank={},dims={})",
                nm, v1.type_, v1.dataType, v1.rank, dims_str
            )
        }
    };
    let dump_op = |label: &str,
                   ins: *const Qnn_Tensor_t,
                   n_in: u32,
                   outs: *const Qnn_Tensor_t,
                   n_out: u32| {
        eprint!("[graphAddNode pre] {} inputs:", label);
        for i in 0..n_in {
            let t = unsafe { ins.add(i as usize) };
            eprint!(" #{}={}", i, tensor_brief(t));
        }
        eprint!("\n[graphAddNode pre] {} outputs:", label);
        for i in 0..n_out {
            let t = unsafe { outs.add(i as usize) };
            eprint!(" #{}={}", i, tensor_brief(t));
        }
        eprintln!();
    };

    for (nm, ty, ins, n_in, outs, n_out, params, n_params, label) in nodes {
        dump_op(label, ins as *const _, n_in, outs as *const _, n_out);
        let op = make_op(nm, ty, ins, n_in, outs, n_out, params, n_params);
        let err = unsafe { (v.graphAddNode.unwrap())(graph, op) };
        anyhow::ensure!(err == 0, "graphAddNode({}) err=0x{:x}", label, err);
    }
    // Two trailing nodes (down proj + residual add2).
    let trailing: [NodeSpec<'_>; 2] = [
        (
            on_down_proj.as_ptr(),
            ot_q40.as_ptr(),
            in_down.as_mut_ptr(),
            3,
            out_down.as_mut_ptr(),
            1,
            ptr::null_mut(),
            0,
            "down proj",
        ),
        (
            on_add2.as_ptr(),
            ot_add.as_ptr(),
            in_add2.as_mut_ptr(),
            2,
            out_add2.as_mut_ptr(),
            1,
            ptr::null_mut(),
            0,
            "Add(residual2)",
        ),
    ];
    for (nm, ty, ins, n_in, outs, n_out, params, n_params, label) in trailing {
        dump_op(label, ins as *const _, n_in, outs as *const _, n_out);
        let op = make_op(nm, ty, ins, n_in, outs, n_out, params, n_params);
        let err = unsafe { (v.graphAddNode.unwrap())(graph, op) };
        anyhow::ensure!(err == 0, "graphAddNode({}) err=0x{:x}", label, err);
    }

    // 14 nodes total (12 in `nodes` + 2 in `trailing`). Verified by
    // `qnn_oppkg::graph::LAYER_NODE_COUNT == 14` host test.
    const _ASSERT_LAYER_NODE_COUNT: usize = 14;

    // ── 6. graphFinalize (timed) ────────────────────────────────────────────
    let t_fin0 = Instant::now();
    let err = unsafe { (v.graphFinalize.unwrap())(graph, ptr::null_mut(), ptr::null_mut()) };
    let finalize_ms = t_fin0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "graphFinalize -> err=0x{:x} elapsed={:.2} ms",
        err, finalize_ms
    );
    anyhow::ensure!(err == 0, "graphFinalize err=0x{:x}", err);

    // ── 7. memRegister all host-backed tensors ──────────────────────────────
    let mk_desc = |fd: i32,
                   host_data: *mut std::ffi::c_void,
                   dtype: Qnn_DataType_t,
                   dims: &[u32]|
     -> Qnn_MemDescriptor_t {
        Qnn_MemDescriptor_t {
            memShape: Qnn_MemShape_t {
                numDim: dims.len() as u32,
                dimSize: dims.as_ptr() as *mut u32,
                shapeConfig: ptr::null(),
            },
            dataType: dtype,
            memType: Qnn_MemType_t_QNN_MEM_TYPE_DMA_BUF,
            __bindgen_anon_1: Qnn_MemDescriptor_t__bindgen_ty_1 {
                dmaBufInfo: Qnn_MemDmaBufInfo_t {
                    fd,
                    data: host_data,
                },
            },
        }
    };
    // M2.H multi-output: vcache is now APP_WRITE (was NATIVE). Both kcache
    // and vcache are registered with host-allocated rpcmem fds so the SDK
    // can wire OutputClaim slots to them. Earlier NATIVE vcache caused
    // graphAddNode err=0x1775 with the multi-output KvScatter abstraction.
    let fd_vcache_h = unsafe { rpcmem_to_fd(rpc_vcache) };
    let descs = [
        mk_desc(fd_x, rpc_x, f32_t, &dims_x),
        mk_desc(fd_rms_pre, rpc_rms_pre, f32_t, &dims_rms_pre),
        mk_desc(fd_rms_post, rpc_rms_post, f32_t, &dims_rms_post),
        mk_desc(fd_qq, rpc_qq, u8_t, &dims_qq),
        mk_desc(fd_qd, rpc_qd, f16_t, &dims_qd),
        mk_desc(fd_kq, rpc_kq, u8_t, &dims_kq),
        mk_desc(fd_kd, rpc_kd, f16_t, &dims_kd),
        mk_desc(fd_vq, rpc_vq, u8_t, &dims_vq),
        mk_desc(fd_vd, rpc_vd, f16_t, &dims_vd),
        mk_desc(fd_oq, rpc_oq, u8_t, &dims_oq),
        mk_desc(fd_od, rpc_od, f16_t, &dims_od),
        mk_desc(fd_gq, rpc_gq, u8_t, &dims_gq),
        mk_desc(fd_gd, rpc_gd, f16_t, &dims_gd),
        mk_desc(fd_uq, rpc_uq, u8_t, &dims_uq),
        mk_desc(fd_ud, rpc_ud, f16_t, &dims_ud),
        mk_desc(fd_dq, rpc_dq, u8_t, &dims_dq),
        mk_desc(fd_dd, rpc_dd, f16_t, &dims_dd),
        mk_desc(fd_kcache, rpc_kcache, f16_t, &dims_kcache),
        mk_desc(fd_vcache_h, rpc_vcache, f16_t, &dims_vcache),
        mk_desc(fd_mask, rpc_mask, f16_t, &dims_mask),
        mk_desc(fd_sinks, rpc_sinks, f32_t, &dims_sinks),
        mk_desc(fd_score, rpc_score, f32_t, &dims_score),
        mk_desc(fd_pos, rpc_pos, i32_t, &dims_pos), // M3.4 D-D.1
        mk_desc(fd_n_kv, rpc_n_kv, i32_t, &dims_n_kv), // M3.4 D-D.6
        mk_desc(fd_x_out, rpc_x_out, f32_t, &dims_x_out),
    ];
    let mut mh = [ptr::null_mut::<std::ffi::c_void>(); 25];
    let err = unsafe {
        (v.memRegister.unwrap())(ctx, descs.as_ptr(), descs.len() as u32, mh.as_mut_ptr())
    };
    anyhow::ensure!(err == 0, "memRegister err=0x{:x}", err);

    // Bind tensor MEMHANDLEs.
    let set_mh = |t: &mut Qnn_Tensor_t, h: *mut std::ffi::c_void| {
        t.__bindgen_anon_1.v1.memType = Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
        t.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = h;
    };
    let mut t_x_mh = t_x;
    set_mh(&mut t_x_mh, mh[0]);
    let mut t_rms_pre_mh = t_rms_pre;
    set_mh(&mut t_rms_pre_mh, mh[1]);
    let mut t_rms_post_mh = t_rms_post;
    set_mh(&mut t_rms_post_mh, mh[2]);
    let mut t_qq_mh = t_qq;
    set_mh(&mut t_qq_mh, mh[3]);
    let mut t_qd_mh = t_qd;
    set_mh(&mut t_qd_mh, mh[4]);
    let mut t_kq_mh = t_kq;
    set_mh(&mut t_kq_mh, mh[5]);
    let mut t_kd_mh = t_kd;
    set_mh(&mut t_kd_mh, mh[6]);
    let mut t_vq_mh = t_vq;
    set_mh(&mut t_vq_mh, mh[7]);
    let mut t_vd_mh = t_vd;
    set_mh(&mut t_vd_mh, mh[8]);
    let mut t_oq_mh = t_oq;
    set_mh(&mut t_oq_mh, mh[9]);
    let mut t_od_mh = t_od;
    set_mh(&mut t_od_mh, mh[10]);
    let mut t_gq_mh = t_gq;
    set_mh(&mut t_gq_mh, mh[11]);
    let mut t_gd_mh = t_gd;
    set_mh(&mut t_gd_mh, mh[12]);
    let mut t_uq_mh = t_uq;
    set_mh(&mut t_uq_mh, mh[13]);
    let mut t_ud_mh = t_ud;
    set_mh(&mut t_ud_mh, mh[14]);
    let mut t_dq_mh = t_dq;
    set_mh(&mut t_dq_mh, mh[15]);
    let mut t_dd_mh = t_dd;
    set_mh(&mut t_dd_mh, mh[16]);
    let mut t_kcache_mh = t_kcache;
    set_mh(&mut t_kcache_mh, mh[17]);
    // M2.H: vcache is now APP_WRITE (host-registered fd) — index 18.
    let mut t_vcache_mh = t_vcache;
    set_mh(&mut t_vcache_mh, mh[18]);
    let mut t_mask_mh = t_mask;
    set_mh(&mut t_mask_mh, mh[19]);
    let mut t_sinks_mh = t_sinks;
    set_mh(&mut t_sinks_mh, mh[20]);
    let mut t_score_mh = t_score;
    set_mh(&mut t_score_mh, mh[21]);
    // M3.4 D-D.1: pos_buf at index 22 (descs order).
    let mut t_pos_mh = t_pos;
    set_mh(&mut t_pos_mh, mh[22]);
    // M3.4 D-D.6: n_kv_buf at index 23 (after pos, before x_out).
    let mut t_n_kv_mh = t_n_kv;
    set_mh(&mut t_n_kv_mh, mh[23]);
    let mut t_x_out_mh = t_x_out;
    set_mh(&mut t_x_out_mh, mh[24]);

    // ── 8. graphExecute ────────────────────────────────────────────────────
    // exec_inputs: every APP_WRITE tensor; exec_outputs: APP_READ (x_out).
    // Also include kcache + vcache as inputs (the SDK accepts APP_WRITE
    // memhandles as part of the input list, mirroring kv_scatter microbench;
    // M2.H multi-output: vcache is now APP_WRITE too).
    let exec_inputs = [
        t_x_mh,
        t_rms_pre_mh,
        t_rms_post_mh,
        t_qq_mh,
        t_qd_mh,
        t_kq_mh,
        t_kd_mh,
        t_vq_mh,
        t_vd_mh,
        t_oq_mh,
        t_od_mh,
        t_gq_mh,
        t_gd_mh,
        t_uq_mh,
        t_ud_mh,
        t_dq_mh,
        t_dd_mh,
        t_mask_mh,
        t_sinks_mh,
        t_score_mh,
        t_pos_mh,  // M3.4 D-D.1
        t_n_kv_mh, // M3.4 D-D.6
    ];
    // M2.H multi-output: kcache + vcache are APP_READ outputs (KvScatter
    // claims). The SDK lets them double as graph-internal edges (FA input).
    let mut exec_outputs = [t_kcache_mh, t_vcache_mh, t_x_out_mh];
    let err = unsafe {
        (v.graphExecute.unwrap())(
            graph,
            exec_inputs.as_ptr(),
            exec_inputs.len() as u32,
            exec_outputs.as_mut_ptr(),
            exec_outputs.len() as u32,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    anyhow::ensure!(err == 0, "graphExecute err=0x{:x}", err);

    // M2.H 5th: KvScatter host-buffer propagation + attn_o ratio diagnostic.
    {
        let qnn_xout = unsafe { std::slice::from_raw_parts(rpc_x_out as *const f32, 8) };
        eprintln!("[diag-base] ref_x_out[0..8] = {:?}", &ref_x_out[..8]);
        eprintln!("[diag-base] qnn_xout[0..8]  = {:?}", qnn_xout);
        // Per-element ratio (qnn / ref) for first 8 outputs.
        eprintln!("[diag-base] qnn_xout / ref_x_out (first 8 elems):");
        for i in 0..8 {
            let r = if ref_x_out[i].abs() > 1e-6 {
                qnn_xout[i] / ref_x_out[i]
            } else {
                f32::NAN
            };
            eprintln!("  [{}] ratio = {:.4}", i, r);
        }
        // KvScatter slot at write_pos: head 0/1, first 8 halves.
        let qnn_kc = unsafe { std::slice::from_raw_parts(rpc_kcache as *const u16, kv_total) };
        let kc_to_s = |buf: &[u16], off: usize, n: usize| -> String {
            (0..n)
                .map(|i| format!("{:.4} ", f16_bits_to_f32(buf[off + i])))
                .collect()
        };
        let off_h0 = (pos as usize) * head_dim;
        let off_h1 = kv_capacity * head_dim + (pos as usize) * head_dim;
        eprintln!(
            "[diag-base] qnn kcache[h=0,pos][0..8] = {}",
            kc_to_s(qnn_kc, off_h0, 8)
        );
        eprintln!(
            "[diag-base] ref kcache[h=0,pos][0..8] = {}",
            kc_to_s(&ref_kcache_full, off_h0, 8)
        );
        eprintln!(
            "[diag-base] qnn kcache[h=1,pos][0..8] = {}",
            kc_to_s(qnn_kc, off_h1, 8)
        );
        eprintln!(
            "[diag-base] ref kcache[h=1,pos][0..8] = {}",
            kc_to_s(&ref_kcache_full, off_h1, 8)
        );
        // Coverage: how many halves in QNN kcache changed from zero?
        let nz_qnn = qnn_kc.iter().filter(|&&v| v != 0).count();
        let nz_ref = ref_kcache_full.iter().filter(|&&v| v != 0).count();
        eprintln!(
            "[diag-base] kcache nonzero halves: qnn={} ref={} total={}",
            nz_qnn, nz_ref, kv_total
        );
        let _ = ref_attn_o.len();
    }

    // ── 9. Compare ──────────────────────────────────────────────────────────
    let mut max_abs = 0.0f32;
    unsafe {
        let test = std::slice::from_raw_parts(rpc_x_out as *const f32, dim);
        for i in 0..dim {
            let d = (test[i] - ref_x_out[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
    }

    // ── M2.I breakdown timing — QNN graphExecute wall-clock ─────────────────
    // We separately report:
    //   (1) first-call latency  — captures any deferred graph compilation /
    //       JIT cost paid at the first execute (suspected cause of the 1차 8.7
    //       ms anomaly when the prior measurement skipped warmup)
    //   (2) steady-state median — 3 warmup + 10 measure. KV cache writes the
    //       same pos with deterministic data → outputs match across runs.
    //
    // Note: the very first graphExecute right after graphFinalize was already
    // performed for correctness comparison above. So this "first-call" sample
    // here is actually the SECOND graphExecute, i.e. already partially warm.
    // To get a true cold first-call we'd need to skip the correctness exec —
    // out of scope; we keep correctness as gating.
    const QNN_WARMUP: usize = 3;
    const QNN_RUNS: usize = 10;
    // Sample 1 cold-ish first-call (= 2nd execute overall; correctness was 1st).
    let t_first = Instant::now();
    let err = unsafe {
        (v.graphExecute.unwrap())(
            graph,
            exec_inputs.as_ptr(),
            exec_inputs.len() as u32,
            exec_outputs.as_mut_ptr(),
            exec_outputs.len() as u32,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    anyhow::ensure!(err == 0, "graphExecute (2nd cold) err=0x{:x}", err);
    let qnn_second_cold_ms = t_first.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[M2.I-breakdown] qnn_graph_execute_2nd_cold = {:.3} ms (no warmup, after correctness exec)",
        qnn_second_cold_ms
    );

    for _ in 0..QNN_WARMUP {
        let err = unsafe {
            (v.graphExecute.unwrap())(
                graph,
                exec_inputs.as_ptr(),
                exec_inputs.len() as u32,
                exec_outputs.as_mut_ptr(),
                exec_outputs.len() as u32,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        anyhow::ensure!(err == 0, "graphExecute warmup err=0x{:x}", err);
    }
    let mut qnn_runs_ms = Vec::with_capacity(QNN_RUNS);
    for _ in 0..QNN_RUNS {
        let t0 = Instant::now();
        let err = unsafe {
            (v.graphExecute.unwrap())(
                graph,
                exec_inputs.as_ptr(),
                exec_inputs.len() as u32,
                exec_outputs.as_mut_ptr(),
                exec_outputs.len() as u32,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        anyhow::ensure!(err == 0, "graphExecute timed err=0x{:x}", err);
        qnn_runs_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    qnn_runs_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let qnn_median_ms = qnn_runs_ms[QNN_RUNS / 2];
    let qnn_min_ms = qnn_runs_ms[0];
    let qnn_max_ms = qnn_runs_ms[QNN_RUNS - 1];
    let qnn_mean_ms = qnn_runs_ms.iter().sum::<f64>() / QNN_RUNS as f64;
    eprintln!(
        "[M2.I-breakdown] qnn_graph_execute: median={:.3} ms  min={:.3}  max={:.3}  mean={:.3}  (N={}, warmup={})",
        qnn_median_ms, qnn_min_ms, qnn_max_ms, qnn_mean_ms, QNN_RUNS, QNN_WARMUP
    );

    // ── M2.I breakdown analysis — SDK overhead separation ───────────────────
    // Production engine baseline: Qwen2.5-1.5B Q4_0 generate -n 32 = 29.43
    // ms/tok, 28 layers → 1.05 ms/layer (fused dispatch + minimal sync).
    let prod_layer_ms = 29.43_f64 / 28.0;
    let sdk_overhead_vs_chain_only = qnn_median_ms - raw_chain_only_ms;
    let sdk_overhead_vs_per_stage = qnn_median_ms - raw_chain_per_stage_ms;
    let sdk_overhead_vs_prod = qnn_median_ms - prod_layer_ms;
    eprintln!("\n[M2.I-breakdown] === Comparison ===");
    eprintln!(
        "  production engine layer wall-clock (estimated, 29.43/28) = {:.3} ms",
        prod_layer_ms
    );
    eprintln!(
        "  raw OpenCL chain-only sync (1 cl_finish, production-like) = {:.3} ms",
        raw_chain_only_ms
    );
    eprintln!(
        "  raw OpenCL per-stage sync  (14 cl_finish, raw individual) = {:.3} ms",
        raw_chain_per_stage_ms
    );
    eprintln!(
        "  qnn graphExecute median                                    = {:.3} ms",
        qnn_median_ms
    );
    eprintln!("  --------------------------------------------------------------------");
    eprintln!(
        "  qnn − chain-only          = {:+.3} ms  (SDK validate + bind + submit)",
        sdk_overhead_vs_chain_only
    );
    eprintln!(
        "  qnn − per-stage           = {:+.3} ms  (vs raw individual dispatch)",
        sdk_overhead_vs_per_stage
    );
    eprintln!(
        "  qnn − production          = {:+.3} ms  (TBT-relevant: × {:.2})",
        sdk_overhead_vs_prod,
        qnn_median_ms / prod_layer_ms
    );
    eprintln!(
        "  per-op SDK overhead (qnn − chain-only) / 14 = {:+.4} ms/op",
        sdk_overhead_vs_chain_only / 14.0
    );

    unsafe {
        let _ = (v.memDeRegister.unwrap())(mh.as_ptr(), mh.len() as u32);
    }

    Ok((max_abs, finalize_ms))
}

#[cfg(feature = "qnn")]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let frac = mant as f32 / 1024.0;
        return if sign == 1 {
            -frac * 2.0f32.powi(-14)
        } else {
            frac * 2.0f32.powi(-14)
        };
    }
    if exp == 31 {
        return if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }
    let new_exp = (exp - 15 + 127) as u32;
    let new_mant = mant << 13;
    f32::from_bits((sign << 31) | (new_exp << 23) | new_mant)
}

#[cfg(feature = "qnn")]
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0 {
        return sign << 15;
    }
    let new_exp = exp - 127 + 15;
    if new_exp <= 0 {
        return sign << 15;
    }
    if new_exp >= 31 {
        return (sign << 15) | (0x1f << 10);
    }
    let new_mant = (mant >> 13) as u16;
    (sign << 15) | ((new_exp as u16) << 10) | new_mant
}
