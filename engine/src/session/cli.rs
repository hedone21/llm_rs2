use clap::Parser;

pub mod eviction;
pub mod kv_mode;

pub use eviction::{
    D2oArgs, EvictionCmd, EvictionCommonArgs, H2oArgs, SlidingArgs, StreamingArgs, TopLevelCmd,
};
pub use kv_mode::{KvMode, KvModeArgs};

/// `--secondary-dtype` CLI 인수 값 (D-3, ENG-ALG-225).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecondaryDtypeArg {
    Auto,
    F16,
    Q4_0,
    F32,
}

/// `--secondary-dtype` value_parser.
pub fn parse_secondary_dtype(s: &str) -> Result<SecondaryDtypeArg, String> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(SecondaryDtypeArg::Auto),
        "f16" => Ok(SecondaryDtypeArg::F16),
        "q4_0" | "q4" => Ok(SecondaryDtypeArg::Q4_0),
        "f32" => Ok(SecondaryDtypeArg::F32),
        other => Err(format!(
            "unknown secondary-dtype '{other}'. Valid values: auto, f16, q4_0, f32"
        )),
    }
}

impl From<SecondaryDtypeArg> for crate::models::weights::SecondaryDtypeChoice {
    fn from(arg: SecondaryDtypeArg) -> Self {
        match arg {
            SecondaryDtypeArg::Auto => Self::Auto,
            SecondaryDtypeArg::F16 => Self::F16,
            SecondaryDtypeArg::Q4_0 => Self::Q4_0,
            SecondaryDtypeArg::F32 => Self::F32,
        }
    }
}

/// `--swap` CLI 인수 — 4 swap 모드 선택용 통합 shorthand (backlog P3, 2026-05-25).
///
/// `--swap` (단독): default = IntraForward 활성 (LISWAP-4, production winner).
/// `--swap <MODE>`: 명시 모드 선택. `intra-forward` / `incremental` /
/// `phase-aware` / `layer-immediate`.
///
/// 기존 4 flag (`--swap-incremental-per-tick`, `--swap-intra-forward`,
/// `--swap-phase-aware`, `--swap-layer-immediate`)는 deprecated. 직접 사용
/// 시 stderr 1회 경고 후 그대로 동작. `--swap` 사용 시 기존 flag보다 우선.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwapMode {
    IntraForward,
    Incremental,
    PhaseAware,
    LayerImmediate,
}

/// `--swap` value_parser.
pub fn parse_swap_mode(s: &str) -> Result<SwapMode, String> {
    match s.to_lowercase().as_str() {
        "intra-forward" | "intra_forward" | "intraforward" => Ok(SwapMode::IntraForward),
        "incremental" => Ok(SwapMode::Incremental),
        "phase-aware" | "phase_aware" | "phaseaware" => Ok(SwapMode::PhaseAware),
        "layer-immediate" | "layer_immediate" | "layerimmediate" => Ok(SwapMode::LayerImmediate),
        other => Err(format!(
            "unknown swap mode '{other}'. Valid: intra-forward, incremental, phase-aware, layer-immediate"
        )),
    }
}

/// `--secondary-layout` CLI 인수 값.
///
/// AUF의 어떤 weights variant로 swap 후 텐서를 만들지 결정한다. 기본은
/// `auto`로, 빌드 환경의 preferred variant(OpenCL→AdrenoSoa) 우선 + AUF에
/// 그게 없으면 CpuAos로 폴백한다. `aos`는 강제로 CpuAos / CudaAos 사용해
/// host pointer를 살려두므로 swap 후 `switch_hw cpu` / partition 호환이
/// 가능하지만 GPU TBT가 SOA 대비 떨어진다 (Adreno 830 실측 33–55%).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecondaryLayoutArg {
    Auto,
    Aos,
    Soa,
}

pub fn parse_secondary_layout(s: &str) -> Result<SecondaryLayoutArg, String> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(SecondaryLayoutArg::Auto),
        "aos" | "cpu_aos" | "cuda_aos" => Ok(SecondaryLayoutArg::Aos),
        "soa" | "adreno_soa" => Ok(SecondaryLayoutArg::Soa),
        other => Err(format!(
            "unknown secondary-layout '{other}'. Valid values: auto, aos, soa"
        )),
    }
}

impl From<SecondaryLayoutArg> for crate::models::weights::SecondaryLayoutChoice {
    fn from(arg: SecondaryLayoutArg) -> Self {
        match arg {
            SecondaryLayoutArg::Auto => Self::Auto,
            SecondaryLayoutArg::Aos => Self::Aos,
            SecondaryLayoutArg::Soa => Self::Soa,
        }
    }
}

/// `--primary-variant` CLI 인수 값 (W-AUF-1 C4). AUF primary backend variant 선택.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimaryVariantArg {
    Auto,
    AdrenoSoa,
    CpuAos,
    CudaAos,
}

pub fn parse_primary_variant(s: &str) -> Result<PrimaryVariantArg, String> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(PrimaryVariantArg::Auto),
        "adreno-soa" | "adreno_soa" => Ok(PrimaryVariantArg::AdrenoSoa),
        "cpu-aos" | "cpu_aos" => Ok(PrimaryVariantArg::CpuAos),
        "cuda-aos" | "cuda_aos" => Ok(PrimaryVariantArg::CudaAos),
        other => Err(format!(
            "unknown primary-variant '{other}'. Valid: auto, adreno-soa, cpu-aos, cuda-aos"
        )),
    }
}

impl From<PrimaryVariantArg> for crate::models::loader::AufVariantChoice {
    fn from(arg: PrimaryVariantArg) -> Self {
        match arg {
            PrimaryVariantArg::Auto => Self::Auto,
            PrimaryVariantArg::AdrenoSoa => Self::AdrenoSoa,
            PrimaryVariantArg::CpuAos => Self::CpuAos,
            PrimaryVariantArg::CudaAos => Self::CudaAos,
        }
    }
}

/// `--primary-dtype` CLI 인수 값 (W-AUF-1 C4). AUF primary dtype 선택.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimaryDtypeArg {
    Auto,
    F16,
    Q4_0,
    Q8_0,
    Bf16,
    F32,
    Q4_1,
}

pub fn parse_primary_dtype(s: &str) -> Result<PrimaryDtypeArg, String> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(PrimaryDtypeArg::Auto),
        "f16" => Ok(PrimaryDtypeArg::F16),
        "q4_0" | "q4" => Ok(PrimaryDtypeArg::Q4_0),
        "q8_0" | "q8" => Ok(PrimaryDtypeArg::Q8_0),
        "bf16" => Ok(PrimaryDtypeArg::Bf16),
        "f32" => Ok(PrimaryDtypeArg::F32),
        "q4_1" => Ok(PrimaryDtypeArg::Q4_1),
        other => Err(format!(
            "unknown primary-dtype '{other}'. Valid: auto, f16, q4_0, q8_0, bf16, f32, q4_1"
        )),
    }
}

impl From<PrimaryDtypeArg> for crate::models::loader::AufDtypeChoice {
    fn from(arg: PrimaryDtypeArg) -> Self {
        match arg {
            PrimaryDtypeArg::Auto => Self::Auto,
            PrimaryDtypeArg::F16 => Self::F16,
            PrimaryDtypeArg::Q4_0 => Self::Q4_0,
            PrimaryDtypeArg::Q8_0 => Self::Q8_0,
            PrimaryDtypeArg::Bf16 => Self::BF16,
            PrimaryDtypeArg::F32 => Self::F32,
            PrimaryDtypeArg::Q4_1 => Self::Q4_1,
        }
    }
}

/// Parse `--qcf-sample-layers` argument into a list of layer indices.
///
/// Accepts:
/// - `"auto"` (default): `[0, n/4, n/2, 3n/4, n-1]` via `compute_auto_sample_layers`.
/// - `"all"`: every layer `[0..n_layers)`.
/// - `"0,8,16,24,31"`: explicit comma-separated indices (sorted + deduped).
pub fn parse_qcf_sample_layers(spec: &str, n_layers: usize) -> Result<Vec<usize>, String> {
    let s = spec.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("auto") {
        return Ok(crate::qcf::compute_auto_sample_layers(n_layers));
    }
    if s.eq_ignore_ascii_case("all") {
        return Ok((0..n_layers).collect());
    }
    let mut out = Vec::new();
    for part in s.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        match p.parse::<usize>() {
            Ok(v) if v < n_layers => out.push(v),
            Ok(v) => return Err(format!("layer index {v} >= n_layers={n_layers}")),
            Err(e) => return Err(format!("failed to parse layer index '{p}': {e}")),
        }
    }
    out.sort();
    out.dedup();
    if out.is_empty() {
        return Err(format!("no valid layer indices in '{s}'"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_llama3_8b() {
        assert_eq!(
            parse_qcf_sample_layers("auto", 32).unwrap(),
            vec![0, 8, 16, 24, 31]
        );
    }

    #[test]
    fn all_small() {
        assert_eq!(parse_qcf_sample_layers("all", 4).unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn explicit() {
        assert_eq!(
            parse_qcf_sample_layers("0,8,16,24,31", 32).unwrap(),
            vec![0, 8, 16, 24, 31]
        );
    }

    #[test]
    fn explicit_unsorted_dedup() {
        assert_eq!(
            parse_qcf_sample_layers("16,0,16,8", 32).unwrap(),
            vec![0, 8, 16]
        );
    }

    #[test]
    fn out_of_range() {
        assert!(parse_qcf_sample_layers("0,32", 32).is_err());
    }

    #[test]
    fn empty_after_trim() {
        assert!(parse_qcf_sample_layers(",", 4).is_err());
    }
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(short, long, default_value = "models/llama3.2-1b")]
    pub model_path: String,

    /// Path to a file containing the prompt. Overrides --prompt if set.
    #[arg(long)]
    pub prompt_file: Option<String>,

    #[arg(short, long, default_value = "Hello, world! I am a")]
    pub prompt: String,

    #[arg(short, long, default_value_t = 20)]
    pub num_tokens: usize,

    /// Backend to use: "cpu", "opencl", "cuda" (build with --features cuda), or
    /// "htp" (build with --features htp_fastrpc, Android only).
    /// Default: Android target → "opencl" (Adreno production path), else → "cpu".
    #[cfg(target_os = "android")]
    #[arg(short, long, default_value = "opencl")]
    pub backend: String,

    #[cfg(not(target_os = "android"))]
    #[arg(short, long, default_value = "cpu")]
    pub backend: String,

    /// Disable zero-copy shared memory (CL_MEM_ALLOC_HOST_PTR).
    ///
    /// Zero-copy is enabled by default on ARM SoC to remove CPU↔GPU memcpy.
    /// Set this flag to fall back to device-only allocations.
    ///
    /// Other features force-enable zero-copy regardless of this flag:
    /// `--resilience-prealloc-switch`, `--tensor-partition > 0`,
    /// `--prefill-cpu-chunk-size > 0`, `--enable-resilience`.
    #[arg(long, default_value_t = false)]
    pub no_zero_copy: bool,

    /// Sprint 2a Phase 2 (ENG-RPCMEM-040): enable rpcmem DMA-BUF zero-copy
    /// allocation for KV cache and precision swap secondary store.
    ///
    /// Adreno Android only — host builds receive a warning and silently
    /// demote. Requires `--backend opencl`.
    ///
    /// When active, OpenCL backend eagerly dlopens `libcdsprpc.so` and shares
    /// a single `Arc<RpcmemAllocator>` between `OpenCLMemory::alloc_kv`
    /// (KV path) and `RpcmemSecondaryStore` (precision swap secondary).
    #[arg(long, default_value_t = false)]
    pub opencl_rpcmem: bool,

    #[arg(long, default_value_t = 2048)]
    pub max_seq_len: usize,

    #[arg(long, default_value_t = 0.8)]
    pub temperature: f32,

    #[arg(long, default_value_t = 0.9)]
    pub top_p: f32,

    #[arg(long, default_value_t = 40)]
    pub top_k: usize,

    #[arg(long, default_value_t = 1.1)]
    pub repetition_penalty: f32,

    #[arg(long, default_value_t = 64)]
    pub repetition_window: usize,

    /// Disable GPU kernel plan for decode (fallback to forward_into every token)
    #[arg(long, default_value_t = false)]
    pub no_gpu_plan: bool,

    /// GPU ratio for tensor partition — fraction of FFN gate/up rows assigned to GPU.
    /// Range (0.0, 1.0): 0.0 = disabled (no split), 1.0 = disabled (no split).
    /// 0.1 = 10% GPU + 90% CPU, 0.9 = 90% GPU + 10% CPU.
    /// NOTE: split_row is clamped to [128, out_dim-128], so extreme values like 0.001
    /// still leave 128 rows on GPU and the rest (CPU-heavy) on CPU — not "almost all GPU".
    /// Use 1.0 or omit the flag for GPU-only execution.
    #[arg(long, default_value_t = 0.0)]
    pub tensor_partition: f32,

    /// Chunked prefill: split long prompts into chunks to limit peak memory.
    /// 0 = auto (default): GPU backend derives a safe size from max_single_alloc()
    ///     to avoid CL_INVALID_BUFFER_SIZE; CPU backend processes entire prompt as one batch.
    #[arg(long, default_value_t = 0)]
    pub prefill_chunk_size: usize,

    /// Inter-chunk yield delay in milliseconds during prefill.
    /// After each prefill chunk, engine calls synchronize() + sleep(yield_ms).
    /// 0 = no yield. Dynamically adjustable via SetPrefillPolicy.
    #[arg(long, default_value_t = 0)]
    pub prefill_yield_ms: u32,

    /// CPU chunk size for GPU-CPU prefill interleaving.
    /// 0 = disabled. After each GPU chunk, CPU processes this many tokens.
    /// Requires --zero-copy or --resilience-prealloc-switch for weight access.
    #[arg(long, default_value_t = 0)]
    pub prefill_cpu_chunk_size: usize,

    /// Enable profiling (per-op timing, latency, score snapshots).
    ///
    /// Legacy mode: inserts two `clFinish()` calls per op on GPU, which
    /// inflates decode ms/tok by ~54 ms on Adreno. Useful for **relative**
    /// per-op ranking only. For apples-to-apples comparison with llama.cpp
    /// per-op GPU timing, use `--profile-events` instead.
    #[arg(long, default_value_t = false)]
    pub profile: bool,

    /// Enable OpenCL event-based per-op profiling.
    ///
    /// Creates the command queue with `CL_QUEUE_PROFILING_ENABLE` and
    /// captures a profiling event per kernel dispatch. At decode-step
    /// boundaries the `End-Start` nanoseconds are aggregated per logical
    /// op label. Unlike `--profile`, this adds no `clFinish()` calls and
    /// closely matches absolute GPU time (same mechanism as
    /// `GGML_OPENCL_PROFILING` in llama.cpp).
    ///
    /// Mutually exclusive with `--profile`.
    #[arg(long, default_value_t = false)]
    pub profile_events: bool,

    /// Enable GPU self-utilization measurement in Heartbeat (MSG-068 Phase 2).
    ///
    /// Turns on OpenCL queue profiling (same mechanism as `--profile-events`)
    /// and feeds the accumulated GPU busy ns into `EngineStatus.self_gpu_pct`
    /// so the Manager / LuaPolicy `ctx.engine.gpu_pct` reflects real usage
    /// instead of the Phase 1 hardcoded 0.0.
    ///
    /// **Overhead**: on Adreno, queue profiling adds ~54 ms/token. Keep OFF
    /// for production TBT measurements. OFF is the default — heartbeat
    /// `self_gpu_pct` stays at 0.0 (INV-092 fallback).
    ///
    /// If `--profile-events` is already set this flag is redundant; both
    /// share the same backend profiling infrastructure.
    #[arg(long, default_value_t = false)]
    pub heartbeat_gpu_profile: bool,

    /// Output directory for profiling data.
    #[arg(long, default_value = "results/profile")]
    pub profile_dir: String,

    /// Score snapshot interval (1 = every step, 10 = every 10th step).
    #[arg(long, default_value_t = 1)]
    pub profile_interval: usize,

    /// Comma-separated list of probes: ops,latency,scores,entropy,cache.
    #[arg(long, default_value = "ops,latency,scores")]
    pub profile_probes: String,

    /// Enable per-KV-head score tracking (for H2O+ analysis).
    #[arg(long, default_value_t = false)]
    pub profile_per_head: bool,

    /// Enable per-op CUDA event profiler (cuda-embedded backend only).
    ///
    /// Wraps each GPU kernel launch in a `cuEventRecord` pair and
    /// aggregates elapsed ms per op label at end-of-run. Label matrix
    /// matches OpenCL's `--profile-events` (matmul_qkv, matmul_wo,
    /// matmul_ffn, rms_norm, rope, attention, kv_update, silu_mul,
    /// lm_head) for apples-to-apples Adreno vs Jetson comparison.
    ///
    /// Independent of `--profile` and `--profile-events`. Writes
    /// `results/profile/cuda_embedded_decode_<timestamp>.json`.
    #[arg(long, default_value_t = false)]
    pub cuda_profile: bool,

    /// Per-category sync policy for the cuda-embedded backend. Lets us
    /// bisect which per-op `cuStreamSynchronize()` calls are load-bearing
    /// for correctness on Jetson UMA versus which are pure overhead.
    ///
    /// Values:
    /// - `all` (default): every launch-site sync stays on (pre-bisect
    ///   behaviour, ~28 tok/s on Xavier).
    /// - `none`: every per-op sync suppressed (equivalent to
    ///   `--cuda-defer-sync`; garbage output).
    /// - `llamacpp`: only the CPU-fallback guard stays on (garbage
    ///   output on Jetson UMA — residual `add_assign` loses cache
    ///   coherency without an intra-layer sync).
    /// - `minimal`: bisection-validated minimal correct set
    ///   (`elem_add` + `fallback`; ~34.8 tok/s on Xavier, +6.4 tok/s
    ///   vs `all`).
    /// - `custom:A,B`: comma-separated category names. Recognised
    ///   categories: `elementwise` (expands to `elem_add` +
    ///   `elem_act` + `elem_misc`), `elem_add`, `elem_act`,
    ///   `elem_misc`, `rmsnorm`, `rope`, `matmul`, `kv_scatter`,
    ///   `attention`, `gather`, `fallback`. Only the listed ones
    ///   keep syncing; everything else is deferred.
    ///
    /// `--cuda-defer-sync` still takes precedence when enabled.
    #[arg(long, default_value = "minimal")]
    pub cuda_sync_policy: String,

    /// Allocate weight tensors in device-only memory (`cuMemAlloc`) instead
    /// of UMA pinned host memory (`cuMemHostAlloc`) on Jetson.
    ///
    /// Jetson integrated GPUs expose the CPU DRAM to CUDA kernels through
    /// a pinned host-mapped alias, which gives zero-copy but weak L2 cache
    /// coherency when kernels read and the CPU writes (see llama.cpp
    /// `ggml-cuda.cu:241`, issue #15034). Weights are written once at load
    /// time and then read from every kernel for the rest of the run, so
    /// moving them off the UMA alias is the strongest lever for cache
    /// ordering without losing zero-copy on per-token activations /
    /// KV cache. No-op on discrete GPUs (managed memory already migrates
    /// weights to VRAM on first touch) and on non-CUDA backends.
    #[arg(long, default_value_t = false)]
    pub cuda_weights_device: bool,

    /// Experimental: bundle each decode step's kernel launches into a
    /// single CUDA Graph (captured and replayed once per token).
    ///
    /// Removes per-kernel driver launch overhead (~5 µs × ~400 launches
    /// = ~2 ms/tok on Jetson Xavier). Pays a per-step graph
    /// instantiate cost (~0.3-1 ms on Xavier) — net win is sensitive
    /// to the instantiate overhead actually measured.
    ///
    /// Currently a per-step re-capture baseline. Incompatible with
    /// `--cuda-profile`, `--profile`, and tensor partition; the
    /// inner decode path must not call `synchronize()`, `read_buffer`,
    /// or any CPU fallback while capture is active.
    #[arg(long, default_value_t = false)]
    pub cuda_graph: bool,

    /// Model weight data type (f16 or q4). f16 = no quantization, q4 = Q4_0 quantization at load time.
    #[arg(long, default_value = "f16")]
    pub weight_dtype: String,

    /// One-shot lm_head quantization at load time (`auto` | `none` | `q4_0`).
    ///
    /// Sprint F (2026-04-26): Recovers the +4.6 ms/tok Adreno gap that
    /// dominates "ratio=1.0 mixed" weight-swap regressions. F16 GGUFs ship
    /// lm_head as F16 (~524 MB), while Q4 GGUFs derive it from Q4_0
    /// embed_tokens. Quantizing lm_head once at load time matches the Q4
    /// baseline cost (~3.8 ms/call) without touching the AUF format.
    /// Embed_tokens stays untouched even on tied-weight models. No-op if
    /// lm_head is already Q4_0.
    ///
    /// `auto` (default): quantize when `--secondary-gguf` is set AND lm_head
    /// is currently F16/F32 (production-safe — pure win, no regression on
    /// Q4 baseline because lm_head is already Q4_0 there).
    /// `q4_0`: force quantize regardless of secondary-gguf presence.
    /// `none`: never quantize (legacy/diagnostic behaviour).
    #[arg(long, default_value = "auto")]
    pub quantize_lm_head: String,

    /// KV cache data type (f32, f16, or q4)
    #[arg(long, default_value = "f16")]
    pub kv_type: String,

    /// KV cache format by registry name (KV_FORMATS). 설정 시 `--kv-type` 보다 우선.
    /// 내장(f32/f16/q4_0/q8_0)은 typed 저장, 그 외 등록 format(예 `synth_q4`)은 opaque 저장(ADR-0008).
    #[arg(long)]
    pub kv_format: Option<String>,

    /// GATE-C: 런타임 stage plugin `.so` 경로(반복 가능, ADR-0009 D6). startup 에 dlopen +
    /// `register_kv_stage_v1` → `KV_CACHE_STAGES` 동적 등록. `--eviction-policy <plugin-name>` 로 선택.
    /// 빌트인과 이름 충돌 시 fail-fast(빌트인 우선).
    #[arg(long = "load-plugin")]
    pub load_plugin: Vec<std::path::PathBuf>,

    // ── Eviction (S-subcmd C2): policy/h2o/d2o/sink/streaming + common
    // 7 params (kv_budget, protected_prefix, memory_threshold_mb,
    // eviction_target_ratio, initial_kv_capacity, min_kv_cache,
    // kv_budget_ratio) moved to EvictionCmd subcommand + EvictionCommonArgs.
    // Existing call sites continue to read via shim accessors on `Args`
    // (see `impl Args` below). ──
    #[clap(flatten)]
    pub eviction_common: EvictionCommonArgs,

    /// Enable resilience manager for adaptive inference.
    /// Legacy generate 기준 flag. argus-cli v1+ 는 default-on 정책이며,
    /// 비활성화는 [`Self::no_resilience`] (`--no-resilience`) 를 사용한다.
    #[arg(long, default_value_t = false)]
    pub enable_resilience: bool,

    /// Disable resilience manager (argus-cli v1+ opt-out).
    /// argus-cli v1 에서는 resilience 가 default-on 이므로 비활성화하려면
    /// 이 flag 를 명시. legacy `generate` binary 는 이 flag 를 무시
    /// (default-off 정책 유지) — argus-cli main 에서만 [`Self::enable_resilience`]
    /// 를 effective 결정한다.
    #[arg(long, default_value_t = false)]
    pub no_resilience: bool,

    /// Pre-allocate dual CPU/GPU buffers for zero-alloc SwitchHw.
    /// Without this flag, only throttle/suspend directives work (no backend switch).
    /// Enables: zero-copy KV memory + weight dual-access rewrap (increases RSS by ~model size).
    #[arg(long, default_value_t = false)]
    pub resilience_prealloc_switch: bool,

    /// Resilience signal transport: "dbus" or "unix:<path>"
    #[arg(long, default_value = "dbus")]
    pub resilience_transport: String,

    // ── Experiment mode ──────────────────────────────
    /// Experiment schedule JSON file (enables experiment mode)
    #[arg(long)]
    pub experiment_schedule: Option<String>,

    /// Experiment output JSONL file path
    #[arg(long)]
    pub experiment_output: Option<String>,

    /// Number of top-K logits to record per token in experiment mode
    #[arg(long, default_value_t = 10)]
    pub experiment_logits_topk: usize,

    /// System metric sampling interval (N tokens, 0=disabled)
    #[arg(long, default_value_t = 1)]
    pub experiment_sample_interval: usize,

    /// Force greedy sampling (temperature=0) for reproducibility
    #[arg(long, default_value_t = false)]
    pub greedy: bool,

    /// Ignore EOS token and continue generating (for long-running experiments)
    #[arg(long, default_value_t = false)]
    pub ignore_eos: bool,

    /// Target TBT in milliseconds for pacing (0=disabled).
    /// After each decode step, sleeps to maintain the target TBT.
    /// Used for fair resource comparison across different actions at the same QoS.
    #[arg(long, default_value_t = 0.0)]
    pub target_tbt: f64,

    /// Fixed per-token throttle delay in milliseconds (0=disabled).
    /// Unconditional sleep after each decode step — useful for co-execution
    /// simulations without running a Manager. Manager `Throttle` directives
    /// override this value when resilience is enabled.
    #[arg(long, default_value_t = 0)]
    pub throttle_delay_ms: u64,

    /// OpenCL command-queue priority hint (`cl_khr_priority_hints`).
    /// "low" yields GPU scheduling to foreground apps (e.g. games) during
    /// co-execution. Falls back to normal priority with a warning if the
    /// driver does not advertise the extension. Also settable via env var
    /// `OCL_QUEUE_PRIORITY`.
    #[arg(long, value_parser = ["low", "medium", "normal", "high"], default_value = "normal")]
    pub gpu_priority: String,

    /// Path to write per-token TBT JSONL log.
    /// Each line: {"token_idx":N,"tbt_ms":X,"forward_ms":Y,"cache_pos":Z,"pacing_ms":W}
    #[arg(long)]
    pub tbt_log: Option<String>,

    /// KV cache memory layout: "head" (head-major) or "seq" (seq-major)
    #[arg(long, default_value = "head")]
    pub kv_layout: String,

    /// Override eviction target_ratio from resilience signals (experiment mode).
    /// When set, all Evict actions will use this ratio instead of the strategy default.
    #[arg(long)]
    pub experiment_eviction_ratio: Option<f32>,

    /// QCF variant to compute: "attn" (default), "caote", or "both".
    #[arg(long, default_value = "attn")]
    pub qcf_mode: String,

    // ── Eval-LL mode (log-likelihood evaluation) ──
    /// Enable log-likelihood evaluation mode (downstream task accuracy)
    #[arg(long, default_value_t = false)]
    pub eval_ll: bool,

    /// Continuation text to evaluate log-likelihood (single task mode)
    #[arg(long)]
    pub eval_continuation: Option<String>,

    /// Path to evaluation batch JSON file: [{"id","prompt","continuation"}, ...]
    #[arg(long)]
    pub eval_batch: Option<String>,

    /// Enable dynamic KV cache quantization for resilience.
    /// Starts with bits=16 (F16-equivalent KiviCache) and allows runtime
    /// transition to Q2/Q4/Q8 via kv_quant_dynamic resilience command.
    #[arg(long, default_value_t = false)]
    pub kv_dynamic_quant: bool,

    /// Number of threads for parallel computation.
    /// Default: auto-detect CPU core count.
    #[arg(long, default_value_t = 0)]
    pub threads: usize,

    /// Path to reference text file for perplexity evaluation (teacher-forcing).
    /// Measures PPL and collects proxy metrics during eviction.
    #[arg(long)]
    pub ppl: Option<String>,

    /// PPL mode 에서 weight swap 을 trigger 할 decode token index (0-based).
    /// 미지정 시 PPL decode loop 은 swap 없음 (baseline).
    /// Requires `--secondary-gguf` to load secondary weights.
    /// 사용 예 (LISWAP-PPL): `--ppl ref.txt --secondary-gguf ... --ppl-swap-at-token 0
    /// --ppl-swap-ratio 0.9 --ppl-swap-per-tick 1`.
    #[arg(long)]
    pub ppl_swap_at_token: Option<usize>,

    /// PPL swap ratio (0.0~1.0). engine `WeightSwapDecider` 가 [0.0, 1.0] 으로 clamp.
    /// 1.0 + `LLMRS_SWAP_ALLOW_BOUNDARY_LAYERS=1` 조합 시 전 layer swap.
    #[arg(long, default_value_t = 0.9)]
    pub ppl_swap_ratio: f32,

    /// PPL incremental swap K (layer/tick). 측정용으로 dynamic-K 비활성화, fixed K.
    /// 기본 1 (한 step 에 1 layer 씩).
    #[arg(long, default_value_t = 1)]
    pub ppl_swap_per_tick: usize,

    /// PPL per-token NLL CSV 출력 경로. 미지정 시 dump 안 함.
    /// CSV columns: phase, token_idx, token_id, nll, swap_state, layers_swapped.
    #[arg(long)]
    pub ppl_nll_csv: Option<std::path::PathBuf>,

    /// PPL prefill 토큰 수 강제 설정 (1..=eval_tokens). 미지정 시 기존 로직
    /// (kv_budget / sliding window / eval_tokens) 그대로. swap 측정 시 decode
    /// loop 을 충분히 길게 돌려야 하므로 이 옵션으로 prefill 을 짧게 만든다.
    /// 예: 1072 token reference 에서 `--ppl-prefill-tokens 32` → prefill 32 +
    /// decode 1040 step.
    #[arg(long)]
    pub ppl_prefill_tokens: Option<usize>,

    /// LISWAP-PPL Scenario E: swap 완료 후 KV cache reset + prefill 다시 시작.
    /// `--ppl-swap-at-token` + `--secondary-gguf` 필요. 워크플로:
    ///   pass 1 (warmup): prefill + swap-driving decode → plan_done 시 종료
    ///   pass 2 (measure): KV cache 0 으로 reset → prefill 다시 → decode (no swap)
    /// pass 2 의 NLL/PPL/CSV 만 기록. 가설: cache mismatch 가 C−D artifact 원인이면
    /// pass 2 결과가 D (Q4 native) 에 수렴해야 함.
    #[arg(long, default_value_t = false)]
    pub ppl_warmup_swap: bool,

    /// LISWAP-PPL Scenario F: `--ppl-warmup-swap` 의 pass 2 prefill 길이를 별도로
    /// 지정. 미지정 시 `--ppl-prefill-tokens` 값을 그대로 사용 (= 시나리오 E).
    /// 예: pass 1 prefill=32 (swap-driving decode 28 step) + pass 2 prefill=1072
    /// (decode loop 없음, batch path 전체) → batch path 만으로 weight 정합 검증.
    #[arg(long)]
    pub ppl_measure_prefill_tokens: Option<usize>,

    /// LISWAP-PPL diagnostic: 모델 로드 직후 모든 layer 의 weight tensor (wq/wk/wv/
    /// wo/w_gate/w_up/w_down) 를 readback 해서 `<dir>/layer{NN}_{name}_{dtype}.bin`
    /// 으로 dump. swap 없는 baseline 측정용 (e.g. Q4_0 native 모델 비교 기준).
    #[arg(long)]
    pub dump_q4_after_load: Option<std::path::PathBuf>,

    /// LISWAP-PPL diagnostic: `--ppl-warmup-swap` Pass 1 의 swap 완료 직후 (cache
    /// reset 직전) 모든 layer 의 weight tensor 를 readback 해서 dump. swap 경로의
    /// Q4 weight 가 standalone Q4 와 비트 단위로 일치하는지 검증용.
    #[arg(long)]
    pub dump_q4_after_swap: Option<std::path::PathBuf>,

    /// Comma-separated layer indices to skip (both attn+mlp).
    /// Example: --skip-layers 1,3,5,7
    #[arg(long, value_delimiter = ',')]
    pub skip_layers: Option<Vec<usize>>,

    /// Skip ratio (0.0-1.0). Uses SkipConfig::uniform_init() to select layers.
    #[arg(long)]
    pub skip_ratio: Option<f32>,

    /// Dump per-layer importance table and exit (no inference).
    /// Runs prefill with ImportanceCollector on the given prompt.
    #[arg(long, default_value_t = false)]
    pub dump_importance: bool,

    /// Path to JSONL file for multi-prompt batch generation.
    /// Each line: {"id":"...", "prompt":"..."} or {"id":"...", "prompt_file":"path"}
    /// Mutually exclusive with --prompt, --prompt-file, --eval-batch.
    #[arg(long)]
    pub prompt_batch: Option<String>,

    /// Loop prompt-batch: restart from beginning when all entries are processed.
    #[arg(long, default_value_t = false)]
    pub prompt_batch_loop: bool,

    /// Maximum iterations for prompt-batch loop (0 = unlimited).
    #[arg(long, default_value_t = 0)]
    pub max_iterations: usize,

    /// Start an interactive multi-turn chat REPL (Llama 3.2 Instruct / Qwen2).
    /// Uses standard (non-KIVI, non-offload) forward path.
    #[arg(long, default_value_t = false)]
    pub chat: bool,

    /// Optional system prompt injected as the first turn when --chat is set.
    #[arg(long)]
    pub system_prompt: Option<String>,

    /// Optional Unix domain socket path. When set, chat mode also accepts
    /// newline-delimited user messages from this socket in addition to stdin,
    /// and streams assistant replies back (terminated by 0x04).
    #[arg(long)]
    pub chat_socket: Option<String>,

    /// Optional TCP listen address (e.g. "127.0.0.1:7878"). Same protocol
    /// as --chat-socket: newline-delimited input, assistant reply bytes
    /// streamed back, 0x04 EOT delimiter per turn. Can be combined with
    /// --chat-socket; both listeners feed the same chat loop.
    #[arg(long)]
    pub chat_tcp: Option<String>,

    /// Directory used by `KvOffload` directives to write out the LRU prefix
    /// of the KV cache. When set, `CacheManager::enable_swap()` registers a
    /// disk-backed `SwapHandler`; without it the `KvOffload` directive is
    /// a warn-only no-op. `RestoreDefaults` triggers recall of offloaded data.
    #[arg(long)]
    pub swap_dir: Option<std::path::PathBuf>,

    /// Optional secondary GGUF path for runtime weight swap (Phase 2).
    /// When specified together with `--force-swap-ratio`, the engine swaps
    /// decoder layer weights from the primary dtype to the secondary dtype
    /// immediately before generation starts.
    /// When omitted, the weight swap path is disabled (ENG-DAT-C09).
    #[arg(long)]
    pub secondary_gguf: Option<std::path::PathBuf>,

    /// AUF primary backend variant 선택 (W-AUF-1 C4).
    /// AUF primary가 아닐 때 무시됨. default: auto.
    #[arg(long, default_value = "auto", value_parser = parse_primary_variant)]
    pub primary_variant: PrimaryVariantArg,

    /// AUF primary dtype 선택 (W-AUF-1 C4).
    /// AUF primary가 아닐 때 무시됨. default: auto (META.default_dtype 우선).
    #[arg(long, default_value = "auto", value_parser = parse_primary_dtype)]
    pub primary_dtype: PrimaryDtypeArg,

    /// AUF TOKENIZER에 eos_id가 비어있을 때 fallback override (W-AUF-1 C5).
    #[arg(long)]
    pub eos_token_id: Option<u32>,

    /// AUF TOKENIZER에 bos_id가 비어있을 때 fallback override (W-AUF-1 C5).
    #[arg(long)]
    pub bos_token_id: Option<u32>,

    /// AUF self-secondary 자동 활성 비활성 (W-AUF-2). 디버그/벤치마크용.
    #[arg(long, default_value_t = false)]
    pub no_self_secondary: bool,

    /// Manually trigger a weight swap before generation starts.
    /// Value: fraction of decoder layers to swap (0.0–1.0).
    /// Example: `--force-swap-ratio 0.5` swaps 50% of layers.
    /// Requires `--secondary-gguf` to be set; exits early with an error
    /// if the secondary path is absent.
    /// Intended for offline testing and debug; not for production use.
    #[arg(long)]
    pub force_swap_ratio: Option<f32>,

    /// Secondary dtype selection for AUF-backed weight swap (ENG-ALG-225, Sprint D).
    ///
    /// Controls which dtype entry is selected from a multi-dtype AUF file:
    ///   auto  — automatically select the best candidate dtype (default).
    ///           If META.default_dtype is set, that is used; otherwise the first
    ///           available candidate is picked.
    ///   q4_0  — explicitly select Q4_0 entries.
    ///   f16   — explicitly select F16 entries.
    ///   f32   — explicitly select F32 entries.
    ///
    /// Ignored for GGUF-backed secondaries (GGUF files carry a single dtype).
    /// Adreno SOA backend rejects f16 (SOA layout is Q4_0-only).
    #[arg(long, default_value = "auto", value_parser = parse_secondary_dtype)]
    pub secondary_dtype: SecondaryDtypeArg,

    /// AUF weights variant 선택 ("auto" | "aos" | "soa").
    ///
    /// `auto` (기본): feature flag 기반 preferred variant 우선 + AUF에 없으면
    /// CpuAos 자동 폴백. OpenCL build에선 AdrenoSoa 우선.
    ///
    /// `aos`: 강제 AOS (`WEIGHTS_CPU_AOS` / `WEIGHTS_CUDA_AOS`). swap 후
    /// `switch_hw cpu` / partition lazy-map / CPU forward가 정상 동작.
    /// GPU TBT는 SOA 대비 30~50% 저하 (Adreno 830 실측).
    ///
    /// `soa`: 강제 SOA (`WEIGHTS_ADRENO_SOA`, OpenCL 전용). 가장 빠르지만
    /// swap 후 host-pointer 부재로 switch_hw cpu / partition 호환 불가.
    ///
    /// GGUF secondary에선 무시됨.
    #[arg(long, default_value = "auto", value_parser = parse_secondary_layout)]
    pub secondary_layout: SecondaryLayoutArg,

    /// Explicit path to tokenizer.json. When omitted, the tokenizer is
    /// resolved automatically via the GGUF basename (e.g.
    /// `<dir>/<stem>.tokenizer.json`, then `<dir>/<stem-without-quant>.tokenizer.json`,
    /// then the legacy `<dir>/tokenizer.json` fallback). Required when
    /// multiple models share the same directory (e.g. `/data/local/tmp/`)
    /// because the legacy fallback can pick up a sibling model's tokenizer
    /// and silently produce garbage outputs.
    #[arg(long)]
    pub tokenizer_path: Option<std::path::PathBuf>,

    /// Dump per-run QCF/NLL/swap_set as a single JSON file (schema_version 1).
    ///
    /// Activates the warmup-prefill workflow: before the main measurement
    /// (--ppl or generation), a short prefill with N tokens collects the
    /// per-layer ImportanceTable used by WeightSwapDecider for accurate
    /// importance × ε bottom-k layer selection.
    ///
    /// When absent, all existing behavior is unchanged (--force-swap-ratio
    /// still uses the uniform fallback path as before).
    #[arg(long)]
    pub qcf_dump: Option<std::path::PathBuf>,

    /// Number of tokens for the warmup prefill that builds ImportanceTable.
    /// Only used when `--qcf-dump` is set. Default: 256.
    #[arg(long, default_value_t = 256)]
    pub qcf_warmup_tokens: usize,

    /// §4.2 decode-X experiment (EuroSys'27). When > 0, the QCF-dump warmup
    /// workflow runs `N` greedy-generation decode steps after the regular
    /// prefill and caches the per-layer hidden state at each decode step in
    /// a fresh collector. Two extra F5 vectors land in the dump JSON:
    /// - `direct_attn_f5_decode_only`: X = decode-only raws (T = N).
    /// - `direct_attn_f5_prefill_decode`: X = concat(prefill raws, decode raws) (T = 256 + N).
    ///
    /// The regular `direct_attn_f5` (prefill X, T = 256) is always written.
    /// Decode token 0 = argmax of prefill's final logits; subsequent decode
    /// tokens = argmax of each previous decode-step logits (greedy).
    /// Only meaningful when a secondary GGUF (Q4) is loaded.
    #[arg(long, default_value_t = 0)]
    pub decode_x_steps: usize,

    /// Layer-selection algorithm for `--qcf-dump` swap path (U5 ablation, EuroSys'27).
    /// Values: `imp` (importance-aware, default — production behavior),
    /// `seq` (sequential 0→N-1), `rev` (reverse N-1→0),
    /// `uni` (evenly spaced), `anti` (importance × ε descending top-k — worst-case).
    /// Only affects the `--qcf-dump` warmup-prefill swap; the manager / live
    /// directive paths always use `imp`.
    #[arg(long, default_value = "imp")]
    pub swap_algorithm: String,

    /// Layer importance formula for the §4 comparison study (EuroSys'27).
    ///
    /// - `mean_pool` (default): `1 − cos(mean_pool(h_in), mean_pool(h_out))`,
    ///   current ARGUS baseline (token-wise mean-pool then cosine).
    /// - `shortgpt_bi`: `1 − (1/T) Σ_t cos(h_in,t, h_out,t)` — ShortGPT BI
    ///   (Men et al., 2024), token-wise cosine then mean.
    /// - `dpllm_proxy`: input-aware perturbation via
    ///   `‖(W_F16 − W_Q4) · x_mean‖ / ‖W_F16 · x_mean‖` on attn_output.weight.
    /// - `compare`: collect all three side-by-side. ImportanceTable still
    ///   uses `mean_pool` for swap decisions; the other two are recorded
    ///   only in the dump JSON's `per_layer_3way` field.
    #[arg(long, default_value = "mean_pool")]
    pub importance_formula: String,

    /// Explicit per-layer swap list (CSV of layer indices) for §4 ground-truth
    /// study. Bypasses `WeightSwapDecider`: when set, the listed layers are
    /// swapped regardless of `--force-swap-ratio` or `--swap-algorithm`.
    /// `--force-swap-ratio` must still be provided (any non-zero value) so
    /// the warmup workflow runs; the ratio itself is ignored. Example:
    /// `--swap-only-layers 5` swaps only layer 5.
    #[arg(long)]
    pub swap_only_layers: Option<String>,

    /// Per-step NLL trajectory mode (U5 mid-swap quality study, EuroSys'27).
    ///
    /// Requires `--qcf-dump`, `--force-swap-ratio`, `--eval-ll`, `--eval-batch`.
    /// Workflow:
    ///   1. warmup prefill → ImportanceTable
    ///   2. WeightSwapDecider.decide(ratio, algorithm) → ordered layer list
    ///      of length K = floor(ratio × num_layers).
    ///   3. for t = 0..=K:
    ///      a. run eval-ll on the eval batch → record EvalOutput_t.
    ///      b. if t < K: SwapExecutor.execute_on_slots(&[selected_layers[t]]).
    ///   4. dump JSON with trajectory: array of K+1 (step, swapped_layers,
    ///      layer_added, eval_ll_output).
    ///
    /// The cumulative swap state at step t mirrors ARGUS's production
    /// incremental swap (one layer per token), letting external analysis
    /// observe the mid-swap NLL trajectory rather than only the final state.
    #[arg(long, default_value_t = false)]
    pub qcf_trajectory: bool,

    /// Enable QCF v3 schema metric dump for EuroSys'27 §3.
    /// Adds qcf_layer_worst_head/qcf_layer_mean_head/qcf_record_*/qcf_d7_*/
    /// qcf_c1_* (schema_version=3) to eval-ll output for both Eviction and KIVI.
    #[arg(long, default_value_t = false)]
    pub enable_qcf_experimental: bool,

    /// Sample layer indices for multi-layer QCF.
    /// Default is "all" (schema v3: every decoder layer) — required for D7/C1.
    /// Other values: "auto" (legacy 5-tuple [0, n/4, n/2, 3n/4, n-1]),
    /// "0,8,16,24,31" (explicit indices).
    #[arg(long, default_value = "all")]
    pub qcf_sample_layers: String,

    /// Eagerly prefault the secondary weight file at model load to remove
    /// per-swap prefault stage cost. Memory commit ≈ AUF size (e.g. 1.2 GB
    /// for Qwen2.5-1.5B Q4_0). Default off; set when --secondary-gguf is
    /// present and on-device app has memory headroom.
    ///
    /// When enabled: immediately after model weights are loaded, the full
    /// secondary weight region is touched (madvise WILLNEED + explicit
    /// page-touch). Subsequent swap invocations find all pages already in
    /// the page cache, eliminating the ~328 ms cold-fault stage measured on
    /// Galaxy S25 (§3.1, swap_overhead_s25.md).
    ///
    /// When `--secondary-gguf` is absent this flag is silently ignored.
    #[arg(long, default_value_t = false)]
    pub eager_prefault_secondary: bool,

    /// Layer-Incremental Swap Stage 1 MVP (LISWAP-1, ENG-ALG-232~234, INV-144~146).
    ///
    /// Number of decoder layers to swap per decode token tick.
    ///
    /// `0` (default): single-shot path — all target layers are swapped at once
    /// before generation, exactly as before. No behavior change.
    ///
    /// `>= 1`: incremental path — when `--force-swap-ratio` is set, instead of
    /// swapping all target layers immediately, an `IncrementalSwapPlan` is
    /// committed and the swap is distributed across decode tokens (N layers/tick).
    ///
    /// **Trade-off**: total swap latency increases by stage gate overhead
    /// (~7.4 ms × N ticks on Galaxy S25), but user-perceived per-token stall
    /// is bounded to `total_swap_latency / ceil(n_layers / per_tick)`.
    ///
    /// Example: 25 layers, per_tick=2 → 13 ticks × ~23 ms stall vs 290 ms
    /// single-shot stall (9 frames vs 0 frames skipped at 30 fps).
    ///
    /// Requires `--force-swap-ratio` to be set. When absent, has no effect.
    /// Per-tick > 0 with no `--force-swap-ratio`: silently ignored (no trigger).
    #[arg(long, default_value_t = 0)]
    pub swap_incremental_per_tick: usize,

    /// LISWAP-2 prototype: Submit incremental swap chunks to a separate
    /// transfer queue/stream so weight H2D writes overlap with the next
    /// token's forward compute.
    ///
    /// Requires `--swap-incremental-per-tick > 0`. When `=0` or absent,
    /// has no effect (silently ignored).
    ///
    /// **Default OFF (2026-05-13)** — async path는 `SwapExecutor`의
    /// sub-batch reactive pause(release_pending > 0 → break)를 우회한다.
    /// dispatcher worker로 release를 위임하기 때문에 main thread는
    /// pending=0인 채 batch 전체를 enqueue → release_worker queue burst →
    /// 메모리 스파이크. production swap default = sync. async는 측정용
    /// (ablation) 시에만 명시적으로 enable.
    #[arg(long, default_value_t = false)]
    pub swap_async_dispatch: bool,

    /// LISWAP-3 prototype (Direction A): use a `CL_MEM_ALLOC_HOST_PTR` slot
    /// pool for swap weight upload. Bypasses the driver staging copy by
    /// running `clEnqueueMapBuffer(MAP_WRITE) → memcpy → Unmap` on a
    /// pre-allocated zero-copy slot.
    ///
    /// **Default OFF**. Requires `LLMRS_OPENCL_HOST_PTR_POOL=1` env-gate
    /// in addition to this flag — the env hard-disables the pool path
    /// independently so the flag alone is insufficient (Stage 4
    /// measurement-driven decision pending).
    ///
    /// Compatible with `--swap-incremental-per-tick > 0` and standalone
    /// (single-shot `--force-swap-ratio`). Falls back to the staging path
    /// when slots are exhausted, the env-gate is unset, the backend is
    /// not OpenCL, or pool init fails. Plan: `compiled-chasing-hopper.md`
    /// Direction A track, Stage 3.
    #[arg(long, default_value_t = false)]
    pub swap_zero_copy: bool,

    /// LISWAP-3 prototype: number of slots in the `CL_MEM_ALLOC_HOST_PTR`
    /// swap pool. Stage 2 measurement on Galaxy S25 (Qwen2.5-1.5B, 28
    /// layers, 7 Q4_0 tensors per layer) reported a sweet spot at 14
    /// slots (= 2 layers worth of in-flight work). Effective only with
    /// `--swap-zero-copy` + `LLMRS_OPENCL_HOST_PTR_POOL=1`.
    #[arg(long, default_value_t = 14)]
    pub swap_pool_slots: usize,

    /// Intra-forward Layer-aligned Swap (LISWAP-4, ENG-ALG-235~238,
    /// INV-147~150).
    ///
    /// `false` (default): no-op — `forward_into` carries
    /// `layer_boundary_hook = None` and the layer loop pays only one
    /// `Option::is_some` branch per layer (INV-147 zero overhead).
    ///
    /// `true`: when `--force-swap-ratio` is set, an `IntraForwardSwapHook`
    /// is committed and dispatches per-layer swap on the layer boundary.
    /// Plan runs to completion across decode tokens; dispatcher drain +
    /// `ratio_generation` bump occurs once on plan retire (INV-150).
    ///
    /// Mutually exclusive with `--swap-incremental-per-tick > 0`
    /// (ENG-DAT-C18). CLI parser rejects the combination.
    #[arg(long, default_value_t = false)]
    pub swap_intra_forward: bool,

    /// Phase-aware Async Weight Swap (LISWAP-5).
    ///
    /// `false` (default): no-op. PHASE_HOOK 미등록 → forward path는
    /// `op_trace::start_op` / `record`에서 atomic load 1회 + 분기로 끝남
    /// (zero overhead).
    ///
    /// `true`: `--force-swap-ratio`가 설정되면 `PhaseAwareSwapDispatcher`가
    /// commit되고, `op_trace` boundary에서 phase를 검사하여:
    /// - `DdrPhase::CacheFit` 끝 → 다음 chunk async H2D enqueue
    /// - `DdrPhase::Heavy` 시작 직전 → in-flight chunk 완료 대기
    ///
    /// `OpKind::ddr_phase()` 분류 + Phase R Scenario B (1.04× of max GREEN) +
    /// production op CV 1.2% 측정 결과를 활용하여 swap H2D를 forward GPU
    /// compute와 overlap.
    ///
    /// `--swap-incremental-per-tick > 0` / `--swap-intra-forward`와 mutually
    /// exclusive.
    #[arg(long, default_value_t = false)]
    pub swap_phase_aware: bool,

    /// LISWAP-6 Phase 6 — Per-layer immediate swap (LISWAP-4 alias-skip variant).
    ///
    /// `false` (default): no-op.
    ///
    /// `true`: when `--force-swap-ratio` is set, an `IntraForwardSwapHook` is
    /// committed (identical infrastructure to `--swap-intra-forward`) but the
    /// log line is tagged `layer-immediate` to make the measurement matrix
    /// distinguishable. With LISWAP-6 Phase 5b alias H2D-skip applied to the
    /// `build_layer_from_mmap_async` weight closure (swap_executor.rs:961~),
    /// every per-layer dispatch returns dummy events and `process_commit`
    /// short-circuits the `wait_event_blocking` fall-through. Result: 28 layer
    /// dispatches with zero `synchronize()` accumulation when the secondary is
    /// rpcmem DMA-BUF aliased.
    ///
    /// Mutually exclusive with `--swap-incremental-per-tick > 0` /
    /// `--swap-intra-forward` / `--swap-phase-aware`.
    #[arg(long, default_value_t = false)]
    pub swap_layer_immediate: bool,

    /// 4 swap 모드 통합 shorthand (backlog P3, 2026-05-25).
    ///
    /// `--swap` (단독): default = IntraForward (LISWAP-4 production winner).
    /// `--swap <MODE>`: 명시 모드 선택. `intra-forward` / `incremental` /
    /// `phase-aware` / `layer-immediate`.
    ///
    /// `--swap` 사용 시 init 단계에서 legacy 4 flag (`--swap-intra-forward`
    /// 등)로 변환되어 기존 dispatch path가 그대로 동작한다. `Incremental` 모드
    /// 선택 시 `--swap-incremental-per-tick K`가 0이면 K=2 default 적용.
    ///
    /// 기존 4 flag 직접 사용은 deprecated — init.rs에서 stderr 1회 경고
    /// (`--swap`으로 마이그레이션 권장). 동작은 그대로 보존.
    #[arg(long, value_parser = parse_swap_mode, num_args = 0..=1, default_missing_value = "intra-forward")]
    pub swap: Option<SwapMode>,

    /// LISWAP-6 Dynamic-K controller — auto-tune `--swap-incremental-per-tick`
    /// based on measured per-layer release cost vs forward wall.
    ///
    /// Requires `--swap-incremental-per-tick > 0`. The explicit value is the
    /// *starting* per_tick; the controller recomputes K from Phase 0 calibration
    /// timing (no static upper cap as of 2026-05-13 — `dynamic_k.rs` hard_upper
    /// removed). Effective only together with `--swap-async-dispatch`.
    ///
    /// Memory-spike avoidance is the hard constraint: K is monotone
    /// non-increasing after calibration and a reactive pause skips swap when
    /// the release queue is non-empty. See `dynamic_k.rs` for the algorithm.
    ///
    /// **Default OFF (2026-05-13)** — async path 동반 flag. async가 default
    /// off로 바뀌면서 dynamic-K도 동반 default off (단독 의미 없음). ARGUS
    /// 측정 시 명시적으로 enable.
    #[arg(long, default_value_t = false)]
    pub swap_dynamic_k: bool,

    /// Probing-K adaptive controller — bottom-up alternative to `--swap-dynamic-k`
    /// (ARGUS). Starts at `K = 1` and probes upward whenever `release_pending`
    /// stays at 0 for a stability window. On any spike the controller drops
    /// `K -= 1` symmetrically. Mutually exclusive with `--swap-dynamic-k`.
    ///
    /// Requires `--swap-incremental-per-tick > 0` (initial value ignored — the
    /// controller starts from 1) and `--swap-async-dispatch`.
    #[arg(long, default_value_t = false)]
    pub swap_probing_k: bool,

    /// Probing-K growth schedule when probing up. `linear` adds 1; `binary`
    /// doubles. Only effective with `--swap-probing-k`.
    #[arg(long, default_value = "linear")]
    pub swap_probing_growth: String,

    /// Probing-K stability window (clean tokens before probing up). Only with
    /// `--swap-probing-k`.
    #[arg(long, default_value_t = 5)]
    pub swap_probing_window: usize,

    /// Phase-aware swap chunk 진단 size (MB). v1 per-tensor chunking에서는
    /// 실제 분할에 사용되지 않고 진단/보고 용도. 측정에 따라 v2에서 sub-tensor
    /// chunking 도입 시 활용 (4 MB sweet spot 기본).
    #[arg(long, default_value_t = 4)]
    pub swap_phase_aware_chunk_mb: usize,

    /// Phase-aware swap throttle — token당 dispatch chunk 수 상한.
    /// 0 = 무제한 (현재 동작 유지, 252 chunks가 첫 3 token에 누적).
    /// N>0 = 매 token N chunks까지만 → 분산을 더 길게 펼쳐 max-stall 단축.
    /// Sweep 측정용 (Phase 2): K = {1,2,4,8,16}.
    #[arg(long, default_value_t = 0)]
    pub swap_phase_aware_max_chunks_per_token: usize,

    /// LISWAP Phase 3 — defer force-swap trigger to decode token N (mid-decode).
    ///
    /// 0 (default) = no delay (swap fires right after prefill, current behavior).
    /// N > 0       = first N decode tokens run on the primary weight, then the
    ///               force-swap trigger (single-shot `run_layer_swap`,
    ///               incremental plan commit, intra-forward hook commit, or
    ///               phase-aware dispatcher arm) fires at the start of token N.
    ///
    /// `prefault_layers` (eager pre-warm) is always executed at prefill end
    /// regardless of this flag — only the actual swap dispatch is deferred.
    ///
    /// Requires `--force-swap-ratio` to be set; otherwise ignored.
    #[arg(long, default_value_t = 0)]
    pub swap_delay_tokens: usize,

    /// Measurement-only (EuroSys 2027 §4.2): bypass the INV-141 release_worker
    /// drain at `SwapExecutor::execute_on_slots` entry so
    /// `--swap-incremental-per-tick K` fires every decode token at the
    /// user-requested K rate instead of being throttled by the previous
    /// batch's release backlog. Equivalent to setting
    /// `LLMRS_SWAP_FORCE_EVERY_TICK=1` (the flag sets the env if unset).
    ///
    /// **Memory-spike risk** — production code path keeps INV-141 to prevent
    /// displaced primary cl_mem from accumulating on a slow release path.
    /// Use only for layer-count predictor accuracy measurement and similar
    /// experiments where chunk-completion-induced throttle distorts the rate.
    #[arg(long, default_value_t = false)]
    pub swap_no_throttle: bool,

    /// Top-level subcommand wrapper.
    ///
    /// `eviction <policy>` form is the only currently registered
    /// subcommand. Omitting the subcommand ≡ `EvictionCmd::None`
    /// (no eviction). See [`crate::session::cli::TopLevelCmd`] and
    /// [`crate::session::cli::EvictionCmd`].
    #[command(subcommand)]
    pub eviction: Option<TopLevelCmd>,

    // ── KV mode subcommand (S-subcmd C4) ─────────────────────────────────
    #[clap(flatten)]
    pub kv_mode_args: KvModeArgs,
}

/// Shim accessors for the eviction subcommand + flatten common args.
///
/// Existing 175+ call sites (`args.eviction_policy`, `args.h2o_keep_ratio`,
/// `args.kv_budget`, ...) read through these methods so the C2 commit
/// changes only `cli/mod.rs`. Call sites migrate to direct enum match in C3.
impl Args {
    /// Normalize `--swap` shorthand to legacy 4 flags (backlog P3, 2026-05-25).
    ///
    /// `Args::parse()` 직후 1회 호출. `--swap` set 시 해당 legacy field 활성화
    /// (Incremental은 `--swap-incremental-per-tick`가 0이면 K=2 default).
    /// `--swap` unset + legacy 4 flag 직접 사용 시 stderr 1회 deprecation 경고.
    /// 이후 dispatch path는 기존 4 field만 읽으면 됨.
    pub fn normalize_swap_shorthand(&mut self) {
        if let Some(mode) = self.swap {
            match mode {
                SwapMode::IntraForward => self.swap_intra_forward = true,
                SwapMode::Incremental => {
                    if self.swap_incremental_per_tick == 0 {
                        self.swap_incremental_per_tick = 2;
                    }
                }
                SwapMode::PhaseAware => self.swap_phase_aware = true,
                SwapMode::LayerImmediate => self.swap_layer_immediate = true,
            }
        } else {
            let used_legacy = self.swap_incremental_per_tick > 0
                || self.swap_intra_forward
                || self.swap_phase_aware
                || self.swap_layer_immediate;
            if used_legacy {
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| {
                    eprintln!(
                        "[deprecation] --swap-incremental-per-tick / --swap-intra-forward / \
                         --swap-phase-aware / --swap-layer-immediate 직접 사용은 향후 제거 예정. \
                         `--swap [intra-forward|incremental|phase-aware|layer-immediate]` 통합 \
                         flag로 마이그레이션 권장 (backlog P3, 2026-05-25)."
                    );
                });
            }
        }
    }

    /// Engine 내부 dispatch default mode 결정.
    ///
    /// Manager `SwapWeights` 수신 시 어느 mode (Incremental / IntraForward /
    /// PhaseAware / LayerImmediate) 로 dispatch 할지의 default. `--swap` enum
    /// 우선, 미지정 시 legacy 4 flag로부터 추론, 모두 미지정이면 LISWAP-4
    /// production winner (IntraForward) 기본.
    ///
    /// `normalize_swap_shorthand()` 후 호출 권장 (legacy field 일치 보장).
    /// 상세 mental model: arch/weight_swap.md §2.8.1.
    pub fn resolved_swap_mode(&self) -> SwapMode {
        if let Some(mode) = self.swap {
            mode
        } else if self.swap_phase_aware {
            SwapMode::PhaseAware
        } else if self.swap_layer_immediate {
            SwapMode::LayerImmediate
        } else if self.swap_intra_forward {
            SwapMode::IntraForward
        } else if self.swap_incremental_per_tick > 0 {
            SwapMode::Incremental
        } else {
            SwapMode::IntraForward
        }
    }

    /// KV mode (단순 reader — legacy fallback 제거됨, 옵션 C 완료).
    pub fn effective_kv_mode(&self) -> KvMode {
        self.kv_mode_args.kv_mode
    }

    /// KIVI quantization bits.
    pub fn effective_kivi_bits(&self) -> u8 {
        self.kv_mode_args.kv_kivi_bits
    }

    /// KIVI residual buffer size.
    pub fn effective_kivi_residual_size(&self) -> usize {
        self.kv_mode_args.kv_kivi_residual_len
    }

    /// Offload storage backend. Offload 모드가 아니면 빈 문자열.
    pub fn effective_kv_offload_storage(&self) -> String {
        match self.kv_mode_args.kv_mode {
            KvMode::Offload => self.kv_mode_args.kv_offload_storage.clone(),
            _ => String::new(),
        }
    }

    /// ENG-RPCMEM-041 / INV-RPCMEM-006: effective `--opencl-rpcmem` 값.
    ///
    /// Sprint 2b: qnn_oppkg backend 제거됨. `--backend qnn_oppkg | qnngpu` 는
    /// 실제 backend init 에서 unknown backend 로 bail 하므로 이 분기는 production
    /// 경로에서 unreachable 하다. INV-RPCMEM-006 spec test 호환을 위해 보존.
    pub fn effective_opencl_rpcmem(&self) -> bool {
        if self.backend == "qnn_oppkg" || self.backend == "qnngpu" {
            false
        } else {
            self.opencl_rpcmem
        }
    }

    /// Returns the nested `EvictionCmd` policy, unwrapping the
    /// `TopLevelCmd::Eviction` wrapper. `None` if no subcommand given.
    fn current_policy(&self) -> Option<&EvictionCmd> {
        match &self.eviction {
            Some(TopLevelCmd::Eviction { policy }) => Some(policy),
            None => None,
        }
    }

    pub fn eviction_policy(&self) -> &'static str {
        self.current_policy()
            .map(|e| e.policy_name())
            .unwrap_or("none")
    }

    pub fn eviction_window(&self) -> usize {
        match self.current_policy() {
            Some(EvictionCmd::Sliding(s)) => s.window,
            _ => 1024,
        }
    }

    pub fn sink_size(&self) -> usize {
        match self.current_policy() {
            Some(EvictionCmd::Streaming(s)) => s.sink,
            _ => 4,
        }
    }

    pub fn streaming_window(&self) -> usize {
        match self.current_policy() {
            Some(EvictionCmd::Streaming(s)) => s.recent_window,
            _ => 0,
        }
    }

    pub fn h2o_keep_ratio(&self) -> f32 {
        match self.current_policy() {
            Some(EvictionCmd::H2o(h)) | Some(EvictionCmd::H2oPlus(h)) => h.keep_ratio,
            _ => 0.5,
        }
    }

    pub fn h2o_tracked_layers(&self) -> usize {
        match self.current_policy() {
            Some(EvictionCmd::H2o(h)) | Some(EvictionCmd::H2oPlus(h)) => h.tracked_layers,
            _ => 0,
        }
    }

    pub fn h2o_decay(&self) -> f32 {
        match self.current_policy() {
            Some(EvictionCmd::H2o(h)) | Some(EvictionCmd::H2oPlus(h)) => h.decay,
            _ => 0.0,
        }
    }

    pub fn h2o_raw_scores(&self) -> bool {
        match self.current_policy() {
            Some(EvictionCmd::H2o(h)) | Some(EvictionCmd::H2oPlus(h)) => h.raw_scores,
            _ => false,
        }
    }

    /// H2O verbose debug output — moved to env var `LLMRS_H2O_DEBUG`
    /// (no longer a CLI flag).
    pub fn h2o_debug(&self) -> bool {
        std::env::var("LLMRS_H2O_DEBUG").is_ok()
    }

    pub fn d2o_keep_ratio(&self) -> f32 {
        match self.current_policy() {
            Some(EvictionCmd::D2o(d)) => d.keep_ratio,
            _ => 0.75,
        }
    }

    pub fn d2o_ema_beta(&self) -> f32 {
        match self.current_policy() {
            Some(EvictionCmd::D2o(d)) => d.ema_beta,
            _ => 0.7,
        }
    }

    pub fn d2o_merge_e(&self) -> f32 {
        match self.current_policy() {
            Some(EvictionCmd::D2o(d)) => d.merge_e,
            _ => 0.1,
        }
    }

    pub fn d2o_layer_alloc(&self) -> bool {
        match self.current_policy() {
            Some(EvictionCmd::D2o(d)) => d.layer_alloc,
            _ => false,
        }
    }

    pub fn d2o_protected_layers(&self) -> Option<Vec<usize>> {
        match self.current_policy() {
            Some(EvictionCmd::D2o(d)) => d.protected_layers.clone(),
            _ => None,
        }
    }

    // ── EvictionCommonArgs shim (flatten field 호출처 호환) ──
    pub fn kv_budget(&self) -> usize {
        self.eviction_common.kv_budget
    }
    pub fn kv_budget_ratio(&self) -> f32 {
        self.eviction_common.kv_budget_ratio
    }
    pub fn protected_prefix(&self) -> Option<usize> {
        self.eviction_common.protected_prefix
    }
    pub fn memory_threshold_mb(&self) -> usize {
        self.eviction_common.memory_threshold_mb
    }
    pub fn eviction_target_ratio(&self) -> f32 {
        self.eviction_common.eviction_target_ratio
    }
    pub fn initial_kv_capacity(&self) -> usize {
        self.eviction_common.initial_kv_capacity
    }
    pub fn min_kv_cache(&self) -> usize {
        self.eviction_common.min_kv_cache
    }
}
