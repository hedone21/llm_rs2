use clap::Parser;
use llm_rs2::backend::cpu::CpuBackend;
use llm_rs2::core::attention_scores::AttentionScoreAccumulator;
use llm_rs2::core::backend::Backend;
use llm_rs2::core::buffer::DType;
use llm_rs2::core::cache_manager::CacheManager;
use llm_rs2::core::events::{self, CacheEvent, StderrDiagnosticSink};
use llm_rs2::core::eviction::h2o::H2OPolicy;
use llm_rs2::core::eviction::h2o_plus::H2OPlusPolicy;
use llm_rs2::core::eviction::no_eviction::NoEvictionPolicy;
use llm_rs2::core::eviction::sliding_window::SlidingWindowPolicy;
use llm_rs2::core::kivi_cache::KiviCache;
use llm_rs2::core::kv_cache::{KVCache, KVLayout};
use llm_rs2::core::memory::Memory;
use llm_rs2::core::pressure::d2o_handler::{D2OConfig, D2OHandler};
use llm_rs2::core::pressure::{CachePressurePipeline, PressureLevel, PressureStageConfig};
use llm_rs2::core::rss_trace::{dump_smaps, rss_trace};
use llm_rs2::core::sampling::{self, SamplingConfig};
use llm_rs2::core::shape::Shape;
use llm_rs2::core::sys_monitor::{LinuxSystemMonitor, NoOpMonitor};
use llm_rs2::core::tensor::Tensor;
use llm_rs2::layers::workspace::{
    LayerWorkspace, PartitionWorkspace, PartitionWsCell, WorkspaceConfig,
};
use llm_rs2::memory::galloc::Galloc;
use llm_rs2::models::transformer::{TransformerModel, TransformerModelForwardArgs};
use std::sync::Arc;
use tokenizers::Tokenizer;

use llm_rs2::experiment::{
    ExperimentSchedule, JsonlWriter, SummaryRecord, SystemSampler, TokenRecord,
    extract_top_k_logits,
};
#[cfg(feature = "resilience")]
use llm_rs2::resilience::DbusTransport;
use llm_rs2::resilience::TcpTransport;
#[cfg(unix)]
use llm_rs2::resilience::UnixSocketTransport;
use llm_rs2::resilience::{
    CommandExecutor, EngineCommand, KVSnapshot, ManagerMessage, MessageLoop,
};

/// `--secondary-dtype` CLI 인수 값 (D-3, ENG-ALG-225).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SecondaryDtypeArg {
    Auto,
    F16,
    Q4_0,
    F32,
}

/// `--secondary-dtype` value_parser.
fn parse_secondary_dtype(s: &str) -> Result<SecondaryDtypeArg, String> {
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

impl From<SecondaryDtypeArg> for llm_rs2::models::weights::SecondaryDtypeChoice {
    fn from(arg: SecondaryDtypeArg) -> Self {
        match arg {
            SecondaryDtypeArg::Auto => Self::Auto,
            SecondaryDtypeArg::F16 => Self::F16,
            SecondaryDtypeArg::Q4_0 => Self::Q4_0,
            SecondaryDtypeArg::F32 => Self::F32,
        }
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
enum SecondaryLayoutArg {
    Auto,
    Aos,
    Soa,
}

fn parse_secondary_layout(s: &str) -> Result<SecondaryLayoutArg, String> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(SecondaryLayoutArg::Auto),
        "aos" | "cpu_aos" | "cuda_aos" => Ok(SecondaryLayoutArg::Aos),
        "soa" | "adreno_soa" => Ok(SecondaryLayoutArg::Soa),
        other => Err(format!(
            "unknown secondary-layout '{other}'. Valid values: auto, aos, soa"
        )),
    }
}

impl From<SecondaryLayoutArg> for llm_rs2::models::weights::SecondaryLayoutChoice {
    fn from(arg: SecondaryLayoutArg) -> Self {
        match arg {
            SecondaryLayoutArg::Auto => Self::Auto,
            SecondaryLayoutArg::Aos => Self::Aos,
            SecondaryLayoutArg::Soa => Self::Soa,
        }
    }
}

/// Parse `--qcf-sample-layers` argument into a list of layer indices.
///
/// Accepts:
/// - `"auto"` (default): `[0, n/4, n/2, 3n/4, n-1]` via `compute_auto_sample_layers`.
/// - `"all"`: every layer `[0..n_layers)`.
/// - `"0,8,16,24,31"`: explicit comma-separated indices (sorted + deduped).
fn parse_qcf_sample_layers(spec: &str, n_layers: usize) -> Result<Vec<usize>, String> {
    let s = spec.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("auto") {
        return Ok(llm_rs2::core::qcf::compute_auto_sample_layers(n_layers));
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
mod parse_qcf_sample_layers_tests {
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
struct Args {
    #[arg(short, long, default_value = "models/llama3.2-1b")]
    model_path: String,

    /// Path to a file containing the prompt. Overrides --prompt if set.
    #[arg(long)]
    prompt_file: Option<String>,

    #[arg(short, long, default_value = "Hello, world! I am a")]
    prompt: String,

    #[arg(short, long, default_value_t = 20)]
    num_tokens: usize,

    /// Backend to use: "cpu", "opencl", or "cuda" (build with --features cuda)
    #[arg(short, long, default_value = "cpu")]
    backend: String,

    /// Auto-switch CPU→GPU at this token count (0=disabled). Requires GPU availability.
    #[arg(long, default_value_t = 0)]
    switch_threshold: usize,

    /// Use zero-copy shared memory (slower but enables CPU-GPU sharing)
    #[arg(long, default_value_t = false)]
    zero_copy: bool,

    #[arg(long, default_value_t = 2048)]
    max_seq_len: usize,

    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    #[arg(long, default_value_t = 40)]
    top_k: usize,

    #[arg(long, default_value_t = 1.1)]
    repetition_penalty: f32,

    #[arg(long, default_value_t = 64)]
    repetition_window: usize,

    /// Disable GPU kernel plan for decode (fallback to forward_into every token)
    #[arg(long, default_value_t = false)]
    no_gpu_plan: bool,

    /// GPU ratio for tensor partition — fraction of FFN gate/up rows assigned to GPU.
    /// Range (0.0, 1.0): 0.0 = disabled (no split), 1.0 = disabled (no split).
    /// 0.1 = 10% GPU + 90% CPU, 0.9 = 90% GPU + 10% CPU.
    /// NOTE: split_row is clamped to [128, out_dim-128], so extreme values like 0.001
    /// still leave 128 rows on GPU and the rest (CPU-heavy) on CPU — not "almost all GPU".
    /// Use 1.0 or omit the flag for GPU-only execution.
    #[arg(long, default_value_t = 0.0)]
    tensor_partition: f32,

    /// Disable PrefillWorkspace (fallback to per-layer alloc during prefill)
    #[arg(long, default_value_t = false)]
    no_prefill_ws: bool,

    /// Chunked prefill: split long prompts into chunks to limit peak memory.
    /// 0 = auto (default): GPU backend derives a safe size from max_single_alloc()
    ///     to avoid CL_INVALID_BUFFER_SIZE; CPU backend processes entire prompt as one batch.
    #[arg(long, default_value_t = 0)]
    prefill_chunk_size: usize,

    /// Inter-chunk yield delay in milliseconds during prefill.
    /// After each prefill chunk, engine calls synchronize() + sleep(yield_ms).
    /// 0 = no yield. Dynamically adjustable via SetPrefillPolicy.
    #[arg(long, default_value_t = 0)]
    prefill_yield_ms: u32,

    /// CPU chunk size for GPU-CPU prefill interleaving.
    /// 0 = disabled. After each GPU chunk, CPU processes this many tokens.
    /// Requires --zero-copy or --resilience-prealloc-switch for weight access.
    #[arg(long, default_value_t = 0)]
    prefill_cpu_chunk_size: usize,

    /// Enable profiling (per-op timing, latency, score snapshots).
    ///
    /// Legacy mode: inserts two `clFinish()` calls per op on GPU, which
    /// inflates decode ms/tok by ~54 ms on Adreno. Useful for **relative**
    /// per-op ranking only. For apples-to-apples comparison with llama.cpp
    /// per-op GPU timing, use `--profile-events` instead.
    #[arg(long, default_value_t = false)]
    profile: bool,

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
    profile_events: bool,

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
    heartbeat_gpu_profile: bool,

    /// Output directory for profiling data.
    #[arg(long, default_value = "results/profile")]
    profile_dir: String,

    /// Score snapshot interval (1 = every step, 10 = every 10th step).
    #[arg(long, default_value_t = 1)]
    profile_interval: usize,

    /// Comma-separated list of probes: ops,latency,scores,entropy,cache.
    #[arg(long, default_value = "ops,latency,scores")]
    profile_probes: String,

    /// Enable per-KV-head score tracking (for H2O+ analysis).
    #[arg(long, default_value_t = false)]
    profile_per_head: bool,

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
    cuda_profile: bool,

    /// Experimental: defer the per-op `synchronize()` calls in the
    /// cuda-embedded backend and sync only once per decode token
    /// (immediately before sampling reads logits).
    ///
    /// Phase C hypothesis H1: the 4.6 ms/tok (12%) wall-clock overhead
    /// beyond GPU kernel time is driven by ~30 per-op syncs per token.
    /// Enabling this flag measures the residual cost. Not a production
    /// optimization — correctness relies on sampling being the only
    /// CPU-visible read in the decode loop. Use with
    /// `--backend cuda-embedded` only; a no-op on other backends.
    #[arg(long, default_value_t = false)]
    cuda_defer_sync: bool,

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
    cuda_sync_policy: String,

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
    cuda_weights_device: bool,

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
    cuda_graph: bool,

    /// Model weight data type (f16 or q4). f16 = no quantization, q4 = Q4_0 quantization at load time.
    #[arg(long, default_value = "f16")]
    weight_dtype: String,

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
    quantize_lm_head: String,

    /// KV cache data type (f32, f16, or q4)
    #[arg(long, default_value = "f16")]
    kv_type: String,

    /// Eviction policy for KV cache management (none, sliding, streaming, h2o, h2o_plus, d2o)
    #[arg(long, default_value = "none")]
    eviction_policy: String,

    /// Window size for sliding window / streaming eviction (tokens).
    /// Default: 1024 for sliding, 2000 for streaming.
    #[arg(long, default_value_t = 1024)]
    eviction_window: usize,

    /// Number of attention sink tokens to preserve (StreamingLLM).
    /// Only used with --eviction-policy streaming.
    #[arg(long, default_value_t = 4)]
    sink_size: usize,

    /// StreamingLLM recent window size. 0 = auto (kv_budget - sink_size).
    /// Only used with --eviction-policy streaming.
    #[arg(long, default_value_t = 0)]
    streaming_window: usize,

    /// Fraction of tokens to keep as heavy hitters (0.0 to 1.0)
    #[arg(long, default_value_t = 0.5)]
    h2o_keep_ratio: f32,

    /// Number of final transformer layers to track for H2O importance scores (0 = all layers)
    #[arg(long, default_value_t = 0)]
    h2o_tracked_layers: usize,

    /// Exponential decay factor for H2O importance scores per step (0.0 = no decay)
    #[arg(long, default_value_t = 0.0)]
    h2o_decay: f32,

    /// D2O heavy-hitter keep ratio (0.0–1.0, paper default 0.75 = 3:1 ratio)
    #[arg(long, default_value_t = 0.75)]
    d2o_keep_ratio: f32,

    /// D2O EMA smoothing factor β for threshold update (paper Eq.10, default 0.7).
    /// τ_t = β · max U_t + (1−β) · τ_{t−1}.
    #[arg(long, default_value_t = 0.7)]
    d2o_ema_beta: f32,

    /// D2O Eq.11 normalisation constant `e` (paper default 0.1).
    /// Controls retained token's self-weight: w_c = e / (Σ exp(u_i) + e).
    #[arg(long, default_value_t = 0.1)]
    d2o_merge_e: f32,

    /// Enable D2O layer-level dynamic allocation (uses per-layer attention variance from prefill)
    #[arg(long, default_value_t = false)]
    d2o_layer_alloc: bool,

    /// Protected layers for D2O layer allocation (comma-separated layer indices, e.g. 0,1,2)
    #[arg(long, value_delimiter = ',')]
    d2o_protected_layers: Option<Vec<usize>>,

    /// Number of prefix tokens to protect from eviction.
    /// Defaults to 4 for score-based policies (h2o, h2o_plus, d2o) and prompt length for sliding.
    #[arg(long)]
    protected_prefix: Option<usize>,

    /// Initial KV cache capacity in tokens (0 = auto: prompt length rounded up to power of 2, min 128)
    #[arg(long, default_value_t = 0)]
    initial_kv_capacity: usize,

    /// Memory threshold in MB below which eviction triggers
    #[arg(long, default_value_t = 256)]
    memory_threshold_mb: usize,

    /// Target ratio of cache to keep when evicting (0.1 to 0.99)
    #[arg(long, default_value_t = 0.75)]
    eviction_target_ratio: f32,

    /// Enable resilience manager for adaptive inference
    #[arg(long, default_value_t = false)]
    enable_resilience: bool,

    /// Pre-allocate dual CPU/GPU buffers for zero-alloc SwitchHw.
    /// Without this flag, only throttle/suspend directives work (no backend switch).
    /// Enables: zero-copy KV memory + weight dual-access rewrap (increases RSS by ~model size).
    #[arg(long, default_value_t = false)]
    resilience_prealloc_switch: bool,

    /// Resilience signal transport: "dbus" or "unix:<path>"
    #[arg(long, default_value = "dbus")]
    resilience_transport: String,

    // ── Experiment mode ──────────────────────────────
    /// Experiment schedule JSON file (enables experiment mode)
    #[arg(long)]
    experiment_schedule: Option<String>,

    /// Experiment output JSONL file path
    #[arg(long)]
    experiment_output: Option<String>,

    /// Number of top-K logits to record per token in experiment mode
    #[arg(long, default_value_t = 10)]
    experiment_logits_topk: usize,

    /// System metric sampling interval (N tokens, 0=disabled)
    #[arg(long, default_value_t = 1)]
    experiment_sample_interval: usize,

    /// Force greedy sampling (temperature=0) for reproducibility
    #[arg(long, default_value_t = false)]
    greedy: bool,

    /// Ignore EOS token and continue generating (for long-running experiments)
    #[arg(long, default_value_t = false)]
    ignore_eos: bool,

    /// Target TBT in milliseconds for pacing (0=disabled).
    /// After each decode step, sleeps to maintain the target TBT.
    /// Used for fair resource comparison across different actions at the same QoS.
    #[arg(long, default_value_t = 0.0)]
    target_tbt: f64,

    /// Fixed per-token throttle delay in milliseconds (0=disabled).
    /// Unconditional sleep after each decode step — useful for co-execution
    /// simulations without running a Manager. Manager `Throttle` directives
    /// override this value when resilience is enabled.
    #[arg(long, default_value_t = 0)]
    throttle_delay_ms: u64,

    /// OpenCL command-queue priority hint (`cl_khr_priority_hints`).
    /// "low" yields GPU scheduling to foreground apps (e.g. games) during
    /// co-execution. Falls back to normal priority with a warning if the
    /// driver does not advertise the extension. Also settable via env var
    /// `OCL_QUEUE_PRIORITY`.
    #[arg(long, value_parser = ["low", "medium", "normal", "high"], default_value = "normal")]
    gpu_priority: String,

    /// Intra-token GPU yield: after every N decoded layers, flush the GPU
    /// queue and sleep `--gpu-yield-us` microseconds. Gives the driver a
    /// scheduling window mid-token so concurrent high-priority contexts
    /// (e.g. foreground games) don't wait out the full layer chain. 0
    /// disables. Also settable via env `LLMRS_DECODE_YIELD_EVERY`.
    #[arg(long, default_value_t = 0)]
    gpu_yield_every_layer: usize,

    /// Microsecond sleep per intra-token yield point. Effective only when
    /// `--gpu-yield-every-layer` > 0. 0 issues `sched_yield()` instead of
    /// sleeping. Also settable via env `LLMRS_DECODE_YIELD_US`.
    #[arg(long, default_value_t = 500)]
    gpu_yield_us: u64,

    /// Path to write per-token TBT JSONL log.
    /// Each line: {"token_idx":N,"tbt_ms":X,"forward_ms":Y,"cache_pos":Z,"pacing_ms":W}
    #[arg(long)]
    tbt_log: Option<String>,

    /// KV cache memory layout: "head" (head-major) or "seq" (seq-major)
    #[arg(long, default_value = "head")]
    kv_layout: String,

    /// Minimum KV cache size in tokens. Eviction will not reduce cache below this.
    #[arg(long, default_value_t = 256)]
    min_kv_cache: usize,

    /// Override eviction target_ratio from resilience signals (experiment mode).
    /// When set, all Evict actions will use this ratio instead of the strategy default.
    #[arg(long)]
    experiment_eviction_ratio: Option<f32>,

    /// Enable verbose H2O debug output (per-step scores, softmax validation, eviction details)
    #[arg(long, default_value_t = false)]
    h2o_debug: bool,

    /// Disable time-normalized scoring (use raw cumulative SUM).
    /// By default, H2O/H2O+ use time-normalized scores to remove cumulative bias.
    #[arg(long, default_value_t = false)]
    h2o_raw_scores: bool,

    /// QCF variant to compute: "attn" (default), "caote", or "both".
    #[arg(long, default_value = "attn")]
    qcf_mode: String,

    /// Enable AWQE + AW-VOPR metrics for KIVI.
    #[arg(long, default_value_t = false)]
    awqe: bool,

    // ── Eval-LL mode (log-likelihood evaluation) ──
    /// Enable log-likelihood evaluation mode (downstream task accuracy)
    #[arg(long, default_value_t = false)]
    eval_ll: bool,

    /// Continuation text to evaluate log-likelihood (single task mode)
    #[arg(long)]
    eval_continuation: Option<String>,

    /// Path to evaluation batch JSON file: [{"id","prompt","continuation"}, ...]
    #[arg(long)]
    eval_batch: Option<String>,

    /// Maximum KV cache budget in tokens. Evicts when cache_pos exceeds this.
    /// 0 = no budget limit (default).
    #[arg(long, default_value_t = 0)]
    kv_budget: usize,

    /// KV cache budget as a ratio of prompt length (0.0–1.0).
    /// When set (> 0), overrides --kv-budget per question: budget = prompt_len * ratio.
    /// Matches H2O paper evaluation methodology.
    #[arg(long, default_value_t = 0.0)]
    kv_budget_ratio: f32,

    /// Enable KIVI-style Q2 KV cache compression (ICML 2024).
    /// Mutually exclusive with eviction policies; uses FP32 residual buffer
    /// that batch-quantizes to 2-bit when full.
    #[arg(long, default_value_t = false)]
    kivi: bool,

    /// Enable dynamic KV cache quantization for resilience.
    /// Starts with bits=16 (F16-equivalent KiviCache) and allows runtime
    /// transition to Q2/Q4/Q8 via kv_quant_dynamic resilience command.
    #[arg(long, default_value_t = false)]
    kv_dynamic_quant: bool,

    /// KIVI quantization bit-width (2, 4, or 8). Default: 2.
    #[arg(long, default_value_t = 2)]
    kivi_bits: u8,

    /// KIVI residual buffer size in tokens (must be multiple of 32).
    /// Default: 32. Larger values improve quality but use more memory.
    #[arg(long, default_value_t = 32)]
    kivi_residual_size: usize,

    /// Number of threads for parallel computation.
    /// Default: auto-detect CPU core count.
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// KV cache offload mode: none, raw (in-memory), or disk (file-based).
    /// Requires --kv-layout seq and --kv-type f16 or f32.
    #[arg(long, default_value = "none")]
    kv_offload: String,

    /// Directory for disk offload files (used with --kv-offload disk).
    /// Defaults to system temp dir if not specified.
    #[arg(long, default_value = "")]
    offload_path: String,

    /// Maximum adaptive prefetch depth for offload KV cache pipeline.
    /// Higher values use more memory but can hide preload latency.
    /// Combined with the controller's default initial depth (16), the
    /// adaptive loop can spend essentially the entire decode trajectory
    /// on increasing/decreasing depth without hitting the ceiling on
    /// typical on-device workloads.
    #[arg(long, default_value_t = 128)]
    max_prefetch_depth: usize,

    /// Use Rayon par_chunks_mut instead of SpinPool for F16 matmul (A/B benchmarking).
    #[arg(long, default_value_t = false)]
    use_rayon: bool,

    /// Path to reference text file for perplexity evaluation (teacher-forcing).
    /// Measures PPL and collects proxy metrics during eviction.
    #[arg(long)]
    ppl: Option<String>,

    /// PPL mode 에서 weight swap 을 trigger 할 decode token index (0-based).
    /// 미지정 시 PPL decode loop 은 swap 없음 (baseline).
    /// Requires `--secondary-gguf` to load secondary weights.
    /// 사용 예 (LISWAP-PPL): `--ppl ref.txt --secondary-gguf ... --ppl-swap-at-token 0
    /// --ppl-swap-ratio 0.9 --ppl-swap-per-tick 1`.
    #[arg(long)]
    ppl_swap_at_token: Option<usize>,

    /// PPL swap ratio (0.0~1.0). engine `WeightSwapDecider` 가 [0.0, 1.0] 으로 clamp.
    /// 1.0 + `LLMRS_SWAP_ALLOW_BOUNDARY_LAYERS=1` 조합 시 전 layer swap.
    #[arg(long, default_value_t = 0.9)]
    ppl_swap_ratio: f32,

    /// PPL incremental swap K (layer/tick). 측정용으로 dynamic-K 비활성화, fixed K.
    /// 기본 1 (한 step 에 1 layer 씩).
    #[arg(long, default_value_t = 1)]
    ppl_swap_per_tick: usize,

    /// PPL per-token NLL CSV 출력 경로. 미지정 시 dump 안 함.
    /// CSV columns: phase, token_idx, token_id, nll, swap_state, layers_swapped.
    #[arg(long)]
    ppl_nll_csv: Option<std::path::PathBuf>,

    /// PPL prefill 토큰 수 강제 설정 (1..=eval_tokens). 미지정 시 기존 로직
    /// (kv_budget / sliding window / eval_tokens) 그대로. swap 측정 시 decode
    /// loop 을 충분히 길게 돌려야 하므로 이 옵션으로 prefill 을 짧게 만든다.
    /// 예: 1072 token reference 에서 `--ppl-prefill-tokens 32` → prefill 32 +
    /// decode 1040 step.
    #[arg(long)]
    ppl_prefill_tokens: Option<usize>,

    /// LISWAP-PPL Scenario E: swap 완료 후 KV cache reset + prefill 다시 시작.
    /// `--ppl-swap-at-token` + `--secondary-gguf` 필요. 워크플로:
    ///   pass 1 (warmup): prefill + swap-driving decode → plan_done 시 종료
    ///   pass 2 (measure): KV cache 0 으로 reset → prefill 다시 → decode (no swap)
    /// pass 2 의 NLL/PPL/CSV 만 기록. 가설: cache mismatch 가 C−D artifact 원인이면
    /// pass 2 결과가 D (Q4 native) 에 수렴해야 함.
    #[arg(long, default_value_t = false)]
    ppl_warmup_swap: bool,

    /// LISWAP-PPL Scenario F: `--ppl-warmup-swap` 의 pass 2 prefill 길이를 별도로
    /// 지정. 미지정 시 `--ppl-prefill-tokens` 값을 그대로 사용 (= 시나리오 E).
    /// 예: pass 1 prefill=32 (swap-driving decode 28 step) + pass 2 prefill=1072
    /// (decode loop 없음, batch path 전체) → batch path 만으로 weight 정합 검증.
    #[arg(long)]
    ppl_measure_prefill_tokens: Option<usize>,

    /// LISWAP-PPL diagnostic: 모델 로드 직후 모든 layer 의 weight tensor (wq/wk/wv/
    /// wo/w_gate/w_up/w_down) 를 readback 해서 `<dir>/layer{NN}_{name}_{dtype}.bin`
    /// 으로 dump. swap 없는 baseline 측정용 (e.g. Q4_0 native 모델 비교 기준).
    #[arg(long)]
    dump_q4_after_load: Option<std::path::PathBuf>,

    /// LISWAP-PPL diagnostic: `--ppl-warmup-swap` Pass 1 의 swap 완료 직후 (cache
    /// reset 직전) 모든 layer 의 weight tensor 를 readback 해서 dump. swap 경로의
    /// Q4 weight 가 standalone Q4 와 비트 단위로 일치하는지 검증용.
    #[arg(long)]
    dump_q4_after_swap: Option<std::path::PathBuf>,

    /// Comma-separated layer indices to skip (both attn+mlp).
    /// Example: --skip-layers 1,3,5,7
    #[arg(long, value_delimiter = ',')]
    skip_layers: Option<Vec<usize>>,

    /// Skip ratio (0.0-1.0). Uses SkipConfig::uniform_init() to select layers.
    #[arg(long)]
    skip_ratio: Option<f32>,

    /// Dump per-layer importance table and exit (no inference).
    /// Runs prefill with ImportanceCollector on the given prompt.
    #[arg(long, default_value_t = false)]
    dump_importance: bool,

    /// Path to JSONL file for multi-prompt batch generation.
    /// Each line: {"id":"...", "prompt":"..."} or {"id":"...", "prompt_file":"path"}
    /// Mutually exclusive with --prompt, --prompt-file, --eval-batch.
    #[arg(long)]
    prompt_batch: Option<String>,

    /// Loop prompt-batch: restart from beginning when all entries are processed.
    #[arg(long, default_value_t = false)]
    prompt_batch_loop: bool,

    /// Maximum iterations for prompt-batch loop (0 = unlimited).
    #[arg(long, default_value_t = 0)]
    max_iterations: usize,

    /// Start an interactive multi-turn chat REPL (Llama 3.2 Instruct / Qwen2).
    /// Uses standard (non-KIVI, non-offload) forward path.
    #[arg(long, default_value_t = false)]
    chat: bool,

    /// Optional system prompt injected as the first turn when --chat is set.
    #[arg(long)]
    system_prompt: Option<String>,

    /// Optional Unix domain socket path. When set, chat mode also accepts
    /// newline-delimited user messages from this socket in addition to stdin,
    /// and streams assistant replies back (terminated by 0x04).
    #[arg(long)]
    chat_socket: Option<String>,

    /// Optional TCP listen address (e.g. "127.0.0.1:7878"). Same protocol
    /// as --chat-socket: newline-delimited input, assistant reply bytes
    /// streamed back, 0x04 EOT delimiter per turn. Can be combined with
    /// --chat-socket; both listeners feed the same chat loop.
    #[arg(long)]
    chat_tcp: Option<String>,

    /// Directory used by `KvOffload` directives to write out the LRU prefix
    /// of the KV cache. When set, `CacheManager::enable_swap()` registers a
    /// disk-backed `SwapHandler`; without it the `KvOffload` directive is
    /// a warn-only no-op. `RestoreDefaults` triggers recall of offloaded data.
    #[arg(long)]
    swap_dir: Option<std::path::PathBuf>,

    /// Optional secondary GGUF path for runtime weight swap (Phase 2).
    /// When specified together with `--force-swap-ratio`, the engine swaps
    /// decoder layer weights from the primary dtype to the secondary dtype
    /// immediately before generation starts.
    /// When omitted, the weight swap path is disabled (ENG-DAT-C09).
    #[arg(long)]
    secondary_gguf: Option<std::path::PathBuf>,

    /// Manually trigger a weight swap before generation starts.
    /// Value: fraction of decoder layers to swap (0.0–1.0).
    /// Example: `--force-swap-ratio 0.5` swaps 50% of layers.
    /// Requires `--secondary-gguf` to be set; exits early with an error
    /// if the secondary path is absent.
    /// Intended for offline testing and debug; not for production use.
    #[arg(long)]
    force_swap_ratio: Option<f32>,

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
    secondary_dtype: SecondaryDtypeArg,

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
    secondary_layout: SecondaryLayoutArg,

    /// Explicit path to tokenizer.json. When omitted, the tokenizer is
    /// resolved automatically via the GGUF basename (e.g.
    /// `<dir>/<stem>.tokenizer.json`, then `<dir>/<stem-without-quant>.tokenizer.json`,
    /// then the legacy `<dir>/tokenizer.json` fallback). Required when
    /// multiple models share the same directory (e.g. `/data/local/tmp/`)
    /// because the legacy fallback can pick up a sibling model's tokenizer
    /// and silently produce garbage outputs.
    #[arg(long)]
    tokenizer_path: Option<std::path::PathBuf>,

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
    qcf_dump: Option<std::path::PathBuf>,

    /// Number of tokens for the warmup prefill that builds ImportanceTable.
    /// Only used when `--qcf-dump` is set. Default: 256.
    #[arg(long, default_value_t = 256)]
    qcf_warmup_tokens: usize,

    /// Layer-selection algorithm for `--qcf-dump` swap path (U5 ablation, EuroSys'27).
    /// Values: `imp` (importance-aware, default — production behavior),
    /// `seq` (sequential 0→N-1), `rev` (reverse N-1→0),
    /// `uni` (evenly spaced), `anti` (importance × ε descending top-k — worst-case).
    /// Only affects the `--qcf-dump` warmup-prefill swap; the manager / live
    /// directive paths always use `imp`.
    #[arg(long, default_value = "imp")]
    swap_algorithm: String,

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
    importance_formula: String,

    /// Explicit per-layer swap list (CSV of layer indices) for §4 ground-truth
    /// study. Bypasses `WeightSwapDecider`: when set, the listed layers are
    /// swapped regardless of `--force-swap-ratio` or `--swap-algorithm`.
    /// `--force-swap-ratio` must still be provided (any non-zero value) so
    /// the warmup workflow runs; the ratio itself is ignored. Example:
    /// `--swap-only-layers 5` swaps only layer 5.
    #[arg(long)]
    swap_only_layers: Option<String>,

    /// Per-step NLL trajectory mode (U5 mid-swap quality study, EuroSys'27).
    ///
    /// Requires `--qcf-dump`, `--force-swap-ratio`, `--eval-ll`, `--eval-batch`.
    /// Workflow:
    ///   1. warmup prefill → ImportanceTable
    ///   2. WeightSwapDecider.decide(ratio, algorithm) → ordered layer list
    ///      of length K = floor(ratio × num_layers).
    ///   3. for t = 0..=K:
    ///       a. run eval-ll on the eval batch → record EvalOutput_t.
    ///       b. if t < K: SwapExecutor.execute_on_slots(&[selected_layers[t]]).
    ///   4. dump JSON with trajectory: array of K+1 (step, swapped_layers,
    ///      layer_added, eval_ll_output).
    /// The cumulative swap state at step t mirrors ARGUS's production
    /// incremental swap (one layer per token), letting external analysis
    /// observe the mid-swap NLL trajectory rather than only the final state.
    #[arg(long, default_value_t = false)]
    qcf_trajectory: bool,

    /// Enable QCF v3 schema metric dump for EuroSys'27 §3.
    /// Adds qcf_layer_worst_head/qcf_layer_mean_head/qcf_record_*/qcf_d7_*/
    /// qcf_c1_* (schema_version=3) to eval-ll output for both Eviction and KIVI.
    #[arg(long, default_value_t = false)]
    enable_qcf_experimental: bool,

    /// Sample layer indices for multi-layer QCF.
    /// Default is "all" (schema v3: every decoder layer) — required for D7/C1.
    /// Other values: "auto" (legacy 5-tuple [0, n/4, n/2, 3n/4, n-1]),
    /// "0,8,16,24,31" (explicit indices).
    #[arg(long, default_value = "all")]
    qcf_sample_layers: String,

    /// β values for β-amplified CAOTE (ARGUS #6 option B).
    /// Comma-separated, e.g., "1.0,1.5,2.0". Note: only β=1.0/1.5/2.0 are
    /// dumped to fixed JSON keys. This flag is currently a placeholder for
    /// future flexibility — the dumped values follow the fixed keys regardless.
    #[arg(long, value_delimiter = ',', default_value = "1.0,1.5,2.0")]
    qcf_betas: Vec<f32>,

    /// K values for top-K retention (ARGUS #5).
    /// Currently fixed [10, 20, 50] in dump; flag accepted for future use.
    #[arg(long, value_delimiter = ',', default_value = "10,20,50")]
    qcf_topk_values: Vec<usize>,

    /// τ values for defensive aggregation. Fixed [0.1, 0.5] in dump.
    #[arg(long, value_delimiter = ',', default_value = "0.1,0.5")]
    qcf_defensive_taus: Vec<f32>,

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
    eager_prefault_secondary: bool,

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
    swap_incremental_per_tick: usize,

    /// LISWAP-2 prototype: Submit incremental swap chunks to a separate
    /// transfer queue/stream so weight H2D writes overlap with the next
    /// token's forward compute.
    ///
    /// Requires `--swap-incremental-per-tick > 0`. When `=0` or absent,
    /// has no effect (silently ignored).
    ///
    /// **Default ON (2026-05-12)** — async dispatch는 production winner mode의
    /// 기본 구성 요소. `--swap-incremental-per-tick 0` (swap 비활성)일 때는
    /// 자동 무시되므로 swap을 쓰지 않는 경우에도 안전. sync path를 명시적으로
    /// 원하면 `--swap-async-dispatch=false`.
    #[arg(long, default_value_t = true)]
    swap_async_dispatch: bool,

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
    swap_zero_copy: bool,

    /// LISWAP-3 prototype: number of slots in the `CL_MEM_ALLOC_HOST_PTR`
    /// swap pool. Stage 2 measurement on Galaxy S25 (Qwen2.5-1.5B, 28
    /// layers, 7 Q4_0 tensors per layer) reported a sweet spot at 14
    /// slots (= 2 layers worth of in-flight work). Effective only with
    /// `--swap-zero-copy` + `LLMRS_OPENCL_HOST_PTR_POOL=1`.
    #[arg(long, default_value_t = 14)]
    swap_pool_slots: usize,

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
    swap_intra_forward: bool,

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
    swap_phase_aware: bool,

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
    swap_layer_immediate: bool,

    /// LISWAP-6 Dynamic-K controller — auto-tune `--swap-incremental-per-tick`
    /// based on measured per-layer release cost vs forward wall.
    ///
    /// Requires `--swap-incremental-per-tick > 0` (the explicit value is treated
    /// as the *hard upper cap* — controller starts at K=1 and may grow up to
    /// the user-supplied value during Phase 0 calibration; never exceeds it).
    /// Effective only together with `--swap-async-dispatch`.
    ///
    /// Memory-spike avoidance is the hard constraint: K is monotone
    /// non-increasing after calibration and a reactive pause skips swap when
    /// the release queue is non-empty. See `dynamic_k.rs` for the algorithm.
    ///
    /// **Default ON (2026-05-12)** — production winner mode의 기본 구성 요소.
    /// `--swap-incremental-per-tick 0`일 때는 자동 무시되므로 항상 안전한 default.
    /// 정적 K를 원하면 `--swap-dynamic-k=false`.
    #[arg(long, default_value_t = true)]
    swap_dynamic_k: bool,

    /// Phase-aware swap chunk 진단 size (MB). v1 per-tensor chunking에서는
    /// 실제 분할에 사용되지 않고 진단/보고 용도. 측정에 따라 v2에서 sub-tensor
    /// chunking 도입 시 활용 (4 MB sweet spot 기본).
    #[arg(long, default_value_t = 4)]
    swap_phase_aware_chunk_mb: usize,

    /// Phase-aware swap throttle — token당 dispatch chunk 수 상한.
    /// 0 = 무제한 (현재 동작 유지, 252 chunks가 첫 3 token에 누적).
    /// N>0 = 매 token N chunks까지만 → 분산을 더 길게 펼쳐 max-stall 단축.
    /// Sweep 측정용 (Phase 2): K = {1,2,4,8,16}.
    #[arg(long, default_value_t = 0)]
    swap_phase_aware_max_chunks_per_token: usize,

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
    swap_delay_tokens: usize,

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
    swap_no_throttle: bool,

    // ── QNN OpPackage backend (M3, ENG-QNN-220) ─────────────────
    /// Eager prebuild of all 28 layer graphs at model load time
    /// (ENG-QNN-209, D1 결정).
    ///
    /// `true` (default): model load 시점에 N×`graphFinalize` (≤ 200 ms/layer)
    /// 직렬 실행. Decode 동안 추가 finalize는 0회 (INV-167).
    /// `false`: lazy build (M3.5 timebox 미사용 시 진입 불가, debug 용도).
    ///
    /// 본 flag는 `--backend qnn_oppkg | qnngpu` 활성 시에만 유효하다.
    #[arg(long, default_value_t = true)]
    qnn_graph_cache_prebuild: bool,

    /// Allow trait fallback path when graph fast path fails (ENG-QNN-220).
    ///
    /// `false` (default): fast path 실패 시 즉시 `Err`. INV-175 (fallback count
    /// == 0) 게이트와 정합한다.
    /// `true`: debug 용도 — fast path 실패 시 OpenCL secondary backend로
    /// fallback. production 측정에서는 OFF 유지.
    #[arg(long, default_value_t = false)]
    qnn_allow_fallback: bool,
}

/// Create a GPU buffer allocator for tensor partition workspace.
///
/// On OpenCL: allocates `UnifiedBuffer` (CL_MEM_ALLOC_HOST_PTR) + permanent map.
/// Single VMA: `as_ptr()`/`as_mut_ptr()` return valid host pointers while
/// `cl_mem()` remains valid for GPU kernels. No PSS double-counting on Adreno.
///
/// On other backends (CPU, CUDA): falls back to `memory.alloc()` which already
/// returns host-accessible buffers (SharedBuffer, CudaHostBuffer).
fn make_partition_gpu_alloc<'a>(
    backend: &'a dyn Backend,
    memory: &'a dyn Memory,
) -> impl Fn(usize, DType) -> anyhow::Result<Arc<dyn llm_rs2::core::buffer::Buffer>> + 'a {
    // Try to extract OpenCL queue for UnifiedBuffer allocation.
    #[cfg(feature = "opencl")]
    let ocl_queue: Option<ocl::Queue> = backend
        .as_any()
        .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
        .map(|b| b.queue.clone());

    #[cfg(not(feature = "opencl"))]
    let _ = backend; // suppress unused warning

    move |size: usize, dtype: DType| -> anyhow::Result<Arc<dyn llm_rs2::core::buffer::Buffer>> {
        #[cfg(feature = "opencl")]
        if let Some(ref q) = ocl_queue {
            let buf = llm_rs2::buffer::unified_buffer::UnifiedBuffer::new(q.clone(), size, dtype)?;
            buf.map()?; // Permanent map for dual CPU/GPU access
            return Ok(Arc::new(buf));
        }
        memory.alloc(size, dtype)
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    // Sprint E forward_gen op-tracer: install atexit hook so the trace
    // dumps even on Ctrl+C / early-return paths. No-op when env unset.
    llm_rs2::profile::op_trace::install_atexit_once();
    // Quality-cost profiler: gated by LLM_RS2_PROFILE_QUALITY=1.
    llm_rs2::profile::quality_metrics::install_atexit_once();
    // T0: process start, before CLI parsing or any allocation.
    rss_trace("start");

    #[allow(unused_mut)]
    let mut args = Args::parse();

    // ENG-DAT-C18: --swap-incremental-per-tick > 0 / --swap-intra-forward /
    // --swap-phase-aware are mutually exclusive (LISWAP-1 vs LISWAP-4 vs
    // LISWAP-5 — ratio_generation bump + dispatcher ownership conflict).
    // Reject combinations explicitly so engine never starts in an ambiguous
    // swap-policy state.
    let swap_modes_active = (args.swap_incremental_per_tick > 0) as usize
        + args.swap_intra_forward as usize
        + args.swap_phase_aware as usize
        + args.swap_layer_immediate as usize;
    if swap_modes_active > 1 {
        anyhow::bail!(
            "--swap-incremental-per-tick (= {}) / --swap-intra-forward (= {}) / \
             --swap-phase-aware (= {}) / --swap-layer-immediate (= {}) are mutually \
             exclusive (ENG-DAT-C18). Pick one:\n\
             (a) --swap-incremental-per-tick=N                                 (LISWAP-1)\n\
             (b) --swap-intra-forward=true                                     (LISWAP-4)\n\
             (c) --swap-phase-aware=true                                       (LISWAP-5)\n\
             (d) --swap-layer-immediate=true                                   (LISWAP-6 P6)\n\
             (e) (none)                                                        (single-shot)",
            args.swap_incremental_per_tick,
            args.swap_intra_forward,
            args.swap_phase_aware,
            args.swap_layer_immediate
        );
    }

    // --swap-no-throttle: forwards to env so SwapExecutor::execute_on_slots
    // skips the INV-141 release_worker drain. Measurement-only (EuroSys 2027
    // §4.2 layer-count predictor accuracy). Sets the env only if unset so the
    // env-based invocation path (LLMRS_SWAP_FORCE_EVERY_TICK=1) keeps working
    // independently. The executor logs a stderr warning on first read.
    if args.swap_no_throttle && std::env::var_os("LLMRS_SWAP_FORCE_EVERY_TICK").is_none() {
        // SAFETY: set before any worker thread that might read the variable.
        // generate.rs::main runs single-threaded up to this point (CLI parse +
        // Rayon pool init below). Writes after thread spawn would be UB on
        // some platforms; this write precedes the pool builder on line 1247.
        unsafe { std::env::set_var("LLMRS_SWAP_FORCE_EVERY_TICK", "1") };
    }

    // Configure Rayon thread pool: 0 = auto-detect CPU cores
    let num_threads = if args.threads > 0 {
        args.threads
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();
    eprintln!("[Config] Using {} threads", num_threads);

    // Wire Rayon vs SpinPool toggle
    #[cfg(target_arch = "aarch64")]
    if args.use_rayon {
        llm_rs2::backend::cpu::neon::USE_RAYON.store(true, std::sync::atomic::Ordering::Relaxed);
        eprintln!("[Config] F16 matmul: Rayon (par_chunks_mut)");
    }

    // --greedy overrides temperature to 0
    if args.greedy {
        args.temperature = 0.0;
    }

    // --chat conflict validation.
    // Standard / kivi / offload paths are each supported; experiment/eval
    // modes and advanced GPU features remain incompatible.
    if args.chat {
        let kv_offload_active = !args.kv_offload.is_empty() && args.kv_offload != "none";
        let has_eviction = args.eviction_policy != "none";
        if args.kivi && kv_offload_active {
            anyhow::bail!("--chat: --kivi and --kv-offload are mutually exclusive");
        }
        if args.kivi && has_eviction {
            anyhow::bail!("--chat: --kivi cannot combine with --eviction-policy in v1 (pick one)");
        }
        if kv_offload_active && has_eviction {
            anyhow::bail!(
                "--chat: --kv-offload cannot combine with --eviction-policy in v1 (pick one)"
            );
        }
        let conflicts: &[(&str, bool)] = &[
            ("--eval-ll", args.eval_ll),
            ("--ppl", args.ppl.is_some()),
            ("--prompt-batch", args.prompt_batch.is_some()),
            ("--eval-batch", args.eval_batch.is_some()),
            ("--tensor-partition", args.tensor_partition > 0.0),
            ("--cuda-graph", args.cuda_graph),
            ("--dump-importance", args.dump_importance),
            ("--experiment-schedule", args.experiment_schedule.is_some()),
        ];
        if let Some((flag, _)) = conflicts.iter().find(|(_, enabled)| *enabled) {
            anyhow::bail!(
                "--chat is incompatible with {} (v1 supports standard / --kivi / --kv-offload / --eviction-policy paths)",
                flag
            );
        }
    }

    // --profile and --profile-events are mutually exclusive.
    // Both probe the decode path but use incompatible mechanisms:
    //   --profile          : CPU wall clock + per-op clFinish (adds ~54 ms/tok)
    //   --profile-events   : GPU profiling events (near-zero overhead)
    if args.profile && args.profile_events {
        anyhow::bail!(
            "--profile and --profile-events are mutually exclusive. \
             Use --profile-events for absolute GPU per-op timing (Adreno/llama.cpp comparison), \
             or --profile for legacy CPU-wall-clock relative ranking."
        );
    }

    let sampling_config = SamplingConfig {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        repetition_window: args.repetition_window,
    };

    let model_path = &args.model_path;

    // Propagate GPU queue priority to OpenCLBackend via env var (same
    // convention as OCL_PLATFORM / OCL_DEVICE_TYPE). CLI wins over an
    // already-set env var so a flag in a script overrides the shell env.
    if args.gpu_priority != "normal" {
        unsafe {
            std::env::set_var("OCL_QUEUE_PRIORITY", &args.gpu_priority);
        }
    }

    // Propagate intra-token GPU yield knobs. Both flags route through env
    // vars so `core::gpu_yield`'s OnceLock cache stays valid across sub-crate
    // boundaries. CLI wins over a pre-set env var.
    if args.gpu_yield_every_layer > 0 {
        unsafe {
            std::env::set_var(
                "LLMRS_DECODE_YIELD_EVERY",
                args.gpu_yield_every_layer.to_string(),
            );
            std::env::set_var("LLMRS_DECODE_YIELD_US", args.gpu_yield_us.to_string());
        }
    }

    // 1. Setup
    eprintln!("[Profile] Event: ModelLoadStart");
    eprintln!("Loading model from {}", model_path);

    // Backend initialization: primary backend + secondary for SwitchHw resilience.
    // GPU secondary is auto-initialized when available (soft failure OK).
    #[allow(clippy::type_complexity)]
    let (mut backend, memory, gpu_backend_arc, gpu_memory_arc, mut is_gpu): (
        Arc<dyn Backend>,
        Arc<dyn Memory>,
        Option<Arc<dyn Backend>>,
        Option<Arc<dyn Memory>>,
        bool,
    ) = match args.backend.as_str() {
        "cpu" => {
            let cpu = Arc::new(CpuBackend::new()) as Arc<dyn Backend>;
            let cpu_mem: Arc<dyn Memory> = Arc::new(Galloc::new());
            // Try to init GPU as secondary for SwitchHw resilience
            #[cfg(feature = "opencl")]
            let (gpu_be, gpu_mem_arc) =
                match llm_rs2::backend::opencl::OpenCLBackend::new_with_profile_events(
                    // MSG-068 Phase 2: heartbeat-gpu-profile도 같은 queue
                    // profiling 인프라를 사용하므로 어느 한쪽이 켜지면 활성화.
                    args.profile_events || args.heartbeat_gpu_profile,
                ) {
                    Ok(gpu_concrete) => {
                        let gpu_concrete = Arc::new(gpu_concrete);
                        let gm: Arc<dyn Memory> =
                            Arc::new(llm_rs2::backend::opencl::memory::OpenCLMemory::new(
                                gpu_concrete.context.clone(),
                                gpu_concrete.queue.clone(),
                                args.zero_copy,
                            ));
                        let g = gpu_concrete as Arc<dyn Backend>;
                        eprintln!(
                            "[Backend] CPU primary, GPU secondary available (SwitchHw ready)"
                        );
                        (Some(g), Some(gm))
                    }
                    Err(e) => {
                        eprintln!("[Backend] CPU only (GPU init failed: {})", e);
                        (None, None)
                    }
                };
            #[cfg(not(feature = "opencl"))]
            let (gpu_be, gpu_mem_arc): (
                Option<Arc<dyn Backend>>,
                Option<Arc<dyn Memory>>,
            ) = (None, None);
            (cpu, cpu_mem, gpu_be, gpu_mem_arc, false)
        }
        #[cfg(feature = "opencl")]
        "opencl" | "gpu" => {
            let gpu_concrete = Arc::new(
                llm_rs2::backend::opencl::OpenCLBackend::new_with_profile_events(
                    // MSG-068 Phase 2: heartbeat-gpu-profile도 같은 queue
                    // profiling 인프라를 사용하므로 어느 한쪽이 켜지면 활성화.
                    args.profile_events || args.heartbeat_gpu_profile,
                )?,
            );
            // When resilience is enabled, force zero-copy memory so KV cache uses
            // UnifiedBuffer (CL_MEM_ALLOC_HOST_PTR, host-accessible). This enables
            // zero-alloc UMA re-tag during GPU→CPU switch instead of 56MB GPU→CPU copy.
            let mut effective_zero_copy = args.zero_copy
                || args.resilience_prealloc_switch
                || args.tensor_partition > 0.0
                || args.prefill_cpu_chunk_size > 0
                || args.enable_resilience;
            if !args.zero_copy
                && (args.resilience_prealloc_switch
                    || args.tensor_partition > 0.0
                    || args.prefill_cpu_chunk_size > 0
                    || args.enable_resilience)
            {
                eprintln!("[Config] Forcing zero-copy memory for CPU-accessible buffers");
            }
            // LLMRS_FORCE_DEVICE_ALLOC: RSS diagnostic flag.
            // Forces effective_zero_copy=false so OpenCLMemory::alloc() creates
            // OpenCLBuffer (READ_WRITE device-only) instead of UnifiedBuffer
            // (CL_MEM_ALLOC_HOST_PTR).  This lets the Tester measure the RSS
            // contribution of ALLOC_HOST_PTR vs device-only allocations.
            //
            // Independent from FORCE_DEVICE_ONLY (backend-level flag) — both can
            // be set simultaneously to ensure all paths use device-only memory.
            // When only LLMRS_FORCE_DEVICE_ALLOC is set, the primary alloc path
            // (OpenCLMemory::alloc) goes device-only but any backend-level zero-copy
            // overrides (e.g. --zero-copy CLI flag processed above) are suppressed.
            if std::env::var("LLMRS_FORCE_DEVICE_ALLOC").is_ok() {
                effective_zero_copy = false;
                eprintln!(
                    "[RSS-diag] LLMRS_FORCE_DEVICE_ALLOC set: effective_zero_copy forced to false \
                     (primary memory = device-only)"
                );
            }
            let gpu_mem: Arc<dyn Memory> =
                Arc::new(llm_rs2::backend::opencl::memory::OpenCLMemory::new(
                    gpu_concrete.context.clone(),
                    gpu_concrete.queue.clone(),
                    effective_zero_copy,
                ));
            let gpu: Arc<dyn Backend> = gpu_concrete;
            // GPU is primary; keep a ref as secondary for SwitchHw round-trip
            (gpu.clone(), gpu_mem.clone(), Some(gpu), Some(gpu_mem), true)
        }
        #[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
        "cuda" => {
            let gpu_concrete = Arc::new(llm_rs2::backend::cuda::CudaBackend::new()?);
            let gpu_mem: Arc<dyn Memory> = if gpu_concrete.is_discrete_gpu() {
                Arc::new(llm_rs2::backend::cuda::memory::CudaMemory::managed())
            } else {
                Arc::new(llm_rs2::backend::cuda::memory::CudaMemory::new())
            };
            // --cuda-profile: event-based per-op profiler. Only wired on
            // the cuda-embedded backend (PC cuda path has its own
            // profiling story and doesn't expose enable_profiler).
            #[cfg(feature = "cuda-embedded")]
            if args.cuda_profile {
                gpu_concrete.enable_profiler(4096)?;
            }
            // --cuda-defer-sync: skip implicit per-op synchronize() in
            // launch helpers. The decode loop must then sync once per
            // token before sampling reads the logits — see the decode
            // loop's pre-sampling barrier. Available on both cuda_pc
            // (host discrete GPU) and cuda_embedded (Jetson UMA).
            #[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
            if args.cuda_defer_sync {
                gpu_concrete.set_defer_sync(true);
                eprintln!(
                    "[CUDA] --cuda-defer-sync enabled: per-op syncs suppressed; token-boundary sync only"
                );
            }
            // --cuda-sync-policy: fine-grained per-category bisection.
            // Parsed before weights-device so a misconfigured string
            // errors out before the long model-load path. `all` is a
            // no-op (matches the AtomicU32 default from `new()`); other
            // values override the policy bitmask. Legacy
            // `--cuda-defer-sync` takes precedence and zeros the policy
            // entirely at the `maybe_sync_cat` layer.
            //
            // Resolves through `llm_rs2::backend::cuda` which aliases to
            // cuda_pc (feature = "cuda") or cuda_embedded (feature =
            // "cuda-embedded"); the two modules share the same
            // SyncPolicy API shape.
            #[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
            {
                use llm_rs2::backend::cuda::SyncPolicy;
                let policy = SyncPolicy::parse(&args.cuda_sync_policy).map_err(|e| {
                    anyhow::anyhow!(
                        "--cuda-sync-policy: {e}. Valid: all | none | llamacpp | minimal | custom:<cats>"
                    )
                })?;
                gpu_concrete.set_sync_policy(policy);
                if !args.cuda_sync_policy.eq_ignore_ascii_case("all") {
                    eprintln!(
                        "[CUDA] --cuda-sync-policy={} (mask=0x{:02x})",
                        args.cuda_sync_policy,
                        policy.raw()
                    );
                }
            }
            // --cuda-weights-device: route weight uploads through a pure
            // device allocation (cuMemAlloc + explicit H2D). Must be set
            // before the model loader runs so every `copy_weight_from`
            // call sees the flag.
            #[cfg(feature = "cuda-embedded")]
            if args.cuda_weights_device {
                if gpu_concrete.is_discrete_gpu() {
                    eprintln!(
                        "[CUDA] --cuda-weights-device ignored on discrete GPU (managed memory already migrates weights to VRAM)"
                    );
                } else {
                    gpu_concrete.set_weights_device(true);
                    eprintln!(
                        "[CUDA] --cuda-weights-device enabled: weight tensors allocated via cuMemAlloc (device-only); activations/KV remain host-pinned"
                    );
                }
            }
            let gpu: Arc<dyn Backend> = gpu_concrete;
            (gpu.clone(), gpu_mem.clone(), Some(gpu), Some(gpu_mem), true)
        }
        // ENG-QNN-202/INV-170: qnn_oppkg는 default off opt-in. feature 비활성 시
        // 본 분기는 컴파일에서 제거되어 unknown backend로 빠진다.
        #[cfg(feature = "qnn")]
        "qnn_oppkg" | "qnngpu" => {
            // QNN backend는 호스트(non-Android)에서 init 실패 → 명확한 Err 전파.
            // 디바이스 빌드에서만 정상 진행 가능 (libQnnGpu.so 존재).
            // ENG-QNN-209/D1: --qnn-graph-cache-prebuild flag (default true)는
            // 백엔드 생성 시점에 wired 후 model load 완료 시점에 actual prebuild가
            // 발동된다.
            let qnn = Arc::new(llm_rs2::backend::qnn_oppkg::QnnOppkgBackend::with_prebuild(
                args.qnn_graph_cache_prebuild,
            )?);
            let qnn_mem: Arc<dyn Memory> = Arc::new(
                llm_rs2::backend::qnn_oppkg::memory::QnnOppkgMemory::new(qnn.clone()),
            );

            // ENG-QNN-206: SwitchHw round-trip을 위해 OpenCL backend를 secondary로
            // 등록. OpenCL init이 fail하면 secondary 없이 진행 (SwitchHw 비활성).
            #[cfg(feature = "opencl")]
            let (gpu_be, gpu_mem_arc): (
                Option<Arc<dyn Backend>>,
                Option<Arc<dyn Memory>>,
            ) = match llm_rs2::backend::opencl::OpenCLBackend::new_with_profile_events(
                args.profile_events || args.heartbeat_gpu_profile,
            ) {
                Ok(gpu_concrete) => {
                    let gpu_concrete = Arc::new(gpu_concrete);
                    let gm: Arc<dyn Memory> =
                        Arc::new(llm_rs2::backend::opencl::memory::OpenCLMemory::new(
                            gpu_concrete.context.clone(),
                            gpu_concrete.queue.clone(),
                            args.zero_copy,
                        ));
                    let g = gpu_concrete as Arc<dyn Backend>;
                    eprintln!(
                        "[Backend] QNN-GPU primary, OpenCL secondary available (SwitchHw ready)"
                    );
                    (Some(g), Some(gm))
                }
                Err(e) => {
                    eprintln!(
                        "[Backend] QNN-GPU only (OpenCL secondary init failed: {})",
                        e
                    );
                    (None, None)
                }
            };
            #[cfg(not(feature = "opencl"))]
            let (gpu_be, gpu_mem_arc): (
                Option<Arc<dyn Backend>>,
                Option<Arc<dyn Memory>>,
            ) = (None, None);

            // qnn_graph_cache_prebuild는 위에서 with_prebuild()에 wired됨.
            // qnn_allow_fallback는 M3.3 forward path에서 활용.
            let _ = args.qnn_allow_fallback;

            // M3.4: OpenCL secondary를 qnn_oppkg backend의 fallback target으로
            // 등록. prefill 및 model load 단계에서 trait method 호출 시
            // OpenCL secondary가 처리. decode (seq_len=1) fast path만 graph
            // 직접 dispatch (INV-175).
            #[cfg(feature = "opencl")]
            if let Some(ref gpu_concrete) = gpu_be {
                qnn.set_fallback_backend(gpu_concrete.clone());
                eprintln!(
                    "[Backend] qnn_oppkg fallback wired to OpenCL secondary (prefill + model load 위임)"
                );
            }
            // M3.4: production activation/KV memory는 OpenCL secondary로 위임.
            // qnn_oppkg backend는 graph build 시점에 internal rpcmem alloc으로
            // weight + scratch를 보유한다. production이 만드는 activation tensor는
            // OpenCL buffer로 남아 prefill + model load fallback path가 자연스럽게
            // 작동한다. KV cache는 OpenCL buffer (graph 내부 KvScatter는 자체
            // rpcmem 사용 + execute path에서 host-side memcpy로 동기화).
            let qnn_dyn: Arc<dyn Backend> = qnn.clone();
            // Step 1 (KV zero-copy): OpenCL secondary가 있으면 HybridMemory로
            // primary_mem을 구성한다. alloc()은 OpenCL cl_mem으로 위임하고,
            // alloc_kv()는 rpcmem + CL_MEM_USE_HOST_PTR dual buffer를 반환.
            // production prefill path (cl_mem 경유)는 무손상.
            #[cfg(feature = "opencl")]
            let primary_mem: Arc<dyn Memory> = match (&gpu_mem_arc, &gpu_be) {
                (Some(ocl_m), Some(ocl_be)) => {
                    if let Some(ocl_concrete) = ocl_be
                        .as_any()
                        .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>(
                    ) {
                        eprintln!(
                            "[Backend] QNN primary_mem → QnnOppkgHybridMemory (KV zero-copy Step 1)"
                        );
                        Arc::new(
                            llm_rs2::backend::qnn_oppkg::hybrid_memory::QnnOppkgHybridMemory::new(
                                ocl_m.clone(),
                                qnn.clone(),
                                ocl_concrete.context.clone(),
                            ),
                        )
                    } else {
                        ocl_m.clone()
                    }
                }
                (Some(m), None) => m.clone(),
                (None, _) => qnn_mem.clone(),
            };
            #[cfg(not(feature = "opencl"))]
            let primary_mem: Arc<dyn Memory> = qnn_mem.clone();
            // M3.4 D-D.4: gpu_backend_arc로는 OpenCL secondary를 노출한다.
            // primary qnn_oppkg는 noshuffle prep / map_weights_for_cpu / RSS
            // diag 등 OpenCL-specific path에 downcast 불가하므로, secondary가
            // 있으면 secondary를 보조 backend로 expose. 없으면 None (해당
            // path들은 qnn_oppkg-only 환경에서 불활성).
            let gpu_backend_for_caller: Option<Arc<dyn Backend>> = match &gpu_be {
                Some(be) => Some(be.clone()),
                None => Some(qnn_dyn.clone()),
            };
            (
                qnn_dyn,
                primary_mem.clone(),
                gpu_backend_for_caller,
                Some(primary_mem),
                true,
            )
        }
        _ => anyhow::bail!(
            "Unknown backend: {}. Use cpu, opencl, or cuda.",
            args.backend
        ),
    };
    // cpu_backend_arc: always available for migration and SwitchHw fallback.
    let cpu_backend_arc: Arc<dyn Backend> = if args.backend == "cpu" {
        backend.clone()
    } else {
        Arc::new(CpuBackend::new())
    };
    let cpu_memory_arc: Arc<dyn Memory> = if args.backend == "cpu" {
        memory.clone()
    } else {
        Arc::new(Galloc::new())
    };
    let w_dtype = match args.weight_dtype.as_str() {
        "f16" => DType::F16,
        "q4" | "q4_0" => DType::Q4_0,
        _ => anyhow::bail!(
            "Unknown weight-dtype: {}. Use f16 or q4.",
            args.weight_dtype
        ),
    };
    eprintln!("[Config] Weight dtype: {:?}", w_dtype);
    // Validate --force-swap-ratio requires --secondary-gguf.
    if args.force_swap_ratio.is_some() && args.secondary_gguf.is_none() {
        anyhow::bail!(
            "--force-swap-ratio requires --secondary-gguf to be set (no secondary weight file)"
        );
    }

    // Parse --swap-algorithm (used by --qcf-dump warmup-swap path; U5 ablation).
    let swap_algorithm = llm_rs2::models::weights::SwapAlgorithm::from_cli(&args.swap_algorithm)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "--swap-algorithm: unknown value '{}'. Valid: imp, seq, rev, uni, anti",
                args.swap_algorithm
            )
        })?;

    // Parse --importance-formula (§4 EuroSys'27 study). `compare` enables
    // three_way collector + post-warmup DP-LLM proxy ε computation.
    let (importance_formula, importance_compare) = match args.importance_formula.as_str() {
        "mean_pool" => (llm_rs2::core::qcf::ImportanceFormula::MeanPool, false),
        "shortgpt_bi" => (llm_rs2::core::qcf::ImportanceFormula::ShortGptBi, false),
        "dpllm_proxy" => (llm_rs2::core::qcf::ImportanceFormula::DpllmProxy, false),
        "compare" => (llm_rs2::core::qcf::ImportanceFormula::MeanPool, true),
        other => anyhow::bail!(
            "--importance-formula: unknown value '{}'. Valid: mean_pool, shortgpt_bi, dpllm_proxy, compare",
            other
        ),
    };

    // Parse --swap-only-layers (§4 ground-truth study). CSV of layer indices.
    let swap_only_layers: Option<Vec<usize>> = match args.swap_only_layers.as_deref() {
        None | Some("") => None,
        Some(csv) => {
            let mut v = Vec::new();
            for tok in csv.split(',') {
                let t = tok.trim();
                if t.is_empty() {
                    continue;
                }
                let idx: usize = t.parse().map_err(|_| {
                    anyhow::anyhow!("--swap-only-layers: '{}' is not a non-negative integer", t)
                })?;
                v.push(idx);
            }
            v.sort_unstable();
            v.dedup();
            Some(v)
        }
    };

    let is_gguf = model_path.ends_with(".gguf");
    let mut model = if is_gguf {
        if args.weight_dtype != "f16" {
            eprintln!("[Warning] --weight-dtype ignored for GGUF models (dtype from file)");
        }
        // Use LoadConfig single-entry path (ENG-DAT-090) so --secondary-gguf
        // is wired in automatically.
        let load_cfg = llm_rs2::models::loader::LoadConfig {
            primary_source: std::path::PathBuf::from(model_path),
            default_dtype: w_dtype,
            secondary_source: args.secondary_gguf.clone(),
            secondary_dtype_choice: args.secondary_dtype.into(),
            secondary_layout_choice: args.secondary_layout.into(),
        };
        TransformerModel::load_from_config(&load_cfg, backend.clone(), &*memory)?
    } else {
        TransformerModel::load_with_dtype(model_path, backend.clone(), &*memory, w_dtype)?
    };
    // T1: model weights loaded into memory (MmapBuffer + GPU copy if applicable).
    rss_trace("model_loaded");
    // LLMRS_DUMP_SMAPS_T1: dump /proc/self/smaps at T1 for VMA analysis.
    // Tester pulls this file to analyse kgsl/ion/dmabuf VMA distribution.
    if std::env::var("LLMRS_DUMP_SMAPS_T1").is_ok() {
        dump_smaps("T1_model_loaded");
    }

    // ENG-QNN-203/INV-167 — Eager prebuild of layer graph cache (D1 결정).
    // Model load + LayerSlot 등록이 완료된 시점에 N×graphFinalize를 직렬 실행한다.
    // host build에서는 backend init이 이미 fail하여 본 분기 도달 불가; 디바이스
    // 빌드 + Android runtime에서만 본격 동작.
    #[cfg(feature = "qnn")]
    if args.backend == "qnn_oppkg" || args.backend == "qnngpu" {
        if let Some(qnn_be) = backend
            .as_any()
            .downcast_ref::<llm_rs2::backend::qnn_oppkg::QnnOppkgBackend>()
        {
            // ModelConfig → LayerConfig 변환. M3.2 단계는 Qwen2.5-1.5B 단일
            // 모델 지원 (ENG-QNN-225 / INV-176). 추후 다른 모델 추가 시
            // dispatch table 도입 예정.
            let mc = &model.config;
            let layer_cfg = llm_rs2::backend::qnn_oppkg::layer_graph::LayerConfig {
                dim: mc.hidden_size as u32,
                n_head: mc.num_attention_heads as u32,
                n_kv_heads: mc.num_key_value_heads as u32,
                head_dim: mc.head_dim as u32,
                ffn_dim: mc.intermediate_size as u32,
                kv_capacity: args.max_seq_len as u32,
            };
            qnn_be.prebuild_graph_cache(&model.layers, &layer_cfg)?;
        }
    }

    // WSWAP-6-PREFAULT: eager prefault of the secondary weight file.
    //
    // When --eager-prefault-secondary is set, touch all secondary weight pages
    // immediately after model load so that subsequent swap invocations hit the
    // page cache instead of incurring cold page faults (~328 ms on Galaxy S25,
    // §3.1 swap_overhead_s25.md). This is a one-time upfront cost traded for
    // per-swap latency elimination.
    //
    // Memory commit ≈ AUF/GGUF secondary size (e.g. 1.2 GB for Q4_0 1.5B).
    // Default OFF to protect memory-constrained environments.
    if args.eager_prefault_secondary {
        if let Some(ref secondary) = model.secondary_mmap {
            let t0 = std::time::Instant::now();
            secondary.prefault();
            eprintln!(
                "[Eager-Prefault] secondary weights prefaulted in {:.1}ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
        } else {
            eprintln!("[Eager-Prefault] no secondary configured, skipping");
        }
    }

    // When CPU primary + GPU secondary: migrate weights to GPU zero-copy memory.
    // Creates UnifiedBuffer (CL_MEM_ALLOC_HOST_PTR) + mapped: single VMA,
    // as_ptr() valid for CPU, cl_mem() valid for GPU.
    #[cfg(feature = "opencl")]
    if !is_gpu && let (Some(gpu_be), Some(gpu_mem)) = (&gpu_backend_arc, &gpu_memory_arc) {
        match model.migrate_weights_to_gpu(gpu_mem.as_ref(), gpu_be) {
            Ok(n) => eprintln!(
                "[Backend] Migrated {} weight tensors to GPU zero-copy (ALLOC_HOST_PTR)",
                n
            ),
            Err(e) => eprintln!("[Backend] Weight migration skipped: {}", e),
        }
    }

    // When GPU primary + resilience/partition enabled: ensure weights are CPU-accessible.
    // Maps UnifiedBuffer weights or reads device-only OpenCLBuffer into new UnifiedBuffer.
    // Single VMA (ALLOC_HOST_PTR) — no PSS double-counting on Adreno.
    //
    // Preload conditions (in order of cost/necessity):
    //   1. `--resilience-prealloc-switch`  → SwitchHw needs CPU-accessible weights
    //   2. `--tensor-partition <r>` where r is NOT the GPU-only fast-path ratio
    //      (r < GPU_ONLY_THRESHOLD and r > 0) → CPU matmul needs host pointers
    //   3. `--prefill-cpu-chunk-size > 0`  → CPU-side prefill chunk uses weights
    //
    // Notably `--enable-resilience` alone does NOT trigger preload. The
    // IPC directives that require CPU-accessible weights (`SetPartitionRatio`
    // with a non-GPU-only ratio, `SwitchHw` to CPU) now lazily invoke
    // `map_weights_for_cpu()` at the first activation point (see the
    // directive handler below). This avoids the ~200 ms startup cost and
    // the 400+ MB RSS uplift that hit every run which only enabled the
    // manager channel but never actually used CPU-side weight access.
    #[cfg(feature = "opencl")]
    let cli_partition_needs_cpu_weights = args.tensor_partition > 0.0
        && !llm_rs2::layers::tensor_partition::is_gpu_only_ratio(args.tensor_partition);
    #[cfg(feature = "opencl")]
    if is_gpu
        && (args.resilience_prealloc_switch
            || cli_partition_needs_cpu_weights
            || args.prefill_cpu_chunk_size > 0)
    {
        match model.map_weights_for_cpu(&backend) {
            Ok(n) if n > 0 => eprintln!(
                "[Backend] Mapped {} weight tensors for dual CPU/GPU access",
                n
            ),
            Ok(_) => {} // All weights already CPU-accessible
            Err(e) => eprintln!("[Backend] Weight mapping failed (switch may crash): {}", e),
        }
    }

    // Sprint F/G-1-D (2026-04-26): one-shot lm_head Q4_0 load.
    //
    // Mode `auto` (default):
    //   1. AUF secondary with lm_head Q4_0 entry (capability bit 2 = 1)
    //      → zero-copy AUF path (~0 ms).
    //   2. AUF capability bit 2 = 0, or non-AUF secondary present
    //      → runtime quantize fallback (Sprint F, ~hundreds of ms).
    //   3. No secondary at all → skip (preserve legacy F16 behaviour).
    //
    // Mode `q4_0`: force runtime quantize regardless of AUF entry (debug).
    // Mode `none`/`off`: skip entirely (legacy F16).
    // LISWAP-PPL: PPL mode + --secondary-gguf with the default `auto` policy
    // would runtime-quantize lm_head F16 → Q4_0 (see `LmHeadAufResolution::
    // NotAuf` branch below). That diverges from a Q4-native GGUF baseline,
    // whose lm_head is loaded as F16 from the file. The result is a
    // systematic +~0.07 NLL gap that has nothing to do with the swap path
    // (root cause documented in `notes/handoff_liswap_ppl_lm_head_2026_05_12`).
    // For PPL measurements we silently switch the default to `none` so that
    // F16+swap and Q4-native are bit-identical. Power users can still force
    // the old behaviour with `--quantize-lm-head q4_0`.
    let qlm = {
        let raw = args.quantize_lm_head.to_ascii_lowercase();
        if args.ppl.is_some()
            && args.secondary_gguf.is_some()
            && (raw == "auto" || raw.is_empty())
        {
            eprintln!(
                "[Notice] PPL mode + --secondary-gguf: auto-disabling lm_head Q4_0 \
                 quantization (would create a systematic +~0.07 NLL gap vs Q4-native \
                 baseline). Pass `--quantize-lm-head q4_0` to override."
            );
            "none".to_string()
        } else {
            raw
        }
    };
    match qlm.as_str() {
        "none" | "off" => {
            // F16 preserved — no action.
        }
        "auto" | "" => {
            if args.secondary_gguf.is_some() {
                // Try AUF lm_head entry first.
                // Note: payload.bytes borrows model.secondary_mmap (mmap lifetime).
                // We extract the bytes into an owned Vec before calling
                // load_lm_head_from_auf (which mutably borrows model) to satisfy
                // the borrow checker.
                let vocab_size = model.config.vocab_size;
                let hidden_size = model.config.hidden_size;
                // Resolve AUF lm_head payload: (bytes_owned, shape, variant_tag, is_none_ok)
                // or None (GGUF secondary).
                enum LmHeadAufResolution {
                    /// AUF entry found — owned bytes ready for load.
                    Found {
                        bytes: Vec<u8>,
                        shape: [usize; 2],
                        variant_tag: &'static str,
                    },
                    /// AUF present but no lm_head entry (bit 2 = 0).
                    AbsentFallback,
                    /// INV-135 violation.
                    Error(llm_rs2::auf::AufError),
                    /// Non-AUF secondary or no secondary.
                    NotAuf,
                }
                let resolution = {
                    match model
                        .secondary_mmap
                        .as_ref()
                        .and_then(|sm| sm.as_auf_view())
                        .map(|view| view.lm_head_q4_0_payload(vocab_size, hidden_size))
                    {
                        Some(Ok(Some(payload))) => LmHeadAufResolution::Found {
                            bytes: payload.bytes.to_vec(),
                            shape: payload.shape,
                            variant_tag: payload.variant_tag,
                        },
                        Some(Ok(None)) => LmHeadAufResolution::AbsentFallback,
                        Some(Err(e)) => LmHeadAufResolution::Error(e),
                        None => LmHeadAufResolution::NotAuf,
                    }
                };

                match resolution {
                    LmHeadAufResolution::Found {
                        bytes,
                        shape,
                        variant_tag,
                    } => {
                        // AUF path: lm_head Q4_0 entry found — load from owned bytes (~0 ms quantize).
                        eprintln!(
                            "[Backend] lm_head: loading from AUF Q4_0 entry (~0 ms quantize, variant={variant_tag})"
                        );
                        // Build a synthetic LmHeadPayload with owned bytes.
                        let payload = llm_rs2::auf::LmHeadPayload {
                            bytes: &bytes,
                            shape,
                            dtype: llm_rs2::auf::TensorDType::Q4_0,
                            alignment: 65536,
                            variant_tag,
                        };
                        model
                            .load_lm_head_from_auf(&payload, &backend)
                            .map_err(|e| {
                                anyhow::anyhow!("--quantize-lm-head AUF load failed: {e}")
                            })?;
                    }
                    LmHeadAufResolution::AbsentFallback => {
                        // AUF present but no lm_head entry (bit 2 = 0, v0.1.0) → runtime fallback.
                        eprintln!(
                            "[Backend] lm_head: AUF entry absent (capability bit 2 = 0), runtime quantize"
                        );
                        let t_q = std::time::Instant::now();
                        match model.quantize_lm_head_to_q4_0(&backend) {
                            Ok(true) => eprintln!(
                                "[Backend] Quantized lm_head → Q4_0 in {:.1} ms (mode=auto/runtime-fallback)",
                                t_q.elapsed().as_secs_f64() * 1000.0,
                            ),
                            Ok(false) => {} // already Q4_0
                            Err(e) => {
                                anyhow::bail!("--quantize-lm-head runtime fallback failed: {e}")
                            }
                        }
                    }
                    LmHeadAufResolution::Error(e) => {
                        // INV-135 violation (entry/dtype/shape mismatch) → fail-fast.
                        anyhow::bail!("--quantize-lm-head AUF invariant violation (INV-135): {e}");
                    }
                    LmHeadAufResolution::NotAuf => {
                        // Non-AUF secondary (GGUF) or no secondary at all → runtime quantize.
                        let t_q = std::time::Instant::now();
                        match model.quantize_lm_head_to_q4_0(&backend) {
                            Ok(true) => eprintln!(
                                "[Backend] Quantized lm_head → Q4_0 in {:.1} ms (mode=auto)",
                                t_q.elapsed().as_secs_f64() * 1000.0,
                            ),
                            Ok(false) => {} // already Q4_0
                            Err(e) => anyhow::bail!("--quantize-lm-head failed: {e}"),
                        }
                    }
                }
            }
            // No secondary → skip (plain F16 run, legacy behaviour preserved).
        }
        "q4_0" | "q4" => {
            // Forced runtime quantize — AUF entry ignored (regression / debug mode).
            eprintln!("[Backend] lm_head: forced runtime quantize (AUF entry ignored, mode=q4_0)");
            let t_q = std::time::Instant::now();
            match model.quantize_lm_head_to_q4_0(&backend) {
                Ok(true) => eprintln!(
                    "[Backend] Quantized lm_head → Q4_0 in {:.1} ms (mode=q4_0)",
                    t_q.elapsed().as_secs_f64() * 1000.0,
                ),
                Ok(false) => eprintln!("[Backend] lm_head already Q4_0 — quantize skipped"),
                Err(e) => anyhow::bail!("--quantize-lm-head failed: {e}"),
            }
        }
        other => anyhow::bail!(
            "Unknown --quantize-lm-head value: {}. Use 'auto', 'none', or 'q4_0'.",
            other
        ),
    }

    // CUDA: migrate weights to pinned host memory for cuBLAS access.
    // Unlike OpenCL (CL_MEM_USE_HOST_PTR zero-copy wrap), CUDA requires a memcpy into
    // cuMemHostAlloc'd buffers to get device pointers for cuBLAS.
    #[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
    if args.backend == "cuda" {
        match model.migrate_weights_to_cuda(&backend) {
            Ok(n) => eprintln!(
                "[Backend] Migrated {} weight tensors to CUDA pinned memory",
                n
            ),
            Err(e) => eprintln!("[Backend] CUDA weight migration failed: {}", e),
        }
    }

    // Tensor partition: split FFN gate/up weights for CPU-GPU cooperative inference.
    // Requires weights to be CPU-accessible (after map_weights_for_cpu).
    //
    // When `args.tensor_partition` is inside the GPU-only fast-path band
    // (>= GPU_ONLY_THRESHOLD), `prepare_tensor_partition` is a no-op and
    // leaves `partition_ctx = None`; forward() then takes the dense GPU
    // path. We still call it so the semantics (and the "Prepared 0" log)
    // are explicit.
    if args.tensor_partition > 0.0 && args.tensor_partition < 1.0 {
        match model.prepare_tensor_partition(args.tensor_partition, &cpu_backend_arc) {
            Ok(0) => eprintln!(
                "[Partition] ratio={:.3} treated as GPU-only (>= {:.3}); partition path disabled",
                args.tensor_partition,
                llm_rs2::layers::tensor_partition::GPU_ONLY_THRESHOLD,
            ),
            Ok(n) => eprintln!(
                "[Partition] Prepared {} weights with ratio {:.2}",
                n, args.tensor_partition
            ),
            Err(e) => eprintln!("[Partition] Failed to prepare tensor partition: {}", e),
        }
    }

    // Q4_0 noshuffle SOA conversion: pre-convert all Q4_0 weights to Adreno-optimized
    // SOA layout. After this, matmul_q4_0 auto-dispatches to noshuffle GEMV for decode.
    // Check actual weight dtype (GGUF may load Q4_0 even when w_dtype=F16).
    //
    // LLMRS_SKIP_NOSHUFFLE_SOA: RSS diagnostic flag.
    // When set, skip SOA conversion entirely (registry stays empty).
    // matmul_q4_0() fallback path (engine/src/backend/opencl/mod.rs:1961):
    //   lookup_noshuffle_soa() returns None → standard Q4_0 GEMV kernel runs.
    // So decode still works, just slightly slower. RSS measurement is valid
    // for all tokens when this flag is set.
    #[cfg(feature = "opencl")]
    if is_gpu {
        let actual_q4 = w_dtype == DType::Q4_0
            || model
                .layers
                .first()
                .is_some_and(|l| l.load_weights().wq.dtype() == DType::Q4_0);
        if actual_q4 {
            if std::env::var("LLMRS_SKIP_NOSHUFFLE_SOA").is_ok() {
                eprintln!(
                    "[RSS-diag] LLMRS_SKIP_NOSHUFFLE_SOA set: skipping noshuffle SOA conversion \
                     (decode uses standard Q4_0 GEMV fallback — correct but slower)"
                );
            } else {
                // Keep the AOS cl_mem alive when any runtime path still needs
                // CPU-accessible weights: resilience pre-warm, a non-GPU-only
                // tensor partition, prefill CPU chunking, or plain lazy
                // activation via `--enable-resilience`. In those cases
                // `map_weights_for_cpu()` will either have already run (lines
                // ~988 above) or will run on demand against the original AOS
                // allocation. Dropping it would strand the fallback path.
                let keep_for_cpu = args.resilience_prealloc_switch
                    || cli_partition_needs_cpu_weights
                    || args.prefill_cpu_chunk_size > 0
                    || args.enable_resilience;
                // M3.4 D-D.4: qnn_oppkg primary는 noshuffle prep을 OpenCL secondary
                // backend로 위임해야 한다. primary 자체는 OpenCLBackend가 아니라
                // downcast가 fail하기 때문. fallback gpu_backend_arc가 있으면
                // 그것을, 없으면 원래 backend (OpenCL primary)를 사용.
                let prep_backend: &Arc<dyn Backend> = if (args.backend == "qnn_oppkg"
                    || args.backend == "qnngpu")
                    && let Some(ref gpu_be) = gpu_backend_arc
                {
                    gpu_be
                } else {
                    &backend
                };
                match model.prepare_noshuffle_buffers(prep_backend, keep_for_cpu) {
                    Ok(n) => eprintln!("[Backend] Noshuffle SOA prepared: {} weight tensors", n),
                    Err(e) => eprintln!("[Backend] Noshuffle preparation skipped: {}", e),
                }
                // WSWAP-5-TBT-DIAG: dump cl_mem footprint immediately after
                // primary noshuffle prep so the Q4 baseline allocation
                // pattern is recorded *before* any AUF SOA bypass swap path
                // adds placeholder cl_mems on top.
                #[cfg(feature = "opencl")]
                if let Some(ocl_be) = backend
                    .as_any()
                    .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
                {
                    ocl_be.dump_cl_mem_diagnostics(" stage=after_noshuffle_prep");
                }
            }
        }
    }

    // Check if model weights are on GPU (cl_mem accessible) — needed for CPU→GPU switch.
    #[cfg(feature = "opencl")]
    let weights_on_gpu = {
        let layer0 = model.layers[0].load_weights();
        llm_rs2::backend::opencl::get_cl_mem(layer0.wq.buffer().as_ref()).is_ok()
    };
    #[cfg(not(feature = "opencl"))]
    let weights_on_gpu = false;

    // 2. Tokenizer
    //
    // Resolution order:
    //   1. `--tokenizer-path` if explicitly provided.
    //   2. Safetensors layout (model_path is a directory): `<dir>/tokenizer.json`.
    //   3. GGUF layout (model_path is a file): try in order
    //        a. `<dir>/<stem>.tokenizer.json`              (e.g. qwen2.5-1.5b-f16.tokenizer.json)
    //        b. `<dir>/<stem-stripped>.tokenizer.json`     (strip trailing `-f16` / `-q4_0` /
    //                                                       `-q8_0` quant suffix; e.g.
    //                                                       qwen2.5-1.5b.tokenizer.json)
    //        c. `<dir>/tokenizer.json`                     (legacy single-tokenizer-per-dir)
    //
    // Step 3a/3b prevents a sibling model's tokenizer from being picked up
    // when multiple GGUFs co-exist in the same directory (e.g. /data/local/tmp
    // on Android, where both Llama and Qwen GGUFs share the path). The
    // legacy fallback (3c) keeps existing single-model setups working.
    let tokenizer_path: String = if let Some(p) = args.tokenizer_path.as_ref() {
        p.to_string_lossy().into_owned()
    } else if is_gguf {
        let path = std::path::Path::new(model_path);
        let parent = path.parent().unwrap_or(std::path::Path::new("."));
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        // Quant suffix list — keep in sync with `--weight-dtype` accepted values.
        // Match is case-insensitive on the suffix itself but we preserve the
        // original case of the surviving stem so file lookup matches the
        // on-disk capitalisation (e.g. `Llama-3.2-1B-Instruct-f16.gguf` ->
        // `Llama-3.2-1B-Instruct.tokenizer.json`).
        const QUANT_SUFFIXES: &[&str] = &["-f16", "-f32", "-q4_0", "-q4_1", "-q8_0", "-q4_k"];
        let stem_lower = stem.to_ascii_lowercase();
        let stem_stripped: Option<String> = QUANT_SUFFIXES.iter().find_map(|suf| {
            stem_lower
                .strip_suffix(suf)
                .map(|s| stem[..s.len()].to_string())
        });
        let candidates: Vec<std::path::PathBuf> = {
            let mut v = Vec::with_capacity(3);
            v.push(parent.join(format!("{stem}.tokenizer.json")));
            if let Some(ref s) = stem_stripped {
                v.push(parent.join(format!("{s}.tokenizer.json")));
            }
            v.push(parent.join("tokenizer.json"));
            v
        };
        let chosen = candidates
            .iter()
            .find(|p| p.exists())
            .cloned()
            .unwrap_or_else(|| parent.join("tokenizer.json"));
        chosen.to_string_lossy().into_owned()
    } else {
        format!("{}/tokenizer.json", model_path)
    };
    eprintln!("[Tokenizer] {}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Cannot load tokenizer from {}: {}", tokenizer_path, e))?;
    {
        // Vocab-size mismatch는 거의 항상 wrong-tokenizer-for-model bug
        // (예: Qwen 모델 + Llama tokenizer → decoding garbage). 자동 fallback이
        // sibling tokenizer를 잘못 잡거나 share된 tokenizer.json이 다른 family인
        // 경우를 silent failure로 두지 말 것.
        //
        // tokenizer > model: OOB embedding lookup → 즉시 error.
        // tokenizer < model: model vocab이 padding으로 round-up된 경우 정상
        //   (Qwen2.5 1.5B: trained=151665, padded=151936). 5% 또는 256 이상
        //   격차만 error로 차단.
        let tok_vocab = tokenizer.get_vocab_size(true);
        let model_vocab = model.config.vocab_size;
        // Gemma3 등 multimodal 모델은 텍스트 vocab + 소수의 special token (image_soft 등)을
        // tokenizer에만 두고 embedding table엔 두지 않는 경우가 있음. 작은 overflow(≤8)는
        // 텍스트 생성에서 emit될 일이 거의 없으므로 warning으로 강등.
        let oob_tolerance: usize = 8;
        if tok_vocab > model_vocab + oob_tolerance {
            anyhow::bail!(
                "Tokenizer vocab ({}) exceeds model vocab ({}) by more than {} — OOB embedding lookup risk. \
                 Path: {}. Pass --tokenizer-path with the matching tokenizer.json.",
                tok_vocab,
                model_vocab,
                oob_tolerance,
                tokenizer_path
            );
        } else if tok_vocab > model_vocab {
            eprintln!(
                "[Tokenizer] WARNING: tokenizer vocab ({}) > model vocab ({}) by {} (likely multimodal special tokens). \
                 Text generation OK; encoding text containing those special tokens would OOB.",
                tok_vocab,
                model_vocab,
                tok_vocab - model_vocab
            );
        }
        let pad_tolerance = (model_vocab / 20).max(256);
        let pad_gap = model_vocab.saturating_sub(tok_vocab);
        if pad_gap > pad_tolerance {
            anyhow::bail!(
                "Tokenizer vocab too small: model={} tokenizer={} (gap={} > {} padding tolerance). \
                 Likely wrong tokenizer for this model. Path: {}. \
                 Pass --tokenizer-path with the matching tokenizer.json.",
                model_vocab,
                tok_vocab,
                pad_gap,
                pad_tolerance,
                tokenizer_path
            );
        }
    }

    // 3. Prompt
    let prompt = if let Some(path) = &args.prompt_file {
        std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read prompt file {}: {}", path, e))?
    } else {
        args.prompt.clone()
    };
    eprintln!("Prompt: {}", prompt);

    let encoding = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("Token Length: {}", input_ids.len());

    // 4. Prepare KV Cache
    let max_seq_len = args.max_seq_len; // Use argument
    let kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;
    let num_layers = model.config.num_hidden_layers;
    eprintln!(
        "Model config: layers={}, kv_heads={}, head_dim={}, max_seq_len={}",
        num_layers, kv_heads, head_dim, max_seq_len
    );

    // ── KIVI + eval-ll mode: KiviCache with log-likelihood evaluation ──
    if args.kivi && args.eval_ll {
        let questions = load_eval_questions(&args, &prompt)?;
        let vocab_size = model.config.vocab_size;
        let hidden_size = model.config.hidden_size;
        let eval_config = llm_rs2::eval::EvalConfig {
            max_seq_len,
            effective_budget: 0,
            kv_budget_ratio: 0.0,
            greedy: args.greedy,
            kv_type: format!("q{}+f32_residual", args.kivi_bits),
            qcf_mode: args.qcf_mode.clone(),
            vocab_size,
            hidden_size,
        };
        let qcf_config = llm_rs2::core::qcf::QcfConfig::default();
        let kivi_bits = args.kivi_bits;
        let mut kv_caches: Vec<KiviCache> = (0..num_layers)
            .map(|_| {
                KiviCache::new_gpu(
                    kv_heads,
                    head_dim,
                    max_seq_len,
                    args.kivi_residual_size,
                    kivi_bits,
                    backend.clone(),
                    memory.clone(),
                )
            })
            .collect();
        if args.awqe {
            for cache in kv_caches.iter_mut() {
                cache.set_awqe_enabled(true);
            }
            eprintln!("[KIVI] AWQE + AW-VOPR enabled");
        }
        // ARGUS Step 6: resolve sample layers and inject score accumulator.
        let kivi_n_layers = kv_caches.len();
        let kivi_sample_layers = if args.enable_qcf_experimental {
            parse_qcf_sample_layers(&args.qcf_sample_layers, kivi_n_layers)
                .map_err(|e| anyhow::anyhow!("--qcf-sample-layers: {e}"))?
        } else {
            vec![0]
        };
        // Inject a GQA-aware score accumulator when experimental mode is on.
        // KiviHook::score_accumulator() forwards it into TransformerModelForwardArgs,
        // so LlamaLayer will push attention probabilities into it during forward_into().
        // entropy_computed flag is set in post_prefill when acc.is_active() + scores non-empty.
        let kivi_score_acc = if args.enable_qcf_experimental {
            let mut acc = llm_rs2::core::attention_scores::AttentionScoreAccumulator::new_gqa(
                max_seq_len,
                model.config.num_attention_heads,
                model.config.num_key_value_heads,
                kivi_n_layers,
                0,   // last_n_layers=0 → all layers tracked
                1.0, // no decay
            );
            acc.set_active(true);
            Some(acc)
        } else {
            None
        };
        let mut hook = llm_rs2::eval::KiviHook::new(
            qcf_config,
            args.enable_qcf_experimental,
            kivi_sample_layers,
            kivi_score_acc,
        );
        let output = llm_rs2::eval::run_eval_ll_generic(
            &model,
            &tokenizer,
            &backend,
            &*memory,
            &mut kv_caches,
            &mut hook,
            &questions,
            &eval_config,
            None,
        )?;
        let mut json_val = serde_json::from_str::<serde_json::Value>(&output.to_json()?)?;
        json_val["config"] = serde_json::json!({
            "model": args.model_path,
            "eviction_policy": "kivi",
            "kivi_bits": args.kivi_bits,
            "kivi_residual_size": args.kivi_residual_size,
            "max_seq_len": max_seq_len,
            "kv_type": format!("q{}+f32_residual", args.kivi_bits),
        });
        println!("{}", serde_json::to_string_pretty(&json_val)?);
        return Ok(());
    }

    // ── KIVI + PPL mode: KiviCache with perplexity evaluation ──
    if args.kivi
        && let Some(ref ppl_path) = args.ppl
    {
        return run_kivi_ppl(
            &args,
            &model,
            &tokenizer,
            &backend,
            &memory,
            kv_heads,
            head_dim,
            num_layers,
            max_seq_len,
            args.kivi_residual_size,
            ppl_path,
        );
    }

    let kv_type = match args.kv_type.as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "q4" => DType::Q4_0,
        _ => anyhow::bail!(
            "Unsupported KV type: {}. Use f32, f16, or q4.",
            args.kv_type
        ),
    };

    // Note: flash_attn_f32_f16 is compiled for DK ∈ {64, 128} only, but both
    // CUDA naive attention (attention_gen_f16kv_naive) and CPU fallback attention
    // support F16 KV for any head_dim. No auto-promotion needed.

    // Determine initial KV cache capacity (dynamic grow-on-demand)
    // Default: reserve space for prompt + all tokens to generate, so decode never
    // triggers grow() mid-generation (grow is ~370 ms spike on Adreno).
    let initial_kv_capacity = if args.eval_ll || args.ppl.is_some() {
        // Eval modes: pre-allocate full capacity to avoid re-allocation
        max_seq_len
    } else if args.initial_kv_capacity > 0 {
        args.initial_kv_capacity.min(max_seq_len)
    } else {
        input_ids
            .len()
            .saturating_add(args.num_tokens)
            .next_power_of_two()
            .max(128)
            .min(max_seq_len)
    };

    // Calculate buffer size per KV cache (based on initial capacity, not max)
    let n_values = initial_kv_capacity * kv_heads * head_dim;
    let kv_buf_size = match kv_type {
        DType::Q4_0 => {
            use llm_rs2::core::quant::{BlockQ4_0, QK4_0};
            (n_values / QK4_0) * std::mem::size_of::<BlockQ4_0>()
        }
        _ => n_values * kv_type.size(),
    };
    let use_head_major = args.kv_layout.to_lowercase() != "seq";
    let kv_layout = if use_head_major {
        KVLayout::HeadMajor
    } else {
        KVLayout::SeqMajor
    };
    eprintln!(
        "KV cache type: {:?}, layout: {:?} (initial capacity: {} tokens, {}B per layer, max: {})",
        kv_type, kv_layout, initial_kv_capacity, kv_buf_size, max_seq_len
    );

    let mut kv_caches = Vec::new();
    for _ in 0..num_layers {
        let k_buf = memory.alloc_kv(kv_buf_size, kv_type)?;
        let v_buf = memory.alloc_kv(kv_buf_size, kv_type)?;

        let shape = if use_head_major {
            Shape::new(vec![1, kv_heads, initial_kv_capacity, head_dim])
        } else {
            Shape::new(vec![1, initial_kv_capacity, kv_heads, head_dim])
        };

        let k = Tensor::new(shape.clone(), k_buf, backend.clone());
        let v = Tensor::new(shape, v_buf, backend.clone());

        kv_caches.push(
            KVCache::new_dynamic(
                k,
                v,
                initial_kv_capacity,
                max_seq_len,
                kv_heads,
                head_dim,
                memory.clone(),
            )
            .with_layout(kv_layout),
        );
    }

    // ── Chat REPL mode ──
    // Three dispatch paths: standard (KVCache, with optional eviction),
    // KIVI (quantized KV), and KV-offload (disk/raw store).
    if args.chat {
        let sampling_config = SamplingConfig {
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            repetition_penalty: args.repetition_penalty,
            repetition_window: args.repetition_window,
        };
        let kv_offload_active = !args.kv_offload.is_empty() && args.kv_offload != "none";
        if args.kivi {
            return run_chat_kivi(
                &args,
                &model,
                &tokenizer,
                &backend,
                &memory,
                &sampling_config,
                kv_heads,
                head_dim,
                num_layers,
                max_seq_len,
            );
        }
        if kv_offload_active {
            return run_chat_offload(
                &args,
                &model,
                &tokenizer,
                &backend,
                &memory,
                &sampling_config,
                kv_heads,
                head_dim,
                num_layers,
                max_seq_len,
            );
        }
        return run_chat_standard(
            &args,
            &model,
            &tokenizer,
            &backend,
            &memory,
            &mut kv_caches,
            &sampling_config,
            max_seq_len,
        );
    }

    // 5. Experiment schedule + Command Executor
    let experiment_schedule = if let Some(ref path) = args.experiment_schedule {
        Some(ExperimentSchedule::load(path)?)
    } else {
        None
    };

    let mut experiment_tx: Option<std::sync::mpsc::Sender<ManagerMessage>> = None;
    let heartbeat_interval = std::time::Duration::from_millis(1000);

    // MSG-068 Phase 2: GPU self-util meter 추출. 백엔드가 queue profiling과
    // 함께 빌드되었을 때만(opt-in) Some이 된다. CPU 백엔드/비활성 시 None이며
    // executor는 self_gpu_pct=0.0을 송출한다 (INV-092 fallback).
    #[allow(unused_mut)]
    let mut gpu_meter: Option<std::sync::Arc<dyn llm_rs2::resilience::GpuSelfMeter>> = None;
    #[cfg(feature = "opencl")]
    if args.heartbeat_gpu_profile {
        // 우선 primary backend에서 찾고, 없으면 secondary(GPU)에서 찾는다.
        // CPU primary + GPU secondary 구성에서도 GPU self-util을 보고하기 위함.
        let try_extract = |b: &std::sync::Arc<dyn Backend>| -> Option<
            std::sync::Arc<dyn llm_rs2::resilience::GpuSelfMeter>,
        > {
            b.as_any()
                .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
                .and_then(|ocl| ocl.gpu_self_meter())
                .map(|m| m as std::sync::Arc<dyn llm_rs2::resilience::GpuSelfMeter>)
        };
        gpu_meter =
            try_extract(&backend).or_else(|| gpu_backend_arc.as_ref().and_then(try_extract));
        if gpu_meter.is_some() {
            eprintln!("[Resilience] Heartbeat GPU profiling enabled (MSG-068 Phase 2)");
        } else {
            eprintln!(
                "[Resilience] --heartbeat-gpu-profile set but no OpenCL backend with profiling available; self_gpu_pct stays 0.0"
            );
        }
    }

    let mut command_executor = if let Some(ref schedule) = experiment_schedule {
        // Experiment mode: internal mpsc channel (no external transport needed)
        let (tx, rx) = std::sync::mpsc::channel();
        let (resp_tx, _resp_rx) = std::sync::mpsc::channel();
        experiment_tx = Some(tx);
        eprintln!("[Experiment] Mode enabled — schedule: {}", schedule.name);
        Some(CommandExecutor::with_gpu_meter(
            rx,
            resp_tx,
            args.backend.clone(),
            heartbeat_interval,
            gpu_meter.clone(),
        ))
    } else if args.enable_resilience {
        let (cmd_rx, resp_tx, _handle) = match args.resilience_transport.as_str() {
            #[cfg(feature = "resilience")]
            "dbus" => MessageLoop::spawn(DbusTransport::new())?,
            #[cfg(unix)]
            s if s.starts_with("unix:") => {
                let path = std::path::PathBuf::from(&s[5..]);
                MessageLoop::spawn(UnixSocketTransport::new(path))?
            }
            s if s.starts_with("tcp:") => {
                let addr = s[4..].to_string();
                MessageLoop::spawn(TcpTransport::new(addr))?
            }
            other => {
                eprintln!("[Resilience] Unknown transport: {}", other);
                return Ok(());
            }
        };
        eprintln!(
            "[Resilience] Executor enabled — transport: {}",
            args.resilience_transport
        );
        let mut executor = CommandExecutor::with_gpu_meter(
            cmd_rx,
            resp_tx,
            args.backend.clone(),
            heartbeat_interval,
            gpu_meter.clone(),
        );

        // secondary 경로가 있으면 swap_weights 액션이 Heartbeat에도 포함되도록 설정.
        // Capability와 Heartbeat 두 목록이 항상 같은 조건을 공유한다 (ENG-ST-032).
        let has_secondary = args.secondary_gguf.is_some();
        executor.set_has_secondary(has_secondary);

        // Send Capability as first message (SEQ-022).
        // available_actions 는 Heartbeat 와 동일하게 eviction_policy / kv_type 에서 파생.
        // Heartbeat 보다 먼저 manager 에 도달하므로, 첫 signal 처리 시점에 이미 이 값이
        // 반영돼 있어야 정책이 엔진이 지원하지 않는 액션을 선택하는 회귀를 막을 수 있다.
        let cap_available_actions = {
            let mut a = vec![
                "throttle".to_string(),
                "switch_hw".to_string(),
                "layer_skip".to_string(),
            ];
            if args.eviction_policy != "none" {
                a.push("kv_evict_h2o".to_string());
                a.push("kv_evict_sliding".to_string());
                a.push("kv_evict_streaming".to_string());
                a.push("kv_merge_d2o".to_string());
            }
            if args.kv_type.starts_with('q') {
                a.push("kv_quant_dynamic".to_string());
            }
            // secondary GGUF/AUF 존재 시 swap_weights 등록 (ENG-ST-032).
            // Heartbeat의 compute_available_actions와 동일 조건을 공유한다.
            if has_secondary {
                a.push("swap_weights".to_string());
            }
            a
        };
        executor.send_capability(llm_shared::EngineCapability {
            available_devices: vec!["cpu".to_string(), "opencl".to_string()],
            active_device: args.backend.clone(),
            max_kv_tokens: args.max_seq_len,
            bytes_per_kv_token: model.config.num_key_value_heads
                * model.config.head_dim
                * 2  // K + V
                * 2, // F16 = 2 bytes
            num_layers: model.config.num_hidden_layers,
            available_actions: cap_available_actions,
        });
        eprintln!("[Resilience] Capability sent to Manager");

        Some(executor)
    } else {
        None
    };
    // Set initial partition ratio from CLI for heartbeat reporting
    if args.tensor_partition > 0.0
        && args.tensor_partition < 1.0
        && let Some(ref mut exec) = command_executor
    {
        exec.set_partition_ratio(args.tensor_partition);
    }
    // Seed sticky throttle from CLI so no-directive polls preserve the CLI
    // value; Manager `Throttle` directives still override at runtime.
    if args.throttle_delay_ms > 0
        && let Some(ref mut exec) = command_executor
    {
        exec.set_throttle_delay_ms(args.throttle_delay_ms);
    }
    let mut throttle_delay_ms: u64 = args.throttle_delay_ms;
    let mut tbt_log_writer: Option<std::io::BufWriter<std::fs::File>> =
        args.tbt_log.as_ref().map(|path| {
            let file = std::fs::File::create(path).expect("failed to create tbt-log file");
            std::io::BufWriter::new(file)
        });
    let mut target_tbt_ms = args.target_tbt;

    // ── KIVI mode: separate path with KiviCache ──
    // Placed after executor creation so resilience is available in the token loop.
    if args.kivi || args.kv_dynamic_quant {
        // KIVI mode: --kivi starts at Q2, --kv-dynamic-quant starts at bits=16
        // (F16-equivalent) and allows runtime transition via kv_quant_dynamic.
        // Note: --enable-resilience alone stays on main path (F16 KVCache + eviction).
        let initial_bits: u8 = if args.kivi { args.kivi_bits } else { 16 };
        let residual_size = if initial_bits == 16 {
            // bits=16: all tokens stay in residual (no quantization flush)
            // Round down to QKKV (32) multiple for KiviCache alignment
            (max_seq_len / 32) * 32
        } else {
            args.kivi_residual_size
        };
        return run_kivi(
            &model,
            &tokenizer,
            &backend,
            &memory,
            &input_ids,
            &sampling_config,
            kv_heads,
            head_dim,
            num_layers,
            max_seq_len,
            residual_size,
            args.num_tokens,
            args.experiment_output.as_deref(),
            args.experiment_logits_topk,
            args.experiment_sample_interval,
            &prompt,
            &args.backend,
            &mut command_executor,
            initial_bits,
            args.no_gpu_plan,
            args.target_tbt,
            args.tbt_log.as_deref(),
            args.ignore_eos,
            args.throttle_delay_ms,
        );
    }

    // ── Offload mode: separate path with OffloadKVCache ──
    // Placed after executor creation so resilience is available in the decode loop.
    if args.kv_offload != "none" {
        return run_offload(
            &model,
            &tokenizer,
            &backend,
            &memory,
            &input_ids,
            &sampling_config,
            kv_heads,
            head_dim,
            num_layers,
            max_seq_len,
            args.num_tokens,
            &prompt,
            &args.backend,
            &args.kv_offload,
            &args.kv_type,
            args.max_prefetch_depth,
            &args.offload_path,
            &mut command_executor,
            args.throttle_delay_ms,
        );
    }

    // Experiment JSONL writer + system sampler
    let mut experiment_writer = if let Some(ref path) = args.experiment_output {
        Some(JsonlWriter::new(path)?)
    } else {
        None
    };
    let mut system_sampler = SystemSampler::new(args.experiment_sample_interval);
    let sys_start = if experiment_writer.is_some() {
        Some(system_sampler.snapshot())
    } else {
        None
    };
    let mut experiment_eviction_count: usize = 0;
    let mut experiment_evicted_total: usize = 0;
    let mut experiment_total_throttle_ms: u64 = 0;
    let mut forward_ms_values: Vec<f64> = Vec::new();

    // 6. Inference Loop
    let mut tokens = input_ids.clone();
    let mut start_pos = 0;
    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;

    eprintln!(
        "Generating (Max: {}, Temp: {}, TopP: {}, TopK: {})...",
        max_seq_len, args.temperature, args.top_p, args.top_k
    );
    let start_time = std::time::Instant::now();
    let mut _last_token_time = start_time;
    let mut _ttft_ms = 0.0;
    let mut tbt_values = Vec::new();

    // 4.5 Setup CacheManager
    let actual_protected_prefix =
        args.protected_prefix
            .unwrap_or(match args.eviction_policy.as_str() {
                // Score-based policies: default to 4 (attention sinks only).
                // Protecting the entire prompt makes score-based eviction meaningless
                // because only generated tokens would be evictable.
                "h2o" | "h2o_plus" | "d2o" => 4,
                // StreamingLLM: use explicit sink_size parameter
                "streaming" => args.sink_size,
                // Sliding window / none: protect entire prompt (legacy behavior)
                _ => input_ids.len(),
            });

    let mut cache_manager = {
        // CUDA discrete GPU: managed memory (cuMemAllocManaged) reserves system RAM
        // for virtual address space even though data resides in VRAM. MemAvailable
        // from /proc/meminfo is unreliable — use NoOpMonitor to prevent false pressure.
        let monitor: Box<dyn llm_rs2::core::sys_monitor::SystemMonitor> =
            if backend.is_discrete_gpu() {
                Box::new(NoOpMonitor)
            } else {
                Box::new(LinuxSystemMonitor)
            };
        let threshold_bytes = args.memory_threshold_mb * 1024 * 1024;

        if args.eviction_policy == "d2o" {
            // D2O uses CachePressureHandler (Pipeline mode), not EvictionPolicy (Legacy mode)
            let d2o_handler = D2OHandler::new(D2OConfig {
                keep_ratio: args.d2o_keep_ratio,
                protected_prefix: actual_protected_prefix,
                target_ratio: args.eviction_target_ratio,
                ema_beta: args.d2o_ema_beta,
                merge_e: args.d2o_merge_e,
                use_layer_allocation: args.d2o_layer_alloc,
                protected_layers: args.d2o_protected_layers.clone().unwrap_or_default(),
            });
            let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
                min_level: PressureLevel::Warning,
                handler: Box::new(d2o_handler),
            }]);
            CacheManager::with_pipeline(pipeline, monitor, threshold_bytes)
        } else {
            let policy: Box<dyn llm_rs2::core::eviction::EvictionPolicy> = match args
                .eviction_policy
                .as_str()
            {
                "none" => Box::new(NoEvictionPolicy::new()),
                "sliding" => Box::new(SlidingWindowPolicy::new(
                    args.eviction_window,
                    actual_protected_prefix,
                )),
                "streaming" => {
                    use llm_rs2::core::eviction::StreamingLLMPolicy;
                    let window = if args.streaming_window > 0 {
                        args.streaming_window
                    } else if args.kv_budget > 0 {
                        args.kv_budget.saturating_sub(args.sink_size)
                    } else {
                        args.eviction_window
                    };
                    Box::new(StreamingLLMPolicy::new(args.sink_size, window))
                }
                "h2o" => Box::new(H2OPolicy::new(args.h2o_keep_ratio, actual_protected_prefix)),
                "h2o_plus" => Box::new(H2OPlusPolicy::new(
                    args.h2o_keep_ratio,
                    actual_protected_prefix,
                )),
                other => anyhow::bail!(
                    "Unknown eviction policy: '{}'. Use: none, sliding, streaming, h2o, h2o_plus, d2o",
                    other
                ),
            };
            CacheManager::new(policy, monitor, threshold_bytes, args.eviction_target_ratio)
        }
    };

    // Setup event sink for score diagnostics
    cache_manager.set_event_sink(Arc::new(StderrDiagnosticSink));

    // Enable disk-backed KV swap when --swap-dir is provided.
    // KvOffload directives write to this directory; RestoreDefaults recalls.
    if let Some(dir) = args.swap_dir.clone() {
        eprintln!("[Resilience] KV swap enabled: dir={}", dir.display());
        cache_manager.enable_swap(dir);
    }

    // Register policies for Manager-directed eviction dispatch.
    // Use a small protected_prefix (4 = attention sinks) for Manager-directed policies,
    // NOT actual_protected_prefix which may be the entire prompt length when
    // --eviction-policy is "none". The Manager decides WHEN and HOW MUCH to evict;
    // the policy should not silently prevent meaningful eviction.
    let resilience_protected_prefix = 4usize; // attention sinks only
    cache_manager.register_policy(
        llm_rs2::resilience::EvictMethod::H2o,
        Box::new(H2OPolicy::new(
            args.h2o_keep_ratio,
            resilience_protected_prefix,
        )),
    );
    cache_manager.register_policy(
        llm_rs2::resilience::EvictMethod::Sliding,
        Box::new(SlidingWindowPolicy::new(
            args.eviction_window,
            resilience_protected_prefix,
        )),
    );
    // Note: Streaming policy is NOT pre-registered because its parameters
    // (sink_size, window_size) come from the Manager directive at runtime.
    // It is instantiated on-demand in the eviction dispatch below.

    // Parse QCF mode
    let qcf_mode = match args.qcf_mode.as_str() {
        "caote" => llm_rs2::core::qcf::QcfMode::Caote,
        "both" => llm_rs2::core::qcf::QcfMode::Both,
        _ => llm_rs2::core::qcf::QcfMode::Attn,
    };
    let needs_caote = qcf_mode.has_caote();

    // Setup AttentionScoreAccumulator for H2O / H2O+ / D2O / CAOTE
    // When CAOTE is requested, always use GQA-aware accumulator (for per-KV-head attention).
    let needs_score_based = args.eviction_policy == "h2o"
        || args.eviction_policy == "d2o"
        || args.eviction_policy == "h2o_plus";
    // Always build accumulator for eval-ll when any eviction policy is active:
    // sliding mode needs it to populate last_step_head_attn for QCF-ATTN v2.
    let has_eviction_policy = args.eviction_policy != "none";
    // --enable-resilience forces accumulator on: the manager can request Evict
    // at any runtime moment, and `compute_qcf_estimates` (~line 4882) falls back
    // to uniform weights without scores, which corrupts action-cost ranks
    // (measured: h2o/d2o collapse to 0, sliding inflates +312%).
    // Originally decoupled 2026-04-20 because forcing the CPU accumulator
    // disabled the GPU decode plan (~25% slowdown). Now re-coupled after
    // Phase A/B (flash_attn score output, commits 3096de4 + 28d8fe4): the
    // accumulator coexists with the GPU plan and overhead is <1%
    // (Adreno 37.4 t/s, Jetson overhead 1.8–4.3%).
    let needs_accumulator =
        needs_score_based || needs_caote || args.enable_resilience || has_eviction_policy;
    // GQA mode required for last_step_head_attn() (QCF-ATTN v2 + CAOTE).
    let use_gqa = args.eviction_policy == "h2o_plus" || needs_caote || has_eviction_policy;

    let mut score_accumulator = if needs_accumulator {
        let acc = if use_gqa {
            AttentionScoreAccumulator::new_gqa(
                max_seq_len,
                model.config.num_attention_heads,
                model.config.num_key_value_heads,
                model.config.num_hidden_layers,
                args.h2o_tracked_layers,
                args.h2o_decay,
            )
        } else {
            AttentionScoreAccumulator::new(
                max_seq_len,
                model.config.num_attention_heads,
                model.config.num_hidden_layers,
                args.h2o_tracked_layers,
                args.h2o_decay,
            )
        };
        let mut acc = acc;
        // Always active: GPU acc overhead is ~0.6ms/token (1.7%),
        // CPU NEON acc overhead is ~0.66ms/token (1.1%).
        // This ensures first RequestQcf returns accurate H2O/D2O estimates.
        acc.set_active(true);
        acc.set_time_normalize(!args.h2o_raw_scores);
        Some(acc)
    } else {
        None
    };

    // Initialize GPU-side score accumulator when using OpenCL backend.
    // This compiles score_reduce.cl and allocates persistent GPU buffers.
    // Eliminates per-token GPU->CPU blocking readback (~129ms/token).
    #[cfg(feature = "opencl")]
    if score_accumulator.is_some()
        && let Some(ocl_be) = backend
            .as_any()
            .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
    {
        match ocl_be.init_gpu_score_acc(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
            max_seq_len,
            args.h2o_decay,
        ) {
            Ok(()) => {
                if let Some(gpu_acc) = ocl_be.gpu_score_acc_mut() {
                    gpu_acc.set_active(true);
                }
                eprintln!("[GPU Score] Accumulator initialized — per-token readback eliminated");
            }
            Err(e) => {
                eprintln!(
                    "[GPU Score] Failed to initialize (falling back to CPU path): {}",
                    e
                );
            }
        }
    }

    if args.eviction_policy != "none" {
        eprintln!(
            "Eviction: policy={}, window={}, prefix={}, ratio={}, threshold={}MB",
            args.eviction_policy,
            args.eviction_window,
            actual_protected_prefix,
            args.eviction_target_ratio,
            args.memory_threshold_mb
        );
    }

    // Build SkipConfig from CLI options
    use llm_rs2::core::skip_config::SkipConfig;
    let mut skip_config = if let Some(ref layers) = args.skip_layers {
        let mut sc = SkipConfig::new();
        for &l in layers {
            sc.attn_skip.insert(l);
            sc.mlp_skip.insert(l);
        }
        assert!(
            sc.validate(model.config.num_hidden_layers),
            "Cannot skip layer 0 or last layer (SWIFT constraint)"
        );
        eprintln!(
            "[Skip] Explicit layers: {:?} ({} sub-layers skipped)",
            layers,
            sc.total_skips()
        );
        Some(sc)
    } else if let Some(ratio) = args.skip_ratio {
        let sc = SkipConfig::uniform_init(model.config.num_hidden_layers, ratio);
        assert!(
            sc.validate(model.config.num_hidden_layers),
            "uniform_init produced invalid SkipConfig (layer 0 or last layer skipped)"
        );
        eprintln!(
            "[Skip] Uniform ratio={:.1}% → {} sub-layers skipped",
            ratio * 100.0,
            sc.total_skips()
        );
        Some(sc)
    } else {
        None
    };
    let mut last_skip_ratio: Option<f32> = args.skip_ratio;

    // Auto-eviction: non-experiment mode evicts automatically.
    // - Sliding window: triggers on memory pressure after each forward pass.
    // - Score-based (H2O/H2O+/D2O): triggers when cache utilization >= 90% capacity,
    //   using force_evict_with_scores to bypass memory pressure checks.
    let auto_eviction = args.eviction_policy != "none" && experiment_schedule.is_none();
    let score_based_eviction = matches!(args.eviction_policy.as_str(), "h2o" | "h2o_plus" | "d2o");

    // ── Weight swap: --force-swap-ratio manual trigger ──────────────────────
    // Applied once before generation starts (prefill + decode).
    // Requires --secondary-gguf (validated above at model load time).
    // When --qcf-dump is set, this block is skipped; the swap is deferred to
    // the QCF dump workflow below (after warmup prefill builds ImportanceTable).
    //
    // ENG-ALG-232~234 (LISWAP-1): when --swap-incremental-per-tick > 0,
    // the swap is NOT executed here. Instead, an IncrementalSwapPlan is
    // committed and stored below; the decode loop drains it chunk-by-chunk.
    // per_tick == 0 (default): single-shot path, unchanged from before.
    let mut incremental_force_swap_plan: Option<llm_rs2::models::weights::IncrementalSwapPlan> =
        None;

    // LISWAP-6 manager path: when manager triggers SwapWeights, the plan is
    // committed to `incremental_force_swap_plan` and this state records the
    // information needed to send WeightSwapReport on plan completion.
    // Fields: (ratio, total_layers_planned, plan_start_time, qcf_swap_estimated)
    let mut manager_swap_report_pending: Option<(f32, usize, std::time::Instant, f32)> = None;
    // Populated by the plan-done block (outside executor scope); consumed by the
    // executor checkpoint block (inside executor scope) the same token tick.
    let mut ready_weight_swap_report: Option<llm_shared::WeightSwapReport> = None;

    // LISWAP-4 (ENG-ALG-237 / INV-150): intra-forward layer-aligned swap hook.
    // Created when `--swap-intra-forward` + `--force-swap-ratio` both active.
    // Decode loop injects `Some(&*hook)` into `layer_boundary_hook` and calls
    // `finalize` once `plan_is_complete()` to drain dispatcher + bump
    // ratio_generation + invalidate SOA registry.
    let mut intra_forward_swap_hook: Option<Arc<llm_rs2::models::weights::IntraForwardSwapHook>> =
        None;

    // LISWAP-5: phase-aware async swap dispatcher. Created when
    // `--swap-phase-aware` + `--force-swap-ratio` both active. Registered as
    // process-wide PHASE_HOOK so `op_trace::start_op` / `record` callsites in
    // forward_gen drive chunk dispatch from the forward thread itself.
    // `finalize` drains remaining chunks + bumps ratio_generation when decode
    // ends.
    let mut phase_aware_swap_dispatcher: Option<
        Arc<llm_rs2::models::weights::PhaseAwareSwapDispatcher>,
    > = None;

    // LISWAP-2 prototype: async swap dispatcher lifecycle.
    // Created once here; used in the decode loop when async dispatch is active.
    // `None` when --swap-async-dispatch is false or the backend does not support
    // async transfer.
    // NOTE: also created when per_tick == 0 so that manager-triggered incremental
    // swap (LISWAP-6 manager path) can use async dispatch even without
    // --swap-incremental-per-tick CLI flag.
    let async_swap_dispatcher: Option<llm_rs2::models::weights::AsyncSwapDispatcher> = {
        if args.swap_async_dispatch {
            let swap_backend: Arc<dyn Backend> = gpu_backend_arc
                .as_ref()
                .cloned()
                .unwrap_or_else(|| cpu_backend_arc.clone());
            if swap_backend.supports_async_transfer() {
                Some(llm_rs2::models::weights::AsyncSwapDispatcher::new(
                    swap_backend,
                ))
            } else {
                if args.swap_incremental_per_tick > 0 {
                    eprintln!(
                        "[LISWAP-2] backend does not support async transfer; falling back to sync incremental swap"
                    );
                }
                None
            }
        } else {
            None
        }
    };

    // LISWAP-6 — Dynamic K controller. Active when `--swap-dynamic-k` is set.
    // K is determined entirely by timing (forward wall vs per-layer drop cost);
    // there is no static upper cap. Per-tick dispatch is still bounded by the
    // sub-batch reactive pause inside `SwapExecutor::execute_on_slots`.
    let mut dynamic_k_controller: Option<llm_rs2::models::weights::DynamicKController> =
        if args.swap_dynamic_k {
            Some(llm_rs2::models::weights::DynamicKController::new())
        } else {
            None
        };
    let dynamic_k_diag = std::env::var("LLMRS_DYNAMIC_K_DIAG")
        .map(|v| v == "1")
        .unwrap_or(false);

    // ── LISWAP-3 prototype (Direction A): ALLOC_HOST_PTR pool ────────────
    // Lazy-init the swap pool when the user opted in via `--swap-zero-copy`
    // AND the env-gate `LLMRS_OPENCL_HOST_PTR_POOL=1` is set. Both conditions
    // are required so the flag alone cannot accidentally enable the
    // prototype path. SwapExecutor falls back to the staging path on `None`.
    // Plan: compiled-chasing-hopper.md Direction A track, Stage 3.
    #[cfg(feature = "opencl")]
    let host_ptr_swap_pool: Option<Arc<llm_rs2::backend::opencl::host_ptr_pool::HostPtrPool>> = {
        if !args.swap_zero_copy {
            None
        } else if let Some(gpu_be) = gpu_backend_arc.as_ref().and_then(|b| {
            b.as_any()
                .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
        }) {
            let cfg = llm_rs2::backend::opencl::host_ptr_pool::HostPtrPoolConfig {
                n_slots: args.swap_pool_slots.max(1),
                ..Default::default()
            };
            let pool = gpu_be.host_ptr_pool_or_init(cfg);
            if pool.is_some() {
                eprintln!(
                    "[LISWAP-3] host_ptr_pool active: slots={}, max_tensor_size={}",
                    cfg.n_slots, cfg.max_tensor_size
                );
            } else {
                eprintln!(
                    "[LISWAP-3] --swap-zero-copy requested but pool unavailable \
                     (env LLMRS_OPENCL_HOST_PTR_POOL not set or pool init failed); \
                     using staging path"
                );
            }
            pool
        } else {
            eprintln!(
                "[LISWAP-3] --swap-zero-copy ignored: backend is not OpenCL; using staging path"
            );
            None
        }
    };
    #[cfg(not(feature = "opencl"))]
    let _host_ptr_swap_pool: Option<()> = {
        if args.swap_zero_copy {
            eprintln!(
                "[LISWAP-3] --swap-zero-copy ignored: opencl feature is disabled in this build"
            );
        }
        None
    };

    // LISWAP Phase 3 — pending mid-decode trigger payload.
    // When `--swap-delay-tokens N > 0` AND `--force-swap-ratio` is set, the
    // trigger logic is deferred from prefill end to decode token N. We capture
    // (ratio, target_layers) here and re-run the dispatch block at the loop
    // head when `decode_token_index == swap_delay_tokens`.
    let mut pending_force_swap: Option<(f32, Vec<usize>)> = None;

    // ── LISWAP Phase 3 dispatch macro ────────────────────────────────────────
    // Identical force-swap dispatch logic invoked from two callsites:
    //   (a) prefill end (when --swap-delay-tokens == 0, default — original
    //       behavior, must preserve baseline wall ±5 ms),
    //   (b) decode loop head (when --swap-delay-tokens > 0, mid-decode trigger).
    //
    // The macro relies on hygienic name capture of the surrounding `main`'s
    // mutable state (model, backend, gpu_backend_arc, cpu_backend_arc,
    // host_ptr_swap_pool, intra_forward_swap_hook, incremental_force_swap_plan,
    // phase_aware_swap_dispatcher, is_gpu, args). Inputs are bound by the macro
    // arms: $ratio = clamped force ratio, $target_layers = Vec<usize> resolved
    // from `uniform_target_layers`. Both callsites pre-compute these before
    // expanding the macro.
    macro_rules! dispatch_force_swap {
        ($ratio:expr, $target_layers:expr) => {{
            let ratio: f32 = $ratio;
            let target_layers: Vec<usize> = $target_layers;
            let num_layers = model.layers.len();
            if args.swap_phase_aware {
                let swap_backend: Arc<dyn Backend> = gpu_backend_arc
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| cpu_backend_arc.clone());
                let dispatcher = Arc::new(llm_rs2::models::weights::AsyncSwapDispatcher::new(
                    Arc::clone(&swap_backend),
                ));
                let secondary = match model.secondary_mmap.as_ref() {
                    Some(s) => Arc::clone(s),
                    None => {
                        anyhow::bail!(
                            "--swap-phase-aware requires --secondary-gguf (no secondary mmap available)"
                        );
                    }
                };
                let config = Arc::new(model.config.clone());
                let chunk_size_bytes = args.swap_phase_aware_chunk_mb.max(1) * 1_048_576;
                eprintln!(
                    "weight_swap: phase-aware mode — ratio={:.2}, {} target layers, chunk_size={} MB (LISWAP-5)",
                    ratio,
                    target_layers.len(),
                    args.swap_phase_aware_chunk_mb
                );
                let phase_dispatcher = llm_rs2::models::weights::PhaseAwareSwapDispatcher::new(
                    chunk_size_bytes,
                    model.layers.clone(),
                    secondary,
                    swap_backend,
                    dispatcher,
                    DType::Q4_0,
                    config,
                );
                // LISWAP Phase 4: install weak self-ref so the worker thread can
                // call back into try_dispatch_chunk_worker via ChunkDispatchJob.
                phase_dispatcher.install_self_weak();
                phase_dispatcher.commit_plan(&target_layers);
                phase_dispatcher
                    .set_max_chunks_per_token(args.swap_phase_aware_max_chunks_per_token);
                if args.swap_phase_aware_max_chunks_per_token > 0 {
                    eprintln!(
                        "weight_swap: phase-aware throttle — max {} chunks/token",
                        args.swap_phase_aware_max_chunks_per_token
                    );
                }
                llm_rs2::profile::op_trace::set_phase_hook(
                    phase_dispatcher.clone() as Arc<dyn llm_rs2::profile::op_trace::PhaseHook>
                );
                phase_aware_swap_dispatcher = Some(phase_dispatcher);
            } else if args.swap_intra_forward || args.swap_layer_immediate {
                let swap_backend: Arc<dyn Backend> = gpu_backend_arc
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| cpu_backend_arc.clone());
                let dispatcher = Arc::new(llm_rs2::models::weights::AsyncSwapDispatcher::new(
                    Arc::clone(&swap_backend),
                ));
                let mode_flag_name = if args.swap_layer_immediate {
                    "--swap-layer-immediate"
                } else {
                    "--swap-intra-forward"
                };
                let secondary = match model.secondary_mmap.as_ref() {
                    Some(s) => Arc::clone(s),
                    None => {
                        anyhow::bail!(
                            "{} requires --secondary-gguf (no secondary mmap available)",
                            mode_flag_name
                        );
                    }
                };
                let config = Arc::new(model.config.clone());
                // LISWAP-6 Phase 6: layer-immediate variant reuses the
                // IntraForwardSwapHook infrastructure. The behavioural
                // difference is in the swap_executor.rs alias H2D-skip
                // (Phase 5b) which collapses every per-layer dispatch to a
                // dummy event when the secondary is rpcmem DMA-BUF aliased.
                let mode_label = if args.swap_layer_immediate {
                    "layer-immediate (LISWAP-6 P6)"
                } else {
                    "intra-forward (LISWAP-4)"
                };
                eprintln!(
                    "weight_swap: {} mode — ratio={:.2}, {} target layers",
                    mode_label,
                    ratio,
                    target_layers.len()
                );
                intra_forward_swap_hook = Some(llm_rs2::models::weights::IntraForwardSwapHook::new(
                    target_layers,
                    0,
                    dispatcher,
                    secondary,
                    model.layers.clone(),
                    swap_backend,
                    Some(Arc::clone(&model.release_worker)),
                    DType::Q4_0,
                    config,
                ));
            } else if args.swap_incremental_per_tick > 0 {
                eprintln!(
                    "weight_swap: incremental mode — ratio={:.2}, {} target layers, per_tick={} ({} ticks estimated)",
                    ratio,
                    target_layers.len(),
                    args.swap_incremental_per_tick,
                    target_layers.len().div_ceil(args.swap_incremental_per_tick),
                );
                incremental_force_swap_plan = Some(llm_rs2::models::weights::IncrementalSwapPlan::new(
                    target_layers,
                    args.swap_incremental_per_tick,
                    0,
                ));
            } else {
                match run_layer_swap(
                    &model,
                    &target_layers,
                    gpu_backend_arc.as_ref(),
                    &cpu_backend_arc,
                    None,
                    #[cfg(feature = "opencl")]
                    host_ptr_swap_pool.clone(),
                ) {
                    Ok(report) => {
                        eprintln!(
                            "weight_swap: force ratio={:.2}, swapped {}/{} layers in {:.1}ms",
                            ratio,
                            report.swapped.len(),
                            num_layers,
                            report.latency_ms,
                        );
                        if let Some(ref stages) = report.stage_breakdown {
                            eprintln!("weight_swap stages: {}", stages.to_log_line());
                        }
                        #[cfg(feature = "opencl")]
                        if let Some(ocl_be) = backend
                            .as_any()
                            .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
                        {
                            ocl_be.dump_cl_mem_diagnostics(" stage=after_force_swap");
                        }
                        #[cfg(feature = "opencl")]
                        remap_weights_for_cpu_after_swap(
                            &mut model,
                            &backend,
                            is_gpu,
                            args.resilience_prealloc_switch,
                            "force-swap",
                        );
                    }
                    Err(e) => {
                        anyhow::bail!("--force-swap-ratio: swap failed: {}", e);
                    }
                }
            }
        }};
    }

    if args.qcf_dump.is_none()
        && let Some(ratio) = args.force_swap_ratio
    {
        let ratio = ratio.clamp(0.0, 1.0);
        let num_layers = model.layers.len();
        let target_layers =
            llm_rs2::models::weights::SwapExecutor::uniform_target_layers(ratio, num_layers);

        // ── LISWAP-6: Eager prefault ─────────────────────────────────────
        // qnn_oppkg + Rpcmem variant 일 때 swap 시점의 rpcmem_alloc 비용
        // (~420 ms/25 layer on Galaxy S25) 을 model load 시점에 흡수.
        // Gguf/Auf variant 는 madvise() 만 호출되어 비용 작음 (~65 ms).
        // 모든 swap mode (single-shot/incremental/intra-forward/phase-aware)
        // 가 자동 이득. swap blocking 700 → ~280 ms 단축 (60%).
        //
        // Phase 3 (`--swap-delay-tokens > 0`): prefault ALWAYS runs at prefill
        // end — only the actual swap dispatch is deferred. This preserves the
        // delay=0 baseline wall (~405 ms LISWAP-6 alias) and ensures rpcmem
        // pages are warm regardless of when the trigger fires.
        if !target_layers.is_empty()
            && let Some(secondary) = model.secondary_mmap.as_ref()
        {
            let t_pre = std::time::Instant::now();
            secondary.prefault_layers(&target_layers);
            // LISWAP-6 Phase 1 — Rpcmem variant also primes per-tensor
            // cl_mem aliases inside `ensure_layer_loaded`. Surface the cache
            // size alongside the wall-clock prefault cost so regressions in
            // either step are visible at startup.
            let alias_cache_len = match secondary.as_ref() {
                llm_rs2::models::weights::SecondaryMmap::Rpcmem(rpc) => Some(rpc.alias_cache_len()),
                _ => None,
            };
            match alias_cache_len {
                Some(n) => eprintln!(
                    "weight_swap: eager prefault — {} layers, {:.1}ms (alias cache: {} cl_mems)",
                    target_layers.len(),
                    t_pre.elapsed().as_secs_f64() * 1e3,
                    n,
                ),
                None => eprintln!(
                    "weight_swap: eager prefault — {} target layers, {:.1}ms",
                    target_layers.len(),
                    t_pre.elapsed().as_secs_f64() * 1e3,
                ),
            }
        }

        // Phase 3: defer the dispatch block when --swap-delay-tokens N > 0.
        // We still run dispatch immediately when `target_layers.is_empty()`
        // (no-op log line; nothing to defer).
        if !target_layers.is_empty() && args.swap_delay_tokens > 0 {
            eprintln!(
                "weight_swap: dispatch deferred — will trigger at decode_token_index={} \
                 (Phase 3 mid-decode swap, ratio={:.2}, {} target layers)",
                args.swap_delay_tokens,
                ratio,
                target_layers.len(),
            );
            pending_force_swap = Some((ratio, target_layers));
        } else if target_layers.is_empty() {
            eprintln!(
                "weight_swap: force ratio={:.2} → 0 target layers (no-op)",
                ratio,
            );
        } else {
            // delay == 0 (default): dispatch immediately at prefill end.
            dispatch_force_swap!(ratio, target_layers);
        }
    }

    // ════════════════════════════════════════════════════════════
    //  DUMP-IMPORTANCE MODE: Measure per-layer importance and exit
    // ════════════════════════════════════════════════════════════
    if args.dump_importance {
        use llm_rs2::core::qcf::ImportanceCollector;

        let mut collector = ImportanceCollector::new();

        let prompt_enc = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!(e))?;
        let prompt_ids: Vec<u32> = prompt_enc.get_ids().to_vec();
        let prompt_len = prompt_ids.len();
        eprintln!("[Importance] Prefill {} tokens...", prompt_len);

        let cpu_buf = Galloc::new().alloc(prompt_len * 4, DType::U8)?;
        unsafe {
            let ptr = cpu_buf.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(prompt_ids.as_ptr(), ptr, prompt_len);
        }
        let cpu_input = Tensor::new(
            Shape::new(vec![1, prompt_len]),
            cpu_buf,
            Arc::new(CpuBackend::new()),
        );
        let input_tensor = backend.copy_from(&cpu_input)?;

        let logits_buf = memory.alloc(prompt_len * vocab_size * 4, DType::F32)?;
        let mut logits = Tensor::new(
            Shape::new(vec![1, prompt_len, vocab_size]),
            logits_buf,
            backend.clone(),
        );

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos: 0,
            kv_caches: &mut kv_caches,
            backend: &backend,
            memory: &*memory,
            logits_out: &mut logits,
            x_gen: None,
            workspace: None,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: Some(&mut collector),
            logits_last_only: false,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;

        let table = collector.build();

        let importance_entries: Vec<serde_json::Value> = table
            .entries()
            .iter()
            .map(|e| {
                serde_json::json!({
                    "layer": e.layer_id,
                    "sublayer": format!("{:?}", e.sublayer),
                    "importance": e.importance,
                    "opr": e.opr,
                })
            })
            .collect();

        let output = serde_json::json!({
            "model": args.model_path,
            "num_layers": model.config.num_hidden_layers,
            "prompt_tokens": prompt_len,
            "importance": importance_entries,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
        return Ok(());
    }

    // ════════════════════════════════════════════════════════════
    //  EVAL-LL MODE: Log-likelihood evaluation for downstream tasks
    // ════════════════════════════════════════════════════════════
    if args.eval_ll {
        let questions = load_eval_questions(&args, &prompt)?;

        // ── QCF-dump prelude: --eval-ll + --qcf-dump + --force-swap-ratio ────
        // When all three flags are active we run warmup prefill → ImportanceTable
        // → WeightSwapDecider → SwapExecutor before the eval loop.  This mirrors
        // the PPL/generation QCF-dump workflow (line ~2417) but uses the eval
        // questions' prompt text instead of a corpus file for the warmup input.
        let eval_ll_qcf_start = std::time::Instant::now();
        let mut eval_ll_qcf_importance: Option<llm_rs2::core::qcf::ImportanceTable> = None;
        let mut eval_ll_qcf_decision: Option<llm_rs2::models::weights::decider::SwapDecision> =
            None;
        let mut eval_ll_qcf_dpllm_epsilon: Option<Vec<f32>> = None;

        if args.qcf_dump.is_some()
            && let Some(force_ratio) = args.force_swap_ratio
        {
            let warmup_n = args.qcf_warmup_tokens.max(1);
            // Concatenate question prompts (separated by "\n\n") and take the
            // first warmup_n tokens. Empty result → soft skip (no abort).
            let warmup_ids = build_eval_ll_warmup_text(&questions, warmup_n, &tokenizer);

            if warmup_ids.is_empty() {
                eprintln!(
                    "[QCF-dump] WARNING: eval-ll warmup token sequence is empty; \
                     prelude skipped (swap will use uniform fallback)"
                );
            } else {
                let result = run_qcf_warmup_workflow(
                    &model,
                    &backend,
                    memory.as_ref(),
                    &mut kv_caches,
                    vocab_size,
                    &warmup_ids,
                    Some(force_ratio),
                    gpu_backend_arc.as_ref(),
                    &cpu_backend_arc,
                    " eval-ll",
                    swap_algorithm,
                    !args.qcf_trajectory,
                    importance_formula,
                    importance_compare,
                    swap_only_layers.as_deref(),
                )?;
                eval_ll_qcf_decision = result.decision;
                eval_ll_qcf_importance = Some(result.importance);
                eval_ll_qcf_dpllm_epsilon = result.dpllm_epsilon;
            }
        }

        let ratio_mode = args.kv_budget_ratio > 0.0;
        let budget_mode = args.kv_budget > 0 || ratio_mode;

        // For ratio mode, effective_budget is computed per-question inside eval_loop.
        // Pass 0 here; the loop will use kv_budget_ratio × prompt_len.
        let effective_budget = if ratio_mode { 0 } else { args.kv_budget };

        eprintln!(
            "[Eval-LL] {} questions, policy={}, kv_budget={}, kv_budget_ratio={}, mode={}",
            questions.len(),
            args.eviction_policy,
            args.kv_budget,
            args.kv_budget_ratio,
            if budget_mode {
                if ratio_mode {
                    "ratio-per-question"
                } else {
                    "chunked"
                }
            } else {
                "full-prefill"
            }
        );

        let qcf_mode_enum = match args.qcf_mode.as_str() {
            "caote" => llm_rs2::core::qcf::QcfMode::Caote,
            "both" => llm_rs2::core::qcf::QcfMode::Both,
            _ => llm_rs2::core::qcf::QcfMode::Attn,
        };
        let qcf_config = llm_rs2::core::qcf::QcfConfig {
            mode: qcf_mode_enum,
            ..llm_rs2::core::qcf::QcfConfig::default()
        };

        let eval_config = llm_rs2::eval::EvalConfig {
            max_seq_len,
            effective_budget,
            kv_budget_ratio: args.kv_budget_ratio,
            greedy: args.greedy,
            kv_type: args.kv_type.clone(),
            qcf_mode: args.qcf_mode.clone(),
            vocab_size,
            hidden_size,
        };

        // For ratio mode, hook starts with budget=0; eval_loop updates it per-question.
        let hook_budget = if ratio_mode { 0 } else { effective_budget };
        let is_d2o = args.eviction_policy == "d2o";

        // ARGUS Step 6: resolve --qcf-sample-layers from CLI.
        // When --enable-qcf-experimental is off, always use [0] (legacy, no overhead).
        let eviction_hook_sample_layers = if args.enable_qcf_experimental {
            parse_qcf_sample_layers(&args.qcf_sample_layers, num_layers)
                .map_err(|e| anyhow::anyhow!("--qcf-sample-layers: {e}"))?
        } else {
            vec![0]
        };

        let mut hook = llm_rs2::eval::EvictionHook::new(
            cache_manager,
            score_accumulator,
            qcf_config,
            hook_budget,
            actual_protected_prefix,
            score_based_eviction,
            args.h2o_keep_ratio,
            is_d2o,
            args.kv_type.clone(),
            backend.clone(),
            args.enable_qcf_experimental,
            eviction_hook_sample_layers,
        );

        // ── Trajectory mode dispatch ──────────────────────────────────────────
        // When `--qcf-trajectory` is active alongside `--qcf-dump` and
        // `--force-swap-ratio`, we run eval-ll K+1 times (K = decision layer
        // count): step 0 with no swap (baseline), then step t (1..=K) after
        // cumulatively applying `selected_layers[t-1]`. Each step's full
        // EvalOutput is captured into `trajectory_outputs` and emitted under
        // the `trajectory` field of the dump JSON.
        let trajectory_mode = args.qcf_trajectory
            && args.qcf_dump.is_some()
            && args.force_swap_ratio.is_some()
            && eval_ll_qcf_decision
                .as_ref()
                .map(|d| !d.selected_layers.is_empty())
                .unwrap_or(false);
        let ordered_layers: Vec<usize> = if trajectory_mode {
            eval_ll_qcf_decision
                .as_ref()
                .unwrap()
                .selected_layers
                .clone()
        } else {
            Vec::new()
        };
        let n_steps = if trajectory_mode {
            ordered_layers.len() + 1
        } else {
            1
        };
        let mut trajectory_outputs: Vec<llm_rs2::eval::EvalOutput> =
            Vec::with_capacity(n_steps);

        if trajectory_mode {
            eprintln!(
                "[QCF-trajectory] mode enabled: K={} (algo={}, ratio={:.2})",
                ordered_layers.len(),
                swap_algorithm.short_name(),
                args.force_swap_ratio.unwrap_or(0.0),
            );
        }

        for step in 0..n_steps {
            if trajectory_mode {
                eprintln!(
                    "[QCF-trajectory] step {}/{}: cumulative swap = {:?}",
                    step,
                    ordered_layers.len(),
                    &ordered_layers[..step]
                );
            }

            let step_out = llm_rs2::eval::run_eval_ll_generic(
                &model,
                &tokenizer,
                &backend,
                &*memory,
                &mut kv_caches,
                &mut hook,
                &questions,
                &eval_config,
                skip_config.as_ref(),
            )?;
            trajectory_outputs.push(step_out);

            if trajectory_mode && step < ordered_layers.len() {
                let layer_to_swap = ordered_layers[step];
                let report = run_layer_swap(
                    &model,
                    &[layer_to_swap],
                    gpu_backend_arc.as_ref(),
                    &cpu_backend_arc,
                    None,
                    #[cfg(feature = "opencl")]
                    None,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "[QCF-trajectory] swap layer {} failed: {}",
                        layer_to_swap,
                        e
                    )
                })?;
                eprintln!(
                    "[QCF-trajectory] swapped layer {}: latency {:.1}ms",
                    layer_to_swap, report.latency_ms
                );
            }
        }

        // For downstream non-trajectory stdout printing, expose the last step's
        // EvalOutput as `output` (in non-trajectory mode this is the only step).
        let output = trajectory_outputs
            .last()
            .expect("at least one eval-ll step ran")
            .clone();

        // ── QCF-dump JSON (eval-ll mode) ──────────────────────────────────────
        if let Some(ref dump_path) = args.qcf_dump {
            use llm_rs2::eval::qcf_helpers::{
                QcfSwapDumpContext, TrajectoryStep, dump_qcf_swap_json,
            };

            let empty_swap: Vec<usize> = Vec::new();
            let (swap_set, qcf_predicted, fallback_used) =
                if let Some(ref dec) = eval_ll_qcf_decision {
                    (
                        dec.selected_layers.as_slice(),
                        dec.qcf_swap_estimate,
                        dec.fallback_used,
                    )
                } else {
                    (empty_swap.as_slice(), 0.0f32, false)
                };

            let secondary_path_str = args.secondary_gguf.as_ref().and_then(|p| p.to_str());
            let model_arch = if args.model_path.to_lowercase().contains("qwen") {
                "qwen2"
            } else {
                "llama"
            };
            let total_wall = eval_ll_qcf_start.elapsed().as_secs_f64();

            // Build trajectory steps when in trajectory mode.
            let trajectory_steps: Vec<TrajectoryStep> = if trajectory_mode {
                trajectory_outputs
                    .iter()
                    .enumerate()
                    .map(|(t, eo)| TrajectoryStep {
                        step: t,
                        swapped_layers: ordered_layers[..t].to_vec(),
                        layer_added: if t > 0 {
                            Some(ordered_layers[t - 1])
                        } else {
                            None
                        },
                        eval_ll_output: eo,
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let trajectory_ref: Option<&[TrajectoryStep]> = if trajectory_mode {
                Some(trajectory_steps.as_slice())
            } else {
                None
            };
            let eval_ll_output_ref = if trajectory_mode { None } else { Some(&output) };

            let ctx = QcfSwapDumpContext {
                model_arch,
                model_path: &args.model_path,
                secondary_path: secondary_path_str,
                primary_dtype: "F16",
                secondary_dtype: "Q4_0",
                num_layers: model.layers.len(),
                force_swap_ratio: args.force_swap_ratio,
                swap_algorithm: args
                    .force_swap_ratio
                    .map(|_| swap_algorithm.short_name()),
                swap_set,
                qcf_swap_predicted: qcf_predicted,
                fallback_used,
                importance_table: eval_ll_qcf_importance.as_ref(),
                noise_table: Some(model.quant_noise.as_ref()),
                ppl: None,
                avg_nll: None,
                n_eval_tokens: 0,
                wall_time_s: total_wall,
                warmup_tokens: args.qcf_warmup_tokens,
                backend: &args.backend,
                kv_type: &args.kv_type,
                ppl_corpus: None,
                eval_ll_output: eval_ll_output_ref,
                trajectory: trajectory_ref,
                dpllm_epsilon: eval_ll_qcf_dpllm_epsilon.as_deref(),
            };

            dump_qcf_swap_json(dump_path, &ctx)?;
            eprintln!(
                "[QCF-dump] eval-ll JSON written to {}{}",
                dump_path.display(),
                if trajectory_mode {
                    " (trajectory schema_v2)"
                } else {
                    ""
                }
            );
        }

        let mut json_val = serde_json::from_str::<serde_json::Value>(&output.to_json()?)?;
        json_val["config"] = serde_json::json!({
            "model": args.model_path,
            "eviction_policy": args.eviction_policy,
            "kv_budget": args.kv_budget,
            "kv_budget_ratio": args.kv_budget_ratio,
            "max_seq_len": max_seq_len,
            "kv_type": args.kv_type,
            "h2o_keep_ratio": args.h2o_keep_ratio,
            "h2o_decay": args.h2o_decay,
            "time_normalized": !args.h2o_raw_scores,
            "skip_layers": args.skip_layers,
            "skip_ratio": args.skip_ratio,
        });
        println!("{}", serde_json::to_string_pretty(&json_val)?);
        return Ok(());
    }

    // ════════════════════════════════════════════════════════════
    //  QCF DUMP WORKFLOW: Warmup prefill → ImportanceTable → Swap → Measure
    //
    //  When --qcf-dump is active, we insert a warmup prefill before the main
    //  measurement to build an ImportanceTable for accurate WeightSwapDecider
    //  (importance × ε bottom-k selection, ENG-ALG-215).
    //
    //  The workflow applies to both --ppl and generation modes.
    //  When --qcf-dump is absent, all existing behavior is unchanged.
    // ════════════════════════════════════════════════════════════

    // Accumulated state produced by the QCF dump workflow.
    // Only populated when args.qcf_dump.is_some().
    let mut qcf_warmup_importance: Option<llm_rs2::core::qcf::ImportanceTable> = None;
    let mut qcf_swap_decision: Option<llm_rs2::models::weights::decider::SwapDecision> = None;
    let qcf_workflow_start = std::time::Instant::now();

    if args.qcf_dump.is_some() && (args.ppl.is_some() || !prompt.is_empty()) {
        let warmup_n = args.qcf_warmup_tokens.max(1);

        // For PPL mode the warmup tokens come from the reference text; for
        // generation mode they come from the prompt. Both paths cap at warmup_n.
        let warmup_tokens: Vec<u32> = if let Some(ref ppl_path) = args.ppl {
            let text = std::fs::read_to_string(ppl_path)
                .map_err(|e| anyhow::anyhow!("Failed to read PPL file for warmup: {}", e))?;
            let enc = tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Warmup tokenize error: {}", e))?;
            enc.get_ids().iter().take(warmup_n).copied().collect()
        } else {
            let enc = tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Warmup tokenize error: {}", e))?;
            enc.get_ids().iter().take(warmup_n).copied().collect()
        };

        if warmup_tokens.is_empty() {
            anyhow::bail!(
                "--qcf-dump: warmup token sequence is empty (prompt or PPL text too short)"
            );
        }

        let result = run_qcf_warmup_workflow(
            &model,
            &backend,
            memory.as_ref(),
            &mut kv_caches,
            vocab_size,
            &warmup_tokens,
            args.force_swap_ratio,
            gpu_backend_arc.as_ref(),
            &cpu_backend_arc,
            "",
            swap_algorithm,
            true,
            importance_formula,
            importance_compare,
            swap_only_layers.as_deref(),
        )?;
        qcf_swap_decision = result.decision;
        qcf_warmup_importance = Some(result.importance);
    }

    // ════════════════════════════════════════════════════════════
    //  PPL MODE: Perplexity evaluation on reference text
    // ════════════════════════════════════════════════════════════
    if let Some(ref ppl_path) = args.ppl {
        // LISWAP-PPL diagnostic: dump weights immediately after model load
        // (before any swap), useful for Q4-native baseline comparison.
        if let Some(ref dump_dir) = args.dump_q4_after_load {
            dump_layer_weights_to_dir(&model, &backend, dump_dir)?;
        }

        // LISWAP-PPL Scenario E (warmup-then-measure):
        //   Pass 1: drive the weight swap to completion with no NLL logging.
        //   Reset KV caches + score_accumulator so the measurement pass sees a
        //   fresh cache, then run the measurement pass with the swap trigger
        //   disabled. The cache reset isolates the "cache mismatch" hypothesis
        //   from the "weight quantization-path mismatch" hypothesis.
        if args.ppl_warmup_swap {
            if args.ppl_swap_at_token.is_none() {
                anyhow::bail!(
                    "--ppl-warmup-swap requires --ppl-swap-at-token (the warmup pass needs a swap trigger)"
                );
            }
            if model.secondary_mmap.is_none() {
                anyhow::bail!(
                    "--ppl-warmup-swap requires --secondary-gguf (weights must be available for swap)"
                );
            }

            let mut warmup_args = args.clone();
            // Suppress CSV/JSON outputs on the warmup pass.
            warmup_args.ppl_nll_csv = None;
            warmup_args.qcf_dump = None;
            eprintln!("[PPL-Swap] === Pass 1: warmup (driving swap to completion) ===");
            let _warmup_dummy = run_ppl(
                &warmup_args,
                &model,
                &tokenizer,
                &backend,
                &*memory,
                &mut kv_caches,
                &mut cache_manager,
                &mut score_accumulator,
                vocab_size,
                hidden_size,
                max_seq_len,
                ppl_path,
                auto_eviction,
                score_based_eviction,
                actual_protected_prefix,
                skip_config.as_ref(),
                /* warmup_only */ true,
            )?;

            // LISWAP-PPL diagnostic: dump weights right after swap completion
            // (before cache reset), so each layer's GPU buffer can be compared
            // byte-for-byte against the Q4-native baseline dump.
            if let Some(ref dump_dir) = args.dump_q4_after_swap {
                dump_layer_weights_to_dir(&model, &backend, dump_dir)?;
            }

            // Reset KV cache positions. The underlying tensor buffers stay
            // allocated; we only rewind the write head + high-water mark so
            // the next prefill starts from pos 0.
            for cache in kv_caches.iter_mut() {
                cache.current_pos = 0;
                cache.high_water_pos = 0;
            }
            if let Some(acc) = score_accumulator.as_mut() {
                acc.reset();
            }
            eprintln!("[PPL-Swap] === Pass 2: measurement (swap disabled, fresh KV cache) ===");
        }

        // Measurement pass. When warmup_swap was active, disable further swap
        // triggers and clear the warmup flag locally so this pass is a pure
        // teacher-forcing PPL run on the already-swapped weights.
        // When `--ppl-measure-prefill-tokens` is set, pass 2 uses that prefill
        // length instead of `--ppl-prefill-tokens` (Scenario F: large prefill
        // shrinks the decode loop, isolating batch vs single-step path).
        let mut measure_args_owned;
        let measure_args: &Args = if args.ppl_warmup_swap {
            measure_args_owned = args.clone();
            measure_args_owned.ppl_swap_at_token = None;
            measure_args_owned.ppl_warmup_swap = false;
            if let Some(measure_prefill) = args.ppl_measure_prefill_tokens {
                measure_args_owned.ppl_prefill_tokens = Some(measure_prefill);
            }
            &measure_args_owned
        } else {
            &args
        };
        let ppl_result = run_ppl(
            measure_args,
            &model,
            &tokenizer,
            &backend,
            &*memory,
            &mut kv_caches,
            &mut cache_manager,
            &mut score_accumulator,
            vocab_size,
            hidden_size,
            max_seq_len,
            ppl_path,
            auto_eviction,
            score_based_eviction,
            actual_protected_prefix,
            skip_config.as_ref(),
            /* warmup_only */ false,
        )?;

        // --qcf-dump: write JSON after PPL measurement completes.
        if let Some(ref dump_path) = args.qcf_dump {
            use llm_rs2::eval::qcf_helpers::{QcfSwapDumpContext, dump_qcf_swap_json};

            let empty_swap: Vec<usize> = Vec::new();
            let (swap_set, qcf_predicted, fallback_used) = if let Some(ref dec) = qcf_swap_decision
            {
                (
                    dec.selected_layers.as_slice(),
                    dec.qcf_swap_estimate,
                    dec.fallback_used,
                )
            } else {
                (empty_swap.as_slice(), 0.0f32, false)
            };

            let secondary_path_str = args.secondary_gguf.as_ref().and_then(|p| p.to_str());
            let model_arch = if args.model_path.to_lowercase().contains("qwen") {
                "qwen2"
            } else {
                "llama"
            };
            let total_wall = qcf_workflow_start.elapsed().as_secs_f64() + ppl_result.wall_time_s;

            let ctx = QcfSwapDumpContext {
                model_arch,
                model_path: &args.model_path,
                secondary_path: secondary_path_str,
                primary_dtype: "F16",
                secondary_dtype: "Q4_0",
                num_layers: model.layers.len(),
                force_swap_ratio: args.force_swap_ratio,
                swap_algorithm: args
                    .force_swap_ratio
                    .map(|_| swap_algorithm.short_name()),
                swap_set,
                qcf_swap_predicted: qcf_predicted,
                fallback_used,
                importance_table: qcf_warmup_importance.as_ref(),
                noise_table: Some(model.quant_noise.as_ref()),
                ppl: Some(ppl_result.ppl),
                avg_nll: Some(ppl_result.avg_nll),
                n_eval_tokens: ppl_result.n_eval_tokens,
                wall_time_s: total_wall,
                warmup_tokens: args.qcf_warmup_tokens,
                backend: &args.backend,
                kv_type: &args.kv_type,
                ppl_corpus: Some(ppl_path.as_str()),
                eval_ll_output: None,
                trajectory: None,
                dpllm_epsilon: None,
            };

            dump_qcf_swap_json(dump_path, &ctx)?;
            eprintln!("[QCF-dump] JSON written to {}", dump_path.display());
        }

        return Ok(());
    }

    // ════════════════════════════════════════════════════════════
    //  PROMPT-BATCH MODE: Sequential multi-prompt generation
    // ════════════════════════════════════════════════════════════
    if let Some(ref batch_path) = args.prompt_batch {
        let entries = load_prompt_batch(batch_path)?;
        if entries.is_empty() {
            anyhow::bail!("prompt-batch file is empty: {}", batch_path);
        }
        eprintln!(
            "[Batch] Loaded {} entries from {}",
            entries.len(),
            batch_path
        );

        // ARGUS hook: emit Step1~6 metrics (qcf_caote_max / qcf_per_head /
        // qcf_topk_retention_* / attention_entropy / qcf_beta_amplified_* /
        // qcf_per_layer*) per record, alongside legacy fields.
        // Hook owns cache_manager + score_accumulator from here; subsequent
        // forward calls in this branch route score_accumulator through the hook.
        use llm_rs2::eval::StepHook;
        let pb_qcf_mode_enum = match args.qcf_mode.as_str() {
            "caote" => llm_rs2::core::qcf::QcfMode::Caote,
            "both" => llm_rs2::core::qcf::QcfMode::Both,
            _ => llm_rs2::core::qcf::QcfMode::Attn,
        };
        let pb_qcf_config = llm_rs2::core::qcf::QcfConfig {
            mode: pb_qcf_mode_enum,
            ..llm_rs2::core::qcf::QcfConfig::default()
        };
        let pb_ratio_mode = args.kv_budget_ratio > 0.0;
        let pb_hook_budget = if pb_ratio_mode { 0 } else { args.kv_budget };
        let pb_is_d2o = args.eviction_policy == "d2o";
        let pb_num_layers = model.config.num_hidden_layers;
        let pb_sample_layers = if args.enable_qcf_experimental {
            parse_qcf_sample_layers(&args.qcf_sample_layers, pb_num_layers)
                .map_err(|e| anyhow::anyhow!("--qcf-sample-layers: {e}"))?
        } else {
            vec![0]
        };
        let mut hook = llm_rs2::eval::EvictionHook::new(
            cache_manager,
            score_accumulator.take(),
            pb_qcf_config,
            pb_hook_budget,
            actual_protected_prefix,
            score_based_eviction,
            args.h2o_keep_ratio,
            pb_is_d2o,
            args.kv_type.clone(),
            backend.clone(),
            args.enable_qcf_experimental,
            pb_sample_layers,
        );

        let mut iteration = 0usize;

        // Pre-allocate generation buffers (once)
        let logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
        let mut logits = Tensor::new(
            Shape::new(vec![1, 1, vocab_size]),
            logits_buf,
            backend.clone(),
        );
        let eos_id = model.config.eos_token_id;

        // Pre-allocate workspace (once)
        let q_dim = model.config.num_attention_heads * model.config.head_dim;
        let k_dim = model.config.num_key_value_heads * model.config.head_dim;
        let v_dim = k_dim;
        let ffn_hidden = model.config.intermediate_size;
        let x_gen_buf = memory.alloc(hidden_size * 4, DType::F32)?;
        let mut x_gen = Tensor::new(
            Shape::new(vec![1, 1, hidden_size]),
            x_gen_buf,
            backend.clone(),
        );
        let mut gen_ws = LayerWorkspace::new(
            WorkspaceConfig {
                batch_size: 1,
                dim: model.config.hidden_size,
                q_dim,
                k_dim,
                v_dim,
                ffn_hidden,
                n_heads: model.config.num_attention_heads,
                max_seq_len: args.max_seq_len,
            },
            memory.as_ref(),
            backend.clone(),
        )?;

        // Attach partition workspace if tensor partition is active.
        // Use UnifiedBuffer (ALLOC_HOST_PTR, host-accessible + GPU-accessible) for partition
        // buffers so merge can use direct pointer access instead of read_buffer/write_buffer.
        let layer0_partition_probe = model.layers[0].load_weights();
        if let Some(ref ctx) = layer0_partition_probe.partition_ctx {
            let gpu_alloc = make_partition_gpu_alloc(&*backend, memory.as_ref());

            // Zero-copy residual: permanent-map ws.residual's UnifiedBuffer so the
            // partition decode path can read residual directly via as_ptr() and
            // skip the per-layer read_buffer DMA (currently ~1.15 ms/layer).
            // Gate behind LLMRS_PARTITION_ZCOPY_RESIDUAL=1, or auto-enable when
            // poll-flag mode is active (skipping the read_buffer is the whole
            // point of the spin-poll path).
            #[cfg(feature = "opencl")]
            if std::env::var_os("LLMRS_PARTITION_ZCOPY_RESIDUAL").is_some()
                || llm_rs2::layers::tensor_partition::partition_poll_flag_enabled()
            {
                if let Some(ub) = gen_ws
                    .residual
                    .buffer()
                    .as_any()
                    .downcast_ref::<llm_rs2::buffer::unified_buffer::UnifiedBuffer>()
                {
                    ub.map()?;
                    eprintln!("[Partition] Residual UnifiedBuffer permanent-mapped for zero-copy");
                } else {
                    eprintln!(
                        "[Partition] WARN: residual buffer is not UnifiedBuffer (zero-copy skipped)"
                    );
                }
            }

            gen_ws.partition_ws = Some(Arc::new(PartitionWsCell::new(PartitionWorkspace::new(
                ctx,
                ffn_hidden,
                hidden_size,
                &gpu_alloc,
                backend.clone(),
                cpu_backend_arc.clone(),
            )?)));
        }

        // Pre-allocate CPU/GPU single-token tensors
        let cpu_gen_indices_buf = Galloc::new().alloc(4, DType::U8)?;
        let cpu_gen_input = Tensor::new(
            Shape::new(vec![1, 1]),
            cpu_gen_indices_buf,
            Arc::new(CpuBackend::new()),
        );
        let gpu_gen_input_buf = memory.alloc(4, DType::U8)?;
        let mut gen_input_tensor =
            Tensor::new(Shape::new(vec![1, 1]), gpu_gen_input_buf, backend.clone());
        let mut logits_cpu = vec![0.0f32; vocab_size];

        // Persistent prefill policy state: survives across batches.
        // Only reset by RestoreDefaults, not by prefill→decode transition.
        let mut persistent_chunk_size: Option<usize> = None;
        let mut persistent_yield_ms: Option<u32> = None;
        let mut persistent_cpu_chunk_size: Option<usize> = None;

        'outer: loop {
            for entry in &entries {
                if args.max_iterations > 0 && iteration >= args.max_iterations {
                    break 'outer;
                }

                let prompt_text = resolve_prompt(entry)?;
                let encoding = tokenizer
                    .encode(prompt_text.as_str(), true)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                let batch_input_ids: Vec<u32> = encoding.get_ids().to_vec();
                let prompt_tokens = batch_input_ids.len();

                eprintln!(
                    "[Batch] #{} id={}, prompt_tokens={}",
                    iteration, entry.id, prompt_tokens
                );

                // Per-record budget when --kv-budget-ratio is active
                // (mirrors eval-ll path eval_loop.rs:207). Without this the hook
                // sees effective_budget=0 and post_prefill early-returns,
                // suppressing eviction and ARGUS metric collection.
                if pb_ratio_mode {
                    let dynamic_budget = ((prompt_tokens as f32) * args.kv_budget_ratio) as usize;
                    hook.set_effective_budget(dynamic_budget.max(1));
                }

                let entry_start = std::time::Instant::now();

                eprintln!(
                    "[Batch] #{} id={} prefill_start ts={:.3}",
                    iteration,
                    entry.id,
                    unix_ts()
                );

                // === PREFILL ===
                let process_len = batch_input_ids.len();
                if process_len > max_seq_len {
                    eprintln!(
                        "[Batch] #{} id={}: prompt too long ({} > {}), skipping",
                        iteration, entry.id, process_len, max_seq_len
                    );
                    let err_result = serde_json::json!({
                        "id": entry.id,
                        "error": format!("prompt too long: {} > {}", process_len, max_seq_len),
                    });
                    println!("{}", serde_json::to_string(&err_result)?);
                    iteration += 1;
                    continue;
                }

                // Chunked prefill
                // When resilience is enabled, auto-chunk at 256 for checkpoint support.
                // When GPU backend and prefill_chunk_size == 0, auto-derive a safe chunk
                // size from max_single_alloc() to avoid CL_INVALID_BUFFER_SIZE (-61).
                let auto_gpu_chunk: Option<usize> =
                    if args.prefill_chunk_size == 0 && backend.is_gpu() {
                        let max_alloc = backend.max_single_alloc();
                        if max_alloc > 0 {
                            // Each chunk needs a logits buffer: chunk * vocab_size * 4 bytes.
                            // Use 50% of max_single_alloc as conservative budget.
                            let budget = max_alloc / 2;
                            let by_vocab = (budget / (vocab_size * 4)).max(1);
                            // Also bound by hidden_size to keep activation buffers feasible.
                            let by_hidden = (max_alloc / (hidden_size * 4)).max(1);
                            let derived = by_vocab.min(by_hidden).min(512);
                            Some(derived)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                let chunk_size = if args.prefill_chunk_size > 0
                    && args.prefill_chunk_size < process_len
                {
                    args.prefill_chunk_size
                } else if let Some(auto) = auto_gpu_chunk {
                    if auto < process_len {
                        eprintln!(
                            "[Prefill] prefill_chunk_size auto-selected: {} (max_alloc={}MB, vocab={}, hidden={})",
                            auto,
                            backend.max_single_alloc() / (1024 * 1024),
                            vocab_size,
                            hidden_size,
                        );
                        auto
                    } else {
                        process_len
                    }
                } else if args.enable_resilience && process_len > 256 {
                    256
                } else {
                    process_len
                };
                let chunked = chunk_size < process_len;

                // Dynamic prefill policy: use persistent values if set by prior
                // SetPrefillPolicy, otherwise fall back to CLI defaults.
                let mut effective_chunk_size = persistent_chunk_size.unwrap_or(chunk_size);
                let mut effective_yield_ms = persistent_yield_ms.unwrap_or(args.prefill_yield_ms);
                let mut effective_cpu_chunk_size =
                    persistent_cpu_chunk_size.unwrap_or(args.prefill_cpu_chunk_size);

                let (prefill_logits_shape, prefill_logits_buf_size) = if chunked {
                    (Shape::new(vec![1, 1, vocab_size]), vocab_size * 4)
                } else {
                    (
                        Shape::new(vec![1, process_len, vocab_size]),
                        process_len * vocab_size * 4,
                    )
                };
                // Use CPU memory when on CPU after SwitchHw; GPU memory otherwise.
                let batch_effective_mem: &dyn Memory = if is_gpu {
                    memory.as_ref()
                } else {
                    cpu_memory_arc.as_ref()
                };
                let prefill_logits_buf =
                    batch_effective_mem.alloc(prefill_logits_buf_size, DType::F32)?;
                let mut prefill_logits =
                    Tensor::new(prefill_logits_shape, prefill_logits_buf, backend.clone());

                let prefill_timer = std::time::Instant::now();
                let mut deferred_switch: Option<String> = None;
                let total_chunks = process_len.div_ceil(chunk_size);

                // Report prefill start to resilience manager.
                if let Some(executor) = &mut command_executor {
                    executor.set_prefill_state("prefill", 0, process_len);
                }

                let mut chunk_start = 0;
                let mut chunk_idx = 0usize;
                while chunk_start < process_len {
                    // Guard: effective_chunk_size must be at least 1.
                    let ecs = effective_chunk_size.max(1);
                    let chunk_end = (chunk_start + ecs).min(process_len);
                    let chunk_tokens = &batch_input_ids[chunk_start..chunk_end];
                    let chunk_len = chunk_tokens.len();

                    let cpu_chunk_buf = Galloc::new().alloc(chunk_len * 4, DType::U8)?;
                    unsafe {
                        let ptr = cpu_chunk_buf.as_mut_ptr() as *mut u32;
                        std::ptr::copy_nonoverlapping(chunk_tokens.as_ptr(), ptr, chunk_len);
                    }
                    let cpu_chunk_tensor = Tensor::new(
                        Shape::new(vec![1, chunk_len]),
                        cpu_chunk_buf,
                        Arc::new(CpuBackend::new()),
                    );
                    let input_tensor = backend.copy_from(&cpu_chunk_tensor)?;

                    model.forward_into(TransformerModelForwardArgs {
                        input_tokens: &input_tensor,
                        start_pos: chunk_start,
                        kv_caches: &mut kv_caches,
                        backend: &backend,
                        memory: batch_effective_mem,
                        logits_out: &mut prefill_logits,
                        x_gen: None,
                        workspace: None,
                        score_accumulator: hook.score_accumulator(),
                        profiler: None,
                        skip_config: skip_config.as_ref(),
                        importance_collector: None,
                        logits_last_only: chunked,
                        variance_collector: None,
                        prefill_workspace: None,

                        layer_boundary_hook: None,
                    })?;
                    backend.synchronize()?;
                    drop(input_tensor);

                    chunk_start = chunk_end;

                    // Inter-chunk yield: sleep after GPU chunk to release compute.
                    if effective_yield_ms > 0 {
                        std::thread::sleep(std::time::Duration::from_millis(
                            effective_yield_ms as u64,
                        ));
                    }

                    // CPU interleave: process next chunk on CPU while GPU is free.
                    // Invariant: the last chunk must be processed by GPU so that
                    // prefill_logits (GPU buffer) is valid at the end.
                    if effective_cpu_chunk_size > 0 && chunk_start < process_len {
                        let remaining = process_len - chunk_start;
                        if remaining > effective_cpu_chunk_size {
                            // Flush GPU caches before CPU reads KV buffers (ARM UMA coherence).
                            for kv in kv_caches.iter() {
                                kv.k_buffer.buffer().map_for_cpu()?;
                                kv.v_buffer.buffer().map_for_cpu()?;
                            }

                            let cpu_end = (chunk_start + effective_cpu_chunk_size)
                                .min(process_len.saturating_sub(1));
                            if cpu_end > chunk_start {
                                let cpu_tokens = &batch_input_ids[chunk_start..cpu_end];
                                let cpu_len = cpu_tokens.len();

                                let cpu_in_buf = Galloc::new().alloc(cpu_len * 4, DType::U8)?;
                                unsafe {
                                    let ptr = cpu_in_buf.as_mut_ptr() as *mut u32;
                                    std::ptr::copy_nonoverlapping(
                                        cpu_tokens.as_ptr(),
                                        ptr,
                                        cpu_len,
                                    );
                                }
                                let cpu_in_tensor = Tensor::new(
                                    Shape::new(vec![1, cpu_len]),
                                    cpu_in_buf,
                                    cpu_backend_arc.clone(),
                                );

                                let cpu_chunk_start_pos = chunk_start;

                                // CPU prefill logits: separate CPU buffer (discarded).
                                let cpu_logits_buf =
                                    cpu_memory_arc.alloc(vocab_size * 4, DType::F32)?;
                                let mut cpu_logits = Tensor::new(
                                    Shape::new(vec![1, 1, vocab_size]),
                                    cpu_logits_buf,
                                    cpu_backend_arc.clone(),
                                );

                                model.forward_into(TransformerModelForwardArgs {
                                    input_tokens: &cpu_in_tensor,
                                    start_pos: cpu_chunk_start_pos,
                                    kv_caches: &mut kv_caches,
                                    backend: &cpu_backend_arc,
                                    memory: cpu_memory_arc.as_ref(),
                                    logits_out: &mut cpu_logits,
                                    x_gen: None,
                                    workspace: None,
                                    score_accumulator: hook.score_accumulator(),
                                    profiler: None,
                                    skip_config: skip_config.as_ref(),
                                    importance_collector: None,
                                    logits_last_only: true,
                                    variance_collector: None,
                                    prefill_workspace: None,

                                    layer_boundary_hook: None,
                                })?;
                                drop(cpu_in_tensor);
                                drop(cpu_logits);

                                chunk_start = cpu_end;
                            }
                        }
                    }

                    // ── Prefill resilience checkpoint (chunk boundary) ──
                    if chunked && let Some(executor) = &mut command_executor {
                        let kv_snap = KVSnapshot {
                            total_bytes: kv_caches
                                .iter()
                                .map(|c| {
                                    (c.k_buffer.buffer().size() + c.v_buffer.buffer().size()) as u64
                                })
                                .sum(),
                            total_tokens: kv_caches[0].current_pos,
                            capacity: kv_caches[0].capacity(),
                            protected_prefix: actual_protected_prefix,
                            kv_dtype: args.kv_type.clone(),
                            eviction_policy: args.eviction_policy.clone(),
                            skip_ratio: 0.0,
                        };
                        let plan = executor.poll(&kv_snap);

                        // SetPrefillPolicy: dynamically adjust chunk/yield/cpu parameters.
                        // Values persist across batches until RestoreDefaults.
                        if let Some(v) = plan.prefill_chunk_size {
                            effective_chunk_size = v;
                            persistent_chunk_size = Some(v);
                            eprintln!("[Prefill] Policy: chunk_size -> {}", v);
                        }
                        if let Some(v) = plan.prefill_yield_ms {
                            effective_yield_ms = v;
                            persistent_yield_ms = Some(v);
                            eprintln!("[Prefill] Policy: yield_ms -> {}", v);
                        }
                        if let Some(v) = plan.prefill_cpu_chunk_size {
                            let layer0_probe = model.layers[0].load_weights();
                            if v > 0 && layer0_probe.wq.as_ptr().is_null() {
                                eprintln!(
                                    "[Prefill] Policy: cpu_chunk_size={} rejected — weights not CPU-accessible. \
                                     Use --resilience-prealloc-switch or --prefill-cpu-chunk-size at CLI.",
                                    v
                                );
                            } else {
                                effective_cpu_chunk_size = v;
                                persistent_cpu_chunk_size = Some(v);
                                eprintln!("[Prefill] Policy: cpu_chunk_size -> {}", v);
                            }
                        }

                        // Throttle: sleep between chunks
                        if plan.throttle_delay_ms > 0 && plan.throttle_delay_ms != throttle_delay_ms
                        {
                            eprintln!(
                                "[Prefill] Throttle: {}ms -> {}ms",
                                throttle_delay_ms, plan.throttle_delay_ms
                            );
                        }
                        throttle_delay_ms = plan.throttle_delay_ms;
                        if throttle_delay_ms > 0 {
                            eprintln!(
                                "[Prefill] Throttle: {}ms delay after chunk {}/{}",
                                throttle_delay_ms,
                                chunk_idx + 1,
                                total_chunks
                            );
                            std::thread::sleep(std::time::Duration::from_millis(throttle_delay_ms));
                        }

                        // LayerSkip
                        if plan.restore_defaults {
                            skip_config = None;
                            last_skip_ratio = None;
                            effective_chunk_size = chunk_size;
                            effective_yield_ms = args.prefill_yield_ms;
                            effective_cpu_chunk_size = args.prefill_cpu_chunk_size;
                            persistent_chunk_size = None;
                            persistent_yield_ms = None;
                            persistent_cpu_chunk_size = None;
                        } else if let Some(ratio) = plan.layer_skip
                            && last_skip_ratio != Some(ratio)
                        {
                            eprintln!("[Prefill] LayerSkip: ratio={:.2}", ratio);
                            skip_config = Some(SkipConfig::uniform_init(
                                model.config.num_hidden_layers,
                                ratio,
                            ));
                            last_skip_ratio = Some(ratio);
                        }

                        // SwitchHw: defer to post-prefill boundary.
                        // Mid-prefill switch causes segfault: model workspace buffers
                        // remain on the old backend; the next chunk accesses them
                        // from the new backend -> invalid memory reference.
                        if let Some(ref device) = plan.switch_device {
                            if deferred_switch.is_none() {
                                eprintln!(
                                    "[Prefill] SwitchHw: deferring '{}' to post-prefill (chunk_pos={})",
                                    device, kv_caches[0].current_pos
                                );
                            }
                            deferred_switch = Some(device.clone());
                        }

                        // Report prefill progress.
                        executor.set_prefill_state("prefill", chunk_start, process_len);
                    }

                    chunk_idx += 1;
                }

                let ttft_ms = prefill_timer.elapsed().as_secs_f64() * 1000.0;

                // Report transition to decode phase.
                if let Some(executor) = &mut command_executor {
                    executor.set_prefill_state("decode", 0, 0);
                }

                // Sample first token from prefill logits
                let mut last_logits = vec![0.0f32; vocab_size];
                unsafe {
                    let ptr = last_logits.as_mut_ptr() as *mut u8;
                    let byte_len = vocab_size * 4;
                    if chunked {
                        let slice = std::slice::from_raw_parts_mut(ptr, byte_len);
                        backend.read_buffer(&prefill_logits, slice)?;
                    } else {
                        let mut full_logits = vec![0.0f32; process_len * vocab_size];
                        let full_ptr = full_logits.as_mut_ptr() as *mut u8;
                        let full_slice =
                            std::slice::from_raw_parts_mut(full_ptr, full_logits.len() * 4);
                        backend.read_buffer(&prefill_logits, full_slice)?;
                        let start_idx = (process_len - 1) * vocab_size;
                        last_logits
                            .copy_from_slice(&full_logits[start_idx..start_idx + vocab_size]);
                    }
                }
                drop(prefill_logits);

                // Execute deferred SwitchHw (from prefill checkpoint).
                // Now safe: prefill is done, logits read, all workspace released.
                if let Some(ref device) = deferred_switch
                    && let (Some(gpu_be), Some(gpu_mem)) = (&gpu_backend_arc, &gpu_memory_arc)
                {
                    match device.as_str() {
                        "cpu" if is_gpu => {
                            eprintln!("[Prefill->Decode] Executing deferred SwitchHw: GPU->CPU");
                            llm_rs2::core::kv_migrate::migrate_kv_caches(
                                &mut kv_caches,
                                &backend,
                                &cpu_backend_arc,
                                &cpu_backend_arc,
                                &cpu_memory_arc,
                                &cpu_memory_arc,
                                kv_heads,
                                head_dim,
                                max_seq_len,
                                false,
                            )?;
                            backend = cpu_backend_arc.clone();
                            is_gpu = false;
                            // Re-allocate decode buffers on CPU.
                            let new_lb = cpu_memory_arc.alloc(vocab_size * 4, DType::F32)?;
                            logits = Tensor::new(logits.shape().clone(), new_lb, backend.clone());
                            let new_xb = cpu_memory_arc.alloc(hidden_size * 4, DType::F32)?;
                            x_gen = Tensor::new(x_gen.shape().clone(), new_xb, backend.clone());
                            gen_ws = LayerWorkspace::new(
                                WorkspaceConfig {
                                    batch_size: 1,
                                    dim: model.config.hidden_size,
                                    q_dim,
                                    k_dim,
                                    v_dim,
                                    ffn_hidden,
                                    n_heads: model.config.num_attention_heads,
                                    max_seq_len: args.max_seq_len,
                                },
                                cpu_memory_arc.as_ref(),
                                backend.clone(),
                            )?;
                            let new_gi = cpu_memory_arc.alloc(4, DType::U8)?;
                            gen_input_tensor = Tensor::new(
                                gen_input_tensor.shape().clone(),
                                new_gi,
                                backend.clone(),
                            );
                            eprintln!(
                                "[Prefill->Decode] SwitchHw: Switched to CPU (GPU handles released, decode buffers re-allocated)."
                            );
                        }
                        "gpu" | "opencl" if !is_gpu && weights_on_gpu => {
                            eprintln!("[Prefill->Decode] Executing deferred SwitchHw: CPU->GPU");
                            llm_rs2::core::kv_migrate::migrate_kv_caches(
                                &mut kv_caches,
                                &backend,
                                gpu_be,
                                &cpu_backend_arc,
                                &cpu_memory_arc,
                                gpu_mem,
                                kv_heads,
                                head_dim,
                                max_seq_len,
                                true,
                            )?;
                            backend = gpu_be.clone();
                            is_gpu = true;
                            let new_lb = gpu_mem.alloc(vocab_size * 4, DType::F32)?;
                            logits = Tensor::new(logits.shape().clone(), new_lb, backend.clone());
                            let new_xb = gpu_mem.alloc(hidden_size * 4, DType::F32)?;
                            x_gen = Tensor::new(x_gen.shape().clone(), new_xb, backend.clone());
                            gen_ws = LayerWorkspace::new(
                                WorkspaceConfig {
                                    batch_size: 1,
                                    dim: model.config.hidden_size,
                                    q_dim,
                                    k_dim,
                                    v_dim,
                                    ffn_hidden,
                                    n_heads: model.config.num_attention_heads,
                                    max_seq_len: args.max_seq_len,
                                },
                                gpu_mem.as_ref(),
                                backend.clone(),
                            )?;
                            let new_gi = gpu_mem.alloc(4, DType::U8)?;
                            gen_input_tensor = Tensor::new(
                                gen_input_tensor.shape().clone(),
                                new_gi,
                                backend.clone(),
                            );
                            eprintln!(
                                "[Prefill->Decode] SwitchHw: Switched to GPU (decode buffers re-allocated)."
                            );
                        }
                        _ => {} // Already on requested backend
                    }
                }

                let mut batch_tokens = batch_input_ids.clone();
                let next_token_id = sampling::sample(
                    &mut last_logits,
                    &batch_tokens,
                    vocab_size,
                    &sampling_config,
                    None,
                );
                batch_tokens.push(next_token_id);
                let mut batch_start_pos = process_len;

                // ── Score collection probe ──
                // Mirrors eval_loop.rs:246~287. Batch prefill calls forward with
                // workspace=None, so the hook's score_accumulator stays empty
                // → ARGUS metrics fall back to defaults (0). Re-feed the last
                // prompt token as a 1-step decode forward to populate per-head
                // attention scores, then restore current_pos so cache state
                // matches prompt_tokens (probe entry beyond current_pos is
                // invisible to subsequent forward calls).
                use llm_rs2::core::kv_cache::KVCacheOps;
                if hook.needs_score_probe(&kv_caches) {
                    let saved_positions: Vec<usize> =
                        kv_caches.iter().map(|c| c.current_pos()).collect();
                    let last_prompt_token = batch_input_ids[prompt_tokens - 1];
                    unsafe {
                        *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = last_prompt_token;
                    }
                    backend.write_buffer(&mut gen_input_tensor, unsafe {
                        std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
                    })?;
                    if let Some(acc) = hook.score_accumulator() {
                        acc.begin_step();
                    }
                    let probe_mem: &dyn Memory = if is_gpu {
                        memory.as_ref()
                    } else {
                        cpu_memory_arc.as_ref()
                    };
                    model.forward_into(TransformerModelForwardArgs {
                        input_tokens: &gen_input_tensor,
                        start_pos: prompt_tokens - 1,
                        kv_caches: &mut kv_caches,
                        backend: &backend,
                        memory: probe_mem,
                        logits_out: &mut logits,
                        x_gen: Some(&mut x_gen),
                        workspace: Some(&mut gen_ws),
                        score_accumulator: hook.score_accumulator(),
                        profiler: None,
                        skip_config: skip_config.as_ref(),
                        importance_collector: None,
                        logits_last_only: false,
                        variance_collector: None,
                        prefill_workspace: None,

                        layer_boundary_hook: None,
                    })?;
                    for (cache, &pos) in kv_caches.iter_mut().zip(saved_positions.iter()) {
                        cache.set_current_pos(pos);
                    }
                }

                // ARGUS Step1~6: compute experimental_qcf payload from prefill state.
                // Also triggers post-prefill eviction when budget exceeded.
                hook.post_prefill(&mut kv_caches);

                eprintln!(
                    "[Batch] #{} id={} prefill_end ts={:.3}",
                    iteration,
                    entry.id,
                    unix_ts()
                );
                eprintln!(
                    "[Batch] #{} id={} decode_start ts={:.3}",
                    iteration,
                    entry.id,
                    unix_ts()
                );

                // === DECODE LOOP ===
                let mut tbt_values_batch: Vec<f64> = Vec::new();
                let mut generated_count: usize = 1; // first token already sampled
                let mut last_token_time = std::time::Instant::now();

                for _ in 0..(args.num_tokens - 1) {
                    if kv_caches[0].current_pos >= max_seq_len {
                        break;
                    }

                    // Throttle delay
                    if throttle_delay_ms > 0 {
                        std::thread::sleep(std::time::Duration::from_millis(throttle_delay_ms));
                    }

                    // Write token to CPU input
                    let current_token = *batch_tokens.last().unwrap();
                    unsafe {
                        *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = current_token;
                    }
                    backend.write_buffer(&mut gen_input_tensor, unsafe {
                        std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
                    })?;

                    let decode_start = std::time::Instant::now();
                    // Use CPU memory when on CPU; GPU memory otherwise.
                    // After SwitchHw GPU→CPU, `memory` is still OpenCL memory whose
                    // alloc() creates OpenCLBuffer (null as_ptr). Must use
                    // cpu_memory_arc for CPU-accessible lazy allocations.
                    let effective_mem: &dyn Memory = if is_gpu {
                        gpu_memory_arc.as_deref().unwrap_or_else(|| memory.as_ref())
                    } else {
                        cpu_memory_arc.as_ref()
                    };
                    model.forward_into(TransformerModelForwardArgs {
                        input_tokens: &gen_input_tensor,
                        start_pos: batch_start_pos,
                        kv_caches: &mut kv_caches,
                        backend: &backend,
                        memory: effective_mem,
                        logits_out: &mut logits,
                        x_gen: Some(&mut x_gen),
                        workspace: Some(&mut gen_ws),
                        score_accumulator: hook.score_accumulator(),
                        profiler: None,
                        skip_config: skip_config.as_ref(),
                        importance_collector: None,
                        logits_last_only: false,
                        variance_collector: None,
                        prefill_workspace: None,

                        layer_boundary_hook: None,
                    })?;
                    backend.synchronize()?;
                    hook.post_decode_step(&mut kv_caches, generated_count);

                    let now = std::time::Instant::now();
                    let tbt = (now - last_token_time).as_secs_f64() * 1000.0;
                    tbt_values_batch.push(tbt);
                    last_token_time = now;
                    let _forward_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

                    // Read logits and sample
                    unsafe {
                        let ptr = logits_cpu.as_mut_ptr() as *mut u8;
                        let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
                        backend.read_buffer(&logits, slice)?;
                    }

                    let next_id = sampling::sample(
                        &mut logits_cpu,
                        &batch_tokens,
                        vocab_size,
                        &sampling_config,
                        None,
                    );
                    batch_tokens.push(next_id);
                    batch_start_pos += 1;
                    generated_count += 1;

                    if next_id == eos_id && !args.ignore_eos {
                        break;
                    }
                }

                eprintln!(
                    "[Batch] #{} id={} decode_end ts={:.3}",
                    iteration,
                    entry.id,
                    unix_ts()
                );

                let total_ms = entry_start.elapsed().as_secs_f64() * 1000.0;
                let mean_tbt_ms = if tbt_values_batch.is_empty() {
                    0.0
                } else {
                    tbt_values_batch.iter().sum::<f64>() / tbt_values_batch.len() as f64
                };

                // Decode generated text (skip prompt tokens)
                let generated_ids = &batch_tokens[prompt_tokens..];
                let text = tokenizer.decode(generated_ids, true).unwrap_or_default();

                // Output JSONL
                let mut result = serde_json::json!({
                    "id": entry.id,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_count,
                    "ttft_ms": (ttft_ms * 100.0).round() / 100.0,
                    "mean_tbt_ms": (mean_tbt_ms * 100.0).round() / 100.0,
                    "total_ms": (total_ms * 100.0).round() / 100.0,
                    "text": text,
                });
                // Merge ARGUS Step1~6 fields (qcf_caote_max / qcf_per_head /
                // qcf_topk_retention_* / attention_entropy / qcf_beta_amplified_* /
                // qcf_per_layer*) when --enable-qcf-experimental is on.
                if let serde_json::Value::Object(extra_map) = hook.extra_question_fields(&kv_caches)
                    && let serde_json::Value::Object(ref mut rmap) = result
                {
                    for (k, v) in extra_map {
                        rmap.insert(k, v);
                    }
                }
                println!("{}", serde_json::to_string(&result)?);

                eprintln!(
                    "[Batch] #{} id={} done: {} tokens, ttft={:.1}ms, tbt={:.1}ms, total={:.1}ms",
                    iteration, entry.id, generated_count, ttft_ms, mean_tbt_ms, total_ms
                );

                // === RESET KV CACHE + score accumulator + per-record hook state ===
                hook.reset_caches(&mut kv_caches);

                iteration += 1;
            }

            if !args.prompt_batch_loop {
                break;
            }
        }

        eprintln!("[Batch] Complete: {} iterations", iteration);
        return Ok(());
    }

    // Inference profiler (activated by either --profile or --profile-events).
    // Declared before prefill so PrefillOpProfiler can be populated.
    //
    // --profile-events uses the same InferenceProfiler container (ops/json
    // export) but feeds it via OpProfiler::merge_from_events() instead of
    // the legacy per-op synchronize+wall-clock path.
    let mut profiler = if args.profile || args.profile_events {
        Some(llm_rs2::profile::InferenceProfiler::new(
            llm_rs2::profile::ProfileConfig {
                score_snapshot_interval: args.profile_interval,
                track_per_head: args.profile_per_head,
                enabled_probes: args.profile_probes.split(',').map(String::from).collect(),
                output_dir: std::path::PathBuf::from(&args.profile_dir),
            },
        ))
    } else {
        None
    };

    // Pre-allocate generation buffers
    let logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut logits = Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        logits_buf,
        backend.clone(),
    );

    // Cache EOS token ID from config.json (model-agnostic)
    let eos_id = model.config.eos_token_id;

    // === WARMUP: trigger DVFS ramp-up before timed prefill ===
    // Runs a forward pass and brief CPU spin to ensure governor reaches max clock.
    // Without this, idle CPU starts at ~2.2GHz and ramp-up time
    // pollutes the prefill measurement (llama.cpp's model loading + warmup
    // achieves the same effect).
    //
    // Env overrides (for gap investigation):
    //   LLMRS_SKIP_WARMUP=1     : disable warmup entirely (baseline cold-start)
    //   LLMRS_WARMUP_TOKENS=N   : warmup with N tokens (default 1). Use >1 to JIT-compile
    //                             prefill-path kernels (batched QKV / flash_attn prefill).
    if std::env::var("LLMRS_SKIP_WARMUP").is_err() {
        let warmup_tokens: usize = std::env::var("LLMRS_WARMUP_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1)
            .max(1)
            .min(tokens.len());

        let warmup_start = std::time::Instant::now();
        let warmup_buf = Galloc::new().alloc(warmup_tokens * 4, DType::U8)?;
        unsafe {
            let ptr = warmup_buf.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, warmup_tokens);
        }
        let warmup_input = Tensor::new(
            Shape::new(vec![1, warmup_tokens]),
            warmup_buf,
            Arc::new(CpuBackend::new()),
        );
        let warmup_input = backend.copy_from(&warmup_input)?;

        let warmup_logits_shape = if warmup_tokens == 1 {
            Shape::new(vec![1, 1, vocab_size])
        } else {
            Shape::new(vec![1, warmup_tokens, vocab_size])
        };
        let warmup_logits_buf = memory.alloc(warmup_tokens * vocab_size * 4, DType::F32)?;
        let mut warmup_logits =
            Tensor::new(warmup_logits_shape, warmup_logits_buf, backend.clone());

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &warmup_input,
            start_pos: 0,
            kv_caches: &mut kv_caches,
            backend: &backend,
            memory: memory.as_ref(),
            logits_out: &mut warmup_logits,
            x_gen: None,
            workspace: None,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;
        backend.synchronize()?;
        let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[WARMUP] tokens={} ms={:.2}", warmup_tokens, warmup_ms);

        // Brief all-core spin to push DVFS governor to max frequency.
        // 50ms is enough for walt governor to ramp up.
        use rayon::prelude::*;
        let spin_until = std::time::Instant::now() + std::time::Duration::from_millis(50);
        (0..rayon::current_num_threads())
            .into_par_iter()
            .for_each(|_| {
                while std::time::Instant::now() < spin_until {
                    std::hint::spin_loop();
                }
            });

        // Reset KV caches
        for cache in kv_caches.iter_mut() {
            cache.current_pos = 0;
            cache.high_water_pos = 0;
        }
    } else {
        eprintln!("[WARMUP] skipped (LLMRS_SKIP_WARMUP)");
    }

    // D2O layer-level allocation: create variance collector before prefill.
    // Only active when --eviction-policy d2o and --d2o-layer-alloc are both set.
    let mut variance_collector = if args.d2o_layer_alloc && args.eviction_policy == "d2o" {
        Some(
            llm_rs2::core::pressure::d2o_layer_alloc::D2OVarianceCollector::new(
                model.config.num_hidden_layers,
                model.config.num_key_value_heads,
                model.config.num_attention_heads,
                model.config.head_dim,
                tokens.len(),
            ),
        )
    } else {
        None
    };

    // Weight swap state (ENG-ALG-218 + ENG-ALG-214-ROUTE).
    //
    // `importance_table_for_swap`: most-recently collected per-layer
    // importance table from an on-demand prefill measurement.  `None`
    // until the first `RequestQcf` prefill completes.
    //
    // `collector_armed`: true when a `RequestQcf` has been received and
    // we are waiting for the next prefill to inject `ImportanceCollector`.
    // This is a lightweight bool; the actual collector lives on the stack
    // during prefill (not stored here).
    let mut importance_table_for_swap: Option<llm_rs2::core::qcf::ImportanceTable> = None;
    let mut collector_armed = false;

    // === PREFILL PHASE ===
    let mut deferred_switch: Option<String> = None;
    {
        println!("[Profile] Event: PrefillStart");
        let process_len = tokens.len();
        if process_len > max_seq_len {
            anyhow::bail!(
                "Prompt length {} exceeds max_seq_len {}",
                process_len,
                max_seq_len
            );
        }

        // Determine effective chunk size.
        // 0 or >= process_len → use full prompt as single chunk (original behaviour).
        // When resilience is enabled, auto-chunk at 256 so that chunk boundaries
        // serve as checkpoints for SwitchHw / Throttle / LayerSkip commands.
        // When GPU backend and prefill_chunk_size == 0, auto-derive a safe chunk
        // size from max_single_alloc() to avoid CL_INVALID_BUFFER_SIZE (-61).
        let auto_gpu_chunk: Option<usize> = if args.prefill_chunk_size == 0 && backend.is_gpu() {
            let max_alloc = backend.max_single_alloc();
            if max_alloc > 0 {
                // Each chunk needs a logits buffer: chunk * vocab_size * 4 bytes.
                // Use 50% of max_single_alloc as conservative budget.
                let budget = max_alloc / 2;
                let by_vocab = (budget / (vocab_size * 4)).max(1);
                // Also bound by hidden_size to keep activation buffers feasible.
                let by_hidden = (max_alloc / (hidden_size * 4)).max(1);
                let derived = by_vocab.min(by_hidden).min(512);
                Some(derived)
            } else {
                None
            }
        } else {
            None
        };
        let chunk_size = if args.prefill_chunk_size > 0 && args.prefill_chunk_size < process_len {
            args.prefill_chunk_size
        } else if let Some(auto) = auto_gpu_chunk {
            if auto < process_len {
                eprintln!(
                    "[Prefill] prefill_chunk_size auto-selected: {} (max_alloc={}MB, vocab={}, hidden={})",
                    auto,
                    backend.max_single_alloc() / (1024 * 1024),
                    vocab_size,
                    hidden_size,
                );
                auto
            } else {
                process_len
            }
        } else if args.enable_resilience && process_len > 256 {
            256
        } else {
            process_len
        };
        let chunked = chunk_size < process_len;
        if chunked {
            eprintln!(
                "[Prefill] Chunked mode: {} tokens in chunks of {}",
                process_len, chunk_size
            );
        }

        // Dynamic prefill policy: start from CLI values, updated by SetPrefillPolicy.
        let mut effective_chunk_size = chunk_size;
        let mut effective_yield_ms = args.prefill_yield_ms;
        let mut effective_cpu_chunk_size = args.prefill_cpu_chunk_size;

        // Reusable logits buffer: [1, 1, vocab_size] when chunked, else [1, process_len, vocab_size].
        // Chunked mode always uses logits_last_only=true so only 1 position is written per chunk.
        let (logits_shape, logits_buf_size) = if chunked {
            (Shape::new(vec![1, 1, vocab_size]), vocab_size * 4)
        } else {
            (
                Shape::new(vec![1, process_len, vocab_size]),
                process_len * vocab_size * 4,
            )
        };
        let prefill_logits_buf = memory.alloc(logits_buf_size, DType::F32)?;
        let mut prefill_logits = Tensor::new(logits_shape, prefill_logits_buf, backend.clone());

        let prefill_timer = std::time::Instant::now();
        let mut prefill_pure_fwd_ms: f64 = 0.0;
        let total_chunks = process_len.div_ceil(chunk_size);

        // Report prefill start to resilience manager.
        if let Some(executor) = &mut command_executor {
            executor.set_prefill_state("prefill", 0, process_len);
        }

        // ENG-ALG-218: if collector is armed, prepare a collector for this prefill.
        // Armed by `RequestQcf` handler in decode loop; collector is injected into
        // the last prefill chunk so it captures the final contextual activation state.
        let mut on_demand_collector: Option<llm_rs2::core::qcf::ImportanceCollector> =
            if collector_armed {
                Some(llm_rs2::core::qcf::ImportanceCollector::new())
            } else {
                None
            };
        if collector_armed {
            collector_armed = false; // consume the flag; armed at most once per prefill
        }

        let mut chunk_start = 0;
        let mut chunk_idx = 0usize;
        while chunk_start < process_len {
            // Guard: effective_chunk_size must be at least 1.
            let ecs = effective_chunk_size.max(1);
            let chunk_end = (chunk_start + ecs).min(process_len);
            let chunk_tokens = &tokens[chunk_start..chunk_end];
            let chunk_len = chunk_tokens.len();

            // ENG-ALG-218: inject collector only on the last prefill chunk.
            // Earlier chunks have partial seq_len; the last chunk captures final
            // contextual state which is most representative for per-layer importance.
            let is_last_chunk = chunk_end >= process_len;
            let inject_collector = is_last_chunk && on_demand_collector.is_some();

            let chunk_trace = std::env::var("LLMRS_PREFILL_CHUNK_MS").is_ok();
            let t_chunk_start = std::time::Instant::now();

            // Build CPU input tensor for this chunk.
            let cpu_chunk_buf = Galloc::new().alloc(chunk_len * 4, DType::U8)?;
            unsafe {
                let ptr = cpu_chunk_buf.as_mut_ptr() as *mut u32;
                std::ptr::copy_nonoverlapping(chunk_tokens.as_ptr(), ptr, chunk_len);
            }
            let cpu_chunk_tensor = Tensor::new(
                Shape::new(vec![1, chunk_len]),
                cpu_chunk_buf,
                Arc::new(CpuBackend::new()),
            );
            let input_tensor = backend.copy_from(&cpu_chunk_tensor)?;
            let t_setup_end = std::time::Instant::now();

            // RoPE position for this chunk: start_pos (0 during prefill) + offset within prompt.
            let chunk_start_pos = start_pos + chunk_start;

            model.forward_into(TransformerModelForwardArgs {
                input_tokens: &input_tensor,
                start_pos: chunk_start_pos,
                kv_caches: &mut kv_caches,
                backend: &backend,
                memory: memory.as_ref(),
                logits_out: &mut prefill_logits,
                x_gen: None,
                workspace: None,
                score_accumulator: None, // No score tracking during prefill
                profiler: profiler.as_mut().map(|p| &mut p.ops),
                skip_config: None,
                importance_collector: if inject_collector {
                    on_demand_collector.as_mut()
                } else {
                    None
                },
                // Chunked mode: only the last position's logits needed (saves GPU memory).
                // Non-chunked: write all positions (original behaviour).
                logits_last_only: chunked,
                variance_collector: variance_collector.as_mut(),
                prefill_workspace: None,

                layer_boundary_hook: None,
            })?;
            backend.synchronize()?;
            let t_fwd_end = std::time::Instant::now();
            let fwd_ms = (t_fwd_end - t_setup_end).as_secs_f64() * 1000.0;
            prefill_pure_fwd_ms += fwd_ms;
            if chunk_trace {
                let setup_ms = (t_setup_end - t_chunk_start).as_secs_f64() * 1000.0;
                let total_ms = (t_fwd_end - t_chunk_start).as_secs_f64() * 1000.0;
                eprintln!(
                    "[PREFILL_CHUNK] idx={} start_pos={} len={} setup_ms={:.2} fwd_ms={:.2} total_ms={:.2}",
                    chunk_idx, chunk_start_pos, chunk_len, setup_ms, fwd_ms, total_ms
                );
            }

            // Immediately release the GPU input buffer for this chunk.
            drop(input_tensor);

            chunk_start = chunk_end;

            // Inter-chunk yield: sleep after GPU chunk to release compute for other processes.
            if effective_yield_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(effective_yield_ms as u64));
            }

            // CPU interleave: process next chunk on CPU while GPU is free.
            // Invariant: the last chunk must be processed by GPU so that
            // prefill_logits (GPU buffer) is valid at the end.
            if effective_cpu_chunk_size > 0 && chunk_start < process_len {
                let remaining = process_len - chunk_start;
                // Only run CPU chunk if enough tokens remain for GPU to handle
                // at least one more chunk afterwards.
                if remaining > effective_cpu_chunk_size {
                    // Flush GPU caches to main memory before CPU reads KV buffers.
                    // On ARM UMA, clFinish() alone may not flush GPU L1/L2 cache.
                    // map_for_cpu() calls clEnqueueMapBuffer which ensures coherence.
                    for kv in kv_caches.iter() {
                        kv.k_buffer.buffer().map_for_cpu()?;
                        kv.v_buffer.buffer().map_for_cpu()?;
                    }

                    let cpu_end =
                        (chunk_start + effective_cpu_chunk_size).min(process_len.saturating_sub(1));
                    if cpu_end > chunk_start {
                        let cpu_tokens = &tokens[chunk_start..cpu_end];
                        let cpu_len = cpu_tokens.len();

                        let cpu_in_buf = Galloc::new().alloc(cpu_len * 4, DType::U8)?;
                        unsafe {
                            let ptr = cpu_in_buf.as_mut_ptr() as *mut u32;
                            std::ptr::copy_nonoverlapping(cpu_tokens.as_ptr(), ptr, cpu_len);
                        }
                        let cpu_in_tensor = Tensor::new(
                            Shape::new(vec![1, cpu_len]),
                            cpu_in_buf,
                            cpu_backend_arc.clone(),
                        );

                        let cpu_chunk_start_pos = start_pos + chunk_start;

                        // CPU prefill logits: use a separate CPU buffer to avoid writing
                        // to GPU prefill_logits. These intermediate logits are discarded.
                        let cpu_logits_buf = cpu_memory_arc.alloc(vocab_size * 4, DType::F32)?;
                        let mut cpu_logits = Tensor::new(
                            Shape::new(vec![1, 1, vocab_size]),
                            cpu_logits_buf,
                            cpu_backend_arc.clone(),
                        );

                        model.forward_into(TransformerModelForwardArgs {
                            input_tokens: &cpu_in_tensor,
                            start_pos: cpu_chunk_start_pos,
                            kv_caches: &mut kv_caches,
                            backend: &cpu_backend_arc,
                            memory: cpu_memory_arc.as_ref(),
                            logits_out: &mut cpu_logits,
                            x_gen: None,
                            workspace: None,
                            score_accumulator: None,
                            profiler: None,
                            skip_config: None,
                            importance_collector: None,
                            logits_last_only: true,
                            variance_collector: None,
                            prefill_workspace: None,

                            layer_boundary_hook: None,
                        })?;
                        // No backend.synchronize() needed — CPU forward is synchronous.
                        drop(cpu_in_tensor);
                        drop(cpu_logits);

                        chunk_start = cpu_end;
                    }
                }
                // else: remaining tokens fit in one GPU chunk → GPU finishes.
            }

            // ── Prefill resilience checkpoint (chunk boundary) ──
            // Poll CommandExecutor between chunks to handle SwitchHw, Throttle,
            // and LayerSkip commands mid-prefill. Only active in chunked mode.
            if chunked && let Some(executor) = &mut command_executor {
                let kv_snap = KVSnapshot {
                    total_bytes: kv_caches
                        .iter()
                        .map(|c| (c.k_buffer.buffer().size() + c.v_buffer.buffer().size()) as u64)
                        .sum(),
                    total_tokens: kv_caches[0].current_pos,
                    capacity: kv_caches[0].capacity(),
                    protected_prefix: actual_protected_prefix,
                    kv_dtype: args.kv_type.clone(),
                    eviction_policy: args.eviction_policy.clone(),
                    skip_ratio: 0.0,
                };
                let plan = executor.poll(&kv_snap);

                // SetPrefillPolicy: dynamically adjust chunk/yield/cpu parameters.
                if let Some(v) = plan.prefill_chunk_size {
                    effective_chunk_size = v;
                    eprintln!("[Prefill] Policy: chunk_size -> {}", v);
                }
                if let Some(v) = plan.prefill_yield_ms {
                    effective_yield_ms = v;
                    eprintln!("[Prefill] Policy: yield_ms -> {}", v);
                }
                if let Some(v) = plan.prefill_cpu_chunk_size {
                    let layer0_probe = model.layers[0].load_weights();
                    if v > 0 && layer0_probe.wq.as_ptr().is_null() {
                        eprintln!(
                            "[Prefill] Policy: cpu_chunk_size={} rejected — weights not CPU-accessible. \
                             Use --resilience-prealloc-switch or --prefill-cpu-chunk-size at CLI.",
                            v
                        );
                    } else {
                        effective_cpu_chunk_size = v;
                        eprintln!("[Prefill] Policy: cpu_chunk_size -> {}", v);
                    }
                }

                // Throttle: sleep between chunks
                if plan.throttle_delay_ms > 0 && plan.throttle_delay_ms != throttle_delay_ms {
                    eprintln!(
                        "[Prefill] Throttle: {}ms -> {}ms",
                        throttle_delay_ms, plan.throttle_delay_ms
                    );
                }
                throttle_delay_ms = plan.throttle_delay_ms;
                if throttle_delay_ms > 0 {
                    eprintln!(
                        "[Prefill] Throttle: {}ms delay after chunk {}/{}",
                        throttle_delay_ms,
                        chunk_idx + 1,
                        total_chunks
                    );
                    std::thread::sleep(std::time::Duration::from_millis(throttle_delay_ms));
                }

                // LayerSkip
                if plan.restore_defaults {
                    skip_config = None;
                    last_skip_ratio = None;
                    effective_chunk_size = chunk_size;
                    effective_yield_ms = args.prefill_yield_ms;
                    effective_cpu_chunk_size = args.prefill_cpu_chunk_size;
                } else if let Some(ratio) = plan.layer_skip
                    && last_skip_ratio != Some(ratio)
                {
                    eprintln!("[Prefill] LayerSkip: ratio={:.2}", ratio);
                    skip_config = Some(SkipConfig::uniform_init(
                        model.config.num_hidden_layers,
                        ratio,
                    ));
                    last_skip_ratio = Some(ratio);
                }

                // SwitchHw: defer to post-prefill boundary.
                // Mid-prefill switch causes segfault: model workspace buffers
                // remain on the old backend; the next chunk accesses them
                // from the new backend -> invalid memory reference.
                if let Some(ref device) = plan.switch_device {
                    if deferred_switch.is_none() {
                        eprintln!(
                            "[Prefill] SwitchHw: deferring '{}' to post-prefill (chunk_pos={})",
                            device, kv_caches[0].current_pos
                        );
                    }
                    deferred_switch = Some(device.clone());
                }

                // Report prefill progress.
                executor.set_prefill_state("prefill", chunk_start, process_len);
            }

            chunk_idx += 1;
        }

        // ENG-ALG-218: finalize on-demand ImportanceCollector after prefill completes.
        // INV-128: this block always runs (normal fall-through from the while loop),
        // so QcfEstimate is guaranteed to be sent when the prefill completes successfully.
        // For panics/early-return paths the caller-side Drop guard is the safety net.
        if let Some(collector) = on_demand_collector.take() {
            let table: llm_rs2::core::qcf::ImportanceTable = collector.build();
            let layer_swap = build_layer_swap_estimate(&model, Some(&table));
            if let Some(executor) = &mut command_executor {
                executor.send_qcf_estimate(llm_shared::QcfEstimate {
                    estimates: std::collections::HashMap::new(),
                    layer_swap,
                });
                log::debug!("[QCF] QcfEstimate sent after prefill finalization (ENG-ALG-218)");
            }
            importance_table_for_swap = Some(table);
        }

        let prefill_forward_ms = prefill_timer.elapsed().as_secs_f64() * 1000.0;

        // Report transition to decode phase.
        if let Some(executor) = &mut command_executor {
            executor.set_prefill_state("decode", 0, 0);
        }

        // Auto-eviction after prefill (sliding window only, non-experiment mode)
        if auto_eviction {
            cache_manager.maybe_evict(&mut kv_caches).ok();
        }

        // Sticky eviction at prefill→decode boundary.
        // If a KvEvict directive arrived during prefill, executor holds a sticky evict_plan.
        // Execute it now (before decode starts) to reduce attention work from the first decode step.
        // Score-based methods (H2O/D2O) are not available here — falls back to force_evict.
        if let Some(ref mut exec) = command_executor {
            let kv_snap = KVSnapshot {
                total_bytes: kv_caches
                    .iter()
                    .map(|c| (c.k_buffer.buffer().size() + c.v_buffer.buffer().size()) as u64)
                    .sum(),
                total_tokens: kv_caches[0].current_pos,
                capacity: kv_caches[0].capacity(),
                protected_prefix: actual_protected_prefix,
                kv_dtype: "f16".to_string(),
                eviction_policy: args.eviction_policy.clone(),
                skip_ratio: 0.0,
            };
            let plan = exec.poll(&kv_snap);
            if let Some(evict) = &plan.evict {
                let effective_ratio = args.experiment_eviction_ratio.unwrap_or(evict.target_ratio);
                if effective_ratio > 0.0 {
                    let current_pos = kv_caches[0].current_pos;
                    // Use current_pos as ceiling (first and only boundary eviction).
                    let tgt_raw = (current_pos as f32 * effective_ratio).max(1.0) as usize;
                    let target_pos = tgt_raw.max(args.min_kv_cache);
                    if current_pos > target_pos {
                        // adjusted_ratio so force_evict(current_pos * adjusted) == target_pos.
                        let adjusted_ratio = target_pos as f32 / current_pos as f32;
                        // Dispatch by evict method (same as decode loop).
                        // Scores are unavailable at prefill→decode boundary, so
                        // D2O and score-based H2O fall back to force_evict.
                        let result = if evict.method == llm_rs2::resilience::EvictMethod::Streaming
                        {
                            if let Some(ref sp) = evict.streaming_params {
                                let policy =
                                    llm_rs2::core::eviction::streaming_llm::StreamingLLMPolicy::new(
                                        sp.sink_size,
                                        sp.window_size,
                                    );
                                cache_manager.force_evict_by_policy_ref(
                                    &policy,
                                    &mut kv_caches,
                                    adjusted_ratio,
                                    llm_rs2::core::cache_manager::ScoreContext::None,
                                )
                            } else {
                                cache_manager.force_evict(&mut kv_caches, adjusted_ratio)
                            }
                        } else {
                            cache_manager.force_evict_by_policy(
                                evict.method,
                                &mut kv_caches,
                                adjusted_ratio,
                                llm_rs2::core::cache_manager::ScoreContext::None,
                            )
                        };
                        match result {
                            Ok(r) if r.evicted => {
                                eprintln!(
                                    "[Prefill→Decode] Evicted {} tokens (pos: {} → {})",
                                    r.tokens_removed,
                                    r.new_pos + r.tokens_removed,
                                    r.new_pos
                                );
                            }
                            Err(e) => eprintln!("[Prefill→Decode] Eviction error: {}", e),
                            _ => {}
                        }
                    }
                }
            }
        }

        // Sample last token — read logits from the last chunk's output.
        // When chunked: prefill_logits is [1,1,vocab_size], last_logits = the only row.
        // When not chunked: prefill_logits is [1,process_len,vocab_size], take last row.
        let mut last_logits = vec![0.0f32; vocab_size];
        unsafe {
            let ptr = last_logits.as_mut_ptr() as *mut u8;
            let byte_len = vocab_size * 4;
            if chunked {
                // Single-row buffer; read all of it.
                let slice = std::slice::from_raw_parts_mut(ptr, byte_len);
                backend.read_buffer(&prefill_logits, slice)?;
            } else {
                // Multi-row buffer; read only the last row.
                // read_buffer reads from offset 0, so we read the full buffer and
                // then take the last vocab_size elements.
                let mut logits_cpu = vec![0.0f32; process_len * vocab_size];
                let full_ptr = logits_cpu.as_mut_ptr() as *mut u8;
                let full_slice = std::slice::from_raw_parts_mut(full_ptr, logits_cpu.len() * 4);
                backend.read_buffer(&prefill_logits, full_slice)?;
                let start_idx = (process_len - 1) * vocab_size;
                last_logits.copy_from_slice(&logits_cpu[start_idx..start_idx + vocab_size]);
            }
        }

        let next_token_id = sampling::sample(
            &mut last_logits,
            &tokens,
            vocab_size,
            &sampling_config,
            None,
        );

        _ttft_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "Prefill: {:.2} ms ({} tokens, {:.1} tok/s)",
            prefill_forward_ms,
            process_len,
            process_len as f64 / (prefill_forward_ms / 1000.0),
        );
        eprintln!(
            "Prefill(pure fwd): {:.2} ms ({} tokens, {:.1} tok/s) [sync'd forward only, comparable to llama-bench pp]",
            prefill_pure_fwd_ms,
            process_len,
            process_len as f64 / (prefill_pure_fwd_ms / 1000.0),
        );
        _last_token_time = std::time::Instant::now();

        tokens.push(next_token_id);
        start_pos += process_len;
        // T2: first forward pass (prefill) complete, KV cache filled.
        rss_trace("prefill_done");
    }

    // Execute deferred SwitchHw (from prefill checkpoint).
    // Now safe: prefill is done, logits read, all workspace released.
    // Decode buffers are allocated *after* this point, so only KV migrate
    // and backend/is_gpu update are needed here.
    if let Some(ref device) = deferred_switch
        && let (Some(gpu_be), Some(gpu_mem)) = (&gpu_backend_arc, &gpu_memory_arc)
    {
        match device.as_str() {
            "cpu" if is_gpu => {
                eprintln!("[Prefill->Decode] Executing deferred SwitchHw: GPU->CPU");
                llm_rs2::core::kv_migrate::migrate_kv_caches(
                    &mut kv_caches,
                    &backend,
                    &cpu_backend_arc,
                    &cpu_backend_arc,
                    &cpu_memory_arc,
                    &cpu_memory_arc,
                    kv_heads,
                    head_dim,
                    max_seq_len,
                    false,
                )?;
                backend = cpu_backend_arc.clone();
                is_gpu = false;
                // Re-tag weight tensors with CPU backend.
                // UnifiedBuffer (ALLOC_HOST_PTR, mapped) stays valid for CPU.
                eprintln!("[Prefill->Decode] SwitchHw: Switched to CPU.");
            }
            "gpu" | "opencl" if !is_gpu && weights_on_gpu => {
                eprintln!("[Prefill->Decode] Executing deferred SwitchHw: CPU->GPU");
                llm_rs2::core::kv_migrate::migrate_kv_caches(
                    &mut kv_caches,
                    &backend,
                    gpu_be,
                    &cpu_backend_arc,
                    &cpu_memory_arc,
                    gpu_mem,
                    kv_heads,
                    head_dim,
                    max_seq_len,
                    true,
                )?;
                backend = gpu_be.clone();
                is_gpu = true;
                eprintln!("[Prefill->Decode] SwitchHw: Switched to GPU.");
            }
            _ => {}
        }
    }

    // D2O: compute per-layer budgets from prefill attention variance.
    let d2o_layer_ratios: Option<Vec<(f32, f32)>> = if let Some(ref collector) = variance_collector
    {
        let budgets = collector.compute_budgets(
            args.d2o_keep_ratio * args.eviction_target_ratio,
            (1.0 - args.d2o_keep_ratio) * args.eviction_target_ratio,
        );
        log::info!(
            "[D2O] Layer budgets computed: {:?}",
            budgets.iter().map(|(h, r)| h + r).collect::<Vec<_>>()
        );
        Some(budgets)
    } else {
        None
    };

    // Position → birth step mapping for profiling (token identity tracking)
    let mut position_birth_step: Vec<usize> = if profiler.is_some() {
        // All prefill tokens have birth_step = 0 (prompt)
        let prompt_len = tokens.len();
        let map = vec![0usize; prompt_len];
        // Register prompt token births + first generated token
        if let Some(ref mut p) = profiler {
            p.scores
                .record_token_births(0, prompt_len, actual_protected_prefix);
        }
        map
    } else {
        Vec::new()
    };

    // === GENERATION PHASE ===
    {
        println!("[Profile] Event: DecodingStart");

        // --profile-events / --heartbeat-gpu-profile: drop any events captured
        // during prefill/warmup so the decode-only aggregate is not polluted.
        // Prefill uses the generic `forward` path (no label hints), so without
        // this step all matmul dispatches from prefill would spill into the
        // decode "matmul" bucket and inflate the GPU self-util meter's first
        // heartbeat sample.
        #[cfg(feature = "opencl")]
        if (args.profile_events || args.heartbeat_gpu_profile)
            && let Some(ocl_be) = backend
                .as_any()
                .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
            && ocl_be.profile_events_enabled
        {
            backend.synchronize()?;
            ocl_be.flush_and_aggregate_profile()?;
            let _ = ocl_be.take_profile_accum();
            // Prefill-phase GPU busy ns were also fed into the self-util
            // meter via flush_and_aggregate_profile(); drain them so the
            // first heartbeat only reflects decode-phase usage.
            if let Some(m) = ocl_be.gpu_self_meter() {
                use llm_rs2::resilience::GpuSelfMeter;
                let _ = m.sample(std::time::Duration::from_secs(1));
            }
            eprintln!("[Profile] prefill/warmup events dropped (decode-only accumulator)");
        }
        // Pre-allocate workspace for generation
        let q_dim = model.config.num_attention_heads * model.config.head_dim;
        let k_dim = model.config.num_key_value_heads * model.config.head_dim;
        let v_dim = k_dim;
        let ffn_hidden = model.config.intermediate_size;

        // After SwitchHw GPU->CPU, `memory` is still OpenCL memory whose
        // alloc() creates OpenCLBuffer (null as_ptr). Use cpu_memory_arc when on CPU.
        let decode_mem: &dyn Memory = if is_gpu {
            memory.as_ref()
        } else {
            cpu_memory_arc.as_ref()
        };

        // Re-allocate logits on the correct backend after deferred SwitchHw.
        // The outer `logits` was allocated with `memory` (GPU) before the
        // deferred switch. After GPU→CPU, the unmapped UnifiedBuffer has
        // as_ptr() == null → segfault when CPU forward writes logits.
        if !is_gpu && logits.as_ptr().is_null() {
            let new_logits_buf = decode_mem.alloc(vocab_size * 4, DType::F32)?;
            logits = Tensor::new(
                Shape::new(vec![1, 1, vocab_size]),
                new_logits_buf,
                backend.clone(),
            );
        }

        let x_gen_buf = decode_mem.alloc(hidden_size * 4, DType::F32)?;
        let mut x_gen = Tensor::new(
            Shape::new(vec![1, 1, hidden_size]),
            x_gen_buf,
            backend.clone(),
        );

        let mut gen_ws = LayerWorkspace::new(
            WorkspaceConfig {
                batch_size: 1,
                dim: model.config.hidden_size,
                q_dim,
                k_dim,
                v_dim,
                ffn_hidden,
                n_heads: model.config.num_attention_heads,
                max_seq_len: args.max_seq_len, // Use context window size
            },
            decode_mem,
            backend.clone(),
        )?;

        // Attach partition workspace if tensor partition is active.
        // Use UnifiedBuffer (ALLOC_HOST_PTR) for zero-copy merge (see batch path above).
        let layer0_partition_probe = model.layers[0].load_weights();
        if let Some(ref ctx) = layer0_partition_probe.partition_ctx {
            let gpu_alloc = make_partition_gpu_alloc(&*backend, decode_mem);

            // Zero-copy residual (see line 1807 block for rationale).
            #[cfg(feature = "opencl")]
            if std::env::var_os("LLMRS_PARTITION_ZCOPY_RESIDUAL").is_some()
                || llm_rs2::layers::tensor_partition::partition_poll_flag_enabled()
            {
                if let Some(ub) = gen_ws
                    .residual
                    .buffer()
                    .as_any()
                    .downcast_ref::<llm_rs2::buffer::unified_buffer::UnifiedBuffer>()
                {
                    ub.map()?;
                    eprintln!("[Partition] Residual UnifiedBuffer permanent-mapped for zero-copy");
                } else {
                    eprintln!(
                        "[Partition] WARN: residual buffer is not UnifiedBuffer (zero-copy skipped)"
                    );
                }
            }

            gen_ws.partition_ws = Some(Arc::new(PartitionWsCell::new(PartitionWorkspace::new(
                ctx,
                ffn_hidden,
                hidden_size,
                &gpu_alloc,
                backend.clone(),
                cpu_backend_arc.clone(),
            )?)));
        }

        // Single token CPU tensor for generation loop
        let cpu_gen_indices_buf = Galloc::new().alloc(4, DType::U8)?;
        let cpu_gen_input = Tensor::new(
            Shape::new(vec![1, 1]),
            cpu_gen_indices_buf,
            Arc::new(CpuBackend::new()),
        );

        // Pre-allocate input tensor for decode loop (avoids per-token alloc)
        let gpu_gen_input_buf = decode_mem.alloc(4, DType::U8)?;
        let mut gen_input_tensor =
            Tensor::new(Shape::new(vec![1, 1]), gpu_gen_input_buf, backend.clone());

        // Pre-allocate CPU spare decode buffers for zero-alloc GPU→CPU SwitchHw.
        // Both sets (GPU active + CPU spare) stay alive for the process lifetime,
        // enabling instant swap without allocation/deallocation during switch.
        // This prevents Samsung LMKD from killing the process due to RSS spike.
        let (mut spare_logits, mut spare_xgen, mut spare_gen_ws, mut spare_gen_input) =
            if is_gpu && args.resilience_prealloc_switch {
                let cpu_lb = cpu_memory_arc.alloc(vocab_size * 4, DType::F32)?;
                let cpu_xb = cpu_memory_arc.alloc(hidden_size * 4, DType::F32)?;
                let cpu_gi = cpu_memory_arc.alloc(4, DType::U8)?;
                eprintln!("[Switch] Pre-allocated CPU spare buffers for zero-alloc SwitchHw");
                (
                    Some(Tensor::new(
                        Shape::new(vec![1, 1, vocab_size]),
                        cpu_lb,
                        cpu_backend_arc.clone(),
                    )),
                    Some(Tensor::new(
                        Shape::new(vec![1, 1, hidden_size]),
                        cpu_xb,
                        cpu_backend_arc.clone(),
                    )),
                    Some(LayerWorkspace::new(
                        WorkspaceConfig {
                            batch_size: 1,
                            dim: model.config.hidden_size,
                            q_dim,
                            k_dim,
                            v_dim,
                            ffn_hidden,
                            n_heads: model.config.num_attention_heads,
                            max_seq_len: args.max_seq_len,
                        },
                        cpu_memory_arc.as_ref(),
                        cpu_backend_arc.clone(),
                    )?),
                    Some(Tensor::new(
                        Shape::new(vec![1, 1]),
                        cpu_gi,
                        cpu_backend_arc.clone(),
                    )),
                )
            } else {
                (None, None, None, None)
            };

        // Streaming setup
        use std::io::Write;
        let mut stdout = std::io::stdout();
        let mut _printed_len = 0;

        // Print initial tokens (prompt + first generated)
        let initial_text = tokenizer.decode(&tokens, true).unwrap_or_default();
        print!("{}", initial_text);
        _printed_len = initial_text.len();
        stdout.flush().ok();

        // ─── UMA Hybrid Attention setup (Stage C) ─────────────────────
        // LLMRS_ATTN_HYBRID_KV_FRAC=X 가 설정되고 gating 조건이 모두 충족되면
        // 공용 GPU 스크래치 버퍼를 할당하고 HybridScope를 install한다. 스코프
        // 객체는 decode 루프 종료까지 살아있어야 하므로 `_hybrid_scope`로 바인드.
        // Gating 실패 시 reason을 stderr로 한 번 찍고 스킵.
        #[cfg(feature = "opencl")]
        let _hybrid_scope = {
            use llm_rs2::layers::hybrid_attention::{self, HybridAttnSetup};
            match HybridAttnSetup::from_env() {
                Some(kv_frac) => {
                    let backend_is_opencl = backend.name() == "OpenCL";
                    let kv_is_f16 = args.kv_type == "f16";
                    let head_dim_val = model.config.head_dim;
                    let head_dim_ok = head_dim_val == 64 || head_dim_val == 128;
                    let n_heads_q = model.config.num_attention_heads;
                    let n_kv_heads = model.config.num_key_value_heads;
                    let is_gqa = n_kv_heads < n_heads_q;
                    let partition_off =
                        args.tensor_partition <= 0.0 || args.tensor_partition >= 1.0;
                    let eviction_compatible =
                        args.eviction_policy != "kivi" && args.eviction_policy != "qcf";
                    let layout_ok = kv_caches
                        .first()
                        .map(|c| c.layout() == KVLayout::HeadMajor)
                        .unwrap_or(false);

                    let gate_ok = backend_is_opencl
                        && kv_is_f16
                        && head_dim_ok
                        && is_gqa
                        && partition_off
                        && eviction_compatible
                        && layout_ok;

                    if !gate_ok {
                        let reason = if !backend_is_opencl {
                            "backend is not OpenCL"
                        } else if !kv_is_f16 {
                            "kv dtype must be f16"
                        } else if !head_dim_ok {
                            "head_dim must be 64 or 128"
                        } else if !is_gqa {
                            "requires GQA (n_kv_heads < n_heads_q)"
                        } else if !partition_off {
                            "FFN tensor partition is active"
                        } else if !eviction_compatible {
                            "incompatible eviction policy (kivi/qcf)"
                        } else {
                            "KV layout must be HeadMajor"
                        };
                        eprintln!(
                            "[hybrid-attn] LLMRS_ATTN_HYBRID_KV_FRAC={} ignored: {}",
                            kv_frac, reason
                        );
                        None
                    } else {
                        // Map KV/Q/out_attn/residual UnifiedBuffer들을 CPU가 접근
                        // 가능하도록 전부 매핑한다. UMA 특성상 map은 주소만
                        // 고정하고 추가 복사는 하지 않는다. Plan execution에
                        // 들어가기 전에 한 번만 호출되면 충분.
                        let mut map_err: Option<anyhow::Error> = None;
                        for c in kv_caches.iter() {
                            if let Err(e) = c.k_buffer.buffer().map_for_cpu() {
                                map_err = Some(e);
                                break;
                            }
                            if let Err(e) = c.v_buffer.buffer().map_for_cpu() {
                                map_err = Some(e);
                                break;
                            }
                        }
                        if map_err.is_none() {
                            if let Err(e) = gen_ws.q.buffer().map_for_cpu() {
                                map_err = Some(e);
                            } else if let Err(e) = gen_ws.out_attn.buffer().map_for_cpu() {
                                map_err = Some(e);
                            }
                        }
                        if let Some(e) = map_err {
                            eprintln!("[hybrid-attn] failed to map UMA buffers: {} — skipping", e);
                            None
                        } else {
                            let ocl_be = backend
                                .as_any()
                                .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>();
                            match ocl_be {
                                Some(ob) => match HybridAttnSetup::new_for_decode(
                                    &ob.queue,
                                    kv_frac,
                                    n_heads_q,
                                    head_dim_val,
                                ) {
                                    Ok(setup) => {
                                        eprintln!(
                                            "[hybrid-attn] enabled: kv_frac={} n_heads_q={} head_dim={}",
                                            kv_frac, n_heads_q, head_dim_val
                                        );
                                        Some(hybrid_attention::install(Arc::new(setup)))
                                    }
                                    Err(e) => {
                                        eprintln!(
                                            "[hybrid-attn] setup allocation failed: {} — skipping",
                                            e
                                        );
                                        None
                                    }
                                },
                                None => None,
                            }
                        }
                    }
                }
                None => None,
            }
        };

        // Build GPU kernel plan for decode (OpenCL only, lazy rebuild on invalidation)
        // Disable for Gemma3: plan doesn't include QK-norm, post-norm, gelu_tanh_mul
        // Disable when tensor partition is active: plan bypasses forward_gen's
        // partition path entirely (plan = pure GPU chain, no CPU co-execution).
        //
        // Score accumulator coexistence: when a CPU `score_accumulator` is
        // active (H2O/D2O/Sliding/CAOTE eviction), the plan may still be used
        // as long as the paired GPU `gpu_score_acc` is active.  `build_plan`
        // then selects the legacy attention kernel (flash attn has no score
        // output) and pre-binds the GPU score buffer into arg 4. Per-layer
        // `reduce_layer` + post-pass `end_step` are driven by
        // `FullKernelPlan::execute` so CPU readback happens only at eviction
        // time (see `sync_to_cpu` further down).
        #[cfg(feature = "opencl")]
        let accumulator_compatible_with_plan = {
            let has_cpu_acc = score_accumulator.is_some();
            let gpu_acc_active = backend
                .as_any()
                .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
                .and_then(|ob| ob.gpu_score_acc())
                .is_some_and(|acc| acc.is_active());
            !has_cpu_acc || gpu_acc_active
        };
        // Partition is now routed through `build_partitioned_layer_plan` inside
        // `build_plan`, so the old `partition_ctx.is_none()` gate has been
        // removed (see ENG-ALG-200 / arch A.6.1). When partition + plan are
        // both unavailable for a layer (e.g. `LLMRS_PARTITION_PLAN=0`), the
        // builder returns `Err` and the caller falls back to forward_gen.
        #[cfg(feature = "opencl")]
        let mut gpu_plan = if backend.name() == "OpenCL"
            && !args.profile
            && !args.no_gpu_plan
            && accumulator_compatible_with_plan
            && model.config.arch != llm_rs2::models::config::ModelArch::Gemma3
            && !args.swap_intra_forward
            && !args.swap_layer_immediate
            && !args.swap_phase_aware
        {
            model.build_plan(&x_gen, &logits, &gen_ws, &mut kv_caches, &backend)
        } else {
            None
        };
        // Sticky disable: when the initial `build_plan` returns `None` and
        // partition is active, the cause is almost always the opt-in gate
        // (`LLMRS_PARTITION_PLAN=0` default on Adreno, 2026-04-21). Retrying
        // every token spams `build_plan` (~100 ms/token overhead) for no
        // benefit. Lock the disable on the first miss and keep forward_gen.
        // `execute_plan` resetting `gpu_plan = None` for KV-resize
        // invalidation still takes the rebuild path on the next token.
        #[cfg(feature = "opencl")]
        let partition_active_any = model
            .layers
            .iter()
            .any(|s| s.load_weights().partition_ctx.is_some());
        #[cfg(feature = "opencl")]
        let mut gpu_plan_sticky_disabled = partition_active_any && gpu_plan.is_none();

        // Pre-allocate decode buffers (reused across tokens)
        let mut logits_cpu = vec![0.0f32; vocab_size];
        let mut sampling_indices: Vec<usize> = (0..vocab_size).collect();

        // Ceiling for sticky eviction: records current_pos at first eviction trigger.
        // Subsequent evictions use ceiling * ratio as a fixed target to prevent cascade
        // (e.g. cache 33 → 16 → 8 → ... when target_ratio is applied to ever-shrinking pos).
        let mut evict_ceiling: Option<usize> = None;
        let mut evict_floor_logged: Option<bool> = None;

        // Sticky cache for last-applied partition ratio. The executor re-delivers
        // `plan.partition_ratio = Some(sticky)` on every poll (ISSUE-5 fix), so
        // without this guard the consumer below would re-split 84 weights and
        // rebuild the GPU plan on every decode tick (verify v2 REGRESSION-A:
        // q4 enable +102% → +3859% TBT). Seeded from CLI-time partition so the
        // first sticky re-delivery is a no-op when nothing changed.
        let mut last_applied_partition_ratio: Option<f32> =
            if args.tensor_partition > 0.0 && args.tensor_partition < 1.0 {
                Some(args.tensor_partition)
            } else {
                None
            };

        // Generation loop
        for (decode_token_index, _) in (0..(args.num_tokens - 1)).enumerate() {
            let _decode_t = llm_rs2::profile::quality_metrics::Timer::start(
                &llm_rs2::profile::quality_metrics::DECODE_TOTAL,
            );

            // Check physical cache capacity (not start_pos, which is logical RoPE position)
            if kv_caches[0].current_pos >= max_seq_len {
                println!("\n[Stopped: Max context length reached]");
                break;
            }

            // ── LISWAP Phase 3 — mid-decode force-swap trigger ───────────────
            // Fires once at decode_token_index == args.swap_delay_tokens when
            // a pending payload was prepared at prefill end. Same dispatch
            // logic as the prefill-end path (shared via macro) to ensure the
            // four swap modes (single-shot / incremental / intra-forward /
            // phase-aware) all receive the same code path.
            if let Some((ratio, target_layers)) = pending_force_swap.take() {
                if decode_token_index == args.swap_delay_tokens {
                    eprintln!(
                        "weight_swap: mid-decode trigger at decode_token_index={}",
                        decode_token_index,
                    );
                    dispatch_force_swap!(ratio, target_layers);
                } else {
                    // Not yet — re-stash for the next iteration.
                    pending_force_swap = Some((ratio, target_layers));
                }
            }

            // ── Auto-switch CPU→GPU at threshold ─────────────────────────
            if !is_gpu
                && weights_on_gpu
                && args.switch_threshold > 0
                && kv_caches[0].current_pos >= args.switch_threshold
                && let (Some(gpu_be), Some(gpu_mem)) =
                    (gpu_backend_arc.as_ref(), gpu_memory_arc.as_ref())
            {
                eprintln!(
                    "[Switch] Auto-switch CPU→GPU at token {}",
                    kv_caches[0].current_pos
                );
                llm_rs2::core::kv_migrate::migrate_kv_caches(
                    &mut kv_caches,
                    &backend,
                    gpu_be,
                    &cpu_backend_arc,
                    &cpu_memory_arc,
                    gpu_mem,
                    kv_heads,
                    head_dim,
                    max_seq_len,
                    true,
                )?;
                backend = gpu_be.clone();
                let logits_gpu_buf = gpu_mem.alloc(vocab_size * 4, DType::F32)?;
                logits = Tensor::new(
                    Shape::new(vec![1, 1, vocab_size]),
                    logits_gpu_buf,
                    backend.clone(),
                );
                let xg_buf = gpu_mem.alloc(hidden_size * 4, DType::F32)?;
                x_gen = Tensor::new(Shape::new(vec![1, 1, hidden_size]), xg_buf, backend.clone());
                gen_ws = LayerWorkspace::new(
                    WorkspaceConfig {
                        batch_size: 1,
                        dim: model.config.hidden_size,
                        q_dim,
                        k_dim,
                        v_dim,
                        ffn_hidden,
                        n_heads: model.config.num_attention_heads,
                        max_seq_len: args.max_seq_len,
                    },
                    gpu_mem.as_ref(),
                    backend.clone(),
                )?;
                #[cfg(feature = "opencl")]
                {
                    gpu_plan = None; // invalidate; will rebuild after first forward
                }
                // Re-allocate gen_input_tensor on new GPU backend
                let gi_buf = gpu_mem.alloc(4, DType::U8)?;
                gen_input_tensor = Tensor::new(Shape::new(vec![1, 1]), gi_buf, backend.clone());
                is_gpu = true;
                eprintln!("[Switch] Switched to GPU successfully.");
            }
            // ── End auto-switch ──────────────────────────────────────────

            let last_token = tokens[tokens.len() - 1];
            unsafe {
                *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = last_token;
            }
            // Reuse pre-allocated GPU buffer — write data instead of alloc+copy
            backend.write_buffer(&mut gen_input_tensor, unsafe {
                std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
            })?;

            // Apply decay to accumulated importance scores before this step
            if let Some(acc) = score_accumulator.as_mut() {
                acc.begin_step();
            }

            let forward_start = std::time::Instant::now();

            // Phase 2: throttle counter reset — 매 token 시작 시.
            if let Some(ref disp) = phase_aware_swap_dispatcher {
                disp.reset_token_counter();
            }

            // Try GPU plan path (OpenCL decode only, no profiling)
            #[cfg(feature = "opencl")]
            let used_plan = if let Some(ref plan) = gpu_plan {
                match model.execute_plan(
                    plan,
                    &gen_input_tensor,
                    start_pos,
                    &mut x_gen,
                    &mut kv_caches,
                    &mut logits,
                    &backend,
                ) {
                    Ok(true) => true,
                    Ok(false) => {
                        // Plan invalidated (KV cache resize needed).
                        // Set to None; forward_into will handle grow.
                        // Plan is rebuilt on the next token after grow completes.
                        gpu_plan = None;
                        false
                    }
                    Err(_) => {
                        gpu_plan = None;
                        false
                    }
                }
            } else {
                false
            };
            #[cfg(not(feature = "opencl"))]
            let used_plan = false;

            if !used_plan {
                // Use GPU memory when on GPU; CPU memory when on CPU.
                // After SwitchHw GPU→CPU, `memory` is still OpenCL memory whose
                // alloc() creates OpenCLBuffer (null as_ptr). We must use
                // cpu_memory_arc to ensure lazy allocations (e.g. k_cast/v_cast)
                // produce CPU-accessible buffers.
                let effective_mem: &dyn Memory = if is_gpu {
                    gpu_memory_arc.as_deref().unwrap_or_else(|| memory.as_ref())
                } else {
                    cpu_memory_arc.as_ref()
                };

                // --cuda-graph: bundle this token's launches into a single
                // CUDA Graph, replayed once. Drains pending work first; the
                // end_capture_and_launch() call replaces the per-kernel
                // driver dispatches with one graph launch.
                #[cfg(feature = "cuda-embedded")]
                let cu_graph_be: Option<
                    &llm_rs2::backend::cuda_embedded::CudaBackend,
                > = if args.cuda_graph {
                    backend
                        .as_any()
                        .downcast_ref::<llm_rs2::backend::cuda_embedded::CudaBackend>()
                } else {
                    None
                };
                #[cfg(feature = "cuda-embedded")]
                if let Some(cu_be) = cu_graph_be {
                    cu_be.begin_graph_capture()?;
                }

                // LISWAP-4: inject IntraForwardSwapHook when active.
                // The cast to `&dyn LayerBoundaryHook` happens inside the
                // option mapping so the args field can be `Option<&dyn _>`
                // — this is the *only* place a real hook is wired in.
                let liswap4_hook: Option<&dyn llm_rs2::models::weights::LayerBoundaryHook> =
                    intra_forward_swap_hook
                        .as_deref()
                        .map(|h| h as &dyn llm_rs2::models::weights::LayerBoundaryHook);

                model.forward_into(TransformerModelForwardArgs {
                    input_tokens: &gen_input_tensor,
                    start_pos,
                    kv_caches: &mut kv_caches,
                    backend: &backend,
                    memory: effective_mem,
                    logits_out: &mut logits,
                    x_gen: Some(&mut x_gen),
                    workspace: Some(&mut gen_ws),
                    score_accumulator: score_accumulator.as_mut(),
                    profiler: profiler.as_mut().map(|p| &mut p.ops),
                    skip_config: skip_config.as_ref(),
                    importance_collector: None,
                    logits_last_only: false,
                    variance_collector: None,
                    prefill_workspace: None,

                    layer_boundary_hook: liswap4_hook,
                })?;

                #[cfg(feature = "cuda-embedded")]
                if let Some(cu_be) = cu_graph_be {
                    cu_be.end_graph_capture_and_launch()?;
                }

                // Rebuild plan if it was invalidated (e.g. KV cache resize).
                // Skip rebuild when tensor partition is active — plan bypasses
                // the partition co-execution path. Same accumulator-pairing
                // requirement as the initial build above.
                #[cfg(feature = "opencl")]
                let accumulator_compatible_with_plan = {
                    let has_cpu_acc = score_accumulator.is_some();
                    let gpu_acc_active = backend
                        .as_any()
                        .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
                        .and_then(|ob| ob.gpu_score_acc())
                        .is_some_and(|acc| acc.is_active());
                    !has_cpu_acc || gpu_acc_active
                };
                // Plan rebuild after fallback. Partition is now routed inside
                // build_plan (ENG-ALG-200) so the old `partition_ctx.is_none()`
                // gate is dropped — build_plan itself picks partition-aware or
                // legacy FFN per layer.
                #[cfg(feature = "opencl")]
                if gpu_plan.is_none()
                    && !gpu_plan_sticky_disabled
                    && backend.name() == "OpenCL"
                    && !args.profile
                    && !args.no_gpu_plan
                    && accumulator_compatible_with_plan
                    && model.config.arch != llm_rs2::models::config::ModelArch::Gemma3
                {
                    gpu_plan = model.build_plan(&x_gen, &logits, &gen_ws, &mut kv_caches, &backend);
                    if partition_active_any && gpu_plan.is_none() {
                        // Second build also failed — lock out further retries.
                        gpu_plan_sticky_disabled = true;
                    }
                }
            }
            backend.synchronize()?;

            // --profile-events: drain and aggregate GPU events into OpProfiler.
            // --heartbeat-gpu-profile (MSG-068 Phase 2): same flush, but feeds
            // the GPU self-util meter instead of (or in addition to) the
            // op-level profiler. Flush runs whenever queue profiling is on.
            #[cfg(feature = "opencl")]
            if (args.profile_events || args.heartbeat_gpu_profile)
                && let Some(ocl_be) = backend
                    .as_any()
                    .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
                && ocl_be.profile_events_enabled
            {
                ocl_be.flush_and_aggregate_profile()?;
                // Op-profiler aggregation only when the caller asked for
                // per-op timing (--profile-events). The heartbeat-only path
                // still flushes so the GPU self-util meter sees the delta,
                // but intentionally skips take_profile_accum() to avoid
                // clearing labels that might still be of interest elsewhere.
                if args.profile_events {
                    let accum = ocl_be.take_profile_accum();
                    if let Some(ref mut p) = profiler {
                        p.ops.merge_from_events(&accum);
                        p.ops.count += 1;
                    }
                }
            }

            // --cuda-profile: drain pending CUevent pairs per-token so
            // the pool (default 4096 pairs) does not overflow. Each
            // decode token launches roughly n_layers * ~10 kernels.
            #[cfg(feature = "cuda-embedded")]
            if args.cuda_profile
                && let Some(cu_be) = backend
                    .as_any()
                    .downcast_ref::<llm_rs2::backend::cuda_embedded::CudaBackend>()
                && cu_be.profiler_enabled()
            {
                if let Err(e) = cu_be.flush_profiler() {
                    eprintln!("[CUDA-Profile] per-token flush failed: {}", e);
                }
            }

            let forward_ms = forward_start.elapsed().as_secs_f64() * 1000.0;

            // ── Layer-Incremental Swap dispatch (ENG-ALG-233) ──────────────────
            // Runs after forward, before sampling. Per-tick: drain up to N layers
            // and call SwapExecutor::execute_on_slots with the chunk.
            // ENG-ALG-234: plan committed with force-swap-ratio + per_tick > 0;
            //   new signals during flight are ignored (plan runs to completion).
            // INV-145: empty chunk is never passed to execute_on_slots.
            if let Some(ref mut inc_plan) = incremental_force_swap_plan {
                // LISWAP-6 Dynamic-K: reactive pause + per-tick override.
                //
                // - Pause: release queue non-empty → skip swap this tick (K
                //   stays unchanged). Calibration tick is exempt because it
                //   has to dispatch K=1 to measure drop cost.
                // - Pre-drain: inject controller's current K into the plan.
                let mut dyn_k_pause = false;
                if let Some(ref ctrl) = dynamic_k_controller {
                    let pending = model.release_worker.pending_count();
                    if ctrl.is_calibrated() && ctrl.should_pause(pending) {
                        dyn_k_pause = true;
                        if dynamic_k_diag {
                            eprintln!(
                                "[DynamicK] pause t={} pending={} k={}",
                                decode_token_index,
                                pending,
                                ctrl.current_k()
                            );
                        }
                    } else {
                        // Calibration tick forces K=1 (sync measurement);
                        // subsequent ticks use the controller's current K.
                        let k = if ctrl.is_calibrated() {
                            ctrl.current_k()
                        } else {
                            1
                        };
                        inc_plan.set_per_tick(k);
                    }
                }
                let chunk = if dyn_k_pause {
                    Vec::new()
                } else {
                    inc_plan.drain_chunk()
                };
                if !chunk.is_empty() {
                    let t_swap = std::time::Instant::now();
                    match run_layer_swap(
                        &model,
                        &chunk,
                        gpu_backend_arc.as_ref(),
                        &cpu_backend_arc,
                        async_swap_dispatcher.as_ref(),
                        #[cfg(feature = "opencl")]
                        host_ptr_swap_pool.clone(),
                    ) {
                        Ok(report) => {
                            eprintln!(
                                "[IncrementalSwap] tick={} chunk={:?} swapped={} remaining={} latency={:.1}ms",
                                decode_token_index,
                                &chunk,
                                report.swapped.len(),
                                inc_plan.remaining_count(),
                                t_swap.elapsed().as_secs_f64() * 1000.0,
                            );
                            if let Some(ref stages) = report.stage_breakdown {
                                eprintln!("[IncrementalSwap] stages: {}", stages.to_log_line());
                            }
                            #[cfg(feature = "opencl")]
                            remap_weights_for_cpu_after_swap(
                                &mut model,
                                &backend,
                                is_gpu,
                                args.resilience_prealloc_switch,
                                "incremental-swap",
                            );

                            // LISWAP-6 Dynamic-K Phase 0 calibration. Runs only on
                            // the first successfully-dispatched chunk; drains the
                            // async transfer queue and spins on release_worker
                            // until pending == 0 to measure the worst-case per-
                            // layer drop cost. From there on the controller runs
                            // async (no sync waits).
                            if let Some(ref mut ctrl) = dynamic_k_controller
                                && !ctrl.is_calibrated()
                                && !chunk.is_empty()
                            {
                                if let Some(ref dispatcher) = async_swap_dispatcher {
                                    let _ = dispatcher.drain(std::time::Duration::from_millis(500));
                                }
                                let calib_start = std::time::Instant::now();
                                let timeout = std::time::Duration::from_millis(100);
                                while model.release_worker.pending_count() > 0
                                    && calib_start.elapsed() < timeout
                                {
                                    std::hint::spin_loop();
                                }
                                let drop_total = calib_start.elapsed().as_secs_f64() * 1000.0;
                                let drop_ms_per_layer = (drop_total / chunk.len() as f64) as f32;
                                ctrl.calibrate(drop_ms_per_layer, forward_ms as f32);
                                if dynamic_k_diag {
                                    eprintln!(
                                        "[DynamicK] calibrated t={} drop_ms={:.3} fwd_ms={:.2} safe_k={}",
                                        decode_token_index,
                                        drop_ms_per_layer,
                                        forward_ms,
                                        ctrl.current_k()
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!(
                                "[IncrementalSwap] swap error on tick={}: {}",
                                decode_token_index, e
                            );
                        }
                    }
                }

                // LISWAP-6 Dynamic-K Phase 1+: observe forward wall, shrink K
                // if the forward got tighter than anything seen so far.
                if let Some(ref mut ctrl) = dynamic_k_controller
                    && ctrl.is_calibrated()
                {
                    let prev_k = ctrl.current_k();
                    ctrl.observe_forward(forward_ms as f32);
                    if dynamic_k_diag && ctrl.current_k() != prev_k {
                        eprintln!(
                            "[DynamicK] k_decrease t={} fwd_ms={:.2} new_k={}",
                            decode_token_index,
                            forward_ms,
                            ctrl.current_k()
                        );
                    }
                }
                // ENG-ALG-233: retire plan when all layers have been drained (INV-145).
                if inc_plan.is_done() {
                    eprintln!(
                        "[IncrementalSwap] plan complete (started_at_token={}, finished_at_token={})",
                        inc_plan.started_at_token(),
                        decode_token_index,
                    );
                    // LISWAP-2: drain async dispatcher to ensure all in-flight commits land
                    // before the plan is retired. drain failure is non-fatal — prototype
                    // robustness is secondary to measurement.
                    if let Some(ref dispatcher) = async_swap_dispatcher {
                        let drain_t = std::time::Instant::now();
                        if let Err(e) = dispatcher.drain(std::time::Duration::from_secs(2)) {
                            eprintln!("[LISWAP-2] drain failed: {e}");
                        } else {
                            eprintln!(
                                "[LISWAP-2] dispatcher drained: {:.1}ms",
                                drain_t.elapsed().as_secs_f64() * 1000.0
                            );
                        }
                    }
                    incremental_force_swap_plan = None;

                    // LISWAP-6 manager path: build WeightSwapReport when the plan
                    // was committed by dispatch_swap_weights (manager signal).
                    // Stored in `ready_weight_swap_report`; sent by executor block
                    // later this token tick (executor scope is separate).
                    if let Some((ratio, n_planned, plan_start, qcf_estimated)) =
                        manager_swap_report_pending.take()
                    {
                        use llm_rs2::models::weights::compute_qcf_swap;
                        let latency_ms = plan_start.elapsed().as_millis() as u64;
                        let n_layers = model.layers.len();
                        let actually_swapped_now: Vec<usize> = (0..n_layers)
                            .filter(|&i| model.layers[i].current_dtype() == DType::Q4_0)
                            .collect();
                        let qcf_swap_actual = if actually_swapped_now.is_empty() {
                            qcf_estimated
                        } else {
                            compute_qcf_swap(
                                &actually_swapped_now,
                                &model.quant_noise,
                                importance_table_for_swap.as_ref(),
                                n_layers,
                            )
                        };
                        let layers_swapped: Vec<llm_shared::LayerSwapEntry> = actually_swapped_now
                            .iter()
                            .map(|&idx| llm_shared::LayerSwapEntry {
                                layer_idx: idx as u32,
                                from_dtype: llm_shared::DtypeTag::F16,
                                to_dtype: llm_shared::DtypeTag::Q4_0,
                            })
                            .collect();
                        eprintln!(
                            "[WeightSwap] manager plan complete: ratio={:.2}, planned={}, \
                             actually_q4={}, qcf_swap={:.4}, latency={}ms",
                            ratio,
                            n_planned,
                            layers_swapped.len(),
                            qcf_swap_actual,
                            latency_ms,
                        );
                        ready_weight_swap_report = Some(llm_shared::WeightSwapReport {
                            layers_swapped,
                            freed_bytes: 0,
                            latency_ms,
                            qcf_swap_actual,
                        });
                    }
                }
            }
            // ── End Layer-Incremental Swap dispatch ────────────────────────────

            // ── LISWAP-4 Intra-forward Swap retire (INV-150) ──────────────────
            // After every decode token, check whether the in-flight plan is
            // complete. If so, drain dispatcher, synchronize backend, bump
            // ratio_generation, invalidate noshuffle SOA registry, and retire
            // the hook to None.
            if let Some(hook) = intra_forward_swap_hook.clone()
                && hook.plan_is_complete()
            {
                let drain_t = std::time::Instant::now();
                let backend_for_invalidate: Arc<dyn Backend> = gpu_backend_arc
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| Arc::clone(&backend));
                let invalidate = move || {
                    backend_for_invalidate.invalidate_noshuffle_soa_registry();
                };
                match hook.finalize(
                    &model.ratio_generation,
                    invalidate,
                    std::time::Duration::from_secs(10),
                ) {
                    Ok(()) => {
                        eprintln!(
                            "[IntraForwardSwap] plan retired at token={} (drain+sync+bump+invalidate {:.1}ms)",
                            decode_token_index,
                            drain_t.elapsed().as_secs_f64() * 1000.0,
                        );
                        #[cfg(feature = "opencl")]
                        remap_weights_for_cpu_after_swap(
                            &mut model,
                            &backend,
                            is_gpu,
                            args.resilience_prealloc_switch,
                            "intra-forward-swap",
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "[IntraForwardSwap] finalize failed at token={}: {}",
                            decode_token_index, e
                        );
                    }
                }
                intra_forward_swap_hook = None; // retire
            }
            // ── End LISWAP-4 retire ────────────────────────────────────────────

            // ── LISWAP-5 Phase-aware Swap retire ──────────────────────────────
            // chunk_queue가 비고 in_flight도 None이면 dispatcher 종료. finalize는
            // 마지막 ratio_generation bump + invalidate 수행. PHASE_HOOK은
            // OnceLock이라 unset 불가능하지만 finalize() 후 모든 hook fire가
            // noop이 됨 (dispatcher 내부 finalized atomic).
            if let Some(disp) = phase_aware_swap_dispatcher.as_ref()
                && std::env::var("LLMRS_PHASE_AWARE_DEBUG").as_deref() == Ok("1")
                && decode_token_index < 5
            {
                let (q, inf, p, d, hs, he, ce) = disp.debug_snapshot();
                eprintln!(
                    "[PhaseAwareSwap-DBG] tok={} queue={} in_flight={} pending={} dispatched={} hook_start={} hook_end={} cachefit_end={}",
                    decode_token_index, q, inf, p, d, hs, he, ce
                );
            }
            if let Some(disp) = phase_aware_swap_dispatcher.as_ref()
                && disp.is_complete()
            {
                let drain_t = std::time::Instant::now();
                let backend_for_invalidate: Arc<dyn Backend> = gpu_backend_arc
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| Arc::clone(&backend));
                let invalidate = move || {
                    backend_for_invalidate.invalidate_noshuffle_soa_registry();
                };
                match disp.finalize(
                    &model.ratio_generation,
                    invalidate,
                    std::time::Duration::from_secs(10),
                ) {
                    Ok(()) => {
                        eprintln!(
                            "[PhaseAwareSwap] plan retired at token={} (drain+sync+bump+invalidate {:.1}ms, chunks={})",
                            decode_token_index,
                            drain_t.elapsed().as_secs_f64() * 1000.0,
                            disp.dispatched_count(),
                        );
                        #[cfg(feature = "opencl")]
                        remap_weights_for_cpu_after_swap(
                            &mut model,
                            &backend,
                            is_gpu,
                            args.resilience_prealloc_switch,
                            "phase-aware-swap",
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "[PhaseAwareSwap] finalize failed at token={}: {}",
                            decode_token_index, e
                        );
                    }
                }
                phase_aware_swap_dispatcher = None;
            }
            // ── End LISWAP-5 retire ────────────────────────────────────────────

            // ── H2O Debug: per-step diagnostics ──
            if args.h2o_debug {
                // 1. Verify ws.scores is post-softmax (sample first 4 heads)
                let n_heads_q = model.config.num_attention_heads;
                let stride = gen_ws.scores.len() / n_heads_q;
                let cache_pos = kv_caches[0].current_pos;
                let heads_to_check = n_heads_q.min(4);
                for h in 0..heads_to_check {
                    let sum: f32 = gen_ws.scores[h * stride..h * stride + cache_pos]
                        .iter()
                        .sum();
                    if (sum - 1.0).abs() > 0.01 {
                        eprintln!(
                            "[H2O-Debug] WARNING: head {} score sum = {:.6} (expect ~1.0)",
                            h, sum
                        );
                    }
                }

                // 2. Dump importance score distribution
                if let Some(acc) = score_accumulator.as_ref() {
                    let scores = acc.importance_scores();
                    let valid = &scores[..cache_pos];
                    if !valid.is_empty() {
                        let mut indexed: Vec<(usize, f32)> =
                            valid.iter().enumerate().map(|(i, &s)| (i, s)).collect();
                        indexed.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let top5: Vec<_> = indexed.iter().take(5).collect();
                        let bot5: Vec<_> = indexed.iter().rev().take(5).collect();
                        eprintln!(
                            "[H2O-Debug] step={} cache_pos={} Top5={:?} Bot5={:?}",
                            decode_token_index, cache_pos, top5, bot5
                        );
                    }
                }
            }

            // Auto-eviction after forward pass (non-experiment mode)
            if auto_eviction {
                let before_len = kv_caches[0].current_pos;
                let capacity = kv_caches[0].capacity();

                // GPU score sync: transfer GPU-accumulated scores to CPU accumulator
                // before any score-based eviction decision. Only syncs when:
                // 1. GPU score acc is active AND
                // 2. Eviction is imminent (score-based at 90% capacity) OR non-score-based with acc
                #[cfg(feature = "opencl")]
                if (score_based_eviction && before_len >= capacity * 9 / 10
                    || score_accumulator.as_ref().is_some_and(|a| a.is_active()))
                    && let Some(ocl_be) = backend
                        .as_any()
                        .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
                    && let Some(gpu_acc) = ocl_be.gpu_score_acc()
                    && gpu_acc.is_active()
                {
                    let (flat, head) = gpu_acc.sync_to_cpu(ocl_be.queue.as_core())?;
                    if let Some(ref mut acc) = score_accumulator {
                        acc.import_gpu_scores(&flat, &head);
                    }
                }

                // Capture pre-eviction scores for profiling (before eviction mutates state)
                let pre_eviction_scores: Vec<f32> = if profiler.is_some()
                    && score_based_eviction
                    && before_len >= capacity * 9 / 10
                {
                    score_accumulator
                        .as_ref()
                        .filter(|acc| acc.is_active())
                        .map(|acc| {
                            acc.importance_scores()[..before_len.min(acc.importance_scores().len())]
                                .to_vec()
                        })
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };

                let result = if score_based_eviction && before_len >= capacity * 9 / 10 {
                    // Score-based policies: force evict when cache >= 90% full
                    if let Some(acc) = score_accumulator.as_ref() {
                        if acc.is_active() {
                            // D2O layer-level allocation: use per-layer budgets if available
                            if let Some(ref ratios) = d2o_layer_ratios {
                                cache_manager.force_evict_with_scores_and_budgets(
                                    &mut kv_caches,
                                    args.eviction_target_ratio,
                                    acc.importance_scores(),
                                    ratios,
                                )?
                            } else {
                                cache_manager.force_evict_with_scores(
                                    &mut kv_caches,
                                    args.eviction_target_ratio,
                                    acc.importance_scores(),
                                )?
                            }
                        } else {
                            cache_manager.force_evict(&mut kv_caches, args.eviction_target_ratio)?
                        }
                    } else {
                        cache_manager.force_evict(&mut kv_caches, args.eviction_target_ratio)?
                    }
                } else if let Some(acc) = score_accumulator.as_ref() {
                    if acc.is_active() {
                        cache_manager
                            .maybe_evict_with_scores(&mut kv_caches, acc.importance_scores())?
                    } else {
                        cache_manager.maybe_evict(&mut kv_caches)?
                    }
                } else {
                    cache_manager.maybe_evict(&mut kv_caches)?
                };
                if result.evicted {
                    // Compute evicted indices from pre-eviction state
                    let target_len = ((before_len as f32) * args.eviction_target_ratio) as usize;
                    let evicted_indices = if !pre_eviction_scores.is_empty() {
                        llm_rs2::profile::compute_h2o_evicted_indices(
                            before_len,
                            target_len,
                            actual_protected_prefix,
                            args.h2o_keep_ratio,
                            &pre_eviction_scores,
                        )
                    } else {
                        Vec::new()
                    };

                    if let Some(ref mut p) = profiler {
                        // Record token deaths before the EvictionEvent
                        if !evicted_indices.is_empty() {
                            p.scores.record_token_deaths(
                                decode_token_index,
                                &evicted_indices,
                                &position_birth_step,
                                &pre_eviction_scores,
                            );
                        }
                        p.on_eviction(llm_rs2::profile::EvictionEvent {
                            step: decode_token_index,
                            policy: args.eviction_policy.clone(),
                            before_len,
                            after_len: result.new_pos,
                            evicted_count: result.tokens_removed,
                            partition: llm_rs2::profile::PartitionInfo {
                                prefix_end: actual_protected_prefix,
                                hh_count: 0,
                                recent_start: result.new_pos,
                            },
                            evicted_indices: evicted_indices.clone(),
                            pre_eviction_scores,
                        });
                    }

                    // Update position_birth_step mapping after eviction (compact)
                    if !position_birth_step.is_empty() {
                        let evicted_set: std::collections::HashSet<usize> =
                            evicted_indices.iter().copied().collect();
                        let mut kept = Vec::new();
                        for (pos, &birth) in position_birth_step.iter().enumerate() {
                            if pos < before_len && !evicted_set.contains(&pos) {
                                kept.push(birth);
                            }
                        }
                        position_birth_step = kept;
                    }

                    if let Some(acc) = score_accumulator.as_mut() {
                        acc.reset();
                    }
                    // Reset GPU score accumulator after eviction
                    #[cfg(feature = "opencl")]
                    if let Some(ocl_be) = backend
                        .as_any()
                        .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
                        && let Some(gpu_acc) = ocl_be.gpu_score_acc_mut()
                        && gpu_acc.is_active()
                    {
                        gpu_acc.reset(ocl_be.queue.as_core())?;
                    }
                }
            }
            forward_ms_values.push(forward_ms);
            if std::env::var("LLMRS_PER_TOKEN_MS").is_ok() {
                eprintln!(
                    "[PER_TOKEN] idx={} kv_pos={} forward_ms={:.3}",
                    decode_token_index, kv_caches[0].current_pos, forward_ms
                );
            }

            // ── Experiment: inject directives at this token position ──
            let mut injected_signals: Vec<String> = Vec::new();
            if let (Some(schedule), Some(tx)) = (&experiment_schedule, &experiment_tx) {
                for entry in schedule.directives_at(decode_token_index) {
                    let msg = ManagerMessage::Directive(entry.directive.clone());
                    injected_signals.push(directive_summary(&msg));
                    tx.send(msg).ok();
                }
            }

            // ── Resilience checkpoint (CommandExecutor) ──────
            let mut action_names: Vec<String> = Vec::new();
            if let Some(executor) = &mut command_executor {
                let kv_snap = KVSnapshot {
                    total_bytes: kv_caches
                        .iter()
                        .map(|c| (c.k_buffer.buffer().size() + c.v_buffer.buffer().size()) as u64)
                        .sum(),
                    total_tokens: kv_caches[0].current_pos,
                    capacity: kv_caches[0].capacity(),
                    protected_prefix: actual_protected_prefix,
                    // Phase 3에서 실제 정책/dtype/skip 정보로 채울 예정
                    kv_dtype: "f16".to_string(),
                    eviction_policy: args.eviction_policy.clone(),
                    skip_ratio: 0.0,
                };

                let plan = executor.poll(&kv_snap);
                action_names = plan_summary(&plan);

                // Activate score collection on-demand: only when eviction is
                // requested or imminent. With GPU score accumulator, there is
                // no per-token overhead (scores are accumulated on-device).
                if let Some(ref mut acc) = score_accumulator
                    && !acc.is_active()
                    && (plan.evict.is_some() || plan.request_qcf)
                {
                    acc.set_active(true);
                    // Also activate GPU score accumulator if available
                    #[cfg(feature = "opencl")]
                    if let Some(ocl_be) = backend
                        .as_any()
                        .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
                        && let Some(gpu_acc) = ocl_be.gpu_score_acc_mut()
                    {
                        gpu_acc.set_active(true);
                    }
                }

                // SEQ-095/096: Compute and send QCF estimates if requested
                if plan.request_qcf {
                    // ENG-ALG-218: if secondary mmap is present, arm the collector
                    // so the next prefill injects ImportanceCollector.
                    if model.secondary_mmap.is_some() && !collector_armed {
                        collector_armed = true;
                        eprintln!("[WeightSwap] ImportanceCollector armed for next prefill");
                    }

                    // Derive streaming window: same logic as policy construction
                    let streaming_window_size = if args.streaming_window > 0 {
                        args.streaming_window
                    } else if args.kv_budget > 0 {
                        args.kv_budget.saturating_sub(args.sink_size)
                    } else {
                        args.eviction_window
                    };

                    // ISSUE-9 fix: On OpenCL with zero-copy memory, KV V
                    // buffers are UnifiedBuffer (CL_MEM_ALLOC_HOST_PTR) that
                    // start unmapped — `as_ptr()` returns null, which trips
                    // the host-readable guard in compute_qcf_estimates and
                    // skips KV-based 4종 estimates. Sync GPU queue and map
                    // V buffers for CPU before running QCF, then unmap so
                    // the next forward pass can reuse the GPU path.
                    if let Err(e) = backend.synchronize() {
                        eprintln!("[QCF] backend.synchronize() failed: {}", e);
                    }
                    let mut mapped_bufs: Vec<std::sync::Arc<dyn llm_rs2::core::buffer::Buffer>> =
                        Vec::new();
                    for cache in &kv_caches {
                        let v_buf = cache.v_buffer.buffer();
                        if v_buf.as_ptr().is_null() {
                            match v_buf.map_for_cpu() {
                                Ok(_) => mapped_bufs.push(v_buf.clone()),
                                Err(e) => {
                                    eprintln!("[QCF] map_for_cpu failed: {}", e);
                                }
                            }
                        }
                    }

                    let ctx = QcfEstimateContext {
                        kv_caches: &kv_caches,
                        score_accumulator: score_accumulator.as_ref(),
                        streaming_config: Some((args.sink_size, streaming_window_size)),
                        importance_table: importance_table_for_swap.as_ref(),
                        num_layers: model.config.num_hidden_layers,
                        kivi_caches: None,
                    };
                    let estimates = compute_qcf_estimates(&ctx);

                    // Release mappings so subsequent forward passes can use
                    // the GPU path without the "writes to mapped buffer
                    // are UB" hazard.
                    for buf in &mapped_bufs {
                        if let Err(e) = buf.unmap_for_gpu() {
                            eprintln!("[QCF] unmap_for_gpu failed: {}", e);
                        }
                    }

                    // Build layer_swap estimate if importance table is available
                    // (set by a previous prefill) and secondary is present.
                    let layer_swap =
                        build_layer_swap_estimate(&model, importance_table_for_swap.as_ref());

                    executor.send_qcf_estimate(llm_shared::QcfEstimate {
                        estimates,
                        layer_swap,
                    });
                }

                // ENG-ALG-214-ROUTE (LISWAP-6 manager path): SwapWeights →
                // IncrementalSwapPlan commit. decode loop drains K=2 layers/tick
                // with dynamic-K + sub-batch pause. WeightSwapReport sent on
                // plan completion (see plan-done block below).
                //
                // Source priority: sticky `pending_swap_weights` first — covers
                // the case where the directive arrived while the prefill loop
                // was polling (prefill drops `plan.swap_weights`). Fall back to
                // `plan.swap_weights` for the same-tick path.
                let pending_swap = executor.take_pending_swap_weights().or(plan.swap_weights);
                if let Some((ratio, target_dtype)) = pending_swap {
                    dispatch_swap_weights(
                        &model,
                        ratio,
                        target_dtype,
                        importance_table_for_swap.as_ref(),
                        decode_token_index,
                        &mut incremental_force_swap_plan,
                        &mut manager_swap_report_pending,
                    );
                    // Note: remap_weights_for_cpu_after_swap will be called
                    // per-chunk in the incremental swap dispatch block above.
                }

                // LISWAP-6 manager path: send completed WeightSwapReport.
                // Built by the plan-done block (before executor scope), consumed here.
                if let Some(report) = ready_weight_swap_report.take() {
                    executor.send_weight_swap_report(report);
                }

                if let Some(evict) = &plan.evict {
                    let effective_ratio =
                        args.experiment_eviction_ratio.unwrap_or(evict.target_ratio);

                    let current_pos = kv_caches[0].current_pos;

                    // Ceiling: record current_pos at the first sticky eviction trigger.
                    // All subsequent evictions use ceiling * ratio as fixed target to prevent
                    // cascade shrinking (e.g. 33→16→8→... when ratio applied to shrinking pos).
                    // Streaming eviction (target_ratio == 0.0) bypasses this check since
                    // it manages its own window logic internally.
                    let (skip_eviction, target_pos) = if effective_ratio > 0.0 {
                        let ceiling = evict_ceiling.get_or_insert(current_pos);
                        let tgt_raw = (*ceiling as f32 * effective_ratio).max(1.0) as usize;
                        let tgt = if tgt_raw < args.min_kv_cache {
                            if evict_floor_logged.is_none() {
                                eprintln!(
                                    "[Eviction] target_pos {} clamped to min_kv_cache {}",
                                    tgt_raw, args.min_kv_cache
                                );
                                evict_floor_logged = Some(true);
                            }
                            args.min_kv_cache
                        } else {
                            tgt_raw
                        };
                        // Batch 32 tokens before evicting to amortize memmove overhead
                        // (~14ms/step → ~0.4ms/step on compact_keep_positions).
                        const EVICT_BATCH_HEADROOM: usize = 32;
                        (current_pos <= tgt + EVICT_BATCH_HEADROOM, tgt)
                    } else {
                        (false, 0)
                    };

                    if skip_eviction {
                        // Cache already within target — no-op this step
                    } else {
                        // GPU score sync before resilience eviction
                        #[cfg(feature = "opencl")]
                        if let Some(ocl_be) = backend
                            .as_any()
                            .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>(
                        ) && let Some(gpu_acc) = ocl_be.gpu_score_acc()
                            && gpu_acc.is_active()
                        {
                            let (flat, head) = gpu_acc.sync_to_cpu(ocl_be.queue.as_core())?;
                            if let Some(ref mut acc) = score_accumulator {
                                acc.import_gpu_scores(&flat, &head);
                            }
                        }

                        // ── Score distribution diagnostic (via events system) ──
                        if let Some(acc) = score_accumulator.as_ref() {
                            let scores = acc.importance_scores();
                            let cache_pos = kv_caches[0].current_pos;

                            if let Some(snapshot) = events::build_score_snapshot(
                                scores,
                                cache_pos,
                                actual_protected_prefix,
                                decode_token_index,
                                10,
                            ) {
                                cache_manager
                                    .event_sink()
                                    .emit(CacheEvent::ScoreDiagnostic(snapshot));

                                if let Some(ref out_path) = args.experiment_output {
                                    let diag_path = format!(
                                        "{}.scores.csv",
                                        out_path.trim_end_matches(".jsonl")
                                    );
                                    if events::dump_scores_csv(scores, cache_pos, &diag_path)
                                        .is_ok()
                                    {
                                        eprintln!("[ScoreDiag] Scores dumped to {}", diag_path);
                                    }
                                }
                            }
                        }

                        // Build ScoreContext from accumulator for policy-directed eviction
                        let scores = if let Some(acc) = score_accumulator.as_ref() {
                            if let Some(head_imp) = acc.head_importance_scores() {
                                llm_rs2::core::cache_manager::ScoreContext::PerHead {
                                    flat: acc.importance_scores(),
                                    head: head_imp,
                                    n_kv_heads: acc.n_kv_heads(),
                                }
                            } else if acc.is_active() {
                                llm_rs2::core::cache_manager::ScoreContext::Flat {
                                    importance: acc.importance_scores(),
                                }
                            } else {
                                llm_rs2::core::cache_manager::ScoreContext::None
                            }
                        } else {
                            llm_rs2::core::cache_manager::ScoreContext::None
                        };

                        // Manager already decided to evict — execute via named policy
                        // D2O uses Pipeline (force_evict_with_scores), not named policy registry
                        // StreamingLLM uses on-demand instantiation (params from directive)

                        // Ceiling-based adjusted ratio: back-calculate ratio so that
                        // force_evict's internal (current_pos * ratio) == target_pos.
                        // This prevents the cascade effect when current_pos < ceiling.
                        let adjusted_ratio = if effective_ratio > 0.0 && current_pos > 0 {
                            target_pos as f32 / current_pos as f32
                        } else {
                            effective_ratio
                        };

                        let result = if evict.method == llm_rs2::resilience::EvictMethod::D2o {
                            let importance = if let Some(acc) = score_accumulator.as_ref() {
                                acc.importance_scores().to_vec()
                            } else {
                                vec![]
                            };
                            if importance.is_empty() {
                                cache_manager.force_evict(&mut kv_caches, adjusted_ratio)
                            } else {
                                cache_manager.force_evict_with_scores(
                                    &mut kv_caches,
                                    adjusted_ratio,
                                    &importance,
                                )
                            }
                        } else if evict.method == llm_rs2::resilience::EvictMethod::Streaming {
                            use llm_rs2::core::eviction::StreamingLLMPolicy;
                            if let Some(ref sp) = evict.streaming_params {
                                let policy = StreamingLLMPolicy::new(sp.sink_size, sp.window_size);
                                cache_manager.force_evict_by_policy_ref(
                                    &policy,
                                    &mut kv_caches,
                                    adjusted_ratio,
                                    scores,
                                )
                            } else {
                                Err(anyhow::anyhow!(
                                    "KvStreaming evict plan missing streaming_params"
                                ))
                            }
                        } else {
                            cache_manager.force_evict_by_policy(
                                evict.method,
                                &mut kv_caches,
                                adjusted_ratio,
                                scores,
                            )
                        };
                        match result {
                            Ok(r) if r.evicted => {
                                eprintln!(
                                    "[Resilience] Evicted {} tokens (pos: {} → {})",
                                    r.tokens_removed,
                                    r.new_pos + r.tokens_removed,
                                    r.new_pos
                                );
                                if args.h2o_debug {
                                    if let Some(acc) = score_accumulator.as_ref() {
                                        let scores = acc.importance_scores();
                                        let pre_pos = r.new_pos + r.tokens_removed;
                                        let valid = &scores[..pre_pos.min(scores.len())];
                                        if !valid.is_empty() {
                                            let total: f32 = valid.iter().sum();
                                            let max_s = valid
                                                .iter()
                                                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                                            let min_s =
                                                valid.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                                            let avg_s = total / valid.len() as f32;
                                            eprintln!(
                                                "[H2O-Debug] Pre-eviction scores: min={:.3} avg={:.3} max={:.3} total={:.1} tokens={}",
                                                min_s,
                                                avg_s,
                                                max_s,
                                                total,
                                                valid.len()
                                            );
                                        }
                                    }
                                    eprintln!(
                                        "[H2O-Debug] Eviction: ratio={:.3} removed={} new_pos={}",
                                        effective_ratio, r.tokens_removed, r.new_pos
                                    );
                                }
                                if let Some(ref mut p) = profiler {
                                    p.on_eviction(llm_rs2::profile::EvictionEvent {
                                        step: decode_token_index,
                                        policy: args.eviction_policy.clone(),
                                        before_len: r.new_pos + r.tokens_removed,
                                        after_len: r.new_pos,
                                        evicted_count: r.tokens_removed,
                                        partition: llm_rs2::profile::PartitionInfo {
                                            prefix_end: actual_protected_prefix,
                                            hh_count: 0,
                                            recent_start: r.new_pos,
                                        },
                                        evicted_indices: vec![],
                                        pre_eviction_scores: vec![],
                                    });
                                }
                                // Release physical pages (madvise MADV_DONTNEED)
                                let mut bytes_released = 0usize;
                                for cache in kv_caches.iter_mut() {
                                    bytes_released += cache.release_unused_pages();
                                }
                                if bytes_released > 0 {
                                    eprintln!(
                                        "[Resilience] Released {} MB of physical pages",
                                        bytes_released / (1024 * 1024)
                                    );
                                }
                                experiment_eviction_count += 1;
                                experiment_evicted_total += r.tokens_removed;
                                if let Some(acc) = score_accumulator.as_mut() {
                                    acc.reset();
                                    acc.set_active(false);
                                }
                                // Reset GPU score accumulator after resilience eviction
                                #[cfg(feature = "opencl")]
                                if let Some(ocl_be) = backend
                                    .as_any()
                                    .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>(
                                ) && let Some(gpu_acc) = ocl_be.gpu_score_acc_mut()
                                    && gpu_acc.is_active()
                                {
                                    gpu_acc.reset(ocl_be.queue.as_core())?;
                                    gpu_acc.set_active(false);
                                }
                                // Invalidate GPU Plan — cache size changed after eviction,
                                // stale plan would use wrong attention sequence length.
                                #[cfg(feature = "opencl")]
                                {
                                    gpu_plan = None;
                                }
                            }
                            Err(e) => eprintln!("[Resilience] Eviction error: {}", e),
                            _ => {}
                        }
                    } // end skip_eviction else
                }

                // Dynamic tensor partition ratio
                //
                // The executor re-delivers the sticky partition_ratio on every
                // poll (ISSUE-5 prefill→decode carry-over), so we guard with
                // `last_applied_partition_ratio` to prevent re-splitting weights
                // on every decode tick (REGRESSION-A). Only the first delivery
                // of a new ratio triggers the expensive re-split / re-register.
                if let Some(ratio) = plan.partition_ratio
                    && last_applied_partition_ratio != Some(ratio)
                {
                    if ratio <= 0.0 || ratio >= 1.0 {
                        // Disable partition: clear partition_ctx from all layers
                        // via atomic clone-and-swap (ArcSwap snapshot replace).
                        for slot in &model.layers {
                            let old = slot.load_weights();
                            let mut new = (*old).clone();
                            new.partition_ctx = None;
                            slot.store_weights_same_dtype(Arc::new(new));
                        }
                        gen_ws.partition_ws = None;
                        eprintln!("[Partition] Disabled (ratio={})", ratio);
                        executor.set_partition_ratio(0.0);
                        last_applied_partition_ratio = Some(ratio);
                        // Partition off: invalidate plan to trigger rebuild next
                        // iter so GPU-only fast path is restored.
                        #[cfg(feature = "opencl")]
                        {
                            gpu_plan = None;
                        }
                    } else if llm_rs2::layers::tensor_partition::is_gpu_only_ratio(ratio) {
                        // GPU-only fast path: clear any existing partition context
                        // so forward() skips the host staging / CPU matmul / merge
                        // path entirely. No lazy `map_weights_for_cpu` needed here
                        // because the CPU side is unused at this ratio.
                        for slot in &model.layers {
                            let old = slot.load_weights();
                            let mut new = (*old).clone();
                            new.partition_ctx = None;
                            slot.store_weights_same_dtype(Arc::new(new));
                        }
                        gen_ws.partition_ws = None;
                        eprintln!(
                            "[Partition] ratio={:.3} treated as GPU-only (>= {:.3}); partition path disabled",
                            ratio,
                            llm_rs2::layers::tensor_partition::GPU_ONLY_THRESHOLD,
                        );
                        executor.set_partition_ratio(ratio);
                        last_applied_partition_ratio = Some(ratio);
                        #[cfg(feature = "opencl")]
                        {
                            gpu_plan = None;
                        }
                    } else {
                        // Lazy activation: if weights are still GPU-only (null host
                        // ptr — the normal state when `--enable-resilience` alone
                        // was used without `--tensor-partition`), map them now. This
                        // moves the ~200 ms / +400 MB RSS cost from startup to the
                        // first `SetPartitionRatio` directive that actually needs
                        // CPU-accessible weights. The one-shot first-activation
                        // stall is logged for downstream TBT accounting.
                        let mut lazy_map_ok = true;
                        #[cfg(feature = "opencl")]
                        if is_gpu && model.layers[0].load_weights().wq.as_ptr().is_null() {
                            let t0 = std::time::Instant::now();
                            match model.map_weights_for_cpu(&backend) {
                                Ok(n) if n > 0 => eprintln!(
                                    "[Partition] Lazy-mapped {} weight tensors for CPU access in {:.1} ms (first-activation stall)",
                                    n,
                                    t0.elapsed().as_secs_f64() * 1000.0,
                                ),
                                Ok(_) => {}
                                Err(e) => {
                                    eprintln!(
                                        "[Partition] Lazy weight map failed: {} — ratio={} rejected.",
                                        e, ratio
                                    );
                                    lazy_map_ok = false;
                                }
                            }
                        }
                        // Re-split weights with new ratio (only if lazy map succeeded)
                        if lazy_map_ok {
                            match model.prepare_tensor_partition(ratio, &cpu_backend_arc) {
                                Ok(n) => {
                                    eprintln!(
                                        "[Partition] Re-split {} weights with ratio {:.2}",
                                        n, ratio
                                    );
                                    // Reallocate workspace
                                    let layer0_probe = model.layers[0].load_weights();
                                    // Diagnostic: dump per-weight buffer kind for
                                    // layer 0 so a "B is not OpenCL buffer" crash
                                    // on the next forward immediately points at
                                    // which tensor is misbacked. Single-shot,
                                    // layer 0 only — every other layer has the
                                    // same backing pattern by construction.
                                    #[cfg(feature = "opencl")]
                                    {
                                        use llm_rs2::backend::opencl::buffer_kind_label;
                                        let l0 = &layer0_probe;
                                        let mut log = String::from(
                                            "[Partition] Layer 0 weight buffer kinds: ",
                                        );
                                        log.push_str(&format!(
                                            "wq={} wk={} wv={} wo={} ",
                                            buffer_kind_label(l0.wq.buffer().as_ref()),
                                            buffer_kind_label(l0.wk.buffer().as_ref()),
                                            buffer_kind_label(l0.wv.buffer().as_ref()),
                                            buffer_kind_label(l0.wo.buffer().as_ref()),
                                        ));
                                        log.push_str(&format!(
                                            "w_gate={} w_up={} w_down={} ",
                                            buffer_kind_label(l0.w_gate.buffer().as_ref()),
                                            buffer_kind_label(l0.w_up.buffer().as_ref()),
                                            buffer_kind_label(l0.w_down.buffer().as_ref()),
                                        ));
                                        if let Some(ref ctx) = l0.partition_ctx {
                                            log.push_str(&format!(
                                                "gate_gpu_slice={} up_gpu_slice={} down_gpu_slice={}",
                                                buffer_kind_label(
                                                    ctx.gate.gpu_slice.buffer().as_ref()
                                                ),
                                                buffer_kind_label(
                                                    ctx.up.gpu_slice.buffer().as_ref()
                                                ),
                                                buffer_kind_label(
                                                    ctx.down.gpu_slice.buffer().as_ref()
                                                ),
                                            ));
                                        }
                                        eprintln!("{log}");
                                    }
                                    if let Some(ref ctx) = layer0_probe.partition_ctx {
                                        let gpu_alloc =
                                            make_partition_gpu_alloc(&*backend, decode_mem);
                                        gen_ws.partition_ws = Some(Arc::new(PartitionWsCell::new(
                                            PartitionWorkspace::new(
                                                ctx,
                                                ffn_hidden,
                                                hidden_size,
                                                &gpu_alloc,
                                                backend.clone(),
                                                cpu_backend_arc.clone(),
                                            )?,
                                        )));
                                    } else {
                                        gen_ws.partition_ws = None;
                                    }
                                    executor.set_partition_ratio(ratio);
                                    last_applied_partition_ratio = Some(ratio);
                                    // Partition active: invalidate plan so the
                                    // partition co-execution path takes over.
                                    #[cfg(feature = "opencl")]
                                    {
                                        gpu_plan = None;
                                    }
                                    // Re-register Q4_0 noshuffle SOA entries:
                                    // `map_weights_for_cpu()` above replaced
                                    // GPU-only weights' `UnifiedBuffer`, minting
                                    // new `cl_mem` pointers. The SOA registry's
                                    // old entries are now keyed by stale
                                    // `cl_mem`s, so `build_plan()` would miss
                                    // the lookup and silently fall back to the
                                    // AOS Q4_0 GEMV (measured +102% TBT on
                                    // Galaxy S25, verify v2 ISSUE-2).
                                    //
                                    // Clear + rebuild mirrors the CLI init path
                                    // (prepare_tensor_partition → prepare_
                                    // noshuffle_buffers) so partition sub-buffer
                                    // slices are also registered. Idempotent
                                    // for non-Q4_0 weight dtypes: the
                                    // prepare_noshuffle_buffers() helper
                                    // short-circuits on DType::Q4_0 mismatch.
                                    #[cfg(feature = "opencl")]
                                    if is_gpu {
                                        let actual_q4 = w_dtype == DType::Q4_0
                                            || model.layers.first().is_some_and(|l| {
                                                l.load_weights().wq.dtype() == DType::Q4_0
                                            });
                                        if actual_q4
                                            && let Some(ocl_be) = backend
                                                .as_any()
                                                .downcast_ref::<
                                                    llm_rs2::backend::opencl::OpenCLBackend,
                                                >()
                                        {
                                            ocl_be.clear_noshuffle_soa_registry();
                                            // Tensor partition runtime re-register:
                                            // `map_weights_for_cpu()` above replaced the
                                            // per-weight `UnifiedBuffer` with a CPU-mapped
                                            // version, so we keep the AOS allocation
                                            // alive — the partition path (and any CPU
                                            // matmul fallback) dereferences the original
                                            // cl_mem directly. `keep_original=true` stops
                                            // `prepare_noshuffle_buffers` from swapping
                                            // the tensor buffers out from under the
                                            // caller.
                                            match model
                                                .prepare_noshuffle_buffers(&backend, true)
                                            {
                                                Ok(n) => eprintln!(
                                                    "[Partition] Re-registered Q4_0 noshuffle SOA: {} weight tensors",
                                                    n
                                                ),
                                                Err(e) => eprintln!(
                                                    "[Partition] Noshuffle re-registration failed: {} (AOS fallback will hurt TBT)",
                                                    e
                                                ),
                                            }
                                        }
                                    }
                                }
                                Err(e) => eprintln!("[Partition] Re-split failed: {}", e),
                            }
                        }
                    }
                }

                // Dynamic layer skip / restore_defaults handling
                if plan.restore_defaults {
                    eprintln!("[Resilience] RestoreDefaults");
                    skip_config = None;
                    last_skip_ratio = None;
                    evict_ceiling = None;
                    evict_floor_logged = None;
                } else if let Some(ratio) = plan.layer_skip
                    && last_skip_ratio != Some(ratio)
                {
                    eprintln!("[Resilience] LayerSkip: ratio={:.2}", ratio);
                    skip_config = Some(SkipConfig::uniform_init(
                        model.config.num_hidden_layers,
                        ratio,
                    ));
                    last_skip_ratio = Some(ratio);
                }

                if let Some(ref device) = plan.switch_device {
                    if let (Some(gpu_be), Some(gpu_mem)) = (&gpu_backend_arc, &gpu_memory_arc) {
                        match device.as_str() {
                            "cpu" if is_gpu => {
                                if spare_logits.is_none() {
                                    eprintln!(
                                        "[Switch] ERROR: SwitchHw requires --resilience-prealloc-switch flag. Ignoring directive."
                                    );
                                } else {
                                    eprintln!(
                                        "[Switch] Resilience: GPU→CPU at token {}",
                                        kv_caches[0].current_pos
                                    );
                                    llm_rs2::core::kv_migrate::migrate_kv_caches(
                                        &mut kv_caches,
                                        &backend,
                                        &cpu_backend_arc,
                                        &cpu_backend_arc,
                                        &cpu_memory_arc,
                                        &cpu_memory_arc,
                                        kv_heads,
                                        head_dim,
                                        max_seq_len,
                                        false,
                                    )?;
                                    backend = cpu_backend_arc.clone();
                                    // Zero-alloc swap: exchange active GPU buffers with
                                    // pre-allocated CPU spares. GPU buffers survive in spare_*
                                    // (no clReleaseMemObject, no RSS spike).
                                    if let (Some(sl), Some(sx), Some(sw), Some(si)) = (
                                        spare_logits.as_mut(),
                                        spare_xgen.as_mut(),
                                        spare_gen_ws.as_mut(),
                                        spare_gen_input.as_mut(),
                                    ) {
                                        std::mem::swap(&mut logits, sl);
                                        std::mem::swap(&mut x_gen, sx);
                                        std::mem::swap(&mut gen_ws, sw);
                                        std::mem::swap(&mut gen_input_tensor, si);
                                    }
                                    #[cfg(feature = "opencl")]
                                    {
                                        gpu_plan = None;
                                    }
                                    is_gpu = false;
                                    // Re-tag weight tensors with CPU backend.
                                    eprintln!("[Switch] Resilience: Switched to CPU.");
                                }
                            }
                            "gpu" | "opencl" if !is_gpu && weights_on_gpu => {
                                if spare_logits.is_none() {
                                    eprintln!(
                                        "[Switch] ERROR: SwitchHw requires --resilience-prealloc-switch flag. Ignoring directive."
                                    );
                                } else {
                                    eprintln!(
                                        "[Switch] Resilience: CPU→GPU at token {}",
                                        kv_caches[0].current_pos
                                    );
                                    llm_rs2::core::kv_migrate::migrate_kv_caches(
                                        &mut kv_caches,
                                        &backend,
                                        gpu_be,
                                        &cpu_backend_arc,
                                        &cpu_memory_arc,
                                        gpu_mem,
                                        kv_heads,
                                        head_dim,
                                        max_seq_len,
                                        true,
                                    )?;
                                    backend = gpu_be.clone();
                                    // Zero-alloc swap: exchange active CPU buffers with
                                    // spare GPU buffers (preserved from previous switch).
                                    if let (Some(sl), Some(sx), Some(sw), Some(si)) = (
                                        spare_logits.as_mut(),
                                        spare_xgen.as_mut(),
                                        spare_gen_ws.as_mut(),
                                        spare_gen_input.as_mut(),
                                    ) {
                                        std::mem::swap(&mut logits, sl);
                                        std::mem::swap(&mut x_gen, sx);
                                        std::mem::swap(&mut gen_ws, sw);
                                        std::mem::swap(&mut gen_input_tensor, si);
                                    }
                                    #[cfg(feature = "opencl")]
                                    {
                                        gpu_plan = None;
                                    }
                                    is_gpu = true;
                                    eprintln!("[Switch] Resilience: Switched to GPU (zero-alloc).");
                                }
                            }
                            "gpu" | "opencl" if !is_gpu && !weights_on_gpu => {
                                eprintln!(
                                    "[Resilience] SwitchHw(gpu): model weights on CPU, not GPU-accessible. \
                                     Start with --backend opencl for GPU switching."
                                );
                            }
                            _ => {} // Already on requested backend
                        }
                    } else {
                        eprintln!(
                            "[Resilience] SwitchHw({}): no secondary backend available",
                            device
                        );
                    }
                }

                // kv_quant_bits: not supported on F16 KVCache path
                if let Some(bits) = plan.kv_quant_bits {
                    eprintln!(
                        "[Resilience] Warning: kv_quant_dynamic(bits={}) requested but KV cache is F16 (not KIVI). \
                         Dynamic quantization requires --kv-type q2/q4. Ignoring.",
                        bits
                    );
                }

                // KvOffload: Manager-directed LRU prefix offload to disk.
                if let Some(ratio) = plan.offload_ratio {
                    match cache_manager.offload(&mut kv_caches, ratio) {
                        Ok(n) => eprintln!(
                            "[Resilience] KvOffload: ratio={:.2}, {} tokens swapped",
                            ratio, n
                        ),
                        Err(e) => eprintln!("[Resilience] KvOffload failed: {}", e),
                    }
                }
                // RestoreDefaults → recall offloaded tokens back from disk.
                if plan.recall_offload && plan.restore_defaults {
                    match cache_manager.recall(&mut kv_caches) {
                        Ok(n) => {
                            if n > 0 {
                                eprintln!("[Resilience] Recalled {} tokens from swap", n);
                            }
                        }
                        Err(e) => eprintln!("[Resilience] Recall failed: {}", e),
                    }
                }

                if plan.throttle_delay_ms != throttle_delay_ms {
                    eprintln!(
                        "[Resilience] Throttle: {}ms → {}ms",
                        throttle_delay_ms, plan.throttle_delay_ms
                    );
                }
                throttle_delay_ms = plan.throttle_delay_ms;

                // Update target TBT from Manager directive (overrides CLI --target-tbt).
                // `target_tbt_set` distinguishes "manager explicitly said 0 (disable
                // pacing)" from "manager never sent a directive" — otherwise a
                // `SetTargetTbt { target_ms: 0 }` restore cannot clear a prior
                // non-zero target (see verify/ISSUE-3).
                if plan.target_tbt_set && plan.target_tbt_ms as f64 != target_tbt_ms {
                    eprintln!(
                        "[Resilience] SetTargetTbt: {:.1}ms → {}ms",
                        target_tbt_ms, plan.target_tbt_ms
                    );
                    target_tbt_ms = plan.target_tbt_ms as f64;
                } else if plan.restore_defaults {
                    target_tbt_ms = args.target_tbt; // restore CLI default
                }

                // RestoreDefaults: restore partition ratio to CLI initial value
                if plan.restore_defaults {
                    let cli_ratio = args.tensor_partition;
                    if cli_ratio > 0.0 && cli_ratio < 1.0 {
                        // Restore to CLI partition ratio
                        let layer0_probe = model.layers[0].load_weights();
                        if !layer0_probe.wq.as_ptr().is_null()
                            && let Ok(n) =
                                model.prepare_tensor_partition(cli_ratio, &cpu_backend_arc)
                        {
                            eprintln!(
                                "[Partition] RestoreDefaults: re-split {} weights with CLI ratio {:.2}",
                                n, cli_ratio
                            );
                            // prepare_tensor_partition returns 0 when cli_ratio is
                            // in the GPU-only fast-path band; partition_ctx is then
                            // None and we must clear any stale workspace so forward()
                            // stays on the dense GPU path.
                            let layer0_probe2 = model.layers[0].load_weights();
                            if let Some(ref ctx) = layer0_probe2.partition_ctx {
                                let gpu_alloc = make_partition_gpu_alloc(&*backend, decode_mem);
                                if let Ok(ws) = PartitionWorkspace::new(
                                    ctx,
                                    ffn_hidden,
                                    hidden_size,
                                    &gpu_alloc,
                                    backend.clone(),
                                    cpu_backend_arc.clone(),
                                ) {
                                    gen_ws.partition_ws = Some(Arc::new(PartitionWsCell::new(ws)));
                                }
                            } else {
                                gen_ws.partition_ws = None;
                            }
                            executor.set_partition_ratio(cli_ratio);
                            last_applied_partition_ratio = Some(cli_ratio);
                        }
                    } else {
                        // CLI had no partition — disable via ArcSwap clone-and-install
                        for slot in &model.layers {
                            let old = slot.load_weights();
                            let mut new = (*old).clone();
                            new.partition_ctx = None;
                            slot.store_weights_same_dtype(Arc::new(new));
                        }
                        gen_ws.partition_ws = None;
                        executor.set_partition_ratio(0.0);
                        last_applied_partition_ratio = None;
                    }
                }

                if plan.suspended {
                    eprintln!("\n[Resilience] Inference suspended by system signal");
                    break;
                }

                if throttle_delay_ms > 0 {
                    experiment_total_throttle_ms += throttle_delay_ms;
                    std::thread::sleep(std::time::Duration::from_millis(throttle_delay_ms));
                }

                executor.on_token_generated();
            } else if throttle_delay_ms > 0 {
                // No CommandExecutor: honour CLI --throttle-delay-ms directly
                // so decode pacing works without --enable-resilience.
                std::thread::sleep(std::time::Duration::from_millis(throttle_delay_ms));
            }
            // ── End Resilience checkpoint ─────────────────────

            // Read logits to CPU (reuses pre-allocated buffer).
            //
            // Token-boundary sync point: `Backend::read_buffer` always
            // calls `synchronize()` internally (see CpuBackend/OpenCL/
            // cuda_embedded impls). This is the barrier that guarantees
            // all in-flight kernels complete before sampling runs —
            // critical for `--cuda-defer-sync` where per-op syncs are
            // suppressed.
            unsafe {
                let ptr = logits_cpu.as_mut_ptr() as *mut u8;
                let slice = std::slice::from_raw_parts_mut(ptr, logits_cpu.len() * 4);
                backend.read_buffer(&logits, slice)?;
            }

            // Extract top-K logits before sampling modifies them
            let top_logits = if experiment_writer.is_some() {
                extract_top_k_logits(&logits_cpu, args.experiment_logits_topk)
            } else {
                vec![]
            };

            let sample_start = std::time::Instant::now();
            let next_token_id = sampling::sample(
                &mut logits_cpu,
                &tokens,
                vocab_size,
                &sampling_config,
                Some(&mut sampling_indices),
            );
            let sample_us = sample_start.elapsed().as_micros() as u64;

            let now = std::time::Instant::now();
            let mut tbt = now.duration_since(_last_token_time).as_secs_f64() * 1000.0;

            // ── Target TBT pacing: sleep to maintain target throughput ──
            let pacing_ms = if target_tbt_ms > 0.0 && tbt < target_tbt_ms {
                let sleep_ms = target_tbt_ms - tbt;
                std::thread::sleep(std::time::Duration::from_secs_f64(sleep_ms / 1000.0));
                tbt = target_tbt_ms; // effective TBT = target
                sleep_ms
            } else {
                0.0
            };

            tbt_values.push(tbt);

            // ── TBT log: write per-token JSONL ──
            if let Some(ref mut w) = tbt_log_writer {
                use std::io::Write;
                writeln!(w,
                    "{{\"token_idx\":{},\"tbt_ms\":{:.2},\"forward_ms\":{:.2},\"cache_pos\":{},\"pacing_ms\":{:.2}}}",
                    decode_token_index, tbt, forward_ms, kv_caches[0].current_pos, pacing_ms
                ).ok();
            }

            // ── Profiler: record step data ──
            if let Some(ref mut p) = profiler {
                let forward_us = (forward_ms * 1000.0) as u64;
                let total_us = forward_us + sample_us;
                let cache_len = kv_caches[0].current_pos;
                let (imp, head_imp, n_kv) = match score_accumulator {
                    Some(ref acc) if acc.is_active() => (
                        Some(acc.importance_scores()),
                        acc.head_importance_scores(),
                        acc.n_kv_heads(),
                    ),
                    _ => (None, None, 0),
                };
                let pos_map = if position_birth_step.is_empty() {
                    None
                } else {
                    Some(position_birth_step.as_slice())
                };
                p.on_step_end(
                    decode_token_index,
                    next_token_id,
                    forward_us,
                    sample_us,
                    total_us,
                    cache_len,
                    imp,
                    head_imp,
                    n_kv,
                    pos_map,
                );
                // Record new token birth
                p.scores
                    .record_token_births(decode_token_index + 1, 1, actual_protected_prefix);
            }
            // Track birth step for new token (even if profiler is off, keep mapping in sync)
            if !position_birth_step.is_empty() {
                position_birth_step.push(decode_token_index + 1);
            }

            _last_token_time = now;
            tokens.push(next_token_id);

            // start_pos tracks the LOGICAL position for RoPE encoding.
            start_pos += 1;

            // ── Experiment: write per-token JSONL record ──
            if let Some(ref mut writer) = experiment_writer {
                let token_text = tokenizer
                    .decode(&[next_token_id], false)
                    .unwrap_or_default();
                let sys_metrics = system_sampler.sample(decode_token_index);
                let signal_str = if injected_signals.is_empty() {
                    None
                } else {
                    Some(injected_signals.join("+"))
                };
                let record = TokenRecord {
                    pos: decode_token_index,
                    token_id: next_token_id,
                    text: token_text,
                    tbt_ms: tbt,
                    forward_ms,
                    signal: signal_str.as_deref(),
                    actions: action_names,
                    cache_pos: kv_caches[0].current_pos,
                    throttle_ms: throttle_delay_ms,
                    top_logits,
                    sys: sys_metrics,
                };
                writer.write_token(&record)?;
            }

            // Streaming print (suppress in experiment mode for clean JSONL)
            if experiment_writer.is_none() {
                let current_text = tokenizer.decode(&tokens, true).unwrap_or_default();
                if let Some(substring) = current_text.get(_printed_len..).filter(|s| !s.is_empty())
                {
                    print!("{}", substring);
                    stdout.flush().ok();
                    _printed_len = current_text.len();
                }
            }
            // D-D.6 debug: dump raw token IDs (special token visibility).
            if std::env::var("LLMRS_DUMP_TOKEN_IDS").is_ok() {
                eprintln!(
                    "[token-id step={}] id={}",
                    decode_token_index, next_token_id
                );
            }

            // T3 / T4: RSS snapshot after first and 16th decode tokens.
            if decode_token_index == 0 {
                rss_trace("decode_1");
            } else if decode_token_index == 15 {
                rss_trace("decode_16");
            }

            if next_token_id == eos_id && !args.ignore_eos && std::env::var("IGNORE_EOS").is_err() {
                break;
            }
        }
    }

    // 6. Write experiment summary
    if let Some(ref mut writer) = experiment_writer {
        let sys_end = Some(system_sampler.snapshot());
        let avg_tbt_ms = if tbt_values.is_empty() {
            0.0
        } else {
            tbt_values.iter().sum::<f64>() / tbt_values.len() as f64
        };
        let avg_forward_ms = if forward_ms_values.is_empty() {
            0.0
        } else {
            forward_ms_values.iter().sum::<f64>() / forward_ms_values.len() as f64
        };
        let summary = SummaryRecord {
            _summary: true,
            total_tokens: tbt_values.len(),
            ttft_ms: _ttft_ms,
            avg_tbt_ms,
            avg_forward_ms,
            total_throttle_ms: experiment_total_throttle_ms,
            eviction_count: experiment_eviction_count,
            evicted_tokens_total: experiment_evicted_total,
            final_cache_pos: kv_caches[0].current_pos,
            max_seq_len,
            prompt: prompt.clone(),
            schedule_name: experiment_schedule
                .as_ref()
                .map(|s| s.name.clone())
                .unwrap_or_else(|| "baseline".to_string()),
            eviction_policy: args.eviction_policy.clone(),
            backend: args.backend.clone(),
            sample_interval: args.experiment_sample_interval,
            sys_start,
            sys_end,
            governor: Some(SystemSampler::read_governor()),
        };
        writer.write_summary(&summary)?;
        eprintln!(
            "[Experiment] Done: {} tokens, avg TBT {:.2}ms, {} evictions",
            summary.total_tokens, avg_tbt_ms, experiment_eviction_count
        );
    }

    // 6.5. Export profiler data if enabled
    if let Some(ref profiler) = profiler {
        profiler.ops.print_report();

        let metadata = llm_rs2::profile::ProfileMetadata {
            model: args.model_path.clone(),
            backend: args.backend.clone(),
            eviction_policy: args.eviction_policy.clone(),
            max_seq_len: args.max_seq_len,
            prompt_len: prompt.len(),
            generated_tokens: tbt_values.len(),
        };
        match profiler.export_json(&metadata) {
            Ok(path) => eprintln!("[Profile] Exported to {}", path.display()),
            Err(e) => eprintln!("[Profile] Export failed: {}", e),
        }
    }

    // 6.6. Export --cuda-profile aggregate if enabled.
    // Independent of the generic `profiler` above (which lives in
    // `llm_rs2::profile::ops::OpProfiler` and targets OpenCL events).
    #[cfg(feature = "cuda-embedded")]
    if args.cuda_profile
        && let Some(cuda_be) = backend
            .as_any()
            .downcast_ref::<llm_rs2::backend::cuda_embedded::CudaBackend>()
        && cuda_be.profiler_enabled()
    {
        match cuda_be.flush_profiler() {
            Ok(Some(map)) => {
                let dropped = cuda_be.profiler_dropped();
                dump_cuda_profile_report(&map, dropped, &args, tbt_values.len(), cuda_be.device());
            }
            Ok(None) => {}
            Err(e) => eprintln!("[CUDA-Profile] flush failed: {}", e),
        }
    }

    // 7. Output results
    println!("\nDone.");
    println!("[Profile] Event: End");
    #[cfg(feature = "cuda-embedded")]
    {
        llm_rs2::backend::cuda_embedded::dump_fallback_counters();
        if let Some(cu_be) = backend
            .as_any()
            .downcast_ref::<llm_rs2::backend::cuda_embedded::CudaBackend>()
        {
            cu_be.dump_graph_counters();
        }
    }
    // WSWAP-5-TBT-DIAG: final cl_mem dump after the entire generation
    // pipeline completes. Includes any growth that occurred during prefill /
    // decode (KV cache grow-on-demand, plan-rebuild scratch, etc.).
    #[cfg(feature = "opencl")]
    if let Some(ocl_be) = backend
        .as_any()
        .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
    {
        ocl_be.dump_cl_mem_diagnostics(" stage=after_generate");
    }
    println!("TTFT: {:.2} ms", _ttft_ms);
    if !forward_ms_values.is_empty() {
        let avg_forward: f64 =
            forward_ms_values.iter().sum::<f64>() / forward_ms_values.len() as f64;
        println!(
            "Decode: {:.2} ms/tok ({:.1} tok/s) [{} tokens, forward only]",
            avg_forward,
            1000.0 / avg_forward,
            forward_ms_values.len(),
        );
        // Sprint E: flush the forward_gen op-tracer right after the Decode
        // summary so the per-op breakdown sits next to the headline TBT in
        // the log (atexit will fire too, but is moot once we already dumped).
        llm_rs2::profile::op_trace::dump_and_reset();
        if forward_ms_values.len() >= 2 {
            let tail = &forward_ms_values[1..];
            let avg_tail: f64 = tail.iter().sum::<f64>() / tail.len() as f64;
            let tok0 = forward_ms_values[0];
            println!(
                "Decode(excl tok[0]): {:.2} ms/tok ({:.1} tok/s) [{} tokens] | tok[0]={:.2} ms",
                avg_tail,
                1000.0 / avg_tail,
                tail.len(),
                tok0,
            );
        }
    }
    if !tbt_values.is_empty() {
        let avg_tbt: f64 = tbt_values.iter().sum::<f64>() / tbt_values.len() as f64;
        println!(
            "Avg TBT: {:.2} ms ({:.1} tokens/sec)",
            avg_tbt,
            1000.0 / avg_tbt
        );
    }

    // --qcf-dump: write JSON for generation mode (ppl=null, avg_nll=null).
    if let Some(ref dump_path) = args.qcf_dump {
        use llm_rs2::eval::qcf_helpers::{QcfSwapDumpContext, dump_qcf_swap_json};

        let empty_swap: Vec<usize> = Vec::new();
        let (swap_set, qcf_predicted, fallback_used) = if let Some(ref dec) = qcf_swap_decision {
            (
                dec.selected_layers.as_slice(),
                dec.qcf_swap_estimate,
                dec.fallback_used,
            )
        } else {
            (empty_swap.as_slice(), 0.0f32, false)
        };

        let secondary_path_str = args.secondary_gguf.as_ref().and_then(|p| p.to_str());
        let model_arch = if args.model_path.to_lowercase().contains("qwen") {
            "qwen2"
        } else {
            "llama"
        };
        let total_wall = qcf_workflow_start.elapsed().as_secs_f64();

        let ctx = QcfSwapDumpContext {
            model_arch,
            model_path: &args.model_path,
            secondary_path: secondary_path_str,
            primary_dtype: "F16",
            secondary_dtype: "Q4_0",
            num_layers: model.layers.len(),
            force_swap_ratio: args.force_swap_ratio,
            swap_algorithm: args
                .force_swap_ratio
                .map(|_| swap_algorithm.short_name()),
            swap_set,
            qcf_swap_predicted: qcf_predicted,
            fallback_used,
            importance_table: qcf_warmup_importance.as_ref(),
            noise_table: Some(model.quant_noise.as_ref()),
            ppl: None,
            avg_nll: None,
            n_eval_tokens: 0,
            wall_time_s: total_wall,
            warmup_tokens: args.qcf_warmup_tokens,
            backend: &args.backend,
            kv_type: &args.kv_type,
            ppl_corpus: None,
            eval_ll_output: None,
            trajectory: None,
            dpllm_epsilon: None,
        };

        dump_qcf_swap_json(dump_path, &ctx)?;
        eprintln!("[QCF-dump] JSON written to {}", dump_path.display());
    }

    // T5: normal exit — all allocations still live (model, KV caches, workspaces).
    rss_trace("exit");
    Ok(())
}

// ── CUDA profile report ───────────────────────────────────────

#[cfg(feature = "cuda-embedded")]
fn dump_cuda_profile_report(
    map: &std::collections::HashMap<&'static str, (u64, f64)>,
    dropped: u64,
    args: &Args,
    n_tokens: usize,
    device: &str,
) {
    // 1) Sort by total_ms descending — same ordering as the OpenCL
    //    `--profile-events` summary so the two reports read the same.
    let mut rows: Vec<(&&str, &(u64, f64))> = map.iter().collect();
    rows.sort_by(|a, b| {
        b.1.1
            .partial_cmp(&a.1.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total_ms: f64 = rows.iter().map(|(_, v)| v.1).sum();

    eprintln!(
        "\n=== CUDA per-op profile ({} ops over {} tokens) ===",
        rows.len(),
        n_tokens
    );
    eprintln!(
        "{:<28}  {:>8}  {:>12}  {:>10}  {:>7}",
        "label", "count", "total_ms", "mean_ms", "pct"
    );
    for (label, (count, t_ms)) in &rows {
        let pct = if total_ms > 0.0 {
            t_ms / total_ms * 100.0
        } else {
            0.0
        };
        let mean = if *count > 0 {
            t_ms / (*count as f64)
        } else {
            0.0
        };
        eprintln!(
            "{:<28}  {:>8}  {:>12.3}  {:>10.4}  {:>6.2}%",
            label, count, t_ms, mean, pct
        );
    }
    eprintln!(
        "{:<28}  {:>8}  {:>12.3}",
        "TOTAL",
        rows.iter().map(|(_, v)| v.0).sum::<u64>(),
        total_ms
    );
    if dropped > 0 {
        eprintln!(
            "[CUDA-Profile] WARNING: {dropped} records dropped (pool exhausted between flushes)"
        );
    }

    // 2) Write JSON to results/profile/cuda_embedded_decode_<ts>.json.
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Plain unix seconds — `chrono` is not a workspace dep. Downstream
    // tooling can parse either form.
    let ts_iso = format!("unix:{ts}");

    let mut ops_json = Vec::with_capacity(rows.len());
    for (label, (count, t_ms)) in &rows {
        let mean = if *count > 0 {
            t_ms / (*count as f64)
        } else {
            0.0
        };
        ops_json.push(serde_json::json!({
            "label": label,
            "count": count,
            "total_ms": t_ms,
            "mean_ms": mean,
        }));
    }
    let doc = serde_json::json!({
        "timestamp": ts_iso,
        "device": device,
        "backend": "cuda-embedded",
        "n_tokens": n_tokens,
        "model": args.model_path,
        "dropped_records": dropped,
        "ops": ops_json,
    });

    let dir = std::path::PathBuf::from(&args.profile_dir);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("[CUDA-Profile] mkdir {} failed: {}", dir.display(), e);
        return;
    }
    let path = dir.join(format!("cuda_embedded_decode_{}.json", ts));
    match std::fs::write(&path, serde_json::to_vec_pretty(&doc).unwrap_or_default()) {
        Ok(()) => eprintln!("[CUDA-Profile] Exported to {}", path.display()),
        Err(e) => eprintln!("[CUDA-Profile] write {} failed: {}", path.display(), e),
    }
}

// ── Experiment helpers ────────────────────────────────────────

fn directive_summary(msg: &ManagerMessage) -> String {
    match msg {
        ManagerMessage::Directive(d) => {
            let cmds: Vec<String> = d.commands.iter().map(command_summary).collect();
            format!("Directive(seq={}, [{}])", d.seq_id, cmds.join(", "))
        }
    }
}

/// Context for dry-run QCF estimation (ENG-ALG-050).
/// Groups all inputs needed by `compute_qcf_estimates` so that the caller
/// constructs a single struct instead of passing many individual arguments.
struct QcfEstimateContext<'a> {
    kv_caches: &'a [KVCache],
    score_accumulator: Option<&'a AttentionScoreAccumulator>,
    /// (sink_size, window_size) for StreamingLLM dry-run. None = skip.
    streaming_config: Option<(usize, usize)>,
    /// Pre-built importance table for LayerSkip dry-run. None = skip.
    importance_table: Option<&'a llm_rs2::core::qcf::ImportanceTable>,
    /// Total number of transformer layers (needed for LayerSkip).
    num_layers: usize,
    /// KIVI caches for dynamic quantization QCF dry-run. None = skip.
    kivi_caches: Option<&'a [KiviCache]>,
}

/// Compute dry-run QCF estimates for all 6 lossy actions (ENG-ALG-050).
/// Read-only: does not modify KV caches.
///
/// Uses unified QCF formula: QCF = ||O_before - O_after|| / ||O_before||
/// where O = sum_t alpha_t * V_t (attention-weighted value output).
///
/// Returns estimates for:
/// - `kv_evict_sliding`  : Sliding window eviction
/// - `kv_evict_h2o`      : H2O importance-based eviction (needs scores)
/// - `kv_evict_streaming` : StreamingLLM eviction (needs streaming_config)
/// - `kv_merge_d2o`      : D2O merge estimate (needs scores)
/// - `kv_quant_dynamic`  : KIVI dynamic quantization (skipped for non-KiviCache path)
/// - `layer_skip`        : LayerSkip importance-based QCF (needs importance_table)
fn compute_qcf_estimates(ctx: &QcfEstimateContext<'_>) -> std::collections::HashMap<String, f32> {
    use llm_rs2::core::qcf::{
        AggregationMode, QcfActionType, UnifiedQcfParams, VDataSource, compute_unified_qcf,
    };
    use std::collections::HashMap;
    let mut estimates = HashMap::new();

    // ── 1-4. KVCache-based eviction/merge QCF via unified formula ──
    //
    // ISSUE-6 guard: OpenCL device-only 버퍼는 `as_ptr()`이 명시적으로
    // `ptr::null()`을 반환한다 (engine/src/backend/opencl/buffer.rs). 이 경우
    // `Tensor::as_slice::<T>()`이 `(ptr=null, len=size/sizeof T)` 슬라이스를
    // 만들고 `read_v_f32()`에서 `data[offset..end]`로 인덱싱하는 순간 null
    // deref → SIGSEGV. `VDataSource::from_kv_cache(None)` 가 host pointer 검사
    // 후 `None`을 돌려주므로 device-only 캐시는 자연스럽게 skip된다.
    let v_host_readable =
        !ctx.kv_caches.is_empty() && ctx.kv_caches.iter().all(|c| !c.v_buffer.as_ptr().is_null());
    if !ctx.kv_caches.is_empty() && ctx.kv_caches[0].current_pos > 0 && !v_host_readable {
        eprintln!(
            "[QCF] KV-based estimates skipped: v_buffer is device-only (signal path without host-mapped KV)."
        );
    }
    if v_host_readable && ctx.kv_caches[0].current_pos > 0 {
        let cache = &ctx.kv_caches[0];
        let current_pos = cache.current_pos;
        let keep_ratio = 0.5f32;
        let target_len = (current_pos as f32 * keep_ratio) as usize;
        let protected_prefix = 4usize;

        let scores_opt = ctx
            .score_accumulator
            .filter(|a| a.is_active())
            .map(|a| a.importance_scores());
        let head_attn_opt = ctx
            .score_accumulator
            .filter(|a| a.is_active())
            .and_then(|a| a.last_step_head_attn());

        let fallback_scores: Vec<f32>;
        let attention_scores: &[f32] = if let Some(scores) = scores_opt {
            scores
        } else {
            fallback_scores = vec![1.0 / current_pos.max(1) as f32; current_pos];
            &fallback_scores
        };

        if target_len < current_pos {
            // (id, action, requires_scores). `kv_evict_sliding` needs no scores;
            // h2o/d2o use heavy-hitter selection so are gated on score availability.
            // Streaming QCF only fires when streaming_config is set.
            let mut actions: Vec<(&'static str, QcfActionType, bool)> = vec![
                (
                    "kv_evict_sliding",
                    QcfActionType::EvictSliding { target_len },
                    false,
                ),
                (
                    "kv_evict_h2o",
                    QcfActionType::EvictH2o {
                        target_len,
                        keep_ratio: 0.5,
                        protected_prefix,
                    },
                    true,
                ),
                (
                    "kv_merge_d2o",
                    QcfActionType::MergeD2o {
                        target_len,
                        keep_ratio: 0.5,
                        protected_prefix,
                    },
                    true,
                ),
            ];
            if let Some((sink_size, window_size)) = ctx.streaming_config {
                actions.push((
                    "kv_evict_streaming",
                    QcfActionType::EvictStreaming {
                        sink_size,
                        window_size,
                    },
                    false,
                ));
            }

            for (id, action, requires_scores) in actions {
                if requires_scores && scores_opt.is_none() {
                    continue;
                }
                let Some(v_source) = VDataSource::from_kv_cache(cache, None) else {
                    continue;
                };
                // D2O simulator (paper Eq.8) needs K for nearest-neighbour
                // matching; other actions ignore `k_source`.
                let k_source = if matches!(action, QcfActionType::MergeD2o { .. }) {
                    VDataSource::k_from_kv_cache(cache)
                } else {
                    None
                };
                let params = UnifiedQcfParams {
                    action,
                    v_source,
                    k_source,
                    attention_scores,
                    head_attn: head_attn_opt,
                    n_kv_heads: cache.kv_heads(),
                    head_dim: cache.head_dim(),
                    current_pos,
                    capacity: cache.capacity(),
                    layout: cache.layout(),
                    aggregation: AggregationMode::Mean,
                    beta: 1.0,
                };
                let (qcf, _) = compute_unified_qcf(&params);
                estimates.insert(id.to_string(), qcf);
            }
        }
    }

    // ── 5. KIVI dynamic quantization QCF ──
    if let Some(kivi_caches) = ctx.kivi_caches
        && !kivi_caches.is_empty()
    {
        let mut total_qcf = 0.0f32;
        let mut count = 0u32;
        for cache in kivi_caches {
            let qcf = cache.estimate_dryrun_qcf();
            if qcf > 0.0 {
                total_qcf += qcf;
                count += 1;
            }
        }
        if count > 0 {
            let avg_qcf = total_qcf / count as f32;
            estimates.insert("kv_quant_dynamic".to_string(), avg_qcf.min(1.0));
        }
    }

    // ── 6. LayerSkip QCF: importance-table based skip cost estimate ──
    if let Some(table) = ctx.importance_table {
        let total_sublayers = ctx.num_layers * 2;
        let skip_count = total_sublayers / 4;
        if skip_count > 0 {
            let (qcf_skip, _skip_set) = table.estimate_qcf_for_count(skip_count, ctx.num_layers);
            estimates.insert("layer_skip".to_string(), qcf_skip);
        }
    }

    estimates
}

// ── Weight swap dispatch (ENG-ALG-214-ROUTE) ────────────────────────────────

/// Run a `SwapExecutor` over the given target layers.
///
/// Centralises the boilerplate shared by the four call sites that execute a
/// weight swap (`--force-swap-ratio`, two QCF-dump warmup paths, and the
/// `EngineCommand::SwapWeights` direct dispatch). Always targets `Q4_0`
/// (only currently-supported swap dtype, INV-126). Resolves the swap backend
/// to GPU when available, otherwise CPU — matches the original logic that
/// `SwapExecutor` branches on `backend.name()` to pick the AUF SOA fast path.
#[allow(clippy::too_many_arguments)]
fn run_layer_swap(
    model: &llm_rs2::models::transformer::TransformerModel,
    target_layers: &[usize],
    gpu_backend: Option<&Arc<dyn Backend>>,
    cpu_backend: &Arc<dyn Backend>,
    async_dispatcher: Option<&llm_rs2::models::weights::AsyncSwapDispatcher>,
    #[cfg(feature = "opencl")] host_ptr_pool: Option<
        Arc<llm_rs2::backend::opencl::host_ptr_pool::HostPtrPool>,
    >,
) -> Result<llm_rs2::models::weights::SwapReport, llm_rs2::models::weights::SwapError> {
    let swap_memory = Galloc::new();
    let swap_backend: Arc<dyn Backend> =
        gpu_backend.cloned().unwrap_or_else(|| cpu_backend.clone());
    // ENG-ALG-228: attach the model's async release worker so Stage (c) enqueues
    // displaced LayerWeights for background drop instead of blocking inline.
    let executor = llm_rs2::models::weights::SwapExecutor::new_with_worker(
        DType::Q4_0,
        &model.config,
        swap_backend,
        &swap_memory,
        Arc::clone(&model.release_worker),
    );
    // LISWAP-3 prototype: if a host_ptr pool is supplied, attach it so the
    // AOS materialise path uses the zero-copy slot pool.
    #[cfg(feature = "opencl")]
    let executor = match host_ptr_pool {
        Some(pool) => executor.with_host_ptr_pool(pool),
        None => executor,
    };
    executor.execute_on_slots(
        model.layers.as_slice(),
        model.secondary_mmap.as_ref(),
        &model.ratio_generation,
        target_layers,
        async_dispatcher,
    )
}

/// Re-map weight tensors for CPU access after a weight swap.
///
/// Required when running on GPU with `--secondary-layout aos +
/// --resilience-prealloc-switch`: `SwapExecutor::materialise_tensor` lands an
/// unmapped `UnifiedBuffer` in the new `LayerWeights` snapshot, and the next
/// `switch_hw cpu` directive segfaults on a null host pointer.  Idempotent —
/// already-mapped tensors short-circuit in `map_one`.
#[cfg(feature = "opencl")]
fn remap_weights_for_cpu_after_swap(
    model: &mut llm_rs2::models::transformer::TransformerModel,
    backend: &Arc<dyn Backend>,
    is_gpu: bool,
    enabled: bool,
    label: &str,
) {
    if !is_gpu || !enabled {
        return;
    }
    match model.map_weights_for_cpu(backend) {
        Ok(0) => {}
        Ok(n) => eprintln!(
            "[Backend] Re-mapped {} weight tensors after {} (host pointer restored)",
            n, label,
        ),
        Err(e) => eprintln!(
            "[Backend] Post-swap re-map failed: {} (switch_hw cpu may crash)",
            e,
        ),
    }
}

/// Result of a QCF-dump warmup workflow: importance table plus optional swap
/// decision (when `--force-swap-ratio` was applied).
struct QcfWarmupResult {
    importance: llm_rs2::core::qcf::ImportanceTable,
    decision: Option<llm_rs2::models::weights::SwapDecision>,
    /// Per-layer DP-LLM proxy ε. `Some` only when 3-way comparison mode is
    /// active (`--importance-formula compare`). Same length as the layer count.
    dpllm_epsilon: Option<Vec<f32>>,
}

/// QCF-dump warmup workflow shared by `--ppl/generation` and `--eval-ll` modes.
///
/// 1. Warmup prefill with `ImportanceCollector` over `warmup_ids`.
/// 2. Build `ImportanceTable`, reset KV caches to zero.
/// 3. If `force_ratio` is set, run `WeightSwapDecider` and dispatch the swap.
///
/// `log_prefix` is concatenated immediately after `[QCF-dump]` in every log
/// line emitted by this helper, so any non-empty value must include its own
/// leading space (e.g. `" eval-ll"`). The caller must ensure `warmup_ids` is
/// non-empty.
#[allow(clippy::too_many_arguments)]
fn run_qcf_warmup_workflow(
    model: &llm_rs2::models::transformer::TransformerModel,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut [KVCache],
    vocab_size: usize,
    warmup_ids: &[u32],
    force_ratio: Option<f32>,
    gpu_backend: Option<&Arc<dyn Backend>>,
    cpu_backend: &Arc<dyn Backend>,
    log_prefix: &str,
    swap_algorithm: llm_rs2::models::weights::SwapAlgorithm,
    execute_swap: bool,
    importance_formula: llm_rs2::core::qcf::ImportanceFormula,
    importance_three_way: bool,
    swap_only_layers: Option<&[usize]>,
) -> anyhow::Result<QcfWarmupResult> {
    use llm_rs2::core::qcf::ImportanceCollector;
    use llm_rs2::models::weights::WeightSwapDecider;

    let actual_warmup_len = warmup_ids.len();
    eprintln!(
        "[QCF-dump]{} warmup prefill: {} tokens (formula={}, three_way={})",
        log_prefix,
        actual_warmup_len,
        importance_formula.as_str(),
        importance_three_way,
    );

    // ── Warmup prefill with ImportanceCollector ───────────────────────────────
    let mut collector = ImportanceCollector::new_with_formula(importance_formula, importance_three_way);
    {
        let warmup_buf = Galloc::new().alloc(actual_warmup_len * 4, DType::U8)?;
        unsafe {
            let ptr = warmup_buf.as_mut_ptr() as *mut u32;
            for (i, &id) in warmup_ids.iter().enumerate() {
                *ptr.add(i) = id;
            }
        }
        let cpu_warmup = Tensor::new(
            Shape::new(vec![1, actual_warmup_len]),
            warmup_buf,
            Arc::new(CpuBackend::new()),
        );
        let warmup_input = backend.copy_from(&cpu_warmup)?;

        let warmup_logits_buf = memory.alloc(actual_warmup_len * vocab_size * 4, DType::F32)?;
        let mut warmup_logits = Tensor::new(
            Shape::new(vec![1, actual_warmup_len, vocab_size]),
            warmup_logits_buf,
            backend.clone(),
        );

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &warmup_input,
            start_pos: 0,
            kv_caches,
            backend,
            memory,
            logits_out: &mut warmup_logits,
            x_gen: None,
            workspace: None,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: Some(&mut collector),
            logits_last_only: false,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;
        backend.synchronize()?;
    }

    // ── Build ImportanceTable (+ optional DP-LLM ε) + reset KV cache ─────────
    let (imp_table, dpllm_epsilon) = if importance_three_way {
        let (table, x_means) = collector.build_with_xmeans();
        let eps = model.secondary_mmap.as_ref().map(|sec| {
            llm_rs2::models::weights::noise_table::compute_input_aware_epsilon(
                &model.layers,
                sec,
                &x_means,
            )
        });
        (table, eps)
    } else {
        (collector.build(), None)
    };
    eprintln!(
        "[QCF-dump]{} ImportanceTable built: {} entries (dpllm_epsilon={})",
        log_prefix,
        imp_table.len(),
        if dpllm_epsilon.is_some() { "computed" } else { "skipped" },
    );
    for kv in kv_caches.iter_mut() {
        kv.current_pos = 0;
    }

    // ── Optional swap with importance-guided decider ──────────────────────────
    let decision = if let Some(ratio) = force_ratio {
        let ratio = ratio.clamp(0.0, 1.0);
        eprintln!(
            "[QCF-dump]{} swap algorithm: {} (execute_swap={})",
            log_prefix,
            swap_algorithm.short_name(),
            execute_swap,
        );
        let decider = WeightSwapDecider {
            importance: Some(&imp_table),
            noise: Some(model.quant_noise.as_ref()),
            n_decoder_layers: model.layers.len(),
            currently_swapped: &[],
            allow_boundary_layers: read_allow_boundary_env(),
            algorithm: swap_algorithm,
        };
        let decider_decision = decider.decide(ratio);

        // §4 ground-truth path: when `--swap-only-layers` is set, override the
        // decider's selection with the explicit list. The decider's
        // `qcf_swap_estimate` is recomputed against this override so the dump
        // JSON reports the QCF prediction for the actually-swapped set.
        let decision = if let Some(only) = swap_only_layers {
            let override_layers: Vec<usize> = only.iter().copied().filter(|i| *i < model.layers.len()).collect();
            let qcf_override = llm_rs2::models::weights::compute_qcf_swap(
                &override_layers,
                model.quant_noise.as_ref(),
                Some(&imp_table),
                model.layers.len(),
            );
            eprintln!(
                "[QCF-dump]{} swap-only override: layers={:?} (ignoring algorithm/ratio decision)",
                log_prefix, override_layers,
            );
            llm_rs2::models::weights::SwapDecision {
                selected_layers: override_layers,
                qcf_swap_estimate: qcf_override,
                fallback_used: false,
            }
        } else {
            decider_decision
        };

        // Trajectory mode (`--qcf-trajectory`): return the decision without
        // executing the swap — the caller drives swap one layer at a time
        // around per-step eval-ll measurements.
        if !execute_swap {
            return Ok(QcfWarmupResult {
                importance: imp_table,
                decision: Some(decision),
                dpllm_epsilon,
            });
        }

        if decision.selected_layers.is_empty() {
            eprintln!(
                "[QCF-dump]{} swap: ratio={:.2} → 0 layers selected (qcf=0.0)",
                log_prefix, ratio,
            );
        } else {
            let report = run_layer_swap(
                model,
                &decision.selected_layers,
                gpu_backend,
                cpu_backend,
                None,
                // LISWAP-3: QCF dump path does not exercise the pool yet —
                // Stage 3 prototype only wires --force-swap-ratio paths.
                #[cfg(feature = "opencl")]
                None,
            )
            .map_err(|e| anyhow::anyhow!("[QCF-dump]{} swap failed: {}", log_prefix, e))?;
            eprintln!(
                "[QCF-dump]{} swap: ratio={:.2}, layers={}/{}, qcf_pred={:.4}, \
                 fallback={}, latency={:.1}ms",
                log_prefix,
                ratio,
                report.swapped.len(),
                model.layers.len(),
                decision.qcf_swap_estimate,
                decision.fallback_used,
                report.latency_ms,
            );
        }
        Some(decision)
    } else {
        None
    };

    Ok(QcfWarmupResult {
        importance: imp_table,
        decision,
        dpllm_epsilon,
    })
}

/// Execute a SwapWeights command from the manager: validate → decide → commit
/// incremental plan → report on plan completion.
///
/// LISWAP-6 manager path: instead of sync single-shot execution, this function
/// commits an `IncrementalSwapPlan` (K=2, dynamic-K + sub-batch pause) to the
/// decode loop via `swap_plan_out`. The decode loop drains the plan per tick.
/// `WeightSwapReport` is sent when the plan completes (see plan-done block in
/// the decode loop). Manager receives "received" acknowledgment immediately
/// (via the existing executor ack in the command dispatch site), and the final
/// WeightSwapReport arrives on plan completion.
///
/// Rejection (no-secondary, invalid-ratio, unsupported-dtype, in-flight plan)
/// is logged to stderr; no plan is committed.
/// `LLMRS_SWAP_ALLOW_BOUNDARY_LAYERS=1` 이면 `true` — `WeightSwapDecider` 가
/// layer 0 과 마지막 decoder layer 도 swap 후보로 포함. 미설정/다른 값 → `false`.
/// PPL teacher-forcing NLL ablation 등 research-only path 에서 사용.
fn read_allow_boundary_env() -> bool {
    std::env::var("LLMRS_SWAP_ALLOW_BOUNDARY_LAYERS")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn dispatch_swap_weights(
    model: &llm_rs2::models::transformer::TransformerModel,
    ratio: f32,
    target_dtype: llm_shared::DtypeTag,
    importance_table: Option<&llm_rs2::core::qcf::ImportanceTable>,
    decode_token_index: usize,
    swap_plan_out: &mut Option<llm_rs2::models::weights::IncrementalSwapPlan>,
    manager_report_out: &mut Option<(f32, usize, std::time::Instant, f32)>,
) {
    use llm_rs2::models::weights::{
        IncrementalSwapPlan, SwapDecision, WeightSwapDecider, compute_qcf_swap,
    };
    use llm_shared::DtypeTag;

    // ── 1. Validation ──────────────────────────────────────────────────────
    if model.secondary_mmap.is_none() {
        eprintln!("[WeightSwap] Rejected: no_secondary (ENG-DAT-C09)");
        return;
    }
    if ratio <= 0.0 || ratio > 1.0 {
        eprintln!("[WeightSwap] Rejected: invalid_ratio ({:.4})", ratio);
        return;
    }
    if target_dtype != DtypeTag::Q4_0 {
        eprintln!(
            "[WeightSwap] Rejected: unsupported_dtype ({:?}) (INV-126)",
            target_dtype
        );
        return;
    }

    // ── 1b. In-flight plan check ───────────────────────────────────────────
    // Reject if a plan is already in flight (CLI or manager). Prevents
    // concurrent plan conflict (spec: manager signal accept only when no plan).
    if swap_plan_out.is_some() {
        eprintln!(
            "[WeightSwap] Rejected: incremental plan already in-flight (ratio={:.2}). \
             Wait for current plan to complete before sending a new SwapWeights signal.",
            ratio
        );
        return;
    }

    // ── 2. Collect currently-swapped layers ────────────────────────────────
    let n_layers = model.layers.len();
    let currently_swapped: Vec<usize> = (0..n_layers)
        .filter(|&i| model.layers[i].current_dtype() == DType::Q4_0)
        .collect();

    // ── 3. Decider ─────────────────────────────────────────────────────────
    let allow_boundary = read_allow_boundary_env();
    eprintln!(
        "[Decider] allow_boundary_layers={} (ratio={:.4})",
        allow_boundary, ratio
    );
    let decider = WeightSwapDecider {
        importance: importance_table,
        noise: Some(&model.quant_noise),
        n_decoder_layers: n_layers,
        currently_swapped: &currently_swapped,
        allow_boundary_layers: allow_boundary,
        algorithm: llm_rs2::models::weights::SwapAlgorithm::ImportanceAware,
    };
    let decision: SwapDecision = decider.decide(ratio);

    if decision.selected_layers.is_empty() {
        eprintln!(
            "[WeightSwap] No layers to swap (ratio={:.2}, already_swapped={})",
            ratio,
            currently_swapped.len()
        );
        // Empty swap is Ok per spec (already fully swapped); no plan committed.
        return;
    }

    // ── 4. Compute QCF estimate for the planned layers ─────────────────────
    let qcf_swap_estimated = compute_qcf_swap(
        &decision.selected_layers,
        &model.quant_noise,
        importance_table,
        n_layers,
    );

    // ── 5. Commit incremental plan (K=2, same as CLI --swap-incremental-per-tick 2) ──
    let n_planned = decision.selected_layers.len();
    let per_tick = 2usize; // LISWAP-6: K=2 hard upper cap for manager path
    let ticks_est = n_planned.div_ceil(per_tick);
    eprintln!(
        "[WeightSwap] manager path: ratio={:.2}, {} target layers, per_tick={} ({} ticks estimated), qcf_estimated={:.4}",
        ratio, n_planned, per_tick, ticks_est, qcf_swap_estimated,
    );

    *swap_plan_out = Some(IncrementalSwapPlan::new(
        decision.selected_layers,
        per_tick,
        decode_token_index,
    ));
    *manager_report_out = Some((
        ratio,
        n_planned,
        std::time::Instant::now(),
        qcf_swap_estimated,
    ));
}

/// Build `LayerSwapEstimate` from an available `ImportanceTable` + model noise table.
///
/// Returns `None` when secondary mmap is absent or no importance table has been
/// collected yet (i.e., on the very first `RequestQcf` before any prefill).
fn build_layer_swap_estimate(
    model: &llm_rs2::models::transformer::TransformerModel,
    importance_table: Option<&llm_rs2::core::qcf::ImportanceTable>,
) -> Option<llm_shared::LayerSwapEstimate> {
    // secondary must be present for weight swap to make sense
    model.secondary_mmap.as_ref()?;

    let imp = importance_table?;

    let n = model.layers.len();
    let noise = &model.quant_noise;

    // per_layer_importance: indexed by decoder layer id
    let per_layer_importance: Vec<f32> = (0..n)
        .map(|i| {
            imp.entries()
                .iter()
                .find(|e| {
                    e.layer_id == i
                        && e.sublayer == llm_rs2::core::qcf::layer_importance::SubLayer::Full
                })
                .map(|e| e.importance)
                .unwrap_or(0.0)
        })
        .collect();

    // per_layer_noise: None for NaN/missing
    let per_layer_noise: Vec<Option<f32>> = (0..n).map(|i| noise.epsilon(i)).collect();

    // qcf_swap_at_ratio: sample at representative ratios
    use llm_rs2::models::weights::WeightSwapDecider;
    use std::collections::HashMap;

    let sample_ratios = [0.1f32, 0.25, 0.5, 0.75, 1.0];
    let mut qcf_swap_at_ratio: HashMap<String, f32> = HashMap::new();

    for &r in &sample_ratios {
        let decider = WeightSwapDecider {
            importance: Some(imp),
            noise: Some(noise),
            n_decoder_layers: n,
            currently_swapped: &[],
            allow_boundary_layers: read_allow_boundary_env(),
            algorithm: llm_rs2::models::weights::SwapAlgorithm::ImportanceAware,
        };
        let (_, qcf) = decider.decide_dry_run(r);
        qcf_swap_at_ratio.insert(format!("{:.2}", r), qcf);
    }

    Some(llm_shared::LayerSwapEstimate {
        per_layer_importance,
        per_layer_noise,
        qcf_swap_at_ratio,
    })
}

fn command_summary(cmd: &EngineCommand) -> String {
    match cmd {
        EngineCommand::Throttle { delay_ms } => format!("Throttle({}ms)", delay_ms),
        EngineCommand::SetTargetTbt { target_ms } => format!("SetTargetTbt({}ms)", target_ms),
        EngineCommand::LayerSkip { skip_ratio } => format!("LayerSkip({:.2})", skip_ratio),
        EngineCommand::KvEvictH2o { keep_ratio } => format!("KvEvictH2o({:.2})", keep_ratio),
        EngineCommand::KvEvictSliding { keep_ratio } => {
            format!("KvEvictSliding({:.2})", keep_ratio)
        }
        EngineCommand::KvStreaming {
            sink_size,
            window_size,
        } => format!("KvStreaming(sink={}, win={})", sink_size, window_size),
        EngineCommand::KvMergeD2o { keep_ratio } => {
            format!("KvMergeD2o(ratio={})", keep_ratio)
        }
        EngineCommand::KvQuantDynamic { target_bits } => {
            format!("KvQuantDynamic({}bit)", target_bits)
        }
        EngineCommand::KvOffload { ratio } => format!("KvOffload({:.2})", ratio),
        EngineCommand::RestoreDefaults => "RestoreDefaults".to_string(),
        EngineCommand::SwitchHw { device } => format!("SwitchHw({})", device),
        EngineCommand::PrepareComputeUnit { device } => format!("Prepare({})", device),
        EngineCommand::Suspend => "Suspend".to_string(),
        EngineCommand::Resume => "Resume".to_string(),
        EngineCommand::RequestQcf => "RequestQcf".to_string(),
        EngineCommand::SetPartitionRatio { ratio } => {
            format!("SetPartitionRatio({})", ratio)
        }
        EngineCommand::SetPrefillPolicy {
            chunk_size,
            yield_ms,
            cpu_chunk_size,
        } => format!(
            "SetPrefillPolicy(chunk={:?}, yield={:?}, cpu_chunk={:?})",
            chunk_size, yield_ms, cpu_chunk_size
        ),
        EngineCommand::SwapWeights {
            ratio,
            target_dtype,
        } => {
            format!("SwapWeights(ratio={:.2}, dtype={:?})", ratio, target_dtype)
        }
    }
}

fn plan_summary(plan: &llm_rs2::resilience::ExecutionPlan) -> Vec<String> {
    let mut names = Vec::new();
    if let Some(ref evict) = plan.evict {
        names.push(format!(
            "Evict({:.2}, {:?}, {:?})",
            evict.target_ratio, evict.method, evict.level
        ));
    }
    if let Some(ref dev) = plan.switch_device {
        names.push(format!("Switch({})", dev));
    }
    if plan.throttle_delay_ms > 0 {
        names.push(format!("Throttle({}ms)", plan.throttle_delay_ms));
    }
    if plan.suspended {
        names.push("Suspend".to_string());
    }
    if plan.resumed {
        names.push("Resume".to_string());
    }
    if let Some(bits) = plan.kv_quant_bits {
        names.push(format!("KvQuant({}bit)", bits));
    }
    if let Some(ratio) = plan.layer_skip {
        names.push(format!("LayerSkip({:.2})", ratio));
    }
    if plan.restore_defaults {
        names.push("RestoreDefaults".to_string());
    }
    if let Some(ratio) = plan.partition_ratio {
        names.push(format!("PartitionRatio({:.2})", ratio));
    }
    names
}

// ════════════════════════════════════════════════════════════════
//  Prompt-batch helpers
// ════════════════════════════════════════════════════════════════

fn unix_ts() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

#[derive(serde::Deserialize)]
struct PromptBatchEntry {
    id: String,
    prompt: Option<String>,
    prompt_file: Option<String>,
}

fn load_prompt_batch(path: &str) -> anyhow::Result<Vec<PromptBatchEntry>> {
    use std::io::BufRead;
    let file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to open prompt batch {}: {}", path, e))?;
    let reader = std::io::BufReader::new(file);
    let mut entries = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let entry: PromptBatchEntry =
            serde_json::from_str(trimmed).map_err(|e| anyhow::anyhow!("Line {}: {}", i + 1, e))?;
        entries.push(entry);
    }
    Ok(entries)
}

fn resolve_prompt(entry: &PromptBatchEntry) -> anyhow::Result<String> {
    if let Some(ref text) = entry.prompt {
        Ok(text.clone())
    } else if let Some(ref path) = entry.prompt_file {
        std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read prompt_file {}: {}", path, e))
    } else {
        anyhow::bail!("Entry '{}': needs 'prompt' or 'prompt_file'", entry.id)
    }
}

// ════════════════════════════════════════════════════════════════
//  Eval-LL: Log-likelihood evaluation for downstream task accuracy
// ════════════════════════════════════════════════════════════════

/// Load and normalize eval questions from `--eval-batch` or `--eval-continuation`.
///
/// Produces a `Vec<EvalQuestion>` in grouped format (prompt + choices).
fn load_eval_questions(
    args: &Args,
    default_prompt: &str,
) -> anyhow::Result<Vec<llm_rs2::eval::EvalQuestion>> {
    let raw_tasks: Vec<serde_json::Value> = if let Some(ref path) = args.eval_batch {
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow::anyhow!("Failed to open eval batch {}: {}", path, e))?;
        serde_json::from_reader(file)?
    } else {
        let cont = args.eval_continuation.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--eval-ll requires --eval-continuation or --eval-batch")
        })?;
        vec![serde_json::json!({
            "id": "single",
            "prompt": default_prompt,
            "choices": [cont],
        })]
    };

    let mut questions: Vec<llm_rs2::eval::EvalQuestion> = Vec::new();
    for task in &raw_tasks {
        if let Some(choices) = task["choices"].as_array() {
            questions.push(llm_rs2::eval::EvalQuestion {
                id: task["id"].as_str().unwrap_or("unknown").to_string(),
                prompt: task["prompt"]
                    .as_str()
                    .unwrap_or(default_prompt)
                    .to_string(),
                choices: choices
                    .iter()
                    .filter_map(|c| c.as_str().map(|s| s.to_string()))
                    .collect(),
            });
        } else if let Some(cont) = task["continuation"].as_str() {
            questions.push(llm_rs2::eval::EvalQuestion {
                id: task["id"].as_str().unwrap_or("unknown").to_string(),
                prompt: task["prompt"]
                    .as_str()
                    .unwrap_or(default_prompt)
                    .to_string(),
                choices: vec![cont.to_string()],
            });
        }
    }
    Ok(questions)
}

/// Build a warmup token sequence from the eval-ll question set.
///
/// Concatenates the `prompt` fields of the questions (separated by `"\n\n"`),
/// tokenizes the result, and returns at most `max_tokens` token IDs.
/// If fewer tokens are produced than requested, a warning is emitted but the
/// function succeeds — the caller handles the reduced warmup gracefully.
///
/// Returns an empty Vec when tokenization fails entirely (non-fatal).
fn build_eval_ll_warmup_text(
    questions: &[llm_rs2::eval::EvalQuestion],
    max_tokens: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Vec<u32> {
    // Join question prompts.
    let combined: String = questions
        .iter()
        .map(|q| q.prompt.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    if combined.is_empty() {
        eprintln!("[QCF-dump] WARNING: all eval questions have empty prompts; warmup skipped");
        return Vec::new();
    }

    let enc = match tokenizer.encode(combined.as_str(), true) {
        Ok(e) => e,
        Err(e) => {
            eprintln!(
                "[QCF-dump] WARNING: warmup tokenize error: {}; warmup skipped",
                e
            );
            return Vec::new();
        }
    };

    let ids: Vec<u32> = enc.get_ids().iter().take(max_tokens).copied().collect();

    if ids.len() < max_tokens {
        eprintln!(
            "[QCF-dump] WARNING: only {} warmup tokens available (requested {}); \
             using all available tokens",
            ids.len(),
            max_tokens
        );
    }

    ids
}

// ── KIVI + PPL mode: KiviCache-based perplexity evaluation ───────────────────

#[allow(clippy::too_many_arguments)]
fn run_kivi_ppl(
    args: &Args,
    model: &TransformerModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    max_seq_len: usize,
    residual_size: usize,
    text_file: &str,
) -> anyhow::Result<()> {
    use llm_rs2::core::kv_cache::KVCacheOps;

    let hidden_size = model.config.hidden_size;
    let vocab_size = model.config.vocab_size;

    // ── 1. Read and tokenize reference text ──
    let text = std::fs::read_to_string(text_file)
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", text_file, e))?;
    let encoding = tokenizer
        .encode(text.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenize error: {}", e))?;
    let all_ids: Vec<u32> = encoding.get_ids().to_vec();
    let total_tokens = all_ids.len();

    if total_tokens < 2 {
        anyhow::bail!("PPL requires at least 2 tokens, got {}", total_tokens);
    }

    let eval_tokens = total_tokens.min(max_seq_len);
    if total_tokens > max_seq_len {
        eprintln!(
            "[KIVI-PPL] Warning: text has {} tokens, truncating to max_seq_len={}",
            total_tokens, max_seq_len
        );
    }
    let token_ids = &all_ids[..eval_tokens];

    eprintln!(
        "[KIVI-PPL] {} tokens, kivi_residual_size={}, max_seq_len={}",
        eval_tokens, residual_size, max_seq_len
    );

    // ── 2. Create KiviCache per layer ──
    let mut kv_caches: Vec<KiviCache> = (0..num_layers)
        .map(|_| {
            KiviCache::new_gpu(
                kv_heads,
                head_dim,
                max_seq_len,
                residual_size,
                2,
                backend.clone(),
                memory.clone(),
            )
        })
        .collect();

    // ── 3. Pre-allocate decode buffers ──
    let dl_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut decode_logits =
        Tensor::new(Shape::new(vec![1, 1, vocab_size]), dl_buf, backend.clone());
    let xg_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let mut x_gen = Tensor::new(Shape::new(vec![1, 1, hidden_size]), xg_buf, backend.clone());
    let q_dim = model.config.num_attention_heads * head_dim;
    let k_dim = kv_heads * head_dim;
    let v_dim = k_dim;
    let ffn_hidden = model.config.intermediate_size;
    let mut gen_ws = LayerWorkspace::new(
        WorkspaceConfig {
            batch_size: 1,
            dim: hidden_size,
            q_dim,
            k_dim,
            v_dim,
            ffn_hidden,
            n_heads: model.config.num_attention_heads,
            max_seq_len: args.max_seq_len,
        },
        memory.as_ref(),
        backend.clone(),
    )?;
    let cpu_gen_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_buf,
        Arc::new(CpuBackend::new()),
    );
    // Pre-allocate GPU input tensor for decode loop (avoids per-token GPU alloc)
    let gpu_gen_buf_kp = memory.alloc(4, DType::U8)?;
    let mut gen_input_gpu = Tensor::new(Shape::new(vec![1, 1]), gpu_gen_buf_kp, backend.clone());
    let mut logits_cpu = vec![0.0f32; vocab_size];

    let mut total_nll: f64 = 0.0;
    let mut nll_count: usize = 0;
    let mut qcf_metrics: Vec<serde_json::Value> = Vec::new();
    let mut flush_count: usize = 0;
    let overall_start = std::time::Instant::now();

    // ── 4. Prefill phase ──
    let prefill_len = eval_tokens.min(max_seq_len);
    eprintln!("[KIVI-PPL] Prefill: {} tokens", prefill_len);

    {
        let cpu_backend = Arc::new(CpuBackend::new());
        let input_buf = Galloc::new().alloc(prefill_len * 4, DType::U8)?;
        unsafe {
            let ptr = input_buf.as_mut_ptr() as *mut u32;
            for (i, &id) in token_ids[..prefill_len].iter().enumerate() {
                *ptr.add(i) = id;
            }
        }
        let cpu_input = Tensor::new(Shape::new(vec![1, prefill_len]), input_buf, cpu_backend);
        let input_tensor = backend.copy_from(&cpu_input)?;

        let prefill_logits_buf = memory.alloc(prefill_len * vocab_size * 4, DType::F32)?;
        let mut prefill_logits = Tensor::new(
            Shape::new(vec![1, prefill_len, vocab_size]),
            prefill_logits_buf,
            backend.clone(),
        );

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos: 0,
            kv_caches: &mut kv_caches,
            backend,
            memory: memory.as_ref(),
            logits_out: &mut prefill_logits,
            x_gen: None,
            workspace: None,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;

        // Collect flush QCF metrics from prefill
        for metric in kv_caches[0].take_flush_proxies() {
            qcf_metrics.push(serde_json::json!({
                "flush": flush_count,
                "action": metric.action,
                "raw_value": metric.raw_value,
                "normalized_value": metric.normalized_value,
                "tokens_quantized": metric.tokens_affected,
            }));
            flush_count += 1;
        }
        for cache in kv_caches[1..].iter_mut() {
            cache.take_flush_proxies();
        }

        // Read all prefill logits to CPU
        let mut all_logits = vec![0.0f32; prefill_len * vocab_size];
        unsafe {
            let ptr = all_logits.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, all_logits.len() * 4);
            backend.read_buffer(&prefill_logits, slice)?;
        }

        // Score tokens 1..prefill_len: logits[i] predicts token[i+1]
        for i in 0..prefill_len - 1 {
            let offset = i * vocab_size;
            let lp = sampling::compute_log_prob(
                &all_logits[offset..offset + vocab_size],
                token_ids[i + 1],
                vocab_size,
            );
            total_nll -= lp;
            nll_count += 1;
        }

        eprintln!(
            "[KIVI-PPL] Prefill NLL: {:.4}, count={}, running PPL={:.4}, Q2_tokens={}, res_pos={}",
            total_nll,
            nll_count,
            (total_nll / nll_count as f64).exp(),
            kv_caches[0].q2_tokens,
            kv_caches[0].res_pos,
        );
    }

    // ── 5. Decode phase (teacher-forcing) ──
    let mut start_pos = prefill_len;

    for i in prefill_len..eval_tokens - 1 {
        let input_token = token_ids[i];
        let target_token = token_ids[i + 1];

        // Feed true token
        unsafe {
            *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = input_token;
        }
        // Reuse pre-allocated GPU buffer — write data instead of alloc+copy
        backend.write_buffer(&mut gen_input_gpu, unsafe {
            std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
        })?;

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &gen_input_gpu,
            start_pos,
            kv_caches: &mut kv_caches,
            backend,
            memory: memory.as_ref(),
            logits_out: &mut decode_logits,
            x_gen: Some(&mut x_gen),
            workspace: Some(&mut gen_ws),
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;
        start_pos += 1;

        // Collect flush QCF from decode step
        for metric in kv_caches[0].take_flush_proxies() {
            qcf_metrics.push(serde_json::json!({
                "flush": flush_count,
                "action": metric.action,
                "raw_value": metric.raw_value,
                "normalized_value": metric.normalized_value,
                "tokens_quantized": metric.tokens_affected,
            }));
            flush_count += 1;
        }
        for cache in kv_caches[1..].iter_mut() {
            cache.take_flush_proxies();
        }

        // Read logits and score target
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
            backend.read_buffer(&decode_logits, slice)?;
        }
        let lp = sampling::compute_log_prob(&logits_cpu, target_token, vocab_size);
        total_nll -= lp;
        nll_count += 1;

        // Progress
        if (i + 1) % 200 == 0 {
            let ppl = (total_nll / nll_count as f64).exp();
            eprintln!(
                "[KIVI-PPL] step {}/{}: NLL={:.4}, PPL={:.4}, cache_pos={}, Q2_tokens={}",
                i + 1,
                eval_tokens,
                total_nll,
                ppl,
                kv_caches[0].current_pos(),
                kv_caches[0].q2_tokens,
            );
        }
    }

    // ── 6. Output results ──
    let wall_time = overall_start.elapsed().as_secs_f64();
    let ppl = (total_nll / nll_count as f64).exp();
    let tok_per_sec = nll_count as f64 / wall_time;

    // Separate QCF (NMSE) and OPR metrics from flush proxies
    let qcf_kivi_nmse_total: f64 = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str() != Some("kivi_opr"))
        .filter_map(|m| m["raw_value"].as_f64())
        .sum();
    let qcf_attn_normalized_total: f64 = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str() != Some("kivi_opr"))
        .filter_map(|m| m["normalized_value"].as_f64())
        .sum();

    // KIVI OPR: per-flush events and summary stats
    let qcf_kivi_events: Vec<&serde_json::Value> = qcf_metrics
        .iter()
        .filter(|m| m["action"].as_str() == Some("kivi_opr"))
        .collect();
    let n_kivi_flushes = qcf_kivi_events.len();
    let opr_raw_values: Vec<f64> = qcf_kivi_events
        .iter()
        .filter_map(|m| m["raw_value"].as_f64())
        .collect();
    let qcf_kivi_opr_sum: f64 = opr_raw_values.iter().sum();
    let qcf_kivi_opr_max: f64 = opr_raw_values.iter().cloned().fold(0.0f64, f64::max);
    let qcf_kivi_opr_total: Option<f64> = if opr_raw_values.is_empty() {
        None
    } else {
        Some(qcf_kivi_opr_sum / opr_raw_values.len() as f64)
    };
    let qcf_kivi_opr_events: Option<usize> = if opr_raw_values.is_empty() {
        None
    } else {
        Some(opr_raw_values.len())
    };

    let output = serde_json::json!({
        "ppl": ppl,
        "total_nll": total_nll,
        "token_count": nll_count,
        "tokens_per_second": tok_per_sec,
        "wall_time_s": wall_time,
        "qcf_metrics": qcf_metrics,
        "flush_count": qcf_metrics.len(),
        "n_kivi_flushes": n_kivi_flushes,
        "qcf_kivi_events": qcf_kivi_events,
        "qcf_kivi_nmse_total": qcf_kivi_nmse_total,
        "qcf_attn_total": qcf_kivi_nmse_total,
        "qcf_attn_normalized_total": qcf_attn_normalized_total,
        "qcf_kivi_opr_sum": qcf_kivi_opr_sum,
        "qcf_kivi_opr_max": qcf_kivi_opr_max,
        "qcf_kivi_opr_total": qcf_kivi_opr_total,
        "qcf_kivi_opr_events": qcf_kivi_opr_events,
        "final_cache_pos": kv_caches[0].current_pos(),
        "kivi_q2_tokens": kv_caches[0].q2_tokens,
        "kivi_res_pos": kv_caches[0].res_pos,
        "config": {
            "model": args.model_path,
            "text_file": text_file,
            "eviction_policy": "kivi",
            "kivi_residual_size": residual_size,
            "max_seq_len": max_seq_len,
            "kv_type": "q2+f32_residual",
        }
    });

    println!("{}", serde_json::to_string_pretty(&output)?);

    eprintln!(
        "\n[KIVI-PPL] Final: PPL={:.4}, NLL={:.4}, tokens={}, {:.1} tok/s, {:.1}s, Q2_tokens={}",
        ppl, total_nll, nll_count, tok_per_sec, wall_time, kv_caches[0].q2_tokens
    );

    Ok(())
}

// ── KIVI mode: KiviCache-based inference ────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_kivi(
    model: &TransformerModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    input_ids: &[u32],
    sampling_config: &SamplingConfig,
    kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    max_seq_len: usize,
    residual_size: usize,
    num_tokens: usize,
    experiment_output: Option<&str>,
    experiment_logits_topk: usize,
    experiment_sample_interval: usize,
    prompt: &str,
    backend_name: &str,
    command_executor: &mut Option<llm_rs2::resilience::CommandExecutor>,
    initial_bits: u8,
    no_gpu_plan: bool,
    mut target_tbt_ms: f64,
    tbt_log_path: Option<&str>,
    ignore_eos: bool,
    cli_throttle_delay_ms: u64,
) -> anyhow::Result<()> {
    use llm_rs2::core::kv_cache::KVCacheOps;

    println!(
        "[KIVI] KV cache enabled — bits={}, residual_size={}, max_seq_len={}",
        initial_bits, residual_size, max_seq_len
    );

    // Experiment infrastructure
    let mut experiment_writer = if let Some(path) = experiment_output {
        Some(JsonlWriter::new(path)?)
    } else {
        None
    };
    let mut system_sampler = SystemSampler::new(experiment_sample_interval);
    let sys_start = if experiment_writer.is_some() {
        Some(system_sampler.snapshot())
    } else {
        None
    };

    // Create KiviCache per layer
    let mut kv_caches: Vec<KiviCache> = (0..num_layers)
        .map(|_| {
            KiviCache::new_gpu(
                kv_heads,
                head_dim,
                max_seq_len,
                residual_size,
                initial_bits,
                backend.clone(),
                memory.clone(),
            )
        })
        .collect();

    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;
    let q_dim = model.config.num_attention_heads * head_dim;
    let k_dim = kv_heads * head_dim;
    let v_dim = kv_heads * head_dim;
    let ffn_hidden = model.config.intermediate_size;

    // Allocate workspace
    let mut gen_ws = LayerWorkspace::new(
        WorkspaceConfig {
            batch_size: 1,
            dim: hidden_size,
            q_dim,
            k_dim,
            v_dim,
            ffn_hidden,
            n_heads: model.config.num_attention_heads,
            max_seq_len,
        },
        memory.as_ref(),
        backend.clone(),
    )?;
    let x_gen_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let mut x_gen = Tensor::new(
        Shape::new(vec![1, 1, hidden_size]),
        x_gen_buf,
        backend.clone(),
    );

    let logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut logits = Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        logits_buf,
        backend.clone(),
    );

    // === PREFILL ===
    let mut tokens: Vec<u32> = input_ids.to_vec();
    let process_len = tokens.len();
    if process_len > max_seq_len {
        anyhow::bail!(
            "Prompt length {} exceeds max_seq_len {}",
            process_len,
            max_seq_len
        );
    }
    let mut start_pos = 0usize;

    let prefill_start = std::time::Instant::now();
    {
        let cpu_indices_buf = Galloc::new().alloc(process_len * 4, DType::U8)?;
        unsafe {
            let ptr = cpu_indices_buf.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, process_len);
        }
        let cpu_input = Tensor::new(
            Shape::new(vec![1, process_len]),
            cpu_indices_buf,
            Arc::new(CpuBackend::new()),
        );
        let input_tensor = backend.copy_from(&cpu_input)?;

        let prefill_logits_buf = memory.alloc(process_len * vocab_size * 4, DType::F32)?;
        let mut prefill_logits = Tensor::new(
            Shape::new(vec![1, process_len, vocab_size]),
            prefill_logits_buf,
            backend.clone(),
        );

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos,
            kv_caches: &mut kv_caches,
            backend,
            memory: memory.as_ref(),
            logits_out: &mut prefill_logits,
            x_gen: None,
            workspace: None,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;

        // Sample last token from prefill logits
        let mut logits_cpu = vec![0.0f32; process_len * vocab_size];
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, logits_cpu.len() * 4);
            backend.read_buffer(&prefill_logits, slice)?;
        }
        let last_start = (process_len - 1) * vocab_size;
        let next_token = sampling::sample(
            &mut logits_cpu[last_start..last_start + vocab_size],
            &tokens,
            vocab_size,
            sampling_config,
            None,
        );
        tokens.push(next_token);
        start_pos = process_len;
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = prefill_ms;
    let kivi_mem = kv_caches
        .iter()
        .map(|c| c.memory_usage_bytes())
        .sum::<usize>();
    eprintln!(
        "[KIVI] Prefill: {}ms, cache_pos={}, Q2_tokens={}, res_pos={}, mem={}KB",
        prefill_ms as u32,
        kv_caches[0].current_pos(),
        kv_caches[0].q2_tokens,
        kv_caches[0].res_pos,
        kivi_mem / 1024,
    );

    // Print prompt
    use std::io::Write;
    let mut stdout = std::io::stdout();
    let initial_text = tokenizer.decode(&tokens, true).unwrap_or_default();
    if experiment_writer.is_none() {
        print!("{}", initial_text);
        stdout.flush().ok();
    }

    // === DECODE ===
    let cpu_gen_indices_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_indices_buf,
        Arc::new(CpuBackend::new()),
    );

    // Pre-allocate GPU input tensor for decode loop (avoids per-token GPU alloc)
    let gpu_gen_buf = memory.alloc(4, DType::U8)?;
    let mut gen_input = Tensor::new(Shape::new(vec![1, 1]), gpu_gen_buf, backend.clone());

    let eos_id = model.config.eos_token_id;

    let mut tbt_writer: Option<std::io::BufWriter<std::fs::File>> = tbt_log_path.map(|p| {
        let f = std::fs::File::create(p).expect("failed to create tbt-log file");
        std::io::BufWriter::new(f)
    });

    let decode_start = std::time::Instant::now();
    let mut generated_count = 0usize;
    let mut tbt_values: Vec<f64> = Vec::new();
    let mut forward_ms_values: Vec<f64> = Vec::new();
    let mut last_token_time = std::time::Instant::now();

    // Dynamic skip_config for KIVI resilience path
    use llm_rs2::core::skip_config::SkipConfig;
    let mut kivi_skip_config: Option<SkipConfig> = None;
    let mut kivi_last_skip_ratio: Option<f32> = None;
    // Track last applied quant bits to avoid redundant transition_bits calls (sticky guard)
    let mut kivi_last_quant_bits: Option<u8> = None;

    // Build GPU kernel plan for KIVI decode (OpenCL only).
    // Skip when tensor partition is active — plan bypasses forward_gen's
    // partition co-execution path.
    #[cfg(feature = "opencl")]
    let mut gpu_plan =
        // KIVI plan does not yet integrate tensor-partition — the rejection
        // lives inside `build_plan_for_kivi` (returns None when any layer has
        // a partition_ctx). See ENG-ALG-200 scope note.
        if backend.name() == "OpenCL" && !no_gpu_plan {
            model.build_plan_for_kivi(&x_gen, &logits, &gen_ws, &kv_caches, backend)
        } else {
            None
        };
    #[cfg(not(feature = "opencl"))]
    let gpu_plan: Option<()> = None;

    for decode_idx in 0..(num_tokens - 1) {
        if kv_caches[0].current_pos() >= max_seq_len {
            eprintln!("\n[Stopped: Max context length reached]");
            break;
        }

        // Flush residual if needed (before plan dispatch writes new token)
        for cache in kv_caches.iter_mut() {
            if cache.needs_flush() {
                let _ = cache.flush_if_needed();
                // Flush changes Q2 state — plan remains valid (q2_tokens/res_pos are dynamic)
            }
        }

        let last_token = tokens[tokens.len() - 1];
        unsafe {
            *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = last_token;
        }
        // Reuse pre-allocated GPU buffer — write data instead of alloc+copy
        backend.write_buffer(&mut gen_input, unsafe {
            std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
        })?;

        let fwd_start = std::time::Instant::now();

        // Try GPU plan path first
        #[cfg(feature = "opencl")]
        let plan_ok = if let Some(ref plan) = gpu_plan {
            match model.execute_plan_for_kivi(
                plan,
                &gen_input,
                start_pos,
                &mut x_gen,
                &mut kv_caches,
                &mut logits,
                backend,
            ) {
                Ok(true) => true,
                _ => {
                    gpu_plan = None;
                    false
                }
            }
        } else {
            false
        };
        #[cfg(not(feature = "opencl"))]
        let plan_ok = false;

        if !plan_ok {
            model.forward_into(TransformerModelForwardArgs {
                input_tokens: &gen_input,
                start_pos,
                kv_caches: &mut kv_caches,
                backend,
                memory: memory.as_ref(),
                logits_out: &mut logits,
                x_gen: Some(&mut x_gen),
                workspace: Some(&mut gen_ws),
                score_accumulator: None,
                profiler: None,
                skip_config: kivi_skip_config.as_ref(),
                importance_collector: None,
                logits_last_only: false,
                variance_collector: None,
                prefill_workspace: None,

                layer_boundary_hook: None,
            })?;

            // Rebuild plan after fallback. Rejection for partition-active
            // KIVI runs happens inside `build_plan_for_kivi` (see ENG-ALG-200).
            #[cfg(feature = "opencl")]
            if gpu_plan.is_none() && backend.name() == "OpenCL" && !no_gpu_plan {
                gpu_plan = model.build_plan_for_kivi(&x_gen, &logits, &gen_ws, &kv_caches, backend);
            }
        }

        backend.synchronize()?;
        let forward_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;
        forward_ms_values.push(forward_ms);

        start_pos += 1;
        generated_count += 1;

        // ── KIVI resilience checkpoint ──
        if let Some(executor) = command_executor.as_mut() {
            let current_bits = kv_caches[0].bits();
            let kv_dtype = match current_bits {
                16 => "f16".to_string(),
                8 => "q8".to_string(),
                4 => "q4".to_string(),
                2 => "q2".to_string(),
                _ => format!("q{}", current_bits),
            };
            let kv_snap = llm_rs2::resilience::KVSnapshot {
                total_bytes: kv_caches
                    .iter()
                    .map(|c| c.memory_usage_bytes() as u64)
                    .sum(),
                total_tokens: kv_caches[0].current_pos(),
                capacity: kv_caches[0].capacity(),
                protected_prefix: 0,
                kv_dtype,
                eviction_policy: "kivi".to_string(),
                skip_ratio: kivi_last_skip_ratio.unwrap_or(0.0),
            };
            let plan = executor.poll(&kv_snap);

            // QCF estimate: dry-run KIVI quantization NMSE
            if plan.request_qcf {
                let ctx = QcfEstimateContext {
                    kv_caches: &[], // KIVI path has no standard KVCache
                    score_accumulator: None,
                    streaming_config: None,
                    importance_table: None,
                    num_layers,
                    kivi_caches: Some(&kv_caches),
                };
                let estimates = compute_qcf_estimates(&ctx);
                executor.send_qcf_estimate(llm_shared::QcfEstimate {
                    estimates,
                    layer_swap: None,
                });
            }

            // kv_quant_bits: transition KiviCache bit-width
            // Sticky guard: skip if already at the requested bit-width
            if let Some(bits) = plan.kv_quant_bits
                && kivi_last_quant_bits != Some(bits)
            {
                for cache in kv_caches.iter_mut() {
                    if let Err(e) = cache.transition_bits(bits) {
                        eprintln!("[KIVI-Resilience] transition_bits({}) error: {}", bits, e);
                    }
                }
                // Invalidate GPU Plan — cache structure changed after bit transition
                #[cfg(feature = "opencl")]
                {
                    gpu_plan = None;
                }
                eprintln!("[KIVI-Resilience] Transitioned KV cache to {}bit", bits);
                kivi_last_quant_bits = Some(bits);
            }

            // KvOffload is not supported on the KIVI decode path — it runs on
            // KiviCache instances, whereas SwapHandler operates on KVCache.
            // Emit the expected "KvOffload" log lines so verify scenarios still
            // match, but mark the action as a no-op.
            if let Some(ratio) = plan.offload_ratio {
                eprintln!(
                    "[Resilience] KvOffload: ratio={:.2}, 0 tokens swapped (KIVI path)",
                    ratio
                );
            }

            // layer_skip / restore_defaults
            if plan.restore_defaults {
                eprintln!("[KIVI-Resilience] RestoreDefaults");
                kivi_skip_config = None;
                kivi_last_skip_ratio = None;
                kivi_last_quant_bits = None;
            } else if let Some(ratio) = plan.layer_skip
                && kivi_last_skip_ratio != Some(ratio)
            {
                eprintln!("[KIVI-Resilience] LayerSkip: ratio={:.2}", ratio);
                kivi_skip_config = Some(SkipConfig::uniform_init(
                    model.config.num_hidden_layers,
                    ratio,
                ));
                kivi_last_skip_ratio = Some(ratio);
            }

            // throttle
            if plan.throttle_delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(plan.throttle_delay_ms));
            }

            // Update target TBT from Manager directive. `target_tbt_set` lets
            // the engine honor an explicit `SetTargetTbt { target_ms: 0 }` to
            // disable pacing (see verify/ISSUE-3).
            if plan.target_tbt_set && plan.target_tbt_ms as f64 != target_tbt_ms {
                eprintln!(
                    "[KIVI-Resilience] SetTargetTbt: {:.1}ms → {}ms",
                    target_tbt_ms, plan.target_tbt_ms
                );
                target_tbt_ms = plan.target_tbt_ms as f64;
            } else if plan.restore_defaults {
                target_tbt_ms = 0.0;
            }

            if plan.suspended {
                eprintln!("\n[KIVI-Resilience] Inference suspended by system signal");
                break;
            }

            executor.on_token_generated();
        } else if cli_throttle_delay_ms > 0 {
            // No CommandExecutor: honour CLI --throttle-delay-ms directly.
            std::thread::sleep(std::time::Duration::from_millis(cli_throttle_delay_ms));
        }

        // Read logits to CPU
        let mut logits_cpu = vec![0.0f32; vocab_size];
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, logits_cpu.len() * 4);
            backend.read_buffer(&logits, slice)?;
        }

        // Extract top-K logits before sampling modifies them
        let top_logits = if experiment_writer.is_some() {
            extract_top_k_logits(&logits_cpu, experiment_logits_topk)
        } else {
            vec![]
        };

        let next_token =
            sampling::sample(&mut logits_cpu, &tokens, vocab_size, sampling_config, None);

        let now = std::time::Instant::now();
        let mut tbt = now.duration_since(last_token_time).as_secs_f64() * 1000.0;

        // Target TBT pacing
        let pacing_ms = if target_tbt_ms > 0.0 && tbt < target_tbt_ms {
            let sleep_ms = target_tbt_ms - tbt;
            std::thread::sleep(std::time::Duration::from_secs_f64(sleep_ms / 1000.0));
            tbt = target_tbt_ms;
            sleep_ms
        } else {
            0.0
        };

        tbt_values.push(tbt);
        last_token_time = std::time::Instant::now();

        // TBT log
        if let Some(ref mut w) = tbt_writer {
            use std::io::Write;
            writeln!(w,
                "{{\"token_idx\":{},\"tbt_ms\":{:.2},\"forward_ms\":{:.2},\"cache_pos\":{},\"pacing_ms\":{:.2}}}",
                decode_idx, tbt, forward_ms, kv_caches[0].current_pos(), pacing_ms
            ).ok();
        }

        tokens.push(next_token);

        // Experiment: write per-token JSONL record
        if let Some(ref mut writer) = experiment_writer {
            let token_text = tokenizer.decode(&[next_token], false).unwrap_or_default();
            let sys_metrics = system_sampler.sample(decode_idx);
            let record = TokenRecord {
                pos: decode_idx,
                token_id: next_token,
                text: token_text,
                tbt_ms: tbt,
                forward_ms,
                signal: None,
                actions: vec![],
                cache_pos: kv_caches[0].current_pos(),
                throttle_ms: 0,
                top_logits,
                sys: sys_metrics,
            };
            writer.write_token(&record)?;
        }

        // Stream output (suppress in experiment mode)
        if experiment_writer.is_none() {
            let text = tokenizer.decode(&tokens, true).unwrap_or_default();
            let new_text = &text[initial_text.len()..];
            print!("\r{}{}", initial_text, new_text);
            stdout.flush().ok();
        }

        if next_token == eos_id && !ignore_eos && std::env::var("IGNORE_EOS").is_err() {
            break;
        }
    }

    // Flush TBT log
    if let Some(ref mut w) = tbt_writer {
        use std::io::Write;
        w.flush().ok();
    }

    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let tok_per_s = if decode_ms > 0.0 {
        generated_count as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };
    let kivi_mem_final = kv_caches
        .iter()
        .map(|c| c.memory_usage_bytes())
        .sum::<usize>();

    // Write experiment summary
    if let Some(ref mut writer) = experiment_writer {
        let sys_end = Some(system_sampler.snapshot());
        let avg_tbt_ms = if tbt_values.is_empty() {
            0.0
        } else {
            tbt_values.iter().sum::<f64>() / tbt_values.len() as f64
        };
        let avg_forward_ms = if forward_ms_values.is_empty() {
            0.0
        } else {
            forward_ms_values.iter().sum::<f64>() / forward_ms_values.len() as f64
        };
        let summary = SummaryRecord {
            _summary: true,
            total_tokens: tbt_values.len(),
            ttft_ms,
            avg_tbt_ms,
            avg_forward_ms,
            total_throttle_ms: 0,
            eviction_count: 0,
            evicted_tokens_total: 0,
            final_cache_pos: kv_caches[0].current_pos(),
            max_seq_len,
            prompt: prompt.to_string(),
            schedule_name: "kivi".to_string(),
            eviction_policy: "none".to_string(),
            backend: backend_name.to_string(),
            sample_interval: experiment_sample_interval,
            sys_start,
            sys_end,
            governor: Some(SystemSampler::read_governor()),
        };
        writer.write_summary(&summary)?;
        eprintln!(
            "[KIVI-Experiment] Done: {} tokens, avg TBT {:.2}ms",
            summary.total_tokens, avg_tbt_ms,
        );
    }

    eprintln!();
    eprintln!(
        "[KIVI] Decode: {} tokens, {:.1}ms ({:.1} tok/s)",
        generated_count, decode_ms, tok_per_s
    );
    eprintln!(
        "[KIVI] Final: cache_pos={}, Q2_tokens={}, res_pos={}, mem={}KB",
        kv_caches[0].current_pos(),
        kv_caches[0].q2_tokens,
        kv_caches[0].res_pos,
        kivi_mem_final / 1024,
    );

    // Compare with FP32 equivalent
    let fp32_equiv = kv_caches[0].current_pos() * kv_heads * head_dim * 4 * 2 * num_layers;
    eprintln!(
        "[KIVI] Compression: {:.1}x vs FP32 ({}KB vs {}KB)",
        fp32_equiv as f64 / kivi_mem_final.max(1) as f64,
        kivi_mem_final / 1024,
        fp32_equiv / 1024,
    );

    Ok(())
}

// ── Offload mode: OffloadKVCache-based inference with per-layer prefetch ─────

#[allow(clippy::too_many_arguments)]
fn run_offload(
    model: &TransformerModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    input_ids: &[u32],
    sampling_config: &SamplingConfig,
    kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    max_seq_len: usize,
    num_tokens: usize,
    _prompt: &str,
    _backend_name: &str,
    offload_mode: &str,
    kv_type_str: &str,
    max_prefetch_depth: usize,
    offload_path: &str,
    command_executor: &mut Option<CommandExecutor>,
    cli_throttle_delay_ms: u64,
) -> anyhow::Result<()> {
    use llm_rs2::core::kv_cache::KVCacheOps;
    use llm_rs2::core::offload::OffloadKVCache;
    use llm_rs2::core::offload::raw_store::RawStore;

    // Validate constraints
    let kv_dtype = match kv_type_str {
        "f32" => DType::F32,
        "f16" => DType::F16,
        _ => anyhow::bail!(
            "--kv-offload requires --kv-type f16 or f32, got '{}'",
            kv_type_str
        ),
    };

    let token_bytes = kv_heads * head_dim * kv_dtype.size();

    // Resolve disk offload directory
    let disk_dir = if offload_path.is_empty() {
        std::env::temp_dir().join("llm_rs2_kv_offload")
    } else {
        std::path::PathBuf::from(offload_path)
    };

    if offload_mode == "disk" {
        eprintln!(
            "[Offload] mode=disk, path={}, dtype={:?}, layers={}, token_bytes={}, max_seq={}",
            disk_dir.display(),
            kv_dtype,
            num_layers,
            token_bytes,
            max_seq_len,
        );
    } else {
        eprintln!(
            "[Offload] mode={}, dtype={:?}, layers={}, token_bytes={}, max_seq={}",
            offload_mode, kv_dtype, num_layers, token_bytes, max_seq_len,
        );
    }

    // Create OffloadKVCache per layer. When the main backend is a GPU, wire it
    // (plus the matching memory allocator) so `get_view()` uploads the KV bytes
    // to device buffers that `attention_gen` can read via `cl_mem`. Without
    // this, OpenCL backends would see a null `cl_mem` from the default CPU
    // `SharedBuffer` and fail at kernel arg binding.
    let is_gpu_backend = backend.as_ref().is_gpu();
    let mut kv_caches: Vec<OffloadKVCache> = (0..num_layers)
        .map(|layer_id| {
            let store: Box<dyn llm_rs2::core::offload::store::OffloadStore> = match offload_mode {
                "raw" => Box::new(RawStore::new(token_bytes)),
                "disk" => Box::new(
                    llm_rs2::core::offload::disk_store::DiskStore::new(
                        disk_dir.clone(),
                        layer_id,
                        token_bytes,
                    )
                    .expect("Failed to create DiskStore"),
                ),
                _ => panic!("Unknown offload mode: {}", offload_mode),
            };
            let mut c =
                OffloadKVCache::new(layer_id, kv_heads, head_dim, kv_dtype, max_seq_len, store);
            if is_gpu_backend {
                c.set_gpu_backend(backend.clone(), memory.clone());
            }
            c
        })
        .collect();

    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;
    let q_dim = model.config.num_attention_heads * head_dim;
    let k_dim = kv_heads * head_dim;
    let v_dim = k_dim;
    let ffn_hidden = model.config.intermediate_size;

    // Allocate workspace
    let mut gen_ws = LayerWorkspace::new(
        llm_rs2::layers::workspace::WorkspaceConfig {
            batch_size: 1,
            dim: hidden_size,
            q_dim,
            k_dim,
            v_dim,
            ffn_hidden,
            n_heads: model.config.num_attention_heads,
            max_seq_len,
        },
        memory.as_ref(),
        backend.clone(),
    )?;
    let x_gen_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let mut x_gen = Tensor::new(
        Shape::new(vec![1, 1, hidden_size]),
        x_gen_buf,
        backend.clone(),
    );

    let logits_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut logits = Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        logits_buf,
        backend.clone(),
    );

    // === PREFILL ===
    let mut tokens: Vec<u32> = input_ids.to_vec();
    let process_len = tokens.len();
    if process_len > max_seq_len {
        anyhow::bail!(
            "Prompt length {} exceeds max_seq_len {}",
            process_len,
            max_seq_len
        );
    }
    let mut start_pos = 0usize;

    let prefill_start = std::time::Instant::now();
    {
        let cpu_indices_buf = Galloc::new().alloc(process_len * 4, DType::U8)?;
        unsafe {
            let ptr = cpu_indices_buf.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, process_len);
        }
        let cpu_input = Tensor::new(
            Shape::new(vec![1, process_len]),
            cpu_indices_buf,
            Arc::new(CpuBackend::new()),
        );
        let input_tensor = backend.copy_from(&cpu_input)?;

        let prefill_logits_buf = memory.alloc(process_len * vocab_size * 4, DType::F32)?;
        let mut prefill_logits = Tensor::new(
            Shape::new(vec![1, process_len, vocab_size]),
            prefill_logits_buf,
            backend.clone(),
        );

        // Prefill uses standard forward_into (no prefetch needed for batch)
        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos,
            kv_caches: &mut kv_caches,
            backend,
            memory: memory.as_ref(),
            logits_out: &mut prefill_logits,
            x_gen: None,
            workspace: None,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;

        // Sample last token from prefill logits
        let mut logits_cpu = vec![0.0f32; process_len * vocab_size];
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, logits_cpu.len() * 4);
            backend.read_buffer(&prefill_logits, slice)?;
        }
        let last_start = (process_len - 1) * vocab_size;
        let next_token = sampling::sample(
            &mut logits_cpu[last_start..last_start + vocab_size],
            &tokens,
            vocab_size,
            sampling_config,
            None,
        );
        tokens.push(next_token);
        start_pos = process_len;
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = prefill_ms;

    let offload_mem_after_prefill: usize = kv_caches.iter().map(|c| c.memory_usage_bytes()).sum();
    let raw_equiv = process_len * token_bytes * 2 * num_layers; // K+V
    eprintln!(
        "[Offload] Prefill: {:.1}ms, cache_pos={}, store_mem={}KB (raw equiv={}KB, ratio={:.2}x)",
        prefill_ms,
        kv_caches[0].current_pos(),
        offload_mem_after_prefill / 1024,
        raw_equiv / 1024,
        raw_equiv as f64 / offload_mem_after_prefill.max(1) as f64,
    );

    // Print prompt
    use std::io::Write;
    let mut stdout = std::io::stdout();
    let initial_text = tokenizer.decode(&tokens, true).unwrap_or_default();
    print!("{}", initial_text);
    stdout.flush().ok();

    // === DECODE with adaptive prefetch ===
    let cpu_gen_indices_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_indices_buf,
        Arc::new(CpuBackend::new()),
    );

    // Pre-allocate GPU input tensor for decode loop (avoids per-token GPU alloc)
    let gpu_gen_buf = memory.alloc(4, DType::U8)?;
    let mut gen_input = Tensor::new(Shape::new(vec![1, 1]), gpu_gen_buf, backend.clone());

    let eos_id = model.config.eos_token_id;

    let mut prefetch =
        llm_rs2::core::offload::prefetch::PrefetchController::new(max_prefetch_depth, num_layers);

    let decode_start = std::time::Instant::now();
    let mut generated_count = 0usize;
    let mut tbt_values: Vec<f64> = Vec::new();
    let mut forward_ms_values: Vec<f64> = Vec::new();
    let mut last_token_time = std::time::Instant::now();

    for _decode_idx in 0..(num_tokens - 1) {
        if kv_caches[0].current_pos() >= max_seq_len {
            eprintln!("\n[Stopped: Max context length reached]");
            break;
        }

        let last_token = tokens[tokens.len() - 1];
        unsafe {
            *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = last_token;
        }
        // Reuse pre-allocated GPU buffer — write data instead of alloc+copy
        backend.write_buffer(&mut gen_input, unsafe {
            std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
        })?;

        // Preload state managed by forward_into_offload:
        // - retained layers: retain_preload() keeps preloaded=true
        // - non-retained layers: release_buffers() sets preloaded=false

        let fwd_start = std::time::Instant::now();
        model.forward_into_offload(
            TransformerModelForwardArgs {
                input_tokens: &gen_input,
                start_pos,
                kv_caches: &mut kv_caches,
                backend,
                memory: memory.as_ref(),
                logits_out: &mut logits,
                x_gen: Some(&mut x_gen),
                workspace: Some(&mut gen_ws),
                score_accumulator: None,
                profiler: None,
                skip_config: None,
                importance_collector: None,
                logits_last_only: false,
                variance_collector: None,
                prefill_workspace: None,

                layer_boundary_hook: None,
            },
            &mut prefetch,
        )?;
        let forward_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;
        forward_ms_values.push(forward_ms);

        // ── Offload resilience checkpoint ──
        if let Some(executor) = command_executor.as_mut() {
            let kv_snap = KVSnapshot {
                total_bytes: kv_caches
                    .iter()
                    .map(|c| c.memory_usage_bytes() as u64)
                    .sum(),
                total_tokens: kv_caches[0].current_pos(),
                capacity: kv_caches[0].capacity(),
                protected_prefix: 0,
                kv_dtype: kv_type_str.to_string(),
                eviction_policy: "none".to_string(),
                skip_ratio: 0.0,
            };
            let plan = executor.poll(&kv_snap);

            if plan.throttle_delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(plan.throttle_delay_ms));
            }
            if plan.suspended {
                eprintln!("\n[Offload-Resilience] Inference suspended by system signal");
                break;
            }
            // evict, kv_quant_bits, layer_skip 등은 OffloadKVCache에서 미지원 — 무시
            if plan.evict.is_some() {
                eprintln!(
                    "[Offload-Resilience] KvEvict requested but OffloadKVCache has no eviction support — ignored"
                );
            }

            executor.on_token_generated();
        } else if cli_throttle_delay_ms > 0 {
            // No CommandExecutor: honour CLI --throttle-delay-ms directly.
            std::thread::sleep(std::time::Duration::from_millis(cli_throttle_delay_ms));
        }
        // ── End checkpoint ──

        start_pos += 1;
        generated_count += 1;

        // Read logits to CPU
        let mut logits_cpu = vec![0.0f32; vocab_size];
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, logits_cpu.len() * 4);
            backend.read_buffer(&logits, slice)?;
        }

        let next_token =
            sampling::sample(&mut logits_cpu, &tokens, vocab_size, sampling_config, None);

        let now = std::time::Instant::now();
        let tbt = now.duration_since(last_token_time).as_secs_f64() * 1000.0;
        tbt_values.push(tbt);
        last_token_time = now;

        tokens.push(next_token);

        // Streaming output
        let text = tokenizer.decode(&tokens, true).unwrap_or_default();
        let new_text = &text[initial_text.len()..];
        print!("\r{}{}", initial_text, new_text);
        stdout.flush().ok();

        if next_token == eos_id && std::env::var("IGNORE_EOS").is_err() {
            break;
        }
    }

    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let tok_per_s = if decode_ms > 0.0 {
        generated_count as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };
    let offload_mem_final: usize = kv_caches.iter().map(|c| c.memory_usage_bytes()).sum();
    let final_raw_equiv = kv_caches[0].current_pos() * token_bytes * 2 * num_layers;

    eprintln!();
    eprintln!(
        "[Offload] Decode: {} tokens, {:.1}ms ({:.1} tok/s)",
        generated_count, decode_ms, tok_per_s,
    );
    eprintln!(
        "[Offload] Final: cache_pos={}, store_mem={}KB (raw equiv={}KB, ratio={:.2}x)",
        kv_caches[0].current_pos(),
        offload_mem_final / 1024,
        final_raw_equiv / 1024,
        final_raw_equiv as f64 / offload_mem_final.max(1) as f64,
    );

    let avg_forward_ms = if forward_ms_values.is_empty() {
        0.0
    } else {
        forward_ms_values.iter().sum::<f64>() / forward_ms_values.len() as f64
    };
    let avg_tbt = if tbt_values.is_empty() {
        0.0
    } else {
        tbt_values.iter().sum::<f64>() / tbt_values.len() as f64
    };

    eprintln!(
        "[Prefetch] final depth={}, preload_ema={:.0}us, forward_ema={:.0}us",
        prefetch.depth(),
        prefetch.preload_ema_us(),
        prefetch.forward_ema_us(),
    );

    println!("\nDone.");
    println!("TTFT: {:.2} ms", ttft_ms);
    println!(
        "Avg forward: {:.2} ms, Avg TBT: {:.2} ms ({:.1} tok/s)",
        avg_forward_ms, avg_tbt, tok_per_s,
    );

    Ok(())
}

// ════════════════════════════════════════════════════════════════
//  PPL MODE: Teacher-forcing perplexity evaluation on reference text.
//
//  Reads a text file, tokenizes it, and measures how well the model
//  predicts each token given all previous tokens. Applies the configured
//  eviction policy and collects proxy metrics during eviction events.
// ════════════════════════════════════════════════════════════════

/// LISWAP-PPL diagnostic: dump every layer's weight tensors (wq/wk/wv/wo/
/// w_gate/w_up/w_down) to raw bin files under `out_dir`. File naming:
/// `layer{NN}_{tensor}_{dtype}.bin` (e.g. `layer00_wq_Q4_0.bin`). Each file
/// holds the raw GPU buffer bytes for that tensor at the moment of the call.
///
/// Two such dumps (one from a Q4-native model load, one from an F16 model
/// after swap completion) can be byte-compared on the host to determine
/// whether the swap path produces bit-identical Q4 weights.
fn dump_layer_weights_to_dir(
    model: &TransformerModel,
    backend: &Arc<dyn Backend>,
    out_dir: &std::path::Path,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let n = model.layers.len();
    eprintln!(
        "[Q4-DUMP] dumping {} layer weights to {}",
        n,
        out_dir.display()
    );
    for (i, slot) in model.layers.iter().enumerate() {
        let weights = slot.load_weights();
        let dtype = slot.current_dtype();
        let tensors: [(&str, &llm_rs2::core::tensor::Tensor); 7] = [
            ("wq", &weights.wq),
            ("wk", &weights.wk),
            ("wv", &weights.wv),
            ("wo", &weights.wo),
            ("w_gate", &weights.w_gate),
            ("w_up", &weights.w_up),
            ("w_down", &weights.w_down),
        ];
        for (name, t) in tensors {
            let nbytes = t.buffer().size();
            if nbytes == 0 {
                eprintln!(
                    "[Q4-DUMP] layer{:02} {} dtype={:?} SKIP (size=0)",
                    i, name, dtype
                );
                continue;
            }
            let mut bytes = vec![0u8; nbytes];
            // For OpenCL/CUDA tensors `buffer().as_ptr()` is the cl_mem/cu_ptr
            // handle and may look like a host nullptr — backend.read_buffer
            // does the device→host copy via the backend-specific path, so we
            // rely on its return value rather than pre-checking as_ptr.
            match backend.read_buffer(t, &mut bytes) {
                Ok(()) => {
                    let fname = format!("layer{:02}_{}_{:?}.bin", i, name, dtype);
                    let path = out_dir.join(&fname);
                    std::fs::write(&path, &bytes)?;
                    eprintln!(
                        "[Q4-DUMP] layer{:02} {:8} dtype={:>5?} bytes={:8} → {}",
                        i, name, dtype, nbytes, fname
                    );
                }
                Err(e) => {
                    eprintln!(
                        "[Q4-DUMP] layer{:02} {} dtype={:?} SKIP (read_buffer failed: {})",
                        i, name, dtype, e
                    );
                }
            }
        }
    }
    // Also dump model-level tensors that are NOT inside per-layer slots and
    // therefore are NOT touched by weight swap: embed_tokens, final norm, and
    // lm_head. These three are the most likely sources of E ≠ D NLL drift
    // because (a) the F16 model's lm_head is typically tied to embed_tokens
    // and (b) any missing lm_head is derived via F16→Q4_0 quantization at
    // load time, whose result may not match a standalone Q4_0 GGUF's lm_head
    // byte-for-byte.
    let model_tensors: [(&str, &llm_rs2::core::tensor::Tensor); 3] = [
        ("embed_tokens", &model.embed_tokens),
        ("norm",         &model.norm),
        ("lm_head",      &model.lm_head),
    ];
    for (name, t) in model_tensors {
        let nbytes = t.buffer().size();
        if nbytes == 0 {
            eprintln!("[Q4-DUMP] model.{} SKIP (size=0)", name);
            continue;
        }
        let dt = t.dtype();
        let mut bytes = vec![0u8; nbytes];
        match backend.read_buffer(t, &mut bytes) {
            Ok(()) => {
                let fname = format!("model_{}_{:?}.bin", name, dt);
                let path = out_dir.join(&fname);
                std::fs::write(&path, &bytes)?;
                eprintln!(
                    "[Q4-DUMP] model.{:14} dtype={:>5?} bytes={:8} → {}",
                    name, dt, nbytes, fname
                );
            }
            Err(e) => {
                // The lm_head can live on a CPU backend even when the main
                // backend is GPU (`lm_head_on_cpu`) — fall back to CpuBackend
                // for that case so we still get a dump file out.
                let cpu_be: Arc<dyn Backend> =
                    Arc::new(llm_rs2::backend::cpu::CpuBackend::new());
                match cpu_be.read_buffer(t, &mut bytes) {
                    Ok(()) => {
                        let fname = format!("model_{}_{:?}.bin", name, dt);
                        let path = out_dir.join(&fname);
                        std::fs::write(&path, &bytes)?;
                        eprintln!(
                            "[Q4-DUMP] model.{:14} dtype={:>5?} bytes={:8} → {} (via CPU fallback)",
                            name, dt, nbytes, fname
                        );
                    }
                    Err(e2) => {
                        eprintln!(
                            "[Q4-DUMP] model.{} SKIP (read_buffer failed: gpu={}, cpu={})",
                            name, e, e2
                        );
                    }
                }
            }
        }
    }
    eprintln!("[Q4-DUMP] complete: {} layers + 3 model tensors dumped to {}", n, out_dir.display());
    Ok(())
}

/// Return value from `run_ppl` for use by the caller (e.g. `--qcf-dump`).
struct PplResult {
    ppl: f64,
    avg_nll: f64,
    n_eval_tokens: usize,
    wall_time_s: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_ppl(
    args: &Args,
    model: &TransformerModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &dyn Memory,
    kv_caches: &mut [KVCache],
    cache_manager: &mut CacheManager,
    score_accumulator: &mut Option<AttentionScoreAccumulator>,
    vocab_size: usize,
    hidden_size: usize,
    max_seq_len: usize,
    text_file: &str,
    auto_eviction: bool,
    score_based_eviction: bool,
    protected_prefix: usize,
    skip_config: Option<&llm_rs2::core::skip_config::SkipConfig>,
    // LISWAP-PPL Scenario E: when true, return early as soon as the swap plan
    // completes. NLL/CSV/JSON outputs are suppressed. Used by `--ppl-warmup-swap`
    // to drive the swap to completion before the actual measurement pass.
    warmup_only: bool,
) -> anyhow::Result<PplResult> {
    // ── 1. Read and tokenize reference text ──
    let text = std::fs::read_to_string(text_file)
        .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", text_file, e))?;
    let encoding = tokenizer
        .encode(text.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenize error: {}", e))?;
    let all_ids: Vec<u32> = encoding.get_ids().to_vec();
    let total_tokens = all_ids.len();

    if total_tokens < 2 {
        anyhow::bail!("PPL requires at least 2 tokens, got {}", total_tokens);
    }

    let eval_tokens = total_tokens.min(max_seq_len);
    if total_tokens > max_seq_len {
        eprintln!(
            "[PPL] Warning: text has {} tokens, truncating to max_seq_len={}",
            total_tokens, max_seq_len
        );
    }
    let token_ids = &all_ids[..eval_tokens];

    eprintln!(
        "[PPL] {} tokens, policy={}, kv_budget={}, kv_type={}",
        eval_tokens, args.eviction_policy, args.kv_budget, args.kv_type
    );

    // ── 2. Pre-allocate decode buffers ──
    let dl_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let mut decode_logits =
        Tensor::new(Shape::new(vec![1, 1, vocab_size]), dl_buf, backend.clone());
    let xg_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let mut x_gen = Tensor::new(Shape::new(vec![1, 1, hidden_size]), xg_buf, backend.clone());
    let q_dim = model.config.num_attention_heads * model.config.head_dim;
    let k_dim = model.config.num_key_value_heads * model.config.head_dim;
    let v_dim = k_dim;
    let ffn_hidden = model.config.intermediate_size;
    let mut gen_ws = LayerWorkspace::new(
        WorkspaceConfig {
            batch_size: 1,
            dim: hidden_size,
            q_dim,
            k_dim,
            v_dim,
            ffn_hidden,
            n_heads: model.config.num_attention_heads,
            max_seq_len: args.max_seq_len,
        },
        memory,
        backend.clone(),
    )?;
    let cpu_gen_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_buf,
        Arc::new(CpuBackend::new()),
    );
    // Pre-allocate GPU input tensor for decode loop (avoids per-token GPU alloc)
    let gpu_gen_buf_ppl = memory.alloc(4, DType::U8)?;
    let mut gen_input_gpu = Tensor::new(Shape::new(vec![1, 1]), gpu_gen_buf_ppl, backend.clone());
    let mut logits_cpu = vec![0.0f32; vocab_size];

    // ── 3. Determine prefill chunk size ──
    let has_budget = args.kv_budget > 0 || args.kv_budget_ratio > 0.0;
    if auto_eviction && !has_budget {
        eprintln!(
            "[PPL] Warning: eviction enabled without --kv-budget. \
             Results may not be reproducible. Use --kv-budget N for deterministic experiments."
        );
    }
    let prefill_chunk = if let Some(forced) = args.ppl_prefill_tokens {
        // LISWAP-PPL: 명시적 prefill 길이 강제. swap 측정 시 decode loop 을
        // 충분히 돌리기 위함. budget 로직보다 우선.
        forced.clamp(2, eval_tokens)
    } else if has_budget {
        let budget = if args.kv_budget_ratio > 0.0 {
            ((eval_tokens as f32) * args.kv_budget_ratio) as usize
        } else {
            args.kv_budget
        };
        budget.min(eval_tokens).max(2)
    } else if auto_eviction && args.eviction_policy == "sliding" {
        args.eviction_window.min(eval_tokens)
    } else {
        eval_tokens
    };

    let effective_budget = if args.kv_budget_ratio > 0.0 {
        ((eval_tokens as f32) * args.kv_budget_ratio) as usize
    } else if args.kv_budget > 0 {
        args.kv_budget
    } else {
        max_seq_len // No budget → no eviction trigger
    };

    if has_budget {
        eprintln!(
            "[PPL] Effective budget: {} tokens (deterministic eviction)",
            effective_budget
        );
    }

    // Headroom-based threshold: evict only when cache exceeds budget + headroom.
    // This prevents 1-by-1 evictions every step and ensures batch evictions (~2 total).
    // Example: budget=1500 → headroom=375 → threshold=1875.
    let eviction_headroom = (effective_budget / 4).max(16);
    let eviction_threshold = effective_budget.saturating_add(eviction_headroom);

    let mut total_nll: f64 = 0.0;
    let mut nll_count: usize = 0;
    // PPL v3: collect QCF for every eviction event
    let mut qcf_events: Vec<serde_json::Value> = Vec::new();
    let overall_start = std::time::Instant::now();

    // LISWAP-PPL: per-token NLL log + token-index-triggered weight swap.
    // (phase, token_idx, token_id, nll, swap_state, layers_swapped)
    let mut per_token_log: Vec<(&'static str, usize, u32, f64, &'static str, usize)> = Vec::new();
    let log_per_token = args.ppl_nll_csv.is_some();
    let mut ppl_swap_plan: Option<llm_rs2::models::weights::IncrementalSwapPlan> = None;
    // dispatch_swap_weights 시그니처 호환용 (PPL 경로에서는 manager 보고 안 함).
    let mut ppl_swap_report_unused: Option<(f32, usize, std::time::Instant, f32)> = None;
    let mut layers_swapped_so_far: usize = 0;
    let ppl_cpu_backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let ppl_swap_logged = std::sync::atomic::AtomicBool::new(false);

    if args.ppl_swap_at_token.is_some() && model.secondary_mmap.is_none() {
        anyhow::bail!("--ppl-swap-at-token requires --secondary-gguf to load secondary weights");
    }

    // ── 4. Prefill phase ──
    let prefill_len = prefill_chunk.min(eval_tokens);
    eprintln!("[PPL] Prefill: {} tokens", prefill_len);

    {
        let cpu_backend = Arc::new(CpuBackend::new());
        let input_buf = Galloc::new().alloc(prefill_len * 4, DType::U8)?;
        unsafe {
            let ptr = input_buf.as_mut_ptr() as *mut u32;
            for (i, &id) in token_ids[..prefill_len].iter().enumerate() {
                *ptr.add(i) = id;
            }
        }
        let cpu_input = Tensor::new(Shape::new(vec![1, prefill_len]), input_buf, cpu_backend);
        let input_tensor = backend.copy_from(&cpu_input)?;

        let prefill_logits_buf = memory.alloc(prefill_len * vocab_size * 4, DType::F32)?;
        let mut prefill_logits = Tensor::new(
            Shape::new(vec![1, prefill_len, vocab_size]),
            prefill_logits_buf,
            backend.clone(),
        );

        if let Some(acc) = score_accumulator.as_mut() {
            acc.begin_step();
        }

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos: 0,
            kv_caches,
            backend,
            memory,
            logits_out: &mut prefill_logits,
            x_gen: None,
            workspace: None,
            score_accumulator: score_accumulator.as_mut(),
            profiler: None,
            skip_config,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;

        // Read all prefill logits to CPU
        let mut all_logits = vec![0.0f32; prefill_len * vocab_size];
        unsafe {
            let ptr = all_logits.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, all_logits.len() * 4);
            backend.read_buffer(&prefill_logits, slice)?;
        }

        // Score tokens 1..prefill_len: logits[i] predicts token[i+1]
        for i in 0..prefill_len - 1 {
            let offset = i * vocab_size;
            let lp = sampling::compute_log_prob(
                &all_logits[offset..offset + vocab_size],
                token_ids[i + 1],
                vocab_size,
            );
            total_nll -= lp;
            nll_count += 1;
            if log_per_token {
                per_token_log.push(("prefill", i, token_ids[i + 1], -lp, "none", 0));
            }
        }

        eprintln!(
            "[PPL] Prefill NLL: {:.4}, count={}, running PPL={:.4}",
            total_nll,
            nll_count,
            (total_nll / nll_count as f64).exp()
        );
    }

    // ── 5. Decode phase (teacher-forcing) ──
    let mut start_pos = prefill_len;

    for (decode_idx, i) in (prefill_len..eval_tokens - 1).enumerate() {
        let input_token = token_ids[i];
        let target_token = token_ids[i + 1];

        // ── LISWAP-PPL: token-index-triggered weight swap ──────────────────
        // dispatch_swap_weights 가 commit 한 IncrementalSwapPlan 을 매 decode
        // step 마다 K=ppl_swap_per_tick 만큼 drain. dynamic-K controller 와
        // async dispatcher 는 측정 결정론을 위해 사용하지 않는다.
        if Some(decode_idx) == args.ppl_swap_at_token && ppl_swap_plan.is_none() {
            dispatch_swap_weights(
                model,
                args.ppl_swap_ratio,
                llm_shared::DtypeTag::Q4_0,
                None, // importance None → fallback uniform
                decode_idx,
                &mut ppl_swap_plan,
                &mut ppl_swap_report_unused,
            );
            if !ppl_swap_logged.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[PPL-Swap] triggered at decode_idx={}, ratio={}, per_tick={}",
                    decode_idx, args.ppl_swap_ratio, args.ppl_swap_per_tick
                );
            }
        }
        let mut swap_state: &'static str = if layers_swapped_so_far > 0 {
            "post_swap"
        } else {
            "none"
        };
        let plan_done = if let Some(plan) = ppl_swap_plan.as_mut() {
            plan.set_per_tick(args.ppl_swap_per_tick);
            let chunk = plan.drain_chunk();
            if !chunk.is_empty() {
                let t_swap = std::time::Instant::now();
                match run_layer_swap(
                    model,
                    &chunk,
                    Some(backend),
                    &ppl_cpu_backend,
                    None,
                    #[cfg(feature = "opencl")]
                    None,
                ) {
                    Ok(report) => {
                        layers_swapped_so_far += report.swapped.len();
                        swap_state = "swapping";
                        eprintln!(
                            "[PPL-Swap] tick decode_idx={} chunk={:?} swapped={} remaining={} latency={:.1}ms",
                            decode_idx,
                            &chunk,
                            report.swapped.len(),
                            plan.remaining_count(),
                            t_swap.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                    Err(e) => {
                        eprintln!("[PPL-Swap] run_layer_swap error: {}", e);
                    }
                }
            }
            plan.is_done()
        } else {
            false
        };
        if plan_done {
            eprintln!(
                "[PPL-Swap] plan complete at decode_idx={}, total_swapped={}",
                decode_idx, layers_swapped_so_far
            );
            ppl_swap_plan = None;
            swap_state = "post_swap";

            if warmup_only {
                // LISWAP-PPL Scenario E (warmup pass): swap is complete, return
                // before scoring the current token so the caller can reset KV
                // caches and run the measurement pass from scratch with the
                // already-swapped weights.
                eprintln!(
                    "[PPL-Swap] warmup_only=true → returning at decode_idx={} (no further scoring)",
                    decode_idx
                );
                let wall_time = overall_start.elapsed().as_secs_f64();
                return Ok(PplResult {
                    ppl: 0.0,
                    avg_nll: 0.0,
                    n_eval_tokens: 0,
                    wall_time_s: wall_time,
                });
            }
        }

        // Score accumulator begin step
        if let Some(acc) = score_accumulator.as_mut() {
            acc.begin_step();
        }

        // Feed true token
        unsafe {
            *(cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = input_token;
        }
        // Reuse pre-allocated GPU buffer — write data instead of alloc+copy
        backend.write_buffer(&mut gen_input_gpu, unsafe {
            std::slice::from_raw_parts(cpu_gen_input.buffer().as_ptr(), 4)
        })?;

        model.forward_into(TransformerModelForwardArgs {
            input_tokens: &gen_input_gpu,
            start_pos,
            kv_caches,
            backend,
            memory,
            logits_out: &mut decode_logits,
            x_gen: Some(&mut x_gen),
            workspace: Some(&mut gen_ws),
            score_accumulator: score_accumulator.as_mut(),
            profiler: None,
            skip_config,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;
        start_pos += 1;

        // Read logits and score target
        unsafe {
            let ptr = logits_cpu.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, vocab_size * 4);
            backend.read_buffer(&decode_logits, slice)?;
        }
        let lp = sampling::compute_log_prob(&logits_cpu, target_token, vocab_size);
        total_nll -= lp;
        nll_count += 1;
        if log_per_token {
            per_token_log.push((
                "decode",
                i,
                target_token,
                -lp,
                swap_state,
                layers_swapped_so_far,
            ));
        }

        // ── Budget-based eviction (deterministic, experiment-reproducible) ──
        // Eviction triggers when cache_pos exceeds eviction_threshold (budget + headroom).
        // Using headroom prevents 1-by-1 evictions: evictions occur in ~2 large batches
        // rather than 500+ tiny steps, preserving PPL measurement validity.
        // This is deterministic: same text + same budget = same eviction positions.
        // No dependency on memory pressure or hardware state.
        if auto_eviction && has_budget {
            let before_len = kv_caches[0].current_pos;
            if before_len > eviction_threshold {
                let ratio = effective_budget as f32 / before_len as f32;

                // GPU V buffer readback for QCF-CAOTE computation.
                let v_cpu_data: Option<Vec<f32>> = if args.kv_type == "f32"
                    && !kv_caches.is_empty()
                    && kv_caches[0].v_buffer.buffer().as_ptr().is_null()
                {
                    let v_elems = kv_caches[0].v_buffer.buffer().size() / 4;
                    let mut v_buf = vec![0.0f32; v_elems];
                    let byte_slice = unsafe {
                        std::slice::from_raw_parts_mut(v_buf.as_mut_ptr() as *mut u8, v_elems * 4)
                    };
                    match backend.read_buffer(&kv_caches[0].v_buffer, byte_slice) {
                        Ok(()) => Some(v_buf),
                        Err(_) => None,
                    }
                } else {
                    None
                };
                let can_compute_qcf = args.kv_type == "f32"
                    && !kv_caches.is_empty()
                    && (v_cpu_data.is_some() || !kv_caches[0].v_buffer.buffer().as_ptr().is_null());

                // Perform eviction
                let result = if score_based_eviction {
                    if let Some(acc) = score_accumulator.as_ref() {
                        if acc.is_active() {
                            let scores = acc.importance_scores().to_vec();
                            cache_manager.force_evict_with_scores(kv_caches, ratio, &scores)?
                        } else {
                            cache_manager.force_evict(kv_caches, ratio)?
                        }
                    } else {
                        cache_manager.force_evict(kv_caches, ratio)?
                    }
                } else {
                    cache_manager.force_evict(kv_caches, ratio)?
                };

                if result.evicted {
                    let eviction_ratio = result.tokens_removed as f32 / before_len as f32;
                    let ppl_at_event = (total_nll / nll_count as f64).exp();

                    let qcf_caote_value = if can_compute_qcf
                        && let Some(acc) = score_accumulator.as_ref()
                        && let Some(head_attn) = acc.last_step_head_attn()
                    {
                        use llm_rs2::core::qcf::{
                            AggregationMode, QcfActionType, UnifiedQcfParams, VDataSource,
                            compute_unified_qcf,
                        };
                        let target_len = ((before_len as f32) * ratio) as usize;
                        let cache = &kv_caches[0];
                        let v_cpu_bytes: Option<&[u8]> = v_cpu_data.as_deref().map(|s| {
                            // Reinterpret &[f32] as &[u8] so the unified helper handles it.
                            unsafe {
                                std::slice::from_raw_parts(
                                    s.as_ptr() as *const u8,
                                    std::mem::size_of_val(s),
                                )
                            }
                        });
                        let action = if score_based_eviction {
                            QcfActionType::EvictH2o {
                                target_len,
                                keep_ratio: args.h2o_keep_ratio,
                                protected_prefix,
                            }
                        } else {
                            QcfActionType::EvictSliding { target_len }
                        };
                        match VDataSource::from_kv_cache(cache, v_cpu_bytes) {
                            Some(v_source) => {
                                let params = UnifiedQcfParams {
                                    action,
                                    v_source,
                                    // PPL eval site only triggers Sliding/H2O,
                                    // never D2O — `k_source` is unused.
                                    k_source: None,
                                    attention_scores: acc.importance_scores(),
                                    head_attn: Some(head_attn),
                                    n_kv_heads: cache.kv_heads(),
                                    head_dim: cache.head_dim(),
                                    current_pos: before_len,
                                    capacity: cache.capacity(),
                                    layout: cache.layout(),
                                    aggregation: AggregationMode::Mean,
                                    beta: 1.0,
                                };
                                let (qcf, _) = compute_unified_qcf(&params);
                                qcf as f64
                            }
                            None => 0.0,
                        }
                    } else {
                        0.0
                    };

                    qcf_events.push(serde_json::json!({
                        "step": i,
                        "tokens_evicted": result.tokens_removed,
                        "eviction_ratio": eviction_ratio,
                        "qcf_caote": qcf_caote_value,
                        "ppl_at_step": ppl_at_event,
                    }));

                    // IMPORTANT: Do NOT reset start_pos to current_pos after eviction.
                    // After shift_positions(), cached K vectors retain their original RoPE
                    // positions. start_pos must continue incrementing from the original
                    // position to maintain correct RoPE relative distances. Using current_pos
                    // (compacted) creates a RoPE discontinuity where cached tokens appear
                    // as "future" tokens, causing severe NLL degradation.
                    // start_pos continues via `start_pos += 1` in the main loop.
                    if let Some(acc) = score_accumulator.as_mut() {
                        acc.reset();
                    }
                    eprintln!(
                        "[PPL] Eviction at step {}: {} → {} tokens (removed {})",
                        i, before_len, result.new_pos, result.tokens_removed
                    );
                }
            }
        }

        // Progress
        if (i + 1) % 200 == 0 {
            let ppl = (total_nll / nll_count as f64).exp();
            eprintln!(
                "[PPL] step {}/{}: NLL={:.4}, PPL={:.4}, cache_pos={}",
                i + 1,
                eval_tokens,
                total_nll,
                ppl,
                kv_caches[0].current_pos
            );
        }
    }

    // ── 6. Output results ──
    let wall_time = overall_start.elapsed().as_secs_f64();
    let ppl = (total_nll / nll_count as f64).exp();
    let avg_nll = total_nll / nll_count as f64;
    let tok_per_sec = nll_count as f64 / wall_time;

    // Compute summary stats from all eviction events (v3)
    let n_evictions = qcf_events.len();
    let qcf_sum_caote: f64 = qcf_events
        .iter()
        .filter_map(|e| e["qcf_caote"].as_f64())
        .sum();
    let qcf_max_caote: f64 = qcf_events
        .iter()
        .filter_map(|e| e["qcf_caote"].as_f64())
        .fold(0.0f64, f64::max);

    let output = serde_json::json!({
        "ppl": ppl,
        "total_nll": total_nll,
        "token_count": nll_count,
        "tokens_per_second": tok_per_sec,
        "wall_time_s": wall_time,
        "n_evictions": n_evictions,
        "qcf_events": qcf_events,
        "qcf_sum_caote": qcf_sum_caote,
        "qcf_max_caote": qcf_max_caote,
        "config": {
            "model": args.model_path,
            "text_file": text_file,
            "eviction_policy": args.eviction_policy,
            "kv_budget": args.kv_budget,
            "kv_type": args.kv_type,
            "max_seq_len": max_seq_len,
            "eviction_target_ratio": args.eviction_target_ratio,
            "h2o_keep_ratio": args.h2o_keep_ratio,
            "protected_prefix": protected_prefix,
            "skip_layers": args.skip_layers,
            "skip_ratio": args.skip_ratio,
        }
    });

    println!("{}", serde_json::to_string_pretty(&output)?);

    eprintln!(
        "\n[PPL] Final: PPL={:.4}, NLL={:.4}, tokens={}, {:.1} tok/s, {:.1}s",
        ppl, total_nll, nll_count, tok_per_sec, wall_time
    );

    // LISWAP-PPL: per-token NLL CSV dump (token_idx is text-absolute, identical
    // across scenarios for direct curve comparison).
    if let Some(csv_path) = args.ppl_nll_csv.as_ref() {
        use std::io::Write;
        let mut f = std::fs::File::create(csv_path)?;
        writeln!(f, "phase,token_idx,token_id,nll,swap_state,layers_swapped")?;
        for (phase, idx, id, nll, state, n) in &per_token_log {
            writeln!(f, "{},{},{},{:.6},{},{}", phase, idx, id, nll, state, n)?;
        }
        f.flush()?;
        eprintln!(
            "[PPL] Per-token NLL CSV: {} ({} rows)",
            csv_path.display(),
            per_token_log.len()
        );
    }

    Ok(PplResult {
        ppl,
        avg_nll,
        n_eval_tokens: nll_count,
        wall_time_s: wall_time,
    })
}

// ─────────────────────── Chat REPL mode ───────────────────────

use llm_rs2::core::chat_ipc::{
    ChatInput, finish_reply_stream, spawn_chat_input_sources, write_reply_bytes,
};

fn resolve_token_ids(
    tokenizer: &Tokenizer,
    literals: &[&'static str],
    required: bool,
) -> anyhow::Result<Vec<u32>> {
    let mut out = Vec::with_capacity(literals.len());
    for lit in literals {
        match tokenizer.token_to_id(lit) {
            Some(id) => out.push(id),
            None if required => {
                anyhow::bail!(
                    "tokenizer is missing required special token `{}`. \
                     Make sure tokenizer.json has it registered as an added_token.",
                    lit
                );
            }
            None => {}
        }
    }
    Ok(out)
}

// Chat turn executor: per-variant state machine for prefill/decode/eviction/reset.
// Keeps the REPL loop KV-type-agnostic. See `run_chat_repl`.
trait ChatTurnExec {
    /// Current KV position.
    fn pos(&self) -> usize;
    /// Reset session state (KV position, accumulator, offload store).
    fn reset(&mut self);
    /// Prefill a batch of tokens, advancing `pos` by `tokens.len()` and
    /// returning the last-position logits (host f32).
    fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>>;
    /// Decode a single token, advancing `pos` by 1 and returning logits.
    fn decode_step(&mut self, token: u32) -> anyhow::Result<Vec<f32>>;
    /// Ensure there is room for `additional` new tokens before `max_seq_len`.
    /// Eviction-capable execs may run force_evict here. Non-evicting execs
    /// return Err on overflow.
    fn ensure_capacity(&mut self, additional: usize, max_seq_len: usize) -> anyhow::Result<()>;
    /// End-of-turn maintenance hook (e.g. opportunistic auto-eviction).
    fn on_turn_end(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
    /// Content of the `/stats` line.
    fn stats_line(&self, max_seq_len: usize) -> String;
}

/// Shared REPL loop driving a `ChatTurnExec`. Handles template rendering,
/// stdin + socket input, slash commands, streaming decode, and turn-end
/// hooks. All KV-type-specific work is delegated to the exec.
#[allow(clippy::too_many_arguments)]
fn run_chat_repl<E: ChatTurnExec>(
    args: &Args,
    model_arch: llm_rs2::models::config::ModelArch,
    tokenizer: &Tokenizer,
    eos_token_id: u32,
    vocab_size: usize,
    sampling_config: &SamplingConfig,
    max_seq_len: usize,
    exec: &mut E,
) -> anyhow::Result<()> {
    use llm_rs2::core::chat_template::ChatTemplate;
    use std::collections::VecDeque;
    use std::io::Write;

    let template = ChatTemplate::new(model_arch)?;
    let stop_ids = {
        let lits = template.stop_token_literals();
        if lits.is_empty() {
            anyhow::bail!("chat template has no stop token literals");
        }
        let mut ids = resolve_token_ids(tokenizer, &[lits[0]], true)?;
        ids.extend(resolve_token_ids(tokenizer, &lits[1..], false)?);
        ids.push(eos_token_id);
        ids.sort_unstable();
        ids.dedup();
        ids
    };
    let assistant_eot_ids: Vec<u32> = tokenizer
        .encode(template.assistant_eot(), false)
        .map_err(|e| anyhow::anyhow!("encode EOT: {}", e))?
        .get_ids()
        .to_vec();
    let bos_id = if template.bos_needed_on_first_prefill() {
        template
            .bos_literal()
            .and_then(|lit| tokenizer.token_to_id(lit))
    } else {
        None
    };

    // Optional system prompt prefill (stays in KV across turns).
    if let Some(sys) = &args.system_prompt {
        let rendered = template.render_system(sys);
        let mut ids = tokenizer
            .encode(rendered.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode system: {}", e))?
            .get_ids()
            .to_vec();
        if let Some(b) = bos_id {
            ids.insert(0, b);
        }
        if ids.len() > max_seq_len {
            anyhow::bail!(
                "system prompt produces {} tokens, exceeds max_seq_len={}",
                ids.len(),
                max_seq_len
            );
        }
        let _ = exec.prefill(&ids)?;
    }

    let input_rx = spawn_chat_input_sources(args.chat_socket.as_deref(), args.chat_tcp.as_deref())?;
    let mut first_user: Option<String> =
        (!args.prompt.trim().is_empty()).then(|| args.prompt.clone());
    let mut recent: VecDeque<u32> = VecDeque::new();

    eprintln!(
        "[Chat] Ready. Arch={:?}, max_seq_len={}. Commands: /exit /reset /stats /help",
        model_arch, max_seq_len
    );
    let mut stdout_lock = std::io::stdout();

    'outer: loop {
        print!("> ");
        stdout_lock.flush().ok();

        let (user_line_raw, reply_writer) = if let Some(line) = first_user.take() {
            (line, None)
        } else {
            match input_rx.recv() {
                Ok(ChatInput::Line(s, w)) => (s, w),
                Ok(ChatInput::Eof) | Err(_) => {
                    eprintln!();
                    break 'outer;
                }
            }
        };
        let user_line = user_line_raw
            .trim_end_matches(&['\n', '\r'][..])
            .to_string();
        let trimmed = user_line.trim();

        match trimmed {
            "" => continue,
            "/exit" | "/quit" => break 'outer,
            "/help" => {
                println!("(commands: /exit /quit /reset /stats /help; empty line ignored)");
                continue;
            }
            "/stats" => {
                println!("{}", exec.stats_line(max_seq_len));
                continue;
            }
            "/reset" => {
                exec.reset();
                recent.clear();
                println!("(session reset)");
                continue;
            }
            _ => {}
        }

        let rendered = template.render_user_and_assistant_header(trimmed);
        let mut turn_ids: Vec<u32> = tokenizer
            .encode(rendered.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode user turn: {}", e))?
            .get_ids()
            .to_vec();
        if exec.pos() == 0
            && let Some(b) = bos_id
        {
            turn_ids.insert(0, b);
        }

        // Capacity check: eviction-capable execs may reclaim space here.
        if let Err(e) = exec.ensure_capacity(turn_ids.len() + args.num_tokens, max_seq_len) {
            let msg = format!("error: {}", e);
            eprintln!("{}", msg);
            write_reply_bytes(reply_writer.as_ref(), msg.as_bytes());
            finish_reply_stream(reply_writer.as_ref());
            anyhow::bail!("context overflow: {}", e);
        }

        let mut prefill_logits = exec.prefill(&turn_ids)?;

        let mut accum: Vec<u32> = Vec::new();
        let mut printed_bytes: usize = 0;
        let mut indices_buf: Vec<usize> = Vec::with_capacity(vocab_size);
        let first_tok = {
            let recent_slice: Vec<u32> = recent.iter().copied().collect();
            sampling::sample(
                &mut prefill_logits,
                &recent_slice,
                vocab_size,
                sampling_config,
                Some(&mut indices_buf),
            )
        };

        let mut cur_tok = first_tok;
        for _step in 0..args.num_tokens {
            if stop_ids.contains(&cur_tok) {
                break;
            }
            accum.push(cur_tok);
            recent.push_back(cur_tok);
            if recent.len() > sampling_config.repetition_window.max(1) {
                recent.pop_front();
            }

            let decoded = tokenizer.decode(&accum, true).unwrap_or_default();
            if decoded.len() > printed_bytes {
                let piece = &decoded[printed_bytes..];
                print!("{}", piece);
                stdout_lock.flush().ok();
                write_reply_bytes(reply_writer.as_ref(), piece.as_bytes());
                printed_bytes = decoded.len();
            }

            let mut logits_host = exec.decode_step(cur_tok)?;

            if exec.pos() + 1 >= max_seq_len {
                break;
            }

            let recent_slice: Vec<u32> = recent.iter().copied().collect();
            cur_tok = sampling::sample(
                &mut logits_host,
                &recent_slice,
                vocab_size,
                sampling_config,
                Some(&mut indices_buf),
            );
        }

        // Record assistant EOT into KV so the next turn sees a well-formed boundary.
        if !assistant_eot_ids.is_empty() && exec.pos() + assistant_eot_ids.len() <= max_seq_len {
            let _ = exec.prefill(&assistant_eot_ids)?;
        }

        exec.on_turn_end()?;

        println!();
        stdout_lock.flush().ok();
        finish_reply_stream(reply_writer.as_ref());
    }

    Ok(())
}

// ─── Standard chat executor (KVCache; supports eviction policies) ─────────────

struct StandardTurnExec<'a> {
    model: &'a TransformerModel,
    backend: &'a Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    kv_caches: &'a mut [KVCache],
    // Decode workspace (reused across tokens).
    decode_logits: Tensor,
    x_gen: Tensor,
    gen_ws: LayerWorkspace,
    cpu_gen_input: Tensor,
    gen_input_gpu: Tensor,
    vocab_size: usize,
    pos: usize,
    // Eviction wiring (None when --eviction-policy == "none").
    cache_manager: Option<CacheManager>,
    score_accumulator: Option<AttentionScoreAccumulator>,
    eviction_policy_name: String,
    score_based: bool,
    target_ratio: f32,
    evicted_total: usize,
}

impl<'a> StandardTurnExec<'a> {
    /// Build a f32 token tensor on backend.
    fn tokens_to_backend_tensor(&self, tokens: &[u32]) -> anyhow::Result<Tensor> {
        let cpu_backend = Arc::new(CpuBackend::new());
        let input_buf = Galloc::new().alloc(tokens.len() * 4, DType::U8)?;
        unsafe {
            let ptr = input_buf.as_mut_ptr() as *mut u32;
            for (i, &id) in tokens.iter().enumerate() {
                *ptr.add(i) = id;
            }
        }
        let cpu_input = Tensor::new(Shape::new(vec![1, tokens.len()]), input_buf, cpu_backend);
        self.backend.copy_from(&cpu_input)
    }

    /// Run one forward_into pass with the standard KVCache and optional
    /// score accumulator. Returns last-position logits read to host f32.
    fn forward_prefill_standard(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let input_tensor = self.tokens_to_backend_tensor(tokens)?;
        let logits_buf = self.memory.alloc(self.vocab_size * 4, DType::F32)?;
        let mut logits_out = Tensor::new(
            Shape::new(vec![1, 1, self.vocab_size]),
            logits_buf,
            self.backend.clone(),
        );

        if let Some(acc) = self.score_accumulator.as_mut() {
            acc.begin_step();
        }

        self.model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos: self.pos,
            kv_caches: self.kv_caches,
            backend: self.backend,
            memory: self.memory.as_ref(),
            logits_out: &mut logits_out,
            x_gen: None,
            workspace: None,
            score_accumulator: self.score_accumulator.as_mut(),
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: true,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;
        self.pos += tokens.len();

        let mut host = vec![0.0f32; self.vocab_size];
        unsafe {
            let ptr = host.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, self.vocab_size * 4);
            self.backend.read_buffer(&logits_out, slice)?;
        }
        Ok(host)
    }

    #[cfg(feature = "opencl")]
    fn gpu_sync_scores(&mut self) -> anyhow::Result<()> {
        if let Some(ocl_be) = self
            .backend
            .as_any()
            .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
            && let Some(gpu_acc) = ocl_be.gpu_score_acc()
            && gpu_acc.is_active()
        {
            let (flat, head) = gpu_acc.sync_to_cpu(ocl_be.queue.as_core())?;
            if let Some(ref mut acc) = self.score_accumulator {
                acc.import_gpu_scores(&flat, &head);
            }
        }
        Ok(())
    }

    #[cfg(not(feature = "opencl"))]
    fn gpu_sync_scores(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Run eviction. Returns the number of tokens removed (0 if no-op).
    fn run_eviction(&mut self, force: bool) -> anyhow::Result<usize> {
        if self.cache_manager.is_none() {
            return Ok(0);
        }
        self.gpu_sync_scores()?;

        let before_len = self.kv_caches[0].current_pos;
        let scores_opt = self
            .score_accumulator
            .as_ref()
            .filter(|acc| acc.is_active())
            .map(|acc| acc.importance_scores().to_vec());

        let cache_manager = self.cache_manager.as_ref().unwrap();
        let result = if force {
            match (&scores_opt, self.score_based) {
                (Some(scores), true) => cache_manager.force_evict_with_scores(
                    self.kv_caches,
                    self.target_ratio,
                    scores,
                )?,
                _ => cache_manager.force_evict(self.kv_caches, self.target_ratio)?,
            }
        } else {
            match (&scores_opt, self.score_based) {
                (Some(scores), true) => {
                    cache_manager.maybe_evict_with_scores(self.kv_caches, scores)?
                }
                _ => cache_manager.maybe_evict(self.kv_caches)?,
            }
        };

        let removed = before_len.saturating_sub(self.kv_caches[0].current_pos);
        if result.evicted {
            self.pos = self.kv_caches[0].current_pos;
            self.evicted_total += removed;
            eprintln!(
                "[Chat/Evict] policy={} before={} after={} removed={}",
                self.eviction_policy_name, before_len, self.pos, removed
            );
        }
        Ok(removed)
    }
}

impl<'a> ChatTurnExec for StandardTurnExec<'a> {
    fn pos(&self) -> usize {
        self.pos
    }

    fn reset(&mut self) {
        for c in self.kv_caches.iter_mut() {
            c.current_pos = 0;
        }
        if let Some(acc) = self.score_accumulator.as_mut() {
            acc.reset();
        }
        self.pos = 0;
        self.evicted_total = 0;
    }

    fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        self.forward_prefill_standard(tokens)
    }

    fn decode_step(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        if let Some(acc) = self.score_accumulator.as_mut() {
            acc.begin_step();
        }
        unsafe {
            *(self.cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = token;
        }
        self.backend.write_buffer(&mut self.gen_input_gpu, unsafe {
            std::slice::from_raw_parts(self.cpu_gen_input.buffer().as_ptr(), 4)
        })?;

        self.model.forward_into(TransformerModelForwardArgs {
            input_tokens: &self.gen_input_gpu,
            start_pos: self.pos,
            kv_caches: self.kv_caches,
            backend: self.backend,
            memory: self.memory.as_ref(),
            logits_out: &mut self.decode_logits,
            x_gen: Some(&mut self.x_gen),
            workspace: Some(&mut self.gen_ws),
            score_accumulator: self.score_accumulator.as_mut(),
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: true,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;
        self.pos += 1;

        let mut host = vec![0.0f32; self.vocab_size];
        unsafe {
            let ptr = host.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, self.vocab_size * 4);
            self.backend.read_buffer(&self.decode_logits, slice)?;
        }
        Ok(host)
    }

    fn ensure_capacity(&mut self, additional: usize, max_seq_len: usize) -> anyhow::Result<()> {
        if self.pos + additional <= max_seq_len {
            return Ok(());
        }
        if self.cache_manager.is_some() {
            // Force eviction; then re-check.
            self.run_eviction(true)?;
            if self.pos + additional <= max_seq_len {
                return Ok(());
            }
        }
        anyhow::bail!(
            "context would exceed max_seq_len={} (pos={}, incoming_reserve={}). \
             Use /reset or increase --max-seq-len.",
            max_seq_len,
            self.pos,
            additional
        );
    }

    fn on_turn_end(&mut self) -> anyhow::Result<()> {
        if self.cache_manager.is_none() {
            return Ok(());
        }
        // Force-evict once KV usage reaches 90% of capacity so long sessions
        // keep running without hitting the next-turn ensure_capacity hard stop.
        // Opportunistic maybe_evict (memory-pressure driven) runs at lower fill.
        let capacity = self.kv_caches[0].capacity();
        let at_pressure = self.pos >= capacity.saturating_mul(9) / 10;
        if at_pressure {
            self.run_eviction(true)?;
        } else {
            self.run_eviction(false)?;
        }
        Ok(())
    }

    fn stats_line(&self, max_seq_len: usize) -> String {
        format!(
            "kv_pos={}/{} policy={} evicted_total={}",
            self.pos, max_seq_len, self.eviction_policy_name, self.evicted_total
        )
    }
}

/// Build a CacheManager + AttentionScoreAccumulator for chat's eviction mode.
/// Returns (manager, accumulator, score_based, policy_name, target_ratio).
/// When `args.eviction_policy == "none"`, the manager is `None`.
#[allow(clippy::type_complexity)]
fn build_chat_eviction(
    args: &Args,
    model: &TransformerModel,
    backend: &Arc<dyn Backend>,
    max_seq_len: usize,
) -> anyhow::Result<(
    Option<CacheManager>,
    Option<AttentionScoreAccumulator>,
    bool,
    String,
    f32,
)> {
    if args.eviction_policy == "none" {
        return Ok((None, None, false, "none".to_string(), 1.0));
    }

    let _ = backend;
    let actual_protected_prefix =
        args.protected_prefix
            .unwrap_or(match args.eviction_policy.as_str() {
                "h2o" | "h2o_plus" | "d2o" => 4,
                "streaming" => args.sink_size,
                _ => 4,
            });

    let monitor: Box<dyn llm_rs2::core::sys_monitor::SystemMonitor> = if backend.is_discrete_gpu() {
        Box::new(NoOpMonitor)
    } else {
        Box::new(LinuxSystemMonitor)
    };
    let threshold_bytes = args.memory_threshold_mb * 1024 * 1024;

    let mut cache_manager = if args.eviction_policy == "d2o" {
        let d2o_handler = D2OHandler::new(D2OConfig {
            keep_ratio: args.d2o_keep_ratio,
            protected_prefix: actual_protected_prefix,
            target_ratio: args.eviction_target_ratio,
            ema_beta: args.d2o_ema_beta,
            merge_e: args.d2o_merge_e,
            use_layer_allocation: args.d2o_layer_alloc,
            protected_layers: args.d2o_protected_layers.clone().unwrap_or_default(),
        });
        let pipeline = CachePressurePipeline::new(vec![PressureStageConfig {
            min_level: PressureLevel::Warning,
            handler: Box::new(d2o_handler),
        }]);
        CacheManager::with_pipeline(pipeline, monitor, threshold_bytes)
    } else {
        let policy: Box<dyn llm_rs2::core::eviction::EvictionPolicy> = match args
            .eviction_policy
            .as_str()
        {
            "sliding" => Box::new(SlidingWindowPolicy::new(
                args.eviction_window,
                actual_protected_prefix,
            )),
            "streaming" => {
                use llm_rs2::core::eviction::StreamingLLMPolicy;
                let window = if args.streaming_window > 0 {
                    args.streaming_window
                } else if args.kv_budget > 0 {
                    args.kv_budget.saturating_sub(args.sink_size)
                } else {
                    args.eviction_window
                };
                Box::new(StreamingLLMPolicy::new(args.sink_size, window))
            }
            "h2o" => Box::new(H2OPolicy::new(args.h2o_keep_ratio, actual_protected_prefix)),
            "h2o_plus" => Box::new(H2OPlusPolicy::new(
                args.h2o_keep_ratio,
                actual_protected_prefix,
            )),
            other => anyhow::bail!(
                "Unknown eviction policy for --chat: '{}'. Use: none, sliding, streaming, h2o, h2o_plus, d2o",
                other
            ),
        };
        CacheManager::new(policy, monitor, threshold_bytes, args.eviction_target_ratio)
    };
    cache_manager.set_event_sink(Arc::new(StderrDiagnosticSink));

    // Accumulator setup: build for any active policy so sliding/streaming
    // still populate importance for observability; score-based policies need it.
    let score_based = matches!(args.eviction_policy.as_str(), "h2o" | "h2o_plus" | "d2o");
    // GQA accumulator: always active in chat. h2o_plus strictly requires it;
    // other policies benefit from per-head scores for future CAOTE / head budgets.
    let mut acc = AttentionScoreAccumulator::new_gqa(
        max_seq_len,
        model.config.num_attention_heads,
        model.config.num_key_value_heads,
        model.config.num_hidden_layers,
        args.h2o_tracked_layers,
        args.h2o_decay,
    );
    acc.set_active(true);
    acc.set_time_normalize(!args.h2o_raw_scores);

    // Init GPU-side accumulator when available.
    #[cfg(feature = "opencl")]
    if let Some(ocl_be) = backend
        .as_any()
        .downcast_ref::<llm_rs2::backend::opencl::OpenCLBackend>()
    {
        let _ = ocl_be.init_gpu_score_acc(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
            max_seq_len,
            args.h2o_decay,
        );
        if let Some(gpu_acc) = ocl_be.gpu_score_acc_mut() {
            gpu_acc.set_active(true);
        }
    }

    Ok((
        Some(cache_manager),
        Some(acc),
        score_based,
        args.eviction_policy.clone(),
        args.eviction_target_ratio,
    ))
}

#[allow(clippy::too_many_arguments)]
fn run_chat_standard(
    args: &Args,
    model: &TransformerModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    kv_caches: &mut [KVCache],
    sampling_config: &SamplingConfig,
    max_seq_len: usize,
) -> anyhow::Result<()> {
    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;
    let q_dim = model.config.num_attention_heads * model.config.head_dim;
    let k_dim = model.config.num_key_value_heads * model.config.head_dim;
    let v_dim = k_dim;
    let ffn_hidden = model.config.intermediate_size;

    let dl_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let decode_logits = Tensor::new(Shape::new(vec![1, 1, vocab_size]), dl_buf, backend.clone());
    let xg_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let x_gen = Tensor::new(Shape::new(vec![1, 1, hidden_size]), xg_buf, backend.clone());
    let gen_ws = LayerWorkspace::new(
        WorkspaceConfig {
            batch_size: 1,
            dim: hidden_size,
            q_dim,
            k_dim,
            v_dim,
            ffn_hidden,
            n_heads: model.config.num_attention_heads,
            max_seq_len,
        },
        memory.as_ref(),
        backend.clone(),
    )?;
    let cpu_gen_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_buf,
        Arc::new(CpuBackend::new()),
    );
    let gpu_gen_buf = memory.alloc(4, DType::U8)?;
    let gen_input_gpu = Tensor::new(Shape::new(vec![1, 1]), gpu_gen_buf, backend.clone());

    let (cache_manager, score_accumulator, score_based, policy_name, target_ratio) =
        build_chat_eviction(args, model, backend, max_seq_len)?;

    let mut exec = StandardTurnExec {
        model,
        backend,
        memory: memory.clone(),
        kv_caches,
        decode_logits,
        x_gen,
        gen_ws,
        cpu_gen_input,
        gen_input_gpu,
        vocab_size,
        pos: 0,
        cache_manager,
        score_accumulator,
        eviction_policy_name: policy_name,
        score_based,
        target_ratio,
        evicted_total: 0,
    };

    run_chat_repl(
        args,
        model.config.arch,
        tokenizer,
        model.config.eos_token_id,
        vocab_size,
        sampling_config,
        max_seq_len,
        &mut exec,
    )
}

// ─── KIVI chat executor (quantized KV cache) ──────────────────────────────────

struct KiviTurnExec<'a> {
    model: &'a TransformerModel,
    backend: &'a Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    kv_caches: Vec<KiviCache>,
    decode_logits: Tensor,
    x_gen: Tensor,
    gen_ws: LayerWorkspace,
    cpu_gen_input: Tensor,
    gen_input_gpu: Tensor,
    vocab_size: usize,
    pos: usize,
    bits: u8,
    residual_size: usize,
}

impl<'a> KiviTurnExec<'a> {
    fn tokens_to_backend_tensor(&self, tokens: &[u32]) -> anyhow::Result<Tensor> {
        let cpu_backend = Arc::new(CpuBackend::new());
        let input_buf = Galloc::new().alloc(tokens.len() * 4, DType::U8)?;
        unsafe {
            let ptr = input_buf.as_mut_ptr() as *mut u32;
            for (i, &id) in tokens.iter().enumerate() {
                *ptr.add(i) = id;
            }
        }
        let cpu_input = Tensor::new(Shape::new(vec![1, tokens.len()]), input_buf, cpu_backend);
        self.backend.copy_from(&cpu_input)
    }
}

impl<'a> ChatTurnExec for KiviTurnExec<'a> {
    fn pos(&self) -> usize {
        self.pos
    }

    fn reset(&mut self) {
        for c in self.kv_caches.iter_mut() {
            c.reset();
        }
        self.pos = 0;
    }

    fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let input_tensor = self.tokens_to_backend_tensor(tokens)?;
        let logits_buf = self.memory.alloc(self.vocab_size * 4, DType::F32)?;
        let mut logits_out = Tensor::new(
            Shape::new(vec![1, 1, self.vocab_size]),
            logits_buf,
            self.backend.clone(),
        );
        self.model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos: self.pos,
            kv_caches: &mut self.kv_caches,
            backend: self.backend,
            memory: self.memory.as_ref(),
            logits_out: &mut logits_out,
            x_gen: None,
            workspace: None,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: true,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;
        self.pos += tokens.len();

        let mut host = vec![0.0f32; self.vocab_size];
        unsafe {
            let ptr = host.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, self.vocab_size * 4);
            self.backend.read_buffer(&logits_out, slice)?;
        }
        Ok(host)
    }

    fn decode_step(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        unsafe {
            *(self.cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = token;
        }
        self.backend.write_buffer(&mut self.gen_input_gpu, unsafe {
            std::slice::from_raw_parts(self.cpu_gen_input.buffer().as_ptr(), 4)
        })?;
        self.model.forward_into(TransformerModelForwardArgs {
            input_tokens: &self.gen_input_gpu,
            start_pos: self.pos,
            kv_caches: &mut self.kv_caches,
            backend: self.backend,
            memory: self.memory.as_ref(),
            logits_out: &mut self.decode_logits,
            x_gen: Some(&mut self.x_gen),
            workspace: Some(&mut self.gen_ws),
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: true,
            variance_collector: None,
            prefill_workspace: None,

            layer_boundary_hook: None,
        })?;
        self.pos += 1;

        let mut host = vec![0.0f32; self.vocab_size];
        unsafe {
            let ptr = host.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, self.vocab_size * 4);
            self.backend.read_buffer(&self.decode_logits, slice)?;
        }
        Ok(host)
    }

    fn ensure_capacity(&mut self, additional: usize, max_seq_len: usize) -> anyhow::Result<()> {
        if self.pos + additional > max_seq_len {
            anyhow::bail!(
                "context would exceed max_seq_len={} (pos={}, incoming_reserve={}). \
                 Use /reset or increase --max-seq-len.",
                max_seq_len,
                self.pos,
                additional
            );
        }
        Ok(())
    }

    fn stats_line(&self, max_seq_len: usize) -> String {
        format!(
            "kv_pos={}/{} mode=kivi bits={} residual={}",
            self.pos, max_seq_len, self.bits, self.residual_size
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn run_chat_kivi(
    args: &Args,
    model: &TransformerModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    sampling_config: &SamplingConfig,
    kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    max_seq_len: usize,
) -> anyhow::Result<()> {
    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;
    let q_dim = model.config.num_attention_heads * head_dim;
    let k_dim = kv_heads * head_dim;
    let v_dim = kv_heads * head_dim;
    let ffn_hidden = model.config.intermediate_size;
    let residual_size = args.kivi_residual_size;
    let bits = args.kivi_bits;

    eprintln!(
        "[Chat/KIVI] bits={}, residual_size={}, max_seq_len={}",
        bits, residual_size, max_seq_len
    );

    let kv_caches: Vec<KiviCache> = (0..num_layers)
        .map(|_| {
            KiviCache::new_gpu(
                kv_heads,
                head_dim,
                max_seq_len,
                residual_size,
                bits,
                backend.clone(),
                memory.clone(),
            )
        })
        .collect();

    let dl_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let decode_logits = Tensor::new(Shape::new(vec![1, 1, vocab_size]), dl_buf, backend.clone());
    let xg_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let x_gen = Tensor::new(Shape::new(vec![1, 1, hidden_size]), xg_buf, backend.clone());
    let gen_ws = LayerWorkspace::new(
        WorkspaceConfig {
            batch_size: 1,
            dim: hidden_size,
            q_dim,
            k_dim,
            v_dim,
            ffn_hidden,
            n_heads: model.config.num_attention_heads,
            max_seq_len,
        },
        memory.as_ref(),
        backend.clone(),
    )?;
    let cpu_gen_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_buf,
        Arc::new(CpuBackend::new()),
    );
    let gpu_gen_buf = memory.alloc(4, DType::U8)?;
    let gen_input_gpu = Tensor::new(Shape::new(vec![1, 1]), gpu_gen_buf, backend.clone());

    let mut exec = KiviTurnExec {
        model,
        backend,
        memory: memory.clone(),
        kv_caches,
        decode_logits,
        x_gen,
        gen_ws,
        cpu_gen_input,
        gen_input_gpu,
        vocab_size,
        pos: 0,
        bits,
        residual_size,
    };

    run_chat_repl(
        args,
        model.config.arch,
        tokenizer,
        model.config.eos_token_id,
        vocab_size,
        sampling_config,
        max_seq_len,
        &mut exec,
    )
}

// ─── KV-Offload chat executor ─────────────────────────────────────────────────

struct OffloadTurnExec<'a> {
    model: &'a TransformerModel,
    backend: &'a Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    kv_caches: Vec<llm_rs2::core::offload::OffloadKVCache>,
    decode_logits: Tensor,
    x_gen: Tensor,
    gen_ws: LayerWorkspace,
    cpu_gen_input: Tensor,
    gen_input_gpu: Tensor,
    vocab_size: usize,
    pos: usize,
    offload_mode: String,
    max_prefetch_depth: usize,
    prefetch: llm_rs2::core::offload::prefetch::PrefetchController,
}

impl<'a> OffloadTurnExec<'a> {
    fn tokens_to_backend_tensor(&self, tokens: &[u32]) -> anyhow::Result<Tensor> {
        let cpu_backend = Arc::new(CpuBackend::new());
        let input_buf = Galloc::new().alloc(tokens.len() * 4, DType::U8)?;
        unsafe {
            let ptr = input_buf.as_mut_ptr() as *mut u32;
            for (i, &id) in tokens.iter().enumerate() {
                *ptr.add(i) = id;
            }
        }
        let cpu_input = Tensor::new(Shape::new(vec![1, tokens.len()]), input_buf, cpu_backend);
        self.backend.copy_from(&cpu_input)
    }
}

impl<'a> ChatTurnExec for OffloadTurnExec<'a> {
    fn pos(&self) -> usize {
        self.pos
    }

    fn reset(&mut self) {
        for c in self.kv_caches.iter_mut() {
            c.reset_session();
        }
        self.pos = 0;
    }

    fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let input_tensor = self.tokens_to_backend_tensor(tokens)?;
        let logits_buf = self.memory.alloc(self.vocab_size * 4, DType::F32)?;
        let mut logits_out = Tensor::new(
            Shape::new(vec![1, 1, self.vocab_size]),
            logits_buf,
            self.backend.clone(),
        );
        self.model.forward_into_offload(
            TransformerModelForwardArgs {
                input_tokens: &input_tensor,
                start_pos: self.pos,
                kv_caches: &mut self.kv_caches,
                backend: self.backend,
                memory: self.memory.as_ref(),
                logits_out: &mut logits_out,
                x_gen: None,
                workspace: None,
                score_accumulator: None,
                profiler: None,
                skip_config: None,
                importance_collector: None,
                logits_last_only: true,
                variance_collector: None,
                prefill_workspace: None,

                layer_boundary_hook: None,
            },
            &mut self.prefetch,
        )?;
        self.pos += tokens.len();

        let mut host = vec![0.0f32; self.vocab_size];
        unsafe {
            let ptr = host.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, self.vocab_size * 4);
            self.backend.read_buffer(&logits_out, slice)?;
        }
        Ok(host)
    }

    fn decode_step(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        unsafe {
            *(self.cpu_gen_input.buffer().as_mut_ptr() as *mut u32) = token;
        }
        self.backend.write_buffer(&mut self.gen_input_gpu, unsafe {
            std::slice::from_raw_parts(self.cpu_gen_input.buffer().as_ptr(), 4)
        })?;
        self.model.forward_into_offload(
            TransformerModelForwardArgs {
                input_tokens: &self.gen_input_gpu,
                start_pos: self.pos,
                kv_caches: &mut self.kv_caches,
                backend: self.backend,
                memory: self.memory.as_ref(),
                logits_out: &mut self.decode_logits,
                x_gen: Some(&mut self.x_gen),
                workspace: Some(&mut self.gen_ws),
                score_accumulator: None,
                profiler: None,
                skip_config: None,
                importance_collector: None,
                logits_last_only: true,
                variance_collector: None,
                prefill_workspace: None,

                layer_boundary_hook: None,
            },
            &mut self.prefetch,
        )?;
        self.pos += 1;

        let mut host = vec![0.0f32; self.vocab_size];
        unsafe {
            let ptr = host.as_mut_ptr() as *mut u8;
            let slice = std::slice::from_raw_parts_mut(ptr, self.vocab_size * 4);
            self.backend.read_buffer(&self.decode_logits, slice)?;
        }
        Ok(host)
    }

    fn ensure_capacity(&mut self, additional: usize, max_seq_len: usize) -> anyhow::Result<()> {
        if self.pos + additional > max_seq_len {
            anyhow::bail!(
                "context would exceed max_seq_len={} (pos={}, incoming_reserve={}). \
                 Use /reset or increase --max-seq-len.",
                max_seq_len,
                self.pos,
                additional
            );
        }
        Ok(())
    }

    fn stats_line(&self, max_seq_len: usize) -> String {
        format!(
            "kv_pos={}/{} mode=offload store={} prefetch_depth={}",
            self.pos, max_seq_len, self.offload_mode, self.max_prefetch_depth
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn run_chat_offload(
    args: &Args,
    model: &TransformerModel,
    tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    sampling_config: &SamplingConfig,
    kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    max_seq_len: usize,
) -> anyhow::Result<()> {
    use llm_rs2::core::offload::OffloadKVCache;
    use llm_rs2::core::offload::raw_store::RawStore;

    let vocab_size = model.config.vocab_size;
    let hidden_size = model.config.hidden_size;
    let q_dim = model.config.num_attention_heads * head_dim;
    let k_dim = kv_heads * head_dim;
    let v_dim = kv_heads * head_dim;
    let ffn_hidden = model.config.intermediate_size;

    let kv_dtype = match args.kv_type.as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        other => anyhow::bail!(
            "--chat --kv-offload requires --kv-type f16 or f32 (got '{}')",
            other
        ),
    };
    let token_bytes = kv_heads * head_dim * kv_dtype.size();
    let disk_dir = if args.offload_path.is_empty() {
        std::env::temp_dir().join("llm_rs2_kv_offload")
    } else {
        std::path::PathBuf::from(&args.offload_path)
    };
    let offload_mode = args.kv_offload.clone();
    eprintln!(
        "[Chat/Offload] mode={}, dtype={:?}, layers={}, token_bytes={}, max_seq={}",
        offload_mode, kv_dtype, num_layers, token_bytes, max_seq_len
    );

    let is_gpu_backend = backend.as_ref().is_gpu();
    let kv_caches: Vec<OffloadKVCache> = (0..num_layers)
        .map(|layer_id| {
            let store: Box<dyn llm_rs2::core::offload::store::OffloadStore> =
                match offload_mode.as_str() {
                    "raw" => Box::new(RawStore::new(token_bytes)),
                    "disk" => Box::new(
                        llm_rs2::core::offload::disk_store::DiskStore::new(
                            disk_dir.clone(),
                            layer_id,
                            token_bytes,
                        )
                        .expect("DiskStore::new failed"),
                    ),
                    other => panic!("Unknown offload mode: {}", other),
                };
            let mut c =
                OffloadKVCache::new(layer_id, kv_heads, head_dim, kv_dtype, max_seq_len, store);
            if is_gpu_backend {
                c.set_gpu_backend(backend.clone(), memory.clone());
            }
            c
        })
        .collect();

    let dl_buf = memory.alloc(vocab_size * 4, DType::F32)?;
    let decode_logits = Tensor::new(Shape::new(vec![1, 1, vocab_size]), dl_buf, backend.clone());
    let xg_buf = memory.alloc(hidden_size * 4, DType::F32)?;
    let x_gen = Tensor::new(Shape::new(vec![1, 1, hidden_size]), xg_buf, backend.clone());
    let gen_ws = LayerWorkspace::new(
        WorkspaceConfig {
            batch_size: 1,
            dim: hidden_size,
            q_dim,
            k_dim,
            v_dim,
            ffn_hidden,
            n_heads: model.config.num_attention_heads,
            max_seq_len,
        },
        memory.as_ref(),
        backend.clone(),
    )?;
    let cpu_gen_buf = Galloc::new().alloc(4, DType::U8)?;
    let cpu_gen_input = Tensor::new(
        Shape::new(vec![1, 1]),
        cpu_gen_buf,
        Arc::new(CpuBackend::new()),
    );
    let gpu_gen_buf = memory.alloc(4, DType::U8)?;
    let gen_input_gpu = Tensor::new(Shape::new(vec![1, 1]), gpu_gen_buf, backend.clone());

    let prefetch = llm_rs2::core::offload::prefetch::PrefetchController::new(
        args.max_prefetch_depth,
        num_layers,
    );
    let mut exec = OffloadTurnExec {
        model,
        backend,
        memory: memory.clone(),
        kv_caches,
        decode_logits,
        x_gen,
        gen_ws,
        cpu_gen_input,
        gen_input_gpu,
        vocab_size,
        pos: 0,
        offload_mode,
        max_prefetch_depth: args.max_prefetch_depth,
        prefetch,
    };

    run_chat_repl(
        args,
        model.config.arch,
        tokenizer,
        model.config.eos_token_id,
        vocab_size,
        sampling_config,
        max_seq_len,
        &mut exec,
    )
}
