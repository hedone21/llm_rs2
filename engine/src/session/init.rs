use std::sync::Arc;

use crate::backend::cpu::CpuBackend;
use crate::core::backend::Backend;
use crate::core::buffer::DType;
use crate::core::memory::Memory;
use crate::core::rss_trace::{dump_smaps, io_trace, rss_trace};
use crate::core::sampling::SamplingConfig;
use crate::memory::galloc::Galloc;
use crate::models::transformer::TransformerModel;
use crate::session::cli::Args;

/// Session 초기화 컨텍스트 (Phase 4-1 외곽 추출).
///
/// `SessionInitCtx::build(&args)`가 generate.rs main()의 args 검증,
/// 환경변수 전파, Rayon 초기화, Backend/Memory init, 모델 로드, weight probe를
/// 수행한다. build() 완료 후 ctx는 모든 하위 decode 경로에서 필요한 값들을 보유한다.
///
/// - `args`는 main()이 owned 보유; build()는 `&Args` borrow + 필요 필드만 clone.
/// - `model`은 owned으로 보유 (KIVI/SwitchHw가 model.layers를 mutation).
/// - `is_gpu`, `weights_on_gpu`는 main()이 mutation할 수 있도록 pub으로 노출.
pub struct SessionInitCtx {
    /// sampling 파라미터 (greedy override 처리 완료).
    pub sampling_config: SamplingConfig,
    /// 모델 파일 경로 (GGUF 또는 safetensors 디렉토리).
    pub model_path: String,
    /// 모델이 GGUF 포맷이면 true.
    pub is_gguf: bool,

    /// 주 backend (CPU 또는 GPU).
    pub backend: Arc<dyn Backend>,
    /// 주 메모리 할당자.
    pub memory: Arc<dyn Memory>,
    /// SwitchHw 용 GPU secondary backend (CPU primary일 때 Some).
    pub gpu_backend_arc: Option<Arc<dyn Backend>>,
    /// SwitchHw 용 GPU secondary 메모리 할당자.
    pub gpu_memory_arc: Option<Arc<dyn Memory>>,
    /// 현재 GPU가 primary backend이면 true (main()이 SwitchHw 후 mutation).
    pub is_gpu: bool,
    /// 모델 weights가 GPU cl_mem에 있으면 true.
    pub weights_on_gpu: bool,

    /// CPU 백엔드 (SwitchHw fallback, tensor partition, weight migration).
    pub cpu_backend_arc: Arc<dyn Backend>,
    /// CPU 메모리 할당자.
    pub cpu_memory_arc: Arc<dyn Memory>,

    /// swap layer 선택 알고리즘 (--swap-algorithm).
    pub swap_algorithm: crate::models::weights::SwapAlgorithm,
    /// layer 중요도 계산 공식 (--importance-formula).
    pub importance_formula: crate::core::qcf::ImportanceFormula,
    /// `compare` 모드 활성 여부 (importance_formula = MeanPool이지만 3-way 수집).
    pub importance_compare: bool,
    /// 명시적 swap 대상 layer 목록 (--swap-only-layers, § 4 ground-truth study).
    pub swap_only_layers: Option<Vec<usize>>,

    /// 로드된 모델 (KIVI/SwitchHw가 model.layers를 swap하므로 owned).
    pub model: TransformerModel,
}

impl SessionInitCtx {
    pub fn build(args: &Args) -> anyhow::Result<Self> {
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
            crate::backend::cpu::neon::USE_RAYON.store(true, std::sync::atomic::Ordering::Relaxed);
            eprintln!("[Config] F16 matmul: Rayon (par_chunks_mut)");
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
                anyhow::bail!(
                    "--chat: --kivi cannot combine with --eviction-policy in v1 (pick one)"
                );
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

        // --greedy overrides temperature to 0 (args is &Args so we apply inline)
        let effective_temperature = if args.greedy { 0.0 } else { args.temperature };
        let sampling_config = SamplingConfig {
            temperature: effective_temperature,
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
        let (backend, memory, gpu_backend_arc, gpu_memory_arc, is_gpu): (
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
                    match crate::backend::opencl::OpenCLBackend::new_with_profile_events(
                        // MSG-068 Phase 2: heartbeat-gpu-profile도 같은 queue
                        // profiling 인프라를 사용하므로 어느 한쪽이 켜지면 활성화.
                        args.profile_events || args.heartbeat_gpu_profile,
                    ) {
                        Ok(gpu_concrete) => {
                            let gpu_concrete = Arc::new(gpu_concrete);
                            let gm: Arc<dyn Memory> =
                                Arc::new(crate::backend::opencl::memory::OpenCLMemory::new(
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
                    crate::backend::opencl::OpenCLBackend::new_with_profile_events(
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
                    Arc::new(crate::backend::opencl::memory::OpenCLMemory::new(
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
                let gpu_concrete = Arc::new(crate::backend::cuda::CudaBackend::new()?);
                let gpu_mem: Arc<dyn Memory> = if gpu_concrete.is_discrete_gpu() {
                    Arc::new(crate::backend::cuda::memory::CudaMemory::managed())
                } else {
                    Arc::new(crate::backend::cuda::memory::CudaMemory::new())
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
                // Resolves through `crate::backend::cuda` which aliases to
                // cuda_pc (feature = "cuda") or cuda_embedded (feature =
                // "cuda-embedded"); the two modules share the same
                // SyncPolicy API shape.
                #[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
                {
                    use crate::backend::cuda::SyncPolicy;
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
                let qnn = Arc::new(crate::backend::qnn_oppkg::QnnOppkgBackend::with_prebuild(
                    args.qnn_graph_cache_prebuild,
                )?);
                let qnn_mem: Arc<dyn Memory> = Arc::new(
                    crate::backend::qnn_oppkg::memory::QnnOppkgMemory::new(qnn.clone()),
                );

                // ENG-QNN-206: SwitchHw round-trip을 위해 OpenCL backend를 secondary로
                // 등록. OpenCL init이 fail하면 secondary 없이 진행 (SwitchHw 비활성).
                #[cfg(feature = "opencl")]
                let (gpu_be, gpu_mem_arc): (
                    Option<Arc<dyn Backend>>,
                    Option<Arc<dyn Memory>>,
                ) = match crate::backend::opencl::OpenCLBackend::new_with_profile_events(
                    args.profile_events || args.heartbeat_gpu_profile,
                ) {
                    Ok(gpu_concrete) => {
                        let gpu_concrete = Arc::new(gpu_concrete);
                        let gm: Arc<dyn Memory> =
                            Arc::new(crate::backend::opencl::memory::OpenCLMemory::new(
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
                            .downcast_ref::<crate::backend::opencl::OpenCLBackend>(
                        ) {
                            eprintln!(
                                "[Backend] QNN primary_mem → QnnOppkgHybridMemory (KV zero-copy Step 1)"
                            );
                            Arc::new(
                                crate::backend::qnn_oppkg::hybrid_memory::QnnOppkgHybridMemory::new(
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
        let swap_algorithm = crate::models::weights::SwapAlgorithm::from_cli(&args.swap_algorithm)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "--swap-algorithm: unknown value '{}'. Valid: imp, seq, rev, uni, anti",
                    args.swap_algorithm
                )
            })?;

        // Parse --importance-formula (§4 EuroSys'27 study). `compare` enables
        // three_way collector + post-warmup DP-LLM proxy ε computation.
        let (importance_formula, importance_compare) = match args.importance_formula.as_str() {
            "mean_pool" => (crate::core::qcf::ImportanceFormula::MeanPool, false),
            "shortgpt_bi" => (crate::core::qcf::ImportanceFormula::ShortGptBi, false),
            "dpllm_proxy" => (crate::core::qcf::ImportanceFormula::DpllmProxy, false),
            "dpllm_multi" => (crate::core::qcf::ImportanceFormula::DpllmMulti, false),
            "dpllm_abs" => (crate::core::qcf::ImportanceFormula::DpllmAbs, false),
            "dpllm_qcf" => (crate::core::qcf::ImportanceFormula::DpllmQcf, false),
            "direct_attn" => (crate::core::qcf::ImportanceFormula::DirectAttn, false),
            "compare" => (crate::core::qcf::ImportanceFormula::MeanPool, true),
            other => anyhow::bail!(
                "--importance-formula: unknown value '{}'. Valid: mean_pool, shortgpt_bi, dpllm_proxy, dpllm_multi, dpllm_abs, dpllm_qcf, direct_attn, compare",
                other
            ),
        };

        // Parse --swap-only-layers (§4 ground-truth study). CSV of layer indices.
        // Order is preserved (trajectory mode swaps in the listed order); duplicates
        // are dropped while keeping the first occurrence.
        let swap_only_layers: Option<Vec<usize>> = match args.swap_only_layers.as_deref() {
            None | Some("") => None,
            Some(csv) => {
                let mut v = Vec::new();
                let mut seen = std::collections::HashSet::new();
                for tok in csv.split(',') {
                    let t = tok.trim();
                    if t.is_empty() {
                        continue;
                    }
                    let idx: usize = t.parse().map_err(|_| {
                        anyhow::anyhow!("--swap-only-layers: '{}' is not a non-negative integer", t)
                    })?;
                    if seen.insert(idx) {
                        v.push(idx);
                    }
                }
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
            let load_cfg = crate::models::loader::LoadConfig {
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
        io_trace("model_loaded");
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
                .downcast_ref::<crate::backend::qnn_oppkg::QnnOppkgBackend>()
            {
                // ModelConfig → LayerConfig 변환. M3.2 단계는 Qwen2.5-1.5B 단일
                // 모델 지원 (ENG-QNN-225 / INV-176). 추후 다른 모델 추가 시
                // dispatch table 도입 예정.
                let mc = &model.config;
                let layer_cfg = crate::backend::qnn_oppkg::layer_graph::LayerConfig {
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
                io_trace("before_prefault");
                let t0 = std::time::Instant::now();
                secondary.prefault();
                eprintln!(
                    "[Eager-Prefault] secondary weights prefaulted in {:.1}ms",
                    t0.elapsed().as_secs_f64() * 1e3
                );
                io_trace("after_prefault");
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
            && !crate::layers::tensor_partition::is_gpu_only_ratio(args.tensor_partition);
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
                        Error(crate::auf::AufError),
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
                            let payload = crate::auf::LmHeadPayload {
                                bytes: &bytes,
                                shape,
                                dtype: crate::auf::TensorDType::Q4_0,
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
                            anyhow::bail!(
                                "--quantize-lm-head AUF invariant violation (INV-135): {e}"
                            );
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
                eprintln!(
                    "[Backend] lm_head: forced runtime quantize (AUF entry ignored, mode=q4_0)"
                );
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
                    crate::layers::tensor_partition::GPU_ONLY_THRESHOLD,
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
                // Phase 4-4.10: noshuffle SOA conversion is now opt-in.
                //
                // Background: noshuffle SOA reclaims ≈702.8 MiB on Qwen 2.5-1.5B
                // Q4_0 but assumes GPU-only inference. Active weights converted to
                // SOA cannot back CPU fallback paths (switch_hw cpu, tensor
                // partition CPU split, prefill CPU chunking) without silent garbage.
                // It also relies on `kernel_gemv_noshuffle_q4_0` which currently
                // regresses Adreno decode TBT by +13% vs the standard Q4_0 GEMV
                // (Phase 4-4.8 measurement).
                //
                // Defaulting to AOS keeps the CPU fallback story intact and avoids
                // the noshuffle GEMV regression. Memory savings can be opted back
                // in once the `.cl` kernel is tuned for Adreno (backlog Path B).
                //
                // Env flags:
                //   LLMRS_ENABLE_NOSHUFFLE_SOA=1 → opt in to SOA conversion
                //   LLMRS_SKIP_NOSHUFFLE_SOA=1   → legacy override, always skip
                let soa_opt_in = std::env::var_os("LLMRS_ENABLE_NOSHUFFLE_SOA").is_some();
                let soa_skip = std::env::var_os("LLMRS_SKIP_NOSHUFFLE_SOA").is_some();
                if !soa_opt_in || soa_skip {
                    let reason = if soa_skip {
                        "LLMRS_SKIP_NOSHUFFLE_SOA set"
                    } else {
                        "default AOS (set LLMRS_ENABLE_NOSHUFFLE_SOA=1 to opt in)"
                    };
                    eprintln!(
                        "[NoShuffle] Skipped: {} — decode uses standard Q4_0 GEMV",
                        reason
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
                        Ok(n) => {
                            eprintln!("[Backend] Noshuffle SOA prepared: {} weight tensors", n)
                        }
                        Err(e) => eprintln!("[Backend] Noshuffle preparation skipped: {}", e),
                    }
                    // WSWAP-5-TBT-DIAG: dump cl_mem footprint immediately after
                    // primary noshuffle prep so the Q4 baseline allocation
                    // pattern is recorded *before* any AUF SOA bypass swap path
                    // adds placeholder cl_mems on top.
                    #[cfg(feature = "opencl")]
                    if let Some(ocl_be) = backend
                        .as_any()
                        .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
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
            crate::backend::opencl::get_cl_mem(layer0.wq.buffer().as_ref()).is_ok()
        };
        #[cfg(not(feature = "opencl"))]
        let weights_on_gpu = false;

        Ok(SessionInitCtx {
            sampling_config,
            model_path: model_path.clone(),
            is_gguf,
            backend,
            memory,
            gpu_backend_arc,
            gpu_memory_arc,
            is_gpu,
            weights_on_gpu,
            cpu_backend_arc,
            cpu_memory_arc,
            swap_algorithm,
            importance_formula,
            importance_compare,
            swap_only_layers,
            model,
        })
    }
}
