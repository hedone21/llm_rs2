//! argus-bench — resilience benchmark / verify 엔진 bin.
//!
//! argus_cli (single-prompt happy path) 와 같은 추론 골격을 공유하되,
//! `--experiment-output` per-token JSONL 산출 + resilience directive(throttle /
//! target_tbt / suspend …) 의 런타임 효과 측정을 지원한다. verify 하네스
//! (`verify/verify.py`) 가 이 bin 을 baseline/action 으로 띄워 SystemSignal →
//! Policy → EngineCommand → Engine 전 경로를 검증한다.
//!
//! ## 단계적 흡수 (argus-bench AB sub-sprint)
//!
//! - **AB-0 (current)**: experiment-output JSONL + `[Experiment] Done` summary,
//!   throttle·target_tbt·suspend 런타임 효과. eviction·quant·offload·partition·
//!   weight-swap 는 후속 단계에서 stage 로 재배선.
//! - AB-1: EvictionStage (plan.evict) 재배선.
//! - AB-2: KIVI dynamic quant. AB-3: KvOffload. AB-4: tensor partition.
//! - AB-6: weight-swap 8종 SwapStage glue.

use anyhow::bail;
use clap::Parser;
use llm_rs2::session::bin_setup::build_inference_ctx;
use llm_rs2::session::cli::{Args, KvMode};
use llm_rs2::session::experiment_run::run_experiment_path;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let mut args = Args::parse();

    // `--swap` shorthand → legacy 4 flag normalize (argus_cli 와 동일).
    args.normalize_swap_shorthand();

    // resilience default-on. `--no-resilience` 명시 시 effective=false.
    args.enable_resilience = !args.no_resilience;

    reject_unsupported_modes_ab0(&args)?;

    // AB-1: standard happy path + resilience runtime effect + CLI `eviction
    // <policy>` (mid-decode KvEvict* directive). throttle/target_tbt/suspend/
    // eviction 은 런타임 directive 이므로 가드를 통과한다. 미지원 모드(skip /
    // d2o-layer-alloc / qcf / profile / partition / swap)만 차단.
    if !bench_supported(&args) {
        bail!(
            "argus-bench AB-1: this combination of args is not yet supported. \
             supported: eviction <none|sliding|streaming|h2o|h2o_plus> + resilience. \
             blocked: qcf_dump, skip_ratio, d2o_layer_alloc, profile, profile_events, \
             tensor_partition, weight-swap (→ AB-2..AB-6)."
        );
    }
    if args.num_tokens < 1 {
        bail!("argus-bench: --num-tokens must be >= 1");
    }

    let ctx = build_inference_ctx(args)?;
    run_experiment_path(ctx)
}

/// AB-1 bench 지원 args 가드. [`is_standard_happy_path`](llm_rs2::session::is_standard_happy_path)
/// 와 동일하되 **eviction_policy != "none" 을 허용**한다 (resilience eviction은
/// `build_bench_loop` 가 처리). 아직 미지원인 skip / d2o-layer-alloc / qcf /
/// profile / tensor_partition / weight-swap 은 그대로 차단.
fn bench_supported(args: &Args) -> bool {
    args.qcf_dump.is_none()
        && args.skip_ratio.unwrap_or(0.0) == 0.0
        && !args.d2o_layer_alloc()
        && !args.profile
        && !args.profile_events
        && args.tensor_partition == 0.0
        && !args.swap_intra_forward
        && !args.swap_layer_immediate
        && !args.swap_phase_aware
}

/// AB-0 에서 미구현인 mode 진입 flag 를 검사하여 즉시 reject 한다.
/// argus_cli 의 reject 와 동일하되 `--experiment-output` 은 허용한다 (argus-bench
/// 의 핵심 기능). eviction/swap/KIVI/offload/partition 은 AB-1..AB-6 에서 해제.
fn reject_unsupported_modes_ab0(args: &Args) -> anyhow::Result<()> {
    if args.chat {
        bail!("argus-bench AB-0: --chat moved to argus-chat (planned)");
    }
    if args.chat_socket.is_some() || args.chat_tcp.is_some() {
        bail!("argus-bench AB-0: --chat-socket / --chat-tcp moved to argus-chat (planned)");
    }
    if args.experiment_schedule.is_some() {
        bail!(
            "argus-bench AB-0: --experiment-schedule (static schedule) unsupported; use mock_manager/manager IPC resilience"
        );
    }
    if args.ppl.is_some() {
        bail!("argus-bench AB-0: --ppl moved to argus-eval --ppl");
    }
    if args.eval_ll || args.eval_batch.is_some() || args.eval_continuation.is_some() {
        bail!(
            "argus-bench AB-0: --eval-ll / --eval-batch / --eval-continuation moved to argus-eval --eval-ll"
        );
    }
    if args.dump_importance {
        bail!("argus-bench AB-0: --dump-importance moved to argus-eval --dump-importance");
    }
    if args.qcf_dump.is_some() {
        bail!(
            "argus-bench AB-0: --qcf-dump moved to argus-eval (--qcf-dump with --eval-ll or --ppl)"
        );
    }
    if !matches!(args.effective_kv_mode(), KvMode::Standard) {
        bail!("argus-bench AB-0: --kv-mode KIVI/Offload land in AB-2/AB-3");
    }
    if args.secondary_gguf.is_some()
        || args.force_swap_ratio.is_some()
        || args.swap_incremental_per_tick > 0
        || args.swap_intra_forward
        || args.swap_layer_immediate
        || args.swap_phase_aware
    {
        bail!("argus-bench AB-0: weight swap CLI options land in AB-6");
    }
    if args.profile || args.profile_events {
        bail!("argus-bench AB-0: --profile / --profile-events not yet supported");
    }
    if args.tensor_partition > 0.0 {
        bail!("argus-bench AB-0: --tensor-partition lands in AB-4");
    }
    Ok(())
}
