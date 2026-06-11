//! AB-0: argus_bench experiment 경로.
//!
//! [`run_standard_happy_path`](crate::session::standard_happy) 와 동일한
//! prefill→sample→run 골격에 per-token JSONL writer + `[Experiment] Done`
//! summary + suspend 로그를 더한다. verify 하네스가 소비하는 산출물:
//! - `--experiment-output` JSONL: token record(token_id) + `_summary` record.
//!   verify 는 token record 를 세고(`count_decoded_tokens`) token_id 를
//!   재디코딩(accuracy)하며 `_summary.avg_tbt_ms` 로 performance 를 본다.
//! - stderr: `[Resilience] Inference suspended ...` (Suspend), `[Experiment] Done`.
//!
//! resilience directive(throttle/target_tbt/suspend) 의 런타임 효과는
//! [`DecodeLoop::run`](crate::session::DecodeLoop) 가 `ExecutionPlan` 을 읽어
//! 적용하므로 본 경로는 별도 처리하지 않는다 (avg_tbt 가 그 효과를 반영).

use crate::experiment::{JsonlWriter, SummaryRecord, SystemSampler, TokenRecord};
use crate::inference::sampling;
use crate::session::assembly::{
    SwapWiringConfig, build_bench_loop, build_local_pressure_source, build_resilience_cache_manager,
};
use crate::session::decode_loop::StopReason;
use crate::session::experiment::ScheduleCommandSource;
use crate::session::standard_happy::StandardHappyCtx;

pub fn run_experiment_path(ctx: StandardHappyCtx) -> anyhow::Result<()> {
    let StandardHappyCtx {
        args,
        backend,
        memory,
        hardware,
        model,
        tokenizer,
        kv_caches,
        max_seq_len,
        sampling_config,
        vocab_size,
        resilience,
        tokens,
    } = ctx;

    use crate::hardware::DeviceTarget;
    let cpu_backend_arc = hardware
        .resolve(DeviceTarget::Cpu)
        .expect("Cpu always resolves")
        .0
        .clone();

    eprintln!(
        "[argus-bench] experiment path → DecodeLoop+ModelForward (tokens={}, budget={})",
        tokens.len(),
        args.num_tokens
    );

    let mut sys_sampler = SystemSampler::new(args.experiment_sample_interval);
    let sys_start = args
        .experiment_output
        .as_ref()
        .map(|_| sys_sampler.snapshot());

    // AB-1: CLI `eviction <policy>` 로 resilience force-eviction CacheManager 구성
    // (eviction=none 이면 None → happy-path 동등). plan.evict directive 가 오면
    // decode 루프가 forward.try_evict 로 mid-decode prune.
    let cache_manager = build_resilience_cache_manager(&args, &backend)?;
    // β-5: graded 압력 source 는 cache_manager 가 있을 때(eviction/swap 활성)만 주입한다 —
    // happy-path(eviction=none + swap-dir 없음 → cache_manager=None)는 무주입해 per-token
    // /proc 읽기를 차단한다(G4). source 가 있어도 pressure 소비자(Persistent EvictionStage)가
    // 등록돼 있어야 실제 발화하며, N-step 캐시로 syscall 빈도를 제한한다.
    let pressure_source = cache_manager
        .as_ref()
        .map(|_| build_local_pressure_source(&args, &backend));
    // ADR-0008: bin_setup이 dispatch한 kv_caches를 소비(과거엔 drop 후 typed 재할당).
    let mut decode_loop = build_bench_loop(
        backend.clone(),
        memory.clone(),
        cpu_backend_arc.clone(),
        hardware.clone(),
        model,
        kv_caches,
        max_seq_len,
        sampling_config.clone(),
        !args.no_gpu_plan,
        resilience,
        cache_manager,
        pressure_source,
        args.eviction_target_ratio(),
        None, // γ-3b: argus-bench 는 schedule 없음 (IPC resilience 만)
        // AB-6: swap dispatch 설정. `--swap` 미지정 시 Incremental(LISWAP-6 production winner).
        SwapWiringConfig {
            default_mode: args
                .swap
                .unwrap_or(crate::session::cli::SwapMode::Incremental),
            phase_chunk_size_bytes: args.swap_phase_aware_chunk_mb * 1024 * 1024,
            phase_max_chunks_per_token: args.swap_phase_aware_max_chunks_per_token,
        },
    )?;

    let t_prefill = std::time::Instant::now();
    let mut last_logits = decode_loop.prefill(&tokens)?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

    let first_token = sampling::sample(
        &mut last_logits,
        &tokens,
        vocab_size,
        &sampling_config,
        None,
    );

    let t_decode = std::time::Instant::now();
    let result = decode_loop.run(args.num_tokens - 1, first_token)?;
    let decode_total_ms = t_decode.elapsed().as_secs_f64() * 1000.0;

    // Suspend 시 break → CommandRequested. legacy 와 동일 문자열을 emit 하여
    // verify thermal_emergency_suspend 의 stderr_pattern 을 충족한다.
    if result.stopped_by == StopReason::CommandRequested {
        eprintln!("\n[Resilience] Inference suspended by system signal");
    }

    let mut final_tokens: Vec<u32> = tokens.clone();
    final_tokens.push(first_token);
    final_tokens.extend_from_slice(&result.tokens_generated);
    let decoded = tokenizer
        .decode(&final_tokens, true)
        .unwrap_or_else(|_| String::from("[decode error]"));
    println!("{}", decoded);

    let decode_tokens = result.tokens_generated.len();
    let total_gen = 1 + decode_tokens;
    let decode_per_tok = if decode_tokens > 0 {
        decode_total_ms / decode_tokens as f64
    } else {
        0.0
    };
    let avg_tbt = (prefill_ms + decode_total_ms) / total_gen as f64;
    println!("TTFT: {:.2} ms", prefill_ms);
    if decode_tokens > 0 {
        println!(
            "Decode: {:.2} ms/tok ({:.1} tok/s) [{} tokens]",
            decode_per_tok,
            1000.0 / decode_per_tok.max(0.001),
            decode_tokens,
        );
    }
    println!(
        "Avg TBT: {:.2} ms ({:.1} tokens/sec)",
        avg_tbt,
        1000.0 / avg_tbt.max(0.001),
    );

    // ── experiment JSONL: per-token record + _summary ──
    if let Some(path) = args.experiment_output.as_ref() {
        let prompt_len = tokens.len();
        let generated: Vec<u32> = std::iter::once(first_token)
            .chain(result.tokens_generated.iter().copied())
            .collect();

        let mut writer = JsonlWriter::new(path)?;
        for (i, &token_id) in generated.iter().enumerate() {
            let pos = prompt_len + i;
            // per-token wall-clock 분해는 보존하지 않는다 — verify 는 token_id 와
            // record 수만 소비하므로 평균값으로 채운다.
            let (tbt_ms, forward_ms) = if i == 0 {
                (prefill_ms, prefill_ms)
            } else {
                (decode_per_tok, decode_per_tok)
            };
            let record = TokenRecord {
                pos,
                token_id,
                text: String::new(),
                tbt_ms,
                forward_ms,
                signal: None,
                actions: Vec::new(),
                cache_pos: pos,
                throttle_ms: 0,
                top_logits: Vec::new(),
                sys: sys_sampler.sample(pos),
            };
            writer.write_token(&record)?;
        }

        let prompt_text = tokenizer
            .decode(&tokens, true)
            .unwrap_or_else(|_| String::new());
        let summary = SummaryRecord {
            _summary: true,
            total_tokens: total_gen,
            ttft_ms: prefill_ms,
            avg_tbt_ms: avg_tbt,
            avg_forward_ms: decode_per_tok,
            total_throttle_ms: 0,
            eviction_count: 0,
            evicted_tokens_total: 0,
            final_cache_pos: result.final_pos,
            max_seq_len,
            prompt: prompt_text,
            schedule_name: String::new(),
            eviction_policy: args.eviction_policy().to_string(),
            backend: args.backend.clone(),
            sample_interval: args.experiment_sample_interval,
            sys_start,
            sys_end: Some(sys_sampler.snapshot()),
            governor: Some(SystemSampler::read_governor()),
        };
        writer.write_summary(&summary)?;

        eprintln!(
            "[Experiment] Done: {} tokens, avg TBT {:.2}ms, {} evictions",
            total_gen, avg_tbt, 0
        );
    }

    eprintln!(
        "[argus-bench] generated={} (first={} + run={}) stopped_by={:?} final_pos={}",
        total_gen, first_token, decode_tokens, result.stopped_by, result.final_pos
    );
    Ok(())
}

/// γ-3b: argus-eval experiment 모드 — 정적 `ScheduleCommandSource` 를 β-4 CommandSource
/// seam 에 주입하여 generation 실행. `run_experiment_path` 와 동일한 prefill→decode 골격에
/// schedule-driven directive 를 더한다.
///
/// ## JSONL 산출
///
/// `--experiment-output` 이 지정된 경우 `run_experiment_path` 와 동일한 JSONL + `_summary`
/// 레코드를 기록한다. verify 하네스 호환.
pub fn run_experiment_schedule_path(
    ctx: StandardHappyCtx,
    schedule_source: ScheduleCommandSource,
) -> anyhow::Result<()> {
    let StandardHappyCtx {
        args,
        backend,
        memory,
        hardware,
        model,
        tokenizer,
        kv_caches,
        max_seq_len,
        sampling_config,
        vocab_size,
        resilience,
        tokens,
    } = ctx;

    use crate::hardware::DeviceTarget;
    let cpu_backend_arc = hardware
        .resolve(DeviceTarget::Cpu)
        .expect("Cpu always resolves")
        .0
        .clone();

    eprintln!(
        "[argus-eval] experiment path → ScheduleCommandSource (tokens={}, budget={})",
        tokens.len(),
        args.num_tokens
    );

    let mut sys_sampler = SystemSampler::new(args.experiment_sample_interval);
    let sys_start = args
        .experiment_output
        .as_ref()
        .map(|_| sys_sampler.snapshot());

    let cache_manager = build_resilience_cache_manager(&args, &backend)?;
    let pressure_source = cache_manager
        .as_ref()
        .map(|_| build_local_pressure_source(&args, &backend));

    let mut decode_loop = build_bench_loop(
        backend.clone(),
        memory.clone(),
        cpu_backend_arc.clone(),
        hardware.clone(),
        model,
        kv_caches,
        max_seq_len,
        sampling_config.clone(),
        !args.no_gpu_plan,
        resilience,
        cache_manager,
        pressure_source,
        args.eviction_target_ratio(),
        Some(schedule_source),
        // AB-6: swap dispatch 설정 (schedule 모드도 secondary 보유 시 swap 활성).
        SwapWiringConfig {
            default_mode: args
                .swap
                .unwrap_or(crate::session::cli::SwapMode::Incremental),
            phase_chunk_size_bytes: args.swap_phase_aware_chunk_mb * 1024 * 1024,
            phase_max_chunks_per_token: args.swap_phase_aware_max_chunks_per_token,
        },
    )?;

    let t_prefill = std::time::Instant::now();
    let mut last_logits = decode_loop.prefill(&tokens)?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

    let first_token = sampling::sample(
        &mut last_logits,
        &tokens,
        vocab_size,
        &sampling_config,
        None,
    );

    let t_decode = std::time::Instant::now();
    let result = decode_loop.run(args.num_tokens - 1, first_token)?;
    let decode_total_ms = t_decode.elapsed().as_secs_f64() * 1000.0;

    if result.stopped_by == StopReason::CommandRequested {
        eprintln!("\n[Resilience] Inference suspended by system signal");
    }

    let mut final_tokens: Vec<u32> = tokens.clone();
    final_tokens.push(first_token);
    final_tokens.extend_from_slice(&result.tokens_generated);
    let decoded = tokenizer
        .decode(&final_tokens, true)
        .unwrap_or_else(|_| String::from("[decode error]"));
    println!("{}", decoded);

    let decode_tokens = result.tokens_generated.len();
    let total_gen = 1 + decode_tokens;
    let decode_per_tok = if decode_tokens > 0 {
        decode_total_ms / decode_tokens as f64
    } else {
        0.0
    };
    let avg_tbt = (prefill_ms + decode_total_ms) / total_gen as f64;
    println!("TTFT: {:.2} ms", prefill_ms);
    if decode_tokens > 0 {
        println!(
            "Decode: {:.2} ms/tok ({:.1} tok/s) [{} tokens]",
            decode_per_tok,
            1000.0 / decode_per_tok.max(0.001),
            decode_tokens,
        );
    }
    println!(
        "Avg TBT: {:.2} ms ({:.1} tokens/sec)",
        avg_tbt,
        1000.0 / avg_tbt.max(0.001),
    );

    if let Some(path) = args.experiment_output.as_ref() {
        let prompt_len = tokens.len();
        let generated: Vec<u32> = std::iter::once(first_token)
            .chain(result.tokens_generated.iter().copied())
            .collect();

        let mut writer = JsonlWriter::new(path)?;
        for (i, &token_id) in generated.iter().enumerate() {
            let pos = prompt_len + i;
            let (tbt_ms, forward_ms) = if i == 0 {
                (prefill_ms, prefill_ms)
            } else {
                (decode_per_tok, decode_per_tok)
            };
            let record = TokenRecord {
                pos,
                token_id,
                text: String::new(),
                tbt_ms,
                forward_ms,
                signal: None,
                actions: Vec::new(),
                cache_pos: pos,
                throttle_ms: 0,
                top_logits: Vec::new(),
                sys: sys_sampler.sample(pos),
            };
            writer.write_token(&record)?;
        }

        let prompt_text = tokenizer
            .decode(&tokens, true)
            .unwrap_or_else(|_| String::new());
        let summary = SummaryRecord {
            _summary: true,
            total_tokens: total_gen,
            ttft_ms: prefill_ms,
            avg_tbt_ms: avg_tbt,
            avg_forward_ms: decode_per_tok,
            total_throttle_ms: 0,
            eviction_count: 0,
            evicted_tokens_total: 0,
            final_cache_pos: result.final_pos,
            max_seq_len,
            prompt: prompt_text,
            schedule_name: String::new(),
            eviction_policy: args.eviction_policy().to_string(),
            backend: args.backend.clone(),
            sample_interval: args.experiment_sample_interval,
            sys_start,
            sys_end: Some(sys_sampler.snapshot()),
            governor: Some(SystemSampler::read_governor()),
        };
        writer.write_summary(&summary)?;

        eprintln!(
            "[Experiment] Done: {} tokens, avg TBT {:.2}ms",
            total_gen, avg_tbt,
        );
    }

    eprintln!(
        "[argus-eval] experiment generated={} (first={} + run={}) stopped_by={:?} final_pos={}",
        total_gen, first_token, decode_tokens, result.stopped_by, result.final_pos
    );
    Ok(())
}
