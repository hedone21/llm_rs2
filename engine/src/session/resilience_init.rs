//! CommandExecutor 생성 헬퍼 (P3.2).
//!
//! `build_command_executor`는 legacy `generate.rs` L596~700의 CommandExecutor
//! 생성 블록을 외과적으로 이식한 함수다. argus-cli 경유로만 호출된다.
//!
//! - experiment_schedule 분기 제외 (argus-cli v0 reject)
//! - `args.enable_resilience` 가 false이면 `Ok(None)` 반환
//! - transport 연결 실패 시 graceful `Err` 전파 (panic 없음)

use std::time::Duration;

use anyhow::Result;

use crate::models::transformer::TransformerModel;
use crate::resilience::{CommandExecutor, MessageLoop, TcpTransport};
use crate::session::cli::Args;

/// Args + TransformerModel 메타에서 CommandExecutor를 생성한다.
///
/// `args.enable_resilience` 가 false이면 `Ok(None)`.
/// transport 연결 실패(connection refused 등)는 `Err` 로 전파된다.
pub fn build_command_executor(
    args: &Args,
    model: &TransformerModel,
) -> Result<Option<CommandExecutor>> {
    if !args.enable_resilience {
        return Ok(None);
    }

    let heartbeat_interval = Duration::from_millis(1000);

    // MSG-068 Phase 2: GPU self-util meter 추출.
    // OpenCL backend + heartbeat_gpu_profile 활성 시만 Some.
    // 그 외는 None → executor가 self_gpu_pct=0.0 송출.
    #[allow(unused_mut)]
    let mut gpu_meter: Option<std::sync::Arc<dyn crate::resilience::GpuSelfMeter>> = None;
    #[cfg(feature = "opencl")]
    if args.heartbeat_gpu_profile {
        // argus-cli는 SessionInitCtx 해체 후 호출되므로 backend Arc를 직접 갖지 않음.
        // gpu_meter 추출 기회가 없어 None 유지 — heartbeat self_gpu_pct=0.0.
        // P5+에서 backend Arc를 인자로 받는 별 overload 도입 예정.
        let _ = args.heartbeat_gpu_profile; // 컴파일러 경고 방지
    }

    // transport 분기
    let (cmd_rx, resp_tx, _handle) = match args.resilience_transport.as_str() {
        #[cfg(feature = "resilience")]
        "dbus" => {
            use crate::resilience::DbusTransport;
            MessageLoop::spawn(DbusTransport::new())?
        }
        #[cfg(unix)]
        s if s.starts_with("unix:") => {
            use crate::resilience::UnixSocketTransport;
            let path = std::path::PathBuf::from(&s[5..]);
            MessageLoop::spawn(UnixSocketTransport::new(path))?
        }
        s if s.starts_with("tcp:") => {
            let addr = s[4..].to_string();
            MessageLoop::spawn(TcpTransport::new(addr))?
        }
        other => {
            anyhow::bail!(
                "[Resilience] Unknown transport: '{}'. Use dbus / unix:<path> / tcp:<addr>",
                other
            );
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
        gpu_meter,
    );

    // argus-cli v0: secondary 없음 → swap_weights 액션 미포함.
    executor.set_has_secondary(false);

    // Capability 송출 (SEQ-022).
    // argus-cli v0 safe-only subset: swap / eviction / kv-quant 제외.
    let available_actions = vec![
        "throttle".to_string(),
        "set_target_tbt".to_string(),
        "suspend".to_string(),
        "reject_new".to_string(),
        "limit_tokens".to_string(),
        "restore_defaults".to_string(),
    ];
    executor.send_capability(llm_shared::EngineCapability {
        available_devices: vec!["cpu".to_string(), "opencl".to_string()],
        active_device: args.backend.clone(),
        max_kv_tokens: args.max_seq_len,
        bytes_per_kv_token: model.config.num_key_value_heads
            * model.config.head_dim
            * 2  // K + V
            * 2, // F16 = 2 bytes
        num_layers: model.config.num_hidden_layers,
        available_actions,
    });
    eprintln!("[Resilience] Capability sent to Manager");

    // tensor_partition seed (happy path에서는 0.0 → no-op이지만 무회귀 보존)
    if args.tensor_partition > 0.0 && args.tensor_partition < 1.0 {
        executor.set_partition_ratio(args.tensor_partition);
    }
    // throttle seed
    if args.throttle_delay_ms > 0 {
        executor.set_throttle_delay_ms(args.throttle_delay_ms);
    }

    Ok(Some(executor))
}
