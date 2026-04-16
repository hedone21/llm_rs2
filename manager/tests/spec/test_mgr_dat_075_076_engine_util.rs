//! MGR-DAT-075, MGR-DAT-076, MSG-069: LuaPolicy `ctx.engine` 하위의
//! Engine self-utilization 필드 노출 계약을 검증한다.
//!
//! 2026-04 Phase 1~2 — Engine이 Heartbeat로 전달한 `self_cpu_pct`/`self_gpu_pct`가
//! Manager 측에서 소실·재해석 없이 `ctx.engine.cpu_pct`/`ctx.engine.gpu_pct`로
//! Lua 평가 컨텍스트에 그대로 전달되는지 확인한다. 또한 Pressure6D나
//! EwmaReliefTable 계산에 섞여 들어가지 않음(영향 없음)을 확인한다.
//!
//! Phase 2에서는 self_gpu_pct가 실측값(0.0~1.0)일 수 있으며, Manager는 여전히
//! 해석/변형하지 않고 그대로 전달해야 한다.
//!
//! 실행 조건: LuaPolicy가 기본 경로이므로 feature gate 없음. 기존
//! `#[cfg(feature = "hierarchical")]` 테스트와 독립적으로 실행된다.

#![allow(clippy::needless_doctest_main)]

use std::io::Write;

use llm_manager::config::AdaptationConfig;
use llm_manager::lua_policy::LuaPolicy;
use llm_manager::pipeline::PolicyStrategy;
use llm_shared::{
    ComputeReason, EngineCommand, EngineMessage, EngineState, EngineStatus, Level, ResourceLevel,
    SystemSignal,
};

// ─── Helpers ──────────────────────────────────────────────────────────────

fn write_script(body: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new().suffix(".lua").tempfile().unwrap();
    f.write_all(body.as_bytes()).unwrap();
    f
}

fn new_policy(script: &str) -> (LuaPolicy, tempfile::NamedTempFile) {
    let f = write_script(script);
    // qcf_penalty_weight=0.0 으로 QCF 선발행 로직을 비활성화한다.
    // 이 파일의 테스트는 ctx.engine / ctx.signal 매핑만 검증하므로 QCF 경로가 개입하면 안 된다.
    let config = AdaptationConfig {
        qcf_penalty_weight: 0.0,
        ..AdaptationConfig::default()
    };
    let p = LuaPolicy::with_system_clock(f.path().to_str().unwrap(), config).unwrap();
    (p, f)
}

fn heartbeat(self_cpu: f64, self_gpu: f64) -> EngineMessage {
    EngineMessage::Heartbeat(EngineStatus {
        active_device: "opencl".to_string(),
        compute_level: ResourceLevel::Normal,
        actual_throughput: 15.0,
        memory_level: ResourceLevel::Normal,
        kv_cache_bytes: 0,
        kv_cache_tokens: 0,
        kv_cache_utilization: 0.0,
        memory_lossless_min: 1.0,
        memory_lossy_min: 0.01,
        state: EngineState::Running,
        tokens_generated: 0,
        available_actions: vec![],
        active_actions: vec![],
        eviction_policy: "none".into(),
        kv_dtype: "f16".into(),
        skip_ratio: 0.0,
        phase: String::new(),
        prefill_pos: 0,
        prefill_total: 0,
        partition_ratio: 0.0,
        self_cpu_pct: self_cpu,
        self_gpu_pct: self_gpu,
    })
}

fn compute_signal(sys_cpu_pct: f64) -> SystemSignal {
    SystemSignal::ComputeGuidance {
        level: Level::Normal,
        cpu_usage_pct: sys_cpu_pct,
        gpu_usage_pct: 0.0,
        reason: ComputeReason::Balanced,
        recommended_backend: llm_shared::RecommendedBackend::Any,
    }
}

/// `ctx.engine.cpu_pct`(f64)를 SetTargetTbt.target_ms(u64)로 인코딩하는 스크립트.
/// 값 * 1000 후 반올림. Lua에서 읽은 값의 raw 접근을 최소 오차로 검증한다.
const SCRIPT_ECHO_CPU_PCT_AS_TBT: &str = r#"
function decide(ctx)
  local v = ctx.engine.cpu_pct or 0.0
  return {{type = "set_target_tbt", target_ms = math.floor(v * 1000 + 0.5)}}
end
"#;

const SCRIPT_ECHO_GPU_PCT_AS_TBT: &str = r#"
function decide(ctx)
  local v = ctx.engine.gpu_pct or 0.0
  return {{type = "set_target_tbt", target_ms = math.floor(v * 1000 + 0.5)}}
end
"#;

/// ctx.signal.compute.cpu_pct 기반 단순 결정. engine.cpu_pct에 의존하지 않음.
const SCRIPT_SIGNAL_ONLY: &str = r#"
function decide(ctx)
  local v = ctx.signal.compute.cpu_pct or 0.0
  return {{type = "set_target_tbt", target_ms = math.floor(v + 0.5)}}
end
"#;

fn extract_target_ms(dir: Option<llm_shared::EngineDirective>) -> Option<u64> {
    dir.and_then(|d| {
        d.commands.into_iter().find_map(|c| match c {
            EngineCommand::SetTargetTbt { target_ms } => Some(target_ms),
            _ => None,
        })
    })
}

// ---------------------------------------------------------------------------
// MGR-DAT-075: ctx.engine.cpu_pct 노출
// ---------------------------------------------------------------------------

#[test]
fn mgr_dat_075_ctx_engine_cpu_pct_reflects_heartbeat() {
    // SPEC: MGR-DAT-075, MSG-069
    let (mut p, _f) = new_policy(SCRIPT_ECHO_CPU_PCT_AS_TBT);
    p.update_engine_state(&heartbeat(0.42, 0.0));
    let dir = p.process_signal(&compute_signal(0.0));
    assert_eq!(
        extract_target_ms(dir),
        Some(420),
        "ctx.engine.cpu_pct가 0.42로 전달되어 420ms로 인코딩되어야 함"
    );
}

#[test]
fn mgr_dat_075_ctx_engine_cpu_pct_is_zero_before_first_heartbeat() {
    // SPEC: MGR-DAT-075
    let (mut p, _f) = new_policy(SCRIPT_ECHO_CPU_PCT_AS_TBT);
    // heartbeat 미주입 상태에서 signal만 송출
    let dir = p.process_signal(&compute_signal(0.0));
    assert_eq!(
        extract_target_ms(dir),
        Some(0),
        "heartbeat 없을 때 ctx.engine.cpu_pct는 0.0 default"
    );
}

// ---------------------------------------------------------------------------
// MGR-DAT-076: ctx.engine.gpu_pct Phase 1 placeholder
// ---------------------------------------------------------------------------

#[test]
fn mgr_dat_076_ctx_engine_gpu_pct_default_is_zero_without_meter() {
    // SPEC: MGR-DAT-076, MSG-068, INV-092
    // Phase 2에서도 meter 미주입(기본) 시 engine은 self_gpu_pct=0.0을 송출하며
    // Manager는 그대로 ctx.engine.gpu_pct=0.0으로 Lua에 노출한다.
    let (mut p, _f) = new_policy(SCRIPT_ECHO_GPU_PCT_AS_TBT);
    p.update_engine_state(&heartbeat(0.5, 0.0));
    let dir = p.process_signal(&compute_signal(0.0));
    assert_eq!(
        extract_target_ms(dir),
        Some(0),
        "meter 미주입 시 ctx.engine.gpu_pct는 0.0 (Phase 1 호환)"
    );
}

#[test]
fn mgr_dat_076_nonzero_self_gpu_pct_is_passed_through_untouched() {
    // SPEC: MGR-DAT-076 (Phase 2 forward compat)
    // Manager는 self_gpu_pct를 해석/변형하지 않고 Lua에 전달한다.
    let (mut p, _f) = new_policy(SCRIPT_ECHO_GPU_PCT_AS_TBT);
    p.update_engine_state(&heartbeat(0.0, 0.30));
    let dir = p.process_signal(&compute_signal(0.0));
    assert_eq!(
        extract_target_ms(dir),
        Some(300),
        "Manager는 self_gpu_pct를 변형하지 않고 0.30 → 300ms로 통과시켜야 함"
    );
}

// ---------------------------------------------------------------------------
// MSG-069: ctx.signal.compute.cpu_pct vs ctx.engine.cpu_pct 분리
// ---------------------------------------------------------------------------

#[test]
fn msg_069_ctx_signal_and_ctx_engine_cpu_are_independent() {
    // SPEC: MSG-069
    // signal 스케일(0~100)과 engine 스케일(0~1)이 혼합되지 않음을 스크립트 결정으로 검증.
    // signal이 80 → target_ms=80, engine이 0.25 → target_ms=250. 두 스크립트 각각 다른 값.
    let (mut p_sig, _f1) = new_policy(SCRIPT_SIGNAL_ONLY);
    p_sig.update_engine_state(&heartbeat(0.25, 0.0));
    let sig_dir = p_sig.process_signal(&compute_signal(80.0));
    assert_eq!(extract_target_ms(sig_dir), Some(80));

    let (mut p_eng, _f2) = new_policy(SCRIPT_ECHO_CPU_PCT_AS_TBT);
    p_eng.update_engine_state(&heartbeat(0.25, 0.0));
    let eng_dir = p_eng.process_signal(&compute_signal(80.0));
    assert_eq!(extract_target_ms(eng_dir), Some(250));
}

#[test]
fn msg_069_lua_can_compute_external_contention_from_raw_values() {
    // SPEC: MSG-069 (non-normative 예시, arch/20-manager.md §10.7)
    // external = (signal.compute.cpu_pct / 100) - engine.cpu_pct
    const SCRIPT: &str = r#"
function decide(ctx)
  local sys = (ctx.signal.compute.cpu_pct or 0.0) / 100.0
  local eng = ctx.engine.cpu_pct or 0.0
  local external = sys - eng
  return {{type = "set_target_tbt", target_ms = math.floor(external * 1000 + 0.5)}}
end
"#;
    let (mut p, _f) = new_policy(SCRIPT);
    p.update_engine_state(&heartbeat(0.30, 0.0));
    let dir = p.process_signal(&compute_signal(80.0));
    assert_eq!(
        extract_target_ms(dir),
        Some(500),
        "external = 0.80 - 0.30 = 0.50 → 500ms"
    );
}

// ---------------------------------------------------------------------------
// Pressure6D / relief 학습 경로 비침투 (regression guard)
// ---------------------------------------------------------------------------

#[test]
fn mgr_dat_075_self_cpu_pct_does_not_leak_into_pressure6d() {
    // SPEC: MGR-DAT-075 (Pressure6D 영향 없음)
    // Pressure6D는 lua_policy 내부 SignalState에서 계산되며 외부 관측 불가.
    // 대안: ctx.signal 기반으로만 결정하는 스크립트가 self_cpu_pct 변화에
    // 무관하게 동일한 EngineDirective를 반환함을 관측해 간접 검증.
    let (mut a, _fa) = new_policy(SCRIPT_SIGNAL_ONLY);
    a.update_engine_state(&heartbeat(0.0, 0.0));
    let dir_a = a.process_signal(&compute_signal(45.0));

    let (mut b, _fb) = new_policy(SCRIPT_SIGNAL_ONLY);
    b.update_engine_state(&heartbeat(0.9, 0.0));
    let dir_b = b.process_signal(&compute_signal(45.0));

    assert_eq!(
        extract_target_ms(dir_a),
        extract_target_ms(dir_b),
        "ctx.signal만 사용하는 스크립트는 self_cpu_pct와 무관해야 함 (Pressure6D 비침투의 간접 증거)"
    );
}

#[test]
fn mgr_dat_076_self_gpu_pct_does_not_leak_into_relief_observations() {
    // SPEC: MGR-DAT-076
    // relief observations는 pressure_with_thermal()에서 유도되며 self_gpu_pct를
    // 사용하지 않는다. 외부 관측 경로: observation이 진행되는 동안 두 시나리오의
    // EngineDirective가 동일한지 비교. signal-only 스크립트로 검증.
    let (mut a, _fa) = new_policy(SCRIPT_SIGNAL_ONLY);
    a.update_engine_state(&heartbeat(0.5, 0.0));
    let dir_a = a.process_signal(&compute_signal(50.0));

    let (mut b, _fb) = new_policy(SCRIPT_SIGNAL_ONLY);
    b.update_engine_state(&heartbeat(0.5, 0.9));
    let dir_b = b.process_signal(&compute_signal(50.0));

    assert_eq!(
        extract_target_ms(dir_a),
        extract_target_ms(dir_b),
        "self_gpu_pct 차이는 EngineDirective에 영향을 주지 않아야 함"
    );
}
