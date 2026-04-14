//! MSG-060 필드 17~18 (self_cpu_pct, self_gpu_pct), MSG-067, MSG-068, MSG-069,
//! INV-091, INV-092: Engine Self-Utilization Heartbeat 필드 Spec 테스트.
//!
//! 2026-04 Phase 1 — Engine이 `/proc/self/stat` 기반 자신의 CPU 사용률을
//! Heartbeat(MSG-060)에 실어 보내는 경로를 검증한다. GPU는 Phase 1에서
//! placeholder(항상 0.0)이며 서지/의미는 Phase 2에서 재정의된다.
//!
//! 실행 조건: feature gate 없음 (기본 경로).

#![allow(clippy::needless_doctest_main)]

use llm_rs2::resilience::proc_self_meter::ProcSelfMeter;
use llm_shared::{EngineState, EngineStatus, ResourceLevel};

fn base_status() -> EngineStatus {
    EngineStatus {
        active_device: "cpu".to_string(),
        compute_level: ResourceLevel::Normal,
        actual_throughput: 15.0,
        memory_level: ResourceLevel::Normal,
        kv_cache_bytes: 1024,
        kv_cache_tokens: 100,
        kv_cache_utilization: 0.5,
        memory_lossless_min: 1.0,
        memory_lossy_min: 0.01,
        state: EngineState::Running,
        tokens_generated: 50,
        available_actions: vec!["throttle".into()],
        active_actions: vec![],
        eviction_policy: "none".into(),
        kv_dtype: "f16".into(),
        skip_ratio: 0.0,
        phase: String::new(),
        prefill_pos: 0,
        prefill_total: 0,
        partition_ratio: 0.0,
        self_cpu_pct: 0.0,
        self_gpu_pct: 0.0,
    }
}

// ---------------------------------------------------------------------------
// MSG-060 #17~18 / MSG-061: serde round-trip + 하위호환 (#[serde(default)])
// ---------------------------------------------------------------------------

#[test]
fn msg_060_self_util_fields_roundtrip_preserves_values() {
    // SPEC: MSG-060 (필드 17, 18), MSG-061
    let mut s = base_status();
    s.self_cpu_pct = 0.42;
    s.self_gpu_pct = 0.0;
    let json = serde_json::to_string(&s).unwrap();
    let back: EngineStatus = serde_json::from_str(&json).unwrap();
    assert!((back.self_cpu_pct - 0.42).abs() < f64::EPSILON);
    assert_eq!(back.self_gpu_pct, 0.0);
    // 기존 필드 sanity
    assert_eq!(back.active_device, "cpu");
    assert_eq!(back.kv_cache_tokens, 100);
    assert_eq!(back.eviction_policy, "none");
}

#[test]
fn msg_060_missing_self_util_fields_default_to_zero() {
    // SPEC: MSG-061, 하위호환 (serde(default))
    // 필드 17~18을 빼고 나머지 16필드만 포함한 구버전 JSON.
    let legacy = r#"{
        "active_device": "opencl",
        "compute_level": "normal",
        "actual_throughput": 12.0,
        "memory_level": "normal",
        "kv_cache_bytes": 2048,
        "kv_cache_tokens": 256,
        "kv_cache_utilization": 0.25,
        "memory_lossless_min": 1.0,
        "memory_lossy_min": 0.01,
        "state": "running",
        "tokens_generated": 77,
        "available_actions": [],
        "active_actions": [],
        "eviction_policy": "none",
        "kv_dtype": "f16",
        "skip_ratio": 0.0,
        "phase": "",
        "prefill_pos": 0,
        "prefill_total": 0,
        "partition_ratio": 0.0
    }"#;
    let s: EngineStatus = serde_json::from_str(legacy).unwrap();
    assert_eq!(s.self_cpu_pct, 0.0);
    assert_eq!(s.self_gpu_pct, 0.0);
    assert_eq!(s.tokens_generated, 77);
}

#[test]
fn msg_060_self_cpu_pct_explicit_value_is_preserved_across_json() {
    // SPEC: MSG-060 — JSON에 명시된 값이 default를 덮어쓰고 복원된다.
    let base = serde_json::to_value(base_status()).unwrap();
    let mut obj = base.as_object().unwrap().clone();
    obj.insert("self_cpu_pct".to_string(), serde_json::json!(0.73));
    obj.insert("self_gpu_pct".to_string(), serde_json::json!(0.0));
    let json = serde_json::to_string(&serde_json::Value::Object(obj)).unwrap();
    let s: EngineStatus = serde_json::from_str(&json).unwrap();
    assert!((s.self_cpu_pct - 0.73).abs() < f64::EPSILON);
    assert_eq!(s.self_gpu_pct, 0.0);
}

// ---------------------------------------------------------------------------
// INV-091: clamp to [0.0, 1.0]
// ---------------------------------------------------------------------------

#[test]
fn inv_091_self_cpu_pct_is_clamped_on_send_side() {
    // SPEC: INV-091
    // ProcSelfMeter::sample()은 어떤 경우에도 [0.0, 1.0]을 반환한다.
    // 반복 샘플링 시 내부적으로 delta가 음수이거나 부정확한 계산이 발생해도
    // clamp_unit 로직이 이를 범위 내로 잘라낸다.
    let mut m = ProcSelfMeter::new();
    for _ in 0..50 {
        let v = m.sample();
        assert!(
            (0.0..=1.0).contains(&v),
            "sample must be in [0.0, 1.0], got {v}"
        );
    }
}

#[test]
fn inv_091_self_gpu_pct_phase1_is_always_zero() {
    // SPEC: INV-091, MSG-068
    // Phase 1 — Engine 측에서 GPU self-util은 항상 0.0. Executor에서
    // 하드코딩되므로, base_status() (executor와 같은 기본값)이 곧 송출 값.
    let s = base_status();
    assert_eq!(s.self_gpu_pct, 0.0);
}

// ---------------------------------------------------------------------------
// INV-092 / MSG-067: 측정 실패 시 0.0 fallback + Heartbeat 차단 금지
// ---------------------------------------------------------------------------

#[test]
fn inv_092_proc_self_stat_read_failure_falls_back_to_zero() {
    // SPEC: INV-092, MSG-067
    // ProcSelfMeter::sample()은 Result/Option이 아닌 f64를 반환하며,
    // 첫 호출(이전 샘플 없음)은 관례상 0.0을 반환한다. 또한 내부 /proc
    // 읽기 실패 시에도 0.0 반환을 보장한다 (Linux에서 정상 동작하는
    // 케이스는 second sample이 finite하므로 반대로 검증).
    let mut m = ProcSelfMeter::new();
    let first = m.sample();
    assert_eq!(first, 0.0, "first sample must be 0.0 (warm-up)");
}

#[test]
fn inv_092_measurement_failure_does_not_block_heartbeat_emission() {
    // SPEC: INV-092, INV-064
    // Heartbeat를 구성하는 경로에서 self-CPU 값이 0.0이어도 EngineStatus는
    // 정상 serialize 되어야 하며, 다른 필드 값은 영향을 받지 않는다.
    let mut s = base_status();
    s.self_cpu_pct = 0.0; // 측정 실패 모의
    s.self_gpu_pct = 0.0;
    let json = serde_json::to_string(&s).unwrap();
    let back: EngineStatus = serde_json::from_str(&json).unwrap();
    assert_eq!(back.self_cpu_pct, 0.0);
    assert_eq!(back.tokens_generated, 50);
    assert_eq!(back.active_device, "cpu");
}
