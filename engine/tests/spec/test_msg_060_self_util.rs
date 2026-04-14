//! MSG-060 필드 17~18 (self_cpu_pct, self_gpu_pct), MSG-067, MSG-068, MSG-069,
//! INV-091, INV-092: Engine Self-Utilization Heartbeat 필드 Spec 테스트.
//!
//! 2026-04 Phase 1 — Engine이 `/proc/self/stat` 기반 자신의 CPU 사용률을
//! Heartbeat(MSG-060)에 실어 보내는 경로를 검증한다. GPU는 Phase 1에서
//! placeholder(항상 0.0)이며 서지/의미는 Phase 2에서 재정의된다.
//!
//! 이 파일은 Architect가 생성한 스켈레톤이다. 본문은 Implementer가 채운다.
//! 필요한 경우 Implementer는 `shared::EngineStatus` 생성 헬퍼 또는
//! `engine::telemetry::self_cpu_probe` (가칭) 내부 타입을 테스트 전용으로
//! 재노출하여 테스트 가능성을 확보한다.
//!
//! 실행 조건: feature gate 없음 (기본 경로).

#![allow(clippy::needless_doctest_main)]

// ---------------------------------------------------------------------------
// MSG-060 #17~18 / MSG-061: serde round-trip + 하위호환 (#[serde(default)])
// ---------------------------------------------------------------------------

#[test]
fn msg_060_self_util_fields_roundtrip_preserves_values() {
    // SPEC: MSG-060 (필드 17, 18), MSG-061
    // GIVEN EngineStatus { self_cpu_pct = 0.42, self_gpu_pct = 0.0 } (기타 필드는 기본)
    // WHEN  serde_json::to_string → from_str 왕복
    // THEN  복원된 struct의 self_cpu_pct == 0.42 (epsilon 이내)
    // AND   self_gpu_pct == 0.0
    // AND   기타 기존 16필드도 손상되지 않음
    todo!("Implementer: 18필드 serde round-trip (정밀도 f64 epsilon)")
}

#[test]
fn msg_060_missing_self_util_fields_default_to_zero() {
    // SPEC: MSG-061, INV-028 (serde(default) 하위호환)
    // GIVEN 필드 17~18이 빠진 구버전 JSON (필드 12~16은 포함)
    // WHEN  serde_json::from_str::<EngineStatus>()
    // THEN  역직렬화 성공
    // AND   self_cpu_pct == 0.0, self_gpu_pct == 0.0 (serde default)
    todo!("Implementer: 구버전 JSON에 self_*_pct 누락 시 0.0 fallback")
}

#[test]
fn msg_060_self_cpu_pct_explicit_value_is_preserved_across_json() {
    // SPEC: MSG-060
    // GIVEN self_cpu_pct = 0.73, self_gpu_pct = 0.0 을 JSON에 명시
    // WHEN  from_str
    // THEN  두 값 모두 JSON에서 읽어온 값으로 복원 (default가 아님)
    todo!("Implementer: 명시 값 보존")
}

// ---------------------------------------------------------------------------
// INV-091: clamp to [0.0, 1.0]
// ---------------------------------------------------------------------------

#[test]
fn inv_091_self_cpu_pct_is_clamped_on_send_side() {
    // SPEC: INV-091
    // GIVEN Engine의 자가 CPU 측정 루틴이 일시적으로 1.3 (>1.0) 또는 -0.05 (<0.0)을 산출
    // WHEN  EngineStatus에 실어 송출 전 Engine 측 clamp 경로를 통과
    // THEN  송출 직전 값은 [0.0, 1.0] 범위 내
    //
    // 구현 노트: Implementer는 engine 측 probe 헬퍼를 public(crate) 혹은
    // test-only 경로로 노출하여 clamp 포인트를 단위 테스트한다.
    todo!("Implementer: Engine 측 clamp 경계(>1.0, <0.0) 검증")
}

#[test]
fn inv_091_self_gpu_pct_phase1_is_always_zero() {
    // SPEC: INV-091, MSG-068
    // WHEN  Engine이 heartbeat를 구성한다
    // THEN  self_gpu_pct == 0.0 (Phase 1 미구현 placeholder)
    todo!("Implementer: Phase 1에서 GPU 측정 함수가 호출되지 않거나 항상 0.0을 반환함을 확인")
}

// ---------------------------------------------------------------------------
// INV-092 / MSG-067: 측정 실패 시 0.0 fallback + Heartbeat 차단 금지
// ---------------------------------------------------------------------------

#[test]
fn inv_092_proc_self_stat_read_failure_falls_back_to_zero() {
    // SPEC: INV-092, MSG-067
    // GIVEN /proc/self/stat 읽기 또는 파싱이 실패하도록 주입 (permission/format)
    // WHEN  Engine heartbeat 샘플링 루틴을 호출
    // THEN  self_cpu_pct = 0.0 반환
    // AND   오류는 panic/propagate 없이 흡수된다
    todo!("Implementer: probe 함수가 Result/Option이 아닌 f64를 반환하며 실패 시 0.0")
}

#[test]
fn inv_092_measurement_failure_does_not_block_heartbeat_emission() {
    // SPEC: INV-092, INV-064 (heartbeat_interval 내 최소 1회 송출)
    // GIVEN self-CPU 측정 실패 상황
    // WHEN  heartbeat 송출 루프를 1 cycle 진행
    // THEN  Heartbeat가 정상적으로 송출된다 (self_cpu_pct만 0.0으로 실림)
    todo!("Implementer: mock transport로 Heartbeat 수신 확인")
}
