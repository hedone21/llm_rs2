//! MGR-DAT-075, MGR-DAT-076, MSG-069: LuaPolicy `ctx.engine` 하위의
//! Engine self-utilization 필드 노출 계약을 검증한다.
//!
//! 2026-04 Phase 1 — Engine이 Heartbeat로 전달한 `self_cpu_pct`/`self_gpu_pct`가
//! Manager 측에서 소실·재해석 없이 `ctx.engine.cpu_pct`/`ctx.engine.gpu_pct`로
//! Lua 평가 컨텍스트에 그대로 전달되는지 확인한다. 또한 Pressure6D나
//! EwmaReliefTable 계산에 섞여 들어가지 않음(영향 없음)을 확인한다.
//!
//! 이 파일은 Architect가 생성한 스켈레톤이다. 본문은 Implementer가 채운다.
//! 필요한 비공개 타입(`LuaPolicy::build_ctx`, `engine_cache`)은
//! `pub(crate)` 재노출 또는 `#[cfg(test)]` helper로 테스트 가능성을 확보한다.
//!
//! 실행 조건: LuaPolicy가 기본 경로이므로 feature gate 없음. 기존
//! `#[cfg(feature = "hierarchical")]` 테스트와 독립적으로 실행된다.

#![allow(clippy::needless_doctest_main)]

// ---------------------------------------------------------------------------
// MGR-DAT-075: ctx.engine.cpu_pct 노출
// ---------------------------------------------------------------------------

#[test]
fn mgr_dat_075_ctx_engine_cpu_pct_reflects_heartbeat() {
    // SPEC: MGR-DAT-075, MSG-069
    // GIVEN LuaPolicy에 self_cpu_pct = 0.42를 담은 mock EngineStatus heartbeat 주입
    // WHEN  build_ctx()로 Lua ctx 구성 후 Lua 스크립트로 `return ctx.engine.cpu_pct` 실행
    // THEN  반환된 값 ≈ 0.42 (f64 epsilon)
    todo!("Implementer: mock heartbeat → Lua에서 ctx.engine.cpu_pct 읽기")
}

#[test]
fn mgr_dat_075_ctx_engine_cpu_pct_is_zero_before_first_heartbeat() {
    // SPEC: MGR-DAT-075
    // GIVEN 신규 LuaPolicy, heartbeat 미수신 상태
    // WHEN  build_ctx()로 ctx 구성
    // THEN  ctx.engine.cpu_pct 는 0.0 또는 nil (설계: Implementer가 결정)
    //       Lua 스크립트가 숫자로 안전하게 취급할 수 있어야 함 (nil이면 tonumber 0 으로 치환 등)
    todo!("Implementer: 초기 상태 노출 정책 확정 + 검증")
}

// ---------------------------------------------------------------------------
// MGR-DAT-076: ctx.engine.gpu_pct Phase 1 placeholder
// ---------------------------------------------------------------------------

#[test]
fn mgr_dat_076_ctx_engine_gpu_pct_phase1_is_zero() {
    // SPEC: MGR-DAT-076, MSG-068
    // GIVEN self_gpu_pct = 0.0 (Phase 1 규약) 을 담은 mock heartbeat
    // WHEN  build_ctx() + Lua에서 ctx.engine.gpu_pct 읽기
    // THEN  반환값 == 0.0
    todo!("Implementer: Phase 1 gpu_pct placeholder 노출")
}

#[test]
fn mgr_dat_076_nonzero_self_gpu_pct_is_passed_through_untouched() {
    // SPEC: MGR-DAT-076 (Phase 2 forward compat)
    // GIVEN Engine이 (가정상) self_gpu_pct = 0.30 을 송출
    // WHEN  Manager가 그대로 ctx.engine.gpu_pct에 싣는지 확인
    // THEN  ctx.engine.gpu_pct ≈ 0.30
    // 의도: Manager 측은 Phase 1에서도 값 가공/제로잉을 하지 않음을 보장
    //       (Phase 2 배선 시 shape 변경 없이 값이 통과되도록)
    todo!("Implementer: Manager는 self_gpu_pct를 해석/변형하지 않고 전달")
}

// ---------------------------------------------------------------------------
// MSG-069: ctx.signal.compute.cpu_pct vs ctx.engine.cpu_pct 분리
// ---------------------------------------------------------------------------

#[test]
fn msg_069_ctx_signal_and_ctx_engine_cpu_are_independent() {
    // SPEC: MSG-069
    // GIVEN ComputeGuidance system cpu_pct = 80.0 (→ ctx.signal.compute.cpu_pct = 80.0)
    // AND   EngineStatus.self_cpu_pct = 0.25
    // WHEN  build_ctx() 후 Lua 스크립트 실행
    // THEN  ctx.signal.compute.cpu_pct == 80.0 (원본 스케일 유지: 0~100)
    // AND   ctx.engine.cpu_pct == 0.25 (0~1 스케일)
    // AND   두 값은 서로 간섭하지 않는다 (한 쪽 업데이트가 다른 쪽을 바꾸지 않음)
    todo!("Implementer: 스케일 차이(0~100 vs 0~1)와 독립성을 모두 검증")
}

#[test]
fn msg_069_lua_can_compute_external_contention_from_raw_values() {
    // SPEC: MSG-069 (non-normative 예시, arch/20-manager.md §10.7)
    // GIVEN ctx.signal.compute.cpu_pct = 80.0, ctx.engine.cpu_pct = 0.30
    // WHEN  Lua 스크립트가 external = (ctx.signal.compute.cpu_pct / 100) - ctx.engine.cpu_pct 계산
    // THEN  external ≈ 0.50
    // 의도: Rust 측이 gap을 계산하지 않고 raw 두 값을 그대로 제공함을 회귀 방지
    todo!("Implementer: Lua 스크립트 실행 후 결과 대조")
}

// ---------------------------------------------------------------------------
// Pressure6D / relief 학습 경로 비침투 (regression guard)
// ---------------------------------------------------------------------------

#[test]
fn mgr_dat_075_self_cpu_pct_does_not_leak_into_pressure6d() {
    // SPEC: MGR-DAT-075 (Pressure6D 영향 없음)
    // GIVEN 동일한 ComputeGuidance/MemoryPressure/ThermalAlert 시퀀스
    // AND   시나리오 A: self_cpu_pct = 0.0
    // AND   시나리오 B: self_cpu_pct = 0.9
    // WHEN  pressure_with_thermal()로 Pressure6D 산출
    // THEN  A와 B의 Pressure6D가 동일해야 한다 (self_cpu_pct는 Pressure6D에 쓰이지 않음)
    todo!("Implementer: 두 시나리오의 Pressure6D 동등성")
}

#[test]
fn mgr_dat_076_self_gpu_pct_does_not_leak_into_relief_observations() {
    // SPEC: MGR-DAT-076
    // GIVEN observe() 경로에 before/after heartbeat 주입 (self_gpu_pct 값만 다름)
    // WHEN  EwmaReliefTable.observe 호출 후 entries["action"] 비교
    // THEN  self_gpu_pct 차이가 relief 6D 벡터에 영향을 주지 않는다
    //       (관측은 Pressure6D만 사용 — 차원 0 gpu는 ctx.signal 기반)
    todo!("Implementer: relief 6D에 engine self-util이 섞이지 않음 확인")
}
