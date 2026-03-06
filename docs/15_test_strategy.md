# Test Strategy — Resilience System

> Phase 0 설계 문서 | D-Bus Resilience 시스템 테스트 전략

## 1. Overview

Resilience 시스템은 외부 시스템(D-Bus, Manager)에 의존하지만,
테스트는 **외부 의존성 없이** 호스트에서 실행 가능해야 한다.

### 1.1 테스트 원칙

- **D-Bus 없이 테스트**: 모든 로직은 `mpsc::channel`을 통해 주입 가능. D-Bus는 전송 계층일 뿐
- **기존 3-Tier 확장**: llm.rs의 Host Unit / Backend Verification / E2E 구조를 유지하며 Resilience tier 추가
- **결정적 재현**: 시그널 시퀀스를 파일로 저장하고 재생하여 동일 결과 보장
- **장애 주입**: 정상 시나리오뿐 아니라 경계 조건, 빠른 전환, 채널 끊김 등을 테스트

### 1.2 확장된 테스트 Tier

| Tier | 환경 | 대상 | 도구 |
|------|------|------|------|
| **T1. Unit** | 호스트 (cargo test) | Strategy, State Machine, Conflict Resolution | mock channel |
| **T2. Integration** | 호스트 (cargo test) | ResilienceManager 전체 파이프라인 | MockSignalSource |
| **T3. D-Bus** | Linux 호스트 (D-Bus 있는 환경) | DbusListener ↔ Manager 연동 | mock-manager 바이너리 |
| **T4. System** | 임베디드 디바이스 | 추론 + Resilience E2E | mock-manager + generate |

---

## 2. T1: Unit Tests

각 컴포넌트를 독립적으로 검증. `#[cfg(test)] mod tests` 내에 작성.

### 2.1 Signal Types (signal.rs)

```rust
#[test]
fn test_level_ordering() {
    // Level은 Ord를 구현. 심각도 순서가 올바른지 검증.
    assert!(Level::Normal < Level::Warning);
    assert!(Level::Warning < Level::Critical);
    assert!(Level::Critical < Level::Emergency);
}

#[test]
fn test_level_max_returns_worst() {
    assert_eq!(Level::Normal.max(Level::Critical), Level::Critical);
    assert_eq!(Level::Emergency.max(Level::Warning), Level::Emergency);
}
```

### 2.2 OperatingMode State Machine (state.rs)

```rust
#[test]
fn test_all_normal_yields_normal_mode() {
    let mode = OperatingMode::from_levels(
        Level::Normal, Level::Normal, Level::Normal, Level::Normal,
    );
    assert_eq!(mode, OperatingMode::Normal);
}

#[test]
fn test_single_warning_yields_degraded() {
    // 4개 중 하나라도 warning이면 Degraded
    let mode = OperatingMode::from_levels(
        Level::Normal, Level::Warning, Level::Normal, Level::Normal,
    );
    assert_eq!(mode, OperatingMode::Degraded);
}

#[test]
fn test_single_critical_yields_minimal() {
    let mode = OperatingMode::from_levels(
        Level::Normal, Level::Normal, Level::Critical, Level::Normal,
    );
    assert_eq!(mode, OperatingMode::Minimal);
}

#[test]
fn test_any_emergency_yields_suspended() {
    // 나머지가 Normal이어도 하나가 Emergency면 Suspended
    let mode = OperatingMode::from_levels(
        Level::Normal, Level::Normal, Level::Normal, Level::Emergency,
    );
    assert_eq!(mode, OperatingMode::Suspended);
}

#[test]
fn test_mixed_levels_worst_wins() {
    let mode = OperatingMode::from_levels(
        Level::Warning, Level::Critical, Level::Normal, Level::Warning,
    );
    assert_eq!(mode, OperatingMode::Minimal);  // Critical이 가장 심각
}
```

### 2.3 Individual Strategies

각 Strategy는 시그널 입력과 반환 액션의 매핑을 검증.

#### MemoryStrategy

```rust
#[test]
fn test_memory_normal_restores_defaults() {
    let mut s = MemoryStrategy::new();
    let actions = s.react(&SystemSignal::MemoryPressure {
        level: Level::Normal,
        available_bytes: 2_000_000_000,
        reclaim_target_bytes: 0,
    }, OperatingMode::Normal);
    assert!(actions.iter().any(|a| matches!(a, ResilienceAction::RestoreDefaults)));
}

#[test]
fn test_memory_critical_triggers_eviction() {
    let mut s = MemoryStrategy::new();
    let actions = s.react(&SystemSignal::MemoryPressure {
        level: Level::Critical,
        available_bytes: 100_000_000,
        reclaim_target_bytes: 50_000_000,
    }, OperatingMode::Minimal);
    assert!(actions.iter().any(|a| matches!(a, ResilienceAction::Evict { target_ratio } if *target_ratio <= 0.5)));
}

#[test]
fn test_memory_emergency_evicts_and_rejects() {
    let mut s = MemoryStrategy::new();
    let actions = s.react(&SystemSignal::MemoryPressure {
        level: Level::Emergency,
        available_bytes: 10_000_000,
        reclaim_target_bytes: 200_000_000,
    }, OperatingMode::Suspended);
    assert!(actions.iter().any(|a| matches!(a, ResilienceAction::Evict { .. })));
    assert!(actions.iter().any(|a| matches!(a, ResilienceAction::RejectNew)));
}
```

#### ComputeStrategy

```rust
#[test]
fn test_compute_critical_switches_backend() {
    let mut s = ComputeStrategy::new();
    let actions = s.react(&SystemSignal::ComputeGuidance {
        level: Level::Critical,
        recommended_backend: RecommendedBackend::Gpu,
        reason: ComputeReason::CpuBottleneck,
        cpu_usage_pct: 95.0,
        gpu_usage_pct: 10.0,
    }, OperatingMode::Minimal);
    assert!(actions.iter().any(|a| matches!(a, ResilienceAction::SwitchBackend { to: RecommendedBackend::Gpu })));
}

#[test]
fn test_compute_warning_does_not_switch() {
    let mut s = ComputeStrategy::new();
    let actions = s.react(&SystemSignal::ComputeGuidance {
        level: Level::Warning,
        recommended_backend: RecommendedBackend::Gpu,
        reason: ComputeReason::CpuBottleneck,
        cpu_usage_pct: 78.0,
        gpu_usage_pct: 20.0,
    }, OperatingMode::Degraded);
    // Warning에서는 준비만. 전환 액션 없음.
    assert!(!actions.iter().any(|a| matches!(a, ResilienceAction::SwitchBackend { .. })));
}
```

#### ThermalStrategy

```rust
#[test]
fn test_thermal_emergency_suspends() {
    let mut s = ThermalStrategy::new();
    let actions = s.react(&SystemSignal::ThermalAlert {
        level: Level::Emergency,
        temperature_mc: 92000,
        throttling_active: true,
        throttle_ratio: 0.3,
    }, OperatingMode::Suspended);
    assert!(actions.iter().any(|a| matches!(a, ResilienceAction::Suspend)));
}

#[test]
fn test_thermal_critical_throttles_proportionally() {
    let mut s = ThermalStrategy::new();
    let actions = s.react(&SystemSignal::ThermalAlert {
        level: Level::Critical,
        temperature_mc: 82000,
        throttling_active: true,
        throttle_ratio: 0.7,
    }, OperatingMode::Minimal);
    // throttle_ratio 0.7 → delay = (1.0 - 0.7) * 100 = 30ms
    assert!(actions.iter().any(|a| matches!(a, ResilienceAction::Throttle { delay_ms: 30 })));
}
```

#### EnergyStrategy

```rust
#[test]
fn test_energy_charging_restores() {
    let mut s = EnergyStrategy::new();
    let actions = s.react(&SystemSignal::EnergyConstraint {
        level: Level::Normal,
        reason: EnergyReason::Charging,
        power_budget_mw: 0,
    }, OperatingMode::Normal);
    assert!(actions.iter().any(|a| matches!(a, ResilienceAction::RestoreDefaults)));
}
```

### 2.4 Conflict Resolution

```rust
#[test]
fn test_cpu_always_wins_over_gpu() {
    let actions = vec![
        ResilienceAction::SwitchBackend { to: RecommendedBackend::Gpu },
        ResilienceAction::SwitchBackend { to: RecommendedBackend::Cpu },
    ];
    let resolved = resolve_conflicts(actions);
    assert_eq!(resolved.len(), 1);
    assert!(matches!(resolved[0], ResilienceAction::SwitchBackend { to: RecommendedBackend::Cpu }));
}

#[test]
fn test_most_aggressive_eviction_wins() {
    let actions = vec![
        ResilienceAction::Evict { target_ratio: 0.85 },
        ResilienceAction::Evict { target_ratio: 0.50 },
        ResilienceAction::Evict { target_ratio: 0.25 },
    ];
    let resolved = resolve_conflicts(actions);
    assert!(matches!(resolved[0], ResilienceAction::Evict { target_ratio } if target_ratio == 0.25));
}

#[test]
fn test_largest_delay_wins() {
    let actions = vec![
        ResilienceAction::Throttle { delay_ms: 30 },
        ResilienceAction::Throttle { delay_ms: 50 },
    ];
    let resolved = resolve_conflicts(actions);
    assert!(matches!(resolved[0], ResilienceAction::Throttle { delay_ms: 50 }));
}

#[test]
fn test_suspend_overrides_all() {
    let actions = vec![
        ResilienceAction::Evict { target_ratio: 0.5 },
        ResilienceAction::SwitchBackend { to: RecommendedBackend::Cpu },
        ResilienceAction::Suspend,
    ];
    let resolved = resolve_conflicts(actions);
    assert_eq!(resolved.len(), 1);
    assert!(matches!(resolved[0], ResilienceAction::Suspend));
}

#[test]
fn test_restore_only_when_no_other_constraints() {
    // RestoreDefaults + 다른 제약 → RestoreDefaults 무시
    let actions = vec![
        ResilienceAction::RestoreDefaults,
        ResilienceAction::Throttle { delay_ms: 30 },
    ];
    let resolved = resolve_conflicts(actions);
    assert!(!resolved.iter().any(|a| matches!(a, ResilienceAction::RestoreDefaults)));
    assert!(resolved.iter().any(|a| matches!(a, ResilienceAction::Throttle { .. })));
}

#[test]
fn test_restore_alone_passes_through() {
    let actions = vec![ResilienceAction::RestoreDefaults];
    let resolved = resolve_conflicts(actions);
    assert_eq!(resolved.len(), 1);
    assert!(matches!(resolved[0], ResilienceAction::RestoreDefaults));
}
```

### 2.5 Action Executor

```rust
#[test]
fn test_execute_evict_reduces_cache_pos() {
    let mut caches = make_test_caches(4, 100);  // 4 layers, pos=100
    let mut ctx = make_test_context(&mut caches);
    execute_action(ResilienceAction::Evict { target_ratio: 0.5 }, &mut ctx);
    assert_eq!(caches[0].current_pos, 50);
}

#[test]
fn test_execute_suspend_sets_flag() {
    let mut caches = make_test_caches(1, 10);
    let mut ctx = make_test_context(&mut caches);
    assert!(!*ctx.suspended);
    execute_action(ResilienceAction::Suspend, &mut ctx);
    assert!(*ctx.suspended);
}

#[test]
fn test_execute_restore_clears_constraints() {
    let mut caches = make_test_caches(1, 10);
    let mut ctx = make_test_context(&mut caches);
    *ctx.throttle_delay_ms = 50;
    *ctx.reject_new = true;
    execute_action(ResilienceAction::RestoreDefaults, &mut ctx);
    assert_eq!(*ctx.throttle_delay_ms, 0);
    assert!(!*ctx.reject_new);
}
```

---

## 3. T2: Integration Tests

ResilienceManager 전체 파이프라인을 `mpsc::channel`로 검증.

### 3.1 기본 시그널 → 액션 파이프라인

```rust
#[test]
fn test_manager_processes_memory_signal() {
    let (tx, rx) = mpsc::channel();
    let mut manager = ResilienceManager::new(rx);

    tx.send(SystemSignal::MemoryPressure {
        level: Level::Critical,
        available_bytes: 100_000_000,
        reclaim_target_bytes: 50_000_000,
    }).unwrap();

    let actions = manager.poll();
    assert!(!actions.is_empty());
    assert_eq!(manager.mode(), OperatingMode::Minimal);
    assert!(actions.iter().any(|a| matches!(a, ResilienceAction::Evict { .. })));
}
```

### 3.2 복수 시그널 동시 도착

```rust
#[test]
fn test_manager_handles_multiple_signals() {
    let (tx, rx) = mpsc::channel();
    let mut manager = ResilienceManager::new(rx);

    // 2개 시그널 동시 전송
    tx.send(SystemSignal::ThermalAlert {
        level: Level::Critical,
        temperature_mc: 82000,
        throttling_active: true,
        throttle_ratio: 0.7,
    }).unwrap();
    tx.send(SystemSignal::ComputeGuidance {
        level: Level::Critical,
        recommended_backend: RecommendedBackend::Gpu,
        reason: ComputeReason::CpuBottleneck,
        cpu_usage_pct: 95.0,
        gpu_usage_pct: 10.0,
    }).unwrap();

    let actions = manager.poll();

    // Thermal은 CPU 요구, Compute는 GPU 권장 → 충돌 해소: CPU 승리
    assert!(actions.iter().any(|a| matches!(
        a, ResilienceAction::SwitchBackend { to: RecommendedBackend::Cpu }
    )));
}
```

### 3.3 상태 전이 시퀀스

```rust
#[test]
fn test_manager_state_transitions() {
    let (tx, rx) = mpsc::channel();
    let mut manager = ResilienceManager::new(rx);
    assert_eq!(manager.mode(), OperatingMode::Normal);

    // Normal → Degraded
    tx.send(SystemSignal::EnergyConstraint {
        level: Level::Warning,
        reason: EnergyReason::BatteryLow,
        power_budget_mw: 3000,
    }).unwrap();
    manager.poll();
    assert_eq!(manager.mode(), OperatingMode::Degraded);

    // Degraded → Minimal
    tx.send(SystemSignal::ThermalAlert {
        level: Level::Critical,
        temperature_mc: 83000,
        throttling_active: true,
        throttle_ratio: 0.6,
    }).unwrap();
    manager.poll();
    assert_eq!(manager.mode(), OperatingMode::Minimal);

    // Minimal → Suspended
    tx.send(SystemSignal::MemoryPressure {
        level: Level::Emergency,
        available_bytes: 5_000_000,
        reclaim_target_bytes: 500_000_000,
    }).unwrap();
    manager.poll();
    assert_eq!(manager.mode(), OperatingMode::Suspended);

    // Suspended → Normal (모든 시그널 normal 복귀)
    tx.send(SystemSignal::MemoryPressure {
        level: Level::Normal, available_bytes: 2_000_000_000, reclaim_target_bytes: 0,
    }).unwrap();
    tx.send(SystemSignal::ThermalAlert {
        level: Level::Normal, temperature_mc: 55000, throttling_active: false, throttle_ratio: 1.0,
    }).unwrap();
    tx.send(SystemSignal::EnergyConstraint {
        level: Level::Normal, reason: EnergyReason::Charging, power_budget_mw: 0,
    }).unwrap();
    manager.poll();
    assert_eq!(manager.mode(), OperatingMode::Normal);
}
```

### 3.4 채널 끊김 (Manager 종료)

```rust
#[test]
fn test_manager_survives_channel_disconnect() {
    let (tx, rx) = mpsc::channel();
    let mut manager = ResilienceManager::new(rx);

    // 시그널 하나 보내고 sender를 drop
    tx.send(SystemSignal::ThermalAlert {
        level: Level::Warning,
        temperature_mc: 70000,
        throttling_active: false,
        throttle_ratio: 1.0,
    }).unwrap();
    drop(tx);  // 채널 끊김

    // 첫 poll: 시그널 처리
    let actions = manager.poll();
    assert!(!actions.is_empty());

    // 이후 poll: 채널 끊김이지만 panic 없이 빈 결과
    let actions = manager.poll();
    assert!(actions.is_empty());
    // 마지막 상태 유지
    assert_eq!(manager.mode(), OperatingMode::Degraded);
}
```

### 3.5 빈 poll (시그널 없음)

```rust
#[test]
fn test_manager_poll_returns_empty_when_no_signals() {
    let (_tx, rx) = mpsc::channel();
    let mut manager = ResilienceManager::new(rx);

    let actions = manager.poll();
    assert!(actions.is_empty());
    assert_eq!(manager.mode(), OperatingMode::Normal);
}
```

---

## 4. T3: D-Bus Integration Tests

실제 D-Bus가 있는 Linux 호스트에서 DbusListener와 mock Manager의 연동을 검증.

### 4.1 mock-manager 바이너리

테스트용 Manager 역할을 하는 간단한 D-Bus 시그널 발행기.

```
src/bin/mock_manager.rs
```

기능:
- System Bus에 `org.llm.Manager1`로 등록
- CLI 인자 또는 시나리오 파일로 시그널 시퀀스 정의
- 지정된 간격으로 시그널 발행

```bash
# 직접 시그널 발행
mock-manager --signal MemoryPressure --level critical --available 100000000 --reclaim 50000000

# 시나리오 파일 재생
mock-manager --scenario scenarios/thermal_spike.json
```

### 4.2 시나리오 파일 형식

```json
{
  "name": "thermal_spike",
  "description": "온도 급상승 후 회복 시나리오",
  "signals": [
    {
      "delay_ms": 0,
      "signal": "ThermalAlert",
      "level": "warning",
      "temperature_mc": 72000,
      "throttling_active": false,
      "throttle_ratio": 1.0
    },
    {
      "delay_ms": 2000,
      "signal": "ThermalAlert",
      "level": "critical",
      "temperature_mc": 84000,
      "throttling_active": true,
      "throttle_ratio": 0.6
    },
    {
      "delay_ms": 5000,
      "signal": "ThermalAlert",
      "level": "warning",
      "temperature_mc": 73000,
      "throttling_active": false,
      "throttle_ratio": 1.0
    },
    {
      "delay_ms": 3000,
      "signal": "ThermalAlert",
      "level": "normal",
      "temperature_mc": 58000,
      "throttling_active": false,
      "throttle_ratio": 1.0
    }
  ]
}
```

### 4.3 D-Bus round-trip 검증

```bash
# 터미널 1: mock-manager 실행
mock-manager --scenario scenarios/thermal_spike.json

# 터미널 2: LLM 추론 + resilience
generate --prompt "Hello" -n 256 --resilience

# 검증: LLM 로그에서 resilience 액션 확인
# [Resilience] ThermalAlert: warning → mode: Degraded
# [Resilience] ThermalAlert: critical → mode: Minimal, action: SwitchBackend(Cpu)
# [Resilience] ThermalAlert: normal → mode: Normal, action: RestoreDefaults
```

### 4.4 D-Bus 연결 불가 테스트

```bash
# D-Bus 없이 실행 → resilience_manager = None, 정상 추론
generate --prompt "Hello" -n 128 --resilience
# 기대: 경고 로그 후 정상 동작
# [WARN] D-Bus listener exited: ... LLM continues without resilience.
```

---

## 5. T4: System Tests (On-device)

임베디드 디바이스에서 실제 추론과 resilience를 함께 검증.

### 5.1 테스트 매트릭스

| 시나리오 | 시그널 시퀀스 | 기대 동작 | 검증 방법 |
|----------|-------------|----------|----------|
| 정상 추론 | 시그널 없음 | 기존과 동일한 출력 | 토큰 비교 |
| 메모리 압박 | MemoryPressure critical | KV eviction 발생, 추론 계속 | 로그 + 토큰 수 확인 |
| 쓰로틀링 | ThermalAlert critical | GPU→CPU 전환, 속도 저하 | 백엔드 로그 + TPS 측정 |
| 배터리 부족 | EnergyConstraint critical | CPU only + 토큰 제한 | 백엔드 로그 + 토큰 수 |
| 비상 중단 | ThermalAlert emergency | 추론 즉시 중단 | 종료 확인 + 부분 출력 |
| 회복 | critical → normal | 제약 해제, 정상 복원 | TPS 회복 확인 |
| 다중 이벤트 | Memory + Thermal 동시 | 충돌 해소 후 안전 동작 | 액션 로그 |
| Manager 크래시 | 시그널 도중 Manager 종료 | 추론 계속 (graceful) | 추론 완료 확인 |

### 5.2 자동화 스크립트

```bash
#!/bin/bash
# scripts/test_resilience.sh
# 시나리오별 자동 실행 및 결과 수집

SCENARIOS_DIR="scenarios/"
RESULTS_DIR="results/data/resilience/"
mkdir -p "$RESULTS_DIR"

for scenario in "$SCENARIOS_DIR"/*.json; do
    name=$(basename "$scenario" .json)
    echo "=== Running scenario: $name ==="

    # mock-manager 백그라운드 실행
    mock-manager --scenario "$scenario" &
    MANAGER_PID=$!

    # LLM 추론
    generate --prompt "The quick brown fox" -n 128 --resilience \
        2>&1 | tee "$RESULTS_DIR/${name}.log"

    # mock-manager 종료
    kill $MANAGER_PID 2>/dev/null
    wait $MANAGER_PID 2>/dev/null

    echo "=== Completed: $name ==="
done
```

### 5.3 성능 회귀 기준

| 항목 | 기준 |
|------|------|
| Resilience 비활성 시 오버헤드 | < 1% TPS 감소 (try_recv 비용) |
| 시그널 수신~액션 실행 지연 | < 10ms (채널 전달 + poll) |
| 백엔드 전환 시간 | < 500ms (KV 마이그레이션 포함) |
| Eviction 후 추론 정확성 | 기존 eviction 테스트와 동일 기준 |

---

## 6. 사전 정의 시나리오

### 6.1 시나리오 목록

| 파일 | 시나리오 | 시그널 수 | 총 시간 |
|------|---------|---------|---------|
| `normal_operation.json` | 시그널 없음 (baseline) | 0 | 0s |
| `memory_gradual.json` | 메모리 점진적 감소 | 4 | 10s |
| `memory_sudden_oom.json` | 갑작스런 OOM 직전 | 2 | 1s |
| `thermal_spike.json` | 온도 급상승 후 회복 | 4 | 10s |
| `thermal_sustained.json` | 장시간 고온 유지 | 6 | 30s |
| `battery_drain.json` | 배터리 점진적 방전 | 4 | 15s |
| `battery_charging.json` | 방전 후 충전 시작 | 5 | 10s |
| `cpu_bottleneck.json` | CPU 과부하 → GPU 권장 | 3 | 5s |
| `gpu_bottleneck.json` | GPU 과부하 → CPU 권장 | 3 | 5s |
| `multi_event_storm.json` | Memory + Thermal + Energy 동시 | 8 | 5s |
| `rapid_oscillation.json` | 빠른 level 전환 (히스테리시스 테스트) | 12 | 6s |
| `manager_crash.json` | 시그널 도중 Manager 중단 | 3 + crash | 5s |
| `full_lifecycle.json` | Normal → 모든 단계 → Normal 복귀 | 16 | 60s |

### 6.2 시나리오 예시: multi_event_storm.json

```json
{
  "name": "multi_event_storm",
  "description": "Memory, Thermal, Energy 시그널이 짧은 간격으로 동시 도착",
  "signals": [
    { "delay_ms": 0,    "signal": "MemoryPressure",   "level": "warning",   "available_bytes": 800000000,  "reclaim_target_bytes": 100000000 },
    { "delay_ms": 200,  "signal": "ThermalAlert",     "level": "warning",   "temperature_mc": 72000, "throttling_active": false, "throttle_ratio": 1.0 },
    { "delay_ms": 300,  "signal": "EnergyConstraint",  "level": "warning",   "reason": "battery_low", "power_budget_mw": 3000 },
    { "delay_ms": 500,  "signal": "MemoryPressure",   "level": "critical",  "available_bytes": 200000000,  "reclaim_target_bytes": 400000000 },
    { "delay_ms": 200,  "signal": "ThermalAlert",     "level": "critical",  "temperature_mc": 83000, "throttling_active": true, "throttle_ratio": 0.6 },
    { "delay_ms": 2000, "signal": "MemoryPressure",   "level": "normal",    "available_bytes": 2000000000, "reclaim_target_bytes": 0 },
    { "delay_ms": 1000, "signal": "ThermalAlert",     "level": "normal",    "temperature_mc": 58000, "throttling_active": false, "throttle_ratio": 1.0 },
    { "delay_ms": 500,  "signal": "EnergyConstraint",  "level": "normal",    "reason": "charging", "power_budget_mw": 0 }
  ]
}
```

---

## 7. 검증 기준 정리

### 7.1 정확성 (Correctness)

| 검증 항목 | 기준 | Tier |
|----------|------|------|
| Level 순서 | Normal < Warning < Critical < Emergency | T1 |
| 상태 전이 | worst level → 해당 mode | T1 |
| Strategy 반응 | level별 올바른 action 반환 | T1 |
| 충돌 해소 | CPU > GPU, min ratio, max delay, Suspend 우선 | T1 |
| 파이프라인 | signal → level 갱신 → mode 갱신 → strategy → action | T2 |
| D-Bus 수신 | Manager 시그널 → SystemSignal 변환 정확성 | T3 |
| E2E 동작 | 시나리오별 기대 동작 일치 | T4 |

### 7.2 안정성 (Robustness)

| 검증 항목 | 기준 | Tier |
|----------|------|------|
| 채널 끊김 | panic 없이 마지막 상태 유지 | T2 |
| D-Bus 연결 실패 | 경고 후 추론 정상 동작 | T3 |
| 빈 poll | 빈 Vec 반환, 상태 변경 없음 | T2 |
| 빠른 시그널 연속 | 모든 시그널 처리, 누락 없음 | T2, T4 |
| Manager 재시작 | 재연결 후 시그널 수신 재개 | T3 |

### 7.3 성능 (Performance)

| 검증 항목 | 기준 | Tier |
|----------|------|------|
| poll() 오버헤드 | < 1% TPS 감소 | T4 |
| 시그널 → 액션 지연 | < 10ms | T4 |
| 컴파일 크기 | resilience feature off 시 바이너리 증가 0 | T1 |

---

## 8. CI 통합

### 8.1 Host CI (T1 + T2)

```bash
# 기존 cargo test에 resilience 모듈 테스트 포함
cargo test                          # 전체 (resilience 포함)
cargo test resilience               # resilience 모듈만
cargo test resilience::strategy     # strategy만
cargo test resilience::manager      # manager 통합만
```

### 8.2 Feature 분리 확인

```bash
# resilience feature 없이 빌드 — 기존 기능 정상 확인
cargo build --no-default-features --features opencl
cargo test --no-default-features --features opencl

# resilience feature 포함 빌드
cargo build --features resilience
cargo test --features resilience
```

### 8.3 Lint / Format

```bash
# 기존 sanity_check.sh에 포함
./.agent/skills/developing/scripts/sanity_check.sh
```

---

## 9. 테스트 구현 우선순위

Phase 1 (기반)에서 작성하는 테스트:

| 순서 | 대상 | 테스트 수 (예상) | 의존성 |
|------|------|-----------------|--------|
| 1 | Level, SystemSignal 타입 | 5 | 없음 |
| 2 | OperatingMode 상태 머신 | 6 | signal.rs |
| 3 | resolve_conflicts() | 7 | signal.rs |
| 4 | MemoryStrategy | 4 | signal.rs, strategy trait |
| 5 | ComputeStrategy | 4 | signal.rs, strategy trait |
| 6 | ThermalStrategy | 4 | signal.rs, strategy trait |
| 7 | EnergyStrategy | 3 | signal.rs, strategy trait |
| 8 | ResilienceManager 통합 | 6 | 위 전체 |
| 9 | Action Executor | 4 | signal.rs |
| **합계** | | **~43** | |

이 43개 테스트는 D-Bus 없이 `cargo test`로 호스트에서 실행 가능.
