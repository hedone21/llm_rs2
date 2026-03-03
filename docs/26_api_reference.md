# 26. Resilience API 레퍼런스

> llm.rs Resilience 시스템의 모든 공개 타입, 트레이트, 함수에 대한 빠른 참조 문서.

---

[이전: 24. Resilience 사용 가이드](./24_resilience_usage_guide.md)

---

## Quick Navigation

| 섹션 | 주요 타입 | 파일 |
|------|----------|------|
| [1. Signal Types](#1-signal-types) | `Level`, `SystemSignal`, `RecommendedBackend`, `ComputeReason`, `EnergyReason` | `src/resilience/signal.rs` |
| [2. State Management](#2-state-management) | `OperatingMode` | `src/resilience/state.rs` |
| [3. Strategy Framework](#3-strategy-framework) | `ResilienceAction`, `ResilienceStrategy`, `resolve_conflicts` | `src/resilience/strategy/mod.rs` |
| [4. Strategy Implementations](#4-strategy-implementations) | `MemoryStrategy`, `ComputeStrategy`, `ThermalStrategy`, `EnergyStrategy` | `src/resilience/strategy/*.rs` |
| [5. Manager & Execution](#5-manager--execution) | `ResilienceManager`, `DbusListener`, `InferenceContext`, `execute_action` | `src/resilience/manager.rs`, `dbus_listener.rs` |
| [6. CLI Integration](#6-cli-integration) | `--enable-resilience`, 초기화, 토큰 루프 체크포인트 | `src/bin/generate.rs` |

---

## 1. Signal Types

### 1.1 `Level`

심각도 수준. 모든 시그널이 공유하며, `Ord`를 derive하여 비교 가능.

**파일**: `src/resilience/signal.rs`

**정의**:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    Normal,
    Warning,
    Critical,
    Emergency,
}
```

**Variants**:

| Variant | 순서 | 의미 | D-Bus 문자열 |
|---------|------|------|-------------|
| `Normal` | 0 (최저) | 정상 상태 | `"normal"` |
| `Warning` | 1 | 경고 — 경미한 제약 | `"warning"` |
| `Critical` | 2 | 위험 — 공격적 제약 | `"critical"` |
| `Emergency` | 3 (최고) | 긴급 — 추론 일시중단 | `"emergency"` |

**Methods**:

| 메서드 | 시그니처 | 설명 |
|--------|---------|------|
| `from_dbus_str` | `fn from_dbus_str(s: &str) -> Option<Self>` | D-Bus 문자열을 `Level`로 변환. 대소문자 구분. 알 수 없는 값은 `None` 반환 |
| `max` (derived) | `fn max(self, other: Self) -> Self` | `Ord` trait으로 두 레벨 중 더 심각한 쪽 반환 |

**예시**:
```rust
use llm_rs2::resilience::Level;

assert!(Level::Normal < Level::Emergency);
assert_eq!(Level::from_dbus_str("critical"), Some(Level::Critical));
assert_eq!(Level::from_dbus_str("unknown"), None);
assert_eq!(Level::Warning.max(Level::Critical), Level::Critical);
```

---

### 1.2 `SystemSignal`

D-Bus Manager(`org.llm.Manager1`)로부터 수신하는 시스템 시그널. 4가지 카테고리.

**파일**: `src/resilience/signal.rs`

**정의**:
```rust
#[derive(Debug, Clone)]
pub enum SystemSignal {
    MemoryPressure {
        level: Level,
        available_bytes: u64,
        reclaim_target_bytes: u64,
    },
    ComputeGuidance {
        level: Level,
        recommended_backend: RecommendedBackend,
        reason: ComputeReason,
        cpu_usage_pct: f64,
        gpu_usage_pct: f64,
    },
    ThermalAlert {
        level: Level,
        temperature_mc: i32,
        throttling_active: bool,
        throttle_ratio: f64,
    },
    EnergyConstraint {
        level: Level,
        reason: EnergyReason,
        power_budget_mw: u32,
    },
}
```

**Variants / Fields**:

| Variant | 필드 | 타입 | 설명 |
|---------|------|------|------|
| `MemoryPressure` | `level` | `Level` | 메모리 압박 심각도 |
| | `available_bytes` | `u64` | 사용 가능한 메모리 (bytes) |
| | `reclaim_target_bytes` | `u64` | 회수 목표 (bytes) |
| `ComputeGuidance` | `level` | `Level` | 연산 자원 심각도 |
| | `recommended_backend` | `RecommendedBackend` | 권장 백엔드 |
| | `reason` | `ComputeReason` | 권장 사유 |
| | `cpu_usage_pct` | `f64` | CPU 사용률 (%) |
| | `gpu_usage_pct` | `f64` | GPU 사용률 (%) |
| `ThermalAlert` | `level` | `Level` | 열 경고 심각도 |
| | `temperature_mc` | `i32` | 온도 (밀리섭씨, e.g. 85000 = 85°C) |
| | `throttling_active` | `bool` | 쓰로틀링 활성 여부 |
| | `throttle_ratio` | `f64` | 쓰로틀 비율 (1.0 = 제한 없음, 0.0 = 완전 제한) |
| `EnergyConstraint` | `level` | `Level` | 에너지 제약 심각도 |
| | `reason` | `EnergyReason` | 에너지 제약 사유 |
| | `power_budget_mw` | `u32` | 전력 예산 (밀리와트) |

**Methods**:

| 메서드 | 시그니처 | 설명 |
|--------|---------|------|
| `level` | `fn level(&self) -> Level` | 모든 variant에서 `level` 필드를 추출 |

**예시**:
```rust
let sig = SystemSignal::MemoryPressure {
    level: Level::Critical,
    available_bytes: 50 * 1024 * 1024,
    reclaim_target_bytes: 100 * 1024 * 1024,
};
assert_eq!(sig.level(), Level::Critical);
```

---

### 1.3 `RecommendedBackend`

Manager가 권장하는 연산 백엔드.

**파일**: `src/resilience/signal.rs`

**정의**:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedBackend {
    Cpu,
    Gpu,
    Any,
}
```

**Variants**:

| Variant | D-Bus 문자열 | 의미 |
|---------|-------------|------|
| `Cpu` | `"cpu"` | CPU 백엔드 사용 권장 |
| `Gpu` | `"gpu"` | GPU 백엔드 사용 권장 |
| `Any` | `"any"` | 제한 없음, 아무 백엔드나 가능 |

**Methods**:

| 메서드 | 시그니처 | 설명 |
|--------|---------|------|
| `from_dbus_str` | `fn from_dbus_str(s: &str) -> Option<Self>` | D-Bus 문자열 변환 |

---

### 1.4 `ComputeReason`

연산 가이던스 시그널의 사유.

**파일**: `src/resilience/signal.rs`

**정의**:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeReason {
    CpuBottleneck,
    GpuBottleneck,
    CpuAvailable,
    GpuAvailable,
    BothLoaded,
    Balanced,
}
```

**Variants**:

| Variant | D-Bus 문자열 | 의미 |
|---------|-------------|------|
| `CpuBottleneck` | `"cpu_bottleneck"` | CPU가 병목 |
| `GpuBottleneck` | `"gpu_bottleneck"` | GPU가 병목 |
| `CpuAvailable` | `"cpu_available"` | CPU 여유 있음 |
| `GpuAvailable` | `"gpu_available"` | GPU 여유 있음 |
| `BothLoaded` | `"both_loaded"` | CPU, GPU 모두 부하 |
| `Balanced` | `"balanced"` | 부하 균형 |

**Methods**:

| 메서드 | 시그니처 | 설명 |
|--------|---------|------|
| `from_dbus_str` | `fn from_dbus_str(s: &str) -> Option<Self>` | D-Bus 문자열 변환 |

---

### 1.5 `EnergyReason`

에너지 제약 시그널의 사유.

**파일**: `src/resilience/signal.rs`

**정의**:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyReason {
    BatteryLow,
    BatteryCritical,
    PowerLimit,
    ThermalPower,
    Charging,
    None,
}
```

**Variants**:

| Variant | D-Bus 문자열 | 의미 |
|---------|-------------|------|
| `BatteryLow` | `"battery_low"` | 배터리 부족 |
| `BatteryCritical` | `"battery_critical"` | 배터리 위험 |
| `PowerLimit` | `"power_limit"` | 전력 제한 도달 |
| `ThermalPower` | `"thermal_power"` | 열 기반 전력 제한 |
| `Charging` | `"charging"` | 충전 중 |
| `None` | `"none"` | 사유 없음 |

**Methods**:

| 메서드 | 시그니처 | 설명 |
|--------|---------|------|
| `from_dbus_str` | `fn from_dbus_str(s: &str) -> Option<Self>` | D-Bus 문자열 변환 |

---

## 2. State Management

### 2.1 `OperatingMode`

4개 시그널 레벨 중 가장 심각한 값에 의해 결정되는 LLM 운영 모드.

**파일**: `src/resilience/state.rs`

**정의**:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatingMode {
    Normal,
    Degraded,
    Minimal,
    Suspended,
}
```

**Variants**:

| Variant | 대응 Level | 의미 |
|---------|-----------|------|
| `Normal` | `Level::Normal` | 정상 동작. 모든 기능 사용 가능 |
| `Degraded` | `Level::Warning` | 성능 저하 모드. 일부 자원 제약 |
| `Minimal` | `Level::Critical` | 최소 모드. 공격적 제약 적용 |
| `Suspended` | `Level::Emergency` | 일시중단 모드. 추론 중지 |

**Methods**:

| 메서드 | 시그니처 | 설명 |
|--------|---------|------|
| `from_levels` | `fn from_levels(memory: Level, compute: Level, thermal: Level, energy: Level) -> Self` | 4개 시그널 레벨에서 최악값을 선택하여 모드 결정 |

**예시**:
```rust
use llm_rs2::resilience::{Level, OperatingMode};

// 모든 레벨이 Normal이면 Normal 모드
let mode = OperatingMode::from_levels(
    Level::Normal, Level::Normal, Level::Normal, Level::Normal
);
assert_eq!(mode, OperatingMode::Normal);

// 하나라도 Critical이면 Minimal 모드
let mode = OperatingMode::from_levels(
    Level::Warning, Level::Critical, Level::Normal, Level::Normal
);
assert_eq!(mode, OperatingMode::Minimal);
```

---

## 3. Strategy Framework

### 3.1 `ResilienceAction`

Strategy가 시그널에 대응하여 반환하는 액션. 추론 루프에서 실행됨.

**파일**: `src/resilience/strategy/mod.rs`

**정의**:
```rust
#[derive(Debug, Clone)]
pub enum ResilienceAction {
    Evict { target_ratio: f32 },
    SwitchBackend { to: RecommendedBackend },
    LimitTokens { max_tokens: usize },
    Throttle { delay_ms: u64 },
    Suspend,
    RejectNew,
    RestoreDefaults,
}
```

**Variants**:

| Variant | 필드 | 설명 |
|---------|------|------|
| `Evict` | `target_ratio: f32` | KV 캐시 축출. `target_ratio`는 현재 캐시 대비 유지 비율 (0.0~1.0). 예: 0.50은 캐시의 50%만 유지 |
| `SwitchBackend` | `to: RecommendedBackend` | 추론 백엔드 전환 |
| `LimitTokens` | `max_tokens: usize` | 최대 생성 토큰 수 제한 |
| `Throttle` | `delay_ms: u64` | 토큰 생성 간 딜레이 삽입 (밀리초) |
| `Suspend` | (없음) | 추론 일시중단 |
| `RejectNew` | (없음) | 새로운 추론 요청 거부 |
| `RestoreDefaults` | (없음) | 이전 제약 해제, 정상 상태 복원 |

---

### 3.2 `ResilienceStrategy` trait

시그널에 대한 반응 전략 인터페이스. 각 시그널 카테고리마다 하나의 Strategy 구현체가 존재.

**파일**: `src/resilience/strategy/mod.rs`

**정의**:
```rust
pub trait ResilienceStrategy: Send + Sync {
    fn react(&mut self, signal: &SystemSignal, mode: OperatingMode) -> Vec<ResilienceAction>;
    fn name(&self) -> &str;
}
```

**Methods**:

| 메서드 | 시그니처 | 설명 |
|--------|---------|------|
| `react` | `fn react(&mut self, signal: &SystemSignal, mode: OperatingMode) -> Vec<ResilienceAction>` | 시그널을 받아 실행할 액션 목록 반환. 빈 `Vec`이면 조치 불필요 |
| `name` | `fn name(&self) -> &str` | 전략 이름 (로깅용) |

---

### 3.3 `resolve_conflicts`

여러 Strategy에서 반환된 액션들의 충돌을 해결하는 병합 함수.

**파일**: `src/resilience/strategy/mod.rs`

**시그니처**:
```rust
pub fn resolve_conflicts(actions: Vec<ResilienceAction>) -> Vec<ResilienceAction>
```

**충돌 해결 규칙**:

| 규칙 | 설명 |
|------|------|
| Suspend 최우선 | `Suspend`가 하나라도 있으면 다른 모든 액션을 무시하고 `[Suspend]`만 반환 |
| CPU 우선 | `SwitchBackend` 충돌 시 `Cpu`가 항상 우선 (안전 우선 원칙) |
| 최소 축출 비율 | 여러 `Evict`가 있으면 가장 공격적인 값(최소 `target_ratio`)을 선택 |
| 최대 딜레이 | 여러 `Throttle`이 있으면 가장 큰 `delay_ms`를 선택 |
| 최소 토큰 수 | 여러 `LimitTokens`가 있으면 가장 작은 `max_tokens`를 선택 |
| RestoreDefaults 조건부 | 다른 제약이 전혀 없을 때만 `RestoreDefaults` 통과. 다른 액션과 함께 있으면 억제됨 |

**예시**:
```rust
use llm_rs2::resilience::strategy::resolve_conflicts;
use llm_rs2::resilience::{ResilienceAction, RecommendedBackend};

// Suspend가 있으면 다른 모든 것을 무시
let actions = vec![
    ResilienceAction::Evict { target_ratio: 0.50 },
    ResilienceAction::Suspend,
    ResilienceAction::Throttle { delay_ms: 100 },
];
let resolved = resolve_conflicts(actions);
assert_eq!(resolved.len(), 1);
assert!(matches!(resolved[0], ResilienceAction::Suspend));

// CPU가 GPU보다 우선
let actions = vec![
    ResilienceAction::SwitchBackend { to: RecommendedBackend::Gpu },
    ResilienceAction::SwitchBackend { to: RecommendedBackend::Cpu },
];
let resolved = resolve_conflicts(actions);
assert!(matches!(resolved[0], ResilienceAction::SwitchBackend { to: RecommendedBackend::Cpu }));
```

---

## 4. Strategy Implementations

### 4.1 `MemoryStrategy`

메모리 압박에 대응하여 KV 캐시 축출을 수행하는 전략.

**파일**: `src/resilience/strategy/memory.rs`

**정의**:
```rust
pub struct MemoryStrategy {
    last_level: Level,  // 마지막으로 처리한 레벨 (중복 Normal 억제용)
}
```

**Methods**:

| 메서드 | 시그니처 | 설명 |
|--------|---------|------|
| `new` | `fn new() -> Self` | 새 인스턴스 생성. `last_level`은 `Level::Normal`로 초기화 |

**Level -> Action 매핑** (코드 기준):

| Level | 반환 Actions | 설명 |
|-------|-------------|------|
| `Normal` | `[RestoreDefaults]` | 제약 해제. 단, 이전 레벨도 Normal이면 빈 Vec 반환 (중복 억제) |
| `Warning` | `[Evict { target_ratio: 0.85 }]` | 캐시의 85% 유지 (15% 축출) |
| `Critical` | `[Evict { target_ratio: 0.50 }]` | 캐시의 50% 유지 (50% 축출) |
| `Emergency` | `[Evict { target_ratio: 0.25 }, RejectNew]` | 캐시의 25% 유지 + 새 요청 거부 |

**예시**:
```rust
let mut strategy = MemoryStrategy::new();

let signal = SystemSignal::MemoryPressure {
    level: Level::Critical,
    available_bytes: 50 * 1024 * 1024,
    reclaim_target_bytes: 100 * 1024 * 1024,
};
let actions = strategy.react(&signal, OperatingMode::Minimal);
// [Evict { target_ratio: 0.50 }]
```

---

### 4.2 `ComputeStrategy`

연산 자원 부하에 대응하여 백엔드 전환 및 쓰로틀링을 수행하는 전략.

**파일**: `src/resilience/strategy/compute.rs`

**정의**:
```rust
pub struct ComputeStrategy {
    current_backend: RecommendedBackend,  // 현재 추적 중인 백엔드 (초기값: Any)
}
```

**Methods**:

| 메서드 | 시그니처 | 설명 |
|--------|---------|------|
| `new` | `fn new() -> Self` | 새 인스턴스 생성. `current_backend`은 `RecommendedBackend::Any`로 초기화 |

**Level -> Action 매핑** (코드 기준):

| Level | 반환 Actions | 설명 |
|-------|-------------|------|
| `Normal` | `[RestoreDefaults]` | 제약 해제. `current_backend`을 `Any`로 리셋 |
| `Warning` | `[]` (빈 Vec) | 준비만 함. `current_backend`을 `recommended_backend`로 업데이트하지만 전환하지 않음 |
| `Critical` | `recommended_backend != current_backend`이면 `[SwitchBackend { to: recommended_backend }]`, 같으면 `[Throttle { delay_ms: 50 }]` | 권장 백엔드로 전환. 이미 같은 백엔드면 쓰로틀링 |
| `Emergency` | `[SwitchBackend { to: Cpu }, Throttle { delay_ms: 100 }]` | 무조건 CPU 전환 + 100ms 쓰로틀링 |

**예시**:
```rust
let mut strategy = ComputeStrategy::new();

// Warning 단계: 아무 조치 없이 내부 상태만 업데이트
let signal = SystemSignal::ComputeGuidance {
    level: Level::Warning,
    recommended_backend: RecommendedBackend::Cpu,
    reason: ComputeReason::GpuBottleneck,
    cpu_usage_pct: 30.0,
    gpu_usage_pct: 95.0,
};
let actions = strategy.react(&signal, OperatingMode::Degraded);
assert!(actions.is_empty());

// Emergency: 무조건 CPU 전환 + 쓰로틀링
let signal = SystemSignal::ComputeGuidance {
    level: Level::Emergency,
    recommended_backend: RecommendedBackend::Cpu,
    reason: ComputeReason::BothLoaded,
    cpu_usage_pct: 90.0,
    gpu_usage_pct: 95.0,
};
let actions = strategy.react(&signal, OperatingMode::Suspended);
// [SwitchBackend { to: Cpu }, Throttle { delay_ms: 100 }]
```

---

### 4.3 `ThermalStrategy`

열 경고에 대응하여 연산 강도를 낮추는 전략. Stateless 구조체 (필드 없음).

**파일**: `src/resilience/strategy/thermal.rs`

**정의**:
```rust
pub struct ThermalStrategy;
```

**Methods**:

| 메서드 | 시그니처 | 설명 |
|--------|---------|------|
| `new` | `fn new() -> Self` | 새 인스턴스 생성 |

**Level -> Action 매핑** (코드 기준):

| Level | 반환 Actions | 설명 |
|-------|-------------|------|
| `Normal` | `[RestoreDefaults]` | 제약 해제 |
| `Warning` | `[SwitchBackend { to: Cpu }]` | GPU 사용 중단, CPU로 전환 |
| `Critical` | `[SwitchBackend { to: Cpu }, Throttle { delay_ms: delay }, LimitTokens { max_tokens: 64 }]` | CPU 전환 + 동적 쓰로틀 + 토큰 제한. `delay = ((1.0 - throttle_ratio) * 100.0) as u64` |
| `Emergency` | `[Suspend]` | 추론 즉시 중단 |

**Critical 레벨 delay 계산**: `throttle_ratio` 값에 따라 딜레이가 동적으로 결정됨.

| `throttle_ratio` | 계산 | `delay_ms` |
|-------------------|------|-----------|
| 1.0 (제한 없음) | `(1.0 - 1.0) * 100` | 0 |
| 0.5 (50% 제한) | `(1.0 - 0.5) * 100` | 50 |
| 0.0 (완전 제한) | `(1.0 - 0.0) * 100` | 100 |

**예시**:
```rust
let mut strategy = ThermalStrategy::new();

let signal = SystemSignal::ThermalAlert {
    level: Level::Critical,
    temperature_mc: 80000,    // 80°C
    throttling_active: true,
    throttle_ratio: 0.5,      // 50% 제한
};
let actions = strategy.react(&signal, OperatingMode::Minimal);
// [SwitchBackend { to: Cpu }, Throttle { delay_ms: 50 }, LimitTokens { max_tokens: 64 }]
```

---

### 4.4 `EnergyStrategy`

에너지 제약에 대응하여 전력 소비를 줄이는 전략. Stateless 구조체 (필드 없음).

**파일**: `src/resilience/strategy/energy.rs`

**정의**:
```rust
pub struct EnergyStrategy;
```

**Methods**:

| 메서드 | 시그니처 | 설명 |
|--------|---------|------|
| `new` | `fn new() -> Self` | 새 인스턴스 생성 |

**Level -> Action 매핑** (코드 기준):

| Level | 반환 Actions | 설명 |
|-------|-------------|------|
| `Normal` | `[RestoreDefaults]` | 제약 해제 |
| `Warning` | `[SwitchBackend { to: Cpu }]` | GPU 대신 CPU 사용 (절전) |
| `Critical` | `[SwitchBackend { to: Cpu }, LimitTokens { max_tokens: 64 }, Throttle { delay_ms: 30 }]` | CPU 전환 + 토큰 제한 + 30ms 쓰로틀링 |
| `Emergency` | `[Suspend, RejectNew]` | 추론 중단 + 새 요청 거부 |

**예시**:
```rust
let mut strategy = EnergyStrategy::new();

let signal = SystemSignal::EnergyConstraint {
    level: Level::Critical,
    reason: EnergyReason::BatteryLow,
    power_budget_mw: 3000,
};
let actions = strategy.react(&signal, OperatingMode::Minimal);
// [SwitchBackend { to: Cpu }, LimitTokens { max_tokens: 64 }, Throttle { delay_ms: 30 }]
```

---

### 4.5 전략 매핑 비교 요약

모든 Strategy의 Level별 액션을 한눈에 비교하는 표.

| Level | Memory | Compute | Thermal | Energy |
|-------|--------|---------|---------|--------|
| **Normal** | `RestoreDefaults` | `RestoreDefaults` | `RestoreDefaults` | `RestoreDefaults` |
| **Warning** | `Evict(0.85)` | `[]` (빈 Vec) | `SwitchBackend(Cpu)` | `SwitchBackend(Cpu)` |
| **Critical** | `Evict(0.50)` | `SwitchBackend(rec)` 또는 `Throttle(50)` | `SwitchBackend(Cpu)` + `Throttle(동적)` + `LimitTokens(64)` | `SwitchBackend(Cpu)` + `LimitTokens(64)` + `Throttle(30)` |
| **Emergency** | `Evict(0.25)` + `RejectNew` | `SwitchBackend(Cpu)` + `Throttle(100)` | `Suspend` | `Suspend` + `RejectNew` |

---

## 5. Manager & Execution

### 5.1 `ResilienceManager`

중앙 오케스트레이터. `mpsc` 채널로 시그널을 수신하고, Strategy에 위임하며, 충돌을 해결하여 액션 목록을 반환.

**파일**: `src/resilience/manager.rs`

**정의**:
```rust
pub struct ResilienceManager {
    rx: mpsc::Receiver<SystemSignal>,
    mode: OperatingMode,
    current_levels: SignalLevels,  // (private) 시그널 카테고리별 최신 레벨 캐시
    strategies: Strategies,        // (private) 4개 Strategy 구현체
}
```

**Methods**:

| 메서드 | 시그니처 | 설명 |
|--------|---------|------|
| `new` | `fn new(rx: mpsc::Receiver<SystemSignal>) -> Self` | 기본 Strategy들로 새 매니저 생성. 초기 모드는 `Normal` |
| `poll` | `fn poll(&mut self) -> Vec<ResilienceAction>` | 논블로킹 폴링: 채널에 쌓인 시그널을 모두 drain하고, Strategy 위임 후 `resolve_conflicts`로 병합 반환. 시그널 없으면 빈 Vec |
| `mode` | `fn mode(&self) -> OperatingMode` | 현재 운영 모드 반환 |

**내부 처리 흐름** (`poll` 호출 시):

1. `rx.try_recv()`로 채널에서 대기 중인 시그널을 모두 꺼냄
2. 각 시그널에 대해:
   - 해당 카테고리의 `current_levels` 업데이트
   - `OperatingMode::from_levels()`로 운영 모드 재계산
   - 해당 Strategy의 `react()` 호출
3. 모든 액션을 `resolve_conflicts()`로 병합
4. 결과 반환

**예시**:
```rust
use std::sync::mpsc;
use llm_rs2::resilience::{ResilienceManager, SystemSignal, Level};

let (tx, rx) = mpsc::channel();
let mut mgr = ResilienceManager::new(rx);

// 시그널 전송
tx.send(SystemSignal::MemoryPressure {
    level: Level::Warning,
    available_bytes: 200 * 1024 * 1024,
    reclaim_target_bytes: 50 * 1024 * 1024,
}).unwrap();

// 폴링 — 논블로킹
let actions = mgr.poll();
assert!(!actions.is_empty());
assert_eq!(mgr.mode(), OperatingMode::Degraded);
```

---

### 5.2 `DbusListener`

별도 스레드에서 D-Bus System Bus에 연결하여 `org.llm.Manager1` 시그널을 수신하는 리스너.

**파일**: `src/resilience/dbus_listener.rs`

**정의**:
```rust
pub struct DbusListener {
    tx: mpsc::Sender<SystemSignal>,
}
```

**D-Bus 상수** (private):

| 상수 | 값 | 설명 |
|------|---|------|
| `MANAGER_DEST` | `"org.llm.Manager1"` | D-Bus well-known name |
| `MANAGER_PATH` | `"/org/llm/Manager1"` | D-Bus object path |
| `MANAGER_IFACE` | `"org.llm.Manager1"` | D-Bus interface |

**Methods**:

| 메서드 | 시그니처 | 설명 |
|--------|---------|------|
| `new` | `fn new(tx: mpsc::Sender<SystemSignal>) -> Self` | Sender 채널을 받아 리스너 생성 |
| `spawn` | `fn spawn(self) -> std::thread::JoinHandle<()>` | 별도 스레드에서 D-Bus 수신 루프 시작. Manager 연결 실패 시 경고 로그 출력 후 정상 종료 (LLM은 Resilience 없이 계속 동작) |

**D-Bus 시그널 파싱**:

| D-Bus 시그널 이름 | 파싱 함수 (private) | D-Bus 시그니처 | 변환 결과 |
|------------------|-------------------|---------------|----------|
| `MemoryPressure` | `parse_memory_pressure` | `(s, t, t)` — level, available_bytes, reclaim_target_bytes | `SystemSignal::MemoryPressure` |
| `ComputeGuidance` | `parse_compute_guidance` | `(s, s, s, d, d)` — level, backend, reason, cpu_pct, gpu_pct | `SystemSignal::ComputeGuidance` |
| `ThermalAlert` | `parse_thermal_alert` | `(s, i, b, d)` — level, temp_mc, throttling, ratio | `SystemSignal::ThermalAlert` |
| `EnergyConstraint` | `parse_energy_constraint` | `(s, s, u)` — level, reason, power_mw | `SystemSignal::EnergyConstraint` |

**예시**:
```rust
use std::sync::mpsc;
use llm_rs2::resilience::DbusListener;

let (tx, rx) = mpsc::channel();
let listener = DbusListener::new(tx);
let _handle = listener.spawn();  // 별도 스레드에서 D-Bus 시그널 수신 시작
```

---

### 5.3 `InferenceContext`

`execute_action` 함수에 전달되는 추론 루프의 가변 상태 참조 묶음.

**파일**: `src/resilience/manager.rs`

**정의**:
```rust
pub struct InferenceContext<'a> {
    pub max_tokens: &'a mut usize,
    pub throttle_delay_ms: &'a mut u64,
    pub suspended: &'a mut bool,
    pub reject_new: &'a mut bool,
}
```

**Fields**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `max_tokens` | `&'a mut usize` | 최대 생성 토큰 수. `LimitTokens` 액션이 현재 값과 min을 취함 |
| `throttle_delay_ms` | `&'a mut u64` | 토큰 간 딜레이 (ms). `Throttle` 액션이 설정 |
| `suspended` | `&'a mut bool` | 중단 플래그. `Suspend` 액션이 `true`로 설정 |
| `reject_new` | `&'a mut bool` | 새 요청 거부 플래그. `RejectNew` 액션이 `true`로 설정 |

---

### 5.4 `execute_action`

단일 `ResilienceAction`을 `InferenceContext`에 적용하는 실행 함수.

**파일**: `src/resilience/manager.rs`

**시그니처**:
```rust
pub fn execute_action(action: &ResilienceAction, ctx: &mut InferenceContext)
```

**액션별 동작**:

| Action | 동작 |
|--------|------|
| `Evict { target_ratio }` | 로그 출력만 수행. Phase 3a에서 CacheManager 통합 예정 |
| `SwitchBackend { to }` | 로그 출력만 수행. Phase 3b에서 generate_hybrid 통합 예정 |
| `LimitTokens { max_tokens }` | `ctx.max_tokens = min(ctx.max_tokens, max_tokens)` — 항상 더 작은 값 적용 |
| `Throttle { delay_ms }` | `ctx.throttle_delay_ms = delay_ms` — 딜레이 설정 |
| `Suspend` | `ctx.suspended = true` |
| `RejectNew` | `ctx.reject_new = true` |
| `RestoreDefaults` | `ctx.throttle_delay_ms = 0`, `ctx.reject_new = false` — 쓰로틀과 거부 플래그 초기화 |

**예시**:
```rust
use llm_rs2::resilience::{InferenceContext, ResilienceAction, execute_action};

let mut max_tokens = 128;
let mut throttle_delay_ms = 0u64;
let mut suspended = false;
let mut reject_new = false;

let mut ctx = InferenceContext {
    max_tokens: &mut max_tokens,
    throttle_delay_ms: &mut throttle_delay_ms,
    suspended: &mut suspended,
    reject_new: &mut reject_new,
};

// LimitTokens는 현재 값과 min을 취함
execute_action(&ResilienceAction::LimitTokens { max_tokens: 64 }, &mut ctx);
assert_eq!(max_tokens, 64);

// 더 큰 값으로 다시 호출해도 기존 값 유지
execute_action(&ResilienceAction::LimitTokens { max_tokens: 200 }, &mut ctx);
assert_eq!(max_tokens, 64);  // min(64, 200) = 64
```

---

## 6. CLI Integration

### 6.1 `--enable-resilience` 플래그

Resilience Manager를 활성화하는 CLI 옵션. `resilience` feature가 활성화된 빌드에서만 동작.

**파일**: `src/bin/generate.rs`

**정의**:
```rust
/// Enable D-Bus resilience manager for adaptive inference
#[arg(long, default_value_t = false)]
enable_resilience: bool,
```

기본값은 `false`. `--enable-resilience`를 명시하면 `true`.

---

### 6.2 초기화 패턴

`generate.rs`의 main 함수에서 Resilience Manager 초기화 코드 (약 293~305행).

```rust
// 5. Resilience Manager (optional)
#[cfg(feature = "resilience")]
let mut resilience_manager = if args.enable_resilience {
    let (tx, rx) = std::sync::mpsc::channel();
    let listener = DbusListener::new(tx);
    let _listener_handle = listener.spawn();
    eprintln!("[Resilience] Manager enabled — listening for D-Bus signals");
    Some(ResilienceManager::new(rx))
} else {
    None
};
#[cfg(feature = "resilience")]
let mut throttle_delay_ms: u64 = 0;
```

**초기화 순서**:

1. `mpsc::channel()` 생성 — `tx`는 DbusListener에, `rx`는 ResilienceManager에 전달
2. `DbusListener::new(tx).spawn()` — 별도 스레드에서 D-Bus 시그널 수신 시작
3. `ResilienceManager::new(rx)` — `Some`으로 감싸서 Optional로 보관
4. `throttle_delay_ms` 별도 변수 — 토큰 루프에서 딜레이 적용에 사용

---

### 6.3 토큰 루프 체크포인트

Generation 루프 내에서 매 토큰마다 `ResilienceManager::poll()`을 호출하는 체크포인트 코드 (약 551~596행).

```rust
// ── Resilience checkpoint ─────────────────────────
#[cfg(feature = "resilience")]
if let Some(rm) = &mut resilience_manager {
    let mut suspended = false;
    let mut reject_new = false;
    let mut num_tokens = args.num_tokens;
    let mut ctx = InferenceContext {
        max_tokens: &mut num_tokens,
        throttle_delay_ms: &mut throttle_delay_ms,
        suspended: &mut suspended,
        reject_new: &mut reject_new,
    };

    for action in rm.poll() {
        if let ResilienceAction::Evict { target_ratio } = &action {
            // Evict: KV 캐시에서 직접 prune_prefix 수행
            let target_len = (kv_caches[0].current_pos as f32 * target_ratio) as usize;
            let remove = kv_caches[0].current_pos.saturating_sub(target_len);
            if remove > 0 {
                for cache in kv_caches.iter_mut() {
                    if let Err(e) = cache.prune_prefix(remove) {
                        eprintln!("[Resilience] Eviction error: {}", e);
                    }
                }
            }
        } else {
            execute_action(&action, &mut ctx);
        }
    }

    args.num_tokens = num_tokens;

    if suspended {
        eprintln!("\n[Resilience] Inference suspended by system signal");
        break;
    }

    if throttle_delay_ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(throttle_delay_ms));
    }
}
// ── End Resilience checkpoint ─────────────────────
```

**체크포인트 처리 순서**:

1. 로컬 변수(`suspended`, `reject_new`, `num_tokens`)를 생성하여 `InferenceContext` 구성
2. `rm.poll()`로 액션 목록 획득
3. 각 액션 실행:
   - `Evict`: `execute_action` 대신 직접 `KVCache::prune_prefix()` 호출하여 캐시 축출
   - 그 외: `execute_action()` 위임
4. `args.num_tokens` 업데이트 (LimitTokens 반영)
5. `suspended`이면 `break`로 생성 루프 탈출
6. `throttle_delay_ms > 0`이면 `thread::sleep`으로 딜레이 삽입

**주의사항**: `Evict` 액션은 `execute_action()`을 거치지 않고 체크포인트에서 직접 처리됨. `execute_action()`의 `Evict` 분기는 로그만 출력하며, 실제 축출 로직은 `generate.rs`에 있음.

---

> 이 문서는 소스 코드에서 직접 추출한 정확한 타입 정의와 매핑을 제공합니다.
> 설계 의도와 사용 방법은 [24. Resilience 사용 가이드](./24_resilience_usage_guide.md)를 참조하세요.
