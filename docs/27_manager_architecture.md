# 27. Manager Service Architecture

> LLM Resource Manager 내부 아키텍처 문서
> Crate: `llm_manager` (`manager/`)

## 1. Overview

Manager는 시스템 리소스를 모니터링하고, 임계값 기반 판단을 거쳐 LLM 엔진에 `SystemSignal`을 전달하는 독립 서비스다. LLM 엔진은 수신한 신호에 따라 KV 캐시 eviction, 백엔드 전환, throttle 등을 자율적으로 수행한다.

**핵심 원칙:**
- **단방향**: Manager → LLM 신호만 존재. LLM은 수신만 함
- **Fail-Safe**: Manager가 죽어도 LLM은 독립적으로 추론 계속 가능
- **OCP**: PolicyEngine trait으로 평가 전략 교체 가능 (알고리즘 미확정)
- **No async**: `std::thread` + `std::sync::mpsc`만 사용

## 2. 3-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Main Thread                                │
│                                                                     │
│  ┌──────────────────┐     ┌────────────────┐     ┌───────────────┐ │
│  │    Collectors     │     │  PolicyEngine  │     │    Emitter    │ │
│  │  (4 threads)      │     │  (main thread) │     │ (main thread) │ │
│  │                   │     │                │     │               │ │
│  │  MemoryCollector  │     │ ThresholdPolicy│     │ DbusEmitter   │ │
│  │  ThermalCollector │ tx  │ (hysteresis)   │     │    or         │ │
│  │  ComputeCollector │────▶│                │────▶│ UnixSocket    │──▶ LLM
│  │  EnergyCollector  │     │ process()      │     │ Emitter       │ │
│  │                   │mpsc │  → Vec<Signal> │     │               │ │
│  └──────────────────┘     └────────────────┘     └───────────────┘ │
│                                                                     │
│  SIGINT/SIGTERM ──▶ AtomicBool shutdown ──▶ all threads exit        │
└─────────────────────────────────────────────────────────────────────┘
```

**데이터 흐름:**

```
/proc/meminfo ──┐
/proc/pressure/ ─┤
                 ├──▶ Reading ──▶ mpsc ──▶ PolicyEngine::process()
/sys/class/      │                              │
  thermal/ ──────┤                         Level 변경?
/proc/stat ──────┤                         ├── Yes → SystemSignal
/sys/class/      │                         └── No  → (skip)
  power_supply/ ─┘                              │
                                                ▼
                                     Emitter::emit(signal)
                                                │
                                     ┌──────────┴──────────┐
                                     │                     │
                                D-Bus Signal       Unix Socket
                              (Linux System Bus)   (length-prefixed JSON)
                                     │                     │
                                     └──────────┬──────────┘
                                                ▼
                                          LLM Engine
                                    (SignalListener<T>)
```

## 3. Module Structure

```
manager/
├── Cargo.toml
└── src/
    ├── lib.rs                    # pub mod 선언
    ├── main.rs                   # CLI, 스레드 오케스트레이션, 메인 루프
    ├── config.rs                 # TOML 설정 (Config, *Thresholds)
    ├── collector/
    │   ├── mod.rs                # Collector trait, Reading, ReadingData
    │   ├── memory.rs             # /proc/meminfo + /proc/pressure/memory
    │   ├── thermal.rs            # /sys/class/thermal/thermal_zone*/temp
    │   ├── compute.rs            # /proc/stat (CPU delta 계산)
    │   └── energy.rs             # /sys/class/power_supply/*/capacity
    ├── policy/
    │   ├── mod.rs                # PolicyEngine trait
    │   └── threshold.rs          # ThresholdPolicy (히스테리시스 기반)
    ├── emitter/
    │   ├── mod.rs                # Emitter trait
    │   ├── dbus.rs               # DbusEmitter (org.llm.Manager1)
    │   └── unix_socket.rs        # UnixSocketEmitter (4B BE len + JSON)
    └── bin/
        └── mock_manager.rs       # 테스트용 D-Bus 신호 발신기
```

**코드 규모:** ~2,800줄 (테스트 포함), 41개 유닛 테스트

## 4. Core Traits

### 4.1 Collector

```rust
pub trait Collector: Send + 'static {
    fn run(&mut self, tx: mpsc::Sender<Reading>, shutdown: Arc<AtomicBool>) -> anyhow::Result<()>;
    fn name(&self) -> &str;
}
```

각 Collector는 전용 스레드에서 실행되며, `Reading`을 중앙 채널로 전송한다. 현재 4개 모두 polling 방식이며, 향후 PSI event나 D-Bus PropertiesChanged 등 이벤트 구동으로 전환 가능하다.

### 4.2 PolicyEngine (OCP 확장 포인트)

```rust
pub trait PolicyEngine: Send {
    fn process(&mut self, reading: &Reading) -> Vec<SystemSignal>;
    fn current_signals(&self) -> Vec<SystemSignal>;
    fn name(&self) -> &str;
}
```

**확장 시나리오:**
- `TrendPolicy` — 이동 평균, 변화율 기반 예측
- `MLPolicy` — 학습된 모델로 선제적 level 판단
- `CompositePolicy` — 여러 sub-policy를 조합

메인 루프와 Emitter 코드는 수정 없이 새 PolicyEngine 구현체만 교체하면 된다.

### 4.3 Emitter

```rust
pub trait Emitter: Send {
    fn emit(&mut self, signal: &SystemSignal) -> anyhow::Result<()>;
    fn emit_initial(&mut self, signals: &[SystemSignal]) -> anyhow::Result<()>;
    fn name(&self) -> &str;
}
```

| 구현체 | 전송 방식 | 플랫폼 |
|--------|----------|--------|
| `DbusEmitter` | System Bus `org.llm.Manager1` 신호 | Linux |
| `UnixSocketEmitter` | `[4B BE u32 len][UTF-8 JSON]` | Android / 범용 |

## 5. Reading Types

`ReadingData` enum이 4종의 리소스 데이터를 표현한다:

| Variant | 주요 필드 | 데이터 소스 |
|---------|----------|------------|
| `Memory` | `available_bytes`, `total_bytes`, `psi_some_avg10` | `/proc/meminfo`, `/proc/pressure/memory` |
| `Thermal` | `temperature_mc`, `throttling_active` | `/sys/class/thermal/thermal_zone*/temp`, `cooling_device*/cur_state` |
| `Compute` | `cpu_usage_pct`, `gpu_usage_pct` | `/proc/stat` (delta 계산), GPU는 미구현 (0.0) |
| `Energy` | `battery_pct`, `charging`, `power_draw_mw` | `/sys/class/power_supply/*/capacity`, `status`, `power_now` |

## 6. ThresholdPolicy — 히스테리시스 설계

### 6.1 히스테리시스 원리

Level 전이 시 진동(oscillation) 방지를 위해 상향/하향 임계값을 다르게 설정한다:

```
          Thermal 예시 (hysteresis = 5000mc = 5°C)

상향(악화):  Normal ──60°C──▶ Warning ──75°C──▶ Critical ──85°C──▶ Emergency
하향(회복):  Normal ◀──55°C── Warning ◀──70°C── Critical ◀──80°C── Emergency
                      △              △               △
                  -5°C gap       -5°C gap         -5°C gap
```

### 6.2 평가 함수

두 가지 방향의 히스테리시스 평가 함수가 존재한다:

| 함수 | 메트릭 방향 | 적용 대상 |
|------|-----------|----------|
| `level_ascending()` | 값이 높을수록 나쁨 | Thermal (온도), Compute (사용률) |
| `level_descending()` | 값이 낮을수록 나쁨 | Memory (가용 %), Energy (배터리 %) |

### 6.3 전이 규칙

- **악화 (escalation)**: 즉시, 다단계 점프 가능 (Normal → Emergency)
- **회복 (recovery)**: recovery threshold 통과 필요. 현재 level의 recovery threshold을 통과하면 값에 맞는 level로 이동 (다단계 회복 가능)

예시 — Emergency에서 온도가 40°C로 급락:
1. `40 < 80 (emergency_down)` → Emergency 탈출
2. `40 < 55 (warning_down)` → Normal로 직행

### 6.4 신호별 특이사항

| 신호 | 특이사항 |
|------|---------|
| `MemoryPressure` | `reclaim_target_bytes`를 level에 따라 총 메모리의 5%/10%/20%로 계산 |
| `ComputeGuidance` | Emergency level 없음 (최대 Critical). level 변경 **또는** `recommended_backend` 변경 시 발신 |
| `ThermalAlert` | `throttle_ratio`를 level에 따라 고정 (Normal/Warning=1.0, Critical=0.7, Emergency=0.3) |
| `EnergyConstraint` | `ignore_when_charging = true`면 충전 중 강제 Normal. 배터리 없는 시스템은 항상 Normal |

## 7. Collector 구현 상세

### 7.1 MemoryCollector

| 항목 | 값 |
|------|-----|
| 데이터 소스 | `/proc/meminfo` (MemTotal, MemAvailable), `/proc/pressure/memory` (PSI some avg10) |
| 파싱 | `strip_prefix` + `trim` (정규식 불필요) |
| 단위 변환 | kB → bytes (×1024) |
| PSI 실패 처리 | `psi_some_avg10 = None` (PSI 미지원 커널) |

### 7.2 ThermalCollector

| 항목 | 값 |
|------|-----|
| 데이터 소스 | `thermal_zone0..31/temp` (최고 온도 선택), `cooling_device0..63/cur_state` |
| 쓰로틀링 판단 | `cur_state > 0`이면 `throttling_active = true` |
| zone 탐색 | 순차 시도, 파일 없으면 중단 (0-based) |

### 7.3 ComputeCollector

| 항목 | 값 |
|------|-----|
| 데이터 소스 | `/proc/stat` 첫 줄 (aggregate CPU) |
| 계산 방식 | 두 스냅샷 간 delta: `(total_delta - idle_delta) / total_delta × 100` |
| 첫 번째 읽기 | 기준 스냅샷 수집만 하고, Reading 발생 안 함 (delta 없음) |
| GPU | 미구현 (0.0). 벤더별 sysfs 필요 |

### 7.4 EnergyCollector

| 항목 | 값 |
|------|-----|
| 배터리 탐색 | `/sys/class/power_supply/*/type` == `"Battery"` && `capacity` 파일 존재 |
| 충전 판단 | `status` == `"Charging"` 또는 `"Full"` |
| 전력 소모 | `power_now` (microWatts → milliWatts, ÷1000) |
| 배터리 없음 | `battery_pct = None`, `charging = false` |

## 8. Configuration

TOML 파일로 설정. 모든 필드에 기본값이 있어 설정 파일 없이도 동작한다.

**기본 경로:** `/etc/llm-manager/config.toml`

```toml
[monitor]
poll_interval_ms = 1000          # 모든 Collector의 폴링 간격

[memory]
warning_available_pct = 40.0     # 가용 메모리 < 40% → Warning
critical_available_pct = 20.0
emergency_available_pct = 10.0
hysteresis_pct = 5.0             # 회복 시 +5%p 필요

[thermal]
warning_temp_mc = 60000          # 60°C → Warning
critical_temp_mc = 75000
emergency_temp_mc = 85000
hysteresis_mc = 5000             # 회복 시 -5°C 필요

[compute]
warning_usage_pct = 70.0         # CPU/GPU 중 높은 쪽 > 70% → Warning
critical_usage_pct = 90.0
hysteresis_pct = 5.0

[energy]
warning_battery_pct = 30.0
critical_battery_pct = 15.0
emergency_battery_pct = 5.0
warning_power_budget_mw = 3000   # Warning 시 LLM에 전달할 전력 예산
critical_power_budget_mw = 1500
emergency_power_budget_mw = 500
ignore_when_charging = true      # 충전 중이면 강제 Normal
```

## 9. CLI Usage

```bash
# Linux D-Bus 모드 (기본)
llm_manager --config /etc/llm-manager/config.toml --transport dbus

# Unix socket 모드 (Android 또는 테스트)
llm_manager --transport unix:/tmp/llm_manager.sock --client-timeout 30

# 기본 설정으로 실행
llm_manager --transport unix:/tmp/llm_manager.sock
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-c, --config` | `/etc/llm-manager/config.toml` | TOML 설정 파일 경로 |
| `-t, --transport` | `dbus` | `dbus` 또는 `unix:<socket_path>` |
| `--client-timeout` | `60` | Unix socket 모드에서 클라이언트 연결 대기 (초) |

## 10. Threading Model

```
Main Thread
├── signal handler (SIGINT/SIGTERM → AtomicBool)
├── emitter 초기화
├── policy engine 초기화
├── collector 스레드 spawn (4개)
│   ├── MemoryCollector thread   ──┐
│   ├── ThermalCollector thread  ──┤ mpsc::Sender<Reading>
│   ├── ComputeCollector thread  ──┤
│   └── EnergyCollector thread   ──┘
│                                   │
│   mpsc::Receiver<Reading>  ◀──────┘
│
└── main loop:
    recv_timeout(1s) → policy.process() → emitter.emit()
```

- Collector 스레드는 각각 독립적으로 동작. 하나가 실패해도 나머지 계속 수집
- `mpsc::Sender<Reading>` clone으로 4개 스레드가 하나의 채널에 전송
- 원본 Sender를 drop하여, 모든 Collector가 종료되면 `Disconnected` 감지
- Shutdown: `SIGINT`/`SIGTERM` → 전역 `AtomicBool` → 메인 루프 break → 공유 `Arc<AtomicBool>` 전파 → Collector 스레드 종료 → `join()`

## 11. Wire Protocols

### 11.1 D-Bus (Linux)

| 항목 | 값 |
|------|-----|
| Bus | System Bus |
| Well-known Name | `org.llm.Manager1` |
| Object Path | `/org/llm/Manager1` |
| Interface | `org.llm.Manager1` |
| 신호 | `MemoryPressure`, `ComputeGuidance`, `ThermalAlert`, `EnergyConstraint` |

각 신호의 인자 타입과 형식은 [docs/20_dbus_ipc_spec.md](20_dbus_ipc_spec.md) 참조.

### 11.2 Unix Socket

| 항목 | 값 |
|------|-----|
| 소켓 타입 | `AF_UNIX`, `SOCK_STREAM` |
| Wire format | `[4-byte BE u32 length][UTF-8 JSON payload]` |
| JSON 형식 | `llm_shared::SystemSignal`의 serde 직렬화 |
| 연결 모델 | 단일 클라이언트 (1:1) |

Engine 측 `UnixSocketTransport`와 동일한 wire format을 사용하여 호환된다.

## 12. SystemSignal 공유 타입

`llm_shared` crate에 정의된 공유 타입. Manager(생산자)와 Engine(소비자) 모두 이 타입을 사용한다.

```rust
pub enum SystemSignal {
    MemoryPressure   { level, available_bytes, reclaim_target_bytes },
    ComputeGuidance  { level, recommended_backend, reason, cpu_usage_pct, gpu_usage_pct },
    ThermalAlert     { level, temperature_mc, throttling_active, throttle_ratio },
    EnergyConstraint { level, reason, power_budget_mw },
}
```

`Level`: `Normal < Warning < Critical < Emergency` (Ord 구현)

## 13. 테스트 전략

### 13.1 유닛 테스트 (41개)

| 영역 | 테스트 수 | 방법 |
|------|----------|------|
| ThresholdPolicy 히스테리시스 | 26 | 직접 Reading 주입, Level 전이 검증 |
| Collector 파싱 | 13 | `tempfile`로 synthetic sysfs 생성 |
| UnixSocket 라운드트립 | 2 | 소켓 생성 → 연결 → 신호 송수신 |

### 13.2 E2E 검증

```bash
# Manager 기동 (Unix socket)
RUST_LOG=info llm_manager --transport unix:/tmp/test.sock --client-timeout 3 &

# Python 클라이언트로 초기 신호 수신 확인
python3 -c "
import socket, struct, json
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect('/tmp/test.sock')
for _ in range(4):
    length = struct.unpack('>I', s.recv(4))[0]
    print(json.loads(s.recv(length)))
"
```

### 13.3 mock_manager

테스트용 D-Bus 신호 발신기. Engine의 `DbusTransport` 수신 테스트에 사용:

```bash
# 단일 신호
mock_manager --signal MemoryPressure --level critical \
    --available-bytes 50000000 --reclaim-target 100000000

# 시나리오 재생
mock_manager --scenario scenarios/thermal_spike.json
```

## 14. 의존성

| Crate | 용도 |
|-------|------|
| `llm_shared` | `SystemSignal`, `Level` 등 공유 타입 |
| `serde` + `serde_json` | 설정/신호 직렬화 |
| `toml` | TOML 설정 파싱 |
| `clap` | CLI 인자 파싱 |
| `log` + `env_logger` | 로깅 |
| `libc` | SIGINT/SIGTERM 핸들러 |
| `zbus` (blocking-api) | D-Bus System Bus 접근 |
| `tempfile` (dev) | 테스트용 임시 파일 |

**async 라이브러리 없음** — `zbus`의 `blocking-api` feature만 사용. 내부적으로 `async-io` (smol)가 blocking wrapper 아래에서 동작하지만, Manager 코드에 async 개념은 노출되지 않는다.

## 15. Known Limitations

| 항목 | 현황 | 개선 방향 |
|------|------|----------|
| GPU 사용률 | 항상 0.0 | 벤더별 sysfs 파싱 필요 (Adreno: `/sys/class/kgsl/`) |
| Collector 방식 | 전부 polling | Memory: PSI `poll()` 이벤트, Energy: UPower D-Bus 이벤트로 전환 가능 |
| 클라이언트 수 | Unix socket 1:1 | 멀티 클라이언트 필요 시 accept loop + Vec<Stream> |
| Config 재로딩 | 시작 시 1회 | SIGHUP으로 hot reload |
| Android 테스트 | 미검증 | sysfs 경로 차이, D-Bus 부재 환경 테스트 필요 |

## 16. 관련 문서

- [20. D-Bus IPC Specification](20_dbus_ipc_spec.md) — 프로토콜 명세, 신호 인자, 히스테리시스 설계 원본
- [21. Resilience Architecture](21_resilience_architecture.md) — Engine 측 신호 수신 및 처리 아키텍처
- [22. Resilience Integration](22_resilience_integration.md) — generate.rs에서의 신호 반응 구현
- [24. Resilience Usage Guide](24_resilience_usage_guide.md) — Resilience 시스템 사용법
