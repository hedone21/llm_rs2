# Manager Overview

> **TL;DR**: Manager는 시스템 리소스를 모니터링하고 LLM Engine에 적응형 제어 디렉티브를 발행하는 독립 프로세스이다. Monitor(센서 수집) → Policy(PI 제어 + 조합 선택) → Emitter(디렉티브 전송) 3-layer 파이프라인으로 구성되며, 4+1개 Monitor 스레드와 1개 메인(정책) 스레드로 동작한다. Engine 없이 독립 시작·종료가 가능하고, Manager 장애가 Engine 추론 루프에 영향을 주지 않는다.

## 1. Purpose and Scope

이 문서는 Manager 프로세스의 **목적, 3-layer 아키텍처, 각 레이어 서브시스템, 스레딩 모델, 설정 체계, CLI**를 정의한다. Manager가 *무엇을 하는지*와 *어떤 부분으로 구성되는지*를 상위 수준에서 정의한다.

**이 파일이 명세하는 것:**

- Manager 프로세스의 책임과 독립성 보증
- Monitor / Policy / Emitter 3-layer 파이프라인 구조
- 각 레이어의 서브시스템 책임과 인터페이스 개요
- 스레딩 모델과 통신 토폴로지
- 초기화 시퀀스와 설정 로딩 우선순위
- CLI 인터페이스

**이 파일이 명세하지 않는 것:**

- 상태 머신 전이 테이블 → `21-manager-state.md`
- PI Controller, ActionSelector, ReliefEstimator 알고리즘 상세 → `22-manager-algorithms.md`
- 설정 스키마 상세, SystemSignal 필드 정의 → `23-manager-data.md`
- 와이어 포맷, 메시지 필드 → `10-protocol.md`, `11-protocol-messages.md`
- 상호작용 시퀀스 → `12-protocol-sequences.md`

## 2. Definitions

| 용어 | 정의 |
|------|------|
| **Manager** | `llm_manager` 크레이트에서 빌드되는 독립 프로세스. 리소스 모니터링과 정책 결정을 담당한다. |
| **Monitor Layer** | Manager 첫 번째 계층. 4+1개 병렬 스레드에서 OS 센서를 읽어 SystemSignal을 생성한다. |
| **Policy Layer** | Manager 두 번째 계층. 메인 스레드에서 PI Controller → Supervisory → Action Selector 파이프라인을 순차 실행한다. |
| **Emitter Layer** | Manager 세 번째 계층. Policy가 생성한 EngineDirective를 Engine에 전송한다. |
| **SystemSignal** | Monitor가 생성하는 도메인별 시스템 상태 메시지. 4종: MemoryPressure, ThermalAlert, ComputeGuidance, EnergyConstraint. |
| **ThresholdEvaluator** | 히스테리시스 기반 임계값 평가기. 원시 측정값 → Level (Normal/Warning/Critical/Emergency) 변환. **D-Bus Emitter 전용**으로, D-Bus 전송 시 raw 값에서 Level을 산출하여 와이어 메시지에 포함한다. 내부 SystemSignal에는 Level이 포함되지 않으며, Policy Layer에서도 사용하지 않는다. |
| **HierarchicalPolicy** | PolicyStrategy trait 구현체. PI + Supervisory + ActionSelector + ReliefEstimator를 조합한다. |
| **ActionRegistry** | 액션 메타데이터 저장소. 종류, 가역성, 파라미터 범위, 배타 그룹, 기본 비용을 관리한다. |

## 3. Specification

### 3.1 Manager Process Overview [MGR-010 ~ MGR-013]

**[MGR-010]** Manager는 독립 프로세스이다. Engine 없이 시작, 실행, 종료가 가능하다. *(MUST)* (SYS-050~054 참조)

**[MGR-011]** Manager 장애는 Engine 추론 루프에 영향을 주지 않는다. *(MUST)* (INV-005 재확인)

**[MGR-012]** Manager는 3-layer 파이프라인을 따른다: Monitor → Policy → Emitter. *(MUST)* (SYS-086 재확인)

```
┌─────────────────────────────────────────────────┐
│                   Manager Process                │
│                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │ Monitor  │───→│  Policy  │───→│ Emitter  │───→ Engine
│  │  Layer   │    │  Layer   │    │  Layer   │   │
│  └──────────┘    └──────────┘    └──────────┘   │
│  (4+1 threads)   (main thread)   (main thread)  │
└─────────────────────────────────────────────────┘
```

**[MGR-013]** Manager는 단일 Engine과 1:1로 통신한다. *(MUST)* (SYS-093 참조)

### 3.2 Monitor Layer [MGR-014 ~ MGR-022]

**[MGR-014]** Monitor Layer는 4+1개 독립 스레드로 구성된다. 각 Monitor는 Monitor trait을 구현한다. *(MUST)* (SYS-087 확장)

| Monitor | 도메인 | SystemSignal |
|---------|--------|-------------|
| MemoryMonitor | memory | MemoryPressure |
| ThermalMonitor | thermal | ThermalAlert |
| ComputeMonitor | compute | ComputeGuidance |
| EnergyMonitor | energy | EnergyConstraint |
| ExternalMonitor | (외부) | 임의 SystemSignal |

**[MGR-015]** Monitor trait 인터페이스는 다음 3개 메서드로 구성된다: *(MUST)*

| 메서드 | 시그니처 | 설명 |
|--------|---------|------|
| `run` | `run(tx: Sender<SystemSignal>, shutdown: Arc<AtomicBool>) -> Result` | 센서 폴링 루프. tx로 신호 전송. shutdown 시 루프 탈출. |
| `initial_signal` | `initial_signal() -> Option<SystemSignal>` | 초기 상태 신호. None 반환 시 해당 도메인의 초기 신호 없음. |
| `name` | `name() -> &str` | Monitor 이름 문자열. |

**[MGR-016]** MemoryMonitor는 `/proc/meminfo`를 읽어 MemoryPressure 신호를 생성한다. *(MUST)*

- Raw 데이터: `available_bytes`, `total_bytes`
- `reclaim_target_bytes`: Level 보조 계산 (Normal=0, Warning=5%, Critical=10%, Emergency=20%)
- Policy는 `available_bytes/total_bytes`에서 직접 압력을 계산한다 (MGR-ALG-013a)

**[MGR-017]** ThermalMonitor는 `/sys/class/thermal/`을 읽어 ThermalAlert 신호를 생성한다. *(MUST)*

- Raw 데이터: `temperature_mc` (밀리섭씨), `throttle_ratio`
- `zone_types` 필터링, 다중 zone 중 최대 온도 기준
- Policy는 `temperature_mc`에서 직접 압력을 계산한다 (PI Controller, MGR-ALG-014)

**[MGR-018]** ComputeMonitor는 `/proc/stat` CPU delta를 계산하여 ComputeGuidance 신호를 생성한다. *(MUST)*

- 방향: Ascending (CPU 사용률이 높을수록 위험)
- Emergency 레벨 없음 (emergency threshold = f64::MAX)
- `gpu_usage_pct`는 GPU 사용률을 보고한다. 구현이 필요하다 *(MUST)*. OpenCL 환경에서 GPU busy 비율을 수집하여 ComputeGuidance 신호에 포함해야 한다.

**[MGR-019]** EnergyMonitor는 `/sys/class/power_supply/`를 읽어 EnergyConstraint 신호를 생성한다. *(MUST)*

- Raw 데이터: `battery_pct` (배터리 잔량 %), `power_budget_mw` (전력 예산)
- `ignore_when_charging=true` 시 충전 중 신호를 발행하지 않는다
- Policy는 `battery_pct`에서 직접 압력을 계산한다 (compute PI 보조 기여, MGR-ALG-015)

**[MGR-020]** ExternalMonitor는 stdin 또는 Unix socket에서 JSON Lines로 SystemSignal을 수신한다. 연구 및 테스트 용도이다. *(MAY)*

**[MGR-021]** ThresholdEvaluator는 D-Bus Emitter 전용 도구이다. D-Bus 전송 경로에서 raw 센서 값을 Level로 변환하여 D-Bus 와이어 메시지(`11-protocol-messages.md` MSG-100~104)에 포함한다. 내부 SystemSignal에는 Level이 포함되지 않는다. *(SHOULD)*

- Direction: Ascending (높을수록 위험) / Descending (낮을수록 위험)
- 상태 전이 테이블 → `21-manager-state.md` MGR-067~073

> **참고 (non-normative)**: Monitor의 유일한 책임은 raw 센서 데이터 수집과 전달이다. 심각도 평가(Level)는 Monitor의 책임이 아니다. D-Bus 전송이 필요한 경우에만 Emitter Layer의 ThresholdEvaluator가 Level을 산출한다. **Policy Layer는 raw 필드(available_bytes, temperature_mc, cpu_usage_pct, battery_pct)에서 직접 압력을 계산한다.** 이 분리는 Monitor를 단순하게 유지하고, 압력 계산 전략을 Policy에서 독립적으로 교체할 수 있게 한다.

**[MGR-022]** 기본 폴링 주기는 `poll_interval_ms = 1000ms`이다. Monitor별 개별 설정이 가능하다. *(SHOULD)*

- **MemoryMonitor는 100ms 이하 주기로 폴링해야 한다. *(MUST)*** OOM은 초 단위로 발생하므로 1000ms 주기로는 감지가 늦다. 임계값 기반 직접 매핑(MGR-024)과 결합하여 100ms 이내에 압력 반영 → Supervisory 모드 전이 → Directive 발행이 가능해야 한다.

### 3.3 Policy Layer [MGR-023 ~ MGR-030]

**[MGR-023]** Policy Layer는 메인 스레드에서 순차 실행된다. PolicyStrategy trait으로 추상화되며, HierarchicalPolicy(Rust 내장)와 LuaPolicy(스크립트 기반, MGR-049)가 구현체이다. *(MUST)* (SYS-088 확장)

**[MGR-024]** 도메인별 압력 계산 — 각 도메인은 독립적인 전략으로 원시 측정값 [0, 1]을 연속 압력 [0, 1]로 변환한다. *(MUST)*

| 도메인 | 전략 | 근거 |
|--------|------|------|
| Compute | PI Controller | 연산 부하 변동이 점진적. P항+I항 평활화가 noise 제거에 효과적. |
| Thermal | PI Controller | 열 관성이 크므로 평활화가 적합. |
| Memory | 임계값 기반 직접 매핑 | OOM은 초 단위로 발생. PI 평활화의 지연이 치명적. 즉각 반응 필요. |

- **PI Controller** (Compute, Thermal): P항 + I항, anti-windup (integral_clamp + can_act), gain scheduling (GainZone)
- **임계값 기반** (Memory): Monitor의 측정값 또는 Level을 압력에 직접 매핑. PI 평활화를 거치지 않는다.
- 알고리즘 상세 → `22-manager-algorithms.md`

> **참고 (non-normative)**: 도메인별 압력 계산 전략은 **전략 패턴으로 교체 가능**하다. 현재 구성(PI 2개 + 임계값 1개)은 하나의 설정이며, 향후 모든 도메인에 PI를 적용하거나, PID로 교체하거나, 학습 기반 전략으로 전환할 수 있다. PolicyStrategy trait(MGR-023)이 정책 전체를, 도메인별 전략이 개별 압력 계산을 각각 추상화한다.

**[MGR-025]** Supervisory — PressureVector → OperatingMode (Normal/Warning/Critical) 변환을 수행한다. *(MUST)*

- Peak pressure (PressureVector.max()) 기반
- 에스컬레이션 즉시 (다단계 건너뛰기 가능)
- 디에스컬레이션 hold_time 후 1단계씩 하강
- 상태 머신 전이 테이블 → `21-manager-state.md` MGR-050~059
- 알고리즘 상세 → `22-manager-algorithms.md`

**[MGR-026]** Action Selector — Cross-domain 조합 최적화를 수행한다. *(MUST)*

- 전수 탐색 (2^N)
- 비용 최소화 + 제약 만족 (latency budget, 배타 그룹, 모드별 허용 액션)
- Stateless
- 알고리즘 상세 → `22-manager-algorithms.md`

**[MGR-027]** ReliefEstimator — 액션 효과를 온라인 학습으로 예측한다. *(MUST)*

- ReliefEstimator trait. 현재 구현체: OnlineLinearEstimator
- 온라인 선형 회귀 (RLS, forgetting factor=0.995)
- 13차원 FeatureVector → 4차원 ReliefVector (compute, memory, thermal, latency)
- 액션별 독립 모델
- 세션 간 모델 영속화
- 상태 수명 → `21-manager-state.md` MGR-083~087
- 알고리즘 상세 → `22-manager-algorithms.md`

**[MGR-028]** ActionRegistry — 액션 메타데이터 저장소이다. `from_config()`로 초기화되며, 설정에 포함된 액션만 등록한다 (최대 8개). *(MUST)* (SYS-095~099 확장)

등록 가능 액션 8종:

| ActionId | 종류 | 파라미터 | 파라미터 범위 |
|----------|------|---------|-------------|
| SwitchHw | Lossless | (없음) | -- |
| Throttle | Lossless | delay_ms | [0.0, 100.0] |
| KvOffloadDisk | Lossless | (없음) | -- |
| KvEvictSliding | Lossy | keep_ratio | [0.3, 0.9] |
| KvEvictH2o | Lossy | keep_ratio | [0.3, 0.9] |
| KvMergeD2o | Lossy | keep_ratio | [0.3, 0.9] |
| KvQuantDynamic | Lossy | target_bits | [4.0, 8.0] |
| LayerSkip | Lossy | skip_layers | [1.0, 8.0] |

각 액션은 종류(Lossless/Lossy), 가역 여부, 파라미터 범위, 배타 그룹, 기본 비용을 메타데이터로 보유한다.

**[MGR-029]** EnergyConstraint 처리 — 별도 PI 인스턴스 없이 compute PI에 보조 기여한다. raw 값(`battery_pct`)에서 직접 측정값을 산출한다. *(MUST)*

```
energy_measurement = clamp(1.0 - battery_pct / 100.0, 0, 1) * 0.5
combined = max(pressure.compute, energy_measurement)
compute_pressure = pi_compute.update(combined, dt)
```

> 전력 상태의 시간 스케일(분~시간)이 PI의 초 단위 제어와 상이하므로 별도 도메인 대신 compute 보조 신호로 반영한다. 0.5 가중치는 energy가 compute를 과도하게 지배하지 않도록 한다. 알고리즘 상세 → `22-manager-algorithms.md` MGR-ALG-015.

**[MGR-030]** Observation Window — 액션 적용 후 OBSERVATION_DELAY_SECS (3.0초) 관찰 대기를 수행한다. *(MUST)*

- 실측 relief = pressure_before - pressure_current
- ReliefEstimator 온라인 업데이트
- 시퀀스 상세 → `12-protocol-sequences.md` SEQ-050~054

### 3.4 Emitter Layer [MGR-031 ~ MGR-034]

**[MGR-031]** Emitter trait 인터페이스: *(MUST)*

| 메서드 | 설명 |
|--------|------|
| `emit(signal)` | SystemSignal 전송 |
| `emit_initial(signals)` | 초기 신호 일괄 전송 (trait에 정의되나 현재 초기화 흐름에서 미사용 — MGR-045 참조) |
| `emit_directive(directive)` | EngineDirective 전송 |
| `name()` | Emitter 이름 |

`emit_directive()`의 기본 구현은 로깅만 수행한다 (no-op). (SYS-089 확장)

**[MGR-032]** EngineReceiver trait — `try_recv() -> Option<EngineMessage>`, `is_connected()`. 양방향 전송(Unix/TCP)에서 Engine 메시지 수신에 사용한다. *(MUST)*

**[MGR-033]** 양방향 구현체 — UnixSocketChannel(기본)과 TcpChannel은 Emitter + EngineReceiver를 겸용한다. ConnectionState 3-state (Listening, Connected, Disconnected)로 연결을 관리한다. *(MUST)*

- 상태 전이 테이블 → `21-manager-state.md` MGR-060~066

**[MGR-034]** 단방향 구현체 — DbusEmitter는 D-Bus System Bus를 통해 SystemSignal만 전송한다. EngineDirective는 로깅만 수행한다. Feature-gated (`dbus`). *(MAY)*

### 3.5 Main Loop [MGR-035 ~ MGR-038]

**[MGR-035]** 메인 루프 구조는 다음 의사코드를 따른다: *(MUST)*

```
loop:
    if SHUTDOWN:
        break

    // Engine 메시지 우선 drain
    while engine_message = receiver.try_recv():
        handle_engine_message(engine_message)

    // Monitor 신호 수신 (50ms timeout)
    match monitor_rx.recv_timeout(50ms):
        signal =>
            directive = policy.process_signal(signal)
            if directive is Some:
                emitter.emit_directive(directive)
        timeout =>
            continue
```

(SEQ-030~035 참조)

**[MGR-036]** 처리 우선순위 — Engine 메시지를 먼저 drain한 후 Monitor 신호를 처리한다. 코드 순서에 의한 암묵적 우선순위이다. *(MUST)*

> **참고 (non-normative)**: 최신 Engine 상태(FeatureVector, available_actions)로 압력 계산 정확도를 보장하기 위함.

**[MGR-037]** Engine 메시지 처리: *(MUST)*

| 메시지 | 처리 |
|--------|------|
| Heartbeat | `update_engine_state()` — FeatureVector 갱신, available_actions 파싱 |
| Response | 로깅 |
| Capability | 로깅 |

**[MGR-038]** Monitor 신호 처리 — `process_signal()` 호출 후 Directive 생성 시 `emit_directive()`로 전송한다. Directive 없으면 continue. *(MUST)*

### 3.6 Threading Model [MGR-039 ~ MGR-041]

**[MGR-039]** Manager 스레드 구성: *(MUST)*

| 스레드 | 역할 | 수명 | 통신 |
|--------|------|------|------|
| Main | 정책 루프 (Policy + Emitter) | 프로세스 수명 | mpsc::Receiver\<SystemSignal\> |
| MemoryMonitor | 메모리 센서 폴링 | 프로세스 수명 | mpsc::Sender\<SystemSignal\> |
| ThermalMonitor | 온도 센서 폴링 | 프로세스 수명 | mpsc::Sender\<SystemSignal\> |
| ComputeMonitor | CPU/GPU 사용량 폴링 | 프로세스 수명 | mpsc::Sender\<SystemSignal\> |
| EnergyMonitor | 배터리/전력 폴링 | 프로세스 수명 | mpsc::Sender\<SystemSignal\> |
| ExternalMonitor | 외부 신호 수신 | 프로세스 수명 (enabled 시) | mpsc::Sender\<SystemSignal\> |
| Reader | Engine 메시지 수신 (양방향 시) | 연결 수명 | mpsc::SyncSender\<EngineMessage\>(cap=64) |

**[MGR-040]** 모든 Monitor는 독립 OS 스레드에서 실행된다. 공유 상태 없이 mpsc 채널로만 통신한다. *(MUST)* (INV-013 재확인)

**[MGR-041]** 종료 프로토콜: *(MUST)*

1. SIGINT 또는 SIGTERM 수신
2. SHUTDOWN AtomicBool 설정
3. 메인 루프 탈출
4. Relief 모델 저장 (`save()`)
5. Monitor 스레드 join
6. 프로세스 종료

### 3.7 Initialization [MGR-042 ~ MGR-045]

**[MGR-042]** 설정 로딩 우선순위 (높은 순서부터): *(MUST)*

1. `--policy-config` CLI 인수로 지정된 별도 정책 설정 파일
2. `config.toml`의 `[policy]` 섹션
3. `PolicyConfig::default()`

**[MGR-043]** Transport 생성 — CLI `--transport` 인수에 따라 결정: *(MUST)*

| 값 | 생성되는 구현체 |
|----|---------------|
| `unix:<path>` | UnixSocketChannel |
| `tcp:<host:port>` | TcpChannel |
| `dbus` | DbusEmitter |

**[MGR-044]** Monitor 생성 — 설정의 `enabled` 플래그에 따라 활성화한다. 비활성 Monitor는 스레드를 생성하지 않는다. *(MUST)*

**[MGR-045]** 초기 신호 방출 — 각 Monitor의 `initial_signal()`을 수집한다. 수집된 초기 신호를 순차적으로 `policy.process_signal()`에 투입하여 초기 상태 기반 Directive를 생성한다. Directive가 생성되면 `emit_directive()`로 전송한다. 이를 통해 Manager 시작 시점의 리소스 상태가 정책 파이프라인에 반영된다. *(SHOULD)*

### 3.8 CLI Interface [MGR-046 ~ MGR-048]

**[MGR-046]** 바이너리명은 `llm_manager`이다 (Cargo.toml 기준). *(MUST)*

**[MGR-047]** CLI 인수: *(MUST)*

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `--config` | `/etc/llm-manager/config.toml` | 메인 설정 파일 경로 |
| `--transport` | `dbus` | 전송 매체 |
| `--client-timeout` | `60` | Engine 연결 대기 시간 (초) |
| `--policy-config` | (없음) | 별도 정책 설정 파일 경로 |
| `--policy-script` | (없음) | Lua 정책 스크립트 경로 (MGR-049). 지정 시 LuaPolicy 사용. |

**[MGR-048]** Relief 모델 영속화 — 종료 시 `save()`, 시작 시 `load()`를 수행한다. 경로: `ReliefModelConfig.storage_dir` (기본 `~/.llm_rs/models`). *(SHOULD)*

**[MGR-049]** Lua 정책 스크립팅 — `--policy-script <path>` 지정 시 Lua 5.4 VM을 임베딩하여 정책 결정을 Lua 스크립트에 위임한다. *(MAY)* Feature-gated (`lua`).

- Lua VM은 Manager 시작 시 1회 생성되며, 세션 동안 상태를 유지한다 (글로벌 변수, EMA 등).
- 스크립트는 `decide(ctx)` 함수를 정의해야 한다 (MUST). 이 함수는 SystemSignal 수신마다 호출된다.
- `ctx` 입력: `ctx.engine` (EngineStatus heartbeat 필드), `ctx.active` (현재 활성 액션 목록).
- `sys.*` 헬퍼: `sys.read(path)`, `sys.meminfo()`, `sys.thermal(zone)`, `sys.gpu_busy()`, `sys.gpu_freq()`, `sys.cpu_freq(n)` — 시스템 센서를 Lua에서 직접 읽는다.
- 반환: EngineCommand 테이블 배열. 빈 배열이면 액션 없음.
- 에러 처리: Lua 런타임 에러 시 `log::error`로 기록하고 빈 액션을 반환한다. Manager는 crash하지 않는다.
- 메모리 제한: 4MB. 샌드박스: TABLE, STRING, MATH 라이브러리만 허용 (IO, OS 차단).
- `--policy-script` 미지정 시 기존 HierarchicalPolicy(MGR-023~028)가 사용된다.

## 4. Alternative Behavior

- **Engine 미연결 상태**: Policy 루프는 계속 실행된다. `emit_directive()` 호출은 skip되어 비치명적 Ok를 반환한다. 압력 계산, 모드 전이, 관찰 업데이트는 계속 수행된다.

- **Monitor 전체 비활성**: 모든 Monitor의 `enabled=false` 시 SystemSignal이 발생하지 않는다. `recv_timeout` 50ms마다 timeout → continue 반복. Directive는 미생성된다.

- **D-Bus 모드**: Emitter만 활성 (단방향). EngineReceiver 없음. Heartbeat/Response 수신 불가. Policy는 Monitor 신호만으로 동작한다.

## 5. Constraints

**[MGR-C01]** Manager는 `std::thread`와 `std::sync::mpsc`만 사용한다. async 런타임(tokio, async-std 등)을 사용하지 않는다. *(MUST NOT)*

**[MGR-C02]** Manager의 Shared 크레이트 의존은 IPC 타입에 한정된다. *(MUST)* (INV-010, INV-011 참조)

## 6. Examples

### 6.1 Manager 시작 시퀀스 trace

1. `config.toml` 로딩 → PolicyConfig 구성
2. UnixSocketChannel 생성 (bind + listen)
3. Monitor 스레드 4+1개 시작
4. `wait_for_client(60s)`
5. Engine 연결 → Capability 수신
6. 메인 루프 진입

### 6.2 Manager 정상 운영 1초 trace

```
t=0ms:   drain engine → Heartbeat 수신 → update_engine_state()
t=0ms:   recv_timeout(50ms) → MemoryPressure Normal
t=0ms:   process_signal() → PI pressure 0.12 → Normal → no action
t=50ms:  timeout → continue
t=100ms: timeout → continue
  ...
t=1000ms: drain engine → 다음 Heartbeat
```

## 7. Rationale (non-normative)

### 왜 3-layer인가

Monitor(센서)와 Policy(결정)와 Emitter(전달)를 분리하여 각 레이어를 독립 테스트 및 교체 가능하게 한다. Monitor는 스레드 격리, Policy는 trait 추상화(PolicyStrategy), Emitter는 전송 매체 교체를 지원한다.

### 왜 EnergyMonitor가 별도 PI 없이 compute에 합산되는가

전력 상태는 시간 스케일(분~시간)이 PI의 초 단위 제어와 상이하다. 별도 도메인 분리 시 4D PressureVector가 필요하나, ActionSelector 조합 공간이 불필요하게 확대된다. Compute의 보조 신호로 반영하면 기존 3D 구조를 유지하면서 전력 압력을 반영할 수 있다.

### 왜 std::thread이고 async가 아닌가

Monitor는 blocking `/proc` 읽기, Policy는 순차 CPU 연산이다. I/O 다중화(epoll/io_uring) 이점이 없다. 스레드 수가 10 미만이므로 OS 스레드 오버헤드는 미미하다.

### 왜 종료 시 relief 모델을 저장하는가

동일 디바이스 + 동일 모델에서 relief 특성이 세션 간 유사하다. 학습 누적으로 cold-start를 완화한다.
