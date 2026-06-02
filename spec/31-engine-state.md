# Engine State Machines

> **TL;DR**: Engine 내부 5개 상태 머신의 전이 조건을 빠짐없이 열거한다. (1) OperatingMode 4-state (Normal/Degraded/Minimal/Suspended): worst-wins 순수 함수, Strategy 경로(D-Bus 레거시) 전용. (2) EngineState 3-state (Idle/Running/Suspended): Directive 경로의 프로토콜 수준 상태. (3) CommandExecutor: 11종 EngineCommand를 ExecutionPlan으로 변환, superseding 규칙, active/available_actions 관리. (4) ExecutionPlan 수명: 1회성 생성-소비-폐기. (5) ResilienceManager: D-Bus 레거시 경로의 Strategy 반응 + resolve_conflicts() 7규칙.

## 1. Purpose and Scope

이 문서는 Engine 내부의 **모든 상태 머신**의 상태, 전이 조건, 불변식을 빠짐없이 열거한다.

**이 파일이 명세하는 것:**

- OperatingMode FSM (Strategy 경로)
- EngineState FSM (Directive 경로)
- CommandExecutor 내부 상태와 EngineCommand 처리 규칙
- ExecutionPlan 수명주기와 소비 순서
- ResilienceManager 상태와 Strategy 반응 (D-Bus 레거시)
- `resolve_conflicts()` 우선순위 규칙

**이 파일이 명세하지 않는 것:**

- Engine 아키텍처 개요 → `30-engine.md`
- Manager 상태 머신 → `21-manager-state.md`
- Transport 연결 상태 → `10-protocol.md`
- 알고리즘 상세 (eviction 수식, QCF 계산 등) → `32-engine-algorithms.md`

## 2. Definitions

| 용어 | 정의 |
|------|------|
| **FSM** | Finite State Machine. 유한 상태와 전이 함수의 쌍. |
| **Transition Guard** | 전이가 발생하기 위한 전제 조건. |
| **worst-wins** | 4종 Level 중 최댓값이 결과 모드를 결정하는 규칙. |
| **Superseding** | 동일 `poll()` 내 복수 Directive에서 후행이 선행의 plan 필드를 덮어쓰는 규칙. |

## 3. Specification

### 3.1 OperatingMode FSM [ENG-ST-010 ~ ENG-ST-015]

**[ENG-ST-010]** 상태 정의 -- 4개 상태. *(MUST)*

| 상태 | 설명 |
|------|------|
| Normal | 정상 운영. 모든 기능 사용 가능 |
| Degraded | 일부 리소스 제약. 경량 적응 액션 적용 (eviction, throttle) |
| Minimal | 최소 리소스. 공격적 제약 (CPU 전환, 토큰 제한, 강한 eviction) |
| Suspended | 추론 일시 정지. Manager Resume 명령 또는 Level 복구 대기 |

**[ENG-ST-011]** 전이 함수 `from_levels()` -- worst-wins 규칙: *(MUST)*

```
function from_levels(memory: Level, compute: Level, thermal: Level, energy: Level) -> OperatingMode:
    worst = max(memory, compute, thermal, energy)
    match worst:
        Normal    -> Normal
        Warning   -> Degraded
        Critical  -> Minimal
        Emergency -> Suspended
```

`from_levels()`는 순수 함수이다. 이전 상태에 의존하지 않으며, 4종 Level의 worst로 다음 상태가 결정된다. 따라서 모든 상태 간 전이가 가능하다 (N^2 - N = 12 전이).

**[ENG-ST-012]** Level에서 OperatingMode로의 매핑: *(MUST)*

| worst Level | OperatingMode |
|-------------|---------------|
| Normal | Normal |
| Warning | Degraded |
| Critical | Minimal |
| Emergency | Suspended |

**[ENG-ST-013]** 전이 테이블 (완전 열거): *(MUST)*

| 출발 상태 | 도착 상태 | 전이 조건 |
|----------|----------|----------|
| Normal | Degraded | any Level >= Warning |
| Normal | Minimal | any Level >= Critical |
| Normal | Suspended | any Level = Emergency |
| Degraded | Normal | all Level = Normal |
| Degraded | Minimal | any Level >= Critical |
| Degraded | Suspended | any Level = Emergency |
| Minimal | Normal | all Level = Normal |
| Minimal | Degraded | worst Level = Warning |
| Minimal | Suspended | any Level = Emergency |
| Suspended | Normal | all Level = Normal |
| Suspended | Degraded | worst Level = Warning |
| Suspended | Minimal | worst Level = Critical |

**[ENG-ST-014]** 사용 위치: *(MUST)*

| 사용자 | 경로 | 호출 시점 |
|--------|------|----------|
| ResilienceManager | D-Bus/Strategy 경로 | `process_signal()`에서 Level 갱신 후 매번 재계산 |
| CommandExecutor | Directive 경로 | **사용하지 않음** |

> OperatingMode는 Strategy 경로(ResilienceManager)에서만 사용된다. Directive 경로(CommandExecutor)에서는 Manager가 이미 모드를 고려한 Directive를 전송하므로 Engine이 별도로 OperatingMode를 계산할 필요가 없다.

**[ENG-ST-015]** Emergency 자율 전이 규칙 (SYS-055, SYS-085a 참조): *(MUST)*

D-Bus 경로에서 Emergency level signal 수신 시:

1. `DbusTransport.signal_to_manager_message()`가 `EngineCommand::Suspend`로 변환한다
2. `CommandExecutor.apply_command(Suspend)`가 `engine_state = Suspended`를 설정한다
3. `ExecutionPlan.suspended = true`로 Inference Loop가 중단된다

이는 Manager의 명시적 Suspend 명령 없이 Engine이 자율적으로 Suspended 상태에 진입하는 것이다. SYS-055 "Emergency 신호 시 추론 즉시 중단" 요구사항을 D-Bus 경로에서 충족한다.

### 3.2 EngineState FSM [ENG-ST-020 ~ ENG-ST-025]

**[ENG-ST-020]** 상태 정의 -- 3개 상태. *(MUST)*

| 상태 | 설명 |
|------|------|
| Idle | 추론 시작 전 또는 추론 완료 후 |
| Running | 추론 진행 중 (decode loop 활성) |
| Suspended | 추론 일시 정지 (Manager 명령에 의해) |

**[ENG-ST-021]** 전이 테이블 (완전 열거): *(MUST)*

| 출발 | 도착 | 트리거 | 코드 위치 |
|------|------|--------|----------|
| Idle | Running | `executor.set_running()` | `generate.rs`: decode loop 진입 시 |
| Running | Suspended | `EngineCommand::Suspend` 수신 | `executor.rs`: `apply_command(Suspend)` |
| Suspended | Running | `EngineCommand::Resume` 수신 | `executor.rs`: `apply_command(Resume)` |
| Running | Idle | 추론 완료 | `generate.rs`: decode loop 종료 |
| Idle | Idle | 자기 전이 (변화 없음) | -- |
| Suspended | Idle | 프로세스 종료 | -- |

**[ENG-ST-022]** 금지 전이: *(SHOULD NOT)*

| 전이 | 이유 |
|------|------|
| Idle -> Suspended | Idle 상태에서 Suspend 명령을 받아도 추론 루프가 없으므로 효과가 없다. 코드상 전이는 일어나지만 의미 없는 전이이다. |

**[ENG-ST-023]** Heartbeat 내 state 필드: *(MUST)*

- `Heartbeat.state`는 EngineState를 Manager에 보고한다
- Manager는 이 값으로 Engine이 Idle/Running/Suspended인지 판단한다

**[ENG-ST-024]** EngineState vs OperatingMode 관계: *(MUST)*

| 항목 | EngineState | OperatingMode |
|------|-------------|---------------|
| 상태 수 | 3 (Idle/Running/Suspended) | 4 (Normal/Degraded/Minimal/Suspended) |
| 사용 위치 | CommandExecutor, Heartbeat, Manager | ResilienceManager |
| 경로 | Directive 경로 (프로토콜 수준) | Strategy 경로 (D-Bus 레거시) |

두 상태 머신은 **독립적**이다. `OperatingMode.Suspended`와 `EngineState.Suspended`가 의미적으로 대응하지만 직접 연결되지 않는다.

**[ENG-ST-025]** EngineState는 `shared` 크레이트에 정의되어 Manager-Engine 프로토콜에서 공유된다. OperatingMode는 Engine 내부(`resilience/state.rs`)에 정의되어 프로토콜에 노출되지 않는다. *(MUST)*

### 3.3 CommandExecutor State [ENG-ST-030 ~ ENG-ST-039]

**[ENG-ST-030]** 내부 상태 필드: *(MUST)*

| 필드 | 타입 | 초기값 | 설명 |
|------|------|--------|------|
| engine_state | EngineState | Idle | 현재 Engine 상태 |
| compute_level | ResourceLevel | Normal | (deprecated) 레거시 호환용 |
| memory_level | ResourceLevel | Normal | (deprecated) 레거시 호환용 |
| active_device | String | CLI `--backend` 값 | 현재 활성 디바이스 |
| throttle_delay_ms | u64 | 0 | 현재 토큰 간 지연 |
| active_actions | Vec&lt;String&gt; | [] | 현재 활성 액션 이름 목록 |
| throughput_ema | f32 | 0.0 | 토큰 처리율 EMA (alpha=0.1) |
| last_token_time | Option&lt;Instant&gt; | None | 마지막 토큰 생성 시각 |
| tokens_generated | usize | 0 | 총 생성 토큰 수 |
| last_heartbeat | Instant | now() | 마지막 Heartbeat 전송 시각 |
| heartbeat_interval | Duration | 1000ms | Heartbeat 주기 |

**[ENG-ST-031]** active_actions 관리 규칙: *(MUST)*

| EngineCommand | active_actions 동작 |
|---------------|-------------------|
| Throttle { delay_ms > 0 } | "throttle" 추가 (중복 방지) |
| Throttle { delay_ms = 0 } | "throttle" 제거 |
| LayerSkip | "weight.skip" 추가 |
| KvEvictH2o | "kv.evict_h2o" 추가 |
| KvEvictSliding | "kv.evict_sliding" 추가 |
| KvQuantDynamic | "kv.quant_dynamic" 추가 |
| RestoreDefaults | `active_actions.clear()` |
| Resume | 초기화하지 않음 (compute_level/memory_level만 리셋) |
| KvStreaming | "kv.evict_streaming" 추가 |
| SwitchHw, PrepareComputeUnit, Suspend | active_actions 변경 없음 |

**[ENG-ST-032]** available_actions 계산 규칙: *(MUST)*

매 Heartbeat 전송 시 `compute_available_actions(eviction_policy, kv_dtype, has_secondary)` 호출:

- 기본: `["throttle", "switch_hw", "weight.skip"]`
- `eviction_policy != "none"` 이면: `+ ["kv.evict_h2o", "kv.evict_sliding", "kv.evict_streaming"]`
- `kv_dtype.starts_with('q')` 이면: `+ ["kv.quant_dynamic"]`
- secondary GGUF/AUF 파일이 로드된 경우(`has_secondary == true`) 이면: `+ ["swap_weights"]`

Capability 송출(`send_capability`)과 Heartbeat의 `available_actions`는 같은 조건(`args.secondary_gguf.is_some()`)을 공유해야 한다.

**[ENG-ST-033]** EngineCommand 처리 결과 상태 전이 -- 13종 전수 열거: *(MUST)*

| 명령 | 내부 상태 변경 | ExecutionPlan 필드 | CommandResult |
|------|--------------|-------------------|---------------|
| Throttle { delay_ms } | throttle_delay_ms 갱신 | plan.throttle_delay_ms | Ok |
| LayerSkip { skip_ratio } | -- | plan.layer_skip = Some(ratio) | Ok |
| KvEvictH2o { keep_ratio } | -- | plan.evict = Some(EvictPlan { H2o, ratio, Critical }) | Ok |
| KvEvictSliding { keep_ratio } | -- | plan.evict = Some(EvictPlan { Sliding, ratio, Critical }) | Ok |
| KvMergeD2o { keep_ratio } | -- | plan.evict = Some(EvictPlan { D2o, ratio, Critical }) | Ok |
| KvStreaming { sink_size, window_size } | -- | plan.evict = Some(EvictPlan { Streaming, 0.0, Critical, streaming_params: Some(StreamingParams { sink_size, window_size }) }) | Ok |
| KvQuantDynamic { target_bits } | -- | plan.kv_quant_bits = Some(bits) | Ok |
| RequestQcf | -- | plan.request_qcf = true | Ok (QcfEstimate를 별도 EngineMessage로 전송, SEQ-095~098) |
| RestoreDefaults | throttle=0, compute/memory=Normal, active_actions=[] | plan.restore_defaults = true, plan.throttle_delay_ms = 0 | Ok |
| SwitchHw { device } | -- | plan.switch_device = Some(device) | Ok |
| PrepareComputeUnit { device } | -- | plan.prepare_device = Some(device) | Ok |
| Suspend | engine_state = Suspended | plan.suspended = true | Ok |
| Resume | engine_state = Running, compute/memory=Normal, throttle=0 | plan.resumed = true, plan.throttle_delay_ms = 0 | Ok |

**[ENG-ST-034]** `poll()` 실행 흐름: *(MUST)*

```
function poll(kv_snap: &KVSnapshot) -> ExecutionPlan:
    plan = ExecutionPlan::default()

    1. Heartbeat 체크: elapsed >= interval -> send_heartbeat(kv_snap)
    2. try_recv drain -> directives 수집
    3. directives 비어 있으면 -> plan.throttle_delay_ms = 기존 throttle 유지, return plan
    4. 각 directive에 대해:
       a. 각 command를 순서대로 apply_command(cmd, &mut plan) -> CommandResult
       b. CommandResponse(seq_id, results) 전송 (directive 당 1회)
    5. plan.suspended == true -> plan.evict = None, plan.switch_device = None,
       plan.prepare_device = None, plan.throttle_delay_ms = 0, plan.resumed = false
    6. 내부 throttle_delay_ms 상태 갱신
    7. return plan
```

**[ENG-ST-035]** Superseding 규칙: *(MUST)*

동일 `poll()` 내 복수 Directive가 존재하면 순서대로 처리한다:

- evict: 후행이 선행을 덮어쓴다 (`plan.evict`는 마지막 값)
- switch_device: 후행이 선행을 덮어쓴다
- throttle: 후행이 선행을 덮어쓴다
- Suspend: 다른 모든 것을 무효화한다 (step 5에서 plan 초기화)
- 각 Directive에 대해 개별 CommandResponse를 전송한다

### 3.4 ExecutionPlan Lifecycle [ENG-ST-040 ~ ENG-ST-043]

**[ENG-ST-040]** 필드 정의: *(MUST)*

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| evict | Option&lt;EvictPlan&gt; | None | KV 캐시 eviction 계획 |
| switch_device | Option&lt;String&gt; | None | 디바이스 전환 대상 |
| prepare_device | Option&lt;String&gt; | None | pre-warm 디바이스 |
| throttle_delay_ms | u64 | 0 | 토큰 간 지연 (ms) |
| suspended | bool | false | 추론 중단 여부 |
| resumed | bool | false | 추론 재개 여부 |
| layer_skip | Option&lt;f32&gt; | None | Layer skip 비율 |
| kv_quant_bits | Option&lt;u8&gt; | None | KV quantization 비트 |
| restore_defaults | bool | false | 모든 액션 상태 초기화 |

**[ENG-ST-041]** EvictPlan 정의: *(MUST)*

| 필드 | 타입 | 설명 |
|------|------|------|
| target_ratio | f32 | 유지할 캐시 비율 (0.0~1.0). Streaming에서는 0.0 (사용되지 않음) |
| level | ResourceLevel | 리소스 레벨 (현재 항상 Critical) |
| method | EvictMethod | H2o, Sliding, Streaming |
| streaming_params | Option\<StreamingParams\> | Streaming 전용. sink_size + window_size. 나머지 method에서는 None |

EvictMethod 변종: `H2o`, `Sliding`, `Streaming`. Engine 내부 타입이며 shared 프로토콜에 포함되지 않는다.

StreamingParams: `struct { sink_size: usize, window_size: usize }`. Streaming eviction의 파라미터를 전달한다.

**[ENG-ST-042]** 수명 규칙: *(MUST)*

1. **생성**: `CommandExecutor.poll()`이 반환하는 시점
2. **소비**: Inference Loop가 plan의 각 필드를 읽고 실행하는 시점 (동일 토큰 내)
3. **폐기**: 다음 `poll()` 호출 전에 plan은 더 이상 참조되지 않음
4. ExecutionPlan은 1회성이다. 저장하거나 재사용하지 않는다 (INV-031)

**[ENG-ST-043]** Inference Loop의 ExecutionPlan 소비 순서: *(MUST)*

```
1. plan.evict           -> cache_manager.force_evict_by_policy() [score accumulator 활성화 포함]
2. plan.switch_device   -> backend 교체
3. plan.prepare_device  -> pre-warm
4. plan.kv_quant_bits   -> (KIVI 경로에서만 효과)
5. plan.layer_skip      -> SkipConfig 갱신
6. plan.restore_defaults -> 모든 액션 상태 초기화
7. plan.suspended       -> 대기 루프 진입
8. plan.resumed         -> 대기 루프 탈출
9. plan.throttle_delay_ms -> sleep (매 토큰)
```

### 3.5 ResilienceManager State [ENG-ST-050 ~ ENG-ST-055]

> **α-W-3 갱신 (§5.4 drift-sync)**: 이 섹션의 strategy 경로는 manager-less `LocalPolicy`(front-door ①, `ResilienceStrategy` 3종 + `resolve_conflicts`)로 재정위된다. `ResilienceAction`/`MemoryStrategy` 삭제, 출력 어휘는 `EngineCommand`(`shared/`)로 통일. manager-full Directive 경로는 종전대로 `signal_to_manager_message()` → `EngineCommand` → `CommandExecutor` 를 사용하며 본 섹션을 경유하지 않는다.

**[ENG-ST-050]** 내부 상태 필드: *(MUST)*

| 필드 | 타입 | 초기값 | 설명 |
|------|------|--------|------|
| mode | OperatingMode | Normal | 현재 운영 모드 |
| current_levels.memory | Level | Normal | 최신 memory signal level |
| current_levels.compute | Level | Normal | 최신 compute signal level |
| current_levels.thermal | Level | Normal | 최신 thermal signal level |
| current_levels.energy | Level | Normal | 최신 energy signal level |
| strategies.compute | Box&lt;dyn ResilienceStrategy&gt; | ComputeStrategy | Compute 전략 |
| strategies.thermal | Box&lt;dyn ResilienceStrategy&gt; | ThermalStrategy | Thermal 전략 |
| strategies.energy | Box&lt;dyn ResilienceStrategy&gt; | EnergyStrategy | Energy 전략 |

> **α-W-3 갱신 (§5.4 drift-sync)**: `strategies.memory` 행 제거 — `MemoryStrategy` 삭제, memory 압력은 graded `Pressure` scalar(`LocalPressureSource`) 경로로 흐른다. strategy 는 thermal/energy/compute 3개만 잔존.

**[ENG-ST-051]** `poll()` 실행 흐름: *(MUST)*

> **α-W-3 갱신 (§5.4 drift-sync)**: 반환 어휘 `Vec<ResilienceAction>` → `Vec<EngineCommand>`. `react()` 가 dead `mode` 인자 제거(2b OperatingMode 재계산은 `react()` 입력이 아닌 보고/상태용으로만 잔존).

```
function poll() -> Vec<EngineCommand>:
    all_cmds = []

    1. try_recv drain -> SystemSignal 수집
    2. 각 signal에 대해 process_signal():
       a. Level 캐시 갱신 (해당 domain)
       b. OperatingMode 재계산 (from_levels) [보고/상태용]
       c. 해당 Strategy.react(signal) -> Vec<EngineCommand>
       d. all_cmds에 추가
    3. all_cmds 비어 있으면 -> return []
    4. return resolve_conflicts(all_cmds)
```

**[ENG-ST-052]** Strategy 반응 테이블: *(MUST)*

> **α-W-3 갱신 (§5.4 drift-sync)**: 출력 어휘를 `ResilienceAction`(폐기) → `EngineCommand`(`shared/`, 18-variant, 유일 이산 어휘)로 통일. **Memory strategy 행 전면 제거** — memory 압력은 이산 strategy 가 아니라 graded `Pressure` scalar(0–100)로 흐른다 (`LocalPressureSource` 가 전 센서 magnitude 를 융합, §5.1/§5.4; `MemoryStrategy` 삭제). 따라서 strategy 는 **thermal/energy/compute 3개**만 남으며, 이들은 scalar 로 환원 불가한 *mode* 이산 명령(switch/suspend)만 낸다 (thermal/energy 의 magnitude 도 `Pressure` scalar 에 융합되지만 그건 `PressureSource` 측 관여이고, `react()` 는 mode 출력 전용). `Evict` 행은 graded 경로로 이관되어 strategy 어휘에서 소멸.

| Strategy | Level | EngineCommand |
|----------|-------|---------------|
| Thermal | Normal | RestoreDefaults |
| Thermal | Warning | SwitchHw { device: "cpu" } |
| Thermal | Critical | SwitchHw { device: "cpu" } + Throttle { delay_ms: (1-throttle_ratio)*100 } |
| Thermal | Emergency | Suspend |
| Compute | Normal | RestoreDefaults |
| Compute | Warning | (없음 -- 준비만, 전환 안 함) |
| Compute | Critical | SwitchHw { device: recommended } 또는 Throttle { delay_ms: 50 } (이미 해당 backend면) |
| Compute | Emergency | SwitchHw { device: "cpu" } + Throttle { delay_ms: 100 } |
| Energy | Normal | RestoreDefaults |
| Energy | Warning | SwitchHw { device: "cpu" } |
| Energy | Critical | SwitchHw { device: "cpu" } + Throttle { delay_ms: 30 } |
| Energy | Emergency | Suspend |

> **superseded (α-W-3, §5.4) — silent 삭제 금지로 명기**:
> - **Memory strategy 4행 전체** (`Normal→RestoreDefaults`, `Warning→Evict{0.85}`, `Critical→Evict{0.50}`, `Emergency→Evict{0.25}+RejectNew`): graded `Pressure` scalar 경로로 이관 (`LocalPressureSource` → `band()` → `EvictionStage`). `MemoryStrategy` 자체가 소멸하므로 행 제거.
> - **Thermal Critical 의 `LimitTokens { 64 }`** 및 **Energy Critical 의 `LimitTokens { 64 }`**: `EngineCommand` 에 등가 변종 부재 + production `CommandExecutor`/`ExecutionPlan` 미처리(한 번도 live 배선 안 됨) → **drop**. throttle 의 token-rate 억제 의도는 잔존 `Throttle` 이 흡수.
> - **Energy Emergency 의 `RejectNew`**: 동일 사유로 **drop**. stop-intent 는 `Suspend` 가 흡수.
> - **`SwitchBackend { Cpu }` → `SwitchHw { device: "cpu" }`** 매핑: `RecommendedBackend::Cpu`→`"cpu"`, `Gpu`→`"gpu"`, `Any`→switch 생략(구체 GPU backend(opencl/cuda) 해석은 dispatcher/`Hardware` 책임).

**[ENG-ST-053]** ResilienceStrategy trait: *(MUST)*

> **α-W-3 갱신 (§5.4 drift-sync)**: 시그니처에서 dead `mode: OperatingMode` 인자 제거(`react()` 가 `mode` 를 소비한 적 없음), 출력을 `Vec<EngineCommand>` 로 통일. trait 은 front-door 확장점 ① 로 **생존**(manager-less `LocalPolicy` 의 정책 단위, §5.4 / arch §7).

```
trait ResilienceStrategy: Send + Sync
    react(&mut self, signal: &SystemSignal) -> Vec<EngineCommand>
    name() -> &str
```

**[ENG-ST-054]** `EngineCommand` → `LoopControl` 적용: *(MUST)*

> **α-W-3 갱신 (§5.4 drift-sync)**: `react()` 출력이 `EngineCommand` 로 통일되므로 변환·적용은 `CommandDispatcher`(L4) → `LoopControl`(②control 채널) 단일 거처로 이동(구 `InferenceContext`/`execute_action` 의 ad-hoc 직접 적용 폐기). ①KV/weight·③switch 명령은 `registry.submit(OneShotStage)` 로 간다 (§5.4 A-1 3역할 분리). `LoopControl` 은 ②control 명령(throttle/target_tbt/suspend/resume/restore)만 보유한다.

`LocalPolicy.poll()` 이 반환한 `Vec<EngineCommand>` 는 `CommandDispatcher` 가 채널별로 분배한다. ②control 채널 적용:

```
LoopControl (구 InferenceContext ② 잔여):
    throttle_delay_ms: u64
    target_tbt_ms: u64
    suspended: bool

function apply(cmd: &EngineCommand, lc: &mut LoopControl):
    Throttle { delay_ms } -> lc.throttle_delay_ms = delay_ms
    SetTargetTbt { target_ms } -> lc.target_tbt_ms = target_ms
    Suspend                -> lc.suspended = true
    Resume                 -> lc.suspended = false
    RestoreDefaults        -> lc.throttle_delay_ms = 0, lc.target_tbt_ms = 0
    SwitchHw / Prepare*    -> ③switch 채널 (registry/Hardware), LoopControl 무관
    Kv* / SwapWeights / LayerSkip / SetPartitionRatio / SetPrefillPolicy -> ①KV/weight 채널 (OneShot Stage)
```

> **superseded (α-W-3, §5.4)**: 구 `InferenceContext` 의 `max_tokens: &mut usize` / `reject_new: &mut bool` 필드와 `execute_action` 의 `LimitTokens`/`RejectNew`/`Evict`/`SwitchBackend` 분기는 **폐기**. `LimitTokens`/`RejectNew` 는 `EngineCommand` 등가 부재로 어휘 자체가 소멸(ENG-ST-052 참조), `Evict` 는 graded `Pressure` 경로로, `SwitchBackend` 는 `SwitchHw` 로 이관. `&mut` borrow 기반 직접 작용 모델은 `LoopControl` 값-기반 적용으로 대체.

**[ENG-ST-055]** ResilienceManager와 CommandExecutor의 관계:

- ResilienceManager는 Strategy 경로 내부에서만 활성이다
- CommandExecutor는 Directive 경로에서 Manager 명령을 ExecutionPlan으로 변환한다
- DbusTransport 사용 시 signal이 ManagerMessage로 변환되어 CommandExecutor로 전달되므로, ResilienceManager를 경유하지 않고 CommandExecutor가 직접 처리한다

### 3.6 resolve_conflicts() Priority Rules [ENG-ST-060 ~ ENG-ST-065]

**[ENG-ST-060]** 충돌 해소 규칙 -- 4규칙 전수 열거: *(MUST)*

> **α-W-3 갱신 (§5.4 drift-sync)**: `resolve_conflicts` 입력·출력 어휘 `Vec<ResilienceAction>` → `Vec<EngineCommand>`. strategy 어휘 축소(Memory·LimitTokens·RejectNew 소멸, SwitchBackend→SwitchHw)로 규칙 수 **7→4**. 어휘에서 사라진 명령에 대응하던 R3(Evict)·R5(LimitTokens)·R7(RejectNew)는 폐기(아래 superseded). manager-less `LocalPolicy` 가 thermal/energy/compute 3 strategy 출력을 cross-domain 해소한다(§5.4).

| 규칙 | 우선순위 | 해소 방식 |
|------|---------|----------|
| R1. Suspend overrides all | 최고 | `Suspend` 존재 시 다른 모든 명령 폐기. `[Suspend]`만 반환 |
| R2. RestoreDefaults suppressed | 최저 | 다른 제약(`SwitchHw`/`Throttle`/`SetTargetTbt` 등)이 하나라도 있으면 `RestoreDefaults` 무시 |
| R3. SwitchHw: "cpu" precedence | -- | 복수 `SwitchHw` 중 `device == "cpu"` 가 하나라도 있으면 `"cpu"`. 아니면 마지막 값 |
| R4. Throttle: largest delay | -- | 복수 `Throttle` 중 `max(delay_ms)` 선택 |

> 그 외 `EngineCommand` 변종(`SetTargetTbt`/`Resume`/`PrepareComputeUnit`/`Kv*`/`SwapWeights`/`LayerSkip`/`SetPartitionRatio`/`SetPrefillPolicy`/`RequestQcf`)은 **pass-through** — strategy 가 생산하지 않거나 충돌 의미가 없어 그대로 통과한다. (strategy 출력은 R1~R4 가 다루는 `Suspend`/`SwitchHw`/`Throttle`/`RestoreDefaults` 에 한정되지만, `resolve_conflicts` 는 임의 `EngineCommand` 벡터에 대해 vacuous-agnostic 하게 정의된다 — 해당 변종이 입력에 없으면 그 규칙은 no-op.)

> **superseded (α-W-3, §5.4) — silent 삭제 금지로 명기**:
> - **구 R3 (Evict: most aggressive, `min(target_ratio)`)**: `Evict` 가 strategy 어휘에서 소멸(graded `Pressure` 경로 이관) → 폐기. graded eviction 강도 결정은 `Pressure` scalar 의 `band()` 가 담당하므로 충돌 해소 불필요.
> - **구 R5 (LimitTokens: smallest, `min(max_tokens)`)**: `LimitTokens` 어휘 소멸(EngineCommand 등가 부재) → 폐기.
> - **구 R7 (RejectNew: any)**: `RejectNew` 어휘 소멸 → 폐기.
> - **구 R4 (SwitchBackend: CPU wins)** → 현 R3 (`SwitchHw "cpu" precedence`)로 재명명·재정의(`RecommendedBackend::Cpu` enum 비교 → `device == "cpu"` 문자열 비교).
> - **구 R6 (Throttle: largest delay)** → 현 R4 로 번호만 이동(의미 불변).

**[ENG-ST-061]** 해소 알고리즘 의사코드: *(MUST)*

> **α-W-3 갱신 (§5.4 drift-sync)**: `EngineCommand` 어휘로 갱신. evict/tokens/reject 누산 제거(어휘 소멸), switch_device 는 enum 폴드 → `device == "cpu"` 문자열 폴드, pass-through 명령(R1~R4 비대상)은 원순서 보존하여 결과에 포함.

```
function resolve_conflicts(cmds: Vec<EngineCommand>) -> Vec<EngineCommand>:
    if cmds is empty: return []

    // Single-pass scan
    max_delay = 0
    target_device = None        // String, "cpu" precedence fold
    has_suspend = false
    has_restore = false
    passthrough = []            // R1~R4 비대상 명령 (원순서 보존)

    for cmd in cmds:
        match cmd:
            SwitchHw { device }  -> target_device = fold ("cpu" always wins, else last)
            Throttle { delay }   -> max_delay = max(max_delay, delay)
            Suspend              -> has_suspend = true
            RestoreDefaults      -> has_restore = true
            _                    -> passthrough.push(cmd)   // SetTargetTbt/Resume/Kv*/... pass-through

    // R1: Suspend overrides all
    if has_suspend: return [Suspend]

    // Build result from non-default values
    result = []
    if target_device is Some:   result.push(SwitchHw { device: target_device })
    if max_delay > 0:           result.push(Throttle { delay_ms: max_delay })
    result.extend(passthrough)

    // R2: RestoreDefaults only when no other constraints
    if has_restore AND result is empty:
        return [RestoreDefaults]

    return result
```

**[ENG-ST-062]** 해소 예시: *(α-W-3 갱신, EngineCommand 어휘)*

| 시나리오 | 입력 EngineCommand | 해소 결과 |
|---------|-------------------|----------|
| Thermal Warning + Energy Warning | [SwitchHw("cpu"), SwitchHw("cpu")] | [SwitchHw("cpu")] (R3) |
| Thermal Critical + Energy Critical | [SwitchHw("cpu"), Throttle(100), SwitchHw("cpu"), Throttle(30)] | [SwitchHw("cpu"), Throttle(100)] (R3 cpu, R4 max) |
| Thermal Emergency + Energy Warning | [Suspend, SwitchHw("cpu")] | [Suspend] (R1) |
| Compute Critical(gpu) + Thermal Warning | [SwitchHw("gpu"), SwitchHw("cpu")] | [SwitchHw("cpu")] (R3: cpu precedence) |
| All Normal: RestoreDefaults x3 | [Restore, Restore, Restore] | [RestoreDefaults] (R2: result empty) |
| Thermal Warning + Compute Normal | [SwitchHw("cpu"), RestoreDefaults] | [SwitchHw("cpu")] (R2: Restore 억제) |

**[ENG-ST-063]** Strategy 출력 EngineCommand 변종 (이산 mode 채널): *(MUST)*

> **α-W-3 갱신 (§5.4 drift-sync)**: 구 "ResilienceAction 7종" 표 → strategy 가 실제 생산하는 `EngineCommand` 부분집합으로 대체. `ResilienceAction` 타입 자체가 삭제되므로(§5.4) "Action → EngineCommand 매핑" 칼럼은 무의미해져 제거. thermal/energy/compute strategy 의 `react()` 는 아래 변종만 낸다 (graded magnitude 는 `Pressure` scalar 가, KV/weight 명령은 manager IPC 또는 OneShot Stage 가 담당).

| EngineCommand | 설명 | 어느 strategy 가 생산 |
|--------------|------|--------------------|
| SwitchHw { device } | 백엔드 전환 ("cpu"/"gpu") | Thermal/Energy/Compute |
| Throttle { delay_ms } | 토큰 간 지연 | Thermal/Energy/Compute |
| Suspend | 추론 중단 | Thermal(Emergency)/Energy(Emergency) |
| RestoreDefaults | 모든 제약 해제 | 모든 strategy(Normal) |

> **superseded (α-W-3, §5.4)**: 구 표의 `Evict`/`LimitTokens`/`RejectNew` 행 폐기 — `Evict` 는 graded `Pressure` 경로(strategy 비생산), `LimitTokens`/`RejectNew` 는 `EngineCommand` 등가 부재로 어휘 소멸. `ResilienceAction` 은 production 소비자 0(test-only)였으며 삭제됨; manager-full 경로(`signal_to_manager_message()`)는 이미 `SystemSignal` → `EngineCommand` 직접 변환을 사용해 왔고 본 변경의 영향을 받지 않는다.

### 3.7 KVSnapshot Reporting [ENG-ST-070]

**[ENG-ST-070]** KVSnapshot 필드와 출처: *(MUST)*

| 필드 | 타입 | 출처 |
|------|------|------|
| total_bytes | u64 | 전 layer KV buffer 크기 합산 |
| total_tokens | usize | `kv_caches[0].current_pos` |
| capacity | usize | `kv_caches[0].capacity()` |
| protected_prefix | usize | CLI `--protected-prefix` 또는 정책 기본값 (H2O: 4, Sliding: prompt_len) |
| kv_dtype | String | "f16", "q4", "q2" -- CLI `--kv-type` / KIVI 모드 |
| eviction_policy | String | CLI `--eviction-policy` |
| skip_ratio | f32 | 현재 layer skip 비율 |

KVSnapshot은 Heartbeat의 EngineStatus 필드를 채우는 데 사용된다 (`send_heartbeat`).

## 4. Alternative Behavior

- **Strategy 경로 비활성 시**: ResilienceManager와 OperatingMode FSM은 인스턴스화되지 않는다. CommandExecutor만 활성이며 EngineState FSM만 동작한다.

- **CommandExecutor 비활성 시** (`command_executor = None`): 모든 상태 머신이 비활성이다. Inference Loop는 resilience checkpoint를 건너뛰고 순수 추론만 수행한다.

- **Experiment 모드**: 외부 Transport 대신 내부 mpsc 채널로 CommandExecutor를 구동한다. 상태 전이 규칙은 동일하다.

## 5. Constraints

**Invariants:**

| ID | 불변식 |
|----|--------|
| INV-070 | `OperatingMode.from_levels()`는 순수 함수이다. 입력 4종 Level만으로 결과가 결정되며 이전 상태에 의존하지 않는다 |
| INV-071 | EngineState 전이는 CommandExecutor 내부에서만 발생한다. 외부에서 직접 변경하지 않는다 |
| INV-072 | `resolve_conflicts()`에서 Suspend 존재 시 반환 리스트는 정확히 `[Suspend]`이다 |
| INV-073 | `resolve_conflicts()`에서 RestoreDefaults는 다른 제약이 하나도 없을 때만 반환된다 |
| INV-074 | `CommandExecutor.poll()`의 `plan.suspended == true`일 때 `plan.evict`, `plan.switch_device`, `plan.prepare_device`는 모두 None이다 |
| INV-075 | Resume 명령은 compute_level, memory_level을 Normal로, throttle_delay_ms를 0으로 초기화한다 |
| INV-076 | RestoreDefaults 명령은 active_actions를 비우고, throttle_delay_ms를 0으로, compute/memory_level을 Normal로 초기화한다 |

## 6. Examples

### 6.1 Directive 경로: Suspend-Resume 사이클

```
Token 100: poll() -> ManagerMessage::Directive(seq_id=5, [Suspend])
  apply_command(Suspend):
    engine_state = Suspended
    plan.suspended = true
  poll() step 5: plan.evict = None, plan.switch_device = None, plan.throttle_delay_ms = 0
  CommandResponse(5, [Ok]) 전송
  Inference Loop: 대기 루프 진입

Token 100 (대기 중): poll() -> ManagerMessage::Directive(seq_id=6, [Resume])
  apply_command(Resume):
    engine_state = Running
    compute_level = Normal, memory_level = Normal, throttle_delay_ms = 0
    plan.resumed = true
  CommandResponse(6, [Ok]) 전송
  Inference Loop: 대기 루프 탈출, 추론 재개
```

### 6.2 Superseding: 동일 poll() 내 복수 Directive

```
Token 50: poll() try_recv -> 2 directives 수집

Directive 1 (seq_id=10): [KvEvictH2o { keep_ratio: 0.7 }]
  plan.evict = Some(EvictPlan { H2o, 0.7, Critical })
  CommandResponse(10, [Ok])

Directive 2 (seq_id=11): [KvEvictSliding { keep_ratio: 0.5 }, Throttle { delay_ms: 50 }]
  plan.evict = Some(EvictPlan { Sliding, 0.5, Critical })  -- 선행 덮어씀
  plan.throttle_delay_ms = 50
  CommandResponse(11, [Ok, Ok])

최종 plan: evict = Sliding(0.5), throttle = 50ms
```

### 6.3 Strategy 경로: Memory Critical + Thermal Warning

> **α-W-3 갱신 (§5.4 drift-sync)**: Memory 압력은 strategy 가 아닌 graded `Pressure` scalar 경로로 분리됐다. 따라서 이 시나리오는 두 경로의 **병렬 흐름**으로 재기술된다 — memory magnitude 는 `LocalPressureSource` 가 `Pressure(0–100)` 으로 융합해 `band()`→`EvictionStage`(graded eviction)를 구동하고, thermal 의 *mode* 출력만 `LocalPolicy`(strategy) → `EngineCommand` 로 흐른다. 두 경로는 같은 thermal 센서를 두 목적(입력 scalar ⊕ 출력 이산명령)으로 읽는다.

```
[연속 채널] LocalPressureSource.poll():
  memory magnitude (Critical급) ⊕ thermal magnitude (Warning급) -> Pressure ≈ <fused>
    -> StepInfo -> EvictionStage.band() -> graded eviction (강도 = Pressure 함수)
  (이산 EngineCommand 미생산)

[이산 채널] LocalPolicy.poll():
  signal: ThermalAlert { level: Warning }
    current_levels.thermal = Warning
    ThermalStrategy.react(signal) -> [SwitchHw { device: "cpu" }]
  (MemoryPressure 는 strategy 비대상 — graded 경로가 흡수)

  all_cmds = [SwitchHw("cpu")]
  resolve_conflicts() -> [SwitchHw("cpu")]   -- CommandDispatcher -> ③switch 채널
```

### 6.4 D-Bus Emergency 자율 전이

```
D-Bus signal: MemoryPressure { level: Emergency }
  DbusTransport.signal_to_manager_message():
    Emergency -> EngineCommand::Suspend
    ManagerMessage::Directive(seq_id=auto, [Suspend])

CommandExecutor.poll():
  apply_command(Suspend):
    engine_state = Suspended
    plan.suspended = true
  step 5: plan 초기화

Inference Loop: 대기 루프 진입 (Manager 명시 명령 없이 자율 진입)
```

## 7. Rationale (non-normative)

### 왜 OperatingMode가 순수 함수인가

이전 상태와 무관하게 현재 Level만으로 모드를 결정하여 상태 관리를 단순화한다. 히스테리시스가 필요한 경우 Manager 측 Supervisory Layer(`21-manager-state.md` MGR-050 ~ MGR-059)에서 처리하므로 Engine은 최신 상태만 반영하면 된다.

### 왜 EngineState와 OperatingMode가 분리되어 있는가

EngineState는 프로토콜 수준(Manager-Engine 공유)의 3-state로 단순하다. OperatingMode는 Engine 내부 Strategy 경로의 4-state로 D-Bus 레거시 환경에서 독립 판단에 필요하다. Directive 경로에서는 Manager가 모드를 고려한 명령을 이미 전송하므로 Engine이 별도 모드를 계산할 필요가 없다.

### 왜 Suspend가 plan을 초기화하는가

Emergency 상황에서 eviction이나 device switch가 실행되면 오히려 위험할 수 있다. Suspend는 추론을 즉시 멈추는 것이 목적이므로 다른 모든 적응 액션을 무효화하여 안전한 중단을 보장한다 (SYS-055 참조).

### 왜 resolve_conflicts()에서 CPU가 항상 우선인가

Safety-first 원칙이다. 복수 도메인(thermal + compute)이 동시에 Backend 전환을 요구할 때, CPU는 항상 사용 가능하고 전력/발열 면에서 안전한 폴백이다. GPU 전환은 단일 도메인 판단일 때만 유효하다.

### 왜 ExecutionPlan이 1회성인가

plan을 저장하면 다음 poll()에서 중복 실행될 위험이 있다. 1회 생성-소비-폐기 패턴으로 멱등성 문제를 원천 차단한다.
