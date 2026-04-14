# Protocol Messages

> **TL;DR**: Manager ↔ Engine IPC의 모든 메시지 타입을 필드 수준으로 정의한다. Manager→Engine: ManagerMessage(1종) 내 EngineDirective(seq_id + EngineCommand 13종 배치). Engine→Manager: EngineMessage(4종) — Capability(5필드), Heartbeat/EngineStatus(16필드), Response/CommandResponse(seq_id + CommandResult 3종), QcfEstimate(per-action QCF 비용). D-Bus 전송용: SystemSignal(4종, externally tagged). 이 문서의 테이블이 프로토콜 구현의 canonical reference이다.

## 1. Purpose and Scope

이 문서는 Manager ↔ Engine 간 모든 IPC 메시지 타입의 필드명, 타입, 범위, serde 태그값, JSON 예시를 완전 구체적으로 정의한다.

**이 파일이 명세하는 것:**

- Envelope 타입 (ManagerMessage, EngineMessage)
- Directive 구조 (EngineDirective, EngineCommand 13종)
- Engine 보고 구조 (EngineCapability, EngineStatus 16필드, CommandResponse, CommandResult 3종)
- 지원 열거형 타입 (ResourceLevel, EngineState, Level, RecommendedBackend, ComputeReason, EnergyReason)
- D-Bus 전송용 SystemSignal 4종

**이 파일이 명세하지 않는 것:**

- 와이어 포맷, 전송 계층 → `10-protocol.md`
- 메시지 시퀀스 → `12-protocol-sequences.md`
- 메시지의 의미론적 처리 로직 → `22-manager-algorithms.md`, `32-engine-algorithms.md`

## 2. Definitions

| 용어 | 정의 |
|------|------|
| **Tag Value** | serde internally tagged enum의 `"type"` 또는 `"status"` 필드 값. |
| **Default** | `#[serde(default)]` 어노테이션. JSON에서 해당 필드 생략 시 적용되는 값. |
| **Wire Name** | JSON 직렬화 시 사용되는 키 이름. 특별한 rename이 없으면 snake_case 필드명과 동일. |

## 3. Specification

### 3.1 Envelope Types [MSG-010 ~ MSG-014]

**[MSG-010]** ManagerMessage — Manager → Engine 최상위 envelope. `tag = "type"`, `rename_all = "snake_case"`. *(MUST)*

| Tag Value | Variant | Payload | 설명 |
|-----------|---------|---------|------|
| `"directive"` | Directive | EngineDirective | 명령 배치 |

JSON 예시:
```json
{
  "type": "directive",
  "seq_id": 1,
  "commands": [{"type": "throttle", "delay_ms": 50}]
}
```

> **참고 (non-normative)**: Internally tagged enum이므로 `EngineDirective`의 필드(`seq_id`, `commands`)가 동일 JSON 객체에 flat merge된다.

**[MSG-011]** EngineMessage — Engine → Manager 최상위 envelope. `tag = "type"`, `rename_all = "snake_case"`. *(MUST)*

| Tag Value | Variant | Payload | 설명 |
|-----------|---------|---------|------|
| `"capability"` | Capability | EngineCapability | 능력 보고 (세션당 1회) |
| `"heartbeat"` | Heartbeat | EngineStatus | 주기적 상태 보고 |
| `"response"` | Response | CommandResponse | Directive 실행 응답 |
| `"qcf_estimate"` | QcfEstimate | QcfEstimate | RequestQcf에 대한 QCF 비용 응답 |

JSON 예시 (Capability):
```json
{
  "type": "capability",
  "available_devices": ["cpu", "opencl"],
  "active_device": "cpu",
  "max_kv_tokens": 2048,
  "bytes_per_kv_token": 256,
  "num_layers": 16
}
```

JSON 예시 (Heartbeat, 간략):
```json
{
  "type": "heartbeat",
  "active_device": "cpu",
  "compute_level": "normal",
  "actual_throughput": 15.0,
  "memory_level": "normal",
  "kv_cache_bytes": 1048576,
  "kv_cache_tokens": 512,
  "kv_cache_utilization": 0.25,
  "memory_lossless_min": 1.0,
  "memory_lossy_min": 0.01,
  "state": "running",
  "tokens_generated": 100
}
```

JSON 예시 (Response):
```json
{
  "type": "response",
  "seq_id": 1,
  "results": [{"status": "ok"}]
}
```

**[MSG-012]** Envelope의 태그 필드 이름은 `"type"` (ManagerMessage, EngineMessage 모두)으로 고정된다. Variant별 payload 필드는 동일 JSON 객체에 flat merge된다. *(MUST)*

**[MSG-013]** 알 수 없는 tag value 수신 시 serde 역직렬화가 실패한다. 이는 `10-protocol.md` PROTO-061에 따라 ParseError로 처리된다. *(MUST)*

**[MSG-014]** QCF 프로토콜 확장이 정의되어 있다. EngineCommand에 `RequestQcf` (MSG-036b), EngineMessage에 `QcfEstimate` (MSG-085~087)가 포함된다. 이 확장은 논문의 C4 기여(QCF 기반 품질 저하 정량화)를 프로토콜 수준에서 지원한다. *(non-normative)*

### 3.2 EngineDirective [MSG-020 ~ MSG-022]

**[MSG-020]** EngineDirective — Manager → Engine 명령 배치. *(MUST)*

| 필드 | 타입 | 범위 | Default | 설명 |
|------|------|------|---------|------|
| seq_id | u64 | ≥ 1, 단조 증가 | (필수) | 디렉티브 식별자 |
| commands | Vec\<EngineCommand\> | 1개 이상 | (필수) | 명령 목록 (순서대로 처리) |

JSON 예시:
```json
{
  "seq_id": 42,
  "commands": [
    {"type": "kv_evict_h2o", "keep_ratio": 0.48},
    {"type": "throttle", "delay_ms": 30}
  ]
}
```

> **참고 (non-normative)**: ManagerMessage envelope에 감싸여 전송되므로 실제 와이어에서는 `"type": "directive"` 태그가 추가된다.

**[MSG-021]** seq_id 불변식 — 세션 내 단조 증가한다 (INV-020 참조). 동일 seq_id를 재사용해서는 안 된다 (INV-021 참조). *(MUST)*

**[MSG-022]** commands 배열 — 순서대로 처리된다. 후속 명령이 선행 명령의 효과를 덮어쓸 수 있다 (예: 동일 배치에 2개 evict 명령 시 마지막이 승리). Suspend 명령이 포함되면 다른 모든 명령의 ExecutionPlan 효과를 초기화한다. *(MUST)*

### 3.3 EngineCommand [MSG-030 ~ MSG-041, MSG-031b]

**[MSG-030]** EngineCommand — Manager → Engine 개별 명령. `tag = "type"`, `rename_all = "snake_case"`. **14종 변형.** *(MUST)*

| Tag Value | Variant | 도메인 | 필드 | 필드 타입 | 범위 | 설명 |
|-----------|---------|--------|------|----------|------|------|
| `"throttle"` | Throttle | Compute | delay_ms | u64 | 0–100 | 토큰 간 삽입 딜레이(ms). 0=해제. |
| `"set_target_tbt"` | SetTargetTbt | Compute | target_ms | u64 | ≥ 0 | 목표 TBT(ms). 동적 pacing. 0=해제. |
| `"layer_skip"` | LayerSkip | Compute | skip_ratio | f32 | [0.0, 1.0] | 건너뛸 레이어 비율. |
| `"kv_evict_h2o"` | KvEvictH2o | Memory | keep_ratio | f32 | (0.0, 1.0] | H2O 정책으로 유지할 KV 비율. |
| `"kv_evict_sliding"` | KvEvictSliding | Memory | keep_ratio | f32 | (0.0, 1.0] | 슬라이딩 윈도우로 유지할 KV 비율. |
| `"kv_merge_d2o"` | KvMergeD2o | Memory | keep_ratio | f32 | (0.0, 1.0] | D2O 정책(eviction + compensation merging)으로 유지할 KV 비율. |
| `"kv_streaming"` | KvStreaming | Memory | sink_size | usize | ≥ 1 | Attention sink 토큰 수. |
| | | | window_size | usize | ≥ 1 | 슬라이딩 윈도우 크기. |
| `"kv_quant_dynamic"` | KvQuantDynamic | Memory | target_bits | u8 | {2, 4, 8} | KV 캐시 양자화 비트 수. |
| `"request_qcf"` | RequestQcf | Query | (없음) | — | — | 각 lossy 액션의 예상 QCF 비용 요청. |
| `"restore_defaults"` | RestoreDefaults | Lifecycle | (없음) | — | — | 모든 액션 기본값 복원. |
| `"switch_hw"` | SwitchHw | Lifecycle | device | String | 디바이스 식별자 | 컴퓨트 유닛 전환. |
| `"prepare_compute_unit"` | PrepareComputeUnit | Lifecycle | device | String | 디바이스 식별자 | 전환 사전 워밍업. |
| `"suspend"` | Suspend | Lifecycle | (없음) | — | — | 추론 즉시 중지. |
| `"resume"` | Resume | Lifecycle | (없음) | — | — | 추론 재개. |

> **참고 (non-normative)**: 위 '도메인' 칼럼은 각 액션의 **주 대상(primary target) 도메인** 분류이다. 실제 cross-domain relief effect(하나의 액션이 여러 도메인에 동시 영향을 미침)는 Action Pool(`01-architecture.md` SYS-095)과 `22-manager-algorithms.md`에서 모델링된다.

#### MSG-031: Throttle

토큰 생성 간 지정된 밀리초만큼 딜레이를 삽입한다. `delay_ms = 0`이면 스로틀을 해제한다.

```json
{"type": "throttle", "delay_ms": 50}
```

#### MSG-031b: SetTargetTbt

목표 TBT(Time Between Tokens)를 설정한다. Engine은 매 토큰 생성 후 `sleep(max(0, target_ms - actual_tbt))`로 동적 pacing을 수행하여 일정한 TBT를 유지한다. `target_ms = 0`이면 pacing을 해제한다. `RestoreDefaults`로도 해제된다.

Throttle과의 차이: Throttle은 고정 delay를 삽입하므로 실제 TBT가 `forward + delay`로 가변적이다. SetTargetTbt는 실제 forward 시간에 따라 sleep을 조절하여 `max(forward, target)`으로 일정한 TBT를 유지한다. 동일 QoS에서 리소스 사용량을 공정 비교할 때 사용한다.

```json
{"type": "set_target_tbt", "target_ms": 150}
```

#### MSG-032: LayerSkip

Transformer 레이어의 지정된 비율을 건너뛴다. `skip_ratio = 0.0`이면 스킵 없음, `1.0`이면 전체 스킵.

```json
{"type": "layer_skip", "skip_ratio": 0.25}
```

#### MSG-033: KvEvictH2o

Heavy-Hitter Oracle(H2O) 정책으로 KV 캐시를 eviction한다. `keep_ratio`만큼의 토큰을 유지한다.

```json
{"type": "kv_evict_h2o", "keep_ratio": 0.48}
```

#### MSG-034: KvEvictSliding

슬라이딩 윈도우 정책으로 KV 캐시를 eviction한다. 가장 최근 토큰 `keep_ratio`만큼 유지한다.

```json
{"type": "kv_evict_sliding", "keep_ratio": 0.6}
```

#### MSG-034b: KvMergeD2o

D2O (Dynamic Discriminative Operations) 정책으로 KV 캐시를 eviction한다. H2O 스타일 3-partition eviction에 compensation merging을 결합한다. eviction 대상 토큰을 가장 유사한 retained 토큰에 scatter-reduce로 병합하여 정보 손실을 줄인다.

```json
{"type": "kv_merge_d2o", "keep_ratio": 0.5}
```

#### MSG-035: KvStreaming

StreamingLLM (attention sink + sliding window) 정책. Engine은 `Ok`를 반환하고, 다음 토큰 생성 전에 sink 영역과 recent window를 유지하며 중간 토큰을 제거한다. executor.rs에서 `EvictPlan { method: Streaming, streaming_params: Some(StreamingParams { sink_size, window_size }) }`를 생성하고, generate.rs에서 `StreamingLLMPolicy`를 즉석 호출하여 실행한다.

```json
{"type": "kv_streaming", "sink_size": 4, "window_size": 256}
```

#### MSG-036: KvQuantDynamic

KV 캐시의 양자화 비트 수를 동적으로 전환한다. KIVI 캐시(`kv_dtype`이 "q2"/"q4"/"q8")에서만 유효하다.

```json
{"type": "kv_quant_dynamic", "target_bits": 4}
```

#### MSG-036b: RequestQcf

Manager가 Critical 전환 시 Engine에 각 lossy 액션의 예상 QCF 비용을 1회 요청한다. Engine은 현재 KV 캐시/모델 상태를 기반으로 읽기 전용 스캔을 수행하여 `QcfEstimate` (EngineMessage)로 응답한다. KV 캐시를 변경하지 않는다.

```json
{"type": "request_qcf"}
```

- RequestQcf는 단독 Directive로 전송하는 것을 권장한다 (SHOULD). 다른 EngineCommand와 같은 Directive에 포함할 수 있으나 (MAY), QCF 계산 결과를 받은 뒤 액션을 선택하는 흐름상 별도 Directive가 자연스럽다.
- Engine은 RequestQcf를 포함한 Directive에 대해 먼저 CommandResponse를 전송하고 (RequestQcf에 대해 `Ok`), 그 다음 별도 EngineMessage로 QcfEstimate를 전송한다 (MUST). 순서: Response → QcfEstimate.
- Engine이 QCF 계산을 지원하지 않는 상태(예: 캐시 비어있음)이면 빈 estimates로 응답한다 (MUST).

#### MSG-037: RestoreDefaults

모든 액션 유도 상태를 기본값으로 복원한다. `active_actions`를 전체 초기화하고, throttle을 0, target_tbt를 0으로 리셋한다.

```json
{"type": "restore_defaults"}
```

#### MSG-038: SwitchHw

활성 컴퓨트 유닛을 전환한다. 동일 배치 내 여러 SwitchHw 명령이 있으면 마지막이 승리한다.

```json
{"type": "switch_hw", "device": "opencl"}
```

#### MSG-039: PrepareComputeUnit

지정된 컴퓨트 유닛을 사전 워밍업한다. 실제 전환은 SwitchHw로 수행한다.

```json
{"type": "prepare_compute_unit", "device": "opencl"}
```

#### MSG-040: Suspend

추론을 즉시 중지한다. 동일 배치 내 다른 명령의 ExecutionPlan 효과를 override한다 (evict, switch_device, throttle 모두 초기화). Engine 상태가 Suspended로 전이한다.

```json
{"type": "suspend"}
```

#### MSG-041: Resume

Suspended 상태에서 추론을 재개한다. compute_level, memory_level을 Normal로, throttle을 0으로 리셋한다.

```json
{"type": "resume"}
```

### 3.4 EngineCapability [MSG-050 ~ MSG-052]

**[MSG-050]** EngineCapability — Engine → Manager 능력 보고. 세션당 1회, 연결 직후 전송. *(MUST)*

| 필드 | 타입 | 범위 | Default | 설명 |
|------|------|------|---------|------|
| available_devices | Vec\<String\> | 1개 이상 | (필수) | 사용 가능한 디바이스 목록 |
| active_device | String | available_devices 중 하나 | (필수) | 현재 활성 디바이스 |
| max_kv_tokens | usize | > 0 | (필수) | KV 캐시 최대 토큰 수 |
| bytes_per_kv_token | usize | > 0 | (필수) | KV 토큰당 바이트 수 |
| num_layers | usize | > 0 | (필수) | 모델 레이어 수 |

**[MSG-051]** device 문자열은 소문자이다. 알려진 값: `"cpu"`, `"opencl"`. 향후 확장 가능하다. *(SHOULD)*

**[MSG-052]** `max_kv_tokens × bytes_per_kv_token` = 총 KV 캐시 바이트 예산이다. *(non-normative)*

JSON 예시:
```json
{
  "type": "capability",
  "available_devices": ["cpu", "opencl"],
  "active_device": "cpu",
  "max_kv_tokens": 2048,
  "bytes_per_kv_token": 256,
  "num_layers": 16
}
```

### 3.5 EngineStatus (Heartbeat) [MSG-060 ~ MSG-066]

**[MSG-060]** EngineStatus — Engine → Manager 주기적 상태 보고. **16 필드.** *(MUST)*

| # | 필드 | 타입 | 범위 | Default | 설명 |
|---|------|------|------|---------|------|
| 1 | active_device | String | — | (필수) | 현재 활성 디바이스 |
| 2 | compute_level | ResourceLevel | 3종 | (필수) | 컴퓨트 리소스 수준 |
| 3 | actual_throughput | f32 | ≥ 0.0 | (필수) | 실측 TPS (EMA) |
| 4 | memory_level | ResourceLevel | 3종 | (필수) | 메모리 리소스 수준 |
| 5 | kv_cache_bytes | u64 | ≥ 0 | (필수) | KV 캐시 사용 바이트 |
| 6 | kv_cache_tokens | usize | ≥ 0 | (필수) | KV 캐시 저장 토큰 수 |
| 7 | kv_cache_utilization | f32 | [0.0, 1.0] | (필수) | kv_cache_tokens / max_kv_tokens |
| 8 | memory_lossless_min | f32 | [0.0, 1.0] | (필수) | 무손실 확보 가능 최소 비율 |
| 9 | memory_lossy_min | f32 | [0.0, 1.0] | (필수) | 유손실 확보 가능 최소 비율 |
| 10 | state | EngineState | 3종 | (필수) | 엔진 상태 |
| 11 | tokens_generated | usize | ≥ 0 | (필수) | 세션 누적 토큰 수 |
| 12 | available_actions | Vec\<String\> | — | `[]` | 실행 가능 액션 목록 |
| 13 | active_actions | Vec\<String\> | — | `[]` | 현재 활성 액션 목록 |
| 14 | eviction_policy | String | — | `""` | 현재 eviction 정책명 |
| 15 | kv_dtype | String | — | `""` | 현재 KV dtype |
| 16 | skip_ratio | f32 | [0.0, 1.0] | `0.0` | 현재 레이어 스킵 비율 |

**[MSG-061]** 하위 호환성 — 필드 12~16 (available_actions, active_actions, eviction_policy, kv_dtype, skip_ratio)에 `#[serde(default)]`가 적용되어 있다. 구 버전 Engine의 JSON에서 이 필드가 생략되어도 역직렬화가 성공한다. *(MUST)*

**[MSG-062]** available_actions — Engine이 현재 상태에서 실행 가능한 액션 목록이다. 값은 EngineCommand tag value와 동일한 문자열이다. 결정 로직은 Engine 내부이며 32번 스펙에서 상세화한다. 여기서는 값 도메인만 정의한다: *(non-normative)*

- 항상 포함: `"throttle"`, `"switch_hw"`, `"layer_skip"`
- eviction 정책이 `"none"`이 아닐 때 추가: `"kv_evict_h2o"`, `"kv_evict_sliding"`
- kv_dtype이 `"q"` 접두사일 때 추가: `"kv_quant_dynamic"`

**[MSG-063]** active_actions — 현재 활성 상태인 액션 문자열 목록이다. 값은 EngineCommand tag value와 동일하다. 예: `["throttle", "kv_evict_h2o"]`. *(MUST)*

**[MSG-064]** eviction_policy 값 — `"none"`, `"h2o"`, `"sliding"`, `"streaming"` 등. Engine 내부 정책명. 비어 있으면(`""`) 구 버전 호환 기본값. *(SHOULD)*

**[MSG-065]** kv_dtype 값 — `"f16"`, `"q8"`, `"q4"`, `"q2"`. Engine의 현재 KV 캐시 데이터 타입. 비어 있으면(`""`) 구 버전 호환 기본값. *(SHOULD)*

**[MSG-066]** actual_throughput — EMA (α=0.1)로 계산된다. 토큰 미생성 시 0.0. *(non-normative)*

**[MSG-067]** self_cpu_pct 계산 — Engine은 `/proc/self/stat`의 `(utime + stime)` 틱 증가량을 직전 Heartbeat 송출 시각과의 wall-clock 경과 및 `sysconf(_SC_CLK_TCK)`, 코어 수(`num_cpus`)로 정규화하여 [0.0, 1.0] 범위로 산출한다. 범위 밖 값은 clamp한다 (INV-091). 측정 실패 시 0.0 fallback이며, Heartbeat 송출은 차단하지 않는다 (INV-092). *(MUST)*

**[MSG-068]** self_gpu_pct 계산 — Engine은 OpenCL queue profiling
(`CL_QUEUE_PROFILING_ENABLE`)로 캡처한 커널 start/end 이벤트에서
`(end - start)` nanoseconds를 누적하고, 직전 Heartbeat 송출 시각과의
wall-clock 경과로 정규화하여 [0.0, 1.0] 범위로 산출한다. 범위 밖 값은 clamp한다
(INV-091). **Opt-in only**: profiling 활성화 시 Adreno 기준 ~54 ms/token
오버헤드가 있으므로 기본은 비활성화(meter 미주입)이며 이 경우 0.0을 그대로
송출한다 (INV-092 fallback, 하위호환). 측정 실패/meter 미주입 시 0.0이며
Heartbeat 송출은 차단하지 않는다 (INV-092). 첫 샘플은 기준점 부재로 0.0
반환(warm-up). CUDA 등 다른 백엔드는 동등한 device-side event timing을
사용할 수 있으며(예: `cudaEventElapsedTime`), 의미는 동일하게 정의된다.
*(MUST when meter is injected; MAY skip otherwise)*

**[MSG-069]** Manager 연결 — `ctx.engine.cpu_pct`, `ctx.engine.gpu_pct`는 LuaPolicy 평가 컨텍스트에 노출된다 (MGR-DAT-075, MGR-DAT-076). 시스템 전체 CPU 사용률(`ComputeGuidance.cpu_pct` 기반 `ctx.signal.compute.cpu_pct`)과 별도 값으로 유지되며, 두 값의 산출/비교(예: 외부 경합량)는 Lua 스크립트 책임이다. Pressure6D 계산식은 변경되지 않는다. *(MUST)*

JSON 예시 (전체 18필드):
```json
{
  "type": "heartbeat",
  "active_device": "cpu",
  "compute_level": "normal",
  "actual_throughput": 15.0,
  "memory_level": "normal",
  "kv_cache_bytes": 1048576,
  "kv_cache_tokens": 512,
  "kv_cache_utilization": 0.25,
  "memory_lossless_min": 1.0,
  "memory_lossy_min": 0.01,
  "state": "running",
  "tokens_generated": 100,
  "available_actions": ["throttle", "switch_hw", "layer_skip", "kv_evict_h2o", "kv_evict_sliding"],
  "active_actions": ["kv_evict_h2o"],
  "eviction_policy": "h2o",
  "kv_dtype": "f16",
  "skip_ratio": 0.0,
  "self_cpu_pct": 0.42,
  "self_gpu_pct": 0.0
}
```

JSON 예시 (7필드 생략 — 구 버전 호환):
```json
{
  "type": "heartbeat",
  "active_device": "cpu",
  "compute_level": "normal",
  "actual_throughput": 10.0,
  "memory_level": "normal",
  "kv_cache_bytes": 0,
  "kv_cache_tokens": 0,
  "kv_cache_utilization": 0.0,
  "memory_lossless_min": 1.0,
  "memory_lossy_min": 0.01,
  "state": "running",
  "tokens_generated": 0
}
```

### 3.6 CommandResponse [MSG-070 ~ MSG-073]

**[MSG-070]** CommandResponse — Engine → Manager Directive 응답. *(MUST)*

| 필드 | 타입 | 범위 | 설명 |
|------|------|------|------|
| seq_id | u64 | 대응 EngineDirective.seq_id | 매칭 식별자 |
| results | Vec\<CommandResult\> | 대응 commands와 1:1 | 명령별 결과 |

**[MSG-071]** results 배열 길이는 대응 `EngineDirective.commands` 길이와 동일해야 한다 (INV-024 참조). *(MUST)*

#### 불변식

- **[INV-025]** `len(CommandResponse.results) == len(EngineDirective.commands)`. *(MUST)*

**[MSG-072]** `results[i]`는 `commands[i]`의 실행 결과이다 (순서 보존). *(MUST)*

**[MSG-073]** seq_id 매칭 불변식 — 수신한 적 없는 seq_id로 Response를 전송해서는 안 된다 (INV-023 참조). *(MUST NOT)*

#### 불변식

- **[INV-026]** Engine은 수신한 EngineDirective.seq_id에 대해서만 CommandResponse를 전송한다. *(MUST)*

JSON 예시:
```json
{
  "type": "response",
  "seq_id": 42,
  "results": [
    {"status": "ok"},
    {"status": "partial", "achieved": 0.7, "reason": "insufficient cache tokens"},
    {"status": "rejected", "reason": "single backend"}
  ]
}
```

### 3.7 CommandResult [MSG-080 ~ MSG-083]

**[MSG-080]** CommandResult — 개별 명령 실행 결과. `tag = "status"`, `rename_all = "snake_case"`. *(MUST)*

| Tag Value | Variant | 추가 필드 | 설명 |
|-----------|---------|----------|------|
| `"ok"` | Ok | (없음) | 정상 실행 |
| `"partial"` | Partial | achieved: f32, reason: String | 부분 실행 |
| `"rejected"` | Rejected | reason: String | 실행 불가 |

**[MSG-081]** Partial — `achieved`는 달성된 비율 [0.0, 1.0]. `reason`은 부분 실행 사유. *(MUST)*

```json
{"status": "partial", "achieved": 0.7, "reason": "insufficient cache tokens"}
```

**[MSG-082]** Rejected — `reason`은 거부 사유. *(MUST)*

```json
{"status": "rejected", "reason": "single backend"}
```

```json
{"status": "rejected", "reason": "D2O handler not configured (requires --eviction-policy d2o)"}
```

**[MSG-083]** Ok — 추가 필드 없음. *(MUST)*

```json
{"status": "ok"}
```

### 3.8 QcfEstimate [MSG-085 ~ MSG-087]

**[MSG-085]** QcfEstimate — Engine → Manager QCF 비용 응답. RequestQcf 명령에 대한 응답으로 전송된다. *(MUST)*

| 필드 | 타입 | 설명 |
|------|------|------|
| estimates | Map\<String, f32\> | 각 lossy 액션의 예상 QCF 비용. 키는 EngineCommand tag value. |

**[MSG-086]** estimates의 키는 Engine이 현재 계산 가능한 lossy 액션에 한정된다. 계산 불가능한 액션(예: KV 캐시 비어있어 eviction QCF 산출 불가)은 키에 포함하지 않는다. *(MUST)*

**[MSG-087]** QCF 값은 0.0 이상이다. 값이 클수록 품질 저하가 크다. 0.0은 저하 없음을 의미한다. *(MUST)*

JSON 예시:
```json
{
  "type": "qcf_estimate",
  "estimates": {
    "kv_evict_h2o": 0.12,
    "kv_evict_sliding": 0.18,
    "kv_merge_d2o": 0.08,
    "kv_quant_dynamic": 0.25,
    "layer_skip": 0.35
  }
}
```

> **참고 (non-normative)**: Manager의 ActionSelector는 이 값을 lossy 액션의 비용으로 사용한다. Lossless 액션의 비용은 0이다. QcfEstimate가 없으면(Engine 미연결 등) ActionRegistry의 default_cost를 fallback으로 사용한다.

### 3.9 Supporting Enums [MSG-090 ~ MSG-095]

**[MSG-090]** ResourceLevel — 프로토콜 수준 3단계 리소스 심각도. `rename_all = "snake_case"`. EngineStatus에서 사용. *(MUST)*

| Wire Value | Variant | 순서 |
|-----------|---------|------|
| `"normal"` | Normal | 0 (최저) |
| `"warning"` | Warning | 1 |
| `"critical"` | Critical | 2 (최고) |

> **참고 (non-normative)**: `PartialOrd`/`Ord` derive로 순서 비교가 가능하다 (Normal < Warning < Critical).

**[MSG-091]** EngineState — Engine 운영 상태. `rename_all = "snake_case"`. EngineStatus에서 사용. *(MUST)*

| Wire Value | Variant | 설명 |
|-----------|---------|------|
| `"idle"` | Idle | 추론 시작 전 |
| `"running"` | Running | 추론 실행 중 |
| `"suspended"` | Suspended | Suspend 명령으로 중지됨 |

**[MSG-092]** Level — SystemSignal(D-Bus)용 4단계 심각도. `rename_all = "snake_case"`. *(MUST)*

| Wire Value | Variant | 순서 |
|-----------|---------|------|
| `"normal"` | Normal | 0 |
| `"warning"` | Warning | 1 |
| `"critical"` | Critical | 2 |
| `"emergency"` | Emergency | 3 |

> **참고 (non-normative)**: ResourceLevel(3단계)과 Level(4단계)의 차이 — Emergency는 프로토콜 수준에서 Suspend 명령으로 대체된다. Level은 D-Bus 전송 경로에서 사용한다.

**[MSG-093]** RecommendedBackend — SystemSignal.ComputeGuidance에서 사용. `rename_all = "snake_case"`. *(MUST)*

| Wire Value | Variant |
|-----------|---------|
| `"cpu"` | Cpu |
| `"gpu"` | Gpu |
| `"any"` | Any |

**[MSG-094]** ComputeReason — 6종. SystemSignal.ComputeGuidance에서 사용. `rename_all = "snake_case"`. *(MUST)*

| Wire Value | Variant |
|-----------|---------|
| `"cpu_bottleneck"` | CpuBottleneck |
| `"gpu_bottleneck"` | GpuBottleneck |
| `"cpu_available"` | CpuAvailable |
| `"gpu_available"` | GpuAvailable |
| `"both_loaded"` | BothLoaded |
| `"balanced"` | Balanced |

**[MSG-095]** EnergyReason — 6종. SystemSignal.EnergyConstraint에서 사용. `rename_all = "snake_case"`. *(MUST)*

| Wire Value | Variant |
|-----------|---------|
| `"battery_low"` | BatteryLow |
| `"battery_critical"` | BatteryCritical |
| `"power_limit"` | PowerLimit |
| `"thermal_power"` | ThermalPower |
| `"charging"` | Charging |
| `"none"` | None |

> ⚠️ `EnergyReason::None`에는 명시적 `#[serde(rename = "none")]`이 적용되어 있다. 다른 변형은 `rename_all = "snake_case"`에 의해 자동 변환되나, `None`은 Rust 키워드와의 충돌 가능성으로 인해 명시적으로 지정되어 있다.

### 3.10 D-Bus SystemSignal [MSG-100 ~ MSG-104]

D-Bus 전송 경로 전용 메시지. **Externally tagged** (serde 기본 방식). `rename_all = "snake_case"`.

**[MSG-100]** SystemSignal — 4종 변형. *(MUST)*

| Tag Value (외부 키) | Variant | 필드 |
|-------------------|---------|------|
| `"memory_pressure"` | MemoryPressure | level, available_bytes, total_bytes, reclaim_target_bytes |
| `"compute_guidance"` | ComputeGuidance | level, recommended_backend, reason, cpu_usage_pct, gpu_usage_pct |
| `"thermal_alert"` | ThermalAlert | level, temperature_mc, throttling_active, throttle_ratio |
| `"energy_constraint"` | EnergyConstraint | level, reason, power_budget_mw |

#### MSG-101: MemoryPressure

| 필드 | 타입 | 범위 | 설명 |
|------|------|------|------|
| level | Level | 4종 | 메모리 압박 심각도 |
| available_bytes | u64 | ≥ 0 | 사용 가능 메모리 바이트 |
| total_bytes | u64 | > 0 | 전체 메모리 바이트 |
| reclaim_target_bytes | u64 | ≥ 0 | 회수 목표 바이트 |

```json
{"memory_pressure": {"level": "critical", "available_bytes": 1024, "total_bytes": 4096, "reclaim_target_bytes": 512}}
```

#### MSG-102: ComputeGuidance

| 필드 | 타입 | 범위 | 설명 |
|------|------|------|------|
| level | Level | 4종 | 컴퓨트 상태 심각도 |
| recommended_backend | RecommendedBackend | 3종 | 권장 컴퓨트 백엔드 |
| reason | ComputeReason | 6종 | 판단 사유 |
| cpu_usage_pct | f64 | [0.0, 100.0] | CPU 사용률 (퍼센트) |
| gpu_usage_pct | f64 | [0.0, 100.0] | GPU 사용률 (퍼센트) |

```json
{"compute_guidance": {"level": "warning", "recommended_backend": "cpu", "reason": "gpu_bottleneck", "cpu_usage_pct": 45.0, "gpu_usage_pct": 92.0}}
```

#### MSG-103: ThermalAlert

| 필드 | 타입 | 범위 | 설명 |
|------|------|------|------|
| level | Level | 4종 | 열 상태 심각도 |
| temperature_mc | i32 | — | 밀리섭씨 (예: 45000 = 45.0°C) |
| throttling_active | bool | true/false | OS 스로틀링 활성 여부 |
| throttle_ratio | f64 | [0.0, 1.0] | 1.0 = 스로틀 없음, 0.0 = 완전 스로틀 |

```json
{"thermal_alert": {"level": "warning", "temperature_mc": 45000, "throttling_active": false, "throttle_ratio": 1.0}}
```

#### MSG-104: EnergyConstraint

| 필드 | 타입 | 범위 | 설명 |
|------|------|------|------|
| level | Level | 4종 | 에너지 상태 심각도 |
| reason | EnergyReason | 6종 | 제약 사유 |
| power_budget_mw | u32 | ≥ 0 | 전력 예산 (밀리와트) |

```json
{"energy_constraint": {"level": "critical", "reason": "battery_low", "power_budget_mw": 3000}}
```

## 4. Alternative Behavior

해당 없음. 메시지 정의 문서이다. 메시지 처리의 대안 동작은 해당 컴포넌트 스펙(`22-manager-algorithms.md`, `32-engine-algorithms.md`)에서 다룬다.

## 5. Constraints

- **[CON-020]** 필드명 변경 금지: `shared` 크레이트의 serde 어노테이션이 와이어 포맷을 결정한다. 필드명이나 태그값을 변경하면 프로토콜 호환성이 파괴된다. *(MUST NOT)*

- **[CON-021]** 추가 필드 허용: `#[serde(default)]`로 신규 필드를 추가하면 구 버전 호환이 유지된다. 새 필드 추가는 하위 호환 변경이다. *(MAY)*

- **[CON-022]** 기존 필드 삭제 금지: 기존 필드를 삭제하면 구 버전 피어의 역직렬화가 실패한다. *(MUST NOT)*

#### 불변식

- **[INV-027]** shared 크레이트의 serde 어노테이션 변경은 프로토콜 버전 변경에 해당한다. *(MUST)*
- **[INV-028]** 새 필드 추가 시 반드시 `#[serde(default)]`를 적용하여 하위 호환성을 유지한다. *(MUST)*

## 6. Examples

### 6.1 완전한 세션 교환 예시

#### Step 1: Engine → Manager: Capability

```json
{"type":"capability","available_devices":["cpu","opencl"],"active_device":"cpu","max_kv_tokens":2048,"bytes_per_kv_token":256,"num_layers":16}
```

#### Step 2: Engine → Manager: Heartbeat

```json
{"type":"heartbeat","active_device":"cpu","compute_level":"normal","actual_throughput":12.5,"memory_level":"normal","kv_cache_bytes":524288,"kv_cache_tokens":256,"kv_cache_utilization":0.125,"memory_lossless_min":1.0,"memory_lossy_min":0.02,"state":"running","tokens_generated":50,"available_actions":["throttle","switch_hw","layer_skip"],"active_actions":[],"eviction_policy":"none","kv_dtype":"f16","skip_ratio":0.0}
```

#### Step 3: Manager → Engine: Directive (cross-domain 조합)

```json
{"type":"directive","seq_id":1,"commands":[{"type":"kv_evict_h2o","keep_ratio":0.48},{"type":"throttle","delay_ms":30}]}
```

#### Step 4: Engine → Manager: Response

```json
{"type":"response","seq_id":1,"results":[{"status":"ok"},{"status":"ok"}]}
```

### 6.2 EngineCommand 13종 전체 JSON

```json
{"type": "throttle", "delay_ms": 50}
{"type": "layer_skip", "skip_ratio": 0.25}
{"type": "kv_evict_h2o", "keep_ratio": 0.48}
{"type": "kv_evict_sliding", "keep_ratio": 0.6}
{"type": "kv_merge_d2o", "keep_ratio": 0.5}
{"type": "kv_streaming", "sink_size": 4, "window_size": 256}
{"type": "kv_quant_dynamic", "target_bits": 4}
{"type": "request_qcf"}
{"type": "restore_defaults"}
{"type": "switch_hw", "device": "opencl"}
{"type": "prepare_compute_unit", "device": "opencl"}
{"type": "suspend"}
{"type": "resume"}
```

### 6.3 CommandResult 3종 JSON

```json
{"status": "ok"}
{"status": "partial", "achieved": 0.7, "reason": "insufficient cache tokens"}
{"status": "rejected", "reason": "single backend"}
```

### 6.4 SystemSignal 4종 JSON (D-Bus 전송 참조용)

```json
{"memory_pressure":{"level":"critical","available_bytes":104857600,"total_bytes":4294967296,"reclaim_target_bytes":52428800}}
{"compute_guidance":{"level":"warning","recommended_backend":"cpu","reason":"gpu_bottleneck","cpu_usage_pct":45.0,"gpu_usage_pct":92.0}}
{"thermal_alert":{"level":"warning","temperature_mc":45000,"throttling_active":false,"throttle_ratio":1.0}}
{"energy_constraint":{"level":"critical","reason":"battery_low","power_budget_mw":3000}}
```

## 7. Rationale (non-normative)

### 왜 EngineCommand가 action-specific인가

초기 설계(docs/37)에서는 `SetComputeLevel`/`SetMemoryLevel` 같은 도메인 수준 명령 6종이었다. Manager가 cross-domain 최적 조합을 선택하는 아키텍처로 진화하면서, action-specific 명령(13종)이 더 정밀한 제어를 제공한다. Manager가 "KV eviction H2O 48% + throttle 30ms"라는 구체적 조합을 하나의 Directive로 전송할 수 있다. 현재 구현은 Manager가 구체적 명령 조합을 선택하여 전송하는(Manager-selects-and-commands) 아키텍처이다. Action Selection 아키텍처의 최종 결정(HR-1)에 따라 EngineDirective 구조가 확장될 수 있다(예: mode/pressure 필드 추가).

> **참고**: 동일 용어 `EngineDirective`가 현재 코드에서는 `{seq_id, commands[]}` 구조이나, 논문 설계(policy-design.md §3.2)에서는 `{mode, pressure, priority}` 구조로 제안되어 있다. HR-1 확정 전까지 이 스펙은 현재 코드 기준으로 기술한다.

### 왜 EngineStatus에 available_actions가 있는가

Engine만이 현재 상태에서 실행 가능한 액션을 알 수 있다. eviction 정책 설정 여부, KIVI 캐시 사용 여부 등은 Engine 내부 상태에 의존한다. Manager는 이 정보를 받아 Action Selector의 탐색 공간을 줄인다.

### 왜 ResourceLevel이 3단계인가

Emergency는 Manager 측에서 Suspend 명령으로 대체된다. Manager의 PI Controller가 Emergency를 감지하면 Suspend EngineCommand를 전송한다. 따라서 프로토콜 수준에서는 Normal/Warning/Critical 3단계로 충분하다.

### 왜 SystemSignal의 Level은 4단계인가

D-Bus 전송 경로 호환. D-Bus Transport가 Emergency Level을 수신하면 내부적으로 Suspend EngineCommand로 변환한다.

### 왜 EngineStatus가 16필드인가

11필드(원래)에서 5필드(available_actions, active_actions, eviction_policy, kv_dtype, skip_ratio)가 추가되었다. Manager의 Action Selector와 ReliefEstimator가 Engine의 현재 액션 상태를 알아야 정확한 의사결정을 할 수 있다. 새 필드는 `#[serde(default)]`로 하위 호환을 유지한다.
