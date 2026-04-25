# Invariants Catalog

> **TL;DR**: llm_rs2 전체 스펙에 산재된 불변식(INV-*)을 한 곳에 수집하고,
> 카테고리(Safety/Correctness/Performance/Compatibility)와
> 검증 방법(static/runtime/test)으로 분류한다.
> INV-001~076 (기존 59개) + INV-066~068 (CUDA 3개) + INV-080~085 (cross-cutting 6개) + INV-086~090 (LuaPolicy 5개, 2026-04) + INV-091~092 (Engine self-util 2개, 2026-04) + INV-093~105 (LuaPolicy DPP 13개, 2026-04) + INV-106~116 (LinUCB 11개, INV-113/114 제거; 9개 유효) + INV-117~119 (QCF × DPP 3개, 2026-04) + INV-120 (Plan × Partition 1개, 2026-04) + INV-121~125 (Dynamic Weight Swap Phase 1/2, 2026-04-24) + INV-126~128 (Weight Swap Phase 3 Manager 통합, 2026-04-24) + INV-129 (Weight Swap Phase 3.5 Plan invalidation, 2026-04-25) = 총 110개.

## 1. Purpose and Scope

이 문서는 전체 스펙에 정의된 모든 불변식을 카탈로그로 수집한다. 각 불변식의 원본 위치, 카테고리, 검증 방법을 테이블로 정리하여 불변식 누락을 방지하고 검증 계획 수립을 지원한다.

스펙 변경 시 이 카탈로그를 갱신하여 불변식의 일관성을 유지한다.

**이 파일이 명세하는 것:**

- 기존 스펙에 정의된 INV-001~076의 전수 수집
- 신규 cross-cutting 불변식 INV-080~085의 정의
- 카테고리별, 검증 방법별 분류
- 재확인 관계(restatement) 정리

**이 파일이 명세하지 않는 것:**

- 불변식의 상세 맥락 -- 원본 스펙 참조
- 개별 컴포넌트 아키텍처 -- 20~33번
- 프로토콜 상세 -- 10~12번

## 2. Definitions

### 카테고리 정의

| 카테고리 | 의미 |
|---------|------|
| **Safety** | 위반 시 시스템 크래시, 데이터 손실, 하드웨어 손상 가능. 최우선 보장. |
| **Correctness** | 위반 시 잘못된 동작(오답, 상태 불일치). 기능 정확성 보장. |
| **Performance** | 위반 시 성능 저하, 리소스 낭비. 효율 보장. |
| **Compatibility** | 위반 시 버전 간 호환 불가. 독립 배포 보장. |

### 검증 방법 정의

| 방법 | 의미 |
|------|------|
| **static** | 컴파일 타임 또는 코드 리뷰로 검증 (타입 시스템, feature gate, Cargo 의존 구조, trait bound, 모듈 의존) |
| **runtime** | 실행 중 assert, clamp, 조건 검사, AtomicU64 등으로 보장 |
| **test** | 단위 테스트, 통합 테스트, 프로퍼티 테스트, 장애 주입 테스트로 검증 |

### 표기 규칙

- 검증 칼럼에서 주 검증 방법을 먼저, 보조 방법을 괄호 없이 쉼표로 구분하여 기술한다.
- 재확인(restatement) 관계는 비고에 `=> INV-xxx` 형식으로 표기한다.

## 3. Specification

### 3.1 System/Component Invariants [INV-001 ~ INV-018]

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-001 | 00-overview SYS-001 | 시스템 = 2 독립 프로세스. Engine-Manager 직접 코드 의존 금지. Shared가 유일한 공유 의존성. | Safety | static | Cargo workspace 의존 구조 |
| INV-002 | 00-overview SYS-023 | NEON SIMD는 ARM64에서만 활성화. x86_64 등에서 NEON 코드 실행 금지. | Safety | static | `#[cfg(target_arch)]` |
| INV-003 | 00-overview SYS-032 | `config.json`의 `architectures`가 지원 목록에 없으면 로딩 거부. | Correctness | runtime | 로딩 시 검증 |
| INV-004 | 00-overview SYS-040 | QCF 수집 활성 상태에서 lossy action 실행 시 QcfMetric 생성 필수. | Correctness | test | CROSS-010 관련 |
| INV-005 | 00-overview SYS-050 | Manager 장애가 Engine 추론 루프를 중단시키지 않음. | Safety | test | 장애 주입. CROSS-010 |
| INV-006 | 00-overview SYS-051 | Engine 장애가 Manager 모니터링 루프를 중단시키지 않음. | Safety | test | 장애 주입. CROSS-010 |
| INV-010 | 01-architecture SYS-071 | Engine-Manager 직접 코드 의존 금지. Shared가 유일한 공유 의존성. | Safety | static | Cargo.toml. CROSS-020 |
| INV-011 | 01-architecture SYS-072 | Shared는 Engine/Manager 내부 구현에 의존 금지. | Safety | static | Cargo.toml. CROSS-020 |
| INV-012 | 01-architecture SYS-081 | Backend trait이 유일한 하드웨어 추상화점. Backend 우회 직접 호출 금지. | Correctness | static | 코드 리뷰 |
| INV-013 | 01-architecture SYS-087 | Monitor 스레드 장애가 다른 Monitor에 전파 금지. 독립 OS 스레드, 공유 상태 없음, mpsc만. | Safety | static, test | 아키텍처. CROSS-010 |
| INV-014 | 01-architecture SYS-094 | EngineDirective.seq_id는 세션 내 단조 증가. | Correctness | runtime | AtomicU64 |
| INV-015 | 01-architecture SYS-094 | Capability는 세션당 정확히 1회 전송. | Correctness | runtime, test | |
| INV-016 | 01-architecture SYS-096 | 동일 배타 그룹 액션 동시 활성화 금지. | Correctness | runtime, test | ActionSelector |
| INV-017 | 01-architecture SYS-098 | QCF 수집 활성 + lossy action 실행 시 QcfMetric 생성 필수. | Correctness | test | => INV-004 재확인 |
| INV-018 | 01-architecture SYS-070 | 추론 루프(Prefill/Decode)는 단일 스레드. Forward pass 중 다른 스레드가 모델/KV 캐시 동시 접근 금지. | Safety | static | 아키텍처 |

### 3.2 Protocol Invariants [INV-020 ~ INV-028]

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-020 | 10-protocol PROTO-074 | seq_id 단조 증가: `seq_id(N+1) > seq_id(N)`. | Correctness | runtime | AtomicU64 fetch_add |
| INV-021 | 10-protocol PROTO-074 | 동일 seq_id 재사용 금지. | Correctness | runtime | |
| INV-022 | 10-protocol PROTO-075 | 모든 Directive는 정확히 1개 Response를 유발. | Correctness | runtime, test | D-Bus 예외: 4.1절 참조 |
| INV-023 | 10-protocol PROTO-075 | `CommandResponse.seq_id == EngineDirective.seq_id`. | Correctness | runtime | |
| INV-024 | 10-protocol PROTO-075 | `len(CommandResponse.results) == len(EngineDirective.commands)`. | Correctness | runtime | |
| INV-025 | 11-protocol-messages MSG-071 | `len(CommandResponse.results) == len(EngineDirective.commands)`. | Correctness | runtime | => INV-024 재확인 |
| INV-026 | 11-protocol-messages MSG-073 | Engine은 수신한 seq_id에 대해서만 Response 전송. | Correctness | runtime | |
| INV-027 | 11-protocol-messages CON-020 | Shared serde 어노테이션 변경 = 프로토콜 버전 변경. | Compatibility | static | 코드 리뷰. CROSS-032 |
| INV-028 | 11-protocol-messages CON-021 | 새 필드 추가 시 `#[serde(default)]` 필수. 하위 호환 유지. | Compatibility | static | 코드 리뷰. CROSS-030 |

### 3.3 Manager Algorithm Invariants [INV-030 ~ INV-051]

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-030 | 22-manager-algorithms ALG-012 | `can_act = false`일 때 integral 미변경. | Correctness | runtime, test | |
| INV-031 | 22-manager-algorithms ALG-012 | `integral in [0, integral_clamp]` 항상 유지. | Correctness | runtime | clamp |
| INV-032 | 22-manager-algorithms ALG-021 | 에스컬레이션은 즉시. Normal에서 Critical 직행 가능. | Correctness | test | |
| INV-033 | 22-manager-algorithms ALG-021 | 디에스컬레이션은 반드시 1단계씩. Critical에서 Normal 직행 불가. | Correctness | test | |
| INV-034 | 22-manager-algorithms ALG-023 | `warning_release < warning_threshold`. | Correctness | runtime | 설정 검증 |
| INV-035 | 22-manager-algorithms ALG-023 | `critical_release < critical_threshold`. | Correctness | runtime | 설정 검증 |
| INV-036 | 22-manager-algorithms ALG-023 | `warning_threshold < critical_threshold`. | Correctness | runtime | 설정 검증 |
| INV-037 | 22-manager-algorithms ALG-031 | Warning 모드에서 Lossy 액션 선택 금지. | Correctness | runtime, test | filter |
| INV-038 | 22-manager-algorithms ALG-031 | 이미 활성 중인 액션은 재선택 금지. | Correctness | runtime | filter |
| INV-039 | 22-manager-algorithms ALG-032 | Lossless 액션의 cost = 항상 0. | Correctness | runtime | |
| INV-040 | 22-manager-algorithms ALG-032 | QCF 값 없는 Lossy 액션 = INFINITY cost (사실상 선택 불가). | Correctness | runtime | |
| INV-041 | 22-manager-algorithms ALG-033 | 동일 배타 그룹 액션은 하나의 조합에 동시 미포함. | Correctness | runtime, test | => INV-016 재확인 |
| INV-042 | 22-manager-algorithms ALG-033 | 조합의 총 latency 악화 > latency_budget이면 배제. | Performance | runtime | |
| INV-043 | 22-manager-algorithms ALG-033 | 완전 해소 가능 조합 > best-effort 조합 (항상 우선). | Correctness | runtime | |
| INV-044 | 22-manager-algorithms ALG-035 | parametrize 출력 value는 [range.min, range.max] 범위 내. | Correctness | runtime | clamp |
| INV-045 | 22-manager-algorithms ALG-035 | primary_domain 매핑: SwitchHw/Throttle/LayerSkip -> Compute, 나머지 -> Memory. | Correctness | static | 코드 |
| INV-046 | 22-manager-algorithms ALG-044 | RLS gain vector k = f(P, phi). lambda는 망각 인수. | Correctness | test | |
| INV-047 | 22-manager-algorithms ALG-044 | bias는 W 갱신 후 잔여 오차에 EMA(lr=0.1) 적용. RLS P matrix는 W만. | Correctness | test | |
| INV-048 | 22-manager-algorithms ALG-044 | P matrix: D x D 대칭 양정치. 초기값 100 * I. | Correctness | runtime | 초기화 |
| INV-049 | 22-manager-algorithms ALG-046 | `lambda in (0, 1]`. lambda=1.0이면 forgetting 없음. | Correctness | runtime | 설정 검증 |
| INV-050 | 22-manager-algorithms ALG-061 | 관찰 relief의 latency 차원 = 항상 0.0. | Correctness | runtime | |
| INV-051 | 22-manager-algorithms ALG-061 | 동시 적용 시 전체 relief가 각 액션에 귀속 (개별 분리 불가). | Correctness | runtime | 설계 한계 |

### 3.4 Engine Architecture Invariants [INV-060 ~ INV-065]

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-060 | 30-engine ENG-032 | `CommandExecutor.poll()`은 토큰당 최대 1회 호출. | Performance | static | 코드 구조 |
| INV-061 | 30-engine ENG-032 | ExecutionPlan: 생성 즉시 소비, 다음 poll() 전 폐기. 1회성. | Safety | static | 코드 구조 |
| INV-062 | 30-engine ENG-032 | Suspend 포함 ExecutionPlan: evict/switch_device/prepare_device = None. | Safety | runtime | poll step 5. => INV-074 |
| INV-063 | 30-engine ENG-023 | MessageLoop 스레드는 Transport의 유일한 소유자. | Safety | static | ownership |
| INV-064 | 30-engine ENG-033 | heartbeat_interval 내 최소 1회 Heartbeat 전송 (poll 호출 시). | Correctness | runtime | CROSS-060 |
| INV-065 | 30-engine ENG-013 | Backend trait 구현체는 `Send + Sync`. | Safety | static | trait bound |
| INV-066 | 34-engine-cuda ENG-CUDA-013 | CudaBackend 초기화 시 CC ≥ sm_72 검증. 미달 시 에러 반환. | Safety | runtime | `cudaGetDeviceProperties` |
| INV-067 | 34-engine-cuda ENG-CUDA-050 | `cuda`와 `opencl` feature는 상호 배타적. 동시 활성화 시 컴파일 에러. | Compatibility | static | `build.rs` compile_error! |
| INV-068 | 34-engine-cuda ENG-CUDA-011 | CudaBackend는 llama.cpp PTX 커널 + cudarc만 사용. 자체 CUDA 커널을 작성하지 않음. | Correctness | static | 코드 리뷰 |

### 3.5 Engine State Machine Invariants [INV-070 ~ INV-076]

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-070 | 31-engine-state ENG-ST-011 | `OperatingMode.from_levels()` = 순수 함수. 이전 상태 미의존. | Correctness | static | 함수 시그니처 |
| INV-071 | 31-engine-state ENG-ST-020 | EngineState 전이는 CommandExecutor 내부에서만. 외부 직접 변경 금지. | Correctness | static | 캡슐화 |
| INV-072 | 31-engine-state ENG-ST-060 | `resolve_conflicts()`: Suspend 존재 시 반환 = `[Suspend]`. | Safety | runtime, test | |
| INV-073 | 31-engine-state ENG-ST-060 | `resolve_conflicts()`: RestoreDefaults는 다른 제약 없을 때만 반환. | Correctness | runtime, test | |
| INV-074 | 31-engine-state ENG-ST-034 | `plan.suspended == true`이면 evict/switch_device/prepare_device = None. | Safety | runtime | poll step 5. => INV-062 |
| INV-075 | 31-engine-state ENG-ST-033 | Resume: compute/memory_level을 Normal로, throttle_delay_ms를 0으로. | Correctness | runtime | |
| INV-076 | 31-engine-state ENG-ST-033 | RestoreDefaults: active_actions 비움, throttle를 0으로, compute/memory를 Normal로. | Correctness | runtime | |

### 3.6 Cross-cutting Invariants [INV-080 ~ INV-085]

기존 스펙에서 명시적 INV-* 태그 없이 MUST/MUST NOT으로 기술되었으나, cross-cutting 관점에서 불변식으로 승격한 항목이다.

| ID | 원본 근거 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|----------|----------|---------|------|------|
| INV-080 | SYS-064, MGR-C01, ENG-070 | async 런타임 사용 금지. std::thread + mpsc만 허용. | Safety | static | Cargo.toml, 코드 리뷰. CROSS-010 |
| INV-081 | SYS-065, ENG-071, CON-011 | IPC 직렬화는 JSON (serde_json) 전용. 바이너리 직렬화 포맷 금지. | Compatibility | static | Cargo.toml. CROSS-022 |
| INV-082 | SYS-093, PROTO-041, ENG-072 | 1:1 단일 클라이언트 연결. 다중 Engine 동시 연결 금지. | Safety | runtime | accept 모델. CROSS-090 |
| INV-083 | MGR-C06 | PI Controller output은 [0, 1] 범위 내. | Correctness | runtime | clamp |
| INV-084 | MGR-C07, MGR-C08 | ActionSelector = stateless. ReliefEstimator.predict = 읽기 전용. | Correctness | static | 코드 구조 |
| INV-085 | MGR-C10 | Normal 모드에서 액션 미발행. | Correctness | runtime, test | 조건 검사 |

### 3.7 LuaPolicy Relief Adaptation Invariants [INV-086 ~ INV-090]

2026-04 기본 정책이 LuaPolicy로 이관되며 도입된 불변식. HierarchicalPolicy의 RLS 기반 INV-046~051과 독립적으로 존재한다. `#[cfg(feature = "hierarchical")]` 여부와 무관하게 LuaPolicy 경로에서 항상 MUST.

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-086 | 20-manager MGR-091, 23-manager-data MGR-DAT-070/071 | `EwmaReliefTable.save()`는 `entries`(EWMA 누적값 + observation_count)만 직렬화한다. raw observation 이력, alpha, defaults는 저장 대상이 아니다. | Correctness | static, test | JSON 스키마 검증 |
| INV-087 | 22-manager-algorithms MGR-ALG-080 | `observation_count == 0`인 액션에 대한 첫 `observe()`는 α 평활을 우회하고 관측값을 직접 대입한다 (cold-start 가속). | Correctness | runtime, test | |
| INV-088 | 22-manager-algorithms MGR-ALG-082, 12-protocol SEQ-055 | `decide()`가 다중 커맨드를 반환하면 `ObservationContext`는 **첫 번째** 커맨드의 action만 기록하고, 나머지 액션에 대한 관측은 수행되지 않는다. 관찰 대기 중 새 커맨드가 발행되면 기존 관찰은 학습 없이 폐기된다. | Correctness | test | 학습 누락을 의도적으로 허용 |
| INV-089 | 22-manager-algorithms MGR-ALG-083, 23-manager-data MGR-DAT-073 | 6D relief 관측에서 차원 0~4(gpu, cpu, memory, thermal, latency)는 `before - after`로, 차원 5(main_app_qos)는 `after - before`로 계산한다. 부호 반전은 차원 5에만 적용된다. | Correctness | runtime, test | |
| INV-090 | 22-manager-algorithms MGR-ALG-081 | `EwmaReliefTable.predict(action)`은 **읽기 전용**이다. `entries`에 새 항목을 생성하거나 기존 값을 수정하지 않는다 (INV-084의 LuaPolicy 구체화). | Correctness | static, test | 코드 구조 |

### 3.8 Engine Self-Utilization Invariants [INV-091 ~ INV-092]

2026-04 Phase 1에서 Engine이 Heartbeat에 자신의 프로세스 단위 CPU/GPU 사용률을 실어 보낼 때 적용된다. 대응 필드는 `EngineStatus.self_cpu_pct`, `self_gpu_pct` (MSG-060, MGR-DAT-075, MGR-DAT-076).

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-091 | 11-protocol-messages MSG-067, 23-manager-data MGR-DAT-075/076 | `self_cpu_pct`, `self_gpu_pct` ∈ [0.0, 1.0]. 범위 밖 값은 송출 전에 clamp 한다. | Correctness | runtime, test | Engine 측 clamp |
| INV-092 | 11-protocol-messages MSG-067/068, 23-manager-data MGR-DAT-075/076 | 측정 실패 시 `self_cpu_pct`/`self_gpu_pct`는 0.0 fallback. Heartbeat 송출은 차단되지 않는다. | Robustness/Correctness | runtime, test | MSG-061 하위호환 규약과 동일한 원칙 |

### 3.9 LuaPolicy DPP Algorithm Invariants [INV-093 ~ INV-105]

2026-04 LuaPolicy DPP (Drift-Plus-Penalty) 결정 알고리즘의 불변식. `policy_default.lua` v2.1.0 기준. 대응 명세: `22-manager-algorithms.md` 3.8절 (MGR-DPP-010~070).

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-093 | 22-manager-algorithms MGR-DPP-010 | DPP objective에서 Z_k=0인 도메인의 relief는 score에 기여하지 않는다. θ_warn 이하 도메인은 결정에 무영향. | Correctness | test | Z_k 가중으로 자연 보장 |
| INV-094 | 22-manager-algorithms MGR-DPP-011 | Z_k >= 0. p_k <= θ_warn이면 Z_k = 0. | Correctness | runtime, test | max(0, ...) 구조 |
| INV-095 | 22-manager-algorithms MGR-DPP-011 | W_WARN < W_CRIT < W_EMERG. emergency 도메인은 항상 더 큰 Z_k를 가진다. | Correctness | static | 상수 관계 |
| INV-096 | 22-manager-algorithms MGR-DPP-012 | 모든 도메인에서 θ_warn < θ_crit < θ_emerg. | Correctness | static, test | 설정 검증 |
| INV-097 | 22-manager-algorithms MGR-DPP-020 | A_safe가 비어있으면 no-op. latency floor 위반 action은 선택 불가. | Safety | test | hard constraint |
| INV-098 | 22-manager-algorithms MGR-DPP-030 | Joint action은 최대 2개의 component action으로 구성. | Correctness | static | 레지스트리 구조 |
| INV-099 | 22-manager-algorithms MGR-DPP-030 | Joint relief fallback은 component relief의 선형 합. 학습 joint relief가 우선. | Correctness | test | cold start 경로 |
| INV-100 | 22-manager-algorithms MGR-DPP-040 | Safety Override는 compute/thermal emergency에서만 throttle을 추가. Memory emergency는 대상 아님. | Safety | test | throttle은 compute 액션 |
| INV-101 | 22-manager-algorithms MGR-DPP-040 | Safety Override는 DPP 결과에 throttle을 **추가**만 한다. 기존 action을 제거하지 않는다. | Safety | test | additive only |
| INV-102 | 22-manager-algorithms MGR-DPP-060 | Trigger가 없으면 새 action을 발행하지 않는다. Step 2~6 미실행. | Correctness | test | restore만 허용 |
| INV-103 | 22-manager-algorithms MGR-DPP-060 | Score 동점 시 먼저 열거된 candidate가 선택된다. (strict greater-than 비교) | Correctness | test | 결정론적 tie-break |
| INV-104 | 22-manager-algorithms MGR-DPP-061 | TriggerEngine hysteresis에서 enter > exit. 양의 간격 보장. | Correctness | static, test | 설정 검증 |
| INV-105 | 22-manager-algorithms MGR-DPP-061 | TBT baseline EWMA warmup 기간(20 tokens) 동안 `tbt_degraded` trigger는 활성화되지 않는다. | Safety | runtime, test | premature trigger 방지 |

### 3.10 LinUCB Exploration Bonus Invariants [INV-106 ~ INV-116]

2026-04 LinUCB additive exploration bonus의 불변식. 대응 명세: `22-manager-algorithms.md` 3.9절 (MGR-LUCB-010~070). v2.1.0 기준.

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-106 | 22-manager-algorithms MGR-LUCB-010 | `linucb_alpha=0`이면 `ucb_bonus=0`이 되어 기존 DPP (MGR-DPP-010)와 수학적으로 동일하다. | Correctness | test | 하위 호환 보장 |
| INV-107 | 22-manager-algorithms MGR-LUCB-011 | `ucb_bonus()`는 읽기 전용이다. P matrix를 변경하지 않는다. | Correctness | test | MGR-C08 정신 계승 |
| INV-108 | 22-manager-algorithms MGR-LUCB-014 | σ >= 0. `sqrt(max(0, φᵀ·P·φ))`로 비음수가 보장된다. | Correctness | test | sqrt 입력 비음수 |
| INV-109 | 22-manager-algorithms MGR-LUCB-012 | Feature vector (13D)의 모든 원소는 [0, 1] 범위이다. | Correctness | runtime, test | 입력 정규화 |
| INV-110 | 22-manager-algorithms MGR-LUCB-013 | P (= V⁻¹)는 항상 positive semi-definite이다. Sherman-Morrison update가 보존한다. | Correctness | static | Sherman-Morrison 보존 |
| INV-111 | 22-manager-algorithms MGR-LUCB-015 | `linucb_alpha >= 0`. AdaptationConfig에서 검증한다. | Correctness | runtime, test | config 검증 |
| INV-112 | 22-manager-algorithms MGR-LUCB-015 | 동일 φ에 대해, P matrix update 후 σ는 단조 감소한다 (P shrink). β_t 스케줄 없음. | Correctness | test | 수렴 보장 |
| INV-113 | 22-manager-algorithms MGR-LUCB-020 | `[REMOVED]` Pessimistic safe set은 구현하지 않는다. 기존 DPP safe set (MGR-DPP-020) 유지. | — | — | v2.1.0에서 제거 |
| INV-114 | 22-manager-algorithms MGR-LUCB-020 | `[REMOVED]` Pessimistic safe set 관련. INV-113과 함께 제거. | — | — | v2.1.0에서 제거 |
| INV-115 | 22-manager-algorithms MGR-LUCB-030 | EWMA observe와 LinUCB update는 동일 시점에 호출된다. 학습 데이터 불일치 없음. | Correctness | test | 학습 데이터 일관성 |
| INV-116 | 22-manager-algorithms MGR-LUCB-070 | `linucb_alpha=0`이면 DPP 결정은 §3.8과 비트 단위 동일 결과를 산출한다. | Correctness | test | INV-106 재확인, 런타임 fallback |

### 3.11 QCF × DPP Quality Penalty Invariants [INV-117 ~ INV-119]

2026-04 QCF × DPP 통합 (v2.2.0)의 불변식. 대응 명세: `22-manager-algorithms.md` 3.8절 (MGR-DPP-010, MGR-DPP-013, MGR-DPP-020). `policy_default.lua` v2.2.0 기준.

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-117 | 22-manager-algorithms MGR-DPP-010 | QCF cache miss인 action의 qcf_cost는 0으로 처리한다 (score에서 penalty 없음). should_request_qcf()가 자동 선발행을 수행한다. | Correctness | test | cache miss fallback |
| INV-118 | 22-manager-algorithms MGR-DPP-020 | Emergency level에서 quality floor은 비활성(∞)이다. lossy action이 항상 safe set에 포함될 수 있어야 한다. | Safety | test | Emergency escape hatch |
| INV-119 | 22-manager-algorithms MGR-DPP-020 | pressure level이 높아질수록 quality floor이 완화된다 (0.30 → 0.60 → 0.90 → ∞). 압박이 심할수록 더 많은 품질 훼손을 허용한다. | Correctness | test | monotonic floor relaxation |

### 3.12 Plan × Tensor Partition Invariants [INV-120]

2026-04 Plan × Tensor Partition 통합 (ENG-ALG-200)의 불변식. 대응 명세: `arch/plan_partition_integration.md` A.6.2, A.11 R-PP2.

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-120 | arch/plan_partition_integration.md A.6.2 | FullKernelPlan이 PartitionStep을 포함할 때, 각 PartitionStep::run 진입 시 PartitionPlanContext.ratio_generation_at_build와 PartitionContext.ratio_generation을 비교한다. mismatch면 PlanInvalidated를 반환하며 caller는 plan을 재빌드하거나 forward_gen으로 fallback해야 한다. | Safety/Correctness | runtime | AtomicU64 generation 비교 |

### 3.13 Weight Swap Invariants [INV-121 ~ INV-128]

2026-04-24 Dynamic Weight Swap (Manager 신호 기반 런타임 교체)의 불변식. 이전 Phase A 정적 노선은 **폐기**되었으며 `ENG-DAT-091` ID는 재사용 금지.

- **Phase 1/2 (Engine 내부 인프라)**: INV-121~125. 대응 명세: `32-engine-algorithms.md` 3.12.1~3.12.7 (ENG-ALG-210~214, ENG-ALG-214-SNAP), `33-engine-data.md` 3.17~3.20 (ENG-DAT-090/092/093/094).
- **Phase 3 (Manager 통합)**: INV-126~128. 대응 명세: `32-engine-algorithms.md` 3.12.8~3.12.12 (ENG-ALG-214-ROUTE, ENG-ALG-215~218), `33-engine-data.md` 3.21 (ENG-DAT-095).
- **Arch**: `arch/weight_swap.md` v4 (Phase 3 Manager 통합 반영).

**교차 참조**:
- **INV-120** (Plan × Partition stale): `TransformerModel::ratio_generation`은 INV-120의 감지 키와 동일 소스이다. `SwapExecutor`가 batch 완료 후 정확히 1회 bump하여 plan invalidation을 일으킨다 (ENG-ALG-211 step (e)).
- **INV-121 ↔ INV-123**: 토큰 경계 기반 per-token snapshot(INV-121)과 `ArcSwap::store` 단일 원자 단계(INV-123)는 **쌍으로** forward 비차단성을 보장한다. 한쪽만 성립해도 안전 보장 불충분.
- **INV-122 ↔ INV-121**: dtype 혼합은 정상 상태이므로 INV-122의 수치 임계값은 "혼합 상태 자체"가 아닌 "혼합된 forward 결과의 근접성"에 적용된다.

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-121 | 32-engine-algorithms 3.12.7, ENG-ALG-214-SNAP | Forward 재진입 금지: 토큰 진입 시 per-layer `Arc<LayerWeights>` snapshot을 한 번 획득하고 토큰 내내 재사용. mid-token swap은 현재 토큰에 관찰 불가 (다음 토큰 경계부터 관측). stale/half-swapped 상태 관찰 0건. | Correctness | test | INV-120과 동일 메커니즘 (ratio_generation은 플랜 경로), per-token snapshot은 forward 경로 |
| INV-122 | 32-engine-algorithms 3.12.6 | Dynamic swap 후 forward 결과는 primary baseline 대비 logit NMSE ≤ 0.01, top-5 overlap ≥ 0.9, top-1 match ratio ≥ 0.95를 충족한다. **layer 간 dtype 불균일(혼합 상태)은 본 설계의 정상 상태이며 불변식 위반이 아니다** — ratio 기반 swap의 본질이다. | Correctness | test | Llama/Qwen 양쪽, ratio 0.25/0.5/1.0. dtype mix allowed. |
| INV-123 | 32-engine-algorithms 3.12.2 | Swap 단위는 `LayerSlot.weights.store()` 호출 1회이며 단일 원자 단계로 완결된다. 토큰 경계 밖(= forward가 snapshot을 재획득하지 않는 구간)에 발생한 swap은 **다음 토큰부터** 관측된다. Partial state(반만 교체된 snapshot 등)는 외부에 절대 노출되지 않는다. | Safety/Correctness | test | lock-free atomicity + per-token snapshot 경계 |
| INV-124 | 33-engine-data 3.18 | `LayerSlot::current_dtype`의 값은 해당 slot이 현재 노출하는 `weights` snapshot의 실제 tensor dtype과 항상 일치한다. Swap 과정에서 `current_dtype` 갱신과 `weights` snapshot 교체는 동일 논리 단계에서 수행된다. | Correctness | test | dtype consistency |
| INV-125 | 33-engine-data 3.19, 3.20 | `TransformerModel.secondary_mmap`(구 `TransformerWeights::secondary_mmap`)이 `Some`인 동안 해당 `Arc<SecondaryMmap>`은 drop되지 않는다. Swap 도중 mmap unmap 금지. 모델 lifetime 동안 생존 보장. | Safety | test | mmap lifetime. 보관 위치는 flat 배치(ENG-DAT-093)의 `TransformerModel` 필드. |
| INV-126 | 32-engine-algorithms 3.12.8 (ENG-ALG-214-ROUTE), MSG-082 | `EngineCommand::SwapWeights.target_dtype`에 `DtypeTag::Q4_0` 이외의 variant가 들어오면 `SwapExecutor`까지 도달하지 않고 `CommandResult::Rejected { reason: "UnsupportedDtype" }`로 처리된다. Phase 3 범위 밖 dtype은 payload 호환성 확보용 reserved variant이며, 실행 경로는 panic 없이 명시적 reject만 수행한다. | Safety/Correctness | test | reserved dtype rejection. `F16`/`F32`/`Q8_0` variant 주입 시 Rejected 응답 확인. |
| INV-127 | 32-engine-algorithms 3.12.9, 3.12.10, 33-engine-data 3.21 | `QuantNoiseTable::epsilon(i).is_none()`인 layer(=계산 실패로 NaN이 저장된 layer)는 `WeightSwapDecider`에서 swap 후보로 선택되지 않는다. `QuantNoiseTable` 자체가 없으면 `SwapWeights` 경로는 INV-126과 상호 배타적으로 더 앞단(NoSecondary reject)에서 막힌다. | Correctness | test | NaN layer exclusion. per_layer[i] = NaN 주입 → decider 출력에 i 미포함 확인. |
| INV-128 | 32-engine-algorithms 3.12.12 (ENG-ALG-218) | `ImportanceCollector`가 `Armed` 또는 `Collecting` 상태로 prefill이 진행되었다면, 해당 prefill 종료 시 반드시 `EngineMessage::QcfEstimate`(MSG-084 확장)가 1회 송출되고 collector 상태는 `Idle`로 복귀한다. armed 상태 누수 금지. | Correctness | test | collector leak 검출. RequestQcf → prefill → QcfEstimate 송출 시퀀스의 완결성. |

### 3.14 Weight Swap × Plan Invalidation Invariants [INV-129]

2026-04-25 Weight Swap Phase 3.5 (`FullKernelPlan` × `TransformerModel::ratio_generation` 통합)의 불변식. 대응 명세: `32-engine-algorithms.md` 3.12.13~3.12.14 (ENG-ALG-219, ENG-ALG-220), `arch/weight_swap.md` §2.2.2.

**교차 참조**:
- **INV-120** (Plan × Partition stale): per-`PartitionPlanContext` 검사. INV-129와 **OR로 결합**되어 redundancy를 형성한다.
- **INV-121** (per-token snapshot): forward 경로의 layer Arc snapshot. INV-129의 plan 경로 검사와 동일 시점(token entry)에 captured되어야 한다 (ENG-ALG-220).

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-129 | 32-engine-algorithms 3.12.13 (ENG-ALG-219) | `FullKernelPlan::execute()` 진입 시 `plan.ratio_generation_at_build`와 `TransformerModel::ratio_generation`(현재) 값을 `Acquire` load로 1회 비교한다. mismatch 시 `PlanInvalidated`를 반환하며 caller는 plan 재빌드 또는 `forward_gen` fallback을 수행한다. INV-120(per-partition context)과 **독립적**으로 동작하며 OR 결합된다. weight swap(ENG-ALG-211 step (e) bump) 및 partition re-prep 모두 trigger가 될 수 있다. | Safety/Correctness | runtime | AtomicU64 Acquire load 비교. Plan 경로 stale 감지의 전역 trigger. |

## 4. Alternative Behavior

### 4.1 INV-022 D-Bus 예외

INV-022 (모든 Directive = 정확히 1개 Response)는 D-Bus 경로에서 완화된다 (SEQ-104). D-Bus는 fire-and-forget이므로 Response 전달이 보장되지 않는다. D-Bus Transport의 seq_id는 자체 카운터로 관리되며, Manager는 D-Bus 경로에서 Response를 기대하지 않는다.

### 4.2 INV-025 중복 참고

INV-025는 INV-024와 동일한 내용이다 (`len(results) == len(commands)`). 11-protocol-messages.md에서 별도 ID를 부여하여 해당 문서 내 참조 편의를 제공한다. 이 카탈로그에서는 양쪽 모두 수록하되 재확인 관계를 명시한다.

### 4.3 재확인(Restatement) 관계

일부 INV는 다른 INV 또는 요구사항의 재확인이다. 중복이 아니라 각 스펙 문서 내에서의 자기 완결성을 위한 것이다.

| INV | 재확인 대상 | 비고 |
|-----|-----------|------|
| INV-017 | INV-004 | 01-architecture에서 00-overview 재확인 |
| INV-025 | INV-024 | 11-protocol-messages에서 10-protocol 재확인 |
| INV-041 | INV-016 | 22-manager-algorithms에서 01-architecture 재확인 |
| INV-062 | INV-074 | 30-engine과 31-engine-state에서 동일 내용 |
| INV-116 | INV-106 | MGR-LUCB-070에서 MGR-LUCB-010 재확인 (linucb_alpha=0 시 동일 결과) |

## 5. Constraints

### 5.1 카테고리별 통계

| 카테고리 | 개수 | 비율 |
|---------|------|------|
| Safety | 18 | 23% |
| Correctness | 57 | 71% |
| Performance | 2 | 3% |
| Compatibility | 3 | 4% |
| **합계** | **80** | **100%** |

> **참고**: INV-113, INV-114는 v2.1.0에서 REMOVED (pessimistic safe set 제거). 카운트에서 제외.
> INV-117~119는 v2.2.0 (QCF × DPP)에서 추가. INV-121~122는 Weight Swap Phase A에서 추가.
> INV-129는 Weight Swap Phase 3.5 (Plan × Weight Swap stale detection)에서 추가. 카테고리는 Safety/Correctness 양쪽이며, 통계는 Safety로 1회 카운트한다.

### 5.2 검증 방법별 통계

| 검증 방법 | 주 검증 | 보조 검증 포함 | 설명 |
|----------|---------|-------------|------|
| static | 21 | 25 | Cargo 의존 구조, feature gate, trait bound, 코드 구조 |
| runtime | 33 | 40 | assert, clamp, 조건 검사, AtomicU64 |
| test | 21 | 37 | 단위/통합/프로퍼티/장애 주입 테스트 |

다수의 불변식이 2개 이상의 검증 방법을 병용한다.

### 5.3 유지보수 규칙

- 새 스펙 문서에 INV-*를 추가할 때 이 카탈로그에도 반드시 항목을 추가한다. *(MUST)*
- 기존 INV-*의 의미를 변경할 때 이 카탈로그의 해당 행을 갱신한다. *(MUST)*
- INV-*를 삭제(폐기)할 때 이 카탈로그에서 행을 제거하지 않고, 비고에 `DEPRECATED` 표기한다. *(SHOULD)*
- ID 번호는 재사용하지 않는다. *(MUST NOT)*

## 6. Examples

### 예시 1: 불변식 위반 탐지 -- INV-016 (배타 그룹)

```
상황: ActionSelector가 KvEvictH2o(C4)와 KvEvictSliding(C5)를 동일 조합에 포함
탐지: runtime -- ActionSelector의 exclusion group 필터가 조합 생성 시 거부
검증: test -- 배타 그룹 조합을 시도하는 단위 테스트

조합 생성 루프:
  for each subset of candidates:
    if subset contains two actions from same exclusion group:
      skip  // INV-016 보장
```

### 예시 2: 불변식 검증 매핑 -- static 검증

```
INV-001 (프로세스 분리):
  검증 위치: Cargo.toml
  검증 방법: engine/Cargo.toml의 [dependencies]에 manager 크레이트 없음
             manager/Cargo.toml의 [dependencies]에 engine 크레이트 없음
             양측 모두 shared만 의존
  자동화: CI의 cargo dependency graph 검사

INV-002 (NEON ARM64 한정):
  검증 위치: engine/src/backend/cpu_neon.rs
  검증 방법: 모든 NEON 코드가 #[cfg(target_arch = "aarch64")] 내부
  자동화: x86_64 빌드에서 NEON 심볼 부재 확인
```

### 예시 3: Safety 카테고리 불변식 전체 목록

Safety 카테고리 (18개) -- 위반 시 크래시/데이터 손실/하드웨어 손상 가능:

| ID | 요약 |
|----|------|
| INV-001 | 2 독립 프로세스, 직접 의존 금지 |
| INV-002 | NEON은 ARM64에서만 |
| INV-005 | Manager 장애가 Engine 추론 미중단 |
| INV-006 | Engine 장애가 Manager 모니터링 미중단 |
| INV-010 | Engine-Manager 직접 의존 금지 (재확인) |
| INV-011 | Shared가 Engine/Manager에 미의존 |
| INV-013 | Monitor 스레드 장애 미전파 |
| INV-018 | 추론 루프 단일 스레드 |
| INV-061 | ExecutionPlan 1회성 |
| INV-062 | Suspend 시 evict/switch/prepare = None |
| INV-063 | MessageLoop = Transport 유일 소유자 |
| INV-065 | Backend = Send + Sync |
| INV-072 | Suspend 존재 시 반환 = [Suspend] |
| INV-074 | suspended이면 evict/switch/prepare = None |
| INV-080 | async 런타임 금지 |
| INV-082 | 1:1 단일 클라이언트 |
| INV-125 | secondary mmap lifetime 보장 |
| INV-129 | Plan × Weight Swap stale detection |

## 7. Rationale (non-normative)

### 왜 불변식을 한 곳에 수집하는가

전체 스펙에 산재된 65개 불변식을 개별 스펙에서만 관리하면, 다음 문제가 발생한다:

1. **누락 위험**: 새 코드가 어떤 불변식을 위반하는지 전체 스펙을 교차 참조해야 알 수 있다.
2. **검증 사각지대**: 어떤 불변식이 테스트로 검증되는지, 어떤 것이 static으로만 보장되는지 파악하기 어렵다.
3. **우선순위 혼동**: Safety 불변식과 Performance 불변식의 중요도 차이를 인식하기 어렵다.

카탈로그로 수집하면 검증 계획 수립, 코드 리뷰 체크리스트 작성, 테스트 커버리지 분석이 가능해진다.

### 왜 INV-080~085를 신규 추가하는가

기존 스펙에서 MUST/MUST NOT으로 기술되었지만 명시적 INV-* ID가 없던 cross-cutting 규칙들이다. 여러 컴포넌트에 걸쳐 적용되므로 개별 스펙에 INV를 부여하기 어려웠으나, 불변식 카탈로그의 완전성을 위해 ID를 부여한다. 원본 근거(SYS-064, MGR-C01, ENG-070 등)는 변경 없이 유지된다.

### 왜 재확인(restatement)을 허용하는가

각 스펙 문서의 자기 완결성을 위해 관련 불변식을 재기술하는 것을 허용한다. 단, 이 카탈로그에서 재확인 관계를 명시하여 원본과 사본이 불일치하는 것을 방지한다. 원본이 변경되면 재확인 INV도 동기화해야 한다.

### Safety 불변식의 우선순위

Safety 카테고리의 16개 불변식은 다른 카테고리에 우선한다. 코드 변경 시 Safety 불변식 위반 여부를 최우선으로 검토한다. Performance 불변식(2개)은 최적화 과정에서 의도적으로 완화될 수 있으나, Safety 불변식은 절대 완화 불가이다.
