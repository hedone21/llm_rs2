# Invariants Catalog

> **TL;DR**: llm_rs2 전체 스펙에 산재된 불변식(INV-*)을 한 곳에 수집하고,
> 카테고리(Safety/Correctness/Performance/Compatibility)와
> 검증 방법(static/runtime/test)으로 분류한다.
> INV-001~076 (기존 59개) + INV-066~068 (CUDA 3개) + INV-080~085 (cross-cutting 6개) + INV-086~090 (LuaPolicy 5개, 2026-04) + INV-091~092 (Engine self-util 2개, 2026-04) + INV-093~105 (LuaPolicy DPP 13개, 2026-04) + INV-106~116 (LinUCB 11개, INV-113/114 제거; 9개 유효) + INV-117~119 (QCF × DPP 3개, 2026-04) + INV-120 (Plan × Partition 1개, 2026-04) + INV-121~125 (Dynamic Weight Swap Phase 1/2, 2026-04-24) + INV-126~128 (Weight Swap Phase 3 Manager 통합, 2026-04-24) + INV-129 (Weight Swap Phase 3.5 Plan invalidation, 2026-04-25) + INV-130 (Weight Swap Phase 3.6 Noshuffle SOA coherence, 2026-04-25) + INV-131~134 (Weight Swap Phase 3.7 SOA re-conversion + AUF format, 2026-04-25) + INV-135~136 (AUF lm_head Q4_0 사전 변환, Phase 6 Sprint G-1, 2026-04-26) + INV-137~139 (AUF v0.2 multi-dtype variant, 2026-04-27) + INV-140~143 (Weight Swap Phase 6.5 Overhead Reduction, 2026-05-07) + INV-147~150 (Intra-forward Layer-aligned Swap LISWAP-4, 2026-05-08) + INV-151~155 (QNN OpPackage M1, 2026-05-09) + INV-156~165 (QNN OpPackage M2 layer graph, 2026-05-09) + INV-166~180 (QNN OpPackage M3 backend wire-up, 2026-05-10) + INV-181~188 (QNN OpPackage M4 async chunk swap placeholder, 2026-05-10) + INV-LAYER-001~005 (Engine internal layering for open-sourcing, 2026-05-16) = 총 166개.

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
- **Arch**: `arch/weight_swap.md` v8 (Phase 3 Manager 통합 + Phase 3.5/3.6/3.7 SOA + Phase 5 INV-122 v2.1 단일-token 측정 단위 고정 반영).

**교차 참조**:
- **INV-120** (Plan × Partition stale): `TransformerModel::ratio_generation`은 INV-120의 감지 키와 동일 소스이다. `SwapExecutor`가 batch 완료 후 정확히 1회 bump하여 plan invalidation을 일으킨다 (ENG-ALG-211 step (e)).
- **INV-121 ↔ INV-123**: 토큰 경계 기반 per-token snapshot(INV-121)과 `ArcSwap::store` 단일 원자 단계(INV-123)는 **쌍으로** forward 비차단성을 보장한다. 한쪽만 성립해도 안전 보장 불충분.
- **INV-122 ↔ INV-121**: dtype 혼합은 정상 상태이므로 INV-122의 수치 임계값은 "혼합 상태 자체"가 아닌 "혼합된 forward 결과의 근접성"에 적용된다.
- **INV-122 v2.1 측정 단위 고정** (Phase 5 Sprint A 진단 기반, 2026-04-26): 두 임계값 모두 **단일-token next-token logit**(prefill 종료 직후 첫 1개 logit)에 대해 측정한다. Decode loop(32-token greedy continuation 등)는 INV-122 게이트에 포함되지 않으며 보조 sanity metric으로만 사용된다. 단일-token 단위 고정은 Sprint A 100-prompt sweep에서 32-token decode 누적 drift로 ratio=0.25에서도 Δtop-1=44.85pp가 관측된 측정-임계값 미스매치를 해결한다. 책임 분리: 단일-token NMSE/Δ top-1 = swap/AUF/quantization 자체 정확성 게이트 / decode window = 양자화 누적 drift(swap 직접 책임 아님).
- **INV-122 v2 임계값 책임 분리** (Phase 4 실측 기반, 2026-04-25): NMSE는 절대값(≤ 0.01)으로 swap이 logit value scale을 손상시키지 않음을 검증. Top-1 ranking은 secondary dtype 단독 baseline 대비 Δ ≤ 1.0 pp로 swap 구현이 양자화 본질 노이즈 외 추가 회귀를 만들지 않음을 검증. 절대 top-1 값(예: ≥ 0.95)은 dtype/모델 본질이며 swap의 책임 범위 밖이다. 측정 방법론은 `arch/weight_swap.md` §5.1 참조.

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-121 | 32-engine-algorithms 3.12.7, ENG-ALG-214-SNAP | Forward 재진입 금지: 토큰 진입 시 per-layer `Arc<LayerWeights>` snapshot을 한 번 획득하고 토큰 내내 재사용. mid-token swap은 현재 토큰에 관찰 불가 (다음 토큰 경계부터 관측). stale/half-swapped 상태 관찰 0건. | Correctness | test | INV-120과 동일 메커니즘 (ratio_generation은 플랜 경로), per-token snapshot은 forward 경로 |
| INV-122 | 32-engine-algorithms 3.12.6 | Dynamic swap 결과의 정확성은 두 조건으로 강제된다: (1) primary baseline 대비 **단일-token logit NMSE ≤ 0.01** (절대값), (2) ratio=1.0 mixed swap 결과의 **단일-token mean top-1 match**가 **secondary dtype 단독 baseline 대비 Δ ≤ 1.0 pp**. **두 조건 모두 prefill 종료 직후 첫 next-token logit 1개**에 대해 측정한다. Decode loop(32-token greedy 등)는 게이트에 포함되지 않으며 보조 sanity로만 사용한다. 절대 top-1/top-5 임계값은 dtype/모델 본질 노이즈이므로 본 invariant의 책임이 아니다. **layer 간 dtype 불균일(혼합 상태)은 정상 상태이며 위반이 아니다.** v2.1 (2026-04-26, Phase 5 Sprint A 진단으로 측정 단위 명시). | Correctness | test | Llama/Qwen, ratio 0.25/0.5/0.75/1.0, 단일-token. ratio=1.0은 single-dtype baseline 비교 필수. 100+ prompt 다중 카테고리. |
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

### 3.15 Noshuffle SOA Registry Coherence [INV-130]

2026-04-25 Weight Swap Phase 3.6 (`OpenCLBackend::noshuffle_soa_registry` × `SwapExecutor` 통합)의 불변식. 대응 명세: `32-engine-algorithms.md` 3.12.15 (ENG-ALG-221), `arch/weight_swap.md` §2.2.3.

**교차 참조**:
- **ENG-ALG-211** (SwapExecutor batch 흐름): step (e) `ratio_generation` bump와 동일 시점에 SOA registry invalidate가 수행된다.
- **ENG-ALG-219 / INV-129** (Plan invalidation): SOA invalidate 후 다음 forward는 plan rebuild 경로를 거치며, rebuild 과정에서 새 cl_mem 주소로 SOA registry가 자연 재등록된다.
- **본 invariant는 디바이스(Adreno 830) 한정**으로 발현된다. 호스트 환경에서는 SOA registry 자체가 사용되지 않아 위반이 관측 불가하다.

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-130 | 32-engine-algorithms 3.12.15 (ENG-ALG-221) | Q4_0 weight swap으로 tensor.buffer(cl_mem)가 교체되는 경우, 교체된 layer의 `OpenCLBackend::noshuffle_soa_registry` entry는 stale 상태로 남아 있으면 안 된다. Swap 이후 해당 layer의 GPU matmul이 실행되기 전 invalidate(전체 clear 또는 per-layer key 제거)가 완료되어야 한다. 디바이스(Adreno 830) 한정으로 발현된다. | Correctness | runtime | HashMap entry removal. 호스트에서는 registry 비어 있어 NoOp. 디바이스 실측 필수. |

### 3.16 Weight Swap Phase 3.7 — SOA Re-conversion & AUF Format [INV-131 ~ INV-134]

2026-04-25 Weight Swap Phase 3.7. **3.7a (SOA 재변환 safety net)**과 **3.7b (AUF 포맷 도입)** 두 갈래로 분리된 작업이다.

- **Phase 3.7a (Adreno-only runtime safety net)**: INV-131. 대응 명세: `32-engine-algorithms.md` 3.12.16 (ENG-ALG-222).
- **Phase 3.7b (AUF v0.1 self-contained format)**: INV-132 ~ INV-134. 대응 명세: `33-engine-data.md` §3.22 (ENG-DAT-096), `32-engine-algorithms.md` 3.12.17 (ENG-ALG-223), `arch/auf_format.md`.

**교차 참조**:
- **INV-130** (Phase 3.6 SOA registry coherence): stale entry 제거. Phase 3.7a의 INV-131은 **새 entry 등록 의무**를 추가하며, INV-130과 함께 짝을 이룬다.
- **INV-129** (Plan invalidation): SOA 등록은 다음 plan rebuild 시점이 아니라 swap 종료 시점에 완료되어야 한다 — 등록 전 forward 진입을 막기 위함.

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-131 | 32-engine-algorithms 3.12.16 (ENG-ALG-222) | Q4_0 weight swap 후 첫 GPU matmul 직전, 해당 layer의 모든 Q4_0 weight tensor의 cl_mem 주소가 `OpenCLBackend::noshuffle_soa_registry`에 SOA descriptor와 함께 등록되어 있어야 한다. AUF cache hit 시 사전 변환된 descriptor를 등록, miss 시 `convert_aos_to_soa()` 런타임 호출로 등록. **디바이스(Adreno) 한정**. 호스트는 NoOp. | Correctness | runtime | swap_executor 내부 + OpenCL backend 등록 API 호출 추적. INV-130(stale 제거)의 dual로 작동. |
| INV-132 | 33-engine-data §3.22 (ENG-DAT-096), 32-engine-algorithms 3.12.17 (ENG-ALG-223 §3.12.17.6) | AUF reader는 (a) magic 불일치, (b) `format_major` reader 한도 초과, (c) `capability_required`의 미인식 비트 set 중 어느 하나라도 발견 시 panic 없이 reject한다. 에러 메시지에는 진단 가능한 정보(format 값, capability bit, 권장 조치)를 포함한다. Mode B 자립성 — `source_hash` 불일치는 reject 사유가 아니며 명시적 verify 명령에서만 비교한다. | Safety/Correctness | runtime, test | reader 진입 시 검증. fuzz/unit test 권장. |
| INV-133 | 33-engine-data §3.22.4 (ENG-DAT-096.4), 32-engine-algorithms 3.12.17.1 | AUF reader는 다음 section이 모두 존재할 것을 강제한다: `META`, `TOKENIZER`, `TENSOR_INDEX` (cross-cutting required) 그리고 reader의 backend에 해당하는 `WEIGHTS_*` (`WEIGHTS_ADRENO_SOA` / `WEIGHTS_CUDA_AOS` / `WEIGHTS_CPU_AOS` 중 하나). 누락 시 명시적 에러 + `auf-tool repack` 안내 메시지를 반환한다. | Correctness | runtime, test | reader 진입 시 검증. self-contained 보증의 핵심. |
| INV-134 | 33-engine-data §3.22.5 (ENG-DAT-096.5), §3.22.4 (ENG-DAT-C14) | 모든 AUF section은 `[header.payload_start_offset, file_size)` 범위 내에 있어야 한다 (`section.offset >= payload_start_offset` 그리고 `section.offset + section.size <= file_size`). 어떤 두 section도 byte range가 overlap 금지. 동일 tag의 section은 최대 1회 등장. | Safety/Correctness | runtime, test | reader 진입 시 검증. writer는 build 시점에 자동 충족. |

### 3.17 AUF lm_head Q4_0 사전 변환 [INV-135 ~ INV-136]

2026-04-26 Phase 6 Sprint G-1 (AUF v0.1.1, lm_head Q4_0 사전 변환)의 불변식. 대응 명세: `33-engine-data.md` §3.22.12 (ENG-DAT-096.12), §3.22.13 (ENG-DAT-096.13), `arch/auf_format.md` §2.5b.

**G-1-F update (INV-135 v2, 2026-04-26)**: 디바이스 측정에서 AUF v0.1.1의 lm_head SOA layout 사용이 OpenCL `CL_DEVICE_IMAGE_MAX_BUFFER_SIZE` 한계 초과로 silent corruption(garbage 출력)을 유발함이 확인되었다. Llama 3.2 1B의 lm_head q_buf(`vocab × hidden / 8` = 32M texels)는 Adreno OpenCL의 image1d_buffer_t 한계를 초과하여 `image` 생성이 실패하고, forward 측의 `m=1` SOA fast path가 standard GEMV로 fall through하면서 SOA의 `d_buf`만 노출된 cl_mem을 AOS layout으로 잘못 해석한다. 따라서 lm_head entry는 **모든 backend variant에서 AOS 18B/block layout으로 동봉**한다 (INV-135 v2).

**Sprint A' update (2026-04-27, v0.2)**: INV-135 v2의 의미가 분리된다. (1) **layout 의무**(Adreno SOA variant 안에서 AOS 강제)는 dtype-agnostic으로 유지되며, lm_head가 어떤 dtype(Q4_0/F16/...)이든 SOA 변환을 적용하지 않는다. (2) **dtype 단일성 의무**(Q4_0 single)는 폐기된다. lm_head도 layer weight와 동일하게 multi-dtype 후보 entry 그룹에 포함되어 dtype별 candidate가 다중 등장 가능하다. ENG-DAT-C16 갱신본과 INV-137 갱신본 참조.

**교차 참조**:
- **INV-132** (AUF reader rejection 의무): `capability_optional` bit 2(LM_HEAD_PRECOMPUTED_Q4_0)는 미인식 시에도 reject 사유가 아니다 (optional). v0.1.0 reader가 v0.1.1 AUF를 읽어도 bit 2를 무시하고 정상 진입. INV-135/136은 신 reader(v0.1.1) 한정 의무.
- **INV-133** (required section 존재 의무): lm_head Q4_0 payload는 별도 section이 아니라 기존 `WEIGHTS_<backend>` section 내부에 동봉되므로 INV-133의 6개 section 카탈로그는 그대로 유지된다.
- **INV-134** (section overlap 금지): lm_head payload는 동일 `WEIGHTS_<backend>` section 내부 layer weight payload와 byte range가 인접/연속이며 overlap 금지 의무는 그대로 적용된다.
- **ENG-DAT-C11** (Cross-layer tensor swap 제외): lm_head는 본 invariant 추가 후에도 swap 대상이 아니다 (ratio_generation 무관, model load 시점 1회 매핑).

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-135 v2 | 33-engine-data §3.22.12 (ENG-DAT-096.12), §3.22.6 (ENG-DAT-096.6) | AUF의 `capability_optional` bit 2(LM_HEAD_PRECOMPUTED_Q4_0)가 1이면 TENSOR_INDEX에 `kind = 11(lm_head)` entry가 적어도 1개 존재해야 하며, shape이 model load 시점의 모델 config(`vocab_size`, `hidden_dim`)와 일치해야 한다. **lm_head payload는 모든 backend variant에서 dtype-agnostic AOS 18B/block layout으로 동봉된다 (G-1-F fix + Sprint A' 일반화)** — `WEIGHTS_ADRENO_SOA` section 내부에서도 lm_head는 dtype에 무관하게(Q4_0이든 F16이든) SOA 변환을 적용하지 않고 AOS bytes 그대로 보존한다. 이유: lm_head q_buf 크기(`vocab × hidden / 8` texels 등 dtype별 환산)가 `CL_DEVICE_IMAGE_MAX_BUFFER_SIZE` 한계를 거의 모든 디바이스에서 초과하여 image1d_buffer_t 생성이 실패하고, 빠른 SOA path가 발동 불가능하므로 AOS layout이 의미 있다. shape mismatch 시 reader는 reject + 명시 에러 반환. AUF 헤더의 `source_hash`(hybrid)는 lm_head payload의 implicit identity 역할을 한다 — build 결정성(ENG-DAT-096.13) 보장 하에 source_hash 일치는 lm_head 일치를 함의한다. 별도 lm_head hash field는 도입하지 않는다. **v0.2 multi-dtype 도입 후 (Sprint A' 반전)**: lm_head는 더 이상 single-dtype 분기가 아니며 layer weight와 동일하게 candidate dtype별 entry로 다중 등장 가능하다 (예: Q4_0 entry + F16 entry). 본 INV는 dtype 단일성을 요구하지 않으며 **layout 강제만 dtype-agnostic으로 적용된다** (ENG-DAT-C16 갱신본). | Correctness | runtime, test | reader 진입 시 검증. shape mismatch는 다른 모델용 AUF가 잘못 사용된 케이스. v1→v2: 2026-04-26 Sprint G-1-F garbage 출력 수정 (silent corruption). v0.2 Sprint A' (2026-04-27): lm_head dtype 단일 의무 폐기, layout 의무만 잔존 (dtype-agnostic AOS 강제). |
| INV-136 | 33-engine-data §3.22.12 (ENG-DAT-096.12), `arch/auf_format.md` §2.5b | AUF의 `capability_optional` bit 2가 0이거나 AUF가 부재한 경우, model load는 기존 `quantize_lm_head_to_q4_0()` runtime fallback 경로로 정상 진행되어야 하며 lm_head dtype은 사용자 지정 `--quantize-lm-head` 값(`auto`/`none`/`q4_0`)에 따라 결정된다. `auto` + `--secondary-gguf`(또는 `--secondary-source`) 미설정 시 quantize skip(F16 유지). bit 0 시 AUF reader가 panic하거나 model load가 abort하면 안 된다. **`--secondary-gguf`는 W-AUF-1 (Sprint 1, 2026-05-19) 도입 후 deprecated alias로 stderr 경고 1회 출력 후 그대로 동작한다. `.gguf`/`.auf` 양쪽 입력은 계속 수용되며 lm_head fallback 의무는 동일하다. AUF self-secondary 자동 활성(W-AUF-2)이 정식 경로이며 그 단계에서 `--secondary-gguf`는 최종 제거 예정.** | Correctness | test | Sprint F 동작(v0.1.0 AUF + 신 코드) 보존. fallback 경로 회귀 방지. W-AUF-1 deprecation 경고 1회 출력은 동작 변경 없음. |

### 3.18 AUF v0.2 Multi-dtype Variant Invariants [INV-137 ~ INV-139]

2026-04-27 AUF v0.2 (multi-dtype variant)의 불변식. 대응 명세: `33-engine-data.md` §3.22.14 (ENG-DAT-097), §3.22.15 (ENG-DAT-098), §3.22.16 (ENG-DAT-099), `32-engine-algorithms.md` §3.12.18 (ENG-ALG-224, ENG-ALG-225), `arch/auf_format.md` §2.5c.

**도입 컨텍스트**: AUF v0.1.x는 backend variant 다중성만 지원했다. v0.2는 동일 (backend, layer, kind) 쌍에 대해 여러 dtype 후보(예: Q4_0 + F16)를 한 AUF에 동시 보관할 수 있게 확장한다. dynamic weight swap의 secondary dtype payload를 GGUF 의존 없이 self-contained로 보관하기 위함. 단방향 swap 가정은 그대로 유지되며 SwapExecutor 인터페이스는 변경되지 않는다 (Q3, ENG-DAT-C17).

**핵심 결정 (재확인, Sprint A' 반영)**:
- Q1=B: section tag에 dtype suffix 안 넣음. TENSOR_INDEX entry-level dtype 필드 활용. SECTION_TAG_SIZE=24 / SectionEntry 48B layout 보존.
- **Q2 (Sprint A' 반전)**: lm_head도 layer weight와 동일하게 multi-dtype 후보 entry 적용. dtype별 candidate entry(예: Q4_0 + F16) 다중 등장 가능. 단, INV-135 v2의 **layout 의무**(Adreno SOA variant 안에서 AOS 18B/block 강제)는 dtype에 무관하게 모든 lm_head entry에 적용된다. v0.1.1 시점의 "lm_head Q4_0 single dtype" 의미는 폐기됨.
- Q5=B: capability_optional bit 3 = `MULTI_DTYPE_VARIANTS` 신설. format_major=0 그대로, format_minor 1→2 bump.
- TensorIndex schema_version=1 보존 (v0.1.x reader 양방향 호환을 위해 schema bump 금지). entry 의미만 호환적으로 확장.

**교차 참조**:
- **INV-132** (AUF reader rejection 의무): `capability_optional` bit 3은 미인식 시에도 reject 사유가 아니다 (optional). v0.1.x reader가 v0.2 AUF를 읽어도 bit 3을 무시하고 first-match로 정상 진입.
- **INV-133** (required section 카탈로그): multi-dtype 도입해도 6개 section tag 카탈로그는 불변. dtype은 section tag가 아닌 TENSOR_INDEX entry로 분기.
- **INV-134** (section overlap 금지): dtype별 sub-payload는 동일 `WEIGHTS_<backend>` section 내부 인접 byte range로 배치되며 overlap 금지 의무는 그대로 적용된다.
- **INV-135 v2** (lm_head AOS layout 의무, **dtype-agnostic**): v0.1.1에서는 lm_head Q4_0 single dtype + AOS layout 의미였으나, v0.2 Sprint A' 이후 **layout 의무만 잔존**한다. lm_head는 모든 candidate dtype 후보에 대해 multi-dtype entry로 등록 가능하지만, `WEIGHTS_ADRENO_SOA` variant section 안에서는 dtype에 무관하게 AOS 18B/block layout으로 동봉되어야 한다 (image1d_buffer_t 한계). ENG-DAT-C16 갱신본 참조.
- **INV-122 v2.1** (단일-token 정확성 게이트): multi-dtype AUF로 swap된 경우에도 게이트 적용. dtype 변환 결정성(ENG-ALG-C12)이 보장되므로 게이트 통과 가능.

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-137 | 33-engine-data §3.22.14 (ENG-DAT-097), ENG-DAT-C15 | AUF의 `capability_optional` bit 3(MULTI_DTYPE_VARIANTS)가 1이면, 동일 (`layer_idx`, `kind`)에 등록된 모든 dtype 후보 entry는 동일 `shape_rank`와 동일 `shape` 값을 가져야 한다. dtype은 byte representation을 바꾸지만 logical shape은 보존하기 때문이다. **lm_head(`kind = 11`)도 multi-dtype 후보 그룹에 포함**되어 layer weight와 동일하게 dtype별 candidate entry로 다중 등장 가능하며, 동일 shape 일치 의무가 그대로 적용된다 (Sprint A' 반전). lm_head SOA variant 내 layout 강제(AOS)는 본 INV의 검사 대상이 아니라 ENG-DAT-C16 / INV-135 v2의 별도 의무이다. 위반 시 reader는 `AufError::ShapeMismatch`로 reject. | Correctness | runtime, test | reader 진입 시 검증 + writer build 시 자동 충족. multi-dtype 일관성. lm_head 포함. |
| INV-138 | 33-engine-data §3.22.15 (ENG-DAT-098), §3.22.16 (ENG-DAT-099), ENG-ALG-224 | AUF의 `capability_optional` bit 3 = 1이면, (a) META JSON에 `default_dtype` 필드가 반드시 존재해야 하며 그 값은 ENG-DAT-096.8의 dtype enum 중 하나여야 한다. (b) writer는 TENSOR_INDEX entries를 (`layer_idx` ASC, `kind` ASC, `is_default` DESC, `dtype` ASC)로 안정 정렬하여 default_dtype entry가 동일 (`layer_idx`, `kind`) 그룹의 가장 앞에 오도록 보장해야 한다. 이는 v0.1.x reader가 first-match 규칙으로 default_dtype을 자동 선택하도록 하는 호환 의무이다. (c) reader의 dtype selection precedence는 [호출자 명시 dtype → META.default_dtype → first-match] 순이다. 위반 시 reader는 `AufError::MalformedMeta { reason: "missing default_dtype with MULTI_DTYPE_VARIANTS" }` 또는 `AufError::DtypeNotAvailable`로 reject. | Correctness | runtime, test | writer 정렬 검증 + reader META 검증 + dispatch 정확성. v0.1.x reader 호환의 핵심. |
| INV-139 | 33-engine-data §3.22.14 (ENG-DAT-097), §3.22.16 (ENG-DAT-099) | `capability_optional` bit 3 = `MULTI_DTYPE_VARIANTS`의 의미: 1 = "이 AUF에는 어딘가에 multi-dtype TENSOR_INDEX entry가 적어도 1쌍 존재하며, META에 `default_dtype`이 정의되어 있다". 0 = "single-dtype 모드이며 동일 (`layer_idx`, `kind`)에 entry가 1번씩만 등장한다". v0.1.x reader는 본 bit를 인식하지 못하지만 `capability_optional`이므로 reject 사유가 아니다 (INV-132와 호환). v0.1.x reader가 v0.2 AUF를 만나면 first-match 규칙으로 default_dtype 단일 모드로 안전하게 동작한다 (INV-138 writer 정렬 의무 덕분). format_minor 1(v0.1.x) ↔ 2(v0.2)는 bit 3 사용 여부와 정합되어야 한다 — bit 3 = 1이면 format_minor ≥ 2 필수, format_minor=2이지만 bit 3 = 0인 경우는 format_minor가 향후 다른 v0.2 변경을 위해 미리 bump된 케이스로 허용. | Correctness, Compatibility | runtime, test | bit 3 의미 + v0.1.x ↔ v0.2 reader 호환 매트릭스. |

### 3.19 Weight Swap Overhead Reduction Invariants [INV-140 ~ INV-143]

2026-05-07 Weight Swap Phase 6.5 (Galaxy S25 1564.6 ms swap overhead 감축)의 불변식. 대응 명세: `32-engine-algorithms.md` §3.12.19~3.12.20 (ENG-ALG-226~231), `33-engine-data.md` §3.23 (ENG-DAT-100), `arch/weight_swap.md` §7.

**도입 컨텍스트**: `papers/eurosys2027/_workspace/experiment/swap_overhead_s25.md` 측정으로 stage breakdown — soa_reconvert 758ms / prefault 328ms / mmap_permute 305ms / primary release (코드의 `madvise_ms` 라벨, 실제 `clReleaseMemObject` chain) 173ms. 목표 stall < 800ms.

**교차 참조**:
- **INV-121** (per-token snapshot): forward 진입 시 토큰 경계 snapshot. ENG-ALG-228의 deferred release 워커가 `Arc::try_unwrap` 시도 시 forward가 토큰 경계에서 snapshot을 잡고 있으면 unwrap 실패 → 워커는 backoff 재시도. 다음 토큰 경계 통과 후 성공.
- **INV-123** (atomic store 단일성): ENG-ALG-228은 store 자체에 영향을 주지 않는다. 변경되는 것은 step (c) 의미 ("inline drop" → "enqueue to async worker")만이다.
- **INV-125** (secondary mmap 생존): ENG-ALG-227 borrow buffer는 secondary `Arc<SecondaryMmap>` clone을 보관하여 INV-125를 직접 강제한다 (INV-143).
- **INV-129/130/131** (Plan invalidation, SOA registry coherence): ENG-ALG-231 stage gate ordering이 ratio_generation bump 직전 queue를 idle 상태로 만들어 등록된 SOA descriptor가 모두 valid함을 보장한다. 위반 시 다음 forward가 반쯤 비어있는 registry를 본다.

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-140 | 32-engine-algorithms §3.12.19 (ENG-ALG-226), arch/weight_swap.md §7.2 | Fused SOA convert kernel(`cvt_q4_0_noshuffle_fused`)의 출력 cl_mem 내용은 기존 4-step path(GPU convert → CPU 2D transpose → write_buffer × 2)의 출력과 **byte-equal**이어야 한다. 동일 src/ne00/ne01에 대해 두 경로의 dst_q와 dst_d를 host로 read한 결과가 비트 단위 일치한다. fused kernel 가용성(`kernel_cvt_q4_0_noshuffle_fused.is_some()`)은 backend 컴파일 결과에 따라 빌드 시 결정되며, 실패 시 4-step fallback이 정확성을 동일하게 보장해야 한다. | Correctness | test | host build에서 두 경로 비교 (random Q4_0 buffer 다양 ne00/ne01). 디바이스에서 fused 경로 활성 검증은 manual + spec test. |
| INV-141 | 32-engine-algorithms §3.12.19 (ENG-ALG-228), 33-engine-data §3.23 (ENG-DAT-100) | `PrimaryReleaseWorker`는 swap batch N+1 시작 시점에 batch N의 모든 enqueued primary 해제 작업을 완료해야 한다. `execute_on_slots` 진입 시 `worker.pending_count() == 0` 검증을 수행하며, non-zero일 경우 짧은 deadline의 `worker.drain()`을 호출 후 재검증한다. drain 실패는 swap을 거부(`SwapError::ReleaseDrainTimeout`)하여 메모리 누수를 방지한다. forward 토큰 경계가 Arc holder를 양보할 때까지 워커는 backoff 재시도하므로, 정상 워크로드에서는 drain이 즉시 완료된다 (forward 토큰 ms 단위). | Correctness | test, runtime | host smoke (`Arc::try_unwrap` 성공 경로) + 토큰 경계 race 시나리오 stress test (forward stub과 동시 enqueue). |
| INV-142 | 32-engine-algorithms §3.12.20 (ENG-ALG-230, ENG-ALG-231), arch/weight_swap.md §7.8 | `execute_on_slots` 흐름에서 `TransformerModel.ratio_generation.fetch_add(1, SeqCst)`(stage e) 호출 직전에 `backend.synchronize()`가 1회 호출되어 OpenCL queue가 idle 상태여야 한다. 이는 (1) 비동기 `enqueue_write_buffer`(ENG-ALG-230), (2) fused convert kernel(ENG-ALG-226)이 모두 GPU에서 완료된 후에야 forward가 새 cl_mem을 읽을 수 있도록 보장한다. invalidate_noshuffle_soa_registry / ensure_noshuffle_soa_registered 호출도 synchronize 이후 단계에 위치해야 한다 (stage gate ordering). 위반 시 다음 forward가 미완성 cl_mem을 보고 garbage 산출 가능 (Adreno UMA에서도 ordering 보장 필요). | Safety/Correctness | runtime, test | swap_executor unit test에서 stage 호출 순서 검증 + 디바이스 e2e 정합성 게이트 (INV-122 v2.1 단일-token 게이트로 회귀 차단). |
| INV-143 | 32-engine-algorithms §3.12.19 (ENG-ALG-227), arch/weight_swap.md §7.3 | AOS 무변환 경로(`needs_qk_unpermute_at_swap()=false`)에서 사용되는 borrow buffer는 자신의 lifetime 동안 secondary `Arc<SecondaryMmap>`의 clone을 보관해야 한다. mmap 슬라이스 참조가 backend `copy_weight_from`/`copy_from` 호출 사이클을 통과하는 동안 secondary mmap이 drop되어 SIGBUS를 유발하지 않는다. 본 invariant는 INV-125(model lifetime mmap 생존)의 강화 표현으로 borrow 경로 한정 강제이다. permutation 경로는 owned `Vec<u8>`를 사용하므로 본 invariant의 범위 밖. | Safety | test | borrow buffer drop 시 mmap Arc strong_count 검증 (Tensor 생존 동안 secondary refcount ≥ 2). |

### 3.21 Intra-forward Layer-aligned Swap Invariants [INV-147 ~ INV-150]

2026-05-08 Intra-forward Layer-aligned Swap (LISWAP-4)의 불변식. 대응 명세: `32-engine-algorithms.md` §3.12.22 (ENG-ALG-235~238), `33-engine-data.md` §3.24 (ENG-DAT-101) + §3.15.16 CLI + ENG-DAT-C18, `arch/weight_swap.md` §10.

**도입 컨텍스트**: §3.19 Phase 6.5(290 ms 단발 stall 절대 감축)와 LISWAP-1(§3.20 예약, total wall-clock +96 ms)에 직교한 트랙. LISWAP-2 prototype 측정에서 forward 직후 일괄 dispatch가 Adreno multi-queue serialize로 0% saving. 본 시리즈는 forward **중간** layer 경계에서 dispatch하여 같은 forward 후속 layer + 다음 토큰 forward 선행 layer와 swap window(~25–28 ms)를 overlap하는 별도 timing 영역을 측정한다 (Adreno serialize의 chunk-크기/timing 의존성 검증).

**교차 참조**:
- **INV-121** (per-token snapshot): forward 진입 시 `layer_snapshots` 배열을 1회 build. layer i가 ArcSwap commit과 race하지 않음 (INV-147 보강).
- **INV-123** (ArcSwap 단일 원자 단계): hook이 dispatcher worker에 commit 위임. store는 worker thread에서 1회 발생, 의미 보존.
- **INV-129/130/131** (Plan invalidation, SOA registry coherence): plan 종료 시 dispatcher drain + synchronize 후 ratio_generation bump 1회로 의미 동일 적용 (INV-150).
- **INV-141** (다음 swap 전 drain 의무): plan 종료 시 `dispatcher.drain(deadline)`이 INV-141의 의미를 그대로 강제.
- **INV-142** (stage gate ordering): plan 종료 시점에서 INV-149로 강화 적용 (synchronize 1회 + ratio_generation bump 1회).

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-147 | 32-engine-algorithms §3.12.22.1 (ENG-ALG-235), arch/weight_swap.md §10.2 | `LayerBoundaryHook` 인자가 `None`일 때 `TransformerModel::forward_into`의 layer loop는 hook과 무관한 forward path와 동일한 hot-path 비용을 가져야 한다. 비용은 layer당 `Option::is_some` 검사 1회로 한정되며, branch predictor가 안정적으로 `None` 분기를 잡아 instruction-level overhead가 measurement noise 이하여야 한다. spec test는 (a) hook=None과 (b) hook=Some(NoOpHook)의 forward 시간을 비교하고, (a)는 baseline forward와 byte-equal 출력 + 시간 차이 < 1% 안에 들어와야 한다. NoOpHook overhead는 별도로 측정하여 <10% 이내여야 한다 (handoff §4.4 risk 항목). | Performance | test, microbench | host smoke (synthetic 1B model) + 디바이스 microbench (Galaxy S25, n≥10). 양쪽 모두 baseline forward_ms 대비 hook=None 차이 < 1%. |
| INV-148 | 32-engine-algorithms §3.12.22.2 (ENG-ALG-236) | 단일 `IntraForwardSwapPlan` instance 내에서 동일 layer index `idx`는 정확히 1회만 dispatch 된다. `should_dispatch(idx)`가 true를 반환한 직후 `mark_dispatched(idx)` 호출 후에는 `should_dispatch(idx) == false`가 영구히 유지된다 (plan retire 전까지). 동일 plan에서 같은 layer가 두 번 forward 통과해도(예: prefill + 첫 decode token이 같은 plan 진행 중에 발생) dispatch는 1회. 위반 시 해당 layer에 대한 cl_event가 `pending_events[idx]`를 두 번 덮어써 race condition 유발. | Correctness | test | unit test: plan 생성 → 같은 idx에 대해 should_dispatch / mark_dispatched / should_dispatch 호출 시퀀스 검증. integration test: 한 plan 안에서 forward를 여러 번 호출해도 dispatch 횟수 = `dispatch_at.len()`. |
| INV-149 | 32-engine-algorithms §3.12.22.4 (ENG-ALG-238), 33-engine-data §3.24 (ENG-DAT-101), arch/weight_swap.md §10.3 | Forward pass에서 layer K가 `LayerSlot::load_weights()`를 호출하기 직전에, `IntraForwardSwapHook::pending_event_for(K)`가 `Some(evt)`이면 `backend.wait_event_blocking(&evt)`이 반드시 호출되어 commit-before-read ordering을 강제한다. `arm_pending(K, evt)`는 `submit_commit` 직전에 store, `clear_pending(K)`는 dispatcher worker가 `slot.swap_weights` 후 store. 양쪽 모두 ArcSwap atomic. forward 스레드가 wait gate에서 잡은 evt가 dispatcher worker에 의해 그 사이 None으로 clear되어도 forward는 자신이 잡은 evt를 wait — completed event에 대한 wait는 fast no-op이므로 정확성 영향 없음. 위반 시 forward가 미완성 cl_mem 위에서 layer K weight를 읽어 garbage 출력 가능. | Safety/Correctness | runtime, test | host smoke: artificial cl_event를 hook의 `pending_events[K]`에 주입 후 forward 진입 시 wait_event_blocking 호출됨을 stub backend로 검증. 디바이스 e2e: INV-122 v2.1 단일-token 게이트로 회귀 차단. |
| INV-150 | 32-engine-algorithms §3.12.22.5 (ENG-ALG-238 후속), arch/weight_swap.md §10.4 | 활성 `IntraForwardSwapHook` plan은 `is_complete()` 가 true가 될 때까지 다음 plan commit을 막는다. plan이 complete되면 decode loop는 (1) `dispatcher.drain(deadline)`, (2) `backend.synchronize()`, (3) `ratio_generation.fetch_add(1, SeqCst)`, (4) `invalidate_noshuffle_soa_registry()`를 이 순서대로 호출하고 hook을 `None`으로 retire한다. ratio_generation bump는 plan당 정확히 1회. 진행 중 도착한 신규 `SwapWeights` 신호는 logged-and-dropped. | Safety/Correctness | runtime, test | drain → synchronize → bump → invalidate 호출 순서 trace. ratio_generation 카운터를 plan 시작 시점 vs retire 시점에 비교하여 정확히 +1. INV-141 동등 의무 검증 (drain 완료 후 hook drop 시 dispatcher pending_count == 0). |

### 3.22 QNN OpPackage cdylib Invariants [INV-151 ~ INV-155]

2026-05-09 QNN OpPackage M1 (production cdylib `crates/qnn_oppkg/`)의 불변식. 대응 명세: `30-engine.md` 부록 A (ENG-QNN-010 ~ ENG-QNN-C04).

**도입 컨텍스트**: Phase R 검증 완료 후 production migration 진입. PoC `crates/qnn_oppkg_poc/`는 회귀 안전망으로 보존. cdylib은 QNN runtime이 dlopen하는 외부 계약 산출물이며 Engine/Manager/Shared 어느 크레이트도 의존하지 않는다 (INV-001/010/011 보존).

**교차 참조**:
- **INV-001** (2 독립 프로세스): qnn_oppkg는 4번째 crate이지만 별도 cdylib 산출물(외부 라이브러리)로 Engine/Manager 프로세스 어디에도 포함되지 않는다. INV-001은 "프로세스 수=2"를 강제할 뿐 cdylib 산출물 수를 강제하지 않으므로 위반이 아니다.
- **INV-010/011** (Engine-Manager 직접 의존 금지, Shared 한정): INV-151이 동일 정신을 cdylib 방향으로 확장한다 (Engine ⊥ qnn_oppkg).

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-151 | 30-engine 부록 A.5 (ENG-QNN-C01) | `qnn_oppkg` crate은 `engine`(`llm_rs2`), `manager`(`llm_manager`), `shared`(`llm_shared`) 어느 크레이트와도 cargo dependency edge를 양방향으로 형성하지 않는다. workspace member로만 등록되며 build graph에서 isolated subgraph를 형성한다. | Safety | static | `crates/qnn_oppkg/Cargo.toml`의 `[dependencies]` + `engine/Cargo.toml`/`manager/Cargo.toml`/`shared/Cargo.toml` 코드 리뷰. CI에서 `cargo tree -p qnn_oppkg`가 engine/manager/shared를 포함하지 않음을 검증 가능. INV-001/010/011 골격 보존. |
| INV-152 | 30-engine 부록 A.3 (ENG-QNN-021) | `qnn_oppkg::registry::OPS.len() == StaticInfo::numOperations`. 정적 슬라이스 길이와 cdylib export 메타데이터 값이 비트 단위 일치한다. M1 범위에서 양쪽 모두 5. | Correctness | static, test | host unit test에서 `assert_eq!(OPS.len() as u32, static_info().num_operations)`. 등록 누락 회귀 즉시 검출. |
| INV-153 | 30-engine 부록 A.3 (ENG-QNN-022) | `OPS` 슬라이스 내 모든 `OpDescriptor.op_type` 문자열은 슬라이스 내에서 고유하다. 동일 op_type 두 번 등록 금지. | Correctness | static, test | host unit test에서 `HashSet<&str>::len() == OPS.len()` 검증. 중복 등록 시 `pkg_create_op_impl` 디스패처가 first-match 기준으로 실리지 않은 entry를 silent drop하여 회귀 위험. |
| INV-154 | 30-engine 부록 A.3 (ENG-QNN-023) | cdylib의 `pkg_get_info()` 반환 메타데이터에서 `backendApiVersion == (3, 7, 0)`. QNN GPU API 버전과 일치하며 mismatch 시 SDK가 cdylib을 reject한다 (Phase R G-1-F 결정적 fix). | Compatibility | static, test | host FFI surface test에서 `pkg_get_info()` 호출 후 backendApiVersion major/minor/patch 검증. SDK header와의 정합성. |
| INV-155 | 30-engine 부록 A.5 + A.9 (ENG-QNN-C04, ENG-QNN-C04') | **2-tier (v2, 2026-05-09 M2 진입 시점에 정밀화)**: (Primary, MUST) 100회 register/free 후 `qnn_oppkg::op_impl::STATE_MAP::lock().len() == 0` — cdylib이 보유한 모든 `Box<State>`는 `pkg_free_op_impl`에 의해 소진. (Secondary, SHOULD) `/proc/self/status::VmRSS` slope (last 50 iter linear regression) < 3 KB/iter — driver 잔여물(GPU compiled kernel cache, command buffer 풀)을 포함한 회귀 detector. M1.8 실측 1.1 KB/iter는 driver 영역으로 cdylib 책임 외이나 회귀 임계값으로 사용. | Safety | test | 디바이스 microbench (`microbench_qnn_oppkg_leak.rs`). primary는 STATE_MAP 검사, secondary는 last 50 iter linear regression slope. M1.8 게이트. |

### 3.23 QNN OpPackage M2 Layer Graph Invariants [INV-156 ~ INV-165]

2026-05-09 QNN OpPackage M2 (layer-level graph)의 불변식. 대응 명세: `30-engine.md` 부록 B (ENG-QNN-101 ~ ENG-QNN-C14).

**도입 컨텍스트**: M1 production cdylib (5 ops) 완료. M2는 5 op 추가 + layer graph builder + KV cache integration. **production engine code 변경 0 유지** (M3에서 backend trait 통합). 본 단계까지 INV-001/010/011 골격은 보존된다.

**교차 참조**:
- **INV-151~155** (M1): M2는 M1 invariant를 모두 보존. INV-151 (의존 격리), INV-152/153 (`OPS.len() == numOperations` + 고유성)은 M2의 OPS 슬라이스 길이가 10이 되어도 그대로 유효.
- **ENG-QNN-C03** (engine/kernels/*.cl 미수정): M2 INV-157이 동일 정신 확장 — 신규 .cl 파일 추가도 금지.
- **INV-160** (production code change == 0): M2 가장 핵심 invariant. M3 진입 전 외형적 격리 보증.

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-156 | 30-engine 부록 B.2 (ENG-QNN-101) | `qnn_oppkg::registry::OPS.len() == 10` (M1 5개 + M2 5개: `CustomMatMulQ40F32`, `CustomFlashAttn`, `CustomRope`, `CustomKvScatter`, `CustomDeqQ40`). M2 종료 게이트. | Correctness | static, test | host unit test에서 `OPS.len() == 10` + `static_info().num_operations == 10`. INV-152의 M2 specialization. |
| INV-157 | 30-engine 부록 B.2 (ENG-QNN-103, ENG-QNN-C10) | M2 신규 op은 모두 `engine/kernels/*.cl` 기존 파일을 `include_str!`로 임베드하며 신규 .cl 파일을 추가하지 않는다. production OpenCL 자산은 source 변경 0. | Safety | static | `git diff master..HEAD -- engine/kernels/`가 빈 출력. INV-001/010/011 (production isolation) 정신 확장. |
| INV-158 | 30-engine 부록 B.3 (ENG-QNN-111) | 단일 layer graph의 op node 수 ≤ 13 (Qwen2.5-1.5B 기준). graph builder 출력의 node count를 카운트하여 검증. | Correctness | test | host unit test에서 `build_layer_graph(...)` 후 graph descriptor의 node count 검사. fusion으로 ≤ 11도 가능. |
| INV-159 | 30-engine 부록 B.3 + B.8 (ENG-QNN-114, ENG-QNN-C12) | KV cache buffer는 `[1, kv_heads, 2048, head_dim]` F16 max-padded fixed shape. dynamic seq_len 미지원. attention mask는 max-padded이며 valid range 외 `-INFINITY`. | Correctness | runtime, test | layer graph 입력 tensor descriptor의 shape 검증. seq_len 변동 시 graph 재빌드 강제. |
| INV-160 | 30-engine 부록 B.5 (ENG-QNN-130) | M2 종료 시점 `engine/`, `manager/`, `shared/` 어느 크레이트의 소스 라인도 변경되지 않는다. test 추가 제외(단, test가 production 모듈을 import해도 production 빌드 산출물에 영향 없음). | Safety | static | CI에서 `git diff master..M2_HEAD -- engine/src/ manager/src/ shared/src/` 빈 출력. INV-151 (cargo edge isolation) 보강. |
| INV-161 | 30-engine 부록 B.6 (ENG-QNN-140) | Qwen2.5-1.5B layer 0을 OpPackage graph로 실행한 결과 출력 `y`가 CPU NEON reference 대비 `max_abs_err < 1e-2`(F16 tolerance). 동일 입력에 대해 1회 forward + 1회 KV write 후 비교. | Correctness | test | 디바이스 accuracy test. CPU 참조는 production engine의 NEON forward path. |
| INV-162 | 30-engine 부록 B.6 (ENG-QNN-141) | OpPackage 1 layer TBT ≤ production OpenCL baseline × 1.10. 동일 디바이스(Galaxy S25), 동일 ratio(GPU 100%), 동일 model. wall-clock only (CL_QUEUE_PROFILING_ENABLE 금지). | Performance | test | 디바이스 microbench. baseline은 `--backend opencl` decode TBT, OpPackage는 layer graph executor. |
| INV-163 | 30-engine 부록 B.6 (ENG-QNN-142) | `QnnGraph_finalize` 호출 시간 ≤ 200 ms (디바이스 측정, layer 1개 기준). | Performance | test | 디바이스 microbench. build → finalize wall-clock. |
| INV-164 | 30-engine 부록 B.4 (ENG-QNN-120) | SiluMul kernel은 production `.cl` 수정 없이 OpPackage abstraction의 intermediate alias 패턴(`ArgSpec::OutputTensorAliased`)으로 graph-safe하게 동작한다. SDK가 동일 backing buffer를 input/output edge에 alias 매핑할 수 있어야 한다. SDK가 거부 시 옵션 B (silu_oop + mul_oop 2단계 분해) fallback 발동. | Correctness | test | M2.4 게이트. layer graph 안에서 SiluMul 출력이 다음 op (ffn_up matmul 또는 final Add)의 입력으로 정상 연결됨 검증. |
| INV-165 | 30-engine 부록 B.2 (ENG-QNN-104) | Q4_0 weight tensor는 `MemoryObjectSpec::RawBytes { block_size: 18, element_count: N/32 }`로 OpPackage abstraction에 노출된다. 32 element block당 18 byte (2 byte F16 scale + 16 byte 4-bit nibbles). Q8_0/Q5_K 등 타 quantization은 별도 RawBytes variant. | Correctness | static, test | host unit test에서 `MemoryObjectSpec` 생성 시 block_size=18 검증. Q4_0 GEMV op `kernel_arg_setter`가 raw bytes를 cl_mem으로 정확히 매핑. |

### 3.24 QNN OpPackage M3 Backend Invariants [INV-166 ~ INV-180]

2026-05-10 QNN OpPackage M3 (production backend wire-up)의 불변식. 대응 명세: `30-engine.md` 부록 C (ENG-QNN-201 ~ ENG-QNN-240).

**도입 컨텍스트**: M2 layer graph cdylib (10 ops + 13~14-node graph) 정확성 GREEN 후 production engine `Backend` trait 구현체 `QnnOppkgBackend`를 추가하여 32-token greedy decode를 전수 통과시킨다. M2 INV-160 (production change == 0)은 본 단계에서 자연 만료하며, 그 정신을 INV-169 (OpenCL backend 무회귀)가 대체한다.

**교차 참조**:
- **INV-151~155 (M1)**, **INV-156~165 (M2)**: 본 단계에서 모두 보존. 단 INV-160 (production change == 0)은 자연 만료.
- **INV-001/010/011** (2 독립 프로세스 + Shared): cdylib binary는 여전히 dlopen 외부 산출물로 격리. host metadata crate (`crates/qnn_oppkg`의 rust source)에 대한 cargo edge는 INV-180으로 명시 허용.

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-166 | 30-engine 부록 C.2 (ENG-QNN-201) | qnn_oppkg backend는 `Backend` trait의 모든 필수 method (matmul, matmul_transposed, rms_norm, rms_norm_oop, rope_inplace, attention_gen, kv_scatter_f32_to_f16_batch, flash_attention_prefill)를 OpenCL과 동일 시그니처로 구현한다. | Correctness | static, test | 컴파일 타임 trait bound + host unit test에서 method 시그니처 동등성 검증. |
| INV-167 | 30-engine 부록 C.2 (ENG-QNN-203) | Layer graph는 model load 시점에 N(=28)회 graphFinalize 후 process lifetime 동안 재사용한다. cache invalidation은 weight swap path에서만 발동 (M4 영역). | Correctness | runtime, test | 디바이스 microbench에서 32-token decode 동안 graphFinalize 호출 == 28회 (load 시점) + 0회 (decode 동안). |
| INV-168 | 30-engine 부록 C.2 (ENG-QNN-204) | KV cache shape는 M2와 동일 `[1, kv_heads, 2048, head_dim] F16` max-padded. M3에서는 dynamic seq_len 미지원이며 prefill (variable seq_len)은 OpenCL backend로 fallback. | Correctness | runtime, test | layer graph 입력 tensor descriptor의 shape 검증. INV-159의 production 확장. |
| INV-169 | 30-engine 부록 C.5 (ENG-QNN-237, ENG-QNN-238) | OpenCL backend 정확성/TBT 무회귀: (a) `cargo test --workspace --features opencl` (qnn 비활성) 0건 회귀, (b) `--backend opencl` decode TBT가 M3.0 진입 직전 baseline 대비 ≤ 1.05× (5% tolerance). M2 INV-160 (production change == 0)을 약화 대체하는 핵심 게이트. | Safety | static, test | CI에서 (a)는 cargo test, (b)는 디바이스 microbench. Backend trait 신규 method 추가가 hot path overhead를 도입하지 않음을 보증. |
| INV-170 | 30-engine 부록 C.6 (ENG-QNN-C20, ENG-QNN-219) | `--backend qnn_oppkg | qnngpu`는 default off (opt-in). unknown backend 거부 (`bail!("Unknown backend")`)는 보존된다. `feature = "qnn"` cargo flag 미활성 시 dispatch에 등장하지 않는다. | Correctness | static, test | host integration test에서 `--backend foo` 실행 시 unknown backend로 reject 검증. feature gate 검증. |
| INV-171 | 30-engine 부록 C.2 (ENG-QNN-204) | KV cache는 rpcmem(DMA-BUF heap)-backed buffer로 alloc되며 mmap된 host pointer를 graph builder의 KvCacheHandle에 노출한다. host-side eviction/quant 정책 동시 가능. | Correctness | runtime, test | 디바이스 test에서 KVCache buffer가 rpcmem fd 기반 mmap 영역인지 검증 (`/proc/self/maps` 확인 또는 buffer 메타데이터). Phase R R-A2/R-Y 결과 기반. |
| INV-172 | 30-engine 부록 C.5 (ENG-QNN-231) | Qwen2.5-1.5B Q4_0 32-token greedy decode (top-1, seed=42, 동일 prompt) = `--backend opencl` 결과와 token sequence 100% 일치. 1개 token이라도 다르면 RED. | Correctness | test | 디바이스 accuracy gate. `microbench_qnn_oppkg_decode32` 또는 `generate` 바이너리 두 backend 결과 diff. |
| INV-173 | 30-engine 부록 C.5 (ENG-QNN-240) | TBT 측정은 wall-clock only. `CL_QUEUE_PROFILING_ENABLE` / `--profile-events` 사용 금지 (driver-specific 패널티 회피, M2 INV-162 정신 보존). | Performance | test | 디바이스 microbench. wall-clock 측정 + `--profile-events` 미사용 enforce. |
| INV-174 | 30-engine 부록 C.3 (ENG-QNN-212) | `Backend::supports_layer_graph()`는 idempotent. model load 후 항상 true 또는 항상 false (backend 내부 cache 상태에 의존하지 않음). | Correctness | runtime, test | host unit test에서 동일 backend 인스턴스에 대해 다중 호출 결과 동일성 검증. |
| INV-175 | 30-engine 부록 C.5 (ENG-QNN-239) | qnn_oppkg backend로 32 token decode 후 trait method (matmul, matmul_transposed, rope_inplace, attention_gen, kv_scatter_f32_to_f16_batch) 호출 instrumentation count == 0. fast path 정상 발동을 보증. | Correctness | test | debug build에서 trait method panic 또는 release build에서 atomic counter. count > 0은 RED. |
| INV-176 | 30-engine 부록 C.4 (ENG-QNN-221) | Layer graph node 수 == 14 (Qwen2.5-1.5B 기준, M2의 13에서 +1: RoPE OOP Q/K 분리 + Add residual 2개 명시). build-time const `LAYER_NODE_COUNT == 14`와 동기화 강제. | Correctness | static, test | host unit test에서 `build_layer_graph(...)` 결과 node count == LAYER_NODE_COUNT. const 변경 시 본 INV도 갱신. |
| INV-177 | 30-engine 부록 C.4 (ENG-QNN-225) | KV layout view transform은 buffer copy 비용 0. production HeadMajor `[head, pos, dim]` cl_mem stride와 graph 입력 `[1, kv_heads, 2048, head_dim]` stride가 동일 메모리 layout이며 reshape만으로 입력 전달이 가능하다. | Performance | test | 디바이스 microbench에서 layer graph 입력 transform 단계 wall-clock < 10 μs/layer. memcpy 발생 시 RED. |
| INV-178 | 30-engine 부록 C.5 (ENG-QNN-235) | 32-token decode 동안 `/proc/self/status::VmRSS` slope < 50 KB/token (token 단위 leak detector). INV-155 v2 secondary tier (3 KB/iter)와 다른 영역. | Safety | test | 디바이스 microbench에서 32 iter linear regression slope. |
| INV-179 | 30-engine 부록 C.5 (ENG-QNN-233) | TBT GREEN ≤ 1.10× / YELLOW 1.10~1.20× / RED > 1.20×. baseline = `--backend opencl` decode TBT, Galaxy S25, 동일 prompt 5회 평균, warm-up 3회 제외. | Performance | test | 디바이스 microbench. M3.4 메인 게이트. RED 시 사용자 호출. |
| INV-180 | 30-engine 부록 C.6 (ENG-QNN-C24) | engine 크레이트는 `crates/qnn_oppkg`의 host metadata (LayerConfig 등)에 cargo dependency edge를 형성한다. 단 cdylib binary 산출물은 여전히 dlopen 외부 라이브러리이며 engine binary가 link하지 않는다. INV-151 본래 정신(cdylib ⊥ engine binary)은 보존. | Compatibility | static | `cargo tree -p llm_rs2 -e features`에서 qnn_oppkg가 host crate dep으로 등장하지만 cdylib link는 형성되지 않음 (Cargo.toml `[dependencies]` 검토). |

### 3.25 QNN OpPackage M4 Async Swap Invariants [INV-181 ~ INV-188]

2026-05-10 QNN OpPackage M4 (phase-aware async chunk swap)의 불변식. 대응 명세: `30-engine.md` 부록 D (ENG-QNN-301 ~ ENG-QNN-320, placeholder).

**도입 컨텍스트**: M3 layer graph cache + Phase 6.5 weight swap 인프라(LayerSlot/SecondaryMmap/IntraForwardSwapHook/AsyncSwapDispatcher/HostPtrPool/AUF) 위에서 14-node DAG의 정적 phase analyzer + chunk swap dispatcher를 도입한다. cache-fit phase 진입 시 weight chunk를 `enqueue_write_async`로 dispatch하고 DDR-heavy phase 시작 직전 `wait_event_blocking`. **Pass-gate**: hide ratio ≥ 20% (chunk size 1점 PASS면 GREEN) + swap on/off token sequence 100% 일치.

**교차 참조**:
- **INV-140~143 (Phase 6.5)**, **INV-147~150 (LISWAP-4)**: chunk dispatcher seam이 동일 인프라(IntraForwardSwapHook/AsyncSwapDispatcher) 위에 build됨. INV-150 (plan run-to-completion) 정신 보존.
- **INV-167** (graph cache lifetime): chunk swap이 graph weight handle rebind를 요구. M4.1 spike 후 정책 결정 (옵션 A SDK API / 옵션 B 재build).
- **INV-176** (LAYER_NODE_COUNT == 14): INV-185가 동일 const 동기화를 chunk dispatcher 측에서도 강제.

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-181 | 30-engine 부록 D.3 (ENG-QNN-307) | Chunk swap path는 qnn_oppkg backend 한정. CPU/OpenCL은 NoOp fallback (chunk dispatcher 호출은 무시). | Correctness | runtime, test | host unit test에서 backend 별 chunk dispatcher.dispatch() 결과 검증. |
| INV-182 | 30-engine 부록 D.3 (ENG-QNN-302) | Chunk dispatch 시작은 cache-fit phase 진입 시점. DDR-heavy phase 진행 중에는 dispatch 금지 (메모리 압력 회피). | Correctness | runtime, test | phase analyzer state machine 검증. dispatch 시점이 cache-fit phase 노드의 begin marker와 일치. |
| INV-183 | 30-engine 부록 D.3 (ENG-QNN-302) | Chunk size sweep = {1, 2, 4, 8, 16} MB. M4.2 단계에서 sweep 측정 후 default 결정. | Performance | test | 디바이스 microbench `microbench_qnn_chunk_sweep`. 5 size × 5 iter 측정. |
| INV-184 | 30-engine 부록 D.3 (ENG-QNN-301) | Phase analyzer는 14-node static DAG 기반 const table — runtime 비용 0. dispatch 시점 결정에 list lookup 외 연산 없음. | Performance | static, test | host unit test에서 phase analyzer가 const table 외 alloc/loop 없이 dispatch hint 반환. |
| INV-185 | 30-engine 부록 D.3 (ENG-QNN-301) | qnn_oppkg::graph::LAYER_NODE_COUNT == 14와 phase analyzer의 enumerate set 동기화 build-time check. fusion 등으로 const 변경 시 phase analyzer도 함께 갱신해야 한다. | Correctness | static, test | host unit test에서 (DDR_HEAVY ∪ CACHE_FIT == FULL_SET) ∧ (DDR_HEAVY ∩ CACHE_FIT == ∅) ∧ (FULL_SET.len() == LAYER_NODE_COUNT). |
| INV-186 | 30-engine 부록 D.3 (ENG-QNN-304) | `wait_event_blocking`은 다음 token decode 시작 전에 호출 (forward 도중 main queue 비차단). chunk dispatch는 비동기 transfer queue를 사용하며 main compute queue를 막지 않는다. | Correctness | runtime, test | 디바이스 trace 검증. main queue wall-clock 동안 transfer queue 진행 검증. INV-149 (commit-before-read ordering) 정신 보존. |
| INV-187 | 30-engine 부록 D.3 (ENG-QNN-303) | Hide ratio = `1 - (overlapped time / forward time)` ≥ 20%. chunk size sweep 5점 중 1점에서라도 PASS면 GREEN. | Performance | test | 디바이스 microbench. M4.2 메인 게이트. 모든 size에서 < 20% 시 phase analyzer 분류 재검토 (M4.0 1회 retry). |
| INV-188 | 30-engine 부록 D.3 (placeholder, M4.1 본문에서 ENG-QNN ID 부여) | swap on/off 토큰 시퀀스 100% 일치. chunk swap 활성/비활성 동일 prompt+seed에서 32-token greedy decode 결과 동일. | Correctness | test | 디바이스 accuracy gate. INV-172 (M3 정확성 게이트)의 chunk swap 확장. |

### 3.26 Engine Internal Layered Architecture Invariants [INV-LAYER-001 ~ INV-LAYER-005]

2026-05-16 외부 공개(open-sourcing)를 위한 Engine 내부 레이어 구조 정규화. 대응 명세: `spec/01-architecture.md` §3.8 (SYS-100 ~ SYS-105). 코드 매핑/예외 처리: `arch/01-architecture.md` §6. 위반 현황/마이그레이션 계획: `ARCHITECTURE.md` §13. §13.8(2026-05-16 RESOLVED) 결정 사항(§A: AUF→shared/auf/, §B: backend-aware pool + WeightStagingPool trait, §C: chat_template→inference/+모델별, chat_ipc→session/, §D: backend-specific buffer→backend/<be>/buffer/, §E: 테스트 점진적)이 본 INV 시리즈 비고에 반영되어 있다.

**도입 컨텍스트**: Engine 크레이트(`llm_rs2`) 내부 모듈 의존 그래프가 단방향 5-layer + 2-cross-cutting 구조를 따라야 한다. 본 INV 시리즈는 시스템 전체 구조(INV-001/010/011 — 2 프로세스 + Shared edge)와는 직교한 layer로, Engine 내부의 모듈 import 그래프만 통제한다.

**ID 컨벤션**: 본 시리즈는 `INV-LAYER-NNN` 별칭 형식을 사용한다. 다른 INV-NNN과 ID space 충돌이 없으며, 의미상 별도 차원(Engine 내부 layering)을 다룬다. INV-LAYER-001~005는 모듈 import 그래프(컴파일 단위 경계)를, INV-LAYER-006~007은 L4 내부 struct/builder의 추상화 결합도(field 타입 수준)를 통제한다.

**교차 참조**:
- **INV-001 / INV-010 / INV-011** (2-프로세스 + Shared 경계): 시스템 전체 구조 — 본 시리즈는 이를 보존하며 Engine 내부 추가 제약을 명시.
- **INV-012** (Backend trait이 유일한 하드웨어 추상화점): 본 시리즈의 INV-LAYER-001/003이 이를 강화 — backend impl을 우회한 NEON/OpenCL 직접 호출은 INV-012와 INV-LAYER-003 양쪽 위반.
- **INV-067** (cuda/opencl feature 상호 배타): 본 시리즈와 직교 — feature gate는 컴파일 타임, layer 규칙은 모듈 의존 그래프.
- **INV-151 / INV-180** (qnn_oppkg cdylib isolation): cargo workspace 단의 cdylib 격리. 본 시리즈와 직교.

**위반 측정 도구**: `grep -rn "use crate::" engine/src/ | python3 scripts/layer_lint.py` (TODO: 도구 신설). HEAD `d8f26156` 기준 실측 위반 31건(V-01 ~ V-31)은 `ARCHITECTURE.md` §13.5 표 참조.

**테스트 위치**: 본 INV 시리즈는 `engine/tests/spec/test_inv_layer_{001..005}.rs`에 각각 1개씩 spec test 파일을 가져야 한다 (feedback: `spec_tests_required` — inline `#[cfg(test)]` 불충분). 각 테스트는 `cargo metadata --format-version 1`로 build graph를 읽거나 `grep "^use crate::"` 결과를 layer 분류 표와 매칭하여 위반 enum을 반환한다. **베이스라인 정책**: 현 시점 위반 31건은 baseline JSON(`engine/tests/spec/inv_layer_baseline.json`)에 기록하고, 테스트는 "baseline 이하의 위반"을 PASS로 처리한다. 마이그레이션 PR마다 baseline을 줄이며, 마지막에는 baseline=0이 되어야 한다. spec coverage 도구 `scripts/check_spec_coverage.sh`에 본 시리즈가 포함되도록 INV-LAYER prefix를 인식하게 확장 필요.

| ID | 원본 | 한줄 요약 | 카테고리 | 검증 | 비고 |
|----|------|----------|---------|------|------|
| INV-LAYER-001 | 01-architecture SYS-100, SYS-103 | Engine L1 backend impl(`backend/<be>/`)은 L2(`shared/`)와 cross-cutting(`observability/`, `resilience/`) 외 import 금지. 동일 layer 내 cross-backend import는 명시 zone(예: `cpu_fallback()` 패턴)에 한해 허용. backend가 backend-aware staging pool(예: `backend/cuda_embedded/pool.rs`의 `layer_object_pool`, `backend/opencl/host_ptr_pool.rs`)을 소유하고, pressure handler는 `WeightStagingPool` trait(L2)을 통해 접근한다 (§13.8-B 결정). 위반 예: `backend/opencl/mod.rs` → `layers::tensor_partition`, `backend/qnn_oppkg/mod.rs` → `models::weights::LayerSlot`. | Safety | static, test | grep "use crate::" 결과를 layer 분류와 매칭. INV-012의 backend 추상화 우회 방지를 backend 측에서도 강화. **테스트 예외 (§13.8-E)**: lib 내부 inline `#[cfg(test)]` 블록의 backend instantiation은 grandfathered exception으로 baseline에 등재된 채 유지. 신규 테스트는 `tests/spec/` 한정. |
| INV-LAYER-002 | 01-architecture SYS-100, SYS-103 | Engine L2 `shared/`는 L3(`pressure/`, `inference/`), L4(`session/`), L5(`bin/`) 어떤 모듈도 import 금지. backend-specific buffer/memory는 `shared/`가 아닌 `backend/<be>/buffer/`에 위치한다 — `cl_*`(opencl), `cuda_*`(cuda_embedded/cuda_pc), `rpcmem_*`(qnn_oppkg) 모두 backend 폴더 산하 (§13.8-D). 단 generic buffer(`shared_buffer`, `slice_buffer`, `mmap_buffer`, `unified_buffer`, `borrowed_mmap_buffer`)와 AUF(`shared/auf/`, §13.8-A)는 `shared/` 산하. 위반 예: `buffer/cuda_mmap_alias_buffer.rs` → `models::weights::SecondaryMmap`. | Safety | static, test | grep 검증. `SecondaryMmap`은 L3 Pressure state로 분류되므로 buffer가 직접 import 시 위반. trait 경유로 inversion 필요. **테스트 예외 (§13.8-E)**: lib 내부 inline `#[cfg(test)]`는 grandfathered. |
| INV-LAYER-003 | 01-architecture SYS-100, SYS-103 | Engine L3 `inference/`와 L3 `pressure/`는 상대 도메인의 **trait만** import할 수 있고 concrete 구현체 import 금지. 위반 예: `core/cache_manager.rs` → `resilience::EvictMethod` (cross-cutting concrete), `models/transformer.rs` → `core::offload::preload_pool` (inference→pressure concrete). 동일 도메인 내 모듈 cross-import는 자유 — 예: `inference/chat_template.rs`(generic)와 `inference/models/<arch>/chat_template.rs`(모델별 구현체) 사이 import 자유 (§13.8-C). pressure handler가 backend pool에 접근할 때는 `WeightStagingPool` trait(L2 정의)을 경유. | Correctness | static, test | trait 노출은 `pressure/mod.rs`, `inference/mod.rs`에 명시적 re-export로 강제. concrete 구조체는 module-private 또는 동일 도메인 내에서만 pub. |
| INV-LAYER-004 | 01-architecture SYS-102, SYS-104 | Cross-cutting 모듈(`observability/`, `resilience/`)이 L3 도메인의 concrete type을 import할 때는 trait/Sink 경유로 inversion한다. `EventSink`, `Transport`, `GpuEventMeter` 등이 표준 inversion 패턴. 예외(허용): `events::CacheEvent` enum이 pressure 결과를 직접 표현하는 경우 — enum이 inversion 매체이므로 동의어. 위반 예: `eval/eviction_hook.rs` → downcast `OpenCLBackend` (L1 concrete를 cross-cutting이 직접 소비). | Correctness | static, test | observability/resilience 모듈의 import 분석. concrete L1/L3 type을 import한 곳마다 trait inversion 가능성 검토. eval은 L4(`session/eval/`)로 격상 시 자연 해소. |
| INV-LAYER-005 | 01-architecture SYS-105 | Engine L5 production binary(`bin/generate.rs`)는 L4 `session/`만 직접 import한다. test/microbench binary(`test_backend`, `signal_injector`, `microbench_*`)는 본 규칙 밖. `chat_ipc`는 L4 책임이므로 `session/chat_ipc.rs`에 위치하며 production binary가 직접 import하지 않는다 (§13.8-C). 위반: 현 `bin/generate.rs` 13,022 LOC monolith가 `core`, `models`, `layers`, `memory`, `experiment`, `resilience` 등 거의 모든 lib 모듈을 직접 import (29건의 `use llm_rs2::*`). | Correctness | static, test | Migration Step 2(L5/L4 분리) 후 강제. 그 전까지는 best-effort. test/microbench는 enforcement 대상 외. |
| INV-LAYER-006 | arch/inference_pipeline.md §3, §8.4, §10 | L4 `session::DecodeLoop`은 concrete backend/manager/profiler를 자기 필드로 직접 참조 금지. 구체적으로 L1 backend impl(`OpenCLBackend`, `CudaBackend`, `CpuBackend` 등) 및 L3 concrete struct(`CacheManager`, `LlamaModel`/`TransformerModel`, `Profiler`, `ManagerClient`, `KiviCache`, `OffloadStore` 등)를 `DecodeLoop` struct 필드로 보유 금지. 6 추상화(`session::Forward`, `session::EvictionStage`, `session::SwapStage`, `session::CommandSource`, `session::TokenSampler`, `session::DecodeObserver`)의 `Box<dyn>` 또는 generic bound만 허용. 위반 예: `DecodeLoop { backend: Arc<OpenCLBackend>, manager: ManagerClient, ... }`. **적용 경계**: 본 INV는 `DecodeLoop` struct 필드에만 적용된다. 6 trait의 구현체 struct(`ModelForward`, `CacheManagerStage` 등) **내부 필드**는 L1/L3 concrete를 owned/borrow로 자유 보유 가능 — builder가 trait object로 추상화 후 주입하는 자연 경로 (Task #4 finalize 2026-05-16). 본 INV는 SYS-100/103과 직교한 *결합도* 제약 — INV-LAYER-005(L5→L4 import 제한)는 모듈 경계, 본 INV는 L4 진입점 struct field의 추상화 보유. | Correctness | static, test | `engine/tests/spec/test_inv_layer_006.rs` — `DecodeLoop` struct 필드를 reflection 없이 source-grep으로 검사. concrete L1/L3 type 이름이 `DecodeLoop` 필드 타입에 등장하면 FAIL (trait impl struct 내부 필드는 대상 외). 빌더는 trait object 또는 generic만 받는다(`Box<dyn Forward>`, `<F: Forward>`). Migration Step 2-4 후 강제. |
| INV-LAYER-007 | arch/inference_pipeline.md §4, §11 | `session::DecodeLoopBuilder`의 필수 컴포넌트(`session::Forward`)는 typestate 패턴으로 컴파일 타임에 강제된다. `.build()` 메서드는 `DecodeLoopBuilder<HasForward, ...>`에서만 호출 가능. Optional 컴포넌트(`session::EvictionStage`, `session::SwapStage`, `session::CommandSource`, `session::DecodeObserver`)는 기본 no-op 구현이 자동 적용. `session::TokenSampler`도 default(`GreedySampler`) 제공. `Forward` trait 자체의 lifecycle hook(`finalize`, `on_kv_prune`)은 trait 정의에서 default no-op 본문을 제공 — 외부 기여자가 `prefill`/`step`만 구현해도 컴파일 성공 (Task #4 finalize 2026-05-16 사용자 결정 #2). 위반 예: `Forward` 누락 상태에서 `build()` 호출이 컴파일된다면 FAIL. | Correctness | static, test | `engine/tests/spec/test_inv_layer_007.rs` — `trybuild` crate로 negative test: (a) `Forward` 없이 `build()` 호출하는 코드가 compile-fail (`expected: HasForward, found: NoForward`), (b) `Forward` trait의 `prefill`+`step`만 구현한 minimal impl이 compile-pass임을 확인 (lifecycle hook default 검증). Migration Step 2-2 후 강제. |

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
| Safety | 27 | 23% |
| Correctness | 80 | 70% |
| Performance | 5 | 4% |
| Compatibility | 4 | 3% |
| **합계** | **115** | **100%** |

> **참고**: 카테고리 통계는 INV-001~188 + INV-LAYER-001~007을 합산하나 일부 항목(INV-129/132/142/149/150/162 등)이 2-카테고리 보유이므로 합계는 누적 1회 카운트 기준이다. INV-LAYER-001/002는 Safety, INV-LAYER-003/004/005/006/007은 Correctness로 1회 카운트.
> INV-113, INV-114는 v2.1.0에서 REMOVED (pessimistic safe set 제거). 카운트에서 제외.
> INV-117~119는 v2.2.0 (QCF × DPP)에서 추가. INV-121~122는 Weight Swap Phase A에서 추가.
> INV-129는 Weight Swap Phase 3.5 (Plan × Weight Swap stale detection)에서 추가. 카테고리는 Safety/Correctness 양쪽이며, 통계는 Safety로 1회 카운트한다.
> INV-130은 Weight Swap Phase 3.6 (Noshuffle SOA registry coherence)에서 추가. 카테고리는 Correctness (디바이스 한정 silent correctness bug — crash/data-loss가 아니므로 Safety 아님).
> INV-131은 Weight Swap Phase 3.7a (Adreno SOA 재변환 safety net)에서 추가. Correctness.
> INV-132~134는 Weight Swap Phase 3.7b (AUF v0.1 포맷)에서 추가. INV-132는 Safety/Correctness 양쪽이며 Safety로 1회 카운트, INV-133/134는 Correctness.
> INV-135~136은 Phase 6 Sprint G-1 (AUF v0.1.1 lm_head Q4_0 사전 변환)에서 추가. 둘 다 Correctness — INV-135는 shape/identity 일치 검증, INV-136은 후방 호환 fallback 보존.
> INV-137~139는 AUF v0.2 multi-dtype variant (2026-04-27)에서 추가. INV-137/138은 Correctness, INV-139는 Correctness/Compatibility 양쪽이며 Compatibility로 1회 카운트한다 (v0.1.x ↔ v0.2 reader 호환 의무).
> INV-140~143은 Weight Swap Phase 6.5 Overhead Reduction (2026-05-07)에서 추가. INV-140/141은 Correctness, INV-142는 Safety/Correctness 양쪽이며 Safety로 1회 카운트, INV-143은 Safety (borrow buffer mmap lifetime).
> INV-147~150은 Intra-forward Layer-aligned Swap (LISWAP-4, 2026-05-08)에서 추가. INV-147은 Performance (hook=None zero overhead), INV-148은 Correctness (plan dispatch 멱등), INV-149/150은 Safety/Correctness 양쪽이며 Safety로 1회 카운트(commit-before-read ordering, plan run-to-completion).
> INV-151~155는 QNN OpPackage M1 cdylib (2026-05-09)에서 추가. INV-151은 Safety (workspace dependency isolation, INV-001/010/011 확장), INV-152/153은 Correctness (정적 일관성), INV-154는 Compatibility (SDK 계약 버전 고정), INV-155는 Safety (leak 부재 보증). **INV-155 v2 (2026-05-09 M2 진입 시점)**: M1.8 실측 cdylib leak == 0 (STATE_MAP) 확인 후 driver 잔여물(1.1 KB/iter)을 별도 tier로 분리. Primary(STATE_MAP=0, MUST) + Secondary(VmRSS slope < 3 KB/iter, SHOULD) 2-tier 구조. ID/카운트 변동 없음.
> INV-156~165는 QNN OpPackage M2 layer-level graph (2026-05-09)에서 추가. INV-156/158/159/161/164/165는 Correctness, INV-157/160은 Safety (production isolation 보강), INV-162/163은 Performance.
> INV-166~180은 QNN OpPackage M3 backend wire-up (2026-05-10)에서 추가. INV-166/167/168/170/172/174/175/176은 Correctness, INV-169/178은 Safety (OpenCL backend 무회귀 + leak detector), INV-171은 Correctness (rpcmem-backed KV), INV-173/177/179는 Performance (TBT 측정 룰 + view transform 비용 0 + TBT verdict band), INV-180은 Compatibility (cdylib ⊥ engine binary 본래 정신 보존). M2 INV-160 (production change == 0)은 본 단계에서 자연 만료. 카운트는 +15.
> INV-181~188은 QNN OpPackage M4 async chunk swap (2026-05-10, placeholder)에서 추가. INV-181/182/185/186/188은 Correctness, INV-183/184/187은 Performance. M4.0~M4.3 단계 진입 시 본문 채움. 카운트는 +8.
> INV-LAYER-001~005는 Engine Internal Layered Architecture (2026-05-16)에서 추가. INV-LAYER-001/002는 Safety (backend/shared import 그래프 보존), INV-LAYER-003/004/005는 Correctness (도메인 경계, cross-cutting trait inversion, L5/L4 분리). 별칭 ID 시리즈로 INV-NNN과 별도 namespace를 형성. 카운트는 +5.
> INV-LAYER-006~007은 Task #4 (`DecodeLoop` SOLID 분해 + 빌더, 2026-05-16, `arch/inference_pipeline.md`)에서 추가. 둘 다 Correctness — INV-LAYER-006은 L4 struct 필드 타입의 추상화 결합도(DIP 강화), INV-LAYER-007은 builder typestate로 필수 컴포넌트 컴파일 타임 강제. 카운트는 +2 (LAYER 시리즈 누적 7).
> INV-122는 v2(2026-04-25, Phase 4 정확성 측정 기반)로 임계값 재정의. 이전 절대값(top-5 ≥ 0.9, top-1 ≥ 0.95)이 Q4_0 + 1B 환경에서 물리적으로 도달 불가함이 확인되어, NMSE ≤ 0.01 (절대) + Δ Top-1 ≤ 1 pp (vs single-dtype baseline)로 변경. ID/카운트는 변동 없음.
> **INV-122 v2.1 (2026-04-26, Phase 5 Sprint A 진단 기반)**: 측정 단위를 **단일-token next-token logit**(prefill 종료 직후 첫 1개 logit)으로 명시적으로 고정. Sprint A 100-prompt × 4-ratio sweep에서 32-token decode 누적 drift로 ratio=0.25에서도 Δtop-1=44.85pp 관측 — 측정-임계값 미스매치 진단. 정확성 회귀(garbage 출력 등)는 0건. 임계값 자체(NMSE ≤ 0.01, Δ top-1 ≤ 1 pp)는 v2와 동일하나 측정 단위가 단일-token으로 고정. Decode window metric은 보조 sanity로 분리. ID/카운트는 변동 없음. Phase 4 자료(NMSE mean=0.0062, Δ top-1=+0.33pp)는 v2.1 기준으로도 PASS.

### 5.2 검증 방법별 통계

| 검증 방법 | 주 검증 | 보조 검증 포함 | 설명 |
|----------|---------|-------------|------|
| static | 25 | 29 | Cargo 의존 구조, feature gate, trait bound, 코드 구조 (INV-LAYER 7건 모두 static 주검증) |
| runtime | 34 | 41 | assert, clamp, 조건 검사, AtomicU64 |
| test | 21 | 44 | 단위/통합/프로퍼티/장애 주입 테스트 (INV-LAYER 7건 모두 test 보조 검증) |

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

Safety 카테고리 (19개) -- 위반 시 크래시/데이터 손실/하드웨어 손상 가능:

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
| INV-132 | AUF reader fail-fast on magic/format/capability mismatch |

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
