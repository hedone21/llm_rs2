# Invariants Catalog

> **TL;DR**: llm_rs2 전체 스펙에 산재된 불변식(INV-*)을 한 곳에 수집하고,
> 카테고리(Safety/Correctness/Performance/Compatibility)와
> 검증 방법(static/runtime/test)으로 분류한다.
> INV-001~076 (기존 59개) + INV-066~068 (CUDA 3개) + INV-080~085 (cross-cutting 6개) = 총 68개.

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

## 5. Constraints

### 5.1 카테고리별 통계

| 카테고리 | 개수 | 비율 |
|---------|------|------|
| Safety | 16 | 25% |
| Correctness | 44 | 68% |
| Performance | 2 | 3% |
| Compatibility | 3 | 5% |
| **합계** | **65** | **100%** |

### 5.2 검증 방법별 통계

| 검증 방법 | 주 검증 | 보조 검증 포함 | 설명 |
|----------|---------|-------------|------|
| static | 20 | 24 | Cargo 의존 구조, feature gate, trait bound, 코드 구조 |
| runtime | 32 | 38 | assert, clamp, 조건 검사, AtomicU64 |
| test | 12 | 26 | 단위/통합/프로퍼티/장애 주입 테스트 |

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

Safety 카테고리 (16개) -- 위반 시 크래시/데이터 손실/하드웨어 손상 가능:

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
