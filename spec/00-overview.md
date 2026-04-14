# System Overview

> **TL;DR**: llm_rs2는 ARM64 모바일/엣지 디바이스에서 LLM 추론을 수행하는 온디바이스 추론 시스템이다. Engine(추론)과 Manager(리소스 모니터링) 두 개의 독립 프로세스로 구성되며, IPC를 통해 양방향 통신한다. 3개 도메인(compute, memory, thermal)의 리소스 압박을 도메인별 전략(PI 제어 + 임계값 직접 매핑)으로 연속 추정하고, cross-domain 액션 조합 선택과 온라인 relief 학습을 통해 최소 품질 손실로 추론을 지속한다. QCF(Quality Cost Function) 기반으로 손실성 액션의 품질 비용을 추적한다. Rust로 구현되고, Cargo workspace 3-crate 구조(engine, manager, shared)를 따른다.

## 1. Purpose and Scope

이 문서는 llm_rs2 시스템 스펙의 진입점이다. 시스템이 무엇이고, 누구를 위한 것이며, 무엇을 보장하는지 정의한다.

**이 파일이 명세하는 것:**

- 시스템의 운영 환경과 외부 인터페이스 (System Context)
- 시스템의 목표와 비목표 (System Goals)
- 지원 플랫폼과 하드웨어 요구사항 (Target Platform)
- 지원 모델 아키텍처와 포맷 (Supported Models)
- QCF 기반 품질 추적 프레임워크 (Quality Assurance)
- 시스템 수준 안전성 요구사항 (Fail-Safety)
- 구현 언어, 의존성, 빌드 설정 (Implementation)
- 전체 스펙에서 공통 사용하는 용어집 (Definitions)

**이 파일이 명세하지 않는 것:**

- 컴포넌트 내부 구조 → `01-architecture.md`
- 프로토콜 와이어 포맷 → `10-protocol.md`
- 알고리즘 상세 → `22-manager-algorithms.md`, `32-engine-algorithms.md`

## 2. Definitions

이 섹션은 전체 스펙에서 공통으로 사용하는 용어를 정의한다. 개별 스펙 파일은 추가 용어를 자체 Definitions 섹션에서 정의할 수 있다.

### 시스템 용어

| 용어 | 정의 |
|------|------|
| **Engine** | LLM 추론을 수행하는 프로세스. 모델 로딩, forward pass, KV 캐시, 백엔드 연산, Resilience 서브시스템을 포함한다. |
| **Manager** | 시스템 리소스를 모니터링하고 정책을 평가하여 Engine에 디렉티브를 전송하는 독립 서비스 프로세스. |
| **Shared** | Engine과 Manager가 공유하는 IPC 타입을 정의하는 라이브러리. |
| **Workspace** | Engine, Manager, Shared 세 크레이트를 포함하는 Cargo workspace. |

### 추론 용어

| 용어 | 정의 |
|------|------|
| **Prefill** | 입력 프롬프트의 모든 토큰을 한 번에 처리하는 초기 단계. KV 캐시를 채우고 첫 출력 토큰을 생성한다. |
| **Decode** | Prefill 이후 토큰을 하나씩 자기회귀적으로 생성하는 반복 단계. |
| **Token** | 텍스트의 최소 처리 단위. Tokenizer가 텍스트를 토큰 시퀀스로 변환한다. |
| **Forward Pass** | 모델의 모든 레이어를 한 번 순방향으로 실행하여 다음 토큰의 확률 분포를 계산하는 과정. |
| **KV Cache** | Transformer 어텐션의 Key/Value 텐서를 저장하는 캐시. 이전 토큰의 K/V를 재사용하여 Decode 시 중복 계산을 제거한다. |
| **RoPE** | Rotary Position Embedding. 토큰의 절대 위치를 인코딩하는 회전 기반 위치 임베딩. `start_pos` 파라미터로 논리적 토큰 위치를 지정한다. |
| **Attention Sink** | 시퀀스 초기 토큰(일반적으로 첫 4개)이 비정상적으로 높은 어텐션 스코어를 받는 현상. StreamingLLM이 이를 활용한다. |

### 리소스 용어

| 용어 | 정의 |
|------|------|
| **Level** | 시스템 리소스 상태의 4단계 심각도: `Normal`, `Warning`, `Critical`, `Emergency`. 순서 관계가 있다 (Normal < Warning < Critical < Emergency). |
| **ResourceLevel** | 프로토콜에서 사용하는 3단계 리소스 수준: `Normal`, `Warning`, `Critical`. |
| **PressureVector** | PI Controller가 출력하는 3차원 압력 벡터 (compute, memory, thermal). 각 차원은 [0.0, 1.0] 범위의 연속 값이다. |
| **OperatingMode (Engine)** | Engine Resilience 서브시스템의 운영 모드. 최악 신호 수준에 의해 결정된다: `Normal`, `Degraded`, `Minimal`, `Suspended`. |
| **OperatingMode (Manager)** | Manager Supervisory 계층이 출력하는 운영 모드: `Normal`, `Warning`, `Critical`. 순서 관계가 있다 (Normal < Warning < Critical). |

> **매핑 관계**: Manager OperatingMode → Engine OperatingMode:
> Normal → Normal, Warning → Degraded, Critical → Minimal.
> Engine의 `Suspended`는 Emergency Level 또는 Suspend 명령으로만 진입한다.

### 품질 용어

| 용어 | 정의 |
|------|------|
| **QCF** | Quality Cost Function. 손실성 액션이 추론 품질에 미치는 비용을 정량화하는 프록시 메트릭. 어텐션 중요도 스코어와 V-norm 비율 기반. Manager가 Critical 전환 시 Engine에 RequestQcf를 전송하면, Engine이 현재 상태에서 각 lossy 액션의 QCF를 1회 계산하여 QcfEstimate로 응답한다. |
| **NLL** | Negative Log-Likelihood. 모델 출력 품질을 측정하는 지표. 낮을수록 좋다. |
| **DegradationEstimator** | QCF 프록시 값을 PPL(Perplexity) 증가량으로 변환하는 보정 모듈. 피스와이즈 선형 곡선과 EMA 보정을 사용한다. |

### 액션 용어

| 용어 | 정의 |
|------|------|
| **Lossless Action** | 추론 품질에 영향을 주지 않는 액션. Backend 전환, 디스크 오프로드, 스로틀링 등. |
| **Lossy Action** | 추론 품질에 영향을 줄 수 있는 액션. KV 캐시 eviction, 양자화, 레이어 스킵 등. |
| **Action Pool** | 시스템이 사용할 수 있는 모든 액션의 집합. Lossless와 Lossy로 분류된다. |
| **Relief** | 액션이 해소하는 압력의 양. ReliefVector로 표현된다. |
| **ReliefVector** | 액션의 4차원 효과 예측: compute, memory, thermal 릴리프(양수 = 압력 감소)와 latency 비용(음수 = 지연 증가). |

### 프로토콜 용어

| 용어 | 정의 |
|------|------|
| **EngineDirective** | Manager가 Engine에 전송하는 명령 묶음. `seq_id`(단조 증가 시퀀스 번호)와 `commands`(EngineCommand 벡터)로 구성된다. |
| **EngineCommand** | 개별 명령. Throttle, LayerSkip, KvEvictH2o, KvEvictSliding, KvMergeD2o, KvStreaming, KvQuantDynamic, RequestQcf, RestoreDefaults, SwitchHw, PrepareComputeUnit, Suspend, Resume. |
| **SystemSignal** | Manager 내부에서 Monitor가 생성하는 리소스 상태 신호. MemoryPressure, ComputeGuidance, ThermalAlert, EnergyConstraint의 4종. |
| **Heartbeat** | Engine이 주기적으로 Manager에 보내는 상태 보고 (EngineStatus). |
| **Capability** | Engine이 연결 시 Manager에 보내는 초기 등록 메시지. 사용 가능 디바이스, KV 캐시 용량 등을 포함한다. |

### 캐시 용어

| 용어 | 정의 |
|------|------|
| **Eviction** | KV 캐시에서 토큰을 제거하여 공간을 확보하는 행위. |
| **H2O** | Heavy-Hitter Oracle. 어텐션 스코어 기반으로 중요 토큰을 보존하고 나머지를 제거하는 eviction 정책. 3-partition 구조 (prefix + heavy-hitter + recent). |
| **StreamingLLM** | Attention Sink 토큰과 슬라이딩 윈도우를 결합한 eviction 정책. sink_size + window_size로 설정한다. |
| **D2O** | Dynamic Discriminative Operations. H2O 변형으로, eviction 시 토큰 병합 보상(scatter-reduce merge)을 수행한다. |
| **SnapKV** | 어텐션 패턴 관찰 기반으로 중요 KV를 선별하여 캐시를 압축하는 기법. |
| **KIVI** | Key-Value Integer Quantization. KV 캐시를 낮은 비트(Q2/Q4/Q8)로 동적 양자화하여 메모리를 절감하는 기법. FP32 잔차를 유지한다. |
| **Offload** | 현재 활성이 아닌 KV 캐시를 외부 저장소(디스크/메모리)로 옮기는 행위. 레이어 단위로 prefetch/release를 수행한다. |

### 하드웨어 용어

| 용어 | 정의 |
|------|------|
| **Backend** | 하드웨어별 연산을 추상화하는 인터페이스. matmul, softmax, RoPE 등 텐서 연산을 디바이스별로 구현한다. |
| **ARM64/NEON** | ARM 64비트 아키텍처와 NEON SIMD 확장. Q4_0 양자화 커널에서 `vdotq_s32` 등 벡터 명령을 사용한다. |
| **OpenCL** | 이기종 컴퓨팅 프레임워크. Adreno GPU에서 matmul, softmax, RoPE 커널을 실행한다. |
| **Zero-copy** | CPU와 GPU가 물리 메모리를 공유하여 데이터 복사 없이 접근하는 메모리 모델. ARM UMA SoC에서 `CL_MEM_ALLOC_HOST_PTR`로 구현한다. |
| **Galloc** | CPU 전용 SharedBuffer 할당자. 테스트와 CPU-only 빌드에서 사용한다. |
| **UMA** | Unified Memory Architecture. CPU와 GPU가 동일 물리 메모리를 공유하는 아키텍처. Snapdragon 등 모바일 SoC의 특성. |

## 3. Specification

### 3.1 System Context [SYS-001 ~ SYS-009]

llm_rs2는 온디바이스 LLM 추론 시스템으로, 모바일/엣지 디바이스에서 대형 언어 모델을 실행한다.

**[SYS-001]** 시스템은 두 개의 독립 OS 프로세스로 구성된다: **Engine**과 **Manager**. 두 프로세스는 IPC를 통해 통신하며, 직접적인 코드 의존성이 없다. *(MUST)*

**[SYS-002]** 시스템은 다음 외부 엔티티와 상호작용한다: *(MUST)*

| 외부 엔티티 | 인터페이스 | 역할 |
|-------------|-----------|------|
| 운영체제 | `/proc/meminfo`, `/proc/stat`, `/sys/class/thermal/`, `/sys/class/power_supply/` | 리소스 상태 수집 |
| 사용자 | CLI (stdin/stdout) | 추론 요청 제출 및 결과 수신 |
| 모델 파일 | HuggingFace Safetensors 포맷 (파일시스템) | 모델 가중치, 설정, 토크나이저 로딩 |
| 하드웨어 | CPU (ARM64 NEON, x86_64), GPU (OpenCL) | 텐서 연산 실행 |

**[SYS-003]** Engine은 LLM 추론 전담 프로세스이다. 모델 로딩, forward pass 실행, KV 캐시 관리, 백엔드 연산 디스패치, 리소스 적응(Resilience)을 담당한다. *(informational)*

**[SYS-004]** Manager는 시스템 리소스 모니터링 전담 서비스 프로세스이다. 센서 수집, 정책 평가, 디렉티브 전송을 담당한다. *(informational)*

**[SYS-005]** Shared는 Engine과 Manager가 공유하는 IPC 메시지 타입을 정의하는 라이브러리이다. 양 프로세스는 Shared에만 의존하며, 서로를 직접 의존하지 않는다. *(MUST)*

**[SYS-006]** 시스템은 사용자 입력(프롬프트)을 받아 토큰 단위로 텍스트를 생성한다. Prefill 단계에서 입력을 처리하고, Decode 단계에서 자기회귀적으로 토큰을 생성한다. *(MUST)*

**[SYS-007]** Manager에서 Engine으로의 통신은 ManagerMessage 타입을, Engine에서 Manager로의 통신은 EngineMessage 타입을 사용한다. 이 타입들은 Shared에서 정의된다. *(MUST)*

**[SYS-008]** 시스템은 단일 디바이스에서 실행된다. 분산 추론은 지원하지 않는다. *(informational)*

#### 불변식

- **[INV-001]** 시스템은 항상 정확히 2개의 독립 프로세스(Engine, Manager)로 구성된다. Engine과 Manager 간에 직접 코드 의존이 존재해서는 안 된다. 양 프로세스의 유일한 공유 의존성은 Shared 라이브러리이다. *(MUST NOT)*

### 3.2 System Goals [SYS-010 ~ SYS-019]

**[SYS-010]** 시스템은 온디바이스 LLM 추론을 제공해야 한다. 인터넷 연결 없이 디바이스 로컬에서 모델을 실행할 수 있어야 한다. *(MUST)*

**[SYS-011]** 시스템은 적응형 리소스 관리를 제공해야 한다. 메모리, 열, 연산 도메인의 리소스 압박을 연속적으로 추정하고, 다수 도메인의 압력을 동시에 고려하는 액션 조합을 선택하여 추론을 지속할 수 있어야 한다. *(MUST)*

> **참고 (non-normative)**: 도메인별 압력 계산 전략은 특성에 따라 다르다. Compute/Thermal은 PI Controller(평활화), Memory는 임계값 기반 직접 매핑(즉각 대응)을 사용한다. 전력(Energy) 도메인은 Compute PI에 보조 기여로 반영된다. 각 전략은 교체 가능하다 (`22-manager-algorithms.md` MGR-ALG-010 참조).

**[SYS-012]** 시스템은 품질 비용 추적(QCF)을 제공해야 한다. *(MUST)*

- Critical 모드 전환 시 Manager는 Engine에 `RequestQcf`를 전송한다 (MUST).
- Engine은 현재 KV 캐시/모델 상태에서 각 lossy 액션의 예상 QCF 비용을 읽기 전용으로 계산하여 `QcfEstimate`로 응답한다 (MUST).
- Manager의 ActionSelector는 이 QCF 값을 lossy 액션의 비용으로 사용하여 최소 품질 저하 조합을 선택한다 (MUST).
- Engine이 미연결이거나 QcfEstimate를 수신하지 못하면 ActionRegistry의 default_cost를 fallback으로 사용한다 (MUST).

**[SYS-013]** 시스템은 다중 컴퓨트 백엔드를 지원해야 한다. CPU와 GPU(OpenCL) 백엔드 간 동적 전환이 가능해야 한다. *(MUST)*

**[SYS-014]** 시스템은 다중 양자화 포맷을 지원해야 한다. F32, F16, BF16, Q4_0, Q4_1 가중치 포맷을 로딩하고 연산할 수 있어야 한다. *(MUST)*

#### 비목표 (Non-Goals)

**[SYS-015]** 시스템은 모델 학습(training)을 지원하지 않는다. 추론 전용 시스템이다. *(informational)*

**[SYS-016]** 시스템은 범용 모델 아키텍처를 지원하지 않는다. Llama 계열 디코더 아키텍처만 대상으로 한다. *(informational)*

**[SYS-017]** 시스템은 분산 추론을 지원하지 않는다. 단일 디바이스에서만 실행된다. *(informational)*

**[SYS-018]** 시스템은 배치(batch) 서빙을 지원하지 않는다. 동시에 하나의 추론 세션만 처리한다. *(informational)*

**[SYS-019]** 시스템은 액션 효과의 온라인 학습을 제공해야 한다. 각 액션이 도메인별 압력에 미치는 릴리프(relief)를 런타임에 예측하고, 실측 관찰로 예측 모델을 보정해야 한다. *(MUST)*

> **구현 경로 (non-normative, 2026-04)**: SYS-019의 MUST 요건은 "액션별 릴리프의 runtime 예측 + 실측 관찰로 보정"이라는 **개념적 계약**이며, 특정 학습 알고리즘이나 특성 차원을 규정하지 않는다. 현재 두 구현 경로가 병존하며, SYS-019는 둘 중 **하나가 활성화되어 있을 것**을 요구한다. 두 경로는 각자의 저장 포맷(디스크 스키마)과 학습 주기(관찰 딜레이, 업데이트 규칙)를 독립적으로 관리한다.
>
> - **기본 경로 — LuaPolicy + `EwmaReliefTable`** (MGR-DAT-071): α=0.875 EWMA, 6차원 릴리프 벡터, 관찰 딜레이 3초. 관련 요구사항: MGR-090~093, MGR-ALG-080~083, MGR-DAT-070~074. 불변식: INV-086~090.
> - **확장 경로 — `#[cfg(feature = "hierarchical")]` HierarchicalPolicy + `ReliefEstimator`**: RLS(λ=0.995), 13차원 feature vector → 4차원 ReliefVector, 관찰 딜레이 3초. 관련 요구사항: MGR-027, MGR-ALG-040~047, MGR-DAT-045~046. 불변식: INV-046~051.

### 3.3 Target Platform [SYS-020 ~ SYS-029]

**[SYS-020]** 기본 타겟 플랫폼은 ARM64 Linux/Android이다. Snapdragon SoC(Adreno GPU 포함)를 주요 대상으로 한다. *(MUST)*

**[SYS-021]** 보조 타겟 플랫폼은 x86_64 Linux 및 macOS(Apple Silicon, ARM64)이다. SIMD 최적화 없이 스칼라 연산으로 동작한다. *(SHOULD)*

**[SYS-022]** ARM64 플랫폼에서 NEON SIMD 명령을 사용하여 양자화 커널을 최적화해야 한다. Q4_0 dot product에 `vdotq_s32` 등 벡터 명령을 활용한다. *(MUST, ARM64 한정)*

**[SYS-023]** NEON SIMD 최적화는 ARM64에서만 활성화된다. x86_64에서는 스칼라 폴백을 사용한다. *(MUST)*

**[SYS-024]** Manager는 다음 OS 인터페이스를 사용하여 리소스 상태를 수집한다: *(MUST)*

| 인터페이스 | 수집 대상 | Monitor |
|-----------|----------|---------|
| `/proc/meminfo` | 가용 메모리, 전체 메모리, swap 사용량 | MemoryMonitor |
| `/proc/stat` | CPU 사용률 (delta 기반) | ComputeMonitor |
| `/sys/class/thermal/` | 열 센서 온도, 스로틀링 상태 | ThermalMonitor |
| `/sys/class/power_supply/` | 배터리 상태, 충전 여부, 전력 예산 | EnergyMonitor |

**[SYS-025]** GPU 연산은 OpenCL 프레임워크를 통해 수행한다. 지원 커널: MatMul (F32, F16, Q4_0, Q8_0), RoPE, Softmax. *(MUST, `opencl` feature 활성 시)*

**[SYS-026]** ARM UMA SoC에서는 zero-copy 메모리 모델을 사용해야 한다. `CL_MEM_ALLOC_HOST_PTR` 플래그로 CPU/GPU 공유 메모리를 할당한다. *(SHOULD, OpenCL + UMA 환경)*

#### 불변식

- **[INV-002]** NEON SIMD 최적화 코드 경로는 ARM64 타겟에서만 활성화된다. x86_64 또는 기타 아키텍처에서 NEON 코드가 실행되어서는 안 된다. *(MUST NOT)*

### 3.4 Supported Models [SYS-030 ~ SYS-039]

**[SYS-030]** 시스템은 Llama 계열 디코더 아키텍처를 지원한다. Llama 3.2, Qwen2.5 등 동일 아키텍처를 따르는 모델이 대상이다. *(MUST)*

**[SYS-031]** 모델 포맷은 HuggingFace Safetensors를 사용한다. 모델 로딩에 필요한 파일: *(MUST)*

| 파일 | 용도 |
|------|------|
| `model.safetensors` (또는 분할된 `model-*.safetensors`) | 모델 가중치 |
| `config.json` | 모델 아키텍처 설정 (hidden_size, num_layers, num_attention_heads 등) |
| `tokenizer.json` | 토크나이저 정의 |

**[SYS-032]** 모델 로딩 시 `config.json`의 아키텍처 필드를 검증해야 한다. 지원하지 않는 아키텍처가 감지되면 로딩을 거부해야 한다. *(MUST)*

**[SYS-033]** 지원하는 가중치 데이터 타입: *(MUST)*

| 타입 | 크기 | 설명 |
|------|------|------|
| F32 | 4 bytes/element | 32비트 부동소수점 |
| F16 | 2 bytes/element | 16비트 부동소수점 |
| BF16 | 2 bytes/element | Brain Float 16 |
| Q4_0 | 18 bytes/32 elements | GGML 호환 4비트 양자화 (F16 scale + 16B quants) |
| Q4_1 | 20 bytes/32 elements | GGML 호환 4비트 양자화 (F16 min + F16 scale + 16B quants) |

**[SYS-034]** Q4_0 양자화는 GGML 호환 블록 포맷을 따른다. 한 블록은 32개 원소로 구성되며, BlockQ4_0 = F16 scale (2바이트) + quants (16바이트) = 18바이트이다. 압축률은 F32 대비 약 7.1배 (0.5625 bytes/element)이다. *(MUST)*

**[SYS-035]** 모델 가중치는 `[out_features, in_features]` 레이아웃(사전 전치)으로 저장된다. 추론 시 `matmul_transposed(activation, weight)` 연산으로 양쪽 피연산자의 행 단위 접근 패턴을 활용한다. *(MUST)*

#### 불변식

- **[INV-003]** 모델 로딩 시 `config.json`의 `architectures` 필드가 지원 목록에 존재하지 않으면, 시스템은 로딩을 거부하고 오류를 반환해야 한다. *(MUST)*

### 3.5 Quality Assurance Framework [SYS-040 ~ SYS-049]

**[SYS-040]** 모든 lossy action은 실행 시 QcfMetric을 생성해야 한다. QCF 수집이 활성화된 상태에서 lossy action이 QcfMetric 없이 실행되어서는 안 된다. *(MUST)*

**[SYS-041]** QcfMetric은 다음 필드를 포함해야 한다: *(MUST)*

| 필드 | 타입 | 범위 | 설명 |
|------|------|------|------|
| action | 문자열 | - | 액션 식별자 (예: "h2o", "kivi", "snapkv") |
| raw_value | 실수 | [0, 1] | 원시 프록시 값 |
| normalized_value | 실수 | ≥ 0 | 정책 간 비교 가능한 정규화 값. Eviction 계열: evicted_importance / remaining_importance로 1 초과 가능. 비-Eviction 계열: raw_value와 동일 [0, 1]. |
| tokens_affected | 정수 | ≥ 0 | 영향을 받은 토큰 수 |

**[SYS-042]** QcfMetric은 선택적으로 `per_head` 필드를 포함할 수 있다. 이는 KV 헤드별 분해 값의 벡터이다. *(MAY)*

**[SYS-043]** QCF 프록시 값에서 PPL 증가량(degradation)으로의 변환 경로가 존재해야 한다. DegradationEstimator가 피스와이즈 선형 곡선을 통해 변환을 수행한다. *(MUST)*

- 변환 함수는 breakpoint를 기준으로 두 구간의 기울기(slope_low, slope_high)를 적용한다.
- EMA(Exponential Moving Average) 보정을 통해 실측과 예측 간 오차를 온라인으로 보정한다.
- 출력은 [0, d_max] 범위로 클램핑된다.

**[SYS-044]** DegradationEstimator는 외부 보정 파일(JSON)에서 곡선 파라미터를 로딩할 수 있어야 한다. 보정 파일이 없으면 기본 1:1 선형 곡선을 사용한다. *(SHOULD)*

**[SYS-045]** QCF 수집은 설정으로 활성화/비활성화할 수 있어야 한다. 비활성 시 QCF 계산 오버헤드가 발생하지 않아야 한다. *(MUST)*

**[SYS-046]** 헤드 수준 QCF를 스칼라로 집계하는 방식은 설정 가능해야 한다: *(MUST)*

| 집계 모드 | 설명 |
|----------|------|
| Mean | 산술 평균. 모든 헤드를 동등하게 취급한다. |
| Defensive | Softmax 가중 평균. temperature 파라미터로 최악 헤드 강조 정도를 조절한다. 낮은 temperature = 최악 헤드에 집중. |

#### 불변식

- **[INV-004]** QCF 수집이 활성화된 상태에서 lossy action이 실행되면, 반드시 QcfMetric이 생성되어야 한다. *(MUST)*

### 3.6 Fail-Safety Requirements [SYS-050 ~ SYS-059]

**[SYS-050]** Manager 장애 시 Engine은 독립적으로 추론을 계속해야 한다 (fail-open). Manager 프로세스의 크래시, 연결 끊김, 무응답이 Engine의 추론 루프를 중단시켜서는 안 된다. *(MUST)*

**[SYS-051]** Engine 장애 시 Manager는 독립적으로 모니터링을 계속해야 한다. Engine 연결 끊김이 Manager의 센서 수집과 정책 평가를 중단시켜서는 안 된다. *(MUST)*

**[SYS-052]** Engine과 Manager 간 IPC 연결이 끊어지면, 양 프로세스는 graceful degradation으로 전환해야 한다: *(MUST)*

| 프로세스 | 동작 |
|---------|------|
| Engine | Resilience 서브시스템 비활성. 기본 추론을 지속한다. |
| Manager | Emitter 호출 건너뜀. Policy 루프는 계속 실행하며, 재연결을 시도한다. |

**[SYS-053]** Resilience 서브시스템의 내부 장애는 추론 루프를 크래시시켜서는 안 된다. Resilience가 실패하더라도 추론은 기본 모드로 계속 진행한다 (fail-open 원칙). *(MUST)*

**[SYS-054]** Manager의 Monitor 스레드 하나의 장애가 다른 Monitor 스레드에 전파되어서는 안 된다. 각 Monitor는 독립 스레드에서 실행된다. *(MUST)*

**[SYS-055]** 어느 도메인이든 Emergency Level에 도달하면, Engine은 자율적으로 안전 대응을 수행해야 한다. 이는 Manager의 Directive 없이도 동작해야 한다. *(MUST)*

| 도메인 | Emergency 대응 | 근거 |
|--------|---------------|------|
| Thermal | 즉시 Suspend (추론 중지) | 과열 보호. 하드웨어 손상 방지. |
| Energy | 즉시 Suspend + 새 요청 거부 | 배터리 보호. 전원 차단 방지. |
| Memory | 공격적 eviction (25%) + 새 요청 거부 | 메모리 해제로 대응. OOM killer 회피. Suspend 불필요. |

> **참고 (non-normative)**: Engine의 Emergency 자율 대응은 D-Bus 전송 경로에서 SystemSignal을 직접 수신하는 Strategy 기반 경로(01-architecture.md SYS-085a 참조)로 구현된다. 양방향 프로토콜(Unix Socket/TCP) 경로에서는 Manager가 Emergency 신호를 pressure=1.0으로 변환하여 Critical 모드의 Directive를 전송하며, 필요시 `Suspend` 명령을 포함한다.

#### 불변식

- **[INV-005]** Manager 장애(크래시, 연결 끊김)가 Engine의 추론 루프를 중단시켜서는 안 된다. *(MUST NOT)*
- **[INV-006]** Engine 장애(크래시, 연결 끊김)가 Manager의 모니터링 루프를 중단시켜서는 안 된다. *(MUST NOT)*

### 3.7 Implementation Language and Dependencies [SYS-060 ~ SYS-069]

**[SYS-060]** 구현 언어는 Rust이다. *(MUST)*

**[SYS-061]** 프로젝트는 Cargo workspace 구조로 구성된다. 3개의 크레이트를 포함한다: *(MUST)*

| 크레이트 | 패키지명 | 유형 | 역할 |
|---------|---------|------|------|
| engine | `llm_rs2` | 바이너리 | LLM 추론 엔진 |
| manager | `llm_manager` | 바이너리 | 리소스 모니터 서비스 |
| shared | `llm_shared` | 라이브러리 | 공유 IPC 타입 |

**[SYS-062]** Feature gate 체계: *(MUST)*

| Feature | 크레이트 | 기본 | 역할 |
|---------|---------|------|------|
| `opencl` | engine | 활성 | OpenCL GPU 백엔드 활성화 |
| `resilience` | engine | 비활성 | D-Bus 기반 Resilience 수신 활성화 |
| `dbus` | manager | 활성 | D-Bus Emitter 활성화 |

**[SYS-063]** 릴리스 빌드 프로파일은 최대 최적화를 적용해야 한다: *(SHOULD)*

| 설정 | 값 | 목적 |
|------|---|------|
| LTO | fat | 전체 프로그램 최적화 |
| codegen-units | 1 | 단일 컴파일 유닛 최적화 |
| opt-level | 3 | 최대 최적화 |
| panic | abort | 언와인딩 제거, 바이너리 크기 축소 |

**[SYS-064]** 시스템은 async 런타임을 사용하지 않는다. 스레딩은 `std::thread`, 스레드 간 통신은 `mpsc::channel`만 사용한다. *(MUST NOT)*

**[SYS-065]** IPC 직렬화는 JSON을 사용한다. 바이너리 직렬화 포맷은 사용하지 않는다. *(MUST)*

## 4. Alternative Behavior

해당 없음. 이 문서는 시스템 수준 개요로서, 대안 동작은 개별 컴포넌트 스펙에서 정의한다.

## 5. Constraints

**메모리 제약**: 시스템은 4~12 GB RAM의 모바일 디바이스 환경을 가정한다. 모델 가중치 + KV 캐시 + 워크스페이스의 총 메모리 사용량이 디바이스 가용 메모리 내에 있어야 한다.

> 참고 메모리 예산 (Llama 3.2 1B, seq=2048):
> - 모델 가중치 (Q4_0): ~1.2 GB
> - KV 캐시 (F16): ~64 MB (16 layers × 4 MB)
> - KV 캐시 (F32): ~128 MB
> - 워크스페이스: ~128 KB (공유)
> - **총합**: ~1.3–1.4 GB

**단일 모델, 단일 세션**: 동시에 하나의 모델만 로딩하고, 하나의 추론 세션만 처리한다. 다중 모델 병렬 로딩이나 다중 세션 동시 처리는 지원하지 않는다.

**No async runtime**: `std::thread` + `mpsc`만 사용한다. tokio, async-std 등의 async 런타임에 의존하지 않는다.

## 6. Examples

### 예시 1: 시스템 배포 시나리오

Android 디바이스(Snapdragon 8 Gen 3, 12 GB RAM)에 시스템을 배포하고 실행하는 시나리오:

```
1. 바이너리 배포
   $ adb push llm_manager /data/local/tmp/
   $ adb push llm_rs2 /data/local/tmp/

2. 모델 파일 배포
   /data/local/tmp/model/config.json
   /data/local/tmp/model/model.safetensors
   /data/local/tmp/model/tokenizer.json

3. Manager 시작
   $ ./llm_manager --socket /tmp/llm.sock --config policy_config.toml

4. Engine 시작 및 추론
   $ ./llm_rs2 --model /data/local/tmp/model/ \
               --socket /tmp/llm.sock \
               --prompt "Hello, world"

5. 실행 흐름
   Engine → Manager: Capability (디바이스 정보 등록)
   Engine → Manager: Heartbeat (주기적 상태 보고, ~100ms 간격)
   Manager → Engine: Directive (리소스 상황에 따른 명령)
   Engine: 토큰 생성 → stdout 출력
```

### 예시 2: 리소스 압박 대응 시나리오

메모리 압박이 발생하여 시스템이 적응하는 시나리오:

```
시간  | 가용 메모리 | Level    | Manager 액션                     | Engine 상태
──────|───────────|──────────|─────────────────────────────────|────────────────
t=0   | 60%       | Normal   | (없음)                          | 정상 Decode
t=10  | 35%       | Warning  | Directive: Throttle(delay=20ms) | Decode + 20ms 지연
t=20  | 20%       | Critical | Directive: KvEvictH2o(keep=0.5) | KV 캐시 50% 유지
t=25  | 40%       | Warning  | (hold_time 대기, 아직 de-escalation 안 함) | 축소된 캐시로 Decode
t=30  | 55%       | Normal   | Directive: RestoreDefaults      | 정상 Decode 복귀
```

이 시나리오에서:
- t=10: PI Controller가 memory pressure > warning_threshold (0.4)를 감지. Supervisory가 Warning 모드로 전환. Warning이므로 lossless action만 선택 → Throttle.
- t=20: pressure > critical_threshold (0.7). Supervisory가 Critical로 즉시 에스컬레이션. Lossy action 허용 → KvEvictH2o.
- t=25~30: pressure가 release 임계값 아래로 떨어지지만, hold_time (4초) 동안 안정화를 확인한 후에야 디에스컬레이션.

## 7. Rationale (non-normative)

### 왜 Rust인가

- **메모리 안전성**: GC 없이 메모리 안전성을 보장한다. 수 GB의 모델 가중치와 KV 캐시를 다루는 환경에서 use-after-free, 이중 해제 등의 메모리 버그를 컴파일 타임에 방지한다.
- **Zero-cost abstraction**: Backend trait 디스패치가 단형화(monomorphization)를 통해 런타임 오버헤드 없이 구현된다.
- **크로스 컴파일**: Cargo와 Android NDK를 통해 ARM64 Android 타겟으로의 크로스 컴파일이 간단하다.
- **FFI**: OpenCL C API와의 FFI 통합이 안전 경계(safety boundary) 내에서 가능하다.

### 왜 2-프로세스 분리인가

- **Fail-safety**: Manager 장애가 Engine을 중단시키지 않고, Engine 장애가 Manager를 중단시키지 않는다 ([SYS-050], [SYS-051]).
- **독립 배포**: Engine과 Manager를 독립적으로 업데이트하고 재시작할 수 있다.
- **리소스 격리**: 모니터링 로직(Monitor 스레드, Policy 평가)과 추론 로직(forward pass, KV 연산)이 별도 프로세스에서 실행되어, 모니터링이 추론 성능에 영향을 주지 않는다.

### 왜 OpenCL인가

- **Adreno 최적화**: Snapdragon SoC의 Adreno GPU는 성숙한 OpenCL 지원을 제공한다.
- **개발 생산성**: `.cl` 파일로 커널을 작성하고 JIT 컴파일하여, Vulkan Compute 대비 보일러플레이트가 적다.
- **MatMul 워크로드 특성**: matmul 중심 워크로드에서 OpenCL과 Vulkan Compute 간 유의미한 성능 차이가 없다.

### 왜 matmul_transposed가 기본인가

- HuggingFace 가중치가 `[out_features, in_features]` (사전 전치) 레이아웃으로 저장된다.
- `matmul_transposed(A, B^T)`는 양쪽 피연산자를 행 단위로 접근하여 캐시 효율이 높다.
- Q4_0 양자화의 32-element 블록이 출력 뉴런 단위로 연속 저장되어 행 단위 접근과 호환된다.
