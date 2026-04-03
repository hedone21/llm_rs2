# Engine CUDA Backend

> **TL;DR**: NVIDIA Jetson (Xavier/Orin) GPU 가속을 위한 CUDA 백엔드 스펙. cudarc 크레이트를 통해 llama.cpp의 CUDA 커널(`.cu` → `.ptx`)을 로딩·실행하여 구현 비용을 최소화한다. `cuda` feature gate로 조건부 컴파일되며, 기존 `Backend` trait을 그대로 구현한다. Unified Memory 기반 zero-copy 메모리 모델을 사용하고, Tensor Core (WMMA/MMA)를 활용한 양자화 GEMV/GEMM을 지원한다.

## 1. Purpose and Scope

이 문서는 Engine의 CUDA 백엔드 서브시스템을 정의한다.

**이 파일이 명세하는 것:**

- CUDA 백엔드의 책임과 경계
- 지원 디바이스 및 Compute Capability 요구사항
- cudarc + llama.cpp PTX 커널 재사용 전략
- 메모리 모델 (Unified Memory)
- Backend trait 연산별 CUDA 매핑
- Feature gate 및 빌드 시스템 요구사항
- CLI 인터페이스 확장

**이 파일이 명세하지 않는 것:**

- Backend trait 자체의 정의 → `30-engine.md` [ENG-013]
- KV 캐시 알고리즘 → `32-engine-algorithms.md`
- OpenCL 백엔드 상세 → 기존 코드베이스
- Manager/Resilience 상호작용 → `20-manager.md` ~ `23-manager-data.md`

## 2. Definitions

| 용어 | 정의 |
|------|------|
| **CUDA** | NVIDIA GPU 프로그래밍 모델. `.cu` 커널 코드를 `nvcc`로 컴파일한다. |
| **Jetson** | NVIDIA의 ARM64 기반 임베디드 AI 플랫폼. CPU-GPU가 물리 DRAM을 공유(UMA)한다. |
| **cudarc** | Rust용 CUDA Driver/Runtime API safe wrapper 크레이트. cuBLAS, cuDNN 등 라이브러리 래핑 포함. |
| **PTX** | Parallel Thread Execution. NVIDIA GPU의 중간 표현(IR). `nvcc`로 `.cu` → `.ptx` 컴파일 후 런타임에 로딩한다. |
| **Compute Capability (CC)** | NVIDIA GPU의 하드웨어 기능 수준. 숫자가 높을수록 최신 기능 지원. |
| **Unified Memory** | CPU와 GPU가 동일한 가상 주소 공간을 공유하는 CUDA 메모리 모델. `cudaMallocManaged()`. |
| **Tensor Core** | NVIDIA GPU의 행렬 곱 전용 하드웨어. WMMA(Volta+), MMA(Turing+) 명령어로 접근. |
| **MMVQ** | Matrix-Multiply Vector Quantized. 양자화된 weight × activation 벡터 곱. |
| **MMQ** | Matrix-Multiply Quantized. 양자화된 weight × activation 행렬 곱 (배치 > 8). |
| **CUDA Graph** | CUDA의 커널 실행 계획 최적화. 커널 시퀀스를 그래프로 캡처 후 단일 launch로 재실행. OpenCL Plan의 CUDA 대응. |

## 3. Specification

### 3.1 CUDA Backend 개요 [ENG-CUDA-010]

**[ENG-CUDA-010]** CudaBackend는 `Backend` trait의 CUDA 구현체이다. `cuda` feature gate로 조건부 컴파일된다. NVIDIA Jetson 디바이스에서 Tensor Core를 활용한 GPU 가속 추론을 제공한다. *(MUST)*

**[ENG-CUDA-011]** CudaBackend는 llama.cpp의 CUDA 커널 소스(`.cu`)를 `nvcc`로 프리컴파일하여 PTX를 생성하고, cudarc를 통해 런타임에 로딩·실행한다. 자체 CUDA 커널을 작성하지 않는다. *(MUST)*

> **Rationale (non-normative)**: llama.cpp의 ggml-cuda 디렉토리에 50+ 커널이 CC별 최적화, Tensor Core 활용, 양자화 지원을 모두 구현하고 있다. cudarc의 `CudaModule::from_ptx()`로 프리컴파일된 PTX를 로딩하면, ggml의 graph/tensor 시스템에 의존하지 않고 개별 커널만 재사용할 수 있다. C FFI 바인딩 수동 작성 대비 Rust safe API로 타입 안전성이 높고, ggml API 변경에도 커널 함수 시그니처만 맞으면 영향 없다.

**[ENG-CUDA-012]** CudaBackend는 다음 Jetson 디바이스를 지원한다: *(MUST)*

| 디바이스 | GPU 아키텍처 | Compute Capability | 주요 기능 |
|---------|-------------|--------------------|----|
| Jetson Xavier / Xavier NX | Volta | sm_72 | WMMA Tensor Core, FP16 MMA, dp4a |
| Jetson Orin / Orin NX / Orin Nano | Ampere 변형 | sm_87 | MMA Tensor Core, INT8 TC, TF32, BF16 |

**[ENG-CUDA-013]** CudaBackend의 최소 Compute Capability는 sm_72 (Volta)이다. sm_72 미만의 GPU(Nano sm_53, TX2 sm_62)는 지원하지 않는다. *(MUST)*

> **Rationale (non-normative)**: sm_72 이상에서 WMMA Tensor Core가 사용 가능하여 양자화 GEMM 성능이 비약적으로 향상된다. sm_53/sm_62는 Tensor Core 미지원으로 CPU 대비 이점이 미미하다.

### 3.2 cudarc + PTX 커널 전략 [ENG-CUDA-020]

**[ENG-CUDA-020]** CUDA 커널 통합은 cudarc 크레이트를 통해 수행한다. ggml-cuda의 `.cu` 소스를 `build.rs`에서 `nvcc`로 PTX 컴파일하고, cudarc의 `CudaModule::from_ptx()`로 로딩한다. *(MUST)*

```
빌드 시: .cu (llama.cpp) → nvcc → .ptx (프리컴파일)
런타임:  .ptx → cudarc CudaModule::from_ptx() → CudaFunction → launch_kernel()
```

**[ENG-CUDA-021]** cudarc의 op-level 커널 launch API를 사용한다. ggml의 graph-based 실행 모델이나 ggml_tensor 구조체는 사용하지 않는다. *(MUST)*

> **Rationale (non-normative)**: llm.rs는 op-by-op 실행 모델이다. cudarc의 `CudaFunction::launch()`로 개별 커널을 직접 호출하면 llm.rs의 실행 흐름(Backend trait → layer-by-layer forward)을 그대로 유지할 수 있다. ggml의 graph/tensor 시스템을 우회하므로 ggml API 변경에 대한 노출이 최소화된다.

**[ENG-CUDA-022]** 커널 소스 매핑 — llama.cpp `.cu` 파일에서 필요한 커널 함수를 추출하여 PTX로 컴파일한다: *(MUST)*

| llm.rs Backend 연산 | 커널 함수 | llama.cpp 소스 파일 |
|---------------------|---------|-------------------|
| `matmul` / `matmul_transposed` | MMVQ (batch≤8) / MMQ (batch>8) | `mmvq.cu`, `mmq.cu` |
| `matmul` (F16/F32) | cuBLAS HGEMM/SGEMM | cudarc cuBLAS 래핑 |
| `rms_norm` | `rms_norm_f32` | `norm.cu` |
| `rope_inplace` | `rope_norm` / `rope_neox` | `rope.cu` |
| `softmax` | `soft_max_f32` | `softmax.cu` |
| `silu_mul` | `silu` + `mul` | `unary.cu`, `bin_bcast.cu` |
| `gelu_tanh_mul` | `gelu` + `mul` | `unary.cu`, `bin_bcast.cu` |
| `attention_gen` | Flash Attention (MMA/WMMA/tile/vec) | `fattn*.cu` |
| `add_assign` | `add_f32` | `bin_bcast.cu` |
| `scale` | `scale_f32` | `scale.cu` |
| `copy_from` / `copy_into` | `cpy` | `cpy.cu` |
| `cast` | `convert` | `convert.cu` |
| `gather` | `get_rows` | `get_rows.cu` |

**[ENG-CUDA-023]** 커널 호출 시 cudarc의 CUDA stream을 사용하여 비동기 실행한다. `synchronize()` 호출 시 `CudaStream::synchronize()`로 동기화한다. *(MUST)*

**[ENG-CUDA-024]** cuBLAS 연산(HGEMM, SGEMM)은 cudarc의 `cudarc::cublas` 모듈을 통해 호출한다. F16/F32 matmul에서 양자화 커널 대신 사용한다. *(MUST)*

**[ENG-CUDA-025]** 커널 PTX 파일은 빌드 산출물로 포함한다. `include_bytes!()` 또는 `CudaModule::from_ptx_file()`로 로딩한다. *(SHOULD)*

### 3.3 메모리 모델 [ENG-CUDA-030]

**[ENG-CUDA-030]** Jetson은 UMA(Unified Memory Architecture)이다. CudaBackend는 cudarc의 `CudaDevice::alloc()` (Unified Memory)을 기본 할당 전략으로 사용한다. *(MUST)*

> **Rationale (non-normative)**: Jetson에서 CPU와 GPU는 물리 DRAM을 공유한다. cudarc의 `CudaSlice<T>`가 Unified Memory를 관리하며, CPU에서 `.device_ptr()`로 GPU 포인터를, host copy로 CPU 접근을 수행한다. ARM SoC의 OpenCL `CL_MEM_ALLOC_HOST_PTR`과 동일한 원리.

**[ENG-CUDA-031]** CudaBuffer는 `Buffer` trait을 구현하며, 다음 속성을 가진다: *(MUST)*

| 메서드 | 동작 |
|--------|------|
| `as_ptr()` | Unified Memory 포인터 반환 (CPU에서 직접 접근 가능) |
| `as_mut_ptr()` | Unified Memory 가변 포인터 반환 |
| `cl_mem()` | `None` 반환 (OpenCL 핸들 없음) |
| `cuda_ptr()` | cudarc `CudaSlice` 디바이스 포인터 반환 (GPU 커널용) |
| `sync_device()` | `CudaDevice::synchronize()` 호출 |
| `is_host_managed()` | `true` (Jetson UMA에서 CPU가 항상 접근 가능) |

**[ENG-CUDA-032]** 모델 가중치는 CPU 메모리(mmap)에서 로딩 후 `cudarc::driver::sys::cuMemHostRegister()`로 GPU에 등록하여 zero-copy로 접근한다. *(SHOULD)*

> **Rationale (non-normative)**: `cuMemHostRegister`는 기존 호스트 메모리를 GPU에 매핑하는 zero-copy 방식으로, `ClWrappedBuffer`의 `CL_MEM_USE_HOST_PTR`과 동일한 역할. 추가 메모리 할당 없이 OOM을 방지한다.

**[ENG-CUDA-033]** KV 캐시 버퍼는 cudarc `CudaDevice::alloc()`으로 할당한다. KV 캐시의 grow-on-demand는 새 `CudaSlice` 할당 + `dtod_copy`로 구현한다. *(MUST)*

### 3.4 양자화 지원 [ENG-CUDA-040]

**[ENG-CUDA-040]** CudaBackend는 다음 양자화 포맷의 matmul을 지원한다: *(MUST)*

| Weight DType | Activation DType | CUDA 커널 경로 | 비고 |
|-------------|-----------------|---------------|------|
| Q4_0 | F32 | MMVQ (`vec_dot_q4_0_q8_1`) | activation을 on-the-fly Q8_1로 양자화 |
| Q8_0 | F32 | MMVQ (`vec_dot_q8_0_q8_1`) | |
| F16 | F32 | cudarc cuBLAS HGEMM | Tensor Core 활용 |
| F32 | F32 | cudarc cuBLAS SGEMM | fallback |

**[ENG-CUDA-041]** 양자화된 KV 캐시(Q4_0, F16)에 대한 Flash Attention은 llama.cpp의 `fattn-vec` 커널(PTX)로 직접 수행한다. 역양자화→F32 변환 없이 양자화 상태에서 attention을 계산한다. *(SHOULD)*

### 3.5 Feature Gate 및 빌드 시스템 [ENG-CUDA-050]

**[ENG-CUDA-050]** CUDA 백엔드는 `cuda` Cargo feature로 게이트된다. `opencl` feature와 상호 배타적이다. *(MUST)*

```toml
[features]
default = ["opencl"]
opencl = ["ocl"]
cuda = ["cudarc"]  # opencl과 동시 활성화 불가
```

> **Rationale (non-normative)**: 단일 바이너리에서 OpenCL과 CUDA를 동시에 지원할 실익이 없다. Jetson은 CUDA만, Adreno/Mali는 OpenCL만 사용한다. 상호 배타적 feature gate로 컴파일 복잡성을 제거한다.

**[ENG-CUDA-051]** 빌드 시스템은 다음 요구사항을 충족한다: *(MUST)*

| 항목 | 요구사항 |
|------|---------|
| CUDA Toolkit | 12.0 이상 (JetPack 6 기준) |
| nvcc | `build.rs`에서 llama.cpp `.cu` → `.ptx` 컴파일 |
| Target CC | `-arch=sm_72 -arch=sm_87` (Xavier + Orin) |
| cudarc | `cudarc = "0.19"` (Cargo.toml) |
| 커널 소스 | `vendor/ggml-cuda/` (llama.cpp에서 필요한 `.cu` 파일만 추출) |

**[ENG-CUDA-052]** `build.rs`는 llama.cpp의 `.cu` 파일을 `nvcc --ptx`로 컴파일하여 PTX 파일을 생성한다. 생성된 PTX는 `OUT_DIR`에 저장되며 바이너리에 포함된다. *(MUST)*

```
build.rs 흐름:
1. vendor/ggml-cuda/*.cu → nvcc --ptx -arch=sm_72 -arch=sm_87
2. → OUT_DIR/*.ptx
3. → include_bytes!() 또는 런타임 파일 로딩
```

> **Rationale (non-normative)**: ggml 전체 라이브러리를 빌드하지 않고 커널 PTX만 생성하므로 빌드 의존성이 최소화된다. ggml의 graph/tensor 시스템, 메모리 관리자 등에 대한 의존이 전혀 없다.

### 3.6 Backend Trait 연산 매핑 [ENG-CUDA-060]

**[ENG-CUDA-060]** CudaBackend는 `Backend` trait의 모든 필수 메서드를 구현한다. 각 메서드는 cudarc를 통해 PTX 커널 또는 cuBLAS로 디스패치된다. *(MUST)*

**[ENG-CUDA-061]** matmul dispatch 우선순위: *(MUST)*

```
1. MMVQ — 양자화 weight, batch ≤ 8 (decode 토큰 생성)
2. MMQ  — 양자화 weight, batch > 8 (prefill)
3. MMVF — F16/F32 weight, thin matrix
4. cuBLAS GEMM — fallback
```

> **Rationale (non-normative)**: llama.cpp의 검증된 dispatch 로직을 그대로 재사용한다. MMVQ는 decode(단일 토큰)에서, MMQ/cuBLAS는 prefill(배치)에서 최적이다.

**[ENG-CUDA-062]** Flash Attention은 CC에 따라 자동 경로 선택: *(MUST)*

| CC 범위 | FA 경로 | 비고 |
|---------|---------|------|
| sm_72 (Volta) | WMMA FA | `fattn-wmma-f16.cuh` |
| sm_87 (Ampere) | MMA FA | `fattn-mma-f16.cuh`, GQA 최적화 |

**[ENG-CUDA-063]** `attention_gen` 호출 시 decode(seq_len=1)이면 `fattn-vec` 경로를, prefill(seq_len>1)이면 CC별 FA 경로를 선택한다. *(SHOULD)*

### 3.7 실행 최적화 [ENG-CUDA-065]

**[ENG-CUDA-065]** 초기 구현은 op-by-op 실행 모델을 사용한다. OpenCL 백엔드의 Plan 시스템(pre-bound kernel arguments)에 해당하는 최적화는 CUDA Graph으로 구현할 수 있으나, 초기 구현에서는 필수가 아니다. *(MAY)*

> **Rationale (non-normative)**: OpenCL의 Plan 시스템은 커널 arg 바인딩 + enqueue 오버헤드를 제거하기 위한 최적화이다. CUDA는 커널 launch 오버헤드가 OpenCL 대비 낮고, 필요 시 CUDA Graph으로 유사한 최적화가 가능하다. 초기에는 op-by-op으로 정확성을 확보한 후, 프로파일링 결과에 따라 CUDA Graph을 도입한다.

### 3.8 CLI 확장 [ENG-CUDA-070]

**[ENG-CUDA-070]** `--backend` 플래그에 `"cuda"` 옵션을 추가한다: *(MUST)*

| 값 | 동작 |
|----|------|
| `"cpu"` | CPU 백엔드 (기존) |
| `"opencl"` | OpenCL GPU 백엔드 (기존) |
| `"cuda"` | CUDA GPU 백엔드 (신규, `cuda` feature 필요) |

**[ENG-CUDA-071]** CUDA 전용 CLI 플래그: *(SHOULD)*

| 플래그 | 타입 | 기본값 | 설명 |
|--------|------|--------|------|
| `--cuda-device` | usize | 0 | CUDA 디바이스 인덱스 |
| `--cuda-no-mma` | bool | false | MMA/WMMA 비활성화 (cuBLAS fallback) |

### 3.9 SwitchHw 호환 [ENG-CUDA-080]

**[ENG-CUDA-080]** CudaBackend는 SwitchHw directive의 대상 백엔드로 사용 가능하다. CPU↔CUDA 전환 시 weight migration은 `cuMemHostRegister` (CPU→CUDA) 및 host pointer 직접 사용 (CUDA→CPU)으로 수행한다. *(SHOULD)*

**[ENG-CUDA-081]** KV 캐시 migration은 기존 `kv_migrate.rs`의 `migrate_kv_cache()` 함수를 재사용한다. CudaBackend의 `copy_from()`이 cudarc Unified Memory로의 복사를 수행한다. *(MUST)*

## 4. Constraints

**[ENG-CUDA-C01]** `cuda` feature와 `opencl` feature는 동시에 활성화할 수 없다. `build.rs`에서 컴파일 타임에 검증한다. *(MUST)*

**[ENG-CUDA-C02]** CudaBackend 초기화 시 GPU의 CC를 확인하고, sm_72 미만이면 에러를 반환한다. *(MUST)*

**[ENG-CUDA-C03]** llama.cpp의 CUDA 커널 소스(`.cu`)는 수정하지 않는다. 업스트림의 커널을 그대로 사용하여 유지보수 비용을 최소화한다. *(SHOULD)*

## 5. Examples (non-normative)

### 5.1 Jetson Orin에서 CUDA 추론

```bash
# 빌드 (Jetson 네이티브)
cargo build --release --features cuda --no-default-features

# 추론
./target/release/generate \
  --model-path models/llama3.2-1b \
  --backend cuda \
  --kv-type f16 \
  --max-seq-len 4096 \
  -n 500 --greedy
```

### 5.2 크로스 컴파일 (호스트 → Jetson)

```bash
source jetson.source  # CUDA toolkit + aarch64 toolchain 설정
cargo build --release --target aarch64-unknown-linux-gnu --features cuda --no-default-features
```

## 6. Rationale (non-normative)

### 왜 cudarc + PTX인가?

| 대안 | 장점 | 단점 | 결론 |
|------|------|------|------|
| **cudarc + PTX 로딩** | Rust safe API, ggml 의존성 없음, 커널만 재사용, 디버깅 용이 | nvcc 빌드 필요, PTX 관리 | **채택** |
| **ggml C FFI 래핑** | 빠른 초기 구현 | ggml API 변경 추적, unsafe 다수, graph/tensor 의존성 노출 위험 | 리스크 높음 |
| **CUDA 커널 직접 포팅** | 완전한 제어, 의존성 없음 | 50+ 커널 작성/최적화, 막대한 개발 비용 | 비현실적 |
| **cuBLAS만 사용** | 간단 | 양자화 미지원, 성능 열위 | 불충분 |

cudarc를 선택한 핵심 이유:

1. **ggml 의존성 제거**: 커널 `.cu` 파일만 가져오고, ggml의 tensor/graph/메모리 시스템은 사용하지 않음. API 변경에 면역.
2. **Rust 타입 안전성**: `extern "C"` 수동 바인딩 대신 cudarc의 safe wrapper 사용. `CudaSlice<T>`, `CudaFunction` 등 제네릭 타입.
3. **생태계 검증**: candle(HuggingFace), burn 등 주요 Rust ML 프로젝트가 cudarc 기반. 활발히 유지보수 중 (v0.19+).
4. **라이브러리 통합**: cuBLAS, cuDNN 래핑이 cudarc에 내장. 별도 FFI 불필요.

### 왜 sm_72 최소인가?

- sm_53 (Nano): Tensor Core 없음, dp4a 없음 → Q4_0 MMVQ가 스칼라로 폴백, CPU 대비 이점 미미
- sm_62 (TX2): dp4a는 있으나 Tensor Core 없음 → GEMM은 빠르지만 Flash Attention 비효율
- sm_72 (Xavier): WMMA Tensor Core + dp4a → 양자화 GEMM과 FA 모두 하드웨어 가속
- sm_87 (Orin): MMA Tensor Core + INT8 TC → 최적 성능

### Jetson UMA 메모리 모델

Jetson은 ARM SoC와 동일한 UMA 구조이다:

```
┌─────────────────────────────────────┐
│           Physical DRAM             │
│                                     │
│  ┌─────────┐       ┌─────────────┐  │
│  │  CPU    │       │  GPU        │  │
│  │  Core   │       │  (Volta/    │  │
│  │  (ARM)  │       │   Ampere)   │  │
│  └────┬────┘       └──────┬──────┘  │
│       │                   │         │
│       └───── Unified ─────┘         │
│              Memory                 │
│      (cudarc CudaSlice<T>)          │
└─────────────────────────────────────┘
```

Adreno `CL_MEM_ALLOC_HOST_PTR` ≈ Jetson `cudaMallocManaged` ≈ cudarc `CudaDevice::alloc()`: 동일한 물리 메모리를 CPU/GPU 양쪽에서 접근.

### llama.cpp 커널 매핑 상세

llama.cpp의 CUDA matmul dispatch 로직:

```
Weight dtype?
├── Q4_0/Q8_0 (양자화)
│   ├── batch ≤ 8 → MMVQ (mmvq.cu)
│   │   └── activation → on-the-fly Q8_1 양자화
│   │   └── vec_dot_q4_0_q8_1 (dp4a, sm_61+)
│   └── batch > 8 → MMQ (mmq.cu)
│       └── Tensor Core MMA (sm_72+)
├── F16
│   ├── thin matrix → MMVF (mmvf.cu)
│   └── large matrix → cuBLAS HGEMM (cudarc cublas)
└── F32
    └── cuBLAS SGEMM (cudarc cublas)
```

### OpenCL Plan vs CUDA Graph

| | OpenCL Plan (현재) | CUDA Graph (향후) |
|---|---|---|
| **원리** | 커널 arg를 미리 바인딩, 변하는 arg만 갱신 | 커널 시퀀스를 그래프로 캡처, 단일 launch |
| **구현** | plan.rs (1769줄), 자체 구현 | CUDA 런타임 API, cudarc 지원 |
| **오버헤드 제거** | arg 바인딩 CPU 비용 | launch + sync CPU 비용 |
| **초기 구현** | 필수 (OpenCL launch 오버헤드 큼) | 선택 (CUDA launch 오버헤드 작음) |
| **동적 인자 갱신** | DynamicArg enum으로 수동 | cudaGraphExecKernelNodeSetParams |

CUDA는 커널 launch 오버헤드가 OpenCL보다 낮으므로, 초기 구현은 op-by-op으로 충분하다. 프로파일링 후 필요 시 CUDA Graph을 도입한다.
