# llm.rs: On-device LLM Inference Framework

본 프로젝트는 ARM64 기반 엣지 디바이스 및 모바일 환경에 최적화된 고성능 **On-device LLM 추론 프레임워크**입니다. Rust 언어로 구현되었으며, 하드웨어 가속기 활용을 위한 유연한 백엔드 구조와 메모리 효율성을 극대화한 Zero-copy 아키텍처를 지향합니다.

## 🚀 Key Features

* **ARM64 Optimized**: Android 및 Linux 환경의 ARM64 SoC 성능을 최대로 활용하도록 설계되었습니다.
* **Zero-copy Memory Management**: `Galloc` 및 `SharedBuffer`를 통해 CPU와 GPU(OpenCL)/NPU 간의 불필요한 데이터 복사를 제거했습니다.
* **Backend Extensibility**: `Backend` 트레이트를 통해 CPU, OpenCL, NPU(TBD) 등 다양한 연산 엔진을 유연하게 교체 및 확장할 수 있습니다.
* **Quantization Support**: GGML(GGUF) 호환 `Q4_0`, `Q4_1` 블록 양자화 및 FP16/BF16 타입을 지원하여 메모리 대역폭 문제를 해결합니다.
* **Llama 3.2 Ready**: Llama 3.2 (1B) 아키텍처 및 GQA(Grouped-Query Attention)를 우선적으로 지원합니다.

## 📖 Documentation

시스템 아키텍처, 데이터 레이아웃, 인터페이스 정의 및 메모리 모델에 대한 상세한 내용은 아래 문서를 참조하십시오.

* **[ARCHITECTURE.md](ARCHITECTURE.md)** — 하이레벨/로레벨 컴포넌트 설계, Zero-copy 메커니즘, 양자화 포맷 및 Trait 정의
* **[docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md)** — 개발 치트시트 (빌드, 배포, 테스트 명령어)

### docs/ 상세 문서

| # | 문서 | 내용 |
|---|------|------|
| 00 | [Build Guide](docs/00_build_guide.md) | 프로젝트를 처음부터 구현하는 단계별 가이드 |
| 01 | [Design Rationale](docs/01_design_rationale.md) | 설계 결정의 근거 (Rust, OpenCL, Q4_0 등) |
| 02 | [Core Abstractions](docs/02_core_abstractions.md) | Tensor, Buffer, Shape, DType, KVCache 상세 |
| 03 | [CPU Backend](docs/03_cpu_backend.md) | CPU 스칼라 + NEON SIMD + AVX2 구현 |
| 04 | [Model Loading](docs/04_model_loading.md) | Safetensors 로딩, HF 이름 매핑, Q4_0 양자화 |
| 05 | [Tokenizer & Sampling](docs/05_tokenizer_and_sampling.md) | 토크나이저 통합 및 샘플링 알고리즘 |
| 06 | [OpenCL Backend](docs/06_opencl_backend.md) | OpenCL 백엔드 초기화, 커널 디스패치 |
| 07 | [Kernel Implementation](docs/07_kernel_implementation.md) | OpenCL 커널 알고리즘, Adreno 최적화 |
| 08 | [Memory Management](docs/08_memory_management.md) | 버퍼 타입, Zero-copy, 전송 패턴 |
| 09 | [Attention Mechanism](docs/09_attention_mechanism.md) | GPU 어텐션 커널, GQA, 성능 분석 |
| 10 | [Model Inference](docs/10_model_inference.md) | Llama 3.2 설정, Forward pass, LayerWorkspace |
| 11 | [KV Cache Management](docs/11_kv_cache_management.md) | KV 캐시 Eviction 시스템 설계 |
| 12 | [Hybrid Inference](docs/12_hybrid_inference.md) | CPU→GPU 동적 전환 전략 |
| 13 | [Testing & Benchmarks](docs/13_testing_and_benchmarks.md) | Oracle 테스트, micro_bench, 프로파일링 |
