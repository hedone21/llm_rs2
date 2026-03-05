# llm.rs Documentation Index

llm.rs (llm_rs2) 프로젝트의 기술 문서 인덱스입니다. 프로젝트 개요는 루트 `ARCHITECTURE.md`를, 빌드 및 기여 가이드는 `CLAUDE.md`를 참조하세요.

## 번호 체계

| 범위 | 영역 | 설명 |
|------|------|------|
| 00-13 | Core Engine | 빌드, 설계, 추상화, 백엔드, 모델, 추론 |
| 14-15 | Quality & Testing | 컴포넌트 품질 게이트, 테스트 전략 |
| 20-26 | Resilience | D-Bus IPC, 아키텍처, 통합, 테스트, 사용 가이드, API |

## 문서 목록

| # | 파일 | 주제 |
|---|------|------|
| 00 | [00_build_guide.md](00_build_guide.md) | 빌드 가이드 (Phase별 구현 순서) |
| 01 | [01_design_rationale.md](01_design_rationale.md) | 설계 결정 근거 (Rust, OpenCL, Q4_0 등) |
| 02 | [02_core_abstractions.md](02_core_abstractions.md) | Tensor, Buffer, Shape, DType, KVCache |
| 03 | [03_cpu_backend.md](03_cpu_backend.md) | CPU 백엔드 (scalar, NEON SIMD, AVX2) |
| 04 | [04_model_loading.md](04_model_loading.md) | Safetensors 로딩, HF 이름 매핑, Q4_0 |
| 05 | [05_tokenizer_and_sampling.md](05_tokenizer_and_sampling.md) | 토크나이저 통합, 샘플링 알고리즘 |
| 06 | [06_opencl_backend.md](06_opencl_backend.md) | OpenCL 백엔드 초기화, 커널 디스패치 |
| 07 | [07_kernel_implementation.md](07_kernel_implementation.md) | OpenCL 커널 알고리즘, Adreno 최적화 |
| 08 | [08_memory_management.md](08_memory_management.md) | 버퍼 타입, Zero-copy, 전송 패턴 |
| 09 | [09_attention_mechanism.md](09_attention_mechanism.md) | GPU attention 커널, GQA |
| 10 | [10_model_inference.md](10_model_inference.md) | Llama 3.2 config, forward pass, LayerWorkspace |
| 11 | [11_kv_cache_management.md](11_kv_cache_management.md) | KV 캐시 eviction 시스템 (Sliding Window, H2O) |
| 12 | [12_hybrid_inference.md](12_hybrid_inference.md) | CPU-GPU 동적 전환 전략 |
| 13 | [13_testing_and_benchmarks.md](13_testing_and_benchmarks.md) | Oracle 테스트, micro_bench, 프로파일링 |
| 14 | [14_component_status.md](14_component_status.md) | 컴포넌트 품질 게이트, 테스트 상태 |
| 15 | [15_test_strategy.md](15_test_strategy.md) | Resilience 테스트 전략 (T1-T4 계층) |
| 20 | [20_dbus_ipc_spec.md](20_dbus_ipc_spec.md) | D-Bus IPC 프로토콜 명세 |
| 21 | [21_resilience_architecture.md](21_resilience_architecture.md) | Resilience 아키텍처, Strategy 패턴 |
| 22 | [22_resilience_integration.md](22_resilience_integration.md) | generate.rs 통합 설계 |
| 23 | [23_resilience_test_strategy.md](23_resilience_test_strategy.md) | Resilience 통합 테스트 요약 |
| 24 | [24_resilience_usage_guide.md](24_resilience_usage_guide.md) | Resilience 시스템 사용 가이드 |
| 25 | [25_troubleshooting.md](25_troubleshooting.md) | 트러블슈팅 가이드 |
| 26 | [26_api_reference.md](26_api_reference.md) | Resilience API 레퍼런스 |

## 추천 읽기 순서

**처음 시작하는 경우**: 00 -> 01 -> 02 -> 03 -> 04 -> 10 -> 11

**기여자**: CLAUDE.md -> ARCHITECTURE.md -> 14 -> 13

**Resilience 기능 관련**: 20 -> 21 -> 22 -> 24 -> 25 -> 26

**특정 주제 심층 탐구**: OpenCL (06 -> 07 -> 08 -> 09), KV Cache (11), Hybrid (12)
