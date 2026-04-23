# llm.rs 프로젝트 가이드

프로젝트의 AI 에이전트 작업 가이드.

## 시스템 지시

- **언어**: 모든 응답, 리포트, 계획, 설명은 한국어로 작성한다. 기술 용어와 코드 식별자는 원문 유지.

## 프로젝트 개요

llm.rs (repo: llm_rs2) — Rust로 작성된 고성능 온디바이스 LLM 추론 프레임워크. ARM64 Android/엣지 디바이스를 타겟으로 하며, HuggingFace Safetensors 포맷의 Llama 3.2 모델을 Q4_0/Q8_0 양자화 및 OpenCL GPU 가속으로 지원한다.

## 아키텍처

**Cargo workspace** — 3개 Rust 크레이트:
- `engine/` — LLM 추론 엔진 (`llm_rs2` 크레이트)
- `shared/` — 공유 시그널 타입 (`llm_shared` 크레이트)
- `manager/` — 시스템 리소스 매니저 서비스 (`llm_manager` 크레이트)

**비-Rust 컴포넌트**:
- `web_dashboard/` — 웹 대시보드 (Python/Flask)

**엔진 모듈 구조** (`engine/src/`): 디렉토리 목록은 `ls`로 확인 가능 — 여기서는 비자명한 관계만 기재.
- `core/`가 백엔드 추상화 + KV cache/eviction/pressure pipeline/QCF를 모두 소유 (백엔드·모델·레이어가 모두 의존).
- `backend/opencl/`은 `engine/kernels/*.cl`을 런타임 로드 (컴파일 타임 링크 아님).
- `resilience/`는 `manager` 크레이트와 D-Bus/UnixSocket IPC로 통신 (별도 프로세스).

**주요 바이너리**:
- `generate` (`engine/src/bin/`) — 메인 추론 바이너리 (단일 백엔드, CPU 또는 OpenCL)
- `test_backend` (`engine/src/bin/`) — 백엔드 정확성 검증 (CPU vs OpenCL 비교, Tier 2 테스트)
- `llm_manager` (`manager/src/main.rs`) — 시스템 리소스 매니저 서비스 (PI 컨트롤러, 정책 엔진)
- `mock_engine` (`manager/src/bin/`) — 매니저 테스트용 엔진 모의 클라이언트
- `mock_manager` (`manager/src/bin/`) — 엔진 테스트용 매니저 모의 서버

**추론 흐름**: Prefill (배치 토큰) → Decode (토큰 단위). 레이어별: RMSNorm → QKV matmul → RoPE → KV 캐시 갱신 → Attention → FFN. 모델은 `forward_into()`로 통합 forward pass 수행; eviction은 `CacheManager`를 통해 호출자 책임. `LlamaLayer::forward()`는 `seq_len == 1`일 때 private `forward_gen()` 경로로 분기.

**Zero-copy 메모리**: ARM SoC에서 `CL_MEM_ALLOC_HOST_PTR`이 GPU 버퍼를 CPU 포인터로 매핑하여 CPU↔GPU 간 memcpy를 제거한다.

**KV 캐시 eviction**: `EvictionPolicy` 트레이트 — `NoEvictionPolicy`, `SlidingWindowPolicy` (최근 N 토큰 유지), `H2OPolicy` (3-파티션: prefix + heavy hitters + recent window), `H2OPlusPolicy` (per-head GQA-aware 변형). `D2OHandler` (merge compensation, `CachePressureHandler` 기반). RoPE 포지션은 eviction 후에도 단조 증가; 물리적 KV 캐시 위치는 `prune_prefix()`로 감소 가능.

**Tensor Partition**: `--tensor-partition <ratio>` (0.0~1.0)로 FFN gate/up matmul을 GPU와 CPU에서 동시 분할 실행. 설계 문서: `arch/tensor_partition.md`. Decode-only (seq_len=1). 자동으로 `--zero-copy` + `rewrap_weights_for_dual_access()` 활성화. 분할 단위: out_dim 축(행), Q4_0 128 배수 정렬. 동기화: `synchronize()` → `read_buffer(residual→CPU)` → GPU/CPU 병렬 matmul → `copy_slice` merge. ARM UMA 캐시 비일관성으로 `read_buffer`/`write_buffer` 필수 (as_ptr 직접 접근 시 stale cache). CUDA는 CudaHostBuffer(pinned)로 동일 코드 동작.

## 핵심 제약사항

- **`.cl` 커널 수정 정책**: 기본적으로 수정을 피하되, 성능 최적화 작업(Senior Implementer)에서는 허용. Adreno 실측 교훈 준수 필수 — (1) DK=128 flash attn은 per-thread 32 float4 초과 시 register spill, (2) `sub_group_reduce_*`는 SLM tree-reduce 대비 33~55% 느림 (Adreno 830 실측), (3) `CL_QUEUE_PROFILING_ENABLE`은 driver-specific 패널티를 가져 엔진 간 비교는 항상 wall-clock 사용.
- `opencl` feature는 기본 활성화. GPU 없는 호스트에서도 컴파일은 되지만 GPU 연산은 실행되지 않는다.
- Release 프로필: `lto = "fat"`, `codegen-units = 1`, `opt-level = 3`.
- Android 타겟: NEON+dotprod 필수; x86 타겟: AVX2+FMA 활성화 (`.cargo/config.toml`에 설정).
- Android 크로스 컴파일: `run_device.py`가 `hosts.toml` 기반으로 NDK env를 자동 주입한다. cargo 직접 호출 시에만 `source android.source`가 필요 (비권장).
- **Android 벤치 스레드**: Galaxy S25는 6T만 사용. 8T는 llm.rs/llama.cpp 양쪽 모두 역효과.
- **테스트 기본 모델 포맷: GGUF**. 모든 추론·디바이스·벤치 테스트는 GGUF(`.gguf`)를 기본으로 사용한다. Safetensors는 (1) GGUF가 준비되지 않은 모델, (2) GGUF/Safetensors 결과 비교가 목적인 테스트에 한해 사용한다. `--model-path`에 `.gguf` 파일을 직접 지정 (generate.rs:779에서 확장자로 자동 분기).
- **성능 측정은 `--profile` 없이** — `--profile`은 `plan.rs`를 비활성화하고 매 op마다 `backend.synchronize()`를 2회 호출하여 ~54 ms/token의 sync 오버헤드를 더한다. Production TBT는 `Decode: X ms/tok` 로그 라인 또는 manager heartbeat의 `actual_throughput`을 사용한다. `--profile`의 per-op breakdown은 **상대 비교**에만 유효하다 (어느 op이 상대적으로 큰가). 절대값은 sync 오버헤드로 부풀려져 있다.

## 커밋 컨벤션

Conventional Commits: `type(scope): subject` — 명령형 현재 시제. Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert.

## 에이전트 시스템

6개 특화 서브에이전트가 `.claude/agents/`에 정의되어 있다. 메인 세션이 오케스트레이터 역할을 하며 에이전트 간 결과를 전달한다.

| 에이전트 | 모델 | 역할 | 범위 |
|---------|------|------|------|
| **PM** | opus | 계획 수립, TODO 관리, 우선순위 조정 | `.agent/todos/*.md`만 수정 |
| **Architect** | opus | 코드 분석, SOLID 설계, Spec/Arch 문서 관리 | `spec/`, `arch/`, `docs/*.md` |
| **Senior Implementer** | opus | GPU 커널(.cl), NEON/SIMD, 성능 최적화, 복잡 알고리즘 | `kernels/`, `backend/opencl/`, `neon.rs`, `kivi_cache.rs`, `qcf/` |
| **Implementer** | sonnet | 일반 Rust 구현, 프로토콜 연결, CLI, Manager, 테스트 | `engine/`, `shared/`, `manager/` (GPU/SIMD 제외) |
| **Tester** | opus | 호스트/디바이스 테스트, 결과 분석, 품질 게이트 | 수정 불가, 실행만 |
| **Researcher** | opus | 논문 분석, 기술 조사, 적용 가능성 평가 | 수정 불가, 조사만 |

**워크플로우**:
```
[PM] 계획/TODO → [Architect] 설계+Spec → [Senior Impl / Impl] 구현+테스트 → [Tester] 검증
                                            ↑
                                    [Researcher] 기법 조사
```

**제약**: 서브에이전트는 다른 서브에이전트를 호출할 수 없다 (최대 1단계). 작업 위임은 메인 세션이 담당.

## 스킬 시스템

`.claude/skills/`에 에이전트별 특화 스킬이 정의되어 있다. 상세 절차와 명령어는 각 스킬 참조.

| 스킬 | 용도 | 주 사용 에이전트 |
|------|------|----------------|
| **sanity-check** | 빌드 + cargo fmt + clippy + test | Implementer |
| **deploy-test** | Android 빌드→배포→테스트 + 디바이스 관리 | Tester |
| **profile** | 온디바이스 프로파일링 + 시각화 | Tester, Implementer |
| **dashboard** | 웹 대시보드 실행/관리 | (공용) |
| **design-review** | SOLID 원칙 기반 코드 구조 검토 | Architect |
| **research** | 논문/기술 조사 + 적용 가능성 평가 | Researcher |
| **spec-manage** | Spec/Arch/Test 3계층 문서 관리 (ID 할당, 동기화) | Architect |
| **develop** | 개발 파이프라인 오케스트레이터 (워크플로우 조율) | 메인 세션 (오케스트레이터) |

## 워크플로우 규칙

- **완료 시 자동 커밋**: Implementer가 작업을 완료하면 자동으로 커밋한다. 미커밋 작업을 남기지 않는다.
- **완료 시 데스크톱 알림**: 작업 완료 후 `notify-send "llm.rs" "<작업 요약>"`으로 알림을 보낸다.

## 빠른 참조

**호스트 빌드/테스트** (스킬 미사용 시):
```bash
cargo build --release -p llm_rs2
cargo test --workspace
cargo fmt --all && cargo clippy --workspace -- -D warnings
```

**Android 크로스 빌드** (run_device.py 경유 권장, NDK env 자동 주입):
```bash
# 최초 1회: hosts.toml 생성
python scripts/device_registry.py bootstrap-host

# 빌드 + 배포
python scripts/run_device.py -d pixel generate

# cargo 직접 호출 시 (비권장, env 수동 설정 필요):
# source android.source && cargo build --release --target aarch64-linux-android -p llm_rs2
```

| 작업 | 스킬/참조 |
|------|----------|
| 빌드, 린트, 유닛 테스트 | `/sanity-check` |
| Android 배포, 디바이스 테스트 | `/deploy-test` |
| 프로파일링 | `/profile` |
| 대시보드 | `/dashboard` |
| Spec 관리 | `/spec-manage` |
| 실험 벤치마크 | `experiments/PLAN.md`, `docs/35_experiment_runner_guide.md` |
| Resilience 검증 하네스 | `resilience_verify/README.md`, `resilience_verify/USAGE.md`, `arch/resilience_verify_v2.md` |
| TODO 관리 | `.agent/todos/` — 형식: `.agent/todos/README.md` |
| 설계 문서 | `ARCHITECTURE.md`, `spec/`, `docs/` |

## Resilience 검증 하네스 요약

`resilience_verify/` — SystemSignal → Policy → EngineCommand → Engine 전 경로를 실제 바이너리로 돌려 4층(functional / crash+progress / performance / accuracy) 판정하는 자동화 매트릭스. 엔트리는 `python resilience_verify/verify.py --device <key> --model <f16,q4> [--scenario-filter ...]`. 시나리오 YAML은 `resilience_verify/scenarios/`. 사용법 상세: `resilience_verify/USAGE.md`.
