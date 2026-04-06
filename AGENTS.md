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

**엔진 모듈 구조** (`engine/src/lib.rs`):
- `core/` — 트레이트와 추상화: `Backend` (17+ ops), `Buffer`, `Tensor`, `KVCache`, eviction 정책
- `backend/cpu/` — CPU 백엔드, ARM64 NEON (`neon.rs`) 및 x86 AVX2 (`x86.rs`) 특화
- `backend/opencl/` — OpenCL GPU 백엔드; 커널 파일: `engine/kernels/*.cl` (~80개)
- `models/llama/` — Llama 3.2 모델 로딩 및 forward pass
- `layers/` — Transformer 레이어, attention (naive + flash), 사전 할당 workspace 버퍼
- `memory/` — Galloc 공유 할당자
- `buffer/` — SharedBuffer (zero-copy GPU↔CPU) 및 UnifiedBuffer
- `resilience/` — Resilience 매니저 (D-Bus/UnixSocket IPC, strategy 패턴)

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

- **`.cl` 커널 파일을 수정하지 않는다** — 명시적 지시가 없는 한. 고도로 최적화되어 있고 안정적이다.
- **`--gpu-attn` 플래그를 사용하지 않는다** — 명시적 지시가 없는 한.
- `opencl` feature는 기본 활성화. GPU 없는 호스트에서도 컴파일은 되지만 GPU 연산은 실행되지 않는다.
- Release 프로필: `lto = "fat"`, `codegen-units = 1`, `opt-level = 3`.
- Android 타겟: NEON+dotprod 필수; x86 타겟: AVX2+FMA 활성화 (`.cargo/config.toml`에 설정).
- Android 크로스 컴파일 시 `cargo build` 전에 반드시 `source android.source` 실행.

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

| 작업 | 스킬/참조 |
|------|----------|
| 빌드, 린트, 유닛 테스트 | `/sanity-check` |
| Android 배포, 디바이스 테스트 | `/deploy-test` |
| 프로파일링 | `/profile` |
| 대시보드 | `/dashboard` |
| Spec 관리 | `/spec-manage` |
| 실험 벤치마크 | `experiments/PLAN.md`, `docs/35_experiment_runner_guide.md` |
| TODO 관리 | `.agent/todos/` — 형식: `.agent/todos/README.md` |
| 설계 문서 | `ARCHITECTURE.md`, `spec/`, `docs/` |
