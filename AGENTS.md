# llm.rs 프로젝트 가이드

프로젝트의 AI 에이전트 작업 가이드.

## 시스템 지시

- **언어**: 모든 응답, 리포트, 계획, 설명은 한국어로 작성한다. 기술 용어와 코드 식별자는 원문 유지.

LLM의 흔한 코딩 실수를 줄이기 위한 행동 가이드라인. 프로젝트별 지시사항과 필요에 따라 병합한다.

**트레이드오프:** 이 가이드라인은 속도보다 신중함에 편향되어 있다. 사소한 작업에는 재량껏 판단한다.

### 1. 코딩 전 사고

**가정하지 말 것. 혼란을 숨기지 말 것. 트레이드오프를 표면화할 것.**

구현 전:
- 가정을 명시적으로 진술한다. 불확실하면 묻는다.
- 다중 해석이 가능하면 제시한다 — 침묵 속 결정 금지.
- 더 단순한 접근이 존재하면 그렇다고 말한다. 정당한 근거가 있으면 이의를 제기한다(push back).
- 불분명한 게 있으면 멈춘다. 무엇이 혼란스러운지 명명한다. 묻는다.

### 2. 단순함 우선

**문제를 해결하는 최소한의 코드. 추측성 코드 금지.**

- 요청된 것 이상의 기능 금지.
- 일회성 코드를 위한 추상화 금지.
- 요청되지 않은 "유연성"이나 "설정 가능성" 금지.
- 불가능한 시나리오에 대한 에러 핸들링 금지.
- 200줄로 쓴 것을 50줄로 쓸 수 있다면, 다시 쓴다.

자문하라: "시니어 엔지니어가 이걸 보고 과복잡하다고 할까?" 그렇다면 단순화한다.

### 3. 외과적 변경

**꼭 필요한 것만 건드린다. 자신이 만든 것만 정리한다.**

기존 코드를 편집할 때:
- 인접한 코드, 주석, 포매팅을 "개선"하지 않는다.
- 망가지지 않은 것을 리팩토링하지 않는다.
- 본인이라면 다르게 작성할지라도, 기존 스타일을 따른다.
- 무관한 dead code를 발견하면 언급만 한다 — 삭제하지 않는다.

본인의 변경이 orphan을 만들 때:
- 본인의 변경으로 인해 미사용이 된 import·변수·함수를 제거한다.
- 기존부터 있던 dead code는 요청받지 않는 한 제거하지 않는다.

검증 기준: 변경된 모든 라인은 사용자 요청으로 직접 추적 가능해야 한다.

### 4. 목표 기반 실행

**성공 기준을 정의한다. 검증될 때까지 반복한다.**

작업을 검증 가능한 목표로 변환한다:
- "validation 추가" → "invalid input에 대한 테스트를 작성한 뒤 통과시킨다"
- "버그 수정" → "버그를 재현하는 테스트를 작성한 뒤 통과시킨다"
- "X 리팩토링" → "리팩토링 전후로 테스트가 통과함을 보장한다"

다단계 작업의 경우 간단한 계획을 진술한다:
~~~
1. [단계] → 검증: [확인 방법]
2. [단계] → 검증: [확인 방법]
3. [단계] → 검증: [확인 방법]
~~~

강한 성공 기준은 독립적 반복을 가능하게 한다. 약한 기준("그냥 되게 하라")은 지속적인 명확화를 요구한다.

---

**이 가이드라인이 작동하고 있다는 신호:** diff에 불필요한 변경이 줄고, 과복잡화로 인한 재작성이 줄며, 명확화 질문이 실수 이후가 아닌 구현 이전에 나온다.

## 프로젝트 개요

llm.rs (repo: llm_rs2) — Rust로 작성된 고성능 온디바이스 LLM 추론 프레임워크. ARM64 Android/엣지 디바이스를 타겟으로 하며, HuggingFace Safetensors 포맷의 Llama 3.2 모델을 Q4_0/Q8_0 양자화 및 OpenCL GPU 가속으로 지원한다.

## 아키텍처

> **도메인 용어**: KV/weight 캐시 관리 핵심 용어(저장 형태=**Format**(noun) ⊥ 관리 동작=**Stage**(verb), handle 3종 등)는 [`CONTEXT.md`](CONTEXT.md) 참조. "Layer"는 transformer 디코더 블록 전용이며, 저장 형태는 "Layer"가 아니라 "Format"이라 부른다.

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
- **Adreno 디바이스 권장 backend = `opencl --opencl-rpcmem`** (Galaxy S25 검증, Sprint 2a 통합 완료 2026-05-26). KV cache + precision swap secondary store 가 rpcmem DMA-BUF heap + OpenCL `CL_MEM_USE_HOST_PTR` alias 로 zero-copy 화. `libcdsprpc.so` 만 필요 (Galaxy S25에 사전 배포). S25 Qwen2.5-1.5B Q4_0 Decode TBT 실측: `--backend opencl` ≈ `--backend opencl --opencl-rpcmem` ≈ 32 ms/tok (Sprint 2a-Gate 측정 2026-05-26, n=3 median). **`CL_MEM_ALLOC_HOST_PTR` (UnifiedBuffer / `--zero-copy` flag)는 Adreno에서 zero-copy 효과 없음** (driver internal pool alloc) — 진짜 zero-copy 원천은 DMA-BUF interop. **`--tokenizer-path` 명시 필수** — 자동 fallback이 sibling 모델의 tokenizer를 잡으면 vocab-size mismatch error로 차단 (commit 45cb489). **Deprecated**: `--backend qnn_oppkg` / `qnngpu` 는 Sprint 2b (2026-05-26) 에서 production 제거 (S25 실측에서 fast path 없이 OpenCL secondary 로 fallback 동작 중이라 가속 source 부재 확인). paper M3/M4/M5 microbench 는 보존 (`crates/qnn_oppkg/` + `engine/microbench/qnn_oppkg_*.rs`, `--features qnn` 빌드).

## QCF 명명 컨벤션 (2026-04-27 결정, rename 완료 — 2026-06-10 실측 확인)

QCF(Quality Cost Function)는 두 패밀리로 명확히 구분한다. 액션마다 측정 공간이 달라 단일 통일이 불가하므로, **이름으로 패밀리를 명시**한다.

| 패밀리 | 측정 공간 | 포함 액션 | 코드 위치 (현재 명) |
|---|---|---|---|
| **QCF_kv** | KV 캐시 → attention output `‖ΔO‖₂ / ‖O‖₂` | sliding/H2O/streaming eviction, KIVI quant, D2O merge | `engine/src/qcf/qcf_kv.rs::compute_qcf_kv` (구 unified_qcf.rs) |
| **QCF_weight** | 모델 forward path (weight/layer 단위) | weight swap (F16→Q4), layer skip (SWIFT) | `engine/src/pressure/weights/decider.rs` (swap 판단), `engine/src/qcf/layer_importance.rs::compute_qcf_weight` |

- **Layer skip은 QCF_weight에 속한다** — skip은 "그 layer의 weight를 쓰지 않음"이라 weight 패밀리. swap과 ImportanceTable을 공유한다.
- **두 패밀리는 직접 비교 불가** — 측정 공간이 다르다. cross-action 비교는 `DegradationEstimator`로 ΔPPL 환산 후 가능 (estimator에 액션별 piecewise-linear 곡선 등록 필요).
- **IPC 메시지** (`shared/src/lib.rs::QcfEstimate`)는 이미 두 필드(`estimates` HashMap + `layer_swap`)로 분리되어 있어 패밀리 구분이 IPC 단계에 자연스럽게 반영되어 있다.
- **rename 완료**: 코드 rename(`qcf_kv.rs` 등, 2026-05-19/21) + estimator key 정규화(`kv.*`/`weight.*` — `estimator.rs::with_defaults` 실측 확인 2026-06-10) + 문서 갱신 전부 적용. backlog "QCF 명명 컨벤션 정리" 항목 RESOLVED 종결.
- **상세 정의/수식**: `docs/qcf_taxonomy.md` (논문 참조용). 두 패밀리 정의, 7개 KV 액션 + swap + skip 수식, 코드 위치 인덱스 수록.

## 커밋 컨벤션

Conventional Commits: `type(scope): subject` — 명령형 현재 시제. Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert.

## 코드 스타일 컨벤션

- **모듈 파일 스타일 = no-`mod.rs`** (Rust 모던 path 스타일, 2026-06-02 결정). 디렉토리 모듈의 루트는 `foo/mod.rs` 가 아니라 형제 `foo.rs` 다 (`foo.rs` + `foo/` 안에 서브모듈). 두 스타일 다 Rust 2024 에서 유효하나, no-`mod.rs` 가 권장 모던 관용구(`cargo new` 기본)이고 이미 top-level (`backend.rs`/`buffer.rs`/`memory.rs`/`quant.rs`)이 채택한 패턴이라 프로젝트 전역 컨벤션으로 채택한다. **신규·이동 모듈은 반드시 이 스타일.** 기존 nested `mod.rs` 38개 일괄 sweep 은 완료 (commit `3895e17d`, 2026-06-02 — engine/src 전수 `git mv`; 잔여 mod.rs 11개는 manager/src 8 + manager/tests 1 + crates/qnn_oppkg 2 로 원 census 범위 밖, 별도 처분). 대응 설계: `arch/pipeline_stage_design_v2.md` §2.1 규칙 C.

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
| **handoff-doc** | 세션·단계 종료 시 handoff 문서 작성 (R1~R6 6요소 + 자기점검) | 모두 (메인 세션, PM, Architect, Implementer) |
| **review** | 구현 전 Plan/Action/Decision/Design 사전 리뷰 (10섹션 골격: Reviewed Items / Alternatives ≥2 + status quo / Risks(RPN≥100) / Premortem / Devil's Advocate) | 메인 세션 |

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
| Resilience 검증 하네스 | `verify/README.md`, `verify/USAGE.md`, `arch/verify_v2.md` |
| TODO 관리 | `.agent/todos/` — 형식: `.agent/todos/README.md` |
| 설계 문서 | `ARCHITECTURE.md`, `spec/`, `docs/` |
| 도메인 용어집 | `CONTEXT.md` — 저장 형태=Format(noun) ⊥ 관리 동작=Stage(verb), "Layer"는 transformer layer 전용 |

## Weight asset 경로 정책 (Sprint 1 W-AUF-1, 2026-05-20)

- **정식**: `--model-path foo.auf` (AUF single-file). multi-dtype/variant capability bit ON이면 W-AUF-2에서 self-secondary 자동 활성 예정.
- **CLI flags (AUF primary)**: `--primary-variant {auto,adreno-soa,cpu-aos,cuda-aos}`, `--primary-dtype {auto,f16,q4_0,q8_0,bf16,f32,q4_1}`, `--eos-token-id <ID>` / `--bos-token-id <ID>` (TOKENIZER section eos/bos가 N/A일 때 fallback), `--no-self-secondary` (W-AUF-2 자동 활성 끄기).
- **AUF 빌드 (`auf_tool build`)**: `--tokenizer-config tokenizer_config.json` + `--bos-token-id` / `--eos-token-id`로 TOKENIZER section 슬롯을 채운다. 미지정 시 sibling `tokenizer_config.json` 자동 탐색.
- **Deprecated**: `--secondary-gguf <PATH>`는 stderr 1회 경고 후 그대로 동작 (`.gguf`/`.auf` 둘 다 수용). 향후 제거 예정 — 기존 호출은 AUF single-file로 마이그레이션 권장.

## Resilience 검증 하네스 요약

`verify/` — SystemSignal → Policy → EngineCommand → Engine 전 경로를 실제 바이너리로 돌려 4층(functional / crash+progress / performance / accuracy) 판정하는 자동화 매트릭스. 엔트리는 `python verify/verify.py --device <key> --model <f16,q4> [--scenario-filter ...]`. 시나리오 YAML은 `verify/scenarios/`. 사용법 상세: `verify/USAGE.md`.
