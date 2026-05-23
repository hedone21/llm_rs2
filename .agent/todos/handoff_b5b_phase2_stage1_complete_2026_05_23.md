# Handoff: B-5b Phase 2 Stage 1 완료 → Stage 2 (호출지 치환 + S25 게이트) 다음 세션

**작성**: 2026-05-23
**HEAD**: `eb1970dc feat(backend): B-5b Phase 2 Stage 1 — Backend trait 4 capability hooks (infrastructure-only)`
**브랜치/Worktree**: `worktree-b5_trait_extension` (`.claude/worktrees/b5_trait_extension`, push 보류 — master FF 정리 중)
**다음 세션 진입 문장**: `"B-5b Phase 2 Stage 2 진행 (호출지 치환 14건 + S25 microbench 게이트)"`
**선행 진입점**: [[handoff_b5b_phase1_complete_2026_05_22]]

---

## TL;DR

B-5b Phase 2를 2 stage로 분할 진행. Stage 1 (Backend trait 4 capability method + struct field + SecondaryStore/CpuKernelSet trait/struct 신설 = **인프라 only**) 완료. 호출지는 미치환이라 baseline 변동 0. **멈춘 이유**: Stage 2가 호출지 14건 치환 + S25 Adreno microbench 게이트 + R-1 RPN 145~180 우려 대응 (사용자 결정 fallback) 포함된 큰 작업이라 컨텍스트 절약 차원에서 다음 세션으로.

**중요 발견 (architect 사전 조사 + Stage 1 진행 중)**:
1. 본 코드베이스 backend 보유 형태 = `Arc<dyn Backend>` ~90건 + `&dyn Backend` ~10건 / generic 0건. **LTO=fat가 vtable에 무력**.
2. **`cpu_companion`은 default impl 불가** (Rust object-safety 제약 — `&Self → &dyn Backend` 안 됨). 모든 backend가 explicit override.
3. R-1 RPN 144 → **145~180으로 악화 가능** (Stage 2 측정 결과 따라). 사용자 결정 C (Phase 0 그대로 진행) 적용.

---

## 진행 상태

### 본 sprint commits (2건 — Stage 1)

| HEAD | scope | 변경 |
|---|---|---|
| `3159fc08` | docs(skill) | review 스킬 v2 — 골격 재배치 + Context + ASCII UML + Premortem 제거 |
| `eb1970dc` | feat(backend) | **B-5b Phase 2 Stage 1** — Backend trait 4 capability hooks + CpuKernelSet + SecondaryStore + OpenClSecondary (placeholder) + 5 backend struct/impl 변경 + lib.rs 등록 |

### 게이트 결과 (Stage 1)

| 게이트 | 결과 |
|---|---|
| `cargo build -p llm_rs2` (default = opencl+profile) | PASS |
| `cargo fmt --all --check` | clean |
| `cargo clippy --lib --no-deps -- -D warnings` | clean 0 |
| `cargo test --lib -p llm_rs2` | non-GPU/non-flaky 회귀 0 (master 1201 → Stage 1 1202~1206, GPU 변동) |
| `cargo test --test spec inv_layer` | **8 PASS / 0 FAIL** |
| `python3 scripts/layer_lint.py --baseline` | **0 new violations** |
| **baseline JSON** | **206 유지 (변동 0)** — 호출지 미치환 |

### 파일 변경 (+318 LOC)

- **신규 모듈 2개**:
  - `engine/src/cpu_kernels.rs` (56 LOC) — `CpuKernelSet` struct + `CPU_KERNEL_SET` static
  - `engine/src/secondary.rs` (61 LOC) — `SecondaryStore` trait + `OpenClSecondary` trait (placeholder)
- **`engine/src/backend.rs`**: +84 LOC (line 1107~1191의 capability extension 블록)
- **5 backend impl**:
  - `engine/src/backend/cpu/neon.rs` (line 218~233) — `cpu_companion` + `cpu_kernels = Some(&CPU_KERNEL_SET)`
  - `engine/src/backend/cpu/{common,avx2}.rs` — `cpu_companion` only
  - `engine/src/backend/opencl/mod.rs` (line 546~552 field, 1537~1540 init, 6577~6598 override + empty `OpenClSecondary` impl)
  - `engine/src/backend/cuda_pc/mod.rs` (line 234~239 field, 322~324 init, 1806~1810 override)
  - `engine/src/backend/cuda_embedded/mod.rs` (line 352~356 field, 523~525 init, 3049~3053 override)
  - `engine/src/backend/qnn_oppkg/mod.rs` (+8 LOC — `cpu_companion = self`)
- **보조 mock 수정** (trait method 의무화로):
  - `engine/src/tensor.rs` 3개 inline mock impl
  - `engine/tests/spec/test_{inv_130,inv_131,inv_142,async_swap_executor}.rs` 4개 mock
- `engine/src/lib.rs` 신규 모듈 2개 등록

### baseline 변동 누적

| 단계 | 변동 | baseline |
|---|---|---|
| 진입 (Step 5b 종결) | — | 296 |
| B-1~B-5b Phase 1 (누적) | -90 | 206 |
| **B-5b Phase 2 Stage 1** | **0 (인프라만)** | **206** |
| Stage 2 예상 (호출지 치환 14건) | -14 (예상) | 192 |

---

## 다음 작업 — Stage 2 (senior-implementer 위임 권장)

### 호출지 치환 14건 (architect 사전 조사 + Stage 1 발견)

**1. `cpu_companion()` (Stage 2 진짜 ROI)**
- `backend/cuda_pc/mod.rs`: 13건의 `cpu_fallback` 진입점 (line 720, 739, 881, 885, 897, 1178, 1259, 1316, 1382, 1401, 1436, 1465)
- `backend/cuda_embedded/mod.rs`: 동일 패턴 ~10건
- 치환 패턴: `if let Some(cpu) = ...downcast_ref::<CpuBackend>()` → `let cpu = backend.cpu_companion();`

**2. `cpu_kernels()` (NEON fused-matmul 인디렉션 — R-1 핵심 위험원)**
- `backend/opencl/plan.rs:696, 717` — 직접 호출
- `layers/transformer_layer/forward_gen.rs:234, 264, 1286, 1307, 1436, 1461` — 직접 호출
- 시그니처: `unsafe fn(*const f32, usize, &[(*const T, *mut f32, usize)])` 그대로 인디렉션
- 치환 패턴: `crate::backend::cpu::neon::fused_matmul_f16(...)` → `(backend.cpu_kernels().expect("cpu_kernels required").fused_matmul_f16)(...)`

**3. `as_opencl_secondary()` (downcast 제거)**
- `backend/qnn_oppkg/mod.rs:132~142` — `with_opencl_secondary` 클로저가 `as_any().downcast_ref::<OpenCLBackend>()` 사용
- **결정 필요**: `OpenClSecondary` trait body 설계. 현재 placeholder. 후보:
  - (a) `fn with_queue<R>(&self, f: impl FnOnce(&Queue) -> R) -> R` (trait object 호환 위해 `Box<dyn FnOnce>` 필요)
  - (b) `&OpenCLBackend` 자체를 trait method로 expose하지 않고, qnn_oppkg가 별도 `Arc<OpenCLBackend>` field 보유로 우회 (downcast 제거되지만 trait 가치 ↓)

**4. `yield_after_layer()` (R-1 hot path)**
- `backend/opencl/plan.rs:1834` — `crate::resilience::gpu_yield::maybe_yield_after_layer(backend, i, true)`
- `models/transformer.rs:1841` — `crate::resilience::gpu_yield::maybe_yield_after_layer(&**backend, i, true)`
- 흡수: `resilience/gpu_yield.rs::maybe_yield_after_layer` (line 55) → `OpenCLBackend::yield_after_layer` override 본문
- 치환 패턴: 위 두 호출 → `backend.yield_after_layer(i, true)`
- 흡수 후 `resilience::gpu_yield` 모듈은 default 활용 시 삭제 가능 (또는 fast-path guard로 잔류)

### Stage 2 게이트 (필수)

1. `cargo build -p llm_rs2` PASS (default + opencl + cuda)
2. `cargo fmt + clippy` clean
3. `cargo test --lib` 회귀 0 + `spec inv_layer` 8 PASS
4. **`python3 scripts/layer_lint.py --baseline` → 206 → 192 (-14)** ← 핵심 게이트
5. **S25 Qwen 2.5 1.5B Q4_0 6T microbench TBT Δ ≤ +3%** (baseline 14.66 ms/tok) ← R-1 게이트
6. (선택) Jetson CUDA Q4_0 inference 1회 비교 (R-3 검증)

### S25 microbench 회귀 시 fallback (사용자 결정 필요)

R-1 RPN 145~180 가능성을 사용자가 명시적으로 수용 (결정 C). 회귀 시 다음 중 하나:

1. **§13.8-K 신설 + 매크로 우회** — `yield_after_layer`만 매크로로 freestanding 호출. arch §13.8-K 정책 신설 비용 있음
2. **`yield_after_layer`만 freestanding 유지** — Stage 2 호출지 치환을 `cpu_companion` + `cpu_kernels` + `as_opencl_secondary` 3개로 한정. baseline 효과 -14 → -12
3. **Stage 2 전체 rollback** — Phase 2 자체 종료. baseline 206 유지 (Stage 1 인프라만 잔류 — dead code)

### Stage 2 위임 prompt 초안 (senior-implementer)

```
## 본 작업 = B-5b Phase 2 Stage 2 (호출지 치환 + S25 microbench 게이트)

### 핵심 컨텍스트

- 선행: handoff_b5b_phase2_stage1_complete_2026_05_23.md (HEAD eb1970dc)
- Stage 1 인프라 완료 — Backend trait 4 capability method + 5 backend struct/impl 변경 + 신규 모듈 2개
- 본 Stage 2 = 호출지 14건 치환 + baseline 206→192 + S25 microbench 게이트

### 작업

1. cpu_companion 치환 (cuda_pc 13건 + cuda_embedded ~10건)
2. cpu_kernels 치환 (opencl/plan.rs 2건 + forward_gen.rs 6건)
3. as_opencl_secondary 치환 (qnn_oppkg/mod.rs 2건) — OpenClSecondary trait body 설계 결정 필요
4. yield_after_layer 치환 (opencl/plan.rs:1834 + transformer.rs:1841) + resilience::gpu_yield 흡수

### 게이트
1. cargo build + fmt + clippy + test --lib + spec inv_layer 8 PASS
2. baseline 206 → 192 (-14)
3. S25 Qwen 2.5 1.5B Q4_0 6T TBT Δ ≤ +3% (baseline 14.66 ms/tok)

### 회귀 시 fallback (사용자 결정)
1) §13.8-K 신설 + 매크로 (yield만 우회)
2) yield_after_layer만 freestanding 유지 (-12 baseline)
3) Stage 2 전체 rollback (인프라 dead code 잔류)

### 미해결 결정
- OpenClSecondary trait body 설계 (placeholder → 실체)
- SecondaryStore vs memory::secondary::SecondaryMmapBytes 통합 여부

### 게이트 통과 후
- commit (호출지 치환 + gpu_yield 흡수)
- handoff 작성 (Stage 2 종결 → Phase 3 (hybrid_attention 2건) 인계)
- task #53 completed
```

---

## Landmines / 미해결

### Stage 1 진행 중 발견 (5건)

1. **`cpu_companion` default impl 불가** (Rust object-safety): `fn cpu_companion(&self) -> &dyn Backend { self }` 형태로 default body를 trait 안에 두면 `&Self → &dyn Backend` coerce가 컴파일 안 됨. 모든 5 backend가 explicit override 강제. Phase 0 결정 (a)의 "default = self" 가정이 실제로는 깨짐 — 단 기능적으로는 동일.

2. **layer_lint이 doc 코멘트의 fully-qualified path를 import으로 오인**: 코멘트에 `crate::backend::cpu::neon::fused_matmul_*` 같은 경로를 적으면 INV-LAYER-001 위반으로 잡힘. Stage 2 호출지 치환 시 주석에 직접 경로 적지 말 것 (이미 1회 발생, 코멘트 표현 변경으로 해결).

3. **`CpuBackend::new()`는 `Self` 반환 (`Result` 아님)**: GPU backend init에서 `?` 안 됨. Stage 1 코드는 `Arc::new(CpuBackend::new())` (infallible 가정). CpuBackend는 OS 의존 없음이라 OK.

4. **테스트 결과 호스트 GPU 환경 의존**: 양 baseline (master, Stage 1) 모두 OpenCL device-required 테스트가 무작위 fail. 회귀 판정은 반드시 "GPU 테스트 제외 + flaky 테스트 제외" 후 비교.

5. **Stage 1 미push** (`worktree-b5_trait_extension` local commit `eb1970dc`만 존재): 사용자가 master worktree에 다른 세션 작업 진행 중. Stage 2 완료 후 일괄 push 또는 사용자 master FF 정리 후 push.

### Stage 2 미해결 결정 사항

1. **`OpenClSecondary` trait body 설계** — 현재 empty placeholder. Stage 2가 `with_opencl_secondary` 클로저를 어떻게 trait method로 분해할지 결정 필요. 가장 자연: `fn with_queue<R>(&self, f: Box<dyn FnOnce(&Queue) -> R>) -> R`. 또는 qnn_oppkg가 별도 `Arc<OpenCLBackend>` field로 우회 (downcast 제거되지만 trait 가치 ↓).

2. **`SecondaryStore` vs `memory::secondary::SecondaryMmapBytes` 통합** — Stage 1은 placeholder. `SecondaryMmapBytes`가 이미 존재한다는 점 발견. Stage 2가 통합 방향 결정 필요 (interface 일치 or `SecondaryStore` 삭제 후 `SecondaryMmapBytes` 직접 사용).

3. **`resilience::gpu_yield` 모듈 처분** — `yield_after_layer` trait method 흡수 후 본 모듈은 dead code. 삭제 vs fast-path guard 잔류 결정 필요.

### 사전 회귀 (Stage 1과 무관, B-2 handoff 이후 누적)

- `cargo build --features cuda` / `--features cuda-embedded`: `swap_dispatch.rs:441 map_weights_for_cpu` 사전 회귀. master에도 존재. 본 sprint와 무관.
- `crates/qnn_oppkg/src/interface.rs` unresolved imports 17건+ (commit d930801a 이후). 본 sprint와 무관.
- `cargo build --no-default-features` 사전 회귀 (B-2 handoff에서도 명시).

---

## 자기점검 (handoff-doc 스킬)

- [x] 진입 문장 한 줄로 다음 세션 첫 명령 가능: `"B-5b Phase 2 Stage 2 진행"`
- [x] 멈춘 이유 명시: Stage 2가 호출지 14건 + S25 microbench 게이트 + R-1 fallback 결정 포함 큰 작업
- [x] Landmines 표면화: 5건 (cpu_companion default 제약 + layer_lint 코멘트 오인 + Result/Self + GPU 테스트 변동 + 미push)
- [x] 검증 게이트 수치: baseline 206→192, S25 TBT Δ ≤ +3%, spec inv_layer 8 PASS
- [x] 본문 적정 길이 (외부 파일/링크 분리 OK)
