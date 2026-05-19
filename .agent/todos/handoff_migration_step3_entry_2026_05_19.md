# Handoff — Migration Step 3 (L1/L2 경계 정리) 진입점 (2026-05-19)

**작성**: 2026-05-19 (Phase 4 종결 직후)
**master HEAD**: `5cc0d87d` (예정: Phase 4 종결 commit 후 갱신)
**대응 단계**: `ARCHITECTURE.md §13.7 Migration Step 3`
**진입 문장**: **"Migration Step 3 진행"** 또는 sub-sprint 단위 **"Step 3-A 진행"**

---

## 1. Step 3 목표

`ARCHITECTURE.md §13.7 Step 3`:

> **L1/L2 경계 정리** — backend impl이 `shared/` 외 import 제거, backend-specific buffer/pool/포맷 재배치

### 해소 대상 violation (실측 31건 중 6건)

| # | 위반 | 해소 sprint |
|---|---|---|
| V-07 | `buffer/host_ptr_pool_buffer.rs` → `backend::opencl::host_ptr_pool` | 3-D |
| V-08 | backend-specific buffer가 `buffer/`에 위치 (`cl_*`, `cuda_*`, `rpcmem_*`) | 3-D |
| V-09 | `buffer/*` → `models::weights::SecondaryMmap` | 3-E (보조) |
| V-19 | `layers/tensor_partition.rs` → `buffer::cl_sub_buffer::ClSubBuffer` | 3-D + V-19 잔여 |
| V-23 | AUF가 `models/weights/secondary_mmap.rs` 등 일반 로딩에 사용 | 3-A |
| V-27 | `models/weights/layer_object_pool.rs` → `CudaBackend` downcast | 3-B |

**목표**: baseline 31건 → **25건** (6건 해소).

---

## 2. 결정 사항 (§13.8 RESOLVED)

본 Step 3 진입 시 모든 결정이 이미 완료되어 있다. 추가 결정 불요.

### §13.8-A — AUF 위치 (3-A)

- **결정**: `auf/` → **`shared/auf/`**
- **이동 대상**:
  - `engine/src/auf/reader.rs`, `header.rs`, `section.rs`, `tensor_index.rs`, `view.rs`, `error.rs` 등 전체
- **import 변경 측**:
  - `models/weights/secondary_mmap.rs:26+`
  - `models/transformer.rs:644,3556,3586,3615`
  - `buffer/borrowed_mmap_buffer.rs:120,216`
  - `models/weights/rpcmem_secondary.rs:46+`
  - `bin/generate.rs` (V-30 일부)

### §13.8-B — backend-aware pool 위치 (3-B)

- **결정**:
  - `models/weights/layer_object_pool.rs` (CUDA pool) → **`backend/cuda_embedded/pool.rs`** 또는 **`backend/cuda_pc/pool.rs`** (양 backend 공유 여부 실측 후)
  - `backend/opencl/host_ptr_pool.rs` 위치 유지
  - **`WeightStagingPool` trait를 `shared/`에 신설** — pressure handler가 의존 역전으로 접근
- **import 변경 측**:
  - `core/pressure/weight_swap_handler.rs:21+` (LayerSlot, SwapExecutor 의존 라인)
  - `models/weights/swap_executor.rs:2139+` (HostPtrPool downcast)

### §13.8-D — backend-specific buffer 위치 (3-D)

- **결정**: `cl_*`, `cuda_*`, `rpcmem_*` 접두어를 가진 모든 buffer → **`backend/<be>/buffer/`**
- **이동 매핑**:
  ```
  buffer/cl_sub_buffer.rs           → backend/opencl/buffer/cl_sub_buffer.rs
  buffer/cl_wrapped_buffer.rs       → backend/opencl/buffer/cl_wrapped_buffer.rs
  buffer/host_ptr_pool_buffer.rs    → backend/opencl/buffer/host_ptr_pool_buffer.rs
  buffer/cuda_buffer.rs             → backend/cuda_embedded/buffer/ 또는 backend/cuda_pc/buffer/
                                       (공유 시 backend/cuda_common/buffer/ 신설 후보)
  buffer/cuda_mmap_alias_buffer.rs  → (동일)
  buffer/rpcmem_alias_buffer.rs     → backend/qnn_oppkg/buffer/rpcmem_alias_buffer.rs
  ```
- **잔존 `shared/buffer/`** (generic만):
  - `shared_buffer.rs`, `slice_buffer.rs`, `mmap_buffer.rs`, `unified_buffer.rs`, `borrowed_mmap_buffer.rs`

### V-09 보조 작업 (3-E)

- **`SecondaryMmap`을 L2의 trait로 추상화** (`SecondaryStore` 등)
- buffer는 trait만 import, 구현은 `pressure/` (또는 현 `models/weights/`) 잔존
- 영향 import:
  - `buffer/cuda_mmap_alias_buffer.rs:22`
  - `buffer/rpcmem_alias_buffer.rs:25`
  - `buffer/host_ptr_pool_buffer.rs:27`
  - `buffer/borrowed_mmap_buffer.rs:19`

---

## 3. Sub-sprint 분해

각 sub-sprint = 단일 worktree + spec test 회귀 0 + (필요 시) bit-identical 32 tok + ff-merge.

### 3-A — `auf/` → `shared/auf/` (V-23 해소)

| 항목 | 값 |
|---|---|
| 영향 LOC | ~100 (import path 변경) |
| 위험 | **LOW** — 순수 file rename + use path 갱신 |
| Dependencies | 없음 |
| 게이트 | spec test 회귀 0, `cargo build --release` 회귀 0 |
| 진입 문장 | "Step 3-A 진행" |

**순서**:
1. `git mv engine/src/auf/ shared/src/auf/`
2. `engine/src/lib.rs`에서 `pub mod auf;` 제거
3. `shared/src/lib.rs`에 `pub mod auf;` 추가
4. 호출처 (`engine/src/{models/transformer,models/weights/secondary_mmap,buffer/borrowed_mmap_buffer,models/weights/rpcmem_secondary,bin/generate}`) 의 `use crate::auf::` → `use llm_shared::auf::`
5. baseline JSON 업데이트 (V-23 제거)
6. spec test `test_inv_layer_002` 검증 (L2→L3 역의존 감소)

### 3-D — backend-specific buffer 이동 (V-07/V-08/V-19 해소)

| 항목 | 값 |
|---|---|
| 영향 LOC | ~300 (file rename + 광범위 import) |
| 위험 | **M** — 두 CUDA backend 공유 여부 실측 결정 필요 |
| Dependencies | 없음 (3-A와 병행 가능) |
| 게이트 | spec test 회귀 0, host CPU + S25 bit-identical 32 tok |
| 진입 문장 | "Step 3-D 진행" |

**선결 결정**: CUDA buffer를 `cuda_embedded`와 `cuda_pc`가 공유하는지 실측 (grep `use crate::buffer::cuda_*`). 공유 시 `backend/cuda_common/buffer/` 신설.

**순서** (OpenCL buffer 기준):
1. `git mv engine/src/buffer/cl_sub_buffer.rs engine/src/backend/opencl/buffer/`
2. `git mv engine/src/buffer/cl_wrapped_buffer.rs engine/src/backend/opencl/buffer/`
3. `git mv engine/src/buffer/host_ptr_pool_buffer.rs engine/src/backend/opencl/buffer/`
4. `engine/src/buffer/mod.rs`에서 export 제거
5. `engine/src/backend/opencl/mod.rs` 또는 `buffer/mod.rs` 신설 + re-export
6. 호출처 `use crate::buffer::cl_*` → `use crate::backend::opencl::buffer::*`
7. (CUDA / rpcmem 동일 패턴)
8. baseline JSON 업데이트 (V-07/V-08/V-19 일부 제거)

### 3-B — backend-aware pool 이동 (V-27 해소)

| 항목 | 값 |
|---|---|
| 영향 LOC | ~200 (LayerObjectPool 이동 + trait 신설) |
| 위험 | **M** — `WeightStagingPool` trait 설계 + `CudaBackend` downcast 제거 |
| Dependencies | 3-D 권장 (CUDA buffer 이동 후) |
| 게이트 | spec test 회귀 0, S25 weight swap 정확성 (top-5 overlap > 99%) |
| 진입 문장 | "Step 3-B 진행" |

**순서**:
1. `shared/src/lib.rs`에 `WeightStagingPool` trait 신설:
   ```rust
   pub trait WeightStagingPool: Send + Sync {
       fn acquire(&self, size: usize) -> Result<StagingHandle>;
       fn release(&self, handle: StagingHandle);
   }
   ```
2. `engine/src/models/weights/layer_object_pool.rs` → `engine/src/backend/cuda_embedded/pool.rs` (또는 `cuda_pc/pool.rs`)
3. `CudaBackend`가 `WeightStagingPool` trait impl
4. `core/pressure/weight_swap_handler.rs`가 `Arc<dyn WeightStagingPool>` 받음 (현 `Arc<LayerObjectPool>` 대체)
5. `models/weights/swap_executor.rs:2139` downcast 제거
6. baseline JSON 업데이트 (V-27 제거)

### 3-E — `SecondaryMmap` trait inversion (V-09 보조)

| 항목 | 값 |
|---|---|
| 영향 LOC | ~150 (trait + 4 buffer 모듈 import 변경) |
| 위험 | **M** — L3 Pressure state ↔ L2 buffer 의존 끊기 |
| Dependencies | 3-D 권장 (buffer 이동 후) |
| 게이트 | spec test, weight swap 정확성 |
| 진입 문장 | "Step 3-E 진행" |

**순서**:
1. `shared/src/lib.rs`에 `SecondaryStore` trait 신설:
   ```rust
   pub trait SecondaryStore: Send + Sync {
       fn map_range(&self, offset: usize, size: usize) -> Result<&[u8]>;
       fn layer_range(&self, layer_id: usize, sublayer: SubLayer) -> Option<(usize, usize)>;
   }
   ```
2. `models/weights/secondary_mmap.rs::SecondaryMmap`이 `SecondaryStore` impl
3. 4 buffer 모듈 (`cuda_mmap_alias_buffer`, `rpcmem_alias_buffer`, `host_ptr_pool_buffer`, `borrowed_mmap_buffer`)이 `&dyn SecondaryStore` 받음
4. baseline JSON 업데이트 (V-09 제거)

### 3-F — baseline JSON 갱신 + spec test 검증 (정리)

| 항목 | 값 |
|---|---|
| 영향 LOC | ~50 (JSON + test 갱신) |
| 위험 | **LOW** |
| Dependencies | 3-A, 3-B, 3-D, 3-E 완료 후 |
| 게이트 | `engine/tests/spec/test_inv_layer_*` PASS |
| 진입 문장 | "Step 3-F 진행" |

**확인 사항**:
- baseline 31 → 25 (6건 감소)
- `test_inv_layer_001` (L1→상위 import 금지): V-07 제거 확인
- `test_inv_layer_002` (L2→L3 역의존 금지): V-09, V-23 제거 확인
- `test_inv_layer_003` (L3→concrete backend impl 금지): V-19, V-27 제거 확인

---

## 4. 의존성 순서

```
       3-A (auf 이동, V-23)
        │
        │  (병행 가능)
        ▼
       3-D (buffer 이동, V-07/V-08/V-19)
        │
        ├─────► 3-B (pool 이동, V-27)
        │
        └─────► 3-E (SecondaryStore trait, V-09)
                 │
                 ▼
                3-F (baseline 갱신 + spec test)
```

**병행 가능**: 3-A는 3-D와 독립. 3-B와 3-E는 3-D 완료 후 병행 가능.

---

## 5. 위험 매트릭스

| R | 항목 | 영향 | 완화 |
|---|---|---|---|
| R1 | cross-crate refactoring (`engine` ↔ `shared`) | M | 3-A부터 시작 (auf 이동만), `Cargo.toml` dependency 추가 검증 |
| R2 | 두 CUDA backend 공유 buffer 결정 | M | 3-D 시작 시 grep 실측, 공유 시 `backend/cuda_common/buffer/` 신설 |
| R3 | `WeightStagingPool` trait 시그니처 | M | LayerObjectPool 기존 API 그대로 모사, 사후 정제 |
| R4 | `SecondaryStore` trait 시그니처 | M | `SecondaryMmap` 기존 public method 그대로 모사 |
| R5 | 31 violation 중 6건 해소 후 spec test 갱신 누락 | L | 3-F에서 baseline JSON diff 검증 |
| R6 | bin/generate.rs의 V-30 일부 (auf import) 변경 | L | 3-A 안에 포함 — mechanical |

---

## 6. 검증 인프라

### 게이트

| 게이트 | 적용 sub-sprint | 기준 |
|---|---|---|
| `cargo build --release -p llm_rs2 -p llm_shared -p llm_manager` | 전부 | 회귀 0 |
| `cargo test --lib -p llm_rs2 -p llm_shared` | 전부 | 회귀 0 |
| `cargo test --test spec -p llm_rs2` | 전부 | 회귀 0 (parallel race 외) |
| `scripts/layer_lint.py` | 3-F | baseline JSON과 일치 |
| S25 bit-identical 32 tok | 3-B, 3-D, 3-E (옵션 3-A) | output 동일 |
| avg_tbt n=5 회귀 ≤5% | 3-B, 3-D, 3-E | TBT 변경 없음 (구조 변경만) |

### baseline JSON

위치: `engine/tests/spec/inv_layer_baseline.json`

```
현재: 309건 (Step 1 시점 측정)
목표: 309 - 6 (3-A: V-23 1건 + 3-B: V-27 1건 + 3-D: V-07/V-08/V-19 3건 + 3-E: V-09 1건) = 303건
```

각 sub-sprint commit 시 JSON에서 해당 entry 제거.

---

## 7. 환경 / 규칙

- **언어**: 모든 응답 한국어, 기술 용어/코드 식별자 원문 유지
- **EnterWorktree**: 코드 변경 작업 시 worktree 격리 필수
- **테스트 기본 모델 포맷**: GGUF
- **Android 벤치 스레드**: Galaxy S25는 6T만
- **TBT metric**: 항상 avg_tbt (tok0 inclusive). rest_tbt 단독 비교 금지
- **성능 측정**: `--profile` 없이
- **신규 spec test**: `engine/tests/spec/`
- **완료 시 자동 commit + `notify-send`**
- **`.cl` 커널 수정**: Senior Implementer만 (본 Step 3에선 적용 없음)
- **Background job 임시 파일**: `$CLAUDE_JOB_DIR`
- **worktree symlink 필수**: `third_party/`, `libs/` (S25 빌드 시)

---

## 8. 추천 진입 순서

1. **Sprint 1**: **"Step 3-A 진행"** (auf 이동) — 가장 mechanical, 위험 LOW, 결과 빠름
2. **Sprint 2**: **"Step 3-D 진행"** (buffer 이동) — V-07/V-08/V-19 일괄 해소, paper critical path 영향 없음
3. **Sprint 3**: **"Step 3-B 진행"** 또는 **"Step 3-E 진행"** (병행 가능)
4. **Sprint 4**: 나머지 + **"Step 3-F 진행"** (baseline JSON 갱신)

총 예상: **3~4 sprint**.

---

## 9. 다음 Step (참고)

| Step | 작업 | 예상 |
|---|---|---|
| **Step 4** | L3 재배치 — `core/` → `pressure/`, `inference/` rename only | Step 3 후 진입 |
| **Step 5** | Cross-cutting 분리 — `observability/`, `resilience/` 확장 | Step 4 후 |
| **Step 6** | `/simplify` 코드 정리 — orphan import, dead code 제거 | 모든 step 후 |

본 Step 3은 외부 공개 인터페이스 stabilization의 critical path. Step 4/5는 후속 sprint.

---

## 10. 참조 문서

- `ARCHITECTURE.md §13.1~13.8` — Layered architecture 사양 (SoT)
  - §13.4 Directory Migration Map
  - §13.5 Violations (실측 31건)
  - §13.7 Migration Plan (Step 1~6)
  - §13.8 Resolved Decisions (§A~E)
- `spec/01-architecture.md §3.8` — SYS-100~105 (INV-LAYER spec)
- `spec/41-invariants.md §3.26` — 베이스라인 정책
- `arch/01-architecture.md §6` — Layered Architecture Mapping
- `scripts/layer_lint.py` — import 그래프 lint
- `engine/tests/spec/inv_layer_baseline.json` — 위반 baseline
- `.agent/todos/handoff_phase4_complete_2026_05_19.md` — Phase 4 종결 (직전)
- `.agent/todos/handoff_layered_arch_step1_complete_2026_05_16.md` — Step 1 doc 완료
