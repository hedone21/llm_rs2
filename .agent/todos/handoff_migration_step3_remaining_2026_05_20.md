# Migration Step 3 잔여 작업 handoff (3-B / 3-E / 3-F)

**작성**: 2026-05-20
**master HEAD**: `c2cb436f` (3-D-b 종결)
**Plan**: `/home/go/.claude/plans/proud-strolling-whale.md` (B안)
**진행 그래프**:
```
[3-A 완료] → [3-D-a 완료] → [3-D-b 완료] → [3-D-c 흡수됨]
                                              │
                                              ├─► [3-B 대기] ─┐
                                              │              ├─► [3-F 대기]
                                              └─► [3-E 대기] ─┘
```

---

## 1. 현재 상태 종합

### 완료된 sub-PR
| Sprint | HEAD | 핵심 변화 |
|---|---|---|
| 3-A | `5ddc66bf` | `auf/` → `shared/auf/` (V-23 해소) |
| 3-D-a | `3afafa06` | Dead code 삭제 + `MmapBuffer` 통합 + `MmapKeepAlive` trait |
| 3-D-b | `c2cb436f` | `memory/<resource>/` 신설 + 72 파일 import 갱신 |
| 3-D-c | (3-D-b sed에 흡수) | tensor_partition import + buffer/mod.rs facade |

### Baseline 변화
- 3-A 진입: 309
- 3-A 후: 297 (-12)
- 3-D-a 후: 294 (-3)
- 3-D-b 후: 294 (path-only 갱신)
- **현재 baseline JSON entry**: 306 *(rebase 후 master 코드 변경 누적으로 stale/regression mismatch 잔존 — 3-F에서 일괄 정리)*
- **layer_lint 현 violation**: 292 (`INV-002: 6 / INV-001: 31 / INV-003: 190 / INV-004: 36 / INV-005: 29`)

### 남은 V-?? 살아 있는 위반 (3-B/3-E/3-F 대상)
**INV-LAYER-002 (L2→상위) 6건** (`scripts/layer_lint.py` 출력):
```
memory/cuda/mmap.rs:22                  -> crate::models::weights::SecondaryMmap
memory/opencl/host_ptr_pool_buffer.rs:25 -> crate::backend::opencl::host_ptr_pool::HostPtrPoolGuard  ← V-07
memory/opencl/host_ptr_pool_buffer.rs:27 -> crate::models::weights::SecondaryMmap
memory/rpcmem/opencl_alias.rs:25         -> crate::models::weights::SecondaryMmap
memory/rpcmem/opencl_alias.rs:57         -> crate::models::weights::rpcmem_secondary::RpcmemLayerRegion
memory/rpcmem/opencl_alias.rs:86         -> crate::models::weights::rpcmem_secondary::RpcmemLayerRegion
```

**V-27 (L3→L1 downcast)** `engine/tests/spec/inv_layer_baseline.json`:
```
models/weights/layer_object_pool.rs:124 -> crate::backend::cuda_embedded::CudaBackend
```

---

## 2. Sprint 3-B — `WeightStagingPool` trait + pool relocation (V-27)

**진입 문장**: `"Step 3-B 진행"`
**LOC**: ~200
**위험**: M
**Dependencies**: 3-D-b 완료 (✅ 충족 — memory/cuda 위치 안정화됨)

### 핵심 위반 (V-27)
`engine/src/models/weights/layer_object_pool.rs:120-127`:
```rust
let cuda_ctx = backend
    .as_any()
    .downcast_ref::<crate::backend::cuda_embedded::CudaBackend>()  // ← L3→L1 downcast
    .ok_or_else(|| anyhow!("LayerObjectPool: backend must be CudaBackend"))?
    .context()
    .clone();
```

### 구현 단계
1. **`shared/src/lib.rs`에 `WeightStagingPool` trait 신설**:
   ```rust
   pub trait WeightStagingPool: Send + Sync {
       fn take(&self) -> Option<StagingLayerHandle>;
       fn depth(&self) -> usize;
       fn target_depth(&self) -> usize;
   }
   ```
   - `StagingLayerHandle`은 opaque handle 또는 associated type (구체 타입은 backend가 정함)
   - 실측된 `LayerObjectPool` public API (`new`, `take`, `depth`, `target_depth`) 그대로 모사
2. **`engine/src/models/weights/layer_object_pool.rs` → `engine/src/backend/cuda_embedded/pool.rs`**
   - `git mv`로 파일 이동
   - `backend/cuda_embedded/mod.rs`에 `pub mod pool;` 추가
3. **`LayerObjectPool`이 `WeightStagingPool` trait impl**
4. **downcast 제거 — `CudaBackend::create_layer_pool(...)` 메서드 추가**
   - `impl CudaBackend { pub fn create_layer_pool(&self, ...) -> Arc<dyn WeightStagingPool> { ... } }`
   - L3에서는 `backend.as_any().downcast_ref::<CudaBackend>()` 없이 trait method 호출
   - 호출처: `bin/generate.rs:1265`, `session/qcf_runtime.rs:62`, `swap_executor.rs:293, 377`
5. **`swap_executor.rs:2139` HostPtrPool downcast** — 3-D-b에서 `memory::opencl::host_ptr_pool_buffer`로 path 변경됐지만 downcast 자체는 잔존. V-27이 아닌 V-25 카테고리. 별도 처리 또는 backlog.
6. **baseline JSON V-27 entry 제거**

### 선결 확인
- `swap_executor`가 `LayerObjectPool` 인스턴스를 어떻게 얻는지 (Arc 주입? 생성?) — 변경 영향 범위 산정
- cuda_pc도 같은 pool이 필요한지 (현재 LayerObjectPool은 CudaBackend 한정 downcast → cuda_embedded만 사용 추정)

### 호출처 (실측)
```
bin/generate.rs:1242, 1265           ← Arc<LayerObjectPool> 생성 + 보관
session/qcf_runtime.rs:62            ← QcfRuntime에 Arc<LayerObjectPool> 주입
models/weights/swap_executor.rs:293, 377  ← SwapExecutor.layer_pool 필드 + with_layer_pool ctor
```

### 게이트
- `cargo build` + `cargo test --lib` 회귀 0
- **S25 weight swap 정확성 top-5 overlap > 99%** (Phase 6 swap 시나리오)
- `cargo test --test spec test_inv_layer` 8/8 PASS
- baseline JSON V-27 1건 제거 검증 (`python3 scripts/layer_lint.py --baseline`)

---

## 3. Sprint 3-E — `SecondaryStore` / `RpcmemRegion` trait inversion (V-09 잔존)

**진입 문장**: `"Step 3-E 진행"`
**LOC**: ~150 (trait + 3 모듈 import 변경)
**위험**: M
**Dependencies**: 3-D-b 완료 (✅) — 3-B와 병행 가능

### 핵심 위반 (INV-LAYER-002 5건, V-09 + 잔존 V-08)
모두 L2(memory) → L3(models/weights) 역방향 import:
```
memory/cuda/mmap.rs:22                  -> SecondaryMmap        (cuda_mmap_alias)
memory/opencl/host_ptr_pool_buffer.rs:27 -> SecondaryMmap        (host_ptr_pool_buffer)
memory/rpcmem/opencl_alias.rs:25         -> SecondaryMmap        (rpcmem_alias)
memory/rpcmem/opencl_alias.rs:57, 86     -> RpcmemLayerRegion    (rpcmem_alias)
```

### 구현 단계
1. **`shared/src/lib.rs`에 `SecondaryStore` trait 신설**:
   ```rust
   pub trait SecondaryStore: Send + Sync {
       fn layer_tensor(&self, layer_idx: usize, subname: &str) -> Option<TensorInfo>;
       fn tensor_bytes(&self, info: &TensorInfo) -> &[u8];
       fn source_path(&self) -> &Path;
       fn is_pre_converted_soa(&self) -> bool;
       fn needs_qk_unpermute_at_swap(&self) -> bool;
       fn prefault(&self);
       fn prefault_layers(&self, target_layers: &[usize]);
   }
   ```
   - `engine/src/models/weights/secondary_mmap.rs:435-550` public API 그대로 모사 (`layer_tensor`, `tensor_bytes`, `source_path`, `is_pre_converted_soa`, `needs_qk_unpermute_at_swap`, `prefault`, `prefault_layers`)
   - `TensorInfo`는 `SecondaryTensorInfo`(line 199)를 shared로 이동 or trait associated type
2. **`SecondaryMmap`이 `SecondaryStore` impl** — `engine/src/models/weights/secondary_mmap.rs:428+`에 trait impl 추가
3. **3 memory 모듈이 `&dyn SecondaryStore` 받기**:
   - `engine/src/memory/cuda/mmap.rs:22` — `use crate::models::weights::SecondaryMmap` → `use llm_shared::SecondaryStore`
   - `engine/src/memory/opencl/host_ptr_pool_buffer.rs:27` — 동일
   - `engine/src/memory/rpcmem/opencl_alias.rs:25` — 동일
   - 생성자 시그니처: `secondary: &SecondaryMmap` → `secondary: &dyn SecondaryStore`
4. **`RpcmemRegion` trait 도입** (V-09 잔존 2건):
   - `shared/src/lib.rs`에:
     ```rust
     pub trait RpcmemRegion: Send + Sync {
         fn host_ptr(&self) -> *mut u8;
         fn fd(&self) -> i32;
         fn size(&self) -> usize;
     }
     ```
   - `engine/src/models/weights/rpcmem_secondary.rs::RpcmemLayerRegion`이 impl
   - `memory/rpcmem/opencl_alias.rs:57, 86` `RpcmemLayerRegion` → `&dyn RpcmemRegion`
5. **호출처 (transformer.rs, swap_executor.rs 등) 동기화** — `SecondaryMmap`은 `SecondaryStore` 구현체이므로 대부분 변경 없음
6. **baseline JSON V-09 + 잔존 V-08 entries 제거**

### 게이트
- `cargo build` + `cargo test --lib` 회귀 0
- **S25 weight swap 정확성 top-5 overlap > 99%**
- `cargo test --test spec test_inv_layer` 8/8 PASS
- baseline JSON 5건 제거 → INV-LAYER-002 위반 6 → 1건 (V-07 `host_ptr_pool::HostPtrPoolGuard`만 잔존)

### V-07 잔존 처리
`memory/opencl/host_ptr_pool_buffer.rs:25` → `crate::backend::opencl::host_ptr_pool::HostPtrPoolGuard`는 backend 내부 자원 직접 의존. 본 sprint 범위 외 — backlog.

---

## 4. Sprint 3-F — baseline JSON 갱신 + spec test 검증 + 문서 동기화

**진입 문장**: `"Step 3-F 진행"`
**LOC**: ~80 (JSON + 문서)
**위험**: LOW
**Dependencies**: 3-B + 3-E 완료

### 구현 단계
1. **`scripts/layer_lint.py` 실행 → 현재 violation 목록 dump**
2. **`engine/tests/spec/inv_layer_baseline.json` 전면 갱신**:
   - 현재 306 entries → 예상 270~280
   - `python3 scripts/layer_lint.py --baseline` 회귀 = 0 보장
   - `--baseline` mismatch 63 stale + 51 regression (master rebase 누적) 해소
   - V-09/V-23/V-27 entries 제거 확인 (이미 완료된 sub-PR의 효과)
3. **`engine/tests/spec/test_inv_layer_001~007.rs` 실행 → 8/8 PASS**
4. **ARCHITECTURE.md §13.5 Violations 표 갱신**:
   - V-07/V-08/V-09/V-19/V-23/V-27에 `RESOLVED in Step 3-X` 주석
   - B안 적용 기록 (`memory/<resource>/` 신설)
5. **ARCHITECTURE.md §13.7 Migration Plan Step 3 표 갱신**:
   - 실제 진행 vs 계획 차이 명시
   - V-07/V-19 잔존 backlog 등록
6. **(옵션) `arch/01-architecture.md §6` Layered Architecture Mapping**:
   - `memory/<resource>/`가 L2 sub-grouping임을 명시

### 게이트
- `cargo test --test spec -p llm_rs2` PASS
- `python3 scripts/layer_lint.py --baseline` 회귀 0

---

## 5. 잔존 위반 (Step 3 외 backlog)

본 Step 3에서 해소되지 않는 위반 — 별도 backlog 등록 권장:

| 위반 | 위치 | 사유 |
|---|---|---|
| V-07 (L2→L1) | `memory/opencl/host_ptr_pool_buffer.rs:25` → `HostPtrPoolGuard` | `backend/opencl/host_ptr_pool.rs`가 `OpenCLBackend` 직접 import → L2 이동 불가. 해소하려면 backend trait 추상화 필요 |
| V-19 본질 | `tensor_partition.rs:196` → `ClSubBuffer`(L1) + `:726` test → `CpuBackend` | L3→L1 본질. tensor_partition을 L2로 이동 시 해소. 별도 sprint |
| V-25 (HostPtrPool downcast) | `swap_executor.rs:2139` | 3-B와 같은 패턴의 downcast (HostPtrPool은 OpenCL 자원). 별도 trait 추출 필요 |
| CUDA `cuda_buffer.rs` 3 struct 분리 | `memory/cuda/buffer.rs` | LOC 측정 후 결정 (Plan 3-D-b.1 옵션). 현재 단일 파일 유지 |

---

## 6. 환경 / 규칙

- 언어: 모든 응답 한국어, 기술 용어/코드 식별자 원문 유지
- EnterWorktree: 코드 변경 작업 시 worktree 격리 필수
- worktree symlink: `third_party/`, `libs/` (S25 빌드 시)
- 테스트 기본 모델 포맷: GGUF
- Android 벤치 스레드: Galaxy S25 6T만
- TBT metric: avg_tbt (tok0 inclusive)
- 성능 측정: `--profile` 없이
- 완료 시: 자동 commit + `notify-send "llm.rs" "<요약>"`
- `.cl` 커널 수정 없음 (본 Step 3은 import path/trait/구조만)

---

## 7. 핵심 수정 파일 인덱스

### 3-B (`WeightStagingPool` trait)
- `shared/src/lib.rs` — trait 신설
- `engine/src/models/weights/layer_object_pool.rs:91-204` → `engine/src/backend/cuda_embedded/pool.rs`
- `engine/src/backend/cuda_embedded/mod.rs` — `pub mod pool;` + `create_layer_pool()` method
- 호출처: `bin/generate.rs:1242,1265`, `session/qcf_runtime.rs:62`, `models/weights/swap_executor.rs:293,377,2139`

### 3-E (`SecondaryStore` + `RpcmemRegion` trait)
- `shared/src/lib.rs` — 2 trait + `TensorInfo` (SecondaryTensorInfo 이동 or assoc type)
- `engine/src/models/weights/secondary_mmap.rs:199,428+` — impl trait
- `engine/src/models/weights/rpcmem_secondary.rs::RpcmemLayerRegion` — impl `RpcmemRegion`
- `engine/src/memory/cuda/mmap.rs:22` — `SecondaryMmap` → `&dyn SecondaryStore`
- `engine/src/memory/opencl/host_ptr_pool_buffer.rs:27` — 동일
- `engine/src/memory/rpcmem/opencl_alias.rs:25,57,86` — `SecondaryMmap` + `RpcmemLayerRegion` → trait

### 3-F (검증)
- `engine/tests/spec/inv_layer_baseline.json` — 전면 갱신
- `ARCHITECTURE.md §13.5, §13.7` — RESOLVED 주석
- `arch/01-architecture.md §6` — memory sub-grouping 기록 (옵션)
- `scripts/layer_lint.py` — 실행, 수정 없음

---

## 8. 진입 권장 순서

3-B와 3-E는 독립 sub-PR로 병행 가능. 권장:

```
1. "Step 3-B 진행"  (V-27, cuda_embedded pool 전용 → 단일 backend)
2. "Step 3-E 진행"  (V-09, 3 memory 모듈 동시 변경 → 큰 PR)
3. "Step 3-F 진행"  (baseline + 문서)
```

또는 병렬 진행 시 3-E를 먼저 (더 큰 변경, conflict 회피).

---

## 9. 참조 문서

- `ARCHITECTURE.md §13.1~13.8` (SoT — Layer Definitions, Violations, Migration Plan, Resolved Decisions)
- `spec/01-architecture.md §3.8` — SYS-100~105 (INV-LAYER)
- `arch/01-architecture.md §6` — Layered Architecture Mapping
- `/home/go/.claude/plans/proud-strolling-whale.md` — 본 Step 3 plan (B안)
- `scripts/layer_lint.py` — `LAYER_RULES`에 `("memory", "L2")` 등록됨
- `engine/tests/spec/inv_layer_baseline.json` — 306 entries (3-D-b 종결, 일부 stale)
