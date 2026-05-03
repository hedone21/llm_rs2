# Engine 리팩토링 분석 — 2026-05

성능과 SOLID 두 축으로 engine 크레이트 비-백엔드 영역(`core/`, `models/`, `layers/`, `eval/`, `resilience/`, `profile/`, `memory/`)을 조사한 결과를 정리한다. 백엔드 구현체(`backend/`, `kernels/`, `kivi_cache.rs`)는 단위 테스트로 회귀 검증이 어려워 분석 범위에서 제외했다.

작성: 2026-05-03 (3개 분석 에이전트 병렬 실행)

---

## 1. 적용 결과 (이 PR)

회귀 위험이 **L**이고 변경 범위가 **surgical**한 항목만 우선 적용했다.

| 항목 | 위치 | 변경 |
|------|------|------|
| **P1** | `engine/src/models/transformer.rs:109` | `Mutex<Option<PreloadPool>>` → `OnceLock<PreloadPool>` |
| **P2** | `engine/src/core/attention_scores.rs:58, 96` | `tracked_layers` Vec::with_capacity 사전 할당 |
| **P3** | `engine/src/eval/eval_loop.rs:194` | `prompt_enc.get_ids().to_vec()` 제거, 슬라이스 직접 사용 |
| **P4** | `engine/src/core/events.rs:218` | `events()` clone → `drain_events()` 추가 (clone 회피) |

L 위험이지만 **별도 PR로 분리할 가치가 있는** 항목은 §4에 따로 모았다.

---

## 2. 성능 관점

### 2.1 즉시 적용 가능 (회귀 위험 L)

| # | 위치 | 현황 | 제안 | 이득 |
|---|------|------|------|------|
| P1 | `transformer.rs:109` | `Mutex<Option<PreloadPool>>` (lazy init) | `OnceLock<PreloadPool>` | lock 제거, decode 중 1회 init만 |
| P2 | `attention_scores.rs:58,96` | `((..)..total).collect()` (Vec::new 재할당) | `Vec::with_capacity(last_n_layers)` | init 알로케이션 1회 절약 |
| P3 | `eval_loop.rs:194` | `prompt_enc.get_ids().to_vec()` 후 슬라이스로만 사용 | `&[u32]` 직접 사용 | 평가 질문당 1 alloc 제거 |
| P4 | `core/events.rs:218` | `events.lock().unwrap().clone()` (Vec 전체 복제) | `drain_events()` 추가 | 진단 모드 메모리 |

### 2.2 중기 (M 위험, 마이크로벤치 필요)

| # | 위치 | 제안 | 이득 |
|---|------|------|------|
| P5 | `kv_cache.rs` Shape::new(vec![..]) per-token | inline-array Shape 또는 cached prototype | decode hot path alloc 제거 |
| P6 | `attention_scores.rs:1060,1086` `non_bos.clone()` | in-place sort + index tracking, SmallVec | H2O score path 메모리/캐시 |
| P7 | `cache_manager.rs:60` `HashMap<EvictMethod, Box<dyn>>` (≤8 entries) | `[(EvictMethod, Box); N]` linear scan | hash overhead 제거 |
| P8 | `eviction/h2o.rs:137` 전체 sort 후 top-k | `select_nth_unstable` partial sort | O(n log n) → O(n + k log k) |

### 2.3 메모리 누적 위험 — 장시간 추론 시 OOM 가능성

| # | 위치 | 패턴 | 제안 |
|---|------|------|------|
| P9 | `profile/scores.rs:19` `snapshots/evictions/lifetimes` Vec 무제한 | step마다 누적 → 4K step 추정 ~400MB | bounded `VecDeque` + capacity 명시 |
| P10 | `eval/eviction_hook.rs:29` `KVCacheSnapshot.data: Vec<Vec<u8>>` | 다중선택 평가에서 choices×layers 복제 | `Arc<[u8]>` 또는 mmap snapshot |

---

## 3. SOLID 관점

### 3.1 SRP (Single Responsibility)

분해가 시급한 god class/file 톱 5.

1. **`TransformerModel` (`transformer.rs` ~2810 LoC, 20+ pub 메서드)**
   - 책임 융합: 모델 로딩(safetensors/gguf/AUF) + 가중치 백엔드 마이그레이션 + zero-copy 매핑 + 텐서 파티션 + 임베딩 gather + lm_head matmul fallback + KIVI plan 빌드 + 일반 plan 빌드 + offload preload pool
   - 분해 제안: `ModelLoader` / `WeightMigrator` / `PartitionInstaller` / `PlanBuilder` / `EmbedGatherer`
   - 위험: **H** (forward_into 호출자 다수). 점진적 helper 추출만 권장
   - 마이그레이션 가능: Yes — 메서드 단위 free function 이전

2. **`CommandExecutor` (`resilience/executor.rs`, 24 필드 + 1533 LoC)**
   - `apply_command`가 16+ EngineCommand variant를 한 거대 match로 처리
   - 분해: `CommandDispatcher` / `EngineStateTracker` / `HeartbeatReporter` / `ThroughputMeter`
   - 위험: **M** — `ExecutionPlan`을 카테고리 sub-struct로 분리하는 건 호환

3. **`KVCache` (`kv_cache.rs` 2797 LoC, 1 파일)**
   - KVCacheOps trait + impl + layout 분기 + utils + tests 모두 한 파일
   - 분해: 파일 분할 + `KVLayout`을 trait로 (현재는 enum + match)
   - 위험: **L–M** — 파일 분할은 re-export로 호환

4. **`AttentionScoreAccumulator` (1915 LoC, 13 필드)**
   - flat + GQA per-head + CAOTE + time-norm 융합
   - 분해: `FlatAccumulator` / `GqaAccumulator` / `CaoteSink` (composition)
   - 위험: **M** — hot path, 마이크로벤치 필수

5. **`CacheManager` 9개 force_evict 변형 (`cache_manager.rs:333-485`)**
   - score 모드(None/Flat/PerHead) × force × budget × named-policy 곱
   - 제안: `EvictionRequest` builder + 단일 `execute()` 진입점
   - 위험: **L** (내부 합성, pub wrapper 유지 가능)

### 3.2 OCP (Open/Closed) — 새 dtype/arch/command 추가 시 N곳 수정

| 위치 | 위반 | 비용 |
|------|------|------|
| `swap_executor.rs:95` `dtype_tag_to_dtype` | Q4_0만 매핑 | dtype 추가 = 함수 수정 |
| `executor.rs:333` `apply_command` | 16+ variant 빅 match | 명령 추가 = 5곳 수정 (enum + arm + plan field + sticky + Restore) |
| `transformer.rs:1252,1273,1494` Gemma3 분기 | forward_into 본체 흩뿌려짐 | 새 arch = forward_into + LayerForwardArgs 양쪽. **`ArchPolicy` trait** 도입 권장 |

### 3.3 ISP (Interface Segregation) — Fat trait

- **`KVCacheOps` (17+ 메서드, 9개가 default no-op)**: KVCache vs KiviCache 합집합
  - 분해: `KVCacheCore` + `Resizable` + `ScoreSink` + `KiviExt`
  - 호환성: KVCacheOps는 super-trait blanket으로 유지
  - 위험: **L** (정적 분해)

### 3.4 DIP (Dependency Inversion) — 구체 의존

1. **`transformer.rs`의 `downcast_ref::<OpenCLBackend>()` 패턴** (43, 718, 1453, 2271, 2293)
   - Backend trait가 있는데 우회. 백엔드별 메서드(`gpu_score_acc`, `set_op_label`, `is_nosub`, `has_kivi_attn_kernel`)를 구체 타입으로 직접 호출
   - 제안: capability sub-trait (`OpLabelHook`, `KiviKernelProvider`)를 backend가 옵션으로 제공
   - 위험: **H** (백엔드 영역, 본 분석 제외 영역과 인접)

2. **환경변수 직접 의존** (`LLMRS_SKIP_GPU_EMBED`, `LLMRS_MADV_DONTNEED`, `LLMRS_PARTITION_FUSED_MERGE`)
   - hot path에서 `std::env::var` 직접 호출 → thread-safety 우려 + 테스트 격리 불가
   - 제안: `RuntimeFlags` struct로 시작 시 1회 읽고 DI
   - 위험: **L**, 테스트성 ↑

3. **`KVCache.{k_buffer, v_buffer, current_pos, high_water_pos, max_seq_len}` 모두 `pub`**
   - 외부에서 직접 mutate 가능 → eviction/plan 빌더가 invariant 깰 위험
   - 제안: `pub(crate)` + setter 메서드
   - 위험: **L** (단, 외부 호출자 grep으로 사전 확인 필요)

---

## 4. 적용 로드맵

### A. 이번 PR (회귀 위험 L, surgical)
- ✅ P1 OnceLock<PreloadPool>
- ✅ P2 Vec::with_capacity
- ✅ P3 prompt_ids slice 사용
- ✅ P4 events drain helper

### B. 다음 PR (회귀 위험 L, 스코프가 큼 — 별도 PR로 분리)
- **DIP-2 RuntimeFlags struct** — 환경변수 hot path 의존 제거. 다파일 영향
- **DIP-3 KVCache 필드 `pub(crate)` 좁히기** — 호출자 cascading 영향 사전 검증 필요
- **SRP-5 EvictionRequest builder** — cache_manager 9개 force_evict 통합
- **ISP-1 KVCacheOps 분해** — KVCacheCore + Resizable + ScoreSink + KiviExt

### C. 중기 (회귀 위험 M, 마이크로벤치 필요)
- **OCP-3 ArchPolicy trait** — Gemma3 분기 정리
- **P5 Shape inline-array** — kv_cache hot path
- **P6 attention_scores in-place sort**
- **P7 cache_manager array dispatch**
- **P8 H2O partial sort**
- **SRP-2 ExecutionPlan 카테고리 sub-struct 분리**
- **SRP-4 AttentionScoreAccumulator composition 분해**

### D. 큰 PR (회귀 위험 H, 설계 문서 선행 필요)
- **SRP-1 TransformerModel 분해** — `arch/transformer_decomposition.md` 설계 후 진행
- **SRP-2 CommandExecutor 전체 분해**
- **SRP-3 KVCache 파일 분할** (re-export로 호환 가능하지만 git history 보존 고려)
- **DIP-1 Backend downcast_ref 제거** — 백엔드 영역 변경 동반

### E. 건드리지 않을 것
- `kivi_cache.rs` (Senior Implementer 영역, GPU 커널 의존)
- `Backend` trait 자체 (백엔드별 ABI 안정성)
- `pressure/CachePressureHandler` (이미 잘 설계됨)
- `mappers/` (이미 잘 추상화)
- `profile/` 모듈 구조 (옵션 probe 패턴 채택됨)

### 메모리 누적 위험 (별도 추적)
- **P9** `profile/scores.rs` bounded ring buffer
- **P10** `eviction_hook.rs` shared snapshot

`.agent/todos/refactoring_followups.md`에 위 항목들을 backlog 등록할 것.

---

## 5. 관련 파일 인덱스

비-백엔드 영역에서 라인 카운트 톱 (분석 시점):

| 파일 | LoC | 비고 |
|---|---:|---|
| `engine/src/core/kivi_cache.rs` | 4073 | Senior Implementer 영역, 분석 제외 |
| `engine/src/models/transformer.rs` | 3430 | SRP-1 god class |
| `engine/src/core/kv_cache.rs` | 2797 | SRP-3 / ISP-1 |
| `engine/src/layers/transformer_layer/forward_gen.rs` | 2007 | decode hot path |
| `engine/src/core/attention_scores.rs` | 1915 | SRP-4 |
| `engine/src/layers/tensor_partition.rs` | 1664 | |
| `engine/src/core/cache_manager.rs` | 1535 | SRP-5 |
| `engine/src/resilience/executor.rs` | 1533 | SRP-2 |
| `engine/src/layers/transformer_layer/forward.rs` | 1300 | prefill hot path |
| `engine/src/models/weights/swap_executor.rs` | 1252 | OCP-1 |
| `engine/src/eval/eval_loop.rs` | 1094 | |
| `engine/src/layers/transformer_layer/mod.rs` | 678 | |
| `engine/src/layers/workspace.rs` | 577 | |
| `engine/src/resilience/manager.rs` | 438 | |
