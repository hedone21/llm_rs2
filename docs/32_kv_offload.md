# Chapter 32: KV Cache Offload

> **이전**: [31. 메모리 아키텍처 개요](31_memory_architecture.md)

KV 캐시 오프로드는 비활성 레이어의 KV 데이터를 외부 저장소로 이전하고, 레이어별 프리페치 파이프라인으로 I/O와 연산을 겹쳐 성능 저하를 최소화하는 시스템입니다.

---

## 32.1 Overview

### 핵심 목표

- **피크 메모리 절감**: 동시에 `depth`개 레이어만 attn 버퍼 보유 (전체 대비 ~75% 절감)
- **무손실**: RawStore 기반 — 압축/양자화 없이 원본 데이터 보존
- **적응형**: EMA 기반 프리페치 깊이 자동 조정
- **제로 코스트 추상화**: `KVCacheOps` trait 제네릭 모노모피즘

### 아키텍처

```
OffloadKVCache (per layer)
├── store: Box<dyn OffloadStore>     # RawStore (Vec<u8>)
├── attn_k_buf: Option<Vec<u8>>      # Lazy 할당, preload() 시 로드
├── attn_v_buf: Option<Vec<u8>>      # Lazy 할당, release_buffers() 시 해제
├── preloaded: bool                   # preload() 완료 플래그
├── store_behind: usize              # 지연 쓰기 토큰 카운터
├── out_k_buf: Option<Arc<SharedBuffer>>  # 재사용 출력 버퍼
└── out_v_buf: Option<Arc<SharedBuffer>>  # 재사용 출력 버퍼

PrefetchController
├── depth: usize                     # 현재 프리페치 깊이 (1~max_depth)
├── preload_ema_us / forward_ema_us  # EMA 기반 타이밍 추적
└── slack_streak / decrease_patience # 진동 방지

PreloadPool
├── task_tx: mpsc::Sender            # 작업 디스패치 채널
├── workers: Vec<JoinHandle>         # 지속성 워커 스레드
└── (submit → PreloadResult)         # 타입 소거 프리로드 함수
```

---

## 32.2 모듈 구조

```
engine/src/core/offload/
├── mod.rs          OffloadKVCache + KVCacheOps/PrefetchableCache impl + tests (2071줄)
├── store.rs        OffloadStore trait 정의 (28줄)
├── raw_store.rs    RawStore: 무압축 Vec<u8> 저장 (160줄)
├── prefetch.rs     PrefetchController: 적응형 프리페치 깊이 (279줄)
└── preload_pool.rs PreloadPool: 지속성 스레드 풀 (241줄)
```

---

## 32.3 OffloadStore Trait

```rust
pub trait OffloadStore: Send {
    /// 전체 KV 데이터를 저장 (prefill 시 사용)
    fn store(&mut self, k_data: &[u8], v_data: &[u8], num_tokens: usize) -> Result<()>;

    /// 저장소에서 사전 할당 버퍼로 로드
    fn load_into(&self, k_buf: &mut [u8], v_buf: &mut [u8]) -> Result<usize>;

    /// 단일 토큰 K/V 추가 (decode 시)
    fn append_token(&mut self, k_token: &[u8], v_token: &[u8]) -> Result<()>;

    fn storage_size(&self) -> usize;
    fn stored_tokens(&self) -> usize;
    fn clear(&mut self);
}
```

### RawStore 구현

```rust
pub struct RawStore {
    k_data: Vec<u8>,       // K 캐시 원본 바이트
    v_data: Vec<u8>,       // V 캐시 원본 바이트
    num_tokens: usize,     // 저장된 토큰 수
    token_bytes: usize,    // 토큰당 바이트 (kv_heads × head_dim × dtype.size())
}
```

- **무압축**: 고엔트로피 F16/F32 데이터에서 LZ4 압축률 ~1.0x이므로 압축 불필요
- **무오버헤드**: `load_into()`는 `copy_from_slice()` 한 번
- **크기**: 정확히 `num_tokens × token_bytes × 2` (K+V)

---

## 32.4 OffloadKVCache

### 구조체 필드

```rust
pub struct OffloadKVCache {
    layer_id: usize,                    // 디버그용 레이어 인덱스
    kv_heads: usize,
    head_dim: usize,
    dtype: DType,                       // F16 또는 F32
    current_pos: usize,                 // 유효 토큰 수
    max_seq_len: usize,
    token_bytes: usize,                 // kv_heads × head_dim × dtype.size()
    store: Box<dyn OffloadStore>,       // RawStore
    attn_k_buf: Option<Vec<u8>>,        // Lazy 할당 K 어텐션 버퍼
    attn_v_buf: Option<Vec<u8>>,        // Lazy 할당 V 어텐션 버퍼
    preloaded: bool,                    // preload() 완료 플래그
    out_k_buf: Option<Arc<SharedBuffer>>,  // 재사용 출력 버퍼 (R-P2)
    out_v_buf: Option<Arc<SharedBuffer>>,  // 재사용 출력 버퍼 (R-P2)
    out_backend: Arc<CpuBackend>,
    store_behind: usize,                // 지연 쓰기 대기 토큰 수
}
```

### 주요 메서드

| 메서드 | 용도 | 비용 |
|--------|------|------|
| `new()` | 생성 (attn 버퍼 미할당) | O(1) |
| `preload()` | store → attn_buf 로드 (lazy 할당) | O(current_pos × token_bytes) |
| `release_buffers()` | attn_buf 해제 + 지연 토큰 플러시 | O(store_behind × token_bytes) |
| `retain_preload()` | 교차 토큰 attn_buf 유지 | O(1) |
| `reset_preload()` | preloaded 플래그 리셋 | O(1) |
| `update(K, V)` | KV 데이터 추가 | O(seq_len × token_bytes) |
| `get_view()` | 어텐션용 텐서 반환 | O(current_pos × token_bytes) |

### 동작 모드

#### Prefill (seq_len > 1)

```
update(K, V) → store.store(k_data, v_data, seq_len)
                또는 store.append_token() × seq_len (이미 데이터 있을 시)
```

Prefill은 표준 `forward_into()` 경로를 사용하므로 프리페치 오버헤드가 없습니다.

#### Decode — 비보유 레이어 (preloaded=false)

```
update(K, V) → store.append_token(k, v)   [즉시 store에 쓰기]
get_view()   → store.load_into(attn_buf)  [동기 로드, 폴백]
```

#### Decode — 보유 레이어 (preloaded=true, depth 이내)

```
update(K, V) → attn_buf에 직접 append    [store_behind++]
get_view()   → attn_buf에서 직접 반환     [store I/O 없음]
```

이것이 **지연 쓰기 패턴 (Deferred Write)**의 핵심입니다.

### 지연 쓰기 상세

```
preload() → store에서 attn_buf로 전체 로드
    ↓
update() (decode, preloaded=true)
    → attn_buf[current_pos]에 직접 쓰기
    → store_behind += 1
    → store에는 안 씀 (I/O 절감)
    ↓
get_view() → attn_buf 데이터 그대로 반환
    ↓
retain_preload() → preloaded=true 유지 (다음 토큰에서도 attn_buf 활성)
    ↓
... (N 토큰 반복) ...
    ↓
release_buffers()
    → flush_deferred(): store_behind개 토큰을 store에 배치 쓰기
    → attn_buf = None (메모리 해제)
```

**효과**: 보유 레이어는 디코드 중 store I/O가 0입니다.

---

## 32.5 PreloadPool

### 설계 동기

이전 구현(`thread::scope`)은 토큰마다 스레드를 생성/조인했습니다. `PreloadPool`은 지속적 워커 스레드로 이를 대체합니다.

### 구조

```rust
pub struct PreloadPool {
    task_tx: Option<mpsc::Sender<PreloadTask>>,  // 작업 디스패치
    workers: Vec<thread::JoinHandle<()>>,         // 지속성 워커
    size: usize,
}

pub struct PreloadTask {
    cache_ptr: *mut (),                           // 타입 소거 캐시 포인터
    preload_fn: unsafe fn(*mut ()) -> Result<()>, // 타입 소거 프리로드 함수
    result_tx: mpsc::SyncSender<PreloadResult>,   // 일회용 결과 채널
}

pub struct PreloadResult {
    pub result: Result<()>,
    pub duration: Duration,     // 프리로드 소요 시간
}
```

### 타입 소거

`PreloadPool`은 `OffloadKVCache` 타입을 모릅니다. 대신 raw pointer + 타입 소거 함수를 사용합니다:

```rust
pub unsafe fn preload_erased<C: PrefetchableCache>(ptr: *mut ()) -> Result<()> {
    unsafe { (*(ptr as *mut C)).preload() }
}
```

### Safety Invariants

1. `cache_ptr`는 유효한 `Vec<C>` 원소를 가리킴
2. 각 레이어는 동시에 최대 하나의 스레드만 접근 (i vs i+depth)
3. preload 작업이 완료된 후에만 해당 레이어 release 가능

---

## 32.6 PrefetchController

### 적응형 깊이 조정

```rust
pub struct PrefetchController {
    depth: usize,              // 현재 깊이 (≥1)
    max_depth: usize,          // 상한
    preload_ema_us: f64,       // 프리로드 시간 EMA (μs)
    forward_ema_us: f64,       // 포워드 시간 EMA (μs)
    alpha: f64,                // EMA 계수 (0.3)
    samples: usize,
    warmup_samples: usize,     // num_layers (워밍업 기간)
    slack_streak: usize,       // 연속 슬랙 관찰 횟수
    decrease_patience: usize,  // 감소 인내심 (3)
}
```

### 조정 알고리즘

```
slack_ratio = 1.0 − (preload_ema / forward_ema)

if slack_ratio < 0:
    → STALL: 프리로드가 포워드보다 느림
    → depth 즉시 증가 (min(depth+1, max_depth))

elif slack_ratio > 0.3:
    → SLACK: 프리로드가 포워드보다 30%+ 빠름
    → slack_streak++
    → slack_streak ≥ 3이면 depth 감소 (max(depth-1, 1))

else:
    → BALANCED: 변경 없음, slack_streak 리셋
```

**진동 방지**: 감소는 3회 연속 슬랙 관찰 후에만 적용됩니다.

### 타이밍 기록

- `record_preload(dur)`: 백그라운드 프리로드 완료 시 호출
- `record_forward(dur)`: 레이어 포워드 완료 시 호출
- `adjust()`: 토큰 경계에서 한 번 호출

---

## 32.7 forward_into_offload() 파이프라인

`LlamaModel::forward_into_offload()` (`models/llama/llama_model.rs`)는 decode 시 레이어별 프리페치 파이프라인을 실행합니다.

### 토큰당 실행 흐름

```
[1] PreloadPool 초기화 (최초 호출 시 한 번)
    pool = PreloadPool::new(max_depth)

[2] 동기 프리로드 [0..depth)
    for j in 0..depth:
        cache[j].preload()    ← retain된 경우 early-exit

[3] 배경 프리로드 [depth..2×depth)
    for j in depth..2*depth:
        pending[j] = pool.submit(cache[j], preload_erased)

[4] Layer Loop (i = 0..num_layers):
    ┌─ (a) pending[i] 결과 수집 → prefetch.record_preload(dur)
    ├─ (b) pending[i+depth] 배경 제출 (존재하면)
    ├─ (c) layer[i].forward(kv_cache=cache[i])
    │       → cache[i].get_view()  ← preloaded면 attn_buf 사용
    │       → prefetch.record_forward(dur)
    ├─ (d) i < depth → cache[i].retain_preload()    [보유]
    └─ (e) i > 0 && i-1 ≥ depth → cache[i-1].release_buffers()  [해제]

[5] 잔여 pending 드레인 + 마지막 레이어 해제

[6] prefetch.adjust()  ← 다음 토큰을 위한 깊이 조정
```

### 동시 접근 안전성

```
depth=2 예시 (16 레이어):

Main thread:    forward(0) forward(1) forward(2) forward(3) ...
Worker 1:                  preload(2) preload(4) preload(6) ...
Worker 2:                  preload(3) preload(5) preload(7) ...

- 레이어 i의 forward와 레이어 i+depth의 preload가 동시 실행
- depth ≥ 1이므로 i ≠ i+depth: 동일 레이어 동시 접근 없음
- raw 포인터 사용하지만, split_at_mut의 안전성 모델과 동등
```

---

## 32.8 CLI 사용법

```bash
cargo run --release --bin generate -- \
  --model-path models/llama3.2-1b \
  --kv-offload raw \                 # 오프로드 모드 (none/raw)
  --max-prefetch-depth 4 \           # 최대 프리페치 깊이
  --kv-type f16 \                    # F16 또는 F32 필수
  --kv-layout seq \                  # SeqMajor 필수
  --prompt "Hello" -n 128
```

### 제약사항

| 조건 | 이유 |
|------|------|
| `--kv-type f16` 또는 `f32` | Q4_0 블록 단위 접근과 바이트 단위 저장소 비호환 |
| `--kv-layout seq` | SeqMajor 연속 레이아웃만 지원 |
| Eviction 정책 비호환 | OffloadKVCache에 `prune_prefix()` 미구현 |

---

## 32.9 메모리 예산

### Llama 3.2 1B (F16, 16 layers, seq=2048)

| 구성요소 | 표준 KVCache | Offload (depth=2) |
|---------|------------|-------------------|
| KV 데이터 (전체) | 64 MB | 64 MB (RawStore) |
| 활성 attn 버퍼 | 64 MB (전 레이어) | 8 MB (2 레이어) |
| 출력 SharedBuffer | — | ~4 MB (재사용) |
| **활성 메모리** | **64 MB** | **~12 MB** |
| **총 메모리** | **64 MB** | **~76 MB** |

> RawStore는 무압축이므로 총 메모리는 오히려 증가하지만, **활성(피크) 메모리**가 핵심입니다.
> 오프로드의 가치는 동시에 활성인 버퍼를 `depth`개 레이어로 제한하는 데 있습니다.

---

## 32.10 테스트 커버리지

### RawStore 테스트 (4개)

| 테스트 | 검증 |
|--------|------|
| `test_raw_store_basic` | store → load_into 왕복 bit-exact |
| `test_raw_store_append_token` | 50토큰 증분 추가 |
| `test_raw_store_clear` | 리셋 후 0 토큰 |
| `test_raw_store_empty` | 빈 store에서 load_into → 0 반환 |

### OffloadKVCache 테스트 (18개)

| 카테고리 | 테스트 | 검증 |
|---------|--------|------|
| 기본 | `test_offload_kvcache_ops` | prefill + get_view 정확성 |
| | `test_offload_kvcache_overflow` | 용량 초과 에러 |
| | `test_offload_kvcache_empty_view` | 빈 캐시 [1,0,h,d] 형상 |
| | `test_offload_kvcache_memory_usage` | 메모리 사용량 추적 |
| 정확도 | `test_integration_base_vs_offload_f16_accuracy` | BASE vs Offload bit-exact |
| 프리로드 | `test_preload_skips_io_in_get_view` | preload 후 get_view에서 재로드 안 함 |
| | `test_preload_update_append_to_attn_buf` | preload 후 update가 attn_buf에 추가 |
| | `test_preload_concurrent_split_at_mut` | thread::scope 동시 접근 |
| | `test_preload_empty_cache` | 빈 캐시 preload 안전 |
| | `test_preload_idempotent` | 다중 preload 멱등성 |
| 교차 토큰 | `test_retain_preload_cross_token` | retain 후 지연 쓰기 작동 |
| | `test_retain_preload_depth_decrease` | retain 후 release 정상 |
| | `test_retain_preload_guards_none_bufs` | 버퍼 없이 retain 안전 |
| 지연 쓰기 | `test_deferred_write_skips_store` | preloaded 시 store 쓰기 건너뜀 |
| | `test_deferred_flush_on_release` | release 시 모든 지연 토큰 플러시 |
| | `test_deferred_preload_after_behind` | store_behind>0에서 preload 정상 |
| | `test_deferred_write_with_raw_store` | 5 지연 토큰 데이터 보존 |
| | `test_non_retained_update_writes_store_immediately` | 비프리로드 시 즉시 쓰기 |

### PrefetchController 테스트 (5개)

| 테스트 | 검증 |
|--------|------|
| `test_warmup_no_adjust` | 워밍업 중 조정 없음 |
| `test_increase_on_stall` | 스톨 시 depth 증가 |
| `test_decrease_with_patience` | 3회 연속 슬랙 후 감소 |
| `test_max_depth_cap` | max_depth 초과 방지 |
| `test_no_oscillation` | 교대 스톨/슬랙에서 진동 없음 |

### PreloadPool 테스트 (4개)

| 테스트 | 검증 |
|--------|------|
| `test_pool_basic` | 단일 작업 완료 |
| `test_pool_concurrent_tasks` | 8작업 × 4워커 동시 실행 |
| `test_pool_drop_joins_workers` | drop 시 진행 중 작업 완료 대기 |
| `test_pool_result_timing` | 결과 duration 정확성 |

### 벤치마크 테스트 (3개)

| 테스트 | 비교 |
|--------|------|
| `test_bench_adaptive_prefetch` | depth=1 vs 적응형 vs 고정 vs 지연 |
| `test_bench_deferred_store_write` | RawStore 비용 vs BASE |
| `test_bench_pool_vs_scope` | PreloadPool vs thread::scope |

---

## 32.11 설계 결정 로그

| 결정 | 근거 |
|------|------|
| RawStore만 유지 (DiskStore/ZramStore 제거) | F16/F32 고엔트로피 데이터에서 LZ4 압축률 ~1.0x, 복잡성만 증가 |
| `std::thread` + `mpsc` (tokio 금지) | 프로젝트 제약: tokio 미사용 |
| 지연 쓰기 패턴 | 보유 레이어의 이중 I/O(store + attn_buf) 제거 → ~40% 디코드 시간 절감 |
| 지속성 PreloadPool | 토큰당 thread::spawn 오버헤드 제거 |
| EMA 기반 적응형 깊이 | 하드웨어/워크로드별 최적 깊이 자동 탐색 |
| 감소 patience=3 | depth oscillation 방지, 안정성 우선 |
| Eviction 미지원 | CacheManager 제네릭화 필요, 별도 작업으로 분리 |
| SeqMajor 전용 | 연속 바이트 저장/로드를 위한 단순화 |
