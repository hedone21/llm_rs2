# KV Cache Layer-wise Offload Design

> **Status**: Draft
> **Date**: 2026-03-12
> **Related**: `docs/11_kv_cache_management.md`, `engine/src/core/kv_cache.rs`

## 1. Overview

두 가지 KV 캐시 오프로드 방식을 레이어 프리페치 파이프라인 위에 구현한다.

| 방식 | 저장소 | 품질 | 메모리 절약 (F16, 16L, seq=2048) |
|------|--------|------|--------------------------------|
| **DiskStore** | 디스크 파일 | 무손실 | 56 MB (64→8 MB) |
| **ZramStore** | LZ4 압축 메모리 | 무손실 | ~30 MB (64→~28+8 MB) |
| (참고) KIVI Q2 | 인메모리 양자화 | **손실** | 61.8 MB (64→2.2 MB) |

**핵심 원칙**: Prefill 단계에서는 일반 KVCache를 사용하고, **이벤트 발생 시**(메모리 압박, 사용자 명령 등) 오프로드 모드로 전환한다.

## 2. Architecture

```
KVCacheOps (trait, 기존)
  ├── KVCache        (인메모리, 기존)
  ├── KiviCache      (Q2 양자화, 기존)
  └── OffloadKVCache (신규, 레이어별 오프로드)
         │
         ├── store: DiskStore    ← 방식 A: 디스크 파일 I/O
         └── store: ZramStore    ← 방식 B: LZ4 압축 메모리

PrefetchPipeline (신규)
  - std::thread + mpsc 채널 (tokio 금지)
  - 청크 단위 더블 버퍼 (chunk_size 레이어 × 2)
  - Chunk N 연산 중 Chunk N+1 비동기 로드
  - 적응형 chunk_size: 런타임 I/O 대비 연산 비율로 자동 조정
```

### 2.1 Module Structure

```
engine/src/core/
  kv_cache.rs              (기존 - KVCacheOps, KVCache)
  kivi_cache.rs            (기존 - KiviCache)
  offload/                 (신규)
    mod.rs                 OffloadKVCache + KVCacheOps impl
    store.rs               OffloadStore trait 정의
    disk_store.rs          DiskStore 구현
    zram_store.rs          ZramStore 구현 (LZ4 + 전처리)
    pipeline.rs            PrefetchPipeline (더블 버퍼 + I/O 스레드)
    preprocess.rs          F16/F32 바이트 재배치 (Zram 압축률 핵심)
```

### 2.2 Key Traits & Structs

```rust
// ── store.rs ──
pub trait OffloadStore: Send {
    /// 전체 KV 데이터를 저장소에 기록
    fn store(&mut self, k_data: &[u8], v_data: &[u8], num_tokens: usize) -> Result<()>;
    /// 저장소에서 KV 데이터를 복원 (사전 할당 버퍼에 기록)
    fn load_into(&self, k_buf: &mut [u8], v_buf: &mut [u8]) -> Result<usize>;
    /// 1 토큰 append (decode용)
    fn append_token(&mut self, k_token: &[u8], v_token: &[u8]) -> Result<()>;
    /// 저장소 사용량 (바이트)
    fn storage_size(&self) -> usize;
    /// 저장된 토큰 수
    fn stored_tokens(&self) -> usize;
    /// 리셋
    fn clear(&mut self);
}

// ── mod.rs ──
pub struct OffloadKVCache {
    layer_id: usize,
    kv_heads: usize,
    head_dim: usize,
    dtype: DType,              // F16 또는 F32
    current_pos: usize,
    max_seq_len: usize,
    token_bytes: usize,        // kv_heads * head_dim * dtype.size()
    state: CacheState,
    store: Box<dyn OffloadStore>,
}

#[derive(Clone, Copy, PartialEq)]
enum CacheState {
    /// 데이터가 활성 버퍼에 로드됨 (연산 가능)
    Active,
    /// 데이터가 저장소에만 존재 (로드 필요)
    Offloaded,
    /// 비동기 로드 진행 중
    Loading,
}

// ── pipeline.rs ──

/// 청크 구성: 성능/메모리 트레이드오프 제어
pub struct ChunkConfig {
    /// 한 번에 프리페치할 레이어 수 (1, 2, 4, 8)
    pub chunk_size: usize,
    /// 파이프라인 버퍼에 할당 가능한 최대 메모리 (바이트)
    /// chunk_size는 이 예산 내에서 자동 결정됨
    pub max_buffer_bytes: Option<usize>,
    /// 적응형 조정 활성화
    pub adaptive: bool,
}

pub struct PrefetchPipeline {
    num_layers: usize,
    chunk_config: ChunkConfig,
    /// 더블 버퍼: 각각 chunk_size 레이어분의 K + V 데이터
    /// buf_a[layer_offset..layer_offset+layer_bytes] 로 레이어별 접근
    buf_a: AlignedBuffer,   // chunk_size × layer_kv_bytes
    buf_b: AlignedBuffer,   // chunk_size × layer_kv_bytes
    layer_kv_bytes: usize,  // 레이어당 K+V 바이트 수
    /// 현재 어떤 버퍼가 활성인지
    active_buf: BufferSlot, // A or B
    /// I/O 스레드 통신 (std::sync::mpsc)
    request_tx: Sender<PipelineCmd>,
    result_rx: Receiver<PipelineResult>,
    io_thread: Option<JoinHandle<()>>,
    /// 적응형 조정을 위한 런타임 통계
    stats: PipelineStats,
}

/// 런타임 성능 측정 (적응형 chunk_size 조정용)
struct PipelineStats {
    /// I/O 시간 EMA (지수이동평균), ms 단위
    io_ema_ms: f32,
    /// Compute 시간 EMA, ms 단위
    compute_ema_ms: f32,
    /// 연속 스톨 횟수 (I/O가 compute보다 오래 걸린 경우)
    consecutive_stalls: usize,
    /// 총 처리된 청크 수
    chunks_processed: usize,
}

enum PipelineCmd {
    /// 청크 로드: layers[start..end]를 지정 버퍼에 로드
    LoadChunk { start_layer: usize, end_layer: usize, buf: BufferSlot },
    /// 청크 저장: 지정 버퍼의 데이터를 저장소에 기록
    StoreChunk { start_layer: usize, end_layer: usize, buf: BufferSlot },
    Shutdown,
}
```

### 2.3 Offload Lifecycle

```
[일반 추론 (Prefill)]
  KVCache (인메모리) ← 기존과 동일, 변경 없음
         │
         ▼ (이벤트: 메모리 압박 / 사용자 명령 / decode 전환)
[Migration]
  KVCache.data → OffloadStore.store()  (레이어별 저장)
  KVCache 메모리 해제
         │
         ▼
[오프로드 추론 (Decode)]
  PrefetchPipeline: Load(L0) → [Compute(L0) | Load(L1)] → ...
         │
         ▼ (이벤트: 메모리 여유 확보)
[Restore] (선택사항)
  OffloadStore.load() → KVCache (인메모리 복귀)
```

### 2.4 Chunked Prefetch Strategy

레이어 하나씩 프리페치하면 I/O 비용이 연산보다 클 때 파이프라인이 스톨된다.
**여러 레이어를 하나의 청크로 묶어서** 프리페치하면 이 문제를 완화할 수 있다.

#### 2.4.1 기본 개념

```
단일 레이어 프리페치 (chunk_size=1, 기존):
  Load(L0) → [Compute(L0) | Load(L1)] → [Compute(L1) | Load(L2)] → ...
  매 스텝 병목: max(io_1layer, compute_1layer)

청크 프리페치 (chunk_size=4):
  Load(L0-3) → [Compute(L0-3) | Load(L4-7)] → [Compute(L4-7) | Load(L8-11)] → ...
  매 스텝 병목: max(io_4layers, compute_4layers)
```

#### 2.4.2 청크가 효과적인 이유

**1. 고정 오버헤드 분산 (amortization)**

각 프리페치 작업에는 데이터 크기와 무관한 고정 비용이 있다:
- 스레드 wake-up: ~0.1-0.5ms
- mpsc 채널 send/recv: ~0.01ms
- 파일 seek (DiskStore): ~0.1-0.5ms
- LZ4 함수 호출 (ZramStore): ~0.01ms

```
chunk_size=1: 16 sync points × ~0.5ms = ~8ms 고정 오버헤드/token
chunk_size=4:  4 sync points × ~0.5ms = ~2ms 고정 오버헤드/token
절감: ~6ms/token
```

**2. 순차 I/O 효율 향상 (DiskStore)**

Android UFS는 대용량 순차 읽기에서 처리량이 향상된다:
```
 8 MB read: ~1.5 GB/s (syscall + 스토리지 컨트롤러 오버헤드)
16 MB read: ~1.6 GB/s
32 MB read: ~1.7 GB/s (오버헤드 분산, 더 큰 DMA 전송)
```

**3. 레이턴시 분산 평균화**

개별 레이어의 I/O 지터나 연산 시간 편차가 청크 내에서 평균화된다.
한 레이어에서 I/O 스파이크가 발생해도 같은 청크의 다른 레이어가 이를 보상.

**4. ZramStore 압축 효율**

더 큰 블록에서 LZ4가 더 긴 패턴을 찾을 수 있고,
바이트 재배치(shuffle)도 NEON 벡터 연산으로 더 효율적으로 처리된다.

#### 2.4.3 메모리-성능 트레이드오프

더블 버퍼 메모리 = `2 × chunk_size × layer_kv_bytes`

| chunk_size | 버퍼 메모리 (F16) | 버퍼 메모리 (F32) | 메모리 절약 (vs 전체 인메모리) | Sync 횟수 |
|------------|------------------|------------------|------------------------------|-----------|
| 1 | 8 MB | 16 MB | 87.5% | 16 |
| 2 | 16 MB | 32 MB | 75.0% | 8 |
| 4 | 32 MB | 64 MB | 50.0% | 4 |
| 8 | 64 MB | 128 MB | 0% (의미 없음) | 2 |

> **상한선**: `chunk_size ≤ num_layers / 2` (그 이상이면 오프로드 의미 없음)

#### 2.4.4 적응형 Chunk Sizing

런타임에 chunk_size를 자동 조정한다:

```rust
impl PrefetchPipeline {
    /// 매 청크 처리 후 호출: I/O vs Compute 시간 비교
    fn adapt_chunk_size(&mut self) {
        let ratio = self.stats.io_ema_ms / self.stats.compute_ema_ms;

        let desired = if ratio <= 1.0 {
            1  // I/O ≤ Compute: 최소 chunk으로 메모리 최적화
        } else if ratio <= 1.5 {
            2  // 약간 I/O bound: chunk 2로 고정 오버헤드 절감
        } else if ratio <= 2.5 {
            4  // 확실히 I/O bound: chunk 4
        } else {
            8  // 심각한 I/O 병목: 최대 chunk
        };

        // 메모리 예산 확인
        let max_by_budget = self.chunk_config.max_buffer_bytes
            .map(|b| b / (2 * self.layer_kv_bytes))
            .unwrap_or(usize::MAX);

        // 레이어 수 상한
        let max_by_layers = self.num_layers / 2;

        self.chunk_config.chunk_size = desired
            .min(max_by_budget)
            .min(max_by_layers)
            .max(1);
    }
}
```

**안정성 규칙**:
- chunk_size 변경은 토큰 경계에서만 (forward pass 도중 변경 금지)
- 증가는 즉시, 감소는 연속 3회 안정 후 적용 (oscillation 방지)
- EMA smoothing factor: α=0.3 (최근 값에 가중)

#### 2.4.5 Forward Loop (청크 기반)

```rust
pub fn forward_into_offload(
    &self,
    args: LlamaModelForwardArgs<OffloadKVCache>,
    pipeline: &mut PrefetchPipeline,
) -> Result<()> {
    // ... embedding lookup ...

    let C = pipeline.chunk_size();
    let num_chunks = (self.layers.len() + C - 1) / C;

    // Load first chunk
    pipeline.load_chunk(0)?;

    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * C;
        let end = (start + C).min(self.layers.len());

        // Wait for current chunk data
        pipeline.wait_chunk_ready(chunk_idx)?;
        let compute_start = Instant::now();

        // Start loading next chunk (overlaps with compute)
        if chunk_idx + 1 < num_chunks {
            pipeline.load_chunk(chunk_idx + 1)?;
        }

        // Compute all layers in this chunk
        for i in start..end {
            let buf_view = pipeline.layer_view(i);  // 버퍼 내 레이어 슬라이스
            layer_forward_with_buffer(
                &self.layers[i], &mut x, buf_view,
                &mut kv_caches[i], backend, memory, ...
            )?;
        }

        let compute_ms = compute_start.elapsed().as_secs_f32() * 1000.0;
        pipeline.record_compute(compute_ms);

        // Async write-back (append new tokens for this chunk)
        pipeline.store_chunk(chunk_idx)?;
    }

    // Adaptive adjustment for next token
    if pipeline.is_adaptive() {
        pipeline.adapt_chunk_size();
    }

    // ... final norm + head ...
}
```

#### 2.4.6 버퍼 레이아웃 (청크 내 레이어 배치)

```
AlignedBuffer (chunk_size=4 예시):
┌──────────────────────────────────────────────────────────┐
│ Layer 0 K data │ Layer 0 V data │ Layer 1 K │ Layer 1 V │...
│ [token_bytes]  │ [token_bytes]  │           │           │
└──────────────────────────────────────────────────────────┘
  ↑ offset = 0    ↑ offset = tb    ↑ 2*tb      ↑ 3*tb

layer_view(i) → &buf[i*2*token_bytes .. (i+1)*2*token_bytes]
  여기서 token_bytes = current_pos × kv_heads × head_dim × dtype.size()
```

DiskStore: 파일 내 레이아웃과 동일한 순서 → 순차 읽기 최적화.
ZramStore: 청크 전체를 하나의 LZ4 블록으로 압축 가능 (더 높은 압축률).

## 3. Risk Analysis

### 3.1 DiskStore Risks

#### R-D1: Android 플래시 I/O 지터 (Severity: HIGH)

**문제**: Android 플래시 스토리지는 GC(Garbage Collection), TRIM, 다른 앱 I/O로 인해
지연 시간이 급증할 수 있다 (정상 3ms → 스파이크 50-200ms).

**영향**: 파이프라인 스톨 → 토큰 생성 지연 급증 (P99 레이턴시 악화)

**대응**:
- **Fallback 타이머**: prefetch에 timeout 설정 (예: 20ms). 초과 시 해당 레이어는
  동기 로드로 전환하고, 경고 로그 출력
- **적응형 버퍼링**: 연속 스파이크 감지 시 더블 버퍼 → 트리플 버퍼로 확장
- **Direct I/O**: `O_DIRECT` 플래그로 OS 페이지 캐시 우회, GC 간섭 최소화
- **모니터링**: 레이어별 로드 시간 히스토그램 기록, 벤치마크에 P50/P99 포함

#### R-D2: 플래시 마모 (Severity: MEDIUM)

**문제**: Decode 매 토큰마다 16 레이어에 각 2KB append → 토큰당 32KB 쓰기.
2048 토큰 생성 시 64MB 누적 쓰기. 장시간 사용 시 플래시 수명 감소.

**영향**: 기기 수명 단축 (특히 저가 eMMC 스토리지)

**대응**:
- **Append-only + 배치 fsync**: 매 토큰마다 fsync 하지 않음. N 토큰마다 한번 fsync
  (데이터 유실 위험은 추론 중단 시에만 발생하므로 수용 가능)
- **tmpfs 우선 사용**: `/dev/shm` 또는 Android의 tmpfs 마운트 포인트 활용.
  가용 시 디스크 대신 RAM 기반 tmpfs 사용 (메모리 여유분만큼)
- **세션 종료 시 정리**: 오프로드 파일은 임시 파일로 관리, 세션 종료 시 자동 삭제
- **쓰기 횟수 제한**: 누적 쓰기량 모니터링, 임계값 초과 시 경고

#### R-D3: 파일 시스템 오버헤드 (Severity: LOW)

**문제**: 레이어당 2개 파일 (K, V) × 16 레이어 = 32 파일 open/manage.
작은 파일 다수보다 큰 파일 소수가 I/O 효율적.

**대응**:
- **단일 파일 + 오프셋**: 전체 레이어를 하나의 파일에 저장, 레이어별 오프셋으로 접근
  ```
  [Header: layer_count, token_bytes, tokens_per_layer]
  [Layer 0 K data][Layer 0 V data]
  [Layer 1 K data][Layer 1 V data]
  ...
  ```
- **mmap 고려**: 대용량 파일은 mmap으로 접근하여 read() syscall 오버헤드 제거.
  단, mmap은 페이지 폴트 지연이 예측 불가하므로 **선택적 사용**

#### R-D4: 저장 공간 부족 (Severity: MEDIUM)

**문제**: F32, seq=2048 기준 128MB, F16 기준 64MB 디스크 공간 필요.
저장 공간이 부족한 기기에서 실패 가능.

**대응**:
- **사전 공간 검사**: 오프로드 시작 전 `statvfs()`로 가용 공간 확인
- **Graceful fallback**: 공간 부족 시 오프로드 중단, 인메모리 유지 + 경고
- **`max_offload_layers` 옵션**: 일부 레이어만 오프로드 가능하도록 설정

#### R-D5: Migration 중 지연 (Severity: MEDIUM)

**문제**: 인메모리 → 디스크 마이그레이션 시 전체 KV 데이터를 한번에 기록.
F16 seq=2048: 64MB 기록 → ~43ms @ 1.5GB/s. 이 시간 동안 추론 중단.

**대응**:
- **비동기 마이그레이션**: I/O 스레드에서 레이어별 순차 기록.
  마이그레이션 완료된 레이어부터 오프로드 모드로 전환
- **Lazy 마이그레이션**: 처음에는 마이그레이션 안 하고, 각 레이어가 다음에
  사용된 후 write-back 시점에 자연스럽게 디스크에 기록

### 3.2 ZramStore Risks

#### R-Z1: 부동소수점 데이터 비압축성 (Severity: CRITICAL)

**문제**: F16/F32 raw 데이터를 LZ4로 직접 압축하면 **압축률 ~1.0x (0% 절감)**.
가수부(mantissa) 엔트로피가 바이트당 ~7.3비트로 거의 랜덤.
**전처리 없이 ZramStore는 무의미함.**

**영향**: 메모리 절감 효과 없음. 오히려 압축/해제 오버헤드만 추가.

**대응**:
- **필수 전처리: 바이트 재배치 (byte-shuffle)**
  ```
  원본 F16: [b0_hi, b0_lo, b1_hi, b1_lo, b2_hi, b2_lo, ...]
  재배치:   [b0_hi, b1_hi, b2_hi, ...] [b0_lo, b1_lo, b2_lo, ...]
  ```
  상위 바이트(지수+부호)가 연속 배치되어 LZ4 패턴 매칭 가능.
  논문 기반 기대 압축률: **F16 ~2.0-2.3x, F32 ~1.5-2.0x**

- **선택적 전처리: 지수-델타 인코딩**
  채널별 base exponent 계산 후 delta만 저장.
  추가 압축률 개선 가능하나 구현 복잡도 증가.

- **Validation gate**: 구현 초기에 실제 KV 캐시 데이터로 압축률 측정.
  **1.5x 미만이면 ZramStore 전략 재검토** (abort 기준 명확화)

#### R-Z2: 압축 오버헤드 (Severity: MEDIUM)

**문제**: Decode 시 매 토큰마다 현재 레이어를 재압축해야 함 (1 토큰 append 후).
LZ4 압축 속도: Snapdragon ~350-500 MB/s. 4MB 레이어 압축: ~8-11ms.

**영향**: 레이어 연산(~3ms)보다 압축이 훨씬 오래 걸림 → 파이프라인 병목

**대응**:
- **잔여 버퍼 (Residual buffer)**: KiviCache 패턴 차용.
  최근 N 토큰 (예: 64)은 미압축 상태로 유지.
  잔여 버퍼가 가득 차면 배치 압축하여 압축 블록에 추가.
  ```
  [압축 블록 0..M] + [미압축 잔여 버퍼: 최근 N 토큰]
  load 시: 압축 블록 해제 + 잔여 버퍼 복사
  ```
  매 토큰 append는 잔여 버퍼에만 기록 → 압축 오버헤드 없음.
  64 토큰마다 한번 압축 → amortized 비용 = 0.17ms/token

- **비동기 압축**: 잔여 버퍼 flush는 I/O 스레드에서 수행,
  다음 레이어 연산과 겹침

#### R-Z3: 해제 시 전처리 역변환 오버헤드 (Severity: LOW)

**문제**: 바이트 재배치 역변환 (unshuffle) 추가 CPU 비용.
4MB 데이터 memcpy 수준: ~1-2ms on ARM64.

**영향**: 해제 시간 증가 (LZ4 해제 0.7ms + unshuffle 1.5ms = ~2.2ms)

**대응**:
- **NEON SIMD 활용**: ARM64 NEON `vzip`/`vuzp` 명령으로 바이트 재배치 가속
  기대 처리량: ~8-10 GB/s → 4MB 처리 ~0.4-0.5ms
- **파이프라인으로 숨김**: 해제+역변환 총 ~1.2ms < 레이어 연산 ~3ms → 여유 있음

#### R-Z4: LZ4 크레이트 edge case (Severity: LOW)

**문제**: `lz4` 크레이트의 알려진 이슈들:
- 빈 입력(0바이트) 시 panic (Issue #55)
- `uncompressed_size`가 `i32` → ~2GB 제한
- `prepend_size` 포맷이 비표준

**대응**:
- 빈 입력 가드: `if data.is_empty() { return Ok(()); }`
- 레이어당 데이터는 최대 수십 MB → i32 한계 내
- `prepend_size: false` 사용, 크기는 메타데이터로 별도 관리

#### R-Z5: 레이어별 압축률 편차 (Severity: LOW)

**문제**: 모델의 초기 레이어와 후기 레이어에서 KV 값 분포가 다르므로
압축률이 레이어마다 다를 수 있음 (1.5x ~ 2.5x 범위).

**영향**: 메모리 사용량 예측 어려움, 최악의 경우 절감 효과 미미

**대응**:
- 레이어별 압축률 통계 수집
- 압축률이 낮은 레이어(< 1.3x)는 미압축 유지 (오버헤드만 추가되므로)
- 적응형 정책: 런타임에 레이어별 압축 여부 결정

### 3.3 Common Risks (공통)

#### R-C1: 파이프라인 데드락 (Severity: HIGH)

**문제**: I/O 스레드와 연산 스레드 간 mpsc 채널 동기화에서 데드락 가능.
예: I/O 스레드가 버퍼 스왑 대기 중 + 연산 스레드가 로드 완료 대기 중 → 교착.

**대응**:
- **단방향 흐름**: 연산 스레드 → (request) → I/O 스레드 → (result) → 연산 스레드.
  양방향 의존성 제거
- **Timeout 기반 수신**: `recv_timeout(Duration::from_millis(50))` 사용.
  타임아웃 시 동기 로드로 fallback
- **상태 머신 설계**: I/O 스레드를 명확한 상태 머신으로 구현
  ```
  Idle → Loading(layer_id) → Loaded → Storing(layer_id) → Idle
  ```
- **유닛 테스트**: 스트레스 테스트 (100회 반복 + 인위적 지연 삽입)

#### R-C2: 레이턴시 편차로 인한 파이프라인 스톨 (Severity: MEDIUM)

**문제**: 파이프라인은 I/O ≤ Compute를 가정하나, 실제로는:
- DiskStore: I/O 스파이크 (GC 등)
- ZramStore: 해제+역변환 시간이 짧은 레이어 연산보다 길 수 있음
- 초기 레이어(seq가 짧을 때)는 연산이 매우 빠름

**영향**: 파이프라인 효과 감소, 실질 오버헤드 증가

**대응**:
- **청크 프리페치 (Section 2.4)**: 여러 레이어를 묶어서 프리페치.
  chunk_size=4면 4개 레이어의 연산 시간 동안 다음 4개를 로드.
  개별 I/O 지터가 청크 내에서 평균화됨
- **적응형 chunk_size**: 런타임 I/O/Compute EMA 비율 기반 자동 조정.
  스톨 비율이 높으면 chunk_size 증가, 낮으면 감소 (메모리 반환)
- **스톨 카운터**: 최대 chunk_size에서도 연속 스톨 N회 이상이면
  오프로드 비활성화 + 인메모리 복귀 고려
- **통계 기반 조정**: `PipelineStats`의 io_ema_ms/compute_ema_ms 추적

#### R-C3: Score Accumulator 비호환 (Severity: MEDIUM)

**문제**: H2O/H2O+ eviction은 `score_accumulator`를 통해 레이어별 attention score를
캡처한다. 오프로드 모드에서 KV 캐시가 파이프라인 버퍼에 있으므로
`kv_caches[i].current_pos()` 접근이 기존과 다를 수 있음.

**대응**:
- `OffloadKVCache`가 `current_pos`를 메타데이터로 항상 유지 (로드 상태 무관)
- Score accumulator는 KV 캐시 내용이 아닌 `current_pos`만 참조하므로 호환 가능
- 테스트: H2O eviction + offload 조합 시나리오 검증

#### R-C4: CacheManager (Eviction) 비호환 (Severity: HIGH)

**문제**: 기존 `CacheManager::maybe_evict()`는 `&mut [KVCache]`를 직접 받아
`prune_prefix()`를 호출한다. `OffloadKVCache`는 다른 타입이므로
기존 eviction 정책과 직접 연동 불가.

**대응**:
- **Phase 1에서는 eviction 미지원**: 오프로드 모드에서는 eviction 비활성화.
  max_seq_len까지만 사용, 초과 시 오류
- **Phase 2 (향후)**: `CacheManager`를 제네릭화하거나
  `OffloadKVCache`에 `prune_prefix()` 구현
- **문서화**: 오프로드 모드의 제약사항 명시

#### R-C5: Migration 실패 시 복구 (Severity: MEDIUM)

**문제**: 인메모리 → 오프로드 마이그레이션 도중 실패 (디스크 풀, 메모리 부족 등).
일부 레이어만 오프로드된 불완전 상태.

**대응**:
- **트랜잭션 패턴**: 마이그레이션은 전체 성공 또는 전체 롤백
- **2단계 커밋**:
  1. 모든 레이어를 저장소에 기록 (원본 유지)
  2. 전체 성공 확인 후 원본 메모리 해제
  3. 어느 단계에서든 실패 시 저장소 정리 + 원본 유지
- **부분 오프로드 허용 (대안)**: 성공한 레이어만 오프로드, 나머지는 인메모리 유지.
  파이프라인은 혼합 모드 지원

#### R-C6: F32 지원 시 I/O 병목 (Severity: MEDIUM)

**문제**: F32는 F16의 2배 데이터. 레이어당 8MB (seq=2048).
DiskStore: 읽기 ~5.3ms > 레이어 연산 ~3ms → **I/O가 병목**.
ZramStore: 해제 ~1.3ms, 전처리 역변환 ~1ms → 총 ~2.3ms, 여유 있음.

**대응**:
- **F32 DiskStore + 청크 프리페치**: chunk_size=4로 고정 오버헤드 절감.
  순차 I/O 효율 향상 (32MB @ ~1.7 GB/s)으로 레이어당 실효 I/O 시간 감소.
  그래도 I/O bound이므로 ~40-50% 오버헤드는 불가피 (Section 5.6 참조)
- **F32 ZramStore**: 문제 없음 (해제 시간 < 연산 시간). F32에서는 ZramStore 권장
- **적응형**: F32 DiskStore 시 adaptive chunk_size가 자동으로 4로 증가
- **사용자 가이드**: F16 사용 권장, F32 DiskStore는 성능 저하 문서화

### 3.4 Risk Summary Matrix

| ID | 리스크 | 심각도 | 발생 확률 | 대응 전략 |
|----|--------|--------|-----------|-----------|
| R-D1 | 디스크 I/O 지터 | HIGH | HIGH | Timeout + fallback + 모니터링 |
| R-D2 | 플래시 마모 | MEDIUM | MEDIUM | tmpfs 우선 + 배치 fsync + 쓰기 제한 |
| R-D3 | 파일 시스템 오버헤드 | LOW | LOW | 단일 파일 + 오프셋 |
| R-D4 | 저장 공간 부족 | MEDIUM | LOW | 사전 검사 + graceful fallback |
| R-D5 | 마이그레이션 지연 | MEDIUM | HIGH | 비동기/Lazy 마이그레이션 |
| R-Z1 | **float 비압축성** | **CRITICAL** | **CERTAIN** | **바이트 재배치 필수** + 압축률 gate |
| R-Z2 | 압축 오버헤드 | MEDIUM | HIGH | 잔여 버퍼 (amortized) + 비동기 |
| R-Z3 | 전처리 역변환 비용 | LOW | MEDIUM | NEON SIMD 가속 |
| R-Z4 | LZ4 edge case | LOW | LOW | 입력 가드 + 크기 관리 |
| R-Z5 | 레이어별 압축률 편차 | LOW | MEDIUM | 적응형 압축 정책 |
| R-C1 | **파이프라인 데드락** | **HIGH** | MEDIUM | 단방향 흐름 + timeout + 상태 머신 |
| R-C2 | 레이턴시 편차 | MEDIUM | HIGH | **청크 프리페치 + 적응형 chunk_size** |
| R-C3 | Score accumulator 비호환 | MEDIUM | LOW | 메타데이터 분리 유지 |
| R-C4 | **Eviction 비호환** | **HIGH** | CERTAIN | Phase 1 미지원 + Phase 2 제네릭화 |
| R-C5 | 마이그레이션 실패 | MEDIUM | LOW | 트랜잭션 패턴 (전체 성공/롤백) |
| R-C6 | F32 I/O 병목 | MEDIUM | HIGH | **청크 프리페치 C=4 (−32%p)** / ZramStore 권장 |
| R-C7 | 청크 크기 oscillation | LOW | MEDIUM | 감소는 3회 안정 후 적용 + EMA 평활화 |

## 4. Implementation Plan

### Phase 1: Common Infrastructure (공통 인프라)

| Step | 내용 | 파일 | 리스크 대응 |
|------|------|------|------------|
| 1-1 | `OffloadStore` trait 정의 | `core/offload/store.rs` | — |
| 1-2 | `OffloadKVCache` + `KVCacheOps` impl | `core/offload/mod.rs` | R-C3 (current_pos 메타) |
| 1-3 | `ChunkConfig` + `PrefetchPipeline` (청크 더블 버퍼 + I/O 스레드) | `core/offload/pipeline.rs` | R-C1 (상태 머신 + timeout) |
| 1-4 | `PipelineStats` + 적응형 chunk_size 조정 로직 | `core/offload/pipeline.rs` | R-C2, R-C7 (EMA + 안정성 규칙) |
| 1-5 | Migration: KVCache → OffloadKVCache 변환 | `core/offload/mod.rs` | R-C5 (트랜잭션 패턴) |
| 1-6 | `forward_into_offload()` (청크 기반 루프) | `models/llama/llama_model.rs` | R-C2 (적응형 청크) |
| 1-7 | 유닛 테스트: 파이프라인 동기화, 데드락 방지, 청크 경계 | 각 모듈 `#[cfg(test)]` | R-C1 (스트레스 테스트) |

### Phase 2A: DiskStore (디스크 오프로드)

| Step | 내용 | 파일 | 리스크 대응 |
|------|------|------|------------|
| 2A-1 | `DiskStore` 구조체 (단일 파일 + 레이어 오프셋) | `core/offload/disk_store.rs` | R-D3 |
| 2A-2 | `store()`: raw 데이터 → 파일 sequential write | | R-D2 (배치 fsync) |
| 2A-3 | `load_into()`: 파일 → 사전 할당 버퍼 read | | R-D1 (timeout) |
| 2A-4 | `append_token()`: 파일 끝 append (token_bytes) | | R-D2 |
| 2A-5 | 공간 사전 검사 + fallback 로직 | | R-D4 |
| 2A-6 | tmpfs 감지 및 우선 사용 로직 | | R-D2 |
| 2A-7 | 테스트: roundtrip, append, I/O 스파이크 시뮬레이션 | | R-D1 |

### Phase 2B: ZramStore (압축 메모리)

| Step | 내용 | 파일 | 리스크 대응 |
|------|------|------|------------|
| 2B-1 | F16/F32 바이트 재배치 (shuffle/unshuffle) | `core/offload/preprocess.rs` | **R-Z1** (필수) |
| 2B-2 | NEON SIMD 바이트 재배치 최적화 | `core/offload/preprocess.rs` | R-Z3 |
| 2B-3 | `ZramStore` 구조체 (압축 블록 + 잔여 버퍼) | `core/offload/zram_store.rs` | R-Z2 (잔여 버퍼) |
| 2B-4 | `store()`: 재배치 → LZ4 압축 → Vec<u8> | | R-Z4 (입력 가드) |
| 2B-5 | `load_into()`: LZ4 해제 → 역재배치 → raw 복원 | | R-Z3 |
| 2B-6 | `append_token()`: 잔여 버퍼 + 배치 재압축 | | R-Z2 |
| 2B-7 | **압축률 검증 gate**: 실제 KV 데이터로 측정 | | **R-Z1** (1.5x 미만 시 재검토) |
| 2B-8 | 적응형 레이어별 압축 정책 | | R-Z5 |
| 2B-9 | Cargo.toml에 `lz4 = "1.28"` 추가 | `engine/Cargo.toml` | R-Z4 (Android 빌드 확인) |
| 2B-10 | 테스트: roundtrip bit-exact, 압축률, 성능 | | R-Z1, R-Z2 |

### Phase 3: Integration (통합)

| Step | 내용 | 파일 | 리스크 대응 |
|------|------|------|------------|
| 3-1 | CLI `--kv-offload disk\|zram` 옵션 | `bin/generate.rs` | — |
| 3-2 | 이벤트 기반 마이그레이션 트리거 | `bin/generate.rs` | R-C5, R-D5 |
| 3-3 | 벤치마크: 메모리, tok/s, P50/P99 레이턴시 | | R-D1, R-C2 |
| 3-4 | 문서: 사용 가이드 + 제약사항 | `docs/` | R-C4 |

### Phase 4: Testing

| 테스트 | 검증 내용 | 대응 리스크 |
|--------|---------|------------|
| `test_store_roundtrip_{disk,zram}` | store→load 데이터 일치 | 기본 정확성 |
| `test_preprocess_roundtrip` | shuffle→unshuffle bit-exact | R-Z1 |
| `test_compression_ratio` | ZramStore 압축률 ≥ 1.5x (abort gate) | **R-Z1** |
| `test_pipeline_no_deadlock` | 100회 반복 + 인위적 지연 (C=1,2,4) | **R-C1** |
| `test_pipeline_chunk_boundary` | 16 layers를 C=3,5,7로 불균등 분할 시 정확성 | R-C2 |
| `test_pipeline_adaptive_sizing` | I/O 느릴 때 C 증가, 빨라지면 C 감소 확인 | R-C2, R-C7 |
| `test_pipeline_timeout_fallback` | I/O 지연 시 동기 fallback | R-D1, R-C2 |
| `test_migration_rollback` | 마이그레이션 실패 시 원본 유지 | R-C5 |
| `test_migration_partial` | 부분 마이그레이션 후 혼합 모드 | R-C5 |
| `test_offload_kvcache_ops` | KVCacheOps trait 계약 준수 | R-C3 |
| `test_offload_f16_f32` | F16/F32 모두 정확한 오프로드 | R-C6 |
| `test_append_decode_loop` | 256 토큰 연속 decode 정확성 | 통합 |
| `test_disk_space_fallback` | 공간 부족 시 graceful 동작 | R-D4 |
| `test_empty_input_guard` | 0 토큰 KV 캐시 처리 | R-Z4 |

## 5. Timing Analysis

### 5.1 DiskStore Pipeline (F16, Llama 3.2 1B, seq=2048)

```
레이어당 KV 크기: 2048 × 8 × 64 × 2B × 2(K+V) = 4 MB
Android UFS 3.1 sequential read: ~1.5 GB/s
Android UFS 3.1 sequential write: ~0.8 GB/s
고정 오버헤드: ~0.5ms/sync

  Read:  4 MB / 1.5 GB/s = 2.7 ms
  Write: 2 KB / 0.8 GB/s ≈ 0 ms (append 1 token)
  Compute per layer: ~3 ms (decode, single token)

chunk_size=1 (I/O < Compute → 최적):
  L0 load (2.7ms) → [L0 compute (3ms) | L1 load (2.7ms)] → ...
  Total: 2.7 + 15 × (3.0 + 0.5) = 2.7 + 52.5 = 55.2 ms/token
  Baseline: 48 ms/token
  Overhead: ~15%
  → C=1이 이미 최적 (I/O가 compute보다 짧으므로 chunk 불필요)

chunk_size=2 (고정 오버헤드 절감):
  청크 I/O: 8 MB / 1.5 GB/s = 5.3 ms
  청크 연산: 2 × 3 ms = 6 ms (여전히 compute bound)
  Total: 5.3 + 7 × (6.0 + 0.5) = 5.3 + 45.5 = 50.8 ms/token
  Overhead: ~6% ✓
  → sync 횟수 절반으로 고정 오버헤드 절감 효과
```

### 5.2 DiskStore Pipeline (F32, seq=2048, chunk_size=1)

```
레이어당 KV 크기: 8 MB
  Read:  8 MB / 1.5 GB/s = 5.3 ms  ← 연산(3ms)보다 느림!
  고정 오버헤드: ~0.5ms/sync (스레드 wake + 채널)

Pipeline (I/O bound, C=1):
  Total: 5.3 + 15 × (5.3 + 0.5) = 5.3 + 87.0 = 92.3 ms/token
  Overhead: ~92% ✗
```

### 5.2.1 DiskStore Pipeline (F32, 청크 프리페치 적용)

```
chunk_size=4:
  청크 I/O: 32 MB / 1.7 GB/s = 18.8 ms (대용량 순차 읽기로 처리량 향상)
  청크 연산: 4 × 3 ms = 12 ms
  고정 오버헤드: ~0.5ms × 4 sync = 2.0 ms

  Total: 18.8 + 3 × (18.8 + 0.5) = 18.8 + 57.9 = 76.7 ms/token
  Overhead: ~60% (C=1의 92%에서 32%p 개선)
  버퍼 메모리: 2 × 4 × 8 MB = 64 MB

chunk_size=2:
  청크 I/O: 16 MB / 1.6 GB/s = 10.0 ms
  청크 연산: 2 × 3 ms = 6 ms

  Total: 10.0 + 7 × (10.0 + 0.5) = 10.0 + 73.5 = 83.5 ms/token
  Overhead: ~74%
  버퍼 메모리: 2 × 2 × 8 MB = 32 MB

비교 (F32 DiskStore):
  C=1: ~92%, 버퍼 16 MB
  C=2: ~74%, 버퍼 32 MB  (−18%p)
  C=4: ~60%, 버퍼 64 MB  (−32%p)
  → F32 DiskStore는 C=4에서도 상당한 오버헤드.
  → F32에서는 ZramStore가 적합 (Section 5.4: 오버헤드 ~5.4%)
```

### 5.3 ZramStore Pipeline (F16, seq=2048)

```
압축률 ~2.0x 가정 (바이트 재배치 + LZ4)
  압축 데이터: 4 MB / 2.0 = 2 MB
  LZ4 해제: 2 MB / 3.0 GB/s = 0.67 ms
  Unshuffle: 4 MB / 8 GB/s (NEON) = 0.5 ms
  Total load: ~1.2 ms

Pipeline:
  Total: 1.2 + 16 × 3.0 = 49.2 ms/token
  Overhead: ~2.5% ✓
```

### 5.4 ZramStore Pipeline (F32, seq=2048)

```
압축률 ~1.7x 가정 (F32는 F16보다 압축률 낮음)
  원본: 8 MB, 압축: ~4.7 MB
  LZ4 해제: 4.7 MB / 3.0 GB/s = 1.6 ms
  Unshuffle: 8 MB / 8 GB/s = 1.0 ms
  Total load: ~2.6 ms

Pipeline:
  Total: 2.6 + 16 × 3.0 = 50.6 ms/token
  Overhead: ~5.4% ✓  ← DiskStore F32 (88%)보다 훨씬 양호
```

### 5.5 Summary (chunk_size별 비교)

| 구성 | C | 오버헤드 | 버퍼 메모리 | 메모리 절약 | 판정 |
|------|---|---------|-----------|-----------|------|
| DiskStore F16 | 1 | ~15% | 8 MB | 56 MB | ✓ 양호 |
| DiskStore F16 | 2 | ~6% | 16 MB | 48 MB | ✓ 최적 (추천) |
| DiskStore F32 | 1 | ~92% | 16 MB | 112 MB | ✗ 느림 |
| DiskStore F32 | 4 | ~60% | 64 MB | 64 MB | △ 개선되나 여전히 느림 |
| ZramStore F16 | 1 | ~2.5% | 8 MB | ~30 MB | ✓ 최적 |
| ZramStore F32 | 1 | ~5.4% | 16 MB | ~50 MB | ✓ 양호 |
| (참고) KIVI Q2 | — | ~0% | — | 61.8 MB | ✓ 최적 (단, 손실) |

**핵심 인사이트**:
- **I/O ≤ Compute (F16 Disk, ZramStore 전체)**: C=1~2가 최적. 메모리 절약 극대화.
- **I/O > Compute (F32 DiskStore)**: C=4로 32%p 개선되나 근본적 한계 존재. → **ZramStore 권장**.
- 적응형 chunk_size는 이 결정을 런타임에 자동으로 수행.

## 6. Decision Log

| 결정 | 근거 |
|------|------|
| F16/F32 모두 지원 | 기존 코드에서 이미 F16/F32 KV 캐시 완전 지원 (`--kv-type f16/f32`) |
| `lz4 = "1.28"` 사용 | C 바인딩 (cc 크레이트 기반), Android ARM64 크로스 컴파일 검증됨, cmake 불필요 |
| Prefill은 인메모리 유지 | 이벤트 기반 오프로드 — 메모리 압박 시에만 마이그레이션 |
| Phase 1에서 eviction 미지원 | CacheManager 제네릭화 필요, 별도 작업으로 분리 |
| 바이트 재배치 필수 (ZramStore) | raw float LZ4 압축률 ~1.0x, 전처리 없이 무의미 |
| 잔여 버퍼 패턴 (ZramStore) | 매 토큰 재압축 방지, KiviCache 패턴 검증됨 |
| std::thread + mpsc | tokio 금지 제약, 파이프라인에 충분한 동기화 |
| 청크 프리페치 (적응형) | I/O > Compute 시 고정 오버헤드 분산 + 순차 I/O 효율. F32 DiskStore에서 92%→60% 개선 |
| chunk_size 상한 = num_layers/2 | 그 이상이면 버퍼가 전체 인메모리와 동일 → 오프로드 의미 없음 |
| 적응형 감소는 3회 안정 후 | chunk_size oscillation 방지, EMA α=0.3으로 평활화 |
