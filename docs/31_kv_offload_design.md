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
  - 더블 버퍼 (2 레이어분 F16/F32 데이터)
  - Layer N 연산 중 Layer N+1 비동기 로드
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
pub struct PrefetchPipeline {
    num_layers: usize,
    /// 더블 버퍼 (각각 1 레이어분 K + V)
    buf_a_k: Vec<u8>,
    buf_a_v: Vec<u8>,
    buf_b_k: Vec<u8>,
    buf_b_v: Vec<u8>,
    /// 현재 어떤 버퍼가 활성인지
    active_buf: BufferSlot, // A or B
    /// I/O 스레드 통신 (std::sync::mpsc)
    request_tx: Sender<PipelineCmd>,
    result_rx: Receiver<PipelineResult>,
    io_thread: Option<JoinHandle<()>>,
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
- **적응형 프리페치 깊이**: 기본 1-레이어 ahead.
  스톨 감지 시 2-레이어 ahead로 확장 (트리플 버퍼 활성화)
- **스톨 카운터**: 연속 스톨 N회 이상이면 오프로드 비활성화 고려
- **통계 기반 조정**: 런타임 중 I/O 시간과 연산 시간 EMA 추적

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
- **F32 DiskStore**: 파이프라인 깊이를 2-ahead로 설정 (트리플 버퍼)
  또는 I/O가 병목임을 수용하고 처리량 저하 문서화
- **F32 ZramStore**: 문제 없음 (해제 시간 < 연산 시간)
- **사용자 가이드**: F16 사용 권장, F32는 ZramStore에 더 적합하다고 안내

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
| R-C2 | 레이턴시 편차 | MEDIUM | HIGH | 적응형 프리페치 깊이 |
| R-C3 | Score accumulator 비호환 | MEDIUM | LOW | 메타데이터 분리 유지 |
| R-C4 | **Eviction 비호환** | **HIGH** | CERTAIN | Phase 1 미지원 + Phase 2 제네릭화 |
| R-C5 | 마이그레이션 실패 | MEDIUM | LOW | 트랜잭션 패턴 (전체 성공/롤백) |
| R-C6 | F32 I/O 병목 | MEDIUM | HIGH | 트리플 버퍼 / ZramStore 권장 |

## 4. Implementation Plan

### Phase 1: Common Infrastructure (공통 인프라)

| Step | 내용 | 파일 | 리스크 대응 |
|------|------|------|------------|
| 1-1 | `OffloadStore` trait 정의 | `core/offload/store.rs` | — |
| 1-2 | `OffloadKVCache` + `KVCacheOps` impl | `core/offload/mod.rs` | R-C3 (current_pos 메타) |
| 1-3 | `PrefetchPipeline` (더블 버퍼 + I/O 스레드) | `core/offload/pipeline.rs` | R-C1 (상태 머신 + timeout) |
| 1-4 | Migration: KVCache → OffloadKVCache 변환 | `core/offload/mod.rs` | R-C5 (트랜잭션 패턴) |
| 1-5 | `forward_into_offload()` 메서드 | `models/llama/llama_model.rs` | R-C2 (적응형 프리페치) |
| 1-6 | 유닛 테스트: 파이프라인 동기화, 데드락 방지 | 각 모듈 `#[cfg(test)]` | R-C1 (스트레스 테스트) |

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
| `test_pipeline_no_deadlock` | 100회 반복 + 인위적 지연 | **R-C1** |
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

  Read:  4 MB / 1.5 GB/s = 2.7 ms
  Write: 2 KB / 0.8 GB/s ≈ 0 ms (append 1 token)
  Compute per layer: ~3 ms (decode, single token)

Pipeline:
  L0 load (2.7ms) → [L0 compute (3ms) | L1 load (2.7ms)] → ...

  Total: 2.7 + 16 × 3.0 = 50.7 ms/token
  Baseline (in-memory): 48 ms/token
  Overhead: ~6% ✓
```

### 5.2 DiskStore Pipeline (F32, seq=2048)

```
레이어당 KV 크기: 8 MB
  Read:  8 MB / 1.5 GB/s = 5.3 ms  ← 연산(3ms)보다 느림!

Pipeline (I/O bound):
  Total: 5.3 + 16 × 5.3 = 90.1 ms/token
  Overhead: ~88% ✗

Triple buffer (2-ahead prefetch):
  Total: ~5.3 + 16 × max(5.3, 3+2.7) = ~90 ms (여전히 I/O bound)
  → F32 DiskStore는 성능 저하 불가피. ZramStore 권장.
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

### 5.5 Summary

| 구성 | 오버헤드 | 메모리 절약 | 판정 |
|------|---------|-----------|------|
| DiskStore F16 | ~6% | 56 MB | ✓ 실용적 |
| DiskStore F32 | ~88% | 112 MB | ✗ 느림 |
| ZramStore F16 | ~2.5% | ~30 MB | ✓ 최적 |
| ZramStore F32 | ~5.4% | ~50 MB | ✓ 양호 |
| (참고) KIVI Q2 | ~0% | 61.8 MB | ✓ 최적 (단, 손실) |

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
