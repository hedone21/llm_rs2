# Chapter 33: SVD Cache

> **이전**: [32. KV 캐시 오프로드](32_kv_offload.md)

SVD 캐시는 K 캐시를 SVD(특이값 분해) 기반 저랭크 근사로 압축하고, V 캐시를 무손실 RawStore로 오프로드하는 하이브리드 전략입니다.

---

## 33.1 Overview

### 동기

KV 캐시에서 K와 V는 압축 특성이 다릅니다:

| 속성 | K 캐시 | V 캐시 |
|------|--------|--------|
| 코사인 유사도 (rank=10) | **0.93** | 0.73 |
| 저랭크 근사 효과 | 높음 | 낮음 |
| 선택 전략 | SVD 압축 (손실) | RawStore 오프로드 (무손실) |

### 핵심 아이디어

```
K: [seq_len × head_dim] → SVD → basis [k × head_dim] + coeffs [seq_len × k]
   → 메모리: seq_len × head_dim → seq_len × k + k × head_dim (k << head_dim 시 절감)
   → 품질: CosSim ≥ 0.93 (rank_k=10)

V: [seq_len × head_dim] → RawStore (무손실, 레이어별 프리페치)
```

---

## 33.2 아키텍처

### SvdOffloadKVCache 구조체

```rust
pub struct SvdOffloadKVCache {
    layer_id: usize,
    kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    current_pos: usize,
    config: SvdConfig,              // rank_k (기본 10)

    // K: SVD 압축 (per-head)
    k_heads: Vec<SvdHeadState>,     // 헤드별 기저+계수
    attn_k_buf: Vec<f32>,           // 증분 재구성 버퍼 [max_seq × kv_heads × head_dim]
    k_reconstructed_tokens: usize,  // 재구성 완료된 토큰 수

    // V: RawStore 오프로드
    v_store: VStoreAdapter,         // OffloadStore 어댑터
    attn_v_buf: Option<Vec<u8>>,    // Lazy V 프리로드 버퍼
    v_preloaded: bool,
    v_token_bytes: usize,           // kv_heads × head_dim × sizeof(F16)

    // Prefill K 임시 버퍼
    prefill_k_f32: Option<Vec<f32>>,
    prefill_tokens: usize,

    // 출력 버퍼 재사용
    out_k_buf: Option<Arc<SharedBuffer>>,
    out_v_buf: Option<Arc<SharedBuffer>>,
    out_backend: Arc<CpuBackend>,
}
```

### SvdHeadState (per-head)

```rust
struct SvdHeadState {
    basis: Vec<f32>,             // [k × head_dim] 기저 벡터
    coeffs: Vec<f32>,            // [compressed_tokens × k] 사영 계수
    compressed_tokens: usize,    // 계수가 계산된 토큰 수
    actual_k: usize,             // 실제 달성 랭크 (≤ config.rank_k)
}
```

### VStoreAdapter

`OffloadStore`는 K+V 쌍을 기대하므로, V만 저장하는 어댑터:

```rust
struct VStoreAdapter {
    store: Box<dyn OffloadStore>,    // 실제 RawStore
    dummy_k: Vec<u8>,                // K 대신 넣는 0 바이트
    token_bytes: usize,
}
```

---

## 33.3 압축 파이프라인

### Prefill 단계

```
입력: F16 K/V 텐서
    ↓
K: F16 → f32 변환 → prefill_k_f32에 누적
V: F16 원본 → v_store.store_v()
    ↓
prefill 완료 시: compute_basis_from_prefill()
```

### 기저 계산 (`compute_basis_from_prefill`)

각 헤드에 대해:

```
1. 헤드별 데이터 추출: data[seq_len × head_dim]
2. Gram 행렬: A^T A = data^T × data  [head_dim × head_dim]
3. 고유분해: svd_eigen_f32(A^T A, head_dim, rank_k)
   → top-k 고유값/고유벡터 (Power iteration, 100회)
4. 기저 저장: basis = eigenvectors [actual_k × head_dim]
5. 모든 prefill 토큰 사영: coeffs[t] = dot(token[t], basis[j]) for j in 0..k
```

### Decode 단계

```
새 토큰 (F16 → f32):
    K: project_decode_token(k_f32)
       → 각 헤드: coeffs[t] = dot(token, basis[j]) for j in 0..k
       → compressed_tokens += 1
    V: v_store.append_v_token(v_f16)
       → v_preloaded면 attn_v_buf에도 추가
```

### 재구성 (`get_view` 시)

```
K: reconstruct_k_incremental()
   → k_reconstructed_tokens..current_pos까지 증분 재구성
   → out[pos, head, d] = Σ(coeffs[pos, j] × basis[head, j, d])
   → f32 → F16 변환
   → SharedBuffer에 복사

V: preloaded면 attn_v_buf에서, 아니면 v_store.load_v_into()
   → SharedBuffer에 복사
```

---

## 33.4 SVD 수학 (`svd_math.rs`)

### 주요 함수

| 함수 | 시그니처 | 용도 |
|------|---------|------|
| `compute_gram_matrix` | `(data: &[f32], num_rows, dim) → Vec<f32>` | A^T A 계산 |
| `svd_eigen_f32` | `(ata: &[f32], n, max_k) → (eigenvalues, eigenvectors)` | Top-k 고유분해 |
| `project_token` | `(token: &[f32], basis, k, dim) → Vec<f32>` | 단일 토큰 사영 |
| `reconstruct_into` | `(basis, coeffs, ..., out: &mut [f32])` | 계수×기저→토큰 재구성 |

### Power Iteration 알고리즘

```
for eigenvalue 1..max_k:
    v = random_init
    for iter 1..100:
        v' = A × v
        v = v' / ||v'||    (정규화)
    λ = v^T A v            (고유값)
    if λ < λ_max × 1e-4:   (상대 임계값)
        break
    A ← A − λ v v^T        (편향: deflation)
```

---

## 33.5 PrefetchableCache 구현

SvdOffloadKVCache는 `PrefetchableCache`를 구현하여 `forward_into_offload()` 경로와 호환됩니다:

| 메서드 | 동작 |
|--------|------|
| `preload()` | V 데이터를 v_store에서 attn_v_buf로 로드 (K는 항상 메모리 내) |
| `release_buffers()` | attn_v_buf 해제 + v_preloaded 리셋 |
| `reset_preload()` | v_preloaded 플래그 리셋 |
| `retain_preload()` | attn_v_buf가 존재하면 v_preloaded 재설정 |

K 캐시는 SVD 계수+기저로 항상 메모리에 있으므로 프리페치 대상이 아닙니다. V 캐시만 preload/release 대상입니다.

---

## 33.6 CLI 사용법

```bash
cargo run --release --bin generate -- \
  --model-path models/llama3.2-1b \
  --kv-offload svd \                # SVD 모드
  --svd-rank 10 \                   # SVD 랭크 (기본 10)
  --max-prefetch-depth 4 \          # V 프리페치 깊이
  --kv-type f16 \                   # F16 필수
  --kv-layout seq \                 # SeqMajor 필수
  --prompt "Hello" -n 128
```

### 제약사항

| 조건 | 이유 |
|------|------|
| F16 입력 전용 | 내부적으로 F16→f32 변환 (F32 입력 미구현) |
| SeqMajor 전용 | V 오프로드와 K 재구성 레이아웃 호환 |
| Eviction 비호환 | SVD 기저는 eviction 후 무효화 |

---

## 33.7 메모리 예산

### Llama 3.2 1B (F16, 16 layers, rank_k=10, seq=2048)

**K 캐시 (per layer, per head):**

| 구성요소 | 크기 |
|---------|------|
| basis: [10 × 64] × f32 | 2,560 bytes |
| coeffs: [2048 × 10] × f32 | 81,920 bytes |
| **합계 (per head)** | **84,480 bytes** |
| **합계 (8 heads × 16 layers)** | **~10.3 MB** |

**비교 (F16 원본 K):**
- 원본: 2048 × 8 × 64 × 2 = 2 MB per layer, 32 MB total
- SVD: ~10.3 MB → **~68% 절감**

**V 캐시:** RawStore에 원본 저장 (32 MB) + 활성 attn_v_buf 2개 레이어 (~4 MB)

**총 메모리:**

| 구성요소 | 크기 |
|---------|------|
| K SVD 데이터 | ~10.3 MB |
| K 재구성 버퍼 (attn_k_buf) | ~2 MB (전 레이어 공유) |
| V RawStore | 32 MB |
| V attn_v_buf (2 layers) | ~4 MB |
| 출력 버퍼 | ~4 MB |
| **합계** | **~52 MB** |

표준 KVCache (64 MB) 대비 **~19% 절감**.

---

## 33.8 품질 특성

### 코사인 유사도 (rank_k별)

| rank_k | K CosSim | 메모리 절감 | 비고 |
|--------|---------|-----------|------|
| 5 | ~0.85 | ~84% | 저품질 |
| 10 | **~0.93** | **~68%** | 기본값, 균형점 |
| 20 | ~0.97 | ~37% | 고품질 |
| 64 (full) | 1.00 | 0% | 무손실 (K 원본과 동일) |

> **주의**: rank_k가 head_dim(64)에 근접하면 SVD의 메모리 이점이 사라집니다.
> rank_k=10은 품질(0.93)과 절감(68%)의 균형점입니다.

### V 캐시

V는 RawStore 오프로드이므로 **완전 무손실** (bit-exact)입니다.

---

## 33.9 테스트 커버리지

| 테스트 | 검증 |
|--------|------|
| `test_svd_cache_basic` | 생성 + dtype 확인 |
| `test_svd_cache_prefill_shape` | 프리필 후 형상 검증 |
| `test_svd_cache_prefill_then_decode` | 프리필+디코드 혼합 |
| `test_svd_cache_k_cosine_similarity` | K 재구성 CosSim > 0.9 |
| `test_svd_cache_v_lossless` | V 바이트 정확 왕복 |
| `test_svd_cache_memory_usage` | 메모리 사용량 계산 |
| `test_svd_cache_overflow` | 용량 초과 에러 |
| `test_svd_cache_preload_release` | preload/release 생명주기 |
| `test_svd_cache_k_lossless_full_rank` | full-rank SVD 무손실 확인 |
| `test_svd_cache_incremental_reconstruction` | 증분 재구성 정확성 |

### SVD 수학 테스트

| 테스트 | 검증 |
|--------|------|
| 대각 행렬 고유분해 | 알려진 고유값 정확성 |
| 사영-재구성 왕복 | CosSim > 0.999 |
| Gram 행렬 대칭성 | A^T A 대칭 확인 |

---

## 33.10 설계 결정 로그

| 결정 | 근거 |
|------|------|
| K만 SVD, V는 RawStore | K CosSim 0.93 vs V CosSim 0.73 — V는 저랭크 근사 비효과적 |
| Power iteration (Jacobi 아님) | 상위 k개만 필요 (전체 고유분해 불필요), 구현 단순 |
| F16→f32→F16 변환 | SVD 수학은 f32 정밀도 필요, 최종 출력은 F16으로 복원 |
| 증분 재구성 | get_view()마다 전체 재구성 대신 새 토큰만 재구성 (O(Δ) vs O(N)) |
| prefill 기저 계산 | decode 토큰만으로는 기저 품질 보장 불가, prefill 데이터가 충분 |
| VStoreAdapter | OffloadStore trait 재사용, K/V 분리 없이 기존 인터페이스 활용 |
| rank_k=10 기본값 | compress_lab 실험 결과 품질-메모리 최적 균형 |
