# Long Context CPU Attention 최적화 — 4K+ 성능 갭 해소

> **Status**: Ready for implementation
> **Priority**: P0 (High impact — 4K context에서 llama.cpp 대비 35% 수준)
> **Owner**: TBD (senior-implementer 권장 — NEON + 복잡한 numerical algorithm)
> **작성일**: 2026-04-13
> **마지막 측정**: 2026-04-13 Galaxy S25 (Snapdragon 8 Elite, Adreno 830)

---

## 1. 문제 정의

### 측정된 성능 갭

**환경**: Galaxy S25 / Snapdragon 8 Elite / CPU backend / 6 threads / Qwen2.5 1.5B Q4_0 (`qwen2.5-1.5b-q4_0-v2.gguf`)

| 컨텍스트 | llm_rs decode | llama.cpp decode | llm_rs/llama.cpp | ms/tok 갭 |
|---------|---------------|------------------|------------------|-----------|
| **Short (~20 tok)** | 22.6 tok/s | 29.98 tok/s (tg128) | **75%** | 44.3 vs 33.3 ms |
| **4K (4096~4472 tok)** | **10.6 tok/s** | **30.50 tok/s** (pp4096+tg128) | **35%** | **94.3 vs 32.8 ms** |

### 관찰된 특성

1. **llama.cpp는 context 길이에 거의 불변**: 29.98 → 30.50 (+1.7%, flat)
2. **llm_rs는 4K에서 심각한 열화**: 22.6 → 10.6 (-53%)
3. **4K context에서 attention overhead 약 50ms/tok** (94.3 − 44.3)
4. **Prefill도 O(n²) 문제**: 4472 tokens prefill 약 400초, 11.1 tok/s

### 재현 명령

디바이스에 GGUF + binary가 이미 배포되어 있음 (2026-04-13 기준):

```bash
# llama.cpp 4K 벤치마크
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./llama-bench \
  -m /data/local/tmp/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-v2.gguf \
  -pg 4096,128 -t 6 -ngl 0 -r 2"

# llm_rs 4K 벤치마크
adb shell "cd /data/local/tmp && ./generate \
  -m /data/local/tmp/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-v2.gguf \
  --prompt-file /data/local/tmp/prompt_6k.txt \
  -n 128 -b cpu --max-seq-len 6144 --threads 6 --ignore-eos"

# 4K prompt 파일 생성 (host에서)
cd /tmp && (for i in 1 2 3 4 5 6 7 8; do cat prompt_base.txt; echo; done) > prompt_6k.txt
adb push /tmp/prompt_6k.txt /data/local/tmp/prompt_6k.txt
```

**참고**: `/data/local/tmp/prompt_6k.txt`는 27KB, 3752 words, Qwen tokenizer로 약 4472 tokens.

---

## 2. 근본 원인 분석

### Qwen2.5 1.5B attention 연산량 (per layer, 4K context)

```
Q:  1 token × 12 Q-heads × 128 dim     = 1.5 KB
K:  4096 tokens × 2 KV-heads × 128 dim = 2.0 MB (F16)
V:  4096 tokens × 2 KV-heads × 128 dim = 2.0 MB (F16)

GQA ratio = 12 Q-heads / 2 KV-heads = 6
→ Q-heads 0~5는 KV-head 0 공유, Q-heads 6~11은 KV-head 1 공유
```

### 현재 구현의 병목 (`engine/src/backend/cpu/neon.rs:235-500` `attention_gen_f16_neon`)

**구조**: Rayon `par_chunks_mut` over Q-heads (12-way 병렬성, 6 코어)

```rust
out_data
    .par_chunks_mut(head_dim)
    .enumerate()
    .for_each(|(h, out_h)| {
        let kv_h = h / gqa_ratio;  // 6 Q-head가 같은 kv_h 참조
        // ... 3-pass attention ...
    });
```

**4가지 병목**:

1. **GQA 중복 읽기 (가장 심각)**
   - 6개 Q-head가 같은 KV-head를 **각자 로드**
   - 레이어당 KV 트래픽: `12 heads × (1MB K + 1MB V) = 24MB` (필요 최소: 4MB)
   - 28 레이어 × 24MB = **672MB/token DRAM 트래픽**
   - LPDDR5X 77GB/s 대역폭 → 이론치 8.7ms/token (BW-bound)
   - 실측 50ms → **cache thrashing + 중복 읽기**가 추가 40ms 원인

2. **L2 Cache 초과**
   - Snapdragon 8 Elite Oryon core: L1D 64KB, L2 2MB (shared)
   - 한 head working set: K(1MB) + V(1MB) = **2MB = L2 전체**
   - 6 thread 동시 실행 → L2 thrash → DRAM fallback

3. **3-Pass Softmax (KV 2회 스캔)**
   ```
   Pass 1: QK^T → scores 배열 전체 할당 (4K × 4B = 16KB)
   Pass 2: max → exp → sum → divide (scores 여러 번 순회)
   Pass 3: weighted V → V 전체 스캔
   ```
   - K와 V가 **다른 pass에서 읽혀** cache reuse 불가
   - Scores 배열이 L1에 fit하지만 불필요한 중간 저장

4. **Thread 활용 미흡**
   - 12 head / 6 core = 2 head per core (순차)
   - Head 처리 중 single-core 성능만 활용
   - Inter-core parallelism은 head 경계에서만 발생

### llama.cpp가 flat한 이유

llama.cpp는 **Flash Decoding** 스타일로 동작:
- KV sequence를 여러 thread에 **분할** (PR #19209, 2025)
- 각 thread: partial (max, sum_exp, weighted_v) 계산
- 최종 log-sum-exp reduction으로 합산
- Online softmax로 1-pass 처리
- Working set이 L1/L2에 fit → cache hit 극대화

---

## 3. 해법 — 3단계 구현 계획

### Step 1: Online Softmax (낮은 난이도, 중간 효과)

**수학적 동등성**: max, sum, output을 점진적으로 업데이트하고 새 max 발견 시 rescale

```python
m = -∞      # running max
l = 0       # running sum of exp
o = 0       # running weighted V

for t in range(n):
    s = Q @ K[t]
    m_new = max(m, s)
    rescale = exp(m - m_new)
    l = l * rescale + exp(s - m_new)
    o = o * rescale + V[t] * exp(s - m_new)
    m = m_new

output = o / l
```

**이점**:
- KV를 **한 번만 스캔** (K와 V 같은 루프 → spatial locality)
- `scores` 버퍼 제거 (L1 스택만 사용)
- Streaming 처리 가능 → Step 2(Flash Decoding)의 전제 조건

**수정 위치**: `engine/src/backend/cpu/neon.rs:235 attention_gen_f16_neon`

**Risk**: 낮음 — 수학적으로 동등, NaN 처리 주의 필요 (현재 코드의 NaN sanitization 로직 유지)

**예상 개선**: 4K decode 94ms → ~70ms (25-30% 향상)

### Step 2: Flash Decoding — KV Split 병렬화 (중간 난이도, 큰 효과)

**아이디어**: Head 병렬 대신 **KV chunk + head 2단계 병렬**

```
4K KV → 4 chunks × 1024 tokens
각 thread: 자기 chunk에서 partial (m_i, l_i, o_i) 계산
Reduction: global max로 rescale 후 합산

for chunk_idx in 0..4:  // parallel
    for head in 0..12:  // all heads, shared KV access
        (m[head, chunk], l[head, chunk], o[head, chunk]) = partial_attention(Q[head], K_chunk, V_chunk)

// Merge
for head in 0..12:
    M = max(m[head, :])
    L = sum(l[head, c] * exp(m[head, c] - M) for c in chunks)
    O = sum(o[head, c] * exp(m[head, c] - M) for c in chunks)
    out[head] = O / L
```

**이점**:
- **GQA 중복 읽기 제거**: 한 chunk의 KV를 12개 head가 **공유**
- **Cache locality**: Chunk 256KB (K+V) → L2 fit
- **DRAM 트래픽**: `12 heads × 4MB = 48MB/layer` → `2 KV-heads × 4MB = 8MB/layer` (**6배 감소**)
- **Core 활용**: KV chunk 병렬 + head inner loop → 6-core 포화

**수정 위치**: `engine/src/backend/cpu/neon.rs` — 새 함수 `attention_gen_f16_neon_flash()` 또는 기존 함수 리팩토링

**Risk**: 중간 — 병합 로직의 numerical stability, chunk size 튜닝 필요

**예상 개선**: 4K decode ~70ms → ~40ms (2x 향상)

**튜닝 포인트**:
- Chunk size: 512/1024/2048 실험 (L2 cache 2MB 기준 1024 권장)
- Chunk 수 임계값: 짧은 context에서는 head 병렬이 더 유리 → `if cache_seq_len < 1024: use_head_parallel()`
- F16 load: `vld1q_f16` / `vcvt_f32_f16` NEON intrinsics 활용

### Step 3: CPU Flash Attention for Prefill (중간-높은 난이도, 매우 큰 효과)

**문제**: Prefill은 현재 O(n²) — 4472 tokens prefill이 400초

**접근**: Tile 단위 QK^T + online softmax + weighted V

```
for tile_i in 0..(n / TILE_I):     // Q tiles
    for tile_j in 0..(n / TILE_J):  // K/V tiles
        S = Q[tile_i] @ K[tile_j]^T
        (m, l, o) = online_softmax_update(S, V[tile_j], previous_state)
```

**참고**: `engine/src/layers/attention.rs`에 prefill flash attention 경로가 이미 존재할 수 있음 — 확인 후 CPU 경로 확장

**수정 위치**:
- `engine/src/layers/attention.rs`
- `engine/src/backend/cpu/neon.rs` (or common.rs)

**Risk**: 중간-높음 — causal mask 처리, tile boundary 에지 케이스

**예상 개선**: 4K prefill **10x 이상** 속도 향상 (11 tok/s → 100+ tok/s)

---

## 4. 구현 전 체크리스트

### 관련 코드 파일

| 파일 | 역할 | 수정 범위 |
|------|------|-----------|
| `engine/src/backend/cpu/neon.rs:186` `attention_gen` (trait impl) | ARM64 NEON attention 진입점 | Step 1, 2 |
| `engine/src/backend/cpu/neon.rs:235` `attention_gen_f16_neon` | F16 KV 전용 3-pass 구현 | Step 1, 2 (메인 수정 대상) |
| `engine/src/backend/cpu/common.rs:362` `attention_gen` | F32/Q4_0 공통 fallback | Step 1 (선택) |
| `engine/src/backend/cpu/x86.rs:176` `attention_gen` | AVX2 경로 (참고용) | Step 1 (선택) |
| `engine/src/layers/transformer_layer/forward_gen.rs:455-679` | Decode attention 호출 | 파라미터 변경 시 |
| `engine/src/layers/attention.rs` | Prefill attention (flash 구조 확인 필요) | Step 3 |
| `engine/src/layers/transformer_layer/forward.rs` | Prefill forward | Step 3 |

### 테스트 포인트

1. **정확도 검증** (필수, Step 1/2 공통)
   - 기존 `attention_gen_f16_neon`과 **bit-exact 비교**는 불가 (부동소수점 순서 차이)
   - **허용 오차**: F16 관점 NMSE < 1e-4, 로짓 top-k 일치율 > 99%
   - 테스트 스크립트: `engine/tests/` 아래 신규 테스트 추가 필요
   - 기존 `tests/test_attention*` 또는 `tests/spec/` 참고

2. **성능 측정**
   - Short context (20 tok) 회귀 방지: 22.6 tok/s 유지 또는 개선
   - 4K context 목표: 30+ tok/s (현재 10.6)
   - 2K, 4K, 8K 지점 측정
   - `adb`로 디바이스 벤치마크 필수 (호스트 벤치는 ARM 특성 반영 안 됨)

3. **회귀 테스트**
   - `cargo test -p llm_rs2 -- attention`
   - `cargo test -p llm_rs2 -- kivi` (KIVI는 attention_gen을 호출)
   - `./scripts/deploy-test.sh` (tier 2 디바이스 테스트)

### 주의사항

- **HeadMajor 레이아웃 가정 유지**: KV cache는 `[batch, kv_heads, capacity, head_dim]`, attention kernel은 이 layout에 의존
- **NaN 처리**: 현재 코드의 NaN sanitization (line 337-355)은 필수 — 이전 레이어 오염 방지
- **`.cl` 커널 수정 금지**: CPU 최적화이므로 OpenCL 변경 없음
- **GQA group_size**: `n_rep = num_heads_q / num_heads_kv`, Qwen2.5 1.5B는 6
- **Qwen2.5 1.5B config**: `num_attention_heads=12, num_key_value_heads=2, head_dim=128, num_hidden_layers=28`

---

## 5. 예상 결과 (전체 로드맵)

### Decode TBT (Qwen2.5 1.5B Q4_0, Snapdragon 8 Elite, CPU 6T)

| 단계 | Short ctx | 4K ctx | 4K / llama.cpp |
|------|-----------|--------|----------------|
| **현재** | 22.6 tok/s | 10.6 tok/s | 35% |
| **+ Step 1 (Online softmax)** | ~24 tok/s | ~14 tok/s | 46% |
| **+ Step 2 (Flash Decoding)** | ~25 tok/s | **~25 tok/s** | **82%** |
| **llama.cpp 기준** | 29.98 tok/s | 30.50 tok/s | 100% |

### Prefill (4K)

| 단계 | Prefill |
|------|---------|
| **현재** | 11 tok/s |
| **+ Step 3 (CPU Flash Attn)** | **100+ tok/s** (예상) |
| **llama.cpp** | 51 tok/s (pp512) — long prefill은 측정 필요 |

---

## 6. 다음 세션 시작 시 해야 할 일

### 우선순위 1: Step 1 (Online Softmax) 구현

1. `engine/src/backend/cpu/neon.rs:235 attention_gen_f16_neon` 분석
2. 3-pass → 1-pass 변환
   - 기존: Pass 1 (QK^T → scores) → Pass 2 (softmax in-place) → Pass 3 (weighted V)
   - 신규: for t in 0..n { QK dot, update (m, l, o), V * exp_weight }
3. NaN handling 유지
4. 유닛 테스트: `tests/test_attention_online_softmax.rs` 신규 작성
5. Host 테스트 + 디바이스 배포 + 4K 측정

### 우선순위 2: Step 2 (Flash Decoding) 설계

1. Architect agent 호출: KV split 전략 설계 문서 작성
2. Chunk size 튜닝 실험 (512/1024/2048)
3. Threshold 결정: `if cache_seq_len < THRESHOLD: head-parallel else: kv-split-flash`

### 사전 준비 사항

- 디바이스에 최신 binary 배포 필요할 수 있음 (현재 배포된 것은 이전 Q4 최적화까지만 포함)
  ```bash
  source android.source && cargo build --release --bin generate
  adb push target/aarch64-linux-android/release/generate /data/local/tmp/generate
  ```
- `/tmp/prompt_6k.txt`는 세션 간 persistent하지 않음 — 필요 시 재생성
  ```bash
  cd /tmp && (for i in 1 2 3 4 5 6 7 8; do cat prompt_base.txt; echo; done) > prompt_6k.txt
  # prompt_base.txt는 bench_prompt_xl.txt에서 추출
  adb shell "cat /data/local/tmp/bench_prompt_xl.txt" > /tmp/prompt_base.txt
  ```

---

## 7. 참고 자료

### 논문/구현 참조
- **Flash Attention 1/2** (Tri Dao) — 원조 flash attention
- **Flash Decoding** (Tri Dao 2023) — KV split for decode inference
- **llama.cpp PR #19209**: "ggml-cpu: FA split across kv for faster TG" (2025)
- **llama.cpp ggml_flash_attn_ext**: CPU/GPU 통합 flash attention

### 외부 링크
- https://deepwiki.com/ggml-org/llama.cpp/7.4-flash-attention-and-optimizations
- https://github.com/ggml-org/llama.cpp/pull/19209
- https://crfm.stanford.edu/2023/10/12/flashdecoding.html

### 프로젝트 내 관련 문서
- `arch/tensor_partition.md` — GPU/CPU 병렬 설계 참고 (패턴 유사)
- `docs/14_component_status.md` — 컴포넌트 상태
- 이전 세션 요약: 메모리 `project_session_2026_04_12.md`

---

## 8. 작업 중 발견한 부가 이슈

### KIVI Q4 partial write 버그 (해결됨)
- 커밋 `c9570a6` — `upload_and_dequant_flush_with` 전체 버퍼 덮어쓰기 수정
- 관련 없음 — 참고용

### Long prefill의 O(n²) 문제
- 4472 tokens prefill = 400초 = 0.09 sec/token
- Attention이 O(n)이 아닌 O(n²)로 scale하는 것처럼 보임 (실제로는 prefill의 QK가 O(n²) 행렬곱)
- Step 3에서 해결

### Prefill과 Decode의 독립성
- Step 1, 2는 **decode만** 해결
- Prefill 문제는 Step 3에서 별도 처리
- 실험 목적(decode benchmark)에는 Step 1-2로 충분

---

## 9. 실패 시나리오 대응

### 정확도 저하 발생 시
- Bitwise comparison이 아닌 sensible tolerance (NMSE, top-k match) 사용
- F32 intermediate 유지 여부 확인 (rescale 곱셈이 F16에서 정밀도 문제 가능)

### 성능 개선이 예상보다 작을 경우
- `perf` 또는 `simpleperf`로 profiling 재확인
- L1/L2 miss rate 측정
- DRAM bandwidth saturation 확인

### Chunk 병렬화로 오히려 느려질 경우
- Threshold 상향 조정 (예: 2K 이상에서만 chunk split)
- Rayon overhead 측정 — crossbeam_utils::thread 직접 사용 고려
- SpinPool 활용 검토 (`engine/src/core/thread_pool.rs`)
