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

---

## 10. 후속 작업 (Phase 2) — Long Context 잔여 성능 갭 해소

Step 1/2/3 (Online Softmax / Flash Decoding / Flash Prefill NEON) 완료 후에도 4K 구간에서 llama.cpp 대비 격차가 큼:

| 시나리오 | llm_rs2 | llama.cpp | 비율 |
|---|---|---|---|
| CPU 4K decode | 11.2 tok/s | 30.5 tok/s | 37% |
| CPU 4K prefill | 37.7 tok/s | ~50+ tok/s | 75% |
| GPU 4K decode | 7.4 tok/s | 14.9 tok/s | 50% |
| GPU 4K prefill | 39.2 tok/s | ~760 tok/s | 5.2% (19x gap) |

외부 분석(코드 검증 완료)으로 다음 3개 근본 원인 식별.

### 0. 환경 / 선행 작업 (다른 세션 착수 시 필독)

**선행 커밋 (master 기준)**:
- Step 1 (Online Softmax): `89a7afd`
- Step 2 (Flash Decoding): `70a059f`
- Step 3 (Flash Prefill NEON): `1d2bd2e`

**디바이스**: Galaxy S25 (Snapdragon 8 Elite, Adreno 830, R3CY408S5SB)

**모델 / 프롬프트 (디바이스 경로)**:
- 모델: `/data/local/tmp/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-v2.gguf` (Qwen2.5 1.5B Q4_0)
- 4K 프롬프트: `/data/local/tmp/prompt_6k.txt` (4472 tokens)
- 2K 프롬프트: `/data/local/tmp/prompt_2k.txt` (2054 tokens)
- 바이너리 push 위치: `/data/local/tmp/generate`, `/data/local/tmp/llama-bench` (대조용)

**측정 규칙** (메모리 `feedback_benchmark_thread_count.md` 참조):
- `--threads 6` 단일 (8T 측정 금지)
- 각 조건 1회만 측정 (반복 금지)
- 측정 전 5~10분 쿨다운, `adb shell dumpsys thermalservice | head -20`로 Status 0 확인

**빌드 절차 (Linux 호스트)**:
```bash
source android.source
# 만약 NDK 경로 에러 시 수동 export (이전 세션에서 발견된 함정):
#   export ANDROID_NDK_HOME=/opt/android-ndk
#   export PATH=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH
cargo build --release --bin generate --target aarch64-linux-android
adb push target/aarch64-linux-android/release/generate /data/local/tmp/generate
adb shell chmod +x /data/local/tmp/generate
```

**기준선 측정 명령 (재현 가능)**:
```bash
# Short decode (회귀 방지 측정)
adb shell "cd /data/local/tmp && ./generate \
  -m /data/local/tmp/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-v2.gguf \
  --prompt 'Hello world' \
  -n 128 -b cpu --threads 6 --ignore-eos"

# 4K (주 지표)
adb shell "cd /data/local/tmp && ./generate \
  -m /data/local/tmp/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-v2.gguf \
  --prompt-file /data/local/tmp/prompt_6k.txt \
  -n 128 -b cpu --max-seq-len 6144 --threads 6 --ignore-eos"
```

**현재 베이스라인 (Step 3 = `1d2bd2e` 기준, 6T 1회 cold)**:
- Short decode: 32.6 tok/s
- 2K decode / prefill: 14.2 / 50.9 tok/s
- 4K decode / prefill: 11.2 / 37.7 tok/s
- llama.cpp CPU 6T 4K combined: 30.50 tok/s (대조군)

---

### 10.1 근본 원인 정리

**원인 A — Prefill F16→F32 bulk 변환 (Step 3 1차 PR 미완성 Stage)**
- 위치: `engine/src/layers/transformer_layer/forward.rs:174-229, 694-752`
- 매 레이어마다 F16 KV를 `read_to_f32` + `bulk_f16_to_f32`로 거대한 Vec<f32>에 통째로 복사
- 4K × 28 레이어 → 수십 MB 할당/복사 반복으로 NEON Flash Prefill 이득을 상쇄
- 해결: `arch/cpu_flash_attention_prefill.md` Stage 9 (F16 직접 경로) 구현

**원인 B — CPU Decode Q4_0 horizontal sum 비효율**
- 위치: `engine/src/backend/cpu/neon.rs:2957-3050` `vec_dot_q4_0_q8_0`
- 블록당 `vaddlvq_s16` 4회 호출 — ARM에서 사이클 큰 명령
- llama.cpp 스타일 `vpaddlq_s16 + vpadal_s16` 체인 미사용
- 참조: `docs/31_perf_comparison_llama_cpp.md` "Phase 1: CPU Q4 Dot Product 최적화"

**원인 C — GPU Adreno F16 커널 부재**
- 현재 F16 GEMM은 범용 템플릿 → 로컬 메모리 낭비 + barrier 과다
- llama.cpp는 Adreno subgroup + texture cache 활용
- GPU 4K prefill 격차의 압도적 비중 차지

**추가 이슈 — GPU prefill chunk auto-sizing**
- `--prefill-chunk-size 0` 기본값에서 `CL_INVALID_BUFFER_SIZE` 실패
- 자동 chunk 결정 로직 필요 (소규모 fix)

---

### 10.1.5 GPU Prefill 실제 동작 분석 (CPU Fallback 삼중 병목) — 2026-04-13 추가

> **⚠️ 2026-04-13 17:00 정정 (부분 철회)**: 아래 원인 **D1 (Q4_0 prefill 커널 미지원)** 분석은 실제 벤치마크 설정(**Weights=Q4_0, KV=F16**)과 맞지 않음. `flash_attention_prefill_gpu`의 `DType::Q4_0 ⇒ Ok(false)` 분기는 **KV dtype 기준**이므로 KV=F16 구성에서는 발동하지 않음. Attention은 GPU F16 커널(`flash_attn_f32_f16_q1` 등)로 정상 실행 중. 따라서 **원인 D1/D3 (풀 카피 fallback)은 현재 벤치마크에서는 발생하지 않음**. 원인 **D2 (F16 prefill 커널 head_dim=64 하드코딩)**만 유효하며 Qwen2.5 1.5B (head_dim=128)에서 여전히 CPU fallback 유발. **현재 주요 진단은 하단 [10.1.6](#101-6-gpu-decode-attention의-구조적-결함--kv-split-부재) 으로 이동**했음. P0-4 (Q4_0 KV prefill 커널)는 현재 기본 벤치 구성과 무관하므로 priority 강등 (아래 P0-4 항목 주석 참조).

**핵심 발견** (정정 후): `-b opencl` 지정 시 prefill attention은 GPU F16 커널로 실행되나, **head_dim=128 (Qwen 계열) 조건에서는 여전히 CPU fallback**. 10.1의 원인 A/B/C는 유효하나, 기존 세션에서 해석했던 "GPU = CPU fallback" 프레임은 과도하게 일반화된 것임. 실제로는 head_dim=64 모델에서는 GPU 경로가 작동. 본질적 long-context 성능 갭의 주원인은 10.1.6의 decode attention KV-split 부재.

**원인 D1 — Q4_0 KV cache: Prefill GPU 커널 미존재** ⚠️ **10.1.6 참조로 대체 — 현재 기본 벤치 구성(Weights=Q4_0, KV=F16)과 무관**
- 위치: `engine/src/backend/opencl/mod.rs::flash_attention_prefill_gpu`
- 코드 구조:
  ```rust
  let kernel = match kv_dtype {
      DType::F32 => match &kernels.kernel_flash_attn_f32 { ... }
      DType::F16 => match &kernels.kernel_flash_attn_f32_f16 { ... }
      _ => return Ok(false), // Q4_0 not supported
  };
  ```
- **정정 (2026-04-13 17:00)**: 이 분기는 **KV dtype 기준**이며, 현재 기본 벤치 설정 (`--kv-type f16`)에서는 발동하지 않음. Weights가 Q4_0이더라도 KV가 F16이면 F16 커널 경로로 진입. 이 원인은 `--kv-type q4_0` 등 KV=Q4_0 구성을 쓰는 다른 사용 사례에만 해당.
- ~~영향: Q4_0 KV (현재 기본 실전 구성)일 때 GPU backend 요청이 무조건 `Ok(false)` → 상위 레이어 CPU fallback.~~ (잘못된 전제 — 기본 구성은 KV=F16)
- 근본 해법 (KV=Q4_0 구성 한정): flash_attn prefill 커널의 Q4_0 지원 (on-the-fly dequant 전략) — **P0-4 참조, 우선순위 강등됨**.

**원인 D2 — F16 Prefill 커널 head_dim=64 하드코딩 (Qwen 계열 차단)**
- 위치: `engine/src/backend/opencl/mod.rs::flash_attention_prefill_gpu` (F16 분기)
- 빌드 시 `DK=DV=64` 매크로로 컴파일된 `flash_attn_f32_f16` 커널만 존재 → `head_dim != 64`인 경우 fallback:
  ```rust
  if head_dim != 64 {
      return Ok(false);
  }
  ```
- Step 2 (최근 커밋)에서 **decode** 경로에는 `head_dim=128` 커널을 추가했으나, **prefill** 경로는 미적용.
- 영향: Qwen2.5 1.5B (head_dim=128)는 F16 KV cache 구성에서도 prefill GPU 경로 불가 → CPU fallback.

**원인 D3 — Fallback 경로의 삼중 병목** ⚠️ **현재 벤치 구성(KV=F16, Qwen head_dim=128)에서는 원인 D2로 인한 fallback 시에만 발생 — 10.1.6 참조**
- 위치: `engine/src/layers/transformer_layer/forward.rs` `!gpu_dispatched` 분기 (prefill path around L174-229, variant L694-752)
- 단계별 비용:
  1. **VRAM → RAM 전량 카피**: K/V 텐서가 device-only일 때 `is_device_only` 분기로 수십~수백 MB를 CPU 메모리로 다운로드
  2. **F32 강제 변환**: Q4_0 → `read_to_f32` 또는 F16 → `bulk_f16_to_f32`로 전부 F32 확장 (O(N))
  3. **스칼라 CPU 연산**: 최적화되지 않은 CPU 경로로 실제 attention 계산 수행
- **정정**: 이 경로는 원인 D2 (head_dim=128) 때문에 Qwen prefill에서만 발동. Llama 3.2 1B (head_dim=64) 기본 벤치에서는 GPU F16 커널로 dispatch 성공 → D3 발생 안 함.
- 이것이 10.1 원인 A("F16→F32 bulk 변환")를 더 심각하게 만드는 실제 호출 컨텍스트 (D2 fallback 시).

**영향 평가**
- 문서/측정 상 "GPU prefill"로 기록된 수치 일부는 실제 CPU 실행 결과 → **재측정 필요**.
- P2-1 (Adreno F16 matmul 커널)만으로는 해결 불가 — prefill flash attn 커널 자체가 dispatch 되지 않는 구간이 존재.
- 본질적 해결은 **GPU 커널 지원 확장 (P0-3, P0-4)** + **fallback 감지 가시성 (P1-2 강화)**.

**코드 경로 요약 (재진입 탐지용)**
| 파일 | 함수 | 역할 |
|---|---|---|
| `engine/src/backend/opencl/mod.rs` | `flash_attention_prefill_gpu` | dtype/head_dim 분기 → `Ok(false)`면 fallback 신호 |
| `engine/src/layers/transformer_layer/forward.rs` | prefill dispatch (L174-229, L694-752) | `!gpu_dispatched` 시 CPU 경로 진입 (VRAM→RAM 카피, F32 변환) |
| `engine/src/backend/cpu/neon.rs` | `flash_prefill_forward_*` | 실제 CPU 계산 (F32 가정) |

---

### 10.1.6 GPU Decode Attention의 구조적 결함: KV-Split 부재 — 2026-04-13 추가 (현재 최우선 진단)

> **2026-04-13 추가 정정** (P0-5 실측 후): KV-split (Flash Decoding) 단독으로는 GQA redundant VRAM read 문제를 해결하지 못함.
> KV-split은 시퀀스 축 병렬화 (GPU 점유율 향상)에 효과적이나, 같은 KV-head를 공유하는 Q-head들이 여전히
> 별도 WG로 dispatch되므로 local memory 공유 불가, VRAM 중복 로드는 절반(6배 → 3배 수준)만 줄어듦.
> 본질적 해결은 P0-5c (GQA-aware KV 공유 커널) 참조.
>
> **2026-04-13 실측 검증 (revert 완료, 커밋 d0d4978)**: P0-5/P0-5b 디바이스 실측 결과, KV-split는 4K Adreno 830 구성에서 net regression (7.3 → 6.5 tok/s, -11%).
> 원인: (1) 추가 kernel launch 오버헤드 (12 Q-head × 4 splits = 48 WG), (2) partials/meta buffer 추가 VRAM 트래픽, (3) 12 Q-head만으로 점유율 충분해 병렬성 이득 미미.
> 결론: KV-split 단독으로는 4K decode 병목 해결 불가. 본질적 해결은 P0-5c (GQA-aware KV 공유 커널).
>
> **2026-04-13 추가 실측 검증 (P0-5c revert, 커밋 84c82d2)**: GQA-aware KV 공유 커널을 구현해 같은 KV-head를 공유하는 Q-head들을 단일 WG로 묶었으나, Qwen 2.5 1.5B (n_heads_kv=2) 에서 **WG=2 under-utilization**으로 오히려 net regression (7.3 → 2.6 tok/s, -65%).
> 6× VRAM KV read 감소 효과 < Adreno 830의 다수 SP가 놀게 되는 idle 손실.
> 본질적 해결은 **GQA + KV-split 결합** (`gws=[WG_SIZE, n_heads_kv, n_kv_splits]` 으로 Qwen 2×8=16, Llama 8×8=64 WG 확보). P0-5 (KV-split) + P0-5c (GQA reuse) 둘 다 부활 + 결합하는 별도 작업 필요 → **신규 P0-5d 참조**.
> llama.cpp는 모든 백엔드에서 decode GQA reuse 미구현 — 우리도 동일 stance 유지.

#### 핵심 발견
- **짧은 컨텍스트(128 토큰)**: llm_rs GPU가 llama.cpp 대비 **빠름** — Adreno-특화 Q4_0 weight matmul 커널(`mul_mv_q4_0_f32.cl`)의 서브그룹/텍스처 최적화 덕분. 이때 시간 99%는 weight matmul, attention 비중 무시 가능.
- **긴 컨텍스트(4K+)**: llm_rs GPU가 llama.cpp 대비 **반토막** (Qwen2.5 1.5B 기준 7.4 vs 14.9 tok/s) — weight matmul 시간은 고정이지만 attention 시간이 폭증.
- Attention 시간 폭증의 원인은 **VRAM 대역폭 포화**이며, 그 원인은 **decode attention 커널의 head-parallel only 디스패치** (KV-split 부재).

#### 코드 증거
`engine/src/backend/opencl/mod.rs` 약 2038 줄 부근 `flash_attention_decode_gpu` 디스패치:
```rust
// Q1_WG_SIZE = 64 (compile-time constant in the kernel)
const Q1_WG_SIZE: usize = 64;

// Global = [Q1_WG_SIZE, n_heads_q * batch (= n_heads_q for decode batch=1), 1]
// Each WG handles one (batch, head) pair.
let global_work_size: [usize; 3] = [Q1_WG_SIZE, n_heads_q, 1];
let local_work_size: [usize; 3] = [Q1_WG_SIZE, 1, 1];
```

- 병렬화 축이 **Q-head 단일 축**이며 **KV 시퀀스 축 분할 없음** (Flash Decoding / KV-split 미구현).
- 워크그룹 간에는 local memory 공유 불가 → GQA 공유가 "논리적"으로만 존재하고 VRAM 레벨에서는 보장되지 않음.

#### 구조적 결함 — GQA 중복 VRAM 읽기
- Qwen2.5 1.5B: n_heads_q=12, n_heads_kv=2 → **6개 Q-head가 1개 KV-head 공유**.
- Llama 3.2 1B: n_heads_q=32, n_heads_kv=8 → **4개 Q-head가 1개 KV-head 공유**.
- 현재 구조에서 각 Q-head가 별도 워크그룹 → **같은 KV를 각자 따로 VRAM에서 로드**.

#### 트래픽 계산 (Qwen 4K 예시)
- KV-head 1개 (F16): 4096 tok × 128 dim × 2B = **1MB** (K만. V까지 2MB)
- Q-head 12개가 각자 로드: 12 × 2MB = **24MB/layer/token**
- 28 레이어: **672MB/token**  ↔ 이상적 공유 시: 2 KV-head × 2MB × 28 = **112MB/token** (6배 감소)

**단계별 VRAM 트래픽 비교 (per layer / token)** — 2026-04-13 정정:

| 단계 | VRAM 트래픽 / layer / token | 중복 배율 |
|---|---|---|
| 이상 (완전 공유) | ~4 MB | 1× |
| Legacy 단일 커널 | 24 MB | 6× |
| KV-split (P0-5) **(측정상 net negative, REVERTED)** | ~12 MB (이론치) | 3× (시퀀스 축 분할 + 페치 재사용 일부) |
| GQA 공유 (P0-5c) **(측정상 net negative on Qwen, REVERTED — WG=2 under-utilization)** | ~4 MB (이상치 근접) | 1× |

→ P0-5는 절반 수준의 개선에 불과. 본질 해결은 P0-5c 의 WG 그룹핑 + local memory KV 공유.
- LPDDR5X ~77GB/s 환경에서 이는 대역폭 포화의 직접 원인

#### 짧은 문맥에선 티 안 나는 이유
- 128 토큰 구성: KV-head 1개 당 K+V ≈ 32KB (F16). 4~6배 중복 읽기해도 ~192KB 수준 → GPU L1/L2 캐시 hit → 문제 없음.
- 4K 문맥: KV-head 당 2MB → L2 초과 → 매번 VRAM 페치 → 중복 요청이 실제 대역폭 비용으로 전환.

#### 결론
- 4K decode 성능 갭(**llm_rs GPU 7.4 tok/s vs llama.cpp 14.9 tok/s**)의 **주원인은 GPU attention 커널의 KV-split 부재**.
- 이는 하드웨어 한계가 아닌 **커널 디스패치 설계 결함**. llama.cpp는 Flash Decoding (KV-split) 으로 해결.
- Prefill 경로에서 발견된 P0-3 (head_dim=128) 문제와는 **다른 이슈**이며 병렬로 존재. 해법은 신규 **P0-5** (아래 10.2 참조).

---

### 10.2 TODO 항목

#### [P0-1] Q4_0 vec_dot horizontal sum 최적화 (vpaddlq + vpadal 체인) + i8mm hot path
- **Status**: PARTIAL DONE (2026-04-13, 커밋 c794292 + 9abd8e0; decode 11.2 → 12.0 tok/s, +7.1%; 목표 14+ tok/s 미달; 잔여 갭은 **P0-5 (GPU KV-split 부재)** 기인 가능성 높음 — CPU-only 구성이 아닌 하이브리드 측정 시 재검증 필요)
- **Sprint**: current
- **담당**: senior-implementer
- **Dependencies**: 없음 (독립 작업)
- **예상 작업량**: 0.5 ~ 1일 (초기) + i8mm 경로 확장 0.5 ~ 1일
- **코드 위치**: `engine/src/backend/cpu/neon.rs:2957-3050` `vec_dot_q4_0_q8_0`, 그리고 i8mm 핫 경로 `vec_dot_q4_0_q8_0_i8mm_4rows_il`, `vec_dot_q4_0_q8_0_i8mm`
- **참조 문서**: `docs/31_perf_comparison_llama_cpp.md` (Phase 1 섹션), llama.cpp `ggml-quants.c` `ggml_vec_dot_q4_0_q8_0`
- **Description**: 블록당 4회 `vaddlvq_s16`를 `vpaddlq_s16 + vpadal_s16` 체인으로 교체하여 horizontal reduction 사이클 단축. 가능하면 i8mm (`vusdotq_s32`) 경로와 함께 dotprod 경로도 같은 기법으로 통일. **확장**: 초기 수정은 plain NEON 함수에만 적용했으나, Snapdragon 8 Elite에서는 i8mm capable이라 `vec_dot_q4_0_q8_0_i8mm*` 경로가 dispatch되고 plain NEON 경로가 사용되지 않음 — 실측에서 성능 변동이 관측되지 않음. 따라서 **i8mm 핫 경로 (`_i8mm_4rows_il`, `_i8mm`)에도 동일 최적화 확대 적용**이 필요.
- **Acceptance Criteria**:
  - [ ] **`vec_dot_q4_0_q8_0_i8mm_4rows_il` 및 `vec_dot_q4_0_q8_0_i8mm`가 dispatch되는 실측 환경 (Snapdragon 8 Elite, Galaxy S25)에서 4K decode 11.4 → 14+ tok/s**
  - [ ] 4K decode (Qwen2.5 1.5B Q4_0, Snapdragon 8 Elite, CPU 6T): 11.2 → 14+ tok/s (+28% 이상)
  - [ ] Unit test `tests/neon_q4_dot.rs` 동일 결과 유지 (abs diff ≤ 1e-5)
  - [ ] `cargo bench` (있다면) 또는 `generate --profile` 기준 `matmul` op latency 감소 확인
  - [ ] x86 AVX2 경로 미영향 (회귀 테스트)
- **구현 패턴** (llama.cpp `ggml_vec_dot_q4_0_q8_0` 참조):

  ```c
  // llama.cpp 패턴 (ggml-cpu/quants.c 또는 ggml-cpu/arch/arm/quants.c)
  // 핵심: vpaddlq_s16 → 짝수 인접 합산 → vpadal_s16으로 i32 누적기에 즉시 페어 더함
  int32x4_t sumv0 = vdupq_n_s32(0);
  int16x8_t p0 = vmull_s8(vget_low_s8(x0), vget_low_s8(y0));
  p0 = vmlal_s8(p0, vget_high_s8(x0), vget_high_s8(y0));
  sumv0 = vpadalq_s16(sumv0, p0);  // <-- 핵심: pairwise add long + accumulate
  // ... 블록 끝에서 한 번만 vaddvq_s32(sumv0)로 horizontal sum
  ```

  **현재 (neon.rs:2957~)**:
  - 블록당 4회 `vaddlvq_s16(...)` 호출 → 각 호출이 별도 horizontal reduction
  - i16 8-element를 i64 하나로 줄이는 비싼 명령

  **변경 후 목표**:
  - 블록 내부에서 `vpaddlq_s16` (i16x8 → i32x4 pairwise) + `vpadal_s16` (in-place i32x4 누적)
  - 마지막 블록 종료 후 단 1회만 `vaddvq_s32` (i32x4 → i32 horizontal sum)
  - 결과: 사이클 절감 + 명령 수 감소

  i8mm 경로(`vusdotq_s32` 사용 영역)도 동일 원리 적용 가능 — `vusdotq` 누적기는 이미 i32x4이므로 horizontal은 마지막 1회만.
- **Notes**: ROI 최고 (작은 작업, decode 28% 개선 추정). Phase 2의 first delivery로 권장. **2026-04-13 재작업 메모**: plain NEON만 수정하면 Snapdragon 8 Elite에서 측정 불변 (i8mm이 우선 dispatch되기 때문). 반드시 dispatch되는 경로에서 실측 검증해야 함.

---

#### [P0-2] Prefill F16 직접 경로 (Step 3 Stage 9 완성)
- **Status**: TODO
- **Sprint**: current
- **담당**: senior-implementer
- **Dependencies**: Step 3 (Flash Prefill NEON) 완료됨 — 선행 조건 충족
- **예상 작업량**: 1 ~ 2일
- **코드 위치**:
  - `engine/src/layers/transformer_layer/forward.rs:174-229` (prefill main path)
  - `engine/src/layers/transformer_layer/forward.rs:694-752` (variant path)
  - `engine/src/backend/cpu/neon.rs` (Flash Prefill 커널 — F16 변종 추가)
- **참조 문서**: `arch/cpu_flash_attention_prefill.md` Stage 9 (F16 직접 경로)
- **Description**: `read_to_f32` + `bulk_f16_to_f32`로 전체 K/V를 F32로 변환하는 대신, Flash Prefill 커널 내부에서 F16 block을 직접 read → decode → softmax tile에 사용. 타일 단위 deq (on-the-fly)로 수십 MB Vec<f32> allocation/copy 제거.
- **Acceptance Criteria**:
  - [ ] Prefill 경로에서 `bulk_f16_to_f32` 호출 제거 (F16 KV 케이스)
  - [ ] 4K prefill (CPU): 37.7 → 60+ tok/s (+60% 이상)
  - [ ] `prompt_alloc` / `memcpy` 관련 heap 프로파일 대폭 감소 (측정 필요)
  - [ ] 수치 정확도: `test_backend` CPU vs baseline 비교, logit cos-sim ≥ 0.999
  - [ ] 기존 F32 경로 회귀 없음
- **F16 직접 경로 신규 함수 시그니처 후보** (Architect와 협의):

  ```rust
  // engine/src/backend/cpu/neon.rs (신규)
  #[cfg(target_arch = "aarch64")]
  pub fn flash_prefill_forward_f16_neon(
      q: &[f32],          // [seq_len, n_heads_q, head_dim] (Q는 F32 유지 — RoPE 출력)
      k_f16: &[u16],      // [kv_heads, capacity, head_dim] F16 raw bits
      v_f16: &[u16],      // 동일 layout
      // ... 기존 F32 시그니처와 동일한 메타파라미터 ...
  ) -> Result<()>;
  ```

  - F16 K/V는 타일 진입 시 NEON `vld1q_u16` + `vcvt_f32_f16`으로 on-the-fly 변환
  - Step 1/2의 `flash_apply_token` F16 변환 패턴 재사용

- **forward.rs 분기 변경 위치**:
  - L174-229: prefill main path — `read_to_f32` 호출 전에 `if F16 KV: flash_prefill_forward_f16_neon(...)` 분기 추가
  - L694-752: variant path — 동일 패턴
  - 기존 F32 경로는 fallback으로 유지 (F32 KV 케이스용)
- **Notes**: forward.rs (dispatch layer)와 neon.rs (kernel layer) 양쪽 수정이 필요. Architect 검토 권장 — 특히 F16 K/V `Tensor` 접근자 trait 안정성. **Scope 재확인 (2026-04-13)**: P0-2는 CPU fallback 경로 자체를 빠르게 만드는 완화책 — GPU backend를 명시적으로 지정한 시나리오의 본질적 해결은 아님. GPU path dispatch 자체는 **P0-3/P0-4**에서 다룸. P0-3/P0-4 완료 후에도 `-b cpu` 실행과 head_dim 미지원 fallback을 위해 P0-2는 여전히 필요.

---

#### [P0-3] OpenCL flash_attn prefill 커널 head_dim=128 지원 (Qwen 계열)
- **Status**: **DONE (기능) / 성능 회귀 (2026-04-13, 커밋 7f41d2f + 37aaab2)** — Qwen prefill GPU dispatch 활성화 확인, 단 prefill 속도 31 tok/s < CPU fallback 43 tok/s (-28%). 남은 격차는 Adreno 특화 matmul 부재 → **P2-1 영역**으로 이관.
- **Sprint**: done (perf 잔여는 P2-1)
- **담당**: senior-implementer
- **Dependencies**: 없음 (Step 2 decode head_dim=128 패턴 참조)
- **예상 작업량**: 1 ~ 2일 (실제 소요 동일)
- **코드 위치**:
  - `engine/src/backend/opencl/mod.rs::flash_attention_prefill_gpu` (F16 분기 — 현재 `if head_dim != 64 { return Ok(false); }`)
  - `engine/kernels/flash_attn_f32_f16*.cl` (prefill 커널; `DK=DV=64` 빌드 매크로)
  - 커널 빌드 및 캐시 로딩부 (`kernel_flash_attn_f32_f16_dk128` 신규 추가 필요)
- **참조**: 최근 커밋에서 decode에 적용된 head_dim=128 커널 (`attention_gen_f16_neon` 계열 Step 2/3 prefill/decode 패턴), llama.cpp Adreno flash attention 커널
- **Description**: Prefill용 F16 flash attention OpenCL 커널을 `DK=DV=128`로 별도 빌드하고, `head_dim == 128` 분기에서 dispatch. Step 2가 decode 경로에 적용한 head_dim 별 커널 컴파일 + 선택 로직을 prefill에 확장. Qwen2.5 1.5B (head_dim=128)가 F16 KV 구성에서 GPU prefill을 사용할 수 있도록 함.
- **Acceptance Criteria**:
  - [x] `flash_attention_prefill_gpu`가 `head_dim == 128, DType::F16` 조합에서 `Ok(true)`로 dispatch됨 (fallback 경고 로그 없음) ✅
  - [ ] 4K prefill (Qwen2.5 1.5B, F16 KV, GPU): 현재 CPU fallback 수치 대비 명확한 개선 — ❌ **미달** (CPU 43 → GPU 31, -28%)
  - [x] head_dim=64 (Llama 3.2 계열) 경로 회귀 없음 ✅
  - [x] `test_backend` 정확도: CPU vs GPU cos-sim ≥ 0.999 (head_dim=128) ✅
  - [x] 커널 컴파일 시간 증가는 허용 범위 (프로그램 캐시로 완화 확인) ✅
- **Notes**:
  - Step 2 decode 패턴 재사용 가능 — 구조적으로 낮은 리스크. P0-4 대비 **선행 권장** (커널 템플릿 매크로화 경험 확보).
  - **2026-04-13 실측**: Qwen prefill GPU 활성화 확인, `[Prefill] flash_attn dispatch: head_dim=128, DType=F16, BLOCK_M=32` 로그 발화, CPU fallback 로그(`[GPU-fallback]`) 제거. 단 prefill 속도 31 tok/s < CPU fallback 43 tok/s.
  - **BLOCK_M=64 vs 32 모두 30.8 ~ 31.0 tok/s** 로 register spill 원인 아님 확정 (튜닝 무효).
  - 남은 격차 (GPU 31 vs llama.cpp 760, **24×**) 는 Adreno subgroup + texture cache 최적화 필요 → **P2-1 영역**으로 이관.
  - Decode 경로 영향 없음 (7.4 → 7.5 tok/s).

---

#### [P0-5] GPU flash_attention_decode KV-split (Flash Decoding) 구현 — ⚠️ **REVERTED** (2026-04-13)
- **Status**: **REVERTED (2026-04-13, 커밋 d0d4978)** — 4K Adreno 830 실측에서 net regression (-11%) 확정, 3개 커밋(5f3dd1f/9a5a75e/f14ddb3) 통합 revert
- **실측 결과 (2026-04-13, Adreno 830, Qwen2.5 1.5B Q4_0, F16 KV, 4K, 6T)**:
  - 커널 컴파일 성공: `split_dk64=Y split_dk128=Y reduce_dk64=Y reduce_dk128=Y`
  - KV-split 발동 확인: `[FlashDecode] KV-split active: kv_splits=4 (n_kv=4473, head_dim=128, n_heads_q=12)`
  - baseline (plan path, legacy single kernel): 7.3 tok/s
  - `--no-gpu-plan` legacy 추정: 6.4 tok/s
  - `--no-gpu-plan` + KV-split active: **6.6 tok/s** (+3% vs legacy, **-10% vs plan baseline**)
  - 결론: 알고리즘은 정상 동작. Plan path가 legacy 단일 커널을 pre-bind 하므로 production 경로에서 KV-split이 활성화되지 않음. → **P0-5b 필수**
  - GQA 본질 해결은 별도 작업 (**P0-5c**)
- **Sprint**: current
- **담당**: researcher 선조사 (짧음, 1일) → senior-implementer 구현 (2~4일)
- **Dependencies**: 없음 (P0-3/P0-4와 독립, GPU kernel 계열 중 **가장 ROI 높음**)
- **예상 작업량**: 3~5일
- **코드 위치**:
  - `engine/src/backend/opencl/mod.rs::flash_attention_decode_gpu` 디스패치 (약 2038줄 부근 `global_work_size` 정의)
  - `engine/kernels/flash_attn_f32_f16_q1*.cl` (head-parallel 전용, KV-split 브랜치 추가 또는 별도 커널)
  - (선택) llama.cpp Adreno용 flash decoding 커널 참조
- **Description**:
  - Global work size 를 `[Q1_WG_SIZE, n_heads_q, kv_splits]` 로 확장.
  - 각 (head, kv_split) 워크그룹이 자기 chunk의 partial `(m_i, l_i, o_i)` 계산.
  - 2nd pass (또는 동일 커널 2단계)에서 global max/log-sum-exp 로 merge.
  - KV-split 수는 컨텍스트 길이에 따라 동적 결정 (예: `ceil(cache_seq_len / 1024)` 또는 GPU CU 수 기반). Short ctx에서는 `kv_splits=1` 로 회귀 방지.
  - GQA-aware: 같은 KV-head 를 공유하는 Q-head 는 같은 KV-split 스케줄 내에서 local memory 로 공유 (가능한 경우).
- **Acceptance Criteria**:
  - [ ] Short context (128 tok) decode 회귀 없음 (`kv_splits=1` fallback 경로)
  - [ ] 4K decode (Qwen2.5 1.5B, Adreno 830): 7.4 → **12+ tok/s** (llama.cpp 14.9의 80% 이상)
  - [ ] Head-parallel vs KV-split 자동 전환 임계값 튜닝 결과 TODO 노트에 기록
  - [ ] 정확도: logits cos-sim ≥ 0.999 vs 기존 커널
  - [ ] `.cl` 커널 수정 허용 (`feedback_cl_modification.md` 근거)
- **Notes**: 이번 세션에서 발견된 **최우선 GPU 성능 이슈**. 10.1.6 의 VRAM thrashing 분석이 직접 근거. Flash Decoding(Tri Dao 2023) 패턴. `.cl` 커널 수정 허용. P0-1 rework 잔여 갭(decode 12.0 → 목표 14+ tok/s)도 이 항목 완료로 회복 가능성 있음 (CPU 구성에서도 GPU KV-split이 weight matmul 대역폭 경쟁을 줄임). researcher 선조사: llama.cpp Adreno flash decoding 커널 구조 + Tri Dao 2023 Flash Decoding 알고리즘 검토.
- **2026-04-13 후속 Notes**: Plan path가 legacy 단일 커널을 pre-bind 하므로 KV-split이 production 경로에서 활성화되지 않음. **P0-5b 필수**. 본질적 GQA 해결은 **P0-5c** (Q-head 그룹 WG 묶기 + local memory KV 공유). 본 항목 완료 전까지 P0-5는 측정상 효과 없음 (확인됨).
- **2026-04-13 REVERT 결정 (커밋 d0d4978)**: 실측 결과 plan path / non-plan 양쪽 모두 net regression (-10~11%):
  - baseline (legacy single kernel, plan ON) 7.4 → 7.3 tok/s
  - `--no-gpu-plan` + KV-split (P0-5): 6.6 tok/s (-10%)
  - plan + KV-split (P0-5b 통합 후): **6.5 tok/s (-11%)**
  - PLAN_DEBUG trace로 모든 16 레이어에서 `flash adaptive: split+reduce (kv_splits=4)` 발동 확인 — 알고리즘/plan 통합은 정상, 그러나 4K 구간에서 구조적으로 비효율.
  - 원인: (1) 12 Q-head × 4 splits = 48 WG launch overhead, (2) partials/meta buffer (~24KB × 16 layer) 추가 VRAM trip, (3) 12 Q-head만으로 Adreno 830 점유율 충분 → 병렬성 이득 미미, (4) KV-split은 시퀀스 병렬화이며 10.1.6의 GQA redundant read 와 직교하는 문제.
  - llama.cpp OpenCL port도 Flash Decoding 미구현 — 동일 stance로 revert.
  - 코드는 5f3dd1f / 9a5a75e / f14ddb3 에 보존. **재개 조건**: 8K+ 컨텍스트 표준 벤치 도입, 또는 더 작은 GPU 타겟 추가 (점유율이 낮아 KV-split 이득이 커질 수 있는 구성).
  - 본질적 해결은 **P0-5c** (GQA-aware KV 공유 커널) — KV-split과 독립적으로 구현 가능.

---

#### [P0-5b] Plan-path KV-split 통합 — ⚠️ **REVERTED** (2026-04-13)
- **Status**: **REVERTED (2026-04-13, 커밋 d0d4978)** — Plan 통합은 정상 동작 확인 (PLAN_DEBUG trace), 단 P0-5 자체가 4K 구간에서 비효율이라 함께 revert
- **Sprint**: current
- **담당**: senior-implementer
- **Dependencies**: P0-5 (커널 + dispatch 코드) — 충족
- **예상 작업량**: 1~2일
- **코드 위치**:
  - `engine/src/backend/opencl/plan.rs::build_flash_attention_step` — legacy 단일 커널 pre-bind 지점
  - `engine/src/backend/opencl/mod.rs::flash_decode_kv_splits` — heuristic 재사용
  - `engine/src/backend/opencl/mod.rs::flash_attention_decode_split_gpu` — split 경로 호출 헬퍼
- **Description**:
  - `plan.rs`가 build 단계에서 `kv_splits`를 결정하고, `> 1`이면 split + reduce 두 커널을 step으로 등록.
  - `DynamicArg::CacheSeqLen` 변경에 따라 `kv_splits` 동적 재계산 가능하도록 구성.
  - `kv_splits == 1`이면 기존 단일 커널 step 유지 (short context 회귀 방지).
  - Partials / meta workspace 는 plan workspace에 사전 등록.
- **Acceptance Criteria**:
  - [ ] Plan path 활성 상태에서 4K decode 시 `[FlashDecode] KV-split active` 로그 발생 확인
  - [ ] 4K decode (Qwen 2.5 1.5B, Adreno 830, plan ON): baseline 7.3 tok/s 대비 회귀 없음, 가급적 +5% 이상
  - [ ] Short ctx (128 tok) 회귀 없음 (`kv_splits=1` legacy step 유지 확인)
  - [ ] `flash_attn_decode_split` 테스트 전체 통과
  - [ ] Plan invalidation 시 (cache_seq_len 점프) `kv_splits` 재계산 검증
- **Notes**: P0-5의 production 경로 활성화. 본 항목 완료 전까지 P0-5는 측정상 효과 없음 (확인됨). P0-5b 완료 후 잔여 GQA redundant read 문제는 P0-5c로 이관.
- **2026-04-13 REVERT 결정 (커밋 d0d4978)**: Plan 통합은 완벽히 동작함을 확인 (PLAN_DEBUG trace로 16 layer 모두 `split+reduce (kv_splits=4)` 발동). 측정치 plan+KV-split 6.5 tok/s (baseline 7.3 대비 -11%). P0-5 자체가 4K 구간에서 net negative라 함께 revert. 재개 시 P0-5 resurrect와 묶어서 진행 (8K+ 컨텍스트 또는 더 작은 GPU 타겟 도입 시).

---

#### [P0-5c] GQA-aware KV 공유 커널 (WG 그룹핑 + local memory 공유) — ⚠️ **REVERTED** (2026-04-13)
- **Status**: **REVERTED (2026-04-13, 커밋 84c82d2)** — Qwen 4K 디바이스 실측에서 net regression (-65%) 확정. PLAN_DEBUG로 GQA kernel 정상 활성화 확인했으나, n_heads_kv=2 → WG=2 under-utilization 으로 6× VRAM 절감 < Adreno 830 SP idle 손실. 코드 보존 (커밋 5f59455). 결합 작업은 신규 P0-5d 로 이관.
- **Sprint**: current
- **담당**: researcher 선조사 (1일) → senior-implementer 구현 (3~5일)
- **Dependencies**: 없음 (P0-5/P0-5b revert로 의존성 해소, 단독 직진)
- **예상 작업량**: researcher 1일 + senior-implementer 3~5일
- **코드 위치**:
  - `engine/kernels/flash_attn_f32_f16.cl` — 신규 GQA-grouped 변종 또는 기존 q1_split 일반화
  - `engine/src/backend/opencl/mod.rs` — dispatch 로직 확장
- **Description**:
  - 같은 KV-head를 공유하는 Q-head 그룹 (`gqa_ratio = n_heads_q / n_heads_kv`)을 **단일 WG**로 묶어 dispatch.
  - WG 내에서 KV를 local memory 또는 register tile에 **한 번** 로드, 그룹 내 모든 Q-head가 reuse.
  - 결과는 head 단위로 출력 (reduce 단계와 호환).
  - **P0-5/P0-5b revert 후 단독 진행**: P0-5 측정으로 KV-split은 GQA redundant VRAM read 를 해결하지 못함이 입증됨 (10.1.6 "2026-04-13 실측 검증" 참조). GQA 공유 커널이 유일한 본질 해결책.
  - **KV-split과 독립적으로 구현** — 복잡도 절감, P0-5/P0-5b 재활성화 불필요.
  - (옵션) KV-split 과 **직교** 가능: 향후 8K+ 시나리오에서 `(kv_split, kv_head_group)` 두 축 dispatch 도 가능. 현재는 `kv_splits=1` 고정 + GQA 공유만 구현.
    - 예 (향후 확장): `global = [WG_SIZE, n_heads_kv * gqa_ratio_blocks, kv_splits]`
- **참조**:
  - llama.cpp `ggml-cuda/fattn-vec-f16.cu`, `fattn-vec-f32.cu` — `parallel_blocks` + `ncols` 그룹핑
  - llama.cpp `ggml-metal.metal::kernel_flash_attn_ext_vec` — `Q_PER_THREAD` 구조
  - 우리 파일: `engine/kernels/flash_attn_f32_f16.cl` — `head_kv_idx = head_idx / gqa_ratio` 부근
- **Acceptance Criteria**:
  - [ ] **4K decode (Qwen 2.5 1.5B, Adreno 830, plan ON)**: baseline 7.4 → **12+ tok/s** (llama.cpp 14.9의 80%+)
  - [ ] **VRAM 트래픽 이론치**: 24 MB/layer/token → **~4 MB/layer/token** (6배 감소, GQA 이상치 근접)
  - [ ] Short ctx 회귀 없음
  - [ ] 정확도: logits cos-sim ≥ 0.999 vs 기존 커널
  - [ ] (선택) VRAM 트래픽 실측 (perfetto/snapdragon profiler 가능 시): 24MB → 4~6MB / layer / token 확인
- **Notes**: P0-5/P0-5b 완료 후 잔여 4K GPU 격차 해소의 **본질 작업**. GQA local memory 공유로 VRAM 대역폭 6배 절감 기대. llama.cpp 의 `Q_PER_THREAD` 매크로 패턴이 가장 직접적 참조. 이 항목이 10.1.6 "GQA 중복 VRAM 읽기" 본질 결함의 최종 해법.
- **2026-04-13 REVERT 결정 (커밋 84c82d2)**:
  - **실측 (Adreno 830, Qwen 2.5 1.5B Q4_0, F16 KV, 4K n_kv=4472, plan ON, cooldown 240s, AP 31.4°C, Status 0)**:
    - baseline (legacy q1, plan ON): 7.4 → 7.3 tok/s 재현
    - **GQA kernel (P0-5c): 2.6 tok/s (-65%)**
  - **PLAN_DEBUG 검증**: `[Plan] L0..L27 GQA flash attention dispatch (attn_seq_len=4473, gws=[64, 2, 1], lws=Some([64, 1, 1]))` — Plan path에서 StandardFlashGqa variant 정상 활성화. 단 `gws=[64, 2, 1]` = **WG 2개만** dispatch (Qwen n_heads_kv=2).
  - **원인**:
    1. Qwen n_heads_kv=2 → WG=2 → Adreno 830 (다수 SP) 심각한 under-utilization (Researcher 조사 §G.4 사전 경고 일치)
    2. 6× VRAM KV read 절감 효과 < GPU SP idle 손실
    3. `o_acc[8][32]` private register 압력 (spill 가능성) 추가 악화
    4. llama.cpp도 decode GQA reuse 미구현 — 동일 stance 유지
  - **Llama 3.2 1B (n_heads_kv=8, WG=8) 미측정**: 상대적으로 유리할 가능성, 그러나 현 시점 주력 벤치 모델이 Qwen이라 검증 불가.
  - **코드 보존**: 커밋 5f59455 (P0-5c GQA 커널 + plan 통합 + 테스트 + research doc 일체).
  - **재개 조건**:
    - Qwen 전용 **GQA + KV-split 결합 커널** 도입 시 (`gws=[WG_SIZE, n_heads_kv, n_kv_splits]` 으로 16+ WG 확보) → **신규 P0-5d 참조**
    - 또는 Llama 3.2 (n_heads_kv=8) 주력 벤치 도입 후 재측정 (WG=8로 실용성 있을 가능성)

---

#### [P0-5d] GQA + KV-split 결합 커널 — 🔶 **DEFERRED (후보, P0-3 완료 후 재평가)** (2026-04-13 신설)
- **Status**: **DEFERRED** (P0-5/P0-5b/P0-5c 결합 재시도 필요 확인 후 활성화)
- **Sprint**: backlog
- **담당**: senior-implementer (P0-5/P0-5b/P0-5c 커밋 resurrect + 확장)
- **Dependencies**: P0-3 완료 권장 (head_dim=128 매크로화 경험 + Qwen prefill GPU dispatch 확보 후 decode 트랙 재진입)
- **예상 작업량**: 3~5일
- **코드 위치** (resurrect 대상):
  - 커밋 5f3dd1f — P0-5 KV-split 커널 (`flash_attn_f32_f16_q1_split.cl` 추정)
  - 커밋 9a5a75e — KV-split dispatch 로그
  - 커밋 f14ddb3 — Plan-path KV-split 통합 (P0-5b)
  - 커밋 5f59455 — P0-5c GQA reuse 커널 + plan 통합 + 테스트 + research doc
- **Description**:
  - P0-5c revert 후 남은 방향. **GQA reuse + KV-split 두 축을 직교 결합**.
  - Dispatch: `gws=[WG_SIZE, n_heads_kv, n_kv_splits]`
    - Qwen 2.5 1.5B (n_heads_kv=2): 2×8 = **16 WG** (ctx 4K, kv_splits=8)
    - Llama 3.2 1B/3B (n_heads_kv=8): 8×8 = **64 WG**
  - Adreno 830 SP 점유율 충분 확보 → P0-5c의 WG=2 under-utilization 문제 + P0-5의 launch overhead 문제 동시 해소.
  - 같은 KV-head를 공유하는 Q-head 그룹은 local memory KV 한 번 로드 (P0-5c 패턴) + 시퀀스 축은 split 으로 추가 병렬화 (P0-5 패턴) + 2nd pass reduce.
  - **권장 검증 순서**: Llama 3.2 3B (n_heads_kv=8, head_dim=128) 벤치에서 먼저 검증 → Qwen 적용. WG 수가 큰 구성에서 먼저 안전 마진 확보.
- **참조**:
  - llama.cpp `ggml-cuda/fattn-vec-f16.cu` — `parallel_blocks` (KV-split) + `ncols` (Q-group) 동시 활용 패턴
  - 우리 보존 커밋 5f3dd1f/f14ddb3/5f59455 의 코드 베이스
  - 10.1.6 "2026-04-13 추가 실측 검증" 의 net negative 분석
- **Acceptance Criteria**:
  - [ ] 4K decode (Qwen 2.5 1.5B, Adreno 830, plan ON): 7.4 → **12+ tok/s** (llama.cpp 14.9의 80%+)
  - [ ] VRAM 트래픽: 24 MB/layer/token → **6~8 MB** (3~4× 감소, GQA 부분 reuse + split 페치 재사용)
  - [ ] Short ctx (128 tok) 회귀 없음
  - [ ] 정확도: cos-sim ≥ 0.999 vs baseline
  - [ ] (선택) Llama 3.2 3B 측정에서 동일 또는 더 큰 개선 확인 (WG=64 풀 점유)
- **Notes**: P0-5c revert 보고에서 명시된 "재개 조건" 의 구체화. 단독 P0-5/P0-5c 모두 net negative 였으므로 결합만이 유의미. 현 sprint는 deferred — P0-3 완료로 Qwen prefill GPU 경로 확보 후, remaining gap 분석에서 decode attention 비중이 여전히 크면 활성화. **현재 우선순위는 P0-3 (prefill head_dim=128)**.

---

#### [P0-4] ~~OpenCL flash_attn prefill 커널 Q4_0 KV 지원~~ → **P2 강등 (future)** — 현재 기본 벤치 구성(KV=F16)과 무관
- **Status**: DEFERRED (2026-04-13 priority 강등; 기본 벤치는 `--kv-type f16`, 이 항목은 `--kv-type q4_0` 등 KV=Q4_0 구성을 쓰는 다른 사용 사례에만 필요)
- **Sprint**: backlog (future — KV=Q4_0 구성이 실전에서 요구될 때 재활성화)
- **담당**: researcher (선조사) → senior-implementer (구현)
- **Dependencies**: P0-3 완료 권장 (커널 매크로화 경험 선행)
- **예상 작업량**: researcher 1 ~ 2일 + senior-implementer 3 ~ 5일
- **코드 위치**:
  - `engine/src/backend/opencl/mod.rs::flash_attention_prefill_gpu` (현재 Q4_0 분기에서 `_ => return Ok(false)`)
  - `engine/kernels/flash_attn_f32_f16*.cl` (신규 `flash_attn_f32_q4_0*.cl` 파생)
  - 참고: 기존 Q4_0 GEMV 커널 (`mul_mv_q4_0_*.cl`), Q4_0 블록 layout
- **참조**:
  - llama.cpp의 Q4 KV cache 지원 상태 조사 (대체로 F16 권장하는 이유 포함)
  - `docs/31_perf_comparison_llama_cpp.md`
  - 프로젝트 내 Q4_0 dequant 커널 (`simple_ops.cl`, KIVI 관련)
- **Description**:
  - **Phase A (Researcher)**: Q4_0 KV를 GPU flash attention에서 처리하는 현실적 전략 평가:
    (a) 타일 로드 시 on-the-fly dequant (Q4_0 → F16 in private/local mem) 후 기존 F16 커널 재사용
    (b) K/V를 F16으로 사전 변환한 shadow buffer 유지 (메모리 2x 비용, 코드 단순)
    (c) Q4_0 블록 단위로 fused matmul (complex, 큰 개발 비용)
  - 각 전략별 성능/정확도/VRAM 트레이드오프 문서화, 권고안 제시.
  - **Phase B (senior-implementer)**: 선택된 전략 구현. 가장 유력한 후보는 (a) — K 타일 로드 지점에서 Q4_0 block을 subgroup 단위로 dequant하여 local/private memory F16 타일 구성, 이후 기존 online softmax + Pv matmul 재사용.
- **Acceptance Criteria**:
  - [ ] Researcher 산출물: Q4_0 GPU flash attention 전략 비교 문서 (전략 (a)/(b)/(c) 성능/메모리/복잡도 분석)
  - [ ] `flash_attention_prefill_gpu`가 `DType::Q4_0` 케이스에서 `Ok(true)` dispatch
  - [ ] 4K prefill (Qwen2.5 1.5B, Q4_0 KV, GPU): 현재 CPU fallback 수치 대비 대폭 개선
  - [ ] 수치 정확도: F16 KV baseline 대비 logit cos-sim ≥ 0.995 (Q4_0 quantization loss 고려)
  - [ ] VRAM 사용량이 F16 KV 대비 +50% 이내 (shadow buffer 전략 시 제약)
  - [ ] CPU Q4_0 prefill fallback 경로는 유지 (비-OpenCL 빌드/호스트 GPU 호환)
- **Notes**: **2026-04-13 17:00 priority 강등** — 10.1.5 정정으로 "Q4_0은 실전 기본 구성" 주장이 **오판**임이 확인됨. 기본 벤치 구성은 Weights=Q4_0 + **KV=F16**이며, `flash_attention_prefill_gpu` Q4_0 분기는 **KV dtype 기준**이므로 현재 벤치에서는 발동하지 않음. 이 항목은 미래에 `--kv-type q4_0` (KV 자체를 Q4_0으로 저장) 구성을 쓰는 사용 사례가 생길 때 재활성화. Researcher 선조사 결과에 따라 작업 범위 / acceptance 수치 재조정. 전략 (b) 선택 시 P1-1 (chunk sizing)의 메모리 계산식도 함께 갱신 필요.

---

#### [P1-1] GPU prefill chunk auto-sizing fix
- **Status**: DONE (2026-04-13, 커밋 15f99e0)
- **Sprint**: current
- **담당**: implementer (sonnet)
- **Dependencies**: 없음
- **예상 작업량**: 0.5일
- **코드 위치**: `engine/src/bin/generate.rs` (prefill 루프, chunk size 결정부) + `engine/src/backend/opencl/mod.rs` (buffer size 계산)
- **Description**: `--prefill-chunk-size 0` (기본값) 시 `CL_INVALID_BUFFER_SIZE`로 실패. GPU 가용 메모리와 모델 hidden dim에서 안전한 기본 chunk size를 자동 산출 (예: 512 또는 사용 가능 메모리 / (hidden_dim * 4) 중 작은 값).
- **Acceptance Criteria**:
  - [ ] `--prefill-chunk-size 0` 호출 시 에러 없이 정상 동작
  - [ ] 4K prompt GPU prefill 성공
  - [ ] 명시적 `--prefill-chunk-size 128` 결과와 수치 동등 (FP tolerance 내)
  - [ ] 로그에 자동 선택된 chunk size 출력
- **검증 명령**:
  ```bash
  # 자동 chunk 결정 정상 동작 확인
  adb shell "cd /data/local/tmp && ./generate \
    -m /data/local/tmp/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-v2.gguf \
    --prompt-file /data/local/tmp/prompt_6k.txt \
    -n 8 -b opencl --kv-type f16 --max-seq-len 6144 --ignore-eos"

  # 명시적 chunk size와 결과 동등성 확인
  adb shell "... --prefill-chunk-size 128 ..."
  ```
- **Notes**: 낮은 위험도, 빠른 개선. GPU 4K prefill 측정 재개를 위한 선결 조건.

---

#### [P1-2] Prefill OpProfiler 확장 + GPU fallback 감지 로그 (중요도 상승)
- **Status**: DONE (2026-04-13, 커밋 f0091cc)
- **Sprint**: current
- **담당**: implementer (sonnet)
- **Dependencies**: 없음 (P0-2와 병행 가능)
- **예상 작업량**: 0.5 ~ 1일 + fallback 로그 0.25일
- **코드 위치**:
  - `engine/src/core/profiler.rs` (OpProfiler)
  - `engine/src/layers/transformer_layer/forward.rs` (prefill path hooks, `!gpu_dispatched` 분기)
  - `engine/src/backend/opencl/mod.rs::flash_attention_prefill_gpu` (Ok(false) 반환 지점들)
- **Description**: 현재 OpProfiler가 decode 중심. Prefill에서도 per-op (qkv_matmul, rope, kv_write, flash_prefill, ffn_*) 타이밍을 남기도록 hook 추가. `--profile` flag 시 prefill 경로에서도 `synchronize()` + timestamp 캡처. **추가 (2026-04-13)**: 10.1.5에서 밝혀진 GPU prefill CPU fallback 문제의 가시성 확보를 위해, GPU 커널이 `Ok(false)` 반환하여 CPU로 fallback할 때 **stderr 경고 로그**를 남긴다 (이유 포함: "Q4_0 not supported", "head_dim=128 not supported" 등).
- **Acceptance Criteria**:
  - [ ] `generate --profile` 실행 시 prefill per-op breakdown이 stderr/JSON에 출력
  - [ ] Decode 결과 포맷과 호환 (dashboard 파싱 가능)
  - [ ] 4K prefill 전/후 P0-2 효과를 per-op 단위로 측정 가능
  - [ ] **GPU backend 지정 상태에서 prefill이 CPU로 fallback될 때마다 stderr에 1회 이상 경고 로그 출력 (dtype, head_dim, 이유 포함)**
  - [ ] **fallback 이벤트가 OpProfiler 출력에도 "cpu_fallback_dispatch" 이벤트로 기록**
  - [ ] 중복 로그 방지: 같은 (dtype, head_dim) 조합은 per-run 1회만 출력 (또는 dedupe 카운터)
- **Notes**: 이전 세션 권고 사항. P0-2의 효과 측정과 원인 C 조사의 필수 전제. **중요도 상승**: 10.1.5 발견으로, fallback 감지 없이는 GPU 최적화 작업(P0-3/P0-4/P2-1)이 실제 유효한지 구분할 수 없음. P0-3/P0-4 착수 **전에** 완료 권장.

---

#### [P2-1] GPU Adreno F16 matmul 커널 (Subgroup + Texture) — 🟡 **Phase B 1차 완료 (2026-04-13)**
- **Status**: **Phase A 완료 + Phase B 1차 완료 (2026-04-13)** — Q4_0 tiled GEMM 커널 적용. Qwen prefill **27.8 → 50.0 tok/s (+80%)**. 목표(150+) 미달이나 회귀 해소 + 의미있는 진전. 추가 최적화 (subgroup intrinsics, texture cache) 는 후속 작업.
- **Sprint**: ~~current~~ → 후속 (추가 최적화 시 재오픈)
- **2026-04-13 Phase B 결과**:
  - 신규 커널: `engine/kernels/mul_mm_q4_0_f32_l4_lm.cl` (llama.cpp 포팅, interleaved BlockQ4_0 레이아웃 적용)
  - Dispatch: `matmul_q4_0()` 에서 m≥32일 때 `matmul_gemm_q4_0()` 로 분기 (GEMV는 m<32, m=1 noshuffle 보존)
  - **Galaxy S25 (Adreno 830) 측정**:
    - Qwen 2.5-1.5B Q4_0 prefill 4472 tok: **27.8 → 50.0 tok/s (+80%)** (master vs Phase B 동일 thermal status 1)
    - Qwen Decode 7.4 → 8.8 tok/s (+19%, thermal variance 포함)
    - Llama 3.2 1B Q4_0 prefill 4449 tok: 136.3 tok/s (회귀 없음 — Llama는 head_dim=64 + FFN 작아 이미 최적)
  - **목표 미달 분석**: 50 tok/s vs 목표 150+. 잔여 갭 원인 추정 — flash_attn (head_dim=128 prefill) 비중 큼, GEMM tile 사이즈가 Adreno에 비최적, subgroup 미사용. 후속 P2-1b 로 분리 검토.
- **담당**: researcher 선조사 → senior-implementer 구현
- **Dependencies**: P0-3 ✅ 완료 (Qwen prefill GPU 경로 활성화로 fallback 아닌 실제 커널 측정 가능). P1-1 / P1-2 ✅ 완료
- **예상 작업량**: 4 ~ 7일 (연구 2일 + 구현 3 ~ 5일)
- **코드 위치**:
  - 기존: `engine/kernels/mul_mv_f16_f32.cl`, `matmul*.cl` 계열, `engine/kernels/flash_attn_f32_f16*.cl` (prefill DK=128)
  - 신규 예정: `engine/kernels/mul_mm_f16_adreno.cl` (Prefill GEMM 특화), 또는 기존 flash attention 커널의 Adreno-특화 버전
- **참조 문서**: 
  - llama.cpp `ggml-opencl.cpp` Adreno subgroup 커널 (`gemv_noshuffle.cl` 등)
  - `mul_mv_q4_0_*.cl` (이미 Adreno 최적화된 Q4_0 GEMV 커널 — F16 계열에 이식 가능성)
  - `arch/gpu_backend.md` (있다면)
  - `docs/31_perf_comparison_llama_cpp.md` Phase 3 섹션
- **Description**:
  - P0-3 완료로 Qwen prefill GPU 활성화됐으나 속도 미달(CPU 43 > GPU 31). Register spill 아님 확정, **Adreno 특화 최적화 부재가 원인**.
  - **Phase A (Researcher)**: llama.cpp Adreno 계열 커널 패턴 분석 — subgroup size, local mem tiling, texture cache(image2d_t) 사용 패턴 정리. `cl_qcom_extended_register_size`, subgroup intrinsics (Qualcomm extensions). 우리 현재 커널과 diff. 우리 프로젝트 내 이미 Adreno-친화로 작성된 `mul_mv_q4_0_*.cl` 패턴을 F16 계열 (특히 `flash_attn_f32_f16` prefill DK=128) 에 이식 가능성 평가.
  - **Phase B (senior-implementer)**: Prefill 핵심 커널 (`flash_attn_f32_f16` prefill DK=128) 의 Adreno 특화 변형. Subgroup reduction + texture(image2d_t)로 K 행 재사용 극대화. 필요 시 decode `_q1` 도 포함. 기존 일반 커널은 fallback으로 보존.
- **Acceptance Criteria**:
  - [ ] Research 산출물: Adreno subgroup 커널 작동 원리 + 적용 가능성 평가 문서
  - [ ] 4K prefill (Qwen 2.5 1.5B, Adreno 830, plan ON, 6T): **31 → 100+ tok/s** (llama.cpp 760의 13%+ 회복; 절대값은 researcher 보고 후 재조정)
  - [ ] 4K decode 회귀 없음 (현재 7.3 tok/s 유지)
  - [ ] Non-Adreno (NVIDIA/Intel OpenCL) fallback 유지 — 기존 범용 커널 보존
  - [ ] `test_backend` 정확도: CPU vs GPU cos-sim ≥ 0.999
- **Notes**:
  - ROI 불확실성과 작업량 때문에 원래 P2였으나, P0-3 완료로 GPU dispatch가 실제 일어나는 상태가 되었고 prefill GPU(31) < CPU fallback(43) 회귀가 확인됨에 따라 **최우선 트랙으로 승격**.
  - 대상 커널: `flash_attn_f32_f16` prefill (DK=128). Adreno subgroup + texture cache 미사용이 회귀의 근본 원인.
  - `.cl` 수정 허용 규칙은 `feedback_cl_modification.md` 참조.

---

### 10.3 권장 실행 순서 (2026-04-13 20:30 최종 정리 — P0-3 완료, P2-1 최우선 승격)

**핵심 업데이트 (2026-04-13 세션 종료 시점)**: P0-3 기능 완료 (Qwen prefill GPU dispatch 활성화)로 CPU fallback 탈출. 그러나 GPU prefill (31 tok/s) < CPU fallback (43 tok/s, -28%)로 회귀가 확인되어 **P2-1 (Adreno F16 matmul 커널)**이 최우선으로 승격됨. P0-5/P0-5b/P0-5c는 모두 revert, P0-5d는 Llama 3.2 3B 도입 후 재평가로 deferred.

**병렬 트랙 A (CPU 성능 / fallback 완화)**:
1. ~~P0-1 rework~~ **PARTIAL DONE (커밋 c794292+9abd8e0; 11.2 → 12.0 tok/s, +7.1%)** — 목표 14+ 미달분은 P0-5 기인 가능성
2. P0-2 F16 직접 경로 prefill (senior-implementer, 1~2일)

**병렬 트랙 A' (GPU 커널 확장 — 본질적 해결, 최우선)** — 2026-04-13 20:00 재재재정렬 (P0-5c revert 후):
**트랙 A' (GPU 커널 확장) — 최종 정리**:
1. 🔥 **P2-1 Adreno F16 matmul 커널** — **최우선** (researcher 2일 → senior-implementer 3~5일). Prefill 31 → 100+ tok/s 목표.
2. ~~**P0-3 head_dim=128 prefill 커널**~~ **DONE 기능 / 성능 회귀 (커밋 7f41d2f + 37aaab2, 2026-04-13)** — Qwen prefill GPU dispatch 활성화 ✅, CPU fallback 로그 제거 ✅. 단 GPU 31 tok/s < CPU 43 tok/s. 잔여 갭은 P2-1로 이관.
3. ~~**P0-5 decode KV-split 커널/dispatch**~~ **REVERTED (커밋 d0d4978)** — 4K Adreno 830 net regression (-11%) 확정. 코드는 5f3dd1f/9a5a75e/f14ddb3 에 보존. **결합 작업은 P0-5d 로 이관**.
4. ~~**P0-5b Plan-path KV-split 통합**~~ **REVERTED (커밋 d0d4978)** — Plan 통합 동작 확인했으나 P0-5 본체가 net negative라 함께 revert. **결합 작업은 P0-5d 로 이관**.
5. ~~**P0-5c GQA-aware KV 공유 커널**~~ **REVERTED (커밋 84c82d2)** — Qwen 4K 측정에서 net regression (7.3 → 2.6 tok/s, -65%). 코드는 커밋 5f59455 에 보존. 결합 작업은 **P0-5d** 로 이관.
6. 🔶 **P0-5d GQA + KV-split 결합 커널** — **DEFERRED (backlog)**. Llama 3.2 3B 등 다른 모델 벤치 인프라 도입 후 재평가. WG 수가 큰 구성에서 먼저 검증 후 Qwen 적용.
7. ~~P0-4 Q4_0 prefill 커널~~ **DEFERRED (P2/future)** — 현재 벤치 구성(KV=F16)과 무관

**병렬 트랙 B (GPU 재개 + 가시성)** — **완료**:
1. ~~P1-1 GPU chunk auto-sizing~~ **DONE (커밋 15f99e0)**
2. ~~P1-2 Prefill OpProfiler 확장 + fallback 감지 로그~~ **DONE (커밋 f0091cc)**

**트랙 C (CPU prefill F16 경로)** — 차순위:
1. **P0-2** CPU prefill F16 직접 경로 (Step 3 Stage 9). GPU prefill 활성화로 우선순위 낮아졌으나 non-GPU 디바이스/fallback 품질 위해 차순위 유지.

**마일스톤 재정의 (2026-04-13 세션 종료)**:
- **M1 (완료)**: P0-1 rework (partial) + P1-1 + P1-2 완료 → decode 11.2 → **12.0 tok/s** (Qwen CPU 4K, +7.1%), GPU prefill fallback 가시화, chunk auto-sizing 정상화.
- **M2 (완료)**: **P0-3 기능 완료 → Qwen prefill GPU 활성화 (head_dim=128 dispatch, CPU fallback 탈출)**. 단 perf 회귀로 P2-1 후속 작업 필수.
- **M3 (현재, 🔥 진행 중 타겟)**: **P2-1 완료 → Adreno F16 matmul 커널 적용으로 prefill 31 → 100+ tok/s**. Subgroup + texture cache 기반.
- **M4 (차후)**: **P0-2 완료 → CPU prefill F16 직접 경로**. Non-GPU fallback 품질 + CPU-only 디바이스 지원.
- **M5 (조건부)**: **P0-5d (GQA + KV-split 결합) 활성화** — Llama 3.2 3B 벤치 도입 후 WG 충분 구성에서 재평가.

---

### 10.4 핵심 acceptance criteria 요약

| ID | 핵심 지표 | 목표 | 현재 상태 |
|---|---|---|---|
| P0-1 | 4K decode (CPU 6T, **i8mm 경로 dispatch되는 실측 환경**) | 11.2 → 14+ tok/s | **PARTIAL DONE** (12.0 tok/s, +7.1%) |
| P0-2 | 4K prefill (CPU) | 37.7 → 60+ tok/s, bulk F16→F32 제거 | TODO |
| P0-3 | F16 KV, head_dim=128 prefill GPU dispatch | `Ok(false)` fallback 제거, Qwen prefill GPU 활성화, CPU fallback 대비 개선 | **DONE 기능 / 성능 회귀 (커밋 7f41d2f + 37aaab2, 2026-04-13) — dispatch ✅, 31 tok/s < CPU 43 (-28%)** |
| ~~P0-5c~~ | ~~GQA-aware KV 공유 커널 (local memory 공유)~~ | ~~baseline 7.3 → 12+ tok/s, VRAM 24→~4 MB/layer~~ | **REVERTED (커밋 84c82d2, 2026-04-13) — Qwen 4K: 7.3 → 2.6 (-65%); n_heads_kv=2 → WG=2 under-utilization on Adreno 830** |
| ~~P0-5~~ | ~~4K decode KV-split 커널/dispatch~~ | ~~커널 컴파일 + dispatch + split active~~ | **REVERTED (커밋 d0d4978, 2026-04-13) — net regression -10~11% @ 4K Adreno 830** |
| ~~P0-5b~~ | ~~Plan-path KV-split 통합~~ | ~~baseline 7.3 대비 회귀 없음~~ | **REVERTED (커밋 d0d4978, 2026-04-13) — plan+KV-split 6.5 tok/s (-11% vs 7.3 baseline)** |
| 🔶 P0-5d | GQA + KV-split 결합 커널 (`gws=[WG_SIZE, n_heads_kv, n_kv_splits]`) | 4K decode 7.4 → 12+ tok/s, VRAM 24→6~8 MB/layer, Qwen 16WG / Llama 64WG | **DEFERRED (backlog, P0-3 완료 후 재평가)** |
| ~~P0-4~~ | ~~Q4_0 KV prefill GPU dispatch~~ | ~~cos-sim ≥ 0.995~~ | **DEFERRED (future, P2)** |
| P1-1 | `--prefill-chunk-size 0` | 실패 → 정상 동작 | **DONE (커밋 15f99e0)** |
| P1-2 | Prefill per-op 프로파일 + fallback 로그 | 출력 가능, GPU→CPU fallback 경고 stderr | **DONE (커밋 f0091cc)** |
| **P2-1** 🟡 | **4K prefill (Qwen, Adreno 830, plan ON, 6T)** | 27.8 → 50.0 tok/s (+80%) — 목표 150+ 미달 | **Phase B 1차 완료 (Q4_0 GEMM)** |
| 공통 | 수치 정확도 | logit cos-sim ≥ 0.999 (test_backend) | — |

---

### 10.5 참고 문서

- 코드 위치 근거: 외부 분석 결과와 현 코드 매칭 완료
- `docs/31_perf_comparison_llama_cpp.md` — Phase 1~3 비교 분석
- `arch/cpu_flash_attention_prefill.md` — Stage 9 F16 직접 경로 설계
- `feedback_cl_modification.md` — `.cl` 커널 수정 허용 정책
- llama.cpp 참조: `reference_llama_cpp_source.md`, `reference_llama_cpp_testing.md`

---

## 11. 다음 세션 시작 가이드 (2026-04-13 21:30 세션 종료 시점 — P2-1 Phase B 1차 완료)

### 현재 master 상태
- 최신 커밋: `2cb9d4a feat(opencl): Q4_0 tiled GEMM kernel for prefill (P2-1 Phase B)`
- 빌드: Android aarch64 release OK, 호스트 `cargo check --all-targets` 클린 (rust-analyzer stale 경고만 잔존)
- 작업트리: tracked files clean (untracked `.agent/research/`, eval_stderr.txt 등만 존재)

### 측정된 현재 성능 (Qwen 2.5 1.5B Q4_0, Adreno 830, F16 KV, 4K, plan ON)
| 항목 | 이전 baseline (37aaab2) | **Phase B 적용 (2cb9d4a)** | Δ |
|------|-----|-----|---|
| Prefill (4472 tok) | 27.8 tok/s | **50.0 tok/s** | +80% |
| Decode | 7.4 tok/s | 8.8 tok/s | +19% |
| TBT | 148 ms | 124 ms | −16% |

- 잔여 격차: Prefill **15× vs llama.cpp(760)**, Decode 1.7× vs llama.cpp(14.9)
- Llama 3.2 1B Q4_0 Prefill: 136.3 tok/s (회귀 없음 — F16 GEMM 경로 무변경)

### 이번 세션(2026-04-13) 완료 작업 요약
**커밋 (master)**:
- `c794292` perf(neon): Q4_0 vec_dot fallback (vpaddlq chain) — P0-1 partial
- `9abd8e0` perf(neon): Q4_0 i8mm hot paths, 2-block unroll — P0-1 partial
- `15f99e0` fix(gpu): prefill chunk auto-sizing — P1-1 DONE
- `f0091cc` feat(profiler): prefill per-op + GPU fallback log — P1-2 DONE
- `7f41d2f` feat(opencl): GPU flash prefill head_dim=128 — P0-3 기능 DONE
- `37aaab2` perf(opencl): tune BLOCK_M=32 for DK=128 — P0-3 튜닝 (effect=0)
- `2cb9d4a` feat(opencl): Q4_0 tiled GEMM kernel for prefill (P2-1 Phase B) — **이번 세션 핵심**

**Revert (history 보존)**:
- `d0d4978` ← `5f3dd1f / 9a5a75e / f14ddb3` — P0-5/P0-5b Flash Decoding KV-split (-11%)
- `84c82d2` ← `5f59455` — P0-5c GQA-aware KV 공유 (-65%, WG=2 under-utilization)

**핵심 발견 (Phase A 선조사)**:
- 24× 갭의 주범은 **Q4_0 타일드 GEMM 커널 부재** (flash_attn 아님)
- F16 GEMM은 이미 llama.cpp와 byte-identical, parity 상태
- Phase B에서 `mul_mm_q4_0_f32_l4_lm.cl` 포팅 + `matmul_q4_0()` m≥32 분기 → +80% 달성
- 포팅 시 interleaved `BlockQ4_0` 직접 사용 (rewrap 없이, byte-level dequant load)
- 보고서: `.agent/research/2026-04-13_p2-1_adreno_f16_matmul_phase_a.md` (untracked)

### 다음 세션 착수 옵션

**A. 🔥 P2-1b Follow-up: Prefill 추가 최적화 (권장, 최우선)**

목표: Prefill 50 → 150+ tok/s. 잔여 15× 갭 해소.

- **A-0. 프로파일 먼저 (필수, 30분)** — Phase B 적용 후 op breakdown 재측정. Q4_0 GEMM이 병목에서 빠졌으니 현재 hot path는 flash_attn DK=128 prefill / LM head GEMV 가능성. `--profile` per-op ms 분포 확인 후 A-1/A-2/A-3 중 선택.

- **A-1. Q4_0 GEMM Adreno subgroup + image2d_t 버전 (High ROI, 중간 난이도)**
  - 현재 `mul_mm_q4_0_f32_l4_lm.cl`은 vanilla local memory only (buf 2×8KB)
  - Adreno 830은 `gemv_noshuffle.cl` 패턴(subgroup 64-wide + image1d_buffer TP cache)이 GEMV에서 2~3× 빠름
  - GEMM에서도 텍스처 캐시로 weight read 가속화 가능 (llama.cpp 대응 커널 부재 → 우리가 먼저)
  - 리스크: Adreno 전용 분기 (`#ifdef ADRENO_GPU`), weight rewrap 추가 필요 (SOA AOS-row-major)
  - 기대: Prefill 50 → 90~120 tok/s

- **A-2. Q4_0 GEMM tile size 튜닝 (Low ROI 예상, 저난이도)**
  - 현재 BM=64, BN=64, BK=32, TM=4, TN=8, nth=128
  - BLOCK_M=32 DK=128 flash tuning 효과 0 전례 → 제한적 가능성
  - Adreno 830 레지스터 압박/WG 한계 기준 BM=32/BK=64 variant 확인 가치 있음
  - 기대: Prefill 50 → 55~70 tok/s (보조)

- **A-3. Flash Attn DK=128 Prefill 최적화 (Medium ROI, 고난이도)**
  - matmul 병목 해소된 지금 flash_attn이 새 병목일 수 있음 — 프로파일 필수 (A-0)
  - 필요 시 subgroup reduction 기반 online softmax 재작성
  - 기대: Prefill 50 → 80~100 tok/s (단독), A-1과 중첩 시 시너지

**권장 순서**: A-0 → A-1 → A-3 재평가 → 필요 시 A-2.

착수 명령: `.agent/todos/long_context_attention_optimization.md §11 읽고 A-0 프로파일부터 해줘`

**B. P0-2 CPU prefill F16 직접 경로 (차순위)**
- senior-implementer 구현 (1~2일). CPU fallback 속도 개선 (37→60+ tok/s).
- GPU prefill 활성화 + Phase B 완료로 우선순위 낮음. non-GPU 디바이스 지원 필요 시 승격.

**C. P0-5d GQA + KV-split 결합 커널 (DEFERRED 유지)**
- Llama 3.2 3B 등 multi-모델 벤치 인프라 도입 후 재평가.
- Qwen 단독(n_heads_kv=2)에서는 P0-5c revert 근거 유효.

### 재현용 벤치마크 명령
```bash
# Android 빌드 (Linux 호스트)
export NDK_HOME=/opt/android-ndk
export HOST_TAG=linux-x86_64
export TOOLCHAIN=$NDK_HOME/toolchains/llvm/prebuilt/$HOST_TAG
export CC_aarch64_linux_android=$TOOLCHAIN/bin/aarch64-linux-android21-clang
export CXX_aarch64_linux_android=$TOOLCHAIN/bin/aarch64-linux-android21-clang++
export AR_aarch64_linux_android=$TOOLCHAIN/bin/llvm-ar
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=$TOOLCHAIN/bin/aarch64-linux-android21-clang
cargo build --target aarch64-linux-android --release -p llm_rs2 --bin generate

adb push target/aarch64-linux-android/release/generate /data/local/tmp/generate
adb shell chmod +x /data/local/tmp/generate

# 쿨다운 + 4K bench (1회 규칙)
sleep 240
adb shell "dumpsys thermalservice | grep -E 'Thermal Status|AP,'"  # Status 0 확인
adb shell "cd /data/local/tmp && ./generate \
  -m /data/local/tmp/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-v2.gguf \
  --prompt-file /data/local/tmp/prompt_6k.txt \
  -n 128 -b opencl --kv-type f16 --max-seq-len 6144 --ignore-eos"
```

### 참고 문서 / 보존된 작업물
- Phase 2 상세: 본 파일 §10 (P2-1 §10.2는 Phase B 결과로 갱신됨)
- §10.1.6: GPU decode 구조적 결함 분석 (KV-split 가설 + 정정 이력 포함)
- P0-5/P0-5b/P0-5c revert 근거: 각 항목 Notes 블록
- **P2-1 Phase A 선조사**: `.agent/research/2026-04-13_p2-1_adreno_f16_matmul_phase_a.md` (untracked, 보존 필요)
- P0-5 선조사 복원: `git show d0d4978~1:.agent/research/2026-04-13_flash_decoding_kv_split.md`
