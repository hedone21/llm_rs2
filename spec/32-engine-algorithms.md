# Engine Algorithms

> **TL;DR**: Engine 내부 알고리즘의 상세를 정의한다. KV 캐시 eviction 4종(H2O, Sliding Window, D2O, StreamingLLM), KIVI 비대칭 양자화(flush, bit transition, incremental dequant, GPU mode), SWIFT 기반 layer skip, Unified QCF (ENG-ALG-051, attention output perturbation 통합 메트릭) + legacy proxy 8종(deprecated) 수식, RequestQcf dry-run 6종 action 스캔, DegradationEstimator의 piecewise-linear 보정, madvise/shrink_to_fit 물리 메모리 해제, chunked prefill, CachePressurePipeline의 Handler 체인, 추론 루프 resilience checkpoint를 기술한다. 변수명과 타입 표현은 자유이나, 의사코드/불변식/example trace를 포함한다.

## 1. Purpose and Scope

이 문서는 Engine 내부 알고리즘의 **핵심 로직, 수식, 불변식, 실행 순서**를 정의한다.

**이 파일이 명세하는 것:**

- KV Cache Eviction 알고리즘 4종 (H2O, Sliding Window, D2O, StreamingLLM)
- KV Cache Quantization (KIVI): flush cycle, bit transition, incremental dequant, GPU mode
- Layer Skip: SkipConfig 초기화, layer importance 계산
- Unified QCF (ENG-ALG-051): attention output perturbation 통합 메트릭 + legacy proxy 8종 (deprecated)
- RequestQcf → QcfEstimate 흐름 (F-01 해결)
- DegradationEstimator: piecewise-linear + EMA 보정
- madvise/release_unused_pages: CPU path, shrink_to_fit, GPU MadviseableGPUBuffer (F-02, F-04 해결)
- Chunked Prefill
- CachePressurePipeline: Handler 체인 6종 (F-03 해결)
- Inference Loop Resilience Checkpoint

**이 파일이 명세하지 않는 것:**

- Engine 아키텍처 개요 → `30-engine.md`
- 상태 머신 전이 테이블 → `31-engine-state.md`
- 데이터 타입, CLI 설정, KV 캐시 구조 → `33-engine-data.md`
- Manager 알고리즘/데이터 → `20-manager.md` ~ `23-manager-data.md`
- 프로토콜 메시지/시퀀스 → `10-protocol.md` ~ `12-protocol-sequences.md`

## 2. Definitions

| 용어 | 정의 |
|------|------|
| **Heavy Hitter** | H2O에서 누적 attention score가 높은 토큰. 캐시 예산 내에서 보존된다. |
| **Protected Prefix** | eviction에서 절대 제거되지 않는 프롬프트 앞부분 (attention sink). |
| **Recent Window** | eviction에서 가장 최근 생성된 토큰 집합. 항상 보존된다. |
| **Residual Buffer** | KIVI에서 아직 양자화되지 않은 최근 R개 토큰의 FP32 저장 영역. |
| **Flush** | KIVI residual buffer가 가득 차서 양자화 압축 저장소로 이동하는 연산. |
| **QCF Proxy** | 손실 액션의 품질 비용을 근사하는 경량 메트릭. PPL 직접 측정 없이 품질 영향을 추정한다. |
| **NMSE** | Normalized Mean Squared Error. quantize → dequantize round-trip 오차를 분산으로 정규화한 값. |
| **OPR** | Output Perturbation Ratio. `||delta_output|| / ||original_output||`. |
| **CAOTE** | Context-Aware Output Token Error. softmax 재분배 효과와 value 방향 오차를 함께 고려하는 proxy. |
| **AWQE** | Attention-Weighted Quantization Error. attention score로 가중된 양자화 오차. |

## 3. Specification

### 3.1 KV Cache Eviction Algorithms

#### 3.1.1 H2O Eviction [ENG-ALG-010]

**[ENG-ALG-010]** H2O (Heavy-Hitter Oracle) 3-partition eviction은 attention score 기반 토큰 중요도 순위를 사용하여 캐시를 축소한다. *(MUST)*

**3-Partition 구조**:

```
[0..prefix)             = Protected Prefix (절대 불가침)
[prefix..recent_start)  = Evictable Zone (score 기반 선별)
[recent_start..current_pos) = Recent Window (항상 보존)
```

**알고리즘 (의사코드)**:

```
function h2o_evict(cache, target_len, importance[], protected_prefix, keep_ratio):
    if current_pos <= target_len:
        return  // eviction 불필요

    available = target_len - protected_prefix
    hh_budget = floor(available * keep_ratio)
    recent_budget = available - hh_budget
    recent_start = current_pos - recent_budget

    // Evictable Zone: [protected_prefix .. recent_start)
    ranked = sort_by_importance_ascending(positions[protected_prefix..recent_start])

    // 상위 hh_budget개 보존, 나머지 제거
    evicted = ranked[0 .. len(ranked) - hh_budget]

    // keep = protected + heavy_hitters + recent
    keep = [0..prefix) ∪ top_hh(ranked) ∪ [recent_start..current_pos)
    sort(keep)

    cache.compact_keep_positions(keep, write_start=0)
    cache.current_pos = len(keep)
    cache.release_unused_pages()
```

**Importance Score 누적**: AttentionScoreAccumulator에서 decode step마다 post-softmax attention 값을 누적한다. `importance[pos] = Sigma_steps attn(pos)`. 선택적으로 time-normalized (step 수로 나눔) 또는 exponential decay (`h2o_decay` factor) 적용.

**compact_keep_positions 최적화**: 연속 keep 위치를 batch로 묶어 단일 `shift_positions()` 호출로 처리한다. 최대 200x 감소된 `buffer_shift` 호출.

**불변식**:

- `|keep| = min(target_len, current_pos)`
- `[0..prefix) ⊂ keep` (prefix 항상 보존)
- `[recent_start..current_pos) ⊂ keep` (recent 항상 보존)
- eviction 후 `current_pos <= target_len`

---

#### 3.1.2 Sliding Window Eviction [ENG-ALG-011]

**[ENG-ALG-011]** Sliding Window Eviction은 가장 오래된 토큰부터 순서대로 제거하는 FIFO 정책이다. *(MUST)*

**알고리즘**:

```
function sliding_evict(cache, target_len):
    if current_pos <= target_len:
        return
    prune_count = current_pos - target_len
    cache.prune_prefix(prune_count)
    // prune_prefix 내부에서 shift + release_unused_pages 수행
```

**불변식**:

- eviction 후 `current_pos = target_len`
- 보존 토큰은 가장 최근 `target_len`개

---

#### 3.1.3 D2O Eviction + Compensation Merging [ENG-ALG-012]

**[ENG-ALG-012]** D2O (Dynamic Discriminative Operations)는 H2O 스타일 3-partition eviction 후, 제거 대상 토큰을 가장 유사한 잔존 토큰에 merge하여 정보 손실을 보상한다. *(MUST)*

**알고리즘**:

```
function d2o_evict(cache, target_len, importance[], config):
    // Phase 1: H2O-style ranking (3-partition)
    evicted_positions = identify_evicted_h2o(importance, config.prefix,
                                              config.keep_ratio, ...)

    // Phase 2: Per-head cosine similarity matching
    for each evicted_pos in evicted_positions:
        matches = find_nearest_retained_per_head(evicted_pos,
                                                  retained_positions, cache)
        mean_sim = average(matches[h].similarity for h in 0..kv_heads)

        // EMA threshold filtering
        if mean_sim >= ema_threshold:
            // Merge: scatter-reduce evicted token into nearest retained per head
            for h in 0..kv_heads:
                retained = matches[h].nearest_pos
                cache.K[h][retained] += cache.K[h][evicted_pos]  // additive merge
                cache.V[h][retained] += cache.V[h][evicted_pos]
            merged_count += 1
        else:
            // Delete: no compensation
            deleted_count += 1

    // Phase 3: EMA threshold update
    current_mean_sim = mean(all match similarities)
    if not initialized:
        ema_threshold = current_mean_sim
    else:
        ema_threshold = alpha * ema_threshold + beta * current_mean_sim

    // Phase 4: Compact remaining positions
    cache.compact_keep_positions(retained_positions, 0)
```

**D2OConfig 파라미터**:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `keep_ratio` | 0.75 | 논문 기본값 N:M = 3:1 |
| `ema_alpha` | 0.5 | old threshold 가중치 |
| `ema_beta` | 0.5 | new mean 가중치 |
| `protected_prefix` | 4 | eviction 보호 prefix |
| `use_layer_allocation` | false | Phase B, per-layer 예산 배분 |

**Cosine similarity**: per-head 단위로 K 벡터 간 cosine distance 계산. F16/Q4_0 dtype 지원 (내부 F32 변환).

**Layer-level dynamic allocation**: D2O Phase B. prefill 시 수집한 per-layer attention variance를 기반으로 layer별 hh_ratio/recent_ratio 차등 배분. D2OVarianceCollector가 prefill 중 데이터 수집.

**현재 상태**: D2OHandler는 CachePressurePipeline 내 Handler로 구현 완료. KvMergeD2o EngineCommand (MSG-034b)는 executor.rs에서 `EvictPlan { method: D2o, target_ratio: keep_ratio, level: Critical }` 생성으로 처리되며, generate.rs의 `EvictMethod::D2o` 분기에서 `CacheManager::force_evict_with_scores(target_ratio)`를 호출하여 Pipeline 내 persistent D2OHandler를 재활용한다. ActionId::KvMergeD2o가 Manager action_registry에 등록되어 eviction 배타 그룹(C4/C5/C7/C8)에 포함된다. 전제조건: `--eviction-policy d2o`로 시작된 Engine에서만 유효하다 (Pipeline에 D2OHandler가 존재해야 함).

---

#### 3.1.4 StreamingLLM Eviction [ENG-ALG-013]

**[ENG-ALG-013]** StreamingLLM Eviction은 attention sink (시퀀스 선두 토큰)과 recent window (시퀀스 후미 토큰)를 유지하고 중간 토큰을 제거한다. target_len 파라미터를 사용하지 않으며, sink_size + window_size로 유지할 범위가 결정된다. *(MUST)*

**알고리즘**:

```
function streaming_evict(cache, sink_size, window_size):
    keep_len = sink_size + window_size
    if current_pos <= keep_len:
        return  // 제거할 토큰 없음

    // sink 영역: positions [0, sink_size)  — 유지
    // 제거 영역: positions [sink_size, current_pos - window_size)
    // recent 영역: positions [current_pos - window_size, current_pos)  — 유지

    remove_start = sink_size
    remove_end = current_pos - window_size
    remove_count = remove_end - remove_start

    // recent window를 sink 바로 뒤로 이동
    cache.shift_positions(remove_end, remove_start, window_size)
    cache.set_current_pos(keep_len)
```

**불변식**:

- eviction 후 `current_pos = sink_size + window_size`
- positions [0, sink_size)는 항상 보존 (attention sink)
- positions [sink_size, sink_size + window_size)는 eviction 전 가장 최근 window_size개 토큰

**프로토콜 경로**: Manager가 `KvStreaming { sink_size, window_size }` EngineCommand를 전송하면, executor.rs에서 `EvictPlan { method: Streaming, streaming_params: Some(StreamingParams { sink_size, window_size }), target_ratio: 0.0, pressure_level: Critical }`을 생성한다. generate.rs의 eviction 분기에서 `StreamingLLMPolicy::new(sink_size, window_size).evict(cache, 0)` 즉석 호출로 실행한다 (target_len은 무시됨).

---

### 3.2 KV Cache Quantization (KIVI)

#### 3.2.1 Residual Buffer + Flush Cycle [ENG-ALG-020]

**[ENG-ALG-020]** KIVI (ICML 2024) 비대칭 양자화는 최근 R개 토큰을 FP32 residual buffer에 유지하고, 꽉 차면 batch quantize하여 압축 저장소로 flush한다. *(MUST)*

**KV update**:

```
function kivi_update(new_k, new_v):
    // Append to residual buffer (FP32)
    res_k[res_pos] = new_k
    res_v[res_pos] = new_v
    res_pos += 1

    // Flush when residual is full
    if res_pos >= res_cap:
        flush_residual()
```

**flush_residual 알고리즘**:

```
function flush_residual():
    assert res_pos >= group_size  // group_size = QKKV = 32

    n_groups = res_pos / group_size
    flush_tokens = n_groups * group_size

    // 1. QCF proxy 수집 (FP32 원본이 있을 때)
    push QcfMetric(compute_flush_qcf(...))   // NMSE
    push QcfMetric(compute_flush_opr(...))   // OPR
    if awqe_enabled and attn_scores available:
        push QcfMetric(compute_flush_awqe(...))  // AWQE

    // 2. Key: per-CHANNEL quantization
    for each head h:
        for each group g (of group_size tokens):
            for each channel ch in 0..head_dim:
                vals[0..32] = res_k[h][g*32..(g+1)*32][ch]  // 32 tokens, 1 channel
                qk.push_quantized(vals)

    // 3. Value: per-TOKEN quantization
    for each head h:
        for each token t in 0..flush_tokens:
            for each block b in 0..head_dim/32:
                chunk = res_v[h][t][b*32..(b+1)*32]  // 1 token, 32 dims
                qv.push_quantized(chunk)

    q2_tokens += flush_tokens

    // 4. Shift remaining tokens to front
    remaining = res_pos - flush_tokens
    shift res_k/res_v: [flush_tokens..res_pos) -> [0..remaining)
    res_pos = remaining
```

**Key vs Value 양자화 비대칭**:

- **Key**: per-channel (같은 채널의 여러 토큰을 하나의 그룹으로). 이유: attention QK^T에서 채널 간 분산이 크므로 채널별 scale/zero가 효과적.
- **Value**: per-token (같은 토큰의 여러 차원을 하나의 그룹으로). 이유: weighted sum V에서 토큰 간 분산이 크므로 토큰별 scale/zero가 효과적.

**불변식**:

- `q2_tokens`는 항상 `res_cap`의 배수
- `res_pos < res_cap` (flush 직후)
- `total_tokens = q2_tokens + res_pos`

---

#### 3.2.2 Dynamic Bit Transition [ENG-ALG-021]

**[ENG-ALG-021]** KiviCache는 런타임에 양자화 bit-width를 전환할 수 있다 (2, 4, 8 bit). *(MUST)*

```
function transition_bits(new_bits):
    assert new_bits in {2, 4, 8}
    if new_bits == current_bits: return

    if q2_tokens == 0:
        // 양자화 데이터 없음: 포맷만 전환
        qk = new QuantizedBlocks(new_bits)
        qv = new QuantizedBlocks(new_bits)
        bits = new_bits
        return

    // Dequantize -> Re-quantize (오차 누적 주의)
    for each block in qk:
        buf = dequantize(block)
        new_qk.push(quantize(buf, new_bits))

    for each block in qv:
        buf = dequantize(block)
        new_qv.push(quantize(buf, new_bits))

    qk = new_qk; qv = new_qv; bits = new_bits
    q2_deq_tokens = 0  // dequant cache 무효화
```

**QuantizedBlocks**: Q2 (BlockQ2_0), Q4 (BlockKVQ4), Q8 (BlockKVQ8) 3종 variant. 모두 QKKV=32 단위 블록. `push_quantized(vals)` / `dequantize_block(idx, out)` 인터페이스.

---

#### 3.2.3 Incremental Dequantization (assemble_view) [ENG-ALG-022]

**[ENG-ALG-022]** assemble_view()는 이전에 dequantize하지 않은 flush만 incremental 처리하여 attention 계산용 FP32 view를 구성한다. *(MUST)*

```
function assemble_view():
    // 1. Incremental: 이전에 dequant하지 않은 flush만 처리
    if q2_tokens > q2_deq_tokens:
        old_flushes = q2_deq_tokens / res_cap
        new_flushes = q2_tokens / res_cap

        for flush in old_flushes..new_flushes:
            dequantize Key blocks -> attn_k_buf[flush*res_cap..(flush+1)*res_cap]
            dequantize Value blocks -> attn_v_buf[flush*res_cap..(flush+1)*res_cap]

        q2_deq_tokens = q2_tokens

    // 2. Residual: FP32 데이터 복사 (매 step 갱신)
    copy res_k/res_v -> attn_k_buf/attn_v_buf[q2_tokens..q2_tokens+res_pos]
```

**성능**: 첫 flush 이후 incremental dequant만 수행하므로, 추가 flush가 없으면 dequant 비용 0. Residual 복사만 매 step 발생.

---

#### 3.2.4 GPU Mode [ENG-ALG-023]

**[ENG-ALG-023]** KiviCache는 GPU 모드를 지원한다 (`new_gpu()` 생성자). *(MAY)*

- **GPU 버퍼**: 영구 GPU 버퍼 6개 (`gpu_res_k/v`, `gpu_attn_k/v`, `gpu_q2k/v`)
- **Hot path** (update, get_view): GPU 커널 (`kivi_gather_update`) 사용
- **Cold path** (flush quantization): 여전히 CPU에서 수행 (빈도 낮음)
- **Fallback**: GPU 할당 실패 시 CPU 모드로 자동 전환
- **Q2 GPU 저장**: 12 bytes/block U8 텐서로 GPU에 직접 저장

---

### 3.3 Layer Skip

#### 3.3.1 SkipConfig 초기화 [ENG-ALG-030]

**[ENG-ALG-030]** SWIFT (arXiv 2024) 기반 자기 추론적 디코딩. Transformer 레이어의 attention과 MLP를 독립적으로 skip 가능하다. *(MUST)*

**SWIFT 제약**: Layer 0과 Layer L-1은 절대 skip 불가.

**skip_ratio -> skip_layers 변환** (`uniform_init`):

```
function uniform_init(num_layers, skip_ratio):
    total_candidates = (num_layers - 2) * 2  // layer 0, L-1 제외, attn+mlp 각각
    num_skip = round(total_candidates * skip_ratio)

    // 홀수 인덱스 레이어 우선 (1, 3, 5, ...), attn -> mlp 교대
    for i in [1, 3, 5, ...] (i < num_layers-1):
        skip attn[i], then mlp[i]
    // 부족하면 짝수 인덱스 (2, 4, 6, ...)
    for i in [2, 4, 6, ...] (i < num_layers-1):
        skip attn[i], then mlp[i]
```

**Forward pass 적용**: skip된 sub-layer는 identity (residual만 통과). 입력이 그대로 출력.

---

#### 3.3.2 Layer Importance [ENG-ALG-032]

**[ENG-ALG-032]** Layer importance는 prefill 시 1회 계산하여 전 decode step에 재사용한다. *(MUST)*

**Importance 계산**:

```
importance(layer_i) = 1 - cosine_similarity(input_i, output_i)
```

여기서 `input_i`, `output_i`는 해당 레이어 전/후의 hidden state (seq dim에 대해 mean-pool).

**OPR (Output Perturbation Ratio)**:

```
OPR(layer_i) = ||output_i - input_i||_2 / ||input_i||_2
```

**ImportanceCollector 사용법**: prefill loop에서 `snapshot_before(x)` → `layer.forward()` → `record_after(x, layer_id)` → `build()` → `ImportanceTable`.

**QCF for skip**:

```
QCF_skip = Sigma importance(skipped_layers) / Sigma importance(all_layers)
```

범위: [0, 1]. `estimate_qcf_for_count()`는 보호 레이어 제외 후 importance 오름차순으로 가장 저렴한 레이어부터 선택.

---

### 3.4 QCF 계산

#### 3.4.1 Eviction QCF -- Attention x V-norm [ENG-ALG-041] [DEPRECATED — superseded by ENG-ALG-051]

**[ENG-ALG-041]** H2O eviction 시 정보 손실을 추정하는 기본 proxy. Per-head importance를 attention score와 V-norm의 곱으로 계산한다. *(MUST)* **[DEPRECATED — superseded by ENG-ALG-051]**

**수식** (per-head):

```
importance(t) = attn(t) * ||V(t)||_1

raw[h] = Sigma_{t in evicted} importance(t) / Sigma_{t in all} importance(t)         // [0, 1]
normalized[h] = Sigma_{t in evicted} importance(t) / Sigma_{t in remaining} importance(t)  // [0, inf)
```

> **코드와 기존 스펙 불일치 (00-overview SYS-043)**: SYS-043은 normalized_value를 "[0,1] 정규화"로 기술하나, 실제 코드에서 `normalized_value = evicted_importance / remaining_importance`이며 1 이상이 가능하다. 이 스펙은 코드를 따른다.

**Head 집계**:

- **Mean**: `raw_value = mean(raw[h] for h)`
- **Defensive**: softmax-weighted mean (온도 파라미터, worst-case head 강조)

```
AggregationMode::Defensive { temperature }:
    weights[h] = softmax(raw[h] / temperature)
    result = Sigma weights[h] * raw[h]
```

---

#### 3.4.2 Sliding Window QCF -- V-norm Only [ENG-ALG-042] [DEPRECATED — superseded by ENG-ALG-051]

**[ENG-ALG-042]** Sliding window eviction 시 V-norm만으로 proxy를 계산한다 (attention score 불필요). *(MUST)* **[DEPRECATED — superseded by ENG-ALG-051]**

```
per head h:
    total_vnorm = Sigma_{t=0..pos} ||V(h,t)||_1
    evicted_vnorm = Sigma_{t in evicted} ||V(h,t)||_1
    raw[h] = evicted_vnorm / total_vnorm
    normalized[h] = evicted_vnorm / (total_vnorm - evicted_vnorm)
```

---

#### 3.4.3 CAOTE Eviction QCF [ENG-ALG-043] [DEPRECATED — superseded by ENG-ALG-051]

**[ENG-ALG-043]** Context-Aware Output Token Error. Softmax 재분배 효과와 value 방향 오차를 함께 고려한다. *(MUST)* **[DEPRECATED — superseded by ENG-ALG-051]**

```
per head h:
    1. o_mean = Sigma_i alpha_i * v_i                         // attention output
    2. alpha_evicted = Sigma_{j in evicted} alpha_j
    3. amplification = 1 / (1 - alpha_evicted)                // redistribution factor
    4. weighted_residual = Sigma_{j in evicted} alpha_j * (o_mean - v_j)
    5. error[h] = amplification * ||weighted_residual||_2 / ||o_mean||_2
```

주의: `alpha_evicted >= 1` 이면 `error = d_max` (상한 클램프).

---

#### 3.4.4 QCF Attention V2 (경량) [ENG-ALG-044] [DEPRECATED — superseded by ENG-ALG-051]

**[ENG-ALG-044]** V-norm 계산 없이 attention score만으로 QCF를 계산하는 경량 variant. *(MAY)* **[DEPRECATED — superseded by ENG-ALG-051]**

```
per head h:
    sum_evicted = Sigma_{pos in evicted} head_attn[h * max_seq_len + pos]
    raw[h] = sum_evicted
    normalized[h] = 2 * sum_evicted / eviction_ratio
```

---

#### 3.4.5 KIVI Flush QCF -- NMSE [ENG-ALG-045] [DEPRECATED — superseded by ENG-ALG-051]

**[ENG-ALG-045]** KIVI residual flush 시 quantize-dequantize round-trip NMSE로 양자화 오차를 추정한다. *(MUST)* **[DEPRECATED — superseded by ENG-ALG-051]**

```
function compute_nmse_block(original[32], bits):
    reconstructed = dequantize(quantize(original, bits))
    mean = mean(original)
    var = Sigma (original[i] - mean)^2 / 32
    if var < epsilon: return 0
    mse = Sigma (original[i] - reconstructed[i])^2 / 32
    return clamp(mse / var, 0, 1)
```

**합산 (per-head)**:

```
NMSE_K[h] = mean(NMSE over all Key channel blocks for head h)
NMSE_V[h] = mean(NMSE over all Value token blocks for head h)
combined[h] = 0.6 * NMSE_K[h] + 0.4 * NMSE_V[h]
```

> **가중 합산 비율 0.6K + 0.4V**: KIVI 논문 Table 2에서 Key quantization이 quality에 더 민감한 것을 근거로 한다.

---

#### 3.4.6 KIVI Flush OPR [ENG-ALG-046] [DEPRECATED — superseded by ENG-ALG-051]

**[ENG-ALG-046]** V cache 양자화의 output perturbation ratio. K는 2차 효과로 무시한다. *(MUST)* **[DEPRECATED — superseded by ENG-ALG-051]**

```
per head h:
    orig_sum[d] = Sigma_t V_orig[h][t][d]                          // 원본 V 합
    delta_sum[d] = Sigma_t (V_quant[h][t][d] - V_orig[h][t][d])    // 오차 합
    OPR[h] = ||delta_sum||_2 / ||orig_sum||_2
```

---

#### 3.4.7 AWQE (Attention-Weighted Quantization Error) [ENG-ALG-047] [DEPRECATED — superseded by ENG-ALG-051]

**[ENG-ALG-047]** Post-softmax attention score로 가중된 V 양자화 오차. `set_awqe_enabled(true)` 시에만 활성화된다 (기본 비활성). *(MAY)* **[DEPRECATED — superseded by ENG-ALG-051]**

```
per head h:
    error_sum[d] = Sigma_t attn(t) * (V_quant[h][t][d] - V_orig[h][t][d])
    base_sum[d] = Sigma_t attn(t) * V_orig[h][t][d]
    AWQE[h] = ||error_sum||_2 / ||base_sum||_2
```

**GQA 처리**: Q head 그룹의 attention score를 평균하여 KV head별 가중치 생성.

---

#### 3.4.8 Layer Skip QCF [ENG-ALG-048]

**[ENG-ALG-048]** SkipQcfTracker: speculative decoding rejection rate 기반 sliding window proxy. *(MUST)*

```
function record(accepted, drafted):
    rejection_rate = 1 - accepted / drafted
    window.push_back(rejection_rate)
    if window.len > window_size: window.pop_front()

function current_proxy():
    raw_value = mean(window)
    normalized_value = raw_value   // 이미 0~1 범위
```

#### 3.4.9 AW-VOPR (Attention-Weighted Vector Output Perturbation Ratio) [ENG-ALG-049] [DEPRECATED — superseded by ENG-ALG-051]

**[ENG-ALG-049]** KIVI residual flush 시, attention-weighted V 양자화 에러의 벡터 합산 norm을 원본 attention output norm으로 나누어 output-level 양자화 영향을 추정한다. AWQE(ENG-ALG-047)의 벡터 확장으로, 반대 방향 에러의 상쇄를 반영한다. `set_awqe_enabled(true)` 시에만 활성화된다. *(MAY)* **[DEPRECATED — superseded by ENG-ALG-051]**

수학적 정의:

```
O_orig_h = Σ_t α_t × V_t                    (head_dim 벡터)
ΔV_t = V_t - dequant(quant(V_t))
ΔO_h = Σ_t α_t × ΔV_t                      (head_dim 벡터)

per-head:  AW_VOPR_h = ‖ΔO_h‖₂ / max(‖O_orig_h‖₂, ε)
per-layer: AW_VOPR = mean_h(AW_VOPR_h)      (norm-first-then-mean)
```

**AWQE와의 차이**:
- AWQE: `Σ_t α × scalar_error` → 스칼라 합산 (방향 무시)
- AW-VOPR: `‖Σ_t α × vector_ΔV‖` → 벡터 합산 후 norm (상쇄 반영)

GQA 처리: kv_head당 gqa_group_size개 Q-head의 attention weight를 평균하여 α로 사용한다. 집계는 norm-first-then-mean: 각 Q-head별 ‖ΔO‖/‖O_orig‖을 계산한 뒤 평균한다.

복잡도: O(kv_heads × flush_tokens × head_dim). AWQE와 동일. 추가 메모리: head_dim floats (임시 벡터 2개).

---

#### 3.4.10 Unified QCF -- Attention Output Perturbation [ENG-ALG-051]

**[ENG-ALG-051]** Unified QCF: 모든 KV cache lossy action의 품질 비용을 attention output perturbation으로 통합 측정한다. *(MUST)*

ENG-ALG-041~047, 049의 개별 proxy를 대체하는 단일 통합 메트릭. Action 종류에 관계없이 동일한 수식 프레임워크로 QCF를 계산하여, action 간 비용 비교를 가능하게 한다.

**수식**:

```
QCF = ‖O_before - O_after‖ / ‖O_before‖

O_before = Σ_t α_t × V_t                       (현재 attention output, head_dim 벡터)
```

**Action별 O_after 정의**:

```
Eviction (Sliding/H2O/Streaming):
    O_after = Σ_{t ∈ retained} (α_t / Σ_{t ∈ retained} α_t) × V_t
    -- softmax 재정규화: evicted 토큰의 attention mass가 retained에 재분배

D2O merge:
    O_after = Σ_{t ∈ retained} (α_t / Σ_{t ∈ retained} α_t) × V'_t
    -- V'_t = merge-compensated V (D2O의 cosine similarity 기반 merge 보상 적용)

KIVI quantization:
    O_after = Σ_t α_t × dequant(quant(V_t))
    -- quantize-dequantize round-trip이 V에 미치는 영향
```

**Per-KV-head 계산 후 집계**:

```
per kv_head h:
    QCF_h = ‖O_before_h - O_after_h‖₂ / max(‖O_before_h‖₂, ε)

aggregate_heads():
    QCF = mean(QCF_h for h in 0..n_kv_heads)
```

**GQA 처리**: Q-head 그룹의 attention weight를 평균하여 KV-head별 α를 생성한다.

```
α_kv[h][t] = mean(α_q[q][t] for q in gqa_group(h))
```

**불변식**:

- `QCF ∈ [0, 1]` — 정규화된 상대 오차
- Action이 V를 변경하지 않으면 `QCF = 0` (항등 변환)
- Eviction 토큰 수 증가 → QCF 단조 증가 (정보 손실 증가)
- KIVI quantization bit 감소에 대해 QCF 단조 증가: `QCF(Q2) > QCF(Q4) > QCF(Q8) > QCF(F16)`

**ENG-ALG-041~047, 049 대체 관계**:

| 대체되는 ID | 기존 proxy | Unified QCF에서의 대응 |
|------------|-----------|----------------------|
| ENG-ALG-041 | Attention x V-norm | Eviction O_after (H2O retained set) |
| ENG-ALG-042 | V-norm only | Eviction O_after (Sliding retained set) |
| ENG-ALG-043 | CAOTE | Eviction O_after (softmax 재분배 효과가 수식에 내재) |
| ENG-ALG-044 | Attention V2 (경량) | Eviction O_after (attention-only 근사는 별도 fast path로 가능) |
| ENG-ALG-045 | KIVI NMSE | KIVI O_after (dequant round-trip) |
| ENG-ALG-046 | KIVI OPR | KIVI O_after (attention-weighted 벡터 perturbation) |
| ENG-ALG-047 | AWQE | KIVI O_after (스칼라 합산 → 벡터 합산으로 통합) |
| ENG-ALG-049 | AW-VOPR | KIVI O_after (동일 수식; AW-VOPR이 ENG-ALG-051의 KIVI 경우와 수학적 동치) |

> **ENG-ALG-048 (Layer Skip QCF)**: speculative decoding rejection rate 기반이며, KV cache lossy action이 아니므로 이 통합 메트릭의 대상이 아니다. ENG-ALG-048은 그대로 유지된다.

---

### 3.5 RequestQcf -> QcfEstimate 흐름 [ENG-ALG-050]

**[ENG-ALG-050]** Manager가 RequestQcf를 보내면 Engine은 **읽기 전용 스캔**으로 per-action QCF 비용을 산출하여 QcfEstimate를 반환한다. *(MUST)*

> **구현 상태**: RequestQcf/QcfEstimate는 프로토콜(MSG-036b, SEQ-095~098)과 스펙에서 정의되며, `EngineCommand::RequestQcf` variant와 `CommandExecutor` 핸들러가 구현 완료되었다 (`shared/src/lib.rs`, `engine/src/resilience/executor.rs`).

**단계**:

1. **Manager -> Engine**: `RequestQcf { budget_ratio: f32 }` (Directive로 전달)
2. **Engine poll()**: CommandExecutor가 RequestQcf를 수신
3. **읽기 전용 스캔**: 캐시 상태를 변경하지 않고 각 액션의 QCF를 시뮬레이션

   **6종 Action dry-run 명세**:

   > 계산 컬럼: Eviction/D2O/KIVI의 QCF 계산은 Unified QCF (ENG-ALG-051)의 attention output perturbation 수식을 따른다. LayerSkip은 ENG-ALG-048 (별도 proxy)을 사용한다.

   | Action | 필요 입력 | 계산 | 출력 범위 | 가용성 조건 |
   |--------|----------|------|-----------|------------|
   | Sliding | `current_pos`, `target_len`, attention weights α, V cache | Unified QCF (ENG-ALG-051): Eviction O_after with Sliding retained set | [0, 1] | 항상 계산 가능 |
   | H2O | `current_pos`, importance scores, `keep_ratio`, attention weights α, V cache | Unified QCF (ENG-ALG-051): Eviction O_after with H2O retained set | [0, 1] | AttentionScoreAccumulator 활성 필요 |
   | Streaming | `current_pos`, `sink_size`, `window_size`, attention weights α, V cache | Unified QCF (ENG-ALG-051): Eviction O_after with Streaming retained set (`sink + window`) | [0, 1] | 항상 계산 가능 (`sink_size`, `window_size`는 config에서 취득) |
   | D2O | importance scores, `keep_ratio`, attention weights α, V cache | Unified QCF (ENG-ALG-051): D2O O_after with merge-compensated V' | [0, 1] | AttentionScoreAccumulator 활성 필요 |
   | KIVI | KiviCache residual 상태, attention weights α, V cache | Unified QCF (ENG-ALG-051): KIVI O_after with `dequant(quant(V))` | [0, 1] | KiviCache 사용 시에만 (F16/F32 KV cache에서는 N/A) |
   | LayerSkip | `ImportanceTable`, `skip_ratio` | `estimate_qcf_for_count()` (ENG-ALG-048, 테이블 참조만) | [0, 1] | prefill 시 ImportanceTable이 수집되어야 함 |

   **Action별 dry-run 상세**:

   - **Sliding**: 제거 대상 위치 계산 → Unified QCF (ENG-ALG-051) Eviction 수식 적용. retained set = `[evict_count..current_pos]`. `evict_count = current_pos - target_len`.
   - **H2O**: `identify_evicted_h2o()` → importance score 기반 하위 토큰 식별 → Unified QCF (ENG-ALG-051) Eviction 수식 적용. retained set = prefix + heavy hitters + recent window.
   - **Streaming**: `sink_size + window_size`로 retained set 결정 (sink + recent window) → Unified QCF (ENG-ALG-051) Eviction 수식 적용.
   - **D2O**: `identify_evicted_h2o()` + merge compensation으로 V' 생성 → Unified QCF (ENG-ALG-051) D2O 수식 적용. merge-compensated V'를 사용하므로 순수 eviction보다 낮은 QCF.
   - **KIVI**: 현재 residual 상태에서 Unified QCF (ENG-ALG-051) KIVI 수식 적용. `dequant(quant(V))` round-trip으로 O_after 계산.
   - **LayerSkip**: `ImportanceTable.estimate_qcf_for_count()` (ENG-ALG-048, 테이블 참조만). Skip 대상 레이어의 importance 합으로 QCF 추정. (Unified QCF 대상 아님)

   **가용성 판정**: 각 action의 가용성 조건이 충족되지 않으면 해당 action의 QcfMetric은 QcfEstimate에 포함되지 않는다 (N/A). Manager는 반환된 estimate 목록에 존재하는 action만으로 의사결정한다.

4. **QcfEstimate 생성**: per-action QcfMetric 리스트 + DegradationEstimator로 PPL 증가 추정
5. **Engine -> Manager**: `EngineMessage::QcfEstimate(...)` 응답

**불변식**: 스캔 중 캐시 데이터 변경 없음. KiviCache의 경우 `set_current_pos()`로 probe step의 update 되돌리기 가능. 모든 dry-run 출력은 [0, 1] 범위 (KIVI의 NMSE 포함).

---

### 3.6 DegradationEstimator [ENG-ALG-060]

**[ENG-ALG-060]** QCF proxy 값을 PPL 증가량(degradation)으로 변환하는 piecewise-linear 함수에 EMA 보정을 적용한다. *(MUST)*

**PiecewiseLinear**:

```
f(x) = slope_low * x                                                if x < breakpoint
      = slope_low * breakpoint + slope_high * (x - breakpoint)      if x >= breakpoint
```

**Estimation**:

```
function estimate(metric: QcfMetric) -> f32:
    curve = curves[metric.action]  // 없으면 linear(slope=1.0) fallback
    base = curve.evaluate(metric.raw_value)
    correction = ema_corrections[metric.action]  // 없으면 1.0
    return clamp(base * correction, 0, d_max)
```

**EMA 보정**:

```
function update_ema(action, proxy_value, actual_d):
    predicted = curve.evaluate(proxy_value)
    observed_ratio = actual_d / predicted
    ema = (1 - alpha) * current_ema + alpha * observed_ratio
```

**JSON 보정 파일** (예시):

```json
{
  "d_max": 5.0,
  "ema_alpha": 0.1,
  "actions": {
    "eviction": { "breakpoint": 0.3, "slope_low": 2.0, "slope_high": 8.0 },
    "kivi": { "breakpoint": 0.05, "slope_low": 10.0, "slope_high": 50.0 }
  }
}
```

---

### 3.7 madvise / release_unused_pages

#### 3.7.1 CPU Path [ENG-ALG-071]

**[ENG-ALG-071]** eviction 후 `current_pos` 이후의 미사용 KV 버퍼 영역에 `madvise(MADV_DONTNEED)`를 호출하여 물리 페이지를 OS에 반환한다. *(MUST)*

**high_water_pos 메커니즘**:

```
invariant: current_pos <= high_water_pos <= capacity
```

- `high_water_pos`: 지금까지 기록된 최대 `current_pos`. `update()` 시 갱신.
- madvise 범위: `[current_pos .. high_water_pos)` (high_water_pos 너머는 미접근 영역이므로 이미 물리 페이지 없음).
- madvise 후: `high_water_pos = current_pos` (다음 madvise 범위 초기화).

**Layout별 처리**:

```
SeqMajor:
    // 연속: [pos0_all_heads | pos1_all_heads | ...]
    row_bytes = kv_heads * head_dim * type_size
    used = current_pos * row_bytes
    hwm = high_water_pos * row_bytes
    madvise_dontneed(k_ptr, used, hwm)
    madvise_dontneed(v_ptr, used, hwm)

HeadMajor:
    // Per-head: [head0: pos0..posN | head1: pos0..posN | ...]
    for h in 0..kv_heads:
        base = h * capacity * head_dim * type_size
        from = base + current_pos * head_dim * type_size
        to = base + high_water_pos * head_dim * type_size
        madvise_dontneed(k_ptr, from, to)
        madvise_dontneed(v_ptr, from, to)
```

**madvise_dontneed 내부**:

```
function madvise_dontneed(base_ptr, from_offset, to_offset):
    // Page alignment: start UP, end DOWN
    aligned_start = round_up(base_ptr + from_offset, PAGE_SIZE)
    aligned_end = round_down(base_ptr + to_offset, PAGE_SIZE)
    if aligned_start >= aligned_end: return 0
    libc::madvise(aligned_start, aligned_end - aligned_start, MADV_DONTNEED)
```

**안전성**: `MADV_DONTNEED`는 anonymous private mapping에서 물리 페이지만 해제한다. 재접근 시 zero-fill page fault. KV 캐시는 사용 전 항상 덮어쓰기하므로 안전.

---

#### 3.7.2 shrink_to_fit [ENG-ALG-072]

**[ENG-ALG-072]** Dynamic KV cache에서 사용량이 capacity의 절반 미만일 때, 더 작은 버퍼로 재할당하여 물리 메모리를 확실히 해제한다. *(MUST)*

```
function shrink_to_fit():
    if memory == None: return 0   // non-dynamic cache는 불가
    new_cap = next_power_of_2(current_pos).max(64)
    if new_cap >= capacity: return 0

    new_k, new_v = allocate(new_cap, ...)
    copy current_pos worth of data from old to new
    freed = (old_buf_size - new_buf_size) * 2  // K + V

    k_buffer = new_k; v_buffer = new_v
    capacity = new_cap
    high_water_pos = current_pos
    return freed
```

**release_unused_pages 통합**: `release_unused_pages()`는 먼저 shrink_to_fit을 시도한다. 조건 미충족 시 madvise fallback.

```
function release_unused_pages():
    if memory.is_some() and current_pos < capacity/2:
        return shrink_to_fit()    // 재할당
    if not is_host_managed():
        return 0                  // GPU driver-pinned buffer는 skip
    // madvise path (ENG-ALG-071)
    ...
```

---

#### 3.7.3 GPU Path -- MadviseableGPUBuffer [ENG-ALG-073]

**[ENG-ALG-073]** GPU 버퍼에서 madvise를 활성화하는 대안 전략. *(MAY)*

**문제**: 표준 OpenCL 버퍼 (`CL_MEM_ALLOC_HOST_PTR`)는 드라이버가 물리 페이지를 pin하므로 madvise가 무효.

**해결: MadviseableGPUBuffer**:

- 앱이 `Vec<u8>`로 호스트 메모리 할당
- `CL_MEM_USE_HOST_PTR`로 CL 버퍼 생성 (앱 메모리를 GPU가 직접 접근)
- `is_host_managed() = true` → madvise 유효
- UMA (Unified Memory Architecture) 환경에서 zero-copy GPU 접근

**Adreno (Qualcomm GPU) 핀 문제**:

- 일부 Adreno 드라이버는 `CL_MEM_USE_HOST_PTR`에서도 물리 페이지를 핀 → madvise 무효
- **대안**: shrink_to_fit (ENG-ALG-072)으로 재할당하여 물리 메모리 확실 해제

**v17 설계 반영**: GPU 경로에서 shrink_to_fit이 madvise 대안으로 동작. `release_unused_pages()`가 자동으로 조건 분기.

---

### 3.8 Chunked Prefill [ENG-ALG-080]

**[ENG-ALG-080]** 긴 프롬프트를 청크 단위로 분할하여 peak 메모리를 제한한다. *(MAY)*

**CLI**: `--prefill-chunk-size N` (0 = 비활성, 전체 프롬프트를 단일 배치로 처리)

**알고리즘**:

```
function chunked_prefill(tokens, chunk_size, model, caches):
    if chunk_size == 0 or chunk_size >= len(tokens):
        // 단일 배치 처리 (기존 경로)
        model.forward(tokens, caches, logits_last_only=false)
        return

    // Chunked mode
    for chunk_start in range(0, len(tokens), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(tokens))
        chunk = tokens[chunk_start..chunk_end]

        // 모든 청크에서 logits_last_only=true (마지막 토큰만 logits 계산)
        model.forward(chunk, caches, logits_last_only=true)
```

**logits_last_only 최적화**: head projection에서 마지막 hidden state만 추출. logits 버퍼: `[1, 1, vocab_size]` (vs. `[1, seq_len, vocab_size]`). GPU 메모리 ~3GB 절감 (5K+ 토큰 시).

**불변식**: 청크 경계에서 KV 캐시는 연속으로 누적. `start_pos`가 청크별로 자동 증가.

---

### 3.9 CachePressurePipeline

#### 3.9.1 Pipeline 구조 [ENG-ALG-091]

**[ENG-ALG-091]** CachePressurePipeline은 여러 Handler를 pressure level에 따라 순차 실행하는 파이프라인이다. *(MUST)*

```
CachePressurePipeline {
    stages: Vec<PressureStageConfig> sorted by min_level ascending
}

PressureStageConfig {
    min_level: PressureLevel,          // 활성화 최소 레벨
    handler: Box<dyn CachePressureHandler>
}
```

**PressureLevel** (심각도 순서): `Normal < Warning < Critical < Emergency`

**실행 규칙**: `stage.min_level <= ctx.pressure_level`인 모든 stage가 순차 실행. 각 handler는 이전 handler가 변경한 캐시 상태를 본다.

---

#### 3.9.2 Handler 체인 (6종) [ENG-ALG-092]

**[ENG-ALG-092]** CachePressurePipeline은 6종 Handler를 지원한다. *(MUST)*

| Handler | 구현 상태 | 동작 | ActionResult |
|---------|----------|------|-------------|
| EvictionHandler | **완료** | EvictionPolicy 래핑, QCF proxy 수집 통합 | `Evicted { tokens_removed, new_pos }` |
| D2OHandler | **완료** | H2O eviction + cosine merge, EMA threshold | `Evicted { tokens_removed, new_pos }` |
| SwapHandler | **완료** | LRU offload via prune_prefix | `Swapped { tokens_swapped }` |
| QuantizeHandler | **완료** | pressure → target bits 매핑 (Warning=8, Critical=4, Emergency=2). KIVI 외부 처리 | `NoOp` (KIVI는 외부) |

**HandlerContext**: 각 handler에 전달되는 컨텍스트. 상세 필드 정의는 `33-engine-data.md` ENG-DAT-080 참조.

**Example pipeline 구성**:

```
[Warning]  EvictionHandler(ratio=0.8)
[Critical] EvictionHandler(ratio=0.5)
```

또는:

```
[Warning]  QuantizeHandler
[Critical] EvictionHandler(ratio=0.7)
```

---

#### 3.9.3 EvictionHandler QCF 통합 [ENG-ALG-093]

**[ENG-ALG-093]** EvictionHandler는 실제 eviction 전에 QCF proxy를 계산하여 qcf_sink에 push한다. *(MUST)*

```
function compute_and_push_proxy(ctx, current_pos, target_len):
    if qcf not enabled or qcf_sink is None: return
    if V buffer is null (GPU): return   // SIGSEGV 방지

    if policy is "sliding_window":
        evicted_positions = [0..prune_count)
        metric = compute_sliding_qcf_attn(evicted_positions, cache, ...)
    else if importance available (H2O):
        evicted = identify_evicted_h2o(importance, prefix=4, keep_ratio=0.5, ...)
        metric = compute_eviction_qcf_attn(evicted, importance, cache, ...)

    qcf_sink.push(metric)
```

---

### 3.10 Inference Loop Resilience Checkpoint [ENG-ALG-095]

**[ENG-ALG-095]** 추론 루프의 매 토큰 생성 후 `CommandExecutor.poll()`을 호출하여 ExecutionPlan을 소비한다. *(MUST)*

**소비 순서** (코드 기준):

```
function per_token_checkpoint(executor, caches, model_state):
    snap = build_kv_snapshot(caches)
    plan = executor.poll(&snap)

    // 1. Suspension check
    if plan.suspended:
        block until Resume signal
        return

    // 2. Eviction
    if plan.evict is Some(evict_plan):
        match evict_plan.method:
            H2o ->
                target_len = floor(current_pos * evict_plan.target_ratio)
                evict_with_scores(cache, target_len, importance)
            Sliding ->
                target_len = floor(current_pos * evict_plan.target_ratio)
                evict(cache, target_len)
            Streaming ->
                params = evict_plan.streaming_params.unwrap()
                streaming_evict(cache, params.sink_size, params.window_size)

    // 3. Device switch
    if plan.switch_device is Some(device):
        switch backend to device

    // 4. Layer skip
    if plan.layer_skip is Some(ratio):
        skip_config = SkipConfig::uniform_init(num_layers, ratio)

    // 5. KV quantization
    if plan.kv_quant_bits is Some(bits):
        for each kivi_cache: transition_bits(bits)

    // 6. Throttle
    if plan.throttle_delay_ms > 0:
        sleep(throttle_delay_ms)

    // 7. Restore defaults
    if plan.restore_defaults:
        reset all action state to initial values

    executor.on_token_generated()
```

**불변식**:

- Suspend는 다른 모든 plan 필드를 무효화 (evict, switch, prepare 모두 None으로 초기화)
- 같은 `poll()`에서 여러 Directive가 도착하면 마지막 evict가 승리 (superseding, ENG-ST-040 참조)
- Heartbeat는 `poll()` 내부에서 interval 기반 자동 전송

### 3.11 GPU Plan × Tensor Partition 협업

#### 3.11.1 LayerKernelPlan에 PartitionStep 통합 [ENG-ALG-200]

**[ENG-ALG-200]** Tensor partition이 활성일 때 GPU 단계는 plan으로 dispatch overhead를 회피하고, CPU FFN slice는 기존 PartitionWorkspace 위에서 동기 직렬 실행한다. *(MUST)*

**목적**: tensor partition이 활성일 때 GPU 단계는 plan으로 dispatch overhead를 회피하고, CPU FFN slice는 기존 PartitionWorkspace 위에서 동기 직렬 실행한다.

**알고리즘 (per layer)**:
1. (선행) 이전 layer의 GPU 단계 완료 보장 (in-order queue).
2. residual을 CPU에 가시화 (synchronize + read_buffer 또는 zcopy).
3. GPU FFN slice 4 KernelStep 연속 enqueue + flush.
4. CPU FFN slice (matmul gate, matmul up, silu/gelu_mul, matmul down).
5. PartitionMerge::Inline이면 copy_slice/add_assign 3 step 실행.
   PartitionMerge::Deferred이면 cpu_merge_staging에 upload만, merge는 다음 layer의
   fused_norm_merge에 위임.

**불변식**:
- INV-120 — PartitionStep::run 진입 시 ratio_generation 검사. mismatch 시 PlanInvalidated 반환.
- INV-082 (Buffer alive guarantee) — KernelStep.retained_bufs로 lifetime 유지.

**연관 ENG-ALG**:
- ENG-ALG-095 (resilience checkpoint) — SetPartitionRatio 처리 시 plan 재빌드 트리거.

**연관 arch**: `arch/plan_partition_integration.md` (전체).
**구현 우선순위**: P1 (decode TBT 회수 -10~-15 ms/tok 목표).

## 4. Alternative Behavior

**KIVI 비활성 (`--kivi` 미지정)**: KiviCache 대신 KVCache를 사용. flush cycle, bit transition, incremental dequant 모두 비적용. QCF의 NMSE/OPR/AWQE proxy도 비생성.

**CachePressurePipeline 비활성**: pipeline이 구성되지 않으면 모든 eviction은 inference loop의 resilience checkpoint (ENG-ALG-095)를 통해서만 발생한다.

**GPU mode 실패**: KiviCache GPU 할당 실패 시 CPU mode로 자동 fallback. MadviseableGPUBuffer 사용 불가 시 shrink_to_fit (ENG-ALG-072)이 대안.

**Manager 미연결 (`--enable-resilience` 없음)**: `command_executor = None`. Resilience checkpoint (ENG-ALG-095)를 건너뛰고 순수 추론만 수행. QCF 메트릭 생성은 계속되나 전송 대상이 없다.

## 5. Constraints

**[ENG-ALG-C01]** H2O eviction의 Protected Prefix와 Recent Window는 절대 제거되지 않는다. *(MUST NOT)*

**[ENG-ALG-C02]** KIVI flush 시 `group_size` (QKKV=32) 단위로만 양자화가 수행된다. `res_pos < group_size`이면 flush 불가. *(MUST)*

**[ENG-ALG-C03]** Layer Skip의 SWIFT 제약: Layer 0과 Layer L-1은 절대 skip하지 않는다. *(MUST NOT)*

**[ENG-ALG-C04]** RequestQcf 스캔 (ENG-ALG-050)은 읽기 전용이다. 캐시 데이터를 변경하지 않는다. *(MUST NOT)*

**[ENG-ALG-C05]** `madvise_dontneed()`의 범위는 반드시 page-aligned이다 (start UP, end DOWN). *(MUST)*

**[ENG-ALG-C06]** `is_host_managed() = false`인 버퍼에 madvise를 호출하지 않는다. *(MUST NOT)*

**[ENG-ALG-C07]** Chunked prefill에서 모든 청크는 `logits_last_only=true`로 처리한다. *(MUST)*

**[ENG-ALG-C08]** CachePressurePipeline의 stage 실행 순서는 `min_level` 오름차순이다. *(MUST)*

## 6. Examples

### 6.1 H2O Eviction Trace (2 heads, 8 positions, evict to 4)

```
초기 상태:
  current_pos=8, target_len=4, protected_prefix=1, keep_ratio=0.5
  available = 4 - 1 = 3
  hh_budget = floor(3 * 0.5) = 1
  recent_budget = 3 - 1 = 2
  recent_start = 8 - 2 = 6

  Evictable Zone: [1..6) = positions [1, 2, 3, 4, 5]
  importance (예시): [0.8, 0.1, 0.3, 0.05, 0.2, 0.15, -, -]

  ranked ascending: [3(0.05), 1(0.1), 5(0.15), 4(0.2), 2(0.3)]
  hh_budget=1이므로 상위 1개 보존: [2(0.3)]
  evicted: [3, 1, 5, 4]

  keep = {0} ∪ {2} ∪ {6, 7} = {0, 2, 6, 7}
  compact_keep_positions([0, 2, 6, 7], 0)
  current_pos = 4
  release_unused_pages()
```

### 6.2 QCF Eviction Trace (ENG-ALG-041, 2 heads, 8 positions, evict [0,1,2,3])

```
head 0: uniform attn 0.125, all V-norms equal
  total_imp = 8 * 0.125 * v_norm = v_norm
  evicted_imp = 4 * 0.125 * v_norm = 0.5 * v_norm
  raw[0] = 0.5, normalized[0] = 0.5/0.5 = 1.0

head 1: pos 0 has attn=0.5, rest 0.5/7
  evicted_imp = 0.5*vn + 3*(0.5/7)*vn = (0.5 + 0.214)*vn = 0.714*vn
  raw[1] = 0.714, normalized[1] = 0.714/0.286 = 2.497

Mean aggregation:
  raw_value = (0.5 + 0.714) / 2 = 0.607
  normalized_value = (1.0 + 2.497) / 2 = 1.749  // 1을 초과
```

### 6.3 KIVI Flush Cycle Trace (res_cap=64, group_size=32)

```
Token 1~64: kivi_update() -> res_pos 1..64
Token 64: res_pos(64) >= res_cap(64) -> flush_residual()
  n_groups = 64/32 = 2
  flush_tokens = 64
  -- QCF: compute_flush_qcf() -> NMSE metric pushed
  -- Key: per-channel quantize (2 groups * head_dim channels per head)
  -- Value: per-token quantize (64 tokens * head_dim/32 blocks per head)
  q2_tokens = 64
  res_pos = 0

Token 65: kivi_update() -> res_pos = 1
  ...
Token 128: flush again -> q2_tokens = 128

assemble_view():
  -- old_flushes=0, new_flushes=2 -> dequant 128 tokens
  -- copy residual (res_pos tokens) to attn_buf[128..]
  q2_deq_tokens = 128

Next decode step (no new flush):
  -- q2_tokens(128) == q2_deq_tokens(128) -> skip dequant
  -- residual copy only (O(res_pos))
```

### 6.4 release_unused_pages Decision Tree

```
release_unused_pages() 호출:
  |
  +--> memory.is_some() AND current_pos < capacity/2?
  |    YES -> shrink_to_fit() (재할당)
  |    NO  +--> is_host_managed()?
  |         YES -> madvise(MADV_DONTNEED, [current_pos..high_water_pos))
  |         NO  -> return 0  // GPU driver-pinned, skip
```

## 7. Rationale (non-normative)

### 왜 H2O 3-partition인가

Attention sink 현상 (시퀀스 초기 토큰의 높은 attention score)에 의해 초기 토큰 제거 시 perplexity 급증이 발생한다. Protected prefix가 이를 방지하고, recent window는 locality of reference를 보존한다. 3-partition은 StreamingLLM과 H2O의 장점을 결합한 설계이다.

### 왜 KIVI는 Key per-channel, Value per-token인가

KIVI 논문 (ICML 2024)의 분석에 따르면, Key 텐서는 채널 방향으로 outlier가 집중되어 per-channel 양자화가 정보 보존에 유리하다. Value 텐서는 토큰 방향으로 분산이 크므로 per-token 양자화가 효과적이다. 이 비대칭 전략이 동일 bit-width에서 더 낮은 perplexity를 달성한다.

### 왜 normalized_value가 unbounded인가

`normalized_value = evicted / remaining`은 eviction이 aggressive할수록 (남은 것보다 제거한 것이 많을 때) 1을 초과한다. 이는 cross-policy 비교에서 eviction severity를 선형적으로 반영하기 위함이다. `raw_value = evicted / total`이 [0,1] 범위로 clamp되어 절대 비용을 제공하고, `normalized_value`는 상대적 위험도를 제공한다.

### 왜 shrink_to_fit이 madvise의 대안인가

Qualcomm Adreno 등 일부 GPU 드라이버는 `CL_MEM_USE_HOST_PTR` 매핑에서도 물리 페이지를 driver-pin하여 madvise가 무효가 된다. shrink_to_fit은 버퍼 재할당으로 이전 메모리를 해제하므로 driver 동작과 무관하게 물리 메모리 반환이 보장된다. 다만 재할당 + 복사 비용이 발생하므로 capacity/2 미만일 때만 트리거한다.

### 왜 AWQE가 기본 비활성인가

AWQE는 가장 정확한 KIVI QCF proxy이나, attention score를 매 decode step에서 수집해야 한다. 이는 attention 커널에서 score를 별도 버퍼에 기록하는 추가 비용을 발생시킨다. 기본 NMSE + OPR 조합이 대부분의 상황에서 충분한 정확도를 제공하므로, AWQE는 세밀한 품질 추적이 필요한 경우에만 `set_awqe_enabled(true)`로 활성화한다.

### KvMergeD2o EngineCommand의 설계 전략

D2O의 merge 동작은 eviction과 동시에 실행되어야 하므로 CachePressurePipeline의 D2OHandler를 재활용한다. `KvMergeD2o { keep_ratio }` 명령 수신 시 executor.rs에서 `EvictPlan { method: D2o, target_ratio: keep_ratio }` 를 생성하고, generate.rs의 `EvictMethod::D2o` 분기에서 `CacheManager::force_evict_with_scores(target_ratio)`를 호출한다. D2OHandler.handle()이 ctx.target_ratio를 우선 사용하므로 Directive의 keep_ratio가 자연스럽게 override된다. ActionId::KvMergeD2o가 Manager action_registry에 등록되어 eviction 배타 그룹(C4/C5/C7/C8)에 포함된다.
