# QCF Taxonomy — `QCF_kv` and `QCF_weight`

본 문서는 llm.rs(crate `llm_rs2`)의 Quality Cost Function(QCF) 측정 체계를 두 패밀리(`QCF_kv` / `QCF_weight`)로 정리한 **참조 문서**다. 논문에 인용될 정의·수식·코드 위치를 동시에 제공한다.

- 작성일: 2026-04-27
- 결정 기록: `CLAUDE.md` "QCF 명명 컨벤션", `.agent/todos/backlog.md` `[P2] QCF 명명 컨벤션 정리`
- 코드 base: HEAD `b52b3ed` (branch `feat/weight`)

---

## 1. 개요

QCF는 **손실성(lossy) 액션이 모델 품질에 미치는 비용**을 [0, 1+] 범위의 무차원 값으로 정량화하는 함수족이다. 각 lossy action은 실행 시 부산물로 `QcfMetric`을 산출하고, 매니저(`llm_manager`)는 이 값을 정책 입력으로 사용한다.

본 코드 베이스의 QCF는 측정 공간이 본질적으로 다른 **두 패밀리**로 분리된다:

| 패밀리 | 측정 공간 | 결과 단위 | 적용 액션 |
|---|---|---|---|
| **QCF_kv** | head별 attention output 벡터 ℝ^{head_dim} | `‖ΔO‖₂ / ‖O‖₂` (per head, then aggregate) | KV 캐시에 가하는 lossy 액션 |
| **QCF_weight** | layer 단위 스칼라 | importance × noise의 가중 비율 | 모델 forward path를 변형하는 액션 |

두 패밀리는 분자/분모의 정의 공간이 달라 **raw 값으로 직접 비교할 수 없다**. cross-action 비교는 `DegradationEstimator`(§5)를 통해 ΔPPL(추정 perplexity 증가량)로 환산한 뒤 가능하다.

---

## 2. QCF_kv — KV 캐시 액션 패밀리

### 2.1 통합 정의

**대상 양**: layer 단일 forward 시점에 만들어지는 attention output

```math
O_h \;=\; \sum_{t=0}^{T-1} \alpha_h(t)\, V(h, t) \;\in\; \mathbb{R}^{d_{\text{head}}}
```

여기서

- `h`: KV head index, `h ∈ [0, n_kv_heads)`
- `t`: 시퀀스 위치, `T = current_pos`
- `α_h(t)`: head h의 attention weight at position t
- `V(h, t)`: KV cache의 V 벡터 (head h, position t)

**QCF_kv (per head)**:

```math
\mathrm{QCF}_{\mathrm{kv}}^{(h)}(\mathcal{A}) \;=\; \frac{\bigl\| O_h^{\text{before}} - O_h^{\text{after}}(\mathcal{A}) \bigr\|_2}{\bigl\| O_h^{\text{before}} \bigr\|_2}
```

여기서 𝒜는 적용된 액션. `O_h^{before}`는 액션 적용 전, `O_h^{after}(𝒜)`는 액션 적용 후의 attention output이다.

**Aggregation**:

```math
\mathrm{QCF}_{\mathrm{kv}}(\mathcal{A}) \;=\; \mathrm{Agg}\Bigl(\bigl\{\mathrm{QCF}_{\mathrm{kv}}^{(h)}(\mathcal{A})\bigr\}_{h=0}^{n_{kv}-1}\Bigr)
```

`Agg`는 **`Mean`** 또는 **`Defensive softmax`** (worst-head 강조; DefensiveKV, 2025 변형):

```math
\mathrm{Agg}_{\text{def}}(x) = \sum_h w_h x_h, \quad w_h = \frac{\exp(x_h / \tau)}{\sum_{h'} \exp(x_{h'} / \tau)}
```

코드 위치:
- 통합 함수: `engine/src/core/qcf/unified_qcf.rs::compute_unified_qcf` (line 80)
- aggregation: `engine/src/core/qcf/mod.rs::aggregate_heads` (line 108)

### 2.2 액션별 `O^after` 구성

5개 KV 액션이 동일 골격에 다른 `O^after`를 끼워 넣어 정의된다.

#### 2.2.1 Sliding Window Eviction

최근 `target_len` 토큰만 유지하고 나머지를 제거. 잔존 토큰 사이에서 attention weight를 재정규화:

```math
O_h^{\text{after}} = \sum_{t \in \mathcal{R}} \frac{\alpha_h(t)}{\sum_{t' \in \mathcal{R}} \alpha_h(t')} \, V(h, t)
```

여기서 `ℛ = [T - target_len, T)`.

코드: `unified_qcf.rs:153~172`, `compute_o_eviction()` line 360.

#### 2.2.2 H2O Eviction

Heavy-hitters + recent window + protected prefix의 3-파티션 잔존 정책:

```math
\mathcal{R} = \mathcal{P} \;\cup\; \mathrm{Top}_k\bigl(\{\alpha_h(t)\}_{t \in \mathcal{E}}\bigr) \;\cup\; \mathcal{W}
```

`𝒫 = [0, p)` (protected prefix), `𝒲 = [T - w, T)` (recent window), `ℰ = [p, T - w)` (evictable zone), `k = ⌊w_HH · |ℰ|⌋` (HH budget).

`O^after`는 §2.2.1과 같은 재정규화 공식.

코드: `unified_qcf.rs:173~199`, `identify_retained_h2o()` line 479.

#### 2.2.3 StreamingLLM Eviction

Sink + recent window:

```math
\mathcal{R} = [0, s) \;\cup\; [T - w, T)
```

(`s`: `sink_size`, `w`: `window_size`). `O^after`는 재정규화 공식.

코드: `unified_qcf.rs:200~222`.

#### 2.2.4 D2O Merge (Eviction with Compensation)

H2O와 동일한 잔존 집합 `ℛ`을 쓰되, evicted 토큰 `e ∈ ℰ ∖ ℛ`을 가장 가까운 retained 토큰에 가중 합성:

```math
\tilde V(h, r) = V(h, r) + \sum_{e \in \mathrm{NN}^{-1}(r)} w_{e \to r} \, V(h, e)
```

```math
w_{e \to r} = \frac{\exp(\sigma_{e,r})}{\exp(\sigma_{e,r}) + \mathrm{e}}, \quad \sigma_{e,r} = \cos\bigl(V(h, e), V(h, r)\bigr)
```

여기서 `NN(e) = argmax_{r ∈ ℛ} cos(V(h,e), V(h,r))`, `e = MERGE_E = 1.0`(default). 그 다음:

```math
O_h^{\text{after}} = \sum_{r \in \mathcal{R}} \frac{\alpha_h(r)}{\sum_{r' \in \mathcal{R}} \alpha_h(r')} \, \tilde V(h, r)
```

코드: `unified_qcf.rs:223~250`, `compute_o_d2o_merge()` line 393, `find_nearest_cosine_with_sim()` line 452.

#### 2.2.5 KIVI Quantization

토큰 집합은 변경 없음. V 벡터를 quantize→dequantize round-trip:

```math
O_h^{\text{after}} = \sum_{t=0}^{T-1} \alpha_h(t) \, Q^{-1}\bigl(Q(V(h, t); b)\bigr)
```

`Q(·; b)`: `b`-bit 양자화기. 지원 비트수 = {2, 4, 8}, block size `QKKV`. block-wise quantize: `BlockQ2_0`, `BlockKVQ4`, `BlockKVQ8`.

코드: `unified_qcf.rs:251~270`, `quantize_dequantize_f32()` line 524.

### 2.3 측정 시점과 데이터

- **측정 시점**: 매 lossy 액션 실행 시점(decode 도중 반복).
- **`α_h(t)`**: 해당 시점의 실제 attention scores. per-head slice가 사용 가능하면 사용, 그렇지 않으면 flat scores fallback (`unified_qcf.rs:99~129`).
- **`V(h, t)`**: KV cache의 V 버퍼. F32/F16/Q4_0 모두 지원 (`VDataSource` enum, line 46).
- **layout**: `KVLayout::HeadMajor` (production 고정). offset = `h · capacity · d_head + t · d_head` (`compute_v_offset` 헬퍼).

### 2.4 출력 자료구조

```rust
pub struct QcfMetric {
    pub action: String,                    // "eviction_attn", "kivi", ...
    pub raw_value: f32,                    // QCF_kv 본체. ‖ΔO‖/‖O‖ ∈ [0, 1+]
    pub normalized_value: f32,             // ‖ΔO‖ / ‖O − ΔO‖ (eviction에서 무경계)
    pub per_head: Option<Vec<f32>>,        // QCF_kv^(h) 분해
    pub tokens_affected: usize,
}
```

코드: `engine/src/core/qcf/mod.rs:30~44`.

---

## 3. QCF_weight — 모델 forward path 변형 패밀리

### 3.1 공통 입력: `ImportanceTable`

ImportanceTable은 prefill 1-pass 동안 layer마다 hidden state의 변화량을 cosine 유사도로 정량화한 것이다.

```math
\mathrm{importance}_i \;=\; \max\bigl(0,\; 1 - \cos\bigl(\bar x_i^{\text{in}},\, \bar x_i^{\text{out}}\bigr)\bigr)
```

여기서 `x̄_i^{in/out} = (1/T) Σ_{t=0}^{T-1} x_i(t)` (시퀀스 차원 mean-pool, `T = warmup_tokens`).

코드: `engine/src/core/qcf/layer_importance.rs:194~228`, `cosine_similarity()` line 269.

빌드 시점: `--qcf-warmup-tokens N`(default 256) 토큰으로 prefill 1회 수행, 매 layer 진입/탈출에서 hidden state mean-pool snapshot.

### 3.2 Weight Swap (F16 → Q4_0)

#### 3.2.1 Per-tensor Quantization Noise

per-tensor relative Frobenius squared error:

```math
\varepsilon_t \;=\; \frac{\bigl\| W_t^{\text{primary}} - W_t^{\text{secondary}} \bigr\|_F^2}{\bigl\| W_t^{\text{primary}} \bigr\|_F^2}
```

여기서 `t ∈ {Q, K, V, O, gate, up, down}` (decoder layer당 7개 tensor; norm tensor는 F32 미양자화로 제외). `‖·‖_F`는 Frobenius norm.

코드: `engine/src/models/weights/noise_table.rs::compute_tensor_epsilon` (line 218), 수식 line 307~318.

#### 3.2.2 Per-layer Quantization Noise

```math
\varepsilon_i \;=\; \frac{1}{|\mathcal{T}_i|} \sum_{t \in \mathcal{T}_i} \varepsilon_t
```

`𝒯_i`: layer i에서 dequant 성공한 tensor 집합. 모든 tensor 실패 시 `ε_i = NaN`(INV-127, swap 후보에서 제외).

코드: `noise_table.rs:109~134`.

#### 3.2.3 QCF_swap 본체

```math
\mathrm{QCF}_{\mathrm{swap}}(\mathcal{S}) \;=\; \frac{\sum_{i \in \mathcal{S}}\, \mathrm{importance}_i \cdot \varepsilon_i}{\sum_{j \in \mathcal{V}}\, \mathrm{importance}_j \cdot \varepsilon_j}
```

- `𝒮`: swap 대상 layer 집합 (F16 → Q4_0로 교체될 layer)
- `𝒱 = {j : ε_j 가 finite}`: 유효 layer 전체. NaN ε layer는 분자/분모 모두에서 제외.
- 결과 ∈ [0, 1] (clamp 적용).

코드: `engine/src/models/weights/decider.rs::compute_qcf_swap` (line 172), 내부 line 212~225.

#### 3.2.4 Swap Set 결정 (Decider)

`WeightSwapDecider`는 `importance_i × ε_i` bottom-k 휴리스틱으로 `𝒮`를 선택. Layer 0과 last layer는 보호되어 후보에서 제외 (sentinel + lm_head 결합 안전 확보).

```math
\mathcal{S} \;=\; \mathrm{argbot}_k\bigl\{ \mathrm{importance}_i \cdot \varepsilon_i \;:\; i \in \{1, \dots, N-2\},\; \varepsilon_i \in \mathbb{R} \bigr\}
```

`k = ⌊\text{ratio} × N⌋`, `N = num_decoder_layers`.

코드: `decider.rs::WeightSwapDecider::decide` (line 57).

### 3.3 Layer Skip (SWIFT)

#### 3.3.1 정의

skip 대상 sub-layer 집합 `𝒮_skip ⊂ {(i, sl) : i ∈ layers, sl ∈ {Full, Attention, Mlp}}` 에 대해:

```math
\mathrm{QCF}_{\mathrm{skip}}(\mathcal{S}_{\mathrm{skip}}) \;=\; \frac{\sum_{(i, sl) \in \mathcal{S}_{\mathrm{skip}}} \mathrm{importance}_{i, sl}}{\sum_{(j, sl') \in \mathcal{E}} \mathrm{importance}_{j, sl'}}
```

`ℰ`: ImportanceTable의 모든 entry (전체 importance의 정규화 분모).

분자/분모 모두 단순 importance의 sum (ε 가중 없음, swap과 차이).

코드: `engine/src/core/qcf/layer_importance.rs::ImportanceTable::compute_qcf` (line 68).

#### 3.3.2 OPR (Output-to-input Perturbation Ratio) — 보조 지표

skip 결정의 보조 지표로 함께 기록되는 layer-wise residual norm:

```math
\mathrm{OPR}_i \;=\; \frac{\bigl\|\bar x_i^{\text{out}} - \bar x_i^{\text{in}}\bigr\|_2}{\bigl\|\bar x_i^{\text{in}}\bigr\|_2}
```

skip QCF에는 직접 들어가지 않지만 ImportanceEntry에 동봉되어 후처리·진단에 사용.

코드: `layer_importance.rs::residual_norm_ratio` (line 246).

### 3.4 측정 시점과 데이터

- **`importance_i`**: warmup prefill 1회로 측정 후 고정 (corpus와 warmup_tokens에 의존).
- **`ε_i`**: secondary mmap 로드 시 1회 측정 후 고정 (모델 weight에만 의존).
- **`𝒮`**: 매니저 정책이 결정 (ratio/protected layer/currently_swapped 입력).
- **반복**: 정적 조합. 점진 swap 운영(`currently_swapped ≠ ∅`) 시 누적 입력에 대해 재계산.

---

## 4. 두 패밀리의 차이 (논문 인용용 요약표)

| 측면 | `QCF_kv` | `QCF_weight` |
|---|---|---|
| 측정 공간 | head별 attention output 벡터 ℝ^{d_head} | layer-level 스칼라 |
| 분자 | `‖ΔO‖₂` (벡터 차분의 L2 norm) | importance·noise의 단순 weighted sum |
| 분모 | `‖O_before‖₂` (원본 attention output L2) | 전 layer importance·noise sum |
| 결과 단위 | dimensionless 상대 L2 오차 ∈ [0, 1+] | dimensionless 가중 비율 ∈ [0, 1] |
| 시간적 성질 | 매 액션 실행마다 동적 계산 | 정적 (importance + ε는 1회 측정 후 고정) |
| Per-head 분해 | 있음 (n_kv_heads) | 없음 (layer 단위만) |
| 입력 의존성 | 현재 KV 상태(α, V)에 의존 | corpus(importance) + 모델 weight(ε)에 의존 |
| 후처리 시뮬레이션 | 불가 (런타임 attention state 의존) | 가능 (table dump → 다른 𝒮로 재계산) |
| 매니저 응답 형태 | scalar per action (`HashMap<String, f32>`) | layer-level table + ratio별 사전 샘플 |

---

## 5. Cross-Action 비교: `DegradationEstimator`

서로 다른 패밀리의 raw 값을 직접 비교할 수 없기에, 모든 액션의 QCF는 액션별 piecewise-linear 곡선을 거쳐 ΔPPL(estimated perplexity increase)로 환산된다.

```math
\Delta\mathrm{PPL}(\mathcal{A}) \;=\; \mathrm{Estimate}_{\mathcal{A}}\bigl(\mathrm{QCF}(\mathcal{A})\bigr)
```

여기서 `Estimate_𝒜(·)`는 액션 𝒜에 대해 오프라인 calibration으로 학습된 단조 piecewise-linear 함수. 곡선은 `(QCF, ΔPPL)` 점들의 ascending 보간. 매니저는 ΔPPL 단위로 cross-action 정책 결정을 수행.

코드:
- `engine/src/core/qcf/estimator.rs::DegradationEstimator::estimate` (line 136)
- 액션별 곡선 등록: `with_defaults()` (line ~71)

**현재 등록된 액션 키**(2026-04-27 기준): `eviction`, `sliding`, `kivi`, `swift`. **swap 미등록** — 후속 작업으로 backlog 등록됨.

매니저 IPC 메시지(`shared/src/lib.rs::QcfEstimate`)는 두 패밀리를 분리해 전송:

```rust
pub struct QcfEstimate {
    pub estimates: HashMap<String, f32>,        // QCF_kv 액션들
    pub layer_swap: Option<LayerSwapEstimate>,  // QCF_weight (swap)
}
```

---

## 6. (참고) 후속 검토: ε 재정의를 통한 측정 형태 정렬

본 분리는 **이름과 IPC 단위에서**의 분리이고, 두 패밀리가 동일 척도가 되는 통일은 의도하지 않는다. 다만 `QCF_weight`의 측정 *형태*만 `QCF_kv`와 닮게 만드는 안이 별도 backlog로 등록되어 있다 (Tier A, 적용 미정):

```math
\varepsilon_i^{\text{output}} \;=\; \frac{\bigl\| W_i^{\text{primary}} \cdot x_i - W_i^{\text{secondary}} \cdot x_i \bigr\|_2}{\bigl\| W_i^{\text{primary}} \cdot x_i \bigr\|_2}
```

여기서 `x_i`는 warmup prefill의 layer i 입력 hidden state. 이렇게 정의하면 ε_i가 `QCF_kv`의 `‖ΔO‖/‖O‖`와 같은 분자/분모 형태(출력 공간 상대 L2 오차)가 된다. `compute_qcf_swap`의 sum도 RMS 누적(`√Σ(imp·ε)²`)으로 변경하면 unified의 `‖ΔO‖` 합성과 동형이 된다.

본 변경은 cross-family raw 비교를 가능하게 하지만 의미론적으로는 여전히 다른 양을 측정하므로, 매니저 정책은 `DegradationEstimator` 경유 ΔPPL 환산을 권장한다.

---

## 7. 코드 위치 인덱스 (인용용)

| 항목 | 파일 | 함수/타입 | 위치 |
|---|---|---|---|
| QCF_kv 통합 함수 | `engine/src/core/qcf/unified_qcf.rs` | `compute_unified_qcf` | line 80 |
| KV 액션 enum | `engine/src/core/qcf/unified_qcf.rs` | `QcfActionType` | line 18 |
| Eviction `O^after` | `engine/src/core/qcf/unified_qcf.rs` | `compute_o_eviction` | line 360 |
| D2O merge `O^after` | `engine/src/core/qcf/unified_qcf.rs` | `compute_o_d2o_merge` | line 393 |
| KIVI quant round-trip | `engine/src/core/qcf/unified_qcf.rs` | `quantize_dequantize_f32` | line 524 |
| H2O retained 식별 | `engine/src/core/qcf/unified_qcf.rs` | `identify_retained_h2o` | line 479 |
| Head aggregation | `engine/src/core/qcf/mod.rs` | `aggregate_heads` | line 108 |
| QcfMetric 구조체 | `engine/src/core/qcf/mod.rs` | `QcfMetric` | line 30 |
| ImportanceTable | `engine/src/core/qcf/layer_importance.rs` | `ImportanceTable` | line 37 |
| Importance 측정 | `engine/src/core/qcf/layer_importance.rs` | `ImportanceCollector::record_after` | line 194 |
| Skip QCF | `engine/src/core/qcf/layer_importance.rs` | `ImportanceTable::compute_qcf` | line 68 |
| OPR | `engine/src/core/qcf/layer_importance.rs` | `residual_norm_ratio` | line 246 |
| QuantNoiseTable | `engine/src/models/weights/noise_table.rs` | `QuantNoiseTable` | line 30 |
| ε_t 계산 | `engine/src/models/weights/noise_table.rs` | `compute_tensor_epsilon` | line 218 |
| ε_i 집계 | `engine/src/models/weights/noise_table.rs` | `new_from_frobenius` | line 73 |
| QCF_swap 본체 | `engine/src/models/weights/decider.rs` | `compute_qcf_swap` | line 172 |
| Swap 결정 | `engine/src/models/weights/decider.rs` | `WeightSwapDecider::decide` | line 57 |
| ΔPPL estimator | `engine/src/core/qcf/estimator.rs` | `DegradationEstimator::estimate` | line 136 |
| IPC 메시지 | `shared/src/lib.rs` | `QcfEstimate`, `LayerSwapEstimate` | line 415, 401 |

---

## 8. 외부 인용 권장 형식

논문에서 본 두 패밀리를 인용할 때 다음 표현을 권장한다:

- "We define two QCF families, **QCF_kv** and **QCF_weight**, distinguished by their measurement spaces. QCF_kv measures relative L2 perturbation in head-level attention output, while QCF_weight measures importance-weighted relative quantization noise at layer granularity."
- 통일성에 대한 주장은 피하고, 패밀리별 정의를 분리해 인용.
- ΔPPL 환산을 거친 비교는 piecewise-linear DegradationEstimator의 calibration 곡선을 통해서만 수행되었음을 명시.

---

## 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-04-27 | 초판 작성. 두 패밀리 정의, 7개 KV 액션 + swap + skip 수식 정리, 코드 위치 인덱스. |
