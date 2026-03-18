# Proxy Degradation Validation Experiment Plan

## 목표

Proxy metric이 실제 PPL 증가량과 얼마나 잘 상관하는지 검증.
핵심 지표: **Spearman rank correlation (ρ)** — proxy 순서가 PPL 순서를 보존하면 ρ ≈ 1.0.

## 실험 구조

### Phase 1: Baseline PPL (eviction 없음)

```bash
cargo run --release --bin generate -- \
  --model-path models/llama3.2-1b \
  --ppl experiments/proxy_validation/texts/wikitext_sample.txt \
  --backend cpu --kv-type f32 --temperature 0 \
  --eviction-policy none --kv-layout head \
  > results/proxy_validation/baseline.json
```

### Phase 2: Eviction PPL Sweep

각 eviction policy × budget 조합에서 PPL + proxy 수집.

#### 2a. Sliding Window

| kv_budget | 예상 eviction % |
|-----------|----------------|
| 1800 | ~10% |
| 1600 | ~20% |
| 1400 | ~30% |
| 1200 | ~40% |
| 1000 | ~50% |
| 800 | ~60% |
| 600 | ~70% |

```bash
for budget in 1800 1600 1400 1200 1000 800 600; do
  cargo run --release --bin generate -- \
    --model-path models/llama3.2-1b \
    --ppl experiments/proxy_validation/texts/wikitext_sample.txt \
    --backend cpu --kv-type f32 --temperature 0 \
    --eviction-policy sliding --eviction-window $budget \
    --kv-budget $budget --kv-layout head \
    > results/proxy_validation/sliding_${budget}.json
done
```

#### 2b. H2O

```bash
for budget in 1800 1600 1400 1200 1000 800 600; do
  cargo run --release --bin generate -- \
    --model-path models/llama3.2-1b \
    --ppl experiments/proxy_validation/texts/wikitext_sample.txt \
    --backend cpu --kv-type f32 --temperature 0 \
    --eviction-policy h2o --h2o-keep-ratio 0.5 \
    --protected-prefix 4 --kv-budget $budget --kv-layout head \
    > results/proxy_validation/h2o_${budget}.json
done
```

### Phase 3: KIVI Quantization PPL

```bash
for bits in 8 4 2; do
  cargo run --release --bin generate -- \
    --model-path models/llama3.2-1b \
    --ppl experiments/proxy_validation/texts/wikitext_sample.txt \
    --backend cpu --temperature 0 --kivi --kivi-residual-size 32 \
    > results/proxy_validation/kivi_q${bits}.json
done
```

(Note: KIVI bits 전환은 현재 CLI에서 지원하지 않으므로 default 2-bit만 가능.
추후 --kivi-bits 옵션 추가 필요.)

### Phase 4: Analysis

#### 4a. Rank Correlation 계산

```python
import json, glob
from scipy.stats import spearmanr

results = []
for f in sorted(glob.glob("results/proxy_validation/sliding_*.json")):
    data = json.load(open(f))
    avg_proxy = sum(m["raw_value"] for m in data["proxy_metrics"]) / max(len(data["proxy_metrics"]), 1)
    results.append({"budget": data["config"]["kv_budget"], "ppl": data["ppl"], "proxy": avg_proxy})

baseline = json.load(open("results/proxy_validation/baseline.json"))["ppl"]
proxies = [r["proxy"] for r in results]
ppl_increases = [r["ppl"] - baseline for r in results]

rho, p_value = spearmanr(proxies, ppl_increases)
print(f"Sliding Window: Spearman ρ = {rho:.4f}, p = {p_value:.4e}")
```

#### 4b. 판단 기준

| ρ 범위 | 판정 | 의미 |
|--------|------|------|
| ≥ 0.85 | 우수 | Proxy가 PPL 순서를 잘 보존 |
| 0.70 ~ 0.85 | 양호 | 대체로 동작, 일부 역전 |
| < 0.70 | 불충분 | Proxy 개선 필요 |

## 테스트 텍스트 준비

WikiText-2 validation set에서 ~2000 토큰 길이 샘플 3개 추출:

```bash
mkdir -p experiments/proxy_validation/texts
# WikiText-2에서 2000토큰 분량 추출 (수동 또는 스크립트)
```

다중 도메인 검증:
- `wikitext_sample.txt`: Wikipedia 백과사전 스타일
- `code_sample.txt`: 코드/기술 문서
- `narrative_sample.txt`: 서사 텍스트

## 예상 소요 시간

| 단계 | 실험 수 | 예상 시간/실험 | 합계 |
|------|---------|--------------|------|
| Baseline | 1 | ~30s | 30s |
| Sliding | 7 | ~60s | ~7min |
| H2O | 7 | ~90s | ~10min |
| KIVI | 1 | ~30s | 30s |
| Analysis | - | - | 1min |
| **합계** | **16** | | **~20min** |

## 성공 기준

1. Sliding window proxy: ρ ≥ 0.85 (위치 기반이므로 높을 것으로 예상)
2. H2O eviction proxy: ρ ≥ 0.70 (score × V-norm이 PPL 순서를 보존하는지)
3. KIVI NMSE proxy: Q2 > Q4 > Q8 순서가 PPL 순서와 일치
4. Proxy 값 범위: 모두 [0, 1] 내에 있는지 확인
