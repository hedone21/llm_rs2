# Layer-Swap QCF↔NLL 측정 가이드

AUF v0.2 multi-quant 기반 layer-swap의 **QCF_swap 예측치 vs 실제 NLL/품질 열화** 상관관계 측정용 인프라 가이드. 외부 harness(`/home/go/Workspace/papers/pact2026/experiments/scripts/`)가 LongBench/NIAH/RACE-h sweep을 짤 때 참고한다.

목차
- [1. 측정 모델](#1-측정-모델)
- [2. CLI 사용법](#2-cli-사용법)
- [3. JSON dump 스키마](#3-json-dump-스키마)
- [4. Sweep 매트릭스](#4-sweep-매트릭스)
- [5. 분석 권장 통계](#5-분석-권장-통계)
- [6. 의사 코드 (외부 harness)](#6-의사-코드-외부-harness)
- [7. 자산 준비](#7-자산-준비)
- [8. 함정/한계](#8-함정한계)

---

## 1. 측정 모델

| 변수 | 의미 |
|------|------|
| **r** | `--force-swap-ratio` (0.33, 0.66) — primary F16에서 secondary Q4_0로 swap할 layer 비율 |
| **swap_set** | `WeightSwapDecider.decide(r)`이 선택한 layer 인덱스 집합. `importance_i × ε_i` bottom-k. layer 0 / last 보호. |
| **QCF_swap** | `Σ_{i∈S} importance_i × ε_i / Σ_{j∈all_valid} importance_j × ε_j` — swap이 모델 품질에 끼친 예측 비용 (∈ [0, 1]) |
| **ΔPPL** | `PPL(swap) − PPL(baseline)` (또는 `ΔNLL = avg_nll(swap) − avg_nll(baseline)`) |
| **품질 metric** | LongBench: PPL/NLL. NIAH: retrieval accuracy. RACE-h: multiple-choice accuracy. |

**핵심 가설**: ρ(QCF_swap, ΔPPL) ≥ 0.85 (PPL); ρ(QCF_swap, 1−accuracy) ≥ 0.60 (QA류). KIVI/H2O QCF 검증과 동일한 임계값 사용.

---

## 2. CLI 사용법

### 2.1 새로 도입된 flag (HEAD `d460a77`)

```
--qcf-dump <PATH>           run 종료 시 단일 JSON 결과 dump.
                              활성화 시 자동 워밍업 prefill 흐름 진입.
--qcf-warmup-tokens <N>     importance 수집용 warmup prefill 길이. default 256.
                              너무 짧으면 importance 부정확, 너무 길면 시간 ↑.
```

### 2.2 자동 워크플로우 (`--qcf-dump` 활성 시)

```
1. Warmup prefill (skip 없음, secondary 미적용)
   → ImportanceCollector inject → ImportanceTable 빌드
2. KV cache 리셋
3. WeightSwapDecider.decide(ratio) [importance + noise 양쪽 주입 → 정확한 휴리스틱]
   → SwapDecision(selected_layers, qcf_swap_estimate, fallback_used=false)
4. SwapExecutor.execute(model, &selected_layers) → 실제 swap
5. run_ppl() 또는 generation 메인 루프 실행 (swap된 모델로 측정)
6. dump_qcf_swap_json(path, ctx) 호출
```

### 2.3 호출 예시

**Baseline** (`r=0.0`, primary F16만):
```bash
./target/release/generate \
  --model-path models/llama-3.2-1b/llama-3.2-1b-f16.gguf \
  --backend cuda --kv-type f16 --max-seq-len 4096 \
  --ppl experiments/prompts/prefill_4096.txt \
  --qcf-dump results/llama1b_r0.00_ppl.json
```
- secondary, force-swap-ratio 모두 미지정. dump JSON에 `swap_count=0, qcf_swap_predicted=0, fallback_used=false, force_swap_ratio=null`.

**Mid-ratio** (`r=0.33`, F16 + 33% Q4 swap):
```bash
./target/release/generate \
  --model-path models/llama-3.2-1b/llama-3.2-1b-f16.gguf \
  --secondary-gguf models/llama-3.2-1b-mixed.auf \
  --secondary-dtype q4_0 \
  --backend cuda --kv-type f16 --max-seq-len 4096 \
  --ppl experiments/prompts/prefill_4096.txt \
  --force-swap-ratio 0.33 \
  --qcf-dump results/llama1b_r0.33_ppl.json \
  --qcf-warmup-tokens 256
```

**ratio=1.0 (전부 Q4)** — `WeightSwapDecider`는 layer 0/last를 보호하므로 `--force-swap-ratio 1.0`만으로는 14/16. **진짜 100% Q4를 보려면 Q4_0 GGUF를 직접 `--model-path`로**:
```bash
./target/release/generate \
  --model-path models/llama-3.2-1b/llama-3.2-1b-q4_0.gguf \
  --backend cuda --kv-type f16 --max-seq-len 4096 \
  --ppl experiments/prompts/prefill_4096.txt \
  --qcf-dump results/llama1b_r1.00_ppl.json
```
- secondary 미지정. dump JSON에 `swap_count=0`이지만 외부 harness 후처리 시 `force_swap_ratio=1.0`으로 라벨링하고 `qcf_swap_predicted`는 `Σ all importance×ε / Σ all importance×ε = 1.0`으로 별도 채워 넣을 것 (현 dump는 swap_set=∅이므로 0.0이 기록됨 — 라벨링 단계에서 보정).

**Generation 모드 (LongBench/NIAH/RACE-h)**:
```bash
./target/release/generate \
  --model-path models/llama-3.2-1b/llama-3.2-1b-f16.gguf \
  --secondary-gguf models/llama-3.2-1b-mixed.auf \
  --secondary-dtype q4_0 \
  --backend cuda --kv-type f16 --max-seq-len 4096 \
  --prompt "@longbench_doc.txt" \
  --num-tokens 128 \
  --force-swap-ratio 0.33 \
  --qcf-dump results/llama1b_r0.33_longbench.json
```
- `ppl`/`avg_nll` 필드는 `null`로 직렬화. 외부 harness가 generated text와 expected answer를 비교해 F1/EM/ROUGE를 별도 JSON에 기록.

---

## 3. JSON dump 스키마

```json
{
  "schema_version": 1,
  "model_arch": "llama|qwen2",
  "model_path": "models/llama-3.2-1b-f16.gguf",
  "secondary_path": "models/llama-3.2-1b-mixed.auf",
  "primary_dtype": "F16",
  "secondary_dtype": "Q4_0",
  "num_layers": 16,
  "force_swap_ratio": 0.33,
  "swap_set": [1, 3, 5, 7, 9],
  "swap_count": 5,
  "qcf_swap_predicted": 0.214,
  "fallback_used": false,
  "importance_table": [
    {"layer": 0, "sublayer": "Full", "importance": 0.42, "opr": 0.31}
  ],
  "noise_table": [
    {"layer": 0, "epsilon": 0.018}
  ],
  "ppl": 12.34,
  "avg_nll": 2.51,
  "n_eval_tokens": 4096,
  "wall_time_s": 18.7,
  "warmup_tokens": 256,
  "backend": "cuda",
  "kv_type": "f16",
  "ppl_corpus": "experiments/prompts/prefill_4096.txt"
}
```

| 필드 | 타입 | 비고 |
|------|------|------|
| `schema_version` | int | 현재 1. 스키마 변경 시 증가 |
| `model_arch` | string | `llama` 또는 `qwen2` |
| `model_path`, `secondary_path` | string \| null | 절대/상대 모두 가능. secondary 없으면 null |
| `primary_dtype`, `secondary_dtype` | string | `F16`, `Q4_0`, `Q8_0`, `F32` 중 |
| `num_layers` | int | decoder layer 수 |
| `force_swap_ratio` | float \| null | CLI 인자 그대로. 미지정 시 null |
| `swap_set` | int[] | 실제 swap된 layer 인덱스 (오름차순) |
| `swap_count` | int | `swap_set.len()` (편의 필드, redundant) |
| `qcf_swap_predicted` | float | `decision.qcf_swap_estimate` ∈ [0, 1] |
| `fallback_used` | bool | importance 또는 noise 부재 시 uniform fallback 사용했는지 |
| `importance_table` | object[] | 각 (layer, sublayer)의 importance/opr. **항상 포함** (외부 harness 후처리용) |
| `noise_table` | object[] | NaN/missing 제외하고 valid layer만. 각 entry `{layer, epsilon}` |
| `ppl` | float \| null | teacher-forcing PPL. generation 모드에서는 null |
| `avg_nll` | float \| null | 토큰당 평균 NLL. generation 모드에서는 null |
| `n_eval_tokens` | int | PPL 계산에 쓰인 토큰 수 (generation 모드에서는 0) |
| `wall_time_s` | float | warmup + swap + 본 측정 + dump 합 |
| `warmup_tokens` | int | `--qcf-warmup-tokens` 그대로 |
| `backend`, `kv_type` | string | 측정 환경 메타 |
| `ppl_corpus` | string \| null | `--ppl <path>` 그대로 |

---

## 4. Sweep 매트릭스

| 차원 | 값 |
|------|-----|
| 모델 | Llama 3.2 1B, Llama 3.1 8B, Qwen2.5 1.5B, Qwen2.5 7B (4종) |
| Ratio | 0.0, 0.33, 0.66, 1.0 (4종) |
| Benchmark | LongBench, NIAH, RACE-h (3종) |
| **총 run 수** | **48** |

### 4.1 Ratio 처리 방식 (외부 harness 책임)

| Ratio | primary | secondary | force-swap-ratio | 비고 |
|-------|---------|-----------|------------------|------|
| 0.0 | F16 GGUF | (미지정) | (미지정) | 순수 F16 baseline |
| 0.33 | F16 GGUF | mixed.auf | 0.33 | 5/16 Q4 (1B 기준) |
| 0.66 | F16 GGUF | mixed.auf | 0.66 | 10/16 Q4 (1B 기준) |
| 1.0 | **Q4_0 GGUF** | (미지정) | (미지정) | 16/16 Q4. Q4_0 GGUF 직접 호출 |

### 4.2 모델별 layer 수 / 예상 swap_count

| 모델 | num_layers | r=0.33 → swap_count | r=0.66 → swap_count |
|------|-----------|---------------------|---------------------|
| Llama 3.2 1B | 16 | 5 | 10 |
| Llama 3.1 8B | 32 | 10 | 21 |
| Qwen2.5 1.5B | 28 | 9 | 18 |
| Qwen2.5 7B | 28 | 9 | 18 |

`floor(ratio × num_layers)` 그대로 (decider.rs:69).

---

## 5. 분석 권장 통계

### 5.1 모델 단위 + 통합

각 모델별로 (qcf_swap_predicted, ΔPPL) 페어 4개(ratio별 1) → 너무 적다. **벤치마크 단위로 다른 corpus chunk나 NIAH depth로 점을 늘려야** 의미 있는 ρ 계산 가능. 권장:
- LongBench: 5~10개 sub-task별 PPL 측정 → `(qcf, ΔPPL)` per sub-task per ratio = 모델당 ~30 점
- NIAH: depth × needle 조합으로 점을 곱 → 모델당 ~50 점
- RACE-h: subject별 accuracy 측정 → 모델당 ~5~10 점

### 5.2 권장 분석 흐름

```python
import json, glob, pandas as pd
from scipy.stats import spearmanr, pearsonr

# 1. 모든 dump JSON 적재
rows = []
for path in glob.glob("results/*.json"):
    j = json.load(open(path))
    rows.append({
        "model": j["model_arch"] + "_" + Path(j["model_path"]).stem,
        "ratio": j["force_swap_ratio"] or 0.0,
        "qcf": j["qcf_swap_predicted"],
        "ppl": j["ppl"],
        "nll": j["avg_nll"],
        "swap_count": j["swap_count"],
        "fallback": j["fallback_used"],
    })
df = pd.DataFrame(rows)

# 2. baseline 대비 ΔPPL
baseline = df[df.ratio == 0.0].set_index("model")["ppl"]
df["delta_ppl"] = df.apply(lambda r: r["ppl"] - baseline[r["model"]], axis=1)

# 3. 통계
for model, g in df.groupby("model"):
    g = g[g.ratio > 0]
    rho_s, p_s = spearmanr(g.qcf, g.delta_ppl)
    rho_p, p_p = pearsonr(g.qcf, g.delta_ppl)
    print(f"{model}: Spearman ρ={rho_s:.3f} (p={p_s:.3g}), Pearson r={rho_p:.3f}")

# 4. 산점도 + per-model 회귀선
# matplotlib + seaborn.lmplot
```

### 5.3 성공 기준 (논문 contribution 기준)

| 주장 | 검증 벤치마크 | 요구 ρ |
|------|---------------|--------|
| QCF_swap이 LM 열화를 예측 | LongBench (PPL) | ≥ 0.85 |
| QCF_swap이 정보 검색 실패를 예측 | NIAH | ≥ 0.70 |
| QCF_swap이 reading comprehension 열화를 예측 | RACE-h | ≥ 0.60 |

KIVI/H2O QCF 검증(`experiments/qcf_validation/PLAN.md`)과 동일 기준.

---

## 6. 의사 코드 (외부 harness)

```python
import subprocess, json
from pathlib import Path

MODELS = {
    "llama-3.2-1b": {
        "f16":  "models/llama-3.2-1b/llama-3.2-1b-f16.gguf",
        "q4_0": "models/llama-3.2-1b/llama-3.2-1b-q4_0.gguf",
        "auf":  "models/llama-3.2-1b-mixed.auf",
    },
    "llama-3.1-8b":  {...},
    "qwen2.5-1.5b":  {...},
    "qwen2.5-7b":    {...},
}

BENCHES = {
    "longbench": {"mode": "ppl", "corpus": "data/longbench/concatenated.txt"},
    "niah":      {"mode": "gen", "prompt_dir": "data/niah/"},
    "race-h":    {"mode": "gen", "prompt_dir": "data/race-h/"},
}

RATIOS = [0.0, 0.33, 0.66, 1.0]

OUT = Path("results/layer_swap_qcf"); OUT.mkdir(parents=True, exist_ok=True)
GENERATE = "/home/go/Workspace/llm_rs2-weight/target/release/generate"

for tag, m in MODELS.items():
    for r in RATIOS:
        for b_name, b in BENCHES.items():
            cmd = [GENERATE,
                "--backend", "cuda", "--kv-type", "f16", "--max-seq-len", "4096"]

            # ratio별 primary/secondary 분기
            if r == 0.0:
                cmd += ["--model-path", m["f16"]]
            elif r == 1.0:
                cmd += ["--model-path", m["q4_0"]]
            else:
                cmd += ["--model-path", m["f16"],
                        "--secondary-gguf", m["auf"],
                        "--secondary-dtype", "q4_0",
                        "--force-swap-ratio", str(r),
                        "--qcf-warmup-tokens", "256"]

            # 벤치마크별 모드 분기
            if b["mode"] == "ppl":
                cmd += ["--ppl", b["corpus"]]
            else:
                # NIAH/RACE-h: 외부 harness가 prompt 생성/평가
                cmd += ["--prompt", "@" + str(b_prompt_for_run(b, ...))]
                cmd += ["--num-tokens", "128"]

            dump = OUT / f"{tag}_r{r:.2f}_{b_name}.json"
            cmd += ["--qcf-dump", str(dump)]

            subprocess.run(cmd, check=True)

# 후처리: ratio=1.0 케이스의 qcf_swap_predicted를 1.0으로 라벨링
# (real 100% Q4는 swap_set=∅로 dump되므로 후처리에서 보정)
for path in OUT.glob("*_r1.00_*.json"):
    j = json.loads(path.read_text())
    j["qcf_swap_predicted"] = 1.0
    j["swap_count"] = j["num_layers"]
    j["force_swap_ratio"] = 1.0
    path.write_text(json.dumps(j, indent=2))
```

---

## 7. 자산 준비

### 7.1 GGUF 변환 (Q4_0 secondary용)

```bash
python scripts/convert_safetensors_to_gguf.py --outtype q4_0 \
    models/llama-3.1-8b/ models/llama-3.1-8b/llama-3.1-8b-q4_0.gguf

python scripts/convert_safetensors_to_gguf.py --outtype q4_0 \
    models/qwen2.5-1.5b/ models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf

python scripts/convert_safetensors_to_gguf.py --outtype q4_0 \
    models/qwen2.5-7b/ models/qwen2.5-7b/qwen2.5-7b-q4_0.gguf
```

Llama 3.2 1B Q4_0는 이미 repo에 있을 가능성 높음 (확인 후 변환 생략).

### 7.2 GGUF 변환 (F16 primary용)

```bash
python scripts/convert_safetensors_to_gguf.py --outtype f16 \
    models/llama-3.1-8b/ models/llama-3.1-8b/llama-3.1-8b-f16.gguf
# Qwen 2종도 동일
```

### 7.3 멀티-dtype AUF (F16 + Q4_0 동봉)

```bash
scripts/convert_to_auf.sh \
    --input models/llama-3.2-1b/ \
    --output models/llama-3.2-1b-mixed.auf \
    --dtypes f16,q4_0 \
    --default-dtype f16

scripts/convert_to_auf.sh \
    --input models/llama-3.1-8b/ \
    --output models/llama-3.1-8b-mixed.auf \
    --dtypes f16,q4_0 \
    --default-dtype f16

# Qwen 2종도 동일 패턴
```

검증:
```bash
./target/release/auf_tool verify models/llama-3.2-1b-mixed.auf
./target/release/auf_tool info models/llama-3.2-1b-mixed.auf
# format=v0.2, capability_opt=0xC, default_dtype=F16, dtypes=[F16, Q4_0] 확인
```

---

## 8. 함정/한계

### 8.1 Warmup token 수
`--qcf-warmup-tokens 256`이 default. 너무 짧으면 importance 부정확(ImportanceCollector는 sequence-mean pooling 사용 — line 168 `snapshot_before`), 너무 길면 시간 ↑. corpus가 매우 짧거나(<256 token) 의미가 균일하면 외부 harness가 줄여도 됨. NIAH처럼 토큰 분포가 critical한 경우 512 권장.

### 8.2 KV cache 일관성
Warmup → swap → 본 측정 사이에 KV cache가 깨끗해야 PPL/generation 정확. `run_ppl()`은 진입 시 새 KV alloc하므로 자연스럽게 해결되지만, generation 모드에서는 메인 흐름의 `kv_caches`가 워밍업으로 오염되지 않도록 시점에 주의 (현재 구현은 안전 — 메인 generate 진입 시 reset).

### 8.3 ratio=1.0 limitation
WeightSwapDecider는 layer 0/last를 무조건 보호. `--force-swap-ratio 1.0`만으로는 14/16(1B 기준), 30/32(8B 기준). **진짜 100% Q4는 Q4_0 GGUF 직접 사용**. 외부 harness가 ratio=1.0 라벨링 시 `qcf_swap_predicted=1.0`, `swap_count=num_layers`로 후처리 (위 §6 의사 코드 참고).

### 8.4 Qwen2 호환성
GGUF loader가 이미 Qwen2 분기 처리 (`engine/src/models/loader/gguf.rs:1230` — Qwen은 llama.cpp permute 미적용). AUF writer도 `general.architecture` 메타 보존. 추가 변경 불필요.

### 8.5 호스트 메모리
- 1B F16 + mixed.auf (F16+Q4) ≈ 4.5 GB
- 8B F16 + mixed.auf (F16+Q4) ≈ 25 GB
- Qwen2.5 7B F16 + mixed.auf ≈ 22 GB

호스트 RAM 32GB 미만이면 8B/7B에서 swap 발생 가능 — 외부 harness가 모델별로 GPU/CPU 분산 정책 결정.

### 8.6 단방향 swap 의무
v0.2 multi-quant 설계 전제. F16 → Q4_0만 가능. swap된 layer를 다시 F16로 되돌리는 시나리오는 본 측정 범위 외.

### 8.7 dump JSON과 stderr 출력
`--qcf-dump`가 켜져 있어도 기존 stderr 진단 출력(예: `[WeightSwap] OK: ratio=...`)은 그대로 유지. 외부 harness는 stderr 파싱 대신 JSON에 의존 권장.

---

## 9. 참고

| 문서/코드 | 위치 |
|-----------|------|
| WeightSwapDecider 구현 | `engine/src/models/weights/decider.rs:57` |
| QuantNoiseTable | `engine/src/models/weights/noise_table.rs` |
| ImportanceCollector | `engine/src/core/qcf/layer_importance.rs:150` |
| dump_qcf_swap_json | `engine/src/eval/qcf_helpers.rs` |
| AUF v0.2 spec | `arch/auf_format.md`, `docs/auf_format_changelog.md` |
| KIVI/H2O QCF 검증 (참고 protocol) | `experiments/qcf_validation/PLAN.md` |
| 본 측정 인프라 commit | `d460a77 feat(qcf): add --qcf-dump CLI + warmup→swap→PPL workflow ...` |
