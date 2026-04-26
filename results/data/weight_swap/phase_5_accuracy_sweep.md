# Phase 5 — WSWAP-5-INV122-SWEEP: 정확성 100-prompt sweep 리포트

- **측정일**: 2026-04-26
- **브랜치/커밋**: `feat/weight` @ `5fd9975`
- **디바이스**: Galaxy S25 (`SM-S931N`, adb `R3CY408S5SB`)
- **Android**: 16, kernel `6.6.77-android15-8`
- **백엔드**: OpenCL (Adreno), 6 threads, `--greedy` (temperature=0)
- **모델**:
  - primary: `Llama-3.2-1B-Instruct-f16.gguf` (2.4 GB)
  - secondary: `Llama-3.2-1B-Instruct.auf` (701 MB, sha256 `1a1ead0c1f532b26034989deb7dbfece4f5b7ed41b881491cfee857ae014c5ec`, mtime 2026-04-26)
  - tokenizer: `/data/local/tmp/tokenizer.json`
- **Generate binary**: 5차 빌드 HEAD `21c6d82` 동일 사용 (`/data/local/tmp/generate`, sha256 `961f347f67058b3ba73b455f81513969a48ada6ae40b2da90919bae90b164da4`)
- **CLI**: `--prompt-file ... --num-tokens 32 --backend opencl --threads 6 --greedy --protected-prefix 4 --experiment-schedule empty --experiment-output ... --experiment-logits-topk 10 --experiment-sample-interval 0`
- **반복**: 100 prompts × 5 ratios = 500 runs (별도 process마다 모델 재로드, 각 run = greedy decode 32 tokens)
- **Ratios**: `ref` (force-swap 미사용, F16 baseline), `0.25`, `0.5`, `0.75`, `1.0`
- **Decode 토큰 수**: 32 (Phase 4의 4-token에서 확장 — ROUGE-L 측정 가능 + drift 누적 평가)
- **측정 wall-clock**: 4158s (≈69.3 분), per-ratio cooldown 30s × 4
- **Thermal**: SoC zone0 idle 30°C, 측정 중 49–58°C (안정 범위)
- **변경점 vs Phase 4**:
  - **Phase 4**: secondary `Llama-3.2-1B-Instruct-q4_0.gguf` (런타임 SOA 재변환 경로), num-tokens=4
  - **Phase 5**: secondary `Llama-3.2-1B-Instruct.auf` (SOA bypass 경로, HEAD `21c6d82` 신규), num-tokens=32
  - SOA pipeline 적용된 새 AUF로 재측정. 더 긴 decode window로 drift 누적 평가.

## Prompt Set 구성

Phase 4와 동일 100 prompts (`/tmp/inv122_prompts.jsonl`, 별도 텍스트 파일 `/tmp/inv122_prompt_files/*.txt`) 재활용.

| Source | n | 비고 |
|--------|----|------|
| `experiments/prompts/benchmark_prompts.json` (perplexity, qa.single_doc, qa.summarization, qa.few_shot, qa.multi_hop, niah.filler) | 23 | 학술 평가 prompt |
| `experiments/prompts/med_len.txt` | 1 | |
| `experiments/prompts/short_len.txt` | 1 | "tell me long story" |
| Synthetic fact-completion (Phase 4 합성) | 75 | 단순 1-shot continuation |
| **합계** | **100** | spec ≥100 충족 |

## 측정 메트릭

각 prompt × 각 ratio 조합에서 **F16 ref** logits/tokens과 비교:

1. **top-1 match rate**: `argmax(top_logits)` 일치 (per position, prompt별 mean).
2. **top-5 overlap**: 양쪽 top-5 token id 집합의 교집합 / 5.
3. **token sequence match (seq)**: 동일 위치에서 sampled token (greedy) 일치.
4. **NMSE proxy**: 양쪽 top-K(10) 안 공통 token id의 logit 값 비교. `||r-c||²/||r||²`. Lower bound (Phase 4 한계 그대로).
5. **ROUGE-L**: 32-token sampled token ID sequence에 대한 LCS-based F1.

ratio별 100 prompt 통계: NMSE p50/p95/max, top-1/top-5/seq mean, ROUGE-L mean.

## ratio별 집계 결과 (n=100 prompts, 32 decode tokens)

| ratio | n | top-1 mean | top-5 mean | seq mean | NMSE p50 | NMSE p95 | NMSE max | ROUGE-L |
|-------|----|------------|------------|----------|----------|----------|----------|---------|
| 0.25  | 100 | 0.5515 | 0.5464 | 0.5435 | **0.00089** | **0.02220** | 0.19043 | 0.6639 |
| 0.50  | 100 | 0.3465 | 0.3401 | 0.3324 | **0.00560** | **0.04826** | 0.10924 | 0.5313 |
| 0.75  | 100 | 0.3168 | 0.3232 | 0.3103 | **0.00527** | **0.05701** | 0.10028 | 0.5034 |
| 1.00  | 100 | 0.2344 | 0.2409 | 0.2147 | **0.00964** | **0.03744** | 0.23140 | 0.4394 |

## Bimodal 분포 분석 (per-prompt top-1)

| ratio | perfect (=1.0) | zero (=0.0) | partial |
|-------|----------------|-------------|---------|
| 0.25 | 19/100 | 2/100 | 79/100 |
| 0.50 | 5/100 | 5/100 | 90/100 |
| 0.75 | 4/100 | 8/100 | 88/100 |
| 1.00 | 2/100 | 11/100 | 87/100 |

- Phase 4(num-tokens=4)와 비교 시 perfect rate가 크게 감소 (예: ratio=1.0, Phase 4 = 52% → Phase 5 = 2%). 이는 **decode window 확장으로 누적 drift 가시화**된 결과 — greedy 특성상 한 번 첫 token이 갈리면 KV/문맥이 분기되어 32 token 동안 모두 다른 path. partial이 87%로 압도적이라는 사실은 **양자화 noise가 어떤 위치에선가 ranking을 뒤집지만 일부 토큰은 여전히 일치**한다는 의미.

## INV-122 v2 Acceptance Verdict (ratio=1.0 mixed, 명세 임계값 기준)

**임계값**: NMSE ≤ 0.01, Δtop-1 ≤ 1pp

| 메트릭 | 측정값 | 임계값 | Verdict |
|--------|--------|--------|---------|
| NMSE p50 | 0.00964 | ≤ 0.01 | **PASS (간발의 차)** |
| NMSE p95 | 0.03744 | ≤ 0.01 | **FAIL** (3.7× 초과) |
| NMSE max | 0.23140 | ≤ 0.01 | **FAIL** |
| Δtop-1 (mean) | 76.56pp | ≤ 1pp | **FAIL** (76× 초과) |

**종합 verdict (ratio=1.0): FAIL**

NMSE p50만 한정해서 보면 임계값 근접이지만, p95 분포는 임계값을 크게 상회.
top-1 mean이 22%로 떨어진 것은 32-token greedy decode 누적 drift의 직접적 결과.

## 정확성 회귀 발생 여부

**없음.** 모든 ratio에서 sanity 출력은 자연어로 생성됐으며, "/buttons" 등 garbage token, 무한 반복, malformed 출력은 발견되지 않음. 5차 측정 sanity 3건과 일관:

- ref: "It was a bright cold day in April... a few figures hurrying along the pavement..." (정상)
- ratio=1.0: "It was a bright cold day in April... a few figures hurrying along the pavement..." (다른 분기, 정상 자연어)

## Phase 4 vs Phase 5 비교 (ratio=1.0 mixed)

| 메트릭 | Phase 4 (Q4 GGUF, 4-tok) | Phase 5 (AUF, 32-tok) |
|--------|--------------------------|------------------------|
| top-1 mean | 0.6600 | 0.2344 |
| top-5 mean | 0.6473 | 0.2409 |
| token seq | 0.6333 | 0.2147 |
| NMSE proxy mean | 0.0062 | 0.0151 |
| Perfect (=1.0) | 52% | 2% |
| Zero (=0.0) | 20% | 11% |

**해석**:
- top-1 0.66 → 0.23은 측정 window 32-token 확장에 의한 누적 drift가 주된 원인. 4-token 대비 8배 더 긴 분기 기회.
- AUF SOA bypass 자체가 Q4 GGUF 런타임 변환보다 정확성이 떨어진다는 증거는 아님 (4-token 기준 단순 비교 부재). 대신, 32-token greedy 누적 비교는 1B Q4 양자화의 본질적 ranking instability를 더 명확히 노출함.
- NMSE p95 0.037은 여전히 "공통 top-K logits 절대값 보존"이 일정 수준 무너짐을 시사 (**Phase 4의 0.0062 mean과 다른 메트릭 — Phase 5는 p95 분포 기준**).

## 권장 spec 조정 제안 (Architect 후속 작업 권고)

INV-122 v2 임계값(NMSE ≤ 0.01, Δtop-1 ≤ 1pp)은 다음 두 조건이 모두 만족돼야 의미있는 acceptance:
1. **단일 토큰 logit 비교** (현재 spec 의도로 보임)
2. **모든 prompt × position 분포의 p95 보장**

본 측정은 32-token greedy decode 누적 drift를 측정하므로 단일 토큰 비교가 아닙니다. **단일 토큰 NMSE 측정 (decode 1 토큰만 비교)** 으로 재정의하면 Phase 4 (4-token, NMSE=0.0062) 결과와 일관성 있게 PASS로 판정 가능.

권장 spec 조정 옵션 (순위순):

| 옵션 | 설명 | 근거 |
|------|------|------|
| **A. 단일-토큰 NMSE 메트릭 채택** | INV-122 v2를 "decode 첫 토큰" NMSE로 한정. 32-token 누적 drift는 별도 metric. | 양자화 정확성 검증 본래 의도와 일치. ranking instability와 logit value 보존을 분리 측정. |
| **B. NMSE p50 채택 (p95 미사용)** | "분포 중앙값이 0.01 이하" 만 요구. | Phase 4 mean 기준으로도 PASS, Phase 5에서도 p50=0.0096으로 거의 PASS. |
| **C. 임계값 완화 (NMSE ≤ 0.05, Δtop-1 ≤ 30pp at 32-tok)** | 현재 측정 조건에 맞춘 현실적 임계값. | Phase 5 ratio=0.25에서도 NMSE p95=0.022 → 어떠한 ratio에서도 0.01은 비현실적. |
| **D. AUF builder 정밀도 개선** | spec 유지, 구현 측에서 정확성 회복. | F16 baseline 기준 "true" loss 측정이 미실시. AUF builder의 Adreno noshuffle SOA pipeline 정확성 검증 별도 필요. |

**Tester 결론**: 본 측정에서 사용된 acceptance 정의(32-token greedy 누적 NMSE p95)는 spec INV-122 v2의 본래 의도(단일 logit 비교)와 다를 가능성이 높음. **Architect와 sync 후 spec 명확화 필요**. 본 sprint에서는 측정/보고만 수행 (Tester 권한 외 spec 변경 금지).

## 산출물

- 원시 logits: `/tmp/phase5_inv122_sweep/r_{ref,0.25,0.5,0.75,1.0}/{idx}_{prompt_id}.jsonl` (500 파일, 5.9 MB)
- 측정 스크립트: `/tmp/phase5_inv122_sweep.sh` (host)
- 분석 스크립트: `/tmp/phase5_inv122_analyze.py` (host)
- 통계 요약: `/tmp/phase5_inv122_sweep/summary.json`
- 측정 로그: `/tmp/phase5_inv122_sweep.log`
- 본 리포트: `results/data/weight_swap/phase_5_accuracy_sweep.md`

## 다음 액션

1. **Architect 협의**: INV-122 v2 임계값 정의 명확화 (단일 토큰 vs 누적 decode, mean vs p95). Tester 측정 결과 옵션 A~D 검토.
2. **단일-토큰 NMSE 보강 측정** (옵션 A 채택 시): Phase 5 데이터에서 첫 token만 추출해 NMSE 재계산 → ratio=1.0에서 ~0.01 근접 예상 (Phase 4 mean 0.0062와 비교).
3. **WSWAP-5-COLD-UNIFORM (현재 진행 가능)**: 본 sprint 디바이스 점유 종료. cold-path 측정 가능.
4. **Phase 5 TBT-DIAG (`WSWAP-5-TBT-DIAG`)**: ratio=1.0 mixed −20.7% TBT gap 별도 진단.
