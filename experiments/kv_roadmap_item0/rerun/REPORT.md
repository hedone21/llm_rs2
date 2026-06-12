# KV roadmap 항목 0 — P3 재측정 REPORT (2026-06-12 rerun)

> 1차 측정(2026-06-12)에서 e2e 배선 누락 4건 → 수정 라운드 7커밋(HEAD c702ff83) 후 재측정.
> 설계서: arch/kv_roadmap_item0_measurement.md (§5 게이트 임계 = SSOT, 변경 금지)
> 실행 명령 전문: rerun/raw/RUN_COMMANDS.md

## 환경
- HEAD c702ff83 / 빌드 `cargo build --release -p llm_rs2 --features rkv`
- 모델 llama3.2-1b-f16.gguf, KV f32, CPU backend, greedy(temp 0.0)
- 5도메인 프롬프트(304~327 tok), --protected-prefix 4, --kv-budget 200, --ppl-prefill-tokens 150

## ⚠ 프로토콜 편차 (지시 vs PPL 모드 구조 충돌)
지시 "prefill을 프롬프트 전체 길이로 고정"은 PPL 모드 구조와 충돌:
PPL 모드는 reference 텍스트(=프롬프트)를 prefill+decode로 분할 → prefill=전체이면 decode=0 →
eviction 미발동(전 도메인 n_evictions=0 실측 확인). eviction 측정 자체가 불가.
→ 차선: prefill 도메인-무관 고정값 150 통일. (1) 1차 budget(200) 가변 절단 제거, (2) 전 도메인 동일,
(3) decode 154~177 step 확보로 eviction 1건 정상 발동. "절단 오염 제거"의 실질(가변 절단 제거)은 달성.

---

## 측정 1: R-KV (cosine redundancy + importance)
### 1단 (RkvStats, redundant fraction)
- redundant fraction (τ=0.5 NN cosine): mean **0.9964** [0.9924, 1.0000], stdev 0.0009 — 전 layer·head·도메인 균일 포화
- MPC (mean pairwise K-cosine): mean 0.4919 [0.180, 0.900], stdev 0.164 — 분산 있음
- **sanity check**: fraction이 99.6%로 포화 = τ=0.5가 1B K벡터에서 변별력 없음. MPC mean 0.49면 NN cosine은 당연 >0.5.
  "redundancy 높다"기보다 K벡터가 high-cosine 영역에 본질적으로 몰림 → τ=0.5 부적절(임계는 설계서 고정값이라 그대로 적용).
- 게이트(fraction < 15% → 보류): 99.6% >> 15% → **형식상 2단 진입 조건 충족**

### 2단 (sliding/h2o/rkv 3-way PPL, EMR 하네스 미산출)
| domain | sliding | h2o | rkv | rkv/sld | rkv vs sliding |
|---|---|---|---|---|---|
| literary | 326.17 | 332.01 | 332.01 | 1.018 | sliding 우세 |
| encyclopedic | 219.40 | 221.35 | 222.47 | 1.014 | sliding 우세 |
| technical | 277.35 | 263.43 | 263.40 | 0.950 | rkv 우세 |
| conversational | 203.46 | 201.30 | 201.43 | 0.990 | rkv 우세 |
| news | 267.20 | 271.54 | 272.37 | 1.019 | sliding 우세 |
- rkv ≈ h2o (PPL 차이 <0.5%) → R-KV가 importance-only(H2O)로 퇴화 확정 (설계서 §3 예측대로)
- 평균 rkv/sliding PPL ratio = **0.9982** (sliding과 사실상 동률), 2/5 도메인만 근소 우세
- **게이트(§5): EMR Δ ≥ +3%p AND PPL ratio ≤ sliding → GO. EMR 하네스 미산출(편차), PPL ratio 0.9982 = +3%p 우세 아님 → 보류**

### 판정: **보류** (1단 fraction 포화로 2단 진입했으나, 2단에서 sliding과 동률 = redundancy-aware 무가치)
### 1차 신호 일치: ✅ fraction 0.9969 → 0.9964 (포화 확정), R-KV↔H2O 퇴화 확정

---

## 측정 2: A2SF (score forgetting factor)
| domain | sld_ppl | bos@d0 | bos@d8 | drop% | ppl@d8 | jaccard |
|---|---|---|---|---|---|---|
| literary | 326.17 | 28.16 | 14.64 | 48.0% | 330.79 | 0.810 |
| encyclopedic | 219.40 | 36.52 | 12.24 | 66.5% | 221.79 | 0.778 |
| technical | 277.35 | 33.70 | 10.92 | 67.6% | 267.62 | 0.786 |
| conversational | 203.46 | 33.26 | 8.95 | 73.1% | 201.66 | 0.786 |
| news | 267.20 | 31.44 | 12.45 | 60.4% | 269.82 | 0.802 |
- 평균 BOS ratio 감소(d0→d8) = **63.1%**, HH Jaccard = 0.792 (선택집합 79% 유지), d8가 sliding 이긴 도메인 = **2/5**
- **게이트(§5): BOS ratio ≥30% 감소 AND EMR Δ ≥ +2%p vs sliding → GO. BOS 63.1% 감소(충족) AND PPL proxy 2/5 win(미충족) → AND 미충족 = 보류**

### 판정: **보류** (BOS 완화는 확인되나 PPL 개선 없음 — 설계서 §2.2-D 예측대로 "완화돼도 sliding 못 이기면 무가치")
### 1차 신호: ❌ **반전 확정**. 1차 스모크 "decay 0.8 BOS ratio 증가(11.34→17.11)" = e2e 미배선 오측정. 본측정은 전 도메인 BOS ratio 감소(63.1%) — forgetting이 실제로 BOS 지배 완화함을 확인. 단 PPL 게이트는 여전히 보류.

---

## 측정 3: head importance 분산 (Ada-KV 전제)
| domain | max/min | max/2nd-min | C_h max | C_h min | sparse(>0.15) | dispersed(<0.05) |
|---|---|---|---|---|---|---|
| literary | 66.5 | 9.0 | 0.229 | 0.003 | 3 | 2 |
| encyclopedic | 121.9 | 11.2 | 0.215 | 0.002 | 2 | 3 |
| technical | 67.0 | 12.6 | 0.268 | 0.004 | 4 | 3 |
| conversational | 37.9 | 11.5 | 0.254 | 0.007 | 3 | 4 |
| news | 25.2 | 4.6 | 0.147 | 0.006 | 0 | 4 |
- 평균 max/min = **63.7배** (>> 게이트 5배). 2nd-min 기준(min head 0수렴 불안정성 배제)으로도 4.6~12.6배(평균 ~9.8배) ≥ 5배
- head 간 분산 명확: sparse head(C_h~0.25)와 dispersed head(C_h~0.004) 공존
- **게이트(§5): max/min < 2배 보류 / ≥5배 개봉 / 2~5배 약한신호 → 63.7배 = 개봉 후보**

### 판정: **개봉 후보** (per-head budget 차등 가치 있음 신호)
### 1차 신호: ✅ 해소. 1차 all C_h=0.0 degenerate → 본측정 실분포 산출(layer 0 슬롯)
### 측정 한계: last_step_head_attn()이 마지막 처리 layer만 반환 → JSON "layer 0" 라벨은 실제로는 최종 디코더 layer의 head attention. layer 1~15는 0(API 한계). **단일 layer 해상도** — 16층 전체 분산은 미측정(설계서 §2.3-B 의도된 단순화).

---

## 측정 4: Demote 모사 (항목 1 게이트)
### 실모델 PPL (sliding 64토큰 F32 vs demote 256토큰, 창밖 192 Q4 왕복)
| domain | sliding PPL | demote Q4 PPL | ratio | verdict |
|---|---|---|---|---|
| literary | 318.66 | 388.35 | 1.219 | RED |
| encyclopedic | 130.14 | 785.57 | 6.036 | RED |
| technical | 229.69 | 564.16 | 2.456 | RED |
| conversational | 138.16 | 243.90 | 1.765 | RED |
| news | 137.63 | 813.29 | 5.909 | RED |
- **5/5 도메인 전부 RED** (demote PPL ratio 1.22~6.04배, 모두 sliding보다 나쁨)
- **게이트(§5): demote PPL ratio < sliding (Q4) → GO, 아니면 RED → 전 도메인 RED**

### 판정: **RED** (항목 1 전체 보류 — 논문의 "4×@4bit > 1×@16bit"가 1B/2048/비reasoning에서 미재현)
### 1차 신호 일치: ✅ RED 방향 확정 (1차 스모크 demote 6.25 vs sliding 2.63 = RED)

### ⚠ NMSE vs PPL 갈림 관찰 (1차 NMSE 신호 vs 실모델 PPL 반전)
보조 NMSE 측정: demote K NMSE=0.1453 < sliding NMSE=0.75 → **NMSE상 demote 우세** (1차 신호와 동일).
그러나 실모델 forward PPL은 **demote RED**. 갈림 원인:
- NMSE는 quant→dequant 단일 왕복 오차(고립 토큰 1차 손실)만 측정.
- 실모델 PPL은 demote된 K/V가 **이후 모든 decode step의 attention에 누적 전파**(설계서 §2.4-D K 2차 효과).
- "동일 메모리 예산"에서 demote는 더 많은 토큰(256 vs 64) 보존하나 전부 저정밀 → 누적 오차가 sliding의 정밀 64토큰 대비 PPL을 악화.
- 결론: NMSE 단순 왕복은 demote 우세로 오도하나, 실 추론 PPL이 정본 게이트. 1B에서 Demote는 RED.

---

## 종합 (P4 입력)
| 측정 | 핵심 수치(5도메인 평균) | 게이트 판정 | 1차 신호 일치 |
|---|---|---|---|
| R-KV | fraction 0.9964 / PPL ratio 0.9982 (rkv≈h2o) | **보류** | ✅ |
| A2SF | BOS drop 63.1% / d8 2/5 win sliding | **보류** | ❌ 반전(BOS 증가→감소) |
| head 분산 | max/min 63.7배 (2nd-min 9.8배) | **개봉 후보** | ✅(degenerate 해소) |
| Demote | PPL ratio 1.22~6.04 全 RED | **RED** | ✅ |
