# 8B host KV 기법 검증 트랙 — 1단계 baseline 측정 + 2단계 게이트 (2026-06-13)

목적: 1B 에서 KV 기법(로드맵 1/2/7)이 RED/검증불가였던 것이, host 에서 8B/long-context 로 가면
KV 가 실제 TBT 병목이 되어 기법 가치가 생기는지 측정. goal 트랙(측정·게이트까지 자율, 구조 구현은 중단).

## 1단계 — host baseline 측정 매트릭스

- 도구: `argus_bench` (release), prompt `/tmp/prompt_{2k,4k,8k}.txt`(Token Length 2161/4321/8701), `-n 32`, `--threads 8`, `--max-seq-len 9000`.
- 모델: `llama3.2-1b-f16.gguf`(1B, 16L) / `llama3.1-8b` q4_0(4.9GB)·f16(15GB) GGUF(8B, 32L).

### Decode TBT (ms/tok) — argus_bench `Decode:` 라인

| 모델 | backend | 2K | 4K | 8K | 2K→8K 배율 |
|---|---|---|---|---|---|
| 1B-f16 | **opencl** | 13.10 | 20.29 | 35.90 | **2.74×** |
| 1B-f16 | cpu | 46.22 | 50.37 | 57.81 | 1.25× |
| 8B-q4 | **opencl** | 49.10 | 81.39 | 144.36 | **2.94×** |
| 8B-q4 | cpu | 157.11 | 166.69 | 203.99 | 1.30× |
| 8B-f16 | **opencl** | 52.78 | 85.04 | 149.24 | **2.83×** |
| 8B-f16 | cpu | 241.04 | 260.38 | (>1500s, 생략) | — |

> 8B-f16 8K cpu 는 prefill 8701 tok × 8B f16 가 1500s 초과 — opencl-f16(149) + cpu-q4(204)로 패턴 충분해 생략.

### KV cache 메모리 (per-layer × n_layers, KV=f16)

| 모델 | 2K | 4K | 8K |
|---|---|---|---|
| 1B (16L) | 64MB | 128MB | 144MB |
| 8B (32L) | 256MB | 512MB | 576MB |

### 1B 대비 8B 배율 (동일 길이·백엔드)

opencl: 8B/1B ≈ 3.7×(2K 49/13) ~ 4.0×(8K 144/36). cpu: ≈ 3.4×(157/46) ~ 3.5×.

## 핵심 발견 — KV 병목 시작 지점

1. **백엔드가 결정적**: opencl(GPU)에서 KV attention 이 길이별 TBT 를 지배 — 전 모델 2K→8K **2.7~2.9배**. cpu 는 **1.25~1.30배**(weight matmul 이 지배, 8B cpu 157~260ms 라 KV 증가 묻힘).
2. **KV 병목 시작 = opencl + 4K 이상**: opencl 은 2K→4K 에서 이미 1.5~1.7배, 8K 지배적. 8B+8K opencl(KV 576MB, TBT 144~149ms)에서 KV 병목 최대.
3. **1B RED 의 원인 규명**: 항목 0 게이트는 1B/짧은 context(KV 64MB)라 KV 비병목 → 기법 무가치. 8B/8K opencl 에서 KV 가 실제 병목 → 1/2/7 의 **속도·메모리 절약 여지 확인**.

> 단 이는 **속도/메모리** 병목이고, 2단계 게이트는 **품질(PPL)**. KV 병목이어도 압축 기법이 sliding 보다 품질 나쁘면 속도 이득 무의미.

## 2단계 — demote 게이트 8B 재실행 (로드맵 1)

- 도구: `demote_measure` 테스트 `test_demote_vs_sliding_real_model_ppl`(DEMOTE_TEST_* env, 코드 무수정). sliding 64토큰 F32 vs demote 256토큰(창밖 192 Q4 왕복) PPL. 8B-q4, ~18min/도메인.

| 도메인 | sliding PPL | demote PPL | ratio | 판정 | 1B ratio(항목0) |
|---|---|---|---|---|---|
| literary (1B 최약 RED) | 97.54 | 286.11 | **2.93** | RED | 1.22 |
| encyclopedic (1B 최강 RED) | 16.98 | 723.19 | **42.6** | RED | 6.04 |

**판정: RED (8B 에서 1B 보다 극심화)** — **양 극단 모두 악화**: 1B 최약 RED literary 1.22→2.93, 1B 최강 RED encyclopedic 6.04→**42.6**. 1B 5도메인 전체 RED + 8B 양 극단 악화 = demote 의 "더 많은 저정밀 토큰 보존"이 8B 에서도 sliding 의 "정밀 64토큰"을 못 이기며, **모델을 키울수록 더 불리**(저정밀 K 의 2차 효과가 이후 모든 decode attention 에 누적 전파되는데, 8B 는 layer/head 가 많아 누적 폭이 큼). 항목 1 보류 **강화**(8B 가 1B RED 의 반전 희망을 오히려 부정).
- 나머지 3도메인(technical/conversational/news) 미측정 — 양 극단(최약·최강 RED)이 모두 8B 에서 악화라 중간 도메인은 RED 자명(보수적 추론). 도메인당 ~18min(8B-q4 cpu) 비용 대비 결론 불변이라 생략.

## 2단계 — 항목 2(WeightedKV)·7(cross-layer): 구조적 변경 필요 → 중단

- **항목 2 WeightedKV**: K/V 비대칭 merge 가중치(`apply_to: KeyOnly/ValueOnly/Both`, `w_k`/`w_v`) **구현 부재**(grep 0). 게이트 실험(WeightedKV ablation) 자체가 `apply_merges` K/V 분리 + `MergeAbi`/`FromPairAbi` 필드 추가 = plan IR/ABI 구조 변경 요구 → goal 제약상 **중단**.
- **항목 7 cross-layer**: "공유 basis 버퍼 + group-level plan 경로" = ADR급 설계 변경 → **중단**.

## 결론

- 1단계: 8B/8K opencl 에서 KV 가 TBT 병목임을 확인(1/2/7 기법의 메모리·속도 정당성 확보).
- 2단계: **demote(1) 는 8B 품질 게이트에서도 RED** → 보류 유지. **WeightedKV(2)·cross-layer(7) 은 게이트 실험 자체가 구조적 구현을 요구** → goal 중단 조건 도달, 사용자 결정 대기.
- 즉 "KV 가 병목이다(속도)"와 "어느 압축 기법이 sliding 을 이긴다(품질)"는 별개 — 8B 에서 속도 병목은 생겼으나 demote 는 여전히 품질 패배. 2/7 은 구현해야 게이트를 볼 수 있는 단계.
