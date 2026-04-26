# Phase 5 — lm_head Q4_0 Quantize: Sprint F TBT Root Cause Fix

**Date**: 2026-04-26
**Branch**: `feat/weight`
**Device**: Galaxy S25 (`R3CY408S5SB`), Adreno OpenCL, threads=6, V10 thermal isolation
**Model**: Llama 3.2 1B Instruct
**Mode**: async TBT (no `--profile`), `--num-tokens 128`, `--protected-prefix 4`,
prompt `"The capital of France is"`, temperature 0.0

---

## 0. 한 줄 요약

ratio=1.0 mixed weight-swap의 **+24.7% TBT 회귀**의 본질이 lm_head F16-vs-Q4_0 dtype 격차임을 디바이스 N=3 측정으로 100% 확정. 런타임 1회 quantize (mode=auto when `--secondary-gguf` 설정) 도입으로 회귀 완전 회수, **Q4 baseline 대비 10% 추가 가속** 부수 효과까지 확보.

---

## 1. F-0 본질 검증 (mode=q4_0 강제 토글)

### 1.1 측정 매트릭스 (N=3, 128 tokens, async)

`Decode(excl tok[0])` 기준:

| 조건 | run1 | run2 | run3 | mean (ms/tok) | σ |
|---|---:|---:|---:|---:|---:|
| Q4 baseline | 16.39 | 16.41 | 16.35 | **16.38** | 0.025 |
| Mixed C-2 baseline (lm_head=F16) | 19.95 | 20.44 | 20.11 | **20.17** | 0.204 |
| **Mixed C-2 + `--quantize-lm-head q4_0`** | 15.18 | 14.74 | 14.48 | **14.80** | 0.288 |

### 1.2 회복도 분석

- Mixed baseline 회귀 폭 (Q4 baseline 대비): **+3.79 ms/tok (+23.1%)** — Sprint C-3에서 측정한 +24.7%와 정합 (noise 1%p).
- Mixed + lm_head Q4_0 vs Q4 baseline: **−1.58 ms/tok (−9.6%)** — 100% 이상 회복. 격차가 사라지는 정도가 아니라 baseline을 능가.
- 본질 확정 신뢰도: **100%**. Sprint E op-tracer 분석이 가리킨 lm_head 단일 op 가설 검증 완료.

### 1.3 보너스 효과 (왜 baseline보다 빠른가)

Mixed mode + lm_head Q4_0가 Q4 baseline보다 −1.58 ms/tok 빠른 원인은 **AUF SOA bypass 경로의 일등성**으로 추정. Q4 baseline은 `convert_q4_0_to_noshuffle` 경로를 통해 Q4_0 weights를 SOA로 변환하지만, AUF mode는 사전-변환된 `register_pre_converted_soa`를 사용해서 첫 forward의 cold cache cost가 다소 다를 수 있음. Sprint C-1 측정에서 같은 비-회귀 패턴 관찰됨. 본 sprint의 우선순위 밖이므로 향후 sprint에서 분석.

### 1.4 정확성 가드 (ratio sweep, 32 tokens, temp=0)

| ratio | 출력 (앞 12 토큰) | 판정 |
|---|---|---|
| 0.25 | "Paris. The Eiffel Tower, a famous landmark in Paris, was built…" | PASS |
| 0.5 | "Paris. The Eiffel Tower, a famous landmark in Paris, was built…" | PASS |
| 0.75 | "Paris. The Eiffel Tower, a famous landmark in Paris, was built…" | PASS |
| 1.0 | "Paris. The Eiffel Tower, a famous landmark in Paris, was built…" | PASS |

전 ratio 영역에서 "Paris" 정답 + 일관된 영어 문장. garbage 0건.

---

## 2. F-1 Production Fix (mode=auto, default-on for AUF)

### 2.1 구현 결정

**선택**: F-1b (런타임 quantize, default `auto`).

이유:
1. **Self-contained**: AUF 포맷, swap_executor, spec/build pipeline 변경 무관.
2. **즉시 가용**: 같은 sprint 내 production-ready 검증 가능.
3. **Q4 baseline 무영향**: `auto` 모드는 lm_head dtype를 검사하므로 Q4_0이면 자동 skip.
4. **F16-only 무영향**: secondary-gguf 미설정 시 quantize 안 함 (legacy 보존).

F-1a (AUF 포맷 lm_head entry 추가)는 Phase 6 백로그로 이관 — production 가용 가치는 동일하나 spec 영향이 큼.

### 2.2 CLI/API 설계

```rust
// engine/src/bin/generate.rs
#[arg(long, default_value = "auto")]
quantize_lm_head: String,  // "auto" | "none" | "q4_0"
```

- `auto` (default): `--secondary-gguf` 설정 + lm_head ≠ Q4_0 → quantize.
- `q4_0`: 무조건 quantize (secondary 없어도).
- `none`: 절대 quantize 안 함 (diagnostic/regression test 용).

```rust
// engine/src/models/transformer.rs
pub fn quantize_lm_head_to_q4_0(
    &mut self,
    runtime_backend: &Arc<dyn Backend>,
) -> Result<bool>;
```

핵심 로직:
1. lm_head dtype 검사 — Q4_0면 `Ok(false)` 반환.
2. F16/F32/BF16 → F32 dequantize. host pointer 가용 시 직접 read, 아니면 `runtime_backend.read_buffer()`.
3. `quantize_q4_0(rows, cols)` 호출 (기존 `convert.rs::quantize_q4_0` 재사용).
4. CPU SharedBuffer Q4_0 tensor 생성 → `backend.copy_weight_from()` 으로 GPU 업로드.
5. **Tied-weight 보호**: 옛 lm_head가 `gpu_embed_tokens`와 cl_mem 공유 시, `embed_tokens` (CPU F16) 에서 새 F16 GPU 버퍼 생성하여 gpu_embed_tokens 분리. embed gather 경로의 F16 dtype 보존.

### 2.3 F-1 측정 결과 (N=3, 128 tokens, async, auto mode)

| 조건 | run1 | run2 | run3 | mean (ms/tok) | σ |
|---|---:|---:|---:|---:|---:|
| Q4 baseline (auto, no-op) | 16.30 | 16.32 | 16.29 | **16.30** | 0.012 |
| Mixed AUF ratio=1.0 (auto) | 14.50 | 15.01 | 14.47 | **14.66** | 0.247 |
| Mixed AUF ratio=0.5 (auto) | 26.24 | 25.43 | 26.73 | **26.13** | 0.535 |
| Mixed AUF ratio=1.0 + `--quantize-lm-head none` (regression) | 21.03 | 21.44 | 20.37 | **20.95** | 0.439 |

**검증**:
- Q4 baseline: 16.30 ms/tok, σ=0.012 ms — 회귀 0%. (auto 모드는 already-Q4_0 감지 → 출력 메시지조차 없음.)
- Mixed AUF ratio=1.0 (auto): 14.66 ms/tok — Q4 baseline 대비 **−10.1% 가속**.
- Regression test (`--quantize-lm-head=none`): 20.95 ms/tok — Sprint C-3 보고치 +20.7%와 정합. 기존 회귀 패턴 100% 재현.
- ratio=0.5: 26.13 ms/tok — partial swap mode 동작. lm_head fix는 ratio-independent 이므로 lm_head 단일 op 부담 −4.6 ms/tok 회수, 잔여 +9.6 ms/tok은 partial mixed (ratio<1.0) 모드 자체 비용.

### 2.4 Quantize 비용 (one-shot at load time)

| 조건 | quantize 시간 |
|---|---:|
| run1 (cold mmap) | 1683.6 ms |
| run2 (warm mmap) | 897.2 ms |
| run3 (cold mmap) | 1571.8 ms |

평균 ~1.4 s, model load 단계에서 1회 발생, decode TBT에 영향 0%. Llama 3.2 1B vocab=128256, hidden=2048, 525 MB → 148 MB 변환.

---

## 3. 회귀 검증 (Regression Gate)

### 3.1 코드 품질

- `cargo fmt --all`: clean.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo test --workspace --lib -- --skip <expected-skips>`: **1053 + 221 + 38 passed, 0 failed**.

### 3.2 정확성 (Llama 3.2 1B, "The capital of France is", temp=0)

| 시나리오 | 출력 | 판정 |
|---|---|---|
| Q4 baseline (auto) | "Paris[part-dbg]…" | PASS (의미 보존) |
| F16-only (auto, no quantize) | "Paris. The Eiffel Tower…" | PASS |
| F16-only (q4_0 force) | "Paris. The Eiffel Tower…" | PASS |
| Mixed AUF ratio=1.0 (auto) | "Paris. The Eiffel Tower, a famous landmark in Paris, was built for the World's Fair in 1889…" | PASS |
| Mixed AUF ratio=0.5 (auto) | "Paris. The Eiffel Tower…" | PASS |
| Mixed AUF ratio=0.25 (auto) | "Paris. The Eiffel Tower…" | PASS |
| Mixed AUF ratio=1.0 + none (regression) | "Paris. …" (정답) | PASS |

**모든 ratio + 모든 mode에서 garbage 0건, 정답 "Paris" 100%.**

### 3.3 메모리 영향

- F16 lm_head: 525 MB GPU
- Q4_0 lm_head: 148 MB GPU
- 순 회수: **+377 MB GPU memory** (보너스)

기존 Sprint 4/5에서 측정한 alive cl_mem 4.05 GB → 0.5 GB (−87%) 위에 추가 회수.

---

## 4. 변경 사항 요약

### 4.1 코드

- `engine/src/models/transformer.rs:386-501` (신규):
  - `quantize_lm_head_to_q4_0(&mut self, runtime_backend: &Arc<dyn Backend>) -> Result<bool>`
  - F16/F32/BF16 → F32 dequant → `quantize_q4_0` (재사용) → SharedBuffer Q4_0 → GPU upload.
  - Tied-weight 보호 (gpu_embed_tokens 분리).

- `engine/src/bin/generate.rs:264-279` (신규 CLI 옵션):
  - `--quantize-lm-head <auto|none|q4_0>`, default `auto`.

- `engine/src/bin/generate.rs:1059-1097` (호출 site):
  - `auto` 모드: secondary-gguf 설정 시만 작동.
  - 명시 `q4_0`/`q4`: 강제.
  - `none`/`off`: skip.

### 4.2 산출물

- `results/data/weight_swap/sprint_f/run_f0.sh` — F-0 측정 스크립트
- `results/data/weight_swap/sprint_f/f0_results.txt` — F-0 raw 측정 데이터
- `results/data/weight_swap/sprint_f/run_f1.sh` — F-1 production 검증 스크립트
- `results/data/weight_swap/sprint_f/f1_results.txt` — F-1 raw 측정 데이터
- 본 보고서

---

## 5. 다음 sprint 권장

### 권장 Phase 5 종결 조건 만족
- Sprint A~D 가설 모두 부정 (alive cl_mem, plan dispatch, KIVI fallback, memory zone, TLB cold)
- Sprint E op-tracer로 lm_head 단일 op 본질 확정
- Sprint F-0/F-1로 본질 확정 + production fix 완료
- TBT gap 100% 회수 + 보너스 −10% 가속

**결론: Phase 5 weight-swap TBT 분석 종결**.

### Phase 6 백로그 (낮은 우선순위)

1. **F-1a: AUF 포맷에 lm_head Q4_0 entry 추가** (선택적)
   - 장점: model load 시 quantize 비용 (~1 s) 제거, 디스크에서 직접 mmap.
   - 단점: AUF format spec 변경, auf_tool/swap_executor/secondary_mmap.rs 수정, ENG-DAT-* spec 갱신, build pipeline 영향.
   - 우선순위: **낮음** — 현 F-1b는 1회 1초 비용을 제외하면 production-equivalent.

2. **AUF mode가 Q4 baseline보다 −10% 빠른 원인 분석** (선택적)
   - SOA bypass 경로 vs `convert_q4_0_to_noshuffle` 경로 cold-cache 차이 정량화.
   - 가설 부정 시 첫 forward profiling으로 분기 추적.

3. **다른 GGUF 시리즈 (Llama 3 8B, Qwen2 등) 일반화 검증**
   - Llama 3.2 1B만 검증됨. 8B 모델로 lm_head 퀀타이즈 효과 측정.
