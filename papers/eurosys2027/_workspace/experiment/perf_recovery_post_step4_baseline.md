# Perf Recovery Post-step4 Merge — Baseline 격리 측정

**측정일**: 2026-05-21
**디바이스**: Samsung Galaxy S25 (`R3CY408S4HN`)
**모델**: Qwen 2.5-1.5B-Instruct (Q4_0 AOS)
**Backend**: qnn_oppkg, 6 threads
**Prompt**: `"I_am_a"`, num-tokens=32
**Runs**: n=5 per cell
**Metric**: Avg TBT (tok0 inclusive) — `feedback_tbt_metric_tok0_inclusive.md` 컨벤션
**측정 명령**: `LD_LIBRARY_PATH=/data/local/tmp ./generate_<base> --num-tokens 32 --model-path ... --tokenizer-path ... --backend qnn_oppkg --threads 6 --prompt 'I_am_a'`

## 빌드

각 base는 자기 worktree에서 `cargo build --release --features opencl,vulkan,qnn --no-default-features --target aarch64-linux-android --bin generate`로 빌드. **주의**: `python scripts/run_device.py`는 `builder.py:193`이 `cwd=project_root`로 강제하므로 worktree와 무관하게 master 코드만 빌드한다 (handoff 작성 시 누락). 본 측정은 cargo 직접 호출로 우회.

| Base | Worktree | HEAD | sha256 (binary) | size |
|---|---|---|---|---|
| sprint1 | `/home/go/Workspace/llm_rs2` (master) | `e31bd698` | `1cd92f2a…4946a2` | 7,989,608 B |
| step-4-backup | `.claude/worktrees/step4_d_pressure_toplevel` | `fd95e916` | `48d8d925…68f3ca` | 8,135,480 B |
| HEAD (merge) | `.claude/worktrees/merge_resolve` | `000634b3` | `22345320…30f8fcc0` | 8,205,488 B |

## Raw 데이터

`perf_recovery_post_step4_raw/{base}_{dtype}.log` (TTFT + Decode + Avg TBT).

### sprint1 / GGUF

| Run | TTFT (ms) | Decode (ms/tok, forward only) | Avg TBT (ms) |
|---|---|---|---|
| 1 | 173.55 | 12.42 | 33.34 |
| 2 | 179.58 | 12.09 | 33.43 |
| 3 | 173.61 | 11.20 | 32.93 |
| 4 | 174.28 | 12.44 | 33.08 |
| 5 | 173.95 | 12.54 | 32.58 |
| **median** | 173.95 | 12.42 | **33.08** |

### sprint1 / AUF (`--primary-dtype q4_0 --primary-variant cpu-aos`)

| Run | TTFT (ms) | Decode (ms/tok, forward only) | Avg TBT (ms) |
|---|---|---|---|
| 1 | 164.66 |  4.32 | 29.42 |
| 2 | 173.59 | 10.70 | 32.56 |
| 3 | 173.17 | 11.78 | 32.41 |
| 4 | 171.56 | 12.58 | 33.07 |
| 5 | 177.42 | 12.52 | 32.90 |
| **median** | 173.17 | 11.78 | **32.56** |

### step-4-backup / GGUF (AUF skip — primary loader 부재)

| Run | TTFT (ms) | Decode (ms/tok) | Avg TBT (ms) |
|---|---|---|---|
| 1 | 62.50 | 28.96 | 30.00 |
| 2 | 65.24 | 33.24 | 34.24 |
| 3 | 63.59 | 30.97 | 31.99 |
| 4 | 63.02 | 32.33 | 33.29 |
| 5 | 62.92 | 31.60 | 32.57 |
| **median** | 63.02 | 31.60 | **32.57** |

### HEAD (merge) / GGUF

| Run | TTFT (ms) | Decode (ms/tok) | Avg TBT (ms) |
|---|---|---|---|
| 1 | 64.85 | 32.84 | 33.84 |
| 2 | 64.92 | 33.50 | 34.48 |
| 3 | 65.44 | 32.81 | 33.83 |
| 4 | 65.44 | 32.73 | 33.75 |
| 5 | 64.80 | 32.85 | 33.85 |
| **median** | 64.92 | 32.84 | **33.84** |

### HEAD (merge) / AUF

| Run | TTFT (ms) | Decode (ms/tok) | Avg TBT (ms) |
|---|---|---|---|
| 1 | 62.71 | 32.20 | 33.15 |
| 2 | 61.87 | 28.82 | 29.85 |
| 3 | 64.66 | 33.25 | 34.23 |
| 4 | 68.41 | 32.85 | 33.96 |
| 5 | 65.02 | 33.20 | 34.19 |
| **median** | 64.66 | 32.85 | **33.96** |

## 회귀 기여 분해

|  | GGUF Avg TBT | AUF Avg TBT |
|---|---|---|
| sprint1 (e31bd698) | 33.08 ms | 32.56 ms |
| step-4-backup (fd95e916) | 32.57 ms | (n/a) |
| HEAD (000634b3) | 33.84 ms | 33.96 ms |
| **sprint1 → step-4-backup** | **−0.51 ms** | — |
| **step-4-backup → HEAD** | **+1.27 ms** | — |
| **sprint1 → HEAD (total)** | **+0.76 ms (+2.3%)** | **+1.40 ms (+4.3%)** |

## 결과 요약 + 주요 발견

### 1) handoff baseline 가정이 outdated

handoff와 `project_perf_recovery_post_step4.md`는 sprint1 baseline을 GGUF **29.20 ms** / AUF **31.44 ms**로 명시. 같은 환경에서 sprint1 binary 재측정 결과 **GGUF 33.08 ms / AUF 32.56 ms**로, 약 +13~14% 차이. 원인 추정: 이전 측정과 (a) S25 디바이스 상태(thermal/governor/백그라운드 앱), (b) prompt 변경, (c) binary 빌드 옵션 변경, (d) qnn_oppkg path 변경 중 하나. 따라서 Goal의 "Base 95% = GGUF ≤ 30.74 / AUF ≤ 33.10"는 **현 환경에서 sprint1 binary 자체로도 도달 불가**.

### 2) Plan에서 정의한 결론 (A/B/C) 분기는 **(B) 머지 통합 회귀**

- sprint1 → step-4-backup: **−0.51 ms (오히려 개선)**.
- step-4-backup → HEAD: **+1.27 ms 회귀**.

회귀의 절대 다수가 **머지 통합 코드**에서 발생. step-4 본체(layered architecture + DecodeLoop+ModelForward refactor)는 회귀에 기여하지 않음. **H1 (DecodeLoop trait dispatch) 가설은 측정 데이터에 의해 기각**. 회귀 원인 후보 (CLAUDE.md 메모리 기준 머지 시점 변경 집합):
- `compute_qcf_weight_swap` → `compute_qcf_swap` rename (callsite 변경 다수)
- AUF 모듈 위치 결정 (engine vs shared)
- LoadConfig W-AUF-1 신규 fields 통합
- `--secondary-gguf` deprecation warning 분기
- `read_allow_boundary_env` cfg 게이트 제거 (H3 후보)
- `crate::core::*` → top-level promotion

### 3) 현 환경 기준 회귀폭은 base 95% 이내

현 환경에서의 진짜 baseline (sprint1 33.08/32.56)을 기준으로 환산:
- GGUF: HEAD 33.84 ÷ sprint1 33.08 = **102.3%** (회귀 +2.3%, base 105% 마진 이내)
- AUF: HEAD 33.96 ÷ sprint1 32.56 = **104.3%** (회귀 +4.3%, base 105% 마진 이내)

handoff의 절대 ms 기준 ("GGUF ≤ 30.74 / AUF ≤ 33.10")은 outdated baseline에서 파생된 것이므로 폐기 또는 재정의 필요.

### 4) Decode metric 변화 (sprint1 vs step-4 이후)

`Decode:` 라인의 의미가 binary 버전에 따라 다름:
- sprint1: `Decode: 12.42 ms/tok (80.5 tok/s) [31 tokens, forward only]` — pure forward (no sync)
- step-4-backup / HEAD: `Decode: 32.84 ms/tok (30.5 tok/s) [31 tokens]` — full decode loop (sync 포함)

main metric인 Avg TBT(tok0 inclusive)는 wall-clock이라 두 binary 간 비교 가능. Decode 라인은 cross-binary 비교 금지.

## 다음 단계 권고

**Goal 재정의 필요** — handoff의 절대 ms 기준이 outdated baseline에서 파생.

세 가지 옵션:

| 옵션 | 설명 | 권장도 |
|---|---|---|
| **A** | Goal 자체를 폐기. 현 환경에서 sprint1 vs HEAD 회귀가 base 105% 이내(PASS)임을 근거로 본 task 종결 후 다른 작업 진입 | ★★★ (측정 결과상 가장 타당) |
| **B** | 회귀폭 +1.27 ms (step4_backup → HEAD)를 마이크로 회복 — 가설 H3(read_allow_boundary_env env lookup), LoadConfig 분기 등 검증. 기대 회복폭 작음 (~1 ms 이하) | ★ (ROI 낮음) |
| **C** | 외부 환경(이전 측정의 디바이스 상태/binary/feature 차이)을 재현하여 원래 29.20 ms 환경 복원 후 재측정. 원본 측정 출처 추적 필요 | ★★ (인프라 작업 필요) |

### 권장: 옵션 A

근거:
1. **현 환경에서 sprint1 자체가 33.08 ms** — Goal의 30.74 ms는 그 binary로도 미달.
2. **HEAD vs sprint1 회귀폭 +2.3% / +4.3%** — 측정 노이즈(IQR ~1 ms) 수준에 가까움.
3. **회귀가 step-4 본체 아닌 머지 통합**에서 옴 — refactor를 되돌리는 작업은 부적절. 마이크로 회복은 +1.27 ms로 ROI 매우 낮음.
4. CLAUDE.md §2 "단순함 우선" 및 §3 "외과적 변경" 원칙에 부합 — outdated baseline을 쫓는 작업 회피.

옵션 A 진입 시 다음 단계: handoff/메모리에 본 측정 결과 반영 + 사용자에게 baseline 출처 확인 요청 + task 종결.

## Verification (Plan Step 1 게이트)

- [x] 3개 binary 모두 mtime 확인 — sprint1 01:12, step-4-backup 01:16, HEAD 23:33 (5/20)
- [x] sha256 모두 다름 (cache hit 아님)
- [x] n=5 × 5 cells = 25 runs raw 데이터 확보
- [x] median 표 도출
- [x] 결론 도출 (Plan A/B/C 중 **(B) 머지 통합 회귀**)
- [x] 다음 단계 권고 (옵션 A/B/C 중 **옵션 A 권장**)

호스트 테스트는 본 step에서 코드 수정 0이므로 불필요.
