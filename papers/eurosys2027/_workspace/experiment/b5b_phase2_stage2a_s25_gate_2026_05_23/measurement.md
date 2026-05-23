# B-5b Phase 2 Stage 2-A — S25 microbench 게이트 측정

**날짜**: 2026-05-23
**디바이스**: Galaxy S25 (R3CY408S5SB, Adreno 830)
**모델**: Qwen 2.5-1.5B Q4_0 GGUF (`/data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf`)
**Backend**: opencl
**Threads**: 6 (Galaxy S25 권장, 8T 금지 feedback 준수)
**Prompt**: "The capital of France is"
**Args**: `--num-tokens 32 --max-seq-len 512 --greedy --repetition-penalty 1.1 --backend opencl --threads 6`
**Runs**: N=5 per HEAD

## 빌드 조건

| 항목 | 값 |
|---|---|
| target | aarch64-linux-android |
| features | opencl (no-default-features) |
| profile | release (lto=fat) |
| NDK | /opt/android-ndk (android-ndk toolchain via hosts.toml) |
| 비고 | devices.toml의 `features = [opencl, vulkan, qnn]`는 사전 회귀(qnn_oppkg interface 미해결 imports 17건+, B-5b 무관)로 빌드 불가 → 본 게이트 작업 명세대로 opencl 단독 빌드 |

## 비교 대상

| HEAD | label | scope |
|---|---|---|
| `b4193bab` | baseline | B-5b Phase 2 Stage 1 종결 (인프라 only — 호출지 미치환) |
| `6cd09f9b` | stage2a | B-5b Phase 2 Stage 2-A (cpu_kernels 8건 + cpu_companion 25건 호출지 치환) |

## 결과 — avg_tbt (tok0 inclusive)

| run | baseline (ms) | stage2a (ms) |
|---|---|---|
| 1 | 32.96 | 32.91 |
| 2 | 32.99 | 32.91 |
| 3 | 33.00 | 32.93 |
| 4 | 32.97 | 32.82 |
| 5 | 32.87 | 32.84 |
| **mean** | **32.958** | **32.882** |
| stdev | 0.052 | 0.049 |
| min | 32.870 | 32.820 |
| max | 33.000 | 32.930 |

## Δ 및 게이트 판정

| metric | baseline | stage2a | Δ | Δ% |
|---|---|---|---|---|
| **avg_tbt (ms)** | 32.958 | 32.882 | **−0.076** | **−0.231%** |
| decode/tok rest (ms) | 30.988 | 30.894 | −0.094 | −0.303% |
| TTFT (ms) | 94.054 | 94.570 | +0.516 | +0.549% |

**게이트 판정: PASS** — avg_tbt Δ = −0.231% (≤ +3% 통과 마진 안에서 오히려 측정상 미세 개선/동등)

## handoff 14.66 ms/tok 수치와의 차이

handoff `handoff_b5b_phase2_stage1_complete_2026_05_23.md`의 14.66 ms/tok은
다른 측정 컨텍스트(`project_weight_swap_tbt_gap_root_cause` Sprint A~F 종결, Mixed 모드,
"Q4 baseline -10.1%" 시점)의 절대값으로, 본 게이트 측정과 직접 비교 불가.

본 게이트는 **동일 컨텍스트 동일 invocation**으로 baseline ↔ Stage 2-A를 재측정하여
Δ%로 판정. 따라서 절대값 33 ms 수준은 normal — 본 invocation의 standard happy path
(`[Phase4-4.5] standard happy path → DecodeLoop+ModelForward`) 기준.

## 출력 정합성

bit-identical: 모든 10회 (5 baseline + 5 stage2a) 동일 출력
> "The capital of France is Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into"

32 토큰 모두 일치. 추론 정확성 회귀 없음.

## 이상 신호

- logcat/stderr에서 `evict|dispatcher|pending|panic|abort|error|WARN` 라인 **0건**
- `[Phase4-4.5] generated=32 stopped_by=BudgetExhausted final_pos=36` 모두 동일
- `[NoShuffle] Skipped: default AOS` 일관

## 결정

R-1 RPN 145~180 우려 — Stage 2-A 한정으로는 **실측 영향 없음 (-0.23%)**.

- Stage 2-A vtable indirection 영향: 실측 noise 이하 → LTO=fat가 `Arc<dyn Backend>` capability method 호출에서도 충분히 디버추얼라이즈 또는 코스트가 측정 가능한 수준이 아님
- Stage 2-B (예정: as_opencl_secondary + yield_after_layer) 진행 시 동일 게이트 재측정 필요

## 데이터

- `raw/baseline_r{1..5}.txt` — 5 measurements of HEAD `b4193bab`
- `raw/stage2a_r{1..5}.txt` — 5 measurements of HEAD `6cd09f9b`
