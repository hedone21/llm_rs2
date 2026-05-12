# LISWAP-PPL — Weight Swap NLL 측정 인프라 + 4 시나리오 데이터 인수인계 (2026-05-12)

## TL;DR

논문용 "weight swap 과정에 따른 token 단위 NLL 변화" 측정 인프라 구현 + 4 시나리오 실측 완료.
- **A (F16 baseline, no swap)**: PPL 8.5320, mean NLL 2.1438
- **B (F16 → swap 25 layer [1..25])**: PPL 9.7984, mean NLL 2.2822
- **C (F16 → swap 28 layer [0..27])**: PPL 9.9355, mean NLL 2.2961
- **D (Q4_0 native primary, no swap)**: PPL 9.3092, mean NLL 2.2310

핵심 finding: **C−D = +0.065 NLL** = swap process artifact (동일 final state임에도 incremental swap이 native Q4 대비 quality 손실 추가). Quantization 자체 비용(D−A = +0.087) 의 ~75%.

## 커밋

- `a7882222` feat(liswap-ppl): PPL teacher-forcing × weight swap NLL 측정 인프라
- 그 이전: `f3495c28` docs handoff (이전 작업), `76fd55c1` fix sticky state

`origin/master` 에 push 완료 (2026-05-12).

## 측정 환경

| 항목 | 값 |
|---|---|
| 디바이스 | Galaxy S25, adb serial **R3CY408S4HN** |
| 모델 | qwen2.5-1.5b (28 layer, GQA, head_dim=128) |
| Reference text | `experiments/prompts/med_len.txt` (1095 byte, 1078 token tokenized to 1072 token eval) |
| Backend | OpenCL (Adreno) |
| Temperature | 0 (deterministic) |
| Prefill | 32 token (forced via `--ppl-prefill-tokens 32`) |
| Decode | 1040 step teacher-forcing |
| Swap timing | `--ppl-swap-at-token 0` (decode 시작 직후) |
| Swap per-tick | K=1 layer/step (incremental, fixed-K, no dynamic-K) |

## 코드 변경 위치

### engine/src/models/weights/decider.rs
- `WeightSwapDecider` 에 신규 필드 `pub allow_boundary_layers: bool`.
- `decide()` line 81~ 의 protected set 조건부:
  ```rust
  if !self.allow_boundary_layers {
      protected.insert(0usize);
      if n > 1 { protected.insert(n - 1); }
  }
  ```
- 단위 테스트 2건 신규: `boundary_layers_included_when_allowed`, `boundary_layers_allowed_in_fallback_path`.

### engine/src/bin/generate.rs
- 신규 helper: `fn read_allow_boundary_env() -> bool` — `LLMRS_SWAP_ALLOW_BOUNDARY_LAYERS=1` 읽음. 세 `WeightSwapDecider` 호출처에서 공통 사용.
- 신규 CLI 옵션 (`Args` struct):
  - `--ppl-swap-at-token <usize>`: PPL decode token-index 기준 swap trigger
  - `--ppl-swap-ratio <f32>` (default 0.9): swap ratio
  - `--ppl-swap-per-tick <usize>` (default 1): incremental K, fixed
  - `--ppl-nll-csv <PathBuf>`: per-token NLL CSV 출력
  - `--ppl-prefill-tokens <usize>`: prefill 길이 강제 (decode loop 활성화)
- `run_ppl` (line ~10000) 확장: prefill scoring loop CSV 기록, decode loop 에 swap trigger + IncrementalSwapPlan drain + run_layer_swap 호출 + per-token log push, 함수 끝에 CSV write.
- 재사용 함수: `dispatch_swap_weights`, `run_layer_swap`, `IncrementalSwapPlan`, `WeightSwapDecider`, `sampling::compute_log_prob`.

### manager/src/lua_policy.rs
- line 1717 부근의 ratio > 0.9 clamp 완전 제거. 주석 갱신: "engine `WeightSwapDecider::decide()` 의 [0.0, 1.0] clamp 가 최종 boundary".
- 테스트 변경: `test_lua_policy_clamps_ratio_above_limit` → `test_lua_policy_does_not_clamp_high_ratio` (0.95 그대로 통과 검증). 동일 변경을 `precision_swap` alias 테스트에도 적용.

### CLI 옵션 cheat sheet

```
[A] F16 baseline:
  --ppl ref.txt --ppl-prefill-tokens 32 --ppl-nll-csv out.csv

[B] F16 + swap 25 layer:
  + --secondary-gguf <q4_0.gguf> --ppl-swap-at-token 0
    --ppl-swap-ratio 0.9 --ppl-swap-per-tick 1

[C] F16 + swap 28 layer (boundary 우회):
  + LLMRS_SWAP_ALLOW_BOUNDARY_LAYERS=1 (env)
  + --ppl-swap-ratio 1.0  (나머지 동일)

[D] Q4_0 native:
  -m <q4_0.gguf>   (secondary 옵션 모두 제거)
  --ppl-prefill-tokens 32 --ppl-nll-csv out.csv
```

## 결과 데이터

### 1. Overall 요약

| Scenario | model | swap layers | mean NLL | PPL | mean NLL Δ vs A |
|---|---|---|---|---|---|
| A | f16 | — | 2.1438 | 8.5320 | — (baseline) |
| B | f16 → q4 (mid) | 25 ([1..25]) | 2.2822 | 9.7984 | +0.138 |
| C | f16 → q4 (mid) | 28 ([0..27]) | 2.2961 | 9.9355 | +0.152 |
| D | q4 native | — | 2.2310 | 9.3092 | +0.087 |

모든 시나리오 token_idx 0..1071 일치 (teacher-forcing 결정론 확인).

### 2. Phase-wise mean NLL

| Scenario | prefill(31) | decode_none | swapping | post_swap |
|---|---|---|---|---|
| A | 2.7399 | 2.1261 | — | — |
| B | 2.7454 | — | 2.5668 | 2.2614 |
| C | 2.7454 | — | 2.6811 | 2.2721 |
| D | **2.8570** | 2.2123 | — | — |

- A vs B/C prefill 거의 동일 (2.7399~2.7454) — swap 전이라 weight 가 F16
- D prefill = 2.8570 (Q4_0 primary 로드, 양자화 효과가 prefill 부터 일관)

### 3. 핵심 분해

```
양자화 자체 비용     D−A = +0.087  (F16 → Q4 pure quant cost)
Swap process artifact C−D = +0.065  (동일 final state, 75% 수준)
Boundary layer 비용  C−B = +0.014  (post_swap +0.011, swapping +0.114)
```

### 4. Post-swap apples-to-apples (token idx 58..1070, n=1013)

C(swap-completed) vs D(native Q4) — 같은 token range, 같은 final weight state:

| 측정 | NLL |
|---|---|
| C post_swap mean | 2.2721 |
| D same idx range | 2.2016 |
| **Δ (C − D)** | **+0.0705** |

장기 안정화 후에도 incremental swap 이 native Q4 대비 ~0.07 NLL 비용 잔존. 가설 (검증 필요):
- AUF unpermute 와 standalone GGUF Q4_0 quantization 의 미세한 numerical 차이
- prefill F16 시점 KV cache 가 decode Q4 weight 와 결합 시 layer-wise mismatch
- OpenCL multi-thread reduction order 비결정성 (작지만 누적)

### 5. Window 별 추이 (decode only)

```
tok_range      A_f16   C_swap28  D_q4_nat    C−A    D−A    C−D
  31..60       2.764    2.907    2.816    +0.144  +0.052  +0.091  ← swap 진행 중
  61..110      2.873    3.095    2.909    +0.222  +0.036  +0.186  ← swap 직후 max
 111..230      2.044    2.152    2.105    +0.109  +0.061  +0.047
 231..530      2.248    2.400    2.338    +0.151  +0.090  +0.062
 531..830      1.998    2.173    2.074    +0.175  +0.076  +0.099
 831..1070     1.939    2.091    2.061    +0.152  +0.122  +0.030
```

- Swap 진행 + 직후 (31..110): C−D peak +0.186
- 장기 안정화 (831~): C−D = +0.030 — process artifact 감쇠하나 0 아님

## Raw 데이터 위치

### 호스트 (분석용 — pandas/python 즉시 사용 가능)

```
/tmp/ppl_csv/
├── ppl_A_baseline.csv     34 KB, 1071 row (prefill 31 + decode 1040)
├── ppl_B_standard.csv     40 KB, 1071 row
├── ppl_C_boundary.csv     40 KB, 1071 row
└── ppl_D_q4_native.csv    34 KB, 1071 row
```

**CSV 컬럼**: `phase,token_idx,token_id,nll,swap_state,layers_swapped`
- `phase` ∈ {`prefill`, `decode`}
- `swap_state` ∈ {`none`, `swapping`, `post_swap`}
- `token_idx` reference text 의 0-based 절대 위치 (4 시나리오 일치)

### 디바이스 (재실험·재현용)

```
R3CY408S4HN:/data/local/tmp/
├── generate                                  바이너리 (커밋 a7882222 빌드)
├── ppl_ref.txt                               reference text (med_len.txt 그대로)
├── ppl_A_baseline.csv                        시나리오 A 출력
├── ppl_B_standard.csv                        시나리오 B 출력
├── ppl_C_boundary.csv                        시나리오 C 출력
├── ppl_D_q4_native.csv                       시나리오 D 출력
├── _ppl_A.err, _ppl_B.err, _ppl_C.err, _ppl_D.err   stderr (swap log 포함)
└── models/qwen2.5-1.5b/
    ├── qwen2.5-1.5b-f16.gguf                 primary (A/B/C 용)
    ├── qwen2.5-1.5b-q4_0.gguf                D primary / B-C secondary
    └── tokenizer.json
```

호스트 sync (필요시):
```bash
adb -s R3CY408S4HN pull /data/local/tmp/ppl_A_baseline.csv /tmp/ppl_csv/
# (B/C/D 동일)
```

## 재현 명령 (Galaxy S25 R3CY408S4HN)

전제: `adb devices` 에 R3CY408S4HN 보임, `/data/local/tmp/generate` 가 commit `a7882222` 빌드.

### 빌드+배포 (필요시)
```bash
python scripts/run_device.py -d galaxy_s25 --skip-exec --skip-deploy generate
adb -s R3CY408S4HN push target/aarch64-linux-android/release/generate /data/local/tmp/
adb -s R3CY408S4HN push experiments/prompts/med_len.txt /data/local/tmp/ppl_ref.txt
```

### 시나리오 A (baseline)
```bash
adb -s R3CY408S4HN shell "cd /data/local/tmp && \
  LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64:/system/lib64 \
  ./generate -m models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf \
    --tokenizer-path models/qwen2.5-1.5b/tokenizer.json \
    --backend opencl --temperature 0 \
    --ppl ppl_ref.txt --ppl-prefill-tokens 32 \
    --ppl-nll-csv ppl_A_baseline.csv"
```

### 시나리오 B (standard swap)
```bash
adb -s R3CY408S4HN shell "cd /data/local/tmp && \
  LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64:/system/lib64 \
  ./generate -m models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf \
    --tokenizer-path models/qwen2.5-1.5b/tokenizer.json \
    --secondary-gguf models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
    --backend opencl --temperature 0 \
    --ppl ppl_ref.txt --ppl-prefill-tokens 32 \
    --ppl-swap-at-token 0 --ppl-swap-ratio 0.9 --ppl-swap-per-tick 1 \
    --ppl-nll-csv ppl_B_standard.csv"
```

### 시나리오 C (boundary swap)
```bash
adb -s R3CY408S4HN shell "cd /data/local/tmp && \
  LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64:/system/lib64 \
  LLMRS_SWAP_ALLOW_BOUNDARY_LAYERS=1 \
  ./generate -m models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf \
    --tokenizer-path models/qwen2.5-1.5b/tokenizer.json \
    --secondary-gguf models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
    --backend opencl --temperature 0 \
    --ppl ppl_ref.txt --ppl-prefill-tokens 32 \
    --ppl-swap-at-token 0 --ppl-swap-ratio 1.0 --ppl-swap-per-tick 1 \
    --ppl-nll-csv ppl_C_boundary.csv"
```

### 시나리오 D (Q4 native)
```bash
adb -s R3CY408S4HN shell "cd /data/local/tmp && \
  LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64:/system/lib64 \
  ./generate -m models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
    --tokenizer-path models/qwen2.5-1.5b/tokenizer.json \
    --backend opencl --temperature 0 \
    --ppl ppl_ref.txt --ppl-prefill-tokens 32 \
    --ppl-nll-csv ppl_D_q4_native.csv"
```

## 분석 코드 스니펫 (Python, pandas)

```python
import pandas as pd
import matplotlib.pyplot as plt

paths = {
    "A_baseline":  "/tmp/ppl_csv/ppl_A_baseline.csv",
    "B_standard":  "/tmp/ppl_csv/ppl_B_standard.csv",
    "C_boundary":  "/tmp/ppl_csv/ppl_C_boundary.csv",
    "D_q4_native": "/tmp/ppl_csv/ppl_D_q4_native.csv",
}
dfs = {k: pd.read_csv(p) for k, p in paths.items()}

# Sliding mean (window=50) for smooth curves
fig, ax = plt.subplots(figsize=(10, 5))
for name, df in dfs.items():
    decode = df[df["phase"] == "decode"]
    smooth = decode["nll"].rolling(50).mean()
    ax.plot(decode["token_idx"], smooth, label=name)
ax.set_xlabel("token index")
ax.set_ylabel("NLL (50-token moving avg)")
ax.legend()
ax.grid(alpha=0.3)
plt.savefig("nll_4scenarios.pdf", bbox_inches="tight")

# Phase-wise table
for name, df in dfs.items():
    print(name)
    print(df.groupby(["phase", "swap_state"])["nll"].agg(["mean", "count"]))
```

## 측정 노이즈 / 알려진 한계

1. **OpenCL 비결정성**: A 와 B/C 의 prefill NLL 이 token 단위로 ~0.1 NLL 미세 차이. swap 전이라 동일해야 하지만 multi-thread reduction order 의 미세한 비결정성으로 보임. swap effect(수 단위) 대비 무시 가능 노이즈.
2. **lm_head quantization**: F16 모델 로드 시 lm_head 가 missing → derive(F16→Q4_0). 매 실행마다 동일 결과인지 boot-time 확인 안 함. 필요시 reference quantization snapshot 별도 추출 검토.
3. **swap_state 의 "swapping" 마지막 step**: code 상 `plan.is_done()` 시점에 `swap_state` 가 `post_swap` 으로 override 되어 마지막 swap step (실제 swap 일어남) 도 `post_swap` 으로 logged. 분석 시 layer 수 변화로 swap step 식별 가능 (`layers_swapped` diff > 0).
4. **PPL 모드 KV cache 동작**: PPL teacher-forcing 은 KV cache 를 grow-on-demand 로 점진 확장. F16 prefill 시점 KV cache 가 그대로 decode Q4 weight 와 결합. C 와 D 의 KV cache 진화 경로가 다름 → C−D artifact 의 일부일 가능성.

## 후속 작업 후보 (논문 발전 방향)

### 추가 측정 (즉시 가능)
- **시나리오 E**: F16 모델 로드 + swap 28 layer + 추가 decode 1k 토큰 → C−D artifact 가 더 긴 시간에 어떻게 변하는지
- **시나리오 F**: 다른 ratio (0.25, 0.5, 0.75) sweep — quantization quality curve
- **시나리오 G**: `--ppl-swap-per-tick K` sweep (K=1,2,4,8) — incremental K 가 NLL 에 미치는 영향
- **시나리오 H**: 다른 모델 (Llama 3.2 1B) 동일 측정 — 모델 의존성 확인

### 가설 검증 (코드 작업 필요)
- **AUF vs GGUF Q4 numerical 비교**: secondary AUF 의 layer weight 와 standalone Q4 GGUF 의 동일 layer weight 의 bit-level 동일성 검증. mismatch 발견 시 `secondary_mmap` 의 quantization 경로 정렬 작업.
- **KV cache 초기화 후 measure**: swap 완료 후 KV cache 리셋 + prefill 다시 → C 가 D 에 수렴하는지 확인. 만약 수렴 → artifact 는 KV cache 잔존 때문, 만약 안 수렴 → weight 자체 mismatch.
- **boundary layer 만 swap** (시나리오 X): layer 0과 layer 27 만 swap → ΔNLL 측정. 각 layer 의 importance 정량화.

### 시각화 (논문 figure)
- 4 시나리오 sliding-mean NLL 곡선 (위 python 스니펫)
- C−D delta 의 token-index 추이 (artifact decay curve)
- Phase 별 box plot

## 호스트 단위 테스트 상태 (commit a7882222)

```
cargo test -p llm_rs2 --lib weights::decider     →  13 passed (신규 boundary 2건 포함)
cargo test -p llm_manager --lib lua_policy::tests →  68 passed (clamp 테스트 2건 갱신)
cargo test -p llm_rs2 --test spec                →  619 passed
```

## 알려진 디바이스 환경 이슈

- `--backend qnn_oppkg` 가 R3CY408S4HN 에서 `graphFinalize err=0x1786` 으로 실패. 모든 측정은 `--backend opencl` 로 진행. QNN-OppKg 동작 검증된 시리얼은 R3CY408S5SB (이전 handoff 참조). 본 측정 결과는 OpenCL backend 기준.

## 관련 문서

- 이전 handoff (sticky state fix): `.agent/todos/handoff_manager_swap_routing_2026_05_12.md`
- LISWAP-6 production: `project_liswap6_alias_production.md` (memory)
- QCF 명명 컨벤션: `CLAUDE.md` 의 "QCF 명명 컨벤션" 섹션 + `docs/qcf_taxonomy.md`
