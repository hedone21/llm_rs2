# Phase 4-4.5 — S25 OpenCL DecodeLoop paradigm 통일 검증

- HEAD post: `6292a9d0` (Phase 4-4.5)
- HEAD baseline: 메인 세션이 만든 `/data/local/tmp/generate_baseline` (timestamp 2026-05-17 15:06, HEAD `6a224c9f` 의도)
- Device: Galaxy S25 (R3CY408S5SB), Adreno OpenCL
- Model: Qwen 2.5 1.5B Q4_0 GGUF
- Prompt: "The capital of France is" (5 tokens)
- Decode budget: 32 tokens
- Sampling: --greedy --temperature 0.0
- 측정 일자: 2026-05-17

## G6 — bit-identical 32 token greedy

### baseline (`generate_baseline`) 출력
```
The capital of France is Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into
```

### post-4-4.5 (`generate`) 출력
```
The capital of France is Paris. Paris is a very big city. It has about 2.2 million people. It is also a very old city. It was built in the
```

### 분기 진입 로그 (post)
```
[Phase4-4.5] standard happy path → DecodeLoop+ModelForward (tokens=5, budget=32)
[Phase4-4.5] generated=32 (first=12095 + run=31) stopped_by=BudgetExhausted final_pos=36
```

### 분기 진입 로그 (baseline)
없음 — baseline binary는 `[Phase4-4-b]` 또는 `[Phase4-4.5]` 진입 라인을 stderr로 출력하지 않음. 즉 **baseline binary 가 happy path 분기를 타지 않고 fallback main path 로 흐름**.

### Verdict
**FAIL (검증 불성립)** — baseline 과 post 는 서로 다른 코드 path (fallback vs happy path) 를 실행. 동일 path 출력을 bit-identical 비교 할 수 없음.

토큰 sequence 분기 위치: 두 경로 모두 첫 sample token id = 12095 (` Paris`) 까지 동일하나, 두 번째 token 부터 분기.
- baseline (fallback path): ` Paris . It has a population of about 2 million ...`
- post (happy path): ` Paris . Paris is a very big city ...`

post happy path 출력의 ` Paris . Paris is` 패턴은 step1 에서 prompt 마지막 token 의 forward 결과로 first_token=` Paris` 를 얻은 뒤, step2 에서 `forward.step(prev_token= Paris)` 의 결과가 ` .` 이고, step3 에서 `forward.step(prev_token= .)` 의 결과가 ` Paris` 인 자연스러운 greedy 결과로도 해석 가능. 즉 post 출력이 자체 모순적이지 않다.

다만 **fallback path 의 출력이 정답이라는 보증이 없음** — fallback main loop 가 Phase 4-4-b/4-4.5 와 동일한 inference 의미론을 따르는지는 별도 검증 대상.

## G7 — avg_tbt Δ ≤ 5%

### baseline (5 runs, 6T)
| run | Decode (ms/tok) | Avg TBT (ms) |
|-----|-----------------|--------------|
| 1   | 27.67           | 29.48        |
| 2   | 27.90           | 29.72        |
| 3   | 27.90           | 29.57        |
| 4   | 28.03           | 29.79        |
| 5   | 27.80           | 29.58        |
| median | 27.90        | 29.58        |

### post-4-4.5 (5 runs, 6T)
TBT 측정 **불가능**. happy path 분기 (`generate.rs` Phase 4-4.5 진입 블록) 는 TBT/Decode 통계를 stderr 로 출력하지 않음 — `[Phase4-4.5] generated=...` 라인만 출력하고 main 종료. fallback main path 의 `Decode: X ms/tok` / `Avg TBT: Y ms` 라인은 진입하지 않음.

### Verdict
**측정 불가** — post happy path 가 TBT 통계 라인을 출력하지 않아 정량 비교 불가.

## 발견 사항 요약

1. **baseline binary 가 happy path 미진입**: `/data/local/tmp/generate_baseline` (2026-05-17 15:06 빌드) 는 5-token prompt + 기본 옵션에서도 `[Phase4-4-b]` 분기 라인을 출력하지 않음. 가능한 원인:
   - baseline binary 가 HEAD `6a224c9f` 가 아닌 더 이전 HEAD (Phase 4-4-b 도입 전) 에서 빌드되었을 가능성
   - 또는 `is_standard_happy_path()` 가드 가 어떤 args 로 fail 했을 가능성 (당 측정에서는 happy path 진입 조건을 모두 만족하는 args 사용)

2. **post-4-4.5 happy path 분기 자체에 TBT 통계 출력 코드 부재**: `generate.rs` PHASE 4-4.5 블록 (라인 3034~3090 부근) 은 token sequence 출력 + `[Phase4-4.5] generated=...` 진단 라인만 출력하고 즉시 main 을 종료. `Decode: X ms/tok` 등은 fallback main loop 의 출력. G7 검증을 위해서는 happy path 분기에도 동일한 TBT 통계 출력 추가 필요.

3. **Sequence 비교는 fallback vs happy path 의 의미론 등가성** 으로 재정의되어야 함. 4-4-b 의 의도된 fix (T5 중복 forward 제거) 가 happy path 출력에 어떻게 반영되는지 별도 oracle (e.g. llama.cpp greedy) 비교가 필요.

## 권장 후속 조치 (메인 세션)

- A. baseline binary 재빌드: HEAD `6a224c9f` checkout → `run_device.py -d galaxy_s25 --skip-exec generate` → `adb push generate /data/local/tmp/generate_baseline`. 재빌드 후 `[Phase4-4-b]` 진입 라인 확인.
- B. happy path 분기에 TBT 통계 출력 코드 추가 (generate.rs PHASE 4-4.5 블록). 또는 DecodeLoop::run 이 step 마다 step_ms 를 누적하므로 result 에서 avg 계산.
- C. fallback vs happy path 의미론 등가성 검증 oracle 마련 (예: llama.cpp greedy 또는 CPU forward 결과 와의 4-token bit-identical).

---

## Fix 1: `prefill_workspace=None` + TBT log (HEAD `7f7c6856`)

### 변경
- `ModelForward::prefill` 가 `forward_into` 인자로 `prefill_workspace: None` 전달 (production fallback 과 동일하게 owned workspace 자동 생성)
- happy path 블록에 `TTFT: X ms` / `Decode: Y ms/tok` / `Avg TBT: Z ms` 출력 추가

### G6 재측정 (32 token greedy)

baseline:
```
The capital of France is Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into
```

post-fix:
```
The capital of France is Paris. Paris is a very big city. It has about 2.2 million people. It is also a very old city. It was built in the
```

분기 진입 확인 (post-fix only): `[Phase4-4.5] standard happy path → DecodeLoop+ModelForward (tokens=5, budget=32)`

#### 첫 5 token 분기 분석
공통 prefix tokens: ` Paris`, `.` (step 0~1)
- baseline step 2: ` It`
- post-fix step 2: ` Paris`

→ Decode step 1 (input token id = `.`) 의 forward 결과가 baseline 과 post-fix 에서 다름. Prefill 직후 첫 sampling (token=` Paris`) 은 일치 → prefill numeric state OK. 첫 decode step 부터 logits 가 다름.

### Verdict — G6 FAIL
prefill_workspace=None fix는 **불충분**. 분기 원인이 `prefill_workspace` 보유가 아니라 다른 곳.

### G7 — 측정 보류
G6 FAIL 이므로 TBT 비교의 의미가 없음. happy path 가 `Avg TBT: 32.00 ms` (5번 측정 안 했지만 단일 run) 출력은 정상 — production fallback baseline median 29.58 ms 대비 약 +8% regression 의심 가능 (단일 run 이라 신뢰 부족, 5-run median 필요).

### 회귀 원인 추정 (G6 FAIL)
prefill 가 일치하므로 (첫 token=` Paris`):
- ❌ `ModelForward::prefill` 의 numeric state 차이 → **이번 fix 로 해소됨**
- ✅ `ModelForward::step` 의 forward_into 인자가 production decode (`generate.rs:4388 부근`) 와 다름 — `decode_workspace`, `kv_cache`, `score_buf`, `decode_logits_buf` ownership/aliasing 차이
- ✅ KV cache 의 초기 capacity / grow 동작이 ModelForward path 에서 다름 (예: ModelForward 가 명시 capacity 로 alloc, production 은 동적 grow)
- ✅ `score_buf` 혹은 `decode_workspace` 가 step 사이에 잘못 reset 되어 attention residual 누수

### 다음 단계 후보
1. `ModelForward::step` 의 `forward_into` 인자를 production decode (`generate.rs:4388`) 와 1:1 매칭 (특히 `decode_workspace` 의 lifecycle — production 은 step 마다 새로 만드는지 reuse 하는지 확인)
2. KV cache 초기 capacity 를 ModelForward path 에서도 production 과 동일 (128 또는 prompt-len-rounded-up) 로 강제
3. 첫 step 후 logits 첫 10개 dump 디버그 로그 일시 추가 (메인 세션 작업)

### 첫 5 token (텍스트 추출, token-id printer 옵션 부재로 정확 id 추출 불가)
- Baseline: ` Paris`, `.`, ` It`, ` has`, ` a`
- Post-fix: ` Paris`, `.`, ` Paris`, ` is`, ` a`

분기점: step 2 (3번째 sampled token).

### 단일 run 통계 비교 (참고, n=1)
| metric | baseline | post-fix | Δ |
|--------|----------|----------|----|
| TTFT (ms) | 220.59 | 92.37 | -58% |
| Decode (ms/tok) | 28.43 | 30.06 | +5.7% |
| Avg TBT (ms) | 30.15 | 32.00 | +6.1% |

Decode/TBT 모두 5% 게이트 초과. 단, n=1 이라 noise 가능. G6 PASS 후 5-run median 으로 재측정 필요.

TTFT 대폭 감소는 happy path 가 prefill warmup 출력 (`[WARMUP] tokens=1 ms=45.07`) 을 skip 하기 때문 — baseline 의 220.59 ms 에는 warmup 시간이 포함되어 있고, post-fix 92.37 ms 는 happy path 가 warmup 없이 prefill 직행한 값. TTFT 비교는 무효.

---

## Fix 2: KV cache initial_capacity 정합 (HEAD `57c0effd`)

### 변경
- `build_standard_loop(.., initial_kv_capacity: usize, ..)` 시그니처에 capacity 인자 추가
- `generate.rs` main()의 production 공식 `(prompt_len+num_tokens).next_power_of_two().max(128).min(max_seq_len)` 결과를 happy path로 전달
- 이전: `max_seq_len` (=512) 그대로 사용 → fix2: 5+32=37 → 128

### 가설 (반증됨)
KV cache 미사용 영역 garbage가 attention softmax/mask 처리에 침투 (특히 Adreno OpenCL).

### G6 재측정 (32 token, --temperature 0 --top-k 1)

baseline (deterministic):
```
The capital of France is Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into
```

post-fix2:
```
The capital of France is Paris. Paris is a very big city. It has about 2.2 million people. It is also a very old city. It was built in the
```

분기 진입 확인 (post-fix2): `[Phase4-4.5] standard happy path → DecodeLoop+ModelForward (tokens=5, budget=32)` + `[Phase4-4.5] generated=32 (first=12095 + run=31) stopped_by=BudgetExhausted final_pos=36`. TBT log 정상 출력.

#### 첫 5 token 분기 분석 (fix1과 동일)
공통 prefix: ` Paris`, `.` (step 0~1)
- baseline step 2: ` It`
- post-fix2 step 2: ` Paris`

→ KV initial_capacity 가설 반증. fix1과 동일한 분기 위치.

### Verdict — G6 FAIL
fix2 (KV capacity 정합) **불충분**. fix1과 동일한 step 2에서 분기 → KV cache 미사용 영역 garbage 가설 아님.

### G7 — 측정 보류 (G6 FAIL)
단일 run 통계:
| metric | baseline | post-fix2 | Δ |
|--------|----------|-----------|----|
| TTFT (ms) | 218.79 | 91.98 | -58% (warmup skip artifact) |
| Decode (ms/tok) | 27.83 | 29.97 | +7.7% |
| Avg TBT (ms) | 29.53 | 31.91 | +8.1% |

n=1, G6 FAIL이므로 G7 비교 무의미.

### Sampling mismatch 발견 (별건)
이전 G6 비교에서 baseline은 `--greedy` flag로 호출되었으나, `generate.rs` standard generate 경로에서 `args.greedy`는 **사용되지 않음** (KIVI/eval-ll 경로에서만 sampling_config에 반영). 즉 이전 baseline 실행은 default sampling (temp=0.8, top_p=0.9, top_k=40) 으로 수행. 이번 측정은 `--temperature 0 --top-k 1` 명시로 deterministic 강제 → baseline 출력은 이전과 동일 (우연인지, 또는 `--greedy` flag 자체가 무의미한지는 별도 분석 필요).

post-fix2 happy path는 `last_logits.iter().enumerate().max_by(...)` argmax이므로 deterministic. → sampling 차이는 G6 분기의 원인이 아님 확정.

### 다음 fix 후보 (분기는 step 2에서 여전히 발생)

prefill 일치 (step 0 ` Paris`) + step 1 일치 (`.`) + step 2 분기 (` It` vs ` Paris`):

1. **ModelForward::step의 forward_into 인자 정밀 비교**
   - production: `generate.rs:4388` 부근
   - happy path: ModelForward 내부
   - 특히 `decode_workspace` ownership: production은 step마다 reuse vs ModelForward는 eager alloc인지 확인
   - `score_buf` / `decode_logits_buf` lifecycle

2. **ModelForward::new의 decode_workspace alloc 시점**
   - eager (one-shot at construction) vs lazy (per-step)
   - production fallback의 lifecycle과 1:1 매칭 여부

3. **`cpu_backend_arc`와 production effective_mem 차이**
   - `build_standard_loop` 인자로 전달된 cpu_backend_arc vs production main()의 effective_mem 구성
   - 같은 Memory instance/clone 여부 확인

4. **KV cache 자체 분리 여부**
   - prefill 시 사용한 KV cache vs decode 시 사용한 KV cache가 동일 인스턴스인지
   - ModelForward 내부에서 KV cache 새로 생성한다면 prefill 결과 손실

5. **첫 step 후 logits dump 디버그**
   - step 1 (input=`.`) 후 logits[:10]을 baseline/post 양쪽에서 비교
   - bit-level diff 위치 식별 → 어느 layer/operation에서 분기되는지 좁힘

### 첫 5 token 비교
- Baseline: ` Paris`, `.`, ` It`, ` has`, ` a`
- Post-fix2: ` Paris`, `.`, ` Paris`, ` is`, ` a`

분기점: step 2 (3번째 sampled token). fix1과 동일 위치.

---

## Fix 3: baseline `--no-gpu-plan` 비교 (plan path 가설 검증)

### 가설
`generate.rs:4347` production decode loop는 OpenCL GPU plan path를 우선 시도:
```
if let Some(plan) = ... { model.execute_plan(plan, ...) } else { forward_into(...) }
```
- step 0: plan 미빌드 → forward_into (post-fix와 동일)
- step 1+: plan 빌드 후 execute_plan (post-fix의 forward_into와 numeric divergence)

ModelForward::step은 plan 미사용 → plan path가 step 2 분기의 원인일 가능성.

검증: baseline에 `--no-gpu-plan` flag 추가하여 plan path 비활성화 후 비교.

### baseline binary `--no-gpu-plan` 지원 확인
✅ baseline binary (`/data/local/tmp/generate_baseline`, 2026-05-17 15:06)는 `--no-gpu-plan` flag 지원.
helper text: "Disable GPU kernel plan for decode (fallback to forward_into every token)"

### G6 재측정 (32 token, --temperature 0 --top-k 1)

baseline `--no-gpu-plan` (deterministic, 3 run 동일):
```
The capital of France is Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into
```

post-fix2 (greedy, deterministic, 3 run 동일):
```
The capital of France is Paris. Paris is a very big city. It has about 2.2 million people. It is also a very old city. It was built in the
```

#### 첫 5 token 분기 분석
공통 prefix: ` Paris`, `.` (step 0~1)
- baseline `--no-gpu-plan` step 2: ` It`
- post-fix2 step 2: ` Paris`

→ **fix1/fix2와 완전 동일한 분기 위치 (step 2, ` It` vs ` Paris`)**.

### Verdict — G6 FAIL
**plan path 가설 반증**. baseline의 plan path를 비활성화해도 (`--no-gpu-plan`) baseline과 post-fix2는 step 2부터 다른 토큰 생성.

→ paradigm gap의 원인은 plan path가 **아님**. 양쪽 모두 `forward_into`만 사용하는데도 numeric 차이 발생.

### G7 — 측정 불필요
G6 FAIL이므로 TBT 비교 무의미. plan path 가설이 반증되어 ModelForward에 plan path 추가 (가상의 4-4.6) 작업은 paradigm gap 해소에 도움 안 됨.

### Determinism 확인 (3 run)
양쪽 모두 deterministic — baseline `--no-gpu-plan` 3회 동일, post-fix2 3회 동일.
→ noise 가능성 배제. 코드 path 자체에 차이 존재.

### 본 phase 종결 가능성
- ❌ plan path 가설: 반증됨 (Fix 3)
- ❌ prefill_workspace=None 가설: 반증됨 (Fix 1)
- ❌ KV initial_capacity 가설: 반증됨 (Fix 2)

3가지 fix 시도 모두 step 2 분기를 해소하지 못함. **본 phase는 종결 불가** — paradigm gap의 진짜 원인을 찾기 위해 추가 deep dive 필요.

### 남은 후보 (Fix 2의 후보 1, 2, 4, 5 — 모두 plan path 무관)
1. **ModelForward::step의 forward_into 인자 정밀 비교**: production fallback (`generate.rs:4388 부근` no-plan branch) vs ModelForward::step 인자 1:1 diff
2. **decode_workspace lifecycle**: production no-plan path가 step마다 reuse vs ModelForward eager alloc
3. **KV cache 인스턴스 동일성**: prefill용 KV cache vs decode용 KV cache가 같은 인스턴스인지 (ModelForward 내부에서 새로 생성하면 prefill 결과 손실)
4. **logits dump 디버그**: step 1 (input=`.`) 후 logits[:10]을 양쪽 path에서 dump하여 bit-level diff 위치 식별

### 측정 데이터 위치
- /tmp/g6_baseline_noplan.log (baseline --no-gpu-plan)
- /tmp/g6_postfix_greedy.log (post-fix2 --temperature 0 --top-k 1)

### 권장 다음 sprint
Fix 4 — `ModelForward::step` 의 `forward_into` 호출 인자를 production no-plan branch와 byte-level diff. 특히:
- `decode_workspace` ownership/reuse 정책
- `score_buf` / `decode_logits_buf` aliasing
- `cache_manager` 호출 위치 (forward 전/후)
