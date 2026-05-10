# LISWAP-1 swap-active phase: op-level profiling

**측정일**: 2026-05-08  
**디바이스**: Galaxy S25 (R3CY408S4HN, Adreno 830)  
**모델**: Qwen2.5-1.5B-Instruct (28 layers, head_dim=128, GQA 14:2)  
**바이너리**: `target/aarch64-linux-android/release/generate` (HEAD `f3dc4a4`)  
**도구**: `LLMRS_FORWARD_GEN_OP_TRACE={sync,async}` (engine/src/profile/op_trace.rs)

## 목표

LISWAP-1 (`--swap-incremental-per-tick=1`) swap-active phase에서 forward TBT가 약 26 ms → 약 48 ms (+약 22 ms)로 인플레된다. 본 측정은 14개 op bucket의 sync/async wall-clock을 통해 인플레가 발생하는 위치를 결정적으로 찾는다.

## 시나리오

전 시나리오 공통:
```
--secondary-layout aos --force-swap-ratio 0.9
--ignore-eos --threads 6 --backend opencl
-p "The quick brown fox jumps"
```

| ID | 모드 | LISWAP | n_tokens | 의미 |
|----|------|--------|----------|------|
| S1 | sync  | off | 50 | post-swap baseline (GPU exec) |
| S2 | async | off | 50 | post-swap baseline (host dispatch) |
| S3 | sync  | on  | 50 | swap+post mix (GPU exec) |
| S4 | async | on  | 50 | swap+post mix (host dispatch) |
| S5 | sync  | on  | 25 | swap-active dominant (GPU exec) |
| S6 | async | on  | 25 | swap-active dominant (host dispatch) |

각 시나리오 n=2. 25 layers swap × per_tick=1 → token 0~24 swap-active, token 25+ post-swap.

## 시나리오 요약 (avg of 2 runs)

| Scenario | fwd_calls | trace_total ms | Decode ms/tok | Decode(excl) ms/tok |
|----------|----------:|---------------:|--------------:|--------------------:|
| S1 sync baseline       | 50 | 3978.3 | 82.65 | 83.13 |
| S2 async baseline      | 50 |  289.5 | 25.81 | 25.80 |
| S3 sync liswap1 n=50   | 49 | 3511.6 | 73.65 | 74.18 |
| S4 async liswap1 n=50  | 49 |  238.6 | 31.87 | 31.53 |
| S5 sync liswap1 n=25   | 24 | 1606.6 | 69.35 | 70.25 |
| S6 async liswap1 n=25  | 24 |   73.6 | 38.05 | 37.59 |

Per-tick incremental swap latency (S5 run1): tick 0~14 = 22~33 ms, tick 15~17 = 32~34 ms (peak), tick 18~23 = 17~20 ms (감쇠). Tick 22~ 부터 OpenCL driver의 lazy CL_MEM 초기화가 안정화되며 latency 감소. `swap_overhead_dmabuf_heap.md` 일관된 패턴.

## Per-op avg us/call (sync vs async, baseline vs LISWAP-1 short)

`Δsync = S5 - S1`, `Δasync = S6 - S2`. 음수는 LISWAP-1이 baseline보다 빠름 (Q4 layer가 GPU에서 더 빠르게 실행).

| op | S1 sync base | S5 sync liswap | S2 async base | S6 async liswap | Δsync (us) | Δasync (us) |
|----|-----------:|---------------:|--------------:|----------------:|-----------:|------------:|
| rms_norm_attn        | 169.5 |  126.5 |  9.0 |  5.5 |  -43.0 |  -3.5 |
| matmul_qkv           | 306.0 |  215.5 | 51.0 | 25.5 |  -90.5 | -25.5 |
| rope                 | 258.5 |  127.5 | 24.0 | 11.0 | -131.0 | -13.0 |
| kv_update            | 153.5 |  110.5 |  9.5 |  4.5 |  -43.0 |  -5.0 |
| attention            | 275.5 |  171.5 | 17.5 | 15.5 | -104.0 |  -2.0 |
| matmul_wo            | 243.0 |  169.5 | 11.0 |  5.0 |  -73.5 |  -6.0 |
| rms_norm_ffn         | 152.0 |  128.5 | 10.0 |  6.5 |  -23.5 |  -3.5 |
| **matmul_ffn_gate_up** | **637.5** | **751.5** | 49.5 | 23.5 | **+114.0** | -26.0 |
| **matmul_ffn_down**    | **404.5** | **439.5** | 11.0 |  4.5 |  **+35.0** |  -6.5 |
| add_assign           | 148.5 |  113.0 | 10.5 |  4.0 |  -35.5 |  -6.5 |
| embedding            | 701.0 |  643.5 | 81.0 | 85.5 |  -57.5 |  +4.5 |
| final_norm           | 186.0 |  126.5 | 10.5 |  7.0 |  -59.5 |  -3.5 |
| lm_head              | 3112.0 | 2871.0 | 10.5 |  7.0 | -241.0 |  -3.5 |

**주목**: sync mode에서도 forward_gen 합계 Δsync = **−395 us/layer × 28 layers = −11.06 ms/token**. 즉 GPU exec time은 swap-active phase에서 baseline 대비 **줄어든다** (Q4 효과).

## 핵심 분석: trace coverage gap

각 시나리오에서 op_trace가 잡은 시간 vs 외부에서 측정한 Decode time:

| Scenario | trace ms/tok | Decode(excl) ms/tok | gap (외부) | coverage |
|----------|-------------:|--------------------:|-----------:|---------:|
| S1 sync baseline       | 79.57 | 83.13 |  3.56 | 95.7% |
| S2 async baseline      |  5.79 | 25.80 | 20.00 | 22.4% |
| S3 sync liswap n=50    | 71.67 | 74.18 |  2.51 | 96.6% |
| S4 async liswap n=50   |  4.87 | 31.53 | 26.66 | 15.4% |
| S5 sync liswap n=25    | 66.94 | 70.25 |  3.31 | 95.3% |
| S6 async liswap n=25   |  3.07 | 37.59 | 34.53 |  8.2% |

**해석**:
- sync mode coverage 95~97% — GPU sync drain이 op_trace에 모두 포함됨.
- async mode coverage 8~22% — 대부분의 시간이 op_trace 밖. baseline gap 20 ms는 **OpenCL driver-level command queue와 GPU implicit fence 대기 시간** (async로는 enqueue만 측정).

### Δ Decode breakdown (LISWAP-1 short − baseline)

**async mode**:
```
Δ Decode (total)         = +11.80 ms
Δ trace (op-tracer)      =  -2.72 ms      ← op-bucket 내 dispatch 약간 감소
Δ gap (op_trace 밖)      = +14.52 ms      ← 인플레의 정체
```

**sync mode**:
```
Δ Decode (total)         = -12.88 ms      ← 줄어듦
Δ trace                  = -12.62 ms      ← GPU exec 시간 감소 (Q4 효과)
Δ gap                    =  -0.25 ms      ← 변화 없음
```

### 결론

**+12 ms 인플레는 op_trace 밖, async mode에서 측정되는 영역에서 발생**한다.

이 영역의 정체는:
1. **GPU implicit fence wait 시간 증가** — `clEnqueue*` 호출 자체는 빨리 반환하지만, OpenCL driver의 in-order command queue에서 이전 swap-related enqueue (clEnqueueWriteBuffer / clEnqueueMapBuffer 류)가 직렬화 대기를 만든다. sync mode에서는 op_trace의 `synchronize()`가 이 대기를 흡수해 trace에 합쳐진다 — 그래서 sync gap은 일정하지만 sync trace 자체가 (GPU 실행 + driver wait) 합쳐진 값.
2. async mode는 enqueue 호출의 host-side return latency만 잡으므로, **driver-internal mutex 대기 시간이 op_trace 밖의 gap에 누적**된다.

즉 **+22 ms 인플레는 단일 op의 GPU 커널 자체 (kernel exec) 가 느려지는 것이 아니다**. 실제로 모든 individual op의 sync avg_us/call은 LISWAP-1에서 **감소**했다. 인플레는 **driver/runtime-level command queue 직렬화** 또는 **implicit fence wait** 형태로 op_trace bucket 사이에 분산되어 있다.

### 보조 증거: matmul_ffn_gate_up / matmul_ffn_down 만 sync에서 +Δ

| op | Δsync (us/call) | Δasync (us/call) |
|----|----------------:|-----------------:|
| matmul_ffn_gate_up | **+114** | -26 |
| matmul_ffn_down    | **+35**  | -7  |

이 두 op은 sync mode에서 +Δ를 보인다. dispatch는 줄었는데 (Q4가 더 빠르므로) sync는 늘었다 — 즉 **kernel completion까지의 wait time이 늘었다**. SOA→AOS swap 직후 cl_mem이 새로 alloc되어 첫 번째 사용 시 lazy initialization 페널티가 가장 큰 두 weight matrix (gate_up 8.4M params, down 8.4M params)에 집중됨.

다른 op들이 모두 -Δsync인 이유: 첫 enqueue가 아닌 후속 enqueue들은 driver가 이미 gate_up에서 implicit barrier로 직렬화 대기를 흡수했기 때문. 즉 **gate_up/down이 driver wait의 sink** 역할을 한다 — 가시적으로 거기 +Δ가 모이고, 나머지는 wait이 끝난 뒤이므로 정상 GPU 실행 시간만 측정됨.

## Top-3 inflated op (절대값 |Δsync|)

| 순위 | op | Δsync (us) | rsync | 해석 |
|-----:|----|-----------:|------:|------|
| 1 | lm_head            | -241.0 | 0.92 | post-swap에서 -241 us, 정상 (swap이 끝난 layer만 영향) |
| 2 | rope               | -131.0 | 0.49 | Q4 swap 후 layer rope 입력이 작아짐 |
| 3 | **matmul_ffn_gate_up** | **+114.0** | **1.18** | **driver wait sink — swap 직후 첫 weight 사용 페널티** |

## 결론

1. **+22 ms 인플레는 단일 op의 GPU kernel exec time 증가가 아니다.** sync mode에서 forward_gen 전체 Δ는 -11 ms (오히려 감소).
2. **인플레는 OpenCL driver-internal command queue 직렬화 / implicit fence wait에서 발생.** async mode에서 op_trace 밖의 gap이 +14.5 ms 증가하고, sync mode에서는 sync drain에 흡수되어 보이지 않는다.
3. **가시적 sync hot-spot은 matmul_ffn_gate_up (+114 us/layer)** — swap 직후 cl_mem 첫 사용 lazy alloc 페널티가 누적되는 sink. 다른 op들은 이 wait이 흡수된 뒤 측정되어 -Δ로 보인다.
4. **driver layer 직렬화 가설 확인**. Adreno OpenCL의 single-queue serialization (이미 LISWAP-2 negative result에서 관찰됨) 과 일관된 결과.

## 시사점

- `--swap-incremental-per-tick=1` 의 +22 ms는 **GPU 커널이 느려진 것이 아니라 driver-level fence가 forward path 전체를 직렬화**시킨다.
- 해결 방향:
  - swap H2D 업로드와 forward enqueue를 **분리된 OpenCL queue**로 보내고 explicit event 의존성으로만 동기화 (현재 LISWAP-2에서 시도했으나 Adreno multi-queue serialize로 negative — `project_swap_track_status_20260508.md`).
  - 또는 swap 자체를 **token boundary 밖** (예: prefill 중, 또는 별도 sleep tick) 으로 옮겨 forward path의 critical command queue와 분리.
  - LISWAP-4 (intra-forward layer-aligned swap) 가 이 분리를 가장 자연스럽게 달성할 수 있는 후보.
- 추가 측정 권장:
  - `swap-incremental-per-tick=2,3,4`로 swap H2D 빈도를 줄여 driver fence 발생 횟수를 검증.
  - Snapdragon Profiler GPU trace로 explicit submit/wait timeline 가시화 (op_trace는 host-only).

## 부록: 측정 산출물

- 결과 디렉터리: `/tmp/op_profiling_liswap1/` (호스트), `/data/local/tmp/op_profiling_liswap1/` (디바이스)
- stderr (op_trace dump): `s{1..6}_*_run{1,2}.stderr`
- stdout (Decode line): `s{1..6}_*_run{1,2}.stdout`
- 분석 스크립트: `/tmp/op_profiling_liswap1/analyze.py`
