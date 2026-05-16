# Weight Swap 측정 종합 정리 — 2026-05-14
Pre-compact handoff. Deep dive 진입 전 모든 측정과 옵션을 한 곳에.

## 1. 실험 환경

### S25 (Galaxy S25, Adreno 8 Gen 4)
- **Device**: R3CY408S5SB
- **Model**: Qwen2.5-1.5B (28 layers, head_dim=128, kv_heads=2)
- **Primary**: F16 GGUF (~2.85 GB)
- **Secondary**: Q4_0 GGUF (~1.21 GB) → **rpcmem alias path 자동 활성** (qnn_oppkg + GGUF)
- **Backend**: qnn_oppkg (Android default after this session — `generate.rs:206` cfg 분기 추가)
- **Prompt**: 71 tokens, num-tokens=30 (대부분), 280 (K=1 async test)
- **Storage**: UFS, ~4 GB/s seq read
- **각 측정 사이**: cold reboot + 180s rest (thermal stabilization)

### Jetson (AGX Xavier)
- **SSH**: nvidia@165.132.107.73:4121
- **Model**: Llama 3.1 8B (32 layers, head_dim=128, kv_heads=8)
- **Primary**: F16 GGUF (~16 GB)
- **Secondary**: Q4_0 GGUF (~4.5 GB) → **alias path 없음** (CUDA backend, 일반 mmap 사용)
- **Backend**: cuda (cuda-embedded features)
- **Storage**: NVMe ~1.5 GB/s cold, RAM 2.9 GB/s memcpy
- **RAM**: 30 GiB total, 26.7 GB peak during swap
- **각 측정 사이**: drop_caches + 30s sleep (cold IO 일관)

## 2. 코드 변경 (이번 세션)

### a. Android default backend
`engine/src/bin/generate.rs:205-211`: Android target → `qnn_oppkg`, else → `cpu`

### b. Alias path batch prefault skip
`engine/src/models/weights/swap_executor.rs:485-491`:
```rust
let alias_path_prefault = matches!(secondary.as_ref(), SecondaryMmap::Rpcmem(_));
if !alias_path_prefault {
    secondary.prefault_layers(target_layers);  // 28 layer batch first-touch 우회
}
```
→ alias path에서 batch prefault stage 제거. dispatch 시점에 `try_alias_materialise` fallback이 per-layer ensure_layer_loaded 자연 호출.

### c. swap_executor.rs:474 prefault 호출 alias path 분기 (위 위치)

## 3. 시도한 옵션 (CLI flags + env)

### CLI flags
| Flag | 값 범위 | 설명 |
|---|---|---|
| `--swap-incremental-per-tick K` | 0~32 | K=0=SS, K≥1=incremental |
| `--swap-intra-forward` | bool | LISWAP-4 IF, forward layer-boundary hook |
| `--swap-async-dispatch` | bool | LISWAP-2 async dispatcher worker (default OFF) |
| `--swap-dynamic-k` | bool | DynamicKController 활성 (incremental 모드에서만) |
| `--force-swap-ratio` | 0.0~1.0 | 실험: 항상 1.0 (28/28 또는 32/32 layer) |
| `--swap-delay-tokens` | int | 0 (decode 시작 동시 swap) |
| `--secondary-gguf <path>` | path | Q4 GGUF (S25 alias 활성 / Jetson 일반 mmap) |
| `--tbt-log <path>` | path | per-token TBT JSONL dump |
| `--num-tokens` | 30 / 50 / 280 | 30 default, 50 Jetson, 280 K=1 swap 완료 보장 |

### Env variables
| Env | 효과 |
|---|---|
| `LLMRS_SKIP_EAGER_PREFAULT=1` | model load 시점 prefault 우회 |
| `LLMRS_SKIP_FINALIZE_BUDGET=1` | INV-167 graphFinalize 200ms budget 우회 |
| `LLMRS_IO_TRACE=1` | /proc/self/io read_bytes/write_bytes per phase |
| `LLMRS_RSS_TRACE=1` | VmRSS/RssFile/RssAnon/RssShmem per phase |
| `LLMRS_SWAP_DRAIN_DIAG=1` | release_worker queue depth max |
| `LLMRS_SUB_BATCH_PAUSE_DIAG=1` | sub-batch reactive wait 발생 추적 |
| `LLMRS_DYNAMIC_K_DIAG=1` | DynamicK controller calibration trace |

## 4. S25 측정 결과 (30 tokens, GGUF Q4 secondary + alias + prefault skip)

원본 데이터: `papers/eurosys2027/_workspace/experiment/swap_k_sweep_2026_05_14_s25/`
- `all_measurements.csv` (26 rows)
- `raw/*.stdout`, `raw/*.stderr`, `raw/*.tbt.jsonl`
- AUF baseline: `swap_k_sweep_2026_05_14_s25_baseline_auf/`

### 표
```
Mode       | TTFT  | Avg TBT  | Decode fwd | Total~  | 비고
-----------|-------|----------|------------|---------|------
Q4 only    |  903  |  30.52   | 12.15      |  1788   | floor (F16 dtype load)
F16 only   |  889  |  53.73   |  3.48      |  2447   | (forward 3.48 의심)
K=1        |  906  |  43.05   |  5.09      |  2153
K=2        |  897  |  48.71   |  3.68      |  2309
K=3        |  900  |  51.31   |  3.52      |  2388
K=4        |  897  |  74.75   |  4.84      |  3065   | anomaly (후반 tick latency spike)
K=5        |  894  |  48.89   |  3.46      |  2272
K=7        |  892  |  39.85   |  3.32      |  2048   | sweet spot (28÷7=4 ticks)
K=7_v3     |  892  |  40.91   |  3.18      |  2079   | retest with tbt-log
K=9        |  914  |  47.28   |  3.64      |  2284
K=11       |  890  |  46.63   |  2.81      |  2241
K=13       |  891  |  50.92   |  3.16      |  2367
K=14       |  898  |  55.98   |  4.03      |  2521
K=15       |  900  |  49.24   |  3.57      |  2328
K=17       |  903  |  60.23   |  4.29      |  2649
K=19       |  918  |  58.71   |  3.66      |  2620
K=21       |  901  |  51.43   |  3.42      |  2393
K=23       |  908  |  49.42   |  3.76      |  2342
K=25       |  892  |  62.15   |  4.16      |  2694   | worst
K=27       |  899  |  53.13   |  3.60      |  2439
SS         | 1465  |  25.51   |  3.60      |  2205   | TTFT 폭증
IF         |  912  |  55.77   |  3.29      |  1875   | 🏆 winner (LISWAP-4)
dynk_v3    |  906  |  40.78   |  3.18      |  2089   | DynamicK→K=1 fall
K1 sync 280|  891  |  28.06   |  3.88      |  9742   | swap 완료 보장
K1 async 280|  910 |  29.15   |  5.11      |  9784   | sub-batch wait로 sync와 동등
```

### K sweep 곡선 (S25)
```
Total (ms, 30 tok)
2700 │                            ●K=25
     │                  ●K=17  ●K=19
2400 │      ●K=3                       ●K=21    ●K=27
     │            ●K=9     ●K=13              ●K=23
     │   ●K=5                  ●K=15
     │                    ●K=11
2200 │  ●K=1                                     ●SS
2100 │
     │     ●K=7  ← global swap-mode min
2000 │
1900 │                                              ★IF (1875) winner
     │  ●●●●●●●●●●●●●●●●●●●●●●●●● Q4 floor (1788)
```

### S25 핵심 발견
1. **Alias path 활성**: Decode forward 17ms (AUF AOS) → 3ms (rpcmem alias) (-80%)
2. **K=7 sweet spot** (불완전한 이론적 정당화 — paper에서 측정 사실로만 인용)
3. **K=4 anomaly** (3065 ms = K=3/5의 30% 더 큼) — thermal accumulation 가설
4. **sub-batch wait가 async를 sync와 동등화** (K=1 sync 280 ≈ K=1 async 280)
5. **DynamicKController는 K=1만 선택** (수식상 K=7 도달 불가)
6. **IF만 진짜 hide** — token-level swap은 sub-batch wait로 무효, layer-level만 우회

## 5. Jetson 측정 결과 (50 tokens)

원본 데이터: `papers/eurosys2027/_workspace/experiment/swap_k_sweep_2026_05_14_jetson/`

### Sync (initial sweep)
```
Mode      | TTFT   | TBT     | Decode fwd | Total~
----------|--------|---------|------------|--------
Q4 only   |  2085  |  68.98  |  66.92     |  5466   floor
F16 only  |   866  | 157.69  | 155.62     |  8593
K=1       |   898  | 240.12  | 118.52     | 12664
K=2       |   896  | 225.28  |  91.21     | 11935
K=4       |   890  | 217.51  |  78.26     | 11548
K=6       |   893  | 216.41  |  76.16     | 11497
K=8       |   898  | 215.56  |  71.90     | 11460
K=12      |   902  | 211.90  |  70.45     | 11285
K=16      |   899  | 211.93  |  69.35     | 11283
K=24      |   898  | 212.03  |  69.26     | 11288
K=32      |   899  | 210.29  |  67.10     | 11203
SS        |  8936  |  70.31  |  68.34     | 12381   TTFT 폭증
IF        |   889  | 129.80  |  67.65     |  7249
dynk      |   888  | 124.92  |  94.30     |  7009   (was async + K=2)
```

### Async (re-measure for fair comparison)
```
Mode      | TTFT   | TBT     | Decode fwd | Total~
----------|--------|---------|------------|--------
K=1 async |   917  | 140.83  | 119.71     |  7798
K=2 async |   892  | 126.70  |  94.68     |  7100
K=4 async |   917  | 119.78  |  82.26     |  6786
K=8 async |   896  | 115.98  |  74.85     |  6580
K=16 async|   904  | 114.98  |  72.30     |  6539
K=32 async|   913  | 114.86  |  69.87     |  6541   🏆 winner
SS async  |  9032  |  73.76  |  69.88     | 12646   (SS는 async 무관)
```

### 메모리 spike 검증 (async sweep)
```
모든 측정에서:
  max_release_pending = 0     ← release queue 안 채움
  max_dispatcher_pending ≤ 1  ← in-flight ≤ 1 layer
  VmRSS peak = 26.7 GB (변동 ≤ 35 MB)
```
→ **3-layer safety net 정확 작동, 메모리 spike 없음** ✓

### Jetson 핵심 발견
1. **K-sweep plateau** (K=8~32, TBT ~115 ms async) — S25 sweet spot 패턴 없음
2. **Async 진짜 hide 효과 -45%** (TBT 210→115 ms)
3. **K=32 async가 winner** (6541 ms) — IF (7249) + dynk (7009)보다 작음
4. **sub-batch wait가 hide와 공존** (S25는 sync 동등 / Jetson은 effective hide)
5. **SS regression huge** (TTFT 8936-9032 ms, swap이 TTFT에 8000ms 추가)

## 6. S25 vs Jetson 비교

```
                       S25 Adreno Qwen 1.5B    Jetson CUDA Llama 8B
─────────────────────────────────────────────────────────────────
Backend                 qnn_oppkg + rpcmem      cuda (no alias)
Alias path              ✓ (forward -80%)        ✗
IO bandwidth            ~4 GB/s (UFS)           1.5 GB/s (NVMe cold)
Swap data               1.21 GB                 4.5 GB
Swap IO time            ~300 ms                 ~3000 ms
Forward Q4              3 ms/tok                67 ms/tok
Forward F16             3.5 ms/tok (의심)        156 ms/tok
F16/Q4 ratio            ~1.0                    ~2.3
swap/forward ratio      30/3 = 10×              250/65 = 4×
K-sweep shape           K=7 sharp dip           K=8~32 plateau
async hide effect       ~0 (sub-batch 무효)      -45% (sub-batch 공존)
winner mode             IF (1875, -13%)         K=32+async (6541, -22%)
spike safety            ✓                       ✓
```

## 7. 핵심 결론

1. **알고리즘은 platform 의존**: 같은 K-policy + async + dynamic-K가 dramatic하게 다른 곡선.
2. **IF가 유일한 hide는 Adreno 전용**: CUDA는 simple async도 효과.
3. **DynamicK 알고리즘이 sweet spot 못 찾음** (S25 K=7, Jetson K=32+ 모두).
4. **메모리 spike 안전망 정확 작동** — release_pending=0, dispatcher_pending≤1 hard constraint.
5. **TBT-forward 차이의 정체**: sampling + swap_blocking_residual. Adreno는 sampling 비중 큼 (small forward), Jetson은 forward dominant.

## 8. 미해결 의문 (Deep Dive 후보)

### Q1. S25 F16 baseline forward 3.48 ms 의심
- Q4 GGUF baseline (12.15 ms)와 같은 모델인데 4× 차이
- 측정 정의 / kernel path 충돌 가능성
- 재측정 + kernel breakdown 필요

### Q2. K=4 anomaly (3065 ms) 본질
- 인접 K=3 (2388), K=5 (2272)와 차이 700 ms
- 가설: 7 ticks 동안 thermal 누적 (후반 tick latency 76→224 ms)
- 재측정 (3회 반복) + thermal log 필요

### Q3. K=7 sweet spot의 이론적 정당성
- 단순 비용 모델로 설명 안 됨 (K=2, 4, 14 정확 분할도 K=7과 안 같음)
- 가능성: mmap warmup + sub-batch wait + thermal 복합 trade-off
- robust test: K=7 3회 반복 측정으로 noise vs robust 분리

### Q4. S25 25 ms TBT tail의 정체
- forward 3 ms인데 TBT 28 ms — 차이 22-25 ms
- sampling + KV grow + dispatch 합산?
- breakdown: top-k sort time, multinomial time, kv_cache mem_alloc time

### Q5. CUDA가 sub-batch wait + async hide 공존하는 이유
- S25 (Adreno): sub-batch wait이 async를 무효화
- Jetson (CUDA): sub-batch wait + async가 같이 작동 → -45% hide
- 코드 경로 차이 분석 필요 (release_worker queue 동작, dispatcher worker timing)

### Q6. DynamicKController 개선 가능?
- 현재 수식 `floor(fwd × 0.5 / drop)`이 너무 보수적
- drop_ms_per_layer를 cold (cold first-touch) vs warm (steady)로 구분?
- IF로 기본 default 변경?

### Q7. Jetson alias path 가능성
- 현재 CUDA backend는 일반 mmap만 → forward 67 ms
- CUDA에 rpcmem 같은 zero-copy alias 추가 가능?
  (cudaHostRegister + cudaHostGetDevicePointer)
- 가능하면 Jetson forward -80% → IF 효과 더 커질 것

### Q8. Paper §4.2 결론 정리
- "K-policy의 sweet spot은 platform 의존" main story
- "IF가 Adreno에서 유일 hide, CUDA에서는 K+async도 OK"
- 각 platform의 production winner 권고

## 9. 추가 측정 후보

- **a. K=4 재측정 ×3** — anomaly noise vs robust
- **b. K=7 재측정 ×3** — sweet spot robust 검증
- **c. S25 F16/Q4 baseline forward 정의 재검증**
- **d. Sampling cost 분리 측정** (`--num-tokens 5 --no-swap`, profile mode)
- **e. Jetson K=32 async 3회 반복** — Jetson plateau robust 확인
- **f. CUDA에 cudaHostRegister alias path 시도** (구현 + 측정)
- **g. dynamic-K hard_upper 풀어서 측정** (K=7 도달 가능?)
- **h. Llama 3.2 1B (S25)에서 같은 sweep — 모델별 sweet spot 비교**
- **i. Llama 3.2 3B (S25)에서 같은 sweep — 중간 크기 sweet spot**

## 10. 파일 경로 인덱스

### 코드 변경
- `engine/src/bin/generate.rs:205-211` Android default backend
- `engine/src/models/weights/swap_executor.rs:485-491` alias prefault skip

### 측정 데이터
- S25: `papers/eurosys2027/_workspace/experiment/swap_k_sweep_2026_05_14_s25/`
- S25 AUF baseline: `papers/eurosys2027/_workspace/experiment/swap_k_sweep_2026_05_14_s25_baseline_auf/`
- Jetson: `papers/eurosys2027/_workspace/experiment/swap_k_sweep_2026_05_14_jetson/`
- 종합: `papers/eurosys2027/_workspace/experiment/SWAP_HANDOFF_2026_05_14.md` (this file)

### Memory (memory system)
- `feedback_swap_sync_default.md` — production default = sync
- `feedback_no_memory_spike.md` — spike avoidance hard constraint
- `project_swap_overhead_opencl_complete_20260509.md` — HW serialize evidence
- `project_swap_io_paradox_handoff_20260514.md` — IO paradox
- `project_qcf_argus_experimental.md` — ARGUS framework

### 가이드
- `docs/48_swap_dynamic_k_guide.md` — Dynamic-K + safety net
- `.agent/todos/handoff_dynamic_k_2026_05_12.md` — dynamic-K production handoff
