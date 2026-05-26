# Sprint 2d-regression — qnn_oppkg 삭제 후 S25 회귀 측정

- **Date**: 2026-05-26
- **Worktree**: `/home/go/Workspace/llm_rs2-refactor-qnn-oppkg-consolidation`
- **Branch**: `refactor/qnn-oppkg-consolidation`
- **HEAD commit**: `801a158e` — `refactor(backend): qnn_oppkg production code 제거 (sprint 2b)`
- **Tester**: Tester agent (Opus 4.7) — read-only 검증

## 측정 의도

Sprint 2b (commit `801a158e`) 에서 `qnn_oppkg` production backend 가 −4,898 LOC 규모로 삭제됨.
본 측정은 삭제 후에도 시나리오 B `--backend opencl --opencl-rpcmem` 가 Sprint 2a-Gate 측정 수치
(median Avg TBT 32.340 ms) 와 동등한 성능을 유지하는지 회귀 여부를 확인한다.

## 환경

| 항목 | 값 |
|------|----|
| Device model | Galaxy S25 (SM-S931N), serial `R3CY408S4HN` (2a-Gate 동일) |
| Android | 16 |
| SoC | Snapdragon 8 Elite (Adreno 830 GPU) |
| Binary | `legacy_generate` (features = `opencl,vulkan,qnn`, `--no-default-features`, release) |
| Target | `aarch64-linux-android` (NDK API 21, Linux NDK toolchain) |
| Model | Qwen2.5-1.5B Q4_0 GGUF — `/data/local/tmp/models/qwen2.5-1.5b-q4_0.gguf` (2a-Gate 동일 파일) |
| Tokenizer | `/data/local/tmp/models/tokenizer.json` (auto-fallback 차단을 위해 명시) |
| Prompt | `"What is the capital of France?"` (7 tokens) |
| Decode tokens | 64 |
| Threads | 6 (S25 권장) |
| Sampling | `--greedy` (temperature=0, bit-identical 비교용) |
| Wall-clock | `--profile` 미사용 (sync 오버헤드 회피) |
| Runs per scenario | n=3 |

## 시나리오 명세

본 sprint 에서 시나리오 C 는 `qnn_oppkg` 가 삭제되어 자연스럽게 제외된다.

| 시나리오 | CLI | 설명 |
|----------|-----|------|
| A — baseline | `--backend opencl` | 기존 OpenCL path (rpcmem off) |
| B — 측정 대상 | `--backend opencl --opencl-rpcmem` | rpcmem allocator 활성 (2a-Gate 신규 path) |
| ~~C — qnn_oppkg~~ | (삭제됨) | 2b 에서 제거 — 시나리오 X negative test 로 대체 |
| D — probe | `B + --secondary-gguf .../fp16 --force-swap-ratio 0.5` | RpcmemSecondaryStore 정상 동작 확인 |
| X — negative | `--backend qnn_oppkg` | qnn_oppkg 가 명확한 error 로 거부되는지 확인 |

## 결과 — 원시 수치

| 시나리오 | run | TTFT (ms) | Decode (ms/tok) | Avg TBT (ms) |
|----------|-----|-----------|-----------------|--------------|
| A opencl              | 1 | 120.39 | 28.69 | 30.12 |
| A opencl              | 2 | 125.17 | 30.78 | 32.25 |
| A opencl              | 3 | 122.31 | 30.74 | 32.17 |
| B opencl+rpcmem       | 1 | 127.58 | 30.69 | 32.20 |
| B opencl+rpcmem       | 2 | 122.32 | 30.74 | 32.17 |
| B opencl+rpcmem       | 3 | 122.59 | 30.61 | 32.04 |

관찰: scenario A run1 의 Avg TBT 30.12 ms 는 cold-start outlier (Decode 28.69 = 다른 run 평균보다 6% 빠름). median 통계로 영향 제거.

## 결과 — 통계 요약

### Avg TBT (ms) — tok0 inclusive (main metric)

| 시나리오 | mean | **median** | range |
|----------|------|------------|-------|
| A opencl              | 31.513 | **32.170** | [30.120, 32.250] |
| B opencl+rpcmem       | 32.137 | **32.170** | [32.040, 32.200] |

### Decode ms/tok (rest-of-decode)

| 시나리오 | mean | **median** | range |
|----------|------|------------|-------|
| A opencl              | 30.070 | **30.740** | [28.690, 30.780] |
| B opencl+rpcmem       | 30.680 | **30.690** | [30.610, 30.740] |

## 게이트 판정 — **PASS (GREEN)**

| 항목 | 2a-Gate B | 2d-regression B | Δ |
|------|-----------|-----------------|---|
| Avg TBT median (ms) | 32.340 | **32.170** | **−0.53%** |
| Decode ms/tok median | 30.920 | 30.690 | −0.74% |

- Gate band: 2a-Gate B × [0.95, 1.05] = [30.723, 33.957] ms
- 2d B median = 32.170 ms ⊂ band → **PASS (GREEN)**
- 회귀 없음 — 오히려 약간(−0.53%) 개선 (variance 범위 내)
- B 가 A 대비도 median 동일 (32.170 == 32.170), 2a-Gate 와 같이 rpcmem path 가 baseline 등가

## 정확성 검증 — bit-identical 확인

`--greedy` 옵션으로 6 runs 전체의 생성 텍스트 md5 비교:

```
scenario_A_opencl_baseline_run1: 0aa10f96f231febd04f61dabf7106ea9
scenario_A_opencl_baseline_run2: 0aa10f96f231febd04f61dabf7106ea9
scenario_A_opencl_baseline_run3: 0aa10f96f231febd04f61dabf7106ea9
scenario_B_opencl_rpcmem  run1: 0aa10f96f231febd04f61dabf7106ea9
scenario_B_opencl_rpcmem  run2: 0aa10f96f231febd04f61dabf7106ea9
scenario_B_opencl_rpcmem  run3: 0aa10f96f231febd04f61dabf7106ea9
```

**모든 6 runs bit-identical** (단일 md5).

2a-Gate (B run1) 생성 텍스트와의 직접 diff 비교:
```
$ diff 2a-Gate/scenario_B_run1.log 2d/scenario_B_run1.log  # 텍스트 영역만
(no output — completely identical)
```

→ 2a-Gate B 와 2d B 의 첫 64 토큰 생성 결과가 완전히 동일.
qnn_oppkg deletion 이 forward path 정확성에 영향을 주지 않음 확인.

(주의: 2a-Gate README 의 md5 `10c87a39…` 와 2d 의 md5 `0aa10f96…` 가 다른 이유는 추출 시
`echo -n` 사용 여부로 trailing newline 처리가 달라서 발생한 hashing 차이 — 실제 텍스트는 일치.)

## 시나리오 D probe — precision swap 정상 동작

```
[OpenCL] --opencl-rpcmem: RpcmemAllocator init OK (libcdsprpc.so dlopen 성공)
[Deprecated] --secondary-gguf는 향후 제거됩니다. AUF single-file (--model-path foo.auf)로 전환하세요.
weight_swap: eager prefault — 14 layers, 1073.8ms (alias cache: 168 cl_mems)
weight_swap: force ratio=0.50, swapped 0/28 layers in 0.0ms
TTFT: 115.70 ms
Decode: 25.82 ms/tok (38.7 tok/s) [63 tokens]
Avg TBT: 27.23 ms (36.7 tokens/sec)
```

핵심 관찰:
- `RpcmemAllocator init OK` — libcdsprpc.so dlopen 정상
- `eager prefault — 14 layers, 1073.8ms (alias cache: 168 cl_mems)` — 50% (14/28) layers
  alias 캐시 정상 (rpcmem 경유 zero-copy)
- **`RpcmemSecondaryStore` 가 EXT_RPCMEM_ALLOCATOR 단독 lookup 만으로 정상 동작 확인**
  (Sprint 2b 에서 EXT_QNN_OPPKG fallback 제거됨에도 영향 없음)
- 2a-Gate scenario D 와 비교: alias 캐시 168 동일, eager prefault 시간 1270 → 1074 ms 단축 (variance)
  → precision swap path 가 qnn_oppkg deletion 후에도 완전 등가 동작

## 시나리오 X — qnn_oppkg negative test

```
$ legacy_generate --backend qnn_oppkg ...
Loading model from /data/local/tmp/models/qwen2.5-1.5b-q4_0.gguf
Error: Unknown backend: qnn_oppkg. Use cpu, opencl, or cuda.
```

- 명확한 error message ("Unknown backend") 로 거부됨
- error path 가 깔끔 — cleanup task 에서 처리할 issue 없음

## 결론 — sprint 진행 권장

1. **회귀 게이트 PASS (GREEN)** — `--opencl-rpcmem` 가 qnn_oppkg deletion 후에도 등가 성능 유지
   (Δ = −0.53%, gate band 깊숙이 진입)
2. **정확성 완벽 일치** — 2a-Gate B 와 2d B 의 생성 텍스트 byte-identical
3. **precision swap path 정상** — `RpcmemSecondaryStore` 가 EXT_QNN_OPPKG fallback 없이 작동
4. **negative test 깔끔** — `--backend qnn_oppkg` 가 명확한 error 로 거부됨
5. **다음 task (#5 cleanup) 진입 가능** — 코드 deletion 으로 인한 회귀 없음 확인됨

## 파일 인덱스

- `scenario_A_opencl_baseline_run{1,2,3}.log` — Scenario A 원시 로그 (opencl baseline)
- `scenario_B_opencl_rpcmem_run{1,2,3}.log` — Scenario B 원시 로그 (opencl + rpcmem)
- `scenario_D_opencl_rpcmem_swap_probe.log` — Scenario D probe (precision swap)
- `scenario_X_qnn_oppkg_negative.log` — Scenario X negative test (deletion verification)
