# Handoff: Q-2.2 옵션 C 종결 — universal wall 부정, NPU 동작 확인, NPU < CPU performance → 옵션 B' / D / 추가 검증 선택

**작성**: 2026-05-26
**HEAD**: `8fd141cf` (β-1.QUEUE.RPC 직후, 본 sprint 코드 변경 없음)
**worktree**: `.claude/worktrees/b5_trait_extension`
**다음 세션 진입 문장**: **"Q-2.2 NPU track 결정 — 옵션 B' (NPU track 종료 + paper negative-performance) / D (host binding 0x80000414 ignore fix) / 추가 검증 (다른 model NPU sweep) 중 선택"**

---

## TL;DR

옵션 C microbench 로 두 가설 직접 결정:
1. **universal wall ✗ 기각** — llama.cpp stock S25 에서 NPU 실측 동작. `tg32: 32.40 tok/s`, 28 layer 모두 HTP0 assign.
2. **0x80000414 fatal ✗ 기각** — llama.cpp 도 같은 error 발생, 그러나 직후 `dspqueue_create` 가 success → driver-internal probe, 우리만 fatal 처리.

새 발견:
- **NPU performance < CPU**: HTP0 32.40 / CPU 63.75 / OpenCL 37.58 tg32 (Qwen2.5-1.5B Q4_0).
- NPU 가 paper performance evidence 로 부적합 (해당 model 한정).
- **옵션 A (signing barrier) 무의미해짐** — llama.cpp 가 unsigned PD 로 동작.

본 sprint scope = host scope 외 wall 가설 검증. 코드 변경 없음. 결정 게이트 도달.

---

## 진행 상태

| Task | 상태 | 결과 |
|---|---|---|
| C.1 device + GGUF path 확인 | ✓ | R3CY408S5SB, Qwen2.5-1.5B-Instruct-q4_0.gguf 준비 |
| C.2 llama.cpp Snapdragon binary push | ✓ | `/data/local/tmp/llamacpp_c/`, 9 file 117 MB |
| C.3 llama-bench hexagon backend dry-run | ✓ | `HTP0: Hexagon (2048 MiB free)` 등록. URI v79 (SM8750 match). |
| C.4 llama-bench HTP0/GPUOpenCL/CPU 실측 + logcat | ✓ | NPU dispatch 28 layer, tg32 측정 matrix 완성 |
| C.5 wall universality 판정 | ✓ | universal wall ✗ + 0x80000414 = non-fatal driver probe |
| C.6 report + commit + handoff | ✓ | `papers/.../qnn_q22_option_c_2026_05_26/report.md` |

---

## 측정 matrix

| backend | tg32 (tok/s) | pp64 (tok/s) | vs HTP0 |
|---|---|---|---|
| HTP0 (NPU) | **32.40** | (미측정) | baseline |
| GPUOpenCL | 37.58 | 459.61 | +16% |
| CPU (-ngl 0) | **63.75** | 419.42 | +97% |

→ Qwen2.5-1.5B Q4_0 한정 **NPU 가 absolute performance 최하위**. paper main result 부적합.

---

## 결정적 logcat 증거

```
19:12:54.494 remote_handle64_open: libggml-htp-v79.so on domain 3 (refs 1)
19:12:54.494 remote_handle64_open: libdspqueue_rpc_skel.so on domain 3 (refs 2)
19:12:54.495 ★ Error 0x80000414: libdspqueue_rpc_skel.so method 3 (sc 0x3010100) ★
19:12:54.495 ★ dspqueue_create: created Queue 0 ... DSP 0x00000000 for domain 3 ★
```

→ 0x80000414 발생 후 dspqueue_create 가 즉시 성공. llama.cpp 의 `ggml-hexagon.cpp:1642-1651` 는 `dspqueue_create` rc 만 check.

---

## 다음 작업: 사용자 결정 게이트 (3 옵션)

### 옵션 B' (조정, ★ 1순위)
- **NPU track production-track 에서 제거**. universal wall 아니라 **NPU absolute performance 부적합** 이유로.
- paper 에 negative-performance result 기록. backlog OpenCL `--opencl-rpcmem` 집중.
- 추정 1-2h (paper writeup + backlog 정리).
- 위임: PM + Researcher.

### 옵션 D (2순위)
- 우리 host binding `0x80000414` ignore fix → NPU dispatch GREEN → rmsnorm/matmul correctness PASS.
- 종착지: 우리 NPU backend 가 동작은 함. performance 는 llama.cpp 수준 (32 tg32).
- 추정 2-4h. 위임: senior-implementer.
- value 평가: paper 의 "self-built NPU binding" narrative vs 시간 투자.

### 추가 검증 (3순위)
- llama.cpp NPU vs Qwen 8B+ / Llama 3.2 1B / 다른 quant 측정 sweep.
- NPU 가 GPU/CPU 를 능가하는 model size sweet spot 발견 가능성.
- 추정 2-3h. 위임: senior-implementer.

### 검증 게이트 (옵션 D)
- microbench_htp_rmsnorm dispatch success rc=0 + correctness `max_abs_err < 1e-3`
- microbench_htp_matmul 동일

---

## Landmines / 미해결

- **NPU < CPU 는 Qwen2.5-1.5B Q4_0 한정**: 다른 model 에서 NPU 가 우위일 가능성. 본 sprint single-model. 옵션 추가검증 으로 covered.
- **`dspqueue_write` actual dispatch event logcat 미관측**: production INFO level 한계. `32.40 tok/s` + 28 layer assign 으로 dispatch 발생 확정.
- **옵션 A (signed test signature) 영구 무의미**: llama.cpp unsigned PD 동작. signing barrier 가설 반증.
- **paper main result 변동 없음**: Sprint 2a 의 OpenCL `--opencl-rpcmem` 32.17 ms/tok 가 production. NPU 는 separate ablation 자리.
- **session leak 동일**: llama.cpp 도 cleanup 시 worker thread exit. 정상.
- **handoff_α status 변동**: "NPU HVX vector unit 0회 실행" → llama.cpp 측정으로는 실행 됨 (32 tg32 measurable). 우리 binding 한정 0회. 옵션 D fix 시 변동 가능.

---

## 핵심 파일 인덱스

- 본 sprint commit: 없음 (분석/측정만)
- llama.cpp build: `/home/go/Workspace/llama.cpp/build-snapdragon/bin/`
- S25 deploy: `/data/local/tmp/llamacpp_c/`
- 측정 raw: `papers/eurosys2027/_workspace/experiment/qnn_q22_option_c_2026_05_26/{bench_htp0.txt,logcat_bench.txt}`
- llama.cpp 0x80000414 handling: `/home/go/Workspace/llama.cpp/ggml/src/ggml-hexagon/ggml-hexagon.cpp:1642-1651`
- 우리 fatal handling: `engine/src/backend/htp_fastrpc/host.rs` (dispatch path)
- 직전 handoff: `.agent/todos/handoff_q22_beta_queue_rpc_2026_05_26.md`
- 본 sprint report: `papers/eurosys2027/_workspace/experiment/qnn_q22_option_c_2026_05_26/report.md`
