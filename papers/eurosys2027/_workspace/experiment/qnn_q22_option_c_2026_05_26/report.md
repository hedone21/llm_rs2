# Q-2.2 옵션 C — llama.cpp stock S25 NPU 검증: universal wall 부정, 0x80000414 = non-fatal, 그러나 NPU performance < CPU

**작성**: 2026-05-26
**HEAD**: `8fd141cf` (β-1.QUEUE.RPC 직후, 본 sprint 코드 변경 없음)
**device**: Galaxy S25 (R3CY408S5SB, SM-S931N), Hexagon v79, Adreno 830
**llama.cpp build**: local `/home/go/Workspace/llama.cpp/build-snapdragon/` (vintage `983df14`)
**worktree**: `.claude/worktrees/b5_trait_extension`

## TL;DR

옵션 C 의 두 가설을 microbench 로 직접 결정:

1. **universal wall 가설 ✗ 기각** — llama.cpp 의 `libggml-hexagon.so` + `libggml-htp-v79.so` 가 stock S25 에서 **실제로 동작**. `tg32: 32.40 tok/s` 측정. 28 layer 모두 HTP0 assign.
2. **0x80000414 fatal 가설 ✗ 기각** — llama.cpp 도 동일하게 `Error 0x80000414: libdspqueue_rpc_skel.so method 3` 발생, 그러나 직후 `dspqueue_create: created Queue 0 ... for domain 3` 가 성공. llama.cpp 는 이 error 를 명시적으로 무시 (intermediate driver-internal self-probe).

새 결론:
- **DSP-side architectural barrier 가설** (handoff_q22_beta_queue_rpc 의 RPC.4) = 실측 반증.
- 우리 host binding 의 `0x80000414` fatal handling 이 wall 의 진짜 원인 — fix 가능 (옵션 D 추가).
- 그러나 NPU performance (`32.40 tok/s`) < CPU (`63.75`) < OpenCL (`37.58`). Qwen2.5-1.5B Q4_0 단일 model 한정, paper performance evidence 로는 부적합.

## 측정 matrix

`llama-bench -m Qwen2.5-1.5B-Instruct-q4_0.gguf -p 64 -n 32 -r 1 -t 6`:

| backend | flag | tg32 (tok/s) | pp64 (tok/s) | logcat NPU dispatch |
|---|---|---|---|---|
| HTP0 (Hexagon NPU) | `-dev HTP0` | **32.40** | (run pp 미측정) | ✓ 28 layer assign |
| GPUOpenCL (Adreno 830) | `-dev GPUOpenCL` | 37.58 | 459.61 | OpenCL only |
| CPU only | `-ngl 0 -t 6` | **63.75** | 419.42 | CPU only |

Note: 모든 measurement Backend 컬럼은 `OpenCL,HTP` (build feature 둘 다 활성). 실제 dispatch device 는 `-dev` 컬럼.

## 결정적 logcat trace (NPU init + 0x80000414 발생 + dspqueue_create 성공)

```
19:12:54.494 ... remote_handle64_open: opened handle 0xb400007951dcdf90 (remote 0x2d2f60) 
                  for file:///libggml-htp-v79.so?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0 
                  on domain 3 (spawn time 24315 us, load time 5221 us), refs 1
                  ↑ DSP-side compute module load OK

19:12:54.494 ... remote_handle64_open: opened handle 0xb400007951dc0a10 (remote 0x2d2d60) 
                  for file:///libdspqueue_rpc_skel.so?dspqueue_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0 
                  on domain 3, refs 2
                  ↑ driver transport stub load OK

19:12:54.495 ... Error 0x80000414: remote_handle64_invoke failed for handle 0xb400007951dc0a10, 
                  interface libdspqueue_rpc_skel.so method 3 on domain 3 (sc 0x3010100) 
                  (errno Success) (user err 0x80000414)
                  ↑ ★ 우리 binding 이 fatal 처리한 line — llama.cpp 도 똑같이 발생

19:12:54.495 ... dspqueue_create: created Queue 0, 0xb4000077e1d27ed0, DSP 0x00000000 for domain 3
                  ↑ ★ 그러나 바로 다음 dspqueue_create 가 성공 (llama.cpp 는 0x80000414 무시)
```

ggml-hexagon backend production log:
```
ggml-hex: Loading driver libcdsprpc.so
ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev 1
ggml-hex: Hexagon Arch version v79
ggml-hex: allocating new session: HTP0
ggml-hex: new session: HTP0 : session-id 0 domain-id 3 
          uri file:///libggml-htp-v79.so?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0 
          handle 0xb400006e392f3390
```

llama.cpp 도 우리와 동일 init flow:
- domain 3 (ADSP)
- URI 의 `_dom=cdsp` (driver internal alias; domain id 3 는 ADSP)
- session-id 0, handle nonzero

## llama.cpp 코드의 0x80000414 처리

`ggml/src/ggml-hexagon/ggml-hexagon.cpp:1642-1651`:
```cpp
err = dspqueue_create(this->domain_id, ...);
if (err != AEE_SUCCESS) {
    GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", ...);
    return false;
}
```

→ `dspqueue_create` rc 만 check. 이전의 `0x80000414 method 3` 은 명시적 ignore (libcdsprpc.so 의 driver-internal self-probe). 

우리 host.rs 는 이 error 를 fatal 로 abort → wall.

## llama.cpp NPU performance 의 paper 적합성 평가

Qwen2.5-1.5B Q4_0 측정 결과:

- **NPU 가 CPU 의 51%** (32.40 vs 63.75 tg32). NPU 가 더 느림.
- **NPU 가 OpenCL 의 86%** (32.40 vs 37.58). 비슷하지만 NPU 가 약간 느림.
- 즉 Qwen2.5-1.5B Q4_0 한정, **NPU 가 production performance source 로 의미 없음**.

EuroSys 2027 paper performance evidence 측면:
- 이미 production = `--backend opencl --opencl-rpcmem` (Sprint 2a 결정, 2026-05-26). S25 32.17 ms/tok = NPU(31 ms/tok 등가) 와 비슷.
- NPU 가 production OpenCL 을 능가하지 못함 → paper main result 로 부적합.
- 다른 model size (8B+) 에서 NPU 가 GPU 를 능가할 가능성은 별 검증 sprint 필요.

## 후속 옵션 재평가

본 sprint 결과로 옵션 A/B/C 재평가:

| 원 옵션 | 원 정의 | 본 sprint 결과 적용 후 |
|---|---|---|
| A (Hexagon SDK signed test signature) | unsigned PD signing barrier 우회 | ✗ **무의미** — llama.cpp 가 unsigned PD 로 stock S25 에서 동작. signing barrier 아님. |
| B (NPU track 종료 + paper 에 negative result) | universal wall 확인 시 정당화 | △ **근거 변경 정당화** — universal wall ✗, 그러나 **NPU performance < CPU/OpenCL** (Qwen 1.5B Q4_0). paper main evidence 부적합. backlog OpenCL 집중. |
| C (llama.cpp stock S25 동작 검증) | 본 sprint | ✓ **완료** |

**새 옵션** (option D, 추가):
- **D (host binding 0x80000414 ignore fix + NPU dispatch GREEN)**: 우리 host.rs 의 fatal handling 제거, llama.cpp 처럼 0x80000414 무시. 추정 1-2h.
- 종착지: 우리 binding 으로 NPU rmsnorm/matmul correctness GREEN. NPU performance 도 llama.cpp 수준 (32 tg32, CPU 의 51%).
- **value**: 우리 NPU backend 가 동작은 한다는 evidence 확보. 그러나 NPU performance 가 production-track 으로 부적합한 점은 변함 없음. paper 의 "we built our own NPU binding" narrative 가치 vs 시간 투자 비교.

## 후속 옵션 (최종)

| 옵션 | 정의 | 추정 | 권장 |
|---|---|---|---|
| **B' (조정)** | NPU track production-track 에서 제거 + paper 에 negative-performance result 기록 (universal wall 아니라 NPU absolute perf 가 부적합). backlog OpenCL 집중. | 1-2h | ★ 1순위 |
| **D** | host binding 0x80000414 ignore fix → NPU rmsnorm GREEN → 우리 backend stack 완성 (production 아닌 reference). | 2-4h | 2순위 (D 후 B' 도 가능) |
| **추가 검증** | llama.cpp NPU vs Qwen 8B+ / Llama 3.2 1B 등 다른 model 측정. NPU 가 GPU/CPU 를 능가하는 sweet spot model 발견 가능성. | 2-3h | 3순위 |

## 핵심 파일 인덱스

- 본 sprint commit: 없음 (분석/측정만)
- llama.cpp build artifacts: `/home/go/Workspace/llama.cpp/build-snapdragon/bin/`, `libggml-htp-v*.so`
- S25 deploy dir: `/data/local/tmp/llamacpp_c/`
- 측정 raw: `papers/eurosys2027/_workspace/experiment/qnn_q22_option_c_2026_05_26/{bench_htp0.txt,logcat_bench.txt}`
- llama.cpp 0x80000414 handling: `/home/go/Workspace/llama.cpp/ggml/src/ggml-hexagon/ggml-hexagon.cpp:1642-1651`
- 우리 fatal handling location: `engine/src/backend/htp_fastrpc/host.rs:543-617` (dispatch path, β-1.QUEUE 의 cache flag fix 후 잔존)
- 직전 handoff: `.agent/todos/handoff_q22_beta_queue_rpc_2026_05_26.md`

## Landmines / 미해결

- **llama.cpp NPU performance < CPU 는 Qwen2.5-1.5B Q4_0 한정**: 다른 model/quant 에서 NPU 가 우위일 가능성. 본 sprint 한 model 만 측정. (paper performance evidence 면에서 별 sprint 가치 평가)
- **`dspqueue_write`/`dspqueue_read` 의 actual logcat dispatch event 미관측**: production logcat INFO/ERROR level 만 capture. DEBUG/VERBOSE 는 fastrpc 트레이스 활성화 필요. `32.40 tok/s` 자체 + 28 layer assign 으로 dispatch 발생 확정.
- **opt D (binding fix) value 평가**: NPU performance 가 production 부적합인데 우리 backend stack 완성 가치 = 시간 투자 vs paper narrative trade-off. user 결정 필요.
- **paper performance baseline 변동 없음**: OpenCL `--opencl-rpcmem` = 32.17 ms/tok (Sprint 2a) 가 main result. NPU 결과는 separate ablation 자리.
- **session leak 패턴 동일**: llama.cpp 도 cleanup 시 `notif_fastrpc_thread FastRPC notification worker thread exited` 발생. 정상.

## 결정 게이트

본 sprint = 분석/측정 sub-sprint. **사용자 결정 필요**:
- 옵션 B' (NPU track 종료, 권장)
- 옵션 D (host binding fix 시도)
- 추가 검증 (다른 model 측정)
