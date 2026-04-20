# Qwen2.5-1.5B host-vs-device bench — Galaxy S25 (2026-04-20)

## 환경
Galaxy S25 (`R3CY408S5SB`, SD 8 Elite), 6T, `-p 128 -n 128 -ctk f16 -ctv f16`, greedy. llama.cpp `983df14` (NDK r29). llm.rs binary mtime 2026-04-20 13:37. 매 run 전 CPUSS/GPUSS max ≤ 38 °C thermal gate.

## 결과 (중앙값, n=3)

| 모델 | 엔진 | prefill tok/s | decode tok/s |
|------|------|--------------:|-------------:|
| F16  | llama.cpp | 108.49 ± 3.63 | 19.76 ± 0.03 |
| F16  | llm.rs    | **132.80** (132.93 ± 1.80) | **17.90** (17.80 ± 0.26) |
| Q4_0 | llama.cpp | 62.06 ± 12.28 | 32.35 ± 0.16 |
| Q4_0 | llm.rs    | **61.40** (61.93 ± 1.47) | **30.40** (30.03 ± 1.00) |

llama.cpp: `llama-bench -r 3` 단일 프로세스. llm.rs: 별개 프로세스 3회.

## 온도 (before / after max, °C)

| run | 모델·엔진 | before | after |
|---|---|---:|---:|
| r3(avg) | F16·llama.cpp | 36.1 | 72.2 |
| r3(avg) | Q4_0·llama.cpp | 36.1 | 65.3 |
| r1/r2/r3 | F16·llm.rs | 36.1/37.3/37.3 | 55.3/64.9/74.2 |
| r1/r2/r3 | Q4_0·llm.rs | 36.9/36.9/36.9 | 66.5/66.1/59.2 |

## llm.rs ↔ manager handshake
- Manager: `llm_manager --policy-script noop_policy.lua --transport tcp:127.0.0.1:38701` (중립 no-op Lua).
- **unix → tcp 대체**: Android 14 shell domain이 `/data/local/tmp/` (`shell_data_file`) unix bind를 `EACCES`로 차단 (tcp 정상). 기능 동일.
- Engine 로그: `[Resilience] Executor enabled — transport: tcp:127.0.0.1:38701` / `Capability sent to Manager` (매 run).
- Manager 로그 (q4_0 r1): `[TcpChannel] Listening on 127.0.0.1:38701` / `LuaPolicy initialized` / `Signal: ThermalAlert { Emergency, 94200 mC }`. **Directive 0건**, 엔진에 `Evict`/`Throttle`/`SwitchHw` 흔적 없음.

## 해석
F16: llm.rs prefill +22 %, decode −9 %. Q4_0: prefill 동률, decode −6 %. llama.cpp Q4_0 prefill stdev ±12.28은 run 내부 thermal 영향.

## Raw 산출물 (`/home/go/Workspace/llm_rs2/.agent/research/2026-04-20_qwen15b_host_vs_device_bench/`)
`results_summary.csv`, `llamacpp_{f16,q4_0}_run1.log`, `llmrs_{f16,q4_0}_run{1,2,3}.log`, `thermal_gate.sh`, `snapshot_temps.sh`, `run_llmrs_once.sh`, `noop_policy.lua`, `prompt_128.txt`.
