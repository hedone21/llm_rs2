Galaxy S25 (R3CY408S5SB) weight swap measurement raw data
2026-05-14

Model: Qwen2.5-1.5B, F16 primary + Q4_0 AOS AUF secondary
Backend: qnn_oppkg
Prompt: /data/local/tmp/argus/layer_count_5p/prompt.txt (71 tokens)
Cmd flags: --backend qnn_oppkg --force-swap-ratio 1.0 --swap-delay-tokens 0
           [+ per-test: --swap-incremental-per-tick K, --secondary-gguf, --num-tokens]
Env: LLMRS_SKIP_EAGER_PREFAULT=1 LLMRS_SKIP_FINALIZE_BUDGET=1 (대부분)
     LLMRS_IO_TRACE=1 LLMRS_RSS_TRACE=1

File index — see summary.csv

[Page cache 거동 검증 (prefault ON, 50 tok, K=5)]
  w1   warm 1차 (간격 둠)
  c1   cold reboot
  w2   warm 2차 즉시 재실행
  w3   opencl backend, no secondary (50 tok decode 30)
  w4   dummy 9G dd read 후

[Cold reboot K sweep (no prefault, 50 tok)]
  k1   K=1
  k2   K=2
  k3   K=3 (Prefill 폭증 anomaly, noise 추정)
  k7   K=7 (Prefill 폭증 anomaly, noise 추정)
  np2  K=5
  k14  K=14 (30 tok)
  k28b K=28 (30 tok)

[30 tok dummy-evict K sweep]
  sw1  K=1
  sw2  K=2
  sw3  K=3
  sw5  K=5
  sw7  K=7
  k28  K=28 (cache hit으로 빠르게 나옴)

[Baseline 30 tok]
  ss   single-shot swap (--swap-incremental-per-tick 0)
  q4b  Q4 only (no swap, secondary 없음)
  f16b F16 only (no swap)

[Failed]
  np   no prefault 첫 시도 — budget abort
  q4   budget bypass flag 누락 (성공 ver = q4b)
  f16  budget bypass flag 누락 (성공 ver = f16b)
  k3b  device USB 끊김으로 측정 실패

Common build: aarch64-linux-android, features=opencl,vulkan,qnn
Commit base: fefaf9e1 + io_trace 추가 + LLMRS_SKIP_EAGER_PREFAULT 패치 + budget bypass
