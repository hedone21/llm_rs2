# Production op-level phase predictability — phase-aware async swap precondition

**Date**: 2026-05-10
**Device**: Galaxy S25 (Adreno 830, 6T)
**Model**: Qwen2.5-1.5B Q4_0
**Backend**: opencl (production)
**선행**: `swap_overhead_negative_2026-05-08.md` (9-track all negative), `qnn_phase_r_summary.md` §5 (Phase R Scenario B GREEN, 1.04× of max)

---

## 0. 동기

9-track 음성 결과는 **production OpenCL이 phase 무시하고 swap 명령을 driver queue에 그냥 쏟아부어** driver FIFO에 묻혀 serialize된 결과로 해석. Phase R microbench는 cache-fit chain에 memcpy를 align했을 때 1.04× of max (near-perfect parallel) 달성.

→ **production decode op-level wall-clock이 충분히 deterministic하면, op_trace boundary를 trigger 삼아 chunk dispatcher를 cache-fit phase에만 enqueue 하는 전략이 viable.**

본 측정은 그 precondition (phase predictability) 검증.

---

## 1. 측정 방법

```bash
export LLMRS_FORWARD_GEN_OP_TRACE=sync
export LLMRS_FORWARD_GEN_OP_TRACE_PATH=/data/local/tmp/optrace_runN.json
./generate -m qwen2.5-1.5b-q4_0.gguf --backend opencl \
    -p "The capital of France is" -n 32 --temperature 0
```

5 runs back-to-back, 각 run에서 28 layer × 31 decode token = 868 layer-op sample.

---

## 2. 결과 — 5-run aggregate

### 2.1 Phase별 per-layer wall-clock (sync mode)

| Phase | mean (us) | stdev (us) | CV |
|---|---:|---:|---:|
| **cache-fit** (rms_norm × 2 + rope + kv_update + add_assign) | **984** | 12 | **1.2%** |
| ddr-heavy (matmul × 4: qkv/wo/ffn_gate_up/ffn_down) | 1706 | 18 | 1.1% |
| medium (attention + silu_mul) | 324 | 7 | 2.2% |
| **TOTAL per layer** | **3014** | — | — |

### 2.2 Per-op stability

| op | mean us | stdev | CV |
|---|---:|---:|---:|
| rms_norm_attn | 174 | 2.8 | 1.6% |
| rms_norm_ffn | 152 | 2.1 | 1.4% |
| rope | 336 | 12.7 | 3.8% |
| kv_update | 169 | 1.4 | 0.8% |
| add_assign | 154 | 1.4 | 0.9% |
| matmul_qkv | 362 | 5.7 | 1.6% |
| matmul_wo | 284 | 6.4 | 2.2% |
| matmul_ffn_gate_up | 638 | 0.0 | 0.0% |
| matmul_ffn_down | 422 | 6.4 | 1.5% |

**모든 op CV < 4%. matmul_ffn_gate_up은 σ=0** (deterministic).

---

## 3. Chunk size 산정

Adreno UMA host-pinned (DMA-BUF interop, Phase C commit `7e84800`) DDR throughput ~7-8 GB/s 가정.

| chunk size | transfer time @ 7 GB/s | per-layer cache-fit window 점유율 | 안전 margin |
|---:|---:|---:|---|
| 4 MB | 570 us | 58% | **30%** ← 권장 |
| 6 MB | 850 us | 87% | 13% |
| 8 MB | 1140 us | 116% | -16% (overrun risk) |

**Sweet spot: 4 MB chunk** (Phase R도 4-8 MB 권장, 본 측정으로 4 MB 보수 확정).

---

## 4. 25-layer swap 시뮬레이션

target_layers = 25 layer × ~36 MB Q4_0 weight = **~900 MB** total H2D.

| 모드 | 토큰당 hide budget | 25-layer 완료 토큰 수 | user-perceived stall |
|---|---|---|---|
| Single-shot (default) | — | 1 (290 ms stall) | 290 ms |
| LISWAP-1 per-tick=2 | 2 layer × 36 MB | 13 token | 23 ms / tick |
| **B (phase-aware async)** | 28 layer × 4 MB = **112 MB / tok** | **~8 token** | **0 ms (fully hidden)** |

이론적으로 완전 hide 가능. 검증 필요.

---

## 5. 다음 단계 — Implementation

### 5.1 신규 모듈
- `engine/src/models/weights/phase_aware_swap.rs` — chunk dispatcher + worker
- `engine/src/profile/op_trace.rs` — `OpKind::ddr_phase()` 추가 (✅ 본 세션 commit)

### 5.2 통합 포인트
- `forward_gen.rs` `tr_record!()` 매크로에 phase hook 콜백 추가 (zero overhead when None)
- `IntraForwardSwapHook` (LISWAP-4 v3) 와 mutually exclusive 또는 통합

### 5.3 CLI
- `--swap-phase-aware` (bool)
- `--swap-phase-aware-chunk-mb=4` (default 4 MB)

### 5.4 Pass-gate
- correctness: top-5 overlap > 99% vs single-shot swap
- TBT regression: ≤ baseline + 5%
- hide ratio: actual swap H2D wall-clock ≥ 80% hidden behind forward
- 25-layer swap user-perceived stall: ≤ 50 ms (vs 290 ms single-shot)

---

**End of measurement report**

Phase predictability GREEN. Implementation 진행 가능.
