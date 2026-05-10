# Handoff: D-D.6 fast path root cause = QKV bias 누락 — Phase A 완료, Phase B 미완

**Date**: 2026-05-10
**Previous handoff**: `.agent/todos/handoff_qnn_oppkg_d_d_6_root_cause_isolated_2026_05_10.md`
**Branch**: master
**Device**: Galaxy S25 (R3CY408S5SB, Adreno 830 + Hexagon V79)

---

## 0. 한 줄 요약

Fast path mismatch root cause는 **graph 14-node에 Qwen2.5 attention QKV bias add op이 누락**된 것. forward_gen은 `add_row_bias` 적용, graph는 무시 → Q/K/V vector wrong → 28-layer 누적 garbage 토큰. Phase A에서 stage-by-stage byte/numerical 비교로 격리 완료. Phase B (graph BiasAdd op + weight load + buffer slot) 진행 중 — op 자체는 추가 완료, layer_graph.rs build integration 미완.

---

## 1. Phase A: 격리 (이번 세션 완료)

### 1.1 Stage-by-stage 인프라 추가
- `engine/src/layers/transformer_layer/forward_gen.rs` — `dump_stage!` macro + stage 0~16 dump (`LLMRS_QNN_OPPKG_DUMP_FALLBACK_PREFIX=path` → `path.stage{N}` 파일)
- `engine/src/bin/microbench_qnn_qwen_layer.rs` — `dump_stage_mb!` macro + stage 1~16 dump (`LLMRS_MICROBENCH_DUMP_PREFIX=path`)

forward_gen와 microbench 1:1 stage 매핑:
| Stage | 의미 | forward_gen 변수 | microbench buffer |
|---|---|---|---|
| 0 | Input x (pre-norm) | `x` | (inject only) |
| 1 | RMS pre out | `ws.residual` | `buf_y1` |
| 2/3/4 | Q/K/V matmul (pre-bias) | `ws.q/k/v` | `buf_q/k/v` |
| 5/6 | RoPE Q/K (post-bias-RoPE) | `ws.q/k` (in-place) | `buf_q/k` (in-place) |
| 7/8 | KV scatter K/V (full cache) | `kv_cache.get_view()` | (skip) |
| 9 | Attention out | `ws.out_attn` | `buf_attn_o` |
| 10 | O proj out | `ws.attn_out` | `buf_o` |
| 11 | RMS post (post add+norm) | `ws.residual` | `buf_y2` |
| 12/13/14 | gate/up/silu_mul | `ws.gate/up/gate` | `buf_gate/up/gate` |
| 15 | down out | `ws.down` | `buf_down` |
| 16 | layer 0 out | `x` | `buf_x_out` |

### 1.2 비교 결과 (numerical drift propagation)

**Run B**: `--backend qnn_oppkg` (fast path off), `LLMRS_SKIP_WARMUP=1`, prompt "The capital of France is", `-n 4`. forward_gen이 dump_fb.stage{0~16}.

**Run C-1** (bias 미적용 microbench, original): forward_gen vs raw chain stage 비교
| Stage | cos | |fb| | |mb| | max_abs | 해석 |
|---|---|---|---|---|---|
| 1 | 1.000000 | 20.33 | 20.20 | 5.1e-2 | RMS LSB drift만 (ok) |
| 2 | 1.000000 | 79.88 | 79.36 | 1.2e-1 | Q matmul propagate drift |
| 5 | **0.43** | 113.20 | 79.36 | **23.81** | **발산 — norm 큰 차이** |
| 11 | 0.59 | 13.91 | 13.65 | 2.73 | 완전 발산 |
| 16 | 0.58 | 23.03 | 24.07 | 3.46 | 완전 발산 |

|fb stage 5| (113.2) ≠ |mb stage 5| (79.36) — RoPE는 norm 보존하므로 input 자체가 다름. 즉 **bias 적용 (forward_gen) vs 미적용 (microbench)** 의심.

**확인**: GGUF tensor list에 `blk.0.attn_q.bias`, `blk.0.attn_k.bias`, `blk.0.attn_v.bias` 존재 (Qwen2.5는 attention QKV bias 사용).

**Run C-2** (microbench에 bias add 추가 후): cos 회복
| Stage | cos (before bias) | cos (after bias) |
|---|---|---|
| 5 | 0.43 | **0.72** ← 큰 회복 |
| 11 | 0.59 | 0.76 |
| 16 | 0.58 | 0.71 |

|mb stage 5| 79.36 → 113.0 ≈ |fb| 113.2 (norm 일치). bias가 root cause의 **큰 부분**.

### 1.3 잔존 발산 (cos 0.72 ≠ 1.0)
bias 추가 후에도 stage 5+에서 cos가 1.0에 도달하지 못함. 추가 발산 원인 후보:
- RoPE kernel ordering 차이 (forward_gen `backend.rope_inplace` vs microbench `kernel_rope_simple`)
- Flash attention 차이 (`backend.attention_gen` vs `flash_attn_f32_f16_q1`)
- KV scatter ordering (Run B의 dump_fb.stage7/8을 inject 사용했으나 layout/순서 미세 차이)

이 잔존 발산은 fast path 활성화 시점에서 **valid divergent walk** 만들지 갈지 미상. bias만 fix해도 garbage 탈출 가능성 있음. **Phase B 완료 후 fast path 토큰 sanity check 필요**.

---

## 2. Phase B: Graph fix (이번 세션 부분 완료)

### 2.1 완료
1. `engine/kernels/simple_ops.cl` — 새 OOP kernel `kernel_add_row_bias_oop(x, bias, y, dim, total_elements)` 추가 (line ~1041 직전).
2. `crates/qnn_oppkg/src/ops/bias_add.rs` 신규 — `CustomBiasAdd` op descriptor:
   - `kernel_add_row_bias_oop` 매핑
   - inputs: x [rows*dim], bias [dim] / output: y [rows*dim]
   - args: InputTensor(0), InputTensor(1), OutputTensor(0), Int(dim), Int(total)
   - global_work=[total, 1, 1], local default
3. `crates/qnn_oppkg/src/ops/mod.rs` — `pub mod bias_add;`
4. `crates/qnn_oppkg/src/registry.rs` — OPS slice에 `crate::ops::bias_add::DESCRIPTOR` 추가
5. 호스트 컴파일 PASS (`cargo check -p qnn_oppkg && cargo check -p llm_rs2 --bin generate`)

### 2.2 미완 (다음 세션 작업)

#### B.1: layer_graph build에 BiasAdd op 3개 삽입
- 파일: `engine/src/backend/qnn_oppkg/layer_graph.rs::build_layer_graph()` (line 501~)
- 현재 chain: `RmsNorm → Q/K/V matmul → RoPE → KvScatter → FlashAttn → O → Add → RmsNorm → gate/up → SiluMul → down → Add` (14 nodes)
- 신규 chain: `RmsNorm → Q matmul → BiasAdd_Q → K matmul → BiasAdd_K → V matmul → BiasAdd_V → RoPE → ... → Add` (17 nodes)
- 각 BiasAdd 후 새 buffer가 RoPE input
- `graphAddNode` 호출 3번 추가 + op_type CString `"CustomBiasAdd"`

#### B.2: bias weight tensor 3개 load
- 파일: `engine/src/backend/qnn_oppkg/layer_slot.rs` (또는 weight_pack.rs)
- `blk.{i}.attn_q.bias`, `blk.{i}.attn_k.bias`, `blk.{i}.attn_v.bias` (FLOAT_32)
- shape: [q_proj_out=1536], [kv_proj_out=256], [kv_proj_out=256] (Qwen2.5-1.5B)
- 모델별 옵셔널 (Llama3.2는 bias 없음 — Some/None handling)

#### B.3: graph_cache buffer slot 3개 추가
- 파일: `engine/src/backend/qnn_oppkg/graph_cache.rs` 또는 layer_graph.rs (rpcmem buffer 정의 영역)
- `q_bias`, `k_bias`, `v_bias` (FLOAT_32, host-aligned alloc)
- DMA-buf upload (rpcmem_alloc + rpcmem_to_fd → QNN tensor bind)

#### B.4: BiasAdd output buffer 3개 추가
- Q matmul 출력 (buf_q) → BiasAdd_Q input → 새 buf `q_biased` → RoPE input
- K, V도 동일
- 또는 OOP write가 input과 같은 buffer (in-place 효과)을 graph framework이 허용하면 buf_q 재사용 가능 (검증 필요)

#### B.5: supports_layer_graph eligibility
- 파일: `engine/src/backend/qnn_oppkg/mod.rs::supports_layer_graph()`
- 현재 모델별 has_qkv_bias 검사 없음. 추가 권장:
  ```rust
  if model.has_qkv_bias() && !graph_supports_bias { return false; }
  ```
  단 Phase B에서 graph_supports_bias = true가 되므로 fall-through OK.

### 2.3 Phase B 검증 단계
1. 빌드 + deploy
2. fast path on + 토큰 확인 (현재 garbage `a and of_res` → bias fix 후 valid sequence 기대)
3. microbench: bias inject + graph 비교 → max_abs_err PASS 확인
4. forward_gen vs microbench stage 5+ 비교 → cos 0.72 → 1.0 도달 확인
5. INV-175: 28-layer fast path 유지 + fallback_call_count=0

---

## 3. 이번 세션 변경사항 (commit 가치)

| 파일 | 변경 |
|---|---|
| `engine/src/layers/transformer_layer/forward_gen.rs` | dump_stage! macro + stage 0~16 dump (env-gated) |
| `engine/src/bin/microbench_qnn_qwen_layer.rs` | dump_stage_mb! macro + stage 1~16 dump + QKV bias add (k_bias, host_q/k/v_bias, buf_q/k/v_bias, dispatch_bias, bias_no_finish!) |
| `engine/kernels/simple_ops.cl` | `kernel_add_row_bias_oop` 신규 (OOP variant for graph) |
| `crates/qnn_oppkg/src/ops/bias_add.rs` | **NEW** CustomBiasAdd op descriptor |
| `crates/qnn_oppkg/src/ops/mod.rs` | bias_add 모듈 등록 |
| `crates/qnn_oppkg/src/registry.rs` | OPS slice에 bias_add::DESCRIPTOR |

---

## 4. 디버깅 env reference (신규/누적)

```bash
# Production fallback 측 (forward_gen)
LLMRS_QNN_OPPKG_DUMP_FALLBACK_PREFIX=/path        # stage 0~16 binary dump
LLMRS_QNN_OPPKG_DUMP_FALLBACK_RMS_OUT=/path       # 단일 RMS dump (legacy)

# Microbench 측
LLMRS_MICROBENCH_GGUF=/path/to/qwen.gguf          # GGUF weight + bias inject
LLMRS_MICROBENCH_X_FILE=/path                     # input x inject
LLMRS_MICROBENCH_KV_K_FILE=/path                  # KV K inject
LLMRS_MICROBENCH_KV_V_FILE=/path
LLMRS_MICROBENCH_POS=k                            # pos override
LLMRS_MICROBENCH_N_KV=k
LLMRS_MICROBENCH_DUMP_PREFIX=/path                # stage 1~16 binary dump

# 이전 세션 누적 (production fast path 측)
LLMRS_QNN_OPPKG_FAST_PATH=1
LLMRS_QNN_OPPKG_FAST_PATH_DUMP_TARGET_POS=k
LLMRS_QNN_OPPKG_FAST_PATH_DUMP_X=/path
LLMRS_QNN_OPPKG_FAST_PATH_DUMP_KV_K=/path
LLMRS_QNN_OPPKG_FAST_PATH_DUMP_KV_V=/path
LLMRS_QNN_OPPKG_FAST_PATH_FORCE_SYNC=1
LLMRS_QNN_OPPKG_FAST_PATH_FRESH_BUILD=1
LLMRS_QNN_OPPKG_PREBUILD_MAX_LAYERS=k
LLMRS_QNN_OPPKG_VERBOSE_LOG=1
```

---

## 5. 다음 세션 진입 명령

```bash
# 1. Phase B.1~B.4 작업 시작 — layer_graph.rs build에 BiasAdd op 3개 삽입
#    (graph node 14 → 17, 새 buffer slot 3개)
#
# 2. layer_slot.rs (또는 weight_pack.rs)에 bias tensor load 추가
#
# 3. 빌드 + 배포
python scripts/run_device.py -d galaxy_s25 --extra-bin microbench_qnn_qwen_layer --skip-exec generate
adb push target/aarch64-linux-android/release/microbench_qnn_qwen_layer /data/local/tmp/

# 4. fast path 토큰 검증
adb shell '
export LD_LIBRARY_PATH=/data/local/tmp/qnn:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=/data/local/tmp/qnn
export LLMRS_QNN_OPPKG_FAST_PATH=1
export LLMRS_SKIP_WARMUP=1
./generate -m /data/local/tmp/models/qwen2.5-1.5b-q4_0.gguf \
    -p "The capital of France is" -n 16 \
    --backend qnn_oppkg --temperature 0 --initial-kv-capacity 2048
'
# Expected (Phase B 완료 후): valid Qwen sequence (production과 다를 수 있으나 garbage 아님)

# 5. microbench: bias inject + graph 비교
adb shell '
export LD_LIBRARY_PATH=/data/local/tmp/qnn:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=/data/local/tmp/qnn
export LLMRS_MICROBENCH_GGUF=/data/local/tmp/models/qwen2.5-1.5b-q4_0.gguf
./microbench_qnn_qwen_layer
'
# Expected: max_abs_err PASS (raw chain == graph, 둘 다 bias 적용)
```

---

## 6. 주의사항

### DO
- Phase B.1 작업 시 layer_graph.rs::build_layer_graph의 기존 14-node 흐름을 깨지 않도록 주의. 3개 BiasAdd 삽입은 Q/K/V matmul 직후 + RoPE 직전 위치만.
- bias tensor가 모델에 없는 경우 (Llama 등) graceful skip — `if let Some(bias)` 패턴.
- 기존 dump 인프라 + microbench bias 코드 그대로 유지 (다음 세션 검증에 도움).

### DO NOT
- production OpenCLBackend의 add_row_bias kernel/flag 변경 금지.
- forward_gen의 stage dump macro 제거 금지 (Phase B 검증에 사용).
- `kernel_add_row_bias_oop`을 in-place로 변경 금지 — graph 14-node OOP chain 패턴 유지.

---

## 7. Git status (이번 세션 commit 예정)

수정:
- `engine/src/layers/transformer_layer/forward_gen.rs`
- `engine/src/bin/microbench_qnn_qwen_layer.rs`
- `engine/kernels/simple_ops.cl`
- `crates/qnn_oppkg/src/ops/mod.rs`
- `crates/qnn_oppkg/src/registry.rs`

신규:
- `crates/qnn_oppkg/src/ops/bias_add.rs`
- `.agent/todos/handoff_qnn_oppkg_root_cause_qkv_bias_2026_05_10.md` (이 파일)

---

**End of Handoff**

Phase A 격리 완료 / Phase B op 자체 추가 완료 / Phase B graph integration 다음 세션.
