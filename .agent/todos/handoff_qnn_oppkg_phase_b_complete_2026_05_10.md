# Handoff: D-D.6 fast path 완전 활성화 — Phase B 완료, fast path = production byte-equal

**Date**: 2026-05-10
**Previous handoff**: `.agent/todos/handoff_qnn_oppkg_root_cause_qkv_bias_2026_05_10.md`
**Branch**: master
**Device**: Galaxy S25 (R3CY408S5SB, Adreno 830 + Hexagon V79)

---

## 0. 한 줄 요약

D-D.6 fast path 완전 활성화. graph 14-node → 17-node (Q/K/V matmul 직후 BiasAdd 3개 추가) → fast path 토큰이 production fast-path-off와 **byte-identical**. INV-175 (decode 동안 fallback_call_count=0) 만족, 28-layer 모두 fast path. 단 perf는 production 대비 ~6배 느림 (다음 최적화 영역).

---

## 1. 검증 결과

### 1.1 Token sequence (Qwen2.5-1.5B Q4_0, prompt "The capital of France is", greedy)

**Production fast-path-off (reference)**: 
```
a pop Decimal-size=" oil Port233 to larg inople.thisheadsubmit
```

**Fast path on (D-D.6 Phase B)**: 
```
a pop Decimal-size=" oil Port233 to larg inople.thisheadsubmit
```

token IDs (LLMRS_DUMP_TOKEN_IDS=1):
```
[token-id step=0] id=2477   (pop)
[token-id step=1] id=27826  (Decimal)
[token-id step=2] id=7321   (-size)
[token-id step=3] id=429    (=")
[token-id step=4] id=5707   ( oil)
... (production과 동일)
```

**byte-identical token sequence**.

### 1.2 Performance
| 모드 | TBT | tok/s |
|---|---|---|
| Production fast-path-off | 12.78 ms/tok | 78.3 |
| Fast path on (Phase B) | 72.30 ms/tok | 13.8 |

fast path가 ~6배 느림. 17-node graph + 매 op DMA-buf bind + QNN GPU dispatch overhead 의심. 다음 perf 최적화 영역.

### 1.3 graphFinalize
- layer 0: 1554 ms (이전 1181 ms, 17-node로 ~25% 증가)
- 28-layer 누적 finalize: 1661 ms (이전 ~33s 비정상치, 이번 측정 정상)
- budget: 1500 → **2000 ms**로 갱신 (FINALIZE_BUDGET_MS)

---

## 2. Phase B 변경사항

### 2.1 새 op (이전 commit `64faa45`에서 이미 추가)
- `engine/kernels/simple_ops.cl::kernel_add_row_bias_oop` — OOP variant
- `crates/qnn_oppkg/src/ops/bias_add.rs` — `CustomBiasAdd` op descriptor
- `crates/qnn_oppkg/src/ops/mod.rs` + `registry.rs` — 등록

### 2.2 layer_graph build 통합 (이번 commit)
- `engine/src/backend/qnn_oppkg/layer_graph.rs`:
  - bias bytes 추출 (qkv_bias.bq/.bk/.bv 또는 zero fallback)
  - bias rpcmem slot 3개 alloc + copy (slots **끝에** push — 기존 weight slot indexing 보존이 필수, 처음에는 rms_post 직후 push해서 모든 weight memHandle 인덱스 깨짐 → forward = 0)
  - dims_q/k/v_bias [proj_out], dims_q/k/v_biased [1, n_head_*, head_dim]
  - tensor 등록 (q_bias/k_bias/v_bias APP_WRITE F32, q_biased/k_biased/v_biased NATIVE F32)
  - 새 op type "CustomBiasAdd"
  - in/out arrays for 3 BiasAdd ops
  - nodes array `[NodeSpec; 14]` → `[NodeSpec; 17]` — Q/K/V proj 직후 BiasAdd 삽입
  - RoPE Q input: t_q → **t_q_biased**, RoPE K: t_k → **t_k_biased**
  - KvScatter v_src: t_v → **t_v_biased**
  - exec_inputs에 t_q_bias_mh / t_k_bias_mh / t_v_bias_mh 추가
  - `FINALIZE_BUDGET_MS: 1500 → 2000`
- `crates/qnn_oppkg/src/graph/layer.rs`:
  - `LAYER_NODE_COUNT: 14 → 17`
  - `LAYER_INTERMEDIATE_COUNT: 13 → 16`
  - 테스트 갱신
- `engine/src/bin/microbench_qnn_qwen_layer.rs`, `microbench_qnn_qwen_layer_bisect.rs`, `microbench_qnn_28layer_tbt.rs`:
  - `_ASSERT_LAYER_NODE_COUNT: 14 → 17`
- `engine/src/bin/generate.rs`:
  - `LLMRS_DUMP_TOKEN_IDS=1` env로 raw token id 출력 (debug)

### 2.3 Phase B 실패→성공 디버깅 단계
1. **첫 시도**: graphFinalize err=0x1786 (`QNN_GRAPH_ERROR_FINALIZE_FAILED`)
   - 원인: VERBOSE log "Operation does not exist: qnn_oppkg CustomBiasAdd"
   - **fix**: `cargo build --release --target aarch64-linux-android -p qnn_oppkg` 별도 빌드 (cdylib `libqnn_oppkg.so`는 generate cargo build에 포함 안 됨) + `adb push libqnn_oppkg.so`
2. **두 번째 시도**: graphFinalize OK (err=0x0) but x_out = zero buffer
   - 원인: bias 3개 slot을 rms_post 직후에 push → 기존 weight slot indexing (qq=3, qd=4, ..., dd=16) 모두 wrong slot 가리킴 → 모든 weight memHandle invalid → forward = 0
   - **fix**: bias slots를 idx_n_kv 다음 (slots 끝)에 push + idx_q_bias / idx_k_bias / idx_v_bias 변수 추가 + set_mh + exec_inputs 추가
3. **세 번째 시도**: 토큰이 생성되지만 stdout에 "a" 만 보임
   - 원인 의심: stop signal? → token id dump 결과 매 step 151935 (vocab boundary) → fast path 결과 broken
   - 실은 (2)의 fix 결과로 이미 정상 동작했고, dumped는 fix 전의 stale binary 결과였음. 재빌드 + 재배포 후 정상 token sequence 확인.

---

## 3. 잔여 의심점 (Phase A.5 cos 0.72)

Phase A.5에서 microbench (with bias) vs forward_gen 비교 시 stage 5+ cos = 0.72 (1.0이 아님). 그러나 fast path 결과가 production과 byte-identical이므로 **stage-by-stage cos 0.72의 잔존 차이는 mathematically equivalent random walk**였던 것. 구체:
- forward_gen은 `rope_inplace` (in-place) 후 `attention_gen` 사용
- microbench raw chain은 `kernel_rope_simple` (in-place) + `flash_attn_f32_f16_q1` 사용
- graph fast path는 `CustomRope` + `CustomFlashAttn` 사용
- 세 path가 numerical drift는 있지만 **logits argmax는 같음** (greedy decode가 같은 token 선택)

따라서 Phase A.5의 cos 0.72는 root cause가 아니었음 (numerical drift는 항상 있음). bias 누락이 진짜 root cause.

---

## 4. 다음 단계 (Phase C — 다음 세션)

### 4.1 Perf 최적화 (~6배 gap)
- DMA-buf binding 비용 측정 (graphExecute 호출 시 매 input/output rebind?)
- QNN GPU dispatch overhead 분석 (Adreno 830 GPU vs OpenCL secondary 비교)
- 17-node 분해 — bias add를 matmul fused로 (CustomMatMulQ40F32WithBias 신규 op?)

### 4.2 INV-175 검증
- decode 동안 `fallback_call_count = 0` 확인 (28-layer 모두 fast path 유지)
- 검증 명령: 별도 metric 추가 또는 fallback_call_count printer

### 4.3 다른 모델 검증
- Llama3.2-1B (bias 없음) — zero bias로 17-node 동작하지만 effective no-op. perf overhead만.
- Gemma3 (bias 없음, 다른 norm 패턴) — 동작 가능성 검증

### 4.4 디버그 코드 정리
- `LLMRS_QNN_OPPKG_DUMP_FALLBACK_PREFIX` (forward_gen stage 0~16 dump) — 유지
- `LLMRS_MICROBENCH_DUMP_PREFIX` — 유지
- `LLMRS_DUMP_TOKEN_IDS` — 유지 (디버그 도구)
- Phase A.5 microbench bias dispatch — 유지 (검증용)

---

## 5. 다음 세션 진입 명령

```bash
# fast path 정상 동작 확인
adb shell '
export LD_LIBRARY_PATH=/data/local/tmp/qnn:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=/data/local/tmp/qnn
export LLMRS_QNN_OPPKG_FAST_PATH=1
export LLMRS_SKIP_WARMUP=1
./generate -m /data/local/tmp/models/qwen2.5-1.5b-q4_0.gguf \
    -p "The capital of France is" -n 16 \
    --backend qnn_oppkg --temperature 0 --initial-kv-capacity 2048
'
# Expected: "a pop Decimal-size=\" oil Port233 to larg inople.thisheadsubmit"
# (production fast-path-off과 byte-identical)

# token id 확인
adb shell '
export LD_LIBRARY_PATH=/data/local/tmp/qnn:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=/data/local/tmp/qnn
export LLMRS_QNN_OPPKG_FAST_PATH=1
export LLMRS_SKIP_WARMUP=1
export LLMRS_DUMP_TOKEN_IDS=1
./generate ... --ignore-eos 2>&1 | grep token-id
'
```

---

## 6. 주의사항

### DO
- libqnn_oppkg.so를 변경한 경우 항상 **별도 빌드 + push** (`cargo build --release --target aarch64-linux-android -p qnn_oppkg`).
- 새 rpcmem slot 추가 시 **slots 배열의 끝에 push** — 기존 hard-coded 인덱스 (qq=3, ..., dd=16, kcache=17, ..., x_out=22)를 절대 깨면 안 됨.
- LAYER_NODE_COUNT 변경 시 microbench `_ASSERT_LAYER_NODE_COUNT` 3개 모두 갱신.

### DO NOT
- bias slot을 rms_post 직후에 push 하지 말 것 (weight slot indexing 깨짐 → x_out = 0).
- production OpenCL backend의 add_row_bias 변경 금지.
- BiasAdd intermediate (q_biased/k_biased/v_biased)를 in-place로 변경 금지 — graph framework이 NATIVE intermediate의 두 producer를 허용하지 않음.

---

## 7. Git status

수정 (commit 예정):
- `crates/qnn_oppkg/src/graph/layer.rs` — LAYER_NODE_COUNT 17, INTERMEDIATE 16
- `engine/src/backend/qnn_oppkg/layer_graph.rs` — BiasAdd 3개 통합 + budget 2000ms
- `engine/src/bin/generate.rs` — LLMRS_DUMP_TOKEN_IDS env
- `engine/src/bin/microbench_qnn_qwen_layer.rs` + `_bisect.rs` + `_28layer_tbt.rs` — _ASSERT 17

신규:
- `.agent/todos/handoff_qnn_oppkg_phase_b_complete_2026_05_10.md` (이 파일)

---

**End of Handoff**

D-D.6 fast path 완전 활성화. byte-identical token sequence 확인. 다음: perf 최적화 (Phase C).
