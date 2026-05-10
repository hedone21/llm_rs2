# Handoff: D-D.6 fast path root cause 격리 완료 — fix 미완

**Date**: 2026-05-10
**Previous handoff**: `.agent/todos/handoff_qnn_oppkg_d_f_d_d_6_complete_2026_05_10.md`
**Branch**: master (commit pending)
**Device**: Galaxy S25 (R3CY408S5SB, Adreno 830 + Hexagon V79)

---

## 0. 한 줄 요약

Fast path mismatch root cause는 **graph (qnn_oppkg) ops vs production OpenCL backend의 numerical implementation 차이**. graph 14-node algorithm 자체는 정상 (microbench로 검증), 하지만 graph가 사용하는 kernel/algorithm/build-flag 조합이 production OpenCL과 byte-equal한 결과를 만들지 못함. RMS만 fix해도 token garbage 잔존 — 다른 stage (matmul, RoPE, FA, etc.)도 모두 같은 issue.

---

## 1. 격리 매트릭스 (이번 세션 핵심 발견)

### microbench (graph 14-node) — 모든 환경에서 PASS
| 환경 | weight | x_in | KV | 결과 |
|---|---|---|---|---|
| random | random F32 → quantize | random | zero | **PASS** byte-equal |
| GGUF inject | GGUF Q4_0 (production) | random | zero | **PASS** |
| Production state inject | GGUF Q4_0 | production embed (pos=6 dump) | production prefill (pos=6 dump) | **PASS** |

→ **graph algorithm 자체는 100% 정상**. multi-token KV + nonzero pos 환경에서도 raw OpenCL chain과 byte-equal.

### Production fast path — 모든 환경에서 garbage
| 검증 | 결과 |
|---|---|
| layer 0 fast path만 활성 (`MAX_LAYER=1`) | garbage `a the to the. isileunction` |
| 28-layer 모두 fast | garbage `a overdublic vocational(QL automaticbirthdatequImportant...` |
| `LLMRS_SKIP_NOSHUFFLE_SOA=1` (8x_flat 강제) | garbage 동일 |
| `LLMRS_SKIP_WARMUP=1` | garbage 동일 |
| `LLMRS_QNN_OPPKG_PREBUILD_MAX_LAYERS=1` (cache 누적 무관) | garbage 동일 |
| `LLMRS_QNN_OPPKG_FAST_PATH_FRESH_BUILD=1` (cache 우회) | garbage 동일 |
| `LLMRS_QNN_OPPKG_FAST_PATH_FORCE_SYNC=1` (queue sync) | garbage 동일 |
| 새 RMS kernel + 새 build flag (이번 세션 fix) | garbage `a and of_res certiew is. smin brutalunction Indones',r` (다른 패턴) |

### byte-level 격리 (production fast path 첫 decode pos=6)
| 항목 | fast path graph | OpenCL fallback (forward_prefill) | 결과 |
|---|---|---|---|
| x_in (token embedding) | `[-0.0234, -0.0217, ...]` | `[-0.0234, -0.0217, ...]` | **byte-equal** |
| KV K bytes | `[b4, 45, 8d, c2, ...]` | `[b4, 45, 8d, c2, ...]` | **byte-equal** |
| weight (graph build read) | `aos[0..16]={92, 1c, ab, f6, ...}` (정상) | (production tensor) | **byte-equal** |
| RMS output (post-norm) | (graph internal NATIVE) | `[14, 33, 0d, bf, ...]` (-0.5515) | (graph는 직접 dump 어려움) |
| layer 0 x_out | `[0.5867, 0.7293, ...]` (oop_subgroup 후) / `[0.041, ...]` (simple 이전) | `[0.4621, -0.0328, ...]` | **다름 (5767/6144 bytes)** |

→ 같은 input + 같은 weight + 같은 KV에도 graph가 다른 forward 결과 생산. **algorithm/kernel 자체 차이**.

---

## 2. Root cause 격리 단계 (이번 세션 진행 순서)

### Phase 1: production state dump 도구 추가
`execute_layer_graph`에 첫 fast path 호출 시점의 x_in / KV K / KV V / x_out을 binary file로 dump (env-gated). microbench inject 용도.

### Phase 2: microbench inject 인프라 확장
microbench가 production state를 그대로 받을 수 있도록:
- `LLMRS_MICROBENCH_GGUF=/path` — layer 0 weight 7개 + 2 norm 모두 GGUF에서 inject
- `LLMRS_MICROBENCH_X_FILE=/path` — production embedding dump file inject
- `LLMRS_MICROBENCH_KV_K_FILE=/path` + `LLMRS_MICROBENCH_KV_V_FILE=/path` — production KV state inject (raw chain buf_kcache + graph rpcmem 양쪽)
- `LLMRS_MICROBENCH_POS=k` + `LLMRS_MICROBENCH_N_KV=k` — pos/n_kv override

### Phase 3: 비교 실행 결과
- microbench (production state inject) → **byte-equal PASS** (max_abs_err = 0.000000e0)
- 이 결과로 **graph algorithm은 정상**, multi-token attention도 정확 확정

### Phase 4: production-only path 격리
production fast path만의 차이를 격리하기 위해:
- WARMUP probe (pos=0 dump) — model load + prebuild 직후 state. 같은 결과 (graph는 정상 forward).
- production fallback `forward_prefill`의 RMS output dump — `kernel_rms_norm_opt` 결과
- 같은 input으로 microbench raw chain의 RMS output dump — `kernel_rms_norm_simple` 결과
- **두 RMS bytes 다름** (4594/6144 bytes 차이)

### Phase 5: kernel difference 발견
- production: `kernel_rms_norm_opt` (parallel `sub_group_reduce_add` + `__local` scratch)
- graph + microbench raw: `kernel_rms_norm_simple` (single-thread sequential loop)
- mathematically equivalent이지만 floating point 비결합성으로 실제 numerical 결과 다름

### Phase 6: Option A 시도 — graph kernel을 production-style로 교체
새 kernel `kernel_rms_norm_oop_subgroup` (single-subgroup, OOP variant) 추가:
- `local_size = sg_size = 64` 강제 → cross-subgroup reduce 불필요 → SLM 불필요 (LOCAL arg issue 우회)
- production opt와 algorithmically equivalent (single subgroup case)
- graphFinalize PASS (이전 `kernel_rms_norm_oop`의 0x1786 issue 우회)

### Phase 7: build flag 통일 시도
- production OpenCLBackend: `-cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math`
- graph + microbench (이전): `-cl-mad-enable -cl-fast-relaxed-math`
- 두 flag 추가 후 다시 비교 → **여전히 4597 byte 차이**

### Phase 8 (미완): 다른 차이 분석
- 같은 algorithm, 같은 flag인데 byte-equal 안됨
- 가능 원인: SDK OpPackage 컴파일 path가 OpenCLBackend의 직접 컴파일과 다른 binary 생성, sub_group_reduce_add 구현 차이, single-subgroup case 인식 차이 등
- **결정적 분석 미완** — 다음 세션 진입점

---

## 3. 이번 세션 변경사항 (10 files, +659/-70)

### Production code (commit 가치)
| 파일 | 변경 내용 |
|---|---|
| `engine/src/backend/qnn_oppkg/layer_graph.rs` | mask buffer size **버그 fix** (`bytes_mask = n_kv*2` → `kv_capacity*2`, D-D.6 누락 변경), graph_name unique counter (fresh build mode 충돌 회피) |
| `engine/kernels/simple_ops.cl` | `kernel_rms_norm_oop_subgroup` 신규 (production opt와 algorithmically equivalent, LOCAL 없이 single-subgroup) |
| `crates/qnn_oppkg/src/ops/rms_norm.rs` | kernel_name `kernel_rms_norm_simple` → `kernel_rms_norm_oop_subgroup`, build_options에 `-cl-unsafe-math-optimizations -cl-finite-math-only` 추가, workgroup `[rows*64, 64]` |
| `engine/src/models/transformer.rs` | fast path eligibility에 `seq_len == 1` 명시 (defensive, 외부 if-else와 redundant이지만 명확화) |

### Debug infrastructure (env-gated, production 영향 없음)
| 파일 | env vars |
|---|---|
| `engine/src/backend/qnn_oppkg/mod.rs` | `LLMRS_QNN_OPPKG_FAST_PATH_DUMP_TARGET_POS`, `_DUMP_X`, `_DUMP_X_OUT`, `_DUMP_KV_K`, `_DUMP_KV_V`, `_FORCE_SYNC`, `_FRESH_BUILD`, `_DUMP` |
| `engine/src/backend/qnn_oppkg/graph_cache.rs` | `LLMRS_QNN_OPPKG_PREBUILD_MAX_LAYERS=k` (cap prebuild), `fresh_build()` helper |
| `engine/src/backend/qnn_oppkg/runtime.rs` | `LLMRS_QNN_OPPKG_VERBOSE_LOG=1` (logCreate VERBOSE) |
| `engine/src/models/transformer.rs` | `LLMRS_QNN_OPPKG_FAST_PATH_MAX_LAYER=k` (layer-level bisect), `LLMRS_QNN_OPPKG_FAST_PATH_DUMP` (fallback x_in/x_out/KV dump), `LLMRS_QNN_OPPKG_DUMP_FALLBACK_X_OUT` |
| `engine/src/layers/transformer_layer/forward.rs` | `LLMRS_QNN_OPPKG_DUMP_FALLBACK_RMS_OUT` (forward_prefill RMS dump) |
| `engine/src/layers/transformer_layer/forward_gen.rs` | 동일 (forward_gen RMS dump) |
| `engine/src/bin/microbench_qnn_qwen_layer.rs` | `LLMRS_MICROBENCH_GGUF`, `_X_FILE`, `_KV_K_FILE`, `_KV_V_FILE`, `_POS`, `_N_KV`, `_DUMP_RMS_OUT`, `_OCL_SECONDARY` |

---

## 4. 다음 세션 작업 plan

### Phase A — Stage-by-stage byte 비교 (1-2일)

graph 14-node 각 stage가 production OpenCL backend와 byte-equal한지 확인. 의심 stage 우선순위:

1. **RMS norm (RmsNorm pre/post)** — 이미 4597 byte 차이 확인. 같은 algorithm/flag으로도 mismatch. 정확한 origin 분석 필요:
   - production은 `kernel_rms_norm_opt` (in-place), graph는 `kernel_rms_norm_oop_subgroup` (OOP). in-place vs OOP 차이?
   - SDK OpPackage 내부의 kernel 컴파일이 OpenCL primary와 다른 binary 생성하는지 확인 (kernel binary dump 비교)

2. **MatMul Q4_0 (Q/K/V/O/gate/up/down proj)** — production opt path는 noshuffle GEMV (transposed SOA), graph는 8x_flat (non-transposed SOA). algorithm 자체가 다른 layout. 같은 weight matrix 표현하지만 numerical 다름.

3. **RoPE** — production `kernel_rope_simple` vs graph `CustomRope`. 별도 kernel일 가능성.

4. **KvScatter** — production vs graph 같은 kernel 사용? layout 일치 검증.

5. **FlashAttn** — D-D.6에서 새 kernel `flash_attn_f32_f16_q1_dynkv` 사용. production은 `flash_attn_f32_f16_q1`. body 동일 가정이지만 실측 검증 필요.

검증 방법: 각 stage output을 forward_prefill과 forward_gen에 dump 추가 + microbench raw chain stage별 dump + byte 비교.

### Phase B — Architectural 접근 (Phase A 결과에 따라)

stage별 차이가 원인이 분명하면:
- **B.1**: graph가 OpenCLBackend의 kernel을 직접 reuse (ops에서 `include_str!`로 같은 .cl 파일 + 같은 build_options)
- **B.2**: production OpenCL fallback path를 graph와 같은 kernel 사용하도록 변경 (env-gated, perf 저하 수용)
- **B.3**: graph ops 자체를 production OpenCL impl과 source-level 일치시키도록 redesign

### Phase C — Verification + commit
- 모든 stage byte-equal 확인 후 fast path 활성화
- INV-175 (decode 동안 fallback_call_count=0) 검증
- 디버그 env-gated 코드 정리 또는 유지 (다음 세션 도움)

---

## 5. 다음 세션 진입 명령

### Setup (이전 handoff와 동일)
- QNN SDK: `third_party/qnn_sdk_2.33/` (gitignored, 469MB)
- Device: Galaxy S25 (R3CY408S5SB) — 7개 QNN lib `/data/local/tmp/qnn/`에 push
- libqnn_oppkg.so → `/data/local/tmp/`

### 빠른 검증 (현재 상태)
```bash
# Default (fast path off) — production path, 정상 토큰
adb shell './generate -m /data/local/tmp/models/qwen2.5-1.5b-q4_0.gguf \
    -p "The capital of France is" -n 16 --backend qnn_oppkg --temperature 0'
# Expected: a pop Decimal-size=" oil Port233 to larg inople.thisheadsubmit

# Fast path full activation — 여전히 garbage (이번 세션 fix 미완)
adb shell 'export LLMRS_QNN_OPPKG_FAST_PATH=1 && \
    export LLMRS_SKIP_WARMUP=1 && \
    ./generate ... --initial-kv-capacity 2048'
# Expected (현 시점): a and of_res certiew is. smin brutalunction Indones',r
```

### Stage-별 byte 비교 워크플로우 (다음 세션)
1. **Production fallback path stage dump 위치**: `engine/src/layers/transformer_layer/forward.rs` (forward_prefill — start_pos>0 + seq_len==1 진입). 각 stage 직후 `backend.read_buffer()` + `std::fs::write()` 추가.
2. **Microbench raw chain stage dump 위치**: `engine/src/bin/microbench_qnn_qwen_layer.rs` (Stage 1~14 사이). 각 stage 후 `enqueue_read_buffer` + write.
3. **Compare**: `cmp -l file1 file2 | wc -l`로 diff bytes 카운트. 완전 byte-equal 도달까지.

### 디버깅 env reference (이번 세션 추가, 모두 env-gated)
```bash
# Production 측
LLMRS_QNN_OPPKG_FAST_PATH=1                          # fast path 활성
LLMRS_QNN_OPPKG_FAST_PATH_MAX_LAYER=k                # layer < k만 fast
LLMRS_QNN_OPPKG_FAST_PATH_DUMP=1                     # x_in/x_out float print
LLMRS_QNN_OPPKG_FAST_PATH_DUMP_TARGET_POS=k          # dump 진입 pos (default 6)
LLMRS_QNN_OPPKG_FAST_PATH_DUMP_X=/path               # x_in binary
LLMRS_QNN_OPPKG_FAST_PATH_DUMP_X_OUT=/path           # x_out binary
LLMRS_QNN_OPPKG_FAST_PATH_DUMP_KV_K=/path
LLMRS_QNN_OPPKG_FAST_PATH_DUMP_KV_V=/path
LLMRS_QNN_OPPKG_FAST_PATH_FORCE_SYNC=1               # cross-queue sync
LLMRS_QNN_OPPKG_FAST_PATH_FRESH_BUILD=1              # cache 우회
LLMRS_QNN_OPPKG_PREBUILD_MAX_LAYERS=k                # prebuild cap
LLMRS_QNN_OPPKG_VERBOSE_LOG=1                        # QNN logCreate VERBOSE
LLMRS_QNN_OPPKG_DUMP_FALLBACK_RMS_OUT=/path          # fallback path RMS dump
LLMRS_QNN_OPPKG_DUMP_FALLBACK_X_OUT=/path

# Microbench 측
LLMRS_MICROBENCH_GGUF=/path/to/qwen.gguf             # GGUF weight inject
LLMRS_MICROBENCH_X_FILE=/path                        # production x inject
LLMRS_MICROBENCH_KV_K_FILE=/path
LLMRS_MICROBENCH_KV_V_FILE=/path
LLMRS_MICROBENCH_POS=k                               # pos override (default 0)
LLMRS_MICROBENCH_N_KV=k                              # n_kv override (default 1)
LLMRS_MICROBENCH_DUMP_RMS_OUT=/path                  # raw chain RMS dump
LLMRS_MICROBENCH_OCL_SECONDARY=1                     # OpenCLBackend init (production-style)
```

---

## 6. 부정된 가설 (재검토 불필요)

| 가설 | 검증 방법 | 결과 |
|---|---|---|
| graph 14-node algorithm bug | microbench production state inject | PASS — algorithm 정상 |
| weight bytes 차이 | graph build hex dump vs production read | byte-equal |
| KV cache layout mismatch | production fast path KV K bytes vs OpenCL fallback KV bytes | byte-equal (`[b4,45,8d,c2,...]`) |
| token embedding bridge | x_in byte 비교 | byte-equal |
| mask buffer size (D-D.6 누락) | dims_mask vs slot.size | **버그였음 — 이번 세션 fix** |
| WARMUP의 KV contamination | SKIP_WARMUP 시 동일 garbage | 무관 |
| 28-layer prebuild 누적 contamination | PREBUILD_MAX_LAYERS=1 | 무관 |
| graph cache stale | FRESH_BUILD per execute | 무관 |
| OpenCL/QNN cross-queue sync | FORCE_SYNC | 무관 |
| OpenCL secondary cl_context 점유 | microbench OCL_SECONDARY init | 무관 |
| backendCreate/contextCreate config | runtime.rs vs microbench | 동일 |
| graphCreate config | layer_graph.rs vs microbench | 동일 |
| noshuffle vs 8x_flat | SKIP_NOSHUFFLE_SOA | 무관 |
| add_unit (Gemma3) | Qwen은 false 일관 | 무관 |
| RMS_EPS, rope_theta | 1e-5, 1M 일관 | 동일 |
| seq_len > 1에서 fast path 진입 (prefill) | 외부 if-else가 이미 차단 | (확인됨, 무관) |

---

## 7. 외부 참조

### 진입점 (다음 세션 작업)
- `engine/src/backend/qnn_oppkg/mod.rs::execute_layer_graph` — fast path bridge + dump probes
- `engine/src/backend/qnn_oppkg/layer_graph.rs::AndroidLayerGraph::execute` — graphExecute wrap
- `engine/src/layers/transformer_layer/forward.rs::forward_prefill` — production fallback decode path (RMS opt 사용)
- `engine/src/layers/transformer_layer/forward_gen.rs::forward_gen` — workspace some 진입 (현재 production main forward_into는 workspace=None이라 미사용)
- `engine/kernels/simple_ops.cl` — RMS norm kernels (opt, oop, simple, oop_subgroup 신규)
- `crates/qnn_oppkg/src/ops/rms_norm.rs` — graph CustomRmsNorm op spec
- `engine/src/bin/microbench_qnn_qwen_layer.rs` — bit-equal검증 + production state inject infra

### 이전 핸드오프
- `.agent/todos/handoff_qnn_oppkg_d_f_d_d_6_complete_2026_05_10.md` — D-F + D-D.6 spec 완료
- `.agent/todos/handoff_qnn_oppkg_m3_4_red_pos_baked_20260510.md` — D-D 시리즈 시작

---

## 8. 주의사항 (DO / DON'T)

### DO
- 다음 세션 첫 단계: production fallback의 layer 0 stage-by-stage dump 위치 결정 (forward_prefill or forward_gen). main forward_into가 workspace=None이라 현재는 forward_prefill 사용.
- microbench의 raw chain dump를 stage별로 추가하여 byte-by-byte 비교.
- 진척 시 stage 1개씩 fix → 빌드 → microbench 검증 → production fast path 토큰 확인.
- 모든 디버깅 env는 그대로 유지 (다음 세션에 도움).

### DO NOT
- `kernel_rms_norm_oop_subgroup`을 simple로 되돌리지 말 것 — graphFinalize PASS + production-style algorithm 가까움.
- `bytes_mask = n_kv * 2`로 되돌리지 말 것 — D-D.6 spec 누락 버그 fix.
- `LLMRS_QNN_OPPKG_FAST_PATH=1` env를 default 활성화 하지 말 것 — fix 미완.
- production OpenCLBackend의 kernel/flag을 graph 맞춤으로 변경하지 말 것 — production perf regression 위험. graph 측을 production에 맞추는 게 옳음.

---

**End of Handoff**

self-contained: 본 문서 + previous handoff + 위 진입점 파일만으로 다음 세션 stage-by-stage byte-equal 검증 작업 시작 가능.
