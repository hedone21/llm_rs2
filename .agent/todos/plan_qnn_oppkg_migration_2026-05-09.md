# Plan: QNN-GPU OpPackage Migration

**작성일**: 2026-05-09
**선행**: `papers/eurosys2027/_workspace/experiment/qnn_phase_r_summary.md`
**기준 commit**: `f864262`
**Device**: Galaxy S25 (Adreno 830 + Hexagon V79)

---

## 0. 목표

production OpenCL forward kernel을 QNN-GPU OpPackage로 wrap하여:
1. **forward 성능 보존** (Adreno-tuned `mul_mv_f16_f32` 등 그대로 사용)
2. **phase-aware async weight swap** 활용 (cache-fit phase에 sync-free chunk swap)
3. KV cache는 max-padded fixed shape (R-A3에 따라)

---

## 1. Phase R 검증 결과 요약 (이미 완료)

| Risk | Verdict | 영향 |
|------|---------|------|
| Forward (OpPackage chain ≥16) | GREEN | raw OpenCL 12% 능가 |
| 정확성 | GREEN | bit-identical |
| Mixed-op graph | GREEN | per-op 0.035 ms (homogeneous 대비 29% 빠름) |
| KV dynamic shape | YELLOW | max-padded fixed shape 전략 필수 |
| Q4_0 호환 | GREEN (조건부) | SDK API 또는 OpPackage wrap path 명확 |
| Async swap (phase-aware) | GREEN | cache-fit phase에 1.04× of max |
| cl_mem 외부 공유 | RED | 모든 forward op이 OpPackage graph 안에 있어야 |

---

## 2. Migration Scope

### 2.1 In-scope (필수)

#### Forward path (graph 안)
- `mul_mv_f16_f32.cl` (production GEMV) → OpPackage `CustomMatMulF16F32`
- `mul_mv_f32_f32.cl` → `CustomMatMulF32F32`
- `mul_mv_q4_0_f32_8x_flat.cl` (Q4_0 GEMV) → `CustomMatMulQ40F32`
- `flash_attn_f32_f16.cl` → `CustomFlashAttnF32F16`
- `attention_scores.cl` → `CustomAttnScores` (또는 graph composition)
- `simple_ops.cl` 안 핵심:
  - `kernel_rms_norm_opt` → `CustomRmsNorm`
  - `kernel_silu_mul_opt` → `CustomSiluMul` (SwiGLU)
  - `kernel_softmax_opt` → `CustomSoftmax`
  - `kernel_rope_opt` → `CustomRope`
  - `kernel_add_assign_opt` → 이미 PoC 완료 (`CustomAdd` 패턴)
  - `kernel_attn_gen` / `kernel_attn_gen_half` → `CustomAttnGen`
- `cvt_q4_0_noshuffle_fused.cl` → `CustomDeqQ40` (R-A4 fallback path)
- `get_rows.cl` → QNN_OP_GATHER (prebuilt)

#### KV cache management
- max-padded layout: `[1, kv_heads, max_seq=2048, head_dim]` 고정 dim
- attention mask로 valid range 처리
- update path: `kv_scatter_f32_to_f16.cl` → `CustomKvScatter`

#### Async swap
- Phase-aware scheduler: forward 분석 → DDR-heavy / cache-fit phase 구분
- chunk swap dispatcher (4-8 MB 권장 sweet spot)
- weight buffer pool (rpcmem-backed, MEMHANDLE swap 지원)

### 2.2 Out-of-scope (후속 작업)

- KIVI quant attention (`kivi_attn.cl`, `kivi_q2.cl`) — 프로토타입, 본 plan 후
- MoE (`mul_mv_id_*`, `gemm_moe_*`) — Qwen2.5-1.5b는 dense
- Multi-device (Pixel/Jetson은 OpenCL backend 유지)
- HTP+QNN-GPU shared rpcmem (R-Y opt 3) — HTP unstable + 추가 트랙

---

## 3. 5-Phase 실행 계획

### Phase M1 — Production OpPackage 인프라 (1~2주)

**목표**: 우리 OpPackage cdylib을 production crate로 격상.

#### M1.1 OpPackage crate 정식화
- `crates/qnn_oppkg/` (poc → production rename)
- Op registration 시스템: 각 .cl kernel을 op_type으로 등록하는 helper
  - 매크로 `register_op!(name, source, args_layout)` 같은 pattern
  - createOpImpl에서 op_type별 dispatch (현재는 if-else; 더 큰 op set이면 hashtable)
- args layout abstraction (TENSOR/DATA/LOCAL kind, 각 kernel별 schema)
- build.rs에서 production .cl을 자동 include

#### M1.2 핵심 op 5개 wrap (smoke test)
- `CustomMatMulF16F32` (이미 PoC 검증됨, 정식화)
- `CustomRmsNorm` (간단 op, smoke test)
- `CustomSoftmax`
- `CustomSiluMul`
- `CustomAdd` (이미 PoC)

각 op:
- args layout
- memory object dtype/shape
- workgroup size
- 정확성 unit test (vs raw OpenCL kernel 결과)

#### Verification
- `cargo test -p qnn_oppkg` (정확성 unit test)
- `microbench_oppkg_op_correctness.rs` (각 op host vs device 비교)

#### Pass-gate
- 5개 op 모두 max_abs_err < 1e-4 (F32)
- 빌드 + push + register OK
- Production code 변경 0 lines (microbench-only)

---

### Phase M2 — Layer-level Graph (2~3주)

**목표**: Qwen 1 layer (12-15 op) 를 단일 OpPackage graph로 build/execute.

#### M2.1 추가 op wrap
- `CustomMatMulQ40F32` (Q4_0 GEMV — production hot path)
- `CustomFlashAttn` (F16/F32 mixed)
- `CustomRope`
- `CustomKvScatter` (F32 → F16 KV cache update with cast)
- `CustomDeqQ40` (Q4_0 dequant fallback)

#### M2.2 Layer graph builder
- production `LlamaLayer::forward()` → QNN graph 생성
- input: `(x, kv_cache, pos)` → output: `(y, updated_kv)`
- 1 layer = 단일 graph
- KV cache: max-padded [1, kv_heads, 2048, head_dim], offset param

#### M2.3 Layer-level 측정
- `microbench_qnn_qwen_layer.rs`
- production OpenCL 1 layer TBT vs QNN graph 1 layer TBT
- Pass: ≤ 1.10× (성능 보존)

#### Pass-gate
- 1 layer accuracy: vs CPU NEON reference, max_abs_err < 1e-2 (F16 tolerance)
- TBT ≤ baseline × 1.10 (성능 보존)
- graphFinalize ≤ 200 ms (1회성, app load time 영향 acceptable)

---

### Phase M3 — Full-model Graph + Production Integration (3~4주)

**목표**: 16 layer를 통합 또는 layer-pool 방식으로 production binary에 통합.

#### M3.1 Backend abstraction 확장
- `engine/src/backend/qnn_oppkg/` 디렉토리 추가
- `Backend` trait 구현 (CPU/OpenCL/CUDA와 동등)
- forward dispatch path: 기존 model.forward() → backend.forward()

#### M3.2 KV cache + weight 관리
- KV cache: graph 외부 별도 rpcmem alloc (MEMHANDLE 등록)
- weight: 16 layer × ~90 MB = 1.4 GB (Q4_0 시 더 작음)
- weight swap pool (Phase 6.5 path 호환)

#### M3.3 generate.rs 통합
- `--backend qnn_oppkg` flag 추가 (default off, opt-in)
- existing OpenCL backend 그대로 유지 (regression 방지)

#### Pass-gate
- Qwen2.5-1.5b 32k token decode (greedy) 정확성: 기존 OpenCL backend와 token sequence 100% 일치
- TBT: ≤ baseline × 1.10
- VmRSS: ≤ baseline × 1.10 (메모리 약간 증가 max-padded KV)

---

### Phase M4 — Phase-aware Async Swap (2~3주)

**목표**: layer N forward 동안 layer N+1 weight 일부 preload.

#### M4.1 Forward phase analyzer
- 각 layer의 op sequence를 정적 분석
- DDR-heavy (large matmul) vs cache-fit (RMSNorm, RoPE, SwiGLU) 분류
- swap 가능 시간 window 식별

#### M4.2 Chunk swap dispatcher
- weight tensor를 chunk (4-8 MB)로 분할
- swap thread: cache-fit phase 시작 시 chunk dispatch
- main thread: DDR-heavy phase 시작 직전 swap chunk wait

#### M4.3 측정 + 튜닝
- `microbench_qnn_async_swap_e2e.rs`
- chunk size sweep (1, 2, 4, 8, 16 MB)
- per-layer swap hide ratio 측정
- production TBT 영향

#### Pass-gate
- swap pause time / forward time ≤ 80% (적어도 20% hide)
- swap 활성/비활성 정확성 동일

---

### Phase M5 — Manager/Resilience 통합 + Production 검증 (1~2주)

**목표**: 전체 시스템 통합 + 회귀 검증.

#### M5.1 Manager IPC
- ResilienceAction → QNN backend 호환 (eviction signal, throttle 등)
- D2O / sliding eviction가 max-padded KV에서 작동

#### M5.2 verify v2 harness 통합
- `--backend qnn_oppkg`로 모든 verify scenario 통과

#### M5.3 Production 회귀
- existing test suite (cargo test --workspace) 100% 통과
- spec 추적성 (`scripts/check_spec_coverage.sh`) 유지

#### Pass-gate
- All verify v2 scenarios PASS
- Spec coverage 유지
- TBT ≥ baseline (regression 없음)

---

## 4. 의존성 그래프

```
M1 (op infra + 5 ops)
   ↓
M2 (10+ ops, layer graph)  ← Phase R Q4_0 SDK 실측 (병행)
   ↓
M3 (full-model + backend integration)
   ↓
M4 (async swap)
   ↓
M5 (verify + spec)
```

총 9~14주 (1.5~3개월). 각 phase fail-fast 시 re-scope.

---

## 5. Critical Files (예상 변경)

### 신규
- `crates/qnn_oppkg/` (production crate)
- `engine/src/backend/qnn_oppkg/mod.rs` (Backend trait 구현)
- `engine/src/backend/qnn_oppkg/graph.rs` (layer graph builder)
- `engine/src/backend/qnn_oppkg/swap.rs` (phase-aware async swap)
- `engine/src/bin/microbench_qnn_qwen_layer.rs` (Layer 측정)
- `engine/src/bin/microbench_qnn_async_swap_e2e.rs`

### 수정
- `engine/Cargo.toml` (qnn_oppkg dep 추가)
- `engine/src/backend/mod.rs` (Backend enum 확장)
- `engine/src/bin/generate.rs` (`--backend qnn_oppkg` flag)
- `engine/src/models/llama/llama_model.rs` (backend dispatch)

### 회귀 위험
- M1, M2: **LOW** (microbench-only)
- M3: **MEDIUM** (production code 변경, 단 default off)
- M4: **MEDIUM** (async path, race 가능)
- M5: **LOW** (verify only)

---

## 6. 결정 트리 (각 phase 종료 시)

```
M1 PASS? → M2 진입
M1 FAIL → op infra 재설계 / 종결

M2 PASS (TBT ≤1.10×)? → M3 진입
M2 YELLOW (1.10~1.20×) → optimization sub-task
M2 RED (>1.20×) → scope 재정의 (부분 wrap, hot path만)

M3 PASS (정확성 100%, TBT 보존)? → M4
M3 정확성 FAIL → kernel/dispatch 디버깅
M3 TBT regression → M2 결과 검토, optimization

M4 PASS (≥20% hide)? → M5
M4 FAIL (no measurable hide) → swap path 재검토 (chunk size, scheduling)

M5 PASS → 마이그레이션 완료
M5 verify FAIL → 종합 검토, 부분 rollback
```

---

## 7. 산출물 (Phase별)

| Phase | 산출물 |
|-------|--------|
| M1 | `crates/qnn_oppkg/` cdylib + 5 op + unit tests |
| M2 | layer graph builder + 10+ op + `microbench_qnn_qwen_layer` 측정 보고서 |
| M3 | `engine/src/backend/qnn_oppkg/` + `--backend qnn_oppkg` + e2e 정확성 |
| M4 | phase-aware swap scheduler + e2e TBT 측정 |
| M5 | verify v2 PASS + spec coverage + paper section |

---

## 8. Out of Scope (다음 plan)

- Multi-device support (S25 only로 시작)
- HTP+QNN-GPU heterogeneous (R-Y opt 3 path)
- KIVI / MoE op wrap
- TFLite Delegate path

---

**End of Migration Plan**

다음 액션: M1 시작 — `crates/qnn_oppkg/` production crate 정식화.
