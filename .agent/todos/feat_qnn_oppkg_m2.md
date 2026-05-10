# Plan: QNN-GPU OpPackage Migration — M2 (Layer-level Graph)

> **작성일**: 2026-05-09
> **작성자**: PM (메인 세션 위임, plan_qnn_oppkg_migration_2026-05-09.md §3.M2 분해)
> **선행 완료**: M1 (`crates/qnn_oppkg/` production crate, 5 ops + abstraction layer + leak-safe free, HEAD `d930801`)
> **Phase R 결과**: `papers/eurosys2027/_workspace/experiment/qnn_phase_r_summary.md`
> **Device**: Galaxy S25 (R3CY408S4HN, Adreno 830 + Hexagon V79)

---

## 0. M2 Goal & Pass-gate

**Goal**: Qwen 1 layer (12-15 op) 를 단일 OpPackage graph로 build/execute.

**Pass-gate (M2 종료 조건)**:
- 5 신규 op 모두 individual correctness GREEN (max_abs_err < 1e-4 F32, < 1e-3 F16)
- Layer graph 1 layer 정확성 max_abs_err < 1e-2 (F16 tolerance) vs CPU NEON reference
- TBT ≤ baseline × 1.10 (성능 보존)
- graphFinalize ≤ 200 ms (1회성, app load time 영향 acceptable)
- production code 변경 0 lines (M3까지)
- 회귀 0 (`cargo test --workspace`, fmt, clippy)
- 새 spec ID 추가 시 추적성 PASS (`scripts/check_spec_coverage.sh`)

---

## 1. 작업 분해 (M2.A ~ M2.J)

각 단계 형식:
- subject (한 줄)
- description (목표 + 산출물)
- 의존성 (선행 단계 / 외부)
- 검증 게이트 (Pass/Fail)
- 회귀 위험 평가 (LOW / MEDIUM / HIGH)
- 추정 소요 (인-일)

---

### [P0] M2.A — Layer op sequence 분석 + spec 갱신
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음 (M1 완료가 전제)
- **Description**: production `LlamaLayer::forward()`(`engine/src/layers/llama_layer.rs`)와 `forward_gen()` 경로의 op sequence를 정적으로 분해. Qwen2.5-1.5b 기준 op DAG 도출:
  - RMSNorm → QKV matmul (Q4_0 GEMV) → RoPE → KvScatter (F32→F16 cast) → FlashAttn (F16/F32 mixed) → Output matmul → Add(residual) → RMSNorm → Gate matmul + Up matmul → SiluMul → Down matmul → Add(residual)
  - 총 12-15 op (Qwen GQA 6:1, head_dim=128, kv_heads=2)
  - 각 op의 input/output dtype, shape, intermediate tensor lifetime 표기
  - **산출물**: `arch/qnn_layer_graph.md` (Architect 영역, PM은 backlog 등록만)
- **검증 게이트**:
  - PASS: op DAG가 production 코드와 정합 (Architect review)
  - FAIL: op 누락/순서 오류 → reject
- **회귀 위험**: LOW (문서만)
- **추정 소요**: 1일
- **담당 권장**: Architect

---

### [P0] M2.B — CustomRoPE op wrap (가장 단순)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: M2.A (op sequence + tensor schema 확정)
- **Description**: `engine/kernels/simple_ops.cl::kernel_rope_opt`을 OpPackage `CustomRoPE`로 wrap.
  - args layout: `[q_in, k_in, q_out, k_out, sin_table, cos_table, head_dim, n_heads, n_kv_heads, pos]`
  - sin/cos table은 graph build time에 미리 계산해 DATA tensor로 주입
  - **산출물**:
    - `crates/qnn_oppkg/src/ops/rope.rs` (op def + register)
    - `crates/qnn_oppkg/tests/rope_correctness.rs` (host vs device unit)
    - `engine/src/bin/microbench_oppkg_rope_correctness.rs` (디바이스 bit-exact)
- **검증 게이트**:
  - PASS: max_abs_err < 1e-4 (F32 path), < 1e-3 (F16 path), 호스트 unit + S25 microbench 모두 GREEN
  - FAIL: bit-exact 미달 → kernel arg layout 디버깅
- **회귀 위험**: LOW (M1과 동일 패턴, in-place sin/cos 적용)
- **추정 소요**: 1.5일
- **담당 권장**: Implementer (sonnet) — M1 패턴 적용

---

### [P0] M2.C — CustomDeqQ40 op wrap (단순, fallback path)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: M2.A (Q4_0 block layout 확인)
- **Description**: `engine/kernels/cvt_q4_0_noshuffle_fused.cl` (또는 dequant-only 변형)을 OpPackage `CustomDeqQ40`으로 wrap.
  - 단독 op으로 wrap (chain 안에서 fallback로만 사용; M2.D MatMulQ40F32가 hot path)
  - 18B/block AOS layout (CUDA swap 호환)
  - **산출물**:
    - `crates/qnn_oppkg/src/ops/deq_q40.rs`
    - `crates/qnn_oppkg/tests/deq_q40_correctness.rs`
    - `engine/src/bin/microbench_oppkg_deq_q40_correctness.rs`
- **검증 게이트**:
  - PASS: F32 dequant 결과 max_abs_err < 1e-5 vs CPU dequant reference
  - FAIL: block scale 디코딩 오류 → AOS layout 재검토
- **회귀 위험**: LOW (단독 op, dequant arithmetic 직관적)
- **추정 소요**: 1.5일
- **담당 권장**: Implementer (sonnet)

---

### [P0] M2.D — CustomMatMulQ40F32 op wrap (production hot path)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: M2.C (dequant 정확성 검증 후), M2.A
- **Description**: `engine/kernels/mul_mv_q4_0_f32_8x_flat.cl` (Q4_0 GEMV)을 OpPackage `CustomMatMulQ40F32`로 wrap.
  - Qwen 16 layer × 7 matmul = 112 호출, 전체 forward의 dominant cost
  - block dequant + F32 accumulate inline
  - **산출물**:
    - `crates/qnn_oppkg/src/ops/matmul_q40_f32.rs`
    - `crates/qnn_oppkg/tests/matmul_q40_f32_correctness.rs`
    - `engine/src/bin/microbench_oppkg_matmul_q40_correctness.rs`
- **검증 게이트**:
  - PASS:
    - 정확성: max_abs_err < 1e-3 (Q4_0 quantization noise floor 감안) vs CPU NEON Q4_0 GEMV
    - 성능: per-op latency ≤ raw OpenCL `mul_mv_q4_0_f32_8x_flat` × 1.10
  - FAIL: 정확성 < 1e-2 → block scale 디코딩 검증; 성능 > 1.20× → kernel arg dispatch 디버깅
- **회귀 위험**: MEDIUM (block dequant + GEMV arithmetic 복잡, Adreno register usage 주의)
- **추정 소요**: 2.5일
- **담당 권장**: Senior Implementer (Adreno 최적화 + Q4_0 block 처리)

---

### [P1] M2.E — CustomKvScatter op wrap (F32 → F16 cast 포함)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: M2.A
- **Description**: `engine/kernels/kv_scatter_f32_to_f16.cl`을 OpPackage `CustomKvScatter`로 wrap.
  - HeadMajor `[1, kv_heads, capacity, head_dim]` 가정 (production 일치)
  - max-padded fixed shape: capacity=2048
  - F32 input → F16 cast + scatter to position offset
  - **산출물**:
    - `crates/qnn_oppkg/src/ops/kv_scatter.rs`
    - `crates/qnn_oppkg/tests/kv_scatter_correctness.rs`
    - `engine/src/bin/microbench_oppkg_kv_scatter_correctness.rs`
- **검증 게이트**:
  - PASS:
    - F16 cast 정확성: max_abs_err < 2^-10 (F16 epsilon)
    - HeadMajor offset 정합: head*capacity*head_dim + pos*head_dim 위치에만 write
  - FAIL: cast 정밀도 미달 → F16 conversion mode 검증; offset 오류 → stride 인자 재확인
- **회귀 위험**: MEDIUM (HeadMajor stride 하드코딩 가능성, INV-121 KV layout 의존)
- **추정 소요**: 2일
- **담당 권장**: Implementer (sonnet) — KV layout 이해 필요

---

### [P1] M2.F — CustomFlashAttn op wrap (F16/F32 mixed, 가장 위험)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: M2.E (KV scatter 정확성 후), M2.A
- **Description**: `engine/kernels/flash_attn_f32_f16.cl` (또는 production 사용 변형)을 OpPackage `CustomFlashAttn`으로 wrap.
  - online softmax 패턴 (max running + exp scale rebalance)
  - DK=128 (Qwen head_dim), Adreno register spill 한계 주의 (per-thread 32 float4 상한)
  - 입력: Q (F32), K/V cache (F16), 출력: O (F32)
  - **산출물**:
    - `crates/qnn_oppkg/src/ops/flash_attn.rs`
    - `crates/qnn_oppkg/tests/flash_attn_correctness.rs`
    - `engine/src/bin/microbench_oppkg_flash_attn_correctness.rs`
- **검증 게이트**:
  - PASS:
    - 정확성: max_abs_err < 1e-3 (online softmax bit-exact 어려움, F16 누적 오차 감안) vs CPU NEON `attention_gen_f16_neon` reference
    - GQA correctness: 6:1 query-to-KV head mapping 정합
  - FAIL: max_abs_err > 1e-2 → online softmax `m_i`/`l_i` 재계산 단계 검증
- **회귀 위험**: HIGH (online softmax + register spill + GQA mapping 동시; M2 최대 risk)
- **추정 소요**: 4일 (정확성 디버깅 buffer 포함)
- **담당 권장**: Senior Implementer (online softmax + Adreno 32-float4 register 한계)

---

### [P0] M2.G — SiluMul OOP refactor 결정 + 적용
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: M2.A
- **Description**: M1 backlog에서 식별된 SiluMul in-place 한계 해소. Layer graph chain 안에서 host-readable output 필요.
  - **3 옵션 trade-off**: 본 문서 §4 결정 트리 참조.
  - Architect에게 옵션 결정 escalate → 결정 후 적용.
  - **산출물 (옵션 결정 후)**:
    - 옵션 A 채택 시: `engine/kernels/simple_ops.cl`에 `kernel_silu_mul_oop` 추가 (Senior Implementer + .cl 정책 위반 명시 승인)
    - 옵션 B 채택 시: `crates/qnn_oppkg/src/ops/silu_mul.rs` 안 inline kernel 합성 (ENG-QNN-011 spec 영향 검토)
    - 옵션 C 채택 시: `crates/qnn_oppkg/src/ops/silu_oop.rs` + `elementwise_mul.rs` 분리, op set +1
  - 모든 옵션에서 host-readable F32 output 보장
- **검증 게이트**:
  - PASS: max_abs_err < 1e-4 vs CPU SiLU(x) ⊙ y; chain 안에서 read_buffer로 output 확인 가능
  - FAIL: in-place 잔존 → 옵션 재선정
- **회귀 위험**: MEDIUM (옵션 A는 production .cl 변경 → CLAUDE.md 정책 예외 승인 필요)
- **추정 소요**: 옵션 A 1.5일 / 옵션 B 2일 (spec 영향 검토 포함) / 옵션 C 2일 (op 1개 추가)
- **담당 권장**: Architect (옵션 결정) → Senior Implementer (옵션 A) 또는 Implementer (옵션 B/C)

---

### [P1] M2.H — Layer graph builder (인프라)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: M2.B, M2.C, M2.D, M2.E, M2.F, M2.G 모두 individual GREEN 후
- **Description**: M2.A에서 도출된 op DAG를 단일 OpPackage graph로 build하는 builder 작성.
  - input: `(x: F32[1, dim], kv_cache: F16[1, kv_heads, 2048, head_dim], pos: i32)`
  - output: `(y: F32[1, dim], updated_kv: in-place)`
  - intermediate tensor: NATIVE (graph-internal)
  - KV cache: graph 외부 별도 buffer (rpcmem-backed, M3에서 mmap 통합 검토)
  - max-padded fixed shape (R-A3 결과 반영)
  - graphFinalize 1회성 호출
  - **산출물**:
    - `crates/qnn_oppkg/src/graph/layer_builder.rs`
    - `engine/src/bin/microbench_qnn_qwen_layer_correctness.rs` (1 layer accuracy vs CPU NEON)
- **검증 게이트**:
  - PASS:
    - 1 layer accuracy: max_abs_err < 1e-2 (F16 tolerance) vs CPU NEON 1 layer reference
    - graphFinalize ≤ 200 ms (S25 wall-clock)
    - graph build/free leak 없음 (Box::leak + 명시 free 검증, M1 패턴)
  - FAIL: accuracy 미달 → 어느 op intermediate에서 누적 오차 발생하는지 isolation; finalize > 200 ms → graph 단순화
- **회귀 위험**: HIGH (intermediate tensor 연결 + KV cache binding + max-padded layout 동시)
- **추정 소요**: 5일
- **담당 권장**: Senior Implementer (graph composition + KV binding)

---

### [P1] M2.I — Layer-level TBT 측정 + 성능 게이트
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: M2.H (builder GREEN 후)
- **Description**: production OpenCL 1 layer TBT vs QNN OpPackage 1 layer TBT.
  - Reference baseline: Qwen2.5-1.5B Q4_0 GGUF (`qwen2.5-1.5b-q4_0-v2.gguf`), 6 threads, S25
  - 측정 방법: wall-clock (CL_QUEUE_PROFILING_ENABLE 금지, 엔진 간 비교 무효)
  - thermal isolation V10 protocol 적용
  - **산출물**:
    - `engine/src/bin/microbench_qnn_qwen_layer_tbt.rs` (이미 plan에 명시; M2.H의 correctness bin과 분리)
    - `papers/eurosys2027/_workspace/experiment/qnn_m2_layer_tbt.md` 측정 보고
- **검증 게이트**:
  - PASS: TBT ≤ raw OpenCL × 1.10 (10% regression 허용)
  - YELLOW (1.10~1.20×): optimization sub-task 등록 (M2 종결 보류, M3 진입 전 해소)
  - FAIL (> 1.20×): scope 재정의 (§6 fallback 참조)
- **회귀 위험**: MEDIUM (production code 변경 0이지만 결과가 fail 시 M2 전체 re-scope)
- **추정 소요**: 2일 (측정 + 분석)
- **담당 권장**: Tester

---

### [P2] M2.J — Spec ID 추가 + 추적성 검증
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: M2.B~M2.H 구현 완료, M2.I PASS 후
- **Description**: 5 신규 op + layer builder + KV binding에 대한 ENG-QNN spec ID 추가.
  - 예상: ENG-QNN-101 (CustomRoPE), 102 (CustomDeqQ40), 103 (CustomMatMulQ40F32), 104 (CustomKvScatter), 105 (CustomFlashAttn), 106 (CustomSiluMul OOP), 110 (LayerGraphBuilder), 111 (KvCacheBinding max-padded)
  - 신규 invariant: INV-156 (graph intermediate lifetime), INV-157 (KV max-padded shape 보존)
  - tests/spec/ 테스트 추가 (inline #[cfg(test)] 불가, feedback_spec_tests_required.md 준수)
  - **산출물**:
    - `spec/30-engine-overview.md` 또는 적절 spec 파일 갱신
    - `spec/41-invariants.md` INV 추가
    - `tests/spec/qnn_oppkg_m2_*.rs`
- **검증 게이트**:
  - PASS: `scripts/check_spec_coverage.sh` 통과, INV coverage 72% → 75%+
  - FAIL: spec ID 미참조 코드 잔존 → ID 부여 누락 식별
- **회귀 위험**: LOW (문서 + 테스트만)
- **추정 소요**: 2일
- **담당 권장**: Architect (spec ID + INV) → Tester (spec test)

---

## 2. 우선순위 + 학습 가치 순서

### 권장 실행 순서 (rationale: 단순한 것 먼저 → 위험한 것 마지막)

```
1. M2.A (1d, Architect)              — op DAG 분석, 모든 wrap의 schema 입력
2. M2.B (1.5d, Impl) + M2.C (1.5d, Impl) — RoPE + DeqQ40 병렬, M1 패턴 검증
3. M2.G 옵션 결정 (Architect, 0.5d)  — Architect escalate 비동기 진행 (B/C 동안)
4. M2.D (2.5d, Senior Impl)          — Q4_0 GEMV (production hot path, B/C 후 dequant 검증된 상태에서)
5. M2.E (2d, Impl) + M2.G 적용 (Senior Impl/Impl, 1.5~2d) — KV scatter + SiluMul 병렬
6. M2.F (4d, Senior Impl)            — FlashAttn (가장 위험, 정확성 디버깅 buffer 포함)
7. M2.H (5d, Senior Impl)            — Layer graph builder (모든 op GREEN 후)
8. M2.I (2d, Tester)                 — TBT 측정 (게이트)
9. M2.J (2d, Architect + Tester)     — Spec + 추적성 (병렬 진행 가능, M2.B~H 끝나는 대로)
```

**학습 가치 정렬**:
- 단순 op (B, C) 먼저 → M1 패턴 재현 + arg layout abstraction 검증 (위험 분산)
- 어려운 op (F, FlashAttn) 마지막 → online softmax 디버깅이 schedule 견인하지 않게
- M2.D는 hot path지만 dequant 단독(C)이 GREEN인 후 진입 → Q4_0 arithmetic 검증된 상태에서 GEMV에 집중

### Fail-fast 정책

| 단계 | RED 시 fallback |
|------|-----------------|
| M2.B | RoPE bit-exact 미달 → kernel args dispatch 검증; M1 abstraction 한계 발견 시 schema 보강 후 재시도 |
| M2.C | DeqQ40 정확성 미달 → AOS 18B/block layout 재확인 (Phase 6 G-1-F fix 참조), 1일 timebox |
| M2.D | 성능 > 1.20× → M2.D를 hot path 외 fallback path로 격하; M2.H에서 Q4_0 dequant + F16 GEMV 분리 path 검토 |
| M2.E | HeadMajor offset 오류 → INV-121 명시 stride parameter pass-through로 수정 |
| M2.F | online softmax accuracy 미달 → §6 fallback 옵션 1 (FlashAttn 미wrap, raw OpenCL kernel을 graph 안 별도 op으로 동봉) |
| M2.G | 옵션 A 거부 시 → 옵션 C 자동 채택 (op set +1, 일정 +0.5d) |
| M2.H | accuracy 미달 → op isolation: Q→QKV→...→FFN 순으로 forward 부분 그래프 단계적 검증 |
| M2.I | TBT > 1.20× → M3 진입 보류, optimization sprint (graph build 최적화, op fusion 검토) |

---

## 3. 의존성 그래프

```
                                M2.A (op DAG)
                                  │
                ┌─────────────────┼──────────────────┐
                ▼                 ▼                  ▼
              M2.B            M2.C  ────────►  M2.D (Q4_0 GEMV)
            (RoPE)         (DeqQ40)            │
                │                │             │
                │                │             │
                └────────┬───────┴─────────────┘
                         │
                         ▼
              ┌──────────┴──────────┐
              ▼                     ▼
            M2.E                  M2.G
         (KvScatter)            (SiluMul OOP)
              │                     │
              ▼                     │
            M2.F                    │
         (FlashAttn)                │
              │                     │
              └──────────┬──────────┘
                         ▼
                       M2.H
                  (Layer Graph)
                         │
                         ▼
                       M2.I
                    (TBT 측정)
                         │
                         ▼
                       M2.J
                  (Spec 추적성)
```

**병렬 가능 단계**:
- M2.B + M2.C (RoPE + DeqQ40 동시 가능)
- M2.D는 M2.C 의존 (dequant 검증 후)
- M2.E + M2.G (KvScatter + SiluMul 동시)
- M2.G의 Architect 옵션 결정은 M2.B 시작과 비동기 진행
- M2.J spec 작업은 M2.B~H 산출물이 안정되는 대로 시작

**Critical path**: M2.A → M2.C → M2.D → M2.E → M2.F → M2.H → M2.I (≈ 18일 wall-clock, 병렬 가정)

---

## 4. SiluMul OOP refactor 결정 트리

### Trade-off 표

| 옵션 | 장점 | 단점 | 정책 영향 |
|------|------|------|-----------|
| **A. production .cl에 `kernel_silu_mul_oop` 추가** | 가장 깔끔한 GPU 경로, host-readable output, 1 op | CLAUDE.md "engine/kernels/* 수정 회피" 정책 예외 필요 | Senior Implementer 승인 (성능 최적화 작업 분류 적용 가능, .cl 정책에 OOP는 wrap 인프라 작업) |
| **B. OpPackage 안 inline kernel composition** | production .cl 변경 0 | OpPackage가 kernel source string을 직접 합성 → ENG-QNN-011 (op = single .cl wrap) 의도 위반 가능 | spec 영향 검토 필수, Architect 판단 |
| **C. SiluMul → silu_oop + elementwise_mul_oop 2 op 분해** | 정책 위반 0, op set +1로 abstraction 깔끔 | intermediate tensor 1개 추가 (메모리/대역폭 +) → M2.I TBT 영향 가능 | 가장 안전, 일정 +0.5d (op 1개 추가) |

### Architect escalate 질문 정리

PM이 Architect에게 전달할 결정 항목:

1. **CLAUDE.md `.cl` 수정 정책의 예외 적용 범위**: M2 wrap 인프라 작업이 "성능 최적화 작업(Senior Implementer)"의 정의에 포함되는가? 또는 production .cl 변경 0 정책을 M2 동안 strict 유지해야 하는가?
2. **ENG-QNN-011 (op = single .cl wrap) 의도**: 옵션 B의 inline kernel composition이 spec 위반인가, 또는 wrap 추상화의 자연스러운 확장으로 인정 가능한가?
3. **M2.I TBT 게이트 영향**: 옵션 C가 intermediate tensor 추가로 TBT × 1.10 게이트를 넘길 위험이 있는가? (graph 안에서 NATIVE tensor라 host로 read하지 않으나 GPU mem write/read 추가)
4. **M3 forward integration 영향**: 옵션 A는 M3에서도 그대로 사용 가능. 옵션 B/C는 M3에서 별도 처리 필요한가?

PM 추천: **옵션 C (가장 안전, 정책 위반 0)** 우선 검토. Architect가 옵션 A를 승인 시 옵션 A 채택 (성능 면에서 가장 깔끔). 옵션 B는 spec risk 때문에 회피 권장.

---

## 5. M2 종료 조건 (요약)

- [ ] 5 신규 op (RoPE, DeqQ40, MatMulQ40F32, KvScatter, FlashAttn) + SiluMul OOP 모두 individual correctness GREEN
- [ ] Layer graph builder 1 layer 정확성 max_abs_err < 1e-2 (F16 tolerance) vs CPU NEON
- [ ] 1 layer TBT ≤ raw OpenCL × 1.10
- [ ] graphFinalize ≤ 200 ms (S25 wall-clock)
- [ ] production code 변경 0 lines (M3까지)
- [ ] 회귀 0: `cargo test --workspace`, fmt, clippy 모두 GREEN
- [ ] 새 spec ID (ENG-QNN-101~111 + INV-156, 157) 추가, 추적성 PASS, INV coverage 72% → 75%+
- [ ] 산출물 보고서 1건: `papers/eurosys2027/_workspace/experiment/qnn_m2_layer_tbt.md`

---

## 6. 실패 시 fallback

### M2.F (FlashAttn) RED 시
- **옵션 1**: FlashAttn을 OpPackage에 wrap하지 않고, raw `flash_attention_forward_strided()` OpenCL kernel을 graph build 시 별도 op으로 외부 enqueue (cl_mem 외부 공유 RED 검증 결과 위반 → 사실상 불가). **이 옵션은 폐기.**
- **옵션 2 (실효)**: M2 scope를 "FlashAttn 제외 attention path"로 재정의. CustomAttnScores + CustomAttnGen 분리 wrap (Phase R Wave 1에서 attention_scores.cl 매핑 GREEN 확인됨). 일정 +3일.
- **옵션 3**: M2를 "12 op 중 hot path 5 op만 wrap"으로 축소, FlashAttn은 M2.5 (별도 sprint)로 격리.

### M2.I (TBT 측정) YELLOW (1.10~1.20×) 시
- M3 진입 보류 → optimization sub-task: graph build cost 분석, op fusion 가능성 (RMSNorm + matmul fuse 등), KV binding cost 측정.
- 최대 1주 timebox, 그 후 M3 진입 결정.

### M2.I FAIL (> 1.20×) 시
- M2 scope 재정의:
  - **부분 wrap (hot path만)**: Q4_0 GEMV + FlashAttn + RMSNorm 3-op만 wrap, 나머지는 OpenCL 유지 → cl_mem 외부 공유 RED 위배 → 폐기
  - **scope 재정의**: layer-level → "single hot op chain" (예: QKV + RoPE + KvScatter 3-op chain)으로 축소. M3 plan 전반 재검토 필요.

### M2.G (SiluMul) Architect 결정 지연 시
- M2.B/C/D/E 진행은 무관 (병렬). M2.H 진입 직전까지 결정 buffer.
- 1주 이내 미결 시 PM이 옵션 C 자동 채택 (가장 안전).

---

## 7. 외부 참조

### Plan / Handoff
- `.agent/todos/plan_qnn_oppkg_migration_2026-05-09.md` (5-phase plan, M2 §3 참조)
- `.agent/todos/handoff_qnn_oppkg_production_2026-05-09.md` (Phase R 결과 + M1 진입 가이드)
- `papers/eurosys2027/_workspace/experiment/qnn_phase_r_summary.md` (Phase R 종합)

### Code 진입점 (M2 신규/수정 예상)
- `crates/qnn_oppkg/src/ops/{rope,deq_q40,matmul_q40_f32,kv_scatter,flash_attn,silu_mul}.rs` (신규)
- `crates/qnn_oppkg/src/graph/layer_builder.rs` (신규)
- `engine/src/bin/microbench_oppkg_*.rs` (신규 6 bin: 5 op + layer correctness/tbt)
- `arch/qnn_layer_graph.md` (신규, Architect)
- `spec/30-engine-overview.md`, `spec/41-invariants.md` (수정)
- `tests/spec/qnn_oppkg_m2_*.rs` (신규)

### Reference (참조만)
- `engine/src/layers/llama_layer.rs::forward()` / `forward_gen()` (op DAG 출처)
- `engine/kernels/{simple_ops,cvt_q4_0_noshuffle_fused,mul_mv_q4_0_f32_8x_flat,kv_scatter_f32_to_f16,flash_attn_f32_f16}.cl` (wrap 대상)
- `crates/qnn_oppkg/src/lib.rs` (M1 abstraction layer, 본 plan에서 재활용)

---

## 8. PM 발급 backlog 항목 (역할별 등록)

본 plan의 작업을 역할별 TODO로 분배:

| 단계 | 담당 | 등록 위치 |
|------|------|-----------|
| M2.A | Architect | `architect.md` |
| M2.B, M2.C, M2.E, M2.J(test) | Implementer (sonnet) | `rust_developer.md` |
| M2.D, M2.F, M2.H, M2.G(옵션 A/C 적용) | Senior Implementer | `rust_developer.md` (별도 섹션) |
| M2.G (옵션 결정), M2.J (spec) | Architect | `architect.md` |
| M2.I | Tester | `tester.md` |

PM이 본 plan 승인 후 메인 세션이 위 TODO 파일에 항목을 추가한다 (PM은 plan 작성만 담당, 실제 분배는 메인 세션 오케스트레이션 영역).

---

**End of M2 Plan**

총 추정 wall-clock: **18~22일** (병렬 가정, FlashAttn 디버깅 buffer 4일 포함). 사용자 승인 후 M2.A 진입.
