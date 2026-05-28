# Sprint: Qwen 2.5-1.5B Full Microbench Matrix

**작성**: 2026-05-28
**상위 sprint**: μ-Q1 4-cell GREEN (`handoff_qnn_microbench_phase_e_complete_2026_05_26.md`)
**연관 plan**: `plan_qnn_microbench_measurement_protocol_2026_05_26.md` (측정 protocol 재사용)
**디바이스**: Galaxy S25 (`R3CY408S5SB`, SoC SM8750, V79 Hexagon + Adreno 830, 6T CPU `taskset 0x3f`)
**진입 문장**: "Full microbench matrix sprint P0 진행"

---

## TL;DR

μ-Q1 4-cell paper-grade conclusion (M3/M4/M6b/M7 GREEN) 종결 후,
paper main evidence 통합 매트릭스로 확장. **12 op × 7 backend × 2 dtype = 168 cell + MUL_MAT extra 2 shape × 7 × 2 = 28 cell = 총 196 cell** (P0 사용자 결정 2026-05-28: SWIGLU 제외 + MUL_MAT 3 shape).
단일 S25 디바이스, thermal-isolated protocol (warmup 3 + measure 10, round-robin shuffle,
8-zone polling, Tukey 1.5×IQR). Tier-A 8 op 우선 측정.

**P0 완료 2026-05-28 (Architect)**: `papers/.../microbench_op_matrix_2026_05_26/matrix.md` v2 섹션 + `matrix_support_table.md` v2 섹션 작성. PM 보류 4건 모두 Architect 결정. 측정 활성 cell = 94 cell (`—`/`✗`/`⚠` 102 제외). 1일 cap 안에 흡수 가능.

**Phase 분해**:

| Phase | Owner | 산출물 | 의존성 | 상태 |
|---|---|---|---|---|
| **P0** 매트릭스 디자인 v2 | Architect | `matrix.md` v2 + cell inventory + tolerance table | 없음 | ✅ 완료 2026-05-28 |
| **P1a** ARM64Neon CPU bin (12 op × 2 dtype, `—` 제외 ≈ 24 cell) | Implementer | `engine/microbench/neon_op_matrix.rs` 또는 sub-bin 세트 | P0 | TODO |
| **P1b** OpenCL GPU bin (12 op × 2 dtype, M3/M5 inherit + 신규) | Implementer | `engine/microbench/opencl_op_matrix.rs` (또는 기존 확장) | P0 | TODO |
| **P1c** ExecuTorch PTE sweep | Implementer | 12 op × {`use_fp16`, `use_8a4w`} = 24 .pte + builder. **W8A8 → W4A8 재빌드** | P0 | TODO |
| **P1d** Ours-NPU HTP F16 dispatch path | Senior Implementer | `engine/src/backend/htp_fastrpc/` F16 host dispatch (Q4_0 fork). **IDL 변경 0**. mm_lmh + mm_qkv K/V 256 dim gate | P0 | TODO |
| **P2** L.cpp `test-backend-ops` driver | Implementer | `scripts/llamacpp_backend_ops_driver.py` (`-b CPU/GPUOpenCL/HTP0`). 경로: **`/home/go/Workspace/llama.cpp/build-snapdragon/bin/test-backend-ops`** (third_party 아님) | P0 cell inventory | TODO |
| **P3** `microbench_qnn_matrix.py` 7-col 확장 | Implementer | 7-col × 12-op × 2-dtype + MM extra = 196 cell 측정 driver (`—`/`✗`/`⚠` skip 마킹 aware) | P1a/b/c/d + P2 | TODO |
| **P4** S25 측정 + 통합 보고서 | Tester | `report.md` (Phase E format 확장) + raw/aggregated/thermal_log. 측정 활성 = **94 cell** | P3 | TODO |

병렬 가능성:
- P1a/P1b/P1c/P1d 4 트랙은 P0 완료 후 **동시 진행 가능** (Implementer 3명 + Senior 1명 분담)
- P2는 P0 cell inventory 결정 후 시작 권장 (Ours bin과 독립이므로 P1과 병렬)

---

## 측정 매트릭스 (확정)

### Op 13개 (Tier 분류)

| Tier | Op | 사용처 (Qwen 2.5-1.5B) | 우선순위 |
|---|---|---|---|
| **A** | `MUL_MAT` | QKV proj, Output proj, FFN gate/up/down, LM head | hot |
| **A** | `RMS_NORM` | Pre-attn, Pre-FFN | hot |
| **A** | `ROPE` | Q/K rotation | hot |
| **A** | `FLASH_ATTN_EXT` | Attention | hot |
| **A** | `GET_ROWS` | Embedding lookup | hot |
| **A** | `SILU` | FFN activation | hot |
| **A** | `MUL` | FFN gate × up elementwise | hot |
| **A** | `ADD` | Residual | hot |
| **B** | `SOFT_MAX` | (fallback when not fused into flash_attn) | warm |
| **B** | `SCALE` | Q × 1/√d_k | warm |
| **B** | `CPY` | KV cache write | warm |
| **D** | `SET_ROWS` | KV cache scatter (HeadMajor) | warm |
| **D** | `SWIGLU` | (fused SILU+MUL when backend supports) | warm |

### Backend 7개

| Backend | 약어 | 자산 위치 | 비고 |
|---|---|---|---|
| ARM64Neon CPU (Ours) | `ours.cpu` | `engine/src/backend/cpu/` | 6T pinning |
| OpenCL raw GPU (Ours) | `ours.gpu` | `engine/src/backend/opencl/`, `engine/kernels/*.cl` | Adreno 830 |
| ExecuTorch HTP NPU | `et.htp` | `/home/go/Workspace/executorch/.venv` + PT2E | `use_fp16` / `use_8a4w` |
| Ours-NPU HTP FastRPC | `ours.htp` | `engine/src/backend/htp_fastrpc/` | Q4_0 path 존재, F16 fork 필요 (P1d) |
| L.cpp.CPU | `lcpp.cpu` | `third_party/llama.cpp/build-snapdragon/bin/test-backend-ops` | `-b CPU` |
| L.cpp.GPU Adreno | `lcpp.gpu` | (동일) | `-b Adreno` |
| L.cpp.HTP0 Hexagon | `lcpp.htp` | (동일) | `-b HTP0` |

### Dtype 2개

| Dtype | Ours / L.cpp | ExecuTorch |
|---|---|---|
| F16 | F16 native | `use_fp16=True` |
| Q4_0 | Q4_0 block-32 | `use_8a4w` (W4A8 PT2E) — fair to Ours-NPU dispatch |

### Shape (Qwen 2.5-1.5B actual)

- hidden=1536, n_heads=12, n_kv=2 (GQA 6:1), head_dim=128, FFN=8960
- vocab=151936, n_layers=28, max_seq=production hot path 1 (decode) + prefill rep

Per-op shape 매핑은 P0 산출물 `matrix.md` v2 에서 cell inventory로 확정.

---

## [P0] 매트릭스 디자인 v2 (Architect) ✅ 완료 2026-05-28

- **Status**: ✅ DONE 2026-05-28
- **Sprint**: current
- **Owner**: Architect
- **Dependencies**: 없음 (μ-Q1 4-cell GREEN handoff + plan_qnn_microbench_measurement_protocol)
- **Entry**: `.agent/todos/handoff_microbench_full_matrix_phaseP0_2026_05_28.md`
- **산출물 (commit 대기)**:
  - `papers/.../microbench_op_matrix_2026_05_26/matrix.md` — v1 deprecation 노트 + **v2 섹션** (12-§1~§12)
  - `papers/.../microbench_op_matrix_2026_05_26/matrix_support_table.md` — v1 deprecation 노트 + **v2 7-backend 196 cell support map**
  - sprint master TODO 갱신 (본 파일)
- **Acceptance Criteria** (모두 충족):
  - [x] 7 backend 정의 명확 (path + 측정 binary 매핑)
  - [x] Op 12개 inventory (Tier-A 8 + Tier-B 4, SWIGLU 제외)
  - [x] MUL_MAT 3 shape 확정 (mm_ffn / mm_lmh / mm_qkv fused)
  - [x] 196 cell status (`✓` 11 + `+` 83 + `⚠` 10 + `✗` 12 + `—` 80)
  - [x] tolerance 표 (F16/Q4_0/W4A8 base + 누적 op override)
  - [x] PM 보류 4건 Architect 결정 (§5.1~§5.5) — Tier-B fair-pair, SWIGLU 제외, shape sweep, MUL_MAT QKV fused, UNSUPPORTED 5-tier 마킹
  - [x] 의존성 검증 (test-backend-ops 경로 정정, IDL F16 이미 존재, ExecuTorch venv 미확인)
  - [x] 추가 리스크 8건 표면화
- **PM 결정 보류 4건 → Architect 결정 결과**:
  1. **Tier-B fair-pair grouping**: SOFT_MAX/SCALE 은 F16 row 만 (activation-only), CPY/SET_ROWS 도 F16 row 만 (dtype-only/scatter-only). Q4_0 row 모두 `—` 마킹.
  2. **SWIGLU vs SILU+MUL**: **SWIGLU 제외** (Qwen 미사용 + Ours/ExecuTorch 미구현 → fair-pair 성립 안 함). backlog 'Future-fused MUL+SILU' 등록 권장.
  3. **per-op shape sweep**: MUL_MAT 만 3 shape (mm_ffn/mm_lmh/mm_qkv). 다른 op 모두 single shape (FLASH_ATTN seq=1024 fixed, GET_ROWS vocab=151936 single 등). 다른 sweep 은 backlog.
  4. **MUL_MAT QKV 분리 vs fused**: **fused 측정 (N=2048 합계)**. paper figure 단순화 + ExecuTorch unified linear export 호환. P1d 가 K/V N=256 dim shape dispatch gate 책임.
  5. **UNSUPPORTED 마킹 규칙**: 5-tier (`✓` existing / `+` new / `⚠` perf abort / `✗` hard 미지원 / `—` fair 부적합)
- **Notes**: 본 P0 sprint cap 1d (Architect 단독). 실제 약 0.5d 소요 (P1 진입 즉시 가능).

---

## [P1a] ARM64Neon CPU microbench bin (Implementer)

- **Status**: TODO
- **Sprint**: current
- **Owner**: Implementer
- **Dependencies**: P0 cell inventory
- **Description**:
  - 13 op × 2 dtype Ours CPU NEON 측정 bin
  - 단일 bin per op (총 13개) 또는 한 bin이 op 선택 인자 받는 형태 — P0 결정 따름
  - 6T pinning (`taskset 0x3f`), `--threads 6`
  - 기존 NEON kernel 재사용 (`engine/src/backend/cpu/neon.rs`, `kivi_cache.rs` 등)
- **Acceptance Criteria**:
  - 호스트 빌드 PASS + `--features cpu_only` (또는 default) 모두 OK
  - dry-run 1 cell GREEN (e.g., `MUL_MAT F16` Qwen FFN gate shape)
  - 출력: trial 별 raw JSON (Phase E format 호환)
- **Notes**: Tier-A 8 op 먼저 구현 → Tier-B/D는 후속. 1.5~2일.

---

## [P1b] OpenCL Q4_0 latency bin = M5 cell (Implementer)

- **Status**: TODO
- **Sprint**: current
- **Owner**: Implementer
- **Dependencies**: P0
- **Description**:
  - μ-Q1 미완 M5 cell (OpenCL Q4_0 production hot path latency) 보강 + 13 op 확장
  - 기존 `microbench_qnn_oppkg_matmul_q40_correct` 또는 신규 `engine/microbench/opencl_op_q40.rs`
  - production Q4_0 block-32 weight + F32 activation
- **Acceptance Criteria**:
  - Adreno 830 GREEN (S25 device)
  - tolerance: `max_abs_err < 0.05` AND `cosine ≥ 0.999`
  - Phase E format raw JSON 출력
- **Notes**: 1일.

---

## [P1c] ExecuTorch PTE build sweep (Implementer)

- **Status**: TODO
- **Sprint**: current
- **Owner**: Implementer
- **Dependencies**: P0 + 외부 자산 `/home/go/Workspace/executorch/.venv` (μ-Q1에서 셋업 완료, **P0 의존성 검증에서 glob fail — P1c 진입 직전 venv path 재검증 필수**)
- **Description**:
  - **12 op** × {`use_fp16`, `use_8a4w`} = 24 .pte 빌드 (SWIGLU 제외, P0 §2 결정)
  - **W4A8 PT2E builder 신규 작성** (μ-Q1 의 W8A8 빌더는 `QuantDtype.use_8a8w` 사용 → `QuantDtype.use_8a4w` 로 enum 교체)
  - 빌드 스크립트: `papers/eurosys2027/_workspace/experiment/qnn_microbench_phase_d/build_pte_matmul.py` 패턴 확장 (op별 sub-dir)
  - **MUL_MAT 3 shape** (mm_ffn/mm_lmh/mm_qkv) × 2 dtype = 6 .pte 추가 (mm_qkv fused N=2048)
  - PT2E quantizer calibration: fixed random seed (μ-Q1과 동일, seed=42)
  - 자산 위치: `papers/eurosys2027/_workspace/experiment/microbench_full_matrix_2026_05_XX/pte/<op>[_<shape_id>]_<dtype>.pte`
- **Acceptance Criteria**:
  - **24 .pte** (Tier-A 8 + Tier-B 4 = 12 op × 2 dtype) **+ 4 추가 .pte** (mm_lmh F16/W4A8 + mm_qkv F16/W4A8) = **28 .pte** 빌드 완료 (build log + SHA256 기록)
  - Android arm64 qnn_executor_runner 실행 dry-run 1 op GREEN
  - W4A4 native 없음 확인 — `use_8a4w` 단일 활용 결정 명시 (P0 v2-§4)
  - **W4A8 quant schema 명시** (`use_8a4w` 의 4-bit weight + 8-bit activation, asymmetric)
- **Notes**: 빌드 시간 op별 ~1-2분, 28 .pte → 30-60분. backend.py path 정합성 확인 필수. **venv 부재 시 μ-Q1 D.1 절차 재실행 (0.5d 추가)**.

---

## [P1d] Ours-NPU HTP F16 dispatch path (Senior Implementer)

- **Status**: ✅ 호스트 빌드 완료 (P4 디바이스 측정 대기) 2026-05-28
- **Sprint**: current
- **Owner**: Senior Implementer
- **Dependencies**: P0 + μ-Q1 Tier-A 8/8 correctness GREEN (`engine/src/backend/htp_fastrpc/`, HEAD `46dae3aa`)
- **Description**:
  - 현 Q4_0 dispatch path를 F16 dtype으로 fork
  - 12 op 중 F16 native HTP 지원 op만 dispatch (P0 결정: Tier-A 8 op 모두 F16, Tier-B 1 op SOFT_MAX 만 F16 GREEN inherit, SCALE/CPY/SET_ROWS `✗` 마킹)
  - `engine/src/backend/htp_fastrpc/` host-side dispatch 변경
  - **IDL 변경 0** (P0 §10 발견): `idl.rs:32` 에 `HTP_TYPE_F16: u32 = 1` 이미 정의됨. **libdsprpc rebuild 불필요**
  - **MUL_MAT mm_lmh (K=1536 N=151936) + mm_qkv (K=1536 N=2048 fused) dim shape gate 추가** (P0 §5.4 결정) — G2 의 N=1536 GREEN inherit 하나 N=151936, N=2048 (K/V N=256 포함) 미검증
- **Acceptance Criteria**:
  - Ours-NPU HTP F16 microbench bin 1 op (e.g., MUL_MAT mm_ffn) S25 GREEN
  - tolerance: `max_abs_err < 1e-2` AND `cosine ≥ 0.999` (MUL_MAT 은 `< 5e-2` Q4_0 inherit)
  - Tier-A 8 op F16 dispatch GREEN gate + FLASH_ATTN_EXT 측정 시도 결과 (GREEN 또는 `✗` 확정)
  - MUL_MAT mm_lmh N=151936 + mm_qkv N=2048 dim shape dispatch GREEN
  - μ-Q1의 8/8 Q4_0 GREEN 회귀 없음 (Tier-A correctness gate)
- **Notes**: IDL 변경 0 으로 작업량 절감. **1~1.5일** (당초 1.5~2.5d 추정 정정).

### P1d 산출물 (2026-05-28 완료)

**핵심 발견 (DSP-side llama.cpp upstream source code grep)**:
HTP DSP-side (`libggml-htp-v79.so` source `llama.cpp/ggml/src/ggml-hexagon/htp/*.c`) 의 F16 native 지원은 op 별로 **비대칭**:

| Op | DSP-side F16 src0 native | 본 sprint 의 결정 |
|---|---|---|
| **MUL_MAT** (weight) | ✓ (`matmul-ops.c:2360` f16-f32 / f16-f16 path) | `+ new` F16 dispatch bin 작성 — `microbench_htp_matmul_f16.rs` (3 shape: mm_ffn / mm_qkv / mm_lmh) |
| **ADD** | ✓ (`binary-ops.c:131,152,173` `hvx_add_f16_*`) | `+ new` — `microbench_htp_add_f16.rs` |
| **MUL** | ✓ (`binary-ops.c:133,154,175` `hvx_mul_f16_*`) | `+ new` — `microbench_htp_mul_f16.rs` |
| **FLASH_ATTN_EXT** | ✓ (`flash-attn-ops.c:618` Q∈{F16,F32}, K/V strict F16) | `+ new` shape attempt — `microbench_htp_flash_attn_ext.rs` (GREEN 또는 ✗ 확정) |
| **SILU / RMS_NORM / ROPE / GET_ROWS** | **✗ F32 only** (`unary-ops.c:368-382 op_unary` `default → HTP_STATUS_NO_SUPPORT`) | `microbench_htp_silu_f16.rs` 만 작성 — NO_SUPPORT status code 검증용 evidence. 다른 op 의 F16 ✗ 는 본 file narrative 로 통일 인용 |
| **SOFT_MAX** | ✗ src0 F32 strict (mask src1 만 F16 가능) | `✗` 확정 — 별도 bin 작성 없이 silu_f16 evidence inherit |

→ matrix.md v2-§6.0 의 `ours.htp` F16 cell 11 cell 가정 → **실제 GREEN attempt 4 cell + NO_SUPPORT 확정 ≥5 cell** 로 재분류. paper narrative 강화 ("NPU upstream DSP-side 가 F16 path 를 op 별로 비대칭 지원").

**작성한 bin 파일 5개**:
- `engine/microbench/htp_matmul_f16.rs` — MUL_MAT F16 weight × F32 act, **3 shape** (mm_ffn N=8960, mm_qkv N=2048, mm_lmh N=151936) `HTP_MM_F16_SHAPES=mm_ffn,mm_qkv` env override
- `engine/microbench/htp_add_f16.rs` — ADD F16 elementwise (DIM=1536)
- `engine/microbench/htp_mul_f16.rs` — MUL F16 elementwise (DIM=8960)
- `engine/microbench/htp_silu_f16.rs` — SILU F16 NO_SUPPORT evidence bin
- `engine/microbench/htp_flash_attn_ext.rs` — FLASH_ATTN_EXT GQA F16 dispatch attempt (head_dim=128, n_heads=12, n_kv=2, ctx=1024)

**host dispatch path 변경 0**: matrix.md v2-§10 발견 인용 — IDL 의 `HTP_TYPE_F16=1` 가 이미 정의되어 있고, `htp_tensor_from_shape(HTP_TYPE_F16, ne, nb)` 만 microbench level 에서 호출하면 dispatch 됨. mod.rs Backend trait impl 의 cpu_companion 위임 layer 는 변경 무관. **mod.rs 의 idl re-exports 에 `HTP_OP_FLASH_ATTN_EXT` 추가 한 줄** 외에 host binding 변경 0.

**sanity check 결과 (host 빌드)**:
- `cargo fmt --all` GREEN
- `cargo clippy -p llm_rs2 --bin microbench_htp_{matmul,add,mul,silu,flash_attn_ext}_f16 --bin microbench_htp_flash_attn_ext --features htp_fastrpc -- -D warnings` GREEN
- `cargo build --release ... 5 bins ... --features htp_fastrpc` GREEN (host PC, 21.44s)
- `cargo test -p llm_rs2 --lib --features htp_fastrpc backend::htp_fastrpc` 26 passed (0 회귀)
- workspace lib test 1206 passed, 24 failed — 모두 `backend::opencl::*` (호스트 GPU 부재). htp_fastrpc 무관

**Android cross-build / P4 측정 대기**:
- 본 sprint 호스트 PC 에서 NDK env (`android.source`) 의 path 가 macOS 용 — Linux build path 분리 필요. P4 측정 단계에서 `python scripts/run_device.py -d s25 push` 로 ND env 자동 주입 사용.
- 5 bin S25 측정 + dim gate (mm_qkv N=2048 / mm_lmh N=151936) GREEN/RED 판정 → matrix.md v2-§6.0 의 `ours.htp` cell status 갱신
- FLASH_ATTN_EXT bin 결과로 matrix.md `ours.htp FLASH_ATTN_EXT` cell 의 ✗ 가정 검증/갱신
- SILU F16 결과로 NO_SUPPORT 패턴 확정 → RMS_NORM/ROPE/GET_ROWS/SOFTMAX F16 cell 5개 모두 `✗` 마킹

---

## [P2] L.cpp test-backend-ops same-device driver (Implementer)

- **Status**: TODO
- **Sprint**: current
- **Owner**: Implementer
- **Dependencies**: P0 cell inventory (병렬 가능, P1과 독립)
- **Description**:
  - `third_party/llama.cpp/build-snapdragon/bin/test-backend-ops` 확인 (존재 여부 risk)
  - Android arm64 빌드 + S25 push (없으면 빌드 추가)
  - 13 op × 3 backend (`-b CPU`, `-b Adreno`, `-b HTP0`) × 2 dtype = 78 cell driver
  - 스크립트: `scripts/llamacpp_backend_ops_driver.py` (또는 `microbench_qnn_matrix.py` 통합)
  - Phase E thermal protocol 호환 (8-zone polling, round-robin)
- **Acceptance Criteria**:
  - `test-backend-ops` Android arm64 binary S25 GREEN dry-run 1 op
  - 3 backend × 1 op 실행 결과 raw JSON Phase E format
  - 빌드 상태 확인 결과 (build-snapdragon 존재 / 신규 빌드 / fail)
- **Notes**: 빌드 부재 시 `cmake .. -DCMAKE_TOOLCHAIN_FILE=...` 신규 빌드 0.5~1일 추가.

---

## [P3] microbench_qnn_matrix.py 7-col × 13-op × 2-dtype 확장 (Implementer)

- **Status**: TODO
- **Sprint**: current
- **Owner**: Implementer
- **Dependencies**: P1a + P1b + P1c + P1d + P2 (모두)
- **Description**:
  - 기존 `scripts/microbench_qnn_matrix.py` (μ-Q1, 9-cell+Ref) 확장
  - **7 backend × 12 op × 2 dtype + MUL_MAT extra 28 = 196 cell** 매트릭스 (`—`/`✗`/`⚠` 102 cell skip 마킹, **활성 94 cell**)
  - 입력: `--ops MUL_MAT,RMS_NORM,...` + `--backends ours.cpu,ours.gpu,...` + `--dtypes F16,Q4_0` + `--mm-shapes mm_ffn,mm_lmh,mm_qkv`
  - Phase E protocol inherit: warmup 3 + measure 10, round-robin shuffle, 8-zone polling (cpu_little/mid/prime, gpu_5/7, hex_vec, hex_mat, ddr), Tukey 1.5×IQR
  - **cell_id 명명**: `<backend>_<op>_<dtype>[_<shape_id>]` (P0 §8 결정)
  - 출력: `papers/eurosys2027/_workspace/experiment/microbench_full_matrix_2026_05_XX/{raw, aggregated.csv, thermal_log.csv, env.json, report.md, summary.json}`
- **Acceptance Criteria**:
  - dry-run: 1 op × 7 backend × 2 dtype × 1 round = 14 cell 측정 PASS (호스트 + S25)
  - Phase E format 호환 (column 매핑 + fair-pair grouping)
  - **skip 마킹 5-tier** (`—` fair 부적합 / `✗` hard / `⚠` perf abort / `?` 미측정 / `✓+` 측정 OK)
  - 활성 94 cell 의 phase owner 매핑 (P1a~d, P2 별)
- **Notes**: 의존성 4 트랙 모두 GREEN 후 시작. 1~1.5일.

---

## [P4] S25 측정 + Phase E format 통합 보고서 (Tester)

- **Status**: TODO
- **Sprint**: current
- **Owner**: Tester
- **Dependencies**: P3
- **Description**:
  - S25 단일 디바이스 thermal-isolated 측정 (warmup 3 + measure 10, 13 round-robin round)
  - 발열 누적 회피: round-robin shuffle + 45~180s inter-cell + 300~600s inter-round
  - **Tier-A 8 op 우선** 측정 (paper figure 우선순위) → Tier-B/D 후속
  - 1~2일 분산 측정 권장 (한 세션 8~12h 한도 초과)
  - 보고서: `papers/.../microbench_full_matrix_2026_05_XX/report.md`
  - Phase E format 확장: fair-pair row × 13 op × 2 dtype matrix + 각 op의 backend별 median/CV/GREEN-YELLOW-RED 분류
- **Acceptance Criteria**:
  - 182 cell 중 UNSUPPORTED 제외 모두 측정 완료 (또는 skip 마킹)
  - 모든 측정된 cell CV < 10% (≥10%는 재측정 1회 후 RED 마킹)
  - peak thermal < 50°C 유지 (50°C 도달 시 trial discard + cooldown 강제)
  - Phase E report format paper-grade 통합
- **Notes**: device-hour 8~12h, 1~2일 분산. 측정 protocol 변경 제안은 PM에 보고 후 사용자 결정.

---

## 리스크 / 의존성 / 결정 보류 사항

### 측정 protocol 변경 제안 (사용자 결정 대기)

본 sprint는 `plan_qnn_microbench_measurement_protocol_2026_05_26.md`의 4-cell protocol을 그대로 inherit한다.
13×7×2 = 182 cell로 확장 시 측정 시간 증가에 따른 protocol 변경 제안이 발생할 경우, PM이 결정하지 않고 사용자 보고 후 결정.

- 후보 변경 1: round 수 13 → 7 단축 (warmup 3 + measure 4)로 device-hour 압축 — Tester가 측정 중 thermal 압박 시 제안 가능
- 후보 변경 2: Tier-A만 본 sprint, Tier-B/D는 backlog 분리 — 측정 시간 초과 시 fallback

### 외부 의존성

- ExecuTorch venv (`/home/go/Workspace/executorch/.venv`): μ-Q1에서 셋업 완료. 본 sprint에서 PTE op 확장 빌드 시 동일 venv 재사용.
- QNN SDK 2.37 (Executorch 자동 download): μ-Q1 완료. 본 sprint 재활용.
- llama.cpp `test-backend-ops`: P2에서 빌드 상태 확인 후 빌드 추가 여부 결정.
- libdsprpc.so (S25 사전 배포): Ours-NPU HTP path에 필요. IDL 변경 시 영향 검토 (P1d).

### 알려진 위험 (P0 결정 반영)

| Risk | 영향 | Mitigation |
|---|---|---|
| ExecuTorch PT2E PTE 빌드 시간 (op별 1-2분, **28 .pte** = 24 op×dtype + 4 MM extra) | P1c 30-60분 추가 | 빌드 병렬화 (4-worker make), build log 캐싱 |
| **W4A8 PT2E builder 신규 작성** (μ-Q1 W8A8 builder inherit 불가) | P1c +0.5d | `QuantDtype.use_8a8w` → `use_8a4w` enum 교체, μ-Q1 build_pte_matmul.py 패턴 변형 |
| L.cpp `test-backend-ops` 경로 오기 (`third_party/...`) | P2 진입 시점 혼선 | **실제 경로 `/home/go/Workspace/llama.cpp/build-snapdragon/bin/test-backend-ops`** 확인됨. P0 §10 검증 결과 |
| ~~Ours-NPU F16 path 가 IDL 변경 수반~~ | **해소** | `idl.rs:32` HTP_TYPE_F16=1 이미 존재 → libdsprpc rebuild 불필요. **P1d 1.5~2.5d → 1~1.5d 정정** |
| ExecuTorch venv glob fail | P1c 진입 직전 검증 필요 | μ-Q1 D.1 GREEN 이므로 존재 추정 (uv hidden venv). 부재 시 D.1 절차 재실행 (0.5d) |
| MUL_MAT mm_lmh (N=151936) thermal drift | P4 trial 간 발열 | 1.1 GB weight load — inter-trial cool-down +5s |
| Ours-NPU K/V proj N=256 + mm_lmh N=151936 dim 미검증 | P1d acceptance gate | G2 sprint N=1536 helper 재사용 — dim variance test 추가 |
| L.cpp HTP0 perf path abort (`dspqueue_write 0x0c`) | P2 lcpp.htp perf cell 10 cell `⚠` | 사용자 결정 2026-05-26 inherit. support 만 reference + graph-level llama-bench tg32 인용 |
| MUL_MAT mm_qkv fused N=2048 (K=1536+256+256) | P1d K/V dim shape gate | (위 항목과 동일 — 한 sprint 안에서 처리) |
| `et.htp` W4A8 (asym int4) vs Ours.htp Q4_0 (sym int4 block-32) schema mismatch | paper narrative 정직성 | paper section 에 "same bit-width, different schema" 명시 |
| S25 측정 시간 활성 94 cell × 13 round × 5s + cooldown ≈ 6~8h | 1일 cap 안에 fit | Tier-A 우선, 1~2일 분산 옵션 유지 |
| UNSUPPORTED 5-tier 마킹 (`✓`/`+`/`⚠`/`✗`/`—`) | paper narrative | matrix.md v2-§5.5 정의 인용 |
| 발열 누적 (round-robin shuffle 시 cell 간 thermal carryover) | CV 부풀림 | 8-zone polling + 50°C trigger + 45~180s inter-cell, Tukey 1.5×IQR |

### 본 sprint 제외 (backlog 분리)

- **Tier-E 11 op** (Qwen 미사용): `OUT_PROD`, `DIAG_MASK_INF`, `ARGMAX`, etc. — backlog `[P3]` 등록 (아래 backlog 항목)
- **LLama 3.2 1B 동일 shape 매트릭스**: backlog `[P2]` 등록 (Qwen sprint 완료 후, 같은 driver로 재실행)

---

## 출력 디렉토리 규칙

- 측정 결과: `papers/eurosys2027/_workspace/experiment/microbench_full_matrix_2026_05_XX/`
  - `XX`는 P4 실측 시작 일자 (sprint 진입 일자 2026-05-28과 다를 수 있음)
- PTE 자산: `papers/eurosys2027/_workspace/experiment/microbench_full_matrix_2026_05_XX/pte/<op>_<dtype>.pte`
- 최종 보고서: 위 dir의 `report.md` (Phase E format 확장)
- raw / aggregated.csv / thermal_log.csv / env.json / summary.json — cell 별 정합 보존

---

## 자기점검

- [x] sprint 목표 검증 가능 (182 cell GREEN gate)
- [x] Phase 의존성 명시 (P0 → P1 병렬 → P2 → P3 → P4)
- [x] Tier-A 8 op 우선순위 명시 (paper figure 우선 사용)
- [x] 측정 protocol 재사용 (`plan_qnn_microbench_measurement_protocol`)
- [x] backlog 정합성 표시 (`.agent/todos/backlog.md` ACTIVE Sprint 등록)
- [x] PM 권한 내: 코드 수정 0, TODO 파일만 신설
