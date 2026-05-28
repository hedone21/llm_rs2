# Handoff: Full Microbench Matrix P0 — Architect 매트릭스 디자인 v2 진입

**작성**: 2026-05-28
**HEAD**: (sprint 진입 시점 master HEAD — `d0bd0802 refactor(events): cross-domain sink trait 인프라 제거 → log macro + action_diag_helper` 추정, 진입 직전 git rev-parse로 확정)
**Sprint master**: `.agent/todos/sprint_microbench_full_matrix_2026_05_28.md`
**선행 handoff**: `.agent/todos/handoff_qnn_microbench_phase_e_complete_2026_05_26.md` (μ-Q1 4-cell GREEN)
**다음 세션 진입 문장**: **"Full microbench matrix sprint P0 진행 — 매트릭스 디자인 v2 작성"**

---

## TL;DR

μ-Q1 4-cell paper-grade conclusion (M3/M4/M6b/M7 GREEN) 종결 후 paper main evidence 통합 매트릭스로 확장.
P0는 Architect가 **13 op × 7 backend × 2 dtype = 182 cell** 디자인 v2 + cell inventory를 확정한다.
P1{a,b,c,d} 4 트랙은 P0 cell inventory 확정 후 병렬 진입.

---

## 진행 상태

| Task | 상태 | 위치 |
|---|---|---|
| P0 매트릭스 디자인 v2 | ✅ **완료 2026-05-28** | matrix.md v2 + matrix_support_table.md v2 (본 handoff 하단 결과 요약) |
| P1a ARM64Neon CPU bin | TODO | sprint master §P1a (`+` 약 18 cell) |
| P1b OpenCL GPU bin | TODO | sprint master §P1b (`+` 약 16 cell) |
| P1c ExecuTorch PTE sweep | TODO | sprint master §P1c (28 .pte 신규, **W4A8 PT2E builder**) |
| P1d Ours-NPU HTP F16 path | TODO | sprint master §P1d (**IDL 변경 0** → 1~1.5d 정정) |
| P2 L.cpp test-backend-ops driver | TODO | sprint master §P2 (lcpp.{cpu,gpu,htp} 약 28 cell) |
| P3 microbench_qnn_matrix.py 확장 | TODO | sprint master §P3 (196 cell skip-aware driver) |
| P4 S25 측정 + 보고서 | TODO | sprint master §P4 (활성 94 cell, 1일 cap) |

---

## P0 산출물 요구사항

### 1. `matrix.md` v2 갱신

위치: `papers/eurosys2027/_workspace/experiment/microbench_op_matrix_2026_05_26/matrix.md` (초안 존재) → v2

내용:
- 7 backend 정의 v2 (Ours-NPU / ExecuTorch / L.cpp 3-way 분리 명시)
  - `ours.cpu` (ARM64Neon CPU), `ours.gpu` (OpenCL Adreno 830), `ours.htp` (HTP FastRPC F16/Q4_0)
  - `et.htp` (ExecuTorch HTP `use_fp16` / `use_8a4w`)
  - `lcpp.cpu`, `lcpp.gpu` (Adreno), `lcpp.htp` (HTP0 Hexagon)
- 13 op (Tier A 8 + B 3 + D 2) × 2 dtype (F16, Q4_0) × 7 backend = **182 cell**
- 각 op의 shape 매핑 (Qwen 2.5-1.5B actual: hidden=1536, n_heads=12, n_kv=2, head_dim=128, FFN=8960, vocab=151936, n_layers=28)
- native unsupported cell 정직 표시 ("✗" 또는 "—") — paper narrative에 그대로 인용

### 2. `matrix_support_table.md` 갱신

위치: `papers/eurosys2027/_workspace/experiment/microbench_op_matrix_2026_05_26/matrix_support_table.md` (초안 존재) → v2

내용:
- 7 backend × 13 op × 2 dtype 셀별 GREEN / YELLOW / UNSUPPORTED 분류
- UNSUPPORTED 결정 근거 (예: `ours.htp` F16 SOFT_MAX 미지원, `et.htp` `use_8a4w`로 ROPE 미지원 등)
- L.cpp `-b HTP0` 지원 op 확인 (llama.cpp 소스: `/home/go/Workspace/llama.cpp/ggml/src/ggml-hexagon/`)

### 3. Cell inventory

182 cell 의 op × dtype × backend × fair-pair grouping을 표 형태로:
- F16 row: `ours.cpu` F16 vs `ours.gpu` F16 vs `ours.htp` F16 vs `et.htp` use_fp16 vs `lcpp.cpu`/`lcpp.gpu`/`lcpp.htp` F16
- Q4_0 row: `ours.cpu` Q4_0 vs `ours.gpu` Q4_0 vs `ours.htp` Q4_0 vs `et.htp` use_8a4w (W4A8) vs `lcpp.*` Q4_0
- per-op 1 row (총 13 row × 2 dtype = 26 fair-pair group)

### 4. Tolerance table per op

`plan_qnn_microbench_measurement_protocol_2026_05_26.md` §정확성 검증 inherit + op별 override:

| dtype row | tolerance | 비고 |
|---|---|---|
| F16 | `max_abs_err < 1e-2` AND `cosine ≥ 0.999` | μ-Q1 inherit |
| Q4_0 (Ours) | `max_abs_err < 0.05` AND `cosine ≥ 0.999` | μ-Q1 M5 inherit |
| W4A8 (ExecuTorch `use_8a4w`) | `max_abs_err < 0.1` AND `cosine ≥ 0.99` | μ-Q1 M7 inherit |
| FLASH_ATTN_EXT | `max_abs_err < 5e-2` (softmax 누적 오차) | 신규 — 결정 |
| SOFT_MAX | `max_abs_err < 1e-3` | 신규 — 결정 |

op별 override는 Architect 결정 (특히 누적 op: FLASH_ATTN_EXT, SOFT_MAX, ROPE).

### 5. 측정 protocol 호환 확인

`plan_qnn_microbench_measurement_protocol_2026_05_26.md` 그대로 inherit:
- warmup 3 + measure 10, round-robin shuffle, 8-zone polling
- 50°C trigger / 60s session warmup / 45~180s inter-cell / 300~600s inter-round
- Tukey 1.5×IQR
- 출력: raw + aggregated.csv + thermal_log.csv + report.md + summary.json

---

## 자산 (재사용)

- μ-Q1 4-cell 매트릭스 자산: `papers/eurosys2027/_workspace/experiment/qnn_microbench_phase_d_prod_v2/` + `qnn_microbench_phase_e_report.md`
- μ-Q1 Tier-A 8/8 correctness GREEN: HEAD `46dae3aa` (G1~G5 sprint)
- ExecuTorch venv 셋업: `/home/go/Workspace/executorch/.venv` (uv 3.12 + executorch + QNN SDK 2.37)
- S25 자산: `/data/local/tmp/executorch/` (qnn_executor_runner + 7 .so + 2 .pte)
- 측정 driver: `scripts/microbench_qnn_matrix.py` (9-cell+Ref 매트릭스용, P3에서 확장)

---

## 결정 보류 사항 (Architect가 결정)

1. **Tier-B/D op의 fair-pair grouping**: SOFT_MAX, SCALE, CPY, SET_ROWS, SWIGLU 의 backend별 native 지원 매트릭스 확정 (사전 조사 필요)
2. **SWIGLU vs SILU+MUL fusion**: SWIGLU native 지원 backend (`et.htp`, `lcpp.htp` 추정) vs SILU+MUL 분리 측정 backend (`ours.*`) — 한 op cell로 통합 측정할지 분리할지
3. **per-op shape 결정**:
   - MUL_MAT: FFN gate `[1, 1536] × [1536, 8960]` (μ-Q1 inherit) + LM head `[1, 1536] × [1536, 151936]` 분리 여부
   - FLASH_ATTN_EXT: seq=128 / seq=512 / seq=1024 sweep 여부
   - ROPE: head_dim=128 single shape
4. **UNSUPPORTED cell 마킹 규칙**: "✗" (backend 정책상 미지원) vs "—" (op 자체가 dtype 미정의)

위 결정은 Architect가 v2 작성 중 사용자 보고 후 확정.

---

## 미해결 question (사용자 결정 대기)

- **측정 시간 한도**: 182 cell × 13 round × ~5s/trial = ~3.3h pure compute. 발열 cooldown 합산 시 8~12h. 1~2일 분산 vs 1 session 강행 결정.
- **L.cpp `test-backend-ops` 빌드 상태**: `third_party/llama.cpp/build-snapdragon/bin/test-backend-ops` 존재 여부 P0에서 확인. 부재 시 P2 빌드 추가 0.5-1일.
- **Ours-NPU F16 path IDL 변경 여부**: P1d 진입 직전 `engine/src/backend/htp_fastrpc/htp_iface.idl` diff 검토 필요. libdsprpc rebuild 영향.

---

## P0 완료 요약 (2026-05-28 Architect)

### 산출물

- `papers/eurosys2027/_workspace/experiment/microbench_op_matrix_2026_05_26/matrix.md` — v1 deprecation 표시 + **v2 섹션** (§1~§12: backend 7개 / op 12개 / MUL_MAT 3 shape / Architect §5 결정 5건 / cell inventory group 28×7 / status breakdown / tolerance / protocol / phase entry / 의존성 / 리스크 / paper narrative)
- `papers/.../matrix_support_table.md` — v1 deprecation + **v2 섹션** (7-backend × 196 cell 5-tier status legend)
- `.agent/todos/sprint_microbench_full_matrix_2026_05_28.md` — TL;DR/Phase 표/P0/P1c/P1d task 갱신

### 196 cell breakdown

| status | count | % |
|---|---:|---:|
| `✓` existing (μ-Q1 + G1~G6) | 11 | 5.6% |
| `+` new (P1a/b/c/d/P2) | 83 | 42.3% |
| `⚠` perf abort (lcpp.htp F16) | 10 | 5.1% |
| `✗` hard | 12 | 6.1% |
| `—` fair 부적합 | 80 | 40.8% |

**측정 활성 cell** = 94 (1일 cap 안에 흡수 가능, 발열 cooldown 포함 6~8h).

### PM 보류 4건 → Architect 결정

1. **Tier-B fair-pair**: F16 row 만, Q4_0 row 모두 `—` (activation-only)
2. **SWIGLU**: 제외 (Qwen 미사용 + Ours/ExecuTorch 미구현)
3. **per-op shape sweep**: MUL_MAT 만 3 shape, 나머지 single
4. **MUL_MAT QKV**: fused 측정 (N=2048 합계 1 cell, K/V N=256 dim gate 는 P1d 책임)
5. **UNSUPPORTED 5-tier 마킹**: `✓` / `+` / `⚠` / `✗` / `—`

### 의존성 검증 결과

| 항목 | 상태 | 비고 |
|---|---|---|
| llama.cpp test-backend-ops | ✓ 존재 | **`/home/go/Workspace/llama.cpp/build-snapdragon/bin/test-backend-ops`** (handoff 의 `third_party/...` 경로는 오타) |
| ExecuTorch venv | ⚠ glob fail | μ-Q1 D.1 GREEN 이므로 존재 추정. P1c 진입 직전 재확인 (uv venv hidden) |
| QNN SDK 2.37 | ✓ 가용 | μ-Q1 D.1 inherit |
| Ours-NPU IDL F16 | ✓ **변경 0** | `idl.rs:32` HTP_TYPE_F16=1 이미 정의. **libdsprpc rebuild 불필요** (P1d 1.5~2.5d → 1~1.5d 정정) |
| engine/microbench/htp_*.rs | ✓ 8개 존재 | G1~G6 Tier-A 7 op F16 + MATMUL Q4_0 inherit |
| μ-Q1 matmul_{f16,w8a8}.pte | ✓ 존재 | W4A8 는 `use_8a8w` → `use_8a4w` 재빌드 (P1c) |

### 추가 발견 리스크 8건 (sprint master + matrix.md v2-§11 에 등록)

1. ExecuTorch W4A8 PT2E 빌더 신규 작성 (μ-Q1 W8A8 builder inherit 불가)
2. MUL_MAT mm_lmh (151936) thermal drift
3. Ours-NPU K/V proj 256 out_dim 미검증 (G2 N=1536 만 GREEN)
4. L.cpp.HTP0 perf path abort (dspqueue_write 0x0000000c)
5. MUL_MAT mm_qkv fused N=2048 Ours-NPU 측 dim gate 미검증
6. CPY Q4_0 row 의미 없음 (`—` 마킹 + paper narrative)
7. ExecuTorch et.htp W4A8 (asym int4) vs Ours.htp Q4_0 (sym int4 block-32) schema 차이 paper 명시
8. 측정 시간 6~8h cap 안에 fit (94 cell × 13 round × 5s + cooldown)

---

## 다음 세션 진입 명령

P0 완료 → P1 4-track 병렬 진입 가능. 진입 명령:

```
"Full microbench matrix P1 진행 — P1a/P1b/P1c/P1d 4-track 분담 시작"
```

또는 (트랙 개별 진입):

```
"P1a ARM64Neon CPU microbench bin 작성 — sprint master §P1a"
"P1b OpenCL GPU microbench bin 작성 — sprint master §P1b"
"P1c ExecuTorch W4A8 PTE builder 작성 — sprint master §P1c"
"P1d Ours-NPU HTP F16 host dispatch — sprint master §P1d (IDL 변경 0)"
"P2 L.cpp test-backend-ops driver — sprint master §P2"
```

### P1 entry handoff (PM 향)

각 P1 트랙별 detail handoff 는 **sprint master 의 해당 § 섹션** 이 entry point. 별도 sub-handoff 파일은 P1 진입 시점에 작성자 (Implementer/Senior Implementer) 가 본인 책임. PM 이 트랙별 owner 배정 후 호출.

---

## 자기점검 (P0 완료 시점)

- [x] P0 산출물 5개 모두 작성 완료 (matrix.md v2 §1~§12 / matrix_support_table.md v2 / cell inventory group 28×7 inline / tolerance op×dtype 표 / protocol inherit)
- [x] PM 보류 4건 → Architect 결정 5건 (UNSUPPORTED 마킹 추가)
- [x] 196 cell status breakdown 5-tier 정밀 산출 (`✓` 11 / `+` 83 / `⚠` 10 / `✗` 12 / `—` 80)
- [x] 의존성 검증 6개 항목 (test-backend-ops 경로 정정, IDL F16 변경 0 발견)
- [x] 추가 리스크 8건 sprint master 등록
- [x] P1 트랙별 entry point 명시 (P1d 작업량 1.5~2.5d → 1~1.5d 정정)
- [x] 다음 세션 진입 명령 P1 4-track 병렬 + 트랙 개별 5종 제시
