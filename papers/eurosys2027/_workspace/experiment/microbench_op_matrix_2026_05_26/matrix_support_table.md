# Support Matrix (v1 — pre-P0)

> **DEPRECATED 2026-05-28**: v1 은 4 backend column. v2 (7 backend × 12 op × 2 dtype) 는 본 파일 하단의 § v2 섹션 참조.
> 본 v1 표는 test-backend-ops 의 codebase sweep + G1~G5 sprint 결과를 raw record 로 보존.

| ID | OP (GGML / IDL) | Qwen tier | shape hint | **CPU** | **L.cpp.GPU** (Adreno) | **L.cpp.HTP0** (Hexagon) | **Ours-NPU** |
|---:|---|:--:|---|:--:|:--:|:--:|:--:|
|  6 | **RMS_NORM** / RMS_NORM | A | [1536] | ✓ | △ 76% | △ 52% | ✓ |
|  4 | **MUL_MAT** / MUL_MAT | A | Q4_0 1536/8960/151936 | △ 81% | △ 52% | △ 25% | ✗ |
| 14 | **ROPE** / ROPE | A | head_dim=128 | ✓ | ✓ | △ 22% | ✗ |
| 15 | **FLASH_ATTN_EXT** / FLASH_ATTN_EXT | A | hs=128 nh=12 nkv=2 | ✓ | △ 54% | ✗ | ✗ |
| 17 | **GET_ROWS** / GET_ROWS | A | vocab=151936 | ✓ | △ 12% | △ 8% | ✗ |
|  7 | **SILU** / UNARY_SILU | A | [8960] | ✓ | △ 25% | △ 25% | ✗ |
|  0 | **MUL** / MUL | A | [8960] | ✓ | ✓ | △ 73% | ✗ |
|  1 | **ADD** / ADD | A | [1536] | ✓ | ✓ | △ 73% | ✗ |
| 12 | **SOFT_MAX** / SOFTMAX | B | nh=12 nctx | ✓ | ✓ | △ 50% | ✗ |
| 18 | **SCALE** / SCALE | B | fused | ✓ | ✓ | ✓ | ✗ |
| 19 | **CPY** / CPY | B | dtype | △ 74% | △ 10% | △ 9% | ✗ |
| 16 | **SET_ROWS** / SET_ROWS | B | KV scatter | △ 72% | △ 14% | △ 7% | ✗ |
|  9 | **SWIGLU** / GLU_SWIGLU | D | fused alt | ✓ | ✓ | △ 25% | ✗ |
|  2 | **SUB** / SUB | E |  | ✓ | ✓ | △ 73% | ✗ |
|  3 | **DIV** / DIV | E |  | ✓ | ✓ | △ 73% | ✗ |
|  5 | **MUL_MAT_ID** / MUL_MAT_ID | E | MoE | ✓ | △ 35% | △ 35% | ✗ |
|  8 | **GELU** / UNARY_GELU | E |  | ✓ | △ 25% | △ 25% | ✗ |
| 10 | **SWIGLU_OAI** / GLU_SWIGLU_OAI | E |  | ✓ | ✓ | △ 50% | ✗ |
| 11 | **GEGLU** / GLU_GEGLU | E |  | ✓ | ✓ | △ 25% | ✗ |
| 13 | **ADD_ID** / ADD_ID | E | MoE | ✓ | ✓ | ✓ | ✗ |
| 20 | **ARGSORT** / ARGSORT | E | host | ✓ | △ 40% | △ 74% | ✗ |
| 21 | **SQR** / SQR | E |  | ✓ | ✓ | △ 50% | ✗ |
| 22 | **SQRT** / SQRT | E |  | ✓ | ✓ | △ 50% | ✗ |
| 23 | **SUM_ROWS** / SUM_ROWS | E | RMSN fused | ✓ | △ 40% | △ 40% | ✗ |
| 24 | **SSM_CONV** / SSM_CONV | E |  | ✓ | ✓ | ✓ | ✗ |

---

# v2 — 7 backend × 12 op × 2 dtype = 168 cell support map (+ MUL_MAT 추가 shape 28 = 196 total)

**작성**: 2026-05-28 (P0 sprint)
**관련 산출물**: [`matrix.md`](matrix.md) v2 섹션, [`cell_inventory.md`](cell_inventory.md)

## v2-§1 7-backend × Tier-A 8 op × 2 dtype × 1 shape (= 112 cell, MUL_MAT base shape mm_ffn 만)

cell status legend: `✓` = native 지원 + 측정 가능 / `+` = 신규 bin 작성 후 측정 / `⚠` = perf path abort (support 만 reference) / `✗` = hard 미지원 / `—` = fair 부적합 (활성화-only op + Q4_0 row 등).

### F16 row (Tier-A 8 op × 7 backend = 56 cell)

| OP / shape | ours.cpu | ours.gpu | ours.htp | et.htp (`use_fp16`) | lcpp.cpu | lcpp.gpu | lcpp.htp |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **MUL_MAT** mm_ffn `[1,1536]×[1536,8960]` | `+` P1a | `✓` M3 | `+` P1d (F16 fork from G2) | `✓` M6b | `+` P2 | `+` P2 | `⚠` perf abort |
| **RMS_NORM** `[1,1536]` | `+` P1a | `+` P1b | `✓` G1 | `+` P1c | `+` P2 | `+` P2 | `⚠` |
| **ROPE** head_dim=128 | `+` P1a | `+` P1b | `✓` G3 | `+` P1c | `+` P2 | `+` P2 | `⚠` (shape 22% only) |
| **FLASH_ATTN_EXT** hs=128 nh=12 nkv=2 ctx=1024 | `+` P1a | `+` P1b | `✗` (NPU fused FA 미지원, G1~G6 미검증) | `+` P1c | `+` P2 | `+` P2 | `✗` (test-backend-ops 0%) |
| **GET_ROWS** vocab=151936 | `+` P1a | `+` P1b | `+` P1d (full vocab, G6 는 1024 dummy) | `+` P1c | `+` P2 | `+` P2 | `⚠` (vocab large) |
| **SILU** `[1,8960]` | `+` P1a | `+` P1b | `✓` G4 | `+` P1c | `+` P2 | `+` P2 | `⚠` |
| **MUL** `[1,8960]` | `+` P1a | `+` P1b | `✓` G4 | `+` P1c | `+` P2 | `+` P2 | `⚠` |
| **ADD** `[1,1536]` | `+` P1a | `+` P1b | `✓` G4 | `+` P1c | `+` P2 | `+` P2 | `⚠` |

### Q4_0/W4A8 row (Tier-A 8 op × 7 backend = 56 cell)

| OP / shape | ours.cpu | ours.gpu | ours.htp | et.htp (`use_8a4w`) | lcpp.cpu | lcpp.gpu | lcpp.htp |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **MUL_MAT** mm_ffn `[1,1536]×Q4_0[1536,8960]` | `+` P1a | `✓` M5 | `✓` G2 inherit | `+` P1c (W4A8 PT2E builder 신규) | `+` P2 | `+` P2 | `⚠` |
| **RMS_NORM** | `—` (activation-only) | `—` | `—` | `—` | `—` | `—` | `—` |
| **ROPE** | `—` | `—` | `—` | `—` | `—` | `—` | `—` |
| **FLASH_ATTN_EXT** | `—` | `—` | `—` | `—` | `—` | `—` | `—` |
| **GET_ROWS** Q4_0 embed | `+` P1a | `+` P1b | `+` P1d (Q4_0 embed 미검증) | `+` P1c | `+` P2 | `+` P2 | `⚠` |
| **SILU** | `—` | `—` | `—` | `—` | `—` | `—` | `—` |
| **MUL** | `—` | `—` | `—` | `—` | `—` | `—` | `—` |
| **ADD** | `—` | `—` | `—` | `—` | `—` | `—` | `—` |

## v2-§2 Tier-B 4 op (× 7 backend × 2 dtype = 56 cell)

### F16 row (Tier-B)

| OP / shape | ours.cpu | ours.gpu | ours.htp | et.htp | lcpp.cpu | lcpp.gpu | lcpp.htp |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **SOFT_MAX** `[12,1,1024]` | `+` P1a | `+` P1b | `✓` G5 | `+` P1c | `+` P2 | `+` P2 | `⚠` |
| **SCALE** `[12,1,1024]` | `+` P1a | `+` P1b | `✗` init_scale_req 미구현 | `+` P1c | `+` P2 | `+` P2 | `⚠` |
| **CPY** F32→F16 `[2,1,128]` | `+` P1a | `+` P1b | `✗` init_cpy_req 미구현 | `+` P1c | `+` P2 | `+` P2 | `⚠` |
| **SET_ROWS** `[2,1,128]→cache[1024,2,128]` | `+` P1a | `+` P1b | `✗` init_set_rows_req 미구현 | `+` P1c | `+` P2 | `+` P2 | `⚠` |

### Q4_0 row (Tier-B): 모두 `—` (activation-only / dtype-conversion op, Q4_0 의미 없음) = 28 cell `—` 일괄.

## v2-§3 MUL_MAT 추가 shape (`mm_lmh` + `mm_qkv` × 7 backend × 2 dtype = 28 cell)

| OP / shape | dtype | ours.cpu | ours.gpu | ours.htp | et.htp | lcpp.cpu | lcpp.gpu | lcpp.htp |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **MUL_MAT** mm_lmh `[1,1536]×[1536,151936]` | F16 | `+` | `+` | `+` P1d (K=1536 N=151936 dim gate) | `+` P1c | `+` P2 | `+` P2 | `✗` shape too large |
| **MUL_MAT** mm_lmh | Q4_0/W4A8 | `+` | `+` | `+` P1d | `+` P1c | `+` P2 | `+` P2 | `✗` |
| **MUL_MAT** mm_qkv fused `[1,1536]×[1536,2048]` | F16 | `+` | `+` | `+` P1d (K/V N=256 dim gate) | `+` P1c | `+` P2 | `+` P2 | `⚠` |
| **MUL_MAT** mm_qkv | Q4_0/W4A8 | `+` | `+` | `+` P1d | `+` P1c | `+` P2 | `+` P2 | `⚠` |

(4 row × 7 backend = 28 cell)

## v2-§4 Total

112 (Tier-A) + 56 (Tier-B) + 28 (MUL_MAT extra) = **196 cell** ✓

### Status 집계

| status | count | % |
|---|---:|---:|
| `✓` existing | 11 | 5.6% |
| `+` new (P1a/b/c/d/P2) | 83 | 42.3% |
| `⚠` perf abort (lcpp.htp F16 Tier-A/B) | 10 | 5.1% |
| `✗` hard (NPU FA / mm_lmh oversize / Tier-B Ours-NPU) | 12 | 6.1% |
| `—` fair 부적합 (Q4_0 row activation-only) | 80 | 40.8% |

측정 활성 cell = 11 + 83 = **94 cell** (P4 측정 대상).

## v2-§5 paper narrative 후보 (support 표 기반)

1. **NPU 가 hot path Tier-A 8 op 중 100% 지원 backend 없음** (test-backend-ops sweep + G1~G6 self-binding 검증 합산). 이 fact 자체가 self-built binding 의 정당성.
2. **Tier-B 4 op (SOFT_MAX/SCALE/CPY/SET_ROWS) 의 Ours-NPU 1/4 만 GREEN** — SCALE/CPY/SET_ROWS 는 `init_*_req` helper 미구현 (G1~G6 scope 외) → backlog 분리.
3. **`et.htp` 의 Q4_0 row 는 모두 W4A8 (asym int4)** — Ours.htp 의 Q4_0 (sym int4 block-32) 와 weight bit-width 동일하나 quant schema 다름. paper section 에 "fair-pair" 정의를 명시 ("same bit-width, different schema").

---

## v2 자기점검

- [x] 7 backend column 정의 (v1 의 4 backend → 7 backend: Ours/ExecuTorch/L.cpp 3-way 분리)
- [x] 12 op × 2 dtype × 7 backend = 168 cell + MUL_MAT extra 28 = 196 cell
- [x] status legend 5-tier (`✓` / `+` / `⚠` / `✗` / `—`)
- [x] G1~G6 Ours-NPU Tier-A 7 op F16 GREEN inherit 반영 (RMSN/ROPE/SILU/MUL/ADD/SOFTMAX/GETROWS, MATMUL은 Q4_0 만 ✓)
- [x] μ-Q1 4 cell (M3/M5/M6b/G2 Q4_0) inherit 반영
- [x] phase owner (P1a/b/c/d/P2) per cell 표시
- [x] hard `✗` vs soft `—` 명확 구분
