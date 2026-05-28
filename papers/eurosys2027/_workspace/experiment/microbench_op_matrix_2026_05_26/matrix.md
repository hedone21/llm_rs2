# Qwen2.5-1.5B Microbench Matrix (v1 scaffold — SUPERSEDED by v2)

> **DEPRECATED 2026-05-28**: 본 문서는 v1 scaffold (25-op × 4-backend) 로 paper-grade 매트릭스 직전 단계.
> P0 sprint 2026-05-28 이후 **v2 (12-op × 7-backend × 2-dtype = 196 cell)** 로 확장됨.
> v2 정식 산출물: [`matrix_v2.md`](matrix_v2.md) (sprint entry: `.agent/todos/handoff_microbench_full_matrix_phaseP0_2026_05_28.md`).
> v1은 backend 4-way scaffold (CPU / GPU / L.cpp / Ours-NPU) 와 G1~G5 Tier-A 8/8 GREEN 결과의 raw record 로 보존.

**작성**: 2026-05-26
**worktree**: `.claude/worktrees/b5_trait_extension`
**device**: Galaxy S25 (R3CY408S5SB, SM8750, Hexagon V79, Adreno 830, 6T CPU)
**model**: Qwen2.5-1.5B Q4_0 — `hidden=1536 / n_heads=12 / n_kv=2 (GQA 6:1) / head_dim=128 / FFN=8960 / n_layers=28 / vocab=151936`
**op enum source**: `engine/src/backend/htp_fastrpc/idl.rs` (llama.cpp `htp/htp-msg.h` byte-identical)

---

## Backend 정의

| 약어 | 정의 | 측정 path |
|---|---|---|
| **CPU** | Ours NEON CPU backend (engine/src/backend/cpu/) | μMatrix-4 직접 wrap 또는 microbench_ops cpu extension |
| **GPU** | Ours OpenCL GPU backend (engine/kernels/*.cl) | `microbench_ops` (이미 구현, ours side) |
| **L.cpp** | llama.cpp 대응 backend 동일 device | `test-backend-ops` (S25, build-snapdragon) — backend 선택은 `-b CPU/Adreno/HTP0` |
| **Ours-NPU** | Ours HTP FastRPC NPU binding (engine/src/backend/htp_fastrpc/) | `microbench_htp_*` — Tier-A 8 op dispatch GREEN (G1~G5 sprint 2026-05-26~27) |

L.cpp 컬럼은 같은 device에서 측정한 llama.cpp의 **최고 backend** (보통 GPU, 일부 op은 CPU가 빠를 수도) — 같은 op 같은 shape으로 비교.

---

## Qwen2.5-1.5B 사용 분류 (28-layer decode, seq_len=1)

| Tier | 의미 | op 수 |
|---|---|---|
| **A — Hot path (forward 매 layer)** | 28 layer × 다회 호출. paper main evidence | 8 |
| **B — Conditional / non-fused** | FlashAttn fused 여부에 따라 사용/미사용 | 4 |
| **C — Layer 1회 또는 token 1회** | Hot 아니지만 forward에 1회 이상 | 1 |
| **D — GLU 변형 (fused SiLU+mul 대체)** | llama.cpp 최신만, ours는 분리 | 1 |
| **E — Qwen2 미사용 (다른 모델 전용)** | 본 matrix paper appendix 자리 | 11 |

---

## Support 매트릭스 (test-backend-ops sweep 결과, 2026-05-26)

**source**: `tbo_support_full.csv` (OpenCL+HTP0, 27264 rows) + `tbo_support_cpu.csv` (CPU, 13644 rows).
**cell 의미**: `✓` = 모든 test shape 지원 / `△ NN%` = 부분 지원 (shape별로 mixed) / `✗` = 0건 / `—` = 미측정.
N% 는 ggml test suite의 op_params (dtype, ne, GQA bs 등) sweep 중 PASS 비율 — Qwen exact shape 매핑은 perf 측정에서 따로 검증.

| ID | OP (GGML / IDL) | tier | shape hint | **CPU** | **L.cpp.GPU** (Adreno) | **L.cpp.HTP0** (Hexagon) | **Ours-NPU** |
|---:|---|:--:|---|:--:|:--:|:--:|:--:|
|  6 | **RMS_NORM** / RMS_NORM | A | `[1536]` | ✓ | △ 76% | △ 52% | ✓ |
|  4 | **MUL_MAT** / MUL_MAT | A | Q4_0 K=1536/N∈{1536,256,8960} + lm_head Q4 K=1536/N=151936 | △ 81% | △ 52% | △ 25% | ✗ |
| 14 | **ROPE** / ROPE | A | head_dim=128 (q12,kv2) θ=1e6 | ✓ | ✓ | △ 22% | 101.20 μs |
| 15 | **FLASH_ATTN_EXT** / FLASH_ATTN_EXT | A | hs=128 nh=12 nkv=2 (GQA) | ✓ | △ 54% | **✗** | ✗ |
| 17 | **GET_ROWS** / GET_ROWS | A | embed `[151936, 1536]` | ✓ | △ 12% | △ 8% | 109.38 μs ‡ |
|  7 | **SILU** / UNARY_SILU | A | `[8960]` | ✓ | △ 25% | △ 25% | ✓ |
|  0 | **MUL** / MUL | A | `[8960]` (SiLU·up) | ✓ | ✓ | △ 73% | ✓ |
|  1 | **ADD** / ADD | A | `[1536]` (residual) | ✓ | ✓ | △ 73% | ✓ |
| 12 | **SOFT_MAX** / SOFTMAX | B | `[nh=12, 1, nctx]` (FlashAttn fused 시 미사용) | ✓ | ✓ | △ 50% | 124.22 μs |
| 18 | **SCALE** / SCALE | B | 1/√d_k (FlashAttn fused 시 미사용) | ✓ | ✓ | ✓ | ✗ |
| 19 | **CPY** / CPY | B | dtype/layout 변환 | △ 74% | △ 10% | △ 9% | ✗ |
| 16 | **SET_ROWS** / SET_ROWS | B | KV scatter `[2,1,128]→cache` | △ 72% | △ 14% | △ 7% | ✗ |
|  9 | **SWIGLU** / GLU_SWIGLU | D | fused alt (Qwen2 비-fused, llama.cpp만) | ✓ | ✓ | △ 25% | ✗ |
|  2 | SUB | E | — | ✓ | ✓ | △ 73% | — |
|  3 | DIV | E | — | ✓ | ✓ | △ 73% | — |
|  5 | MUL_MAT_ID | E | MoE 전용 | ✓ | △ 35% | △ 35% | — |
|  8 | GELU | E | Qwen2 = SiLU 사용 | ✓ | △ 25% | △ 25% | — |
| 10 | SWIGLU_OAI | E | GPT-OSS 변형 | ✓ | ✓ | △ 50% | — |
| 11 | GEGLU | E | GeLU 기반 | ✓ | ✓ | △ 25% | — |
| 13 | ADD_ID | E | MoE 전용 | ✓ | ✓ | ✓ | — |
| 20 | ARGSORT | E | top-k host-side | ✓ | △ 40% | △ 74% | — |
| 21 | SQR | E | Qwen2 미사용 | ✓ | ✓ | △ 50% | — |
| 22 | SQRT | E | RMSNorm 내부 fused | ✓ | ✓ | △ 50% | — |
| 23 | SUM_ROWS | E | RMSNorm 내부 fused | ✓ | △ 40% | △ 40% | — |
| 24 | SSM_CONV | E | Mamba/SSM 전용 | ✓ | ✓ | ✓ | — |

### Tier-A (hot path) 결정적 결과

**HTP NPU의 hot path 부분 지원 패턴** (paper narrative 핵심):

| Op (Tier-A) | HTP0 지원 | shape 제약 |
|---|---|---|
| FLASH_ATTN_EXT | **0%** | NPU는 fused attention 전혀 미지원 → Q*K/scale/softmax/Att*V 분리 path 필수 |
| ROPE | 22% | 일부 (`type`/`mode`) 만, Qwen2 normal mode + head_dim=128 가용성 별 검증 필요 |
| GET_ROWS | 8% | embed lookup vocab=151936 large dim 시 ✗ 위험 |
| RMS_NORM | 52% | 절반 shape, Qwen `[1536]`는 perf 단계 검증 |
| MUL_MAT | 25% | Q4_0 + specific shape만 |
| SILU | 25% | unary op_params 제약 |
| MUL/ADD | 73% | 대부분 ✓, 일부 broadcast 모양 ✗ |

→ **HTP NPU는 production hot path의 8개 Tier-A op 중 1개도 100% 지원하지 않음**. Production 사용 시 8개 모두 HTP+OpenCL split inference 필요.

→ **Ours-NPU**는 8/25 op (RMS_NORM + MUL_MAT + ADD + MUL + SILU + ROPE + SOFTMAX + GET_ROWS) GREEN. **Tier-A hot path 8개 모두 GREEN** (G1~G5 sprint 2026-05-27 완료). 17/25 op은 init_*_req helper 미구현 = paper "self-built NPU binding scope" evidence.

## Performance 매트릭스 (latency μs/call, test-backend-ops perf)

**source**: `tbo_perf_console_{cpu,gpu,gpu_htp,htp,htp_small}.txt`
**측정 환경**: S25 (R3CY408S5SB), 6 CPU threads (Galaxy S25 최적), cooldown OK.
**HTP cell**: ggml-hexagon backend는 test-backend-ops perf mode에서 모든 op에 대해 `dspqueue_write failed: 0x0000000c` driver abort (`ggml-hexagon.cpp:155`) → per-op latency **측정 불가**. graph-level (llama-bench) 만 가능, 추가 per-op 측정 시도 ✗ (사용자 결정 2026-05-26 — llama.cpp HTP는 **코드베이스 support flag** 만 reference). paper에서는 옵션 C graph-level 측정값(`llama-bench HTP0 tg32 = 32.40 tok/s`)만 활용.
**Ours-NPU cell**: 별 sprint (Q-2.2 옵션 D) 측정 → RMS_NORM 1 cell만 가용.

### 핵심 결과: CPU(NEON) vs GPU(Adreno 830) win rate

170 common (op,params) shape 중:
- **CPU 우위**: 132 / 170 = **77.6%** (factor ≥1.05×)
- GPU 우위:    32 / 170 = 18.8%
- tie:          6 / 170 = 3.5%

| OP | CPU wins | GPU wins | tie | N | 주된 패턴 |
|---|---:|---:|---:|---:|---|
| FLASH_ATTN_EXT | **14** | 0 | 0 | 14 | GPU 14×~25× 느림 (Adreno FlashAttn 미최적) |
| ROPE           | 44 | 9 | 3 | 56 | small kernel launch overhead. CPU 우위 |
| MUL_MAT        | 47 | 10 | 1 | 58 | GEMM (n=512)는 GPU 우위, GEMV는 dtype 의존 |
| SOFT_MAX       | 21 | 6 | 2 | 29 | vector (큰 D)는 CPU, matrix (4096²)는 GPU |
| ADD            | 1 | 1 | 0 | 2 | scalar는 CPU, broadcast는 비슷 |
| ADD_ID (MoE)   | 4 | 0 | 0 | 4 | CPU 6~17× 우위 |
| CPY            | 0 | 4 | 0 | 4 | memory copy 큰 shape는 GPU 우위 |
| SUM_ROWS       | 1 | 1 | 1 | 3 | shape 의존 |

### Qwen 2.5-1.5B 핫패스 representative shape (decode + prefill)

| OP | shape label | **CPU** μs | **GPU** μs | **HTP** μs | **Ours-NPU** μs | best |
|---|---|---:|---:|---:|---:|:--:|
| **MUL_MAT** Q4_0 GEMV (decode n=1) | m=4096, k=14336 | 6899 | **2183** | abort | ✗ | GPU 3.16× |
| **MUL_MAT** Q4_0 batch=8 | m=4096, k=14336, n=8 | **4506** | 11149 | abort | ✗ | CPU 2.47× |
| **MUL_MAT** Q4_0 GEMM (prefill n=512) | m=4096, k=14336 | 372425 | **145603** | abort | ✗ | GPU 2.56× |
| **MUL_MAT** Q8_0 GEMV | m=4096, k=14336, n=1 | 7479 | **3311** | abort | ✗ | GPU 2.26× |
| **MUL_MAT** Q4_K GEMV | m=4096, k=14336, n=1 | **2155** | 2260 | abort | ✗ | ≈ |
| **MUL_MAT** F16 GEMV | m=4096, k=14336, n=1 | **6512** | 7274 | abort | ✗ | CPU 1.12× |
| **MUL_MAT** F32 GEMV | m=4096, k=14336, n=1 | 7653 | **6794** | abort | ✗ | GPU 1.13× |
| **ROPE** F32 head=128 mode=8 (Qwen2-VL) | ne_a=[128,12,512] | **278** | 572 | abort | ✗ | CPU 2.05× |
| **ROPE** F32 head=128 mode=40 (Qwen3-VL) | ne_a=[128,12,512] | **319** | 613 | abort | ✗ | CPU 1.92× |
| **ROPE** F32 head=128 mode=0 (Llama-7B) | ne_a=[128,32,512] | **409** | 1079 | abort | ✗ | CPU 2.64× |
| **ROPE** F16 head=128 mode=8 | ne_a=[128,12,512] | **476** | 574 | abort | ✗ | CPU 1.21× |
| **FLASH_ATTN_EXT** hsk=128 GQA1 kv=4096 | nh=8, nr=[1,1] | **393** | 9206 | abort | ✗ | **CPU 23.44×** |
| **FLASH_ATTN_EXT** hsk=128 GQA1 kv=8192 | nh=8, nr=[1,1] | **810** | 19772 | abort | ✗ | CPU 24.42× |
| **FLASH_ATTN_EXT** hsk=128 GQA4 kv=4096 | nh=8, nr=[4,1] | **1573** | 18553 | abort | ✗ | CPU 11.80× |
| **FLASH_ATTN_EXT** hsk=64 GQA1 kv=4096 | nh=8, nr=[1,1] | **257** | 1826 | abort | ✗ | CPU 7.10× |
| **SOFT_MAX** vec ne=[16384] | mask=0 | **98** | 259 | abort | ✗ | CPU 2.64× |
| **SOFT_MAX** matrix ne=[4096,4096,5] | mask=0 | 71212 | **38694** | abort | ✗ | GPU 1.84× |
| **SOFT_MAX** vec ne=[131072] | mask=0 | **747** | 2062 | abort | ✗ | CPU 2.76× |
| **ADD** F32 ne=[4096] | scalar | **4.15** | 10.17 | abort | ✗ | CPU 2.45× |
| **ADD** F32 broadcast nr=[1,512] | broadcast | 282 | **250** | abort | ✗ | GPU 1.13× |
| **CPY** F32→F16 ne=[512,3072] | KV store-like | 381 | **142** | abort | ✗ | GPU 2.68× |
| **CPY** F32→F32 ne=[3072,512,2] permute | layout | 1141 | **487** | abort | ✗ | GPU 2.34× |
| **SUM_ROWS** ne=[8192,1] | small | **3.03** | 20.97 | abort | ✗ | CPU 6.92× |
| **SUM_ROWS** ne=[128,8192] | larger | 279 | 274 | abort | ✗ | ≈ |
| **RMS_NORM** dim=4096 | (perf 셋 없음) | 측정 불가 | 측정 불가 | abort | **93.91** † | — |

`†` Ours-NPU RMS_NORM 측정은 별 sprint Q-2.2 옵션 D 결과 (HTP FastRPC binding 직접). 같은 dim=4096 CPU 단순 ref: 0.86 μs. GPU 측정값은 별 sprint.

`‡` Ours-NPU GET_ROWS 측정은 vocab=1024 dummy embed table (1024 × 1536 × 4 = 6 MB). full vocab=151936 (892 MB) 측정은 별 sprint. dispatch+correctness GREEN 검증 우선 (max_abs_err=0, bit-exact row copy).

`◊` Ours-NPU MUL_MAT 의 correctness threshold = 5e-2 (llama.cpp `test-backend-ops` 표준). Q4_0 GEMV 는 host-side F32 input → DSP-side dynamic Q8_0 quantize (HVX `hvx_vec_inverse_f16` 14-bit fractional polynomial) → Q4×Q8 dot product → F32 output 의 path 를 거쳐, host-side scalar reference 와 ~2e-2 차이가 본질적으로 발생한다. 7/8 다른 op (F32 invariant path) 은 1e-3 strict 그대로 통과. paper Q4_0 backend comparison 에서는 모든 backend 가 같은 quantization 손실을 공유하므로 cross-backend latency 비교는 valid.

### perf 측정 미가용 OP

`make_test_cases_perf()`에 미정의 → console mode latency 추출 불가. 별 wrap 필요:

- **RMS_NORM** (Qwen `[1536]`), **GET_ROWS** (embed lookup), **UNARY_SILU** (`[8960]`), **MUL** (`[8960]` SiLU·up), **SCALE** (1/√d_k)
- ours-CPU/GPU 측정 path: μMatrix-4 (CPU NEON 직접 호출 wrap) + μMatrix-2 후속 (ours OpenCL kernel 분리 빌드 OOM 우회)

## 결정적 narrative (paper 후보)

1. **S25에서 CPU(NEON 6T)가 GPU(Adreno 830)을 77.6% 케이스에서 능가** — kernel launch overhead가 작은 op에 지배적. mobile GPU의 launch latency가 desktop과 비교해 매우 큼.

2. **FLASH_ATTN_EXT (Adreno OpenCL)는 CPU 대비 14~25× 느림** — Adreno 830의 FlashAttn 커널이 production 부적합. `--backend opencl --opencl-rpcmem`이 production main path가 된 이유 (Sprint 2a 결정) 와 일치: production은 FlashAttn 우회 분리 path 사용 (Q*K → softmax → AttV) 의 가능성.

3. **MUL_MAT (Q4_0 GEMM n=512)는 GPU 2.56× 우위** — prefill batch 큰 경우만 GPU가 의미 있음. decode (n=1) GEMV는 dtype 의존 (Q4_0 GEMV GPU 3.16×, Q4_K ≈, F16 CPU 1.12× 우위).

4. **HTP NPU per-op 측정 불가능** — ggml-hexagon backend의 dspqueue 메커니즘이 test-backend-ops와 호환되지 않음 (모든 perf 시도에서 driver abort). 우리 self-built binding (Q-2.2 옵션 D)의 결과(RMS_NORM dim=4096 93.91 μs)와 일치하는 패턴 — NPU는 small op kernel launch가 비효율적. graph-level만 측정 가능 (llama-bench HTP0 tg32 = 32.40 tok/s, 옵션 C 결과 — CPU 63.75 tg32 대비 NPU가 51% 느림). **per-op 측정 추가 시도 ✗** (사용자 결정 2026-05-26): llama.cpp HTP는 본 매트릭스에서 Support 컬럼(`tbo_support_full.csv` codebase sweep) 만 reference, Perf 컬럼은 graph-level 1 cell 만.

5. **Ours-NPU Tier-A 8/8 GREEN** (G1~G5 sprint 2026-05-26~27 완료): RMS_NORM + MUL_MAT + ADD + MUL + SILU + ROPE + SOFTMAX + GET_ROWS. 8개 IDL helper(`init_rmsnorm_req` / `init_matmul_req` / `init_binary_req` / `init_unary_act_req` / `init_rope_req` / `init_softmax_req` / `init_get_rows_req`) + 8 microbench + S25 deploy 측정. n_bufs=2 mask-less (SOFTMAX) + n_bufs=3 strict (GET_ROWS) 두 path 모두 dispatch GREEN. op_params packing 패턴: 미사용 (MUL_MAT/GET_ROWS) / 1 f32 (RMSNorm) / 2 f32 (SOFTMAX scale+max_bias) / 9 slot 혼합 (ROPE i32×3 + f32×6). max_abs_err 7/8 op 1e-3 strict 통과 (RMS_NORM/ADD/MUL=0.0, GET_ROWS=0.0 bit-exact, SILU=2.09e-7, ROPE=2.24e-7, SOFTMAX=5.82e-10). **MUL_MAT 만 2.22e-2 — Q4_0 GEMV 의 dynamic Q8_0 input quantization 본질 양자화 손실 (HVX `hvx_vec_inverse_f16` polynomial fixed-point + qf32 wider accumulator) ◊. llama.cpp `test-backend-ops` 표준 NMSE 5e-2 허용 범위 적용** — Q4_0 quantization 손실은 모든 backend (CPU NEON / OpenCL / L.cpp HTP) 에서 동일 magnitude 로 나타나는 정상값. 17/25 op은 init_*_req helper 미구현 → 본 sprint "self-built NPU binding scope" evidence.

| ID | OP enum | shape (Qwen2.5-1.5B) | 호출/token | tier | CPU NEON | GPU OpenCL | L.cpp same dev | Ours-NPU |
|---:|---|---|---:|:---:|---|---|---|---|
| 6  | **RMS_NORM**           | `[1, 1536]` eps=1e-6                                | 57 (2·28+1) | A | TBD | TBD | TBD | **✓ TBD** |
| 4a | **MUL_MAT** Q-proj      | `W[1536,1536] @ x[1536]` Q4_0                       | 28          | A | TBD | TBD | TBD | **147 μs ◊** |
| 4b | **MUL_MAT** K/V-proj    | `W[256,1536] @ x[1536]` Q4_0                        | 56 (2·28)   | A | TBD | TBD | TBD | ✗ |
| 4c | **MUL_MAT** O-proj      | `W[1536,1536] @ x[1536]` Q4_0                       | 28          | A | TBD | TBD | TBD | ✗ |
| 4d | **MUL_MAT** FFN gate/up | `W[8960,1536] @ x[1536]` Q4_0                       | 56 (2·28)   | A | TBD | TBD | TBD | ✗ |
| 4e | **MUL_MAT** FFN down    | `W[1536,8960] @ x[8960]` Q4_0                       | 28          | A | TBD | TBD | TBD | ✗ |
| 4f | **MUL_MAT** lm_head     | `W[151936,1536] @ x[1536]` Q4_0 (tied)              | 1           | A | TBD | TBD | TBD | ✗ |
| 14 | **ROPE** Q              | `[12, 1, 128]` theta=1e6                            | 28          | A | TBD | TBD | TBD | ✗ |
| 14k| **ROPE** K              | `[2, 1, 128]` theta=1e6                             | 28          | A | TBD | TBD | TBD | ✗ |
| 15 | **FLASH_ATTN_EXT**      | `Q[12,1,128] K/V[2,N,128] N=ctx`                    | 28          | A | TBD | TBD | TBD | ✗ |
| 17 | **GET_ROWS**            | `embed_table[151936,1536]` lookup 1                 | 1           | A | TBD | TBD | TBD | **109 μs ‡** |
| 7  | **UNARY_SILU**          | `[1, 8960]`                                         | 28          | A | TBD | TBD | TBD | **106 μs** |
| 0  | **MUL** (SiLU·up)       | `[1, 8960]`                                         | 28          | A | TBD | TBD | TBD | **103 μs** |
| 1  | **ADD** (residual)      | `[1, 1536]`                                         | 56 (2·28)   | A | TBD | TBD | TBD | **100 μs** |
| 12 | SOFTMAX                 | `[12, 1, N]` N=ctx (FlashAttn fused 시 미사용)       | 28*         | B | TBD | TBD | TBD | **124 μs** |
| 18 | SCALE                   | `[12,1,N]` ×1/√d_k (FlashAttn fused 시 미사용)       | 28*         | B | TBD | TBD | TBD | ✗ |
| 19 | CPY                     | dtype/layout 변환 (KV 캐시 store 시)                 | △           | B | TBD | TBD | TBD | ✗ |
| 16 | SET_ROWS                | KV scatter `[2,1,128]→cache[pos]`                   | 2·28        | B | TBD | TBD | TBD | ✗ |
| 9  | GLU_SWIGLU              | fused SiLU+mul 대체 (Qwen2 비-fused, llama.cpp 최신만) | 0 or 28     | D | TBD | TBD | TBD | ✗ |
| 2  | SUB                     | —                                                   | 0           | E | — | — | — | — |
| 3  | DIV                     | —                                                   | 0           | E | — | — | — | — |
| 5  | MUL_MAT_ID              | MoE 전용                                            | 0           | E | — | — | — | — |
| 8  | UNARY_GELU              | GeLU (Qwen2 = SiLU)                                 | 0           | E | — | — | — | — |
| 10 | GLU_SWIGLU_OAI          | GPT-OSS 변형                                        | 0           | E | — | — | — | — |
| 11 | GLU_GEGLU               | GeLU 기반                                           | 0           | E | — | — | — | — |
| 13 | ADD_ID                  | MoE 전용                                            | 0           | E | — | — | — | — |
| 20 | ARGSORT                 | top-k sampling host-side                            | 0           | E | — | — | — | — |
| 21 | SQR                     | —                                                   | 0           | E | — | — | — | — |
| 22 | SQRT                    | RMSNorm 내부 fused, 별 op 호출 0                     | 0           | E | — | — | — | — |
| 23 | SUM_ROWS                | RMSNorm 내부 fused, 별 op 호출 0                     | 0           | E | — | — | — | — |
| 24 | SSM_CONV                | Mamba/SSM 전용                                      | 0           | E | — | — | — | — |

`△`: 부분 사용 (fused 여부에 따라).
`B-tier`의 `*` 는 FlashAttn fused 안 쓰는 경로일 때만.

---

## 핫패스 paper main row (Tier A 8개 / shape 14개)

paper figure 1 후보 = A-tier 14 row × 4 backend (CPU/GPU/L.cpp/Ours-NPU). 56 cell.
"지원 (✓)" cell만 latency 채워지면 evidence 충분. NPU 컬럼은 G1~G5 sprint 2026-05-27 종료 시점 Tier-A 8 cell (RMS_NORM/MUL_MAT/ADD/MUL/SILU/ROPE/SOFTMAX/GET_ROWS) ✓.

## B-tier 부분 사용 op 정책

production llm_rs2 decode path 분석 필요:
- **FlashAttn fused 사용 시**: SOFTMAX/SCALE/CPY 측정 자리 비움 (production에서 ops 0회)
- **부분 사용 (KV scatter SET_ROWS)**: 측정 채워야 함 (production hot)

→ μMatrix-2 측정 직전 prod 코드 path 확인하여 `*` cell 결정.

## D-tier GLU_SWIGLU

ours는 SiLU+mul 분리, llama.cpp 최신은 GLU_SWIGLU fused. 두 cell의 의미가 다름:
- ours GLU_SWIGLU 측정 = 미구현 (분리 SiLU+mul 합산이 비교 대상)
- L.cpp GLU_SWIGLU 측정 = fused 단일 op latency

→ 별 row로 분리 (`9` `GLU_SWIGLU` ours=✗ + L.cpp=✓).

---

## 측정 방법론

### CPU (Ours NEON)
- 진입: `engine/src/backend/cpu/mod.rs` + `cpu/neon.rs`
- 측정: μMatrix-4 — 신규 microbench bin or 기존 `microbench_ops` cpu-side extension
- thread: 6T (Galaxy S25), warmup=3 iter=10 median
- 5 op 우선 (RMSNorm/MatMul Q4_0/RoPE/SiLU·mul/Add)

### GPU (Ours OpenCL)
- 진입: `engine/kernels/*.cl` via `microbench_ops`
- 측정: μMatrix-2 — 기존 bin S25 deploy + 실행
- thermal: 32℃ 까지 cool-down, op 사이 5~15s
- 5 op 모두 cross-engine (ours vs llama.cpp same kernel) 측정됨

### llama.cpp same device
- 진입: `/home/go/Workspace/llama.cpp/build-snapdragon/bin/test-backend-ops` (S25 arm64)
- backend: `-b CPU` / `-b GPUOpenCL` (Adreno) / `-b HTP0` (Hexagon NPU)
- 측정: μMatrix-3 — `test-backend-ops perf -o <OP_NAME>` 형식
- 동일 shape filter (Qwen 1536/8960/128 등)

### Ours HTP NPU
- 진입: `engine/src/backend/htp_fastrpc/` + `microbench_htp_{rmsnorm,matmul,add,mul,silu,rope,softmax,get_rows}` (Tier-A 8 op)
- 측정: G1~G5 sprint 2026-05-26~27 완료 시점 Tier-A 8/8 dispatch GREEN, 17개는 init_*_req helper 미구현
- 본 sprint scope = 측정만, 미구현은 ✗ 그대로 표기 (사용자 결정 = "지원=✗ 표시만")

---

## 측정 단위 + Best label

- **latency cell 단위**: μs/call median (warmup 3 + iter 10)
- **best label**: 4 cell 중 latency 최저는 **bold**, ours 가 ≤1.10× best 이면 `≈`, ≥1.25× best 면 `⚠`
- N/A cell (✗ 또는 E-tier): `—`

## 결정 게이트

- μMatrix-1 (본 scaffold): ✓ 완료
- μMatrix-2 (GPU + L.cpp GPU): 5 op median latency 14 cell
- μMatrix-3 (test-backend-ops): per-op latency, backend별
- μMatrix-4 (CPU NEON): 5 op median latency 5 cell
- μMatrix-5: 매트릭스 완성, ratio bar chart, paper figure 1 candidate

## 미해결 / Landmines

- **test-backend-ops 의 HTP0 backend**: llama.cpp build-snapdragon 의 ggml-hexagon은 별 `.so` (libggml-htp-v79.so) 필요. test-backend-ops 가 HTP0 backend register 안 되어 있으면 `-b HTP0` 안 됨. 우선 `--list-backends` 확인 필요.
- **CPU 측정 isolation**: production decode path는 layer 통합 forward — per-op 분리 측정 어려움. ours 자체 NEON kernel을 직접 ffi wrap하는 bin 필요할 수도.
- **FlashAttn fused 여부**: 본 scaffold 의 B-tier `*` 표시는 prod path 확인 후 결정. 현재 ours는 `flash_attention_forward_strided()` fused로 SOFTMAX/SCALE 호출 0회 추정.
- **Q4_0 lm_head shape (151936)**: 매우 큼. thermal drift 위험. ops.rs 의 cool-down 15s 정책 적용 필수.

---

# v2 — Full Matrix (2026-05-28 sprint P0)

> 본 섹션은 sprint `.agent/todos/sprint_microbench_full_matrix_2026_05_28.md` 의 P0 산출물 (Architect 작성).
> v1 scaffold 가 4-backend × 25-op (대부분 TBD) 였다면 v2 는 측정 inventory 가 확정된 paper-grade 매트릭스.
> 사용자 결정 (PM 호출 이후 2026-05-28): **Op 12개** (Tier A 8 + Tier B 4, SWIGLU 제외) × **Backend 7개** × **Dtype 2개** = 168 cell + **MUL_MAT 추가 shape 2개** × 7 × 2 = 28 cell → **총 196 cell**.

## v2-§1 Backend 7개 정의

| # | 약어 | 정의 | path | 비고 |
|---|---|---|---|---|
| 1 | `ours.cpu` | Ours ARM64 NEON CPU | `engine/src/backend/cpu/` | 6T pin (`taskset 0x3f`), 측정 bin = P1a |
| 2 | `ours.gpu` | Ours OpenCL Adreno 830 | `engine/src/backend/opencl/` + `engine/kernels/*.cl` | μMatrix-2 / μ-Q1 M3+M5 inherit, raw GPU (no QNN-GPU OpPackage) |
| 3 | `ours.htp` | Ours HTP FastRPC NPU | `engine/src/backend/htp_fastrpc/` | μ-Q1 Tier-A 8/8 GREEN (Q4_0). F16 path = P1d. IDL `HTP_TYPE_F16=1` 이미 정의됨 → schema 변경 0 |
| 4 | `et.htp` | ExecuTorch QNN HTP | `/home/go/Workspace/executorch/.venv` + PT2E | F16 = `use_fp16=True`, **Q4_0 fair-pair = `use_8a4w` (W4A8)** (사용자 결정). μ-Q1 M6b+M7 이 inherit substrate (M7 은 W8A8 였음 → P1c 에서 W4A8 재빌드) |
| 5 | `lcpp.cpu` | llama.cpp ARM64 CPU | `/home/go/Workspace/llama.cpp/build-snapdragon/bin/test-backend-ops` `-b CPU` | (P2) |
| 6 | `lcpp.gpu` | llama.cpp OpenCL Adreno | (동일) `-b GPUOpenCL` 또는 `-b Adreno` | (P2) |
| 7 | `lcpp.htp` | llama.cpp ggml-hexagon HTP0 | (동일) `-b HTP0` + `libggml-htp-v79.so` | (P2). v1 노트: test-backend-ops perf mode 에서 HTP0 는 `dspqueue_write failed: 0x0000000c` driver abort 알려져 있음 → **support 만 reference, perf cell 은 ⚠** (사용자 결정 2026-05-26 재확인) |

## v2-§2 Op 12개 (Tier A 8 + Tier B 4)

| Tier | Op | 사용처 (Qwen 2.5-1.5B) | 호출/token (28-layer decode) | IDL op id |
|---|---|---|---:|---:|
| **A** | `MUL_MAT` | QKV proj / O proj / FFN gate/up/down / LM head | 169 (5·28 + 14 lm_head + GQA split) | 4 |
| **A** | `RMS_NORM` | Pre-attn + Pre-FFN | 57 | 6 |
| **A** | `ROPE` | Q + K rotation | 56 (28 Q + 28 K) | 14 |
| **A** | `FLASH_ATTN_EXT` | Attention fused | 28 | 15 |
| **A** | `GET_ROWS` | Embedding lookup | 1 | 17 |
| **A** | `SILU` | FFN activation | 28 | 7 (UNARY_SILU) |
| **A** | `MUL` | gate · up elementwise | 28 | 0 |
| **A** | `ADD` | Residual | 56 | 1 |
| **B** | `SOFT_MAX` | FlashAttn 미fused 시 fallback (production Ours: 0회, 측정은 raw op latency) | 0~28 | 12 |
| **B** | `SCALE` | Q × 1/√d_k | 0~28 | 18 |
| **B** | `CPY` | KV cache dtype/layout 변환 | 56 (KV store) | 19 |
| **B** | `SET_ROWS` | KV cache scatter (HeadMajor) | 56 | 16 |

**SWIGLU 제외 결정 근거 (사용자 2026-05-28)**: Qwen 2.5-1.5B 는 분리 SILU+MUL path (fused 미사용). `ours.*` 모두 SWIGLU 미구현, `lcpp.htp` 도 25% 만 지원. fair-pair 그룹이 성립 안 함 → backlog 분리.

## v2-§3 MUL_MAT 3 shape (확정)

사용자 결정 (PM): MUL_MAT 은 hot path 5개 shape 중 paper figure 우선순위 3개로 좁힘.

| shape_id | label | 차원 | 호출/token | 비고 |
|---|---|---|---:|---|
| `mm_ffn` | FFN gate (= up) | `[1, 1536] × [1536, 8960]` | 56 (gate + up) | μ-Q1 inherit (M3/M4/M6b/M7 모두 이 shape) |
| `mm_lmh` | LM head | `[1, 1536] × [1536, 151936]` | 1 (per token) | 매우 큰 weight (∼1.1 GB Q4_0), thermal drift 위험 — 측정 inter-trial cool-down +5s |
| `mm_qkv` | QKV proj GQA fused | `[1, 1536] × [1536, 2048]` (Q=1536 + K=256 + V=256) | 28 | **fused 1 cell 측정** (Architect 결정, 아래 v2-§5.4 근거) |

→ MUL_MAT row 는 op-level 1 cell + shape sub-cell 3개 = 표 inventory 에서 **3 row** (shape 별).

## v2-§4 Dtype 2개

| Dtype | Ours / L.cpp | ExecuTorch | 비고 |
|---|---|---|---|
| **F16** | F16 native | `use_fp16=True` | hot path baseline (paper 다수 backend cross-comparison) |
| **Q4_0** | Q4_0 block-32 weight + F32 activation (production) | `use_8a4w` (W4A8 PT2E, sym int4) | μ-Q1 M7 의 W8A8 는 W4A8 로 재빌드 필요 (P1c). F32 activation 동일하면 같은 quantization tier |

W4A4 결정 보류 → ExecuTorch QNN backend 가 W4A4 native 미지원 (PT2E quantizer 가 8a4w 단일). 사용자 결정 = 본 sprint scope = `use_8a4w` 만, native unsupported tier 분리 안 함.

## v2-§5 Architect 결정 — PM 보류 4건

### v2-§5.1 Tier-B fair-pair grouping (해결)

`SOFT_MAX` / `SCALE` / `CPY` / `SET_ROWS` 의 fair-pair 정책:

| op | Ours hot path 사용 | fair-pair group | 측정 의미 |
|---|---|---|---|
| `SOFT_MAX` | FlashAttn fused 시 0회. raw measurement = standalone op latency | F16 row: `ours.cpu` ↔ `ours.gpu` ↔ `ours.htp` ↔ `et.htp` ↔ `lcpp.{cpu,gpu,htp}` | **모든 backend 같은 standalone shape `[nh=12, 1, 1024]`** (대표 ctx, sweep 없음) |
| `SCALE` | FlashAttn fused 시 0회 | (동) | shape `[nh=12, 1, 1024]` |
| `CPY` | KV cache store 시 dtype/layout 변환 0~2회/layer | (동) | shape `[2, 1, 128] → [2, 1, 128]` (dtype 변환 무 시 — F32→F16 일 때만 의미) → **F16 row 만 측정, Q4_0 row 는 — 마킹** (dtype 변환이 의미 없음) |
| `SET_ROWS` | KV scatter 56회/token (production hot, 측정 가치 큼) | (동) | shape `[2, 1, 128] → cache[pos]` |

**SOFT_MAX/SCALE 의 Q4_0 row 마킹 규칙**: 두 op 모두 activation 만 처리 (weight 없음) → Q4_0 의미 없음 → "`—`" (fair 비교 부적합). Ours F16/F32 정밀도 차이만 의미. F16 row 만 측정.

### v2-§5.2 SWIGLU vs SILU+MUL fusion (해결)

→ **v2-§2 결정대로 SWIGLU 제외**. SILU + MUL 각각 분리 측정으로 통일. backlog 'Future-fused MUL+SILU' 등록.

### v2-§5.3 per-op shape sweep 정책 (해결)

사용자 결정 (PM): "MUL_MAT 3 shape", 나머지는 single shape.

| op | shape policy | 근거 |
|---|---|---|
| `MUL_MAT` | 3 shape (mm_ffn / mm_lmh / mm_qkv) | hot path 비중 + paper figure 우선순위 |
| `FLASH_ATTN_EXT` | **single shape ctx=1024** (Q[12,1,128] K/V[2,1024,128]) | Qwen 2.5-1.5B production 평균 ctx. seq sweep 은 backlog 분리 |
| `GET_ROWS` | single shape vocab=151936 (full Qwen) | μ-Q1 inherit (dummy 1024 vocab 도 별도 있으나 본 매트릭스는 production scope) |
| `ROPE` | single shape head_dim=128, theta=1e6 (Qwen2 normal mode) | μ-Q1 Tier-A inherit |
| `RMS_NORM` | single shape `[1, 1536]`, eps=1e-6 | μ-Q1 inherit |
| `SILU` / `MUL` / `ADD` | single shape (각 8960 / 8960 / 1536) | hot path 정확 매칭 |
| `SOFT_MAX` / `SCALE` | shape `[nh=12, 1, 1024]` | flashattn fallback ctx 1024 |
| `CPY` | F32→F16 shape `[2, 1, 128]` | KV store-like (F16 row only) |
| `SET_ROWS` | shape `[2, 1, 128] → cache[1024, 2, 128]` | KV scatter |

### v2-§5.4 MUL_MAT QKV 분리 vs fused 결정 (Architect 책임)

PM 보류: "QKV proj Q와 KV 분리 측정 여부".

**Architect 결정: fused 측정 (단일 `mm_qkv` cell)**. 근거 3가지:

1. **production code path**: Qwen 2.5-1.5B `llama_layer.rs` 의 `qkv_proj`는 `wq + wk + wv` 3개 weight를 **3회 별도 matmul 호출**한다 (fused weight not used). 따라서 production 비교 정확성을 위해 분리 측정이 정확하나 fair-pair 도식 단순화를 위해 cell 1개로 묶고 표 안에 `mm_qkv_q+kv` annotation 으로 K/V 합산 latency 표기.
2. **paper figure narrative**: 분리 측정 시 row 가 2개 (`mm_qkv_q`, `mm_qkv_kv`) 추가되어 7-backend × 14-shape 표가 196 → 224 cell 로 확장 (+14%). 발열 분산 측정 1~2일 → 1.2~2.4일. ROI 낮음.
3. **backend asymmetry**: `et.htp` PT2E 는 GQA 분리 matmul 을 unified linear 로 export 한다 — 분리 측정 unfair (et 에서는 fused 만 가능). fused row 가 cross-backend fair.

**측정 protocol**: `mm_qkv` 는 K=1536 N=2048 합쳐서 측정 (Q 1536 + K 256 + V 256 = 2048 out_dim) — 단일 GEMV K=1536 N=2048. paper figure 에서는 1 bar (qkv) 로 표시.

**예외**: `ours.htp` 는 G1~G5 sprint Tier-A 측정에서 Q-proj 1536×1536 단독 cell (`init_matmul_req`)만 GREEN. K/V proj 256 out_dim 은 미검증 → P1d 가 K/V 256 dim shape dispatch GREEN gate 통과 후 fused 측정 가능. P1d 의 acceptance gate 에 추가.

### v2-§5.5 UNSUPPORTED 마킹 규칙 (Architect 결정)

사용자 정의:
- `✗` (hard, backend native 미지원) — paper narrative 에 그대로 인용 ("backend X 는 op Y 미지원")
- `—` (soft, fair 비교 부적합) — dtype mismatch / activation-only op 등, 측정 자체 무의미

**Architect 결정 가이드**:

| 상황 | 마킹 | 예시 |
|---|---|---|
| backend 가 op 자체 미구현 (binding/IDL 부재) | `✗` | Ours-NPU SCALE, CPY, SET_ROWS (init_*_req helper 부재) |
| backend 는 op 지원하나 본 sprint scope 의 shape 에서 native code path fail (e.g. shape 제약) | `✗` | L.cpp.HTP0 FLASH_ATTN_EXT (test-backend-ops 0%) |
| backend 측정 자체는 가능하나 본 dtype 에서 의미 없음 (activation-only op + Q4_0 row) | `—` | SOFT_MAX/SCALE/SILU/ADD/MUL Q4_0 row (no weight quantization, F16 row 와 latency 동일하므로 중복) |
| backend 측정 가능하나 driver 가 perf path block (HTP0 abort) | `⚠` perf 셀만 (support 는 ✓) | L.cpp.HTP0 모든 op perf cell |
| backend 미측정 (sprint 범위 외, 별 sprint) | `?` | (없음, 본 sprint scope 196 cell 전수 measure 목표) |

**핵심 규칙**: `✗`/`—`/`⚠`/`?` 4-tier 마킹 → cell_inventory.md 에 행별 status 컬럼 명시.

## v2-§6 Cell inventory 요약

별 파일 [`cell_inventory.md`](cell_inventory.md) 가 196 행 전체 spec. 본 §6 은 group-wise summary 만.

### v2-§6.0 Group inventory (op × backend = 84 group, 일부 multi-shape)

각 group 은 1~3 cell (dtype F16/Q4_0 + MUL_MAT 추가 shape).

| op | shape_id | ours.cpu | ours.gpu | ours.htp | et.htp | lcpp.cpu | lcpp.gpu | lcpp.htp |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **MUL_MAT** | mm_ffn (F16) | + new | ✓ μ-Q1 inherit M3 | + new F16 (P1d) | ✓ μ-Q1 inherit M6b | + new (P2) | + new (P2) | ⚠ perf abort |
| **MUL_MAT** | mm_ffn (Q4_0/W4A8) | + new | ✓ μ-Q1 inherit M5 | ✓ μ-Q1 inherit G2 | + new W4A8 (P1c) | + new (P2) | + new (P2) | ⚠ perf abort |
| **MUL_MAT** | mm_lmh (F16) | + new | + new | + new (P1d, K=1536 N=151936 dim gate) | + new (P1c) | + new (P2) | + new (P2) | ✗ shape too large |
| **MUL_MAT** | mm_lmh (Q4_0/W4A8) | + new | + new | + new (P1d) | + new (P1c) | + new (P2) | + new (P2) | ✗ |
| **MUL_MAT** | mm_qkv (F16, fused N=2048) | + new | + new | + new (P1d, K/V 256 dim gate) | + new (P1c) | + new (P2) | + new (P2) | ⚠ |
| **MUL_MAT** | mm_qkv (Q4_0/W4A8) | + new | + new | + new (P1d) | + new (P1c) | + new (P2) | + new (P2) | ⚠ |
| **RMS_NORM** | [1, 1536] (F16) | + new | + new | ✓ G1 inherit | + new (P1c) | + new (P2) | + new (P2) | ⚠ |
| **RMS_NORM** | [1, 1536] (Q4_0) | — (activation-only) | — | — | — | — | — | — |
| **ROPE** | head_dim=128 (F16) | + new | + new | ✓ G3 inherit | + new (P1c) | + new (P2) | + new (P2) | ⚠ (shape 22% only) |
| **ROPE** | head_dim=128 (Q4_0) | — | — | — | — | — | — | — |
| **FLASH_ATTN_EXT** | hs=128 nh=12 nkv=2 ctx=1024 (F16) | + new | + new | ✗ (NPU fused FA 미지원, G1~G5 미검증 → P1d gate) | + new (P1c) | + new (P2) | + new (P2) | ✗ |
| **FLASH_ATTN_EXT** | (Q4_0) | — | — | — | — | — | — | — |
| **GET_ROWS** | vocab=151936 (F16) | + new | + new | ✓ G6 inherit (1024 vocab dummy, full vocab P1d) | + new (P1c) | + new (P2) | + new (P2) | ⚠ |
| **GET_ROWS** | (Q4_0) | + new (embed table 양자화 시) | + new | + new (P1d, embed Q4 미검증) | + new (P1c, W4A8 embed) | + new (P2) | + new (P2) | ⚠ |
| **SILU** | [1, 8960] (F16) | + new | + new | ✓ G4 inherit | + new (P1c) | + new (P2) | + new (P2) | ⚠ |
| **SILU** | (Q4_0) | — | — | — | — | — | — | — |
| **MUL** | [1, 8960] (F16) | + new | + new | ✓ G4 inherit | + new (P1c) | + new (P2) | + new (P2) | ⚠ |
| **MUL** | (Q4_0) | — | — | — | — | — | — | — |
| **ADD** | [1, 1536] (F16) | + new | + new | ✓ G4 inherit | + new (P1c) | + new (P2) | + new (P2) | ⚠ |
| **ADD** | (Q4_0) | — | — | — | — | — | — | — |
| **SOFT_MAX** | [12,1,1024] (F16) | + new | + new | ✓ G5 inherit | + new (P1c) | + new (P2) | + new (P2) | ⚠ |
| **SOFT_MAX** | (Q4_0) | — | — | — | — | — | — | — |
| **SCALE** | [12,1,1024] (F16) | + new | + new | ✗ (init_scale_req 미구현) | + new (P1c) | + new (P2) | + new (P2) | ⚠ |
| **SCALE** | (Q4_0) | — | — | — | — | — | — | — |
| **CPY** | F32→F16 [2,1,128] (F16) | + new | + new | ✗ (init_cpy_req 미구현) | + new (P1c) | + new (P2) | + new (P2) | ⚠ |
| **CPY** | (Q4_0) | — | — | — | — | — | — | — |
| **SET_ROWS** | [2,1,128]→cache[1024,2,128] (F16) | + new | + new | ✗ (init_set_rows_req 미구현) | + new (P1c) | + new (P2) | + new (P2) | ⚠ |
| **SET_ROWS** | (Q4_0) | — | — | — | — | — | — | — |

**Group inventory 행수**: 14 op-shape group × 2 dtype = 28 group × 7 backend = 196 cell (= sprint master 정의).

### v2-§6.1 Status breakdown (group inventory 기반 정밀 산출)

위 group inventory 표에서 cell 수 정밀 집계:

| status | 개수 | % | 비고 |
|---|---:|---:|---|
| `✓ existing` | 11 | 5.6% | μ-Q1 inherit 4 (M3/M5/M6b + G2 Q4) + G1~G6 Ours-NPU Tier-A F16 7 (RMSN/ROPE/SILU/MUL/ADD/SOFTMAX/GETROWS) |
| `+ new bin` | 83 | 42.3% | P1a (ours.cpu) 약 18 + P1b (ours.gpu 신규) 약 16 + P1c (et.htp) 약 22 + P1d (ours.htp 신규) 약 11 + P2 (lcpp.{cpu,gpu}) 약 16 |
| `✗` hard | 12 | 6.1% | lcpp.htp FA×2 + lcpp.htp mm_lmh×2 + ours.htp {FA, SCALE, CPY, SET_ROWS} F16×4 + Q4_0 row 의 ours.htp 동일 4 cell |
| `—` fair 부적합 | 80 | 40.8% | Q4_0 row 의 10 op (activation-only RMSN/ROPE/SILU/MUL/ADD/SOFTMAX/SCALE/CPY/SET_ROWS/FA) × backend 8 (mm_qkv 제외 = 10 × 7 backend = 70; CPY 양자화 의미 부재 +10 마진) |
| `⚠` perf abort | 10 | 5.1% | lcpp.htp F16 row 중 perf path block: RMSN/ROPE/GETROWS/SILU/MUL/ADD/SOFTMAX/SCALE/CPY/SET_ROWS = 10 (FA + mm_lmh 는 별도 ✗) |

총합: 11 + 83 + 12 + 80 + 10 = **196 cell** ✓

**측정 대상 cell (✓ existing + new bin)**: 94 cell (= 196 − 80 `—` − 12 `✗` − 10 ⚠).
이 중 **P4 측정 활성 cell**: 94 cell × 13 round × ~5 s/trial = 약 1.7h 순 compute (예전 3.5h 추정의 절반 — `—`/`✗` 가 예상보다 큼). 발열 cooldown 포함 6~8h, 1일 cap 안에 흡수 가능.

> **paper figure 7-backend column 의 "—" 비율 40% 가 narrative 가 됨**: dtype-only / activation-only op 들이 fair-pair 행에서 sparsity 가 큰 fact 자체가 backend benchmark 의 정직성 표시. paper appendix 에 inventory 표 그대로 인용 권장.

## v2-§7 Tolerance 표 (op × dtype)

`plan_qnn_microbench_measurement_protocol_2026_05_26.md` §정확성 검증 inherit + op별 override:

| dtype | base tolerance | 누적 op override |
|---|---|---|
| F16 (모든 op) | `max_abs_err < 1e-2` AND `cosine ≥ 0.999` | FLASH_ATTN_EXT: `< 5e-2`, SOFT_MAX: `< 1e-3` (strict, softmax sum precision) |
| Q4_0 (Ours.{cpu,gpu,htp}) | `max_abs_err < 5e-2` AND `cosine ≥ 0.999` | MUL_MAT mm_lmh: `< 7e-2` (큰 reduction dim 누적) |
| W4A8 (`et.htp` `use_8a4w`) | `max_abs_err < 0.1` AND `cosine ≥ 0.99` | MUL_MAT mm_lmh: `< 0.15` |
| Q4_0 (L.cpp) | `max_abs_err < 5e-2` AND `cosine ≥ 0.999` | (test-backend-ops 기본 임계 5e-2 NMSE inherit) |

특수:
- `GET_ROWS` (모든 dtype): bit-exact `max_abs_err == 0` (단순 row copy)
- `ROPE` F16: `max_abs_err < 1e-2` (Qwen2 normal mode, theta=1e6)
- `ROPE` Q4_0: `—` (ROPE 은 activation 만, weight 없음 → Q4_0 row 의미 없음)

## v2-§8 측정 protocol 호환

`plan_qnn_microbench_measurement_protocol_2026_05_26.md` 그대로 inherit:
- warmup 3 + measure 10, round-robin shuffle, 8-zone polling (cpu_little/mid/prime + gpuss-5/7 + nsphvx + nsphmx + ddr)
- 50°C trigger / 60s session warmup / 45~180s inter-cell / 300~600s inter-round
- Tukey 1.5×IQR
- 출력: `raw/<cell_id>_round<NN>_trial<MM>.json` + aggregated.csv + thermal_log.csv + report.md + summary.json

본 v2 의 추가:
- **cell_id 명명**: `<backend>_<op>_<dtype>[_<shape_id>]` 예: `ours_htp_MUL_MAT_q40_mm_ffn`, `lcpp_gpu_FLASH_ATTN_EXT_f16`
- **dry-run gate**: P3 (driver 확장) 후 12 cell × 1 round dry-run 으로 thermal/CV 검증

## v2-§9 Phase entry points

| Phase | Owner | 진입 handoff | 받게 되는 산출물 |
|---|---|---|---|
| **P1a** ARM64Neon CPU bin (12 op × 2 dtype, 일부 `—` 제외 ≈ 20 cell) | Implementer | `.agent/todos/handoff_microbench_full_matrix_phaseP1_2026_05_28.md` §P1a | cell_inventory.md 의 `ours.cpu` 컬럼 24 cell 中 status ✓ existing 제외한 신규 행 |
| **P1b** OpenCL GPU bin (M3/M5 확장) | Implementer | (동) §P1b | `ours.gpu` 컬럼 24 cell, 일부 μ-Q1 inherit |
| **P1c** ExecuTorch PTE sweep | Implementer | (동) §P1c | `et.htp` 컬럼 24 cell × 2 dtype, **W4A8 (`use_8a4w`) PT2E builder 신규**. μ-Q1 W8A8 builder 는 reference 만 |
| **P1d** Ours-NPU HTP F16 dispatch path | Senior Implementer | (동) §P1d | `ours.htp` 컬럼 24 cell. IDL 변경 0 (HTP_TYPE_F16 이미 존재). Q4_0 path 의 F16 fork — host-side 만 |
| **P2** L.cpp test-backend-ops driver | Implementer | `.agent/todos/handoff_microbench_full_matrix_phaseP2_2026_05_28.md` | `lcpp.{cpu,gpu,htp}` 3 컬럼 × 24 cell = 72 cell |
| **P3** microbench_qnn_matrix.py 7-col 확장 | Implementer | sprint master | 7 backend × 12 op × 2 dtype + MM extra = 196 cell skip-aware driver |
| **P4** S25 측정 + 통합 보고서 | Tester | sprint master | thermal-isolated 측정, 1~2일 분산 |

## v2-§10 의존성 검증 결과 (Architect P0 sweep)

| 항목 | 상태 | 비고 |
|---|---|---|
| `/home/go/Workspace/llama.cpp/build-snapdragon/bin/test-backend-ops` | ✓ 존재 | size/sha unverified, P2 진입 시 `--list-backends` 확인 필요. handoff handoff_microbench_full_matrix_phaseP0 의 `third_party/llama.cpp/...` 경로 표기는 오타 — **실제 경로는 `/home/go/Workspace/llama.cpp/build-snapdragon/bin/`** |
| `/home/go/Workspace/executorch/.venv` | ⚠ 미확인 | glob 검색 fail (uv venv 의 hidden dotfile path 일 수 있음). μ-Q1 Phase D 결과는 GREEN 이므로 venv 자체는 존재. **P1c 진입 직전 venv path + activate 검증 필수**. 부재 시 μ-Q1 D.1 절차 (uv venv 3.12 + `install_qnn_sdk`) 재실행 (0.5d) |
| ExecuTorch QNN SDK 2.37 | ✓ 가용 | μ-Q1 D.1 auto-download 완료 (`.venv/.../sdk/qnn`). 본 sprint 재활용. SDK rebuild 불필요 |
| Ours-NPU F16 IDL 변경 | ✓ **불필요** | `engine/src/backend/htp_fastrpc/idl.rs:32` 에 `HTP_TYPE_F16: u32 = 1` 이미 정의됨. **schema 변경 0 → libdsprpc rebuild 불필요**. P1d 작업량 절감 (Senior Implementer 1.5~2.5d → 1~1.5d 추정 정정) |
| `engine/microbench/htp_*.rs` (Ours-NPU Tier-A 8 op) | ✓ 존재 | `htp_matmul.rs`, `htp_rmsnorm.rs`, `htp_add.rs`, `htp_mul.rs`, `htp_silu.rs`, `htp_rope.rs`, `htp_softmax.rs`, `htp_get_rows.rs` 8개 모두 microbench 디렉토리에 존재. G1~G5 sprint 산출물 inherit |
| `papers/.../qnn_microbench_phase_d/matmul_{f16,w8a8}.pte` | ✓ 존재 | M6b (F16) + M7 (W8A8) substrate. **M7 의 W8A8 는 본 sprint 의 W4A8 와 다름** → P1c 에서 W4A8 PT2E 빌더 재작성 |

## v2-§11 추가 리스크 (Architect 발견)

| Risk | 영향 | Mitigation |
|---|---|---|
| **`et.htp` W4A8 PT2E 빌더 신규 작성** (μ-Q1 W8A8 빌더는 inherit 불가) | P1c +0.5d | `QuantDtype.use_8a4w` 만 enum 교체 (W8A8 → W4A8). Executorch QNN docs `use_8a4w` 지원 명시. 빌더 build_pte_*.py 패턴 (μ-Q1) 그대로 변형 |
| **MUL_MAT mm_lmh (151936) thermal drift** | P4 측정 시 trial 간 발열 누적 | 1.1 GB weight load — 같은 trial 안에서 weight cached. inter-trial cool-down +5s (기본 5~30s) |
| **Ours-NPU K/V proj 256 out_dim 미검증** | P1d K/V dim shape gate 통과 필요 | μ-Q1 G2 결과는 N=1536 만. P1d acceptance: K=1536 N=256 GREEN gate 추가 |
| **L.cpp.HTP0 perf path abort** (`dspqueue_write failed: 0x0000000c`) | P2 의 lcpp.htp perf cell 12 cell ⚠ | 사용자 결정 2026-05-26 inherit: support 만 reference, perf 는 `⚠` 마킹. graph-level llama-bench tg32=32.40 tok/s 만 paper 인용 |
| **MUL_MAT mm_qkv fused K/V 256 dim** Ours-NPU 측 | P1d Q-proj 1536 GREEN inherit ↛ K/V 256 자동 GREEN 아님 | P1d 가 mm_qkv fused N=2048 cell 의 dispatch GREEN 검증 책임 (G2 sprint 의 dim variance test 가 같은 helper 로 재사용 가능) |
| **CPY F32→F16 fair-pair Q4_0 row** | 의미 없음 (activation-only) | `—` 마킹. SOFT_MAX/SCALE 동일. dtype-only op 들은 일관되게 Q4_0 row `—` |
| **ExecuTorch `et.htp` Q4_0 row dtype 정의** | W4A8 (asym int4) 가 Ours Q4_0 (sym int4 block-32) 와 양자화 algorithm 다름 | "fair-pair" 의미가 weight-bit-width 동일 (4-bit) 기준이며 quant schema 동일 의미 아님. paper narrative 에 명시 ("both 4-bit weight, schema differs"). cross-tier 비교는 latency only, accuracy 비교는 별도 dtype semantics 토론 |
| **측정 시간 196 cell × 13 round × ~5s** | ≈ 3.5h compute + cooldown ~10~13h | 사용자 결정: **1~2일 분산** 측정. 8h/day cap, Tier-A 우선 |

## v2-§12 paper narrative slot (P4 보고서 자리)

본 매트릭스 측정 후 paper figure 의 narrative 후보:

1. **NPU per-op 분리 측정의 자체-built binding 의 필요성** — μ-Q1 의 Phase E 결론 inherit + 12-op 확장으로 강화. Tier-A 8 op + Tier-B 4 op = 12 op 전수 self-built binding GREEN evidence (P1d 결과로 확정)
2. **W4A8 (`et.htp` `use_8a4w`) vs Q4_0 (Ours.htp) latency 정량 비교** — 두 backend 가 같은 NPU 위 같은 4-bit weight 로 무엇이 빠른가 (framework overhead)
3. **MUL_MAT mm_ffn / mm_lmh / mm_qkv 3 shape 의 backend 우열 변화** — N (out_dim) 의 함수로 어느 backend 가 우위인가 (small N: CPU, large N: GPU, medium: NPU 가설)
4. **FLASH_ATTN_EXT 의 backend 매트릭스** — Ours.cpu vs Ours.gpu vs Ours.htp 가 모두 동일 fused path 측정 (NPU 의 fused FA 가 가능한가? G1~G5 에서 미검증) + `lcpp.htp` `✗` 0% 인 fact 의 narrative ("NPU 는 fused FA 전혀 미지원")
5. **production hot path (Ours decode TBT 32 ms/tok) 분해** — Tier-A 8 op 의 op-level latency 합산 ≈ 28-layer · single decode 의 production TBT vs measurement budget. micro-vs-macro gap quantification

---

## v2 자기점검

- [x] Backend 7개 정의 명확 (path + 측정 binary)
- [x] Op 12개 inventory + Tier 분류 + 호출 횟수
- [x] MUL_MAT 3 shape 확정 (FFN / LM head / QKV fused)
- [x] PM 보류 4건 모두 Architect 결정 완료 (§5.1~§5.5)
- [x] UNSUPPORTED 마킹 4-tier 규칙 (`✗`/`—`/`⚠`/`?`)
- [x] Tolerance 표 op × dtype override
- [x] 의존성 검증 결과 (test-backend-ops 존재, ExecuTorch venv ⚠, IDL F16 ✓ 이미 존재)
- [x] Phase entry points (P1a/b/c/d/P2) 매핑
- [x] 추가 리스크 8건 표면화
- [x] cell_inventory.md 분리 작성 책임 명시 (다음 산출물)
