# Handoff: Ours-NPU op 지원 sprint 시작 (MUL_MAT 단독 우선)

**작성**: 2026-05-26
**HEAD**: `53b1ef6a docs(microbench): Qwen2.5-1.5B 25-op × 4-backend Support + Perf 1차 매트릭스`
**worktree**: `.claude/worktrees/b5_trait_extension`
**다음 세션 진입 문장**: **"Ours-NPU MUL_MAT sprint 진행"**

---

## TL;DR

μMatrix 1차 매트릭스(`53b1ef6a`) 완성 — Ours-NPU 컬럼은 RMS_NORM 1/25 cell만 채워짐. 나머지 24 op은 `init_*_req` helper + `op_params` packing 미구현 = 본 sprint 시리즈의 작업 범위.

사용자 결정 (2026-05-26):
- **순서**: G2 **MUL_MAT 단독 우선** (paper main = Q-proj decode latency)
- **단위**: 2~3 op per sprint (그룹별, G1 unary 그룹 등은 후속)
- **매트릭스 갱신**: sprint 끝마다 `matrix.md` 해당 cell 갱신 + commit

NPU compute path는 Q-2.2 옵션 D fix(`b1d374ac`)로 GREEN — 새로운 op = IDL packing + microbench wrap만 작성하면 dispatch 가능.

---

## 진행 상태

| 항목 | 상태 | reference |
|---|---|---|
| μMatrix 1차 commit + push | ✓ | `53b1ef6a` |
| NPU compute path GREEN | ✓ | `b1d374ac` (옵션 D DspQueueBuffer layout fix) |
| Ours-NPU op coverage | 1/25 (RMS_NORM) | `engine/src/backend/htp_fastrpc/idl.rs:247` |
| llama.cpp HTP per-op 측정 | ✗ (사용자 결정) | codebase support flag만 reference |

## 다음 sprint plan (MUL_MAT 단독)

### 검증 게이트

- `microbench_htp_matmul` (신규 또는 기존 `htp_matmul_correctness` 확장) dispatch rc=0
- correctness: max_abs_err < 1e-3 vs CPU NEON ref (Q4_0 dequant→F32 matmul)
- S25 latency 측정 → matrix.md cell 갱신

### Shape 우선순위 (Qwen 2.5-1.5B 6 variant)

| # | shape | 사용처 | paper 가치 |
|---|---|---|---|
| 1 | `W[1536,1536] Q4_0 @ x[1536]` | Q-proj / O-proj | **main** (decode hot) |
| 2 | `W[256,1536] Q4_0 @ x[1536]` | K-proj / V-proj | hot |
| 3 | `W[8960,1536] Q4_0 @ x[1536]` | FFN gate / up | hot |
| 4 | `W[1536,8960] Q4_0 @ x[8960]` | FFN down | hot |
| 5 | `W[151936,1536] Q4_0 @ x[1536]` | lm_head | 1 회/token |
| 6 | `W[1536,1536] Q4_0 @ X[1536,n]` n=512 | prefill GEMM | paper supplementary |

**권장**: shape 1 (Q-proj GEMV)만 1차 commit → shape 2-4 합쳐서 동 sprint 안에서 후속 → shape 5/6은 별 sprint.

### 작업 항목

1. `engine/src/backend/htp_fastrpc/idl.rs::init_matmul_req` helper 구현 (RMS_NORM 패턴 참조)
   - op_params 구조: dtype (Q4_0), K, N, transpose flag 등
   - llama.cpp `htp/main.c::mul_mat_*` 시그니처 binary 비교
2. `engine/microbench/htp_matmul.rs` 또는 기존 확장 — dispatch + correctness + timing
3. S25 deploy → measure (warmup 3 + iter 10 median)
4. `matrix.md` 의 `MUL_MAT` row 의 Ours-NPU cell 갱신
5. commit + push + notify

### Landmines / 미해결

- **Q4_0 dequant on HTP HVX**: llama.cpp ggml-hexagon의 MUL_MAT op이 native Q4_0을 지원하는지 unclear. `htp/main.c` 확인 필요. native 미지원이면 host-side dequant → F16 dispatch 경로 (CPU pre-dequant 비용 발생, paper narrative 영향).
- **dtype 매트릭스 폭**: Qwen2.5-1.5B 는 Q4_0만 쓰지만 paper에서 Q8_0/F16 비교를 원하면 dtype × shape sweep. 본 sprint 는 Q4_0만 하고 dtype sweep은 별 sprint 권장.
- **paper main metric ≠ single op latency**: production decode TBT는 layer-level 통합값. MUL_MAT cell은 단일 op 측정 → paper figure 1 은 cell 단위, paper Table은 layer-level. 두 표 분리 유지.
- **shape 5 (lm_head 151936)**: 매우 큼, HTP HMX register tile 제약으로 dispatch fail 가능. shape 4까지 GREEN 확인 후 진입.
- **CPU NEON ref garbage 리스크**: Qwen2.5-1.5B의 CPU baseline은 garbage 출력 issue 있음 ([[project_cpu_baseline_qwen25_5b]]). 단일 op correctness는 dim 작아 영향 없지만, 큰 dim에서 ref 신뢰성 확인.

## 후속 sprint candidate (G1~G5)

| sprint | 그룹 | op | 추정 |
|---|---|---|---|
| 본 sprint | G2 | MUL_MAT (shape 1~4) | 1~2일 |
| 다음 | G1 | ADD + MUL + SILU + SCALE (unary helper factory) | 1일 |
| +1 | G3 | ROPE Q + K (head_dim=128, θ=1e6) | 1일 |
| +2 | G4a | SOFTMAX + 분리 attn path (FLASH_ATTN_EXT 회피) | 1~2일 |
| +3 | G5 | GET_ROWS + SET_ROWS + CPY | 1일 |
| +4 | G4b | FLASH_ATTN_EXT fused (선택) | 2~3일 |

paper main figure 완성 = G1+G2+G3+G4까지 완료 시 Ours-NPU 컬럼 ~12/25 cell.

## 핵심 파일 인덱스

- 본 sprint 커밋: `53b1ef6a` (매트릭스 1차)
- NPU compute GREEN 커밋: `b1d374ac` (Q-2.2 옵션 D)
- IDL: `engine/src/backend/htp_fastrpc/idl.rs` (현재 RMS_NORM helper 1건)
- host binding: `engine/src/backend/htp_fastrpc/host.rs`
- microbench: `engine/microbench/htp_rmsnorm.rs` (확장 또는 신규 htp_matmul.rs)
- llama.cpp ref: `/home/go/Workspace/llama.cpp/ggml/src/ggml-hexagon/htp/main.c::mul_mat_*`
- 매트릭스: `papers/eurosys2027/_workspace/experiment/microbench_op_matrix_2026_05_26/matrix.md`
