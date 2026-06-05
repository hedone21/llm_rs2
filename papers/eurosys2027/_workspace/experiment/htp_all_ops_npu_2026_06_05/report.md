# HTP all-ops-NPU — decode 전 연산 op 를 Hexagon DSP 로 (2026-06-05)

**디바이스**: Galaxy S25 (R3CY408S5SB) · **모델**: Qwen2.5-1.5B-Instruct q4_0 · **백엔드**: `--backend htp` · **6T** · **env**: `ADSP_LIBRARY_PATH=/data/local/tmp`

**목표**: 기존 CPU 위임이던 decode 연산 op(silu_mul, rope, rms_norm, add, attention)를 전부 NPU 로 dispatch. 느려져도 무방(전 op NPU 배치 측정/연구용). 각 op `LLMRS_HTP_NPU_<OP>` env 게이트(기본 OFF=CPU, 회귀 0).

---

## 1. op 별 NPU 정확성 (env OFF baseline 대비)

| op | HTP_OP | NPU 정확도 | token-id (isolation) |
|---|---|---|---|
| matmul (q/k/v/o/gate/up/down + lm_head) | MUL_MAT | 기존(검증됨) | — |
| **silu_mul** | UNARY_SILU + MUL | 일치 | 16/16 |
| **rope** | ROPE (NeoX mode=2) | max_err **1e-5** (F32 rounding) | (flip, 아래 ★) |
| **rms_norm** | RMS_NORM + MUL(gamma) | 일치 | 16/16 |
| **add** (residual/bias) | ADD | 일치 | 16/16 |
| **attention** | FLASH_ATTN_EXT | max_err **4e-4** (F16) | 16/16 |
| **ALL 동시** | — | — | **16/16** (결정론적 2회) |

- **kv_scatter 제외**: 순수 memcpy(compute 0). host→rpcmem write 가 이미 DSP 가 읽는 형태라 NPU 化 무의미.
- **★ ROPE isolation flip**: ROPE 단독 enable 시 "...very big city..."(baseline "...a population...")로 갈림. 단 **rope-diag max_err=1e-5 (bit-exact 수준)** — 이는 버그가 아니라 greedy decode 가 1e-5 수치차를 28층×16토큰 누적해 borderline 토큰을 뒤집는 **본질적 민감성**. ALL 경로는 16/16(다른 op 들과 합쳐진 numeric 이 baseline trajectory 유지).

## 2. decode TBT (all-NPU vs baseline)

| 구성 | decode ms/tok |
|---|---|
| baseline (연산 op 전부 CPU, matmul 만 NPU) | ~66 |
| **all-NPU** (silu+rope+rms_norm+add+attn NPU) | ~85 |

→ **~1.3× 느림** (의도된 trade-off). pointwise/norm 은 CPU NEON 이 NPU 보다 빠르고(op microbench: 6~2600×), attention 도 per-kv-head 분리로 dispatch 수↑. floor 누적 + DSP throughput 열위. **목표는 속도가 아니라 "전 op NPU 배치 가능 + 정확"**.

## 3. 발견한 v79 skel 한계 2건 (device 실측, 우회 완료)

1. **element-wise op dispatch 상한 ~12032 f32**: unary/binary op 이 dispatch 당 1 VTCM 타일(~12032 f32)만 처리하고 전체 텐서를 loop 안 함. prefill silu(numel=44800) 시 idx 12032 부터 입력 passthrough(미처리)→garbage. **우회**: flat helper 를 `HTP_OP_CHUNK=8192` 분할 dispatch. (microbench DIM=8960<상한 이라 가려졌던 버그.)
2. **flash_attn GQA 미지원 (항상 kv_h=0)**: nhkv>1 시 v79 가 모든 q-head 에 kv_h=0 만 사용 → kv_h>0 group(q-head 6~11) garbage(max_err 2.98, first_bad=768). **우회**: kv-head 별 분리 dispatch(그룹당 gqa q-head + 단일 kv-head, HeadMajor single head 는 contiguous). 수정 후 max_err 4e-4.

## 4. 재현

```bash
# 각 op enable: LLMRS_HTP_NPU_{SILU,ROPE,RMSNORM,ADD,ATTN}=1
bash verify.sh           # baseline + 개별 isolation + ALL + decode TBT (q4)
OPS="..." bash verify.sh # 일부 op 만
```

산출물: `verify.sh`, `raw/` (device 로그). 코드: `engine/src/backend/htp_fastrpc.rs` (commit c5f46e60 pointwise + 065d2797 attention).
