# P0-5c 선조사: GQA-aware KV 공유 Attention 커널

**작성**: 2026-04-13 (Researcher 서브에이전트)
**대상**: senior-implementer (P0-5c Phase B)
**관련 TODO**: `.agent/todos/long_context_attention_optimization.md` 10.1.6, P0-5c

---

## A. llama.cpp / FlashInfer GQA 패턴

### A.1 llama.cpp CUDA `fattn-vec.cuh` — GQA reuse 없음

- `/home/go/Workspace/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh:19-510`
- `ncols`는 **Q token 축**(speculation). `ggml_cuda_flash_attn_ext_vec_case` 에서 `Q->ne[1] == 1 → cols_per_block = 1`.
- GQA 매핑: `const int gqa_ratio = ne02 / ne12; K += nb12*(head / gqa_ratio)` — 각 Q head 별 별도 block, 같은 KV head를 `gqa_ratio`× 독립 로드.
- shared memory는 SG간 score 교환용. KV는 L1/L2 cache 의존.

### A.2 llama.cpp Metal `kernel_flash_attn_ext_vec` — GQA reuse 없음

- `/home/go/Workspace/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6318-6748`
- Dispatch: `tgpig[0]=q row, [1]=Q head, [2]=batch×NWG+iwg`.
- GQA: `ikv2 = iq2/(ne02/ne_12_2)` — Q head별 TG 분리, KV 중복 로드.
- shared mem 용도: Q 로드(`sq4`), softmax scratch(`ss`), partial O(`so4`). **KV tile shared 없음**.

### A.3 llama.cpp OpenCL `flash_attn_f32_f16.cl` — 우리 커널과 동형

- `/home/go/Workspace/llama.cpp/ggml/src/ggml-opencl/kernels/flash_attn_f32_f16.cl`
- 우리 `engine/kernels/flash_attn_f32_f16.cl` 와 동일한 `flash_attn_f32_f16_q1` 구조. GQA reuse 없음.

### A.4 FlashInfer — 유일하게 GQA reuse 구현

- `include/flashinfer/attention/decode.cuh` (FlashInfer)
- 핵심 패턴:
  - **prefill multi-query kernel을 decode에 재활용** — `gqa_ratio` 개 Q head 를 1개 multi-query 요청으로 처리.
  - Thread block 매핑: `qo_head_idx = kv_head_idx * bdy + threadIdx.y`, `bdy = GROUP_SIZE = gqa_ratio`.
  - Shared memory:
    - `K tiles: num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim * sizeof(DTypeKV)`
    - `V tiles: 동일`
    - `smem_md: 2 * bdy * bdz * sizeof(float)` (per-Q-head softmax state)
  - `cp_async + num_stages_smem` 으로 K/V tile prefetch
- Operational intensity O(gqa_ratio) byte reuse → compute-bound 전환.
- Paper: arXiv:2501.01005

### A.5 핵심 결론

- **llama.cpp는 decode GQA reuse 미구현**. NVIDIA/Apple은 L1 cache로 흡수.
- Adreno는 buffer L1 없음(image 전용 TP). L2만 거치므로 redundant load 비용이 크게 누적.
- **모바일 GPU에서 GQA reuse는 우리가 먼저 해야 할 최적화**. FlashInfer 패턴 OpenCL 이식.

---

## B. Adreno 830 / Snapdragon 8 Elite 제약

출처: Qualcomm OpenCL Programming Guide (80-NB295-11 Rev. C)

| 속성 | 값 / 권장 |
|---|---|
| Wave size | 8/16/32/**64/128** 컴파일러 선택. 런타임 `clGetKernelSubGroupInfo` 확인 필수. Adreno 830 premium은 128 추정 (불확실). |
| Local memory | Adreno X1-85 = 32 KB/kernel 확인. Adreno 830 동급 추정. 런타임 `CL_DEVICE_LOCAL_MEM_SIZE` 확인 필수. |
| L2 cache line | 64 bytes |
| Local mem 사용 규칙 | 자주 쓰일 때만, 1-2회면 사용 자제. subgroup shuffle 우선. 과다 시 동시 WG 수 감소 → latency hiding 악화. |
| Barrier 비용 | ALU stall 유발, local memory 이점 상쇄 가능. **barrier 수 최소화**. |
| Image vs buffer | image가 일반적으로 빠름(L1 + 자동 boundary + TP HW). 그러나 KV는 매 토큰 write 발생이라 buffer 권장(P0-5 결론 유지). |
| Vectorized load | 128-bit (vload4 float / vload8 half) 권장. |

### B.1 Local memory 버짓

**Qwen head_dim=128, KV_TILE=32**:
- K tile (F16): 32 × 32 × 8B = 8 KB
- V tile (F16): 32 × 32 × 8B = 8 KB
- Q local: gqa_ratio=6 × 128 × 2B = 1.5 KB
- 총 ~18 KB (32 KB 한도 내)

**Llama head_dim=64, KV_TILE=64**:
- K tile: 64 × 16 × 8B = 8 KB
- V tile: 64 × 16 × 8B = 8 KB
- Q local: gqa_ratio=4 × 64 × 2B = 512 B
- 총 ~17 KB

### B.2 Bank conflict
- Adreno bank 구조 공식 미공개. `half2`/`half4` access 로 회피.

### B.3 Image2d
- KV는 write 빈번 → image 비효율. 1차 buffer 유지, 후속 측정 후 평가.

---

## C. 현재 커널 분석

### C.1 구조
- `engine/kernels/flash_attn_f32_f16.cl::flash_attn_f32_f16_q1` (line 214-377)
- Dispatch: `engine/src/backend/opencl/mod.rs:2038` — `global = [Q1_WG_SIZE=64, n_heads_q, 1]`
- `head_idx = get_global_id(1)`, `head_kv_idx = head_idx / gqa_ratio` — GQA 인지하나 WG 분리.

### C.2 KV 중복 분석
Pass 1 (max scan) + Pass 2 (softmax + V) 모두 K 로드 → **K 2× 중복** 추가 존재.
- Qwen (gqa=6): 6 Q-heads × 2 passes = K 12회, V 6회 (per k_idx)
- Llama (gqa=4): 8 + 4 = 12회

### C.3 head_dim 64/128
- `kernel_flash_attn_f32_f16_q1_dk{64,128}` 두 바이너리, 매크로 `DK`/`DV` 분기.

### C.4 현재 local 사용량
- `local_m[64]`, `local_l[64]`, `local_o_comp[64]` (float4) ≈ **1.5 KB** (32 KB 중 ~5%). 타일링 여유 매우 큼.

---

## D. 설계 선택지 + 권장안

### Option 1: WG = 1 KV-head, gqa_ratio Q-heads
```
global = [WG_SIZE, n_heads_kv, batch]
local  = [WG_SIZE, 1, 1]
```
- KV 트래픽 gqa_ratio× 감소 (Qwen 6×, Llama 4×)
- 단점: WG 수 감소 (Llama 32→8, Qwen 12→2) — SP utilization 우려

### Option 2: gqa 를 sub-group에 분배
- Qwen에서 2 sub-group (각 3 Q-head). WG 수 증가하지만 KV 부분 reuse만.
- Llama에 과도. **Option 1 채택, Qwen 부족시 Option 2 fallback**.

### Option 3: Tile prefetch + double buffer
- OpenCL `cp_async` 없음. 1차 생략, 측정 후 추가.

### D.1 권장: Option 1 + KV tile streaming + single stage
근거: Local mem 32 KB 여유, double buffer 복잡도 대비 효과 불확실.

### D.2 결합 (권장 최종)
```
global = [WG_SIZE, n_heads_kv, n_kv_splits]
```
- `n_kv_splits = ceil(n_kv / 512)` (Metal NWG 패턴)
- Qwen 2×8=16 WG (ctx 4096), Llama 8×8=64 WG. SP 8+ 매치.
- ⚠️ **KV-split 결합 필요** — P0-5/P0-5b revert 했지만 Qwen 병렬성 위해 부활 가능성.

**대안 권장**: Llama 우선 구현 + 측정. Qwen은 GQA 단독으로도 6× bandwidth 절감이라 SP 부족에도 net positive 가능. 둘 다 측정 후 KV-split 결합 결정.

---

## E. Local memory 타일링 전략

### E.1 Qwen head_dim=128, KV_TILE=32 (권장)
- single stage K_tile + V_tile = 16 KB
- Q local 1.5 KB
- 총 ~18 KB

### E.2 Llama head_dim=64, KV_TILE=64 (권장)
- K_tile + V_tile = 16 KB
- Q local 512 B
- 총 ~17 KB. double buffer 가능 (KV_TILE=32로 낮춤).

### E.3 Barrier 배치
- Tile 당 barrier 최소화: K+V 같은 iteration에서 cooperative load 후 단일 barrier로 공개.
- Final reduction (m, l, o) 시 log₂(WG_SIZE) 단계.

### E.4 Q 배치
- **Option A (권장)**: Q 전체를 kernel 시작 시 local에 1회 로드.
- Option B: 각 thread reg 분배. 복잡도 ↑.

---

## F. 구현 의사코드 (Llama head_dim=64 기준)

```c
#define WG_SIZE 64
#define KV_TILE 32
#define DK 64
#define DV 64
#define DK4 16
#define DV4 16
// GQA_RATIO, N_HEADS_KV — kernel build option

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void flash_attn_f32_f16_q1_gqa(
    const global void* q_void, ulong q_offset,
    const global void* k_void, ulong k_offset,
    const global void* v_void, ulong v_offset,
    global void* o_void, ulong o_offset,
    const float scale, const int n_kv,
    const int n_head, const int n_head_kv, const int gqa_ratio,
    /* strides */ ...
) {
    const int tid     = get_local_id(0);
    const int kv_head = get_global_id(1);
    const int batch   = get_global_id(2);

    // Q local cache
    __local float4 q_local[GQA_RATIO_MAX][DK4];
    if (tid < gqa_ratio * DK4) {
        const int g  = tid / DK4;
        const int d4 = tid % DK4;
        const int q_head = kv_head * gqa_ratio + g;
        q_local[g][d4] = q_ptr[d4];  // load from global
    }

    // Per-thread accumulators (registers)
    float m_i[GQA_RATIO_MAX], l_i[GQA_RATIO_MAX];
    float4 o_acc[GQA_RATIO_MAX][DV4];
    for (int g = 0; g < gqa_ratio; ++g) {
        m_i[g] = -INFINITY; l_i[g] = 0.0f;
        for (int i = 0; i < DV4; ++i) o_acc[g][i] = (float4)(0.0f);
    }

    // Shared KV tiles
    __local half4 k_tile[KV_TILE][DK4];
    __local half4 v_tile[KV_TILE][DV4];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Stream over KV
    for (int tile_base = 0; tile_base < n_kv; tile_base += KV_TILE) {
        const int tile_n = min(KV_TILE, n_kv - tile_base);

        // Cooperative K/V load (KV_TILE * DK4 = 512 elements / 64 threads = 8 each)
        for (int i = tid; i < KV_TILE * DK4; i += WG_SIZE) {
            const int t = i / DK4, d = i % DK4;
            if (t < tile_n) k_tile[t][d] = K_global[t][d];
            else k_tile[t][d] = (half4)(0.0h);
        }
        for (int i = tid; i < KV_TILE * DV4; i += WG_SIZE) {
            const int t = i / DV4, d = i % DV4;
            if (t < tile_n) v_tile[t][d] = V_global[t][d];
            else v_tile[t][d] = (half4)(0.0h);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute Q·K + online softmax for each Q-head
        for (int g = 0; g < gqa_ratio; ++g) {
            float s_local[KV_TILE / WG_SIZE + 1];
            int s_count = 0;
            float m_new = m_i[g];

            for (int t = tid; t < tile_n; t += WG_SIZE) {
                float4 qk = (float4)(0.0f);
                for (int d = 0; d < DK4; ++d) {
                    qk = mad(q_local[g][d], convert_float4(k_tile[t][d]), qk);
                }
                float s = (qk.s0 + qk.s1 + qk.s2 + qk.s3) * scale;
                s_local[s_count++] = s;
                m_new = max(m_new, s);
            }
            m_new = wg_reduce_max(m_new);

            const float m_scale = exp(m_i[g] - m_new);
            l_i[g] *= m_scale;
            for (int i = 0; i < DV4; ++i) o_acc[g][i] *= m_scale;

            s_count = 0;
            for (int t = tid; t < tile_n; t += WG_SIZE) {
                float p = exp(s_local[s_count++] - m_new);
                l_i[g] += p;
                for (int i = 0; i < DV4; ++i) {
                    o_acc[g][i] = mad((float4)(p), convert_float4(v_tile[t][i]), o_acc[g][i]);
                }
            }
            m_i[g] = m_new;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Final reduction + write
    for (int g = 0; g < gqa_ratio; ++g) {
        float l_final = wg_reduce_sum(l_i[g]);
        for (int i = 0; i < DV4; ++i) {
            float4 o_sum = wg_reduce_sum_float4(o_acc[g][i]);
            if (tid == 0) {
                const int q_head = kv_head * gqa_ratio + g;
                O_global[q_head][i] = o_sum / l_final;
            }
        }
    }
}
```

### F.1 세부 주의

1. **`wg_reduce_*` helpers**: 기존 패턴 재활용. gqa_ratio 동시 reduction이라 scratch `[gqa_ratio][WG_SIZE]` 또는 loop-wise.
2. **`s_local[]` 크기**: `ceil(KV_TILE / WG_SIZE)`. KV_TILE=32, WG=64 → 1.
3. **Online softmax**: 1-pass 전환. Tile마다 rescale. **F32 accumulator 필수** (P0-5 결론).
4. **Tile 경계**: `tile_n < KV_TILE` 시 0-padding + score=-INF.
5. **Register 압력**: Qwen `o_acc[6][32]` = 768 B private/thread. Adreno GPR 여유 확인 필수 — 빌드 후 `CL_KERNEL_PRIVATE_MEM_SIZE` 체크, spill 시 KV_TILE 축소.
6. **MHA fallback (gqa_ratio=1)**: 호스트에서 기존 q1로 분기.

---

## G. 정확도 / 테스트 전략

### G.1 검증
- `test_backend`: GQA kernel vs CPU NEON ref (`attention_gen_f16_neon`) cos-sim ≥ 0.999
- vs 기존 `flash_attn_f32_f16_q1` cos-sim ≥ 0.9999

### G.2 매트릭스
| Model | n_q | n_kv | gqa | head_dim | 우선 |
|---|---|---|---|---|---|
| Llama 3.2 1B | 32 | 8 | 4 | 64 | P0 |
| Qwen 2.5 1.5B | 12 | 2 | 6 | 128 | P0 |
| MHA edge | 32 | 32 | 1 | 64 | P1 (fallback) |

### G.3 KV sweep
{128, 512, 1024, 2048, 4096}

### G.4 병렬성
- Qwen 2 KV × 1 batch = WG 2 → Adreno 830 (8+ SP) 부족. KV-split 결합 검토.
- Llama 8 KV × 1 batch = WG 8 → 적정.

### G.5 회귀 방지
- Short ctx에서 tile overhead로 느려질 수 있음. host에서 `n_kv < 128` 등 threshold 이하는 기존 q1로 dispatch 유지.

---

## 권장 구현 방향 (1문단 요약)

FlashInfer GQA decode 패턴 + Metal flash-decoding NWG split을 OpenCL로 이식한다. 신규 커널 `flash_attn_f32_f16_q1_gqa` 추가, dispatch는 `global = [WG_SIZE, n_heads_kv, batch]` (1차) 또는 `[WG_SIZE, n_heads_kv, n_kv_splits]` (2차, Qwen 병렬성 확보 시). WG 하나가 `gqa_ratio` 개 Q-head 담당, K/V 타일 (Qwen head_dim=128 → KV_TILE=32, Llama head_dim=64 → KV_TILE=64) 을 local memory에 cooperative load → gqa_ratio× 재사용. Online softmax는 F32 accumulator 유지(P0-5 정확도 교훈), single-stage tiling으로 시작. Adreno 32 KB local mem 여유 (~18 KB Qwen, ~17 KB Llama). Qwen `o_acc[6][32]` private 압력 빌드 후 spill 확인 필수. Host dispatch는 `gqa_ratio == 1` MHA 또는 `n_kv < 128` short ctx면 기존 q1로 fallback. 검증은 `test_backend` CPU NEON ref vs cos-sim ≥ 0.999, 모델 3종 × KV 길이 5종 매트릭스. 예상 효과: Llama 4×, Qwen 6× VRAM KV read 감소 → 4K decode throughput 개선 (절대값은 L2 hit ratio 의존, 실측 필요).

---

## 출처

**우리 리포**:
- `engine/kernels/flash_attn_f32_f16.cl:214-377`
- `engine/src/backend/opencl/mod.rs:1897-2060, 2038`

**llama.cpp**:
- `ggml-cuda/fattn-vec.cuh:19-510`
- `ggml-metal/ggml-metal.metal:6318-6748, 5361-5469`
- `ggml-opencl/kernels/flash_attn_f32_f16.cl`

**외부**:
- FlashInfer blog: https://flashinfer.ai/2024/02/02/introduce-flashinfer.html
- FlashInfer paper: https://arxiv.org/pdf/2501.01005
- FlashInfer decode.cuh: https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/decode.cuh
- NSA paper: https://arxiv.org/html/2508.18224v1
- Qualcomm OpenCL Guide (80-NB295-11): https://docs.qualcomm.com/bundle/publicresource/80-NB295-11_REV_C_Qualcomm_Snapdragon_Mobile_Platform_Opencl_General_Programming_and_Optimization.pdf
- GTA/GQA paper: https://arxiv.org/html/2505.21487v1

---

## 불확실 / 추가 검증 필요

1. Adreno 830 실제 `CL_DEVICE_LOCAL_MEM_SIZE` (32 KB 가정).
2. Adreno 830 wave size — `CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE` 디바이스 쿼리.
3. Local memory bank 수/크기 (공식 미기재) — 128-bit access로 회피.
4. Register spill: Qwen `o_acc[6][32]` 빌드 후 spill 확인.
5. Qwen 2 KV-head 병렬성 부족 — KV-split 결합 필요할 수 있음 (P0-5 부활 가능).
6. FlashInfer decode.cuh 직접 코드 확인 권장 (블로그/논문 요약만 검토).
