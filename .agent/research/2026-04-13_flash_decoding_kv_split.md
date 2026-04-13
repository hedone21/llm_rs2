# P0-5 선조사: GPU Decode KV-Split (Flash Decoding) 설계 자료

**작성**: 2026-04-13 (Researcher 서브에이전트)
**대상**: senior-implementer (P0-5 Phase B 구현자)
**관련 TODO**: `.agent/todos/long_context_attention_optimization.md` 10.1.6, P0-5

---

## A. Tri Dao 2023 Flash Decoding 알고리즘 정확한 수식

### A.1 Online softmax (single pass, query 1개 기준)

- 초기: m = -∞, ℓ = 0, o = 0
- 각 KV 원소 i에 대해 score s_i = <q, k_i>/√d 계산 후:
  - m' = max(m, s_i), p_i = exp(s_i - m'), α = exp(m - m')
  - o ← α·o + p_i·v_i, ℓ ← α·ℓ + p_i, m ← m'
- 최종: out = o / ℓ

### A.2 Split 간 merge (Flash Decoding 핵심)

KV 축을 S개 split으로 쪼개 각 split s가 [a_s, b_s) 범위를 독립적으로 처리해 부분 통계 (m_s, ℓ_s, o_s)를 만든다. **o_s는 이미 exp(·-m_s)로 정규화된 weighted sum, NOT yet divided by ℓ_s**.

Merge (log-sum-exp stable):
1. m* = max_s m_s
2. α_s = exp(m_s - m*)
3. ℓ* = Σ_s α_s · ℓ_s
4. o* = Σ_s α_s · o_s
5. out = o* / ℓ*

검증 참조: `llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh:730-779` `flash_attn_combine_results`, Metal `ggml-metal.metal:6865-6905` `kernel_flash_attn_ext_vec_reduce`.

### A.3 수치 안정성

- **Accumulator dtype 분리 필수**: partial o_s는 반드시 **F32**. Partial은 ℓ_s로 아직 나누지 않았으므로 값 범위가 O(C·e^{s_max-m_s})까지 커질 수 있음 (C=128, 점수 편차 10 → ~10^5). F16 (max 65504)에서는 overflow 위험.
- **Meta (m_s, ℓ_s)는 F32 pair** (llama.cpp `float2`). F16은 ℓ 정밀도 손실.
- 최종 dst는 F32 (기존 유지).
- Underflow: split 범위가 모두 mask로 -∞면 m_s=-∞, ℓ_s=0. Merge 시 exp(-∞ - m*)=0 자동 safe. 모든 split이 이 상태면 ℓ*=0 → fallback `dst=0`.
- `exp(m_s - m*)` ≤ 1 (overflow 없음).

---

## B. llama.cpp Flash Decoding 구현 패턴

### B.1 경로별 현황

| 백엔드 | 파일 | KV-split 여부 | 비고 |
|--------|------|----------------|------|
| **OpenCL** | `ggml-opencl/kernels/flash_attn_f32_f16.cl` `flash_attn_f32_f16_q1` | **없음** (우리 커널과 동일) | Adreno 전용 경로 없음. `Q1_WG_SIZE=64` single-WG decode |
| **CUDA (vec)** | `ggml-cuda/fattn-common.cuh` `launch_fattn()` + `fattn-vec.cuh` | **있음** (`parallel_blocks` axis) | occupancy 기반 자동 선택 |
| **Metal (vec)** | `ggml-metal.metal::kernel_flash_attn_ext_vec` + `kernel_flash_attn_ext_vec_reduce` | **있음** (`NWG` function constant) | 별도 reduce kernel, `nwg=32` 고정 |

**핵심 발견**: **llama.cpp의 OpenCL 포트는 Flash Decoding 미구현**. CUDA/Metal만 있음. 우리가 OpenCL에 구현하면 llama.cpp 대비 독자적 최적화.

### B.2 Split 수 결정

- **CUDA** (`fattn-common.cuh:916-966`): `min(max_blocks_per_sm_from_occupancy, ntiles_KQ)` + efficiency loop.
- **Metal** (`ggml-metal-ops.cpp:2919-2935`): `nwg = 32` 고정, `nsg *= 2 while 2*nwg*nsg*ncpsg < ne11 && nsg < 4`.
- **vLLM PagedAttention V2** (`csrc/attention/attention_kernels.cu`): `PARTITION_SIZE = 512` 고정, `max_num_partitions = ceil(context_len / 512)`.

### B.3 Merge 방식

- CUDA/Metal 모두 **별도 kernel** (`flash_attn_combine_results` / `kernel_flash_attn_ext_vec_reduce`).
- 단일 커널 내 atomics 방식은 둘 다 사용 안 함 — cross-WG barrier 없고 atomic_add float으로 online-softmax rescaling 불가능.

---

## C. 현재 엔진 커널 분석

### C.1 파일 목록

```
engine/kernels/flash_attn_f16.cl       — K/V=F16, Q=F16 (프리필 + Q1 decode)
engine/kernels/flash_attn_f32.cl       — 모두 F32
engine/kernels/flash_attn_f32_f16.cl   — K/V=F16, Q=F32, O=F32 (현재 주 경로)
```

각 파일은 2개 kernel: (1) n_q>1 prefill, (2) `*_q1` decode (n_q==1).

### C.2 `flash_attn_f32_f16_q1` 구조

- 시그니처: `engine/kernels/flash_attn_f32_f16.cl:214-243`.
- `global_work_size = [Q1_WG_SIZE=64, n_head, 1]`, `local_work_size = [64, 1, 1]`.
- Pass 1 (KV loop): `m_i = max(score)` 부분 → local reduction → `m_final`.
- Pass 2 (KV loop): `p = exp(score - m_final)`, `l_i += p`, `o_acc[i] = mad(p, v[i], o_acc[i])`.
- Head_dim 축 reduction → `tid==0`이 `o_row = o_acc / l_final` 쓰기.
- `ACC_TYPE = float` (F32 accumulator).

### C.3 GQA 중복 로드

- `head_kv_idx = head_idx / gqa_ratio`로 K,V index 계산. WG 간 공유 없음.
- Qwen2.5 1.5B gqa_ratio=6 → 6배 redundant VRAM read (10.1.6 분석 근거).

### C.4 Split 관련 흔적

- 기존 커널에 split/partial 관련 매크로·주석 **없음**. 깨끗한 상태에서 새로 도입.
- `BLOCK_N, BLOCK_M` (prefill용) 매크로는 있으나 q1은 `Q1_WG_SIZE`만 사용.

### C.5 Decode batch=1 가정

- `flash_attn_f32_f16_q1`은 query row 1 고정.
- `get_global_id(1)` = head. KV axis는 loop만 있고 dispatch 축에 없음.
- **kv_splits 축을 `get_global_id(2)`로 추가가 자연스러움**.

---

## D. Adreno 830 / Snapdragon 8 Elite 특성

- **SIMD width**: 128 (Qualcomm 공식). `CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` 확인. Q1_WG_SIZE=64 half-wave지만 기존 local reduction 트리가 64 기준이라 유지가 안전.
- **Local memory**: Adreno 830 약 32 KiB 보고 (런타임 쿼리 필요). 현재 q1 사용량 ≈ 1.5 KiB (매우 여유).
- **Texture cache (image2d_t)**: Adreno L1T는 gather 유리하나 flash attention은 이미 coalesced. **이득 불확실, 첫 버전은 buffer 유지 권장**.
- **CU 수**: Adreno 830 공식 미공개. 런타임 `CL_DEVICE_MAX_COMPUTE_UNITS` 확인. occupancy 자동 쿼리 없음 → heuristic 대체.

---

## E. 설계 선택지 — 권장: **옵션 3 (옵션 1 + threshold)**

### 비교

| 항목 | 옵션 1: split + merge kernel | 옵션 2: atomics 단일 kernel | 옵션 3: 긴 ctx 전용 별도 kernel |
|------|-------------------------------|-------------------------------|--------------------------------|
| 구현 복잡도 | 중 | 높음 (atomic float 필요) | 중 |
| 정확도 | 완전 안전 | **비결정/부정확** | 완전 안전 |
| Short ctx 회귀 | 낮음 | 낮음 | **없음** |
| Adreno 실행 | 확실 | OpenCL 2.0 atomics 제한 | 확실 |

**권장 근거**:
1. Atomic float rescaling 불가 (o' = α·o + p·v 순차 업데이트 필요).
2. Short ctx에서 split overhead (partial write + 2nd launch) 가 net negative.
3. CUDA `launch_fattn`과 정확히 일치하는 구조 — 조건부 reduce 호출.
4. VRAM 부담 최악 256 KiB 수용 가능.

### 스케치

```rust
// engine/src/backend/opencl/mod.rs
pub fn flash_attention_decode_gpu(...) {
    let n_kv = cache_seq_len;
    let kv_splits = compute_kv_splits(n_kv);
    if kv_splits == 1 {
        return self.launch_q1_legacy(...);
    }
    let partials = self.tmp_buffer(n_head * DV * kv_splits * 4);
    let meta     = self.tmp_buffer(n_head * kv_splits * 8);
    self.launch_q1_split(
        global = [Q1_WG_SIZE, n_head, kv_splits],
        local  = [Q1_WG_SIZE, 1, 1],
        args: [..., kv_splits, partials, meta],
    );
    self.launch_q1_reduce(
        global = [DV, n_head, 1],
        local  = [DV, 1, 1],
        args: [partials, meta, kv_splits, n_head, o_out],
    );
}
```

### 커널 스케치

```c
__kernel void flash_attn_f32_f16_q1_split(
    ... // 기존 q1 인자 동일
    const int kv_splits,
    __global float* partials_void,  // [n_head, kv_splits, DV]
    __global float2* meta_void      // [n_head, kv_splits]
) {
    const int split_idx = get_global_id(2);
    const int kv_per_split = (n_kv + kv_splits - 1) / kv_splits;
    const int kv_start = split_idx * kv_per_split;
    const int kv_end = min(kv_start + kv_per_split, n_kv);
    if (kv_start >= kv_end) { /* write m=-INF, l=0, o=0 */ return; }

    // ... 기존 pass1/pass2 loop을 [kv_start, kv_end) 범위로 제한 ...
    // 단, o_acc / l_final 나누지 않음!
    //       sinks 처리 skip (merge에서 한 번만 적용)
    //       mask/slope/softcap 그대로

    if (tid == 0) {
        meta[head_idx * kv_splits + split_idx] = (float2)(m_final, l_final);
    }
    // partials 쓰기 (NOT divided)
    for (int i = 0; i < DV_VEC; ++i) {
        partials[head_idx * kv_splits * DV + split_idx * DV + ...] = o_acc_lane;
    }
}

__kernel void flash_attn_q1_reduce(
    __global const float* partials,  // [n_head, kv_splits, DV]
    __global const float2* meta,     // [n_head, kv_splits]
    __global const float* sinks,     // optional
    __global float* dst,             // [n_head, DV]
    const int kv_splits,
    const int DV
) {
    const int head = get_group_id(1);
    const int tid = get_local_id(0);  // 0 <= tid < DV

    // 1) 전역 m* (local reduce over kv_splits)
    __local float l_m[MAX_SPLITS];
    float my_m = (tid < kv_splits) ? meta[head*kv_splits + tid].x : -INFINITY;
    // reduce max → m_star

    // 2) ℓ* = Σ α_s * ℓ_s, o_tid = Σ α_s * partials[s, tid]
    float l_star = 0.0f, o_tid = 0.0f;
    for (int s = 0; s < kv_splits; ++s) {
        float2 ml = meta[head*kv_splits + s];
        float alpha = exp(ml.x - m_star);
        l_star += alpha * ml.y;
        o_tid  += alpha * partials[head*kv_splits*DV + s*DV + tid];
    }

    // 3) sinks (optional)
    if (sinks) { l_star += exp(sinks[head] - m_star); }

    // 4) write
    dst[head*DV + tid] = (l_star > 0.0f) ? o_tid / l_star : 0.0f;
}
```

---

## F. 정확도 검증 전략

### F.1 Accumulator

- Partial o_s: F32 필수.
- Meta (m_s, ℓ_s): F32 pair 필수.
- 최종 dst: F32.
- 전체 F32 accumulator이므로 split=1 경로와 수치적으로 거의 동일해야 함. Split 수에 따라 cos-sim 감소 시 버그.

### F.2 흔한 버그 패턴

1. **m_s=-∞, ℓ_s=0 케이스 누락**: split 범위가 전부 causal-masked 시. Merge에서 exp(-∞-m*)=0 자동 safe. 모든 split 이 상태면 dst=0.
2. **o_s를 ℓ_s로 한번 나눈 뒤 merge** — 잘못. 반드시 unnormalized partial 저장.
3. **sinks 중복 적용**: split kernel에서 skip, reduce에서만 처리.
4. **Causal mask 경계**: decode는 pos=n_kv-1. is_causal 체크는 k_row<=pos. Split boundary에서 동일 조건 유지.
5. **Mask offset**: q1은 row=0 고정. split에서도 동일.

### F.3 테스트 추가 (권장)

`engine/src/bin/test_backend.rs`:
- KV sweep: {128, 512, 1024, 2048, 4096, 8192}.
- 각 길이에서 CPU ref vs GPU split=1 vs GPU split={2,4,8,16,32} cos-sim.
- 허용 오차: logits cos-sim ≥ **0.9995**.
- 엣지 케이스: n_kv<kv_per_split, n_kv%kv_splits!=0, is_causal=1+n_kv=1, mask/sinks/max_bias 조합.

---

## G. Split 수 자동 결정 휴리스틱

### G.1 권장 단순식 (Metal 스타일)

```rust
fn compute_kv_splits(n_kv: usize) -> usize {
    if n_kv < 512 { return 1; }
    let s = (n_kv / 1024).max(1).min(32);   // 1K token당 1 split, 상한 32
    s.next_power_of_two().min(32)
}
```

- 1K: 1, 2K: 2, 4K: 4, 8K: 8, 16K: 16, 32K+: 32
- 튜닝은 실측 기반. 초기 구현은 단순식, 측정 후 조정.

### G.2 파라미터 근거

| 상수 | 값 | 근거 |
|------|------|------|
| CHUNK | 1024 | vLLM 512와 Metal 2K 경계의 중간. 4K에서 splits=4 확보 |
| MIN_CTX_FOR_SPLIT | 512 | short ctx 회귀 방지. 128 tok에서는 갭 없음 |
| MAX_SPLITS | 32 | Metal nwg=32와 동일. VRAM 상한 256 KiB 수용 가능 |

---

## 권장 구현 방향 (1문단 요약)

현재의 `flash_attn_f32_f16_q1` 커널을 **옵션 3 (threshold + 2-kernel split/reduce)** 으로 확장한다. 구체적으로 (a) 기존 커널을 `_q1_split` 으로 일반화하여 `global_work_size = [Q1_WG_SIZE, n_head, kv_splits]`, 세 번째 축이 KV 범위 `[split_idx*chunk, (split_idx+1)*chunk)` 만 처리하고 partial `(m_s, ℓ_s, o_s=unnormalized)` 를 F32 버퍼에 쓰게 한다. (b) `flash_attn_q1_reduce` 신규 커널로 partial 을 log-sum-exp merge하여 최종 F32 dst에 쓴다 (`kv_splits=1`이면 호출 스킵). (c) `flash_attention_decode_gpu` 에서 `n_kv < 512` 면 기존 단일 커널 fallback, 그 외 `kv_splits = next_pow2(min(32, max(1, n_kv/1024)))`. partials/meta 는 사전 할당 workspace 배치 권장. (d) F32 accumulator 유지, sinks 는 reduce kernel에서 단일 적용, causal/mask/softcap 은 split kernel에서 기존 로직 재사용. (e) `test_backend`에 KV 길이 sweep + split sweep cos-sim 테스트 추가해 `>=0.9995` 검증. llama.cpp OpenCL 포트에는 아직 구현되지 않은 경로라 독자적 이점이며, CUDA `fattn-common.cuh:730-779` 의 `flash_attn_combine_results` 와 Metal `kernel_flash_attn_ext_vec_reduce` 가 수식·메모리 레이아웃 최상의 참조 구현이다.

---

## 출처 / 참조

**우리 리포**:
- 타겟 커널: `engine/kernels/flash_attn_f32_f16.cl:214-243` `flash_attn_f32_f16_q1`
- 동반 커널: `engine/kernels/flash_attn_f16.cl`, `engine/kernels/flash_attn_f32.cl`
- Dispatch: `engine/src/backend/opencl/mod.rs:1905` `flash_attention_decode_gpu`, 호출부 `:2715`

**llama.cpp 참조**:
- CUDA: `llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh:730-779, 781-1035`
- Metal: `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6318-6748, 6865-6905`
- Metal dispatch: `llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2919-3033`

**외부**:
- Flash Decoding 블로그: https://crfm.stanford.edu/2023/10/12/flashdecoding.html
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- vLLM PagedAttention V2: https://github.com/vllm-project/vllm/blob/main/csrc/attention/attention_kernels.cu

---

## 불확실성

- Adreno 830 CU 수, wave size, local_mem_size 공식값 — 런타임 `clGetDeviceInfo` 확인 필요
- Qwen2.5 1.5B의 정확한 `n_head_q / n_head_kv / head_dim` — 모델 config 확인 필요
- Image2d_t 변환 이득 — 실측 전 도입 권장 안 함
- `CHUNK=1024` 최적값 — 초기 구현 후 512/2K 비교 측정 필요
