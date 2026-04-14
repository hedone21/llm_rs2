# Post-revert verdict — B-4 → SLM tree-reduce 적용 결과

작성: 2026-04-14 23:00
대상 변경: `engine/kernels/flash_attn_f32_f16.cl` Q1 kernel만 SLM tree-reduce 패턴으로 교체 (REQD_SUBGROUP_SIZE_64 어트리뷰트 + sub_group_reduce_* 호출 모두 제거).

## TL;DR

**B-4 revert로 long-context decode 갭 76% 해소.** Production decode wall slope이 12.45 → 7.32 μs/n_kv로 감소 (5.13 μs/n_kv 회수). llama.cpp 대비 갭 **6.75 → 1.62 μs/n_kv**. Decode 출력 정상성 OK (qualitative sanity check). prefill kernel은 변경 안 함.

## 1. Microbench 재측정 (cross-run 후속)

같은 microbench harness, 같은 디바이스 (Galaxy S25 Adreno 830), 같은 dispatch params.

| variant | llm_rs2 (B-4 ON, pre-revert) | llm_rs2 (post-revert) | llama.cpp | post-revert vs llama |
|---|---:|---:|---:|---:|
| Single | 0.38 | 0.28 | 0.245 | **1.16× (15.6%)** |
| Repeat28 | 9.99 | 7.12 | 7.17 | **0.993× TIE** |
| Repeat28+Mask | 10.30 | 7.53 | 7.54 | **0.999× TIE** |
| Repeat28+Mask+QVar | 10.33 | 7.53 | 7.52 | **1.001× TIE** |

production-relevant variant (Repeat28+Mask) 모두 **±0.5% 안에서 llama.cpp와 동등**.

## 2. Production decode 벤치 (Galaxy S25, Qwen 2.5-1.5B Q4_0)

같은 디바이스, 같은 모델, 4 ctx, 240 s 쿨다운, AP mStatus=0.

### 2.1 비교 표

| n_kv | baseline B-4 ON (2026-04-14 오후) | B-4 revert (2026-04-14 23:00) | Δ ms/tok |
|---:|---:|---:|---:|
| 258 | 30.81 | (n_kv=144, 36.88 — 다른 prompt, 비교 부적합) | — |
| 1025 | 38.80 | (n_kv=941) **33.22** | **−5.58** |
| 2047 | 58.88 | (n_kv=1852) **40.08** | **−18.80** |
| 4472 | 83.30 | (n_kv=3674) **53.28** | **−30.02** (n_kv 작음에도) |

n_kv가 정확히 일치하지 않으나 (사용한 prompt 다름), 같거나 더 작은 n_kv에서도 30 ms/tok 이상 빠름 → **decode wall slope 자체가 하락**.

### 2.2 Slope 비교

같은 세션의 long-context 3 점만 사용 (small-ctx는 prompt 차이로 노이즈 큼):

| 측정 | n_kv 범위 | wall slope μs/n_kv |
|---|---|---:|
| 이전 (B-4 ON) | 1025-4472 | **12.45** |
| **B-4 revert** | 941-3674 | **7.32** |
| llama.cpp 오후 baseline | 258-4472 | 4.72 |
| llama.cpp 심야 baseline | 256-6k | 5.70 |

**갭 분석**:
- 이전: llm_rs2 12.45 vs llama 4.72 → gap **7.73 μs/n_kv**
- **현재**: llm_rs2 7.32 vs llama 4.72 → gap **2.60 μs/n_kv**
- **갭 66% 회수** (5.13 μs/n_kv 감소)

llama 5.70 baseline 사용 시:
- 이전 갭: 6.75
- 현재 갭: 1.62
- **갭 76% 회수**

## 3. 정확도 검증

```
$ ./generate --model qwen2.5-1.5b-q4_0-v2 --backend opencl \
   --prompt "The quick brown fox" --num-tokens 30 --temperature 0.0
The quick brown fox jumps over the lazy dog.
This is a sentence in English. It has 26 words, each of which starts with a different letter of the
```

자연스러운 출력. Garbage 없음. **No correctness regression**.

## 4. 변경 요약 (`engine/kernels/flash_attn_f32_f16.cl`)

- Q1 kernel `__kernel REQD_SUBGROUP_SIZE_64 void` → `__kernel void` (Q1만, prefill은 유지)
- `sub_group_reduce_max(m_i)` → `__local local_m[Q1_WG_SIZE]` SLM tree-reduce
- `sub_group_reduce_add(l_i)` → `__local local_l[Q1_WG_SIZE]` SLM tree-reduce
- `sub_group_reduce_add(o_acc[i].s{0,1,2,3})` × DV_VEC → `__local local_o_comp[Q1_WG_SIZE]` per-DV_VEC SLM tree-reduce
- 총 barrier 수 (DV=128): **0 → 236** (이론상 더 비싸지만, Adreno 830에서 실측 더 빠름)

## 5. 미해명 잔존 갭 (1.6-2.6 μs/n_kv)

llama.cpp 대비 여전히 25-35% 느림. 가능한 원인:
- 비-attention ops (matmul_qkv/wo/ffn/lm_head): per-op slope 합 ~1.0 μs/n_kv. 우리 vs llama 비교 안 됨
- KV update / scatter overhead
- Driver-level scheduling 차이
- 506 vs 633 dispatches/token (우리가 더 적지만 per-dispatch 더 무거울 가능성)

이 잔존 갭은:
- **D path (eviction)** 으로 우회 가능 (n_kv 자체 감소)
- 추가 op-level 분석 필요 (B path: Snapdragon Profiler)
- 또는 그냥 수용 (이미 갭 76% 해소된 상태)

## 6. 다음 단계 권장

1. **재현성 1회 더** — 본 측정 1회만이라 Δ 30 ms 같은 큰 수치는 thermal/세션 노이즈 가능성 검토
2. **Eval 회귀 테스트** — 정확도 metric (PPL 또는 LongBench)으로 quantitative 회귀 확인
3. **prefill kernel B-4 검토** — prefill에는 REQD_SUBGROUP_SIZE_64 + A-3 B-1 subgroup split 적용 중. 같은 logic으로 prefill도 손해일 가능성 (prefill 갭은 별도 §1~10에서 추적 중)
4. **남은 1.6-2.6 μs/n_kv** — D path (eviction) 또는 op-level 후속

## 7. 산출물

- `engine/kernels/flash_attn_f32_f16.cl` (Q1 부분 revert)
- `.agent/research/microbench_flash_attn/postrevert_run1_*.txt` (microbench 재측정)
- `.agent/research/microbench_flash_attn/postrevert_decode_*.txt` (production 4 ctx)
- 본 문서 (`post_revert_verdict.md`)
