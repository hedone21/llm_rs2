# A1 Phase A 재감사 — llm_rs2 vs llama.cpp flash_attn Q1 kernel 격차 원인 규명

작성: 2026-04-14, researcher 위임 결과 (메인 세션 저장)

## 1. TL;DR

**Phase A(2026-04-13)의 "byte-identical" 결론은 prefill kernel에만 적용되며, decode Q1 kernel에는 적용되지 않는다.** 현재 우리 `flash_attn_f32_f16_q1`과 llama.cpp의 동 이름 kernel은 **구조적으로 다르다**. 우리 쪽에는 `REQD_SUBGROUP_SIZE_64` qcom attribute + `sub_group_reduce_max`/`sub_group_reduce_add` 기반 barrier-free reduction 최적화가 추가돼 있고, llama.cpp는 여전히 `__local` SLM + tree-reduce + barrier 패턴이다. 즉 **우리가 이미 Adreno-최적화된 버전을 쓰고 있으며 llama.cpp가 unoptimized baseline**이다.

이와 독립적으로, **본 세션 microbench 결과(10.30 μs/n_kv)와 llama.cpp attention 추정(~4.5 μs/n_kv)의 2.3× 갭 전제 자체가 의심스럽다**. Option 3 Phase B의 attn-event 직접 측정은 llama.cpp attention = 17.61 μs/n_kv로 **우리보다 더 가파르다**고 기록했는데, 동시에 같은 디바이스 wall-slope은 4.79 μs/n_kv. 이 두 값은 내부적으로 모순(attention이 total의 367%)이다. **본 재감사 결론: 비교 전제인 "llama.cpp attention ~4.5 μs/n_kv" 자체가 유효 근거 없음**. 현 상황에서 "우리 kernel이 2.3× 느리다"는 주장을 지탱할 확정적 evidence가 없다.

**가설 verdict 요약**: H1 REJECT (kernel 파일 동일), H2 REJECT (dispatch 동일), H3 REJECT (옵션 동일), H4 UNDETERMINED (binary 추출 미수행), H5 REJECT (llama.cpp도 단일 dispatch). **진짜 발견**: **Phase A가 prefill만 비교했고 Q1은 비교하지 않았으며**, 비교해보니 **우리 Q1이 더 고도화**됨.

## 2. llama.cpp Adreno decode flash attention dispatch 경로 트레이스

### 2.1 호출 함수 체인
`ggml_cl_flash_attn()` @ `/home/go/Workspace/llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp:8566`

### 2.2 n_q==1, Q=F32, K/V=F16 분기 (Qwen 2.5-1.5B decode)
- @8597-8604: `n_q==1 && is_mixed` → `kernels_flash_attn_f32_f16_q1.at({128,128})` 선택. 분기 끝.
- 다른 variant 선택 조건 없음 (device vendor check 없음). Adreno 830도 NVIDIA GPU도 같은 kernel 선택.

### 2.3 Kernel 이름 & 컴파일 위치
- kernel 이름: `flash_attn_f32_f16_q1` (ggml-opencl.cpp:1841)
- source file: `kernels/flash_attn_f32_f16.cl` (@1801)
- 컴파일 루프(1811-1847): 9개 dk/dv 테이블 × {f16, f32, f32_f16} × {prefill, q1} = 54개 kernel. 벤더 분기 없음.

### 2.4 Dispatch 파라미터 (@8688-8692)
```cpp
if (n_q == 1) {
    const size_t wg_size = 64;
    size_t local_work_size[] = { wg_size, 1 };
    size_t global_work_size[] = { wg_size, (size_t)(n_head * n_batch) };
    backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);
}
```
- work_dim=2, gws=[64, n_head*n_batch]=[64, 12], lws=[64, 1]
- 40 args (@8655-8686)

### 2.5 컴파일 옵션 (@837-839, 1816-1820)
```
"-cl-std=CLX.Y -cl-mad-enable -cl-unsafe-math-optimizations"
" -cl-finite-math-only -cl-fast-relaxed-math"
" -D DK=128 -D DV=128 -D BLOCK_M=32 -D BLOCK_N=32"
```
X.Y는 device CL_DEVICE_OPENCL_C_VERSION에서 runtime-extraction. Adreno 830 = CL C 2.0 → 우리 CL2.0 하드코딩과 실효 동일.

## 3. Source diff 재확인

### 3.1 Prefill kernel `flash_attn_f32_f16` (n_q>1)
- 우리: A-3 B-1 subgroup split layout (DK==128 일 때)
- llama.cpp: 비-분할 단일 구현
- **Phase A 시점 이후 prefill에 변경 들어감 → "1 byte도 다르지 않음" 진술 stale**

### 3.2 Decode Q1 kernel `flash_attn_f32_f16_q1` (n_q==1) — **핵심**

| 지점 | 우리 (llm_rs2 L466-618) | llama.cpp (L210-373) |
|------|------------------------|----------------------|
| kernel 속성 | `__kernel REQD_SUBGROUP_SIZE_64 void` | `__kernel void` |
| subgroup ext | `cl_qcom_reqd_sub_group_size : enable` | 없음 |
| m_i reduce | `sub_group_reduce_max(m_i)` (L555) | `__local local_m[64]` + 6단 tree + 6 barrier |
| l_i reduce | `sub_group_reduce_add(l_i)` (L589) | `__local local_l[64]` + 6단 tree + 6 barrier |
| o_acc reduce | DV_VEC × 4개 `sub_group_reduce_add` (L603-608) | DV_VEC × (`local_o_comp` + 6단 tree + 7 barrier/iter) |
| barrier 수 (DV=128) | **0** | m=6 + l=6 + DV_VEC(32) × 7 = **236** |
| SLM 사용량 | 0 | ~1.25 KB |
| 예상 이론 성능 | 빠름 (barrier-free) | 느림 (heavy barriers) |

우리 kernel L552-554 주석 ("B-4: subgroup reduce — Q1_WG_SIZE=64 == subgroup width on Adreno. Replaces 6-stage SLM tree-reduce (7 barriers) with single barrier-free shuffle."): **의도적 교체였음**.

### 3.3 Phase A 보고서 섹션 2.6 재해석
Phase A L45 "DV=128에서 32회 반복 × 7단 barrier = 224 barrier" 진술은 작성 시점 우리 kernel을 서술. 이후 B-4 sprint에서 0으로 줄임. Phase A 보고서 자체에는 "decode Q1 diff 수행했다"는 기록 없음.

## 4. Dispatch 파라미터 비교 표

| 항목 | llm_rs2 | llama.cpp | 동일? |
|------|---------|-----------|-------|
| kernel 이름 | flash_attn_f32_f16_q1 | flash_attn_f32_f16_q1 | ✓ |
| work_dim | 2 | 2 | ✓ |
| gws | [64, 12, 1] | [64, 12] | ✓ (실효) |
| lws | [64, 1, 1] | [64, 1] | ✓ |
| arg count | 40 | 40 | ✓ |
| arg layout | 동일 | 동일 | ✓ |
| is_causal | 0 (하드코딩) | 0 (n_q==1이므로) | ✓ |
| mask | NULL | NULL (decode 일반) | ✓ |

**dispatch 측면 기능적 완전 동일**.

## 5. 컴파일 옵션 비교 표

| 옵션 | llm_rs2 | llama.cpp | 동일? |
|------|---------|-----------|-------|
| `-cl-std=` | CL2.0 (조건부) | CL{ver} (runtime) | Adreno=2.0이면 동일 |
| `-cl-mad-enable` | ✓ | ✓ | ✓ |
| `-cl-unsafe-math-optimizations` | ✓ | ✓ | ✓ |
| `-cl-finite-math-only` | ✓ | ✓ | ✓ |
| `-cl-fast-relaxed-math` | ✓ | ✓ | ✓ |
| `-DDK=128` | (공백X) | `-D DK=128` (공백O) | OpenCL spec 동일 |
| 외 DK/DV/BLOCK_M/N | 동일 | 동일 | ✓ |

## 6. 가설별 Verdict

| 가설 | 결론 | 근거 |
|------|------|------|
| H1 (Phase A가 잘못된 파일 비교) | **REJECT** | 같은 파일, 벤더 분기 없음 |
| H2 (Dispatch 파라미터 다름) | **REJECT** | gws/lws/args 모두 동일 |
| H3 (컴파일 옵션 다름) | **REJECT** | 의미적 동일 |
| H4 (드라이버 컴파일 차이) | **UNDETERMINED** | binary 추출 미수행. 그러나 §3에서 **source 자체가 다르므로** moot |
| H5 (단일 vs 다중 kernel) | **REJECT** | 양쪽 모두 단일 dispatch |

### 진짜 발견 1: Phase A는 prefill만 비교했으며, Q1 kernel은 우리가 더 최적화됨
§3.2 표 참조. 우리는 llama.cpp의 SLM tree-reduce 패턴을 `sub_group_reduce_*` + `REQD_SUBGROUP_SIZE_64`로 교체한 B-4 최적화 적용. **이론상 우리가 빨라야 함**.

### 진짜 발견 2: 비교 전제 ("llama.cpp attention ~4.5 μs/n_kv") 가 phantom target
- Option 2B: llama.cpp attention = 17.22 μs/n_kv (profile build)
- Option 3 Phase B: llama.cpp attention = 17.61 μs/n_kv (event-only, profile zero-overhead 주장)
- Wall slope: llama.cpp TOTAL = 4.79 μs/n_kv (non-profile)
- attn-event 빌드 wall eval: TOTAL slope ≈ 18.03 μs/n_kv (Phase B 보고서)

**llama.cpp의 "빠른 4.79"는 비-event·비-profile queue에서만 관측**. event flag 추가 시 3-4× 성능 저하. 즉 Adreno 드라이버에서 `CL_QUEUE_PROFILING_ENABLE` 자체가 flash attn 경로에 막대한 패널티. 우리 `--profile-events`도 같은 flag → **양쪽 다 profile overhead 포함값**이라 사과-사과 비교 가능. 그 비교에서 우리(13.23)가 llama(17.22~17.61)보다 4.4 μs/n_kv 더 빠름.

**"우리가 2.3× 느리다" 주장의 근거인 4.5 μs/n_kv는 wall 5.70 × 0.8 추정값**. 직접 측정된 적 없음. **microbench 10.30과 직접 비교 가능한 값이라는 evidence 없음**.

## 7. 다음 액션 권장

### 7.1 즉시 — 비교 baseline 자체 재검증
1. **microbench cross-run** (가장 결정적): llama.cpp Q1 kernel을 우리 ocl harness에서 직접 컴파일·dispatch해서 같은 입력 같은 디바이스에서 직접 비교. **유일한 결정적 방법**.
2. **B-4 최적화 정당성 검증**: 우리 Q1을 임시 SLM tree-reduce로 revert해서 production slope 변화 측정. 느려지면 B-4 이득, 빨라지면 손해.
3. **non-profile attention contribution 측정**: `--profile-events` ON/OFF 비교로 attention의 profile bias 정량.

### 7.2 B (Snapdragon Profiler) — 보류 권고
§6에서 "우리가 kernel 수준 더 고도화" static analysis 결과 얻음. SP trace는 개선 방향 가이드는 되지만 "왜 llama가 더 빠른가"의 답은 아님 (애초에 더 빠른지 자체가 불확실). 7.1 cross-run이 선행되어야 함.

### 7.3 D (Eviction) — 계속 유효
Kernel-level 갭 원인 미확정이어도 eviction은 독립 전략. n_kv를 줄여 slope × n_kv 감소. 우선순위 높음.

### 7.4 미결 조사 (본 세션 범위 초과)
- `clGetProgramInfo(CL_PROGRAM_BINARIES)` 두 kernel ISA 추출 (H4 최종 판정)
- `CL_KERNEL_PRIVATE_MEM_SIZE`, `CL_KERNEL_LOCAL_MEM_SIZE`, `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` 비교 (register/SLM pressure)
- Adreno에서 `sub_group_reduce_max/add`가 SLM tree-reduce보다 빠른지 단일 microbench (B-4 정당성)

## 8. 메인 세션에 남기는 질문

1. **microbench 10.30 vs llama.cpp 4.5의 비교 자체가 타당한가?** 4.5는 wall 5.70에 0.8을 곱한 추정. wall 5.70 (baseline non-profile non-event) 자체가 attn-event 측정 시 18.03으로 바뀌는 변동성 고려하면, "llama.cpp attention = 4.5"라는 숫자는 **어떤 측정 조건에서도 재현된 적 없음**.
2. **우리 B-4 최적화(sub_group_reduce)가 Adreno 830에서 정말 이득인가?** `cl_qcom_reqd_sub_group_size("half")` 속성이 실제로 존중됐는지 `clGetKernelSubGroupInfo` 확인 필요.
3. **A-3 B-1 subgroup split (prefill용)을 Q1에 이식 가능한가?** Q1은 WG 전체가 단일 Q-row 처리 → A-3 B-1 "Q-row를 2 lanes로 쪼갬"이 구조적으로 무의미. **적용 불가**.

## 9. 참고 파일 경로

**우리 엔진**:
- `engine/kernels/flash_attn_f32_f16.cl` (Q1: L466-618, prefill: L64-464)
- `engine/src/backend/opencl/mod.rs:74-102, 687-710` (컴파일 옵션)
- `engine/src/backend/opencl/plan.rs:962-1069` (dispatch)

**llama.cpp**:
- `ggml/src/ggml-opencl/kernels/flash_attn_f32_f16.cl` (Q1: L210-373, prefill: L29-208)
- `ggml/src/ggml-opencl/ggml-opencl.cpp:506-513` (kernel map)
- `ggml/src/ggml-opencl/ggml-opencl.cpp:831-839` (base compile_opts)
- `ggml/src/ggml-opencl/ggml-opencl.cpp:1786-1851` (flash attn 컴파일)
- `ggml/src/ggml-opencl/ggml-opencl.cpp:8566-8700` (`ggml_cl_flash_attn` dispatcher)
- `ggml/src/ggml-opencl/ggml-opencl.cpp:707-739` (`enqueue_ndrange_kernel`, profile/attn-event 분기)
