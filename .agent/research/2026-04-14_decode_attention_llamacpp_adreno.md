# Decode Attention state-invariant 최적화 조사 (llama.cpp Adreno 대조)

**작성일**: 2026-04-14
**계기**: c446ca8 실측에서 Decode만 갭 1.37× (우리 82.6 ms/tok vs llama.cpp 60.2 ms/tok = 22 ms/tok 차).
Decode 프로파일: attention 43.8% (~36 ms/tok) 지배.
**조사 제약**: Adreno 830 per-thread state 32 float4 상한. state 증가 기법은 부정적 가설 (B-2 −23%, B-3 −38% revert 이력).

## 핵심 결론

**우리(c446ca8) Q1 flash attention은 llama.cpp보다 이미 앞서 있다.** 22 ms/tok 갭의 주 원인은 attention 내부 기법 차이가 **아닐 확률이 매우 높다**.

### 근거 (파일:라인)

| 기법 | llm_rs2 (c446ca8) | llama.cpp |
|---|---|---|
| subgroup reduction | ✓ `sub_group_reduce_max/add` (engine/kernels/flash_attn_f32_f16.cl:555, 589) | ✗ SLM barrier tree reduce 6단계 × 3회 (ggml/src/ggml-opencl/kernels/flash_attn_f32_f16.cl:296-303, 337-345, 357-364) |
| wavefront 고정 | ✓ `qcom_reqd_sub_group_size("half")` (466) | ✗ attribute 없음 (210) |
| Adreno 전용 분기 | — | ✗ flash_attn에 분기 없음 (`use_adreno_kernels`는 matmul 전용: ggml-opencl.cpp:4178-4188) |
| dispatch (wg=64, global=[64,n_head,1]) | = | = |
| per-thread state | 64 float4 (q_priv 32 + o_acc 32) | 64 float4 (동일) |

**즉 llama.cpp Q1은 우리 B-4 이전 버전에 해당**. B-4로 추월 완료.

## state-invariant 튜닝 Top 3 (확실성 순)

### 1. mask NULL 분기 제거 + `opencl_unroll_hint` (공짜)
- **근거**: 우리 커널 466-549줄에서 K loop 내부마다 `if (mask_base != NULL)` 분기 2회. 호스트에서 mask는 decode 시 항상 NULL (`mod.rs:2190-2191` `ArgVal::mem_null()`). 컴파일 타임 상수화 가능.
- **해결**: `-DHAS_MASK=0` 옵션 + `#if HAS_MASK` 가드 + causal/slope/ALiBi 경로 제거.
- **효과**: 4472 K iter × 2 pass = 8944회 분기 제거. `opencl_unroll_hint`는 llama.cpp gemv_noshuffle.cl:112에서 사용 패턴 확인.
- **예상**: 0~1.5 ms/tok
- **위험**: 매우 낮음 (순수 코드 삭제)
- **난이도**: 낮음 (1일)

### 2. `sub_group_broadcast`로 Q state 64 lane 분산
- **근거**: llama.cpp gemv_noshuffle.cl:14-63 패턴 — `sub_group_broadcast(y.s0, lane_id)`로 activation을 1 lane만 load하고 나머지 63 lane이 register로 공유.
- **적용**: flash attention Q1에서도 `q_priv[DK_VEC=32]`를 64 lane에 분산 → lane당 평균 0.5 float4 (state 감소).
- **효과**: per-lane state 64 → 33 float4. Adreno register pressure 완화 → occupancy 상승. 이후 state-증가 기법 허용 기반공사.
- **예상**: 2~5 ms/tok
- **위험**: 중간 (broadcast 호출이 K-row loop 내부마다 발생, 드라이버 shuffle 구현 의존)
- **난이도**: 중간 (3~5일)

### 3. GQA Q-head time-multiplex (n_kv 기반 dispatch)
- **근거**: Qwen 2.5-1.5B n_heads_q=12, n_kv=2, gqa_ratio=6. 현재 K/V를 6번 중복 load (llama.cpp도 동일 낭비). 시간축 직렬화로 K/V traffic 1/6.
- **주의**: B-2(o_acc 6세트 동시 유지)와 다름. **o_acc를 시간축으로 갈아끼워 state 불변 유지**.
- **K/V traffic**: 105 MB → 17 MB 이론.
- **예상**: 0~10 ms/tok (WG occupancy 저하 트레이드오프)
- **위험**: 높음 (WG 수 12→2 급감, Slice 점유율 문제). n_kv*2 타협안 필요.
- **난이도**: 높음 (1~2주, 파일럿 micro-bench 선행 필수)

## 메타 결론 — attention 외 영역이 ROI 더 클 가능성

**22 ms/tok 갭을 attention 내부 튜닝만으로 메우기 어렵다**. 다음 영역을 후속 조사해야 한다:

### A. matmul_ffn 18.8% (Decode GEMV, Q4_0)
- llama.cpp `mul_mv_q4_0_f32_8x_flat` + Adreno `gemv_noshuffle_q4_0_f32` (gemv_noshuffle.cl:14-63)의 sub_group_broadcast activation 공유 패턴.
- 우리 `mul_mv_q4_0_noshuffle` 계열 반영도 재점검 필요.

### B. matmul_wo 8.2% (512×2048 F16 GEMV)
- llama.cpp의 `image1d_buffer_t` 텍스처 캐시 경로 (`mul_mat_Ab_Bi_8x4.cl:23`).
- 우리 `mul_mv_f16_f32`는 buffer load만 사용 — texture L1 미활용.

### C. Kernel fusion (overhead 감축)
- llama.cpp는 `norm_mul_add`, `rope_norm` 등 fuse.
- 우리는 RMSNorm만 fused. RoPE+QKV, silu+mul 후보.

### D. kv_update 3.3%
- llama.cpp는 `set_rows`. 우리 `kv_scatter_f32_to_f16` 구조 비교.

## 착수 권장

1. **즉시**: Top 1 (mask/unroll 정리) — 공짜.
2. **병행**: Top B/A 후속 researcher — matmul_ffn/matmul_wo Adreno 기법 조사.
3. **유보**: Top 2/3 — 선행 조사 결과 보고 결정.

## 파일 경로

### llama.cpp 소스 (/home/go/Workspace/llama.cpp/)
- `ggml/src/ggml-opencl/kernels/flash_attn_f32_f16.cl` (Q1: 210-373)
- `ggml/src/ggml-opencl/kernels/gemv_noshuffle.cl` (broadcast 패턴: 14-63)
- `ggml/src/ggml-opencl/kernels/mul_mat_Ab_Bi_8x4.cl` (image1d_buffer_t: 20-48)
- `ggml/src/ggml-opencl/ggml-opencl.cpp` (Q1 dispatch: 8509-8643, use_adreno: 4178-4188)

### llm_rs2 소스
- `engine/kernels/flash_attn_f32_f16.cl` (Q1: 466-618, B-4 reduction: 555/589/605-608)
- `engine/src/backend/opencl/mod.rs` (Q1 dispatch: 2101-2216)
- `engine/src/backend/opencl/plan.rs` (Plan Q1: 886-990)
