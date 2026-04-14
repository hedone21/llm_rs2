# Host OpenCL on NVIDIA CUDA: Q5+ NaN / CL_OUT_OF_RESOURCES

**Date filed**: 2026-04-14
**Filed by**: feat/gemma3-4b-support branch investigation
**Scope**: NVIDIA host OpenCL (RTX 3090 Ti, OpenCL 1.2), all Gemma3 models (1B + 4B 확인)
**Follow-up branch**: fix/host-opencl-nvidia-fallback (TBD)

## Symptom

- Gemma3 4B on NVIDIA OpenCL: eval-ll Q1–Q4 NLLs match CPU to 3 decimals, Q5+ all NaN. Single-prompt generation produces immediate EOS (blank).
- Gemma3 1B on NVIDIA OpenCL: eval-ll Q1–Q4 numeric, Q5 crashes with `clEnqueueReadBuffer → CL_OUT_OF_RESOURCES`.
- Gemma3 4B on PoCL (`OCL_PLATFORM=portable`): eval-ll 40/40 numeric (655s), single-prompt "a city that never sleeps...", exit 0. **Engine state management is correct.**
- CPU backend: eval-ll 40/40 numeric (447s).

## Root cause (hypothesis)

NVIDIA OpenCL 1.2 (no `cl_khr_subgroups`, no native `convert_float(half)`) fails to JIT-compile 9 Adreno-specific kernels at model load time:

1. `mul_mv_f32_f32.cl` — `get_sub_group_local_id`, `sub_group_reduce_add` undeclared
2. `simple_ops.cl` — same subgroup intrinsics
3. `mul_mv_q4_0_f32.cl` — undeclared macros `N_SIMDGROUP`, `N_DST`, `N_SIMDWIDTH`
4. `mul_mv_f16_f32_l4.cl` — `convert_float(half)` no match (x2 variants)
5. `flash_attn_f32_f16.cl` DK=64 — `convert_float(half)` no match
6. `flash_attn_f32_f16.cl` DK=128 — `convert_float(half)` no match
7. `mul_mm_f16_f32_l4_lm.cl` — `convert_float(half)` no match
8. `kivi_q2.cl` — same

Engine reports "4 warnings and 8 errors generated. / 2 warnings and 7 errors generated. (x2) / 1 error generated." at model load, activates fallback paths. Fallback runs on NVIDIA but accumulates state corruption from Q5 onward (exact trigger op TBD).

## Reproduction (x86_64 Linux + NVIDIA CUDA OpenCL 1.2)

```bash
# 4B — Q5+ NaN
./target/release/generate \
  --model-path /home/go/Workspace/llm_rs2/models/gemma3-4b \
  --backend opencl --kv-type f32 --max-seq-len 4096 \
  --qcf-mode both --eval-ll \
  --eval-batch /tmp/race_h_smoke_10q.json --greedy

# 1B — Q5 CL_OUT_OF_RESOURCES
./target/release/generate \
  --model-path /home/go/Workspace/llm_rs2/models/gemma3-1b \
  --backend opencl --kv-type f32 --max-seq-len 4096 \
  --qcf-mode both --eval-ll \
  --eval-batch /tmp/race_h_smoke_10q.json --greedy
```

## Workaround

`OCL_PLATFORM=portable OCL_DEVICE_TYPE=cpu` forces PoCL — 엔진의 기존 platform-selection env vars (`backend/opencl/mod.rs` 플랫폼 선택 로직)가 그대로 동작하므로 즉시 사용 가능.

## Suggested fix path

**Option A — OpenCL C 1.2 compliant kernel porting (권장)**:
1. 9개 실패 커널을 OpenCL C 1.2 표준으로 포팅: subgroup intrinsics 제거, `convert_float(half)` → `vload_half`/`vstore_half` 또는 비트캐스트(`as_float(...)`)로 대체.
2. Host-side capability check 추가: `cl_khr_subgroups` 미지원 시 non-subgroup 경로로 명시적 라우팅(JIT 오류 fallback 아닌 컴파일 타임 분기).

**Option B — 단기 NaN 원인 bisect**:
1. per-layer NaN 발생 지점 이분 탐색으로 특정 op 식별 (Q5 번째 질문이 시작하는 레이어 N부터 순서대로 CPU 우회).
2. 단일 offending op 수정 — 노력: 4–8시간.

## Effort estimate

- Option A (전체 커널 포팅): 1–2일
- Option B (단기 bisect + 단일 fix): 4–8시간

## Artifacts (this session)

- `notes/gemma3_4b_task10_status.md` — Task 10 raw output, NVIDIA 실행 로그 발췌
- `docs/40_gemma3_support.md` §9.3 — 최종 백엔드별 검증 결과 표
- `notes/gemma3_4b_final_status.md` — Phase 4 전체 최종 상태

## Impact

Gemma3-4B PR (`feat/gemma3-4b-support`)은 **CPU 및 PoCL 경로 기준 merge-ready**. 프로덕션 타겟인 Android Adreno는 별도 테스트 파이프라인에서 검증. 본 이슈는 4B 지원의 차단 요인이 아니라 호스트 개발 환경 제약.
