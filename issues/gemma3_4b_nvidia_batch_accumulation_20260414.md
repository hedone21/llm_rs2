# Gemma3 4B on NVIDIA OpenCL: batch-size ≥ 10에서 Q5 CL_OUT_OF_RESOURCES

**Date filed**: 2026-04-14
**Filed by**: fix/host-opencl-nvidia-fallback bisect follow-up
**Parent issue**: `host_opencl_nvidia_fallback_20260414.md`
**Scope**: Gemma3 4B, NVIDIA RTX 3090 Ti, OpenCL 1.2
**Severity**: medium — 호스트 개발 환경 제약. 프로덕션 Android Adreno에는 영향 없음.

## Symptom

Gemma3 4B `--backend opencl` `--eval-ll` 사용 시 **배치 크기가 특정 임계치 이상일
때만** Q5(다섯 번째 질문)에서 `clEnqueueReadBuffer → CL_OUT_OF_RESOURCES (-5)`로
파열. Q1-Q4는 수치 정상.

## Reproduction (fix/host-opencl-nvidia-fallback HEAD @ 24789da 이상)

```bash
OCL_PLATFORM=NVIDIA ./target/release/generate \
  --model-path /home/go/Workspace/llm_rs2/models/gemma3-4b \
  --backend opencl --kv-type f32 --max-seq-len 4096 \
  --qcf-mode both --eval-ll \
  --eval-batch /tmp/race_h_smoke_10q.json --greedy
```

## Bisect results (배치 크기별)

| 배치 | 결과 |
|------|------|
| Q1만 (1Q) | PASS |
| Q1→Q5 2개만 | PASS |
| Q1-Q4 (4Q) | PASS |
| Q2-Q3-Q4-Q5 (4Q, same-prompt×3 + transition) | PASS |
| Q1-Q5 (5Q) | PASS |
| Q1-Q6 (6Q) | PASS |
| Q1-Q8 (8Q) | PASS |
| **Q1-Q10 (10Q) | FAIL at Q5 (3회 연속 재현)** |
| Q1-Q40 (40Q) | FAIL at Q5 |

Q5 자체는 여러 시나리오에서 통과했고, 배치에 Q9/Q10(다른 프롬프트 race_h_1494)이
포함될 때만 Q5가 파열. **Q1-Q4가 실제로 실행되는 중간에 발생**하므로 Q9/Q10의
내용이 당장 실행되는 건 아님 — 배치 크기가 영향을 주는 up-front 할당/드라이버
상태가 원인으로 추정.

Q1-Q4 같은 프롬프트(race_h_108) PASS 후 Q5(다른 프롬프트 race_h_418) 첫 번째
`read_buffer` 호출에서 터진다는 패턴은 parent issue의 1B 경우와 동일하지만,
parent fix (shrink 경로 제거)로는 해소되지 않는다.

## Hypothesis

- Parent fix 이후 `shrink_to_fit` 경로는 제거됐으므로, 남은 원인은 NVIDIA
  드라이버의 **질문 간 리소스 누적**. `eval_loop.rs:155`의
  `backend.synchronize()`가 누적을 완전히 털어내지 못하는 것으로 보임.
- 질문 수가 적을 때는 Q1-Q4 처리 중 누적이 임계치에 못 미쳐서 PASS. 배치 list
  길이가 늘면 JSON 파싱·토크나이저 상태·임의의 warm-up 변화로 누적 경계가
  앞당겨짐 가능성.
- Gemma3 4B는 head_dim=256로 GPU flash attn 경로 없음 (`log: GPU-fallback prefill
  attn: dtype=F32 head_dim=256 reason="head_dim not in {64, 128}"`). 1B도 동일
  head_dim=256이지만 레이어 수(26 vs 34) 및 intermediate dim 차이로 4B에서만
  누적이 임계 초과.

## Investigation notes / eval_loop (pre-existing 방어 코드)

```rust
// eval_loop.rs:88-108
let mut prefill_ws = if backend.is_gpu() {
    PrefillWorkspace::new(&WorkspaceConfig { ... }, max_seq_len, memory, backend.clone()).ok()
} else { None };

// eval_loop.rs:151-155
hook.reset_caches(kv_caches);
// Flush GPU queue to release deferred OpenCL buffers from previous question.
// Without this, NVIDIA's runtime accumulates pending buffer releases → OOM.
backend.synchronize()?;
```

엔진은 이미 (a) PrefillWorkspace를 한 번만 할당하고 (b) 질문 간 synchronize로
deferred release 유도를 시도하지만, 4B + 10Q 이상에서는 부족한 것으로 보임.

## Next steps (별도 세션 권장)

1. **Instrumentation**: `ocl` 호출 지점에서 `clCreateBuffer`/`clReleaseMemObject`
   counter를 유지, Q1-Q5 사이 Live OpenCL 객체 수 추적. NVIDIA 드라이버가
   deferred release를 언제 수행하는지 확인.
2. **Force release 실험**: `clFinish` 대신 `clReleaseCommandQueue` 후 재생성,
   또는 질문 간 `clFlush + clFinish + N ms wait`로 driver GC 유도.
3. **Prefill path 축소 테스트**: head_dim=256 → CPU fallback로 우회하는 op이
   어느 GPU 버퍼를 남기는지 식별. `prefill_ws.k_cast/v_cast` lazy init 버퍼가
   질문 간 재사용되지 않고 재할당될 가능성 검토.
4. **배치 크기 실제 영향 확인**: 동일 10Q를 중복 없이 "Q1×10"으로 만든 배치로
   실험 — 동일 프롬프트 반복만으로 Q5 OOR 재현되는지.

## Workaround

**PoCL**: `OCL_PLATFORM=portable OCL_DEVICE_TYPE=cpu`로 즉시 회피 (확인됨, 40/40
numeric, ~650s).

**CPU backend**: `--backend cpu`도 안정적 (40/40 numeric, ~450s).

**NVIDIA GPU + 배치 분할**: `scripts/eval_ll_batched.py`로 chunk당 프로세스를 재시작하여 드라이버 누적 회피. 40/40 numeric, ~260s (CPU 대비 43% 단축). 상세 가이드: `docs/gemma3_4b_nvidia_batched_eval_guide.md`.

## Impact

- 호스트 개발 환경에서 4B 대규모 eval-ll이 NVIDIA GPU로 안 돌아간다는 제약.
- 프로덕션 타겟(Android Adreno)에는 영향 없음 — 해당 환경은 cl_khr_subgroups
  지원되므로 fallback 경로 자체가 안 탄다.
- Gemma3 1B는 parent fix로 NVIDIA에서 완전 복구.
