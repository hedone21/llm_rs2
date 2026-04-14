# Long Context CPU Attention 최적화 — 4K+ 성능 갭 해소

(과거 섹션 1~10은 이전 커밋 참고; §11만 최신 상태)

## 11. 다음 세션 시작 가이드 (2026-04-14 14:30 — Decode 갭 분석 6연속 네거티브, 작업 중단)

### 🎯 세션 최종 상태

**Decode 갭 30.6 ms/tok (우리 82.3 ms vs llama.cpp 51.7 ms = 우리 63% 수준) 확정**. 원인 조사 6연속 네거티브, 추가 조사 ROI 낮음 판정으로 **일단 중단**. 다음 세션에서 이어 받을 수 있도록 조사 결과 전량 보존.

### 현재 master 상태
- HEAD: `1d8b2ef docs(skills/worktree): add cwd-in-target guard + stale entry recovery` (Gemma 3 4B 지원 작업이 06212ab 이후 머지됨)
- Phase 1 커밋 `06212ab feat(opencl): event-based per-op profiling (--profile-events)` — 본 세션 산출물, 메인 브랜치 merged
- 빌드: `cargo check` 클린, Android aarch64 release OK
- 디바이스(/data/local/tmp): 최신 `generate` 배포됨 (profile-events 기능 포함)

### 🔥 2026-04-14 오후 재확정 실측 (쿨다운 4분 + thermal 0 대기)

**안정 조건에서 두 엔진 완벽 일관**:
| 엔진 | Prefill tok/s | Decode ms/tok |
|---|---|---|
| llm_rs2 Run 1 | 102.2 | 82.19 |
| llm_rs2 Run 2 | 102.2 | 82.42 |
| llama.cpp Run 1 | 102.85 | 51.64 |
| llama.cpp Run 2 | 103.10 | 51.69 |

**갭 확정**:
- **Prefill**: 거의 parity (우리 99%)
- **Decode**: 30.6 ms 느림 (우리 63% 수준)

이전 §11의 "22 ms 갭"은 thermal-throttled llama.cpp 측정과 fresh llm_rs2 측정을 비교한 결과로 **잘못된 수치**였음. 정확한 갭은 **30.6 ms**.

### 갭 패턴 — Launch count 기반 가설의 증거

| | llm_rs2 | llama.cpp | 갭 |
|---|---|---|---|
| Prefill (큰 GEMM, 적은 launch) | 102 tok/s | 103 tok/s | 0.6% |
| Decode (작은 GEMV, 364 launches/token) | 82 ms | 52 ms | **59%** |

Launch 수에 비례한 갭 증가 → per-kernel launch overhead 또는 per-launch host work가 원인이라는 가설. **하지만 실측으로 반증됨** (아래 §Phase 2 참고).

---

### ✅ 이번 세션 성과

#### 1. Event-based per-op profiling 구현 (`06212ab`)
- CLI flag `--profile-events` 추가
- OpenCL `CL_QUEUE_PROFILING_ENABLE` + event capture로 synchronize 없이 per-op GPU μs 측정
- 6파일 +582/−309 LOC
- llama.cpp `GGML_OPENCL_PROFILING=ON` 빌드와 apples-to-apples 비교 가능

#### 2. llama.cpp profile 빌드 + CSV 측정 확보
- 재빌드: `/home/go/Workspace/llama.cpp/build-android-profile/bin/llama-cli`
- 측정 결과: `/tmp/llama_cl_profiling.csv` (79,274 이벤트)
- decode 구간 per-op 분해: attention 85ms, FFN 12ms, lm_head 10.6ms 등

#### 3. Microbench 도구 추가 (untracked)
- `engine/src/bin/microbench_launch.rs` — Rust ocl::core로 noop kernel 10000회 launch
- `experiments/benchmarks/microbench_launch.c` — 동일 로직 C raw OpenCL
- Adreno 830 실측: **Rust 7.93 μs vs C 7.82 μs (+0.11 μs 노이즈)**. ocl crate wrapper 오버헤드 없음 확인.

---

### 6연속 네거티브 조사 요약

| # | 가설 | 조사 결과 | 보고서 |
|---|---|---|---|
| 1 | attention 커널 느림 | ✗ 우리 Q1 flash attn가 llama.cpp보다 앞섬 (B-4 subgroup reduction) | `.agent/research/2026-04-14_decode_attention_llamacpp_adreno.md` |
| 2 | matmul_ffn/matmul_wo 미반영 기법 | ✗ Q4_0 GEMV 완전 동등, F16 GEMV 우리가 더 공격적 | `.agent/research/2026-04-14_decode_matmul_llamacpp_adreno.md` |
| 3 | kernel fusion 부족 | ✗ 대부분 갖춤. `add_rms_norm_oop` / `kv_scatter_f32_to_f16`은 우리가 앞섬 | `.agent/research/2026-04-14_decode_kernel_fusion_llamacpp_adreno.md` |
| 4 | per-layer `backend.flush()` × 32 | ✗ Qwen production은 `execute_plan` 경로 사용 → flush 애초에 호출 안 됨 | `.agent/research/2026-04-14_decode_sync_audit.md` |
| 5 | decode 경로 sync point audit | ✗ plan.rs 0건, backend method hot path 0건. llama.cpp와 동일 구조 | 위 같은 문서 + 수동 감사 |
| 6 | Rust ocl wrapper launch overhead | ✗ raw C와 0.11 μs 차이 (1.4%). 364 × 7.8 μs = 2.85 ms만 설명 | 위 microbench 바이너리 |

### 조사/측정 자료 전량 보존 위치

`.agent/research/`:
- `2026-04-14_decode_attention_llamacpp_adreno.md` — attention 기법 대조
- `2026-04-14_decode_matmul_llamacpp_adreno.md` — Q4_0/F16 GEMV 대조
- `2026-04-14_decode_kernel_fusion_llamacpp_adreno.md` — fusion 기법 대조
- `2026-04-14_decode_microbench_plan.md` — micro-bench 설계 (Phase 1/2/3/4)
- `2026-04-14_decode_sync_audit.md` — decode 경로 sync 감사

`/tmp/` (세션-local):
- `llama_cl_profiling.csv` — llama.cpp per-op 79k 이벤트
- `llama_decode_per_op.md` — llama.cpp 집계 표
- `llm_rs2_profile_events/*.json` — llm_rs2 event-based decode JSON
- `bench_gap_verify_v2_results.txt` — 5x5 초기 갭 측정 (thermal variance 문제 있음)
- `gpu_freq_v3_results.txt` — GPU 주파수 분포 (llm_rs2 1096 MHz vs llama.cpp 1051 MHz 평균)
- `freq_llm_rs2_r*.csv`, `freq_llamacpp_r*.csv` — raw freq traces
- `bench_gap_verify_v2.sh`, `gpu_freq_bench_v3.sh` — 벤치 스크립트 (재사용 가능)

---

### 📊 핵심 수치 정리 (다음 세션 참고용)

**Production 실측 (Qwen 2.5-1.5B Q4_0, Adreno 830, 4472 prefill tok)**:
- llm_rs2: Prefill 102 tok/s, Decode 82.3 ms/tok (12.2 tok/s)
- llama.cpp: Prefill 103 tok/s, Decode 51.7 ms/tok (19.3 tok/s)
- **Gap: Prefill 0.6%, Decode 30.6 ms (40%)**

**Profile 모드 (`--profile-events` / `GGML_OPENCL_PROFILING=ON`)**:
- llm_rs2: Decode 96 ms/tok, GPU kernel sum 92 ms (96.4%)
- llama.cpp: Decode 114.94 ms/tok, GPU kernel sum 111.56 ms (97.1%)
- 주의: CL_QUEUE_PROFILING_ENABLE이 per-kernel GPU HW time을 **부풀리는** 것으로 확인됨 (llama.cpp 115→52 ms compression 불가능). Profile 수치를 production 절대값으로 사용 금지 — **상대 %만 유효**.

**GPU 주파수 (production decode 중)**:
- llm_rs2 평균 1096 MHz (48% at 1200 MHz) — 더 공격적 사용
- llama.cpp 평균 1051 MHz (20% at 1200 MHz) — 여유 있게 사용
- 우리만 3% 시간 525 MHz로 드랍 (순간 throttling 또는 idle sleep)

**Raw OpenCL per-launch (noop kernel)**: 7.82 μs (Adreno 830 하한)

---

### 🤔 남은 미해결 가능성 (다음 세션 시작점)

6연속 네거티브 후에도 30.6 ms 갭은 실재. 남은 가능성:

**가능성 A — Production 실제 kernel GPU time 차이 (측정 미완)**
- Profile 모드가 왜곡되므로 production kernel sum은 알 수 없음
- **실험**: event-based profiling을 **최소 커널만** 선택해 켜기 (전체 activate가 아닌 sparse)
- 또는: production binary에서 `clEnqueueMarker` + `clWaitForEvents`로 특정 op만 측정
- 예상 가치: 높음 (정확한 갭 분포 확정)

**가능성 B — Launch count 차이**
- 우리 decode token당 launch 수를 정확히 세어 llama.cpp와 비교
- 우리 `execute_plan`의 step 수 × 각 step의 backend method 내부 kernel 수 = N
- llama.cpp의 `ggml_backend_opencl_graph_compute` iteration 수 + fusion 후 실제 enqueue 수 = M
- `N - M` × 7.82 μs = launch 수 차이로 설명 가능한 시간
- 예상 가치: 중간

**가능성 C — Per-launch host work 차이**
- 우리 backend method는 `get_cl_mem()` downcast chain × 4 variant 체크
- 각 launch마다 ~1 μs × 364 launches = 0.36 ms (미미)
- 하지만 `set_kernel_arg` 호출 수 차이는 측정 필요
- 예상 가치: 낮음

**가능성 D — UMA buffer cache coherence 문제**
- 우리는 zero-copy (CL_MEM_ALLOC_HOST_PTR) 사용
- 일부 버퍼가 CPU 포인터 통해 접근 → kernel 간 cache flush 강제 가능성
- llama.cpp는 device-local buffer 위주
- 예상 가치: 높음 (가설 검증 안 된 영역)

**가능성 E — OpenCL driver optimization hint**
- llama.cpp는 특정 kernel에 `cl_qcom_priority_hint` 같은 Qualcomm extension 사용 여부
- 우리가 놓친 driver-level 최적화 플래그
- 예상 가치: 중간

### 다음 세션 착수 옵션

**Option 1 (권장)**: **가능성 A** — production 경로에서 특정 op만 event profiling. `plan.execute` 내부에 선택적 event capture 추가.

**Option 2**: **가능성 D** — UMA cache coherence 감사. `CL_MEM_USE_HOST_PTR` / `CL_MEM_ALLOC_HOST_PTR` 사용 지점 전수 + llama.cpp의 버퍼 flag 정책 대조.

**Option 3**: **가능성 B** — launch count 직접 비교. llama.cpp `ggml-opencl.cpp` decode path + llm_rs2 `plan.execute` 한 iteration의 `clEnqueueNDRangeKernel` 호출 수 카운트.

**Option 4**: **작업 전환** — 30.6 ms 갭 수용, 다른 타겟으로 이동 (모델 범용성 확장, quality, 메모리 최적화 등).

---

### 🔥 기존 핵심 교훈 (여전히 유효)

**Adreno 830 DK=128 flash attention per-thread state 상한 = 32 float4** (A-3 B-1/B-4 성공, B-2/B-3 revert 이력).

**방법론**:
1. per-thread 배열 키우는 변경은 실측 전까지 부정적 가설
2. 대역폭 vs register 트레이드오프는 Adreno에서 거의 항상 register 쪽이 진다
3. state 불변 + reduction/barrier 비용만 줄이는 최적화가 ROI 가장 높음
4. llama.cpp가 구현하지 않은 기법은 Adreno 구조적 한계 가능성 우선 의심
5. 쿨다운 **240초 이상 + Thermal Status 0 확인** 후 측정 (120초는 부족)
6. **실측 게이트 = tok/s + 출력 품질 둘 다**
7. **Profile 모드 수치는 상대 비교만 유효**, 절대값은 production과 다름

### 다음 세션 진입 명령

```
.agent/todos/long_context_attention_optimization.md §11 읽고 Option 1부터 (또는 Option 4 결정)
```

### 재현용 명령

**벤치** (쿨다운 240초 + thermal 0 대기 후 안정 측정):
```bash
adb shell "cd /data/local/tmp && ./generate \
  -m /data/local/tmp/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-v2.gguf \
  --prompt-file /data/local/tmp/prompt_6k.txt \
  -n 32 -b opencl --kv-type f16 --max-seq-len 6144 --ignore-eos"
```

**llama.cpp 벤치**:
```bash
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp \
  ./llama-cli-orig \
  -m /data/local/tmp/Qwen2.5-1.5B-Instruct-q4_0.gguf \
  -f /data/local/tmp/prompt_6k.txt \
  -n 32 -ngl 99 -c 6144 --no-display-prompt \
  --temp 0.8 --top-p 0.9 --top-k 40 -no-cnv"
```

**Event-based profiling**:
```bash
adb shell "cd /data/local/tmp && ./generate \
  -m /data/local/tmp/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-v2.gguf \
  --prompt-file /data/local/tmp/prompt_6k.txt \
  -n 128 -b opencl --kv-type f16 --max-seq-len 6144 --ignore-eos \
  --profile-events --profile-dir /data/local/tmp/results/profile_events"
```

**Microbench (launch overhead)**:
```bash
adb shell "cd /data/local/tmp && ./microbench_launch"
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./microbench_launch_c"
```

### 실측 장비 상태 기록 (재현성)

- Device: Galaxy S25 (Adreno 830, Snapdragon 8 Elite for Galaxy)
- 쿨다운: **벤치 간 240초 + Thermal Status 0 확인 필수**. 120초로는 분산 큼.
- Qwen Q4_0 gguf on device: `/data/local/tmp/Qwen2.5-1.5B-Instruct-q4_0.gguf` (1011 MiB)
- llm_rs2 모델: `/data/local/tmp/llm_rs2/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-v2.gguf`
- 프롬프트: `/data/local/tmp/prompt_6k.txt` (4472 tokens)
- llama.cpp 바이너리: `/data/local/tmp/llama-cli-orig` (production, profile OFF)
- llama.cpp profile 빌드: `/home/go/Workspace/llama.cpp/build-android-profile/bin/llama-cli` (필요시 디바이스 `llama-cli-prof`로 재배포)
