# Cross-run verdict — llm_rs2 Q1 vs llama.cpp Q1 (Adreno 830)

작성: 2026-04-14 22:30
원본: `crossrun_run1_*.txt`, `crossrun_run2_*.txt`

## TL;DR

**llama.cpp Q1 kernel이 우리 production Q1보다 Adreno 830에서 33-55% 빠름.** 같은 디바이스, 같은 KV/Q/O 버퍼, 같은 dispatch params 격리 측정. **B-4 sprint의 "sub_group_reduce + REQD_SUBGROUP_SIZE_64" 최적화가 Adreno 830에서 OFFENSE가 아니라 LOSS였음 확정**. A1 researcher의 "이론상 우리가 빠를 것" 추정이 실측 정반대.

§12 갭 6.75 μs/n_kv 중 약 **2.5 μs/n_kv (37%)가 우리 Q1 kernel choice의 직접 손실**. 즉시 B-4 revert로 회수 가능.

## 측정 환경

- 디바이스: Galaxy S25, QUALCOMM Adreno(TM) 830
- AP thermal mStatus=0, 시작온도 60.2°C
- 빌드: `target/aarch64-linux-android/release/microbench_flash_attn` (release)
- 같은 microbench harness가 두 program (production + vendored llama.cpp)을 컴파일하고 동일 buffer/dispatch로 호출
- 4 variants × 2 layouts × 4 n_kv × 30 iters × 2 engines = 1920 measurements/run × 2 runs

## 결과 표 (HeadMajor, per-token slope)

| variant | llm_rs2 (B-4) μs/n_kv | llama.cpp (SLM tree) μs/n_kv | ratio | llama 우세 |
|---|---:|---:|---:|---:|
| **Run 1** ||||
| Single | 0.368 | 0.264 | 1.391× | 39.1% |
| Repeat28 | 10.086 | 7.446 | 1.355× | 35.5% |
| Repeat28Mask | 10.311 | 7.751 | 1.330× | 33.0% |
| Repeat28MaskQ | 10.370 | 7.766 | 1.335× | 33.5% |
| **Run 2** ||||
| Single | 0.386 | 0.249 | 1.551× | 55.1% |
| Repeat28 | 9.904 | 7.046 | 1.406× | 40.6% |
| Repeat28Mask | 10.290 | 7.637 | 1.347× | 34.7% |
| Repeat28MaskQ | 10.289 | 7.645 | 1.346× | 34.6% |
| **Mean (Repeat28+Mask)** | **10.30** | **7.69** | **1.34×** | **34.0%** |

## 핵심 발견

### 1. B-4 최적화 (sub_group_reduce + qcom subgroup attribute) 가 Adreno 830에서 LOSS

A1 researcher 보고서의 §3.2 표:
- 우리 Q1: 0 barriers, 0 SLM, sub_group_reduce_*
- llama.cpp Q1: 236 barriers, ~1.25 KB SLM, SLM tree-reduce

**이론**: 우리가 빨라야 함 (barrier-free).
**실측**: 우리가 33-55% 느림.

가능한 원인:
1. **Adreno OpenCL의 `sub_group_reduce_*` 구현이 실제로 비효율적**. 드라이버가 내부적으로 SLM 구현을 사용하면서 추가 오버헤드 발생.
2. **`cl_qcom_reqd_sub_group_size("half"=64)` 어트리뷰트가 occupancy를 제한**. 64-wide wavefront 강제 → SP당 동시 실행 가능 wavefront 수 감소.
3. **Register pressure**: sub_group_reduce는 broadcast된 결과를 모든 lane에 유지하나, SLM tree-reduce는 lane 0만 결과 보유 → SLM tree 쪽이 register 여유 더 큼 → occupancy 유리.
4. **Compile path 차이**: REQD_SUBGROUP_SIZE_64 어트리뷰트가 컴파일 hint로 다른 ISA 생성 패턴을 유도.

### 2. §12 갭 6.75 μs/n_kv의 분해 (개정판)

| 기여 원인 | μs/n_kv | % of gap |
|---|---:|---:|
| **B-4 sub_group_reduce LOSS** | ~2.5 (= (10.30-7.69) × 12.45/14.74 wall scaling) | **37%** |
| Production 환경 (FFN cache thrashing) | ~1.5 (Option A 2.93 × wall scaling) | 22% |
| 미해명 (다른 op, KV update, profile overhead 외) | ~2.75 | 41% |
| **Total** | 6.75 | 100% |

**즉시 B-4 revert로 갭 37% (2.5 μs/n_kv) 회수 가능**.

### 3. PosMajor에서도 동일 패턴

| variant | llm_rs2 (PosMajor) | llama.cpp (PosMajor) | ratio |
|---|---:|---:|---:|
| Repeat28 | 10.21 | 7.15 (run1) / 6.98 (run2) | 1.43-1.46× |

PosMajor에서도 llama가 30%+ 빠름. 즉 KV stride 무관한 kernel 자체 차이.

## 권장 액션 (우선순위)

### 즉시 (저비용, 큰 효과): B-4 revert 검증
1. **`engine/kernels/flash_attn_f32_f16.cl`의 Q1 kernel만 SLM tree-reduce 패턴으로 교체** (llama.cpp 버전 그대로 차용)
2. production decode 벤치 재측정 (Galaxy S25, Qwen 2.5-1.5B Q4_0, 같은 4 ctx)
3. 예상 효과: attention slope 13.23 → ~10 μs/n_kv (25% 감소), wall slope 12.45 → ~10.3 μs/n_kv (17% 감소)
4. **단** prefill kernel은 영향 없음 (B-4은 Q1만 적용). prefill은 별도 작업.

### 후속 (조사): 왜 Adreno에서 sub_group_reduce가 느린가
- `clGetKernelSubGroupInfo`로 두 kernel의 실제 subgroup 사이즈 확인
- `CL_KERNEL_PRIVATE_MEM_SIZE`, `CL_KERNEL_LOCAL_MEM_SIZE` 비교
- `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` 비교
- Snapdragon Profiler trace로 wavefront occupancy + register spill 확인

### 미해명 4.6 μs/n_kv 잔존
- B-4 revert 후에도 wall 10.3 vs llama 5.7 = **4.6 μs/n_kv 갭 남음**
- 이는 attention 외부 또는 production 환경 효과 (FFN/matmul 차이)
- 별도 후속 작업 (D path eviction이 합리적)

## 산출물

- `engine/src/bin/microbench_flash_attn.rs` (cross-run 확장)
- `.agent/research/microbench_flash_attn/llamacpp_q1_flash_attn.cl` (vendored)
- `.agent/research/microbench_flash_attn/crossrun_run1_2228.txt`, `crossrun_run2_2230.txt` (raw)
- `.agent/research/microbench_flash_attn/cross_run_verdict.md` (본 문서)
