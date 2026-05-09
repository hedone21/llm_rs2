# Handoff: OpenCL → QNN-GPU Backend Migration (다음 세션 작업)

**작성일**: 2026-05-09
**기준 commit**: `a110920` (origin/master push 완료)
**Device**: Galaxy S25 (Snapdragon 8 Elite for Galaxy, Adreno 830 GPU + Hexagon V79 HTP)

---

## 0. 한 줄 요약

R-Y 검증 완료 → **HTP+QNN-GPU shared rpcmem zero-copy가 Adreno 800 series에서 유일한 작동 path**로 확인됨. 사용자가 OpenCL → QNN-GPU **마이그레이션 진행** 결정. 계획 세우기 전에 **risk 확인 단계 우선**, 그 후 순차 plan. 본 세션은 /compact 예정이므로 self-contained handoff.

---

## 1. 결정 배경 — 왜 마이그레이션이 필요한가

### Phase 6/9: GPU multi-queue HW serialize 확정
- OpenCL same-ctx 1.93x, Vulkan same-family 1.945x (Phase 6 + Phase 9)
- 같은 chip 안에서 진정 parallel 불가 (Adreno single-issue command processor)

### Phase 32b: HTP+GPU heterogeneous parallel은 가능
- Phase 32b-1: HTP prebuilt MatMul 정확성 ✓
- Phase 32b-2: HTP MatMul + GPU GEMV concurrent → C3/(C1+C2) = 0.510 (진정 H/W parallel)
- 핵심 fact: 다른 chip이면 진정 parallel

### R-Y 검증: OpenCL과 HTP zero-copy 불가, QNN-GPU와는 가능
| Path | 결과 |
|------|------|
| `cl_khr_external_memory_dma_buf` + acquire/release | ✗ rpcmem fd format 비호환 |
| `cl_qcom_ext_host_ptr` (vendor) | ✗ Adreno 800 series에서 모든 alloc_type reject |
| **HTP + QNN-GPU shared rpcmem (다른 memType, 같은 fd)** | **✓ 1024/1024 정확성** |

### 결론
- Adreno 800에서 **raw OpenCL과 HTP 간 zero-copy는 vendor가 막음**
- 유일한 작동 path = **QNN runtime의 GPU backend (libQnnGpu.so) 통한 cross-backend ION/DMA-BUF sharing**
- 따라서 production HTP+GPU heterogeneous async swap을 진정 활용하려면 **forward path를 raw OpenCL → QNN-GPU로 마이그레이션** 필요

---

## 2. 현재 코드 상태

### 검증된 microbench (commit `a110920`)
- `engine/src/bin/microbench_qnn_probe.rs` — QNN runtime probe
- `engine/src/bin/microbench_htp_correctness.rs` — Phase B Q1 (HTP ElementWiseAdd 정확성)
- `engine/src/bin/microbench_htp_gpu_parallel.rs` — Phase C Q2 (HTP+GPU compute parallel)
- `engine/src/bin/microbench_htp_throughput.rs` — Phase D Q3 (raw 0.04 GB/s, FAIL)
- `engine/src/bin/microbench_htp_graph_reuse.rs` — Phase 11 (graph reuse OK)
- `engine/src/bin/microbench_htp_rpcmem_throughput.rs` — Phase 12 (rpcmem alloc OK, exec 여전히 느림)
- `engine/src/bin/microbench_htp_matmul_correctness.rs` — Phase 32b-1 (MatMul 정확성)
- `engine/src/bin/microbench_htp_gpu_matmul_concurrent.rs` — Phase 32b-2 (DDR-heavy parallel)
- `engine/src/bin/microbench_htp_opencl_interop.rs` — R-Y opt 1 (raw OpenCL ↔ HTP 실패)
- `engine/src/bin/microbench_htp_qnngpu_share.rs` — **R-Y opt 3 PASS (HTP + QNN-GPU)**

### Production code 변경
- 0 lines (모든 microbench bin에 격리)

### 환경
- QNN SDK 2.33 ($USER_SERVER:~/Workspace/qnn/qairt/2.33.0.250327/) — `third_party/qnn_sdk_2.33/` (gitignored)
- bindgen build.rs 동작 (NDK sysroot 자동)
- `/data/local/tmp/qnn/` 에 5개 SDK lib push 됨:
  - libQnnHtp.so, libQnnHtpV79Stub.so, libQnnHtpPrepare.so, libQnnSystem.so
  - libQnnHtpV79Skel.so (DSP skel)
  - **libQnnGpu.so (GPU backend, 새로 추가됨)**

### Cargo features
- `qnn = ["libloading", "bindgen"]` — microbench-only PoC
- `vulkan = ["ash"]` — Phase 9 archived
- 기본 production: `opencl` (변경 없음)

### Workspace state
- HEAD: `a110920` origin/master (push 완료)
- working tree: clean

---

## 3. 다음 세션의 첫 작업 — Risk 확인

사용자 명시: **"리스크 확인부터 하고 하나씩 순차적으로 계획을 세워서 진행"**

마이그레이션 plan 들어가기 **전에** risk를 먼저 식별·평가해야 한다. 계획은 그 다음.

### 알려진/예상 Risk 카테고리

#### R-A: QNN-GPU backend functional coverage
- **R-A1**: 우리 production이 사용하는 모든 OpenCL kernel을 QNN-GPU prebuilt op으로 표현 가능한가
  - 우리 kernel list (`engine/kernels/*.cl`):
    - mul_mv (matmul), mul_mv_f16, dequantize_q4_0, flash_attn_v75, rmsnorm, rope, residual_add, swiglu, soft_max 등
  - QNN OpDef.h에 prebuilt op 매칭: MatMul, RMSNorm, Softmax, Sigmoid, Tanh 등 — 일부는 ✓, attention/flash_attn은 ✗
- **R-A2**: prebuilt 부재 op의 **QNN custom op** 작성 비용 (Hexagon SDK + HVX intrinsic? 또는 OpenCL-only kernel registration?)
- **R-A3**: KV cache 관리 (HeadMajor layout, dynamic shape) QNN graph가 표현 가능한가
- **R-A4**: Q4_0 quantization scheme — QNN의 quantization 표현이 우리 GGUF Q4_0과 호환되는가

#### R-B: Performance parity (raw OpenCL → QNN-GPU)
- **R-B1**: 우리 production GEMV (Adreno-tuned, mul_mv_f16) vs QNN-GPU MatMul TBT 비교
- **R-B2**: 우리 flash_attn 커널 vs QNN-GPU attention op (있다면) 또는 graph composition
- **R-B3**: 모델 forward 1 step e2e 비교 (Qwen2.5-1.5B Q4): 현 OpenCL vs QNN-GPU only
- **R-B4**: graph build/finalize cost (production load time 영향)

#### R-C: Memory model
- **R-C1**: KV cache (HeadMajor, dynamic grow-on-demand) QNN graph로 표현 — variable shape 지원 여부
- **R-C2**: weight buffer (cl_mem 600MB) → QNN tensor mapping. swap path 영향
- **R-C3**: workspace tensor (forward 중간 결과) QNN graph 안에 포함 가능한가, 별도 buffer인가

#### R-D: Migration scope/cost
- **R-D1**: 전체 backend rewrite 시 일정 (researcher: 8~12주)
- **R-D2**: 부분 마이그레이션 가능한가 (예: FFN만 QNN-GPU, attention은 OpenCL 유지)
  - 단점: 매 layer마다 backend 전환 cost
- **R-D3**: production 회귀 risk (현재 OpenCL backend 유지하면서 QNN-GPU 추가)

#### R-E: 통합 architecture
- **R-E1**: 두 backend가 같은 weight buffer share 가능한가 (HTP write → QNN-GPU read 검증됨, 다른 path 미확인)
- **R-E2**: Manager/Resilience signaling을 QNN-GPU에서도 사용 가능 (현재는 OpenCL backend 전용)
- **R-E3**: KV eviction policy / D2O / pressure pipeline이 QNN graph와 호환

#### R-F: 외부 의존성/제약
- **R-F1**: QNN SDK 2.33 license (재배포 금지, EULA 동의 사용자만)
- **R-F2**: device QNN runtime version (vendor api 2.20.0 vs SDK 2.25)
- **R-F3**: 다른 device (Pixel/Jetson 등)에서는 QNN-GPU 미사용 — 별도 backend 유지 필요

#### R-G: 정확성/품질
- **R-G1**: QNN-GPU forward output 정확성 vs OpenCL baseline
- **R-G2**: Q4_0 dequantization 정확도 (rounding/saturation 차이)
- **R-G3**: KV cache state 일치 (mid-decode swap에서 backend 전환 시)

### 권장 risk 확인 절차

다음 세션 시작 시:

```
Phase R: Risk Assessment (마이그레이션 plan 전 단계)
  R-A: Functional coverage 확인 (3~5일)
    - 우리 OpenCL kernel 모두 list-up
    - QNN OpDef.h에서 매칭 op 검색
    - 부재 op 정리 (custom op 또는 graph composition 필요)
  R-B: Performance probe (2~3일)
    - QNN-GPU MatMul TBT vs 우리 GEMV TBT
    - 단순 1-layer forward 비교
  R-C: Memory model 검토 (1~2일, 코드 read-only)
  R-D, R-E, R-F, R-G: 코드 분석 + 사용자 토론 (1~2일)

→ Phase R 결과로 마이그레이션 scope 결정 + 정식 plan
```

각 risk 확인 결과로 다음 결정:
- **GREEN**: 모든 risk 해소 가능 → 8~12주 마이그레이션 plan
- **YELLOW**: 부분 마이그레이션만 가능 → 제한된 scope (예: FFN만)
- **RED**: 핵심 risk 미해결 → paper로 정리 + 마이그레이션 보류

---

## 4. 핵심 Reference

### 우리 측정 결과 보고서
- `papers/eurosys2027/_workspace/experiment/swap_overhead_phase10_htp_feasibility.md` (Phase A~E)
- 본 handoff 후 작성 예정: `papers/.../swap_overhead_phase32b_*.md` (Phase 32b 결과)

### 외부 참고
- HeteroInfer (SOSP'25): https://arxiv.org/abs/2501.14794 — NPU+GPU heterogeneous LLM inference
- QNN GPU backend tutorial: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/gpu_qnnmem_api_tutorial.html
- QNN HTP shared buffer tutorial: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/htp_shared_buffer_tutorial.html
- ORT QNN EP HTP shared memory PR: https://github.com/microsoft/onnxruntime/pull/23136
- llama.cpp QNN backend WIP PR: https://github.com/ggml-org/llama.cpp/pull/12063 (active, reference 가능)

### 우리 production code 진입점 (마이그레이션 영향 범위)
- `engine/src/backend/mod.rs` — Backend trait
- `engine/src/backend/opencl/mod.rs` — 현 OpenCL backend
- `engine/src/backend/cpu/mod.rs`, `engine/src/backend/cuda_pc/mod.rs` — 다른 backend (parallel structure)
- `engine/kernels/*.cl` — 11개 OpenCL kernel
- `engine/src/models/llama/llama_model.rs` — forward dispatch
- `engine/src/layers/llama_layer.rs` — per-layer forward

### Risk R-A1 사전 작업 — kernel 매핑 list
```
mul_mv             → QNN MatMul (확실)
mul_mv_f16         → QNN MatMul (FP16, 확실)
dequantize_q4_0    → QNN Dequantize 또는 prebuilt 미지원 → custom op
flash_attn_v75     → QNN ScaledDotProductAttention (확실하지 않음, custom 가능성)
rmsnorm            → QNN RmsNorm (확실)
rope               → QNN RoPE 또는 custom op
residual_add       → QNN ElementWiseAdd (확실)
swiglu             → QNN SwiGLU 또는 ElementWiseMul + Sigmoid composition
soft_max           → QNN Softmax (확실)
KV cache update    → QNN graph variable input + Concat 또는 별도 path
```

각 kernel을 prebuilt op + custom으로 분류해 R-A1 결과 만들어야 함.

---

## 5. /compact 후 다음 세션 시작 절차

```bash
cd /home/go/Workspace/llm_rs2

# 1. 상태 확인
git log --oneline -5         # HEAD = a110920
git status                    # clean

# 2. 본 handoff 읽기
cat .agent/todos/handoff_qnn_gpu_migration_2026-05-09.md

# 3. Risk 확인 단계 시작
ls engine/kernels/*.cl                                 # 우리 OpenCL kernel list
grep "QNN_OP_" third_party/qnn_sdk_2.33/include/QNN/QnnOpDef.h | head -50  # QNN prebuilt ops

# 4. (필요 시) device 환경 확인
adb devices                                            # R3CY408S4HN
adb shell ls /data/local/tmp/qnn/                      # SDK libs 5개 + GPU
```

### 사용자 요청 흐름
1. 사용자가 plan을 직접 명시하기 전, **risk 확인 단계**부터 시작
2. risk 카테고리 (R-A~R-G) 각각을 확인 완료 후 결과 정리
3. 결과 기반으로 **순차적 plan** 작성 (사용자와 합의)
4. plan 승인 후 implementation 단계 진입

---

## 6. 잠정 알고 있는 것 (다음 세션에서 검증)

- ✓ HTP backend가 ION fd로 register 가능 (Phase 12)
- ✓ QNN-GPU backend가 DMA_BUF fd로 register 가능 (R-Y opt 3)
- ✓ 같은 rpcmem fd를 두 backend가 다른 memType으로 동시 register OK
- ✓ HTP write → QNN-GPU read 정확성 100% (1024/1024)
- ✓ QNN graph 재사용 가능 (Phase 11 R8)
- ✓ HTP MatMul prebuilt op 1024×4096 FP32 정확성 1.13e-6 max abs error
- ? QNN-GPU MatMul vs 우리 production GEMV 성능 (R-B1, 다음 세션)
- ? flash_attn 같은 복잡 op QNN custom 작성 가능성 (R-A2)
- ? KV cache dynamic shape QNN 표현 가능성 (R-C1)

---

## 7. Do / Do Not

### DO
- Risk 확인을 **반드시** 마이그레이션 plan 전에 수행
- 각 risk를 measurable + falsifiable로 정의
- Microbench bin으로 격리 측정 (production code 미변경)
- QNN docs (HTP shared buffer, GPU QnnMem) 우선 reference
- 결과 commit + push 빈번하게

### DO NOT
- risk 확인 없이 production code 변경
- 전체 backend rewrite 시도 (8~12주 작업, plan 없이 진입 금지)
- HTP raw transfer path (LISWAP-5 falsified) 재시도
- raw OpenCL ↔ HTP zero-copy 재시도 (R-Y opt 1, 2 검증 완료, 막힘)
- /compact 전 production code 수정

---

**End of Handoff**

이 문서는 self-contained — /compact 후 새 세션에서 본 문서만 읽으면 risk 확인 + 마이그레이션 plan을 즉시 시작 가능.
