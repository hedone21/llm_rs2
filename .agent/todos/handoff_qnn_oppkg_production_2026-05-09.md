# Handoff: QNN-GPU OpPackage Production Migration (다음 세션)

**작성일**: 2026-05-09
**기준 commit**: `8477087` (origin/master push 완료)
**Device**: Galaxy S25 (R3CY408S4HN, Adreno 830 + Hexagon V79)
**SDK**: QNN 2.33 (`third_party/qnn_sdk_2.33/`, gitignored)

---

## 0. 한 줄 요약

Phase R 완료 (모든 risk 검증). **OpPackage path = GREEN**. 다음 세션은 `plan_qnn_oppkg_migration_2026-05-09.md` 기준 **M1 (OpPackage crate 정식화)** 진입. Architect → Implementer → Tester 파이프라인으로 진행.

---

## 1. 현재 상태

### 1.1 commit (origin/master 동기화)
```
8477087 docs: QNN OpPackage migration plan (5-phase, 9~14 weeks)
f864262 feat(qnn): Phase R Wave 3 — KV dynamic + mixed-op layer PoCs
f8acc05 docs(qnn): Phase R 종합 결과 — OpPackage + phase-aware swap GREEN
459ad15 feat(qnn): Phase R Wave 2 — OpPackage path + async swap microbenches
81a0d90 docs(qnn): Phase R Wave 1 — kernel 매핑 + license + runtime gate (GREEN)
c469387 docs: QNN-GPU migration risk assessment plan (Phase R)
```

### 1.2 working tree 상태
- 미커밋: `engine/src/bin/microbench_htp_rpcmem_throughput.rs` (untracked, ignore)
- third_party/ (gitignored, 그대로 유지)

### 1.3 디바이스 상태
- S25 ADB 연결 (R3CY408S4HN)
- `/data/local/tmp/qnn/`: 7 SDK lib (libQnnHtp/Gpu/HtpV79Skel/Stub/Prepare/System/Cpu)
- `/data/local/tmp/`: 9개 microbench bin + `libqnn_oppkg_poc.so`

### 1.4 HTP 환경 이슈 (별도 트랙)
- HTP backend 사용 microbench 모두 segfault (reboot으로도 안 fix)
- `microbench_htp_correctness`, `microbench_htp_qnngpu_share` 등
- 본 마이그레이션 path는 HTP 미사용이라 무관, 별도 디버깅 필요

---

## 2. Phase R 검증 결과 (전체)

### 2.1 Verdict 표

| Risk | Verdict | 근거 |
|------|---------|------|
| R-A1 kernel 매핑 | GREEN | 17 핵심 op 모두 표현 가능 (14 prebuilt + 3 composition) |
| R-A3 KV dynamic shape | YELLOW | graphFinalize 30~32ms (TBT 28ms 초과). max-padded 전략 필수 |
| R-A4 Q4_0 호환 | GREEN (조건부) | SDK API (`Qnn_BlockEncoding_t`) + Fallback (OpPackage wrap) |
| R-B1 raw QNN MatMul forward | RED → reframed | single-op artifact, multi-op graph는 GREEN |
| R-F1 SDK license | GREEN | device push 모델 |
| R-F2 runtime 호환 | GREEN | SDK 2.33 lib push로 vendor 2.20 우회 |
| **OpPackage chain amortize** | **GREEN** | N=16 이상에서 raw OpenCL 12% 능가 |
| **OpPackage 정확성** | **GREEN** | bit-identical (max_abs_err = 0) |
| **R-Y1 mixed-op layer** | **GREEN** | mixed 0.035 ms/op vs homogeneous 0.049 ms/op |
| **Async swap (phase-aware)** | **GREEN** | cache-fit phase에 1.04× of max (near-perfect parallel) |
| cl_mem 외부 공유 (graph 외) | RED | 모든 path 막힘 → 모든 forward op이 graph 안에 있어야 |

### 2.2 결정적 fix 발견
**`backendApiVersion`**: `5.33.0`이 아니라 **`3.7.0`**.
- `backendGetApiVersion`이 보고하는 5.33은 backend builder ID
- OpPackage가 expect하는 GPU API Version은 `QnnGpuCommon.h` 매크로 `QNN_GPU_API_VERSION` = 3.7.0
- 이 fix로 `registerOpPackage` segfault → SUCCESS 전환

---

## 3. 산출물 (재현 가능)

### 3.1 Code (commit `459ad15`, `f864262`)

**Crate**:
- `crates/qnn_oppkg_poc/` — cdylib OpPackage (V1.4 + V2.0 interface)
  - `Cargo.toml`, `build.rs`, `src/lib.rs`
  - export symbol: `QnnOpPackage_InitInterface`
  - 등록 ops: `CustomAdd`, `CustomMatMul` (mul_mv_f16_f32 wrap)

**Microbenches** (`engine/src/bin/`, all `--features qnn,opencl`):
- `microbench_qnngpu_matmul_tbt.rs` — R-B1 raw QNN MatMul (4 dim)
- `microbench_qnngpu_clmem_share.rs` — cl_mem 직접 공유 시도 (failed)
- `microbench_qnngpu_htp_concurrent.rs` — HTP+QNN-GPU concurrent (env 이슈)
- `microbench_qnn_oppkg_test.rs` — OpPackage 1024 elem ElementWiseAdd PoC
- `microbench_oppkg_gemv_vs_baseline.rs` — single GEMV wrap 비교
- `microbench_oppkg_chain_amortize.rs` — chain N=1/4/16/64 amortize
- `microbench_oppkg_async_swap.rs` — phase-aware swap (CLI args: dim chain memcpy_mb)
- `microbench_oppkg_dynamic_shape.rs` — graphFinalize cost vs K
- `microbench_oppkg_mixed_layer.rs` — homogeneous vs mixed graph

### 3.2 Reports
- `papers/eurosys2027/_workspace/experiment/qnn_kernel_mapping.md` (Wave 1 R-A1)
- `papers/eurosys2027/_workspace/experiment/qnn_risk_wave1_gate.md` (Wave 1 종합)
- `papers/eurosys2027/_workspace/experiment/qnn_phase_r_summary.md` (Phase R 종합)

### 3.3 Plans
- `.agent/todos/plan_qnn_gpu_risk_assessment_2026-05-09.md` (Phase R plan, 완료)
- `.agent/todos/plan_qnn_oppkg_migration_2026-05-09.md` (Migration plan, 진입 대기)
- `.agent/todos/handoff_qnn_oppkg_production_2026-05-09.md` (본 문서)

---

## 4. 핵심 측정 데이터 (paper용)

### 4.1 R-B1 raw QNN MatMul (RED, single-op artifact)

| Dim K×N | Baseline OpenCL | QNN MatMul | Ratio |
|---------|-----------------|------------|-------|
| 1536×1536 | 0.198 ms | 0.220 ms | 1.110× |
| 1536×256 | 0.132 ms | 0.204 ms | 1.549× |
| 1536×8960 | 0.607 ms | 0.643 ms | 1.060× |
| 8960×1536 | 0.548 ms | 1.373 ms | 2.504× |
| Layer 가중평균 | 2.422 ms | 3.507 ms | **1.448×** |

### 4.2 OpPackage chain amortize (GREEN turning point)

| N_op | raw OpenCL | OpPackage | Ratio | per-op (oppkg) |
|------|-----------|-----------|-------|---------------|
| 1 | 0.147 ms | 0.214 ms | 1.453× | 0.214 |
| 4 | 0.276 ms | 0.314 ms | 1.136× | 0.079 |
| **16** | 0.948 ms | **0.829 ms** | **0.874×** | **0.052** |
| **64** | 3.177 ms | **2.813 ms** | **0.885×** | **0.044** |

### 4.3 Phase-aware async swap (GREEN)

| Scenario | C1 graph | C2 memcpy | C3 concurrent | C3/max | C3/(C1+C2) |
|----------|----------|-----------|----------------|--------|------------|
| Large matmul (DDR-heavy) | 1.124 ms | 1.182 ms | 1.431 ms | 1.210× | 0.620 |
| **Cache-fit (256×256)** | 0.909 ms | 0.457 ms | **0.946 ms** | **1.040×** | 0.692 |
| 1MB chunk (small chunk) | 1.085 ms | 0.055 ms | 1.455 ms | 1.341× | 1.276 |

### 4.4 Mixed-op layer (R-Y1, GREEN)

| Config | num_ops | exec mean | per-op |
|--------|---------|-----------|--------|
| Homogeneous (12 mm) | 12 | 0.590 ms | 0.049 ms |
| **Mixed (6 mm + 6 add)** | 12 | **0.420 ms** | **0.035 ms** |
| Homogeneous (16 mm) | 16 | 0.724 ms | 0.045 ms |

### 4.5 R-A3 graphFinalize (YELLOW)

| K | finalize (ms) |
|---|---------------|
| 128 | 32.7 |
| 512 | 30.7 |
| 1024 | 30.9 |
| 2048 | 31.0 |
| 4096 | 31.3 |

→ TBT (28 ms) 초과. max-padded fixed shape 전략 필수.

---

## 5. 다음 세션 시작 절차

### 5.1 환경 확인
```bash
cd /home/go/Workspace/llm_rs2
git log --oneline -5            # HEAD = 8477087
git status                       # working tree clean (third_party/ 무시)
adb devices                      # R3CY408S4HN 연결 확인
ls third_party/qnn_sdk_2.33/include/QNN/  # SDK 헤더 존재
```

### 5.2 Plan 읽기
```bash
cat .agent/todos/plan_qnn_oppkg_migration_2026-05-09.md
cat papers/eurosys2027/_workspace/experiment/qnn_phase_r_summary.md
```

### 5.3 작업 시작 흐름

**M1 (OpPackage crate 정식화, 1~2주)**:

1. **Architect** (먼저)
   - `crates/qnn_oppkg/` (poc → production rename) 구조 설계
   - Op registration abstraction (각 .cl kernel을 op_type으로 등록하는 helper)
   - args layout schema (TENSOR/DATA/LOCAL kind, kernel별)
   - createOpImpl dispatcher (op_type → impl 매핑)
   - Spec 추가 필요 시 `spec/` 갱신
   - 참조: 현 `crates/qnn_oppkg_poc/src/lib.rs` (PoC가 이미 V1.4/V2.0 구현)

2. **Implementer (senior)** (Architect 후)
   - crate rename + 구조 적용
   - 5 핵심 op wrap: MatMulF16F32, RmsNorm, Softmax, SiluMul, Add
   - 각 op host vs device 정확성 unit test
   - sanity check (fmt + clippy + test)

3. **Tester** (Implementer 후)
   - 각 op 정확성 검증 (host calc vs device output)
   - 5 op 모두 max_abs_err < 1e-4
   - Pass-gate 충족 확인

### 5.4 M1 Pass-gate
- 5 op 모두 정확성 max_abs_err < 1e-4 (F32)
- 빌드 + push + register OK
- production code 변경 0 lines (microbench-only)

---

## 6. 주의사항 (Do / Don't)

### DO
- PoC crate (`qnn_oppkg_poc`)를 production rename할 때 commit history 보존
- `backendApiVersion = 3.7.0` 유지 (이게 핵심 fix)
- 모든 Box::leak으로 'static lifetime 보장 (PoC 패턴 유지)
- Spec ID 추가 시 `tests/spec/` 테스트 작성 필수 (CLAUDE.md 규칙)
- 매 phase commit + push (작업 손실 방지)

### DO NOT
- raw QNN MatMul 마이그레이션 시도 (R-B1 RED 확정, OpPackage path만)
- HTP backend 사용 (현재 unstable + 별도 트랙)
- cl_mem 외부 공유 시도 (모든 path 막힘 검증됨)
- production OpenCL backend 즉시 변경 (M3 진입 후 default off로 도입)

---

## 7. 다음 단계 외부 참조

### Code 진입점 (production 변경 필요)
- `engine/src/backend/mod.rs` (Backend trait + enum)
- `engine/src/backend/opencl/mod.rs` (현 OpenCL backend)
- `engine/src/models/llama/llama_model.rs` (forward dispatch)
- `engine/src/layers/llama_layer.rs` (1 layer forward, M2 graph builder 참고)
- `engine/kernels/*.cl` (wrap 대상)

### SDK 헤더 (참조 필수)
- `third_party/qnn_sdk_2.33/include/QNN/GPU/QnnGpuOpPackage.h` (`QnnGpu_Kernel_t`)
- `third_party/qnn_sdk_2.33/include/QNN/GPU/QnnGpuCommon.h` (`QNN_GPU_API_VERSION 3.7.0`)
- `third_party/qnn_sdk_2.33/include/QNN/QnnOpPackage.h` (V1.4/V2.0 interface)
- `third_party/qnn_sdk_2.33/include/QNN/QnnTypes.h` (BlockEncoding for Q4_0)

### 외부 reference
- llama.cpp QNN backend WIP PR: https://github.com/ggml-org/llama.cpp/pull/12063
- ONNX Runtime QNN EP: https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/qnn
- HeteroLLM (paper baseline): https://arxiv.org/abs/2501.14794

---

## 8. 잠재 risk (M1~M5 진행 시 발견될 수 있음)

### 알려진 risk (Phase R에서 부분 검증)
- KV cache max-padded → memory cost 증가 (16 layer × max_seq=2048 × kv_heads × head_dim × 2 byte)
- Q4_0 GPU backend 실측 성능 (헤더 분석만, 실제 kernel은 Phase M2에서)
- flash_attn OpPackage wrap 정확성 (online softmax 패턴)

### 잠재 risk (미검증, 진입 후 처리)
- Manager/Resilience IPC 호환 (D2O signal, eviction 등)
- mid-decode swap 시 KV cache state 일치
- multi-token generation full-graph vs layer-by-layer trade-off
- production app 패키징 (SDK lib deployment)

---

## 9. Phase R에서 확정된 architecture 결정

1. **OpPackage 안에 모든 forward op wrap** (외부 cl_mem 공유 막힘)
2. **layer-level graph** (full-model graph는 너무 큰 build cost; layer-level이 sweet spot)
3. **max-padded KV cache** (graph re-finalize 비용 회피)
4. **phase-aware async swap** (cache-fit phase에 chunk swap)
5. **chunk size 4-8 MB** (1 MB는 thread overhead로 fail)
6. **Backend trait + qnn_oppkg backend** (existing OpenCL backend 보존)
7. **default off + opt-in flag** (`--backend qnn_oppkg`)

---

**End of Handoff**

이 문서는 self-contained — /compact 후 새 세션에서 본 문서만 읽으면 M1 즉시 시작 가능.
