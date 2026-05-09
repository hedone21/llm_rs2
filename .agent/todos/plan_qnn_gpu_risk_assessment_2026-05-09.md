# Plan: QNN-GPU Migration — Phase R (Risk Assessment)

**작성일**: 2026-05-09
**기준 commit**: `cc34ce5` (origin/master)
**선행 handoff**: `.agent/todos/handoff_qnn_gpu_migration_2026-05-09.md`
**Device**: Galaxy S25 (Adreno 830 + Hexagon V79 HTP) — **단일 타겟**

---

## 0. 목표

OpenCL forward backend → QNN-GPU 마이그레이션이 **정확도 + 성능 무손실**로 가능한지 falsifiable 측정으로 결정.

- **GREEN**: 전체 forward QNN-GPU 표현 가능 + TBT ≤ baseline + 정확성 OK → 마이그레이션 plan 진입
- **YELLOW**: 부분 마이그레이션이 net 무손실 → 제한된 scope plan
- **RED**: 어떤 형태로도 정확도 또는 성능 손실 → paper 정리 + 마이그레이션 보류

**Production code 변경 0 lines** 유지. 모든 검증은 microbench + 코드 read-only 분석.

---

## 1. 결정 기준 (사용자 명시)

| 기준 | 통과 조건 |
|------|----------|
| 정확도 | QNN-GPU forward 출력 vs OpenCL baseline element-wise drift 무시 가능 (max abs <1e-2 FP16) |
| 성능 | QNN-GPU TBT ≤ OpenCL baseline TBT × 1.0 (margin 없음, 손실 시 RED) |
| Device scope | S25 only (Pixel/Jetson은 OpenCL backend 유지, R-F3 자동 결정) |

---

## 2. Wave 구조

```
Wave 1 (1~2일, 병렬, fail-fast gate)
   ├─ R-A1 kernel ↔ QNN op 매칭
   ├─ R-F1 SDK license
   └─ R-F2 runtime 호환

      ↓ GREEN

Wave 2 (2~3일, 병렬)
   ├─ R-A2 custom op 비용
   ├─ R-B1 GEMV vs MatMul TBT
   ├─ R-G1 단순 forward 정확성
   └─ R-C1/2/3 메모리 모델 (read-only)

      ↓ GREEN

Wave 3 (2~3일, 병렬)
   ├─ R-B3 e2e 1 layer FFN TBT
   ├─ R-A3 KV cache dynamic shape
   ├─ R-A4 Q4_0 호환
   ├─ R-B2 flash_attn 대체
   └─ R-B4 graph build cost

      ↓ GREEN

Wave 4 (1일, 종합 결정)
   ├─ R-D scope/cost (사용자 토론)
   ├─ R-E 통합 영향도 (code analysis)
   ├─ R-G2 full-model accuracy
   └─ R-G3 mid-decode swap 호환성

총 6~9일, fail-fast 시 단축
```

---

## 3. Wave 1 — Fail-fast gate (1~2일, 병렬)

여기서 RED면 Phase R 즉시 종결, paper로.

### R-A1 kernel ↔ QNN op 매칭 (4시간)

**검증 방법**:
```bash
# 우리 kernel 11개
ls engine/kernels/*.cl

# QNN prebuilt op 검색
grep -E "QNN_OP_|^#define" third_party/qnn_sdk_2.33/include/QNN/QnnOpDef.h
```

산출물: `papers/eurosys2027/_workspace/experiment/qnn_kernel_mapping.md` 매핑 표.

**Pass/Fail 기준**:
- ≥ 9개 prebuilt 매칭 → **GREEN** (Wave 2 진입)
- 6~8개 매칭, 부재 op는 graph composition 가능 → **YELLOW** (Wave 2 진입, R-A2 강화)
- < 6개 매칭 또는 핵심 op (MatMul, RMSNorm) 부재 → **RED** (Phase R 종결)

### R-F1 SDK license (1시간)

**검증 방법**: `third_party/qnn_sdk_2.33/LICENSE.pdf` 또는 EULA 텍스트 검토.

**Pass/Fail**:
- 임베드 가능 (lib만 device push) → **GREEN**
- 사용 불가 / 재배포 일체 금지 → **RED**

### R-F2 runtime 버전 호환 (2시간)

**검증 방법**:
```bash
# device vendor api 확인
adb shell strings /vendor/lib64/libQnnHtp.so | grep -E "version|Version"
# vs SDK 2.33 (api 2.25.0)
```
PoC 단계에서 SDK 2.33 .so push로 우회 검증 완료. production scenario 영향만 정리.

**Pass/Fail**:
- SDK lib push 영구적으로 acceptable → **GREEN**
- SDK lib push 불가 (앱 sandbox 등) → **YELLOW** (vendor 2.20 기반 재검증 필요)

---

## 4. Wave 2 — Core feasibility (2~3일, 병렬)

Wave 1 GREEN 후 진입. 신규 microbench 4개 작성.

### R-A2 부재 op custom 비용 (1일, code/docs 분석)

**검증 방법**:
- Wave 1 R-A1 결과 부재 op 리스트
- QNN UDO (User-Defined Op) docs 검토: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/
- Hexagon SDK + HVX intrinsic 필요 여부 확인
- ORT QNN EP custom op 사례 (PR #23136) reference

**Pass/Fail**:
- UDO C++ binding 가능 + 작성 cost ≤ 1주/op → **GREEN**
- HVX intrinsic 필수 → **YELLOW** (cost 산정만)
- UDO API 미지원 op → **RED**

### R-B1 GEMV vs QNN-GPU MatMul TBT (1일)

**신규 microbench**: `engine/src/bin/microbench_qnngpu_matmul_tbt.rs`

**측정 조건**:
- 차원: production 사용 차원 (예: 2048×2048, 2048×5632 — Qwen2.5-1.5b FFN)
- precision: FP16 (production과 동일)
- baseline: 현 OpenCL `mul_mv_f16` kernel 동일 차원 동일 precision
- iter: n=30, σ/mean check
- wall-clock 측정 (`CL_QUEUE_PROFILING_ENABLE` 사용 안 함)

**Pass/Fail**:
- QNN-GPU TBT ≤ OpenCL baseline → **GREEN**
- 1.0~1.1× → **YELLOW** (Wave 3 e2e 결과 보고 결정)
- > 1.1× → **RED** (성능 손실 = RED)

### R-G1 단순 forward 정확성 (4시간)

**신규 microbench**: `engine/src/bin/microbench_qnngpu_correctness.rs`

**측정 조건**:
- FFN layer (RMSNorm + gate matmul + up matmul + SwiGLU + down matmul)
- baseline: OpenCL forward 동일 weight, 동일 input
- element-wise diff

**Pass/Fail**:
- max abs error < 1e-2 (FP16) → **GREEN**
- < 1e-1 → **YELLOW**
- 그 이상 → **RED**

### R-C1/2/3 메모리 모델 (read-only, 1일)

**검증 방법**: docs + 우리 code 분석.
- R-C1: QNN graph variable shape support → KV cache (HeadMajor, dynamic grow)
- R-C2: weight buffer (cl_mem 600MB) → QNN tensor mapping. 현 swap path 영향
- R-C3: workspace tensor (forward 중간 결과) graph 안 vs 별도

산출물: `papers/eurosys2027/_workspace/experiment/qnn_memory_model_review.md`

**Pass/Fail**:
- 표현 + 매핑 가능 → **GREEN**
- 부분 표현 (e.g., KV는 외부 buffer로) → **YELLOW**
- 표현 불가 → **RED**

---

## 5. Wave 3 — End-to-end + edge cases (2~3일, 병렬)

### R-B3 e2e 1 layer FFN TBT (1일)

**신규 microbench**: `engine/src/bin/microbench_qnngpu_ffn_layer.rs`
- Qwen2.5-1.5b 1 FFN layer 전체 (RMSNorm + 3 matmul + SwiGLU + residual)
- baseline: 동일 layer OpenCL
- 현 production decode TBT ~28ms/tok (16 layer 평균 1.75ms/layer)

**Pass/Fail**:
- QNN-GPU TBT/layer ≤ baseline → **GREEN**
- > baseline → **RED** (성능 손실)

### R-A3 KV cache dynamic shape (1일)

**검증**: QNN graph에 variable input + Concat op으로 KV cache update path 표현 가능한가.
- 우리 production: HeadMajor `[1, kv_heads, capacity, head_dim]`, prefill+decode 단계에서 pos slice 동적 변경
- QNN graph executor가 매 step 같은 shape 강요한다면 incremental update 표현 어려움

**Pass/Fail**:
- variable shape 또는 max-shape + slice 가능 → **GREEN**
- 매 token re-finalize 필요 (>1초) → **RED** (graph build cost 폭발)

### R-A4 Q4_0 호환 (1일)

**검증 방법**:
- QNN Dequantize prebuilt op 검색
- 우리 GGUF Q4_0 packing (32 elem/block, scale FP16) → QNN tensor quantization scheme 매핑
- 정확성 측정: `microbench_qnngpu_q4_dequant.rs` (필요 시)

**Pass/Fail**:
- 매핑 OK + 정확성 → **GREEN**
- 우리 dequant kernel을 UDO로 작성 가능 → **YELLOW**
- 표현 불가 → **RED**

### R-B2 flash_attn 대체 (1일, code/docs 분석)

**검증 방법**:
- QNN ScaledDotProductAttention prebuilt op 존재 여부
- 없다면 graph composition (matmul + softmax + matmul) 시도
- 우리 flash_attn_v75 (DK=128, online softmax)와의 numerical 차이 추정

**Pass/Fail**:
- prebuilt op 존재 + 우리 차원 지원 → **GREEN**
- composition 가능 + 성능 추정 OK → **YELLOW**
- 표현 불가 → **RED**

### R-B4 graph build/finalize cost (4시간)

**측정**: Wave 2 microbench의 graph build 시간 + production 모델 16 layer 추정.

**Pass/Fail**:
- 모델 로드 시 1회 build < 1초/layer → **GREEN**
- > 5초/layer → **RED** (production 로드 시간 영향 심각)

---

## 6. Wave 4 — Scope/integration 결정 (1일)

측정이 아니라 **결정** 영역. Wave 1~3 결과와 사용자 토론 결합.

### R-D scope/cost
Wave 1~3 종합 후:
- 모든 op GREEN → 전체 마이그레이션 8~12주 plan
- 부분 GREEN (예: FFN만, attention은 OpenCL) → 부분 마이그레이션 plan + backend switching cost 측정 필요

### R-E 통합 영향도 (code analysis)
- Manager/Resilience IPC (현 OpenCL backend 전제)
- D2O / KV eviction / pressure pipeline → backend abstraction 영향
- weight swap (현 ALLOC_HOST_PTR pool) → QNN tensor와 호환

산출물: 영향 모듈 list + 변경 추정 LOC.

### R-G2 full-model accuracy
- Qwen2.5-1.5b 32k token decode greedy → output token sequence diff vs OpenCL baseline
- 후순위 (Wave 3 R-G1 PASS 후 sanity check)

### R-G3 mid-decode swap 호환성
- production async swap (Phase 6.5) 시 backend 전환 시 KV cache state 일치
- S25 only이므로 현재 weight swap path가 QNN-GPU에서 동작해야 함
- swap path 자체가 마이그레이션 plan에 포함될 항목

---

## 7. 산출물

### Wave별 보고서
- `papers/eurosys2027/_workspace/experiment/qnn_risk_wave1_gate.md` (R-A1, R-F1, R-F2)
- `papers/eurosys2027/_workspace/experiment/qnn_risk_wave2_core.md` (R-A2, R-B1, R-G1, R-C)
- `papers/eurosys2027/_workspace/experiment/qnn_risk_wave3_e2e.md` (R-B3, R-A3, R-A4, R-B2, R-B4)
- `papers/eurosys2027/_workspace/experiment/qnn_risk_wave4_decision.md` (R-D, R-E, R-G2, R-G3)

### 신규 microbench (engine/src/bin/)
- `microbench_qnngpu_matmul_tbt.rs` (Wave 2 R-B1)
- `microbench_qnngpu_correctness.rs` (Wave 2 R-G1)
- `microbench_qnngpu_ffn_layer.rs` (Wave 3 R-B3)
- (선택) `microbench_qnngpu_q4_dequant.rs` (Wave 3 R-A4)
- (선택) `microbench_qnngpu_kv_dynamic.rs` (Wave 3 R-A3)

### 매핑/리뷰 문서
- `papers/eurosys2027/_workspace/experiment/qnn_kernel_mapping.md` (R-A1)
- `papers/eurosys2027/_workspace/experiment/qnn_memory_model_review.md` (R-C)

### 최종 결정 보고서
- `papers/eurosys2027/_workspace/experiment/qnn_migration_decision_2026-05-XX.md`
  - GREEN/YELLOW/RED + 다음 plan (마이그레이션 또는 paper 정리)

---

## 8. Critical Files

### 분석 대상 (read-only)
- `engine/kernels/*.cl` (R-A1)
- `engine/src/backend/opencl/mod.rs` (R-E)
- `engine/src/backend/mod.rs` (R-E)
- `engine/src/core/cache_manager.rs`, `engine/src/core/kv_cache.rs` (R-C, R-G3)
- `engine/src/models/llama/llama_model.rs` (R-B3)
- `third_party/qnn_sdk_2.33/include/QNN/QnnOpDef.h` (R-A1)

### 신규 생성 (Wave 2/3, microbench-only)
- `engine/src/bin/microbench_qnngpu_*.rs`

### Cargo features
- 기존 `qnn = ["libloading", "bindgen"]` 활용 (Cargo.toml 변경 불필요)

### Production 영향
- **0 lines 변경** (Phase R 동안). 변경은 마이그레이션 plan 진입 후.

---

## 9. 결정 트리

```
Wave 1 결과
├─ RED (R-A1 핵심 op 부재 또는 R-F1 license 불가)
│   └─ Phase R 종결 → paper 정리
│       paper: "QNN-GPU 마이그레이션 NOT feasible — root cause"
│
├─ YELLOW
│   └─ Wave 2 진입 (R-A2에 강한 추가 검증)
│
└─ GREEN
    └─ Wave 2 진입

Wave 2 결과
├─ R-B1 또는 R-G1 RED
│   └─ Phase R 종결 (성능/정확도 손실)
│
├─ YELLOW (R-B1 1.0~1.1×, R-G1 borderline)
│   └─ Wave 3 e2e 결과 보고 최종 결정
│
└─ GREEN
    └─ Wave 3 진입

Wave 3 결과
├─ R-B3 RED (e2e 성능 손실)
│   └─ Phase R 종결 (paper)
│
├─ R-A3/A4/B2 일부 RED
│   └─ Wave 4에서 부분 마이그레이션 scope 검토 (YELLOW)
│
└─ 모두 GREEN
    └─ Wave 4 + 마이그레이션 plan 작성

Wave 4 결과
├─ 마이그레이션 진행 결정 (GREEN/YELLOW)
│   └─ 별도 plan_qnn_gpu_migration_<scope>_<date>.md 작성
│
└─ 보류 결정 (RED 또는 비용 과다)
    └─ paper 정리
```

---

## 10. Verification

### 각 Wave 완료 시
1. **Sanity** (microbench bin 추가 시):
   ```bash
   cargo fmt --all
   cargo clippy --features qnn,opencl --bin microbench_qnngpu_* -- -D warnings
   ```
2. **회귀 확인**:
   ```bash
   cargo test -p llm_rs2  # production code 변경 없음 → 항상 통과
   ```
3. **device 측정 commit + push**:
   - microbench bin
   - measurement report (`papers/.../qnn_risk_wave*.md`)

### Phase R 종료 조건
- Wave 4 결과 + 최종 결정 보고서 작성
- GREEN → 마이그레이션 plan 별도 작성 후 진입
- RED → paper 정리, Phase R 마감

---

## 11. Out of Scope

- production code 변경 (마이그레이션 plan 진입 후)
- multi-device backend (S25 only)
- 전체 backend rewrite implementation (Phase R는 검증만)
- Pixel/Jetson에서 QNN-GPU 동작 확인 (해당 device는 OpenCL backend 유지)
- LISWAP-4 / LISWAP-5 / Vulkan PoC 재시도 (모두 종결)

---

## 12. Risk → Pass/Fail 요약

| ID | Wave | 핵심 측정 | GREEN | YELLOW | RED |
|----|------|----------|-------|--------|-----|
| R-A1 | 1 | kernel 매칭 | ≥9 | 6~8 | <6 또는 핵심 부재 |
| R-A2 | 2 | custom op | UDO ≤1주/op | HVX 필요 | API 미지원 |
| R-A3 | 3 | dynamic shape | OK | max-shape+slice | re-finalize 필요 |
| R-A4 | 3 | Q4_0 | 매핑 OK | UDO 필요 | 표현 불가 |
| R-B1 | 2 | MatMul TBT | ≤baseline | ≤1.1× | >1.1× |
| R-B2 | 3 | flash_attn | prebuilt OK | composition | 표현 불가 |
| R-B3 | 3 | FFN layer TBT | ≤baseline | — | >baseline |
| R-B4 | 3 | graph build | <1s/layer | <5s/layer | >5s/layer |
| R-C1 | 2 | KV shape | OK | 외부 buffer | 표현 불가 |
| R-C2 | 2 | weight tensor | OK | partial | 불가 |
| R-C3 | 2 | workspace | graph 안 | 별도 buffer | 불가 |
| R-D | 4 | scope | 전체 | 부분 | 보류 |
| R-E | 4 | 통합 | 영향 한정 | 부분 rework | 전면 rework |
| R-F1 | 1 | license | 임베드 OK | lib only | 사용 불가 |
| R-F2 | 1 | runtime | SDK push OK | vendor만 | 부적합 |
| R-G1 | 2 | 단순 정확성 | <1e-2 | <1e-1 | drift |
| R-G2 | 4 | full-model | seq 일치 | <1% diff | drift |
| R-G3 | 4 | swap 호환 | OK | 부분 | 불가 |

(R-F3 multi-device는 S25 only로 자동 결정 → 제거)

---

**End of Plan**

다음 액션: Wave 1 시작 (R-A1, R-F1, R-F2 병렬). 첫 산출물은 `qnn_kernel_mapping.md`.
