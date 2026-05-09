# Handoff: QNN-GPU OpPackage M3 Production Wire-up

**작성일**: 2026-05-10
**기준 commit**: 본 핸드오프 직전 commit (M2 산출물 push 완료)
**Device**: Galaxy S25 (R3CY408S4HN, Adreno 830 + Hexagon V79)
**SDK**: QNN 2.33 (`third_party/qnn_sdk_2.33/`, gitignored)

---

## 0. 한 줄 요약

M2 (layer-level graph) 완료. 5 op + 14-node single-layer graph **정확성 GREEN (bit-exact)**. M2.I TBT는 **microbench 측정 1.74× / logcat 직접 측정 1.24× YELLOW**. 다음 세션은 **M3 production engine wire-up**으로 진짜 verdict 확정 + production 통합 결정.

---

## 1. 현재 상태

### 1.1 commit 상태
- HEAD: M2 산출물 push 완료 (origin/master 동기화 — 본 handoff 직전 commit)
- 이전: `d930801` (M1 OpPackage production crate)

### 1.2 working tree
- third_party/ (gitignored, SDK)
- microbench bin 다수 존재 (M2.B~M2.I 검증분)

### 1.3 디바이스 상태
- S25 ADB 연결 확인 필요
- `/data/local/tmp/`: `libqnn_oppkg.so` (M2 변경분), `qnn/` SDK libs 7개

---

## 2. M2 결과 종합

### 2.1 단계별 verdict

| 단계 | 결과 | 측정값 |
|------|------|--------|
| M2.B RoPE OOP | GREEN | 3 케이스 max_abs_err = 0 |
| M2.C DeqQ40 | YELLOW | dual-output abstraction 한계 (M3 영역) |
| M2.D MatMulQ40F32 | GREEN | Qwen hot path 2 케이스 max_abs_err = 0 |
| M2.E KvScatter (multi-output abstraction) | GREEN | k_dst+v_dst max_abs_err = 0 |
| M2.F FlashAttn | GREEN | 3 n_kv 케이스 max_abs_err 9.3e-5~4e-7 |
| M2.G SiluMul OOP | GREEN | 3 케이스 max_abs_err = 0 |
| **M2.H Layer graph 정확성** | **GREEN** | **14-node bit-exact (max_abs_err = 0)** |
| **M2.I TBT** | **YELLOW** | **logcat 1.24× (microbench 1.74× artifact)** |

### 2.2 핵심 발견 (paper evidence 가치 큼)

#### F1. SDK alias semantics 측정 (Phase R 가설 정밀화)
- **가설**: SDK는 graph 안 마지막 op output만 host APP_READ로 sync
- **검증**: 헤더(`QnnTensor.h:124`) "Graph tensors cannot be of type APP_READWRITE. R/W tensors connect multiple graphs" 직접 명시
- **6 sub-graph bisect**로 stage별 drift 측정 → vcache PASS/q_rope=0/kcache 정확성 분리 확인

#### F2. In-place vs OOP OpPackage 패턴 (chain composition)
- **B4 prebuilt ElementWiseAdd × 2**: GREEN (SDK 일반 chain composition 정상)
- **B1 RoPE in-place chain**: RED (in-place op이 마지막 노드면 host buffer 0)
- **B1' OpPackage OOP chain (CustomAdd × 2)**: GREEN (Phase R Wave 2 0.874× 재현 → 0.92×)
- **결론**: in-place OpPackage 패턴이 SDK chain composition과 호환 안 됨. **OOP variant 도입이 본질 fix**

#### F3. FlashAttn sinks mem-object size 결함
- 결정적 fix 발견: `sinks` mem_object `flat_dims: vec![1]` → `vec![n_head]`
- root cause: kernel은 `sinks_ptr[head_idx]`로 indexing, OpPackage는 `mem_null()` binding 불가 → zero-init buffer 들어감 → `m_i = 0` + `l_final += exp(0)` → **ratio ~0.5**
- 한 줄 fix로 14-node layer graph 정확성 회복 (max_abs_err 2.228 → 0)

#### F4. TBT 측정 — microbench artifact vs 진짜 SDK overhead
- **logcat warm steady-state graphExecute: 1.3 ms/layer**
- 28 layer × 1.3 = **36.4 ms/tok 진짜 cost**
- baseline production OpenCL: 29.43 ms/tok
- **진짜 ratio: 1.24× (YELLOW)**
- microbench 측정 51.16 ms/tok = 36.4 + 14.76 ms harness overhead (Instant::now/elapsed, token loop simulation 등)
- **CPU fallback / GPU throttle 없음** (logcat 직접 확인)

#### F5. Phase R Wave 2 결과와 일관성
- Phase R N=16 chain amortize: 0.874× (12% faster)
- 본 측정 chain-only vs QNN: 0.92× ✓ 재현
- Phase R 가설 valid

---

## 3. 산출물 (재현 가능)

### 3.1 Code

#### Crate `crates/qnn_oppkg/`
- `src/args.rs` — `ArgSpec::OutputTensorAliased` 신규, `OpImplLayout::output_claims` 신규, `OutputClaimSpec` 신규
- `src/op_impl.rs` — multi-output dispatch (`output_claims`), `OutputTensorAliased` 분기
- `src/ops/`
  - `rope.rs` (M2.B): kernel_rope_simple_oop 사용, OOP 7-arg
  - `deq_q40.rs` (M2.C): dual-output, YELLOW
  - `matmul_q40_f32.rs` (M2.D): Q4_0 GEMV, 15-arg, rank-flexible
  - `kv_scatter.rs` (M2.E): multi-output (Tier 1 InOutTensor + OutputTensor → multi-output)
  - `flash_attn.rs` (M2.F): 44-arg, sinks mem-object n_head 크기 수정
  - `silu_mul.rs` (M2.G): kernel_silu_mul_simple_oop 사용
  - `matmul_f16_f32.rs`: rank-flexible 갱신
  - `add.rs`, `rms_norm.rs`, `softmax.rs`: output_claims 추가 (legacy 동작)
- `src/graph/{mod.rs, layer.rs}` — Qwen 1 layer DAG metadata (LAYER_NODE_COUNT=14)
- `tests/ops_metadata.rs` — host unit test 50+ (M2 신규 ~25)

#### Microbenches (`engine/src/bin/`)
- `microbench_qnn_oppkg_rope_correct` (M2.B)
- `microbench_qnn_oppkg_deq_q40_correct` (M2.C)
- `microbench_qnn_oppkg_matmul_q40_correct` (M2.D)
- `microbench_qnn_oppkg_kv_scatter_correct` (M2.E)
- `microbench_qnn_oppkg_flash_attn_correct` (M2.F)
- `microbench_qnn_oppkg_flash_attn_qwen` (M2.F replay)
- `microbench_qnn_oppkg_chain5_correct` (M2.G)
- `microbench_qnn_qwen_layer` (M2.H 14-node)
- `microbench_qnn_qwen_layer_2graph` (M2.H 6차 시도, archival)
- `microbench_qnn_qwen_layer_bisect` (M2.H 4차 bisect)
- `microbench_qnn_qwen_layer_bisect2` (M2.H 8차 sub-graph bisect)
- `microbench_qnn_oppkg_oop_chain_correct` (B1' OOP chain GREEN)
- `microbench_qnn_oppkg_reduced3_correct` (B1 reduced 3-op)
- `microbench_qnn_prebuilt_chain_correct` (B4 prebuilt GREEN)
- `microbench_qnn_28layer_tbt` (M2.I 28-layer simulation)

#### Production .cl (engine/kernels/simple_ops.cl)
- 신규 OOP variant 추가 (in-place 한계 우회):
  - `kernel_rope_simple_oop` — 별도 x_in/x_out
  - `kernel_silu_mul_simple_oop` — 별도 x_in/y_in/out

### 3.2 Spec/Arch (Architect M2.A)
- `spec/30-engine.md` 부록 B (ENG-QNN-101~150 + ENG-QNN-C10~C14)
- `spec/41-invariants.md` §3.23 (INV-156~165)
- `spec/COVERAGE.md` 갱신
- `arch/30-engine.md` §17 (Layer graph 컴포넌트 다이어그램)

### 3.3 Plans + Handoff
- `.agent/todos/feat_qnn_oppkg_m2.md` (M2 plan 본문, PM)
- `.agent/todos/handoff_qnn_oppkg_m3_production_wireup_20260510.md` (본 문서)

---

## 4. 핵심 측정 데이터

### 4.1 Standalone op (max_abs_err = 0 if not noted)

| Op | Cases | max_abs_err |
|----|-------|-------------|
| Add | 3 (N ∈ {64, 1024, 16384}) | 0 |
| MatMulF16F32 | 3 (M, N, K) | 0 |
| RmsNorm | 3 (rows, dim) | 0 |
| Softmax | 3 (rows, cols) | 0 |
| SiluMul OOP | 3 (rows, dim) | 0 |
| RoPE OOP | 3 (start_pos) | 0 |
| MatMulQ40F32 | 2 (Qwen QKV/FFN) | 0 |
| KvScatter (multi-output) | 1 | 0 |
| FlashAttn | 3 (n_kv ∈ {128, 1024, 2048}) | 9.3e-5~4e-7 |
| FlashAttn (Qwen n_kv=1, sinks fix) | 1 | 0 |

### 4.2 Chain composition

| Chain | Pattern | Result |
|-------|---------|--------|
| 4-op (RmsNorm→MatMul→Add→Softmax) | OOP only | bit-exact (M1.9) |
| 5-op chain (with SiluMul OutputTensorAliased) | mixed | RED 1.34e-1 (M2.G chain endpoint 한계) |
| **14-node layer (sinks fix 후)** | **OOP variant + multi-output** | **bit-exact (M2.H)** |

### 4.3 TBT 측정

| 측정 | 값 |
|------|-----|
| Production OpenCL baseline (`generate -n 32`) | 29.43 ms/tok |
| QNN graphExecute warm steady-state (logcat) | 1.3 ms/layer |
| QNN 28-layer logcat sum | 36.4 ms/tok (1.24× YELLOW) |
| QNN microbench measured 28-layer × 32 token | 51.16 ms/tok (1.74×, microbench wrap overhead 14.76 ms 포함) |
| graphFinalize (1회성 app load) | 1191 ms |

### 4.4 6 sub-graph bisect (M2.H 디버그)

| # | endpoint | max_abs_err |
|---|----------|-------------|
| 1 | RmsNorm + Q + RoPE Q | 0 |
| 2 | KvScatter k+v | 0 |
| 3 | FlashAttn (sinks fix 전) | 1.56 (ratio 0.507) |
| 3 (after fix) | FlashAttn (sinks fix) | 0 |
| 4 | O proj + Add | 0 |
| 5 | Norm + ffn + SiluMul | 0 |
| 6 | Down + Add | 0 |

---

## 5. 다음 세션 시작 절차 (M3)

### 5.1 환경 확인
```bash
cd /home/go/Workspace/llm_rs2
git log --oneline -10           # M2 커밋 확인
git status                       # working tree clean
adb devices                      # R3CY408S4HN 연결
ls third_party/qnn_sdk_2.33/include/QNN/   # SDK 헤더
```

### 5.2 M3 plan 읽기
```bash
cat .agent/todos/handoff_qnn_oppkg_m3_production_wireup_20260510.md  # 본 문서
cat .agent/todos/plan_qnn_oppkg_migration_2026-05-09.md §M3
cat .agent/todos/feat_qnn_oppkg_m2.md
```

### 5.3 M3 작업 흐름

**M3 (Backend trait + production wire-up, 3~4주)**:

1. **Architect** (먼저)
   - `engine/src/backend/qnn_oppkg/` 디렉토리 신규 설계
   - `Backend` trait 구현 — CPU/OpenCL/CUDA와 동등
   - KV cache 관리 (graph 외부 max-padded buffer, rpcmem alloc)
   - weight pool (16 layer × ~90 MB Q4_0 = ~1.4 GB)
   - Spec 추가: ENG-QNN-200~ (production integration)

2. **Senior Implementer** (Architect 후)
   - `engine/src/backend/qnn_oppkg/{mod,backend,layer_graph,kv_cache,weight_pool}.rs`
   - `--backend qnn_oppkg` flag 추가 (default off, opt-in)
   - existing OpenCL backend 그대로 유지 (regression 방지)

3. **Tester**
   - 정확성: token sequence vs OpenCL backend 100% 일치
   - TBT: 진짜 ratio 측정 (microbench wrap overhead 제거)
   - VmRSS: ≤ baseline × 1.10

### 5.4 M3 Pass-gate (핵심 결정 게이트)

- 정확성 100% (token sequence)
- **TBT ratio ≤ 1.10×** (GREEN) → production 통합 결정
- 1.10~1.20× (YELLOW) → optimization sub-task 1주 timebox
- > 1.20× (RED) → scope 재정의 또는 OpPackage path 폐기

---

## 6. 주의사항 (Do / Don't)

### DO
- M2.H sinks fix (`flat_dims: vec![n_head]`) 유지
- OOP variant kernel (rope_simple_oop, silu_mul_simple_oop) 사용
- multi-output abstraction 사용 (KvScatter)
- M3 wire-up은 default off + opt-in flag (`--backend qnn_oppkg`)
- production engine 변경 시 OpenCL backend regression 검증 필수
- 매 phase commit + push (작업 손실 방지)

### DO NOT
- in-place OpPackage 패턴 다시 사용 (chain composition 한계)
- `OutputTensorAliased` chain endpoint에 두지 말 것 (1.34e-1 RED)
- raw QNN MatMul (R-B1 RED, OpPackage path만)
- HTP backend 사용 (현재 unstable)
- DeqQ40 dual-output 우회 시도 자제 (Tier 1 InOutTensor + OutputTensor 패턴이 그래도 한계 — KvScatter처럼 multi-output abstraction 적용 가능하나 본질 한계)
- microbench wrap overhead 제외 측정 위해 production wire-up 필수 (1.74× → 1.24× → 진짜 GREEN/YELLOW 결정)

---

## 7. 외부 참조

### Code 진입점 (M3 production 변경 영역)
- `engine/src/backend/mod.rs` (Backend trait + enum 확장)
- `engine/src/backend/opencl/mod.rs` (현 OpenCL backend, 보존)
- `engine/src/models/llama/llama_model.rs` (forward dispatch)
- `engine/src/layers/llama_layer.rs` (1 layer forward, M2.H builder 참고)

### SDK 헤더 (참조 필수)
- `third_party/qnn_sdk_2.33/include/QNN/GPU/QnnGpuOpPackage.h` (`QnnGpu_Kernel_t`, `QnnGpu_KernelArgType_t`)
- `third_party/qnn_sdk_2.33/include/QNN/GPU/QnnGpuCommon.h` (`QNN_GPU_API_VERSION 3.7.0`)
- `third_party/qnn_sdk_2.33/include/QNN/GPU/QnnGpuGraph.h` (`QnnGpuGraph_CustomConfig_t`, `disableMemoryOptimizations`)
- `third_party/qnn_sdk_2.33/include/QNN/QnnTensor.h:124` (APP_READWRITE 금지 — 핵심)
- `third_party/qnn_sdk_2.33/include/QNN/QnnGraph.h` (graphCreate/Finalize/Execute)

---

## 8. 잠재 risk (M3 진행 시)

### 검증된 사실 (mini-PoC)
- **Weight MUTABLE 확정** (2026-05-10): memRegister 1회 후 host에서 rpcmem에 새 weight memcpy → 다음 execute가 새 weight 즉시 사용. SDK 캐싱/freeze 없음. → **M4 async swap path fundamental risk PASS**.

### 알려진 risk
- KV cache max-padded `[1, kv_heads, 2048, head_dim]` F16: 16 layer × 1 MB = 16 MB
- Q4_0 weight 16 layer = ~1.4 GB (rpcmem alloc 필요)
- multi-token generation full-graph vs layer-by-layer trade-off
- Manager/Resilience IPC 호환 (D2O signal, eviction 등 — M5 영역)
- M4 stress test 미검증: layer scale (수백 MB) chunk swap, graphExecute 진행 중 동시 chunk write race

### 잠재 risk
- production wire-up 시 OpenCL backend regression
- KV cache + weight pool memory pressure
- `--backend qnn_oppkg` 활성화 시 thermal/throttle (M2.I 28-layer 측정에선 안 보였음)
- production app 패키징 (SDK lib deployment, app load time 1.2s amortize)

---

## 9. M2에서 확정된 결정 (M3에 그대로 적용)

1. **OpPackage 안 모든 forward op wrap** (외부 cl_mem 공유 막힘 — Phase R)
2. **layer-level graph** (full-model graph는 build cost 너무 큼 — M2.H 검증, finalize 1.2s)
3. **max-padded KV cache** (graph re-finalize 비용 회피)
4. **OOP variant kernel** (in-place + chain composition 한계 우회)
5. **multi-output abstraction** (KvScatter 패턴)
6. **Backend trait + qnn_oppkg backend** (existing OpenCL backend 보존)
7. **default off + opt-in flag** (`--backend qnn_oppkg`)

---

## 10. M2 알려진 한계 (M3/M4/M5 backlog)

1. **DeqQ40 dual-output abstraction 한계**: M2.C YELLOW. M3에서 production이 inline dequant (MatMulQ40 안)이라 우선순위 낮음.
2. **SiluMul `OutputTensorAliased` chain endpoint 한계**: M2.G에서 발견. 본 한계는 layer graph endpoint가 아닌 intermediate에선 OOP variant로 우회 — M2.H에서 GREEN 검증.
3. **graphFinalize 1.2s**: app load 1회성 cost. M3 cache amortize 검증 필요.
4. **microbench harness overhead**: M2.I 측정에서 14.76 ms 식별. M3 production wire-up이 정확한 측정 path.
5. **KvScatter Tier 1 InOutTensor**: standalone GREEN. layer graph에선 multi-output 패턴으로 GREEN. M3에서 정확성 재검증.
6. **CustomMatMulF16F32 사용 안 됨**: Qwen2.5-1.5B는 Q4_0 weight 사용. F16 weight 모델 (예: Llama 3.2-1B F16) 처리 시 검증 필요.

---

**End of Handoff**

이 문서는 self-contained — /compact 후 새 세션에서 본 문서만 읽으면 M3 즉시 시작 가능.
