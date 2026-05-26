# Handoff: μ-Q1 — QNN HTP matmul microbench PoC 진입

**작성**: 2026-05-26
**HEAD**: `eca09d55 fix(arch): WeightSwapHandler dormant 마킹 — 잘못된 wire edge 제거` (직전 swap track 종료)
**브랜치**: `worktree-b5_trait_extension`
**다음 세션 진입 문장**: "μ-Q1 진행" 또는 "HTP microbench S25 verify 진행"

---

## TL;DR

QNN HTP backend 신규 추가 sprint 의 microbench-first PoC. **Backend trait
통합은 안 함** — microbench (`engine/microbench/htp_*.rs`) 만으로 production
hot path matmul 측정 후 결과 GREEN 일 때 Q-2 (Backend trait + 14-node
single-layer) 진입. **scope cap 2~3 영업일**. 측정 매트릭스 = **7-cell
(HTP FP32 / HTP W8A8 / OpenCL F16 / QNN-GPU F16 / OpenCL Q4_0 / Executorch FP32 / Executorch W8A8)** + CPU NEON reference. shape = Qwen2.5-1.5B FFN gate
`[1,1536] × [1536,8960]`. Tolerance: atol < 1e-3 (FP32 row) / atol < 0.1
(W8A8 row, Executorch §3 미확보로 cosine ≥ 0.99 보조).

본 세션 완료: μ-Q1.0-1 (SDK 셋업, host 빌드 PASS, build.rs worktree
fallback 동작 확인) + μ-Q1.1-1 (기존 `htp_matmul_correctness.rs` =
FP32 + args[K,N] 지원, Qwen shape 직접 적용 가능). 다음 세션 = S25
디바이스 작업 + Executorch 셋업 + 7-cell 측정.

---

## 진행 상태

### 본 세션 완료

| Task | 작업 | 결과 |
|---|---|---|
| #192 (μ-Q1.0-1) | SDK 위치 + host 빌드 | main repo `third_party/qnn_sdk_2.33/` 보유. `cargo build --release --bin microbench_htp_matmul_correctness --features qnn` **PASS** (24.09s) |
| #196 (μ-Q1.1-1) | 기존 bin shape 점검 | FP32 dtype, args[K,N] customizable. **신규 bin 작성 일부 불필요** — μ-Q1.1-2는 W8A8 로 재정의 |

### Plan + Task 자산

- Plan 파일: `.agent/todos/plan_qnn_htp_microbench_2026_05_26.md`
- Task: #192~#205 (μ-Q1.0~μ-Q1.3 + μ-Q1.E1~E4 + μ-Q1.M 매트릭스)
- Researcher 보고: 본 세션 transcript (Executorch HTP 1 matmul PoC 조사, 4 섹션)

---

## 측정 매트릭스 (7-cell + reference)

| ID | 측정 대상 | Bin / 도구 | Op | dtype | Status |
|---|---|---|---|---|---|
| **M1** | HTP FP32 (본 프로젝트) | `microbench_htp_matmul_correctness` | `qti.aisw/MatMul` | FP32 | 기존 자산 |
| **M2** | HTP W8A8 (본 프로젝트, 신규) | `microbench_htp_matmul_w8a8` (작성 필요) | `qti.aisw/MatMul` + `Qnn_QuantizeParams_t` | W8A8 sym | μ-Q1.1-2 |
| **M3** | OpenCL F16 baseline | `microbench_oppkg_gemv_vs_baseline` (raw OpenCL part) | `mul_mv_f16_f32` 커널 | F16 weight + F32 act | 기존 자산 (Qwen 1×1536×8960) |
| **M4** | QNN-GPU OpPackage F16 | `microbench_oppkg_gemv_vs_baseline` (OpPackage part) | OpPackage CustomMatMul | F16 + F32 | 기존 자산 |
| **M5** | OpenCL Q4_0 (본 production) | `microbench_qnn_oppkg_matmul_q40_correct` | OpPackage CustomMatMulQ40F32 | Q4_0 + F32 | 기존 자산 |
| **M6** | Executorch HTP FP32 | `qnn_executor_runner` + `.pte` | `aten.matmul` → QNN MatMul | FP32 | μ-Q1.E3/E4 |
| **M7** | Executorch HTP W8A8 | `qnn_executor_runner` + `.pte` (PT2E quantizer) | `aten.matmul` (W8A8) | W8A8 | μ-Q1.E3/E4 |
| **Ref** | CPU NEON | host f32 matmul (Rust ndarray 또는 직접) | f32 matmul | F32 | numerical reference 만 |

### Shape

- 1차: Qwen2.5-1.5B FFN gate **`M=1, K=1536, N=8960`** (production hot path)
- 2차 (선택): Llama 3.2 1B `M=1, K=2048, N=8192`
- 3차 (small, VTCM fit 확인용): `M=1, K=128, N=128`

### Tolerance

| Row | atol | rtol | 추가 |
|---|---|---|---|
| FP32 (M1 / M3 / M4 / M6) | < 1e-3 | < 1e-3 | 코드 주석 pass gate |
| W8A8 (M2 / M7) | < 0.1 | < 0.1 | Executorch §3 floating 클래스 기준 활용. cosine ≥ 0.99 추가 |
| Q4_0 (M5) | < 0.05 | < 0.05 | 기존 production 검증된 범위 |

---

## 측정 항목 (각 cell 공통)

**Numerical (vs CPU NEON Ref)**:
- max_abs_error
- mean_abs_error
- cosine_similarity
- relative_l2_norm `‖X - Ref‖₂ / ‖Ref‖₂`

**Latency** (워밍업 10 + 측정 30, **graph_execute 만 측정**):
- median (p50)
- p95, p99
- stddev
- end-to-end (mem_register/dereg 포함, 비교 cell 만)

**Memory** (HTP cells M1/M2/M6/M7 만):
- VTCM utilization (8/16/32 MB sweep)
- rpcmem alloc size

**Throughput**:
- GFLOPS = `2 × M × K × N / latency_seconds`

---

## 다음 작업 (다음 세션 진입 순서)

### 1차 단계: μ-Q1.0 디바이스 verify (0.5일)

| Task | 작업 | 명령 |
|---|---|---|
| #193 | S25 라이브러리 push | `python3 scripts/run_device.py -d galaxy_s25 push --src third_party/qnn_sdk_2.33/lib/aarch64-android/libQnnHtp.so /data/local/tmp/qnn/` + V79Stub + V79Skel (hexagon-v79/unsigned) |
| #194 | 9 HTP microbench sweep | 각 bin: `cargo build --release --target aarch64-linux-android --features qnn --bin <bin>` + push + run. 결과 표 (segfault / PASS / FAIL) |
| #195 | Segfault risk 확정 | Phase R 시점 2026-05-09 의 HTP microbench segfault 재현 여부. YES면 risk 분석 (SDK 버전 / Skel push 누락 / vendor lib conflict / soc_model 미일치) |

### 2차 단계: μ-Q1.1 production-shape 측정 (1일)

| Task | 작업 |
|---|---|
| #197 | `microbench_htp_matmul_w8a8.rs` 신규 작성 (htp_matmul_correctness 의 W8A8 변형 + `Qnn_QuantizeParams_t` scaleOffsetEncoding). symmetric per-tensor, max-abs scale calibration |
| #198 | S25 HTP vs OpenCL Q4_0 latency 측정. shape Qwen 1×1536×8960. 워밍업 10 + 측정 30 |

### 3차 단계: μ-Q1.E Executorch 셋업 + 측정 (1일)

| Task | 작업 |
|---|---|
| #201 | Executorch repo clone (`https://github.com/pytorch/executorch`) + Python venv (torch, executorch, transformers, qnn 의존성) |
| #202 | `qnn_executor_runner` Android arm64 빌드 — Executorch CMake + QNN backend + NDK r26 |
| #203 | `.pte` 빌드: single MatMul `[1,1536]×[1536,8960]` FP32 1개 + W8A8 PT2E 1개. `examples/qualcomm/scripts/export_example.py` 변형 |
| #204 | S25 push + 실행. 30 runs median + p50/p95/p99 |

### 4차 단계: μ-Q1.M 매트릭스 통합 + 결정 (0.5일)

| Task | 작업 |
|---|---|
| #205 | 7-cell 매트릭스 통합 표 + 보고서 `papers/eurosys2027/_workspace/experiment/htp_matmul_microbench_2026_05_26.md` |
| #199 | 결정 게이트: HTP/OpenCL ratio 기준 다음 트랙 (Q-2 진입 / heterogeneous / abandon) |
| #200 | commit + handoff + push + notify |

---

## Landmines / 미해결

### 1. soc_model SM8750 (V79) 여부 미확정 (Researcher §4-5)

Snapdragon 8 Elite (V79) 의 정확한 `QcomChipset` enum 값 미확보. Executorch
master 에서 `serialization/` 또는 `qnn_constants.py` 확인 필요. 우선 SM8650
(V75) 로 fallback 시도, 작동 시 SM8750 retry.

### 2. HTP microbench segfault risk (Memory 2026-05-09)

Phase R 시점 (16일 전) HTP microbench 모두 segfault, reboot 무효. SDK 2.33 +
V79 Skel push 셋업 후 재현 여부 #195 에서 확정. Skel 미푸시 또는 SDK header
↔ vendor lib 버전 불일치 (vendor 2.20 vs SDK 2.25) 의심.

### 3. VTCM tile budget 미튜닝

Qwen 1×1536×8960 matmul = ~14M elements. VTCM 8MB 로 spill 가능. `vtcm_size_in_mb`
8/16/32 sweep 필요. M2 (W8A8) 의 경우 더 큰 buffer 들어가므로 spill 가능성 ↑.

### 4. Executorch W8A8 PT2E pipeline 미경험

PT2E (`prepare_pt2e` → calibrate → `convert_pt2e` → `to_backend(QnnPartitioner)`)
는 본 프로젝트에 처음. calibration data 가 deterministic (single forward pass)
이면 scale/zero_point 결정 가능하지만, dtype 정밀도가 정확성 게이트 통과 못할
시 fallback path 정의 필요.

### 5. mem_handle 등록 비용 분리 (Researcher §4-6)

Executorch Issue #3949: "Simple op = 53 ms" — rpcmem alloc + mem_register +
graph_execute 합산. 측정 시 graph_execute 만 워밍업/측정 분리 필수.

### 6. context binary version mismatch (Researcher §4-1)

Error 30010 ("Failed to interpret QNN context binary"). 빌드 시 SDK 2.33 +
V79 binary header magic 검증. Executorch 측 .pte 도 같은 risk.

### 7. SDK 2.33 vendor 2.20 incompatibility (Memory)

vendor runtime API 2.20.0 ↔ SDK header 2.25 incompatible. SDK .so 우선
dlopen (`LD_LIBRARY_PATH=/data/local/tmp/qnn:/vendor/lib64`) 필요. 본 세션
빌드는 host 였으나 디바이스에서는 ordering critical.

---

## 자기점검

- [x] 진입 문장 한 줄만으로 다음 세션이 첫 명령 가능 — "μ-Q1 진행"
- [x] 왜 멈췄는가 명시 — 디바이스 push + Executorch 셋업이 1턴 초과 작업
- [x] 가장 큰 landmine 표면화 — segfault risk + soc_model V79 미확정
- [x] 검증 게이트 수치 표현 — atol/rtol 표 + GFLOPS 공식 + cell 별 status
- [x] 본문 길이 적정 — ~500 라인 inline, 상세는 plan 파일 link

---

## 즉시 재현 명령 (다음 세션 시작점)

### S25 push (μ-Q1.0-2)

```bash
# QNN HTP runtime libs (host arm64)
adb -s galaxy_s25 push third_party/qnn_sdk_2.33/lib/aarch64-android/libQnnHtp.so /data/local/tmp/qnn/
adb -s galaxy_s25 push third_party/qnn_sdk_2.33/lib/aarch64-android/libQnnHtpV79Stub.so /data/local/tmp/qnn/
adb -s galaxy_s25 push third_party/qnn_sdk_2.33/lib/aarch64-android/libQnnSystem.so /data/local/tmp/qnn/

# Hexagon DSP skel
adb -s galaxy_s25 push third_party/qnn_sdk_2.33/lib/hexagon-v79/unsigned/libQnnHtpV79.so /data/local/tmp/qnn/
adb -s galaxy_s25 push third_party/qnn_sdk_2.33/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so /data/local/tmp/qnn/

# 검증
adb -s galaxy_s25 shell ls /data/local/tmp/qnn/
```

### Android 빌드 + 첫 실행 (μ-Q1.0-3 시작)

```bash
# Cross-build
python3 scripts/run_device.py -d galaxy_s25 build microbench_htp_matmul_correctness --features qnn

# Push + run (Qwen FFN scale)
python3 scripts/run_device.py -d galaxy_s25 run microbench_htp_matmul_correctness -- 1536 8960
# 기대: "HTP graph (MatMul) finalize: OK" + numerical pass + latency log
```

### OpenCL 비교 baseline (M3 + M4 동시)

```bash
python3 scripts/run_device.py -d galaxy_s25 run microbench_oppkg_gemv_vs_baseline --features qnn,opencl
# 기대: raw OpenCL TBT + OpPackage TBT 양쪽 출력
```

---

## 진입 명령 (다음 세션)

```
"μ-Q1 진행"                                # 풀 sprint
"μ-Q1.0-2 진행" 또는 "S25 라이브러리 push 진행"  # 디바이스 verify 시작
"μ-Q1.E1 진행"                             # Executorch 셋업부터 분리 진입
```

---

## 관련 자산

- Plan: `.agent/todos/plan_qnn_htp_microbench_2026_05_26.md`
- Memory: `[[project_liswap5_phase10_htp_feasibility_20260509]]` (16일 전 HTP feasibility)
- Memory: `[[project_qnn_oppkg_phase_r_complete_20260509]]` (M1 진입 직전 상태)
- 기존 9 HTP microbench: `engine/microbench/htp_*.rs`
- 기존 28 QNN OpPackage bin: `engine/microbench/qnn_*.rs`
- OpenCL F16/OpPackage baseline: `engine/microbench/oppkg_gemv_vs_baseline.rs` (이미 Qwen 1×1536×8960 shape)
- SDK: `/home/go/Workspace/llm_rs2/third_party/qnn_sdk_2.33/` (main repo, gitignored)
- Researcher 보고: 본 세션 transcript (4 섹션 — Executorch 트리 / matmul flow / tolerance / 함정)
