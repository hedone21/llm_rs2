# Plan: QNN HTP Microbench 비교 측정 protocol

**작성**: 2026-05-26
**HEAD**: `82758e7d docs(handoff): μ-Q1 QNN HTP matmul microbench PoC sprint 진입 plan + handoff`
**브랜치**: `worktree-b5_trait_extension`
**관련 plan**: [`plan_qnn_htp_microbench_2026_05_26.md`](plan_qnn_htp_microbench_2026_05_26.md) (sprint phase 분할)
**진입 문장**: "QNN microbench protocol Phase A 진행"

---

## 목적

μ-Q1 sprint 의 **9-cell + Ref** 매트릭스를 신뢰성 있게 측정. 4 요소를 보장:

1. **반복**: cell 당 warmup 3 + measure 10 회, median + CV 보고
2. **발열 제어**: 4 thermal zone polling + 50°C 임계 + cooldown 강제
3. **순서 randomize**: cell 순서 round-robin 으로 발열 누적 편향 제거
4. **실패 처리**: 1차 재시도 → 2차 환경 강화 재시도 → 3차 cell skip (사용자 지시 "해결 안 되면 넘어가")

본 protocol 은 측정 방법론에 한정. sprint 진행 phase 는 자매 plan 참고.

---

## 측정 매트릭스 (확정)

Shape 전 cell 공통: **Qwen 2.5-1.5b FFN gate `[1, 1536] × [1536, 8960]`** (M=1 GEMV hot path).

| Cell | dtype | Backend / Runtime | 기존 자산 | 신규 필요 |
|---|---|---|---|---|
| **M1** | FP32 | QNN HTP | `microbench_htp_matmul_correctness` (args[K,N] customizable) | dtype default 유지 |
| **M1b** | F16 | QNN HTP | M1 + `Qnn_DataType_t_QNN_DATATYPE_FLOAT_16` | dtype enum 변경 1줄 |
| **M2** | W8A8 | QNN HTP | (없음) | **신규**: `microbench/htp_matmul_w8a8.rs` |
| **M3** | F16 | OpenCL raw (`mul_mv_f16_f32`) | `microbench_oppkg_gemv_vs_baseline` (baseline part) | 그대로 |
| **M4** | F16 | QNN-GPU OpPackage | 위 bin (OpPackage part) | 그대로 |
| **M5** | Q4_0 | OpenCL production | `microbench_qnn_oppkg_matmul_q40_correct` | 그대로 |
| **M6** | FP32 | Executorch HTP | (없음) | `.pte` + `qnn_executor_runner` |
| **M6b** | F16 | Executorch HTP | (없음) | `.pte` (`use_fp16=True`) |
| **M7** | W8A8 | Executorch HTP | (없음) | `.pte` (PT2E pipeline) |
| **Ref** | F32 | CPU NEON 6T | CPU baseline (정확성 oracle) | 그대로 |

### Fair pairing (paper 도식 근거)

| Row | Cells | 비교 의미 |
|---|---|---|
| **FP32** | M1 ↔ M6 | QNN runtime native vs Executorch 의 framework overhead |
| **F16** | M1b ↔ M3 ↔ M4 ↔ M6b | HTP vs OpenCL raw vs QNN-GPU OpPackage vs Executorch |
| **W8A8** | M2 ↔ M7 | QNN raw vs Executorch PT2E pipeline |
| **production reference** | M5 | OpenCL Q4_0 (현재 production hot path) — 단독 reference |

---

## 신뢰성 protocol

### 반복 횟수

| 단계 | 횟수 | 처리 |
|---|---|---|
| Warmup | 3 | 측정에 미포함 (cache/JIT/kernel 컴파일/mem_handle register 워밍업) |
| Measure | 10 | median + p25/p75 + p95 + min/max + CV (%) 계산 |

### 통계 처리

- **Median** 을 primary metric 으로 보고 (mean 은 outlier 민감)
- **CV (coefficient of variation)** = `stddev / median × 100`
  - CV < 5% : GREEN (신뢰)
  - 5% ≤ CV < 10% : YELLOW (재측정 옵션 또는 보고서 명시)
  - CV ≥ 10% : RED (재측정 필수, 원인 분석)
- **Outlier rejection**: Tukey 1.5×IQR rule. raw + filtered 양쪽 기록.

### 측정 순서 randomize

- 10 cell × (3+10) trial = 130 trial
- **Round-robin shuffle**: `for round in 1..13: for cell in shuffle(cells): run`
- 이유: 한 cell 의 마지막 trial 만 발열 누적으로 부풀어 median 왜곡되는 아티팩트 제거

### 정확성 검증

- 동일 input matrix (rand seeded fixed, e.g. seed=42)
- CPU NEON F32 (Ref) 가 정답 oracle
- tolerance:
  - FP32 (M1, M6): `max_abs_err < 1e-3`
  - F16 (M1b, M3, M4, M6b): `max_abs_err < 1e-2` AND `cosine ≥ 0.999`
  - W8A8 (M2, M7): `max_abs_err < 0.1` AND `cosine ≥ 0.99`
  - Q4_0 (M5): `max_abs_err < 0.05` AND `cosine ≥ 0.999`

---

## 발열 제어 protocol

### 측정 환경

| 항목 | 설정 | 이유 |
|---|---|---|
| 디바이스 | S25 standby | thermal 변동 최소화 |
| USB 충전 | OFF (배터리 ≥ 60%) | 충전 자체가 thermal load |
| 방 온도 | 22~26°C | 측정 시작 시점 기록 |
| Airplane mode | ON | RF 발열 + background sync 제거 |
| Screen | OFF (AOD OFF) | display thermal load |
| Background apps | force-stop | `adb shell am force-stop com.samsung.*` |
| Wi-Fi / BT / GPS | OFF | 동일 사유 |

### Thermal zone monitoring

기존 `bench_strict_thermal_isolation.sh` 패턴 재사용:
```
zone 1  = cpu-0-0-0 (little cluster)
zone 10 = cpu-0-4-1 (big cluster)
zone 28 = gpuss-5   (Adreno)
zone 30 = gpuss-7   (Adreno)
```

추가로 HTP 측정 시 NPU 관련 zone 사전 확인 — S25 의 `cat /sys/class/thermal/thermal_zone*/type | grep -i hexagon` 로 npu/hexagon zone 식별 후 monitoring 추가.

### 온도 임계

| 임계 | 값 | 동작 |
|---|---|---|
| **시작 임계** | 모든 zone ≤ 40°C | 측정 시작 가능 |
| **trigger 임계** | 측정 중 ≥ 50°C | 현재 trial discard + cooldown |
| **위험 임계** | ≥ 60°C | 측정 전면 중단 (디바이스 보호) |
| **회복 목표** | ≤ 38°C | trigger 후 cooldown 종료 조건 |

기존 스크립트는 `THERMAL_THRESHOLD_MC=50000` (50°C) 를 채용 — 동일하게 적용.

### Cooldown 정책

| 구간 | 최소 시간 | 최대 대기 | 동작 |
|---|---|---|---|
| **inter-trial** (같은 cell 내) | 5s | 30s | 빠른 cool, CV>10% 면 자동 증가 |
| **inter-cell** (cell 전환) | 45s | 180s | 기존 `COOLDOWN_MIN_SEC` 재사용 |
| **inter-round** (130 trial 의 round 사이) | 300s | 600s | 누적 발열 회복 |
| **세션 시작** | 60s + 1 trial discard | — | first-run cold-start 보정 |
| **trigger 임계 도달** | — | 600s | zone ≤ 38°C 까지 polling |

### Thermal throttling 감지

- 같은 cell 의 trial N latency > trial 1 latency × 1.15 → throttling 의심
- 자동 alert + cooldown 강제 + 해당 round 재시도 (재시도 후 동일 패턴이면 fixed 보고)

### Zombie process 사전 점검

기존 패턴 그대로:
```bash
adb shell 'ps -A | grep -E "(generate|microbench|llama-cli|qnn_executor)"' | grep -v grep
```
검출 시 측정 중단 + 사용자 알림 (자동 kill 안 함 — 사용자 in-progress 작업 보호).

---

## 자동화 스크립트

### 신규: `scripts/microbench_qnn_matrix.py`

기존 `bench_strict_thermal_isolation.sh` 의 thermal logic + `run_device.py` 의 ADB 추상화 합성.

#### 입력
- `--cells M1,M1b,M2,...` (default: 전체)
- `--rounds 13` (warmup 3 + measure 10)
- `--seed 42`
- `--cooldown-min 45 --cooldown-max 180`
- `--thermal-threshold-mc 50000`
- `--out papers/eurosys2027/_workspace/experiment/qnn_microbench_<YYYY_MM_DD>/`

#### 출력 구조
```
qnn_microbench_2026_05_27/
├── raw/
│   ├── M1_round01_trial00.json    # trial 별 raw timing + accuracy
│   ├── M1_round01_trial01.json
│   └── ...
├── aggregated.csv                 # cell, n, median, p25, p75, p95, CV, max_err, cosine
├── thermal_log.csv                # timestamp, zone, temp_c, current_cell, trial_id
├── round_sequence.json            # 실제 실행된 round-robin 순서 (reproducibility)
├── env.json                       # device info, library SHA, build profile
└── report.md                      # auto-generated table + fair-pair 분석
```

#### 실행 단위 (pseudocode)
```python
preflight_zombie_check()
preflight_thermal_ok()
preflight_airplane_mode_check()
session_warmup_cooldown(60s)

cells_order = []
for r in range(rounds):
    cells_order.append(shuffle(cells, seed=seed+r))

for round_idx, round_cells in enumerate(cells_order):
    for cell in round_cells:
        wait_thermal_below(40)
        result = run_cell(cell, trial=round_idx)
        if result.is_outlier or result.cv_so_far > 10:
            mark_for_retry(cell, round_idx)
        sleep(inter_trial=5..30)
    sleep(inter_cell=45..180)
sleep(inter_round=300..600)

aggregate_and_report()
```

### CPU 6T pinning

- `taskset 0x3f` (little 4 + big mid 2, big primary 1 제외)
- governor: `performance` (max freq pin, 측정 안정성)
- 측정 종료 시 governor 복원 (cleanup hook)

---

## Per-cell 측정 내용

### Common
- timing: **wall-clock 만** (OpenCL profile event 금지 — feedback 준수, driver-specific 패널티)
- accuracy: max_abs_err, mean_abs_err, cosine_similarity vs Ref
- per-trial 추가 메타: thermal at start/end, trial duration, mem usage delta

### Per-cell 분리 항목

#### M1 / M1b / M2 (HTP)
- Pre-warm: mem_handle register × 3 회 (Executorch issue #3949: simple op = 53 ms overhead 분리)
- Per-trial 측정: `Qnn_GraphExecute` 만
- VTCM sub-sweep: 8MB / 16MB / 32MB 각 3 trial best 채택, 결정 후 fixed
- soc_model: SM8750 (V79) — `qnn_constants.py` 확인 후 hardcode

#### M3 / M4 (OpenCL)
- Pre-warm: kernel build cached 후 3 회 dispatch
- Per-trial: `clEnqueueNDRangeKernel` + `clFinish` wall-clock
- OpPackage 측정 (M4) 시 fast-path / non-fast-path 둘 다 trial 보고

#### M5 (OpenCL Q4_0)
- production hot path setting 그대로
- block size 32 weight + F32 activation (W4-only)

#### M6 / M6b / M7 (Executorch)
- pre-built `.pte` 적재 후 `Module::init()` 분리
- Per-trial: `module.execute(inputs)` 만 (load 미포함)
- W8A8 (M7): PT2E pipeline (`torch.export` + `convert_pt2e`) — calibration 데이터 fixed seed

#### Ref (CPU NEON)
- 6T pinned, single batch
- F32 정확성 oracle
- timing 도 표에 포함 (NEON F32 GEMV 절대 기준)

---

## 검증 게이트

### Per-cell GREEN 조건
1. **정확성**: 위 tolerance 표 만족
2. **신뢰성**: CV < 5% (최대 10% 까지 허용 — 보고서에 명시)
3. **발열 안전**: thermal log 에서 trigger 임계(50°C) 도달 없이 완주

### Matrix-level GREEN 조건
- 9 cell + Ref 모두 위 조건 통과
- Fair-pair row (FP32 / F16 / W8A8) 각각 결론 도출 가능

### 실패 처리 (사용자 지시 "해결 안 되면 넘어가")

| 단계 | 조치 |
|---|---|
| **1차** | VTCM sweep / kernel param 재시도 1회 |
| **2차** | 환경 강화 (전체 cooldown 600s + reboot + airplane mode 재확인) 재측정 1회 |
| **3차** | 실패 보고 + 다음 cell 진행 (skip), 보고서에 `FAILED` 마킹 |

- Matrix 한 셀 fail 은 보고서에 명시. fair-pair 분석은 그 셀 제외하고 진행.
- 디바이스 ≥ 60°C 위험 임계 도달 시: 측정 전면 중단 + 사용자 알림 + 30분 cooling 권고.

---

## Phase 분할

자매 plan 의 sprint phase 와 본 protocol phase 매핑:

| Plan phase | 본 protocol phase | 내용 | 예상 |
|---|---|---|---|
| μ-Q1.0-1 (SDK verify) | — | (이미 plan_qnn_htp_microbench 에 위임) | 0.5d |
| μ-Q1.0-2/3 (libs push + sweep) | **A** | `microbench_qnn_matrix.py` 작성 + thermal monitoring + dry-run (Ref + M5 만) | 0.5d |
| μ-Q1.1 (htp w8a8 신규) | **B** | M2 신규 bin + M1/M1b/M5 측정 (기존 자산 시리즈) | 0.5d |
| (oppkg verify) | **C** | M3/M4 측정 (OpenCL + QNN-GPU OpPackage) | 0.5d |
| μ-Q1.E1~E4 (Executorch) | **D** | Executorch repo + `.pte` 빌드 + M6/M6b/M7 측정 | **변동 1~2d** |
| μ-Q1.M / μ-Q1.2 | **E** | 통합 분석 + 결정 보고서 + handoff | 0.5d |

- **총 예상**: 3~4 영업일 (sprint cap 2~3d 초과 가능 — Executorch 가 variance)
- **단축 옵션**: D phase 가 Executorch 의 .pte 빌드 path 에서 막히면 1차 skip → M6/M6b/M7 RED 마킹 후 Matrix-level 결정. 후속 sprint 로 위임.

---

## Landmines

1. **온도 zone 이름 디바이스별 차이** — S25 의 zone naming 사전 확인 필수. `bench_strict_thermal_isolation.sh` 의 zone 1/10/28/30 가 S25 (Snapdragon 8 Elite) 에서도 동일한지 검증.
2. **VTCM sweep 비용 폭증** — 8/16/32 MB × 3 trial 만으로 best 결정. 본 측정 (× 13 trial) 은 best VTCM 만.
3. **Executorch runner crash** — 1 trial crash 면 module re-init. 2회 재발 시 cell skip (실패 처리 3차).
4. **adb shell session timeout** — long run 시 keep-alive (`adb shell -t` 또는 chunked invocation).
5. **mem_handle register 비용 의 graph 외부 분리** — measure 시 `graph_execute` 만. Executorch 도 `module.init()` 분리.
6. **W8A8 quantize/dequantize boundary** — input quantize (CPU) + output dequantize (CPU) 는 graph 외부 → 측정 제외. cell 간 일관성 위해.
7. **OpenCL profile event 금지** (feedback `feedback_opencl_profile_events_cross_engine.md`) — driver-specific 패널티. wall-clock 만.
8. **TBT metric tok0 inclusive** (feedback `feedback_tbt_metric_tok0_inclusive.md`) — 본 측정은 single-op latency 라 무관. 단, 향후 multi-op extension 시 주의.
9. **Galaxy S25 6T fixed** (feedback `feedback_benchmark_thread_count.md`) — Ref CPU NEON 8T 금지.
10. **Background app thermal pollution** — 매 round 시작 전 `am force-stop` 강제.
11. **충전 케이블 thermal load** — USB cable disconnect 권장. 어렵다면 USB charging disable (`dumpsys battery set charge 0`).
12. **방 온도 변동** — 시작 시점 + 종료 시점 기록. 4°C 이상 차이 시 보고서에 noise factor 명시.

---

## 산출물

- **신규 스크립트**: `scripts/microbench_qnn_matrix.py`
- **신규 microbench**: `engine/microbench/htp_matmul_w8a8.rs` (M2)
- **신규 microbench (선택)**: `engine/microbench/htp_matmul_f16.rs` (M1b — M1 enum 변경 1줄이면 별 bin 불필요)
- **측정 결과 디렉토리**: `papers/eurosys2027/_workspace/experiment/qnn_microbench_<YYYY_MM_DD>/`
  - `raw/<cell>_round<NN>_trial<MM>.json` × ~130
  - `aggregated.csv`
  - `thermal_log.csv`
  - `round_sequence.json`
  - `env.json`
  - `report.md`
- **handoff**: `.agent/todos/handoff_qnn_microbench_results_<date>.md`

---

## 진행 권장 순서

1. **Phase A** (자동화 + dry-run): Implementer
   - `microbench_qnn_matrix.py` 작성 + Ref + M5 dry-run 2 round 측정
   - dry-run 결과로 thermal protocol 검증 (zone naming, threshold 적합성, cooldown 시간)
2. **Phase B** (HTP raw 시리즈): Tester (디바이스 게이트)
   - M2 신규 bin → Senior Implementer (W8A8 quantization 처리는 비자명)
   - M1 / M1b / M5 측정 → Tester
3. **Phase C** (OpenCL + OpPackage): Tester
4. **Phase D** (Executorch): Researcher 선행 (.pte 빌드 path 조사) → Implementer (셋업) → Tester (측정)
5. **Phase E** (분석 + 보고서): 메인 세션 (오케스트레이터) + Architect (결정 게이트)

---

## 다음 액션 (Phase A step 1)

1. `scripts/microbench_qnn_matrix.py` 골격 작성
   - `bench_strict_thermal_isolation.sh` 의 thermal logic 을 Python 으로 포팅 + ADB 추상화
   - cell registry (bin path + arg + parser) 정의
2. S25 `/sys/class/thermal/thermal_zone*/type` 으로 NPU 관련 zone 식별
3. Ref + M5 dry-run 2 round 측정 (예상 ~15 분)
4. Dry-run 결과 분석:
   - CV 가 5% 이내 들어오는가?
   - thermal threshold 50°C 가 적합한가?
   - cooldown 45s 충분한가?
5. 결과 따라 protocol 파라미터 조정 후 Phase B 진입

---

## 관련 자료

- Sprint plan: [plan_qnn_htp_microbench_2026_05_26.md](plan_qnn_htp_microbench_2026_05_26.md)
- Entry handoff: [handoff_qnn_htp_microbench_entry_2026_05_26.md](handoff_qnn_htp_microbench_entry_2026_05_26.md)
- Thermal control 패턴: `scripts/bench_strict_thermal_isolation.sh`, `scripts/bench_thermal_controlled.sh`
- ADB 추상화: `scripts/run_device.py`
- Memory:
  - `feedback_opencl_profile_events_cross_engine.md` (profile event 금지)
  - `feedback_benchmark_thread_count.md` (S25 6T)
  - `feedback_adreno_subgroup_reduce.md` (이론보다 실측)
  - `reference_microbench_flash_attn.md` (microbench 도구 선례)
