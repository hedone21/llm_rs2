# Partition P2 — sync_drain 재조사 및 추가 회수

**상태**: 착수 대기 (다음 세션 시작점)
**담당 후보**: senior-implementer (OpenCL event 처리 + trace 확장)
**우선순위**: P2 (P1 = CPU NEON Q4_0 GEMV 최적화는 Q4_0-only, 본 작업은 dtype 무관)
**예상 이득**: Q4_0 -2~-5 ms/tok, F16 -1~-3 ms/tok (이론 상한 10.9 ms/tok 전체 회수)

---

## 0. Context — 이전 세션 완료 사항

### 직전 커밋

- `cb3908e` — `perf(tensor_partition): fused norm-merge kernel to eliminate layer barrier`
- `9c0281a` — `test(opencl): add microbench_async_write for Adreno overlap gate`
- `5ca7ba0` — `perf(tensor_partition): add async residual read (LLMRS_PARTITION_ASYNC_READ)` (diagnostic)

### Fused norm-merge 통합 결과 (greedy, Galaxy S25 / Adreno 830, Qwen 2.5-1.5B, r=0.7)

| 경로 | Q4_0 TBT | F16 TBT | sync_drain | merge |
|---|---|---|---|---|
| baseline (fused OFF) | 68.87 ms | 68.45 ms | 0.40 ms/layer | 0.13 ms/layer |
| fused ON | 62.99 ms | 68.08 ms | 0.39~0.45 ms/layer | 0.01 ms/layer |
| Δ | -5.88 (-8.5%) | -0.37 (-0.54%) | ~0 | **-0.13 (제거 완료)** |

- Merge kernel 제거는 dtype 무관 구조적 이득으로 확정.
- Q4_0는 CPU critical path → merge 회수가 TBT에 직접 반영됨.
- F16은 GPU bandwidth-bound → 이득 대부분이 `gpu_wait`로 흡수.

### 여전히 남은 병목 (fused ON 기준)

| 성분 | ms/tok | 점유 | 비고 |
|---|---|---|---|
| **CPU matmul** | 34.4 | 55% | P1 영역 — Q4_0 NEON 최적화 |
| **sync_drain** | 10.9 | 17% | **본 작업 P2 목표** |
| gpu_wait | 5.0 | 8% | CPU/GPU 타이밍 차 |
| attn + norm + lm_head | ~12 | 19% | |
| merge | 0.3 | 0.5% | 제거됨 |

---

## 1. 문제 정의

fused norm-merge로 `merge` 성분(3.6 ms/tok)은 제거했으나, `sync_drain` 11 ms/tok은 거의 그대로 남음. Plan에서 기대했던 B+C+D 성분(5.6 ms) 회수가 실측 0에 수렴.

### sync_drain 재구성 (추정)

A1 실패(+36 ms) 당시 sync_drain 11.2 ms/tok을 4성분으로 분해:

| 성분 | 추정 ms/tok | 성격 |
|---|---|---|
| A. Prior work 대기 (이전 layer GPU 잔존) | 5.6 | 구조적 — CPU-GPU race |
| B. clFinish ioctl | 1.4 | 시스템콜 오버헤드 |
| C. Driver kernel prefetch 상실 | 2.8 | clFinish 후 queue empty |
| D. Post-sync re-dispatch | 1.4 | 다음 kernel enqueue 재개 |

Fused merge로 B+C+D는 **이론적으로** 제거 대상이었으나 실측 ~0 회수. 이유는 `backend.synchronize()` 호출이 **fused 경로에서도 여전히 존재**하기 때문으로 추정 (`forward_gen.rs:1091, 1116`).

### 실패 이유 가설

1. **Residual을 CPU가 읽어야 하므로** `synchronize + read_buffer` 필요. Zero-copy 경로에서도 GPU write 완료 대기는 필요 (cache coherence).
2. **CPU matmul 수행 전** residual 가시화 보장 필요 → synchronize는 구조적으로 제거 어려움.
3. 단, synchronize의 real cost가 실제로 **'prior work wait'가 대부분**이고 B+C+D는 merge 제거 후 이미 자동 흡수되었을 가능성도 존재 — 즉 plan의 A/B/C/D 분해 자체가 재검증 필요.

---

## 2. 본 작업 목표

### 2.1 진단 (Phase 1, 0.5d)

**trace 확장**: 현재 partition trace에 `zcopy_residual` 여부와 synchronize의 **실제 wait 길이**를 분리 기록.

대상 파일: `engine/src/layers/tensor_partition.rs` (trace 구조), `engine/src/layers/transformer_layer/forward_gen.rs:1082-1135` (synchronize 지점).

구체 항목:
- `zcopy_used`: bool (layer 당)
- `synchronize_cost_ns`: backend.synchronize() 호출 직전/직후 wall-clock 차
- `read_cost_ns`: read_buffer 호출 비용 (zero-copy면 0)

현재는 `sync_ns`가 `synchronize + read_buffer`를 합친 측정인지, 분리되어 있는지 재확인 필요 (이전 세션에서 분리했으나 zero-copy 경로 명시적 로깅 없음).

**목표 산출**:
- zero-copy 경로 실제 활성 여부 확인 (ARM UMA + HOST_PTR 설정)
- synchronize 단독 비용 (prior work wait 성분 A의 실측값)
- read_buffer 단독 비용 (zero-copy 아닌 경로의 DMA 부담)

**Abort 조건**: zero-copy가 이미 활성인데도 sync_drain 11 ms/tok이면 A 성분이 전부(구조적) → 본 P2 작업 중단 권고. P1(CPU NEON Q4_0) 선회.

### 2.2 개입 (Phase 2, 1d, Phase 1 결과에 조건부)

**Case A**: zero-copy 미활성 발견 시 → 활성화 조건 검토 (rewrap, HOST_PTR).

**Case B**: zero-copy 활성이지만 synchronize 비용 ≥ 5 ms 확인 시 → **event 기반 wait 대체** 재시도.

A1 실패 교훈(+36ms 회귀)은 '전체 synchronize 제거'였으나, P2는 더 세밀:
- **Layer N의 `add_rms_norm_oop` event만 wait**: 이전 모든 GPU 작업이 아니라 residual을 쓰는 특정 kernel만.
- 이는 OpenCL `clSetEventCallback` 또는 user-event로 더 좁은 barrier를 만드는 접근.
- 구현 복잡도 높음. A1과 차이: A1은 synchronize 전면 제거, 본 접근은 특정 event만 wait로 교체.

**Case C**: synchronize 비용이 실제로 낮고 (e.g., <1 ms/layer) 대부분이 A 성분(구조적 CPU-GPU race) → 추가 회수 불가. P1 선회 권고.

### 2.3 측정 + 채택 판단 (Phase 3, 0.5d)

- Galaxy S25 Q4_0 + F16 각각 실측
- 토큰 bit-exact 유지 확인 (fused norm-merge와 동일 원칙)
- 채택 기준:
  - Q4_0 TBT -2 ms 이상 + 토큰 일치 → 기본 on 전환
  - -1~-2 ms → diagnostic flag
  - 미달 또는 회귀 → rollback, P1으로 선회

---

## 3. 관련 파일 (변경 대상 예상)

| 파일 | 예상 변경 |
|---|---|
| `engine/src/layers/tensor_partition.rs` | trace 구조에 zcopy/sync 분리 필드 |
| `engine/src/layers/transformer_layer/forward_gen.rs:1082-1135` | synchronize 세분화, event wait 시도 |
| `engine/src/backend/opencl/mod.rs` | event-based wait API (A1 산물 재활용 가능) |
| `engine/src/core/backend.rs` | 필요 시 trait 메서드 추가 (enqueue_*_with_event류) |

---

## 4. 참고 자료 (우선순위 순)

### 4.1 직전 커밋 diff

```bash
git show cb3908e --stat    # fused norm-merge 통합
git show 9c0281a --stat    # Step 0 microbench
git show 5ca7ba0 --stat    # A1 async read (diagnostic, 실패 사례)
```

### 4.2 메모리 (이전 세션 결과)

- `project_partition_bottleneck` (memory) — DRAM BW 상한 분석 (17~23%만 사용, GPU kernel cap이 실제 ceiling)
- `project_partition_a1_async_read_failed` (memory) — A1 실패 원인 분해 (merge barrier 이전 + 메모리 경합)
- `project_partition_next_step` (memory) — fused norm-merge 계획 (현재 완료, 본 P2가 후속)

### 4.3 구조 문서

- `arch/tensor_partition.md` — partition 설계 R1 (contention mitigation 포지셔닝)
- `.agent/todos/partition_fused_norm_merge.md` — P1 완료 기록, 특히 §1 Design의 sync_drain 분해표

### 4.4 현재 코드의 핵심 지점

- `engine/src/layers/transformer_layer/forward_gen.rs:1082` (`zcopy_residual = !ws.residual.as_ptr().is_null()`)
- `engine/src/layers/transformer_layer/forward_gen.rs:1091, 1116` (synchronize 호출 2곳)
- `engine/src/layers/transformer_layer/forward_gen.rs:142` (fused_norm_merge 분기 진입점, L>0 + fused flag on)
- `engine/src/layers/tensor_partition.rs` — `record_partition_timing`, `PART_SYNC_NS`

### 4.5 측정 명령 (recheck baseline)

```bash
# Thermal gate < 42°C
adb shell cat /sys/class/thermal/thermal_zone0/temp

# Q4_0 baseline
adb shell "cd /data/local/tmp && LLMRS_PARTITION_TRACE=1 LLMRS_PARTITION_FUSED_MERGE=1 \
  ./generate --model-path /data/local/tmp/models/qwen2.5-1.5b --weight-dtype q4 \
  -b opencl --threads 6 --no-gpu-plan \
  --prompt-file prompts/prefill_128.txt \
  --tensor-partition 0.7 -n 128 --ignore-eos --greedy 2>&1" | tail -40

# F16 baseline
# 위와 동일, --weight-dtype f16
```

기대 기준선 (현재 fused ON):
- Q4_0: TBT ~63 ms/tok, sync_drain ~0.4 ms/layer
- F16: TBT ~68 ms/tok, sync_drain ~0.4 ms/layer

---

## 5. 착수 시 첫 명령

```bash
# 환경 재확인
cd /Users/li/Workspace/llm_rs2
git log --oneline -5
git status

# 현재 baseline 재현
adb devices
python scripts/run_device.py -d galaxy_s25 --build-only generate  # 빌드 확인

# Phase 1 진단 시작: 우선 trace에 zcopy_residual 로깅만 추가
# (최소 침습, fused 경로 유지)
```

---

## 6. 완료 시

- 커밋 예시 (채택 시): `perf(tensor_partition): recover sync_drain via event-based wait`
- 커밋 예시 (구조적 하한 확인 시): `docs(analysis): sync_drain structural floor — event wait inconclusive`
- 메모리 업데이트:
  - `project_partition_p2_sync_drain_result.md` — 실측 + 판정
  - `project_partition_next_step.md` 업데이트 (P1 CPU NEON이 다음 레버인지 or 또 다른 방향)
- `arch/tensor_partition.md` — sync_drain 구조적 하한 명시 (확인 시)
- 데스크톱 알림: `notify-send "llm.rs" "P2 sync_drain: <결과>"`

---

## 7. 대안 선회 기준

P2에서 실측상 이득 없음 확인 시 **즉시 P1으로 전환**:

- **P1 — CPU NEON Q4_0 GEMV 최적화**: 34.4 ms/tok critical path 단축. 10% = -3.4 ms, 20% = -6.8 ms 잠재.
- 대상: `engine/src/backend/cpu/neon/` 내 `fused_matmul_q4_0`, `fused_matmul_f16`
- 기법: SDOT unroll/tile, Q4 dequant pipeline overlap, `prfm pldl1keep` prefetch

P3 이후 레버:
- **P3 — GPU Q4 GEMV Adreno 튜닝**: `gemv_noshuffle_q4_0.cl` 레지스터/LDS 최적화
- **P4 — partition ratio 재튜닝**: CPU 빨라진 후 r=0.6~0.65 실험
- **P5 — Gemma3 fused merge 확장**: post_ffn_norm 포함 4-input 커널
