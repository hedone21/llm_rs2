# Tensor Partition — Fused Norm-Merge (N-layer batching)

**상태**: 착수 대기 (compact 후 재개)
**담당**: Senior Implementer (OpenCL 커널 + Backend trait + forward_gen 재편)
**우선순위**: P1 (partition을 plan path 대비 순이득 구간으로 진입시키는 유일한 구조 변경)

---

## 0. Context (왜 이 작업인가)

### 지금까지 확인된 사실 (2026-04-20 실측)

Galaxy S25 / Qwen 2.5-1.5B Q4_0 / r=0.7 / Strategy B whole-FFN partition:

| 경로 | TBT |
|---|---|
| Plan path (GPU-only, 연속 chain) | 64.15 ms/tok |
| Partition r=0.7 (현재) | 66.45 ms/tok (+2.3 ms 손실) |

**r sweep 확정**: r=0.7이 실측 sweet spot (r↑ 시 GPU slice 커져 단조 악화).

### Trace breakdown (r=0.7)

| 단계 | ms/layer | ms/token |
|---|---|---|
| sync_drain | 0.40 | 11.2 |
| dma_read | 0.00 | ~0 (zero-copy) |
| cpu_matmul | 1.23 | 34.4 (critical path) |
| gpu_wait | 0.18 | 5.0 |
| merge | 0.13 | 3.6 |

### A1 (async residual read) 실패 — `commit 5ca7ba0`

`LLMRS_PARTITION_ASYNC_READ=1`로 `synchronize()` 제거 + `enqueue_read_buffer(blocking=false, event)` 시도: **+36 ms 회귀**. 원인: merge step이 새 barrier가 되어 drain이 merge 시점으로 이전, CPU↔GPU 메모리 경합 추가.

**확정된 결론**:
1. sync_drain 11.2 ms는 단순 host wait가 아님 — pipeline prefetch 상실 + clFinish ioctl + re-dispatch overhead + prior work 대기의 **합성**.
2. Merge를 독립 kernel로 두는 한 OpenCL in-order queue에서 barrier로 동작.
3. **국소 async 전환은 무효**. 구조 변경(merge fusion)만이 실효 레버.

### Merge 자체 비용 분해 (baseline)

| 항목 | μs/layer |
|---|---|
| copy_slice(GPU→GPU, 6KB) | ~15 |
| copy_slice(CPU→GPU, 6KB) | ~25 |
| add_assign(GPU kernel) | ~10 |
| **실제 작업** | **~50** |
| kernel launch × 3 | ~80 |
| **측정 합계** | **130 μs (0.13 ms)** |

Merge "자체"는 싸다. 비용의 본질은 **별도 kernel로 enqueue된다는 사실 자체가 만드는 barrier**.

---

## 1. 설계 — Fused norm-merge

### 아이디어

다음 layer의 pre-attention RMSNorm kernel이 3-input 합산 + norm을 한꺼번에 수행:

```
기존:
  L_N end: copy_slice(gpu_partial→down) → copy_slice(cpu→staging) → add_assign(down,staging) → residual+=down
  L_{N+1}: RMSNorm(residual) → attn → ...

변경:
  L_N end: (merge step 삭제. gpu_partial은 그대로, cpu_partial은 async write_buffer로 staging에 도달)
  L_{N+1}: fused_norm_merge(prior_residual, gpu_partial, cpu_staging, weight, eps) → normalized → attn → ...
```

**GPU queue 관점**:
- Layer 경계 barrier 소멸 → plan path처럼 연속 chain
- Host synchronize 불필요
- Driver kernel prefetch 체인 유지

### 예상 이득 breakdown

| 성분 | 현재 | Fused | 회수 |
|---|---|---|---|
| Prior work 대기 (성분 A) | 5.6 ms | ~3 ms | 2.6 ms |
| clFinish ioctl (B) | 1.4 ms | 0 | 1.4 ms |
| Prefetch 상실 (C) | 2.8 ms | 0 | 2.8 ms |
| Re-dispatch (D) | 1.4 ms | 0 | 1.4 ms |
| Merge launch × 3 | 2.2 ms | 0 | 2.2 ms |
| **합계 회수** | | | **~10 ms** |

**이론 TBT**: 66.4 → 56 ms. Plan path(64ms) 대비 **-8 ms 순이득** (처음으로 partition이 plan을 이기는 구간).
**현실 예상 (할인 50-70%)**: 60-62 ms TBT, plan path 대비 -2~-4 ms.

### 필수 timing 조건

Layer N+1의 `fused_norm_merge` 실행 시점까지 `cpu_staging` buffer에 CPU partial이 도달해야 함.

- CPU matmul: 1.24 ms/layer
- write_buffer(6 KB, async): ~50 μs
- → CPU side 완료 ~1.3 ms
- GPU layer N FFN slice (r=0.7): 1.85 × 0.7 = 1.3 ms
- → 거의 동률. CPU가 10-20% 빨라야 drain 없음, 느리면 fused_norm_merge가 buffer를 기다리며 GPU stall (이 경우 기존 gpu_wait 수준)

---

## 2. 구현 단계 — Abort 조건 포함

### Step 0: Adreno async overlap 마이크로벤치 (1시간)

**목표**: Adreno OpenCL driver가 `enqueue_write_buffer(non-blocking) + enqueue_ndrange` 시퀀스에서 실제 compute/DMA overlap을 제공하는지 선행 검증.

**방법**:
- 간단한 test 바이너리: 6KB write_buffer(non-blocking) enqueue 후 1ms짜리 dummy kernel enqueue, clFinish 시간 측정
- 비교군: blocking write_buffer 후 kernel enqueue
- 차이가 <100 μs면 overlap 안됨 → **Step 1 이후 전체 중단, CPU NEON 최적화 선회**

**Abort 조건**: write_buffer가 진짜 async가 아니면 fused 설계의 timing이 깨짐. 마이크로벤치 결과 overlap 없으면 즉시 중단.

### Step 1: `fused_norm_merge` OpenCL 커널 + CPU fallback (0.5일)

**대상 파일**:
- 신규: `engine/kernels/fused_norm_merge.cl`
- 수정: `engine/src/backend/opencl/kernel_loader.rs` (로드 엔트리 추가)
- 수정: `engine/src/backend/opencl/mod.rs` (dispatch 함수)
- 수정: `engine/src/backend/cpu/mod.rs` (fallback 구현)
- 수정: `engine/src/core/backend.rs` (trait 메서드 추가)

**커널 서명**:
```c
__kernel void fused_norm_merge(
    __global const float* prior_residual,   // [hidden]
    __global const float* gpu_partial,      // [hidden]
    __global const float* cpu_staging,      // [hidden]
    __global const float* norm_weight,      // [hidden]
    __global float* out,                    // [hidden]
    __global float* residual_out,           // [hidden] — residual 갱신 (prior + gpu + cpu)
    float eps,
    int hidden
) {
    // 1) r = prior + gpu_partial + cpu_staging
    // 2) residual_out <- r
    // 3) out <- rmsnorm(r) * norm_weight
}
```

**Work-group 설계**: hidden=1536 → WG size 256, work-item 당 6 elem. 기존 `add_rms_norm_oop` 패턴 재사용.

**검증**:
- 단위 테스트: CPU backend 구현과 bit-identical (ULP 허용)
- `cargo test --lib -p llm_rs2 fused_norm_merge`

**Abort 조건**: 정확성 테스트 fail → 설계 재검토.

### Step 2: Forward_gen 구조 재편 — fused 경로 (기본 off 플래그) (0.5일)

**환경 플래그**: `LLMRS_PARTITION_FUSED_MERGE=1` (기본 off, off 시 기존 경로 완전 유지)

**대상 파일**: `engine/src/layers/transformer_layer/forward_gen.rs` partition 블록 (1044-1240)

**변경 요지**:
1. Partition 블록 끝의 merge 3-step 삭제 (fused 경로일 때)
2. Layer 진입부 pre-attention RMSNorm을 fused variant로 교체
3. CPU partial upload는 기존 `copy_slice(cpu→staging)`을 그대로 유지 (async 여부는 Step 3에서)
4. `synchronize()` 호출 제거 (merge barrier 없으므로)
5. Last layer(28) 경계 처리: 다음 layer 없으므로 final merge + 기존 lm_head 경로

**호출 지점 찾기**: `add_rms_norm_oop` 호출 위치를 pre-attention norm으로 대체.

**검증 — 중간 측정**:
- async 없이 fused만 (blocking write_buffer + 매 layer fused_norm_merge)
- 기대: sync_drain 일부 회수 (B+C+D 일부), A 성분 + CPU wait 여전히 존재
- 목표 TBT: 63-65 ms (일부 이득)
- **회귀 시 Abort**: fused kernel 자체가 기존 norm보다 느리면 설계 실패.

### Step 3: Async write_buffer + ping-pong staging (1일)

**추가 변경**: CPU→GPU upload를 non-blocking으로 + staging buffer 2개 교대

**대상 파일**: `engine/src/layers/workspace.rs` (staging buffer 2개), `forward_gen.rs`

**동작**:
- `cpu_merge_staging_A`, `cpu_merge_staging_B` ping-pong
- Layer N: A에 write_buffer(non-blocking), layer N+1 fused_norm_merge가 A 읽음
- Layer N+1: B에 write, layer N+2 fused 읽음
- Swap pointer per layer

**검증 — 최종 측정**:
```bash
# Baseline (현재 강제)
adb shell "cd /data/local/tmp && LLMRS_PARTITION_TRACE=1 ./generate \
  --model-path /data/local/tmp/models/qwen2.5-1.5b -b opencl --threads 6 \
  --no-gpu-plan --prompt-file prompts/prefill_128.txt \
  --tensor-partition 0.7 -n 128 --ignore-eos \
  --experiment-output /data/local/tmp/trace_out/baseline.jsonl \
  --experiment-sample-interval 10 --experiment-logits-topk 0 2>&1" | tail -30

# 30초 쿨다운

# Fused + async
adb shell "cd /data/local/tmp && LLMRS_PARTITION_TRACE=1 LLMRS_PARTITION_FUSED_MERGE=1 ./generate \
  --model-path /data/local/tmp/models/qwen2.5-1.5b -b opencl --threads 6 \
  --no-gpu-plan --prompt-file prompts/prefill_128.txt \
  --tensor-partition 0.7 -n 128 --ignore-eos \
  --experiment-output /data/local/tmp/trace_out/fused.jsonl \
  --experiment-sample-interval 10 --experiment-logits-topk 0 2>&1" | tail -30
```

**채택 기준**:
- TBT 개선 ≥ 3 ms → 기본 on 전환 검토
- 1-3 ms → diagnostic flag로 유지
- 미달 또는 회귀 → rollback

---

## 3. 측정 및 보고

### 필수 측정

1. **정확성**: Fused path vs baseline path의 생성 토큰 비교 (동일 prompt, 동일 seed)
2. **성능 breakdown**: sync_drain, cpu_matmul, gpu_wait, merge 각 변화
3. **Plan path 회귀 확인**: partition 비활성 시 64.15 ms 유지

### 비교표 템플릿

| 메트릭 | Plan | Partition baseline | Partition fused |
|---|---|---|---|
| sync_drain | — | 11.2 | ? |
| merge | — | 3.6 | 0 |
| TBT | 64.15 | 66.45 | **?** |
| Plan 대비 | 0 | +2.3 | **?** |

---

## 4. 리스크 및 완화

| 리스크 | 확률 | 영향 | 완화 |
|---|---|---|---|
| Adreno async write_buffer 미작동 | 중 | 높음 | Step 0 마이크로벤치로 선행 판정 |
| Fused kernel 정확성 | 중 | 높음 | Step 1 단위 테스트 |
| 3-input 커널 register spill | 낮 | 중 | hidden=1536, DK=128 spill 조건 해당 없음 |
| Ping-pong 버퍼 swap 버그 | 중 | 중 | 토큰 단위 정확성 검증 (생성 결과 비교) |
| Barrier가 다른 위치로 이전 (A1 재연) | 중 | 높음 | Step 2 중간 측정에서 회귀 시 즉시 rollback |
| CPU timing miss (stall) | 중 | 낮 | gpu_wait로 관찰, 허용 가능 수준 |

---

## 5. 관련 파일 참조

### 현재 상태 (수정 전)
- `engine/src/layers/transformer_layer/forward_gen.rs:1044-1240` — partition 블록
- `engine/src/layers/tensor_partition.rs:461-495` — trace 기록 + summary
- `engine/src/layers/workspace.rs` — `PartitionWorkspace` (gate_gpu, up_gpu, down_partial_gpu, down_partial_cpu, cpu_merge_staging)
- `engine/src/backend/opencl/mod.rs:2610-2660` — read_buffer + enqueue_read_buffer_async (A1 산물)
- `engine/src/core/backend.rs:5-28, 385-405` — GpuEvent + trait 메서드
- `engine/kernels/` — OpenCL 커널 모음

### 관련 커밋
- `5ca7ba0` — A1 async read (diagnostic only, default off)
- `fc1b033` — Strategy B whole-FFN slice + plan-gating

### 실측 결과
- `experiments/results/tensor_partition/` — 기존 sweep
- `/tmp/ratio_sweep_results.txt` — 2026-04-20 r sweep 결과

---

## 6. 착수 시 첫 명령

```bash
# 빌드 환경 확인
adb devices
ls models/qwen2.5-1.5b/
cat hosts.toml

# Step 0 마이크로벤치 준비 — 기존 generate 바이너리에 LLMRS_PARTITION_TRACE=1로 선행 측정
# (재확인: 현재 baseline이 여전히 66.45 ms인지)
bash -c "adb shell 'cd /data/local/tmp && LLMRS_PARTITION_TRACE=1 ./generate \
  --model-path /data/local/tmp/models/qwen2.5-1.5b -b opencl --threads 6 \
  --no-gpu-plan --prompt-file prompts/prefill_128.txt \
  --tensor-partition 0.7 -n 128 --ignore-eos \
  --experiment-output /data/local/tmp/trace_out/recheck.jsonl \
  --experiment-sample-interval 10 --experiment-logits-topk 0 2>&1' | tail -5"
```

---

## 7. 완료 시

- 커밋 메시지: `perf(tensor_partition): fused norm-merge kernel to eliminate layer barrier`
- 메모리 업데이트: `project_partition_fused_norm_merge_result.md` — 실측 결과 + 채택/비채택 여부
- `arch/tensor_partition.md` — Strategy B + fused path 업데이트
- 데스크톱 알림: `notify-send "llm.rs" "partition fused merge result: ..."`
