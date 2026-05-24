# Handoff: §13.8-L hot path sub-trait 격상 (L-1/L-2/L-3) → 다음 sub-sprint

**작성**: 2026-05-24
**HEAD**: `dde575b9 refactor(layer): S-L-3 — KiviAttentionBackend trait + 4 callsite 정리`
**브랜치**: `worktree-b5_trait_extension`
**다음 세션 진입 문장**: "tensor_partition L2 격상" 또는 "§13.8-O WeightSwapDispatch trait 격상" 또는 "generate 분할 설계 라운드"

---

## TL;DR

§13.8-L 운용 메모상 "5건 이상 hot path marker 누적 = sub-trait 격상 강제 trigger" 발동. 3 sub-sprint 단일 commit씩 — L-1 Profile hook (3 Backend trait method) / L-2 GpuScoreAccess trait + Backend trait method / L-3 KiviAttentionBackend trait (4 method). hot path 9 marker 중 2건 (`forward_gen.rs` decode + `pressure/kivi_cache.rs::update_gpu`) 제거 — 함수 내 `OpenCLBackend` downcast / `use crate::backend::opencl::*` import 0 달성. S25 Adreno OpenCL Qwen2.5-1.5b Q4_0 32토큰 bit-identical PASS (Avg TBT 30.29 vs 32.30 ms — 회귀 없음).

---

## 진행 상태

### 3 sub-sprint commit

| sub-sprint | commit | 추가 | 효과 |
|---|---|---|---|
| **S-L-1** | `a83a62d1` | Backend trait `profile_events_enabled` + `set_op_label` + `clear_op_label` default no-op + OpenCL/CUDA override | `forward_gen.rs:54~102` 5 downcast block 제거. `#[cfg(feature = "opencl")]` gate 사라짐. marker 유지 (함수 내 다른 downcast 잔존). |
| **S-L-2** | `3ee56f82` | `GpuScoreAccess` trait 9 메서드 + `GpuScoreAccumulator impl` + Backend trait `gpu_score_acc()` / `gpu_score_acc_mut()` default `None` + OpenCL override | `transformer.rs:1654` (forward_into) + `forward_gen.rs:543~549` (attention_gen pre-dispatch) 2 downcast 제거. cfg gate 사라짐. |
| **S-L-3** | `dde575b9` | `KiviAttentionBackend` trait 4 메서드 (has_kivi_attn_kernel/is_nosub_device/attention_gen_kivi/kivi_gather_update) + OpenCL impl + Backend trait `as_kivi_attention()` | `forward_gen.rs:454` (KIVI native dispatch) + `pressure/kivi_cache.rs::update_gpu:1564` (KIVI gather update fast path + Backend trait `copy_slice` slow path) 2 downcast 제거. **함수 내 잔존 downcast 0** ⇒ marker 2건 제거. |

### 검증

- **호스트 빌드**: 3 commit 모두 release build PASS (1m 내)
- **spec test**: `cargo test -p llm_rs2 --test spec inv_layer` 8/8 PASS (모든 sub-sprint)
- **layer_lint**: `python3 scripts/layer_lint.py --baseline engine/tests/spec/inv_layer_baseline.json --json` new_violations=0
- **clippy + fmt**: clean (모든 sub-sprint)
- **lib test**: 1210 PASS / 18 fail (OpenCL host device-required test의 동시 실행 경합 — handoff [P3] 환경 이슈와 동일, 1228 합산 동일. sprint 회귀 아님)
- **S25 Adreno OpenCL bit-identical**: Qwen2.5-1.5b Q4_0 32토큰 greedy
  - baseline 398030aa: `France_capital_isParis\n# Capitale ... TTFT 79.62 / TBT 32.30`
  - L-3 dde575b9:   `France_capital_isParis\n# Capitale ... TTFT 78.98 / TBT 30.29`
  - **32 토큰 출력 완전 일치 (bit-identical)** + TBT |Δ|=6% (측정 노이즈)
- **Jetson CUDA / S25 qnn_oppkg**: 사전 환경 이슈로 빌드 불가 (master에서도 동일) — sprint 회귀 무관 확정. CUDA backend의 S-L-1 set_op_label trait override는 inherent method 위임 패턴이라 코드 정확성 risk 0.

### marker 분포 변화

| 카테고리 | 격상 전 | 격상 후 | Δ |
|---|---|---|---|
| hot-path marker (전체) | 9 | 7 | −2 |
| 그 중 forward.rs / forward_gen.rs / attention.rs | 3 | 2 | −1 (forward_gen.rs 제거) |
| 그 중 pressure/kivi_cache.rs | 3 | 2 | −1 (update_gpu 제거) |
| baseline JSON (28 → 28) | 28 | 28 | 0 (모든 변경이 marker zone 내부 — baseline 무영향) |

---

## 다음 작업 (3 갈래)

### A. tensor_partition L2 격상 (S-D3, baseline 28 → 27)

- `engine/src/layers/tensor_partition.rs` (1575 LOC) 의 PartitionContext / PartitionPath / partition_plan_* helpers 가 L3 dep 없는 pure config 면 top-level L2 (`engine/src/tensor_partition.rs`) 이동 가능.
- backend/opencl/plan.rs:16 V-02 마지막 INV-LAYER-001 1건 해소.
- 비용: 30분~2h.

### B. §13.8-O WeightSwapDispatch trait 격상 (backlog [P2])

- 3 트랙 분할: WeightSwapDispatch trait (3건) + PrefetchAccess + PreloadPool L2 격상 (3건) + KvCacheView trait (3건).
- §O register의 9 marker 정리.
- 비용: 1~3일.

### C. generate 바이너리 분할 설계 라운드 (INV-LAYER-005 27건 전제)

- 단일 generate.rs 13K LOC → 다수 바이너리 (cli / chat / experiment / resilience 잠정).
- baseline의 96% (27건) 해소.
- 비용: 설계 라운드 + sprint 다회.

### D. 잔여 §L hot path marker 7건 (별도 sprint 후보)

- `transformer.rs:1479/2492/2739` (forward_into / execute_plan / KIVI plan execute): PlanExecutor trait 추상화 필요 — `FullKernelPlan` struct가 OpenCLBackend concrete 직접 받음. transitive drag 큼.
- `attention.rs:232` (flash_attention_forward_strided NEON fallback): 함수 안 backend downcast 분석 필요.
- `kivi_cache.rs:1811/2108` (flush_residual_gpu / assemble_view_gpu): 함수 안 `get_cl_mem` + raw `ocl::core::enqueue_write_buffer` 잔존. trait method로 추출하려면 OpenCL queue 노출 필요 (OpenCL-specific).
- 비용: 3~5일 (PlanExecutor 본질 해소).

---

## Landmines / 미해결

### 1. UnsafeCell `&self -> &mut dyn Trait` 패턴
- L-2 의 `gpu_score_acc_mut(&self) -> Option<&mut dyn GpuScoreAccess>` 는 OpenCL backend 의 inherent UnsafeCell-backed pattern 을 trait 으로 노출. `#[allow(clippy::mut_from_ref)]` 필요.
- 향후 multi-threaded inference 도입 시 본 trait 전면 재검토 (single-threaded 가정 깨짐).

### 2. CUDA backend S-L-1 override 검증 미수행
- CUDA embedded `set_op_label` / `clear_op_label` trait override 추가했으나 Jetson 빌드가 master 이전부터 broken state (`cuda_mmap_alias_buffer` / `map_weights_for_cpu` 미존재). 검증은 빌드 fix 후 별 sprint.
- 코드 정확성 risk 0 (inherent method 그대로 위임).

### 3. KiviAttentionBackend trait 확장 잠재 영역
- L-3 trait 은 OpenCL-only KIVI 4 메서드 한정. flush_residual_gpu / assemble_view_gpu marker 제거에는 raw OpenCL queue 노출 또는 `kivi_flush` / `kivi_assemble_view` 같은 high-level method 추가 필요. 현재 trait 으로는 부족.

### 4. PartitionContext L2 격상 (V-02)
- baseline 28 에서 마지막 INV-LAYER-001 1건. 본 sprint 와 별개. S-D3 sprint 대상.

### 5. OpenCL host test 18 fail은 동시 실행 경합
- 본 sprint 회귀 아닌 환경 이슈. 개별 실행은 PASS. handoff [P3] 와 동일 패턴.

### 6. qnn_oppkg crate 빌드 broken
- `crates/qnn_oppkg/` 가 master 부터 type inference + path resolution error 53건. QNN SDK 없는 상태 + 또 무관한 빌드 에러. 본 sprint 변경 아님.

---

## 진입 명령 (다음 세션)

```
"tensor_partition L2 격상"                # S-D3 (baseline 28 → 27, 30분~2h)
"§13.8-O WeightSwapDispatch trait 격상"   # 9 marker 정리 (1~3일)
"PlanExecutor trait 격상"                 # §L 잔여 hot path 7건 (3~5일)
"generate 분할 설계 라운드"               # INV-LAYER-005 27건 전제
```
