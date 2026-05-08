# LISWAP-4 v3 측정 결과 — Plan-path 우회 후 intra-forward swap 평가

- **일자**: 2026-05-08
- **디바이스**: Galaxy S25 (SM-S931N), serial R3CY408S4HN
- **빌드**: HEAD (`f3dc4a4` + 미커밋 patch: `generate.rs:5115`에 `&& !args.swap_intra_forward` 가드 추가)
- **모델**: Qwen2.5-1.5B-Instruct, primary F16 GGUF + secondary Q4_0 GGUF (`--secondary-layout aos`)
- **시나리오**: 3개 × n=3, 200 토큰 (`--ignore-eos`), prompt = "The quick brown fox jumps", 6 threads, OpenCL backend
- **공통 플래그**: `--force-swap-ratio 0.9` → 28레이어 중 25개 swap 대상

---

## 0. 변경 동기

이전 LISWAP-4 측정에서 `IntraForwardSwapHook`이 **한 번도 호출되지 않는** 결함이 있었다. 원인: `forward_into`의 layer loop는 `gpu_plan.is_none()`일 때만 사용되며, 평소 OpenCL backend에서는 plan path가 활성이라 hook이 우회됨. 이번 v3에서는 `--swap-intra-forward` 활성 시 plan을 강제로 비활성화 (`gpu_plan = None`) → forward_gen layer loop가 사용 → hook이 매 layer 종료 시 호출되도록 보장.

또한 plan path의 perf 영향을 분리하기 위해 `sync_plan_off` baseline (`--no-gpu-plan`)을 추가하여 3-way 비교를 수행.

---

## 1. Raw 결과 요약

### 1.1 Decode TBT (excl tok[0]) — settle 후 토큰당 시간

| 시나리오 | n | mean (ms/tok) | std | runs |
|---|---|---|---|---|
| sync_plan_on  | 3 | 26.72 | 0.091 | 26.73, 26.80, 26.62 |
| sync_plan_off | 3 | 26.64 | 0.090 | 26.55, 26.64, 26.73 |
| liswap4_v3    | 3 | **26.95** | 0.042 | 26.98, 26.96, 26.90 |

**핵심**:
- `sync_plan_off` ≈ `sync_plan_on` (Δ −0.29%) → S25 Qwen2.5-1.5B에서 plan path는 settle TBT에 영향 없음.
- `liswap4_v3` vs `sync_plan_off`: **+0.31 ms/tok (+1.15%)** — 무시 가능한 수준이지만, 일관되게 약간 느림.
- `liswap4_v3` vs `sync_plan_on`: +0.23 ms/tok (+0.86%).

### 1.2 tok[0] — swap+첫 디코드 통합

| 시나리오 | mean (ms) | std | raw |
|---|---|---|---|
| sync_plan_on  | 26.56 | 0.23 | 26.52, 26.35, 26.81 |
| sync_plan_off | 27.14 | 0.70 | 27.92, 26.95, 26.56 |
| liswap4_v3    | **578.96** | 21.04 | 603.25, 566.45, 567.18 |

**Δ tok[0] (liswap4_v3 − sync_plan_on) = +552.4 ms** — swap 비용 거의 전부가 tok[0]에 흡수됨.

### 1.3 TTFT

| 시나리오 | mean (ms) | std | 해석 |
|---|---|---|---|
| sync_plan_on  | 531.38 | 49.68 | swap (≈350ms) + prefill (≈104ms) + 오버헤드 |
| sync_plan_off | 538.23 | 18.21 | 동일 (plan 무관) |
| liswap4_v3    | 345.35 |  1.15 | prefill만 (swap은 tok[0]로 미뤄짐) — std 매우 안정적 |

**TTFT 자체는 짧아 보이지만**, swap 비용이 tok[0]로 이동했을 뿐 → 실제 사용자 경험상 "첫 토큰까지" 시간 = TTFT + tok[0]:

| 시나리오 | TTFT + tok[0] (ms) | vs sync_plan_on |
|---|---|---|
| sync_plan_on  | 557.94 | (baseline) |
| liswap4_v3    | **924.31** | **+366.4 ms (+65.7%)** |

---

## 2. DIAG 로그 분석 — hook 동작 검증

3 runs 모두 동일 패턴:

```
[DecodingStart]
The quick brown fox jumps over[IntraForwardSwap-DIAG] token=0 pending_layers=0 dispatcher_pending=0 plan_complete=true
[IntraForwardSwap] plan retired at token=0 (drain+sync+bump+invalidate 0.0ms)
```

### 2.1 발견 사항

- **DIAG는 token=0에서 단 한 번만 출력됨** — `pending_layers=0`, `dispatcher_pending=0`, `plan_complete=true`.
- **plan retire가 즉시(0.0ms) 발생** — drain/sync/bump/invalidate 비용 0.
- **이 의미**: token=0의 forward 시작 시점에 이미 모든 25개 layer의 swap이 완료되어 있음. dispatcher는 prefill 또는 그 직전(`weight_swap` 헬퍼 등록 시점)에 모든 작업을 완료.
- 결과적으로 **intra-forward "overlap" 동작이 실효적으로 일어나지 않음** — swap이 GPU 동시 수행 중에 진행되는 것이 아니라, prefill 전후의 dispatcher 큐에서 sync 누적.

### 2.2 timeline 추정

`liswap4_v3`의 prefill = 221.4 ms vs `sync_plan_on`의 prefill = 104.4 ms (+117 ms) 차이가 있고, tok[0] = 579 ms vs 27 ms (+552 ms). 합산:
- liswap4_v3: prefill(221) + tok[0](579) = 800 ms
- sync_plan_on: swap(352) + prefill(104) + tok[0](27) = 483 ms

→ **liswap4_v3가 더 비쌈**. plan path 사용 못 하는 비용 + dispatcher가 forward와 진정한 병렬을 달성 못 하는 비용이 합산됨.

---

## 3. 판정

| 항목 | 결과 |
|---|---|
| settle TBT 차이 (vs sync_plan_off) | +1.15% (≈ noise) |
| swap retire 시점 | token=0 (즉시) — overlap 미발생 |
| swap 비용 위치 | tok[0]로 이동 (총 비용은 오히려 +366 ms) |
| **종합 판정** | **NEGATIVE** |

### 3.1 가설 결과

- ❌ **Strong positive 미충족**: swap 비용이 GPU 작업과 overlap되지 않음. 즉 pending_layers가 점진적으로 감소하는 양상이 관측되지 않고, prefill 단계에서 이미 완료.
- ❌ **Partial positive 미충족**: settle 후 TBT inflation 자체는 미미하나, tok[0]에 swap 비용 전부가 누적되어 총 사용자 latency 증가.
- ✅ **Negative 확정**: hardware/driver 한계로 dispatcher의 H2D upload + permute가 GPU forward와 진정 병렬 실행되지 않음. **LISWAP-2의 "Adreno multi-queue serialize + H2D 99.9% upload" 결론을 v3에서 재확인**.

### 3.2 LISWAP-1/2 결과와 일관성

- LISWAP-1 (incremental): 옵션 적합 (token-spread).
- LISWAP-2 (background): negative — driver serialize.
- LISWAP-4 v1/v2 (forward-coupled): hook 미호출로 무효 측정.
- **LISWAP-4 v3** (plan 우회, hook 정상): swap이 forward와 overlap되지 않음. dispatcher 큐가 forward GPU 작업과 직렬화되어 prefill 직후 모든 swap이 완료되며, tok[0]에 sync penalty가 흡수.

---

## 4. 추가 관측 (참고용)

- liswap4_v3는 `weight_swap stages` 통계 라인이 출력되지 않음 (sync 모드는 `prefault/mmap_permute/...` 분해 출력). intra-forward 모드는 stage timer가 forward path로 분산되어 합산 출력 없음.
- liswap4_v3 prefill +117 ms는 forward_gen path 고유 오버헤드 + dispatcher contention 가능성. sync_plan_off도 forward_gen path지만 prefill ~104 ms (sync_plan_on과 동일) → contention이 원인일 가능성이 큼.
- `Avg TBT`: liswap4_v3 = 34.7 ms (tok[0] 포함), sync_plan_on = 31.7 ms (tok[0] 포함). 200토큰 평균 +3 ms/tok = liswap4_v3의 tok[0] 580ms / 200tok ≈ 2.9ms. 분배 일치.

---

## 5. 결론 / 권장

1. **LISWAP-4 (intra-forward swap)는 S25 Adreno 환경에서 효익 없음**. 사용자 측 총 latency는 sync 모드보다 +66% 악화.
2. 본질 원인: dispatcher의 H2D + mmap_permute 작업이 GPU forward와 overlap되지 않아 forward 진입 시점에 sync 비용이 누적됨. 이는 LISWAP-2의 결과와 일관 (Adreno multi-queue serialize, H2D 대역폭 99.9% 점유).
3. **paper 메인 메시지 변경 불요**: Phase 6.5 sync 모드 (-81%) + LISWAP-1 (incremental, token-spread)이 최선. LISWAP-4 ablation은 negative 보강 자료로 활용.
4. **다음 단계**: 별도 큐 / non-blocking enqueue / vendor-specific async copy API 등 driver-level 우회 외에는 software 영역 추가 시도 무효.

---

## 6. Raw artifact

- `papers/eurosys2027/_workspace/experiment/liswap4_v3_raw/sync_plan_on_run{1,2,3}.log`
- `papers/eurosys2027/_workspace/experiment/liswap4_v3_raw/sync_plan_off_run{1,2,3}.log`
- `papers/eurosys2027/_workspace/experiment/liswap4_v3_raw/liswap4_v3_run{1,2,3}.log`

