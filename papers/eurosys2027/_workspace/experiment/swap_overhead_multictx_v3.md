# Multi-Context Swap v3 측정 (정상 작동 확인)

- 디바이스: Galaxy S25 (R3CY408S4HN), 6T, OpenCL backend
- 모델: qwen2.5-1.5b-instruct f16 + q4_0 secondary, AOS layout, force-swap-ratio 0.9
- 프롬프트: "The quick brown fox jumps", n_tokens=200, ignore-eos
- 빌드: `target/aarch64-linux-android/release/generate` (mtime 21:09, HEAD `0c8951e` + 미커밋 host_ptr_pool/swap fix)
- 측정 일자: 2026-05-08, 21:12 ~ 21:20 KST
- 시나리오 6개 × 3 runs = 18 logs, **swap_errors = 0/18 (모두 통과)**

## 핵심 검증

이전 v1/v2에서 ocl 0.19의 `&Queue::as_ptr()` 잘못된 trait impl(`ClContextPtr`)로 cl_command_queue 대신 cl_context를 반환 → `clEnqueueMapBuffer failed (errcode=-36)` 25/25 fail. v3에서 `&CommandQueueCore` 명시 사용으로 수정 후 측정.

| 검증 항목 | v2 결과 | v3 결과 |
|---|---|---|
| swap error count (per-run) | 25/25 fail | **0/18 (전부 통과)** |
| `[LISWAP-3] host_ptr_pool active` log | 출력됨 (path 활성화) | 출력됨 (path 활성화) |
| Decode 완료 (crash 없음) | OK | **18/18 OK** |
| 출력 품질 (force-swap-ratio 0.9) | swap 실패 fallback → 평범한 영문 | swap 실제 동작 → 의도된 quality 저하 (정상) |

v2와 v3의 출력 양상 차이 자체가 **v3에서 swap이 실제로 발생함**을 입증.

## 결과 표 (decode_excl tok[0] 기준, mean ± stdev, n=3)

| # | 시나리오 | Avg TBT (ms) | Decode excl (ms/tok) | Per-tick swap (mean / p50 / max) | swap err |
|---|---|---|---|---|---|
| 1 | sync_baseline | 42.29 ± 3.04 | **31.43 ± 2.86** | 47.1 / 48.8 / 62.3 ms (n=75) | 0 |
| 2 | liswap1+3 singlectx (DMA-BUF) | 49.73 ± 10.72 | **38.96 ± 10.45** | 45.4 / 46.7 / 62.2 ms | 0 |
| 3 | liswap1 multictx | 51.07 ± 4.03 | **40.80 ± 2.20** | 42.3 / 47.0 / 59.9 ms | 0 |
| 4 | multictx + DMABUF_SYNC | 46.19 ± 8.80 | **34.98 ± 8.32** | 49.2 / 50.9 / 60.9 ms | 0 |
| 5 | multictx + skipfinish | 47.03 ± 7.89 | **36.02 ± 7.17** | 47.5 / 50.0 / 59.3 ms | 0 |
| 6 | multictx + sync + skipfinish | 46.26 ± 7.11 | **34.95 ± 6.56** | 50.1 / 52.7 / 62.4 ms | 0 |

per-tick latency: 25 tick × 3 run = 75 samples per scenario. swap-active window는 token 0~24 (`plan complete (started_at_token=0, finished_at_token=24)` 모든 시나리오 동일).

## 핵심 비교 (vs sync_baseline 31.43 ms/tok)

| 비교 | Δ ms/tok | Δ % |
|---|---|---|
| S2 liswap1+3 singlectx (DMA-BUF baseline) vs S1 | +7.53 | +24.0% |
| S3 liswap1 **multictx** vs S1 | +9.37 | +29.8% |
| S4 multictx + DMABUF_SYNC vs S1 | +3.55 | +11.3% |
| S5 multictx + skipfinish vs S1 | +4.60 | +14.6% |
| S6 multictx full (sync + skipfinish) vs S1 | +3.53 | +11.2% |

### 결정적 비교: S3 vs S2 (multi-context vs single-context, 둘 다 zero-copy + DMA-BUF backed)

```
S3 multictx:   40.80 ms/tok
S2 singlectx:  38.96 ms/tok
Δ:             +1.84 ms/tok  (+4.7%)
```

**판정: multi-context는 single-context 대비 약간 느림 (+4.7%).**

→ 사용자가 사전에 정의한 3단계 판정 중 **"3 > 2: multi-context overhead가 큼 (Worst)"** 에 해당. cl_context 분리가 driver-level isolation을 제공하지 않으며, 추가 컨텍스트 관리 오버헤드만 발생.

다만 stdev 고려 시 차이가 통계적으로 미미할 가능성도 있음:
- S2 stdev 10.45 (변동성 큼: 28.92 / 38.19 / 49.77)
- S3 stdev 2.20 (변동성 작음: 38.37 / 41.37 / 42.65)
- 95% CI 겹침: S2 [18.0, 59.9], S3 [36.4, 45.2] → 겹침 → **유의차 없음에 가까움**.
- 즉 S3 ≈ S2의 약화된 형태 (둘 다 baseline 대비 +25~30% overhead, 둘 사이엔 본질적 차이 없음).

## DMABUF_SYNC / skip_finish 옵션 효과

S3 (40.80) → S4 +sync (34.98): **−5.82 ms/tok (−14.3%)**. cache flush가 오히려 TBT를 낮춤. (해석: kernel이 stale data를 보지 않게 되어 retry/refetch 비용이 줄어들었거나, 통계적 변동의 일부.)

S3 (40.80) → S5 +skipfinish (36.02): −4.78 ms/tok. host pointer pool 반환 시 `clFinish` 생략 효과.

S3 (40.80) → S6 sync + skipfinish (34.95): −5.85 ms/tok. S4와 거의 같음 → 두 옵션은 독립적이지 않고, sync가 dominant.

다만 S4/S5/S6 모두 stdev가 7~8 수준으로 크고 (S3 stdev 2.20 대비) — 디바이스 thermal 또는 background activity 변동을 잡아냈을 가능성. **본 결과는 평균 기반 정성 해석에 그쳐야 함**.

## per-tick swap latency 관찰

- 모든 swap 시나리오 (S2~S6)에서 mean tick latency 42~50ms 범위. mmap_permute가 전체의 ~50% (`stages: prefault=0.2ms mmap_permute=20ms ... 나머지 0ms`).
- multi-context (S3)가 mean tick lat **42.3ms로 최저** — single-ctx (S2) 45.4ms 대비 −7%. swap 자체는 multi-context에서 약간 빠름.
- 그러나 forward TBT는 multi-context가 약간 느려 (+4.7%) — swap 단축 < forward 영향, 결국 net negative.
- → multi-context의 driver-level isolation이 swap upload 자체에는 작은 양성 효과를 주지만, OpenCL 컨텍스트 전환·동기화 비용으로 forward path 손실이 더 큼.

## paper 메시지 영향

이 측정으로 다음 결론이 강화됨:

1. **Multi-context는 Adreno 832 환경에서 silver bullet이 아니다.**
   사전 가설 "cl_context 분리가 driver-level isolation 제공" 은 **유의미한 근거 없음**으로 종결. S3 ≈ S2 (둘 다 +25~30% overhead).
   - 추가 가설 검증 (3 < 2)이 부정됨 → ablation 챕터에서 multi-context는 **negative result**로 보고.
2. **DMA-BUF baseline (S2)**: zero-copy + DMA-BUF heap 단독으로 baseline 대비 +24% overhead. swap 통합에서 unavoidable한 비용.
3. **DMABUF_SYNC + skip_finish 조합 (S6)**: best case 11.2% overhead. 단 stdev 큼 → 추가 측정으로 신뢰구간 좁힐 필요.
4. **mmap_permute가 per-tick swap latency의 ~50% 차지** — 향후 zero-copy 접근에서 가장 큰 최적화 대상은 secondary mmap의 page-fault + permute 단계 (multi-context와 무관).

## 이전 트랙 비교 맥락

- LISWAP-2 / Direction A: negative (Adreno multi-queue serialize, H2D 99.9% upload). v3 multi-context도 같은 카테고리에 추가 = **isolation 가설 자체가 Adreno에서는 약함**.
- Phase 6.5 메인 (HEAD `f3dc4a4`): −81% swap overhead 달성. 그것 대비 본 v3 측정의 multi-context는 진전을 만들지 못함.

## 한계

- per-token TBT 미수집 → "swap-active TBT (token 0~24 평균)" / "saturated TBT (token 30+)" 직접 분리 불가. `Decode(excl tok[0])`이 200 토큰 평균이라 swap window (25 tokens) 영향이 12% 가중치로 희석됨. 향후 측정에서는 per-token timestamp 활성화 권장.
- n=3은 stdev 7~10 수준에서 95% CI 폭이 매우 큼. 정량 결론에는 n≥10 또는 thermal-controlled bench가 필요.
- output token quality는 force-swap-ratio 0.9의 의도된 garbage이므로 정확성 비교는 swap_errors와 종료 코드에 한정.

## raw 데이터

`papers/eurosys2027/_workspace/experiment/multictx_v3_raw/` 18개 로그 + `runner.log`.

추출 metric:
```
v3_1_sync_baseline_{1,2,3}.log              tbt={43.31, 38.88, 44.69}
v3_2_liswap1_3_singlectx_{1,2,3}.log        tbt={49.02, 60.79, 39.38}
v3_3_liswap1_multictx_{1,2,3}.log           tbt={46.50, 52.62, 54.09}
v3_4_liswap1_multictx_sync_{1,2,3}.log      tbt={55.88, 38.70, 43.98}
v3_5_liswap1_multictx_skipfinish_{1,2,3}.log tbt={49.15, 53.64, 38.30}
v3_6_liswap1_multictx_full_{1,2,3}.log      tbt={47.24, 52.82, 38.71}
```
