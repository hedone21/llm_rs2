# Handoff: HTP async batch fusion 통합 진입 (PoC 확정 → backend 통합)

**작성**: 2026-06-05 (메인 세션 오케스트레이터)
**HEAD**: `d84d432a` (사용자 ADR 커밋; HTP work 와 ADR/techniques M2-M4 가 interleave)
**브랜치**: `master` — **origin 대비 14 커밋 미push** (내 HTP + 사용자 ADR/techniques 혼재)
**관련 HTP 커밋**: `24fa53c7`(op microbench) · `568ada7e`(fusion 조사) · `b54855ae`(batch microbench) · `316a19df`(PoC 결과). 앞선 push 분: F16 dispatch `65c147c5`, device 검증 `7ae77e85` 등.

**다음 세션 진입 문장**: "HTP async batch fusion 통합 — matmul_transposed_via_htp enqueue/flush 분리부터"

---

## TL;DR

HTP graph fusion 의 핵심 전제(**async queue batching 으로 ~100µs dispatch floor 회수**)가 PoC 로 **device-확정**됐다(N=28 2.8× faster, per-op 146→47µs, 정확성 보존). 다음 = production backend 에 enqueue/flush 분리 후 **FFN(KV-free) 블록을 layer-loop 에서 batch 통합**. 멈춘 이유 = production 코드 변경은 PoC 대비 위험(async buffer lifetime / ordering / 정확성 회귀)이라 사용자가 handoff 후 신중 진입을 택함.

---

## 진행 상태 — 이번 세션 완료 (전부 device 실측)

| 작업 | 결과 | commit |
|---|---|---|
| S4+S5+Y device 검증 (D/1/2 게이트) | 🟢 전 GREEN (prof_usecs=159, OOM 0, token-id 16/16) | `7ae77e85` |
| CPU/GPU/NPU × {q4,f16} × {prefill,decode} 매트릭스 | GPU>CPU>NPU; Q4 decode>F16(BW) | `8e7a64b9` |
| F16 weight NPU dispatch (default-on) | OOM 0(rpcmem 3.3GB OK), 16/16 정합, decode 6.9 tok/s(최저, memory-bound) | `65c147c5` |
| HTP op microbench (8 op) | **~100µs dispatch floor 발견**. 경량 op CPU 6~2600× 승, softmax(long-ctx)만 NPU 1.4× | `24fa53c7` |
| graph fusion 타당성 조사 (Workflow 4각) | feasible-with-caveats: async batching 가능, true graph op 불가(v79 skel) | `568ada7e` |
| **batch-dispatch PoC** | **floor 회수 확정: N=7 r0.44 / N=28 r0.36, err 2.2e-2** | `b54855ae`+`316a19df` |

**측정 핵심**: per-op dispatch 비용 sync 146µs → batch 47µs (회수 ~100µs = 예측 floor 일치). matmul-only batch ~31ms/tok(1.5×), 전 NPU-op fusion best-case ~11~16ms/tok(3~4×, 추정 medium). **단 floor 제거지 NPU 절대속도 향상 아님**(여전히 CPU 1.4× 느림).

---

## 다음 작업 (PoC 통과 → 통합, 순서대로)

1. **backend enqueue/flush 분리** — `htp_fastrpc.rs::matmul_transposed_via_htp` 의 dspqueue_write→read 쌍(현 ~258-322행)을 `enqueue()`(write + op_pending.fetch_add) / `flush()`(op_pending 0까지 drain) 로 분리. 현 no-op `synchronize()`/`flush()` 를 drain 지점으로 활용. → **검증**: 단일 op 경로 token-id 16/16 회귀 0 (기존 동기와 동일 결과).
2. **FFN(KV-free) 블록 layer-loop batch** — gate/up/down matmul + rmsnorm + silu_mul 을 한 batch 로 enqueue → 블록 끝 flush. **attention/kv_scatter/eviction 은 CPU 유지**(R6 참조). → **검증**: decode TBT 가 floor 회수만큼 감소 + token-id 16/16 유지.
3. **device e2e 재측정** — argus_cli f16/q4 decode TBT 전후 비교 (목표: matmul-only ~31ms 근처) + 정확성.

**위임**: `senior-implementer`(backend async dispatch + buffer lifetime) + 메인 세션(device 측정, 직렬).

---

## Landmines / 미해결

- **★ async in-flight 버퍼 lifetime**: `RpcmemBuffer` single-owner RAII(Clone 차단). batch flush 완료까지 N개 dst buffer 동시 alive 필요 — 소유 구조 설계 주의(PoC 는 Vec<RpcmemBuffer> 로 해결).
- **★ 데이터 의존성 ordering**: layer 내 op 간 dependency(matmul→add→rmsnorm). dspqueue FIFO 가 DSP 처리 순서는 보장하나, dst→다음 src 의 **cache invalidate 타이밍** 검증 필수. batch 내 중간 buffer 는 host coherency 불요(DSP-local)지만 경계 buffer 는 flush/invalidate 필요.
- **★ 정확성 회귀**: 현 16/16 token-id 는 strict 1:1 동기에서 확보. async 전환 후 race/ordering → DSP OOB 가 **crash 아니라 wrong-answer**(spec INV-HTP-FRPC-003). device 재검증 필수.
- **★ attention/KV-touching op 은 fuse 금지**: attention 을 DSP 로 fuse 하면 DSP 커널이 eviction/KIVI/D2O 가 만든 KV dtype/layout/position 과 **강결합**(현 attention 이 CPU 인 이유). fusion 경계를 **KV 바깥**(FFN 등 KV-free 블록)에 둘 것.
- **★ true graph op packet 불가**: `HtpGeneralReq` 312B 단일 op assert. newer master 의 `htp_opbatch_req`/256-op/`flush_batch` 따라가면 device v79 binary 와 **schema mismatch → silent garbage**. **v79 의 op_pending+flush async pipeline 만** 사용.
- **ADSP_LIBRARY_PATH 필수**: `--backend htp`/microbench 실행 시 `ADSP_LIBRARY_PATH=/data/local/tmp` 없으면 `remote_handle64_open` SKIP. run_device.py 미설정 → 수동 adb (레시피 = `[[htp-device-verification-green]]` 메모리 / phase1 handoff).
- **eviction+htp 자체 미검증**: 모든 htp run 이 resilience off(KV 그냥 grow). fusion 전에 "현 htp backend 에서 eviction 정상" 부터 확인 권장.
- **F16 NPU default-on**: 현재 f16+htp 가 F16 weight 전부 rpcmem(3.3GB) + NPU dispatch (decode 6.9 tok/s, 느림). 의도된 동작(사용자 결정).
- **14 커밋 미push** + 사용자 ADR/techniques 와 interleave — push 시 함께 나감.
- **서사**: NPU < CPU 절대속도(20.9<30 tok/s). fusion 이득 = "floor 페널티 제거"지 "빠르다" 금지.

---

## 참고 링크

- 메모리: `[[htp-device-verification-green]]` (env 레시피 + 전 결과 + fusion PoC 요약), `[[htp-production-backend-feasibility]]`
- 조사/PoC 문서: `papers/eurosys2027/_workspace/experiment/htp_graph_fusion_feasibility_2026_06_05.md` (verdict + PoC 섹션)
- 실험: `.../experiment/{backend_phase_matrix, htp_op_microbench, htp_batch_dispatch_poc}_2026_06_05/` (figure + raw)
- PoC 계측기: `engine/microbench/htp_batch_dispatch.rs` (enqueue/flush 분리 참조 구현)
- backend: `engine/src/backend/htp_fastrpc.rs` (matmul_transposed_via_htp, copy_weight_from, dispatchable×2)
- GREEN reference: `engine/microbench/htp_matmul.rs::run_htp`
- 설계: `arch/htp_fastrpc.md`, `spec/htp_fastrpc.md` (INV-HTP-FRPC-001~005)
