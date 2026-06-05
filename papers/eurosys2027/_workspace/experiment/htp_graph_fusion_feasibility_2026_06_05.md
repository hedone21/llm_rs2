# HTP layer graph fusion 타당성 조사 (2026-06-05)

**방법**: Workflow 4각 병렬 조사(IDL/protocol · upstream skel · repo prior-art · 이득 정량화) + 종합. 조사 agent 가 실제 skel 소스(`/home/go/Workspace/llama.cpp`, Mar 13 빌드)까지 ground-truth 검증.

**Verdict: `feasible-with-caveats`** — 단, "true graph op packet" 이 아니라 **async queue batching** 으로.

---

## 핵심 발견 (decisive)

1. **true single-packet graph op = v79 skel 로 영구 불가.** `HtpGeneralReq` 는 단일 op schema (op:u32 + src0~4 + dst, **312B compile-time assert**, `idl.rs:153-225`). batch schema(`htp_opbatch_req`/256-op/`flush_batch`)는 **newer upstream master 에만** 있고 device 의 `libggml-htp-v79.so` binary 에는 **부재**. graph op 원하면 skel 재빌드(Hexagon SDK, β scope) 필수.

2. **그러나 async queue batching 은 현 v79 skel + 현 IDL 로 가능.** 조사가 실제 skel 소스를 확인:
   - DSP callback (`main.c:1007 htp_packet_callback`)이 `dspqueue_read_noblock` while-loop 로 큐를 drain — **host round-trip 없이** 연속 packet 처리, op 당 응답 1개 발행(`main.c:372`).
   - upstream host shim(`ggml-hexagon.cpp:141`)이 **동일 단일 op schema + 동일 v79 binary** 로 `enqueue(write, op_pending++)` → `flush(while op_pending: read)` = "**N writes → 1 drain**" 패턴을 이미 실증.
   - 즉 메커니즘은 256-op batch packet 이 아니라 **v79 의 op_pending + flush async pipeline**.

3. **현 llm.rs 는 이 async 성을 전혀 안 씀.** 매 op `dspqueue_write` 직후 `dspqueue_read` 10s blocking = strict 1:1 동기 round-trip (`htp_fastrpc.rs:158-198, 280-323`), `synchronize`/`flush` 는 no-op(`:776,784`). **host-side 재구성만으로 floor 누적 회수 가능** (skel 재빌드 불요).

---

## 이득 정량화 (angle-4)

| 시나리오 | floor 누적 | best-case decode |
|---|---|---|
| 모든 op 개별 NPU dispatch (naive) | 레이어당 15~18 dispatch × 100µs × 28 = **42~50ms/tok 순수 floor** | 현 48ms 와 비슷하거나 나쁨 → **순손실** |
| matmul-only async batch (7 write→1 flush) | (7-1)×100µs×28 = ~16.8ms 회수 | **~31ms/tok (~1.5×)** |
| 전 NPU-op layer fusion (1 flush/layer) | 42~50ms → **2.8ms** | **~11~16ms/tok (~3~4×)**, confidence medium |

**★ caveat**: 이 이득은 "NPU 가 CPU 보다 빨라져서"가 **아니라** "dispatch floor 누적 제거". NPU 절대속도는 여전히 CPU 의 ~1.4× 느림(20.9 vs 30.0 tok/s). paper 서사 = "heterogeneous 협력 시 NPU leg 의 latency floor 페널티 제거"지 "빠르다" 아님.

**손익분기**: 레이어당 NPU dispatch ≥ 2회 + op 당 평균 compute < ~200µs 구간에서 fusion 이 floor 누적을 상쇄. 현 matmul-only backend(레이어당 7 dispatch = 19.6ms/tok 순수 floor)는 이미 분기점을 한참 넘어 **"matmul 들만 한 batch 로 묶는 부분 fusion"이 최고 ROI**.

---

## prior-art (angle-3)

- QNN-GPU OpPackage 에서 chain/layer fusion + amortization **이미 실증** (단일 op 1.45× → N16 0.874×, 14-node Qwen layer 단일 graph warm 1.3ms/layer, graph reuse PASS). **단 거의 전부 Adreno GPU 경로지 HTP FastRPC 아님** — 수치 직접 전이 불가.
- in-place op(SiluMul/RoPE/KvScatter)이 multi-node chain composition 을 깸 → QNN-GPU 는 OOP variant + multi-output 으로 우회. **fusion 시 동일 함정 주의**.

---

## 권장 다음 단계 (검증 가능 PoC)

`matmul_transposed_via_htp` 를 **enqueue/flush 분리**:
- (a) enqueue: `dspqueue_write` + `op_pending.fetch_add(1)`
- (b) flush: `op_pending` 0 될 때까지 `dspqueue_read` while-loop (현 no-op `synchronize():776` 을 여기로)

microbench(`engine/microbench/htp_matmul.rs`)에서 **"7회 연속 write → 1회 flush" vs "7× (write+read)"** wall-clock 을 S25 6T n=3 비교.

**검증 게이트**: (1) 7-op batch wall-clock < 7× 개별의 **80%** (floor 회수 실증), (2) max_abs_err < 5e-2 (정확성). PASS → transformer layer-loop 통합 확대.

**effort**: ~2-3주, host-side only, skel 재빌드 불요.

---

## Risks

- best-case 11~16ms 는 confidence medium (floor/compute 비중첩 가정 — FFN matmul compute 가 크면 이득 축소).
- async in-flight N개 dst buffer 의 aliasing/coherency: 현 `buffer.rs` single-owner RAII(Clone 차단) — N in-flight lifetime + cache flag(CpuWriteDspRead/DspWriteCpuRead) 관리가 1:1 동기보다 까다로움. layer 내 data dependency(matmul→add→rmsnorm) 순서는 dspqueue FIFO 가 보장하나 dst→다음 src cache invalidate 타이밍 검증 필요.
- 정확성 회귀: 현 16/16 token-id 는 strict 1:1 동기에서 확보. async 전환 후 race/ordering → DSP OOB=wrong-answer(crash 아님). device 재검증 필수.
- **schema 함정**: newer-master 의 `htp_opbatch_req`/256-op/`flush_batch` 를 그대로 따라가면 v79 binary 와 schema mismatch → silent garbage. **반드시 v79 의 op_pending+flush 방식만**.

---

**조사 산출물**: Workflow `wf_b361ab5e-e42` (5 agent, 490K tok, 149 tool calls). raw = task output.
