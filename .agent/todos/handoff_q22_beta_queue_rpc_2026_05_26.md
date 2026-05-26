# Handoff: Q-2.2-β.QUEUE.RPC 종료 — 3 host-side 가설 모두 기각, DSP-side architectural barrier 확정 → Q-2.2 decision gate

**작성**: 2026-05-26
**HEAD**: `05a70824` (β-1.QUEUE) — 본 sprint 분석만, 코드 변경 없음
**worktree**: `.claude/worktrees/b5_trait_extension`
**다음 세션 진입 문장**: **"Q-2.2 NPU track decision — 옵션 A (Hexagon SDK signed test signature 시도) / B (NPU track 종료 + 결과 paper 에 기록 + OpenCL 집중) / C (llama.cpp 의 stock S25 동작 검증으로 wall universality 확인) 중 선택"**

---

## TL;DR

β-1.QUEUE.RPC 의 3 host-side 가설 (RPC.1 packet schema / RPC.2 sess_id 의미 / RPC.3 init sequence 누락) **모두 기각** — 우리 코드는 llama.cpp local build (`983df14`) 와 byte-identical schema + 동일 lifecycle sequence.

logcat trace 의 timing 분석으로 fail 지점 확정: **`libdspqueue_rpc_skel.so` (driver-side dspqueue transport stub) 의 method 3 reject**. 우리 compute module (`libggml-htp-v79.so`) 의 packet handler 진입 0회 — driver-level 의 PD-policy wall.

본 sprint = β-1.QUEUE.RPC scope (host-side fix 후보) **소진**. β-2.MM (matmul) 진입 무의미 (rmsnorm 도 안 가는 wall 에 matmul 시도 동일 fail).

Q-2.2 NPU track 의 decision gate 도달. 3 후속 옵션 중 선택 필요.

---

## 진행 상태

| RPC | 가설 | 결과 | 근거 |
|---|---|---|---|
| RPC.1 | HtpGeneralReq 312B mismatch | ✗ 기각 | local llama.cpp htp-msg.h 와 우리 idl.rs byte-identical |
| RPC.2 | sess_id routing 의미 | ✗ 기각 | main.c::htp_iface_start sess_id 는 FARF logging 만 |
| RPC.3 | init sequence 누락 step | ✗ 기각 | host.rs GET_URI + LATENCY 이미 구현, ggml-hexagon.cpp 와 1-to-1 일치 |
| RPC.4 (잔존) | DSP-side dspqueue_rpc_skel method 3 driver reject | △ host scope 외 | logcat: handle 은 libdspqueue_rpc_skel.so, libggml-htp-v79.so FARF 0줄 |

---

## 결정적 logcat (18:25:01)

```
.430 remote_handle64_open: libggml-htp-v79.so (refs 1)
.431 remote_handle64_open: libdspqueue_rpc_skel.so (refs 2)   ← driver 자동 load
.431 fastrpc_notif worker thread launched
.432 Error 0x80000414: remote_handle64_invoke failed
     interface libdspqueue_rpc_skel.so method 3 on domain 7 (sc 0x3010100)
.436 remote_handle_close_domain libdspqueue_rpc_skel.so
```

핵심:
- fail handle = `libdspqueue_rpc_skel.so` (driver transport)
- libggml-htp-v79.so FARF log 0줄 — DSP compute module 진입 자체 0회
- sc 0x3010100 decode: method 3, n_in=1, n_out=1 — IDL contract match
- 0x80000414 = `AEE_EUNABLETOLOAD` 의 high-bit set variant. unsigned PD policy reject

---

## 다음 작업: Q-2.2 decision gate (3 옵션)

### 옵션 A: Hexagon SDK signed test signature 시도

- Hexagon SDK Docker 의 `mdsp-test-sign` 같은 dev signing tool 사용
- libggml-htp-v79.so + libdspqueue_rpc_skel.so 를 dev-signed PD 로 push
- stock S25 가 dev signing 을 accept 하는지 별 검증 필요
- 추정 4-6h
- risk: stock device 가 dev signing 자체를 거부할 수 있음 (production policy)
- 권장 위임: senior-implementer

### 옵션 B: NPU track 종료 + 결과 paper 에 기록

- Q-2.2 architectural barrier 자체가 paper 의 negative-result evidence
- handoff_α + β-1.START + β-1.MAP + β-1.QUEUE + β-1.QUEUE.RPC 4 sub-sprint 의 host-side 시도 + DSP-side wall 의 정리된 narrative
- backlog 의 OpenCL 최적화 항목으로 전환 (이미 production backend)
- 추정 1-2h (paper writeup 만)
- risk: NPU 카드 자체를 paper 에서 제거 — submission timeline 영향 분석 필요
- 권장 위임: PM + Researcher (paper integration)

### 옵션 C: llama.cpp 의 stock S25 동작 검증

- 우리 build (`983df14`) 의 llama.cpp main 을 동일 S25 device 에서 직접 실행
- NPU compute 가 동작하는지 검증
- universal wall 인지 (옵션 B 정당화) vs 우리 binding detail 차이 (추가 sprint)
- 추정 2-3h
- risk: llama.cpp 자체 빌드는 이미 완료 (Q-2.2-α.P2). main binary deploy + 실행 protocol 필요
- 권장 위임: senior-implementer

### 검증 게이트 (모든 옵션 공통)

- 옵션 별 결정 → 새 sprint plan 수립 → 진입

---

## Landmines / 미해결

- **`Error 0xd: open_shell failed for domain 3 (Permission denied)`**: 정상. cross-domain ADSP. cdsp (7) 만 쓰는 본 path 와 무관.
- **`Error 0xffffffff: fastrpc_enable_kernel_optimizations failed (Bad address)`**: cross-issue. unsigned PD kernel optimization tuning 권한 부족 가능성. 본 fail 의 contributing factor 일 수도.
- **session leak 잔존**: 측정 시 sleep 10s 우회. 별 issue.
- **handoff_q22_alpha_complete_2026_05_26.md status 와 일관**: "NPU HVX vector unit 0회 실행". 4 sub-sprint 모든 host infra 추가도 이 status 변동 없음.
- **mainline llama.cpp 는 schema migration 완료 (htp_opbatch_req)**: 우리는 구 vintage. 만약 옵션 C 가 fail 이면 mainline 으로 우리도 migrate 후 재시도 가능 — 별 sprint level work.

---

## 핵심 파일 인덱스

- 직전 commit: `05a70824` (β-1.QUEUE)
- 본 sprint commit: 없음 (분석만)
- 우리 schema: `engine/src/backend/htp_fastrpc/idl.rs:127-181` (HtpGeneralReq 312B)
- ground truth: `/home/go/Workspace/llama.cpp/ggml/src/ggml-hexagon/htp/htp-msg.h:118-138`
- llama.cpp init sequence: `/home/go/Workspace/llama.cpp/ggml/src/ggml-hexagon/ggml-hexagon.cpp:1535-1680`
- DSP-side htp_iface_start: `/home/go/Workspace/llama.cpp/ggml/src/ggml-hexagon/htp/main.c:249-313`
- 우리 host init: `engine/src/backend/htp_fastrpc/host.rs:404-680`
- 본 sprint report: `papers/eurosys2027/_workspace/experiment/qnn_q22_beta_queue_rpc_2026_05_26/report.md`
- 직전 handoff: `.agent/todos/handoff_q22_beta_queue_2026_05_26.md`
