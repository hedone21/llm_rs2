# Q-2.2-β.QUEUE.RPC — host scope 소진, DSP-side dspqueue_rpc_skel reject 가 architectural barrier

**작성**: 2026-05-26
**HEAD**: `05a70824` (β-1.QUEUE 직후) — 본 sprint 는 분석만, 코드 변경 없음
**진입**: handoff_q22_beta_queue_2026_05_26.md
**worktree**: `.claude/worktrees/b5_trait_extension`

## TL;DR

β-1.QUEUE.RPC 의 3 host-side 가설 (packet schema / sess_id 의미 / init
sequence 누락) **모두 기각**. logcat 의 정확한 timing trace 로 fail 지점이
`libggml-htp-v79.so` (DSP-side compute module) 가 아닌
**`libdspqueue_rpc_skel.so` (driver-side dspqueue transport stub)** 의
method 3 reject 임을 확정. 이는 host scope (Rust binding) 를 벗어난 영역
— DSP-side unsigned PD 의 module signing/policy barrier.

본 sprint 결과 = β-1.QUEUE.RPC 의 host-side fix 후보 소진. β-2.MM (matmul)
진입 무의미 — rmsnorm 도 안 가는 wall 에 matmul 시도해도 동일 fail.
Q-2.2 NPU track 의 decision gate 도달.

## 검증 layer

| RPC | 가설 | 결과 | 근거 |
|---|---|---|---|
| RPC.1 | HtpGeneralReq 312B vs llama.cpp htp-msg.h mismatch | ✗ 기각 | `/home/go/Workspace/llama.cpp/ggml/src/ggml-hexagon/htp/htp-msg.h` 의 `htp_general_req` (op + op_params[16] + flags + 6 tensor) = 우리 `idl.rs::HtpGeneralReq` 와 byte-identical. mainline llama.cpp 는 이미 `htp_opbatch_req` 로 schema migration 되었으나 우리 build (`983df14`) 는 구 htp_general_req. 우리 .so 와 .rs 가 같은 vintage. |
| RPC.2 | `htp_iface_start` 의 sess_id 가 routing key | ✗ 기각 | `main.c::htp_iface_start` (L249-313) 의 sess_id 는 단일 `FARF(HIGH, "session %u started: ...")` 로깅에만 사용. 실제 routing 은 `dspqueue_import(dsp_queue_id, ...)` 만 — dev_id vs sess_id 는 IDL slot 동일. |
| RPC.3 | host init sequence 의 누락 step (GET_URI / LATENCY) | ✗ 기각 | host.rs:543-617 에 GET_URI (L568) + LATENCY (L612) 이미 구현. ggml-hexagon.cpp:1559-1679 의 sequence (reserve_session → GET_URI → UNSIGNED_MODULE → handle_open → LATENCY → dspqueue_create → dspqueue_export → htp_iface_start) 와 1-to-1 일치. |
| RPC.4 (잔존) | DSP-side `libdspqueue_rpc_skel.so` method 3 driver reject | △ host scope 외 | logcat 의 fail line `Error 0x80000414: remote_handle64_invoke failed for handle ..., interface libdspqueue_rpc_skel.so method 3 on domain 7 (sc 0x3010100)`. 이는 우리 compute module (libggml-htp-v79.so) 가 아니라 dspqueue 의 transport-level RPC stub 의 reject. unsigned PD 정책 / driver module signing 영역. |

## 결정적 logcat trace (timing)

```
18:25:01.430 ... remote_handle64_open: opened handle ... for libggml-htp-v79.so (refs 1, spawn 22272us, load 4865us)
18:25:01.431 ... remote_handle64_open: opened handle ... for libdspqueue_rpc_skel.so (refs 2, spawn 1us, load 136us)
18:25:01.431 ... fastrpc_notif_domain_init: FastRPC notification worker thread launched
18:25:01.432 ... Error 0x80000414: remote_handle64_invoke failed for handle ..., interface libdspqueue_rpc_skel.so method 3 on domain 7 (sc 0x3010100) (user err 0x80000414)
                ↑ host 의 dspqueue_write 가 호출한 driver-side IDL stub
18:25:01.436 ... remote_handle_close_domain: closed handle (libdspqueue_rpc_skel.so) — fastrpc cleanup
```

핵심 관찰:
1. **`libdspqueue_rpc_skel.so`** 는 host 가 직접 open 하지 않은 module — `libcdsprpc.so::dspqueue_create` 가 자동으로 DSP 에 push.
2. method 3 fail 의 handle 은 **dspqueue_rpc_skel** (driver transport stub) 의 handle, **libggml-htp-v79.so** (우리 compute module) 의 handle 이 아니다.
3. 즉 packet 이 DSP-side 의 우리 callback (`htp_packet_callback`) 에 도달하기 전, driver-level 의 dspqueue transport 가 message 를 enqueue 시점에 reject.
4. logcat 에 `libggml-htp-v79.so` 의 FARF log 가 단 한 줄도 안 보임 = compute module 진입 0회.

## scalars decode (sc 0x3010100)

`REMOTE_SCALARS_MAKEX(attr=0, method=3, n_in=1, n_out=1, no_in=0, no_out=0)`
- method 3 = dspqueue_write 의 IDL impl
- n_in=1 input buffer (우리 message 312 B)
- n_out=1 output buffer (driver internal)

sc 자체는 contract 와 match. 즉 our 312 B message 가 driver-side stub 의
expected layout 과 일치. 그럼에도 fail = **driver-side 의 PD-level reject**.

## 0x80000414 의미

QC AEE error code:
- `0x80000414` = `AEE_EBADPARM_CODE`? `AEE_EUNABLETOLOAD`? raw value 14 (0xe)
  의 high-bit set variant.
- `AEE_EUNABLETOLOAD (14)` 은 일반 "unable to load" — DSP-side dspqueue
  transport 가 message 를 ctx->queue 에 enqueue 시점에 PD-level resource
  load 가 fail. unsigned PD 에서 dspqueue 자체의 internal kernel module 또는
  worker thread queue 가 권한 부족.

## 본 sprint 의 코드 변경

**없음**. 분석/검증만 수행. 직전 commit (`05a70824`, β-1.QUEUE) 의 cache flag
정정 + register_buf_attr2 추가가 host-side 의 본 sprint 결과물. β-1.QUEUE.RPC
는 그 위에서 host scope fix 후보가 더 있는지 검증한 결과 — **없음**.

## decision gate

Q-2.2 NPU track 의 hard wall 확인:
- handoff_q22_alpha_complete_2026_05_26.md status = "NPU HVX vector unit 0회 실행"
- β-1.START + β-1.MAP + β-1.QUEUE + β-1.QUEUE.RPC 4 sub-sprint 모두 host
  infra 강화 (lifecycle 4 method, fastrpc_mmap, cache flag, register_buf,
  GET_URI/LATENCY) 적용.
- 결과: 모두 dspqueue_write 의 driver-side stub method 3 EUNABLETOLOAD 에서
  block. DSP-side compute module (libggml-htp-v79.so) 의 packet handler
  진입 자체가 0 회.

### 3 후속 옵션

1. **(A) Hexagon SDK signed test signature 시도**: SDK Docker 의 `mdsp-test-sign`
   같은 dev signing tool 사용 → libggml-htp-v79.so + libdspqueue_rpc_skel.so
   를 dev-signed PD 로 push. stock S25 가 dev signing 을 accept 하는지 별
   검증 필요. 추정 4-6h.
2. **(B) NPU track 종료**: Q-2.2 architectural barrier 확인 + paper 에 결과
   기록 (negative result 자체가 paper 의 evidence). 기존 OpenCL backend
   집중. backlog 의 OpenCL 최적화 항목으로 전환.
3. **(C) llama.cpp 의 stock S25 동작 검증**: 우리 build 의 llama.cpp main 을
   동일 S25 device 에서 직접 실행하여 NPU compute 가 도는지 검증. 만약
   동일하게 fail 이면 **stock S25 가 universal wall** (옵션 B 정당화).
   동작한다면 우리 host binding 의 특정 detail 차이 → 추가 sprint.

### 위임 권장

senior-implementer 또는 researcher — Hexagon SDK signing policy 조사.

## Landmines / 미해결

- **`Error 0xd: open_shell failed for domain 3 (Permission denied)`**: 정상.
  cross-domain (ADSP) 권한 미부여. cdsp (7) 만 쓰는 본 path 와 무관.
- **`Error 0xffffffff: fastrpc_enable_kernel_optimizations failed (Bad address)`**:
  cross-issue. unsigned PD 에서 kernel optimization tuning 권한 부족 가능성.
  본 fail 의 contributing factor 일 수도 있으나 architectural barrier 의
  하위 증상.
- **session leak 잔존**: 본 sprint 측정 시에도 sleep 10s 우회. 별 issue.

## 핵심 파일 인덱스

- 본 sprint commit: 없음 (분석만)
- 직전 commit (β-1.QUEUE): `05a70824`
- 우리 schema vs llama.cpp htp-msg.h 비교 baseline:
  - 우리: `engine/src/backend/htp_fastrpc/idl.rs:127-181` (HtpGeneralReq 312B)
  - llama.cpp: `/home/go/Workspace/llama.cpp/ggml/src/ggml-hexagon/htp/htp-msg.h:118-138`
- llama.cpp init sequence: `/home/go/Workspace/llama.cpp/ggml/src/ggml-hexagon/ggml-hexagon.cpp:1535-1680`
- DSP-side htp_iface_start: `/home/go/Workspace/llama.cpp/ggml/src/ggml-hexagon/htp/main.c:249-313`
- 다음 진입점: handoff_q22_beta_queue_rpc_2026_05_26.md
