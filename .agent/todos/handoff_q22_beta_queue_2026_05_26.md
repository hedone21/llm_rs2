# Handoff: Q-2.2-β.QUEUE 완료 (ref count GREEN) → β-1.QUEUE.RPC 진입

**작성**: 2026-05-26
**HEAD**: (커밋 직후 갱신)
**worktree**: `.claude/worktrees/b5_trait_extension`
**다음 세션 진입 문장**: **"Q-2.2-β.QUEUE.RPC 진입 — DSP-side libdspqueue_rpc_skel.so method 3 EUNABLETOLOAD 의 root cause (packet size mismatch / htp_iface_start silent fail / packet handler 등록 누락) 분석 + rmsnorm correctness PASS"**

---

## TL;DR

β-1.QUEUE 의 두 정정 완료:
- (Q.1) `idl.rs` `DSPQUEUE_BUFFER_FLAG_*` const `1<<0` / `1<<1` → `1<<4` (0x10) / `1<<7` (0x80) 정정 (QC dspqueue.h ground truth)
- (Q.2) `host.rs` + `buffer.rs` 에 `remote_register_buf_attr2` dlsym + alloc 직후 register + Drop deregister 추가

S25 logcat 검증:
- 이전 핵심 fail 라인 **`fastrpc_buffer_ref ... ref -1` 완전 사라짐** — driver-internal ref count underflow 해결 (logcat evidence)
- 그러나 dispatch full GREEN 아님 — 새 root cause: `libdspqueue_rpc_skel.so method 3 (sc 0x3010100) EUNABLETOLOAD`. DSP-side IDL impl 자체가 module load 불가.

β-1.QUEUE scope ("DspQueueBuffer.flags cache bit + register_buf 추가") = **완료**. 추가 검증 (rmsnorm correctness) 은 **β-1.QUEUE.RPC** 로 위임.

---

## 진행 상태

| Task | 상태 | 작업 | 증거 |
|---|---|---|---|
| β-1.QUEUE.1 cache flag const 정정 | ✓ | idl.rs:90/91 (1<<0/1 → 1<<4/7) | QC dspqueue.h WebFetch 확인 |
| β-1.QUEUE.2 remote_register_buf_attr2 dlsym + alloc/Drop | ✓ | host.rs:240-260/272-274/472-478, buffer.rs:21/145-160/240-252 | libcdsprpc.so strings 에 export 확인, logcat ref=-1 사라짐 |
| β-1.QUEUE.3 호스트 빌드 + clippy + fmt | ✓ | cargo build/clippy clean | — |
| β-1.QUEUE.4 S25 logcat ref=-1 미발생 확인 | ✓ | logcat 17:38:08 의 fail 라인 set 변경 | 이전 ref=-1 line 사라지고 method 3 EUNABLETOLOAD 가 새 fail |
| β-1.QUEUE.5 (별 sub-sprint) rmsnorm correctness | □ | dispatch full GREEN 아님 — DSP-side issue | β-1.QUEUE.RPC 로 위임 |

---

## 다음 작업: β-1.QUEUE.RPC

**검증 게이트**: `dspqueue_write rc=0` + `dspqueue_read rc=0` + `rsp.status == HTP_STATUS_OK` + `max_abs_err < 1e-3` vs CPU baseline.

### 작업 분해 (가설 우선순위)

| 단계 | 작업 | 추정 | Risk | Fallback |
|---|---|---|---|---|
| RPC.1 | llama.cpp `htp-msg.h::htp_general_req` size + layout 검증 (우리 312 byte 와 일치 여부) | 30m | 낮음 | size 다르면 idl.rs `HtpGeneralReq` 보정 |
| RPC.2 | htp_iface_start invoke 의 logcat method 2 (sc decode) success 여부 분석 | 20m | 낮음 | host stdout `OK` 가 invoke rc=0 만 검증할 가능성 |
| RPC.3 | llama.cpp `ggml-hexagon.cpp` 의 dspqueue + libggml-htp-v79.so init sequence transcribe (우리 4-step handshake 가 누락한 step 검색) | 30m | 중간 | 누락 step 있으면 host.rs 에 추가 |
| RPC.4 | S25 재실행 + dispatch GREEN 확인 + rmsnorm correctness | 30m | 낮음 | 가설 모두 negative 시 senior-implementer 위임 |
| RPC.5 | report + commit + β-2.MM handoff | 30m | 낮음 | — |

**예상 시간**: 2~3h.

### 위임 권장

senior-implementer — DSP-side IDL stub method 3 의 module load 실패 root cause 분석. packet layout binary debugging + llama.cpp init sequence diff.

### 위험 시나리오

- **DSP-side libggml-htp-v79.so 가 host-side 와 분리된 module init 이 필요할 가능성**: llama.cpp 의 `htp_iface_start` 외에 별도 module register call (예: `remote_handle_open(libggml-htp-v79.so?init_handler)`) 가 있을 수 있다.
- **HtpGeneralReq layout 미세 mismatch**: 우리는 `#[repr(C)]` + field 별 size_of 계산. llama.cpp 의 packed struct 또는 padding 정책이 다르면 312 vs 다른 byte. binary diff 필요.
- **packet message 의 vendor-specific tail data**: llama.cpp 는 일부 buffer 를 packet message 끝에 inline 으로 추가할 수도. `dspqueue_write` 의 message_length 만으로 부족할 가능성.

---

## Landmines / 미해결

- **`Error 0xd: open_shell failed for domain 3 (Permission denied)`**: 정상. cross-domain (ADSP) 권한 미부여. cdsp (7) 만 쓰는 본 path 와 무관.
- **`Error 0xffffffff: fastrpc_enable_kernel_optimizations failed (Bad address)`**: cross-issue. dispatch 와 무관.
- **`flags 0x3` (dspqueue_write_noblock) 은 driver-internal packet flag**: DSPQUEUE_PACKET_FLAG_MESSAGE | _BUFFERS = 1 | 2 = 3. 우리가 보낸 `0x0` 이 lower layer 에서 자동 set. 정상. 본 sprint 의 ref=-1 issue 와 무관.
- **session leak 잔존**: 측정 시 wait 10s 우회. session.start 이후 process kill 시 driver-side session release race. 별 issue.
- **n_hvx 0 vs 4 둘 다 동일 fail**: n_hvx 자체는 root cause 아님 — DSP-side dispatch 진입조차 못 함.

---

## 핵심 파일 인덱스

- 본 sprint commit: (commit 직후 갱신)
- idl: `engine/src/backend/htp_fastrpc/idl.rs:76-101` (cache flag 정확값 + doc)
- host: `engine/src/backend/htp_fastrpc/host.rs:240-260, 272-274, 472-478, 661`
- buffer: `engine/src/backend/htp_fastrpc/buffer.rs:21-24, 145-160, 240-252`
- microbench: `engine/microbench/htp_rmsnorm.rs:506-575` (dispatch closure)
- report: `papers/eurosys2027/_workspace/experiment/qnn_q22_beta_queue_2026_05_26/report.md`
- QC fastrpc 헤더: `github.com/quic/fastrpc/master/inc/{remote,dspqueue}.h`
- llama.cpp htp-msg.h (다음 sub-sprint WebFetch 대상): build-snapdragon/ggml/src/ggml-hexagon/htp/htp-msg.h
