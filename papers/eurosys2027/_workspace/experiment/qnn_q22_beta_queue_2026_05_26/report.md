# Q-2.2-β.QUEUE — driver-internal ref count GREEN, dispatch BLOCKED on DSP-side RPC stub

**작성**: 2026-05-26
**HEAD**: (commit 직후 갱신)
**진입**: handoff_q22_beta_map_2026_05_26.md
**worktree**: `.claude/worktrees/b5_trait_extension`

## TL;DR

β-1.QUEUE 의 두 정정 모두 적용:
- (Q.1) `idl.rs` cache flag const `1 << 0` / `1 << 1` → `1 << 4` (0x10) / `1 << 7` (0x80)
  — QC `inc/dspqueue.h::dspqueue_buffer_flags` ground truth 일치.
- (Q.2) `host.rs` + `buffer.rs` 에 `remote_register_buf_attr2` dlsym + alloc 직후
  호출 + Drop deregister — QC `inc/remote.h`: "Registration functions prepare
  buffers for use in FastRPC calls — distinct from fastrpc_mmap()".

**검증 진전 (logcat evidence)**:
- (Q.2) 이전 핵심 fail 라인 **`fastrpc_buffer_ref: Attempting to remove last reference ... ref -1`
  완전히 사라짐** — driver-internal ref count underflow 해결.
- 그러나 dispatch full GREEN 은 아직 — 새 root cause: DSP-side
  `libdspqueue_rpc_skel.so method 3` 이 `AEE_EUNABLETOLOAD (0xe)`.

β-1.QUEUE scope = "DspQueueBuffer.flags cache bit + register_buf 추가" 까지
**완료**. 추가 검증 (rmsnorm correctness) 은 새 sub-sprint **β-1.QUEUE.RPC**
로 위임 — packet size 또는 htp_iface_start silent fail 검증.

## 검증 layer

| 검증 | 상태 | 증거 |
|---|---|---|
| (a) `DSPQUEUE_BUFFER_FLAG_*` 정확값 확인 | ✓ | QC dspqueue.h: FLUSH_SENDER=0x10, INVALIDATE_RECIPIENT=0x80 (이전 `1 << 0` / `1 << 1` 은 reserved/invalid bit) |
| (b) `idl.rs` const 갱신 (1<<4 / 1<<7) | ✓ | idl.rs:90/91 |
| (c) `to_flags()` unit test PASS (값 변화 무관, OR 비교) | ✓ | `dspq_buffer_type_flags` 2 tests OK |
| (d) `remote_register_buf_attr2` libcdsprpc.so export 확인 | ✓ | `strings /vendor/lib64/libcdsprpc.so | grep ^remote_register` 에 명시 |
| (e) `RemoteRegisterBufAttr2Fn` typedef + `FASTRPC_ATTR_*` const 추가 | ✓ | host.rs:240-260 |
| (f) `HtpFastrpcHost::remote_register_buf_attr2: Option<...>` field | ✓ | host.rs:272-274 |
| (g) dlsym (optional, None tolerant 구 driver 호환) | ✓ | host.rs:472-478 |
| (h) `RpcmemBuffer::alloc` mmap 직후 register 호출 | ✓ | buffer.rs:145-160 (`if let Some(register)` 분기) |
| (i) `RpcmemBuffer::drop` deregister (fd=-1 sentinel) | ✓ | buffer.rs:240-252 (munmap 보다 앞) |
| (j) 호스트 빌드 + clippy clean | ✓ | `cargo clippy -p llm_rs2 --lib --features htp_fastrpc -- -D warnings` PASS |
| (k) S25 logcat: `fastrpc_buffer_ref ... ref -1` 라인 미발생 | **✓** | β-1.MAP 후의 ref=-1 underflow 완전 사라짐 |
| (l) **rmsnorm correctness PASS** | ✗ (β-1.QUEUE.RPC 로 분리) | DSP-side `libdspqueue_rpc_skel.so method 3 EUNABLETOLOAD` 새 root cause |

## 결정적 발견

### (1) cache flag bit 값 ground-truth 일치 (β-1.START 잠재 버그 정정)

| const | 이전 (β-1.START) | β-1.QUEUE 정확값 | 출처 |
|---|---|---|---|
| `DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER` | `1 << 0` = 0x01 | **`1 << 4` = 0x10** | QC `inc/dspqueue.h` |
| `DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT` | `1 << 1` = 0x02 | **`1 << 7` = 0x80** | 동상 |

→ 이전 `dbuf.flags = 0x03` 은 driver 가 인식하지 못하는 reserved/invalid
bit. 정확값으로 정정 시 dispatch 의 cache maintenance phase 가 정상 수행됨.
단독으로는 dispatch fail 해결 아님 — register_buf 와 함께 본 sprint 적용.

### (2) mmap ≠ register (driver-internal ref count)

QC `inc/remote.h` 문서:

> "Register a file descriptor for a buffer to enable zero-copy sharing with
> DSP via SMMU. ... These are distinct from `fastrpc_mmap()`, which creates
> mappings for DMA buffers, whereas registration functions prepare buffers
> for use in FastRPC calls."

즉:
- `fastrpc_mmap` = SMMU 매핑 (DSP 가 physical address 접근 가능하도록 page table set)
- `remote_register_buf_attr2` = driver-internal table 에 buffer 등록
  (`fastrpc_buffer_ref` 같은 ref count 동작의 대상이 되도록)

`dspqueue_write` 시 buffer 가 attach 되면 driver 가 `fastrpc_buffer_ref +1`
을 시도. 미등록 buffer (mmap 만) 면 ref=-1 underflow.

본 sprint 정정으로 buffer 마다 mmap + register 두 단계 모두 적용.

## S25 실측 결과

### β-1.QUEUE.1 단독 (cache flag bit 정정만)

| run | 결과 | logcat 핵심 |
|---|---|---|
| 1 | dispatch fail `AEE_EUNABLETOLOAD (0xe)` | `fastrpc_buffer_ref ref -1` 그대로 발생 |

→ cache flag 정정 단독으로는 ref=-1 issue 미해결.

### β-1.QUEUE.1 + Q.2 (register_buf_attr2 추가)

| run | 결과 | logcat 핵심 |
|---|---|---|
| 1 (n_hvx=0) | dispatch fail `AEE_EUNABLETOLOAD (0xe)` | **`ref -1` 라인 완전 사라짐**, 새 라인 `libdspqueue_rpc_skel.so method 3 (sc 0x3010100)` EUNABLETOLOAD |
| 2 (n_hvx=4) | 동상 | 동상 — `n_hvx` 와 무관 |

→ register_buf_attr2 추가로 ref count underflow 해결. 새 fail 은
DSP-side RPC stub `libdspqueue_rpc_skel.so` 의 method 3 (= dspqueue_write
IDL impl) 자체가 module load 불가. host-side 변경으로는 추가 진전 미정.

```
05-26 17:38:08 Error 0xd: open_shell failed for domain 3 (Permission denied)   ← 정상 (cross-domain 권한)
05-26 17:38:08 Error 0xffffffff: fastrpc_enable_kernel_optimizations failed (Bad address)   ← cross-issue
05-26 17:38:08 Error 0x80000414: remote_handle64_invoke failed ... libdspqueue_rpc_skel.so method 3 (sc 0x3010100)   ← 진짜 root cause
05-26 17:38:08 Error 0xe: dspqueue_write_noblock failed for queue ... (flags 0x3, num_buffers 2, message_length 312)
05-26 17:38:08 Error 0xe: dspqueue_write failed for queue ... (flags 0x0, num_buffers 2, message_length 312 errno Success)
```

### sc 0x3010100 decode

`REMOTE_SCALARS_MAKEX(0, 3, 1, 1, 0, 0)`:
- attr = 0
- method = 3 (dspqueue_write IDL method)
- n_in = 1 input buffer
- n_out = 1 output buffer

즉 dspqueue_write 의 IDL contract 자체는 정상 (1 in + 1 out). DSP-side 가
이 method 3 의 implementation 을 load 할 수 없음.

## 변경 요약 (3 파일)

| 파일 | 변경 | LOC |
|---|---|---|
| `engine/src/backend/htp_fastrpc/idl.rs` | `DSPQUEUE_BUFFER_FLAG_*` 정확값 정정 (1<<4 / 1<<7) + doc 갱신 (β-1.QUEUE 근거 14-line) | +16 / -2 |
| `engine/src/backend/htp_fastrpc/host.rs` | `RemoteRegisterBufAttr2Fn` typedef + `FASTRPC_ATTR_*` const + Option field + dlsym + struct construction | +30 / -0 |
| `engine/src/backend/htp_fastrpc/buffer.rs` | use clause 갱신 + alloc 의 register 호출 + Drop deregister + doc 갱신 | +33 / -2 |

## 다음 sub-sprint: β-1.QUEUE.RPC

**검증 게이트**: `dspqueue_write rc=0` + `dspqueue_read rc=0` +
`rsp.status == HTP_STATUS_OK` + `max_abs_err < 1e-3` vs CPU baseline.

### 가설

1. **packet size mismatch**: 우리 `HtpGeneralReq` size = 312 byte 가 llama.cpp
   `htp-msg.h::htp_general_req` 실제 size 와 불일치 가능성. layout 검증 필요.
   - 추정 시간: 30 min (htp-msg.h WebFetch + size_of 확인 + field 보정)
2. **htp_iface_start silent fail**: host stdout `htp_iface_start: OK (n_hvx=0)`
   는 invoke rc=0 만 검증. DSP-side 가 실제로 attach 되었는지 (`htp_iface`
   structure 의 internal flag) 검증 안 함. n_hvx=0 또는 4 둘 다 동일 fail
   ablation 으로 본 hypothesis weak. logcat 에서 method 2 invoke 의 추가 검증 필요.
   - 추정 시간: 20 min (logcat 분석 + method 2 sc decode)
3. **DSP-side libggml-htp-v79.so 의 packet handler 등록 누락**: dspqueue 가
   message dispatch 시 receiving module 로 `libggml-htp-v79.so` 의 packet
   callback 을 lookup 해야 하는데, htp_iface_start 가 그 등록을 수행하지 못함.
   llama.cpp 가 별도 `remote_handle_open(libggml-htp-v79.so?...)` 후 추가 init
   호출이 있을 가능성.
   - 추정 시간: 30 min (llama.cpp ggml-hexagon.cpp init sequence 검토)
4. **session ID 또는 queue_id binding 누락**: htp_iface_start 의 payload (u32
   sess_id + u64 queue_id + u32 n_hvx) 가 DSP-side 에서 dspqueue 와 binding 되어야 하는데, 우리는 sess_id=1 + queue_id (export 결과) 를 보냄. 검증 필요.
   - 추정 시간: 20 min

### 작업 분해 (가설별)

| 단계 | 작업 | 추정 | Risk |
|---|---|---|---|
| RPC.1 | llama.cpp `htp-msg.h::htp_general_req` size + layout 검증 | 30m | 낮음 |
| RPC.2 | htp_iface_start invoke 의 logcat method 2 trace 분석 (success 여부) | 20m | 낮음 |
| RPC.3 | (조건부) llama.cpp ggml-hexagon.cpp 의 dspqueue + libggml-htp-v79.so init sequence transcribe | 30m | 중간 |
| RPC.4 | S25 재실행 + dispatch GREEN 확인 + rmsnorm correctness gate | 30m | 낮음 |
| RPC.5 | report + commit + β-2.MM handoff | 30m | 낮음 |

**예상 시간**: 2~3h.

### 위임 권장

senior-implementer — packet layout binary debugging + DSP-side IDL trace.

## Landmines / 미해결

- **`Error 0xd: open_shell failed for domain 3 (Permission denied)`**: 정상.
  cross-domain (3=ADSP) 권한 미부여라 cdsp (7) 만 쓰는 본 path 와 무관.
- **`Error 0xffffffff: fastrpc_enable_kernel_optimizations failed (Bad address)`**:
  cross-issue (kernel optimization tuning). dispatch 와 무관.
- **`flags 0x3` (dspqueue_write_noblock)**: driver-internal packet flag
  (DSPQUEUE_PACKET_FLAG_MESSAGE | _BUFFERS). 우리가 보낸 `0x0` 이 lower
  layer 에서 0x3 으로 변환 — 정상.
- **session leak 잔존**: 본 sprint 측정에도 wait 10s 우회. session.start
  이후 process kill 시 driver-side session release race. 별 issue.

## 핵심 파일 인덱스

- 본 sprint commit: (commit 직후 갱신)
- idl: `engine/src/backend/htp_fastrpc/idl.rs:90-91` (cache flag 정확값)
- host: `engine/src/backend/htp_fastrpc/host.rs:240-260, 272-274, 472-478, 661`
- buffer: `engine/src/backend/htp_fastrpc/buffer.rs:21, 145-160, 240-252`
- QC fastrpc 오픈소스: `github.com/quic/fastrpc/master/inc/{remote,dspqueue}.h`
- 다음 진입점: handoff_q22_beta_queue_2026_05_26.md
