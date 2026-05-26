# Handoff: Q-2.2-β.MAP 완료 (mmap GREEN) → β-1.QUEUE 진입

**작성**: 2026-05-26
**HEAD**: (커밋 직후 갱신)
**worktree**: `.claude/worktrees/b5_trait_extension`
**다음 세션 진입 문장**: **"Q-2.2-β.QUEUE 진입 — DspQueueBuffer.flags cache maintenance bit 명시 세팅 + remote_register_buf_attr2 dlsym 후 dspqueue_write GREEN + rmsnorm correctness PASS"**

---

## TL;DR

β-1.MAP 핵심 검증 게이트 **fastrpc_mmap rc=0 GREEN**:
- root cause = `FASTRPC_MAP_FD` enum 값 오인 (16 = NOMAP → 정확값 2)
- QC fastrpc 오픈소스 `inc/remote.h::enum fastrpc_map_flags` 직접 확인
- 동시에 `RPCMEM_HEAP_NOREG` 정정 (`0x80000000` = HEAP_DEFAULT → 정확값 `0x40000000`)
- backend.rs `pub mod htp_fastrpc;` 누락 보정 (β-1.START commit 의 buggy 부분)

β-1 의 "rmsnorm correctness PASS" 추가 goal 은 **별 sub-sprint β-1.QUEUE 로 분리**:
- 새 fail: `dspqueue_write rc=AEE_EUNABLETOLOAD (0xe)` + driver-side `fastrpc_buffer_ref ref=-1`
- mmap 자체는 GREEN, dispatch packet 의 buffer attach 가 fail
- 가설: `DspQueueBuffer.flags` cache maintenance bit 누락 또는 `remote_register_buf_attr2` 추가 필요

본 sprint 결과 = β-1.MAP scope 완료. dispatch correctness 는 β-1.QUEUE 로 위임.

---

## 진행 상태

| Task | 상태 | 작업 | 증거 |
|---|---|---|---|
| β-1.MAP.1 RPCMEM_HEAP_NOREG 정확값 확정 | ✓ | QC `inc/rpcmem.h` WebFetch | NOREG=0x40000000 (이전 0x80000000=HEAP_DEFAULT 였음) |
| β-1.MAP.2 FASTRPC_MAP_FD 정확값 확정 | ✓ | QC `inc/remote.h::enum fastrpc_map_flags` WebFetch | FD=2 (이전 16=FD_NOMAP) |
| β-1.MAP.3 host.rs/buffer.rs 상수 갱신 | ✓ | host.rs:73/76/79, buffer.rs:30 | clippy clean |
| β-1.MAP.3.5 backend.rs htp_fastrpc 선언 추가 | ✓ | backend.rs:29 (master merge 직후 누락 보정) | cross-build PASS |
| β-1.MAP.4 S25 fastrpc_mmap rc=0 확인 | ✓ | logcat `Error 0x1 fastrpc_mmap failed` 미발생 | stdout: `htp_iface_start: OK` 다음 dispatch attempt 진입 |
| β-1.MAP.5 NOREG ablation | ✓ | NOREG 포함/제거 두 변형 동일 mmap GREEN | dispatch 단계 동일 fail — NOREG 는 본 issue 와 무관 |
| β-1.MAP.6 report + handoff | ✓ | qnn_q22_beta_map_2026_05_26/report.md + 본 handoff | — |
| (별 sub-sprint) β-1.QUEUE rmsnorm correctness | □ | dspqueue_write rc=0xe + ref=-1 발견 | β-1.QUEUE 로 위임 |

---

## 다음 작업: β-1.QUEUE

**검증 게이트**: microbench_htp_rmsnorm 재실행 → `dspqueue_write rc=0` +
`dspqueue_read rc=0` + `rsp.status == HTP_STATUS_OK` + max_abs_err < 1e-3
vs CPU baseline.

### 작업 분해 (가설)

| 단계 | 작업 | 추정 | Risk | Fallback |
|---|---|---|---|---|
| β-1.QUEUE.1 | `DspQueueBuffer.flags` 명시 세팅 — `host.rs:154 reserved` 영역에서 cache maintenance bit (`DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER`, `_INVALIDATE_RECIPIENT`) 정확값 확인 + llama.cpp `ggml-hexagon.cpp:2242-2255` `htp_req_buff_init` 의 `dspqbuf_type` 분기 transcribe | 30 min | 낮음 | dspqueue.h 헤더 dump 필요 시 QC `inc/dspqueue.h` WebFetch |
| β-1.QUEUE.2 | `remote_register_buf_attr2` dlsym 추가 + `RpcmemBuffer::alloc` 의 mmap 직후 호출 (mmap 만으로 부족할 가능성 검증) | 45 min | 중간 | signature: `void remote_register_buf_attr2(void* buf, size_t size, int fd, int attr)`. attr=0 first |
| β-1.QUEUE.3 | S25 재실행 → logcat `fastrpc_buffer_ref ref=-1` 미발생 + rsp.status 확인 | 15 min | 낮음 | 여전히 fail 시 logcat의 추가 단서 분석 |
| β-1.QUEUE.4 | rmsnorm correctness gate (max_abs_err < 1e-3) + report.md + commit + β-2.MM handoff | 30 min | 낮음 | — |

### 위임 권장

senior-implementer — dspqueue cache flag semantics + libcdsprpc.so internal
ref count behavior + IDL register_buf signature dlsym.

### 위험 시나리오

- **AEE_EUNABLETOLOAD 가 cache maintenance 미세 이슈가 아닌 다른 원인일 경우**:
  예를 들어 packet size mismatch (우리 312 byte vs llama.cpp htp_general_req
  실제 size). msg_length=312 의 정확성 검증 필요. → microbench source 의
  `HtpGeneralReq` size_of 와 llama.cpp htp-msg.h::htp_general_req 비교.
- **register_buf_attr2 가 fastrpc_mmap 과 충돌**: mmap + register 동시 호출
  시 driver 가 reject 가능. mmap 제거 후 register 만 시도하는 ablation 필요.

---

## Landmines / 미해결

- **dspqueue_write 의 `fastrpc_buffer_ref ref=-1` 의 의미**: driver internal
  buffer table 에서 fd 의 ref count 가 mmap 이후 0 또는 1 (정상은 ≥1).
  dspqueue_write 가 decrement 호출 시 -1 underflow. 즉 mmap 이 ref 를 정상
  증가시키지 못한 것. mmap 자체 rc=0 였으나 ref count side effect 부재.
- **packet flags `0x3`**: 이건 driver-side 의 internal flag. host 가 보낸
  `flags=0` (mod.rs:157) 은 별 의미. driver 가 자체적으로 cache flag 를
  추론한 결과. 이 부분은 정상.
- **session leak 잔존**: β-1.START handoff 의 동일 issue. 본 sprint 측정
  중에도 5-10s wait 으로 우회. fix 는 별 issue.
- **β-1.START commit (58bb805a) 의 `pub mod htp_fastrpc;` 누락**: master
  merge (9c33b59c) 시 backend.rs 충돌 처리에서 누락. 본 sprint 에서 보정.
  β-1.START 측정이 어떻게 통과했는지는 미스터리 — 본 sprint 의 cross-build
  GREEN 으로 인프라 회복.
- **`RPCMEM_TRY_MAP_STATIC = 0x04000000` 정의만 추가, 미사용**: β-1.QUEUE
  의 fallback 후보로 보유.

---

## 핵심 파일 인덱스

- 본 sprint commit: (commit 직후 갱신)
- host: `engine/src/backend/htp_fastrpc/host.rs` (3 const 정정 — NOREG/FD/TRY_MAP_STATIC)
- buffer: `engine/src/backend/htp_fastrpc/buffer.rs` (doc 갱신, alloc/mmap 로직 무변경)
- backend export: `engine/src/backend.rs:29` (`pub mod htp_fastrpc;` 추가)
- microbench: `engine/microbench/htp_rmsnorm.rs` (변경 없음, β-1.START 의 try_start_iface 명시 호출 그대로)
- report: `papers/eurosys2027/_workspace/experiment/qnn_q22_beta_map_2026_05_26/report.md`
- QC fastrpc 오픈소스 참조: `github.com/quic/fastrpc/master/inc/{rpcmem,remote}.h`
- 다음 dspqueue 헤더 후보: `github.com/quic/fastrpc/master/inc/dspqueue.h` (β-1.QUEUE 진입 시 WebFetch)
