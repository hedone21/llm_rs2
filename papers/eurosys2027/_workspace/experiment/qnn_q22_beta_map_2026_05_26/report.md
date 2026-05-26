# Q-2.2-β.MAP — fastrpc_mmap GREEN, dispatch BLOCKED on dspqueue ref count

**작성**: 2026-05-26
**HEAD**: (commit 직후 갱신)
**진입**: handoff_q22_beta_start_2026_05_26.md
**worktree**: `.claude/worktrees/b5_trait_extension`

## TL;DR

β-1.MAP 핵심 검증 게이트 **fastrpc_mmap rc=0 GREEN**. root cause = `FASTRPC_MAP_FD`
enum 값 오인 (16 → 정확값 2). 이전 β-1.START 의 `rc=0x1 EIO` 는 사실 우리가
`FASTRPC_MAP_FD_NOMAP` (= 16, "등록만 하고 실제 mmap 안 함") 변종을 보냈기
때문. QC fastrpc 오픈소스 `inc/remote.h` 의 `enum fastrpc_map_flags` 정의
직접 확인.

β-1.MAP scope 의 추가 검증 (rmsnorm correctness PASS) 은 **별 sub-sprint
β-1.QUEUE 로 분리**. dispatch 단계에서 `dspqueue_write rc=AEE_EUNABLETOLOAD
(0xe)` + driver-side `fastrpc_buffer_ref ref=-1` 발생. mmap 후 buffer 의
ref count 관리 또는 packet 의 `dspqueue_buffer.flags` cache maintenance bit
가 의심. β-1.MAP 의 mmap 자체는 GREEN 이므로 본 sprint 검증 통과.

## 검증 layer

| 검증 | 상태 | 증거 |
|---|---|---|
| (a) `RPCMEM_HEAP_NOREG` 정확값 확인 | ✓ | QC `inc/rpcmem.h`: `RPCMEM_HEAP_NOREG = 0x40000000` (이전 코드 `0x80000000` 은 `RPCMEM_HEAP_DEFAULT`). 본 root cause 와는 무관하나 정정. |
| (b) `FASTRPC_MAP_FD` 정확값 확인 | ✓ | QC `inc/remote.h`: `enum fastrpc_map_flags { FASTRPC_MAP_STATIC=0, _RESERVED=1, **FD=2**, _DELAYED=3, ..., FD_NOMAP=16 }`. 이전 `16` 은 NOMAP 변종. |
| (c) host.rs `RPCMEM_HEAP_NOREG: u32 = 0x4000_0000` 갱신 | ✓ | host.rs L73 |
| (d) host.rs `FASTRPC_MAP_FD: u32 = 2` 갱신 | ✓ | host.rs L79 |
| (e) host.rs `RPCMEM_TRY_MAP_STATIC: u32 = 0x0400_0000` 추가 (fallback 후보) | ✓ | host.rs L76 |
| (f) backend.rs `pub mod htp_fastrpc;` 선언 추가 | ✓ | backend.rs L29 (master merge 직후 누락분 보정) |
| (g) S25 logcat: `fastrpc_mmap` error 라인 미발생 | ✓ | β-1.START 의 `Error 0x1 fastrpc_mmap failed ... domain 7 flags 0x10 ioctl ret -1 errno EIO` 사라짐. flags 0x10 (16=NOMAP) → 새 attempt 0x02 (2=FD). |
| (h) stdout: `htp_iface_start: OK` + dispatch attempt 진입 | ✓ | β-1.START 의 SKIP-on-alloc 이 사라지고 SKIP-on-write 로 전진 |
| (i) **rmsnorm correctness PASS** | ✗ (별 sub-sprint) | `dspqueue_write rc=AEE_EUNABLETOLOAD (0xe)` — β-1.QUEUE 로 분리 |

## 결정적 발견 (β-1.START 의 mmap fail root cause)

| const | β-1.START 값 | β-1.MAP 정확값 | 출처 |
|---|---|---|---|
| `RPCMEM_HEAP_NOREG` | `0x8000_0000` | **`0x4000_0000`** | QC `inc/rpcmem.h` |
| `RPCMEM_HEAP_DEFAULT` | (= NOREG, 오인) | `0x8000_0000` | 동상 |
| `RPCMEM_HEAP_ID_SYSTEM` | `25` ✓ | `25` | 동상 |
| `RPCMEM_DEFAULT_FLAGS` | `1` ✓ | `1` | `ION_FLAG_CACHED` |
| `RPCMEM_TRY_MAP_STATIC` | (정의 부재) | `0x0400_0000` | 동상 |
| `FASTRPC_MAP_FD` | `16` (= `FD_NOMAP`) | **`2`** | QC `inc/remote.h` `enum fastrpc_map_flags` |
| `FASTRPC_MAP_FD_NOMAP` | (= `FD`, 오인) | `16` | 동상 |

→ β-1.START 의 `flags 0x10` 은 driver-side 에서 `FD_NOMAP` 으로 해석 →
buffer 를 등록만 하고 실제 mmap 은 수행하지 않음 → 후속 driver-internal
검증이 mmap 결과를 요구하는 단계에서 ioctl EIO. 정확한 `FASTRPC_MAP_FD = 2`
사용 시 mmap 완전 수행, error 라인 미발생.

## S25 실측 결과

5 회 run, 모두 동일 패턴.

stdout:
```
[3/3] QNN HTP NPU (FastRPC + libggml-htp-v79.so dspqueue)
  host: libcdsprpc.so=/vendor/lib64/libcdsprpc.so, domain_id=7, session_id=1, queue_id=...
  htp_iface_start: OK (n_hvx=0)                                            ← β-1.START GREEN
  SKIP — dspqueue_write: htp_fastrpc: AEEResult=AEE_EUNABLETOLOAD (0xe, raw=14)   ← β-1.QUEUE issue
```

logcat (β-1.MAP 후 attempt):
```
05-26 17:00:48.794 ... fastrpc_buffer_ref: Attempting to remove last reference to buffer 18 on domain 7
05-26 17:00:48.794 ... Error 0xe: fastrpc_buffer_ref failed (domain 7, fd 18, ref -1)
05-26 17:00:48.794 ... dspqueue_write_noblock failed for queue ... (flags 0x3, num_buffers 2, message_length 312)
```

key 관찰:
- β-1.START 의 `Error 0x1: fastrpc_mmap failed` 라인 **완전 사라짐** — mmap GREEN
- 새 fail 지점: dspqueue_write 가 buffer ref count 를 decrement 시도 시 -1
  (즉 이전에 ref 가 0 또는 1 였다는 의미)
- packet flags `0x3` = `DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
  _INVALIDATE_RECIPIENT` (cache maintenance) — flags 자체는 정상값

### NOREG 포함/제거 ablation

| variant | mmap 결과 | dspqueue_write 결과 |
|---|---|---|
| `DEFAULT_FLAGS \| HEAP_NOREG` (llama.cpp 일치, β-1.MAP final) | GREEN | `AEE_EUNABLETOLOAD` |
| `DEFAULT_FLAGS only` (NOREG 제거) | GREEN (silent) | `AEE_EUNABLETOLOAD` |

→ NOREG bit 은 본 issue 와 무관. dispatch 단계 ref count 관리가 진짜 culprit.
final: llama.cpp 패턴 일치 위해 `NOREG` 유지.

## 변경 요약 (3 파일)

| 파일 | 변경 | LOC |
|---|---|---|
| `engine/src/backend/htp_fastrpc/host.rs` | `RPCMEM_HEAP_NOREG = 0x40000000` (0x80000000 정정), `FASTRPC_MAP_FD = 2` (16 정정), `RPCMEM_TRY_MAP_STATIC = 0x04000000` 추가, doc 주석 갱신 | ~+8 / -8 |
| `engine/src/backend/htp_fastrpc/buffer.rs` | RPCMEM_ALLOC_FLAGS doc 갱신 (β-1.MAP 검증 메모), fastrpc_mmap 주석 갱신 (NOMAP root cause 명시) | ~+10 / -10 |
| `engine/src/backend.rs` | `#[cfg(feature = "htp_fastrpc")] pub mod htp_fastrpc;` 선언 추가 (master merge 직후 누락 보정 — β-1.START commit 은 host build 만 했고 cross-build path 가 broken 이었음) | +3 |

## 다음 sub-sprint: β-1.QUEUE

**검증 게이트**: `dspqueue_write rc=0` + `dspqueue_read rc=0` +
`rsp.status == HTP_STATUS_OK` + `max_abs_err < 1e-3` vs CPU baseline.

### 작업 분해 (가설)

| 단계 | 작업 | 가설 |
|---|---|---|
| β-1.QUEUE.1 | `DspQueueBuffer.flags` 명시 세팅 (현재 0 → `DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER`/`_INVALIDATE_RECIPIENT` 조합). llama.cpp `ggml-hexagon.cpp:2242-2255` 의 `htp_req_buff_init` 의 dspqbuf_type 분기 그대로 transcribe. | flags=0 (CONSTANT 변종) 시 driver 가 buffer ref 를 다르게 관리할 가능성 |
| β-1.QUEUE.2 | `remote_register_buf_attr2(ptr, size, fd, 0)` 명시 dlsym + 호출 (libcdsprpc.so 에 export 확인됨) | mmap 후 추가 register 필요 가능성 |
| β-1.QUEUE.3 | 위 GREEN 후 logcat에서 `fastrpc_buffer_ref ref=-1` 미발생 + rsp.status 확인 | — |
| β-1.QUEUE.4 | rmsnorm correctness gate + report.md + commit + β-2.MM handoff | — |

**예상 시간**: 1.5 ~ 3h (flag 변경 + 측정 1, register_buf 추가 검토 1).

### 위임 권장

senior-implementer — dspqueue cache flag semantics + ref count driver
behavior 분석.

## Landmines / 미해결

- **`AEE_EUNABLETOLOAD` 의 wider 의미**: 0xe 는 일반 "unable to load" 에러 코드.
  driver 가 buffer descriptor 로부터 DSP-side 로 buffer 를 attach 하려다 fail.
  ref -1 은 그 fail 의 부산물. 진짜 root cause 는 attach 단계의 다른 조건일 가능성.
- **session leak 잔존**: β-1.START handoff 에 명시된 동일 issue. run_device.py 의
  cleanup 과 device-side 의 session release race. 5-10s wait 으로 우회 중.
- **`pub mod htp_fastrpc;` 누락은 β-1.START commit (58bb805a) 의 buggy 부분**:
  본 sprint 에서 보정. β-1.START 측정이 어떻게 통과했는지는 미스터리 —
  host build 만 했거나, 이전 staged binary 의 결과를 봤거나. cross-build path
  가 보정되어 본 sprint 가 자체 검증 가능.
- **β-1.START 의 `RPCMEM_HEAP_NOREG = 0x80000000` 은 사실 `RPCMEM_HEAP_DEFAULT`**.
  β-1.MAP 에서 정정. fastrpc_mmap fail 의 root cause 와는 무관 (mmap rc 는
  enum FASTRPC_MAP_FD 값에 좌우, NOREG 는 alloc 시점만 영향).

## 핵심 파일 인덱스

- 본 sprint commit: (commit 직후 갱신)
- host: `engine/src/backend/htp_fastrpc/host.rs` (3 const 정정, doc 갱신)
- buffer: `engine/src/backend/htp_fastrpc/buffer.rs` (alloc/mmap doc 갱신)
- backend mod 선언: `engine/src/backend.rs:29` (htp_fastrpc 모듈 export)
- QC fastrpc 오픈소스: `github.com/quic/fastrpc/master/inc/{rpcmem,remote}.h`
- 다음 진입점: handoff_q22_beta_map_2026_05_26.md
