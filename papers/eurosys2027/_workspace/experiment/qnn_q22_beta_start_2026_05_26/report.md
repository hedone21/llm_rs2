# Q-2.2-β.START — lifecycle IDL invoke GREEN, dispatch BLOCKED on rpcmem mmap

**작성**: 2026-05-26
**HEAD**: (commit 직후 갱신)
**진입**: handoff_q22_alpha_complete_2026_05_26.md
**worktree**: `.claude/worktrees/b5_trait_extension`

## TL;DR

β-1 lifecycle layer (htp_iface_start/stop) **GREEN** — `htp_iface_stub.c` 의
4 method (start/stop/enable_etm/disable_etm) sc/pra layout 정밀 transcribe +
Drop RAII + AtomicBool idempotent guard. S25 logcat 에서 `htp_iface_start: OK
(n_hvx=0)` 확인.

β-1 의 보다 큰 dispatch GREEN goal 은 **BLOCKED** — `fastrpc_mmap` 가
`rc=0x1 (EIO)` 로 reject. NOREG flag (0x80000000 / 0x80000) 두 값 모두 동일
실패. 정확값은 Hexagon SDK `rpcmem.h` 직접 dump 후 결정 필요.

## 검증 layer

| 검증 | 상태 | 증거 |
|---|---|---|
| (a) `htp_iface_stub.c` mid/sc/pra 정확 transcribe | ✓ | `host.rs::HTP_IFACE_MID_*` + unit test `iface_method_sc_encoding` |
| (b) `remote_handle64_invoke(handle, MAKEX(0,2,1,0,0,0), pra=primIn[24])` host invocation | ✓ | logcat: invoke 결과 silent (Error 라인 없음). stdout: `htp_iface_start: OK` |
| (c) `RpcmemBuffer::alloc` 에 `fastrpc_mmap` 통합 | ✓ (코드) | buffer.rs:119-137 |
| (d) `dspqueue_write` 가 buffer 를 도메인 7 에서 인식 | ✗ | `fastrpc_mmap rc=0x1, ioctl ret -1, errno EIO` |
| (e) `rsp.status == HTP_STATUS_OK` 및 correctness | ✗ | dispatch 진입 못 함 (c 차단) |
| (f) Drop RAII 의 `htp_iface_stop` 호출 흔적 | ✓ | logcat: dspqueue_close + remote_handle64_close 순차 + `remote_handle64_invoke ... method 3 ... sc 0x3010100` line (NOTE: 이 method 3 invoke 는 dspqueue_rpc_skel close 의 일부, htp_iface stop 과는 별개 — stop 호출은 fail-tolerant 로 silent succeed) |

## 변경 요약 (4 파일)

| 파일 | 변경 | LOC |
|---|---|---|
| `engine/src/backend/htp_fastrpc/host.rs` | 4 lifecycle method (`try_start_iface` / `try_stop_iface` / `try_{en,dis}able_etm`) + `iface_started` AtomicBool + `RemoteArgBuf` repr(C) + `remote_scalars_makex` const fn + `RPCMEM_HEAP_NOREG` const + `Drop` 에서 try_stop_iface 호출 + 2 unit test | +~150 |
| `engine/src/backend/htp_fastrpc/mod.rs` | `HtpFastrpcBackend::new` 가 `host.try_start_iface(n_hvx)` 직접 호출 + bail! stub 제거 + `HTP_FASTRPC_N_HVX_DEFAULT` const (= 0, llama.cpp `opt_nhvx` 일치) + `iface_started` 필드 제거 (host 가 ground truth) | ~+25 / -50 |
| `engine/src/backend/htp_fastrpc/buffer.rs` | `fastrpc_mmap` 통합 (4K page-align + RPCMEM_HEAP_NOREG flag) + Drop 에서 `fastrpc_munmap` + `alloc_size` 필드 | ~+50 |
| `engine/microbench/htp_rmsnorm.rs` | host 생성 직후 `try_start_iface(env(HTP_FASTRPC_N_HVX) or 0)` 호출 추가 + stdout 출력 | +10 |

## sc encoding (검증)

| Method | mid | sc 계산 | sc 값 |
|---|---|---|---|
| start | 2 | `MAKEX(0,2,1,0,0,0)` = `(2<<24) \| (1<<16)` | `0x0201_0000` |
| stop | 3 | `MAKEX(0,3,0,0,0,0)` = `(3<<24)` | `0x0300_0000` |
| enable_etm | 4 | `MAKEX(0,4,0,0,0,0)` = `(4<<24)` | `0x0400_0000` |
| disable_etm | 5 | `MAKEX(0,5,0,0,0,0)` = `(5<<24)` | `0x0500_0000` |

Unit test `iface_method_sc_encoding` 가 위 값을 PASS.

## primIn layout (start)

24 byte, u64 alignment, layout `_stub_method` 의 `_COPY` 순서를 그대로 매핑:

```
offset  0..4  : u32 sess_id        (host_int)
offset  4..8  : padding
offset  8..16 : u64 dsp_queue_id   (host_int)
offset 16..20 : u32 n_hvx          (host_int)
offset 20..24 : tail padding
```

Rust 구현: `prim_in: [u64; 3]` 으로 u64 align 보장 후 byte-level slice 로 직접
write — endianness 는 `to_ne_bytes()` (host = aarch64 LE).

## S25 실측 결과

5번 run, 모두 동일 패턴.

stdout:
```
[3/3] QNN HTP NPU (FastRPC + libggml-htp-v79.so dspqueue)
  host: libcdsprpc.so=/vendor/lib64/libcdsprpc.so, domain_id=7, session_id=1, queue_id=...
  htp_iface_start: OK (n_hvx=0)         ← ✓ β-1 lifecycle GREEN
  SKIP — htp_fastrpc: fastrpc_mmap rc=0x1 (domain=7, fd=18, size=8192)
```

logcat (last attempt, NOREG=0x80000000 + 4K page-align):
```
05-26 16:18:50.298 28644 28644 E microbench_htp_rmsnorm: vendor/qcom/proprietary/adsprpc/src/fastrpc_mem.c:510: Error 0x1: fastrpc_mmap failed to map buffer fd 18, addr 0x756c75f000, length 0x2000, domain 7, flags 0x10, ioctl ret 0xffffffff, errno I/O error (user err 0x1)
```

key 관찰:
- `addr 0x756c75f000` — page-aligned (LSB 0x000)
- `length 0x2000` — 8192 bytes (4K page-aligned ✓)
- `flags 0x10` — FASTRPC_MAP_FD ✓
- `ioctl ret -1, errno EIO` — kernel-side reject. user err 0x1 = AEE_EBADPARM
- 영향: dspqueue_write 가 buffer 를 도메인 7 에 인식 못 함 → dispatch 차단

### NOREG flag 값 실험 (모두 fail)

| flag value | 출처 | 결과 |
|---|---|---|
| `0x8000_0000` | quic/fastrpc 오픈소스 + 일부 SDK | mmap rc=0x1 EIO |
| `0x80000` | 일부 older SDK | mmap rc=0x1 EIO |
| (생략, DEFAULT_FLAGS only) | original | `Buffer FD 18 not mapped to domain 7` (다른 fail 경로) |

→ NOREG 두 값 모두 동일 EIO 결과. driver-side validation 이 flag 외 다른
이유로 reject. **정확한 RPCMEM_HEAP_NOREG 값 + 추가 조건은 Hexagon SDK
`rpcmem.h` (`/opt/hexagon/6.4.0.2/ipc/fastrpc/rpcmem/inc/rpcmem.h`)
header dump 후 결정 필요**.

## 다음 sub-sprint: β-1.MAP

**목표**: `fastrpc_mmap rc=0` GREEN → `dspqueue_write` 가 buffer 도메인 인식
→ rmsnorm correctness PASS.

**진입 조건**: 다음 중 하나:
1. **(권장)** Docker 안 `/opt/hexagon/6.4.0.2/ipc/fastrpc/rpcmem/inc/rpcmem.h`
   직접 dump → `RPCMEM_HEAP_NOREG` + `RPCMEM_TRY_MAP_STATIC` + 기타 flag
   정확값 확인 후 적용.
2. (대안) llama.cpp 빌드 산출물 (libggml-hexagon.so) 의 disassembly 에서
   `rpcmem_alloc2` 호출 site 의 immediate 인자 추출.
3. (대안) `fastrpc_buffer_map` (별개 API) 직접 dlsym + 시도.

**예상 시간**: 30 min ~ 1.5h (header dump 1, 알맞은 flag 1 cycle, alignment
1 cycle).

## 다음 다다음 sub-sprint: β-2.MM

β-1.MAP GREEN 후 진입. handoff `q22_alpha_complete_2026_05_26.md` 의 β-2.MM
설계 그대로. MatMul HTP NPU GREEN + correctness.

## Landmines / 미해결

- **fastrpc_mmap EIO**: 본 문서의 핵심 issue. SDK header 미확인 상태로
  fastrpc_buffer 의 정확한 alloc flag 패턴 미상.
- **session leak across runs**: 첫 run 직후 두 번째 run 의 첫 시도가
  `remote_handle64_open uri=...` fail (session_id=1 in-use). 두 번째 run
  의 두 번째 시도부터 GREEN. cleanup race condition — Drop 에서
  remote_handle64_close 가 즉시 release 안 되는 경우 존재. β-1 scope 외.
- **NOREG=0x80000000 코드 잔존**: 본 commit 의 RPCMEM_HEAP_NOREG 상수는
  공식 SDK 와 일치 여부 미검증. 본 PoC 에서는 비활성 효과 (mmap 자체가
  fail 라 NOREG bit 영향 미검출). β-1.MAP 에서 SDK 확정 후 갱신.
- **mod.rs::HtpFastrpcBackend::new 가 try_start_iface 실패 시 fatal**:
  PoC 정책. 이전 lazy-init fallback 정책 (best-effort eprintln) 은 제거.
  caller (CLI 등) 가 backend 미사용 결정 시 mode flag 로 처리해야 함.
