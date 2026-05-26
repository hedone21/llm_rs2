# Handoff: Q-2.2-β.START 완료 (lifecycle GREEN, dispatch BLOCKED) → β-1.MAP 진입

**작성**: 2026-05-26
**HEAD**: (커밋 직후 갱신)
**worktree**: `.claude/worktrees/b5_trait_extension`
**다음 세션 진입 문장**: **"Q-2.2-β.MAP 진입 — Hexagon SDK rpcmem.h header dump 로 RPCMEM_HEAP_NOREG 정확값 확정 + fastrpc_mmap GREEN + rmsnorm correctness PASS"**

---

## TL;DR

β-1.START 의 lifecycle layer (htp_iface_start/stop) **GREEN**:
- `htp_iface_stub.c` 자동 생성 stub 의 mid/sc/pra 정밀 transcribe
- start sc = `0x0201_0000`, stop sc = `0x0300_0000`, enable_etm = `0x0400_0000`, disable_etm = `0x0500_0000`
- primIn 24 byte layout (u32 sess_id @0, u64 queue_id @8, u32 n_hvx @16)
- `iface_started: AtomicBool` 로 idempotent + `Drop` 에서 try_stop_iface 호출
- S25 logcat 에서 `htp_iface_start: OK (n_hvx=0)` 확인

β-1 의 보다 큰 "rmsnorm dispatch GREEN" goal 은 별 issue **BLOCKED**:
- `fastrpc_mmap(domain=7, fd=18, ptr, 0, alloc_size, FASTRPC_MAP_FD=16)` 가
  `rc=0x1 ioctl ret=-1 errno EIO user_err=AEE_EBADPARM` 로 fail
- NOREG flag 0x80000000 / 0x80000 두 값 모두 동일 fail
- 정확한 `RPCMEM_HEAP_NOREG` 값 + 추가 조건은 SDK header 직접 dump 후 결정

본 sprint 결과 = β-1 lifecycle layer 완전 통합. dispatch 는 별 sub-sprint
β-1.MAP 으로 분리.

---

## 진행 상태

| Task | 상태 | 작업 | 증거 |
|---|---|---|---|
| β-1.1 stub layout 추출 | ✓ | host.rs:267-292 의 method id/sc encoding 표 + 주석 | 자동 생성 stub `_stub_method` 의 sc/pra 정밀 transcribe |
| β-1.2 host lifecycle 4 method | ✓ | `try_start_iface` + `try_stop_iface` + `try_{en,dis}able_etm` + AtomicBool + Drop | 2 unit test PASS (sc encoding + RemoteArgBuf size) |
| β-1.2-b mod.rs caller | ✓ | HtpFastrpcBackend::new 가 host.try_start_iface(n_hvx) 직접 호출 | 16 unit test PASS |
| β-1.3 build + clippy | ✓ | host release + Android cross-build PASS, lib clippy clean | cargo test/clippy 출력 |
| β-1.4 S25 실행 | △ | lifecycle GREEN, dispatch BLOCKED on mmap | stdout: `htp_iface_start: OK`; logcat: invoke 결과 silent (no error on method 2) |
| β-1.5 report.md + commit | ✓ | `papers/.../qnn_q22_beta_start_2026_05_26/report.md` 작성 | 본 handoff 와 동시 commit |
| β-1.6 RpcmemBuffer fastrpc_mmap | △ (코드 통합 GREEN, 실행 fail) | 4K page align + NOREG flag + Drop munmap | rc=0x1 — 정확한 NOREG 값 미상 |

---

## 다음 작업: β-1.MAP

**검증 게이트**: microbench_htp_rmsnorm 재실행 → `fastrpc_mmap rc=0` +
dspqueue_write/read 성공 + `rsp.status == HTP_STATUS_OK` + max_abs_err < 1e-3
vs CPU baseline.

### 작업 분해

| 단계 | 작업 | 추정 | Risk | Fallback |
|---|---|---|---|---|
| β-1.MAP.1 | **Hexagon SDK `rpcmem.h` 직접 dump** — `/opt/hexagon/6.4.0.2/ipc/fastrpc/rpcmem/inc/rpcmem.h` 의 `RPCMEM_HEAP_NOREG`, `RPCMEM_TRY_MAP_STATIC`, `RPCMEM_HEAP_*` 상수 정확값 추출. 사용자가 Docker 권한 또는 host SDK install 결정. | 15 min | 낮음 | 대안: libggml-hexagon.so disassembly 에서 immediate 추출 |
| β-1.MAP.2 | `host.rs::RPCMEM_HEAP_NOREG` 값 갱신 + `buffer.rs::RPCMEM_ALLOC_FLAGS` 검토 (다른 flag 필요 여부 확인 — TRY_MAP_STATIC 등) | 10 min | 낮음 | — |
| β-1.MAP.3 | S25 재실행 → fastrpc_mmap rc=0 확인 | 10 min | 낮음 | mmap 여전히 fail 시 대안 시도 |
| β-1.MAP.4 (조건부) | 위 step 2/3 의 fallback path: `fastrpc_buffer_map` (별개 API) 시도 또는 explicit `fastrpc_mmap` skip 후 rpcmem auto-register 의존 (단 Original error 가 그 경로에 있었음) | 30-60 min | 중간 | β-1.MAP 종결 + retro 분석 |
| β-1.MAP.5 | rmsnorm correctness gate + report.md 보강 + commit + β-2.MM handoff | 30 min | 낮음 | — |

### 위임 권장

senior-implementer — SDK header read + FFI binding + 정밀 driver-side
flag mapping.

### 위험 시나리오

- **fastrpc_buffer_map 가 SDK header 에 별 entry 로 존재하는 경우**: dlsym
  + signature 결정 + RpcmemBuffer 통합 추가 (~+50 LOC)
- **stock S25 unsigned PD 가 사용자 fastrpc_mmap 자체를 차단**: 가능성 낮으나
  검증 필요. logcat 의 "open_shell failed for domain 3" 경고는 unrelated
  (CDSP domain 7 은 정상 진행됨).
- **Docker SDK 가 SDK 6.4.0.2 가 아닌 다른 버전**: 변종 flag value 가능성.
  llama.cpp 빌드 사용 환경 = 6.4.0.2 (기존 handoff 확인).

---

## Landmines / 미해결

- **NOREG=0x80000000 코드 잔존**: 본 sprint 의 host.rs:`RPCMEM_HEAP_NOREG`
  상수는 공식 SDK 확정 후 갱신 필요. mmap 자체가 fail 라 NOREG bit 의
  영향은 검출 불능.
- **session leak across runs**: 첫 run 직후 두 번째 run 의 첫 시도가
  `remote_handle64_open uri=...` fail (session_id=1 in-use). Drop 의
  remote_handle64_close 가 즉시 release 안 되는 timing race. β-1 scope 외,
  workaround: 직접 shell run 사이 1~2s 대기.
- **mod.rs::HtpFastrpcBackend::new 가 try_start_iface 실패 시 fatal**:
  이전 best-effort (lazy init fallback) 정책 제거. caller (CLI 등) 가
  backend 미사용 결정 시 fatal 처리해야 함. β-2.MM 이전에 CLI 통합 결정
  필요 — backend 선택 plumbing 의 graceful 분기 위치 미정.
- **logcat 의 `Error 0x80000414: ... method 3 on libdspqueue_rpc_skel.so
  sc 0x3010100`**: dspqueue_close 의 일부, htp_iface_stop (sc 0x03000000)
  과 별개. handoff 의 stop 호출 추적은 stdout 의 `dspqueue_close: closed
  Queue` 메시지로 검증 가능.
- **path A (자체 HVX skel) 와 path B (llama.cpp skel 차용) 모두 본 sprint
  결과로 영향 받지 않음**: 양 path 모두 host-side rpcmem alloc/mmap 동일
  무관. β-1.MAP GREEN 이후 양 path 모두 진행 가능.

---

## 핵심 파일 인덱스

- 본 sprint 의 commit: (commit 직후 갱신)
- host: `engine/src/backend/htp_fastrpc/host.rs` (+150 LOC: lifecycle 4 method, AtomicBool, RemoteArgBuf, sc helper, RPCMEM_HEAP_NOREG, Drop 갱신, 2 unit test)
- backend: `engine/src/backend/htp_fastrpc/mod.rs` (try_start_iface stub 제거, host.try_start_iface 직접 호출, HTP_FASTRPC_N_HVX_DEFAULT const)
- buffer: `engine/src/backend/htp_fastrpc/buffer.rs` (fastrpc_mmap + munmap + alloc_size 필드 + 4K align)
- microbench: `engine/microbench/htp_rmsnorm.rs` (host 직후 try_start_iface 명시 호출)
- report: `papers/eurosys2027/_workspace/experiment/qnn_q22_beta_start_2026_05_26/report.md`
- llama.cpp 자동 생성 stub: `/home/go/Workspace/llama.cpp/build-snapdragon/ggml/src/ggml-hexagon/htp_iface_stub.c`
- Hexagon SDK header (다음 세션에서 dump): `/opt/hexagon/6.4.0.2/ipc/fastrpc/rpcmem/inc/rpcmem.h` (Docker 안)
