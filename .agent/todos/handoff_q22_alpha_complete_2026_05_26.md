# Handoff: Q-2.2-α PoC 종결 → Q-2.2-β.IDL 진입

**작성**: 2026-05-26
**HEAD**: `222f5016 feat(htp-fastrpc): Q-2.2-α Phase 6 — stock S25 raw HTP path 부분 GREEN`
**worktree**: `.claude/worktrees/b5_trait_extension`
**다음 세션 진입 문장**: **"Q-2.2-β.IDL 진입 — htp_iface_start IDL stub 정확 구현"**

---

## TL;DR

Q-2.2-α PoC (8 Phase) 완료. 핵심 검증 3건 모두 GREEN — (a) llama.cpp ggml-hexagon path stock S25 동작 / (b) 우리 host Rust binding 1772 LOC 으로 4-step handshake + GET_URI + handle_open + dspqueue create/export 모두 stock device PASS / (c) 자체 빌드 DSP skel binary `libggml-htp-v79.so` (271 KB) DSP-side dlopen 성공 (logcat 직접 증명). 미완: dspqueue_write rc=0x48 (DSP worker 미시작, htp_iface_start IDL stub 누락). β 단계 시작 작업으로 IDL stub 정확 구현 → 진짜 HTP rms_norm dispatch GREEN.

---

## 진행 상태

8 Phase 모두 완료:

| Phase | 작업 | Commit | 결과 |
|---|---|---|---|
| 0 | Docker SDK pull (8 GB image) | — | GREEN |
| 1 | spec/htp_fastrpc.md + arch/htp_fastrpc.md (Path B patch) | `3fb23a65` | GREEN, 5 INV |
| 2 | llama.cpp ggml-hexagon Docker 빌드 | — | `libggml-htp-v79.so` 271 KB |
| 3 | host binding 5 파일 1339 LOC, 13 test PASS | `4feb0b0c` | GREEN |
| 4 | HtpFastrpcBackend trait impl, 14 test PASS | `91b0c84e` | GREEN |
| 5 | microbench_htp_rmsnorm 3-way 585 LOC | `5842fe74` | GREEN |
| 6 | S25 deploy + 실측 + Phase 6 patch (GET_URI + n_bufs=2) | `222f5016` | **부분 GREEN** (Stage 1-4 OK, Stage 5 fail) |
| 7 | handoff + commit + push + notify | (본 문서) | 진행 중 |

**총 LOC 신설** (Q-2.2 sprint 전체):
- engine/src/backend/htp_fastrpc/ (5 파일): 1772 LOC
- engine/microbench/htp_fastrpc_dryrun.rs: 166 LOC
- engine/microbench/htp_rmsnorm.rs: 580 LOC
- spec/htp_fastrpc.md + arch/htp_fastrpc.md: 476 LOC
- papers/.../report.md × 3 + logs: 측정 데이터
- **합계**: ~3,000 LOC

---

## 측정 결과 (Phase 6 S25 실측)

### Stage 1-4 GREEN (logcat 직접 증명)

```
I microbench_htp_rmsnorm: ... multidsplib_env_init: libcdsprpc.so loaded
I microbench_htp_rmsnorm: ... remote_session_control Unsigned PD enable 1 request for domain 7
I microbench_htp_rmsnorm: ... open success: libdspqueue_rpc_skel.so handle 0x2d2d60 domain 7
I microbench_htp_rmsnorm: ... open success: libggml-htp-v79.so handle 0x2d2f60   ← ★ key
I microbench_htp_rmsnorm: ... dspqueue_close: closed Queue 0, 0xb400007e3943f010, DSP 0x00000000
```

`E adsprpc` 라인 0건. stock S25 가 자체 빌드 unsigned PD skel 을 받아들임.

### Stage 5 fail signature

```
[3/3] QNN HTP NPU (FastRPC + libggml-htp-v79.so dspqueue)
  host: libcdsprpc.so=/vendor/lib64/libcdsprpc.so, domain_id=7, session_id=1, queue_id=33002528702464
  SKIP — dspqueue_write: htp_fastrpc: AEEResult=unknown (0x48, raw=72)
```

`rc=0x48 (72)` — DSP-side worker thread 가 dspqueue 를 attach 안 함. `htp_iface_start(handle, sess_id, queue_id, n_hvx)` IDL invocation 이 host queue 와 DSP worker 를 connect 하는 step 인데 본 sprint 에서 누락.

---

## 다음 작업: Q-2.2-β.IDL 진입점

### 단일 task

**`htp_iface_start` IDL stub 정확 구현** (1 method, ~150 LOC)

검증 게이트: `microbench_htp_rmsnorm` 재실행 → Stage 5 (dspqueue_write) rc=0 GREEN + correctness `max_abs_err < 1e-3` vs CPU baseline (gamma=1).

### 구현 detail

1. **`REMOTE_SCALARS_MAKEX` macro 값 확정**:
   - llama.cpp 의 IDL stub binary 또는 Hexagon SDK 의 `remote.h` 에서 직접 추출
   - `REMOTE_SCALARS_MAKEX(attr, method, in_h, out_h, in_b, out_b)` = `((attr)<<24 | (method)<<16 | (in_h)<<12 | (out_h)<<8 | (in_b)<<4 | (out_b))`
   - `htp_iface_start` = method idx 0 (htp_iface.idl 의 첫 method), `attr=0, in_h=0, out_h=0`
   - parameter type: `sess_id u32 (in), dsp_queue_id u64 (in), n_hvx u32 (in)` → primitive `in_b=3, out_b=0` 일 가능성, 또는 buffer `in_b=0, out_b=0` 가능성
   - 정확값 검증: Docker 안에서 `qaic -E htp_iface.idl` 실행 후 자동 생성 stub C source 읽기 (~10분 작업)

2. **`remote_arg` struct 정확 layout**:
   ```c
   struct remote_arg {
       union {
           struct remote_buf buf;
           struct remote_handle h;
           struct remote_handle64 h64;
       } u;
       struct remote_buf {
           void *pv;
           size_t nLen;
       };
   };
   ```
   sess_id (u32) 와 n_hvx (u32) 는 primitive — `pra[].buf.pv = &val; pra[].buf.nLen = sizeof(val)` 형태 또는 직접 scalar packing.

3. **호출 site**: `engine/src/backend/htp_fastrpc/host.rs::HtpFastrpcHost::new` 의 Step 7 (`htp_iface_start`) 위치. 현재 `try_start_iface` 가 best-effort 반환 — 그것을 정확 호출로 교체.

4. **`n_hvx` 값 선택**: llama.cpp default = 4 (`opt_nhvx=4`). 그대로 차용.

### 위임 권장

Senior implementer 위임 (FFI + unsafe + IDL stub):
- Hexagon SDK Docker 안에서 `qaic` 실행 + stub C 분석
- `host.rs` 의 `try_start_iface` 교체 (~150 LOC)
- `microbench_htp_rmsnorm` 재실행 (S25)
- correctness gate (max_abs_err < 1e-3) + timing 측정

작업일 추정: 2-3 시간 (qaic 출력 분석이 가장 큰 unknown).

### β 진입 결정 게이트

- htp_iface_start GREEN → β.A: 진짜 HTP rms_norm GREEN → 3-way 측정 complete
- htp_iface_start FAIL → β.B 차선: llama.cpp 빌드한 `libggml-hexagon.so` host shim 의 `init_dsp` symbol 을 dlopen 해서 setup → 우리 binding 은 그 후 dspqueue handle 만 차용 (더 복잡하지만 IDL macro 의존 0)

---

## Landmines / 미해결

- **FASTRPC_GET_URI 누락** (Phase 6 first run 실패 root cause): host.rs patch 로 해결. 향후 다른 sprint 에서 raw URI 직접 사용 시 default domain 3 fallback → `Error 0xe domain >= 0` 동일 함정.
- **llama.cpp `HTP_OP_RMS_NORM` 은 gamma 미적용** (unary-ops.c::rms_norm_f32). 본 sprint 의 3-way correctness 는 CPU/OpenCL baseline 도 gamma=1.0 fill 로 비교. β 단계의 진짜 gamma 적용 RMSNorm 은 HTP_OP_MUL chain 필요 — 별 작업.
- **dspqueue_write rc=0x48** (DSP worker not attached): 본 sprint 의 미해결 issue. β.IDL 진입점.
- **`htp_iface.idl` start method 의 `attr` byte**: SDK header 의 attr enum 정확값 미확정 (보통 0 이지만 IDL annotation 따라 다름). qaic 자동 stub 실행 시 정확값 도출 필요.
- **path A (자체 IDL + qaic + HVX skel) 는 본 sprint 에서 미진입**. β 의 IDL stub 작업 성공 후에도 path A 는 별 sprint (matmul 등 자체 op 작성 시점에 필요).
- **dry-run prototype (microbench_htp_fastrpc_dryrun) 의 fastrpc mod**: 본 sprint backend skeleton 으로 흡수 완료. dry-run 자체는 그대로 microbench 유지 — β 진입 후 polishing 또는 삭제 결정.

---

## 핵심 파일 인덱스

- 본 sprint 의 모든 commit: `da22ea82` (Q-2.1) → `446487fa` (handoff) → `4138efd4` (dry-run B) → `2e09efbc` (interface matrix) → `3fb23a65` (spec/arch) → `4feb0b0c` (Phase 3) → `91b0c84e` (Phase 4) → `5842fe74` (Phase 5) → `222f5016` (Phase 6)
- spec: `spec/htp_fastrpc.md` (219 LOC, 5 INV)
- arch: `arch/htp_fastrpc.md` (236 LOC, mermaid 3개)
- host binding: `engine/src/backend/htp_fastrpc/{host,idl,buffer,error,mod}.rs` (~1772 LOC)
- microbench: `engine/microbench/htp_rmsnorm.rs` (580 LOC)
- Phase 6 report: `papers/eurosys2027/_workspace/experiment/qnn_q22_alpha_rmsnorm_2026_05_26/report.md`
- 모든 측정 stdout/logcat: 같은 디렉토리
- 진입 base 분석: `papers/.../backend_interface_matrix.md` (60 method 분류 + 16 microbench plan)
- llama.cpp 빌드 산출물: `/home/go/Workspace/llama.cpp/build-snapdragon/ggml/src/ggml-hexagon/libggml-htp-v{68,69,73,75,79,81}.so` + `libggml-hexagon.so`
