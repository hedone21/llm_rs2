# Handoff: Q-2.2-α PoC 종결 (AMBER) → Q-2.2-β.IDL 진입

**작성**: 2026-05-26 (2026-05-26 grill-me 검토 후 정정)
**HEAD**: `222f5016 feat(htp-fastrpc): Q-2.2-α Phase 6 — stock S25 raw HTP path 부분 GREEN`
**worktree**: `.claude/worktrees/b5_trait_extension`
**다음 세션 진입 문장**: **"Q-2.2-β.IDL 진입 — qaic 자동 stub 추출 후 htp_iface_start 정확 구현 + rmsnorm dispatch GREEN"**

---

## ⚠ Status 정정 (grill-me 검토 결과)

**평가: AMBER** (이전 "부분 GREEN" 표현은 misleading).

근거:
- PoC 의 사용자 원 정의 = "간단한 MatMul 이 HTP NPU 에서 동작 + 정확성 검증" (= 실제 NPU compute 동작)
- 본 sprint 결과 = **NPU 의 HVX vector unit 한 줄도 실행 안 됨** (logcat: skel `libggml-htp-v79.so` dlopen 까지만, `hvx_fast_rms_norm_f32` 호출 0회)
- 본 sprint 의 사용자 원 의도 (3-way CPU/GPU/NPU 성능 비교) 부합도 = ~30% (NPU column SKIP)
- 인프라/handshake/capability 검증 GREEN, dispatch/execution 검증 MISS

본 sprint 의 산출물 = **재사용 가능한 인프라 + capability 검증** 으로 한정.

---

## TL;DR

Q-2.2-α PoC (8 Phase) 완료. 검증 layer:

| 검증 | 상태 |
|---|---|
| (a) llama.cpp ggml-hexagon path stock S25 동작 | ✓ |
| (b) host Rust binding 1772 LOC 으로 4-step handshake + GET_URI + handle_open + dspqueue create/export | ✓ |
| (c) 자체 빌드 DSP skel binary `libggml-htp-v79.so` (271 KB) stock S25 DSP-side dlopen | ✓ |
| (d) **DSP-side HVX 함수 실제 실행** | **✗ (0 회)** |
| (e) MatMul correctness on HTP NPU | ✗ |

PoC 본질 = (d)(e) 까지 검증. 본 sprint 는 (a)(b)(c) 까지. **β 단계로 분할 진행 필요** — 단순 1 sprint 보강 아닌 sprint scope 재정의 (rmsnorm IDL PoC → matmul dispatch + 정확성).

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

## 다음 작업: β 단계 분할 (β-1.IDL → β-2.MM)

PoC 본질 = MatMul HTP NPU GREEN. 단일 sprint 가 아닌 2 sub-sprint 분할:

### β-1.IDL: htp_iface_start IDL stub + rmsnorm dispatch GREEN (3-4h)

**검증 게이트**: `microbench_htp_rmsnorm` 재실행 → Stage 5 (dspqueue_write) rc=0 + DSP rsp.status == HTP_STATUS_OK + correctness `max_abs_err < 1e-3` vs CPU baseline (gamma=1).

작업 분해 (qaic 가용성 사전 verify 완료, `/opt/hexagon/6.4.0.2/ipc/fastrpc/qaic/bin/qaic`):

| 단계 | 작업 | 추정 | Risk | Fallback |
|---|---|---|---|---|
| β-1.1 | Docker 안 `qaic` 로 `htp_iface.idl` 컴파일 → `htp_iface_stub.c` 자동 생성 + `htp_iface_start` 의 `sc` macro + `pra` packing 코드 추출 | 30 min | 낮음 (qaic 동작 verify 됨) | SDK header `remote.h` 의 macro manual 분석 (+1-2h) |
| β-1.2 | `engine/src/backend/htp_fastrpc/host.rs::try_start_iface` 정확 구현 교체 (`remote_arg pra[]` Rust struct + `remote_handle64_invoke(handle, sc, pra)`) | 1-1.5h | 중간 (C union → Rust repr(C) 정확 매핑) | n_hvx sweep (1/2/4) |
| β-1.3 | host build + clippy + Android cross-build | 15 min | 낮음 | — |
| β-1.4 | microbench_htp_rmsnorm S25 재실행 + correctness gate | 30 min | 중간 (packet schema mismatch 시 rsp.status != OK 가능) | logcat 분석 + llama.cpp `htp_tensor_init` byte 비교 |
| β-1.5 | report.md + commit + β-2.MM handoff | 30 min | 낮음 | — |

n_hvx default = 4 (llama.cpp `opt_nhvx`).

### β-2.MM: matmul Q-proj HTP NPU GREEN + 정확성 (4-5h)

**PoC 종결 게이트**: HTP_OP_MUL_MAT (=4) 호출 → output max_abs_err < 1e-3 vs CPU baseline + 3-way timing complete (CPU/OpenCL/HTP).

작업 분해:
- HtpGeneralReq.op = HTP_OP_MUL_MAT (=4)
- src0 = input tensor [1, K], src1 = weight tensor [K, N] (F32 first; Q4_0 dequant 은 추후)
- dst = output [1, N]
- n_bufs = 3 (input + weight + output)
- weight RpcmemBuffer alloc + upload (~9MB for Qwen Q-proj [1536, 1536])
- shape: `[1, 1536] × [1536, 1536] = [1, 1536]` (Qwen2.5-1.5B Q-proj, "간단한 MatMul" 정의)
- microbench_htp_matmul 신설 (~400 LOC) 또는 microbench_htp_rmsnorm 에 matmul case 추가
- llama.cpp `matmul-ops.c::matmul_f32` 의 op_params + packet schema 직접 reference

### 두 sub-sprint 합산 추정: 7-9h

각 sub-sprint 별로 별 session + commit + handoff. β-2.MM 완료 시 본 PoC 진짜 GREEN.

### 위임 권장

각 sub-sprint senior-implementer 위임 (FFI + unsafe + IDL + HVX schema 매핑).

### 위험 시나리오

- β-1 IDL stub fail (qaic 출력이 기대와 다름) → SDK header manual 분석으로 fallback (+1-2h)
- β-1 PASS 후 β-2 packet schema mismatch → llama.cpp `init_matmul_req` 의 byte-level 비교 verify
- HVX op 가 stock device 에서 차단 (signed-only 강제) → 가능성 낮음 (libggml-htp-v79.so dlopen 자체는 PASS 했으므로 unsigned PD 정책 적용 중)

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
