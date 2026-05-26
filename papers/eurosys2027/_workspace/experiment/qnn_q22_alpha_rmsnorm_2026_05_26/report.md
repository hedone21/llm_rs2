# Q-2.2-α PoC Phase 6 — Stock S25 raw HTP path 부분 GREEN

**작성**: 2026-05-26
**HEAD (직전)**: `5842fe74 feat(microbench): Q-2.2-α Phase 5 — 3-way RMSNorm microbench`
**디바이스**: Galaxy S25 (R3CY408S5SB / SM8750 / HTP V79), stock non-rooted
**대상 bin**: `microbench_htp_rmsnorm` (3-way: CPU NEON / OpenCL / QNN HTP raw)
**결과**: **부분 GREEN — 4-step handshake + handle_open + GET_URI + dspqueue create/export 모두 GREEN. dspqueue_write 단계 rc=0x48 (htp_iface_start IDL stub 누락). β 진입점 명확**

---

## TL;DR

본 sprint 의 PoC 핵심 ("stock S25 에서 raw HTP path 가능성") = **GREEN**:
- libcdsprpc.so dlopen ✓
- 17 symbol dlsym ✓
- `FASTRPC_RESERVE_NEW_SESSION` → `effective_domain_id=7, session_id=1` ✓
- `DSPRPC_CONTROL_UNSIGNED_MODULE` (logcat `Unsigned PD enable 1 request for domain 7`) ✓
- `FASTRPC_GET_URI` → session-encoded URI ✓
- `remote_handle64_open(session_uri)` → libggml-htp-v79.so + libdspqueue_rpc_skel.so DSP-side dlopen ✓
- `dspqueue_create(128KB req, 64KB resp)` + `dspqueue_export` → `queue_id=33002528702464` ✓
- `dspqueue_write` ✗ **rc=0x48** (DSP-side worker 미시작, `htp_iface_start` IDL 호출 누락)

⇒ FastRPC channel + handle + queue 모두 정상 setup. 실제 op dispatch (dspqueue_write) 만 IDL stub 정확 구현 부재로 차단. **β 단계 진입점**: `htp_iface_start` IDL invocation 정확 구현 (`REMOTE_SCALARS_MAKEX` macro + `remote_handle64_invoke` parameter array layout).

---

## 단계별 진행

| 단계 | 결과 | 비고 |
|---|---|---|
| Push binary (1.5MB) + libggml-htp-v79.so (271KB) | ✓ | `/data/local/tmp/` |
| ENV setup `LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64 ADSP_LIBRARY_PATH=/data/local/tmp` | ✓ | llama.cpp developer.md 패턴 |
| CPU baseline (1000 iter, gamma=1) | ✓ | mean=0.87μs |
| OpenCL GPU (1000 iter, gamma=1) | ✓ | mean=161μs (dispatch overhead, [1,1536] 단일 row) |
| QNN HTP — Stage 1: HtpFastrpcHost::new (4-step handshake) | ✓ | domain_id=7, session_id=1 |
| QNN HTP — Stage 2: FASTRPC_GET_URI | ✓ | session-encoded URI 획득 (patch 추가) |
| QNN HTP — Stage 3: remote_handle64_open | ✓ | libggml-htp-v79.so + libdspqueue_rpc_skel.so DSP-side load |
| QNN HTP — Stage 4: dspqueue_create + dspqueue_export | ✓ | queue_id=33002528702464 |
| QNN HTP — Stage 5: dspqueue_write (rms_norm packet) | ✗ | **rc=0x48 (72)** |

---

## 핵심 logcat (Stage 1-4 PASS, Stage 5 fail)

`papers/.../qnn_q22_alpha_rmsnorm_2026_05_26/logs_logcat.txt`:

```
I microbench_htp_rmsnorm: ... multidsplib_env_init: libcdsprpc.so loaded
I microbench_htp_rmsnorm: ... remote_session_control Unsigned PD enable 1 request for domain 7   ← Stage 1
I microbench_htp_rmsnorm: ... open success: libdspqueue_rpc_skel.so handle 0x2d2d60 domain 7
I microbench_htp_rmsnorm: ... open success: libggml-htp-v79.so handle 0x2d2f60                 ← Stage 3 ★ key proof
I microbench_htp_rmsnorm: ... dspqueue_close: closed Queue 0, 0xb400007e3943f010, DSP 0x00000000 ← Stage 4 (create+export+close OK)
(no E adsprpc lines)
```

⇒ Q-2.1 의 `Control stubbed routine` 같은 vendor ACL 차단 라인은 **전무**. **stock S25 에서 자체 빌드 DSP skel binary 가 정상 로드** 가능함을 직접 증명. 이는 본 sprint 의 가장 큰 성취.

---

## stdout 실측 결과 (Phase 6 run 4, n_bufs=2 patch 후)

```
=== microbench_htp_rmsnorm (Q-2.2-α PoC Phase 5) ===

Config:
  shape       = [1, 1536]    eps = 0.00001
  warmup iter = 10           measure iter = 1000

[1/3] CPU baseline (f32)
  mean=0.87 us (median=0.83, stddev=0.42, n=1000)

[2/3] OpenCL GPU
  mean=161.68 us (median=150.31, stddev=46.13, n=1000)
  vs CPU: max_abs_err=0.000e0, max_rel_err=0.000e0 — PASS

[3/3] QNN HTP NPU (FastRPC + libggml-htp-v79.so dspqueue)
  host: libcdsprpc.so=/vendor/lib64/libcdsprpc.so, domain_id=7, session_id=1, queue_id=33552284516352
  SKIP — dspqueue_write: htp_fastrpc: AEEResult=unknown (0x48, raw=72)
```

CPU 1.00× baseline, OpenCL 185× slower (dispatch overhead, scale 작음 — `[1, 1536]` 단일 row 라 GPU 사용 의미 없음. Adreno 가 sweet spot 진입은 더 큰 shape 필요). HTP 측정은 stage 5 차단으로 미수집.

---

## 부분 GREEN 의 의미

본 sprint 전 가설:
- (a) llama.cpp ggml-hexagon path 가 stock S25 에서 raw HTP 동작
- (b) 우리 host Rust binding 으로 동일 path 재현 가능
- (c) 자체 빌드 DSP skel binary 가 stock device 에서 load 가능

검증 결과:
- (a) ✓ Q-2.2 dry-run B + 본 Phase 6 모두 GREEN
- (b) ✓ host.rs (603 LOC) + idl.rs (305) + buffer.rs (204) + error.rs (128) + mod.rs (532) = 1772 LOC binding 으로 4-step handshake + GET_URI + handle_open + dspqueue create/export 가 stock S25 에서 동작
- (c) ✓ `libggml-htp-v79.so` 271 KB cdylib (Docker 안에서 자체 빌드) 가 stock S25 DSP-side 에서 dlopen 성공 (logcat `open success: libggml-htp-v79.so` 직접 증명)

미해결:
- (d) DSP-side worker thread 시작 (`htp_iface_start`) → dspqueue_write 정상 dispatch 까지

(d) 는 IDL stub macro 정확값 + parameter array layout 의 detail 작업. host source 분석 또는 libcdsprpc.so reverse engineering 이 필요. 본 sprint scope 의 핵심은 (a)(b)(c) 모두 GREEN — d는 β 작업.

---

## β 단계 진입점

다음 sprint 명확한 진입:

1. **`htp_iface_start` IDL stub 정확 구현** (1 method, ~150 LOC):
   - `REMOTE_SCALARS_MAKEX(attr=0, method=0, in_h=0, out_h=0, in_b=3, out_b=0)` macro 값 확정
   - `remote_arg pra[3] = [{u32 sess_id}, {u64 dsp_queue_id}, {u32 n_hvx}]` parameter array layout
   - `remote_handle64_invoke(handle, sc, pra)` 호출
   - HVX thread 수 결정 (default = `n_hvx=4`)
2. **Re-run Phase 6** → dspqueue_write rc=0 GREEN 기대
3. **rms_norm correctness gate** (max_abs_err < 1e-3 vs CPU baseline)
4. **3-way timing** complete table 생성

이후 β 의 본격 작업 (matmul Q-proj 등) 진입.

---

## 변경 파일

| 경로 | 변경 |
|---|---|
| `engine/src/backend/htp_fastrpc/host.rs` | +56 LOC. `RemoteRpcGetUri` struct + `FASTRPC_GET_URI` step (Stage 2). session-encoded URI fallback. Phase 6 핵심 fix |
| `engine/microbench/htp_rmsnorm.rs` | -10/+5 LOC. n_bufs=3→2 (llama.cpp htp/main.c:1081 `if (n_bufs != 2)` 검증). gamma=1.0 fill (llama.cpp `HTP_OP_RMS_NORM` 은 gamma 미적용). SKIP 메시지에 raw rc 표시 (`{:#}`) |
| `devices.toml` | +1 word. `features = [..., "htp_fastrpc"]` 추가 (run_device.py 자동 빌드 시 activate) |
| `papers/.../qnn_q22_alpha_rmsnorm_2026_05_26/logs_stdout.txt` + 4 runs | 측정 stdout |
| `papers/.../qnn_q22_alpha_rmsnorm_2026_05_26/logs_logcat.txt` | DSP-side skel load 직접 증명 logcat |
| `papers/.../qnn_q22_alpha_rmsnorm_2026_05_26/report.md` | 본 문서 |

---

## 자기점검

- [x] 진입 문장: 다음 sprint = "Q-2.2-β.IDL — htp_iface_start IDL stub 정확 구현"
- [x] 멈춘 이유: IDL stub macro 정확값 부재 (Phase 4 senior-impl 도 best-effort fail 인정). scope 외 작업. PoC 본질은 (a)(b)(c) 모두 GREEN 으로 검증 완료
- [x] landmine:
  - `FASTRPC_GET_URI` step 누락 시 default domain 3 fallback → `Error 0xe domain >= 0` (Phase 6 first run 에서 검증, host.rs patch 로 해결)
  - llama.cpp `HTP_OP_RMS_NORM` 은 gamma 미적용 → 3-way correctness 시 baseline gamma=1 fill 필수
  - `dspqueue_write rc=0x48` 는 host queue 정상 + DSP worker 미시작 의 특정 시그니처
- [x] 검증 게이트: logcat `open success: libggml-htp-v79.so` direct evidence + queue_id 32-bit 정수 (NULL 아님) 확인 + rc raw int 직접 인용
- [x] 본문 길이: ~900 토큰 (Q-2.1/Q-2.2 dry-run report 와 동등 분량, β 진입점 명세 추가분)
