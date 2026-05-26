# Q-2.2 옵션 D — host-side fix 가설 3개 모두 ✗ 반증, root cause 가 host scope 안인지 밖인지 불확실. senior-implementer 위임.

**작성**: 2026-05-26
**worktree**: `.claude/worktrees/b5_trait_extension`
**진입 commit**: `47bb5380` (옵션 C handoff 직후)
**device**: Galaxy S25 (R3CY408S5SB)
**model**: Qwen2.5-1.5B 미사용 (microbench_htp_rmsnorm 단독)

## TL;DR

옵션 D 의 frame 정정 + ablation 3개 실측:

- **frame 정정**: 옵션 D 원 가설은 "0x80000414 fatal handling 제거" 였으나 실제 우리 fail = `dspqueue_write rc=0xe (AEE_EUNABLETOLOAD, raw=14)`. `0x80000414` 는 logcat 에 출력되는 driver-internal probe error (llama.cpp 도 동일 발생) 이지만 우리 abort point 가 아님.
- **3 host-side fix 가설 모두 ✗ 반증**:
  - 가설 1: `remote_register_buf_attr2` 호출 차이 (β-1.QUEUE.2 잔존) → env OFF default → 동일 fail.
  - 가설 2: dspqueue_create callback context NULL vs llama.cpp `(void*)this` → dummy non-null 시도 → 동일 fail.
  - 가설 3: RESERVE_NEW_SESSION 으로 우리는 `domain_id=7 session_id=1`, llama.cpp 는 RESERVE skip + `domain_id=3 session_id=0` → env-gated dev0_path 추가 → 동일 fail.
- **본 sprint 결과 = 단순 host-side 추측 fix scope 소진**. 다음 step = senior-implementer 위임 (binary diff / strace / DspQueueBuffer struct layout 정밀 비교 / Hexagon SDK driver 동작 분석).

본 sprint 코드 변경 = ablation infrastructure (env-gated) 만 유지. default 동작 변동 없음 — handoff_q22_beta_queue_rpc 와 같은 fail state.

## ablation 측정 matrix

device: R3CY408S5SB. command: `LD_LIBRARY_PATH=/data/local/tmp ./microbench_htp_rmsnorm --rows 1 --dim 4096 --eps 1e-5 --warmup 1 --measure 1`

| run | env | host log | result |
|---|---|---|---|
| baseline | (default, β-1.QUEUE.RPC 상태) | domain_id=7 session_id=1, htp_iface_start OK | dspqueue_write rc=14 |
| attempt1 | register_buf OFF | 동일 | rc=14 |
| attempt2 | `HTP_FASTRPC_NONNULL_CTX=1` | 동일 + non-null context | rc=14 |
| attempt3 | `HTP_FASTRPC_DEV0_PATH=1` | domain_id=3 session_id=0 | rc=14 |

→ **세 변수 (register_buf / callback ctx / domain_id) 모두 dspqueue_write 의 root cause 아님**.

## logcat trace (정확한 fail line)

```
.385 Error 0x80000414: remote_handle64_invoke failed for handle 0xb400007e8b038f10, 
                      interface libdspqueue_rpc_skel.so method 3 on domain 7 (sc 0x3010100)
                      (errno Success) (user err 0x80000414)
                      ↑ driver-internal probe (llama.cpp 도 발생, dspqueue_create 시점)

.386 Error 0xe: dspqueue_write_noblock failed for queue 0xb400007efafc0b50 
                (flags 0x3, num_buffers 2, message_length 312)
                ↑ ★ 실제 우리 abort point ★

.386 Error 0xe: dspqueue_write failed for queue 0xb400007efafc0b50 
                (flags 0x0, num_buffers 2, message_length 312 errno Success)
```

핵심:
- `message_length 312` = `HtpGeneralReq` size — llama.cpp local build `htp-msg.h` 와 byte-identical 확인 (β-1.QUEUE.RPC).
- `flags 0x3` = driver auto-set (MESSAGE | BUFFERS), `0x0` = 우리가 보낸 outer flags. 정상.
- `num_buffers 2` = bufs[input, output]. ggml-hexagon `init_unary_req` 와 동일.

## llama.cpp 와의 비교 (옵션 C 결과 기반)

| 항목 | 우리 | llama.cpp (stock S25) |
|---|---|---|
| 0x80000414 method 3 발생 | ✓ | ✓ |
| 0x80000414 의 fatal 처리 | ✗ (driver-internal, dspqueue_create 시점) | ✗ (동일) |
| dspqueue_create rc | success | success |
| dspqueue_write rc | **0xe (14) fail** | **0 (success)** |
| NPU compute (tg32) | SKIP | **32.40 tok/s** |

→ 우리 binding 의 dspqueue_write 호출이 driver 의 method 3 dispatch table lookup 시점에 fail. llama.cpp 는 같은 시점에 성공.

## 후보 가설 (잔여, senior-implementer 위임)

| # | 가설 | 추정 |
|---|---|---|
| 4 | `DspQueueBuffer` struct sizeof / layout 미세 차이. Rust `#[repr(C)]` padding vs QC `inc/dspqueue.h` 의 packed/aligned. | 1-2h (binary diff) |
| 5 | `HtpGeneralReq` 의 `op_code` (RMS_NORM enum 값) 또는 `op_params[0]=eps` encoding 차이. llama.cpp `htp/main.c::rms_norm_f32` 의 expect 값 비교. | 1h |
| 6 | host-side init 시 추가 누락된 driver API 호출. llama.cpp 의 binary 자체에 우리에게 없는 라이브러리 init step 가 있을 가능성 (e.g., `fastrpc_set_param` 류). | 2-3h (strace + binary symbol diff) |
| 7 | dspqueue_buffer 의 fd 가 driver internal mapping 과 mismatch — `rpcmem_to_fd` 결과가 fastrpc domain 등록 fd 와 다를 가능성. `fastrpc_mmap` 시점의 fd 값 추적. | 1-2h |
| 8 | 우리 process 의 SElinux/seccomp 정책이 dspqueue_write 의 ioctl 차단. llama.cpp 와 binary 권한 비교. | 30m-1h |

## senior-implementer 위임 brief

- **목표**: `dspqueue_write rc=14` 의 root cause 식별 + fix → `microbench_htp_rmsnorm` HTP path GREEN (`max_abs_err < 1e-3`).
- **접근**: 본 sprint 의 3 가설 (register_buf / non-null ctx / dev0_path) 모두 ✗ 확정. 가설 4-8 sequential 시도 또는 strace/symbol binary diff 로 시야 넓게.
- **준비물**:
  - 우리 binary: `/data/local/tmp/microbench_htp_rmsnorm` (1576544 byte, 본 sprint 빌드)
  - llama.cpp binary: `/data/local/tmp/llamacpp_c/llama-bench` (옵션 C 검증 GREEN)
  - 둘 다 같은 `libggml-htp-v79.so` (271KB) 사용
  - env gating: `HTP_FASTRPC_REGISTER_BUF=1` / `HTP_FASTRPC_NONNULL_CTX=1` / `HTP_FASTRPC_DEV0_PATH=1` 로 ablation 재시도 가능 (default 는 기존 동작).
- **검증 게이트**: 한 단일 ablation 으로 dspqueue_write rc=0 → 그 변경을 default 화 + correctness PASS → backend GREEN.
- **fail 시**: handoff 에 "host scope 외 (DSP-side / driver-level)" 확정 + 옵션 B' (negative result paper 기록) 으로 transition.

## 본 sprint 코드 변경

env-gated ablation infrastructure 만 추가 (default 동작 변동 없음):

| 파일 | 변경 |
|---|---|
| `engine/src/backend/htp_fastrpc/host.rs:470-487` | `remote_register_buf_attr2` dlsym 을 `HTP_FASTRPC_REGISTER_BUF=1` 일 때만 lookup. default = None (β-1.QUEUE.2 revert). |
| `engine/src/backend/htp_fastrpc/host.rs:512-541` | RESERVE_NEW_SESSION 을 `HTP_FASTRPC_DEV0_PATH=1` 시 skip + domain_id=3, session_id=0 hardcode. default = RESERVE 호출. |
| `engine/src/backend/htp_fastrpc/host.rs:633-665` | dspqueue_create callback context = `HTP_FASTRPC_NONNULL_CTX=1` 시 non-null dummy. default = NULL. |

본 sprint commit 만으로는 dispatch 동작 변경 없음. ablation infrastructure 만 추가.

## 핵심 파일 인덱스

- 본 sprint commit: (commit 직후 갱신)
- host: `engine/src/backend/htp_fastrpc/host.rs` (3 ablation site)
- microbench: `engine/microbench/htp_rmsnorm.rs:518-583` (dispatch closure)
- llama.cpp 비교 baseline: `/home/go/Workspace/llama.cpp/ggml/src/ggml-hexagon/ggml-hexagon.cpp:1535-1680`
- llama.cpp dispatch: `ggml-hexagon.cpp:141-161` (enqueue), `2199-2262` (htp_req_buff_init)
- 측정 raw: `papers/eurosys2027/_workspace/experiment/qnn_q22_option_d_2026_05_26/{run,logcat}_d4_attempt[1-3].txt`
- 직전 handoff: `.agent/todos/handoff_q22_option_c_2026_05_26.md`

## Landmines / 미해결

- **0x80000414 frame 정정**: 본 sprint 초기 가설 ("fatal handling 제거") 가 잘못. logcat 의 출력 line ≠ 우리 binding 의 abort point. 실제 fail = `dspqueue_write rc=14`. 옵션 D 의 작업 범위 정의가 처음부터 mis-frame 되었음 (handoff_q22_option_c_2026_05_26 의 옵션 D 정의 갱신 필요).
- **3 가설 ✗ 의 의미**: register_buf / callback ctx / domain_id 모두 root cause 아님 = host-side 의 명백한 차이 가능성 모두 검증. 남은 가설 4-8 은 더 정밀한 binary diff 필요. simple-fix 가능성 낮음.
- **NPU performance < CPU 는 변동 없음**: 옵션 D GREEN 달성 후에도 우리 binding 의 NPU performance 는 llama.cpp 32 tg32 수준 = CPU 51%. paper main result 부적합 → 백엔드 확장 가치는 있으나 production-track 가치는 별 평가.
- **session leak 잔존**: 본 sprint 측정 시도 사이 별 wait 없이 진행. fail state 라 leak 가 더 누적 가능. 다음 측정 시 sleep 5s 권장.

## 결정 게이트

본 sprint 완료. 다음 세션 = senior-implementer 위임. 진입 문장: `Q-2.2 옵션 D continue — senior-implementer 위임. 가설 4~8 (DspQueueBuffer layout / op_code / 추가 init API / fd mismatch / SElinux) 정밀 분석 + binary diff 로 dspqueue_write rc=14 root cause 식별`.
