# Handoff: Q-2.2 옵션 D — host-side 3 가설 모두 ✗ 반증. senior-implementer 위임.

**작성**: 2026-05-26
**worktree**: `.claude/worktrees/b5_trait_extension`
**진입 commit**: `47bb5380` (옵션 C handoff 직후)
**다음 세션 진입 문장**: **"Q-2.2 옵션 D continue — senior-implementer 위임. 가설 4~8 (DspQueueBuffer layout / op_code / 추가 init API / fd mismatch / SElinux) 정밀 분석 + binary diff 로 `dspqueue_write rc=14` root cause 식별"**

---

## TL;DR

옵션 D 의 host-side 추측 fix 3개 시도, 모두 ✗ 반증:

| 가설 | 변경 | 결과 |
|---|---|---|
| 1 `remote_register_buf_attr2` 호출 차이 (β-1.QUEUE.2 잔존) | env-gate, default OFF | ✗ dspqueue_write rc=14 동일 |
| 2 dspqueue_create callback context NULL vs llama.cpp `(void*)this` | dummy non-null env-gate | ✗ rc=14 동일 |
| 3 RESERVE_NEW_SESSION → `domain_id=7 session=1` vs llama.cpp `domain=3 session=0` | env-gate dev0_path (RESERVE skip) | ✗ rc=14 동일 |

→ 단순 host-side 추측 fix scope 소진. **root cause = 더 깊은 binary-level 차이** (DspQueueBuffer layout / op_code / 추가 init API / fd mismatch / SElinux 등).

본 sprint 코드 변경 = env-gated ablation infrastructure 만 (default 동작 변동 없음). 다음 세션 senior-implementer 위임.

---

## frame 정정 (중요)

옵션 D 의 원 가설은 "0x80000414 fatal handling 제거" 였으나 본 sprint 첫 측정 (D.1) 에서:

```
SKIP — dspqueue_write: htp_fastrpc: AEEResult=AEE_EUNABLETOLOAD (0xe, raw=14)
```

→ 우리 abort point = `dspqueue_write rc=14`, NOT `0x80000414`. logcat 의 `0x80000414` 는 driver-internal probe (llama.cpp 도 같은 line 발생) 로 fatal 아님.

옵션 C 의 handoff (handoff_q22_option_c_2026_05_26 의 옵션 D 정의) 가 잘못 frame 되어 있었음. 본 handoff 가 correct frame.

---

## 진행 상태

| Task | 상태 | 결과 |
|---|---|---|
| D.1 fatal catch site 식별 | ✓ | `dspqueue_write rc=14` (frame 정정) |
| D.2 host-side 가설 3개 ablation | ✓ | 모두 ✗ |
| D.3 호스트/Android 빌드 | ✓ | check + fmt + clippy clean |
| D.4 S25 microbench 3 attempt | ✓ | 동일 fail rc=14 |
| D.5 correctness gate | ✗ | dispatch GREEN 미달성 |
| D.6 report + handoff + commit | ✓ | 본 sprint |

---

## 다음 작업: senior-implementer 위임 (Q-2.2 옵션 D continue)

### 위임 prompt 초안

```
프로젝트: llm.rs (worktree b5_trait_extension)
trait/scope: HTP FastRPC backend (engine/src/backend/htp_fastrpc/)
trigger: dspqueue_write rc=0xe (AEE_EUNABLETOLOAD, raw=14) 의 root cause 식별 + fix.

상황:
- 우리 host.rs lifecycle (RESERVE/GET_URI/UNSIGNED/handle_open/LATENCY/dspqueue_create/export/htp_iface_start) 모두 GREEN.
- 우리 dispatch (dspqueue_write) 만 fail (rc=14).
- llama.cpp `build-snapdragon/bin/llama-bench` 는 stock S25 에서 같은 path 로 NPU GREEN (옵션 C 검증, tg32 32.40).
- 우리 sprint 의 host-side 3 가설 (register_buf / non-null ctx / domain_id 매칭) 모두 ✗ 반증 (qnn_q22_option_d_2026_05_26/report.md).

목표: dspqueue_write rc=0 + correctness `max_abs_err < 1e-3` vs CPU baseline → backend GREEN.

가설 잔여 (가능성 순):
1. DspQueueBuffer struct layout (Rust #[repr(C)] padding vs QC `inc/dspqueue.h`). sizeof / 각 field offset binary 검증.
2. HtpGeneralReq op_code/op_params encoding (우리 RMS_NORM enum 값 vs llama.cpp htp/main.c expect).
3. host-side init 의 추가 누락 API (예: fastrpc_set_param, fastrpc tunable). strace + libcdsprpc symbol diff.
4. rpcmem_to_fd 결과의 fd 가 fastrpc domain register fd 와 mismatch.
5. SElinux/seccomp 정책 차이.

준비물:
- 우리 binary: /data/local/tmp/microbench_htp_rmsnorm (1576544 byte)
- llama.cpp baseline: /data/local/tmp/llamacpp_c/llama-bench (옵션 C GREEN)
- 두 binary 모두 같은 libggml-htp-v79.so 사용
- env: HTP_FASTRPC_{REGISTER_BUF,NONNULL_CTX,DEV0_PATH}=1 로 본 sprint ablation 재현 가능

권장 접근: strace 또는 fastrpc trace 활성화 (`FASTRPC_LOG_FILES=1` env 또는 .debugconfig file) 로 두 binary 의 driver call sequence binary diff. 첫 발산 지점이 root cause.

검증 게이트: dspqueue_write rc=0 + rsp.status == HTP_STATUS_OK + max_abs_err < 1e-3.

fail 시 (가설 1-5 모두 ✗): handoff 에 "host scope 외, DSP-side/driver-level barrier 확정" + 옵션 B' transition.

본 sprint 코드 위치:
- host: engine/src/backend/htp_fastrpc/host.rs (470-665, 3 ablation site)
- microbench: engine/microbench/htp_rmsnorm.rs:518-583
- llama.cpp 비교: /home/go/Workspace/llama.cpp/ggml/src/ggml-hexagon/ggml-hexagon.cpp:1535-1680 + 141-161 (enqueue)
- 본 sprint report: papers/eurosys2027/_workspace/experiment/qnn_q22_option_d_2026_05_26/report.md
```

### 검증 게이트

- single ablation 으로 `dspqueue_write rc=0` + correctness PASS → backend GREEN.
- 5 가설 모두 ✗ → handoff "host scope 외" 확정 + 옵션 B' transition.

---

## 본 sprint 코드 변경

env-gated ablation infrastructure 만 (default 동작 변동 없음):

| 파일 | 라인 | 변경 |
|---|---|---|
| `engine/src/backend/htp_fastrpc/host.rs` | 470-487 | `remote_register_buf_attr2` dlsym → `HTP_FASTRPC_REGISTER_BUF=1` 일 때만 lookup. default = None (β-1.QUEUE.2 effective revert). |
| `engine/src/backend/htp_fastrpc/host.rs` | 512-541 | RESERVE_NEW_SESSION → `HTP_FASTRPC_DEV0_PATH=1` 시 skip + domain_id=3 / session_id=0 hardcode. default = RESERVE. |
| `engine/src/backend/htp_fastrpc/host.rs` | 633-665 | dspqueue_create callback ctx → `HTP_FASTRPC_NONNULL_CTX=1` 시 non-null dummy. default = NULL. |

본 변경 자체로는 dispatch 동작 변경 없음. 다음 세션 senior-implementer 가 같은 env 로 ablation 재현 가능.

---

## Landmines / 미해결

- **frame 정정 반영 필요**: handoff_q22_option_c_2026_05_26 의 옵션 D 정의 "0x80000414 fatal handling 제거" 는 잘못. 본 handoff 가 correct frame. 다음 세션은 본 handoff 만 reference.
- **NPU performance 변동 없음**: 옵션 D GREEN 후에도 NPU < CPU 51% (Qwen2.5-1.5B Q4_0 한정, llama.cpp 측정). 백엔드 확장 가치는 있으나 production-track 가치는 별 평가.
- **session leak 잔존**: 본 sprint 측정 시도 사이 wait 없이 진행. fail state 라 더 누적. 다음 측정 sleep 5s 권장.
- **fast-fail risk**: 가설 4-8 도 단순 추측 fix 로 해결 안 될 가능성. strace/binary diff 가 진짜 첫 step. 단순 코드 추측 → 가설 ✗ 반복 안 하기.

---

## 핵심 파일 인덱스

- 본 sprint commit: (commit 직후 갱신)
- host: `engine/src/backend/htp_fastrpc/host.rs` (3 ablation site)
- microbench: `engine/microbench/htp_rmsnorm.rs:518-583`
- llama.cpp 비교: `/home/go/Workspace/llama.cpp/ggml/src/ggml-hexagon/ggml-hexagon.cpp:1535-1680, 141-161, 2199-2262`
- 측정 raw: `papers/eurosys2027/_workspace/experiment/qnn_q22_option_d_2026_05_26/`
- 본 sprint report: 같은 폴더 `report.md`
- 직전 handoff: `.agent/todos/handoff_q22_option_c_2026_05_26.md`
