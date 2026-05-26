# Handoff: Q-2.2 옵션 D continue — DspQueueBuffer layout fix, NPU compute GREEN

**작성**: 2026-05-26
**worktree**: `.claude/worktrees/b5_trait_extension`
**commit**: `dde08248`
**다음 세션 진입 문장 후보**: "Q-2.2 옵션 D complete — backend correctness GREEN. NPU production-track 가치 평가 (shape sweep + 진짜 hot path op 측정) 또는 옵션 B' transition 결정."

---

## TL;DR

senior-implementer 1 회 위임. `DspQueueBuffer` struct layout 정정 + fix:

- root cause = `DspQueueBuffer` 의 field 선언 순서 mismatch (이전 `fd/ptr/offset/size/flags/reserved[3]` 40B → 정확 `fd/size/offset/flags/ptr` 24B per QC SDK `inc/dspqueue.h` BSD-3-Clause).
- 직전 sprint 의 3 host-side 가설 ablation 모두 ✗ 였던 이유 = root cause 가 더 깊은 ABI layout corruption. 3 env 별 logcat assertion 이 모두 같은 corruption 의 다른 manifestation.
- 검증 GREEN: `dspqueue_write rc=0` + `rsp.status==HTP_STATUS_OK` + `max_abs_err=0.0` (bit-identical vs CPU) + multi-shape stability + logcat clean.

본 sprint 의 자세한 발견 경로는 commit message + 본 문서의 "발견 경로" 섹션 참조.

---

## 진행 상태

| Task | 상태 | 결과 |
|---|---|---|
| 가설 1 binary layout 검증 | ✓ | QC SDK 헤더 fetch + layout diff |
| layout fix 적용 | ✓ | host.rs DspQueueBuffer + buffer.rs dsp_buf |
| host fmt/clippy/test | ✓ | 17 htp_fastrpc tests pass |
| device build + deploy | ✓ | aarch64-linux-android release OK |
| S25 microbench correctness | ✓ | max_abs_err=0.0 (bit-identical) |
| multi-shape stability | ✓ | 1536 / 4096 / 2048 모두 PASS |
| logcat clean 확인 | ✓ | `buffer_ref` / `b->offset` / `DEREF` assertion 사라짐 |
| commit + handoff | ✓ | 본 sprint |

---

## 측정 결과

device: Galaxy S25 (R3CY408S5SB), shape `[1, 4096]`, eps=1e-5, n=1000

```
[1/3] CPU baseline (f32)
  mean=0.86 us (median=0.83, stddev=0.05, n=1000)
[2/3] OpenCL GPU
  mean=151.37 us (median=136.25, stddev=46.57, n=1000) — PASS
[3/3] QNN HTP NPU (FastRPC + libggml-htp-v79.so dspqueue)
  mean=93.91 us (median=90.62, stddev=76.72, n=1000) — PASS
  err: max_abs=0.0, max_rel=0.0
```

logcat (post-fix): `dspqueue_write` 관련 error line 0 개 (이전 `0xe: fastrpc_buffer_ref failed` / `b->offset == 0` / `DSPQUEUE_BUFFER_FLAG_DEREF` 모두 clean).

NPU > CPU 51% 미해결 — 본 sprint 는 backend correctness GREEN 만 확정. CPU 0.86 µs 는 microbench overhead dominant — production 가치는 별 평가 (아래 미해결 참조).

---

## struct 변경 (핵심 1 hunk)

### 이전 (broken — 40 B)

```rust
#[repr(C)]
pub struct DspQueueBuffer {
    pub fd: c_int,              // @0  (ok)
    pub ptr: *mut c_void,       // @8  (alignment padding inserted)  WRONG ORDER
    pub offset: u32,            // @16
    pub size: u32,              // @20
    pub flags: u32,             // @24
    pub reserved: [u32; 3],     // @28 (존재하지 않는 field)
}
```

### 새 (GREEN — 24 B per QC SDK)

```rust
#[repr(C)]
pub struct DspQueueBuffer {
    pub fd: u32,                // @0
    pub size: u32,              // @4
    pub offset: u32,            // @8
    pub flags: u32,             // @12
    pub ptr: *mut c_void,       // @16 (union with uint64_t address)
}
```

`offset_of!` 기반 `dspq_buffer_layout_matches_qc_sdk` test 추가 — 재발 방지.

---

## fail mode 통일 설명 (직전 sprint 의 3 env 가설)

| env | 이전 logcat assertion | 진짜 의미 (layout 관점) |
|---|---|---|
| default (`domain=7`) | `dspqueue_cpu.c:1196 nErr==0 / fastrpc_buffer_ref ref=-1` | driver 가 `(size, offset)` 잘못 읽어 buffer registry lookup fail |
| `HTP_FASTRPC_REGISTER_BUF=1` | `dspqueue_cpu.c:1185 (b->flags & DEREF) == 0` | flags 슬롯이 우리 `offset` (=0) 으로 read 됨. register_buf 가 ref 설정 후 driver 가 자동 deref 시도해 mismatch |
| `HTP_FASTRPC_DEV0_PATH=1` (`domain=3`) | `dspqueue_cpu.c:1208 b->offset == 0` | offset 슬롯이 우리 `ptr` 상위 32bit 으로 read 됨, non-zero |

세 환경 모두 같은 layout corruption 의 다른 측면을 보고. 직전 sprint 의 ablation 매트릭스에서 root cause 단정이 어려웠던 이유 = manifestation 이 enviroment 별로 다르게 보였음.

---

## 발견 경로 (회고)

1. **on-device 환경 sanity**: S25 non-rooted + SELinux enforcing + strace 미설치 → `FASTRPC_LOG_FILES` / `.debugconfig` env 도 RFS 디렉토리 권한 차단. logcat (`vendor/qcom/proprietary/adsprpc/...`) 의 verbose 가 유일한 trace 채널.
2. **3 assertion 의 공통점**: 직전 sprint 의 3 env 별 logcat assertion 모두 `dspqueue_buffer` field access. "different sites of same struct corruption" 가설 도출.
3. **SDK 헤더 회수**: hexagon SDK 가 호스트에 없음. QC fastrpc github 의 `inc/dspqueue.h` 를 `wget https://raw.githubusercontent.com/quic/fastrpc/master/inc/dspqueue.h` 로 직접 fetch (BSD-3-Clause).
4. **layout 매칭 + fix**: SDK 의 `fd/size/offset/flags/ptr` (24B) ↔ 우리 `fd/ptr/offset/size/flags/reserved` (40B) 차이. 이전 sprint 가 ggml-hexagon 의 **field 사용 순서** (`d->fd, d->ptr, d->offset, d->size, d->flags`) 를 struct **선언 순서** 로 오인.
5. **부수 발견**: `microbench_htp_rmsnorm` 의 `rsp.status != 0` check 가 `HTP_STATUS_OK = 1` 인 환경에서 always-pass false positive (이전엔 모든 run 이 dispatch fail 이라 reach 안 함). fix 후 status check 가 정상 동작.

직전 sprint handoff 의 "권장 접근" (strace/binary diff/SDK 헤더 정밀 비교) 의 마지막 항목 정확히 적용.

---

## 코드 변경 (commit `dde08248`)

| 파일 | 변경 |
|---|---|
| `engine/src/backend/htp_fastrpc/host.rs` | `DspQueueBuffer` field 순서/sizeof 수정 (24B). `pub fd: u32` (이전 `c_int`). docstring 에 root cause 설명. `dspq_buffer_layout_matches_qc_sdk` test 추가. |
| `engine/src/backend/htp_fastrpc/buffer.rs` | `dsp_buf` literal 의 field 순서 갱신. `self.fd as u32` cast. `reserved` 제거. |
| `engine/microbench/htp_rmsnorm.rs` | `rsp.status != 0` → `!= HTP_STATUS_OK`. import 에 `HTP_STATUS_OK` 추가. |

env-gated ablation infrastructure (`HTP_FASTRPC_REGISTER_BUF` / `_NONNULL_CTX` / `_DEV0_PATH`) 는 그대로 유지 — 회귀 측정용.

---

## Landmines / 미해결

- **NPU 성능 평가는 별 sprint**: 본 sprint 는 backend correctness GREEN 만. dim=4096 단일 shape 에서 NPU > CPU 51% 는 microbench overhead dominant — production 평가는:
  - (a) larger shape sweep (rmsnorm 만은 의미 제한적, FFN/attn 으로 op 확장 필요)
  - (b) MatMul Q-proj 등 진짜 hot path op 측정
  - (c) batch 효과 (rows ≫ 1)
- **production backend wire-up**: 본 sprint 는 microbench 의 dispatch closure 만 검증. `engine/src/backend/htp_fastrpc/mod.rs::dispatch_rmsnorm` 도 같은 fix 가 적용 가능한지 점검 필요 (struct 변경은 transparent 이지만 status check 패턴 확인).
- **NPU vs CPU 51% 결과**: paper main result 부적합은 변동 없음. 옵션 B' (negative result paper 기록) transition vs 옵션 D 후속 (real hot path op 측정) 결정 필요. **결정 권한은 사용자 sprint 정책**.
- **`reserved[3]` 제거 side effect**: 본 backend 는 internal only 라 외부 caller 영향 없음. ABI 외부 노출이 생기면 sizeof 변경 brake 가능.

---

## 검증 게이트 (사전 정의 vs 결과)

| gate | result |
|---|---|
| `dspqueue_write rc=0` | ✓ |
| `rsp.status == HTP_STATUS_OK` | ✓ |
| `max_abs_err < 1e-3` vs CPU | ✓ (= 0.0, bit-identical) |
| host fmt + clippy + lib test | ✓ (17 htp_fastrpc tests pass, layout test 포함) |
| logcat clean (3 assertion 사라짐) | ✓ |
| multi-shape stability | ✓ (1536 / 4096 / 2048) |

→ **GATE PASS**. 옵션 D continue sprint 종결.

---

## 핵심 파일 인덱스

- 본 sprint commit: `dde08248`
- host: `engine/src/backend/htp_fastrpc/host.rs:143-180` (DspQueueBuffer 새 정의)
- host test: 같은 파일 `tests::dspq_buffer_layout_matches_qc_sdk`
- buffer: `engine/src/backend/htp_fastrpc/buffer.rs:232-244` (dsp_buf literal)
- microbench: `engine/microbench/htp_rmsnorm.rs:573-577` (status check fix)
- SDK 헤더 참조 (호스트 캐시): `/tmp/fastrpc_check/dspqueue.h` (BSD-3-Clause, github 직접 fetch)
- 직전 sprint report: `papers/eurosys2027/_workspace/experiment/qnn_q22_option_d_2026_05_26/report.md`
- 직전 sprint handoff: `.agent/todos/handoff_q22_option_d_2026_05_26.md`
