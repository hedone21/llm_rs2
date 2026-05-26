# Q-2.2 dry-run (B scope) — FastRPC IDL path GREEN

**작성**: 2026-05-26
**HEAD (직전)**: `446487fa docs(handoff): Q-2.2 진입 plan — llama.cpp FastRPC IDL backend`
**디바이스**: Galaxy S25 (R3CY408S5SB / SM8750 / HTP V79), stock non-rooted
**대상 bin**: `microbench_htp_fastrpc_dryrun` (신설, +166 LOC)
**결과**: **GREEN — 가설 (5) FastRPC ACL 정밀화 완전 확정. Q-2.2-α Architect spec 진입 가능**

---

## TL;DR

Q-2.1 의 `E adsprpc remote.c:40 Control stubbed routine` 는 **QNN backend 가 호출한 vendor-extension control 에 한정된 차단**이고, **FastRPC layer 자체는 stock S25 에서 정상 동작**함을 단일 microbench 로 직접 증명. 핵심 시퀀스 4 단계 모두 PASS:

1. `dlopen libcdsprpc.so` OK (`/vendor/lib64/libcdsprpc.so`, 497624 bytes)
2. `dlsym remote_session_control` OK
3. `FASTRPC_RESERVE_NEW_SESSION` → return 0, **effective_domain_id=7**, session_id=1
4. `DSPRPC_CONTROL_UNSIGNED_MODULE` (domain=7, enable=1) → return 0

총 실행 시간 ~60 ms. logcat 에 `E adsprpc` 전무, 대신 `I microbench_htp_fastrpc_dryrun: fastrpc_apps_user.c:2890: remote_session_control Unsigned PD enable 1 request for domain 7` 가 깔끔히 기록.

⇒ Q-2.2 sprint 의 핵심 전제 (llama.cpp ggml-hexagon path 차용 가능) 가 본 디바이스에서 직접 검증됨. Architect 단계 진입 GREEN.

---

## 단계별 진행

| 단계 | 결과 | 비고 |
|---|---|---|
| Prototype 작성 — `microbench/htp_fastrpc_dryrun.rs` (+166 LOC) | 적용 | SDK header 의존 0, struct/매크로 직접 정의 (qualcomm/fastrpc remote.h 인용) |
| Cargo.toml bin 등록 | 적용 | `microbench_htp_fastrpc_dryrun` (qnn feature gate, 패밀리 통일) |
| host `cargo build --release --features qnn` | PASS | 20.95s, error 0 |
| Android cross-build `aarch64-linux-android` | PASS | binary 414 KB |
| Runtime — `dlopen /vendor/lib64/libcdsprpc.so` | OK | first candidate hit |
| Runtime — `dlsym remote_session_control` | OK | C ABI, no mangling |
| Runtime — `FASTRPC_RESERVE_NEW_SESSION` (req_id=13, struct=40B) | **OK** | **effective_domain_id=7, session_id=1** |
| Runtime — `DSPRPC_CONTROL_UNSIGNED_MODULE` (req_id=2, struct=8B) | **OK** | **return 0, unsigned PD 활성화 성공** |
| 총 wall time | ~60 ms | 12:18:02.482 → 12:18:02.543 |

---

## 핵심 logcat (Q-2.1 대비 결정적 대조)

`papers/.../qnn_q22_dryrun_fastrpc_2026_05_26/logs_logcat.txt` 핵심 라인:

```
I microbench_htp_fastrpc_dryrun: rpcmem_android.c:210: set up allocator ... for DMA buf heap system
I microbench_htp_fastrpc_dryrun: fastrpc_config.c:336: Reading configuration file
I microbench_htp_fastrpc_dryrun: fastrpc_apps_user.c:4957: fastrpc_apps_user_init done  with default domain:3
I microbench_htp_fastrpc_dryrun: fastrpc_apps_user.c:5071: multidsplib_env_init: libcdsprpc.so loaded
I microbench_htp_fastrpc_dryrun: fastrpc_apps_user.c:2890: remote_session_control Unsigned PD enable 1 request for domain 7   ← ★ DIRECT EVIDENCE
I microbench_htp_fastrpc_dryrun: fastrpc_apps_user.c:4532: close_dev: unloading library libcdsprpc.so
I microbench_htp_fastrpc_dryrun: fastrpc_apps_user.c:3504: domain_deinit for domain 0..15: dev -1
I microbench_htp_fastrpc_dryrun: rpcmem_android.c:357: rpcmem_deinit_internal done
I microbench_htp_fastrpc_dryrun: fastrpc_apps_user.c:4858: fastrpc_apps_user_deinit done
```

**Q-2.1 대비 차이점**:
- Q-2.1: `E adsprpc : remote.c:40:Control stubbed routine - Return failure` (QNN 의 `domain_init` 가 호출한 vendor-extension control 이 ACL fail)
- Q-2.2: **`E adsprpc` 라인 0건**. 대신 `I` (Info) 레벨로 `Unsigned PD enable 1 request for domain 7` 가 정상 logging
- ⇒ FastRPC layer 자체는 ACL 검사를 통과. QNN backend 가 호출한 일부 control code 만 vendor-side 에서 차단된 것이 확정

---

## 가설 ranking 최종 갱신

| 가설 | Q-2.1 후 | Q-2.2 후 | 근거 |
|---|---|---|---|
| (1) `.farf` sentinel | low | low | — |
| (2) RTLD_GLOBAL 미사용 | 반증 | 반증 | (Q-2.1) |
| (3) PlatformInfo block 누락 | 반증 | 반증 | (Q-2.1) |
| (3b) backendApiVersion 3.7.0 | 부적용 | 부적용 | — |
| (4) skel 경로/이름 | low | low | deviceCreate 미진입 |
| (5) OS-level FastRPC ACL | ★ 확정 | **★ 정밀화 — QNN domain_init vendor control 만 차단. FastRPC 자체는 stock S25 GREEN** | Q-2.2 logcat 의 `Unsigned PD enable 1 request for domain 7` 정상 처리 |

가설 (5) 의 의미가 두 단계 거쳐 명확해짐:
- Q-2.1: FastRPC layer 어딘가에서 차단 (범위 미정)
- **Q-2.2: 차단 범위 = QNN backend `domain_init` 가 호출하는 vendor-extension control code 만. FastRPC `remote_session_control` 의 표준 req_id 13/2 는 ACL bypass 가능**

이 정밀화는 **Q-2.2 sprint 의 핵심 가정** (llama.cpp ggml-hexagon path 차용 가능) 의 **직접 검증**.

---

## struct/매크로 정확값 (Hexagon SDK 의존 없이 직접 정의)

출처: `github.com/qualcomm/fastrpc/inc/remote.h` (open source).

```rust
// session_control_req_id (enum, 0-indexed)
pub const FASTRPC_RESERVE_NEW_SESSION: u32 = 13;
pub const FASTRPC_GET_URI: u32 = 15;
pub const DSPRPC_CONTROL_UNSIGNED_MODULE: u32 = 2;

// handle_control_req_id
pub const DSPRPC_CONTROL_LATENCY: u32 = 1;

// domain.h
pub const CDSP_DOMAIN_NAME: &str = "cdsp";

#[repr(C)]
pub struct RemoteRpcReserveNewSession {
    pub domain_name: *mut u8,
    pub domain_name_len: u32,
    pub session_name: *mut u8,
    pub session_name_len: u32,
    pub effective_domain_id: u32, // [out]
    pub session_id: u32,          // [out]
}

#[repr(C)]
pub struct RemoteRpcControlUnsignedModule {
    pub domain: i32,
    pub enable: i32,
}
```

크기 검증: `RemoteRpcReserveNewSession=40B` (pointer 8B × 2 + u32 × 4 = 16+16 padded), `RemoteRpcControlUnsignedModule=8B` (i32 × 2). 실측 stdout 도 동일 (`struct size=40` / `struct size=8`).

⇒ Hexagon SDK 도 없이 minimum binding 으로 FastRPC reserve_session + unsigned PD enable 까지 동작 확정. Q-2.2-β host binding 의 dlopen + struct 정의 layer 는 본 prototype 으로 0→1 검증 완료.

---

## 사용자 결정 게이트

본 dry-run 결과로 Q-2.2-α Architect spec 진입 가능. 다음 결정 항목 (Q-2.2-α 진입 시 동시 확정 필요):

1. `spec/htp_fastrpc_backend.md` ID prefix — `INV-HTP-FRPC-*` 또는 `INV-FASTRPC-*` 중 무엇?
2. backend 이름 — `qnn_fastrpc` 유지 (handoff 표기) 또는 `htp_fastrpc` (QNN 의존 0 이므로 더 정확) 중 무엇?
3. handoff 의 6 trade-off (vendor strategy / IDL reuse / SDK dependency / weight repack timing / multi-session / DSP arch coverage) 모두 본 dry-run 영향 없음 — 그대로 진행

⇒ 진입 문장: **"Q-2.2-α 진입 — FastRPC IDL backend Architect spec"**.

---

## 변경 파일

| 경로 | 변경 |
|---|---|
| `engine/microbench/htp_fastrpc_dryrun.rs` (신규) | +166 LOC. fastrpc mod + 4-step dry-run main (`dlopen` → `dlsym` → `FASTRPC_RESERVE_NEW_SESSION` → `DSPRPC_CONTROL_UNSIGNED_MODULE`). SDK header 의존 0 |
| `engine/Cargo.toml` | +4 LOC. `[[bin]] microbench_htp_fastrpc_dryrun` 등록 |
| `papers/.../qnn_q22_dryrun_fastrpc_2026_05_26/logs_stdout.txt` | runtime stdout (18 lines) |
| `papers/.../qnn_q22_dryrun_fastrpc_2026_05_26/logs_logcat.txt` | runtime logcat (25 lines, `E adsprpc` 0건) |
| `papers/.../qnn_q22_dryrun_fastrpc_2026_05_26/report.md` | 본 문서 |

---

## 진단 가치 보존 정책

본 prototype 은 Q-2.2-β host binding sprint 의 **starting point**:
- 4-step sequence 가 이미 검증 완료 → β 는 binding 확장 (7 symbol → 17 symbol) + handle_open / dspqueue 만 추가
- struct 정의는 직접 vendor 한 minimum 셋. Q-2.2-α Architect 에서 17 symbol 풀 binding 결정 시 본 prototype 의 fastrpc mod 를 확장

⇒ commit + 보존.

---

## 자기점검

- [x] 진입 문장: 다음 sprint = "Q-2.2-α 진입 — FastRPC IDL backend Architect spec"
- [x] 멈춘 이유: dry-run scope (B) 완수. handle_open 은 자체 빌드 skel 필요 → α/β/γ 에서 진행
- [x] landmine: 가설 (5) 정밀화 (QNN domain_init vendor control 만 차단). FastRPC `remote_session_control` 표준 req_id 는 stock S25 ACL bypass.
- [x] 검증 게이트: logcat raw 인용 (`Unsigned PD enable 1 request for domain 7` 정상 + `E adsprpc` 0건)
- [x] 본문 길이: ~750 토큰 (Q-2.1 report 750 토큰과 동일. dry-run 패턴 일관성 유지)
