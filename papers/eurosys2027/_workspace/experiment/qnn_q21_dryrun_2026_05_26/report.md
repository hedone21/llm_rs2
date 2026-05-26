# Q-2.1 dry-run — PlatformInfo + RTLD_GLOBAL 패치 RED

**작성**: 2026-05-26
**HEAD (직전)**: `204c7fbd docs(handoff): Executorch + llama.cpp QNN backend 분석 종결 → Q-2 진입 plan`
**디바이스**: Galaxy S25 (SM8750 / Adreno 830 / HTP V79), stock non-rooted
**대상 bin**: `microbench_htp_matmul_correctness`
**결과**: **RED — but FastRPC ACL 차단 mechanism 직접 증명 (진단 가치 ★)**

---

## TL;DR

직전 Q-2.0 dry-run (HEAD `56f059c5`) 의 +99 LOC device config (arch=V79 + SIGNEDPD) 만으로는 `deviceCreate err=0x36b1` 차단. 본 Q-2.1 에서 Researcher 분석 기반 2-shot 패치 (PlatformInfo block + RTLD_GLOBAL) 적용 후에도 `deviceCreate err=0x36b1` 동일. fail point 만 이동 (Q-2.0 의 contextCreate fail → 본 dry-run 의 deviceCreate fail). **그러나 PlatformInfo 추가 덕분에 transport 시도 단계까지 진입하여 `adsprpc remote.c:40 Control stubbed routine` logcat 으로 OS-level FastRPC ACL 차단 직접 증명**. 가설 (5) (stock S25 ACL) 확정, 가설 (2)(3) 반증.

---

## 단계별 진행

| 단계 | 결과 | 비고 |
|---|---|---|
| Patch 1 — `QnnDevice_PlatformInfo_t` block + `QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO` entry (+74 LOC) | 적용 | values=Executorch HtpDevicePlatformInfoConfig.cpp:13-58 차용 (vtcm=8MB, signedPdSupport=false, socModel=57=SM8750, arch=V79, dlbcSupport=true) |
| Patch 2 — `dlopen_global` helper + 6 sibling lib RTLD_NOW\|RTLD_GLOBAL preload | 적용 | libloading 0.8 `libloading::os::unix::Library` 사용. Cargo.toml 변경 없음 |
| host `cargo build --release --features qnn` | PASS | — |
| Android cross-build `aarch64-linux-android` | PASS | — |
| Runtime — RTLD_GLOBAL preload 6 lib | 4 OK / 2 skip | `libQnnHtpV79.so` + `libQnnHtpV79Skel.so` = **32-bit** lib (64-bit host process 라 skip). DSP skel side lib 이므로 host RTLD_GLOBAL 무관 |
| Runtime — `dlopen libQnnHtp.so RTLD_GLOBAL` | OK | — |
| Runtime — `backendCreate` | OK | — |
| Runtime — **`deviceCreate (ARCH + SIGNEDPD + PLATFORM_INFO)`** | **FAIL** | **err=0x36b1 (= 14001)** |
| contextCreate / graphCreate | blocked | — |

---

## 핵심 logcat (Q-2.0 대비 신규 단서)

`papers/.../qnn_q21_dryrun_2026_05_26/logs_logcat.txt`:

```
W QnnDsp  : Initializing HtpProvider
W QnnDsp  : Specified config ARCH, ignoring on real target          ← Q-2.0 ARCH config 도 no-op
E adsprpc : remote.c:40:Control stubbed routine - Return failure    ← ★ FastRPC ACL 차단 직접 증명
E QnnDsp  : Fail to get effective domain id from rpc with DeviceId 0 coreId 0 pdId 0
E QnnDsp  : error in creation of transport instance
E QnnDsp  : Failed to create transport for device, error: 1002
E QnnDsp  : Failed to load skel, error: 1002
E QnnDsp  : Transport layer setup failed: 14001
E QnnDsp  : Failed to parse default platform info: 14001            ← PlatformInfo 자체는 parsing 진입 성공
E QnnDsp  : Failed to parse platform config: 14001
```

`logs_stdout.txt`:

```
preload RTLD_GLOBAL skip: libQnnHtpV79.so (dlopen failed: ... is 32-bit instead of 64-bit)
preload RTLD_GLOBAL skip: libQnnHtpV79Skel.so (dlopen failed: ... is 32-bit instead of 64-bit)
dlopen libQnnHtp.so RTLD_GLOBAL OK
HTP custom config: arch=V79, useSignedProcessDomain=false
Error: deviceCreate err=0x36b1
```

---

## 가설 ranking 결정적 갱신

| 가설 | 직전 (handoff R1) | Q-2.1 후 | 근거 |
|---|---|---|---|
| (1) `.farf` sentinel | low | low | — (Researcher 2 cross-check 시점부터 격하) |
| (2) RTLD_GLOBAL 미사용 | medium | **반증** | 6 lib 중 4 lib `RTLD_NOW\|RTLD_GLOBAL` preload 성공 후에도 동일 fail. 2 lib skip (32-bit) 은 host RTLD 무관 (DSP skel side) |
| (3) PlatformInfo block 누락 | ★ high | **반증** | PlatformInfo parsing 진입 자체는 성공 (`Failed to parse default platform info` 는 transport layer 의 downstream error). transport (`Control stubbed routine`) 가 그 전에 죽음 |
| (3b) backendApiVersion 3.7.0 | 부적용 | 부적용 | — (직전 handoff 에서 정정 완료) |
| (5) **OS-level FastRPC ACL** | unverifiable | **★ 확정** | `adsprpc remote.c:40 Control stubbed routine - Return failure` 가 직접 증명. stock S25 non-rooted process 의 FastRPC handle 가 ACL 차단됨 |

PlatformInfo 추가의 부수 효과: backend 가 transport 시도 단계까지 진입하여 Q-2.0 에서 unverifiable 이던 가설 (5) 를 logcat 으로 확정. **그 자체로 본 dry-run 의 main contribution**.

---

## 보너스 finding — `Specified config ARCH, ignoring on real target`

Q-2.0 의 +99 LOC 중 `QNN_DEVICE_CONFIG_OPTION_HTP_ARCH` entry (ARCH=V79) 는 logcat 에 따르면 **real target 에서는 ignored**. simulator path 전용 config 였음. Q-2.0 의 1/3 패치가 본디 no-op. Q-2.1 에서 PlatformInfo + SIGNEDPD 만 effective.

---

## 사용자 결정 (2026-05-26)

직접 인용: **"llama.cpp의 방법 사용해. fastrpc 사용하면 해결 가능한거 아니야?"**

근거:
- llama.cpp ggml-hexagon mainline (Researcher 2 분석) 은 QNN API 우회 + **FastRPC IDL 직접 사용** + `DSPRPC_CONTROL_UNSIGNED_MODULE` (`ggml-hexagon.cpp:1607-1616`) 로 unsigned PD 활성화. stock S25 GREEN (Llama-3.2-1B Q4_0 tg64=51.54 t/s @ S8 Elite).
- 본 dry-run 의 차단 layer = QNN API `domain_init` 의 ACL 검사. FastRPC layer 자체는 살아있음 (llama.cpp 가 같은 stock S25 에서 동작).
- path 자체를 QNN HTP API → FastRPC IDL 로 갈아끼면 ACL 차단 mechanism 자체를 피해갈 수 있음.

**다음 sprint = Q-2.2 (재정의)**: 독립 FastRPC + HVX skel backend. llama.cpp ggml-hexagon 의 init flow 와 `libggml-htp-vNN.so` 자체 skel 구조를 reference 로 차용.

---

## 변경 파일

| 경로 | 변경 |
|---|---|
| `engine/microbench/htp_matmul_correctness.rs` | +142 LOC / -10 LOC. `qnn_htp` mod 에 `QnnHtpDevice_OnChipDeviceInfoExtension_t` + `QnnHtpDevice_DeviceInfoExtension_t` 추가. `dlopen_global` helper + 6 sibling lib preload sequence. PlatformInfo block (CoreInfo → HardwareDeviceInfo → PlatformInfo) + `QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO` config entry |
| `engine/Cargo.toml` | 변경 없음 (libloading 0.8 이미 적용) |
| `papers/.../qnn_q21_dryrun_2026_05_26/logs_stdout.txt` | runtime stdout (12 lines) |
| `papers/.../qnn_q21_dryrun_2026_05_26/logs_logcat.txt` | runtime logcat (14 lines) |
| `papers/.../qnn_q21_dryrun_2026_05_26/report.md` | 본 문서 |

---

## 진단 가치 보존 정책

본 sprint commit 정책 (사용자 결정 — 2026-05-26):

- patch (+142 LOC) 는 dead path 가 아니라 **다음 sprint Q-2.2 (FastRPC) 가 QNN API path 와의 fair-pair 비교 base 로 활용 가능**. cleanup 하지 않음.
- 옵션 1 (rooted/dev-fused S25 hardware 확보) 재시도 시 starting point.
- logs raw + report.md 모두 paper supplementary data 로 보존.

---

## 자기점검

- [x] 진입 문장: 다음 sprint Q-2.2 = "llama.cpp FastRPC IDL path 차용 + 독립 backend 신설"
- [x] 멈춘 이유: FastRPC ACL 확정 → 다음 sprint 는 path 교체로 우회 (사용자 결정)
- [x] landmine: 가설 (3) PlatformInfo 반증, (2) RTLD_GLOBAL 반증, (5) FastRPC ACL 확정. Q-2.0 ARCH config no-op 보너스
- [x] 검증 게이트: logcat raw 인용 + `adsprpc Control stubbed` direct evidence
- [x] 본문 길이: ~700 토큰 (handoff-doc 권장 500 토큰 초과 — 직접 증명 logcat 보존 가치)
