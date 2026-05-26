# Handoff: Executorch + llama.cpp 분석 종결 → Q-2 독립 QNN backend 신설

**작성**: 2026-05-26
**HEAD**: `13e0849d docs(handoff): 다음 세션 작업 1+2 진입 plan`
**브랜치 / Worktree**: `master` (`/home/go/Workspace/llm_rs2/.claude/worktrees/b5_trait_extension`)
**작성자**: main session (researcher 위임 2건 + cross-check)

**다음 세션 진입 문장**: "Q-2 진입 — 독립 QNN backend 신설"

---

## TL;DR

Executorch HTP path 와 llama.cpp ggml-hexagon backend 정밀 분석 종결. 사용자 방침 = **본 프로젝트 자체 QNN backend 소유** (wrap path abandon). Executorch / llama.cpp 는 init flow reference 로만 활용. 차단 mechanism 가설은 PlatformInfo block + RTLD_GLOBAL 2가지로 좁혀짐. Q-2.0 dry-run 의 +99 LOC 패치 (878f0ae3) 는 dead — 이제 PlatformInfo + RTLD_GLOBAL 재시도. INV-154 `backendApiVersion 3.7.0` fix 는 OpPackage path 의 spec 이지 raw HTP client path 에는 부적용 (정정).

---

## 진행 상태

### Task

| ID | 상태 | 작업 | Commit |
|---|---|---|---|
| Researcher 1 | ✅ completed | Executorch HTP init flow 정밀 분석 (HtpDevice.cpp:295~427 + PlatformInfo + RTLD_GLOBAL + `.farf`) | - (in-session) |
| Researcher 2 | ✅ completed | llama.cpp ggml-hexagon mainline 분석 + 3-way diff | - (in-session) |
| INV-154 정정 | ✅ completed | `crates/qnn_oppkg/tests/spec/test_inv_154_api_version.rs:10-12` 는 OpPackage path 전용으로 확인 | - |
| **Q-2.1 dry-run** | **⏳ 이번 세션 진입 대상** | PlatformInfo block + RTLD_GLOBAL 2-shot 패치 → `microbench_htp_matmul_correctness` 부활 검증 | - |
| Q-2.2 backend 신설 | ⏳ blocked-by Q-2.1 | `engine/src/backend/qnn_htp/` Backend trait 신설 (Q-2.1 PASS 후) | - |
| Q-2.3 model 연결 | ⏳ blocked-by Q-2.2 | Llama 3.2 1B / Qwen 2.5-1.5B end-to-end | - |
| Q-2.4 device gate | ⏳ blocked-by Q-2.3 | S25 token gen 정확성 + TBT vs `--backend qnn_oppkg` (13 ms/tok) 비교 | - |

### 측정 / 검증 결과

Researcher 분석 기반 (확정 사실):

| 항목 | 값 | 근거 |
|---|---|---|
| Executorch HtpDevice MakeConfig | socModel + PlatformInfo (vtcm=8MB + signedPdSupport=false + arch=V79 + dlbcSupport=true) + SIGNEDPD + ARCH | `/home/go/Workspace/executorch/backends/qualcomm/runtime/backends/htp/host/HtpDevicePlatformInfoConfig.cpp:13-58` |
| Executorch dlopen flags | `RTLD_NOW \| RTLD_GLOBAL` | `executorch/backends/qualcomm/runtime/backends/QnnImplementation.cpp:60-80` |
| `.farf` 의미 | FARF logging mask (`0x0C` Executorch, `0xffff` llama.cpp) — signing/권한 무관 | `executorch/backends/qualcomm/export_utils.py:469` + `llama.cpp/scripts/snapdragon/adb/llama-cli.farf` |
| llama.cpp ggml-hexagon | mainline 머지 + stock S25 GREEN. **QNN API 미사용** (FastRPC IDL + `libggml-htp-vNN.so` 자체 skel). Llama-3.2-1B Q4_0 tg64=51.54 t/s @ S8 Elite | `/home/go/Workspace/llama.cpp/ggml/src/ggml-hexagon/ggml-hexagon.cpp:1535-1700` + mainline README |
| llama.cpp dlopen | `RTLD_NOW \| RTLD_LOCAL` | `llama.cpp/ggml/src/ggml-hexagon/libdl.h:65-68` |
| 본 프로젝트 raw bin (878f0ae3) | `arch=V79 + SIGNEDPD=false` 만 추가. PlatformInfo block 통째 누락 | `engine/microbench/htp_matmul_correctness.rs:122-167` |
| 본 프로젝트 dlopen | libloading default = `RTLD_LAZY \| RTLD_LOCAL` | `htp_matmul_correctness.rs:100` |
| Q-2.0 dry-run sweep | 9 bin 모두 RED (deviceCreate err=0x36b1 AEE_ENOSUCHMOD) | `papers/eurosys2027/_workspace/experiment/qnn_q20_dryrun_2026_05_26/report.md` |

### 가설 ranking (Researcher 2 cross-check 후 갱신)

| 가설 | 직전 (Researcher 1) | 갱신 후 | 근거 |
|---|---|---|---|
| (1) `.farf` sentinel 누락 | 유력 | **격하 (low)** | llama.cpp 도 `.farf=0xffff` 만 — logging mask. signing 무관 확정 |
| (2) RTLD_GLOBAL 미사용 | 유력 | **유력 (medium)** | llama.cpp 는 `RTLD_LOCAL` 로 동작 → 결정적 아닐 가능성 ↑. 단 QNN HTP API path 에서 cross-lib symbol 해결 필요성은 별개 — 1 line fix 라 같이 시도 |
| (3) PlatformInfo block 통째 누락 | 유력 | **★ 유력 (high)** | QNN API layer 의 device-side dispatch 가 socModel/vtcm/signedPdSupport 명시를 요구할 가능성. llama.cpp 는 QNN 우회라 cross-check 불가. Executorch GREEN path 에 mandatory |
| (3b) backendApiVersion 3.7.0 | 신규 후보 | **부적용 (격하)** | INV-154 spec 은 OpPackage lib 의 declare 용. raw HTP client 의 `backendApiVersion` 은 SDK 가 반환하는 query 값 — user 셋팅 대상 아님 |
| (4) skel 경로/이름 | low | low | deviceCreate 실패라 skel 검색 단계 미진입 |
| (5) SELinux / process signature | unverifiable | unverifiable | stock S25 root 없이 추적 불가 (기존 R6 보존) |

---

## 다음 작업 (다음 세션이 그대로 사용 가능)

### 액션

1. **Q-2.1 dry-run — PlatformInfo block + RTLD_GLOBAL 2-shot 패치** → 검증: `microbench_htp_matmul_correctness` S25 PASS (deviceCreate + contextCreate + graphCreate 까지 진행)
   - PlatformInfo: Executorch `HtpDevicePlatformInfoConfig.cpp:13-58` 의 `QnnDevice_PlatformInfo_t` block 그대로 차용 — `version=1, numHwDevices=1, deviceId=0, numCores=1, devType=ON_CHIP, vtcm_size_in_mb=8 (V79=8MB), signedPdSupport=false, socModel=57 (SM8750), arch=V79, dlbcSupport=true, coreId=0, coreType=0`
   - RTLD_GLOBAL: libloading 의 `Library::open_with_flags` 또는 직접 `libc::dlopen` 사용으로 변경. libloading 0.8+ 가 `OpenFlags::NOW | OpenFlags::GLOBAL` 지원
   - `engine/microbench/htp_matmul_correctness.rs:99-167` 영역만 수정. 다른 8 bin 은 한 bin PASS 후 일괄 패치
2. **Q-2.1 결과에 따라 분기**:
   - PASS → Q-2.2 본격 진입 (`engine/src/backend/qnn_htp/` Backend trait 신설, Architect spec 단계 선행)
   - FAIL → 가설 (5) OS-level 정책 영역 → 옵션 2 (Executorch wrap) 또는 옵션 3 (HTP 포기) 재결정. handoff R6 의 stock-root 한계 그대로
3. **Q-2.2 (Q-2.1 PASS 시)** → Backend trait 신설 → 검증: `cargo test -p llm_rs2 --features qnn --lib` PASS
4. **Q-2.3** → Llama 3.2 1B / Qwen 2.5-1.5B end-to-end (model load + forward + KV cache) → 검증: S25 token gen 정확성 + Phase E 의 M7 W8A8 0.276 ms / M6b F16 0.456 ms 재현
5. **Q-2.4 device gate** → 검증: `--backend qnn_htp` vs `--backend qnn_oppkg` (13 ms/tok) TBT 비교

### 위임 prompt (Q-2.1 dry-run, Senior Implementer)

> **에이전트**: `senior-implementer`
> **모델**: `opus`
> **권한**: 수정 가능 경로 — `engine/microbench/htp_matmul_correctness.rs`, `engine/Cargo.toml` (libloading version bump 필요 시), `papers/eurosys2027/_workspace/experiment/qnn_q21_dryrun_2026_05_26/`

```
Q-2.1 dry-run 진입: PlatformInfo block + RTLD_GLOBAL 2-shot 패치로 raw QNN HTP API path 의 stock Galaxy S25 차단 우회 검증.

# 배경
직전 Q-2.0 dry-run (878f0ae3) 의 +99 LOC device config (arch=V79 + SIGNEDPD=false) 만으로는 deviceCreate err=0x36b1 (AEE_ENOSUCHMOD) 발생. handoff_qnn_backend_research_2026_05_26.md 의 R1/R2 Researcher 분석에 따라 진짜 차단 mechanism 후보는 (3) PlatformInfo block 통째 누락 + (2) RTLD_GLOBAL 미사용.

# Reference (read-only)
- Executorch HtpDevicePlatformInfoConfig: `/home/go/Workspace/executorch/backends/qualcomm/runtime/backends/htp/host/HtpDevicePlatformInfoConfig.cpp:13-58`
- Executorch QnnImplementation dlopen: `/home/go/Workspace/executorch/backends/qualcomm/runtime/backends/QnnImplementation.cpp:60-80`
- Executorch SocInfo (SM8750→V79 매핑): `/home/go/Workspace/executorch/backends/qualcomm/utils/utils.py:1278-1315`
- 본 프로젝트 raw bin 현 상태: `engine/microbench/htp_matmul_correctness.rs:99-167` (878f0ae3 이후)

# 검증 게이트
1. Patch 1 — PlatformInfo block 추가:
   - QnnDevice_PlatformInfo_t struct 정의 (qnn-sys bindgen 에 있으면 reuse, 없으면 본 file 안에 local definition)
   - 값: version=1, numHwDevices=1, deviceId=0, numCores=1, devType=ON_CHIP, vtcm_size_in_mb=8, signedPdSupport=false, socModel=57 (SM8750), arch=V79, dlbcSupport=true, coreId=0, coreType=0
   - QnnDevice_Config_t array 에 `QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO` entry 추가 (기존 ARCH + SIGNEDPD 뒤)
2. Patch 2 — RTLD_GLOBAL 전환:
   - libloading 0.8+ 의 `Library::open_with_flags` 또는 unsafe `libc::dlopen("...", RTLD_NOW | RTLD_GLOBAL)` 직접 사용
   - 7 lib 모두 (libQnnHtp + libQnnSystem + libQnnHtpV79Stub + libQnnHtpV79 + libQnnHtpV79Skel + libQnnHtpPrepare + libQnnHtpV79CalculatorStub) 같은 패턴
3. S25 device run:
   - `python scripts/run_device.py -d s25 microbench_htp_matmul_correctness`
   - Expected: deviceCreate OK + contextCreate OK + graphCreate OK
   - logcat 캡처 (Skel publish 확인)
4. 측정 결과 보고서: `papers/eurosys2027/_workspace/experiment/qnn_q21_dryrun_2026_05_26/report.md`
   - PASS/FAIL + 단계별 진행 (deviceCreate/contextCreate/graphCreate) + err code
   - PASS 시: 1B/1.5B 모델 통합의 다음 sprint 진입 게이트 통과
   - FAIL 시: 가설 (5) OS-level 정책 → stock-root 한계 → handoff R6 그대로 + 옵션 2/3 결정 사용자 위임

# 제약
- 본 sprint 의 다른 8 bin 패치는 single bin PASS 확인 후 진행 (1 bin 부터 검증)
- baseline `arch=V79 + SIGNEDPD=false` 는 그대로 유지 — 추가일 뿐
- libloading version bump 필요 시 Cargo.toml minimal change 만 (other crate dep 영향 검증)
- 본 dry-run 의 cap: 1d (FAIL 시 빠른 abandon)
```

---

## Landmines / 미해결 / 안 가본 길

- **PlatformInfo + RTLD_GLOBAL 양쪽 다 fail 시 stock S25 root 없이 추적 불가**. `handoff_q20_dryrun_complete_2026_05_26.md` R6 #1 의 한계 그대로 유지. 옵션 2 (Executorch wrap) / 옵션 3 (HTP 포기) 결정 사용자 위임
- **llama.cpp ggml-hexagon 은 QNN API 우회 path** — mainline 머지된 stock-GREEN 사실은 본 프로젝트 raw QNN path 의 mechanism 추적에 직접 도움 안 됨. **개념 reference 만** (RTLD_LOCAL 도 가능 / `.farf` 는 logging only). 본 프로젝트가 QNN HTP API path 를 고수하면 mainline 코드 직접 차용 안 됨
- **시도했지만 실패한 방향** (격하):
  - 가설 (1) `.farf` sentinel — Researcher 2 가 logging mask 로 확정, signing 무관
  - 가설 (3b) `backendApiVersion = 3.7.0` fix — INV-154 spec 은 OpPackage path 전용. raw HTP client path 에서는 SDK 가 반환하는 query 값, user 셋팅 대상 아님 (정정 완료)
- **결정 대기** — 사용자: Q-2.1 PASS 후 Q-2.2 spec 단계에서 Backend trait 의 정확한 scope (OpPackage 와 별도 backend, 또는 통합 backend with feature flag)
- **이 길은 가지 말 것** — Researcher 1 의 "1d 안에 3-shot 패치 (.farf + RTLD_GLOBAL + PlatformInfo)" 권장. `.farf` 는 logging mask 라 첫 시도에 굳이 포함 안 함 (단순화)
- **LOC 증가 예상**: dry-run +60~100 LOC (PlatformInfo struct + RTLD_GLOBAL dlopen). Q-2 본격 진입 시 `engine/src/backend/qnn_htp/` 200~400 LOC + Backend trait impl 100~200 LOC 추가
- **paper figure 영향**: Q-2.1 PASS 시 cross-backend chart 에 `--backend qnn_htp` row 추가 필요. Phase E 의 M6b/M7 결과는 그대로 paper main evidence (Executorch path 측정값) — 본 프로젝트 raw path 와의 fair-pair 비교 base 로 활용

---

## 참고 링크

- 상위 plan / spec: `arch/inference_pipeline.md` (Phase 4 종결 시 backend matrix 정의)
- 직전 handoff: `.agent/todos/handoff_q20_dryrun_complete_2026_05_26.md` (raw QNN HTP API path RED 확정)
- 메모리:
  - `project_qnn_oppkg_phase_r_complete_20260509.md` (INV-154 backendApiVersion 3.7.0 — OpPackage spec)
  - `project_qnn_oppkg_m2_complete_20260510.md` (M2 layer-level graph)
  - `reference_llama_cpp_source.md` (llama.cpp clone path)
- Researcher 1 분석 산출물: in-session response (Executorch HtpDevice init flow + PlatformInfo block + RTLD_GLOBAL + `.farf` 분석)
- Researcher 2 분석 산출물: in-session response (llama.cpp ggml-hexagon mainline 분석 + 3-way diff + INV-154 정정 + 옵션 2b 평가)
- PR #11 (직전 sprint): https://github.com/hedone21/llm_rs2/pull/11
