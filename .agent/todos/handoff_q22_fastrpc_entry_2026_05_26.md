# Handoff: Q-2.1 RED 종결 → Q-2.2 FastRPC IDL backend 진입 plan

**작성**: 2026-05-26
**HEAD**: `da22ea82 exp(microbench): Q-2.1 dry-run RED — FastRPC ACL 차단 직접 증명`
**브랜치 / Worktree**: `worktree-b5_trait_extension` (`/home/go/Workspace/llm_rs2/.claude/worktrees/b5_trait_extension`)
**작성자**: main session (Researcher 위임 결과 종합)

**다음 세션 진입 문장**: "Q-2.2 진입 — FastRPC IDL path 독립 backend Architect spec"

---

## TL;DR

Q-2.1 dry-run 결과 **stock S25 의 raw QNN HTP API path 가 OS-level FastRPC ACL 로 차단됨** 확정 (logcat `adsprpc remote.c:40 Control stubbed routine - Return failure`). 사용자 결정 = **llama.cpp ggml-hexagon 처럼 QNN API 우회 + FastRPC IDL 직접 path** 차용. Researcher 분석 (mainline llama.cpp) 결과: host-side 는 `libcdsprpc.so` 의 17 symbols dlopen + `DSPRPC_CONTROL_UNSIGNED_MODULE` sequence 만으로 GREEN. DSP-side skel 은 llama.cpp `htp/` 통째 차용 (MIT, C 작성, 6 arch variant). 예상 sprint 규모 3-4 주 (α/β/γ/δ 4 단계). Architect spec 단계가 진입 게이트.

---

## 진행 상태

### Task

| ID | 상태 | 작업 | 산출물 |
|---|---|---|---|
| Q-2.0 dry-run | ✅ completed | 9 bin sweep + device config fix → RED | `handoff_q20_dryrun_complete_2026_05_26.md` |
| Q-2 mechanism 분석 | ✅ completed | Executorch + llama.cpp 정밀 분석 → 가설 ranking | `handoff_qnn_backend_research_2026_05_26.md` + `project_qnn_backend_research.md` |
| **Q-2.1 dry-run** | ✅ completed | PlatformInfo + RTLD_GLOBAL 2-shot → RED + FastRPC ACL 확정 | commit `da22ea82` + report.md |
| **Q-2.2-R1 Researcher 분석** | ✅ completed | llama.cpp FastRPC mechanism 6 항목 정밀 분석 | in-session response (본 handoff R4 압축) |
| Q-2.2-α Architect spec | ⏳ **다음 세션 진입 대상** | spec/htp_fastrpc_backend.md + arch/htp_fastrpc.md | - |
| Q-2.2-α SDK env | ⏳ blocked-by spec | Hexagon SDK 6.4+ Docker pull + HEXAGON_SDK_ROOT 표준화 | - |
| Q-2.2-β host binding | ⏳ blocked-by α | libcdsprpc dlopen + 17 symbol Rust binding | - |
| Q-2.2-γ skel + repack | ⏳ blocked-by β | 자체 skel build + GGUF→q4x4x2 repack + first-layer GREEN | - |
| Q-2.2-δ (optional) | ⏳ blocked-by γ | rope + flash_attn + 전체 16 layer 추론 | - |

### Q-2.1 측정 결과 (확정)

| 단계 | 결과 |
|---|---|
| PlatformInfo block + 6 lib RTLD_GLOBAL preload | 적용 (host build PASS) |
| backendCreate | OK |
| **deviceCreate (ARCH + SIGNEDPD + PLATFORM_INFO)** | **FAIL err=0x36b1** (= 14001) |

logcat 결정적 증거: `E adsprpc : remote.c:40:Control stubbed routine - Return failure`.

가설 ranking 최종:
- (2) RTLD_GLOBAL: **반증** (preload 후에도 fail)
- (3) PlatformInfo: **반증** (parsing 진입 성공, transport 가 그 전에 죽음)
- (5) OS-level FastRPC ACL: **★ 확정**
- 보너스: ARCH=V79 custom config 는 real target 에서 no-op (`Specified config ARCH, ignoring on real target`)

### Researcher 분석 결과 (6 항목 압축)

**A. IDL 단일 source** — `llama.cpp/ggml/src/ggml-hexagon/htp/htp_iface.idl` (17줄, lifecycle 4 메서드만). op dispatch 는 dspqueue out-of-band path → IDL 우회. qaic IDL compiler 는 Hexagon SDK ship.

**B. Host symbol surface** — `libcdsprpc.so` 17 symbols (htp-drv.cpp:32-360):
- rpcmem_alloc/free/to_fd, rpcmem_alloc2
- fastrpc_mmap/munmap
- dspqueue_create/close/export/write/read
- remote_handle64_open/invoke/close, remote_handle_control, remote_handle64_control
- **remote_session_control** (FASTRPC_RESERVE_NEW_SESSION + FASTRPC_GET_URI + **DSPRPC_CONTROL_UNSIGNED_MODULE**)

본 프로젝트 Rust `libloading` 으로 100% 가능, Hexagon SDK 정적 링크 불필요.

**C. DSP entrypoint** — `htp_iface_skel_handle_invoke`. main.c:1007 `htp_packet_callback` 가 op enum switch. HVX intrinsic (`Q6_V_vand_VV`, `Q6_Vub_vlsr_VubR`) + VTCM 8MB scratchpad. 4 worker threads (n_hvx).

**D. ★ unsigned PD sequence** (차단 우회 핵심, ggml-hexagon.cpp:1559-1616):
1. (optional) `FASTRPC_RESERVE_NEW_SESSION` → effective_domain_id, session_id
2. `FASTRPC_GET_URI` → session_uri (file://libggml-htp-v79.so?...)
3. **`DSPRPC_CONTROL_UNSIGNED_MODULE.enable=1`** ← QNN API 가 호출 못 하는 그 호출
4. `remote_handle64_open(session_uri, &handle)` GREEN

stock S25 GREEN 증거: `tg64=51.54 t/s @ S8 Elite` (README.md:135-225). **QNN API 가 차단된 이유** = vendor ACL 의 QNN-specific block (Q-2.1 logcat). `libcdsprpc.so` 직접 호출은 ACL 우회.

**E. 차용 challenge**:
- **DSP skel 은 Rust 불가** (QURT RTOS + HVX intrinsic). C 로 llama.cpp `htp/` 통째 vendor (MIT)
- Hexagon SDK 6.4+ 필수 (Docker `ghcr.io/snapdragon-toolchain/arm64-android:v0.3` 5.2GB)
- **Weight repack 필수**: GGUF Q4_0 (32-block) → HTP q4x4x2 (256 super-block). Phase 1 runtime / Phase 2 AUF secondary variant
- 4 arch variant (v75 + v79 ship 권장): 8 Gen3 + S8 Elite/S25

**F. Sprint scope** ~6200 LOC (C 3000 차용 + Rust 3200 신규):
- α(1주): Architect spec + SDK env + DSP skel vendor + host Rust binding 1차
- β(1주): session lifecycle + rpcmem + dspqueue + MatMul microbench (DSP skel 은 llama.cpp 빌드물 임시 사용)
- γ(1주): 자체 skel 빌드 + GGUF→q4x4x2 repack + Backend trait + first-layer FP32 동일성 GREEN
- δ(1주, optional): rope + flash_attn + 전체 16 layer 추론 GREEN

---

## 다음 작업 (다음 세션 진입)

### 액션 (Q-2.2-α Architect spec 단계)

1. **Architect 위임** → spec 작성. 검증: spec ID 할당 + 도식 + 결정 항목 답변 완료
   - 파일: `spec/htp_fastrpc_backend.md` (요구사항/제약/inv) + `arch/htp_fastrpc.md` (구조/모듈/data flow)
   - 입력: 본 handoff 의 R4 (A-F) + 본 handoff 의 R6 risks
   - 결정해야 할 trade-off 6건 (Researcher F 항목):
     - llama.cpp `htp/` 통째 vendor vs 부분 차용 → 권장 = **통째 vendor + LICENSE.MIT 명시**
     - IDL: llama.cpp 의 것 그대로 → 권장 = **그대로 차용** (lifecycle 4 메서드)
     - Hexagon SDK runtime dep: docker 강제 vs 자체 mini-sdk → 권장 = **docker image + HEXAGON_SDK_ROOT env check**
     - Weight repack: runtime vs AUF build-time → 권장 = **Phase 1 runtime, Phase 2 AUF secondary variant**
     - Multi-session: dev_id != 0 skip 여부 → 권장 = **Phase 1 single session (dev_id=0)**
     - DSP arch coverage: 6 종 vs 선택 → 권장 = **v75 + v79 2 종 ship** (Phase 1 은 v79 단독)

2. **SDK env 셋업** (spec 확정 후) → 검증: docker pull 성공 + `cmake --version` (in container) + Hexagon SDK 6.4+ 확인

3. **Q-2.2-α 산출물** → 검증:
   - `spec/htp_fastrpc_backend.md` (스펙)
   - `arch/htp_fastrpc.md` (구조)
   - `htp_skel/` 디렉토리 (llama.cpp vendor) + LICENSE.MIT
   - `crates/qnn_fastrpc/` (host-side Rust binding crate 신설) — Cargo.toml 만, 코드는 β 단계

### 위임 prompt (Q-2.2-α Architect spec, 다음 세션 사용)

> **에이전트**: `architect`
> **모델**: `opus`
> **권한**: `spec/`, `arch/`, `docs/*.md`, `ARCHITECTURE.md` (코드 수정 불가, design only)

```
Q-2.2-α Architect spec: HTP FastRPC IDL backend 독립 신설.

# 입력
- 본 handoff: `.agent/todos/handoff_q22_fastrpc_entry_2026_05_26.md` (전체 R4 + R6)
- Researcher raw (in-session, 본 handoff R4 에 압축)
- 참고: llama.cpp ggml-hexagon (mainline, MIT) - `/home/go/Workspace/llama.cpp/ggml/src/ggml-hexagon/`

# 산출물
1. `spec/htp_fastrpc_backend.md` — INV-HTP-FRPC-001 등 ID 할당. 요구사항 + 제약 + invariants
2. `arch/htp_fastrpc.md` — Mermaid 다이어그램 + 모듈 구조 + data flow + Backend trait integration point
3. `ARCHITECTURE.md` §13 (or §14) 추가 — backend matrix 갱신 (qnn_oppkg / opencl / cuda_* + qnn_fastrpc 행)

# 결정해야 할 6 trade-off (본 handoff R5 에 권장값 명시)
- DSP skel vendor 정책 (통째 vs 부분)
- IDL 차용 정책 (그대로 vs 자체)
- Hexagon SDK 의존 (docker vs vendored)
- Weight repack 시점 (runtime vs build-time)
- Multi-session (Phase 1 dev_id=0 single vs 처음부터 multi)
- DSP arch coverage (v75+v79 vs 6 종)

# 제약
- 코드 수정 금지 (spec/docs 만)
- 본 sprint scope = α 단계까지 (β γ δ 는 후속 sprint)
- 본 프로젝트 기존 Backend trait 와 충돌 없게 (engine/src/backend/* 파일은 read-only 참고)
- 본 프로젝트 기존 AUF/GGUF loader 와 호환 (Phase 1 은 runtime repack 만)

# 검증 게이트
- spec test (`/spec-manage`) 통과
- ARCHITECTURE.md 의 backend matrix 갱신 후 layer_lint 회귀 0
- arch/htp_fastrpc.md 의 Mermaid 가 정상 render

cap: 1 일
```

---

## Landmines / 미해결 / 안 가본 길

### 가장 큰 risk (R1) — Hexagon SDK 환경 자체

hexagon-clang + QURT RTOS + qaic IDL + cdsp domain — 모두 Qualcomm 전유 toolchain. 본 프로젝트의 기존 NDK + CUDA + OpenCL 토얼체인과 완전 별도 라인. Docker image (5.2GB) 의존 사실상 강제. CI 통합 시간 + 신규 build target maintainer 비용 발생.

**대응**: α 단계의 SDK env 셋업 task 가 spec 단계와 병렬 진행. docker pull 실패 / hexagon-clang 버전 mismatch / qaic compile error 등 환경 이슈는 spec 확정 전에 격리.

### R2 — q4x4x2 repack 정확성

GGUF Q4_0 (32-block, FP16 scale + 16 byte qs = 18B/block) → HTP q4x4x2 (256 super-block = 8 GGUF blocks, 128B qs + 16B scale concat). off-by-one 빈발 영역.

**대응**: Phase γ 의 first-layer FP32 동일성 검증을 microbench 부터. llama.cpp `repack_row_q4x4x2` (ggml-hexagon.cpp:402-465) 의 verbose level 2/3 dump 패턴 차용.

### R3 — dspqueue cache 일관성

ggml-hexagon 는 SystemHeap rpcmem + `fastrpc_mmap(FASTRPC_MAP_FD)` 1회 mmap + 매 packet `dspqueue_buffer{fd,ptr,size,offset,flags}` 동봉. `DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | INVALIDATE_RECIPIENT` 강제 (main.c:412, 1001). 본 프로젝트 Phase 6.5 의 cache 비일관 함정과 유사 패턴.

**대응**: Phase β 의 microbench 부터 flush/invalidate flag 정확성 확인. `read_buffer` 누락 시 stale cache 확보.

### R4 — ABI mismatch (낮음, monitor)

`remote_handle64` 가 IDL 측 `uint64_t` vs C 측 weak `void*` alias. bindgen 자동 변환 시 type alias 확인 필수.

**대응**: spec 단계에서 ABI 명시 + Phase β 의 Rust binding 첫 commit 에 단위 테스트 (handle 값 64-bit cast 확인).

### 시도했지만 실패한 방향 (반증)

- 가설 (1) `.farf` sentinel — Researcher 2 가 logging mask 로 확정 (signing 무관)
- 가설 (2) RTLD_GLOBAL — Q-2.1 에서 preload 후에도 동일 fail (반증)
- 가설 (3) PlatformInfo block — Q-2.1 에서 parsing 진입은 성공, transport 가 그 전에 죽음 (반증)
- 가설 (3b) backendApiVersion 3.7.0 — INV-154 spec 은 OpPackage path 전용, raw HTP client 부적용 (정정)
- 가설 (4) skel 경로/이름 — deviceCreate 실패라 skel 검색 단계 미진입 (low 유지)

### 결정 대기

- 사용자: Q-2.2-α spec 확정 후 β 진입 게이트 (Architect 산출물 review)
- Architect: 본 handoff R5 의 6 trade-off 권장값 채택 vs 별 결정
- 본 sprint 전체 cap: **3-4 주** (α/β/γ + optional δ). 2 주 sprint 로는 viable backend 까지 미달

### 이 길은 가지 말 것

- **raw QNN HTP API path 재시도** — Q-2.1 에서 root cause 확정 (vendor ACL). 옵션 1 (rooted/dev-fused S25) 외에는 우회 불가
- **Rust 로 DSP skel 작성 시도** — Researcher 명시: hexagon-clang 만 HVX intrinsic + QURT link 가능. rustc 의 hexagon target 은 RTOS link 불가
- **Hexagon SDK in-tree vendor** — 5GB+ git LFS 부담. Docker image 의존이 정도
- **Phase E (Executorch M7 W8A8 0.276 ms) 와의 1:1 비교** — Phase E 는 별 path (Executorch wrap). Q-2.2 의 FastRPC backend 는 별도 row. paper chart 에 추가 행만

---

## 참고 링크

- 직전 handoff: `.agent/todos/handoff_qnn_backend_research_2026_05_26.md` (Q-2.0 + 가설 ranking 종결)
- Q-2.1 report: `papers/eurosys2027/_workspace/experiment/qnn_q21_dryrun_2026_05_26/report.md`
- Researcher (in-session) 분석 raw: 본 세션 transcript (Q-2.2-R1)
- llama.cpp ggml-hexagon: `/home/go/Workspace/llama.cpp/ggml/src/ggml-hexagon/` (전체)
- 핵심 인용 위치:
  - IDL: `llama.cpp/ggml/src/ggml-hexagon/htp/htp_iface.idl` (17줄)
  - 17 symbols: `htp-drv.cpp:32-360`
  - unsigned PD sequence: `ggml-hexagon.cpp:1559-1616`
  - DSP entrypoint: `htp/main.c:28-313` (lifecycle), `main.c:1007-1198` (packet dispatch)
  - q4x4x2 repack: `ggml-hexagon.cpp:402-465`
  - stock S25 GREEN: `docs/backend/snapdragon/README.md:135-225`
- 본 프로젝트 Q-2.1 baseline: `engine/microbench/htp_matmul_correctness.rs:99-167` (QNN API path RED, Q-2.2 의 FastRPC path GREEN target)
- 메모리:
  - `project_qnn_backend_research.md` (Q-2.0 + Q-2.1 + Q-2.2 scope)
  - `project_qnn_oppkg_phase_r_complete_20260509.md` (INV-154 OpPackage spec)
  - `reference_llama_cpp_source.md` (llama.cpp clone path)
