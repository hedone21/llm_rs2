# Handoff: 다음 세션 작업 1 + 2 (Executorch HTP 분석 + qnn_oppkg → opencl 통합)

**작성**: 2026-05-26
**HEAD**: `878f0ae3 exp(qnn-htp): Q-2.0 dry-run 종결 — raw HTP API path 차단 확정`
**브랜치**: `worktree-b5_trait_extension`
**다음 세션 진입 문장**: "Executorch HTP 분석 + qnn_oppkg → opencl 통합 진행"

**supersedes**: [[handoff_q20_dryrun_complete_2026_05_26]] 의 R5 "Q-2 재설계 결정 진행". 본 handoff 가 결정 후 진입.

---

## TL;DR

Q-2 sprint 가 raw HTP API path 차단으로 막혔고 본 세션의 mechanism 추적은 grep level 까지만 (cpp 본문 미열람). 사용자 결정으로 다음 세션에서 2 작업 진행:

1. **Executorch HTP 사용 mechanism 정밀 분석** — cpp 본문 Read 로 stock S25 publish path 확정 (0.5~1d, Researcher 위임)
2. **qnn_oppkg → opencl 통합** — production 실효 = rpcmem KV cache 만 옵션화, OpPackage 자체 제거 (1주~, dry-run 부터 분할)

권장 순서: **1 → 2** (또는 1 background + 2 dry-run main 병행). 작업 1 결과가 Q-2 방향 (raw 부활 / Executorch wrap / abandon) 을 결정하고, 그에 따라 작업 2 의 backend matrix 최종 모습이 그려짐.

---

## 진행 상태 (둘 다 pending)

| Task | 상태 | 위치 |
|---|---|---|
| 작업 1 Executorch HTP 분석 | pending (Researcher 위임 권장) | `/home/go/Workspace/executorch/backends/qualcomm/runtime/` |
| 작업 2a libQnnGpu.so 의존 dry-run | pending (메인) | `engine/src/backend/qnn_oppkg/`, `crates/qnn_oppkg/` |
| 작업 2b rpcmem allocator 추출 | pending | (작업 2a 결과 따라 spec) |
| 작업 2c opencl backend 통합 + flag | pending | `engine/src/backend/opencl/` + CLI 추가 |
| 작업 2d S25 회귀 + qnn_oppkg crate 제거 | pending | deploy-test gate |

---

## 다음 액션

### 작업 1: Executorch HTP 사용 mechanism 분석

**목적**: stock S25 (non-root, unsigned shell) 에서 Executorch `qnn_executor_runner` 가 어떻게 `libQnnHtpV79Skel.so` 를 DSP application PD 에 publish 하는지 진짜 mechanism 식별. 본 프로젝트 raw bin 의 fail (deviceCreate err=0x36b1 AEE_ENOSUCHMOD) 을 해소할 setup 차이 발견 또는 OS-level 차단 확정.

**Researcher 위임 prompt 골격**:
```
Executorch QNN HTP backend 의 stock device publish mechanism 추적.

배경: 본 프로젝트 raw QNN HTP API bin 이 stock S25 에서 contextCreate/deviceCreate
err=0x36b1 (AEE_ENOSUCHMOD) 로 fail. device custom config (kHtpUnsignedPd + ARCH=V79)
+ ADSP_LIBRARY_PATH 정확히 설정 후에도 fail 동일. 그러나 Executorch 의
qnn_executor_runner 는 동일 디바이스/lib/Skel.so 로 동작. mechanism 차이 미파악.

읽어야 할 핵심 파일 (Read 로 본문 전체):
1. /home/go/Workspace/executorch/backends/qualcomm/runtime/QnnExecuTorchBackend.cpp
   — init() / execute() entry point
2. /home/go/Workspace/executorch/backends/qualcomm/runtime/QnnManager.cpp
   — backend lifecycle
3. /home/go/Workspace/executorch/backends/qualcomm/runtime/backends/QnnBackendFactory.cpp
   — backend creation
4. /home/go/Workspace/executorch/backends/qualcomm/runtime/backends/QnnImplementation.cpp
   — libQnnHtp.so dlopen + provider lookup
5. /home/go/Workspace/executorch/backends/qualcomm/runtime/backends/htp/HtpDevice.cpp
   — createDeviceCustomConfig() 전체 본문, init order
6. /home/go/Workspace/executorch/backends/qualcomm/runtime/backends/htp/HtpBackend.cpp
   (있는 경우)
7. qnn_executor_runner main() — Executorch 자체 또는 examples/ 에 위치

확정해야 할 사항:
(a) backendCreate / contextCreate / deviceCreate 호출 순서 (본 프로젝트와 비교)
(b) device config 외 추가 setProperty / config 호출
(c) farf manifest 또는 .farf 파일의 효과 (export_utils.py:469 의 echo 0x0C > .farf)
(d) Skel publish 를 가능케 하는 코드 경로 식별

보고 형식:
- 호출 순서 + config 호출 표 (Executorch vs 본 프로젝트)
- 본 프로젝트가 빠뜨린 setup 목록 (있으면)
- mechanism 식별 verdict:
  GREEN = 식별됨 (구체적 setup 차이 + 본 프로젝트 적용 plan)
  YELLOW = 후보 식별 (검증 필요)
  RED = code 차이 없음 → OS-level 차단 확정 (platform signature 등)

수정 금지 (source 분석만). 본 프로젝트 코드 변경 없음. 보고 길이 ~600~800 토큰.
```

**시간 cap**: Researcher 1d. Background 실행 가능.

**결과의 두 분기**:
- GREEN/YELLOW → 본 프로젝트 microbench 적용 후 1 bin 검증 → Q-2.1 raw path spec 진입 가능
- RED → Q-2 = Executorch wrap 결정 확정, 작업 2 진입 (qnn_oppkg → opencl 통합 의 의미 명확)

### 작업 2: qnn_oppkg → opencl 통합 (sub-sprint 분할)

**목적**: production 실효 가치 = rpcmem KV cache 의 zero-copy alias 만 보존, OpPackage 자체 제거. backend matrix 단순화 (5 → 4), 코드 정직성 회복 (이름이 misleading 이었던 문제 해소).

#### 작업 2a (메인 직접, 0.5~1d): libQnnGpu.so 의존 dry-run

확인 사항:
- `crates/qnn_oppkg/` 의 5926 LOC 중 rpcmem allocation 만 분리 가능한 부분
- `libQnnGpu.so` 가 실제로 dlopen 되는 path (rpcmem alloc 만 쓸 거면 필요한가?)
- `libcdsprpc.so` 만으로 rpcmem allocator 가 동작하는 microbench 작성 가능성
- `engine/src/backend/qnn_oppkg/{kv_buffer,hybrid_memory,memory}.rs` 의 rpcmem alloc lifecycle

검증 게이트:
- rpcmem allocator + `CL_MEM_USE_HOST_PTR` alias 만 사용하는 minimal microbench bin 작성 (libQnnGpu.so dlopen 없이)
- S25 에서 동작 확인 — `--backend qnn_oppkg` fast_path OFF 의 효과 (13 ms/tok zero-copy) 와 비교

#### 작업 2b (Senior Implementer, 1~2d): rpcmem allocator 추출

작업 2a 결과 기반:
- `crates/qnn_oppkg/` 에서 rpcmem 전용 모듈 추출 → 새 crate `crates/rpcmem/` 또는 `engine/src/memory/rpcmem.rs`
- QNN/OpPackage 의존성 분리

#### 작업 2c (메인 또는 Implementer, 1d): opencl backend 통합 + CLI flag

- `engine/src/backend/opencl/mod.rs` 에 rpcmem KV cache alloc path 추가
- CLI flag 신설: `--rpcmem-kv-cache` 또는 `--kv-zero-copy` (naming 은 spec 단계)
- 사용 예: `--backend opencl --rpcmem-kv-cache` 가 현 `--backend qnn_oppkg` fast_path OFF 와 등가

#### 작업 2d (Tester, 0.5d): S25 회귀 + qnn_oppkg crate 제거

- S25 Qwen2.5-1.5B Q4_0 Decode TBT 측정: `--backend opencl --rpcmem-kv-cache` 가 13 ms/tok 재현 확인
- 회귀 통과 시 `engine/src/backend/qnn_oppkg/` + `crates/qnn_oppkg/` 디렉토리 제거
- backend matrix 4 개로 축소

**Microbench 보존**: paper main evidence 에 해당하는 M3/M4 microbench bin 은 그대로 둠 (production code 제거 ≠ microbench 제거).

**Ripple**:
- `--backend qnn_oppkg` 사용 docs/scripts (스크립트 8개+ 추정) 갱신
- handoff/메모리의 qnn_oppkg 언급 정리

---

## Landmines / 미해결

### 1. 작업 1 의 RED 시나리오

Executorch 가 platform signature / qnn manifest / SELinux context 등 코드 외 요소로만 동작하면 raw path 부활 불가. 이 경우 Q-2 = Executorch wrap 으로 확정 → 본 프로젝트의 model loader (AUF/GGUF/Safetensors) 와 .pte (Executorch schema) 사이의 weight share 경로가 부재한 점이 추가 spec 필요. Llama 3.2 / Qwen 2.5 전체 모델의 .pte 변환 가능성도 미검증 (Phase D 는 단일 MatMul 만 검증).

### 2. 작업 2a 의 libQnnGpu.so 진짜 필요성

CLAUDE.md 메모리 명시: "qnn_oppkg backend = libQnnGpu.so + libqnn_oppkg.so + libcdsprpc.so 필요". libQnnGpu.so 가 rpcmem alloc 자체에 관여하는지, 아니면 OpPackage 그래프 dispatch (fast_path ON, 79 ms/tok dead code) 에만 관여하는지 검증 필요. 후자라면 libQnnGpu.so 의존 제거 가능 (통합 깔끔).

### 3. 두 작업의 순서 trade-off

- **1 → 2 순서**: 작업 1 RED 시 작업 2 진입 명분 강화. 작업 1 GREEN 시 backend matrix 가 (qnn_htp 추가 + qnn_oppkg 제거) 동시 진행 — 정합성 ↑.
- **병렬**: Researcher 위임 (작업 1) background + 메인 작업 2a dry-run 시작. 시간 효율 ↑, 단 작업 2a 가 작업 1 결과 의존 안 함 확인 필요 (확인됨 — libQnnGpu.so 분석은 작업 1 와 독립).

### 4. paper figure 영향

`--backend qnn_oppkg` 와 `--backend opencl` 비교가 paper 측 cross-backend chart 의 한 축이었다면 통합 후 단일 backend 가 됨. paper figure 재구성 필요. M4 (QNN-GPU OpPackage microbench) 는 paper main evidence 로 보존.

### 5. 본 세션의 device config code

`engine/microbench/htp_matmul_correctness.rs` 의 99 LOC (HTP custom config + deviceCreate, commit `878f0ae3` 에 포함) 는 dead path. 작업 1 GREEN 시 정상 path 로 cleanup, RED 시 명시적 revert 또는 진단 evidence 로 보존.

---

## 산출물 매핑 (commit/push 후)

| 항목 | 위치 |
|---|---|
| 본 handoff | `.agent/todos/handoff_next_session_executorch_qnn_consolidation_2026_05_26.md` |
| 직전 Q-2.0 종결 handoff | `.agent/todos/handoff_q20_dryrun_complete_2026_05_26.md` |
| Q-2.0 sweep raw | `papers/eurosys2027/_workspace/experiment/qnn_q20_dryrun_2026_05_26/` |
| device config diff (dead path) | `engine/microbench/htp_matmul_correctness.rs` (commit `878f0ae3`) |
| Executorch shallow clone | `/home/go/Workspace/executorch/` (외부 자산) |
| qnn_oppkg crate (분석 대상) | `crates/qnn_oppkg/` (5926 LOC), `engine/src/backend/qnn_oppkg/` (853 LOC mod.rs + 9 files) |

---

## 자기점검

- [x] 진입 문장 한 줄만으로 다음 세션 첫 명령 가능
- [x] 멈춘 이유 명시 (사용자 결정 by design, 다음 세션이 두 작업 진행)
- [x] 가장 큰 landmine 표면화 (5건: 작업 1 RED, libQnnGpu 의존, 순서, paper figure, device config dead path)
- [x] 검증 게이트 명시 (작업 1 verdict, 작업 2 각 단계의 PASS 조건, S25 13 ms/tok 재현)
- [x] 본문 길이 ~1000 토큰 (handoff-doc 권장 500 토큰 초과 — 두 작업의 sub-sprint 분할 + Researcher prompt 골격 포함을 위해 의도적 확장)
