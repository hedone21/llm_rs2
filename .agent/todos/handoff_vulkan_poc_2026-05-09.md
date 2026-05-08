# Handoff: Vulkan Compute PoC — Adreno HW vs OpenCL ICD Serialization 가설 검증

**작성일**: 2026-05-09
**기준 commit**: `60b95aa` (cleanup 완료, working tree clean, origin/master push 완료)
**Device**: Galaxy S25 (Snapdragon 8 Elite for Galaxy, Adreno 830)

---

## STATUS (2026-05-09 갱신): CLOSED — Phase 9 fast-feasibility 결과 RED

본 1주 PoC 계획 진입 전, 사용자 요청으로 3-question fast feasibility check (Phase 9, 1일 소요)를 먼저 수행했다.

**결과**:
- Q1 correctness ✓
- Q2 async swap **✗** (Vulkan cross-family 1.895x ≈ OpenCL 1.93x — HW level serialize 확정)
- Q3 throughput ✓ (Vulkan 0.979x parity, OpenCL ICD overhead negligible)

→ **본 1주 PoC 진입 취소**. Phase 9 측정만으로 paper Section 4 evidence 결정적 강화. 상세: `papers/eurosys2027/_workspace/experiment/swap_overhead_phase9_vulkan_feasibility.md`.

아래는 원본 1주 PoC 계획 (참고용 보존). 향후 Vulkan production migration 검토 시 재참조 가능.

---

---

## 0. 한 줄 요약

OpenCL 스택 안의 모든 가능성 검증 완료 (Phase 0-8, 16 트랙). 이제 **Vulkan transfer queue가 OpenCL ICD를 우회해 GPU compute와 진정한 병렬 실행이 가능한지** 직접 microbench로 측정. **1주 PoC, 코드 ~300 LOC, 측정 binary 1개, production 코드 0줄 변경**. PoC 성공/실패 어느 쪽이든 paper main evidence 강화.

---

## 1. 배경 — 왜 Vulkan PoC인가

### Phase 6에서 확정한 사실
Adreno 830 OpenCL은 HW level single-issue FIFO command processor:
- Same-ctx in-order × 2: **1.93x** (HW serialize)
- Same-ctx OoO × 2: **2.00x** (OoO hint 무시)
- Multi-context × 2: **2.55x** (context switch overhead 추가)

→ OpenCL ICD layer에서 진짜 우회 path 없음.

### 미확정 가설
**이 직렬화가 (a) OpenCL ICD에 한정된 SW 직렬화인가, (b) HW level 강제인가**?

기존 측정은 모두 OpenCL stack 안에서 수행. Vulkan은 다른 driver path:
- Adreno 7xx 이후 command processor가 PFP/ME/BV/BR 두 microcontroller로 분리 (chipsandcheese 분석)
- Vulkan은 graphics/compute/**transfer** queue family를 별도로 expose
- transfer queue가 별도 HW 큐(GMU 또는 secondary CP)에 매핑된다면 OpenCL과 다른 schedule 정책 가능

### PoC 결과 분기
| Vulkan transfer queue overlap | Paper 영향 |
|-------------------------------|-----------|
| O (parallel ~1ms) | 본 finding을 **OpenCL ICD에 한정**으로 좁힘 → contribution 강화 + Vulkan production migration 정당화 |
| X (serial ~2ms) | **HW level 직렬화 확정** → Phase 6 evidence 더 강력 |

어느 쪽이든 paper 가치 ↑. **부정 결과여도 publishable**.

---

## 2. 현재 코드 상태

### 정리 완료
- 32 commits pushed to origin/master (HEAD: `60b95aa`)
- Working tree clean
- All Phase 0-8 measurements documented in `papers/eurosys2027/_workspace/experiment/swap_overhead_phase*.md`
- Phase 4 production fix (HOST_WRITE_ONLY -35%) integrated
- Dead code (`--swap-pre-stage`, `DIAG/DEBUG` logs, hook trace instrumentation) 제거

### 검증된 baselines (Galaxy S25, Qwen2.5-1.5B Q4_0)
- GPU forward TBT: 27.86 ms/tok
- CPU forward TBT: 29.30 ms/tok (CPU/GPU = 1.05x)
- 600MB H2D wall-clock: 22.28ms (27.5 GB/s, 8% of 290ms swap stall)
- 290ms swap 분해: H2D 22ms + Q4 변환/ArcSwap/cl_mem alloc/page pinning 268ms

---

## 3. PoC 목표

**한 줄**: Vulkan compute kernel + Vulkan transfer queue가 동시 실행될 때 wall-clock이 single queue 1.0x인지 2.0x인지 측정.

### Success criteria
- σ/mean < 7% (n=30, deterministic config)
- Result definitive (1.0x vs 2.0x 명확 구분)

### Acceptance bands
- **0.95-1.10x**: parallel 확인 — paper Vulkan path 새 contribution
- **1.10-1.80x**: partial — driver mixed schedule
- **1.80-2.20x**: HW serialize 확정 — paper Phase 6 evidence 더 강력
- **>2.20x**: Vulkan에서도 추가 overhead — driver issue 별개 finding

---

## 4. 환경 셋업

### Galaxy S25 측 (이미 셋업됨)
- Vulkan loader: `/system/lib64/libvulkan.so` (이미 device에 있음)
- Adreno 830 Vulkan 1.3 driver는 production 일반 사용 가능 (모든 Android 게임이 사용)

### Host 측 신규 작업
1. **Vulkan crate 추가**: `engine/Cargo.toml`에 `ash = "0.38"` 또는 `vulkano = "0.34"` 추가
   - 권장: **`ash`** — 얇은 raw FFI wrapper, OpenCL의 `ocl::ffi`와 비슷한 추상화 레벨
   - vulkano는 무거움 (validation, queue family abstraction, etc.)
2. **NDK shaderc 또는 glslang**: GLSL → SPIR-V 컴파일
   - 가장 단순: 호스트에서 `glslangValidator` (Linux 패키지 또는 Vulkan SDK)로 SPIR-V 사전 컴파일 → 바이너리에 `include_bytes!` 임베드
   - 동적 컴파일 불필요 (PoC는 1개 커널만 사용)
3. **Cross-compile 확인**: `cargo build --release --target aarch64-linux-android --bin microbench_vulkan_two_queue`
   - `ash`는 dynamic loading으로 device의 `libvulkan.so` 사용 → NDK link 추가 작업 없음

### 가능한 build issue
- `ash`는 platform-specific extensions (Android surface)을 별도 feature로 갈라놓음. 본 PoC는 **headless compute only**라 surface extensions 불필요. `default-features = false` 권장.

---

## 5. 작업 단계 (1주 estimate)

### Day 1: 환경 셋업 + Hello Vulkan (4-6시간)
**목표**: Adreno 830에서 단순 compute kernel이 Vulkan으로 실행되는지 확인

1. `engine/Cargo.toml`에 ash dependency 추가
2. `engine/src/bin/microbench_vulkan_hello.rs` 신규
   - VkInstance 생성 (no validation layers in release)
   - VkPhysicalDevice 열거 + Adreno 830 선택
   - **`vkGetPhysicalDeviceQueueFamilyProperties` 출력** ⭐ — graphics/compute/transfer family 분리 확인
   - 단순 add kernel SPIR-V 로드, 실행, 결과 검증
3. cross-compile + adb push + 실행
4. **Critical output**: queue family list. transfer-only queue family가 별도로 나오는지 확인

### Day 2: Two-queue concurrent microbench (8시간)
**목표**: Phase 6 OpenCL과 동일 패턴을 Vulkan으로 구현

1. `engine/src/bin/microbench_vulkan_two_queue.rs` 신규
   - 동일한 busy compute kernel (~1ms target)
   - Configs (Phase 6와 1:1 대응):
     - Single compute queue (baseline)
     - 2× compute queue (same family)
     - Compute queue + transfer queue (다른 family) ← **핵심 측정**
2. Timeline semaphore (Vulkan 1.2+) 사용
   - `vkCmdCopyBuffer` (transfer queue) + `vkCmdDispatch` (compute queue) 동시 submit
   - 두 큐의 wait_semaphore_value로 wall-clock 측정
3. n=30 측정, σ/mean check

### Day 3: 결과 분석 + 문서화 (4시간)
1. `papers/eurosys2027/_workspace/experiment/swap_overhead_phase9_vulkan_serialize.md` 작성
2. Phase 6 OpenCL과 직접 비교 테이블
3. paper Section 4 update draft

### Day 4-5: contingency + extended measurement
- 만약 Day 1에서 transfer queue family 없으면:
  - graphics + compute pair 측정으로 fallback
  - "Adreno 830 Vulkan은 graphics/compute만 분리, transfer는 graphics와 통합"이 자체 finding
- 만약 Day 2 결과가 noisy하면 n 확대 + warmup 강화

---

## 6. 핵심 코드 스케치 (Day 2 microbench)

```rust
// engine/src/bin/microbench_vulkan_two_queue.rs
//! Two-queue concurrent kernel — Vulkan version of Phase 6 OpenCL bench.
//! Tests whether Adreno 830 HW serialization is OpenCL ICD-specific
//! or HW-level by routing two simultaneous workloads through Vulkan.

use ash::{vk, Entry, Instance};

fn main() -> anyhow::Result<()> {
    let entry = unsafe { Entry::load()? };  // dlopen libvulkan.so
    let instance = create_instance(&entry)?;  // headless
    let phys = pick_adreno_device(&instance)?;
    let queue_families = unsafe {
        instance.get_physical_device_queue_family_properties(phys)
    };

    // Print queue families — critical observation
    for (i, qf) in queue_families.iter().enumerate() {
        eprintln!("Queue family {}: count={}, flags={:?}",
            i, qf.queue_count, qf.queue_flags);
    }
    // Expected on Adreno 830:
    //   family 0: GRAPHICS | COMPUTE | TRANSFER (1 queue)
    //   family 1: COMPUTE | TRANSFER (1 queue)  ← if exists
    //   family 2: TRANSFER (1 queue)            ← if exists ← KEY
    //
    // If transfer-only family exists, route H2D copy there + compute
    // dispatch on graphics family. Compare wall-clock to Phase 6 OpenCL.

    // ... device creation, queue retrieval, SPIR-V kernel load,
    //     timeline semaphore setup, dispatch, finish, measure ...
}
```

### Pre-built SPIR-V (busy.spv)
```glsl
#version 450
layout(local_size_x = 1024) in;
layout(std430, binding = 0) buffer Out { float data[]; };
layout(push_constant) uniform PC { int iters; } pc;
void main() {
    uint id = gl_GlobalInvocationID.x;
    float v = float(id);
    for (int i = 0; i < pc.iters; i++) {
        v = v * 1.00001 + 0.5;
        v -= 0.5;
    }
    data[id] = v;
}
```

호스트에서 컴파일:
```bash
glslangValidator -V busy.comp -o busy.spv
# 또는 Vulkan SDK의 glslc:
glslc --target-env=vulkan1.2 busy.comp -o busy.spv
```

바이너리에 임베드:
```rust
const BUSY_SPV: &[u8] = include_bytes!("../../shaders/busy.spv");
```

---

## 7. 측정 절차 (Day 2)

```bash
# 호스트
cargo build --release --target aarch64-linux-android --bin microbench_vulkan_two_queue
adb push target/aarch64-linux-android/release/microbench_vulkan_two_queue /data/local/tmp/

# 디바이스 (LD_LIBRARY_PATH는 Vulkan 자동 로드, OpenCL과 다름)
adb shell /data/local/tmp/microbench_vulkan_two_queue 30
# 또는
python scripts/run_device.py -d galaxy_s25 microbench_vulkan_two_queue 30
```

**측정 5 cell** (Phase 6와 1:1 매칭):
1. Single compute queue baseline (1.00x)
2. Same family compute × 2 (Vulkan equivalent of OpenCL same-ctx in-order × 2)
3. Different family (compute + transfer) ⭐
4. Timeline semaphore concurrent submit
5. With validation layers ON (instrumentation 영향 측정)

---

## 8. 위험 / 알려진 이슈

### Vulkan-specific
- **Adreno 830 transfer queue family 부재 가능성**: Adreno 730 forum에서 "compute and graphics queue only" 보고가 있음. 830에서도 동일하면 PoC scope를 graphics/compute pair로 변경
- **timeline semaphore 미지원 driver**: Vulkan 1.2 feature. Adreno 830 production driver는 1.3 지원하므로 OK 예상, 단 binary semaphore fallback도 준비
- **Validation layer 부재**: production driver는 release-only. 디버깅 어려움 → Day 1에 모든 alloc/free path 검증 철저

### 본 프로젝트 specific
- 본 PoC는 **microbench only**, 추론 path와 분리됨 → production 회귀 위험 zero
- Cargo `ash` dep 추가는 default-features = false면 OpenCL/CUDA path와 직교
- shader 컴파일 의존성: `glslangValidator` 호스트 설치 필요. Linux 패키지명: `glslang-tools` (apt) / `glslang` (arch)

---

## 9. Out of Scope

다음은 본 PoC 범위 밖:
- Vulkan compute backend production 이식 (8-12주, 별도 paper)
- OpenCL ↔ Vulkan buffer interop (`cl_khr_external_memory` + `VK_KHR_external_memory_fd`)
- VkQueue priority hints
- Adreno KGSL ioctl 직접 호출 (보안 정책 차단됨)

---

## 10. 다음 세션 시작 절차

```bash
cd /home/go/Workspace/llm_rs2

# 0. 상태 확인
git log --oneline -5
# HEAD = 60b95aa (cleanup 완료)
git status  # clean

# 1. handoff 읽기
cat .agent/todos/handoff_vulkan_poc_2026-05-09.md

# 2. 환경 점검
which glslangValidator || sudo pacman -S glslang  # Arch
adb devices  # R3CY408S4HN

# 3. Day 1: ash dep + hello vulkan
# Day 2: two-queue microbench
# Day 3: 결과 분석 + paper section
```

### 우선순위
1. **Day 1 가장 중요**: queue family enumeration 결과만으로도 paper 가치 있음
2. transfer-only family 존재 여부에 따라 Day 2 plan 갈림
3. failure mode (`ash` cross-compile, `vkGetInstanceProcAddr` 실패 등) 가능, contingency 필요

---

## 11. Success / Failure 종결 기준

### Success
- microbench_vulkan_two_queue 5 configs 측정 완료
- Phase 6 OpenCL 표와 1:1 비교 가능
- σ/mean < 7%
- swap_overhead_phase9_vulkan_serialize.md 작성 완료
- paper Section 4 update draft

### Acceptable failure
- transfer-only queue family 부재 → "Adreno 830 Vulkan은 transfer queue 분리 안 함" 자체 finding
- 모든 queue가 1.93x → "HW level 직렬화 확정" Phase 6 강화
- driver issue로 측정 불가 → 별도 documenting (PoC scope 제한)

### Hard fail (rare)
- `ash` cross-compile 불가능 → 우선 vulkano로 fallback
- libvulkan.so 로드 실패 → device-specific issue, 별도 path 필요

---

## 12. 메모리 file 갱신 권장

다음 세션 자동 인지를 위해:

```
경로: /home/go/.claude/projects/-home-go-Workspace-llm-rs2/memory/
신규 파일: project_vulkan_poc_handoff_20260509.md (이 handoff 요약)
MEMORY.md에 라인 추가:
- [Vulkan PoC handoff 2026-05-09](project_vulkan_poc_handoff_20260509.md) — 1주 PoC, transfer queue family 검증 → HW vs ICD 직렬화 가설 확정
```

(실제 메모리 파일은 다음 세션에서 검증 결과와 함께 작성 권장)

---

**End of Handoff**

이 문서는 self-contained — 새 세션에서 본 문서만 읽으면 Vulkan PoC를 즉시 시작 가능.
