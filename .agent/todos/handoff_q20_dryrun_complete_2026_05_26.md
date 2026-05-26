# Handoff: Q-2.0 dry-run 종결 — raw HTP API path 차단 확정

**작성**: 2026-05-26
**HEAD (commit 직전)**: `56f059c5 exp(microbench): μ-Q1 sprint 종결 — Phase D production + Phase E report + handoff`
**브랜치**: `worktree-b5_trait_extension`
**다음 세션 진입 문장**: "Q-2 재설계 결정 진행"

---

## TL;DR

Q-2 (HTP backend trait) 의 dry-run 게이트 = **RED**. raw QNN HTP API path 가 stock S25 에서 fastrpc Skel publish OS 정책으로 차단됨. device custom config (`kHtpUnsignedPd` + `arch=V79`) 보강 후에도 fail point 가 contextCreate → deviceCreate 로 이동만 하고 err code 0x36b1 동일. **Executorch path 는 Phase E 에서 paper-grade GREEN 검증됨** (M7 W8A8 0.276 ms) — 같은 디바이스, 같은 lib, 같은 Skel.so 위에서 동작. mechanism 차이는 root 없는 환경에서 추적 불가. 다음 세션이 Q-2 재설계 방향 결정.

---

## 진행 상태

| Task | 상태 | 위치 |
|---|---|---|
| Q-2.0a host qnn feature 빌드 sanity | ✅ PASS | `cargo build --release --features qnn` warning only |
| Q-2.0b 9 bin Android arm64 cross-compile | ✅ PASS | `target/aarch64-linux-android/release/microbench_htp_*` 9건 460~680 KB |
| Q-2.0c S25 9 bin verify sweep | ✅ DONE | 0 PASS / 5 FAIL / 4 SEGFAULT (모두 contextCreate err=0x36b1) |
| Q-2.0d device config fix 1 bin 검증 | ✅ DONE | deviceCreate err=0x36b1 — fail point 만 이동, root cause 해소 안 됨 |

### Tester sweep 결과 (9 bin)

```
1 microbench_htp_correctness              139 SEGFAULT  contextCreate err=0x36b1
2 microbench_htp_gpu_matmul_concurrent      1 FAIL      contextCreate err=0x36b1
3 microbench_htp_gpu_parallel               1 FAIL      contextCreate err=0x36b1
4 microbench_htp_graph_reuse              139 SEGFAULT  contextCreate err=0x36b1
5 microbench_htp_matmul_correctness       139 SEGFAULT  contextCreate err=0x36b1
6 microbench_htp_opencl_interop             1 FAIL      contextCreate err=0x36b1
7 microbench_htp_qnngpu_share               1 FAIL      contextCreate err=0x36b1
8 microbench_htp_rpcmem_throughput          1 FAIL      contextCreate err=0x36b1
9 microbench_htp_throughput               139 SEGFAULT  contextCreate err=0x36b1
```

SEGFAULT 4건 = contextCreate fail 후 libQnnHtp destructor 의 partially-initialized state cleanup race (host-side bug, root cause 자체와 별개).

### Device config fix 시도 결과

`engine/microbench/htp_matmul_correctness.rs` 99 LOC 추가 (QnnHtpDevice_CustomConfig_t reverse-engineering + deviceCreate + ARCH=V79 + SIGNEDPD config):

- Run 1 (unsigned PD): `deviceCreate err=0x36b1` + fastrpc `domain_deinit for domain 3: dev 6` (cDSP domain bind ✓, Skel publish fail)
- Run 2 (signed PD): `deviceCreate err=0x36b1` + `domain 3: dev -1` (domain bind 자체 fail)

→ unsigned PD 가 fastrpc 측에서 더 진전 (Executorch 와 동일 path) — 그럼에도 DSP-side `AEE_ENOSUCHMOD (0x80000406)` 로 Skel publish fail.

### 핵심 logcat (대표)

```
QnnDsp <E> DspTransport.openSession qnn_open failed, 0x80000406, prio 100
QnnDsp <E> IDspTransport: Unable to load lib 0x80000406
QnnDsp <E> Failed to load skel, error: 1002
QnnDsp <E> Transport layer setup failed: 14001
QnnDsp <E> default device creation failed
QnnDsp <E> Failed to create context with err 0x36b1 (= 14001)
```

---

## 다음 액션 (다음 세션이 결정)

다음 세션 진입 문장: **"Q-2 재설계 결정 진행"**

3 옵션 중 사용자 결정:

| 옵션 | scope | 효과 |
|---|---|---|
| **Q-2 Executorch wrap 재설계** | `engine/src/backend/qnn_htp/` Backend trait 이 .pte runtime call wrap. PyTorch → .pte AOT 변환 spec + model loader 연결. 예상 3~5d. | M7 W8A8 0.276 ms paper-grade 결과 그대로 production 으로 통합. native HTP path 확보 |
| **Q-2 abandon** | HTP path 통합 없음. Phase E 수치는 paper baseline 으로만 사용. backend matrix 는 cuda_embedded/cuda_pc/opencl/qnn_oppkg 4개 유지 | paper 측 비교 base 는 외부 측정 (Phase E) 으로 충분 |
| **Q-2 보류, 다른 sprint** | INV-LAYER A+B (`/home/go/.claude/plans/greedy-snacking-oasis.md`, baseline 176→163) 또는 다른 backlog 으로 전환 | HTP 통합은 root 환경 / paper deadline 도래 시 재개 |

### 권장 (메인 의견)

**Executorch wrap 재설계** — paper main path 가 이미 paper-grade GREEN 으로 확정되어 있고, Backend trait 도입의 productive value 가 가장 높음. 다만 PyTorch model → .pte AOT 변환 단계가 model loader pipeline 과 어떻게 결합되는지 (Llama 3.2/Qwen 2.5 정식 모델 lower-to-edge 가능성) Architect spec 단계에서 명확히 해야 함.

---

## Landmines / 미해결

### 1. raw QNN HTP API path 가 차단된 정확한 mechanism

- Executorch (`qnn_executor_runner`) 가 동일 stock S25 + 동일 lib + 동일 Skel.so 로 작동
- 본 프로젝트 raw bin 은 device config (kHtpUnsignedPd, ARCH=V79) 완비 후에도 fail
- 차이 후보: (a) Executorch process 가 platform signature/manifest 보유, (b) farf manifest 또는 `.farf` 파일 (`echo 0x0C > <runner>.farf` 의 Executorch 패턴), (c) Executorch 의 추가 init path (예: HtpBackendCustomConfig 의 estimated_inference_time 또는 weight_sharing), (d) SELinux/permission 차이
- root 권한 / dmesg 접근 / `ps -Z` SELinux context 확인이 가능하면 정확한 추적 가능. **현재 stock S25 환경에서는 추적 불가**

### 2. Executorch wrap path 가 production 모델과 결합 가능한지

- PyTorch → .pte AOT 변환은 단일 MatMul (Phase D) 에서는 검증됨
- Llama 3.2 1B / Qwen 2.5 1.5B 전체 모델을 .pte 로 변환 가능한지 검증 안 됨 — Executorch 의 lowering 제약 (Q4_0 weight 미지원, dynamic shape, custom op) 등이 막을 가능성
- 본 프로젝트의 AUF/GGUF loader 와 .pte (Executorch 자체 schema) 사이의 weight share 경로 부재 — model loader 와 spec 재설계 필요

### 3. device config fix 코드의 사용처 부재

- `engine/microbench/htp_matmul_correctness.rs` 의 99 LOC 추가는 dead path (PASS 안 됨)
- 가치: 향후 root device 또는 다른 backend (HTP V79 외) 에서 같은 API 검증 시 참고
- 다음 세션이 Executorch wrap 결정하면 별도 sprint 에서 cleanup 가능

### 4. μ-Q1 phase_d_prod 측정 raw 가 commit 누락됨

- `papers/.../qnn_microbench_phase_d_prod/` (production v1, μ-Q1 commit 이전 untracked)
- 본 commit 에 함께 묶어 정리 (Phase D production 측정 raw 보존)

---

## 산출물 매핑

| 항목 | 위치 |
|---|---|
| Tester 9 bin sweep 보고 | `papers/eurosys2027/_workspace/experiment/qnn_q20_dryrun_2026_05_26/report.md` |
| Per-bin logs + logcat | `papers/.../qnn_q20_dryrun_2026_05_26/logs/` |
| Device config fix diff | `engine/microbench/htp_matmul_correctness.rs` (99 LOC 추가, dead path) |
| Phase D production v1 측정 (μ-Q1 잔여) | `papers/.../qnn_microbench_phase_d_prod/` |
| third_party SDK symlink | worktree-local (gitignored) |

---

## 자기점검

- [x] 진입 문장 한 줄만으로 다음 세션 첫 명령 가능 ("Q-2 재설계 결정 진행")
- [x] 멈춘 이유 명시: raw path OS 정책 차단 + 결정 분기 사용자 위임
- [x] 가장 큰 landmine 표면화: 4건 (mechanism 차이 미추적 / Executorch wrap 모델 결합 검증 부재 / device config code dead path / phase_d_prod 누락)
- [x] 검증 게이트: Tester 9 bin 표 + senior-implementer fix run 1/2 logcat raw 인용
- [x] 본문 길이 ~800 토큰 (handoff-doc 권장 500 토큰 초과 — landmine 추적 가치 + 다음 세션 결정 보조 위해 의도적 확장)
