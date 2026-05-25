# Plan: μ-Q1 — QNN HTP matmul microbench-first PoC

**작성**: 2026-05-26
**브랜치**: `worktree-b5_trait_extension`
**진입 문장**: "μ-Q1 진행" 또는 "HTP microbench PoC 진행"
**Scope cap**: 2~3 영업일

---

## 목표

S25 Hexagon V79 HTP 에서 single matmul op (production-shape) 을 microbench
로 실행, **CPU reference 대비 numerical match** (atol ≤ 0.1, cosine ≥ 0.99) +
**latency 측정** (OpenCL Q4_0 대비). microbench 결과 GREEN 일 때만 후속
sprint Q-2 (Backend trait `engine/src/backend/qnn_htp/` 신설) 진입.

**비목표 (μ-Q1 에서 명시적 제외)**:
- `engine/src/backend/qnn_htp/` 디렉토리 신설 — μ-Q1.2 GREEN 후 후속 Q-2
- 14-node single-layer (M2 HTP 버전) — Q-2
- KV cache layout — Q-3
- Architect spec/arch 문서 — μ-Q1.2 GREEN 후 Q-2 진입 시
- `crates/qnn_oppkg_poc` 정리

---

## 자산 활용

| 기존 자산 | μ-Q1 에서 활용 |
|---|---|
| `engine/microbench/htp_matmul_correctness.rs` 등 9 HTP bin | μ-Q1.0 verify sweep |
| `crates/qnn_oppkg` (5926 LOC, rpcmem allocator) | rpcmem zero-copy alloc + `QnnMem_register` 재사용 |
| `engine/build.rs` bindgen + `qnn` feature | 그대로 |
| `crates/qnn_oppkg_poc` (810 LOC) | **deprecated 후보** — μ-Q1 에서는 건드리지 않음 |

---

## Phase 분할

### μ-Q1.0 — 기존 자산 verify (0.5일)

| Step | 작업 | 검증 |
|---|---|---|
| 1 | SDK 위치 확정 (`third_party/qnn_sdk_2.33/` 부재 → 외부 link 또는 env var) | `cargo build --release --bin microbench_htp_matmul_correctness --features qnn` PASS |
| 2 | S25 에 `libQnnHtp.so` + `libQnnHtpV79Stub.so` + `libQnnHtpV79Skel.so` push | adb shell `ls /data/local/tmp/qnn/` 3개 파일 확인 |
| 3 | 기존 9 HTP microbench 빌드/실행 sweep | 각 bin 결과 표 (segfault / PASS / FAIL) |
| 4 | Memory 의 segfault risk (Phase R 시점 2026-05-09) 재현 여부 확정 | YES 면 risk 분석, NO 면 다음 phase |

**μ-Q1.0 결과만 보고 abandon 결정 가능**:
9개 microbench 가 전부 segfault 면 SDK/디바이스 환경 본질, PoC 보류.

### μ-Q1.1 — production-shape microbench 추가 (1일)

- 기존 `htp_matmul_correctness.rs` shape 점검 (Researcher §4-5 = small matmul 권장).
- toy shape 만 있으면 신규 bin 추가:
  - **신규**: `engine/microbench/htp_matmul_qwen_ffn.rs`
  - Shape: Qwen 2.5-1.5b 의 `[1, 1536] × [1536, 8960]` (FFN gate/up hot path)
  - Quantization: **W8A8 1차** (Executorch tolerance 미확보 → atol 0.1 + cosine ≥ 0.99)
  - VTCM budget tuning (`vtcm_size_in_mb` 8/16/32 sweep)
- vs OpenCL Q4_0 matmul latency 비교 (S25 production hot path)

### μ-Q1.2 — 결정 게이트 (0.5일)

| 결과 (HTP / OpenCL Q4_0 latency) | 다음 트랙 |
|---|---|
| < 50% (2× 이상 가속) | Q-2 진입 — Backend trait + 14-node single-layer |
| 50~100% | YELLOW — heterogeneous (Phase 10 Q2 재활용, HTP+GPU 동시) 트랙 |
| ≥ 100% | RED — abandon, microbench 만 보존 |

---

## Landmines (Researcher §4)

1. **Context binary 호환성 (Error 30010)** — SDK 2.33 + V79 magic 검증 binary header 기록
2. **`libQnnHtpV79{Stub,Skel}.so` push 누락** — μ-Q1.0 step 2 검증
3. **VTCM 초과 silent perf cliff** — μ-Q1.1 vtcm sweep
4. **mem_handle 등록 비용 graph 외부** — 측정 시 `graph_execute` 만 분리, 워밍업/측정 분리
5. **soc_model SM8750 (v79)** — μ-Q1.0 verify (Researcher 불확실 항목)
6. **Simple op = 53 ms overhead** (Executorch Issue #3949) — rpcmem alloc/register 합산 비용

---

## 산출물

- μ-Q1.0: 9 microbench verify 결과 표 + 1 commit (SDK link/env 조정 필요시)
- μ-Q1.1: 신규 `htp_matmul_qwen_ffn.rs` bin + 1 commit
- μ-Q1.2: 결정 보고서 `papers/eurosys2027/_workspace/experiment/htp_matmul_microbench_2026_05_26.md` + handoff

---

## 진행 권장 순서

PM → Implementer (μ-Q1.0~μ-Q1.1) → Tester (μ-Q1.2 디바이스 게이트)

Architect spec/arch 단계는 μ-Q1.2 GREEN 후 Q-2 진입 시.

---

## 다음 액션 (μ-Q1.0 step 1)

1. SDK 위치 점검 — `find / -name "QnnHtp.h" 2>/dev/null` + `~/.qnn-sdk*` 확인
2. 없으면 main repo (`/home/go/Workspace/llm_rs2/`) 확인 후 worktree symlink
3. `qnn` feature 빌드 trial: `cargo build --release --bin microbench_htp_matmul_correctness --features qnn` (host)

---

## 관련 자료

- Executorch HTP backend 조사: 본 세션 Researcher 보고 (Phase R Phase 32b 인사이트 합본)
- Memory: [[project_liswap5_phase10_htp_feasibility_20260509]], [[project_qnn_oppkg_phase_r_complete_20260509]]
- 기존 9 HTP microbench: `engine/microbench/htp_*.rs`
