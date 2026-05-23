# Handoff: B-5b Phase 2 종결 → Phase 3 진입 다음 세션

**작성**: 2026-05-23
**HEAD**: `28bd7724 refactor(backend): B-5b Phase 2 Stage 2-B — yield_after_layer 단독 치환 (hot-path)`
**브랜치/Worktree**: `worktree-b5_trait_extension` (`.claude/worktrees/b5_trait_extension`)
**다음 세션 진입 문장**: `"B-5b Phase 3 진행 (hybrid_attention 2건 §J 확장 + as_opencl_secondary 치환)"`
**선행 진입점**: [[handoff_b5b_phase2_stage1_complete_2026_05_23]]

---

## TL;DR

B-5b Phase 2 (Backend trait 4 capability method 인프라 + 호출지 치환) 전체 완료. Stage 1 (인프라) + Stage 2-A (cpu_kernels 8건 + cpu_companion 25건) + Stage 2-B (yield_after_layer 단독 hot-path) 3 commit. S25 microbench 게이트 양쪽 모두 PASS (Δ avg_tbt = −0.231% / −0.018%). **R-1 vtable 우려 실측 0** — LTO=fat가 `Arc<dyn Backend>` dispatch cost 완전 흡수 확인. **멈춘 이유**: Phase 2 완료, Phase 3 (hybrid_attention 2건 §J 확장 + Stage 2에서 이연된 as_opencl_secondary 치환)로 자연스러운 경계. 컨텍스트 절약.

**중요 발견 (Phase 2 전반)**:
1. `cpu_companion`은 trait default impl 불가 (Rust object-safety, `&Self → &dyn Backend` coerce 안 됨). 모든 5 backend explicit override
2. **R-1 RPN 144~180 우려는 실측에서 미실현** — Stage 2-A Δ −0.231%, Stage 2-B Δ −0.018% (둘 다 noise 이하). 향후 capability hook 추가 시 +3% 게이트 유지 권장
3. layer_lint baseline은 **architecture invariant 강도 ≠ 절대 카운트** — Stage 2-B에서 자유함수 dispatch → trait override로 inversion 시 +2 발생. 새 violation 0이지만 4 backend가 helper import → +4. 본질은 **개선** (INV-LAYER-006 source-grep 친화)

---

## 진행 상태

### 본 sprint commits (3건)

| HEAD | scope | 변경 |
|---|---|---|
| `eb1970dc` | feat(backend) | **B-5b Phase 2 Stage 1** — Backend trait 4 capability hooks + CpuKernelSet + SecondaryStore + OpenClSecondary (placeholder) + 5 backend struct/impl + lib.rs 등록 (+318 LOC, baseline 변동 0) |
| `6cd09f9b` | refactor(backend) | **B-5b Phase 2 Stage 2-A** — cpu_kernels (8건) + cpu_companion (25건) 호출지 치환. cuda_pc + cuda_embedded 자유함수 `cpu_fallback*` dead code 제거. R-3 결정 적용 (tag counter `cpu_fallback_log(tag)` 분해) (+83 / -45) |
| `28bd7724` | refactor(backend) | **B-5b Phase 2 Stage 2-B** — yield_after_layer 자유함수 → trait method (hot-path, ~4000 호출/inference). `gpu_yield::maybe_yield_after_layer` → `gpu_yield_impl` rename. 4 GPU backend (OpenCL/CudaPc/CudaEmbedded/QnnOppkg) 1줄 override (+86 / -38) |

### S25 microbench 게이트 결과 (Galaxy S25, 6T, Qwen 2.5 1.5B Q4_0)

| 단계 | HEAD | avg_tbt (ms ± stddev) | Δ vs baseline | 판정 |
|---|---|---|---|---|
| baseline (Stage 1 종결) | `b4193bab` | 32.958 ± 0.052 | — | — |
| Stage 2-A | `6cd09f9b` | 32.882 ± 0.049 | **−0.231%** | PASS |
| Stage 2-B (yield disabled) | `28bd7724` | 32.876 ± 0.083 | **−0.018%** vs Stage 2-A | PASS |
| Stage 2-B (yield enabled, `EVERY=4 US=500`) | `28bd7724` | 40.376 ± 0.094 | +22.8% (sync+sleep 비용, vtable 아님) | 게이트 외 |

- N=5 per HEAD, 15 runs 출력 bit-identical
- 로그 클린 (`evict|dispatcher|pending|panic|abort|error|WARN` grep 0건)
- raw 데이터:
  - `papers/eurosys2027/_workspace/experiment/b5b_phase2_stage2a_s25_gate_2026_05_23/`
  - `papers/eurosys2027/_workspace/experiment/b5b_phase2_stage2b_s25_gate_2026_05_23/`

### 호스트 게이트 결과 (3개 Stage 모두 PASS)

| Gate | Stage 1 | Stage 2-A | Stage 2-B |
|---|---|---|---|
| `cargo build` | PASS | PASS | PASS |
| `cargo fmt` | clean | clean | clean |
| `cargo clippy -D warnings` | clean 0 | clean 0 | clean 0 |
| `cargo test --lib` | 회귀 0 | 회귀 0 | 회귀 0 |
| `cargo test --test spec inv_layer` | 8 PASS | 8 PASS | 8 PASS |
| `layer_lint --baseline` | 새 violation 0 | 새 violation 0 | 새 violation 0 |

### baseline 변동 누적

| 단계 | baseline (절대) | 메모 |
|---|---|---|
| 진입 (Stage 1 종결) | 209 | master `b4193bab` 시점 |
| Stage 2-A `6cd09f9b` | 196 | -13 (L001 -7, L003 -6) |
| Stage 2-B `28bd7724` | **208** | +12 (자유함수 dispatch → 4 backend trait override inversion). 새 violation 0건 |

**baseline 절대 카운트의 한계**: layer_lint은 fully-qualified path 사용 빈도를 셈. trait method override가 helper를 호출하면 import 횟수 = backend 수만큼 늘어남. 본 sprint의 진짜 ROI는 (1) **자유함수 dispatch 25+ 건 제거** (2) **downcast 0건 유지** (3) **trait method 경유로 architecture invariant 강화** — 절대 baseline은 +/− 0건 수준 (Phase 2 시작 시 209, 종결 시 208).

---

## 다음 작업 — Phase 3 (task #54)

### 범위 (3 subgroup)

**A. hybrid_attention 2건 §J 확장**

`models/transformer.rs` 또는 layer 측의 `hybrid_attention` 관련 cross-layer 호출 2건. 자세한 호출지는 `arch/layered_architecture.md §13.8-J` 확장 정책 따라 결정. Architect 1회 라운드 권장.

**B. as_opencl_secondary 치환 (Stage 2-A에서 이연)**

- `engine/src/backend/qnn_oppkg/mod.rs:132~142` — `with_opencl_secondary` 클로저의 `as_any().downcast_ref::<OpenCLBackend>()` 제거
- R-2 결정 보류 상태: `OpenClSecondary` trait body 설계 옵션 α/β/γ 중 택1
  - (α) `fn with_queue<R>(&self, f: &dyn Fn(&Queue) -> R)` — return value 처리 위해 외부 mutable slot 캡처 필요
  - (β) `fn queue_handle(&self) -> &Queue` — `Queue` 타입 OpenCL feature-gated, lifetime 복잡
  - (γ) qnn_oppkg가 `Arc<OpenCLBackend>` field 직접 보유 — downcast 제거되지만 trait 가치 ↓ + INV-LAYER-001 새 형태 도입
- **Backend trait 분할 정책 재논의 시점**: 본 Phase 3 진입 직전 사용자 결정 필요 (memory: "현재 수준으로 R-2, R-3 정리, 추후에 Backend 트레잇 분할 정책 재 논의")

**C. 잔여 cleanup**

- `engine/src/cpu_kernels.rs` + `engine/src/secondary.rs` 신규 모듈 2개 검토 (`SecondaryStore` placeholder가 Phase 3에서 실체화 안 되면 처분 결정)
- `intra_token_yield_enabled()` workspace 내 caller 0건 (Stage 2-B 보고). dead code 처분 별도 sprint 가능
- `gpu_yield_impl` 자유함수가 4 backend override 본문에서 import — Phase 3에서 본문 4번 중복 흡수 (default impl로) 재고 여부

### Phase 3 게이트

1. 호스트 게이트 (Stage 1/2와 동일 7개)
2. layer_lint baseline 새 violation 0 (절대값은 hybrid_attention 분해 + as_opencl_secondary 결과에 따라 변동)
3. S25 microbench — A/B 둘 다 hot-path 직접 영향 적음, 게이트 microbench 생략 가능 (Stage 2-A처럼 LOC 적은 단계 패턴). 다만 변경량 큰 경우 +3% 게이트 1회 측정

### Phase 3 위임 prompt 초안

```
## 본 작업 = B-5b Phase 3 (hybrid_attention §J 확장 + as_opencl_secondary + 잔여 cleanup)

### 핵심 컨텍스트
- 선행: handoff_b5b_phase2_complete_2026_05_23.md (HEAD 28bd7724)
- Phase 2 종결 — Backend trait 4 capability hook 인프라 + 호출지 33+2건 모두 치환 완료
- Phase 3 = hybrid_attention 2건 + as_opencl_secondary 1 메서드 + cleanup

### 작업
1. hybrid_attention 2건 §J 확장 (architect 1회 라운드 권장)
2. as_opencl_secondary 치환 (R-2 결정 — OpenClSecondary trait body 설계 필요)
3. cleanup: SecondaryStore placeholder 처분 + intra_token_yield_enabled() dead code 처분

### 게이트
1. 호스트 게이트 7개
2. layer_lint baseline 새 violation 0
3. S25 microbench (선택) Δ ≤ +3%

### 미해결 결정 (Phase 3 진입 직전)
- Backend trait 분할 정책 (사용자: Phase 2 종결 후 재논의 명시)
- OpenClSecondary trait body 설계 (α/β/γ 택1)
- hybrid_attention §J 확장 범위 (architect 의견)
```

### Phase 4 (task #55) — Phase 3 후속

- 누적 handoff 통합 (Phase 1~3)
- master FF 정리 (사용자 master worktree 작업 완료 시)
- 잔여 spec/arch 문서 갱신

---

## Landmines / 미해결

### Phase 2 진행 중 발견

1. **`cpu_companion` default impl 불가** (Stage 1 발견): Rust object-safety 제약. `fn cpu_companion(&self) -> &dyn Backend { self }` 본문 컴파일 안 됨. 모든 backend explicit override 강제. 동작상 등가, ISP 누적 +0.

2. **layer_lint 절대 카운트는 architecture invariant 강도와 1:1 비례 안 함**: Stage 2-B에서 자유함수 → trait override inversion으로 +2 발생 (4 backend가 helper import). 새 violation 0 + dispatch는 trait method 경유로 더 깔끔. **baseline 절대 변동에 과민하지 말 것** — 새 violation 0 + 호출지 잔재 0이 진짜 게이트.

3. **`Arc<dyn Backend>` LTO=fat에서도 vtable cost 측정 불가**: Phase 2 시작 시 R-1 RPN 145~180 우려가 있었지만 Stage 2-A (낮은 빈도) + Stage 2-B (~4000회/inference hot-path) 둘 다 Δ < 0.5%. **향후 capability hook 추가 시 +3% 게이트 유지** 권장 — worst case 발생 시점이 model/backend 조합 변경 시점에 가까울 수 있음.

4. **`OpenClSecondary` trait body placeholder 잔류**: Stage 1에서 empty trait body로 들임, Phase 3에서 실체화 필요. R-2 결정 보류 상태. Backend trait 분할 정책 재논의 시점과 동기화.

5. **handoff baseline 14.66 ms/tok는 다른 컨텍스트 절대값**: Stage 2-A 측정에서 tester가 발견 — `project_weight_swap_tbt_gap_root_cause` Sprint A~F 종결 시점의 Mixed 모드 수치. 본 capability migration sprint baseline = 32.9 ms/tok (호스트 OpenCL stub, 32 tokens, 6T). 미래 sprint에서 baseline 인용 시 측정 컨텍스트 확인 필수.

### 사전 회귀 (Phase 2와 무관, B-2 handoff 이후 누적)

- `cargo build --features cuda` / `--features cuda-embedded`: `swap_dispatch.rs:441 map_weights_for_cpu` 사전 회귀
- `crates/qnn_oppkg/src/interface.rs` unresolved imports 17건+ (commit d930801a 이후)
- `cargo build --no-default-features` 사전 회귀
- 호스트 NVIDIA OpenCL `gpu_buffer_shift` 2건 flaky (호스트 GPU stub 환경 의존, S25 device test에선 무관)

### Phase 3 진입 직전 결정 사항 (3건)

1. **Backend trait 분할 정책** — 사용자 명시 보류 항목. trait 비대화 우려 시점 도달 여부 판단
2. **`OpenClSecondary` trait body 설계** — α (with_queue closure) vs β (queue_handle direct) vs γ (qnn_oppkg가 Arc<OpenCLBackend> field 보유) 택1
3. **hybrid_attention §J 확장 범위** — `arch/layered_architecture.md §13.8-J` 확장 정책 (architect 1회 라운드 권장)

---

## 자기점검 (handoff-doc 스킬)

- [x] 진입 문장 한 줄로 다음 세션 첫 명령 가능: `"B-5b Phase 3 진행"`
- [x] 멈춘 이유 명시: Phase 2 완료, Phase 3 (hybrid_attention + 잔여) 진입 직전 architect/사용자 결정 3건 대기
- [x] Landmines 표면화: 5건 (default impl 제약 + baseline 카운트 한계 + vtable cost 실측 0 + OpenClSecondary placeholder + baseline 컨텍스트 혼동)
- [x] 검증 게이트 수치: Stage 2-A Δ −0.231%, Stage 2-B Δ −0.018% (둘 다 +3% 게이트 PASS), 호스트 게이트 7개 × 3 stage 모두 PASS
- [x] 본문 적정 길이 (외부 파일/링크 분리 OK)
