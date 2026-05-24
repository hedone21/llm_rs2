# Handoff: Backend extension sprint 종결 → 다음 sprint 진입 결정 대기

**작성**: 2026-05-24
**HEAD**: `8b9a3c9a refactor(backend): SecondaryStore placeholder 제거 (R-EXT-2 a)`
**브랜치/Worktree**: `worktree-b5_trait_extension` (`.claude/worktrees/b5_trait_extension`)
**다음 세션 진입 문장**: `"INV-LAYER-003 143건 해소 sprint 진입"` (권장) 또는 `"KiviCache hot path downcast sub-sprint 진입"`
**선행 진입점**: [[handoff_b5b_complete_2026_05_23]]

---

## TL;DR

B-5b sprint 후속 — Backend trait 분할 정책 + ISP 재점검을 사용자
**절충안** (hot path 보존 + cold path만 string lookup)으로 종결.
ggml `ggml_backend_reg_get_proc_address(name)` 차용해 `Backend::get_extension(name)`
도입, 외부 모듈 downcast 14건(OpenCLBackend 11 + QnnOppkgBackend 3) →
**string-keyed trait method 한 군데로 통일**. 본 sprint 동안 hot path 16건
유지(forward_gen 5 / forward.rs:589 / transformer 3 / kivi 3 / 기타).
placeholder `SecondaryStore` (사용처 0) + `engine/src/secondary.rs` 제거로
ISP 누적 -1. **멈춘 이유**: 본 sprint 결정 정책 (절충안) 완수. 다음
sprint 후보 결정 사용자 라운드 필요.

**중요 발견**:
1. **외부 downcast 호출지가 architect 라운드 사전 측정의 거의 3배**
   (4건 → 11건). Phase 3.5 추가 sub-sprint로 흡수 — eviction_trigger /
   transformer.rs:377/814 / init.rs 3건. 사용자 옵션 B 결정으로 확장
2. **kivi_cache 3건은 R2 게이트로 hot path 확정** — KIVI 모드 진입 시
   per-token × 16 layer 호출. 본 sprint 제외, sub-sprint 후보 ([[kivi-downcast-resolve]])
3. **transformer.rs:45 `with_op_label`는 hot path** — `profile_events_enabled`
   분기 wrapper로 매 op 호출. `--profile` 모드일 때만 실효성. 유지
4. **partition_workspace.rs 2건은 Buffer trait downcast** (UnifiedBuffer) —
   Backend trait 무관, 별도 backlog ([[buffer-trait-extension]])

---

## 진행 상태

### 본 sprint commits (3건)

| HEAD | scope | LOC | 변경 요약 |
|---|---|---|---|
| `4e9369d6` | docs(arch) | +342 | architect 결정 라운드 — R-EXT-1 (α 평면 const) / R-EXT-2 (a placeholder 제거) / R-EXT-3 (ⅰ+ⅲ rustdoc + COLD-EXT 마커) |
| `51ff0bd0` | refactor(backend) | +112/-21 | Phase 1+2+3+3.5 통합 — extension 인프라 + cold path 11건 치환 (qnn_oppkg / secondary_mmap×2 / transformer×3 / init×3 / eviction_trigger×2) |
| `8b9a3c9a` | refactor(backend) | -30 | Phase 5 — `SecondaryStore` placeholder + `engine/src/secondary.rs` 제거 |

### Phase 진행 표

| Phase | scope | 결과 | 측정 |
|---|---|---|---|
| 0 (architect 라운드) | 3 결정 단일안 | 4e9369d6 | architect 위임, 290 LOC 산출물 |
| 1 (인프라) | `Backend::get_extension` default impl `None` | 51ff0bd0 포함 | 5 backend 변경 0, baseline 변동 0 |
| 2 (OpenCL+QNN impl) | OpenCL/QnnOppkg trait method impl + 3 const | 51ff0bd0 포함 | baseline 변동 0 |
| 3 (cold path 4건) | qnn_oppkg / secondary_mmap×2 / transformer:1057 | 51ff0bd0 포함 | baseline 변동 0 |
| 3.5 (추가 cold 7건) | init×3 / transformer:377/814 / eviction_trigger×2 | 51ff0bd0 포함 | baseline 변동 0 |
| 4 (`name()` 정리) | **취소** (사용자 옵션 B에 미포함) | — | backlog 후보 |
| 5 (placeholder 제거) | SecondaryStore + secondary.rs | 8b9a3c9a | ISP -1, baseline 변동 0 |

### 호스트 게이트 결과 (HEAD `8b9a3c9a`)

| Gate | 결과 |
|---|---|
| cargo build --release -p llm_rs2 | PASS (57s) |
| cargo fmt --all -- --check | clean |
| cargo clippy -p llm_rs2 --lib --bin generate -- -D warnings | 0 |
| cargo test --test spec -- inv_layer | 8 PASS (test_inv_layer_001~007 + 추가) |
| cargo test --lib (병렬) | 22 backend::opencl host-flakiness (pre-existing P3 backlog, serial PASS) |
| cargo test --test spec (병렬) | 2 host-OpenCL flakiness (inv_140 / partition_split_backend_retag, serial PASS) |
| layer_lint --baseline | 0 새 violation |
| baseline (INV-LAYER 합계) | **196 → 196** (변동 0; downcast 패턴 통일은 layer_lint 규칙과 무관) |

### S25 microbench

생략 — 본 sprint 11건 모두 cold path (loader/init/swap/eviction trigger).
forward decode hot path 변경 0 (16건 hot downcast 모두 유지). 게이트는
architect 라운드 §"S25 microbench 생략 정당화"에 명시됨.

### ROI 측정

| 지표 | 진입 | 종결 |
|---|---|---|
| 외부 production `downcast_ref::<OpenCLBackend>()` (cold/sub-hot) | 11건 | 0건 |
| 외부 production `downcast_ref::<QnnOppkgBackend>()` (cold) | 3건 | 0건 |
| hot path downcast (forward/decode loop) | 16건 | 16건 (의도 유지) |
| Backend trait method 수 | 61 | **61** (+1 `get_extension` / -0 / +0 net) |
| ISP 누적 (B-5b 진입 시 57 → 본 sprint 진입 시 61) | 61 | **61** (extension +1 / placeholder trait `SecondaryStore` -1 가 다른 module 이라 net 0) |
| placeholder file 수 | 1 (`secondary.rs`) | 0 |

---

## 다음 작업 — 사용자 결정 라운드 필요

### 후보 sprint A: INV-LAYER violation 196건 본격 해소 (권장)

baseline 196 (B-5b 종결 시점 그대로):
- INV-LAYER-003: **143건** ← main 해소 대상
- INV-LAYER-005: 27건
- INV-LAYER-001: 14건
- INV-LAYER-002: 8건
- INV-LAYER-004: 4건

B-5b Phase 3 A 패턴 (`§G shared identifier promotion` mechanical 적용)
재활용으로 mechanical 진행 가능. ARCHITECTURE.md §13.8 정책 4종
(F/G/H/I + §J 폐기) 활용.

**진입 prompt 초안**:
```
선행: handoff_backend_extension_2026_05_24 (HEAD 8b9a3c9a)
INV-LAYER-003 143건 violation top-source 파일 식별 후
§G/§H 적용으로 mechanical 해소. baseline 196 → ~150 1차 목표.
```

### 후보 sprint B: KiviCache hot path downcast resolve sub-sprint

KIVI 모드에서만 진입하지만 진입 시 per-token × 16 layer hot.
3건 (kivi_cache.rs:1559/1842/2108)을 `KiviCache::new()` 시점 1회
downcast + `Arc<OpenCLBackend>` field 보존으로 해소.

ROI: production hot downcast 3 → 0. 단 KIVI 모드 한정이므로 우선순위 P2.

### 후보 sprint C: Buffer trait extension (partition_workspace 등)

`as_any().downcast_ref::<UnifiedBuffer>()` 패턴 — Backend trait 가 아닌
Buffer trait 외부 진입. partition_workspace.rs 2건 + 기타.

본 sprint와 동일 패턴 (`Buffer::get_extension(name)`) 적용 가능하지만
별도 trait 이므로 scope 분리 필요.

### 후보 sprint D: `name() == "OpenCL"` / `is_gpu()` 정리

호출지 22+건 (loader / generate / prologue 등). 분기 자체는 boolean
property 이므로 ROI 중간. backlog [P3] 등록 가능.

---

## Landmines / 미해결

### L1. 사전 측정 누락 (architect 라운드 4건 vs 실측 11건)

architect 라운드 진행 직전 `grep downcast_ref::<OpenCLBackend>`
sweep 이 충분치 않아 사전 측정 4건만 인지. Phase 3 진입 후 후속
grep 에서 init / transformer / eviction_trigger 추가 7건 발견.
사용자 옵션 B (전체 확장)로 흡수했지만, **향후 architect 라운드
사전에는 `grep -rn` + `wc -l` 로 절대 카운트 확정 후 옵션 제시**.

### L2. KiviCache 3건 hot path 분류 (R2 게이트)

frequency 측정 결과 KIVI 모드 진입 시 per-token × 16 layer 호출.
본 sprint 제외 결정.
- 호출지: `pressure/kivi_cache.rs:1559` (update_gpu) / `1842` (flush_residual_gpu) / `2108` (assemble_view_gpu)
- 후속: KiviCache struct 에 `Option<Arc<OpenCLBackend>>` field 추가 +
  `new_gpu` 에서 1회 downcast → hot path lookup 0. 별도 sub-sprint
  ([[kivi-downcast-resolve]] post)

### L3. partition_workspace 2건은 Buffer trait downcast

`.buffer().as_any().downcast_ref::<UnifiedBuffer>()` 패턴. Backend
trait 무관 → 본 sprint scope 분리.
- 호출지: `partition_workspace.rs:149, 175`
- 후속: Buffer trait 도 `get_extension(name)` 도입 가능 (별도 sprint).
  ROI 작음 — UnifiedBuffer 외 downcast 후보 별로 없음

### L4. transformer.rs:45 `with_op_label` 는 hot path

`profile_events_enabled` 분기 op wrapper. forward 매 op 진입.
`--profile-events` 비활성 시 fast path 이지만 매 호출 downcast 비용
0이 아님. 본 sprint 유지 결정, 별도 sub-sprint 없음. 향후 production
profile-events default OFF 가정에서 ROI 낮음.

### L5. transformer.rs:1642, 1901 `gpu_score_active` 체크는 hot path

forward_into 매 토큰 진입/종료 시점 호출. score accumulator 활성 시만
실효성이지만 분기 자체는 매 토큰 발생. 본 sprint 유지.

### L6. Backend trait method count 변동 (+1 extension, +0 net)

architect 라운드는 -3~-4 (placeholder 제거 포함) 예상했지만 실측
**net 0** (`get_extension` +1 / `SecondaryStore` trait 제거는 다른
모듈이라 Backend trait 자체엔 영향 없음). ISP 누적 우려는 잔존
(여전히 61 method). 다음 sprint 후보 D (`name()` 정리) 진행 시 추가
helper 도입 가능성 — Backend trait 자체는 더 안 늘리도록 주의.

### L7. host OpenCL parallel test flakiness (pre-existing P3)

`cargo test --lib --release` 병렬 실행 시 22 fail (backend::opencl::*).
`--test-threads=1` 격리 시 PASS. backlog `[P3] backend::opencl::* host
test 24개 device-required fail` 그대로. 본 sprint 무관.

### L8. partition_workspace.rs:149/175 `as_any` 패턴 잔여

본 sprint scope 외이지만 grep 결과에 남음. 향후 Buffer trait
extension sprint 진행 시 일괄 정리 대상. handoff landmine 으로만
기록.

---

## 권장 진입

**다음 세션 진입 문장**: `"INV-LAYER-003 143건 해소 sprint 진입"`

- backend extension 정책은 본 sprint 로 종결 — `get_extension(name)`
  패턴이 향후 backend-specific 진입점 표준
- ISP 재점검은 완수 (placeholder 제거 / hot path 결정 / cold path
  통일). 추가 trait 분할은 진정한 capability 누적 추세 발생 시 재검토
- INV-LAYER violation 196건 해소는 paper 사이클 종결 후 default 메인
  트랙 ([[eurosys2027-post-paper]])

대안 진입 문장:
- `"KiviCache hot path downcast sub-sprint 진입"` (L2 해소, P2)
- `"Buffer trait extension sprint"` (L3, ROI 작음)

---

## 참조

- 본 sprint architect 라운드: `arch/sprint_backend_extension_round.md`
- 선행 handoff: `.agent/todos/handoff_b5b_complete_2026_05_23.md`
- 차용 출처: ggml `ggml_backend_reg_get_proc_address(reg, name)` —
  `https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-backend.h`
- 외부 비교 자료: `engine/src/backend.rs` (61 method) vs ggml-backend
  (5-way 분할, struct당 ≤14 fn ptr) vs ExecuTorch (`BackendInterface`
  6 virtual method)
