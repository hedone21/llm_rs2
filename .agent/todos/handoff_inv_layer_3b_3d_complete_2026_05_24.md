# Handoff: INV-LAYER sub-sprint 3B + 3D 종결 → 다음 후보 선택

**작성**: 2026-05-24
**HEAD**: `7df6c0b6 refactor(session): 3B — production CpuBackend alloc cleanup (Minimal)`
**브랜치**: `worktree-b5_trait_extension` (master FF 대기)
**다음 세션 진입 문장**: "INV-LAYER 다음 sub-sprint 후보 리뷰" 또는 "INV-LAYER-001 31건 정리 진입"

---

## TL;DR

3D (bulk_f16_to_f32 L2 promotion, 옵션 B')와 3B (CpuBackend DI cleanup, Minimal) 두 sub-sprint 종결. 3D는 INV-LAYER-003 -6 (182 → 176) 달성, 3B는 baseline 0 효과지만 정성적 cleanup (CpuBackend single Arc 공유) 완료. 멈춘 이유: 한 turn에서 두 sub-sprint 모두 종결, 다음 후보 선택 대기.

---

## 진행 상태

| Sub-sprint | Status | Commit | Baseline 영향 | 정성 효과 |
|---|---|---|---|---|
| 3D bulk_f16_to_f32 L2 (옵션 B') | ✅ DONE | `6c0f5dd2` | INV-LAYER-003 129→123 (-6) | NEON SIMD utility를 L1 backend::cpu → L2 quant 모듈로 격상. caller `#[cfg(aarch64)]` 분기 4건 제거 → non-aarch64 silent garbage 위험도 동시 해소 |
| 3B CpuBackend DI (Minimal) | ✅ DONE | `7df6c0b6` | 0건 (L4→L1 invariant 미정의) | session/batch + qcf_runtime에서 `Arc::new(CpuBackend::new())` 6건 → single Arc 공유. import 2건 정리. |

**baseline 추이**: 196 (3A 직전) → 182 (3A 후) → **176** (3B+3D 후, -6 누계)

**INV-LAYER 카테고리별 (HEAD 7df6c0b6)**:
- INV-LAYER-001: 14
- INV-LAYER-002: 8
- INV-LAYER-003: **123** (-6 from 3D)
- INV-LAYER-004: 4
- INV-LAYER-005: 27

**검증 (호스트)**:
- `cargo test -p llm_rs2 --lib` (host opencl skip): 1199 PASS, 회귀 0
- `cargo test -p llm_rs2 --test spec inv_layer`: 8 PASS
- `cargo fmt --all` + `cargo clippy -p llm_rs2 --lib -- -D warnings`: clean

**디바이스 검증**: 미수행 (mechanical refactor, host gate로 충분 판단).

---

## 다음 작업

직전 §2 Context의 후보 비교 (3E qcf:: architect / 3B / 3D 제외):

### 1순위: **INV-LAYER-001 31건 정리** (해당 baseline 영역에서 비중 8%)

**가설**: L1 backend → 상위 레이어 (L3/L4/L5) 역방향 import 14건. 본문 가장 명확한 위반 패턴.
- 검증: `python3 scripts/layer_lint.py --json | python3 -c "import json,sys; d=json.load(sys.stdin); [print(v) for v in d['violations'] if v.get('rule')=='INV-LAYER-001'][:5]"`
- 게이트: baseline -10 이상, test 회귀 0
- 예상 작업량: 0.5~1일 (mechanical)

### 2순위: **INV-LAYER-005 27건** (generate.rs L5 → 상위 직접 import)

- generate.rs 4,953 LOC를 손대지 않고 줄이려면 session 모듈로 더 격상.
- **위험**: generate.rs 자체가 legacy 보존 방향 ([P2] `generate 바이너리 분할 + Manager 통합`)이라 손대는 자체가 향후 분할 작업과 충돌 가능.
- 다음 sprint 진입 전 backlog 결정 필요.

### 3순위: **INV-LAYER-002 8건** (L2 → 상위)

- 적은 건수, 외과적 처리 가능. ROI 낮음 (4%).

### 4순위 (별도 진행): **arch §13.8 cleanup**

- 3A에서 KVLayout §G 처리 완료. §H trait inversion / §I L4 promotion이 미진행 항목 (직전 backend extension sprint handoff).

---

## Landmines / 미해결

1. **L4 → L1 import는 layer_lint invariant 미정의**: `session/*` → `backend::cpu/opencl/...` 직접 import는 baseline 카운트되지 않는다. 3B에서 처음 발견. 향후 sub-sprint에서 "L4 cleanup으로 baseline 감소"를 추정할 때 반드시 사전 측정 (1분 grep + layer_lint --json 비교) 후 진행할 것. **이 패턴이 INV-LAYER-001 31건 검증에도 영향 — L1 source가 L4 target import한 경우는 잡힘 (V-01~V-02), 그 반대는 안 잡힘**.

2. **bulk_f16_to_f32 host scalar fallback 미검증**: 3D에서 비-aarch64 scalar fallback을 추가했지만 x86 호스트에서 실제로 실행되는 곳은 forward.rs KV F16 dequant 경로다 — host에서 KV F16 모드로 실행하는 통합 테스트가 없어 fallback이 동작 안 해도 회귀 안 잡힐 수 있다. 신규 unit test 2건 (`f16_bulk::tests::round_trip_basic`, `lengths_16_8_residual`)이 fallback 자체는 검증.

3. **Hammered 옵션 (trait method 추가) ROI 0 확정**: P2 backlog #122 entry에 반영. 향후 동일한 trait method 추가 제안이 나오면 반드시 사전에 layer_lint check_violation 함수에서 L4 source 규칙 존재 여부 확인.

4. **`Backend::cpu_companion()` 시그니처 `&dyn` 유지**: 3B 진행 중 Hammered (Arc<dyn> 반환으로 변경)를 검토했으나 caller가 이미 `cpu_backend_arc` 인프라를 갖고 있어 trait 변경이 redundant. 향후 cuda_pc/cuda_embedded 등 GPU backend 내부 호출 패턴이 `self.cpu_companion().matmul(...)` 형태로 이미 동작 — trait sig 변경은 호출지 ~25건 ripple만 발생하고 이득 0.

5. **§2 Context "정량 추정"의 한계**: 3A/3B/3D 모두 §2 Context의 "X건 해소" 추정과 실측이 다르게 나왔다 (3A는 정확히 일치, 3B는 0건, 3D는 정확히 일치). 추정이 정확하려면 사전 측정 (layer_lint --json + grep) 1분 투자 필요. review 스킬 §2 Context에 "정량 표현은 layer_lint --json 등 실측 도구로 확인 후 작성" 보강 후보.

---

## 진입 게이트 (다음 세션)

- 1순위 INV-LAYER-001 진입 시:
  - 사전 측정: `python3 scripts/layer_lint.py --json | jq '.violations[] | select(.rule=="INV-LAYER-001")' | head -50`
  - 14건의 (source, target) 분포 분석 → 그룹화
  - mechanical vs design 분기 결정
- 다른 후보 진입 시: review 스킬로 Plan 사전 리뷰 (스코프 본문 1·2·4·5·7·8·9)

## 참고 파일

- `engine/src/quant/f16_bulk.rs` (3D 신규)
- `engine/src/backend/cpu/neon.rs:2528` (3D 정의 제거 후 line shift, 이전 위치)
- `engine/src/layers/transformer_layer/forward.rs:344/425/430/1009/1086/1091` (3D path 재지향)
- `engine/src/session/batch/runner.rs:202/364` (3B cleanup)
- `engine/src/session/qcf_runtime.rs:202/237/284/295/326` (3B cleanup)
- `engine/tests/spec/inv_layer_baseline.json` (재생성, 176 baseline)
- `scripts/layer_lint.py:171-247` (check_violation 본문 — L4 미정의 확인)
- `.agent/todos/backlog.md` [P2] CpuBackend 생성 책임 통일 — DROP 후보 갱신
