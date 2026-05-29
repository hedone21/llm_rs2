# Backend Conformance Harness — 설계 노트

> **상태**: **구조 확정 / 내용물 보류** (2026-05-29 `/improve-codebase-architecture` grill 산출물). design-in-progress.
> **출처**: improve-codebase-architecture 후보 ② (deep-module 렌즈 — "interface 가 곧 test surface"). 후보 리포트 맥락은 `.agent/todos/handoff_kv_weight_grill_2026_05_28.md` R0′.
> **대응 spec**: `spec/41-invariants.md` — `INV-BACKEND-COMPUTE-FALLBACK` (본 harness 가 그 실행 검증기), `INV-STAGE-ORDER-SAFETY` (동일 패턴이 후보 ⑤ stage harness 로 확장).
> **관련 설계**: `arch/pipeline_stage_design_v2.md` §3.1 (compute auto-default), §3.2 (fallback profiling).

## 1. 문제 — 현재 `test_backend.rs` 의 shallowness

`engine/src/bin/test_backend.rs` (1,041 LOC) 는 부분적 harness 다 — `backends: Vec<Arc<dyn Backend>>` 를 CLI 로 골라 여러 backend 에 같은 op 를 돌려 비교하는 구조는 이미 있다 (line 59~106). 그러나 deep-module 렌즈에서 4 마찰:

1. **커버리지 구멍** — compute op ~24 중 **5개**(MatMul / MatMulTransposed / MatMulSlice / Softmax / RMSNorm) 만. `attention_gen` / `flash_attention_prefill` / `silu_mul` / `rope_inplace` 등 hot op 누락.
2. **oracle 가 op 마다 손으로 박힘** — line 384~449 hand-rolled reference. 24 op 확장 시 폭증 + 그 자체 버그원. "무엇이 correct 인가" 의 단일 출처 부재.
3. **계약 테스트 0** — `INV-BACKEND-COMPUTE-FALLBACK` (compute default → `cpu_companion` 위임, `unimplemented!` 금지, `fallback_profile::note` 호출) 을 실행 검증하는 코드 없음. 문서상 계약일 뿐.
4. **완전성 보장 없음** — "새 backend = trait 구현 + harness 통과 = 정확성 증명" 닫힘 부재. 정확성 책임이 사용자에게 이양 (위험).

## 2. 확정 구조 (deep-module deepening)

핵심: `Backend` trait 을 그 interface 로 검증 — **interface 가 곧 test surface**.

### 2.1 재사용 lib 함수 (bin 아님)

`engine` 크레이트 lib 위치에:

```rust
pub fn assert_backend_conformance(
    backend: &dyn Backend,
    oracle: &dyn Oracle,        // §2.3 seam — 보류한 oracle 결정의 착지점
    opts: ConformanceOpts,      // tolerance 등 (보류 내용물)
) -> ConformanceReport;
```

op 순회 + 비교 + `ConformanceReport` 빌드가 본 함수에 locality. **세 진입점이 재사용**: host `#[test]`, device bin, 새 backend 자기 test. (bin inline 로직이면 재사용 불가 → deepening 실패.)

### 2.2 계약(contract) 검증 — 런타임 spy (정적 grep 없음)

`FallbackOnlyBackend` (required floor 만 구현, compute op override 0) 를 **counting CPU companion** 으로 감싸고 `LLMRS_FALLBACK_PROFILE` on 으로 전 op 실행:

- **위임** — companion 카운터 op 별 > 0
- **no-panic** — 실행이 panic 없이 완료. (`unimplemented!` default 가 있으면 그 op 실행 시 패닉 → 실패. spy 가 전 op 을 exercise 한다는 coverage 전제 — §3 보류.)
- **note** — profile 에 op 별 기록

정적 source-grep 은 **채택 안 함** (source 텍스트를 테스트 = brittle, code smell — 사용자 결정 2026-05-29). grep 이 잡으려던 "실행 안 한 op 의 panic default" 는 "spy 가 전 op 실행" 으로 런타임에서 닫힌다.

`FallbackOnlyBackend` 은 테스트 도구 이상 — **"아무것도 가속 안 한 갓 만든 backend" 그 자체**. 여기에 conformance 를 돌리면 floor(required + auto-default) 정상 동작 = 새 backend bring-up baseline.

### 2.3 oracle = seam (보류 결정의 착지점)

`Oracle` 을 인자 seam 으로 두어, 흡수가 oracle 결정을 **선점하지 않음**. 기존 hand-rolled reference 는 임시 `Oracle` adapter 로 꽂아둔다 (삭제 X, blocking X). 보류한 oracle 결정(CPU backend / golden vector / 층위 D)이 이 seam 에 들어온다.

### 2.4 host vs device 진입 — 둘 다, 같은 코어

- **host `#[test]`** (`cargo test --workspace`): CPU oracle 앵커 + `FallbackOnlyBackend` spy + 계약. platform-independent → 매 커밋 CI 검증.
- **device bin** (`test_backend.rs` thin wrapper, `run_device.py` 배포): Adreno(S25) / CUDA(Jetson) 가속 경로 vs CPU oracle. device 별 tolerance 는 `opts` 인자.

둘이 같은 `assert_backend_conformance` 호출 → 로직 중복 0.

### 2.5 `test_backend.rs` 흡수 partition

| 현재 내용 | 처리 |
|---|---|
| op 순회 + 비교 + `ConformanceReport` | **lib 로 이동** |
| CLI 파싱 / backend 구성 / 결과 출력 | **bin thin 유지** (~80 LOC) |
| hand-rolled per-op reference (384~449) | **oracle seam 의 임시 adapter** (보류 결정까지) |
| `run_kivi_attention_test` (689) | **분리 → `assert_kivi_conformance(&dyn KiviAttentionBackend)`** (capability conformance, earmark) |

base `Backend` conformance 는 **paradigm-agnostic 유지** (KIVI 는 capability conformance 로 — 후보 ④ 정신).

## 3. 보류 (내용물 — content)

구조와 독립. `assert_backend_conformance` 시그니처/seam 안에 나중에 채운다:

- **oracle** — CPU backend / golden vector / **층위 D**(가속 backend = CPU 비교 + CPU = 소수 golden·analytic 앵커). 추천 D, 미확정.
- **coverage 경계** — compute op 전부? memory/sync op? capability sub-trait 포함?
- **tolerance 정책** — 정확 일치 vs per-(backend, dtype, op) tolerance (Q4_0 양자화 오차, Adreno fp16 rounding 등).

## 4. 효과

- **deletion test ✓** — 1,041 LOC mixed concern → lib 함수 한 곳 + thin bin (~80).
- `INV-BACKEND-COMPUTE-FALLBACK` 가 죽은 글자 → **실행 검증**.
- ①(ScoreCollector) / ③(hook 흡수) / ④(KVCacheLayer) migration 의 **회귀 그물** (de-risk 전제 — 그래서 후보 중 first).
- 패턴이 후보 ⑤ (PipelineStage 순서-안전 property test) 로 확장.
