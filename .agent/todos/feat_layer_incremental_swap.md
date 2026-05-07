# Layer-Incremental Swap — Stage 1 MVP plan

> **작성일**: 2026-05-07
> **작성자**: PM (메인 세션 분석 요약 기반)
> **Backlog 항목**: `LISWAP-1` (P0, current sprint), `LISWAP-2~4` (Stage 2+, backlog)
> **목적**: Galaxy S25 단발 swap stall 290 ms를 N decode tick에 분산하여 frame budget(33 ms @ 30 fps) 안에 욱여넣는다.

---

## 0. Spec ID Backref (Architect 발급, 2026-05-07)

본 plan의 작업 단위에 대응하는 spec ID는 다음과 같다. Implementer는 각 ID의 spec 본문(`spec/32-engine-algorithms.md`, `spec/41-invariants.md`)과 arch 컴포넌트(`arch/weight_swap.md` §8)를 참조하여 구현 + spec test를 작성한다.

### Spec IDs

| ID | 종류 | 위치 | 한줄 의도 |
|----|------|------|----------|
| **ENG-ALG-232** | Algorithm | `spec/32-engine-algorithms.md` §3.12.21.1 | `IncrementalSwapPlan` 자료구조 + drain/is_empty/len 의미론 |
| **ENG-ALG-233** | Algorithm | `spec/32-engine-algorithms.md` §3.12.21.2 | Decode loop forward → drain → execute_on_slots → retire dispatch |
| **ENG-ALG-234** | Algorithm | `spec/32-engine-algorithms.md` §3.12.21.3 | Plan commit lifecycle: commit-then-run-to-completion, in-flight signal 무시 |
| **INV-144** | Correctness | `spec/41-invariants.md` §3.20 | Forward-snapshot 정합성 (INV-121 보존, byte-equality 비요구) |
| **INV-145** | Correctness | `spec/41-invariants.md` §3.20 | drain 단조 감소 + retire 정확성 + 빈 chunk 호출자 가드 |
| **INV-146** | Safety/Correctness | `spec/41-invariants.md` §3.20 | tick 경계 = batch 경계, ENG-ALG-231 stage gate × N tick 적용 |

### Arch 매핑

- `arch/weight_swap.md` §8 (Phase 6.6 LISWAP-1): 컴포넌트 분해 + Mermaid 다이어그램 + INV-144 byte-equality 비요구 명시.
- `arch/weight_swap.md` §3: CLI 표에 `--swap-incremental-per-tick` 추가.

### CLI

- `--swap-incremental-per-tick` (`usize`, default `0`) — `spec/33-engine-data.md` §3.15.16 표 갱신. ID는 ENG-ALG-232~234 / INV-144~146로 cross-ref.

### Implementer가 작성할 spec test (feedback_spec_tests_required 준수)

| 테스트 파일 | 검증 ID | 핵심 케이스 |
|-------------|---------|-------------|
| `engine/tests/spec/test_eng_alg_232_incremental_plan.rs` | ENG-ALG-232, INV-145 | drain progression (per_tick=1/3/5), is_empty/len, 빈 chunk 반환 (n=0 또는 retired) |
| `engine/tests/spec/test_eng_alg_233_decode_dispatch.rs` | ENG-ALG-233, INV-146 | per_tick=0 회귀 (single-shot byte-equal), per_tick≥1 chunk 진행 + retire 시퀀스, ratio_generation bump 횟수 = chunk 수 |
| `engine/tests/spec/test_eng_alg_234_plan_lifecycle.rs` | ENG-ALG-234 | commit 후 새 SwapWeights signal 무시, plan empty 후 새 신호는 수용 |
| `engine/tests/spec/test_inv_144_snapshot_consistency.rs` | INV-144 | chunk 진행 시퀀스 동일 시 forward determinism, INV-121 stress test 재사용 |

### 비범위 (Stage 2+ 후속 spec ID 시리즈)

- LISWAP-2: PI hysteresis 4-state 상태기계 — 별도 spec 시리즈로 발급 예정.
- LISWAP-3: ImportanceTable 역순 ordering — ENG-ALG-232의 입력 순서 재정의.
- LISWAP-4: Background dispatcher thread — ENG-ALG-233 dispatch 위치 변경.

---

## 1. 배경 및 문제 정의

### 1.1 현재 상태 (Phase 6.5 + AOS swap fix 후)

- **측정 보고**: `papers/eurosys2027/_workspace/experiment/swap_overhead_s25_phase6_5.md`
- **단발 swap stall (S25, n=5)**: 290 ms (σ=36)
- **30 fps frame budget**: 33 ms — 290 ms는 약 9 frame 분량 stall
- **EuroSys 2027 paper critical path** — latency-budget 표가 깨지면 논문 contribution 자체가 흔들림

### 1.2 Stage breakdown (290 ms)

| Stage | ms | 비중 | 비고 |
|---|---:|---:|---|
| mmap_permute | 282.6 | 97% | host→GPU 업로드 ~900 MB, **단일 병목** |
| prefault | 6.8 | 2% | mmap 페이지 fault-in |
| synchronize | 0.6 | 0% | 마지막 clFinish |
| 기타 | ~0 | 0% | dispatch/release |

### 1.3 후보 평가

| 옵션 | 절감 추정 | risk | 검증 부담 |
|---|---:|---|---|
| **A. Zero-copy USE_HOST_PTR** | ~200 ms | high (R1~R7 unknown) | high |
| **B. Layer-incremental swap** | 0 ms (분산만) | low | low |
| **A+B 결합** | 90+200 = 미정 | medium | medium |

**Stage 1 결정**: B를 먼저 구현. 이유:
- Architecture가 자연스럽게 지원 (ArcSwap LayerSlot이 layer 단위 atomic)
- A의 R1~R7 (UMA 캐시 비일관성, Adreno 드라이버 zero-copy 실효, mmap page eviction, AUF page alignment 등) 검증 부담이 크다
- B는 total을 줄이지 못하지만 user-perceived stall은 frame budget 안으로 가져옴 → 충분히 contribution 성립

---

## 2. Stage 1 (B-1, synchronous interleave) MVP scope

### 2.1 작업 범위

1. **CLI 플래그**:
   - `--swap-incremental-per-tick=N` (default = 0, single-shot 호환 유지)
   - N = 0: 기존 단발 경로 (회귀 없음 보장)
   - N > 0: incremental 경로

2. **자료구조** (신규 파일 `engine/src/models/weights/incremental_plan.rs`):
   ```rust
   pub struct IncrementalSwapPlan {
       remaining: VecDeque<usize>,   // 남은 layer index
       per_tick: usize,              // tick당 처리할 layer 수
       started_at_token: usize,      // 진단/측정용
   }

   impl IncrementalSwapPlan {
       pub fn new(layers: Vec<usize>, per_tick: usize, token: usize) -> Self;
       pub fn drain(&mut self, n: usize) -> Vec<usize>;  // 다음 tick chunk
       pub fn is_empty(&self) -> bool;
       pub fn len(&self) -> usize;
   }
   ```

3. **Decode loop integration** (`engine/src/bin/generate.rs`):
   - 기존: 신호 발생 → `SwapExecutor::execute_on_slots(all_target_layers, ...)` 한 번 호출 (290 ms stall)
   - 신규: 신호 발생 → `IncrementalSwapPlan::new(target_layers, per_tick, current_token)` 생성 → 매 forward 직후 `plan.drain(per_tick)` → `SwapExecutor::execute_on_slots(chunk, ...)` 호출 → `plan.is_empty()` 시 None으로 retire

4. **Cooldown 정책 (Stage 1 한정 단순화)**:
   - **Plan을 commit하면 끝까지 실행**. 진행 중 새 압력 신호는 무시 (혹은 plan 종료 후 일정 cooldown 동안 거부)
   - 진행 중 압력 변화 대응 (Forward / Backward / Cooldown 상태기계)은 **Stage 2 (LISWAP-2)** 로 분리

### 2.2 수치 추정 (Galaxy S25)

| per_tick | 토큰 수 | 토큰당 stall (ms) | 30fps frame skip |
|---:|---:|---:|---:|
| 25 (현재 single-shot) | 1 | 290 | 47 |
| 5 | 5 | ~58 | 2 |
| **3** | **9** | **~35** | **1** |
| 2 | 13 | ~23 | **0 (budget 안)** |

### 2.3 Trade-off (명시)

- 매 tick마다 **stage 고정비용** 부과: prefault 6.8 + synchronize 0.6 = **7.4 ms / tick**
- per_tick=2 (13 tick) 케이스: 13 × 7.4 = **96 ms 추가** total overhead
- 결과 추정: single-shot 290 ms → incremental 386 ms total (33% 증가)
- 그러나 **user-perceived stall = 23 ms (frame budget 이내)**
- Trade를 측정 보고에 명시할 것: "total swap latency는 늘어나지만 frame skip 0"

---

## 3. 작업 순서 및 의존

```
[1] PM (현재) — backlog 등록 + plan 작성        ✅ 완료
   ↓
[2] Architect — Spec ID 발급 (LISWAP-1-*) + IncrementalSwapPlan API 설계 검토
   - 신규 spec: incremental_plan.rs 인터페이스 (drain 의미, is_empty 정의, 수명 관리)
   - 신규 spec: --swap-incremental-per-tick CLI 의미
   - 신규 spec: cooldown 정책 (plan 진행 중 신호 무시)
   - tests/spec/ 하위 스펙 ID 등록 (feedback_spec_tests_required 준수)
   ↓
[3] Implementer — 코드 + spec test 작성
   - 신규 모듈: engine/src/models/weights/incremental_plan.rs
   - generate.rs decode loop 수정 (drain 호출)
   - spec test (engine/tests/spec_layer_incremental_swap.rs):
     · per_tick=25(single-shot) vs per_tick={1,2,3,5} forward logits byte-equal
     · per_tick=0(default) → 기존 경로 회귀 없음
     · plan.is_empty() 시 retire 정확성
     · 진행 중 새 신호 무시 동작
   - cargo fmt + cargo clippy --workspace -- -D warnings
   - sanity-check 스킬 통과 후 자동 커밋
   ↓
[4] Tester — Galaxy S25 디바이스 측정
   - run_device.py + deploy-test 스킬
   - per_tick ∈ {0, 1, 2, 3, 5} sweep
   - n=5 반복, 90% trim
   - 수집: per-token wall-clock, swap stage breakdown, decode 정확성(top-5 overlap)
   - 보고서: papers/eurosys2027/_workspace/experiment/swap_incremental_s25.md (신규)
```

### 의존성 그래프

```
PM (LISWAP-1 register) ──→ Architect (spec ID 발급)
                              ↓
                          Implementer (코드 + spec test)
                              ↓
                           Tester (S25 측정)
                              ↓
                       측정 보고 → 다음 결정 (LISWAP-2 진행 여부, A 트랙 진행 여부)
```

---

## 4. Acceptance Criteria

### 4.1 Spec test (host-isolated)

- [ ] `per_tick=25(single-shot)` vs `per_tick={1,2,3,5}` forward logits **byte-equal**
  - synthetic weight (deterministic, host에서 격리 가능한 작은 모델)
  - decode N tokens, 모든 token logits 일치
- [ ] `per_tick=0` (default) → 기존 경로와 100% 동일 (회귀 없음)
- [ ] `IncrementalSwapPlan::is_empty()` 후 retire 정확성 (None으로 전환, 다음 tick에 drain 호출 안 됨)
- [ ] Plan 진행 중 새 swap 신호 들어오면 무시 (Stage 1 단순화)

### 4.2 Device test (Galaxy S25)

- [ ] **정확성**: per_tick ∈ {1,2,3,5} 모두 single-shot 대비 top-5 overlap > 99%
- [ ] **Token-time CDF**: per_tick=3에서 worst-case stall **≤ 35 ms** (1 frame skip 이내)
- [ ] **Token-time CDF**: per_tick=2에서 worst-case stall **≤ 25 ms** (frame budget 이내)
- [ ] **Total swap latency**: per_tick=3 기준 single-shot 290 ms → 315 ms 이내 (overhead +25 ms 허용)
- [ ] **Total swap latency**: per_tick=2 기준 386 ms 이내 (96 ms overhead 추정과 일치)
- [ ] 메모리 회수: swap 후 PSS는 single-shot과 동일 수준 (회귀 없음)

### 4.3 측정 환경

- 디바이스: Galaxy S25 (Snapdragon 8 Elite, Adreno 830)
- 모델: Llama 3.2 1B F16 + Q4_0 GGUF 두 벌
- 스레드: 6T (CLAUDE.md `feedback_benchmark_thread_count` 준수)
- 측정: wall-clock (`Decode: X ms/tok`), `--profile` 사용 금지 (sync overhead 부풀림)
- swap 트리거: 외부 압력 injection 또는 manager directive (기존 path)

---

## 5. 영향 파일 (예상)

| 파일 | 변경 종류 | LOC | 비고 |
|---|---|---:|---|
| `engine/src/models/weights/incremental_plan.rs` | 신규 | ~80 | `IncrementalSwapPlan` 구현체 |
| `engine/src/models/weights/mod.rs` | 수정 | ~5 | `pub mod incremental_plan` 등록 |
| `engine/src/bin/generate.rs` | 수정 | ~80 | CLI parse + decode loop drain 호출 |
| `engine/tests/spec_layer_incremental_swap.rs` | 신규 | ~150 | 4개 spec test 케이스 |
| `arch/layer_incremental_swap.md` | 신규 (Architect) | ~100 | 상태 다이어그램 + cooldown 정책 |
| `spec/` 하위 LISWAP-1-* | 신규 (Architect) | — | spec ID 발급 |

**총 예상**: ~415 LOC (코드 165 + 테스트 150 + 문서 100)

기존 `SwapExecutor::execute_on_slots(chunk, ...)` 의 chunk 인터페이스가 이미 존재한다고 가정. 부재 시 단일 layer 기존 path를 loop으로 호출하는 thin wrapper로 충분 (구현 상세는 Implementer가 결정).

---

## 6. 위험 및 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| `SwapExecutor::execute_on_slots`가 chunk를 받지 않음 | 구현 복잡도 증가 | 단일 layer 호출을 loop로 감싸는 wrapper로 대응. Architect 검토 시점에 확인. |
| Stage 고정비용 7.4 ms/tick이 추정보다 큼 | per_tick=2도 frame budget 초과 | per_tick=3, 4로 후퇴. 이 경우 1 frame skip은 발생하지만 47 → 1로 충분. |
| Plan 진행 중 새 압력 신호 무시가 과도하게 보수적 | 압박 변화 미대응 | Stage 1 한정 단순화로 명시. LISWAP-2에서 상태기계 도입. |
| Spec test의 byte-equality가 floating point 비결정성으로 실패 | 정확성 검증 불가 | 동일 weight + 동일 sampler seed → bit-exact 가능. 실패 시 ULP 허용으로 완화. |
| `IncrementalSwapPlan` lifetime이 generate.rs main loop과 얽힘 | borrow checker 이슈 | `Option<IncrementalSwapPlan>` + `Option::take()` 패턴 권장. Implementer 재량. |

---

## 7. Stage 2+ 분리 항목 (이번에는 안 함)

- **LISWAP-2 (P2)**: PI controller hysteresis 상세화. Idle/Forward/Backward/Cooldown 상태기계.
- **LISWAP-3 (P3)**: Layer-ordering 정책 (QCF_weight ImportanceTable 역순 → abort 시 품질 손실 최소).
- **LISWAP-4 (P3)**: B-2 Background dispatcher thread (GPU queue 경합 risk, 보상 < 복잡도 가능성).
- **LISWAP-OPT-A (P3)**: Zero-copy USE_HOST_PTR 검증 트랙 (별 트랙). LISWAP-1 측정 후 frame budget 미달 시 후순위.

---

## 8. 다음 단계

1. **Architect**: 본 plan 문서를 입력으로 받아 spec ID `LISWAP-1-*` 발급, `arch/layer_incremental_swap.md` 작성, `IncrementalSwapPlan` API 설계 확정.
2. **Implementer**: Architect spec 기반 구현 + spec test, sanity-check 통과 후 자동 커밋.
3. **Tester**: deploy-test 스킬로 S25 측정, `swap_incremental_s25.md` 보고서 작성.
4. **PM (재진입)**: Tester 보고를 받아 LISWAP-2 진행 여부 / A 트랙 진행 여부 결정.

---

**참조**:
- `papers/eurosys2027/_workspace/experiment/swap_overhead_s25_phase6_5.md` — 290 ms 실측
- `.agent/todos/feat_weight_swap.md` — Phase 1~6 weight swap history
- `.agent/todos/backlog.md` — `WSWAP-6-*` 시리즈 (290 ms 자체 감축, 직교)
- `CLAUDE.md` — Tensor Partition / `--zero-copy` / ARM UMA 캐시 일관성 노트
