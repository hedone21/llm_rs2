# Handoff: Phase α-K substep (2) 완료 (2a+2b, S25 device-gated PASS) → substep (2c)

**작성**: 2026-06-03
**HEAD**: `b9d024bc refactor(pressure): Phase α-K substep (2b) — eviction guard+dispatch 코어를 EvictionHandler/registry 공유로 통합`
**브랜치**: `master` (origin 미push — ahead 5: `1502f568` `30067871` `b9d024bc` + 2 선행)
**다음 세션 진입 문장**: **"α-K substep (2c) 진행"** (`determine_pressure_level` → `PressureSource` 추출, **β-경계 재census 선행**).

> SSOT = `arch/pipeline_stage_design_v2.md` (§4.1 / §5 / §9.1). ADR = `docs/adr/0001-kv-dispatch-paradigm.md` (Accepted). 트랙 메모리 = [[project-pipeline-alpha-k]].

---

## TL;DR — 이번 세션 arc

1. **substep (2) scope 판정 = rescope-narrower, β 미결합** (census→synthesis→adversarial workflow `wf_c03a913a`, verdict_holds=true). §9.1 표 괄호 "(Eviction/KvMerge/SwapDispatch)"는 두 해석 — A(신 §5 `PipelineStage::on_phase` impl 입주 → `PipelineRegistry` 소스 0 + StepInfo carrier + DecodeLoop 재작성 = **α-W-3b와 동일 β-trap**) vs B(실제 명세 "generic flip 아닌 구조 분해", carrier 불변). **B로 고정**: 신 PipelineStage 입주 금지(β 연기), grep 게이트(substep 2 diff에 `impl PipelineStage`/`PipelineRegistry`/`StageContext` 0건).
2. **§5 인프라 현황**: `engine/src/pipeline.rs`에 PipelineStage trait/LifecyclePhase/StageContext/StageOutcome/PipelineDispatcher **타입만** 존재(α-W-1 `0d12c81d`). **impl 0 / `PipelineRegistry` struct 부재**. 라이브 production = 구 `CachePressureHandler`+`CachePressurePipeline`(pressure.rs). 두 추상 공존.
3. **2a ✅ `30067871`** + **2b ✅ `b9d024bc`** 구현·host-gated·커밋. **S25 device-gate PASS**(아래).
4. prerequisite 해소: baseline 동결 `9b350609`, SnapKV 게이트 재정의(구현체 0 → 실행가능 4구성).

---

## substep (2) 증분 진행 + 게이트

| 증분 | 내용 | commit | host | device(S25) |
|---|---|---|---|---|
| **2a** | budget 인라인 복제 제거 — `execute_dispatch`에 `layer_ratios` 파라미터 추가, `force_evict_with_scores_and_budgets` 위임 축약 | `30067871` | ✅ byte-identical + 회귀 가드 테스트 | (byte-identical라 device 불요) |
| **2b** | `run_policy_eviction`↔`EvictionHandler::handle` 통합 — `run_policy_eviction`을 `target_len` 파라미터+`pub(crate)` 코어로 승격(ratio→target_len은 `resolve_target_len` 헬퍼), EHH 인라인 guard/dispatch/result 제거→코어 위임+ScoreContext 평탄화 | `b9d024bc` | ✅ eviction_handler 9 + cache_manager 29 (StreamingLLM ratio≤0 + budgets 회귀 가드) | ✅ **PASS** |
| **2c** | `determine_pressure_level`(cache_manager.rs:153) → `PressureSource` 추출 | 미착수 | — | — (β-경계 재census 선행) |

- **2b 동등성 증명**(8 diff 1:1 대조): 유일 실거동 차 = D3(ratio≤0 정책결정/StreamingLLM, RPE→target_len 0 vs EHH→1) → 호출자가 자기 target_len 선해소+코어는 target_len만 받음으로 소멸. 잔여 = 진단 로그(EHH undershoot warn 획득+MIN_EVICT prefix 통일) + new_pos `caches[0]→max_cache_pos`(lockstep 등가).

## S25 device gate 결과 (2026-06-03)

- **방법**: current `b9d024bc` vs baseline `9b350609`(detached checkout 빌드) 두 android `legacy_generate`, Qwen2.5-1.5B Q4_0, `--backend opencl --greedy`(결정적).
- **bit-identical 4/4 PASS**: none/sliding/h2o/d2o 생성 텍스트 완전 일치. **결정적·thermal 무관 = 핵심 correctness 게이트.**
- **avg_tbt**: cool-state hot-path(none) current 31.38 vs baseline 31.42 = **+0.1%** (≤+3% PASS). eviction-config tbt는 **thermal-confounded**(eviction 없는 none이 발열만으로 31→73ms 변동 입증) — 코드 무관, current ≤ baseline 도처(회귀 방향 아님).
- **eviction 실발화 확정**: `-n 256 --memory-threshold-mb 999999 --eviction-target-ratio 0.5` → `pressure=Emergency` 매 step → EvictionHandler::handle → run_policy_eviction(2b 코어) 실행. d2o는 "[KVCache] Shrunk 512→64" + 출력 분기까지 확인(sensitive).
- **⚠️ §9.1 "32-tok" 게이트 한계 발견**: `MIN_EVICT_TOKENS=64 > 33토큰`이라 32-tok에선 eviction **구조적 미발화**(표준 path만 검증). 2b 코어 검증엔 **-n 256+ 강제 필수** — 미래 device gate 절차에 반영.
- **Jetson 건너뜀**(사용자 결정). **KIVI 건너뜀**(2a+2b와 직교 + substep 1 KIVIFormat unwired → substep 3 wiring으로 연기).

## 다음 — substep (2c)

- **2c 작업**: `determine_pressure_level`(cache_manager.rs:153, threshold 대비 mem 4-level 양자화)을 함수/trait로 추출. §5.1이 이를 미래 `LocalPressureSource`로 지목.
- **β-경계 재census 필수**(2b처럼): §5 `PressureSource` trait가 `pipeline.rs`에 **이미 정의**돼 있음 → 2c가 (a)함수 추출만(자체완결, B-style)인지 (b)`PressureSource` trait 배선+`band()` 강등(β 결합)인지 census로 가른다. **band() 배선=β로 연기, 순수 함수 추출만 2c.**
- **device gate**: 2c도 step-tier non-byte-identical 가능 → S25 게이트(위 방법 재사용, **-n 256+ 강제** 필수). Jetson은 사용자 요청 시.

## Landmines / 교훈
- **device gate eviction 발화 조건**: 32-tok은 MIN_EVICT_TOKENS(64) 미달로 vacuous. **-n 256+ & `--memory-threshold-mb 999999` & `--eviction-target-ratio 0.5`** 로 강제해야 2b 코어 발화.
- **thermal이 tbt를 지배**: S25는 sustained load + 빌드 직후 발열로 tbt가 2배 변동(none 31→73). tbt 비교는 **cool-state first-run** 또는 cooldown 필수. bit-identical(결정적)이 신뢰 가능한 게이트.
- **device run gotcha**: `--prompt`은 단일 토큰만(run_device.py adb 공백분리). CLI는 **subcommand 구조**(`--greedy eviction sliding --window N`), 구 `--eviction-policy` flat flag 아님. tokenizer-path 명시 필수.
- **baseline 빌드**: detached checkout(`git checkout 9b350609`)으로 build+deploy 후 `git checkout master` 복원. gitignored devices.toml/hosts.toml 유지됨. worktree는 gitignored config 부재로 비권장.
- **세션 외 untracked `arch/pipeline/`**: companion 문서, 내 작업 무관(transcript Write 0 확인), 미커밋·미삭제.
- **커밋 금지 untracked**: `.antigravitycli/`, `.claude/scheduled_tasks.lock`, `papers/.../microbench_*`, `.agent/todos/handoff_microbench_*.md`. 명시 파일 add(`git add -A` 금지).
- **§9.1 SnapKV 각주 갱신 권고**(Architect, 미반영): SnapKV 구현체 0 → device 게이트 4구성(none/sliding/h2o/d2o)+KIVI로 재정의 문서화.

## 자기점검
- 진입 문장: ✓ "α-K substep (2c) 진행" (β-경계 재census 선행)
- 왜 멈췄나: ✓ 사용자가 S25 게이트 완료 후 handoff 갱신 + 다음 세션 연속 요청
- 최대 landmine: ✓ 2c의 β-결합 여부(PressureSource band() 배선) + device gate eviction 발화 조건(-n 256+)
- 검증 게이트: ✓ 2a/2b host PASS 수치 + S25 bit-identical 4/4 PASS commit/handoff 기록
- device 가용: ✓ S25 USB(`R3CY408S5SB`). Jetson ssh alias 미해결(165.132.107.73:4121 45일 stale, 사용자 요청 시 복구). cargo-zigbuild 미설치.
