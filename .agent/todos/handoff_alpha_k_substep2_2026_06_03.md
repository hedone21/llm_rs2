# Handoff: Phase α-K substep (2) 종결 (2a+2b, S25 device-gate PASS; 2c=β-throwaway 판정 skip) → substep (3)

**작성**: 2026-06-03 (closure 갱신)
**HEAD**: `b9d024bc` (substep 2b) — closure 커밋은 본 handoff 갱신 직후
**브랜치**: `master` (origin 미push)
**다음 세션 진입 문장**: **"α-K substep (3) 진행"** (forward-path generic→trait object flip; 3c=`LlamaLayer::forward`+`attention_into` perf 위험 crux, **device round 필요**).

> **substep (2) 종결 결정 (2026-06-03)**: 계획됐던 2c(`determine_pressure_level` 추출)는 β-경계 재census 워크플로우(`wf_57ad2f94-a93`, `verdict_holds=false`)에서 **cosmetic throwaway**로 판정 — byte-identical하나 β에서 4중 불일치(L3↔L4 / `PressureLevel`↔`Pressure(0–100)` / 단일함수↔`band()`분할 / memory-only↔전센서융합)로 통째 폐기. ADR-0001:203 substep(2) 정의(Eviction/KvMerge/SwapDispatch Stage)에도 미포함. **사용자 결정 = substep(2)를 2a+2b로 종결, `determine_pressure_level → LocalPressureSource` 마이그레이션은 β(LocalPressureSource 신설 시점) 흡수.** (상세 = 아래 "§2c 종결" 섹션.)

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
| ~~**2c**~~ | `determine_pressure_level` 추출 — **β-경계 재census 결과 cosmetic throwaway 판정 → 미수행, β 흡수** (사용자 결정 2026-06-03) | skip | — | — |

- **2b 동등성 증명**(8 diff 1:1 대조): 유일 실거동 차 = D3(ratio≤0 정책결정/StreamingLLM, RPE→target_len 0 vs EHH→1) → 호출자가 자기 target_len 선해소+코어는 target_len만 받음으로 소멸. 잔여 = 진단 로그(EHH undershoot warn 획득+MIN_EVICT prefix 통일) + new_pos `caches[0]→max_cache_pos`(lockstep 등가).

## S25 device gate 결과 (2026-06-03)

- **방법**: current `b9d024bc` vs baseline `9b350609`(detached checkout 빌드) 두 android `legacy_generate`, Qwen2.5-1.5B Q4_0, `--backend opencl --greedy`(결정적).
- **bit-identical 4/4 PASS**: none/sliding/h2o/d2o 생성 텍스트 완전 일치. **결정적·thermal 무관 = 핵심 correctness 게이트.**
- **avg_tbt**: cool-state hot-path(none) current 31.38 vs baseline 31.42 = **+0.1%** (≤+3% PASS). eviction-config tbt는 **thermal-confounded**(eviction 없는 none이 발열만으로 31→73ms 변동 입증) — 코드 무관, current ≤ baseline 도처(회귀 방향 아님).
- **eviction 실발화 확정**: `-n 256 --memory-threshold-mb 999999 --eviction-target-ratio 0.5` → `pressure=Emergency` 매 step → EvictionHandler::handle → run_policy_eviction(2b 코어) 실행. d2o는 "[KVCache] Shrunk 512→64" + 출력 분기까지 확인(sensitive).
- **⚠️ §9.1 "32-tok" 게이트 한계 발견**: `MIN_EVICT_TOKENS=64 > 33토큰`이라 32-tok에선 eviction **구조적 미발화**(표준 path만 검증). 2b 코어 검증엔 **-n 256+ 강제 필수** — 미래 device gate 절차에 반영.
- **Jetson 건너뜀**(사용자 결정). **KIVI 건너뜀**(2a+2b와 직교 + substep 1 KIVIFormat unwired → substep 3 wiring으로 연기).

## §2c 종결 — β-경계 재census 결과 (2026-06-03)

- **워크플로우 `wf_57ad2f94-a93`**: census(3각도: consumer-ripple / SSOT-scope / scaffolding-state) → synthesis → adversarial(3 lens: secretly-β / trivial-noop / behavior-equiv). 결과 `verdict_holds=false`, `refuted_count=1` (trivial-noop이 `hybrid_minimal_2c` 판정을 SSOT 인용으로 refute).
- **판정 = cosmetic throwaway**: `determine_pressure_level`(cache_manager.rs:153) → pressure/ free fn 추출은 byte-identical·consumer ripple 0로 **기계적으로 가능하나** β에서 통째 폐기될 중간물. **4중 β-불일치**(소스 직접 검증):
  1. layer: pressure/**L3** vs `LocalPressureSource` session/**L4** (`design_v2.md:191`)
  2. 반환: `PressureLevel`(=`llm_shared::Level`) vs `Pressure(0–100)`
  3. 로직: 단일 함수 vs `Pressure::band()` 분할 (`pipeline.rs:44` 이미 placeholder draft, 주석이 "구 `determine_pressure_level` day-1 carry"라 자인)
  4. 입력: memory-only vs memory+thermal+energy 융합 (**`docs/adr/0002:53`** "현 `LocalPressureSource`= 구 `cache_manager.rs::determine_pressure_level`, memory만 계산 → 전 센서 융합으로 확장해야")
- **substep(2) 정의 밖**: **`docs/adr/0001:203`** substep(2) = "CacheManager → **EvictionStage/KvMergeStage/SwapDispatchStage** 분해" — `determine_pressure_level`은 이름조차 미포함. dispatch *진입 전* level 선택 로직이지 Stage가 아니다. 추출해도 SRP 이득 0(CacheManager가 `threshold_bytes` 필드 + 유일 호출처 line 239 잔류).
- **사용자 결정 (AskUserQuestion)**: **substep(2)를 2a+2b로 종결**. `determine_pressure_level → LocalPressureSource` 마이그레이션은 **β(LocalPressureSource 신설 시점)에서 trait 배선과 일괄 처리**.
- **종결 근거 (grounding)**: 세 Stage가 기존 핸들러로 이미 존재(`EvictionHandler`/`D2OHandler`=KvMerge/`SwapHandler`=SwapDispatch) + 2a/2b가 마지막 인라인 eviction 로직 carve-out → `cache_manager.rs`(1558 LOC)는 thin dispatcher(전 pub fn이 `execute_dispatch`/`run_policy_eviction`로 깔때기).

## 다음 — substep (3) (forward-path generic→trait object flip)

- **(3a)** eval-LL island boundary flip (device, ripple 격리 검증).
- **(3b)** `LlamaModel::forward` 진입점 → `&[Arc<dyn KVCacheFormat>]` + loop 직전 concrete 흡수.
- **(3c) `LlamaLayer::forward` + `attention_into` 흡수 = ★perf 위험 crux** — layer-tier dyn dispatch vs `INV-HOTPATH-DISPATCH`. 해소 가설: concrete-handle 흡수(④-a) + plan `AttentionVariant` enum static 유지(④-b 연기). **production hot path = plan enum이라 trait는 cold path** 예상이나 **호스트 GPU 부재로 device round 전까지 perf 불확정 → ADR-0001 §6.5 revoke 가능**.
- **(3d)** plan path 정렬 (`execute<C>` 흡수, `build_plan` concrete 2갈래 유지).
- **게이트 (§9.1)**: layer-tier = **device bit-identical + perf(avg_tbt)**. (3)은 regression이 *예상 가능*(layer-tier vtable)이라 Δ>+3% 시 root-cause 의무가 아니라 예측 범위. **(3) 세부 cut point은 진입 시 type-flip ripple 보고 결정**(미리 고정=speculative, §9.1 R3).
- **device 환경 블로커** (잔존): S25 adb(`R3CY408S5SB`) OK. Jetson은 ssh 키 미등록 + IP `165.132.107.73:4121` stale + cargo-zigbuild 미설치 → 사용자 개입 필요(사용자가 Jetson skip 가능). device gate는 substep 2의 `-n 256+ --memory-threshold-mb 999999 --eviction-target-ratio 0.5` 강제 방법 재사용 + cool-state tbt.

## Landmines / 교훈
- **device gate eviction 발화 조건**: 32-tok은 MIN_EVICT_TOKENS(64) 미달로 vacuous. **-n 256+ & `--memory-threshold-mb 999999` & `--eviction-target-ratio 0.5`** 로 강제해야 2b 코어 발화.
- **thermal이 tbt를 지배**: S25는 sustained load + 빌드 직후 발열로 tbt가 2배 변동(none 31→73). tbt 비교는 **cool-state first-run** 또는 cooldown 필수. bit-identical(결정적)이 신뢰 가능한 게이트.
- **device run gotcha**: `--prompt`은 단일 토큰만(run_device.py adb 공백분리). CLI는 **subcommand 구조**(`--greedy eviction sliding --window N`), 구 `--eviction-policy` flat flag 아님. tokenizer-path 명시 필수.
- **baseline 빌드**: detached checkout(`git checkout 9b350609`)으로 build+deploy 후 `git checkout master` 복원. gitignored devices.toml/hosts.toml 유지됨. worktree는 gitignored config 부재로 비권장.
- **세션 외 untracked `arch/pipeline/`**: companion 문서, 내 작업 무관(transcript Write 0 확인), 미커밋·미삭제.
- **커밋 금지 untracked**: `.antigravitycli/`, `.claude/scheduled_tasks.lock`, `papers/.../microbench_*`, `.agent/todos/handoff_microbench_*.md`. 명시 파일 add(`git add -A` 금지).
- **§9.1 SnapKV 각주 갱신 권고**(Architect, 미반영): SnapKV 구현체 0 → device 게이트 4구성(none/sliding/h2o/d2o)+KIVI로 재정의 문서화.

## 자기점검
- 진입 문장: ✓ "α-K substep (3) 진행" (forward-path generic→trait flip; 3c=perf 위험 crux, device round 필요)
- 왜 멈췄나: ✓ substep(2) 종결 결정 후 checkpoint — 2c는 β-경계 재census가 throwaway 판정, 사용자가 substep(2) 종결+2c β흡수 선택. 다음 substep(3)은 layer-tier perf crux라 device round 환경 setup 동반 → 새 세션 진입.
- 최대 landmine: ✓ substep(3) 3c의 layer-tier dyn dispatch perf 회귀(INV-HOTPATH; 호스트 GPU 부재로 device 전 불확정, ADR §6.5 revoke 가능) + device 환경 블로커(Jetson ssh/zigbuild)
- 검증 게이트: ✓ 2a/2b host PASS 수치 + S25 bit-identical 4/4 PASS commit/handoff 기록
- device 가용: ✓ S25 USB(`R3CY408S5SB`). Jetson ssh alias 미해결(165.132.107.73:4121 45일 stale, 사용자 요청 시 복구). cargo-zigbuild 미설치.
