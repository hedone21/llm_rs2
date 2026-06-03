# Handoff: Phase α-K substep (3d) 진입 — prefill flip + plan 평가 + World A eviction 발화 seam

**작성**: 2026-06-03 (작성자: 메인 세션)
**HEAD**: `0c81badf docs(handoff): Phase α-K (3c-evict) 완료 체크포인트 (다음 3d)`
**브랜치**: `master` (origin 동기화 — `0c81badf` push 완료)
**다음 세션 진입 문장**: **"α-K substep (3d) 진행"**

> 배경 전체(census §1·SSOT 위치 §2·cut-point 이력 (3a~3c) §3·device cross-build 절차 §5)는 **`handoff_alpha_k_substep3_census_2026_06_03.md`** 참조. 본 문서는 (3d) 진입에 필요한 것만 추린다. SSOT = `arch/pipeline_stage_design_v2.md` §9.1 + §9.1-EVICT(-DECISION). ADR = `docs/adr/0001-kv-dispatch-paradigm.md` §8.3. 트랙 메모리 = [[project-pipeline-alpha-k]].

---

## TL;DR
(3c-evict)까지 완료 — eviction `evict(&mut KVCache)` → `compact(keep, merges)` flip 의 **함수-레벨 경로(`EvictionPolicy::plan_keep` + host 등가성 게이트)**를 순수 additive·unwired 로 안착(`2f014163`). 다음 = **(3d)**: prefill path flip + plan 평가((3p) 분기 결정) **+ World A eviction 발화 seam 구축(Phase 4-4)** → (3c-evict)가 defer 한 device bit-identical eviction 게이트. **왜 멈췄나**: (3d)는 단순 flip 이 아니라 *World A 에 한 번도 없던 eviction 발화 seam*을 처음 설계·구축해야 해서(아래 R6 F1) 별도 설계 라운드가 필요 — 세션 경계로 분리.

---

## 진행 상태 (검증된 게이트)
| substep | 상태 | commit | 게이트 |
|---|---|---|---|
| (1) trait 정의 | ✅ | `9d858cf9` | host |
| (2) CacheManager→Stage | ✅ | 2a+2b | S25 PASS |
| (3a) write_kv backend 흡수 | ✅ | `5ea8ad47` | host |
| (3b) write_kv cast scratch | ✅ | `3bc03e59` | host |
| (3c-fwd) forward_into_fmt flip | ✅ | `c2b05aff` | **S25 device bit-identical** (F16/Q4_0 rpcmem 32tok) |
| **(3c-evict) plan_keep + compact 등가** | ✅ | `2f014163` (+SSOT `cd917319`) | **host: compact_parity 9/9** (4 정책 × 3 dtype) + eviction 124 + standard_format 10 회귀 0, fmt+clippy(`--workspace -D warnings`) clean |
| (3d) | ★다음 | — | device |
| (3p) plan-flip / (4) KVCacheOps 폐기 | 대기 | — | device (perf crux) |

(3c-evict) = `EvictionPolicy::plan_keep(current_pos, target_len, importance) -> Option<(Vec<usize>, Vec<Merge>)>` default(`None`) + 4 정책 override(sliding/h2o/streaming/no_eviction, prefix-포함 ascending keep-list). in-place `evict*`와 공존(refactor 0). 테스트 = `engine/src/pressure/eviction/compact_parity.rs`.

---

## 다음 작업 (3d) — 세 갈래가 한 substep
1. **World A eviction 발화 seam 구축 (Phase 4-4, 핵심·선결)** → 검증: (3c-evict) `plan_keep→compact`를 live 발화시켜 S25 OFF/ON **bit-identical**(eviction 발화 + `--no-gpu-plan`, F16/Q4_0). **단 OFF/ON 은 반드시 같은 World·같은 타이밍**(R6 F3).
2. **prefill path flip** (forward.rs prefill → trait object) → 검증: host build + S25 device bit-identical(기존 forward 게이트 재사용 §census).
3. **plan 평가 = (3p) 분기 결정** → plan path 를 `KVCacheOps` 유지할지 trait flip 할지 판단(결정만, 구현은 (3p)).

설계 결정 필요(구현 전 Architect 라우팅 권장): **EvictionStage→cache 도달 방식** = (a) `EvictionStage::before_step` 에 cache handle 주입(StepCtx 확장) vs (b) `Forward::try_evict` 를 DecodeLoop 에 배선 + fmt-aware. + `is_standard_happy_path` 가 eviction 을 World A 로 받아들이는 정책(γ 가 미룬 부분).

---

## Landmines / 미해결 (R6)
- **F1 — World A eviction 발화 seam 자체가 미구축 (3d 의 진짜 난점).** `EvictionStage::before_step(&StepCtx)`(decode_loop.rs:159)는 캐시 접근 권한 없음 + `build_standard_loop` 는 `NoOpEvictionStage` 만 주입 + `Forward::try_evict`(model_forward.rs:542)는 DecodeLoop 이 **호출조차 안 함**. → World A 에서 eviction 은 한 번도 발화한 적 없다. 이걸 짓는 게 (3d)/Phase 4-4.
- **F3 — 발화 타이밍.** World A DecodeLoop = forward **전**(before_step), World B legacy = forward **후**(`run_auto_eviction` legacy:2278). bit-identical device 게이트를 World 분할로 구성하면 한 토큰 어긋나 발산 → **OFF/ON 을 같은 World·타이밍 안에서만** 비교. (γ 가 (3c-evict) device 게이트를 (3d)로 미룬 이유.)
- **try_evict fmt 분기 미결.** fmt active 시 `self.kv_caches` 가 빈 Vec(mem::take→fmt_caches)라 현 `try_evict` 가 no-op. fmt 분기로 `plan_keep → fmt.compact` 호출해야. + active policy 도달 경로 — CacheManager 가 policy 를 EvictionHandler 에 캡슐화 → accessor 추가 또는 try_evict 시그니처 확장 필요.
- **deferred (§9.1-EVICT-DEFER, (3d) 범위 밖)**: H2O+ per-head(`plan_keep`=None) → (3c-evict-perhead) per-head compact seam / D2O 가중 merge → (3c-evict-d2o) `apply_merges` 가중 확장 + D2OHandler keep-list 산출.
- **기존 부채(내 변경 무관)**: `--all-targets` clippy 12건 = 기존 spec test(test_inv_rpcmem_006/test_model_forward_parity:286/test_qnn_*), MEMORY [[project-pipeline-alpha-k]] line 40 기록. 표준 게이트(`--workspace -D warnings`)는 clean.
- **커밋 금지 untracked**: `arch/pipeline/`(companion, 내 작업 무관·미커밋·미삭제), `.antigravitycli/`, `.claude/scheduled_tasks.lock`, `papers/.../microbench_*`. 명시 파일만 add (`git add -A` 금지).

---

## 자기점검
- 진입 문장 한 줄로 시작 가능? ✓ "α-K substep (3d) 진행"
- 왜 멈췄나 명시? ✓ World A eviction 발화 seam 신설 설계 라운드 필요(F1), 세션 경계
- 최대 landmine 표면화? ✓ F1(seam 미구축) + F3(타이밍) + try_evict fmt 분기
- 검증 게이트 수치/명령? ✓ compact_parity 9/9, S25 OFF/ON bit-identical(eviction 발화 + --no-gpu-plan)
- 길이 적정? ✓ 배경은 census handoff 링크로 분리
