# ADR-0003 구현 진행 원장 (/loop dynamic)

**대상**: ADR-0003 + **ADR-0004** 구현 완수
**SSOT**: ADR-0003(확장 메커니즘) + **ADR-0004(KVCacheStage trait 설계)** + `arch/pipeline_stage_design_v2.md`
**시작 HEAD**: `d331d01b` / 현 HEAD: `5f81bace docs(adr): ADR-0004 …`
**진입**: `/loop` dynamic 모드, 완료 시 자가 종료

> ## ▶ 재개 진입점 (compact 후 여기서 시작)
> **설계 전부 확정·커밋됨**(M0 8c23a72a / M1 136f7cdd / 설계 5f81bace). 분기 F1~F6 + 네이밍 닫힘 (아래 "M2-B 설계 분기").
> **M2-B① ✅ 완료**(`caeca4f2`): technique-api 가 ADR-0004 표면(KVCacheStage/StageCtx 9-accessor/KVCachePlan/KeepSpec/WeightedMerge/StageParams/find_stage)으로 재구성됨. 표면은 workflow `wf_21533739` 가 6기법 all_expressible·dyn-safe 적대 검증 완료.
> **사용자 결정**(iter-5): (1) **지금 World A→B 전환(역어댑터)** — ②a→②b. (2) **name-key = 전부 CLI 이름으로 통일**(`policy.name()` 출력도 CLI 이름; 주로 `"sliding_window"→"sliding"`, 의존 테스트 4곳 갱신).
> **M2-B②a ✅ 완료**(`2676acd2`): 엔진→technique-api(+linkme) 의존 + `EvictionPolicyAsStage` 어댑터 + 3 빌트인(sliding/streaming/h2o) `KV_CACHE_STAGES` 등록. **등록만(unwired)**, 프로덕션 무변경. test 1223/0.
> **다음 = M2-B②b**: name 통일(`SlidingWindowPolicy::name()` "sliding_window"→"sliding" + 의존 테스트 `sliding_window.rs:247`·`cache_manager.rs:762,1163`·`eviction_handler.rs:267` 갱신) + `KVCachePlan` executor(LayerWide+merges→apply_merges(가중)+compact_keep_positions; PerHead→_for_head; PerHead+merges→bail) + `StageBackedPolicy{stage}` impl EvictionPolicy(plan→executor 실행, World B) + session.rs:620-650·build_bench_loop.rs:88-117 의 **sliding/streaming/h2o 3 arm 만** `find_stage`→make→StageBackedPolicy fallback 으로 교체(h2o_plus arm·d2o if·bail 잔류; bail 메시지·streaming window 유도·protected_prefix 기본값 match 는 caller 잔류) + startup self-test(release fail-fast). 게이트 = `cargo test -p llm_rs2 --lib -- --skip backend::opencl --skip memory::opencl` (≥1223 passed, 0 failed) + build + fmt(내 파일) + clippy(--workspace, --all-targets 금지) + compact_parity + ② 완료 시 release smoke. **무회귀 invariant**: 버퍼 bit-identical(compact_parity transitively)·target_len 의미·NoOp 가드(run_policy_eviction 선처리)·undershoot·score dispatch·name() 보존(CLI 이름으로 일관).
> 재개 명령 예: "M2-B②b 진행".

---

## 기준선 (M0, 2026-06-05)

- **게이트 명령**: `cargo test -p llm_rs2 --lib -- --skip backend::opencl --skip memory::opencl`
- **기준선 결과**: `ok. 1220 passed; 0 failed; 39 filtered out`
- **GPU 테스트 제외 사유**: 이 호스트는 OpenCL 디바이스 부재(CLAUDE.md). `backend::opencl::*`는 flaky-fail(런마다 19~24개), `memory::opencl::unified::test_map_write_unmap_cycle`는 null-ptr **SIGABRT로 테스트 프로세스 전체를 죽임**. 둘 다 환경적 — 회귀 아님. 따라서 게이트에서 제외하고 "skip 셋 밖 실패 0"으로 회귀를 판정한다.

## 게이트 (각 마일스톤 커밋 전)

1. **빌드**: `cargo build -p llm_rs2` (dev) OK. + **release 빌드는 M2·M5에서만** (`cargo build --release -p llm_rs2`) — fat-LTO 빌드가 비싸 매 마일스톤 반복은 비효율. release가 의미 있는 지점(linkme fat-LTO 생존 smoke)에서만 수행. ← prompt 게이트의 *취지*(빌드+LTO 생존) 유지하며 cadence 최적화 (의도적 판단).
2. **테스트**: 위 게이트 명령 → `0 failed` (skip 셋 밖). 떨어지면 STOP.
3. **fmt**: `cargo fmt --all`.
4. **clippy**: `cargo clippy --workspace -- -D warnings` (`--all-targets` 금지) clean.
5. **M2 이후 release smoke**: release bin으로 각 정책 이름 해석(`--eviction-policy h2o` 등) "Unknown policy" bail 안 함 — linkme가 `--gc-sections`에서 생존했는지. 실패 시 ADR §4 build.rs codegen 폴백.

## 가드레일

- 한국어. Conventional Commits + `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **`git add -A` 금지** — 명시 파일만. push 금지.
- 커밋 금지 untracked: `.antigravitycli/`, `.claude/scheduled_tasks.lock`, `papers/**/microbench_*`, `.agent/todos/handoff_microbench_*.md`, `arch/pipeline/`, `docs/refactoring_report/`(별건).
- 신규/이동 모듈 = no-`mod.rs`. 외과적 변경. 테스트 약화/우회 금지.

## 마일스톤 체크리스트

- [x] **M0** 기준선(1220) + 원장 + ADR-0003·README 커밋
- [x] **M1** `crates/technique-api/` 신설 — `EvictionPlan`(planning trait) + `Merge` + `PolicyParams` + `EvictionPolicyReg` + linkme `EVICTION_POLICIES` distributed_slice + `find_eviction`/`registered_names`. workspace member 추가, linkme 0.3 dep. **엔진 의존 0**(단방향). 테스트 2/2(dummy 등록·조회). 커밋 예정.
- [~] **M2** ⏸ **STOP — 사용자 결정 대기** (아래 "M2 fork"). 조사 완료: `compact_parity.rs`(통과 중)가 sliding/streaming/h2o/no_eviction × {F32,F16,Q4_0}에서 plan_keep→compact ≡ in-place evict 증명 → 어댑터 경로 검증됨. 그러나 **h2o_plus(plan_keep→None)·d2o(EvictionPolicy 아님)는 planning 레지스트리에 안 들어감** → registry scope 결정 필요.
- [ ] **M3** 정책 impl을 `crates/techniques/<name>/` per-crate 이전 + linkme 등록, workspace glob `crates/techniques/*`, bin 의존(D4 1줄/기법), 더미 crate로 "폴더만 추가" 검증
- [ ] **M4** d2o `plan_keep` 이전 + `Merge` 가중치 필드. **동등성 테스트 선작성 필수**, 못 세우면 STOP+human-review 플래그 (senior-implementer 위임 가능)
- [ ] **M5** 기여자 문서 "기법 추가법"(hook·시그니처·등록 + 동작 예제 crate, 컴파일·등록 게이트)

## 완료 조건

M1·M2·M3·M5 커밋 + 전체 `/sanity-check` green + release self-test 통과 + 본 체크리스트 done.
(M4 = 동등성 테스트 통과 done, 또는 "human-review 필요" 명시 플래그.) → 최종 보고 + `notify-send` + 루프 종료.

## Iteration 로그

- **iter-1 (M0, 2026-06-05)**: repo 상태 확인(HEAD d331d01b, 7 worktree). baseline 측정 — GPU flaky/SIGABRT 발견 → 게이트를 "GPU skip + skip밖 실패 0"으로 정의. baseline=1220 passed/0 failed. 원장 작성. ADR-0003+README+원장 커밋(8c23a72a). → 다음 iter: M1.
- **iter-2 (M1, 2026-06-05)**: 사용자 체크인("중간된거야?")으로 M0↔M1 경계 일시정지 후 재개. **설계 발견**: `plan_keep`(planning 표면)이 이미 코드에 스캐폴딩(Sliding/H2O/Streaming/NoEviction 구현, unwired). 단 **H2O+(per-head)는 `None` 반환, D2O는 EvictionPolicy 아님** → ADR-0003 §D2 "h2o+/d2o 덮음"은 낙관적. M1 = technique-api crate(planning trait `EvictionPlan`, 엔진 의존 0). 게이트: technique-api 2/2, clippy workspace clean, 엔진 1220/0 무회귀. `cargo fmt --all`이 무관한 htp_fastrpc.rs(기존 fmt drift) 122줄 건드려 revert(외과적 변경). 커밋 136f7cdd. → 다음 iter: M2.
- **iter-3 (M2, 2026-06-05)**: M2 조사 — `compact_parity.rs`가 이미 baseline(1220) 안에서 통과 중, plan_keep→compact 경로가 4개 정책 × 3 dtype 에서 bit-identical 증명됨(어댑터 경로 안전). **STOP**: h2o_plus/d2o 가 planning 레지스트리에 안 들어가 registry scope 가 ADR 미명세 → 사용자 결정 대기(M2 fork). 코드 변경 없음(조사만). 원장만 갱신.
- **iter-4 (M2-B①, 2026-06-05)**: 사용자 (B) 선택 → 재설계(ADR-0004) 확정 후 `/loop` 재개. **코드 쓰기 전 적대 검증**: workflow `wf_21533739`(8 agent, 680K tok)로 6기법(sliding/streaming/h2o/no_eviction/h2o_plus/d2o)이 `KVCacheStage::plan` 단일 표면으로 표현 가능한지 + 각자 StageCtx 에서 뭘 읽는지 기법별 병렬 검증 → **all_expressible=true, dyn_safety_ok=true**. 9-accessor StageCtx 합집합 도출. `crates/technique-api/src/lib.rs` 를 ADR-0004 표면으로 재작성(EvictionPlan→KVCacheStage 등). StageParams d2o 필드는 M4 연기(open question). 게이트: technique-api 2/2, clippy workspace clean, 엔진 회귀 1220/0. 커밋 `caeca4f2`. → 다음 iter: M2-B②(엔진 배선).
- **iter-5 (M2-B②, 2026-06-05) — ⏸ STOP, 사용자 결정 대기**: 엔진 배선 **코드 쓰기 전 적대 검증** workflow `wf_108af60f`(4 agent, 338K tok) — 프로덕션 호출그래프 매핑 + 역어댑터 가설 검증. 발견:
  - **프로덕션 eviction = in-place `evict*`** (EvictionHandler→run_policy_eviction→per-cache policy.evict*). `plan_keep`/compact 는 **unwired**(compact_parity 테스트만). → KVCacheStage 배선 = **3정책 프로덕션 decode 를 World A(in-place)→World B(plan→compact) 전환**.
  - **`should_evict` 는 프로덕션 dead**(테스트 전용). 트리거는 run_policy_eviction 의 target_len/MIN_EVICT(64) 가드 + 메모리 압력.
  - **ledger "4 정책" 은 실제 3개**: `plan_keep` 구현 = sliding/streaming/h2o/no_eviction 4개지만 no_eviction("none")은 match 밖(happy-path early-return). 따라서 session.rs:620-650 match 에서 **sliding/streaming/h2o 3 arm 만 제거 가능**. **h2o_plus 잔류**(plan_keep→None, per-head; head_score source=F5 미완이라 degenerate 회귀 위험 → ⑤로 deferred). d2o if-선분기 잔류(M4). bail 잔류.
  - 역어댑터(StageBackedPolicy) verdict = **validated_with_changes**. risks: merge-uniform(Q4=M4), dead-code, name-key.
  - **stop_flag 2개**: (1) ADR-0004 §2/§4 가 "엔진이 plan→compact 실행"은 mandate 하나 **역어댑터로 프로덕션 World A→B 전환을 지금 할지**는 미명세(decode 경로 첫 pivot). (2) **name-key**: CLI "sliding" vs `policy.name()` "sliding_window" 불일치(reg key 결정 + name() 출력 보존 여부).
  - 코드 변경 0(검증만). 사용자 결정 후 ②a(등록+self-test, 무변경)→②b(match arm 교체) 재개.
- **iter-6 (M2-B②a, 2026-06-05)**: 사용자 결정 — (1)지금 World A→B 전환(역어댑터), (2)name-key 전부 CLI 통일. ②a 구현(순수 additive): engine→technique-api(+linkme 0.3) dep + `eviction/stage_registry.rs` 신설(`EvictionPolicyAsStage` 어댑터 = plan_keep→KVCachePlan 위임 + uniform Merge→WeightedMerge) + 3 빌트인(sliding/streaming/h2o) `#[distributed_slice(KV_CACHE_STAGES)]` 등록(reg key=CLI 이름). **등록만, unwired** — 프로덕션 여전히 in-place evict. 게이트: build OK, test 1223/0(신규 3: 등록·어댑터 faithful×2), clippy workspace clean, fmt 내 파일만. 커밋 `2676acd2`. → 다음 iter: M2-B②b(name 통일 + executor + StageBackedPolicy + match arm 3 교체 + self-test).

## M2-B 재설계 — ground truth (workflow wf_a9f025a7, 4축 surface)

사용자가 (B) 선택 → `EvictionPlan` 을 per-head keep + 가중 merge 까지 확장 재설계. 실측 사실:
1. **두 확장은 직교**: h2o+ = **per-head keep**(head마다 다른 토큰, merge 없음), d2o = **가중 merge**(layer-wide keep). 동시에 둘 다 필요한 정책은 현재 없음(d2o+h2o+ 융합 = 미래).
2. **가중 merge 적용 코드는 반쯤 존재**: `StandardFormat::apply_merges`(standard_format.rs:481)가 `w_c·into + Σ w_e·from` 적용하나 **현재 uniform weight**(`Merge{into,from}`에 가중치 없음). d2o 의 진짜 Eq.11 weight 는 `scatter_reduce_merge_layer_wide`(d2o_handler.rs:555)에 별도. **F32/F16만, Q4_0 merge disabled**.
3. **per-head 실행 primitive 존재**: `compact_keep_positions_for_head`(kv_cache.rs:905). per-head plan 실행 가능.
4. **★ d2o가 어려운 핵심**: (a) **stateful EMA**(τ_t = β·max+()·prev, `Mutex<D2OState>`, 호출 간 누적 — 순수함수 아님), (b) **raw K snapshot 필요**(cosine nearest, layer-wide concat), (c) **per-layer**(layer_ratios 예산). 즉 "pure plan(scores)→plan" 모델에 안 맞음 — planner가 K + 상태 + layer ctx 를 받아야 함. 반면 h2o+는 head_importance만 있으면 순수.
5. head_importance 는 accumulator가 계산하나 **session.rs가 현재 forward 안 함**(flat만) — h2o+ plan 경로 켜려면 session 배선 추가.

## M2-B 설계 분기 (grill 진행 중)
- **F1 ✅ (가) 확정**: 단일 plan-returning trait. 입력 = `&KVCache`(읽기) + scores(flat/head) + budget(target_len/layer_idx). **상태는 impl**(Mutex, d2o EMA). 엔진이 반환 Plan을 `compact`로 실행(캐시 변형 독점). "순수성" 논거는 약해서 기각(캐싱/리플레이 YAGNI, C-ABI는 stateful 객체도 OK) — 진짜 축은 **plan-returning(엔진 변형) vs self-mutating(플러그인 변형)**. (a)EMA=impl 상태로 해소, (b)K=정적단계엔 `&KVCache` 읽기 borrow(복사 스냅샷 불필요; **스냅샷은 미래 `.so` C-ABI 경계에서만 필요** — K 입력을 "읽기 접근" 추상으로 두면 borrow→flat 교체 가능), (c)layer=스칼라.
- **F2 ✅**: `KVCachePlan { keep: KeepSpec(LayerWide(Vec)|PerHead(Vec<Vec>)), merges: Vec<WeightedMerge{into,into_weight,from:Vec<(pos,w)>}> }`. keep=배타 enum, merge=직교 필드. new_pos는 keep.len() 도출.
- **F3 ✅**: `Merge`→`WeightedMerge` 통합. 가중치 plan에 baked, `apply_merges`가 plan 가중치 사용(현 uniform 대체).
- **F4 ✅**: per-head+merge 융합(d2o+h2o+)은 타입상 표현 가능, executor는 당분간 `bail!`(promotion-trigger).
- **F5**: head_importance session forward 배선 추가(현재 flat만) — M2/M3 구현 항목.
- **F6 ✅ (강제된 귀결)**: trait이 `technique-api`에 살아 `&KVCache`(엔진 타입) 직접 못 받음(단방향 의존). → `technique-api`가 정의하는 읽기 추상 **`StageCtx`**(geometry+scores+dequant K accessor)를 받고 엔진이 `&KVCache` 위로 impl. 정적=borrow, 미래 `.so`=C accessor/flat. forward-compatible.
- **네이밍 ✅** (ADR-0004, CONTEXT.md 반영): `EvictionPlan`(trait)→`KVCacheStage` / 반환struct→`KVCachePlan` / `Merge`→`WeightedMerge` / `EvictionPolicyReg`→`KVCacheStageReg` / `EVICTION_POLICIES`→`KV_CACHE_STAGES` / `PolicyParams`→`StageParams` / `find_eviction`→`find_stage`. `KVCacheFormat`의 형제. 세션 `EvictionStage` deprecated, legacy `EvictionPolicy` phase-out.

**설계 확정 — 결정 박제**: ADR-0004 신설, ADR-0003 §D2 정정, CONTEXT.md(KVCacheStage/KVCachePlan 항목 + Flagged ambiguities) 갱신.

## M2(B) 구현 plan (설계 확정 후 — 루프 재개 대상)
1. ✅ **(iter-4, `caeca4f2`)** M1 technique-api rename + 재구성: `KVCacheStage` trait(`fn name`, `fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan>`) + `KVCachePlan`/`KeepSpec`/`WeightedMerge` + `StageCtx` 읽기 추상 + `KVCacheStageReg`/`KV_CACHE_STAGES`/`StageParams`/`find_stage`.
2. 엔진: `StageCtx`를 `&KVCache` 위로 impl + `KVCachePlan` executor(LayerWide→apply_merges(가중)+compact_keep_positions / PerHead→compact_keep_positions_for_head / PerHead+merges→bail) + 빌트인 4정책을 `KVCacheStage`로(어댑터 또는 직접) + `KV_CACHE_STAGES` 레지스트리로 session.rs:621·init.rs match arm 제거 + startup self-test.
3. compact_parity를 가중 merge·per-head로 확장.
4. (M4) d2o를 `KVCacheStage`로 재구현(plan에 nearest+Eq.11 가중치, EMA=impl Mutex, K=StageCtx). 기존 D2OHandler와 동등성 테스트 선행, 미확립 시 STOP.
5. (F5) head_importance session forward 배선.

## 검증된 표면·스윕 intel (workflow wf_21533739, iter-4)

**M2-B②~④ 구현의 SSOT 보조** — 6기법 적대 검증 + match-site 전수 스윕 결과. (원본: workflow 결과, 680K tok)

### 검증된 9-accessor StageCtx (전부 dyn-safe, 코드 근거 확인됨)
| accessor | 소비 기법 | 엔진 impl 원천 |
|---|---|---|
| `current_pos(&self)->usize` | 6개 전부 | `KVCache::current_pos()` kv_cache.rs:958 |
| `target_len(&self)->usize` | sliding/streaming/h2o/d2o | EvictionHandler ratio→len 환산(eviction_handler.rs:48-51)을 ctx impl 이 필드로 보유 |
| `layer_idx(&self)->usize` | d2o | 엔진이 layer 순회 중 주입(protect 판정용; budget 은 target_len 으로 흡수) |
| `importance(&self)->Option<&[f32]>` | h2o/d2o | HandlerContext.importance(pressure.rs:53-54) |
| `n_kv_heads(&self)->usize` | h2o_plus/d2o | `KVCache::kv_heads()` kv_cache.rs:122 |
| `head_dim(&self)->usize` | d2o | `KVCache::head_dim()` kv_cache.rs:126 |
| `head_score(&self,kv_head,pos)->f32` | h2o_plus | head_importance row-major(cache_manager.rs:594-596) 인덱싱을 엔진이 내부화 |
| `has_head_scores(&self)->bool` | h2o_plus | h2o_plus 폴백(scores 부재) 복원 — false 시 LayerWide degenerate |
| `dequant_k(&self,pos,head,out:&mut[f32])` | d2o | dequantize_k(d2o_handler.rs:414-441) dtype 분기 위임. **out-param 필수**(슬라이스 반환=Q4 임시버퍼 수명 불가, Vec=할당 폭증) |
- keep 모양: sliding/streaming/h2o/no_eviction/d2o = **LayerWide**, h2o_plus = **PerHead**. merge: d2o만 가중(나머지 빈 Vec).
- executor 매핑(엔진): LayerWide+merges→`apply_merges`(가중)+`compact_keep_positions`(kv_cache.rs:863) / PerHead→`compact_keep_positions_for_head`(kv_cache.rs:905) / PerHead+merges→`bail!`(promotion-trigger).

### ★ M4(d2o) STOP 게이트 — ADR-0004 §4 개정 필요 (설계 모순, 사용자/PM 결정)
- **Q4_0 merge 불일치**: ADR-0004 §4 = "Q4_0 merge 비활성 유지"인데, 현 D2OHandler 는 `scatter_reduce_q4`(d2o_handler.rs:585-594)로 **Q4_0에서 실제 가중 merge 수행 중**. executor `apply_merges`(standard_format.rs:481-535)는 Q4_0 silently 스킵(:532). d2o 를 plan→executor 로 옮기면 Q4_0 KV 에서 merge drop → 기존 경로와 bit-level 불일치 → §4 "동등성 선행, 미확립 시 STOP" 위반.
- **두 갈래**: (A) §4 고수(Q4 비활성)→d2o-on-Q4_0 은 keep-only 로 paper(Eq.11 정렬) 회귀 / (B, **권고**) executor `apply_merges` 를 가중+Q4_0 지원으로 확장하고 §4 "Q4 비활성" 문구를 ADR 개정으로 해제. plan 표면은 이미 dtype-agnostic 으로 옳음.
- **M4 진입 시 이 결정 먼저** → ADR-0004 §4 개정 또는 사용자 확인. 미해결로 d2o-on-Q4_0 진행 금지(STOP).

### M2-B② match-site 스윕 (레지스트리로 제거 대상)
**활성 closed match = 2곳 복제** (drift 위험, 단일 factory 로 통합이 1차 목표):
- (A) `engine/src/session/chat/session.rs` — :604 `if eviction_policy=="d2o"` 선분기 / :620-650 `match eviction_policy.as_str()`(sliding/streaming/h2o/h2o_plus + unknown bail) / :590-594 protected_prefix 기본값 match / :654 `score_based = matches!(...)`.
- (B) `engine/src/session/assembly/build_bench_loop.rs` — :72 d2o 선분기 / :88-117 정책 match(+none 포함, d2o 파라미터는 CLI 아닌 상수) / :57-61 protected_prefix match. **(A)와 거의 동일 손복제** — 미묘차(none arm, d2o param 출처, bail 메시지).
- **부수 분기**(descriptor 속성으로 흡수): score_based(session.rs:654), is_d2o(eval/runner.rs:158, batch/runner.rs:111), prefill 특수화(ppl/runner.rs:705), happy-path "none" 게이트(build_standard_loop.rs:60-71, init.rs:124), protected_prefix 기본값. → descriptor 에 `needs_scores()`/`is_noop()`/`is_merge_based()`/`default_protected_prefix()` 두면 일괄 제거.
- **이름 정의 지점**: `engine/src/session/cli/eviction.rs:71-77` EvictionCmd→name 변환표 — 레지스트리 키와 동기화 필요.
- **D2O 구조 주의**: d2o 는 EvictionPolicy 가 아니라 D2OHandler→Pipeline→`CacheManager::with_pipeline`. 단순 `name→Box<dyn EvictionPolicy>` factory 로 흡수 불가 → ADR-0004 의도대로 d2o 를 **KVCacheStage 로 재구현**(M4)하면 if-선분기 소멸. 그 전(② 단계)엔 4 빌트인만 레지스트리화하고 d2o if-분기는 잔류 가능.
- **별개 축(이번 범위 밖)**: backend match(init.rs:203-424), EvictMethod enum(method.rs:9-14)+resilience/executor.rs:397-463(production dispatch 끊김 — decode_loop.rs:174 가 method 무시하고 CLI 정책 재사용).
- **⚠ legacy generate.rs caveat**: ppl/eval/batch 는 자체 match 없이 *구성된 ctx*(PplRunCtx 등)만 받음. 그 ctx 채우는 정책 구성 match 는 메인 트리에 없고 `.claude/worktrees/b5_trait_extension/engine/legacy/generate.rs` 에 추정 → **전수 OCP 완성하려면 그 파일도 레지스트리 호출로 전환 필요**(별도 검수). MEMORY.md 의 `engine/legacy/generate.rs` 경로는 메인 트리 기준 stale.

### open questions (M3+ 재검토)
- StageParams 비대화(공용 struct 미사용 필드 vs per-technique opaque params) — 현재 5필드 유지, M4 에서 d2o 필드 추가 시 재결정.
- h2o_plus 폴백 정책: `has_head_scores()=false` 시 LayerWide(sliding 등가) vs LayerWide(H2O 등가) 무엇 반환할지 명세 필요.
- 레지스트리 산출 단위: `Box<dyn KVCacheStage>` vs 완성된 `CacheManager`(needs_scores 장착 등 조립은 레지스트리 밖 공통 후처리).

## (구) M2 fork — registry scope (※ B 선택으로 대체됨)

## M2 fork — registry scope (※ B 선택으로 대체됨)

planning 레지스트리(`EVICTION_POLICIES` = `EvictionPlan`)는 plan_keep 가 Some 인 정책만 어댑터로 구동 가능. 검증된 4개(sliding/streaming/h2o/no_eviction)는 OK. **h2o_plus**=per-head divergence 로 `plan_keep`→None, **d2o**=`EvictionPolicy` 아님(가중 scatter-merge, `Merge` 가중치 부재=M4). ADR-0003 §D2 "covers h2o+/d2o" 는 코드와 불일치(낙관적).
- **(A)** 4개만 레지스트리화. h2o_plus·d2o 는 엔진 내부 특수경로 유지(부분 OCP, 즉시 ship). ADR §D2 정직하게 축소.
- **(B)** `EvictionPlan` 확장: per-head keep-list(h2o_plus) + 가중 merge(d2o, M4 선당김). 완전 OCP, 큰 trait 변경.
- **(C, 권장)** A 즉시 + h2o_plus/d2o 를 promotion-trigger 로 문서화(외부 per-head/merge 플러그인 실수요 등장 시 B). ADR §D2 정정. ADR-0002 의 source-tag promotion-trigger 와 동일 YAGNI 패턴.

## 미해결 fork (M3에서 STOP+보고)

- **H2O+/D2O 패키징**: planning 표면(`plan_keep` 단일 keep-list)은 Sliding/H2O/Streaming/NoEviction만 덮는다. H2O+(per-head, `plan_keep`→None)와 D2O(가중 merge, EvictionPolicy 아님)는 안 덮임. M3에서 결정 필요 — (가) 더 풍부한 plan(per-head keep-lists + 가중 merge)으로 표면 확장, (나) H2O+/D2O는 technique crate화 보류하고 엔진에 잔류, (다) D2O는 M4(가중 Merge)로, H2O+는 별도. ADR-0003 §D2 갱신 동반.
- **htp_fastrpc.rs fmt drift** (mention-only): HEAD에서 이미 unformatted. M1 무관이라 미수정 — 별도 `style:` 커밋 대상.
