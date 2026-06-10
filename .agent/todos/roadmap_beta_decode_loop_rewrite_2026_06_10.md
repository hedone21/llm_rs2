# Phase β substep roadmap — DecodeLoop 재작성 (확정)

**작성**: 2026-06-10 (메인 세션). 설계 = 워크플로우 `wf_56289e25-f8b` 21-agent(census 6 → 설계 3안 → judge 3-lens → 합성 → substep별 적대 검증 + critic) + **사용자 grill 7건 해소(G1~G7, 아래 결정 기록)**.
**상태**: **확정 — β-1 착수 가능.**
**HEAD**: `428ed2d6` (코드 상태 = `1f93d6b1` 동일, docs-only 2커밋 차)
**SSOT**: `arch/pipeline_stage_design_v2.md` §5/§5.4/§2.1/§9.1 · 진입 handoff `handoff_beta_decode_loop_rewrite_entry_2026_06_10.md`
**검증 감사**: 적대 검증 refuted 3건(β-2/β-4/β-5) + critic 8건 → 본문에 전부 보정 반영(soft "[보정]" 표기).

---

## ⚠️ 선결 사실 — master 가 이미 RED (β-1 이 해소)

`cargo test -p llm_rs2 --test spec test_inv_layer_006` **FAIL (HEAD 직접 재현)**. 원인 = AB-1(`94ef643f`)이 추가한 `decode_loop.rs:45 cache_manager: Option<CacheManager>` concrete 필드가 BANNED substring 검사(test_inv_layer_006.rs:28)에 걸림. 당시 게이트가 lib test 만 돌고 `--test spec` integration target 미실행으로 유출. **[G7 결정] β-1 에서 INV-LAYER-006 spec+test 개정으로 해소**(과도기 필드 명시 허용 + β-3 에서 필드 제거 후 금지 복원).

---

## Grill 결정 기록 (2026-06-10, 사용자 확정)

| # | 결정 |
|---|---|
| **G1** | **AB- mode dispatch owner = β.** AB-2(KIVI)/AB-4(partition)/AB-6(weight-swap) 트랙 일시 동결 — β-4(OneShot Stage+LoopControl) 확정 후 Stage 모델로 직접 구현. AB-2 는 D8 ABI 변경(2026-06-10) 후 계획 유효성 재확인 동반. |
| **G2** | **SSOT normative 문언 3건 전부 승인**: (i) v2:534 "5 hook 흡수" → "2종(DecodeObserver/StopCondition) 흡수 + 3종(StepHook/PhaseHook/LayerBoundaryHook, 발화 경로 0 orphan) 삭제·phase 어휘 승계 + pull형 표면 5종은 on_phase 환원 불가" (ii) §2.1:196 "Forward SUPERSEDED" → 슬림 내부 seam 잔존(3-impl 다형성) (iii) §9 γ 재정의(rename 189ref/53file · nested mod.rs 38개 sweep · argus-eval/chat bin화 · batch orphan 처분). → **β-1 normative 커밋에 포함.** |
| **G3** | **orphan hook 3종 + eval/batch orphan 코드 = β-7 동반 삭제.** spec INV-147/149/150 은 phase 어휘 등가 승계 또는 폐기 표기. |
| **G4** | **ManagerPressureSource/LocalPolicy = defer(seam 만).** β-5 는 LocalPressureSource(memory-only)까지. 엔진측 융합(wire 무확장 가능 — 검증 확인)·로컬 센서 인프라는 β 밖 후속. v2:651 "manager-less 이산 정책 1급" 문언 긴장은 SSOT 에 명시(β 범위 manager-less = memory graded 만 1급). |
| **G5** | **chat max_pos break 마지막 토큰 미포함 거동 = bit-identical 보존**(β-6 거동 고정 테스트 기준). 버그 여부 판단은 β 이후 분리. |
| **G6** | **suspend = loop break 보존**(β-4). Resume 유의미화(pause/park)는 수요 발생 시 후속 — LoopControl seam 이 확장 지점. |
| **G7** | **master RED = β-1 에서 INV-LAYER-006 개정으로 해소**(즉시 hotfix 안 함). |

---

## TL;DR

7 substep 직선 위상(dependency-first 골격, judge 2/3 승자 + 타 안 이식): **계약(β-1) → 빈 배선(β-2) → eviction seam(β-3) → command 분해(β-4) → pressure 일원화(β-5) → chat 수렴(β-6) → v1 삭제(β-7)**. 매 substep 컴파일+실행(branch-by-abstraction §9.1 R3). device 왕복 ≈ 6회. tbt Δ≤+3% 기준은 **항상 α-K frozen 절대값**(54.22/54.04/53.79 ms/tok — rolling 재기준화 금지). eviction 경로는 (γ) 선례대로 host 정식 + β-3 **신규 eviction baseline 동결**(`frozen_baseline_beta_evict_*.md`)이 β-4/5/7 회귀 기준.

| id | 제목 | 게이트 tier | 의존 |
|---|---|---|---|
| β-1 ✅ | 계약·어휘 마감 — PipelineRegistry + 계약 + SSOT 정오표·normative(G2) + INV-LAYER-006 개정(G7) — **완료 2026-06-10** (`c5e0fdae` docs + `9ec108c0` feat + `653f56e5` fix + `ba0b343f` style) | host ✅ | — |
| β-2 | driver phase 배선 — run()/prefill() empty-registry dispatch (run_until_stop 미접촉) | device | β-1 |
| β-3 | World A eviction 발화 seam(F1) — wrap-당김 + stages/kv/eviction.rs UER 경유 + baseline 동결 | host 정식 + device | β-2 |
| β-4 | α-W-3b 흡수 — 매핑표 선행 + CommandSource→CommandDispatcher→LoopControl + evict 4종 OneShot | device | β-3 |
| β-5 | PressureLevel→Pressure(band) ripple 10파일 + LocalPressureSource(happy-path 무주입) | host + device-sanity | β-3, β-4 |
| β-6 | chat 수렴 — 거동 고정 테스트 선행 + run_until_stop 단일 driver 통합 + prev_token 승격 | host 정식 + device 1회 | β-4, β-5 |
| β-7 | v1 표면 삭제 + builder 대체 + TokenSampler 이동 + orphan 삭제(G3) — 최종 게이트 | device (최종) | β-6 |

---

## substep 상세

### β-1 — 계약·어휘 마감 (additive, unwired) — ✅ 완료 2026-06-10

> **완료 기록**: `c5e0fdae`(docs: spec/41 + SSOT §5.2.1 신설·정오표·normative 3건) + `9ec108c0`(feat: pipeline_registry.rs + spec test 11건 + INV-LAYER-006 개정 — master RED 해소) + `653f56e5`(fix: BANNED CacheManager 재추가 — 금지 복원 구조, Tester P1 지적) + `ba0b343f`(style: fmt-dirty 10파일). Tester 독립 검증: lib 1261→1267(+6 registry unit, 회귀 0), spec test 11/11, full suite GREEN(GPU 부재 21건 환경 실패 제외). **β-3 인계 사항**: cache_manager 필드 제거 시 `ALLOWLISTED_TRANSITIONAL` 동시 제거(BANNED 복원은 기성립). **백로그**: `scripts/check_spec_coverage.sh` line 90 octal 버그 + INV-DECODE-STAGE 시리즈 추출 로직 부재(기존 버그, 비차단).
> ★다음 착수 = **β-2**.

1. **신설 `engine/src/session/pipeline_registry.rs`** (L4, §2.1 규칙 C no-`mod.rs`): `PipelineRegistry{stages: Mutex<Vec<Arc<dyn PipelineStage+Send+Sync>>>, len: AtomicUsize}` + `submit()` + `impl PipelineDispatcher` — v2:585-599 그대로(submit 순서 순회·Consumed→OneShot GC·Stop→break·Err→panic fail-fast). **빈-registry fast-path**: dispatch 진입부 `len.load(Relaxed)==0` 즉시 반환. 기존 trait 시그니처(`pipeline.rs:183-185`) 기계적 실현에 한정 — 신규 설계 0.
2. **OpProfiler 수명 계약**: `StageContext{step, profiler: &'a mut OpProfiler}` — DecodeLoop 이 OpProfiler 1개 소유, dispatch 호출부마다 `&mut` 재대여. registry unit test 로 고정.
3. **spec-gap 계약 확정 + spec test 신설**(INV-DECODE-STAGE-004/005/006/007): (i) `on_phase` `Result` ↔ dispatcher Err→panic = fail-fast 고정, (ii) StopReason v1 5-variant(traits.rs:48-55) ↔ v2 4-variant(pipeline.rs:106) 수렴 매핑 — StopFlag 는 driver 자체 stop_flag 체크 잔존·variant 무추가, (iii) **eviction pos-환류 계약** — StageOutcome 무변경, driver 가 eviction phase dispatch 후 held handle `current_pos` 비교로 pos 갱신 + `forward.on_kv_prune`(model_forward.rs:474-486) 호출 의무. **EvictionStage 트리거 phase 명시 확정**(v1 발화 지점=forward 전 → `PreEviction`/`PostEviction` dispatch 지점 + OneShot 소비 phase) + 미발화 phase variant(PreLayer/PostLayer/Fine 등) 처분 doc 기재. StepInfo 3필드 유지(prev_token 승격은 β-6).
4. **INV-LAYER-006 개정 [G7]**: (i) master RED 해소 — 과도기 필드(`cache_manager`) 명시 허용 표기 + β-3 제거 후 금지 복원 예정 기재, (ii) OpProfiler 보유 계약 반영(BANNED 'Profiler' + spec/41:499 normative 개정), (iii) `pipeline.rs` 헤더 L2 ↔ spec/41:499 "PipelineDispatcher trait = L4" 불일치 해소(배치 확정 + 정오표).
5. **SSOT 정오표 + normative 커밋 [G2 승인]**: (a) 사실 정오표 — §9.1 F1 'try_evict 미호출' stale(AB-1 기배선, happy-path 는 cache_manager=None 미발동), 'fmt.compact()' 서술→`execute_kv_plan` 정본, line drift(:558→:654, kv_cache.rs:857→:1113), ExecutionPlan 15→19필드, §9 γ stale, **"manager Level-only 송출" 오기 정정**(SystemSignal 은 graded magnitude 기운반 — 편차는 manager측 융합 부재뿐). census 전제 3건(try_evict 기배선·ResilienceAction/MemoryStrategy 기삭제·compact_parity 9종 완비) 전제 절 고정. (b) normative — G2 승인 문언 3건((i) hook 흡수 재서술 (ii) Forward 잔존 (iii) γ 재정의) 반영.

- **게이트(host)**: build + `cargo test --workspace` + fmt + clippy `-D warnings`. registry unit(순서/Consumed GC 1회성/Stop break/Err panic/len==0 무lock). INV-DECODE-STAGE-004~007 green. **INV-LAYER-006 개정 후 full suite green 복구**(현 master RED 해소 확인 포함). 소비자-0 grep = 옛 경로 byte-identical. SSOT 정오표·normative 커밋 동행 의무.
- **위험**: 계약 오고정 시 β-2~7 재작업(RPN≈120 — 실패 모드는 재작업). 완화 = 계약 spec test 선박.

### β-2 — driver phase 배선 (empty-registry, 거동-0) — ✅ 완료 2026-06-10

> **완료 기록**: `367b56f7`(feat: DecodeLoop prefill/run 에 PipelineRegistry dispatch 9건 삽입 + stale doc 2건 + unit 4건). 발화 = PrefillStart/PrefillEnd + per-token 6종(DecodeStart·PreForward·PostForward·PreSample·PostSample·DecodeEnd) + Finalize; run_until_stop 무접촉(diff hunk 부재로 Tester 검증); v2→v1 StopReason 4-variant 수렴 매핑 구현(per-token Stop→break, PostSample Stop 은 미push break = chat 시맨틱 정합, DecodeEnd Stop 은 token 포함 break). **host 게이트**: lib 1287 PASS(decode_loop 12/12 — 신규 4건 포함, 회귀 0), chat spec 20/20, spec suite 682 PASS(잔여 6 FAIL = GPU 3 + INV-LAYER-001/002/003 `inv_layer_baseline.json` 미갱신 — β-1 worktree 대조로 위반 동수(8/3/12) 입증, pre-existing), fmt/clippy clean. **device 게이트(S25)**: sig 15/15 MATCH(3 dtype × 5 runs, frozen `.out`(`blA_argus_*.out`) 직접 라인 대조 + 동일 recipe md5 EQ — frozen 문서의 md5 절대값(`304f4ada..` 등)은 당시 추출 recipe 미상으로 재현 불가, `.out` 원본 대조를 정본 판정으로 사용) + avg_tbt n=5 median f16 **54.73**(Δ+0.94%)/f32 **54.25**(Δ+0.39%)/q4 **53.50**(Δ−0.54%) — 전부 α-K frozen 절대 기준(54.22/54.04/53.79) Δ≤+3% 충족, per-token 6 phase 오버헤드 위험 비발현(len==0 fast-path 유효 입증) + non-vacuous(LLMRS_FWD_TRACE=1 run: build_plan SUCCESS·wrap 3/3 — fwd-trace 는 env 게이트라 기본 출력엔 미표시, 후속 substep 도 trace run 으로 확인할 것). **백로그 추가**: INV-LAYER-001/002/003 만성 FAIL(baseline.json 미갱신, β-1 이전부터) — 별도 chore.
> ★다음 착수 = **β-3**.

`decode_loop.rs` run(:124-346)/prefill(:79-105)에만 `Arc<PipelineRegistry>` 슬롯(builder `with_pipeline`, default empty) + phase 발화: PrefillStart/PrefillEnd/DecodeStart/PreForward/PostForward/PreSample/PostSample/DecodeEnd/Finalize. **run_until_stop(:356-497) 미접촉**(단 chat 도 prefill() 공유(chat/session.rs:100)라 Prefill 2종 phase 는 chat 에 닿음 — empty registry 거동-0, chat spec 2종 커버). StepInfo{pos, decode_step, pressure=`Pressure::default()`} 스냅샷. v1 trait 호출 (a)~(h) 전부 보존. per-token 발화 = 6 phase. PreLayer/PostLayer/Fine 미발화(orphan + INV-HOTPATH-DISPATCH layer-tier dyn 금지) — β 범위 제외.

- **게이트(device)**: host 선행(full suite + chat spec 2종). device(S25): `argus_cli --no-resilience` frozen baseline 3-dtype sig md5(f16 `304f4ada..`/f32 `684d01d9..`/q4 `1cfba273..`) + avg_tbt n=5 median **Δ≤+3%(α-K frozen 절대값 기준 — 전 substep 공통, rolling 금지)**.
- **위험**: per-token 6 phase 오버헤드. 완화 = len==0 fast-path + 절대 기준 실측; 회귀 시 emission 축소로 substep 내 후퇴.

### β-3 — World A eviction 발화 seam (§9.1 F1) — ✅ 완료 2026-06-10

> **완료 기록**: commit A `603ff2ee`(fmt wrap construction 이동 + fmt_caches accessor + wrap_kv_caches free fn + unit 3) / commit B `96113e3e`(stages/kv/eviction.rs EvictionStage + driver PreEviction·PostEviction 배선 + kv_pos_handle pos-환류(§5.2.1 (가)) + 등가 9종+1 + loop-level 환류 test) / 마감 docs(이 커밋). **host**: lib 1296 PASS(+신규 7: wrap 3·stage unit 3·loop 환류 1, 회귀 0 — 21 FAIL=GPU 부재), 등가 9종+1 **10/10**(anchor=v1 force_evict 직접, 자기 비교 금지 충족, 비-vacuous removed=84≥64), spec 684 PASS(잔여 FAIL ⊆ pre-existing 6), clippy/fmt clean. **device(S25)**: commit A 단독 = sig 15/15 MATCH + tbt f16 54.43(Δ+0.39%)/f32 54.25(+0.39%)/q4 53.63(−0.30%) [f16 은 열누적 재측정, 28.2°C]. commit B 후 happy-path 재검증 = sig 15/15 MATCH + tbt f16 54.14(Δ−0.15%)/f32 54.10(+0.11%)/q4 53.38(−0.76%) — per-token +2 dispatch(PreEviction/PostEviction) 오버헤드 비발현. **eviction baseline 동결** = `frozen_baseline_beta_evict_2026_06.md`(command-driven KvEvictSliding, mock_manager TCP directive — f16 sig `c41930a5..`·q4 `84db59fb..` 각 3/3 결정적, CacheEvent `removed=459, new_pos=459`, crash-free 6/6. `[CacheEvent]` 는 RUST_LOG=info 필수). **정정 1건**: `cache_manager` 필드 제거+INV-LAYER-006 복원은 **β-4 이월**(roadmap 자기 모순 해소 — (a.6) offload 가 β-4 전까지 필드 요구 + β-4 KvOffload 등가 anchor 전제; spec/41·test allowlist 주석 동시 정정). v1 (a.5) live 경로는 유지 — cutover 는 β-4.
> ★다음 착수 = **β-4**.

- **commit A(단독 revert 가능)**: model_forward.rs `ensure_fmt_wrapped`(:229-246) lazy wrap → construction 시점 이동 + accessor `fn fmt_caches(&self)->&[Arc<StandardFormat>]` 신설(INV-STAGE-LAYER-HANDLE 충족). `kv_caches.is_empty()` 가드 보존 + chat reset_kv/빈 캐시 회귀 test.
- **commit B**: 신설 `engine/src/stages/kv/eviction.rs`(§2.1 최종 위치 직행): v2 concrete `EvictionStage`(PipelineStage impl, Persistent⊕OneShot 同코드), register 시점 `Vec<Arc<StandardFormat>>` 보유. 적용 경로 = **CacheManager UER 경유**(take_inner/put_inner — v1 try_evict 와 산출 동일: madvise/CacheEvent/min-floor 회계 보존; `plan_keep→execute_kv_plan` 직접 전환은 friction-triggered 후속). 트리거 1차 cut = v1 AB-1 동일 조건(OneShot). pos 환류 = β-1 계약. wiring: build_bench_loop eviction 시나리오 제출 → v1 AB-1 경로(decode_loop.rs:168-194) 등가 비교. compact_parity.rs 9종 자산 승계. ~~β-3 완료 시 **`cache_manager` 과도기 필드 제거 + INV-LAYER-006 금지 복원**(G7 마감).~~ **[정정 2026-06-10, β-3 census]: 필드 제거+금지 복원은 β-4 로 이월** — (a.6) `try_offload`/`try_recall` 가 β-4 의 (a.5)/(a.6) plan 소비 교체(§β-4 item 3) 전까지 필드를 요구하고, β-4 게이트의 KvOffload 구==신 등가가 구 경로 생존을 전제(자기 모순 해소). spec/41 INV-LAYER-006 행 + test allowlist 주석 동시 정정 완료.
- **게이트**: host 정식((γ) 선례) — stage 발화 vs **v1 try_evict 산출** 등가 integration test: **sliding + h2o + streaming(`sink_size.max(1)` 경계 포함) × F32/F16/Q4_0 = 9종 + min-floor 발화 경계 1종**. **자기 비교 금지 — anchor = v1 live 경로 산출.** device: commit A 단독 = 3-dtype sig md5 + avg_tbt Δ≤+3%(production prefill/step 접촉 → bit-identical 필수). commit B = argus_bench `--eviction-policy sliding` S25 crash-free + `[CacheEvent] Eviction completed` 발화 확인(비-vacuous) + **신규 eviction baseline 동결 → `frozen_baseline_beta_evict_2026_06.md`** — **command-driven evict(manager directive, (a.5)) 시나리오 포함**(β-4 게이트 전제; AB-1 + build_bench_loop `with_cache_manager` + mock_manager 실재 — handoff 의 "AB- 재구현 대기" landmine 은 이 경로 한정 stale).
- **위험**: wrap 시점 변경 → plan lazy build(:405-437) 전 상태 변화 silent garbage(2026-04-14 선례). 완화 = commit A 단독 격리 + 3-dtype bit-identical + on_kv_prune 재사용 + 'eviction 직후 1-step 출력 일치' unit.

### β-4 — α-W-3b 흡수 (ENG-ST-054, v2 §5.4 A-1) — ✅ 완료 2026-06-10

> **완료 기록**: 선행 산출물 `ed733df0`(arch/beta4_command_channel_mapping.md — 18/18 variant×19/19 필드 역방향 검증 + sticky 등가표 + method-drop + heartbeat 채택안(가) + roadmap 라인 drift 정오) / 본 구현 `685fc147`(CommandSource pure poll retarget + command_dispatcher.rs 신설(LoopControl+CommandDispatcher, exhaustive 18-arm match) + (a.5) 제거→OneShot submit + (a.6) Arc<Mutex<CacheManager>> 경유 + **G7 마감: cache_manager 필드+allowlist 제거, INV-LAYER-006 금지 복원 GREEN** + G6 suspend=break 보존 + heartbeat held-handle query) / 회귀 수정 `c8aeda79`(dispatcher 구성 조건 — resilience-on 시 CM 없이도 구성. **device smoke 가 적발**: v1 은 eviction=none+resilience-on 에서 control 디렉티브 적용 — 구현 초판은 무소비 드롭. host e2e(Throttle 30ms: 83.18→106.22 ms/tok)+device(54→86.64 ms/tok) 양검증). **host**: dispatcher unit 10/10 + beta4 채널 5/5(매핑 행별 등가·sticky·heartbeat 연속성·method-drop) + beta3 등가 10/10 유지 + lib 1308 PASS(실패 19=전수 GPU 부재) + spec(잔여 FAIL ⊆ pre-existing) + `cargo test --no-run` + clippy/fmt clean. **device(S25)**: happy-path sig 15/15 MATCH + tbt f16 54.10(Δ−0.22%)/f32 54.00(Δ−0.07%)/q4 53.34(Δ−0.84%) + fix 후 sig 사니티 3/3 MATCH. **β-3 eviction baseline 재확인(신 dispatcher→OneShot 경로)**: f16 sig `c41930a5..` MATCH + q4 `84db59fb..` MATCH + marker `removed=459, new_pos=459` MATCH. **resilience-on smoke**: Throttle directive 수신+적용(86.64 ms/tok ≈ 54+30) crash-free. **참고**: mid-decode TCP directive 도착이 device 에서 수신 지연되는 현상은 transport.rs(β-4 무접촉) 기존 특성 — verify 시나리오는 전부 prefill-중 도착이라 비차단(잔여 관찰 항목으로만 기록).
> ★다음 착수 = **β-5**.

**선행 산출물(코드 diff 전 단독 커밋)**: EngineCommand 18-variant(shared/src/lib.rs:189-278) × ExecutionPlan 19필드(executor.rs:19-65) 구→신 채널 전수 매핑표 + sticky 3-상태(evict_applied decode_loop.rs:48,:169 / sticky carry executor.rs:306-317 / RestoreDefaults reset :484-502) 전이 등가표 — arch/ 부속 문서. **(a.5) method-drop 시맨틱(evict_plan.method 무시, CacheManager CLI 정책 force_evict)도 적시.**

1. `CommandSource`(traits.rs:171) retarget: `poll()->Vec<EngineCommand>`(pure). ManagerCommandSource = 구 CommandExecutor::poll(:289-358) mpsc drain 분리.
2. 신설 `engine/src/session/command_dispatcher.rs`(L4): 구 apply_command(:360-571) 이동. ② Throttle/SetTargetTbt/Suspend/Resume/RestoreDefaults/RequestQcf/SetPrefillPolicy → **LoopControl** 신설. ① **evict-family 4종**(KvEvictH2o :397/KvEvictSliding :411/KvStreaming :428/KvMergeD2o :451) → `registry.submit(OneShot EvictionStage)`. KvQuantDynamic/KvOffload/SwapWeights/SetPartitionRatio/LayerSkip = LoopControl 과도기 필드 잔존(deprecated 표기 — **[G1] β가 owner: AB-2/4/6 Stage 구현 후속 substep 에서 이전**, 미실재 stage 선축조 금지). ③ SwitchHw/PrepareComputeUnit = Hardware resolve seam 만.
3. run() (a)(a.5)(a.6)(h) plan 소비 → dispatcher+LoopControl 교체. **+ [β-3 이월분] `cache_manager` 과도기 필드 제거 + `ALLOWLISTED_TRANSITIONAL` 동시 제거(INV-LAYER-006 금지 복원, G7 마감)** — (a.5)/(a.6) 교체와 동일 커밋. **컴파일 ripple 전수**: run_until_stop(:382-389) 소비부 동시 교체(chat 거동 무변 = chat spec 게이트), batch/runner.rs:495(orphan 이나 컴파일 필수 — 동시 마이그레이션/shim), NoOpCommandSource(defaults.rs:39), ResilienceAdapter/CmdSrcWrapper(resilience_adapter.rs:37/:72). **suspend = break 보존 [G6].** sticky evict → OneShot Consumed GC 등가.
4. **heartbeat conduit**: 현 유일 트리거 = CmdSrcWrapper.poll → executor.poll 상단(:292-296), KVSnapshot 이 poll 인자로만 흐름 → poll pure 화로 단절 위험. **매핑표에 송출 경로 보존 설계 명시**(EngineReport 는 heartbeat 메서드 부재 + DecodeLoop.report dead_code) + **host 게이트에 heartbeat 연속성 test(actual_throughput 송출 등가)**.
5. ResilienceAction/MemoryStrategy/ResilienceStrategy 시그니처 = 기완료 — 작업 0.

- **게이트(device)**: host — 매핑표 행별 등가 test(mock_manager directive 시퀀스): suspend/throttle/tbt/evict 4종 + **live 소비 경로 보유 SwapWeights(:246)/KvQuantDynamic(:465)/KvOffload(:476) 포함 — 과도기 필드도 구==신 등가 의무** + dispatcher EngineCommand exhaustive match(18-variant 컴파일 강제) + sticky 전이 unit + heartbeat 연속성 test + full suite + `cargo test --no-run`. device: argus_cli --no-resilience 3-dtype sig md5 + avg_tbt Δ≤+3% + resilience-on smoke + **β-3 eviction baseline 재확인(command-driven + pressure-driven 무변 1줄 — β-5 단독 귀속 전제)**.
- **위험**: suspend/sticky/heartbeat 미세 시맨틱 변형 → manager-full 회귀. 완화 = 매핑표 선행 + exhaustive match + heartbeat 잔류·연속성 test + baseline 검증.

### β-5 — Pressure 일원화 (Level→Pressure+band)

1. **ripple 단독 commit(behavior-preserving)**: PressureLevel→`Pressure`+`band()` — **실측 10파일 = src 8**(pressure.rs :61/:154, handler 4종, cache_manager :153, build_bench_loop :27/:85, chat/session.rs :29/:617) **+ tests 2**(tests/spec/test_eng_alg_060_092.rs 16곳, tests/test_action_pool.rs 8곳 — integration target, `cargo build` 로 안 잡힘). determine_pressure_level 계단(t, t/2, t/4) 산식 그대로 이식 — 4-level↔band 전사 매핑.
2. LocalPressureSource 신설(session/): canonical band cutoff 소유(pipeline.rs:44-49 placeholder 교체), memory-only(/proc/meminfo).
3. **주입 정책 = 구조적 무접촉**: build_standard_loop 무주입(happy-path pressure 소비자 0 — per-token syscall 차단), StepInfo.pressure source 부재 시 0. build_bench_loop 만 주입 — 내부 N-step 캐시(N=8, 실측 확정).
4. EvictionStage Persistent band-driven 발화 + OneShot 同코드 test 증명.
5. **ManagerPressureSource = defer [G4]** — seam 만. SSOT 에 "β 범위 manager-less = memory graded 만 1급" 명시.

- **게이트(host + device-sanity)**: host — level↔band 경계값 전수 unit(4-level × t/t÷2/t÷4 — eviction/swap trigger 임계 불변) + full suite + **`cargo test --no-run` 전수** + SystemMonitor mock pressure-driven eviction 시나리오. device: argus_cli happy-path sig 무변(무접촉 증명) + argus_bench pressure-driven eviction 1회 — **β-3 동결 baseline 대비**(단독 측정 지점).
- **위험**: band cutoff off-by-one(silent graded 변화). 완화 = 산식 그대로 + 경계값 전수 unit + ripple 단독 commit.

### β-6 — chat 수렴 (run_until_stop 통합)

- **commit A(선행)**: chat 거동 고정 테스트 — run_until_stop(:485-489) max_pos break 마지막 토큰 미포함(**[G5] 보존 확정**), stop 체크 pos-증가-후 타이밍, **turn-boundary score-fed try_evict(chat/session.rs:194/:286 — live 소비자, AttentionScoreAccumulator scores 전달)** 를 수렴 diff 이전에 박음. turn-boundary eviction = decode loop 밖 경로라 **직접 호출 보존**(stage 화 안 함 — TurnStart/TurnEnd 관계만 doc).
- **commit B**: run_until_stop(:356-497, ~140줄 중복) → run 골격 단일 driver 통합 — chat 차이는 registry 등록물 + LoopControl 구성으로 표현, driver 본문 분기 신설 금지. StopCondition(chat/stop_condition.rs:13) → PostSample 구독 stage `StageOutcome::Stop(StopReason)` 반환. **StepInfo `prev_token: u32` 승격 동반**(v2:557 trigger 정식 발동, driver 기보유 ripple 0). TurnStart/TurnEnd 발화. ChatSession 3종 빌드(:439/:494/:553) registry 이식.
- **commit C**: TokenTickSink(live = ResilienceAdapter TickWrapper) → 신설 `engine/src/stages/system/tick.rs` PostSample 구독 stage — heartbeat token count 채널 보존.
- **게이트(host 정식 + device)**: host — chat spec 2종 + commit A 거동 고정 전부 green + token tick 등가 unit(구 on_token_generated 횟수 == 신 PostSample 발화 횟수) + full suite. device: argus_cli happy-path 3-dtype sig md5 + avg_tbt Δ≤+3% 1회 + **resilience-on 재검증(host mock_manager e2e tick 등가 + device resilience-on smoke 1회)**.
- **위험**: chat 분기 happy-path 누설 / chat 미세 거동 발산. 완화 = 거동 고정 선행 + 차이를 등록물/구성으로 + device bit-identical.

### β-7 — v1 표면 삭제 (순수 삭제 단독 편성, 최종 게이트)

1. `session/traits.rs` 삭제: EvictionStage/SwapStage/TokenTickSink/ResilienceBundle/DecodeObserver/StepCtx/EvictionOutcome/SkipReason/StopReason(v1) — 등가물 pipeline.rs 수렴. **DecodeResult 는 삭제 아님 — §2.1:193 session/(L4) 이동만**(소비자 chat/session.rs:37·experiment_run.rs). run() v1 슬롯 (b)/(c)/(e)/(g) 제거 — production impl 전부 NoOp(거동-무변). defaults.rs NoOp 7종 제거.
2. 생존 3종: TokenSampler→`inference/sampling.rs` 이동(front-door ①, Greedy/RepetitionPenalty impl 동행) / CommandSource(β-4 신 시그니처) / EngineReport(CommandExecutor 잔류). Forward = 슬림 내부 seam 잔존(G2-(ii) SSOT 문언 기반영).
3. typestate builder(NoForward/HasForward :24-27,:648-679) → 신 assembly 대체 + INV-LAYER-007 compile_fail 등가 승계. **잔존 소비자 재배선 명시: experiment_run.rs(:87 + :19 v1 StopReason import) · standard_happy.rs(:70) · microbench/probe_inference_loop.rs:222.**
4. **[G3 확정] orphan hook 3종(StepHook/PhaseHook/LayerBoundaryHook) + eval/batch orphan 코드(run_eval_ll/run_prompt_batch) 동반 삭제** — spec INV-147/149/150 은 phase 어휘 등가 승계 또는 폐기 표기.
5. stages/*.rs doc stale 갱신 + no-`mod.rs` 최종 확인. 삭제 단일 커밋군 격리.

- **게이트(device 최종)**: host — full suite + `cargo test --no-run` + clippy + compile_fail 승계 + grep `session::traits|StepCtx|EvictionOutcome` production 0. device(S25): argus_cli happy-path 3-dtype sig md5 + avg_tbt n=5 — **α-K frozen 절대 기준 누적 Δ≤+3%**('진짜 최종 perf') + β-3 eviction baseline MATCH + test_backend 5 op PASS + KIVI 오라클 Q2/Q4/Q8 + resilience-on smoke. 순수 삭제 = sig 무변, 불일치 = tripwire.
- **위험**: 삭제 ripple orphan 비대화. 완화 = β-6 종료 시 소비자-0 grep census 선행 + 잔존 소비자 사전 이식. (G3 확정으로 잔존-허용 조항은 폐기 — 동반 삭제가 기본.)

---

## 후속 (β 범위 외, 기록)

- **AB-2/4/6 Stage 모델 구현** — G1 결정에 따라 β-4 완료 후 β 후속 substep 또는 별도 트랙으로 재개(과도기 LoopControl 필드 이전 포함).
- **ManagerPressureSource(엔진측 융합)·LocalPolicy·로컬 센서 인프라** — G4 defer 분.
- **γ 재정의 잔여**(G2-(iii) SSOT 반영 후): kv/·weight/ rename(189ref/53file), nested mod.rs 38개 sweep, argus-eval/chat bin화, batch orphan 처분.
- **chat 마지막 토큰 거동 버그 여부 판단** — G5 보존 분.
- **suspend pause(park) 전환** — G6 보존 분.

## 참조

- SSOT: `arch/pipeline_stage_design_v2.md` §5(:532-655) · §5.4(:611-655) · §2.1(:168-207) · §9.1(:719-917)
- 진입: `.agent/todos/handoff_beta_decode_loop_rewrite_entry_2026_06_10.md`
- baseline: `.agent/todos/frozen_baseline_alpha_k_5f_2026_06_05.md` (+ β-3 산출 예정 `frozen_baseline_beta_evict_2026_06.md`)
- spec: ENG-ST-054 (α-W-3b) · INV-LAYER-006/007 · INV-DECODE-STAGE-004~007
