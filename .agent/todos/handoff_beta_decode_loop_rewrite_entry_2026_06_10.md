# Handoff: Phase β 진입 — DecodeLoop 재작성 (단일 PipelineStage 모델)

**작성**: 2026-06-10
**HEAD**: `1f93d6b1 docs(arch): Backend 축 C12 device 부분검증 GREEN — S25 KIVI 오라클 L2=0`
**브랜치**: master (origin 동기 — `1f93d6b1` push 완료)
**작성자**: 메인 세션
**다음 세션 진입 문장**: **"β 트랙 진행 — DecodeLoop 재작성. 시작 = 구현 cut 설계(α-K 처럼 runnable substep 분해). SSOT = `arch/pipeline_stage_design_v2.md` §5(PipelineStage 모델) + §9/§9.1."**

> 이 문서는 **β 진입점**. β 는 3–4주 大단계 + 미설계라 첫 작업은 **구현 cut 설계 grill**이지 바로 구현이 아니다. SSOT 가 모델(§5)·roadmap(§9)을 이미 확정했으므로, 다음 세션은 그것을 **runnable substep 로 분해**하는 것부터.

---

## TL;DR

- **끝남**: α-K BC 종결(`KVCacheOps` trait 완전 폐기, frozen baseline `frozen_baseline_alpha_k_5f_2026_06_05.md`) · GATE-C 3축(Stage⊥Format⊥Backend) dlopen plugin 완비(Backend 축 device 비트정확 `1f93d6b1`) · α-W 인프라 5 substep 중 **α-W-3b 1개만 잔여**(아래 흡수).
- **다음 = β**: 현 live `DecodeLoop`(v1 typestate builder + `session/traits.rs` 7-trait + 5 hook trait)를 **단일 `PipelineStage` + `LifecyclePhase` enum + `PipelineRegistry`** 모델로 재작성(§5). α-W 가 `pipeline.rs`(PipelineStage)를 additive 로 스캐폴딩만 해뒀고(live 미배선), **실제 교체가 β**.
- **왜 지금/시작점**: 선행 트랙(α-W 인프라·α-K KV 패러다임·GATE-C plugin) 전부 완료라 DecodeLoop 재작성의 전제가 갖춰짐. β 는 大단계라 §9.1 R3 게이트 철학으로 substep 분해 설계부터(위임 아님, 설계 grill 먼저).

---

## 진행 상태 (시작 state, 코드 실측)

| 요소 | 위치 | 상태 |
|---|---|---|
| `PipelineStage` trait (대체 어휘) | `engine/src/pipeline.rs:165` | ✅ α-W-1 스캐폴딩(additive, **live loop 미배선**) + `engine/src/stages.rs` skeleton |
| 현재 live `DecodeLoop` | `engine/src/session/decode_loop.rs:30` (struct), `:73` impl | v1 typestate builder(`NoForward`→`HasForward`), run 루프 |
| run 루프 순서(§9.1-c) | `decode_loop.rs:159` eviction `before_step` → swap before → `:194` forward `step()` → swap after → sample | `Forward::try_evict` 는 run 에서 **미호출**(grep 0) |
| v1 7-trait (β 흡수·삭제 대상) | `engine/src/session/traits.rs` | Forward/EvictionStage/SwapStage/CommandSource/EngineReport/TokenTickSink/ResilienceBundle/DecodeObserver |
| α-W-3b (β 로 흡수) | spec ENG-ST-054 (코드 미구현) | `CommandSource`/`CommandDispatcher`/`LoopControl` 신설 + `CommandExecutor` 분해 + `PressureSource`(Manager/Local) + `LocalPolicy`. 매핑 raw = workflow `wq32w40ql` |
| frozen baseline (bit-identical 기준) | `.agent/todos/frozen_baseline_alpha_k_5f_2026_06_05.md` | α-K 종결 baseline |
| HEAD / host | `1f93d6b1` master / **host GPU 없음**(device=S25 `R3CY408S5SB`) | — |

---

## 다음 작업 (β substep 설계부터)

1. **β 구현 cut 설계 grill** — §5(PipelineStage 모델) + §9.1 R3 게이트 철학으로 **runnable substep 분해**(α-K 의 step-1~5 / α-W 의 W-1~5 선례). 산출 = substep 표(내용·게이트 tier·위험) + roadmap todo. 각 substep 이 컴파일+실행되는 엔진을 남길 것(branch-by-abstraction).
2. **α-W-3b 흡수 계획** — `CommandSource`/`CommandDispatcher`/`LoopControl` 신설 + `CommandExecutor`(executor.rs) 분해. β 가 live DecodeLoop 본문을 재작성하므로 여기서 통합(ENG-ST-054 spec + §5.4). manager-less 이산 정책(switch/suspend) 1급 요구.
3. **생존/삭제 trait 확정** — 삭제: v1 7-trait(흡수). **생존(삭제 금지)**: `TokenSampler`(→`inference/sampling.rs`, front-door ①), `CommandSource`(이산 명령 source seam, §5.4), `EngineReport`(heartbeat/status → `CommandExecutor` 잔류).
4. **World A eviction 발화 seam 구축** (§9.1 F1 — 핵심 난제) — 현재 DecodeLoop 가 real eviction 을 한 번도 발화 못 함(seam 미구축). β 가 이 seam(`StepInfo`→held cache handle 도달)을 짓는다. Phase 4-4 잔여가 β 로 folds.
5. **게이트 tier별** — host(additive substep) / device(거동 변경 substep, S25). **device 게이트 bin = `argus_cli`(happy-path) + `test_backend`** — landmine 참조(legacy 삭제됨).

---

## Landmines / 미해결 / 안 가본 길

- **★ device 게이트 bin 변경 (메모리 stale)**: `[[project_pipeline_alpha_w]]` 의 "게이트 bin = `legacy_generate`" 절차는 **무효** — α-K 5-F(`d5ed71d2`)에서 legacy bin/generate.rs 삭제됨. 현재 device 게이트 = `argus_cli`(standard happy-path 만) + `test_backend`(op 정확성). **eviction/KIVI/swap/offload 모드는 CLI 부재**(argus reject·legacy 삭제 → AB- argus-bench 재구현 대기) → β 의 그 경로 device 게이트는 host-test 또는 argus-bench 의존.
- **★ World A eviction 발화 seam 미구축 (§9.1 F1, β 핵심 난제)**: `decode_loop.rs::run` 이 real eviction 을 한 번도 발화한 적 없음 — `build_standard_loop`(`assembly/build_standard_loop.rs`)이 eviction stage 미주입 → default `NoOpEvictionStage`(`decode_loop.rs:558`). `EvictionStage::before_step(&StepCtx)` 가 `ModelForward` 캐시(`kv_caches`/`fmt_caches`)에 도달할 seam 구조적 부재(`model_forward.rs:227` "Phase 4-4 will reach …"). β 가 이걸 짓는 게 가장 큰 작업.
- **생존 trait 삭제 금지**: `TokenSampler`/`CommandSource`/`EngineReport` 는 흡수 대상 아님(§2.1 v1 잔재 처리, §5.4). 흡수 후 동명 충돌(v1 `EvictionStage`/`SwapStage` trait ↔ v2 concrete Stage)은 자연 소멸.
- **α-W-3b manager-less 정책 의존성**: 이산 정책(switch/suspend)은 **로컬 센서 모니터(thermal/battery/usage 자율 수집) 인프라 동반** 필요 — 현 production `SystemSignal` 생산자는 `dbus_transport`(manager)뿐(§5.4 연혁). β 에서 `LocalPressureSource`/`LocalPolicy` 가 그 seam.
- **cargo 가 권위**: IDE/rust-analyzer 진단(E0061/E0063/E0560 등) mid-edit stale 빈번(α-W·α-K 4회 재확인). integration test arity 는 `cargo build` 로 안 잡힘 → `cargo test --no-run` 필수.
- **argus-bench(AB-) 병렬 트랙**: AB-2(KIVI)/AB-4(tensor partition)/AB-5(verify)/AB-6(weight-swap)이 mode dispatch 재구현 — β 의 PipelineStage 모델과 공유 표면이라 **조율 필요**(어느 쪽이 mode dispatch 의 owner 인지 설계 시 결정).
- **§9 roadmap γ stale**: §9 가 β 후 "γ = legacy generate.rs 잔여 마이그레이션"으로 적었으나 legacy 는 이미 5-F 에서 삭제 → **γ scope 재평가 필요**(PACT2026 PoC 만 잔존?).

---

## 참조
- 설계 SSOT: `arch/pipeline_stage_design_v2.md` — **§5**(PipelineStage 모델, trait 정의 :539) · **§9**(roadmap, β=line 712) · **§9.1**(R3 게이트 철학 + (3c-evict)/(3d) eviction seam census) · §2.1(v1 잔재 처리) · §5.4(CommandSource/PressureSource 대칭).
- 코드: `engine/src/pipeline.rs`(PipelineStage) · `engine/src/stages.rs`(skeleton) · `engine/src/session/decode_loop.rs`(현 live) · `engine/src/session/traits.rs`(v1 7-trait, 삭제 대상) · `engine/src/session/forward/model_forward.rs`(eviction seam gap :227).
- 메모리: [[project_pipeline_alpha_w]](α-W 5 substep, α-W-3b 상세 — 단 device 게이트 절차 stale) · [[project-pipeline-alpha-k]](α-K BC 종결, frozen baseline).
- 선행 완료: GATE-C Backend 축 [[project-backend-axis-capability-dlopen]] · α-K `frozen_baseline_alpha_k_5f_2026_06_05.md` · `handoff_alpha_k_bc_complete_2026_06_05.md`.
