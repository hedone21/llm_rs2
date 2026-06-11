# Handoff: AB-2/4/6 Stage 모델 재개 진입 (β/γ 종결 후)

**작성**: 2026-06-11
**HEAD**: `ecd07549 docs(todo): Phase γ 종결 — γ-4 RESOLVED + generate 분할 부분 해소 + 파생 후속 5건 등록`
**브랜치**: master (worktree 없음)
**다음 세션 진입 문장**: **"AB-4 device 게이트 진행 — handoff_ab246_stage_entry 기준"** (host 단계는 2026-06-11 완료 — 아래 다음 액션 2번 참조)
**착수 순서 (사용자 확정 2026-06-11)**: **AB-4 → AB-6 → AB-2**

> 본 문서가 AB-2/4/6 재개의 SSOT. 구 계획 `handoff_argus_bench_ab0_ab3_2026_06_05.md` §AB-2/4/6 은
> **배선 계층 전부 stale**(β-4/β-7 로 전제 소멸) — 도메인 인벤토리·verify 시나리오·landmine 목록만 승계.
> 근거 = 재개 census (workflow `wf_867d2c2b`, 4 survey + adversarial verify, 2026-06-11).

---

## TL;DR

- G1 동결 해제 조건(β-4 확정) 충족 + γ 종결. census 결과 **세 트랙 모두 차단 없음** — 유효/무효 분류 완료.
- 공통 골격: 각 트랙 = `command_dispatcher.rs:96-106` deprecated LoopControl 필드 1개 →
  `stages/` 하위 OneShot PipelineStage 이전(`submit_evict` :304-317 거울) + 필드 삭제 + 구==신 등가 테스트 승계.
- 왜 이 시점: 사용자 요청 "AB-2/4/6 Stage 모델 재개" (2026-06-11). 순서는 AskUserQuestion 으로 확정.

## Track 판정 (census verdict 압축)

| 트랙 | 판정 | 첫 결정/액션 |
|---|---|---|
| **AB-4** partition | **최소 재설계 — 1순위.** 구 최대 landmine(`prepare_tensor_partition(&mut)` vs `Arc<Model>`)은 α-W-5 가 본질 해소: `LayerSlot::apply_dispatch(&self)` RCU + INV-120 gen-counter(slot.rs:193) + `build_partitioned_layer_plan` 의 `PlanInvalidated` 런타임 자동 무효화. 트리거만 greenfield | (a) `transformer.rs:990` `&mut`→`&self` 1줄 완화 vs Stage 가 `Vec<Arc<LayerSlot>>` 직접 보유 택1, (b) 결정층 = `"partition"` WeightStage 빌트인(3축 정합) vs 엔진 직결 OneShot(결정이 항등이라 단순) — status quo 포함 비교 후 확정 |
| **AB-6** weight-swap | **최대 재설계 — 즉시 착수 가능.** 정식 경로 = ADR-0006 **Seam B**: `stages/weight.rs`(8줄 골격 예약석)에 `WeightSwapStage: PipelineStage`(OneShot, D5 multi-tick Continue→Consumed) 신설. Seam A 는 **MW-A~D 4/4 완료**(`0ff9f609`, `weight/stage_ctx.rs` 실재). `EngineSwapRuntime::handle_swap_weights`(orphan, swap_runtime.rs:137)가 Stage 본문 §1-7 명세 | 발화 phase 결정 — 미발화 PreSwap/PostSwap* 에 driver 발화 신설 vs 기발화 PostEviction/PostForward 구독(decode_loop.rs:337/:362 주석이 v1 (c)/(e) 슬롯 등가 자리 명시, driver 무수정) |
| **AB-2** KIVI | **부분 재설계 — 3순위.** D8 ABI 는 re-export 로 caps 경로·TypeId 보존(저위험; 신규 직접 호출은 `KiviAttnArgs` 패킹 — 참조: kivi_format.rs `attention_native`/kivi_cache.rs `update_gpu`). **C12 잔여 production e2e 를 AB-2 가 흡수**(완료 게이트에 포함 의무) | 1순위 쟁점 = KiviForward 의 owned `Vec<KiviCache>`(Arc 아님)에 Stage 가 도달하는 법: handle Arc 화 vs (a.6) try_offload 선례의 Forward seam 메서드. + 검증 수단 택1: argus_bench KIVI 분기(원계획 step1-2 유효) vs argus-chat bin화 선행 |

## 유효 자산 (전 트랙 공통)

- **선례 패턴**: `stages/kv/eviction.rs` EvictionStage(OneShot/Persistent 同코드, register 시점 Arc handle, armed edge-trigger, driver pos-환류) + `stages/system/tick.rs` + ChatStopStage(phase 선택은 v1 등가 시점 census 필수 — PostSample vs DecodeEnd 실사례).
- **dispatch tier**: AB Stage 전부 step-tier 라 dyn 자유(INV-HOTPATH-DISPATCH). 단 AB-6 intra-forward/phase-aware 는 layer-tier — PipelineStage 로 못 올림, `forward_into` 의 layer_boundary_hook(별개 live 서브시스템, INV-147/149/150) 유지. AB-4 는 plan path 에 dyn 신설 금지(`Option<PartitionContext>` 정적 분기).
- **도메인 인프라 전부 생존**: KIVI `transition_bits`(kv/kivi_cache.rs:1034)·`alloc_kivi_kv_caches`·`--kv-dynamic-quant`(cli.rs:634) / partition `PartitionContext`+INV-120+`apply_dispatch` / swap 4-mode(IncrementalSwapPlan/IntraForwardSwapHook/PhaseAwareSwapDispatcher)+pacing(dynamic_k)+`WeightStageModelCtx`.
- **MEMORY 제약 승계**: swap production winner = per-tick=2 + async + dynamic-K + sub-batch pause (LISWAP-6, 메모리 spike 회피 hard constraint).

## 검증 게이트

| 게이트 | 수치/명령 | 비고 |
|---|---|---|
| happy-path 무회귀 | α-K frozen: S25 sig md5(f16=304f4ada…/f32=684d01d9…/q4=1cfba273…) + tbt 54.22/54.04/53.79 ms/tok, Δ≤+3% (`frozen_baseline_alpha_k_5f_2026_06_05.md`) | **착수 직전 현 HEAD 재측정 필수** — 캡처 HEAD 8e7ffc67(β 이전)이라 β/γ drift 와 AB 회귀 합산 방지 |
| command-driven 경로 | β evict baseline: mock_manager TCP + KvEvictSliding 0.5 → sig + `removed=459, new_pos=459` + `final_pos=586` (`frozen_baseline_beta_evict_2026_06.md`) | RUST_LOG=info 필수 |
| AB 액션 자체 | **frozen baseline 부재** → 각 트랙 구현 직후 mock_manager TCP 시나리오 틀 복제로 신규 동결 | mock_manager 가 KvQuantDynamic/SetPartitionRatio/SwapWeights 송신 지원 |
| KIVI 회귀(AB-2) | `test_backend --backends opencl` KIVI 오라클 Q2/Q4/Q8 3/3 PASS, L2=0.000000 bit-exact | MatMulTransposed 16/MatMulSlice 8 FAIL + RoPE 2 ERROR 는 pre-β 하네스 이슈 — AB 회귀로 오판 금지 |
| host 등가 | beta4_command_channel.rs 과도기 5종 등가 테스트(:494+) → Stage 이전 시 승계 | armed 시맨틱: swap=transient(OneShot 직역), quant/partition=sticky(**last-applied 비교** 게이트 — evict_armed 복사 금지) |

## Landmines

1. **"Stage" 어휘 충돌**: `WeightSwapDeciderAsStage`(weight/stage_registry.rs:26)는 technique-api **WeightStage**(plan-returning plugin, Seam A) — PipelineStage 아님. "AB-6 이미 있다" 오독 금지.
2. **stale 문서 함정**: `handoff_weight_stage_unification_adr0006_2026_06_07.md` 는 MW-D 완료 직전 작성 — "MW-D 잔여" 로 오독해 stage_ctx.rs 재작성 금지(census survey 1 도 빠졌던 함정). 코드(`0ff9f609`)가 정본.
3. **silent no-op**: PreSwap/PostSwapBefore/PostSwapAfter/PrefillChunkBoundary 는 enum 정의만 있고 **driver 발화 0** — 구독 Stage 만 만들면 컴파일 되지만 영원히 안 불림.
4. **v2 prefill 은 단일 호출**(command poll/chunk 경계 부재, decode_loop.rs:193-213) — `prefill_midway_partition` 시나리오의 mid-prefill 적용은 v2 에선 decode step 0 으로 늦춰짐. AB-4/AB-5 에서 chunked prefill 신설 여부 명시 결정 필요.
5. **로그 문자열 = 계약**: verify YAML 이 `[KIVI-Resilience] Transitioned KV cache to 4bit` / `[Partition] (Lazy-mapped|ratio=0\.3)` 을 regex marker 로 직접 검사(kvquant_to_q4.yaml:26 / partition_enable.yaml:26). 신규 Stage 글자단위 재현 or AB-5 에서 YAML 동시 갱신 — roadmap 에 미리 못박을 것.
6. **host green ≠ 기능 검증**: partition 은 CPU silent no-op(GPU 전용), AB-2 GPU realloc·AB-6 intra-forward/phase-aware 도 device 필수.
7. **AB-4 참조 구현 소실**: batch/runner.rs:172-211 은 γ-4(`2e53cf44`) 삭제 — 대체 참조 = `session/init.rs:818-851`(α-W-5 정적 CLI 경로, live) 또는 `git show 2e53cf44^:engine/src/session/batch/runner.rs`.
8. **AB-2 검증 수단**: `build_chat_kivi`(chat/session.rs:491) 호출처 0 orphan. 유일 live KIVI e2e = argus_eval 2모드(run_eval_ll_kivi/run_ppl_kivi) — AB-2 가 KiviCache/KiviForward 표면 변경 시 즉시 영향권, 무회귀 smoke 게이트 포함.
9. **stale 주석 2건**: init.rs:603 'directive handler below 에서 lazy 호출'(handler 는 삭제된 legacy) / stages/weight.rs '예정 입주자(Phase α-K)'(실 owner = AB-6, 진입 시 정오). 반대로 transformer.rs:990 `&mut` 보고 과대 대응 금지 — 본문은 `&self` 만 사용, 1줄 완화로 끝.

## 다음 액션 (AB-4 기준)

1. ~~**AB-4 설계 결정 2건**~~ ✅ **사용자 확정 (2026-06-11)**: (a-2) fan-out 을 `layers/tensor_partition.rs` 자유 함수로 추출 + Stage 는 `Vec<Arc<LayerSlot>>` 보유(EvictionStage 동형, CLI 정적 경로와 함수 공유) / (b-1) 엔진 직결 OneShot(plugin 결정층 없음 — 결정이 항등, EvictionStage method-drop 선례 동형, `"partition"` WeightStage 는 per-layer 알고리즘 등장 시 재검토). lazy mapping 은 slot-only `&self` 변형 분리(norm/lm_head 는 SwitchHw 전용이라 불필요).
2. ~~**AB-4 구현**~~ ✅ **host 완료 (2026-06-11)**: 설계 `a9b9f4b2`(arch v2 §5.5 신설 + spec §3.28 + ADR-0006 §6 + beta4 매핑표, 신규 INV 0건) / 구현 `359b9a29`(fan-out 자유 함수 `apply_partition_dispatch` + slot-only `map_layer_slots_for_host_access` 추출, 동작 불변) + `6c0616cb`(`stages/weight/partition.rs` PartitionStage OneShot·PreForward + dispatcher `submit_partition` last-applied 게이트 + `LoopControl.partition_ratio` 삭제 + RestoreDefaults Full 복원) + `1ba1a6f7`(argus_bench 가드 해제 + 로그 계약). 신규 테스트 13종 green, Tester 교차 검증 **host 게이트 GREEN(조건부)** — fan-out/INV-123 보존 정독 확인, sticky 값-비교 게이트 비-vacuous, 로그 regex 글자단위 MATCH(rustc 실측), layer_lint 30=baseline. disable(ratio≤0) 케이스 = Stage 책임(Architect 승인, §5.5.3). **조건**: macOS OpenCL 1.2 frozen 으로 lib 테스트 이름 단위 대조는 Linux/device host 재실행으로 확정 필요.
3. **AB-4 device 게이트 (다음)**: α-K frozen 재측정(MATCH 확인 선행) → S25 mock_manager `SetPartitionRatio` 시나리오 + `[Partition]` marker + 신규 baseline 동결 → 검증: sig·marker·tbt Δ≤+3%. + Linux/device host 에서 `cargo test -p llm_rs2 --lib` 이름 단위 대조 1회.
4. AB-6 진입 (발화 phase 결정부터) → AB-2 진입 (Vec<KiviCache> 접근 모델 + 검증 수단 결정부터).
