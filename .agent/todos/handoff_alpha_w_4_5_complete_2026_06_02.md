# Handoff: Phase α-W-5 + α-W-4 완료 (format/capability 축) → 잔여 α-W-3b (Phase β 경계)

**작성**: 2026-06-02
**HEAD**: `42cb9066 feat(engine): Phase α-W-4 — capability 물리 정착 + CapabilityRegistry 배선 + KiviCache.kivi handle`
**브랜치**: `master` (push 완료, origin 동기화)
**다음 세션 진입 문장**: **"α-W-3b 진행"**(resilience 2-source, **α-W 트랙 最종·最위험, Phase β 경계**). 단 진입 전 아래 §α-W-3b scope 결정 필요.

> 설계 SSOT = `arch/pipeline_stage_design_v2.md`. 트랙 = [[project_pipeline_alpha_w]].

---

## TL;DR

확장 파이프라인 α-W 를 5 substep 분해 진행. **이번 세션 2 substep 완료**: α-W-5(format/weight 축, 5a `3e5dc49f`/5b `ef723da7`) + α-W-4(capability 축, `42cb9066`). 누적 = α-W-1·2·3a·3-arch·**5·4**. **잔여 = α-W-3b 단 1개**(resilience 2-source). **멈춤 지점**: α-W-3b 가 Phase β(PipelineRegistry+새 DecodeLoop)에 깊게 의존하는 phase-boundary 라 scope 결정 후 진입.

---

## 진행 상태

| substep | commit | 게이트 | 결과 |
|---|---|---|---|
| α-W-1/2/3a/3-arch | `0d12c81d`/`1a1cd444`/`57629f26`/`4ca4e6ff` | host·device | (이전 세션) |
| **α-W-5a** PartitionedWeight→Vec<WeightSlice> | `3e5dc49f` | host | byte-identical 구조 |
| **α-W-5b** WeightFormat::apply_dispatch 배선 | `ef723da7` | device | S25 `--tensor-partition 0.5` bit-identical |
| **α-W-4** capability 물리정착+Registry+KiviCache.kivi | `42cb9066` | device | S25 `--kv-mode kivi` Q4_0+F16 bit-identical |

- 게이트 절차·수치 상세 = 각 commit 메시지 + [[project_pipeline_alpha_w]] 구현 메모.

## ★ α-W-3b scope 결정 (진입 전 필수)

α-W-3b = `CommandSource`/`CommandDispatcher`/`LoopControl` 신설 + `CommandExecutor` 분해 + `PressureSource` impl(Manager/Local) + `LocalPolicy`. **§5.4 2-source 모델**(PressureSource ∥ CommandSource).

**Phase β 의존 발견 (이번 세션 스카우트):**
- 현 라이브 resilience 경로 = `ResilienceAdapter`(`session/resilience_adapter.rs`) → `CommandExecutor`(`resilience/executor.rs`) → `ExecutionPlan` → session traits(StepHook 등), `BatchRunCtx.command_executor`(`batch/args.rs:38`)로 decode 루프 배선.
- §5.4 line 643: `CommandDispatcher → ①KV/weight·③switch = registry.submit(OneShotStage) / ②control = LoopControl`. **`registry.submit(OneShotStage)` 는 PipelineRegistry + 새 DecodeLoop = Phase β** (현 코드에 PipelineRegistry 부재; v2 PipelineStage 모델은 미배선).
- §5.4 연혁 650: "코드 적용 = Phase α-W(CommandSource/CommandDispatcher/LoopControl 신설 + CommandExecutor 분해 ...)". **타입 신설은 α-W 이나 라이브 루프 배선(registry.submit)은 β** — α-W-4 와 동일한 "타입은 지금 / 완전 배선은 후행 phase" 패턴이나 **β 의존이 더 깊다**(α-W-4 는 KiviCache 라는 기존 객체에 얹었지만, 3b 의 dispatch 는 아직 없는 registry 를 요구).

**achievable α-W-3b 후보 (architect scope 확정 필요):**
- (A) **타입 + decompose만**: CommandSource trait / LocalPolicy / PressureSource impls(Manager/Local) / LoopControl struct 신설 + CommandExecutor::apply_command 을 source/dispatch/report 로 분해 — **단 ExecutionPlan/라이브 경로 보존**(registry.submit 대신 현 ExecutionPlan 생산 유지, β 까지). 대부분 host-gateable, 라이브 동작 무변.
- (B) **Phase β 로 이연**: 3b 가 β DecodeLoop 재작성의 일부 — α-W 트랙은 4/5/3a 로 종결, 3b 는 β 진입 시.

**→ α-W-4 처럼 architect scope round 로 (A) 의 정확한 achievable 경계(β 미의존분)를 확정한 뒤 진입 권장.** (A) 가 의미있는 self-contained 단위인지, 아니면 registry 없이는 decompose 가 hollow 한지가 핵심 판정.

## device 게이트 절차 (재사용)
- bin=`legacy_generate`, S25(`galaxy_s25`, R3CY408S5SB). 단일토큰 prompt. baseline=현 master(미커밋이면 stash dance) vs post 생성텍스트 diff.
- α-W-5: `--tensor-partition 0.5`. α-W-4: `--kv-mode kivi`(Q4_0 forward_gen + F16 plan, 둘 다 update_gpu 매 토큰). α-W-3b: standard happy + manager 경로(resilience).

## Landmines / 교훈
- **cargo 가 권위 (이번 세션 4회 재확인)**: α-W-5/4 양쪽에서 IDE 진단이 E0560/E0615/E0277/E0063/E0061 다수 표시했으나 전부 **mid-edit stale**. `cargo build`(lib+bin) + **`cargo test -p llm_rs2 --no-run`(integration spec arity 검출 필수 — `cargo build` 는 tests/ 미컴파일)** 로 확정.
- **architect scope round 가치(α-W-4)**: §3.6/§4.1/α-K/β 경계가 미묘할 때 architect 가 achievable scope(forward hot-path 무변, KiviCache.kivi만)를 정밀 carving → 위험 HIGH→중 하향 + bit-identical 보장. α-W-3b 도 동일 권장.
- **S25 Adreno `is_nosub()=true` 기본** (F16 GEMV 커널 변형 플래그, device subgroup 아님) → forward_gen:419/backend.rs:125 "Adreno fallthrough" 주석 stale (코드 무변, 별도 docs 정정 backlog).
- **커밋 금지 untracked**: `.antigravitycli/`, `.claude/scheduled_tasks.lock`, `papers/.../microbench_*`, `.agent/todos/handoff_microbench_*.md`. 명시 파일 add(`git add -A` 금지).

## 자기점검
- 진입 문장: ✓ "α-W-3b 진행" (단 scope 결정 선행)
- 왜 멈췄나: ✓ α-W-3b 는 Phase β 경계 — 자율 강행보다 scope 결정(A 부분진행 vs B β이연) 필요
- 최대 landmine: ✓ 3b 의 CommandDispatcher→registry.submit 이 미존재 PipelineRegistry(β) 의존
- 검증 게이트: ✓ α-W-5/4 host+device 수치 commit 메시지에 명시
- device 가용: ✓ S25 USB, 게이트 절차(partition/kivi) 확립
