# Handoff — 설계 구체화 (ResilienceStrategy 가지 닫힘 → pressure/ 해체 reconciliation)

**작성**: 2026-06-01 (ResilienceStrategy grill 종료 + arch doc 반영·커밋·푸쉬 후)
**HEAD**: `84880e7e docs(arch): ResilienceStrategy 2-source 모델 + EngineCommand 단일 어휘 (§5.4 신설)` (= origin/master, **pushed**)
**브랜치**: `master` (ahead 0, clean)
**작성자**: 메인 세션 (Claude)
**다음 세션 진입 문장**: **"pressure/ 해체 reconciliation"**

> 설계 SSOT = `arch/pipeline_stage_design_v2.md`. v1(동결) = `arch/pipeline_stage_design.md` (grill 이력·R-5 원본). **코드 아님 — 설계 문서 구체화 트랙.** Phase α-W/α-K 는 코드 진입(아직 아님).

---

## TL;DR

ResilienceStrategy 가지 grill 완료 → arch doc §5.4 신설 + front-door/§2.1 정정으로 반영 → 커밋(`84880e7e`)·푸쉬 완료. 다음 = **pressure/ 해체 reconciliation**: §2.1 은 `pressure/ 유지`(format impl+offload+eviction policy 거주)인데 직전 세션 dialogue 결정은 `pressure/ 완전 삭제`(→ `format/` + `stages/`). 두 'Q1-Q6' 가 충돌. 멈춘 이유 = 이번 세션은 ResilienceStrategy 만 재결정, pressure 는 손대지 않음(doc 은 현 `pressure/ 유지` 구조로 내부 정합 유지).

---

## ResilienceStrategy 가지 — 닫힌 결정 (doc §5.4 반영·커밋 완료, **재grill 금지**)

| # | 결정 |
|---|---|
| 1 | `ResilienceAction` **삭제** → `EngineCommand`(shared/) 단일 이산 어휘 |
| 2 | `ResilienceStrategy` **생존**(front-door ①) — `react(&mut self, signal) -> Vec<EngineCommand>` (dead `mode` 제거, 출력 통일). manager-less `LocalPolicy` 정책 단위 |
| 3 | `MemoryStrategy` **소멸** (graded → `LocalPressureSource`). 이산 잔존 = Thermal/Energy/Compute |
| 4 | `resolve_conflicts` 생존, `EngineCommand` 로 retarget, `LocalPolicy` 내부 (manager-full 은 manager Lua 가 병합) |
| 5 | **2-source 대칭**: `PressureSource`(연속) ∥ `CommandSource`(이산), 각 Manager/Local |
| 6 | **A-1** 3분할: `CommandSource`(pure 생산) / `CommandDispatcher`(변환,L4) / `EngineReport`(보고) |
| 7 | KV/weight 명령 **OneShot Stage化** → registry.submit, pressure-driven persistent Stage 와 동일 코드 |
| 8 | `ExecutionPlan` → `LoopControl` 축소 (②control 만, 15→~9 필드) |

**근거 핵심 (재확인 불요)**: `ResilienceAction`/`ResilienceManager`/4-strategy/`resolve_conflicts` 는 production 소비자 0(**test-only**). 라이브 경로 = `EngineCommand`(`shared/src/lib.rs:189`) → `CommandExecutor::apply_command`(`engine/src/resilience/executor.rs:360`, `ExecutionPlan` 생산). cross-domain 해소는 manager PolicyEngine(Lua DPP)이 이미 수행("preserve Manager's cross-domain selection" 주석). **manager-less 이산 정책(switch/suspend)이 1급 요구**(사용자 확정)라 `ResilienceStrategy`/`CommandSource`/`LocalPolicy` 가 생존.

---

## 다음 작업 (순서)

1. **pressure/ 해체 reconciliation** (진입 문장). 결정할 것: `pressure/` 를 **완전 삭제**(format impl → `format/{standard,kivi}.rs`, offload → `format/offload/`, eviction → `stages/kv/eviction/`) vs **유지**(데이터/정책 home, 트리거만 `stages/`). 영향 = §0.4 "새 KV Format→`pressure/`"·"새 eviction→`pressure/eviction/`" 행 + §2.1 `pressure/`/`pressure/eviction/` 거주자 행 + `format/` 디렉토리 신설 여부. **갈등 상태**: 구 handoff(이번에 덮어씀) = 완전 삭제 / doc §2.1 連歴(line 195) = 유지. 어느 쪽 canonical 인지 user 재확정부터.
2. 결정 후 §0.4/§2.1 patch + 連歴 + 커밋.
3. **G4**(수치/임계) — handoff landmine: **R3/R4 사용자 리스크 트랙 결과 먼저 확인**.
4. **G5**(`DeviceTarget` variant + `StepInfo` 필드).
5. 이후 코드 = **Phase α-W**(`hardware.rs`/`capability/`/`pipeline.rs` 신설 + `stages/` 골격 + `CommandSource`/`CommandDispatcher`/`LoopControl` + `CommandExecutor` 분해 + `ResilienceAction`/`MemoryStrategy` 삭제 + `ResilienceStrategy` 시그니처 변경).

---

## Landmines / 미해결

- **pressure/ 해체 = 두 'Q1-Q6' 충돌 (최대 landmine)** — doc §2.1 連歴(type→file grill, "pressure/ 유지") vs 구 handoff(dissolution grill, "완전 삭제"). 이번 세션 미해결. doc 은 현재 유지 구조로 내부 정합. **어느 쪽 canonical 인지부터 결정.**
- **ResilienceStrategy 가지 = 재grill 금지** — §5.4 확정·커밋(`84880e7e`). friction-triggered 잔여 1개: setup-1회 명령(`SwapWeights`/`SetPartitionRatio`)이 OneShot Stage vs `LayerSlot` 직접 적용 — Phase α-W 진입 시 판정.
- **manager-less 이산 정책 = 로컬 센서 모니터 미구축 의존** — thermal/battery/usage 자율 수집 인프라 필요(현 production `SystemSignal` 생산자 = `dbus_transport`=manager 뿐).
- **리스크 트랙 = 사용자 소유** — R1~R6 재실행 금지. G4 전 R3/R4 결과 확인.
- **커밋 금지 대상**(working tree untracked): `.antigravitycli/`, `.claude/scheduled_tasks.lock`, `papers/.../microbench_full_matrix_2026_05_28/*`, `.agent/todos/handoff_microbench_full_matrix_p4_sweep_wait_2026_05_29.md`. 커밋 시 **파일명 지정**(`git add -A` 금지).

## 선행 문서
① 본 handoff ② SSOT `arch/pipeline_stage_design_v2.md`(§5.4 resilience 2-source·§0.4/§7 front-door·§2.1 type→file) ③ v1 동결 `arch/pipeline_stage_design.md`(R-5 원본 §5.3/§6.2) ④ 용어 `/CONTEXT.md`(3축) ⑤ 코드 근거: `shared/src/lib.rs:189`(`EngineCommand`), `engine/src/resilience/executor.rs`(`CommandExecutor`/`ExecutionPlan`), `engine/src/resilience/strategy/*`(4 strategy, test-only), `engine/src/session/resilience_adapter.rs`(`CommandSource` v1 trait).

## 자기점검
- 진입 문장으로 첫 명령 가능: ✓ "pressure/ 해체 reconciliation"
- 왜 멈췄나: ✓ 이번 세션 결론은 ResilienceStrategy, pressure 는 재결정 안 함
- 최대 landmine: ✓ 두 'Q1-Q6' 충돌(pressure/ 유지 vs 삭제)
- 검증: 설계 구체화라 게이트 = doc 자기완결 (코드 게이트 아님)
