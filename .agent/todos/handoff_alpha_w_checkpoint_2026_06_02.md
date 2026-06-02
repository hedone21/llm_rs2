# Handoff: Phase α-W 진입 (α-W-1 + α-W-3a 완료) → 잔여 substep

**작성**: 2026-06-02
**HEAD**: `57629f26 feat(resilience): Phase α-W-3 — ResilienceAction/MemoryStrategy 폐기, EngineCommand 단일 이산 어휘`
**브랜치**: `master` (worktree 아님 — 메인 `/home/go/Workspace/llm_rs2`)
**작성자**: 메인 세션 (Claude, 오케스트레이터)
**다음 세션 진입 문장**: **"arch/ drift 정리"** (host, 바로 가능 — α-W-3 거버넌스 마무리) · 대안 **"α-W-2 진행"** (device, Hardware 흡수 — 4/5 선행)

> 설계 SSOT = `arch/pipeline_stage_design_v2.md`. 트랙 전체 = [[project_pipeline_alpha_w]] (memory). substep 분해·매핑 raw = workflow `wq32w40ql` 결과(`/tmp/claude-1000/.../tasks/wq32w40ql.output`, 2504 lines).

---

## TL;DR

Phase α-W(확장 추론 파이프라인, §9 기준 2–3주 大단계)를 **5 substep으로 분해**해 진입. **α-W-1**(L2 타입 스캐폴딩, additive) + **α-W-3a**(ResilienceAction/MemoryStrategy 폐기 → EngineCommand 단일 어휘) 완료·커밋·검증. **멈춘 이유**: 사용자 체크포인트 요청. 남은 substep 중 **α-W-2/3b/4/5는 device-gated**인데 이 호스트는 GPU 부재(`CL_DEVICE_NOT_FOUND`)라 bit-identical 검증 불가 → S25/Jetson 필요. 즉시 가능한 host 작업 = **arch/ drift 정리**(α-W-3가 노출한 구 어휘 미러링 4개 arch 파일).

---

## 진행 상태 (검증 완료)

| substep | commit | 게이트 | 결과 |
|---|---|---|---|
| **α-W-1** L2 스캐폴딩(`pipeline`/`hardware`/`capability`/`format`/`stages`) | `0d12c81d` | host | release build + clippy -D + fmt clean / layer_lint 신규 0 / 적대적 리뷰 2종 PASS. byte-identical(미배선) |
| **α-W-3a** ResilienceAction·MemoryStrategy 폐기 + EngineCommand 단일 어휘 + spec drift-sync | `57629f26` | host | release build + clippy --workspace -D + fmt clean / ENG-ST-052(11)·060(5)·INV-072(4)·073(5)·074~076/071 PASS / resilience lib 87 PASS / `--features resilience` 컴파일 OK / layer_lint 신규 0 |

- **신규 실패 0** (양 커밋). 잔존 실패 = 호스트 GPU 부재(`test_inv_140`/`test_host_ptr_pool`/`test_inv_130`, run마다 flaky) + pre-existing baseline drift(`test_inv_layer_001` — backend/opencl·htp_fastrpc cross-backend import 6건, master 동일·α-W 무관).
- α-W-3a canonical 결정: **LimitTokens/RejectNew drop**(EngineCommand 등가 부재·미배선; Throttle/Suspend 흡수). memory→graded `Pressure`, strategy 3개(thermal/energy/compute) 잔존. spec ENG-ST-052/060 7규칙→4규칙, SSOT §5.4 "20-variant"→실측 18 정정.

---

## 다음 작업 (택1, 우선순위 순)

1. **arch/ drift 정리** (host, 바로 가능 — α-W-3 거버넌스 미완분). `arch/{31-engine-state,30-engine,40-cross-cutting,weight_swap}.md`가 구 `ResilienceAction`/`MemoryStrategy`/`react(.., mode)` 어휘를 canonical로 미러링 → §5.4 superseded 포인터로 동기화. **Architect 도메인** → `architect` 에이전트 위임 권장. `arch/weight_swap.md §2.8`의 `ResilienceAction::SwapWeights` "engine internal enum" 전제 붕괴(→ `EngineCommand::SwapWeights` 단일). 검증: 문서만(빌드 영향 0) + `git grep -n ResilienceAction arch/` = 0(또는 superseded 주석만).

2. **α-W-2 진행** (device-gated). `Hardware`(`hardware.rs`, 이미 신설)가 `SessionInitCtx`(`session/init.rs:22`) 4 Arc(`cpu_backend_arc`/`gpu_backend_arc`/`cpu_memory_arc`/`gpu_memory_arc`) 흡수 + `resolve()` 호출지(switch/migrate/partition) 배선. 소비자 ~7 파일(eval/runner·batch/args·prefill·decode_fallback/swap_dispatch). **α-W-4/5의 선행.** 검증 게이트: 호스트 build+test + **S25 OpenCL bit-identical**(GPU 필요).

3. **α-W-3b** (device, production 접촉). `CommandSource`/`CommandDispatcher`/`LoopControl` 신설 + `CommandExecutor`(executor.rs) 분해(A-1 3역할, ENG-ST-054) + `PressureSource` impl(Manager/Local) + `LocalPolicy`. spec ENG-ST-054는 이미 LoopControl 기술(spec-ahead-of-code). `CommandSource::poll`(v1 session/traits.rs)→Vec<EngineCommand> 변경은 DecodeLoop 본문 손대므로 Phase β 경계 주의.

(α-W-4 CapabilityRegistry 배선[hot path, 高위험], α-W-5 WeightFormat::apply_dispatch는 4=2 후행.)

---

## Landmines / 미해결 / 안 가본 길

- **device 게이트 = S25/Jetson 필수**: 이 호스트 GPU 없음. α-W-2/3b/4/5는 author는 가능하나 bit-identical은 디바이스에서. host에서 끝낼 수 있는 건 arch/ drift + (additive면) 스캐폴딩뿐.
- **implementer 위임 + auto-commit 함정** ([[feedback_delegated_agent_autocommit]]): 미커밋 작업 든 채 implementer 위임 시 자기 파일만 자동커밋(`76cfc1f0` = 비컴파일 중간 커밋) → `git reset --soft`로 통합함. **위임 전 내 작업 커밋 OR "커밋 금지" 명시 OR worktree isolation.** 서브에이전트 자기보고·IDE diagnostic 둘 다 lag 가능 — `cargo`가 권위.
- **α-W-1 미완 layout**: `capability/`는 `kivi_attention.rs`/`gpu_score.rs`(backend.rs re-export shim)만 — `KiviAttentionBackend`/`GpuScoreAccess` 물리 이동 + `score_collector`/`tier_movable`(빈 trait 금지로 미생성)는 α-W-4. `format/`은 `weight_format.rs`만 — `kv_cache_format.rs`(KVCacheOps rename)는 α-K. `stages/`는 빈 골격(layer_lint 미분류 unknown — 실 Stage 입주하는 α-K에서 분류 확정).
- **arch/ drift는 α-W-3 커밋(`57629f26`)에 미포함** — spec/docs는 포함, arch/만 후속. 위 다음작업 #1.
- **`pressure/`→`kv/`, KVCacheFormat trait 이동은 α-K**(α-W 아님). α-W-2/4/5는 신설·배선 위주.
- **커밋 금지 untracked**(working tree): `.antigravitycli/`, `.claude/scheduled_tasks.lock`, `papers/.../microbench_full_matrix_2026_05_28/*`, `.agent/todos/handoff_microbench_full_matrix_p4_sweep_wait_2026_05_29.md`. 커밋 시 **파일명 명시**(`git add -A` 금지).

## 자기점검
- 진입 문장으로 첫 명령 가능: ✓ "arch/ drift 정리" (또는 "α-W-2 진행")
- 왜 멈췄나: ✓ 사용자 체크포인트 요청 + 잔여 substep device-gated(호스트 GPU 부재)
- 최대 landmine: ✓ device 게이트 = S25/Jetson 필수 + implementer auto-commit 함정
- 검증 게이트: ✓ host(build+clippy+test+layer_lint) 명령·수치 명시 / device는 S25 bit-identical
