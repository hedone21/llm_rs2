# Handoff: Phase α-W-2 완료 (Hardware 흡수, S25 게이트 통과) → 잔여 3b/4/5

**작성**: 2026-06-02
**HEAD**: `1a1cd444 feat(engine): Phase α-W-2 — Hardware가 SessionInitCtx 4 secondary Arc 흡수`
**브랜치**: `master` (worktree 아님 — 메인 `/home/go/Workspace/llm_rs2`)
**작성자**: 메인 세션 (Claude, 오케스트레이터)
**다음 세션 진입 문장**: **"α-W-5 진행"**(partition 일반화, 중위험) · 대안 **"α-W-4 진행"**(Capability, hot-path 高위험) · **"α-W-3b 진행"**(resilience 2-source, Phase β 경계). **셋 다 device-gated, S25 USB 연결 시 가용.**

> 설계 SSOT = `arch/pipeline_stage_design_v2.md`. 트랙 전체 = [[project_pipeline_alpha_w]] (memory, 게이트 절차·stale 정정 포함).

---

## TL;DR

Phase α-W(확장 추론 파이프라인)를 5 substep으로 분해 진행 중. **이번 세션에 2 substep 추가 완료**: α-W-3 arch drift-sync(`4ca4e6ff`, host) + **α-W-2 Hardware 흡수**(`1a1cd444`, **device 게이트 통과**). 누적 완료 = α-W-1·3a·3-arch·2. **멈춘 이유**: 사용자 체크포인트 요청. **결정적 발견**: device(Galaxy S25)가 USB로 연결돼 있어 device-gated 작업 검증 가능(이전 handoff의 "device 부재" 가정은 틀렸음 — 그건 호스트 *로컬 GPU* 얘기였고 게이트는 S25/Jetson). 잔여 = α-W-3b/4/5, 전부 device-gated이나 S25 가용.

---

## 진행 상태 (검증 완료)

| substep | commit | 게이트 | 결과 |
|---|---|---|---|
| **α-W-1** L2 스캐폴딩 | `0d12c81d` | host | build+clippy+fmt clean, additive |
| **α-W-3a** resilience 어휘(ResilienceAction/MemoryStrategy 폐기→EngineCommand) | `57629f26` | host | spec drift-sync + 87 PASS |
| **α-W-3 arch drift-sync** | `4ca4e6ff` | host | workflow(4 arch 파일 edit→adversarial verify, 3 PASS + 30-engine §7 흐름 모순 1건 직접 보정) + 폐기 식별자 7종 arch/ 전수 sweep 살아있는 타입 0건 + .rs 무변경 |
| **α-W-2** Hardware가 SessionInitCtx 4 secondary Arc 흡수 | `1a1cd444` | **device** | host build+clippy clean / cargo test 신규 실패 0(24 실패 전부 opencl GPU-부재 + kv_cache RSS flaky, session/ 0건) / **S25 Adreno OpenCL greedy gen=32 bit-identical**(생성 텍스트 baseline 완전 일치) |

- **신규 실패 0** (전 커밋). α-W-2 = 9 파일(+147/-68): SessionInitCtx + BatchRunCtx/DecodePrologueCtx/PrefillCtx/StandardHappyCtx 4 Arc→`Arc<Hardware>`, SwapDispatchCtx(borrowed)→`&'a Hardware`, init.rs `Hardware::new(cpu,gpu,None,cpu_mem,gpu_mem)`, argus_cli+legacy_generate ctx 생성부 배선.
- **저위험 전략**(핵심): 각 소비자에서 `hardware` destructure 직후 동일 이름 로컬을 `resolve()`로 재바인딩 → 본문 ~40 사용처 **무변경**. 의미 동등성: gpu_backend/gpu_memory Option이 init.rs tuple에서 항상 동기(both Some/both None)라 `resolve(Gpu)`의 `device.unwrap_or(host)` fallback 불일치 불가.

---

## ★ device 게이트 절차 (이번 세션 확립 — 잔여 3b/4/5 에 재사용)

- **연결**: Galaxy S25 USB (`adb -s R3CY408S5SB`). registry key = **`galaxy_s25`** (`devices.toml`, features=`[opencl,vulkan,qnn,htp_fastrpc]`).
- **게이트 bin = `legacy_generate`** (NOT `generate` — 분할됨). `legacy_generate`(`engine/legacy/generate.rs`, 동결 monolith)가 4-Arc 소비자 전부(batch/decode_fallback/prefill/standard_happy/swap)를 타는 **완전 게이트**. `argus_cli`(신규 single-prompt)는 standard happy만 = 부분 게이트.
- **명령**: `python scripts/run_device.py -d galaxy_s25 [--skip-build] legacy_generate --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json --backend opencl --prompt Hello -n 32 --temperature 0`
- **gotcha**: `--prompt`은 **단일 토큰**만(run_device.py adb shell 전달이 공백 분리 → multi-word는 "quick"이 subcommand로 오인). greedy(temp 0)→결정적→생성 텍스트 diff로 bit-identical 판정.
- **절차**: baseline(현 HEAD) 빌드+gen → 코드변경 → 재빌드+gen → 두 생성 텍스트 diff. android 빌드 ~55s.
- 잔여 substep별 게이트 시나리오: α-W-4=`--kv-mode kivi`, α-W-5=`--tensor-partition <r>`, α-W-3b=standard happy(+ manager 경로).

---

## 다음 작업 (택1)

1. **α-W-5** (device, 중위험). `WeightFormat::apply_dispatch` = `prepare_tensor_partition` 일반화 + `PartitionedWeight`(2-fixed)→`Vec<WeightSlice>`. 비교적 self-contained. 게이트 = `--tensor-partition`.
2. **α-W-4** (device, **高위험 hot path**). `CapabilityRegistry` factory 배선 + `KiviAttentionBackend`/`GpuScoreAccess` capability/ 물리 정착 + `KiviCache.gpu_backend` 좁히기(per-call `as_kivi_attention` 제거). 게이트 = `--kv-mode kivi` bit-identical.
3. **α-W-3b** (device, **Phase β 경계**). `CommandSource`/`CommandDispatcher`/`LoopControl` 신설 + `CommandExecutor` 분해(ENG-ST-054 spec-ahead) + `PressureSource` impl + `LocalPolicy`. live DecodeLoop 본문 손대므로 가장 신중. spec ENG-ST-054 이미 LoopControl 기술.

---

## Landmines / 미해결 / 안 가본 길

- **stale 컨텍스트 (세션 중 발견·정정)**: `engine/src/bin/generate.rs` **삭제됨** → `legacy_generate`+`argus_cli` 분할 완료(`7065196c`). no-mod.rs sweep 완료(`3895e17d`). **CLAUDE.md(generate.rs:779 등) + MEMORY.md는 일부 stale** — MEMORY.md "Key File Paths"는 정정함(이번 세션). CLAUDE.md 본문은 미수정(별도 작업). 진입 전 grep으로 파일 존재 재확인 권장.
- **cargo가 권위** ([[feedback_delegated_agent_autocommit]] 재확인): α-W-2 구현 후 IDE 진단이 다수 E0560/E0026 컴파일 에러를 표시했으나 **mid-edit stale**였고 `cargo build`는 clean. senior-implementer 자기보고·IDE 진단 **둘 다 lag** — `cargo`로 직접 검증해야 함.
- **senior-implementer 위임 + auto-commit**: 위임 프롬프트에 **"커밋 금지 — 오케스트레이터가 device 게이트 후 커밋"** 명시 → 정상 동작(에이전트 커밋 안 함, 게이트 후 내가 단일 커밋). 위임 전 tree 청결(.rs 무변경) 확인이 안전 기준선.
- **게이트 커버리지**: α-W-2의 S25 게이트는 standard happy path(cpu_backend_arc + Hardware 구성 + resolve(Cpu))를 exercise. decode_fallback의 gpu/cpu_memory runtime 경로(SwitchHw migration)는 `--switch` 시나리오라 미실행 — 대신 host build(타입: backend↔memory 오매핑 컴파일 불가) + diff 리뷰(cpu↔gpu 명시) + 의미 동등성 증명으로 커버. 잔여 substep도 동일 원칙(타입+리뷰+가능한 시나리오 게이트).
- **커밋 금지 untracked**(working tree): `.antigravitycli/`, `.claude/scheduled_tasks.lock`, `papers/.../microbench_full_matrix_2026_05_28/*`, `.agent/todos/handoff_microbench_*.md`. 커밋 시 **파일명 명시**(`git add -A` 금지).
- **8T vs 6T**: 게이트 실행이 8 threads로 돎(CLAUDE.md는 벤치 6T 권장). bit-identical은 thread 수와 무관(greedy 토큰 동일)이라 게이트엔 영향 없음. 단 **성능 측정 시엔 6T**.

## 자기점검
- 진입 문장으로 첫 명령 가능: ✓ "α-W-5 진행" (또는 4/3b)
- 왜 멈췄나: ✓ 사용자 체크포인트 요청 (α-W-2 device 게이트 통과 직후, 잔여는 더 큰 규모)
- 최대 landmine: ✓ stale 컨텍스트(generate 분할) + cargo 권위(IDE/에이전트 lag) + 게이트 bin=legacy_generate·단일토큰 prompt
- 검증 게이트: ✓ host(build+clippy+test) 명령·수치 + S25 bit-identical 절차·결과 명시
- device 가용: ✓ S25 USB 연결, galaxy_s25 registry key, 게이트 절차 확립
