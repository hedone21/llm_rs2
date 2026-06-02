# Handoff: Phase α-W-5 완료 (WeightFormat::apply_dispatch 배선, S25 partition bit-identical) → 잔여 3b/4

**작성**: 2026-06-02
**HEAD**: `ef723da7 feat(engine): Phase α-W-5b — WeightFormat::apply_dispatch 배선`
**브랜치**: `master` (메인 `/home/go/Workspace/llm_rs2`)
**다음 세션 진입 문장**: **"α-W-4 진행"**(CapabilityRegistry, hot-path 高위험) · 대안 **"α-W-3b 진행"**(resilience 2-source, Phase β 경계). **둘 다 device-gated, S25 USB 연결 시 가용.**

> 설계 SSOT = `arch/pipeline_stage_design_v2.md` §4.2. 트랙 전체 = [[project_pipeline_alpha_w]] (memory).

---

## TL;DR

확장 추론 파이프라인 Phase α-W 를 5 substep 으로 분해 진행 중. **이번 세션에 α-W-5 완료** (5a `3e5dc49f` 구조 + 5b `ef723da7` 배선). 누적 = α-W-1·2·3a·3-arch·**5**. **멈춘 이유**: 사용자가 메뉴에서 α-W-5 단일 선택 → 완료. **잔여 = α-W-3b/4** (전부 device-gated, S25 가용).

α-W-5 = `WeightFormat`(α-W-1 trait 표면)의 **첫 소비자** 생성 + tensor partition 설치 일반화. `PartitionedWeight` 2-fixed 필드(`gpu_slice`/`cpu_slice`) → `Vec<WeightSlice{tensor,backend,rows}>`, `prepare_tensor_partition` → per-layer `LayerSlot::apply_dispatch(&Hardware)` 위임. **bit-identical 게이트 통과** (S25 OpenCL `--tensor-partition 0.5`).

---

## 진행 상태 (검증 완료)

| substep | commit | 게이트 | 결과 |
|---|---|---|---|
| α-W-1 L2 스캐폴딩 | `0d12c81d` | host | additive |
| α-W-2 Hardware 흡수 | `1a1cd444` | device | S25 bit-identical |
| α-W-3a resilience 어휘 | `57629f26` | host | spec drift-sync + 87 PASS |
| α-W-3 arch drift-sync | `4ca4e6ff` | host | workflow + 7종 sweep |
| **α-W-5a** PartitionedWeight→Vec<WeightSlice> | `3e5dc49f` | host | build/clippy/test clean, 동작·시그니처 무변 |
| **α-W-5b** apply_dispatch 배선 | `ef723da7` | **device** | **S25 OpenCL `--tensor-partition 0.5` greedy gen=32 bit-identical** |

**α-W-5b 검증 수치**: host build clean / clippy `--workspace -- -D warnings` exit 0 / tensor_partition 21 + apply_dispatch focused 2 + inv_120 3 + eng_dat_092 5 PASS / layer_lint 신규 violation 0 / **S25 baseline `3aab681f` vs 5a+5b 생성 텍스트 "Hello！I'm a boy...Jack" 완전 일치**.

---

## α-W-5 구현 핵심 (재현·후속 참고)

- **설계**: `/tmp/alpha_w5_design.md` (세션 한정 — 핵심은 본 handoff + [[project_pipeline_alpha_w]] 에 흡수). §4.2 item 2 의 6 결정 전부 반영.
- **5 차원 adversarial 검증 워크플로우**(`wppt8b1t8`)로 blocker 2 + major 5 **사전** 적발: ① legacy/generate.rs 는 **활성 device-gate bin**(소비처 누락 위험), ② LayerSlot 에 idx 필드 부재, ③ init.rs Hardware::new 가 호출보다 뒤, ④ resolve() Option, ⑤ LayerDispatch/SliceSpec Clone 부재, ⑥ Full 분기 가드 verbatim, ⑦ INV-120 게이트 공백→host test.
- **impl 위치 = `impl WeightFormat for LayerSlot`** (`models/weights/slot.rs`). 레이어: slot(L3-inference)→format/hardware(L2 하향 허용)→tensor_partition(L3 동일도메인) = clean.
- **2-commit 분해**: 5a(구조, host — 접근자 #[inline] 로 hot-path bit-identical) → 5b(배선, device).
- **companion** = `hw.resolve(spec.hardware)` (baked-in cpu_backend 파라미터 제거). PartitionContext.cpu_backend **필드는 유지**(forward 매 토큰 resolve 회피 = INV-HOTPATH-DISPATCH).
- **INV-120**: prepare_tensor_partition per-slot 로직(gen counter/RCU/Release ordering)을 apply_dispatch 로 **verbatim** 이식. slot.rs focused unit test 2개가 회귀 가드.

## ★ device 게이트 절차 (α-W-5 확정 — partition 시나리오)

- bin = `legacy_generate`(완전 게이트). 명령: `python scripts/run_device.py -d galaxy_s25 legacy_generate --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf --tokenizer-path .../tokenizer.json --backend opencl **--tensor-partition 0.5** --prompt Hello -n 32 --temperature 0`
- baseline vs post: `git stash` 5b → `git checkout <pre-commit>` → 빌드+gen → `git checkout master` + `git stash pop` → 빌드+gen → 생성 텍스트 diff. (5b 미커밋 보호.)
- `--tensor-partition <r>` 이 zero-copy+rewrap+partition forward(forward_gen)+SOA+plan 전부 exercise. `[Partition] Prepared 84 weights with ratio 0.50` 로그로 path 활성 확인.

## 다음 작업 (택1)

1. **α-W-4** (device, **高위험 hot path**). `CapabilityRegistry` factory 배선 + `KiviAttentionBackend`/`GpuScoreAccess` capability/ 물리 정착 + `KiviCache.gpu_backend` 좁히기(per-call `as_kivi_attention` 제거). 게이트 = `--kv-mode kivi` bit-identical.
2. **α-W-3b** (device, **Phase β 경계**). `CommandSource`/`CommandDispatcher`/`LoopControl` 신설 + `CommandExecutor` 분해(ENG-ST-054 spec-ahead) + `PressureSource` impl + `LocalPolicy`. live DecodeLoop 본문 손대므로 가장 신중.

## Landmines / 교훈

- **cargo 가 권위 (3회째 재확인)**: 5a·5b 양쪽에서 IDE 진단이 E0560/E0615/E0277/E0061 다수 표시(mid-edit stale)했으나 `cargo build` + `cargo test --no-run` clean. **특히 `cargo build`는 tests/(integration spec) 미컴파일** → LayerSlot::new arity 같은 test 파일 오류는 **`cargo test -p llm_rs2 --no-run`(integration 포함) 으로만 검출**. clippy 게이트는 `--workspace -- -D warnings`(integration test 미포함)이 공식; `--all-targets` 의 spec test 경고 11건은 pre-existing(qnn/rpcmem/parity, 5a/5b 무관).
- **senior-implementer 위임 + "커밋 금지"**: 정상 동작(에이전트 커밋 안 함, 게이트 후 내가 단일 커밋). 5b 는 LayerSlot::new 시그니처 변경이 spec test 16개로 ripple → 에이전트가 전수 갱신.
- **커밋 금지 untracked**(working tree): `.antigravitycli/`, `.claude/scheduled_tasks.lock`, `papers/.../microbench_full_matrix_2026_05_28/*`, `.agent/todos/handoff_microbench_*.md`. 커밋 시 **파일명 명시**(`git add -A` 금지).
- **8T vs 6T**: 게이트가 8 threads 로 돎(CLAUDE.md 벤치 6T 권장). bit-identical 은 thread 무관(greedy 동일). 성능 측정 시엔 6T.

## 자기점검
- 진입 문장으로 첫 명령 가능: ✓ "α-W-4 진행" (또는 3b)
- 왜 멈췄나: ✓ 사용자가 α-W-5 단일 선택 → 완료
- 최대 landmine: ✓ cargo 권위(특히 `cargo test --no-run` 으로 integration arity 검출) + legacy 활성 bin
- 검증 게이트: ✓ host(build+clippy+test) 수치 + S25 `--tensor-partition 0.5` bit-identical 절차·결과
- device 가용: ✓ S25 USB, galaxy_s25 key, partition 게이트 절차 확립
