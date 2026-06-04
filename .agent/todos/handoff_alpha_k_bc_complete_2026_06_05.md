# Handoff: α-K BC 종결 ✅ — KVCacheOps trait 완전 폐기 (5-F 완료, device PASS)

**작성**: 2026-06-05 (메인 세션)
**HEAD**: `9326a096` refactor(engine): 5-F F4 — fmt 접미사 제거 — **origin/master push 완료(ahead 0)**
**브랜치**: `master`
**다음 세션 진입 문장**: **"α-K BC 종결 — 다음 트랙 결정 (β / argus-bench / F4 잔여 중 택1)"**

---

## TL;DR
α-K BC **완주**. `KVCacheOps`(generic monomorphization) → `KVCacheFormat`(concrete-handle/StandardFormat) 패러다임 통일 완료. 5-F 비가역 cutover(legacy bin + OLD generic forward chain + KVCacheOps trait 삭제) 를 F0~F4 7 커밋으로 수행하고 **S25 device 게이트 PASS**(fmt-only ≡ frozen baseline bit-identical + avg_tbt Δ≤±0.5%). `grep KVCacheOps engine/src` 함수/trait/impl **0건**. **왜 멈췄나**: α-K BC 전체 목표(trait 폐기) 달성 — 다음은 별개 트랙(β/argus-bench)이라 사용자 결정 대기.

---

## 진행 상태 (5-F, 전부 push됨)
| 단계 | commit | 게이트 |
|---|---|---|
| HB-3 verify parking | `265f52b0` | verify=argus-bench 대기 parking(엔진 bin 폐기) |
| F3-defer offload inherent | `b54898db` | offload 62 test PASS |
| F0 게이트 상수화 | `8e7ffc67` | fmt=production 기본, non-opencl 1239 PASS |
| frozen baseline | `fec8ad23` | S25 캡처(F16 weight, f16/f32/q4 sig+avg_tbt) |
| F1 legacy 폐기 (★비가역) | `d5ed71d2` | build+clippy --workspace clean |
| F2a ModelForward fmt-only | `ba33ac86` | decode_fallback/prefill/parity 삭제, 1220 PASS |
| F2b OLD chain 삭제 | `05c019ee` | forward_into<C>/forward_gen<C>/execute<C> 등, 1218 PASS |
| F2c 잔여 generic 소비자 | `7623f441` | execute<C>+execute_plan_for_kivi 삭제+struct bound |
| F3 KVCacheOps trait+impl×3 폐기 | `102f0461` | grep 코드 0건, 1217 PASS |
| F4 fmt 접미사 제거 | `9326a096` | *_fmt→base rename, fmt clean |

### device 게이트 결과 (S25 OpenCL `--opencl-rpcmem`, 비가역 acceptance)
- **bit-identical 3/3**: argus(fmt-only, post-cutover) sig == frozen baseline(`fec8ad23`) — f16 `304f4ada` / f32 `684d01d9` / q4 `1cfba273`, build_plan SUCCESS + wrap 발화(non-vacuous).
- **avg_tbt Δ ≤ ±0.5%**(median n=5): f16 −0.31% / f32 +0.22% / q4 −0.41% ≪ +3%. monomorphization perf 회귀 0(설계 예측 neutral 확인).
- 모델 = `qwen2.5-1.5b-f16.gguf`(F16 weight 필수 = build_plan SUCCESS = 비-vacuous). baseline 레코드: `.agent/todos/frozen_baseline_alpha_k_5f_2026_06_05.md`.

---

## 다음 작업 (α-K 후행 트랙 — 사용자 택1)
α-K BC 자체는 **완료**. 아래는 별개 트랙:
1. **β (DecodeLoop PipelineStage 재작성)** — α-K 완주가 선결이었음. **α-W-3b(defer-B, resilience 2-source)가 β 안에서 해소**. SSOT `arch/pipeline_stage_design_v2.md`. 大작업.
2. **argus-bench (비-happy 모드 fmt 재구현)** — 5-F 가 `decode_fallback`(eviction-during-decode + weight-swap, legacy 전용) 삭제 → 해당 production 모드가 **reject-only**. argus_cli 가 reject 하는 eviction/swap/offload/KIVI/chat 을 fmt 경로로 재구현하는 family bin. **verify.py 재배선(HB-3 parking 해제)도 여기서**.
3. **F4 잔여 (forward_gen_fmt / forward_prefill_fmt rename)** — 메서드명이자 모듈명(`forward_gen_fmt.rs`/`forward_prefill_fmt.rs`)이라 sed 충돌(`forward_gen.rs` 기존). 파일/mod 재구성 필요한 trivial chore. 우선순위 낮음.
4. **Jetson device 게이트** — α-K flip 은 opencl 전용(plan path CUDA 부재). Jetson 은 dyn fallback 만. 최종 게이트에 Jetson 포함 여부 미결.

---

## Landmines / 미해결 (R6)
- **★비가역 완료**: F1~F4 는 legacy + OLD chain + trait 영구 삭제. revert 는 `git revert 102f0461~5..102f0461` 류(비권장 — device 게이트 PASS 확정). frozen baseline `.out` 은 S25 `/data/local/tmp/blA_*.out`(legacy 삭제됐어도 출력 보존) + `cut_*.out`.
- **decode_fallback 삭제 = capability 축소**: weight-swap / eviction-during-decode production 모드가 5-F 후 **사라짐**(argus_cli reject). argus-bench 재구현 전까지 불가. "legacy disposable" 결정 + 문서화된 plan 과 정합이나 명시 인지 필요.
- **score-driven UER arm 미검증**: chat 은 eviction 에 score 미공급(h2o≡sliding) → `force_evict_with_scores` arm 은 chat dead-path. eval/비-chat fmt 경로의 score-driven eviction 동치는 별도 검증 필요(future, (3d) S4 스코프 밖).
- **F2c 보존 struct**: `OffloadForwardArgs<'a, C>`(구 TransformerModelForwardArgs, bound 제거) = offload fmt 전용. `TransformerModelForwardArgs<'a>`(구 FmtArgs) = standard. 2개 공존(정상).
- **KVCacheOps 주석 잔존**: branch-by-abstraction migration historical 노트(standard_format/kivi_format/offload_format 등 module doc) archival 보존 — 코드 0건이라 무해.
- **OffloadForward(chat offload) prefill preload 사전 버그**(5-B 식별, broken+비결정) = 5-F 무관 별도 트랙.
- **host opencl test abort**: `memory::opencl::unified` 등은 host GPU 부재로 abort/fail(비-회귀). 회귀 게이트 = `cargo test -p llm_rs2 --lib -- --skip opencl`(RSS madvise 1건 = 병렬압박 flaky, 격리 PASS).

---

## 자기점검
- 진입 문장? ✓ "α-K BC 종결 — 다음 트랙 결정 (β / argus-bench / F4 잔여 중 택1)"
- 왜 멈췄나? ✓ α-K BC 목표(KVCacheOps 폐기) 달성, 다음은 별개 트랙(사용자 결정)
- 최대 landmine? ✓ decode_fallback 삭제로 weight-swap/eviction-decode 모드 reject-only(argus-bench 재구현 대상)
- 게이트 수치? ✓ device bit-identical 3/3 + avg_tbt Δ≤±0.5% / grep KVCacheOps 0건 / non-opencl 1217 PASS
- 길이? ✓ 상세 = 10 커밋 메시지 + `frozen_baseline_alpha_k_5f_2026_06_05.md` + 각 design_alpha_k_*.md
