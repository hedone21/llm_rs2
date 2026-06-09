# Handoff: ADR-0008 e2e eviction bin 배선 — false-positive 정정 + opaque eviction e2e 증명

**작성**: 2026-06-09
**HEAD**: `f0d92c66 fix(session): loop builder가 ctx.kv_caches 소비 — --kv-format opaque가 decode loop 도달`
**브랜치**: master (미푸시)
**작성자**: 메인 세션

**다음 세션 진입 문장**: **"ADR-0008 e2e eviction 배선 완료(opaque eviction 실증) — 다음은 (a) q4_0 typed eviction SIGSEGV 조사 vs (b) GATE-C(.so dlopen, 북극성 최종) 착수 우선순위 grill 후 착수."**

---

## TL;DR

직전 세션이 "B 먼저 → A"(e2e eviction bin 배선 → GATE-C) 를 선택. grill 결과 task B 의 진짜 gap 은 "bin 배선"이 아니라 **loop builder dead-wiring** 이었음을 발견: `build_inference_ctx` 가 `--kv-format` opaque 를 올바로 dispatch 했으나 `standard_happy.rs`/`experiment_run.rs` 가 `ctx.kv_caches` 를 drop 후 builder 가 typed 를 재할당 → **opaque 가 decode loop 에 도달 못 함**. 따라서 **지난 세션 ADR-0008 의 "synth_q4 == q4_0 token-identical e2e" 는 false positive**(둘 다 typed). 수정(builder 가 ctx.kv_caches 소비) 후 opaque 가 실제 decode loop 를 타고, **argus_bench + signal_injector host e2e 로 opaque eviction 정상 발화 실증**. **멈춘 이유**: task B 완주. 부수로 q4_0 typed eviction SIGSEGV(pre-existing) 발견 → 별도 결정.

---

## 진행 상태 (1 커밋, 명시 5파일, 무회귀)

| 커밋 | 내용 | 게이트 |
|---|---|---|
| `f0d92c66` | `build_standard_loop`/`build_bench_loop` 가 `ctx.kv_caches` 소비(내부 `alloc_standard_kv_caches` 제거) + `[DecodeLoop] kv storage` 로그 + orphan 필드(`StandardHappyCtx.initial_kv_capacity`/`kv_type`) 제거 | build OK · clippy `--workspace` clean · lib **1258/0** |

**e2e 실측** (Qwen2.5-1.5B q4_0, CPU greedy, argus_bench + signal_injector TCP loopback):

| kv-format | opaque-flow | eviction(h2o 0.5, 337-tok prompt) | 출력 |
|---|---|---|---|
| **synth_q4 (opaque)** | `[DecodeLoop] kv storage = OPAQUE` ✅ | `final_pos` 337→**215** (~50% prune), EXIT 0 ✅ | coherent (f16-KV 급) |
| f16 (typed) | typed | `final_pos` 215 ✅ (opaque와 parity) | coherent |
| **q4_0 (typed)** | typed | 🔴 **SIGSEGV (EXIT 139)** | — |

token-identity 차등(eviction 없이, n=24): opaque "Paris. The population ... 2 million..." (coherent) ≠ typed q4_0 "Paris.改错"(degraded). **opaque ≠ typed q4_0 는 정상**(f32 floor attention vs native kernel; opaque 가 오히려 고품질). ADR-0008 Status/§4 정정 완료.

---

## 다음 작업 (새 결정 필요 — grill 먼저)

1. **q4_0 typed eviction SIGSEGV 조사** — `argus_bench --kv-format q4_0 ... eviction h2o --keep-ratio 0.5` + signal_injector `kv.evict_h2o`(긴 프롬프트, eviction floor 초과 시) → exit 139. f16-typed·synth_q4-opaque 는 정상이므로 **q4_0 native eviction 경로 한정**. 의심 = q4_0 attention_scores(score-free h2o recency degrade) 또는 q4_0 compact/prune. **pre-existing**(본 fix 는 typed arm 무수정). 심각도 高(production q4_0 KV + h2o eviction crash). gdb 백트레이스 미확보(타이밍 의존, gdb 하 미재현).
2. **GATE-C** = `.so` cdylib dlopen 승격(북극성 최종, ADR-0007 D6). 런타임 registry + register_plugin C-ABI dual-wiring + dlopen 배선 + bit-identical 재증명. host-implementable, GPU/perf 만 device. `panic=abort`↔`catch_unwind` 충돌이 난점.
3. (잔여 deferred) write encoder family(q8_0/q4_1) / GPU opaque arm(device).

---

## Landmines / 미해결 / 핵심 발견

- **opaque-flow 증거는 alloc-시점 로그가 아니다**: `build_inference_ctx` 의 `"KV format: ... (opaque)"` 는 caches 가 drop 돼도 찍힌다 → 지난 false positive 를 못 잡은 직접 원인. 진짜 증거 = `[DecodeLoop] kv storage = OPAQUE`(builder 가 ModelForward 소비 직전, `build_*_loop.rs`). 미래 리팩터가 다시 끊으면 이 로그가 즉시 노출.
- **opaque ≠ typed q4_0 는 버그 아님**: opaque = q4_0 저장 + f32 floor attention(dequant→f32), typed q4_0 = native kernel. compute 경로가 달라 token 발산 정상(둘 다 valid). bit-identity 주장은 **unit gate(F32 round-trip)** 한정. e2e 에서 token-identity 게이트 금지(재발 방지).
- **eviction min-cache floor**: 짧은 프롬프트(≤~40 token)는 `--eviction-target-ratio`/min_kv_cache floor 미달로 eviction no-op(format 무관). e2e 는 **긴 프롬프트 필수**(≥~300 token). 디버깅 시 이거 모르면 "eviction 안 됨"으로 오판(실제로 floor 정상 동작).
- **argus_bench eviction = signal-driven**: `self.eviction`(EvictionStage) 은 NoOp(`with_eviction` 미호출). eviction 은 오직 `plan.evict`(resilience KvEvict directive, `cmd_source.poll`→executor) 경로. host e2e = `signal_injector --socket tcp:127.0.0.1:PORT -f schedule.json`(listen) → argus_bench `--resilience-transport tcp:...`(connect). schedule = `[{delay_sec, directive:{seq_id, commands:[{"type":"kv.evict_h2o","keep_ratio":0.5}]}}]`. directive 는 sticky(evict_applied 1회 gate). CLI `eviction h2o --keep-ratio 0.5` + `--protected-prefix 4` 도 필요(CacheManager 빌드).
- **q4_0 typed eviction SIGSEGV** (위 §1) — production 심각 버그, 별도 조사.
- **legacy_generate 는 main tree 부재**: worktree(`/.claude/worktrees/b5_trait_extension/`)에만 존재. eviction vehicle = `argus_bench`(main tree). 핸드오프의 "argus vs legacy" 프레이밍은 무효였음.
- **engine/Cargo.toml drift 유지**: `microbench_score_readback` bin 엔트리(untracked `microbench/score_readback.rs`) 미커밋 보존. 이번 커밋은 src 5파일만(Cargo.toml 미포함).

---

## 참조

- SSOT: `docs/adr/0008-opaque-kv-production-integration.md`(Status 정정 + §4 e2e eviction 완료/q4_0 segfault).
- 코드 앵커: `engine/src/session/assembly/build_standard_loop.rs`·`build_bench_loop.rs`(kv_caches 소비 + 로그) / `standard_happy.rs`·`experiment_run.rs`(drop 제거, ctx.kv_caches 전달) / `bin_setup.rs`(orphan 필드 제거) / `engine/src/resilience/executor.rs:411`(KvEvictSliding→EvictPlan) / `engine/src/session/decode_loop.rs:168`(plan.evict→try_evict).
- e2e schedule: `/tmp/evict_h2o_schedule.json`(미커밋, 내용은 위 Landmines 참조).
- 선행 handoff: `handoff_opaque_kv_production_adr0008_2026_06_08.md`(ADR-0008 full scope).
