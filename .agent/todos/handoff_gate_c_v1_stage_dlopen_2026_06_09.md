# Handoff: GATE-C v1 — Stage 축 `.so` dlopen plugin 승격 완주

**작성**: 2026-06-09
**HEAD**: `5a6a44bc test(engine): GATE-C C5 — dlopen plan-identity re-proof 게이트`
**브랜치**: master (미푸시)
**작성자**: 메인 세션

**다음 세션 진입 문장**: **"GATE-C v1(Stage 축 .so dlopen) 완주 — 다음은 GATE-C v2(Format 축 .so, ADR-0009 D4) 설계 grill, 또는 argus_bench eviction-firing e2e 검증(directive→policy 매핑 확인 선행)."**

---

## TL;DR

북극성("zero-compile `.so` plugin 설치")의 **Stage 축**을 런타임 dlopen 으로 승격 완료. 설계 grill(Q1–Q7 잠금) → ADR-0009 → C1–C5 구현. plugin 은 단일 `register_kv_stage_v1` C-ABI 심볼 export, `&dyn StageCtx` 는 `#[repr(C)]` fn-ptr 테이블로 평탄화, `KVCachePlan` 은 plugin-arena→host-copy→plan_free 마샬링. 정적 linkme 경로 보존(가산). **완료 게이트 = plan-identity**(dlopen stage 의 plan == 알려진 정답) 통과. **멈춘 이유**: ADR-0009 D7 정의 게이트 충족 + production CLI smoke green → v1 마일스톤 완주. 다음 = Format 축(v2) 또는 full eviction e2e.

---

## 진행 상태 (5 커밋, 무회귀)

| 커밋 | 내용 | 게이트 |
|---|---|---|
| `8836f88f` | ADR-0009 작성 (Q1–Q7 + ADR-0007 D6 해결) | docs |
| `851e4d03` | **C1** technique-api: StageCtxAbi/PlanAbi/PluginVTableAbi(`#[repr(C)]`) + AbiStageCtx 어댑터 + PlanArena + `register_kv_stage!` 매크로 | technique-api 15/15 |
| `c1e4b986` | **C2** engine: DYN_REGISTRY OnceLock + register_dynamic_stages(dlopen+abi+충돌 fail-fast) + make_stage + DynStage(vtable 마샬링) + shim 8종. libloading → non-optional 코어 | lib 1260/0, clippy clean |
| `88926e0c` | **C3** 호출부 2곳 find_stage→make_stage + `--load-plugin` CLI flag(build_inference_ctx 배선) | lib 1260/0, --help 노출 |
| `a5ae6fd3` | **C4** example-keep-recent cdylib화(`crate-type=["cdylib","rlib"]` + plugin-cdylib feature) + register_kv_stage! | rlib 2/2, cdylib nm `register_kv_stage_v1` T, feature OFF→심볼 부재 |
| `5a6a44bc` | **C5** `engine/tests/gate_c_dlopen_equivalence.rs` plan-identity + merge + reject 2종 | 통합 테스트 1/1 |

**최종 게이트(전부 green)**: technique-api 15/15 · lib **1260/0**(opencl skip) · clippy `--workspace -D warnings` clean · GATE-C 통합 테스트 1/1 · release fat-LTO 빌드 42s green · **production `argus_cli --load-plugin target/release/libexample_keep_recent.so` → startup dlopen 등록 + coherent 추론**("...Paris. It has a population...").

---

## 다음 작업 (택1, grill 권장)

1. **GATE-C v2 = Format 축 `.so`** (ADR-0009 D4 순서). `KVFormat` trait(2-method: name+`KVLayoutDesc` descriptor)는 콜백 0이라 C-ABI 표면이 Stage 보다 단순(register_kv_format_v1 + descriptor POD). **단, WRITE encoder 가 per-token hot-path** → LTO 단절 + S25 TBT Δ≤+3% **device 게이트** 필요(ADR-0007 L2). vehicle = synth-q4-format(이미 KV_FORMATS 등록). C1 인프라(vtable 패턴/OnceLock 병합/마샬링) 재사용.
2. **full eviction-firing e2e** (선택, 검증 보강). `argus_bench --load-plugin .so --eviction-policy example_keep_recent` + signal_injector(`/tmp/evict_h2o_schedule.json` 잔존) + 긴 프롬프트(≥300 tok, min-cache floor)로 `.so` stage 가 **실추론 중 실제 eviction** 하는지. **선행 확인 필수**: EngineCommand eviction directive(`kv.evict_h2o`/`kv.evict_sliding`)가 `--eviction-policy` 설정 정책(example_keep_recent)을 실행하는지, 아니면 directive 가 정책을 재선택하는지(`engine/src/resilience/executor.rs` + build_bench_loop). 후자면 example_keep_recent 트리거 불가 → 새 directive 어휘 또는 generic evict 필요.
3. (잔여 deferred) GATE-C v3 = Backend capability 축(`BackendCapability` trait 메서드 미확정 + GPU device-only, 최후순위).

---

## Landmines / 핵심 발견

- **linkme proc-macro 는 `linkme::` 경로 하드코딩** → plugin crate 가 linkme 를 **직접 dep 해야** 한다(technique-api 재노출로 못 없앤다). 확장 비용 = technique-api + linkme 2 deps. `register_kv_stage!` 는 `#[$crate::distributed_slice(...)]`(매크로 재노출, `pub use linkme::distributed_slice`) + `#[unsafe(no_mangle)]`(Rust 2024) 사용.
- **`#[no_mangle]` 충돌 차단 = `plugin-cdylib` feature 게이트**: 매크로의 동적 C-export 는 `#[cfg(feature="plugin-cdylib")]`. 정적 force-link 빌드(dev-dep)는 feature OFF → 심볼 미emit → 다중 정적 plugin 충돌 없음. `.so` 빌드만 `--features plugin-cdylib`.
- **자기-충돌**: example_keep_recent 가 정적 등록(dev-dep linkme)된 바이너리에서 같은 이름 `.so` dlopen 시 충돌 reject. C5 통합 테스트는 example_keep_recent 를 **Rust 미참조**(`.so` 만 dlopen)해 정적 등록을 안 끌어와 회피. plan-identity 는 known-answer(KeepRecent 알고리즘)로 검증.
- **directive→policy 매핑 미확인**(다음작업 #2 선행): argus_bench eviction 은 signal-driven(plan.evict). example_keep_recent 가 signal 로 트리거되는지 미검증 — C5 는 unit(plan) 레벨, full e2e 는 이 매핑 확인 필요.
- **panic=abort parity**(Q1=B): plugin panic = 정적 stage panic = 프로세스 abort(동일). catch_unwind 격리는 untrusted `.so` 요구 시 defer(Cargo 가 멤버별 panic override 불가).
- **PerHead 미지원**: `keep_kind==1` → `planabi_to_plan` 이 host bail(v1, promotion-trigger 전).
- **engine/Cargo.toml drift 유지**: `microbench_score_readback` `[[bin]]`(untracked `microbench/score_readback.rs`) 미커밋. C2 의 libloading 변경은 stash 격리 커밋으로 분리(drift 보존). 커밋 금지.

---

## 참조

- SSOT: `docs/adr/0009-gate-c-stage-dlopen-plugin.md`(D1–D7 = Q1–Q7, Status=구현완주). ADR-0007 D6(GATE-C device-gated 정정 — Stage 축은 host-implementable).
- 코드 앵커: `crates/technique-api/src/lib.rs`(ABI 타입/AbiStageCtx/PlanArena/register_kv_stage! 매크로, GATE-C 섹션) / `engine/src/pressure/eviction/stage_registry.rs`(DYN_REGISTRY/register_dynamic_stages/make_stage/DynStage/shim, GATE-C 섹션) / `engine/src/session/{cli.rs(--load-plugin), bin_setup.rs(register 배선), assembly/build_bench_loop.rs:~101, chat/session.rs:~636}`(make_stage 호출) / `crates/techniques/example-keep-recent/`(cdylib vehicle) / `engine/tests/gate_c_dlopen_equivalence.rs`(C5 게이트).
- 빌드: `.so` = `cargo build --release -p example-keep-recent --features plugin-cdylib` → `target/release/libexample_keep_recent.so`. smoke = `argus_cli --load-plugin <.so> --model-path <gguf> -b cpu --prompt ... -n 8`.
- 선행 handoff: `handoff_opaque_kv_e2e_eviction_wiring_2026_06_09.md`(ADR-0008 e2e — GATE-C 전 단계).
