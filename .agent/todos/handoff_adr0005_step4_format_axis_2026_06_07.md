# Handoff: ADR-0005 step4 (format 축 spine) — S4-1·S4-2 완주 → S4-3 (compute→backend) 설계부터

**작성**: 2026-06-07
**HEAD**: `8a02b834 refactor(format): ADR-0005 S4-2 — KVCacheFormat::compact 폐기, execute_kv_plan 단일 정본`
**브랜치**: master (미푸시)
**작성자**: 메인 세션

**다음 세션 진입 문장**: **"ADR-0005 step4 S4-3 — compute→backend descriptor dispatch 설계부터 (KIVI=escape, forward_gen_fmt 메커니즘 변경, device L2 게이트). 그 후 점진 trait 정리 → 최종 KVCacheFormat 삭제."**

---

## TL;DR

3-axis plugin 리팩토링의 **format 축**을 북극성(zero-compile .so format plugin) 직접 전진으로 진행. `KVCacheFormat 해체`(step4) 중 **host-additive/cold 증분 2개 완주** — S4-1(KV_FORMATS 내장 등록) + S4-2(compact 폐기→execute_kv_plan 정본). **멈춘 이유**: 남은 S4-3(compute→backend descriptor dispatch)는 device-gated(L2=S25 TBT) + 설계 필요(KIVI escape, forward 메커니즘 변경, System A 결합)라 phase 경계. S4-1/S4-2 와 성격이 다름(작은 host 증분 → 큰 설계+device 라운드).

이번 세션의 **대형 발견**(아래 §Landmines 필독): α-K BC 완주 확인, Offload=2 시스템, taxonomy 확정, closed-enum=anti-.so 모순, 인터페이스 3결정.

---

## 진행 상태 (이번 세션, 2 커밋 — 명시 파일만, 무회귀)

| 증분 | 커밋 | 내용 | 게이트(독립 재검증) |
|---|---|---|---|
| **S4-1** | `c414a38e` | `engine/src/format/builtin_kv_formats.rs` 신설 — `StandardKvFormat{name,desc}` f32/f16/q4_0/q8_0 를 `#[distributed_slice(KV_FORMATS)]` 등록(descriptor=`dtype_to_layout_desc` 단일원천) + `ensure_builtin_kv_formats_registered()` self-test. **순수 additive·unwired** | `format::builtin_kv_formats` 1/1(round-trip+4종) + clippy clean |
| **S4-2** | `8a02b834` | `KVCacheFormat::compact` trait 메서드 + `StandardFormat::compact`/`apply_merges`(균등) + KIVI/Offload compact + 단위테스트 폐기 → trait **7→6 method**. production 정본=`execute_kv_plan`. compact_parity Path B 를 execute_kv_plan 으로 retarget. ADR-0005 D3 + technique-api NOTE 갱신 | **compact_parity 9/9 bit-identical**(4정책×3dtype) + lib 1249/0(RSS flaky 격리) + clippy clean |

전체 게이트: `cargo test -p llm_rs2 --lib`(skip opencl/memory) **1249 passed 0 failed**, `cargo clippy --workspace -- -D warnings` 0, fmt clean(drift 2파일 미변경).

---

## 다음 작업 (S4-3+, 전부 device/설계 의존)

1. **S4-3 = compute→backend descriptor dispatch** (cold→device). `write_kv`/`attention_into` 를 descriptor-구동 backend dispatch 로. **설계 패스 선행 필수** — 미결: (a) KIVI(Q2+residual)는 block-quant descriptor 어휘 밖 → escape(특화 kernel 유지), (b) `forward_gen_fmt`/`forward_prefill_fmt` 가 `&Arc<dyn KVCacheFormat>` 로 compute 디스패치하는 메커니즘 변경, (c) buffer holder(StandardFormat) vs descriptor plugin 분리 완성(bridge: StandardFormat 에 `format_name` 필드). 게이트 = **L2 device**(plan-path bit-identical + S25 TBT Δ≤+3%).
2. **점진 trait 정리 → 최종 KVCacheFormat 삭제** — S4-3 후 format 멤버가 {Standard dtypes, KIVI} descriptor+backend 로 수렴하면 trait 축소·삭제. **단 System A(OffloadFormat)가 trait impl 이라 최종 삭제 전 처리 필요**(§Landmines).
3. **(곁가지, defer) System A 처리** — OffloadFormat port-or-delete. **(곁가지, defer) System B tier_move stage 정리**(arch §4.1, trait 삭제와 직교).

---

## Landmines / 핵심 발견 (다음 세션 필독 — 재발견 방지)

- **α-K BC 완주됨** — `KVCacheOps` trait 삭제(`102f0461`), legacy generate 폐기(`d5ed71d2`), production 은 engine `KVCacheFormat`(6-method now)으로 통일. 종결 handoff `e23518e3`. (memory `project_pipeline_alpha_k` 는 stale — Step 2/3 까지만 기록, 실제 5-F 완주.)
- **Offload = 독립 2 시스템** (offload map `wf_bb953e9b`): **System A**=`OffloadKVCache`+`OffloadFormat`(KV 대체 storage, `OffloadFormat`=KVCacheFormat impl=**trait 삭제 blocker**) vs **System B**=`SwapHandler`(일반 KVCache prefix→disk swap, **이미 CachePressureHandler**, `offload_one`/`recall_one` 보유=arch §4.1 tier_move 원천). **둘은 별개.**
- **System A 는 production 도달 불가** — `argus_cli.rs:79`/`argus_bench.rs:104` 가 offload 모드 `bail!`("planned for v1"), `build_chat_offload` 호출자 0, legacy bin 삭제됨. `LLMRS_OFFLOAD_FMT` env 게이트 **dead**(0 read, stale 주석). offload mode 자체가 unreachable v1-pending.
- **taxonomy 확정**: KIVI=**format**(CONTEXT.md:13/94 명시, Q2+residual=format+동적quant stage 합성), Standard dtypes=format, **Offload≠format**(precision-직교 `offload.rs:47`, arch §4.1 tier_move=stage). ⚠ **CONTEXT.md 에 Offload taxonomy 미기록**(glossary 공백 — System A/B 구분 + tier_move 기록 권고).
- **closed-enum=anti-.so 모순**: "KVCacheFormat 완전 삭제"를 enum `FmtCache{Standard,Kivi,Offload}` 로 하면 닫힌 집합→.so format 추가 불가(북극성 위배). 해법=format 축은 descriptor-plugin 열린 집합 유지, compute=backend dispatch+floor. **enum 으로 닫지 말 것.**
- **인터페이스 3결정**(설계 워크플로우 `wf_448db4f2` 종합, 코드로 자명): (1) bridge=별도 thin plugin(StandardKvFormat) — StandardFormat 직접 impl 안 함, `KVFormatReg.make`가 zero-arg 라 buffer 주입 불가 / (2) compute=레이어가 버퍼+descriptor 로 **기존** backend 저수준(kv_scatter/attention_gen/flash) dispatch, backend 신규 메서드 0(M+N+K) / (3) compact=plan-returning(execute_kv_plan), api KVFormat 에 미추가.
- **engine `Merge` 보존**(삭제 금지) — `EvictionPolicy::plan_keep -> Option<(Vec<usize>, Vec<Merge>)>`(sliding/h2o/streaming/no_eviction)가 사용(항상 빈). 전면 Merge→WeightedMerge dedup 은 EvictionPolicy 5파일 번지는 별도 정리.
- **RSS flaky**: `pressure::kv_cache::tests::test_*release_unused_pages*` 류 — full lib 집계 시 간헐 1-fail, 단독·재실행 PASS = 환경성(회귀 아님). lib 게이트는 1249 passed 0 failed 가 clean.
- **미커밋 보존**: `engine/Cargo.toml`(microbench_score_readback bin, M) + `engine/microbench/score_readback.rs`(??) — pre-existing, 건드리지/커밋하지 말 것. fmt drift 2파일(`htp_fastrpc.rs`/`transformer.rs`) — `cargo fmt --all` 이 건드리면 checkout.

---

## 참조

- SSOT: `docs/adr/0005-...md`(D3 S4-2 갱신 NOTE 포함 / D5 floor / D6 registry / L1·L2), `docs/adr/0004-...md`(execute_kv_plan plan-returning), `CONTEXT.md`(3축 — Offload taxonomy 미기록), `arch/pipeline_stage_design_v2.md` §4.1(tier_move)/§8(L2 게이트).
- 코드 앵커: `engine/src/format/builtin_kv_formats.rs`(S4-1) / `crates/technique-api/src/lib.rs:472`(KVFormat 2-method) / `engine/src/format/kv_cache_format.rs`(KVCacheFormat 6-method) / `engine/src/pressure/eviction/stage_registry.rs:88`(execute_kv_plan) / `engine/src/format/dtype_layout.rs`(descriptor floor).
- 워크플로우 산물: understand `wf_297467b9` / 설계 `wf_448db4f2`(3안→심사→종합) / offload map `wf_bb953e9b`.
- 선행 handoff: `handoff_format_backend_plugin_unification_2026_06_06.md`(M-F1~3), `handoff_3axis_plugin_refactor_host_complete_2026_06_07.md`.
