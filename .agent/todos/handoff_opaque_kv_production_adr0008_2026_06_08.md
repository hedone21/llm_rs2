# Handoff: ADR-0008 opaque KV production 통합 — full scope(Stage 1+2+3) 완주

**작성**: 2026-06-08
**HEAD**: `923f1868 docs(adr): ADR-0008 — D2 floor 패턴 정정 + Stage 1+2+3 완주 상태 + e2e eviction bin 한계 기록`
**브랜치**: master (미푸시)
**작성자**: 메인 세션

**다음 세션 진입 문장**: **"ADR-0008 full scope 완주 — 다음은 GATE-C(.so cdylib dlopen 승격, 북극성 최종) 착수 vs 잔여 deferred(e2e eviction bin 배선 / write encoder family / GPU opaque arm) 우선순위 grill 후 착수."**

---

## TL;DR

ADR-0007 GATE-B(격리 opaque format 호스트 증명) 후속으로, 사용자가 "북극성 우선순위 grill"에서 **production KVCache 통합 먼저**(GATE-C보다)를 선택 → opaque(.so block-quant) format 이 **실제 추론 전 경로**(alloc→grow→attention→eviction→D2O merge)를 `DType` variant 없이 q4_0 bit-identical 로 타게 만들었다. **ADR-0008 full scope(Stage 1+2+3) 완주**: 4 커밋, 전 단계 bit-identical 게이트 + Stage 1 e2e token-identical. **멈춘 이유**: full scope 가 이번 작업의 종착(사용자 확정 범위). 다음(GATE-C 등)은 새 결정.

---

## 진행 상태 (이번 세션, 5 커밋 — 명시 파일만, 무회귀)

| | 커밋 | 내용 | 게이트 |
|---|---|---|---|
| ADR | `4e20aa64` | ADR-0008 설계 SSOT (D1~D6, grill 4결정) | 문서 |
| S1 | `6b4eaef5` | KVCache opaque 흡수 + descriptor-keyed alloc/grow + `--kv-format` + attention floor + synth-q4 force-link + OpaqueKvFormat 삭제 | opaque KVCache(grow 포함) == q4_0 round-trip **bit-identical** + **`--kv-format synth_q4` == `q4_0` token-identical**(Qwen2.5-1.5B CPU greedy n=24) |
| S2 | `d1b6cb55` | eviction primitive(prune/shift/shift_for_head/shrink_to_fit_opaque) descriptor-keyed | opaque prune_prefix(+shrink) attention == q4_0 bit-identical |
| S3 | `114b9b12` | `decode_via_descriptor` + D2O `apply_weighted_merges_opaque` | opaque weighted merge == q4_0 round-trip bit-identical (K/V) |
| 정정 | `923f1868` | ADR D2 floor 패턴 정정 + 완주 상태 + bin 한계 | 문서 |

전체 게이트: `cargo test -p llm_rs2 --lib`(skip opencl/memory) = **1258/0** (RSS flaky 환경성 재실행 PASS). `cargo clippy --workspace -- -D warnings` clean.

**핵심 설계(ADR-0008)**: 구조 (A) — `KVCache` 가 `OpaqueBuffer`(U8+desc) 1급 흡수(forward+eviction 둘 다 기존 경로 재사용). **이유**: eviction 이 concrete `&mut [KVCache]`(UER seam, D2O cross-layer)에 묶여 opaque 가 evict 되려면 KVCache 여야 함. byte-회계=**floor 패턴**(opaque arm 추가, typed arm 무변=byte-identical-by-construction). dispatch=format **이름→Option<DType>**(layout 아님). `OpaqueKvFormat` 삭제.

---

## 다음 작업 (전부 새 결정 필요 — grill 먼저)

1. **GATE-C = `.so` cdylib dlopen 승격** (북극성 최종, ADR-0007 D6). 런타임 registry(OnceLock 동적 + static linkme 합산) + `register_plugin` C-ABI dual-wiring + dlopen 배선 + dlopen 경로 bit-identical 재증명. **host-implementable**(직전 적대 검증 `wf_ebcad9c1`: handoff 의 "device-gated" 오분류 정정 — 코어는 호스트, GPU arm/perf 만 device). `panic=abort`↔`catch_unwind` 충돌이 최대 난점.
2. **e2e eviction bin 배선** — opaque eviction(prune/shift/merge)은 unit 게이트로 증명됐고 production 파이프라인은 dtype-agnostic 이나 CLI e2e 미검증: `argus_cli`(--kv-format 보유)는 happy-path 전용(eviction 미지원), `legacy_generate`(eviction 보유)는 --kv-format 미배선. 둘 중 하나 배선.
3. **write encoder family 확장** — `encode_via_descriptor`(+ opaque merge 재인코딩) 현재 PerBlockF16/Nibble(q4_0)만. q8_0(Byte)/q4_1(WithMin)/q2_0 확장.
4. **GPU opaque arm** — `opencl.rs:6098` attention_gen `_=>kernel_attn_gen`(F32 가정 silent garbage) / `:5577` matmul dtype-guard + readback. **device-gated**.

---

## Landmines / 미해결 / 핵심 발견

- **`engine/Cargo.toml` 커밋 = score_readback drift 동반 위험**: 사용자가 이번에 제약 해제(synth-q4-format dep 추가 위해). 미커밋 `microbench_score_readback` `[[bin]]`(untracked `microbench/score_readback.rs` 참조)은 **선택 스테이징으로 제외**하고 커밋했다(fresh checkout 빌드 보존). working tree 엔 drift 유지(`git status`에 ` M engine/Cargo.toml`). 향후 engine/Cargo.toml 커밋 시 동일 주의.
- **floor 패턴 ≠ 전면 통일**: byte-회계/copy/shift/merge 의 typed arm(q4_0/f16/f32)은 **1바이트도 안 건드렸다**. opaque arm 만 추가(descriptor-keyed, byte 단위). `copy_slice`/`buffer_shift` 의 U8 `type_size=1` 이라 opaque count 는 byte; Q4_0 은 block(18B), F16/F32 는 element. 단위 혼동 시 silent corruption.
- **opaque grow 는 desc 재포장 필수**: `memory.alloc_kv(_, U8)` 는 plain U8 버퍼 → `OpaqueBuffer::new(inner, desc)` 로 재포장 안 하면 sidecar 소실 → garbage. grow/shrink/alloc 모두 재포장.
- **shrink_to_fit_opaque 는 release_unused_pages → shrink 경유로도 호출**됨(opaque 의 madvise 는 `_=>return 0` skip 이라 안전). prune 후 underutilized 면 shrink 발동(테스트가 cap 128→64 로 검증).
- **eviction policy 직접 read 는 전부 테스트 코드** (sliding/h2o_plus/streaming 의 `as_slice::<f32>`) — production 정책은 외부 importance score + primitive 만 사용. 그래서 정책 코드 무변으로 opaque 동작.
- **D2O merge bit-identity 근거**: opaque merge 가 q4_0 merge 와 byte-identical 인 건 (a) decode floor == BlockQ4_0 dequant(G3), (b) encode floor == BlockQ4_0::quantize(G4), (c) 동일 weighted-sum 순서/associativity. 셋 중 하나라도 깨지면 불성립.
- **synth_q4 == q4_0 attention 은 CPU 에서 token-identical 실측**(Stage 1 e2e). 단 이는 CPU Q4_0 attention kernel == F32 floor 우연/성질이지 보장된 불변식 아님 — GPU/다른 kernel 에선 floor(opaque) vs native(typed)가 다를 수 있음(둘 다 correct). unit 게이트는 안전하게 **F32 round-trip reference** 기준.

---

## 참조

- SSOT: `docs/adr/0008-opaque-kv-production-integration.md`(D1 흡수 / D2 floor / D3 name→DType / D4 attention floor / D5 범위 full / D6 vehicle / §3 Risks / §4 Deferred). 선행=`docs/adr/0007-...md`(GATE-B), `0005`(D5 generic floor), `0004`(D1 compact 금지=구조 B 기각 근거).
- 코드 앵커: `engine/src/pressure/kv_cache.rs`(is_opaque/opaque_desc/opaque_bytes_per_head + grow/shrink_to_fit_opaque/update_opaque/prune/shift opaque arm) / `engine/src/pressure/standard_format.rs`(write_inner+attention_into opaque 분기, apply_weighted_merges_opaque, opaque 게이트 테스트 3종) / `engine/src/format/dtype_layout.rs`(decode_via_descriptor) / `engine/src/format/builtin_kv_formats.rs`(builtin_format_dtype + synth_q4 force-link) / `engine/src/session/bin_setup.rs`(--kv-format dispatch + alloc_opaque_kv_caches) / `engine/src/session/cli.rs`(--kv-format).
- 설계 워크플로우: 북극성 3-방향 매핑 + GATE-C host-testability 적대 검증 `wf_ebcad9c1`.
- 선행 handoff: `handoff_opaque_dtype_gate_b_2026_06_08.md`(GATE-B).
