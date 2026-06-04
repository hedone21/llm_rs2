# Roadmap: Phase α-K BC 완주 — `KVCacheOps` trait 완전 폐기((4))

**작성**: 2026-06-04 (작성자: PM) — 사용자 "BC 완주" 결정(2026-06-04)의 정식 TODO화.
**SSOT**: `arch/pipeline_stage_design_v2.md` §9.1 "⚠️ α-K (3p)/(4) 방향" 블록(line 758~761, 2026-06-04 BC 확정) + §9.1-EVICT(-DECISION).
**ADR**: `docs/adr/0001-kv-dispatch-paradigm.md` §8.3 (line 216~218, ④-a viability cut + 5 cluster census).
**진입 handoff**: `.agent/todos/handoff_alpha_k_3d_entry_2026_06_03.md` (직전 (3c-evict) 완료 체크포인트).
**트랙 메모리**: [[project-pipeline-alpha-k]].

> PM 비-수정 영역: 본 roadmap 은 `.agent/todos/`만 수정한다. 코드·SSOT·ADR 는 미수정. 각 step 의 실제 구현/검증은 Architect(설계)·Implementer/Senior Impl(구현)·Tester(device 게이트)가 분담 — Step별 "권장 역할" 참조.

---

## 목표·원칙 (전역)

**최종 목표**: `KVCacheOps` trait 완전 삭제((4)). 현재 `KVCacheOps`(generic monomorphization, production hot=plan path) ∥ `KVCacheFormat`(trait object, forward-fallback + eviction host 증명) 가 공존(branch-by-abstraction). production 은 휴면 상태(`LLMRS_KV_FMT` OFF + eviction unwired). BC 완주 = production 을 `KVCacheFormat` 패러다임으로 통일 후 `KVCacheOps` 삭제.

**확정 사실(소스 검증, ADR §8.3 census)**:
- (4) 컴파일 차단자 = **5 cluster**. B-1 plan `execute<C>`(=(3p)) / B-2 forward chain full-surface(prefill 포함) / B-3 offload(`PrefetchableCache: KVCacheOps` + `preload_erased` fn-ptr) / B-4 eval(`run_eval_ll_generic<C>` + `StepHook<C>`) / B-5 legacy(✅사용자 결정으로 disposable=해소).
- (3p) = **④-a concrete-handle**(`Arc<StandardFormat>`/`Arc<KIVIFormat>`, static dispatch, **vtable 0**). dyn trait object flip 아님. production plan path 는 이미 generic monomorphization(vtable 0 + lock 없음 = perf-optimal)이라 (3p)/④-a 는 **perf neutral-or-slightly-worse**(getter/advance 의 layer당 Mutex lock 추가 — cleanup 목적, gain 아님).
- (3c-fwd) ✅ `c2b05aff` (forward_into_fmt, S25 device bit-identical) / (3c-evict) ✅ `2f014163` (`EvictionPolicy::plan_keep` + compact host 등가 9/9). 둘 다 순수 additive·unwired(production 무변).

**전역 원칙**:
1. **branch-by-abstraction** — 매 substep 이 컴파일+실행되는 엔진을 남긴다(`KVCacheOps` ∥ `KVCacheFormat` 공존 전제).
2. **legacy = reference baseline** — migration 내내 `legacy_generate` 출력을 reference 로 유지. 각 cluster flip 은 legacy 출력과 **bit-identical** 비교. legacy 폐기는 **마지막**(Step 5).
3. **production World B 무회귀** — Step 3((3p) hot-path flip) **전까지** hot path(plan path) 무변. Step 1·2 는 cold-path(prefill/eval/offload)만 접촉.
4. **외과적** — 변경 라인은 해당 cluster flip 으로 직접 추적 가능. 무관 리팩토링 금지.
5. **회귀 시 격리 revert** — Step 3 회귀 시 (3p)만 revert, Step 1·2 cold cluster 정리는 유지(ADR §6.5 substep 단위 조기 후퇴).

**전역 device 게이트 정의** (Step별 "검증 게이트"에서 참조):
- **device-gate(full)** = 5 KV 구성(Sliding/H2O/D2O/KIVI/SnapKV) × 32-tok token-id 완전 일치(bit-identical) + avg_tbt Δ≤+3% (n≥5 median, tok0 inclusive — `feedback_tbt_metric_tok0_inclusive`), S25 OpenCL(`opencl --opencl-rpcmem`) + Jetson CUDA.
- **baseline 동결**: BC 진입 commit 에서 1회 캡처(ADR §6.1/6.2) → 매 device 게이트가 frozen 기준과 비교. **legacy 출력 = reference baseline**.
- **bit-identical 비교 명령(예시, S25)**: device `/data/local/tmp/models/qwen2.5-1.5b/{f16,q4_0}.gguf` 사용.
  ```
  # reference (legacy):  ./legacy_generate -m <gguf> -b opencl --opencl-rpcmem --greedy -n 32 -p "..."
  # flip 경로:           LLMRS_KV_FMT=1 ./argus_cli  -m <gguf> -b opencl --opencl-rpcmem --greedy -n 32 -p "..."
  #   (Step 4 이전엔 device-gate bin = legacy_generate; Step 4 후 argus_cli)
  ```

---

## [P1] Step 1 — B-2/B-4 cold-path flip (forward_into prefill + eval)
- **Status**: IN PROGRESS (①-a + C3 + **①-b ✅ `2bf5c500`** + **①-c ✅ host `1e4f20fe`** + **①-d ✅ host `84bed97e`**(10 site); **①-e 다음** = run_kivi_ppl(KIVIFormat prefill arm 신설 후))
- **Sprint**: current
- **Dependencies**: (3c-fwd) ✅ `c2b05aff` + (3c-evict) ✅ `2f014163` (선결 완료). Architect 설계 라운드 권장(B-2 full-surface fork 형태).
- **차단 cluster**: B-2 (forward chain full-surface) + B-4 (eval 다형성). **hot path 미접촉**(prefill/eval = cold tier).
- **Description**: `KVCacheOps` 폐기 차단자 중 production hot 이 아닌 두 cluster 를 먼저 flip. (a) **B-2**: `forward_into<C: KVCacheOps>`(transformer.rs:1491) + 레이어 chain(`forward_gen<C>` forward_gen.rs:24 / `forward_prefill<C>` forward.rs:41 / `update_kv_cache<C>` transformer_layer.rs:33 + Args struct) 의 **prefill + 비-decode forward** 경로를 `&[Arc<dyn KVCacheFormat>]` 기반으로 전환. decode-only fmt fork(`forward_into_fmt`/`forward_gen_fmt`)는 이미 (3c-fwd)에 존재 → prefill·warmup·qcf_runtime·batch 호출처를 fmt entry 로 단일화. (b) **B-4**: `run_eval_ll_generic<C>`(eval_loop.rs:45) + `StepHook<C>`(eviction_hook.rs:217 `StepHook<KVCache>` / kivi_hook.rs:137 `StepHook<KiviCache>`) + `CacheSnapshot<C>` 의 KVCache/KiviCache **런타임 다형성**을 trait object 또는 enum 으로 전환.
- **권장 역할**: Architect (B-2 full-surface fork 설계 — base-trait creep 금지, `INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC`) → Implementer (구현 + host test) → Tester (device-cold 게이트).
- **Acceptance Criteria**:
  - host: `cargo build` + `cargo test --workspace` (옛 경로 byte-불변, fmt+clippy `--workspace -D warnings` clean).
  - device-cold: prefill+eval 경로가 legacy 출력과 **bit-identical** (S25, `--no-gpu-plan` 불필요 — prefill 은 plan 무관; eval 은 ppl/nll 수치 일치). 5 KV 구성 중 Sliding/H2O/NoEviction 최소 커버.
  - hot path(plan decode TBT) **무변 확인** — avg_tbt Δ≈0 (이 step 은 hot 미접촉, surprise tripwire).
- **Notes**: B-2 가 full-surface 라 ④-a(getter+advance만 닿는 plan) 로 못 벗긴다(ADR §8.3 B-2). prefill flip 선결이 Step 3((3p)) 의 일부 부담을 미리 던다. **sub-item (2026-06-04 정정 — eval-first 역전: eval 의 `C` = forward_into cache 타입이라 forward_into flip 이 선결, SSOT §9.1-BC1' ★반증 1~3)**:
  - **①-a (3a)/(3b) trait-gap + write_kv/attention_into GPU fast-path**: ✅ **phantom/완료** — `write_kv`(standard_format.rs:120 GPU scatter)·`attention_into`(:235) 이미 wired((3c-fwd) device PASS, 3a/3b ✅ `5ea8ad47`/`3bc03e59`). 추가 작업 없음.
  - **C3 = `write_kv_batch` GPU prefill batch scatter**: ✅ **host `2e6b50fb`** + **device bit-identical ①-b 게이트에서 실증**(F16 GPU batch scatter / Q4_0 cast / F32 GPU batch scatter 모두 PASS).
  - **①-b = B-2 prefill batch entry (C1)**: ✅ **완료 `2bf5c500`** — `forward_into_fmt` multi-token prefill dispatch + `forward_prefill_fmt`(신규) + `StandardFormat::attention_into` prefill arm(`prefill_attention`) + `TransformerModelForwardFmtArgs.logits_last_only` + `ModelForward::prefill` 게이트 배선. 설계+적대검증 = `design_alpha_k_1b_cut_2026_06_04.md`(workflow wfceex20u). **host**: build + standard_format 13/13 + non-opencl 회귀 0 + fmt + clippy clean. **S25 device 게이트 PASS**: F16(rpcmem GPU flash)/Q4_0(rpcmem CPU dequant)/F32(device-only GPU flash) **3 dtype × `--no-gpu-plan` bit-identical**(텍스트+first token 49689+count 31 일치) + avg_tbt Δ∈[−1.9%, +0.6%]·TTFT Δ∈[−3.1%, +1.4%] (모두 ≤+3%). hot path(gate OFF) 무변. **Jetson CUDA prefill = optional follow-on**(같은 Backend trait `flash_attention_prefill`, S25 가 GPU-flash+CPU-fallback+e2e 커버). NoEviction(happy-path) — Sliding/H2O eviction-during-prefill 은 ①-b 범위 밖(3c-evict).
  - **①-c ✅ = B-4 eval flip** (`1e4f20fe`): **Strategy A (transient per-call fmt-wrap)** — eval 이 concrete `Vec<KVCache>`/`Vec<KiviCache>` 를 계속 소유, forward 1회만 `EvalCacheKind::forward_fmt_roundtrip`(fmt_bridge.rs 신규) 로 wrap→`forward_into_fmt`→into_inner 복귀. hook/snapshot/`force_evict` 무수정(forward↔hook 시퀀셜). `forward_into_fmt` 에 score_accumulator/skip_config/importance_collector/cache_self_need_scores 4필드 additive 확장(forward_into 미러). `StepHook<C>`/`CacheSnapshot<C>` 바운드 0. **scope 정정: SSOT "thin" 은 census 빈틈 — 실제는 中규모(~290 LOC), forward_into_fmt feature parity 확장 필요**(eval 이 ModelForward 안 거치고 forward_into 직접 호출 7곳). **host 게이트**: legacy `--eval-ll` BEFORE vs AFTER — **KVCache F16 basic+H2O bit-identical**, **KIVI flush 정수회계(flush_count/q2_tokens/res_pos/predicted) 완전일치**(nll Δ~1e-6~2e-4 = ★2 KIVI get_view=F32 carve-out, device-only bit-identical). KVCacheOps trait 생존(Step 5 폐기) — `EvalCacheKind for KiviCache` 의 `use KVCacheOps` 1개 잔여(Step 5 정리). 설계+적대검증 = `design_alpha_k_1c_cut_2026_06_04.md`(workflow `wdrcgtqwz`, 5 lens).
  - **①-d ✅ = B-2 비-decode 잔여 10 site** (`84bed97e`): warmup·qcf_runtime(212/300)·batch(383/452/742/817)·run_ppl(771/935)·dump_importance 의 forward_into → forward_into_fmt(EvalCacheKind round-trip 재사용). **forward_into_fmt decode 분기에 workspace=None fallthrough 추가**(발산 A — 구 layer.forward→forward_prefill fall-through 를 forward_prefill_fmt 로 미러, bit-identical, production 미발화 additive). **slice→Vec 시그니처 3곳**(run_warmup/run_ppl/QcfWarmupCtx, caller 0 변경; clone unsound 기각). **host 게이트**: build+clippy+fmt clean·lib 1241 pass(13 fail 전부 opencl GPU 부재)·비-opencl 회귀 0·batch/ppl(KVCache NLL=173.1049)/dump_importance bit-identical + warmup seq_len=1 fallthrough 런타임 PASS. 설계+적대검증 = `design_alpha_k_1d_cut_2026_06_04.md`(workflow `w12qx2ybg`).
  - **①-e ★다음 = run_kivi_ppl(KIVI 2 site) fmt 전환** — ①-d 게이트에서 발견: **KIVIFormat::attention_into 는 multi-token prefill arm 부재**(attention_gen=single-query decode 전용, `kivi_format.rs:95-173`)라 forward_prefill_fmt 가 KIVI multi-token prefill 시 panic(eval KIVI 는 token-by-token prefill 이라 안 걸림). **KIVIFormat multi-token prefill arm 신설**(KiviCache::get_view compact view aware `prefill_attention` + bits 2/4/8 CPU SeqMajor + bits16 GPU HeadMajor + native 경로, device 검증) 선결 후 run_kivi_ppl(ppl:367 prefill + 448 decode) fmt 전환 + AWQE cache_self_need_scores 주입(①-c 미러). prefill.rs(run_chunked_prefill, profiler+variance)도 ①-e/Step5 후보.

---

## [P1] Step 2 — B-3 offload 분리 (`PrefetchableCache` KVCacheOps 비의존 재정의)
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Step 1 (forward chain flip — `forward_into_offload<C>` 가 B-2 forward chain 에 의존). 독립 분리 가능하면 Step 1 과 병렬 평가.
- **차단 cluster**: B-3 (offload). **production hot 아님** (opt-in `--kv-offload`, plan path 미사용).
- **Description**: `PrefetchableCache: KVCacheOps` supertrait(kv_cache.rs:16) 의 supertrait bound 를 제거하여 `PrefetchableCache` 를 `KVCacheOps` 비의존으로 **재정의**. + `forward_into_offload<C>`(transformer.rs:2906) + `preload_erased<C>` fn-ptr(preload_pool.rs:177) 의 generic 소비를 trait object/비-generic 으로 전환. 권장 처리(ADR §8.3 B-3) = offload 를 (4) 트랙에서 **분리**(KVCacheOps 비의존화)하여 KVCacheOps 폐기의 컴파일 차단을 해소.
- **권장 역할**: Architect (`PrefetchableCache` 재정의 형태 — KVCacheOps 의 어느 method 가 offload 에 실제 필요한지 census, supertrait → 독립 trait 또는 KVCacheFormat 위임) → Implementer (구현 + host test) → Tester (`--kv-offload` K-sweep crash-free + sane).
- **Acceptance Criteria**:
  - host: build + test, `PrefetchableCache` 가 `KVCacheOps` 를 supertrait 로 더 이상 요구하지 않음(grep 확인).
  - device: `--kv-offload` 경로 crash-free + sane output (offload 는 production hot 아님 → bit-identical 보다 정확성+안정성 우선). 가능하면 legacy offload 출력과 bit-identical.
  - hot path 무변 확인.
- **Notes**: offload 는 `--kv-offload` opt-in 이라 device 게이트는 sanity 중심. fn-ptr(`preload_erased`)의 generic erasure 방식이 설계 포인트.

---

## [P1] Step 3 — (3p) ④-a hot-path flip (plan path concrete-handle)
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Step 1 (B-2 prefill flip — plan 평가가 forward chain 정합 필요) + Step 2 권장 선행. **(3d) prefill flip + plan 평가 = (3p) 분기 결정** (handoff_alpha_k_3d_entry 의 미해결 — 이 step 의 진입 전 확정 필요).
- **차단 cluster**: B-1 (plan `execute<C: KVCacheOps>` plan.rs:1257). **production hot layer-tier crux** — BC 완주의 유일한 perf 위험 지점.
- **Description**: `plan.rs::execute<C: KVCacheOps>`(:1257) 를 **④-a concrete-handle**(`Arc<StandardFormat>`/`Arc<KIVIFormat>`, static dispatch, vtable 0)로 flip. C 가 닿는 표면 = 6 스칼라 getter(capacity :1286/1293 / current_pos :1286/1290/1291/1851 / res_pos :1294 / q2_tokens :1295) + `advance_pos` 1회(:1828)뿐(K/V 버퍼 데이터 접근 0, attention 은 `AttentionVariant` enum static — `attention_into` 호출 0건). dyn trait object 가 **아니다**(ADR §8.3 정정 1). perf 측정 대상 = **vtable 아님 = ④-a getter/advance 의 layer당 Mutex lock**(`StandardFormat` 의 `inner.lock().unwrap()` 패턴).
- **권장 역할**: Architect (④-a concrete-handle 도입 형태 + `AttentionVariant` 평탄화(④-b) 묶음 여부 friction-triggered 판단) → Senior Implementer (plan.rs 핫패스 flip — GPU/lock 비용 민감) → Tester (device-gate full + avg_tbt 실측).
- **Acceptance Criteria** (★perf crux):
  - **device-gate(full)**: 5 KV 구성 × 32-tok **bit-identical** + **avg_tbt Δ≤+3%** (S25 OpenCL + Jetson CUDA, n≥5 median tok0-inclusive). frozen baseline 대비.
  - lock-cost 실측: avg_tbt 회귀가 +3% 초과 시 root-cause = `StandardFormat` getter Mutex lock (vtable 아님 — ④-a vtable 0). 격리 microbench 폐기(측정 대상 부재 — ADR §8.3 정정 1).
- **Notes**: **회귀 시 (3p)만 revert, Step 1·2 cold cluster 정리는 유지**(전역 원칙 5, SSOT line 761). perf 방향 = neutral-or-slightly-worse(현 plan path 가 이미 vtable 0 + lock 없음 = perf-optimal — ④-a 는 lock 추가). ADR §6.5 [갱신 주 2026-06-04]: (3p) 6.2 게이트 fail(Δ>+3%) **AND** lock-cost 실측 확인이 정정된 perf revoke trigger. **선결 = (3d) plan 평가** — handoff_alpha_k_3d_entry §"다음 작업 (3d)" 3번(plan 평가 = (3p) 분기 결정)을 이 step 진입 전 확정.

---

## [P2] Step 4 — device-gate 를 legacy_generate → argus_cli 로 이주
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: Step 1·2·3 (flip 들이 argus_cli 경로에서 동작해야 device-gate 이주 가능). **legacy 폐기(Step 5) 의 선결**.
- **차단 cluster**: B-5 부수효과 (legacy_generate 가 현 device-gate bin — MEMORY [[project_pipeline_alpha_w]]). legacy 폐기 전 device 게이트 매체 이전 필수.
- **Description**: 현 device 게이트 bin = `legacy_generate`(`engine/legacy/generate.rs`). legacy 를 폐기(Step 5)하려면 device bit-identical/avg_tbt 게이트를 **`argus_cli`**(`engine/src/bin/argus_cli.rs`)로 먼저 이주해야 한다. argus_cli 가 5 KV 구성 × eviction subcommand × `--opencl-rpcmem` × greedy 32-tok 측정을 legacy 와 동등하게 수행하는지 확인 + 갭 메움.
- **권장 역할**: Architect (argus_cli 가 device-gate 요구사항 충족하는지 gap census) → Implementer (argus_cli CLI 갭 메움) → Tester (argus_cli 로 device-gate full 재현 — legacy 출력과 bit-identical 확인).
- **Acceptance Criteria**:
  - `argus_cli` 가 device-gate(full) 5 KV 구성 × 32-tok 측정을 legacy 와 동등 수행.
  - argus_cli 출력 == legacy_generate 출력 bit-identical (이주 전 등가 증명 — reference baseline 연속성).
  - device-gate 절차 문서(handoff/SSOT) 가 argus_cli 명령으로 갱신(Tester→Tech Writer 협업).
- **Notes**: argus_cli 는 신규 single-prompt bin(`7065196c` 분할). legacy 는 동결 monolith. device 게이트 이주가 legacy 폐기의 hard 선결 — 이주 없이 legacy 삭제 시 device 검증 매체 소실.

---

## [P2] Step 5 — legacy 폐기 + `KVCacheOps` trait 삭제
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: Step 1·2·3·4 전부. **컴파일 차단자 0 확인이 진입 조건**(B-1~B-4 flip 완료 + B-5 legacy 이주).
- **차단 cluster**: 없음 (deletion). B-5 legacy(`legacy/generate.rs`)가 `forward_into`/`execute_plan`/`execute_plan_for_kivi`/`forward_into_offload`/`run_eval_ll_generic` + `use KVCacheOps`(:4044/4645) 전부 호출 → legacy 폐기로 마지막 소비자 제거.
- **Description**: (a) `legacy_generate` bin + `engine/legacy/generate.rs` 폐기(사용자 결정: legacy disposable, 호환성 불필요). (b) `KVCacheOps` trait 삭제 — `kv_cache_ops.rs:53` 의 generic monomorphization 정책 주석 제거(ADR-0001 §8 item 1). 컴파일 차단자 0(B-1~B-4 flip 완료) 확인 후 진행.
- **권장 역할**: Architect (KVCacheOps 잔존 소비자 0 최종 census — `grep KVCacheOps` 전수) → Senior Implementer (trait 삭제 + 잔여 정리) → Tester (device-gate full 최종 — *진짜 최종 perf*, parallel path 제거 후 monomorphization 드러남).
- **Acceptance Criteria**:
  - `grep -r "KVCacheOps" engine/` 0건 (trait + 모든 generic bound + use 삭제).
  - host: build + test (fmt+clippy `--workspace -D warnings` clean).
  - **device-gate(full) 최종**: 5 KV 구성 × 32-tok bit-identical + avg_tbt Δ≤+3% (S25 + Jetson). **(4) avg_tbt = 진짜 최종 perf** — parallel path(KVCacheOps∥KVCacheFormat) 제거 후라야 실제 monomorphization 이 드러남(SSOT line 735, table line 731).
  - `kv_cache_ops.rs` 파일 자체 삭제 또는 `KVCacheFormat`-only 로 정리(`KVCacheFormat` 으로 rename 동행 — ADR-0001 §title 명칭 정리).
- **Notes**: 트레이트 명 `KVCacheOps` → `KVCacheFormat` 동행 rename(ADR line 10). INV ID(`INV-KVCACHELAYER-*` / `INV-STAGE-LAYER-HANDLE`)는 추적용 안정 키 유지. 삭제 commit 후 ADR-0001 §6 종료 게이트 충족 확인 + 트랙 메모리 [[project-pipeline-alpha-k]] 종결 배너.

---

## 의존 그래프 요약

```
(3c-fwd ✅) ─┐
(3c-evict ✅)─┴→ Step 1 (B-2 prefill + B-4 eval, cold) ─┬→ Step 3 ((3p) B-1 plan, HOT crux)
                Step 2 (B-3 offload, cold) ─────────────┘        │
                                                                 ↓
                                          Step 4 (device-gate → argus_cli)
                                                                 ↓
                                          Step 5 (legacy 폐기 + KVCacheOps 삭제)
```

- **위험 순서**: Step 1·2 (cold, hot 무접촉) → Step 3 (hot crux, 회귀 시 (3p)만 revert) → Step 4·5 (legacy 폐기).
- **revoke 연동**(ADR §6.5): Step 3 avg_tbt Δ>+3% **AND** lock-cost 실측 확인 = perf revoke trigger. Step 1·2 회귀는 cold-path 한정(revoke 강도 낮음). bit-identical 회귀는 cold/hot 무관 6.1 발동.
- **handoff cross-ref**: 직전 (3c-evict) = `handoff_alpha_k_3d_entry_2026_06_03.md` / 배경 census = `handoff_alpha_k_substep3_census_2026_06_03.md`. 각 step 완료 시 handoff 작성(`handoff-doc` 스킬 R1~R6).
