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
- **Status**: ✅ **COMPLETE** (①-a + C3 + **①-b ✅ `2bf5c500`** + **①-c ✅ `1e4f20fe`** + **①-d ✅ `84bed97e`**(10 site) + **①-e ✅ `2941edca`**(run_kivi_ppl KIVI 2 site + KIVIFormat prefill arm)). B-2 forward chain(prefill 포함) + B-4 eval cold-path 전부 fmt 전환. **다음 = Step 2(B-3 offload)** 또는 Step 3((3p) hot crux). production hot(plan path) 무접촉 유지.
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
  - **①-e ✅ = run_kivi_ppl(KIVI 2 site) fmt 전환 + KIVIFormat prefill arm 신설** (`2941edca`): ①-d 게이트가 발견한 갭(KIVIFormat::attention_into multi-token prefill arm 부재) 해소. **(a)** `prefill_attention`(standard_format.rs free fn) private→`pub(crate)` 승격. **(b)** KIVIFormat::attention_into 에 `seq_len>1` prefill arm 신설 — KIVI 는 multi-token native 커널 부재라 dequant view(`get_view`) + `prefill_attention` 재사용(DRY). OLD `forward_prefill<C>` KIVI 경로와 bit-identical(`q_start_pos=cache_seq_len-seq_len`, CPU SeqMajor/GPU bits16 HeadMajor·bits2/4/8 assembled 모두 `kv_layout`/`kv_capacity` 인자 분기 → 별도 경로 불요). prefill score 무시(`_need_scores` 동일). +단위테스트(seq=4<res_cap Q2 flush 미발생 → residual F32 exact, causal mean 0..=r=r/2 bit-exact + panic 회귀 가드). **(c)** run_kivi_ppl(ppl/runner.rs:368 prefill + 456 decode) → `KiviCache::forward_fmt_roundtrip`+`forward_into_fmt`(①-c 미러), `cache_self_need_scores=needs_attn_scores()`(AWQE 미활성=false). decode 는 single-shot prefill 제어흐름상 unreachable(`prefill_len==eval_tokens`)이나 forward_into<C> 소비 제거 위해 전환. `TransformerModelForwardArgs` import 제거. **host 게이트**(qwen2.5-1.5b-q4_0, prefill_512.txt=456 tok, `--kv-mode kivi --ppl`, CPU): BEFORE(forward_into) vs AFTER(forward_into_fmt) **완전 bit-identical**(NLL=56.9390/total_nll 전정밀/flush_count=6/q2_tokens=384/res_pos=72/flush qcf_metrics 전부 일치), **panic 0**. ★2 carve-out 미발동(prefill=flash exact + single-shot=decode 미진입). build+fmt+clippy clean, 비-opencl·비-RSS 회귀 0. 설계+적대검증 = workflow `wej192c82`(3 verifier — V1 bit-identity CONFIRMED 라인근거, V2/V3 "refuted"는 "arm 미구현" 시제 아티팩트=작업 순서 확인). **잔여(Step5)**: prefill.rs(run_chunked_prefill, profiler+variance) forward_prefill<C> 소비 + KIVIFormat GPU prefill device 검증(host CPU-mode only).

---

## [P1] Step 2 — B-3 offload 분리 (`PrefetchableCache` KVCacheOps 비의존 재정의)
- **Status**: ✅ **COMPLETE** (`936d0c99`, Option A = supertrait bound 제거안). **다음 = Step 3((3p) ④-a hot crux)**. production hot(plan path) 무접촉 유지.
- **Sprint**: current → done
- **Dependencies**: Step 1 ✅ (forward chain flip). offload 는 `forward_into<C>` caller 가 아닌 **별개 forward** 라 실제로는 Step 1 과 독립했으나, B-2 OLD chain(`layer.forward<OffloadKVCache>`) 소비는 공유.
- **차단 cluster**: B-3 (offload). **production hot 아님** (opt-in `--kv-offload`, plan path 미사용).
- **Description**: `PrefetchableCache: KVCacheOps` supertrait(kv_cache.rs:22) 제거하여 standalone 재정의 + `forward_into_offload<C>`(transformer.rs)·`preload_erased::<C>` 호출의 generic 소비를 concrete `OffloadKVCache` 로 monomorphize. 권장 처리(ADR §8.3 B-3 + SSOT §9.1:753) = offload 를 (4) 트랙에서 **분리**(supertrait bound 제거안).
- **권장 역할**: Architect (census) → Implementer (구현+host) → Tester (`--kv-offload` sanity). **실제 수행**: 메인 세션(census + Option A 구현 + host gate + 적대검증 workflow `w06swlxi9`).
- **Acceptance Criteria** — ✅ 충족:
  - ✅ host: build + clippy `--workspace -D warnings` + fmt clean. offload 58 + preload 14 + base-vs-offload accuracy test pass. `grep "PrefetchableCache:"` supertrait 0 / `forward_into_offload<` generic 0.
  - ✅ device(host CPU 대체): `--kv-mode offload --kv-offload-storage raw --kv-type f16` host CPU greedy n=32 crash-free + sane + BEFORE/AFTER **생성텍스트 bit-identical**(md5 `568a03e...` 동일, timing 라인만 wall-clock 노이즈). full lib 1229 pass / 26 fail(24 opencl GPU부재 + 2 RSS flaky, 둘다 비-회귀). **S25/Jetson device 게이트는 forward 무변(monomorphization-only)이라 불요** — Option A 는 코드 경로를 안 바꿈.
  - ✅ hot path 무변(plan path 미접촉).
- **수행 결과 (Option A 정밀 범위 — V3 적대검증 반영, 과장 금지)**: Step 2 = **B-3 cluster 분리(5 차단자 중 1개)** — offload-**specific** KVCacheOps 결합(supertrait + offload-전용 generic `forward_into_offload<C>`/`preload_erased::<C>`)만 제거. **(4) KVCacheOps 폐기 자체는 미진전** — B-1(plan)/B-2(forward chain)/B-4(eval)/B-5(legacy) 차단자 잔존.
- **★잔여 = B-2 OLD-chain 삭제 prerequisite (Step 5 선결, 강제 유지지 설계 종착 아님)**: Option A 후에도 `forward_into_offload` 본체가 **공유 OLD B-2 layer chain**(`layer.forward<OffloadKVCache>` → `forward_gen<C>`/`forward_prefill<C>`)을 계속 소비 + `impl KVCacheOps for OffloadKVCache`(offload.rs:263) 유지. 이는 설계 의도가 아니라 **B-2 OLD chain 이 살아있는 한 강제 유지**다. 최종 `KVCacheOps` 삭제(Step 5: OLD layer chain 삭제) 이전에 **`forward_into_offload` 의 fmt 이주**(`forward_gen_fmt`/`forward_prefill_fmt` + `OffloadKVCache: KVCacheFormat` interior-mutability + preload pool aliasing 재설계 = **Option B**, **device GPU 재검증 필수**)가 필요하다. **순서 명시**: `forward_into_offload` 는 `forward_into<C>` 의 caller 가 **아니라 자체 layer loop 를 가진 별개 forward** 라 ①-a~①-d(=forward_into caller 이주)가 미접촉 → **Step 5 의 OLD-chain 삭제 단위에 포함**(또는 그 직전 전용 증분). `run_chunked_prefill`(prefill.rs)과 동급의 B-2 OLD-chain 잔여 소비자.
- **Notes**: `preload_erased<C: PrefetchableCache>` 는 standalone bound 로 generic 유지(KVCacheOps 비의존 = 차단자 아님; roadmap 의 "비-generic 전환"은 KVCacheOps 결합 제거가 목적이었고 supertrait 제거로 달성). PrefetchableCache trait 은 standalone 으로 보존(`재정의`, 삭제 아님).

---

## [P1] Step 3 — (3p) ④-a hot-path flip (plan path concrete-handle)
- **Status**: **설계 ✅** (`design_alpha_k_3p_cut_2026_06_04.md`, workflow `wf_2be25cb8-bc9` 3 design + 3 verify) → **device 구현·게이트 대기**. (3d) plan-eval = flip 확정(선결 충족). **acceptance = device 전용**(plan GPU-only) — host 세션 미완료 가능.
- **Sprint**: next
- **Dependencies**: Step 1 (B-2 prefill flip — plan 평가가 forward chain 정합 필요) + Step 2 권장 선행. **(3d) plan 평가 = (3p) 분기 결정 → flip 확정**(SSOT line 761 BC 결정 + ①-b S25 device PASS 정합). (3p) 실작업 = step() fmt/plan 상호배타 해소. World A eviction seam(F1)은 hard 선결 아님(decode_loop try_evict 0건).
- **설계 결론(`design_alpha_k_3p_cut`)**: ④-a = plan-local 최소 trait `PlanCacheHandle`(plan_geometry 1 lock + plan_advance 1 lock + plan_kv_bufs read seam) + concrete-handle monomorphize(`execute_fmt<H>`, **vtable 0**). flip 표면 4개 = execute(C 6지점) + StandardFormat/KIVIFormat inherent + **build_plan(★V1 적대검증이 잡은 누락 — KVCache pub k_buffer/v_buffer 직접 접근 :2577/2760 → fmt seam 필요)** + ModelForward wiring(fmt/plan 상호배타 해소). perf = neutral-or-slightly-worse(2 lock/layer, ~32 lock/tok, <0.01% TBT 예상·device 실측). ④-b(AttentionVariant 평탄화) defer 확정(attention=enum static, C 미접촉). host scaffold = device 라운드 동행(독립 land 안 함 — orphan dead_code 위험 + revert 격리).
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
- **차단 cluster**: 없음 (deletion). B-5 legacy(`legacy/generate.rs`)가 `forward_into`/`execute_plan`/`execute_plan_for_kivi`/`forward_into_offload`/`run_eval_ll_generic` + `use KVCacheOps`(:4044/4645) 전부 호출 → legacy 폐기로 *legacy 측* 소비자 제거. **⚠️ legacy 폐기 ≠ 모든 소비자 제거**: 아래 B-2 OLD-chain 잔여(non-legacy) 가 별도로 남는다.
- **★B-2 OLD-chain 잔여 소비자 (legacy 외, Step 5 진입 전 반드시 migrate)**: `KVCacheOps`/OLD layer chain(`forward_gen<C>`/`forward_prefill<C>`)을 직접 쓰는 **non-legacy** 소비자 2개 — (1) **`forward_into_offload`**(transformer.rs, **OffloadForward** offload_forward.rs:156/190 = `chat/session.rs:547` 사용)가 본체에서 `layer.forward<OffloadKVCache>` + `OffloadKVCache: KVCacheOps`(offload.rs:263) 소비 → **fmt 이주(Option B: `forward_gen_fmt` + `OffloadKVCache: KVCacheFormat` interior-mut + preload pool aliasing 재설계, device GPU 재검증) 필요**. (2) **`run_chunked_prefill`**(prefill.rs, profiler+variance_collector) → `forward_prefill<C>` 소비(①-e 잔여). 둘 다 Step 2(B-3 분리)·①-a~①-d(forward_into caller 이주)에 미포함 → **여기(Step 5) 또는 그 직전 전용 증분에서 처리**.
- **Description**: (a) `legacy_generate` bin + `engine/legacy/generate.rs` 폐기(사용자 결정: legacy disposable, 호환성 불필요). (b) **위 B-2 OLD-chain 잔여 2 소비자(`forward_into_offload`/`run_chunked_prefill`) fmt 이주** → OLD layer chain(`forward_gen<C>`/`forward_prefill<C>`/`forward_into<C>`) + `impl KVCacheOps for OffloadKVCache` 삭제 가능. (c) `KVCacheOps` trait 삭제 — `kv_cache_ops.rs:53` 의 generic monomorphization 정책 주석 제거(ADR-0001 §8 item 1). 컴파일 차단자 0(B-1~B-4 flip + 위 잔여 migrate 완료) 확인 후 진행.
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
(3c-evict ✅)─┴→ Step 1 ✅ (B-2 prefill + B-4 eval, cold) ─┬→ Step 3 ((3p) B-1 plan, HOT crux) ★다음
                Step 2 ✅ (B-3 offload 분리, cold) ───────┘        │
                                                                 ↓
                                          Step 4 (device-gate → argus_cli)
                                                                 ↓
                                          Step 5 (legacy 폐기 + B-2 OLD-chain 잔여 migrate
                                                  [forward_into_offload/run_chunked_prefill]
                                                  + KVCacheOps 삭제)
```

- **위험 순서**: Step 1·2 (cold, hot 무접촉) → Step 3 (hot crux, 회귀 시 (3p)만 revert) → Step 4·5 (legacy 폐기).
- **revoke 연동**(ADR §6.5): Step 3 avg_tbt Δ>+3% **AND** lock-cost 실측 확인 = perf revoke trigger. Step 1·2 회귀는 cold-path 한정(revoke 강도 낮음). bit-identical 회귀는 cold/hot 무관 6.1 발동.
- **handoff cross-ref**: 직전 (3c-evict) = `handoff_alpha_k_3d_entry_2026_06_03.md` / 배경 census = `handoff_alpha_k_substep3_census_2026_06_03.md`. 각 step 완료 시 handoff 작성(`handoff-doc` 스킬 R1~R6).
