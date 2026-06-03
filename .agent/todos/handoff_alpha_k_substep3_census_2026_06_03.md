# Handoff: Phase α-K substep (3) — (3a)(3b)(3c-fwd)(3c-evict) 완료, 다음 (3d)

**작성**: 2026-06-03 (3c-evict 구현 완료 — plan_keep keep-list seam + compact 등가성 host gate)
**HEAD**: `2f014163` (3c-evict 구현). cut 확정 SSOT `cd917319`. 3c-fwd `c2b05aff`, 3b `3bc03e59`, 3a `5ea8ad47`.
**브랜치**: `master` (origin 미push)
**다음 세션 진입 문장**: **"α-K substep (3d) 진행"** — prefill path flip + plan 평가(여기서 (3p) 분기 결정 — plan path 가 `KVCacheOps` 유지인지 trait flip 인지) + **World A eviction 발화 배선(EvictionStage→cache 도달 seam, Phase 4-4) + device bit-identical eviction 게이트**(3c-evict 에서 defer 된 분). device.

> **★★ Architect 게이트 재확정 (2026-06-03, F1~F4 재평가 → 갈래 (γ) 확정, SSOT §9.1-EVICT-DECISION)**: 직전 핸드오프의 "eviction 발화 device 게이트(OFF=World B vs ON=World A bit-identical)"는 **F3 으로 성립 불가** — World A eviction 은 forward *전*(`EvictionStage::before_step` line 159, 캐시 권한 부재 + `build_standard_loop` 미주입 = `NoOpEvictionStage`) 발화, World B 는 forward *후*(`run_auto_eviction` legacy:2278) 발화 → 타이밍 한 토큰 어긋나 발산. F4 = compact 가 World B 와 동일 `buffer_shift` 재사용이라 device 증명 가치 낮음. **확정 = (γ)**: **(1) 1차 게이트 = host unit test**(`CpuBackend` 동일 buffer_shift 경로로 정책별 in-place evict 버퍼 == keep-list compact 버퍼 bit-identical, 4 정책 × 3 dtype). **(2) DecodeLoop 발화 배선 + device bit-identical eviction = (3c-evict) 범위 밖 → (3d)/Phase 4-4 로 defer**(World A eviction 발화 seam 미구축). **(3) device 는 sanity 보조만**(crash-free + sane, bit-identical 아님). **소스 hot-path/DecodeLoop/legacy generate.rs 미수정**(발화 배선 후속).

> **★(3c) staging 결정 (2026-06-03, 사용자 위임)**: census 가 "eviction 발화 + fmt flip 동시 성립 불가"(StandardFormat by-value 소유 ⊥ eviction 의 연속 `&mut [KVCache]` 요구)를 잡음. 해소 = ADR §4.2 interior-mutability(eviction 도 `&self` compact 경유)로 **(3c)를 (3c-fwd: forward flip) + (3c-evict: eviction→compact flip)** 두 increment 로 분리. (3c-fwd) = NoEviction happy-path 에서 forward flip 정확성만 검증(완료). **Architect 후속 SSOT 개정 필요**: §9.1:750 (3c) 게이트 조건 — (i) "eviction 발화"를 (3c-evict) 로 이동(forward flip 은 NoEviction), (ii) 게이트 명령에 eviction *subcommand* 누락 정정(`--eviction-target-ratio` 는 일반 필드라 happy-path 탈출 못 함 → `eviction sliding` 등 subcommand 필수), (iii) F32 host-mapped(rpcmem/zero-copy/CPU) bit-identical carve-out 명시(F16/Q4_0/F32-device-only 만 동치).

> **이번 세션 결과 (2026-06-03, (3c-evict) 구현 완료)**: `EvictionPolicy::plan_keep(current_pos, target_len, importance) -> Option<(Vec<usize>, Vec<Merge>)>` **default method 신설**(eviction.rs, default `None` = 미지원) + **4 정책 override**(sliding/h2o/streaming/no_eviction — prefix-포함 ascending keep-list 산출, merges 빈; in-place `evict*` 와 **공존**(refactor 0, production World B 무회귀)). **정식 게이트 = host unit test** `engine/src/pressure/eviction/compact_parity.rs`(신규, 9 test): 정책별 in-place `evict*(&mut KVCache)` 버퍼 == `plan_keep → StandardFormat::compact(keep, merges)` 버퍼 **bit-identical**(4 정책 × {F32,F16,Q4_0} 유효영역 byte + current_pos) + H2O+ `plan_keep`=None(deferred) 검증. **모두 PASS**. host build+fmt+clippy(`--workspace -D warnings`) clean, eviction 124 + standard_format 10 회귀 0, compact_parity.rs clippy 0건(잔존 12 = 기존 spec test 부채). **순수 additive·unwired**(어떤 decode loop 도 plan_keep 미호출) — hot-path/DecodeLoop/legacy generate.rs 무수정. **device = N/A**(unwired 라 실행 경로 부재; `LLMRS_KV_FMT`+eviction subcommand 도 World B 라우팅돼 fmt 미engage → device bit-identical eviction 은 (3d) 발화배선 후). 등가성 근거(소스 직접 증명): `compact_keep_positions(keep,0)` 가 contiguous run batching → `shift_positions`(sliding/streaming 의 그 호출) / H2O 는 `evict_with_scores` 가 이미 `compact_keep_positions(keep_all, prefix)` 사용 → prefix-포함 keep 으로 등가.

> **이번 세션 결과 (2026-06-03, (3c-fwd))**: 신 entry `forward_into_fmt` + `ForwardGenFmtArgs`/`forward_gen_fmt`(forward_gen 라이브 decode arm fork, KV→write_kv·attention→attention_into 위임) + `attention_into` Q4-GPU-fallback 흡수(`attention_q4_gpu_fallback` pub(crate)+`&dyn Backend` 일반화, body 무변) + `ModelForward` fmt-cache wiring(`fmt_caches` mem::take wrap + `fmt_eligible` 플래그). **HEAD `c2b05aff`**. 게이트 = env `LLMRS_KV_FMT`(기본 OFF). 설계 워크플로우 `wf_c7b4afc0-204`(첫 설계 = eviction 전제 오류, 사용자 정정), 적대 검증 `wf_e0536e49-397`(must-fix 2건: F32 doc honesty + chat footgun → fmt_eligible 반영). **host**: build+clippy(--workspace -D warnings)+fmt clean, standard_format 10/10, CPU bit-identical(F16/Q4_0 16~32tok). **★S25 device PASS**(2026-06-03): F16(rpcmem) 32tok / Q4_0(rpcmem, Q4-GPU-fallback) 32tok / device-only F16 24tok 전부 bit-identical(forward_into vs forward_into_fmt, `--no-gpu-plan` 강제로 fmt fallback 실행 — KV_FMT ON 28캐시 wrap 확인, non-vacuous). aarch64 빌드 = `/opt/android-ndk` linux-x86_64(android.source 는 macOS deprecated, hosts.toml 참조).

> **이번 세션 결과 (2026-06-03)**: substep(3) 진입 작업 = type-flip ripple census(워크플로우 `wf_c2e4bf13-9e3`, 5 island + synthesis + 3-lens adversarial) 완료. census가 **§9.1/ADR의 substep(3) perf-crux 규정이 부정확함**을 소스로 반증 → Architect 라우팅으로 SSOT(arch §9.1·§8·§1.1 / ADR §8.3·§6.5 / spec/41) 재정의 완료. **impl 코드는 미착수**(사용자 "Architect 라우팅 먼저" 선택). 핵심 발견 = 아래 §1.

> SSOT = `arch/pipeline_stage_design_v2.md` (§9.1 substep 게이트 표 + §1.1/§8 INV-HOTPATH-DISPATCH cold/hot 정정). ADR = `docs/adr/0001-kv-dispatch-paradigm.md` §8.3 (substep 목록 + (3p) 신설). 트랙 메모리 = [[project-pipeline-alpha-k]].

---

## §1. 핵심 발견 (소스 직접 검증 — 권위)

핸드오프/§9.1 기존 규정: substep(3) = "forward path → trait object, **layer-tier (N×/token) ★perf 위험 집결**". **census가 이를 반증:**

- **production GPU decode hot path = plan path, NOT forward_gen.** `session/forward/model_forward.rs::step`이 매 decode step `execute_plan` 먼저 시도 → `Ok(true)`면 즉시 return. `forward_into`(→`forward_gen`)는 **plan invalidation / build 실패 / `--no-gpu-plan` 시에만** 도는 **cold/fallback tier**. (`try_build_plan` model_forward.rs:161 `if !self.plan_enabled` — `--no-gpu-plan`이 plan 완전 우회시키는 실재 메커니즘.)
- **`plan.rs::execute<C: KVCacheOps>` (plan.rs:1257)**: attention을 `AttentionVariant` enum **static dispatch**로 처리, **`attention_into` 호출 0건**. plan은 **여전히 `<C: KVCacheOps>` generic**.
- **`StandardFormat` (standard_format.rs:8-9 docstring)**: "§4.1 R4 상 **cold-path 라 lock 비용 무관**" — SSOT가 Mutex lock 비용을 명시적으로 무관 처리.

**귀결:**
1. **substep(3) forward_gen flip은 production-hot crux가 아니다.** `Arc<dyn>::attention_into` vtable + `StandardFormat` lock은 cold path라 production TBT 미영향, `INV-HOTPATH-DISPATCH` 위반 아님(ADR §5.2 R-G1 명시 허용).
2. **(3) device 게이트는 `--no-gpu-plan` 강제 없이 vacuous** — production이 plan만 타므로 flip 코드 미실행. 강제해도 측정 대상이 cold path라 production perf를 직접 증명 못 함. (3) 게이트 역할 = fallback path 정확성(bit-identical) + fallback 자체 perf 회귀 부재 한정.
3. **진짜 layer-tier crux = (3p) plan-flip** (신규 substep). plan path를 trait object로 flip하는 것이 production hot. census 이전 어느 substep에도 없어 신설.
4. **(4) `KVCacheOps` 폐기는 (3p) 선결 필수** — `plan.rs::execute<C: KVCacheOps>`가 KVCacheOps의 마지막 generic 소비자(plan-dep). (3p) 전 (4) 진입 시 컴파일 차단.

적대 검증이 잡은 합성 오류 3건(정정 반영): ① in-place flip 불가(`ModelForward` 단일 `kv_caches: Vec<KVCache>` 공유) → **신 entry `forward_into_fmt` + 신 args struct** branch-by-abstraction. ② `KvFormatHandle` enum = 합성 발명(ADR §5.4 OCP 위반 REJECTED) → `&[Arc<dyn KVCacheFormat>]` slice. ③ vacuous gate → `--no-gpu-plan` 강제.

## §2. SSOT 수정 내역 (커밋됨)
- `arch/pipeline_stage_design_v2.md`: §1.1(layer-tier 적용 대상=plan path 정정 노트) + §8 표(INV-HOTPATH-DISPATCH cold-tier dyn 허용) + §9.1(substep 표: (3) cold 강등 / (3p) 신설 / (4) plan-dep / cold-hot ⚠️ 블록 / (3) cut-point 3a-3d 확정 / write_kv 권고).
- `docs/adr/0001-kv-dispatch-paradigm.md`: §8.3(substep 목록 (3) cold + (3p) + (4) plan-dep + 2026-06-03 census 갱신 주) + §6.5(revoke 무게가 (3p)에, (3)은 cold라 trigger 약함).
- `spec/41-invariants.md`: 후보 INV 섹션 INV-HOTPATH-DISPATCH에 cold-tier 허용 명시(여전히 Phase α-K 구현 시점 등록, normative 문구 예고만 — 기존 패턴 정합, spec test 불요).

## §3. 확정된 substep(3) cut-point (branch-by-abstraction)
in-place flip 불가 → **신 entry `forward_into_fmt` + 신 args struct + `ModelForward` fmt-cache wiring**으로 fallback 분기 후 단일 호출처 전환.

- **(3a) ✅ 완료 (`5ea8ad47`)** — `write_kv`/`write_kv_batch`에 `backend: &dyn Backend`(form A) 추가. StandardFormat::write_inner가 GPU F16/F32 HeadMajor decode scatter fast-path 흡수(host 미진입, device 3c) + 표준 경로 KVCache::update fallback. KIVIFormat::write_inner가 update_kv_cache CPU-only(sync+update) 흡수. MISSING method는 impl 내부 소비(base-trait creep 0). host build+13 test+fmt+clippy clean + 적대 검증 3 lens refuted=0. **비-F32 cast scratch + device readback + Q4-GPU-fallback은 3b/3c 연기**(unwired라 무회귀).
- **(3b) ✅ 완료 (`3bc03e59`)** — `write_kv` 비-F32 CPU cast scratch(forward_gen `memory.alloc`+`ws.k_cast`) 흡수. **설계 결정 = 옵션 ①**(StandardFormat 내부 lazy scratch, allocator는 inner `KVCache.memory()`; write_kv signature에 memory 미추가 — KVCache가 이미 동일 allocator 보유, ② trait 표면 오염 기각). struct `Mutex<KVCache>`→`Mutex<StandardFormatInner{cache,k_cast,v_cast}>`(단일 lock). buf_size 3-arm(F16=*2/Q4_0=block/else=*4) + **shape 가드**(write_kv/write_kv_batch가 cast 분기 공유 → batch↔decode 혼용 시 scratch 재할당; forward_gen은 decode-only라 불필요했던 추가). `KVCache::memory()` pub(crate) read-only Arc clone(additive). host build+10 test(신규 3: cast/batch-decode-realloc/requires-dynamic)+fmt+clippy(`--workspace -D warnings`) clean + 적대 검증 3 lens refuted=0. **여전히 unwired(production write_kv 호출처 0)라 무회귀**. ⚠️ **`attention_into` Q4-GPU-fallback 정밀 재현은 device(3c)로 잔여 연기** — 3b는 write_kv cast 만 흡수, attention Q4 fallback은 미착수.
- **(3c-fwd) ✅ 완료 (`c2b05aff`)** — 신 entry `forward_into_fmt`(transformer.rs) + `TransformerModelForwardFmtArgs` + `forward_gen_fmt`(transformer_layer/forward_gen_fmt.rs, forward_gen 라이브 decode arm fork) + `attention_into` Q4-GPU-fallback 흡수(standard_format.rs, `attention_q4_gpu_fallback` pub(crate)+`&dyn Backend` 일반화) + `ModelForward` fmt-cache wiring(`fmt_caches` mem::take wrap + `fmt_eligible`(single-prompt happy-path 만 true) + step() fmt 분기 + reset_kv 라우팅). 게이트 = env `LLMRS_KV_FMT`(기본 OFF). **NoEviction happy-path 한정**(eviction 은 (3c-evict)). host bit-identical(CPU F16/Q4) + **S25 device PASS**(F16/Q4_0 rpcmem 32tok + device-only F16 24tok 전부 forward_into 와 일치, `--no-gpu-plan` 강제). 적대 검증 must-fix 2건 반영(F32 doc honesty: F16/Q4_0/F32-device-only 만 bit-identical, F32 host-mapped 발산 명시 / chat footgun: fmt_eligible 로 멀티턴 prefill panic 차단).
- **(3c-evict) ✅ 완료 (`2f014163`)** — `EvictionPolicy::plan_keep` default method(`None`) + 4 정책 override(sliding/h2o/streaming/no_eviction, prefix-포함 ascending keep-list, merges 빈) + host unit test `compact_parity.rs`(9 test, 4 정책 × 3 dtype bit-identical + H2O+ None). in-place `evict*` 와 공존(refactor 0). 정식 게이트 = host(γ 확정, 위 ★★). **순수 additive·unwired** — DecodeLoop/legacy/hot-path 무수정. **deferred(§9.1-EVICT-DEFER)**: H2O+(per-head, 단일 keep-list 불가) → (3c-evict-perhead) / D2O(가중 merge ≠ 균등평균, `EvictionPolicy` 아님) → (3c-evict-d2o). **DecodeLoop 발화 배선 + device bit-identical = (3d)/Phase 4-4**.
- **(3d) ★다음** prefill path flip + plan 평가(여기서 (3p) 분기 결정). **+ World A eviction 발화 seam 구축**(F1: `EvictionStage::before_step` 가 캐시 권한 부재 + `build_standard_loop` 미주입 = `NoOpEvictionStage` → real EvictionStage 또는 `Forward::try_evict` 배선 필요, Phase 4-4 영역) → plan_keep→compact 를 live 발화시켜 **device bit-identical eviction 게이트**(3c-evict defer 분) 수행. ADR §4.2 write-path 마이그레이션(CacheManager/D2OHandler)도 발화 배선과 함께.

**write_kv GPU scatter 흡수 형태 = (A) `backend: &dyn Backend` 인자 흡수** (Architect 권고). (B) concrete inherent는 불가 — forward_gen이 `Arc<dyn>`만 들고 `as_any()` 차단이라 concrete 도달 불가. `attention_into`가 이미 backend를 per-call 받으므로 대칭 흡수 자연(backend = execution-owned 범용 핸들, format⊥hardware, agnostic 위반 아님). §4.1 `write_kv` placeholder를 (A)로 구현 시점 확정. (forward_gen 외 `write_kv` 호출처가 backend 보유하는지 (3a)에서 재확인 — 미확인 시 미결.)

## §4. 미결 / open question
1. forward_gen 외 `write_kv` 호출처의 backend 보유 여부 ((3a) census 재확인).
2. forward_prefill(forward.rs) CPU NEON/AVX2/Rayon inline attention이 `attention_into` impl 흡수 시 벡터화 특화 보존되는가 → (3d) 마지막 배치 + device 게이트 실패 시 prefill 옛 경로 격리.
3. `preload_erased::<C>`(transformer.rs:2872,2899) fn-ptr↔trait-object 충돌 = substep(4) 선결(본 (3) 범위 밖, backlog). `forward_into_offload<C: PrefetchableCache>`는 옛 generic 경로로 공존.
4. (3p) ④-b(`AttentionVariant` enum 평탄화)는 (3p)와 한 묶음 가능하나 friction-triggered 별도 평가.

## §5. device 환경 — (3c-fwd) 게이트 검증 절차 (재사용 가능)
- **S25 adb `R3CY408S5SB` OK.** 권장 backend = `opencl --opencl-rpcmem`.
- **aarch64-android cross-build (검증된 절차)**: `android.source` 는 macOS 경로라 **deprecated** — Linux host 는 `hosts.toml` 의 `/opt/android-ndk`(linux-x86_64, api 21) 사용. env 직접 설정:
  ```
  NDK=/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64
  export CC_aarch64_linux_android=$NDK/bin/aarch64-linux-android21-clang
  export CXX_aarch64_linux_android=$NDK/bin/aarch64-linux-android21-clang++
  export AR_aarch64_linux_android=$NDK/bin/llvm-ar
  export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=$NDK/bin/aarch64-linux-android21-clang
  cargo build --release --target aarch64-linux-android -p llm_rs2 --bin legacy_generate
  adb push target/aarch64-linux-android/release/legacy_generate /data/local/tmp/legacy_generate
  ```
  (devices.toml 에 s25 항목 부재라 run_device.py 대신 직접 adb push. 빌드 ~57s.)
- **(3c-fwd) 게이트 명령 (bit-identical, happy-path)**: device 의 `/data/local/tmp/models/qwen2.5-1.5b/{f16,q4_0}.gguf` 사용. eviction subcommand **없이**(happy-path), `--no-gpu-plan` 강제:
  ```
  # gate OFF (forward_into):  ./legacy_generate -m <gguf> -b opencl --opencl-rpcmem --no-gpu-plan --greedy -n 32 -p "..."
  # gate ON  (forward_into_fmt): LLMRS_KV_FMT=1 <동일>
  # LLMRS_FWD_TRACE=1 시 "KV_FMT ON: wrapped N KVCache" 로 fmt 경로 engage 확인(non-vacuous).
  ```
  → 두 출력 텍스트 bit-identical 이면 PASS. **F16/Q4_0 KV 만**(F32 host-mapped 은 발산 — §1 doc honesty). **(3c-fwd) 결과 = F16/Q4_0 rpcmem 32tok + device-only F16 24tok 전부 PASS.**
- **Jetson 블로커**: `devices.toml` jetson/s25 항목 부재 / ssh 키 / cargo-zigbuild 미설치. S25 단독 1차.
- **baseline 동결 = `9b350609`** (substep 1·2a·2b·3a·3b·3c-fwd 모두 additive/게이트 OFF 라 현 HEAD 와 production 거동 동일).

## §6. 자기점검 (3d 진입용)
- 진입 문장: ✓ "α-K substep (3d) 진행" — prefill path flip + plan 평가((3p) 분기 결정) **+ World A eviction 발화 seam 구축(Phase 4-4) → (3c-evict) defer 한 device bit-identical eviction 게이트**.
- 왜 멈췄나: ✓ (3c-evict) = `plan_keep` keep-list seam(4 정책) + compact 등가성 **host unit test 9개 전부 PASS**(γ 확정 정식 게이트) + 순수 additive·unwired 완료한 클린 체크포인트(`2f014163`). 발화 배선은 World A eviction seam(미구축, Phase 4-4)이 선결이라 (3d)로 의도적 분리 — (3c-evict) 는 함수-레벨 등가성 증명에 한정(SSOT §9.1-EVICT-DECISION 결정 1).
- 최대 landmine: ✓ (1) **(3d) eviction 발화 배선의 진짜 난점 = F1** — `EvictionStage::before_step(&StepCtx)` 는 캐시 접근 권한이 없고(model_forward.rs:227,522 "Phase 4-4 TODO"), `build_standard_loop` 는 `NoOpEvictionStage` 만 주입, `Forward::try_evict`(model_forward.rs:542)는 DecodeLoop 이 호출조차 안 함(decode_loop.rs:159 는 `eviction.before_step` 만). **즉 World A 에서 eviction 은 한 번도 발화한 적 없고 seam 자체가 미구축** — 이걸 짓는 게 (3d)/Phase 4-4 의 핵심. (2) **발화 타이밍 = forward 전(DecodeLoop) vs 후(World B legacy)** — bit-identical device 게이트를 World 분할로 구성하면 한 토큰 어긋나 발산(F3). 같은 World·타이밍 안에서만 OFF/ON 비교 성립. (3) plan_keep 은 fmt active 시만 의미 — `try_evict` 가 fmt 활성 시 빈 `self.kv_caches`(mem::take)를 봐서 no-op 되는 것을 fmt 분기로 고쳐야(plan_keep→`fmt.compact`). active policy 도달 경로(CacheManager 가 policy 를 EvictionHandler 캡슐화)도 (3d) 미결.
- 검증 게이트: ✓ (3c-evict) host build+fmt+clippy(`--workspace -D warnings`) clean + compact_parity 9/9 + eviction 124 + standard_format 10 회귀 0. compact_parity.rs clippy 0(잔존 12 = 기존 spec test 부채, alpha-k 메모리 line 40).
- device 가용: ✓ S25 USB(`R3CY408S5SB`) + cross-build 절차 §5 확립. (3c-evict) 는 device N/A(unwired). (3d) 발화 배선 후 §5 절차로 게이트. Jetson 블로커 잔존.

## §7. 커밋 금지 untracked (반복)
`.antigravitycli/`, `.claude/scheduled_tasks.lock`, `papers/.../microbench_*`, `.agent/todos/handoff_microbench_*.md`, 세션 외 `arch/pipeline/`(companion, 내 작업 무관·미커밋·미삭제). 명시 파일만 add (`git add -A` 금지).
