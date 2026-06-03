# Handoff: Phase α-K substep (3) — ripple census + SSOT 재정의 + (3a)(3b) 구현 완료

**작성**: 2026-06-03 (3b 완료 갱신)
**HEAD**: `3bc03e59` (3b 구현). 3a `5ea8ad47`, SSOT 재정의 `90bb424f`+`b8937153`, census `27f9fa6c` 이후.
**브랜치**: `master` (origin 미push, ahead=6)
**다음 세션 진입 문장**: **"α-K substep (3c) 진행"** — **첫 device round**. 신 entry `forward_into_fmt` + 신 args struct + `ModelForward` fmt-cache wiring 으로 decode fallback 단일 호출처를 trait object(`&[Arc<dyn KVCacheFormat>]`)로 전환. **★device 게이트: `--no-gpu-plan` 강제 필수**(미강제 시 production 이 plan path 만 타 vacuous, §1 귀결2) + eviction 발화(`-n 256 --memory-threshold-mb 999999 --eviction-target-ratio 0.5`). S25 OpenCL bit-identical 검증.

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
- **(3c) ★다음** 신 entry `forward_into_fmt` + 신 args struct + `ModelForward` fmt-cache wiring으로 decode fallback 단일 호출처 trait 전환 + `attention_into` Q4-GPU-fallback 흡수. **★device — `--no-gpu-plan` 강제 필수** + eviction 발화(`-n 256 --memory-threshold-mb 999999 --eviction-target-ratio 0.5`). **production cache=항상 `new_dynamic`(memory=Some)** 확인됨(model_forward.rs:544) → 3b의 memory=None 에러 경로는 production 미발화(오용 방어 가드).
- **(3d)** prefill path flip + plan 평가(여기서 (3p) 분기 결정).

**write_kv GPU scatter 흡수 형태 = (A) `backend: &dyn Backend` 인자 흡수** (Architect 권고). (B) concrete inherent는 불가 — forward_gen이 `Arc<dyn>`만 들고 `as_any()` 차단이라 concrete 도달 불가. `attention_into`가 이미 backend를 per-call 받으므로 대칭 흡수 자연(backend = execution-owned 범용 핸들, format⊥hardware, agnostic 위반 아님). §4.1 `write_kv` placeholder를 (A)로 구현 시점 확정. (forward_gen 외 `write_kv` 호출처가 backend 보유하는지 (3a)에서 재확인 — 미확인 시 미결.)

## §4. 미결 / open question
1. forward_gen 외 `write_kv` 호출처의 backend 보유 여부 ((3a) census 재확인).
2. forward_prefill(forward.rs) CPU NEON/AVX2/Rayon inline attention이 `attention_into` impl 흡수 시 벡터화 특화 보존되는가 → (3d) 마지막 배치 + device 게이트 실패 시 prefill 옛 경로 격리.
3. `preload_erased::<C>`(transformer.rs:2872,2899) fn-ptr↔trait-object 충돌 = substep(4) 선결(본 (3) 범위 밖, backlog). `forward_into_offload<C: PrefetchableCache>`는 옛 generic 경로로 공존.
4. (3p) ④-b(`AttentionVariant` enum 평탄화)는 (3p)와 한 묶음 가능하나 friction-triggered 별도 평가.

## §5. device 환경 (3c부터 필요, 3a/3b는 host-only)
- **S25 adb `R3CY408S5SB` OK.** CLAUDE.md 권장 backend = `opencl --opencl-rpcmem`. device 게이트 절차 = substep(2) 교훈 재사용 + **`--no-gpu-plan` 강제**(미강제 시 vacuous, §1 귀결2).
- **Jetson 블로커**: `devices.toml`에 jetson/s25 항목 부재(`[devices.host]`만) / ssh 키 미등록 / IP `165.132.107.73:4121` stale / cargo-zigbuild 미설치. → S25 OpenCL 단독 1차 게이트 가능, Jetson CUDA는 보드 직접 접속 빌드 선행(사용자 개입).
- **baseline 동결 = `9b350609`** (substep 1·2a·2b additive/byte-identical라 현 HEAD와 거동 동일).

## §6. 자기점검
- 진입 문장: ✓ "α-K substep (3c) 진행" — 첫 device round. forward_into_fmt 신 entry + ModelForward fmt-cache wiring + attention_into Q4-GPU-fallback 흡수, `--no-gpu-plan` 강제 device 게이트.
- 왜 멈췄나: ✓ census→SSOT 재정의→3a(write_kv backend seam+GPU scatter)→3b(write_kv 비-F32 cast scratch)까지 host-gated+적대검증 완료한 클린 체크포인트. 3c는 첫 device round라 자연 경계(host 작업 종결). 3a+3b 합쳐 write_kv 흡수 완결(GPU scatter fast-path + 비-F32 cast 모두), 남은 건 wiring(3c)과 attention Q4-fallback(3c).
- 최대 landmine: ✓ (3c) device 게이트 `--no-gpu-plan` 미강제 시 vacuous(production plan path가 flip 코드 가림). 진짜 perf revoke는 (3p)에서. (4)는 (3p) 선결 차단(plan-dep). + branch-by-abstraction 필수(ModelForward 단일 `kv_caches: Vec<KVCache>` 공유라 in-place flip 불가 → 신 entry).
- 검증 게이트: ✓ 3b host build+10 test+fmt+clippy(`--workspace -D warnings`) clean. 적대 검증 3 lens(cast-fidelity/additive/safety) refuted=0. production cache=new_dynamic(memory=Some) 직접 검증(model_forward.rs:544, static `new` 호출처는 전부 `#[cfg(test)]`).
- device 가용: ✓ S25 USB(`R3CY408S5SB`). Jetson 블로커 잔존(devices.toml 항목 부재/ssh 키/cargo-zigbuild, 사용자 요청 시 복구). 3c부터 device 필요.

## §7. 커밋 금지 untracked (반복)
`.antigravitycli/`, `.claude/scheduled_tasks.lock`, `papers/.../microbench_*`, `.agent/todos/handoff_microbench_*.md`, 세션 외 `arch/pipeline/`(companion, 내 작업 무관·미커밋·미삭제). 명시 파일만 add (`git add -A` 금지).
