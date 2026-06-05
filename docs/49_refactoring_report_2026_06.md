# 추론 엔진 리팩토링 리포트 (2026-06) — Layered Pipeline / KVCacheFormat 통일 / argus-bench

> **상태**: 리포트 초안 (다음 세션에서 정식 리포트로 정제 예정). 추가 리팩토링은 본 문서 작성 시점에 일시 중단.
> **대상 아크**: Phase 4(레이어드) → bin 분할 → α-W → α-K BC → 5-F(비가역 cutover) → argus-bench(AB-0/1/3)
> **기준 HEAD**: `1bb4c006` (origin/master, 2026-06-05)
> **관련 문서**: `arch/inference_pipeline.md`, `arch/pipeline_stage_design_v2.md`, `docs/47_refactoring_analysis_2026_05.md`, `.agent/todos/handoff_argus_bench_ab0_ab3_2026_06_05.md`

---

## 0. Executive Summary

llm.rs 추론 엔진을 **모놀리식 + 제네릭 monomorphization** 설계에서 **레이어드 파이프라인 + concrete-handle 포맷 추상화**로 통일하는 다단계 리팩토링. 핵심 성과 3가지:

1. **모놀리식 해체**: `generate.rs`(5098줄)를 `DecodeLoop` + 6개 SOLID trait + `ModelForward` + assembly 헬퍼로 분해. 추론 step이 고정된 trait 호출 시퀀스가 됨.
2. **패러다임 통일**: KV 캐시 forward 경로의 `<C: KVCacheOps>` 제네릭 monomorphization을 `Arc<dyn KVCacheFormat>` trait-object(concrete handle + interior mutability)로 이주. `KVCacheOps` trait 완전 폐기.
3. **bin 패밀리화 + resilience 재구현**: 모놀리식 `generate` → `argus_cli`(single-prompt) + `argus_bench`(resilience benchmark/verify). 5-F에서 삭제된 resilience 모드(eviction/offload/...)를 fmt 경로에 재구현 중.

검증은 전 단계 **host 게이트(build + clippy `-D warnings` + 1217 test) + S25 device bit-identical**로 통과. 비가역 삭제 전 **frozen baseline 동결**. 누적 ~8000줄 삭제, `grep KVCacheOps engine/src` 코드 0건 달성.

---

## 1. 배경 / 동기

리팩토링 이전의 3가지 구조적 문제:

### 1.1 모놀리식 비대화
`engine/legacy/generate.rs` 한 파일에 prefill / decode / eviction / weight-swap / KV-offload / resilience IPC / tensor-partition / profiling이 전부 인라인. **5098줄**. 변경 1건이 전체에 파급되고, 단위 테스트가 불가능(전 경로가 `main()` 스코프 지역변수에 의존).

### 1.2 제네릭 폭증 (`<C: KVCacheOps>`)
KV 캐시를 추상화한 `KVCacheOps` trait의 제네릭 바운드가 forward chain 전반에 전파:
```rust
fn forward_into<C: KVCacheOps>(&self, caches: &mut [C], ...)
fn execute<C: KVCacheOps>(plan, caches: &mut [C], ...)   // GPU plan
fn forward_gen<C: KVCacheOps>(...)                        // layer
```
→ 새 캐시 타입(`KiviCache`, `OffloadKVCache`) 추가 시 monomorphize 사본이 증식하고, 캐시 종류별 코드 경로 분기가 폭발. `<C>` 바운드가 호출 그래프 상위로 전파되어 시그니처를 오염.

### 1.3 resilience 어휘 산만
manager↔engine IPC에 `ResilienceAction`(이산 액션) + `MemoryStrategy`(graded 매핑)가 중복 존재. 같은 의도(메모리 압박 → eviction)가 두 어휘로 표현되어 정합성 검증이 어려움.

---

## 2. 전략 — branch-by-abstraction + device 게이트

모든 단계가 동일한 안전 패턴을 따름:

```
   ① additive 추가      신규 경로(fmt)를 env-gate(LLMRS_KV_FMT) OFF 상태로 OLD와 병존
        ↓ host + S25 device bit-identical 검증
   ② flip               게이트 기본값을 신규 경로로 전환 (OLD는 dead-but-present)
        ↓ frozen baseline 동결 (device 캡처)
   ③ delete             OLD 영구 삭제 (비가역, baseline으로 회귀 검증)
```

**검증 게이트(전 단계 공통)**:
- host: `cargo build --workspace` + `cargo clippy --workspace -- -D warnings` + `cargo test -p llm_rs2 --lib -- --skip opencl`(1217 PASS)
- device: **S25(Galaxy, Adreno, OpenCL `--opencl-rpcmem`)에서 bit-identical** — 동일 prompt·seed로 생성 텍스트 + summary md5 일치, avg_tbt Δ 허용범위
- exact-string 로그 보존: verify가 정규식 매칭하므로 로그 문자열 글자단위 유지

---

## 3. 단계별 상세

### 3.1 Phase 4 — 레이어드 추론 파이프라인

모놀리식 decode 루프를 **6개 trait + `DecodeLoop`**로 분해 (SSOT: `arch/inference_pipeline.md`).

**6 trait** (`engine/src/session/traits.rs`):
| trait | 책임 | default |
|---|---|---|
| `Forward` (필수) | `prefill` / `step` / `try_evict` / `on_kv_prune` / `finalize` | — |
| `EvictionStage` | `before_step` → `EvictionOutcome` | NoOp |
| `SwapStage` | `before_step` / `after_step` / `pending_report` | NoOp |
| `CommandSource` | `poll(ctx, kv_snap)` → `ExecutionPlan` | NoOp |
| `TokenSampler` | `sample` / `observe_token` | Greedy |
| `DecodeObserver` | `on_step_end` / `on_eviction` / `finalize` | NoOp |

**`DecodeLoop`** (`engine/src/session/decode_loop.rs`, typestate builder `NoForward`→`HasForward`): 매 decode step이 고정 시퀀스
```
(a) cmd_source.poll → ExecutionPlan
(b) eviction.before_step → Pruned 시 forward.on_kv_prune
(c) swap.before_step
(d) forward.step → logits (+ step_ms 측정)
(e) swap.after_step
(f) sampler.sample
(g) observers.on_step_end
```

**구현체 / 조립**:
- `ModelForward` (`session/forward/model_forward.rs`): 표준 `KVCache` 기반 Forward.
- `build_standard_loop` (`session/assembly/`): `SessionInitCtx` unpack → ModelForward + sampler + resilience 조립. `is_standard_happy_path` 가드로 진입 제한.
- eviction을 model에서 분리(SRP): `LlamaModelForwardArgs`에서 `cache_manager` 제거, 호출자(generate)가 forward 후 별도 eviction.

### 3.2 bin 분할 (`7065196c`)

모놀리식 `generate` → **`argus_cli`**(single-prompt happy-path, 신규 bin) + legacy `generate` 동결. ARGUS bin 패밀리의 시작점. argus_cli는 `is_standard_happy_path` 가드 통과 args만 DecodeLoop로 위임하고 나머지는 legacy로 fallback.

### 3.3 Phase α-W — 확장 파이프라인 + 어휘 통일 (SSOT: `arch/pipeline_stage_design_v2.md`)

| 서브 | commit | 변경 |
|---|---|---|
| α-W-1 | `0d12c81d` | 확장 파이프라인 L2 타입 스캐폴딩 신설 |
| α-W-2 | `1a1cd444` | **`Hardware`**가 `SessionInitCtx`의 흩어진 cpu/gpu backend·memory secondary Arc 4개를 흡수 — `hardware.resolve(DeviceTarget)` → `(backend, memory)` |
| α-W-3 | `57629f26` | **`ResilienceAction`/`MemoryStrategy` 폐기 → `EngineCommand` 단일 이산 어휘**. memory도 graded `Pressure`로. `strategy/memory.rs` 삭제 |
| α-W-4 | `42cb9066` | **`CapabilityRegistry`** 물리 정착 — OpenCL backend에 `KiviAttentionBackend` handle 등록(`SessionInitCtx.caps`), L2 capability 축 배선 |
| α-W-5 | `3e5dc49f`·`ef723da7` | `PartitionedWeight` 2-fixed 필드 → `Vec<WeightSlice>` 일반화, `WeightFormat::apply_dispatch` 배선 (`prepare_tensor_partition` 일반화) |

> 잔여 **α-W-3b**(resilience 2-source defer-B)는 Phase β로 이월.

### 3.4 Phase α-K BC — KVCacheOps → KVCacheFormat ★핵심

**제네릭 monomorphization → concrete-handle trait-object** 패러다임 전환.

```rust
// BEFORE: 컴파일타임 monomorphize, <C> 바운드가 forward chain 전반 전파
fn forward_into<C: KVCacheOps>(&self, caches: &mut [C], ...)

// AFTER: 런타임 trait-object. KVCache를 Format으로 wrap
fn forward_into(&self, fmts: &[Arc<dyn KVCacheFormat>], ...)
//   StandardFormat(KVCache) / KIVIFormat(KiviCache) / OffloadFormat(OffloadKVCache)
//   interior mutability → forward·eviction 모두 &self 통과 (ADR-0001 §4.2)
```

**핵심 아이디어**: `KVCache`를 `StandardFormat`으로 by-value wrap(`Arc::new(StandardFormat::new(i, c))`). `StandardFormat`이 내부 가변성(`with_cache_mut` / `take_inner` / `put_inner`)을 제공하므로, forward(읽기)와 eviction(쓰기)이 같은 `Arc` 공유 핸들을 통해 `&self`로 동작. `<C>` 바운드 소멸.

**단계별 flip** (env-gate `LLMRS_KV_FMT` OFF → additive → flip):
| 단계 | commit | 내용 |
|---|---|---|
| ①-c~①-e | `1e4f20fe`·`84bed97e`·`2941edca` | eval / 비-decode forward(10 site) / KIVI prefill을 `forward_into_fmt`로 flip |
| Step 2 | `936d0c99` | offload를 `PrefetchableCache`로 분리 (KVCacheOps 비의존) |
| Step 3 | `004147bf`·`ae9fc460` | **GPU plan hot-path** concrete-handle flip — `build_plan`/`execute_plan`이 `StandardFormat` handle slice 수용. non-opencl 빌드 복구 위해 `cfg(opencl)` 게이트 |
| Step 4 | `b4e2deee` | device 게이트 → argus_cli S25 등가 검증 PASS |
| Step 5-B | `7a22bb63`·`bb8a200f` | offload `KVCacheFormat` 이주 (decode arm bit-identical). cast scratch 영속화로 S25 avg_tbt +5.8%→+0.58% |
| Step 5-C | `641dc932` | `KiviForward`를 `forward_into_fmt`로 이주 (OLD-chain 소비자 제거) |
| Step 5-E | `cbb5f376` | `KVCacheOps` 본문 메서드를 각 캐시 **inherent**로 이전 + 소비자 rewire |
| (3d) S1~S4 | `928a95ee`·`1eae6c46`·`1d8027ae` | chat eviction을 **UER(Unwrap-Evict-Rewrap)** seam으로 fmt-wrap (S25 γ-sanity PASS) |

### 3.5 Phase 5-F — 비가역 cutover (F0~F4) ★대삭제

fmt를 production 유일 경로로 확정하고 OLD를 영구 삭제.

| F | commit | 삭제/변경 | 게이트 |
|---|---|---|---|
| F0 | `8e7ffc67` | `LLMRS_KV_FMT` 게이트 상수화 — fmt = production 기본 | non-opencl 1239 PASS |
| baseline | `fec8ad23` | legacy 삭제 전 S25 frozen baseline 동결 (F16 weight = build_plan SUCCESS = 비-vacuous) | f16/f32/q4 sig+avg_tbt 캡처 |
| **F1 ★** | `d5ed71d2` | **legacy `generate.rs`(5098줄) + bin 폐기** (비가역) | build+clippy clean |
| F2a | `ba33ac86` | `ModelForward` fmt-only + **`decode_fallback`(prologue 668 / eviction_trigger 207 / swap_dispatch 504) + prefill.rs(746) + parity test 삭제** | 1220 PASS |
| F2b/c | `05c019ee`·`7623f441` | OLD `forward_into<C>` / `execute<C>` / `forward_gen<C>` generic chain + 잔여 소비자 삭제 | 1218 PASS |
| **F3 ★** | `102f0461` | **`KVCacheOps` trait + impl×3(KVCache/KiviCache/OffloadKVCache) 영구 폐기** | grep 0건, 1217 PASS |
| F4 | `9326a096` | `*_fmt` 접미사 제거 (mechanical rename) | fmt clean |

**device 게이트 결과** (S25 OpenCL, 비가역 acceptance):
- **bit-identical 3/3**: argus(fmt-only) ≡ frozen baseline — f16 `304f4ada` / f32 `684d01d9` / q4 `1cfba273`. build_plan SUCCESS + wrap 발화(non-vacuous).
- **avg_tbt Δ ≤ ±0.5%** (median n=5): f16 −0.31% / f32 +0.22% / q4 −0.41% — monomorphization 제거의 perf 회귀 0(설계 예측 neutral 확인).

### 3.6 argus-bench — decode_fallback 모드 fmt 재구현 (AB-0/1/3, 현재)

5-F에서 삭제된 `decode_fallback`(eviction-during-decode + weight-swap orchestration glue)을 fmt `DecodeLoop`에 **재구현** + PARKED된 verify 하네스 재배선. legacy 자리를 happy-path 전용 argus_cli가 아닌, **experiment-output + resilience runtime effect를 지원하는 `argus_bench`**로 대체.

| AB | commit | 핵심 변경 (file) |
|---|---|---|
| **AB-0** | `f18d0b6e` | 신규 `argus_bench` bin (`bin/argus_bench.rs`) + `bin_setup`(`session/bin_setup.rs`, argus_cli와 셋업 공유 — `build_inference_ctx`) + `run_experiment_path`(`session/experiment_run.rs`, per-token JSONL `TokenRecord`/`SummaryRecord` + `[Experiment] Done`) + **`DecodeLoop::run` target_tbt pacing** + executor `SetTargetTbt` 로그 + Suspend 로그 + verify orchestrator host binary 재배선(`ENGINE_BIN`) |
| **AB-1** | `94ef643f` | **mid-decode eviction** — `DecodeLoop`에 `cache_manager`/`evict_applied` 필드 + poll 후 `plan.evict`(EvictPlan) 소비 → `forward.try_evict`(fmt UER) + `build_bench_loop`/`build_resilience_cache_manager`(`session/assembly/build_bench_loop.rs`) 신설 + `ModelForward::on_kv_prune` GPU plan 무효화 + verify `_engine_env` RUST_LOG=info 주입 |
| **AB-3** | `95908f57` | **KvOffload/recall** — `Forward::try_offload`/`try_recall`(UER) + `DecodeLoop` 소비 + `build_resilience_cache_manager` `--swap-dir`→`enable_swap` 글루 |

**검증**: host verify 6 시나리오 green — throttle_smoke / target_tbt(×2) / thermal_emergency_suspend / memory_critical_evict / prefill_midway_injection. AB-3 offload host smoke PASS.

---

## 4. 핵심 before → after 변환 (요약 매트릭스)

| 축 | BEFORE | AFTER | 위치 |
|---|---|---|---|
| 추론 루프 | 모놀리식 `generate.rs` 5098줄 | `DecodeLoop` + 6 trait (고정 step 시퀀스) | `session/decode_loop.rs`, `traits.rs` |
| KV 추상화 | `<C: KVCacheOps>` 제네릭 | `Arc<dyn KVCacheFormat>` trait-object | `format/`, `pressure/standard_format.rs` |
| forward 호출 | `forward_into<C>(&mut [C])` | `forward_into(&[Arc<dyn KVCacheFormat>])` | `models/transformer.rs` |
| GPU plan | OLD `execute<C>` | `build_plan`/`execute_plan`(handle slice) | `backend/opencl/plan.rs` |
| eviction | model 내장 / legacy trigger | `forward.try_evict` UER(take→evict→put) | `forward/model_forward.rs` |
| resilience 어휘 | `ResilienceAction` + `MemoryStrategy` | `EngineCommand`(18 variant) → `ExecutionPlan` | `shared/`, `resilience/executor.rs` |
| secondary Arc | `SessionInitCtx` 4개 산재 | `Hardware.resolve(target)` 흡수 | `session/init.rs`, `hardware.rs` |
| bin | `generate` 단일 | `argus_cli` + `argus_bench` | `bin/` |

---

## 5. 정량 지표 / 검증

- **삭제 LOC**: legacy generate.rs 5098줄(F1) + decode_fallback 1379줄 + prefill 746줄 + OLD generic chain + KVCacheOps trait. 누적 ~8000줄.
- **`grep KVCacheOps engine/src`**: 함수/trait/impl 정의 **0건** (주석/마이그레이션 노트만 archival).
- **5-F device**: bit-identical 3/3(f16/f32/q4), avg_tbt Δ ≤ ±0.5%.
- **argus-bench host verify**: 6/6 green (AB-0 4 + AB-1 2).
- **non-opencl 회귀 게이트**: `cargo test -p llm_rs2 --lib -- --skip opencl` = **1217 PASS**. (`test_prune_prefix_calls_release_unused_pages`는 병렬압박 flaky — 격리 PASS=비-회귀.)
- **clippy**: `--workspace -- -D warnings` clean (전 단계).

---

## 6. 현재 상태 / 잔여

**완료** (`origin/master` = `1bb4c006`): Phase 4 → α-W → α-K BC 5-F(KVCacheOps 완전 폐기, device PASS) → argus-bench AB-0/1/3.

**잔여 (device phase — verify 시나리오 전부 galaxy_s25/jetson 전용)**:
- **AB-2** KIVI dynamic quant (KiviForward 분기 + caps plumbing + `Forward::try_quant`)
- **AB-4** tensor partition 동적 enable (**GPU 전용** — host smoke 불가; `prepare_tensor_partition(&mut self)` vs `Arc<TransformerModel>` get_mut 불가 landmine)
- **AB-6** weight-swap 8종 (SwapStage trait 시그니처 변경 + intra-forward hook 주입)
- **AB-5** verify adb 재배선(_build/_deploy_remote PARKED 해제 + 크로스빌드) + S25 16/16 device 게이트
- 별개 후행: **Phase β** (DecodeLoop PipelineStage 재작성, α-W-3b 흡수)

S25(R3CY408S5SB) adb 연결 + 모델(f16/q4 gguf) + android-ndk 준비됨. Jetson ssh 오프라인.
상세 구현 맵·landmine: `.agent/todos/handoff_argus_bench_ab0_ab3_2026_06_05.md` + workflow `waq2nsy1b`.

---

## 7. 교훈 / landmines

- **RoPE-after-prune 의미**: AB-1 eviction의 `self.pos = new_pos`(physical) 동기화는 functional_only 시나리오로 가려짐(rouge 0.082 통과). score-driven eviction(AttentionScoreAccumulator 미장착, force_evict score-free, H2O≡recency) 동치는 `pass_criteria: all` 시나리오에서 별도 device 검증 필요.
- **로그 문자열 글자단위 보존**: verify가 정규식 매칭하므로 legacy 로그(`[CacheEvent] Eviction completed: policy='h2o'`, `[Partition] Disabled (ratio=0)` 등)를 글자단위 유지해야 함. F4 rename에서 `forward_gen_fmt`/`forward_prefill_fmt`는 메서드명=모듈명이라 sed 충돌 → 미rename(보류).
- **log::info! vs RUST_LOG**: `[CacheEvent]` 등 effect 로그는 `log::info!` → `RUST_LOG=info` 필요(legacy도 외부 의존). verify orchestrator `_engine_env`로 주입.
- **GPU plan 무효화**: eviction/partition으로 KV pos가 shift되면 보유 `FullKernelPlan`이 stale → `on_kv_prune`에서 `gpu_plan=None` 무효화. CPU(host)는 plan 부재라 no-op.
- **KVLayout HeadMajor 고정**: eviction/d2o/score 경로가 HeadMajor 하드코딩. SeqMajor 선택 시 Plan 경로 silent garbage.
- **AB-4 구조적 landmine**: `prepare_tensor_partition`이 `&mut self`인데 ModelForward는 `Arc<TransformerModel>` 보유(get_mut 불가). LayerSlot ArcSwap으로 &self 토글 가능한지 확인 필요 — 불가 시 비국소 변경.

---

## 부록: 커밋 인덱스 (시간 역순, 주요)

```
1bb4c006 handoff argus-bench AB-0/1/3 → device phase
95908f57 AB-3 KvOffload/recall
94ef643f AB-1 mid-decode eviction
f18d0b6e AB-0 experiment-output bin
e23518e3 α-K BC 종결 handoff
9326a096 5-F F4 fmt 접미사 제거
102f0461 5-F F3 KVCacheOps trait 폐기 ★
7623f441 5-F F2c / 05c019ee 5-F F2b OLD chain 삭제
ba33ac86 5-F F2a ModelForward fmt-only + decode_fallback 삭제
d5ed71d2 5-F F1 legacy generate.rs 폐기 ★비가역
fec8ad23 5-F frozen baseline 동결
8e7ffc67 5-F F0 게이트 상수화
cbb5f376 α-K BC 5-E KVCacheOps inherent 이전
641dc932 α-K BC 5-C KiviForward fmt
7a22bb63 α-K BC 5-B offload fmt
004147bf α-K BC Step 3 plan hot-path flip
936d0c99 α-K BC Step 2 offload 분리
2941edca α-K BC ①-e KIVIFormat prefill
84bed97e α-K BC ①-d 비-decode flip
1e4f20fe α-K BC ①-c eval flip
42cb9066 α-W-4 CapabilityRegistry
3e5dc49f/ef723da7 α-W-5 WeightFormat
1a1cd444 α-W-2 Hardware 흡수
57629f26 α-W-3 EngineCommand 단일 어휘
0d12c81d α-W-1 L2 스캐폴딩
7065196c bin 분할 argus_cli
```
