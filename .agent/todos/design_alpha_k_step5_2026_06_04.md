# Phase α-K BC Step 5 설계 — KVCacheOps trait 폐기 + KVCacheFormat rename

문서 SSOT: `.agent/todos/roadmap_alpha_k_bc_completion_2026_06_04.md` §Step 5
선행: Step 1~4 완료 (cold-flip + offload 분리 + plan hot-path flip + argus_cli 등가)
검증 상태: 본 문서는 4개 census + 4개 설계 + 3개 적대검증 렌즈를 종합. 적대검증 blocking 4건(렌즈1 B1, 렌즈2 BLOCKING-1·2, 렌즈3 BL-1·2) 전부 반영하여 수정된 설계임. 호스트 GPU 부재 → device 검증 = S25 OpenCL + Jetson CUDA.

---

## 1. TL;DR + Step 5 범위

**한 줄 결론**: KVCacheOps trait 삭제는 *동작 변경*이 아니라 *타입 시스템 표면 제거*다. Step 1~4가 모든 production forward 경로를 fmt(`TransformerModelForwardFmtArgs`, non-generic)로 이미 flip 했으므로, 남은 일은 (a) generic `<C: KVCacheOps>`의 마지막 4개 소비자 제거, (b) trait 메서드를 각 cache의 inherent fn으로 사전 이주, (c) atomic cutover로 trait·OLD-chain·legacy 동시 삭제다. 단 1개 구간(offload Option B fmt 이주)만 진짜 설계+device 재검증을 요하고, 나머지는 기계적 rewire다.

**채택한 핵심 결정**:
- **갈래 A (inherent-only)** — micro-trait `KVCachePrimitive` 신설(갈래 B) 기각. fmt 래퍼가 이미 타입별 분리(StandardFormat↔KVCache, KIVIFormat↔KiviCache)라 generic 코드 공유 수요 0 (YAGNI).
- **ADR-0001 "rename"은 역할 이전으로 이미 실현됨** — `format/kv_cache_format.rs::KVCacheFormat`(7-method base trait)이 ADR이 의도한 dispatch 표면. KVCacheOps는 rename 대상이 아니라 **삭제 대상**. KVCacheFormat으로의 중복 rename 금지.
- **일괄 cutover 불가피**(trait 삭제는 atomic)하나, 선행 증분(inherent 보강 + OLD-chain 잔여 제거 + legacy 폐기)으로 cutover의 컴파일 차단자를 0으로 선확인.

**Step 5 범위 (roadmap §Step 5 (a)/(b)/(c))**:
- (a) `legacy_generate` bin + `engine/legacy/generate.rs` 폐기.
- (b) B-2 OLD-chain 잔여 소비자 fmt 이주: **offload(forward_into_offload) + run_chunked_prefill + KiviForward(★roadmap 누락분, 본 설계 신규 식별)**.
- (c) `KVCacheOps` trait 삭제 + `kv_cache_ops.rs` 정리.

**범위 밖 (별도 트랙)**:
- argus family bin(argus-chat/eval/bench) 실제 구현 — 본 설계는 "session 함수 라이브러리 보존(거취 A1)"을 권고하며 family bin은 backlog. **단 offload device 게이트 매체로 argus-bench(offload 지원)가 5-B의 hard 선결로 격상됨**(렌즈3 BL-1).
- tensor-partition session 모듈 추출 (main inline 미추출 → defer).

---

## 2. KVCacheOps 97 참조 분류 요약표 (영역 × disposition)

분류 키:
- **dies-with-legacy**: legacy/generate.rs 폐기(5-F의 F1) 또는 OLD-chain 삭제(5-F의 F2)로 자동 소멸. 주석 참조 포함.
- **needs-rewire**: concrete 타입의 inherent 메서드 호출로 전환 (5-E).
- **migrate-fmt**: fmt 경로로 forward 이주 (5-B/5-C/5-D).
- **removable**: 미사용 import / 재export 라인 삭제 (5-E/5-F).
- **trait-def**: 삭제 본체 (5-F).

| 영역 | 대표 위치 | 참조 성격 | disposition | 처리 증분 |
|---|---|---|---|---|
| trait 정의 | `kv_cache_ops.rs:55-160` | KVCacheOps 정의 (20 메서드) | trait-def (삭제) | 5-F (F3) |
| impl × 3 | `kv_cache.rs:1006` / `kivi_cache.rs:2237` / `offload.rs:263` | KVCache/KiviCache/OffloadKVCache impl 블록 | 본문 inherent 이주(5-E) 후 삭제(5-F) | 5-E→5-F |
| fmt 위임 (Standard) | `standard_format.rs:76,139,158,195,217,304,353,366,394` | `KVCacheOps::current_pos`(7) / `get_view`(2) | needs-rewire (inner inherent) | 5-E (E3) |
| fmt 위임 (KIVI) | `kivi_format.rs:72,76,119-124,168-169` | `current_pos`(3)/`capacity`(2)/`layout`(1)/`get_view`(2) | needs-rewire | 5-E (E3) |
| fmt 위임 (eval bridge) | `fmt_bridge.rs:129,137` | `current_pos`/`needs_attn_scores`(KiviCache) | needs-rewire (①-c 잔여 정식 해소) | 5-E (E3) |
| OLD generic forward | `transformer.rs:1509` `forward_into<C>` / `transformer_layer.rs` `forward<C>` / `forward_gen.rs:24` / `forward.rs:41` / `plan.rs:1277` `execute<C>` | generic 경로 (fmt fork가 live) | dies-with-legacy | 5-F (F2) |
| OLD-chain 잔여 소비자 (offload) | `offload_forward.rs:157,191` / `transformer.rs:3717` `forward_into_offload` / `chat/session.rs:547` | concrete OffloadKVCache, OLD layer chain 소비 | migrate-fmt (Option B) | **5-B** |
| OLD-chain 잔여 소비자 (KIVI) | `kivi_forward.rs:157,188` | `forward_into<KiviCache>` (★roadmap 누락) | migrate-fmt | **5-C** |
| OLD-chain 잔여 소비자 (chunked prefill) | `prefill.rs:348,446` | `forward_into<KVCache>` (profiler/variance) | migrate-fmt | **5-D** |
| 게이트 OFF fallback | `model_forward.rs:429,589` | `LLMRS_KV_FMT` OFF → forward_into\<C\> | dies-with-legacy (게이트 상수화) | 5-F (F2) |
| 비-forward 잡소비 (swap) | `swap_handler.rs:302` | `cache.ensure_capacity()` (**KVCache**, ★렌즈2 정정: OffloadKVCache 아님) | needs-rewire | 5-E (E4) |
| 비-forward 잡소비 (batch) | `batch/runner.rs:726` | `c.current_pos()` (Vec\<KVCache\>) | needs-rewire | 5-E (E4) |
| test (별도 파일) | `test_eng_dat_012_031.rs` / `test_eng_alg_020_022.rs` / `test_backend.rs` / `test_action_pool.rs` | concrete 메서드/필드 | needs-rewire / removable | 5-E (E4) |
| **test (impl 파일 내, ★렌즈1 B1 + 렌즈2 BLOCKING-1 누락분)** | `offload.rs:701` `compare_views<A,B>` (generic) + `offload.rs:1558` / `kivi_cache.rs:2929` (inline test) | **generic 헬퍼** + concrete test | needs-rewire (generic → 단형화 또는 test-local trait) | **5-E (E5 신설)** |
| 재export (★렌즈1 W1 누락분) | `kv_cache.rs:8-10` `pub use ...KVCacheOps` | 재export 라인 (test 4종이 이 경로로 import) | removable (동반 삭제) | 5-F (F3) |
| 주석 참조 | `eval/hook.rs:22` / `eval_loop.rs:5` / `layer_boundary_hook.rs:7` / `format.rs:12` / `ppl/runner.rs:256` | 주석 또는 미사용 import | dies-with-legacy / removable | 5-F / 5-E |
| 문자열 false-positive (★렌즈2 추가확인) | `transformer.rs:1843` `.expect("KVCacheOps::get_buffers_mut...")` | 문자열 리터럴 (코드 아님) | grep 게이트에서 제외 명시 | 5-E 게이트 정의 |
| 잔존 타입 (★렌즈1 W3) | `KVLayout`(151 refs) / `KiviRawBuffers`(10 refs) | trait 무관, inherent 시그니처가 계속 사용 | 잔존 (파일 운명 §5.3) | 5-F (F3) |

**검증되지 않은 추측 명시**: "97 참조"는 census grep 집계값이며, 위 표 행 수와 1:1 대응하지 않는다(한 행이 다수 호출 포함). 5-E 착수 시 `grep -rn "KVCacheOps" engine/`로 재집계 필요.

---

## 3. 핵심 설계: KVCache inherent-primitives 전환 전략 (메서드별 목적지)

### 3.1 전환 본질 (적대검증 렌즈1 W2 반영 정정)

**[정정] KVCacheFormat base trait이 모든 위임을 흡수하지 않는다.** fmt 래퍼가 위임하는 4개 메서드 중 `layout`(kivi_format:120)과 `get_view`(standard:366/394, kivi:124/168)는 **KVCacheFormat base trait(7 메서드: idx/current_pos/capacity/write_kv/write_kv_batch/compact/attention_into)에 부재**한다. 즉 base trait은 write/attention/geometry(current_pos/capacity)만 흡수하고, **layout/get_view는 반드시 inner cache의 inherent fn으로 rewire되어야 한다.** §0[F2]/ADR 정합의 "이미 충족됨" 서술은 layout/get_view에 대해서는 거짓이며, 본 설계는 이를 inherent 신설로 다룬다(아래 표).

분류: **[INH]** inherent 이미 존재(thin-forward) → impl 본문 그대로. **[T-ONLY]** trait에만 존재 → 5-E에서 inherent 신설. **[FIELD]** pub 필드. **[DEAD]** default no-op, 해당 cache 미구현 → 삭제.

### 3.2 KVCache (StandardFormat가 소비)

| 메서드 | 현 상태 | 목적지 | 비고 |
|---|---|---|---|
| `current_pos` | [FIELD] pub `self.current_pos` | inherent `fn current_pos(&self)` 신설 | StandardFormat이 `KVCacheOps::current_pos(&cache)` 호출 중 → inherent fn 신설이 rewire 최소 |
| `set_current_pos` | [T-ONLY] | inherent fn 신설 (pos 대입 + pos==0 high_water reset) | compact에서 사용 |
| `capacity`/`kv_heads`/`head_dim`/`layout`/`update`/`memory_usage_bytes` | [INH] 존재 | 그대로 | trait이 inherent 재호출 |
| `kv_dtype` | [T-ONLY] = `self.k_buffer.dtype()` | inherent fn 신설 | write_inner |
| `get_view` | [T-ONLY] `&mut self` — **inherent `get_view(&self, seq_len)`(kv_cache.rs:534)와 동명·다른 시그니처** | **신규 이름 `view(&self)` 로 신설 (★렌즈2 NON-BLOCKING-1 / R3)** | 같은 impl 블록 중복 정의 컴파일 에러 회피. 기존 `get_view(&self,seq_len)` caller는 5-E 착수 전 grep 확정 후 무변 |
| `get_buffers_mut`/`advance_pos`/`ensure_capacity` | [T-ONLY] | inherent fn 신설 | write_inner GPU scatter / plan_advance |
| KIVI 5종 + needs_attn_scores/set_attn_scores/get_kivi_raw_buffers | [DEAD] default no-op | 삭제 (KVCache 미사용) | |

### 3.3 KiviCache (KIVIFormat + fmt_bridge가 소비) — 가장 무거움

KiviCache는 inherent가 거의 전무(`bits`/`reset`/`set_awqe_enabled`만). 5-E에서 ~18개 inherent fn 신설이 작업량의 대부분이나, 본문은 KVCacheOps impl에서 **기계적 복사**(byte-identical diff 확인) — 위험 낮음.

| 메서드 | 현 상태 | 목적지 | 비고 |
|---|---|---|---|
| `current_pos` | [T-ONLY] = `self.total_tokens()` (**private**, kivi_cache.rs:423) | **pub fn `total_tokens()`** + inherent `current_pos` | fmt_bridge ①-c 수용 잔여 정식 해소점 |
| `needs_attn_scores` | [T-ONLY] = `self.awqe_enabled` (**private field**, kivi_cache.rs:247) | **pub fn `is_awqe_enabled()`** + inherent | fmt_bridge `needs_scores` 위임 해소 |
| `capacity`/`layout` | [T-ONLY] 조건부(bits=16 GPU⇒HeadMajor) | inherent fn 신설 (**조건부 로직 기계적 복사 필수**, R2) | |
| `kv_heads`/`head_dim`/`kv_dtype`/`memory_usage_bytes`/`update`/`get_view`/`get_buffers_mut`/`advance_pos`/`set_attn_scores`/`set_current_pos` | [T-ONLY] | inherent fn 신설 (`get_view`는 `&mut` 유지=assemble 필요) | |
| `res_pos`/`q2_tokens` | [FIELD] pub | 그대로 (필드 직접) | plan execute seam |
| `res_cap`/`needs_flush`/`flush_if_needed`/`get_kivi_raw_buffers` | [T-ONLY] | inherent fn 신설 | |

### 3.4 OffloadKVCache (forward_into_offload_fmt가 소비)

13개 override 전부 inherent fn 신설 + **`impl KVCacheFormat for OffloadKVCache`(또는 `OffloadFormat` wrapper) 신설**(5-B). 상세는 §4. swap_handler의 `ensure_capacity`는 **KVCache 대상**(렌즈2 정정)이라 여기 무관.

### 3.5 비-forward 잡소비자 rewire (5-E)

| 위치 | 호출 | 대상 타입 | rewire |
|---|---|---|---|
| `swap_handler.rs:302` | `cache.ensure_capacity()` | **KVCache** (★렌즈2 정정, OffloadKVCache 아님 — recall_one:271 `&mut KVCache`) | KVCache inherent 호출 |
| `batch/runner.rs:726` | `c.current_pos()` | Vec\<KVCache\> | KVCache inherent 호출 |
| `fmt_bridge.rs:129,137` | `KVCacheOps::current_pos/needs_attn_scores` | KiviCache | `g.total_tokens()` / `g.is_awqe_enabled()` |
| `offload.rs:701` `compare_views<A,B>` | generic 2-타입 헬퍼 (★렌즈1 B1) | KVCache + OffloadKVCache | **단형화** `compare_views(base: &mut KVCache, offload: &mut OffloadKVCache, ...)` (호출처 offload.rs:759가 단일 조합뿐) |
| `offload.rs:1558` / `kivi_cache.rs:2929` (inline test/bench) | `KVCacheOps::get_view` | KVCache | inherent `view()` 호출 (★렌즈2 BLOCKING-1) |
| test 별도 파일 3종 + ppl runner | 메서드/필드 혼합 | concrete | inherent 호출 + `use` 제거 |

---

## 4. offload Option B fmt 이주 (5-B) + device 게이트

### 4.1 단순 "OffloadKVCache: KVCacheFormat"은 성립하지 않음 — Option B′(하이브리드) 채택

forward_into_offload는 forward_into_fmt와 **본문이 완전히 다른 별개 forward**(preload pool + adaptive prefetch depth + cross-token retain/release + 자체 layer loop)다. lifecycle 메서드(preload/release_buffers/retain_preload)는 KVCacheFormat 표면에 없고 전부 `&mut self`다. 따라서:

- **재사용**: `forward_gen_fmt`/`forward_prefill_fmt`의 layer 본문(write_kv + attention_into 위임 두 지점만 fmt).
- **재사용 불가**: forward_into_fmt의 layer loop(preload pool 없음) → **forward_into_offload의 loop 골격은 보존**, 내부 `layer.forward(...)`만 `layer.forward_gen_fmt`/`forward_prefill_fmt`로 교체.
- **preload pool aliasing은 유지** — preload는 KVCacheFormat이 아니라 `PrefetchableCache`(Step 2 standalone) 경로로 계속 concrete monomorphization.

**Option B′ 정의 = "forward_into_offload는 살아남되 generic `<C: KVCacheOps>` layer chain 의존만 끊는다."** 이것이 roadmap §Step 5(b)의 정확한 달성 방법.

### 4.2 interior-mut 후보 A — `Mutex<OffloadKVCache>` wrapper (`OffloadFormat`)

```
// pressure/offload_format.rs (no-mod.rs, 형제 offload.rs 옆)
pub struct OffloadFormat { idx: usize, inner: Mutex<OffloadKVCache> }
impl KVCacheFormat for OffloadFormat {
    fn write_kv(&self, ...)      -> { self.inner.lock().unwrap().update(...) }       // &mut OK (guard)
    fn attention_into(&self, ...) -> { let mut g=self.inner.lock().unwrap();
                                       let (kc,vc)=g.view_mut(); be.attention_gen(...) }
    fn current_pos(&self) -> usize { self.inner.lock().unwrap().current_pos() }
    fn capacity(&self) -> usize    { ... }
    fn compact(...)      -> Result<()> { bail!("offload: eviction 미지원") } // on_kv_prune no-op 일치
}
// preload/release/retain 는 KVCacheFormat 밖 → OffloadFormat::preload(&self){inner.lock().preload()}
// + impl PrefetchableCache for OffloadFormat
```

- 장점: OffloadKVCache 본문 무변. update/get_view의 `&mut self`가 lock guard 안에서 성립.
- C4(검증됨): `OffloadStore: Send` → `Mutex<OffloadKVCache>: Sync`(T: Send면 충분). b-0 컴파일이 강제.

**구현 경로 결정 (★렌즈2 BLOCKING-2 반영)**: 5-B 시점에 OffloadKVCache는 current_pos/update/get_view의 inherent가 **없고 KVCacheOps impl(263)만** 있다. OffloadFormat이 inner를 부르는 방식 2경로:
- **(경로 X, 권장)** 5-B에서 OffloadFormat이 inner 호출 시 **트레이트 잔존을 활용**(`KVCacheOps::current_pos(&*g)`). Step 1~3의 standard_format.rs가 정확히 이 패턴이고 5-E까지 trait가 살아있으므로 합법. OffloadKVCache inherent 신설은 5-E로 미루고, 5-E에서 OffloadFormat의 `KVCacheOps::` 호출도 inner inherent로 rewire. → 5-B↔5-E 의존은 "5-E가 5-B의 OffloadFormat 코드를 rewire 대상으로 포함"(역방향, 정상).
- (경로 Y) 5-B 안에서 OffloadKVCache inherent 선신설 후 OffloadFormat이 inherent 직호출. 5-E의 OffloadKVCache 항목이 5-B로 흡수.

land 순서 `5-B → 5-E`는 두 경로 모두 유지. **(★렌즈2 정정)** D4 §3/§4의 "swap_handler ensure_capacity = OffloadKVCache" 오기는 제거 — 실제 KVCache이며 5-B와 무관.

### 4.3 preload aliasing 안전성 (C2/C3 검증 결과)

- `preload_erased::<C>`가 `*(ptr as *mut C)` concrete cast(preload_pool.rs:177). `Vec<OffloadFormat>` 전환 시 caches_ptr 타입 + 제네릭 인자를 **OffloadFormat으로 동시 치환**(불일치 시 raw cast가 UB silent 통과) → b-1 host test에서 preload→forward round-trip 실검증 필수.
- 안전 불변식 `far_idx = i + depth(≥1) ≠ i`(transformer.rs:3797) 보존 시 **Mutex lock 경합 0**(서로 다른 인스턴스 = 서로 다른 Mutex). Mutex가 오히려 raw-ptr Send 안전성 강화.
- ★cross-token retain(retain_preload, 3893): main thread가 layer i 만진 뒤 다음 토큰에서 같은 layer가 preload 대상 — 토큰 간 sequential이라 안전하나 interior-mut 전환 시 가장 미묘 → 게이트에서 multi-token 발화 필수(G3).

### 4.4 build_chat_offload / OffloadForward 영향

`alloc_offload_kv_caches`(offload_forward.rs:277) 반환 `Vec<OffloadKVCache>` → `Vec<OffloadFormat>`. 내부에서 `OffloadKVCache::new` 후 wrap, `set_gpu_backend`는 wrap 전 inner에 호출. `OffloadForward.kv_caches`/`reset_kv`(224)/`kv_caches_mut`(105) 시그니처 따라감. build_chat_offload(session.rs:547) 자체는 alloc 시그니처만 따라가면 무변.

### 4.5 이주 substep (additive + 게이트)

| substep | 작업 | 검증 |
|---|---|---|
| b-0 | `OffloadFormat` 신설(pressure/offload_format.rs). KVCacheFormat + PrefetchableCache impl. **purely additive, unwired** | host unit: write_kv→attention_into round-trip == OffloadKVCache 직접 (CPU bit-identical). compact bail |
| b-1 | `forward_into_offload_fmt` **copy-fork**(transformer.rs). loop 골격 보존, `layer.forward`→`forward_gen_fmt`/`forward_prefill_fmt`. `preload_erased::<OffloadFormat>`. env 게이트 `LLMRS_OFFLOAD_FMT` OFF default | host CPU: `--kv-mode offload --kv-type f16` greedy n=32 ON vs OFF md5 동일 |
| b-2 | OffloadForward/alloc/build_chat_offload를 `Vec<OffloadFormat>` 전환 (게이트 ON 경로) | host: chat offload smoke + reset_kv |
| b-3 (device) | **device 게이트(§4.6)** 통과 후 게이트 제거 + 기존 forward_into_offload + `impl KVCacheOps for OffloadKVCache` 삭제 | S25/Jetson bit-identical + TBT |

### 4.6 device GPU 재검증 게이트 (★렌즈3 BL-1·BL-2 + G1~G4 전면 반영)

**[BL-1 — 매체 영속성, hard 선결]**: argus_cli가 offload를 reject → offload device 발화 매체 = `legacy_generate`(5-F 폐기 대상) 또는 build_chat_offload(family bin 미구현). **Step 3 함정보다 심각**: offload GPU 경로(get_view:437 gpu_backend 분기)는 host CpuBackend에서 `is_gpu()=false`라 **단 1줄도 실행 안 됨**(C1) + argus가 reject라 baseline 매체 부재. 따라서:
- **argus-bench(offload 지원)를 5-B의 hard 선결로 격상** 또는 **legacy_generate를 offload 한정으로 5-F에서 부분 제외** 중 택1을 5-A에서 명시.
- frozen baseline은 **legacy의 offload GPU 출력**으로 5-B 이전에 S25+Jetson 각각 캡처(argus OFF는 offload reject라 baseline 불가).
- **5-F 진입 gate에 "offload device 매체 영속성"을 추가**: 5-F = (5-E grep 0건) ∧ (offload device 매체가 argus-bench/chat로 확보됨). 미확보 시 5-F 이후 offload regression 재현 영구 불가.

**[BL-2 — Mutex poisoning 신규 failure mode]**: OffloadFormat::attention_into의 lock guard 안에서 get_view가 GPU 버퍼 alloc(offload.rs:447/452 `.expect()`)을 수행 → alloc 실패 시 panic이 **Mutex poisoning** 유발(legacy `&mut self`엔 없던 신규 failure mode). StandardFormat 선례는 guard 내 panic 경로가 없어 **선례 비대칭**. host CpuBackend는 GPU 분기 미진입 → 이 경로 0 커버. 게이트 명시 항목:
- VRAM 압박(max_seq × token_bytes × 2 × num_layers) 하 alloc 실패가 device에서 발생 가능한지 — 단순 bit-identical로 미검출.
- `lock().unwrap()` poisoning 복원 전략(또는 panic=abort 의존) 명시. 이 비대칭을 5-B 설계에 박을 것.

**게이트 정의**:
- 장비: S25 `--backend opencl --opencl-rpcmem` + Jetson `--backend cuda`.
- 모델: Qwen2.5-1.5B, `--kv-mode offload`, **F16·F32 둘 다**(★G1: C1 dtype 무관 발화, upload 크기만 상이 — 한쪽만 검증 시 다른 dtype upload 회귀 미검출). raw 모드 필수 + disk smoke 선택.
- bit-identical: `LLMRS_OFFLOAD_FMT=1`(ON) vs unset(OFF) greedy(temp=0). **첫 토큰 logit + 텍스트 완전 일치**.
  - ★G2 prefill arm: forward_into_offload prefill(seq_len>1)은 get_view가 store 전체→GPU upload total_bytes로 decode와 패턴 상이 → **prompt ≥64 토큰 포함하여 GPU upload 전체 경로 발화**.
  - ★G3 retain: depth≥1로 retain 항상 발화하나 **32-tok decode + depth < num_layers 구성**(retain된 layer가 다음 토큰 deferred write로 가는 경로)으로 검증.
- avg_tbt: ON vs OFF Δ ≤ +3%, n≥5 median, wall-clock(`--profile` 금지). ★G4: **raw 모드가 worst-case**(in-memory store라 disk I/O 없어 lock 2회/layer가 상대적으로 더 드러남) → raw를 disk와 분리 측정. Step 3 +0.24%는 plan(GPU 지배)이라 직접 전이 불가.
- device-only 필수(host 미발화): (1) get_view의 GPU upload가 lock guard 내 정상, (2) preload worker + main attention_into Mutex deadlock 부재, (3) prefill_attention GPU flash가 offload SeqMajor + device-resident KV 호환, (4) BL-2 poisoning 미발생.
- 회귀 시: `LLMRS_OFFLOAD_FMT` OFF default라 production 무영향 → b-3 device 게이트만 revert(나머지 fmt 이주 유지).

---

## 5. 비-happy 모드 거취 + 권고 + legacy 폐기 hard blocker

### 5.1 모드 거취 표

★census 정정: 영역2가 eval/ppl/batch/dump/qcf/warmup을 "OLD-chain 소비자"로 분류했으나, **Step 1(①-c/①-d/①-e)에서 전부 `forward_into_fmt`로 flip 완료** — 이들은 OLD generic을 더 이상 소비하지 않는다. 잔여 `use KVCacheOps`는 concrete `current_pos()` + import 잔재일 뿐. **거취 ≠ KVCacheOps 의존**: 거의 모든 모드 진입 함수가 `session/` 공유 모듈로 추출되어 fmt-clean.

| 모드 | 실로직 위치 | OLD-chain 의존 | 이주 비용 | 권고 |
|---|---|---|---|---|
| eval-ll / ppl / batch / dump / qcf | `eval,ppl,batch,dump_importance,qcf_runtime` (fmt flip ✅) | **미의존** | 낮음 (배선만) | **argus-eval 묶음 이주** |
| prompt-batch | `batch/runner.rs` (fmt ✅) | 미의존 | 낮음 | argus-eval |
| eviction (sliding/h2o/d2o/snapkv) | `decode_fallback/eviction_trigger` + `pressure/` + (3c-evict) compact | 미의존(forward fmt 가능, eviction은 후처리) | 中 (resilience hook 배선) | **argus-bench 이주** (verify + paper 핵심) |
| weight swap (8종) | `decode_fallback/swap_dispatch` + `swap_runtime` | **미의존**(weight mmap만, forward 무호출) | 中 | **argus-bench 이주** (paper perf 핵심) |
| **offload** | `offload_forward.rs` → forward_into_offload | **★OLD-chain hard 의존** | **높음** (Option B + device) | **이주 (5-B)** — drop 시 KVSwap 연구축 소멸 |
| **KIVI** (KiviForward 추론) | `kivi_forward.rs:157/188` | **★혼재**: ppl/eval=fmt-clean ✅, **KiviForward=OLD** | 中~高 | **이주 (5-C)** = Step 5 작업과 중첩 |
| profile | `prefill.rs` profiler | 간접(run_chunked_prefill OLD) | 中 | 이주 (argus-bench, 우선순위 낮음) |
| chat | `chat/repl.rs` + `chat/session.rs` | 혼재(Standard=fmt, KIVI/Offload chat=OLD) | 中 | **이주(argus-chat) 또는 drop** (데모용, 측정 무관) |
| tensor-partition | main inline (**미추출**) | 간접(partition 분기 dead in fmt) | **높음** (session 모듈 부재) | **defer** (별도 추출 라운드) |

### 5.2 권고 (우선순위)

1. **저비용 (fmt-clean, 배선만)**: eval-ll/ppl/dump/qcf/batch → **argus-eval** 1개 bin. 로직 이미 session/에 존재.
2. **중비용 (모듈 존재, decode-loop/resilience 배선)**: eviction + weight swap → **argus-bench**. verify 하네스 + paper perf 측정 매체.
3. **고비용 (forward fmt 재이주 + device)**: offload(5-B) + KIVI(5-C) — "모드 이주"가 아니라 **KVCacheOps 삭제 자체의 선결**.
4. **drop 후보**: chat(데모, paper 미사용) — argus-chat 보류 가능.
5. **미추출 = defer**: tensor-partition.

**거취 결정 = A1(session 함수 라이브러리 보존) 권고**: 사용자 "legacy disposable"은 *bin* 폐기지 *기능* 폐기가 아님. KIVI/offload/eval은 라이브러리 API로 생존하고 미래 family bin의 backend. **이 선택이 5-B/5-C/5-D를 필수로 확정**한다(A2 drop이면 fmt 이주가 "삭제"로 대체되어 범위 급감하나 기능 손실). 미해결: §7-(1).

### 5.3 legacy 폐기 hard blocker (3건)

legacy/generate.rs는 main() dispatcher일 뿐 실로직은 `session/`에 산다. 폐기를 막는 진짜 blocker:

- **HB-1 (compile, KVCacheOps 삭제 차단)**: offload forward OLD-chain — `forward_into_offload` + `impl KVCacheOps for OffloadKVCache` + OffloadForward + chat/session.rs:547. legacy bin과 무관하게 `session/`에 남아 trait 소비 → **5-B로 해소**(device 필수).
- **HB-2 (compile, KVCacheOps 삭제 차단)**: KiviForward OLD-chain — `kivi_forward.rs:157/188`이 `forward_into<C>` 소비. ppl/eval KIVI는 fmt-clean이나 추론 KiviForward 미이주 → **5-C로 해소**.
- **HB-3 (검증 blocker, non-compile)**: device 게이트 매체. run_device.py 게이트 bin = `legacy_generate`. argus_cli는 happy-path 전용 → eviction/swap/KIVI/offload/partition device 게이트는 legacy 필요. **★verify.py:99가 `generate` bin을 빌드하는데 이 bin은 Cargo.toml에 없다 → verify는 이미 stale**(2026-04-23 마지막 results). legacy 폐기 전 verify를 argus-bench/eval로 재배선하거나 verify 폐기를 명시 결정해야 함.

**hard blocker 본질**:
- **compile hard blocker (KVCacheOps 삭제 차단) = HB-1 + HB-2.** legacy bin 삭제로 해소 안 됨(non-legacy session/ 소비자). = Step 5(b) 그 자체.
- **legacy bin 삭제(5-F의 F1)만의 blocker = HB-3.** eviction/swap/partition 측정 bin이 argus_cli에 없음 → **argus-bench로 eviction+swap(+offload device 매체) 이주가 legacy bin 삭제의 실질 선결**.

---

## 6. sub-increment 시퀀싱 (6 증분 + 게이트 + 의존 + 비가역 안전장치)

원칙 = Step 1~3 선례(additive + 게이트 + 증분별 device 게이트 + 회귀 시 격리 revert). 비가역 cutover(5-F)는 맨 마지막에 격리, 그 앞은 전부 additive·revert-safe.

### 6.1 증분 목록

| # | 제목 | 범위 | 게이트 | 의존 |
|---|---|---|---|---|
| **5-A** | 비-happy 모드 거취 결정 (코드 0) | A1(session 보존)/A2(drop)/A3(family bin) 중 택1. **이 결정이 나머지 5 증분 범위를 잠금**. **★offload device 매체(argus-bench)를 5-B 이전 확보할지 결정**(BL-1) | 없음 (결정 문서 = ADR-0001 보강 + roadmap 갱신). PM/Architect | 없음 (첫 게이트) |
| **5-C** | KiviForward fmt 이주 (roadmap 누락 보강) | `kivi_forward.rs:157/188` `forward_into<KiviCache>` → forward_into_fmt (①-c/①-e 선례). KIVIFormat prefill arm은 ①-e 기존 | host: build + KIVI test. `--kv-mode kivi` n=32 BEFORE/AFTER bit-identical + flush 정수회계 일치. **device 권장(KIVI GPU bits16 첫 검증, host-only land 가능)** | 5-A=A1 |
| **5-D** | run_chunked_prefill fmt 이주 | `prefill.rs:348/446` → forward_into_fmt (①-d 동형, EvalCacheKind round-trip). profiler/variance cold | host: build + clippy/fmt. NLL 정밀 bit-identical. **device 불요**(prefill은 plan 무관, ①-b 선례) | 5-A=A1 |
| **5-B** | offload Option B fmt 이주 (★device 필수) | §4. b-0~b-3. preload pool aliasing 재설계 = Step 5 최난도 | §4.6 (host bit-identical → S25/Jetson bit-identical + Δ≤+3%) | 5-A=A1 (+ BL-1 매체) |
| **5-E** | inherent 전환 + rewire (★additive, trait 미삭제) | §3 (E1 KVCache inherent / E2 KiviCache pub / E3 fmt rewire / E4 잡소비 + test 별도파일 / **E5 ★offload.rs:701 compare_views 단형화 + offload.rs:1558·kivi_cache.rs:2929 inline test**) | host: build + `cargo test --workspace` 전체 pass + fmt/clippy clean. **grep 게이트(§6.2)**. device 불요(동작 불변) | 5-B·5-C·5-D 완료 |
| **5-F** | legacy 폐기 + trait 삭제 + rename (★비가역) | F1 legacy bin/파일 + F2 OLD chain(forward_into\<C\>/forward\<C\>/forward_gen\<C\>/forward_prefill\<C\>/execute\<C\> + TransformerModelForwardArgs\<C\> + impl×3) + **F3 kv_cache_ops.rs 삭제 + ★kv_cache.rs:8-10 재export 블록 삭제**(W1) + F4 rename(별도 chore commit) | host: `grep KVCacheOps engine/` 0건 + build + test pass + clippy. **device-gate(full)**: 5 KV × 32-tok bit-identical + avg_tbt Δ≤+3% (S25+Jetson, frozen baseline) | 5-A~5-E 전부 + **offload 매체 영속성**(BL-1) |

**land 순서 (의존 준수)**: `5-A → 5-C ∥ 5-D → 5-B(device) → 5-E → 5-F(device, 비가역)`
**위험 순서 (난도)**: 5-A(無) < 5-D(낮음) < 5-C(낮음~중) < 5-E(중, 26파일 광범위) < 5-B(높음, preload aliasing+device) < 5-F(높음, 비가역).

### 6.2 5-E grep 게이트 표현식 (★렌즈2 BLOCKING-1 + 추가확인 반영)

5-F 진입 = 5-E 종료. 다음을 **모두 0건**(주석·문자열 제외 `KVCacheOps::` 호출):
- standard_format.rs(9) / kivi_format.rs(8) / fmt_bridge.rs(2) — E3
- swap_handler.rs:302 / batch/runner.rs:726 — E4
- **★offload.rs:701(compare_views generic) + offload.rs:1558 + kivi_cache.rs:2929** — E5 (누락분, impl 파일 내 test도 0건이어야 함)
- test 별도 4종 — E4
- **제외**: `transformer.rs:1843` `.expect("KVCacheOps::...")` 문자열 리터럴(컴파일 무해, false-positive) → 게이트 표현식을 "주석·문자열 제외 호출 0건"으로 정의.

### 6.3 비가역 단계(5-F) 안전장치 (4중)

1. **차단자 0 선확인(5-E 전담)**: 5-F의 trait/impl 삭제 전 §6.2 grep 게이트로 호출 0건 실측 → 5-F 컴파일 실패 위험이 `<C: KVCacheOps>` bound 제거 + `use` 제거로만 한정(기계적).
2. **격리 commit + 즉시 revert**: 5-F는 5-E 위 단일 commit. device 게이트 회귀 시 `git revert 5-F` 1회로 5-E 상태(컴파일·동작 무변, trait 잔존, 모든 호출 이미 inherent) 복귀. 5-E 자체는 additive라 revert 불요.
3. **frozen baseline**: BC 진입 commit의 출력을 동결. legacy가 5-F에서 삭제되므로 **baseline은 5-F 이전 캡처 필수**(5-E 시점). argus OFF 출력도 병행(Step 4가 argus≡legacy 등가 증명). **단 offload는 argus reject라 legacy offload 출력을 5-B 이전 별도 캡처**(BL-1).
4. **rename 분리(F4)**: `KVCacheFormat` rename은 trait 삭제(F2/F3)와 별도 chore commit(mechanical sed) — 삭제 회귀 분석 오염 방지. device 게이트는 F1~F3에서 통과 후 F4는 host build/test만.

**비가역성 본질**: 5-F가 비가역인 건 "코드 삭제"이지 "동작 변경"이 아니다 — 5-E까지 동작은 이미 fmt 단일 경로로 수렴(production 게이트 ON화는 Step 3/4 device 검증). 따라서 5-F device 게이트의 위험은 신규 회귀가 아니라 **parallel path 제거 후 monomorphization perf 노출**(avg_tbt)이 유일. bit-identical은 5-E까지의 fmt 경로가 이미 증명. **단 offload(5-B)는 예외** — host 미발화 GPU 경로라 bit-identical을 5-B device 게이트에서만 증명 가능.

---

## 7. Landmines / 미해결

**미해결 결정사항 (사용자/PM 확인 필요)**:
1. **[5-A 핵심] KIVI/offload/swap/batch의 production 진입점 거취**: A1(session 보존, fmt 이주 강제) vs A2(drop, 5-B/5-C/5-D가 "삭제"로 축소 + R1=180 소멸 + 기능 손실) vs A3(family bin 신설, 범위 폭발). **이 결정이 Step 5 난이도를 좌우.** A2 drop이면 Step 5가 순수 host rewire로 축소.
2. **[BL-1] offload device 매체 영속성**: argus-bench(offload 지원)를 5-B hard 선결로 확보 vs legacy_generate를 offload 한정 5-F 부분 제외. 5-A에서 명시 필요.
3. **[HB-3] verify 하네스**: verify.py:99의 stale `generate` bin → argus-bench/eval 재배선 vs verify 폐기. legacy 폐기 전 결정.
4. **[§5.3 W3] `kv_cache_ops.rs` 파일 운명**: KVCacheOps만 삭제(파일·import 유지) vs `format/kv_layout.rs` rename(KVLayout 151 + KiviRawBuffers 10 = 최대 161 참조 import 변경). ADR 정합은 둘 다 충족. **최소 변경 = 파일명 유지**(KVCacheOps trait + 재export만 삭제, KVLayout/KiviRawBuffers 정의 잔존) 강하게 권고.

**Landmines (구현 중 함정)**:
- **get_view 이름 충돌**(R3): KVCache inherent `get_view(&self,seq_len)`(534) ↔ trait `get_view(&mut)`(1046). 5-E에서 trait 변형은 **반드시 다른 이름 `view`로 신설**. 기존 `get_view(&self,seq_len)` caller는 5-E 착수 전 grep 확정 후 무변.
- **KiviCache 조건부 로직**(R2): bits=16 GPU⇒HeadMajor capacity/layout 분기 → inherent 복사 시 **byte-identical diff 확인**. KVCacheOps impl을 5-E에서 즉시 삭제하지 말고 5-F까지 잔존시켜 "impl이 inherent 호출" 형태로 두면 5-E 단독 회귀 0.
- **preload raw cast UB**(C2/R4): `*mut OffloadKVCache`↔`*mut OffloadFormat` 타입 불일치가 raw cast로 silent 통과 가능 → b-1에서 caches_ptr + preload_erased 제네릭 인자 **동시 치환** + host round-trip test.
- **Mutex poisoning**(BL-2): offload guard 내 GPU alloc panic이 신규 failure mode(StandardFormat 선례 비대칭). device 게이트 명시 항목.
- **검증 안 된 추측**: (i) run_chunked_prefill의 실제 호출처가 legacy/generate.rs:1848 단독인지(nonhappy census 주장) — 5-D 착수 전 재확인. standard happy는 decode_loop.prefill()(ModelForward) 사용으로 chunked 미사용이라는 census 주장이 맞으면 5-D는 cold-only. (ii) "97 참조"는 grep 집계값, 표 행과 1:1 아님 — 5-E 착수 시 재집계.

---

## 8. 적대검증 요약 (반영한 수정)

| 렌즈 | 이슈 | 등급 | 본 설계 반영 |
|---|---|---|---|
| 1 | **B1**: `offload.rs:701 compare_views<A,B>` generic test 헬퍼 누락 (갈래 A 유일 컴파일 차단 예외) | BLOCKING | §3.5 / §6.1 5-E에 **E5 신설**(단형화), §6.2 grep 게이트에 포함 |
| 1 | W1: `kv_cache.rs:8-10` 재export 라인 누락 (test 4종이 이 경로 import) | non-blocking | §6.1 5-F **F3에 재export 블록 삭제 명시** |
| 1 | W2: "KVCacheFormat 흡수안"이 layout/get_view 미커버 | non-blocking | §3.1에 **정정 명시** — layout/get_view는 base trait 부재, inherent rewire 필수 |
| 1 | W3: KVLayout 151 / KiviRawBuffers 10 refs 파일 운명 실비용 | non-blocking | §7-(4) **파일명 유지 강하게 권고** |
| 2 | **BLOCKING-1**: 5-E rewire에 impl 파일 내 test 2곳(kivi_cache:2929, offload:1558) 누락 → grep-0 게이트 위반 | BLOCKING | §3.5 / §6.1 E5 / §6.2 grep 게이트 정밀화 |
| 2 | **BLOCKING-2**: D4 §3 "swap_handler ensure_capacity=OffloadKVCache" 오기 (실제 KVCache) → 5-B↔5-E 의존 근거 오류 | BLOCKING | §3.5 / §4.2 **정정**, 의존 근거를 경로 X/Y로 재정의 |
| 2 | NON-BLOCKING-1: get_view 이름 충돌 명문화 | non-blocking | §3.2 / §7 Landmine **"다른 이름 `view` 강제"** |
| 2 | 추가: transformer.rs:1843 문자열 false-positive | — | §6.2 grep 게이트 "문자열 제외" 정의 |
| 2 | 순환 의존·5-F 차단자 0·fmt args 분리 | 통과 | 골격 무수정 |
| 3 | **BL-1**: offload device 매체 부재 (Step 3 함정보다 심각, argus reject + host 미발화) | BLOCKING | §4.6 / §6.1 5-F 진입 gate에 **매체 영속성 추가**, argus-bench 5-B hard 선결 격상, baseline 5-B 이전 캡처 |
| 3 | **BL-2**: Mutex poisoning 신규 failure mode (guard 내 GPU alloc panic, 선례 비대칭) | BLOCKING | §4.6 **device 게이트 명시 검증 항목 추가** |
| 3 | G1: F16·F32 둘 다 게이트 (dtype 무관 발화) | 보강 | §4.6 게이트 정의 |
| 3 | G2: prefill arm ≥64 prompt 토큰 발화 | 보강 | §4.6 |
| 3 | G3: retain cross-token 32-tok + depth<num_layers | 보강 | §4.6 |
| 3 | G4: raw 모드 worst-case avg_tbt 분리 측정 | 보강 | §4.6 |

**적대검증으로 확인된 census 정확 항목** (무수정): dyn KVCacheOps = 0건(전부 monomorphization), fmt 경로(forward_gen_fmt/forward_prefill_fmt/execute_fmt) trait-clean(주석만), 매크로 생성 코드 없음, KVCacheFormat ⊥ KVCacheOps(supertrait 무관), OffloadStore: Send → Mutex<OffloadKVCache>: Sync 충족(C4), 안전 불변식 far_idx≠i가 Mutex 도입 후 보존(C3).

---

**관련 파일 (절대경로)**:
- SSOT: `/home/go/Workspace/llm_rs2/.agent/todos/roadmap_alpha_k_bc_completion_2026_06_04.md` §Step5
- 삭제 대상: `/home/go/Workspace/llm_rs2/engine/src/kv_cache_ops.rs`, `/home/go/Workspace/llm_rs2/engine/legacy/generate.rs`
- base trait(rename 무관, 유지): `/home/go/Workspace/llm_rs2/engine/src/format/kv_cache_format.rs:61-108`
- 5-B: `/home/go/Workspace/llm_rs2/engine/src/pressure/offload.rs:263-501,437-483,701,759,1558`, `/home/go/Workspace/llm_rs2/engine/src/pressure/offload/preload_pool.rs:177-180`, `/home/go/Workspace/llm_rs2/engine/src/models/transformer.rs:3717,3797-3900`, `/home/go/Workspace/llm_rs2/engine/src/session/forward/offload_forward.rs:157,191,277,300-302`, `/home/go/Workspace/llm_rs2/engine/src/session/chat/session.rs:547`, `/home/go/Workspace/llm_rs2/engine/src/pressure/offload/store.rs:9`
- 5-C: `/home/go/Workspace/llm_rs2/engine/src/session/forward/kivi_forward.rs:157,188`
- 5-D: `/home/go/Workspace/llm_rs2/engine/src/session/prefill.rs:348,446`
- 5-E inherent: `/home/go/Workspace/llm_rs2/engine/src/pressure/kv_cache.rs:534,1006,1046`, `/home/go/Workspace/llm_rs2/engine/src/pressure/kivi_cache.rs:247,423,2237,2929`
- 5-E rewire: `/home/go/Workspace/llm_rs2/engine/src/pressure/standard_format.rs:366,394`, `/home/go/Workspace/llm_rs2/engine/src/pressure/kivi_format.rs:120`, `/home/go/Workspace/llm_rs2/engine/src/session/eval/fmt_bridge.rs:129,137`, `/home/go/Workspace/llm_rs2/engine/src/pressure/swap_handler.rs:271,302`, `/home/go/Workspace/llm_rs2/engine/src/session/batch/runner.rs:726`
- 5-F OLD chain: `/home/go/Workspace/llm_rs2/engine/src/layers/transformer_layer/forward_gen.rs`, `forward.rs`, `/home/go/Workspace/llm_rs2/engine/src/backend/opencl/plan.rs:1277`, `/home/go/Workspace/llm_rs2/engine/src/session/forward/model_forward.rs:429,589`, `/home/go/Workspace/llm_rs2/engine/src/pressure/kv_cache.rs:8-10`(재export)
- 검증 매체: `/home/go/Workspace/llm_rs2/verify/verify.py:99`(stale generate bin), `/home/go/Workspace/llm_rs2/engine/src/bin/argus_cli.rs:158-207`(reject 목록)
- ADR: `/home/go/Workspace/llm_rs2/docs/adr/0001-kv-dispatch-paradigm.md:10,124`