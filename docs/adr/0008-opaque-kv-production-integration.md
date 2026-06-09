# ADR-0008: opaque KV format production 통합 — `KVCache` 흡수 + `is_q4`→descriptor-keyed (실추론 grow/eviction/D2O)

> **Status**: Accepted · **구현 완주 2026-06-08** (Stage 1 `6b4eaef5` / Stage 2 `d1b6cb55` / Stage 3 `114b9b12`; lib 1258/0 + clippy clean)
> **정정 2026-06-09** (`f0d92c66`): 위 "`--kv-format synth_q4` == `q4_0` token-identical" e2e 는 **false positive** 였다. `build_inference_ctx` 가 opaque 를 올바로 dispatch 했으나 `standard_happy.rs:75`/`experiment_run.rs:54` 가 `ctx.kv_caches` 를 drop 한 뒤 loop builder 가 typed 를 재할당 → **opaque 가 decode loop 에 도달 못 함**(둘 다 typed q4_0 → 자명한 동일). loop builder 가 `ctx.kv_caches` 를 소비하도록 수정(§4 e2e eviction 항목 참조). 수정 후 실측: opaque 는 decode loop 에 도달하며(`[DecodeLoop] kv storage = OPAQUE`) coherent, 단 native typed q4_0 와는 **token-identical 아님**(opaque f32 floor attention vs native q4_0 kernel — compute 경로 차이, 둘 다 valid). bit-identity 는 unit gate(`standard_format.rs`, F32 round-trip)가 담당.
> **Date**: 2026-06-08
> **Decision-makers**: 사용자 + 메인 세션 (ADR-0007 GATE-B 후속 "북극성 우선순위 grill" 세션; 3-방향 실측 매핑 + GATE-C host-testability 적대 검증 워크플로 — `wf_ebcad9c1`)
> **Selected**: GATE-B 의 격리된 `OpaqueKvFormat`(자체 capacity pre-size 저장)을 **production `KVCache` 에 흡수** — `KVCache` 가 `OpaqueBuffer`(U8 + `KVLayoutDesc` sidecar)를 1급 저장으로 받아들이고, `is_q4` 이진 byte-회계를 **descriptor-keyed** 로 일반화한다. 그 결과 opaque(DType 없는) format 이 **실제 추론 전 경로**(alloc → grow-on-demand → attention → eviction(sliding/h2o) → D2O merge)를 코드 변경 0으로 탄다. 범위 = **full**(Stage 1+2+3, D2O 포함). 검증 vehicle = 외부 `synth-q4-format` crate force-link(`--kv-format synth_q4`)가 `--kv-format q4_0`(typed)와 **bit-identical**.
> **Related**: ADR-0007(opaque-dtype 해금, GATE-B — 본 ADR 의 직전 단계), ADR-0004(D1 변형=엔진 독점, `compact` base-trait 금지 → 구조 (B) 기각 근거), ADR-0005(D3 format=descriptor-only, D5 generic floor, D6 3축 registry), `arch/pipeline_stage_design_v2.md` §4.1(`KVCacheFormat` 6-method base trait), `/CONTEXT.md`(format 축).

---

## 1. Context

ADR-0007 GATE-B 는 닫힌 `DType` enum 을 우회한 opaque KV format(`synth_q4`)이 **격리 host 환경**에서 write(encode+scatter)+attention(dequant floor→F32) 을 q4_0 round-trip 과 bit-identical 로 수행함을 증명했다(7 커밋, CPU end-to-end). 단 `OpaqueKvFormat`(`pressure/opaque_format.rs`)은 **자체 최소 저장**(capacity pre-size, grow/eviction 부재)이라 실제 추론에는 못 쓴다 — grow/eviction 은 production 단계로 deferred 였다.

"북극성(zero-compile `.so` plugin 설치) 우선순위 grill" 에서 세 후보(GATE-C / production KVCache 통합 / 다른 축)를 실측 매핑한 결과, 사용자는 **production KVCache 통합 먼저**를 선택했다 — "메커니즘(dlopen) 증명(GATE-C)보다, plugin format 이 **실제 워크로드를 돈다**를 먼저 확보". GATE-C(.so dlopen)는 그 다음(ADR-0007 D6).

**실측으로 확정된 핵심 제약** (`wf_ebcad9c1` + 직접 코드 확인):
- **forward 는 trait-generic, eviction 은 concrete-bound**. `forward_into` 는 `&[Arc<dyn KVCacheFormat>]` 를 소비(`model_forward.rs:360`)하나 항상 `StandardFormat::new(i, KVCache)` 로 wrap(`ensure_fmt_wrapped:237`). eviction 은 `HandlerContext.caches: &mut [KVCache]`(concrete, `pressure.rs:53`)를 요구하고, `StandardFormat::take_inner/put_inner`(UER seam, `standard_format.rs:125`)가 fmt wrapper 에서 연속 `Vec<KVCache>` 를 복원해 `force_evict(&mut [KVCache])`(D2O cross-layer argmax = 연속 슬라이스 필수)에 넘긴다. → **opaque 가 evict 되려면 `KVCache` 여야 한다.** `OpaqueKvFormat` 의 `OpaqueInner`(≠`KVCache`)는 trait object 라 forward 엔 꽂히지만 UER 슬라이스엔 못 들어간다.
- `is_q4 = dtype==Q4_0` 이진 분기가 `KVCache` 전반에 산재: `alloc_standard_kv_caches`(`bin_setup.rs:216`), `grow`(`:196`), `shrink_to_fit`(`:312`), prune shift(`:560`), `shift_positions`(`:645`), `shift_positions_for_head`(`:738`), `memory_usage_bytes`(`:627`). opaque(U8)는 `_` 분기로 빠져 `dtype.size()=1` → packed block bytes 를 1바이트/element 로 **오산**(silent corruption).
- D2O `apply_weighted_merges`(`standard_format.rs:468`)는 f32/f16/`BlockQ4_0` **typed in-place 3분기**(dequant→weighted sum→requantize). opaque 엔 4번째 descriptor-generic 분기 = **실질 재작성**.

**이미 깔린 토대**: `KVLayoutDesc::bytes_for_elems/block_bytes`(G1 단일원천), `dequant_via_descriptor` 의 OpaqueBuffer sidecar 인식(G3), `encode_via_descriptor`(G4 write encoder), `KV_FORMATS` registry + `find_kv_format`(S4-1), `dequant_to_f32_tensor` floor(ADR-0005 D5).

---

## 2. Decisions

### D1 — 구조: `KVCache` 가 `OpaqueBuffer` 를 1급 저장으로 흡수 (택 A)

`KVCache.k_buffer/v_buffer` 가 `OpaqueBuffer`(U8 tag + `KVLayoutDesc` sidecar)를 들 수 있게 한다. opaque = `StandardFormat(KVCache{OpaqueBuffer})` → **forward(trait)·eviction(UER→concrete `KVCache`) 둘 다 기존 경로 그대로**. `OpaqueKvFormat` 은 **삭제**(dead code) — encode/scatter 오케스트레이션은 `KVCache` write 경로로 이동, GATE-B bit-identical 증명은 `KVCache` 경로 테스트로 **retarget**(증명 소실 X, git history 보존).

**기각 (B)**: `OpaqueKvFormat` 자체 저장 유지 + eviction 을 `KVCacheFormat` trait 표면으로 일반화(`compact`/`shift`/`grow` 추가). ADR-0004 D1 정면 위반("`compact` base-trait 금지, `execute_kv_plan` 독점", `kv_cache_format.rs:8`) + D2O 가 연속 `&mut [KVCache]` 슬라이스를 못 만들어 막힘 + base trait 6→9+ method = `INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC` 압력. `StandardFormat` 이 이미 concrete `KVCache` 를 wrap 하므로 opaque-in-`KVCache` 가 **동형**(구조 일관) — (A)가 외과적.

### D2 — byte-회계: **floor 패턴** (opaque arm 추가, typed arm 무변)

구현은 "모든 사이트 단일 helper 통일"이 아니라 **ADR-0005 D5 floor 패턴**으로 정착했다(더 외과적): `KVCache` 에 `is_opaque()`(pub(crate)) / `opaque_desc()`(sidecar downcast) / `opaque_bytes_per_head()`(`= (head_dim/block_elems)·block_bytes`) helper 를 신설하고, byte-회계·copy·shift·scatter·merge 사이트마다 **opaque arm 을 추가**한다(descriptor-keyed byte 회계). **typed arm(q4_0/f16/f32)은 1바이트도 변경하지 않는다** → byte-identical-by-construction(회귀 표면 0). opaque arm 의 count 단위는 byte(`copy_slice`/`buffer_shift` 의 U8 `type_size=1`). 신규 struct field 0(derive-on-demand). 이유: `copy_slice` element 단위가 dtype-의존(Q4_0=block 18B, U8=1B)이라 copy/shift loop 의 전면 통일은 위험·무이득 — opaque 가 곧 descriptor-keyed 생성 floor 이고 typed 는 특화 arm 으로 보존(D5 거울).

### D3 — dispatch: format **name → `Option<DType>`** (layout 아님)

`synth_q4` 는 `q4_0` 와 **동일 layout**(block_elems:32/bits:4/Nibble/PerBlockF16) — layout 으론 typed/opaque 구분 불가. **format 이름(identity)이 가른다**. 엔진 `builtin_format_dtype(name) -> Option<DType>`: `f32/f16/q4_0/q8_0` → `Some`(typed 경로 무변), 그 외 등록 format → `None`(opaque 저장). 미래 `.so` plugin format 도 정확히 이 "name→no-DType→opaque" 경로를 친다. CLI `--kv-format <name>`(registry `find_kv_format`) 신설, 설정 시 `--kv-type` 보다 우선; 미설정 시 `--kv-type` 하위호환.

### D4 — attention: opaque 는 floor 재사용

opaque-in-`KVCache` 의 attention 은 `dequant_to_f32_tensor`(G3 floor) → 기존 `backend.attention_gen`/`prefill_attention`(ADR-0005 D5 generic floor, ADR-0007 G3 동형). typed dtype 은 기존 특화 경로(Q4_0 on-the-fly deq / GPU flash) 유지. dispatch 지점 = `StandardFormat::attention_into`(또는 그 위임처)에서 buffer 가 `OpaqueBuffer` 면 floor 분기.

### D5 — 범위: **full** (Stage 1+2+3, D2O 포함, gate 없음)

| Stage | 내용 (descriptor-keyed 전환 사이트) | bit-identical 게이트 |
|---|---|---|
| **1 runnable** | `alloc`(D6 opaque alloc) + `grow`/`shrink` byte-회계 + attention floor(D4) + CLI `--kv-format`(D3) | q4_0 grow/shrink byte-identical + opaque decode 도달(`[DecodeLoop] kv storage = OPAQUE`) + coherent (⚠️ 당초 "== q4_0 token-identical" 게이트는 false positive 였음 — Status 정정 참조; 실제 게이트 = unit F32 round-trip + coherence) |
| **2 eviction** | `shift_positions` / prune shift / `shift_positions_for_head` / `memory_usage_bytes` | sliding/h2o opaque evict == q4_0 (host) |
| **3 D2O** | `apply_weighted_merges` typed 3분기 → descriptor-generic 4번째(dequant→weighted sum→`encode_via_descriptor`), `scatter_reduce_q4` mirror | opaque d2o merge == q4_0 round-trip |

구현은 **단계별 커밋·게이트**(S1 커밋 → S2 → S3). 각 단계는 같은 `KVCache` 파일을 만지므로 **순차**(병렬 fan-out 아님).

### D6 — 검증 vehicle: 외부 `synth-q4-format` crate force-link

`engine/Cargo.toml` 에 `synth-q4-format` dep 추가 + format force-link 사이트(`ensure_builtin_kv_formats_registered` 인근)에 `use synth_q4_format as _;`. `--kv-format synth_q4` → **진짜 외부 plugin crate** 의 format 이 production 추론을 돎(builtin demo 보다 북극성 정합 — "외부 plugin → 실추론", GATE-C 직전). e2e 게이트 = `synth_q4`(opaque 저장) == `q4_0`(typed) bit-identical(같은 layout). **커밋 위생**: `engine/Cargo.toml` 의 기존 미커밋 변경(`microbench_score_readback` bin 엔트리, untracked `microbench/score_readback.rs` 참조)은 커밋 시 **선택 스테이징으로 제외**(fresh checkout 빌드 보존, 사용자 microbench WIP 미커밋 유지).

---

## 3. Risks

- **Q4_0 production silent corruption** (RPN 高): descriptor 변환 산술이 1바이트라도 어긋나면 KV garbage → PPL 폭발. 방어 = 사이트별 byte-identical 게이트 + 기존 `KVCache` 단위테스트(prune/shift/grow/cross-layout) 불변 + S1 e2e oracle(`synth_q4`==`q4_0`).
- **U8 type_size=1 함정**: `buffer_shift` 가 opaque 를 pos당 `head_dim` **바이트**만 옮김(실제 `blocks_per_pos*block_bytes` 필요) → block 절단. → D2 descriptor-keyed shift 가 모든 shift 사이트에 필수(Stage 2).
- **`OpaqueKvFormat` 삭제 = GATE-B 산출물 폐기**: 완화 = bit-identical 증명을 `KVCache` opaque 경로 테스트로 retarget(소실 X), git history 보존. `pressure/opaque_format.rs` 제거.
- **D2O merge 재작성 복잡도**: 완화 = `scatter_reduce_q4`(`standard_format.rs:585`)를 descriptor-generic 으로 mirror, q4_0 desc 일 때 bit-identical 게이트. 필요 시 senior-implementer 위임.
- **`engine/Cargo.toml` 커밋 ↔ microbench drift**: D6 선택 스테이징으로 격리.
- **typed-access 비강제**(ADR-0007 잔존): `Tensor::as_slice::<f32>()` 가 opaque 버퍼에 호출되면 raw 재해석. eviction policy 의 layout-direct read(있으면)를 floor 경유로 전환 + audit. `OpaqueTensor` wrapper 는 미도입(다음 단계).

---

## 4. Deferred (본 ADR 범위 밖)

- **GATE-C** (.so cdylib dlopen + register_plugin C-ABI dual-wiring + 런타임 registry) — ADR-0007 D6. host-implementable 이나 별 인프라.
- **GPU opaque arm** — `opencl.rs:6098` attention_gen `_=>kernel_attn_gen`(F32 가정 silent garbage) / `:5577` matmul dtype-guard + readback floor. **device-gated**(host GPU 부재).
- **write encoder family 확장** — `encode_via_descriptor` 현재 PerBlockF16/Nibble(q4_0)만. q8_0(Byte)/q4_1(WithMin)/q2_0(asymmetric)는 범위 한정 `Err` → 필요 시 확장. opaque merge 재인코딩도 동일 제약.
- ~~**e2e eviction 런 (bin 배선)**~~ — **완료 2026-06-09 (`f0d92c66`)**. 핵심은 "bin 배선"이 아니라 **loop builder dead-wiring**(opaque 가 애초에 decode loop 에 도달 못 함, Status 정정 참조)이었다. `argus_bench`(experiment_run, eviction 보유) + `signal_injector`(TCP loopback, `kv.evict_h2o` directive) host e2e 실측(Qwen2.5-1.5B q4_0, 337-token prompt, h2o keep_ratio 0.5):
>   - `--kv-format synth_q4`(opaque): eviction 정상 발화 — `final_pos` 337→215(~50% prune), no crash, coherent. **opaque eviction e2e 증명**.
>   - `--kv-format f16`(typed): 동일(`final_pos` 215) — opaque/typed parity.
>   - 짧은 프롬프트(≤~40 token)는 min-cache floor 미달로 eviction no-op(format 무관).
> - 🔴 **부수 발견: `--kv-format q4_0`(typed) + h2o force_evict mid-decode → SIGSEGV(exit 139)**. f16-typed·synth_q4-opaque 는 정상이므로 **q4_0 native eviction 경로 한정 pre-existing 버그**(본 fix 는 typed arm 무수정 + 동일 q4_0 캐시가 흐를 뿐이라 무관). repro: `argus_bench --kv-format q4_0 ... eviction h2o --keep-ratio 0.5` + signal_injector `kv.evict_h2o`(긴 프롬프트). 별도 조사 필요(q4_0 attention_scores/compact 경로 의심).
> - `argus_cli`(happy-path, eviction 미지원)는 무영향 — eviction 은 `argus_bench` 경로.
