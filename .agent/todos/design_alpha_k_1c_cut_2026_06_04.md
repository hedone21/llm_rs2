# 설계 cut: Phase α-K BC ①-c — eval flip (forward_into_fmt 전환)

**작성**: 2026-06-04 (메인 세션, Architect 역할)
**근거 워크플로우**: `wdrcgtqwz` (3 독립 설계안 → 판정·합성 → 5 적대 검증). 판정 = **Strategy A (transient per-call fmt-wrap)**.
**SSOT**: `arch/pipeline_stage_design_v2.md` §9.1-BC1'(line 794). roadmap = `roadmap_alpha_k_bc_completion_2026_06_04.md` Step 1 ①-c.
**전제**: ①-b `2bf5c500`(forward_into_fmt prefill+decode flip, S25 device PASS).

---

## 채택 접근 = Strategy A (transient fmt-wrap, hook 무수정)

eval 은 지금처럼 concrete `Vec<KVCache>`(EvictionHook) / `Vec<KiviCache>`(KiviHook) 를 **계속 소유**한다.
fmt 는 소유 전환이 아니라 **forward 호출 1회짜리 transient borrow-wrap**:
1. `mem::take(caches)` 로 Vec 통째 꺼냄 (caches=[]).
2. `Arc<StandardFormat>`(또는 `KIVIFormat`) 로 enumerate-wrap → `Vec<Arc<dyn KVCacheFormat>>` 로 clone.
3. `forward_into_fmt(&dyn_slice, ...)`.
4. dyn_slice drop(refcount→1) → `Arc::try_unwrap().into_inner()` 로 concrete 복귀 → `*caches = recovered`.

**hook 은 forward 와 interleave 하지 않으므로**(forward → post_prefill/snapshot/restore 시퀀셜) hook 시점엔 concrete `Vec` 가 복귀해 있다 → **EvictionHook/KiviHook/KVCacheSnapshot/KiviCacheSnapshot/CacheManager::force_evict 전부 1바이트 무수정**. 이것이 enum 대안(B/C: force_evict 의 cross-layer `&mut [KVCache]` 를 fmt-owned Mutex 에서 재구성 = lock-all 어댑터 + deadlock risk + hook 전면 재작성)을 누른 결정적 이유.

### 왜 enum(SSOT 문구) 대신 A
- B/C 는 SSOT 문구("enum EvalCache over Arc<dyn>")에 가까우나, force_evict(`&mut [KVCache]`)·snapshot raw-byte 경로(`k_buffer`/`v_buffer` 직접 + `current_pos` pub 필드 write)를 fmt Mutex seam 안으로 재배선해야 함 → LOC + bit-identical 검증 표면 + reentrant-lock deadlock risk. CLAUDE.md "단순함·외과적" 위반.
- A 는 SSOT 문구에서만 약간 벗어나나(transient-wrap), **정신**(KVCacheOps 를 forward 시그니처에서 제거)엔 부합. hook/eviction 0 변경 = 무회귀 최강.

---

## census 가 드러낸 SSOT 빈틈 (정정)

SSOT line 794 는 ①-c 를 "thin 소비자 교체"라 했으나 **부분 반증**:
- eval 은 `ModelForward` 를 거치지 않고 `model.forward_into()` 를 **직접** 호출(7곳: live 5 = probe `eval_loop.rs:265`/choice-decode `:357`/importance-pass `:577`/token-by-token-prefill `:715`/full-prefill `:783` + dead 2 = run_chunked_prefill `:873/:929` `#[allow(dead_code)]`).
- 이 호출들은 `forward_into_fmt` 가 **미지원**하는 `score_accumulator`·`skip_config`·`importance_collector` 를 넘긴다. `TransformerModelForwardFmtArgs`(현 9필드)엔 없음.
- ⇒ ①-c 는 thin 이 아니라 **fmt forward 의 feature parity 확장(中 규모, ~290 LOC)** 이 필요.

---

## 구현 (판정 7-step + 적대 검증 3 정정 반영)

### [정정 1] 수용 기준 (Verify 1)
①-c 는 `KVCacheOps` **trait 을 삭제하지 않는다**(Step 5 소관, branch-by-abstraction 공존). `EvalCacheKind for KiviCache` 의 `cur_pos`/`needs_scores` 위임은 `KiviCache::current_pos`(=`total_tokens()` private fn)·`needs_attn_scores`(`awqe_enabled` private 필드) 호출이 필요 → **`fmt_bridge.rs` 에 `use KVCacheOps` 1개 잔존**(Step 5 inherent 화로 정리). KVCache 측은 `current_pos`/`high_water_pos` 가 pub 필드라 trait 불요.
**수용 기준** = `grep "forward_into\b" eval_loop.rs`→0 (전부 forward_into_fmt) + `StepHook<C>`/`CacheSnapshot<C>` 의 `C: KVCacheOps` 바운드 제거 + legacy `--eval-ll` EvalOutput JSON bit-identical. (`grep KVCacheOps engine/` 0 은 Step 5.)

### [정정 2] fmt args breaking (Verify 2)
`TransformerModelForwardFmtArgs` 에 4 필드 추가 = breaking change → production fmt 호출처 2곳(`model_forward.rs:399` prefill / `:465` decode)에 `score_accumulator: None, skip_config: None, importance_collector: None, cache_self_need_scores: false` 명시 추가 필수. "byte-불변"은 **런타임 출력 한정**(source 는 호출처 touch). `forward_into_fmt` 본문에 `gpu_score_active = backend.gpu_score_acc().is_some_and(|a|a.is_active())` 계산 추가.

### [정정 3] need_scores 에 AWQE 항 복원 (Verify 3)
`forward_gen_fmt` 의 `need_scores = args.need_scores`(AWQE 항 없음 — base trait 에 `needs_attn_scores` 부재, §4.1 R4 ③). 따라서 KIVI AWQE self-need 를 **forward_into_fmt 레벨에서 주입**: 새 args 필드 `cache_self_need_scores: bool` (eval 이 `caches[0].needs_attn_scores()` 로 산출). decode arm:
```
let acc_need = if gpu_score_active { false } else { score_acc.as_ref().is_some_and(|a| a.should_track_layer(i)) };
let need_scores = acc_need || args.cache_self_need_scores;   // forward_gen.rs:409 미러
```
누락 시 `LLMRS_KIVI_AWQE=1 + experimental off` 에서 AWQE proxy(kivi_flush_count 등) silent 발산.

### 구현 단계
1. **seam**: `StandardFormat::into_inner(self)->KVCache`(`StandardFormatInner` destructure) + `KIVIFormat::into_inner(self)->KiviCache`(`Mutex::into_inner`). `pub(crate)`. base trait 무변(creep 0).
2. **fmt_bridge.rs**(신규, `session/eval/`): `EvalCacheKind` trait(`forward_fmt_roundtrip` + `cur_pos`/`set_cur_pos`/`needs_scores`) + `KVCache`/`KiviCache` impl. KVCache: 필드 직접(`set_cur_pos` 는 `KVCache::set_current_pos` 식 미러 = `self.current_pos=p; if p==0 {high_water=0}`). KiviCache: `use KVCacheOps`(정정1). round-trip 무손실 unit test.
3. **forward_into_fmt 확장**(transformer.rs): args +4필드(score_accumulator/skip_config/importance_collector/cache_self_need_scores). 본문: `gpu_score_active` 계산; decode arm `need_scores`(정정3)+skip(`skip_config.map_or((false,false),|sc|(sc.skip_attn(i),sc.skip_mlp(i)))`); prefill arm skip 동일; layer 전 `coll.snapshot_before`; layer 후 `coll.record_after` + CPU score-feed(`fmts[i].current_pos()`, `forward_into.rs:1894-1922` 미러); loop 후 `acc.end_step()` + GPU end_step. forward_prefill_fmt **소스 무변**(skip 값만 다르게 전달, importance/score 는 forward_into_fmt layer-loop 가 x 읽어 처리). 호출처 2곳(정정2).
4. **hook.rs**: `StepHook<C: KVCacheOps>`→`StepHook<C>`, `CacheSnapshot<C: KVCacheOps>`→`<C>`, `use KVCacheOps` 삭제. eviction_hook/kivi_hook impl 0 변경.
5. **eval_loop.rs flip**: `<C: KVCacheOps>`→`<C: EvalCacheKind>`(run_eval_ll_generic + 4 prefill 헬퍼). 7 forward_into→`C::forward_fmt_roundtrip(kv_caches, |fmts| forward_into_fmt{fmts, score_accumulator, skip_config, importance_collector, cache_self_need_scores, ...})`. 직접 C 호출 3곳(`cur_pos`/`set_cur_pos`/`needs_scores`). `use KVCacheOps` 삭제.
6. **host 게이트**: legacy `--eval-ll` flip 전 vs 후 EvalOutput JSON diff=0 — (i) **F16 KV** + H2O eviction(★2: F32-host-decode 는 inline-NEON vs attention_gen 으로 NOT bit-identical → F16/Q4 강제), (ii) Q4_0 KV, (iii) **KIVI(LLMRS_KIVI_AWQE=1 + experimental off)** 경로. choice_nlls/predicted/predicted_raw + (KIVI)kivi_flush_count. + score-feed accumulate 동치 unit test.
7. 정리 + `cargo fmt`/clippy + commit + notify.

---

## Landmines (R6)
- **★2 F32-host decode**: F16/Q4 게이트 강제. F32 KV eval bit-identical 은 device-only 에서만(①-b carve-out 상속, 신규 위험 0).
- **fmt_bridge round-trip panic-safety**: `let r = run(&dyn); *caches = recover; r` — run 이 `?` early-return 해도 복귀-후-return. run 내부 panic 은 caches 손실(=process abort 라 실해 없음).
- **fmt_bridge 의 `use KVCacheOps`**(KiviCache impl): ①-c 수용된 잔여, Step 5 inherent 화로 정리. `grep KVCacheOps engine/src/session/eval/`=0 은 ①-c 목표 아님(JSON bit-identical 이 목표).
- **forward_into_fmt additive-fork 중복 증가**: score-feed/importance 미러가 forward_into 중복본 키움 → Step 5 dedup 부담. 미러 코드에 `forward_into:NNNN 미러` 주석으로 추적성.
- **per-forward Arc 재할당**: round-trip 이 매 forward 마다 N StandardFormat::new + k_cast/v_cast scratch 재생성. eval=cold 라 무시(정확성 무관).
