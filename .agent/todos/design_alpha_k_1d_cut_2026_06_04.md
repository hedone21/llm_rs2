# 설계 cut: α-K BC ①-d — B-2 비-decode forward_into→forward_into_fmt flip

**작성**: 2026-06-04 (메인 세션 오케스트레이터)
**설계+적대검증**: workflow `w12qx2ybg` (3 design lens[Explore] → judge → 5 adversarial verify[Explore]). refuted_count=1(V4=lens-level 모호성, judge 합성이 전부 해소).
**SSOT**: `arch/pipeline_stage_design_v2.md` §9.1-BC1' · roadmap `roadmap_alpha_k_bc_completion_2026_06_04.md` Step 1 ①-d.
**선행**: ①-c `1e4f20fe`(`EvalCacheKind::forward_fmt_roundtrip` + `forward_into_fmt` 4필드 확장).

---

## 결정 요약

**범위 (included 10 site, 최종)** — handoff(warmup/qcf/batch/ppl-KVCache) + dump_importance(구조적 쌍둥이). **run_kivi_ppl(KIVI 2 site)는 게이트에서 defer 확정(아래 ★게이트 발견)**:
| site | 함수 | seq_len | workspace | 분기 | cache | 비고 |
|---|---|---|---|---|---|---|
| warmup.rs:69 | run_warmup | 1(기본) | None | **NEW fallthrough** | KVCache | Vec sig change |
| qcf_runtime.rs:212 | run_qcf_warmup_workflow | >1 | None | prefill_fmt | KVCache | Vec sig(QcfWarmupCtx) + importance |
| qcf_runtime.rs:300 | run_qcf_warmup_workflow(decode-X) | 1 | None | **NEW fallthrough** | KVCache | importance, llo=true |
| batch/runner.rs:383 | run_prompt_batch | >1 | None | prefill_fmt | KVCache(owned) | score+skip |
| batch/runner.rs:452 | run_prompt_batch | >1 | None | prefill_fmt | KVCache | score+skip, llo=true |
| batch/runner.rs:742 | run_prompt_batch | 1 | Some | gen_fmt | KVCache | score+skip, x_gen=Some |
| batch/runner.rs:817 | run_prompt_batch | 1 | Some | gen_fmt | KVCache | score+skip |
| ppl/runner.rs:771 | run_ppl | >1 | None | prefill_fmt | KVCache | Vec sig, score, begin_step 선행 |
| ppl/runner.rs:935 | run_ppl | 1 | Some | gen_fmt | KVCache | Vec sig, score |
| dump_importance.rs:65 | run_dump_importance | >1 | None | prefill_fmt | KVCache(owned ctx) | importance, sig 불요 |

### ★ 게이트 발견 (설계/적대검증이 놓친 갭) — run_kivi_ppl defer
설계 단계는 ppl:367/448(KIVI)을 included 로 잡았으나, **host 게이트에서 panic 발견**(`x86.rs:228`, slice index 15232 > 15104=59×256): **KIVIFormat::attention_into 는 prefill arm 이 없다**(`kivi_format.rs:95-173` — 항상 `attention_gen`=single-query decode 전용). forward_prefill_fmt 가 multi-token(59) prefill query 를 넘기면 attention_gen 이 single-query 가정으로 인덱스 초과. ①-c eval KIVI 가 안 걸린 이유 = eval 은 KIVI+AWQE 시 **token-by-token prefill**(eval_loop.rs:641-658, seq_len=1 forward_gen_fmt)이라 multi-token prefill_fmt 를 한 번도 안 탐. run_kivi_ppl 은 AWQE off(needs_scores=false) → full multi-token prefill → 최초로 갭에 진입. **StandardFormat 의 `prefill_attention` 재사용은 불가**(KiviCache::get_view 가 compact view + bits 별 layout/capacity 상이 + GPU native·bits=16 HeadMajor device 검증 필요) → KIVIFormat multi-token prefill arm 신설 = 별도 feature 증분(**①-e 후보**). **run_kivi_ppl(prefill+decode)은 ①-d 에서 generic forward_into 유지**(decode 만 fmt 가능하나 함수 단위 일관성). 적대검증(V1/V3)이 forward_prefill_fmt bit-identity(헤더 주장)만 보고 KIVIFormat prefill arm 부재를 못 봄 → **게이트가 잡음**.

**excluded (근거 명시)**:
- `ppl/runner.rs:367/448`(run_kivi_ppl): KIVIFormat multi-token prefill arm 부재(★게이트 발견) → **defer(①-e)**.
- `prefill.rs:348/446`(run_chunked_prefill): profiler+variance_collector 사용 → forward_into_fmt 미지원. legacy-only. **defer**(Step 5/①-e). 추가 시 fmt instrumentation 필드 신설 = 외과적 위반.
- `kivi_forward.rs:157/188`(KiviForward chat): **production decode**(chat/session.rs:487). 원칙3 cold-only 위반 → device-gate 동반 별도 step defer.
- `model_forward.rs:399/470`: production hot. Step 3 (3p) ④-a 영역.

---

## 핵심 변경

### 1. forward_into_fmt — workspace=None decode fallthrough (발산 A)
**문제**: 구 `forward_into` decode(seq_len==1)는 `layer.forward`(transformer_layer.rs:261)로 → `seq_len==1 && workspace.is_some()` 일 때만 forward_gen, **workspace==None 이면 forward_prefill fall-through**(:287, degenerate 1-token, flash). `forward_into_fmt`(transformer.rs:2148-2150)는 무조건 forward_gen_fmt + `.expect()` → workspace=None 패닉.

**해결**: decode 분기(2137)에 workspace 유무 분기 추가. workspace=None → `forward_prefill_fmt`(degenerate seq_len=1, 구 forward_prefill 과 bit-identical — forward_prefill_fmt.rs:1-22 헤더). `ws_cfg` 를 함수 상단으로 끌어올려 prefill arm + fallthrough 공유. `owned_prefill_ws`/`needs_ws_sync` 재사용.

```rust
if is_decode {
    if workspace.is_none() {
        // 발산 A: 구 layer.forward → forward_prefill fall-through 미러.
        if owned_prefill_ws.is_none() {
            owned_prefill_ws = Some(PrefillWorkspace::new(&ws_cfg, seq_len /*=1*/, memory, backend.clone())?);
            needs_ws_sync = backend.is_gpu();
        }
        let pws = owned_prefill_ws.as_mut().unwrap();
        layer.forward_prefill_fmt(ForwardPrefillFmtArgs { x:&mut x, fmt:&fmts[i], start_pos, backend, pws, rms_norm_eps, rope_theta, head_dim, is_local, skip_attn:s_attn, skip_mlp:s_mlp })?;
    } else {
        // 기존 forward_gen_fmt (else 로 이동, 동작 불변)
    }
}
```

**production 무영향 (전수 검증)**: forward_into_fmt 현 호출처 — model_forward.rs:399(prefill seq_len>1), :470(decode workspace=Some), eval_loop decode 4곳 전부 workspace=Some, prefill 3곳 전부 seq_len>1. ⇒ workspace==None && seq_len==1 만드는 기존 호출처 0건. 새 분기는 ①-d warmup/qcf:300 에서만 발화. 순수 additive(else = 기존 코드 이동).

### 2. slice→Vec 시그니처 (4곳, caller 0 변경)
clone slice-variant 는 **unsound 기각**: KVCache non-Clone(derive 없음); Tensor clone 은 buffer Arc shallow alias → 복제가 같은 버퍼 오염; Format 은 cache by-value 소유라 in-place wrap 불가.
- `run_warmup(kv_caches: &mut [KVCache])` → `&mut Vec<KVCache>` (caller prefill.rs:176 `&mut kv_caches` owned 무변경)
- `run_ppl(kv_caches: &mut [KVCache])` → `&mut Vec<KVCache>` (caller run_ppl_dispatch :100/:158 owned)
- `QcfWarmupCtx.kv_caches: &'a mut [KVCache]` → `&'a mut Vec<KVCache>` (caller eval/runner.rs:81, legacy:1718 owned)
- dump_importance/batch/run_kivi_ppl: 이미 owned Vec → 불요.
함수 내 비-forward 사용(iter_mut/.len()/[i])은 `&mut Vec` 의 Deref<[T]> 로 그대로. sub-slice 전달 0건(grep).

### 3. KIVI AWQE 주입 — **defer(①-e)와 함께 보류**
원설계: ppl:367/448 이 roundtrip 전 `cache_self_need_scores = kv_caches.first().is_some_and(|c| c.needs_scores())` 선계산. **그러나 run_kivi_ppl 자체가 KIVIFormat prefill arm 부재로 defer(★게이트 발견)** → AWQE 주입도 ①-e 로 함께 이월. ①-d 의 KVCache site 는 전부 `cache_self_need_scores: false` 고정(KVCache::needs_scores=false).

### 4. per-site 전환 패턴
`model.forward_into(TransformerModelForwardArgs {...})` → `C::forward_fmt_roundtrip(&mut kv_caches, |fmts| model.forward_into_fmt(TransformerModelForwardFmtArgs {...}))`. forward 1점만 클로저로 감싸고 전후 kv_caches 접근(migrate/total_bytes/q2_tokens)은 클로저 밖 시퀀셜 유지(borrowck).

---

## host 게이트 (device 불요 — 전부 cold path)
1. build -p llm_rs2 + clippy --workspace -D warnings + fmt --check. sig 변경 4곳 caller 무변경 컴파일.
2. cargo test --workspace (fmt_bridge 3 test 유지 + forward_into_fmt fallthrough 단위 test 1 신설 권장).
3. legacy bit-identical (CPU, qwen2.5-1.5b):
   - **warmup**: 표준 생성 `legacy_generate -m q4_0 -b cpu -p "..." -n 16 --temperature 0` → 출력 텍스트 BEFORE/AFTER 동일(run_warmup 발화).
   - **batch**: `--prompt-batch test_batch.jsonl` → 출력 동일.
   - **ppl(KVCache)**: `--ppl <text>` Q4_0 → NLL bit-identical. **실측 ✅**: NLL=173.1049 BEFORE==AFTER.
   - **ppl(KIVI)**: defer(forward_into 유지) → **bit-identical by construction. 실측 ✅** total_nll/ppl/flush_count/q2/res_pos 완전일치.
   - **dump_importance**: `--dump-importance` → importance table 동일. **실측 ✅ IDENTICAL**.
   - **warmup fallthrough**: `eviction sliding --window 1024`로 run_chunked_prefill→run_warmup 강제 → `[WARMUP] tokens=1` 발화 + 출력 "Paris..." 정상 + **패닉 없음. 실측 ✅**(seq_len=1 fallthrough 런타임 검증).
   - (qcf:300 decode-X: secondary GGUF + --decode-x-steps>0 필요 — 미실행; warmup fallthrough 가 동일 seq_len=1 fallthrough 경로를 런타임 커버.)
4. carve-out: F32+host-mapped 는 forward_gen_fmt NOT bit-identical(host 기본 F16, fallthrough 는 prefill_fmt=flash 라 F32 도 OK) → 게이트에서 f32-host 배제.

**게이트 실측 종합 (HEAD pre-①-d BEFORE vs post-①-d AFTER, CPU qwen2.5-1.5b-q4_0)**: build clean / clippy -D warnings clean / fmt clean / lib test 1241 pass·13 fail(전부 backend::opencl GPU 부재 pre-existing)·비-opencl 회귀 0 / 변경모듈 74 pass / batch·ppl(KVCache)·ppl(KIVI)·dump_importance bit-identical + warmup fallthrough 런타임 PASS.

---

## Landmines (ranked)
0. [★게이트 발견·해소됨] **KIVIFormat::attention_into 는 multi-token prefill arm 이 없다**(attention_gen=single-query). forward_prefill_fmt 가 KIVI multi-token prefill query 를 넘기면 panic(slice index 초과). run_kivi_ppl 을 ①-d 에서 defer(forward_into 유지)로 해소. **①-e** = KIVIFormat multi-token prefill arm 신설(KiviCache compact view aware prefill_attention + bits/layout/GPU-native/bits16-HeadMajor device 검증) 후 run_kivi_ppl(2 site) fmt 전환 + AWQE 주입. → roadmap Step 1 잔여.
1. [P0 패닉] fallthrough 미선행 시 warmup(기본 seq_len=1)/qcf:300 이 `.expect()` 즉시 패닉. **발산 A 는 엣지가 아니라 기본 경로.** (해소: forward_into_fmt decode 분기에 workspace=None → forward_prefill_fmt fallthrough 추가, 런타임 검증 ✅)
2. [P0 정합성] fallthrough 를 forward_gen_fmt(ws 새 할당)로 보내면 bit 깨짐 — 반드시 **forward_prefill_fmt**.
3. [P1 KIVI 무음] ppl:367/448 cache_self_need_scores 주입 누락 → AWQE 무음 오염(패닉 없음).
4. [P1 slice/Vec] sig 변경 안 하면 컴파일 실패. clone 우회 unsound.
5. [P2 borrowck] roundtrip 클로저가 &mut kv_caches 잡는 동안 클로저 밖 접근(batch migrate:619 등) 겹치면 에러 — forward 1점만 감싸기.
6. [P2 qcf:300 루프] decode-X 매 step roundtrip(take/wrap/unwrap) — cold 허용, step-단위 유지가 단순.
7. [P3 graph fast-path] forward_into_fmt 는 qnn graph fast-path 없음 — ①-d site 전부 host CPU eval 라 무관.
8. [P3 importance decode] qcf:300 fallthrough 의 importance 기록은 레이어루프 레벨(2132/2197) — 구 forward_into(1743/1893)와 동작 일치, BEFORE/AFTER 로 확인.
9. [P3 F32-host carve-out] decode workspace=Some site(batch:742/817, ppl:935) --kv-type f32+host-mapped 시 NOT bit-identical — 게이트 배제.
