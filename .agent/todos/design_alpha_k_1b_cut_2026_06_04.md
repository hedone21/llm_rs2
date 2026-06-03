# 설계 cut: α-K BC ①-b — forward_into_fmt multi-token prefill entry

**작성**: 2026-06-04 (메인 세션, 직접 코드 전수 후). **상태**: 설계안 → 적대적 검증 대기.
**SSOT**: roadmap `roadmap_alpha_k_bc_completion_2026_06_04.md` line 55(①-b 경계) + `arch/pipeline_stage_design_v2.md` §4.1(trait 7 method 불변) / §9.1-BC1'.

---

## 핵심 발견 (trait 표면 무변)

prefill attention 은 `attention_into` 의 **시그니처 변경 없이** 흡수 가능:
- `seq_len = q.shape().dims()[1]` — decode=1, prefill>1 (q 텐서에서 도출).
- `q_start_pos = current_pos() - seq_len` — causal mask offset. write_kv_batch 가 current_pos 를 seq_len 만큼 전진시킨 직후라, 새 토큰들의 절대 시작 위치 = current_pos - seq_len. **happy-path(prefill 중 eviction 없음) 게이트 하에서만 성립** (게이트가 happy-path 전제이므로 OK).
- impl 내부 분기: seq_len==1 → 기존 decode(attention_gen / Q4 fallback, **무변**); seq_len>1 → 신규 prefill(flash_attention_prefill GPU + CPU fallback).
- **§4.1 "attention 1" 불변 보존** — 새 trait method 0, dtype/codebook 누출 0.

## 변경 구성요소 (전부 additive·게이트 OFF 시 production 무변)

### C-1. `StandardFormat::attention_into` multi-token 분기 (standard_format.rs)
- 맨 위 `let seq_len = q.shape().dims()[1];`. seq_len==1 → 기존 코드 byte-불변. seq_len>1 → 신규.
- 신규 prefill 분기 = forward_prefill(forward.rs:259-585)의 attention 블록을 미러:
  - GPU: `backend.flash_attention_prefill(q, k_cache, v_cache, out, n_heads_q, n_heads_kv, seq_len, cache_seq_len, head_dim, kv_capacity, batch_size, is_head_major)`.
  - dispatched=false(Q4_0/head_dim 미지원/CPU) → CPU fallback: device-only readback OR Q4/F16 dequant + `flash_attention_forward_strided`(strides·start_pos·br/bc=32·window 동일) + device-only writeback.
  - q_start_pos 는 strided fallback 의 causal mask 인자.
- **DRY**: prefill attention+fallback 로직을 신규 free 함수로 추출(fmt 전용 caller). forward_prefill 은 **무수정**(acceptance line 49 "옛 경로 byte-불변"). 중복은 Step 5(KVCacheOps 삭제 시 forward_prefill<C> 제거)에서 자연 해소.

### C-2. `TransformerLayer::forward_prefill_fmt` 신규 (forward_prefill_fmt.rs, forward_gen_fmt 의 prefill 짝)
- forward_prefill 의 **PrefillWorkspace 라이브 경로**(partition off, variance_collector off) 미러:
  rms_norm → QKV matmul → bias → QK-norm → RoPE(`[batch,seq_len,n_heads,head_dim]`) →
  `fmt.write_kv_batch(&k_rope,&ws.v,backend)`(C3) →
  `fmt.attention_into(&q_rope,backend,&mut ws.out_attn,AttnDims{n_heads_q,window},None)`(multi-token) →
  O-proj → residual → FFN(gate/up/silu/down) → residual.
- 생략(라이브 미진입): partition 분기, variance_collector, profiler instrumentation(env-gated 수치-무관).

### C-3. `TransformerModel::forward_into_fmt` multi-token dispatch (transformer.rs:2015)
- `debug_assert_eq!(seq_len,1)` 제거. seq_len==1 → 기존 decode loop(forward_gen_fmt, LayerWorkspace). seq_len>1 → prefill loop(forward_prefill_fmt, PrefillWorkspace).
- PrefillWorkspace = forward_into 의 owned_prefill_ws 블록(transformer.rs:1583-1607) 미러로 **owned 할당**(args 구조체 최소 변경). needs_ws_sync(drop 전 synchronize) 동반.
- logits_last_only: prefill 은 마지막 토큰 hidden 만 lm_head(forward_into:1960 미러).

### C-4. `ModelForward::prefill` 배선 (model_forward.rs:355)
- 게이트 ON(`fmt_eligible && standard_format_gate_enabled()`): chunk loop **전** `ensure_fmt_wrapped()` 호출(현재 step()에서만 호출 → prefill 시작으로 이동, decode 호출은 idempotent no-op) + 청크별 `forward_into_fmt`(dyn_fmts) 호출.
- 게이트 OFF: 기존 forward_into 경로 무변.

## 적대적 검증 결과 (2026-06-04 workflow wfceex20u, 5 lens + param 추출)

핵심 아키텍처 **승인** (trait 시그니처 무변, q_start_pos 도출, forward_prefill 미러). 3건 반증 = cut **명세 완전성 갭**(설계 오류 아님). 검증-반영 정정:

### 정정 A (logits-last-only, major) — C-3 보강
- `TransformerModelForwardFmtArgs`(transformer.rs:203)에 **`pub logits_last_only: bool` 필드 신설**. C-4(ModelForward::prefill)=true, decode step()=false.
- forward_into_fmt prefill 분기(seq_len>1) final-norm 후 lm_head 단계에 transformer.rs:1960-1977 **그대로 미러**: `if logits_last_only && seq_len>1 { last_off=(seq_len-1)*hidden; memory.alloc(hidden*4); copy_slice(&x,&mut x_last,last_off,0,hidden); lm_head(&x_last) } else { lm_head(&x) }`. 누락 시 logits_prefill_last([1,1,vocab]) OOB/wrong-token.

### 정정 B (bit-identical, major) — C-1 prefill arm 명세
- attention_into seq_len>1 분기는 **decode delegate(attention_gen / attention_q4_gpu_fallback) 재사용 절대 금지** (둘 다 single-query, causal mask 부재). forward_prefill(forward.rs:262-585) **통째 미러**(~270 LOC):
  - GPU: `backend.flash_attention_prefill(q, k_cache, v_cache, out, n_heads_q, n_heads_kv, seq_len, cache_seq_len, head_dim, kv_capacity, batch_size, layout==HeadMajor)` → bool dispatched.
  - dispatched=false/CPU: device-only readback(316-375) | Q4_0(376-412)/F16(413-453)/F32(454-461) dequant → per-batch `flash_attention_forward_strided`(18인자: forward.rs:515-534 정확 복제, q_start_pos=current_pos()-seq_len, br=bc=32, window) → device-only writeback(538-583, opencl get_cl_mem+enqueue_write).
- scores=None 전달(prefill 은 score 누적 안 함, `let _ = scores;`).

### 정정 C (causal-mask minor + F32 caveat) — window/F32 처리
- prefill 분기는 decode 진입부의 `effective_cache_len = cache_seq_len.min(window)` clamp + `kv_start_pos` **우회**. 전체 cache_seq_len K + window 를 flash 에 직접 전달(flash 내부 마스킹). q_start_pos=current_pos()-seq_len 그대로.
- **F32 prefill = bit-identical** (decode 와 달리 forward_prefill 도 inline-NEON 아닌 flash 사용 → fmt arm 도 flash → 동일). 단 host parity test 로 확인. device 게이트 dtype = F16/Q4_0/F32 모두(prefill 한정).

### 정정 D (scope) — C-2 모델 분기 1:1 체크리스트
- forward_prefill_fmt 는 forward_gen_fmt 커버리지 1:1 미러: qkv_bias / Gemma3 q_norm·k_norm / pre_ffn_norm·post_ffn_norm / rms_norm_add_unit / gelu_tanh / is_local_attn+window / embed_scale(모델 진입점). 누락 시 Gemma3/Qwen2 prefill garbage.
- partition_ctx / variance_collector / profiler 생략 **정당**(happy-path 미진입 확인).

### 정정 E (wiring minor) — C-4 분기 가드
- prefill chunk loop = step():423-445 패턴: `let dyn_fmts = fmt_caches.as_ref().map(...); if Some(dyn_fmts) { forward_into_fmt } else { forward_into(kv_caches) }`. ensure_fmt_wrapped 단독 호출 후 forward_into(빈 kv_caches) 시 panic. chat(fmt_eligible=false)·eval/ppl/batch(ModelForward 미경유) 무영향 확인.

## DRY 결정 (additive-fork)
- prefill attention arm 은 **forward_prefill_fmt/attention_into 전용 신규 코드**. forward_prefill(generic)은 **무수정**(byte-불변, acceptance line 49). 중복 ~270 LOC 는 host parity test(forward_prefill out_attn == fmt arm out_attn, F16/Q4_0/F32 byte-identical)로 bit-identical 증명, **Step 5(forward_prefill<C> 삭제)에서 자연 해소**. (3c-fwd 의 forward_gen_fmt fork 선례 동일.)

## 구현 순서 (host 게이트 각 단계)
1. C-1: standard_format.rs attention_into seq_len>1 prefill arm(private free fn) + host parity test(vs forward_prefill 동일 입력 byte-identical).
2. C-2: forward_prefill_fmt.rs 신규(forward_gen_fmt 모델분기 1:1) — additive·unwired.
3. C-3: forward_into_fmt seq_len>1 dispatch + args.logits_last_only + owned PrefillWorkspace + logits-last 미러.
4. C-4: ModelForward::prefill fmt 분기 배선 + host 회귀 test(LLMRS_KV_FMT=1, multi-token, panic 부재).
5. device 게이트(S25 R3CY408S5SB + Jetson): `--no-gpu-plan` prefill bit-identical(F16/Q4_0/F32) + avg_tbt Δ≤+3%(TTFT).
