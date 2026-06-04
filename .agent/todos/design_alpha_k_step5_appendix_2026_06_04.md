# Step 5 설계 — 부속 산출물 (census / designs / adversarial verify 원본)

## census (JSON)
```json
[
 {
  "area": "KVCacheOps trait 폐기 (Step 5) Census",
  "summary": "\nKVCacheOps trait 정의 분석 + 3개 cache 타입 impl + fmt 래퍼 분석:\n\n(1) **trait 정의** (engine/src/kv_cache_ops.rs):\n  - 20개 메서드 (기본 구현 8개 + override 가능 12개)\n  - Send bound 만 요구 (동기 컨텍스트 전용)\n  - default impl 8개: get_buffers_mut/advance_pos/ensure_capacity/needs_attn_scores/set_attn_scores/get_kivi_raw_buffers/res_pos/q2_tokens/res_cap/needs_flush/flush_if_needed (모두 no-op)\n\n(2) **cache 타입별 impl 상황**:\n  - KVCache (kv_cache.rs:1006): 모든 메서드 override (current_pos/set_current_pos/capacity/kv_heads/head_dim/layout/kv_dtype/memory_usage_bytes/update/get_view/get_buffers_mut/advance_pos/ensure_capacity) — 12개 필수\n  - KiviCache (kivi_cache.rs:2237): 모든 메서드 override + KIVI 전용 5개 (needs_attn_scores/set_attn_scores/get_kivi_raw_buffers/res_pos/q2_tokens/res_cap/needs_flush/flush_if_needed/advance_pos) — 18개 override\n  - OffloadKVCache (offload.rs:263): 13개 override (current_pos/set_current_pos/capacity/kv_heads/head_dim/layout/kv_dtype/memory_usage_bytes/update/get_view) — 13개\n\n(3) **fmt 래퍼 내 KVCacheOps 위임 패턴**:\n  - StandardFormat (pressure/standard_format.rs): 내부 KVCache에 KVCacheOps::current_pos/KVCacheOps::get_view/KVCacheOps::capacity/KVCacheOps::layout/cache.kv_heads()/cache.head_dim() 위임 (inherent + trait 혼재)\n  - KIVIFormat (pressure/kivi_format.rs): 내부 KiviCache에 KVCacheOps::current_pos/KVCacheOps::capacity/KVCacheOps::layout/KVCacheOps::get_view 위임\n  - 두 wrapper 모두 Mutex 내부에서만 호출 → 생산 경로 미진입(unwired, unit test 전용)\n\n(4) **구 경로 (OLD forward_gen<C>/forward_prefill<C>) 소비자**:\n  - forward_into_offload (TransformerModel::forward_into_offload → forward.rs 제네릭 루프): OffloadKVCache monomorphization (현재 활성)\n  - run_chunked_prefill (session/prefill.rs:101): KVCache 직접 소비 (추가 조사 필요)\n\n(5) **메서드별 호출 빈도** (non-test grep):\n  - current_pos(): 74회 호출 (trait delegation + inherent 혼재)\n  - set_current_pos(): 4회 (eviction/reset seam만 사용)\n  - capacity/layout/kv_heads/head_dim: 전체 geometry getter (대량 호출)\n  - get_view(): 114회 (attention 열람)\n  - update(): 215회 (KV 쓰기)\n  - advance_pos(): 20회 (위치 증분)\n  - KIVI 전용 (res_pos/q2_tokens/res_cap/needs_flush): 10~116회 (KIVI Plan seam)\n",
  "references": [
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/kv_cache_ops.rs",
    "lines": "55-160",
    "kind": "trait-def",
    "detail": "KVCacheOps trait 정의. 20개 메서드: current_pos/set_current_pos/capacity/kv_heads/head_dim/layout/kv_dtype/memory_usage_bytes/update/get_view/get_buffers_mut/advance_pos/ensure_capacity/needs_attn_scores/set_attn_scores/get_kivi_raw_buffers + KIVI 5종 (res_pos/q2_tokens/res_cap/needs_flush/flush_if_needed). 기본 구현 8개 (모두 no-op 또는 0 반환).",
    "disposition": "needs-rewire"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/pressure/kv_cache.rs",
    "lines": "1006-1073",
    "kind": "impl-for-cache",
    "detail": "impl KVCacheOps for KVCache: 12개 메서드 override (all non-default). inherent 메서드와 trait 메서드 동명: current_pos/set_current_pos/capacity/kv_heads/head_dim/layout/kv_dtype/memory_usage_bytes. No override for KIVI methods (inherited no-op).",
    "disposition": "migrate-fmt"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/pressure/kivi_cache.rs",
    "lines": "2237-2453",
    "kind": "impl-for-cache",
    "detail": "impl KVCacheOps for KiviCache: 18개 override. 유일하게 needs_attn_scores/set_attn_scores/get_kivi_raw_buffers/res_pos/q2_tokens/res_cap/needs_flush/flush_if_needed/advance_pos override. set_current_pos는 no-op (pos 파생). layout은 조건부 (bits=16 GPU⇒HeadMajor, 아니면 SeqMajor).",
    "disposition": "migrate-fmt"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/pressure/offload.rs",
    "lines": "263-374",
    "kind": "impl-for-cache",
    "detail": "impl KVCacheOps for OffloadKVCache: 13개 override (current_pos/set_current_pos/capacity/kv_heads/head_dim/layout/kv_dtype/memory_usage_bytes/update/get_view). No override for KIVI/buffer direct methods (all no-op). generic forward monomorphization 진입점 (현재 활성).",
    "disposition": "migrate-fmt"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/pressure/standard_format.rs",
    "lines": "56-120",
    "kind": "fmt-delegation",
    "detail": "StandardFormat::with_cache_mut + plan_geometry/plan_advance: inner KVCache의 KVCacheOps getter를 Mutex interior-mut으로 위임. production 미진입(unwired, unit test 전용). inherent 메서드와 trait 메서드 모두 호출 (KVCacheOps::current_pos vs cache.kv_heads() 혼재).",
    "disposition": "removable"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/pressure/kivi_format.rs",
    "lines": "66-102",
    "kind": "fmt-delegation",
    "detail": "KIVIFormat::current_pos/capacity 구현: KVCacheOps::current_pos/capacity를 lock guard 안에서 위임. production 미진입. inherent 메서드 부재(trait만 호출).",
    "disposition": "removable"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/layers/transformer_layer/forward_gen.rs",
    "lines": "24",
    "kind": "generic-consumer",
    "detail": "forward_gen<C: KVCacheOps>: B-2 OLD-chain 진입점. forward 본문(transformer.rs execute 루프)에서 generic monomorphization. Step 3 후 NEW plan path(execute_fmt/build_plan_fmt)로 flip 완료되어 B-2 OLD는 이제 offload만 소비.",
    "disposition": "dies-with-legacy"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/session/forward/offload_forward.rs",
    "lines": "156-176, 190-200",
    "kind": "generic-consumer",
    "detail": "OffloadForward::prefill/step: model.forward_into_offload 호출. Vec<OffloadKVCache> 소비. forward_into_offload 내부는 generic loop(forward_gen<C: KVCacheOps>). Step 5 목표 B-2 OLD-chain 2번 소비자.",
    "disposition": "needs-rewire"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/session/prefill.rs",
    "lines": "101-200",
    "kind": "generic-consumer",
    "detail": "run_chunked_prefill: PrefillCtx에서 Vec<KVCache> 직접 소비. KVCacheOps::forward_gen<C> 호출 여부 확인 필요. Step 5 목표 B-2 OLD-chain 2번 소비자 후보 (추가 확인 필요).",
    "disposition": "needs-rewire"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/models/transformer.rs",
    "lines": "1,3717",
    "kind": "plan",
    "detail": "TransformerModel: forward_into_offload 메서드 정의 위치. OffloadKVCache + generic forward loop 진입점. forward 경로 3가지(plan/forward_into/forward_into_offload) 중 forward_into_offload만 KVCacheOps 직접 소비(Step 4 완료).",
    "disposition": "migrate-fmt"
   }
  ],
  "blockers": [
   "run_chunked_prefill이 forward_gen<C: KVCacheOps>을 호출하는지 명확 확인 필요 (prefill.rs에서 forward 경로 호출 지점 추적)",
   "PrefetchableCache trait vs KVCacheOps의 의존성 분리 상태 확인 (Step 2 완료 선언이지만 inherent method 호출 위치 재확인)",
   "StandardFormat/KIVIFormat이 fmt래퍼일지라도 inner cache의 inherent 메서드 호출 시 trait delegation 필요 여부 (Mutex interior 경계)",
   "forward_into_offload 내부 loop가 여전히 generic<C: KVCacheOps>를 사용하는지 아니면 구체 concrete OffloadKVCache를 사용하는지 명확화",
   "KIVI 전용 메서드(res_pos/q2_tokens/res_cap) 호출 위치 추적 (Plan hot-path vs cold-path 구분)"
  ]
 },
 {
  "area": "Census 영역 2: OLD generic forward chain + plan KVCacheOps 소비",
  "summary": "\nStep 5 설계(KVCacheOps trait 폐기)의 핵심 타겟: 97 참조/26 파일의 KVCacheOps generic 경로.\n\n**현황 (Census 결과)**\n\nPhase α-K BC Step 1~3에서 copy-fork로 남긴 중복:\n- forward_into_fmt (new, 2026; trait-object) ↔ forward_into (old, generic C: KVCacheOps)\n- forward_gen_fmt (new, 라이브 decode 부분만) ↔ forward_gen (old, generic)\n- forward_prefill_fmt (new, 라이브 prefill 부분만) ↔ forward_prefill (old, generic)\n- execute_fmt (new, StandardFormat 핸들) ↔ execute (old, generic C: KVCacheOps)\n\n**OLD 경로의 현재 호출처 (Step 1~3 이후에도 남아있는 소비)**\n\n1. **forward_into<C: KVCacheOps>** (transformer.rs:1509)\n   - (1a) model_forward.rs:429 (prefill chunked, fallback용 — LLMRS_KV_FMT OFF)\n   - (1b) model_forward.rs:589 (decode fallback용 — plan invalidation 또는 미빌드)\n   - (1c) prefill.rs:348 (legacy profiler/variance 콜렉션용 — 미래: family bin 이주)\n   - (1d) prefill.rs:446 (CPU 청크 폴백용 — 미래: family bin 이주)\n   - (1e) kivi_forward.rs:157 (KiviCache prefill — 미래: family bin 이주, 또는 KiviCache → KVCacheFormat wrapper)\n   - (1f) kivi_forward.rs:188 (KiviCache decode — 위와 동일)\n   **결론**: argus_cli 게이트 OFF → legacy_generate bin 폐기 후 자동 dead code (단, 미래 family bin 또는 wrapper필요)\n\n2. **forward_gen<C: KVCacheOps>** (transformer_layer/forward_gen.rs:24, pub(super))\n   - (2a) layer.forward() (transformer_layer.rs:264) — seq_len==1 && workspace.is_some() 시 호출\n   - **call chain**: forward_into → layer loop → layer.forward → forward_gen (generic)\n   - **중복 상태**: forward_gen_fmt(라이브 decode arm만)는 partition/fused/KIVI/inline-NEON dead 부분 생략 — forward_gen의 큰 부분(554~1068 inline-NEON F32 attention, partition·fused 분기)은 forward_gen_fmt에 없음 → **dead code 아님, 정상 작동 중**\n\n3. **forward_prefill<C: KVCacheOps>** (transformer_layer/forward.rs:41, pub(crate))\n   - (3a) layer.forward() (transformer_layer.rs:287) — seq_len>1 또는 workspace 미제공 시 호출\n   - **call chain**: forward_into → layer loop → layer.forward → forward_prefill (generic)\n   - **중복 상태**: forward_prefill_fmt(라이브 PrefillWorkspace 행복 경로만)는 partition/variance/profiler dead 부분 생략 → **전체 기능 정상, 일부 instrumentation dead**\n\n4. **execute<C: KVCacheOps>** (backend/opencl/plan.rs:1277)\n   - (4a) transformer.rs:3371 (execute_plan_try 폴백)\n   - (4b) transformer.rs:3681 (execute_plan_for_kivi path)\n   - **중복 상태**: execute_fmt(plan.rs:1945)는 plan geometry snapshot으로 4개 KVCacheOps getter 호출을 1 lock으로 통합\n   - **제거 경로**: model_forward.rs:484/555 == plan 시도 → execute_plan_fmt(데이터 경로) or execute_plan(data path) 모두 현존, 두 call site 모두 OK(game 게이트 기반 분기)\n\n**KVCacheOps 메서드 사용 통계 (Step 5 제거 대상 → inherent/rename trait)**\n\ngeneric forward 체인에서 호출되는 KVCacheOps 메서드 (래퍼/format에서 델리게이트):\n- current_pos (좌표/마스크 계산) — standard_format.rs:76,139,158,195,217 / kivi_format.rs:72,119-121,169\n- capacity (검증/마스크) — standard_format.rs (암묵적) / kivi_format.rs:76,119\n- layout (attention GPU dispatch 결정) — standard_format.rs (미사용) / kivi_format.rs:120\n- get_view (K,V buffer 접근) — standard_format.rs:366,394 / kivi_format.rs:124,168\n- res_pos, q2_tokens (eviction 상태) — execute path만, forward layer 체인에서 미사용\n- update (KV write) → **fmt 위임으로 제거** (StandardFormat::write_kv / KiviFormat::write_kv)\n\n계층별 사용:\n- transformer.rs (forward_into_fmt/forward_into_offload/execute_plan_fmt) — format 래퍼로 대체 ✓\n- layer.forward<C: KVCacheOps> (seq_len dispatch) — 제거 불가(layer.forward_gen/forward_prefill은 generic 경계)\n- layer.forward_gen<C> (decode path) — forward_gen_fmt 로 복제됨 (partition/fused dead 부분 있음)\n- layer.forward_prefill<C> (prefill path) — forward_prefill_fmt 로 복제됨 (variance/partition dead 부분 있음)\n- backend/opencl/plan.rs execute<C> — execute_fmt 로 복제됨 (geometry snapshot으로 개선)\n",
  "references": [
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/models/transformer.rs",
    "lines": "1509-1843",
    "kind": "trait-def",
    "detail": "forward_into<C: KVCacheOps> — OLD generic, layer 루프에서 forward_into → layer.forward → seq_len dispatch → forward_gen/forward_prefill",
    "disposition": "dies-with-legacy (forward_into_fmt로 완전 대체, argus_cli gamepath 사용 후 삭제)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/layers/transformer_layer.rs",
    "lines": "236-311",
    "kind": "generic-consumer",
    "detail": "layer.forward<C: KVCacheOps> — seq_len 분기(1→forward_gen, >1→forward_prefill), layer.forward_gen/forward_prefill 호출",
    "disposition": "needs-rewire (forward_gen/forward_prefill 제거 후 이 함수 자체 삭제, 또는 fmt dispatch로 변경)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/layers/transformer_layer/forward.rs",
    "lines": "41-585",
    "kind": "impl-for-cache",
    "detail": "forward_prefill<C: KVCacheOps> — prefill seq_len>1 경로, KV write (update) + attention 수행. dead 블록: partition(626-740) / variance(499-513) / profiler / fallback(789-1271)",
    "disposition": "dies-with-legacy (forward_prefill_fmt 라이브 경로와 복제 — step 5 함수 전체 삭제, fmt로 대체)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/layers/transformer_layer/forward_gen.rs",
    "lines": "24-1068",
    "kind": "impl-for-cache",
    "detail": "forward_gen<C: KVCacheOps> — decode seq_len==1 경로, KV write + attention. dead 블록: partition·fused(300+라인) / KIVI(400+) / inline-NEON F32 attention(554-1068) / kv_start_pos 유도(404)",
    "disposition": "dies-with-legacy (forward_gen_fmt 라이브 decode arm와 복제 — forward_gen 전체 삭제, fmt로 대체)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/layers/transformer_layer/forward_gen_fmt.rs",
    "lines": "1-250",
    "kind": "fmt-delegation",
    "detail": "forward_gen_fmt — forward_gen의 decode arm 복제(partition/fused/KIVI/inline-NEON dead), KV write+attention을 fmt 위임. live arm: norm/QKV/RoPE/O-proj/FFN/residual = backend 직호출",
    "disposition": "migrate-fmt (forward_gen 삭제 후 forward_gen_fmt를 \"forward_gen\" 함수로 rename, fmt 인자 제거 → &Arc<dyn KVCacheFormat> format 구체화)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/layers/transformer_layer/forward_prefill_fmt.rs",
    "lines": "1-800+",
    "kind": "fmt-delegation",
    "detail": "forward_prefill_fmt — forward_prefill의 PrefillWorkspace 라이브 경로 복제(partition/variance/profiler dead), KV write+attention을 fmt 위임. live arm: norm/QKV/RoPE/O-proj/FFN = backend 직호출",
    "disposition": "migrate-fmt (forward_prefill 삭제 후 forward_prefill_fmt를 \"forward_prefill\" 함수로 rename, fmt 인자 제거 → format 구체화)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/backend/opencl/plan.rs",
    "lines": "1277-1450",
    "kind": "generic-consumer",
    "detail": "execute<C: KVCacheOps> — plan 레이어 루프, 4개 KVCacheOps getter(current_pos/capacity/res_pos/q2_tokens) → dispatch steps. execute_fmt(1945)는 handle.plan_geometry()로 단일 lock 치환",
    "disposition": "dies-with-legacy (execute_fmt로 완전 대체, model_forward.rs 게이트 완성 후 삭제)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/backend/opencl/plan.rs",
    "lines": "1945-2100",
    "kind": "fmt-delegation",
    "detail": "execute_fmt — execute의 byte-identical fork, plan geometry snapshot(1개 lock) 사용, step dispatch 동일",
    "disposition": "migrate-fmt (execute 삭제 후 execute_fmt를 \"execute\" 함수로 rename, fmt 인자 삭제)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/pressure/standard_format.rs",
    "lines": "1-1000",
    "kind": "eval-session",
    "detail": "StandardFormat KVCacheFormat impl — write_kv/attention_into/plan_geometry. 내부 KVCache:KVCacheOps에 위임 (current_pos/capacity/get_view 호출, execute<C> 대체용)",
    "disposition": "removable (execute_fmt 완성 후 내부 getter 호출 유지, 외부 KVCacheOps 게이트 불필요)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/pressure/kivi_format.rs",
    "lines": "1-250",
    "kind": "eval-session",
    "detail": "KiviFormat KVCacheFormat impl — write_kv/attention_into. 내부 KiviCache:KVCacheOps에 위임 (execute<C> 대체용, future: forward_into 호출처 래핑)",
    "disposition": "removable (execute_fmt 완성 후 유지, future family bin이주 시 재평가)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/session/forward/model_forward.rs",
    "lines": "429, 589, 478-495, 555",
    "kind": "plan",
    "detail": "ModelForward::step/prefill — LLMRS_KV_FMT 게이트 기반 분기: (1) ON → execute_plan_fmt + forward_into_fmt fallback, (2) OFF → execute_plan + forward_into fallback (l.429/589 게이트 OFF 폴백)",
    "disposition": "removable (게이트 ON default화 후 OFF 분기 제거, l.429/589/555 → l.484 통합)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/session/prefill.rs",
    "lines": "101-600",
    "kind": "legacy",
    "detail": "run_chunked_prefill — legacy generate.rs 분리된 block, forward_into<Vec<KVCache>> 호출(l.348/446). future: family bin(argus-bench/eval) 이주 또는 forward_into_fmt 래핑",
    "disposition": "migrate-fmt (forward_into → forward_into_fmt로 변경, 또는 family bin 이주 후 삭제)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/session/forward/kivi_forward.rs",
    "lines": "150-200",
    "kind": "legacy",
    "detail": "KiviForward::prefill/step — forward_into<Vec<KiviCache>> 호출(l.157/188). future: family bin 이주 또는 KiviCache → KVCacheFormat wrapper 설계",
    "disposition": "migrate-fmt (Option A: forward_into → KiviFormat 래퍼로 변경, Option B: family bin 이주)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/session/forward/offload_forward.rs",
    "lines": "150-200",
    "kind": "legacy",
    "detail": "OffloadForward::prefill/step — forward_into_offload<Vec<OffloadKVCache>> 호출(l.156/190). forward_into_offload는 구체 monomorphization(BC Step 2, 완료됨)",
    "disposition": "removable (forward_into_offload 그대로 유지, Step 5 scope 밖 — forward_into 레이어만 삭제)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/kv_cache_ops.rs",
    "lines": "1-300+",
    "kind": "trait-def",
    "detail": "KVCacheOps trait — 20개 메서드(current_pos/capacity/layout/kv_heads/head_dim/update/get_view/get_buffers_mut/advance_pos/etc). impl: KVCache/KiviCache/OffloadKVCache",
    "disposition": "dies-with-legacy (Step 5 최종 타겟 — 모든 제네릭 소비자 fmt 완성 후 삭제)"
   }
  ],
  "blockers": [
   "model_forward.rs l.429/589 (LLMRS_KV_FMT OFF fallback) — argus_cli gamepath 완성되어야 legacy 경로 제거 가능",
   "prefill.rs l.348/446 (run_chunked_prefill) — future family bin(argus-bench/eval) 이주 결정 필요 또는 fmt 래핑",
   "kivi_forward.rs l.157/188 (KiviCache forward) — KiviCache → KVCacheFormat wrapper 설계 필요 또는 family bin 이주",
   "forward_gen dead 블록 (partition/fused/KIVI/inline-NEON) — forward_gen_fmt 완성 후 forward_gen 삭제 시 dead 블록도 함께 제거됨",
   "forward_prefill dead 블록 (partition/variance/profiler) — forward_prefill_fmt 완성 후 forward_prefill 삭제 시 dead 블록도 함께 제거됨"
  ]
 },
 {
  "area": "Phase α-K BC Step 5: KVCacheOps trait 폐기 — fmt-side 위임 관계 census (영역 3)",
  "references": [
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/pressure/standard_format.rs",
    "lines": "76, 139, 158, 195, 217, 304, 353, 366, 394",
    "kind": "fmt-delegation",
    "detail": "StandardFormat (KVCache wrapper) → KVCacheOps trait 위임: current_pos (7x line 76/139/158/195/217/304/353), get_view (2x line 366/394). plan_geometry()/write_inner()/attention_into() 메서드 내부에서 inner cache의 KVCacheOps trait 메서드 호출.",
    "disposition": "migrate-fmt (inherent method로 rewire: cache.current_pos 필드 직접 접근, cache.get_view() pub method 호출)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/pressure/kivi_format.rs",
    "lines": "72, 76, 119–124, 168–169",
    "kind": "fmt-delegation",
    "detail": "KIVIFormat (KiviCache wrapper) → KVCacheOps trait 위임: current_pos (3x line 72/121/169), capacity (2x line 76/119), layout (1x line 120), get_view (2x line 124/168). KVCacheFormat impl method들(current_pos/capacity)과 attention_into(prefill/decode arm)에서 inner KiviCache의 trait 메서드 호출.",
    "disposition": "migrate-fmt (inherent method + new pub fn으로 rewire: KiviCache.total_tokens() pub method 추가, awqe_enabled 접근용 pub fn 추가)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/session/eval/fmt_bridge.rs",
    "lines": "129, 137",
    "kind": "fmt-delegation",
    "detail": "EvalCacheKind impl for KiviCache → KVCacheOps trait 위임: current_pos (line 129), needs_attn_scores (line 137). fmt_bridge는 KVCache와 KiviCache의 다형성을 EvalCacheKind trait으로 추상화하되, KiviCache 경우에만 KVCacheOps 경유 (KVCache는 pub 필드로 직접 접근 가능).",
    "disposition": "migrate-fmt (KiviCache pub method 추가 후 직접 호출로 rewire: cur_pos()는 new total_tokens() 호출, needs_scores()는 new is_awqe_enabled() 호출)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/format/kv_cache_format.rs",
    "lines": "1–108",
    "kind": "trait-def",
    "detail": "KVCacheFormat base trait (Phase α-K substep 1): geometry(idx/current_pos/capacity) + mutation(write_kv/write_kv_batch/compact) + attention(attention_into) = 7 methods. KVCacheOps와 **독립적 설계** — format impl이 inner cache의 KVCacheOps를 위임하는 seam이지만, base trait 자체는 storage-format-agnostic (INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC). Step 5 후 KVCacheOps 삭제 시 KVCacheFormat은 무변.",
    "disposition": "removable (trait 자체 불필요 아님; fmt-wrapper 위임만 변경될 뿐 base trait 유지)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/kv_cache_ops.rs",
    "lines": "1–160",
    "kind": "trait-def",
    "detail": "KVCacheOps trait (L2 shared identifier): KVCache/KiviCache 양쪽을 추상화 (~20 method). 단 fmt-side 위임 + legacy consumer (forward_into_offload/run_chunked_prefill) 총 97 reference / 26 file에 산재. Step 5 목표 = trait 전체 삭제 + 97 reference 전부 rewire (inherent method 또는 이미 wiring된 legacy consumer 제거).",
    "disposition": "removable (Step 5 primary target; 위임 끊은 후 전면 삭제 가능)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/pressure/kv_cache.rs",
    "lines": "41–1078 (pub struct + pub fn definitions)",
    "kind": "impl-for-cache",
    "detail": "KVCache impl (KVCacheOps impl 포함): pub field current_pos(직접 접근 가능), pub fn capacity/layout/kv_heads/head_dim/get_view. StandardFormat 위임 시 trait 경유 불가 → inherent method로 바꾸면 됨 (이미 pub method들이 존재).",
    "disposition": "needs-rewire (StandardFormat의 KVCacheOps::current_pos() call를 cache.current_pos 필드/cache.capacity() etc 호출로 변경; KVCacheOps impl 삭제)"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/pressure/kivi_cache.rs",
    "lines": "205–2400+ (pub struct + impl KVCacheOps)",
    "kind": "impl-for-cache",
    "detail": "KiviCache impl (KVCacheOps impl 포함): pub q2_tokens/res_pos, PRIVATE bits/kv_heads/head_dim/awqe_enabled/res_cap. KVCacheOps impl이 total_tokens() private fn + awqe_enabled 필드 접근. KIVIFormat/fmt_bridge 위임 시 trait 경유 불가 → pub inherent method 추가 필요.",
    "disposition": "needs-rewire (new pub fn total_tokens() + new pub fn is_awqe_enabled() 추가; KIVIFormat/fmt_bridge의 KVCacheOps::method() call을 pub fn으로 rewire; KVCacheOps impl 삭제)"
   }
  ],
  "blockers": [
   "KiviCache private state (total_tokens() fn / awqe_enabled field) 접근: fmt_bridge/KIVIFormat이 inner cache의 KVCacheOps 메서드 호출 → pub method 추가 필수 (privacy break-glass, 정당성=fmt wrapper 캡슐화 내 구현 detail)",
   "fmt wrapper 내부 Mutex lock guard 생존 기간: write_inner/attention_into에서 guard를 lock하는 동안 KVCacheOps 메서드 호출 → inherent method로 바꾼 후에도 동일 guard 내 실행되도록 sequencing 필요 (no change, just method call style)",
   "KVCacheFormat base trait과 KVCacheOps의 관계 명확화: Step 3에서 이미 독립적으로 설계됨 (KVCacheFormat ⊥ KVCacheOps) → Step 5는 '위임 경로 끊기'만 할 뿐 base trait 자체는 유지"
  ],
  "summary": "\nKVCacheOps trait의 fmt-side 위임 (census 영역 3)은 3 파일에서 이루어짐:\n\n1. **StandardFormat.rs** (12 위임): inner KVCache에 KVCacheOps::current_pos (7x) / get_view (2x) 호출.\n   - Rewire: cache.current_pos 필드 직접 접근 + cache.get_view() pub method 호출로 대체 가능.\n\n2. **KIVIFormat.rs** (10 위임): inner KiviCache에 KVCacheOps::current_pos (3x) / capacity (2x) / layout (1x) / get_view (2x) 호출.\n   - Rewire: new pub fn total_tokens() (current_pos 대체) + pub fn is_awqe_enabled() + 기존 pub method (capacity/layout/get_view) 호출로 대체.\n\n3. **fmt_bridge.rs** (2 위임, KiviCache only): EvalCacheKind impl for KiviCache에서 KVCacheOps::current_pos / needs_attn_scores 호출.\n   - Rewire: new pub fn total_tokens() + is_awqe_enabled() 호출로 대체.\n\n**핵심 난관**: KiviCache의 private state (total_tokens() private fn, awqe_enabled 필드)를 fmt wrapper가 trait 경유로 접근하고 있음. Step 5는 이 위임을 끊으려면:\n- Option A (추천): KiviCache에 pub fn total_tokens() + pub fn is_awqe_enabled() 추가 → fmt/fmt_bridge에서 직접 호출 → KVCacheOps trait 완전 삭제 (97 reference 모두 rewire).\n- Option B: micro-trait KVCachePrimitive 신설 → 위임 abstraction 유지하되 KVCacheOps 삭제 (trait count 중립).\n\n**KVCacheFormat base trait은 무변**. Step 3에서 이미 KVCacheOps와 독립적으로 설계됨 (supertrait 관계 없음, format impl이 inner cache 접근만 다를 뿐). 위임 끊은 후 format 자체는 불변 (§4.1 INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC 유지).\n"
 },
 {
  "area": "Census 영역 4: eval/session/legacy/tests KVCacheOps 소비자 (Phase α-K BC Step 5)",
  "references": [
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/session/eval/hook.rs",
    "lines": "22-23",
    "kind": "comment-reference",
    "detail": "CacheSnapshot<C> 트레이트 문서 주석에서 KVCacheOps 폐기 언급만 (Phase α-K ①-c 관련). 실제 사용처 없음.",
    "disposition": "dies-with-legacy"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/session/eval/eval_loop.rs",
    "lines": "5",
    "kind": "comment-reference",
    "detail": "파일 헤더 주석: run_eval_ll_generic<C: EvalCacheKind> 경로에서 KVCacheOps 바운드 이미 제거됨. 주석 참조만.",
    "disposition": "dies-with-legacy"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/session/ppl/runner.rs",
    "lines": "256",
    "kind": "unused-import",
    "detail": "Line 256: use 선언만. run_kivi_ppl 함수 본문(376-393)에서 KiviCache::forward_fmt_roundtrip 및 fmt 래퍼 사용. KVCacheOps 메서드 호출 없음.",
    "disposition": "removable"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/session/batch/runner.rs",
    "lines": "723-726",
    "kind": "generic-consumer",
    "detail": "use 선언 + score probe 블록에서 Vec<KVCache> iteration 시 c.current_pos() 호출(line 726). Hook이 &kv_caches (concrete Vec<KVCache>)를 받으므로 generic C 아님. KVCacheOps 트레이트는 형식적만 남음.",
    "disposition": "needs-rewire"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/layer_boundary_hook.rs",
    "lines": "7",
    "kind": "comment-reference",
    "detail": "LayerBoundaryHook 설명 주석에서 KVCacheOps 패턴 참조. 실제 trait 사용처 없음.",
    "disposition": "dies-with-legacy"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/pressure/swap_handler.rs",
    "lines": "301-302",
    "kind": "trait-method-on-concrete",
    "detail": "recall() 함수에서 cache.ensure_capacity(existing + count) 호출. cache는 OffloadKVCache 구체 타입이나 KVCacheOps 트레이트 메서드로 호출.",
    "disposition": "needs-rewire"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/legacy/generate.rs",
    "lines": "4044, 4645",
    "kind": "legacy-generic-template",
    "detail": "Line 4044 (run_kivi_generate 내): use 선언 + KiviCache로 generic 인스턴스화된 forward. Line 4645 (run_with_offload 내): 동일 패턴. 파일 전체 폐기 대상(Step 5-a).",
    "disposition": "dies-with-legacy"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/tests/test_action_pool.rs",
    "lines": "9, 49",
    "kind": "generic-consumer",
    "detail": "use 선언(line 9) + make_seqmajor_cache 함수(line 49): cache.current_pos 필드 직접 할당. KVCacheOps 메서드 아닌 public field 접근.",
    "disposition": "removable"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/tests/spec/test_eng_dat_012_031.rs",
    "lines": "9, 37-56",
    "kind": "generic-consumer",
    "detail": "use 선언(line 9) + make_cache() 반환 KVCache에 대해 current_pos(), capacity(), kv_heads(), head_dim(), memory_usage_bytes(), get_view() 호출. 모두 KVCacheOps 트레이트 메서드.",
    "disposition": "needs-rewire"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/tests/spec/test_eng_alg_020_022.rs",
    "lines": "7, 22-105",
    "kind": "generic-consumer",
    "detail": "use 선언(line 7) + KiviCache 테스트에서 current_pos(), kv_heads(), head_dim(), bits(), memory_usage_bytes(), reset(), q2_tokens(field), res_pos(field), kv_dtype() 호출. 트레이트 메서드 + 필드.",
    "disposition": "needs-rewire"
   },
   {
    "file": "/home/go/Workspace/llm_rs2/engine/src/bin/test_backend.rs",
    "lines": "13, 913-925",
    "kind": "generic-consumer",
    "detail": "use 선언(line 13) + KiviCache 벤치마크에서 current_pos(), q2_tokens(), res_pos() 호출. 모두 트레이트 메서드.",
    "disposition": "needs-rewire"
   }
  ],
  "blockers": [
   "KVCacheOps trait 자체는 monomorphic generic <C: KVCacheOps>에서만 사용 — dyn KVCacheOps trait object 없음",
   "모든 concrete 타입(KVCache/KiviCache/OffloadKVCache)은 inherent 메서드 또는 필드로 전환 가능",
   "Step 5-b (forward_into_offload 이주) 단계: OffloadKVCache는 fmt 래퍼 미지원 → concrete impl 보존 필요",
   "test 파일들의 KVCacheOps 메서드 호출(9개 파일)은 각 cache impl 타입의 inherent 메서드로 변환 필요"
  ],
  "summary": "Census 영역 4에서 발견된 KVCacheOps 참조는 총 11개 지점 (11개 파일). \n\n**분류 결과:**\n- dies-with-legacy (Step 5-a에서 자동 제거): 5개 (eval/hook.rs 주석, eval/eval_loop.rs 주석, layer_boundary_hook.rs 주석, legacy/generate.rs 2곳)\n- removable (불필요 import): 2개 (session/ppl/runner.rs, tests/test_action_pool.rs)  \n- needs-rewire (concrete type inherent 메서드로 변환): 4개 실제 호출처 (session/batch/runner.rs, pressure/swap_handler.rs, tests/spec/test_eng_dat_012_031.rs, tests/spec/test_eng_alg_020_022.rs, src/bin/test_backend.rs 5개 파일)\n\n**핵심 발견:**\n★ dyn KVCacheOps trait object 사용처 NONE — monomorphization만 사용\n★ 모든 실제 소비는 concrete 타입(KVCache/KiviCache/OffloadKVCache)에 대한 메서드/필드 접근\n★ Step 1~4에서 이미 fmt 래퍼(KVCacheFormat 등)로 제거된 경로와 eval 차단 해소(①-c EvalCacheKind) 완료\n★ Step 5-a (legacy/generate.rs 폐기) + Step 5-b (forward_into_offload fmt 이주) 이후 trait 삭제 자체는 가능하나, 각 concrete 타입의 trait 메서드를 inherent로 변환하는 rewire 작업 필요 (9개 호출처)"
 }
]
```

## designs
## 설계 inherent-primitives
Task 갱신은 부수적이므로 설계 결과를 바로 반환하겠습니다. 모든 검증이 완료되었습니다.

---

# Step 5 설계: KVCacheOps trait 폐기 + KVCacheFormat rename

## 0. 핵심 발견 (census 정정 — 설계 전제를 바꾸는 3건)

**[F1] dyn-object 강제 지점 = 0. polymorphism 을 강제하는 것은 generic `<C: KVCacheOps>` 뿐이다.**
모든 소비자가 `&mut [C]` (contiguous concrete slice) 또는 단일 concrete type 으로 monomorphize 됨. `dyn KVCacheOps` 사용처 NONE (census 영역 4 확인). 즉 KVCacheOps 는 "런타임 다형성"이 아니라 "컴파일 타임 코드 공유(KVCache·KiviCache·OffloadKVCache 한 함수로)" 용도. **이 코드 공유를 누가 대체하는가가 설계의 분기점.**

**[F2] fmt 래퍼의 KVCacheOps 위임은 KVCacheFormat base trait 과 무관하다.**
`format/kv_cache_format.rs::KVCacheFormat` 은 이미 7-method base trait 으로 존재하고 KVCacheOps 와 **supertrait 관계 없음**(독립). StandardFormat/KIVIFormat 이 KVCacheOps 를 쓰는 것은 base trait 요구가 아니라 **inner cache(KVCache/KiviCache)에 도달하는 수단**일 뿐이다. 따라서 ADR-0001 의 "rename KVCacheOps→KVCacheFormat" 은 **이미 충족됨** — base trait `KVCacheFormat` 이 그 역할을 한다. 남은 KVCacheOps 는 rename 대상이 아니라 **삭제 대상**이다 (아래 §3에서 ADR 정합).

**[F3] KVCacheOps impl 본문은 거의 전부 inherent thin-forward 다.**
`impl KVCacheOps for KVCache::update` = `self.update(...)` (inherent 호출). KiviCache `current_pos` = `self.total_tokens()`. 즉 trait 메서드의 다수가 이미 존재하는 inherent fn/pub 필드로 위임됨. **단 inherent 가 부재한 trait-only 동작이 있다** (아래 표 `[T-ONLY]` 행) — 이들만 trait→inherent 로 본문을 옮기면 된다.

---

## 1. 전환 전략

### 1.1 본질: 일괄 cutover 다 (additive+gated 불가)

trait 삭제는 정의상 atomic 하다. `KVCacheOps` 정의를 지우는 순간 97 참조/26 파일이 동시에 깨진다. Step 1~3 의 "additive fork + LLMRS_KV_FMT 게이트 OFF" 패턴은 **forward 동작 경로**(hot/cold path 의 실행 흐름)를 두 벌 유지할 때만 성립한다. trait 폐기는 동작 경로가 아니라 **타입 시스템 표면**의 제거라 게이팅이 불가능하다.

단, 일괄 cutover 의 **범위를 최소화**할 수 있다. 핵심은 (B) OLD-chain 의존을 먼저 끊어 cutover 시점에 남은 참조를 "순수 메서드 호출 rewire"로 환원하는 것이다.

### 1.2 두 갈래 선택 — **갈래 A (inherent-only) 채택**

| | 갈래 A: inherent-only (채택) | 갈래 B: micro-trait `KVCachePrimitive` 신설 |
|---|---|---|
| KVCacheOps 메서드 destination | 각 cache 의 inherent `impl` 블록 | 신규 작은 trait 1개 (KVCache·KiviCache impl) |
| fmt 래퍼 위임 | `cache.method()` 직접 inherent 호출 | `KVCachePrimitive::method(cache)` |
| trait 개수 순변화 | −1 (KVCacheOps 삭제) | 0 (KVCacheOps→KVCachePrimitive 개명+축소) |
| 코드 공유(KVCache+KiviCache 한 함수) | **불가** (각 타입 별 메서드) | 가능 (generic `<C: KVCachePrimitive>`) |
| 적합성 | ★ fmt 래퍼가 이미 타입별로 분리됨(StandardFormat↔KVCache, KIVIFormat↔KiviCache). 코드 공유 불필요 | OLD generic forward 가 살아있을 때만 의미 |

**채택 근거**: Step 5 이후 production forward 경로는 **타입별 fmt 래퍼**(StandardFormat·KIVIFormat)로 완전히 분리된다. StandardFormat 은 KVCache 만, KIVIFormat 은 KiviCache 만 본다 — 한 함수가 두 타입을 모두 처리할 generic 수요가 사라진다. micro-trait(갈래 B)은 "generic 으로 코드 공유"가 필요할 때의 가치인데, OLD generic forward 를 모두 삭제하면 그 수요가 0이 된다. 갈래 B 는 CLAUDE.md "요청되지 않은 유연성 금지" 위반(YAGNI). 단 **예외 1건**은 §2.3 참조(테스트 헬퍼의 의도적 generic).

### 1.3 컴파일 차단자 0 을 만드는 순서 (4 phase, 단일 PR 내 sequential commit)

```
S5-P0 (선행): inherent 보강 — additive, 무삭제
  → KVCache/KiviCache/OffloadKVCache 에 [T-ONLY] 메서드를 inherent fn 으로 추가.
    KVCacheOps impl 은 그대로 두고(이중 존재 일시 허용), 신규 inherent 만 추가.
  → 검증: cargo build (KVCacheOps impl 의 self.method() 가 inherent 를 가리키게 됨 — 무회귀)

S5-P1: B-2 OLD-chain 잔여 2 소비자 fmt 이주 (★device 재검증 필수 구간)
  → (1) forward_into_offload → forward_into_offload_fmt (OffloadKVCache: KVCacheFormat impl)
  → (2) run_chunked_prefill → forward_into_fmt 경유 (KiviForward 도 동반)
  → OffloadForward/KiviForward 의 forward_into 호출 제거
  → 검증: S25 OpenCL + Jetson CUDA offload/KIVI bit-identical (Option B device gate)

S5-P2: legacy 폐기 (대량 dead 참조 제거)
  → engine/legacy/generate.rs + legacy_generate bin 삭제 (Cargo.toml [[bin]] 제거)
  → 이 시점에 forward_into<C>/forward_gen<C>/forward_prefill<C>/execute<C> 의
    유일한 production caller 가 모두 사라짐 (model_forward 의 OFF-fallback 제외)
  → 검증: cargo build --workspace (legacy 참조 0 확인)

S5-P3: cutover (atomic) — OLD generic 삭제 + KVCacheOps 삭제 + rename
  → model_forward.rs LLMRS_KV_FMT OFF 분기 제거 (게이트 상수화: 항상 fmt)
  → forward_into<C>/forward_gen<C>/forward_prefill<C>/layer.forward<C>/execute<C> 삭제,
    *_fmt 들을 비-fmt 이름으로 rename
  → KVCacheOps impl 3개 블록 삭제 (본문은 P0 에서 inherent 로 이주 완료)
  → fmt 래퍼/test/swap_handler/batch 의 KVCacheOps:: 호출 → inherent 호출 rewire
  → kv_cache_ops.rs: KVCacheOps trait 삭제. KVLayout/KiviRawBuffers 만 잔존 →
    파일을 kv_layout.rs 로 rename 하거나 format/ 로 이동 (§3)
  → 검증: cargo build (차단자 0) + host test + S25/Jetson 최종 bit-identical
```

**차단자 0 의 메커니즘**: P3 의 trait 삭제가 atomic 이지만, P0 에서 inherent 를 미리 깔아 두면 trait 삭제 직후 모든 호출처가 inherent 로 1:1 치환 가능하다. P1/P2 가 generic-consumer 를 먼저 제거해 P3 의 rewire 표면을 "fmt 래퍼 위임 + 비-forward 잡호출(swap/batch/test)"로 축소한다.

---

## 2. 메서드별 목적지 표

분류: **[INH]** inherent 이미 존재(thin-forward) → impl 본문 그대로 inherent 로. **[T-ONLY]** trait 에만 존재 → P0 에서 inherent 신설. **[FIELD]** pub 필드라 메서드 불요. **[DEAD]** default no-op, 해당 cache 가 미구현 → 삭제.

### 2.1 KVCache (StandardFormat 가 소비)

| 메서드 | 현 상태 | 목적지 | 비고 |
|---|---|---|---|
| `current_pos` | [FIELD] `self.current_pos` pub | inherent `fn current_pos(&self)` 신설 (또는 필드 직접) | StandardFormat 은 `KVCacheOps::current_pos(&cache)` 호출 중 → inherent fn 신설이 호출처 변경 최소 |
| `set_current_pos` | [T-ONLY] | inherent fn 신설 (본문 = pos 대입 + pos==0 시 high_water reset) | compact 에서 사용 |
| `capacity` | [INH] `self.capacity()` 존재 | 그대로 | |
| `kv_heads`/`head_dim`/`layout` | [INH] 존재 | 그대로 | |
| `kv_dtype` | [T-ONLY] = `self.k_buffer.dtype()` | inherent fn 신설 | write_inner 에서 사용 |
| `memory_usage_bytes` | [INH] 존재 | 그대로 | trait 은 inherent 재호출 |
| `update` | [INH] 존재 | 그대로 | |
| `get_view(&mut)` | [T-ONLY] (inherent 는 `get_view(&self,seq_len)` 시그니처 다름) | inherent `fn view(&self)->(Tensor,Tensor)` 신설 (이름 충돌 회피) | KVCache 는 `(k_buffer.clone, v_buffer.clone)` — `&self` 로 가능. 신규 이름 `view` 권장 |
| `get_buffers_mut` | [T-ONLY] = `Some((&mut k,&mut v))` | inherent fn 신설 | write_inner GPU scatter |
| `advance_pos` | [T-ONLY] | inherent fn 신설 | plan_advance/write_inner |
| `ensure_capacity` | [T-ONLY] | inherent fn 신설 | write_inner |
| KIVI 5종(res_pos/q2_tokens/res_cap/needs_flush/flush_if_needed) | [DEAD] default no-op | 삭제 (KVCache 미사용) | |
| needs_attn_scores/set_attn_scores/get_kivi_raw_buffers | [DEAD] default | 삭제 | |

### 2.2 KiviCache (KIVIFormat + fmt_bridge 가 소비) — 가장 무거움

| 메서드 | 현 상태 | 목적지 | 비고 |
|---|---|---|---|
| `current_pos` | [T-ONLY] = `self.total_tokens()` (private) | inherent `fn current_pos(&self)` 신설 (= total_tokens) | fmt_bridge 의 ①-c 수용 잔여 해소점 |
| `set_current_pos` | [T-ONLY] no-op | inherent no-op fn (또는 호출처 제거) | KIVI position 파생 |
| `capacity` | [T-ONLY] 조건부(bits=16 GPU) | inherent fn 신설 | |
| `kv_heads`/`head_dim` | [T-ONLY] | inherent fn 신설 | `bits`/`reset` 은 이미 inherent 존재 — 동일 패턴 |
| `layout` | [T-ONLY] 조건부 | inherent fn 신설 | |
| `kv_dtype` | [T-ONLY] = F32 | inherent fn 신설 | |
| `memory_usage_bytes` | [T-ONLY] | inherent fn 신설 | |
| `update`/`get_view(&mut)` | [T-ONLY] | inherent fn 신설 (get_view 는 `&mut` 유지 — assemble 필요) | |
| `get_buffers_mut`/`advance_pos` | [T-ONLY] | inherent fn 신설 | |
| `needs_attn_scores` | [T-ONLY] = `self.awqe_enabled` | inherent `fn needs_attn_scores(&self)` (또는 `is_awqe_enabled`) | fmt_bridge `needs_scores` 위임 해소 |
| `set_attn_scores`/`get_kivi_raw_buffers` | [T-ONLY] | inherent fn 신설 | |
| `res_pos`/`q2_tokens`/`res_cap`/`needs_flush`/`flush_if_needed` | [T-ONLY] | inherent fn 신설 | plan execute seam |

**KiviCache 는 inherent 가 거의 전무**(`bits`/`reset`/`set_awqe_enabled`만). P0 에서 ~18개 inherent fn 신설이 작업량의 대부분. 단 본문은 KVCacheOps impl 에서 그대로 복사 — 위험 낮음.

### 2.3 OffloadKVCache (forward_into_offload_fmt 가 소비)

| 메서드 | 목적지 | 비고 |
|---|---|---|
| 13개 override 전부 | inherent fn 신설 + **`impl KVCacheFormat for OffloadKVCache` 신설** (P1) | Option B: interior-mut(`&self`) + preload pool aliasing 재설계 필요 (★device 재검증 구간) |

OffloadKVCache 는 갈래 A 의 예외처럼 보이나, fmt 이주(P1)로 `KVCacheFormat` impl 을 얻으면 forward_into_offload_fmt 가 `Arc<dyn KVCacheFormat>` 슬라이스로 통합되어 OLD generic 불요. swap_handler 의 `cache.ensure_capacity()` (영역4) 는 OffloadKVCache inherent 호출로 rewire.

### 2.4 비-forward 잡소비자 (P3 rewire)

| 위치 | 호출 | rewire |
|---|---|---|
| `swap_handler.rs:301` | `cache.ensure_capacity()` (OffloadKVCache) | inherent 호출 + `use KVCacheOps` 제거 |
| `batch/runner.rs:726` | `c.current_pos()` (Vec\<KVCache\>) | inherent 호출 + `use` 제거 |
| `fmt_bridge.rs:129,137` | `KVCacheOps::current_pos/needs_attn_scores(KiviCache)` | inherent 호출 (①-c 잔여 정식 해소) |
| test 5파일(test_eng_dat/alg, test_backend, test_action_pool, ppl runner) | 메서드/필드 혼합 | inherent 호출 rewire; test 헬퍼가 generic 이면 갈래 A 예외로 작은 test-local trait 허용 가능하나, 대부분 concrete 라 불요 |

---

## 3. ADR-0001 정합 — rename 의 정체

ADR-0001 line 10/124 은 "코드 trait 명 `KVCacheOps` → `KVCacheFormat` 동행 rename"을 명시한다. **그러나 실제 코드는 ADR 작성 후 분기했다**:

- ADR 비전: KVCacheOps(15 method) 를 **축소·개명**하여 KVCacheFormat(5 method) 으로 만든다 (= 동일 trait 의 진화).
- 실제 구현: `format/kv_cache_format.rs::KVCacheFormat`(7 method base trait) 이 **별도 신설**되었고, KVCacheOps 는 그것과 독립으로 남았다. fmt 래퍼가 양쪽 모두 구현/소비.

→ **정합 결론**: ADR 의 "rename" 은 *명칭 이전*이 아니라 *역할 이전*으로 이미 실현되었다. `KVCacheFormat` base trait 이 ADR 이 의도한 "Trait object dispatch 표면"이고, `KVCacheOps` 는 그 표면이 inner cache 에 닿기 위한 **임시 가교(generic 코드 공유)** 였다. Step 5 는 이 가교를 inherent 로 내재화하고 KVCacheOps 를 삭제한다 — 이것이 ADR §8 item 1("kv_cache_ops.rs:53 정책 주석 제거")과 §6 종료 게이트의 실제 의미.

**rename 실행 항목 (P3)**:
1. `KVCacheOps` trait 정의 삭제. **`KVCacheFormat` 으로 rename 하지 않는다** — 이름은 이미 base trait 이 점유. 중복 rename 은 충돌.
2. `kv_cache_ops.rs` 파일: `KVLayout`·`KiviRawBuffers` 만 잔존 (이들은 inherent 메서드 시그니처가 계속 사용). 파일을 `kv_layout.rs` 로 rename 하거나 `format/` 로 이동 — no-mod.rs 컨벤션상 `format/kv_cache_format.rs` 옆 `format/kv_layout.rs` 가 자연. (KiviRawBuffers 는 KiviCache 옆 `pressure/` 가 응집도상 더 맞으나, 현 import 표면 변경 최소화 우선 시 kv_cache_ops.rs 잔존+개명이 안전.)
3. ADR-0001 §6 종료 게이트 충족 배너 + spec INV-KVCACHELAYER-* 안정 키 유지(rename 영향 없음 — INV ID 는 추적 키).

---

## 4. 위험 (RPN = 심각도×발생×미검출, ≥100 강조)

| ID | 위험 | RPN | 완화 |
|---|---|---|---|
| **R1** | **OffloadKVCache fmt 이주(P1, Option B)의 device 회귀**: interior-mut(`&self`) 전환 + preload pool aliasing 재설계가 OLD `&mut [C]` 의미를 깨뜨림. preload 버퍼 aliasing 은 `&self` 하에서 borrow 규칙 충돌 가능 → unsafe/Mutex 필요. **host GPU 부재라 host 검증 불가** | **9×4×5=180** | P1 을 독립 commit + S25/Jetson 양쪽 offload bit-identical 게이트 필수. 회귀 시 P1 만 revert(P0/P2/P3 무관). Option B 설계는 별도 cut 문서로 분리 권장 |
| **R2** | **KiviCache inherent 18개 신설 시 본문 미세 변형**: 복사 중 조건부 로직(bits=16 GPU capacity/layout) 손상 → KIVI silent garbage | 7×3×6=**126** | P0 은 KVCacheOps impl 본문을 **기계적 복사**(diff 로 byte-identical 확인). KVCacheOps impl 을 즉시 삭제하지 말고 P3 까지 잔존시켜 "impl 이 inherent 를 호출"하는 형태로 두면 P0 단독 회귀 0 |
| **R3** | `get_view` 이름 충돌(KVCache inherent `get_view(&self,seq_len)` ↔ trait `get_view(&mut self)`) → rewire 시 잘못된 오버로드 선택 | 6×4×4=96 | trait 변형을 `view(&self)` 또는 `attention_view(&mut)` 로 **다른 이름**으로 신설. 기존 inherent `get_view(&self,seq_len)` 은 무변(별도 caller 존재 여부 P0 에서 grep 확인) |
| **R4** | **LLMRS_KV_FMT OFF 분기 제거(P3)가 fallback 경로의 숨은 의존 노출**: model_forward.rs:429/589 OFF-fallback 이 plan 미빌드/invalidation 시 forward_into<C> 로 강하하는데, fmt-only 화 후 그 강하 대상이 forward_into_fmt 로 바뀜 — fmt 경로가 plan-invalidation 케이스를 커버하는지 미검증 | 7×3×5=**105** | P3 전에 forward_into_fmt 의 plan-fallback arm 이 OFF 경로와 동치인지 확인(Step 3 가 이미 fmt-ON 에서 검증했으나 ON default 화는 별개). S25 에서 plan-invalidation 유발 시나리오(긴 context) bit-identical |
| R5 | `kv_cache_ops.rs` import 표면(26 파일 `use crate::kv_cache_ops::...`) 의 KVLayout/KiviRawBuffers 잔존 처리 — 파일 rename 시 26 파일 import 경로 동시 수정 | 4×4×3=48 | P3 마지막. rename 을 생략하고 파일명 유지(KVCacheOps 만 삭제) 도 허용 — ADR 정합상 파일 rename 은 선택. import 변경 최소화 우선 시 KVCacheOps 만 삭제하고 파일은 잔존 |
| R6 | test 파일 generic 헬퍼가 갈래 A 에서 컴파일 불가(여러 cache 타입 받던 generic) | 4×3×3=36 | 대부분 concrete(영역4 확인). generic 잔존 시 test-local 작은 trait 1개 허용(production 무오염) |
| R7 | argus_cli happy-path 전용 — KIVI/offload/swap/batch 모드가 legacy 폐기 후 진입점 상실. family bin(argus-chat/eval/bench) 미구현 상태 | 5×5×2=50 | **Step 5 범위 정정 필요**: KIVIFormat/OffloadKVCache fmt 이주는 하되, 그 production 진입점(family bin)은 별도 트랙. P1 의 fmt 이주는 "코드 보존+컴파일 가능"이 목표지 production 발화가 아님. 사용자 결정 필요 — drop vs family bin 이주 |

**최상위 위험 R1(180)**: forward_into_offload 의 Option B fmt 이주가 Step 5 전체에서 유일한 "진짜 설계+device 재검증" 구간이다. 나머지(P0/P2/P3)는 기계적 rewire 라 위험이 낮다. **권고: P1(offload/kivi fmt 이주)을 Step 5 본체에서 분리해 선행 전용 증분(별도 cut 문서 + device gate)으로 처리**하고, P0/P2/P3(trait 삭제 본체)는 그 후 단일 PR. 이렇게 하면 trait 삭제 PR 은 device 위험 0(host-verifiable rewire)이 된다.

---

## 5. 미해결 결정사항 (사용자/PM 확인 필요)

1. **R7 — KIVI/offload/swap/batch 의 production 진입점**: legacy 폐기 후 이 모드들은 argus_cli 에서 reject 됨. Step 5 의 P1 fmt 이주를 (a) "코드 보존만, 발화는 family bin 트랙으로 연기" 할지, (b) 이 모드들을 아예 drop(코드 삭제)할지. **drop 이면 P1 자체가 불필요해지고 R1(180) 이 소멸** — Step 5 가 순수 host rewire 로 축소된다. 이 결정이 Step 5 난이도를 좌우한다.
2. **§3 R5 — `kv_cache_ops.rs` 파일 운명**: KVCacheOps 만 삭제(파일·import 표면 유지) vs `format/kv_layout.rs` 로 rename(26 파일 import 수정). ADR 정합은 둘 다 충족. 최소 변경 = 전자.

## 6. 결론

- **갈래 A (inherent-only)** 채택 — fmt 래퍼가 이미 타입별 분리라 generic 코드 공유 수요 0.
- **일괄 cutover** 불가피(trait 삭제는 atomic)하나, P0(inherent 선보강)+P1(OLD-chain 잔여 제거)+P2(legacy 폐기)로 P3 cutover 의 rewire 표면을 "fmt 위임 + 잡호출"로 축소 → 차단자 0.
- **ADR rename 은 이미 실현됨** — `KVCacheFormat` base trait 이 ADR 의 dispatch 표면. KVCacheOps 는 rename 대상이 아니라 삭제 대상. KVCacheFormat 으로의 중복 rename 금지.
- **최대 위험 = forward_into_offload Option B fmt 이주(R1=180, device-only)**. R7 결정(KIVI/offload drop 여부)에 따라 이 위험이 소멸 가능 — Step 5 진입 전 사용자 확인 권장.

관련 파일(절대경로): `/home/go/Workspace/llm_rs2/engine/src/kv_cache_ops.rs`, `/home/go/Workspace/llm_rs2/engine/src/format/kv_cache_format.rs`, `/home/go/Workspace/llm_rs2/engine/src/pressure/kv_cache.rs:1006`, `/home/go/Workspace/llm_rs2/engine/src/pressure/kivi_cache.rs:2237`, `/home/go/Workspace/llm_rs2/engine/src/pressure/offload.rs:263`, `/home/go/Workspace/llm_rs2/engine/src/pressure/standard_format.rs`, `/home/go/Workspace/llm_rs2/engine/src/pressure/kivi_format.rs`, `/home/go/Workspace/llm_rs2/engine/src/session/eval/fmt_bridge.rs:127-139`, `/home/go/Workspace/llm_rs2/engine/src/models/transformer.rs:1509,2034,3717`, `/home/go/Workspace/llm_rs2/engine/src/backend/opencl/plan.rs:1277,1938`, `/home/go/Workspace/llm_rs2/docs/adr/0001-kv-dispatch-paradigm.md:10,124`, `/home/go/Workspace/llm_rs2/.agent/todos/roadmap_alpha_k_bc_completion_2026_06_04.md` §Step5.

## 설계 offload-optionB
모든 핵심 정보를 확보했다. 결정적 발견을 정리한다:

1. **OffloadKVCache 의 모든 핵심 메서드는 `&mut self`** — `update`(296), `get_view`(376, store load + GPU upload + preloaded reset), `preload`/`release_buffers`/`retain_preload`(lifecycle). 반면 `KVCacheFormat::write_kv`/`attention_into` 는 `&self`. 따라서 OffloadKVCache 를 KVCacheFormat 으로 만들려면 `Mutex<OffloadKVCacheInner>` interior-mut 필수.

2. **raw pointer aliasing 의 진짜 문제** — forward_into_offload 는 `caches_ptr: *mut OffloadKVCache` 를 (a) main thread 가 `&mut *caches_ptr.add(i)` 로 forward 중 `update`+`get_view` 호출, (b) preload pool worker 가 `caches_ptr.add(far_idx) as *mut ()` + `preload_erased::<OffloadKVCache>` 로 `(*ptr).preload()` 호출. 안전성은 "far_idx != i, 한 element 는 한 thread 만" 으로 보장. `preload_erased` 는 **concrete 타입 monomorphization** 이라 `*mut OffloadKVCache` 가 필수.

3. **Mutex 로 감싸면 aliasing 모델이 근본적으로 바뀐다** — `Arc<dyn KVCacheFormat>` 핸들은 `&self` 라 raw `*mut` 불요지만, preload(`&mut self`)는 KVCacheFormat 표면에 없다 → preload 경로는 여전히 별도. Mutex 를 쓰면 main thread 의 attention_into(lock 잡음)와 preload worker(같은 Mutex lock 시도)가 **다른 layer 라도 동일 타입 핸들** 일 때 실제 lock 경합은 없으나(서로 다른 Arc), preload 가 `&self` 가 아니면 Arc 핸들로 호출 불가.

이제 설계를 종합한다.

---

# Step 5(b) offload OLD-chain fmt 이주 — Option B 설계

## 0. 결론 요약 (TL;DR)

**Option B 를 "OffloadKVCache 를 KVCacheFormat 으로 직접 구현"으로 좁게 정의하면 성립하지 않는다.** forward_into_offload 는 forward_into_fmt 와 **본문이 완전히 다른 별개 forward**(preload pool + prefetch adaptive depth + retain/release cross-token lifecycle + 자체 layer loop)이고, 그 lifecycle 메서드(preload/release/retain)는 KVCacheFormat 표면에 없으며 전부 `&mut self` 다. 따라서 forward_gen_fmt/forward_prefill_fmt 를 **그대로 재사용할 수 없다** — write/attention 두 지점만 fmt 위임이고, 나머지 lifecycle 골격은 offload 전용으로 남는다.

**권장 설계 = Option B′ (하이브리드)**: 
- OffloadKVCache 를 `Mutex<...>` 없이, **interior-mut 를 최소 표면(write_kv/attention_into 가 만지는 store/attn_buf/gpu_buf/current_pos)에만** 도입한 `OffloadFormat` wrapper 로 감싼다.
- forward_into_offload 의 **layer loop 골격(preload pool, prefetch, retain/release)은 보존**하되, 내부 `layer.forward(...)` 호출만 `layer.forward_gen_fmt`/`forward_prefill_fmt` 로 교체.
- preload pool 의 raw-pointer aliasing 은 **그대로 유지** — preload 는 KVCacheFormat 이 아니라 `PrefetchableCache`(Step 2 standalone) 경로로 계속 `&mut OffloadKVCache` concrete monomorphization.

즉 **forward_into_offload 는 살아남되 generic `<C: KVCacheOps>` layer chain 의존만 끊는다.** 이것이 roadmap §Step 5(b) 가 요구하는 "forward_gen<C>/forward_prefill<C> + impl KVCacheOps for OffloadKVCache 소비 제거" 의 정확한 달성 방법이다.

---

## 1. 왜 단순 Option B(OffloadKVCache: KVCacheFormat)가 안 되는가

| 항목 | forward_into_fmt (Standard) | forward_into_offload |
|---|---|---|
| layer loop | 단순 순차 16 layer | preload pool + adaptive prefetch depth + cross-token retain/release |
| cache 핸들 | `&fmts[i]: &Arc<dyn KVCacheFormat>` (`&self`) | `&mut *caches_ptr.add(i)` (`&mut`, raw ptr aliasing) |
| 동시성 | 없음 | preload worker thread 가 `caches_ptr.add(far_idx)` 를 동시 접근 |
| lifecycle | 없음 | preload/release_buffers/retain_preload (전부 `&mut self`, KVCacheFormat 표면 밖) |
| write/attention | `fmt.write_kv` / `fmt.attention_into` | `kv_cache.update` / `kv_cache.get_view`+`backend.attention_gen` (generic) |

forward_gen_fmt 는 write/attention 두 지점만 fmt 위임하고 나머지(norm/QKV/RoPE/FFN)는 backend 직호출이다 (forward_gen_fmt.rs:139, 166). **이 두 지점은 offload 에도 그대로 적합**하다. 문제는 lifecycle 이다 — forward_gen_fmt 는 preload/retain/release 를 전혀 모른다. 따라서:

- **재사용 가능**: forward_gen_fmt / forward_prefill_fmt 의 layer 본문 (write_kv + attention_into 위임 포함).
- **재사용 불가**: forward_into_fmt 의 layer loop (preload pool 없음). → forward_into_offload 의 loop 골격은 유지.

---

## 2. interior-mut 재설계 (OffloadFormat)

### 2.1 표면 분리: forward 가 만지는 상태 vs prefetch 가 만지는 상태

OffloadKVCache 의 mutable 상태를 두 그룹으로 나눈다:

- **forward 그룹** (write_kv / attention_into 가 만짐): `current_pos`, `store`, `attn_k_buf`/`attn_v_buf`, `preloaded`, `store_behind`, `out_*_buf`, `gpu_*_buf`. → `&self` 로 호출돼야 함.
- **prefetch 그룹** (preload pool worker 가 만짐): preload/release_buffers/retain_preload — 이것도 동일 `attn_buf`/`preloaded`/`store` 를 만진다.

→ **두 그룹이 동일 필드를 공유**한다 (attn_buf, preloaded, store). 따라서 forward(main thread)와 preload(worker thread)가 **다른 layer 일 때만** 안전하다는 기존 불변식이 그대로 필요. Mutex 로 감싸도 이 불변식은 깨지지 않지만(같은 layer 를 두 thread 가 동시에 만지지 않음), Mutex 는 **forward 경로의 `&self` 요구만 충족**시킬 뿐이다.

### 2.2 권장: `RefCell` 불가 → `Mutex` 또는 `&mut` 우회

`KVCacheFormat: Send + Sync` 이므로 `RefCell` 불가(StandardFormat 도 같은 이유로 Mutex 사용, standard_format.rs:8). 두 후보:

**후보 A — `Mutex<OffloadKVCache>` 단일 lock (StandardFormat 패턴 모방)**
```
pub struct OffloadFormat {
    idx: usize,
    inner: Mutex<OffloadKVCache>,   // 기존 OffloadKVCache 무변
}
impl KVCacheFormat for OffloadFormat {
    fn write_kv(&self, k, v, be) -> Result<()> { self.inner.lock().unwrap().update(...) }
    fn attention_into(&self, q, be, out, dims, scores) -> Result<()> {
        let mut g = self.inner.lock().unwrap();
        let (kc, vc) = g.get_view();          // &mut self OK (lock guard)
        be.attention_gen(q, &kc, &vc, out, ...) // SeqMajor, window clamp
    }
    fn current_pos(&self) -> usize { self.inner.lock().unwrap().current_pos() }
    fn capacity(&self) -> usize { ... }
    fn compact(...) -> Result<()> { bail!("offload: eviction 미지원") } // on_kv_prune no-op 과 일치
}
```
- **장점**: OffloadKVCache 본문 1바이트 무변. update/get_view 의 `&mut self` 가 lock guard 안에서 그대로 성립.
- **단점**: preload pool worker 가 `OffloadFormat` 의 inner 를 `&mut` 로 만지려면 — preload 는 KVCacheFormat 에 없으니 `OffloadFormat::preload(&self)` 인헤런트 메서드를 추가하고 그 안에서 `self.inner.lock().unwrap().preload()`. **그러나 preload_erased 는 raw `*mut` 로 lock 우회 중** → Mutex 를 도입하면 preload worker 도 같은 Mutex 를 lock 해야 함. main thread 의 attention_into(layer i, lock 잡음)와 preload worker(layer far_idx, **다른 OffloadFormat** 의 lock)는 서로 다른 Mutex 라 경합 0. **단, prefetch 가 `&Arc<OffloadFormat>` 를 thread 로 보내려면 Arc clone + Send 필요** → Mutex<T: Send> 는 Sync 라 OK.

**후보 B — Mutex 없이, forward_into_offload 가 `&mut OffloadKVCache` 를 직접 들고 attention 만 free fn 으로**
- KVCacheFormat 을 아예 구현하지 않고, forward_gen_fmt 가 받는 `fmt: &Arc<dyn KVCacheFormat>` 대신 offload 전용 **별도 args**(`ForwardGenOffloadArgs { kv_cache: &mut OffloadKVCache, ... }`)를 만들어 write/attention 을 concrete 호출.
- **이것은 fmt 이주가 아니다** — KVCacheOps 의존만 OffloadKVCache 인헤런트 메서드로 바꾸는 것 (current_pos/update/get_view 를 inherent 로). roadmap §Step 5(b) "forward_gen<C> 소비 제거" 는 달성하지만 "fmt 이주" 는 아님.

### 2.3 권장 = 후보 A (Mutex<OffloadKVCache>), 단 preload 는 fmt 우회 유지

후보 A 가 SSOT 의 "Option B: OffloadKVCache: KVCacheFormat interior-mut" 의도에 가장 부합하고 StandardFormat 선례와 일관된다. **핵심 결정**: 
- write_kv/attention_into/current_pos/capacity/compact 5개는 KVCacheFormat 으로 (lock).
- **preload/release_buffers/retain_preload 는 KVCacheFormat 에 넣지 않는다** — `OffloadFormat` 인헤런트 메서드 또는 별도 `PrefetchableCache` impl 로 둔다. 단 preload_erased 의 monomorphization 대상이 `OffloadKVCache` → `OffloadFormat` 으로 바뀐다 (raw ptr 은 `*mut OffloadFormat`).

---

## 3. preload/prefetch 경로와 fmt 의 상호작용 (Step 2 standalone 의 여파)

Step 2 에서 `PrefetchableCache` 가 `KVCacheOps` supertrait 를 떼고 standalone 이 됐다 (kv_cache.rs:22). 이게 Step 5(b) 를 **크게 단순화**한다:

- preload pool 은 `preload_erased::<C: PrefetchableCache>` (preload_pool.rs:177) 로 동작 — KVCacheFormat 와 무관.
- **OffloadFormat 가 `PrefetchableCache` 를 구현**하면 (preload → `self.inner.lock().preload()`), `preload_erased::<OffloadFormat>` 으로 그대로 raw-ptr submit 가능.
- transformer.rs:3813/3844 의 `preload_erased::<OffloadKVCache>` → `preload_erased::<OffloadFormat>` 단순 치환.
- caches_ptr 타입 `*mut OffloadKVCache` → `*mut OffloadFormat`.

**상호작용 위험 (핵심)**: preload worker(`OffloadFormat::preload` → `inner.lock()`)와 main thread forward(layer i 의 `attention_into` → 같은 layer i 의 `inner.lock()`)가 **동일 layer 의 Mutex 를 동시에 잡을 가능성**. 기존 불변식은 "far_idx = i + depth, depth ≥ 1, 따라서 far_idx ≠ i" 로 layer 충돌을 막는다. Mutex 도입 후에도 **이 불변식이 유효하면 lock 경합 0**(서로 다른 OffloadFormat 인스턴스 = 서로 다른 Mutex). 단 release_buffers(`i-1`)도 main thread 가 호출하므로 layer i-1 의 preload 가 이미 collect 된 뒤(transformer.rs:3897 `i>0 && (i-1)>=depth`) 호출 → 순서 보존됨.

**→ Mutex 는 lock 경합을 만들지 않는다(layer 분리 불변식이 그대로 성립). 단 Mutex 가 `Send + Sync` 를 제공해 raw-ptr `unsafe impl Send for PreloadTask` 의 안전성 근거를 강화한다 — 오히려 안전성이 개선된다.**

---

## 4. chat/session.rs:547 (build_chat_offload) 영향

`build_chat_offload` 가 `alloc_offload_kv_caches` 로 `Vec<OffloadKVCache>` 를 만들어 `OffloadForward::new` 에 넘긴다 (session.rs:529/547). 영향:

- `alloc_offload_kv_caches` (offload_forward.rs:265) 반환 타입 `Vec<OffloadKVCache>` → `Vec<OffloadFormat>` 변경. 내부에서 `OffloadKVCache::new(...)` 후 `OffloadFormat::new(layer_id, cache)` 로 wrap. `set_gpu_backend` 는 wrap **전**에 inner 에 호출하거나 OffloadFormat 에 위임 메서드 추가.
- `OffloadForward.kv_caches: Vec<OffloadKVCache>` → `Vec<OffloadFormat>` (offload_forward.rs:39).
- `OffloadForward::reset_kv` (offload_forward.rs:224) `cache.reset_session()` → OffloadFormat 위임 메서드 또는 `with_inner_mut`.
- `kv_caches_mut()` (offload_forward.rs:105) 시그니처 변경.
- **build_chat_offload 자체는 alloc 함수 시그니처만 따라가면 무변** — caller 영향 최소.

**중요**: OffloadForward::prefill/step 이 `forward_into_offload` 에 `&mut self.kv_caches` 를 넘기는 구조(offload_forward.rs:160/194)는 유지. forward_into_offload 시그니처가 `Vec<OffloadKVCache>` → `Vec<OffloadFormat>` 으로만 바뀐다.

---

## 5. 이주 단계 (additive + 게이트, Step 1~3 선례)

Step 1~3 의 `LLMRS_KV_FMT` OFF-default + copy-fork 선례를 따른다. 단 forward_into_offload 는 **production 유일 경로**(OFF fallback 이 없는 단일 forward)라 게이트 전략이 다르다:

| substep | 작업 | 검증 |
|---|---|---|
| **b-0** | `OffloadFormat` 신설 (`pressure/offload_format.rs`, no-mod.rs). `Mutex<OffloadKVCache>` + KVCacheFormat impl (write/attn/geometry/compact-bail) + PrefetchableCache impl + set_gpu_backend/reset_session/store 위임. **purely additive, unwired.** | host unit test: write_kv→attention_into round-trip == OffloadKVCache 직접 (CPU bit-identical). compact bail 확인. |
| **b-1** | `forward_into_offload_fmt` **copy-fork** 신설 (transformer.rs). loop 골격(preload pool/prefetch/retain/release) 보존, `layer.forward(LayerForwardArgs{kv_cache})` → `layer.forward_gen_fmt`/`forward_prefill_fmt(fmt: &fmts[i])` 교체. `preload_erased::<OffloadFormat>`. **OFF-default env 게이트 `LLMRS_OFFLOAD_FMT`** 로 OffloadForward 가 분기. 기존 forward_into_offload 무변. | host CPU: `--kv-mode offload --kv-type f16` greedy N=32 ON vs OFF md5 동일. |
| **b-2** | OffloadForward / alloc_offload_kv_caches / build_chat_offload 를 `Vec<OffloadFormat>` 로 전환 (게이트 ON 경로). | host: chat offload smoke + reset_kv. |
| **b-3 (device)** | **device 게이트** (§6) 통과 후 게이트 제거 + 기존 forward_into_offload + `impl KVCacheOps for OffloadKVCache` 삭제. | S25/Jetson bit-identical + TBT (§6). |
| **b-4** | KVCacheOps trait 삭제 (Step 5(c)) — offload 가 마지막 소비자였다면 여기서 trait 제거 + KVCacheFormat rename. | 전체 `grep KVCacheOps engine/` = 0. |

**run_chunked_prefill(prefill.rs:348/446)는 Step 5(b) 의 별개 소비자** — `Vec<KVCache>` + forward_into(generic) 소비. 이건 offload 아니라 **Standard** 경로라 forward_into_fmt 로 이주 (①-d 의 batch/ppl site 와 동형, profiler/variance_collector 가 살아있는 점만 추가 고려). offload 와 분리해 별도 substep 으로 다룬다 (D2 범위는 offload 만).

---

## 6. device GPU 재검증 게이트 정의 (S25 / Jetson)

forward_into_offload 의 GPU 경로는 `OffloadKVCache::get_view` 가 store→`gpu_*_buf` 로 `write_buffer_range` upload 후 `backend.attention_gen` 호출(offload.rs:437-482). 이게 fmt 이주 후 `OffloadFormat::attention_into` 안의 lock guard 에서 동일하게 실행돼야 한다.

**게이트 매체**: argus_cli 가 offload 를 reject 하므로 (happy-path 전용), **device 게이트는 family bin 또는 legacy_generate** 를 매체로 한다. legacy_generate 가 Step 5(a)에서 폐기되므로 **순서 의존**: device 게이트(b-3)를 **Step 5(a) legacy 폐기 이전**에 수행하거나, offload 를 지원하는 family bin(argus-chat/argus-bench)을 먼저 확보해야 한다.

**게이트 정의**:
- **장비**: S25 `--backend opencl --opencl-rpcmem` + Jetson `--backend cuda`.
- **모델/모드**: Qwen2.5-1.5B, `--kv-mode offload --offload-mode raw`, F16 KV (default). disk 모드는 별도 smoke(선택).
- **bit-identical**: `LLMRS_OFFLOAD_FMT=1` (ON) vs unset (OFF), greedy(temp=0) 32 tokens. **첫 토큰 logit + 32 토큰 텍스트 완전 일치**. 동일 프롬프트.
  - 단 forward_gen_fmt 의 알려진 carve-out 적용: **F32 KV + host-mapped 버퍼는 inline-NEON vs attention_gen FP 누산 순서 상이로 NOT bit-identical**(forward_gen_fmt.rs:16-19). offload 는 get_view 가 항상 SeqMajor + `backend.attention_gen` 으로 가므로(offload.rs / OffloadFormat::attention_into) **OLD forward_into_offload 도 동일하게 attention_gen 경로** → F16/F32 모두 bit-identical 기대. **단 prefill arm 은 prefill_attention 으로 가므로 OLD forward_prefill 의 attention 블록과 동일 식인지 device 에서 확인**(SeqMajor flash, q_start_pos).
- **avg_tbt**: ON vs OFF Δ ≤ +3% (Step 3 선례). Mutex lock 2회/layer(write_kv + attention_into) + current_pos lock 추가분이 prefetch I/O 대비 무시 가능해야 함. n=3 median, wall-clock(`--profile` 금지).
- **회귀 시**: `LLMRS_OFFLOAD_FMT` OFF 가 default 라 production 무영향 → b-3 device 게이트만 revert (forward_into_offload_fmt unwire).

**device-only 검증 필수 항목 (host 미발화)**:
1. `OffloadKVCache::get_view` 의 `set_gpu_backend` GPU upload(`write_buffer_range`)가 lock guard 안에서 정상 — host CpuBackend 는 이 경로 미진입.
2. preload pool worker(`OffloadFormat::preload` → `inner.lock().preload()`)와 main thread attention_into 의 Mutex 가 GPU backend 에서 deadlock/경합 없음 (layer 분리 불변식 device 재확인).
3. prefill_attention 의 GPU flash(`flash_attention_prefill`, standard_format.rs:579) 가 offload 의 SeqMajor + device-resident KV 와 호환 (kv_is_gpu 판정).

---

## 7. interior-mut / aliasing 위험 정리 (RPN 순)

| # | 위험 | 근거 | 완화 | 잔여 |
|---|---|---|---|---|
| R1 | **preload worker + main thread 가 동일 layer Mutex 동시 lock** → deadlock 또는 데이터 경합 | far_idx 불변식이 깨지면 같은 OffloadFormat 을 두 thread 가 lock | far_idx=i+depth(depth≥1)≠i 불변식 보존; release(i-1)는 collect 후 호출. Mutex 가 오히려 race 를 막음 | device 게이트 R2(deadlock 부재) 명시 검증 |
| R2 | **Mutex lock 비용이 TBT 회귀** | write_kv+attention_into+current_pos 각 lock | offload 는 store I/O(ms 단위)가 지배적 → lock(ns) 무시 가능. Step 3 plan 도 2 lock/layer 로 Δ+0.24% | device avg_tbt Δ≤+3% 게이트 |
| R3 | **get_view 의 `&mut self` interior(gpu_buf 재할당, preloaded reset)가 lock guard 밖으로 새는 Tensor 핸들** | attention_into 가 `(kc, vc) = g.get_view()` 후 lock 잡은 채 attention_gen 호출 | guard 를 attention_gen 종료까지 유지(StandardFormat::attention_into 와 동일 패턴). Tensor 는 gpu_buf 의 Arc clone 이라 guard 풀려도 유효하나, guard 유지가 안전 | host round-trip test |
| R4 | **preload_erased monomorphization 타입 불일치** (`*mut OffloadKVCache` vs `*mut OffloadFormat`) | raw ptr cast 가 타입 안전성 우회 | caches_ptr 타입 + preload_erased 제네릭 인자를 OffloadFormat 으로 **동시** 치환 (한 substep b-1). 불일치 시 UB | b-1 host 컴파일+test |
| R5 | **OffloadFormat 가 `Send+Sync` 인데 Mutex<OffloadKVCache>** 의 OffloadKVCache 가 `Box<dyn OffloadStore>` 등 비-Send 필드 보유 가능 | KVCacheFormat: Send+Sync 요구 | OffloadKVCache 는 이미 preload pool 로 thread 간 이동 중(unsafe Send) → 실질 Send. Mutex<T> Sync 는 T:Send 필요. OffloadStore: Send 확인 필요 | b-0 컴파일이 강제 |
| R6 | **prefill_attention(SeqMajor) 가 OLD forward_prefill 의 offload attention 과 비-동일** | OLD 는 forward_prefill<C> 의 attention 블록, fmt 는 prefill_attention free fn | 둘 다 SeqMajor + flash_attention_forward_strided 라 동일 식 기대. **device 게이트가 prefill 토큰 포함 32-tok 으로 검증** | device bit-identical |

---

## 8. 미해결/결정 필요 (caller 에게)

1. **device 게이트 매체 순서**: legacy_generate(Step 5a 폐기 대상)가 현재 유일한 offload device 매체. b-3 device 게이트를 (i) Step 5a 이전 수행, (ii) argus-chat/argus-bench family bin 선확보 중 택1. **권장 (i)** — Step 5a 와 5b-device 를 한 device round 에 묶고, legacy 폐기는 그 직후.
2. **compact / on_kv_prune**: OffloadForward::on_kv_prune 는 no-op(eviction 미지원, offload_forward.rs:219). OffloadFormat::compact 는 `bail!` 또는 `unreachable` — eviction 이 offload 에 도달 불가하므로 bail 권장.
3. **OffloadFormat 위치**: no-mod.rs 컨벤션 → `engine/src/pressure/offload_format.rs` (형제 `offload.rs` 옆) 또는 `pressure/offload/format.rs`. StandardFormat 이 `pressure/standard_format.rs` 이므로 **`pressure/offload_format.rs` 권장** (일관성).
4. **run_chunked_prefill** 은 D2 범위 밖(Standard forward_into 소비) — 별도 substep 으로 분리 명시.

---

**관련 파일 경로** (모두 절대경로):
- `/home/go/Workspace/llm_rs2/engine/src/pressure/offload.rs` (OffloadKVCache, KVCacheOps impl:263-500, PrefetchableCache impl:502)
- `/home/go/Workspace/llm_rs2/engine/src/pressure/offload/preload_pool.rs` (PreloadAccess:46, preload_erased:177)
- `/home/go/Workspace/llm_rs2/engine/src/pressure/kv_cache.rs` (PrefetchableCache standalone:22-39)
- `/home/go/Workspace/llm_rs2/engine/src/pressure/standard_format.rs` (Mutex interior-mut 선례, prefill_attention:556)
- `/home/go/Workspace/llm_rs2/engine/src/format/kv_cache_format.rs` (KVCacheFormat: Send+Sync, `&self`:57-108)
- `/home/go/Workspace/llm_rs2/engine/src/models/transformer.rs` (forward_into_offload:3717-3930, forward_into_fmt:2034, preload_erased::<OffloadKVCache>:3813/3844)
- `/home/go/Workspace/llm_rs2/engine/src/layers/transformer_layer/forward_gen_fmt.rs` (write_kv:139 / attention_into:166 위임)
- `/home/go/Workspace/llm_rs2/engine/src/layers/transformer_layer/forward_prefill_fmt.rs`
- `/home/go/Workspace/llm_rs2/engine/src/session/forward/offload_forward.rs` (OffloadForward:34, kv_caches:39, forward_into_offload 호출:156/190, alloc_offload_kv_caches:265, reset_kv:224)
- `/home/go/Workspace/llm_rs2/engine/src/session/chat/session.rs` (build_chat_offload:516, OffloadForward::new:547)
- `/home/go/Workspace/llm_rs2/engine/src/session/prefill.rs` (run_chunked_prefill:101, forward_into:348/446 — D2 범위 밖, 별도 substep)

## 설계 nonhappy-disposition
확정되었습니다. 핵심 사실:
- `forward_into_offload`는 `TransformerModelForwardArgs<'_, OffloadKVCache>`를 받아 OLD `forward_into<C>` 본문 (layer.forward → forward_gen/forward_prefill<C>)을 공유하는 별개 forward. KVCacheOps 의존.
- `swap_dispatch`/`swap_runtime`은 forward 경로를 직접 부르지 않음 → **weight mmap 조작 only, OLD-chain forward 미의존**.
- `run_chunked_prefill`(prefill.rs)은 **legacy/generate.rs:1848에서만 호출** → legacy 폐기 시 호출처 소멸. 단 standard_happy는 `decode_loop.prefill()` (ModelForward, fmt 게이트) 사용 → chunked prefill 미사용.

모든 census가 완료되었습니다. 분석 결과를 정리합니다.

---

# D3: argus_cli reject 모드 거취 분석 + legacy 폐기 hard blocker

## 핵심 발견 (census 정정)

★**원 census(영역 1~4)의 중대한 빈틈을 정정한다**: 영역 2가 eval/ppl/batch/run_chunked_prefill을 "forward_into<C> OLD-chain 소비자"로 분류했으나, **소스 직접 검증 결과 Step 1(①-c/①-d/①-e)에서 eval/ppl/batch/dump/qcf/warmup이 전부 `forward_into_fmt`(fmt 경로)로 이미 flip 완료**되었다. 이들은 `forward_into<C>`(OLD generic)를 **더 이상 소비하지 않는다**. 잔여 `use KVCacheOps`는 concrete `current_pos()` 호출 + import 잔재일 뿐(영역 4와 일치). 따라서 **모드 거취 ≠ KVCacheOps 의존**: 거의 모든 모드 진입 함수가 이미 `session/` 공유 모듈로 추출되어 fmt-clean하다.

**legacy/generate.rs는 main() dispatcher일 뿐, 실로직은 `session/`에 산다.** 모드 거취는 "로직 이주"가 아니라 "**bin entry 배선 + reject 해제 + 게이트**" 문제다.

**`generate` bin은 이미 Cargo.toml에 존재하지 않는다**(legacy_generate만). verify.py:99는 `binary_name="generate"`를 빌드하려 하나 실패 — **verify 하네스는 현재 stale**(2026-04-23 마지막 results). 이건 blocker 분류에 결정적이다.

## 모드별 거취 표

| 모드 | legacy 진입(generate.rs) | 실로직 위치 | KVCacheOps / OLD-chain 의존 | 실사용도 | 이주 비용 | 권고 |
|---|---|---|---|---|---|---|
| **eviction** (sliding/h2o/d2o/h2o_plus/snapkv) | main inline (846~1100, decode loop의 eviction_trigger) | `decode_fallback/eviction_trigger.rs` + `pressure/`(handler/policy 전부) + (3c-evict) `compact` 호스트 등가 완료 | **OLD-chain 미의존** — forward는 ModelForward fmt 경로 가능, eviction은 decode loop 후처리. 단 happy_path가 `eviction_policy()=="none"` 요구 → plan path 비활성, forward_into fallback 발화 | **높음** — verify(memory_critical_evict, signal_memory_critical) + Round 1~15 실험 핵심 + paper의 주 contribution축 | 中 — `decode_fallback` 모듈 재사용. argus_cli에 eviction subcommand 배선 + resilience hook | **이주** (argus-bench 또는 argus-cli v1 eviction subcommand) |
| **KIVI** (`--kv-mode kivi`) | main 274/372/739~755 (`run_kivi`) + ppl(`run_kivi_ppl`) + eval | `forward/kivi_forward.rs`(KiviForward) + `ppl/runner.rs::run_kivi_ppl`(fmt flip ✅ ①-e) | **혼재**: ppl/eval 경로=fmt-clean ✅. 단 **`KiviForward::prefill/step`(kivi_forward.rs:157/188)는 여전히 `forward_into<C>`(OLD generic) 소비** | 中 — verify(direct_cmd_kvquant) + paper M3/M4 microbench(별 bin) | 中~高 — KiviForward를 fmt 이주(forward_into→forward_into_fmt) 필요. KIVIFormat prefill arm은 ①-e에서 이미 구축 | **이주** (단 KiviForward fmt 이주 = Step 5 작업과 중첩) |
| **offload** (`--kv-mode offload`) | main 487/786 (`run_with_offload`) + chat 547 | `forward/offload_forward.rs`(OffloadForward) → `forward_into_offload<OffloadKVCache>` | **★OLD-chain hard 의존** — `forward_into_offload` 본체가 공유 OLD layer chain(layer.forward→forward_gen/forward_prefill<C>) + `impl KVCacheOps for OffloadKVCache` 소비. Step 2는 supertrait만 분리, forward는 미이주 | 中 — verify(direct_cmd_kvoffload) + KVSwap 연구(MEMORY: 1B에선 실익 미미, 8B+ 용) | **높음** — Option B(forward_gen_fmt + OffloadKVCache: KVCacheFormat interior-mut + preload pool aliasing 재설계 + **device GPU 재검증**) | **이주 또는 defer** — Step 5 (b)의 명시 작업. drop 시 KVSwap 연구축 소멸 |
| **weight swap** (8종: secondary_gguf/force_swap_ratio/incremental/intra_forward/layer_immediate/phase_aware/+ swap shorthand) | main 1138~1441(dispatcher 셋업) + decode loop 2201~2233 | `decode_fallback/swap_dispatch.rs` + `swap_runtime.rs` + `pressure/weights/` | **OLD-chain 미의존** — swap은 weight mmap 조작 only(forward 호출 안 함). happy_path가 swap_intra/layer/phase=off 요구하나 forward 자체는 무관 | **높음** — paper 주 contribution(weight swap Phase 1~6, TBT gap, LISWAP 트랙 다수) + perf 측정 핵심 | 中 — swap_dispatch 모듈 재사용, argus_cli decode loop에 dispatcher 배선. (이미 v1-3 계획) | **이주** (argus-bench / argus_cli v1-3) |
| **profile** (`--profile`/`--profile-events`) | main 2160~2172 + 3645(profile_dir) + prefill.rs profiler | `prefill.rs`(profiler) + `observability/profile/` + decode loop inline | **간접** — `run_chunked_prefill`(prefill.rs:348/446 `forward_into<C>` OLD)가 profiler 보유 | 中 — `/profile` skill + Tester 워크플로. CLAUDE.md: "성능 측정은 --profile 없이" → production TBT 측정엔 불요 | 中 — run_chunked_prefill fmt 이주 필요(profiler/variance 경로) | **이주** (argus-bench) — 단 우선순위 낮음 |
| **prompt-batch** (`--prompt-batch`) | main 1778 (`run_batch`) | `batch/runner.rs`(fmt flip ✅ ①-d) | **OLD-chain 미의존** — batch는 forward_into_fmt 사용 ✅ | 낮음~中 — 실험 throughput 측정 | **낮음** — batch/runner.rs는 이미 fmt-clean. argus-eval에 배선만 | **이주** (argus-eval, 저비용) |
| **eval-ll** (`--eval-ll`/`--eval-batch`/`--eval-continuation`) | main 274/347/1647 (`run_eval_ll`/`run_eval_ll_generic`) | `eval/runner.rs` + `eval/eval_loop.rs`(fmt flip ✅ ①-c) | **OLD-chain 미의존** — `run_eval_ll_generic<C: EvalCacheKind>` + forward_fmt_roundtrip → forward_into_fmt ✅. generic은 EvalCacheKind(KVCache/KiviCache 다형성)일 뿐 KVCacheOps 아님 | **높음** — eval methodology(docs/30) + PPL/NIAH/LongBench Tier 1~3 + 실험 NLL 측정 | **낮음** — eval_loop는 fmt-clean. argus-eval에 배선만 | **이주** (argus-eval, 저비용) |
| **ppl** (`--ppl`) | main 372/1747 (`run_ppl_dispatch`/`run_kivi_ppl`) | `ppl/runner.rs`(fmt flip ✅ ①-d/①-e) | **OLD-chain 미의존** — forward_into_fmt ✅ | **높음** — perplexity 측정 핵심 | **낮음** — fmt-clean. argus-eval 배선만 | **이주** (argus-eval, 저비용) |
| **dump-importance** (`--dump-importance`) | main 1629 | `dump_importance.rs`(fmt flip ✅ ①-d) | **OLD-chain 미의존** — forward_into_fmt ✅ | 낮음 — QCF 분석 도구 | **낮음** — fmt-clean | **이주** (argus-eval dump) 또는 **drop**(저사용) |
| **qcf-dump** (`--qcf-dump`) | main 1554/1693/3488 | `qcf_runtime.rs`(fmt flip ✅ ①-d) | **OLD-chain 미의존** — forward_into_fmt ✅ | 낮음 — QCF 곡선 분석 | **낮음** — fmt-clean | **이주**(argus-eval dump qcf) 또는 **drop** |
| **chat** (`--chat`/`--chat-socket`/`--chat-tcp`) | main 473~583 (`run_chat_repl_v2`) | `chat/repl.rs` + `chat/session.rs` | **혼재**: Standard chat=ModelForward(fmt_eligible=false, plan OFF) → forward_into fallback. KIVI chat=KiviForward(OLD). Offload chat=OffloadForward(OLD-chain) | 낮음~中 — 데모/대화. paper 측정 미사용 | 中 — chat/session.rs 재사용, argus-chat 신규 bin. KIVI/Offload chat은 위 모드 이주에 종속 | **이주**(argus-chat) 또는 **drop**(데모용, 측정 무관) |
| **tensor-partition** (`--tensor-partition>0`) | main 716/2747~3130 (`prepare_tensor_partition`) | `layers/tensor_partition/` + main inline (미추출) | **간접** — forward_prefill/forward_gen의 partition 분기(dead in fmt). happy_path가 partition=0 요구 | 中 — verify(partition_ratio×2, prefill_midway_partition) + paper(Direction A failed, partition baseline) + experiments/run_tensor_partition.sh | **높음** — main에 미추출(session/ 모듈 없음). 추출+이주 비용 큼 | **이주 또는 defer** — argus-bench. 미추출이라 가장 무거움 |

## 권고 요약 (우선순위)

1. **저비용 이주 (이미 fmt-clean, 배선만)**: eval-ll / ppl / dump-importance / qcf-dump / prompt-batch / batch → **argus-eval** 신규 bin 1개로 묶음. 로직은 `session/{eval,ppl,batch,dump_importance,qcf_runtime}.rs`에 이미 존재, fmt 경로 완료. main dispatcher 배선만 복제.

2. **중비용 이주 (모듈 존재, resilience/decode-loop 배선 필요)**: eviction + weight swap → **argus-bench** 신규 bin. `decode_fallback/{eviction_trigger,swap_dispatch}.rs` 재사용. verify 하네스 + paper perf 측정의 핵심 수요.

3. **고비용 이주 (forward fmt 재이주 + device GPU 재검증 필요)**: offload(OffloadForward) + KIVI(KiviForward) → Step 5 (b)의 OLD-chain fmt 이주와 **물리적으로 동일 작업**. offload는 Option B(interior-mut + preload pool aliasing). 이 둘은 "모드 이주"가 아니라 "**KVCacheOps 삭제 자체의 선결**".

4. **drop 후보 (저사용 + 측정 무관)**: chat(데모용, paper 미사용) — argus-chat 보류 가능. dump/qcf도 사용도 낮으나 fmt-clean이라 이주 비용이 미미해 묶어 이주가 합리적.

5. **미추출 = 가장 무거움**: tensor-partition은 main inline에 남아 session/ 모듈 부재 → 별도 추출 라운드 필요. defer 권장.

## ★legacy 폐기 hard blocker 목록

legacy/generate.rs 폐기 = `legacy_generate` bin 삭제. 이를 막는 **진짜 hard blocker**는 모드 로직이 아니라(대부분 session/에 추출 완료), 다음 3가지다:

**HB-1. offload forward의 OLD-chain fmt 이주 (compile blocker)**
- `forward_into_offload`(transformer.rs:3717) + `impl KVCacheOps for OffloadKVCache`(offload.rs:263) + `OffloadForward`(offload_forward.rs:156/190) + `chat/session.rs:547`이 OLD layer chain을 소비. 이건 legacy 외 **non-legacy production 경로**(argus-chat/argus-bench가 offload 지원 시 살아남음).
- legacy bin만 삭제해도 OffloadForward는 `session/`에 남아 KVCacheOps를 계속 소비 → **KVCacheOps trait 삭제 불가**. roadmap Step 5(b) 명시. **device GPU 재검증 필수**.
- ※ 단 "legacy bin 삭제" 자체만 보면 HB가 아님(legacy main이 run_with_offload를 호출하지만, 로직은 OffloadForward에 산다). **KVCacheOps trait 삭제(Step 5c)의 hard blocker**다.

**HB-2. KiviForward의 OLD-chain fmt 이주 (compile blocker)**
- `KiviForward::prefill/step`(kivi_forward.rs:157/188)이 `forward_into<C>`(OLD generic) 소비. ppl/eval KIVI 경로는 fmt-clean이나 KiviForward(REPL/run_kivi)는 미이주.
- HB-1과 동일 성격: legacy bin과 무관하게 `session/`에 남아 KVCacheOps 소비 → trait 삭제 차단.

**HB-3. device 게이트 매체 + verify 하네스 (검증 blocker, non-compile)**
- run_device.py 게이트 bin = `legacy_generate`(MEMORY). Step 4에서 argus_cli 등가 확인됐으나 **argus_cli는 happy-path 전용** → eviction/swap/KIVI/offload/partition device 게이트는 여전히 legacy_generate 필요.
- **verify.py는 `generate` bin을 빌드(verify.py:99)하는데 이 bin은 Cargo.toml에 없다** → verify는 이미 stale. legacy 폐기 전 verify를 argus-bench/argus-eval로 재배선하거나, verify 폐기를 명시 결정해야 한다. verify scenario(KvOffload/eviction/partition + resilience)는 위 이주 모드들에 직접 의존.

### 결론: hard blocker 본질

- **compile hard blocker (KVCacheOps 삭제 차단)** = HB-1(offload) + HB-2(KIVI KiviForward). 이 둘은 "모드 이주"가 아니라 **OLD-chain forward의 fmt 재이주 = Step 5(b) 그 자체**. legacy bin 삭제로는 해소 안 됨(non-legacy session/ 소비자).
- **legacy bin 삭제(Step 5a)만의 blocker** = HB-3(device 게이트 매체). eviction/swap/partition을 측정할 bin이 argus_cli에 없음 → **argus-bench로 eviction+swap(+partition) 이주가 legacy bin 삭제의 실질 선결**. 이주 없이 legacy 삭제 시 paper perf 측정 + verify 하네스 매체 소실.
- **모드 로직 이주는 대부분 저비용** — eval/ppl/batch/dump/qcf는 fmt-clean(Step 1 완료 효과), 배선만 남음. 이들은 blocker가 아니다.

**최단 경로 권고**: (1) HB-1/HB-2(offload+KIVI fmt 이주)를 Step 5(b)로 먼저 수행 → KVCacheOps 삭제 차단 해소. (2) argus-bench(eviction+swap)를 device 게이트 매체로 구축 → HB-3 해소. (3) eval/ppl/batch는 argus-eval로 묶어 저비용 이주. (4) tensor-partition은 미추출이므로 defer. (5) chat은 측정 무관이라 drop 가능. 이후 legacy bin + KVCacheOps trait 동시 삭제.

핵심 파일: `/home/go/Workspace/llm_rs2/engine/src/session/forward/offload_forward.rs:156,190` (HB-1), `/home/go/Workspace/llm_rs2/engine/src/session/forward/kivi_forward.rs:157,188` (HB-2), `/home/go/Workspace/llm_rs2/engine/src/models/transformer.rs:3717` (forward_into_offload), `/home/go/Workspace/llm_rs2/engine/src/pressure/offload.rs:263` (impl KVCacheOps for OffloadKVCache), `/home/go/Workspace/llm_rs2/verify/verify.py:99` (stale `generate` bin), `/home/go/Workspace/llm_rs2/engine/src/bin/argus_cli.rs:158-207` (reject 목록).

## 설계 sequencing
# 설계 D4: Phase α-K BC Step 5 — sub-increment 시퀀싱 + 증분별 게이트

## 0. Census 재검증 결과 (roadmap blocker → 코드 fact)

설계 전 5개 핵심 blocker 를 소스로 확정했다 (roadmap 의 census 와 차이 명시):

| # | blocker | 확정 fact | 영향 |
|---|---|---|---|
| B1 | run_chunked_prefill forward 경로 | `prefill.rs:348/446` = `model.forward_into(TransformerModelForwardArgs<C>)` 직접 호출 (KVCache). **forward_gen 직접 아님 — forward_into caller** | 5-D 에 포함, host-only |
| B2 | forward_into_offload generic 잔존 | **이미 concrete** (`transformer.rs:3717 forward_into_offload(args: ...<OffloadKVCache>)`, Step 2 `936d0c99` 완료). 단 본체가 `layer.forward<OffloadKVCache>` → OLD chain 소비 + `impl KVCacheOps for OffloadKVCache`(offload.rs:263) 유지. `preload_erased::<OffloadKVCache>`(3813) | 5-B = **device GPU 재검증 증분** |
| B3 | **kivi_forward.rs (KiviForward) 미이주** | `kivi_forward.rs:157/188` = `forward_into<KiviCache>` **여전히 OLD**. ①-e 는 `run_kivi_ppl` 만 고침. KiviForward(`--kv-mode kivi` 추론 happy-ish) 는 미접촉 | **roadmap 누락** — 5-C 신설 |
| B4 | KiviCache private state | `total_tokens()` = private fn (kivi_cache.rs:423), `awqe_enabled` = private field. fmt/fmt_bridge 가 `KVCacheOps::current_pos`/`needs_attn_scores` 경유 | 5-E 에 pub inherent 추가 |
| B5 | argus family bin | **파일 부재**. argus-chat/argus-eval = "planned" reject 만 (argus_cli.rs:160~183). KiviForward·eval·chat·offload·batch 의 실제 비-legacy 진입점은 **session 모듈 함수**(legacy 가 호출) + **chat/session.rs:547 OffloadForward** | 거취 결정: family bin 신설 불필요, session 함수 자체 이주 |

추가 확정:
- **dyn KVCacheOps trait object 0건** — 전부 generic monomorphization. → trait 삭제 = 모든 `<C: KVCacheOps>` bound 제거 + concrete inherent 전환. 일괄 cutover.
- KVCache inherent 메서드 **부분 존재** (capacity/layout/kv_heads/head_dim/update/memory_usage_bytes) — 단 `current_pos`/`set_current_pos`/`kv_dtype`/`get_buffers_mut`/`advance_pos`/`ensure_capacity` 는 trait-only, `get_view` 는 시그니처 상이(`&self,seq_len` vs trait `&mut`). → inherent 보강 필요.
- forward_gen/forward_prefill OLD 의 KVCacheOps 의존(update/get_view/advance_pos/get_buffers_mut/get_kivi_raw_buffers) 은 fmt 로 "대체"가 아니라 **forward_gen_fmt/forward_prefill_fmt 가 이미 live** → OLD 삭제 시 동반 소멸 (5-F).

---

## 1. 증분 목록 (번호 + 제목 + 범위 + 게이트 + 의존)

설계 전략: roadmap 의 Step 5 를 **6 증분(5-A ~ 5-F)** 으로 분해. 원칙 = Step 1~3 선례(additive + 게이트 + 증분별 device 게이트 + 회귀 시 격리 revert). 비가역 cutover(5-E/5-F)는 **맨 마지막에 격리**하고 그 앞 증분은 전부 additive·revert-safe 로 만든다.

---

### 5-A. 비-happy 모드 거취 결정 (코드 0, 결정 문서만)
- **범위**: argus_cli 가 reject 하는 비-happy 모드(eviction/KIVI/offload/swap/profile/batch/chat/eval/ppl) 의 Step 5 후 운명 결정. **3 옵션**:
  - (A1) **session 함수 직접 보존** — legacy bin 만 삭제, 비-happy session 함수(run_kivi_*, OffloadForward, run_chunked_prefill, eval/batch)는 라이브러리 함수로 잔존. argus family bin 은 backlog 유지. **이 옵션이 5-B/5-C/5-D 의 fmt 이주를 강제** (legacy 외 소비자가 살아있으므로).
  - (A2) **drop** — 비-happy 모드를 Step 5 에서 함께 폐기. KiviForward/OffloadForward/eval/batch session 코드 삭제 → fmt 이주 불필요, trait 삭제 간소화. 단 기능 손실 (논문 microbench/KIVI/offload ablation 매체 소실 위험).
  - (A3) **family bin 신설** — argus-chat/argus-eval/argus-bench 를 이 증분에서 실제 구현 후 이주. 범위 폭발 (Step 5 밖 작업).
- **권장**: **A1** — 사용자 "legacy disposable" 은 *bin* 폐기를 의미하지 *기능* 폐기가 아님. KIVI/offload/eval 은 라이브러리 API 로 살아있고 미래 family bin 의 backend. 단 이 선택이 5-B/5-C/5-D 를 **필수 작업으로 확정**한다 (A2 였다면 5-B/5-C/5-D 의 fmt 이주가 "삭제"로 대체됨).
- **게이트**: 없음 (결정 문서 = ADR-0001 보강 + roadmap 갱신). PM/Architect.
- **의존**: 없음 (Step 5 진입 전 첫 게이트). **이 결정이 나머지 5 증분의 범위를 잠근다.**

---

### 5-B. forward_into_offload fmt 이주 (★device GPU 재검증)
- **범위**: `forward_into_offload`(transformer.rs:3717) 본체가 소비하는 OLD chain 을 fmt 로 이주. roadmap 명시 Option B = `forward_gen_fmt` + `OffloadKVCache: KVCacheFormat`(interior-mut) + **preload pool aliasing 재설계**. 영향: offload.rs(`impl KVCacheFormat for OffloadKVCache` 신설, `impl KVCacheOps` 는 5-E 까지 잔존), transformer.rs(`forward_into_offload_fmt` 신규 fork), offload_forward.rs:156/190(OffloadForward::prefill/step 게이트 배선), chat/session.rs:547(OffloadForward 사용처).
- **추가 정밀**: OffloadKVCache 는 prefetch pool 의 **buffer aliasing** 이 있어 StandardFormat 의 단순 `Mutex<KVCache>` interior-mut 패턴 그대로는 안 됨 — preload 가 K/V buffer 를 pool 에서 빌려오는 동안 fmt 의 `write_kv`/`attention_into` 가 같은 buffer 를 잡는 lifetime 충돌 검토 필요. **이 증분이 Step 5 에서 가장 설계 난도 높음.**
- **선례**: (3c-fwd) additive fork + 게이트. 신규 env 게이트(`LLMRS_KV_FMT` 재사용 or `LLMRS_OFFLOAD_FMT`) OFF default → production(`--kv-offload`) byte-불변.
- **게이트**:
  - host: `cargo build` + offload test(58) + preload test(14) pass + fmt/clippy `--workspace -D warnings` clean. `--kv-mode offload --kv-offload-storage raw --kv-type f16` host CPU greedy n=32 — gate ON vs OFF **bit-identical** (md5, Step 2 의 `568a03e...` 연속성).
  - **device(필수)**: **S25 OpenCL** offload GPU 경로 — `--kv-mode offload --opencl-rpcmem` n=32 gate ON ≡ OFF bit-identical + avg_tbt Δ≤+3%(n≥5 median). Jetson CUDA 동일. **offload 는 preload H2D + GPU forward 라 host-only 증명 불충분** (Step 2 는 monomorphization-only 라 device 불요였으나, 5-B 는 forward 경로를 실제로 바꿈 → device 필수).
- **의존**: 5-A=A1 확정 (drop 이면 이 증분 삭제).

---

### 5-C. KiviForward fmt 이주 (roadmap 누락 보강, host-only)
- **범위**: `kivi_forward.rs:157/188`(KiviForward::prefill/step) 의 `forward_into<KiviCache>` → `forward_into_fmt`(KiviCache round-trip, ①-c/①-e 선례 재사용). ①-e 가 `run_kivi_ppl` 만 고쳤고 **추론 경로 KiviForward 는 미이주** — 이게 살아있으면 `forward_into<C>` 소비자가 남아 trait 삭제 차단.
- **선례**: ①-c `EvalCacheKind::forward_fmt_roundtrip` 또는 ①-e KIVIFormat wrap 직접 재사용. KIVIFormat prefill arm 은 ①-e 에서 이미 신설됨.
- **게이트**:
  - host: build + KIVI test pass. `--kv-mode kivi` host CPU greedy n=32 BEFORE/AFTER **bit-identical** (텍스트 + ①-c 의 KIVI flush 정수회계 flush_count/q2_tokens/res_pos 완전일치). ★2 carve-out(KIVI get_view F32, nll Δ~1e-6) 는 기존 known.
  - **device(권장)**: S25 `--kv-mode kivi --opencl-rpcmem` n=32 ON ≡ OFF — KIVI GPU bits16 HeadMajor 경로 검증. ①-e 가 "KIVIFormat GPU prefill = host CPU-mode only(device defer)" 로 남겼으므로 **이 증분에서 device 첫 검증**. host-only 로 land 후 device 는 follow-on 가능(KIVI 는 production hot 아님).
- **의존**: 5-A=A1. 5-B 와 독립(병렬 가능) — 단 둘 다 forward_into 계열 소비자 제거라 순서 무관.

---

### 5-D. run_chunked_prefill fmt 이주 (host-only)
- **범위**: `prefill.rs:348/446`(run_chunked_prefill, profiler + variance_collector + CPU 청크 폴백) 의 `forward_into<KVCache>` → `forward_into_fmt`. ①-d 의 10-site 이주 패턴(EvalCacheKind round-trip)과 동형. profiler/variance instrumentation 은 dead-or-cold.
- **선례**: ①-d 직접 재사용. forward_into_fmt 에 ①-d 가 추가한 workspace=None fallthrough 이미 존재.
- **게이트**:
  - host: build + clippy/fmt clean. run_chunked_prefill 경유 경로(profiler 모드 또는 ppl multi-chunk) BEFORE/AFTER **bit-identical** (NLL 정밀일치, ①-d 의 KVCache NLL=173.1049 류).
  - **device 불요** — prefill 은 plan 무관(①-b 선례: prefill flip 은 `--no-gpu-plan` 불요), chunked prefill 은 host profiler/variance 전용 cold. (단 5-A=A2 drop 이면 이 증분 자체 삭제 = run_chunked_prefill 폐기.)
- **의존**: 5-A=A1. 5-B/5-C 와 독립.

---

### 5-E. KVCache/KiviCache/OffloadKVCache inherent 전환 + fmt/소비자 rewire (★additive, trait 미삭제)
- **범위**: trait 삭제 **직전**의 마지막 additive 증분. trait 은 아직 살아있는 채로, **모든 trait 메서드 호출처를 inherent/pub-fn 호출로 사전 전환**한다. 이게 "비가역 cutover 의 컴파일 차단자 0 선확인" 안전장치의 핵심.
  - (E1) **KVCache inherent 보강**: trait-only 였던 `current_pos`/`set_current_pos`/`kv_dtype`/`get_buffers_mut`/`advance_pos`/`ensure_capacity` 를 inherent `pub fn` 으로 추가 (trait impl 은 inherent 로 위임 = 중복이지만 일시 공존). `get_view` 는 시그니처 통일(`&mut`). current_pos 는 pub 필드 직접 접근도 가능.
  - (E2) **KiviCache pub inherent 추가**: `pub fn total_tokens()`(현 private) + `pub fn is_awqe_enabled()`(awqe_enabled 노출). KIVI 전용 메서드(res_pos/q2_tokens 는 이미 pub field, res_cap/needs_flush/flush_if_needed/get_kivi_raw_buffers) inherent 화.
  - (E3) **fmt-side rewire**: standard_format.rs(12), kivi_format.rs(10), fmt_bridge.rs(2) 의 `KVCacheOps::method(&*lock)` → `lock.method()` inherent 호출. lock guard 생존 기간 무변(동일 guard 내 method style 만 변경).
  - (E4) **concrete 소비자 rewire**: swap_handler.rs:302(`cache.ensure_capacity`, OffloadKVCache concrete), batch/runner.rs:726(`c.current_pos()`, Vec<KVCache>), test 3종(test_eng_dat_012_031/test_eng_alg_020_022/test_backend.rs), `use KVCacheOps` import 삭제.
- **선례**: **순수 additive** — inherent 추가는 기존 trait 과 공존. 호출처 rewire 는 동일 동작(method resolution 이 inherent 우선이지만 동치). 게이트 OFF/ON 개념 불필요(동작 불변).
- **게이트**:
  - host: build + `cargo test --workspace` **전체 pass**(test 파일 rewire 포함) + fmt/clippy clean. **이 시점에 `grep -rn "KVCacheOps::" engine/ --include=*.rs` 가 trait *정의* + *impl 블록* 외 0건** 이어야 함 = 비가역 단계 차단자 0 선확인.
  - device 불요 (동작 불변, 코드 경로 미변경).
- **의존**: 5-B/5-C/5-D 완료(모든 generic forward 소비자가 fmt 또는 concrete 로 전환된 후라야 호출처 rewire 가능). **5-F 의 hard 선결.**

---

### 5-F. legacy 폐기 + KVCacheOps trait 삭제 + KVCacheFormat rename (★비가역 cutover)
- **범위**: 단일 commit (또는 2 commit: legacy 삭제 / trait 삭제) 의 비가역 일괄 cutover.
  - (F1) `legacy_generate` bin 정의(engine/Cargo.toml) + `engine/legacy/generate.rs` 삭제.
  - (F2) OLD layer chain 삭제: `forward_into<C>`(transformer.rs:1509) + `forward<C>`(1388/transformer_layer.rs) + `forward_gen<C>`(forward_gen.rs) + `forward_prefill<C>`(forward.rs) + `execute<C>`(plan.rs:1277) + `TransformerModelForwardArgs<C>` struct + `impl KVCacheOps for {KVCache,KiviCache,OffloadKVCache}` (3 impl).
  - (F3) `engine/src/kv_cache_ops.rs` 파일 삭제 + `format.rs`/lib.rs 의 mod 선언 제거.
  - (F4) `KVCacheFormat` rename 동행 (ADR-0001 §title) — INV ID 안정 키 유지. (rename 은 sed-level mechanical, 별도 chore commit 권장.)
- **선례**: **비가역** — additive 불가. Step 1~3 의 "게이트 OFF 폴백" 안전장치 소멸. → §2 의 안전장치로 보강.
- **게이트**:
  - host: `grep -rn "KVCacheOps" engine/` **0건**. build + `cargo test --workspace` pass + fmt/clippy `--workspace -D warnings` clean.
  - **device-gate(full) 최종 = 진짜 최종 perf**: 5 KV 구성(Sliding/H2O/D2O/KIVI/SnapKV) × 32-tok **bit-identical** + **avg_tbt Δ≤+3%** (S25 OpenCL `opencl --opencl-rpcmem` + Jetson CUDA, n≥5 median tok0-inclusive). frozen baseline 대비. **parallel path(KVCacheOps∥KVCacheFormat) 제거 후라야 실제 monomorphization 이 드러남** — Step 3 의 +0.24% 가 여기서 재확인되거나 악화 노출. **fmt 게이트(`LLMRS_KV_FMT`)도 이 시점에 제거 = 항상 ON** (단일 경로).
  - canonical 명령(S25): `argus_cli -b opencl --opencl-rpcmem --greedy -n 32 --no-resilience --model-path <F16.gguf> --tokenizer-path <tok.json> --kv-type {f16,f32,q4} --prompt "..."` vs frozen baseline 출력.
- **의존**: 5-A·5-B·5-C·5-D·5-E 전부. **5-E 의 grep 0건(차단자 0) 확인이 진입 gate.**

---

## 2. 비가역 단계(5-F) 안전장치

5-F 는 일괄 cutover 라 Step 1~3 의 "게이트 OFF 로 즉시 무력화" 가 불가능하다. 4중 안전장치:

1. **차단자 0 선확인(5-E 가 전담)**: 5-F 이전에 5-E 가 모든 `KVCacheOps::` *호출* 을 inherent 로 옮겨 둠 → 5-F 의 trait/impl 삭제는 "정의 + impl + bound + use 선언" 만 지운다. 삭제 전 `grep -rn "KVCacheOps::" engine/` 가 trait 정의/impl 파일 외 **0건** 이어야 진입 (5-E 게이트). 이러면 5-F 컴파일 실패 위험이 `<C: KVCacheOps>` bound 제거 + `use` 제거로만 한정 — 기계적.
2. **격리 commit + 즉시 revert**: 5-F 는 5-E 위의 단일 commit. device 게이트 회귀(bit-identical 깨짐 또는 Δ>+3%) 시 `git revert 5-F` 1회로 5-E 상태(컴파일·동작 무변, trait 잔존, 모든 호출은 이미 inherent)로 복귀. 5-E 자체는 동작 불변 additive 라 revert 불요.
3. **frozen baseline 비교**: BC 진입 commit 의 legacy 출력을 reference 로 동결(roadmap §전역 device 게이트). 5-F device 게이트는 frozen 과 bit-identical 비교 — legacy 가 5-F 에서 삭제되므로 **baseline 은 5-F 이전에 캡처 필수** (5-E 시점 또는 진입 시점). argus_cli OFF 출력도 baseline 으로 병행 캡처(Step 4 가 argus≡legacy 등가 증명함).
4. **rename 분리(F4)**: `KVCacheFormat` rename 은 trait 삭제(F2/F3)와 **별도 chore commit** — mechanical sed 변경이 삭제 회귀 분석을 오염시키지 않게. device 게이트는 F1~F3 commit 에서 통과시킨 뒤 F4 rename 은 host build/test 만으로 land.

**비가역성의 본질**: 5-F 가 비가역인 건 "코드 삭제" 이지 "동작 변경" 이 아니다 — 5-E 까지 동작은 이미 fmt 단일 경로로 수렴(production 게이트 ON 화는 Step 3/4 에서 device 검증됨). 따라서 5-F device 게이트의 위험은 **신규 회귀가 아니라 "parallel path 제거 후 monomorphization perf 노출"** (avg_tbt) 이 유일. bit-identical 은 5-E 까지의 fmt 경로가 이미 증명.

---

## 3. 전체 예상 위험 순서 (낮음 → 높음)

```
5-A (거취 결정, 코드 0)                         ── 위험 無 (문서)
  ↓
5-D (run_chunked_prefill, host-only cold)        ── 낮음 (①-d 동형, device 불요)
5-C (KiviForward, host-only + device 권장)       ── 낮음~중 (①-c/①-e 동형, KIVI GPU device 첫 검증)
  ↓
5-E (inherent 전환 + rewire, additive)           ── 중 (동작 불변이나 호출처 광범위 26파일, test 3종 rewire)
  ↓
5-B (offload fmt 이주, ★device 필수)             ── 높음 (preload pool aliasing 재설계 = Step 5 최난도, GPU H2D+forward device 재검증)
  ↓
5-F (legacy 폐기 + trait 삭제, ★비가역)          ── 높음 (비가역 cutover + 최종 perf 노출, revert 격리 의존)
```

**순서 주석**:
- 5-B 와 5-E 의 순서는 트레이드오프. 위 그래프는 5-E(rewire) 를 5-B(offload) 앞에 두지만, **실제 의존은 5-E 가 5-B/5-C/5-D 완료를 요구**(§1 의존 참조) → 정확한 land 순서는 `5-A → {5-C, 5-D 병렬} → 5-B → 5-E → 5-F`. 위 "위험 순서" 는 *난도* 정렬이고, *land 순서* 는 의존 그래프 준수.
- **5-B(offload) 를 5-E 앞에** 두는 이유: 5-E 의 E4(swap_handler `OffloadKVCache::ensure_capacity` rewire)와 E3(OffloadKVCache fmt 위임)이 5-B 의 `impl KVCacheFormat for OffloadKVCache` 신설에 의존. 즉 5-B 가 offload 의 fmt impl 을 만들고, 5-E 가 그 호출처를 inherent 로 정리.
- **revert 격리 경계**: 5-B device 게이트 회귀 → 5-B revert(offload 만, 나머지 fmt 이주 유지). 5-F device 게이트 회귀 → 5-F revert(trait 부활, 5-A~5-E 유지). Step 3 perf revoke trigger(Δ>+3% AND lock-cost 실측)는 5-F 의 최종 perf 게이트에 그대로 적용.

**land 순서 (의존 준수)**:
```
5-A → 5-C ∥ 5-D → 5-B(device) → 5-E → 5-F(device, 비가역)
```

---

## 4. roadmap 대비 발견·수정 사항 (Architect/PM 반영 필요)

1. **roadmap 누락**: KiviForward(`kivi_forward.rs:157/188`) 가 `forward_into<KiviCache>` 를 **여전히 OLD 로 소비** — roadmap Step 5 §B-2 OLD-chain 잔여는 `forward_into_offload` + `run_chunked_prefill` 2개만 명시했으나 **KiviForward 가 3번째 소비자**. ①-e 는 `run_kivi_ppl`(eval)만 고쳤고 추론 KiviForward 는 미접촉. → **5-C 신설 필수**.
2. **argus family bin 부재 확정**: argus-chat/argus-eval/argus-bench 는 파일 없음(planned reject 만). 5-A 거취 결정이 "session 함수 보존(A1)" 이어야 KiviForward/OffloadForward/eval 이 라이브러리로 생존 — 이게 5-B/5-C/5-D 를 강제. drop(A2) 선택 시 5-B/5-C/5-D 가 "fmt 이주" → "삭제" 로 바뀌어 범위 급감하나 기능 손실.
3. **device 재검증 증분 = 5-B(offload) 단 1개**(GPU forward 경로 실변경). 5-C(KIVI)는 device 권장이나 host-only land 가능(production hot 아님). 5-F 최종 perf 는 device-gate(full) 필수. 5-D/5-E 는 host-only.
4. **관련 파일 절대경로**:
   - roadmap SSOT: `/home/go/Workspace/llm_rs2/.agent/todos/roadmap_alpha_k_bc_completion_2026_06_04.md`
   - trait 삭제 대상: `/home/go/Workspace/llm_rs2/engine/src/kv_cache_ops.rs`
   - rename 대상 base trait: `/home/go/Workspace/llm_rs2/engine/src/format/kv_cache_format.rs`
   - 5-B: `/home/go/Workspace/llm_rs2/engine/src/pressure/offload.rs`, `/home/go/Workspace/llm_rs2/engine/src/models/transformer.rs`(forward_into_offload:3717), `/home/go/Workspace/llm_rs2/engine/src/session/forward/offload_forward.rs`(156/190), `/home/go/Workspace/llm_rs2/engine/src/session/chat/session.rs`(547)
   - 5-C: `/home/go/Workspace/llm_rs2/engine/src/session/forward/kivi_forward.rs`(157/188)
   - 5-D: `/home/go/Workspace/llm_rs2/engine/src/session/prefill.rs`(348/446)
   - 5-E inherent 추가: `/home/go/Workspace/llm_rs2/engine/src/pressure/kv_cache.rs`(impl 1006), `/home/go/Workspace/llm_rs2/engine/src/pressure/kivi_cache.rs`(total_tokens:423 private, awqe_enabled:247 private)
   - 5-E rewire: `/home/go/Workspace/llm_rs2/engine/src/pressure/standard_format.rs`, `kivi_format.rs`, `/home/go/Workspace/llm_rs2/engine/src/session/eval/fmt_bridge.rs`, `/home/go/Workspace/llm_rs2/engine/src/pressure/swap_handler.rs`(302), `/home/go/Workspace/llm_rs2/engine/src/session/batch/runner.rs`(726), `/home/go/Workspace/llm_rs2/engine/tests/spec/test_eng_dat_012_031.rs`, `test_eng_alg_020_022.rs`, `/home/go/Workspace/llm_rs2/engine/src/bin/test_backend.rs`
   - 5-F OLD chain: `/home/go/Workspace/llm_rs2/engine/src/layers/transformer_layer/forward_gen.rs`, `forward.rs`, `/home/go/Workspace/llm_rs2/engine/src/backend/opencl/plan.rs`(execute:1277), `/home/go/Workspace/llm_rs2/engine/legacy/generate.rs`

## adversarial verify
## 검증 hidden-consumers
핵심 검증이 완료되었습니다. 정리합니다:

- **fmt 경로(forward_gen_fmt/forward_prefill_fmt)는 KVCacheOps 를 주석으로만 참조** — 실제 trait 소비 없음. fmt 경로 자체는 trait-clean (설계 주장 확인).
- **KVCacheFormat base trait 는 7 메서드** (idx/current_pos/capacity/write_kv/write_kv_batch/compact/attention_into). 그러나 fmt 래퍼의 위임은 `KVCacheOps::current_pos/capacity/layout/get_view` 를 호출 — 이 중 **`layout`(kivi_format:120)과 `get_view`(standard:366/394, kivi:124/168)는 KVCacheFormat base trait 에 없습니다**. 즉 fmt 래퍼는 base trait 메서드가 아니라 **inner cache 의 KVCacheOps** 를 통해 layout/get_view 에 도달합니다. 설계의 "KVCacheFormat 흡수안"은 이 두 메서드를 커버하지 못하며, 반드시 inner cache 의 inherent fn 으로 rewire 되어야 합니다 (설계 §2 표가 이를 [T-ONLY]/inherent 신설로 다루나, "KVCacheFormat 이 흡수"한다는 표현은 부정확).
- **KVLayout 151 refs / KiviRawBuffers 10 refs** — trait 삭제 후에도 대량 잔존하는 타입. 설계 §3 의 "kv_cache_ops.rs 파일 운명" 결정이 151 + 10 refs 에 영향. 단 이들은 `pub use` 재export 로 `pressure::kv_cache::KVLayout` 경로도 살아있어 파일 rename 시 양쪽 경로 모두 갱신 필요.

검증 완료. 결과를 반환합니다.

---

# 적대검증 렌즈 1 — 숨은 소비자/갭 (코드 수정 없음, grep 전수 재확인)

## 검증 방법
`grep -rn "KVCacheOps" engine/ --include="*.rs"` 전수(97 참조) + import 경로(`use`/`::KVCacheOps`/`KVCacheOps,`) + 재export 경유 경로(`pressure::kv_cache::`) + dyn object + where-bound + fmt 위임 호출 + base trait 시그니처를 직접 대조.

## blocking 이슈 (설계 진입 전 반드시 반영)

### B1 [HARD] census·설계가 누락한 진짜 generic where-bound 소비자 — `offload.rs:701` `compare_views<A: KVCacheOps, B: KVCacheOps>`
- 위치: `engine/src/pressure/offload.rs:701` (정의), `:759` (인스턴스화 `compare_views(&mut base: KVCache, &mut offload: OffloadKVCache, DType::F16)`).
- 성격: **테스트 모듈 내 진짜 generic 코드 공유** — 서로 다른 두 cache 타입(KVCache vs OffloadKVCache)을 한 함수로 받아 `get_view()/current_pos()/kv_heads()/head_dim()` 호출 후 byte 비교. 이것이 갈래 A(inherent-only)의 **유일한 실제 컴파일 차단 예외**다.
- census 갭: 영역1/2/3 어디에도 없음. 영역4 가 `offload.rs` 의 KVCacheOps impl(263)은 잡았으나 **같은 파일 test 모듈의 `compare_views` generic 헬퍼(701)는 놓쳤다**. 영역4 의 `test_backend.rs:913`/`test_eng_*` 는 전부 단일 concrete 타입이라 갈래 A 로 inherent rewire 가능하지만, `compare_views` 만은 **두 타입을 동시에 받으므로 inherent rewire 불가**.
- 설계 갭: D4 §1 5-E·E4 의 test rewire 대상은 `test_eng_dat_012_031` / `test_eng_alg_020_022` / `test_backend.rs` 3종만 명시. `offload.rs:701` 누락. 설계 §2.3 R6 이 "test 헬퍼 generic 잔존 시 test-local trait 1개 허용"으로 추상 처리했으나 **구체 지점을 식별 못 해 5-F 컴파일 차단자가 됨**.
- 권고: 5-E 에 **E5 신설** — `offload.rs:701 compare_views` 를 (a) test-local micro-trait(`KVViewProbe { fn view; fn pos; fn heads; fn dim }`, KVCache·OffloadKVCache 2 impl, production 무오염), 또는 (b) `compare_views` 를 두 concrete 오버로드(non-generic)로 분리. 둘 다 production 무영향. 이 결정이 5-E 게이트 "grep KVCacheOps:: 0건" 충족의 선결.

## non-blocking 권고 수정 (설계 정확성 보강)

### W1 census 가 명시 안 한 재export 표면 — `kv_cache.rs:10 pub use ... KVCacheOps`
- `engine/src/pressure/kv_cache.rs:10` 가 `KVCacheOps`(+KVLayout/KiviRawBuffers)를 재export. test 4종(`test_action_pool:9`, `test_backend:13`, `test_eng_dat:9`, `test_eng_alg:7`)이 **`kv_cache_ops::` 가 아니라 `pressure::kv_cache::KVCacheOps` 경로로 import**한다. census 가 이들을 영역4 에서 잡긴 했으나 "재export 경유"라는 사실을 명시하지 않아, 5-F 의 trait/use 삭제 시 **재export 라인(kv_cache.rs:10)도 동반 삭제 + import 경로 갱신 대상**임이 설계에 빠짐. 설계 5-F 의 "F3: kv_cache_ops.rs 파일 삭제 + mod 선언 제거"에 **`pressure/kv_cache.rs:8-10` 재export 블록 삭제**를 명시 추가 권고.

### W2 "KVCacheFormat 흡수안"은 layout/get_view 를 커버하지 못함 (rename 흡수 부정확)
- fmt 래퍼가 위임하는 메서드 = `current_pos`(standard 6회/kivi 3회), `capacity`(kivi 2회), `layout`(kivi:120 1회), `get_view`(standard 2회/kivi 2회). **`layout` 과 `get_view` 는 KVCacheFormat base trait(7 메서드: idx/current_pos/capacity/write_kv/write_kv_batch/compact/attention_into)에 부재**.
- 즉 "KVCacheFormat base trait 이 ADR rename 을 이미 실현 = 모든 위임을 흡수" 라는 설계 §0[F2]/§3 결론은 **layout/get_view 위임에 대해 거짓**. 이 둘은 base trait 이 아니라 **반드시 inner cache 의 inherent fn 으로 rewire**(설계 §2 표가 [T-ONLY] get_view → `view(&self)` 신설로 다루긴 하나, §3 의 "이미 충족됨" 서술과 모순). 권고: §3 의 "rename 이미 실현" 문장을 "base trait 은 write/attention/geometry 만 흡수, layout/get_view 는 inherent rewire 필요"로 정정.

### W3 KVLayout 151 / KiviRawBuffers 10 refs — 파일 rename 결정의 실비용
- trait 삭제 후에도 `KVLayout`(151 refs) `KiviRawBuffers`(10 refs)가 잔존하며 **`kv_cache_ops.rs` 와 `pressure::kv_cache::` 재export 양쪽 경로로 소비**된다. 설계 §3 R5 의 "파일 rename vs 유지" 결정 시, rename 선택하면 **151+10 = 최대 161 참조의 import 경로 변경**(단 재export 가 흡수하면 직접 import 만). 설계가 "26 파일 import 수정"이라 했으나 실제 KVLayout 만 28+ 파일에 산재. 최소 변경 = 파일명 유지(KVCacheOps trait + 재export 만 삭제, KVLayout/KiviRawBuffers 정의 잔존) 권고를 §3 에서 더 강하게.

## 갭 아님 — 확인된 census 정확 항목
- **dyn KVCacheOps trait object = 0건** (grep `dyn.*KVCacheOps`/`Box<.*KVCacheOps`/`Arc<.*KVCacheOps` 전부 0). census F1/설계 §0[F1] 정확.
- **fmt 경로(forward_gen_fmt/forward_prefill_fmt)는 KVCacheOps 를 주석으로만 참조**, 실제 trait 소비 없음. 설계의 "fmt 경로 trait-clean" 정확.
- **매크로 생성 코드 = 없음** (KVCacheOps 는 매크로 외부 수기 정의/impl 만).
- `format.rs:12` / `layer_boundary_hook.rs:7` / `eval/hook.rs:22` / `eval_loop.rs:5` = 전부 주석 only (census 영역3/4 일치).
- kivi_cache.rs test 내부 use 2건(`:2814 test_compare_kivi_vs_baseline`, `:3445 test_kivi_needs_attn_scores`) = 단일 KiviCache concrete, inherent rewire 가능 (B1 같은 generic 아님). 단 설계 5-E test rewire 목록에 **이 2건도 누락** — non-blocking이나 추가 권고.

## 종합 권고
1. **B1 = 단일 blocking**: 5-E 에 E5(offload.rs:701 compare_views generic 해소) 신설. 미반영 시 5-F 컴파일 실패.
2. 5-F 삭제 목록에 `pressure/kv_cache.rs:8-10` 재export 블록 명시 추가 (W1).
3. §3 "rename 이미 실현" 서술을 layout/get_view 예외 포함으로 정정 (W2).
4. 5-E test rewire 목록에 `offload.rs:701` + `kivi_cache.rs:2814,3445` 추가 (B1+보강).

핵심 파일(절대경로): `/home/go/Workspace/llm_rs2/engine/src/pressure/offload.rs:701,759` (B1 generic 헬퍼), `/home/go/Workspace/llm_rs2/engine/src/pressure/kv_cache.rs:8-10` (W1 재export), `/home/go/Workspace/llm_rs2/engine/src/format/kv_cache_format.rs:61-108` (W2 base trait 7메서드), `/home/go/Workspace/llm_rs2/engine/src/pressure/kivi_format.rs:120` (W2 layout 위임), `/home/go/Workspace/llm_rs2/engine/src/pressure/standard_format.rs:366,394` (W2 get_view 위임), `/home/go/Workspace/llm_rs2/engine/src/pressure/kivi_cache.rs:2814,3445` (test use 누락).

## 검증 compile-blocker-ordering
결정적 확정입니다. fmt 경로는 `TransformerModelForwardFmtArgs`(non-generic)를 쓰고, `TransformerModelForwardArgs<C>`(generic, KVCacheOps bound)의 잔존 소비자는 정확히 4곳:

- prefill.rs:348/446 (run_chunked_prefill) → **5-D**
- model_forward.rs:429/589 (LLMRS_KV_FMT OFF fallback) → **5-F 게이트 제거**
- kivi_forward.rs:157/188 → **5-C**
- offload_forward.rs:157/191 (forward_into_offload, concrete OffloadKVCache) → **5-B**

이것이 D4 census의 정확성을 입증합니다. 모든 fmt 소비자는 `TransformerModelForwardFmtArgs`로 깨끗합니다. 충분한 데이터가 모였습니다. 적대검증 결과를 종합합니다.

---

# 적대검증 결과 — 렌즈 2: 컴파일 차단자 순서

## 판정: 시퀀싱 골격은 건전(SOUND). 단 blocking 1건 + non-blocking 3건의 census 오류가 설계 본문에 박혀 있어 수정 필요.

5-A → {5-C ∥ 5-D} → 5-B → 5-E → 5-F 의 land 순서는 각 증분 끝 `cargo build` 통과를 보장한다. trait 삭제 cutover(5-F)는 5-E의 grep-0 게이트로 차단자 0이 실측 검증 가능하다. 순환 의존(fmt↔inherent)은 **없음**(단방향 확정). 그러나 아래 오류들이 그대로 구현으로 가면 5-B/5-E 경계에서 컴파일 실패 또는 잘못된 작업 순서를 유발한다.

---

## BLOCKING-1 — 5-E rewire 표면에서 test 모듈 2곳 누락 → 5-F 진입 게이트(grep 0건) 위반, 5-F cutover 시 `cargo test` 컴파일 실패

D4 census 영역 4 + §1 5-E의 E4는 KVCacheOps 메서드 호출 test 파일을 3종(`test_eng_dat_012_031.rs`, `test_eng_alg_020_022.rs`, `test_backend.rs`)으로만 나열했다. 그러나 소스 직접 검증 결과 **인라인 test/bench 모듈 2곳이 추가로 `KVCacheOps::get_view`를 호출**한다:

- `engine/src/pressure/kivi_cache.rs:2929` — `test_compare_kivi_vs_baseline` (`#[cfg(test)]` mod tests 내, KVCache 대상 `KVCacheOps::get_view(&mut cache)`)
- `engine/src/pressure/offload.rs:1558` — `#[cfg(test)]` (line 520~) bench 내 `KVCacheOps::get_view(c)` (KVCache 대상) + `offload.rs:701` `compare_views<A: KVCacheOps, B: KVCacheOps>` **generic test 헬퍼**

영향:
- 5-F가 trait를 삭제하는 순간 이 2 파일의 test가 컴파일 실패 → `cargo test --workspace`(5-F host 게이트) 깨짐.
- 더 심각하게, 5-E의 진입/종료 게이트로 정의된 `grep -rn "KVCacheOps::" engine/`가 "trait 정의/impl 파일 외 0건"을 요구하는데, `kivi_cache.rs`/`offload.rs`는 **impl 파일이면서 동시에 test 호출처를 품고 있어** grep이 0건을 못 만든다 → 게이트 자체가 통과 불가 상태로 설계됨.
- 특히 `offload.rs:701 compare_views<A: KVCacheOps, B: KVCacheOps>`는 **generic 헬퍼**다. D4 §1 R6은 "test 헬퍼가 generic이면 test-local 작은 trait 허용"을 언급했으나, 이 구체 지점을 5-E rewire 목록에 넣지 않았다. 갈래 A(inherent-only)에서 이 generic은 컴파일 불가 — concrete `compare_views(base: &mut KVCache, offload: &mut OffloadKVCache, ...)`로 단형화하거나 test-local trait가 필요.

**권고 수정**: 5-E의 E4 목록에 다음 2 파일을 명시 추가하고, `compare_views`의 generic 처리(concrete 2-arg 단형화 권장 — 호출처 offload.rs:759가 `(&mut base, &mut offload)` 단일 조합뿐)를 5-E 작업 항목으로 편입. 5-E 게이트의 grep 표현식을 "trait 정의 파일(kv_cache_ops.rs) + 3 impl 파일의 *impl 블록* 외 0건"으로 정밀화하되, **impl 파일 내 test 모듈도 rewire 완료 후 0건**이어야 함을 명시.

---

## BLOCKING-2 (경계) — D4 §3의 "5-B → 5-E 의존" 근거가 오류. 의존이 실제로는 없으나, 잘못된 근거가 land 순서 강제를 정당화하고 있음

D4 §3 순서 주석: *"5-B(offload)를 5-E 앞에 두는 이유: 5-E의 E4(swap_handler `OffloadKVCache::ensure_capacity` rewire)와 E3(OffloadKVCache fmt 위임)이 5-B의 `impl KVCacheFormat for OffloadKVCache` 신설에 의존."*

소스 검증 결과 **이 근거는 틀렸다**:
- `swap_handler.rs:302`의 `cache.ensure_capacity()`는 **OffloadKVCache가 아니라 KVCache** 대상이다 (`recall_one(cache: &mut KVCache)`, line 271; `recall_caches(&mut [KVCache])`, line 103). 따라서 E4의 이 항목은 KVCache의 inherent ensure_capacity(E1, offload 무관)에 의존하지, 5-B에 의존하지 않는다.
- D4 §4 4번 파일 목록도 "swap_handler.rs:302 (`cache.ensure_capacity`, OffloadKVCache concrete)"로 오기.

실제 의존 관계:
- E3(OffloadKVCache fmt 위임)은 5-E가 아니라 **5-B 자체의 일부**다 — `impl KVCacheFormat for OffloadKVCache`를 만드는 것이 5-B. 5-E의 E3은 "StandardFormat/KIVIFormat의 KVCacheOps:: 호출을 inner inherent로 rewire"이지 OffloadKVCache fmt 위임이 아니다.
- 따라서 5-B와 5-E 사이의 진짜 의존은 단 하나: **5-B가 OffloadKVCache의 inherent 메서드(current_pos/update/get_view 등)를 신설해야 그 위에서 `impl KVCacheFormat for OffloadKVCache`가 inner를 부를 수 있다.** 그런데 5-E의 E-step이 OffloadKVCache inherent 보강을 담당하므로, 정확한 의존은 **"5-B가 OffloadKVCache inherent 신설을 선행 요구"**다.

이것이 시퀀싱에 주는 실제 영향 — D4의 land 순서 `5-B → 5-E`는 결과적으로 맞지만 **근거가 틀려서 위험하다**:
- 5-B에서 `impl KVCacheFormat for OffloadKVCache`(후보 A = `Mutex<OffloadKVCache>` wrapper)를 만들 때, 그 본문은 `self.inner.lock().current_pos()` 등 inner 호출이 필요하다. OffloadKVCache는 current_pos/update/get_view의 **inherent가 없고 KVCacheOps impl(263~501)만** 있다 (실측 확인). → 5-B 시점에 OffloadFormat이 inner를 부르려면 `KVCacheOps::current_pos(&inner)`를 쓰거나(트레이트 잔존 의존 — 5-F까지 OK), 또는 OffloadKVCache inherent를 먼저 신설해야 한다.

**권고 수정**: D4 §3의 의존 근거를 정정. 두 가지 일관된 구현 경로 중 택1을 설계에 명시:
- (경로 X, 권장) 5-B의 OffloadFormat이 inner 호출 시 **트레이트 잔존을 활용**(`KVCacheOps::current_pos(&*g)`) — Step 1~3의 standard_format.rs가 정확히 이 패턴(KVCacheOps:: 호출)을 이미 쓰고 있고 5-E까지 trait가 살아있으므로 합법. 그러면 OffloadKVCache inherent 신설은 5-E의 E-step으로 미뤄지고, 5-E에서 OffloadFormat의 `KVCacheOps::` 호출도 inner inherent로 rewire. 이 경우 5-B↔5-E 의존은 "5-E가 5-B의 OffloadFormat 코드를 rewire 대상으로 포함"(역방향, 정상).
- (경로 Y) 5-B 안에서 OffloadKVCache inherent를 먼저 신설 후 OffloadFormat이 inherent 직호출. 이러면 5-E의 OffloadKVCache 항목이 5-B로 흡수.

어느 쪽이든 land 순서 `5-B → 5-E`는 유지되나, **현재 D4 본문의 "swap_handler ensure_capacity가 OffloadKVCache" 오류는 반드시 제거**해야 E4 작업 시 엉뚱한 타입에 inherent를 찾는 혼선을 막는다.

---

## NON-BLOCKING-1 — get_view 이름 충돌은 R3에서 식별됨. 단 5-E E1 작업 항목에 "rename 강제"를 못박아야 함

실측: KVCache는 inherent `get_view(&self, _seq_len: usize)`(line 534)와 trait `get_view(&mut self)`(line 1046)를 **동명·다른 시그니처**로 보유. 5-E에서 trait 본문을 inherent impl 블록으로 옮기면 같은 블록에 `get_view` 2개 → E0277 아닌 **중복 정의 컴파일 에러**. R3가 정확히 식별했으나, 5-E E1의 작업 텍스트는 "get_view는 시그니처 통일(`&mut`)"로만 적혀 있어 모호하다 — 통일하면 기존 `get_view(&self, seq_len)` 호출처가 깨진다. inherent `get_view(&self, seq_len)`의 caller 존재 여부를 5-E 착수 전 grep으로 확정하고, trait 변형은 **반드시 다른 이름**(`view`/`attention_view`)으로 신설할 것을 E1에 못박아야 한다.

## NON-BLOCKING-2 — fmt↔inherent 순환 의존 없음 (검증 통과)

forward_gen_fmt/forward_prefill_fmt/execute_fmt 본문은 KVCacheOps를 **주석으로만** 참조하고 실호출 0건(실측). fmt → `Arc<dyn KVCacheFormat>::write_kv/attention_into` → (StandardFormat/KIVIFormat 내부) inner cache inherent. inherent는 fmt를 부르지 않음. 단방향 DAG 확정. KVCacheFormat base trait도 KVCacheOps와 supertrait 무관계(format.rs:12 "공존" 주석 + kv_cache_format.rs 독립 정의 확인). 5-F에서 KVCacheOps 삭제가 KVCacheFormat/fmt 경로를 깨지 않음.

## NON-BLOCKING-3 — 5-F의 진짜 차단자 0 검증 통과 (단 fmt args 분리 확인)

`TransformerModelForwardArgs<C: KVCacheOps>`(generic)의 잔존 소비자는 정확히 4곳(prefill.rs 348/446=5-D, model_forward.rs 429/589=5-F 게이트제거, kivi_forward.rs 157/188=5-C, offload_forward.rs 157/191=5-B). 모든 fmt 소비자는 `TransformerModelForwardFmtArgs`(non-generic)로 분리되어 trait 삭제 영향 없음. 5-C/5-D/5-B 완료 시 generic args struct의 production 소비자는 model_forward OFF-fallback 2곳만 남고, 5-F의 게이트 상수화가 이를 제거 → `forward_into<C>`/`forward<C>`/`forward_gen<C>`/`forward_prefill<C>`/`execute<C>`/`TransformerModelForwardArgs<C>`/`update_kv_cache<C>`(transformer_layer.rs:35)/`ForwardGenArgs<C>`/`LayerForwardArgs<C>`가 동시 dead → 일괄 삭제 가능. 차단자 0 메커니즘 건전.

---

## 추가 확인 — 5-E grep 게이트 표현식 정밀화 필요 (BLOCKING-1 연동)

5-E 종료 게이트 "`grep KVCacheOps::` 0건"은 다음을 모두 0으로 만들어야 진입 가능:
- standard_format.rs (9건 `KVCacheOps::current_pos`/`get_view`) — E3
- kivi_format.rs (8건) — E3
- fmt_bridge.rs:129/137 — E3
- transformer.rs:1843 `expect("KVCacheOps::get_buffers_mut...")` (문자열 — grep `KVCacheOps::`에 걸리나 코드 아님, 문자열이라 게이트에서 제외/허용 명시 필요)
- kivi_cache.rs:2929 (test) / offload.rs:1558/701 (test/bench) — **BLOCKING-1, 누락분**

transformer.rs:1843은 `.expect()` 문자열 리터럴이라 컴파일에 무해하나 grep에 매칭됨 — 게이트 표현식이 문자열을 걸러내지 못하면 5-E가 false-positive로 통과 불가 판정될 수 있다. 게이트를 "주석·문자열 제외 `KVCacheOps::` 호출 0건"으로 정의할 것.

---

## 종합 권고

1. **(BLOCKING-1)** 5-E E4에 `kivi_cache.rs` 인라인 test + `offload.rs` 인라인 test/bench(`compare_views` generic 단형화 포함) 2 파일 추가. 5-E grep 게이트를 impl 파일 내 test까지 포함해 0건으로 정의.
2. **(BLOCKING-2)** D4 §3·§4의 "swap_handler ensure_capacity = OffloadKVCache" 오기 정정(실제 KVCache). 5-B↔5-E 의존 근거를 경로 X(트레이트 잔존 활용, Step1~3 선례) 또는 경로 Y(5-B 내 OffloadKVCache inherent 선신설)로 명시. land 순서 `5-B → 5-E`는 유지.
3. **(NON-BLOCKING-1)** 5-E E1에 "trait `get_view(&mut)` 변형은 inherent `get_view(&self,seq_len)`과 충돌 → 다른 이름(`view`) 강제" 명문화.
4. 순환 의존·5-F 차단자 0·fmt args 분리는 검증 통과 — 골격 수정 불요.

핵심 파일(절대경로): `/home/go/Workspace/llm_rs2/engine/src/pressure/kivi_cache.rs:2929`(누락 test), `/home/go/Workspace/llm_rs2/engine/src/pressure/offload.rs:701,1558`(누락 test/generic helper), `/home/go/Workspace/llm_rs2/engine/src/pressure/swap_handler.rs:271,302`(KVCache 대상, D4 오기), `/home/go/Workspace/llm_rs2/engine/src/pressure/kv_cache.rs:534,1046`(get_view 충돌), `/home/go/Workspace/llm_rs2/engine/src/pressure/offload.rs:263-501`(impl KVCacheOps, inherent 부재), `/home/go/Workspace/llm_rs2/engine/src/models/transformer.rs:1843`(문자열 false-positive), `/home/go/Workspace/llm_rs2/engine/src/session/forward/{kivi_forward.rs:157,188,offload_forward.rs:157,191,model_forward.rs:429,589}`, `/home/go/Workspace/llm_rs2/engine/src/session/prefill.rs:348,446`(generic args 잔존 4 소비자).

## 검증 device-reverify-soundness
모든 핵심 사실을 확인했습니다. 검증 결과를 반환합니다.

---

# 적대검증 렌즈 3 — device 재검증 건전성 (offload Option B)

## 판정: **AMBER — blocking 이슈 2건 + 게이트 정의 보강 4건**

설계(offload-optionB §6 device 게이트)는 방향이 옳으나, **device 게이트 정의가 "host 미발화→device 전용" 함정을 Step 3보다 더 심각하게 재발할 구조**다. 코드 소스로 4개 결정적 사실을 확인했고, 그 중 2개는 설계의 전제를 깨는 blocking 이슈다.

---

## 코드로 확정한 4개 사실 (설계 주장 검증)

**[C1] offload GPU 발화 = backend.is_gpu() 단일 게이트, dtype 무관** (offload_forward.rs:277, 300-302)
`alloc_offload_kv_caches`는 `is_gpu = backend.as_ref().is_gpu()` 가 true 면 **무조건** `set_gpu_backend` 호출. **dtype(F16/F32) 분기 없음.** 따라서 설계 §6의 "F16 KV (default)" 게이트 표현은 부정확 — **F16/F32 모두 동일하게 GPU 경로(get_view:437 `gpu_backend.is_some()`) 발화**. dtype은 발화 여부와 무관하고, get_view의 `write_buffer_range` upload 바이트 수(token_bytes)만 다르다. → 게이트는 F16·F32 **둘 다** 필수 (한쪽만 검증 시 다른 dtype의 upload 크기 회귀 미검출).

**[C2] preload aliasing = `preload_erased::<C>` 가 `*(ptr as *mut C)` concrete cast** (preload_pool.rs:177-180, transformer.rs:3811-3814)
설계 §3/R4 주장 그대로 확인. `caches_ptr.add(far_idx) as *mut ()` + `preload_erased::<OffloadKVCache>`. **Option B′가 `Vec<OffloadFormat>`로 전환하면 caches_ptr 타입 + 제네릭 인자를 `OffloadFormat`으로 동시 치환해야 하며, 불일치 시 raw cast가 UB를 silent하게 통과**(타입 검사 우회). 이것이 R4의 본질. → 게이트가 아니라 **컴파일+host test가 강제**하지만, raw cast이므로 **타입 불일치가 컴파일을 통과할 수 있는 경로**(`*mut ()` 중간 캐스팅)가 존재 → b-1 host test에서 실제 preload→forward round-trip 검증 필수.

**[C3] aliasing 안전 불변식 = far_idx = i + depth, depth≥1 ⇒ far_idx ≠ i** (transformer.rs:3797-3803, 3839-3847, 3897-3900)
설계 §3 R1 주장 확인. main thread `&mut *caches_ptr.add(i)`(3850) ∥ worker `caches_ptr.add(far_idx)`(3843). release는 `(i-1) >= depth`(3897)에서만 → 이미 collect된 후. **Mutex 도입 시 이 불변식이 유효하면 lock 경합 0**(서로 다른 OffloadFormat 인스턴스 = 서로 다른 Mutex). 단 ★주의: `retain_preload`(3893, i<depth)는 main thread가 layer i를 만지고, **같은 layer i가 다음 토큰에서 preload worker 대상이 될 수 있다**. cross-token 경계에서 retain된 layer의 buffer를 다음 토큰 preload worker가 만질 때 — 토큰 간 sequential 실행이므로 안전하나, **Mutex/interior-mut 전환 시 이 cross-token retain 경로가 가장 미묘**. 게이트에서 retain(depth≥1, 즉 항상)이 실제 발화하는 multi-token 시나리오 필수.

**[C4] OffloadStore: Send only (Sync 아님)** (offload/store.rs:9 `pub trait OffloadStore: Send`)
설계 §7 R5 검증: `Mutex<OffloadKVCache>`가 `Sync`이려면 `OffloadKVCache: Send`면 충분(Mutex<T>: Sync ⟺ T: Send). `store: Box<dyn OffloadStore>`는 Send ✓ → **R5는 충족**(b-0 컴파일이 강제). 단 `gpu_backend: Option<Arc<dyn Backend>>`의 `Backend: Send + Sync` 여부가 추가 전제 — 현재 `Arc<dyn Backend>`가 thread로 안 넘어가므로(preload worker는 store만 만짐) 미검증 영역. b-0 컴파일이 잡지만, **OpenCL/CUDA backend의 Send+Sync 실제 만족 여부는 host(CpuBackend)에서 검증 불가** → device-only 위험.

---

## Blocking 이슈 2건

### **BL-1 (RPN ≈ 200): device 게이트 매체 부재 — Step 3 함정의 직접 재발, 그러나 더 심각**

설계 offload-optionB §6과 nonhappy-disposition HB-3, sequencing 5-B가 모두 지적했으나 **충분히 blocking으로 격상되지 않았다.**

- **사실**: argus_cli는 offload를 reject(happy-path 전용). offload device 발화 매체 = `legacy_generate`(run_with_offload, generate.rs:4708 set_gpu_backend) 또는 `build_chat_offload`(chat). **둘 다 Step 5(a)에서 폐기 대상이거나(legacy), family bin 미구현(argus-chat 부재).**
- **Step 3 함정과의 차이**: Step 3는 host에서 fmt-ON이 build_plan_fmt None→forward_into_fmt 폴백으로 "host 미발화"였으나, **plan 경로 자체는 host CpuBackend에서 컴파일·실행은 됐다**(GPU만 미발화). 반면 **offload GPU 경로(get_view:437 gpu_backend 분기)는 host CpuBackend에서 `is_gpu()=false`라 아예 진입 자체가 0**(C1). 즉 `set_gpu_backend`→`write_buffer_range`→`attention_gen` 전체가 **host에서 단 1줄도 실행 안 됨**. Mutex guard 안에서 GPU upload가 도는 신규 코드(OffloadFormat::attention_into)는 **host test가 CPU fallback(offload.rs:485-498)만 커버** → device 게이트가 유일한 검증.
- **blocking 본질**: device 게이트를 돌릴 bin이 사라지는 순서 의존. sequencing의 land 순서 `5-A → 5-C∥5-D → 5-B(device) → 5-E → 5-F`에서 **5-B device 게이트가 legacy 폐기(5-F의 F1) 이전**이라 표면상 OK. 그러나 **5-F가 legacy를 삭제하면 그 이후 offload regression을 재현할 매체가 영구 소실** — 5-F 이후 offload GPU 경로는 device에서 다시 돌릴 수 없다(argus-chat 미구현, frozen baseline도 legacy 출력).

**권고 수정 (blocking 해소)**:
1. **5-B device 게이트 매체를 명시적으로 고정**: `legacy_generate`로 5-B device gate를 통과시키되, **5-A 거취 결정에서 "offload device 매체"를 argus-chat 또는 argus-bench로 5-B 이전에 확보**할지 결정. 설계 offload-optionB §8.1의 권장 (i)("5-A 이전 수행 + legacy 폐기는 그 직후")가 맞으나, **"그 직후"가 위험** — 5-F 이후 offload 매체 0. → **argus-bench(offload 지원)를 5-B의 hard 선결로 격상**하거나, **legacy_generate를 offload 한정으로 5-F에서 제외(부분 폐기)** 중 택1을 5-A에서 명시.
2. frozen baseline 캡처를 **legacy의 offload GPU 출력**으로 5-B 이전에 device에서 캡처(S25+Jetson 각각). argus OFF 출력은 offload를 reject하므로 baseline 불가 — **이것이 Step 3와 결정적으로 다른 점**(Step 3는 argus≡legacy 등가를 Step 4에서 증명했으나 offload는 argus가 reject).

### **BL-2 (RPN ≈ 168): Mutex guard 생존 기간 — GPU upload + attention_gen이 lock 내에서 도는 신규 동시성 표면이 host 미검증**

- **사실**: OffloadFormat::attention_into는 `let mut g = inner.lock(); let (kc,vc) = g.get_view(); be.attention_gen(...)`(설계 §2.2 후보 A). **get_view(C1, offload.rs:437-482)가 GPU 버퍼 alloc + `write_buffer_range` 2회(K,V upload)를 guard 안에서 수행** → lock 보유 시간이 **GPU H2D enqueue 동안 지속**. main thread가 layer i의 lock을 GPU upload+attention 동안 잡고 있을 때, **preload worker가 layer far_idx의 lock(다른 Mutex)을 잡는 건 무경합**(C3 불변식). 그러나:
  - **신규 위험**: get_view가 guard 내에서 `self.gpu_k_buf = Some(gpu_mem.alloc(...).expect(...))`(offload.rs:444-453) — **alloc 실패 시 `.expect()` panic이 Mutex poisoning**을 유발. main thread panic → Mutex poisoned → 다음 토큰 lock().unwrap() panic 연쇄. legacy 경로(`&mut self`)는 poisoning 개념 자체가 없었으므로 **이것은 Mutex 도입으로 신규 생성되는 failure mode**.
  - host CpuBackend는 GPU 분기(437) 미진입 → **alloc/upload/poisoning 경로 전체가 host에서 0 커버**.

**권고 수정**:
1. OffloadFormat::attention_into의 lock guard 안에서 `.expect()`/`.unwrap()` panic이 도는 경로(get_view의 alloc:447/452, write_buffer_range:467/474는 이미 log::error로 swallow)를 **device 게이트의 명시 검증 항목으로 추가** — 특히 GPU 버퍼 alloc 실패가 device에서 발생 가능한지(max_seq_len × token_bytes × 2 × num_layers VRAM 압박). 단순 bit-identical로는 미검출.
2. `lock().unwrap()` 대신 poisoning 복원 전략(또는 panic=abort 의존성) 명시. 단 설계 §2.2가 StandardFormat 선례(`lock().unwrap()`)를 따른다 했으므로 — **StandardFormat은 guard 내 panic 경로가 없다**(plan_geometry는 단순 getter). offload는 guard 내 alloc panic이 있어 **선례 비대칭**. 이 비대칭을 5-B 설계에서 명시.

---

## device 게이트 정의 보강 4건 (non-blocking, 충분성 부족)

**G1**: 설계 §6 "F16 KV (default)" → **F16·F32 둘 다 게이트 필수**(C1: dtype 무관 발화, upload 크기만 상이). 설계 §6의 carve-out("offload는 항상 attention_gen 경로라 F16/F32 모두 bit-identical 기대")은 맞으나 **검증 대상에서 F32를 빼면 안 됨**.

**G2**: **prefill arm device 검증 미흡**. 설계 §6/R6이 지적했으나 게이트 명령에 미반영. forward_into_offload의 prefill(seq_len>1)은 layer.forward→forward_prefill<C>(C 경로, transformer.rs:3866 LayerForwardArgs). fmt 이주 후 OffloadFormat::attention_into의 prefill arm(prefill_attention, SeqMajor)으로 가는데 — **get_view가 prefill 시 store.store(전체)→GPU upload total_bytes**(offload.rs:358-359, 387). decode와 upload 패턴이 다름. → 게이트 32-tok에 **prefill 토큰(prompt) 포함 필수**이며, prompt 길이를 GPU upload 전체 경로가 발화하도록 (예: ≥64 prompt 토큰).

**G3**: **retain_preload cross-token 경로(C3)가 게이트에 미반영**. depth≥1이므로 retain은 항상 발화하나, **multi-token decode에서 retain된 layer가 다음 토큰 update의 deferred write(offload.rs:343-351, preloaded&&attn_buf 경로)로 가는지** 32-tok로 검증. 1~2 토큰만으론 retain 경로 미발화. → **32-tok decode + depth가 num_layers보다 작은 구성**(retain 발화 조건) 명시.

**G4**: **avg_tbt 게이트의 lock 비용 측정 정밀화**. 설계 §6/R2가 "offload는 store I/O ms 지배 → lock ns 무시"라 주장하나, **raw 모드(RawStore, in-memory)는 disk I/O가 없다** — store가 메모리 복사라 lock(2회/layer × 16 = 32/tok)이 상대적으로 더 드러날 수 있다. Step 3의 +0.24%는 plan 경로(GPU 지배)였으므로 직접 전이 불가. → raw 모드 avg_tbt를 별도로 측정(disk 모드와 분리), Δ≤+3% 게이트에 **raw가 worst-case**임을 명시.

---

## sequencing 설계(D4)에 대한 교차 검증

- **5-B가 5-E 앞** 배치(sequencing §3 주석)는 옳음 — 5-E의 E3/E4가 OffloadKVCache fmt impl(5-B 산출물)에 의존. ✓
- **단 BL-1로 인해 5-B의 device 게이트가 5-F(legacy 폐기)와 강하게 결합** — sequencing의 "revert 격리 경계"(5-B revert = offload만)는 **5-F 이전에만 유효**. 5-F 이후 offload device regression은 revert해도 재현 매체가 없다. → **sequencing §2 안전장치에 "offload device 매체 영속성"을 5-F의 진입 gate로 추가** 권고: 5-F는 (5-E grep 0건) ∧ (offload device 매체가 argus-bench/chat로 확보됨) 둘 다 만족해야 진입.

---

## 결론

설계 offload-optionB(후보 A: Mutex<OffloadKVCache> + preload는 PrefetchableCache 우회 유지)는 **아키텍처상 건전하며 C1~C4가 전제를 뒷받침**한다. data race/aliasing UB는 C3 불변식(far_idx≠i)이 Mutex 도입 후에도 보존되어 **위험 낮음**(오히려 Mutex가 raw-ptr Send 안전성 강화). 

그러나 **device 게이트 정의는 불충분하며, Step 3 함정이 더 심각한 형태로 재발**한다:
- **Step 3**: host에서 plan 미발화(GPU only)였으나 plan 경로는 host에서 컴파일·CPU 실행됨, argus≡legacy 등가로 baseline 확보 가능.
- **offload 5-B**: host에서 GPU 경로 **0 실행**(C1, is_gpu=false), argus가 offload **reject**라 baseline 매체 부재, 신규 Mutex guard 내 GPU alloc panic(BL-2)이 host 0 커버.

**최종 권고**: (1) BL-1 — offload device 매체(argus-bench/chat)를 5-B의 hard 선결로 격상하고 5-F 진입 gate에 매체 영속성 추가. (2) BL-2 — Mutex poisoning 신규 failure mode를 device 게이트 명시 검증 항목으로 추가. (3) G1~G4 — F16·F32 양쪽, prefill 포함 ≥64 prompt, retain 발화 32-tok, raw 모드 worst-case avg_tbt로 게이트 정의 보강.

관련 파일(절대경로): `/home/go/Workspace/llm_rs2/engine/src/session/forward/offload_forward.rs:277,300-302` (C1 is_gpu 발화), `/home/go/Workspace/llm_rs2/engine/src/pressure/offload.rs:437-483` (get_view GPU upload + alloc panic:447/452), `/home/go/Workspace/llm_rs2/engine/src/pressure/offload/preload_pool.rs:177-180` (C2 preload_erased concrete cast), `/home/go/Workspace/llm_rs2/engine/src/models/transformer.rs:3797-3803,3839-3900` (C3 aliasing 불변식 + retain), `/home/go/Workspace/llm_rs2/engine/src/pressure/offload/store.rs:9` (C4 OffloadStore: Send), `/home/go/Workspace/llm_rs2/engine/legacy/generate.rs:4708` (offload device 매체, 5-F 폐기 대상), `/home/go/Workspace/llm_rs2/engine/src/bin/argus_cli.rs` (offload reject).