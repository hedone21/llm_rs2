# Handoff: α-K BC (3d) S4 ✅ — chat eviction-fmt S25 device γ-sanity 게이트 PASS → (3d) 종결 → 5-F 진입

**작성**: 2026-06-05 (메인 세션)
**HEAD**: `a65aebec` docs(handoff): (3d) S3 #2 — host-CPU carve-out 확정 — **소스 변경 0 (S4=순수 device 검증)**
**브랜치**: `master` (origin 대비 ahead 0; 본 handoff 커밋 시 +1, push 미실행 — 사용자 요청 시)
**디바이스**: Galaxy S25(R3CY408S5SB), `opencl --opencl-rpcmem`, 배포 바이너리 `/data/local/tmp/legacy_generate`(`1eae6c46` 포함, 22:48 빌드)
**다음 세션 진입 문장**: **"BC Step 5-F 진입 — legacy bin + KVCacheOps trait 폐기 (비가역, S25 device 단독)"**

---

## TL;DR
(3d) S4(chat eviction-fmt UER 경로 device γ-sanity) **PASS**. UER(Unwrap-Evict-Rewrap) eviction 메커니즘이 production 경로(f16 KV, Adreno GPU flash)에서 OLD와 **bit-identical**(전 5정책)이고, **메커니즘 자체가 dtype-무관하게 SOUND**임을 GPU-flash 2모델(qwen f16 + llama f32)로 입증. 적대적 검증(workflow `wf_e64f4a27-d49`, 3 blocking)으로 verdict를 교정·강화: f32/q4 의 end-to-end ON≠OFF 는 **eviction 아니라 CPU-attention forward carve-out**(W-2/(3d)S3 #2 동류, head_dim 128 은 F32/Q4 GPU flash 부재→CPU 폴백, fmt `attention_gen` vs OLD NEON 차)임을 llama(head_dim 64, F32 GPU flash) 격리로 확정. **(3d) 전체 종결** → 다음은 5-F. 멈춘 이유: (3d) 마지막 deferred(S4) 완료, 5-F는 비가역이라 별도 설계 선결.

---

## 진행 상태 / 게이트 데이터 (md5 = stdout 전문 해시, device 파일 `/data/local/tmp/{m_,c_,d_,q4_,deep_,lla_,f32big_,f16big_}*.{out,err}`)

| 항목 | 결과 | 판정 |
|---|---|---|
| **f16 KV 5정책 × ON×2+OFF** (max256/20턴) | 정책마다 **ON1==ON2==OFF bit-identical**. sliding=`54ee1331`(1evict,bail) / streaming·h2o·h2o-plus=`de2b7161`(2evict,bail) / d2o=`6a71997e`(13evict,**exit0 완주** /stats evicted_total=518) | ✅ 동치·결정성·완주 |
| **fmt 비-vacuous 발화** | 전 ON 런 `[fwd-trace] wrapped 28 KVCache → StandardFormat`(wrap=1) | ✅ 발화 확인 |
| **UER panic** | 전 런 0 (panic/SIGSEGV/abort/assertion/unwrap-none/index-oob grep) | ✅ |
| **f16 max512/28턴 sliding** (400토큰 eviction) | ON==OFF=`a5f4cd80`, removed=400/new_pos=68 | ✅ 대규모 eviction 동치 |
| **f32 eviction-MECHANISM 격리 (llama3.2-1b, head_dim64, F32 GPU flash)** | none: ON==OFF=`6c88b0a7` / sliding: **ON==OFF=`e69c4708`**(removed=176/new_pos=68, gpufb=0) | ✅ **f32 UER 메커니즘 SOUND** |
| **eviction 이벤트(removed/new_pos)** | 전 dtype·모델 ON≡OFF 동일 | ✅ selection/회계 동치 |

**S4 PASS 결론(스코프 명시)**: chat eviction-fmt UER 경로는 (1) production f16/Adreno-GPU-flash 에서 end-to-end ON≡OFF, (2) 메커니즘이 dtype-무관하게 무손상(f16-qwen + f32-llama GPU-flash 양쪽 bit-identical), (3) crash-free/panic-0/pos↓/evicted_total>0/turn-2-append 정합/d2o 13사이클 완주.

---

## 적대적 검증으로 교정된 2개 carve-out / 스코프 (R6 핵심)

### ① f32/q4 KV end-to-end ON≠OFF = **CPU-attention forward carve-out (UER 무관)**
- **원인(코드 확정 `opencl.rs:3287-3312`)**: F32 flash 커널은 head_dim **64·256만**(128 부재), Q4_0 flash 부재. qwen2.5-1.5b=head_dim **128** → f32/q4 KV attention 은 **ON·OFF 양쪽 CPU 폴백**. fmt(`prefill_attention`/`attention_gen`)와 OLD(`forward_prefill`/inline-NEON) 의 **CPU 구현 차**가 누적되어 greedy flip → ON≠OFF.
- **eviction 아님 입증**: (a) **llama f32(head_dim64, F32 GPU flash) eviction-sliding = ON==OFF bit-identical** → UER 메커니즘 무죄. (b) qwen q4 는 **eviction-none(28턴)에서도 ON≠OFF**(`7c574be9`vs`486a69e9`) → eviction 없이 forward 발산 = forward 귀속 확정. (c) qwen f32 는 eviction-none-deep(28턴)=ON==OFF(`d6b7175b`)인데 sliding 만 ON≠OFF → **post-eviction compacted-cache 의 CPU-attention 차**(contiguous 캐시엔 fmt==OLD, 압축 후 갈림).
- **W-2 / (3d)S3 #2 carve-out 동류** — host-CPU 발산이 device 의 CPU-폴백 dtype(f32/q4@head_dim128)에서 재현. **f16(GPU flash)·device production 경로는 무영향**.
- ⚠️ 직전 세션 verdict 의 "f32 fmt=GPU flash 일관 / OLD=CPU 폴백" 서사는 **오류**(GPU-fallback 카운터 ON=0/OFF=1 은 OLD 전용 로깅 아티팩트, 실제론 양쪽 CPU). 교정됨.

### ② chat 은 eviction 에 score 미공급 → **score-driven UER arm 은 chat dead-path** (스코프 한정)
- **코드 확정(`model_forward.rs:424/440/534/600`)**: ModelForward forward 가 fmt·OLD 양쪽 `score_accumulator: None` → chat forward 는 attention score 누적 안 함. chat `score_accumulator`(session.rs:46) 는 활성/feed 되지 않음(`set_active(true)` 는 test 뿐).
- **실증**: f16 에서 **streaming==h2o==h2o-plus 동일 md5 `de2b7161`** = score 차별화 부재(전부 recency 폴백).
- **귀결**: chat 에서 **h2o/h2o-plus ≡ sliding(recency)**. `try_evict` 의 score-driven `Some(sc)` arm(`force_evict_with_scores`, model_forward.rs:661/666)은 **chat 에서 미발화**. S4(chat) 가 커버한 것은 **score-free UER arm**(=chat 실제 경로). score-driven arm 동치는 **eval/experiment(비-chat) fmt 경로에서 별도 검증 필요** — S4 스코프 밖, 5-F/future 항목.
- d2o 는 예외적으로 동작 다름(13evict): D2OHandler 내부 EMA/cosine 자체 계산(외부 accumulator 무의존).

### 기타 R6
- **bail(graceful)**: sliding/streaming/h2o/h2o-plus 는 "2차 eviction removed=0 → `anyhow::bail`(context overflow)" — **panic 아닌 graceful 거부, ON≡OFF 동일 = pre-existing OLD chat-eviction edge**(RoPE 단조증가 vs 물리 occupancy 추정). S4 범위 밖. d2o 는 반복 eviction 으로 회피→완주.
- **출력=degenerate 반복**("Go Go"/"Then Then"): q4_0 1.5B greedy echoing **모델 아티팩트**(ON/OFF 동일). UER 정확성과 무관 — 동치는 md5 로 판정.
- **커버리지**: f32/q4 는 sliding+d2o 만(streaming/h2o/h2o-plus 미; 결론은 dtype-driven 이라 무관). q4_0 **weight** 전용(chat=plan path 미사용→weight dtype 무관). S25 단독(Jetson CUDA=flip 미적용, 범위 밖).
- **재현**: 게이트 스크립트 `/data/local/tmp/s4_{matrix,clean,dtype}.sh`(host `/tmp/`에도). `--kv-type` 유효값은 **`q4`**(❌`q4_0`). eviction subcmd 는 **`h2o-plus`**(kebab, ❌`h2o_plus`). device 스크립트 env 주입은 **`env RUST_LOG=... VAR=...`** 필수(`VAR=$expanded cmd` 는 POSIX sh 가 명령어로 취급→exit 127).

---

## 다음 작업 — 5-F (legacy bin + KVCacheOps trait 폐기, 비가역, S25 device)
- (3d) 종결로 chat fmt eviction 배선 완성. 5-F 차단자(직전 메모리): production 기본=OLD `forward_into<C>`(LLMRS_KV_FMT 기본 OFF), chat-standard 가 trait 마지막 소비자.
- **5-F 선결 결정**: ① chat fmt env-gate 제거(=production fmt-default flip; happy+chat 동시) — host F32-CPU carve-out(本 handoff ①과 동류) 존재하나 **S25 device 가 acceptance**. ② OLD-chain 잔여 소비자 3(offload `forward_into_offload`+chunked-prefill+KiviForward) 처리. ③ score-driven UER arm(本 handoff ②) — eval fmt 경로 검증을 5-F 에 포함할지 결정.
- 설계부터(5-F census 4결정 확정됨: Full 폐기 / S25 단독 / chat=fmt 이주 / (3d) 완료). 진입 설계 문서 골격: `design_alpha_k_step5_2026_06_04.md` + appendix.

---

## 자기점검
- 진입 문장? ✓ "BC Step 5-F 진입 — legacy bin + KVCacheOps trait 폐기 (비가역, S25 device 단독)"
- 왜 멈췄나? ✓ (3d) 마지막 deferred(S4) PASS 완료; 5-F 비가역이라 설계 선결
- 최대 landmine? ✓ ① f32/q4 ON≠OFF=forward CPU-attention carve-out(UER 무죄, llama GPU-flash 격리 입증) ② chat score 미공급→h2o≡sliding·score-driven arm dead-path
- 게이트 수치? ✓ f16 5정책 ON≡OFF(md5 in 표) / llama f32 eviction ON==OFF `e69c4708` / panic=0 / d2o evicted_total=518 완주
- 길이? ✓ 상세=device `*.err` 로그 + workflow `wf_e64f4a27-d49`
