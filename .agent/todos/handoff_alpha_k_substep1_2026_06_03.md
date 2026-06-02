# Handoff: Phase α-K substep (1) 완료 (KVCacheFormat trait + impl) → substep (2) 첫 device round

**작성**: 2026-06-03
**HEAD**: `9d858cf9 feat(engine): Phase α-K substep (1) — KVCacheFormat trait + Standard/KIVIFormat impl`
**브랜치**: `master` (origin 미push — ahead 3: `5493330e` style, `9d858cf9` feat, 본 handoff)
**다음 세션 진입 문장**: **"α-K substep (2) 진행"** (CacheManager→Stage 분해, **첫 device round** — S25 OpenCL + Jetson CUDA).

> 설계 SSOT = `arch/pipeline_stage_design_v2.md` (§4.1 KVCacheFormat / §9 Sprint / §9.1 α-K 게이트). ADR = `docs/adr/0001-kv-dispatch-paradigm.md` (Accepted). 트랙 메모리 = [[project-pipeline-alpha-k]].

---

## TL;DR — 이번 세션 arc

1. **α-W 종결**: α-W-1·2·3a·4·5 완료. **α-W-3b = defer-B 판정** (adversarial workflow) — resilience 2-source 5 조각 전부 β-결합(registry/StepInfo carrier/로컬센서 미구축), 지금 하면 hollow/추측성코드. **재논의 불요.**
2. **사용자가 defer 대신 proceed 선택** — "갈래 2로 설계 구체화하고 전진". 재프레이밍: "β 전진" = **α-K(4-6주) → β(3-4주) ≈ 7-10주**. ADR-0001 이미 Accepted. β에서 α-W-3b 해소(α-K가 β의 OneShot KV/weight Stage 전제).
3. **α-K entry plan** 구체화 (architect workflow, §9.1) — 7 substep. **(3c)=perf 위험 crux.**
4. **substep (1) 완료 + 커밋** (`9d858cf9`). host-gated, 내가 cargo 직접 검증.

---

## α-K 로드맵 (§9 / §9.1) — 7 substep

| substep | 내용 | tier | 게이트 |
|---|---|---|---|
| **(1)** ✅ `9d858cf9` | KVCacheFormat trait + Standard/KIVIFormat impl (additive, unwired) | 없음 | **host** PASS |
| (2) | CacheManager→Stage 분해 (Eviction/KvMerge/SwapDispatch). census: pressure 계층 이미 concrete `&mut[KVCache]` → generic flip 아닌 구조 분해 | step | **device** (첫 round) |
| (3a) | eval-LL island boundary flip (ripple 격리 검증) | boundary | device |
| (3b) | LlamaModel::forward 진입점 → `&[Arc<dyn>]` + loop 직전 concrete 흡수 | layer 경계 | device |
| **(3c)** | LlamaLayer::forward + attention_into 흡수 — **★perf 위험 crux** | layer | device |
| (3d) | plan path 정렬 (execute<C> 흡수, build_plan concrete 2갈래 유지) | step | device |
| (4) | KVCacheOps 폐기 + rename (진짜 최종 perf) | 없음 | device |

- **단계적 commit**(사용자 결정): 매 device round 전 재판단. (2) 진입 = 첫 디바이스 검증.
- **(3c) crux**: layer-tier dyn vs `INV-HOTPATH-DISPATCH`. 해소 = concrete-handle 흡수(④-a) + plan `AttentionVariant` enum static 유지(④-b 연기). production hot path=plan enum이라 trait는 cold path. **호스트 GPU 부재로 device round 전까지 perf 불확정, ADR §6.5 revoke 가능**(Δ>+3% AND vtable root-cause 동시 충족 시만, 1차 대응=flamegraph).
- **디커플 발견**: β DecodeLoop 재작성은 Forward trait(boundary) 뒤라 α-K KV flip과 독립. ripple = forward-path/eval-LL/offload/plan 4 격리 island뿐(ADR §1 "~10 component" 교정).

## substep (1) 구현 핵심 (커밋 `9d858cf9`, style 선행 `5493330e`)

- 신규 `engine/src/format/kv_cache_format.rs`: KVCacheFormat **7-method** base trait(§4.1 verbatim) + `AttnDims` + `Merge`. geometry(idx/current_pos/capacity)+mutation(write_kv/write_kv_batch/compact)+attention(attention_into). **base에 needs_attn_scores/dtype/as_any 없음**(PRIMITIVE-AGNOSTIC, R4 ③).
- 신규 `engine/src/pressure/{standard_format,kivi_format}.rs`: `StandardFormat`/`KIVIFormat` = `Mutex<KVCache/KiviCache>` wrap. 내부가변성=Mutex(Send+Sync 요구, cold-path라 무관). attention_into→`backend.attention_gen` 위임. KIVI=kivi-native 게이팅 보존+F32 fallback+**AWQE 자가 흡수**. KIVI compact=no-op.
- **KIVI 편향 8 method → base 잔류 0** (adversarial 검증): absorb-attention_into 3 / drop 1(res_cap dead) / concrete-handle-inherent 4(res_pos·q2_tokens=KIVI plan, needs_flush·flush_if_needed=legacy live). 새 capability 추출 0.
- **기존 KVCacheOps/forward_gen/캐시 무변** (purely additive, unwired — production 생성·호출 0). host: build + clippy -D warnings + test --no-run + 11 unit test PASS.

## ★ wiring substep 미결 (substep (1) 무차단 — 후속 처리)
- **`Merge` 가중치 부재** → compact merge 균등 평균만. D2O Eq.11 가중은 wiring 시 해소(Merge+weight 추가 vs D2O가 pre-merge 후 compact keep-only).
- StandardFormat compact **Q4_0 merge skip**(eviction-only). KIVI compact no-op.
- attention_into **GPU score acc layer_idx 미설정**(plan-path 최적화, 후속).
- **GPU 경로(kivi-native/GPU attention) host 컴파일만** — device 검증은 (3c) wiring.

## device 게이트 절차 (substep (2)~ 재사용)
- §9.1: 5 KV 구성(Sliding/H2O/D2O/KIVI/SnapKV) × 32-tok token-id 완전일치 + avg_tbt Δ≤+3% (n≥5 median, tok0 inclusive), S25 OpenCL + Jetson CUDA. baseline = α-K 진입 commit 1회 동결.
- bin=`legacy_generate`. **진입 전 prerequisite**: (a) SnapKV 실가용성 확인(CompressHandler stub이면 게이트 vacuous → 실제 출력 내는 구성으로 재정의), (b) `verify/scenarios/` 12 YAML 회귀 게이트 재사용 가능.

## Landmines / 교훈
- **§4.1 SSOT 선독**: substep (1) trait 표면은 §4.1에 이미 완전 명시돼 있었는데 미독해로 재발견(workflow 1회 낭비). design 섹션 먼저 읽어라.
- **cargo가 권위 (재확인)**: IDE E0583(file not found)=stale(파일 존재+build clean). senior-implementer "기존파일 무변" 보고가 부정확 — `cargo fmt --all`이 α-W-4/5 fmt 잔재 5파일 정규화 + 앞선 architect workflow agent가 handoff.md 자체 편집. **git diff/cargo 직접 검증 + git status 풀 확인(grep 필터 주의) 필수.**
- α-W-4/5 fmt 잔재는 `5493330e` style로 분리 정리 완료(test_backend/tensor_partition/forward_gen/init/kivi_cache).
- **커밋 금지 untracked**: `.antigravitycli/`, `.claude/scheduled_tasks.lock`, `papers/.../microbench_*`, `.agent/todos/handoff_microbench_*.md`. 명시 파일 add(`git add -A` 금지).

## 자기점검
- 진입 문장: ✓ "α-K substep (2) 진행" (첫 device round)
- 왜 멈췄나: ✓ 사용자가 substep (1) 완료 후 세션 종료 + 다음 세션 연속 요청
- 최대 landmine: ✓ (3c) layer-tier perf(아직 멀음, (2)→(3b) 먼저) + cargo/git 직접검증
- 검증 게이트: ✓ substep (1) host PASS 수치 commit 메시지. (2)+ device 절차 명시
- device 가용: ✓ S25 USB + Jetson, §9.1 절차. (2) baseline 동결 + SnapKV 가용성 선확인 필요
