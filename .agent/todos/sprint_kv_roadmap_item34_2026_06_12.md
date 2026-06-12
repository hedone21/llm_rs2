# Sprint: KV 로드맵 항목 3+4 — QueryStats TensorKind 구현 + read-plan ADR (워크트리 분리 게이트)

**작성**: 2026-06-12 (PM)
**상위 트랙**: `[트랙] KV 캐시 관리 확장성 로드맵` (`backlog.md` L865~947, 2026-06-10 등록) — 항목 3(L901) + 항목 4(L908)
**선행 스프린트**: 항목 0 측정 스프린트 ✅ 종결(`sprint_kv_roadmap_item0_2026_06_12.md`) — 판정 4종 확정, D4 결정으로 인프라 항목 3·4는 1B 게이트 RED와 **무관하게 진행**
**진입 handoff (정본)**: `handoff_kv_roadmap_item34_entry_2026_06_12.md` (작업 순서·금지선·landmine 6건)
**진입 문장**: **"KV 로드맵 항목 3 진행 — QueryStats TensorKind 구현부터"** (항목 4 ADR은 3 완료 후 같은 흐름에서)
**Status**: ✅ **종결 (2026-06-12)** — P1~P5 전부 DONE. "KV 구조 확정" 게이트 도달 + 분리 위생 3종 완료(① 푸시는 종결 커밋과 함께 수행) → 사용자 병렬 리팩토링 분기 준비 완료
**ADR 예고 자리**: `docs/adr/0004-kvcachestage-plan-returning-trait.md` §7 L121 — `query_state(Quest): decode-step Q 캡처 미배선 → drop(후속 PR)` (= 본 항목 3이 닫는 자리)

---

## TL;DR

항목 0 측정 종결 후 차기 = **항목 3(QueryStats TensorKind) 구현 → 항목 4(read-plan ADR)**. 이 둘이 끝나는 시점이 사용자가 정의한 **"KV 캐시 구조 확정" 게이트** — 직후 사용자는 별도 워크트리에서 대형 리팩토링을 병렬 시작한다. 따라서:

1. **항목 3은 구현까지** (실모델 e2e 1회 + α-K frozen 재검증 포함).
2. **항목 4는 ADR 작성까지만** — **구현 착수 절대 금지**. ADR-선행의 목적은 미래 표면(read-plan trait)을 **문서 제약으로 고정**해 병렬 리팩토링과의 의미적 충돌을 예방하는 것. read-plan 구현은 리팩토링 머지 후 별도 항목으로 재등록(ADR 산출물 = backlog 재등록).
3. **분리 위생 3종** (항목 3+4 완료 직후): ① origin 푸시 ② 공통 회귀 게이트 명문화 ③ rename 상호 배제 규칙.

**순서 고정 (사용자 결정 2026-06-12, 변경 불가)**: 항목 3 구현 → 항목 4 ADR → 분리 위생 3종.

---

## 사용자 결정 (2026-06-12 확정 — 변경 불가)

| ID | 결정 | 근거 |
|----|------|------|
| **U1** | 순서 = 항목 3 구현 → 항목 4 ADR → 분리 위생 3종 | "KV 구조 확정" 게이트를 명확한 마일스톤으로 — 직후 병렬 리팩토링 시작. |
| **U2** | 항목 4는 **ADR까지** — 구현 착수 금지 | ADR-선행 = 미래 표면을 문서 제약으로 고정해 의미적 충돌 예방. 구현은 리팩토링 머지 후 별도 등록. |
| **U3** | Q 캡처는 **기본 off / score-active 시만** | hot path 비용 0 원칙(v1 `need_scores` 교훈). score-active는 이미 plan path 우회 경로(`model_forward.rs:471` `score_active` 게이트)라 캡처 hook이 자연 정합. |
| **U4** | accumulator 영역 **commit 격리 + 누적 로직 무수정** | `attention_scores.rs`/`qcf_runtime.rs`가 QCF_kv 설계 라운드(`backlog.md` L1112, 공동 검토 대기)와 겹침 — 동시 작업 회귀 추적 위험. |
| **U5** | 완료 게이트에 **실모델 e2e 1회 필수** | 항목 0 교훈 — 단위 게이트만으로 완료 선언 금지(R-KV/A2SF 미배선 허상 사례). |
| **U6** | 완료 게이트에 **α-K frozen 재검증 필수** | forward hot path 접촉 → happy path 무회귀 증명(3-dtype byte-identical). |

---

## 목표

- **항목 3**: `TensorKind::QueryStats` variant + 엔진 누적 배선(per layer·kv_head Q running mean/var, forward 경로 Q 캡처 1지점) + host 통계 정확성 테스트 + 기존 `tensor()` 소비자 무영향 + 실모델 e2e 1회 + α-K frozen 재검증. ADR-0004 §7이 예고한 자리를 닫는다.
- **항목 4**: read-plan 표면 ADR 1건 — `KVReadStage::read_plan(ctx) → KVReadPlan{granularity, select}`. 대안 비교 + grill 통과 + **구현 단계 분해를 backlog 재등록까지만** (구현 착수 금지).
- **분리 위생 3종**: 병렬 워크트리 분기 직전 공통 베이스 정리(푸시 + 회귀 게이트 명문화 + rename 배제 규칙).

---

## 코드 접점 (사전 조사 완료 — 2026-06-12)

| 영역 | 위치 | 역할 |
|---|---|---|
| TensorKind enum | `crates/technique-api/src/lib.rs` L33 (Key/Value/AttnWeights/Scores, `#[repr(u32)]`) | **QueryStats variant 추가 지점** — `#[repr(u32)]` discriminant 4→5 (가산적, 기존 0~3 불변) |
| TensorShape | `crates/technique-api/src/lib.rs` L48 (`#[repr(C)]`, rows/cols/per_head) | QueryStats shape 정의(per layer·kv_head running mean/var — cols/per_head 의미 P1 결정) |
| StageCtx::tensor() | `crates/technique-api/src/lib.rs` L123 (trait 메서드) | default sugar는 무수정 — `tensor(QueryStats)` 접근만 추가 |
| AbiStageCtx | `crates/technique-api/src/lib.rs` L466~ (kind 인덱스 4종 하드코딩, `handles: [_; 4]`) | **배열 크기 4→5 + kinds 배열에 QueryStats 추가** — C-ABI 가산적 확장 |
| KVStageCtx (엔진 impl) | `engine/src/kv/eviction/stage_registry.rs` L197 struct + L260 `tensor()` match | **QueryStats 핸들 필드 추가 + match arm 추가** (handoff의 `stage_registry.rs L197/L260`과 일치) |
| 캡처 패턴 재사용 | `engine/src/inference/attention_scores.rs` `AttentionScoreAccumulator` (L11 struct, L366 `last_step_head_attn`) | Q running mean/var 누적기의 **패턴 참조** — 누적 로직 자체는 **무수정**(U4), 신규 누적 필드/메서드는 commit 격리 |
| forward Q 계산 | `engine/src/layers/transformer_layer.rs` (RoPE 직후 Q) + score 누적 진입 `engine/src/session/forward/model_forward.rs` L471 `score_active` 게이트 | **Q 캡처 1지점** — score-active 시만(U3). plan path 우회 시맨틱과 정합(L466~470 주석) |

> **핵심 정합**: `score_active`(`model_forward.rs:471`)가 active이면 이미 plan path를 우회하고 `forward_into` layer loop로 폴백한다. Q 캡처를 이 경로(score-active)에 한정하면 (a) hot path 비용 0(U3), (b) plan path와의 시맨틱 충돌 없음 — 둘 다 충족. P1 설계가 이 정합을 SSOT로 고정해야 함.

---

## Phase 분해

| Phase | Owner | 산출물 | 의존성 | 상태 |
|---|---|---|---|---|
| **P1** 항목 3 설계 + spec triage | Architect | QueryStats 설계 노트(variant/shape/캡처 seam/off-게이트) + spec 영향 판정 | 없음 | ✅ DONE (2026-06-12, `docs/adr/0004*.md` §10 M-Q amendment — 결정 6건 MQ-1~6 + 캡처 seam 코드검증 §10.1 + P2 task 7분해 §10.2 + 테스트 명세 TQS-1~9 §10.3 + Spec Triage = spec 무관/새 INV 불요. arch 별도 파일 미신설(ADR=SSOT), `arch/README.md` 등록) |
| **P2** 항목 3 구현 | Senior Implementer | TensorKind variant + AbiStageCtx 확장 + KVStageCtx 핸들 + Q 누적 배선(off 기본/score-active) | P1 | ✅ DONE (2026-06-12, 커밋 `783bcadd` technique-api + `a98cd679` 엔진 배선 — QueryStats 전용 격리 2커밋. attention_scores.rs/qcf_runtime.rs 누적 로직 무수정 U4 충족. P2-5 편차: production 소비자 부재로 QueryStatsCell 미신설, `query_stats_accumulator: None` 명시 전달로 충족 — YAGNI) |
| **P3** 항목 3 검증 | Tester + Implementer | host 통계 정확성 테스트 + 기존 `tensor()` 소비자 무회귀 + α-K frozen 3-dtype byte-identical + 실모델 e2e 1회 | P2 | ✅ DONE (2026-06-12, **전 게이트 PASS**: G1 host lib 1369/2-baseline + technique-api 20/0 + TQS-1~9·compact_parity 8·d2o_stage_eq 3 PASS / G2 실모델 e2e llama3.2-1b f16 `query_stats.non_empty=true` 128행(16L×8kv) 전부 비0·finite + pre-change(757ca2a8) 대비 greedy token-id **byte-identical** / G3 S25 α-K frozen 3-dtype 생성출력 byte-identical + avg_tbt Δ f16 −0.26%·f32 −0.04%·q4 −0.50% 전부 음수 / G4 spec 680/4-baseline 신규 실패 0) |
| **P4** 항목 4 read-plan ADR | Architect | ADR 신규 1건(대안 비교 + grill) + 구현 단계 분해를 backlog 재등록 (**구현 착수 금지**) | P3 (항목 3 완료 = KV 구조 확정 일부) | ✅ DONE (2026-06-12, `docs/adr/0011-kv-read-plan-surface.md` Proposed — `KVReadStage::read_plan(ctx)→Option<KVReadPlan{granularity: Token\|Page, select}>` 4번째 plan-returning 형제 + `SelectiveRead` capability opt-in(미지원 full read 폴백) + page 메타=read stage 자기 갱신(tensor(Key)+QueryStats) + read 비변형이라 eviction과 race 없음. grill 충족: status quo+5대안 기각, RPN≥100 4건 완화(최대 R1=378 layer-tier hot-path — dispatch 전략은 구현 amendment로 위임), Premortem·DA·C-ABI·spec 무관 판정. backlog 항목 4 RESOLVED + 신규 "4-impl"(S1~S6, 리팩토링 머지 후 착수) 재등록. **구현 코드 0줄 git 확인**) |
| **P5** 분리 위생 3종 | PM + Implementer | ① origin 푸시 ② 공통 회귀 게이트 명문화 ③ rename 배제 규칙 | P4 | ②③ ✅ DONE (2026-06-12, PM — `worktree_split_hygiene_2026_06_12.md` 신설: ② 게이트 3항 재앵커 명문화 + ③ rename 상호 배제 규칙 고정) / ① origin 푸시 = **메인 세션 수행** |

**병렬 가능성**:
- P1(설계)과 P4(ADR)는 둘 다 Architect 작업이나 **순서 고정**(U1) — P4는 P3 완료(항목 3 구현 완결) 후 착수. 항목 4 설계가 항목 3의 QueryStats를 "신호 공급원"으로 참조하기 때문(backlog L905 시너지).
- P3 내부: host 단위 테스트 + α-K frozen은 host 게이트라 동시, 실모델 e2e 1회는 그 직후.

---

## Phase 상세

### [P1] 항목 3 설계 + spec triage (Architect)
- **Status**: TODO / **Sprint**: current / **Dependencies**: 없음
- **Owner**: Architect
- **Description**: QueryStats TensorKind 추가의 설계를 고정한다. PM은 설계 결정을 내리지 않는다 — 전부 Architect 위임.
  - **variant 정의**: `TensorKind::QueryStats` discriminant(4, `#[repr(u32)]` 가산). `TensorShape` 의미(per layer·kv_head Q running mean/var — cols=head_dim×2? mean/var 분리? per_head=true 여부) 고정.
  - **캡처 seam**: forward 경로 Q 캡처 1지점 위치(`transformer_layer.rs` RoPE 직후 Q vs 직전 — 통계 의미 결정). `AttentionScoreAccumulator` 패턴 재사용 방식(신규 누적 필드 격리, 기존 누적 로직 **무수정** U4).
  - **off-게이트 정합**: 캡처를 score-active 경로에만 배선하는 설계(U3). `model_forward.rs:471` `score_active` 게이트 + plan path 우회 시맨틱과의 정합 명문화 — happy path(score 비활성) 비용 0 보장.
  - **AbiStageCtx 확장**: `handles: [_; 4]` → `[_; 5]` + kinds 배열 추가의 C-ABI 가산성 확인(기존 `.so` 호환).
  - **spec triage**: technique-api는 plugin 어휘(spec 영향 가능) — TensorKind 추가가 어느 spec INV/표면 문서를 건드리는지 판정. `/spec-manage`로 ID 영향 분석.
- **Acceptance Criteria**: QueryStats 설계 노트 1건(variant/shape/캡처 seam/off-게이트 정합 4항 고정) + spec 영향 판정(수정 대상 spec 목록 또는 "영향 없음" 명시). P2 착수 차단 해제 게이트.
- **Notes**: 설계 노트는 `arch/` 또는 `docs/`에 Architect가 배치(PM 범위 밖). off-게이트 정합이 hot path 리스크의 핵심 — 측정/설계 전 고정.

### [P2] 항목 3 구현 (Senior Implementer)
- **Status**: TODO / **Sprint**: current / **Dependencies**: P1
- **Owner**: Senior Implementer (forward hot path + score accumulator 접촉 = QCF/qcf 영역 인접)
- **Description**: P1 설계대로 QueryStats 배선.
  - **technique-api** (`crates/technique-api/src/lib.rs`): `TensorKind::QueryStats` variant(L33) + `AbiStageCtx` 배열 4→5 확장(L467/L482) + kinds 배열 추가(L476). default sugar는 무수정.
  - **엔진 impl** (`engine/src/kv/eviction/stage_registry.rs`): `KVStageCtx` struct에 QueryStats 핸들 필드 추가(L197) + `tensor()` match arm 추가(L260) + `new()` 공급 경로.
  - **Q 누적 배선**: per layer·kv_head Q running mean/var 누적기(신규, `AttentionScoreAccumulator` 패턴 참조). forward 경로 Q 캡처 1지점 — **score-active 시만**(U3). 기존 `AttentionScoreAccumulator` 누적 로직 **무수정**(U4) — 신규 누적은 별도 필드/구조체로 격리.
- **⚠ commit 격리 제약 (필수, U4)**: `attention_scores.rs`/`session/qcf_runtime.rs`는 `backlog.md` L1112 **QCF_kv 정규화 비대칭 설계 라운드**(공동 검토 대기)가 만질 예정. Q 캡처 배선은 (a) 기존 누적 로직 무수정, (b) commit을 QueryStats 전용으로 격리(QCF 변경과 한 commit에 섞지 않음). 두 작업이 같은 표면을 동시 건드리면 회귀 추적 불가.
- **Acceptance Criteria**: 빌드 GREEN + `tensor(QueryStats)`가 score-active 시 `Some`, 비활성 시 `None` + 기존 0~3 kind 동작 불변 + clippy --workspace clean + 자동 커밋(QueryStats 전용 격리 commit).
- **Notes**: AbiStageCtx 배열 확장은 C-ABI 가산적(기존 `.so` 무수정 호환) — P1이 가산성 확인. score-active 게이트가 hot path 비용 0의 유일 근거.

### [P3] 항목 3 검증 (Tester + Implementer)
- **Status**: TODO / **Sprint**: current / **Dependencies**: P2
- **Owner**: Tester (게이트 실행/판정) + Implementer (host 통계 테스트 작성)
- **Description**: 항목 3 완료 게이트를 수치/명령으로 충족 증명.
  - **host 통계 정확성 테스트**: Q running mean/var가 ground-truth(수동 계산 reference)와 일치 — per layer·kv_head 좌표계 정합. dtype 분기(F32 최소, 가능 시 F16/Q4_0). `cargo test -p llm_rs2` lib 게이트 통과.
  - **기존 `tensor()` 소비자 무영향**: 0~3 kind(Key/Value/AttnWeights/Scores) 소비 경로(d2o/CAOTE host 테스트/compact_parity) 무회귀 — `compact_parity` + `d2o_stage_eq_handler_*` GREEN.
  - **α-K frozen 재검증** (U6, forward hot path 접촉): α-K frozen 3-dtype byte-identical 게이트 통과 — happy path(score 비활성) 출력이 변경 전과 비트 동일. 절차: `project_pipeline_alpha_w.md` / α-K frozen census 경로.
  - **실모델 e2e 1회** (U5): 실모델(llama3.2-1b GGUF) 1회 추론에서 score-active 경로(h2o/d2o eviction 등)로 Q 캡처가 실제 채워지는지 + greedy token-id 무변경 확인. 단위 게이트만으로 완료 선언 금지(항목 0 미배선 허상 교훈).
- **Acceptance Criteria** (전 항목 수치/명령):
  - host lib: `cargo test -p llm_rs2` GREEN (신규 통계 테스트 포함, lib N/0).
  - 무회귀: `compact_parity` + `d2o_stage_eq_handler_*` PASS.
  - α-K frozen: 3-dtype(F32/F16/Q4_0) byte-identical PASS.
  - 실모델 e2e: 1회 추론 + Q 캡처 non-empty 확인 + greedy token-id 무변경.
  - fmt + clippy --workspace clean.
- **Notes**: host 단독 충분(품질·정확성 지표는 디바이스 무관 — 항목 0 D2 선례). device 게이트 불요. e2e가 "미배선 허상" 방지의 핵심.

### [P4] 항목 4 read-plan 표면 ADR (Architect) — ★구현 착수 금지
- **Status**: TODO / **Sprint**: current / **Dependencies**: P3 (항목 3 완료 = KV 구조 확정 일부 + QueryStats가 신호 공급원)
- **Owner**: Architect
- **Description**: read-plan 표면 ADR 작성. **ADR까지가 본 Phase의 전부 — 구현 코드 0줄**(U2).
  - **표면 정의**: `KVCacheStage`와 대칭인 plan-returning trait `KVReadStage::read_plan(ctx) → KVReadPlan{granularity, select}`. `attention_into`가 캐시 전체 읽기를 가정하고 selection 인자가 없는 한계를 해소(Quest류 선택 읽기 + InfiniGen/KVSwap류 prefetch).
  - **핵심 통찰 반영**: layer i−1 read plan = Quest에겐 읽기 마스크, KVSwap에겐 layer i prefetch 목록 — 표면 하나로 두 군집(~9건) 동시 해소(backlog L913).
  - **설계 쟁점**: page 메타데이터(K min/max) 유지 주체 — 1차안 = read stage가 `tensor(Key)`(+항목 3 `tensor(QueryStats)`)로 자기 상태 incremental 갱신, 코어 무수정. format `attention_into_selected` capability opt-in(미지원 format full read 폴백).
  - **대안 비교 ≥2 + status quo**: `/review` 골격(Alternatives ≥2 + status quo / Risks RPN≥100 / Premortem / Devil's Advocate). grill 통과 필수.
  - **구현 단계 분해 → backlog 재등록**: ADR 산출물로 read-plan 구현 단계를 `backlog.md`에 별도 항목으로 재등록(리팩토링 머지 후 착수). 본 Phase는 등록까지.
- **Acceptance Criteria**: ADR 신규 1건(`docs/adr/`에 Architect 배치, 대안 ≥2 비교 + status quo + grill 통과) + 구현 단계 분해를 `backlog.md`에 재등록(항목 4 자리 또는 신규 항목) + **구현 코드 0줄 확인**.
- **Notes (★금지선)**: 구현 착수 = 병렬 리팩토링과의 의미적 충돌 위험(U2). ADR-선행의 목적이 미래 표면을 문서 제약으로 고정하는 것 — 구현하면 목적 자체가 무너진다. 항목 2(K/V 비대칭 merge)·항목 5(persistence)도 동일 이유로 리팩토링 머지 후 권고(backlog 표기 유지).

### [P5] 분리 위생 3종 (PM + Implementer)
- **Status**: ②③ ✅ DONE (2026-06-12) / ① = 메인 세션 수행 / **Sprint**: current / **Dependencies**: P4 (항목 3+4 완료 직후)
- **Owner**: PM(②③ 합의·명문화) + Implementer(① 푸시 실행)
- **Description**: 병렬 워크트리 분기 직전 공통 베이스 정리(handoff L27).
  - **① origin 푸시**: 누적된 미푸시 commit(QCF 세션분 + 측정 스프린트분 + 항목 3 + 항목 4 ADR)을 origin/master로 푸시 — 공통 베이스 앵커(handoff "origin 미푸시 누적" 해소).
  - **② 공통 회귀 게이트 명문화**: 양 브랜치(메인 + 리팩토링 워크트리) 머지 판정 공통 기준을 문서로 고정 — (a) host lib `cargo test -p llm_rs2` N/0, (b) α-K frozen 3-dtype byte-identical, (c) S25 verify 매트릭스 28/30(known-fail 2 = QCF_kv 결함, 별도 라운드). 어디에 명문화할지 PM이 배치(`.agent/todos/` 또는 handoff).
    - **⚠ P3 발견 (2026-06-12)**: `frozen_baseline_alpha_k_5f_2026_06_05.md`의 sig md5 3종(`304f4ada/684d01d9/1cfba273`)은 보존된 `.out`로부터 **추출 방식이 재현 불가**(추출 스크립트 미기록) — P3 G3는 device 보존 baseline `.out`과 결정론 라인 직접 byte 비교로 대체 판정함. ② 명문화 시 **게이트 정의를 재앵커**할 것(비교 대상 파일 + 추출/비교 명령을 명시, md5 사전계산값 의존 제거).
  - **③ rename 상호 배제 규칙**: 양 브랜치가 동일 파일을 `git mv`하면 머지 충돌 폭발 → 한쪽만 rename하는 규칙 합의(no-`mod.rs` sweep 등 대형 rename은 리팩토링 워크트리 전담). 규칙 문서화.
- **Acceptance Criteria**: ① origin/master 푸시 완료(`git log origin/master`로 확인) + ② 공통 회귀 게이트 3항 명문화 문서 1건 + ③ rename 배제 규칙 합의 기록 1건.
- **진행 (2026-06-12)**:
  - **②③ ✅ DONE (PM)**: `worktree_split_hygiene_2026_06_12.md` 신설. ② 공통 회귀 게이트 3항 = (a) host lib `cargo test -p llm_rs2 --lib` 신규 실패 0(baseline 실패 2 + OpenCL GPU 부재 21 제외) + technique-api 20/0 + fmt/clippy clean, (b) α-K frozen 3-dtype byte-identical — **재앵커 적용**(md5 사전계산값 의존 제거 → device 보존본 `blA_argus_{f16,f32,q4}.out` 결정론 라인 직접 byte 비교 + tbt Δ≤+3%), (c) S25 verify 28/30(known-fail 2 = QCF_kv 결함, 머지 차단 아님). ③ rename 상호 배제 = 대형 `git mv`는 리팩토링 워크트리 전담 + 메인 KV 표면 동결(plan ABI/format trait/technique-api) + QCF_kv 라운드는 메인 진행 가능 + 충돌 우선순위(구조=워크트리/내용=메인).
  - **① origin 푸시 = 메인 세션 수행** (PM 범위 밖 — 본 스프린트 미푸시 commit 누적분).
- **Notes**: ②③는 코드 0줄(문서/규칙). ①만 푸시 실행(메인 세션). 이 Phase 완료 = 사용자 병렬 리팩토링 분기 준비 완료(= "KV 구조 확정" 게이트 종결).

---

## 담당 에이전트 매핑 (요약)

| Phase | 에이전트 | 사유 |
|---|---|---|
| P1 항목 3 설계 | Architect | TensorKind 어휘 추가 = 설계/spec triage |
| P2 항목 3 구현 | Senior Implementer | forward hot path + score accumulator 인접(qcf 영역) |
| P3 항목 3 검증 | Tester + Implementer | host 게이트 실행 + 통계 테스트 작성 |
| P4 항목 4 ADR | Architect | read-plan 표면 설계(구현 금지) |
| P5 분리 위생 | PM + Implementer | 규칙 명문화(PM) + 푸시(Implementer) |

---

## 제약 (필독)

1. **순서 고정 (U1)**: 항목 3 구현 → 항목 4 ADR → 분리 위생 3종. 병렬 단축 금지.
2. **★항목 4 구현 착수 금지 (U2)**: ADR까지가 본 항목. 구현 = 병렬 리팩토링과 의미적 충돌. read-plan 구현은 리팩토링 머지 후 별도 등록.
3. **Q 캡처 hot path 비용 0 (U3)**: 기본 off / score-active 시만. `model_forward.rs:471` `score_active` 게이트 정합 — happy path 무비용. α-K frozen 재검증으로 증명(P3).
4. **accumulator commit 격리 (U4)**: `attention_scores.rs`/`qcf_runtime.rs`는 `backlog.md` L1112 QCF_kv 설계 라운드와 겹침. 기존 누적 로직 무수정 + QueryStats 전용 commit 격리.
5. **실모델 e2e 1회 필수 (U5)**: 단위 게이트만으로 완료 선언 금지(항목 0 미배선 허상 교훈).
6. **α-K frozen 재검증 필수 (U6)**: forward hot path 접촉 → 3-dtype byte-identical(happy path 무회귀).
7. **AbiStageCtx 확장은 C-ABI 가산적**: `handles: [_; 4]`→`[_; 5]`는 기존 0~3 discriminant 불변 → 기존 `.so` dlopen 호환(P1 확인).
8. **spec triage 선행 (P1)**: technique-api 어휘 추가라 spec 영향 판정 먼저.

---

## 리스크

| ID | 리스크 | 영향 | 완화 |
|----|--------|------|------|
| **R1** | **forward hot path 회귀** — Q 캡처가 score 비활성(happy path)에 비용을 더함 | TBT 저하(production 직격) | U3 score-active 게이트 + U6 α-K frozen 3-dtype byte-identical 게이트(P3). 설계(P1)가 off-게이트 정합을 SSOT로 고정. |
| **R2** | **accumulator 머지 충돌** — L1112 QCF_kv 설계 라운드가 같은 `attention_scores.rs`/`qcf_runtime.rs` 동시 작업 | 회귀 추적 불가 | U4 기존 누적 로직 무수정 + QueryStats 전용 commit 격리. 두 작업 commit 순서/격리 상태를 handoff에 명시(P5). |
| **R3** | **항목 4 구현 유혹** — ADR 작성 중 "바로 구현하면 빠르다"로 U2 위반 | 병렬 리팩토링 의미적 충돌 | P4 AC에 "구현 코드 0줄 확인" 명시. ADR-선행 목적(문서 제약 고정)을 Notes에 재강조. |
| **R4** | **워크트리 게이트 일정** — 항목 3+4가 늦으면 사용자 병렬 리팩토링 착수 지연(전체 일정 블로커) | 일정 직격 | 항목 3은 단일 variant + 캡처 1지점(범위 작음). 항목 4는 ADR(설계 작업, 구현 없음)로 가벼움. P1·P4 Architect 연속 배치로 설계 컨텍스트 유지. P5 분리 위생이 명확한 종결점. |
| **R5** | **e2e 미배선 허상** — 항목 0에서 P2 구현 4종 전부 e2e 미배선으로 1차 측정 오염된 선례 | 완료 오선언 | U5 실모델 e2e 1회 필수 — Q 캡처 non-empty 실제 확인. host 단위 테스트만으로 완료 금지. |
| **R6** | **origin 미푸시 로컬 유일본** — QCF 세션분 + 측정 스프린트분 누적 미푸시 | 유실 위험 | P5 ① origin 푸시가 해소. 그 전까지 로컬 유일본 인지(handoff landmine). |

---

## 동결 권고 (리팩토링 기간 KV 표면 무변경 보장)

- 항목 2(K/V 비대칭 merge)·항목 5(persistence)는 리팩토링 머지 후로 — 리팩토링 기간 중 KV 표면(plan ABI·format trait) 무변경 보장이 목적(handoff landmine, backlog 해당 항목 표기 유지).
- 측정 인프라 신규 표면(rkv feature, `--score-decay`, `--dump-a2sf`, C_h 덤프)은 전부 기본 off — 리팩토링이 정리 대상으로 오인 금지(측정 재현용 보존).
