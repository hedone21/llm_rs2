# Handoff: pipeline_stage_design_v2 평가 (C1·C2·R1·R5 종결) → R4 grill

**작성**: 2026-06-02
**HEAD**: `01e3b101 docs: R5 cross-doc drift sync (spec/41 + ADR-0001 ↔ v2 SSOT)`
**브랜치**: master (worktree 아님)
**작성자**: 메인 세션 (오케스트레이터)
**다음 세션 진입 문장**: ~~**"R4 진행"**~~ → **종결됨 (아래 CLOSED 참조)**

> **✅ CLOSED 2026-06-02** — R4·R2·R3 전부 grill-me 종결. 외부 2차 평가 발견사항 **전체 종결**(C1·C2·R1·R5·R4·R2·R3). 신규 3 커밋:
> - `93f6fa75` R4 — `attention_into` 시그니처 확정(q/backend/out/`AttnDims`/scores; backend per-call; scores=생산 seam; `needs_attn_scores` 흡수; "#12" 해소→α-K). v2 §4.1.
> - `649bbd87` R2 — `INV-STAGE-ORDER-SAFETY` **폐기** + totality 미신설(compact Result 흡수). §1.2 "Safety over policy"→**"Mechanism over policy"** 재서술. v2 §0.2/§1.2/§5.3/§8 + spec/41.
> - `22b6539a` R3 — α-K substep별 중간 bit-identical 게이트 신설(게이트 강도=dispatch tier; avg_tbt를 (2)/(3)/(4) 전부). v2 §9.1 + ADR-0001 §8.3 annotation.
>
> **설계 grill 트랙 완료.** 다음 트랙 = Phase α-W *구현*(코드: `format/`·`capability/`·`hardware.rs`·`pipeline.rs`·`stages/` 신설 + `Hardware` resolve Option) 또는 backlog [P2]. 코드는 전부 미착수(설계만 확정).

---

## TL;DR

외부 2차 평가 리포트로 발견한 `arch/pipeline_stage_design_v2.md`의 **blocking 3건(C1·C2·R1) + R5(cross-doc drift)** 를 grill-with-docs로 종결, **3 커밋**. 다음 = **R4**(`attention_into` 시그니처가 "impl 단계 #12"로 연기됨 — hot path이자 ADR-0001 성능 게이트가 수렴하는 인터페이스가 미정). **멈춘 이유**: R4는 논의가 길어질 수 있어 사용자가 새 세션 분리를 요청. R2·R3는 R4 다음 순서.

---

## 진행 상태 (이번 세션 3 커밋, 전부 문서/구조 — 코드 로직 무변경)

| 이슈 | 결정 | 커밋 | 기록 |
|---|---|---|---|
| **C1** `Hardware::resolve` 반환 타입 모순(non-Option 튜플 ↔ "Npu None") | `-> Option<(&Arc<Backend>, &Arc<Memory>)>`. Cpu 항상 Some / Gpu(CPU-only 빌드)·Npu(backend 부재) None. 부재=호출자 정책(§1.2) | `3a9f104e` | v2 §3.5·§2.1·연혁 |
| **C2** §2.1 SSOT가 KVCacheFormat/WeightFormat trait *정의 파일* 미지정 | `format/` 응집 모듈 신설(L2): `format.rs` + `format/{kv_cache_format,weight_format}.rs`. impl은 `kv/`·`models/weights/`(L3). `Merge` 동시 해소 | `3a9f104e` | v2 §2.1 표 |
| **모듈 스타일** | no-mod.rs 모던 path 스타일 전역 컨벤션 + 기존 nested **38개 mod.rs sweep 실행·검증** | `3895e17d` | AGENTS.md·v2 §2.1 규칙 C |
| **R1** Pressure scalar §5.1↔§5.4 모순 | **입력 융합(전 센서 magnitude→단일 0–100) ⊥ mode 출력 분리(switch/suspend=이산 EngineCommand)**. 의도적 lossy 일원화(사용자 갈래 a) | `3a9f104e` | v2 §5.1·§5.4·CONTEXT.md·**ADR-0002 신설** |
| **R5** cross-doc drift | spec/41·ADR-0001이 폐기된 `view`/`apply_storage` 나열하던 것 sync. `LifecyclePhase` canonical→v2 §5.1 | `01e3b101` | spec/41·ADR-0001 annotation·v2 §5.1 |

**검증**: sweep = `cargo check -p llm_rs2` OK + rpcmem spec **21/21 PASS** + `cargo fmt` clean (커밋 `3895e17d`). 나머지는 문서(prose) — 빌드 영향 0.

---

## 다음 작업

**R4 grill — `attention_into` 시그니처 연기** (v2 §4.1, `arch/pipeline_stage_design_v2.md:463`):
```rust
fn attention_into(&self, /* q, backend, out, dims, scores */) -> Result<()>;
// 정확한 시그니처 = impl 단계(#12)   ← 미정
```
- **왜 teeth**: forward→KV attention 유일 진입점이자 ADR-0001(KV Generic→trait object) 수렴점. ADR-0001 게이트("S25 bit-identical + avg_tbt Δ≤+3%")가 *검증하는 인터페이스 자체가 미정*. §1.1 "layer-tier dyn 금지" ↔ ADR-0001 화해책(④-a concrete-handle fast path)과 ④-b(`AttentionVariant` enum 평탄화)도 α-K 연기(`:475`).
- **결정 2갈래**: (a) **α-K 진입 commit에서 시그니처 확정**(추천) vs (b) #12 연기 유지. + **score 출력 경로를 시그니처가 담보하나**(H2O/eviction은 attention score 필요 — 누락 시 별도 pass ~6배 오버헤드).
- **검증 게이트**: 시그니처 확정 시 기존 attention 경로(`engine/src/layers/transformer_layer/forward_gen.rs:419`의 `as_kivi_attention`, `attention_gen`, `flash_attention_forward_strided`)에서 도출 가능함을 확인 → "모르는 게 아니라 안 적은 것".

**그 후 (이번 세션 미착수)**:
- **R2** `INV-STAGE-ORDER-SAFETY`(v2 §1.2·§5.3·§8): "임의 순서 crash-safe"(창발 속성)를 "7 primitive 개별 totality 계약"으로 재배치할지.
- **R3** Phase α-K(v2 §9, ADR-0001 §8.3) 4–6주 atomic 전환에 substep별 중간 bit-identical 게이트 삽입할지.

---

## Landmines / 미해결 / 안 가본 길

- **R2·R3는 기계적 전파 불가** — 아직 *결정 안 내려진 설계 질문*이라 전파할 settled decision이 없음. 자의로 결정하지 않았음. grill 필요.
- **R5 [iii] 출처 컬럼 9곳**(spec/41 INV-DECODE-STAGE-*/KVCACHELAYER-* 등): 행별 v1 §번호→v2 §번호 재지정 **미적용** — §3.28 intro **일괄 redirect 주석**으로 대응. 잘못된 §번호 9곳 주입이 stale 포인터보다 나쁘다 판단. v1↔v2 절 정밀 매핑이 필요하면 후속(spec-manage).
- **"#12" dangling 참조**: v2 §4.1 attention_into 주석의 "impl 단계(#12)"는 가리키는 단계 목록이 문서에 없음 — R4에서 실제 단계/commit으로 명시 필요.
- **전부 문서 결정, 코드 미적용**: C1 resolve Option, C2 `format/`, R1 PressureSource/CommandSource는 **Phase α-W/α-K 구현 대기**. 현 코드엔 미반영(`format/`·`capability/`·`stages/`·`kv/`·`weight/` 전부 미생성, `Hardware`·`DeviceTarget` 미존재).
- **38 mod.rs sweep 부수효과**: `backend/opencl.rs`의 `include_str!` 25개 경로 `../../../kernels`→`../../kernels` 정정됨(디렉토리 한 단계 상승). 향후 nested 모듈 이동 시 동일 상대경로 함정 주의. spec test 3개(rpcmem_001/002/008)도 `opencl/mod.rs`→`opencl.rs` 경로 갱신됨.
- **ADR 상태**: ADR-0002(pressure lossy unification) Accepted 신설. ADR-0001은 본문 보존 + annotation(`:43`, method 집합 진화 → 현재 7 method, SSOT=v2 §4.1).
