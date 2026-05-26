# Handoff: precision swap §13.8-O 갈래 1 Sprint A/B 완료 → Sprint B-fixup 진입

**작성**: 2026-05-26
**HEAD**: `cea6c594 docs(arch): shared/ 페이퍼 목표 폐기, L2 운영 모델 정직 표기 (옵션 C)`
**브랜치**: `master`
**작성자**: 메인 세션 (오케스트레이터)

**다음 세션 진입 문장**: "Sprint B-fixup 진행"

---

## TL;DR

precision swap §13.8-O 갈래 1 통합 sprint A(trait 격상)/B(ModelConfig L2 격상)/위계 명시 모두 완료. **단 Sprint B 부작용으로 INV-LAYER-002 위반 2건 발생** — `ModelConfig::from_gguf_metadata`가 L3 GGUF parser 의존. 옵션 (i) 함수 이동(`models/loader/gguf.rs`)으로 정리 후 Sprint C(weight_slot pressure 이전) 진입. C 시작 전 design doc 재논의 (사용자 결정 #2, 2026-05-26).

---

## 진행 상태

### Task

| ID | 상태 | 작업 | Commit |
|---|---|---|---|
| #0 | ✅ completed | 위계 명시 (INV-LAYER-003 보조 + §13.8-O 우선순위) | `ab9e1f35` |
| #1 | ✅ completed | Sprint A — KVCacheOps default 제거 + PreloadAccess trait 신설 | `a9dcb5be` |
| #2 | ✅ completed | Sprint B 코드 — ModelConfig → engine/src/model_config.rs (engine 직속) | `6dcba548` |
| #3 | ✅ completed | Sprint B arch — shared/ 폐기, L2 운영 모델 정직 표기 (옵션 C) | `cea6c594` |
| **#4** | **⏳ 이번 세션 진입 대상** | **Sprint B-fixup — `from_gguf_metadata` L3 이전 (옵션 i)** | - |
| #5 | ⏳ blocked-by #4 | Sprint C design doc 재논의 + architect 위임 | - |
| #6 | ⏳ blocked-by #5 | Sprint C 본 작업 (weight_slot pressure 이전) | - |

### 측정 / 검증 결과

| 항목 | 값 | 비고 |
|---|---|---|
| `cargo check -p llm_rs2` (release) | PASS | rust-analyzer diagnostics는 stale cache (실제 PASS) |
| `cargo test --lib -p llm_rs2` | 1223 PASS / 18~22 FAIL | OpenCL 관련만 (host GPU 부재 정상) |
| `layer_lint.py` 총 위반 | 6 → **8건** | -1 ModelConfig escape + **+2 GGUF L3 의존** |
| WeightSwapHandler §13.8-O marker | 3 → 2건 | 1건(ModelConfig) 해소 |
| Android S25 게이트 | 미실시 | 사용자 결정 #3에 따라 A/B host gate만 |

### 신규 위반 (Sprint B 부작용, fixup 대상)

```
INV-LAYER-002  model_config.rs:200  crate::models::loader::gguf::GgufFile
INV-LAYER-002  model_config.rs:236  crate::models::loader::gguf::GgufValue::Array
```

### Sprint A 잔여 (backlog 흡수 대상, fixup 대상 아님)

| 잔여 import | 위치 | 처리 |
|---|---|---|
| `PreloadResult` | `models/transformer.rs:18` | §G L2 promotion 또는 trait associated type |
| `preload_erased` | `models/transformer.rs:20` | trait method 통합 또는 caller inversion |
| `PrefetchController` | `models/transformer.rs:2809` | 별 trait 격상 sprint |

→ 기존 backlog `[P2] §13.8-O cross-L3 vocabulary trait inversion`의 "PrefetchAccess + PreloadPool L2 격상" 트랙으로 흡수. PreloadAccess는 sprint A에서 격상 완료, 잔여 marker 3건은 후속 sprint.

---

## 다음 작업 (다음 세션이 그대로 사용 가능)

### 액션

1. **Sprint B-fixup — `from_gguf_metadata` L3 이전 (옵션 i)** → 검증: `python3 scripts/layer_lint.py` 위반 수 6건 (sprint A 직후 수준) 복귀
2. **Sprint C 시작 전 design doc 재논의** (사용자 결정 #2, 2026-05-26) → 사용자에게 옵션 제시: design doc architect 위임 vs ad-hoc 진행
3. **Sprint C 본 작업 위임** → 검증: WeightSwapHandler 잔여 2 marker 해소 + S25 bit-identical 게이트

### 위임 prompt — Sprint B-fixup

> **에이전트**: `implementer`
> **모델**: `sonnet`
> **권한**: 수정 가능 — `engine/src/model_config.rs`, `engine/src/models/loader/gguf.rs`, 11 호출자 파일

```
Sprint B-fixup — ModelConfig::from_gguf_metadata L3 이전

목적: Sprint B(커밋 6dcba548)에서 ModelConfig를 engine/src/model_config.rs (L2)로 격상했으나, ModelConfig::from_gguf_metadata가 L3 GgufFile/GgufValue를 직접 사용하여 INV-LAYER-002 위반 2건 발생. 함수 이동으로 정공법 해소.

타깃 위반:
- model_config.rs:200  crate::models::loader::gguf::GgufFile
- model_config.rs:236  crate::models::loader::gguf::GgufValue::Array

작업:
1. engine/src/model_config.rs:200 `impl ModelConfig { pub fn from_gguf_metadata(...) }` 함수 전체(line 200~끝까지의 from_gguf_metadata block)를 engine/src/models/loader/gguf.rs로 이동
2. 새 위치 결정 (implementer 재량):
   - 옵션 a: free fn `pub fn parse_model_config(gguf: &GgufFile) -> Result<ModelConfig>`
   - 옵션 b: `impl GgufFile { pub fn to_model_config(&self) -> Result<ModelConfig> }`
   - 권장 a (free fn, 단방향 변환 명확)
3. ModelConfig 내부 field가 private이면 visibility 정정 (필요 시 builder/setter 추가) — 단 외과적 변경 원칙 (CLAUDE.md §3) 준수
4. 11 호출자 갱신:
   - engine/src/models/loader/gguf.rs (8건, in-file 호출은 자연 정합)
   - engine/src/models/weights/secondary_mmap.rs:860
   - engine/src/models/weights/rpcmem_secondary.rs:223
   - engine/src/bin/auf_tool.rs:557, 720

검증 게이트:
- cargo build --release -p llm_rs2 -p llm_shared -p llm_manager PASS
- cargo test --lib -p llm_rs2 회귀 0 (OpenCL 실패는 기존과 동일)
- cargo fmt --all + cargo clippy --workspace -- -D warnings PASS
- python3 scripts/layer_lint.py:
  - model_config.rs INV-LAYER-002 위반 2건 → 0건
  - 총 위반 8 → 6건 (sprint A 직후 수준 복귀)
  - 새 위반 발생 0

제약 (CLAUDE.md §3):
- 함수 이동만. 인접 코드 리팩토링 금지.
- ModelConfig struct 자체는 변경 없음 (data definition 유지)
- visibility 정정 필요 시 최소 범위만

커밋: refactor(model_config): Sprint B-fixup — from_gguf_metadata L3 이전 (INV-LAYER-002 위반 2건 해소)

완료 시 notify-send "llm.rs" "Sprint B-fixup 완료" + 보고.
```

### Sprint C 진입 전 design 결정 항목 (참고 — fixup 후 사용자에게 다시 제시)

사용자 결정 #2 (2026-05-26): "C 시작 전에 다시 논의". 풀어야 할 핵심 질문:

1. **`models/weights/` 16개 파일 도메인 분류**:
   - pressure 측 이전: swap_executor, phase_aware_swap, async_swap, release_worker, intra_forward_swap, dynamic_k, probing_k, incremental_plan, decider, noise_table (swap orchestration)
   - pressure state: rpcmem_secondary, secondary_mmap, slot, backing
   - backend 측 (§13.8-B 기 결정): layer_object_pool → backend/cuda_embedded/pool.rs
2. **인접 의존 처리 (cross-L3 재발 risk)**:
   - `models/loader/gguf/*` (4 파일 import) — GGUF parsing은 inference 잔존
   - `layers/transformer_layer::TransformerLayer` (6 파일 import) — forward path 핵심
   - `models/transformer::TransformerModel` (swap_executor 1 import) — forward 진입점
3. **WeightSwapHandler 잔여 marker 2건**:
   - LayerSlot/SecondaryMmap (line 25)
   - SwapExecutor (line 23, 또는 marker line numbering 재확인)

→ design doc 위임 prompt 초안은 Sprint B-fixup 완료 후 사용자 컨펌 받고 작성.

---

## Landmines / 미해결 / 안 가본 길

- **rust-analyzer stale cache**: Sprint B 코드 이전 후 `unresolved import crate::models::config` diagnostics가 다수 표시되나 실제 cargo check는 PASS. rust-analyzer가 자동 갱신될 때까지 무시 OK.
- **layer_lint baseline 미증가 정책**: Sprint B 부작용으로 위반이 6→8건 증가. baseline JSON(`engine/tests/spec/inv_layer_baseline.json`)을 갱신하지 말 것 — Sprint B-fixup으로 6건으로 복귀시키는 게 정공법.
- **ModelConfig private field**: `from_gguf_metadata`를 free fn으로 옮기면 ModelConfig 내부 field 접근 권한 이슈 가능. visibility 정정 필요 시 builder 패턴 또는 `pub(crate)` 최소 노출.
- **Sprint A 잔여 marker 3건은 fixup 대상이 아님**: PreloadResult/preload_erased/PrefetchController는 backlog 트랙. Sprint B-fixup 시 같이 처리하지 말 것 (scope 폭증).
- **시도했지만 실패한 방향 — 옵션 A (`shared/` 신설)**: 직전 grill에서 폐기. layer_lint이 디렉토리 이름을 모름 + 실용 가치 0. 옵션 C(현실 정합) 채택.
- **이 길은 가지 말 것 — LAYER-EXEMPT marker로 escape**: 위계 우선순위 #1/#2/#3 모순. fixup은 trait 격상/함수 이전/L2 promotion 중 하나로 처리.
- **결정 대기 (Sprint C)**: design doc 위임 vs ad-hoc — 사용자 결정 #2 명시. Sprint B-fixup 완료 후 다시 제시.

---

## 참고 링크

- 상위 plan / 직전 handoff: `.agent/todos/handoff_precision_swap_o1_entry_2026_05_26.md` (compact 전 단계)
- 위계 명시 spec: `spec/01-architecture.md` INV-LAYER-003 NOTE (line 600), `arch/01-architecture.md §6.3/§6.4` 역할 paragraph
- L2 운영 모델: `arch/01-architecture.md §6.2` "L2 위치 정책 (2026-05-26 정정)" paragraph (line 376~378)
- §13.8-G register: `ARCHITECTURE.md:2000` ModelConfig CANDIDATE 행
- layer_lint 도구: `scripts/layer_lint.py:128` TOP_LEVEL_L2 set + line 46~88 LAYER_RULES
- 관련 backlog: `.agent/todos/backlog.md` `[P2] §13.8-O cross-L3 vocabulary trait inversion` (PreloadAccess 트랙은 sprint A에서 부분 해소)
- 관련 커밋:
  - `ab9e1f35` 위계 명시
  - `a9dcb5be` Sprint A (KVCacheOps/PreloadAccess trait 격상)
  - `6dcba548` Sprint B 코드 (ModelConfig L2 격상)
  - `cea6c594` Sprint B arch (shared/ 폐기)
