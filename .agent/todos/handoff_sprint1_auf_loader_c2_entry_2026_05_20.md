# Handoff — Sprint 1 W-AUF-1 C2 진입 (2026-05-20)

## 진입 문장

**"W-AUF-1 C2 진행"** — context-warm 상태로 재개. 본 handoff + plan 파일
(`/home/go/.claude/plans/proud-strolling-whale.md`) 두 개만 읽으면 컨텍스트 복원.

## 현재 상태

- **Worktree**: `.claude/worktrees/sprint1_auf_loader` (브랜치 `worktree-sprint1_auf_loader`)
- **HEAD**: `6adfefbf feat(loader): AufSource + primary format detection (W-AUF-1 C1)`
- **Base**: origin/master (step4_d_pressure_toplevel과 별개 line of work)

## Sprint 1 전체 진행도

| Commit | 상태 | 산출물 |
|---|---|---|
| **C1** | ✅ `6adfefbf` | `loader/auf/{mod,source,variant_select}.rs` + `LoadConfig` 확장 |
| **C2** | ⏳ 다음 | secondary_mmap AUF 코드 이동 + zero-copy buffer + load_tensor 본격 구현 |
| **C3** | pending | `generate.rs:1791` 3-way dispatch + self-secondary stub |
| **C4** | pending | `--primary-variant`/`--primary-dtype`/`--no-self-secondary` flag + `--secondary-gguf` deprecation |
| **C5** | pending | `auf_tool build` TOKENIZER eos/bos 슬롯 + CLI `--eos-token-id` fallback |
| **C6** | pending | 문서 동기화 (USAGE.md 15회, spec/41-invariants 등 12 파일) |

W-AUF-2 (self-secondary 자동 활성, Eager-Flattened Adapter 패턴) + W-AUF-1B
(auf_tool multi-dtype mode) + Sprint 2 (weights/ 리뷰)는 본 sprint 후 진행.

## C2 작업 세부

### 목표

`engine/src/models/weights/secondary_mmap.rs`(1873 LOC)에서 AUF 전용 코드를
`loader/auf/secondary.rs`로 이동하고, primary `AufSource`의 `load_tensor`를
zero-copy로 본격 구현한다.

### 이동 대상 (secondary_mmap.rs → loader/auf/secondary.rs, ~700 LOC)

| 항목 | 현재 위치 |
|---|---|
| `auf_dtype_to_engine` | secondary_mmap.rs:304 |
| `open_secondary_auf` | secondary_mmap.rs:926 |
| `build_auf_secondary_from_view` | secondary_mmap.rs:1037 |
| `check_auf_metadata` | secondary_mmap.rs:1297 |
| `is_auf_path` | secondary_mmap.rs:787 부근 |
| `detect_backend_tag` | secondary_mmap.rs:880 |
| `resolve_backend_tag_candidates` | secondary_mmap.rs:895 |

호출처 (re-export로 변경 0 보장):
- `secondary_mmap.rs::open_secondary_with_options` (line 662) — AUF 분기 (line 669) 그대로 유지
- `secondary_mmap.rs::open_secondary_with_backend` (line 683) — AUF 분기 그대로
- `models/transformer.rs::load_from_config` (line 219) — 호출 시그니처 무변경

### 신설: zero-copy buffer for primary

`engine/src/buffer/auf_view_buffer.rs` (또는 기존 `BorrowedMmapBuffer` 확장):

```rust
pub struct AufViewBuffer {
    view: Arc<crate::auf::AufView>,
    /// File-absolute offset (weights_section_offset + variant_offset).
    abs_offset: usize,
    len: usize,
    dtype: DType,
}

impl Buffer for AufViewBuffer {
    fn as_ptr(&self) -> *const u8 {
        unsafe { self.view.raw_bytes().as_ptr().add(self.abs_offset) }
    }
    // ...
}
```

대안: `MmapBuffer` 패턴 (`Arc<Mmap>` 직접 보유)을 따라 `AufView`에서 `Arc<Mmap>`을
노출하는 accessor 추가 → 기존 `MmapBuffer::new` 재활용. **AufView가 `_mmap`을
private으로 보유 중**이라 accessor 추가 필요 (`shared/src/auf/reader.rs`).

### `AufSource::load_tensor` 본격 구현 (source.rs `todo!()` 교체)

알고리즘:
1. `tensor_id_to_auf(id)` → `(layer_idx, kind)`
2. `view.lookup_tensor(layer_idx, kind.as_u32(), self.primary_dtype)` → `&TensorEntry`
3. `variant_idx = view.tensor_index.variant_index_for_tag(self.variant_tag.as_str())`
4. `var_offset = entry.variant_offsets[variant_idx]`, `var_size = entry.variant_sizes[variant_idx]`
5. AUF는 outermost-first shape → `Shape::new(entry.shape.iter().map(|&d| d as usize).collect())`
   (GGUF처럼 `.rev()` 적용하지 **않음**)
6. `auf_dtype_to_engine(entry.dtype)` → `DType`
7. `AufViewBuffer::new(view.clone(), weights_section_offset + var_offset, var_size, dtype)` → CPU tensor
8. GgufSource 패턴 따라 `is_cpu`/`is_weight` 분기로 backend 업로드

참고: `secondary_mmap.rs::build_auf_secondary_from_view`의 dtype 선택 + variant
검증 로직 거의 그대로 재활용. 단, primary는 single dtype 선택 (multi-dtype filter
필요 없음 — `lookup_tensor`가 처리).

### Qwen2 bias 처리 결정 필요

AUF v0.1 `TensorKind`에 bias variant 없음. 현재 `tensor_id_to_auf`는
`LayerBias`를 `None` 반환. Qwen2 모델은 `has_qkv_bias=true`라 `load_model`이
bias를 로드 시도 → 에러 가능성.

옵션:
- (A) AUF v0.1.x에서 Qwen2 bias 미지원 — Llama만 primary로 지원
- (B) GGUF로 fallback (AUF + GGUF hybrid — 본 sprint 정책 위배)
- (C) AUF v0.2에서 `TensorKind::AttnQBias` 등 추가 (W-AUF-1B + auf_tool 재빌드 필요)

**현재 가용 AUF 파일은 Qwen 2.5-1.5B/7B (둘 다 qkv_bias 필요)** — A로 가면
primary AUF 검증 불가. C가 정답이지만 본 sprint는 A로 일단 시작하고 검증 시
Llama 3.2 AUF 사용 (구할 수 있다면) 또는 C2 게이트에서 명시적 결정.

## C2 게이트

| # | 기준 | 측정 |
|---|---|---|
| G1 | `cargo build -p llm_rs2 --lib` | exit 0 |
| G2 | `cargo build -p llm_rs2` (모든 bin) | exit 0 |
| G3 | `cargo test -p llm_rs2 --lib auf` | 회귀 0 + 신규 test 추가 |
| G4 | 기존 secondary_mmap 호출처 동작 변화 0 | re-export로 path 안 변경 |
| G5 | `AufSource::load_tensor` unit test 추가 | 신규 PASS |

## 관련 파일 위치 (master HEAD 기준)

본 worktree의 master HEAD는 step 4-D worktree와 다른 구조:
- AUF 포맷 코드: `engine/src/auf/` (이전 `shared/src/auf/`)
- AUF dtype_convert: `engine/src/auf/dtype_convert.rs` (이미 이동됨)
- CLI 진입: `engine/src/bin/generate.rs` (session/init.rs 없음)
- core 모듈: `engine/src/core/` (cache_manager/eviction/kv_cache/pressure/qcf 다 흡수)

## R8 (재확정)

AUF self-secondary 도입 시 RpcMem alias path 비호환. W-AUF-2.3에서
**Eager-Flattened Adapter** 패턴 (`SecondaryWeightsBacking` trait +
`GgufBacking`/`AufBacking` impl)으로 해결. SOLID 5원칙 모두 만족; `SecondaryMmap`
enum의 OCP 위배는 backlog.

상세: plan 파일 (`/home/go/.claude/plans/proud-strolling-whale.md`) 끝부분 §"R8 확정 +
W-AUF-2.3 융합안 (2026-05-20)".

## 메모

- 자동 commit + `notify-send "llm.rs" "<요약>"`는 매 commit마다 진행.
- `.cl` 커널 수정 없음 (본 sprint 범위 외).
- TBT metric은 avg_tbt (tok0 inclusive).
- 성능 측정 `--profile` 없이.
