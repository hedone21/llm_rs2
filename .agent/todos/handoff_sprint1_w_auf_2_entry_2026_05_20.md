# Handoff — Sprint 1 W-AUF-2 진입 (2026-05-20)

## 진입 문장

**"W-AUF-2 진행"** — context-warm 상태로 재개. 본 handoff + plan 파일
(`/home/go/.claude/plans/proud-strolling-whale.md` § "W-AUF-2") + 종결 handoff
(`.agent/todos/handoff_sprint1_auf_loader_complete_2026_05_20.md`) 세 개만 읽으면 복원.

## 현 상태

- **Worktree**: `.claude/worktrees/sprint1_auf_loader` (브랜치 `worktree-sprint1_auf_loader`)
- **HEAD**: `febfa031 docs(handoff): W-AUF-1B 완료 기록 — auf_tool multi-dtype 인프라 실측 검증`
- **Base**: origin/master (step4_d_pressure_toplevel과 별개 line of work)
- **Sprint 1 W-AUF-1**: 7 commits로 C1~C6 종결 (정식 entry `--model-path foo.auf`, CLI flags, 문서 동기화 완료)
- **Sprint 1 W-AUF-1B**: 완료 ✅ — multi-dtype 인프라가 이미 master에 있었음 + Qwen2.5-1.5B 실측 빌드 PASS

## W-AUF-2 검증 인프라 (✅ 확보)

```
models/qwen2.5-1.5b/qwen2.5-1.5b-multi-dtype.auf  (4.24 GiB)
```
- format v0.2.1, capability_opt 0x0c (LM_HEAD_PRECOMPUTED_Q4_0 + MULTI_DTYPE_VARIANTS)
- 453 dtype entries (q4_0 + f16 multi-dtype), META.default_dtype=Q4_0, TOKENIZER eos=151643
- `auf_tool verify` 19/19 PASS (INV-137/138 포함)

본 파일이 W-AUF-2 self-secondary 자동 활성 통합 테스트의 1순위 fixture.

## W-AUF-2 작업 범위

### W-AUF-2.1 — `SecondaryMmap::from_auf_self_secondary`

```rust
impl SecondaryMmap {
    pub fn from_auf_self_secondary(
        view: Arc<AufView>,
        primary_variant_tag: BackendTag,
        primary_dtype: TensorDType,
        config: &ModelConfig,
    ) -> Result<Arc<Self>>;
}
```

- primary가 AUF일 때 **같은 `Arc<AufView>`를 secondary로 재포장** — mmap 1회 보장 (R6 위험 완화).
- `primary_variant_tag` / `primary_dtype`은 swap 후보 entry에서 제외.
- 위치: `engine/src/models/loader/auf/secondary.rs` (W-AUF-1 C2에서 이미 분리한 모듈).
- 호출 시 `AufSource::view_arc()`가 이미 노출되어 있음 (`source.rs:92`).

### W-AUF-2.2 — `resolve_secondary` stub 본격 구현

현재 stub: `engine/src/models/loader/mod.rs::resolve_secondary` (W-AUF-1 C3에서 None만 반환).

```rust
pub fn resolve_secondary(
    cfg: &LoadConfig,
    source: &dyn TensorSource,
    backend: &Arc<dyn Backend>,
) -> Result<Option<Arc<SecondaryMmap>>> {
    // 1. explicit --secondary-gguf 우선 (deprecated alias로 유지).
    if let Some(path) = &cfg.secondary_source {
        return Some(open_secondary_with_backend(path, ..., backend)?);
    }
    // 2. AUF primary + multi-dtype/variant + !disable_self_secondary → 자동 활성.
    if let Some(auf) = source.as_auf()
        && cfg.primary_format == PrimaryFormat::Auf
        && !cfg.disable_self_secondary
        && auf.has_swap_candidate()
    {
        return Some(SecondaryMmap::from_auf_self_secondary(
            auf.view_arc(),
            auf.primary_variant_tag(),
            ..,
            source.config(),
        )?);
    }
    Ok(None)
}
```

문제: `dyn TensorSource`에서 `AufSource`로 downcast 필요. **`TensorSource` trait에 `as_auf(&self) -> Option<&AufSource>` default method 추가** 또는 `Any` 사용.

추천: `as_auf()` default method (다른 source는 None 반환). LSP 안전 + Box<dyn Trait>에서 정상 동작.

### W-AUF-2.3 — Eager-Flattened Adapter 패턴 (R8 대응)

**문제**: 현재 `secondary_mmap.rs:702`이 `is_auf_path(path) || !backend_supports_rpcmem_secondary(backend)` → AUF는 standard mmap path로 강제 분기. AUF self-secondary 도입 시 LISWAP-6 H2D 0-copy 가속(rpcmem alias path)을 잃음 (S25 qnn_oppkg에서 swap 22ms 회귀 가능).

**해결**: `SecondaryWeightsBacking` trait + `GgufBacking` / `AufBacking` impl. 각 impl이 생성자에서 layer_index 평탄화 (LSP 안정). `RpcmemSecondaryStore`가 trait abstraction에 의존.

| Step | 산출물 | LOC |
|---|---|---|
| 2.3.1 | `SecondaryWeightsBacking` trait + `SecondaryTensorInfo` 위치 정리 (`weights/mod.rs`) | +40 |
| 2.3.2 | `GgufBacking` struct + impl. layer_index 빌드 로직을 `GgufBacking::new`로 이동 | +90 |
| 2.3.3 | `AufBacking` struct + impl. variant_idx/dtype 생성 시 고정, layer_index 평탄화 | +110 |
| 2.3.4 | `RpcmemSecondaryStore.backing: Arc<dyn SecondaryWeightsBacking>` 교체 | +50 |
| 2.3.5 | `open_secondary_with_backend` AUF 분기 — primary AUF + self-secondary + qnn_oppkg → RpcMem alias 경로 | +30 |
| 2.3.6 | S25 게이트 측정 (Phase 6.5 baseline 대비 ≤5%) | (측정) |
| 2.3.7 | Backlog: `SecondaryMmap` enum의 OCP 위배 해결 (trait화) | (문서) |

SOLID 5원칙 만족 (handoff complete §"R8 — Eager-Flattened Adapter" 상세 참조).

## 게이트

| # | 기준 | 측정 |
|---|---|---|
| G1 | `cargo build -p llm_rs2 --lib` | exit 0 |
| G2 | `cargo build -p llm_rs2` (모든 bin) | exit 0 |
| G3 | `cargo test -p llm_rs2 --lib --no-default-features` | ≥1154 PASS, 회귀 0 |
| G4 | 신규 spec test: AUF self-secondary swap 결과가 explicit GGUF secondary와 동등 | PASS (multi-dtype AUF fixture 사용) |
| G5 | `--no-self-secondary` 동작 검증 | flag set → None 반환 |
| G6 | Heartbeat가 `swap_weights` 액션 정상 등록 — `executor.set_has_secondary(true)` 트리거 확인 | PASS |
| G7 (R8) | S25 device — Phase 6.5 baseline 대비 swap latency ≤5% 변동 | 디바이스 측정 |
| G8 (R8) | S25 — top-5 overlap > 99% | PASS |

## 관련 파일 위치 (master HEAD 기준)

| 항목 | 경로 |
|---|---|
| AufSource (primary) | `engine/src/models/loader/auf/source.rs` |
| AUF secondary helpers | `engine/src/models/loader/auf/secondary.rs` |
| LoadConfig + resolve_secondary stub | `engine/src/models/loader/mod.rs` |
| TransformerModel load_auf_from_config | `engine/src/models/transformer.rs::load_auf_from_config` |
| SecondaryMmap enum | `engine/src/models/weights/secondary_mmap.rs:406` |
| RpcmemSecondaryStore | `engine/src/models/weights/rpcmem_secondary.rs` |
| AUF magic + view | `engine/src/auf/{header,reader}.rs` |

## 메모

- `Tensor`가 `Debug` 미구현이라 `.unwrap_err()` 안 됨 — match arm 으로 unwrap_err 우회 (C2에서 학습).
- LayerBias는 AufSource에서 fail-fast 에러 (`tensor_id_to_auf` None → `materialise_cpu_tensor` 명시 에러). W-AUF-2 검증은 Qwen2.5 multi-dtype 사용 시 bias 텐서 로딩 분기를 어떻게 처리할지 결정 필요 — primary는 그냥 빌드된 GGUF에서 받지만 self-secondary가 이를 처리해야 하므로.
- `Tensor::buffer()`로 Arc<dyn Buffer> 접근, `buf.size()`로 byte 길이, `buf.as_ptr()`로 raw pointer (test에서 사용).
- 자동 commit + `notify-send "llm.rs" "<요약>"`은 매 commit마다 진행.
- `.cl` 커널 수정 없음 (본 sprint 범위 외).

## Plan 파일

`/home/go/.claude/plans/proud-strolling-whale.md` § "W-AUF-2 — AUF self-secondary
자동 활성 (정식 경로 완성)" + § "R8 확정 + W-AUF-2.3 융합안 (2026-05-20)" 본격
참조.

## 재진입

새 세션에서 plan + 본 handoff + complete handoff 세 파일 읽고 **"W-AUF-2 진행"** 입력.
