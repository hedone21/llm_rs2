# Handoff — Sprint 1 W-AUF-2 종결 (2026-05-20)

## Sprint 1 W-AUF-2 — 완료 ✅ (호스트), 디바이스 게이트 측정 대기

`AUF self-secondary 자동 활성` sprint. 6 commits로 C1~C6 종결. R8 (Eager-
Flattened Adapter 패턴 — `WeightSectionView` trait) 부분 적용 완료.

| Commit | 내용 |
|---|---|
| `b71fc340` | C1 — `TensorSource::as_auf` default method + AufSource override |
| `d149604d` | C2 — `AufSecondaryMmap.view` → `Arc<AufView>` 일반화 |
| `c86bf06c` | C3 — `from_auf_self_secondary` + `resolve_secondary` 본격 |
| `db6f3494` | C4 — AUF self-secondary 통합 테스트 4건 (multi-dtype fixture) |
| `b81fdf83` | C5 — `WeightSectionView` trait + `GgufBacking`/`AufBacking` 분리 |
| `c94bb385` | C6 — `try_promote_auf_self_secondary_to_rpcmem` + resolve_secondary 분기 |

## 호스트 게이트 (PASS)

- `cargo build -p llm_rs2` PASS (lib + bins, default features)
- `cargo build -p llm_rs2 --lib` PASS (default + no-default-features)
- `cargo test -p llm_rs2 --lib --no-default-features` **1159 PASS, 회귀 0**
- 신규 unit test: 9건 (C1 0 + C2 0 + C3 2 + C4 4 + C5 0 + C6 0, 그 외는 기존 회귀)
  - `resolve_secondary_non_auf_returns_none`, `resolve_secondary_rejects_explicit_for_auf_primary`
  - `from_auf_self_secondary_selects_other_dtype`, `from_auf_self_secondary_errors_when_only_primary_dtype`
  - `from_auf_self_secondary_rejects_adreno_soa_with_only_f16`
  - `from_auf_self_secondary_rejects_reverse_swap_q4_to_f16`

## 디바이스 게이트 (미실행 — 별도 세션)

G7, G8 측정은 본 commit chain에서는 진행하지 않음. 다음 세션 진입 시:

```bash
# G7: S25 qnn_oppkg, AUF primary multi-dtype, self-secondary 자동 활성
python scripts/run_device.py -d s25 generate -- \
  --model-path /data/local/tmp/qwen2.5-1.5b-multi-dtype.auf \
  --backend qnn_oppkg --gen 64 --enable-resilience

# 기준: Phase 6.5 baseline TBT 대비 ≤ 5% 변동
# 검증 포인트:
# - stderr에서 "[AUF self-secondary] auto-activated" 라인 출력
# - swap_weights 액션이 heartbeat에 등록되는지
# - swap latency 회귀 없는지 (RpcMem alias 경로 활성)
```

검증 인프라:
- `models/qwen2.5-1.5b/qwen2.5-1.5b-multi-dtype.auf` (4.24 GiB, multi-dtype Q4_0+F16)
- `auf_tool verify` 19/19 PASS (W-AUF-1B에서 확보)

## 본 sprint 종결 코드 변경 요약

| 파일 | LOC 변동 |
|---|---|
| `engine/src/models/loader/mod.rs` | +84 / -19 |
| `engine/src/models/loader/auf/mod.rs` | +1 / -1 |
| `engine/src/models/loader/auf/source.rs` | +20 / -0 |
| `engine/src/models/loader/auf/secondary.rs` | +501 / -1 |
| `engine/src/models/transformer.rs` | +73 / -22 |
| `engine/src/models/weights/backing.rs` (NEW) | +85 |
| `engine/src/models/weights/mod.rs` | +2 / -0 |
| `engine/src/models/weights/secondary_mmap.rs` | +69 / -6 |
| `engine/src/models/weights/rpcmem_secondary.rs` | +56 / -8 |

순증 약 +850 LOC (테스트 +331 포함, 순수 production +519).

## 정책 안내

### CLI 동작

- **AUF primary**: `--model-path foo.auf` (정식)
- **AUF multi-dtype**: `auf_tool build --dtypes q4_0,f16 ...`로 빌드 시 self-secondary 자동 활성
- **--no-self-secondary**: 의도적 비활성 (디버그/벤치마크)
- **AUF + --secondary-gguf**: 명시 거부 (R10) — AUF는 self-secondary가 정식 경로
- **GGUF + --secondary-gguf**: 기존 deprecated alias 그대로 동작 (W-AUF-1 정책 유지)

### 아키텍처

- **R6 위험 해결**: primary `AufSource`와 self-secondary가 같은 `Arc<AufView>` 공유.
  mmap 1회 보장.
- **R8 적용 (부분)**: `WeightSectionView` trait + `GgufBacking`/`AufBacking`.
  `RpcmemSecondaryStore`는 trait abstraction에 의존 (LSP/DIP 만족).
  **남는 OCP 위배** (`SecondaryMmap` enum의 닫힌 variant 집합)는
  `.agent/todos/backlog.md`로 이관 권장 (`SecondaryMmap` trait화).

### 단방향 swap 정합성

`from_auf_self_secondary`가 같은 게이트를 재사용:
- AdrenoSoa + F16 secondary → `LoadError::AdrenoSoaF16Rejected`
- primary=Q4_0 + secondary=F16 → `LoadError::ReverseSwapRejected`
- primary와 같은 dtype만 가용 → `LoadError::AufInvariantViolation`

## 미완료 항목 (backlog 등록 권장)

| ID | 항목 |
|---|---|
| **B-1** | `SecondaryMmap` enum → trait 통일 (R8.2.3.7, OCP 잔여) |
| **B-2** | AUF self-secondary `prefault_layers` 정합성 (현재 동작 검증 필요) |
| **B-3** | S25 device 게이트 측정 (G7/G8, Phase 6.5 baseline ≤5%) |
| **B-4** | Llama 3.2-3B에 대한 AUF multi-dtype 빌드 (검증 인프라 확장) |
| **B-5** | swap_executor가 `is_pre_converted_soa=false`인 AUF self-secondary와의 정합성 (현재 동작은 GGUF와 동일 경로) |

## 다음 세션 진입

**"W-AUF-2 디바이스 측정 진행"** — S25 게이트 G7/G8 실행 + 결과 handoff 갱신.
또는 Sprint 2 (`weights/` 디렉토리 리뷰)로 직행.

## Plan 파일

`/home/go/.claude/plans/proud-strolling-whale.md` — Sprint 1 (W-AUF-0~2) 본격
모두 완료. Sprint 2 plan은 별도 작성.

## Worktree

`/home/go/Workspace/llm_rs2/.claude/worktrees/sprint1_auf_loader` (브랜치
`worktree-sprint1_auf_loader`). 본 sprint 종결 후 master에 ff-merge 가능
(C1~C6 모두 호스트 게이트 PASS). 다음 sprint는 별도 worktree 권장.
