# Handoff — Sprint 1 W-AUF-2 종결 (2026-05-20)

## Sprint 1 W-AUF-2 — 호스트 완료 ✅, 디바이스 게이트 차단 (B-3a 의존)

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

## 디바이스 게이트 (G7/G8 실행 시도 — 차단 발견, backlog로 이관)

### 실행 결과 (2026-05-20 S25)

3-way control 측정 (qnn_oppkg backend, 6T):

| 구성 | Decode (ms/tok) | Avg TBT (ms) | 출력 정확성 |
|---|---|---|---|
| GGUF primary (qwen2.5-1.5b-q4_0.gguf) | 3.51 | 29.24 | ✅ "I am a high school senior..." |
| AUF primary (qwen2.5-1.5b-q4_0.auf, ADRENO_SOA) | 7.55 | 25.63 | ❌ garbage ("Parameter(Parameter(..." 반복) |
| AUF primary (multi-dtype CPU_AOS F16) + self-secondary 자동 활성 | 3.84 | 50.04 | ❌ garbage (다른 패턴) |

### 정상 동작 항목 ✅

- **AUF self-secondary 자동 활성** — `[AUF self-secondary] auto-activated:
  variant=WEIGHTS_CPU_AOS primary_dtype=F16 secondary_dtype=Q4_0 layers=28` 라인 정상 출력
- **ReverseSwapRejected 게이트** — `--primary-dtype q4_0` device에서 정확히 reject
  ("primary=Q4_0, secondary=F16. Weight swap only supports F16→Q4_0")
- **qnn_oppkg graphFinalize** — 28 layer 모두 PASS (총 1158 ms)
- **norm/cross-layer dtype lenient lookup** — 본 sprint에서 추가 fix
  (`is_dtype_strict_kind`, commit `69fd5991`) 후 multi-dtype AUF에서 AttnNorm F32 lookup 정상

### 차단 원인 (G7/G8 측정 불가)

**AUF primary + qnn_oppkg device forward 정확성 자체가 W-AUF-1 시점부터 미해결**.
baseline 자체가 garbage라 swap latency 비교가 무의미.

진단:
- GGUF primary는 정상 → 차단 위치는 `AufSource::materialise_cpu_tensor` 또는
  `qnn_oppkg + AUF view-buffer` 경계의 weight layout 변환 불일치 의심
- multi-dtype AUF는 CPU_AOS variant만 빌드 (W-AUF-1B `--variants cpu_aos`).
  CPU_AOS variant도 backend가 받기는 하지만 forward 정확성 위반은 다른 원인
- ADRENO_SOA variant AUF (q4_0.auf, no swap)도 동일하게 garbage → variant 무관, AUF path 자체 문제

### Backlog 신설 / 갱신

| ID | 항목 | 우선순위 |
|---|---|---|
| **B-3a** (신설) | AUF primary qnn_oppkg device forward 정확성 회복. GGUF↔AUF byte-identical 검증 (W-AUF-1 device path 미검증 분 회수). | P0 (G7 차단) |
| **B-3b** (신설) | B-3a 해결 후 multi-dtype AUF (with ADRENO_SOA + CPU_AOS variant) 재빌드 + W-AUF-2 G7/G8 게이트 측정. | P1 |
| B-4 | Llama 3.2-3B AUF multi-dtype 빌드 (검증 확장) | P2 |
| B-5 | swap_executor의 AUF self-secondary 정합성 검증 (B-3a 의존) | P2 |

### 검증 인프라 (✅ 그대로 유효)

- 호스트: `models/qwen2.5-1.5b/qwen2.5-1.5b-multi-dtype.auf` (4.24 GiB, multi-dtype Q4_0+F16)
- 디바이스 push 완료: `/data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-multi-dtype.auf`
- `auf_tool verify` 19/19 PASS

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

## 미완료 항목 (backlog)

| ID | 항목 | 우선순위 |
|---|---|---|
| **B-1** | `SecondaryMmap` enum → trait 통일 (R8.2.3.7, OCP 잔여) | P2 |
| **B-2** | AUF self-secondary `prefault_layers` 정합성 (현재 동작 검증 필요) | P2 |
| **B-3a** | AUF primary qnn_oppkg device forward 정확성 회복 (W-AUF-1 device path 잔여) | **P0** |
| **B-3b** | B-3a 해결 후 W-AUF-2 G7/G8 디바이스 게이트 측정 | P1 |
| **B-4** | Llama 3.2-3B AUF multi-dtype 빌드 | P2 |
| **B-5** | swap_executor의 AUF self-secondary 정합성 (B-3a 의존) | P2 |

## 다음 세션 진입

**"W-AUF-1 device 정확성 회복 진행"** (B-3a) — AUF primary qnn_oppkg path가
GGUF↔AUF byte-identical을 만족하도록 forward path 수정. 그 후 B-3b로 W-AUF-2
G7/G8 측정 가능.

또는 Sprint 2 (`weights/` 디렉토리 리뷰)로 직행. 본 sprint 호스트 게이트는
모두 충족 상태이므로 ff-merge에 차단 없음.

## Plan 파일

`/home/go/.claude/plans/proud-strolling-whale.md` — Sprint 1 (W-AUF-0~2) 본격
모두 완료. Sprint 2 plan은 별도 작성.

## Worktree

`/home/go/Workspace/llm_rs2/.claude/worktrees/sprint1_auf_loader` (브랜치
`worktree-sprint1_auf_loader`). 본 sprint 종결 후 master에 ff-merge 가능
(C1~C6 모두 호스트 게이트 PASS). 다음 sprint는 별도 worktree 권장.
