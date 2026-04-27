# Layer-Swap QCF↔NLL Phase 4 Sanity Report

작성일: 2026-04-27
HEAD: `d460a77 feat(qcf): add --qcf-dump CLI + warmup→swap→PPL workflow for layer-swap measurement`

## 코드 Sanity (호스트, default features)

| 검증 | 결과 | 비고 |
|------|------|------|
| `cargo fmt --all` | PASS | 변경 없음 |
| `cargo clippy --workspace --all-targets -- -D warnings` | PASS | warning 0 |
| `cargo check --workspace` | PASS | 빌드 OK |
| 신규 spec test (`test_qcf_swap_dump`) | **7/7 PASS** | dump schema, ratio sweep, NaN ε 제외, generation 모드 null 직렬화 모두 검증 |
| spec 통합 test (전체) | **474/474 PASS** | 회귀 없음 |
| `cargo test --workspace` lib tests | ⚠️ SIGABRT | `unified_buffer::tests::test_map_write_unmap_cycle` — **본 변경 무관**한 baseline 환경 이슈 (NVIDIA OpenCL host limit, 메모리에 기록됨) |

## 모델 자산 가용성

호스트(`/home/go/`)에서 발견된 자산:
- `/home/go/auf_e/llama-3.2-1b-mixed.auf` ✅ (이전 v0.2 multi-quant validation 자산)
- `/home/go/auf_e/mixed_all_2.auf` ✅
- Primary GGUF (F16/Q4_0): **부재**
- Llama 3.1 8B, Qwen2.5 1.5B/7B: **부재**

## 실모델 Smoke Test 보류 사유

다음 두 의존성 때문에 호스트에서 실제 측정 smoke을 보류:

1. **Primary GGUF 부재** — `--model-path <f16.gguf>`가 없어 swap 워크플로우 진입 불가.
2. **CUDA 빌드 환경 미검증** — `--features cuda`는 nvcc/CUDA toolkit 필요. 호스트에서 build 가능 여부 미확인.

이 두 의존성은 모두 외부 harness 측 (`/home/go/Workspace/papers/pact2026/experiments/scripts/`)에서 자체 처리되므로, **실모델 smoke은 외부 harness 인계 후 첫 sweep run에서 자연스럽게 검증된다**.

외부 harness가 실행 시 첫 dump JSON에서 다음 sanity 체크 권장:

| 필드 | 기대값 (1B, ratio=0.33 기준) |
|------|-----------------------------|
| `schema_version` | 1 |
| `model_arch` | `"llama"` 또는 `"qwen2"` |
| `num_layers` | 16 (1B), 32 (8B), 28 (1.5B/7B) |
| `swap_count` | floor(ratio × num_layers). 1B r=0.33 → 5 |
| `qcf_swap_predicted` | ∈ [0, 1] |
| `fallback_used` | **`false`** (importance + noise 양쪽 주입됨) |
| `importance_table.length` | 보통 num_layers (Full sub-layer) |
| `noise_table.length` | num_layers - (NaN ε 제외) |
| `ppl` (`--ppl` 모드) | baseline 대비 ratio↑ 시 단조 증가 |

## 인계 (Phase 4 종결)

- 가이드 문서: `docs/layer_swap_qcf_measurement.md` (Phase 3 산출물)
- CLI 인터페이스: `--qcf-dump <PATH>`, `--qcf-warmup-tokens <N>`
- JSON 스키마: schema_version=1
- Helper: `engine/src/eval/qcf_helpers.rs::dump_qcf_swap_json`
- 외부 harness 위치: `/home/go/Workspace/papers/pact2026/experiments/scripts/`

다음 작업 (외부 harness agent 또는 사용자):
1. Phase 2 모델 자산 준비 (Llama 8B + Qwen 2종 GGUF/AUF 변환).
2. CUDA 빌드 (`cargo build --release --features cuda -p llm_rs2 --bin generate`).
3. Sweep 실행 (4 model × 4 ratio × 3 bench = 48 run).
4. 결과 분석 (Spearman ρ, Pearson r, 산점도).
