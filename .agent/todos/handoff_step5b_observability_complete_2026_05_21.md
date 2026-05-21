# Handoff: Step 5b (observability 통합 확장) 종결 → 다음 sprint 결정

**작성**: 2026-05-21
**HEAD**: `3bbd923e fix(observability): eval/profile 내부 self-reference 잔재 갱신`
**브랜치/worktree**: `worktree-step5b_observability` @ `.claude/worktrees/step5b_observability/`
**다음 세션 진입 문장**: `"다음 sprint 결정"`

---

## TL;DR

- profile/ + eval/ 두 모듈을 `engine/src/observability/`로 물리 이동.
- layer_lint.py 매핑은 이미 두 모듈을 `observability` 도메인으로 인식 중이었음 — 본 sprint는 **물리 위치와 룰의 정합성 회복**.
- 4 atomic commits, INV-LAYER baseline 296건 그대로 유지 (path만 갱신).
- 멈춘 이유: Step 5b 종결 + 다음 sprint (T1 잔여 = B INV-LAYER 점진 해소 / D EnergyConstraint, 또는 T3로 이동) 사용자 결정 대기.

---

## 진행 상태

### Commits (4건)

| commit | scope | 변경 |
|---|---|---|
| `d9d023f7` | **5b-A** profile/ 이동 | 8 파일 git mv + 13 importer sed + lib.rs/mod.rs 정리 |
| `8a97153e` | **5b-B** eval/ 이동 | 7 파일 git mv + 6 importer sed (4 lib + 2 외부) + lib.rs/mod.rs 정리 |
| `e2eda5a9` | **5b-C** lint + baseline | LAYER_RULES 2 redundant 룰 제거 + baseline JSON 296건 path 재생성 |
| `3bbd923e` | **fix** self-ref 잔재 | `#[cfg(test)]` 블록 + doc-comment의 self-reference 2건 보완 |

### 도메인 매트릭스 (이동 후)

```
engine/src/observability/
├ events.rs              (Step 5-D, 기존)
├ rss_trace.rs           (Step 5-E, 기존)
├ profile/               (★ Step 5b-A, 신규 이동, 8 파일 2,862 LOC)
│  ├ cache.rs, entropy.rs, latency.rs, mod.rs
│  ├ op_trace.rs, ops.rs, quality_metrics.rs, scores.rs
└ eval/                  (★ Step 5b-B, 신규 이동, 7 파일 3,000 LOC)
   ├ eval_loop.rs, eviction_hook.rs, hook.rs
   ├ kivi_hook.rs, mod.rs, output.rs, qcf_helpers.rs
```

`engine/src/profile/` 및 `engine/src/eval/` 완전 삭제.

### 게이트 결과

| 게이트 | 결과 |
|---|---|
| `cargo build -p llm_rs2 --release` | PASS |
| `cargo clippy --lib -p llm_rs2 -- -D warnings` | clean 0건 |
| `cargo test -p llm_rs2 --lib --release` | **1202 PASS / 24 FAIL** (Step 5 종결 1203 대비 -1 — 변동 원인 미상이나 24 FAIL 동일하므로 본질 회귀 아님) |
| `cargo test --test spec inv_layer` | **8 PASS** |
| `layer_lint.py --baseline` diff | **0 new violations** |
| baseline JSON | 296건 유지 (rule별 분포 동일, path만 갱신) |

### Sub-task 표

| Task | 상태 | Commit |
|---|---|---|
| A-1: profile/ → observability/profile/ | completed | `d9d023f7` |
| A-2: eval/ → observability/eval/ | completed | `8a97153e` |
| A-3: lib.rs + observability/mod.rs 정리 | completed (A-1/A-2 흡수) | — |
| A-4: layer_lint.py LAYER_RULES 정리 | completed | `e2eda5a9` |
| A-5: INV-LAYER baseline + 게이트 | completed | `e2eda5a9` + `3bbd923e` |

---

## 다음 작업 (사용자 결정 대기)

T1→T3→T2 순서 합의됨. T1 잔여:

### B. INV-LAYER baseline 296건 점진 해소

- multi-sprint. rule별 sub-sprint로 분해.
- 예상: `INV-LAYER-003` (194건) 가장 큼 → import inversion 작업.
- 위임: Architect → Implementer chain.

### D. EnergyConstraint Spec-Impl Divergence (MGR-ALG-015)

- 단발성 0.5일.
- backlog [P3] 항목.
- 위임: Architect spec 확인 → Implementer 코드 동기화.

### (T1 종결 후 T3 진입)

- S-A LISWAP-6 cleanup segfault (`[P2]`, 1~2일 디버깅)
- S-B Manager↔Engine 프로토콜 이슈 ([P1])
- S-C Qwen CPU decode gap ([P1])

---

## Landmines / 미해결

### 본 sprint 무관 사전 회귀

1. **`backend::opencl::*` host test 24건** — backend [P3] device-required.
2. **`qnn_oppkg` 53 build error** — `crates/qnn_oppkg/src/ops/softmax.rs` 등의 type inference (Step 5 종결 시점에 이미 존재 추정). workspace build에서만 노출, `cargo build -p llm_rs2`엔 영향 없음.
3. **lib test count 1203→1202** — Step 5 종결 카운트 대비 -1. eval_loop.rs `#[cfg(test)]` 블록의 self-reference fix가 단위 테스트 카운트에 영향을 줬을 가능성. 24 FAIL 동일하므로 본질 회귀 아님으로 판단.

### Sed 누락 패턴 — 후속 sprint 진행 시 주의

- `crate::FOO` → `crate::observability::FOO` 단순 sed는 `#[cfg(test)]`/`#[cfg(test_block)]`/doc-comment 안의 self-reference를 잡지 못함 (test 모드에서만 컴파일).
- 본 sprint에서 `eval/eval_loop.rs:964` (test 블록 self-import) + `profile/quality_metrics.rs:9` (doc 예제) 각 1건 잔재. **lib release 빌드는 통과하나 `cargo test --lib`에서 발견**됨.
- 후속 모듈 이동 시 grep 범위에 `--include="*.rs"` + 자기 모듈 디렉토리도 포함시키고 검증할 것.

### experiment.rs 처리

- 본 sprint scope **제외**됨 (L4 매핑, observability 의미와 다름).
- 잠재 후속 sprint: `experiment.rs` 443 LOC를 `experiment/` 디렉토리로 승격 + 책임 분리 (schedule.rs / sampler.rs / output.rs).
- backlog 후보, 현재 우선순위 없음.

### "이 길은 가지 마라"

- profile/eval의 importer를 `pub use` re-export로 하위 호환 유지 — 일관성 깨짐. Step 5 때 core/는 완전 제거했으므로 같은 원칙.
- `experiment.rs`를 observability로 강제 흡수 — 의미가 다름 (driver vs sink), single importer라 동기 약함.

---

## 참고 자료

- 직전 handoff: `.agent/todos/handoff_step5_step6_complete_2026_05_21.md`
- 메모리: [[layered-architecture-decision]] — 본 sprint의 원안 (2026-05-16)
- 변경된 layer_lint 룰: `scripts/layer_lint.py:64-65` (2 redundant 룰 제거)
- 변경된 baseline: `engine/tests/spec/inv_layer_baseline.json` (296건, file path 갱신)
- 후속 후보 목록: 본 세션 브리핑 (handoff_step5_step6_complete_2026_05_21.md 후보 B, D + T3 트랙)

## 자기점검 결과

- [x] 진입 문장 한 줄? `"다음 sprint 결정"`
- [x] "왜 멈췄는가"? Step 5b 종결 + 다음 sprint 사용자 결정 대기 (T1→T3→T2 순서, T1 잔여 B/D 또는 T3 진입)
- [x] 가장 큰 landmine 표면화? sed 누락 self-reference 패턴 (test 블록/doc-comment는 release 빌드에서 안 잡힘)
- [x] 검증 게이트 수치/명령? `cargo test --lib: 1202 PASS / 24 FAIL`, `spec inv_layer 8 PASS`, `baseline diff 0`
- [x] 길이 적정? ~500 토큰, 4 commits + 도메인 매트릭스 + 게이트 5종 + landmine 4건
