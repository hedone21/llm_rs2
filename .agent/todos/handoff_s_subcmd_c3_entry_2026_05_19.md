# Handoff — S-cleanup 완료 + S-subcmd C1/C2 종결, C3-C10 진입점 (2026-05-19)

**Worktree HEAD**: `f0a21d2c` (master `9527c3f9` + 8 commits)
**선행 doc**:
- `handoff_phase4_DABC_complete_2026_05_18.md` (Phase 4-D/A/B/C 종결, 다음 sprint plan)
- `sprep_args_matrix_2026_05_18.md` (142 field 매트릭스 + 17 sub-struct 분류)
- `s_subcmd_design_2026_05_19.md` (EvictionCmd + KvMode 사양 + C1~C10 분해)

---

## TL;DR

| Sprint | Commit | 작업 | 결과 |
|---|---|---|---|
| S-prep doc | `574fc833` | 142 field 정정 + 17 sub-struct 분류 확정 + S-cleanup 진입점 | 138 field |
| S-cleanup B1 | `1eaa7c04` | dead 4 field 삭제 (no_prefill_ws + qcf placeholder 3) | 138 |
| S-cleanup B2 | `5e5d1743` | legacy/measurement 5 field 삭제 + zero_copy→no_zero_copy 패턴 전환 | 133 |
| S-cleanup doc | `11f146da` | 그룹 1-4 review 결과 반영 + dump_importance 재분류 | doc |
| S-cleanup B3 | `a283b35a` | awqe → LLMRS_KIVI_AWQE env 이전 | 132 |
| S-subcmd design | `ad658a61` | EvictionCmd + KvMode 사양 + C1~C10 분해 | doc |
| S-subcmd C1 | `887037d1` | session/cli/eviction.rs 신설 (326 LOC, 9 unit test PASS) | dead lib |
| S-subcmd C2 | `f0a21d2c` | Args 통합 + 21 field 제거 + 22 shim accessor + 175 호출처 자동 마이그레이션 | **111** |

**총 변경**: 142 → 111 production field (-22%). Eviction 도메인은 추가로 sub-args struct 4개 (Sliding/Streaming/H2o/D2o) + 7 EvictionCommonArgs로 분리.

---

## 현재 상태

### 코드 구조

```
engine/src/session/
├── cli/
│   ├── mod.rs    (1,150 LOC: Args struct 111 field + 22 shim accessor impl)
│   └── eviction.rs   (322 LOC: EvictionCmd enum + 4 SubArgs + EvictionCommonArgs + 9 test)
```

### Args 분류 (S-cleanup 결과)

`.agent/data/sprep_args_matrix/args_matrix.tsv` v2 (142 field) 기준:

| 그룹 | 상태 | 처리 |
|---|---|---|
| 1 ModelLoadArgs (4) | ✓ done | production 보존 |
| 2 BackendArgs (13→9) | ✓ done | 5 삭제 + zero_copy→no_zero_copy 패턴 |
| 3 SamplingArgs (6) | ✓ done | production 보존 |
| 4 ProfileArgs (12→10) | ✓ done | profile_per_head 보존, dump_importance→QcfArgs 재분류 |
| 5 KiviArgs (5→4) | ✓ done | awqe→env 이전 |
| 6 OffloadArgs (3) | — | production 예상 |
| 7 EvictionArgs (21→0+Cmd+Common) | ✓ done | C2에서 subcommand로 이관 |
| 8 PrefillArgs (5→4) | partial | no_prefill_ws 삭제 |
| 9 ResilienceArgs (3) | — | production |
| 10 TensorPartitionArgs (1) | — | production |
| 11 SwapArgs (24) | — | 대대적 cleanup, swap_bench binary 분리 후보 |
| 12 QcfArgs (13→9 + dump_importance) | partial | placeholder 3 삭제 |
| 13 ExperimentArgs (5) | — | bin/experiment_runner 분리 후보 |
| 14 ChatArgs (4) | — | chat flag bin 분리 시 제거 |
| 15 PplArgs (10) | — | dump_q4_* 등 측정 종결 시 삭제 |
| 16 BatchArgs (3) | — | production |
| 17 EvalLlArgs (3) | — | eval_ll flag 제거 |

---

## 다음 진입 — 우선순위

### 🔴 [P0] C8 — scripts/verify CLI 마이그레이션 (호환성 깨짐)

C2 commit 후 **모든 기존 invocation 패턴이 clap parse-fail**:

```
이전: --eviction-policy h2o --h2o-keep-ratio 0.5 --h2o-decay 0.0
이후: eviction h2o --keep-ratio 0.5 --decay 0.0
```

`--eviction-policy`/`--h2o-*`/`--d2o-*`/`--sink-size`/`--streaming-window` flag는
**clap이 unknown arg로 reject**. 영향:
- `scripts/run_device.py` invocation (단, generate가 받는 argv는 사용자가 입력 → CLI 호환성은 사용자 책임)
- `verify/scenarios/*.yaml` (scenario CLI 문자열)
- `experiments/*.toml`, `scripts/run_benchmark_suite.py`
- CLAUDE.md / docs/*.md 예시
- Android `hosts.toml`의 invocation argv (사용자 hosts.toml에 hard-coded면 깨짐)
- 사용자 shell history / ad-hoc scripts

**가장 먼저 진행 필요**. C8 미진행 시 production 측정 / verify / Android E2E 모두 동작 불가.

### 🟡 [P1] C3 — shim accessor 호출처 cleanup

C2에서 22 shim method로 175 호출처를 무변경 처리했지만, **장기적으로는 enum match 패턴이 자연스러움**:

```rust
// 현재 (C2)
let kr = args.h2o_keep_ratio();  // shim → 0.5 default if not H2O

// C3 후 (호출처가 enum 명시)
let kr = match &args.eviction {
    Some(EvictionCmd::H2o(h)) | Some(EvictionCmd::H2oPlus(h)) => h.keep_ratio,
    _ => return Err(...),  // H2O 분기에서만 의미
};
```

shim 그대로 두고 점진 마이그레이션도 가능. 우선순위 [P1].

### 🟡 [P1] C4-C7 — KvMode + h2o_debug

| C | 작업 |
|---|---|
| C4 | `session/cli/kv_mode.rs` 신설 (KvMode ValueEnum + sub-args flatten) |
| C5 | cli/mod.rs Args에 KvModeArgs flatten + 6 field 제거 (kivi/kv_offload 등) |
| C6 | init.rs / generate.rs / chat REPL의 KIVI/Offload 분기 마이그레이션 |
| C7 | (skip — h2o_debug는 C2에서 이미 env 이전 완료) |

KvMode는 EvictionCmd 패턴 복제. clap subcommand 단일 제약 때문에 ValueEnum + sub-args flatten.

### 🟢 [P2] C9-C10 — 정리

- C9: spec test 신규 (KvMode parse + Args integration)
- C10: handoff doc 갱신 + sprep_args_matrix doc 정정

### 🟢 [P2] SwapArgs / QcfArgs / ExperimentArgs — 별도 sprint

design doc에 명시. 측정 ablation을 별도 binary (`swap_bench` / `argus_bench` / `experiment_runner`)로 분리하는 sprint. S-subcmd C1~C10 완료 후 진행.

### 🟢 [P2] S-1 (BaseArgs 추출) — S-subcmd 완료 후

기존 sprint plan의 S-1은 S-subcmd 완료 후 자연스럽게 흡수. EvictionCmd + KvModeArgs + EvictionCommonArgs가 이미 sub-struct 형태라 BaseArgs flatten 컨테이너로 정리 쉬움.

---

## 즉시 진행 권장 순서

1. **C8 (CLI 마이그레이션)** — 호환성 회복. 다음 session 첫 작업.
2. C4-C6 (KvMode 도입) — C8 후 자연스러운 다음 단계.
3. C9-C10 (test + doc)
4. 별도 sprint — SwapArgs / QcfArgs / ExperimentArgs binary 분리
5. 별도 sprint — S-1 BaseArgs 추출

---

## Risk

| R | 항목 | 상태 |
|---|---|---|
| R1 | clap subcommand가 chat 등 다른 binary와 conflict | 해소 (generate 단일 binary 한정 도입, 사용자 결정대로 chat/ppl/batch/eval-ll binary 분리는 별도 sprint) |
| R2 | `EvictionCmd::None` vs `Option<EvictionCmd>::None` 이중성 | shim `Args::eviction_policy()`이 정규화 — 사용자 체감 0 |
| R3 | 기존 shell history / scripts CLI 마이그레이션 누락 | **P0 C8에서 일괄 처리 필요** |
| R4 | KvMode `requires_ifs` clap 4 호환성 | 미검증 (C4에서 cargo check) |
| R5 | h2o_debug 사용자 환경 하드코딩 | C2에서 env 이전 (shim 메서드가 `LLMRS_H2O_DEBUG` env 검사). 기존 `--h2o-debug` flag는 reject |
| R6 | Galaxy S25 / Jetson 디바이스 회귀 | C8 완료 후 측정 필요 (S25 generate 32 tok smoke + ppl baseline) |

---

## 검증 게이트

각 commit 후 자동:
- `cargo check --workspace --bins --release` PASS
- `cargo test -p llm_rs2 --lib --release session::` PASS (현 52건)

C8 후 추가:
- `cargo build --workspace --bins --release` PASS
- Galaxy S25 generate 32 tok smoke (qwen2.5-1.5b-q4_0.gguf + greedy + repetition_penalty 1.1)
- ppl 1 line smoke (NLL ε<1e-6 vs baseline)

---

## 다음 session 진입 명령

```
"C8 진행"     ← scripts/verify CLI 마이그레이션 (호환성 회복, 최우선)
"C4 진행"     ← KvMode subcommand 도입 (C8 후 권장 순서)
"C3 진행"     ← shim 호출처를 enum match로 정리 (선택, 점진 가능)
"전체 sprint review" ← 우선순위 + 디바이스 게이트 일정 결정
```

---

## 환경 + 재현

```bash
# Worktree
cd /home/go/Workspace/llm_rs2/.claude/worktrees/s1_args_cleanup
git log --oneline master..HEAD   # 8 commits since master

# 호스트 sanity
cargo check --workspace --bins --release
cargo test -p llm_rs2 --lib --release session::

# 작업 종료 후
# (사용자가 별도 결정 시)
git checkout master
git merge --ff-only worktree-s1_args_cleanup
```

---

## 데이터 아티팩트

- `.agent/data/sprep_args_matrix/args_fields_unique.txt` — 142 field (v2, 숫자 포함)
- `.agent/data/sprep_args_matrix/args_matrix.tsv` — 8 bucket × 142 field
- `.agent/data/sprep_args_matrix/args_matrix.txt` — padded 매트릭스
- `.agent/data/sprep_args_matrix/refs_<bucket>.txt` — 분기별 refs

C8 작업 시 마이그레이션 sed 패턴은 `s_subcmd_design_2026_05_19.md` §마이그레이션 매핑에 명시되어 있음.
