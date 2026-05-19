# Handoff — S-subcmd C2.1/C8 종결, C3/C4 진입점 (2026-05-19)

**Worktree HEAD**: `ba7d2cff` (master `c4caedd6` + 2 commits)
**선행 doc**:
- `handoff_s_subcmd_c3_entry_2026_05_19.md` (이전 entry — S-cleanup + C1/C2 종결)
- `sprep_args_matrix_2026_05_18.md` (142 field 매트릭스 + 17 sub-struct 분류)
- `s_subcmd_design_2026_05_19.md` (EvictionCmd + KvMode 사양 + C1~C10 분해)

---

## TL;DR

| Sprint | Commit | 작업 | 결과 |
|---|---|---|---|
| S-subcmd C2.1 | `51f09c16` | `eviction` wrapper subcommand 도입 (TopLevelCmd + shim helper) | clap이 `eviction h2o --keep-ratio 0.5` 형태 수용 |
| S-subcmd C8 | `ba7d2cff` | docs/scripts/verify 17 파일 마이그레이션 + invocation reorder | 호환성 회복 — production binary 사용 가능 |

C2 직후 모든 invocation parse-fail이었던 상태 → 표준 호출 형태 복귀.

---

## 발견 / 결정

### D1 — clap derive subcommand 위치 제약

clap derive `#[command(subcommand)]`는 subcommand 뒤의 parent flag를 **reject**한다 (unknown arg). 따라서 표준 호출 형태는 다음과 같다 (canonical form):

```
generate <base + EvictionCommonArgs> eviction <policy> <policy-args>
```

예시:
```bash
generate --model-path ... -p "..." --kv-budget 1024 --protected-prefix 4 eviction h2o --keep-ratio 0.5
generate --model-path ... -p "..." eviction sliding --window 1024
generate --model-path ... -p "..." --kv-budget-ratio 0.4 eviction d2o --keep-ratio 0.75
```

base Args (`-p`/`-n`/`--greedy`/`--model-path`/...) + EvictionCommonArgs (`--kv-budget`/`--protected-prefix`/`--memory-threshold-mb`/`--kv-budget-ratio`/`--initial-kv-capacity`/`--min-kv-cache`/`--eviction-target-ratio`)는 **subcommand 앞**에 와야 한다.

policy-specific 인자 (Sliding `--window`, Streaming `--sink`/`--recent-window`, H2O/H2O+ `--keep-ratio`/`--decay`/`--tracked-layers`/`--raw-scores`, D2O `--keep-ratio`/`--ema-beta`/`--merge-e`/`--layer-alloc`/`--protected-layers`)는 subcommand 뒤에.

### D2 — wrapper subcommand 패턴 (`TopLevelCmd::Eviction`)

clap derive에서 enum variant들은 직접 top-level subcommand로 등록되므로, design doc이 명시한 `eviction h2o` 패턴은 wrapper 없이는 동작하지 않는다 (variant 이름인 `h2o`/`sliding`/...이 subcommand가 된다).

`session/cli/eviction.rs`에 wrapper enum 추가:
```rust
#[derive(Subcommand, Debug, Clone)]
pub enum TopLevelCmd {
    Eviction {
        #[command(subcommand)]
        policy: EvictionCmd,
    },
}
```

`Args::eviction` field type: `Option<EvictionCmd>` → `Option<TopLevelCmd>`.
22 shim accessor는 helper `current_policy() -> Option<&EvictionCmd>`를 거쳐 unwrap 한 단계 추가만으로 호환 유지 (175+ 호출처 무변경).

---

## 검증

### CLI E2E (호스트 cargo build --release)
6 invocation 패턴 모두 PASS (`[Config] Using N threads` 로그 확인):
- `eviction h2o --keep-ratio 0.5` (+ `--kv-budget` / `--protected-prefix` 앞에)
- `eviction sliding --window 1024`
- `eviction streaming --sink 4 --recent-window 512`
- `eviction h2o-plus --keep-ratio 0.4`
- `eviction d2o --keep-ratio 0.75 --ema-beta 0.7 --merge-e 0.1 --layer-alloc`
- `eviction none`
- subcommand 생략 (`EvictionCmd::None` 등가)

### cargo test
- `cargo test -p llm_rs2 --lib session::` 52 PASS, 0 FAIL
- `cargo test -p llm_rs2 --lib cli::` 15 PASS, 0 FAIL (eviction::tests 9개 포함)

### 변경 통계
- C2.1: 3 파일 (eviction.rs + mod.rs + assembly test), 56 line+ 27 line-
- C8: 18 파일 (8 docs + 5 scripts + 3 verify + 2 experiment), 698 line+ 667 line-

---

## 다음 진입 — 우선순위

### 🔴 [P0] 디바이스 게이트 — Galaxy S25 / Jetson 회귀 확인

C8 후 production CLI 인터페이스가 바뀌었으므로 디바이스 측 invocation도 함께 갱신 필요.

- `scripts/run_device.py`는 사용자 입력 argv를 그대로 generate에 전달 — 코드 변경 X
- `hosts.toml`의 invocation argv (있다면 사용자 hard-coded) — 사용자 확인 필요
- Android `adb shell`로 직접 호출하는 ad-hoc shell script — 사용자 confirm

검증:
```bash
# Galaxy S25 smoke (qwen2.5-1.5b-q4_0.gguf, 6T)
adb -s R3CY408S5SB shell 'cd /data/local/tmp && LD_LIBRARY_PATH=. ./generate \
    --model-path qwen2.5-1.5b-q4_0.gguf --tokenizer-path qwen-tokenizer.json \
    --backend qnn_oppkg -p "The capital of France is" -n 32 --greedy --repetition-penalty 1.1 \
    eviction none'

# Galaxy S25 eviction smoke (KV cache 압박)
adb shell '... -p "..." -n 64 --kv-budget 512 --protected-prefix 4 eviction h2o --keep-ratio 0.5'
```

### 🟡 [P1] C4-C6 — KvMode subcommand 도입

design doc `s_subcmd_design_2026_05_19.md` §"KvMode — 사양" 참조.

| C | 작업 |
|---|---|
| C4 | `session/cli/kv_mode.rs` 신설 — `KvMode` ValueEnum + `KvModeArgs` flatten struct |
| C5 | cli/mod.rs Args에 KvModeArgs flatten. 기존 `kivi`/`kv_offload` field 제거 (helper accessor만) |
| C6 | init.rs / generate.rs / chat REPL의 KIVI/Offload 분기 마이그레이션 |
| C9 | spec test (KvMode parse + Args integration) |
| C10 | handoff doc + sprep_args_matrix doc 정정 |

KvMode는 clap 단일 subcommand 제약 (한 binary에 `#[clap(subcommand)]` 한 번만)으로 EvictionCmd처럼 subcommand 형태가 아니라 ValueEnum + flatten 패턴.

### 🟡 [P1] C3 — shim accessor 호출처 cleanup (점진)

22 shim method를 enum match 패턴으로 점진 마이그레이션. 우선순위 낮음.

```rust
// 현재 shim
let kr = args.h2o_keep_ratio();
// C3 후
let kr = match args.current_policy() {
    Some(EvictionCmd::H2o(h)) | Some(EvictionCmd::H2oPlus(h)) => h.keep_ratio,
    _ => return Err(...),
};
```

`current_policy()` 헬퍼가 private이라 `pub` 승격 또는 helper 노출이 선행 필요.

### 🟢 [P2] SwapArgs / QcfArgs / ExperimentArgs binary 분리 sprint

별도 sprint. design doc 참조.

### 🟢 [P2] S-1 BaseArgs 추출

S-subcmd C4-C10 완료 후. EvictionCmd + KvModeArgs + EvictionCommonArgs가 sub-struct 형태라 BaseArgs flatten 컨테이너로 정리 쉬움.

---

## Risk

| R | 항목 | 상태 |
|---|---|---|
| R1 | clap subcommand 뒤 parent flag reject | 해소 — 표준 호출 형태로 정착 (canonical = base 앞 / policy 뒤) |
| R2 | `EvictionCmd::None` vs `Option<TopLevelCmd>::None` 이중성 | shim `Args::eviction_policy()`가 둘 다 `"none"`으로 정규화 |
| R3 | 기존 shell history / 미커밋 ad-hoc script 마이그레이션 누락 | C8에서 docs + scripts + verify 일괄 처리. 사용자 ad-hoc은 본 handoff §"디바이스 게이트"로 안내 |
| R4 | `--sink-size` manager mock 도메인 false positive | C8 휴리스틱이 mock_manager_commands 마커로 보존. 수동 검토 OK |
| R5 | 디바이스 회귀 가능성 | 본 worktree에선 호스트 sanity만 확인. S25/Jetson 회귀는 별도 게이트 필요 (P0) |
| R6 | scripts/android_profile.py / scripts/local_profile.py 로그 parsing regex 미갱신 | log emit 형식이 새 subcommand 형태인지 확인 후 갱신. 별도 작업 |

---

## 검증 게이트

각 commit 후:
- `cargo check --workspace --bins --release` PASS (C2.1 + C8 모두)
- `cargo test -p llm_rs2 --lib session:: cli::` PASS (52 + 15 = 67)

다음 작업 (C4-C6) 후 추가:
- 새 spec test (`engine/tests/spec/test_kv_mode_parse.rs`)
- Galaxy S25 KIVI smoke (`--kv-mode kivi --kivi-bits 2`)
- Galaxy S25 offload smoke (`--kv-mode offload --offload-mode raw`)

---

## 다음 session 진입 명령

```
"C4 진행"        ← KvMode subcommand 도입 (다음 자연 순서)
"디바이스 게이트" ← Galaxy S25 / Jetson invocation 회귀 확인 (먼저 검증 필요 시)
"C3 진행"        ← shim 호출처를 enum match로 정리 (선택, 점진 가능)
"전체 sprint review" ← 우선순위 + 디바이스 게이트 일정 결정
```

`"디바이스 게이트"` 또는 `"C4 진행"` 권장 — C8 후 production binary 사용 가능하므로 디바이스 측 갱신 함께 진행이 자연스럽다.

---

## 환경 + 재현

```bash
cd /home/go/Workspace/llm_rs2

# C2.1 + C8 검증
cargo check --workspace --bins --release
cargo test -p llm_rs2 --lib session:: cli::
./target/release/generate --model-path /tmp/x -p "test" eviction h2o --keep-ratio 0.5
# Expected: [Config] Using N threads ... (parse OK)

# 잔여 stale CLI 형태 grep
grep -rnE '(--eviction-policy|--h2o-keep-ratio|--d2o-keep-ratio|--eviction-window|--streaming-window|--h2o-recent-window|--h2o-debug)' \
    docs/ scripts/ verify/ experiments/PLAN.md experiments/benchmarks/ experiments/prompts/README.md 2>/dev/null
# Expected: (legacy 도메인 + Python argparse parameter만 — manager mock client + scripts internal helper)
```

---

## 데이터 아티팩트

- `.agent/data/sprep_args_matrix/args_fields_unique.txt` (142 field)
- `.agent/data/sprep_args_matrix/args_matrix.tsv` (8 bucket × 142 field)
- `/home/go/.claude/jobs/77310ddb/migrate_c8.py` (평면 sed, 47 라인 변경)
- `/home/go/.claude/jobs/77310ddb/reorder_invocations.py` (invocation reorder, 17 block)

(`$CLAUDE_JOB_DIR` 안 artifact는 session 종료 시 cleanup. 보존 필요시 .agent/data/ 로 이동.)
