# Handoff — step-4 ↔ sprint1 통합 머지 (2026-05-20)

## 진입 문장

**"step-4 머지 진행"** — `step-4-backup` 브랜치(107 commits)를 master(=sprint1 PR
머지본) 위로 통합한 새 브랜치 `merge-step4-into-master`를 생성·해결·PR.

## 현재 상태

- **master (local + origin/master)** = `e31bd698 Merge pull request #1` (sprint1 W-AUF + B-3a 통합).
- **step-4-backup (local 전용)** = `fd95e916 refactor(layer): step 4-D ...` (107 commits, 미푸시 보존).
- **worktree-sprint1_auf_loader (origin push 완료)** = `24c6af46`. 본 worktree는 sprint1 마무리 직후 상태.
- 통합 방향: **옵션 B** (base=origin/master, merge in step-4-backup) — sprint1 보전 + step-4 통합 merge commit.
- **.antigravitycli/ gitignore 추가는 별건** (사용자 직접 master에 push 예정).

## 단계 1 — 새 worktree + 브랜치 생성

```bash
cd /home/go/Workspace/llm_rs2
git worktree add ../merge-resolve -b merge-step4-into-master origin/master
cd ../merge-resolve
# 또는 .claude/worktrees/ 하위에 두려면:
# git worktree add .claude/worktrees/merge_resolve -b merge-step4-into-master origin/master
```

## 단계 2 — step-4-backup merge 시도

```bash
git merge step-4-backup --no-edit 2>&1 | tail -10
# → 5 파일, 10 hunks conflict 예상
```

## 단계 3 — 충돌 해결 (작은 → 큰 순서)

### 작은 hunks 7개 (~30분)

| 파일 | hunk | 결정 |
|---|---|---|
| `engine/src/models/weights/mod.rs:30-35` | `compute_qcf_swap` + `backing` re-export | **origin/master 채택**. `compute_qcf_weight_swap` 이름 변경분은 step-4 callsite를 새 이름으로 일괄 변경. |
| `engine/src/models/weights/swap_executor.rs:304-308` | `cuda_mmap_alias_buffer::CudaMmapRegistration` path | **origin/master 채택**. step-4가 옮기려 한 `memory::cuda::mmap` 경로 변경분은 별도 후속 PR 또는 통합 본 브랜치에서 buffer/ 모듈을 일괄 재배치. |
| `engine/src/models/weights/secondary_mmap.rs:27-38` | `crate::auf::*` import (sprint1) vs `llm_shared::auf::*` (step-4) | **origin/master 채택**. AUF crate 이동 정착. |
| `secondary_mmap.rs:729-738` | `backend_supports_rpcmem_secondary` 시그니처 | **origin/master 채택** (`pub(crate)` + `core::backend::Backend`). |
| `secondary_mmap.rs:841-868` | `is_auf_path` 함수 | **origin/master 채택** (loader/auf/secondary에 동일 함수). |
| `secondary_mmap.rs:1387-1428` | `check_auf_metadata` 함수 | **origin/master 채택** (loader/auf/secondary에 동일). |

각 충돌은 `git checkout --theirs <file>` 후 step-4가 추가한 logical 변경만 patch하는 것이 빠를 수도. 단 weights/mod.rs는 두 변경 모두 살려야 하니 수동 편집.

### 중간 hunk 1개 — secondary_mmap.rs:957-1359 (400+ 줄, ~1시간)

step-4가 secondary_mmap.rs 안에 정의한 helpers (`detect_backend_tag`,
`resolve_backend_tag_candidates`, `auf_dtype_to_engine` 등)가 origin/master의
`engine/src/models/loader/auf/secondary.rs`에 모두 있는지 확인:

```bash
# 새 worktree 안에서
grep -rn "fn detect_backend_tag\|fn resolve_backend_tag_candidates" \
  engine/src/models/loader/auf/
```

- 모두 존재 → **origin/master 채택** (한 줄 re-export로 끝).
- 일부 누락 → 누락된 함수를 `loader/auf/secondary.rs`에 이전(step-4의
  변경 내용 그대로).

`detect_backend_tag`는 step-4 단독 변경일 가능성 높다 (sprint1은 다른 path 사용). **누락 가능성 높음 → 옮겨야**.

### 거대 hunk 2개 — generate.rs (~2~4시간)

#### hunk 1: line 49-1441 (~1390줄)

- **master(step-4)**: main() 앞이 거의 비고 `let ctx = llm_rs2::session::init::SessionInitCtx::build(&args)?;` 한 줄 후 ctx unpack.
- **origin/master(sprint1)**: 모든 init 로직 inline + W-AUF-1 CLI flags 처리 (primary_variant/primary_dtype/no_self_secondary/eos_token_id).

**통합 절차**:
1. step-4가 만든 `engine/src/session/init.rs::SessionInitCtx::build` 함수가 보유하는 fields와 책임을 우선 확인.
2. sprint1이 inline에 추가한 W-AUF-1 CLI 처리 로직을 **SessionInitCtx::build 안으로 이동**.
3. `SessionInitCtx` struct에 W-AUF-1 신규 fields (예: `primary_variant_choice`, `primary_dtype_choice`, `disable_self_secondary`, `eos_token_override`) 추가.
4. main()에서는 step-4 style로 `ctx.field` unpack.

#### hunk 2: line 1491-2440 (~950줄)

- main() backend init + LoadConfig dispatch.
- step-4가 ctx로 추출한 backend 초기화 + sprint1의 3-way primary_format dispatch가 같은 영역에서 충돌.

**통합 절차**:
1. backend init 부분도 ctx로 추출 (이미 step-4가 추출했을 가능성 — `session/init.rs` 확인).
2. sprint1의 `let primary_format = detect_primary_format(...)` + LoadConfig 분기를 ctx 또는 main() 안에 유지 (step-4 패턴과 일관).
3. `--secondary-gguf deprecated` warning + secondary_source 처리 path도 ctx에 통합.

#### hunk 3: line 3622-3627 (5줄)

별도 확인 후 단순 결정.

## 단계 4 — 호스트 게이트 검증

```bash
cargo build -p llm_rs2 --no-default-features --lib 2>&1 | tail -3
cargo test -p llm_rs2 --lib --no-default-features 2>&1 | tail -3
cargo test -p llm_rs2 --test test_auf_gguf_byte_equivalence --no-default-features 2>&1 | tail -3
cargo fmt -p llm_rs2
cargo fmt --check -p llm_rs2
```

**목표**: 1159+ PASS, byte-equivalence 2/2 PASS, fmt clean. clippy는 사전 회귀
22~23건 허용 (본 통합으로 추가 회귀 0).

## 단계 5 — 디바이스 회귀 검증

```bash
python scripts/run_device.py -d galaxy_s25 generate -- \
  --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-aos.auf \
  --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json \
  --backend qnn_oppkg --threads 6 --num-tokens 32 --prompt "I am"
```

**목표**: B-3a fix 회귀 0. fluent semantic output, Avg TBT ≤ 32.12 ms.

(필요 시 디바이스 binary는 worktree 안에서 빌드 + push.
`scripts/run_device.py --skip-exec`가 deploy를 자주 누락하므로
`adb push target/.../generate /data/local/tmp/generate` 수동 push 검증 필수)

## 단계 6 — push + PR

```bash
git push -u origin merge-step4-into-master
# GitHub UI에서 PR 생성 + 머지
# 머지 후 step-4-backup 정리: git branch -d step-4-backup
```

## 충돌 해결 정책 요약

- **default 채택**: origin/master (sprint1 결과).
- **step-4 변경 보전 대상**: 
  - `compute_qcf_weight_swap` rename callsites (decider.rs, qcf module 등)
  - `SessionInitCtx` 추출 패턴 (session/init.rs)
  - `engine/src/buffer/{auf_view_buffer,...}` 14개 파일 위치 정착 후 step-4가 옮기려 한 path 일관 적용
  - `memory::cuda::mmap` 등 step-4 신규 경로 → buffer/ 통합 정책에 맞춰 정리
- **버려도 됨**:
  - master worktree 자체의 `is_auf_path`, `check_auf_metadata` 등 (sprint1에 이미 있음)

## 위험 / 검증

| 위험 | 완화 |
|---|---|
| SessionInitCtx 통합 중 W-AUF-1 CLI flag 누락 | 호스트 + S25 device로 `--primary-variant`/`--primary-dtype`/`--no-self-secondary`/`--eos-token-id` 4종 flag 모두 동작 확인 |
| Qwen2 qkv bias forward path 회귀 | byte-equivalence test 통과 + S25 generate fluent output 확인 |
| step-4 buffer/ 분산 vs sprint1 buffer/ 통합 충돌 | sprint1 정책(`engine/src/buffer/{14 files}`) 채택. step-4가 옮기려 한 backend별 위치는 후속 sprint로 |

## 관련 파일

- 본 handoff: `.agent/todos/handoff_merge_step4_into_master_2026_05_20.md` (현재 파일, sprint1 worktree 내)
- step-4 작업: `step-4-backup` 브랜치 (local 전용)
- sprint1 종결: `.agent/todos/handoff_sprint1_w_auf_2_complete_2026_05_20.md` + `handoff_b_3a_auf_device_correctness_2026_05_20.md`
- master HEAD: `e31bd698 Merge pull request #1`

## 메모리 갱신

`/home/go/.claude/projects/-home-go-Workspace-llm-rs2/memory/` 에
`project_merge_step4_into_master.md` 신설. MEMORY.md에 한 줄 인덱스 추가.

## 재진입

**"step-4 머지 진행"** — 단계 1 (새 worktree 생성)부터 시작.
