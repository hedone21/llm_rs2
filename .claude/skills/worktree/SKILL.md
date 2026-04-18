---
name: worktree
description: git worktree를 사용해 격리된 작업 공간을 만들고, 작업 완료 후 master에 병합·정리한다. '워크트리', 'worktree', '격리 작업', '분리된 작업', '새 작업 공간', '병렬 작업', 'worktree 병합', 'worktree 삭제' 등의 요청 시 반드시 이 스킬을 사용. 명시적 요청일 때만 사용 (일반 브랜치 작업에는 사용하지 않음).
allowed-tools: Bash, Read
argument-hint: "create <name> [--type feat|fix|refactor|...] | merge [--squash] | list | clean <name>"
---

# Worktree

git worktree 기반 격리 작업 공간 관리. 긴 빌드/테스트 실행 중 다른 작업을 병렬로 진행하거나, 실험적 변경을 메인 레포에서 격리하고 싶을 때 사용한다.

## 설계 원칙

- **명시적 요청 시만**: 일반적인 브랜치 작업(간단한 수정, 원샷 커밋)에는 사용하지 않는다. 사용자가 "worktree"/"격리 작업"/"분리"를 명시한 경우에만 사용.
- **워크트리 = 새 브랜치**: 기존 브랜치 체크아웃은 지원하지 않는다. 항상 새 브랜치를 만든다. 단순함과 일관성 우선.
- **병합 전 검증 필수**: `/sanity-check` 실패 시 병합을 진행하지 않는다. master는 항상 빌드 가능해야 한다.
- **삭제는 확인 후**: 워크트리와 브랜치는 지운 뒤 복구가 까다로우므로, 병합 성공 후 사용자에게 반드시 물어본다.

## 규약

**워크트리 경로**: `../llm_rs2-<slug>/` (프로젝트 디렉토리의 형제)
- `<slug>` = 브랜치명의 `/` → `-` (예: `feat/qcf-audit` → `../llm_rs2-feat-qcf-audit/`)

**브랜치명**: `<type>/<name>` — Conventional Commits type 접두사
- 기본 type: `work` (미지정 시)
- 허용 type: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `work`

**병합 방식**: `git merge --no-ff` (default)
- 머지 커밋 보존 → 브랜치 범위 추적 가능, 기존 히스토리 스타일(`Merge branch 'feat/gemma3-4b-support'`)과 일관
- `--squash` 옵션 전달 시 `git merge --squash` 사용 (커밋 1개로 압축, 작은 작업용)

## 커맨드

### `create <name> [--type <type>]`

새 워크트리와 브랜치를 만든다.

```bash
# 현재 위치가 메인 레포 루트인지 확인
git rev-parse --show-toplevel
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# master가 최신인지 확인 (로컬 master 기준 분기)
git fetch origin master 2>/dev/null || true
BASE_BRANCH=master

# 브랜치명 구성
TYPE="${TYPE:-work}"          # 인자 없으면 work
BRANCH="${TYPE}/${NAME}"
SLUG=$(echo "$BRANCH" | tr '/' '-')
WT_PATH="../llm_rs2-${SLUG}"

# 이미 존재하면 에러
if git worktree list | grep -q "$WT_PATH"; then
  echo "Error: worktree already exists at $WT_PATH" >&2
  exit 1
fi
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "Error: branch $BRANCH already exists" >&2
  exit 1
fi

# 워크트리 + 새 브랜치 생성 (master에서 분기)
git worktree add -b "$BRANCH" "$WT_PATH" "$BASE_BRANCH"

# 생성 직후 자동으로 워크트리로 이동 (절대경로로 resolve해서 pwd 안정화)
cd "$WT_PATH"
WT_ABS=$(pwd)

echo "✓ Created worktree: $WT_ABS (branch: $BRANCH)"
echo "✓ cwd switched to: $WT_ABS"
```

**사용 예시**:
- "qcf 리팩토링 작업을 worktree에서 해줘" → `NAME=qcf-refactor TYPE=refactor`
- "새 실험용 worktree 만들어줘 이름은 adreno-test" → `NAME=adreno-test TYPE=work`

**cwd 처리 (중요 — 하네스 동작)**:

Claude Code 하네스는 Bash 도구 호출이 끝날 때마다 **cwd를 프로젝트 루트로 리셋**한다 ("Shell cwd was reset to …" 메시지 참고). 즉, 단일 Bash 호출 안에서의 `cd`만 유효하고, 다음 호출은 다시 메인 레포에서 시작한다. 이 현실에 맞춰 다음 두 가지를 함께 지킨다:

1. **생성 직후 기록**: `create`가 성공하면 `WT_ABS` 절대경로를 대화 메모리에 기록한다 (예: "worktree: /home/.../llm_rs2-<slug>"). Read/Edit/Write는 절대경로로 워크트리 파일을 직접 다룬다.
2. **Bash 호출은 cd 프리픽스**: 이후 워크트리 안에서 수행해야 하는 Bash 명령은 한 줄 복합 커맨드로 `cd "$WT_ABS" && <실제 명령>` 형태로 감싼다. 예:
   ```bash
   cd /home/go/Workspace/llm_rs2-fix-foo && cargo check
   cd /home/go/Workspace/llm_rs2-fix-foo && git add -A && git commit -m "..."
   ```
   이 패턴은 상태(cwd)에 의존하지 않으므로 실패 모드가 단순하고, 하네스 리셋에 영향받지 않는다.

Read/Edit/Write 도구는 절대경로를 받으므로 cwd와 무관하다 — 워크트리 내부 파일 편집 시 `/home/.../llm_rs2-<slug>/engine/...` 형태 절대경로를 사용한다.

사용자의 대화형 셸 cwd는 바뀌지 않는다. 사용자가 터미널에서 직접 작업하려면 `cd $WT_ABS`를 안내한다.

**주의**:
- 메인 레포의 미커밋 변경은 새 워크트리에 복제되지 않는다 (git 기본 동작). 이게 의도한 격리다.
- Android 크로스 컴파일 시 워크트리 내부에도 `hosts.toml`이 필요하다 — `python scripts/device_registry.py bootstrap-host`로 생성하거나 메인 레포에서 복사 (`hosts.toml`은 gitignored이므로 워크트리 체크아웃에 자동 포함되지 않는다).

### `merge [--squash]`

워크트리의 브랜치를 master에 병합한다. **반드시 워크트리 내부에서 실행**하며, 하네스 cwd 리셋 때문에 단일 Bash 호출 전체를 `cd "$WT_ABS" && { ... }` 로 감싼다 (아래 스크립트를 통째로).

```bash
# 0. 현재가 워크트리인지 확인 (메인 레포에서 merge 호출 차단)
CURRENT_PATH=$(git rev-parse --show-toplevel)
MAIN_PATH=$(git worktree list --porcelain | awk '/^worktree / {print $2; exit}')
if [ "$CURRENT_PATH" = "$MAIN_PATH" ]; then
  echo "Error: run 'worktree merge' from inside a worktree, not the main repo" >&2
  exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" = "master" ] || [ "$BRANCH" = "HEAD" ]; then
  echo "Error: current branch is $BRANCH — merge from a feature branch" >&2
  exit 1
fi

# 1. 미커밋 변경 차단
if ! git diff-index --quiet HEAD -- || [ -n "$(git ls-files --others --exclude-standard)" ]; then
  echo "Error: uncommitted changes in worktree. Commit or stash them first." >&2
  git status --short
  exit 1
fi

# 2. Sanity check (워크트리 내부에서 실행)
echo "Running sanity-check before merge..."
# /sanity-check 스킬을 호출하거나 직접 실행
cargo fmt --all -- --check || { echo "fmt failed"; exit 1; }
cargo clippy --workspace --all-targets -- -D warnings || { echo "clippy failed"; exit 1; }
cargo test --workspace || { echo "tests failed"; exit 1; }

# 3. 메인 레포로 이동해서 master 병합
cd "$MAIN_PATH"
git checkout master

# 선택: master가 뒤처져 있으면 경고 (강제 pull은 하지 않음)
BEHIND=$(git rev-list --count master..origin/master 2>/dev/null || echo 0)
if [ "$BEHIND" -gt 0 ]; then
  echo "Warning: local master is $BEHIND commits behind origin/master"
  echo "Consider: git pull --ff-only origin master  (ask user first)"
fi

# 4. 병합 (기본 --no-ff)
if [ "$SQUASH" = "1" ]; then
  git merge --squash "$BRANCH"
  echo "Squashed changes staged. Review and commit:"
  echo "  git commit"
else
  git merge --no-ff "$BRANCH" -m "Merge branch '$BRANCH'"
  echo "✓ Merged $BRANCH into master (no-ff)"
fi

# 5. 삭제 여부는 사용자에게 물어본다 (스크립트로 자동 삭제 금지)
echo ""
echo "Merge complete. cwd: $(pwd) (main repo on master)"
echo "Ask user before running 'worktree clean $BRANCH'."
```

**병합 후 cwd**: 병합은 메인 레포에서 수행되므로 에이전트 cwd는 자연스럽게 메인 레포(master)로 이동한다. 삭제하지 않기로 결정한 경우, 사용자가 해당 워크트리에서 추가 작업을 이어가려면 다시 `cd $WT_PATH` 하거나, 스킬이 다시 호출되면 `create` 대신 해당 워크트리로 명시적으로 이동한다.

**병합 방식 선택**:
- 기본 `--no-ff`: 일반적인 feature/refactor 작업. 머지 커밋 남김.
- `--squash`: 작은 fix, 많은 작은 커밋을 하나로 압축하고 싶을 때.

**실패 시 동작**:
- fmt/clippy/test 실패 → 즉시 중단. master 체크아웃도 하지 않음. 사용자가 워크트리에서 수정 후 재시도.
- 머지 충돌 → 사용자에게 알리고 중단. 자동 resolve 시도 금지.

### `list`

현재 worktree 목록을 보여준다.

```bash
git worktree list
```

### `clean <name>`

병합된 워크트리와 브랜치를 삭제한다. **병합 성공 후 사용자 확인을 거친 뒤에만 호출**.

```bash
# name은 브랜치명 (예: feat/qcf-audit) 또는 slug
BRANCH="$NAME"
SLUG=$(echo "$BRANCH" | tr '/' '-')

# 0. 에이전트 cwd가 삭제 대상 워크트리 내부라면 거부
#    (삭제 후 harness의 shell이 없어진 디렉토리에 묶여 모든 bash 명령이
#    "Path does not exist"로 실패하는 edge case 방지. 세션 재시작 전까지 복구 불가.)
CUR_ABS=$(pwd -P)
case "$CUR_ABS" in
  */llm_rs2-${SLUG}|*/llm_rs2-${SLUG}/*)
    echo "Error: current cwd ($CUR_ABS) is inside the worktree to be deleted." >&2
    echo "Run 'cd <main-repo-path>' (or start a new session in the main repo) before 'worktree clean'." >&2
    exit 1 ;;
esac

# 메인 레포로 이동 (cwd를 워크트리로 두면 아래 remove가 실패한다)
MAIN_PATH=$(git worktree list --porcelain | awk '/^worktree / {print $2; exit}')
cd "$MAIN_PATH"

# 워크트리 실제 경로를 git이 추적하는 목록에서 조회 (추정 경로에 의존하지 않음)
WT_PATH=$(git worktree list --porcelain \
  | awk -v br="refs/heads/$BRANCH" '
      /^worktree / {p=$2}
      $0=="branch "br {print p; exit}')
if [ -z "$WT_PATH" ]; then
  # fallback: 규약 경로
  WT_PATH="$MAIN_PATH/../llm_rs2-${SLUG}"
fi
WT_ABS=$(cd "$WT_PATH" 2>/dev/null && pwd || echo "$WT_PATH")

# 병합 여부 확인 (safety) — 병합되지 않은 브랜치는 거부
if ! git branch --merged master | grep -q "^[* ]*$BRANCH$"; then
  echo "Error: branch $BRANCH is NOT merged into master" >&2
  echo "Refusing to delete. Merge first, or use 'git branch -D' manually if intentional." >&2
  exit 1
fi

# 1. 워크트리 제거 — target/, .idea/ 등 gitignored 파일 때문에 거부되지 않도록 --force
#    (git worktree remove는 워크트리 디렉토리 자체도 함께 삭제한다)
git worktree remove --force "$WT_PATH" || {
  echo "Warning: git worktree remove failed; falling back to manual cleanup"
  git worktree prune
}

# 2. 디렉토리 잔재 제거 (git이 무언가 남겨둔 경우 대비)
#    절대경로가 메인 레포 내부가 아닌지 sanity-check 후 rm
if [ -d "$WT_ABS" ] \
   && [ "$WT_ABS" != "$MAIN_PATH" ] \
   && [ "$WT_ABS" != "/" ] \
   && [ "$WT_ABS" != "$HOME" ]; then
  rm -rf "$WT_ABS"
  echo "✓ Removed leftover directory: $WT_ABS"
fi

# 3. 브랜치 제거 (merged check는 위에서 통과했으므로 -d 가능)
git branch -d "$BRANCH"

echo "✓ Removed worktree $WT_ABS and branch $BRANCH"
```

**삭제 범위**: `git worktree remove --force`는 워크트리 디렉토리 **전체를 폴더째 삭제**한다 (gitignored 빌드 산출물 `target/` 포함). 뒤이은 `rm -rf`는 git이 남긴 잔재가 있을 경우의 안전망이다 — 루트/홈/메인 레포 경로로 착각해 실행되지 않도록 가드를 둔다.

**사용자 확인 절차 (스킬 사용자가 따를 것)**:
1. `merge` 성공 후 스크립트가 자동으로 `clean`을 호출하지 **않는다**.
2. 에이전트는 사용자에게 "$BRANCH 워크트리를 삭제할까요?" 확인한다.
3. 승인 시에만 `clean <branch>` 실행.

## 전체 워크플로우 예시

**시나리오**: "QCF 모듈 리팩토링을 worktree에서 해줘"

```bash
# 1. 생성
worktree create qcf-refactor --type refactor
# → ../llm_rs2-refactor-qcf-refactor/ 생성, 브랜치 refactor/qcf-refactor

cd ../llm_rs2-refactor-qcf-refactor
# ... 코드 수정, 커밋 반복 ...

# 2. 병합 (sanity-check 자동 실행)
worktree merge
# → fmt/clippy/test 통과 시 master에 no-ff 머지

# 3. 사용자 확인 후 정리
# 에이전트: "refactor/qcf-refactor 워크트리를 삭제할까요?"
# 사용자: "응"
cd ../llm_rs2
worktree clean refactor/qcf-refactor
```

## 실패 케이스 처리

| 상황 | 동작 |
|------|------|
| `create` 시 워크트리/브랜치 이미 존재 | 에러 후 중단. 사용자가 다른 이름 지정 또는 기존 것 사용. |
| 메인 레포에서 `merge` 호출 | 에러. 워크트리 내부에서 실행하도록 안내. |
| 워크트리에 미커밋 변경 | 에러. `git status` 출력 후 사용자가 커밋/stash. |
| sanity-check 실패 | 에러. master 체크아웃 안 함. 사용자가 수정 후 재시도. |
| 머지 충돌 | git이 멈춘다. 사용자에게 알리고 자동 resolve 시도 금지. |
| `clean` 시 미병합 브랜치 | 거부. 실수로 작업 유실 방지. |
| `clean` 시 현재 cwd가 삭제 대상 워크트리 | 거부. harness shell이 사라진 디렉토리에 묶여 세션 전체가 잠김. 메인 레포로 이동 후 재시도. |
| 외부에서 이미 디렉토리 삭제 → `git worktree list`에 stale entry | `git worktree prune`으로 정리. 브랜치는 `git branch -d <name>` 별도 처리. |

## 참고

- **CLAUDE.md 규칙**: 완료 시 자동 커밋 → worktree 내부에서도 동일하게 적용.
- **Android 빌드**: 워크트리에서도 `hosts.toml`이 필요 — `bootstrap-host`로 생성하거나 메인 레포에서 복사 (gitignored이라 자동 포함 안 됨). `run_device.py`가 NDK env를 자동 주입하므로 셸 환경변수는 별도로 설정할 필요 없다.
- **빌드 캐시**: 각 워크트리는 자체 `target/`을 가진다. 디스크 사용량 주의. cargo의 `CARGO_TARGET_DIR` 공유도 가능하지만 lock 경합 위험으로 권장하지 않음.
