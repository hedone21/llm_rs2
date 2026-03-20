---
name: sanity-check
description: cargo fmt + clippy + unit test를 실행하여 코드 품질을 검증한다. 코드 구현 완료 후 반드시 실행.
allowed-tools: Bash, Read
argument-hint: "[--no-test]"
---

# Sanity Check

코드 품질 검증을 수행한다. Implementer가 구현 완료 후 반드시 실행해야 하는 게이트.

## 실행

```bash
# 전체 검사 (fmt + clippy + test)
./.agent/skills/developing/scripts/sanity_check.sh

# 개별 실행
cargo fmt --all -- --check        # 포매팅 검사
cargo clippy --workspace -- -D warnings   # 린트 검사
cargo test -p llm_rs2             # 엔진 유닛 테스트
cargo test -p llm_shared          # 공유 타입 유닛 테스트
```

## 실패 시 대응

- **fmt 실패**: `cargo fmt --all`로 자동 수정 후 재확인
- **clippy 경고**: 경고 내용을 분석하고 코드 수정
- **test 실패**: 실패한 테스트를 분석하고 원인 보고

## 규칙

- 커밋 전에 반드시 통과해야 한다
- clippy 경고를 `#[allow(...)]`로 억제하지 않는다 (정당한 사유 제외)
