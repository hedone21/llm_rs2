---
name: sanity-check
description: cargo fmt + clippy + unit test를 실행하여 코드 품질을 검증한다. 코드 구현 완료 후 반드시 실행. '빌드', 'cargo check', 'cargo test', '컴파일', '린트', 'clippy', 'fmt' 등의 요청 시에도 이 스킬을 사용.
allowed-tools: Bash, Read
argument-hint: "[--no-test]"
---

# Sanity Check

코드 품질 검증을 수행한다. Implementer가 구현 완료 후 반드시 실행해야 하는 게이트.

## 빌드

```bash
# 호스트 빌드 (CPU-only, 개발용)
cargo check --workspace              # 전체 크레이트 문법 검사
cargo build -p llm_rs2               # 엔진 빌드
cargo build -p llm_manager           # 매니저 빌드

# Android 크로스 컴파일 (반드시 source 먼저)
source android.source
cargo build --target aarch64-linux-android --release -p llm_rs2 --bin generate
```

## 전체 검사 (fmt + clippy + test)

```bash
# 원커맨드 (권장)
./.agent/skills/developing/scripts/sanity_check.sh

# 개별 실행
cargo fmt --all -- --check                  # 포매팅 검사
cargo clippy --workspace -- -D warnings     # 린트 검사
cargo test -p llm_rs2                       # 엔진 유닛 테스트
cargo test -p llm_shared                    # 공유 타입 유닛 테스트
cargo test -p llm_manager                   # 매니저 유닛 테스트
```

## 호스트 추론 테스트

```bash
# 모델 다운로드 (최초 1회)
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir models/llama3.2-1b

# 추론 실행
cargo run --release --bin generate -- --model-path models/llama3.2-1b --prompt "Hello" -n 128
```

모델 경로: `models/llama3.2-1b/` (gitignored)

## 실패 시 대응

- **fmt 실패**: `cargo fmt --all`로 자동 수정 후 재확인
- **clippy 경고**: 경고 내용을 분석하고 코드 수정
- **test 실패**: 실패한 테스트를 분석하고 원인 보고

## 규칙

- 커밋 전에 반드시 통과해야 한다
- clippy 경고를 `#[allow(...)]`로 억제하지 않는다 (정당한 사유 제외)
- Unit tests: `#[cfg(test)] mod tests` 블록에 작성. 모든 feature/fix에 테스트 필수
