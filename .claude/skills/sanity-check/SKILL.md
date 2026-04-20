---
name: sanity-check
description: cargo fmt + clippy + unit test를 실행하여 코드 품질을 검증한다. --spec으로 Spec 불변식 테스트 + 커버리지도 실행 가능. 코드 구현 완료 후 반드시 실행. '빌드', 'cargo check', 'cargo test', '컴파일', '린트', 'clippy', 'fmt' 등의 요청 시에도 이 스킬을 사용.
allowed-tools: Bash, Read
argument-hint: "[--no-test] [--spec]"
---

# Sanity Check

코드 품질 검증을 수행한다. Implementer가 구현 완료 후 반드시 실행해야 하는 게이트.

## 빌드

```bash
# 호스트 빌드 (CPU-only, 개발용)
cargo check --workspace              # 전체 크레이트 문법 검사
cargo build -p llm_rs2               # 엔진 빌드
cargo build -p llm_manager           # 매니저 빌드

# Android 크로스 컴파일 (run_device.py가 hosts.toml로 NDK env 자동 주입)
python scripts/run_device.py -d pixel --skip-exec --skip-deploy generate
# (cargo 직접 호출은 비권장: source android.source && cargo build --target aarch64-linux-android ...)
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

**기본 포맷: GGUF**. `--model-path`는 `.gguf` 파일을 직접 지정한다 (generate.rs가 확장자로 포맷 자동 판별). Safetensors는 GGUF 미준비 모델 또는 포맷 비교 시에만 사용.

```bash
# 추론 실행 (GGUF 기본)
cargo run --release --bin generate -- \
    --model-path models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf \
    --prompt "Hello" -n 128
```

모델 경로: `models/<model>/*.gguf` (gitignored)

## 실패 시 대응

- **fmt 실패**: `cargo fmt --all`로 자동 수정 후 재확인
- **clippy 경고**: 경고 내용을 분석하고 코드 수정
- **test 실패**: 실패한 테스트를 분석하고 원인 보고

## Spec 검증 (--spec 플래그)

`--spec` 플래그가 지정되면 기본 검사에 추가하여 Spec 관련 테스트를 실행한다.

```bash
# Spec 통합 테스트
cargo test -p llm_rs2 --test spec          # Engine INV 테스트
cargo test -p llm_manager --test spec      # Manager INV 테스트

# 3계층 커버리지 통합 검사 (Static INV + 비-INV 추적성 + 품질)
scripts/check_spec_coverage.sh
```

### --spec 실패 시 대응

- **spec 테스트 실패**: 실패한 INV를 식별하고 관련 spec/ 문서와 대조
- **static INV 위반**: 위반된 INV에 대해 Cargo.toml 또는 코드 구조 수정 필요
- **커버리지 누락**: 새 INV 추가 시 tests/spec/ 테스트 작성이 필요함을 보고

## 규칙

- 커밋 전에 반드시 통과해야 한다
- clippy 경고를 `#[allow(...)]`로 억제하지 않는다 (정당한 사유 제외)
- Unit tests: `#[cfg(test)] mod tests` 블록에 작성. 모든 feature/fix에 테스트 필수
