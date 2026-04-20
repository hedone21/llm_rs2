---
name: deploy-test
description: Android 디바이스에 빌드→배포→테스트를 수행한다. 백엔드 검증, E2E 추론, 스트레스 테스트, 디바이스 관리 지원. '디바이스 테스트', 'Android 빌드', '디바이스 배포', 'E2E', '스트레스 테스트', 'adb', '디바이스 목록' 등의 요청 시 이 스킬을 사용.
allowed-tools: Bash, Read
argument-hint: "<binary> [args...] 또는 stress --duration 60"
---

# Deploy & Test

Android 디바이스에서 빌드, 배포, 테스트를 수행한다. Tester 에이전트가 주로 사용.

## 3-tier 테스트 전략

| Tier | 환경 | 명령 | 검증 대상 |
|------|------|------|----------|
| **T1** | 호스트 | `cargo test` (→ /sanity-check) | 토크나이저, shape 추론, 플랫폼 무관 로직 |
| **T2** | 디바이스 | `run_device.py test_backend` | CPU vs OpenCL 커널 정확성 |
| **T3** | 디바이스 | `run_device.py generate` | 전체 추론 파이프라인 |

## 빠른 실행

```bash
# Device Registry 기반 (권장)
python scripts/run_device.py -d pixel <binary> [args...]
python scripts/run_device.py -d pixel --skip-build <binary> [args...]

# 레거시 스크립트 (빌드→푸시→실행 일체형)
./.agent/skills/testing/scripts/run_android.sh <binary> [args...]
```

## 주요 테스트 시나리오

### 백엔드 검증 (Tier 2)
```bash
python scripts/run_device.py -d pixel test_backend
```

### E2E 추론 (Tier 3)

**기본 포맷: GGUF** — `.gguf` 파일을 직접 지정한다. Safetensors는 GGUF 미준비 모델 또는 포맷 비교 시에만 사용.

```bash
python scripts/run_device.py -d pixel generate \
    --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf \
    --prompt "Hello" -n 128
```

### 백엔드별 테스트
```bash
python scripts/run_device.py -d pixel generate -b cpu --prompt "Hello" -n 128
python scripts/run_device.py -d pixel generate -b opencl --prompt "Hello" -n 128
```

### 스트레스 테스트
```bash
python3 ./.agent/skills/testing/scripts/stress_test_adb.py \
    --cmd "/data/local/tmp/generate --model-path ... -n 256" \
    --duration 60 --switch-interval 5
```

## 모델 가중치

**기본 포맷은 GGUF**. `--model-path`는 `.gguf` 파일을 직접 가리킨다 (generate.rs가 확장자로 포맷 자동 판별).

| 경로 | 용도 |
|------|------|
| `models/<model>/*.gguf` | 호스트 PC 테스트용 (gitignored) |
| `/data/local/tmp/models/<model>/*.gguf` | Android 디바이스용 |
| `models/llama3.2-1b/` (safetensors) | Safetensors 비교/폴백 전용 |

```bash
# 호스트에 다운로드
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir models/llama3.2-1b

# 디바이스에 푸시 (최초 1회)
adb push models/llama3.2-1b /data/local/tmp/models/llama3.2-1b
```

## Device Registry

TOML 기반 디바이스 설정: `devices.toml` (프로젝트 루트).

```bash
# 디바이스 관리
python scripts/device_registry.py discover    # 스캔 & 등록
python scripts/device_registry.py list        # 등록 목록
python scripts/device_registry.py validate    # 스키마 검증

# 벤치마크 스크립트
python scripts/stress_test_device.py --device pixel --phases 1,4
python scripts/run_benchmark_suite.py --device pixel --dry-run
python scripts/run_comparison_benchmark.py --device pixel --dry-run
```

패키지: `scripts/device_registry/` — config.py, connection.py, builder.py, deployer.py, discover.py.

## Deploy-only 모드 (실행 없이 빌드+푸시)

generate + manager 두 바이너리를 디바이스에 올리되 실행하지 않는다:
```bash
python scripts/run_device.py -d pixel --skip-exec generate --extra-bin llm_manager
```

## 사전 조건

- `adb devices`로 디바이스 연결 확인
- Android 빌드 환경 설정 (최초 1회):
  ```bash
  # 자동 프로빙 (권장)
  python scripts/device_registry.py bootstrap-host

  # 또는 수동으로 템플릿 복사 후 편집
  cp hosts.toml.example hosts.toml
  ```
- 모델 가중치가 디바이스에 존재해야 함

## 결과 보고

테스트 완료 후 다음을 보고:
1. 통과/실패 여부
2. 실패 시 에러 메시지 전문
3. 성능 수치 (tok/s, 메모리 등) — 해당 시
