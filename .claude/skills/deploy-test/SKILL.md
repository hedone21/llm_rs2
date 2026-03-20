---
name: deploy-test
description: Android 디바이스에 빌드→배포→테스트를 수행한다. 백엔드 검증, E2E 추론, 스트레스 테스트 지원.
allowed-tools: Bash, Read
argument-hint: "<binary> [args...] 또는 stress --duration 60"
---

# Deploy & Test

Android 디바이스에서 빌드, 배포, 테스트를 수행한다. Tester 에이전트가 주로 사용.

## 빠른 실행

```bash
# 방법 1: 레거시 스크립트 (빌드→푸시→실행 일체형)
./.agent/skills/testing/scripts/run_android.sh <binary> [args...]

# 방법 2: Device Registry 기반 (권장)
python scripts/run_device.py -d pixel <binary> [args...]
python scripts/run_device.py -d pixel --skip-build <binary> [args...]
```

## 주요 테스트 시나리오

### 백엔드 검증 (Tier 2)
CPU vs OpenCL 커널 정확성 비교:
```bash
python scripts/run_device.py -d pixel test_backend
```

### E2E 추론 (Tier 3)
전체 추론 파이프라인 검증:
```bash
python scripts/run_device.py -d pixel generate \
    --model-path /data/local/tmp/models/llama3.2-1b \
    --prompt "Hello" -n 128
```

### 백엔드별 테스트
```bash
python scripts/run_device.py -d pixel generate -b cpu --prompt "Hello" -n 128
python scripts/run_device.py -d pixel generate -b opencl --prompt "Hello" -n 128
```

### 스트레스 테스트
배경 앱 전환으로 실사용 환경 시뮬레이션:
```bash
python3 ./.agent/skills/testing/scripts/stress_test_adb.py \
    --cmd "/data/local/tmp/generate --model-path ... -n 256" \
    --duration 60 --switch-interval 5
```

## 사전 조건

- `adb devices`로 디바이스 연결 확인
- Android 빌드 시 `source android.source` 필수 (run_device.py는 자동 처리)
- 모델 가중치가 디바이스에 존재: `/data/local/tmp/models/llama3.2-1b`

## 결과 보고

테스트 완료 후 다음을 보고:
1. 통과/실패 여부
2. 실패 시 에러 메시지 전문
3. 성능 수치 (tok/s, 메모리 등) — 해당 시
