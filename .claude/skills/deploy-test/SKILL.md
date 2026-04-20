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
# 호스트에 Safetensors 다운로드
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir models/llama3.2-1b

# 디바이스에 푸시 (최초 1회)
adb push models/llama3.2-1b /data/local/tmp/models/llama3.2-1b
```

### Safetensors → GGUF 변환

프로젝트 내장 변환 스크립트 사용 (llm.rs + llama.cpp 양쪽 호환):

```bash
# F16 (모든 2D weight + embed F16, norm F32)
python scripts/convert_safetensors_to_gguf.py --outtype f16 \
    models/qwen2.5-1.5b \
    models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf

# Q4_0 (2D weight quantized, embed F16, norm F32 — 기본값)
python scripts/convert_safetensors_to_gguf.py --outtype q4_0 \
    models/qwen2.5-1.5b \
    models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf
```

지원 아키텍처: llama, qwen2, gemma/gemma2/gemma3. 기본 `--outtype`은 `q4_0`.

### GGUF 호스트 CPU 비교 테스트 (llm.rs vs llama.cpp)

용도별 도구 매트릭스:

| 비교 항목 | 도구 | 비고 |
|----------|------|------|
| **성능** (prefill/decode t/s) | `llama-bench` | 표준. 커스텀 prompt 불가, 길이·KV dtype·depth 자유 |
| **텍스트 출력 sanity** | `llama-simple` | greedy만. seed/temp 제어 불가 |
| **품질 (PPL)** | `llama-perplexity` | 표준 dataset 필요 |

**성능 비교 (기본 워크플로우)**:

```bash
# llama.cpp 기준값
/home/go/Workspace/llama.cpp/build-host/bin/llama-bench \
    -m models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
    -p 128,512,2048 -n 32,128 -t 8 -ctk f16 -ctv f16 -r 3 -o md

# llm.rs 동일 조건
./target/release/generate -b cpu \
    -m models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
    -p "$(printf 'x %.0s' {1..512})" -n 128 --temperature 0
# → stderr의 "Prefill: ... tok/s"와 "Decode: ... tok/s" 라인 수집
```

**텍스트 sanity check**:

```bash
/home/go/Workspace/llama.cpp/build-host/bin/llama-simple -m <gguf> -n 24 \
    "The capital of France is"
./target/release/generate -b cpu -m <gguf> -p "The capital of France is" \
    -n 24 --temperature 0
```

**주의**: `llama-cli` (983df14+)는 `--no-conversation` 미지원, `llama-completion`은 Qwen2 base에서 토큰 반복 버그. `llama-bench`/`llama-simple`은 영향 없음. 미빌드 시: `cd /home/go/Workspace/llama.cpp/build-host && cmake --build . --target llama-bench llama-simple -j$(nproc)`.


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
