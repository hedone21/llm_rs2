---
name: profile
description: 온디바이스 프로파일링을 실행하고 결과를 시각화한다. 성능 병목 분석 시 사용.
allowed-tools: Bash, Read
argument-hint: "<android_command> <output_name>"
---

# Profiling

온디바이스 프로파일링 → 시각화 → 벤치마크 기록을 자동화한다.

## 자동 프로파일링 (권장)

```bash
./.agent/skills/profiling/scripts/auto_profile.sh \
    --cmd "<디바이스에서 실행할 전체 명령>" \
    --output-name "<식별자>"
```

### 예시
```bash
# CPU 프로파일링
./.agent/skills/profiling/scripts/auto_profile.sh \
    --cmd "/data/local/tmp/generate --model-path /data/local/tmp/models/llama3.2-1b --prompt 'Hello' -n 128 -b cpu" \
    --output-name "cpu_baseline"

# OpenCL 프로파일링
./.agent/skills/profiling/scripts/auto_profile.sh \
    --cmd "/data/local/tmp/generate --model-path /data/local/tmp/models/llama3.2-1b --prompt 'Hello' -n 128 -b opencl" \
    --output-name "opencl_baseline"
```

## 파이프라인 단계

1. `scripts/android_profile.py` — 디바이스에서 프로파일 데이터 수집 → `results/data/*.json`
2. `scripts/visualize_profile.py` — JSON → 그래프 생성 → `results/plots/*.png`
3. `scripts/update_benchmark_summary.py` — 벤치마크 요약 갱신

## 수동 프로파일링

개별 스크립트로 세밀한 제어가 필요할 때:
```bash
python3 scripts/android_profile.py --cmd "..." --output results/data/run.json
python3 scripts/visualize_profile.py results/data/run.json -o results/plots/run.png
```

## 주의사항

- 바이너리가 디바이스에 배포된 상태여야 한다 (deploy-test 스킬 선행)
- 디바이스에서 다른 무거운 작업이 실행 중이면 결과가 왜곡된다
- 결과 JSON은 `results/data/`에 커밋 (테스트 데이터), 그래프는 gitignored
