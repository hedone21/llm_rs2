# llm.rs 실험 실행 가이드 — 에이전트 인수인계 문서

> **목적**: 이 문서 하나로 실험 에이전트가 호스트 PC와 타겟 디바이스에서 llm.rs의 모든 실험을 독립적으로 수행할 수 있도록 한다.
> **마지막 업데이트**: 2026-03-19

---

## Changelog

| 날짜 | 변경 내용 |
|------|----------|
| 2026-03-19 | 초판. QCF Round 1 결과 포함. 전체 바이너리/러너/디바이스 가이드 통합. |

---

## 1. 바이너리 목록

| 바이너리 | 용도 |
|---------|------|
| `generate` | 메인 추론. PPL 평가, eval-ll, 텍스트 생성, 실험 모드, 프로파일링 |
| `generate_hybrid` | CPU↔GPU 동적 전환 추론 |
| `test_backend` | CPU vs OpenCL 커널 정확성 비교 |
| `micro_bench` | 개별 연산자 벤치마크 (MatMul, Softmax, RMSNorm, RoPE) |
| `test_model` | 모델 로딩 검증 |
| `signal_injector` | Resilience 신호 주입 (스트레스 테스트용) |

---

## 2. 빌드

### 호스트 (x86_64)

```bash
cargo build --release -p llm_rs2 --bin generate
# 결과: target/release/generate
```

### Android (aarch64)

```bash
source android.source
cargo build --target aarch64-linux-android --release -p llm_rs2 --bin generate
# 결과: target/aarch64-linux-android/release/generate
```

### 전체 바이너리 한번에

```bash
cargo build --release --bin generate --bin generate_hybrid --bin test_backend --bin micro_bench --bin test_model --bin signal_injector
```

### CPU 특화 플래그 (`.cargo/config.toml`에 설정됨)

| 타겟 | SIMD |
|------|------|
| x86_64 | AVX2 + FMA |
| aarch64-linux-android | NEON + dotprod |

---

## 3. 디바이스 레지스트리

### devices.toml 구조

```toml
[devices.host]
name = "Dev Workstation"
[devices.host.connection]
type = "local"                          # local | adb | ssh
[devices.host.build]
binary_dir = "target/release"
[devices.host.paths]
work_dir = "/tmp/llm_rs2"
model_dir = "models/llama3.2-1b"

[devices.pixel]
name = "Pixel Phone"
[devices.pixel.connection]
type = "adb"
serial = ""                             # 빈 문자열 = 첫 연결 디바이스
[devices.pixel.build]
target = "aarch64-linux-android"
env_file = "android.source"
binary_dir = "target/aarch64-linux-android/release"
[devices.pixel.paths]
work_dir = "/data/local/tmp"
model_dir = "/data/local/tmp/models/llama3.2-1b"
eval_dir = "/data/local/tmp/llm_rs2/eval"
lib_dir = "/data/local/tmp"             # LD_LIBRARY_PATH (OpenCL 라이브러리)
```

### 호스트 vs 디바이스 차이

| | 호스트 | Android 디바이스 |
|---|--------|-----------------|
| 빌드 타겟 | 네이티브 | `aarch64-linux-android` |
| 환경 소싱 | 없음 | `source android.source` |
| 모델 경로 | `models/llama3.2-1b/` | `/data/local/tmp/models/llama3.2-1b` |
| 실행 | 직접 `./target/release/generate` | `adb shell "LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/generate"` |
| GPU | 없음 | Adreno (OpenCL) |
| CPU SIMD | AVX2 | NEON+dotprod |

---

## 4. 실행 방법

### 4-1. 통합 러너 (권장): `scripts/run_device.py`

빌드 → 배포 → 실행을 한번에 처리.

```bash
# 디바이스 목록
python scripts/run_device.py --list-devices

# 호스트 실행
python scripts/run_device.py -d host generate --prompt "Hello" -n 128

# Android 실행 (OpenCL)
python scripts/run_device.py -d pixel generate --prompt "Hello" -n 128 -b opencl

# 빌드 건너뛰기
python scripts/run_device.py -d pixel --skip-build generate --prompt "Hello" -n 128

# 드라이런 (실행 없이 명령만 출력)
python scripts/run_device.py -d pixel --dry-run generate --prompt "Hello" -n 128

# 평가 파일 함께 배포
python scripts/run_device.py -d pixel --deploy-eval generate --prompt "Hello" -n 128

# 추가 바이너리 함께 빌드/배포
python scripts/run_device.py -d pixel --extra-bin test_backend generate --prompt "Hello" -n 128

# test_backend 실행
python scripts/run_device.py -d pixel test_backend --backends auto,opencl
```

### 4-2. 레거시 셸 러너

```bash
./.agent/skills/testing/scripts/run_android.sh generate --prompt "Hello" -n 128
./.agent/skills/testing/scripts/run_android.sh -s <serial> test_backend --backends auto,opencl
```

### 4-3. 직접 실행 (호스트)

```bash
./target/release/generate --model-path models/llama3.2-1b --prompt "Hello" -n 128 -b cpu
```

---

## 5. generate 바이너리 실행 모드

### (A) 텍스트 생성 — 기본 모드

```bash
generate --model-path <path> --prompt "..." -n 128 -b cpu --temperature 0
```

stdout에 생성된 텍스트 출력 (`\r` 스트리밍).

### (B) PPL 평가 — `--ppl`

```bash
generate --model-path <path> --ppl <text_file> -b cpu --kv-type f32 --temperature 0 \
  --eviction-policy sliding --kv-budget 900 --protected-prefix 4
```

stdout에 JSON 출력:
```json
{
  "ppl": 118.66,
  "total_nll": 4010.66,
  "token_count": 2047,
  "tokens_per_second": 69.15,
  "wall_time_s": 29.60,
  "qcf_metrics": [{"step":900, "action":"sliding", "raw_value":0.0011, "tokens_affected":1, "cache_pos_before":901, "cache_pos_after":900}],
  "eviction_count": 1147,
  "config": { "model":"...", "eviction_policy":"sliding", "kv_budget":900, ... }
}
```

### (C) Eval-LL — `--eval-ll --eval-batch`

다지선다 NLL 평가 (MMLU 등).

```bash
generate --model-path <path> --eval-ll --eval-batch batch.json -b cpu --kv-type f32 \
  --eviction-policy sliding --kv-budget 900 --protected-prefix 4
```

입력 (`batch.json`):
```json
[{"id":"q1", "prompt":"Q: ...\nA. ...\nB. ...\nAnswer:", "choices":[" A"," B"," C"," D"]}]
```

출력 (stdout JSON):
```json
{
  "results": [{"id":"q1", "choice_nlls":[1.2,2.3,1.5,3.4], "predicted":0, "eviction_count":3}],
  "config": {...},
  "wall_time_s": 12.34
}
```

### (D) 실험 모드 — `--experiment-output`

토큰별 상세 기록 (JSONL).

```bash
generate --model-path <path> --prompt "..." -n 128 -b opencl \
  --experiment-output results.jsonl \
  --experiment-logits-topk 10 \
  --experiment-sample-interval 1
```

각 토큰별 JSONL 레코드:
```json
{"pos":0, "token_id":892, "text":" time", "tbt_ms":35.38, "forward_ms":35.0, "cache_pos":19, "top_logits":[[892,16.67],[3347,16.20]], "sys":{"rss_mb":1768, "cpu_pct":634}}
```

마지막 줄에 summary:
```json
{"_summary":true, "total_tokens":128, "ttft_ms":1200, "avg_tbt_ms":52.3, "eviction_count":2}
```

### (E) 실험 스케줄 — `--experiment-schedule`

Resilience 신호를 토큰 위치 기반으로 주입.

```bash
generate --model-path <path> --prompt "..." -n 2048 -b opencl \
  --experiment-output results.jsonl \
  --experiment-schedule schedule.json \
  --enable-resilience --resilience-transport unix:/data/local/tmp/resilience.sock
```

스케줄 JSON:
```json
[
  {"at_token": 100, "signal": "ThermalAlert", "level": "Warning"},
  {"at_token": 500, "signal": "MemoryPressure", "level": "Critical"},
  {"at_token": 800, "signal": "ThermalAlert", "level": "Normal"}
]
```

### (F) 프로파일링 — `--profile`

```bash
generate --model-path <path> --prompt "..." -n 128 \
  --profile --profile-dir results/profile \
  --profile-probes ops,latency,scores,cache \
  --profile-interval 1
```

---

## 6. generate 핵심 CLI 플래그

### 모델 & 입력

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--model-path` | `models/llama3.2-1b` | HuggingFace Safetensors 모델 경로 |
| `--prompt` | `"Hello, world! I am a"` | 입력 프롬프트 |
| `--prompt-file` | — | 프롬프트 파일 (--prompt 오버라이드) |
| `-n` | 20 | 생성 토큰 수 |

### 백엔드 & 메모리

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `-b, --backend` | `cpu` | `cpu` 또는 `opencl` |
| `--max-seq-len` | 2048 | 최대 시퀀스 길이 |
| `--threads` | 0 (자동) | CPU 스레드 수 |

### 샘플링

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--temperature` | 0.8 | 0 = greedy |
| `--greedy` | false | temperature=0 강제 |
| `--top-p` | 0.9 | Nucleus sampling |
| `--top-k` | 40 | Top-K sampling |

### KV Cache

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--kv-type` | `f16` | `f32`, `f16`, `q4`. **QCF에는 f32 필수** |
| `--kv-layout` | `head` | `head` (HeadMajor) 또는 `seq` (SeqMajor) |
| `--kv-budget` | 0 (무제한) | 고정 KV budget (토큰). 초과 시 eviction |
| `--kv-budget-ratio` | 0.0 | prompt 길이 비율. kv-budget과 상호배타 |

### Eviction 정책

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--eviction-policy` | `none` | `none`, `sliding`, `streaming`, `h2o`, `h2o_plus`, `d2o` |
| `--protected-prefix` | 자동 | score-based→4, sliding→prompt길이. **실험 시 명시 필수** |
| `--eviction-window` | 1024 | sliding window 크기 |
| `--eviction-target-ratio` | 0.75 | eviction 시 유지 비율 |
| `--sink-size` | 4 | StreamingLLM sink tokens |

### H2O 전용

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--h2o-keep-ratio` | 0.5 | Heavy-hitter 유지 비율 |
| `--h2o-decay` | 0.0 | 중요도 점수 지수 감소 |
| `--h2o-tracked-layers` | 0 (전체) | score 추적 레이어 수 |
| `--h2o-raw-scores` | false | 시간 정규화 비활성화 |

### D2O 전용

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--d2o-keep-ratio` | 0.75 | 유지 비율 |
| `--d2o-beta` | 0.7 | EMA 임계값 |
| `--d2o-merge-e` | 1.0 | 병합 안정성 상수 |

### KIVI

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--kivi` | false | Q2 KV 압축 활성화. eviction과 상호배타 |
| `--kivi-residual-size` | 32 | 잔여 버퍼 크기 (32의 배수) |

### 가중치

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--weight-dtype` | `f16` | `f16` 또는 `q4` (로딩 시 양자화) |

---

## 7. 실험 러너 스크립트

### 스트레스 테스트: `scripts/stress_test_device.py`

6단계 종합 테스트.

```bash
python scripts/stress_test_device.py --device pixel                       # 전체
python scripts/stress_test_device.py --device pixel --phases 1,4,5        # 선택 단계
python scripts/stress_test_device.py --device pixel --dry-run             # 미리보기
python scripts/stress_test_device.py --device pixel --cooldown 120        # 냉각 간격
python scripts/stress_test_device.py --device pixel --thermal-limit 45.0  # 열 임계
```

| Phase | 테스트 |
|-------|--------|
| 0 | 빌드 & 배포 |
| 1 | 열 안정성 (3× 2048토큰, OpenCL) |
| 2 | 성능 지속성 (10× 128토큰, TBT 변동계수) |
| 3 | 메모리 안정성 (긴 시퀀스 + eviction, RSS 추적) |
| 4 | 백엔드 정확성 (CPU vs OpenCL 수치 오류) |
| 5 | 출력 품질 (eviction vs no-eviction 텍스트 비교) |
| 6 | Resilience 신호 (MemoryPressure, ThermalAlert 주입) |

### 벤치마크 스위트: `scripts/run_benchmark_suite.py`

매개변수화된 매트릭스 (백엔드 × 프리필 × 디코드 × eviction).

```bash
python scripts/run_benchmark_suite.py --device pixel                      # 전체
python scripts/run_benchmark_suite.py --device pixel --skip-build         # 빌드 건너뛰기
python scripts/run_benchmark_suite.py --device pixel --skip-eviction      # eviction 제외
python scripts/run_benchmark_suite.py --device pixel --dry-run
```

### 비교 벤치마크: `scripts/run_comparison_benchmark.py`

CPU vs OpenCL 3시나리오 비교.

```bash
python scripts/run_comparison_benchmark.py --device pixel --backend all
python scripts/run_comparison_benchmark.py --device pixel --backend cpu
```

### 프로파일링: `scripts/android_profile.py`

추론 중 온디바이스 모니터링 (온도, CPU/GPU 주파수, RSS).

```bash
python scripts/android_profile.py \
  --cmd "/data/local/tmp/generate --model-path /data/local/tmp/models/llama3.2-1b \
         --prompt 'Hello' -n 128 -b opencl" \
  --output-dir results/data
```

### QCF 검증: `experiments/qcf_validation/run_all.sh`

```bash
./experiments/qcf_validation/run_all.sh              # 전체 (~4.5시간)
./experiments/qcf_validation/run_all.sh --phase 1    # PPL sweep (~15분)
./experiments/qcf_validation/run_all.sh --phase 2    # NIAH (~40분)
./experiments/qcf_validation/run_all.sh --phase 3    # QA (~30분)
./experiments/qcf_validation/run_all.sh --phase 4    # MMLU (~3시간)
./experiments/qcf_validation/run_all.sh --phase 5    # 분석만
```

---

## 8. 모델 정보

### Llama 3.2 1B

| 파라미터 | 값 |
|---------|-----|
| hidden_size | 2048 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 |
| head_dim | 64 |
| num_hidden_layers | 16 |
| intermediate_size | 8192 |
| vocab_size | 128256 |
| rope_theta | 500000.0 |

| 경로 | 용도 |
|------|------|
| `models/llama3.2-1b/` | 호스트 PC (gitignored) |
| `/data/local/tmp/models/llama3.2-1b` | Android 디바이스 |

```bash
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir models/llama3.2-1b
```

---

## 9. Eviction 정책 상세

| 정책 | CLI | 동작 | QCF 수집 |
|------|-----|------|---------|
| `none` | `--eviction-policy none` | eviction 없음 | — |
| `sliding` | `--eviction-policy sliding --kv-budget N` | budget 초과 시 oldest 1토큰 삭제 | `compute_sliding_qcf()` |
| `streaming` | `--eviction-policy streaming --sink-size 4` | sink 보존 + sliding | `compute_sliding_qcf()` |
| `h2o` | `--eviction-policy h2o --h2o-keep-ratio 0.5` | 3-partition: prefix+HH+recent | `compute_eviction_qcf()` |
| `h2o_plus` | `--eviction-policy h2o_plus` | per-head GQA-aware H2O | `compute_eviction_qcf()` |
| `d2o` | `--eviction-policy d2o --d2o-keep-ratio 0.75` | H2O + cosine merge | D2OHandler |

---

## 10. QCF 시스템

### 개요

```
D(action) = α × Q(action)
Q: QCF 값, 추가 forward pass 없이 eviction 시 부수적으로 수집
α: 오프라인 calibration 계수
```

### QCF 유형

| 유형 | 수식 |
|------|------|
| Eviction | `Σ_evicted attn(t)×‖V(t)‖₁ / Σ_all attn(t)×‖V(t)‖₁` |
| Sliding | `prune_count / total_active` |
| Quantization | `0.6×NMSE(K) + 0.4×NMSE(V)` |

### 핵심 규칙

- **`--kv-type f32` 필수**: QCF V-norm 계산은 F32에서만 동작
- **`--temperature 0`**: 결정론적 실험 필수
- **cumulative total 사용**: avg는 sliding에서 비단조. 항상 `sum(m["raw_value"] for m in metrics)`
- **`--protected-prefix 4` 명시**: 미지정 시 sliding은 전체 prompt 보호 → eviction 안 일어남
- JSON 키: 신버전 `"qcf_metrics"`, 구버전 `"proxy_metrics"` — 둘 다 확인

### Round 1 결과 요약

| Policy | ρ(QCF_total, ΔPPL) |
|--------|-------------------|
| Sliding | **1.000** |
| H2O | **1.000** |

NIAH/QA: sliding eviction은 prefill 이후라 task에 무해. MMLU: 1B 모델 ICL이 random 수준.

---

## 11. 벤치마크 프롬프트

| 파일 | 내용 |
|------|------|
| `experiments/prompts/benchmark_prompts.json` | PPL 5도메인 + NIAH 10needle/8filler + QA 10문항 |
| `experiments/prompts/assemble_niah.py` | NIAH 프롬프트 자동 조합 |
| `experiments/prompts/prefill_*.txt` | 합성 프롬프트 (128/512/1024 tokens) |
| `experiments/proxy_validation/texts/eval_text.txt` | PPL 평가 텍스트 (~2047 tokens) |

NIAH 조합:
```bash
python experiments/prompts/assemble_niah.py --needle N-PASS --depth 0.25 --blocks 20
```

MMLU 다운로드:
```bash
pip install datasets
python experiments/qcf_validation/scripts/run_mmlu.py --download
```

---

## 12. 분석 도구

| 스크립트 | 용도 |
|---------|------|
| `experiments/analysis/quality_metrics.py` | EMR, FDT, Suffix EMR, ROUGE-L, BLEU-4 |
| `experiments/analysis/compare.py` | Baseline vs experiment 비교 리포트 |
| `experiments/qcf_validation/scripts/analyze.py` | QCF Spearman ρ, α fitting, 시각화 |
| `scripts/visualize_profile.py` | 프로파일 JSON → 그래프 |

---

## 13. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| QCF 전부 0 | `--kv-type f16` | `--kv-type f32` |
| Eviction 안 일어남 | budget 미설정 | `--kv-budget N` (N < prompt length) |
| Sliding eviction 안 일어남 | protected-prefix가 prompt 전체 | `--protected-prefix 4` 명시 |
| PPL JSON 파싱 실패 | stderr 혼입 | `2>/dev/null` 또는 stderr 분리 |
| `qcf_metrics` 키 없음 | 구버전 바이너리 | `proxy_metrics`도 확인 |
| adb push 실패 | 디바이스 미연결 | `adb devices` 확인 |
| OpenCL 초기화 실패 | GPU 드라이버 | `-b cpu`로 폴백 |
| 모델 로딩 실패 | 경로 오류 | model_dir 확인, config.json 존재 여부 |

---

## 14. 파일 경로 참조

```
/home/go/Workspace/llm_rs2/
├── engine/src/bin/                      # 6개 바이너리
│   ├── generate.rs                      # 메인 추론 (모든 모드)
│   ├── generate_hybrid.rs               # CPU↔GPU 동적 전환
│   ├── test_backend.rs                  # 커널 정확성 비교
│   ├── micro_bench.rs                   # 연산자 벤치마크
│   ├── test_model.rs                    # 모델 로딩 테스트
│   └── signal_injector.rs              # Resilience 신호 주입
├── engine/src/core/qcf/                 # QCF 모듈
├── scripts/
│   ├── run_device.py                    # **통합 러너 (권장)**
│   ├── device_registry/                 # 디바이스 설정 모듈
│   ├── stress_test_device.py            # 6단계 스트레스 테스트
│   ├── run_benchmark_suite.py           # 매개변수화된 벤치마크
│   ├── run_comparison_benchmark.py      # CPU vs GPU 비교
│   ├── android_profile.py               # 온디바이스 프로파일링
│   └── visualize_profile.py             # 프로파일 시각화
├── .agent/skills/testing/scripts/
│   └── run_android.sh                   # 레거시 셸 러너
├── experiments/
│   ├── prompts/                         # 벤치마크 프롬프트
│   ├── analysis/                        # 분석 스크립트
│   ├── qcf_validation/                  # QCF 검증 실험
│   └── reports/                         # 리포트
├── devices.toml                         # 디바이스 레지스트리
├── android.source                       # NDK 크로스 컴파일 환경
└── results/data/                        # 실험 결과 JSON (커밋됨)
```
