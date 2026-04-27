# llm.rs 사용 매뉴얼

> 복사-붙여넣기 가능한 명령어 중심의 사용 가이드.
> 개별 플래그 설명은 `--help`를 참조하고, 이 문서는 **"이것을 하려면 이 조합을 쓰세요"** 시나리오 중심으로 구성한다.

---

## 목차

1. [Quick Start](#1-quick-start)
2. [generate 모드별 가이드](#2-generate-모드별-가이드)
   - [2.1 기본 추론 (CPU / GPU)](#21-기본-추론-cpu--gpu)
   - [2.2 eval-ll (NLL 평가)](#22-eval-ll-nll-평가)
   - [2.3 PPL (Perplexity)](#23-ppl-perplexity)
   - [2.4 KIVI (양자화 캐시)](#24-kivi-양자화-캐시)
   - [2.5 Eviction (KV 캐시 관리)](#25-eviction-kv-캐시-관리)
   - [2.6 Resilience (Manager 연동)](#26-resilience-manager-연동)
   - [2.7 Adaptive Cooperative Prefill (GPU 양보)](#27-adaptive-cooperative-prefill-gpu-양보)
   - [2.8 Tensor Partition (CPU-GPU 협업 추론)](#28-tensor-partition-cpu-gpu-협업-추론)
   - [2.9 KV Offload (캐시 오프로드)](#29-kv-offload-캐시-오프로드)
   - [2.10 Prompt Batch (배치 추론)](#210-prompt-batch-배치-추론)
   - [2.11 Layer Skip (레이어 건너뛰기)](#211-layer-skip-레이어-건너뛰기)
   - [2.12 Chat (멀티턴 REPL / 소켓 IPC)](#212-chat-멀티턴-repl--소켓-ipc)
   - [2.13 Weight Swap (AUF / Secondary GGUF)](#213-weight-swap-auf--secondary-gguf)
     - [2.13.5 QCF↔NLL 측정 (Spearman 상관관계)](#2135-qcfnll-측정-spearman-상관관계)
3. [Manager 가이드](#3-manager-가이드)
   - [3.1 기본 실행](#31-기본-실행)
   - [3.2 Lua 정책 스크립팅](#32-lua-정책-스크립팅)
   - [3.3 mock_manager 사용법](#33-mock_manager-사용법)
4. [실험 워크플로우 레시피](#4-실험-워크플로우-레시피)
   - [4.1 Action Resource Profile (TBT 기록)](#41-action-resource-profile-tbt-기록)
   - [4.2 NLL/QCF 비교 실험](#42-nllqcf-비교-실험)
   - [4.3 벤치마크 (llm.rs vs llama.cpp)](#43-벤치마크-llmrs-vs-llamacpp)
   - [4.4 회귀 테스트](#44-회귀-테스트)
5. [온디바이스 배포](#5-온디바이스-배포)
   - [5.1 빌드 & 배포](#51-빌드--배포)
   - [5.2 모델 배포](#52-모델-배포)
   - [5.3 디바이스 관리](#53-디바이스-관리)
6. [플래그 레퍼런스](#6-플래그-레퍼런스)

---

## 1. Quick Start

### 지원 모델 포맷

| 포맷 | 확장자 | 지원 dtype | 특징 |
|------|--------|-----------|------|
| **HuggingFace Safetensors** | `.safetensors` (디렉토리) | F16, BF16, F32 | `--weight-dtype q4`로 로드 시 Q4_0 변환 가능 |
| **GGUF** | `.gguf` (단일 파일) | Q4_0, Q8_0, F16, F32 | 사전 양자화 모델 직접 로드 (zero-copy, 변환 불필요) |

GGUF의 K-quant 텐서 (Q4_K, Q6_K 등)는 로드 시 F32로 자동 dequant된다.

### Safetensors 모델 (F16)

```bash
cargo build --release

# CPU 추론
./target/release/generate -m models/qwen2.5-1.5b --prompt "Hello" -n 50

# GPU + Q4 양자화 (F16→Q4 로드 시 변환)
./target/release/generate -m models/qwen2.5-1.5b -b opencl --weight-dtype q4 --prompt "Hello" -n 50
```

### GGUF 모델 (사전 양자화)

```bash
# .gguf 파일을 --model-path에 직접 지정
./target/release/generate -m models/llama3.2-1b-q4_0.gguf --prompt "Hello" -n 50

# GPU
./target/release/generate -m models/llama3.2-1b-q4_0.gguf -b opencl --prompt "Hello" -n 50
```

> GGUF 모드에서 `--weight-dtype`은 무시된다 (파일 내 dtype 사용).
> `tokenizer.json`이 GGUF 파일과 같은 디렉토리에 있어야 한다.

**GGUF 모델 다운로드 방법:**

```bash
# HuggingFace에서 Q4_0 GGUF 다운로드
hf download bartowski/Llama-3.2-1B-Instruct-GGUF \
  --include "Llama-3.2-1B-Instruct-Q4_0.gguf" --local-dir models/

# 또는 기존 safetensors에서 순수 Q4_0 GGUF 생성
python scripts/convert_safetensors_to_gguf.py models/llama3.2-1b models/llama3.2-1b-q4_0.gguf
```

### 온디바이스 (Android, Adreno GPU) 추론

```bash
# 최초 1회: 호스트 toolchain 등록 (NDK 자동 감지)
python scripts/device_registry.py bootstrap-host

# 빌드 + 배포 (run_device.py가 hosts.toml로 NDK env 자동 주입)
python scripts/run_device.py -d pixel --skip-exec generate

# 모델 가중치는 한 번만
adb push -r models/qwen2.5-1.5b /data/local/tmp/models/qwen2.5-1.5b

# 실행 (Safetensors + GPU + Q4)
adb shell "LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/generate \
  -m /data/local/tmp/models/qwen2.5-1.5b -b opencl --weight-dtype q4 -n 50 \
  --prompt 'Hello'"

# 실행 (GGUF + GPU)
adb push models/llama3.2-1b-q4_0.gguf /data/local/tmp/models/
adb push models/llama3.2-1b/tokenizer.json /data/local/tmp/models/
adb shell "LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/generate \
  -m /data/local/tmp/models/llama3.2-1b-q4_0.gguf -b opencl -n 50 \
  --prompt 'Hello'"
```

> **권장 조합**: Android + OpenCL + Q4 weight. Adreno GPU에서 llama.cpp CPU 대비 약 +23% 속도.

### 온디바이스 (Jetson) 추론 — cargo-zigbuild 경로 (권장)

`devices.toml`에 `[devices.jetson]`이 등록되어 있고 `run_device.py`가 빌드→배포→실행을 자동화한다.
`cargo-zigbuild`가 호스트의 최신 glibc(예: Arch 2.41)와 보드의 구버전 glibc(JetPack 5.1.x = 2.31) 간 호환을 처리한다.

```bash
# CPU
python scripts/run_device.py -d jetson generate -b cpu --prompt "'Hello'" -n 50

# CUDA (GPU)
python scripts/run_device.py -d jetson generate -b cuda --prompt "'Hello'" -n 50
```

> **prompt에 공백이 있으면 외곽 쌍따옴표 + 내부 홑따옴표** (`"'…'"`)로 감싼다 — SSH 전송 시 셸 토큰화 보호.

#### 호스트 일회성 셋업 (cargo-zigbuild)

`docs/jetson_setup.md` 참조. 요약:
1. `sudo pacman -S zig` (또는 OS별 zig 설치)
2. `cargo install cargo-zigbuild`
3. `~/.cargo/bin`을 PATH에 포함
4. `ssh-copy-id -p <port> user@jetson` (passwordless ssh 키 등록)
5. `python scripts/device_registry.py bootstrap-host`로 hosts.toml 생성

#### 백엔드 비고

- Jetson에는 OpenCL 런타임이 없으므로 `cuda` feature를 쓴다 (`devices.jetson.build.features = ["cuda"]`).
- cudarc는 `fallback-dynamic-loading`이라 호스트에 CUDA SDK 불필요. 보드의 `LD_LIBRARY_PATH=/usr/local/cuda/lib64`만 필요(자동 주입).
- Xavier(Carmel ARMv8.2)는 dotprod/fhm 미지원이지만 zigbuild 타겟 `aarch64-unknown-linux-gnu.2.31`로 통일 (`+neon,+fp16`만 활성).

---

## 2. generate 모드별 가이드

### 2.1 기본 추론 (CPU / GPU)

**CPU 추론 (기본)**

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --prompt "Explain quantum computing" \
  -n 200
```

**Q4 weight로 메모리 절약 — Safetensors (로드 시 F16→Q4 변환)**

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --weight-dtype q4 \
  --prompt "Explain quantum computing" \
  -n 200
```

**Q4 weight — GGUF (사전 양자화, 변환 없이 직접 로드)**

```bash
# GGUF Q4_0 모델을 직접 로드 — 로딩 시간 단축, peak RSS 감소
./target/release/generate \
  -m models/llama3.2-1b-q4_0.gguf \
  --prompt "Explain quantum computing" \
  -n 200
```

> **Safetensors Q4 vs GGUF Q4_0**: 추론 결과와 속도는 동일. GGUF는 로딩 시 F32 중간 버퍼가 없어 peak RSS가 ~6배 낮다.

**OpenCL GPU 추론 (Android/NVIDIA)**

```bash
# Safetensors
./target/release/generate \
  -m models/qwen2.5-1.5b \
  -b opencl --weight-dtype q4 \
  --prompt "Explain quantum computing" -n 200

# GGUF
./target/release/generate \
  -m models/llama3.2-1b-q4_0.gguf \
  -b opencl \
  --prompt "Explain quantum computing" -n 200
```

**CPU→GPU 자동 전환 (CPU prefill → GPU decode)**

prefill은 CPU, decode는 GPU로 자동 전환. `--switch-threshold N` 토큰에서 GPU로 전환.
GPU가 사용 가능하면 보조 백엔드로 자동 초기화되어 SwitchHw 명령도 동작.

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  -b cpu \
  --switch-threshold 50 \
  --weight-dtype q4 \
  --prompt "Explain quantum computing" \
  -n 200
```

**재현 가능한 greedy 생성**

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --greedy \
  --prompt "The capital of France is" \
  -n 20
```

**프롬프트 파일 사용 (긴 프롬프트)**

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --prompt-file experiments/prompts/prefill_1024.txt \
  -n 100
```

**긴 프롬프트에서 피크 메모리 제한 (chunked prefill)**

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --prompt-file experiments/prompts/prefill_1024.txt \
  --prefill-chunk-size 256 \
  -n 100
```

---

### 2.2 eval-ll (NLL 평가)

다지선다 태스크에서 각 선택지의 NLL(Negative Log-Likelihood)을 평가한다.
가장 낮은 NLL을 가진 선택지를 정답으로 예측한다.

**eval-batch JSON 포맷**

```json
[
  {
    "id": "q1",
    "prompt": "Q: What is 2+2?\nA. 3\nB. 4\nC. 5\nAnswer:",
    "choices": [" 3", " 4", " 5"]
  }
]
```

**기본 eval-ll (eviction 없음)**

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --eval-ll \
  --eval-batch data.json \
  --kv-type f32 \
  --greedy
```

**Sliding window eviction + ratio 버짓**

prompt 길이의 75%만 KV 캐시에 유지. 벤치마크 간 공정 비교에 권장.

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --eval-ll \
  --eval-batch data.json \
  --kv-type f32 \
  --eviction-policy sliding \
  --kv-budget-ratio 0.75 \
  --protected-prefix 4 \
  --greedy
```

**H2O eviction + QCF 메트릭**

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --eval-ll \
  --eval-batch data.json \
  --kv-type f32 \
  --eviction-policy h2o \
  --kv-budget-ratio 0.5 \
  --protected-prefix 4 \
  --greedy \
  --qcf-mode both
```

**출력 예시 (stdout JSON)**

```json
{
  "accuracy": 0.72,
  "results": [
    {
      "id": "q1",
      "choice_nlls": [1.23, 0.87, 2.41],
      "predicted": 1,
      "eviction_count": 2,
      "effective_budget": 320
    }
  ],
  "config": {
    "eviction_policy": "h2o",
    "kv_budget_ratio": 0.5
  },
  "wall_time_s": 12.34
}
```

---

### 2.3 PPL (Perplexity)

teacher-forcing 방식으로 참조 텍스트 전체의 perplexity를 평가한다.

**기본 PPL 평가**

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --ppl experiments/prompts/prefill_1024.txt \
  --kv-type f32
```

**Sliding window eviction + 고정 KV 버짓**

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --ppl experiments/prompts/prefill_1024.txt \
  --kv-type f32 \
  --eviction-policy sliding \
  --kv-budget 512 \
  --protected-prefix 4
```

**출력 예시 (stdout JSON)**

```json
{
  "ppl": 12.34,
  "total_nll": 1500.5,
  "token_count": 1023,
  "tokens_per_second": 85.2,
  "wall_time_s": 12.0,
  "eviction_count": 511,
  "config": {
    "eviction_policy": "sliding",
    "kv_budget": 512
  }
}
```

---

### 2.4 KIVI (양자화 캐시)

KV 캐시를 Q2/Q4/Q8로 양자화하여 메모리를 절약한다. eviction 정책과 상호배타적이다.

**정적 KIVI (Q2, residual=32)**

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --kivi \
  --kivi-bits 2 \
  --kivi-residual-size 32 \
  --prompt "Once upon a time" \
  -n 200
```

**KIVI Q4 (품질과 메모리 균형)**

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --kivi \
  --kivi-bits 4 \
  --kivi-residual-size 64 \
  --prompt "Once upon a time" \
  -n 200
```

**KIVI + eval-ll**

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --kivi \
  --kivi-bits 2 \
  --eval-ll \
  --eval-batch data.json \
  --greedy
```

**KIVI + PPL**

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --kivi \
  --kivi-bits 2 \
  --ppl experiments/prompts/prefill_1024.txt
```

**AWQE + AW-VOPR 품질 메트릭 (KIVI 분석용)**

KIVI 양자화 품질을 측정하는 AWQE(Attention Weight Quantization Error)와
AW-VOPR(Attention Weight Variance Over Precision Ratio) 메트릭을 활성화한다.
PPL 평가나 eval-ll과 함께 사용하여 양자화 품질을 정량화한다.

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --kivi \
  --kivi-bits 2 \
  --awqe \
  --ppl experiments/prompts/prefill_1024.txt
```

**동적 KV 양자화 (Resilience 연동)**

F16으로 시작하고 Manager 신호에 따라 Q2/Q4/Q8로 전환.

```bash
# Terminal 1: mock_manager로 KvQuantDynamic 명령 전송
./target/release/mock_manager \
  --tcp 127.0.0.1:19999 \
  --command KvQuantDynamic \
  --target-bits 2 \
  --wait-secs 30

# Terminal 2: generate (kv-dynamic-quant 활성화)
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --kv-dynamic-quant \
  --enable-resilience \
  --resilience-transport tcp:127.0.0.1:19999 \
  --prompt "Once upon a time" \
  -n 200
```

---

### 2.5 Eviction (KV 캐시 관리)

KV 캐시가 버짓을 초과할 때 오래된 토큰을 제거하여 메모리를 확보한다.

**Sliding window** — 가장 최근 N 토큰만 유지. Llama 3.2 1B에서 품질 최고.

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --eviction-policy sliding \
  --kv-budget 512 \
  --protected-prefix 4 \
  --prompt-file experiments/prompts/prefill_1024.txt \
  -n 200
```

**StreamingLLM (streaming)** — attention sink + recent window 구조.

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --eviction-policy streaming \
  --sink-size 4 \
  --kv-budget 512 \
  --prompt-file experiments/prompts/prefill_1024.txt \
  -n 200
```

**H2O** — Heavy Hitters Oracle (3-partition: prefix + heavy hitters + recent).

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --eviction-policy h2o \
  --kv-budget 512 \
  --h2o-keep-ratio 0.5 \
  --protected-prefix 4 \
  --kv-type f32 \
  --prompt-file experiments/prompts/prefill_1024.txt \
  -n 200
```

**D2O** — H2O + cosine merge compensation (evicted 토큰 정보를 retained에 병합).

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --eviction-policy d2o \
  --kv-budget 512 \
  --d2o-keep-ratio 0.75 \
  --protected-prefix 4 \
  --kv-type f32 \
  --prompt-file experiments/prompts/prefill_1024.txt \
  -n 200
```

> **주의**: H2O, H2O+, D2O는 attention score가 필요하므로 `--kv-type f32`를 권장한다.
> `--protected-prefix`를 명시하지 않으면 score 기반 정책에서 기본값 4가 적용된다.

**초기 KV 캐시 용량 및 메모리 임계값 제어**

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --eviction-policy sliding \
  --kv-budget 1024 \
  --initial-kv-capacity 256 \
  --memory-threshold-mb 512 \
  --prompt-file experiments/prompts/prefill_1024.txt \
  -n 200
```

- `--initial-kv-capacity 0` (기본값): 프롬프트 길이를 2의 거듭제곱으로 올림, 최소 128 토큰
- `--memory-threshold-mb 256` (기본값): 가용 메모리가 이 값 이하로 내려가면 eviction 트리거

**H2O raw scores (시간 정규화 없이 누적 합산)**

기본적으로 H2O/H2O+는 시간 정규화(time-normalized) 점수를 사용한다.
`--h2o-raw-scores`를 지정하면 정규화 없이 raw 누적 합산 점수를 사용한다.

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --eviction-policy h2o \
  --kv-budget 512 \
  --h2o-keep-ratio 0.5 \
  --h2o-raw-scores \
  --kv-type f32 \
  --prompt-file experiments/prompts/prefill_1024.txt \
  -n 200
```

**Sticky eviction 동작 (Manager 신호)**

Manager를 통해 eviction 요청(`KvEvictH2o`, `KvEvictSliding`, `KvMergeD2o`) 또는
KV 양자화 요청(`KvQuantDynamic`)이 수신되면 해당 상태가 **sticky**하게 유지된다.
즉, `RestoreDefaults` 커맨드가 올 때까지 매 decode step마다 해당 eviction/양자화 정책이 적용된다.
CLI의 `--eviction-policy`와 달리 Manager 신호 기반 eviction은 일회성이 아닌 지속적 제약이다.

---

### 2.6 Resilience (Manager 연동)

Manager 서비스(또는 mock_manager)로부터 런타임 명령을 받아 추론 동작을 조절한다.

**mock_manager + generate 조합 (TCP)**

```bash
# Terminal 1: mock_manager 시작 (30초 후 H2O eviction 명령 전송)
./target/release/mock_manager \
  --tcp 127.0.0.1:19999 \
  --command KvEvictH2o \
  --keep-ratio 0.5 \
  --wait-secs 30

# Terminal 2: generate (resilience 활성화)
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --enable-resilience \
  --resilience-transport tcp:127.0.0.1:19999 \
  --prompt-file experiments/prompts/prefill_1024.txt \
  -n 500
```

**Unix socket 사용 (호스트 전용)**

> **Android 참고**: Android 디바이스에서는 Unix domain socket `bind()`가 Permission denied로 실패할 수 있다.
> Android에서 Resilience 연동 시 **TCP transport를 사용**해야 한다 (`--tcp` / `--resilience-transport tcp:`).
> 위의 TCP 예제를 참조.

```bash
# Terminal 1
./target/release/mock_manager \
  --socket /tmp/llm_manager.sock \
  --command KvEvictSliding \
  --keep-ratio 0.7 \
  --wait-secs 20

# Terminal 2
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --enable-resilience \
  --resilience-transport unix:/tmp/llm_manager.sock \
  --prompt "Hello" \
  -n 300
```

**Throttle 명령 (TBT 패딩)**

```bash
./target/release/mock_manager \
  --tcp 127.0.0.1:19999 \
  --command Throttle \
  --delay-ms 100 \
  --wait-secs 15
```

**LayerSkip 명령 (레이어 건너뛰기)**

```bash
./target/release/mock_manager \
  --tcp 127.0.0.1:19999 \
  --command LayerSkip \
  --skip-ratio 0.3 \
  --wait-secs 10
```

**PrepareComputeUnit (SwitchHw 전 사전 준비)**

백엔드 전환(`SwitchHw`) 전에 대상 compute unit을 미리 워밍업한다.
Lua policy에서 미리 GPU를 준비하고 이후 SwitchHw를 발송하는 패턴:

```lua
-- 첫 번째 호출: GPU 준비 지시
table.insert(actions, {type = "switch_hw", device = "opencl"})
```

시나리오 JSON에서 SwitchHw와 조합:

```json
{
  "name": "prepare-then-switch",
  "commands": [
    {"delay_ms": 5000, "command": "SwitchHw", "device": "opencl"}
  ]
}
```

**SwitchHw 사전 할당 (`--resilience-prealloc-switch`)**

zero-alloc SwitchHw를 위해 CPU/GPU 듀얼 버퍼를 미리 할당한다.
이 플래그 없이는 throttle/suspend 지시만 가능하며 백엔드 전환이 불가하다.
활성화 시 RSS가 모델 크기만큼 증가한다 (zero-copy KV 메모리 + weight dual-access rewrap).

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  -b cpu \
  --resilience-prealloc-switch \
  --enable-resilience \
  --resilience-transport tcp:127.0.0.1:19999 \
  --prompt-file experiments/prompts/prefill_1024.txt \
  -n 500
```

**SetPartitionRatio (tensor partition 동적 제어)**

실행 중 FFN 분할 비율을 변경한다.
`ratio=0.0`이면 partition 비활성화(GPU-only), `ratio=1.0`이면 GPU 100% 할당.
Lua policy에서 `set_partition_ratio` action으로, mock_manager CLI에서 `--command SetPartitionRatio --ratio <값>`으로 사용한다.

```lua
-- Lua policy: 메모리 압박 시 partition 비율 낮추기
if mem.available / mem.total < 0.15 then
    table.insert(actions, {type = "set_partition_ratio", ratio = 0.3})
end
```

```bash
# mock_manager CLI: partition 비율 50%로 설정
./target/release/mock_manager --tcp 127.0.0.1:19999 \
  --command SetPartitionRatio --ratio 0.5 --wait-secs 10
```

**시나리오 재생 (여러 명령 순차 전송)**

시나리오 JSON 형식의 상세 설명은 [§3.3 mock_manager 사용법](#33-mock_manager-사용법) 참조.

```json
{
  "name": "eviction-then-restore",
  "commands": [
    {"delay_ms": 10000, "command": "KvEvictSliding", "keep_ratio": 0.6},
    {"delay_ms": 20000, "command": "RestoreDefaults"}
  ]
}
```

```bash
./target/release/mock_manager \
  --scenario scenario.json
```

---

### 2.7 Adaptive Cooperative Prefill (GPU 양보)

게임 등 GPU를 사용하는 앱과 LLM이 공존할 때, prefill 중 GPU 점유를 줄여 게임 FPS를 보호한다.
3가지 메커니즘을 조합한다:

| 메커니즘 | 효과 | CLI 플래그 |
|----------|------|-----------|
| **Dynamic Chunk Size** | GPU 1회 점유 시간 단축 | `--prefill-chunk-size 64` |
| **Inter-chunk Yield** | chunk 간 GPU를 게임에 양보 | `--prefill-yield-ms 16` |
| **GPU-CPU Interleave** | CPU가 일부 chunk 처리, GPU 완전 해방 | `--prefill-cpu-chunk-size 16` |

**정적 설정 (CLI)**

```bash
# chunk만 축소 (가장 단순)
./target/release/generate \
  -m models/qwen2.5-1.5b -b opencl \
  --prompt-file experiments/prompts/prefill_1024.txt \
  --prefill-chunk-size 64 \
  -n 100

# chunk + yield (chunk 사이에 16ms GPU 양보)
./target/release/generate \
  -m models/qwen2.5-1.5b -b opencl \
  --prompt-file experiments/prompts/prefill_1024.txt \
  --prefill-chunk-size 64 \
  --prefill-yield-ms 16 \
  -n 100

# chunk + yield + CPU interleave (GPU chunk 48tok → yield → CPU chunk 16tok → 반복)
./target/release/generate \
  -m models/qwen2.5-1.5b -b opencl \
  --prompt-file experiments/prompts/prefill_1024.txt \
  --prefill-chunk-size 48 \
  --prefill-yield-ms 10 \
  --prefill-cpu-chunk-size 16 \
  -n 100
```

CPU interleave 사용 시 weight가 CPU에서도 접근 가능해야 한다.
`--prefill-cpu-chunk-size > 0`이면 자동으로 zero-copy 메모리가 활성화된다.

**동적 제어 (Manager)**

Manager가 `SetPrefillPolicy` command를 보내 런타임에 조절 가능.
partial update: 지정한 필드만 변경, 나머지는 현재 값 유지.

```json
{"type": "set_prefill_policy", "chunk_size": 48, "yield_ms": 10, "cpu_chunk_size": 16}
{"type": "set_prefill_policy", "chunk_size": 64}
{"type": "set_prefill_policy", "cpu_chunk_size": 0}
```

Engine은 heartbeat에 prefill 상태를 보고한다:
- `phase`: `"idle"` / `"prefill"` / `"decode"`
- `prefill_pos`: 처리 완료 토큰 수
- `prefill_total`: 전체 prompt 토큰 수

**Lua policy 예시 (FPS 기반)**

`sys.foreground_fps(pkg)`는 SurfaceFlinger frame counter delta로 게임 FPS를 측정한다.
패키지명을 인자로 전달한다 (예: `"com.pubg.krmobile"`).

```lua
function decide(ctx)
    local actions = {}

    if ctx.engine.phase == "prefill" then
        local fps = sys.foreground_fps("com.pubg.krmobile")

        if fps == nil then
            -- 게임 미실행 또는 첫 호출: 개입 안 함
        elseif fps < 30 then
            -- 심각한 FPS 하락: chunk 축소 + yield + CPU interleave
            table.insert(actions, {
                type = "set_prefill_policy",
                chunk_size = 48, yield_ms = 10, cpu_chunk_size = 16
            })
        elseif fps < 50 then
            -- 경미한 FPS 하락: chunk 축소 + yield
            table.insert(actions, {
                type = "set_prefill_policy",
                chunk_size = 64, yield_ms = 5, cpu_chunk_size = 0
            })
        else
            -- FPS 양호: 최대 성능
            table.insert(actions, {
                type = "set_prefill_policy",
                chunk_size = 256, yield_ms = 0, cpu_chunk_size = 0
            })
        end
    end

    return actions
end
```

`sys.foreground_fps` 주의사항:
- `dumpsys SurfaceFlinger` 호출 비용 ~50-100ms. 매초 1회 정도 적절
- 첫 호출은 기준값 저장 후 `nil` 반환 (delta 계산 불가)
- UE4 게임 (PUBG 등)은 `gfxinfo`가 동작하지 않아 이 방식 사용
- 게임이 미실행이면 `nil` 반환

**수치 참고 (Qwen 2.5 1.5B, 1073 tokens prefill)**

| 구성 | Prefill 시간 | FPS drop |
|------|-------------|----------|
| GPU only (chunk=1073) | 31.6s | -44% |
| chunk=64, yield=5ms | ~35s (+10%) | ~-30% |
| chunk=48, yield=10ms, cpu=16 | ~45s (+42%) | ~-23% |
| CPU only (SwitchHw) | 84.5s (+167%) | -0% |

---

### 2.8 Tensor Partition (CPU-GPU 협업 추론)

UMA SoC(Adreno 등)에서 FFN gate/up matmul을 GPU와 CPU가 동시에 분할 실행한다.
GPU가 enqueue 후 비동기 실행되는 동안 CPU가 자신의 partition을 처리하므로 별도 스레드 없이
decode 단계(seq_len=1)의 TBT(Time Between Tokens)를 단축할 수 있다.

**전제조건**

- `--tensor-partition > 0.0` 지정 시 `--zero-copy`가 자동 활성화됨
- ARM UMA SoC (Adreno/Mali) 권장 — CPU와 GPU가 동일 DRAM 공유
- `--backend opencl` 조합에서 사용

**기본 사용법 (GPU 70%, CPU 30%)**

```bash
generate \
  --model-path /data/local/tmp/models/llama3.2-1b \
  --backend opencl \
  --tensor-partition 0.7 \
  --prompt "Hello" \
  -n 128
```

`--tensor-partition 0.7`은 FFN gate/up matmul의 out_dim을 기준으로 70%를 GPU, 30%를 CPU에
할당한다. Q4_0 128-배수 정렬을 자동으로 맞춘다.

**동적 ratio 변경 (Manager)**

Manager가 런타임에 비율을 조정할 수 있다.

```bash
# mock_manager CLI
./target/release/mock_manager --tcp 127.0.0.1:19999 \
  --command SetPartitionRatio --ratio 0.5 --wait-secs 10
```

```lua
-- Lua policy
table.insert(actions, {type = "set_partition_ratio", ratio = 0.5})
```

엔진은 다음 decode step부터 새 ratio를 적용한다.
`ratio=0.0`이면 partition을 비활성화하고 GPU-only로 복귀한다.

**주의사항**

- partition 대상: FFN gate/up matmul만 (prefill + decode 모두)
- QKV projection, output projection(`wo`), FFN down projection은 GPU-only
- `--zero-copy` 없이 실행하면 자동 활성화되므로 RSS가 model weight 크기만큼 증가

---

### 2.9 KV Offload (캐시 오프로드)

KV cache를 디스크 또는 인메모리 raw 모드로 오프로드하여 DRAM 사용량을 줄이고
긴 시퀀스를 처리한다.

**모드**

| 모드 | 설명 |
|------|------|
| `none` | 오프로드 없음 (기본값) |
| `raw` | 인메모리 raw 버퍼로 오프로드 |
| `disk` | 파일 기반 오프로드 (`--offload-path` 디렉토리 사용) |

**전제조건**

`--kv-offload`는 `--kv-layout seq`와 `--kv-type f16` 또는 `--kv-type f32` 조합이 필요하다.

**디스크 오프로드 예제**

```bash
generate \
  --model-path /data/local/tmp/models/llama3.2-1b \
  --backend opencl \
  --kv-layout seq \
  --kv-type f16 \
  --kv-offload disk \
  --offload-path /tmp/kv_cache \
  --prompt "Long context prompt..." \
  -n 256
```

`--offload-path`를 생략하면 시스템 임시 디렉토리(`/tmp/llm_rs2_kv_offload`)를 사용한다.

**prefetch 깊이 조정**

```bash
--max-prefetch-depth 8
```

기본값은 `4`다. 값이 클수록 더 많은 레이어를 미리 로드하여 latency를 숨기지만
메모리 사용량이 증가한다.

**KV 메모리 레이아웃 (`--kv-layout`)**

| 값 | 설명 |
|----|------|
| `head` | Head-major 레이아웃 (기본값). 일반 추론 권장. |
| `seq` | Seq-major 레이아웃. `--kv-offload` 사용 시 필수. |

---

### 2.10 Prompt Batch (배치 추론)

JSONL 파일로 여러 프롬프트를 순차 처리한다. 각 프롬프트마다 독립적으로 추론을 수행하고
결과를 출력한다.

**JSONL 형식**

각 줄은 다음 두 형식 중 하나다:

```jsonl
{"id": "q1", "prompt": "What is the capital of France?"}
{"id": "q2", "prompt_file": "/path/to/long_prompt.txt"}
```

- `id`: 식별자 (필수)
- `prompt`: 인라인 프롬프트 텍스트 (`prompt` 또는 `prompt_file` 중 하나 필수)
- `prompt_file`: 프롬프트를 읽어올 파일 경로
- `#`으로 시작하는 줄과 빈 줄은 무시된다

**기본 사용법**

```bash
generate \
  --model-path /data/local/tmp/models/llama3.2-1b \
  --backend opencl \
  --prompt-batch prompts.jsonl \
  -n 128
```

**루프 모드 (연속 처리)**

```bash
generate \
  --model-path /data/local/tmp/models/llama3.2-1b \
  --backend opencl \
  --prompt-batch prompts.jsonl \
  --prompt-batch-loop \
  --max-iterations 100 \
  -n 128
```

`--prompt-batch-loop`를 지정하면 파일 끝에 도달했을 때 처음부터 반복한다.
`--max-iterations 0`은 무제한 반복이다 (기본값 `0`).

**`--prompt-batch`는 `--prompt`, `--prompt-file`, `--eval-batch`와 함께 사용할 수 없다.**

---

### 2.11 Layer Skip (레이어 건너뛰기)

특정 레이어의 attention + MLP 연산을 건너뛰어 추론 속도를 높인다.
건너뛴 레이어의 입력이 그대로 출력으로 전달되므로 품질 트레이드오프가 있다.

**레이어 지정 방식 — 명시적 인덱스**

```bash
generate \
  --model-path /data/local/tmp/models/llama3.2-1b \
  --backend opencl \
  --skip-layers 1,3,5,7 \
  --prompt "Hello" \
  -n 128
```

레이어 인덱스 `0`과 마지막 레이어는 건너뛸 수 없다 (SWIFT 제약).

**레이어 지정 방식 — 비율 기반**

```bash
# 전체 레이어의 25%를 균등하게 skip
generate \
  --model-path /data/local/tmp/models/llama3.2-1b \
  --skip-ratio 0.25 \
  --prompt "Hello" \
  -n 128
```

`--skip-ratio`는 `SkipConfig::uniform_init()`으로 균등하게 레이어를 선택한다.

**레이어 중요도 분석 (`--dump-importance`)**

추론 없이 prefill만 실행하여 레이어별 중요도 테이블을 출력하고 종료한다.
어떤 레이어를 skip할지 결정하는 데 활용할 수 있다.

```bash
generate \
  --model-path /data/local/tmp/models/llama3.2-1b \
  --backend opencl \
  --dump-importance \
  --prompt "Representative prompt text here"
```

**동적 skip 변경 (Manager)**

Manager가 런타임에 skip ratio를 조정할 수 있다:

```json
{"type": "layer_skip", "skip_ratio": 0.25}
```

`skip_ratio=0.0`으로 보내면 skip을 비활성화한다.

---

### 2.12 Chat (멀티턴 REPL / 소켓 IPC)

`--chat` 플래그로 멀티턴 대화 REPL을 연다. 이전 턴의 KV 캐시가 그대로 유지되어
다음 턴의 prefill이 새 사용자 메시지만 추가로 처리한다. stdin 외에 Unix domain
socket / TCP listener를 병행해 외부 프로세스에서 프롬프트를 주입할 수 있다.

**지원 아키텍처**: Llama 3.2 Instruct (`<|begin_of_text|>`, `<|eot_id|>`),
Qwen2 (`<|im_start|>...<|im_end|>`). Gemma3는 현재 미지원 — 템플릿 준비되지 않음.

**호환 경로**: standard (기본 KVCache) / `--kivi` / `--kv-offload` / `--eviction-policy`
중 하나. 상호 배타이며, 아래와는 같이 쓸 수 없다:
`--eval-ll`, `--ppl`, `--prompt-batch`, `--eval-batch`, `--tensor-partition`,
`--cuda-graph`, `--dump-importance`, `--experiment-schedule`.

#### 기본 사용법 (stdin REPL)

```bash
generate \
  --model-path models/llama3.2-1b/llama3.2-1b-instruct-q4_0.gguf \
  --backend opencl \
  --chat \
  --system-prompt "You are a concise assistant." \
  -n 256 \
  --temperature 0.7 --top-p 0.9
```

- `-n` (= `--num-tokens`)은 **턴당** 최대 생성 토큰 수.
- `--system-prompt`는 선택. 지정 시 첫 턴 전에 prefill되어 세션 내내 KV에 유지된다.
- 시작 직후 `[Chat] Ready. Arch=..., max_seq_len=..., Commands: /exit /reset /stats /help`이 출력되고
  `>` 프롬프트가 뜬다.

**슬래시 커맨드**:

| 커맨드 | 동작 |
|--------|------|
| `/exit`, `/quit` | REPL 종료 |
| `/reset` | KV 위치·recent window·offload store 리셋 (새 세션) |
| `/stats` | 현재 KV 사용량/evicted 토큰 수 등 exec 상태 출력 |
| `/help` | 커맨드 목록 |
| (빈 줄) | 무시 (프롬프트 재출력) |

`--prompt`를 같이 주면 그 값이 **첫 번째 사용자 턴**으로 자동 투입된다
(대화 시작 자동화용).

첫 턴에서 `--max-seq-len`을 초과하는 입력+출력이 예상되면 `ensure_capacity()`가
동작해 eviction-capable exec(`--eviction-policy` 활성 시)은 공간을 회수하고,
비-eviction exec는 `context overflow` 에러로 턴을 중단한다(REPL은 종료).

#### 소켓 IPC로 테스트하기

stdin과 독립적으로 동작하는 **두 번째 입력 채널**을 띄운다. 두 채널은 동일한
REPL 루프로 머지되며, 동시에 활성화 가능하다.

**프로토콜**:
- 입력: newline(`\n`) 종료 사용자 메시지. 한 줄 = 한 턴.
- 출력: 생성된 어시스턴트 텍스트가 **바이트 스트리밍**된다 (stdout에 찍히는 것과 동일).
- 턴 경계: 턴 종료 시 `0x04` (EOT, `^D`) 1바이트가 회신 스트림 끝에 붙는다.
- 끊어진 클라이언트는 조용히 무시되며, 모델 루프는 계속 돈다.
- 각 연결은 독립 세션이 아니라 **동일 REPL에 멀티플렉싱**된다. 동시 접속은 가능하지만
  턴은 직렬 처리되고, 모든 클라이언트가 같은 KV 문맥을 공유한다.

##### (A) Unix domain socket — `--chat-socket`

```bash
generate \
  --model-path models/llama3.2-1b/llama3.2-1b-instruct-q4_0.gguf \
  --backend opencl \
  --chat \
  --chat-socket /tmp/llm_chat.sock \
  -n 256
```

엔진 시작 시 `[Chat] Listening for Unix socket input at /tmp/llm_chat.sock`가 뜬다.
기존 소켓 파일은 바인드 전에 자동 제거된다.

다른 터미널에서 `socat`이나 `ncat`으로 테스트:

```bash
# 한 줄 프롬프트 보내고 응답 스트리밍 + EOT까지 수신
printf 'What is 2+2?\n' | socat - UNIX-CONNECT:/tmp/llm_chat.sock

# 대화형: 라인을 입력할 때마다 한 턴 실행
socat - UNIX-CONNECT:/tmp/llm_chat.sock
```

`ncat` 버전:

```bash
ncat -U /tmp/llm_chat.sock
```

##### (B) TCP — `--chat-tcp`

```bash
# 로컬 루프백 (권장)
generate ... --chat --chat-tcp 127.0.0.1:7878 -n 256

# 포트 자동 할당 (커널이 고른 포트를 stderr에 로그)
generate ... --chat --chat-tcp 127.0.0.1:0 -n 256
```

엔진 시작 시 `[Chat] Listening for TCP input at 127.0.0.1:7878`가 뜬다.
**비-루프백 주소에 바인드하면** `[Chat] WARNING: ... non-loopback address ...`가
출력된다 — 네트워크 노출 시 프롬프트 주입 공격면이 생기므로 가능하면 루프백 유지.

테스트:

```bash
# 한 턴
printf 'Give me a haiku about tensors.\n' | ncat 127.0.0.1 7878

# 대화형
ncat 127.0.0.1 7878
```

##### (C) 병행 사용

`--chat-socket`과 `--chat-tcp`는 동시에 켤 수 있다. stdin도 항상 함께 활성화되어
총 3개 채널이 같은 루프에 merge된다.

```bash
generate ... --chat \
  --chat-socket /tmp/llm_chat.sock \
  --chat-tcp 127.0.0.1:7878 \
  -n 256
```

#### 스크립트로 소비하기

EOT 바이트(`\x04`)까지 읽으면 한 턴이 끝난 것으로 판정한다. Python 예:

```python
import socket

s = socket.create_connection(("127.0.0.1", 7878))
s.sendall(b"List three Rust crates for LLM inference.\n")

buf = bytearray()
while True:
    chunk = s.recv(4096)
    if not chunk:
        break
    buf += chunk
    if 0x04 in chunk:
        break

reply, _, _ = buf.partition(b"\x04")
print(reply.decode("utf-8", errors="replace"))
```

여러 턴을 이어서 돌리려면 같은 소켓에 새 라인을 보내고 다시 EOT까지 읽으면 된다.
세션 재시작이 필요하면 `/reset`을 평문 라인으로 전송한다.

#### 자주 쓰는 조합

```bash
# KIVI 양자화 KV + 소켓
generate ... --chat --kivi --kv-quant-bits 4 --chat-socket /tmp/llm_chat.sock -n 256

# Sliding window eviction + TCP
generate ... --chat \
  --eviction-policy sliding --eviction-window 1024 --protected-prefix 4 \
  --chat-tcp 127.0.0.1:7878 -n 256

# D2O + 시스템 프롬프트 + stdin
generate ... --chat \
  --eviction-policy d2o --d2o-keep-ratio 0.75 \
  --system-prompt "Respond in Korean." -n 256
```

#### 트러블슈팅

- **`--chat is incompatible with --X`**: v1 호환 경로만 허용. `--tensor-partition`,
  `--cuda-graph`, `--dump-importance`, 실험/배치 모드는 모두 제외된다.
- **`--chat: --kivi and --kv-offload are mutually exclusive`**: 둘 중 하나만.
- **소켓이 안 열림**: `/tmp/llm_chat.sock`이 다른 프로세스에 의해 잡혀 있거나
  권한 문제. 엔진은 바인드 전에 `remove_file`을 시도한다.
- **TCP 바인드 실패**: 포트 점유. `127.0.0.1:0`으로 자동 할당을 쓰고 stderr 로그로
  실제 포트 확인.
- **컨텍스트 오버플로**: 턴 중 `context overflow: ...` 출력 후 REPL 종료.
  `--eviction-policy sliding --eviction-window <N>`을 같이 붙이면 자동 회수.
- **Gemma3에서 실패**: 챗 템플릿 미구현. Llama/Qwen2만 사용.
- **Windows**: `--chat-socket`은 Unix 전용. 윈도우에서는 `--chat-tcp`만 동작.

---

### 2.13 Weight Swap (AUF / Secondary GGUF)

런타임에 decoder layer weight를 secondary 자산(다른 dtype)으로 교체하는 기능. Phase 2~Phase 6에서 단계적으로 도입되었다. **Sprint G-1 (2026-04-26) 종결** 기준으로 v0.1.1 AUF 포맷 + lm_head Q4_0 사전 변환을 지원한다.

#### 2.13.1 무엇이고 언제 쓰는가

| 입력 | 자산 | 비고 |
|------|------|------|
| Primary | `--model-path <X>.gguf` 또는 Safetensors 디렉토리 | 추론 시작 시 로드되는 모델 |
| Secondary | `--secondary-gguf <Y>.gguf` 또는 `<Y>.auf` | swap 시점에 layer가 교체될 대안 dtype 자산 |

**언제**:
- Memory 압박/정확도 trade-off 실험 (Q4 ↔ F16 swap)
- 디바이스 배포 — primary는 빠른 dtype, secondary는 정밀 dtype을 보관
- Phase 6 mixed-mode benchmark — `--force-swap-ratio`로 전 layer를 secondary로 교체하여 baseline TBT 측정

**확장자 자동 분기**: `.gguf`는 GGUF reader, `.auf`는 AUF reader (zero-copy mmap, backend variant 사전 변환). 기능적으로 동등하나 AUF는 디바이스 배포 시 GGUF 부재 환경에서도 동작.

#### 2.13.2 핵심 CLI 플래그

| 플래그 | 설명 |
|--------|------|
| `--secondary-gguf <PATH>` | secondary 자산 (`.gguf` 또는 `.auf`). 미지정 시 weight swap 경로 비활성. |
| `--force-swap-ratio <FLOAT>` | 0.0–1.0. 추론 시작 직전에 decoder layer의 X 비율을 secondary로 교체. `--secondary-gguf` 필수. |
| `--secondary-dtype <auto\|q4_0\|f16>` | v0.2 multi-quant AUF에서 swap 시 선택할 dtype. 기본 `auto` — AUF META `default_dtype` 우선, 없으면 TENSOR_INDEX first-match. 단일 dtype AUF (v0.1.x)에서는 무시됨. |
| `--swap-dir <DIR>` | swap 시 mmap에 사용할 임시 디렉토리. 미지정 시 OS 기본. |
| `--quantize-lm-head <auto\|q4_0\|none>` | 로드 시점 lm_head Q4_0 변환 (Sprint F). 기본 `auto` — `--secondary-gguf` 지정 + lm_head가 F16/F32일 때만 변환. |
| `--tokenizer-path <PATH>` | tokenizer.json 경로. 같은 디렉토리에 sibling 모델이 있을 때 명시 권장 (자동 감지가 다른 tokenizer를 잡으면 garbage 출력). |
| `--qcf-dump <PATH>` | run 종료 시 QCF_swap / NLL / swap_set / importance·noise table을 단일 JSON으로 dump. 활성화 시 자동 워밍업 prefill (importance 기반 swap 결정) 흐름 진입. |
| `--qcf-warmup-tokens <N>` | warmup prefill 길이. 기본 256. `--qcf-dump` 활성 시에만 사용. |

#### 2.13.3 GGUF secondary 시나리오 (가장 단순)

```bash
# F16 primary + Q4_0 secondary, 50% layer swap
./target/release/generate \
  -m models/llama3.2-1b-f16.gguf \
  --secondary-gguf models/llama3.2-1b-q4_0.gguf \
  --force-swap-ratio 0.5 \
  -b opencl --prompt "Hello" -n 50

# 100% swap (mixed-mode baseline)
./target/release/generate \
  -m models/llama3.2-1b-f16.gguf \
  --secondary-gguf models/llama3.2-1b-q4_0.gguf \
  --force-swap-ratio 1.0 \
  --quantize-lm-head q4_0 \
  -b opencl --prompt "Hello" -n 50
```

> Adreno GPU에서 ratio=1.0 mixed 측정 시 `--quantize-lm-head q4_0`을 명시하면 Q4 baseline 비용에 맞춘다. `auto`는 `--secondary-gguf` 지정 시 자동 활성이지만, 명시적으로 강제하는 게 진단용으로 안전.

#### 2.13.4 AUF secondary 시나리오 (디바이스 배포 권장)

GGUF 두 개를 들고 다니는 대신 AUF 한 개로 통합. **권장 경로는 `scripts/convert_to_auf.sh`** — Safetensors/GGUF 입력에서 한 번에 AUF를 만든다 (auf_tool 빌드/호출, 중간 GGUF 정리, tokenizer 자동 탐색까지 포함).

```bash
# Safetensors → AUF (한 번에)
scripts/convert_to_auf.sh \
    --input  models/llama3.2-1b/ \
    --output models/llama3.2-1b.auf \
    --variants all

# 이미 GGUF가 있으면 단계 1 건너뜀
scripts/convert_to_auf.sh \
    --input  models/llama3.2-1b-q4_0.gguf \
    --output models/llama3.2-1b.auf \
    --variants all

# v0.2 multi-quant: Q4_0 + F16 동시 보관 (default = Q4_0)
scripts/convert_to_auf.sh \
    --input  models/llama3.2-1b-q4_0.gguf \
    --output models/mixed.auf \
    --variants all \
    --dtypes q4_0,f16 \
    --default-dtype q4_0
```

세부 옵션 (`--outtype`, `--include-lm-head`, `--keep-gguf` 등)은 `scripts/convert_to_auf.sh --help` 참조. 저수준 단계별 호출이 필요하면 `auf_tool` 바이너리를 직접 사용:

```bash
# 워크스테이션에서 1회 (수동 경로)
cargo build --release -p llm_rs2 --bin auf_tool
./target/release/auf_tool build \
    --input     models/llama3.2-1b-q4_0.gguf \
    --tokenizer models/llama3.2-1b/tokenizer.json \
    --output    models/llama3.2-1b.auf \
    --variants  all \
    --include-lm-head auto

# 디바이스에 push (디바이스 backend variant만 남기고 strip)
./target/release/auf_tool strip \
    --keep META,TOKENIZER,TENSOR_INDEX,WEIGHTS_ADRENO_SOA \
    models/llama3.2-1b.auf
adb push models/llama3.2-1b.auf /data/local/tmp/

# 디바이스 추론 — AUF는 zero-copy mmap, GGUF 부재 환경에서도 동작
adb shell "LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/generate \
  -m /data/local/tmp/models/llama3.2-1b-f16.gguf \
  --secondary-gguf /data/local/tmp/llama3.2-1b.auf \
  --force-swap-ratio 1.0 \
  -b opencl --prompt 'Hello' -n 50"
```

상세 옵션 (`info`, `verify`, lm_head AOS 예외, 트러블슈팅): `docs/auf_tool_guide.md` 참조.

#### 2.13.5 QCF↔NLL 측정 (Spearman 상관관계)

`--qcf-dump`로 swap된 layer set의 **QCF_swap 예측치**와 실제 **NLL/PPL 열화**를 단일 JSON에 함께 dump 가능. 외부 harness가 ratio sweep 후 Spearman ρ를 산출하는 용도. 자동 워크플로우:

1. 첫 N개 토큰으로 warmup prefill (skip/swap 없음, ImportanceCollector inject)
2. `WeightSwapDecider`가 `importance × ε` bottom-k로 swap_set 결정 (uniform fallback 회피)
3. swap 실행 후 본 측정 (PPL / generation / eval-ll 모두 지원)
4. JSON dump (schema_version=1)

```bash
# PPL 모드 (LongBench 같은 corpus)
./target/release/generate \
  -m models/qwen2.5-1.5b-f16.gguf \
  --secondary-gguf models/qwen2.5-1.5b-mixed.auf \
  --secondary-dtype q4_0 --force-swap-ratio 0.33 \
  -b cuda --kv-type f16 --max-seq-len 4096 \
  --ppl experiments/prompts/prefill_4096.txt \
  --qcf-dump results/qwen1_5b_r0.33_ppl.json

# eval-ll 모드 (RACE-h / NIAH per-question NLL)
./target/release/generate \
  -m models/qwen2.5-1.5b-f16.gguf \
  --secondary-gguf models/qwen2.5-1.5b-mixed.auf \
  --secondary-dtype q4_0 --force-swap-ratio 0.33 \
  -b cuda --kv-type f16 --max-seq-len 4096 \
  --eval-ll --eval-batch data/race_h_300q.jsonl \
  --greedy --qcf-mode both \
  --qcf-dump results/qwen1_5b_r0.33_race_h.json
```

eval-ll 모드 JSON에는 `eval_ll_output.results[i].choice_nlls`로 per-question NLL이 포함된다 — baseline run과 비교해 ΔNLL을 계산하고 `(qcf_swap_predicted, ΔNLL)` Spearman ρ로 산출.

상세 가이드 (스키마 정의, 4 model × 4 ratio × 3 bench sweep 매트릭스, 분석 권장 통계): `docs/layer_swap_qcf_measurement.md`.

#### 2.13.6 트러블슈팅

- **`--force-swap-ratio requires --secondary-gguf`**: secondary 자산을 같이 지정해야 함.
- **AUF 추론 시 garbage 출력 (lm_head 관련)**: AUF v0.1.0 (capability_opt=0)을 사용 중일 가능성. v0.1.1 AUF로 다시 build (`--include-lm-head auto`). 또는 이미 v0.1.1이지만 `WEIGHTS_ADRENO_SOA` 안에 lm_head가 SOA로 들어 있으면 Adreno image limit (~16M texels) 때문에 GEMV가 fall-through되어 발생. Sprint G-1-F fix(INV-135 v2) 이후 모든 variant에서 lm_head는 AOS로 동봉되므로 최신 toolchain으로 재빌드/재AUF build로 해결.
- **TBT 측정 시 swap 비용 포함**: `Decode: X ms/tok` 로그는 swap 완료 이후의 정상 decode 비용. 초기 prefill 직전 `swap` 단계 비용은 stderr "WeightSwap..." 로그 참조.
- **메모리 회수 부족**: Phase 4 Sprint C-1/2/3 이후 PRIMARY drop 자동화. `--force-swap-ratio < 1.0`이면 일부 primary layer 유지가 정상.

세부 메커니즘과 실측: `arch/weight_swap.md`, `results/data/weight_swap/`.

---

## 3. Manager 가이드

### 3.1 기본 실행

`llm_manager`는 시스템 리소스(메모리, 온도, CPU/GPU 사용률)를 모니터링하여
LLM 엔진에 지시를 자동으로 전송하는 데몬이다.
**`--policy-script`가 필수**이며, Lua 스크립트로 정책을 정의한다.

**TCP으로 시작 (기본 정책 스크립트 사용)**

```bash
./target/release/llm_manager \
  --policy-script manager/scripts/policy_example.lua \
  --transport tcp:127.0.0.1:19999
```

**설정 파일 + Unix socket**

```bash
./target/release/llm_manager \
  --policy-script manager/scripts/policy_example.lua \
  --config manager/policy_config.toml \
  --transport unix:/tmp/llm.sock
```

**generate와 함께 사용**

```bash
# Terminal 1: Manager 시작
./target/release/llm_manager \
  --policy-script manager/scripts/policy_example.lua \
  --transport unix:/tmp/llm.sock

# Terminal 2: generate (resilience 활성화)
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --enable-resilience \
  --resilience-transport unix:/tmp/llm.sock \
  --prompt-file experiments/prompts/prefill_1024.txt \
  -n 500
```

---

### 3.2 Lua 정책 스크립팅

Lua 스크립트로 커스텀 정책을 정의한다. **기본 정책 모드**이며, `--policy-script`가 필수이다.

> **참고**: 내장 HierarchicalPolicy (PI Controller + Supervisory + ActionSelector)는
> deprecated되었다. `--features hierarchical`로 빌드 시에만 사용 가능.

```bash
# 기본 빌드 (lua feature 포함)
cargo build --release -p llm_manager

./target/release/llm_manager \
  --policy-script manager/scripts/policy_example.lua \
  --transport tcp:127.0.0.1:19999
```

**policy_config.toml로 초기 relief 값 + trigger 임계값 설정**

```bash
./target/release/llm_manager \
  --policy-script manager/scripts/policy_example.lua \
  --config manager/policy_config.toml \
  --transport tcp:127.0.0.1:19999
```

#### ctx 구조

Lua `decide(ctx)` 함수에 전달되는 `ctx` 테이블:

| 필드 | 내용 | 예시 |
|------|------|------|
| `ctx.engine` | Engine heartbeat 상태 | `ctx.engine.throughput`, `ctx.engine.phase` |
| `ctx.active` | 현재 활성 액션 목록 | `{"kv_evict_h2o"}` |
| `ctx.signal` | SystemSignal 원시값 | `ctx.signal.memory.available` |
| `ctx.coef` | 동적 계수 (핵심) | pressure, trigger, relief |

#### ctx.coef 상세

```lua
ctx.coef = {
    pressure = {            -- 6D 정규화 압력 (0.0~1.0)
        gpu = 0.82,         -- GPU 사용률
        cpu = 0.45,         -- CPU 사용률 (big cluster)
        memory = 0.25,      -- 메모리 사용률 (1 - available/total)
        thermal = 0.41,     -- 열 (temp_safe~temp_critical 정규화)
        latency = 0.35,     -- TBT baseline 대비 열화율
        main_app = 0.0,     -- (미래 확장용)
    },
    trigger = {             -- 경합 증거 (hysteresis 적용)
        tbt_degraded = true,   -- TBT 30%↑ 시 true, 10%↓ 시 false
        mem_low = false,       -- 메모리 80%↑ 시 true, 60%↓ 시 false
        temp_high = false,     -- 온도 70%↑ 시 true, 50%↓ 시 false
    },
    relief = {              -- per-action 학습된 relief (EWMA)
        switch_hw = { gpu=0.5, cpu=-0.3, mem=0.0, therm=0.3, lat=-0.1, qos=0.0 },
        kv_evict_h2o = { gpu=0.1, cpu=0.0, mem=0.4, therm=0.1, lat=0.0, qos=0.0 },
        throttle = { gpu=0.0, cpu=0.3, mem=0.0, therm=0.2, lat=-0.2, qos=0.0 },
        -- ... (10개 액션)
    },
}
```

- **pressure**: Rust가 SystemSignal에서 계산한 정규화 압력. GPU/CPU가 분리되어 `switch_hw`의 트레이드오프를 판단할 수 있다.
- **trigger**: 하드코딩된 임계값이 아니라 hysteresis(enter/exit 분리)로 판정. 3개 중 하나라도 true이면 경합이 발생한 것.
- **relief**: 각 액션이 얼마나 압력을 줄이는지. EWMA로 런타임에 학습되며, 초기값은 config에서 설정. 음수는 해당 도메인에 부하 증가를 의미 (예: `switch_hw`의 `cpu=-0.3`).

#### policy_example.lua (기본 정책)

```lua
function decide(ctx)
    local c = ctx.coef
    local t = c.trigger

    -- 경합 증거 없으면 개입하지 않음
    if not t.tbt_degraded and not t.mem_low and not t.temp_high then
        if #ctx.active > 0 then
            local p = c.pressure
            if p.gpu < 0.3 and p.memory < 0.3 and p.thermal < 0.3 then
                return {{type = "restore_defaults"}}
            end
        end
        return {}
    end

    -- 최고 압력 도메인 → relief 최대 액션 선택
    local p = c.pressure
    local domains = {gpu = p.gpu, cpu = p.cpu, memory = p.memory, thermal = p.thermal}
    local max_domain, max_val = nil, 0
    for k, v in pairs(domains) do
        if v > max_val then max_domain, max_val = k, v end
    end

    -- relief 테이블 키 매핑
    local domain_key = max_domain
    if domain_key == "memory" then domain_key = "mem" end
    if domain_key == "thermal" then domain_key = "therm" end

    -- latency 예산 내에서 최적 액션 선택
    local best, best_val = nil, -999
    for action, r in pairs(c.relief) do
        local rv = r[domain_key] or 0
        if rv > best_val and (r.lat or 0) >= -0.15 then
            best, best_val = action, rv
        end
    end

    if best and best_val > 0 then
        local cmd = {type = best}
        -- 액션별 기본 파라미터
        if best == "kv_evict_h2o" or best == "kv_evict_sliding" then
            cmd.keep_ratio = 0.5
        elseif best == "throttle" then cmd.delay_ms = 50
        elseif best == "switch_hw" then cmd.device = "cpu"
        end
        return {cmd}
    end
    return {}
end
```

#### 커스텀 정책 작성 팁

- **trigger만 보면 된다**: `ctx.coef.trigger`의 3개 bool로 "개입할지" 결정
- **pressure로 "어디가 문제인지" 판단**: 6D 중 가장 높은 도메인이 병목
- **relief로 "뭘 할지" 결정**: 병목 도메인의 relief가 가장 큰 액션 선택
- **음수 relief 주의**: `switch_hw`는 GPU를 해방하지만 CPU 부하 증가 (`cpu=-0.3`)
- **relief는 자동 학습**: 처음에는 config의 초기값 사용, 액션 실행 후 3초 뒤 실제 relief 측정하여 EWMA 갱신

#### EWMA Relief 학습 동작

```
1. Lua가 액션 선택 → Manager가 Engine에 전달
2. 실행 전 pressure 스냅샷 (before)
3. 3초 대기 (settling time)
4. 실행 후 pressure 스냅샷 (after)
5. observed_relief = before - after
6. relief_table[action] = 0.875 × 이전값 + 0.125 × observed
7. 다음 decide()에서 ctx.coef.relief에 업데이트된 값 노출
```

- 첫 관측 시 observed 값으로 직접 교체 (EWMA 아님)
- 복수 액션 동시 적용 시 개별 EWMA 업데이트 건너뜀
- `relief_table_path` config 설정 시 세션 종료 시 JSON으로 저장, 다음 세션에서 로드

#### sys.* 헬퍼 (보조)

`ctx.coef`와 별개로, Lua 스크립트에서 직접 sysfs를 읽을 수 있다:

| 함수 | 반환 | 용도 |
|------|------|------|
| `sys.meminfo()` | `{total, available, free}` (KB) | 메모리 상태 |
| `sys.thermal(zone)` | `float` (°C) | 온도 (zone 번호 지정) |
| `sys.gpu_busy()` | `int` (0-100) | GPU 사용률 (Adreno) |
| `sys.gpu_freq()` | `int` (Hz) | GPU 주파수 |
| `sys.cpu_freq(n)` | `int` (KHz) | CPU 주파수 (코어 번호) |
| `sys.foreground_fps(pkg)` | `float\|nil` | 포그라운드 앱 FPS |
| `sys.read(path)` | `string` | sysfs 파일 읽기 |

#### policy_config.toml 설정

```toml
[adaptation]
ewma_alpha = 0.875              # EWMA 평활 인수 (7/8, Jacobson TCP RTT)
relief_table_path = ""          # 학습 테이블 저장 경로 (빈 문자열 = 비활성화)
temp_safe_c = 35.0              # 열 정규화 안전 온도 (°C)
temp_critical_c = 50.0          # 열 정규화 임계 온도 (°C)

[adaptation.trigger]
tbt_enter = 0.30                # TBT 30% 악화 시 trigger 진입
tbt_exit = 0.10                 # TBT 10% 이내로 회복 시 trigger 해제
tbt_warmup_tokens = 20          # baseline 설정까지의 토큰 수
mem_enter = 0.80                # 메모리 80% 사용 시 진입
mem_exit = 0.60                 # 60% 미만 시 해제
temp_enter = 0.70               # 열 정규화 70% 시 진입
temp_exit = 0.50                # 50% 미만 시 해제

[adaptation.default_relief]
# [gpu, cpu, memory, thermal, latency, main_app_qos]
switch_hw = [0.5, -0.3, 0.0, 0.3, -0.1, 0.0]
kv_evict_h2o = [0.1, 0.0, 0.4, 0.1, 0.0, 0.0]
throttle = [0.0, 0.3, 0.0, 0.2, -0.2, 0.0]
layer_skip = [0.2, 0.1, 0.0, 0.1, -0.1, 0.0]
kv_quant_dynamic = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
# ... (전체 목록: manager/policy_config.toml 참조)
```

**지원 action 타입**

| action.type | 필수 파라미터 | 선택 파라미터 | 설명 |
|-------------|-------------|-------------|------|
| `kv_evict_h2o` | `keep_ratio` | — | H2O eviction |
| `kv_evict_sliding` | `keep_ratio` | — | Sliding eviction |
| `kv_streaming` | `sink_size`, `window_size` | — | StreamingLLM (attention sink + recent window) |
| `kv_merge_d2o` | `keep_ratio` | — | D2O merge eviction |
| `kv_quant_dynamic` | `target_bits` | — | KV 양자화 비트 전환 (2, 4, 8) |
| `set_target_tbt` | `target_ms` | — | TBT 목표값 설정 |
| `layer_skip` | `skip_ratio` | — | 레이어 건너뛰기 비율 |
| `throttle` | `delay_ms` | — | 토큰 생성 지연 추가 |
| `switch_hw` | `device` | — | 백엔드 전환 (`"cpu"` / `"opencl"`) |
| `suspend` | — | — | 추론 일시정지 |
| `resume` | — | — | 추론 재개 |
| `restore_defaults` | — | — | 모든 제약 해제 |
| `set_partition_ratio` | `ratio` | — | tensor partition GPU ratio (0.0~1.0) |
| `set_prefill_policy` | — | `chunk_size`, `yield_ms`, `cpu_chunk_size` | prefill 정책 (생략된 필드는 현재 값 유지) |

> **참고**: `request_qcf`는 Lua policy에서 지원되지 않는다. mock_manager `--command RequestQcf`로만 발송 가능하다.

---

### 3.3 mock_manager 사용법

프로토콜 검증과 실험에 사용하는 가짜 Manager 서버.

**빌드**

```bash
cargo build --release --bin mock_manager
```

**지원 command 전체 목록**

| Command | 필수 파라미터 | 설명 |
|---------|------------|------|
| `KvEvictSliding` | `--keep-ratio` | Sliding eviction 강제 |
| `KvEvictH2o` | `--keep-ratio` | H2O eviction 강제 |
| `KvStreaming` | `--sink-size`, `--window-size` | StreamingLLM 전환 |
| `KvMergeD2o` | `--keep-ratio` | D2O merge 강제 |
| `Throttle` | `--delay-ms` | decode 단계에 지연 추가 |
| `SetTargetTbt` | `--target-ms` | TBT 목표값(ms) 설정 |
| `SwitchHw` | `--device` | 백엔드 전환 (cpu/opencl) |
| `KvQuantDynamic` | `--target-bits` | KV 양자화 비트 전환 |
| `LayerSkip` | `--skip-ratio` | 레이어 건너뛰기 비율 |
| `SetPartitionRatio` | `--ratio` | FFN tensor partition GPU 비율 (0.0~1.0) |
| `SetPrefillPolicy` | (모두 선택) | prefill 정책 동적 변경 |
| `Suspend` | — | 추론 일시 정지 |
| `Resume` | — | 추론 재개 |
| `RestoreDefaults` | — | 모든 제약 해제 |
| `RequestQcf` | — | QCF 메트릭 요청 |

> **참고**: `PrepareComputeUnit`은 `EngineCommand` 프로토콜에 정의되어 있으나
> mock_manager CLI에서는 아직 지원되지 않는다.

**SetPrefillPolicy 파라미터**

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `--chunk-size` | usize | ❌ | prefill chunk 크기 (미지정 시 현재 값 유지) |
| `--yield-ms` | u32 | ❌ | chunk 간 GPU yield 시간 ms |
| `--cpu-chunk-size` | usize | ❌ | CPU interleave chunk 크기 |

**사용 예시**

```bash
# KV 캐시 70% 유지 (Sliding)
./target/release/mock_manager --tcp 127.0.0.1:19999 \
  --command KvEvictSliding --keep-ratio 0.7 --wait-secs 10

# StreamingLLM으로 전환 (sink=4, window=500)
./target/release/mock_manager --tcp 127.0.0.1:19999 \
  --command KvStreaming --sink-size 4 --window-size 500 --wait-secs 10

# decode 100ms 지연
./target/release/mock_manager --tcp 127.0.0.1:19999 \
  --command Throttle --delay-ms 100 --wait-secs 5

# CPU로 백엔드 전환
./target/release/mock_manager --tcp 127.0.0.1:19999 \
  --command SwitchHw --device cpu --wait-secs 5

# KV 캐시를 Q4로 양자화
./target/release/mock_manager --tcp 127.0.0.1:19999 \
  --command KvQuantDynamic --target-bits 4 --wait-secs 10

# TBT 목표 150ms 설정
./target/release/mock_manager --tcp 127.0.0.1:19999 \
  --command SetTargetTbt --target-ms 150 --wait-secs 5

# Tensor partition GPU 비율 50%로 설정
# (engine을 --tensor-partition 0.001 이상으로 기동해야 zero-copy 활성화됨)
./target/release/mock_manager --tcp 127.0.0.1:19999 \
  --command SetPartitionRatio --ratio 0.5 --wait-secs 10

# Prefill 정책 변경 (chunk + yield + CPU interleave)
./target/release/mock_manager --tcp 127.0.0.1:19999 \
  --command SetPrefillPolicy --chunk-size 48 --yield-ms 10 --cpu-chunk-size 16 --wait-secs 10

# Prefill chunk 크기만 변경 (나머지는 현재 값 유지)
./target/release/mock_manager --tcp 127.0.0.1:19999 \
  --command SetPrefillPolicy --chunk-size 64 --wait-secs 5
```

**시나리오 JSON (여러 명령 순차 전송)**

mock_manager는 `--scenario` 옵션으로 JSON 파일을 입력받아 여러 command를 시간 간격을 두고 순차 전송할 수 있다. 단일 `--command` 대신 복합 실험 시나리오를 스크립트 없이 실행할 때 유용하다.

파일 형식:

```json
{
  "name": "시나리오 이름",
  "description": "선택적 설명",
  "commands": [
    { "delay_ms": <시작부터 대기 ms>, "command": "<Command 이름>", ...파라미터... },
    ...
  ]
}
```

- `delay_ms`: 이 command 전송 전에 대기할 시간 (ms). 대기 중 수신되는 Heartbeat는 드레인된다.
- `command`: 위 command 테이블의 이름 (예: `KvEvictSliding`, `SetPartitionRatio` 등).
- 나머지 필드: 해당 command의 파라미터를 JSON 키로 직접 지정한다 (CLI flag 이름에서 `--` 제거, 하이픈을 언더스코어로).

파라미터 키 매핑:

| CLI flag | JSON 키 | 사용 command |
|----------|---------|-------------|
| `--keep-ratio` | `keep_ratio` | KvEvictSliding, KvEvictH2o, KvMergeD2o |
| `--sink-size` | `sink_size` | KvStreaming |
| `--window-size` | `window_size` | KvStreaming |
| `--delay-ms` | `delay_ms_param` | Throttle |
| `--device` | `device` | SwitchHw |
| `--target-bits` | `target_bits` | KvQuantDynamic |
| `--skip-ratio` | `skip_ratio` | LayerSkip |
| `--ratio` | `ratio` | SetPartitionRatio |
| `--chunk-size` | `chunk_size` | SetPrefillPolicy |
| `--yield-ms` | `yield_ms` | SetPrefillPolicy |
| `--cpu-chunk-size` | `cpu_chunk_size` | SetPrefillPolicy |

> **주의**: Throttle의 delay_ms 파라미터는 시나리오의 `delay_ms`(대기 시간)와 구분하기 위해 JSON 키가 `delay_ms_param`이다.

예시 1 — Eviction 후 복원:

```json
{
  "name": "eviction-then-restore",
  "commands": [
    {"delay_ms": 10000, "command": "KvEvictSliding", "keep_ratio": 0.6},
    {"delay_ms": 20000, "command": "RestoreDefaults"}
  ]
}
```

예시 2 — Tensor partition + Prefill 정책 순차 적용:

```json
{
  "name": "partition-and-prefill",
  "description": "GPU 비율 50% 설정 후 cooperative prefill 활성화",
  "commands": [
    {"delay_ms": 5000, "command": "SetPartitionRatio", "ratio": 0.5},
    {"delay_ms": 3000, "command": "SetPrefillPolicy", "chunk_size": 48, "yield_ms": 10, "cpu_chunk_size": 16},
    {"delay_ms": 10000, "command": "RestoreDefaults"}
  ]
}
```

예시 3 — SwitchHw + Throttle 복합:

```json
{
  "name": "switch-then-throttle",
  "commands": [
    {"delay_ms": 5000, "command": "SwitchHw", "device": "opencl"},
    {"delay_ms": 3000, "command": "Throttle", "delay_ms_param": 50},
    {"delay_ms": 10000, "command": "RestoreDefaults"}
  ]
}
```

실행:

```bash
./target/release/mock_manager --tcp 127.0.0.1:19999 \
  --scenario scenario.json --wait-secs 5
```

`--wait-secs`는 시나리오 재생 전 Heartbeat 수신 대기 시간이다. 시나리오 내부의 `delay_ms`와는 별개로 동작한다.

**Command별 Engine 전제조건**

각 command가 실제 효과를 내려면 `generate`를 적절한 플래그로 기동해야 한다. 전제조건이 미충족되면 command는 `Ok`를 반환하지만 효과 없이 경고 로그만 출력한다 (graceful degradation).

| Command | 필수 generate 플래그 | 없을 때 동작 |
|---------|---------------------|-------------|
| `KvEvictSliding` | `--enable-resilience` | score accumulator 미활성, eviction 효과 제한 |
| `KvEvictH2o` | `--enable-resilience` | 위와 동일 |
| `KvStreaming` | `--enable-resilience` | 위와 동일 |
| `KvMergeD2o` | `--enable-resilience` | 위와 동일 |
| `KvQuantDynamic` | `--kv-dynamic-quant` | KIVI 코드 경로 자체가 비활성, 명령 미도달 |
| `Throttle` | 없음 | 항상 작동 |
| `SetTargetTbt` | 없음 | 항상 작동 |
| `SwitchHw` | `--resilience-prealloc-switch` | 경고 로그 후 무시 |
| `LayerSkip` | 없음 | 항상 작동 |
| `SetPartitionRatio` | `--tensor-partition 0.001` 이상 (또는 `--resilience-prealloc-switch`) | weight CPU 미접근 시 "rejected" 경고 |
| `SetPrefillPolicy` | `--chunk-size`/`--yield-ms`는 없음. `--cpu-chunk-size`는 zero-copy 필요 | cpu_chunk만 "rejected" 경고, 나머지 적용 |
| `Suspend` | 없음 | 항상 작동. **주의: generate가 single-shot이라 Suspend 시 세션 종료 후 프로세스 exit** |
| `Resume` | 없음 | 프로토콜 정상이나, Suspend 후 engine이 이미 exit한 상태에서는 도달 불가 |
| `RestoreDefaults` | 없음 | 항상 작동. throttle, eviction, skip_ratio, partition_ratio, prefill 정책 모두 초기화 |
| `RequestQcf` | `--enable-resilience` (score accumulator 활성화) | accumulator 미활성 시 비어 있는 QcfEstimate 반환 |

**표준 Engine 기동 설정 (검증용)**

command 검증 시 아래 4종 설정을 조합한다. 모든 설정에 `--ignore-eos`를 추가하면 EOS 토큰에 의한 조기 종료를 방지할 수 있어 장시간 시나리오 테스트에 유리하다.

```bash
# C1: 기본 — KV eviction, Throttle, Suspend/Resume, SetTargetTbt, RequestQcf 등
./generate -m $MODEL --prompt Hello -n 500 --greedy --ignore-eos \
  --enable-resilience --resilience-transport tcp:127.0.0.1:19999

# C2: GPU + prealloc — SwitchHw 테스트용
./generate -m $MODEL --prompt Hello -n 500 --greedy --ignore-eos \
  -b opencl --resilience-prealloc-switch \
  --enable-resilience --resilience-transport tcp:127.0.0.1:19999

# C3: GPU + partition — SetPartitionRatio, SetPrefillPolicy(cpu_chunk) 테스트용
./generate -m $MODEL --prompt Hello -n 500 --greedy --ignore-eos \
  -b opencl --resilience-prealloc-switch --tensor-partition 0.001 \
  --enable-resilience --resilience-transport tcp:127.0.0.1:19999

# C4: KIVI — KvQuantDynamic 테스트용
./generate -m $MODEL --prompt Hello -n 500 --greedy --ignore-eos \
  --kv-dynamic-quant \
  --enable-resilience --resilience-transport tcp:127.0.0.1:19999
```

**동작 특성 (검증 설계 시 참고)**

- **정책 교체**: KV eviction 정책 (Sliding, H2O, Streaming)은 누적되지 않고 후속 정책이 이전을 대체한다.
- **RestoreDefaults**: 모든 active_actions를 클리어한다 — throttle, eviction, skip_ratio, partition_ratio, prefill 정책, compute/memory level 모두 초기화.
- **Suspend = 세션 종료**: `generate`는 single-shot 바이너리이므로, Suspend 수신 시 generation 종료 후 프로세스가 exit한다. 후속 command (Resume 등)는 broken pipe가 되며, 이는 정상 동작이다.
- **Heartbeat 관찰**: mock_manager는 command 전송 전후로 Heartbeat를 로그한다. `kv_util`, `active_actions`, `active_device`, `state` 필드로 command 효과를 확인할 수 있다.
- **delay_ms 산정**: 시나리오에서 command 간 `delay_ms`는 Heartbeat 1~2회를 관찰할 수 있을 만큼 (2000~3000ms) 설정한다. `--wait-secs`는 초기 연결 안정화용으로 3초가 적절하다. engine의 `-n` 토큰 수는 전체 시나리오 소요 시간보다 충분히 길어야 한다 (23 tok/s 기준: 10초 시나리오 → 최소 -n 300).

**자동 검증 스크립트**

`scripts/test_mock_commands.py`로 53개 테스트 케이스를 자동 실행할 수 있다:

```bash
# 빌드 포함 전체 실행
python scripts/test_mock_commands.py

# 빌드 건너뛰기 (바이너리가 이미 디바이스에 있을 때)
python scripts/test_mock_commands.py --skip-build

# 특정 Phase만
python scripts/test_mock_commands.py --skip-build --phase 3

# 특정 테스트만
python scripts/test_mock_commands.py --skip-build --test 3-07 -v
```

---

## 4. 실험 워크플로우 레시피

### 4.1 Action Resource Profile (TBT 기록)

동일한 QoS(TBT)에서 서로 다른 액션의 자원 효율을 비교한다.

**Baseline (액션 없음)**

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --prompt-file experiments/prompts/prefill_1024.txt \
  -n 200 \
  --greedy \
  --tbt-log results/baseline.jsonl
```

**H2O eviction (mock_manager 신호 주입)**

```bash
# Terminal 1
./target/release/mock_manager \
  --tcp 127.0.0.1:19999 \
  --command KvEvictH2o \
  --keep-ratio 0.5 \
  --wait-secs 30 &

# Terminal 2
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --kv-type f32 \
  --protected-prefix 4 \
  --enable-resilience \
  --resilience-transport tcp:127.0.0.1:19999 \
  --prompt-file experiments/prompts/prefill_1024.txt \
  -n 200 \
  --greedy \
  --tbt-log results/h2o_evict.jsonl
```

**Target TBT 정규화 비교**

동일한 TBT 제약하에서 여러 정책 비교.

```bash
for POLICY in none sliding h2o; do
  ./target/release/generate \
    -m models/qwen2.5-1.5b \
    --eviction-policy $POLICY \
    --kv-budget 512 \
    --protected-prefix 4 \
    --kv-type f32 \
    --target-tbt 150 \
    --prompt-file experiments/prompts/prefill_1024.txt \
    -n 200 \
    --greedy \
    --tbt-log results/tbt_${POLICY}.jsonl
done
```

**TBT log 포맷** (각 줄 JSONL):

```json
{"token_idx": 5, "tbt_ms": 48.2, "forward_ms": 45.1, "cache_pos": 85, "pacing_ms": 3.1}
```

---

### 4.2 NLL/QCF 비교 실험

여러 eviction 정책을 동일한 KV 버짓 비율로 비교한다.

```bash
#!/bin/bash
MODEL=models/qwen2.5-1.5b
DATA=experiments/benchmarks/data/arc_easy.0shot.json
RATIO=0.5

for POLICY in none sliding h2o d2o; do
  echo "=== Policy: $POLICY ==="
  ./target/release/generate \
    -m $MODEL \
    --eval-ll \
    --eval-batch $DATA \
    --kv-type f32 \
    --eviction-policy $POLICY \
    --kv-budget-ratio $RATIO \
    --protected-prefix 4 \
    --greedy \
    --qcf-mode both \
    > results/eval_${POLICY}_r${RATIO}.json
done

# 정확도 요약
for FILE in results/eval_*.json; do
  echo "$FILE: $(python3 -c "import json,sys; d=json.load(open('$FILE')); print(d.get('accuracy', 'N/A'))")"
done
```

---

### 4.3 벤치마크 (llm.rs vs llama.cpp)

**llm.rs 성능 측정 (TBT + TTFT)**

```bash
# CPU Q4
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --weight-dtype q4 \
  --prompt "tell me a short story" \
  -n 128 \
  --greedy \
  --experiment-output results/bench_cpu_q4.jsonl

# OpenCL Q4 (Android)
adb shell "LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/generate \
  -m /data/local/tmp/models/qwen2.5-1.5b \
  -b opencl --weight-dtype q4 \
  --prompt 'tell me a short story' \
  -n 128 --greedy \
  --experiment-output /data/local/tmp/bench_opencl_q4.jsonl"
adb pull /data/local/tmp/bench_opencl_q4.jsonl results/
```

**Python 스크립트로 자동화**

```bash
# 호스트 CPU vs Android CPU vs Android GPU 비교
python scripts/run_comparison_benchmark.py --device pixel --backend all

# Android 전체 벤치마크 스위트
python scripts/run_benchmark_suite.py --device pixel

# 빌드 없이 재실행
python scripts/run_benchmark_suite.py --device pixel --skip-build
```

**실험 결과 summary 추출**

```bash
# JSONL 마지막 줄이 summary
tail -1 results/bench_opencl_q4.jsonl | python3 -m json.tool
```

---

### 4.4 회귀 테스트

빌드 후 NLL 단조성과 KIVI 비트 단조성을 검증한다.

```bash
# 사전 조건: cargo build --release 완료
python3 scripts/test_regression.py --model-path models/qwen2.5-1.5b
```

테스트 항목:
1. **NLL 단조성**: ratio=0.75(높은 버짓)의 NLL <= ratio=0.25(낮은 버짓)의 NLL
2. **KIVI 비트 단조성**: NLL(Q2) >= NLL(Q4) >= NLL(Q8)

**스트레스 테스트 (Android)**

```bash
# 전체 6단계 스트레스 테스트
python scripts/stress_test_device.py --device pixel

# 특정 단계만 (4=정확성, 5=품질)
python scripts/stress_test_device.py --device pixel --phases 4,5
```

---

## 5. 온디바이스 배포

### 5.1 빌드 & 배포

**Android 크로스 컴파일** (run_device.py 경유 권장 — 아래 "통합 러너" 섹션 참조)

```bash
# 최초 1회: 호스트 toolchain 등록 (NDK 자동 감지 → hosts.toml 생성)
python scripts/device_registry.py bootstrap-host

# generate + manager 동시 빌드 + 푸시 (실행은 안 함)
python scripts/run_device.py -d pixel --skip-exec generate --extra-bin llm_manager

# 결과물 위치
# target/aarch64-linux-android/release/generate
# target/aarch64-linux-android/release/llm_manager
```

> cargo 직접 호출이 필요하면 `source android.source && cargo build --target aarch64-linux-android ...`도 가능하나
> `android.source`는 mac/linux 호스트 토글이 수동이라 **deprecated**. `hosts.toml` 흐름을 권장.

**Jetson 크로스 컴파일**

```bash
# Orin (Cortex-A78AE, dotprod/fhm 지원)
cargo build --release --target aarch64-unknown-linux-gnu --no-default-features --bin generate
scp target/aarch64-unknown-linux-gnu/release/generate user@jetson:~/

# Xavier (Carmel ARMv8.2, dotprod/fhm 미지원) — musl 정적 링킹
cargo build --release --target aarch64-unknown-linux-musl --no-default-features --bin generate
scp target/aarch64-unknown-linux-musl/release/generate user@jetson:~/
```

**Jetson 네이티브 빌드 (권장)**

```bash
# Jetson에 SSH 접속 후:
# Rust 미설치 시: wget -qO- https://sh.rustup.rs | sh -s -- -y && source ~/.cargo/env
cd ~/llm_rs2 && git pull
RUSTFLAGS='-C target-feature=+neon,+fp16' cargo build --release --no-default-features
# 결과물: target/release/generate
```

> Jetson에는 OpenCL이 없으므로 `--no-default-features --features cuda`로 빌드하여 CUDA 백엔드를 사용한다.

**통합 러너 사용 (권장, Android)**

```bash
# 빌드 + 배포 + 실행 한번에
python scripts/run_device.py -d pixel generate --prompt "Hello" -n 50 -b opencl

# 빌드 건너뛰기
python scripts/run_device.py -d pixel --skip-build generate --prompt "Hello" -n 50

# 드라이런 (명령 미리 확인)
python scripts/run_device.py -d pixel --dry-run generate --prompt "Hello" -n 50
```

**수동 배포** (run_device.py 미사용 시)

```bash
adb push target/aarch64-linux-android/release/generate /data/local/tmp/llm_rs2/
adb push target/aarch64-linux-android/release/mock_manager /data/local/tmp/llm_rs2/
# chmod +x 은 필요 없음 (adb push가 실행 권한 유지)
```

**Android에서 Resilience (Manager) 연동**

> **반드시 TCP transport를 사용한다.** Android의 `/data/local/tmp/` 등에서 Unix domain socket `bind()`가 Permission denied로 실패할 수 있다.
> D-Bus도 Android에서는 사용할 수 없다. TCP가 유일하게 안정적인 transport이다.

```bash
# Terminal 1: mock_manager를 TCP로 시작
adb shell "/data/local/tmp/llm_rs2/mock_manager \
  --tcp 127.0.0.1:9999 \
  --command KvEvictSliding \
  --keep-ratio 0.7 \
  --wait-secs 30"

# Terminal 2: generate를 TCP resilience로 연결
adb shell "/data/local/tmp/llm_rs2/generate \
  --model-path /data/local/tmp/models/llama3.2-1b \
  --prompt-file /data/local/tmp/llm_rs2/prompt.txt \
  -n 512 --backend cpu --threads 6 \
  --enable-resilience \
  --resilience-transport tcp:127.0.0.1:9999"
```

---

### 5.2 모델 배포

#### Safetensors 모델 (F16)

**다운로드:**

```bash
# Llama 3.2 1B (모델 접근 권한 필요)
huggingface-cli download meta-llama/Llama-3.2-1B \
  --local-dir models/llama3.2-1b

# Qwen 2.5 1.5B
huggingface-cli download Qwen/Qwen2.5-1.5B \
  --local-dir models/qwen2.5-1.5b
```

**필수 파일:**

| 파일 | 용도 |
|------|------|
| `model.safetensors` (또는 여러 개) | 가중치 (F16/BF16) |
| `tokenizer.json` | 토크나이저 |
| `config.json` | 모델 설정 |

**Android 배포:**

```bash
adb shell mkdir -p /data/local/tmp/models/qwen2.5-1.5b
adb push models/qwen2.5-1.5b/. /data/local/tmp/models/qwen2.5-1.5b/
```

#### GGUF 모델 (사전 양자화)

**다운로드 (HuggingFace에서 직접):**

```bash
# Llama 3.2 1B Q4_0 GGUF
hf download bartowski/Llama-3.2-1B-Instruct-GGUF \
  --include "Llama-3.2-1B-Instruct-Q4_0.gguf" \
  --local-dir models/llama3.2-1b-gguf
```

> HuggingFace의 GGUF 파일은 혼합 양자화를 포함할 수 있다 (예: embed이 Q6_K).
> llm.rs는 K-quant 텐서를 자동으로 F32 dequant하여 처리하지만,
> 순수 Q4_0 파일이 최적이다.

**Safetensors에서 순수 Q4_0 GGUF 생성 (권장):**

```bash
python scripts/convert_safetensors_to_gguf.py \
  models/llama3.2-1b \
  models/llama3.2-1b-q4_0.gguf
```

이 방법은 weight 텐서만 Q4_0, norm/embed은 F32로 저장하여
혼합 양자화 없는 깨끗한 GGUF를 생성한다.

**필수 파일:**

| 파일 | 용도 |
|------|------|
| `*.gguf` | 가중치 + 모델 설정 (단일 파일) |
| `tokenizer.json` | 토크나이저 (**같은 디렉토리에 배치**) |

> GGUF에는 `config.json`이 불필요 — 모델 설정이 GGUF 메타데이터에 포함되어 있다.
> 단, `tokenizer.json`은 GGUF 파일과 같은 디렉토리에 있어야 한다.

**Android 배포:**

```bash
adb push models/llama3.2-1b-q4_0.gguf /data/local/tmp/models/
adb push models/llama3.2-1b/tokenizer.json /data/local/tmp/models/
```

#### 지원 GGUF 양자화 타입

| GGUF ggml_type | 지원 | 로드 방식 |
|----------------|------|----------|
| `F32` (0) | O | 직접 참조 (zero-copy) |
| `F16` (1) | O | 직접 참조 (zero-copy) |
| `Q4_0` (2) | O | 직접 참조 (zero-copy) — 권장 |
| `Q4_1` (3) | O | 로드 시 F32 dequant |
| `Q8_0` (8) | O | 직접 참조 (zero-copy) |
| `Q4_K` (12) | △ | 로드 시 F32 dequant (fallback) |
| `Q6_K` (14) | △ | 로드 시 F32 dequant (fallback) |
| `Q2_K`, `Q3_K`, `Q5_K` 등 | X | 에러 반환 |

> Q4_0과 Q8_0이 zero-copy 로드되어 가장 효율적이다.
> K-quant (Q4_K, Q6_K)는 지원하지만 로드 시 F32 변환 오버헤드가 있다.

---

### 5.3 디바이스 관리

**devices.toml 구조** (프로젝트 루트)

```toml
[devices.pixel]
name = "Pixel Phone"

[devices.pixel.connection]
type = "adb"
serial = ""   # 빈 문자열 = 첫 연결 디바이스

[devices.pixel.build]
target = "aarch64-linux-android"
toolchain = "android-ndk"   # hosts.toml에 정의된 toolchain id
binary_dir = "target/aarch64-linux-android/release"

[devices.pixel.paths]
work_dir = "/data/local/tmp"
model_dir = "/data/local/tmp/models/qwen2.5-1.5b"
eval_dir = "/data/local/tmp/llm_rs2/eval"
lib_dir = "/data/local/tmp"
```

**adb 유용한 명령어**

```bash
# 연결된 디바이스 확인
adb devices

# 디바이스 로그 (LLM 관련만 필터)
adb logcat | grep -i llm

# 디바이스 메모리 상태
adb shell cat /proc/meminfo | head -5

# 온도 모니터링
adb shell cat /sys/class/thermal/thermal_zone*/temp

# 온디바이스 추론 실행 (OpenCL 라이브러리 경로 포함)
adb shell "LD_LIBRARY_PATH=/data/local/tmp \
  /data/local/tmp/generate \
  -m /data/local/tmp/models/qwen2.5-1.5b \
  -b opencl --weight-dtype q4 \
  --prompt 'Hello' -n 50"

# 결과 파일 pull
adb pull /data/local/tmp/results.jsonl ./results/
```

**eval 파일 배포**

```bash
python scripts/run_device.py -d pixel --deploy-eval generate --prompt "Hello" -n 10
```

---

## 6. 플래그 레퍼런스

모드별로 자주 쓰는 플래그 조합. 전체 목록은 `./target/release/generate --help`.

### 모델 & 입력

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `-m, --model-path` | `models/llama3.2-1b` | 모델 경로. 디렉토리=Safetensors, `.gguf` 파일=GGUF |
| `-p, --prompt` | `"Hello, world! I am a"` | 입력 프롬프트 |
| `--prompt-file` | — | 프롬프트 파일 (`--prompt` 오버라이드) |
| `-n, --num-tokens` | 20 | 생성 토큰 수 |
| `--weight-dtype` | `f16` | `f16` 또는 `q4` (Safetensors용. GGUF에서는 무시) |

### 백엔드 & 실행

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `-b, --backend` | `cpu` | `cpu`, `opencl`, 또는 `cuda` (`--features cuda` 빌드 필요) |
| `--switch-threshold` | 0 | CPU→GPU 자동 전환 토큰 수 (0=비활성) |
| `--threads` | 자동 | CPU 스레드 수 (0=auto) |
| `--tensor-partition` | 0.0 | FFN gate/up matmul GPU 비율 (0.0=비활성, >0 시 zero-copy 자동 활성) |
| `--gpu-attn` | false | OpenCL GPU kernel로 attention 계산 (OpenCL 전용) |
| `--no-gpu-plan` | false | decode 단계 GPU 커널 계획 비활성화 (매 토큰 forward_into fallback) |
| `--zero-copy` | false | CPU-GPU zero-copy 공유 메모리 활성화 |
| `--no-prefill-ws` | false | PrefillWorkspace 비활성화 (per-layer 할당 fallback) |
| `--use-rayon` | false | F16 matmul에 SpinPool 대신 Rayon 사용 (A/B 벤치마크용) |
| `--prefill-chunk-size` | 0 | chunked prefill 청크 크기 (0=비활성) |
| `--prefill-yield-ms` | 0 | chunk 간 GPU yield 시간 (ms, 0=비활성) |
| `--prefill-cpu-chunk-size` | 0 | CPU interleave 청크 크기 (0=비활성, >0 시 zero-copy 자동 활성) |

### 샘플링

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--temperature` | 0.8 | 0 = greedy |
| `--greedy` | false | temperature=0 강제 |
| `--top-p` | 0.9 | Nucleus sampling |
| `--top-k` | 40 | Top-K sampling |
| `--ignore-eos` | false | EOS 무시 (장시간 실험용) |

### KV 캐시

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--kv-type` | `f16` | `f32`, `f16`, `q4` (QCF/H2O에는 `f32` 권장) |
| `--kv-layout` | `head` | `head` (head-major) 또는 `seq` (seq-major, offload 시 필수) |
| `--kv-budget` | 0 | 고정 KV 버짓 (토큰, 0=무제한) |
| `--kv-budget-ratio` | 0.0 | prompt 길이 비율 버짓. `--kv-budget`과 상호배타 |
| `--max-seq-len` | 2048 | 최대 시퀀스 길이 |
| `--initial-kv-capacity` | 0 | 초기 KV 캐시 용량 토큰 수 (0=auto: prompt 길이 2의 거듭제곱, min 128) |
| `--memory-threshold-mb` | 256 | eviction 트리거 메모리 임계값 (MB) |
| `--kv-offload` | `none` | KV 캐시 오프로드 모드: `none`, `raw`, `disk` |
| `--offload-path` | 시스템 임시 디렉토리 | disk offload 파일 디렉토리 (`--kv-offload disk` 시 사용) |
| `--max-prefetch-depth` | 4 | offload KV 캐시 적응형 prefetch 깊이 (높을수록 latency 감소, 메모리 증가) |

### Eviction 정책

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--eviction-policy` | `none` | `none`, `sliding`, `streaming`, `h2o`, `h2o_plus`, `d2o` |
| `--protected-prefix` | 자동 | eviction에서 보호할 prefix 토큰 수 (score 기반→4, sliding→prompt 길이) |
| `--eviction-window` | 1024 | sliding window 크기 |
| `--eviction-target-ratio` | 0.75 | eviction 시 유지 비율 |
| `--sink-size` | 4 | StreamingLLM attention sink 토큰 수 |
| `--streaming-window` | 0 | StreamingLLM recent window (0=auto: `kv_budget - sink_size`) |

### H2O 전용

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--h2o-keep-ratio` | 0.5 | Heavy Hitter 유지 비율 |
| `--h2o-decay` | 0.0 | 중요도 점수 지수 감소 |
| `--h2o-tracked-layers` | 0 | score 추적 레이어 수 (0=전체) |
| `--h2o-raw-scores` | false | 시간 정규화 없이 raw 누적 합산 점수 사용 |

### D2O 전용

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--d2o-keep-ratio` | 0.75 | 유지 비율 (논문 기본값) |
| `--d2o-ema-alpha` | 0.5 | EMA old-threshold 가중치 |
| `--d2o-ema-beta` | 0.5 | EMA new-mean 가중치 |
| `--d2o-layer-alloc` | false | 레이어별 동적 할당 활성화 (prefill attention variance 기반) |
| `--d2o-protected-layers` | — | D2O 레이어 할당 시 보호할 레이어 인덱스 (콤마 구분, 예: `0,1,2`) |

### KIVI

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--kivi` | false | Q2 KV 압축 활성화 (eviction과 상호배타) |
| `--kivi-bits` | 2 | 양자화 비트폭 (2, 4, 8) |
| `--kivi-residual-size` | 32 | 잔여 버퍼 크기 (32의 배수) |
| `--kv-dynamic-quant` | false | 동적 KV 양자화 (Resilience 연동) |
| `--awqe` | false | AWQE + AW-VOPR 품질 메트릭 활성화 |

### Resilience

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--enable-resilience` | false | Resilience 시스템 활성화 |
| `--resilience-transport` | `dbus` | `dbus`, `unix:<path>`, `tcp:<host:port>` |
| `--resilience-prealloc-switch` | false | SwitchHw zero-alloc을 위한 CPU/GPU 듀얼 버퍼 사전 할당 (RSS +model size) |

### 평가 모드

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--eval-ll` | false | NLL 평가 모드 활성화 |
| `--eval-batch` | — | 평가 배치 JSON 파일 경로 |
| `--ppl` | — | Perplexity 평가 텍스트 파일 경로 |
| `--qcf-mode` | `attn` | `attn`, `caote`, `both` |

### 실험 & 프로파일링

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--tbt-log` | — | per-token TBT JSONL 로그 경로 |
| `--target-tbt` | 0.0 | TBT 목표값 ms (0=비활성) |
| `--experiment-output` | — | 실험 결과 JSONL 경로 |
| `--experiment-schedule` | — | 신호 주입 스케줄 JSON |
| `--experiment-eviction-ratio` | — | Resilience Evict 액션에 강제 적용할 target_ratio 오버라이드 |
| `--experiment-sample-interval` | 1 | 시스템 메트릭 샘플링 간격 (N 토큰마다, 0=비활성) |
| `--experiment-logits-topk` | 10 | 토큰당 기록할 상위 K 로짓 수 |
| `--profile` | false | per-op 타이밍 프로파일링 |
| `--profile-dir` | `results/profile` | 프로파일 출력 디렉토리 |
| `--profile-probes` | `ops,latency,scores` | 수집할 probe 목록 (ops, latency, scores, entropy, cache) |
| `--profile-interval` | 1 | score 스냅샷 간격 (1=매 step, N=N번째 step마다) |
| `--profile-per-head` | false | per-KV-head score 추적 (H2O+ 분석용) |

### 배치 & 레이어

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--prompt-batch` | — | 다중 프롬프트 JSONL 파일 경로 (`--prompt`/`--eval-batch`와 상호배타) |
| `--prompt-batch-loop` | false | 배치 완료 후 처음부터 반복 |
| `--max-iterations` | 0 | 배치 루프 최대 반복 횟수 (0=무제한) |
| `--eval-continuation` | — | 단일 태스크 NLL 평가용 continuation 텍스트 |
| `--dump-importance` | false | 레이어 중요도 테이블 출력 후 종료 (추론 없음) |
| `--skip-layers` | — | 건너뛸 레이어 인덱스 (콤마 구분, 예: `1,3,5,7`) |
| `--skip-ratio` | — | 건너뛸 레이어 비율 (0.0~1.0, `SkipConfig::uniform_init()` 사용) |

### Chat (멀티턴 REPL)

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--chat` | false | 멀티턴 REPL 진입. Llama Instruct / Qwen2 전용. standard / `--kivi` / `--kv-offload` / `--eviction-policy` 중 하나와만 호환 |
| `--system-prompt` | — | 세션 시작 시 프리필되는 system 턴 문자열 |
| `--chat-socket` | — | Unix domain socket 경로. newline-delimited 입력, 응답 바이트 스트리밍 + `0x04` EOT 종결 (Unix 전용) |
| `--chat-tcp` | — | TCP listen 주소(예: `127.0.0.1:7878`, `127.0.0.1:0`). `--chat-socket`과 동시 사용 가능 |

### Weight Swap (AUF / Secondary GGUF)

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--secondary-gguf` | — | secondary 자산 경로 (`.gguf` 또는 `.auf`). 미지정 시 weight swap 비활성 |
| `--force-swap-ratio` | — | 0.0–1.0. 추론 시작 직전 swap할 decoder layer 비율. `--secondary-gguf` 필수 |
| `--secondary-dtype` | `auto` | v0.2 multi-quant AUF에서 swap 시 선택할 dtype (`auto` / `q4_0` / `f16`). `auto`는 AUF META `default_dtype` 우선 → first-match. 단일 dtype AUF (v0.1.x)에서는 무시됨 |
| `--swap-dir` | — | swap mmap에 사용할 디렉토리 (미지정 시 OS 기본) |
| `--quantize-lm-head` | `auto` | `auto` / `q4_0` / `none`. `auto`는 `--secondary-gguf` 지정 + lm_head F16/F32일 때만 Q4_0 변환 |
| `--tokenizer-path` | 자동 감지 | tokenizer.json 경로. 디렉토리에 sibling 모델이 있으면 명시 권장 |
