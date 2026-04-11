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

### 호스트 CPU 추론

```bash
cargo build --release
./target/release/generate -m models/qwen2.5-1.5b --prompt "Hello" -n 50
```

### 호스트 GPU (NVIDIA OpenCL) 추론

```bash
./target/release/generate -m models/qwen2.5-1.5b -b opencl --weight-dtype q4 --prompt "Hello" -n 50
```

### 온디바이스 (Android, Adreno GPU) 추론

```bash
# 빌드
source android.source
cargo build --release --target aarch64-linux-android

# 배포
adb push target/aarch64-linux-android/release/generate /data/local/tmp/
adb push -r models/qwen2.5-1.5b /data/local/tmp/models/qwen2.5-1.5b

# 실행
adb shell "LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/generate \
  -m /data/local/tmp/models/qwen2.5-1.5b -b opencl --weight-dtype q4 -n 50 \
  --prompt 'Hello'"
```

> **권장 조합**: Android + OpenCL + Q4 weight. Adreno GPU에서 llama.cpp CPU 대비 약 +23% 속도.

### 온디바이스 (Jetson, CPU) 추론

```bash
# 방법 1: Jetson에서 네이티브 빌드 (권장)
# Jetson에 Rust 설치 후:
RUSTFLAGS='-C target-feature=+neon,+fp16' cargo build --release --no-default-features
./target/release/generate -m models/qwen2.5-1.5b --prompt "Hello" -n 50

# 방법 2: 호스트에서 크로스 컴파일 (Orin — dotprod/fhm 지원)
cargo build --release --target aarch64-unknown-linux-gnu --no-default-features
scp target/aarch64-unknown-linux-gnu/release/generate user@jetson:~/

# 방법 3: 호스트에서 크로스 컴파일 정적 링킹 (Xavier — dotprod/fhm 미지원, glibc 2.31)
cargo build --release --target aarch64-unknown-linux-musl --no-default-features
scp target/aarch64-unknown-linux-musl/release/generate user@jetson:~/
```

> **주의**: Jetson에는 OpenCL 런타임이 없으므로 `--no-default-features`로 빌드. CUDA 백엔드는 향후 지원 예정.
> Xavier(Carmel ARMv8.2)는 dotprod/fhm 미지원 — 크로스 컴파일 시 musl 타겟 사용 또는 네이티브 빌드.

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

**Q4 weight로 메모리 절약 (CPU)**

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  --weight-dtype q4 \
  --prompt "Explain quantum computing" \
  -n 200
```

**OpenCL GPU 추론 (Android/NVIDIA)**

```bash
./target/release/generate \
  -m models/qwen2.5-1.5b \
  -b opencl \
  --weight-dtype q4 \
  --prompt "Explain quantum computing" \
  -n 200
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

**Unix socket 사용 (Android에서도 동작)**

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

**Android 크로스 컴파일**

```bash
# NDK 환경 설정 (프로젝트 루트의 android.source 사용)
source android.source

# generate 빌드
cargo build --release --target aarch64-linux-android --bin generate

# manager 관련 빌드
cargo build --release --target aarch64-linux-android \
  --bin mock_manager -p llm_manager --no-default-features

# 결과물 위치
# target/aarch64-linux-android/release/generate
# target/aarch64-linux-android/release/mock_manager
```

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

> Jetson에는 OpenCL이 없으므로 `--no-default-features`(OpenCL 제외)로 빌드한다. CUDA 백엔드는 향후 `--features cuda`로 지원 예정.

**통합 러너 사용 (권장, Android)**

```bash
# 빌드 + 배포 + 실행 한번에
python scripts/run_device.py -d pixel generate --prompt "Hello" -n 50 -b opencl

# 빌드 건너뛰기
python scripts/run_device.py -d pixel --skip-build generate --prompt "Hello" -n 50

# 드라이런 (명령 미리 확인)
python scripts/run_device.py -d pixel --dry-run generate --prompt "Hello" -n 50
```

**수동 배포**

```bash
adb push target/aarch64-linux-android/release/generate /data/local/tmp/
chmod +x 은 필요 없음 (adb push가 실행 권한 유지)
```

---

### 5.2 모델 배포

**HuggingFace에서 모델 다운로드**

```bash
# Llama 3.2 1B (모델 접근 권한 필요)
huggingface-cli download meta-llama/Llama-3.2-1B \
  --local-dir models/llama3.2-1b

# Qwen 2.5 1.5B
huggingface-cli download Qwen/Qwen2.5-1.5B \
  --local-dir models/qwen2.5-1.5b
```

**Android로 모델 배포**

```bash
# 디렉토리 생성
adb shell mkdir -p /data/local/tmp/models/qwen2.5-1.5b

# 모델 파일 전송 (수 GB, 시간 소요)
adb push models/qwen2.5-1.5b/. /data/local/tmp/models/qwen2.5-1.5b/

# 전송 확인
adb shell ls -la /data/local/tmp/models/qwen2.5-1.5b/
```

**필수 모델 파일**

| 파일 | 용도 |
|------|------|
| `model.safetensors` (또는 여러 개) | 가중치 |
| `tokenizer.json` | 토크나이저 |
| `config.json` | 모델 설정 |

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
env_file = "android.source"
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
| `-m, --model-path` | `models/llama3.2-1b` | HuggingFace Safetensors 모델 경로 |
| `-p, --prompt` | `"Hello, world! I am a"` | 입력 프롬프트 |
| `--prompt-file` | — | 프롬프트 파일 (`--prompt` 오버라이드) |
| `-n, --num-tokens` | 20 | 생성 토큰 수 |
| `--weight-dtype` | `f16` | `f16` 또는 `q4` |

### 백엔드 & 실행

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `-b, --backend` | `cpu` | `cpu` 또는 `opencl` (GPU 보조 자동 초기화) |
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
