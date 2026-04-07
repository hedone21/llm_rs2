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

**시나리오 재생 (여러 명령 순차 전송)**

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

## 3. Manager 가이드

### 3.1 기본 실행

`llm_manager`는 시스템 리소스(메모리, 온도, CPU/GPU 사용률)를 모니터링하여
LLM 엔진에 지시를 자동으로 전송하는 데몬이다.

**Unix socket으로 시작**

```bash
./target/release/llm_manager \
  --transport unix:/tmp/llm.sock
```

**TCP으로 시작**

```bash
./target/release/llm_manager \
  --transport tcp:127.0.0.1:19999
```

**TOML 설정 파일 사용**

```bash
./target/release/llm_manager \
  --config /etc/llm-manager/config.toml \
  --transport unix:/tmp/llm.sock
```

**generate와 함께 사용**

```bash
# Terminal 1: Manager 시작
./target/release/llm_manager --transport unix:/tmp/llm.sock

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

내장 정책 대신 Lua 스크립트로 커스텀 정책을 정의한다.
`lua` feature로 빌드해야 한다.

```bash
# Lua feature 포함 빌드
cargo build --release -p llm_manager --features lua

./target/release/llm_manager \
  --policy-script manager/scripts/policy_example.lua \
  --transport tcp:127.0.0.1:19999
```

**policy_example.lua 핵심 구조**

```lua
local ema_temp = nil

function decide(ctx)
    local actions = {}
    local temp = sys.thermal(0)
    local mem = sys.meminfo()

    -- EMA 온도
    if ema_temp == nil then ema_temp = temp
    else ema_temp = 0.2 * temp + 0.8 * ema_temp
    end

    -- 메모리 15% 미만 → H2O eviction
    if mem.total > 0 and (mem.available / mem.total) < 0.15 then
        table.insert(actions, {type = "kv_evict_h2o", keep_ratio = 0.5})
        return actions
    end

    -- 온도 42°C 초과 → TBT 200ms 제한
    if ema_temp > 42 then
        table.insert(actions, {type = "set_target_tbt", target_ms = 200})
        return actions
    end

    -- 조건 회복 → 기본값 복원
    if #ctx.active > 0 and ema_temp < 38 then
        if mem.total > 0 and (mem.available / mem.total) > 0.3 then
            table.insert(actions, {type = "restore_defaults"})
        end
    end

    return actions
end
```

**지원 action 타입**

| action.type | 필수 파라미터 | 설명 |
|-------------|-------------|------|
| `kv_evict_h2o` | `keep_ratio` | H2O eviction |
| `kv_evict_sliding` | `keep_ratio` | Sliding eviction |
| `set_target_tbt` | `target_ms` | TBT 목표값 설정 |
| `kv_quant_dynamic` | `target_bits` | KV 양자화 비트 전환 |
| `layer_skip` | `skip_ratio` | 레이어 건너뛰기 비율 |
| `restore_defaults` | — | 모든 제약 해제 |

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
| `Suspend` | — | 추론 일시 정지 |
| `Resume` | — | 추론 재개 |
| `RestoreDefaults` | — | 모든 제약 해제 |
| `RequestQcf` | — | QCF 메트릭 요청 |

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
| `--kv-budget` | 0 | 고정 KV 버짓 (토큰, 0=무제한) |
| `--kv-budget-ratio` | 0.0 | prompt 길이 비율 버짓. `--kv-budget`과 상호배타 |
| `--max-seq-len` | 2048 | 최대 시퀀스 길이 |

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

### D2O 전용

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--d2o-keep-ratio` | 0.75 | 유지 비율 (논문 기본값) |
| `--d2o-ema-alpha` | 0.5 | EMA old-threshold 가중치 |
| `--d2o-ema-beta` | 0.5 | EMA new-mean 가중치 |

### KIVI

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--kivi` | false | Q2 KV 압축 활성화 (eviction과 상호배타) |
| `--kivi-bits` | 2 | 양자화 비트폭 (2, 4, 8) |
| `--kivi-residual-size` | 32 | 잔여 버퍼 크기 (32의 배수) |
| `--kv-dynamic-quant` | false | 동적 KV 양자화 (Resilience 연동) |

### Resilience

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--enable-resilience` | false | Resilience 시스템 활성화 |
| `--resilience-transport` | `dbus` | `dbus`, `unix:<path>`, `tcp:<host:port>` |

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
| `--profile` | false | per-op 타이밍 프로파일링 |
| `--profile-dir` | `results/profile` | 프로파일 출력 디렉토리 |
| `--profile-probes` | `ops,latency,scores` | 수집할 probe 목록 |
