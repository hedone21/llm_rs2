# auf-tool — User Guide

> AUF (Argus Unified Format) 자산을 만들고 검사하고 수정하는 CLI 도구. Weight Swap Phase 3.7b에서 도입, **Sprint G-1 (2026-04-26)에서 v0.1.1 + lm_head Q4_0 사전 변환 추가**.
>
> 정식 spec: `spec/33-engine-data.md` §3.22, `spec/32-engine-algorithms.md` §3.12.17.
> 컴포넌트: `arch/auf_format.md`. Changelog: `docs/auf_format_changelog.md`.
>
> **바이너리 이름**: `auf_tool` (cargo target name; underscore). 빌드: `cargo build --release -p llm_rs2 --bin auf_tool`.

---

## 1. AUF 개요 (사용자 관점)

**AUF는 GGUF의 self-contained 파생 자산**이다. GGUF 원본 + 모든 backend variant (Adreno SOA, CUDA AOS, CPU AOS)별로 사전 변환된 weight payload를 단일 `.auf` 파일에 보관한다. 디바이스 배포 시 GGUF가 부재해도 Engine이 동작한다.

**왜 필요한가**:
- Adreno noshuffle Q4_0 GEMV kernel은 SOA layout(`q_buf` + `d_buf` 분리, image2d 정렬)을 입력으로 가정한다. GGUF는 AOS 원본 byte를 보관하므로 swap 시점에 변환이 필요하며, 변환 누락 시 fallback kernel로 전환되어 정확도 임계 미달 가능.
- AUF는 빌드 시점에 한 번만 변환하고 디바이스에서는 zero-copy mmap으로 직접 사용.

**언제 쓰는가**:
- `--secondary-gguf <path>.auf`로 Engine에 전달. `.gguf`와 `.auf`는 확장자로 자동 분기 (플래그 이름은 GGUF 시절의 호환을 위해 유지).
- Phase 3.7b 시점에는 secondary weight 자산용. v0.1.1부터 lm_head Q4_0 entry도 보관. 향후 primary 모델 전체를 AUF로 대체 가능 (v0.2+).

**v0.1.1 (Sprint G-1) 신기능 요약**:
- **lm_head Q4_0 entry**: 별도 capability bit (`LM_HEAD_PRECOMPUTED_Q4_0` = bit 2)로 표시. F16 GGUF에서도 build 시 사전 양자화하여 동봉 → swap 시 dequant→quant 비용 제거.
- **lm_head AOS 강제** (INV-135 v2): `WEIGHTS_ADRENO_SOA` 안에서도 lm_head는 SOA 변환을 건너뛰고 AOS bytes를 그대로 동봉. 이유: Llama 3.2 1B의 lm_head q_buf가 32M texels로 OpenCL `CL_DEVICE_IMAGE_MAX_BUFFER_SIZE` (~16M)를 초과하여 noshuffle SOA GEMV가 silent fall-through되며 garbage 출력 발생. AOS는 fall-through 없이 표준 GEMV 경로 사용.
- **`--include-lm-head` 플래그**: `auto` (기본) / `on` / `off`. v0.1.0 byte-level 호환은 `off`로 생성.

---

## 2. 명령어 레퍼런스

### 2.1 `auf_tool build`

GGUF + tokenizer.json에서 AUF를 새로 생성한다.

```sh
auf_tool build \
    --input     <path>.gguf \
    --tokenizer <path>/tokenizer.json \
    --output    <path>.auf \
    --variants  <list> \
    [--include-lm-head <auto|on|off>] \
    [--created-by "<custom string>"] \
    [--quiet]
```

| 플래그 | 필수 | 설명 |
|--------|------|------|
| `--input` | ✓ | GGUF 원본 파일. |
| `--tokenizer` | ✓ | tokenizer.json 경로. AUF에 TOKENIZER section으로 동봉 (디바이스에서 별도 tokenizer 파일 불필요). |
| `--output` | ✓ | AUF 출력 경로. atomic rename으로 부분 write 노출 금지. |
| `--variants` | ✓ | 포함할 backend variant. comma-separated 또는 `all`. 예: `adreno_soa,cpu_aos`, `WEIGHTS_ADRENO_SOA,WEIGHTS_CPU_AOS`, `all`. (lower / upper / FULL_TAG 모두 허용) |
| `--dtypes` | — | v0.2 multi-quant: 동봉할 dtype 목록 (comma-separated). 예: `q4_0,f16`. 미지정 시 source GGUF의 dtype 1개(single-dtype, v0.1.x 호환). 2개 이상 지정 시 `capability_optional` bit 3 (`MULTI_DTYPE_VARIANTS`) 자동 set + `format_minor=2` 자동. |
| `--default-dtype` | — | v0.2 multi-quant 모드에서 `META.default_dtype` 명시. `--dtypes`에 포함된 값이어야 한다. 미지정 시 `--dtypes` 첫 번째 값. |
| `--include-lm-head` | — | v0.1.1 lm_head Q4_0 entry 정책. `auto` (기본) / `on` / `off`. 아래 §2.1.1 참조. |
| `--created-by` | — | 헤더 `created_by` (32B UTF-8). 기본 `"llm_rs2 auf-tool v{VERSION}"`. |
| `--quiet` | — | 진행 로그 출력 억제. |

**동작**:
1. GGUF parse → metadata, layer tensor 추출.
2. tokenizer.json 로드.
3. Hybrid `source_hash` 계산 (`sha256(size || mtime || head_8MB || tail_8MB)`).
4. lm_head 결정 (v0.1.1): `select_lm_head_source()` — tied embedding (`token_embd.weight`)이면 그것을 quantize, 아니면 `output.weight` 사용. `--include-lm-head off`면 건너뜀.
5. 각 variant별로 weight 변환:
   - `WEIGHTS_ADRENO_SOA`: Q/K permute → SOA 분리 → q_img alignment.
     - **예외 (INV-135 v2)**: lm_head는 SOA 변환을 건너뛰고 AOS bytes를 그대로 동봉.
   - `WEIGHTS_CUDA_AOS`: Q/K permute → 128B align padding.
   - `WEIGHTS_CPU_AOS`: Q/K permute → 64B align padding.
6. META JSON / TOKENIZER blob / TENSOR_INDEX 직렬화.
7. Section layout 결정 (cursor 기반, 64KB align for `WEIGHTS_*`).
8. Header finalize → tempfile에 write → atomic rename.

**출력**: 성공 시 stdout에 요약 (총 크기, capability flag, section 수, 각 section size).

#### 2.1.1 `--include-lm-head` 모드

| 모드 | 동작 |
|------|------|
| `auto` (기본) | GGUF lm_head dtype != Q4_0이면 quantize하여 entry 추가. 이미 Q4_0이면 AOS bytes 그대로 동봉. capability_opt bit 2 = 1, format_patch = 1. |
| `on` | 강제 quantize (이미 Q4_0이면 AOS bytes 동봉). 결과는 `auto`와 동일하지만 명시적 의도 표현. |
| `off` | lm_head Q4_0 entry 미포함. v0.1.0 byte-level 호환 출력 (capability_opt bit 2 = 0, format_patch = 0). |

**예시**:

```sh
# 모든 backend variant 포함 + auto lm_head (워크스테이션 빌드, v0.1.1)
auf_tool build \
    --input     models/llama-3.2-1b-q4_0.gguf \
    --tokenizer models/llama-3.2-1b/tokenizer.json \
    --output    models/llama-3.2-1b.auf \
    --variants  all

# F16 GGUF에서 강제 lm_head 사전 양자화 (Sprint F/G-1 권장)
auf_tool build \
    --input     models/llama-3.2-1b-f16.gguf \
    --tokenizer models/llama-3.2-1b/tokenizer.json \
    --output    models/llama-3.2-1b-f16-with-lmhead.auf \
    --variants  WEIGHTS_ADRENO_SOA \
    --include-lm-head on

# v0.1.0 호환 출력 (구 reader 지원이 필요한 경우)
auf_tool build \
    --input     models/llama-3.2-1b-q4_0.gguf \
    --tokenizer models/llama-3.2-1b/tokenizer.json \
    --output    models/llama-3.2-1b-v010.auf \
    --variants  all \
    --include-lm-head off
```

---

### 2.2 `auf_tool info`

AUF 파일의 헤더, section 목록, capability flag를 출력한다.

```sh
auf_tool info <path>.auf
```

**출력 예시 (v0.1.1)**:

```
File: models/llama-3.2-1b.auf
Size: 849297408 bytes (810 MiB)

Header:
  magic            : "ARGUS_W\0"
  format           : v0.1.1 (experimental)
  created_by       : "llm_rs2 auf-tool v0.5.0"
  source_hash      : 3f8a91b2c4d5e6f7... (hybrid)
  source_size      : 705689600 bytes
  source_mtime     : 2026-04-25 14:32:11 UTC
  capability_req   : 0x0000000000000000
  capability_opt   : 0x0000000000000004  (LM_HEAD_PRECOMPUTED_Q4_0)
  section_count    : 4
  payload_start    : 0x10000 (65536)

Sections:
  Tag                       Offset      Size           Flags    Version
  META                      0x10000     2048           REQUIRED 1
  TOKENIZER                 0x10800     6291456        REQUIRED 1
  TENSOR_INDEX              0x620800    81920          REQUIRED 1
  WEIGHTS_ADRENO_SOA        0x634800    842452992      STRIPPABLE 1

Variants present: ADRENO_SOA
LM head entry    : present (Q4_0, AOS, ~263 MiB)
```

> v0.1.0 AUF는 `format`에 `v0.1.0`, `capability_opt`에 `0x...0` (bit 2 = 0)으로 출력되며 LM head entry 줄이 없다.

**출력 예시 (v0.2 mixed.auf, multi-quant Q4_0+F16)**:

```
File: models/mixed.auf
Size: 1554677760 bytes (1.45 GiB)

Header:
  magic            : "ARGUS_W\0"
  format           : v0.2.0 (experimental)
  created_by       : "llm_rs2 auf-tool v0.5.0"
  source_hash      : 3f8a91b2c4d5e6f7... (hybrid)
  source_size      : 705689600 bytes
  source_mtime     : 2026-04-25 14:32:11 UTC
  capability_req   : 0x0000000000000000
  capability_opt   : 0x000000000000000C  (LM_HEAD_PRECOMPUTED_Q4_0 | MULTI_DTYPE_VARIANTS)
  section_count    : 4
  payload_start    : 0x10000 (65536)

META:
  default_dtype    : Q4_0

Sections:
  Tag                       Offset      Size           Flags    Version
  META                      0x10000     2048           REQUIRED 1
  TOKENIZER                 0x10800     6291456        REQUIRED 1
  TENSOR_INDEX              0x620800    163840         REQUIRED 1  (2x entries per tensor)
  WEIGHTS_ADRENO_SOA        0x644800    1547407360     STRIPPABLE 1

Variants present: ADRENO_SOA
LM head entry    : present (Q4_0 + F16, AOS, ~263 MiB Q4_0 / ~526 MiB F16)
Dtypes available : Q4_0 (default), F16
```

> `capability_opt=0xC` = bit 2 (`LM_HEAD_PRECOMPUTED_Q4_0`) + bit 3 (`MULTI_DTYPE_VARIANTS`). v0.1.x reader는 bit 3을 unknown으로 처리하지만 reject하지 않고 first-match(= default_dtype Q4_0)로 안전하게 진행한다 (INV-139).

---

### 2.3 `auf_tool strip`

특정 section을 제거하여 파일 크기를 줄인다. **단순 truncation이 아니라 rewrite** (section table offset 재계산).

```sh
auf_tool strip --keep <list> [--no-backup] <path>.auf
```

| 플래그 | 설명 |
|--------|------|
| `--keep` | 유지할 section tag (comma-separated). lower / upper / FULL_TAG 모두 허용 (예: `adreno_soa`, `WEIGHTS_ADRENO_SOA`). required section (META/TOKENIZER/TENSOR_INDEX)은 자동 보존. |
| `--no-backup` | 기본은 `<path>.auf.bak`을 자동 생성. 이 플래그 지정 시 백업 생략. |

**동작**:
1. 기존 AUF 읽기 (backend 검증 우회).
2. `--keep` 검증: required section 누락 또는 strippable 비트 없는 section 제거 시도 시 reject.
3. 백업 (`--no-backup` 미지정 시).
4. 유지 section만으로 새 AUF 빌드 (writer 경로 재사용, `source_hash` / `created_by` 보존).
5. `capability_optional`에서 제거된 variant bit 클리어 (v0.1에서는 N/A — 향후 capability 추가 시 활성).
6. atomic rename으로 in-place 교체.

**예시**:

```sh
# Adreno만 남기고 strip (mobile 배포)
auf_tool strip \
    --keep WEIGHTS_ADRENO_SOA \
    models/llama-3.2-1b.auf

# 결과: 2 GB → ~810 MB (v0.1.1, lm_head entry 포함)
# 백업: models/llama-3.2-1b.auf.bak

# 짧은 형태도 동일
auf_tool strip --keep adreno_soa models/llama-3.2-1b.auf

# 백업 생략
auf_tool strip \
    --keep cpu_aos \
    --no-backup \
    models/server.auf
```

---

### 2.4 `auf_tool verify`

AUF 무결성을 검증한다. 선택적으로 source GGUF와 hash 비교.

```sh
auf_tool verify [--source <path>.gguf] <path>.auf
```

**검증 항목** (모두 INV-132/133/134 매핑):

| 검증 | INV |
|------|-----|
| Magic byte | INV-132 |
| `format_major` reader 한도 | INV-132 |
| `capability_required` 알려진 비트만 | INV-132 |
| Required section 존재 (META/TOKENIZER/TENSOR_INDEX) | INV-133 |
| Section offset/size 무결성 + overlap 금지 + tag unique | INV-134 |

**`--source` 옵션**: 제공 시 hybrid source_hash 재계산하여 헤더 값과 비교. 불일치는 **경고**(reject 아님).

**예시**:

```sh
# 무결성만 검증
auf_tool verify models/llama-3.2-1b.auf

# source GGUF와 일치 검증
auf_tool verify \
    --source models/llama-3.2-1b-q4_0.gguf \
    models/llama-3.2-1b.auf
```

---

### 2.5 `auf_tool repack` (미구현)

Stripped AUF에 누락된 variant section을 source GGUF로부터 재생성하여 추가.

> v0.1.x에서는 **미구현**. 사용자가 strip 전 `.auf.bak`을 보관하면 동등한 결과를 얻을 수 있다. v0.2 후보.

향후 사용 예 (참고):

```sh
auf_tool repack \
    --input  models/server.auf \
    --source models/llama-3.2-1b-q4_0.gguf \
    --add WEIGHTS_ADRENO_SOA \
    --output models/server-with-adreno.auf
```

이 명령은 source GGUF의 hybrid hash가 AUF의 `source_hash`와 일치할 때만 진행된다. 불일치 시 fail-fast (의도된 동작 — 다른 모델로 augment 방지).

---

## 3. 운영 시나리오

### 3.1 워크스테이션 빌드 → Galaxy S25 배포

> 빠른 경로: `scripts/convert_to_auf.sh --input <model-dir-or-gguf> --output <path>.auf`. Safetensors/GGUF 양쪽 입력을 자동 분기하며 tokenizer 자동 탐색 + auf_tool 빌드까지 처리.

```sh
# 워크스테이션 (v0.1.1 기본, 저수준 직접 호출)
auf_tool build \
    --input     models/llama-3.2-1b-q4_0.gguf \
    --tokenizer models/llama-3.2-1b/tokenizer.json \
    --output    models/llama-3.2-1b.auf \
    --variants  all

# 디바이스 변종만 strip
auf_tool strip --keep adreno_soa models/llama-3.2-1b.auf
# 결과: ~810 MB (원본 ~2 GB; v0.1.1 lm_head entry 포함)

# 디바이스 push
adb push models/llama-3.2-1b.auf /data/local/tmp/
adb push models/llama-3.2-1b-f16.gguf /data/local/tmp/  # primary (선택)

# Engine 실행 (AUF를 secondary로 지정 — `--secondary-gguf` 플래그가 .auf도 받음)
adb shell '/data/local/tmp/generate \
    -m /data/local/tmp/llama-3.2-1b-f16.gguf \
    --secondary-gguf /data/local/tmp/llama-3.2-1b.auf \
    --force-swap-ratio 1.0 \
    -b opencl --prompt "Hello" -n 50'
```

### 3.2 Multi-target CI/CD

```sh
# 1회 build, 여러 변종 strip
auf_tool build \
    --input     model.gguf \
    --tokenizer model.tokenizer.json \
    --output    model.auf \
    --variants  all

cp model.auf model-mobile.auf
auf_tool strip --keep adreno_soa --no-backup model-mobile.auf

cp model.auf model-server.auf
auf_tool strip --keep cuda_aos --no-backup model-server.auf

cp model.auf model-cpu.auf
auf_tool strip --keep cpu_aos --no-backup model-cpu.auf
```

### 3.3 F16 GGUF에서 lm_head 사전 양자화 AUF 만들기 (Sprint G-1)

ratio=1.0 mixed-mode 측정에서 Q4 baseline에 맞추기 위해 lm_head를 사전 Q4_0으로 양자화하여 AUF에 동봉:

```sh
auf_tool build \
    --input     models/llama-3.2-1b-f16.gguf \
    --tokenizer models/llama-3.2-1b/tokenizer.json \
    --output    models/llama-3.2-1b-f16-with-lmhead.auf \
    --variants  WEIGHTS_ADRENO_SOA \
    --include-lm-head on

# 디바이스에서 secondary로 지정 — 추가 `--quantize-lm-head` 불필요 (AUF에 이미 Q4_0 lm_head 존재)
generate \
    -m models/llama-3.2-1b-f16.gguf \
    --secondary-gguf models/llama-3.2-1b-f16-with-lmhead.auf \
    --force-swap-ratio 1.0 \
    -b opencl
```

### 3.4 v0.2 Multi-quant AUF 만들기 (Q4_0 + F16 동시 보관)

동적으로 dtype을 바꾸려면 v0.2 multi-quant AUF를 사용한다. 단일 파일에 Q4_0·F16 두 가지 dtype을 보관하고 Engine 측에서 `--secondary-dtype`으로 런타임 선택이 가능하다.

```sh
# Q4_0 GGUF 기반으로 Q4_0 + F16 동시 동봉 (빠른 경로: convert_to_auf.sh)
scripts/convert_to_auf.sh \
    --input        models/llama-3.2-1b-q4_0.gguf \
    --output       models/mixed.auf \
    --variants     all \
    --dtypes       q4_0,f16 \
    --default-dtype q4_0

# 저수준 직접 호출 (auf_tool)
auf_tool build \
    --input        models/llama-3.2-1b-q4_0.gguf \
    --tokenizer    models/llama-3.2-1b/tokenizer.json \
    --output       models/mixed.auf \
    --variants     all \
    --dtypes       q4_0,f16 \
    --default-dtype q4_0
# → capability_opt bit 3 = 1 (MULTI_DTYPE_VARIANTS), format_minor = 2 자동 설정

# 검증
auf_tool info models/mixed.auf
# Dtypes available: Q4_0 (default), F16 표시 확인

# Engine에서 기본 dtype (Q4_0) 사용
generate -m primary.gguf --secondary-gguf models/mixed.auf --force-swap-ratio 1.0 -b opencl

# Engine에서 F16 dtype 명시 선택 (--secondary-dtype)
generate -m primary.gguf --secondary-gguf models/mixed.auf \
    --secondary-dtype f16 --force-swap-ratio 1.0 -b opencl
```

> v0.1.x reader가 mixed.auf를 읽으면 first-match = default_dtype(Q4_0)로 자동 진행한다. capability bit 3은 optional이므로 reject 없이 단일 dtype 모드로 안전하게 동작한다 (INV-139).

### 3.5 모델 업데이트

GGUF가 갱신된 경우, AUF도 처음부터 다시 build. v0.1.x에서는 incremental update 미지원 (가능하지만 source_hash 변경으로 의미 모호).

---

## 4. 트러블슈팅

### 4.1 source_hash mismatch

**증상**:
```
auf_tool verify --source model.gguf model.auf

Warning: AUF source_hash differs from given GGUF
  AUF says : 3f8a91b2c4d5e6f7...
  Computed : a1b2c3d4e5f60718...
```

**원인 후보**:
1. GGUF가 AUF build 이후 수정됨 (mtime 변경, 또는 head/tail 8MB 영역 변경).
2. AUF가 다른 GGUF로부터 build됨.
3. AUF가 strip되면서 `source_hash`는 보존됨 (정상).

**조치**:
- 1번/2번 경우: 원본 GGUF로 다시 build (`auf_tool build`).
- 3번 경우: strip은 source_hash를 보존하므로 정상. `auf_tool info` 확인.
- 무시하고 사용 가능 (verify의 source_hash 비교는 informational, reject 아님 — INV-132).

### 4.2 missing WEIGHTS_* section

**증상** (Engine 실행 시):
```
Error: AUF does not contain WEIGHTS_ADRENO_SOA section.
       Run 'auf_tool repack --add WEIGHTS_ADRENO_SOA' to add it from source GGUF.
```

**원인**: 디바이스의 backend가 요구하는 `WEIGHTS_*` section이 strip되었거나 처음부터 build되지 않음.

**조치**:
- v0.1.x (repack 미구현): `.auf.bak`이 있으면 복원 후 재strip.
  ```sh
  cp model.auf.bak model.auf
  auf_tool strip --keep adreno_soa model.auf
  ```
- backup이 없으면 source GGUF에서 다시 build:
  ```sh
  auf_tool build --input model.gguf --tokenizer tokenizer.json --output model.auf --variants all
  auf_tool strip --keep adreno_soa model.auf
  ```
- v0.2+ repack 가능 후: `auf_tool repack --add WEIGHTS_ADRENO_SOA --source model.gguf model.auf`.

### 4.3 lm_head garbage 출력 (v0.1.0 또는 옛 toolchain)

**증상**: AUF 사용 추론 시 토큰 출력이 무작위 글자/이모지/Unicode (예: "θα364.Edit-प AssemblyProduct...")로 나오며 정상 텍스트가 안 나옴. 같은 모델의 GGUF secondary로 바꾸면 정상.

**원인 후보 1 — v0.1.0 옛 AUF 사용**:
- `auf_tool info` 출력에 `format: v0.1.0` + LM head entry 줄 부재. Engine이 lm_head를 SOA로 변환하여 사용했으나, Llama 3.2 1B의 lm_head q_buf (32M texels)가 OpenCL `CL_DEVICE_IMAGE_MAX_BUFFER_SIZE` (~16M)를 초과 → noshuffle SOA GEMV가 silent fall-through되어 SOA d_buf를 AOS로 잘못 해석.

**원인 후보 2 — Sprint G-1-F 이전 toolchain으로 만든 v0.1.1 AUF**:
- `WEIGHTS_ADRENO_SOA` 안에 lm_head가 SOA로 동봉됨 (INV-135 v1). 같은 image limit 문제 발생.

**조치**: 최신 toolchain (Sprint G-1-F 이후, INV-135 v2 적용)으로 AUF를 다시 build:
```sh
git pull && cargo build --release -p llm_rs2 --bin auf_tool
auf_tool build \
    --input     model.gguf \
    --tokenizer tokenizer.json \
    --output    model.auf \
    --variants  all
# v0.1.1 AUF: lm_head는 모든 variant에서 AOS로 동봉됨 → image limit 회피
```

**검증**: `auf_tool info` 출력에 `format: v0.1.1` + `LM head entry: present (Q4_0, AOS, ...)` 표시.

### 4.4 format_major mismatch

**증상**:
```
Error: AUF format_major=2 but reader supports up to 1. Update llm_rs2.
```

**원인**: AUF가 새 버전 도구로 만들어졌으나 reader는 구버전.

**조치**:
- llm_rs2 / auf-tool 업데이트.
- 또는 구버전 도구로 AUF를 다시 build (가능하다면).
- v0.x 기간(format_major=0)에는 minor 버전 변경도 reject 가능 (`AUF format_major=0 is experimental` 경고 후 진행 권장이지만 reader 구현이 거부 가능).

### 4.5 corrupt section table

**증상**:
```
Error: Section TOKENIZER overlaps with TENSOR_INDEX (offset/size invalid)
```
또는
```
Error: Section WEIGHTS_ADRENO_SOA exceeds file size
```

**원인**: 파일 손상 또는 부분 write (atomic rename 실패 시 발생 가능 — writer 버그).

**조치**:
- 백업 또는 source GGUF에서 다시 build.
- writer 버그 의심 시: 재현 가능한 케이스를 `engine/tests/spec/test_inv_134_auf_section_integrity.rs`에 추가하고 issue 보고.

### 4.6 unknown capability bit

**증상**:
```
Error: AUF requires capability bit 5 (zstd compression) but reader does not support it
```

**원인**: 새 capability flag가 추가된 AUF를 구버전 reader가 만남.

**조치**:
- llm_rs2 업데이트.
- 또는 zstd 미사용으로 AUF 재build (auf-tool 옵션 변경).

---

## 5. 자주 묻는 질문

**Q: AUF와 GGUF 둘 다 보관해야 하는가?**

A: Mode B (v0.1 채택)에서는 AUF만으로 동작 가능. GGUF는 워크스테이션의 source 자산으로만 보관. 배포/모바일 디바이스에서는 AUF만 필요.

**Q: AUF가 stripped 상태인지 어떻게 아는가?**

A: `auf_tool info`로 `Variants present` 줄 확인. 모든 variant가 빠져 있으면 strip된 것이 아니라 build 시점에 일부만 포함된 것일 수 있다.

**Q: v0.1.0 AUF를 v0.1.1 reader가 읽을 수 있나? 반대는?**

A: v0.1.1 reader는 v0.1.0 AUF를 호환 읽기 가능 (lm_head entry 부재 허용). 하지만 lm_head SOA fall-through 위험 때문에 v0.1.1로 다시 build 권장. v0.1.0 reader는 v0.1.1 AUF를 읽으려 하면 `capability_optional` bit 2를 unknown으로 처리할 수 있어 reject 가능 — 이 경우 reader 업데이트.

**Q: lm_head를 AUF에 동봉하지 않고 런타임에 양자화할 수도 있나?**

A: 가능. `auf_tool build --include-lm-head off`로 v0.1.0 호환 AUF를 만들고 generate 측에서 `--quantize-lm-head q4_0`로 강제 변환. 다만 매 실행마다 ~수백 ms 비용 발생. v0.1.1 사전 변환이 권장.

**Q: AUF 자체를 또 다른 컨테이너 (예: zip)에 넣어도 되는가?**

A: 동작은 한다. 다만 mmap 효율이 사라지므로 zero-copy의 의미가 없어진다. 일반 배포에서는 raw `.auf` 파일을 권장.

**Q: format_major=0인 동안 AUF 파일을 production 배포해도 되는가?**

A: 권장하지 않는다. v1.0 stable 선언 (Phase 4 디바이스 실측 + multi-variant 검증 + tool stable 4주 후) 이후에 production 사용. v0.x 기간 중 발견된 spec 결함이 format_major bump를 유발할 수 있다.

**Q: 한 모델의 AUF가 두 종류의 quantization (예: Q4_0 + Q8_0)을 동시에 보관할 수 있는가?**

A: v0.2 (2026-04-27)부터 지원. `--dtypes q4_0,q8_0`으로 지정하면 동일 TENSOR_INDEX entry 그룹에 dtype별 candidate가 다중 등장하며 `capability_optional` bit 3 (`MULTI_DTYPE_VARIANTS`)가 set된다 (ENG-DAT-097, INV-137). Engine 측에서 `--secondary-dtype q8_0`으로 런타임 선택 가능.

**Q: v0.2 multi-quant AUF를 v0.1.x reader가 읽으면 어떻게 되는가?**

A: v0.1.x reader는 `capability_optional` bit 3을 unknown으로 처리하지만 optional이므로 reject하지 않는다. TENSOR_INDEX에서 first-match 규칙을 적용하며, writer가 default_dtype entry를 가장 앞에 정렬하도록 보장하므로(INV-138) v0.1.x reader는 default_dtype 단일 모드로 안전하게 동작한다. 즉 backward-compatible (INV-139).

---

## 6. 교차 참조

| 항목 | 위치 |
|------|------|
| AUF 포맷 정의 (binary layout) | `spec/33-engine-data.md` §3.22 (ENG-DAT-096) |
| Reader/Writer/Stripper 알고리즘 | `spec/32-engine-algorithms.md` §3.12.17 (ENG-ALG-223) |
| 무결성 불변식 (META/section/capability) | `spec/41-invariants.md` §3.16 (INV-132~134) |
| lm_head AOS 불변식 (v0.1.1, G-1-F) | `spec/41-invariants.md` §3.17 (INV-135 v2) |
| v0.2 multi-dtype entry 의미 | `spec/33-engine-data.md` §3.22.14 (ENG-DAT-097) |
| v0.2 dtype selection precedence | `spec/33-engine-data.md` §3.22.15 (ENG-DAT-098) |
| v0.2 META `default_dtype` 필드 | `spec/33-engine-data.md` §3.22.16 (ENG-DAT-099) |
| v0.2 multi-dtype writer 알고리즘 | `spec/32-engine-algorithms.md` §3.12.18 (ENG-ALG-224) |
| v0.2 multi-dtype reader dispatch | `spec/32-engine-algorithms.md` §3.12.18 (ENG-ALG-225) |
| v0.2 multi-dtype 불변식 | `spec/41-invariants.md` §3.18 (INV-137~139) |
| 컴포넌트 매핑 | `arch/auf_format.md` |
| 변경 이력 | `docs/auf_format_changelog.md` (v0.1.0 → v0.1.1 → v0.2) |
| Adreno SOA 재변환 (3.7a fallback) | `spec/32-engine-algorithms.md` §3.12.16 (ENG-ALG-222) |
| Sprint G-1 종결 보고 + 디바이스 측정 | `results/data/weight_swap/phase_6_g1_auf_lmhead.md` |
| Engine 측 weight swap CLI (`--secondary-dtype` 포함) | `docs/USAGE.md` §2.13 |
| 통합 변환 스크립트 (Safetensors→AUF / GGUF→AUF, `--dtypes`/`--default-dtype`) | `scripts/convert_to_auf.sh` |
| Safetensors→GGUF 단계 변환 | `scripts/convert_safetensors_to_gguf.py` |
| TODO / sprint 추적 | `.agent/todos/feat_weight_swap.md` |
