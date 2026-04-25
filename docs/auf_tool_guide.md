# auf-tool — User Guide

> AUF (Argus Unified Format) 자산을 만들고 검사하고 수정하는 CLI 도구. Weight Swap Phase 3.7b에서 도입.
>
> 정식 spec: `spec/33-engine-data.md` §3.22, `spec/32-engine-algorithms.md` §3.12.17.
> 컴포넌트: `arch/auf_format.md`. Changelog: `docs/auf_format_changelog.md`.

---

## 1. AUF 개요 (사용자 관점)

**AUF는 GGUF의 self-contained 파생 자산**이다. GGUF 원본 + 모든 backend variant (Adreno SOA, CUDA AOS, CPU AOS)별로 사전 변환된 weight payload를 단일 `.auf` 파일에 보관한다. 디바이스 배포 시 GGUF가 부재해도 Engine이 동작한다.

**왜 필요한가**:
- Adreno noshuffle Q4_0 GEMV kernel은 SOA layout(`q_buf` + `d_buf` 분리, image2d 정렬)을 입력으로 가정한다. GGUF는 AOS 원본 byte를 보관하므로 swap 시점에 변환이 필요하며, 변환 누락 시 fallback kernel로 전환되어 정확도 임계 미달 가능.
- AUF는 빌드 시점에 한 번만 변환하고 디바이스에서는 zero-copy mmap으로 직접 사용.

**언제 쓰는가**:
- `--secondary-source <path>.auf`로 Engine에 전달. `.gguf`와 `.auf`는 확장자로 자동 분기.
- Phase 3.7b 시점에는 secondary weight 자산용. 향후 primary 모델 전체를 AUF로 대체 가능 (v0.2+).

---

## 2. 명령어 레퍼런스

### 2.1 `auf-tool build`

GGUF에서 AUF를 새로 생성한다.

```sh
auf-tool build \
    --input  <path>.gguf \
    --output <path>.auf \
    --variants <list> \
    [--created-by "<custom string>"]
```

| 플래그 | 설명 |
|--------|------|
| `--input` | GGUF 원본 파일. 필수. |
| `--output` | AUF 출력 경로. atomic rename으로 부분 write 노출 금지. 필수. |
| `--variants` | 포함할 backend variant. comma-separated 또는 `all`. 필수. 예: `WEIGHTS_ADRENO_SOA,WEIGHTS_CPU_AOS` 또는 `all`. |
| `--created-by` | 헤더의 `created_by` 필드 (32B UTF-8). 기본은 `"llm_rs2 v{CARGO_PKG_VERSION}"`. |

**동작**:
1. GGUF parse → metadata, tokenizer, layer tensor 추출.
2. Hybrid `source_hash` 계산 (`sha256(size || mtime || head_8MB || tail_8MB)`).
3. 각 variant별로 weight 변환:
   - `WEIGHTS_ADRENO_SOA`: Q/K permute → SOA 분리 → q_img alignment.
   - `WEIGHTS_CUDA_AOS`: Q/K permute → 128B align padding.
   - `WEIGHTS_CPU_AOS`: Q/K permute → 64B align padding.
4. META JSON / TOKENIZER blob / TENSOR_INDEX 직렬화.
5. Section layout 결정 (cursor 기반, 64KB align for `WEIGHTS_*`).
6. Header finalize → tempfile에 write → atomic rename.

**출력**: 성공 시 stdout에 요약 (총 크기, section 수, 각 section size).

**예시**:

```sh
# 모든 backend variant 포함 (워크스테이션 빌드)
auf-tool build \
    --input  models/llama-3.2-1b-q4_0.gguf \
    --output models/llama-3.2-1b.auf \
    --variants all

# Adreno + CPU만 (Android 디바이스 배포 후보)
auf-tool build \
    --input  models/llama-3.2-1b-q4_0.gguf \
    --output models/llama-3.2-1b-mobile.auf \
    --variants WEIGHTS_ADRENO_SOA,WEIGHTS_CPU_AOS
```

---

### 2.2 `auf-tool info`

AUF 파일의 헤더, section 목록, capability flag를 출력한다.

```sh
auf-tool info <path>.auf
```

**출력 예시**:

```
File: models/llama-3.2-1b.auf
Size: 2147483648 bytes (2.00 GiB)

Header:
  magic            : "ARGUS_W\0"
  format           : v0.1.0 (experimental)
  created_by       : "llm_rs2 v0.4.0"
  source_hash      : 3f8a91b2c4d5e6f7... (hybrid)
  source_size      : 705689600 bytes
  source_mtime     : 2026-04-25 14:32:11 UTC
  capability_req   : 0x0000000000000000
  capability_opt   : 0x0000000000000000
  section_count    : 6
  payload_start    : 0x10000 (65536)

Sections:
  Tag                       Offset      Size           Flags    Version
  META                      0x10000     2048           REQUIRED 1
  TOKENIZER                 0x10800     6291456        REQUIRED 1
  TENSOR_INDEX              0x620800    65536          REQUIRED 1
  WEIGHTS_ADRENO_SOA        0x630000    734003200      STRIPPABLE 1
  WEIGHTS_CUDA_AOS          0x2c4d0000  734003200      STRIPPABLE 1
  WEIGHTS_CPU_AOS           0x586a0000  734003200      STRIPPABLE 1

Variants present: ADRENO_SOA, CUDA_AOS, CPU_AOS
```

---

### 2.3 `auf-tool strip`

특정 section을 제거하여 파일 크기를 줄인다. **단순 truncation이 아니라 rewrite** (section table offset 재계산).

```sh
auf-tool strip --keep <list> [--no-backup] <path>.auf
```

| 플래그 | 설명 |
|--------|------|
| `--keep` | 유지할 section tag (comma-separated). 모든 `SECTION_REQUIRED` section을 반드시 포함해야 함. |
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
auf-tool strip \
    --keep META,TOKENIZER,TENSOR_INDEX,WEIGHTS_ADRENO_SOA \
    models/llama-3.2-1b.auf

# 결과: 2 GB → ~700 MB
# 백업: models/llama-3.2-1b.auf.bak

# 백업 생략
auf-tool strip \
    --keep META,TOKENIZER,TENSOR_INDEX,WEIGHTS_CPU_AOS \
    --no-backup \
    models/server.auf
```

---

### 2.4 `auf-tool verify`

AUF 무결성을 검증한다. 선택적으로 source GGUF와 hash 비교.

```sh
auf-tool verify [--source <path>.gguf] <path>.auf
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
auf-tool verify models/llama-3.2-1b.auf

# source GGUF와 일치 검증
auf-tool verify \
    --source models/llama-3.2-1b-q4_0.gguf \
    models/llama-3.2-1b.auf
```

---

### 2.5 `auf-tool repack` (Phase 5로 미룸)

Stripped AUF에 누락된 variant section을 source GGUF로부터 재생성하여 추가.

> v0.1에서는 **미구현**. 사용자가 strip 전 `.auf.bak`을 보관하면 동등한 결과를 얻을 수 있다.

향후 사용 예 (참고):

```sh
auf-tool repack \
    --input  models/server.auf \
    --source models/llama-3.2-1b-q4_0.gguf \
    --add WEIGHTS_ADRENO_SOA \
    --output models/server-with-adreno.auf
```

이 명령은 source GGUF의 hybrid hash가 AUF의 `source_hash`와 일치할 때만 진행된다. 불일치 시 fail-fast (의도된 동작 — 다른 모델로 augment 방지).

---

## 3. 운영 시나리오

### 3.1 워크스테이션 빌드 → Galaxy S25 배포

```sh
# 워크스테이션
auf-tool build \
    --input  llama-3.2-1b-q4_0.gguf \
    --output llama-3.2-1b.auf \
    --variants all

# 디바이스 변종만 strip
auf-tool strip \
    --keep META,TOKENIZER,TENSOR_INDEX,WEIGHTS_ADRENO_SOA \
    llama-3.2-1b.auf
# 결과: ~700 MB (원본 2 GB)

# 디바이스 push
adb push llama-3.2-1b.auf /data/local/tmp/

# Engine 실행
adb shell '/data/local/tmp/generate \
    --secondary-source /data/local/tmp/llama-3.2-1b.auf \
    --eviction-policy ... '
```

### 3.2 Multi-target CI/CD

```sh
# 1회 build, 여러 변종 strip
auf-tool build --input model.gguf --output model.auf --variants all

cp model.auf model-mobile.auf
auf-tool strip --keep META,TOKENIZER,TENSOR_INDEX,WEIGHTS_ADRENO_SOA \
    --no-backup model-mobile.auf

cp model.auf model-server.auf
auf-tool strip --keep META,TOKENIZER,TENSOR_INDEX,WEIGHTS_CUDA_AOS \
    --no-backup model-server.auf

cp model.auf model-cpu.auf
auf-tool strip --keep META,TOKENIZER,TENSOR_INDEX,WEIGHTS_CPU_AOS \
    --no-backup model-cpu.auf
```

### 3.3 모델 업데이트

GGUF가 갱신된 경우, AUF도 처음부터 다시 build. v0.1에서는 incremental update 미지원 (가능하지만 source_hash 변경으로 의미 모호).

---

## 4. 트러블슈팅

### 4.1 source_hash mismatch

**증상**:
```
auf-tool verify --source model.gguf model.auf

Warning: AUF source_hash differs from given GGUF
  AUF says : 3f8a91b2c4d5e6f7...
  Computed : a1b2c3d4e5f60718...
```

**원인 후보**:
1. GGUF가 AUF build 이후 수정됨 (mtime 변경, 또는 head/tail 8MB 영역 변경).
2. AUF가 다른 GGUF로부터 build됨.
3. AUF가 strip되면서 `source_hash`는 보존됨 (정상).

**조치**:
- 1번/2번 경우: 원본 GGUF로 다시 build (`auf-tool build`).
- 3번 경우: strip은 source_hash를 보존하므로 정상. `auf-tool info` 확인.
- 무시하고 사용 가능 (verify의 source_hash 비교는 informational, reject 아님 — INV-132).

### 4.2 missing WEIGHTS_* section

**증상** (Engine 실행 시):
```
Error: AUF does not contain WEIGHTS_ADRENO_SOA section.
       Run 'auf-tool repack --add WEIGHTS_ADRENO_SOA' to add it from source GGUF.
```

**원인**: 디바이스의 backend가 요구하는 `WEIGHTS_*` section이 strip되었거나 처음부터 build되지 않음.

**조치**:
- v0.1 (repack 미구현): `.auf.bak`이 있으면 복원 후 재strip.
  ```sh
  cp model.auf.bak model.auf
  auf-tool strip --keep META,TOKENIZER,TENSOR_INDEX,WEIGHTS_ADRENO_SOA model.auf
  ```
- backup이 없으면 source GGUF에서 다시 build:
  ```sh
  auf-tool build --input model.gguf --output model.auf --variants all
  auf-tool strip --keep META,TOKENIZER,TENSOR_INDEX,WEIGHTS_ADRENO_SOA model.auf
  ```
- v0.2+ repack 가능 후: `auf-tool repack --add WEIGHTS_ADRENO_SOA --source model.gguf model.auf`.

### 4.3 format_major mismatch

**증상**:
```
Error: AUF format_major=2 but reader supports up to 1. Update llm_rs2.
```

**원인**: AUF가 새 버전 도구로 만들어졌으나 reader는 구버전.

**조치**:
- llm_rs2 / auf-tool 업데이트.
- 또는 구버전 도구로 AUF를 다시 build (가능하다면).
- v0.x 기간(format_major=0)에는 minor 버전 변경도 reject 가능 (`AUF format_major=0 is experimental` 경고 후 진행 권장이지만 reader 구현이 거부 가능).

### 4.4 corrupt section table

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

### 4.5 unknown capability bit

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

A: `auf-tool info`로 `Variants present` 줄 확인. 모든 variant가 빠져 있으면 strip된 것이 아니라 build 시점에 일부만 포함된 것일 수 있다.

**Q: AUF 자체를 또 다른 컨테이너 (예: zip)에 넣어도 되는가?**

A: 동작은 한다. 다만 mmap 효율이 사라지므로 zero-copy의 의미가 없어진다. 일반 배포에서는 raw `.auf` 파일을 권장.

**Q: format_major=0인 동안 AUF 파일을 production 배포해도 되는가?**

A: 권장하지 않는다. v1.0 stable 선언 (Phase 4 디바이스 실측 + multi-variant 검증 + tool stable 4주 후) 이후에 production 사용. v0.x 기간 중 발견된 spec 결함이 format_major bump를 유발할 수 있다.

**Q: 한 모델의 AUF가 두 종류의 quantization (예: Q4_0 + Q8_0)을 동시에 보관할 수 있는가?**

A: v0.1에서는 single quantization 가정. multi-quantization 보관은 별도 section tag (예: `WEIGHTS_CPU_AOS_Q4_0`, `WEIGHTS_CPU_AOS_Q8_0`)로 가능하나 spec에 명시되지 않음. v0.2 후보.

---

## 6. 교차 참조

| 항목 | 위치 |
|------|------|
| AUF 포맷 정의 (binary layout) | `spec/33-engine-data.md` §3.22 (ENG-DAT-096) |
| Reader/Writer/Stripper 알고리즘 | `spec/32-engine-algorithms.md` §3.12.17 (ENG-ALG-223) |
| 무결성 불변식 | `spec/41-invariants.md` §3.16 (INV-132~134) |
| 컴포넌트 매핑 | `arch/auf_format.md` |
| 변경 이력 | `docs/auf_format_changelog.md` |
| Adreno SOA 재변환 (3.7a fallback) | `spec/32-engine-algorithms.md` §3.12.16 (ENG-ALG-222) |
| Phase 3.7 TODO | `.agent/todos/feat_weight_swap.md` |
