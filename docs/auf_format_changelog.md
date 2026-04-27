# AUF Format Changelog

> AUF (Argus Unified Format) 버전별 변경 이력. 본 changelog는 **포맷 그 자체**의 변경만 기록한다 — 구현(`auf-tool` CLI, Engine reader)의 변경은 일반 git log를 따른다.
>
> 정식 spec: `spec/33-engine-data.md` §3.22 (ENG-DAT-096) + `spec/32-engine-algorithms.md` §3.12.17 (ENG-ALG-223).
> 컴포넌트 매핑: `arch/auf_format.md`.

---

## 변경 형식

각 버전은 다음 형식을 따른다:

```
## v<MAJOR>.<MINOR>.<PATCH> — YYYY-MM-DD

### Added
- 새 section tag, capability flag, 헤더 필드 등 (additive 변경)

### Changed
- 기존 필드 의미 변경, schema 확장 (interpretation 변경)

### Removed / Deprecated
- 폐기된 section, 폐기된 flag bit (제거 사유 + 영향)

### Compatibility
- 이전 버전과의 호환성 영향. format_major 증가 시 migration note.

### Capability bits (snapshot)
- 현재 알려진 capability_required / capability_optional 비트 정의

### Section catalog (snapshot)
- 현재 알려진 section tag 목록과 의미

### Implementation notes
- Reader/Writer/Stripper에서 변경된 알고리즘 (있는 경우)
```

---

## v0.2 — 2026-04-27 (Multi-dtype Variant)

> **상태**: Experimental. `format_major = 0`이므로 forward/backward 호환성 보장 안 함.
> **도입 컨텍스트**: Dynamic weight swap의 secondary dtype payload를 GGUF 의존 없이 self-contained AUF에 보관하기 위함. 동일 (`backend`, `layer_idx`, `kind`) 쌍에 대해 여러 dtype 후보(예: Q4_0 + F16)를 한 파일에 동시 보관 가능. SwapExecutor 인터페이스는 변경되지 않음 (단방향 swap 가정 유지).
> **호환성**: v0.1.x와 **byte-level 호환 출력 가능**. 단일 dtype 빌드(`--dtypes Q4_0`)는 v0.1.x와 동일 출력. 다중 dtype 빌드는 capability_optional bit 3 = 1 + format_minor=2이며, v0.1.x reader가 만나면 bit 3을 무시하고 first-match로 default_dtype payload를 단일 모드로 안전하게 사용 (writer의 INV-138 안정 정렬 의무 덕분).
> **format 변경**: format_major=0 그대로, format_minor 1→2 bump, format_patch 0으로 리셋.

### Added

- **capability_optional bit 3 = `MULTI_DTYPE_VARIANTS`** 신설.
  - 1: TENSOR_INDEX에 동일 (`layer_idx`, `kind`)에 대해 dtype별 다중 entry가 등장하며 META에 `default_dtype` 필드가 정의되어 있다.
  - 0: single-dtype 모드. 동일 (`layer_idx`, `kind`)에 entry가 1번씩만 등장. v0.1.x와 동일 의미.
  - reader는 bit 3 미인식 시 ignore (optional). v0.1.x reader가 v0.2 AUF를 만나면 first-match 규칙으로 default_dtype을 단일 모드로 사용.
- **TENSOR_INDEX entry 의미 확장 (ENG-DAT-097)**.
  - 동일 (`layer_idx`, `kind`) 쌍에 대해 dtype이 다른 entry가 여러 개 등장 가능. entry 자체의 binary layout(`(layer_idx, kind, dtype, shape_rank, shape, alignment, variant_offsets, variant_sizes)`)은 변경 없음.
  - **schema_version = 1 그대로 유지** (v0.1.x reader 양방향 호환 위해 schema bump 금지).
  - Writer는 entries를 (`layer_idx` ASC, `kind` ASC, `is_default` DESC, `dtype` ASC)로 안정 정렬하여 default_dtype entry가 그룹 첫 번째에 오도록 보장 (INV-138).
- **META JSON `default_dtype` 필드 추가 (ENG-DAT-099)**.
  - capability bit 3 = 1일 때 의무 (INV-138).
  - 값: `"F32"` / `"F16"` / `"BF16"` / `"Q4_0"` / `"Q4_1"` / `"Q8_0"` / `"U8"` 중 하나.
  - 기존 META JSON 키 뒤에 append하여 byte prefix 보존 (v0.1.x 호환).
- **dtype selection precedence (ENG-DAT-098)** 정의.
  - [호출자 명시 dtype → META.default_dtype → first-match] 순.
  - v0.1.x reader는 명시 dtype API가 없으므로 first-match 단일 진입.
- **Writer dtype 변환 파이프라인 (ENG-ALG-224)** 정의.
  - 입력 GGUF source dtype × 요청 candidate dtype에 따라 identity / dequant→requant 분기.
  - 모든 변환은 deterministic (quantize_q4_0 round-half-to-even, F16↔F32 bit-for-bit).
- **Reader dtype dispatch (ENG-ALG-225)** 정의.
  - `lookup_tensor(layer, kind, requested_dtype)` API로 dtype 선택.

### Changed

- TENSOR_INDEX entry 의미: "동일 (`layer_idx`, `kind`) 쌍은 단 1개 entry" → "dtype별 N개 entry 가능" (capability bit 3 = 1일 때).
- META JSON: `default_dtype` 필드가 multi-dtype AUF에서 의무 추가 필드로 정의됨.
- format_minor: 1 → 2 (additive 의미 확장).
- 기존 capability_optional bit 위치 권고 갱신:
  - bit 3 = `MULTI_DTYPE_VARIANTS` (신규 할당).
  - bit 4 = `UNIGRAM_TOKENIZER` (구 권고 bit 3에서 이동).
  - bit 5 = `BIDIRECTIONAL_SWAP` (예약, 양방향 swap 도입 시).

### Removed / Deprecated

- 없음.

### Compatibility

- **v0.1.0 / v0.1.1과 byte-level 호환 출력 가능** (single-dtype 빌드).
  - `--dtypes Q4_0` 단독 빌드 → capability bit 3 = 0, format_minor = 1 그대로 (v0.1.1 호환 출력) 또는 format_minor = 2로 출력하지만 의미적으로 single-dtype.
  - 본 changelog v0.2는 multi-dtype 사용 시점만 다루며, single-dtype 빌드는 v0.1.x 동작과 의미 동일.
- **호환성 매트릭스 (Reader × AUF)**:

  | Reader | v0.1.0 AUF | v0.1.1 AUF | v0.2 AUF (multi-dtype) |
  |--------|-----------|-----------|----------------------|
  | v0.1.0 | OK | OK (bit 2 ignore) | OK (bit 3 ignore + first-match → default_dtype) |
  | v0.1.1 | OK | OK | OK (bit 3 ignore + first-match → default_dtype) |
  | v0.2 | OK | OK | OK |

- **v0.1.x reader 호환의 핵심**:
  1. bit 3은 `capability_optional`이므로 미인식 reader는 reject하지 않는다 (INV-132 호환).
  2. v0.2 writer는 INV-138 안정 정렬 의무를 지켜 first-match가 default_dtype과 일치하게 보장한다.
  3. META의 `default_dtype` 키는 unknown JSON key로 v0.1.x reader가 무시한다 (단, JSON parser의 unknown_fields = ignore 동작이 implementation 검증 항목).
- **lm_head 처리 (Sprint A' 반전, 2026-04-27)**: lm_head도 layer weight와 동일하게 multi-dtype 후보 entry로 등록 가능하다 (예: Q4_0 + F16 등 모든 candidate dtype). reader dispatch precedence(ENG-ALG-225)가 lm_head에도 동일 적용된다. 다만 **INV-135 v2의 layout 의무**(`WEIGHTS_ADRENO_SOA` 안에서 AOS 18B/block 강제, image1d_buffer_t 한계로 인함)는 dtype에 무관하게 모든 lm_head candidate entry에 강제된다. v0.1.1 시점의 "lm_head Q4_0 single dtype" 결정은 폐기되었다. 디스크 영향: lm_head F16 entry 추가 시 1B 모델 기준 ~+260 MB / variant.
- **SwapExecutor 인터페이스 변경 없음** (Q3 단방향 swap 가정, ENG-DAT-C17).

### Capability bits (snapshot — v0.2)

| Field | Bit | Name | 의미 | 상태 |
|-------|-----|------|------|------|
| `capability_required` | 0..63 | (none) | 사용 안 함 | 모두 0 |
| `capability_optional` | 0 | `SOURCE_HASH_FULL_SHA256` | reserved | 미할당 |
| `capability_optional` | 1 | `IMAGE2D_PRECOMPUTED` | reserved | 미할당 |
| `capability_optional` | 2 | `LM_HEAD_PRECOMPUTED_Q4_0` | lm_head Q4_0 사전 변환 (v0.1.1) | v0.1.1 도입 |
| `capability_optional` | 3 | `MULTI_DTYPE_VARIANTS` | TENSOR_INDEX dtype별 다중 entry + META default_dtype | **v0.2 도입** |
| `capability_optional` | 4..63 | (none) | reserved | 모두 0 |

### Section catalog (snapshot — v0.2)

v0.1.x와 동일 (6개 tag). multi-dtype payload는 신규 section을 만들지 않고 기존 `WEIGHTS_<backend>` section 내부에 dtype별 sub-payload로 동봉한다 (Q1=B 결정).

| Tag | required | strippable | 내용 (v0.2) |
|-----|----------|------------|-------------|
| `META` | yes | no | architecture, dims, RoPE config, RMSNorm epsilon + **`default_dtype`** (multi-dtype 시) |
| `TOKENIZER` | yes | no | vocab + BPE merges + special tokens + chat template |
| `TENSOR_INDEX` | yes | no | layer index → tensor 메타데이터 매핑. **multi-dtype 시 dtype별 다중 entry, schema_version=1 그대로** |
| `WEIGHTS_ADRENO_SOA` | no | yes | Adreno SOA. **multi-dtype 시 dtype별 sub-payload 인접 배치** + lm_head Q4_0 AOS |
| `WEIGHTS_CUDA_AOS` | no | yes | CUDA AOS. **multi-dtype 시 dtype별 sub-payload** + lm_head Q4_0 AOS |
| `WEIGHTS_CPU_AOS` | no | yes | CPU AOS. **multi-dtype 시 dtype별 sub-payload** + lm_head Q4_0 AOS |

### Implementation notes

- Writer (`auf-tool build`):
  - 신규 옵션 `--dtypes <list>` (예: `Q4_0,F16`). default: GGUF source dtype 단일.
  - 신규 옵션 `--default-dtype <DTYPE>`. `--dtypes`에 포함된 값이어야 함. default: `--dtypes` 첫 번째.
  - 단일 dtype 빌드는 v0.1.x 호환 출력 (bit 3 = 0, format_minor 1 또는 2).
  - 다중 dtype 빌드는 bit 3 = 1 + format_minor=2.
  - dtype 변환은 ENG-ALG-224 §3.12.18.1의 deterministic 함수만 사용.
  - TENSOR_INDEX entry 안정 정렬: (`layer_idx` ASC, `kind` ASC, `is_default` DESC, `dtype` ASC).
- Reader: `lookup_tensor(layer, kind, requested_dtype: Option<DType>) -> Result<&TensorIndexEntry>`.
  - capability bit 3 검사 + dtype dispatch (ENG-ALG-225).
  - panic 없음. 모든 위반은 명시적 `AufError`.
- AufView에 `multi_dtype_enabled() -> bool` 추가.
- ModelMeta에 `default_dtype: Option<DType>` 추가.
- SwapExecutor 인터페이스: **변경 없음**. multi-dtype payload는 reader가 dispatch하여 제공한 byte slice를 그대로 받음.
- lm_head 처리 (Sprint A' 반전): lm_head도 layer weight와 동일하게 multi-dtype 후보 적용. Writer는 `convert_to_<variant>_for_dtype` 안에서 lm_head tensor에 대해 SOA variant라도 AOS bytes로 직렬화하는 분기를 가진다 (INV-135 v2 layout 의무, dtype-agnostic). Reader는 lm_head에 대해서도 ENG-DAT-098 precedence(호출자 명시 → META.default_dtype → first-match)를 동일 적용한다. ENG-DAT-C16 갱신본 참조.

### Tests required

- **Multi-dtype roundtrip**: `--dtypes Q4_0,F16` build → read → 양 dtype payload byte-level 일치.
- **v0.1.x reader 호환**: v0.1.0/v0.1.1 reader가 v0.2 AUF(bit 3=1)를 무시하고 default_dtype 단일 모드로 first-match 진입 → 동작 정상.
- **dtype dispatch precedence**: requested_dtype 명시/미명시 × default_dtype 일치/불일치 매트릭스 (3 × 2 = 6 케이스).
- **Shape consistency (INV-137)**: 동일 (layer, kind)의 다중 dtype entry shape 불일치 → reject.
- **Default_dtype 누락 (INV-138)**: capability bit 3 = 1이지만 META에 default_dtype 부재 → reject.
- **Writer 안정 정렬 (INV-138)**: TENSOR_INDEX의 default_dtype entry가 그룹 첫 번째에 옴 → byte-level 검증.
- **lm_head AOS layout 강제 (INV-135 v2 + ENG-DAT-C16 갱신본, Sprint A')**: multi-dtype AUF에서 `WEIGHTS_ADRENO_SOA` variant 안의 lm_head sub-payload가 SOA layout으로 동봉되면 reject (`AufError::LmHeadSoaForbidden`). dtype은 Q4_0 / F16 등 다중 후보 모두 허용되며 dtype 단일성은 검증 대상이 아니다. lm_head 다중 dtype entry shape 일치(INV-137)는 별도 검증.
- **lm_head multi-dtype roundtrip (Sprint A')**: `--dtypes Q4_0,F16`로 lm_head 양 dtype entry가 등록되고, 각각 AOS bytes로 동봉되었음을 byte-level 검증.
- **dtype 변환 결정성 (ENG-ALG-C12)**: 동일 GGUF + 동일 `--dtypes` → 2회 build → byte-level 일치.
- **format 호환성**: format_minor=2 + bit 3 = 0 케이스 (single-dtype 빌드인데 format_minor만 bump됨)는 호환 동작 (INV-139 비고).
- **신규 invariants 테스트**: INV-137 / INV-138 / INV-139 직접 검증 테스트.

### Source hash 정의

- 변경 없음. AUF 헤더의 hybrid `source_hash`(GGUF 전체) 그대로 재사용.
- multi-dtype은 같은 GGUF에서 deterministic 변환된 다중 표현이므로 source_hash 일치는 모든 dtype payload 일치를 함의 (ENG-DAT-096.13 결정성 보장).

---

## v0.1.1 — 2026-04-26 (Sprint G-1: lm_head Q4_0 사전 변환)

> **상태**: Experimental. `format_major = 0`이므로 forward/backward 호환성 보장 안 함.
> **도입 컨텍스트**: Phase 6 Sprint G-1. Sprint F에서 lm_head F16→Q4_0 runtime quantize로 ratio=1.0 mixed weight-swap TBT 회귀(+24.7%) 회수 완료. 잔여 비용 = model load 시점 ~1.4 s quantize. 본 v0.1.1은 build 시점에 lm_head Q4_0 payload를 backend variant section에 동봉하여 cold-start latency 제거.
> **호환성**: v0.1.0과 **byte-level 호환**. 신규 capability bit 2는 optional이므로 구 reader가 무시한다. 신 reader가 v0.1.0 AUF를 만나면 capability bit 2 = 0으로 판정하여 runtime quantize fallback 정상 동작. format_major / format_minor는 변경 없음 (additive 변경, patch 증가).

### Added
- **capability_optional bit 2 = `LM_HEAD_PRECOMPUTED_Q4_0`** 신설.
  - 1: lm_head를 build 시점에 Q4_0으로 사전 quantize하여 backend variant section에 동봉.
  - 0: lm_head는 GGUF 원본 dtype 그대로 (또는 backend variant section에 미포함).
  - reader는 bit 0 시 runtime quantize fallback. bit 1 시 AUF section에서 직접 매핑.
- **TENSOR_INDEX `kind = 11(lm_head)` entry 의미 확장** (ENG-DAT-096.12 추가).
  - lm_head Q4_0 사전 변환 시 entry의 `dtype = Q4_0`, `shape = [vocab_size, hidden_dim]` (GGUF 원본 그대로), `variant_offsets`가 backend variant section 내부의 사전 변환 payload offset을 가리킨다.
  - cross-layer tensor의 `layer_idx = u32::MAX` 규칙은 그대로 적용.
  - **layout (G-1-F update, INV-135 v2)**: lm_head Q4_0 entry는 모든 backend variant에서 **AOS 18B/block layout**으로 동봉된다. `WEIGHTS_ADRENO_SOA` section 내부에서도 lm_head는 SOA 변환을 적용하지 않는다 — `vocab × hidden / 8` texels의 `q_buf`가 OpenCL `CL_DEVICE_IMAGE_MAX_BUFFER_SIZE` 한계를 거의 모든 디바이스에서 초과하여 image1d_buffer_t 생성이 실패하고 빠른 SOA path를 발동시킬 수 없기 때문이다 (Sprint G-1-F 디바이스 측정에서 garbage 출력 회귀로 확인됨). layer weight는 SOA 변환을 그대로 적용.
- **결정성 요구사항** ENG-DAT-096.13 명시. 동일 GGUF + 동일 build option + 동일 host → byte-level 동일 AUF.

### Changed
- 기존 capability_optional bit position 권고 갱신:
  - bit 2 = `LM_HEAD_PRECOMPUTED_Q4_0` (신규 할당).
  - bit 3 = `UNIGRAM_TOKENIZER` (구 권고 bit 2에서 이동).
- TENSOR_INDEX cross-layer tensor entry 의미를 "GGUF 원본 dtype 보존"에서 "build 시점 backend-specific dtype downgrade 결과"로 확장.

### Removed / Deprecated
- 없음.

### Compatibility
- **v0.1.0과 byte-level 호환**.
  - 구 reader(v0.1.0) + 신 AUF(v0.1.1, bit 2 set): bit 2는 optional이므로 reject 사유 아님. 구 reader는 lm_head를 GGUF에서 다시 quantize하는 fallback 경로를 거치며 동작 정상 (cold-start latency 비용은 남는다).
  - 신 reader(v0.1.1) + 구 AUF(v0.1.0, bit 2 = 0): runtime quantize fallback. Sprint F 동작 그대로 보존.
- format_major / format_minor 변경 없음. format_patch만 증가.

### Capability bits (snapshot — v0.1.1)

| Field | Bit | Name | 의미 | 상태 |
|-------|-----|------|------|------|
| `capability_required` | 0..63 | (none) | 사용 안 함 | 모두 0 |
| `capability_optional` | 0 | `SOURCE_HASH_FULL_SHA256` | reserved | 미할당 |
| `capability_optional` | 1 | `IMAGE2D_PRECOMPUTED` | reserved | 미할당 |
| `capability_optional` | 2 | `LM_HEAD_PRECOMPUTED_Q4_0` | lm_head Q4_0 사전 변환 | **v0.1.1 도입** |
| `capability_optional` | 3..63 | (none) | reserved | 모두 0 |

### Section catalog (snapshot — v0.1.1)

v0.1.0과 동일 (6개 tag). lm_head Q4_0 payload는 신규 section을 만들지 않고 기존 `WEIGHTS_<backend>` 내부에 layer weight와 동일 layout으로 동봉.

### Implementation notes
- Writer (`auf-tool build`):
  - 신규 옵션 `--include-lm-head <on|off|auto>` (default `auto`).
  - `auto`: GGUF lm_head dtype이 Q4_0이 아니면 quantize.
  - `on`: 강제 (이미 Q4_0이면 no-op).
  - `off`: skip → v0.1.0 호환 출력, capability_optional bit 2 = 0.
  - quantize는 `quantize_q4_0` 결정성 함수 사용. SOA 변환은 host에서 deterministic kernel.
- Reader: `lm_head_q4_0_payload() -> Option<LmHeadPayload>` accessor 신설. capability bit 2 검사 + TENSOR_INDEX entry lookup.
- model load 분기 (`transformer.rs`): AUF entry 우선 → 없으면 runtime quantize fallback (Sprint F 동작 보존).
- Engine CLI `--quantize-lm-head` 의미 갱신:
  - `auto` (default): AUF entry 우선 → 없으면 runtime quantize.
  - `none`: F16 유지.
  - `q4_0`: 강제 runtime quantize (AUF entry 무시, 회귀 비교용).

### Source hash 정의
- 변경 없음. AUF 헤더의 hybrid `source_hash`(GGUF 전체)를 그대로 재사용. lm_head 단일 tensor에 대한 별도 hash 미도입.
- 근거: deterministic build이므로 source_hash 일치는 lm_head Q4_0 일치를 함의. Llama 3.2 1B 수준에서 GGUF tail 8 MB가 lm_head 영역을 거의 항상 포함하므로 hybrid hash로도 충분. 8 B+ 모델은 v0.x에서 그대로 유지(§3.22.6 채택 결정 시점부터의 한계 그대로). 향후 `SOURCE_HASH_FULL_SHA256` capability bit으로 보강 가능.

### Tests required
- AUF roundtrip: lm_head Q4_0 entry write → read → bytes 일치.
- 후방 호환: v0.1.0 AUF + 신 reader → bit 2 = 0 → runtime fallback.
- 결정성: 동일 GGUF → auf-tool 2회 실행 → byte-level 일치 (host 한정).
- 신규 INV-135 (source_hash 일치) + INV-136 (fallback 정상 동작) 테스트.

---

## v0.1.0 — 2026-04-25 (Initial Draft)

> **상태**: Experimental. `format_major = 0`이므로 forward/backward 호환성 보장 안 함.
> **도입 컨텍스트**: Weight Swap Phase 3.7b. Adreno noshuffle Q4_0 SOA layout을 사전 변환하여 self-contained 자산으로 보관.

### Added
- 포맷 magic `"ARGUS_W\0"` (8B).
- 256B 고정 헤더 (`AufHeader`):
  - `magic` (8B), `format_major` (u16), `format_minor` (u16), `format_patch` (u16), `_pad0` (u16).
  - `created_by` (32B UTF-8, NUL-padded).
  - `source_hash` (32B), `source_size` (u64), `source_mtime` (u64).
  - `capability_required` (u64), `capability_optional` (u64).
  - `section_count` (u32), `_pad1` (u32), `section_table_offset` (u64), `payload_start_offset` (u64).
  - `_reserved` (120B, 0 fill).
- 가변 길이 section table. 각 entry 48B:
  - `tag` (16B UTF-8 ASCII, NUL-padded), `offset` (u64), `size` (u64), `flags` (u32), `version` (u32), `_reserved` (8B).
- Section flag bits:
  - bit 0: `SECTION_REQUIRED` — reader가 인식 못 하면 reject.
  - bit 1: `SECTION_STRIPPABLE` — `auf-tool strip`이 안전하게 제거 가능.
  - bit 2: `SECTION_COMPRESSED` — reserved (v0.1 미사용).
- Hybrid `source_hash` 알고리즘: `sha256(size_le8 || mtime_le8 || head_8MB || tail_8MB)`.
- Endianness: little-endian 전 필드.
- Section payload alignment: 64 KB (Linux/Android THP 친화).

### Section catalog (snapshot — v0.1)

| Tag | required | strippable | 내용 |
|-----|----------|------------|------|
| `META` | yes | no | architecture, dims, RoPE config, RMSNorm epsilon (JSON-in-binary) |
| `TOKENIZER` | yes | no | vocab + BPE merges + special tokens + chat template |
| `TENSOR_INDEX` | yes | no | layer index → tensor 메타데이터 매핑 |
| `WEIGHTS_ADRENO_SOA` | no | yes | Adreno SOA: q_buf + d_buf + q_img alignment, Q/K permute applied |
| `WEIGHTS_CUDA_AOS` | no | yes | CUDA AOS: 18B block + 128B align, Q/K permute applied |
| `WEIGHTS_CPU_AOS` | no | yes | CPU AOS: 18B block + 64B align, Q/K permute applied |

### Capability bits (snapshot — v0.1)

| Field | Bit | Name | 의미 | 상태 |
|-------|-----|------|------|------|
| `capability_required` | 0..63 | (none) | v0.1은 required capability 미사용 | 모두 0 |
| `capability_optional` | 0..63 | (none) | v0.1은 optional capability 미사용 | 모두 0 |

향후 v0.x에서 추가될 후보:
- `capability_optional` bit 0: `SOURCE_HASH_FULL_SHA256` (hybrid 대신 full SHA256).
- `capability_optional` bit 1: `IMAGE2D_PRECOMPUTED` (Adreno q_img를 device-specific texture format으로 사전 인코딩).
- `capability_required` bit 0: `ZSTD_COMPRESSION` (section payload zstd 압축).
- `capability_required` bit 1: `LORA_DELTAS` (multi-asset bundle, Mode C).

### Tokenizer schema

`TOKENIZER` section payload 내부:

| Offset | Field | 의미 |
|--------|-------|------|
| 0..16 | magic | `"ARGUS_TOK\0\0\0\0\0\0\0"` |
| 16..20 | schema_version (u32) | 1 |
| 20..24 | tokenizer_kind (u32) | 0 = BPE (v0.1 유일 지원) |
| 24..28 | vocab_size (u32) | |
| 28..32 | merges_count (u32) | BPE pair 수, 0이면 부재 |
| 32..40 | special_tokens_offset (u64, section-local) | |
| 40..48 | chat_template_offset (u64, section-local, 0 = absent) | |
| 48..56 | tokens_blob_offset (u64, section-local) | |
| 56..64 | merges_blob_offset (u64, section-local, 0 = absent) | |

특수 토큰 ID (12B): `bos_id` (i32), `eos_id` (i32), `pad_id` (i32), `unk_id` (i32). `-1`이면 미설정.

### TENSOR_INDEX schema

`TENSOR_INDEX` section payload 내부:

| Offset | Field | 의미 |
|--------|-------|------|
| 0..16 | magic | `"ARGUS_TIDX\0\0\0\0\0\0"` |
| 16..20 | schema_version (u32) | 1 |
| 20..24 | variant_count (u32) | section table에 등장한 `WEIGHTS_*` 수 |
| 24..28 | tensor_count (u32) | |
| 28..32 | _pad (u32) | 0 |
| 32..(32 + variant_count * 16) | variant tag list | 각 16B `[u8; 16]` (section table tag와 일치) |
| 그 이후 | tensor entries | 가변 길이 (shape_rank에 따름) |

각 tensor entry: `(layer_idx u32, kind u32, dtype u32, shape_rank u32, shape [u64; rank], alignment u64, variant_offsets [u64; variant_count], variant_sizes [u64; variant_count])`.

### Compatibility
- 이전 버전 없음 (initial release).
- `format_major = 0`인 동안 다음 버전(v0.2 등)으로의 호환성 보장 없음.
- Reader는 `format_major = 0`을 만나면 stderr에 명시적 경고 출력: `"AUF format_major=0 is experimental"`.

### Implementation notes
- Reader: panic 없이 모든 무결성 위반을 `Result::Err`로 반환 (ENG-ALG-C10).
- Writer: atomic rename으로 partial state 외부 노출 금지 (ENG-ALG-C11).
- Stripper: in-place rewrite + 기본 backup (`<name>.auf.bak`). `--no-backup` 시 백업 생략.
- Repacker (`auf-tool repack`)는 v0.1에서 **미구현**, Phase 5로 미룸.

---

## (Reserved for future versions)

향후 v0.2 이후의 entry는 위 형식을 따라 본 changelog 상단에 추가한다. 가장 최신 버전이 가장 위에 오도록 유지.
