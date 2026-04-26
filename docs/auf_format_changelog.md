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
  - layout: layer weight와 동일 (Adreno SOA / CUDA AOS / CPU AOS).
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
