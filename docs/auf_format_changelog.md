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
