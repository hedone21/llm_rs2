# AUF (Argus Unified Format) — Self-Contained Weight Asset Architecture

> **상태**: Draft v0.2 (2026-04-27, multi-dtype variant 추가).
> **대상 spec**: `spec/33-engine-data.md` §3.22 (ENG-DAT-096, .12, .13, .14, .15, .16), `spec/32-engine-algorithms.md` §3.12.17 (ENG-ALG-223), §3.12.18 (ENG-ALG-224, ENG-ALG-225), `spec/41-invariants.md` §3.16~3.18 (INV-132~139).
> **연관 작업**:
> - Phase 3.7a (ENG-ALG-222 / INV-131) — runtime SOA 재변환 safety net. AUF 부재 시 fallback 경로로 사용.
> - Phase 6 Sprint G-1 (ENG-DAT-096.12 / INV-135) — lm_head Q4_0 사전 변환으로 model load 시점 ~1.4 s runtime quantize 비용 제거.
> - **v0.2 multi-quant** (ENG-DAT-097~099 / INV-137~139) — dynamic weight swap의 secondary dtype payload를 self-contained AUF로 보관하여 GGUF 의존을 제거. SwapExecutor 인터페이스는 변경되지 않음 (단방향 swap 가정).
> **작성**: 2026-04-25 (v0.1), 2026-04-26 (v0.1.1), 2026-04-27 (v0.2).

---

## 0. 컨텍스트

Phase 3.6 디바이스 실측에서 Q4_0 weight swap 후 첫 토큰("Paris")은 정답이지만 후속 토큰이 garbage("(Parameter" 반복)인 현상이 관측되었다. 근본 원인:

- **OpenCL backend의 `copy_from`은 Q4_0 weight를 AOS(Array-of-Structures) 원본 바이트 그대로 GPU로 업로드**한다.
- **Adreno noshuffle Q4_0 GEMV kernel은 SOA(Struct-of-Arrays) layout** (`q_buf` + `d_buf` 분리, `q_img` image2d 정렬)을 입력으로 가정한다.
- Swap 시점에 AOS→SOA 변환이 누락되면 noshuffle kernel은 매칭 SOA descriptor를 찾지 못해 일반 fallback kernel로 전환되며, 정확도가 임계치를 미달한다.

**해결안 두 갈래**:
- **3.7a (런타임 safety net)**: swap 직후 `convert_aos_to_soa()`를 명시 호출하여 매번 변환. 호환성 안전, 변환 비용 매번 발생.
- **3.7b (AUF 포맷)**: 빌드 시점에 모든 backend variant payload를 사전 생성하여 단일 self-contained 파일에 보관. 런타임 변환 비용 0, 다만 빌드 도구(`auf-tool`)와 spec 안정화 필요.

본 문서는 **3.7b** 갈래의 컴포넌트 매핑이다. 3.7a는 `arch/weight_swap.md`에서 다룬다.

---

## 1. 개요

### 1.1 AUF의 역할

AUF는 GGUF의 **derived but independent self-contained 자산**이다. 다음 3가지 운영 모드 중 본 v0.1 spec은 **Mode B (self-contained)** 단일을 채택한다.

| Mode | 설명 | v0.1 채택 여부 |
|------|------|---------------|
| Mode A | AUF는 GGUF 옆에 있는 cache. AUF 부재/stale 시 GGUF에서 자동 재생성 | 미채택 |
| **Mode B** | **AUF가 자립적 자산. GGUF 없이도 동작. 모든 metadata/tokenizer/tensor를 포함** | **채택** |
| Mode C | AUF는 multi-asset bundle (모델 + adapters + finetune deltas) | 미채택 |

**B-2 (multi-variant single file) + Selective Strip**: 빌드 시점에 모든 backend variant(`WEIGHTS_ADRENO_SOA` / `WEIGHTS_CUDA_AOS` / `WEIGHTS_CPU_AOS`)를 한 파일에 동시 보관. 배포 시 target device variant만 남기고 나머지 strip.

### 1.2 GGUF와의 관계

```mermaid
flowchart LR
    GGUF[(GGUF<br/>원본)] -->|auf-tool build| AUF[(AUF<br/>self-contained)]
    AUF -->|auf-tool strip| AUFD[(AUF<br/>device-specific)]
    AUFD -->|deploy| Device((Device))
    GGUF -.optional.-> Device

    AUFD -->|reader| Engine[Engine<br/>Q4_0 swap]

    style AUF fill:#fff3e0
    style AUFD fill:#c8e6c9
```

핵심 관계:
- **build 시점**: GGUF 입력 필수. variant 변환(`Q/K permute`, SOA reorder, alignment padding) 모두 수행.
- **deploy 시점**: AUF 단일 파일만 디바이스로 전송. GGUF는 워크스테이션에 잔류.
- **runtime**: Engine은 AUF만 mmap. `source_hash`는 정보성 메타데이터로 보존되지만 GGUF가 부재해도 정상 동작 (INV-132).

### 1.3 v0.1 범위와 비범위

**범위**:
- 256B 헤더 + 가변 section table + payload sections.
- Magic `"ARGUS_W\0"` + 3-tier 버저닝 (semver + capability flags + section table 확장점).
- 6개 section tag: META / TOKENIZER / TENSOR_INDEX / WEIGHTS_ADRENO_SOA / WEIGHTS_CUDA_AOS / WEIGHTS_CPU_AOS.
- Hybrid `source_hash` (size + mtime + head/tail 8MB sha256).
- Reader / Writer / Stripper 알고리즘. Repacker는 Phase 5.

**비범위 (v0.x로 미룸)**:
- zstd 압축 section (`SECTION_COMPRESSED` flag bit 2 reserved).
- Image2d_t precomputed payload (Phase 4 디바이스 실측 후 결정).
- Unigram tokenizer (`tokenizer_kind = 1`).
- Cross-asset bundling (LoRA delta, fine-tune snapshot 등) — Mode C.
- 자동 strip / 자동 cache 정책 — 사용자 피드백 후 v0.2에서 결정.

### 1.3b v0.2 컴포넌트 매핑 (multi-dtype variant)

v0.2에서 신규/변경된 컴포넌트:

| 컴포넌트 | v0.1.x 동작 | v0.2 변경 | 대응 spec |
|---------|------------|----------|----------|
| `AufHeader` | format_minor=1, capability_optional bits 0~2 | format_minor=2, bit 3 = `MULTI_DTYPE_VARIANTS` 신설 | ENG-DAT-099 |
| `SectionTable` | 6개 tag 카탈로그 | 변경 없음 (dtype은 section tag로 표현 안 함, Q1=B 결정) | ENG-DAT-097 |
| `TENSOR_INDEX` | 동일 (layer, kind)에 entry 1개 | 동일 (layer, kind)에 dtype별 entry N개 (schema_version=1 보존) | ENG-DAT-097 |
| `META` JSON | 모델 메타데이터만 | `default_dtype` 필드 추가 (capability bit 3 = 1일 때 의무) | ENG-DAT-099 |
| `AufView::lookup_tensor` | `(layer, kind)` 기반 단일 lookup | `(layer, kind, requested_dtype: Option<DType>)` 기반 dispatch + precedence | ENG-DAT-098, ENG-ALG-225 |
| `AufWriter::auf_build` | 단일 dtype 변환 | dtype별 dequant/requant 파이프라인 + 안정 정렬 | ENG-ALG-224 |
| `auf-tool build` CLI | `--variants` | `--variants` + `--dtypes` + `--default-dtype` | ENG-ALG-224, §7.1 |
| `SwapExecutor` | primary/secondary byte slice 받음 | **변경 없음** (Q3 단방향 swap 가정, ENG-DAT-C17) | INV-137~139 |

**lm_head 처리 (Sprint A' 반전)**: lm_head도 layer weight와 동일하게 multi-dtype 후보 entry로 등록 가능하다. multi-dtype AUF에서 lm_head TENSOR_INDEX entry는 dtype별 candidate(예: Q4_0 + F16)로 다중 등장하며, reader dispatch precedence(ENG-ALG-225)도 lm_head에 동일 적용된다. 다만 **INV-135 v2의 layout 의무**(Adreno SOA variant 안에서 AOS 18B/block 강제, image1d_buffer_t 한계로 인함)는 dtype-agnostic으로 유지되며 모든 dtype 후보 entry에 동일하게 적용된다. v0.1.1의 "lm_head Q4_0 single dtype" 결정은 폐기되었으며, layout 강제만 잔존한다 (§2.5b는 v0.1.1 시점 의미를 보존하고, §2.5c가 v0.2 Sprint A'의 일반화된 처리를 다룬다).

---

## 2. 핵심 컴포넌트

### 2.1 AufHeader

**역할**: 파일 식별자, 포맷 버전, source 메타데이터, section table 위치를 256B 고정 영역에 보관.

**책임**:
- Magic 검증 (`"ARGUS_W\0"`).
- format_major/minor/patch 노출 (semver 진화 정책).
- capability_required/optional 노출 (capability flag 진화 정책).
- section_table_offset / payload_start_offset 노출.

**진화 정책**:

| 변경 유형 | 처리 | 예 |
|-----------|------|-----|
| 새 section tag 추가 (additive) | format_minor 증가, 기존 reader는 무시 | v0.1 → v0.2: `WEIGHTS_INTEL_AVX512` 추가 |
| 새 capability flag bit | format_minor 증가, 의미에 따라 required/optional 분류 | v0.1 → v0.2: bit 0 = `SOURCE_HASH_FULL_SHA256` |
| 헤더 필드 의미 변경 | format_major 증가 (breaking) | v0.x → v1.0: `_reserved` 영역의 일부를 신규 필드로 사용 |
| Magic 변경 | 새 포맷 (별도 tool로 처리) | 발생하지 않을 것 (영구 reserved) |

**불변 필드**: 한번 v1.0 stable 선언 이후에는 다음 필드의 byte offset이 변경되지 않는다.
- `magic`, `format_major`, `format_minor`, `format_patch`, `capability_required`, `capability_optional`, `section_count`, `section_table_offset`, `payload_start_offset`.

**가변 영역**: `_reserved [120B]`은 v0.x 기간 동안 자유롭게 사용 가능하지만, v1.0 이후의 변경은 format_major bump 필요.

**인터페이스 (개념)**:

```rust
pub struct AufHeader {
    pub magic: [u8; 8],
    pub format_major: u16,
    pub format_minor: u16,
    pub format_patch: u16,
    pub created_by: [u8; 32],
    pub source_hash: [u8; 32],
    pub source_size: u64,
    pub source_mtime: u64,
    pub capability_required: u64,
    pub capability_optional: u64,
    pub section_count: u32,
    pub section_table_offset: u64,
    pub payload_start_offset: u64,
    // private padding
}

impl AufHeader {
    /// pre: bytes.len() >= 256.
    /// post: magic이 "ARGUS_W\0"가 아니면 Err. _pad/_reserved 무시.
    /// INV-132: format_major > READER_MAX → Err.
    pub fn parse(bytes: &[u8]) -> Result<Self, AufError>;

    pub fn serialize(&self) -> [u8; 256];

    /// post: 알려진 capability bit만 set. 모르는 비트 set 시 Err.
    /// INV-132 매핑.
    pub fn validate_capabilities(&self) -> Result<(), AufError>;
}
```

### 2.2 SectionTable

**역할**: 파일 내 section의 위치/크기/속성을 보관. ELF의 section header table과 유사한 확장 지점.

**책임**:
- 각 section의 `tag`, `offset`, `size`, `flags`, `version` 보관.
- Reader에 lookup API 제공 (tag 기반 검색).
- INV-134 무결성 검증 (overlap 금지, file_size 내, tag unique).

**인터페이스**:

```rust
pub struct SectionEntry {
    pub tag: [u8; 16],          // UTF-8 ASCII, NUL-padded
    pub offset: u64,
    pub size: u64,
    pub flags: u32,             // SECTION_REQUIRED | SECTION_STRIPPABLE | SECTION_COMPRESSED | ...
    pub version: u32,
    // private reserved [u8; 8]
}

pub struct SectionTable {
    entries: Vec<SectionEntry>,
}

impl SectionTable {
    pub fn parse(bytes: &[u8], count: u32) -> Result<Self, AufError>;
    pub fn serialize(&self) -> Vec<u8>;

    /// post: tag 일치하는 entry 반환. NUL trimming 후 비교.
    pub fn find(&self, tag: &str) -> Option<&SectionEntry>;

    /// pre: file_size 인자 = 실제 파일 크기.
    /// INV-134: offset + size <= file_size, no overlap, unique tag.
    pub fn validate(&self, file_size: u64, payload_start: u64) -> Result<(), AufError>;
}
```

**카탈로그 (v0.1)**:

| Tag | required | strippable | 내용 요약 |
|-----|----------|------------|----------|
| META | yes | no | architecture, dims, RoPE config (JSON-in-binary) |
| TOKENIZER | yes | no | vocab + BPE merges + special tokens + chat template |
| TENSOR_INDEX | yes | no | layer → tensor 메타데이터 매핑 |
| WEIGHTS_ADRENO_SOA | no | yes | Adreno SOA: q_buf + d_buf + q_img alignment |
| WEIGHTS_CUDA_AOS | no | yes | CUDA AOS: 18B block + 128B align |
| WEIGHTS_CPU_AOS | no | yes | CPU AOS: 18B block + 64B align |

### 2.3 AufReader

**역할**: AUF 파일을 mmap하고, 자기 backend variant 한정으로 lazy access를 제공.

**책임**:
- `path` + `backend_tag` 입력 → `AufView` 출력.
- mmap-first: payload는 byte slice로만 노출 (zero-copy 의도). META/TOKENIZER만 즉시 파싱.
- INV-132/133/134 검증 — 모두 reader 진입 시 1회 수행.
- 에러 메시지에 진단 정보 + 권장 조치 (INV-132).

**처리 흐름**:

```mermaid
flowchart TD
    A[auf_read path, backend_tag] --> B[mmap open]
    B --> C{magic == ARGUS_W?}
    C -->|No| ZA[Reject: Not an AUF]
    C -->|Yes| D{format_major <= READER_MAX?}
    D -->|No| ZB[Reject: format too new]
    D -->|Yes| E{capability_required<br/>known bits only?}
    E -->|No| ZC[Reject: unknown capability]
    E -->|Yes| F[parse section table]
    F --> G{INV-134:<br/>offsets valid<br/>+ no overlap<br/>+ unique tag?}
    G -->|No| ZD[Reject: corrupt section table]
    G -->|Yes| H{META + TOKENIZER<br/>+ TENSOR_INDEX 존재?}
    H -->|No| ZE[Reject: missing required]
    H -->|Yes| I{WEIGHTS_<backend><br/>존재?}
    I -->|No| ZF[Reject: missing weights<br/>+ repack 안내]
    I -->|Yes| J[parse META JSON<br/>+ TOKENIZER blob<br/>+ TENSOR_INDEX]
    J --> K[Return AufView<br/>weights_payload = mmap slice]

    style ZA fill:#ffcdd2
    style ZB fill:#ffcdd2
    style ZC fill:#ffcdd2
    style ZD fill:#ffcdd2
    style ZE fill:#ffcdd2
    style ZF fill:#ffe0b2
    style K fill:#c8e6c9
```

**인터페이스**:

```rust
pub struct AufView<'a> {
    pub header: AufHeader,
    pub sections: SectionTable,
    pub meta: ModelMeta,                // META JSON 파싱 결과
    pub tokenizer: Tokenizer,           // TOKENIZER blob 파싱 결과
    pub tensor_index: TensorIndex,      // TENSOR_INDEX 파싱 결과
    pub weights_payload: &'a [u8],      // backend variant section의 byte slice
    _mmap: Mmap,                        // lifetime 유지용
}

pub fn auf_read(path: &Path, backend_tag: BackendTag) -> Result<AufView, AufError>;
```

**예외 처리** (모든 케이스 panic 없이 `Err` 반환):

| 케이스 | Err variant | 메시지 |
|--------|-------------|--------|
| 파일 < 256 B | `AufError::Truncated` | "AUF file too small (header < 256 B)" |
| Magic 불일치 | `AufError::NotAuf` | "Not an AUF file (magic mismatch)" |
| format_major > READER_MAX | `AufError::FormatTooNew { found, max }` | "AUF format_major=2 but reader supports up to 1. Update llm_rs2." |
| Unknown required capability | `AufError::UnknownCapability { bits }` | "AUF requires capability bit 5 (zstd compression) but reader does not support it" |
| Section overlap | `AufError::SectionOverlap { tag_a, tag_b }` | "Sections X and Y overlap" |
| Section out of file | `AufError::SectionOutOfBounds { tag }` | "Section X exceeds file size" |
| Required section 누락 | `AufError::RequiredMissing { tag }` | "META section missing — file is not a valid AUF" |
| 자기 backend WEIGHTS_* 누락 | `AufError::WeightsMissing { tag, repack_hint }` | "WEIGHTS_ADRENO_SOA missing. Run 'auf-tool repack ...'" |

### 2.4 AufWriter

**역할**: GGUF 입력 → variant 변환 → AUF 파일 atomic write.

**책임**:
- GGUF metadata + tokenizer + 모든 layer tensor를 읽음.
- 요청된 variants(`--variants WEIGHTS_ADRENO_SOA WEIGHTS_CUDA_AOS WEIGHTS_CPU_AOS`)별로 weight payload 변환:
  - `WEIGHTS_ADRENO_SOA`: Q/K permute → SOA 분리 → q_img alignment.
  - `WEIGHTS_CUDA_AOS`: Q/K permute → 128B align padding.
  - `WEIGHTS_CPU_AOS`: Q/K permute → 64B align padding.
- META JSON + TOKENIZER blob + TENSOR_INDEX 직렬화.
- Section layout 결정 (cursor 기반 단조 진행, 64KB align for `WEIGHTS_*`).
- Header finalize → 단일 파일 atomic write (`tempfile + rename`).

**처리 흐름**:

```mermaid
flowchart TD
    A[auf_build gguf, out, variants] --> B[parse GGUF]
    B --> C[compute hybrid source_hash]
    C --> D[serialize META JSON]
    D --> E[serialize TOKENIZER blob]
    E --> F[for each variant:<br/>convert weights]
    F --> G[build TENSOR_INDEX<br/>with variant_offsets]
    G --> H[plan section layout:<br/>cursor base + alignment]
    H --> I[allocate header<br/>+ section table]
    I --> J[write tempfile:<br/>header + table + payloads]
    J --> K[fsync]
    K --> L[atomic rename to out_path]

    style L fill:#c8e6c9
```

**인터페이스**:

```rust
pub struct VariantTag(pub &'static str);   // "WEIGHTS_ADRENO_SOA" 등

pub struct AufBuildOptions {
    pub variants: Vec<VariantTag>,
    pub created_by: String,                // 보통 "llm_rs2 v{CARGO_PKG_VERSION}"
    pub source_path: PathBuf,
    pub output_path: PathBuf,
}

/// pre: source_path는 valid GGUF.
/// pre: variants는 비어있지 않음.
/// post: output_path에 valid AUF 생성. atomic rename. partial state 외부 노출 금지 (ENG-ALG-C11).
/// post: source_hash는 hybrid (size + mtime + head/tail 8MB sha256).
pub fn auf_build(opts: &AufBuildOptions) -> Result<(), AufError>;
```

**Variant 변환 모듈**: backend-specific. 코드 위치는 implementation 단계에서 결정 (예: `engine/src/auf/variant_adreno_soa.rs`). 각 변환 함수는 GGUF의 layer tensor → variant payload byte vector를 생성하며, 같은 입력에 대해 deterministic을 보장해야 한다 (실험 재현성, source_hash 의미 유지).

### 2.5 AufStripper

**역할**: 기존 AUF에서 일부 strippable section을 제거하여 새 AUF를 만든다 (in-place atomic replace).

**책임**:
- `--keep <tags>` 또는 `--remove <tags>` 입력 받음.
- `SECTION_REQUIRED` 비트 set인 section은 절대 제거 거부.
- `SECTION_STRIPPABLE` 비트 set이 아닌 section은 제거 거부.
- 유지할 section만으로 새 AUF 파일 build (writer 경로 재사용 + source_hash/created_by 보존).
- 기본 백업 생성 (`<name>.auf.bak`), `--no-backup`으로 비활성화.
- atomic rename으로 in-place 교체.

**처리 흐름**:

```mermaid
flowchart TD
    A[auf_strip in_path, keep, no_backup] --> B[auf_read in_path<br/>backend_check 우회]
    B --> C{모든 keep section이<br/>SECTION_STRIPPABLE 또는<br/>이미 SECTION_REQUIRED?}
    C -->|No| Z[Reject: cannot strip required<br/>또는 not strippable]
    C -->|Yes| D{no_backup?}
    D -->|No| E[copy in_path to .auf.bak]
    D -->|Yes| F[skip backup]
    E --> G[rewrite tempfile<br/>with subset sections]
    F --> G
    G --> H[update capability_optional<br/>clear removed bits]
    H --> I[atomic rename tempfile to in_path]

    style I fill:#c8e6c9
    style Z fill:#ffcdd2
```

**인터페이스**:

```rust
pub struct AufStripOptions {
    pub keep_tags: Vec<String>,
    pub no_backup: bool,
}

/// pre: in_path는 valid AUF.
/// pre: keep_tags의 모든 tag는 SECTION_STRIPPABLE이거나 SECTION_REQUIRED 중 하나.
/// post: in_path가 새 AUF로 교체됨. atomic rename. source_hash 보존 (향후 repack 가능).
/// post: capability_optional에서 제거된 variant bit 클리어.
pub fn auf_strip(in_path: &Path, opts: &AufStripOptions) -> Result<(), AufError>;
```

**중요 — Strip은 truncation이 아니다**:

```mermaid
flowchart LR
    subgraph BEFORE["Before strip (4 sections)"]
        H1[Header 256B]
        T1[Section Table<br/>4 entries × 48B]
        P1[Padding to 64KB]
        S1A[META]
        S1B[TOKENIZER]
        S1C[TENSOR_INDEX]
        S1D[WEIGHTS_ADRENO_SOA]
        S1E[WEIGHTS_CUDA_AOS]
        S1F[WEIGHTS_CPU_AOS]
        H1 --> T1 --> P1 --> S1A --> S1B --> S1C --> S1D --> S1E --> S1F
    end

    subgraph AFTER["After strip (Adreno only)"]
        H2[Header 256B<br/>section_count=4<br/>capability_optional 갱신]
        T2[Section Table<br/>4 entries × 48B<br/>NEW offsets]
        P2[Padding to 64KB]
        S2A[META]
        S2B[TOKENIZER]
        S2C[TENSOR_INDEX]
        S2D[WEIGHTS_ADRENO_SOA]
        H2 --> T2 --> P2 --> S2A --> S2B --> S2C --> S2D
    end

    BEFORE -->|rewrite| AFTER
```

단순히 file end를 truncate하면 section table에는 여전히 6 entries가 있고, 마지막 두 entries의 offset은 file_size를 넘어가게 되어 INV-134 위반. 따라서 **반드시 rewrite**.

### 2.5b lm_head Q4_0 사전 변환 (v0.1.1, Sprint G-1)

**역할**: AUF build 시점에 GGUF의 F16 lm_head를 Q4_0으로 사전 변환하여 backend variant section에 동봉. Engine load 시점 ~1.4 s runtime quantize 비용 제거.

**설계 결정 근거 (Sprint G-1-A)**:

```mermaid
flowchart LR
    A[GGUF lm_head<br/>F16, ~525 MB] --> B{build option<br/>--include-lm-head}
    B -->|on / auto| C[F32 dequantize<br/>F16 → F32]
    B -->|off| D[skip<br/>v0.1.0 호환]
    C --> E[Q4_0 quantize<br/>quantize_q4_0]
    E --> F{target backend<br/>variant}
    F -->|ADRENO_SOA| G[convert_aos_to_soa<br/>q_buf + d_buf + q_img]
    F -->|CUDA_AOS| H[18B block<br/>+ 128B align]
    F -->|CPU_AOS| I[18B block<br/>+ 64B align]
    G --> J[WEIGHTS_ADRENO_SOA<br/>section 내부<br/>layer weight와 동일 layout]
    H --> K[WEIGHTS_CUDA_AOS<br/>section 내부]
    I --> L[WEIGHTS_CPU_AOS<br/>section 내부]
    J --> M[TENSOR_INDEX entry<br/>kind=11 lm_head, dtype=Q4_0<br/>variant_offsets]
    K --> M
    L --> M
    M --> N[capability_optional<br/>bit 2 set]

    style D fill:#ffe0b2
    style M fill:#c8e6c9
    style N fill:#c8e6c9
```

**핵심 결정**:

1. **Section type 신설하지 않음**. lm_head Q4_0 payload는 기존 `WEIGHTS_<backend>` section 내부에 layer weight와 동일한 layout으로 동봉. 근거: lm_head는 transformer.rs `prepare_noshuffle_buffers()`(line 914-917)에서 layer weight와 **동일한 SOA 변환 함수**를 호출하므로 layout이 layer weight와 byte-level 동일하다. 별도 section type을 두면 strip / capability / reader switch logic이 두 배로 복잡해진다.

2. **TENSOR_INDEX entry 재사용**. spec ENG-DAT-096.8에 이미 `kind = 11(lm_head)` enum이 정의되어 있다. 이 entry의 `dtype = Q4_0`, `shape = [vocab_size, hidden_dim]`(GGUF 원본 그대로), `variant_offsets[i]`가 backend variant section 내부의 lm_head payload offset을 가리킨다. cross-layer tensor의 `layer_idx = u32::MAX` 규칙은 그대로 적용.

3. **lm_head는 모든 variant에서 AOS 18B block layout** (G-1-F update, INV-135 v2). `WEIGHTS_ADRENO_SOA` section 내부에서도 lm_head는 SOA 변환을 적용하지 않고 raw GGUF Q4_0 bytes 그대로 동봉한다. **이유**: lm_head q_buf 크기(`vocab × hidden / 8` texels, Llama 3.2 1B에서 32M texels)가 OpenCL `CL_DEVICE_IMAGE_MAX_BUFFER_SIZE` 한계를 거의 모든 디바이스에서 초과하여 `image1d_buffer_t` 생성이 실패하고, 빠른 SOA GEMV path(`m=1` decode)가 standard GEMV로 fall through하면서 SOA의 `d_buf`만 노출된 cl_mem을 AOS layout으로 잘못 해석 → garbage 출력 (Sprint G-1-F 디바이스 측정에서 확인됨). 따라서 lm_head는 image 한계로 인해 **빠른 SOA path를 사용할 수 없으며**, AOS 동봉이 정확성과 단순성 모두 충족시킨다. layer weight (`vocab × hidden`보다 훨씬 작음)는 SOA 변환을 그대로 적용. `WEIGHTS_CUDA_AOS` / `WEIGHTS_CPU_AOS`는 처음부터 AOS이므로 lm_head 처리는 일반 weight와 동일.

4. **capability_optional bit 2 = `LM_HEAD_PRECOMPUTED_Q4_0`** 신설. reader는 bit 미인식 시 ignore (`capability_optional`의 의미상). 후방 호환 보장.

5. **source_hash 재사용**. lm_head 단일 tensor 별도 hash 없음. AUF 헤더의 hybrid `source_hash`(GGUF 전체)가 일치하면 lm_head Q4_0 payload도 신뢰. v0.1 채택 결정에서 이미 hybrid hash가 부분 변조에 약함을 인지했고(spec §3.22.6 근거), lm_head는 GGUF tail 8 MB에 거의 항상 포함됨.

**인터페이스 (개념)**:

```rust
// AUF reader (Sprint G-1-C 산출)
pub struct AufView<'a> {
    // 기존 필드들...
    pub lm_head_precomputed_q4_0: bool,   // capability_optional bit 2 검사 결과
}

impl<'a> AufView<'a> {
    /// post: capability_optional bit 2 = 1이고 TENSOR_INDEX에 kind=lm_head + dtype=Q4_0
    /// entry가 존재하면 backend variant section 내부의 byte slice 반환.
    /// 그 외(bit 0 또는 entry 미존재)는 None — caller는 runtime quantize fallback.
    pub fn lm_head_q4_0_payload(&self) -> Option<LmHeadPayload<'a>>;
}

pub struct LmHeadPayload<'a> {
    pub shape: [u64; 2],          // [vocab_size, hidden_dim]
    pub alignment: u64,
    pub bytes: &'a [u8],          // backend variant에 따라 SOA 또는 AOS 직렬화
    pub variant_tag: VariantTag,  // 호출자가 SOA vs AOS 분기에 사용
}
```

**model load 분기 (transformer.rs Sprint G-1-D 산출)**:

```mermaid
flowchart TD
    A[model load 진입] --> B{secondary_source<br/>== Some?}
    B -->|No| C[기존 quantize_lm_head_to_q4_0<br/>auto 분기 그대로]
    B -->|Yes, AUF| D{auf_view<br/>.lm_head_q4_0_payload<br/>= Some?}
    D -->|Yes| E[AUF section에서 직접 매핑<br/>NoshuffleWeightBuffer 또는<br/>SharedBuffer Q4_0<br/>runtime quantize SKIP]
    D -->|No| F[runtime quantize<br/>~1.4s fallback]
    C --> G[lm_head 준비 완료]
    E --> G
    F --> G

    style E fill:#c8e6c9
    style F fill:#ffe0b2
```

**예외 처리 / fallback**:

| 케이스 | 동작 |
|--------|------|
| AUF에 capability bit 2 = 0 | runtime quantize fallback (Sprint F 동작 그대로) |
| AUF에 bit 2 = 1이지만 `kind=lm_head` entry shape이 model config와 불일치 | reject + 명시 에러. AUF가 다른 model용. |
| AUF에 bit 2 = 1이지만 `dtype != Q4_0` | reject + 명시 에러. capability bit 의미 위반. |
| `--quantize-lm-head q4_0` 강제 (debug) | AUF entry 무시 + runtime quantize 강제 (회귀 비교용) |
| `--quantize-lm-head none` | AUF entry도 사용하지 않고 lm_head F16 유지 |

**결정성 요구사항 (ENG-DAT-096.13)**:

- 동일 GGUF + 동일 build option + 동일 host → byte-level 동일 AUF 출력.
- `quantize_q4_0`은 round-half-to-even로 결정성 보장.
- SOA 변환은 host 환경에서 deterministic kernel(reference CPU 또는 host OpenCL)로 수행. 디바이스간 portability는 v0.1.x 범위 외이며 v1.0 conformance에서 별도 검증.

**메모리/디스크 영향 (Llama 3.2 1B 기준)**:

- F16 lm_head 보관 시 (v0.1.0): 525 MB (per backend variant).
- Q4_0 lm_head 보관 시 (v0.1.1): 148 MB (per backend variant).
- 디스크 절감: 377 MB / variant. 3-variant build 시 ~1.1 GB 절감.
- model load 시간 절감: ~1.4 s (Galaxy S25 실측, Sprint F).

### 2.5c Multi-dtype Variant (v0.2)

**역할**: AUF build 시점에 동일 (`backend`, `layer_idx`, `kind`) 쌍에 대해 여러 dtype 후보(예: Q4_0 + F16)를 한 파일에 동시 보관. dynamic weight swap의 secondary dtype payload를 self-contained로 보관하여 deploy 시 GGUF 의존을 제거. **Sprint A' 반전 이후 lm_head(`kind = 11`)도 multi-dtype 후보 그룹에 포함**되어, layer weight와 동일한 dispatch 흐름을 거친다 (ENG-DAT-C16 갱신본). 단, Adreno SOA variant section 내부에서 lm_head sub-payload는 dtype에 무관하게 AOS 18B/block layout으로 동봉되며, 이는 INV-135 v2 layout 의무가 dtype-agnostic으로 유지되기 때문이다.

**핵심 결정 (이미 채택, 재검토 X)**:

| 결정 | 채택 | 근거 요약 |
|------|------|----------|
| **Q1=B**: section tag에 dtype suffix 안 넣음 | TENSOR_INDEX entry-level dtype 필드 활용. SECTION_TAG_SIZE=24 / SectionEntry 48B layout 보존 | section tag 카탈로그 폭발 방지. v0.1.x reader가 tag를 그대로 파싱 가능 (forward compat). |
| **Q2 (Sprint A' 반전)**: lm_head도 multi-dtype 후보 entry 적용 | lm_head는 layer weight와 동일하게 dtype별 candidate(Q4_0 + F16 등) 다중 entry 등록 가능. INV-135 v2 **layout 의무만 dtype-agnostic으로 유지** | lm_head는 image1d_buffer_t 한계로 SOA path 사용 불가 (G-1-F)이지만, AOS layout이라면 dtype 다양성은 정확성에 영향이 없다. self-contained dynamic swap이 lm_head에도 적용 가능하도록 single-dtype 의무를 폐기. swap 대상 여부와는 별개. |
| **Q3**: 단방향 swap 가정 | SwapExecutor 인터페이스 변경 없음 | `primary → secondary` 단일 방향 흐름 유지. AUF 내부에 candidate 양쪽 보관해도 runtime swap은 unchanged. |
| **Q5=B**: capability_optional bit 3 신설 + format_minor 1→2 | 의미적으로 minor change (additive 의미 확장). format_major=0 그대로 | v0.1.x reader가 bit 3을 무시하고 first-match로 정상 진입 (INV-132 호환). |
| TensorIndex schema_version | 1 그대로 유지 | v0.1.x reader 양방향 호환 위해 schema bump 금지. entry 의미만 호환적 확장. |

**처리 흐름 (writer)**:

```mermaid
flowchart LR
    A[GGUF 원본<br/>source_dtype = Q4_0 or F16] --> B{candidate_dtypes<br/>요청}
    B -->|--dtypes Q4_0,F16| C{각 dtype별<br/>변환 파이프라인}
    C -->|src=Q4_0, tgt=Q4_0| D[identity]
    C -->|src=Q4_0, tgt=F16| E[dequant Q4_0→F32<br/>→ cast to F16]
    C -->|src=F16, tgt=Q4_0| F[dequant F16→F32<br/>→ quantize_q4_0]
    C -->|src=F16, tgt=F16| G[identity]
    D --> H[per-variant convert<br/>SOA / CUDA AOS / CPU AOS]
    E --> H
    F --> H
    G --> H
    H --> I[backend variant section<br/>내부 sub-payload 배치<br/>cursor 기반 단조]
    I --> J[TENSOR_INDEX entries<br/>dtype별 N개 per layer×kind<br/>안정 정렬<br/>layer ASC, kind ASC,<br/>is_default DESC, dtype ASC]
    J --> K[META JSON<br/>+ default_dtype 키 append]
    K --> L[Header<br/>format_minor=2<br/>capability_optional bit 3=1]
    L --> M[atomic write]

    style M fill:#c8e6c9
    style J fill:#bbdefb
```

**처리 흐름 (reader dispatch)**:

```mermaid
flowchart TD
    A[AufView::lookup_tensor<br/>layer, kind, requested_dtype] --> B[filter TENSOR_INDEX<br/>by layer + kind]
    B --> C{candidates<br/>비어있음?}
    C -->|Yes| ZA[Err TensorNotFound]
    C -->|No| D{requested_dtype<br/>== Some?}
    D -->|Yes| E{matching entry<br/>존재?}
    E -->|Yes| F[return entry]
    E -->|No| ZB[Err DtypeNotAvailable<br/>+ available list]
    D -->|No| G{capability bit 3<br/>= 1 AND<br/>multi-dtype 모드?}
    G -->|Yes| H{META.default_dtype<br/>matching entry?}
    H -->|Yes| F
    H -->|No| I[fall-through]
    G -->|No| I
    I --> J[return candidates 0<br/>first-match by sort order]

    style F fill:#c8e6c9
    style ZA fill:#ffcdd2
    style ZB fill:#ffcdd2
```

**v0.1.x reader 호환 동작 (중요)**:

```mermaid
flowchart LR
    A[v0.1.x reader] --> B[v0.2 AUF 진입]
    B --> C{capability_optional<br/>bit 3 인식?}
    C -->|미인식| D[ignore<br/>optional이므로 reject 아님<br/>INV-132 호환]
    D --> E{TENSOR_INDEX<br/>다중 entry per layer,kind?}
    E -->|Yes| F[first-match 규칙 사용<br/>requested_dtype API 없음]
    F --> G{writer 안정 정렬<br/>덕분에 첫 entry는<br/>default_dtype}
    G --> H[default_dtype payload<br/>단일 모드로 안전 동작<br/>INV-138 writer 의무]

    style H fill:#c8e6c9
```

**INV-138의 writer 정렬 의무**가 v0.1.x reader 호환의 핵심이다. writer가 entries를 (`layer_idx` ASC, `kind` ASC, `is_default` DESC, `dtype` ASC)로 안정 정렬하면, v0.1.x reader의 first-match가 자연스럽게 default_dtype과 일치하게 된다.

**인터페이스 (개념, ENG-ALG-225 매핑)**:

```rust
pub struct TensorIndexEntry {
    pub layer_idx: u32,
    pub kind: u32,
    pub dtype: u32,             // v0.2: 동일 (layer_idx, kind)에 dtype별로 여러 entry
    pub shape_rank: u32,
    pub shape: Vec<u64>,
    pub alignment: u64,
    pub variant_offsets: Vec<u64>,
    pub variant_sizes: Vec<u64>,
}

impl<'a> AufView<'a> {
    /// pre: layer_idx, kind는 valid.
    /// post (v0.2): requested_dtype 명시 시 해당 dtype entry 반환,
    ///              미명시 시 META.default_dtype → first-match 순.
    /// post (v0.1.x reader compat): requested_dtype 매개변수 자체가 없으므로 항상 first-match.
    /// INV-137: 동일 (layer, kind)의 모든 dtype entry는 동일 shape.
    /// INV-138: writer는 entries를 안정 정렬하여 default_dtype이 그룹 첫 번째.
    /// INV-139: capability_optional bit 3 의미 + v0.1.x ↔ v0.2 호환.
    pub fn lookup_tensor(&self, layer_idx: u32, kind: u32,
                          requested_dtype: Option<DType>) -> Result<&TensorIndexEntry, AufError>;

    /// post: capability_optional bit 3 검사 결과
    pub fn multi_dtype_enabled(&self) -> bool;
}

pub struct ModelMeta {
    // 기존 필드들...
    /// v0.2: capability bit 3 = 1일 때 의무. None이면 single-dtype 모드.
    pub default_dtype: Option<DType>,
}
```

**Variant 변환 모듈 확장 (ENG-ALG-224)**: `convert_to_<variant>_for_dtype(tensors, dtype)` 형태로 dtype 매개변수를 추가한다. 기존 `convert_to_adreno_soa(layers)` 등은 v0.1.x 호환을 위해 wrapper로 유지하거나 deprecate한다 (implementation 단계에서 결정).

**예외 처리 / fallback**:

| 케이스 | 동작 |
|--------|------|
| AUF에 capability bit 3 = 0 (single-dtype) | v0.1.x 동작 그대로. requested_dtype 무시 또는 single dtype 일치 검증. |
| capability bit 3 = 1이지만 META에 `default_dtype` 부재 | reject (`AufError::MalformedMeta`). INV-138 위반. |
| `requested_dtype` 명시했지만 해당 entry 부재 | reject (`AufError::DtypeNotAvailable`) + available list. |
| 동일 (layer, kind, dtype) 쌍에 entry가 2개 이상 | reject (`AufError::DuplicateTensorEntry`). writer build 시점에 자동 충족. |
| 동일 (layer, kind) 다중 dtype shape 불일치 | reject (`AufError::ShapeMismatch`). INV-137 위반. |
| Adreno SOA variant 안에서 lm_head sub-payload가 SOA layout으로 동봉됨 | reject (`AufError::LmHeadSoaForbidden`). INV-135 v2 layout 의무 + ENG-DAT-C16 갱신본 위반. dtype에 무관하게 AOS 강제. |
| lm_head 다중 dtype entry shape 불일치 | reject (`AufError::ShapeMismatch`). INV-137이 lm_head 포함하도록 갱신됨 (Sprint A'). |
| v0.1.x reader가 v0.2 AUF 만남 | optional bit 3 무시 + first-match → default_dtype 단일 모드 (INV-139). |

**메모리/디스크 영향 (Llama 3.2 1B 기준, dual-dtype 예시)**:

- Q4_0 layer weights: ~700 MB / variant.
- F16 layer weights: ~2.6 GB / variant (4x 크기).
- lm_head Q4_0 AOS: ~148 MB / variant.
- lm_head F16 AOS: ~525 MB / variant (Sprint A' 반전: F16 후보가 추가될 때 동봉됨).
- Q4_0 + F16 dual-dtype 1 variant (lm_head dtype별 동봉 포함): ~3.3 GB + lm_head F16 추가 ~+260 MB ≈ ~3.6 GB.
- Q4_0 + F16 dual-dtype 3 variants (Adreno + CUDA + CPU): ~10.7 GB. Strip 후 1 variant: ~3.6 GB.
- 비교: GGUF Q4_0 (~700 MB) + GGUF F16 (~2.6 GB) 양쪽 보관 시 디스크 사용량과 유사. AUF의 우위는 self-contained 단일 파일 + dynamic swap secondary 즉시 사용 가능. lm_head F16 추가분 ~+260 MB는 lm_head dynamic dtype 선택의 비용.

### 2.6 auf-tool CLI

**역할**: AUF 자산을 만들고 검사하고 수정하는 사용자 인터페이스.

**위치 결정 (Architect 권고)**: `engine/src/bin/auf_tool.rs`로 배치하여 cargo workspace 단순성 유지. 별도 crate로 분리하면 GGUF parser, OpenCL convert_aos_to_soa 등 engine 내부 함수에 의존하기 위해 pub API 노출이 추가로 필요하며, 이는 engine crate의 표면적을 늘린다. binary 격리 필요성은 v1.0 이후 재평가.

**서브커맨드**:

| 명령 | 동작 | Phase 3.7 필수 |
|------|------|---------------|
| `auf-tool build --input <gguf> --output <auf> --variants <list>` | GGUF → AUF 빌드. variants 예: `WEIGHTS_ADRENO_SOA WEIGHTS_CUDA_AOS WEIGHTS_CPU_AOS` 또는 `all` | yes |
| `auf-tool info <auf>` | header, sections, sizes, capability flags 출력 | yes |
| `auf-tool strip --keep <tags> [--no-backup] <auf>` | 지정 section만 남기고 나머지 strip. atomic rename. | yes |
| `auf-tool verify [--source <gguf>] <auf>` | 무결성 검증 (INV-132/133/134). `--source` 제공 시 source_hash 일치도 비교 | yes (선택적) |
| `auf-tool repack --input <stripped> --source <gguf> --add <tag> --output <out>` | stripped AUF에 누락된 variant section 재생성 추가 | **Phase 5로 미룸** |

**예시 사용 시나리오** (`docs/auf_tool_guide.md`에 상세):

```sh
# 1) 워크스테이션에서 모든 variants 포함 build
auf-tool build --input model.gguf --output model.auf --variants all

# 2) Galaxy S25용 strip
auf-tool strip --keep META TOKENIZER TENSOR_INDEX WEIGHTS_ADRENO_SOA model.auf

# 3) 디바이스로 전송
adb push model.auf /data/local/tmp/

# 4) Engine에서 사용
generate --secondary-source /data/local/tmp/model.auf ...
```

---

## 3. 운영 시나리오

### 3.1 워크스테이션 빌드 → 배포 워크플로우

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant Tool as auf-tool
    participant WS as Workstation FS
    participant Adb as adb
    participant Phone as Galaxy S25

    Dev->>Tool: auf-tool build --variants all
    Tool->>WS: read model.gguf
    Tool->>Tool: compute source_hash (hybrid)
    Tool->>Tool: convert SOA / CUDA AOS / CPU AOS
    Tool->>WS: write model.auf (atomic)

    Dev->>Tool: auf-tool strip --keep WEIGHTS_ADRENO_SOA
    Tool->>WS: backup model.auf.bak
    Tool->>WS: rewrite model.auf

    Dev->>Adb: adb push model.auf
    Adb->>Phone: /data/local/tmp/model.auf

    Dev->>Phone: generate --secondary-source model.auf
    Phone->>Phone: auf_read → mmap → SwapExecutor uses pre-converted SOA
    Phone-->>Dev: 정상 추론 + zero conversion overhead
```

### 3.2 모바일 배포 시 strip 효과

Llama 3.2 1B Q4_0 기준 대략적 크기 (실측 예상):

| 구성 | 파일 크기 |
|------|----------|
| GGUF 원본 (Q4_0) | ~700 MB |
| AUF (all variants) | ~2.1 GB (3 variants × ~700 MB + meta overhead) |
| AUF (Adreno only after strip) | ~700 MB + meta + 64KB align |
| AUF (CPU only after strip) | ~700 MB + meta + 64KB align |

배포 시점에는 strip된 AUF만 디바이스로 전송하므로 전송량은 GGUF와 거의 동일. variant 변환 비용(SOA reorder, alignment)은 워크스테이션에서 1회 수행.

### 3.3 HF Hub 스타일 (미래 v0.x)

향후 HF Hub 같은 모델 호스팅 사이트에서 multi-variant AUF를 직접 공개할 수 있다. 사용자는 디바이스에 맞는 strip된 변종을 다운받거나, full multi-variant AUF를 받은 뒤 로컬 strip한다. v0.1 spec 자체는 이 시나리오를 막지 않으나, 명시적 지원 도구는 v1.0 이후 추가.

---

## 4. 진화 Roadmap

### 4.1 v0.1 (현재)

- 본 spec.
- 6개 section tag.
- Mode B 단일.
- Hybrid source_hash.
- Reader/Writer/Stripper 구현. Repacker는 Phase 5로 미룸.

### 4.2 v0.x 실험적 기간

`format_major = 0`. forward/backward compat 보장 안 함. 이 기간 동안 가능한 변경:

- 새 section tag 추가 (예: `WEIGHTS_INTEL_AVX512`, `WEIGHTS_QNN_HTP`).
- Capability flag bit 추가 (예: bit 0 = `SOURCE_HASH_FULL_SHA256`, bit 1 = `ZSTD_COMPRESSION`).
- TENSOR_INDEX schema 확장 (예: per-tensor quantization params 보존).
- TOKENIZER blob의 unigram 지원 (`tokenizer_kind = 1`).

### 4.3 v1.0 stable 선언 조건

1. **Phase 4 디바이스 실측 통과**: Galaxy S25에서 INV-122 (logit NMSE ≤ 0.01, top-5 overlap ≥ 0.9, top-1 match ≥ 0.95) 통과.
2. **Multi-variant 검증**: 최소 1개 추가 디바이스 변종(예: CUDA AOS Jetson 또는 Intel CPU AVX2)이 추가되어 multi-variant 시나리오가 cross-device로 동작 검증됨.
3. **Tool stable**: `auf-tool` CLI가 인터페이스 변경 없이 사용된 기간 ≥ 4주.

### 4.4 v1.0 이후 호환성 규칙

| 변경 유형 | 허용 | 절차 |
|-----------|------|------|
| 새 section tag (additive) | yes | format_minor++. 기존 reader는 무시. |
| 새 capability_optional bit | yes | format_minor++. 기존 reader는 무시. |
| 새 capability_required bit | yes (조건부) | format_minor++. 기존 reader는 reject (의도된 동작). |
| 헤더 reserved → 신규 필드 | yes | format_minor++. additive 의미 유지. |
| 헤더 필드 의미 변경 | no | format_major++. migration tool 제공 의무. |
| Magic 변경 | no | 별도 포맷으로 처리 (사실상 발생 안 함). |

### 4.5 호환성 매트릭스 (v0.1.0 / v0.1.1 / v0.2 reader × AUF)

| Reader \ AUF | v0.1.0 (single-dtype) | v0.1.1 (lm_head Q4_0 동봉) | v0.2 (multi-dtype) |
|--------------|----------------------|---------------------------|---------------------|
| **v0.1.0 reader** | OK | OK (bit 2 optional, ignore + runtime quantize fallback) | OK (bit 3 optional, ignore + first-match → default_dtype 단일 모드, INV-139) |
| **v0.1.1 reader** | OK (bit 2 = 0, runtime fallback) | OK (lm_head Q4_0 직접 매핑) | OK (bit 3 optional, ignore + first-match → default_dtype 단일 모드, INV-139) |
| **v0.2 reader** | OK (single-dtype 인식) | OK (single-dtype + lm_head precomputed) | OK (multi-dtype dispatch + lm_head precomputed) |

**호환성 보장의 핵심**:
- v0.1.x reader가 v0.2 AUF를 안전하게 사용 가능한 이유는 (1) `capability_optional` bit 3은 미인식 시 reject 사유 아님 (INV-132), (2) v0.2 writer가 INV-138 안정 정렬 의무를 지켜 first-match가 default_dtype과 일치하도록 보장, (3) META의 `default_dtype` 필드는 unknown JSON key로 v0.1.x reader가 무시 (단, JSON parser가 unknown_fields = ignore 동작이어야 함 — 본 의무는 implementation 검증 항목).
- v0.2 writer가 v0.1.x reader 호환 출력을 만들려면 `--dtypes Q4_0` 단독 빌드 (single-dtype, bit 3 = 0)를 사용한다. 이는 v0.1.1과 byte-level 동일한 출력 (lm_head Q4_0 동봉 옵션은 별도).
- **lm_head 호환 (Sprint A' 반전 후)**: v0.1.x reader가 v0.2 mixed.auf를 만났을 때, lm_head의 dtype별 candidate entry 중 first-match가 자연스럽게 `default_dtype`의 lm_head entry와 일치한다 (writer 안정 정렬 의무). 즉 v0.1.x reader는 multi-dtype lm_head AUF를 default_dtype 단일 lm_head 모드로 안전하게 사용한다. 단, v0.1.x reader는 INV-135 v2의 layout 의무(AOS 강제)를 알지 못하므로, writer는 default_dtype의 lm_head entry를 v0.1.1 호환 layout(Q4_0 AOS)으로 채워두는 것을 권장한다 — `default_dtype = Q4_0`이면 v0.1.1과 byte-level 동일.

### 4.6 미래 capability 후보

| 후보 | 용도 | bit position 권고 |
|------|------|-------------------|
| `SOURCE_HASH_FULL_SHA256` | source_hash가 hybrid 대신 full SHA256 | optional bit 0 |
| `ZSTD_COMPRESSION` | section payload zstd 압축 | required bit 0 |
| `IMAGE2D_PRECOMPUTED` | Adreno q_img가 device-specific texture format으로 사전 인코딩 | optional bit 1 |
| `LM_HEAD_PRECOMPUTED_Q4_0` | lm_head 사전 Q4_0 양자화 (v0.1.1, **할당 완료**) | optional bit 2 |
| `MULTI_DTYPE_VARIANTS` | 동일 (layer, kind)에 dtype별 다중 entry (v0.2, **할당 완료**) | optional bit 3 |
| `LORA_DELTAS` | section tag `LORA_<name>` 도입, multi-asset bundle | required bit 1 (Mode C) |
| `UNIGRAM_TOKENIZER` | TOKENIZER가 SentencePiece unigram 지원 | optional bit 4 (구 권고 bit 3에서 이동) |
| `BIDIRECTIONAL_SWAP` | 양방향 swap 가정 (v0.2 단방향 가정의 확장) | optional bit 5 (예약) |

---

## 5. 다이어그램 모음

### 5.1 Multi-variant AUF 파일 layout (v0.2 multi-dtype)

```mermaid
flowchart TB
    subgraph FILE["model.auf (multi-variant + multi-dtype, post-build)"]
        H["[0..256) Header<br/>magic ARGUS_W<br/>format_major=0, format_minor=2<br/>section_count=6<br/>capability_optional bit 2=1 (lm_head)<br/>capability_optional bit 3=1 (multi-dtype)"]
        T["[256..544) SectionTable<br/>6 entries × 48B"]
        P0["[544..65536) Padding<br/>0x00 fill"]
        S1["META<br/>JSON ~2 KiB<br/>+ default_dtype = Q4_0<br/>required, not strippable"]
        S2["TOKENIZER<br/>~6 MiB BPE blob<br/>required, not strippable"]
        S3["TENSOR_INDEX<br/>~128 KiB tensor metadata<br/>per layer×kind: dtype별 N entries<br/>안정 정렬 default first<br/>kind=11 lm_head: dtype별 entry (Sprint A')<br/>required, not strippable"]
        S4["WEIGHTS_ADRENO_SOA<br/>~3.6 GB strippable<br/>= Q4_0 sub-payload (~700 MB)<br/>+ F16 sub-payload (~2.6 GB)<br/>+ lm_head Q4_0 AOS (~148 MB)<br/>+ lm_head F16 AOS (~525 MB) — INV-135 v2 dtype-agnostic"]
        S5["WEIGHTS_CUDA_AOS<br/>~3.6 GB strippable<br/>= Q4_0 + F16 sub-payloads<br/>+ lm_head dtype별 AOS sub-payloads"]
        S6["WEIGHTS_CPU_AOS<br/>~3.6 GB strippable<br/>= Q4_0 + F16 sub-payloads<br/>+ lm_head dtype별 AOS sub-payloads"]
        H --> T --> P0 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6
    end

    style S3 fill:#fff3e0
    style S4 fill:#c8e6c9
    style S5 fill:#bbdefb
    style S6 fill:#ffe0b2
```

**dtype별 sub-payload 배치 (단일 variant 내부, Sprint A' 일반화)**:

```mermaid
flowchart LR
    subgraph SECTION["WEIGHTS_ADRENO_SOA section payload"]
        Q4["Q4_0 layer-weight sub-payload<br/>per-layer offsets<br/>SOA layout"]
        F16["F16 layer-weight sub-payload<br/>per-layer offsets<br/>SOA layout"]
        LMHQ["lm_head Q4_0<br/>AOS layout<br/>(INV-135 v2 layout 의무)"]
        LMHF["lm_head F16<br/>AOS layout<br/>(INV-135 v2 dtype-agnostic)"]
        Q4 --> F16 --> LMHQ --> LMHF
    end

    TIDX["TENSOR_INDEX entries<br/>layer 0 attn_q dtype=Q4_0 → variant_offset[0] = 0x0...<br/>layer 0 attn_q dtype=F16 → variant_offset[0] = 0x29...<br/>...<br/>layer_idx=u32::MAX kind=11 lm_head dtype=Q4_0 → variant_offset[0] = 0xN0...<br/>layer_idx=u32::MAX kind=11 lm_head dtype=F16 → variant_offset[0] = 0xN1..."]
    TIDX -.references.-> Q4
    TIDX -.references.-> F16
    TIDX -.references.-> LMHQ
    TIDX -.references.-> LMHF

    style Q4 fill:#c8e6c9
    style F16 fill:#bbdefb
    style LMHQ fill:#ffe0b2
    style LMHF fill:#ffe0b2
```

lm_head는 Adreno SOA variant 안에서도 모든 dtype에 대해 AOS layout으로 동봉된다 (image1d_buffer_t 한계). dtype별 candidate가 layer weight와 동일하게 ENG-ALG-225 reader dispatch precedence(호출자 명시 → META.default_dtype → first-match)를 따라 lookup된다.

### 5.2 Strip 전후 비교

```mermaid
flowchart LR
    subgraph BEFORE["Before: ~2.1 GB (Adreno+CUDA+CPU)"]
        BH[Header]
        BT[SectionTable<br/>6 entries]
        BMETA[META]
        BTOK[TOKENIZER]
        BTIDX[TENSOR_INDEX]
        BSOA[WEIGHTS_ADRENO_SOA]
        BCUDA[WEIGHTS_CUDA_AOS]
        BCPU[WEIGHTS_CPU_AOS]
        BH --> BT --> BMETA --> BTOK --> BTIDX --> BSOA --> BCUDA --> BCPU
    end

    BEFORE -->|"auf-tool strip<br/>--keep META TOKENIZER<br/>TENSOR_INDEX WEIGHTS_ADRENO_SOA"| AFTER

    subgraph AFTER["After: ~700 MB (Adreno only)"]
        AH["Header<br/>section_count=4<br/>capability_optional 갱신"]
        AT["SectionTable<br/>4 entries (offsets recomputed)"]
        AMETA[META]
        ATOK[TOKENIZER]
        ATIDX[TENSOR_INDEX]
        ASOA[WEIGHTS_ADRENO_SOA]
        AH --> AT --> AMETA --> ATOK --> ATIDX --> ASOA
    end

    BEFORE -->|"backup before strip<br/>(default)"| BAK[(model.auf.bak)]

    style AFTER fill:#c8e6c9
    style BAK fill:#fff3e0
```

### 5.3 Reader / Writer dataflow

```mermaid
flowchart TB
    subgraph WRITE["Write path (auf-tool build)"]
        GGUF[(GGUF)]
        GGUF --> Parse[parse GGUF]
        Parse --> META_W[META JSON]
        Parse --> TOK_W[TOKENIZER blob]
        Parse --> Layers[layer tensors]
        Layers --> Adreno_W[Adreno SOA<br/>convert]
        Layers --> Cuda_W[CUDA AOS<br/>permute + align]
        Layers --> Cpu_W[CPU AOS<br/>permute + align]
        Adreno_W --> TIDX_W[TENSOR_INDEX<br/>build]
        Cuda_W --> TIDX_W
        Cpu_W --> TIDX_W
        META_W --> Layout[plan section layout<br/>cursor + 64KB align]
        TOK_W --> Layout
        TIDX_W --> Layout
        Adreno_W --> Layout
        Cuda_W --> Layout
        Cpu_W --> Layout
        Layout --> Write[write tempfile<br/>+ atomic rename]
        Write --> AUF_W[(model.auf)]
    end

    subgraph READ["Read path (Engine init)"]
        AUF_R[(model.auf)]
        AUF_R --> Mmap[mmap]
        Mmap --> Header_R[parse header<br/>INV-132 verify]
        Header_R --> Table[parse section table<br/>INV-134 verify]
        Table --> Required[locate META + TOKENIZER<br/>+ TENSOR_INDEX<br/>+ WEIGHTS_<backend><br/>INV-133 verify]
        Required --> Meta_R[parse META JSON]
        Required --> Tok_R[parse TOKENIZER]
        Required --> Tidx_R[parse TENSOR_INDEX]
        Required --> Weights_R[zero-copy slice<br/>weights_payload]
        Meta_R --> View[AufView]
        Tok_R --> View
        Tidx_R --> View
        Weights_R --> View
        View --> Engine[Engine: SwapExecutor<br/>+ noshuffle SOA register]
    end

    style AUF_W fill:#c8e6c9
    style View fill:#c8e6c9
```

---

## 6. 코드-스펙 차이 (Phase 3.7 진입 시점)

본 시점에는 코드가 아직 작성되지 않았다 (Architect 단계). 다음은 implementation 단계에서 발생할 수 있는 차이 후보의 사전 예측이며, 실제 차이 발견 시 본 §6에 추가한다.

| 예상 차이 | 사유 | 처리 방향 |
|-----------|------|----------|
| Variant 변환 함수가 GGUF parser와 강결합 | 기존 GGUF loader 재사용 효율 | engine crate 내 모듈로 배치, auf-tool은 동일 crate의 binary로 |
| Reader/Writer trait 추상화 부재 | v0.1은 단일 포맷이라 trait 불필요 | v0.2에서 trait 검토 (zstd reader 등 추가 시) |
| Section payload alignment 64KB가 작은 모델(<100MB)에서 비효율 | 헤더 패딩만 ~63KB | 작은 모델은 4KB align fallback (v0.2 capability) — v0.1은 64KB 일괄 |

---

## 7. Config / CLI

### 7.1 `auf-tool` CLI 플래그

```
auf-tool build
    --input <PATH>          GGUF 입력
    --output <PATH>         AUF 출력
    --variants <list>       "all" 또는 comma-separated (예: WEIGHTS_ADRENO_SOA,WEIGHTS_CPU_AOS)
    [--created-by <STR>]    custom created_by (기본: "llm_rs2 v{CARGO_PKG_VERSION}")
    [--include-lm-head <on|off|auto>]   lm_head Q4_0 사전 변환 (v0.1.1, default: auto)
                                        auto: GGUF lm_head dtype != Q4_0이면 quantize
                                        on:   강제 quantize (이미 Q4_0이면 no-op)
                                        off:  skip (v0.1.0 호환 출력, capability bit 2 = 0)

    [--dtypes <list>]                   v0.2 multi-dtype variant. comma-separated.
                                        (예: Q4_0,F16). default: 단일 dtype = GGUF source dtype.
                                        2개 이상이면 capability_optional bit 3 set, format_minor=2.
                                        지원 dtype: Q4_0, F16, F32 (Phase 1 범위).
    [--default-dtype <DTYPE>]           v0.2 multi-dtype variant. META.default_dtype 값.
                                        --dtypes에 포함된 값이어야 함. default: --dtypes 첫 번째.

auf-tool info <PATH>        헤더 + section 목록 + capability flags 출력
                            v0.2: TENSOR_INDEX entry의 dtype별 분포도 표시 (예: layer 0 attn_q: [Q4_0, F16])

auf-tool strip <PATH>
    --keep <list>           유지할 section tag (comma-separated)
    [--keep-dtype <DTYPE>]  v0.2: 특정 dtype의 entry만 유지하고 나머지 dtype payload 제거 (선택, Phase 5)
    [--no-backup]           기본은 .auf.bak 자동 생성

auf-tool verify <PATH>
    [--source <GGUF>]       제공 시 source_hash 일치도 검증
    [--check-shapes]        v0.2: multi-dtype entry 간 shape 일치 검증 (INV-137)

auf-tool repack             [Phase 5로 미룸. v0.1에서는 미구현]
    --input <STRIPPED>
    --source <GGUF>
    --add <list>
    --output <PATH>
```

**v0.2 사용 예**:

```sh
# Q4_0 + F16 dual-dtype build (default = Q4_0)
auf-tool build --input model.gguf --output model.auf \
               --variants WEIGHTS_ADRENO_SOA \
               --dtypes Q4_0,F16 --default-dtype Q4_0

# v0.1.x 호환 build (single-dtype, capability bit 3 = 0)
auf-tool build --input model.gguf --output model_legacy.auf \
               --variants WEIGHTS_ADRENO_SOA \
               --dtypes Q4_0
```

### 7.2 Engine CLI 통합

기존 `--secondary-source <PATH>` 플래그가 GGUF 또는 AUF 둘 다 받도록 확장:

- 확장자 `.gguf` → 기존 GGUF loader.
- 확장자 `.auf` → AUF reader (본 spec).
- 그 외 → 에러.

이 분기는 implementation 단계에서 결정. 본 spec/arch는 reader 인터페이스만 명시.

---

## 8. 교차 참조

| 항목 | 위치 |
|------|------|
| 포맷 정의 (binary layout) | `spec/33-engine-data.md` §3.22 (ENG-DAT-096) |
| Reader/Writer/Stripper 알고리즘 (v0.1.x) | `spec/32-engine-algorithms.md` §3.12.17 (ENG-ALG-223) |
| **Multi-dtype Writer/Reader 알고리즘 (v0.2)** | `spec/32-engine-algorithms.md` §3.12.18 (ENG-ALG-224, ENG-ALG-225) |
| **Multi-dtype entry 의미 (v0.2)** | `spec/33-engine-data.md` §3.22.14 (ENG-DAT-097) |
| **Dtype Selection Precedence (v0.2)** | `spec/33-engine-data.md` §3.22.15 (ENG-DAT-098) |
| **META `default_dtype` 필드 (v0.2)** | `spec/33-engine-data.md` §3.22.16 (ENG-DAT-099) |
| Adreno SOA 재변환 (3.7a) | `spec/32-engine-algorithms.md` §3.12.16 (ENG-ALG-222), `arch/weight_swap.md` |
| Phase 3.6 SOA registry coherence | `spec/32-engine-algorithms.md` §3.12.15 (ENG-ALG-221), `arch/weight_swap.md` |
| 무결성 불변식 (v0.1.x) | `spec/41-invariants.md` §3.16~3.17 (INV-131~136) |
| **Multi-dtype 불변식 (v0.2)** | `spec/41-invariants.md` §3.18 (INV-137~139) |
| CLI 사용 가이드 | `docs/auf_tool_guide.md` |
| 버전별 변경 이력 | `docs/auf_format_changelog.md` |
| Phase 3.7 TODO | `.agent/todos/feat_weight_swap.md` |
