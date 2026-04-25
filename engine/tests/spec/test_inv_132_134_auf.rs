/// Spec 테스트: INV-132, INV-133, INV-134 — AUF v0.1 reader fail-fast invariants.
///
/// INV-132: magic/format_major/capability_required 불일치 시 panic 없이 reject.
/// INV-133: required section (META/TOKENIZER/TENSOR_INDEX) 누락 시 reject + backend WEIGHTS_* 누락 시 reject.
/// INV-134: section offset/size 무결성 — payload_start_offset 미만 / 파일 범위 초과 / overlap / 중복 tag.
///
/// spec: `spec/33-engine-data.md` §3.22 (ENG-DAT-096, ENG-DAT-C13, ENG-DAT-C14)
///       `spec/32-engine-algorithms.md` §3.12.17 (ENG-ALG-223)
///       `spec/41-invariants.md` §3.16 (INV-132~134)
use llm_rs2::auf::{
    AufError, AufMeta, AufTokenizer, AufWriter, BackendTag, SECTION_REQUIRED, SECTION_STRIPPABLE,
    SectionEntry, SectionTable, TAG_WEIGHTS_ADRENO_SOA, TAG_WEIGHTS_CPU_AOS, TOKENIZER_KIND_BPE,
};

// ── 헬퍼 ──────────────────────────────────────────────────────────────────

fn make_meta() -> AufMeta {
    AufMeta {
        architecture: "llama".to_owned(),
        n_layers: 2,
        n_heads_q: 4,
        n_kv_heads: 2,
        head_dim: 8,
        hidden_dim: 32,
        ffn_dim: 64,
        vocab_size: 3,
        max_seq_len: 128,
        rope_theta: 10000.0,
        rotary_dim: 8,
        rope_scaling: 1.0,
        rms_norm_epsilon: 1e-5,
    }
}

fn make_tokenizer() -> AufTokenizer {
    AufTokenizer {
        kind: TOKENIZER_KIND_BPE,
        tokens: vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()],
        merges: vec![],
        bos_id: 0,
        eos_id: 1,
        pad_id: -1,
        unk_id: -1,
        chat_template: None,
    }
}

fn build_valid_auf(weights_tag: &str) -> Vec<u8> {
    AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 0, 0)
        .add_weights_section(weights_tag, vec![42u8; 64])
        .build()
        .unwrap()
}

fn open(bytes: Vec<u8>, backend: BackendTag) -> Result<llm_rs2::auf::AufView, AufError> {
    llm_rs2::auf::reader::open_from_bytes(bytes, backend)
}

// ── INV-132: magic/format/capability reject ───────────────────────────────

/// INV-132.1: magic 불일치 → `MagicMismatch`
#[test]
fn inv132_magic_mismatch_rejected() {
    let mut bytes = build_valid_auf(TAG_WEIGHTS_CPU_AOS);
    // offset 0..8 = magic
    bytes[0] = 0xFF;
    let err = open(bytes, BackendTag::CpuAos).unwrap_err();
    assert!(
        matches!(err, AufError::MagicMismatch),
        "expected MagicMismatch, got: {err}"
    );
}

/// INV-132.2: format_major > READER_MAX → `UnsupportedFormatMajor`
#[test]
fn inv132_format_major_unsupported_rejected() {
    let mut bytes = build_valid_auf(TAG_WEIGHTS_CPU_AOS);
    // format_major offset = 8..10
    bytes[8] = 2;
    bytes[9] = 0;
    let err = open(bytes, BackendTag::CpuAos).unwrap_err();
    assert!(
        matches!(err, AufError::UnsupportedFormatMajor { found: 2, .. }),
        "expected UnsupportedFormatMajor, got: {err}"
    );
}

/// INV-132.3: capability_required 미인식 bit → `UnknownRequiredCapability`
#[test]
fn inv132_unknown_required_capability_rejected() {
    let mut bytes = build_valid_auf(TAG_WEIGHTS_CPU_AOS);
    // capability_required offset = 96..104 (u64 LE)
    bytes[96] = 0x01; // bit 0 set
    let err = open(bytes, BackendTag::CpuAos).unwrap_err();
    assert!(
        matches!(err, AufError::UnknownRequiredCapability { bit: 0 }),
        "expected UnknownRequiredCapability bit=0, got: {err}"
    );
}

/// INV-132.4: source_hash 불일치는 reject 사유가 아니다 (Mode B 자립성).
///
/// source_hash를 임의로 변경해도 reader는 정상 진행해야 한다.
#[test]
fn inv132_source_hash_mismatch_not_rejected() {
    let mut bytes = build_valid_auf(TAG_WEIGHTS_CPU_AOS);
    // source_hash offset = 48..80
    for b in bytes[48..80].iter_mut() {
        *b ^= 0xFF;
    }
    // 정상 파싱되어야 함
    let view = open(bytes, BackendTag::CpuAos).unwrap();
    assert_eq!(view.meta.architecture, "llama");
}

/// INV-132.5: 에러 메시지에 진단 가능한 정보가 포함된다.
#[test]
fn inv132_error_messages_are_diagnostic() {
    // magic mismatch
    let mut bytes = build_valid_auf(TAG_WEIGHTS_CPU_AOS);
    bytes[0] = 0x00;
    let msg = open(bytes, BackendTag::CpuAos).unwrap_err().to_string();
    assert!(
        msg.contains("magic") || msg.contains("AUF"),
        "MagicMismatch message should mention magic: {msg}"
    );

    // format_major
    let mut bytes = build_valid_auf(TAG_WEIGHTS_CPU_AOS);
    bytes[8] = 5;
    let msg = open(bytes, BackendTag::CpuAos).unwrap_err().to_string();
    assert!(
        msg.contains("format_major") && msg.contains('5'),
        "UnsupportedFormatMajor message should contain format_major and value: {msg}"
    );
}

// ── INV-133: required section 존재 확인 ───────────────────────────────────

/// INV-133.1: META section 누락 시 `RequiredSectionMissing`
///
/// 직접 section table을 조작하여 META tag를 ZFOO로 변경.
#[test]
fn inv133_meta_missing_rejected() {
    let bytes = build_valid_auf(TAG_WEIGHTS_CPU_AOS);
    // section table: 256 부터 시작, 첫 entry (META) tag 오염
    let section_table_start = 256usize;
    // tag = bytes[256..272], 'META' = [77, 69, 84, 65, 0, ...]
    let mut bytes = bytes;
    bytes[section_table_start] = b'Z'; // 'META' → 'ZETA'
    let err = open(bytes, BackendTag::CpuAos).unwrap_err();
    assert!(
        matches!(err, AufError::RequiredSectionMissing { .. }),
        "expected RequiredSectionMissing, got: {err}"
    );
}

/// INV-133.2: backend WEIGHTS_* 누락 시 `WeightsSectionMissing` + repack 안내
#[test]
fn inv133_weights_missing_gives_repack_hint() {
    // CPU_AOS AUF를 AdrenoSoa backend로 열기 시도
    let bytes = build_valid_auf(TAG_WEIGHTS_CPU_AOS);
    let err = open(bytes, BackendTag::AdrenoSoa).unwrap_err();
    assert!(
        matches!(err, AufError::WeightsSectionMissing { .. }),
        "expected WeightsSectionMissing, got: {err}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("WEIGHTS_ADRENO_SOA"),
        "message should mention the missing tag: {msg}"
    );
    assert!(
        msg.contains("auf-tool") || msg.contains("repack"),
        "message should hint at repack: {msg}"
    );
}

/// INV-133.3: BackendTag::Any이면 WEIGHTS_* 체크를 건너뛴다
#[test]
fn inv133_any_backend_skips_weights_check() {
    let bytes = build_valid_auf(TAG_WEIGHTS_CPU_AOS);
    let view = open(bytes, BackendTag::Any).unwrap();
    assert!(view.weights_range.is_none());
    // META는 파싱됨
    assert_eq!(view.meta.architecture, "llama");
}

/// INV-133.4: 모든 required section이 있어야 valid
#[test]
fn inv133_all_required_sections_ok() {
    let bytes = build_valid_auf(TAG_WEIGHTS_ADRENO_SOA);
    let view = open(bytes, BackendTag::AdrenoSoa).unwrap();
    assert_eq!(view.meta.architecture, "llama");
    assert_eq!(view.tokenizer.tokens.len(), 3);
    assert!(view.weights_range.is_some());
}

// ── INV-134: section offset/size 무결성 ──────────────────────────────────

/// INV-134.1: 중복 section tag → `DuplicateSectionTag`
#[test]
fn inv134_duplicate_tag_rejected() {
    let entries = vec![
        SectionEntry::new("META", 65536, 100, SECTION_REQUIRED, 1).unwrap(),
        SectionEntry::new("META", 65700, 100, SECTION_REQUIRED, 1).unwrap(),
    ];
    let table = SectionTable { entries };
    let err = table.validate_unique_tags().unwrap_err();
    assert!(
        matches!(err, AufError::DuplicateSectionTag { .. }),
        "expected DuplicateSectionTag, got: {err}"
    );
}

/// INV-134.2: section offset < payload_start_offset → `SectionRangeInvalid`
#[test]
fn inv134_offset_below_payload_start_rejected() {
    let entries = vec![
        SectionEntry::new(
            "META",
            100, // 64KB 미만
            100,
            SECTION_REQUIRED,
            1,
        )
        .unwrap(),
    ];
    let table = SectionTable { entries };
    let err = table.validate_ranges(65536, 1_000_000).unwrap_err();
    assert!(
        matches!(err, AufError::SectionRangeInvalid { .. }),
        "expected SectionRangeInvalid, got: {err}"
    );
}

/// INV-134.3: section.offset + section.size > file_size → `SectionRangeInvalid`
#[test]
fn inv134_offset_beyond_file_size_rejected() {
    let entries = vec![
        SectionEntry::new(
            "META",
            65536,
            1_000_000, // file 크기 초과
            SECTION_REQUIRED,
            1,
        )
        .unwrap(),
    ];
    let table = SectionTable { entries };
    let err = table.validate_ranges(65536, 100_000).unwrap_err();
    assert!(
        matches!(err, AufError::SectionRangeInvalid { .. }),
        "expected SectionRangeInvalid, got: {err}"
    );
}

/// INV-134.4: 두 section byte range overlap → `SectionOverlap`
#[test]
fn inv134_overlapping_sections_rejected() {
    let entries = vec![
        SectionEntry::new("META", 65536, 1000, SECTION_REQUIRED, 1).unwrap(),
        SectionEntry::new("TOKENIZER", 66000, 1000, SECTION_REQUIRED, 1).unwrap(),
        // 65536+1000=66536 > 66000 → overlap
    ];
    let table = SectionTable { entries };
    let err = table.validate_ranges(65536, 1_000_000).unwrap_err();
    assert!(
        matches!(err, AufError::SectionOverlap { .. }),
        "expected SectionOverlap, got: {err}"
    );
}

/// INV-134.5: REQUIRED + STRIPPABLE 동시 설정 → `ContradictoryFlags`
#[test]
fn inv134_contradictory_flags_rejected() {
    let entries = vec![
        SectionEntry::new("META", 65536, 100, SECTION_REQUIRED | SECTION_STRIPPABLE, 1).unwrap(),
    ];
    let table = SectionTable { entries };
    let err = table.validate_flags().unwrap_err();
    assert!(
        matches!(err, AufError::ContradictoryFlags { .. }),
        "expected ContradictoryFlags, got: {err}"
    );
}

/// INV-134.6: 정상 AUF는 전 검증 통과
#[test]
fn inv134_valid_auf_passes_all_checks() {
    let bytes = build_valid_auf(TAG_WEIGHTS_CPU_AOS);
    let view = open(bytes, BackendTag::CpuAos).unwrap();
    assert_eq!(view.meta.architecture, "llama");
    assert!(view.weights_range.is_some());
}

// ── ENG-ALG-C10: panic 없이 에러 반환 ───────────────────────────────────

/// ENG-ALG-C10: 모든 무결성 위반은 panic 없이 Result::Err 반환.
///
/// 다양한 corrupt bytes에 대해 unwrap_err()만 호출 (panic 발생 시 실패).
#[test]
fn alg_c10_no_panic_on_corrupt_input() {
    let valid = build_valid_auf(TAG_WEIGHTS_CPU_AOS);

    // 1. 빈 파일
    assert!(open(vec![], BackendTag::CpuAos).is_err());

    // 2. 헤더만 있고 section table 없음
    assert!(open(vec![0u8; 256], BackendTag::CpuAos).is_err());

    // 3. magic 오염
    let mut b = valid.clone();
    b[3] = 0xFF;
    assert!(open(b, BackendTag::CpuAos).is_err());

    // 4. section_count를 매우 크게
    let mut b = valid.clone();
    b[112] = 0xFF;
    b[113] = 0xFF;
    assert!(open(b, BackendTag::CpuAos).is_err());
}

// ── Round-trip 통합 테스트 ────────────────────────────────────────────────

/// Writer → Reader round-trip: 빌드된 AUF를 다시 reader로 열 수 있어야 한다.
#[test]
fn writer_reader_round_trip() {
    let source_hash = [42u8; 32];
    let payload = (0u8..=63).collect::<Vec<_>>();
    let bytes = AufWriter::new(make_meta(), make_tokenizer(), source_hash, 999, 12345)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, payload.clone())
        .build()
        .unwrap();

    let view = open(bytes, BackendTag::CpuAos).unwrap();

    // meta
    assert_eq!(view.meta.n_layers, 2);
    assert_eq!(view.meta.vocab_size, 3);
    // tokenizer
    assert_eq!(view.tokenizer.tokens.len(), 3);
    assert_eq!(view.tokenizer.bos_id, 0);
    // source hash 보존
    assert_eq!(view.header.source_hash, source_hash);
    assert_eq!(view.header.source_size, 999);
    assert_eq!(view.header.source_mtime, 12345);
    // weights
    let wb = view.weights_bytes().unwrap();
    assert_eq!(wb, payload.as_slice());
}

/// Writer → Stripper → Reader: strip 후 AUF가 여전히 valid.
#[test]
fn writer_stripper_reader_round_trip() {
    use llm_rs2::auf::strip_bytes;

    let bytes = AufWriter::new(make_meta(), make_tokenizer(), [7u8; 32], 100, 200)
        .add_weights_section(TAG_WEIGHTS_CPU_AOS, vec![1u8; 64])
        .add_weights_section(TAG_WEIGHTS_ADRENO_SOA, vec![2u8; 128])
        .build()
        .unwrap();

    // ADRENO_SOA만 유지
    let stripped = strip_bytes(&bytes, &[TAG_WEIGHTS_ADRENO_SOA]).unwrap();
    let view = open(stripped, BackendTag::AdrenoSoa).unwrap();

    assert_eq!(view.meta.architecture, "llama");
    assert_eq!(view.header.source_hash, [7u8; 32]);
    let wb = view.weights_bytes().unwrap();
    assert_eq!(wb.len(), 128);
    assert!(wb.iter().all(|&b| b == 2));
}
