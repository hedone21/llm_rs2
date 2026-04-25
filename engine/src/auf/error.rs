/// AUF 파일 파싱/직렬화 에러 타입.
///
/// 모든 에러는 `Result::Err`로 반환되며, panic 없이 진단 메시지를 포함한다 (ENG-ALG-C10).
use std::fmt;

#[derive(Debug)]
pub enum AufError {
    /// 파일이 AUF가 아니다 (magic 불일치).
    MagicMismatch,
    /// format_major가 reader 지원 범위를 초과한다.
    UnsupportedFormatMajor { found: u16, max_supported: u16 },
    /// capability_required 필드에 reader가 인식하지 못하는 비트가 set되어 있다.
    UnknownRequiredCapability { bit: u32 },
    /// required section이 파일에 없다.
    RequiredSectionMissing { tag: String },
    /// backend에 맞는 WEIGHTS_* section이 없다.
    WeightsSectionMissing { weights_tag: String },
    /// section offset/size가 파일 범위를 벗어나거나 payload_start_offset 미만이다 (INV-134).
    SectionRangeInvalid { tag: String, detail: String },
    /// 두 section의 byte range가 overlap된다 (INV-134).
    SectionOverlap { tag_a: String, tag_b: String },
    /// 동일 tag가 두 번 등장한다 (INV-134).
    DuplicateSectionTag { tag: String },
    /// section flag 모순 (REQUIRED + STRIPPABLE 동시 set).
    ContradictoryFlags { tag: String },
    /// SECTION_COMPRESSED가 set되었으나 v0.1 reader는 압축 미지원.
    CompressedSectionUnsupported { tag: String },
    /// section table 영역이 파일 크기를 벗어난다.
    SectionTableTruncated,
    /// 파일이 헤더(256B) 미만이다.
    FileTooSmall,
    /// TOKENIZER section 포맷이 잘못되었다.
    TokenizerFormat { detail: String },
    /// TENSOR_INDEX section 포맷이 잘못되었다.
    TensorIndexFormat { detail: String },
    /// WEIGHTS_* section이 존재하지만 TENSOR_INDEX가 해당 variant를 cover하지 않는다.
    ///
    /// `auf-tool build`가 TensorIndex를 채우지 않았거나, 수동 조작으로 빈 인덱스가 주입된 경우.
    TensorIndexMissingVariant { weights_tag: String },
    /// 기타 IO/파싱 에러.
    Io(std::io::Error),
    /// 기타 메시지 에러.
    Other(String),
}

impl fmt::Display for AufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AufError::MagicMismatch => {
                write!(f, "Not an AUF file (magic mismatch)")
            }
            AufError::UnsupportedFormatMajor {
                found,
                max_supported,
            } => write!(
                f,
                "AUF format_major={found} but reader supports up to {max_supported}. Update llm_rs2."
            ),
            AufError::UnknownRequiredCapability { bit } => write!(
                f,
                "AUF requires capability bit {bit} which is not understood by this reader"
            ),
            AufError::RequiredSectionMissing { tag } => {
                write!(f, "{tag} section missing — file is not a valid AUF")
            }
            AufError::WeightsSectionMissing { weights_tag } => write!(
                f,
                "AUF does not contain {weights_tag}. Run 'auf-tool repack --add {weights_tag}' to add it from source GGUF."
            ),
            AufError::SectionRangeInvalid { tag, detail } => {
                write!(f, "Section {tag} has invalid offset/size: {detail}")
            }
            AufError::SectionOverlap { tag_a, tag_b } => {
                write!(
                    f,
                    "Section {tag_a} overlaps with {tag_b} (offset/size invalid)"
                )
            }
            AufError::DuplicateSectionTag { tag } => {
                write!(f, "Duplicate section tag: {tag}")
            }
            AufError::ContradictoryFlags { tag } => {
                write!(
                    f,
                    "Section {tag} has contradictory flags REQUIRED+STRIPPABLE"
                )
            }
            AufError::CompressedSectionUnsupported { tag } => write!(
                f,
                "Section {tag} uses compression which is not supported in format_minor=1"
            ),
            AufError::SectionTableTruncated => {
                write!(f, "AUF section table extends beyond file size")
            }
            AufError::FileTooSmall => write!(f, "AUF file is too small to contain a valid header"),
            AufError::TokenizerFormat { detail } => {
                write!(f, "TOKENIZER section format error: {detail}")
            }
            AufError::TensorIndexFormat { detail } => {
                write!(f, "TENSOR_INDEX section format error: {detail}")
            }
            AufError::TensorIndexMissingVariant { weights_tag } => write!(
                f,
                "AUF file invariant violation: TENSOR_INDEX does not list variant '{}' — \
                 empty index detected. Rebuild with 'auf-tool build'.",
                weights_tag
            ),
            AufError::Io(e) => write!(f, "AUF I/O error: {e}"),
            AufError::Other(msg) => write!(f, "AUF error: {msg}"),
        }
    }
}

impl std::error::Error for AufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let AufError::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

impl From<std::io::Error> for AufError {
    fn from(e: std::io::Error) -> Self {
        AufError::Io(e)
    }
}

pub type AufResult<T> = Result<T, AufError>;
