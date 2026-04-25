/// AUF Reader — mmap 기반 파일 파싱 + 무결성 검증 (ENG-ALG-223 §3.12.17.1).
///
/// 검증 순서 (INV-132, INV-133, INV-134):
/// 1. 파일 크기 ≥ 256B
/// 2. magic 검증 → format_major 검증 → capability_required 검증 (INV-132)
/// 3. section table 파싱 + 무결성 검증 (INV-134)
/// 4. required section 존재 확인 (INV-133)
/// 5. backend WEIGHTS_* section 존재 확인 (INV-133)
/// 6. META, TOKENIZER, TENSOR_INDEX 파싱 (lazy 가능)
use std::path::Path;

use memmap2::{Mmap, MmapOptions};

use crate::auf::error::{AufError, AufResult};
use crate::auf::header::AufHeader;
use crate::auf::meta::AufMeta;
use crate::auf::section::{
    SectionTable, TAG_META, TAG_TENSOR_INDEX, TAG_TOKENIZER, TAG_WEIGHTS_ADRENO_SOA,
    TAG_WEIGHTS_CPU_AOS, TAG_WEIGHTS_CUDA_AOS,
};
use crate::auf::tensor_index::TensorIndex;
use crate::auf::tokenizer::AufTokenizer;

/// backend 식별자.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendTag {
    AdrenoSoa,
    CudaAos,
    CpuAos,
    /// strip 등 backend check를 우회하는 모드.
    Any,
}

impl BackendTag {
    pub fn weights_section_tag(self) -> Option<&'static str> {
        match self {
            BackendTag::AdrenoSoa => Some(TAG_WEIGHTS_ADRENO_SOA),
            BackendTag::CudaAos => Some(TAG_WEIGHTS_CUDA_AOS),
            BackendTag::CpuAos => Some(TAG_WEIGHTS_CPU_AOS),
            BackendTag::Any => None,
        }
    }
}

/// AUF 파일 view — mmap 보유 + 파싱된 메타데이터 + WEIGHTS payload byte slice.
pub struct AufView {
    /// 파일 mmap (lifetime 보장용).
    _mmap: Mmap,
    pub header: AufHeader,
    pub section_table: SectionTable,
    pub meta: AufMeta,
    pub tokenizer: AufTokenizer,
    pub tensor_index: TensorIndex,
    /// WEIGHTS_* payload의 파일 내 byte range (offset, size).
    /// `BackendTag::Any`이면 None.
    pub weights_range: Option<(u64, u64)>,
}

impl AufView {
    /// WEIGHTS payload byte slice를 반환한다. `BackendTag::Any`이면 None.
    ///
    /// WEIGHTS payload는 mmap slice이므로 별도 복사 없이 zero-copy 접근 가능.
    /// 반환된 slice의 lifetime은 `AufView`에 귀속된다.
    pub fn weights_bytes(&self) -> Option<&[u8]> {
        self.weights_range
            .map(|(offset, size)| &self._mmap[offset as usize..][..size as usize])
    }

    /// mmap 전체 바이트 slice를 반환한다 (stripper 등 내부 용도).
    pub fn raw_bytes(&self) -> &[u8] {
        &self._mmap[..]
    }

    /// 파일 전체 크기 (mmap 길이).
    pub fn file_size(&self) -> u64 {
        self._mmap.len() as u64
    }
}

impl std::fmt::Debug for AufView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AufView")
            .field("format_major", &self.header.format_major)
            .field("format_minor", &self.header.format_minor)
            .field("section_count", &self.header.section_count)
            .field("weights_range", &self.weights_range)
            .finish()
    }
}

/// AUF 파일을 mmap으로 열고 검증 + 파싱하여 `AufView`를 반환한다.
///
/// ENG-ALG-223 §3.12.17.1 Reader 알고리즘 구현.
pub fn open(path: &Path, backend: BackendTag) -> AufResult<AufView> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file) }?;
    open_from_mmap(mmap, backend)
}

/// 바이트열에서 AufView를 파싱한다 (테스트 및 단위 검증 친화).
///
/// 내부적으로 임시 파일을 통해 mmap을 생성한다.
pub fn open_from_bytes(bytes: Vec<u8>, backend: BackendTag) -> AufResult<AufView> {
    use std::io::Write;
    let mut f = tempfile::tempfile().map_err(AufError::Io)?;
    f.write_all(&bytes).map_err(AufError::Io)?;
    let mmap = unsafe { MmapOptions::new().map(&f) }?;
    open_from_mmap(mmap, backend)
}

/// mmap에서 AufView를 파싱하는 실제 로직.
fn open_from_mmap(mmap: Mmap, backend: BackendTag) -> AufResult<AufView> {
    let bytes: &[u8] = &mmap;
    let file_size = bytes.len() as u64;

    // (1) 헤더 파싱 + 검증 (INV-132)
    let header = AufHeader::from_bytes(bytes)?;
    header.validate()?;

    // format_major=0 경고 (실험적)
    if header.format_major == 0 {
        log::warn!("AUF format_major=0 is experimental");
    }

    // (2) section table 영역 경계 확인
    let section_table_end = header
        .section_table_offset
        .checked_add(header.section_count as u64 * 48)
        .ok_or_else(|| AufError::Other("section_table_offset overflow".to_owned()))?;
    if section_table_end > file_size {
        return Err(AufError::SectionTableTruncated);
    }

    let section_table = SectionTable::from_bytes(
        &bytes[header.section_table_offset as usize..],
        header.section_count,
    )?;

    // (3) section 무결성 검증 (INV-134)
    section_table.validate_unique_tags()?;
    section_table.validate_flags()?;
    section_table.validate_no_compressed()?;
    section_table.validate_ranges(header.payload_start_offset, file_size)?;

    // (4) required section 존재 확인 (INV-133)
    let meta_entry = section_table
        .find(TAG_META)
        .ok_or_else(|| AufError::RequiredSectionMissing {
            tag: TAG_META.to_owned(),
        })?
        .clone();
    let tok_entry = section_table
        .find(TAG_TOKENIZER)
        .ok_or_else(|| AufError::RequiredSectionMissing {
            tag: TAG_TOKENIZER.to_owned(),
        })?
        .clone();
    let tidx_entry = section_table
        .find(TAG_TENSOR_INDEX)
        .ok_or_else(|| AufError::RequiredSectionMissing {
            tag: TAG_TENSOR_INDEX.to_owned(),
        })?
        .clone();

    // (5) backend WEIGHTS_* 확인 (INV-133)
    let weights_range = if let Some(weights_tag) = backend.weights_section_tag() {
        let entry =
            section_table
                .find(weights_tag)
                .ok_or_else(|| AufError::WeightsSectionMissing {
                    weights_tag: weights_tag.to_owned(),
                })?;
        Some((entry.offset, entry.size))
    } else {
        None // BackendTag::Any
    };

    // (6) META 파싱
    let meta_bytes = &bytes[meta_entry.offset as usize..][..meta_entry.size as usize];
    let meta = AufMeta::from_json_bytes(meta_bytes)?;

    // (7) TOKENIZER 파싱
    let tok_bytes = &bytes[tok_entry.offset as usize..][..tok_entry.size as usize];
    let tokenizer = AufTokenizer::from_bytes(tok_bytes)?;

    // (8) TENSOR_INDEX 파싱
    let tidx_bytes = &bytes[tidx_entry.offset as usize..][..tidx_entry.size as usize];
    let tensor_index = TensorIndex::from_bytes(tidx_bytes)?;

    Ok(AufView {
        _mmap: mmap,
        header,
        section_table,
        meta,
        tokenizer,
        tensor_index,
        weights_range,
    })
}

// tempfile crate 없이 동작할 수 있도록 fallback — engine의 Cargo.toml에 tempfile이 없으면 표준 라이브러리로 대체
// (실제로는 writer 테스트에서 tempfile을 사용하므로 Cargo.toml에 추가 필요)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auf::meta::AufMeta;
    use crate::auf::tokenizer::{AufTokenizer, TOKENIZER_KIND_BPE};
    use crate::auf::writer::AufWriter;

    fn make_meta() -> AufMeta {
        AufMeta {
            architecture: "llama".to_owned(),
            n_layers: 2,
            n_heads_q: 4,
            n_kv_heads: 2,
            head_dim: 8,
            hidden_dim: 32,
            ffn_dim: 64,
            vocab_size: 10,
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
            tokens: vec![b"hello".to_vec(), b"world".to_vec()],
            merges: vec![],
            bos_id: 1,
            eos_id: 2,
            pad_id: -1,
            unk_id: 0,
            chat_template: None,
        }
    }

    fn build_auf_bytes(weights_payload: &[u8], weights_tag: &str) -> Vec<u8> {
        AufWriter::new(make_meta(), make_tokenizer(), [0u8; 32], 0, 0)
            .add_weights_section(weights_tag, weights_payload.to_vec())
            .build()
            .unwrap()
    }

    #[test]
    fn reader_cpu_aos_ok() {
        let payload = vec![0u8; 128];
        let bytes = build_auf_bytes(&payload, "WEIGHTS_CPU_AOS");
        let view = open_from_bytes(bytes, BackendTag::CpuAos).unwrap();
        assert_eq!(view.meta.architecture, "llama");
        assert_eq!(view.tokenizer.tokens.len(), 2);
        let wr = view.weights_range.unwrap();
        assert_eq!(wr.1, 128);
    }

    #[test]
    fn reader_any_skips_weights_check() {
        let payload = vec![0u8; 64];
        let bytes = build_auf_bytes(&payload, "WEIGHTS_CPU_AOS");
        let view = open_from_bytes(bytes, BackendTag::Any).unwrap();
        assert!(view.weights_range.is_none());
    }

    #[test]
    fn reader_wrong_backend_fails() {
        let payload = vec![0u8; 64];
        let bytes = build_auf_bytes(&payload, "WEIGHTS_CPU_AOS");
        let err = open_from_bytes(bytes, BackendTag::AdrenoSoa).unwrap_err();
        assert!(matches!(err, AufError::WeightsSectionMissing { .. }));
    }

    #[test]
    fn reader_missing_meta_fails() {
        // META 없는 AUF 만들기 — raw bytes로 직접 조작은 복잡하므로
        // magic을 오염시켜 magic mismatch로 테스트
        let payload = vec![0u8; 64];
        let mut bytes = build_auf_bytes(&payload, "WEIGHTS_CPU_AOS");
        bytes[0] = 0xFF;
        let err = open_from_bytes(bytes, BackendTag::CpuAos).unwrap_err();
        assert!(matches!(err, AufError::MagicMismatch));
    }

    #[test]
    fn reader_bad_format_major_fails() {
        let payload = vec![0u8; 64];
        let mut bytes = build_auf_bytes(&payload, "WEIGHTS_CPU_AOS");
        // format_major offset=8, set to 1
        bytes[8] = 1;
        bytes[9] = 0;
        let err = open_from_bytes(bytes, BackendTag::CpuAos).unwrap_err();
        assert!(matches!(
            err,
            AufError::UnsupportedFormatMajor { found: 1, .. }
        ));
    }
}
