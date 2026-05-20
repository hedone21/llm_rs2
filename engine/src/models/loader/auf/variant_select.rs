//! AUF primary backend variant + dtype 선택 정책.
//!
//! `--primary-variant` / `--primary-dtype` CLI flag를 통해 사용자가 명시할 수 있고,
//! 명시 없으면 build feature로부터 자동 결정한다.

use crate::auf::{BackendTag, TensorDType};

/// AUF primary backend variant 선택.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AufVariantChoice {
    /// build feature 기반 자동 선택 (opencl→AdrenoSoa, cuda*→CudaAos, CPU→CpuAos).
    #[default]
    Auto,
    AdrenoSoa,
    CpuAos,
    CudaAos,
}

impl AufVariantChoice {
    /// build feature 기반 default backend tag.
    pub fn default_tag() -> BackendTag {
        #[cfg(feature = "opencl")]
        {
            return BackendTag::AdrenoSoa;
        }
        #[cfg(any(feature = "cuda", feature = "cuda-embedded"))]
        {
            return BackendTag::CudaAos;
        }
        #[allow(unreachable_code)]
        BackendTag::CpuAos
    }

    /// 명시 선택을 `BackendTag`로 변환. `Auto`는 build feature로 결정.
    pub fn to_backend_tag(self) -> BackendTag {
        match self {
            AufVariantChoice::Auto => Self::default_tag(),
            AufVariantChoice::AdrenoSoa => BackendTag::AdrenoSoa,
            AufVariantChoice::CpuAos => BackendTag::CpuAos,
            AufVariantChoice::CudaAos => BackendTag::CudaAos,
        }
    }
}

/// AUF primary dtype 선택.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AufDtypeChoice {
    /// `META.default_dtype` 사용. 없으면 entry first-match.
    #[default]
    Auto,
    F16,
    F32,
    BF16,
    Q4_0,
    Q4_1,
    Q8_0,
}

impl AufDtypeChoice {
    /// 명시 선택을 `TensorDType`로 변환. `Auto`는 `None`을 반환하여 `AufView::lookup_tensor`가
    /// META.default_dtype precedence를 사용하게 한다.
    pub fn to_tensor_dtype(self) -> Option<TensorDType> {
        match self {
            AufDtypeChoice::Auto => None,
            AufDtypeChoice::F16 => Some(TensorDType::F16),
            AufDtypeChoice::F32 => Some(TensorDType::F32),
            AufDtypeChoice::BF16 => Some(TensorDType::BF16),
            AufDtypeChoice::Q4_0 => Some(TensorDType::Q4_0),
            AufDtypeChoice::Q4_1 => Some(TensorDType::Q4_1),
            AufDtypeChoice::Q8_0 => Some(TensorDType::Q8_0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_choices() {
        assert_eq!(AufVariantChoice::default(), AufVariantChoice::Auto);
        assert_eq!(AufDtypeChoice::default(), AufDtypeChoice::Auto);
    }

    #[test]
    fn explicit_dtype_round_trip() {
        assert_eq!(
            AufDtypeChoice::Q4_0.to_tensor_dtype(),
            Some(TensorDType::Q4_0)
        );
        assert_eq!(
            AufDtypeChoice::F16.to_tensor_dtype(),
            Some(TensorDType::F16)
        );
        assert_eq!(AufDtypeChoice::Auto.to_tensor_dtype(), None);
    }

    #[test]
    fn explicit_variant_round_trip() {
        assert_eq!(
            AufVariantChoice::CpuAos.to_backend_tag(),
            BackendTag::CpuAos
        );
    }
}
