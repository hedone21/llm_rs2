//! HTP FastRPC backend — AEEResult → engine error mapping (INV-HTP-FRPC-004).
//!
//! 모든 FastRPC FFI 호출의 return code (i32, AEEResult) 는 본 모듈의
//! [`map_aee_err`] 로 wrap 된다. spec `engine::error::Error::BackendDeviceError`
//! variant 는 후속 sprint 에서 도입 예정이라 PoC 단계에서는 `anyhow::Error`
//! 의 컨텍스트 문자열에 raw code 를 hex + decimal 양쪽으로 보존한다.
//!
//! 자세한 정책: `spec/htp_fastrpc.md` §3 INV-HTP-FRPC-004.

#![cfg(feature = "htp_fastrpc")]

use anyhow::anyhow;

// ── AEEResult known constants ──
//
// 출처: Qualcomm FastRPC open-source headers (BSD-3-Clause).
// 본 sprint scope 에서 발생 가능성 있는 8 ~ 10 known signature 만 유지한다.

/// `AEE_SUCCESS` — 모든 FastRPC 호출의 OK return.
pub const AEE_SUCCESS: i32 = 0;

/// `AEE_EUNABLETOLOAD` — dlopen / library 로드 실패. htp-drv.cpp `htpdrv_init`
/// 도 이 값으로 실패를 보고한다.
pub const AEE_EUNABLETOLOAD: i32 = 0x0E;

/// `AEE_EBADPARM` — packet schema mismatch 또는 invalid argument.
pub const AEE_EBADPARM: i32 = 0x02;

/// `AEE_ENOSUCH` — 요청한 method / handle 이 존재하지 않음. URI typo 의심.
pub const AEE_ENOSUCH: i32 = 0x05;

/// `AEE_ENOMEMORY` — rpcmem heap 부족 (S25 stock system heap 한계).
pub const AEE_ENOMEMORY: i32 = 0x04;

/// `AEE_EEXPIRED` — dspqueue_read timeout. ggml-hexagon enqueue path 에서
/// 정상적인 polling skip 신호로 사용된다.
pub const AEE_EEXPIRED: i32 = 0x0F;

/// `AEE_EFAILED` — 일반 실패 (DSP 측 op execute 실패 포함).
pub const AEE_EFAILED: i32 = 0x01;

/// `AEE_EUNSUPPORTED` — vendor extension control 차단 (Q-2.1 dry-run 패턴
/// 일부 변형).
pub const AEE_EUNSUPPORTED: i32 = 0x0A;

/// Q-2.1 dry-run RED 의 known signature — `deviceCreate err=0x36b1`. QNN
/// backend 의 vendor-extension `domain_init` 차단. 본 backend 는 QNN SDK
/// 의존이 0 이라 직접 만나면 안 되는 코드지만 진단 식별자로 보존.
pub const KNOWN_QNN_DEVICE_CREATE_FAIL: u32 = 0x36b1;

/// HTP DSP 응답의 `status` 가 OK 가 아닐 때 wrap 하는 sentinel. raw
/// AEEResult 와는 다른 axis 라 별도 namespace 유지.
pub const HTP_STATUS_OK: u32 = 1;

/// 알려진 AEE code → 사람 읽을 수 있는 짧은 라벨. 미등록 코드는 `"unknown"`.
pub fn aee_label(code: i32) -> &'static str {
    match code {
        AEE_SUCCESS => "AEE_SUCCESS",
        AEE_EFAILED => "AEE_EFAILED",
        AEE_EBADPARM => "AEE_EBADPARM",
        AEE_ENOMEMORY => "AEE_ENOMEMORY",
        AEE_ENOSUCH => "AEE_ENOSUCH",
        AEE_EUNSUPPORTED => "AEE_EUNSUPPORTED",
        AEE_EUNABLETOLOAD => "AEE_EUNABLETOLOAD",
        AEE_EEXPIRED => "AEE_EEXPIRED",
        _ if (code as u32) == KNOWN_QNN_DEVICE_CREATE_FAIL => "KNOWN_QNN_DEVICE_CREATE_FAIL",
        _ => "unknown",
    }
}

/// FastRPC FFI return code 를 engine error 로 변환한다.
///
/// raw code 는 hex + decimal 양쪽 표기로 보존되어 caller 가 logcat 의
/// `E adsprpc remote.c:NN ...` 라인과 cross-reference 할 수 있다.
///
/// PoC scope 에서는 `anyhow::Error` 를 반환. 후속 sprint 에서
/// `engine::error::Error::BackendDeviceError { backend, code, msg }` variant
/// 가 추가되면 본 함수만 교체하면 된다 (call site 변경 없음).
pub fn map_aee_err(code: i32) -> anyhow::Error {
    anyhow!(
        "htp_fastrpc: AEEResult={} (0x{:x}, raw={})",
        aee_label(code),
        code as u32,
        code,
    )
}

/// HTP DSP 응답의 `status` field 를 engine error 로 변환한다. dspqueue
/// transport 자체는 OK 였지만 DSP-side op execute 가 실패한 케이스.
pub fn map_htp_status(status: u32) -> anyhow::Error {
    let label = match status {
        1 => "HTP_STATUS_OK",
        2 => "HTP_STATUS_INTERNAL_ERR",
        3 => "HTP_STATUS_NO_SUPPORT",
        4 => "HTP_STATUS_INVAL_PARAMS",
        5 => "HTP_STATUS_VTCM_TOO_SMALL",
        _ => "unknown",
    };
    anyhow!("htp_fastrpc: DSP rsp.status={} ({})", status, label)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aee_label_known() {
        assert_eq!(aee_label(AEE_SUCCESS), "AEE_SUCCESS");
        assert_eq!(aee_label(AEE_EUNABLETOLOAD), "AEE_EUNABLETOLOAD");
        assert_eq!(aee_label(0x36b1), "KNOWN_QNN_DEVICE_CREATE_FAIL");
        assert_eq!(aee_label(0x7fff_ffff), "unknown");
    }

    #[test]
    fn map_aee_err_includes_raw_code() {
        let err = map_aee_err(AEE_EBADPARM);
        let s = format!("{err}");
        assert!(s.contains("AEE_EBADPARM"), "missing label: {s}");
        assert!(s.contains("0x2"), "missing hex: {s}");
    }

    #[test]
    fn map_htp_status_includes_label() {
        let err = map_htp_status(4);
        let s = format!("{err}");
        assert!(s.contains("HTP_STATUS_INVAL_PARAMS"), "missing label: {s}");
    }
}
