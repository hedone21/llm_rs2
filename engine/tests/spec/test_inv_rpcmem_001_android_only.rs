//! INV-RPCMEM-001 — `RpcmemAllocator::new()` 는 호스트(non-Android) 빌드에서
//! 컴파일 자체에서 제외되거나 호출 시 `Err` 반환. host 빌드에서
//! `OpenCLBackend::new_with_options(_, true)` 호출 시 자동 강등 + stderr warning 1회.
//!
//! 대응 spec: `spec/30-engine.md` 부록 E.2 (ENG-RPCMEM-011), `spec/41-invariants.md` §3.27.
//! 대응 arch: `arch/rpcmem_allocator.md` §2.2, `arch/opencl_backend.md` §2.2.
//!
//! 본 파일은 Sprint 2a Phase 2 spec/arch 단계의 **테스트 skeleton** 이다.
//! Implementer (task #3) 가 RpcmemAllocator + OpenCLBackend::new_with_options
//! 구현 후 본 테스트 본문을 채운다. 현재는 컴파일 가능한 `unimplemented!` placeholder.

#![allow(dead_code, unused_imports)]

// 호스트 빌드 검증:
// (a) `RpcmemAllocator::new()` 호출 시 Err 반환 (또는 type 자체가 사용 불가).
// (b) `OpenCLBackend::new_with_options(false, true)` 호출 시 `opencl_rpcmem`
//     필드가 false 로 강등됨 + stderr 에 "강등" 문자열 포함.
//
// Android 빌드는 본 테스트의 대상 외 (별도 디바이스 microbench 에서 검증).

#[cfg(not(target_os = "android"))]
#[test]
fn host_build_rpcmem_allocator_returns_err_or_excluded() {
    // TODO Implementer: 다음 중 하나로 구현
    //   1. `RpcmemAllocator::new().is_err()` 검증 (host 에서 컴파일 가능 case).
    //   2. `#[cfg(target_os = "android")]` 로 type 자체가 host 에서 부재 →
    //      별도의 host stub (Infallible 등) 부재를 컴파일 타임 검증.
    eprintln!("[INV-RPCMEM-001 skeleton] TODO: Implementer 가 본 테스트 본문 채움");
}

#[cfg(not(target_os = "android"))]
#[test]
fn host_build_opencl_backend_demotes_rpcmem_flag() {
    // TODO Implementer:
    //   1. `OpenCLBackend::new_with_options(false, true)` 호출 (host 에서 opencl
    //      backend init 가능 — GPU 부재 시 backend init 자체가 Err 일 수 있으므로
    //      mock context 사용 또는 skip).
    //   2. stderr capture (예: `gag::BufferRedirect`) 로 "강등" 또는 "demote" 문자열 포함 검증.
    //   3. backend.get_extension(EXT_RPCMEM_ALLOCATOR) == None 검증.
    eprintln!("[INV-RPCMEM-001 skeleton] TODO: Implementer 가 본 테스트 본문 채움");
}

#[cfg(target_os = "android")]
#[test]
fn android_build_test_runs_on_device() {
    // Android 디바이스 microbench 에서만 의미. 호스트 cargo test 에서는 skip.
    // 실 검증은 `microbench_rpcmem_smoke` (task #3 산출물) 또는 디바이스 e2e.
    eprintln!("[INV-RPCMEM-001 skeleton] Android target — 디바이스 측정 대상.");
}
