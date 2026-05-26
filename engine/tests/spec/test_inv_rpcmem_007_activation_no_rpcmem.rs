//! INV-RPCMEM-007 — `OpenCLMemory::alloc` (activation tensor) 는 `opencl_rpcmem`
//! 값과 무관하게 rpcmem heap 을 사용하지 않는다. KV cache (`alloc_kv`) 와
//! precision swap secondary 전용.
//!
//! 대응 spec: `spec/30-engine.md` 부록 E.3 (ENG-RPCMEM-023), `spec/41-invariants.md` §3.27.
//! 대응 arch: `arch/opencl_backend.md` §3.3.
//!
//! 본 파일은 Sprint 2a Phase 2 spec/arch 단계의 **테스트 skeleton** 이다.

#![allow(dead_code, unused_imports)]

// 검증 방식:
//   1. OpenCLMemory::new_with_rpcmem(_, _, _, Some(allocator)) 로 메모리 생성.
//   2. mem.alloc(1024, DType::F32) 호출 (activation tensor).
//   3. 반환된 Arc<dyn Buffer> 가 RpcmemKvBuffer 변형이 **아님** 검증
//      (downcast `as_any().downcast_ref::<RpcmemKvBuffer>().is_none()`).
//   4. 대신 OpenCLBuffer 또는 UnifiedBuffer 변형이어야 함.
//   5. mem.alloc_kv(1024, DType::F16) 는 RpcmemKvBuffer 변형이 맞아야 함 (대조군).

#[cfg(target_os = "android")]
#[test]
fn activation_alloc_does_not_use_rpcmem_heap() {
    // TODO Implementer: device test.
    //   1. OpenCLBackend::new_with_options(_, true) + allocator 추출.
    //   2. OpenCLMemory::new_with_rpcmem(...).
    //   3. alloc vs alloc_kv 의 반환 Buffer downcast 결과 비교.
    eprintln!("[INV-RPCMEM-007 skeleton] TODO: Implementer device test 본문 작성");
}

#[cfg(not(target_os = "android"))]
#[test]
fn host_build_static_check() {
    // host 빌드는 rpcmem allocator init 불가 → 대신 alloc 메서드의
    // source-grep 으로 rpcmem path 부재 검증 (옵션).
    eprintln!("[INV-RPCMEM-007 skeleton] host: source-grep 검증은 INV-RPCMEM-008 와 중복 — skip");
}
