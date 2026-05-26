//! INV-RPCMEM-003 — rpcmem alloc 실패 시 per-buffer fallback (UnifiedBuffer 또는
//! SecondaryUnavailable → GGUF mmap). session 전체 abort 금지.
//!
//! 대응 spec: `spec/30-engine.md` 부록 E.3 (ENG-RPCMEM-022), `spec/41-invariants.md` §3.27.
//! 대응 arch: `arch/opencl_backend.md` §3.2, `arch/precision_swap.md` §4.
//!
//! 본 파일은 Sprint 2a Phase 2 spec/arch 단계의 **테스트 skeleton** 이다.

#![allow(dead_code, unused_imports)]

// 검증 방식 (host 빌드에서도 가능 — mocked allocator):
//
// 1. RpcmemAllocator 또는 그 trait 를 mocked 구현으로 교체 (test-only feature
//    `rpcmem-mock` 또는 trait inversion).
// 2. mocked allocator 가 `alloc()` 호출 시 즉시 Err 반환.
// 3a. OpenCLMemory::alloc_kv(size, F16) 호출 → 반환된 Buffer 가 RpcmemKvBuffer
//     변형이 **아님** (UnifiedBuffer/OpenCLBuffer 변형). downcast 또는
//     `Buffer::is_host_managed` 등으로 구분.
// 3b. RpcmemSecondaryStore::from_gguf(...) 호출 → Err 반환 + caller
//     (secondary_mmap::open_secondary) 가 일반 GGUF mmap path 로 fallback 성공.
// 4. 두 경우 모두 session 진행 (panic / abort 없음).

#[test]
fn rpcmem_alloc_failure_falls_back_to_unified_buffer() {
    // TODO Implementer: mocked RpcmemAllocator trait 또는 fault-injection 으로
    // OpenCLMemory::alloc_kv 가 UnifiedBuffer 반환함을 검증.
    eprintln!("[INV-RPCMEM-003 skeleton] TODO: Implementer 가 mocked allocator 본문 작성");
}

#[test]
fn rpcmem_secondary_alloc_failure_falls_back_to_gguf_mmap() {
    // TODO Implementer: RpcmemSecondaryStore::from_gguf 가 Err 반환 시
    // secondary_mmap::open_secondary 가 GGUF mmap 경로로 fallback 함을 검증.
    eprintln!("[INV-RPCMEM-003 skeleton] TODO: Implementer 가 본문 작성");
}
