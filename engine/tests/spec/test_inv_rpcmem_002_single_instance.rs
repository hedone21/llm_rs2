//! INV-RPCMEM-002 — `--opencl-rpcmem` 활성 시 OpenCLBackend 와 RpcmemSecondaryStore
//! 가 보유하는 `Arc<RpcmemAllocator>` 는 동일 인스턴스 (single allocator per session).
//!
//! 대응 spec: `spec/30-engine.md` 부록 E.3/E.4 (ENG-RPCMEM-012, ENG-RPCMEM-032).
//! 대응 arch: `arch/rpcmem_allocator.md` §3, `arch/precision_swap.md` §4.
//!
//! 본 파일은 Sprint 2a Phase 2 spec/arch 단계의 **테스트 skeleton** 이다.

#![allow(dead_code, unused_imports)]

// 검증 방식:
// 1. OpenCLBackend::new_with_options(_, true) 로 backend 생성.
// 2. backend.rpcmem_allocator() 또는 EXT_RPCMEM_ALLOCATOR extension 으로 Arc#A 추출.
// 3. RpcmemSecondaryStore::from_gguf(... backend, allocator) 로 store 생성.
// 4. store 내부 allocator field (test-only accessor 추가 필요) 의 Arc#B 추출.
// 5. `Arc::as_ptr(&A) == Arc::as_ptr(&B)` 검증.
//
// Android 디바이스에서만 의미. host 빌드는 RpcmemAllocator init 자체 불가.

#[cfg(target_os = "android")]
#[test]
fn single_allocator_instance_shared_across_consumers() {
    // TODO Implementer:
    //   1. Mock or real OpenCLBackend init (Galaxy S25 디바이스).
    //   2. backend 와 RpcmemSecondaryStore 의 allocator Arc ptr equality 검증.
    //   3. 추가: OpenCLMemory 도 같은 ptr 검증 (3-way equality).
    eprintln!("[INV-RPCMEM-002 skeleton] TODO: Implementer 가 device test 본문 작성");
}

#[cfg(not(target_os = "android"))]
#[test]
fn host_build_skip() {
    // Host 에선 RpcmemAllocator init 불가 → 테스트 skip.
    // 실 검증은 디바이스에서.
    eprintln!("[INV-RPCMEM-002 skeleton] host build skip (Android-only)");
}
