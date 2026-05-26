//! INV-RPCMEM-004 — `RpcmemAllocator` 모듈은 `libQnnGpu.so` / `libqnn_oppkg.so` 를
//! dlopen 하지 않는다. libcdsprpc.so 단독 의존.
//!
//! 대응 spec: `spec/30-engine.md` 부록 E.6 (ENG-RPCMEM-C01), `spec/41-invariants.md` §3.27.
//! 대응 arch: `arch/rpcmem_allocator.md` §2.
//!
//! 본 파일은 Sprint 2a Phase 2 spec/arch 단계의 **테스트 skeleton** 이다.
//! source-grep 기반 — host 빌드에서도 무조건 실행.

#![allow(dead_code, unused_imports)]

use std::fs;
use std::path::PathBuf;

fn allocator_source_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("memory")
        .join("rpcmem")
        .join("allocator.rs")
}

#[test]
fn allocator_source_does_not_reference_libqnngpu() {
    let path = allocator_source_path();
    if !path.exists() {
        // Implementer 가 아직 allocator.rs 를 만들지 않은 단계 — placeholder PASS.
        eprintln!(
            "[INV-RPCMEM-004 skeleton] {} 미존재 (Implementer task #3 대기) — PASS",
            path.display()
        );
        return;
    }
    let src = fs::read_to_string(&path).expect("read allocator.rs");
    let forbidden = ["libQnnGpu.so", "libqnn_oppkg.so", "QNN_BACKEND_LIB", "QNN_OPPKG_LIB"];
    for literal in forbidden {
        assert!(
            !src.contains(literal),
            "INV-RPCMEM-004 위반: allocator.rs 가 '{literal}' literal 포함. libcdsprpc.so 만 dlopen 해야 함."
        );
    }
}
