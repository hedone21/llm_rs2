//! INV-RPCMEM-006 — `--opencl-rpcmem` + `--backend qnn_oppkg | qnngpu` 동시 지정 시
//! 후자가 우선되며 전자는 무시 (stderr 경고 1회). Sprint 2a 호환 mutex.
//! Sprint 2b 에서 qnn_oppkg backend 삭제 시 본 mutex 도 자연 만료.
//!
//! 대응 spec: `spec/30-engine.md` 부록 E.5 (ENG-RPCMEM-041), `spec/41-invariants.md` §3.27.
//! 대응 arch: `arch/rpcmem_allocator.md` §3.1, `arch/opencl_backend.md` §5.
//!
//! 본 파일은 Sprint 2a Phase 2 spec/arch 단계의 **테스트 skeleton** 이다.
//! CLI parser test — host 빌드에서 실행 가능.

#![allow(dead_code, unused_imports)]

// 검증 방식:
//   1. CliArgs::try_parse_from(&["llm_rs2", "--backend", "qnn_oppkg", "--opencl-rpcmem", ...])
//      가 OK 반환 (parse 실패하지 않음).
//   2. session::init 또는 CliArgs::resolve_effective() 에서 effective config
//      의 `opencl_rpcmem == false`.
//   3. stderr capture (gag / assert_cmd) 로 "무시" 또는 "ignored" 또는
//      "qnn_oppkg" 문자열 포함 검증.

#[test]
fn cli_opencl_rpcmem_demoted_when_qnn_oppkg_backend_used() {
    // TODO Implementer:
    //   1. clap::CliArgs::try_parse_from 또는 session::cli::parse_args 호출.
    //   2. effective config 의 opencl_rpcmem == false 검증.
    //   3. stderr capture 로 warning 메시지 검증.
    eprintln!("[INV-RPCMEM-006 skeleton] TODO: Implementer 가 CLI parser 본문 작성");
}

#[test]
fn cli_opencl_rpcmem_active_when_backend_opencl() {
    // TODO Implementer:
    //   1. --backend opencl --opencl-rpcmem 조합 parse.
    //   2. effective config 의 opencl_rpcmem == true.
    //   3. stderr 에 warning 부재.
    eprintln!("[INV-RPCMEM-006 skeleton] TODO: Implementer 가 본문 작성");
}
