//! GATE-C v2 재증명 게이트 (ADR-0009 D4/D7 — Format 축) — dlopen 된 format plugin 의
//! **descriptor-identity**.
//!
//! dlopen 이 바꾸는 유일한 것 = `KVFormat` 인스턴스가 정적 `KV_FORMATS` 슬라이스에서 오느냐
//! `FormatVTableAbi` vtable 에서 오느냐다. 따라서 게이트 = 동적(vtable) 경로로 만든 format 의
//! `KVLayoutDesc` 가 알려진 정답(= 정적 등록이 내는 q4_0-like descriptor)과 동일한지. encoder
//! byte-identity(floor round-trip)로 그 descriptor 가 엔진 generic floor 를 정상 구동함도 확인한다.
//!
//! **force-link 비대칭(GATE-C v1 과 다름)**: `synth_q4` 는 엔진 lib 이 `use synth_q4_format as _;`
//! (`format/builtin_kv_formats.rs`)로 **무조건 정적 등록**한다 → 이 테스트 바이너리에서
//! `find_kv_format("synth_q4") == Some` → `register_dynamic_formats([synth_q4.so])` 는
//! **builtin-collision reject**(빌트인 우선). 따라서:
//!   - `synth_q4` → builtin-collision reject 를 담당(v1 stage 축이 못 했던 추가 reject 경로).
//!   - `example_kv_format`(동일 q4_0-like layout, 엔진이 **force-link 안 함**) → 동적 등록 성공 +
//!     `make_format` fallback + descriptor-identity + 중복 reject 를 담당.
//! 두 vehicle 의 layout 은 동일(q4_0-like)하므로 동적 descriptor == 정적 `synth_q4` descriptor 가
//! 성립해 목표("dlopen 의 descriptor == 정적 synth_q4")를 그대로 만족한다.
//!
//! **process-global 레지스트리**: `DYN_FORMAT_REGISTRY` 가 OnceLock 전역이라, 병렬 테스트 간섭을
//! 피하려고 모든 단언을 단일 `#[test]` 안에서 순차 수행한다.
//!
//! fat-LTO 생존(정적 builtin 의 `--gc-sections` 잔존)은 기존 `ensure_builtin_kv_formats_registered`
//! self-test 가 담당한다(ADR-0005 D6) — 동적 경로는 런타임 dlopen 이라 LTO 무관.

use std::path::PathBuf;
use std::process::Command;

use llm_rs2::format::dynamic_format_registry::{
    dynamic_registered_format_names, make_format, register_dynamic_formats,
};
use llm_rs2::format::{decode_via_descriptor, encode_via_descriptor};
use technique_api::{KVLayoutDesc, Packing, ScaleLayout, find_kv_format};

/// q4_0-like 알려진 정답 descriptor — synth_q4 / example_kv_format 가 공유(block_elems 32 / bits 4 /
/// PerBlockF16 / Nibble).
fn known_q4_like_desc() -> KVLayoutDesc {
    KVLayoutDesc {
        block_elems: 32,
        bits: 4,
        scale_layout: ScaleLayout::PerBlockF16,
        packing: Packing::Nibble,
    }
}

/// `cargo build -p <pkg> [--features plugin-cdylib] --message-format=json` 으로 `.so` 를 산출하고,
/// 산출 경로를 `CARGO_TARGET_TMPDIR` 의 고유 이름으로 복사해 반환한다(feature ON/OFF `.so` 가 같은
/// 산출 경로를 덮어쓰는 것을 회피).
fn build_plugin_so(pkg: &str, with_export: bool, dst_name: &str) -> PathBuf {
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["build", "-p", pkg, "--message-format=json"]);
    if with_export {
        cmd.args(["--features", "plugin-cdylib"]);
    }
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("cargo build -p {pkg} 실행 실패: {e}"));
    assert!(
        out.status.success(),
        "cargo build {pkg} .so 실패:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // --message-format=json: <pkg> artifact 라인에서 .so 경로 추출(serde_json 무의존 스캔).
    let underscore = pkg.replace('-', "_");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let src = stdout
        .lines()
        .filter(|l| l.contains("compiler-artifact") && l.contains(&underscore))
        .flat_map(|l| l.split('"'))
        .find(|tok| tok.ends_with(".so") && tok.contains(&underscore))
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{pkg} .so 산출 경로(filenames) 미검출"));
    let dst = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(dst_name);
    std::fs::copy(&src, &dst)
        .unwrap_or_else(|e| panic!("{} → {} 복사 실패: {e}", src.display(), dst.display()));
    dst
}

/// GATE-C v2 완료 게이트: dlopen descriptor-identity + floor round-trip + reject 3종.
/// (전역 DYN_FORMAT_REGISTRY 간섭 회피를 위해 단일 테스트에서 순차 단언.)
#[test]
fn gate_c_format_dlopen_descriptor_identity_and_rejects() {
    let synth_on = build_plugin_so("synth-q4-format", true, "libsynth_q4_format_on.so");
    let ex_on = build_plugin_so("example-kv-format", true, "libexample_kv_format_on.so");
    let ex_off = build_plugin_so("example-kv-format", false, "libexample_kv_format_off.so");

    // ── (1) builtin-collision reject: synth_q4 는 엔진 force-link 정적 등록 → 동적 등록 거부 ──
    // (이 단언은 레지스트리를 변경하지 않는다 — push 전 bail.)
    assert!(
        find_kv_format("synth_q4").is_some(),
        "전제: synth_q4 가 엔진 force-link 로 정적 등록돼 있어야 한다(builtin_kv_formats)"
    );
    let collide = register_dynamic_formats(std::slice::from_ref(&synth_on));
    assert!(
        collide.is_err(),
        "synth_q4 .so 는 빌트인 충돌로 거부돼야 한다"
    );
    let msg = collide.unwrap_err().to_string();
    assert!(
        msg.contains("충돌") || msg.contains("빌트인"),
        "builtin-collision 거부 메시지: {msg}"
    );

    // ── (2) 동적 등록 성공: example_kv_format(비force-link) ──
    // 이 바이너리엔 example_kv_format 정적 미등록 → 충돌 없이 성립.
    assert!(
        find_kv_format("example_kv_format").is_none(),
        "전제: example_kv_format 는 정적 미등록이어야 한다(엔진 dep 아님)"
    );
    register_dynamic_formats(std::slice::from_ref(&ex_on))
        .expect("example_kv_format .so 동적 등록 성공해야 한다");

    // ── (3) registry merge: 동적 이름 가시화 ──
    assert!(
        dynamic_registered_format_names().contains(&"example_kv_format".to_string()),
        "dynamic_registered_format_names 에 example_kv_format 가 있어야 한다: {:?}",
        dynamic_registered_format_names()
    );

    // ── (4) descriptor-identity: 동적(vtable) make_format 의 descriptor == 정적 synth_q4 == 알려진 정답 ──
    let dyn_fmt = make_format("example_kv_format")
        .expect("make_format 가 동적 format 을 돌려줘야 한다(정적 miss → vtable fallback)");
    assert_eq!(dyn_fmt.name(), "example_kv_format");
    let dyn_desc = dyn_fmt.layout();
    assert_eq!(
        dyn_desc,
        known_q4_like_desc(),
        "dlopen vtable 경로의 KVLayoutDesc 가 알려진 q4_0-like 정답과 동일(무손실 마샬링)"
    );
    let synth_static_desc = (find_kv_format("synth_q4").unwrap().make)().layout();
    assert_eq!(
        dyn_desc, synth_static_desc,
        "dlopen 의 descriptor == 정적 synth_q4 의 descriptor(둘 다 q4_0-like)"
    );

    // ── (5) descriptor-floor round-trip byte-identical: dlopen desc 와 정적 desc 가 엔진 floor 를 동일 구동 ──
    let numel = 64usize; // block_elems(32) 배수
    let src: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 3.0).collect();
    let nbytes = dyn_desc
        .bytes_for_elems(numel)
        .expect("q4_0-like 64 elems 바이트 수");
    let mut enc_dyn = vec![0u8; nbytes];
    encode_via_descriptor(&dyn_desc, &src, &mut enc_dyn).expect("dlopen desc encode");
    let mut enc_static = vec![0u8; nbytes];
    encode_via_descriptor(&synth_static_desc, &src, &mut enc_static).expect("정적 desc encode");
    assert_eq!(
        enc_dyn, enc_static,
        "dlopen 과 정적 descriptor 의 encode 바이트가 동일(floor 동일 구동)"
    );
    // decode round-trip sanity (q4_0 lossy지만 동일 descriptor라 결과 유한·동일 경로).
    let mut dec = vec![0.0f32; numel];
    decode_via_descriptor(&dyn_desc, &enc_dyn, &mut dec);
    assert!(
        dec.iter().all(|v| v.is_finite()),
        "decode 결과가 유한해야 한다"
    );

    // ── (6) reject: 동일 이름 재등록 = 중복 fail-fast ──
    let dup = register_dynamic_formats(std::slice::from_ref(&ex_on));
    assert!(dup.is_err(), "동일 format 이름 재등록은 거부돼야 한다");
    let dup_msg = dup.unwrap_err().to_string();
    assert!(
        dup_msg.contains("중복") || dup_msg.contains("충돌"),
        "중복 거부 메시지: {dup_msg}"
    );

    // ── (7) reject: register_kv_format_v1 심볼 없는 .so = 심볼 부재 fail-fast ──
    let no_sym = register_dynamic_formats(std::slice::from_ref(&ex_off));
    assert!(
        no_sym.is_err(),
        "register_kv_format_v1 부재 .so 는 거부돼야 한다"
    );
    assert!(
        no_sym
            .unwrap_err()
            .to_string()
            .contains("register_kv_format_v1"),
        "심볼 부재 메시지에 register_kv_format_v1 언급"
    );
}
