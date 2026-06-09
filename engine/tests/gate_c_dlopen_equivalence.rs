//! GATE-C 재증명 게이트 (ADR-0009 D7) — dlopen 된 stage plugin 의 **plan-identity**.
//!
//! dlopen 이 바꾸는 유일한 것 = `KVCacheStage` 인스턴스가 정적 슬라이스에서 오느냐 vtable 에서
//! 오느냐다. 따라서 게이트 = 동일 입력 ctx 에 대해 산출 `KVCachePlan` 이 알려진 정답(=정적 stage 가
//! 내는 것과 동일)인지. token-identity 는 틀린 게이트(ADR-0008) — 여기서 쓰지 않는다.
//!
//! **자기-충돌 회피**: 본 통합 테스트 바이너리는 `example_keep_recent` 를 Rust dep 으로 참조하지
//! 않는다(`.so` 를 cargo subprocess 로 빌드 후 dlopen). 따라서 그 이름의 정적 linkme 등록이 이
//! 바이너리에 끌려오지 않아(`find_stage("example_keep_recent")==None`) dlopen 등록이 충돌 없이 성립한다.
//!
//! **process-global 레지스트리**: `DYN_REGISTRY` 가 OnceLock 전역이라, 병렬 테스트 간섭을 피하려고
//! 모든 단언을 단일 `#[test]` 안에서 순차 수행한다.
//!
//! fat-LTO 생존(정적 builtin 의 `--gc-sections` 잔존)은 기존 release `ensure_builtin_stages_registered`
//! self-test 가 담당한다(ADR-0003 §4) — 동적 경로는 런타임 dlopen 이라 LTO 무관.

use std::path::PathBuf;
use std::process::Command;

use llm_rs2::pressure::eviction::stage_registry::{
    dynamic_registered_names, make_stage, register_dynamic_stages,
};
use technique_api::{KVCachePlan, KeepSpec, StageCtx, StageParams, TensorHandle, TensorKind};

/// KeepRecent plan 산출이 읽는 최소 ctx 스텁(current_pos/target_len 만 의미).
struct MockCtx {
    cur: usize,
    tgt: usize,
}
impl StageCtx for MockCtx {
    fn current_pos(&self) -> usize {
        self.cur
    }
    fn target_len(&self) -> usize {
        self.tgt
    }
    fn layer_idx(&self) -> usize {
        0
    }
    fn importance(&self) -> Option<&[f32]> {
        None
    }
    fn n_kv_heads(&self) -> usize {
        1
    }
    fn head_dim(&self) -> usize {
        64
    }
    fn tensor(&self, _kind: TensorKind) -> Option<&dyn TensorHandle> {
        None
    }
}

fn default_params() -> StageParams {
    StageParams {
        eviction_window: 0,
        protected_prefix: 0,
        keep_ratio: 0.0,
        sink_size: 0,
        streaming_window: 0,
    }
}

/// `cargo build -p example-keep-recent [--features plugin-cdylib]` 로 `.so` 를 산출하고, 산출 경로를
/// `CARGO_TARGET_TMPDIR` 의 고유 이름으로 복사해 반환한다(feature ON/OFF `.so` 가 같은 산출 경로를
/// 덮어쓰는 것을 회피).
fn build_plugin_so(with_export: bool, dst_name: &str) -> PathBuf {
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args([
        "build",
        "-p",
        "example-keep-recent",
        "--message-format=json",
    ]);
    if with_export {
        cmd.args(["--features", "plugin-cdylib"]);
    }
    let out = cmd
        .output()
        .expect("cargo build -p example-keep-recent 실행 실패");
    assert!(
        out.status.success(),
        "cargo build .so 실패:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // --message-format=json: example_keep_recent artifact 라인에서 .so 경로 추출(serde_json 무의존 스캔).
    let stdout = String::from_utf8_lossy(&out.stdout);
    let src = stdout
        .lines()
        .filter(|l| l.contains("compiler-artifact") && l.contains("example_keep_recent"))
        .flat_map(|l| l.split('"'))
        .find(|tok| tok.ends_with(".so"))
        .map(PathBuf::from)
        .expect("example_keep_recent .so 산출 경로(filenames)");
    let dst = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(dst_name);
    std::fs::copy(&src, &dst)
        .unwrap_or_else(|e| panic!("{} → {} 복사 실패: {e}", src.display(), dst.display()));
    dst
}

/// GATE-C v1 완료 게이트: dlopen plan-identity + registry merge + fail-fast reject 2종.
/// (전역 DYN_REGISTRY 간섭 회피를 위해 단일 테스트에서 순차 단언.)
#[test]
fn gate_c_dlopen_stage_plan_identity_and_rejects() {
    let so_on = build_plugin_so(true, "libexample_keep_recent_on.so");
    let so_off = build_plugin_so(false, "libexample_keep_recent_off.so");

    // ── (1) 동적 등록 성공 (정적 충돌 없음 — 이 바이너리엔 example_keep_recent 정적 미등록) ──
    register_dynamic_stages(std::slice::from_ref(&so_on))
        .expect("plugin .so 동적 등록 성공해야 한다");

    // ── (2) registry merge: 동적 이름 가시화 ──
    assert!(
        dynamic_registered_names().contains(&"example_keep_recent".to_string()),
        "dynamic_registered_names 에 example_keep_recent 가 있어야 한다: {:?}",
        dynamic_registered_names()
    );

    // ── (3) plan-identity: dlopen stage 의 plan 이 KeepRecent 알려진 정답과 동일 ──
    // KeepRecent: current_pos=100, target_len=30 → keep [70..100] ascending, merges 없음.
    let stage = make_stage("example_keep_recent", &default_params())
        .expect("make_stage 가 동적 stage 를 돌려줘야 한다");
    let plan: KVCachePlan = stage
        .plan(&MockCtx { cur: 100, tgt: 30 })
        .expect("plan Some");
    match plan.keep {
        KeepSpec::LayerWide(keep) => {
            assert_eq!(
                keep,
                (70..100).collect::<Vec<usize>>(),
                "keep LayerWide [70..100]"
            );
        }
        KeepSpec::PerHead(_) => panic!("LayerWide 여야 한다(PerHead 아님)"),
    }
    assert!(plan.merges.is_empty(), "merges 비어야 한다");
    // current <= target → no-op(None).
    assert!(
        stage.plan(&MockCtx { cur: 20, tgt: 30 }).is_none(),
        "current<=target → no-op"
    );

    // ── (4) reject: 동일 이름 재등록 = 중복 fail-fast(빌트인 충돌과 동일 코드 경로) ──
    let dup = register_dynamic_stages(std::slice::from_ref(&so_on));
    assert!(dup.is_err(), "동일 stage 이름 재등록은 거부돼야 한다");
    let msg = dup.unwrap_err().to_string();
    assert!(
        msg.contains("동적 등록") || msg.contains("충돌"),
        "중복 거부 메시지: {msg}"
    );

    // ── (5) reject: register_kv_stage_v1 심볼 없는 .so = 심볼 부재 fail-fast ──
    let no_sym = register_dynamic_stages(std::slice::from_ref(&so_off));
    assert!(
        no_sym.is_err(),
        "register_kv_stage_v1 부재 .so 는 거부돼야 한다"
    );
    assert!(
        no_sym
            .unwrap_err()
            .to_string()
            .contains("register_kv_stage_v1"),
        "심볼 부재 메시지에 register_kv_stage_v1 언급"
    );
}
