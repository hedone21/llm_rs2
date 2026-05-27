// Generates QNN SDK bindings into OUT_DIR. Mirrors engine/build.rs.

/// SDK include path resolution (B-4):
/// 1. `QNN_SDK_ROOT` env var (worktree / CI 명시 override).
/// 2. `<workspace_root>/third_party/qnn_sdk_2.33/include/QNN` (default).
/// 3. git worktree 시 main repo `<main>/third_party/...` fallback —
///    `.git` file에 적힌 `gitdir: <main>/.git/worktrees/<name>` 추적.
///
/// 모두 실패 시 primary path 반환 (warning + empty bindings 출력).
fn resolve_qnn_sdk_inc(workspace_root: &std::path::Path) -> std::path::PathBuf {
    use std::env;
    use std::path::PathBuf;

    if let Ok(env_root) = env::var("QNN_SDK_ROOT") {
        let p = PathBuf::from(env_root).join("include/QNN");
        if p.exists() {
            return p;
        }
    }
    let primary = workspace_root.join("third_party/qnn_sdk_2.33/include/QNN");
    if primary.exists() {
        return primary;
    }
    let git_file = workspace_root.join(".git");
    if let Ok(contents) = std::fs::read_to_string(&git_file)
        && let Some(line) = contents.lines().next()
        && let Some(gitdir) = line.strip_prefix("gitdir: ")
    {
        let gitdir = PathBuf::from(gitdir.trim());
        // gitdir layout: <main>/.git/worktrees/<name>
        if let Some(main_repo) = gitdir
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
        {
            let fallback = main_repo.join("third_party/qnn_sdk_2.33/include/QNN");
            if fallback.exists() {
                return fallback;
            }
        }
    }
    primary
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    use std::env;
    use std::path::PathBuf;

    let workspace_root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let sdk_inc = resolve_qnn_sdk_inc(&workspace_root);
    if !sdk_inc.exists() {
        println!("cargo:warning=QNN SDK not found at {}", sdk_inc.display());
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        std::fs::write(
            out_path.join("qnn_bindings.rs"),
            b"// QNN SDK headers missing\n",
        )
        .unwrap();
        return;
    }
    println!("cargo:rerun-if-changed={}", sdk_inc.display());

    let target = env::var("TARGET").unwrap_or_else(|_| "x86_64-unknown-linux-gnu".to_string());
    // Process GPU specialization first so that the full struct definitions for
    // _QnnOpPackage_Node_t and _QnnOpPackage_OpImpl_t are emitted (they are forward
    // declared as opaque in QnnOpPackage.h).
    let mut builder = bindgen::Builder::default()
        .header(sdk_inc.join("GPU/QnnGpuOpPackage.h").to_string_lossy())
        .header(sdk_inc.join("QnnInterface.h").to_string_lossy())
        .clang_arg(format!("-I{}", sdk_inc.display()))
        .clang_arg("-x")
        .clang_arg("c")
        .clang_arg(format!("--target={}", target))
        .allowlist_function("Qnn.*")
        .allowlist_type("Qnn.*")
        .allowlist_type("QnnInterface_.*")
        .allowlist_type("QnnGpu.*")
        .allowlist_type("QnnOpPackage.*")
        .allowlist_type("_Qnn.*")
        .allowlist_var("QNN_.*")
        .layout_tests(false)
        .derive_default(false)
        .derive_debug(false)
        .generate_comments(false);

    if target.contains("android") {
        if let Ok(ndk) = env::var("ANDROID_NDK_HOME").or_else(|_| env::var("NDK_HOME")) {
            let sysroot = format!("{}/toolchains/llvm/prebuilt/linux-x86_64/sysroot", ndk);
            if std::path::Path::new(&sysroot).exists() {
                builder = builder.clang_arg(format!("--sysroot={}", sysroot));
            }
        } else {
            let sysroot = "/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot";
            if std::path::Path::new(sysroot).exists() {
                builder = builder.clang_arg(format!("--sysroot={}", sysroot));
            }
        }
    }

    let bindings = builder.generate().expect("bindgen QNN headers failed");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("qnn_bindings.rs"))
        .expect("write QNN bindings");
}
