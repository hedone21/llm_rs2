// Generates QNN SDK bindings into OUT_DIR. Mirrors engine/build.rs.

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
    let sdk_inc = workspace_root.join("third_party/qnn_sdk_2.33/include/QNN");
    if !sdk_inc.exists() {
        println!("cargo:warning=QNN SDK not found at {}", sdk_inc.display());
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        std::fs::write(out_path.join("qnn_bindings.rs"), b"// QNN SDK headers missing\n").unwrap();
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
