// build.rs — generates QNN SDK Rust bindings into OUT_DIR when `qnn` feature is on.
// SDK headers live in `third_party/qnn_sdk_2.33/include/QNN/` (gitignored, EULA-bound).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "qnn")]
    qnn_bindgen::run();
}

#[cfg(feature = "qnn")]
mod qnn_bindgen {
    use std::env;
    use std::path::PathBuf;

    pub fn run() {
        // SDK location (relative to workspace root, since cargo runs build.rs in crate dir)
        let workspace_root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
            .parent()
            .unwrap()
            .to_path_buf();
        let sdk_inc = workspace_root.join("third_party/qnn_sdk_2.33/include/QNN");
        if !sdk_inc.exists() {
            println!(
                "cargo:warning=QNN SDK not found at {} — set up `third_party/qnn_sdk_2.33/include/QNN/` for `qnn` feature builds",
                sdk_inc.display()
            );
            // Emit empty bindings so include!() works even without SDK
            let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
            std::fs::write(
                out_path.join("qnn_bindings.rs"),
                b"// QNN SDK headers missing\n",
            )
            .unwrap();
            return;
        }
        println!("cargo:rerun-if-changed={}", sdk_inc.display());

        let header = sdk_inc.join("QnnInterface.h");
        let target = env::var("TARGET").unwrap_or_else(|_| "x86_64-unknown-linux-gnu".to_string());
        let mut builder = bindgen::Builder::default()
            .header(header.to_string_lossy())
            .clang_arg(format!("-I{}", sdk_inc.display()))
            .clang_arg("-x")
            .clang_arg("c")
            .clang_arg(format!("--target={}", target))
            .allowlist_function("Qnn.*")
            .allowlist_type("Qnn_.*")
            .allowlist_type("QnnInterface_.*")
            .allowlist_var("QNN_.*")
            .layout_tests(false)
            .derive_default(false)
            .derive_debug(false)
            .generate_comments(false);

        // Cross-compile: use NDK sysroot to avoid host gnu/stubs-32.h dependency
        if target.contains("android") {
            if let Ok(ndk) = env::var("ANDROID_NDK_HOME").or_else(|_| env::var("NDK_HOME")) {
                let sysroot =
                    format!("{}/toolchains/llvm/prebuilt/linux-x86_64/sysroot", ndk);
                if std::path::Path::new(&sysroot).exists() {
                    builder = builder.clang_arg(format!("--sysroot={}", sysroot));
                } else {
                    println!("cargo:warning=NDK sysroot {} not found", sysroot);
                }
            } else {
                // Fallback: hardcoded /opt/android-ndk (matches hosts.toml default)
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
}
