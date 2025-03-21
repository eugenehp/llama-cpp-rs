use cmake::Config;
use glob::glob;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

macro_rules! debug_log {
    ($($arg:tt)*) => {
        if std::env::var("BUILD_DEBUG").is_ok() {
            println!("cargo:warning=[DEBUG] {}", format!($($arg)*));
        }
    };
}

fn get_cargo_target_dir() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    let profile = std::env::var("PROFILE")?;
    let mut target_dir = None;
    let mut sub_path = out_dir.as_path();
    while let Some(parent) = sub_path.parent() {
        if parent.ends_with(&profile) {
            target_dir = Some(parent);
            break;
        }
        sub_path = parent;
    }
    let target_dir = target_dir.ok_or("not found")?;
    Ok(target_dir.to_path_buf())
}

fn copy_folder(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).expect("Failed to create dst directory");
    if cfg!(unix) {
        std::process::Command::new("cp")
            .arg("-rf")
            .arg(src)
            .arg(dst.parent().unwrap())
            .status()
            .expect("Failed to execute cp command");
    }

    if cfg!(windows) {
        std::process::Command::new("robocopy.exe")
            .arg("/e")
            .arg(src)
            .arg(dst)
            .status()
            .expect("Failed to execute robocopy command");
    }
}

fn extract_lib_names(out_dir: &Path, build_shared_libs: bool) -> Vec<String> {
    let lib_pattern = if cfg!(windows) {
        "*.lib"
    } else if cfg!(target_os = "macos") {
        if build_shared_libs {
            "*.dylib"
        } else {
            "*.a"
        }
    } else {
        if build_shared_libs {
            "*.so"
        } else {
            "*.a"
        }
    };
    let libs_dir = out_dir.join("lib*");
    let pattern = libs_dir.join(lib_pattern);
    debug_log!("Extract libs {}", pattern.display());

    let mut lib_names: Vec<String> = Vec::new();

    // Process the libraries based on the pattern
    for entry in glob(pattern.to_str().unwrap()).unwrap() {
        match entry {
            Ok(path) => {
                let stem = path.file_stem().unwrap();
                let stem_str = stem.to_str().unwrap();

                // Remove the "lib" prefix if present
                let lib_name = if stem_str.starts_with("lib") {
                    stem_str.strip_prefix("lib").unwrap_or(stem_str)
                } else {
                    stem_str
                };
                lib_names.push(lib_name.to_string());
            }
            Err(e) => println!("cargo:warning=error={}", e),
        }
    }
    lib_names
}

fn extract_lib_assets(out_dir: &Path) -> Vec<PathBuf> {
    let shared_lib_pattern = if cfg!(windows) {
        "*.dll"
    } else if cfg!(target_os = "macos") {
        "*.dylib"
    } else {
        "*.so"
    };

    let shared_libs_dir = if cfg!(windows) { "bin" } else { "lib" };
    let libs_dir = out_dir.join(shared_libs_dir);
    let pattern = libs_dir.join(shared_lib_pattern);
    debug_log!("Extract lib assets {}", pattern.display());
    let mut files = Vec::new();

    for entry in glob(pattern.to_str().unwrap()).unwrap() {
        match entry {
            Ok(path) => {
                files.push(path);
            }
            Err(e) => eprintln!("cargo:warning=error={}", e),
        }
    }

    files
}

fn macos_link_search_path() -> Option<String> {
    let output = Command::new("clang")
        .arg("--print-search-dirs")
        .output()
        .ok()?;
    if !output.status.success() {
        println!(
            "failed to run 'clang --print-search-dirs', continuing without a link search path"
        );
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.contains("libraries: =") {
            let path = line.split('=').nth(1)?;
            return Some(format!("{}/lib/darwin", path));
        }
    }

    println!("failed to determine link search path, continuing without it");
    None
}

fn main() {
    let target = env::var("TARGET").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let target_dir = get_cargo_target_dir().unwrap();
    let llama_dst = out_dir.join("llama.cpp");
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR");
    let llama_src = Path::new(&manifest_dir).join("llama.cpp");
    let build_shared_libs = cfg!(feature = "cuda") || cfg!(feature = "dynamic-link");

    let build_shared_libs = std::env::var("LLAMA_BUILD_SHARED_LIBS")
        .map(|v| v == "1")
        .unwrap_or(build_shared_libs);
    let profile = env::var("LLAMA_LIB_PROFILE").unwrap_or("Release".to_string());
    let static_crt = env::var("LLAMA_STATIC_CRT")
        .map(|v| v == "1")
        .unwrap_or(false);

    debug_log!("TARGET: {}", target);
    debug_log!("CARGO_MANIFEST_DIR: {}", manifest_dir);
    debug_log!("TARGET_DIR: {}", target_dir.display());
    debug_log!("OUT_DIR: {}", out_dir.display());
    debug_log!("BUILD_SHARED: {}", build_shared_libs);

    if !llama_dst.exists() {
        debug_log!("Copy {} to {}", llama_src.display(), llama_dst.display());
        copy_folder(&llama_src, &llama_dst);
    }
    // Speed up build
    // TODO: Audit that the environment access only happens in single-threaded code.
    unsafe {
        env::set_var(
            "CMAKE_BUILD_PARALLEL_LEVEL",
            std::thread::available_parallelism()
                .unwrap()
                .get()
                .to_string(),
        )
    };

    // point to CC and CXX binaries on macOS
    if cfg!(all(feature = "mpi", target_os = "macos")) {
        // TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { env::set_var("CC", "/opt/homebrew/bin/mpicc") };
        // TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { env::set_var("CXX", "/opt/homebrew/bin/mpicxx") };
    }

    // Bindings
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .generate_comments(true)
        // https://github.com/rust-lang/rust-bindgen/issues/1834
        // "fatal error: 'string' file not found" on macOS
        .clang_arg("-xc++")
        .clang_arg("-std=c++11")
        // .raw_line("#![feature(unsafe_extern_blocks)]") // https://github.com/rust-lang/rust/issues/123743
        .clang_arg(format!("-I{}", llama_dst.join("include").display()))
        .clang_arg(format!("-I{}", llama_dst.join("ggml/include").display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_partialeq(true)
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_function("llama_.*")
        .allowlist_function("llama_lora_.*")
        .allowlist_type("llama_.*")
        .allowlist_function("common_token_to_piece")
        .allowlist_function("common_tokenize")
        // .allowlist_item("common_.*")
        // .allowlist_function("common_tokenize")
        // .allowlist_function("common_detokenize")
        // .allowlist_type("common_.*")
        // .allowlist_item("common_params")
        // .allowlist_item("common_sampler_type")
        // .allowlist_item("common_sampler_params")
        .allowlist_item("LLAMA_.*")
        // .opaque_type("common_lora_adapter_info")
        .opaque_type("llama_grammar")
        .opaque_type("llama_grammar_parser")
        .opaque_type("llama_sampler_chain")
        // .opaque_type("llama_context_deleter")
        // .blocklist_type("llama_model_deleter")
        .opaque_type("std::.*")
        // .layout_tests(false)
        // .derive_default(true)
        // .enable_cxx_namespaces()
        .use_core()
        .prepend_enum_name(false)
        .generate()
        .expect("Failed to generate bindings");

    // Write the generated bindings to an output file
    let bindings_path = out_dir.join("bindings.rs");
    bindings
        .write_to_file(bindings_path.clone())
        .expect("Failed to write bindings");

    // temporary fix for https://github.com/rust-lang/rust/issues/123743 in
    // cargo +nightly build
    let contents = std::fs::read_to_string(bindings_path.clone()).unwrap();
    let contents = contents.replace("unsafe extern \"C\" {", " extern \"C\" {");
    fs::write(bindings_path, contents).unwrap();

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=./sherpa-onnx");

    debug_log!("Bindings Created");

    // Build with Cmake

    let mut config = Config::new(&llama_dst);

    // Would require extra source files to pointlessly
    // be included in what's uploaded to and downloaded from
    // crates.io, so deactivating these instead
    config.define("LLAMA_BUILD_TESTS", "OFF");
    config.define("LLAMA_BUILD_EXAMPLES", "OFF");
    config.define("LLAMA_BUILD_SERVER", "OFF");

    config.define(
        "BUILD_SHARED_LIBS",
        if build_shared_libs { "ON" } else { "OFF" },
    );

    // use BLAS instead of OpenMP
    // if cfg!(target_os = "macos") {
    //     config.define("GGML_BLAS", "OFF");
    // }

    // see https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md
    if cfg!(all(target_os = "windows", target_arch = "arm")) {
        config.define("GGML_OPENMP", "OFF");
    }

    if cfg!(windows) {
        config.static_crt(static_crt);
    }

    if target.contains("android") && target.contains("aarch64") {
        // build flags for android taken from this doc
        // https://github.com/ggerganov/llama.cpp/blob/master/docs/android.md
        let android_ndk = env::var("ANDROID_NDK")
            .expect("Please install Android NDK and ensure that ANDROID_NDK env variable is set");
        config.define(
            "CMAKE_TOOLCHAIN_FILE",
            format!("{android_ndk}/build/cmake/android.toolchain.cmake"),
        );
        config.define("ANDROID_ABI", "arm64-v8a");
        config.define("ANDROID_PLATFORM", "android-28");
        config.define("CMAKE_SYSTEM_PROCESSOR", "arm64");
        config.define("CMAKE_C_FLAGS", "-march=armv8.7a");
        config.define("CMAKE_CXX_FLAGS", "-march=armv8.7a");
        config.define("GGML_OPENMP", "OFF");
        config.define("GGML_LLAMAFILE", "OFF");
    }

    if cfg!(feature = "vulkan") {
        config.define("GGML_VULKAN", "ON");
        if cfg!(windows) {
            let vulkan_path = env::var("VULKAN_SDK")
                .expect("Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set");
            let vulkan_lib_path = Path::new(&vulkan_path).join("Lib");
            println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
            println!("cargo:rustc-link-lib=vulkan-1");
        }

        if cfg!(target_os = "linux") {
            println!("cargo:rustc-link-lib=vulkan");
        }
    }

    if cfg!(feature = "cuda") {
        config.define("GGML_CUDA", "ON");
    }

    if cfg!(feature = "openmp") {
        config.define("GGML_OPENMP", "ON");
    } else {
        config.define("GGML_OPENMP", "OFF");
    }

    if cfg!(all(feature = "mpi")) {
        config.define("LLAMA_MPI", "ON");
    }

    // General
    config
        .profile(&profile)
        .very_verbose(std::env::var("CMAKE_VERBOSE").is_ok()) // Not verbose by default
        .always_configure(false);

    let build_dir = config.build();

    // Search paths
    println!("cargo:rustc-link-search={}", out_dir.join("lib").display());
    println!(
        "cargo:rustc-link-search={}",
        out_dir.join("lib64").display()
    );
    println!("cargo:rustc-link-search={}", build_dir.display());

    // Link libraries
    let llama_libs_kind = if build_shared_libs { "dylib" } else { "static" };
    let llama_libs = extract_lib_names(&out_dir, build_shared_libs);
    assert_ne!(llama_libs.len(), 0);

    for lib in llama_libs {
        debug_log!(
            "LINK {}",
            format!("cargo:rustc-link-lib={}={}", llama_libs_kind, lib)
        );
        println!(
            "{}",
            format!("cargo:rustc-link-lib={}={}", llama_libs_kind, lib)
        );
    }

    // OpenMP
    if cfg!(feature = "openmp") {
        if target.contains("gnu") {
            println!("cargo:rustc-link-lib=gomp");
        }
    }

    // Windows debug
    if cfg!(all(debug_assertions, windows)) {
        println!("cargo:rustc-link-lib=dylib=msvcrtd");
    }

    // macOS
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=c++");
    }

    // Linux
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    if target.contains("apple") {
        // On (older) OSX we need to link against the clang runtime,
        // which is hidden in some non-default path.
        //
        // More details at https://github.com/alexcrichton/curl-rust/issues/279.
        if let Some(path) = macos_link_search_path() {
            println!("cargo:rustc-link-lib=clang_rt.osx");
            println!("cargo:rustc-link-search={}", path);
        }
    }

    // copy DLLs to target
    if build_shared_libs {
        let libs_assets = extract_lib_assets(&out_dir);
        for asset in libs_assets {
            let asset_clone = asset.clone();
            let filename = asset_clone.file_name().unwrap();
            let filename = filename.to_str().unwrap();
            let dst = target_dir.join(filename);
            debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
            if !dst.exists() {
                std::fs::hard_link(asset.clone(), dst).unwrap();
            }

            // Copy DLLs to examples as well
            if target_dir.join("examples").exists() {
                let dst = target_dir.join("examples").join(filename);
                debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
                if !dst.exists() {
                    std::fs::hard_link(asset.clone(), dst).unwrap();
                }
            }

            // Copy DLLs to target/profile/deps as well for tests
            let dst = target_dir.join("deps").join(filename);
            debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
            if !dst.exists() {
                std::fs::hard_link(asset.clone(), dst).unwrap();
            }
        }
    }
}
