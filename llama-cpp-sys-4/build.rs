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

/// Return a string that uniquely identifies the current state of the llama.cpp
/// submodule so we know when a re-copy is needed.
///
/// Priority:
/// 1. The commit hash from the submodule's git HEAD (most precise).
/// 2. The mtime of `CMakeLists.txt` (fallback for non-git trees).
fn llama_src_version(src: &Path) -> String {
    // In a git submodule the `.git` entry is a *file* whose content is:
    //   gitdir: ../../.git/modules/llama-cpp-sys-4/llama.cpp
    let git_file = src.join(".git");
    if git_file.is_file() {
        if let Ok(text) = std::fs::read_to_string(&git_file) {
            if let Some(rel) = text.strip_prefix("gitdir:").map(str::trim) {
                let head_path = git_file.parent().unwrap().join(rel).join("HEAD");
                if let Ok(head) = std::fs::read_to_string(&head_path) {
                    // HEAD is either a commit hash or "ref: refs/heads/…"
                    let head = head.trim();
                    if head.starts_with("ref:") {
                        // Resolve the ref to the actual commit hash.
                        let ref_path = head
                            .strip_prefix("ref:")
                            .map(str::trim)
                            .unwrap_or(head);
                        let commit_path =
                            git_file.parent().unwrap().join(rel).join(ref_path);
                        if let Ok(hash) = std::fs::read_to_string(commit_path) {
                            return hash.trim().to_owned();
                        }
                    }
                    return head.to_owned();
                }
            }
        }
    }
    // Fallback: modification time of the top-level CMakeLists.txt.
    src.join("CMakeLists.txt")
        .metadata()
        .and_then(|m| m.modified())
        .map(|t| format!("{t:?}"))
        .unwrap_or_else(|_| "unknown".to_owned())
}

/// Copy a directory tree.  This runs on the *host*, so cfg!(unix/windows) is correct here.
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

/// Extract library names from the build output directory.
///
/// `target` is the Rust target triple of the *cross-compilation target* so
/// that the correct file extensions are chosen even when cross-compiling.
fn extract_lib_names(out_dir: &Path, build_shared_libs: bool, target: &str) -> Vec<String> {
    let lib_pattern = if target.contains("windows") {
        "*.lib"
    } else if target.contains("apple") {
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

/// Extract shared-library asset paths from the build output directory.
///
/// `target` is the Rust target triple of the *cross-compilation target*.
fn extract_lib_assets(out_dir: &Path, target: &str) -> Vec<PathBuf> {
    let shared_lib_pattern = if target.contains("windows") {
        "*.dll"
    } else if target.contains("apple") {
        "*.dylib"
    } else {
        "*.so"
    };

    let shared_libs_dir = if target.contains("windows") { "bin" } else { "lib" };
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

/// Ask a clang binary for its library search path (macOS link helper).
///
/// `clang_binary` should be the bare name or full path of the clang binary to
/// query — e.g. `"clang"` for native builds or `"aarch64-apple-darwin-clang"`
/// for a cross-compiler.
fn macos_link_search_path(clang_binary: &str) -> Option<String> {
    let output = Command::new(clang_binary)
        .arg("--print-search-dirs")
        .output()
        .ok()?;
    if !output.status.success() {
        println!(
            "failed to run '{clang_binary} --print-search-dirs', continuing without a link search path"
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

/// Map a Rust target triple to the CMake `CMAKE_SYSTEM_NAME` value.
fn cmake_system_name(target: &str) -> &'static str {
    if target.contains("-android") || target.contains("android-") {
        "Android"
    } else if target.contains("-apple-ios") {
        "iOS"
    } else if target.contains("-apple-") {
        "Darwin"
    } else if target.contains("-windows") {
        "Windows"
    } else if target.contains("-linux") {
        "Linux"
    } else {
        // Generic UNIX-like fallback
        "Linux"
    }
}

/// Derive a MinGW cross-compiler binary name from a Rust `windows-gnu` target triple.
///
/// Rust uses `x86_64-pc-windows-gnu` / `x86_64-pc-windows-gnullvm` while the
/// MinGW toolchain conventionally uses `x86_64-w64-mingw32`.  The `gnullvm`
/// variant uses Clang instead of GCC.
///
/// Returns `None` for `windows-msvc` targets — MSVC cannot cross-compile from
/// a non-Windows host and users must supply `CC`/`CXX` themselves.
fn mingw_compiler(target: &str, cxx: bool) -> Option<String> {
    if !target.contains("windows-gnu") {
        return None;
    }
    let arch = if target.contains("x86_64") {
        "x86_64"
    } else if target.contains("i686") || target.contains("i586") {
        "i686"
    } else if target.contains("aarch64") {
        "aarch64"
    } else {
        target.split('-').next()?
    };
    // `gnullvm` targets use LLVM/Clang; plain `gnu` targets use GCC.
    let compiler = if target.contains("gnullvm") {
        if cxx { "clang++" } else { "clang" }
    } else {
        if cxx { "g++" } else { "gcc" }
    };
    Some(format!("{}-w64-mingw32-{}", arch, compiler))
}

/// Map a Rust target triple to the CMake `CMAKE_SYSTEM_PROCESSOR` value.
fn cmake_system_processor(target: &str) -> String {
    let arch = target.split('-').next().unwrap_or("unknown");
    match arch {
        "x86_64" => "x86_64".to_owned(),
        "i686" | "i386" => "x86".to_owned(),
        "aarch64" | "arm64" => "aarch64".to_owned(),
        "armv7" | "armv7s" | "armv7k" => "armv7-a".to_owned(),
        "arm" => "arm".to_owned(),
        "riscv64gc" | "riscv64" => "riscv64".to_owned(),
        "powerpc64le" => "ppc64le".to_owned(),
        "powerpc64" => "ppc64".to_owned(),
        "s390x" => "s390x".to_owned(),
        "wasm32" => "wasm32".to_owned(),
        other => other.to_owned(),
    }
}

fn main() {
    let target = env::var("TARGET").unwrap();
    let host = env::var("HOST").unwrap();
    let is_cross = host != target;
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

    debug_log!("HOST: {}", host);
    debug_log!("TARGET: {}", target);
    debug_log!("CROSS_COMPILING: {}", is_cross);
    debug_log!("CARGO_MANIFEST_DIR: {}", manifest_dir);
    debug_log!("TARGET_DIR: {}", target_dir.display());
    debug_log!("OUT_DIR: {}", out_dir.display());
    debug_log!("BUILD_SHARED: {}", build_shared_libs);

    // ── Source copy with version tracking ────────────────────────────────────
    // The copy only ran when the OUT_DIR was fresh, so updating the submodule
    // (which adds/removes files like ggml-cpu/) would silently use stale data.
    // We now store the current submodule HEAD in a sentinel file and re-copy
    // whenever it changes.
    let sentinel = out_dir.join(".llama-src-version");
    let current_version = llama_src_version(&llama_src);
    let stored_version = std::fs::read_to_string(&sentinel).unwrap_or_default();
    let needs_copy = !llama_dst.exists() || stored_version.trim() != current_version.trim();
    if needs_copy {
        if llama_dst.exists() {
            debug_log!("Source version changed — removing stale OUT_DIR copy");
            std::fs::remove_dir_all(&llama_dst).ok();
        }
        debug_log!("Copy {} to {}", llama_src.display(), llama_dst.display());
        copy_folder(&llama_src, &llama_dst);
        std::fs::write(&sentinel, &current_version)
            .expect("failed to write source version sentinel");
    }
    // Tell cargo to rerun this script when the submodule HEAD changes.
    // In a git submodule, llama.cpp/.git is a file pointing at the real HEAD.
    let submodule_git = llama_src.join(".git");
    if submodule_git.is_file() {
        // .git file contains "gitdir: ../../.git/modules/llama-cpp-sys-4/llama.cpp"
        if let Ok(contents) = std::fs::read_to_string(&submodule_git) {
            if let Some(gitdir) = contents.strip_prefix("gitdir:").map(|s| s.trim()) {
                let head = submodule_git.parent().unwrap().join(gitdir).join("HEAD");
                if head.exists() {
                    println!("cargo:rerun-if-changed={}", head.display());
                }
            }
        }
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

    // Point CC/CXX at the MPI wrappers when building with MPI on macOS.
    // Check the *target* OS, not the host, so that cross-compilation from a
    // macOS host to a non-Apple target does not accidentally set these.
    if cfg!(feature = "mpi") && target.contains("apple") {
        // TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { env::set_var("CC", "/opt/homebrew/bin/mpicc") };
        // TODO: Audit that the environment access only happens in single-threaded code.
        unsafe { env::set_var("CXX", "/opt/homebrew/bin/mpicxx") };
    }

    // ── Bindgen ──────────────────────────────────────────────────────────────
    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .generate_comments(true)
        // https://github.com/rust-lang/rust-bindgen/issues/1834
        // "fatal error: 'string' file not found" on macOS
        .clang_arg("-xc++")
        .clang_arg("-std=c++17")
        // When cross-compiling, tell libclang/bindgen the target triple so
        // that layout, pointer sizes, and type widths are computed for the
        // *target* architecture rather than the host.
        .clang_arg(format!("--target={}", target))
        // .raw_line("#![feature(unsafe_extern_blocks)]") // https://github.com/rust-lang/rust/issues/123743
        .clang_arg(format!("-I{}", llama_dst.join("include").display()))
        .clang_arg(format!("-I{}", llama_dst.join("ggml/include").display()))
        .clang_arg(format!("-I{}", llama_dst.join("src").display()))
        .clang_arg(format!("-I{}", llama_dst.join("common").display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_partialeq(true)
        // Do not derive PartialEq on types that contain function-pointer fields.
        // Deriving PartialEq on those triggers the
        // `unpredictable_function_pointer_comparisons` lint on newer rustc
        // because function addresses are not stable across codegen units.
        // macOS FILE internals (function pointers _close/_read/_seek/_write)
        .no_partialeq("__sFILE")
        .no_partialeq("ggml_cplan")
        .no_partialeq("ggml_type_traits")
        .no_partialeq("ggml_type_traits_cpu")
        .no_partialeq("ggml_context")
        .no_partialeq("ggml_opt_params")
        .no_partialeq("llama_model_params")
        .no_partialeq("llama_context_params")
        .no_partialeq("llama_sampler_i")
        .no_partialeq("llama_opt_params")
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
        .opaque_type("std::.*");

    // Add RPC support if feature is enabled
    if cfg!(feature = "rpc") {
        builder = builder
            .clang_arg("-DRPC_SUPPORT")
            .allowlist_function("ggml_backend_rpc_.*")
            .allowlist_type("ggml_backend_rpc_.*");
    }

    // Add mtmd (multimodal) support if feature is enabled
    if cfg!(feature = "mtmd") {
        builder = builder
            .clang_arg("-DMTMD_SUPPORT")
            .clang_arg(format!("-I{}", llama_dst.join("tools/mtmd").display()))
            .allowlist_function("mtmd_.*")
            .allowlist_type("mtmd_.*")
            .allowlist_item("MTMD_.*")
            .no_partialeq("mtmd_context_params");
    }

    let bindings = builder
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

    // ── CMake build ──────────────────────────────────────────────────────────

    let mut config = Config::new(&llama_dst);

    // Would require extra source files to pointlessly
    // be included in what's uploaded to and downloaded from
    // crates.io, so deactivating these instead
    config.define("LLAMA_BUILD_TESTS", "OFF");
    config.define("LLAMA_BUILD_EXAMPLES", "OFF");
    config.define("LLAMA_BUILD_SERVER", "OFF");

    // Build tools (including the mtmd library) only when the mtmd feature is
    // requested.  Common is also required because the CMakeLists gate for
    // tools is `if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS)`.
    if cfg!(feature = "mtmd") {
        config.define("LLAMA_BUILD_TOOLS", "ON");
        config.define("LLAMA_BUILD_COMMON", "ON");
    } else {
        config.define("LLAMA_BUILD_TOOLS", "OFF");
    }

    config.define(
        "BUILD_SHARED_LIBS",
        if build_shared_libs { "ON" } else { "OFF" },
    );

    // ── Cross-compilation CMake configuration ────────────────────────────────
    // When building for a different target than the host, tell CMake the
    // target system so that it does not auto-detect the host as the target.
    // Android is handled separately below via its NDK toolchain file.
    if is_cross && !target.contains("android") {
        let system_name = cmake_system_name(&target);
        let system_processor = cmake_system_processor(&target);
        debug_log!("Cross-compiling: CMAKE_SYSTEM_NAME={system_name} CMAKE_SYSTEM_PROCESSOR={system_processor}");
        config.define("CMAKE_SYSTEM_NAME", system_name);
        config.define("CMAKE_SYSTEM_PROCESSOR", &system_processor);

        // CMake only sets CMAKE_CROSSCOMPILING=TRUE automatically when
        // CMAKE_SYSTEM_NAME differs from the host OS name.  For same-OS
        // cross-arch builds (e.g. x86_64-linux → aarch64-linux) the OS
        // names are identical, so CMAKE_CROSSCOMPILING stays FALSE and
        // ggml's guard (`if (CMAKE_CROSSCOMPILING)` in ggml/CMakeLists.txt)
        // never fires — leaving GGML_NATIVE_DEFAULT=ON and causing
        // `-march=native` (tuned for the build host) to be baked into the
        // target binary, which crashes with SIGILL on the target.
        // Force the flag explicitly so ggml always sees it.
        config.define("CMAKE_CROSSCOMPILING", "TRUE");

        if target.contains("apple") {
            // ── Apple cross-arch (e.g. x86_64-apple-darwin → aarch64-apple-darwin) ──
            //
            // Apple's Clang is already a universal cross-compiler; switching
            // to a different compiler binary is neither needed nor possible
            // (there is no `aarch64-apple-darwin-gcc` in Xcode).  The right
            // CMake knob for same-SDK Apple cross-arch builds is
            // CMAKE_OSX_ARCHITECTURES, which makes Clang add the `-arch`
            // flag automatically.
            let osx_arch = if target.contains("aarch64") || target.contains("arm64") {
                "arm64"
            } else if target.contains("x86_64") {
                "x86_64"
            } else if target.contains("i686") {
                "i386"
            } else {
                // Fallback: strip the vendor/OS suffix and use the raw arch.
                target.split('-').next().unwrap_or("arm64")
            };
            config.define("CMAKE_OSX_ARCHITECTURES", osx_arch);
            debug_log!("Apple cross-arch: CMAKE_OSX_ARCHITECTURES={osx_arch}");

            // Propagate an explicit SDK path when the caller provides one.
            if let Ok(sdk) = env::var("CMAKE_OSX_SYSROOT") {
                config.define("CMAKE_OSX_SYSROOT", &sdk);
            }
            // Honour an explicit compiler override (e.g. osxcross), but do
            // NOT guess a compiler name: the system Clang is always correct
            // for same-SDK cross-arch and osxcross users set CC themselves.
            if let Ok(cc) = env::var("CC") {
                config.define("CMAKE_C_COMPILER", &cc);
            }
            if let Ok(cxx) = env::var("CXX") {
                config.define("CMAKE_CXX_COMPILER", &cxx);
            }
        } else {
            // ── Non-Apple cross-compilation ───────────────────────────────────────
            //
            // Honour CC / CXX set by the caller (e.g. cargo cross, zig cc, …).
            // If they are not set:
            //  • Windows GNU targets  → derive the MinGW triple name
            //    (e.g. x86_64-pc-windows-gnu → x86_64-w64-mingw32-gcc)
            //  • Windows MSVC targets → no safe default; MSVC cannot
            //    cross-compile from a non-Windows host, so the user must
            //    supply CC/CXX (e.g. clang-cl via a sysroot).
            //  • Everything else      → {target-triple}-gcc / g++
            if let Ok(cc) = env::var("CC") {
                config.define("CMAKE_C_COMPILER", &cc);
            } else if let Some(cc) = mingw_compiler(&target, false) {
                config.define("CMAKE_C_COMPILER", &cc);
            } else if !target.contains("windows-msvc") {
                config.define("CMAKE_C_COMPILER", format!("{}-gcc", target));
            }

            if let Ok(cxx) = env::var("CXX") {
                config.define("CMAKE_CXX_COMPILER", &cxx);
            } else if let Some(cxx) = mingw_compiler(&target, true) {
                config.define("CMAKE_CXX_COMPILER", &cxx);
            } else if !target.contains("windows-msvc") {
                config.define("CMAKE_CXX_COMPILER", format!("{}-g++", target));
            }

            // Propagate a sysroot when provided (e.g. via --sysroot or
            // CARGO_TARGET_<TRIPLE>_RUSTFLAGS / CMAKE_SYSROOT env var).
            if let Ok(sysroot) = env::var("CMAKE_SYSROOT") {
                config.define("CMAKE_SYSROOT", &sysroot);
            }
        }
    }

    // ── GGML_NATIVE ──────────────────────────────────────────────────────────
    // GGML_NATIVE=ON tells ggml to detect and use the *build host's* CPU
    // features (e.g. -march=native, check_cxx_source_runs for ARM NEON/SVE,
    // FindSIMD.cmake for MSVC).  That is wrong for cross-compilation: the
    // probed features belong to the build host, not the target, so the
    // resulting binary would crash with SIGILL on a different microarch.
    //
    // Override the cmake default explicitly so a stale CMakeCache.txt can
    // never re-enable it after the user switches from a native to a cross
    // build in the same OUT_DIR.
    if is_cross {
        // Belt-and-suspenders: even though CMAKE_CROSSCOMPILING=TRUE above
        // already causes ggml to default GGML_NATIVE to OFF, we pin it here
        // too so the cmake crate's cache-skip path cannot resurrect a
        // previously cached ON value.
        config.define("GGML_NATIVE", "OFF");
    } else if cfg!(feature = "native") {
        // The `native` Cargo feature explicitly opts in to host-CPU
        // optimisation for non-cross builds.
        config.define("GGML_NATIVE", "ON");
    } else {
        // Default native builds to OFF so that the resulting library is
        // portable across machines of the same architecture (matching the
        // behaviour users expect from a Rust crate).
        config.define("GGML_NATIVE", "OFF");
    }

    // Disable OpenMP on 32-bit ARM Windows (compiler support is absent).
    // Use the TARGET env var, not cfg!(), so the check works when
    // cross-compiling from a non-Windows host.
    if target.contains("windows") && target.starts_with("arm") && !target.starts_with("aarch64") {
        config.define("GGML_OPENMP", "OFF");
    }

    // static_crt (MSVC /MT vs /MD) is meaningless for MinGW; only set it for
    // MSVC targets to avoid confusing CMake on windows-gnu cross builds.
    if target.contains("windows-msvc") {
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
        if target.contains("windows") {
            let vulkan_path = env::var("VULKAN_SDK")
                .expect("Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set");
            let vulkan_lib_path = Path::new(&vulkan_path).join("Lib");
            println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
            println!("cargo:rustc-link-lib=vulkan-1");
        }

        if target.contains("linux") {
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

    if cfg!(feature = "mpi") {
        config.define("LLAMA_MPI", "ON");
    }

    if cfg!(feature = "rpc") {
        config.define("GGML_RPC", "ON");
    }

    // General
    config
        .profile(&profile)
        .very_verbose(std::env::var("CMAKE_VERBOSE").is_ok()) // Not verbose by default
        .always_configure(false);

    // The cmake crate skips re-configuration when CMakeCache.txt already exists
    // (always_configure = false).  If a previous run left a CMakeCache.txt but
    // never wrote the actual build-system files (Makefile / build.ninja), the
    // subsequent `cmake --build` call fails with "No such file or directory".
    // Detect that broken state and remove CMakeCache.txt so cmake is forced to
    // configure from scratch.
    {
        let cmake_build_dir = out_dir.join("build");
        let cache = cmake_build_dir.join("CMakeCache.txt");
        if cache.exists() {
            let has_makefile = cmake_build_dir.join("Makefile").exists();
            let has_ninja = cmake_build_dir.join("build.ninja").exists();
            if !has_makefile && !has_ninja {
                debug_log!(
                    "CMakeCache.txt exists but no Makefile/build.ninja found — \
                     removing cache to force reconfiguration"
                );
                std::fs::remove_file(&cache)
                    .expect("failed to remove stale CMakeCache.txt");
            }
        }
    }

    let build_dir = config.build();

    // ── Link search paths ────────────────────────────────────────────────────
    println!("cargo:rustc-link-search={}", out_dir.join("lib").display());
    println!(
        "cargo:rustc-link-search={}",
        out_dir.join("lib64").display()
    );
    println!("cargo:rustc-link-search={}", build_dir.display());

    // ── Link libraries ───────────────────────────────────────────────────────
    let llama_libs_kind = if build_shared_libs { "dylib" } else { "static" };
    let llama_libs = extract_lib_names(&out_dir, build_shared_libs, &target);
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

    // OpenMP: link gomp when the cmake build enabled it (GGML_OPENMP_ENABLED=ON).
    // This can happen even without the "openmp" feature because cmake's FindOpenMP
    // is invoked unconditionally on some platforms (e.g. ARM) when the
    // ggml-cpu CMakeLists includes OpenMP support at the variant level.
    let cmake_cache_path = out_dir.join("build").join("CMakeCache.txt");
    let openmp_enabled_in_cmake = std::fs::read_to_string(&cmake_cache_path)
        .map(|contents| contents.contains("GGML_OPENMP_ENABLED:INTERNAL=ON"))
        .unwrap_or(false);

    if cfg!(feature = "openmp") || openmp_enabled_in_cmake {
        if target.contains("gnu") || target.contains("musl") {
            println!("cargo:rustc-link-lib=gomp");
        }
    }

    // msvcrtd is the MSVC debug CRT — it does not exist in MinGW toolchains.
    if cfg!(debug_assertions) && target.contains("windows-msvc") {
        println!("cargo:rustc-link-lib=dylib=msvcrtd");
    }

    // macOS frameworks and libc++
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=c++");
    }

    // Linux libstdc++
    if target.contains("linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // On (older) macOS / Apple targets we may need to link against the clang
    // runtime, which is hidden in a non-default path.
    // More details at https://github.com/alexcrichton/curl-rust/issues/279.
    if target.contains("apple") {
        // For same-SDK Apple cross-arch builds (e.g. x86_64-apple-darwin →
        // aarch64-apple-darwin) the host's plain `clang` is still the right
        // binary to ask: both arches share the same Xcode SDK and therefore
        // the same library search directories.
        //
        // For osxcross (Linux → macOS) the user sets CC, so we honour that;
        // we do NOT guess a `{target}-clang` name because it is not a stable
        // convention and the SDK paths it would report are likely wrong anyway.
        let clang_bin = env::var("CC").unwrap_or_else(|_| "clang".to_owned());
        if let Some(path) = macos_link_search_path(&clang_bin) {
            println!("cargo:rustc-link-lib=clang_rt.osx");
            println!("cargo:rustc-link-search={}", path);
        }
    }

    // ── Copy shared-library assets to the Cargo target directory ─────────────
    if build_shared_libs {
        let libs_assets = extract_lib_assets(&out_dir, &target);
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
