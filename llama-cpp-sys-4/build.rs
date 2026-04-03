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

/// Compute a short hash over the contents of all `*.patch` files in `patches_dir`.
/// Returns an empty string when the directory does not exist or contains no patches.
fn patches_hash(patches_dir: &Path) -> String {
    if !patches_dir.is_dir() {
        return String::new();
    }
    let mut entries: Vec<_> = std::fs::read_dir(patches_dir)
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).collect())
        .unwrap_or_default();
    entries.sort();
    let mut hasher_val: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for path in &entries {
        if path.extension().map(|e| e == "patch").unwrap_or(false) {
            if let Ok(bytes) = std::fs::read(path) {
                for &b in &bytes {
                    hasher_val ^= b as u64;
                    hasher_val = hasher_val.wrapping_mul(0x100000001b3);
                }
            }
        }
    }
    format!("{:016x}", hasher_val)
}

/// Apply all `*.patch` files in `patches_dir` (sorted alphabetically) to `dst`.
/// Uses the `patch -p1` command, which must be available on PATH.
/// Skips the patches directory entirely when it does not exist or is empty.
fn apply_patches(patches_dir: &Path, dst: &Path) {
    if !patches_dir.is_dir() {
        return;
    }
    let mut entries: Vec<_> = std::fs::read_dir(patches_dir)
        .expect("failed to read patches dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "patch").unwrap_or(false))
        .collect();
    entries.sort();
    for patch in &entries {
        println!("cargo:warning=Applying patch: {}", patch.display());
        let status = Command::new("patch")
            .arg("-p1")
            .arg("--forward")
            .arg("--directory")
            .arg(dst)
            .arg("--input")
            .arg(patch)
            .status()
            .unwrap_or_else(|e| panic!("failed to run `patch` for {}: {e}", patch.display()));
        if !status.success() {
            panic!(
                "Patch {} failed to apply. The patch may need rebasing against the \
                 current llama.cpp submodule commit.",
                patch.display()
            );
        }
    }
}

/// Return a string that uniquely identifies the current state of the llama.cpp
/// submodule so we know when a re-copy is needed.
///
/// Priority:
/// 1. The commit hash from the submodule's git HEAD (most precise).
/// 2. The mtime of `CMakeLists.txt` (fallback for non-git trees).
fn llama_src_version(src: &Path, patches_dir: &Path) -> String {
    let ph = patches_hash(patches_dir);
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
                        let ref_path = head.strip_prefix("ref:").map(str::trim).unwrap_or(head);
                        let commit_path = git_file.parent().unwrap().join(rel).join(ref_path);
                        if let Ok(hash) = std::fs::read_to_string(commit_path) {
                            return format!("{}:{}", hash.trim(), ph);
                        }
                    }
                    return format!("{}:{}", head, ph);
                }
            }
        }
    }
    // Fallback: modification time of the top-level CMakeLists.txt.
    let base = src
        .join("CMakeLists.txt")
        .metadata()
        .and_then(|m| m.modified())
        .map(|t| format!("{t:?}"))
        .unwrap_or_else(|_| "unknown".to_owned());
    // Mix in patch contents so that updating patches forces a re-copy+re-patch.
    format!("{}:{}", base, patches_hash(patches_dir))
}

/// Copy a directory tree.  This runs on the *host*, so cfg!(unix/windows) is correct here.
/// Copy a directory tree using hardlinks where possible (same-device),
/// falling back to a regular copy.
///
/// Hardlinks are essentially instant (no data is duplicated) and the CMake
/// build writes into its own `build/` subdirectory, so it never modifies the
/// linked source files.
fn copy_folder(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).expect("Failed to create dst directory");
    if cfg!(unix) {
        // Try cp -rl (hardlink) first; fall back to cp -rf if the flag is not
        // supported (e.g. older macOS cp, cross-device copies).
        let status = std::process::Command::new("cp")
            .arg("-rl")
            .arg(src)
            .arg(dst.parent().unwrap())
            .status();
        let ok = status.map(|s| s.success()).unwrap_or(false);
        if !ok {
            std::process::Command::new("cp")
                .arg("-rf")
                .arg(src)
                .arg(dst.parent().unwrap())
                .status()
                .expect("Failed to execute cp command");
        }
    } else if cfg!(windows) {
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
    // MSVC produces `.lib` for both static archives and import libraries.
    // MinGW/GCC (windows-gnu / windows-gnullvm) produces `.a` for static
    // archives and `.dll.a` for import libraries — both end in `.a`, so the
    // single pattern covers both cases.
    let lib_pattern = if target.contains("windows-msvc") {
        "*.lib"
    } else if target.contains("windows") {
        // MinGW / GCC-based Windows toolchain (cross or native).
        // Static libs: libfoo.a  |  Shared import libs: libfoo.dll.a
        "*.a"
    } else if target.contains("apple") {
        if build_shared_libs {
            "*.dylib"
        } else {
            "*.a"
        }
    } else if build_shared_libs {
        "*.so"
    } else {
        "*.a"
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

                // For MinGW import libraries the file is named `libfoo.dll.a`.
                // `file_stem()` strips the final `.a` extension, leaving
                // `libfoo.dll`.  We additionally need to strip the trailing
                // `.dll` so that the link name becomes `foo` rather than
                // `foo.dll`.
                let stem_str = if target.contains("windows")
                    && !target.contains("msvc")
                    && stem_str.ends_with(".dll")
                {
                    &stem_str[..stem_str.len() - 4]
                } else {
                    stem_str
                };

                // Remove the "lib" prefix if present (Unix/MinGW convention).
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

    let shared_libs_dir = if target.contains("windows") {
        "bin"
    } else {
        "lib"
    };
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

fn extract_prebuilt_lib_names(
    prebuilt_root: &Path,
    use_shared_libs: bool,
    target: &str,
) -> Vec<String> {
    let lib_pattern = if target.contains("windows-msvc") {
        "*.lib"
    } else if target.contains("windows") {
        "*.a"
    } else if target.contains("apple") {
        if use_shared_libs {
            "*.dylib"
        } else {
            "*.a"
        }
    } else if use_shared_libs {
        "*.so"
    } else {
        "*.a"
    };

    let mut lib_names = Vec::new();
    for dir in [
        prebuilt_root.to_path_buf(),
        prebuilt_root.join("lib"),
        prebuilt_root.join("lib64"),
        prebuilt_root.join("bin"),
    ] {
        if !dir.exists() {
            continue;
        }
        let pattern = dir.join(lib_pattern);
        let pattern_s = match pattern.to_str() {
            Some(v) => v,
            None => continue,
        };

        for entry in glob(pattern_s).unwrap() {
            match entry {
                Ok(path) => {
                    let stem = match path.file_stem().and_then(|s| s.to_str()) {
                        Some(v) => v,
                        None => continue,
                    };

                    let stem = if target.contains("windows")
                        && !target.contains("msvc")
                        && stem.ends_with(".dll")
                    {
                        &stem[..stem.len() - 4]
                    } else {
                        stem
                    };

                    let lib_name = if let Some(stripped) = stem.strip_prefix("lib") {
                        stripped
                    } else {
                        stem
                    };

                    if !lib_names.iter().any(|n| n == lib_name) {
                        lib_names.push(lib_name.to_string());
                    }
                }
                Err(e) => eprintln!("cargo:warning=error={}", e),
            }
        }
    }

    lib_names
}

fn extract_prebuilt_shared_assets(prebuilt_root: &Path, target: &str) -> Vec<PathBuf> {
    let shared_pattern = if target.contains("windows") {
        "*.dll"
    } else if target.contains("apple") {
        "*.dylib"
    } else {
        "*.so"
    };

    let mut files = Vec::new();
    for dir in [
        prebuilt_root.to_path_buf(),
        prebuilt_root.join("lib"),
        prebuilt_root.join("lib64"),
        prebuilt_root.join("bin"),
    ] {
        if !dir.exists() {
            continue;
        }
        let pattern = dir.join(shared_pattern);
        let pattern_s = match pattern.to_str() {
            Some(v) => v,
            None => continue,
        };
        for entry in glob(pattern_s).unwrap() {
            match entry {
                Ok(path) => {
                    if !files.iter().any(|p| p == &path) {
                        files.push(path);
                    }
                }
                Err(e) => eprintln!("cargo:warning=error={}", e),
            }
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
        if cxx {
            "clang++"
        } else {
            "clang"
        }
    } else {
        if cxx {
            "g++"
        } else {
            "gcc"
        }
    };
    Some(format!("{}-w64-mingw32-{}", arch, compiler))
}

fn command_exists(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn apple_vulkan_available() -> bool {
    let has_glslc = command_exists("glslc");

    let sdk = match env::var("VULKAN_SDK") {
        Ok(v) => PathBuf::from(v),
        Err(_) => return false,
    };

    // LunarG macOS SDK layout:
    //   $VULKAN_SDK/include/vulkan/vulkan.h
    //   $VULKAN_SDK/lib/libvulkan*.dylib
    let header_ok = sdk.join("include").join("vulkan").join("vulkan.h").exists();
    let lib_dir = sdk.join("lib");
    let lib_ok = lib_dir.join("libvulkan.dylib").exists()
        || lib_dir.join("libvulkan.1.dylib").exists()
        || lib_dir.join("libMoltenVK.dylib").exists();

    has_glslc && header_ok && lib_ok
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

/// Find a direct child directory of `parent` whose name matches `name`
/// case-insensitively.  Returns the actual path on disk.
fn find_child_dir_ci(parent: &Path, name: &str) -> Option<PathBuf> {
    let lower = name.to_ascii_lowercase();
    std::fs::read_dir(parent)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.is_dir()
                && p.file_name()
                    .map(|n| n.to_string_lossy().to_ascii_lowercase() == lower)
                    .unwrap_or(false)
        })
}

/// Find a file inside `dir` whose name matches `name` case-insensitively.
fn find_file_ci(dir: &Path, name: &str) -> Option<PathBuf> {
    let lower = name.to_ascii_lowercase();
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.is_file()
                && p.file_name()
                    .map(|n| n.to_string_lossy().to_ascii_lowercase() == lower)
                    .unwrap_or(false)
        })
}

/// Search for `glslc` (or `glslc.exe`) inside the Vulkan SDK.
///
/// Search order:
/// 1. `<sdk>/Bin/glslc.exe` or `<sdk>/bin/glslc` (case-insensitive dir match).
/// 2. Recursively walk `<sdk>` looking for the binary (covers non-standard
///    layouts where `glslc` lives under e.g. `x86_64-windows-msvc/bin/`).
/// 3. Check if `glslc` is already on `PATH`.
fn find_glslc(sdk: &Path) -> Option<PathBuf> {
    let exe_name = if cfg!(windows) { "glslc.exe" } else { "glslc" };

    // 1. Standard location: <sdk>/Bin/ or <sdk>/bin/
    if let Some(bin_dir) = find_child_dir_ci(sdk, "bin") {
        if let Some(found) = find_file_ci(&bin_dir, exe_name) {
            return Some(found);
        }
    }

    // 2. Recursive search (bounded to 3 levels to avoid slow scans).
    if let Some(found) = find_file_recursive(sdk, exe_name, 3) {
        return Some(found);
    }

    // 3. Fall back to PATH.
    if let Ok(output) = std::process::Command::new("where").arg(exe_name).output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = stdout.lines().next() {
                let p = PathBuf::from(line.trim());
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }

    // Also try Unix-style `which` (e.g. Git-for-Windows, MSYS2).
    if let Ok(output) = std::process::Command::new("which").arg(exe_name).output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = stdout.lines().next() {
                let p = PathBuf::from(line.trim());
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }

    None
}

/// Recursively search for a file by name (case-insensitive) up to `max_depth`
/// directory levels.
fn find_file_recursive(dir: &Path, name: &str, max_depth: u32) -> Option<PathBuf> {
    if max_depth == 0 {
        return None;
    }
    let lower = name.to_ascii_lowercase();
    let entries: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();

    // Check files first at this level.
    for p in &entries {
        if p.is_file()
            && p.file_name()
                .map(|n| n.to_string_lossy().to_ascii_lowercase() == lower)
                .unwrap_or(false)
        {
            return Some(p.clone());
        }
    }
    // Then recurse into subdirectories.
    for p in &entries {
        if p.is_dir() {
            if let Some(found) = find_file_recursive(p, name, max_depth - 1) {
                return Some(found);
            }
        }
    }
    None
}

/// Try to locate the Vulkan SDK on Windows automatically.
///
/// Search order:
/// 1. `VULKAN_SDK` environment variable (explicit user override).
/// 2. Windows registry (`HKLM\SOFTWARE\LunarG\VulkanSDK` → latest version key).
/// 3. Default install directory `C:\VulkanSDK\<latest version>`.
///
/// A candidate directory is accepted only when it contains `Lib\vulkan-1.lib`.
fn find_vulkan_sdk_windows() -> Option<PathBuf> {
    // Helper: validate that the directory actually contains the Vulkan library.
    let is_valid = |p: &Path| -> bool { p.join("Lib").join("vulkan-1.lib").exists() };

    // 1. Explicit environment variable.
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        let p = PathBuf::from(&sdk);
        if is_valid(&p) {
            return Some(p);
        }
        // The env var is set but doesn't look right — keep trying the
        // automatic paths so a stale/wrong VULKAN_SDK doesn't block the
        // build when the SDK is actually installed elsewhere.
        debug_log!(
            "VULKAN_SDK env var is set to '{}' but Lib/vulkan-1.lib was not found there; \
             trying automatic detection",
            sdk
        );
    }

    // 2. Windows registry (LunarG installer writes version keys here).
    #[cfg(windows)]
    {
        use winreg::enums::*;
        use winreg::RegKey;
        // The LunarG installer writes per-version keys under:
        //   HKLM\SOFTWARE\LunarG\VulkanSDK\<version>  (InstallDir = …)
        if let Ok(hklm) =
            RegKey::predef(HKEY_LOCAL_MACHINE).open_subkey("SOFTWARE\\LunarG\\VulkanSDK")
        {
            // Enumerate version sub-keys and pick the latest valid one.
            let mut candidates: Vec<(String, PathBuf)> = Vec::new();
            for name in hklm.enum_keys().filter_map(Result::ok) {
                if let Ok(ver_key) = hklm.open_subkey(&name) {
                    if let Ok(install_dir) = ver_key.get_value::<String, _>("InstallDir") {
                        let p = PathBuf::from(&install_dir);
                        if is_valid(&p) {
                            candidates.push((name, p));
                        }
                    }
                }
            }
            // Sort descending by version string so the latest wins.
            candidates.sort_by(|a, b| b.0.cmp(&a.0));
            if let Some((_, p)) = candidates.into_iter().next() {
                return Some(p);
            }
        }
    }

    // 3. Scan the default install directory C:\VulkanSDK\<version>.
    let vulkan_base = PathBuf::from("C:\\VulkanSDK");
    if vulkan_base.is_dir() {
        let mut versions: Vec<PathBuf> = std::fs::read_dir(&vulkan_base)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir() && is_valid(p))
            .collect();
        // Sort descending by directory name so the latest version wins.
        versions.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
        if let Some(p) = versions.into_iter().next() {
            return Some(p);
        }
    }

    None
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

    // ── Windows MAX_PATH workaround ──────────────────────────────────────────
    // Cargo's OUT_DIR already contains a long hashed path segment.  The Vulkan
    // backend adds a deeply-nested ExternalProject sub-build
    // (ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-build/
    //  CMakeFiles/3.x.x/VCTargetsPath/x64/Debug/VCTargetsPath.tlog/…)
    // that pushes the total path well beyond Windows' MAX_PATH of 260 chars,
    // causing MSBuild error MSB3491.
    //
    // Fix: on Windows, redirect cmake's build+install tree to a short path
    // under %LOCALAPPDATA% that is derived from a stable hash of OUT_DIR so
    // different crate versions / OUT_DIRs get their own isolated build trees.
    let cmake_out_dir: PathBuf = if target.contains("windows") {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        out_dir.to_string_lossy().as_ref().hash(&mut hasher);
        // Use 8 hex digits (32-bit) — enough collision-resistance for local builds.
        let hash = hasher.finish() as u32;
        // Prefer %LOCALAPPDATA% (~26 chars on a default Windows install) over
        // %TEMP% because TEMP sometimes points to a longer UNC path.
        let base = std::env::var("LOCALAPPDATA")
            .or_else(|_| std::env::var("TEMP"))
            .or_else(|_| std::env::var("TMP"))
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("C:\\Temp"));
        // Final cmake_out_dir ≈ base(~26) + \llcb\(6) + 8hex(8) = ~40 chars.
        // The deepest vulkan sub-path adds ~167 chars → total ~207 < 260. ✓
        let short_dir = base.join("llcb").join(format!("{:08x}", hash));
        std::fs::create_dir_all(&short_dir)
            .expect("Failed to create short cmake output dir (Windows MAX_PATH workaround)");
        short_dir
    } else {
        out_dir.clone()
    };

    // ── Shared CMake build cache (non-Windows) ────────────────────────────────
    // Cargo assigns a different OUT_DIR hash for every distinct feature
    // combination, so toggling --features q1 lands in a brand-new empty dir
    // and forces a full CMake rebuild (~10 min).  To avoid this we redirect
    // the CMake install/build tree to a shared directory keyed by:
    //   <source-commit> + <active C++ feature flags>
    // so that different Cargo OUT_DIRs that represent the same C++ build can
    // share the compiled artefacts.  The OUT_DIR is still used for bindgen
    // output and the source copy; only the CMake build tree is shared.
    //
    // Skip on Windows because the MAX_PATH workaround already handles this
    // with a short LOCALAPPDATA path derived from OUT_DIR.
    let cmake_out_dir: PathBuf = if !target.contains("windows") {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        // Source version is the submodule commit + patches hash, already
        // computed later; recompute it here so we can use it as the cache key.
        let src_ver = {
            let patches_dir_tmp = Path::new(&manifest_dir).join("patches");
            llama_src_version(&llama_src, &patches_dir_tmp)
        };
        // C++ feature flags that affect the compiled output.
        let cpp_features = format!(
            "cuda={},metal={},vulkan={},webgpu={},blas={},opencl={},hip={},openmp={},rpc={},q1={},native={},shared={}",
            cfg!(feature = "cuda"),
            cfg!(feature = "metal"),
            cfg!(feature = "vulkan"),
            cfg!(feature = "webgpu"),
            cfg!(feature = "blas"),
            cfg!(feature = "opencl"),
            cfg!(feature = "hip"),
            cfg!(feature = "openmp"),
            cfg!(feature = "rpc"),
            cfg!(feature = "q1"),
            cfg!(feature = "native"),
            build_shared_libs,
        );
        let mut hasher = DefaultHasher::new();
        src_ver.hash(&mut hasher);
        cpp_features.hash(&mut hasher);
        target.hash(&mut hasher);
        let hash = hasher.finish();
        // Place the shared cmake dir in the Cargo target directory so it is
        // cleaned by `cargo clean` and lives alongside other build artefacts.
        let shared = target_dir
            .parent() // target/<profile> → target
            .unwrap_or(&target_dir)
            .join("llama-cmake-cache")
            .join(format!("{:016x}", hash));
        std::fs::create_dir_all(&shared).expect("failed to create shared cmake cache dir");
        debug_log!("Shared cmake dir: {}", shared.display());
        shared
    } else {
        cmake_out_dir // Windows already set above
    };

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
    // We now store the current submodule HEAD (plus a hash of any local patches)
    // in a sentinel file and re-copy+re-patch whenever either changes.
    let patches_dir = Path::new(&manifest_dir).join("patches");
    let sentinel = out_dir.join(".llama-src-version");
    let current_version = llama_src_version(&llama_src, &patches_dir);
    let stored_version = std::fs::read_to_string(&sentinel).unwrap_or_default();
    let needs_copy = !llama_dst.exists() || stored_version.trim() != current_version.trim();
    if needs_copy {
        if llama_dst.exists() {
            debug_log!("Source version changed — removing stale OUT_DIR copy");
            std::fs::remove_dir_all(&llama_dst).ok();
        }
        debug_log!("Copy {} to {}", llama_src.display(), llama_dst.display());
        copy_folder(&llama_src, &llama_dst);
        // Apply local patches (only those gated by active Cargo features).
        if cfg!(feature = "q1") {
            let q1_patch = patches_dir.join("0001-q1-quantization.patch");
            if q1_patch.exists() {
                let single_dir = out_dir.join("patches-q1");
                std::fs::create_dir_all(&single_dir).ok();
                std::fs::copy(&q1_patch, single_dir.join("0001-q1-quantization.patch"))
                    .expect("failed to stage q1 patch");
                apply_patches(&single_dir, &llama_dst);
            }
        }
        std::fs::write(&sentinel, &current_version)
            .expect("failed to write source version sentinel");
    }
    // Tell cargo to rerun when the submodule HEAD changes.
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
    // Rerun when any patch file is added, removed, or modified.
    if patches_dir.is_dir() {
        println!("cargo:rerun-if-changed={}", patches_dir.display());
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
    println!("cargo:rerun-if-env-changed=LLAMA_PREBUILT_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_PREBUILT_SHARED");

    debug_log!("Bindings Created");

    // ── Optional prebuilt path (skip CMake compile) ─────────────────────────
    //
    // When LLAMA_PREBUILT_DIR points to a directory containing precompiled
    // llama/ggml libraries, use those directly and skip the full CMake build.
    // Expected layout is flexible; files may be in <dir>, <dir>/lib,
    // <dir>/lib64, or <dir>/bin.
    if let Ok(prebuilt_dir_raw) = env::var("LLAMA_PREBUILT_DIR") {
        let prebuilt_dir = PathBuf::from(&prebuilt_dir_raw);
        if prebuilt_dir.exists() {
            let use_shared_libs = env::var("LLAMA_PREBUILT_SHARED")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on"))
                .unwrap_or(build_shared_libs);

            println!(
                "cargo:warning=Using prebuilt llama libs from {}",
                prebuilt_dir.display()
            );

            for p in [
                prebuilt_dir.clone(),
                prebuilt_dir.join("lib"),
                prebuilt_dir.join("lib64"),
                prebuilt_dir.join("bin"),
            ] {
                if p.exists() {
                    println!("cargo:rustc-link-search={}", p.display());
                }
            }

            let llama_libs_kind = if use_shared_libs { "dylib" } else { "static" };
            let llama_libs = extract_prebuilt_lib_names(&prebuilt_dir, use_shared_libs, &target);
            if llama_libs.is_empty() {
                panic!(
                    "LLAMA_PREBUILT_DIR was set to '{}' but no linkable libraries were found",
                    prebuilt_dir.display()
                );
            }

            for lib in llama_libs {
                println!("cargo:rustc-link-lib={llama_libs_kind}={lib}");
            }

            // Platform runtime links (same as normal CMake path).
            if cfg!(feature = "openmp") && (target.contains("gnu") || target.contains("musl")) {
                println!("cargo:rustc-link-lib=gomp");
            }
            if target.contains("apple") {
                println!("cargo:rustc-link-lib=framework=Foundation");
                if cfg!(feature = "metal") {
                    println!("cargo:rustc-link-lib=framework=Metal");
                    println!("cargo:rustc-link-lib=framework=MetalKit");
                }
                println!("cargo:rustc-link-lib=framework=Accelerate");
                println!("cargo:rustc-link-lib=c++");
            }
            if target.contains("linux") {
                println!("cargo:rustc-link-lib=dylib=stdc++");
            }
            if target.contains("windows") && !target.contains("msvc") {
                println!("cargo:rustc-link-lib=static=stdc++");
                println!("cargo:rustc-link-lib=static=winpthread");
            }
            if target.contains("windows") {
                println!("cargo:rustc-link-lib=advapi32");
            }
            if target.contains("apple") {
                let clang_bin = env::var("CC").unwrap_or_else(|_| "clang".to_owned());
                if let Some(path) = macos_link_search_path(&clang_bin) {
                    println!("cargo:rustc-link-lib=clang_rt.osx");
                    println!("cargo:rustc-link-search={}", path);
                }
            }

            // Copy dynamic runtime assets next to test/example/app outputs.
            if use_shared_libs {
                let libs_assets = extract_prebuilt_shared_assets(&prebuilt_dir, &target);
                for asset in libs_assets {
                    let filename = asset
                        .file_name()
                        .and_then(|f| f.to_str())
                        .expect("invalid prebuilt asset file name");

                    for dst in [
                        target_dir.join(filename),
                        target_dir.join("examples").join(filename),
                        target_dir.join("deps").join(filename),
                    ] {
                        if let Some(parent) = dst.parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        if !dst.exists() {
                            let _ = std::fs::hard_link(&asset, &dst)
                                .or_else(|_| std::fs::copy(&asset, &dst).map(|_| ()));
                        }
                    }
                }
            }

            return;
        }
        panic!(
            "LLAMA_PREBUILT_DIR was set to '{}' but that path does not exist",
            prebuilt_dir.display()
        );
    }

    // ── CMake build ──────────────────────────────────────────────────────────

    let mut config = Config::new(&llama_dst);

    // ── sccache launcher ────────────────────────────────────────────────────
    // When sccache is on PATH (or SCCACHE_PATH is set) and the caller has not
    // explicitly disabled it (LLAMA_NO_SCCACHE=1), wrap every C/C++ compiler
    // call with sccache.  This makes feature-flag toggles essentially free
    // after the first build: toggling --features q1 only recompiles the ~5
    // files changed by the patch; the other 459 are cache hits.
    if env::var("LLAMA_NO_SCCACHE").as_deref() != Ok("1") {
        let sccache = env::var("SCCACHE_PATH")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.exists())
            .or_else(|| {
                let exe = if cfg!(windows) {
                    "sccache.exe"
                } else {
                    "sccache"
                };
                env::var_os("PATH").and_then(|paths| {
                    std::env::split_paths(&paths)
                        .map(|d| d.join(exe))
                        .find(|p| p.exists())
                })
            });
        if let Some(sc) = sccache {
            debug_log!("sccache found at {}", sc.display());
            config.define("CMAKE_C_COMPILER_LAUNCHER", sc.to_str().unwrap());
            config.define("CMAKE_CXX_COMPILER_LAUNCHER", sc.to_str().unwrap());
        }
    }

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

    // ── Apple: always pin CMAKE_OSX_ARCHITECTURES ─────────────────────────
    // When a build orchestrator (e.g. Tauri / Electron) spawns cargo with an
    // explicit `--target aarch64-apple-darwin` on an Apple-Silicon Mac, the
    // host and target triples are identical so `is_cross` is false.  However,
    // the orchestrator's own process may run under Rosetta (x86_64 Node.js)
    // or inject environment variables (ARCHFLAGS, CFLAGS with `-arch x86_64`,
    // etc.) that leak into the cmake build.  Without an explicit architecture
    // pin, cmake would honour those leaked flags and silently produce x86_64
    // object code.  Rust then links that x86_64 code into the arm64 binary,
    // and the first x86 instruction executed at runtime triggers SIGILL.
    //
    // Setting CMAKE_OSX_ARCHITECTURES unconditionally for all Apple targets
    // — even non-cross builds — ensures cmake always produces code for the
    // correct architecture regardless of inherited environment pollution.
    if target.contains("apple") && !is_cross {
        let osx_arch = if target.contains("aarch64") || target.contains("arm64") {
            "arm64"
        } else if target.contains("x86_64") {
            "x86_64"
        } else {
            target.split('-').next().unwrap_or("arm64")
        };
        config.define("CMAKE_OSX_ARCHITECTURES", osx_arch);
        debug_log!("Apple native: CMAKE_OSX_ARCHITECTURES={osx_arch}");
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
    let want_native = cfg!(feature = "native") && !is_cross;
    if is_cross {
        // Belt-and-suspenders: even though CMAKE_CROSSCOMPILING=TRUE above
        // already causes ggml to default GGML_NATIVE to OFF, we pin it here
        // too so the cmake crate's cache-skip path cannot resurrect a
        // previously cached ON value.
        config.define("GGML_NATIVE", "OFF");
    } else if want_native {
        // The `native` Cargo feature explicitly opts in to host-CPU
        // optimisation for non-cross builds.
        config.define("GGML_NATIVE", "ON");
    } else {
        // Default native builds to OFF so that the resulting library is
        // portable across machines of the same architecture (matching the
        // behaviour users expect from a Rust crate).
        config.define("GGML_NATIVE", "OFF");
    }

    // ── ARM portable baseline ────────────────────────────────────────────────
    // When GGML_NATIVE=OFF and no explicit GGML_CPU_ARM_ARCH is set, ggml's
    // cmake does not emit any -march/-mcpu flag.  On Alpine/Debian GCC that
    // is fine because GCC is configured with --with-arch=armv8-a.  But on
    // Apple Clang the arm64-apple-macosx target triple *implicitly* defines
    // __ARM_FEATURE_DOTPROD / __ARM_FEATURE_MATMUL_INT8 (because every
    // M-series chip has them), so cmake's check_cxx_source_compiles sees
    // those macros as defined and compiles dotprod/i8mm intrinsics into the
    // ggml-cpu library — which then crashes with SIGILL on any aarch64 chip
    // that lacks those extensions (Cortex-A53, older Graviton, etc.).
    //
    // Pinning GGML_CPU_ARM_ARCH=armv8-a passes -march=armv8-a to the
    // compiler, which overrides the target-triple default and prevents any
    // higher-arch macros from being defined.  Users who want native
    // performance on their own machine should add --features native, which
    // takes the GGML_NATIVE=ON path above and skips this block entirely.
    let is_arm_target = target.starts_with("aarch64") || target.starts_with("arm");
    if is_arm_target && !want_native && !target.contains("android") {
        // Don't override if the caller already set a custom arch via env var
        // (e.g. GGML_CPU_ARM_ARCH=armv8.2-a+dotprod for a specific fleet).
        if env::var("GGML_CPU_ARM_ARCH").is_err() {
            config.define("GGML_CPU_ARM_ARCH", "armv8-a");
        }
    }

    // ── x86 portable baseline ─────────────────────────────────────────────
    // When GGML_NATIVE=OFF on a *native* (non-cross) build, cmake computes
    // GGML_NATIVE_DEFAULT=ON (because CMAKE_CROSSCOMPILING is FALSE).  The
    // INS_ENB variable then becomes ON:
    //
    //   if (GGML_NATIVE OR NOT GGML_NATIVE_DEFAULT)
    //       set(INS_ENB OFF)
    //   else()
    //       set(INS_ENB ON)          # ← this path
    //   endif()
    //
    // This causes GGML_AVX, GGML_AVX2, GGML_FMA, GGML_F16C, GGML_BMI2,
    // and GGML_SSE42 to all default to ON.  The resulting library is compiled
    // with -mavx2 -mfma etc. and crashes with SIGILL ("illegal hardware
    // instruction") on any CPU that lacks those extensions — e.g. older
    // Pentium/Celeron, Atom, first-gen Xeon, or VMs that mask AVX.
    //
    // For a distributable Rust crate the default must be portable: only
    // assume SSE4.2 (baseline for all x86_64 CPUs since Nehalem, 2008).
    // Users who want full host-CPU optimisation should use --features native.
    let is_x86_target =
        target.starts_with("x86_64") || target.starts_with("i686") || target.starts_with("i586");
    if is_x86_target && !want_native {
        // SSE4.2 is the effective baseline for x86_64; leave it ON.
        config.define("GGML_SSE42", "ON");
        // Everything above SSE4.2 is not universally available — disable.
        config.define("GGML_AVX", "OFF");
        config.define("GGML_AVX2", "OFF");
        config.define("GGML_AVX_VNNI", "OFF");
        config.define("GGML_FMA", "OFF");
        config.define("GGML_F16C", "OFF");
        config.define("GGML_BMI2", "OFF");
        config.define("GGML_AVX512", "OFF");
        config.define("GGML_AVX512_VBMI", "OFF");
        config.define("GGML_AVX512_VNNI", "OFF");
        config.define("GGML_AVX512_BF16", "OFF");
        config.define("GGML_AMX_TILE", "OFF");
        config.define("GGML_AMX_INT8", "OFF");
        config.define("GGML_AMX_BF16", "OFF");
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

    let mut enable_metal = cfg!(feature = "metal");

    if cfg!(feature = "vulkan") {
        // On macOS, allow `--features vulkan` to gracefully fall back to
        // Metal when Vulkan SDK tooling is not available.
        if target.contains("apple") && !apple_vulkan_available() {
            println!(
                "cargo:warning=Vulkan SDK not found or incomplete on macOS (need VULKAN_SDK, Vulkan headers/library, and glslc). Falling back to Metal backend."
            );
            config.define("GGML_VULKAN", "OFF");
            enable_metal = true;
        } else {
            config.define("GGML_VULKAN", "ON");
        }

        if target.contains("windows") {
            let vulkan_path = find_vulkan_sdk_windows()
                .expect("Could not find Vulkan SDK. Please install it from https://vulkan.lunarg.com/sdk/home and either set the VULKAN_SDK environment variable or install to the default C:\\VulkanSDK\\ location.");
            debug_log!("Vulkan SDK: {}", vulkan_path.display());
            let vulkan_lib_path =
                find_child_dir_ci(&vulkan_path, "lib").unwrap_or_else(|| vulkan_path.join("Lib"));
            println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
            println!("cargo:rustc-link-lib=vulkan-1");
            // Ensure CMake can also find the SDK.
            // Set both the CMake variable and the environment variable –
            // FindVulkan.cmake reads the *environment* variable VULKAN_SDK,
            // not the CMake cache variable.
            config.define("VULKAN_SDK", vulkan_path.to_str().unwrap());
            unsafe { env::set_var("VULKAN_SDK", &vulkan_path) };

            // Explicitly point CMake at the Vulkan components so that
            // FindVulkan.cmake succeeds even when the SDK's Bin directory
            // (containing glslc) is not on PATH.
            //
            // We search for each component with case-insensitive directory
            // matching so the build works regardless of the SDK layout
            // (e.g. "Include" vs "include", "Lib" vs "lib").
            if let Some(inc) = find_child_dir_ci(&vulkan_path, "include") {
                config.define("Vulkan_INCLUDE_DIR", inc.to_str().unwrap());
            }
            if let Some(lib) = find_file_ci(&vulkan_lib_path, "vulkan-1.lib") {
                config.define("Vulkan_LIBRARY", lib.to_str().unwrap());
            }
            if let Some(glslc) = find_glslc(&vulkan_path) {
                config.define("Vulkan_GLSLC_EXECUTABLE", glslc.to_str().unwrap());
            }
        }

        if target.contains("linux") {
            println!("cargo:rustc-link-lib=vulkan");
        }
    }

    if enable_metal {
        config.define("GGML_METAL", "ON");
    } else {
        config.define("GGML_METAL", "OFF");
    }

    if cfg!(feature = "webgpu") {
        config.define("GGML_WEBGPU", "ON");
    } else {
        config.define("GGML_WEBGPU", "OFF");
    }

    if cfg!(feature = "blas") {
        config.define("GGML_BLAS", "ON");
    } else {
        config.define("GGML_BLAS", "OFF");
    }

    if cfg!(feature = "opencl") {
        config.define("GGML_OPENCL", "ON");
    } else {
        config.define("GGML_OPENCL", "OFF");
    }

    if cfg!(feature = "hip") {
        config.define("GGML_HIP", "ON");
    } else {
        config.define("GGML_HIP", "OFF");
    }

    if cfg!(feature = "cuda") && !cfg!(target_os = "macos") {
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
        .out_dir(&cmake_out_dir)
        .profile(&profile)
        .very_verbose(std::env::var("CMAKE_VERBOSE").is_ok()) // Not verbose by default
        .always_configure(false);

    // The cmake crate skips re-configuration when CMakeCache.txt already exists
    // (always_configure = false).  Detect and clear any stale cache that
    // would cause the wrong cmake settings to be used:
    //
    //  1. CMakeCache.txt exists but no Makefile / build.ninja: the previous
    //     configure run was interrupted.  cmake --build would fail with "No
    //     such file or directory".
    //
    //  2. CMakeCache.txt has GGML_NATIVE set to the wrong value: this happens
    //     when build.rs is updated (e.g. the GGML_NATIVE=OFF fix) without
    //     bumping the crate version, so Cargo reuses the same OUT_DIR and the
    //     cmake crate skips configuration entirely — leaving the old ON value
    //     in the cache and baking -mcpu=native into the library, which then
    //     crashes with SIGILL on any chip that lacks the build host's ISA
    //     extensions.
    {
        let cmake_build_dir = cmake_out_dir.join("build");
        let cache = cmake_build_dir.join("CMakeCache.txt");
        if cache.exists() {
            let has_makefile = cmake_build_dir.join("Makefile").exists();
            let has_ninja = cmake_build_dir.join("build.ninja").exists();
            if !has_makefile && !has_ninja {
                debug_log!(
                    "CMakeCache.txt exists but no Makefile/build.ninja found — \
                     removing cache to force reconfiguration"
                );
                std::fs::remove_file(&cache).expect("failed to remove stale CMakeCache.txt");
            } else {
                // Check whether the cached GGML_NATIVE value matches what we
                // are about to configure.  A mismatch means a previous build
                // used a different build.rs (or a different `native` feature
                // state) and the cmake crate's cache-skip path will silently
                // use the wrong value.
                let desired_native_str = if want_native { "ON" } else { "OFF" };
                let cache_contents = std::fs::read_to_string(&cache).unwrap_or_default();

                // 2a. GGML_NATIVE mismatch.
                let cached_native_on = cache_contents.contains("GGML_NATIVE:BOOL=ON");
                let cached_native_off = cache_contents.contains("GGML_NATIVE:BOOL=OFF");
                let native_mismatch =
                    (want_native && cached_native_off) || (!want_native && cached_native_on);

                // 2b. GGML_CPU_ARM_ARCH in cache doesn't match what we intend
                //     to set (ARM non-native non-android builds).  Without an
                //     explicit -march flag, Apple Clang silently enables
                //     dotprod/i8mm for arm64-apple-macosx targets, producing
                //     a binary that crashes with SIGILL on older ARMv8 chips.
                //
                //     The cache entry looks like "GGML_CPU_ARM_ARCH:STRING="
                //     (empty) when cmake used its default.  We want "armv8-a".
                let is_arm_target_local =
                    target.starts_with("aarch64") || target.starts_with("arm");
                let we_set_arm_arch = is_arm_target_local
                    && !want_native
                    && !target.contains("android")
                    && env::var("GGML_CPU_ARM_ARCH").is_err();
                // Extract the cached GGML_CPU_ARM_ARCH value.  The line is
                // "GGML_CPU_ARM_ARCH:STRING=<value>" so we look for that prefix.
                let cached_arm_arch = cache_contents
                    .lines()
                    .find(|l| l.starts_with("GGML_CPU_ARM_ARCH:"))
                    .and_then(|l| l.split_once('=').map(|x| x.1))
                    .unwrap_or("");
                let arm_arch_mismatch = we_set_arm_arch && cached_arm_arch != "armv8-a";

                // 2c. x86 ISA options: when !want_native on x86, we now
                //     force AVX/AVX2/FMA/F16C/BMI2 to OFF.  A stale cache
                //     from a previous build (or from before this fix) may
                //     still have them ON, causing SIGILL on older CPUs.
                let is_x86_target_local = target.starts_with("x86_64")
                    || target.starts_with("i686")
                    || target.starts_with("i586");
                let x86_isa_mismatch = is_x86_target_local && !want_native && {
                    // Check if any of the ISA options we now force OFF are
                    // still cached as ON.
                    let stale_options = [
                        "GGML_AVX:BOOL=ON",
                        "GGML_AVX2:BOOL=ON",
                        "GGML_FMA:BOOL=ON",
                        "GGML_F16C:BOOL=ON",
                        "GGML_BMI2:BOOL=ON",
                    ];
                    stale_options.iter().any(|opt| cache_contents.contains(opt))
                };

                // 2d. GGML_METAL mismatch: on macOS, llama.cpp defaults
                //     GGML_METAL to ON, so a stale cache from a previous
                //     `--features metal` build (or from the cmake default)
                //     would keep Metal enabled even when the feature is off.
                let metal_mismatch = {
                    let cached_metal_on = cache_contents.contains("GGML_METAL:BOOL=ON");
                    let cached_metal_off = cache_contents.contains("GGML_METAL:BOOL=OFF");
                    (enable_metal && cached_metal_off) || (!enable_metal && cached_metal_on)
                };

                // 2e. GGML_WEBGPU mismatch: stale cache may keep WebGPU ON/OFF
                //     after feature toggles.
                let webgpu_mismatch = {
                    let cached_webgpu_on = cache_contents.contains("GGML_WEBGPU:BOOL=ON");
                    let cached_webgpu_off = cache_contents.contains("GGML_WEBGPU:BOOL=OFF");
                    (cfg!(feature = "webgpu") && cached_webgpu_off)
                        || (!cfg!(feature = "webgpu") && cached_webgpu_on)
                };

                // 2f. CMAKE_OSX_ARCHITECTURES mismatch: a stale cache from
                //     a previous build (or from an environment-polluted cmake
                //     run) may have the wrong architecture, producing x86_64
                //     object code in an arm64 binary (or vice versa).
                let osx_arch_mismatch = target.contains("apple") && {
                    let want_arch = if target.contains("aarch64") || target.contains("arm64") {
                        "arm64"
                    } else if target.contains("x86_64") {
                        "x86_64"
                    } else {
                        ""
                    };
                    if !want_arch.is_empty() {
                        cache_contents
                            .lines()
                            .find(|l| l.starts_with("CMAKE_OSX_ARCHITECTURES:"))
                            .and_then(|l| l.split_once('=').map(|x| x.1))
                            .map(|cached| cached != want_arch)
                            .unwrap_or(false)
                    } else {
                        false
                    }
                };

                let mismatch = native_mismatch
                    || arm_arch_mismatch
                    || x86_isa_mismatch
                    || metal_mismatch
                    || webgpu_mismatch
                    || osx_arch_mismatch;
                if mismatch {
                    debug_log!(
                        "CMakeCache.txt is stale (GGML_NATIVE: cache={} want={}; \
                         GGML_CPU_ARM_ARCH: cache={:?} want={}) — removing cache \
                         to force reconfiguration",
                        if cached_native_on {
                            "ON"
                        } else if cached_native_off {
                            "OFF"
                        } else {
                            "?"
                        },
                        desired_native_str,
                        cached_arm_arch,
                        if we_set_arm_arch {
                            "armv8-a"
                        } else {
                            "(not set)"
                        },
                    );
                    std::fs::remove_file(&cache).expect("failed to remove stale CMakeCache.txt");
                }
            }
        }
    }

    let build_dir = config.build();

    // ── Link search paths ────────────────────────────────────────────────────
    println!(
        "cargo:rustc-link-search={}",
        cmake_out_dir.join("lib").display()
    );
    println!(
        "cargo:rustc-link-search={}",
        cmake_out_dir.join("lib64").display()
    );
    println!("cargo:rustc-link-search={}", build_dir.display());

    // ── Link libraries ───────────────────────────────────────────────────────
    let llama_libs_kind = if build_shared_libs { "dylib" } else { "static" };
    let llama_libs = extract_lib_names(&cmake_out_dir, build_shared_libs, &target);
    assert_ne!(llama_libs.len(), 0);

    for lib in llama_libs {
        debug_log!(
            "LINK {}",
            format!("cargo:rustc-link-lib={}={}", llama_libs_kind, lib)
        );
        println!("cargo:rustc-link-lib={llama_libs_kind}={lib}");
    }

    // OpenMP: link gomp when the cmake build enabled it (GGML_OPENMP_ENABLED=ON).
    // This can happen even without the "openmp" feature because cmake's FindOpenMP
    // is invoked unconditionally on some platforms (e.g. ARM) when the
    // ggml-cpu CMakeLists includes OpenMP support at the variant level.
    let cmake_cache_path = cmake_out_dir.join("build").join("CMakeCache.txt");
    let openmp_enabled_in_cmake = std::fs::read_to_string(&cmake_cache_path)
        .map(|contents| contents.contains("GGML_OPENMP_ENABLED:INTERNAL=ON"))
        .unwrap_or(false);

    if (cfg!(feature = "openmp") || openmp_enabled_in_cmake)
        && (target.contains("gnu") || target.contains("musl"))
    {
        println!("cargo:rustc-link-lib=gomp");
    }

    // Removed: Rust already links the appropriate CRT. Explicitly linking
    // msvcrtd causes CRT mismatch when another crate (e.g. whisper-rs) links
    // the release CRT — both msvcrt and msvcrtd get loaded into the same
    // process, corrupting the file descriptor table and causing
    // `_osfile(fh) & FOPEN` assertion failures in read.cpp.

    // macOS frameworks and libc++
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Foundation");
        if enable_metal {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=c++");
    }

    // Linux libstdc++
    if target.contains("linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // Windows MinGW (GCC-based, not MSVC): link the C++ and threading runtimes.
    // MSVC handles its own C++ runtime via the CRT; MinGW needs explicit flags
    // because Rust's linker driver does not add them automatically.
    // `winpthread` provides POSIX threading support on Windows with MinGW.
    if target.contains("windows") && !target.contains("msvc") {
        println!("cargo:rustc-link-lib=static=stdc++");
        println!("cargo:rustc-link-lib=static=winpthread");
    }

    // ggml-cpu.cpp uses Windows Registry APIs (RegOpenKeyExA, RegQueryValueExA,
    // RegCloseKey) to detect CPU features. These live in advapi32.lib which is
    // not linked by default by Rust's linker driver on Windows.
    if target.contains("windows") {
        println!("cargo:rustc-link-lib=advapi32");
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
        let libs_assets = extract_lib_assets(&cmake_out_dir, &target);
        for asset in libs_assets {
            let asset_clone = asset.clone();
            let filename = asset_clone.file_name().unwrap();
            let filename = filename.to_str().unwrap();

            // Helper: remove stale destination before hard-linking.
            // On Linux, cmake creates versioned symlink chains (e.g. libllama.so -> libllama.so.0).
            // When the library version changes between builds, these become broken symlinks.
            // Path::exists() returns false for broken symlinks, but hard_link() still fails
            // with EEXIST because the directory entry is occupied. Using symlink_metadata()
            // detects both regular files and symlinks (broken or valid).
            let force_hard_link = |src: &Path, dst: &Path| {
                if dst.symlink_metadata().is_ok() {
                    let _ = std::fs::remove_file(dst);
                }
                std::fs::hard_link(src, dst).unwrap();
            };

            let dst = target_dir.join(filename);
            debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
            force_hard_link(&asset, &dst);

            // Copy DLLs to examples as well
            if target_dir.join("examples").exists() {
                let dst = target_dir.join("examples").join(filename);
                debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
                force_hard_link(&asset, &dst);
            }

            // Copy DLLs to target/profile/deps as well for tests
            let dst = target_dir.join("deps").join(filename);
            debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
            force_hard_link(&asset, &dst);
        }
    }
}
