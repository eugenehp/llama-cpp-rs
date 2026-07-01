//! Download and cache prebuilt llama.cpp libraries from GitHub releases.
//!
//! Used when the `prebuilt` Cargo feature is enabled. Artifacts are produced by
//! `.github/workflows/prebuilt-llama.yml` and named:
//! `llama-prebuilt-{linux|macos|windows}-{target}-{variant}-{static|dynamic}.tar.gz`

use std::env;
use std::fs::{self, File};
use std::io::{copy, BufReader};
use std::path::{Path, PathBuf};

const DEFAULT_REPO: &str = "eugenehp/llama-cpp-rs";

/// Resolve the prebuilt tarball file name for the current build configuration.
pub fn asset_name(target: &str, use_shared_libs: bool) -> Option<String> {
    let os = platform_os(target)?;
    let suffix = variant_suffix()?;
    let library_type = if use_shared_libs { "dynamic" } else { "static" };
    Some(format!(
        "llama-prebuilt-{os}-{target}-{suffix}-{library_type}.tar.gz"
    ))
}

/// Ensure prebuilt libraries are available; download and extract when missing.
///
/// Returns `None` when auto-download is disabled, the platform/variant is
/// unsupported, or no matching release asset exists (caller should compile locally).
pub fn ensure_prebuilt(target: &str, use_shared_libs: bool) -> Option<PathBuf> {
    if is_disabled() {
        return None;
    }

    if env::var("LLAMA_PREBUILT_DIR").is_ok() {
        return None;
    }

    let asset = asset_name(target, use_shared_libs)?;
    let tag = release_tag();
    let cache_root = cache_root()?;
    let extract_dir = cache_root
        .join(tag.trim_start_matches('v'))
        .join(asset.strip_suffix(".tar.gz").unwrap_or(&asset));

    if is_valid_prebuilt_root(&extract_dir) {
        println!(
            "cargo:warning=Using cached prebuilt llama libs from {}",
            extract_dir.display()
        );
        return Some(extract_dir);
    }

    let url = download_url(&tag, &asset);
    println!("cargo:warning=Downloading prebuilt llama libs: {url}");

    match download_and_extract(&url, &extract_dir) {
        Ok(()) if is_valid_prebuilt_root(&extract_dir) => {
            println!(
                "cargo:warning=Prebuilt llama libs ready at {}",
                extract_dir.display()
            );
            Some(extract_dir)
        }
        Ok(()) => {
            println!(
                "cargo:warning=Prebuilt archive extracted but no libraries found at {}; falling back to local compile",
                extract_dir.display()
            );
            let _ = fs::remove_dir_all(&extract_dir);
            None
        }
        Err(err) => {
            println!(
                "cargo:warning=Prebuilt download failed ({err}); falling back to local compile"
            );
            let _ = fs::remove_dir_all(&extract_dir);
            None
        }
    }
}

fn is_disabled() -> bool {
    matches!(
        env::var("LLAMA_PREBUILT_OFF").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("on") | Ok("ON")
    )
}

fn release_tag() -> String {
    env::var("LLAMA_PREBUILT_TAG").unwrap_or_else(|_| {
        format!(
            "v{}",
            env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.4.0".into())
        )
    })
}

fn github_repo() -> String {
    env::var("LLAMA_PREBUILT_REPO").unwrap_or_else(|_| DEFAULT_REPO.to_string())
}

fn download_url(tag: &str, asset: &str) -> String {
    if let Ok(url) = env::var("LLAMA_PREBUILT_URL") {
        return url;
    }
    format!(
        "https://github.com/{}/releases/download/{}/{}",
        github_repo(),
        tag,
        asset
    )
}

fn cache_root() -> Option<PathBuf> {
    let out_dir = env::var("OUT_DIR").ok()?;
    let profile = env::var("PROFILE").ok()?;
    let mut target_dir = None;
    let mut sub_path = Path::new(&out_dir);
    while let Some(parent) = sub_path.parent() {
        if parent.ends_with(&profile) {
            target_dir = Some(parent);
            break;
        }
        sub_path = parent;
    }
    Some(target_dir?.join("llama-prebuilt-cache"))
}

fn platform_os(target: &str) -> Option<&'static str> {
    if target.contains("linux") {
        Some("linux")
    } else if target.contains("windows") {
        Some("windows")
    } else if target.contains("apple") {
        Some("macos")
    } else {
        None
    }
}

/// Map enabled backend features to the CI variant suffix.
fn variant_suffix() -> Option<String> {
    if cfg!(feature = "cuda")
        || cfg!(feature = "hip")
        || cfg!(feature = "webgpu")
        || cfg!(feature = "opencl")
        || cfg!(feature = "q1")
    {
        // No published artifacts for these variants yet.
        return None;
    }

    if cfg!(feature = "metal") {
        return Some("metal".to_string());
    }
    if cfg!(feature = "vulkan") {
        return Some("vulkan".to_string());
    }
    if cfg!(feature = "blas") {
        return Some("blas".to_string());
    }
    Some("cpu".to_string())
}

fn is_valid_prebuilt_root(root: &Path) -> bool {
    if !root.is_dir() {
        return false;
    }
    for dir in [
        root.to_path_buf(),
        root.join("lib"),
        root.join("lib64"),
        root.join("bin"),
    ] {
        if !dir.is_dir() {
            continue;
        }
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if !file_type.is_file() && !file_type.is_symlink() {
                continue;
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if is_llama_lib_name(name) {
                return true;
            }
        }
    }
    false
}

fn is_llama_lib_name(name: &str) -> bool {
    let base = name
        .strip_prefix("lib")
        .unwrap_or(name)
        .split('.')
        .next()
        .unwrap_or(name);
    matches!(
        base,
        "llama" | "ggml" | "ggml-base" | "ggml-cpu" | "common" | "mtmd"
    ) || base.starts_with("ggml-")
}

fn download_and_extract(url: &str, extract_dir: &Path) -> Result<(), String> {
    if extract_dir.exists() {
        fs::remove_dir_all(extract_dir).map_err(|e| e.to_string())?;
    }
    fs::create_dir_all(extract_dir).map_err(|e| e.to_string())?;

    let archive_path = extract_dir.with_extension("tar.gz");
    download_file(url, &archive_path)?;

    let file = File::open(&archive_path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    let decoder = flate2::read::GzDecoder::new(reader);
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(extract_dir).map_err(|e| e.to_string())?;
    let _ = fs::remove_file(&archive_path);
    Ok(())
}

fn download_file(url: &str, dest: &Path) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let partial = dest.with_extension("partial");
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("HTTP GET {url}: {e}"))?;
    if !(200..300).contains(&response.status()) {
        return Err(format!("HTTP {} for {url}", response.status()));
    }
    let mut reader = response.into_reader();
    let mut file = File::create(&partial).map_err(|e| e.to_string())?;
    copy(&mut reader, &mut file).map_err(|e| e.to_string())?;
    fs::rename(&partial, dest).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(feature = "prebuilt")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linux_cpu_static_name() {
        let name = asset_name_for("x86_64-unknown-linux-gnu", false, "cpu");
        assert_eq!(
            name,
            Some("llama-prebuilt-linux-x86_64-unknown-linux-gnu-cpu-static.tar.gz".into())
        );
    }

    #[test]
    fn macos_metal_dynamic_name() {
        let name = asset_name_for("aarch64-apple-darwin", true, "metal");
        assert_eq!(
            name,
            Some("llama-prebuilt-macos-aarch64-apple-darwin-metal-dynamic.tar.gz".into())
        );
    }

    fn asset_name_for(target: &str, shared: bool, suffix: &str) -> Option<String> {
        let os = platform_os(target)?;
        let library_type = if shared { "dynamic" } else { "static" };
        Some(format!(
            "llama-prebuilt-{os}-{target}-{suffix}-{library_type}.tar.gz"
        ))
    }

    #[test]
    fn llama_lib_name_detection() {
        assert!(is_llama_lib_name("libllama.a"));
        assert!(is_llama_lib_name("libggml-cpu.so.0.0.1"));
        assert!(is_llama_lib_name("llama.lib"));
        assert!(!is_llama_lib_name("libssl.a"));
    }
}
