# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.20] - 2026-04-01

### Added

- **`llama_cpp_4::quantize` module** — fully typed Rust API for model quantization:
  - `LlamaFtype` enum covering all 34 quantization types (`Q4_K_M`, `Q5_K_M`, `IQ2_XXS`, …)
    with `name()`, `description()`, `from_name()`, and `all()` helpers.
  - `GgmlType` enum for raw tensor storage types with `From`/`TryFrom` conversions.
  - `QuantizeParams` builder: `nthread`, `ftype`, `output_tensor_type`,
    `token_embedding_type`, `allow_requantize`, `quantize_output_tensor`,
    `only_copy`, `pure`, `keep_split`, `dry_run`.
  - `Imatrix` / `ImatrixEntry` for per-tensor importance-matrix data.
  - `TensorTypeOverride` for per-glob-pattern tensor type overrides (`output=F16`).
  - `KvOverride` / `KvOverrideValue` for GGUF metadata injection.
  - `with_pruned_layers()` to remove layers from the output model.
  - `set_attn_rot_disabled()` / `attn_rot_disabled()` for process-level
    TurboQuant control.
- **`LlamaContextParams::with_cache_type_k(GgmlType)`** and
  **`with_cache_type_v(GgmlType)`** — typed setters for KV cache storage type.
- **`LlamaContextParams::with_attn_rot_disabled(bool)`** — per-context
  TurboQuant (Hadamard attention rotation) toggle; applied safely around
  `llama_new_context_with_model` without leaking to other threads.
- **`examples/turbo-quant`** — new example demonstrating TurboQuant:
  `--show-api` prints the full API reference and PPL quality table;
  `--model`/`--kv-type`/`--n-predict` runs side-by-side rotation on/off
  inference so the quality difference is directly visible.
- **`examples/quantize`** rewritten with the new typed API: `--tensor-type`,
  `--prune-layer`, `--disable-attn-rot`, `--list-types` flags.

### Changed

- **`model_quantize()`** now takes `&QuantizeParams` and returns
  `Result<(), u32>` instead of the raw sys struct.
- **`llama-cpp-sys-4` bumped to 0.2.19** — submodule updated to
  llama.cpp `c30e01225` (April 2026), including:
  - [PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038)
    **TurboQuant** — Hadamard rotation of Q/K/V into KV cache for
    dramatically better quantized-cache quality (Q5_0: +0.55 PPL vs +17.28
    PPL without rotation; 2.91× smaller KV cache than F16).
  - [PR #20346](https://github.com/ggml-org/llama.cpp/pull/20346)
    pure-C `llama_model_quantize_params` interface — `void *` fields
    replaced with typed struct pointers (`llama_model_imatrix_data`,
    `llama_model_tensor_override`, `llama_model_kv_override`);
    `tensor_types` field renamed to `tt_overrides`.

### Fixed

- **Windows CI** — old `quantize` example assigned a `u32` ftype literal
  directly to the `i32` `llama_ftype` field that MSVC bindgen emits,
  causing a type error on Windows only. The new `LlamaFtype` enum with
  `as`-casts fixes this on all platforms.

### Deprecated

- `model_quantize_default_params()` — use `QuantizeParams::new(ftype)` instead.

## [0.2.16] - 2026-03-30

### Fixed

- **Windows build**: link `advapi32.lib` to resolve `RegOpenKeyExA`,
  `RegQueryValueExA`, and `RegCloseKey` symbols used by the updated
  `ggml-cpu.cpp` for CPU feature detection via the Windows Registry.
- **Updated llama.cpp submodule** to latest upstream.

## [0.2.14] - 2026-03-29

### Changed

- **Updated all dependencies** to latest compatible versions (128 packages).
- **Updated llama.cpp submodule** from b8533 to b8575.
- **Fixed all clippy warnings** across the entire workspace (~190 → 0).
- **Fixed all `cargo doc` warnings** (8 → 0): broken intra-doc links, bare URLs,
  malformed code block.
- **Improved docs.rs metadata**: replaced non-existent `sampler` feature with
  `mtmd`, added `keywords` and `categories` to both crates.
- **Expanded crate-level documentation**: added feature flag docs for `metal`,
  `vulkan`, `native`, `openmp`; added links to all examples.
- **Added `# Errors` and `# Panics` doc sections** to public API functions.
- **Added `Default` impl** for `LlamaSampler`.
- **Changed `apply_chat_template` signature** to accept `Option<&str>` and
  `&[LlamaChatMessage]` instead of owned types.
- **Removed `rpc-example`** from README examples table (not in workspace).
- **Formatted all code** with `cargo fmt`.

## [0.2.13] - 2026-03-21

### Changed

- **Updated llama.cpp submodule** to latest upstream.
- **Renamed `audio_sample_rate()` → `audio_bitrate()`** on `MtmdContext` to
  follow the upstream rename of `mtmd_get_audio_sample_rate` →
  `mtmd_get_audio_bitrate`. The old `audio_sample_rate()` method is kept as a
  deprecated alias.

## [0.2.9] - 2026-03-12

### Added

- **Case-insensitive Vulkan SDK directory lookup on Windows** — the build
  script now uses case-insensitive matching when locating `Lib`, `Include`,
  and `Bin` directories inside the Vulkan SDK, accommodating SDK layouts
  that use varying capitalisation (e.g. `Lib` vs `lib`).
- **Explicit CMake Vulkan component hints** — `Vulkan_INCLUDE_DIR`,
  `Vulkan_LIBRARY`, and `Vulkan_GLSLC_EXECUTABLE` are now passed directly
  to CMake so `FindVulkan.cmake` succeeds even when the SDK's `Bin`
  directory is not on `PATH`.
- **Recursive `glslc` search** — `find_glslc()` searches up to 3 directory
  levels inside the SDK, then falls back to `where`/`which` on `PATH`,
  covering non-standard SDK layouts.
- **`VULKAN_SDK` environment variable forwarding** — the discovered SDK path
  is now also set as a process environment variable (in addition to the CMake
  define) so that `FindVulkan.cmake` picks it up correctly.

### Fixed

- **`simple` example sampler** — removed the `LlamaSampler::dist(seed)`
  call that was stacked before `LlamaSampler::greedy()`, which had no
  effect on greedy decoding and caused an unused-variable warning for
  `seed`.

### Changed

- All crate versions bumped to `0.2.9`.
- **README examples table** — added a "Package name" column and the `mtmd`
  multimodal example; added Vulkan GPU acceleration example; improved
  multimodal section formatting with sub-headings and the standalone `mtmd`
  CLI example.

## [0.2.8] - 2026-03-12

### Added

- **Automatic Vulkan SDK detection on Windows** — the build script now
  automatically locates the Vulkan SDK by checking (in order):
  1. The `VULKAN_SDK` environment variable.
  2. The Windows registry (`HKLM\SOFTWARE\LunarG\VulkanSDK`).
  3. The default install directory `C:\VulkanSDK\<latest version>`.

  Previously a missing `VULKAN_SDK` env var caused an immediate panic; now
  the SDK is found automatically when installed to the default location.
  Added `winreg` as a Windows-only build dependency for registry access.
- The discovered Vulkan SDK path is now forwarded to CMake via
  `VULKAN_SDK` define, ensuring CMake's `FindVulkan` also picks it up.

### Fixed

- **README `cargo run` command** — the interactive chat example referenced
  `-p llama-cpp-chat` which does not exist; corrected to `-p chat` matching
  the actual package name in `examples/chat/Cargo.toml`.

### Changed

- All crate versions bumped to `0.2.8`.
- Updated `WINDOWS.md` to document automatic Vulkan SDK detection and
  demote the manual `VULKAN_SDK` env var setup to a fallback.

## [0.2.7] - 2026-03-11

### Added

- **macOS Vulkan graceful fallback** — `--features vulkan` on macOS now
  auto-detects whether the Vulkan SDK is installed (checks `VULKAN_SDK`,
  headers, library, and `glslc`). When the SDK is missing or incomplete,
  the build automatically falls back to the Metal backend instead of
  failing.
- **Platform setup guides** — added `LINUX.md`, `MAC.md`, and `WINDOWS.md`
  with step-by-step build instructions and Vulkan prerequisites for each OS.
- CI workflow updated with a Windows Vulkan build check.

### Changed

- `llama-cpp-sys-4` and `llama-cpp-4` bumped to `0.2.7`.

## [0.2.6] - 2026-03-10

### Added

- **Windows `MAX_PATH` workaround** — on Windows, the cmake build/install
  tree is redirected to a short path under `%LOCALAPPDATA%` (derived from a
  stable hash of `OUT_DIR`) to avoid exceeding the 260-character `MAX_PATH`
  limit.  The Vulkan backend's deeply-nested `ExternalProject` sub-build was
  pushing total path lengths beyond 260 chars, causing MSBuild error MSB3491.

### Changed

- `llama-cpp-sys-4` and `llama-cpp-4` bumped to `0.2.6`.
- All cmake output paths (`lib`, `lib64`, `build`, shared-library assets)
  updated to use the shortened `cmake_out_dir` on Windows.

## [0.2.5] - 2026-03-10

### Fixed

- **Windows MinGW cross-compilation (critical)** — `extract_lib_names` in
  `build.rs` used `*.lib` for every Windows target, but MinGW/GCC toolchains
  (`windows-gnu`, `windows-gnullvm`) produce `.a` static archives, not `.lib`
  files.  Cross-compiling to any `*-windows-gnu` target would find zero
  libraries and immediately panic on `assert_ne!(llama_libs.len(), 0)`.  Fixed
  by splitting on `windows-msvc` (→ `*.lib`) vs. all other Windows targets
  (→ `*.a`).
- **MinGW import-library name extraction** — MinGW shared import libraries are
  named `libfoo.dll.a`.  `file_stem()` strips the final `.a`, leaving
  `libfoo.dll`; stripping the `lib` prefix then produced the incorrect link
  name `foo.dll`.  Added a second strip of the trailing `.dll` suffix for
  Windows non-MSVC targets so the correct name `foo` is emitted to
  `cargo:rustc-link-lib`.
- **Missing C++ and threading runtime linkage for Windows MinGW** — Linux
  builds emit `cargo:rustc-link-lib=dylib=stdc++` and macOS emits
  `cargo:rustc-link-lib=c++`, but Windows MinGW targets had no equivalent.
  Added `cargo:rustc-link-lib=static=stdc++` and
  `cargo:rustc-link-lib=static=winpthread` for all `windows` non-`msvc`
  targets.  Static linkage avoids a runtime dependency on MinGW DLLs.
- **`copy_folder` double-evaluation guard** — the two consecutive
  `if cfg!(unix)` / `if cfg!(windows)` blocks could theoretically both
  evaluate on a host where neither macro resolves.  Changed to `if / else if`
  to make the mutual exclusion explicit.
- **Stale `rerun-if-changed=./sherpa-onnx`** — this watch directive pointed
  at a path that does not exist in this repository (leftover from a different
  project).  Cargo re-evaluates `build.rs` on every run when a watched path
  is missing, slowing incremental builds.  Removed.
- **Invalid `target_arch = "arm64"` in `llama-cpp-4/Cargo.toml`** — Rust's
  `cfg` predicate system has no `target_arch = "arm64"`; the correct
  identifier for 64-bit ARM (including Apple Silicon) is `aarch64`.  The
  stale `any(target_arch = "aarch64", target_arch = "arm64")` condition was
  silently identical to `target_arch = "aarch64"` alone.  Cleaned up to a
  single, correctly documented entry.

### Changed

- All example crate versions synced to `0.2.5` (were stale at `0.2.1`).

## [0.2.3] - 2026-03-10

### Added
- **Cross-compilation support** in `llama-cpp-sys-4/build.rs`:
  - All target-conditional logic now uses the `TARGET` environment variable
    instead of `cfg!(target_os/arch/...)` macros, which evaluate against the
    *host* at compile time and break cross builds.
  - `HOST` and `is_cross` (`HOST != TARGET`) are computed once and threaded
    through all platform-specific branches.
  - **`CMAKE_CROSSCOMPILING=TRUE`** is explicitly passed to CMake when
    cross-compiling. CMake only sets this flag automatically when
    `CMAKE_SYSTEM_NAME` changes, so same-OS cross-arch builds
    (e.g. `x86_64-linux` → `aarch64-linux`) would previously leave it unset,
    causing ggml to default `GGML_NATIVE` to `ON` and bake host CPU
    instruction sets into the target binary (SIGILL).
  - **`GGML_NATIVE=OFF`** is explicitly forced for all cross builds, ensuring
    ggml never adds `-march=native` or runs `check_cxx_source_runs` CPU-probing
    tests (which execute on the build host, not the target) regardless of any
    stale `CMakeCache.txt` values.
  - **Apple cross-arch** (`x86_64-apple-darwin` → `aarch64-apple-darwin` and
    vice versa): uses `CMAKE_OSX_ARCHITECTURES` (e.g. `arm64`) instead of
    attempting to find a non-existent `aarch64-apple-darwin-gcc` binary.
    Apple's Clang is already a universal cross-compiler; `-arch` is the correct
    mechanism.
  - **MinGW cross-compilation** (`windows-gnu` / `windows-gnullvm` targets):
    the C/C++ compiler is derived using the MinGW triple convention
    (`x86_64-w64-mingw32-gcc`), not the Rust target triple
    (`x86_64-pc-windows-gnu-gcc` which does not exist).
  - `CMAKE_SYSTEM_NAME` and `CMAKE_SYSTEM_PROCESSOR` are set from the Rust
    target triple via new `cmake_system_name()` and `cmake_system_processor()`
    helpers.
  - `CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER` honour `CC` / `CXX` environment
    variables first (compatible with `cargo cross`, `zig cc`, osxcross, etc.),
    then fall back to sensible defaults per platform.
  - `CMAKE_SYSROOT` / `CMAKE_OSX_SYSROOT` are forwarded from environment
    variables when provided.
  - bindgen gains a `--target {TARGET}` clang argument so that struct layouts,
    pointer widths, and type sizes are computed for the *target* ABI, not the
    host.
  - `extract_lib_names` and `extract_lib_assets` now accept a `target: &str`
    parameter and select file extensions (`.a` / `.so` / `.dylib` / `.lib` /
    `.dll`) based on the target triple rather than the host.
  - `macos_link_search_path` accepts the clang binary name as a parameter;
    for same-SDK Apple cross-arch builds it uses the system `clang` (correct),
    and for osxcross it honours `CC`.

### Fixed
- **SIGILL on cross-compiled binaries**: caused by `GGML_NATIVE=ON` baking
  host CPU features (AVX2, SVE, i8mm, …) into a binary destined for a
  different microarchitecture. Fixed by forcing `CMAKE_CROSSCOMPILING=TRUE`
  and `GGML_NATIVE=OFF` for all cross builds.
- `config.static_crt()` (MSVC `/MT` vs `/MD`) is now only applied to
  `windows-msvc` targets; calling it for `windows-gnu` (MinGW) builds was
  incorrect.
- `cargo:rustc-link-lib=dylib=msvcrtd` (MSVC debug CRT) is now only emitted
  for `windows-msvc` targets; MinGW toolchains use their own CRT and do not
  have `msvcrtd`.
- MPI `CC`/`CXX` override now checks `target.contains("apple")` instead of
  `cfg!(target_os = "macos")` so it does not fire when cross-compiling *from*
  macOS to a non-Apple target.

### Changed
- The `native` Cargo feature is now wired up to `GGML_NATIVE=ON` in CMake.
  Previously the feature existed in `Cargo.toml` but had no effect in
  `build.rs`. Non-cross builds default to `GGML_NATIVE=OFF` (portable); pass
  `--features native` to opt into host-CPU optimisation.

## [0.2.2] - 2026-03-09

### Changed
- Source version tracking: re-copy llama.cpp sources when the submodule HEAD
  changes, not just when `OUT_DIR` is fresh, preventing stale builds after a
  submodule update.
- CMake stale-cache detection: remove `CMakeCache.txt` when it exists without
  a corresponding `Makefile` or `build.ninja`, forcing a clean reconfiguration.

## [0.2.1] - 2026-03-09

### Added
- `mtmd` feature: multimodal (image + text) support via the `tools/mtmd`
  library.
- Vendor directory and additional tool subdirectories included in the crate
  package so `cmake` can find all required source files.

## [0.2.0] - 2026-03-09

### Added
- Initial public release of `llama-cpp-4` and `llama-cpp-sys-4`.
- Raw bindgen bindings to llama.cpp and ggml.
- High-level safe Rust API: model loading, context management, sampling,
  tokenisation, embeddings, LoRA adapters.
- Features: `cuda`, `metal`, `vulkan`, `openmp`, `rpc`, `mpi`, `dynamic-link`.
- Examples: `simple`, `chat`, `embeddings`, `split_model`, `server`, `rpc`.

[0.2.14]: https://github.com/eugenehp/llama-cpp-rs/compare/v0.2.13...v0.2.14
[0.2.13]: https://github.com/eugenehp/llama-cpp-rs/compare/v0.2.9...v0.2.13
[0.2.9]: https://github.com/eugenehp/llama-cpp-rs/compare/v0.2.8...v0.2.9
[0.2.8]: https://github.com/eugenehp/llama-cpp-rs/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/eugenehp/llama-cpp-rs/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/eugenehp/llama-cpp-rs/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/eugenehp/llama-cpp-rs/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/eugenehp/llama-cpp-rs/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/eugenehp/llama-cpp-rs/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/eugenehp/llama-cpp-rs/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/eugenehp/llama-cpp-rs/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/eugenehp/llama-cpp-rs/releases/tag/v0.2.0
