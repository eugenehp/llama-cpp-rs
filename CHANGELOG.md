# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.3]: https://github.com/eugenehp/llama-cpp-rs/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/eugenehp/llama-cpp-rs/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/eugenehp/llama-cpp-rs/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/eugenehp/llama-cpp-rs/releases/tag/v0.2.0
