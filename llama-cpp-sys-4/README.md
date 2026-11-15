# llama-cpp-sys-4

[![Crates.io](https://img.shields.io/crates/v/llama-cpp-sys-4.svg)](https://crates.io/crates/llama-cpp-sys-4)
[![License](https://img.shields.io/crates/l/llama-cpp-sys-4.svg)](https://crates.io/crates/llama-cpp-sys-4)

Raw `bindgen`-generated bindings to [llama.cpp](https://github.com/ggml-org/llama.cpp),
plus the C/C++ build logic that compiles the library.

**llama.cpp version:** `1c2adc75f6 (b12388, incl. v1.0.0)` · **Crate version:** 0.12.0

Unless you need access to a symbol not yet exposed by [`llama-cpp-4`](../llama-cpp-4/),
use that crate instead — it provides a safe API over these raw bindings.

For application code, start with:

```rust
use llama_cpp_4::prelude::*;
```

---

## What's included

- `llama_*` functions and types from `llama.h`
- `ggml_*` functions and types from `ggml/include/ggml.h`
- `LLAMA_*` constants
- `common_tokenize` and `common_token_to_piece` from `common/common.h`
- The entire llama.cpp static library (or shared, with `dynamic-link`)

---

## Feature flags

| Feature | Description |
|---|---|
| `openmp` | OpenMP multi-threading (default on; auto-detected on ARM platforms) |
| `cuda` | NVIDIA GPU (requires CUDA toolkit) |
| `metal` | Apple GPU (macOS/iOS only) |
| `vulkan` | Vulkan GPU backend |
| `native` | `-march=native` — tune for the build machine's CPU |
| `rpc` | Remote compute backend |
| `dynamic-link` | Link against a pre-installed shared `libllama` instead of building from source |
| `prebuilt` | Request a compatible precompiled build when one can be verified (see below) |

---

## DFlash2

DFlash2 drafting (grouped dynamic depthwise convolution + candidate selector)
merged upstream in [#27342](https://github.com/ggml-org/llama.cpp/pull/27342)
(`b10658`) and ships in the vendored `b11829`, so it needs no build feature — the `dflash2`
feature and `patches/0006-dflash2.patch` that carried the pre-merge code are
gone. Converting a checkpoint uses upstream's own `convert_hf_to_gguf.py`.

DFlash2 checkpoints are detected from GGUF metadata, so no separate speculative
type is needed — select a DFlash draft via `Eagle3Session::new_dflash` in
`llama-cpp-4`.

---

## Prebuilt libraries

The exact speculative-state, decode-lifecycle, and fail-closed EAGLE-3 process
patches require native libraries built from the same patched source. No
published release archive carries a patch-identity envelope yet, so `prebuilt`
currently emits a warning and uses the verified source build.
`LLAMA_PREBUILT_DIR` fails closed while those patches are active rather than
mixing patched headers with unverifiable native objects.

Once release assets carry an exact patch identity, the intended interface is:

```bash
# Automatic download + cache, when a compatible release asset is available
cargo build -p llama-cpp-sys-4 --features prebuilt

# Prefetch manually, then build
./scripts/fetch-prebuilt.sh
export LLAMA_PREBUILT_DIR=target/llama-prebuilt-cache/0.12.0/llama-prebuilt-...
cargo build -p llama-cpp-sys-4
```

Release assets are named `llama-prebuilt-{linux|macos|windows}-{target}-{cpu|vulkan|blas|metal}-{static|dynamic}.tar.gz`
and published by `.github/workflows/prebuilt-llama.yml` on version tags.

| Variable | Purpose |
|---|---|
| `LLAMA_PREBUILT_DIR` | Local directory with `lib/` (or `lib64/`, `bin/`) |
| `LLAMA_PREBUILT_TAG` | Release tag (default: `v{CARGO_PKG_VERSION}`) |
| `LLAMA_PREBUILT_OFF` | `1` disables auto-download |
| `LLAMA_PREBUILT_SHARED` | Force dynamic linking when using `LLAMA_PREBUILT_DIR` |

`cuda`, `hip`, `webgpu`, `opencl`, and `q1` have no published prebuilts yet — those builds always compile locally.

---

## Building

The crate compiles llama.cpp from the vendored submodule at build time using
`cc` + `cmake`-style flags. No external llama.cpp installation is required.

```bash
# CPU only (default)
cargo build -p llama-cpp-sys-4

# Metal (macOS)
cargo build -p llama-cpp-sys-4 --features metal

# CUDA
cargo build -p llama-cpp-sys-4 --features cuda

# OpenMPI (distributed inference)
brew install open-mpi   # or apt install libopenmpi-dev
cargo build -p llama-cpp-sys-4 --features mpi
```

### Configuring llama.cpp's own CMake options

llama.cpp has dozens of `LLAMA_*` CMake options and gains more every release.
Rather than mirror each one as a Cargo feature — which does not scale, and
could not carry a value anyway — any `LLAMA_*` option can be set as data, from
three places. Lowest precedence first:

**1. `llama-cpp.toml`**, found by walking up from this crate or from `OUT_DIR`,
or named outright by `LLAMA_CONFIG`. Good for a machine-local or gitignored
override:

```toml
[cmake]
LLAMA_OPENSSL = false      # or "OFF"; 0/1, on/off, true/false, yes/no all work
LLAMA_CURL = "OFF"

[cmake.target."aarch64-apple-darwin"]
LLAMA_CURL = "ON"          # per-target overrides win over the shared table
```

**2. `[package.metadata.llama-cpp-sys-4.cmake]`** in the consuming project's
manifest (or `[workspace.metadata.…]`, since a workspace root is often a
virtual manifest). Good for a setting that belongs to the project and should be
checked in:

```toml
[package.metadata.llama-cpp-sys-4.cmake]
LLAMA_OPENSSL = "OFF"

[package.metadata.llama-cpp-sys-4.cmake.target."x86_64-unknown-linux-gnu"]
LLAMA_SANITIZE_ADDRESS = "ON"
```

**3. A `LLAMA_*` environment variable**, which always wins, so CI and one-off
builds can override anything:

```bash
LLAMA_OPENSSL=0 cargo build -p llama-cpp-sys-4
```

Notes:

- Only `LLAMA_*` is configurable. The backend options (`GGML_METAL`,
  `GGML_CUDA`, …) are driven by this crate's Cargo features, which also decide
  what the Rust side links against; enabling a backend behind the feature
  system's back would configure a library the link step does not expect. A
  `GGML_*` key in a config file is reported and skipped.
- The build script's own knobs (`LLAMA_PATCH`, `LLAMA_PREBUILT_*`,
  `LLAMA_LIB_PROFILE`, `LLAMA_STATIC_CRT`, `LLAMA_NO_SCCACHE`,
  `LLAMA_BUILD_SHARED_LIBS`, `LLAMA_TEST_MODEL`, …) are not forwarded.
- `LLAMA_OPENSSL=0` is the motivating case: upstream defaults it `ON`, so any
  OpenSSL CMake can find gets linked into `common` — a second TLS stack that
  applications downloading models themselves never use, but must still ship and
  patch. With it off, an `https://` request through `common/http.h` fails with
  an explicit "HTTPS is not supported" error rather than falling back to plain
  HTTP.
- Config files are tracked with `rerun-if-changed`, and every `LLAMA_*` variable
  that is *set* gets a `rerun-if-env-changed`. Exporting a `LLAMA_*` variable
  for the *first* time cannot be detected — Cargo has to be told the name in
  advance — so run `cargo clean -p llama-cpp-sys-4` after that, or configure it
  in a file instead.

### Build dependencies

- `clang` — required by `bindgen` to parse the C++ headers
- A C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- `cmake` and a supported CMake generator (Ninja is preferred)

### Regenerating bindings

Bindings are regenerated automatically whenever `build.rs` or `wrapper.h`
changes. The allowlist covers `llama_*`, `ggml_*`, `LLAMA_*`, and the two
`common_*` functions.

```bash
# Force a full rebuild including binding regeneration
touch llama-cpp-sys-4/wrapper.h
cargo build -p llama-cpp-sys-4
```

---

## Notable upstream API changes handled here

Breaks this crate absorbed as the vendored llama.cpp moved forward. The
`CHANGELOG.md` entry for each release covers that release's window; this is the
running summary.

| Removed / renamed | Replacement | Since |
|---|---|---|
| `llama_kv_cache_*` functions | `llama_memory_*` via `llama_get_memory(ctx)` | b4689 → b8249 |
| `llama_set_adapter_lora` + `llama_rm_adapter_lora` | `llama_set_adapters_lora` (batch API) | b4689 → b8249 |
| `context_params.flash_attn: bool` | `context_params.flash_attn_type: llama_flash_attn_type` | b4689 → b8249 |
| `llama-sampling.h` | `llama-sampler.h` | b4689 → b8249 |
| C++11 build flag | C++17, required by `common.h`'s `std::string_view` | b4689 → b8249 |
| `common_peg_arena::save` / `load` | the arena stays in C++ behind a handle (`chat_shim_parser`) | b11557 |
| `common_chat_parse(const std::string &)` | `common_chat_parse(const common_chat_input &)` | b11557 |
| `common_speculative_process(const llama_batch &)` | `common_speculative_process(const common_batch &)` | b11557 |
| `common_speculative_draft_params::n_past` | `::pos0` | b11557 |
| `common_log_set_jsonl(log, bool)` | `common_log_set_jsonl(bool)` — now a global | b11557 |
| `llama_copy_state_data` / `llama_set_state_data` | `llama_state_get_data` / `llama_state_set_data` — take a size, so the safe wrappers are no longer `unsafe` | b12388 |
| `llama_load_model_from_file` / `llama_new_context_with_model` | `llama_model_load_from_file` / `llama_init_from_model` | b12388 |
| `llama_set_warmup` | none — do warmup runs without a flag | b12388 |
| `llama_vocab_cls` | `llama_vocab_bos` — the same token wherever either was defined | b12388 |
| `common_fit_params(const char *)` | `common_fit_params(const std::vector<std::string> &, …, const common_fit_extra_model *)` — shimmed as `common_fit_params_one`, since bindgen renders a by-reference `std::vector` as an opaque byte array | b12388 |
| `common_get_device_memory_data` (8 args) | 9 args: paths became a `std::vector<std::string>`, plus an `hp_ngl_skip` out-param | b12388 |

---

## Bindgen configuration

Key decisions in `build.rs`:

- **`derive_partialeq(true)`** with `no_partialeq(...)` overrides for structs
  containing function-pointer fields (avoids the
  `unpredictable_function_pointer_comparisons` lint).
- **`opaque_type("std::.*")`** — C++ STL types are opaque pointers.
- **Functions taking an STL type by value or reference are blocklisted and
  shimmed**, never bound directly. bindgen does not fail on them; it silently
  degrades the parameter (a `const std::vector<std::string> &` becomes
  `*const __BindgenOpaqueArray8<[u8; 24]>`, a returned `std::string` becomes
  `u8`), which is an ABI mismatch that compiles. `ext_shim`'s
  `common_fit_params_one` and `common_device_memory_collect` exist for exactly
  this reason: they take C types and build the STL values on the C++ side.
- **OpenMP auto-detection** — reads `GGML_OPENMP_ENABLED` from the CMake
  cache rather than relying solely on the `openmp` feature flag, because
  some ARM toolchains enable OpenMP unconditionally.
- **Optional GPU backends probe for their toolchain and fall back.** These
  checks are fatal, not soft: `enable_language(HIP)` and
  `enable_language(CUDA)` abort the configure step outright when the toolkit
  is missing, `find_package(Dawn REQUIRED)` does the same for WebGPU, and
  OpenCL gets all the way to the compiler before dying on `'CL/cl.h' file not
  found`. That used to take down any build that merely had one of them
  switched on alongside other features — `--all-features` failed on macOS at
  HIP and on a ROCm box at Dawn. The build script looks first:
  - HIP — `ROCM_PATH`, `HIP_PATH`, `hipconfig`, `hipcc`, `/opt/rocm`. The
    driver binaries matter, not just the paths: a distro-packaged ROCm has
    neither `ROCM_PATH` nor `/opt/rocm`, and reports `/usr` from
    `hipconfig --rocmpath`.
  - CUDA — `CUDACXX`, `CMAKE_CUDA_COMPILER`, `CUDA_PATH`, `CUDA_HOME`,
    `CUDA_TOOLKIT_ROOT_DIR`, `nvcc`, `/usr/local/cuda`.
  - Dawn — `Dawn_DIR`, `CMAKE_PREFIX_PATH`, then the usual prefixes, in both
    config-file spellings CMake accepts.
  - OpenCL — the Khronos `CL/cl.h`, because `find_package(OpenCL)` succeeds on
    macOS (the framework is there) and only the *compile* fails. Apple spells
    it `<OpenCL/cl.h>`, so install `opencl-headers` to build this backend on a
    Mac.
  - Vulkan — `VULKAN_SDK` plus headers, library and `glslc`.
  - BLAS — a pkg-config `blas`/`openblas`/`blis`/`mkl` module, or the library
    itself in the usual (including multiarch) lib dirs. `ggml-blas` ends its
    `find_package(BLAS)` with `message(FATAL_ERROR ...)`, and off Apple it
    additionally reaches `find_package(PkgConfig REQUIRED)` whenever
    `FindBLAS` leaves `BLAS_INCLUDE_DIRS` empty — which it did on both test
    hosts, so pkg-config is needed in practice whenever BLAS is. Apple always
    qualifies (Accelerate); Windows is passed through untouched, since BLAS
    there comes from MKL or vcpkg; and an explicit `GGML_BLAS_VENDOR`,
    `MKLROOT` or `OPENBLAS_HOME` also passes through, because a caller who has
    pointed CMake at a BLAS knows more about it than the probe does.

  A missing toolchain warns and builds without that backend, selecting Metal
  instead on Apple targets. The shared CMake cache key records the backends
  actually built, so a fallback build never shares artefacts with a real one.
  The target triple drives the Apple checks, not `cfg!(target_os = ...)`,
  which inside a build script describes the host doing the building.
