# llama-cpp-sys-4

[![Crates.io](https://img.shields.io/crates/v/llama-cpp-sys-4.svg)](https://crates.io/crates/llama-cpp-sys-4)
[![License](https://img.shields.io/crates/l/llama-cpp-sys-4.svg)](https://crates.io/crates/llama-cpp-sys-4)

Raw `bindgen`-generated bindings to [llama.cpp](https://github.com/ggerganov/llama.cpp),
plus the C/C++ build logic that compiles the library.

**llama.cpp version:** b8249 · **Crate version:** 0.2.5

Unless you need access to a symbol not yet exposed by [`llama-cpp-4`](../llama-cpp-4/),
use that crate instead — it provides a safe API over these raw bindings.

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

### Build dependencies

- `clang` — required by `bindgen` to parse the C++ headers
- A C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- `cmake` is **not** required — the build is driven entirely by `build.rs`

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

## Notable API changes (b4689 → b8249)

These are the upstream llama.cpp breaks handled in this crate:

| Removed / renamed | Replacement |
|---|---|
| `llama_kv_cache_*` functions | `llama_memory_*` via `llama_get_memory(ctx)` |
| `llama_set_adapter_lora` + `llama_rm_adapter_lora` | `llama_set_adapters_lora` (batch API) |
| `context_params.flash_attn: bool` | `context_params.flash_attn_type: llama_flash_attn_type` |
| `llama-sampling.h` | `llama-sampler.h` |
| C++11 build flag | C++17 required by new `common.h` (`std::string_view`) |

---

## Bindgen configuration

Key decisions in `build.rs`:

- **`derive_partialeq(true)`** with `no_partialeq(...)` overrides for structs
  containing function-pointer fields (avoids the
  `unpredictable_function_pointer_comparisons` lint).
- **`opaque_type("std::.*")`** — C++ STL types are opaque pointers.
- **OpenMP auto-detection** — reads `GGML_OPENMP_ENABLED` from the CMake
  cache rather than relying solely on the `openmp` feature flag, because
  some ARM toolchains enable OpenMP unconditionally.
