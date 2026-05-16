# Changelog

## [0.2.56] - 2026-05-16

### Changed
- **llama.cpp**: bumped vendored submodule to `64b38b561` (master, 2026-05-16),
  which now includes upstream MTP support (PR #22673,
  `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` / `LLAMA_CONTEXT_TYPE_MTP`).
- **Patch removed**: `llama-cpp-sys-4/patches/0002-mtp.patch` is gone — its
  functionality is now upstream. The `mtp` Cargo feature has been removed
  from both `llama-cpp-sys-4` and `llama-cpp-4`.

### Added
- New `LlamaContextType { Default, Mtp }` enum and
  `LlamaContextParams::with_ctx_type` / `ctx_type` wrapping upstream's
  `llama_context_type` (use `Mtp` to load a draft head as the MTP context for
  upstream's `--spec-type draft-mtp` speculative decoder).
- `LlamaContextParams::with_n_rs_seq` / `n_rs_seq` and
  `LlamaContext::n_rs_seq` are now always available (no feature gate).
- New `llama_cpp_4::mtp::MtpSession` — Rust-callable MTP speculative-decoding
  draft loop. Wraps a small C++ shim
  (`llama-cpp-sys-4/mtp_shim/mtp_shim.cpp`) that re-exports upstream's
  `common_speculative_*` MTP path with stable C linkage. Smoke-tested
  end-to-end on Qwen3.6-27B-IQ2_M with 94% draft acceptance.
- New `examples/mtp/` — without `--predict` configures contexts (smoke test);
  with `--predict N` drives the full draft loop via `MtpSession`.

### Removed (breaking)
- `mtp` Cargo feature on both crates.
- `LlamaContext::set_mtp` — upstream removed the `llama_set_mtp` C API; MTP is
  now configured via `ctx_type` on the context, not by post-hoc attachment.
- `LlamaModelParams::with_override_arch` / `override_arch` — the corresponding
  `override_arch` field on `llama_model_params` is gone upstream; MTP head
  architecture is detected automatically from the GGUF metadata.
- `llama_cpp_sys_4::llama_context_seq_rm` — the patched alias is gone; use
  `llama_get_memory` + `llama_memory_seq_rm` (which `clear_kv_cache_seq`
  already does internally).

### Migration
- Drop `features = ["mtp"]` from your `Cargo.toml`.
- Replace `set_mtp(Some(&draft_ctx))` with constructing the draft context from
  `LlamaContextParams::default().with_ctx_type(LlamaContextType::Mtp)`.
- Replace `with_override_arch(...)` calls with nothing — autodetected.
- `scripts/bench-mtp.sh` now passes `--spec-type draft-mtp` (was `mtp`).

## [0.2.43] - 2026-04-10

### Changed

- **Build System**: Changed default library type from dynamic to static
  - Default builds now produce static libraries (.a files)
  - Shared libraries are only built when the `dynamic-link` feature is explicitly enabled
  - Backend features (cuda, metal, blas, vulkan, etc.) no longer force shared library builds
  - The `dynamic-link` feature can be combined with any backend feature to produce shared libraries
  - Environment variable `LLAMA_BUILD_SHARED_LIBS` can override the default behavior

### Backward Compatibility

This change maintains backward compatibility for most use cases:
- Applications using default builds will now get static libraries instead of dynamic ones
- Applications explicitly using the `dynamic-link` feature will continue to work as before
- All backend features (cuda, metal, blas, etc.) continue to work as expected
- The `LLAMA_BUILD_SHARED_LIBS` environment variable provides an escape hatch for special requirements

### Migration Guide

If you were relying on the old behavior (dynamic libraries by default):

1. **Explicitly enable dynamic-link feature**:
   ```bash
   cargo build --features dynamic-link
   ```

2. **Or set the environment variable**:
   ```bash
   LLAMA_BUILD_SHARED_LIBS=1 cargo build
   ```

3. **For Cargo.toml**:
   ```toml
   [dependencies.llama-cpp-sys-4]
   version = "0.2.43"
   features = ["dynamic-link"]
   ```

### Benefits

- **Smaller distribution size**: Static libraries are self-contained
- **Easier deployment**: No need to manage separate .dylib/.so files
- **Better compatibility**: Static linking avoids library version conflicts
- **Explicit control**: Developers can choose the linking strategy that best fits their needs
