# Changelog

## Unreleased

## [0.3.2] - 2026-06-20

### Changed
- **llama.cpp**: vendored submodule `94a220cd6` → `c57607016` (master, 2026-06-21;
  [PR #256](https://github.com/eugenehp/llama-cpp-rs/pull/256)). The public `llama.h`
  C API is unchanged; the `mtmd` helper API was reworked (see below).

### Added
- **EAGLE-3 speculative decoding** (`llama-cpp-4`): new [`eagle::Eagle3Session`]
  driving upstream `COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3`. Pairs a target context
  with a separate EAGLE-3 draft-model context. The `mtp_shim` is generalised with a
  `spec_type` selector shared by MTP and EAGLE-3 (`MtpSession` is unchanged).
- **`examples/eagle`** (+ README) and weight-fetch scripts: runnable EAGLE-3 demo plus
  `scripts/setup-eagle3.sh` (one command: bootstraps a convert env, then downloads +
  converts a target/draft pairing to GGUF) over `scripts/fetch-eagle3.sh` (download +
  convert). Defaults to the open Qwen3-8B + `RedHatAI/Qwen3-8B-speculator.eagle3`.
  Verified end-to-end on Apple M4 Pro (Metal): coherent generation at ~53% draft
  acceptance.
- **mtmd video input** (`llama-cpp-4`, `mtmd` feature): [`mtmd::MtmdVideo`] (+
  `MtmdVideoParams`, `MtmdVideoInfo`, `MtmdVideoItem`) wrapping the new
  `mtmd_helper_video_*` API for frame-by-frame decoding via ffmpeg. Plus
  `MtmdContext::supports_video()` and `MtmdContext::marker()` (per-context marker).

### Fixed
- **mtmd bindings** (`llama-cpp-4`): adapt to the reworked `mtmd` helper API —
  `mtmd_helper_bitmap_init_from_file`/`_from_buf` now take a `placeholder` flag and
  return a `mtmd_helper_bitmap_wrapper`; `mtmd_helper_decode_image_chunk` takes a
  post-decode callback. `MtmdBitmap::from_file`/`from_buf` also no longer leak the
  `video_ctx` returned for video inputs.

## [0.3.1] - 2026-06-04

### Changed
- **llama.cpp**: vendored submodule `b28a2f372` → `94a220cd6` (master, 2026-06-04;
  ~250 commits). Notable upstream: unified Gemma 4 FPE fix
  ([#24088](https://github.com/ggml-org/llama.cpp/pull/24088)), `LLAMA_BUILD_APP`
  unified binary, embedding API rename to **next-n**
  (`llama_set_embeddings_nextn`, `common_speculative_need_embd_nextn`).
- **Bindings** (`llama-cpp-4`): `LlamaContext` embedding getters/setters call upstream
  `llama_*_nextn` FFI; Rust method names stay `*_pre_norm` for compatibility.
- **`llama-cpp-sys-4` build** (`build.rs`): set `LLAMA_BUILD_APP=OFF` always; set
  `LLAMA_BUILD_COMMON=OFF` when the `mtmd` feature is disabled so the OUT_DIR CMake
  copy builds library targets only (fixes build failure after the submodule bump).
- **`mtp_shim`**: `mtp_session_need_embd_pre_norm` delegates to
  `common_speculative_need_embd_nextn`.

### Added
- **`openai-server`** ([`examples/server/README.md`](examples/server/README.md)):
  - `GET /v1/health` (alias of `/health`, both public when `--api-key` is set)
  - Legacy llama.cpp paths: `/chat/completions`, `/completions`, `/embeddings`
  - `POST /tokenize`, `POST /detokenize` (same JSON shape as upstream server)
  - `max_completion_tokens` accepted as an alias for `max_tokens` on chat/completion routes
  - Integration tests: `/v1/health`, tokenize/detokenize roundtrip, `max_completion_tokens`
- **Docs**: server endpoint tables and MTP next-n naming notes in root `README.md`,
  `llama-cpp-4/README.md`, `llama-cpp-sys-4/README.md`, and rustdoc on `mtp` / `context`.

### Fixed
- **`openai-server`**: compile against regenerated bindings after the llama.cpp bump
  (`llama_get_embeddings_nextn` / related symbols).

## [0.3.0] - 2026-05-19

### Changed
- **llama.cpp**: bumped vendored submodule to `b28a2f372` (includes MTP clean-up
  [#23269](https://github.com/ggml-org/llama.cpp/pull/23269)).
- **MTP draft API**: `MtpSession::new_with_config` and [`MtpSessionConfig`]
  expose `n_min` and `p_min`; upstream default `p_min` is now `0.0`.
- **CI**: Linux dynamic prebuilt collection now includes versioned `.so` files
  and symlinks.

### Added
- [`MtpSession::need_embd_pre_norm`], [`MtpSession::print_stats`],
  [`MtpSession::config`], [`MtpSession::n_min`], [`MtpSession::p_min`].
- `examples/mtp`: `--p-min` CLI flag; session stats printed after generation.

[`MtpSessionConfig`]: llama-cpp-4/src/mtp.rs
[`MtpSession::need_embd_pre_norm`]: llama-cpp-4/src/mtp.rs
[`MtpSession::print_stats`]: llama-cpp-4/src/mtp.rs
[`MtpSession::config`]: llama-cpp-4/src/mtp.rs
[`MtpSession::n_min`]: llama-cpp-4/src/mtp.rs
[`MtpSession::p_min`]: llama-cpp-4/src/mtp.rs

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
