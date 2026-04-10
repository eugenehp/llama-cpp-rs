# Changelog

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
