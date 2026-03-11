# macOS setup notes

## Prerequisites

Install Xcode Command Line Tools:

```bash
xcode-select --install
```

Install Homebrew (if needed):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install Rust (stable):

```bash
brew install rustup-init
rustup-init -y
source "$HOME/.cargo/env"
```

Install Git and CMake:

```bash
brew install git cmake
```

## Clone with submodules

```bash
git clone --recursive https://github.com/eugenehp/llama-cpp-rs
cd llama-cpp-rs
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Build

```bash
cargo build
```

## Vulkan on macOS (`--features vulkan`)

This project now supports a graceful fallback on macOS:

- If Vulkan SDK is available and complete, `--features vulkan` builds with Vulkan.
- If Vulkan SDK is missing/incomplete, `--features vulkan` automatically falls back to Metal.

The Vulkan availability check expects:

- `VULKAN_SDK` set
- Vulkan headers at `$VULKAN_SDK/include/vulkan/vulkan.h`
- Vulkan or MoltenVK library in `$VULKAN_SDK/lib`
- `glslc` available in `PATH`

Build command:

```bash
cargo build --features vulkan
```

When Vulkan is unavailable on macOS, you should see a cargo warning indicating fallback to Metal.

## Optional: explicit Metal build

```bash
cargo build --features metal
```

## Notes

- These instructions are documented for macOS but were not executed in this Ubuntu environment.
- CI currently validates Vulkan builds on Linux and Windows; macOS fallback behavior is implemented in `llama-cpp-sys-4/build.rs`.
