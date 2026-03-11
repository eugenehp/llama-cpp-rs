# Windows setup notes

## Prerequisites

Install Rust (stable):

```powershell
winget install --id Rustlang.Rustup -e
```

Install Visual Studio Build Tools (C++ workload):

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
```

Install Git:

```powershell
winget install --id Git.Git -e
```

## Clone with submodules

```powershell
git clone --recursive https://github.com/eugenehp/llama-cpp-rs
cd llama-cpp-rs
```

If already cloned without submodules:

```powershell
git submodule update --init --recursive
```

## Vulkan build prerequisites

Install Vulkan SDK (includes headers, loader, and `glslc`):

```powershell
choco install vulkan-sdk -y
```

Set SDK environment variables for the current session (PowerShell):

```powershell
$latest = Get-ChildItem 'C:\VulkanSDK' -Directory | Sort-Object Name -Descending | Select-Object -First 1
$env:VULKAN_SDK = $latest.FullName
$env:Path = "$($latest.FullName)\Bin;$env:Path"
```

Verify tools:

```powershell
glslc --version
vulkaninfo --summary
```

## Build

```powershell
cargo build
cargo build --features vulkan
```

## Common Vulkan error

If you see:

- `Could NOT find Vulkan (missing: Vulkan_LIBRARY Vulkan_INCLUDE_DIR glslc)`

then the Vulkan SDK is not installed correctly, or `VULKAN_SDK`/`PATH` are not set in your current shell.

## CI check

The repository CI includes a Windows Vulkan build check in:

- `.github/workflows/llama-cpp-rs-check.yml`
