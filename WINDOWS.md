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

The build script **automatically detects** the Vulkan SDK by checking:

1. The `VULKAN_SDK` environment variable (if set).
2. The Windows registry (`HKLM\SOFTWARE\LunarG\VulkanSDK`).
3. The default install directory `C:\VulkanSDK\<latest version>`.

If automatic detection fails, set the SDK path manually for the current session (PowerShell):

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

then the Vulkan SDK is not installed correctly. The build script tries to find it automatically, but if that fails, ensure `VULKAN_SDK` is set and `glslc` is on your `PATH`.

## CI check

The repository CI includes a Windows Vulkan build check in:

- `.github/workflows/llama-cpp-rs-check.yml`
