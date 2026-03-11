# Linux setup notes

## Vulkan build prerequisites

If `cargo build --features vulkan` fails with:

- `Could NOT find Vulkan (missing: Vulkan_LIBRARY Vulkan_INCLUDE_DIR glslc)`

install the Vulkan development packages and GLSL compiler:

```bash
sudo apt update
sudo apt install -y libvulkan-dev vulkan-tools glslc
```

Notes:

- On Ubuntu Questing, `shaderc` is not a valid package name.
- If needed, use `libshaderc-dev` instead:

```bash
sudo apt install -y libshaderc-dev
```

## Verify installation

```bash
command -v glslc
command -v vulkaninfo
ls /usr/include/vulkan/vulkan.h
```

## Build with Vulkan

```bash
cargo clean
cargo build --features vulkan
```
