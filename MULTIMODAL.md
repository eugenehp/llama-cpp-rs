# Multimodal Support in llama-cpp-rs

This document describes the multimodal support added to llama-cpp-rs, enabling processing of images and audio alongside text using vision and audio-capable models.

## Features

The multimodal support is implemented as an optional feature flag and provides:

- **Vision Support**: Process images with models like LLaVA, Qwen2-VL, Gemma3, SmolVLM, Pixtral, etc.
- **Audio Support**: Process audio with models that support audio input
- **Mixed Tokenization**: Seamlessly combine text and media inputs
- **Thread-Safe API**: Safe Rust wrappers around the libmtmd C API

## Building with Multimodal Support

To enable multimodal support, build with the `multimodal` feature flag:

```bash
cargo build --features multimodal
```

Or add to your `Cargo.toml`:

```toml
[dependencies]
llama-cpp-4 = { version = "0.1", features = ["multimodal"] }
```

## Usage

### Basic Example

```rust
use llama_cpp_4::multimodal::{
    MtmdContext, MtmdContextParams, Bitmap, InputChunks, InputText
};

// Load the main model
let model = LlamaModel::load_from_file("model.gguf", params)?;

// Initialize multimodal context with projector
let mtmd_params = MtmdContextParams::default();
let mtmd_context = MtmdContext::new_from_file(
    "mmproj.gguf",
    model,
    mtmd_params
)?;

// Load and prepare an image
let image_data = load_rgb_image("image.jpg")?; // Your image loading
let bitmap = Bitmap::new_image(width, height, &image_data)?;

// Prepare text with media placeholder
let text = InputText::new("Describe this image: <__media__>");
let bitmaps = vec![&bitmap];

// Tokenize mixed input
let mut chunks = InputChunks::new()?;
chunks.tokenize(&mtmd_context, text, &bitmaps)?;

// Process chunks...
```

### Processing Audio

```rust
// Load PCM F32 audio samples
let audio_samples: Vec<f32> = load_audio("audio.wav")?; // Your audio loading
let bitmap = Bitmap::new_audio(&audio_samples)?;

// Use same tokenization process
let text = InputText::new("Transcribe this audio: <__media__>");
let bitmaps = vec![&bitmap];
chunks.tokenize(&mtmd_context, text, &bitmaps)?;
```

## Architecture

The multimodal support consists of several components:

### 1. Build System (`llama-cpp-sys-4/build.rs`)
- Conditionally compiles `libmtmd` and `clip.cpp` when the feature is enabled
- Generates FFI bindings for multimodal functions
- Links the multimodal libraries

### 2. Rust API (`llama-cpp-4/src/multimodal/`)
- `context.rs`: `MtmdContext` for managing multimodal processing
- `bitmap.rs`: `Bitmap` for handling images and audio
- `chunks.rs`: `InputChunks` for mixed tokenization
- `error.rs`: Error types for multimodal operations

### 3. Example (`examples/multimodal/`)
- Complete example showing image processing with a vision model

## Supported Models

The multimodal support works with models that have corresponding multimodal projector (`mmproj`) files:

### Models with Native Support
- Gemma 3
- SmolVLM / SmolVLM2
- Pixtral 12B
- Qwen 2 VL / Qwen 2.5 VL
- Mistral Small 3.1 24B
- InternVL 2.5 / InternVL 3

### Legacy Models (require conversion scripts)
- LLaVA
- MobileVLM
- GLM-Edge
- MiniCPM-V 2.5/2.6
- MiniCPM-o 2.6
- IBM Granite Vision

## Obtaining Multimodal Projector Files

For supported models, use `convert_hf_to_gguf.py` with the `--mmproj` flag:

```bash
python convert_hf_to_gguf.py model_dir --mmproj mmproj.gguf
```

For legacy models, refer to the specific conversion scripts in `llama.cpp/tools/mtmd/legacy-models/`.

## API Reference

### MtmdContext

```rust
pub struct MtmdContext { ... }

impl MtmdContext {
    pub fn new_from_file(path: &str, model: LlamaModel, params: MtmdContextParams) -> Result<Self>;
    pub fn supports_vision(&self) -> bool;
    pub fn supports_audio(&self) -> bool;
    pub fn get_audio_bitrate(&self) -> Option<i32>;
    pub fn decode_use_non_causal(&self) -> bool;
    pub fn decode_use_mrope(&self) -> bool;
}
```

### Bitmap

```rust
pub struct Bitmap { ... }

impl Bitmap {
    pub fn new_image(width: u32, height: u32, data: &[u8]) -> Result<Self>;
    pub fn new_audio(samples: &[f32]) -> Result<Self>;
    pub fn set_id(&mut self, id: &str) -> Result<()>;
    pub fn bitmap_type(&self) -> BitmapType;
}
```

### InputChunks

```rust
pub struct InputChunks { ... }

impl InputChunks {
    pub fn new() -> Result<Self>;
    pub fn tokenize(&mut self, context: &MtmdContext, text: InputText, bitmaps: &[&Bitmap]) -> Result<()>;
    pub fn iter(&self) -> ChunksIter;
}
```

## Limitations and Notes

1. **Feature Flag Required**: Multimodal support is only available when built with the `multimodal` feature flag
2. **Model Compatibility**: Requires models specifically trained for multimodal input
3. **Memory Usage**: Processing images/audio requires additional memory
4. **Thread Safety**: The API is thread-safe but contexts should not be shared during active processing

## Future Improvements

- [ ] Add support for batch processing of multiple images
- [ ] Implement streaming for large audio files
- [ ] Add preprocessing utilities for common image formats
- [ ] Support for video frame extraction
- [ ] Integration with popular image/audio libraries

## Contributing

When contributing to multimodal support:
1. Ensure changes are gated behind the `multimodal` feature flag
2. Update both the low-level bindings and high-level API
3. Add tests for new functionality
4. Update this documentation

## References

- [llama.cpp multimodal documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd)
- [libmtmd API reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/mtmd.h)