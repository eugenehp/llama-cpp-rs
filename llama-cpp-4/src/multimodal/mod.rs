//! Multimodal support for llama-cpp
//! 
//! This module provides support for processing images and audio alongside text
//! using the libmtmd multimodal library from llama.cpp.

#[cfg(feature = "multimodal")]
pub mod context;

#[cfg(feature = "multimodal")]
pub mod bitmap;

#[cfg(feature = "multimodal")]
pub mod chunks;

#[cfg(feature = "multimodal")]
pub mod error;

#[cfg(feature = "multimodal")]
pub use context::MtmdContext;

#[cfg(feature = "multimodal")]
pub use bitmap::{Bitmap, BitmapType};

#[cfg(feature = "multimodal")]
pub use chunks::{InputChunk, InputChunks, InputText, InputChunkRef, ChunkType};

#[cfg(feature = "multimodal")]
pub use error::MultimodalError;