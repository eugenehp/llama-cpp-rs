//! Error types for multimodal functionality

use std::ffi::NulError;
use thiserror::Error;

/// Errors that can occur in multimodal operations
#[derive(Debug, Error)]
pub enum MultimodalError {
    /// Failed to initialize multimodal context
    #[error("Failed to initialize multimodal context")]
    InitializationFailed,
    
    /// Invalid image dimensions
    #[error("Invalid image dimensions: {width}x{height}")]
    InvalidImageDimensions { width: u32, height: u32 },
    
    /// Invalid audio sample count
    #[error("Invalid audio sample count: {count}")]
    InvalidAudioSamples { count: usize },
    
    /// Tokenization failed
    #[error("Tokenization failed: {reason}")]
    TokenizationFailed { reason: String },
    
    /// Bitmap count mismatch
    #[error("Number of bitmaps ({provided}) doesn't match markers in text ({expected})")]
    BitmapCountMismatch { provided: usize, expected: usize },
    
    /// Image preprocessing error
    #[error("Image preprocessing failed")]
    PreprocessingFailed,
    
    /// Null pointer error
    #[error("Null pointer encountered")]
    NullPointer,
    
    /// String conversion error
    #[error("Failed to convert C string: {0}")]
    StringConversion(#[from] NulError),
    
    /// UTF-8 conversion error
    #[error("Invalid UTF-8: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),
    
    /// Feature not supported
    #[error("Feature not supported: {feature}")]
    NotSupported { feature: String },
}