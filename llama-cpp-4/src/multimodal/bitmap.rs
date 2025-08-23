//! Bitmap handling for images and audio

use crate::multimodal::error::MultimodalError;
use llama_cpp_sys_4 as sys;
use std::ffi::{CStr, CString};
use std::ptr::NonNull;

/// Type of bitmap data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitmapType {
    /// RGB image data
    Image,
    /// PCM F32 audio data
    Audio,
}

/// A bitmap that can represent either an image or audio data
#[derive(Debug)]
pub struct Bitmap {
    ptr: NonNull<sys::mtmd_bitmap>,
}

impl Bitmap {
    /// Create a new image bitmap from RGB data
    /// 
    /// # Arguments
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `data` - RGB data in RGBRGBRGB... format (must be width * height * 3 bytes)
    pub fn new_image(width: u32, height: u32, data: &[u8]) -> Result<Self, MultimodalError> {
        let expected_size = (width * height * 3) as usize;
        if data.len() != expected_size {
            return Err(MultimodalError::InvalidImageDimensions { width, height });
        }
        
        let bitmap = unsafe {
            sys::mtmd_bitmap_init(width, height, data.as_ptr())
        };
        
        NonNull::new(bitmap)
            .map(|ptr| Self { ptr })
            .ok_or(MultimodalError::InitializationFailed)
    }
    
    /// Create a new audio bitmap from PCM F32 samples
    /// 
    /// # Arguments
    /// * `samples` - Audio samples in PCM F32 format
    pub fn new_audio(samples: &[f32]) -> Result<Self, MultimodalError> {
        if samples.is_empty() {
            return Err(MultimodalError::InvalidAudioSamples { count: 0 });
        }
        
        let bitmap = unsafe {
            sys::mtmd_bitmap_init_from_audio(samples.len(), samples.as_ptr())
        };
        
        NonNull::new(bitmap)
            .map(|ptr| Self { ptr })
            .ok_or(MultimodalError::InitializationFailed)
    }
    
    /// Get the width of an image bitmap (or 0 for audio)
    pub fn width(&self) -> u32 {
        unsafe { sys::mtmd_bitmap_get_nx(self.ptr.as_ptr()) }
    }
    
    /// Get the height of an image bitmap (or 0 for audio)
    pub fn height(&self) -> u32 {
        unsafe { sys::mtmd_bitmap_get_ny(self.ptr.as_ptr()) }
    }
    
    /// Get the raw data of the bitmap
    pub fn data(&self) -> &[u8] {
        unsafe {
            let ptr = sys::mtmd_bitmap_get_data(self.ptr.as_ptr());
            let size = sys::mtmd_bitmap_get_n_bytes(self.ptr.as_ptr());
            if ptr.is_null() || size == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(ptr, size)
            }
        }
    }
    
    /// Check if this is an audio bitmap
    pub fn is_audio(&self) -> bool {
        unsafe { sys::mtmd_bitmap_is_audio(self.ptr.as_ptr()) }
    }
    
    /// Get the type of this bitmap
    pub fn bitmap_type(&self) -> BitmapType {
        if self.is_audio() {
            BitmapType::Audio
        } else {
            BitmapType::Image
        }
    }
    
    /// Set an ID for this bitmap (useful for KV cache tracking)
    pub fn set_id(&mut self, id: &str) -> Result<(), MultimodalError> {
        let c_id = CString::new(id)
            .map_err(|e| MultimodalError::StringConversion(e))?;
        unsafe {
            sys::mtmd_bitmap_set_id(self.ptr.as_ptr(), c_id.as_ptr());
        }
        Ok(())
    }
    
    /// Get the ID of this bitmap if set
    pub fn id(&self) -> Option<String> {
        unsafe {
            let ptr = sys::mtmd_bitmap_get_id(self.ptr.as_ptr());
            if ptr.is_null() {
                None
            } else {
                CStr::from_ptr(ptr)
                    .to_str()
                    .ok()
                    .map(|s| s.to_string())
            }
        }
    }
    
    /// Get the raw pointer for FFI calls
    pub(crate) fn as_ptr(&self) -> NonNull<sys::mtmd_bitmap> {
        self.ptr
    }
}

impl Drop for Bitmap {
    fn drop(&mut self) {
        unsafe {
            sys::mtmd_bitmap_free(self.ptr.as_ptr());
        }
    }
}

// Safety: Bitmap can be sent between threads
unsafe impl Send for Bitmap {}

/// Builder for creating bitmaps with additional options
#[derive(Debug, Clone)]
pub struct BitmapBuilder {
    id: Option<String>,
}

impl BitmapBuilder {
    /// Create a new bitmap builder
    pub fn new() -> Self {
        Self { id: None }
    }
    
    /// Set the ID for the bitmap
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
    
    /// Build an image bitmap
    pub fn build_image(self, width: u32, height: u32, data: &[u8]) -> Result<Bitmap, MultimodalError> {
        let mut bitmap = Bitmap::new_image(width, height, data)?;
        if let Some(id) = self.id {
            bitmap.set_id(&id)?;
        }
        Ok(bitmap)
    }
    
    /// Build an audio bitmap
    pub fn build_audio(self, samples: &[f32]) -> Result<Bitmap, MultimodalError> {
        let mut bitmap = Bitmap::new_audio(samples)?;
        if let Some(id) = self.id {
            bitmap.set_id(&id)?;
        }
        Ok(bitmap)
    }
}

impl Default for BitmapBuilder {
    fn default() -> Self {
        Self::new()
    }
}