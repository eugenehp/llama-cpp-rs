//! Multimodal context management

use crate::model::LlamaModel;
use crate::multimodal::error::MultimodalError;
use llama_cpp_sys_4 as sys;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr::NonNull;
use std::sync::Arc;

/// Parameters for creating a multimodal context
#[derive(Debug, Clone)]
pub struct MtmdContextParams {
    /// Whether to use GPU for processing
    pub use_gpu: bool,
    /// Whether to print timing information
    pub print_timings: bool,
    /// Number of threads to use
    pub n_threads: i32,
    /// Verbosity level for logging
    pub verbosity: i32,
    /// Media marker in text (defaults to "<__media__>")
    pub media_marker: Option<String>,
}

impl Default for MtmdContextParams {
    fn default() -> Self {
        Self {
            use_gpu: true,
            print_timings: false,
            n_threads: 4,
            verbosity: 0,
            media_marker: None,
        }
    }
}

/// Multimodal context for processing images and audio alongside text
#[derive(Debug)]
pub struct MtmdContext {
    ptr: NonNull<sys::mtmd_context>,
    /// Keep a reference to the model to ensure it outlives the context
    _model: Arc<LlamaModel>,
}

impl MtmdContext {
    /// Create a new multimodal context from a projector file
    pub fn new_from_file(
        mmproj_path: &str,
        model: Arc<LlamaModel>,
        params: MtmdContextParams,
    ) -> Result<Self, MultimodalError> {
        let c_path = CString::new(mmproj_path)
            .map_err(|e| MultimodalError::StringConversion(e))?;
        
        let mut sys_params = unsafe {
            // Safety: mtmd_context_params_default returns a valid struct
            sys::mtmd_context_params_default()
        };
        
        sys_params.use_gpu = params.use_gpu;
        sys_params.print_timings = params.print_timings;
        sys_params.n_threads = params.n_threads;
        // Safe conversion with bounds checking
        sys_params.verbosity = params.verbosity.min(3).max(0) as sys::ggml_log_level;
        
        // Store CString to keep it alive during the call
        let c_marker = params.media_marker
            .map(|marker| CString::new(marker))
            .transpose()
            .map_err(|e| MultimodalError::StringConversion(e))?;
        
        if let Some(ref marker) = c_marker {
            sys_params.media_marker = marker.as_ptr();
        }
        
        let ctx = unsafe {
            // Safety: All pointers are valid for the duration of this call
            // - c_path is kept alive until after the call
            // - model pointer is valid as long as Arc keeps it alive
            // - sys_params is a valid struct
            sys::mtmd_init_from_file(
                c_path.as_ptr(),
                model.as_ptr().as_ptr(),
                sys_params,
            )
        };
        
        NonNull::new(ctx)
            .map(|ptr| Self { ptr, _model: model })
            .ok_or(MultimodalError::InitializationFailed)
    }
    
    /// Check if the model needs non-causal mask for decoding
    pub fn decode_use_non_causal(&self) -> bool {
        unsafe { sys::mtmd_decode_use_non_causal(self.ptr.as_ptr()) }
    }
    
    /// Check if the model uses M-RoPE for decoding
    pub fn decode_use_mrope(&self) -> bool {
        unsafe { sys::mtmd_decode_use_mrope(self.ptr.as_ptr()) }
    }
    
    /// Check if the model supports vision input
    pub fn supports_vision(&self) -> bool {
        unsafe { sys::mtmd_support_vision(self.ptr.as_ptr()) }
    }
    
    /// Check if the model supports audio input
    pub fn supports_audio(&self) -> bool {
        unsafe { sys::mtmd_support_audio(self.ptr.as_ptr()) }
    }
    
    /// Get audio bitrate in Hz (e.g., 16000 for Whisper)
    /// Returns None if audio is not supported
    pub fn get_audio_bitrate(&self) -> Option<i32> {
        let bitrate = unsafe { sys::mtmd_get_audio_bitrate(self.ptr.as_ptr()) };
        if bitrate >= 0 {
            Some(bitrate)
        } else {
            None
        }
    }
    
    /// Get the default media marker string
    pub fn default_marker() -> &'static str {
        unsafe {
            let ptr = sys::mtmd_default_marker();
            CStr::from_ptr(ptr)
                .to_str()
                .unwrap_or("<__media__>")
        }
    }
    
    /// Get the raw pointer for FFI calls
    pub(crate) fn as_ptr(&self) -> NonNull<sys::mtmd_context> {
        self.ptr
    }
}

impl Drop for MtmdContext {
    fn drop(&mut self) {
        unsafe {
            sys::mtmd_free(self.ptr.as_ptr());
        }
    }
}

// Safety: MtmdContext can be sent between threads
unsafe impl Send for MtmdContext {}
// Safety: MtmdContext can be shared between threads (the C API is thread-safe)
unsafe impl Sync for MtmdContext {}