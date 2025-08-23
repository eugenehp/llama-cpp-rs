//! Input chunk management for mixed text and media tokenization

use crate::multimodal::{Bitmap, MtmdContext, MultimodalError};
use crate::token::LlamaToken;
use llama_cpp_sys_4 as sys;
use std::ffi::CString;
use std::ptr::NonNull;

/// Text input configuration
#[derive(Debug, Clone)]
pub struct InputText {
    /// The text content
    pub text: String,
    /// Whether to add special tokens
    pub add_special: bool,
    /// Whether to parse special tokens
    pub parse_special: bool,
}

impl InputText {
    /// Create a new text input with default settings
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            add_special: true,
            parse_special: true,
        }
    }
    
    /// Set whether to add special tokens
    pub fn with_add_special(mut self, add: bool) -> Self {
        self.add_special = add;
        self
    }
    
    /// Set whether to parse special tokens
    pub fn with_parse_special(mut self, parse: bool) -> Self {
        self.parse_special = parse;
        self
    }
}

/// Type of input chunk
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkType {
    Text,
    Image,
    Audio,
}

/// A reference to a chunk owned by InputChunks
#[derive(Debug)]
pub struct InputChunkRef<'a> {
    ptr: NonNull<sys::mtmd_input_chunk>,
    _phantom: std::marker::PhantomData<&'a sys::mtmd_input_chunk>,
}

impl<'a> InputChunkRef<'a> {
    /// Get the type of this chunk
    pub fn chunk_type(&self) -> ChunkType {
        let sys_type = unsafe { 
            sys::mtmd_input_chunk_get_type(self.ptr.as_ptr()) 
        };
        
        match sys_type {
            sys::mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_TEXT => ChunkType::Text,
            sys::mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_IMAGE => ChunkType::Image,
            sys::mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_AUDIO => ChunkType::Audio,
            _ => ChunkType::Text,
        }
    }
    
    /// Get the tokens for a text chunk
    pub fn get_text_tokens(&self) -> Option<Vec<LlamaToken>> {
        if self.chunk_type() != ChunkType::Text {
            return None;
        }
        
        unsafe {
            let mut n_tokens = 0;
            let tokens_ptr = sys::mtmd_input_chunk_get_tokens_text(
                self.ptr.as_ptr(),
                &mut n_tokens,
            );
            
            if tokens_ptr.is_null() || n_tokens == 0 {
                None
            } else {
                let tokens = std::slice::from_raw_parts(tokens_ptr, n_tokens)
                    .iter()
                    .map(|&t| LlamaToken(t))
                    .collect();
                Some(tokens)
            }
        }
    }
    
    /// Get the total number of tokens in this chunk
    pub fn n_tokens(&self) -> usize {
        unsafe { sys::mtmd_input_chunk_get_n_tokens(self.ptr.as_ptr()) }
    }
    
    /// Get the number of temporal positions
    pub fn n_pos(&self) -> i32 {
        unsafe { sys::mtmd_input_chunk_get_n_pos(self.ptr.as_ptr()) }
    }
    
    /// Get the ID of this chunk (if any)
    pub fn id(&self) -> Option<String> {
        unsafe {
            let ptr = sys::mtmd_input_chunk_get_id(self.ptr.as_ptr());
            if ptr.is_null() {
                None
            } else {
                std::ffi::CStr::from_ptr(ptr)
                    .to_str()
                    .ok()
                    .map(|s| s.to_string())
            }
        }
    }
}

/// A single chunk of input (text, image, or audio)
#[derive(Debug)]
pub struct InputChunk {
    ptr: NonNull<sys::mtmd_input_chunk>,
    /// Keep ownership to prevent deallocation
    _owned: bool,
}

impl InputChunk {
    /// Get the type of this chunk
    pub fn chunk_type(&self) -> ChunkType {
        let sys_type = unsafe { 
            sys::mtmd_input_chunk_get_type(self.ptr.as_ptr()) 
        };
        
        match sys_type {
            sys::mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_TEXT => ChunkType::Text,
            sys::mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_IMAGE => ChunkType::Image,
            sys::mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_AUDIO => ChunkType::Audio,
            _ => ChunkType::Text,
        }
    }
    
    /// Get the tokens for a text chunk
    pub fn get_text_tokens(&self) -> Option<Vec<LlamaToken>> {
        if self.chunk_type() != ChunkType::Text {
            return None;
        }
        
        unsafe {
            let mut n_tokens = 0;
            let tokens_ptr = sys::mtmd_input_chunk_get_tokens_text(
                self.ptr.as_ptr(),
                &mut n_tokens,
            );
            
            if tokens_ptr.is_null() || n_tokens == 0 {
                None
            } else {
                let tokens = std::slice::from_raw_parts(tokens_ptr, n_tokens)
                    .iter()
                    .map(|&t| LlamaToken(t))
                    .collect();
                Some(tokens)
            }
        }
    }
    
    /// Get the total number of tokens in this chunk
    pub fn n_tokens(&self) -> usize {
        unsafe { sys::mtmd_input_chunk_get_n_tokens(self.ptr.as_ptr()) }
    }
    
    /// Get the number of temporal positions
    /// (always 1 for M-RoPE, n_tokens otherwise)
    pub fn n_pos(&self) -> i32 {
        unsafe { sys::mtmd_input_chunk_get_n_pos(self.ptr.as_ptr()) }
    }
    
    /// Get the ID of this chunk (if any)
    pub fn id(&self) -> Option<String> {
        unsafe {
            let ptr = sys::mtmd_input_chunk_get_id(self.ptr.as_ptr());
            if ptr.is_null() {
                None
            } else {
                std::ffi::CStr::from_ptr(ptr)
                    .to_str()
                    .ok()
                    .map(|s| s.to_string())
            }
        }
    }
    
    /// Create a copy of this chunk (for custom KV cache management)
    pub fn copy(&self) -> Result<Self, MultimodalError> {
        let ptr = unsafe { sys::mtmd_input_chunk_copy(self.ptr.as_ptr()) };
        NonNull::new(ptr)
            .map(|p| Self { ptr: p, _owned: true })
            .ok_or(MultimodalError::NullPointer)
    }
}

impl Drop for InputChunk {
    fn drop(&mut self) {
        if self._owned {
            unsafe {
                sys::mtmd_input_chunk_free(self.ptr.as_ptr());
            }
        }
    }
}

/// Collection of input chunks for mixed tokenization
#[derive(Debug)]
pub struct InputChunks {
    ptr: NonNull<sys::mtmd_input_chunks>,
}

impl InputChunks {
    /// Create a new empty chunks collection
    pub fn new() -> Result<Self, MultimodalError> {
        let ptr = unsafe { sys::mtmd_input_chunks_init() };
        NonNull::new(ptr)
            .map(|p| Self { ptr: p })
            .ok_or(MultimodalError::InitializationFailed)
    }
    
    /// Tokenize text with media placeholders
    /// 
    /// # Arguments
    /// * `context` - The multimodal context
    /// * `text` - Input text with media markers
    /// * `bitmaps` - Bitmaps to replace markers with
    /// 
    /// The text should contain media markers (default: "<__media__>") that will
    /// be replaced with the corresponding bitmaps.
    pub fn tokenize(
        &mut self,
        context: &MtmdContext,
        text: InputText,
        bitmaps: &[&Bitmap],
    ) -> Result<(), MultimodalError> {
        let c_text = CString::new(text.text)
            .map_err(|e| MultimodalError::StringConversion(e))?;
        
        let sys_text = sys::mtmd_input_text {
            text: c_text.as_ptr(),
            add_special: text.add_special,
            parse_special: text.parse_special,
        };
        
        let bitmap_ptrs: Vec<*const sys::mtmd_bitmap> = bitmaps
            .iter()
            .map(|b| b.as_ptr().as_ptr() as *const _)
            .collect();
        
        let result = unsafe {
            sys::mtmd_tokenize(
                context.as_ptr().as_ptr(),
                self.ptr.as_ptr(),
                &sys_text,
                bitmap_ptrs.as_ptr(),
                bitmaps.len(),
            )
        };
        
        match result {
            0 => Ok(()),
            1 => Err(MultimodalError::BitmapCountMismatch {
                provided: bitmaps.len(),
                expected: 0, // We don't know the expected count
            }),
            2 => Err(MultimodalError::PreprocessingFailed),
            _ => Err(MultimodalError::TokenizationFailed {
                reason: format!("Unknown error code: {}", result),
            }),
        }
    }
    
    /// Get the number of chunks
    pub fn len(&self) -> usize {
        unsafe { sys::mtmd_input_chunks_size(self.ptr.as_ptr()) }
    }
    
    /// Check if the chunks collection is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get a chunk by index
    /// 
    /// Returns None if index is out of bounds
    /// 
    /// # Safety
    /// The returned reference is valid as long as self is not mutated
    pub fn get(&self, index: usize) -> Option<InputChunkRef<'_>> {
        if index >= self.len() {
            return None;
        }
        
        unsafe {
            let chunk_ptr = sys::mtmd_input_chunks_get(self.ptr.as_ptr(), index);
            NonNull::new(chunk_ptr as *mut _)
                .map(|ptr| InputChunkRef {
                    ptr,
                    _phantom: std::marker::PhantomData,
                })
        }
    }
    
    /// Iterate over all chunks
    pub fn iter(&self) -> ChunksIter {
        ChunksIter {
            chunks: self,
            index: 0,
        }
    }
}

impl Drop for InputChunks {
    fn drop(&mut self) {
        unsafe {
            sys::mtmd_input_chunks_free(self.ptr.as_ptr());
        }
    }
}

impl Default for InputChunks {
    fn default() -> Self {
        Self::new().expect("Failed to create InputChunks")
    }
}

/// Iterator over input chunks
#[derive(Debug)]
pub struct ChunksIter<'a> {
    chunks: &'a InputChunks,
    index: usize,
}

impl<'a> Iterator for ChunksIter<'a> {
    type Item = InputChunkRef<'a>;
    
    fn next(&mut self) -> Option<Self::Item> {
        let chunk = self.chunks.get(self.index);
        if chunk.is_some() {
            self.index += 1;
        }
        chunk
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.chunks.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for ChunksIter<'a> {}