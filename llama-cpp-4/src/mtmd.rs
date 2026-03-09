//! Safe wrappers for the `libmtmd` multimodal support library.
//!
//! `libmtmd` extends llama.cpp with the ability to encode image and audio
//! inputs (bitmaps) into token embeddings that can then be fed into a
//! standard [`llama_decode`] call alongside normal text tokens.
//!
//! # Quick-start
//!
//! ```no_run
//! # #[cfg(feature = "mtmd")]
//! # {
//! use std::path::Path;
//! use llama_cpp_4::{
//!     llama_backend::LlamaBackend,
//!     model::{LlamaModel, params::LlamaModelParams, AddBos},
//!     context::params::LlamaContextParams,
//!     mtmd::{MtmdContext, MtmdContextParams, MtmdBitmap, MtmdInputChunks, MtmdInputText},
//! };
//!
//! let backend  = LlamaBackend::init().unwrap();
//! let model    = LlamaModel::load_from_file(&backend, Path::new("model.gguf"),
//!                                            &LlamaModelParams::default()).unwrap();
//! let mut lctx = model.new_context(&backend, LlamaContextParams::default()).unwrap();
//!
//! // Load the multimodal projector (mmproj) model.
//! let ctx_params = MtmdContextParams::default();
//! let mtmd_ctx   = MtmdContext::init_from_file(Path::new("mmproj.gguf"), &model, ctx_params)
//!                               .unwrap();
//!
//! // Load an image from a file.
//! let bitmap = MtmdBitmap::from_file(&mtmd_ctx, Path::new("image.jpg")).unwrap();
//!
//! // Tokenize a prompt that contains the media marker.
//! let marker  = MtmdContext::default_marker();
//! let prompt  = format!("Describe this image: {marker}");
//! let text    = MtmdInputText::new(&prompt, true, true);
//! let bitmaps = [&bitmap];
//!
//! let mut chunks = MtmdInputChunks::new();
//! mtmd_ctx.tokenize(&text, &bitmaps, &mut chunks).unwrap();
//!
//! // Evaluate / decode all chunks.
//! let n_batch = lctx.n_batch() as i32;
//! let mut n_past = 0i32;
//! mtmd_ctx.eval_chunks(lctx.as_ptr(), &chunks, 0, 0, n_batch, true, &mut n_past).unwrap();
//! # }
//! ```
//!
//! # Feature flag
//!
//! This module is only compiled when the `mtmd` Cargo feature is enabled.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::NonNull;
use std::slice;

use llama_cpp_sys_4 as sys;

use crate::model::LlamaModel;

// ─────────────────────────────────────────────────────────────────────────────
// Error types
// ─────────────────────────────────────────────────────────────────────────────

/// All errors that can be returned by the mtmd module.
#[derive(Debug, thiserror::Error)]
pub enum MtmdError {
    /// The context could not be created (e.g. bad mmproj file).
    #[error("failed to create mtmd context (null return from mtmd_init_from_file)")]
    ContextCreateFailed,

    /// The bitmap could not be created.
    #[error("failed to create mtmd bitmap")]
    BitmapCreateFailed,

    /// A path could not be converted to a valid C string (embedded NUL byte or non-UTF-8).
    #[error("invalid path: {0}")]
    InvalidPath(#[from] std::ffi::NulError),

    /// A path was not representable as UTF-8.
    #[error("path is not valid UTF-8")]
    PathNotUtf8,

    /// `mtmd_tokenize` returned an error code.
    #[error("tokenize error: code {0} (1 = bitmap count mismatch, 2 = preprocessing error)")]
    TokenizeError(i32),

    /// `mtmd_encode_chunk` returned a non-zero code.
    #[error("encode error: code {0}")]
    EncodeError(i32),

    /// `mtmd_helper_eval_chunks` (or single-chunk variant) returned a non-zero code.
    #[error("eval error: code {0}")]
    EvalError(i32),
}

/// A convenience `Result` alias for this module.
pub type Result<T> = std::result::Result<T, MtmdError>;

// ─────────────────────────────────────────────────────────────────────────────
// MtmdContextParams
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters used when creating an [`MtmdContext`].
///
/// Obtain a default-initialised instance via [`MtmdContextParams::default()`].
pub struct MtmdContextParams {
    pub(crate) params: sys::mtmd_context_params,
}

impl std::fmt::Debug for MtmdContextParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtmdContextParams")
            .field("use_gpu", &self.params.use_gpu)
            .field("print_timings", &self.params.print_timings)
            .field("n_threads", &self.params.n_threads)
            .field("warmup", &self.params.warmup)
            .field("image_min_tokens", &self.params.image_min_tokens)
            .field("image_max_tokens", &self.params.image_max_tokens)
            .finish()
    }
}

impl Default for MtmdContextParams {
    fn default() -> Self {
        let params = unsafe { sys::mtmd_context_params_default() };
        Self { params }
    }
}

impl MtmdContextParams {
    /// Whether to run the vision/audio encoder on the GPU (default: `true`).
    #[must_use]
    pub fn use_gpu(mut self, v: bool) -> Self {
        self.params.use_gpu = v;
        self
    }

    /// Whether to print timing info after each encode (default: `false`).
    #[must_use]
    pub fn print_timings(mut self, v: bool) -> Self {
        self.params.print_timings = v;
        self
    }

    /// Number of threads used for the vision encoder (default taken from
    /// `mtmd_context_params_default`).
    #[must_use]
    pub fn n_threads(mut self, n: i32) -> Self {
        self.params.n_threads = n;
        self
    }

    /// Whether to run a warm-up encode pass after initialisation.
    #[must_use]
    pub fn warmup(mut self, v: bool) -> Self {
        self.params.warmup = v;
        self
    }

    /// Minimum number of image tokens (0 = use model default).
    #[must_use]
    pub fn image_min_tokens(mut self, n: i32) -> Self {
        self.params.image_min_tokens = n;
        self
    }

    /// Maximum number of image tokens (0 = use model default).
    #[must_use]
    pub fn image_max_tokens(mut self, n: i32) -> Self {
        self.params.image_max_tokens = n;
        self
    }

    /// Override the media marker string (e.g. `"<image>"`).
    ///
    /// The provided string must not contain interior NUL bytes.  Pass `None`
    /// to use the library default (`mtmd_default_marker()`).
    ///
    /// **Note:** the `CString` is stored inside the params so the pointer
    /// remains valid as long as this `MtmdContextParams` lives.
    pub fn media_marker(mut self, marker: Option<&str>) -> std::result::Result<Self, MtmdError> {
        match marker {
            None => {
                self.params.media_marker = std::ptr::null();
                Ok(self)
            }
            Some(s) => {
                let cs = CString::new(s)?;
                self.params.media_marker = cs.as_ptr();
                // Leak the CString so the raw pointer stays valid; the caller
                // must ensure the params don't outlive the string.  Since
                // MtmdContextParams is consumed by MtmdContext::init_from_file,
                // this is safe.
                std::mem::forget(cs);
                Ok(self)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MtmdContext
// ─────────────────────────────────────────────────────────────────────────────

/// The main multimodal context.
///
/// Wraps a `mtmd_context *`.  This context is tied to a specific mmproj model
/// file and a loaded [`LlamaModel`].  It is safe to share across threads for
/// `tokenize` calls (read-only), but `encode_chunk` / eval helpers mutate
/// internal state and must not be called concurrently.
pub struct MtmdContext {
    ptr: NonNull<sys::mtmd_context>,
}

// The underlying mtmd_context is internally synchronised for tokenize().
// encode / decode must be called from a single thread at a time (caller's
// responsibility, enforced by the inference semaphore in the server).
unsafe impl Send for MtmdContext {}
unsafe impl Sync for MtmdContext {}

impl std::fmt::Debug for MtmdContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtmdContext")
            .field("ptr", &self.ptr)
            .finish()
    }
}

impl Drop for MtmdContext {
    fn drop(&mut self) {
        unsafe { sys::mtmd_free(self.ptr.as_ptr()) }
    }
}

impl MtmdContext {
    /// Returns the default media marker string used in prompts
    /// (currently `"<__media__>"`).
    #[must_use]
    pub fn default_marker() -> &'static str {
        let ptr = unsafe { sys::mtmd_default_marker() };
        unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .unwrap_or("<__media__>")
    }

    /// Initialise a multimodal context from an mmproj GGUF file.
    ///
    /// # Parameters
    ///
    /// * `mmproj_path` – path to the mmproj `.gguf` file
    /// * `text_model`  – the already-loaded text model
    /// * `params`      – context parameters (use [`MtmdContextParams::default()`])
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::ContextCreateFailed`] if the underlying C call
    /// returns a null pointer.
    pub fn init_from_file(
        mmproj_path: impl AsRef<Path>,
        text_model: &LlamaModel,
        params: MtmdContextParams,
    ) -> Result<Self> {
        let path = mmproj_path
            .as_ref()
            .to_str()
            .ok_or(MtmdError::PathNotUtf8)?;
        let c_path = CString::new(path)?;

        let ptr = unsafe {
            sys::mtmd_init_from_file(
                c_path.as_ptr(),
                text_model.model.as_ptr(),
                params.params,
            )
        };

        let ptr = NonNull::new(ptr).ok_or(MtmdError::ContextCreateFailed)?;
        Ok(Self { ptr })
    }

    // ── Capability queries ────────────────────────────────────────────────

    /// Returns `true` if the model supports vision (image) input.
    #[must_use]
    pub fn supports_vision(&self) -> bool {
        unsafe { sys::mtmd_support_vision(self.ptr.as_ptr()) }
    }

    /// Returns `true` if the model supports audio input.
    #[must_use]
    pub fn supports_audio(&self) -> bool {
        unsafe { sys::mtmd_support_audio(self.ptr.as_ptr()) }
    }

    /// Returns the audio sample rate in Hz (e.g. 16 000 for Whisper), or
    /// `-1` if audio is not supported.
    #[must_use]
    pub fn audio_bitrate(&self) -> i32 {
        unsafe { sys::mtmd_get_audio_bitrate(self.ptr.as_ptr()) }
    }

    /// Whether `llama_decode` must use a non-causal attention mask when
    /// decoding image embeddings for this model.
    #[must_use]
    pub fn decode_use_non_causal(&self) -> bool {
        unsafe { sys::mtmd_decode_use_non_causal(self.ptr.as_ptr()) }
    }

    /// Whether the model uses M-RoPE for `llama_decode`.
    #[must_use]
    pub fn decode_use_mrope(&self) -> bool {
        unsafe { sys::mtmd_decode_use_mrope(self.ptr.as_ptr()) }
    }

    // ── Core API ──────────────────────────────────────────────────────────

    /// Tokenize a text prompt that contains one or more media markers.
    ///
    /// The number of `bitmaps` must equal the number of media markers in the
    /// prompt text, otherwise [`MtmdError::TokenizeError(1)`] is returned.
    ///
    /// This call is **thread-safe** (shared `&self`).
    ///
    /// # Parameters
    ///
    /// * `text`    – text + tokenisation options
    /// * `bitmaps` – slice of [`MtmdBitmap`] references, one per media marker
    /// * `output`  – an [`MtmdInputChunks`] that will be populated with the result
    pub fn tokenize(
        &self,
        text: &MtmdInputText<'_>,
        bitmaps: &[&MtmdBitmap],
        output: &mut MtmdInputChunks,
    ) -> Result<()> {
        // The C signature is: mtmd_tokenize(..., mtmd_bitmap ** bitmaps, ...)
        // where each element is a `const mtmd_bitmap *`.  We build a Vec of
        // `*const mtmd_bitmap` and pass a mutable pointer to its first element
        // (i.e. `*mut *const mtmd_bitmap`) to satisfy the C API.
        let mut bitmap_ptrs: Vec<*const sys::mtmd_bitmap> =
            bitmaps.iter().map(|b| b.ptr.as_ptr() as *const _).collect();

        let c_text = sys::mtmd_input_text {
            text: text.c_text.as_ptr(),
            add_special: text.add_special,
            parse_special: text.parse_special,
        };

        let ret = unsafe {
            sys::mtmd_tokenize(
                self.ptr.as_ptr(),
                output.ptr.as_ptr(),
                &c_text,
                bitmap_ptrs.as_mut_ptr(),
                bitmap_ptrs.len(),
            )
        };

        if ret != 0 {
            return Err(MtmdError::TokenizeError(ret));
        }
        Ok(())
    }

    /// Encode a single input chunk (image or audio) and store the resulting
    /// embeddings inside the context.
    ///
    /// After a successful call, the embeddings can be retrieved with
    /// [`MtmdContext::output_embd`].
    ///
    /// This call is **NOT thread-safe**.
    pub fn encode_chunk(&self, chunk: &MtmdInputChunk<'_>) -> Result<()> {
        let ret = unsafe { sys::mtmd_encode_chunk(self.ptr.as_ptr(), chunk.ptr) };
        if ret != 0 {
            return Err(MtmdError::EncodeError(ret));
        }
        Ok(())
    }

    /// Return a slice over the embeddings produced by the last
    /// [`encode_chunk`](Self::encode_chunk) call.
    ///
    /// The length (in `f32` elements) is:
    /// ```text
    /// n_embd_inp(model)  *  chunk.n_tokens()
    /// ```
    ///
    /// # Safety
    ///
    /// The returned slice is valid until the next call that mutates the
    /// context (e.g. another `encode_chunk`).
    #[must_use]
    pub fn output_embd(&self, n_elements: usize) -> &[f32] {
        let ptr = unsafe { sys::mtmd_get_output_embd(self.ptr.as_ptr()) };
        if ptr.is_null() || n_elements == 0 {
            return &[];
        }
        unsafe { slice::from_raw_parts(ptr, n_elements) }
    }

    // ── Helper API ────────────────────────────────────────────────────────

    /// High-level helper: evaluate (decode) all chunks in sequence.
    ///
    /// * Text chunks are decoded via `llama_decode`.
    /// * Image/audio chunks are first encoded with `mtmd_encode_chunk` and
    ///   then decoded via `llama_decode`.
    ///
    /// On success `new_n_past` is updated with the new past position.
    ///
    /// This call is **NOT thread-safe**.
    ///
    /// # Parameters
    ///
    /// * `lctx`        – raw pointer to the llama context (from [`LlamaContext::as_ptr`])
    /// * `chunks`      – the tokenized chunks to evaluate
    /// * `n_past`      – current KV-cache position
    /// * `seq_id`      – sequence ID
    /// * `n_batch`     – maximum batch size (must be ≥ 1)
    /// * `logits_last` – if `true`, compute logits only for the final token
    /// * `new_n_past`  – updated KV-cache position after the call
    #[allow(clippy::too_many_arguments)]
    pub fn eval_chunks(
        &self,
        lctx: *mut sys::llama_context,
        chunks: &MtmdInputChunks,
        n_past: i32,
        seq_id: i32,
        n_batch: i32,
        logits_last: bool,
        new_n_past: &mut i32,
    ) -> Result<()> {
        let ret = unsafe {
            sys::mtmd_helper_eval_chunks(
                self.ptr.as_ptr(),
                lctx,
                chunks.ptr.as_ptr(),
                n_past,
                seq_id,
                n_batch,
                logits_last,
                new_n_past,
            )
        };
        if ret != 0 {
            return Err(MtmdError::EvalError(ret));
        }
        Ok(())
    }

    /// High-level helper: evaluate a single chunk.
    ///
    /// Works identically to [`eval_chunks`](Self::eval_chunks) but operates on
    /// one chunk at a time.
    #[allow(clippy::too_many_arguments)]
    pub fn eval_chunk_single(
        &self,
        lctx: *mut sys::llama_context,
        chunk: &MtmdInputChunk<'_>,
        n_past: i32,
        seq_id: i32,
        n_batch: i32,
        logits_last: bool,
        new_n_past: &mut i32,
    ) -> Result<()> {
        let ret = unsafe {
            sys::mtmd_helper_eval_chunk_single(
                self.ptr.as_ptr(),
                lctx,
                chunk.ptr,
                n_past,
                seq_id,
                n_batch,
                logits_last,
                new_n_past,
            )
        };
        if ret != 0 {
            return Err(MtmdError::EvalError(ret));
        }
        Ok(())
    }

    /// Returns a raw pointer to the underlying `mtmd_context`.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for the lifetime of this `MtmdContext`.
    /// The caller must not free it.
    #[must_use]
    pub fn as_ptr(&self) -> *mut sys::mtmd_context {
        self.ptr.as_ptr()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MtmdInputText
// ─────────────────────────────────────────────────────────────────────────────

/// Text input for [`MtmdContext::tokenize`].
///
/// The prompt string must contain the media marker (see
/// [`MtmdContext::default_marker`]) once for every bitmap to be embedded.
#[derive(Debug)]
pub struct MtmdInputText<'a> {
    c_text: CString,
    add_special: bool,
    parse_special: bool,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> MtmdInputText<'a> {
    /// Create a new `MtmdInputText`.
    ///
    /// * `text`          – the prompt (must not contain interior NUL bytes)
    /// * `add_special`   – whether to add BOS/EOS tokens
    /// * `parse_special` – whether to parse special tokens embedded in the text
    ///
    /// # Panics
    ///
    /// Panics if `text` contains an interior NUL byte.
    #[must_use]
    pub fn new(text: &'a str, add_special: bool, parse_special: bool) -> Self {
        let c_text = CString::new(text).expect("MtmdInputText: text must not contain NUL bytes");
        Self {
            c_text,
            add_special,
            parse_special,
            _marker: std::marker::PhantomData,
        }
    }

    /// Try to create a new `MtmdInputText`, returning an error if `text`
    /// contains an interior NUL byte.
    pub fn try_new(
        text: &'a str,
        add_special: bool,
        parse_special: bool,
    ) -> std::result::Result<Self, std::ffi::NulError> {
        let c_text = CString::new(text)?;
        Ok(Self {
            c_text,
            add_special,
            parse_special,
            _marker: std::marker::PhantomData,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MtmdBitmap
// ─────────────────────────────────────────────────────────────────────────────

/// An image or audio bitmap ready for multimodal encoding.
///
/// # Image bitmaps
///
/// The raw pixel data must be in RGBRGBRGB… (interleaved) format.  The total
/// number of bytes must be `nx * ny * 3`.
///
/// # Audio bitmaps
///
/// The raw sample data must be little-endian `f32` PCM samples.  The total
/// number of bytes must be `n_samples * 4`.
pub struct MtmdBitmap {
    ptr: NonNull<sys::mtmd_bitmap>,
}

unsafe impl Send for MtmdBitmap {}
unsafe impl Sync for MtmdBitmap {}

impl std::fmt::Debug for MtmdBitmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtmdBitmap")
            .field("nx", &self.nx())
            .field("ny", &self.ny())
            .field("n_bytes", &self.n_bytes())
            .field("is_audio", &self.is_audio())
            .finish()
    }
}

impl Drop for MtmdBitmap {
    fn drop(&mut self) {
        unsafe { sys::mtmd_bitmap_free(self.ptr.as_ptr()) }
    }
}

impl MtmdBitmap {
    /// Create a bitmap from raw RGB pixel data.
    ///
    /// * `nx`   – image width in pixels
    /// * `ny`   – image height in pixels
    /// * `data` – raw pixel bytes in RGBRGB… format; must be `nx * ny * 3` bytes
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::BitmapCreateFailed`] if the underlying C call
    /// returns null.
    pub fn from_rgb(nx: u32, ny: u32, data: &[u8]) -> Result<Self> {
        let ptr = unsafe { sys::mtmd_bitmap_init(nx, ny, data.as_ptr()) };
        let ptr = NonNull::new(ptr).ok_or(MtmdError::BitmapCreateFailed)?;
        Ok(Self { ptr })
    }

    /// Create an audio bitmap from PCM `f32` samples.
    ///
    /// * `samples` – slice of PCM float samples
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::BitmapCreateFailed`] if the underlying C call
    /// returns null.
    pub fn from_audio(samples: &[f32]) -> Result<Self> {
        let ptr = unsafe { sys::mtmd_bitmap_init_from_audio(samples.len(), samples.as_ptr()) };
        let ptr = NonNull::new(ptr).ok_or(MtmdError::BitmapCreateFailed)?;
        Ok(Self { ptr })
    }

    /// Load a bitmap from a file (image or audio).
    ///
    /// Supported image formats: JPEG, PNG, BMP, GIF, and others handled by
    /// `stb_image`.  Supported audio formats: WAV, MP3, FLAC (via miniaudio).
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::BitmapCreateFailed`] if the file cannot be loaded.
    pub fn from_file(ctx: &MtmdContext, path: impl AsRef<Path>) -> Result<Self> {
        let path = path
            .as_ref()
            .to_str()
            .ok_or(MtmdError::PathNotUtf8)?;
        let c_path = CString::new(path)?;

        let ptr = unsafe {
            sys::mtmd_helper_bitmap_init_from_file(ctx.ptr.as_ptr(), c_path.as_ptr())
        };
        let ptr = NonNull::new(ptr).ok_or(MtmdError::BitmapCreateFailed)?;
        Ok(Self { ptr })
    }

    /// Load a bitmap from an in-memory buffer containing a file.
    ///
    /// The format is auto-detected (image vs audio via magic bytes).
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::BitmapCreateFailed`] if decoding fails.
    pub fn from_buf(ctx: &MtmdContext, buf: &[u8]) -> Result<Self> {
        let ptr = unsafe {
            sys::mtmd_helper_bitmap_init_from_buf(ctx.ptr.as_ptr(), buf.as_ptr(), buf.len())
        };
        let ptr = NonNull::new(ptr).ok_or(MtmdError::BitmapCreateFailed)?;
        Ok(Self { ptr })
    }

    // ── Getters ───────────────────────────────────────────────────────────

    /// Width in pixels (for images) or 0 (for audio).
    #[must_use]
    pub fn nx(&self) -> u32 {
        unsafe { sys::mtmd_bitmap_get_nx(self.ptr.as_ptr()) }
    }

    /// Height in pixels (for images) or 0 (for audio).
    #[must_use]
    pub fn ny(&self) -> u32 {
        unsafe { sys::mtmd_bitmap_get_ny(self.ptr.as_ptr()) }
    }

    /// Total number of bytes in the bitmap data.
    #[must_use]
    pub fn n_bytes(&self) -> usize {
        unsafe { sys::mtmd_bitmap_get_n_bytes(self.ptr.as_ptr()) }
    }

    /// Returns `true` if this bitmap contains audio (rather than image) data.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        unsafe { sys::mtmd_bitmap_is_audio(self.ptr.as_ptr()) }
    }

    /// Return the raw pixel / sample data.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        let n = self.n_bytes();
        if n == 0 {
            return &[];
        }
        let ptr = unsafe { sys::mtmd_bitmap_get_data(self.ptr.as_ptr()) };
        unsafe { slice::from_raw_parts(ptr, n) }
    }

    /// Return the optional ID string attached to this bitmap (used for KV
    /// cache tracking), or `None` if no ID has been set.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        let ptr = unsafe { sys::mtmd_bitmap_get_id(self.ptr.as_ptr()) };
        if ptr.is_null() {
            return None;
        }
        unsafe { CStr::from_ptr(ptr) }.to_str().ok()
    }

    /// Attach an optional ID string to this bitmap (used for KV cache
    /// tracking).
    ///
    /// # Errors
    ///
    /// Returns an error if `id` contains an interior NUL byte.
    pub fn set_id(&mut self, id: &str) -> std::result::Result<(), std::ffi::NulError> {
        let cs = CString::new(id)?;
        unsafe { sys::mtmd_bitmap_set_id(self.ptr.as_ptr(), cs.as_ptr()) };
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MtmdInputChunks
// ─────────────────────────────────────────────────────────────────────────────

/// A list of tokenized input chunks produced by [`MtmdContext::tokenize`].
///
/// Each chunk is either a text token sequence or a set of image/audio tokens.
pub struct MtmdInputChunks {
    ptr: NonNull<sys::mtmd_input_chunks>,
}

impl std::fmt::Debug for MtmdInputChunks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtmdInputChunks")
            .field("len", &self.len())
            .finish()
    }
}

impl Drop for MtmdInputChunks {
    fn drop(&mut self) {
        unsafe { sys::mtmd_input_chunks_free(self.ptr.as_ptr()) }
    }
}

impl MtmdInputChunks {
    /// Create a new, empty chunk list.  Populated by
    /// [`MtmdContext::tokenize`].
    ///
    /// # Panics
    ///
    /// Panics if the underlying C allocation fails (OOM).
    #[must_use]
    pub fn new() -> Self {
        let ptr = unsafe { sys::mtmd_input_chunks_init() };
        let ptr = NonNull::new(ptr).expect("mtmd_input_chunks_init returned null");
        Self { ptr }
    }

    /// Number of chunks in this list.
    #[must_use]
    pub fn len(&self) -> usize {
        unsafe { sys::mtmd_input_chunks_size(self.ptr.as_ptr()) }
    }

    /// Returns `true` if there are no chunks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the `idx`-th chunk.  Returns `None` if `idx >= len()`.
    #[must_use]
    pub fn get(&self, idx: usize) -> Option<MtmdInputChunk<'_>> {
        if idx >= self.len() {
            return None;
        }
        let ptr = unsafe { sys::mtmd_input_chunks_get(self.ptr.as_ptr(), idx) };
        if ptr.is_null() {
            return None;
        }
        Some(MtmdInputChunk {
            ptr,
            _marker: std::marker::PhantomData,
        })
    }

    /// Iterate over all chunks.
    pub fn iter(&self) -> impl Iterator<Item = MtmdInputChunk<'_>> {
        (0..self.len()).filter_map(|i| self.get(i))
    }

    /// Total number of tokens across all chunks.
    ///
    /// Equivalent to `mtmd_helper_get_n_tokens`.
    #[must_use]
    pub fn n_tokens(&self) -> usize {
        unsafe { sys::mtmd_helper_get_n_tokens(self.ptr.as_ptr()) }
    }

    /// Total number of *positions* across all chunks (used for KV-cache
    /// tracking with M-RoPE models where positions ≠ tokens).
    ///
    /// Equivalent to `mtmd_helper_get_n_pos`.
    #[must_use]
    pub fn n_pos(&self) -> i32 {
        unsafe { sys::mtmd_helper_get_n_pos(self.ptr.as_ptr()) }
    }
}

impl Default for MtmdInputChunks {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MtmdInputChunkType
// ─────────────────────────────────────────────────────────────────────────────

/// The type of an [`MtmdInputChunk`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtmdInputChunkType {
    /// Plain text tokens.
    Text,
    /// Image tokens (embeddings produced by the vision encoder).
    Image,
    /// Audio tokens (embeddings produced by the audio encoder).
    Audio,
}

impl From<sys::mtmd_input_chunk_type> for MtmdInputChunkType {
    fn from(v: sys::mtmd_input_chunk_type) -> Self {
        // mtmd_input_chunk_type is a plain C `typedef unsigned int`.
        // The variants are exported as free-standing constants.
        if v == sys::MTMD_INPUT_CHUNK_TYPE_IMAGE {
            Self::Image
        } else if v == sys::MTMD_INPUT_CHUNK_TYPE_AUDIO {
            Self::Audio
        } else {
            Self::Text
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MtmdInputChunk
// ─────────────────────────────────────────────────────────────────────────────

/// A single tokenized input chunk (text, image, or audio).
///
/// Instances are borrowed from an [`MtmdInputChunks`] list and live as long
/// as that list.
#[derive(Debug)]
pub struct MtmdInputChunk<'chunks> {
    ptr: *const sys::mtmd_input_chunk,
    _marker: std::marker::PhantomData<&'chunks MtmdInputChunks>,
}

impl<'chunks> MtmdInputChunk<'chunks> {
    /// The type of this chunk.
    #[must_use]
    pub fn chunk_type(&self) -> MtmdInputChunkType {
        let t = unsafe { sys::mtmd_input_chunk_get_type(self.ptr) };
        MtmdInputChunkType::from(t)
    }

    /// Total number of tokens in this chunk.
    #[must_use]
    pub fn n_tokens(&self) -> usize {
        unsafe { sys::mtmd_input_chunk_get_n_tokens(self.ptr) }
    }

    /// Number of temporal positions (equals `n_tokens` for non-M-RoPE models).
    #[must_use]
    pub fn n_pos(&self) -> i32 {
        unsafe { sys::mtmd_input_chunk_get_n_pos(self.ptr) }
    }

    /// Return the raw llama token IDs for a **text** chunk.
    ///
    /// Returns `None` if this chunk is not a text chunk.
    #[must_use]
    pub fn text_tokens(&self) -> Option<&[i32]> {
        if self.chunk_type() != MtmdInputChunkType::Text {
            return None;
        }
        let mut n: usize = 0;
        let ptr = unsafe { sys::mtmd_input_chunk_get_tokens_text(self.ptr, &mut n) };
        if ptr.is_null() || n == 0 {
            return Some(&[]);
        }
        Some(unsafe { slice::from_raw_parts(ptr, n) })
    }

    /// Return the image token metadata for an **image** or **audio** chunk.
    ///
    /// Returns `None` for text chunks.
    #[must_use]
    pub fn image_tokens(&self) -> Option<MtmdImageTokens<'chunks>> {
        match self.chunk_type() {
            MtmdInputChunkType::Image | MtmdInputChunkType::Audio => {}
            _ => return None,
        }
        let ptr = unsafe { sys::mtmd_input_chunk_get_tokens_image(self.ptr) };
        if ptr.is_null() {
            return None;
        }
        Some(MtmdImageTokens {
            ptr,
            _marker: std::marker::PhantomData,
        })
    }

    /// Optional ID attached to this chunk (used for KV cache tracking).
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        let ptr = unsafe { sys::mtmd_input_chunk_get_id(self.ptr) };
        if ptr.is_null() {
            return None;
        }
        unsafe { CStr::from_ptr(ptr) }.to_str().ok()
    }

    /// Returns the raw `*const mtmd_input_chunk` pointer.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for the lifetime of the parent
    /// `MtmdInputChunks`.
    #[must_use]
    pub fn as_ptr(&self) -> *const sys::mtmd_input_chunk {
        self.ptr
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MtmdImageTokens
// ─────────────────────────────────────────────────────────────────────────────

/// Image/audio token metadata attached to a non-text [`MtmdInputChunk`].
#[derive(Debug)]
pub struct MtmdImageTokens<'chunks> {
    ptr: *const sys::mtmd_image_tokens,
    _marker: std::marker::PhantomData<&'chunks MtmdInputChunks>,
}

impl<'chunks> MtmdImageTokens<'chunks> {
    /// Total number of embedding tokens.
    #[must_use]
    pub fn n_tokens(&self) -> usize {
        unsafe { sys::mtmd_image_tokens_get_n_tokens(self.ptr) }
    }

    /// Width of the token grid.
    #[must_use]
    pub fn nx(&self) -> usize {
        unsafe { sys::mtmd_image_tokens_get_nx(self.ptr) }
    }

    /// Height of the token grid.
    #[must_use]
    pub fn ny(&self) -> usize {
        unsafe { sys::mtmd_image_tokens_get_ny(self.ptr) }
    }

    /// Number of temporal positions (M-RoPE variant; equals `n_tokens` otherwise).
    #[must_use]
    pub fn n_pos(&self) -> i32 {
        unsafe { sys::mtmd_image_tokens_get_n_pos(self.ptr) }
    }

    /// Optional ID for KV cache tracking.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        let ptr = unsafe { sys::mtmd_image_tokens_get_id(self.ptr) };
        if ptr.is_null() {
            return None;
        }
        unsafe { CStr::from_ptr(ptr) }.to_str().ok()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LlamaContext extension
// ─────────────────────────────────────────────────────────────────────────────

use crate::context::LlamaContext;

impl LlamaContext<'_> {
    /// Expose the raw `llama_context` pointer for use with mtmd helpers.
    ///
    /// # Safety
    ///
    /// The pointer is valid for the lifetime of this `LlamaContext` and must
    /// not be freed by the caller.
    #[must_use]
    pub fn as_ptr(&self) -> *mut sys::llama_context {
        self.context.as_ptr()
    }
}
