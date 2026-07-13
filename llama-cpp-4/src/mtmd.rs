//! Safe wrappers for the `libmtmd` multimodal support library.
//!
//! `libmtmd` extends llama.cpp with the ability to encode image and audio
//! inputs (bitmaps) into token embeddings that can then be fed into a
//! standard [`crate::context::LlamaContext::decode`] call alongside normal text tokens.
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
use std::os::raw::c_void;
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

    /// A video stream could not be opened. Common causes: the build lacks
    /// video support (`MTMD_VIDEO` was OFF), `ffmpeg`/`ffprobe` is not on
    /// `PATH`, or the file is unreadable.
    #[error("failed to open video stream (null return from mtmd_helper_video_init)")]
    VideoInitFailed,

    /// `mtmd_helper_video_read_next` returned an error code (`-2`).
    #[error("video read error: code {0}")]
    VideoReadError(i32),
}

/// A convenience `Result` alias for this module.
pub type Result<T> = std::result::Result<T, MtmdError>;

/// Progress callback invoked while the CLIP/mmproj weights are loading.
///
/// Receives a value in `[0.0, 1.0]`. Return `true` to continue loading or
/// `false` to abort immediately.
pub type MtmdProgressCallback = unsafe extern "C" fn(progress: f32, user_data: *mut c_void) -> bool;

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

    /// Maximum number of multimodal output tokens per batch.
    ///
    /// Maps to `mtmd_context_params.batch_max_tokens`. The upstream default
    /// is `1024`. Increase for large images or long audio segments.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "mtmd")]
    /// # {
    /// use llama_cpp_4::mtmd::MtmdContextParams;
    /// let params = MtmdContextParams::default().with_batch_max_tokens(2048);
    /// assert_eq!(params.batch_max_tokens(), 2048);
    /// # }
    /// ```
    #[must_use]
    pub fn with_batch_max_tokens(mut self, n: i32) -> Self {
        self.params.batch_max_tokens = n;
        self
    }

    /// Get the configured batch token cap (`batch_max_tokens`).
    #[must_use]
    pub fn batch_max_tokens(&self) -> i32 {
        self.params.batch_max_tokens
    }

    /// Set flash-attention mode for the vision encoder.
    ///
    /// Maps to `mtmd_context_params.flash_attn_type`. Uses the same
    /// [`crate::context::params::LlamaFlashAttnType`] enum as text contexts.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "mtmd")]
    /// # {
    /// use llama_cpp_4::context::params::LlamaFlashAttnType;
    /// use llama_cpp_4::mtmd::MtmdContextParams;
    /// let params = MtmdContextParams::default()
    ///     .with_flash_attn_type(LlamaFlashAttnType::Auto);
    /// assert_eq!(params.flash_attn_type(), LlamaFlashAttnType::Auto);
    /// # }
    /// ```
    #[must_use]
    pub fn with_flash_attn_type(
        mut self,
        flash_attn_type: crate::context::params::LlamaFlashAttnType,
    ) -> Self {
        self.params.flash_attn_type = flash_attn_type.into();
        self
    }

    /// Get flash-attention mode for the vision encoder.
    #[must_use]
    pub fn flash_attn_type(&self) -> crate::context::params::LlamaFlashAttnType {
        crate::context::params::LlamaFlashAttnType::from(self.params.flash_attn_type)
    }

    /// Register a callback invoked while mmproj weights load.
    ///
    /// Maps to `mtmd_context_params.progress_callback`. Pass `None` to disable
    /// progress reporting. The callback may return `false` to abort loading
    /// early; see [`MtmdProgressCallback`].
    ///
    /// `user_data` is forwarded to each invocation and must remain valid until
    /// [`MtmdContext::init_from_file`] returns.
    #[must_use]
    pub fn with_progress_callback(
        mut self,
        callback: Option<MtmdProgressCallback>,
        user_data: *mut c_void,
    ) -> Self {
        self.params.progress_callback = callback;
        self.params.progress_callback_user_data = user_data;
        self
    }

    /// Override the media marker string (e.g. `"<image>"`).
    ///
    /// The provided string must not contain interior NUL bytes.  Pass `None`
    /// to use the library default (`mtmd_default_marker()`).
    ///
    /// **Note:** the `CString` is stored inside the params so the pointer
    /// remains valid as long as this `MtmdContextParams` lives.
    /// # Errors
    ///
    /// Returns [`MtmdError`] if the marker string contains a NUL byte.
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
    #[allow(clippy::needless_pass_by_value)]
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
            sys::mtmd_init_from_file(c_path.as_ptr(), text_model.model.as_ptr(), params.params)
        };

        let ptr = NonNull::new(ptr).ok_or(MtmdError::ContextCreateFailed)?;
        Ok(Self { ptr })
    }

    // ── Logging ──────────────────────────────────────────────────────────

    /// Silence all clip/mtmd log output by installing a no-op callback.
    ///
    /// Call this right after [`init_from_file`](Self::init_from_file) to
    /// suppress the verbose `clip_model_loader: tensor[N]…` lines that
    /// clip.cpp emits to its own private logger (separate from `llama_log_set`).
    pub fn void_logs() {
        unsafe extern "C" fn noop(
            _level: sys::ggml_log_level,
            _text: *const ::std::os::raw::c_char,
            _ud: *mut ::std::os::raw::c_void,
        ) {
        }
        unsafe { sys::mtmd_log_set(Some(noop), std::ptr::null_mut()) };
    }

    /// Like [`void_logs`](Self::void_logs), but additionally silences logs
    /// emitted by the `mtmd_helper_*` layer (e.g. eval/decode helpers).
    ///
    /// Internally calls `mtmd_helper_log_set` which also routes through
    /// `mtmd_log_set`, so this is a strict superset of `void_logs`.
    pub fn void_helper_logs() {
        unsafe extern "C" fn noop(
            _level: sys::ggml_log_level,
            _text: *const ::std::os::raw::c_char,
            _ud: *mut ::std::os::raw::c_void,
        ) {
        }
        unsafe { sys::mtmd_helper_log_set(Some(noop), std::ptr::null_mut()) };
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

    /// Returns `true` if this build and model support video input.
    ///
    /// Video support additionally requires `ffmpeg`/`ffprobe` to be available
    /// at runtime (see [`MtmdVideo`]). Wraps `mtmd_helper_support_video`.
    #[must_use]
    pub fn supports_video(&self) -> bool {
        unsafe { sys::mtmd_helper_support_video(self.ptr.as_ptr()) }
    }

    /// Returns the media marker string configured for *this* context.
    ///
    /// Unlike [`default_marker`](Self::default_marker) (the library-wide
    /// default), this reflects any override passed via
    /// [`MtmdContextParams::media_marker`]. Wraps `mtmd_get_marker`.
    #[must_use]
    pub fn marker(&self) -> &str {
        let ptr = unsafe { sys::mtmd_get_marker(self.ptr.as_ptr()) };
        if ptr.is_null() {
            return Self::default_marker();
        }
        unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .unwrap_or_else(|_| Self::default_marker())
    }

    /// Returns the audio sample rate in Hz (e.g. `16_000` for Whisper), or `-1` if
    /// audio is not supported.
    #[must_use]
    pub fn audio_sample_rate(&self) -> i32 {
        unsafe { sys::mtmd_get_audio_sample_rate(self.ptr.as_ptr()) }
    }

    /// Whether `llama_decode` must use a non-causal attention mask when
    /// decoding image embeddings for this model.
    #[must_use]
    pub fn decode_use_non_causal(&self, chunk: &MtmdInputChunk<'_>) -> bool {
        unsafe { sys::mtmd_decode_use_non_causal(self.ptr.as_ptr(), chunk.as_ptr()) }
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
    /// prompt text, otherwise [`MtmdError::TokenizeError`] with code `1` is returned.
    ///
    /// This call is **thread-safe** (shared `&self`).
    ///
    /// # Parameters
    ///
    /// * `text`    – text + tokenisation options
    /// * `bitmaps` – slice of [`MtmdBitmap`] references, one per media marker
    /// * `output`  – an [`MtmdInputChunks`] that will be populated with the result
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::TokenizeError`] if tokenization fails.
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
        let mut bitmap_ptrs: Vec<*const sys::mtmd_bitmap> = bitmaps
            .iter()
            .map(|b| b.ptr.as_ptr().cast_const())
            .collect();

        let c_text = sys::mtmd_input_text {
            // Upstream reads exactly `text_len` bytes from `text`
            // (llama.cpp #25548), so the prompt is length-delimited and interior
            // NUL bytes are preserved instead of truncating it.
            text: text.text.as_ptr().cast(),
            text_len: text.text_len,
            add_special: text.add_special,
            parse_special: text.parse_special,
        };

        let ret = unsafe {
            sys::mtmd_tokenize(
                self.ptr.as_ptr(),
                output.ptr.as_ptr(),
                &raw const c_text,
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
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::EncodeError`] if encoding fails.
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
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::EvalError`] if evaluation fails.
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
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
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::EvalError`] if evaluation fails.
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
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

    /// Decode an image/audio chunk whose embeddings have already been
    /// computed (e.g. via [`encode_chunk`](Self::encode_chunk) followed by
    /// [`output_embd`](Self::output_embd)).
    ///
    /// Unlike [`eval_chunk_single`](Self::eval_chunk_single), this helper
    /// handles batching plus the non-causal-attention setup required by
    /// some models (e.g. Gemma 3, Gemma 4 audio) and the M-RoPE position
    /// layout. Use it when the embeddings are already in hand and you want
    /// the helper to take care of `llama_decode` plumbing.
    ///
    /// `encoded_embd` must contain `mtmd_image_tokens_get_n_tokens(chunk) *
    /// llama_model_n_embd_inp(model)` `f32` elements. This call is **NOT
    /// thread-safe**.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::EvalError`] with code `-1` if `chunk` is not an
    /// image/audio chunk, or `1` if `llama_decode` fails.
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn decode_image_chunk(
        &self,
        lctx: *mut sys::llama_context,
        chunk: &MtmdInputChunk<'_>,
        encoded_embd: &[f32],
        n_past: i32,
        seq_id: i32,
        n_batch: i32,
        new_n_past: &mut i32,
    ) -> Result<()> {
        let ret = unsafe {
            sys::mtmd_helper_decode_image_chunk(
                self.ptr.as_ptr(),
                lctx,
                chunk.ptr,
                encoded_embd.as_ptr().cast_mut(),
                n_past,
                seq_id,
                n_batch,
                new_n_past,
                // No post-decode callback; preserves prior single-shot behavior.
                None,
                std::ptr::null_mut(),
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
///
/// The prompt is passed to llama.cpp as an explicit pointer + length
/// (`mtmd_input_text::text_len`), so interior NUL bytes are preserved rather
/// than truncating the prompt — use [`MtmdInputText::from_bytes`] when the
/// prompt is not guaranteed NUL-free.
#[derive(Debug)]
pub struct MtmdInputText<'a> {
    /// Prompt bytes followed by a trailing NUL sentinel. The sentinel keeps the
    /// buffer usable by any C code that still treats `text` as a C string; it is
    /// excluded from `text_len`.
    text: Vec<u8>,
    /// Prompt length in bytes, excluding the trailing NUL sentinel. Passed
    /// verbatim as `mtmd_input_text::text_len`, so interior NULs are honoured.
    text_len: usize,
    add_special: bool,
    parse_special: bool,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> MtmdInputText<'a> {
    /// Create a new `MtmdInputText` from a string prompt.
    ///
    /// * `text`          – the prompt (interior NUL bytes are permitted and
    ///   preserved)
    /// * `add_special`   – whether to add BOS/EOS tokens
    /// * `parse_special` – whether to parse special tokens embedded in the text
    #[must_use]
    pub fn new(text: &'a str, add_special: bool, parse_special: bool) -> Self {
        Self::from_bytes(text.as_bytes(), add_special, parse_special)
    }

    /// Create a new `MtmdInputText` from raw prompt bytes.
    ///
    /// Unlike a C string, the prompt length is carried explicitly, so `text`
    /// may contain interior NUL bytes without truncating the prompt. The bytes
    /// are copied into an owned, NUL-terminated buffer.
    ///
    /// * `text`          – the prompt bytes (typically UTF-8)
    /// * `add_special`   – whether to add BOS/EOS tokens
    /// * `parse_special` – whether to parse special tokens embedded in the text
    #[must_use]
    pub fn from_bytes(text: &'a [u8], add_special: bool, parse_special: bool) -> Self {
        let text_len = text.len();
        let mut buf = Vec::with_capacity(text_len + 1);
        buf.extend_from_slice(text);
        buf.push(0); // NUL sentinel, not counted in `text_len`
        Self {
            text: buf,
            text_len,
            add_special,
            parse_special,
            _marker: std::marker::PhantomData,
        }
    }

    /// Try to create a new `MtmdInputText` from a string prompt.
    ///
    /// Retained for backwards compatibility. Interior NUL bytes are now
    /// permitted (see [`MtmdInputText::new`]), so this never returns `Err`;
    /// prefer [`new`](MtmdInputText::new).
    ///
    /// # Errors
    ///
    /// Never returns an error; the `Result` is kept for API stability.
    pub fn try_new(
        text: &'a str,
        add_special: bool,
        parse_special: bool,
    ) -> std::result::Result<Self, std::ffi::NulError> {
        Ok(Self::new(text, add_special, parse_special))
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

    /// Build an `MtmdBitmap` from a `mtmd_helper_bitmap_wrapper`, taking
    /// ownership of the `bitmap` and freeing any `video_ctx`.
    ///
    /// The `from_file`/`from_buf` constructors only support image/audio input.
    /// When the input is a video the helper returns a non-null `video_ctx`
    /// (an open ffmpeg stream) which is not representable as an `MtmdBitmap`;
    /// we free it here to avoid leaking it. Use [`MtmdVideo`] for video input.
    fn from_wrapper(wrapper: sys::mtmd_helper_bitmap_wrapper) -> Result<Self> {
        if !wrapper.video_ctx.is_null() {
            unsafe { sys::mtmd_helper_video_free(wrapper.video_ctx) };
        }
        let ptr = NonNull::new(wrapper.bitmap).ok_or(MtmdError::BitmapCreateFailed)?;
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
        let path = path.as_ref().to_str().ok_or(MtmdError::PathNotUtf8)?;
        let c_path = CString::new(path)?;

        // `placeholder = false`: load the real bitmap data (not a token-count
        // placeholder). For image/audio the returned `video_ctx` is always null.
        let wrapper = unsafe {
            sys::mtmd_helper_bitmap_init_from_file(ctx.ptr.as_ptr(), c_path.as_ptr(), false)
        };
        Self::from_wrapper(wrapper)
    }

    /// Load a bitmap from an in-memory buffer containing a file.
    ///
    /// The format is auto-detected (image vs audio via magic bytes).
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::BitmapCreateFailed`] if decoding fails.
    pub fn from_buf(ctx: &MtmdContext, buf: &[u8]) -> Result<Self> {
        // `placeholder = false`: load the real bitmap data (not a token-count
        // placeholder). For image/audio the returned `video_ctx` is always null.
        let wrapper = unsafe {
            sys::mtmd_helper_bitmap_init_from_buf(ctx.ptr.as_ptr(), buf.as_ptr(), buf.len(), false)
        };
        Self::from_wrapper(wrapper)
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
// Video input
// ─────────────────────────────────────────────────────────────────────────────

// `free()` from libc — used to release the heap-allocated text returned by
// `mtmd_helper_video_read_next` (the C side allocates it with strdup/malloc and
// documents that the caller must release it with `free()`).
extern "C" {
    fn free(ptr: *mut std::os::raw::c_void);
}

/// Parameters controlling how a [`MtmdVideo`] stream is opened and sampled.
///
/// Obtain a default-initialised instance via [`MtmdVideoParams::default()`]
/// (which mirrors `mtmd_helper_video_init_params_default`: ~4 fps, native
/// `ffmpeg`/`ffprobe` from `PATH`, and a 5 s timestamp interval) and tweak it
/// with the builder methods.
pub struct MtmdVideoParams {
    params: sys::mtmd_helper_video_init_params,
    // Keeps the `ffmpeg_bin_dir` C string alive for as long as `params`
    // borrows it via a raw pointer.
    ffmpeg_bin_dir: Option<CString>,
}

impl std::fmt::Debug for MtmdVideoParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtmdVideoParams")
            .field("fps_target", &self.params.fps_target)
            .field("timestamp_interval_ms", &self.params.timestamp_interval_ms)
            .field("ffmpeg_bin_dir", &self.ffmpeg_bin_dir)
            .finish()
    }
}

impl Default for MtmdVideoParams {
    fn default() -> Self {
        let params = unsafe { sys::mtmd_helper_video_init_params_default() };
        Self {
            params,
            ffmpeg_bin_dir: None,
        }
    }
}

impl MtmdVideoParams {
    /// Desired output frame rate. Values `<= 0` mean "use the video's native
    /// fps" (the default is ~4 fps).
    #[must_use]
    pub fn fps_target(mut self, fps: f32) -> Self {
        self.params.fps_target = fps;
        self
    }

    /// Interval, in milliseconds, between inserted timestamp text chunks (e.g.
    /// `"[10m50.5s]"`). Values `<= 0` disable timestamps (default 5000 ms).
    #[must_use]
    pub fn timestamp_interval_ms(mut self, ms: i64) -> Self {
        self.params.timestamp_interval_ms = ms;
        self
    }

    /// Directory containing the `ffmpeg`/`ffprobe` binaries. Pass `None` to
    /// search `PATH` (the default).
    ///
    /// # Errors
    ///
    /// Returns an error if `dir` contains an interior NUL byte.
    pub fn ffmpeg_bin_dir(mut self, dir: Option<&str>) -> Result<Self> {
        match dir {
            None => {
                self.params.ffmpeg_bin_dir = std::ptr::null();
                self.ffmpeg_bin_dir = None;
            }
            Some(d) => {
                let cs = CString::new(d)?;
                self.params.ffmpeg_bin_dir = cs.as_ptr();
                // Store the owner so the pointer above stays valid.
                self.ffmpeg_bin_dir = Some(cs);
            }
        }
        Ok(self)
    }
}

/// Metadata describing an open [`MtmdVideo`] stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MtmdVideoInfo {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Effective frames-per-second (the `fps_target` if set, else native fps).
    pub fps: f32,
    /// Estimated total frame count at the effective fps (`-1` if unknown).
    pub n_frames: i32,
}

/// One item read from a [`MtmdVideo`] stream by [`MtmdVideo::read_next`].
#[derive(Debug)]
pub enum MtmdVideoItem {
    /// A decoded video frame, ready to be tokenized like any other image
    /// [`MtmdBitmap`].
    Frame(MtmdBitmap),
    /// A timestamp text marker (e.g. `"[10m50.5s]"`) to be inserted into the
    /// prompt between frames.
    Text(String),
}

/// An open video stream, decoded frame-by-frame via `ffmpeg`.
///
/// The notion of "video" exists only at the helper level — it is decoded into
/// a sequence of image [frames](MtmdVideoItem::Frame) and timestamp
/// [text markers](MtmdVideoItem::Text) which are then fed through the normal
/// multimodal pipeline.
///
/// Requires a build with video support (see [`MtmdContext::supports_video`])
/// and `ffmpeg`/`ffprobe` available at runtime.
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "mtmd")]
/// # fn run(mtmd_ctx: &llama_cpp_4::mtmd::MtmdContext) -> Result<(), llama_cpp_4::mtmd::MtmdError> {
/// use std::path::Path;
/// use llama_cpp_4::mtmd::{MtmdVideo, MtmdVideoParams, MtmdVideoItem};
///
/// let mut video = MtmdVideo::from_file(mtmd_ctx, Path::new("clip.mp4"),
///                                      &MtmdVideoParams::default())?;
/// while let Some(item) = video.read_next()? {
///     match item {
///         MtmdVideoItem::Frame(bitmap) => { /* tokenize the frame */ }
///         MtmdVideoItem::Text(ts)      => { /* insert the timestamp marker */ }
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub struct MtmdVideo {
    ptr: NonNull<sys::mtmd_helper_video>,
}

impl std::fmt::Debug for MtmdVideo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtmdVideo")
            .field("info", &self.info())
            .finish()
    }
}

impl Drop for MtmdVideo {
    fn drop(&mut self) {
        unsafe { sys::mtmd_helper_video_free(self.ptr.as_ptr()) }
    }
}

impl MtmdVideo {
    /// Open a video file for frame-by-frame decoding.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::VideoInitFailed`] if the stream cannot be opened
    /// (no video support compiled in, `ffprobe` not found, file unreadable,
    /// …), or [`MtmdError::InvalidPath`] / [`MtmdError::PathNotUtf8`] for a bad
    /// path.
    pub fn from_file(
        ctx: &MtmdContext,
        path: impl AsRef<Path>,
        params: &MtmdVideoParams,
    ) -> Result<Self> {
        let path = path.as_ref().to_str().ok_or(MtmdError::PathNotUtf8)?;
        let c_path = CString::new(path)?;
        let ptr = unsafe {
            sys::mtmd_helper_video_init(ctx.ptr.as_ptr(), c_path.as_ptr(), params.params)
        };
        let ptr = NonNull::new(ptr).ok_or(MtmdError::VideoInitFailed)?;
        Ok(Self { ptr })
    }

    /// Open a video from an in-memory buffer. The buffer is copied internally,
    /// so it need not outlive this call.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::VideoInitFailed`] if the stream cannot be opened.
    pub fn from_buf(ctx: &MtmdContext, buf: &[u8], params: &MtmdVideoParams) -> Result<Self> {
        let ptr = unsafe {
            sys::mtmd_helper_video_init_from_buf(
                ctx.ptr.as_ptr(),
                buf.as_ptr(),
                buf.len(),
                params.params,
            )
        };
        let ptr = NonNull::new(ptr).ok_or(MtmdError::VideoInitFailed)?;
        Ok(Self { ptr })
    }

    /// Return metadata (resolution, effective fps, estimated frame count) for
    /// this stream.
    #[must_use]
    pub fn info(&self) -> MtmdVideoInfo {
        let info = unsafe { sys::mtmd_helper_video_get_info(self.ptr.as_ptr()) };
        MtmdVideoInfo {
            width: info.width,
            height: info.height,
            fps: info.fps,
            n_frames: info.n_frames,
        }
    }

    /// Read the next item from the stream.
    ///
    /// Returns `Ok(Some(item))` for each frame or timestamp marker, and
    /// `Ok(None)` once the end of the stream is reached.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::VideoReadError`] on a decode error.
    pub fn read_next(&mut self) -> Result<Option<MtmdVideoItem>> {
        let mut out_bitmap: *mut sys::mtmd_bitmap = std::ptr::null_mut();
        let mut out_text: *mut std::os::raw::c_char = std::ptr::null_mut();
        let ret = unsafe {
            sys::mtmd_helper_video_read_next(
                self.ptr.as_ptr(),
                &raw mut out_bitmap,
                &raw mut out_text,
            )
        };
        match ret {
            0 => {
                if let Some(ptr) = NonNull::new(out_bitmap) {
                    Ok(Some(MtmdVideoItem::Frame(MtmdBitmap { ptr })))
                } else if !out_text.is_null() {
                    let text = unsafe { CStr::from_ptr(out_text) }
                        .to_string_lossy()
                        .into_owned();
                    // The C side allocated this with strdup/malloc; release it.
                    unsafe { free(out_text.cast()) };
                    Ok(Some(MtmdVideoItem::Text(text)))
                } else {
                    // Success but nothing produced — treat as end of stream.
                    Ok(None)
                }
            }
            -1 => Ok(None), // EOF
            other => Err(MtmdError::VideoReadError(other)),
        }
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
        let ptr = unsafe { sys::mtmd_input_chunk_get_tokens_text(self.ptr, &raw mut n) };
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
            MtmdInputChunkType::Text => return None,
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
// MtmdDecoderPos
// ─────────────────────────────────────────────────────────────────────────────

/// Per-token position used by M-RoPE decoder attention.
///
/// `t` is the temporal axis, `x`/`y` the spatial axes. `z` is reserved for
/// future use. Values are *relative* to a base `pos_0` provided when the
/// position is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct MtmdDecoderPos {
    /// Temporal index.
    pub t: u32,
    /// Spatial X.
    pub x: u32,
    /// Spatial Y.
    pub y: u32,
    /// Reserved.
    pub z: u32,
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

impl MtmdImageTokens<'_> {
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

    /// Compute the per-token decoder positions used by M-RoPE models.
    ///
    /// Returns a vector of length [`n_tokens`](Self::n_tokens). Each entry
    /// is relative to `pos_0`; for non-M-RoPE models this typically reduces
    /// to `(0, i, 0, 0)` for the i-th token.
    ///
    /// Wraps `mtmd_helper_image_get_decoder_pos`.
    #[must_use]
    pub fn decoder_positions(&self, pos_0: i32) -> Vec<MtmdDecoderPos> {
        let n = self.n_tokens();
        let mut out = vec![MtmdDecoderPos::default(); n];
        if n == 0 {
            return out;
        }
        unsafe {
            sys::mtmd_helper_image_get_decoder_pos(
                self.ptr,
                pos_0,
                out.as_mut_ptr().cast::<sys::mtmd_decoder_pos>(),
            );
        }
        out
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_pos_layout_matches_sys() {
        // The Rust MtmdDecoderPos is cast to sys::mtmd_decoder_pos at the
        // FFI boundary in `MtmdImageTokens::decoder_positions`. Verify the
        // assumption.
        assert_eq!(
            std::mem::size_of::<MtmdDecoderPos>(),
            std::mem::size_of::<sys::mtmd_decoder_pos>(),
        );
        assert_eq!(
            std::mem::align_of::<MtmdDecoderPos>(),
            std::mem::align_of::<sys::mtmd_decoder_pos>(),
        );
        assert_eq!(std::mem::offset_of!(MtmdDecoderPos, t), 0);
        assert_eq!(std::mem::offset_of!(MtmdDecoderPos, x), 4);
        assert_eq!(std::mem::offset_of!(MtmdDecoderPos, y), 8);
        assert_eq!(std::mem::offset_of!(MtmdDecoderPos, z), 12);
    }

    #[test]
    fn input_text_records_byte_length_and_nul_terminates() {
        let input = MtmdInputText::new("hello", true, false);
        // text_len is the prompt length, excluding the trailing NUL sentinel.
        assert_eq!(input.text_len, 5);
        assert_eq!(input.text, b"hello\0");
        assert!(input.add_special);
        assert!(!input.parse_special);
    }

    #[test]
    fn input_text_preserves_interior_nul() {
        // The whole point of upstream's `text_len`: a prompt with an embedded
        // NUL must keep its full length rather than truncating at the NUL.
        let input = MtmdInputText::from_bytes(b"a\0b", false, true);
        assert_eq!(input.text_len, 3);
        assert_eq!(input.text, b"a\0b\0");
    }

    #[test]
    fn input_text_try_new_is_infallible() {
        let input = MtmdInputText::try_new("marker \u{1} data", true, true)
            .expect("try_new no longer rejects any input");
        assert_eq!(input.text_len, "marker \u{1} data".len());
    }
}
