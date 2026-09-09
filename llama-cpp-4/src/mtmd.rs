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

    /// `mtmd_input_chunk_save` returned a non-zero code.
    #[error("chunk save error: code {0}")]
    ChunkSaveFailed(i32),

    /// `mtmd_input_chunk_load` returned null — the buffer was not a chunk
    /// this build can restore.
    #[error("failed to load an input chunk from the buffer")]
    ChunkLoadFailed,

    /// A chunk could not be added to a batch.
    #[error("batch add error: code {0} (2 = batch full, 3 = incompatible with existing chunks)")]
    BatchAddFailed(i32),

    /// `mtmd_batch_init` returned null.
    #[error("failed to create an mtmd batch")]
    BatchCreateFailed,

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

        // Length-delimited (llama.cpp #25548), so interior NULs are preserved.
        let c_text = text.as_raw();

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

    /// Tokenize an explicit sequence of parts, without media markers.
    ///
    /// [`Self::tokenize`] splices bitmaps in wherever the prompt contains the
    /// media marker, which means the marker string has to be embedded in the
    /// text and cannot itself be user content. This takes the interleaving
    /// directly, so:
    ///
    /// - a marker appearing in user text is just text, not a splice point;
    /// - `parse_special` is per text part, so a system prompt can enable
    ///   special tokens while user content does not.
    ///
    /// `add_special` applies once to the whole sequence — upstream ignores the
    /// per-part flag.
    ///
    /// Wraps `mtmd_tokenize_from_parts`.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::TokenizeError`] — code `1` means a part carried
    /// both text and a bitmap, or neither.
    pub fn tokenize_from_parts(
        &self,
        parts: &[MtmdInputPart<'_>],
        add_special: bool,
        output: &mut MtmdInputChunks,
    ) -> Result<()> {
        // Three levels have to stay alive across the call: the C text structs,
        // the parts that point at them, and the array of pointers to those
        // parts. Building them in that order keeps every borrow valid.
        let raw_texts: Vec<sys::mtmd_input_text> = parts
            .iter()
            .filter_map(|part| match part {
                MtmdInputPart::Text(text) => Some(text.as_raw()),
                MtmdInputPart::Bitmap(_) => None,
            })
            .collect();

        let mut next_text = 0usize;
        let raw_parts: Vec<sys::mtmd_input_part> = parts
            .iter()
            .map(|part| match part {
                MtmdInputPart::Text(_) => {
                    let raw = &raw_texts[next_text];
                    next_text += 1;
                    sys::mtmd_input_part {
                        text: std::ptr::from_ref(raw),
                        bitmap: std::ptr::null(),
                    }
                }
                MtmdInputPart::Bitmap(bitmap) => sys::mtmd_input_part {
                    text: std::ptr::null(),
                    bitmap: bitmap.ptr.as_ptr().cast_const(),
                },
            })
            .collect();
        let part_ptrs: Vec<*const sys::mtmd_input_part> =
            raw_parts.iter().map(std::ptr::from_ref).collect();

        let ret = unsafe {
            sys::mtmd_tokenize_from_parts(
                self.ptr.as_ptr(),
                output.ptr.as_ptr(),
                part_ptrs.as_ptr(),
                part_ptrs.len(),
                add_special,
            )
        };
        if ret != 0 {
            return Err(MtmdError::TokenizeError(ret));
        }
        Ok(())
    }

    /// Audio-generation capabilities of the loaded mmproj.
    ///
    /// Returns `None` when this projector cannot generate audio, which is the
    /// case for every vision-only mmproj.
    ///
    /// Wraps `mtmd_gen_audio_get_info`.
    #[must_use]
    pub fn gen_audio_info(&self) -> Option<MtmdGenAudioInfo> {
        let info = unsafe { sys::mtmd_gen_audio_get_info(self.ptr.as_ptr()) };
        if info.type_ == sys::MTMD_GEN_AUDIO_TYPE_NONE {
            return None;
        }
        let variant = if info.model_variant.is_null() {
            None
        } else {
            Some(
                unsafe { CStr::from_ptr(info.model_variant) }
                    .to_string_lossy()
                    .into_owned(),
            )
        };
        Some(MtmdGenAudioInfo {
            pipeline: MtmdGenAudioType::from_raw(info.type_),
            sample_rate: info.sample_rate,
            model_variant: variant,
        })
    }

    /// Whether this model and projector can be used for chat.
    ///
    /// Wraps `mtmd_helper_model_can_chat`.
    #[must_use]
    pub fn model_can_chat(&self, ctx: &crate::context::LlamaContext<'_>) -> bool {
        unsafe { sys::mtmd_helper_model_can_chat(ctx.context.as_ptr(), self.ptr.as_ptr()) }
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
    /// Borrow this as the C struct.
    ///
    /// The result points into `self`, so it must not outlive it.
    pub(crate) fn as_raw(&self) -> sys::mtmd_input_text {
        sys::mtmd_input_text {
            // Upstream reads exactly `text_len` bytes, so interior NULs survive.
            text: self.text.as_ptr().cast(),
            text_len: self.text_len,
            add_special: self.add_special,
            parse_special: self.parse_special,
        }
    }

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
            sys::mtmd_helper_bitmap_init_from_file(
                ctx.ptr.as_ptr(),
                c_path.as_ptr(),
                false,
                sys::mtmd_helper_init_opt_default(),
            )
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
            sys::mtmd_helper_bitmap_init_from_buf(
                ctx.ptr.as_ptr(),
                buf.as_ptr(),
                buf.len(),
                false,
                sys::mtmd_helper_init_opt_default(),
            )
        };
        Self::from_wrapper(wrapper)
    }

    /// Mark this bitmap as mergeable with an adjacent mergeable bitmap.
    ///
    /// Video-capable models such as Qwen-VL merge consecutive frames into one
    /// chunk (a temporal merge). [`MtmdVideo::read_next`] already sets this on
    /// the frames it produces; you only need it when you build frames yourself
    /// with [`Self::from_rgb`] and expect them to merge. Without it each frame
    /// becomes its own chunk, which costs tokens and loses temporal structure.
    ///
    /// Wraps `mtmd_bitmap_set_mergeable`.
    pub fn set_mergeable(&mut self, mergeable: bool) {
        unsafe { sys::mtmd_bitmap_set_mergeable(self.ptr.as_ptr(), mergeable) }
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
    /// `strdup` from libc. Used for the text a lazy-bitmap callback yields:
    /// mtmd releases it with `free()`, which Rust's allocator is not
    /// compatible with, so the copy has to come from malloc.
    fn strdup(s: *const std::os::raw::c_char) -> *mut std::os::raw::c_char;
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

    /// Restore a chunk previously serialized with
    /// [`MtmdInputChunk::save`], returning it as an owned placeholder.
    ///
    /// The result carries only metadata — token count, position count, type —
    /// so it can line a restored KV cache up with the prompt that produced it.
    /// It cannot be re-encoded; the pixels are gone.
    ///
    /// Wraps `mtmd_input_chunk_load`.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::ChunkLoadFailed`] if the buffer is not a chunk
    /// this build can restore.
    pub fn load_chunk(buf: &[u8]) -> Result<OwnedMtmdInputChunk> {
        let ptr = unsafe {
            sys::mtmd_input_chunk_load(buf.as_ptr().cast::<std::os::raw::c_char>(), buf.len())
        };
        NonNull::new(ptr)
            .map(|ptr| OwnedMtmdInputChunk { ptr })
            .ok_or(MtmdError::ChunkLoadFailed)
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

    /// Serialize this chunk's metadata to a byte buffer.
    ///
    /// Only metadata is saved — never the image or audio payload. A chunk
    /// restored with [`MtmdInputChunks::load_chunk`] is a *placeholder*: it
    /// carries the token and position counts needed to line a cached KV state
    /// back up with its prompt, but cannot be re-encoded. That is the intended
    /// use, and it is why this is cheap enough to store alongside a session
    /// file written by
    /// [`state_seq_save_file`](crate::context::LlamaContext::state_seq_save_file).
    ///
    /// Wraps `mtmd_input_chunk_save`.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::ChunkSaveFailed`] if llama.cpp cannot serialize
    /// this chunk.
    pub fn save(&self) -> Result<Vec<u8>> {
        // Two-call protocol: query the length, then fill.
        let mut needed: usize = 0;
        let rc = unsafe {
            sys::mtmd_input_chunk_save(self.ptr, std::ptr::null_mut(), 0, &raw mut needed)
        };
        if rc != 0 && needed == 0 {
            return Err(MtmdError::ChunkSaveFailed(rc));
        }
        let mut buf = vec![0u8; needed];
        let rc = unsafe {
            sys::mtmd_input_chunk_save(
                self.ptr,
                buf.as_mut_ptr().cast::<std::os::raw::c_char>(),
                buf.len(),
                &raw mut needed,
            )
        };
        if rc != 0 {
            return Err(MtmdError::ChunkSaveFailed(rc));
        }
        buf.truncate(needed);
        Ok(buf)
    }

    /// Copy this chunk, payload and all, into an owned handle.
    ///
    /// [`MtmdInputChunk`] borrows from the [`MtmdInputChunks`] list holding it,
    /// so it dies with that list. Take a copy when a chunk has to outlive the
    /// tokenization it came from — caching encoded media across requests, for
    /// instance. Unlike [`Self::to_placeholder`], the result is still usable
    /// for encoding.
    ///
    /// Wraps `mtmd_input_chunk_copy`.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::ChunkLoadFailed`] if llama.cpp returns null.
    pub fn to_owned_chunk(&self) -> Result<OwnedMtmdInputChunk> {
        let ptr = unsafe { sys::mtmd_input_chunk_copy(self.ptr) };
        NonNull::new(ptr)
            .map(|ptr| OwnedMtmdInputChunk { ptr })
            .ok_or(MtmdError::ChunkLoadFailed)
    }

    /// Copy this chunk as a standalone placeholder — metadata only, no payload.
    ///
    /// Same shape as a round trip through [`Self::save`] and
    /// [`MtmdInputChunks::load_chunk`], without the serialization. Useful for
    /// keeping a prompt's structure alive after the pixels have been dropped.
    ///
    /// Wraps `mtmd_input_chunk_get_placeholder`.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::ChunkLoadFailed`] if llama.cpp returns null.
    pub fn to_placeholder(&self) -> Result<OwnedMtmdInputChunk> {
        let ptr = unsafe { sys::mtmd_input_chunk_get_placeholder(self.ptr) };
        NonNull::new(ptr)
            .map(|ptr| OwnedMtmdInputChunk { ptr })
            .ok_or(MtmdError::ChunkLoadFailed)
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

    /// Restoring a chunk from garbage must return an error, not abort. The
    /// underlying C returns null on failure, and a null deref here would take
    /// the process with it.
    /// Probing a file that is not an mmproj must report "no modalities"
    /// rather than crash — a server calls this on a user-supplied path.
    #[test]
    fn mmproj_caps_on_a_non_mmproj_file_reports_nothing() {
        let caps = mmproj_caps("/definitely/not/a/model.gguf").expect("no NUL in path");
        assert!(!caps.vision);
        assert!(!caps.audio);
    }

    #[test]
    fn mmproj_caps_rejects_interior_nul() {
        assert!(mmproj_caps("a\0b").is_err());
    }

    #[test]
    fn load_chunk_rejects_garbage() {
        let err = MtmdInputChunks::load_chunk(b"not a serialized chunk").unwrap_err();
        assert!(matches!(err, MtmdError::ChunkLoadFailed), "got {err:?}");
    }

    #[test]
    fn load_chunk_rejects_empty_input() {
        assert!(MtmdInputChunks::load_chunk(&[]).is_err());
    }

    /// A truncated buffer is the realistic corruption case for a chunk read
    /// back off disk beside a session file.
    #[test]
    fn load_chunk_rejects_truncated_input() {
        assert!(MtmdInputChunks::load_chunk(&[0u8; 4]).is_err());
    }

    #[test]
    fn input_text_try_new_is_infallible() {
        let input = MtmdInputText::try_new("marker \u{1} data", true, true)
            .expect("try_new no longer rejects any input");
        assert_eq!(input.text_len, "marker \u{1} data".len());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OwnedMtmdInputChunk
// ─────────────────────────────────────────────────────────────────────────────

/// A chunk that owns its allocation, as returned by
/// [`MtmdInputChunks::load_chunk`] or [`MtmdInputChunk::to_placeholder`].
///
/// [`MtmdInputChunk`] borrows from the [`MtmdInputChunks`] list that holds it;
/// this one stands alone and frees itself on drop.
pub struct OwnedMtmdInputChunk {
    ptr: NonNull<sys::mtmd_input_chunk>,
}

impl std::fmt::Debug for OwnedMtmdInputChunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedMtmdInputChunk")
            .field("chunk_type", &self.chunk_type())
            .field("n_tokens", &self.n_tokens())
            .finish()
    }
}

impl Drop for OwnedMtmdInputChunk {
    fn drop(&mut self) {
        unsafe { sys::mtmd_input_chunk_free(self.ptr.as_ptr()) }
    }
}

impl OwnedMtmdInputChunk {
    /// The type of this chunk.
    #[must_use]
    pub fn chunk_type(&self) -> MtmdInputChunkType {
        MtmdInputChunkType::from(unsafe { sys::mtmd_input_chunk_get_type(self.ptr.as_ptr()) })
    }

    /// Total number of tokens in this chunk.
    #[must_use]
    pub fn n_tokens(&self) -> usize {
        unsafe { sys::mtmd_input_chunk_get_n_tokens(self.ptr.as_ptr()) }
    }

    /// Number of temporal positions.
    #[must_use]
    pub fn n_pos(&self) -> i32 {
        unsafe { sys::mtmd_input_chunk_get_n_pos(self.ptr.as_ptr()) }
    }

    /// Serialize this chunk's metadata, as [`MtmdInputChunk::save`] does.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::ChunkSaveFailed`] if llama.cpp cannot serialize it.
    pub fn save(&self) -> Result<Vec<u8>> {
        let mut needed: usize = 0;
        let rc = unsafe {
            sys::mtmd_input_chunk_save(
                self.ptr.as_ptr(),
                std::ptr::null_mut(),
                0,
                &raw mut needed,
            )
        };
        if rc != 0 && needed == 0 {
            return Err(MtmdError::ChunkSaveFailed(rc));
        }
        let mut buf = vec![0u8; needed];
        let rc = unsafe {
            sys::mtmd_input_chunk_save(
                self.ptr.as_ptr(),
                buf.as_mut_ptr().cast::<std::os::raw::c_char>(),
                buf.len(),
                &raw mut needed,
            )
        };
        if rc != 0 {
            return Err(MtmdError::ChunkSaveFailed(rc));
        }
        buf.truncate(needed);
        Ok(buf)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MtmdBatch
// ─────────────────────────────────────────────────────────────────────────────

/// Encode several media chunks in one pass.
///
/// [`MtmdContext::encode_chunk`] handles one chunk at a time; this batches
/// them, which is what you want for a multi-image prompt or a run of video
/// frames — the vision encoder runs once over the whole set instead of once per
/// image.
///
/// A batch belongs to the context that created it and borrows it for its
/// lifetime. Chunks are *not* owned by the batch, so they must outlive it too.
///
/// Wraps `mtmd_batch_init` / `mtmd_batch_add_chunk` / `mtmd_batch_encode`.
pub struct MtmdBatch<'ctx> {
    ptr: NonNull<sys::mtmd_batch>,
    _ctx: std::marker::PhantomData<&'ctx MtmdContext>,
}

impl std::fmt::Debug for MtmdBatch<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtmdBatch").finish_non_exhaustive()
    }
}

impl Drop for MtmdBatch<'_> {
    fn drop(&mut self) {
        unsafe { sys::mtmd_batch_free(self.ptr.as_ptr()) }
    }
}

impl<'ctx> MtmdBatch<'ctx> {
    /// Start a batch against `ctx`.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::BatchCreateFailed`] if llama.cpp returns null.
    pub fn new(ctx: &'ctx MtmdContext) -> Result<Self> {
        let ptr = unsafe { sys::mtmd_batch_init(ctx.ptr.as_ptr()) };
        NonNull::new(ptr)
            .map(|ptr| Self {
                ptr,
                _ctx: std::marker::PhantomData,
            })
            .ok_or(MtmdError::BatchCreateFailed)
    }

    /// Add a media chunk. Text chunks are rejected.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::BatchAddFailed`] — code `2` means the batch is
    /// full and the chunk was not added (start a new batch), code `3` means it
    /// cannot be batched with what is already there (differing image
    /// geometry, say).
    pub fn add_chunk(&mut self, chunk: &MtmdInputChunk<'_>) -> Result<()> {
        let rc = unsafe { sys::mtmd_batch_add_chunk(self.ptr.as_ptr(), chunk.ptr) };
        if rc == 0 {
            Ok(())
        } else {
            Err(MtmdError::BatchAddFailed(rc))
        }
    }

    /// Encode every chunk added so far.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::EncodeError`] on failure.
    pub fn encode(&mut self) -> Result<()> {
        let rc = unsafe { sys::mtmd_batch_encode(self.ptr.as_ptr()) };
        if rc == 0 {
            Ok(())
        } else {
            Err(MtmdError::EncodeError(rc))
        }
    }

    /// Borrow the embeddings produced for `chunk` by the last [`Self::encode`].
    ///
    /// Returns `None` if the chunk was not part of this batch or encoding has
    /// not run. The slice is owned by the batch and is invalidated by the next
    /// `encode`.
    ///
    /// # Safety of the returned length
    ///
    /// llama.cpp reports only a pointer, so the length is derived from the
    /// chunk's token count and the context's embedding dimension.
    #[must_use]
    pub fn output_embd(&self, chunk: &MtmdInputChunk<'_>, n_embd: usize) -> Option<&[f32]> {
        let ptr = unsafe { sys::mtmd_batch_get_output_embd(self.ptr.as_ptr(), chunk.ptr) };
        if ptr.is_null() {
            return None;
        }
        let len = chunk.n_tokens().checked_mul(n_embd)?;
        if len == 0 {
            return Some(&[]);
        }
        Some(unsafe { slice::from_raw_parts(ptr, len) })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio generation
// ─────────────────────────────────────────────────────────────────────────────

/// Container for generated audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtmdAudioOutType {
    /// Raw PCM samples.
    Pcm,
    /// A complete WAV file: PCM 16-bit little-endian, mono.
    Wav,
}

impl MtmdAudioOutType {
    fn as_raw(self) -> sys::mtmd_helper_gen_audio_outtype {
        match self {
            Self::Pcm => sys::MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM,
            Self::Wav => sys::MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV,
        }
    }
}

/// What to synthesize, and how.
#[derive(Debug, Clone)]
pub struct MtmdAudioRequest {
    /// Sequence id to generate under.
    pub seq_id: i32,
    /// Text to speak.
    pub prompt: String,
    /// BCP-47-ish language hint, if the pipeline takes one.
    pub lang: Option<String>,
    /// Top-k for the backbone sampler.
    pub top_k: i32,
    /// Top-p for the backbone sampler.
    pub top_p: f32,
    /// Seed; `u32::MAX` means random.
    pub seed: u32,
    /// Container for [`MtmdAudioGen::output`].
    pub out_type: MtmdAudioOutType,
}

impl MtmdAudioRequest {
    /// A request to speak `prompt` with upstream's defaults.
    #[must_use]
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            seq_id: 0,
            prompt: prompt.into(),
            lang: None,
            top_k: 40,
            top_p: 0.9,
            seed: u32::MAX,
            out_type: MtmdAudioOutType::Wav,
        }
    }
}

/// Text-to-speech through an mmproj audio-generation pipeline.
///
/// This is the *other* direction of multimodal: where [`MtmdBitmap`] feeds
/// audio in, this drives a pipeline that emits it. The loop is explicitly
/// stateless on llama.cpp's side, so it runs in two phases:
///
/// 1. [`Self::set_input`], then [`Self::step_prompt`] until it returns `0` —
///    the prompt is consumed `n_batch` tokens at a time.
/// 2. [`Self::step_gen`] per frame until it reports stop.
/// 3. [`Self::output`] for the finished audio.
///
/// Wraps `mtmd_helper_gen_audio_*`.
pub struct MtmdAudioGen<'ctx> {
    ptr: NonNull<sys::mtmd_helper_gen_audio>,
    _ctx: std::marker::PhantomData<&'ctx MtmdContext>,
}

impl std::fmt::Debug for MtmdAudioGen<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtmdAudioGen").finish_non_exhaustive()
    }
}

impl Drop for MtmdAudioGen<'_> {
    fn drop(&mut self) {
        unsafe { sys::mtmd_helper_gen_audio_free(self.ptr.as_ptr()) }
    }
}

impl<'ctx> MtmdAudioGen<'ctx> {
    /// Attach a generator to a llama context and an mtmd context.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::ContextCreateFailed`] if the mmproj has no
    /// audio-generation pipeline.
    pub fn new(
        lctx: &mut crate::context::LlamaContext<'_>,
        mctx: &'ctx MtmdContext,
    ) -> Result<Self> {
        let ptr = unsafe {
            sys::mtmd_helper_gen_audio_init(lctx.context.as_ptr(), mctx.ptr.as_ptr())
        };
        NonNull::new(ptr)
            .map(|ptr| Self {
                ptr,
                _ctx: std::marker::PhantomData,
            })
            .ok_or(MtmdError::ContextCreateFailed)
    }

    /// Clear all state, ready for another utterance.
    pub fn reset(&mut self) {
        unsafe { sys::mtmd_helper_gen_audio_reset(self.ptr.as_ptr()) }
    }

    /// Set what to synthesize. `speaker_ref` is an optional voice reference for
    /// pipelines that support cloning.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::EvalError`] if llama.cpp rejects the request, or
    /// [`MtmdError::InvalidPath`] if a string contains an interior NUL.
    pub fn set_input(
        &mut self,
        request: &MtmdAudioRequest,
        speaker_ref: Option<&MtmdBitmap>,
    ) -> Result<()> {
        let prompt = CString::new(request.prompt.as_str())?;
        let lang = request.lang.as_deref().map(CString::new).transpose()?;
        let inp = sys::mtmd_helper_gen_audio_inp {
            seq_id: request.seq_id,
            prompt: prompt.as_ptr(),
            prompt_len: request.prompt.len(),
            speaker_ref: speaker_ref.map_or(std::ptr::null_mut(), |b| b.ptr.as_ptr()),
            lang: lang.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            top_k: request.top_k,
            top_p: request.top_p,
            seed: request.seed,
            out_type: request.out_type.as_raw(),
        };
        let rc = unsafe { sys::mtmd_helper_gen_audio_set_input(self.ptr.as_ptr(), &raw const inp) };
        if rc == 0 {
            Ok(())
        } else {
            Err(MtmdError::EvalError(rc))
        }
    }

    /// Consume up to `n_batch` prompt tokens.
    ///
    /// Returns the number of prompt tokens still outstanding; call again until
    /// it returns `0`, then move on to [`Self::step_gen`].
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::EvalError`] if llama.cpp reports a negative code.
    pub fn step_prompt(&mut self, n_batch: i32) -> Result<i32> {
        let rc = unsafe { sys::mtmd_helper_gen_audio_step_prompt(self.ptr.as_ptr(), n_batch) };
        if rc < 0 {
            return Err(MtmdError::EvalError(rc));
        }
        Ok(rc)
    }

    /// Generate one audio frame.
    ///
    /// `sampled` is the backbone token just sampled, or `None` for pipelines
    /// with no discrete backbone token. `h_state_in` is the hidden state fed
    /// back from the previous step.
    ///
    /// Returns `(hidden_state, stop)`. `stop` marks end-of-speech: the caller
    /// must break the loop. The hidden state borrows generator memory that the
    /// next `step_gen` or [`Self::reset`] invalidates, hence the `&mut self`
    /// borrow being released before you can call again.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::EvalError`] on a negative code.
    pub fn step_gen(
        &mut self,
        sampled: Option<crate::token::LlamaToken>,
        h_state_in: Option<&[f32]>,
        n_text_embd: usize,
    ) -> Result<(Option<Vec<f32>>, bool)> {
        let token = sampled.map_or(sys::LLAMA_TOKEN_NULL, |t| t.0);
        let in_ptr = h_state_in.map_or(std::ptr::null(), <[f32]>::as_ptr);
        let mut out_ptr: *const f32 = std::ptr::null();
        let mut stop = false;
        let rc = unsafe {
            sys::mtmd_helper_gen_audio_step_gen(
                self.ptr.as_ptr(),
                token,
                in_ptr,
                &raw mut out_ptr,
                &raw mut stop,
            )
        };
        if rc < 0 {
            return Err(MtmdError::EvalError(rc));
        }
        // Copy rather than borrow: upstream documents the buffer as valid only
        // until the next step_gen/reset, which a returned slice could outlive.
        let state = if out_ptr.is_null() || n_text_embd == 0 {
            None
        } else {
            Some(unsafe { slice::from_raw_parts(out_ptr, n_text_embd) }.to_vec())
        };
        Ok((state, stop))
    }

    /// Collect the generated audio.
    ///
    /// Returns `(sample_rate, bytes, n_samples)`. `bytes` is raw PCM or a
    /// complete WAV file depending on the request's
    /// [`MtmdAudioOutType`].
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::EvalError`] if nothing has been generated.
    pub fn output(&mut self) -> Result<(i32, Vec<u8>, i64)> {
        let mut sample_rate: i32 = 0;
        let mut data: *const std::os::raw::c_char = std::ptr::null();
        let mut data_len: usize = 0;
        let mut n_samples: i64 = 0;
        let rc = unsafe {
            sys::mtmd_helper_gen_audio_get_output(
                self.ptr.as_ptr(),
                &raw mut sample_rate,
                &raw mut data,
                &raw mut data_len,
                &raw mut n_samples,
            )
        };
        if rc != 0 {
            return Err(MtmdError::EvalError(rc));
        }
        let bytes = if data.is_null() || data_len == 0 {
            Vec::new()
        } else {
            // Copied for the same reason as step_gen: valid only until the next
            // get_output/reset.
            unsafe { slice::from_raw_parts(data.cast::<u8>(), data_len) }.to_vec()
        };
        Ok((sample_rate, bytes, n_samples))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit input parts
// ─────────────────────────────────────────────────────────────────────────────

/// One element of a marker-free prompt, for
/// [`MtmdContext::tokenize_from_parts`].
///
/// Borrows rather than owns, so the caller keeps control of bitmap lifetimes —
/// a bitmap is usually reused across several prompts.
#[derive(Debug)]
pub enum MtmdInputPart<'a> {
    /// A run of text, with its own `parse_special` setting.
    Text(&'a MtmdInputText<'a>),
    /// An image or audio bitmap spliced in at this position.
    Bitmap(&'a MtmdBitmap),
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio-generation capabilities
// ─────────────────────────────────────────────────────────────────────────────

/// Which audio-generation pipeline an mmproj implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtmdGenAudioType {
    /// Qwen3-TTS.
    Qwen3Tts,
    /// `PocketTTS`.
    PocketTts,
    /// A pipeline this crate does not know, added upstream since this release.
    Unknown,
}

impl MtmdGenAudioType {
    fn from_raw(raw: sys::mtmd_gen_audio_type) -> Self {
        match raw {
            sys::MTMD_GEN_AUDIO_TYPE_QWEN3TTS => Self::Qwen3Tts,
            sys::MTMD_GEN_AUDIO_TYPE_POCKETTTS => Self::PocketTts,
            _ => Self::Unknown,
        }
    }
}

/// What [`MtmdContext::gen_audio_info`] reports about a speech pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtmdGenAudioInfo {
    /// The pipeline implemented by this projector.
    pub pipeline: MtmdGenAudioType,
    /// Output sample rate in Hz, e.g. 24000 for Qwen3-TTS. Needed to write a
    /// correct WAV header or resample.
    pub sample_rate: i32,
    /// Weight-variant name, when the pipeline has variants.
    pub model_variant: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Projector capability probe
// ─────────────────────────────────────────────────────────────────────────────

/// Which modalities an mmproj file accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtmdCaps {
    /// Accepts image input.
    pub vision: bool,
    /// Accepts audio input.
    pub audio: bool,
}

/// Read an mmproj file's input capabilities without loading it.
///
/// [`MtmdContext::init_from_file`] builds the full projector — weights, compute
/// buffers, the lot. This only reads enough metadata to answer "does this
/// accept images, audio, or both", which is what a server needs at startup to
/// decide whether a request is even servable.
///
/// Wraps `mtmd_get_cap_from_file`. Returns both flags `false` for a file that
/// is not a readable mmproj.
///
/// # Errors
///
/// Returns [`MtmdError::InvalidPath`] if the path contains an interior NUL, or
/// [`MtmdError::PathNotUtf8`] if it is not UTF-8.
pub fn mmproj_caps(path: impl AsRef<Path>) -> Result<MtmdCaps> {
    let path = path.as_ref().to_str().ok_or(MtmdError::PathNotUtf8)?;
    let c_path = CString::new(path)?;
    let caps = unsafe { sys::mtmd_get_cap_from_file(c_path.as_ptr()) };
    Ok(MtmdCaps {
        vision: caps.inp_vision,
        audio: caps.inp_audio,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Lazy bitmaps
// ─────────────────────────────────────────────────────────────────────────────

/// What a lazy-bitmap callback yields for one chunk index.
#[derive(Debug)]
pub enum MtmdLazyChunk {
    /// An image or audio bitmap. Ownership passes to llama.cpp.
    Bitmap(MtmdBitmap),
    /// A run of text to splice in at this position.
    Text(String),
    /// No more chunks; the placeholder removes itself from the prompt.
    End,
}

/// A bitmap whose contents are produced on demand, during tokenization.
///
/// An ordinary [`MtmdBitmap`] holds decoded pixels or samples from the moment
/// it is built. This holds a callback instead, invoked with `0, 1, 2, …` while
/// the prompt is tokenized and expanding into however many chunks it yields.
/// That matters for two cases:
///
/// - a long video, where materialising every frame up front would not fit in
///   memory;
/// - media that may never be reached, because a stop sequence or a token
///   budget cuts the prompt short first.
///
/// The callback must outlive tokenization, so this owns it alongside the
/// bitmap and drops them in that order.
///
/// Wraps `mtmd_bitmap_init_lazy`.
pub struct MtmdLazyBitmap {
    // Declaration order is the drop order, and it matters: llama.cpp may touch
    // `user_data` while freeing the bitmap, so the bitmap must go first.
    bitmap: MtmdBitmap,
    _callback: Box<LazyCallbackState>,
}

impl std::fmt::Debug for MtmdLazyBitmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtmdLazyBitmap").finish_non_exhaustive()
    }
}

/// Heap home for the user closure, pointed at by `user_data`.
struct LazyCallbackState {
    func: Box<dyn FnMut(usize) -> MtmdLazyChunk>,
}

impl MtmdLazyBitmap {
    /// Build a lazy bitmap identified by `id` (conventionally a file hash).
    ///
    /// `callback` is called with increasing chunk indices until it returns
    /// [`MtmdLazyChunk::End`].
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::BitmapCreateFailed`] if llama.cpp returns null, or
    /// [`MtmdError::InvalidPath`] if `id` contains an interior NUL.
    pub fn new<F>(ctx: &MtmdContext, id: &str, callback: F) -> Result<Self>
    where
        F: FnMut(usize) -> MtmdLazyChunk + 'static,
    {
        let c_id = CString::new(id)?;
        let mut state = Box::new(LazyCallbackState {
            func: Box::new(callback),
        });
        let user_data = std::ptr::from_mut(state.as_mut()).cast::<std::os::raw::c_void>();

        let ptr = unsafe {
            sys::mtmd_bitmap_init_lazy(
                ctx.ptr.as_ptr(),
                c_id.as_ptr(),
                user_data,
                Some(lazy_trampoline),
            )
        };
        let bitmap = MtmdBitmap {
            ptr: NonNull::new(ptr).ok_or(MtmdError::BitmapCreateFailed)?,
        };
        Ok(Self {
            bitmap,
            _callback: state,
        })
    }

    /// Borrow this as an ordinary bitmap, for passing to
    /// [`MtmdContext::tokenize`].
    #[must_use]
    pub fn as_bitmap(&self) -> &MtmdBitmap {
        &self.bitmap
    }
}

/// C entry point for [`MtmdLazyBitmap`].
///
/// Returns `0` when a chunk was produced, `-1` at EOF, `-2` on error. A Rust
/// panic must not unwind into C, so it is caught and reported as `-2`.
extern "C" fn lazy_trampoline(
    chunk_idx: usize,
    user_data: *mut std::os::raw::c_void,
    out_bitmap: *mut *mut sys::mtmd_bitmap,
    out_text: *mut *mut std::os::raw::c_char,
) -> std::os::raw::c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if user_data.is_null() {
            return -2;
        }
        let state = unsafe { &mut *user_data.cast::<LazyCallbackState>() };
        match (state.func)(chunk_idx) {
            MtmdLazyChunk::Bitmap(bitmap) => {
                // Ownership moves to llama.cpp, which frees it with
                // `mtmd_bitmap_free`; skip our own Drop.
                let raw = bitmap.ptr.as_ptr();
                std::mem::forget(bitmap);
                unsafe { *out_bitmap = raw };
                0
            }
            MtmdLazyChunk::Text(text) => {
                let Ok(c_text) = CString::new(text) else {
                    return -2;
                };
                // llama.cpp releases this with `free()`, so it must come from
                // malloc — a Rust-allocated buffer would be freed by the wrong
                // allocator.
                let dup = unsafe { strdup(c_text.as_ptr()) };
                if dup.is_null() {
                    return -2;
                }
                unsafe { *out_text = dup };
                0
            }
            MtmdLazyChunk::End => -1,
        }
    }));
    result.unwrap_or(-2)
}
