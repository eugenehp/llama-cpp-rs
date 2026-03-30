//! Bindings to the llama.cpp library.
//!
//! As llama.cpp is a very fast moving target, this crate does not attempt to create a stable API
//! with all the rust idioms. Instead it provides safe wrappers around nearly direct bindings to
//! llama.cpp. This makes it easier to keep up with the changes in llama.cpp, but does mean that
//! the API is not as nice as it could be.
//!
//! # Examples
//!
//! - [simple](https://github.com/eugenehp/llama-cpp-rs/tree/main/examples/simple)
//! - [chat](https://github.com/eugenehp/llama-cpp-rs/tree/main/examples/chat)
//! - [embeddings](https://github.com/eugenehp/llama-cpp-rs/tree/main/examples/embeddings)
//! - [server](https://github.com/eugenehp/llama-cpp-rs/tree/main/examples/server)
//!
//! # Feature Flags
//!
//! - `cuda` enables CUDA GPU support.
//! - `metal` enables Apple Metal GPU support.
//! - `vulkan` enables Vulkan GPU support (AMD / Intel / cross-platform).
//! - `native` enables host-CPU optimisations (`-march=native`).
//! - `openmp` enables OpenMP multi-core CPU parallelism (on by default).
//! - `rpc` enables RPC backend support for distributed inference across multiple machines.
//! - `mtmd` enables multimodal (image + audio) support via `libmtmd`.
use std::ffi::NulError;
use std::fmt::Debug;
use std::num::NonZeroI32;

use crate::llama_batch::BatchAddError;
use std::os::raw::c_int;
use std::path::PathBuf;
use std::string::FromUtf8Error;

pub mod common;
pub mod context;
#[cfg(feature = "ggml")]
pub mod ggml;
pub mod llama_backend;
pub mod llama_batch;
pub mod model;
pub mod sampling;
pub mod token;
pub mod token_type;

#[cfg(feature = "rpc")]
pub mod rpc;

#[cfg(feature = "mtmd")]
pub mod mtmd;

/// A failable result from a llama.cpp function.
pub type Result<T> = std::result::Result<T, LLamaCppError>;

/// All errors that can occur in the llama-cpp crate.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LLamaCppError {
    /// The backend was already initialized. This can generally be ignored as initializing the backend
    /// is idempotent.
    #[error("BackendAlreadyInitialized")]
    BackendAlreadyInitialized,
    /// There was an error while get the chat template from model.
    #[error("{0}")]
    ChatTemplateError(#[from] ChatTemplateError),
    /// There was an error while decoding a batch.
    #[error("{0}")]
    DecodeError(#[from] DecodeError),
    /// There was an error while encoding a batch.
    #[error("{0}")]
    EncodeError(#[from] EncodeError),
    /// There was an error loading a model.
    #[error("{0}")]
    LlamaModelLoadError(#[from] LlamaModelLoadError),
    /// There was an error creating a new model context.
    #[error("{0}")]
    LlamaContextLoadError(#[from] LlamaContextLoadError),
    /// There was an error adding a token to a batch.
    #[error["{0}"]]
    BatchAddError(#[from] BatchAddError),
    /// see [`EmbeddingsError`]
    #[error(transparent)]
    EmbeddingError(#[from] EmbeddingsError),
}

/// There was an error while getting the chat template from a model.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ChatTemplateError {
    /// the buffer was too small.
    #[error("The buffer was too small. However, a buffer size of {0} would be just large enough.")]
    BuffSizeError(usize),
    /// gguf has no chat template
    #[error("the model has no meta val - returned code {0}")]
    MissingTemplate(i32),
    /// The chat template was not valid utf8.
    #[error(transparent)]
    Utf8Error(#[from] std::str::Utf8Error),
}

/// Error retrieving a string from the model (e.g. description, metadata key/value).
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum StringFromModelError {
    /// The C function returned a negative error code.
    #[error("llama.cpp returned error code {0}")]
    ReturnedError(i32),
    /// The returned bytes were not valid UTF-8.
    #[error(transparent)]
    Utf8Error(#[from] std::str::Utf8Error),
}

/// Failed to Load context
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaContextLoadError {
    /// llama.cpp returned null
    #[error("null reference from llama.cpp")]
    NullReturn,
}

/// Failed to decode a batch.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum DecodeError {
    /// No kv cache slot was available.
    #[error("Decode Error 1: NoKvCacheSlot")]
    NoKvCacheSlot,
    /// The number of tokens in the batch was 0.
    #[error("Decode Error -1: n_tokens == 0")]
    NTokensZero,
    /// An unknown error occurred.
    #[error("Decode Error {0}: unknown")]
    Unknown(c_int),
}

/// Failed to decode a batch.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum EncodeError {
    /// No kv cache slot was available.
    #[error("Encode Error 1: NoKvCacheSlot")]
    NoKvCacheSlot,
    /// The number of tokens in the batch was 0.
    #[error("Encode Error -1: n_tokens == 0")]
    NTokensZero,
    /// An unknown error occurred.
    #[error("Encode Error {0}: unknown")]
    Unknown(c_int),
}

/// When embedding related functions fail
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum EmbeddingsError {
    /// Embeddings weren't enabled in the context options
    #[error("Embeddings weren't enabled in the context options")]
    NotEnabled,
    /// Logits weren't enabled for the given token
    #[error("Logits were not enabled for the given token")]
    LogitsNotEnabled,
    /// The given sequence index exceeds the max sequence id
    #[error("Can't use sequence embeddings with a model supporting only LLAMA_POOLING_TYPE_NONE")]
    NonePoolType,
}

/// Decode a error from llama.cpp into a [`DecodeError`].
impl From<NonZeroI32> for DecodeError {
    fn from(value: NonZeroI32) -> Self {
        match value.get() {
            1 => DecodeError::NoKvCacheSlot,
            -1 => DecodeError::NTokensZero,
            i => DecodeError::Unknown(i),
        }
    }
}

/// Encode a error from llama.cpp into a [`EncodeError`].
impl From<NonZeroI32> for EncodeError {
    fn from(value: NonZeroI32) -> Self {
        match value.get() {
            1 => EncodeError::NoKvCacheSlot,
            -1 => EncodeError::NTokensZero,
            i => EncodeError::Unknown(i),
        }
    }
}

/// An error that can occur when loading a model.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaModelLoadError {
    /// There was a null byte in a provided string and thus it could not be converted to a C string.
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),
    /// llama.cpp returned a nullptr - this could be many different causes.
    #[error("null result from llama cpp")]
    NullResult,
    /// Failed to convert the path to a rust str. This means the path was not valid unicode
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
}

/// An error that can occur when loading a model.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaLoraAdapterInitError {
    /// There was a null byte in a provided string and thus it could not be converted to a C string.
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),
    /// llama.cpp returned a nullptr - this could be many different causes.
    #[error("null result from llama cpp")]
    NullResult,
    /// Failed to convert the path to a rust str. This means the path was not valid unicode
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
}

/// An error that can occur when loading a model.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaLoraAdapterSetError {
    /// llama.cpp returned a non-zero error code.
    #[error("error code from llama cpp")]
    ErrorResult(i32),
}

/// An error that can occur when loading a model.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaLoraAdapterRemoveError {
    /// llama.cpp returned a non-zero error code.
    #[error("error code from llama cpp")]
    ErrorResult(i32),
}

/// get the time (in microseconds) according to llama.cpp
/// ```
/// # use llama_cpp_4::llama_time_us;
/// let time = llama_time_us();
/// assert!(time > 0);
/// ```
#[must_use]
pub fn llama_time_us() -> i64 {
    unsafe { llama_cpp_sys_4::llama_time_us() }
}

/// get the max number of devices according to llama.cpp (this is generally cuda devices)
/// ```
/// # use llama_cpp_4::max_devices;
/// let max_devices = max_devices();
/// assert!(max_devices >= 0);
/// ```
#[must_use]
pub fn max_devices() -> usize {
    unsafe { llama_cpp_sys_4::llama_max_devices() }
}

/// is memory mapping supported according to llama.cpp
/// ```
/// # use llama_cpp_4::mmap_supported;
/// let mmap_supported = mmap_supported();
/// if mmap_supported {
///   println!("mmap_supported!");
/// }
/// ```
#[must_use]
pub fn mmap_supported() -> bool {
    unsafe { llama_cpp_sys_4::llama_supports_mmap() }
}

/// is memory locking supported according to llama.cpp
/// ```
/// # use llama_cpp_4::mlock_supported;
/// let mlock_supported = mlock_supported();
/// if mlock_supported {
///    println!("mlock_supported!");
/// }
/// ```
#[must_use]
pub fn mlock_supported() -> bool {
    unsafe { llama_cpp_sys_4::llama_supports_mlock() }
}

/// An error that can occur when converting a token to a string.
#[derive(Debug, thiserror::Error, Clone)]
#[non_exhaustive]
pub enum TokenToStringError {
    /// the token type was unknown
    #[error("Unknown Token Type")]
    UnknownTokenType,
    /// There was insufficient buffer space to convert the token to a string.
    #[error("Insufficient Buffer Space {0}")]
    InsufficientBufferSpace(c_int),
    /// The token was not valid utf8.
    #[error("FromUtf8Error {0}")]
    FromUtf8Error(#[from] FromUtf8Error),
}

/// Failed to convert a string to a token sequence.
#[derive(Debug, thiserror::Error)]
pub enum StringToTokenError {
    /// the string contained a null byte and thus could not be converted to a c string.
    #[error("{0}")]
    NulError(#[from] NulError),
    #[error("{0}")]
    /// Failed to convert a provided integer to a [`c_int`].
    CIntConversionError(#[from] std::num::TryFromIntError),
}

/// Failed to apply model chat template.
#[derive(Debug, thiserror::Error)]
pub enum NewLlamaChatMessageError {
    /// the string contained a null byte and thus could not be converted to a c string.
    #[error("{0}")]
    NulError(#[from] NulError),
}

/// Failed to apply model chat template.
#[derive(Debug, thiserror::Error)]
pub enum ApplyChatTemplateError {
    /// the buffer was too small.
    #[error("The buffer was too small. Please contact a maintainer and we will update it.")]
    BuffSizeError,
    /// the string contained a null byte and thus could not be converted to a c string.
    #[error("{0}")]
    NulError(#[from] NulError),
    /// the string could not be converted to utf8.
    #[error("{0}")]
    FromUtf8Error(#[from] FromUtf8Error),
}

/// Get the time in microseconds according to ggml
///
/// ```
/// # use std::time::Duration;
/// use llama_cpp_4::ggml_time_us;
///
/// let start = ggml_time_us();
///
/// std::thread::sleep(Duration::from_micros(10));
///
/// let end = ggml_time_us();
///
/// let elapsed = end - start;
///
/// assert!(elapsed >= 10)
#[must_use]
pub fn ggml_time_us() -> i64 {
    unsafe { llama_cpp_sys_4::ggml_time_us() }
}

/// Checks if mlock is supported.
///
/// ```
/// # use llama_cpp_4::llama_supports_mlock;
///
/// if llama_supports_mlock() {
///   println!("mlock is supported!");
/// } else {
///   println!("mlock is not supported!");
/// }
/// ```
#[must_use]
pub fn llama_supports_mlock() -> bool {
    unsafe { llama_cpp_sys_4::llama_supports_mlock() }
}

/// Checks if GPU offload is supported.
///
/// Returns `true` if the library was compiled with GPU support (CUDA, Metal, Vulkan, etc.).
#[must_use]
pub fn supports_gpu_offload() -> bool {
    unsafe { llama_cpp_sys_4::llama_supports_gpu_offload() }
}

/// Checks if RPC backend is supported.
///
/// Returns `true` if the library was compiled with RPC support.
#[must_use]
pub fn supports_rpc() -> bool {
    unsafe { llama_cpp_sys_4::llama_supports_rpc() }
}

/// Get system information string.
///
/// Returns a string containing CPU features, build info, and other system details.
///
/// # Panics
///
/// Panics if the returned string is not valid UTF-8.
#[must_use]
pub fn print_system_info() -> String {
    let c_str = unsafe { llama_cpp_sys_4::llama_print_system_info() };
    let c_str = unsafe { std::ffi::CStr::from_ptr(c_str) };
    c_str.to_str().expect("system info is not valid UTF-8").to_owned()
}

/// Get the maximum number of parallel sequences supported.
#[must_use]
pub fn max_parallel_sequences() -> usize {
    unsafe { llama_cpp_sys_4::llama_max_parallel_sequences() }
}

/// Get the maximum number of tensor buffer type overrides.
#[must_use]
pub fn max_tensor_buft_overrides() -> usize {
    unsafe { llama_cpp_sys_4::llama_max_tensor_buft_overrides() }
}

/// Get the name of a flash attention type.
///
/// # Panics
///
/// Panics if the returned string is not valid UTF-8.
#[must_use]
pub fn flash_attn_type_name(flash_attn_type: i32) -> String {
    let c_str = unsafe { llama_cpp_sys_4::llama_flash_attn_type_name(flash_attn_type) };
    let c_str = unsafe { std::ffi::CStr::from_ptr(c_str) };
    c_str.to_str().expect("flash_attn_type_name is not valid UTF-8").to_owned()
}

/// Get the string representation of a model metadata key.
///
/// # Panics
///
/// Panics if the returned string is not valid UTF-8.
#[must_use]
pub fn model_meta_key_str(key: u32) -> String {
    let c_str = unsafe { llama_cpp_sys_4::llama_model_meta_key_str(key.try_into().unwrap()) };
    let c_str = unsafe { std::ffi::CStr::from_ptr(c_str) };
    c_str.to_str().expect("meta_key_str is not valid UTF-8").to_owned()
}

/// Quantize a model file.
///
/// # Parameters
///
/// - `fname_inp`: Path to the input model file.
/// - `fname_out`: Path to the output quantized model file.
/// - `params`: Quantization parameters. Use `None` for defaults.
///
/// # Returns
///
/// Returns 0 on success, non-zero on failure.
///
/// # Panics
///
/// Panics if the paths contain null bytes.
#[must_use]
pub fn model_quantize(
    fname_inp: &str,
    fname_out: &str,
    params: Option<&llama_cpp_sys_4::llama_model_quantize_params>,
) -> u32 {
    let c_inp = std::ffi::CString::new(fname_inp).expect("input path contains null bytes");
    let c_out = std::ffi::CString::new(fname_out).expect("output path contains null bytes");
    let default_params = unsafe { llama_cpp_sys_4::llama_model_quantize_default_params() };
    let params = params.unwrap_or(&default_params);
    unsafe { llama_cpp_sys_4::llama_model_quantize(c_inp.as_ptr(), c_out.as_ptr(), params) }
}

/// Get default quantization parameters.
#[must_use]
pub fn model_quantize_default_params() -> llama_cpp_sys_4::llama_model_quantize_params {
    unsafe { llama_cpp_sys_4::llama_model_quantize_default_params() }
}

/// Set the log callback.
///
/// # Safety
///
/// The callback and user data must remain valid for the lifetime of the application
/// or until the callback is replaced.
pub unsafe fn log_set(
    callback: llama_cpp_sys_4::ggml_log_callback,
    user_data: *mut std::ffi::c_void,
) {
    llama_cpp_sys_4::llama_log_set(callback, user_data);
}

/// Get the current log callback and user data.
///
/// # Safety
///
/// The caller must ensure the pointers are valid.
pub unsafe fn log_get(
    log_callback: *mut llama_cpp_sys_4::ggml_log_callback,
    user_data: *mut *mut std::ffi::c_void,
) {
    llama_cpp_sys_4::llama_log_get(log_callback, user_data);
}

/// Initialize optimizer state for fine-tuning.
///
/// # Safety
///
/// The context and model must be valid and compatible.
pub unsafe fn opt_init(
    ctx: *mut llama_cpp_sys_4::llama_context,
    model: *mut llama_cpp_sys_4::llama_model,
    params: llama_cpp_sys_4::llama_opt_params,
) {
    llama_cpp_sys_4::llama_opt_init(ctx, model, params);
}

/// Run one training epoch.
///
/// # Safety
///
/// All pointers and handles must be valid.
#[allow(clippy::too_many_arguments)]
pub unsafe fn opt_epoch(
    ctx: *mut llama_cpp_sys_4::llama_context,
    dataset: llama_cpp_sys_4::ggml_opt_dataset_t,
    result_train: llama_cpp_sys_4::ggml_opt_result_t,
    result_eval: llama_cpp_sys_4::ggml_opt_result_t,
    idata_split: i64,
    callback_train: llama_cpp_sys_4::ggml_opt_epoch_callback,
    callback_eval: llama_cpp_sys_4::ggml_opt_epoch_callback,
) {
    llama_cpp_sys_4::llama_opt_epoch(
        ctx,
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval,
    );
}

/// Parameter filter that accepts all tensors (for use with [`opt_init`]).
///
/// # Safety
///
/// The tensor pointer must be valid.
pub unsafe fn opt_param_filter_all(
    tensor: *const llama_cpp_sys_4::ggml_tensor,
    userdata: *mut std::ffi::c_void,
) -> bool {
    llama_cpp_sys_4::llama_opt_param_filter_all(tensor, userdata)
}

/// Auto-fit model and context parameters for available memory.
///
/// # Safety
///
/// All pointers must be valid.
#[allow(clippy::too_many_arguments)]
pub unsafe fn params_fit(
    path_model: *const std::ffi::c_char,
    mparams: *mut llama_cpp_sys_4::llama_model_params,
    cparams: *mut llama_cpp_sys_4::llama_context_params,
    tensor_split: *mut f32,
    tensor_buft_overrides: *mut llama_cpp_sys_4::llama_model_tensor_buft_override,
    margins: *mut usize,
    n_ctx_min: u32,
    log_level: llama_cpp_sys_4::ggml_log_level,
) -> llama_cpp_sys_4::llama_params_fit_status {
    llama_cpp_sys_4::llama_params_fit(
        path_model,
        mparams,
        cparams,
        tensor_split,
        tensor_buft_overrides,
        margins,
        n_ctx_min,
        log_level,
    )
}
