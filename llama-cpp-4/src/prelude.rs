//! Convenience re-exports for typical inference workflows.
//!
//! # Quick start
//!
//! ```
//! use llama_cpp_4::prelude::*;
//! ```
//!
//! Core types are also available at the crate root (`llama_cpp_4::LlamaModel`, …)
//! if you prefer explicit paths over a glob import.
//!
//! # What's included
//!
//! | Category | Re-exported types |
//! |---|---|
//! | Inference | [`LlamaBackend`], [`LlamaModel`], [`LlamaModelParams`], [`LlamaContext`], [`LlamaContextParams`], [`LlamaBatch`], [`LlamaSampler`], [`LlamaSamplerParams`], [`LlamaToken`], [`LlamaTokenDataArray`] |
//! | Tokenising | [`AddBos`], [`Special`] |
//! | Chat | [`LlamaChatMessage`] |
//! | Model introspection | [`LlamaBackendDevice`], [`LlamaBackendDeviceType`] |
//! | Context params | [`LlamaFlashAttnType`], [`LlamaContextType`], [`LlamaAttentionType`], [`RopeScalingType`], [`LlamaPoolingType`], [`ParamsCloneError`] |
//! | KV overrides | [`ParamOverrideValue`] |
//! | Errors | [`Result`], [`LLamaCppError`], [`DecodeError`], [`EncodeError`], [`EmbeddingsError`], [`BatchAddError`], [`ApplyChatTemplateError`], [`NewLlamaChatMessageError`] |
//! | Memory / fit | [`get_device_memory_data`], [`fit_params`], [`FitParams`], [`FitParamsResult`], [`FitParamsError`], [`DeviceMemoryReport`], [`MemoryBreakdownEntry`] |
//! | Tensor capture | [`TensorCapture`], [`CapturedTensor`] |
//! | Speculative | [`MtpSession`], [`MtpSessionConfig`], [`Eagle3Session`], [`Eagle3SessionConfig`] |
//! | Quantization | [`QuantizeParams`], [`TensorTypeOverride`], [`GgmlType`], [`LlamaFtype`], [`model_quantize`], [`attn_rot_disabled`], [`set_attn_rot_disabled`] |
//! | Utilities | [`ggml_time_us`], [`llama_time_us`], [`print_system_info`], [`supports_gpu_offload`], [`max_devices`] |
//!
//! With the `mtmd` feature: [`MtmdContext`], [`MtmdBitmap`], …\
//! With the `rpc` feature: `RpcBackend`, `RpcServer`, and `RpcError` in `llama_cpp_4::rpc`.
//!
//! # Text generation
//!
//! ```no_run
//! use llama_cpp_4::prelude::*;
//! use std::num::NonZeroU32;
//!
//! fn main() {
//!     let backend = LlamaBackend::init().unwrap();
//!     let model = LlamaModel::load_from_file(
//!         &backend,
//!         "model.gguf",
//!         &LlamaModelParams::default(),
//!     )
//!     .unwrap();
//!     let mut ctx = model
//!         .new_context(
//!             &backend,
//!             LlamaContextParams::default().with_n_ctx(NonZeroU32::new(2048)),
//!         )
//!         .unwrap();
//!
//!     let tokens = model.str_to_token("The answer is", AddBos::Always).unwrap();
//!     let n_prompt = tokens.len();
//!     let mut batch = LlamaBatch::new(2048, 1);
//!     for (i, &tok) in tokens.iter().enumerate() {
//!         batch
//!             .add(tok, i as i32, &[0], i == n_prompt - 1)
//!             .unwrap();
//!     }
//!     ctx.decode(&mut batch).unwrap();
//!
//!     let sampler = LlamaSampler::chain_simple([
//!         LlamaSampler::temp(0.8),
//!         LlamaSampler::dist(0),
//!     ]);
//!     let token = sampler.sample(&ctx, 0);
//!     let _text = model.token_to_bytes(token, Special::Plaintext).unwrap();
//! }
//! ```
//!
//! # Chat template
//!
//! ```no_run
//! use llama_cpp_4::prelude::*;
//!
//! fn main() {
//!     let backend = LlamaBackend::init().unwrap();
//!     let model = LlamaModel::load_from_file(
//!         &backend,
//!         "model.gguf",
//!         &LlamaModelParams::default(),
//!     )
//!     .unwrap();
//!
//!     let messages = vec![
//!         LlamaChatMessage::new("system".into(), "You are helpful.".into()).unwrap(),
//!         LlamaChatMessage::new("user".into(), "What is 2+2?".into()).unwrap(),
//!     ];
//!     let prompt = model.apply_chat_template(None, &messages, true).unwrap();
//!     let _tokens = model.str_to_token(&prompt, AddBos::Always).unwrap();
//! }
//! ```
//!
//! # Embeddings
//!
//! ```no_run
//! use llama_cpp_4::prelude::*;
//! use std::num::NonZeroU32;
//!
//! fn main() {
//!     let backend = LlamaBackend::init().unwrap();
//!     let model = LlamaModel::load_from_file(
//!         &backend,
//!         "model.gguf",
//!         &LlamaModelParams::default(),
//!     )
//!     .unwrap();
//!     let mut ctx = model
//!         .new_context(
//!             &backend,
//!             LlamaContextParams::default()
//!                 .with_embeddings(true)
//!                 .with_n_ctx(NonZeroU32::new(512)),
//!         )
//!         .unwrap();
//!
//!     let tokens = model.str_to_token("Hello", AddBos::Always).unwrap();
//!     let mut batch = LlamaBatch::new(512, 1);
//!     for (i, &tok) in tokens.iter().enumerate() {
//!         batch
//!             .add(tok, i as i32, &[0], i == tokens.len() - 1)
//!             .unwrap();
//!     }
//!     ctx.decode(&mut batch).unwrap();
//!     let _vec = ctx.embeddings_seq_ith(0).unwrap();
//! }
//! ```
//!
//! # Memory estimation (before loading fully)
//!
//! ```no_run
//! use llama_cpp_4::prelude::*;
//! use std::path::Path;
//!
//! fn main() {
//!     let report = get_device_memory_data(
//!         Path::new("model.gguf"),
//!         &LlamaModelParams::default(),
//!         &LlamaContextParams::default(),
//!         llama_cpp_sys_4::GGML_LOG_LEVEL_ERROR,
//!     )
//!     .unwrap();
//!     for entry in &report.entries {
//!         println!("projected used: {} bytes", entry.used());
//!     }
//! }
//! ```

// ── Core inference ────────────────────────────────────────────────────────────

pub use crate::context::params::{
    LlamaAttentionType, LlamaContextParams, LlamaContextType, LlamaFlashAttnType, LlamaPoolingType,
    ParamsCloneError, RopeScalingType,
};
pub use crate::context::{CapturedTensor, LlamaContext, MemoryBreakdownEntry, TensorCapture};
pub use crate::llama_backend::LlamaBackend;
pub use crate::llama_batch::{BatchAddError, LlamaBatch};
pub use crate::model::params::kv_overrides::ParamOverrideValue;
pub use crate::model::params::LlamaModelParams;
pub use crate::model::{
    AddBos, LlamaBackendDevice, LlamaBackendDeviceType, LlamaChatMessage, LlamaModel, Special,
};
pub use crate::sampling::{LlamaSampler, LlamaSamplerParams};
pub use crate::token::data_array::LlamaTokenDataArray;
pub use crate::token::LlamaToken;

// ── Errors & results ────────────────────────────────────────────────────────

pub use crate::{
    ApplyChatTemplateError, DecodeError, EmbeddingsError, EncodeError, LLamaCppError,
    LlamaContextLoadError, LlamaModelLoadError, NewLlamaChatMessageError, Result,
};

// ── Memory / fit helpers ────────────────────────────────────────────────────

pub use crate::fit::{
    fit_params, get_device_memory_data, DeviceMemoryEntry, DeviceMemoryError,
    DeviceMemoryHyperParams, DeviceMemoryReport, FitParams, FitParamsError, FitParamsResult,
};

// ── Speculative decoding ────────────────────────────────────────────────────

pub use crate::eagle::{Eagle3Session, Eagle3SessionConfig};
pub use crate::mtp::{MtpSession, MtpSessionConfig};

// ── Multimodal (feature `mtmd`) ─────────────────────────────────────────────

#[cfg(feature = "mtmd")]
pub use crate::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText,
    MtmdProgressCallback,
};

// ── Remote backend (feature `rpc`) ──────────────────────────────────────────

#[cfg(feature = "rpc")]
pub use crate::rpc::{RpcBackend, RpcError, RpcServer};

// ── Quantization ────────────────────────────────────────────────────────────

pub use crate::quantize::{
    attn_rot_disabled, set_attn_rot_disabled, GgmlType, LlamaFtype, QuantizeParams,
    TensorTypeOverride,
};

// ── Utilities ───────────────────────────────────────────────────────────────

pub use crate::{
    ggml_time_us, llama_time_us, max_devices, model_quantize, print_system_info,
    supports_gpu_offload,
};
