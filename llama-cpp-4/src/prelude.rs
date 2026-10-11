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
//! The rule is that anything a prelude-exported API takes, returns or fails
//! with is exported too — including every error type — so `use
//! llama_cpp_4::prelude::*` is enough to write a complete call, `match` arm
//! included. Lower-level corners stay out: the `ggml` module (feature `ggml`)
//! for raw graph building, and [`crate::quantize`] for the imatrix and
//! quantization-preview types.
//!
//! | Category | Re-exported |
//! |---|---|
//! | Backend & model | [`LlamaBackend`], [`LlamaModel`], [`LlamaModelParams`], [`LlamaBackendDevice`], [`LlamaBackendDeviceType`], [`LlamaLoraAdapter`] |
//! | Context | [`LlamaContext`], [`LlamaContextParams`], [`LlamaContextType`], [`LlamaAttentionType`], [`LlamaFlashAttnType`], [`LlamaPoolingType`], [`RopeScalingType`], [`ParamOverrideValue`] |
//! | Batching | [`LlamaBatch`] for token batches; [`LlamaBatchExt`] with [`BatchEntry`], [`EmbeddingRows`], [`LlamaProcessType`], [`LlamaDecisionOrder`] and [`MROPE_SECTIONS`] for per-entry embeddings and M-RoPE |
//! | Tokens | [`LlamaToken`], [`LlamaTokenData`], [`LlamaTokenDataArray`], [`AddBos`], [`Special`], [`StreamDetokenizer`] |
//! | Sampling | [`LlamaSampler`], [`LlamaSamplerParams`], [`CommonSampler`], [`CommonSamplerParams`], [`CommonSamplerType`], [`GrammarSource`], [`ReasoningBudget`], [`ReasoningBudgetState`] |
//! | Chat & tools | [`LlamaChatMessage`], [`ChatTemplates`], [`ChatApplyParams`], [`ChatParams`], [`GrammarTrigger`], [`ReasoningFormat`], [`ToolChoice`], [`json_schema_to_grammar`] |
//! | Speculative decoding | [`MtpSession`], [`MtpSessionConfig`], [`Eagle3Session`], [`Eagle3SessionConfig`], [`DFlashSession`], [`NgramCache`], [`NgramMap`], [`ngram_simple_draft`], [`ngram_cache_draft`], [`SpeculativeType`], [`speculative_types_from_gguf`] |
//! | Memory & fit | [`fit_params`], [`FitParams`], [`FitParamsResult`], [`get_device_memory_data`], [`DeviceMemoryReport`], [`DeviceMemoryEntry`], [`DeviceMemoryHyperParams`], [`MemoryBreakdownEntry`] |
//! | Tensor capture | [`TensorCapture`], [`CapturedTensor`], [`TensorTransactions`], [`TensorTransaction`], [`TensorSelector`], [`TensorAccess`], … |
//! | Quantization | [`QuantizeParams`], [`TensorTypeOverride`], [`GgmlType`], [`LlamaFtype`], [`model_quantize`], [`attn_rot_disabled`], [`set_attn_rot_disabled`] |
//! | Errors | [`Result`], [`LLamaCppError`], [`ShimError`] (the type behind the shim-backed aliases) and one per area: [`DecodeError`], [`EncodeError`], [`ProcessError`], [`EmbeddingsError`], [`BatchAddError`], [`BatchExtError`], [`ChatError`], [`CommonSamplerError`], [`NgramError`], [`RuntimeError`], [`MtpSessionError`], [`Eagle3SessionError`], [`SpeculativeStateError`], [`DetokenizeError`], [`FitParamsError`], [`DeviceMemoryError`], [`FtypeNameError`], [`QuantPreviewError`], [`LlamaModelLoadError`], [`LlamaContextLoadError`], [`LlamaLoraAdapterInitError`], [`LlamaLoraAdapterSetError`], [`LlamaLoraAdapterRemoveError`], [`ApplyChatTemplateError`], [`NewLlamaChatMessageError`], [`ParamsCloneError`] |
//! | Utilities | [`ggml_time_us`], [`llama_time_us`], [`print_system_info`], [`supports_gpu_offload`], [`max_devices`] |
//!
//! With the `mtmd` feature: [`MtmdContext`], [`MtmdBitmap`], [`MtmdContextParams`],
//! [`MtmdInputChunks`], [`MtmdInputText`], [`MtmdProgressCallback`].\
//! With the `rpc` feature: `RpcBackend`, `RpcError`, `add_rpc_server`, `serve`.
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
pub use crate::context::{
    CapturedTensor, CapturedTensorData, LlamaContext, MemoryBreakdownEntry, TensorAccess,
    TensorBatchRow, TensorCallbackFailure, TensorCapture, TensorDataMut, TensorElementType,
    TensorFiniteValidation, TensorRowMapping, TensorSelector, TensorShape, TensorTransaction,
    TensorTransactionError, TensorTransactionHandler, TensorTransactions, TensorWriteback,
    TransactionalTensorCapture,
};
pub use crate::llama_backend::LlamaBackend;
pub use crate::llama_batch::{BatchAddError, LlamaBatch};
pub use crate::llama_batch_ext::{
    BatchEntry, BatchExtError, EmbeddingRows, LlamaBatchExt, LlamaDecisionOrder, LlamaProcessType,
    MROPE_SECTIONS,
};
pub use crate::model::params::kv_overrides::ParamOverrideValue;
pub use crate::model::params::LlamaModelParams;
pub use crate::model::{
    AddBos, LlamaBackendDevice, LlamaBackendDeviceType, LlamaChatMessage, LlamaLoraAdapter,
    LlamaModel, Special,
};
pub use crate::sampling::{LlamaSampler, LlamaSamplerParams};
pub use crate::speculative::{SpeculativeStateError, MAX_SPECULATIVE_STATE_BYTES};
pub use crate::token::data::LlamaTokenData;
pub use crate::token::data_array::LlamaTokenDataArray;
pub use crate::token::detokenizer::{DetokenizeError, StreamDetokenizer};
pub use crate::token::LlamaToken;

// ── Errors & results ────────────────────────────────────────────────────────
//
// Every error type a prelude-exported API can return is itself re-exported, so
// a caller never has to reach past the prelude just to write a `match` arm.

pub use crate::common_sampler::CommonSamplerError;
pub use crate::eagle::Eagle3SessionError;
pub use crate::mtp::MtpSessionError;
pub use crate::ngram::NgramError;
pub use crate::quantize::{FtypeNameError, QuantPreviewError};
pub use crate::runtime::RuntimeError;
pub use crate::ShimError;
pub use crate::{
    ApplyChatTemplateError, DecodeError, EmbeddingsError, EncodeError, LLamaCppError,
    LlamaContextLoadError, LlamaLoraAdapterInitError, LlamaLoraAdapterRemoveError,
    LlamaLoraAdapterSetError, LlamaModelLoadError, NewLlamaChatMessageError, ProcessError, Result,
};

// ── Memory / fit helpers ────────────────────────────────────────────────────

pub use crate::fit::{
    fit_params, get_device_memory_data, DeviceMemoryEntry, DeviceMemoryError,
    DeviceMemoryHyperParams, DeviceMemoryReport, FitParams, FitParamsError, FitParamsResult,
};

// ── Chat templates, tool calling, structured output ─────────────────────────

pub use crate::chat::{
    json_schema_to_grammar, ChatApplyParams, ChatError, ChatParams, ChatTemplates, GrammarTrigger,
    ReasoningFormat, ToolChoice,
};

// ── Sampler chains ──────────────────────────────────────────────────────────

pub use crate::common_sampler::{
    CommonSampler, CommonSamplerParams, CommonSamplerType, GrammarSource, ReasoningBudget,
    ReasoningBudgetState,
};

// ── Speculative decoding ────────────────────────────────────────────────────

pub use crate::eagle::{DFlashSession, Eagle3Session, Eagle3SessionConfig};
pub use crate::mtp::{MtpSession, MtpSessionConfig};
pub use crate::ngram::{ngram_cache_draft, ngram_simple_draft, NgramCache, NgramMap};
pub use crate::runtime::{speculative_types_from_gguf, SpeculativeType};

// ── Multimodal (feature `mtmd`) ─────────────────────────────────────────────

#[cfg(feature = "mtmd")]
pub use crate::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText,
    MtmdProgressCallback,
};

// ── Remote backend (feature `rpc`) ──────────────────────────────────────────

#[cfg(feature = "rpc")]
pub use crate::rpc::{add_rpc_server, serve, RpcBackend, RpcError};

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
