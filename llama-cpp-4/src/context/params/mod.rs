//! A safe wrapper around `llama_context_params`.
//!
//! Use [`LlamaContextParams`] to configure context size, batching, KV layout,
//! `RoPE` / `YaRN` scaling, flash attention, per-sequence samplers, and pairing
//! with another context (`ctx_other`).
mod advanced;
mod types;

pub use types::*;

use std::num::NonZeroU32;

use thiserror::Error;

use crate::sampling::LlamaSampler;

/// Error returned when [`LlamaContextParams::try_clone`] cannot duplicate state.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParamsCloneError {
    /// Per-sequence sampler chains cannot be duplicated.
    #[error("cannot clone params that own per-sequence sampler chains")]
    SamplerChains,
}

/// Builder for [`llama_context_params`](llama_cpp_sys_4::llama_context_params).
///
/// Construct with [`Default::default()`], chain `with_*` setters, then pass the
/// value to [`crate::model::LlamaModel::new_context`]. Getter methods mirror
/// the fields that exist on the underlying C struct.
///
/// # Sampler ownership
///
/// [`Self::with_sampler_seq_configs`] stores owned [`LlamaSampler`] chains inside
/// this struct until the context is created. [`Clone`] clears sampler configs
/// because the underlying chains cannot be duplicated safely.
///
/// # Examples
///
/// ```rust
/// # use std::num::NonZeroU32;
/// use llama_cpp_4::context::params::LlamaContextParams;
///
/// let ctx_params = LlamaContextParams::default()
///     .with_n_ctx(NonZeroU32::new(2048));
///
/// assert_eq!(ctx_params.n_ctx(), NonZeroU32::new(2048));
/// ```
#[derive(Debug)]
#[allow(
    missing_docs,
    clippy::struct_excessive_bools,
    clippy::module_name_repetitions
)]
pub struct LlamaContextParams {
    pub(crate) context_params: llama_cpp_sys_4::llama_context_params,
    /// When `true`, the `TurboQuant` attention rotation (PR #21038) will be
    /// disabled for any context created from these params.
    pub(crate) attn_rot_disabled: bool,
    /// Keeps sampler chains alive while `context_params.samplers` points at them.
    owned_samplers: Vec<LlamaSampler>,
    sampler_configs: Vec<llama_cpp_sys_4::llama_sampler_seq_config>,
}

/// SAFETY: we do not currently allow setting or reading the pointers that cause this to not be automatically send or sync.
unsafe impl Send for LlamaContextParams {}
unsafe impl Sync for LlamaContextParams {}

impl LlamaContextParams {
    /// Set the side of the context
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::num::NonZeroU32;
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default();
    /// let params = params.with_n_ctx(NonZeroU32::new(2048));
    /// assert_eq!(params.n_ctx(), NonZeroU32::new(2048));
    /// ```
    #[must_use]
    pub fn with_n_ctx(mut self, n_ctx: Option<NonZeroU32>) -> Self {
        self.context_params.n_ctx = n_ctx.map_or(0, std::num::NonZeroU32::get);
        self
    }

    /// Get the size of the context.
    ///
    /// [`None`] if the context size is specified by the model and not the context.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_4::context::params::LlamaContextParams::default();
    /// assert_eq!(params.n_ctx(), std::num::NonZeroU32::new(512));
    #[must_use]
    pub fn n_ctx(&self) -> Option<NonZeroU32> {
        NonZeroU32::new(self.context_params.n_ctx)
    }

    /// Set the maximum number of independent sequence states in the context.
    ///
    /// This maps to llama.cpp's `llama_context_params.n_seq_max` and must match
    /// the highest sequence id used by batched decoding.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_n_seq_max(16);
    /// assert_eq!(params.n_seq_max(), 16);
    /// ```
    #[must_use]
    pub fn with_n_seq_max(mut self, n_seq_max: u32) -> Self {
        self.context_params.n_seq_max = n_seq_max.max(1);
        self
    }

    /// Get the configured maximum number of independent sequence states.
    #[must_use]
    pub fn n_seq_max(&self) -> u32 {
        self.context_params.n_seq_max
    }

    /// Set the `n_batch`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::num::NonZeroU32;
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_n_batch(2048);
    /// assert_eq!(params.n_batch(), 2048);
    /// ```
    #[must_use]
    pub fn with_n_batch(mut self, n_batch: u32) -> Self {
        self.context_params.n_batch = n_batch;
        self
    }

    /// Get the `n_batch`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default();
    /// assert_eq!(params.n_batch(), 2048);
    /// ```
    #[must_use]
    pub fn n_batch(&self) -> u32 {
        self.context_params.n_batch
    }

    /// Set the `n_ubatch`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::num::NonZeroU32;
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_n_ubatch(512);
    /// assert_eq!(params.n_ubatch(), 512);
    /// ```
    #[must_use]
    pub fn with_n_ubatch(mut self, n_ubatch: u32) -> Self {
        self.context_params.n_ubatch = n_ubatch;
        self
    }

    /// Get the `n_ubatch`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default();
    /// assert_eq!(params.n_ubatch(), 512);
    /// ```
    #[must_use]
    pub fn n_ubatch(&self) -> u32 {
        self.context_params.n_ubatch
    }

    /// Set the context type (e.g. [`LlamaContextType::Mtp`] for the draft context in
    /// [`crate::mtp::MtpSession`]).
    #[must_use]
    pub fn with_ctx_type(mut self, ctx_type: LlamaContextType) -> Self {
        self.context_params.ctx_type = ctx_type.into();
        self
    }

    /// Get the configured context type.
    #[must_use]
    pub fn ctx_type(&self) -> LlamaContextType {
        self.context_params.ctx_type.into()
    }

    /// Set the number of recurrent-state snapshots per sequence (MTP rollback).
    ///
    /// Must be `>=` [`MtpSessionConfig::n_draft_max`](crate::mtp::MtpSessionConfig::n_draft_max)
    /// on the draft context. See [`crate::mtp`].
    #[must_use]
    pub fn with_n_rs_seq(mut self, n_rs_seq: u32) -> Self {
        self.context_params.n_rs_seq = n_rs_seq;
        self
    }

    /// Get the number of recurrent-state snapshots per sequence used for MTP rollback.
    #[must_use]
    pub fn n_rs_seq(&self) -> u32 {
        self.context_params.n_rs_seq
    }

    /// Set the `offload_kqv` parameter to control offloading KV cache & KQV ops to GPU
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///     .with_offload_kqv(false);
    /// assert_eq!(params.offload_kqv(), false);
    /// ```
    #[must_use]
    pub fn with_offload_kqv(mut self, enabled: bool) -> Self {
        self.context_params.offload_kqv = enabled;
        self
    }

    /// Get the `offload_kqv` parameter
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default();
    /// assert_eq!(params.offload_kqv(), true);
    /// ```
    #[must_use]
    pub fn offload_kqv(&self) -> bool {
        self.context_params.offload_kqv
    }

    /// Set the type of rope scaling.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::{LlamaContextParams, RopeScalingType};
    /// let params = LlamaContextParams::default()
    ///     .with_rope_scaling_type(RopeScalingType::Linear);
    /// assert_eq!(params.rope_scaling_type(), RopeScalingType::Linear);
    /// ```
    #[must_use]
    pub fn with_rope_scaling_type(mut self, rope_scaling_type: RopeScalingType) -> Self {
        self.context_params.rope_scaling_type = i32::from(rope_scaling_type);
        self
    }

    /// Get the type of rope scaling.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_4::context::params::LlamaContextParams::default();
    /// assert_eq!(params.rope_scaling_type(), llama_cpp_4::context::params::RopeScalingType::Unspecified);
    /// ```
    #[must_use]
    pub fn rope_scaling_type(&self) -> RopeScalingType {
        RopeScalingType::from(self.context_params.rope_scaling_type)
    }

    /// Set the rope frequency base.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///    .with_rope_freq_base(0.5);
    /// assert_eq!(params.rope_freq_base(), 0.5);
    /// ```
    #[must_use]
    pub fn with_rope_freq_base(mut self, rope_freq_base: f32) -> Self {
        self.context_params.rope_freq_base = rope_freq_base;
        self
    }

    /// Get the rope frequency base.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_4::context::params::LlamaContextParams::default();
    /// assert_eq!(params.rope_freq_base(), 0.0);
    /// ```
    #[must_use]
    pub fn rope_freq_base(&self) -> f32 {
        self.context_params.rope_freq_base
    }

    /// Set the rope frequency scale.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///   .with_rope_freq_scale(0.5);
    /// assert_eq!(params.rope_freq_scale(), 0.5);
    /// ```
    #[must_use]
    pub fn with_rope_freq_scale(mut self, rope_freq_scale: f32) -> Self {
        self.context_params.rope_freq_scale = rope_freq_scale;
        self
    }

    /// Get the rope frequency scale.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_4::context::params::LlamaContextParams::default();
    /// assert_eq!(params.rope_freq_scale(), 0.0);
    /// ```
    #[must_use]
    pub fn rope_freq_scale(&self) -> f32 {
        self.context_params.rope_freq_scale
    }

    /// Get the number of threads.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_4::context::params::LlamaContextParams::default();
    /// assert_eq!(params.n_threads(), 4);
    /// ```
    #[must_use]
    pub fn n_threads(&self) -> i32 {
        self.context_params.n_threads
    }

    /// Get the number of threads allocated for batches.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_4::context::params::LlamaContextParams::default();
    /// assert_eq!(params.n_threads_batch(), 4);
    /// ```
    #[must_use]
    pub fn n_threads_batch(&self) -> i32 {
        self.context_params.n_threads_batch
    }

    /// Set the number of threads.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///    .with_n_threads(8);
    /// assert_eq!(params.n_threads(), 8);
    /// ```
    #[must_use]
    pub fn with_n_threads(mut self, n_threads: i32) -> Self {
        self.context_params.n_threads = n_threads;
        self
    }

    /// Set the number of threads allocated for batches.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///    .with_n_threads_batch(8);
    /// assert_eq!(params.n_threads_batch(), 8);
    /// ```
    #[must_use]
    pub fn with_n_threads_batch(mut self, n_threads: i32) -> Self {
        self.context_params.n_threads_batch = n_threads;
        self
    }

    /// Check whether embeddings are enabled
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_4::context::params::LlamaContextParams::default();
    /// assert!(!params.embeddings());
    /// ```
    #[must_use]
    pub fn embeddings(&self) -> bool {
        self.context_params.embeddings
    }

    /// Enable the use of embeddings
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default()
    ///    .with_embeddings(true);
    /// assert!(params.embeddings());
    /// ```
    #[must_use]
    pub fn with_embeddings(mut self, embedding: bool) -> Self {
        self.context_params.embeddings = embedding;
        self
    }

    /// Set the evaluation callback.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// extern "C" fn cb_eval_fn(
    ///     t: *mut llama_cpp_sys_4::ggml_tensor,
    ///     ask: bool,
    ///     user_data: *mut std::ffi::c_void,
    /// ) -> bool {
    ///     false
    /// }
    ///
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default().with_cb_eval(Some(cb_eval_fn));
    /// ```
    #[must_use]
    pub fn with_cb_eval(
        mut self,
        cb_eval: llama_cpp_sys_4::ggml_backend_sched_eval_callback,
    ) -> Self {
        self.context_params.cb_eval = cb_eval;
        self
    }

    /// Set the evaluation callback user data.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default();
    /// let user_data = std::ptr::null_mut();
    /// let params = params.with_cb_eval_user_data(user_data);
    /// ```
    #[must_use]
    pub fn with_cb_eval_user_data(mut self, cb_eval_user_data: *mut std::ffi::c_void) -> Self {
        self.context_params.cb_eval_user_data = cb_eval_user_data;
        self
    }

    /// Attach a [`TensorCapture`](super::tensor_capture::TensorCapture) to
    /// intercept intermediate tensor outputs during [`crate::LlamaContext::decode`].
    ///
    /// Sets `cb_eval` to copy tensors matching the capture filter (layer outputs,
    /// named nodes, prefix, or all). After `decode()`, read results from the
    /// capture — see [`crate::TensorCapture`] and [`crate::context::tensor_capture`].
    ///
    /// The capture must outlive the context. Call [`TensorCapture::clear`](crate::TensorCapture::clear) before
    /// reusing it on another batch.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_4::prelude::*;
    ///
    /// fn main() {
    ///     let backend = LlamaBackend::init().unwrap();
    ///     let model = LlamaModel::load_from_file(
    ///         &backend,
    ///         "model.gguf",
    ///         &LlamaModelParams::default(),
    ///     )
    ///     .unwrap();
    ///
    ///     let mut capture = TensorCapture::for_layers(&[13, 20, 27]);
    ///     let ctx_params = LlamaContextParams::default().with_tensor_capture(&mut capture);
    ///     let _ctx = model.new_context(&backend, ctx_params).unwrap();
    /// }
    /// ```
    #[must_use]
    pub fn with_tensor_capture(self, capture: &mut super::tensor_capture::TensorCapture) -> Self {
        self.with_cb_eval(Some(super::tensor_capture::tensor_capture_callback))
            .with_cb_eval_user_data(
                std::ptr::from_mut::<super::tensor_capture::TensorCapture>(capture)
                    .cast::<std::ffi::c_void>(),
            )
    }

    /// Set the storage type for the **K** (key) KV cache tensors.
    ///
    /// The default is `GgmlType::F16`.  Quantized types like `GgmlType::Q5_0`
    /// or `GgmlType::Q4_0` reduce VRAM usage significantly; combining them with
    /// `TurboQuant` attention rotation (the default) keeps quality high.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// use llama_cpp_4::quantize::GgmlType;
    /// let params = LlamaContextParams::default()
    ///     .with_cache_type_k(GgmlType::Q5_0);
    /// ```
    #[must_use]
    pub fn with_cache_type_k(mut self, ty: crate::quantize::GgmlType) -> Self {
        self.context_params.type_k = ty as llama_cpp_sys_4::ggml_type;
        self
    }

    /// Get the K-cache storage type.
    #[must_use]
    pub fn cache_type_k(&self) -> llama_cpp_sys_4::ggml_type {
        self.context_params.type_k
    }

    /// Set the storage type for the **V** (value) KV cache tensors.
    ///
    /// See [`with_cache_type_k`](Self::with_cache_type_k) for details.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// use llama_cpp_4::quantize::GgmlType;
    /// let params = LlamaContextParams::default()
    ///     .with_cache_type_v(GgmlType::Q5_0);
    /// ```
    #[must_use]
    pub fn with_cache_type_v(mut self, ty: crate::quantize::GgmlType) -> Self {
        self.context_params.type_v = ty as llama_cpp_sys_4::ggml_type;
        self
    }

    /// Get the V-cache storage type.
    #[must_use]
    pub fn cache_type_v(&self) -> llama_cpp_sys_4::ggml_type {
        self.context_params.type_v
    }

    /// Control the `TurboQuant` attention-rotation feature (llama.cpp PR #21038).
    ///
    /// By default, llama.cpp applies a Hadamard rotation to Q/K/V tensors
    /// before writing them into the KV cache.  This significantly improves
    /// quantized KV-cache quality at near-zero overhead, and is enabled
    /// automatically for models whose head dimension is a power of two.
    ///
    /// Set `disabled = true` to opt out (equivalent to `LLAMA_ATTN_ROT_DISABLE=1`).
    /// The env-var is applied just before the context is created and restored
    /// afterwards, so this is safe to call from a single thread.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// // Disable rotation for this context only:
    /// let params = LlamaContextParams::default().with_attn_rot_disabled(true);
    /// assert!(params.attn_rot_disabled());
    /// ```
    #[must_use]
    pub fn with_attn_rot_disabled(mut self, disabled: bool) -> Self {
        self.attn_rot_disabled = disabled;
        self
    }

    /// Returns `true` if `TurboQuant` attention rotation is disabled for this context.
    ///
    /// ```rust
    /// let params = llama_cpp_4::context::params::LlamaContextParams::default();
    /// assert!(!params.attn_rot_disabled());
    /// ```
    #[must_use]
    pub fn attn_rot_disabled(&self) -> bool {
        self.attn_rot_disabled
    }

    /// Set the type of pooling.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::{LlamaContextParams, LlamaPoolingType};
    /// let params = LlamaContextParams::default()
    ///     .with_pooling_type(LlamaPoolingType::Last);
    /// assert_eq!(params.pooling_type(), LlamaPoolingType::Last);
    /// ```
    #[must_use]
    pub fn with_pooling_type(mut self, pooling_type: LlamaPoolingType) -> Self {
        self.context_params.pooling_type = i32::from(pooling_type);
        self
    }

    /// Get the type of pooling.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let params = llama_cpp_4::context::params::LlamaContextParams::default();
    /// assert_eq!(params.pooling_type(), llama_cpp_4::context::params::LlamaPoolingType::Unspecified);
    /// ```
    #[must_use]
    pub fn pooling_type(&self) -> LlamaPoolingType {
        LlamaPoolingType::from(self.context_params.pooling_type)
    }

    /// Clone these params, failing when sampler chains are attached.
    ///
    /// Prefer this over [`Clone::clone`] when you need to detect dropped sampler
    /// configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ParamsCloneError::SamplerChains`] when per-sequence sampler
    /// chains are attached and cannot be duplicated.
    pub fn try_clone(&self) -> Result<Self, ParamsCloneError> {
        if self.sampler_configs.is_empty() {
            Ok(self.clone())
        } else {
            Err(ParamsCloneError::SamplerChains)
        }
    }
}

/// Default parameters for `LlamaContext`. (as defined in llama.cpp by `llama_context_default_params`)
/// ```
/// # use std::num::NonZeroU32;
/// use llama_cpp_4::context::params::{LlamaContextParams, RopeScalingType};
/// let params = LlamaContextParams::default();
/// assert_eq!(params.n_ctx(), NonZeroU32::new(512), "n_ctx should be 512");
/// assert_eq!(params.rope_scaling_type(), RopeScalingType::Unspecified);
/// ```
impl Default for LlamaContextParams {
    fn default() -> Self {
        let context_params = unsafe { llama_cpp_sys_4::llama_context_default_params() };
        Self {
            context_params,
            attn_rot_disabled: false,
            owned_samplers: Vec::new(),
            sampler_configs: Vec::new(),
        }
    }
}

/// Duplicate context params for reuse.
///
/// Sampler chains attached via [`LlamaContextParams::with_sampler_seq_configs`]
/// are **not** cloned — the copy clears `samplers` / `n_samplers` because the
/// underlying C chains cannot be duplicated safely.
impl Clone for LlamaContextParams {
    fn clone(&self) -> Self {
        let mut context_params = self.context_params;
        // Sampler chains cannot be duplicated here; cloned params omit them.
        context_params.samplers = std::ptr::null_mut();
        context_params.n_samplers = 0;
        Self {
            context_params,
            attn_rot_disabled: self.attn_rot_disabled,
            owned_samplers: Vec::new(),
            sampler_configs: Vec::new(),
        }
    }
}
