use super::{LlamaAttentionType, LlamaContextParams, LlamaFlashAttnType};
use crate::sampling::LlamaSampler;

impl LlamaContextParams {
    /// Set the flash-attention mode (`Auto`, `Enabled`, or `Disabled`).
    ///
    /// Maps to `llama_context_params.flash_attn_type`. Use
    /// [`LlamaFlashAttnType::Auto`] to match llama.cpp defaults.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::{LlamaContextParams, LlamaFlashAttnType};
    /// let params = LlamaContextParams::default()
    ///     .with_flash_attn_type(LlamaFlashAttnType::Auto);
    /// assert_eq!(params.flash_attn_type(), LlamaFlashAttnType::Auto);
    /// ```
    #[must_use]
    pub fn with_flash_attn_type(mut self, flash_attn_type: LlamaFlashAttnType) -> Self {
        self.context_params.flash_attn_type = flash_attn_type.into();
        self
    }

    /// Get the configured flash-attention mode.
    #[must_use]
    pub fn flash_attn_type(&self) -> LlamaFlashAttnType {
        LlamaFlashAttnType::from(self.context_params.flash_attn_type)
    }

    /// Set the attention type used when extracting embeddings.
    ///
    /// Maps to `llama_context_params.attention_type`. Embedding models often
    /// need [`LlamaAttentionType::NonCausal`]; generative decoding uses
    /// [`LlamaAttentionType::Causal`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::{LlamaAttentionType, LlamaContextParams};
    /// let params = LlamaContextParams::default()
    ///     .with_attention_type(LlamaAttentionType::Causal);
    /// assert_eq!(params.attention_type(), LlamaAttentionType::Causal);
    /// ```
    #[must_use]
    pub fn with_attention_type(mut self, attention_type: LlamaAttentionType) -> Self {
        self.context_params.attention_type = attention_type.into();
        self
    }

    /// Get the attention type used when extracting embeddings.
    #[must_use]
    pub fn attention_type(&self) -> LlamaAttentionType {
        LlamaAttentionType::from(self.context_params.attention_type)
    }

    /// Set the maximum number of outputs per micro-batch.
    ///
    /// Maps to `llama_context_params.n_outputs_max`. When `0`, llama.cpp uses
    /// `n_batch` as the cap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default().with_n_outputs_max(256);
    /// assert_eq!(params.n_outputs_max(), 256);
    /// ```
    #[must_use]
    pub fn with_n_outputs_max(mut self, n_outputs_max: u32) -> Self {
        self.context_params.n_outputs_max = n_outputs_max;
        self
    }

    /// Get the maximum number of outputs per micro-batch.
    #[must_use]
    pub fn n_outputs_max(&self) -> u32 {
        self.context_params.n_outputs_max
    }

    /// Use a unified KV buffer across input sequences.
    ///
    /// Maps to `llama_context_params.kv_unified`. Disabling can improve
    /// throughput for batched decoding when sequences do not share a long prefix.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default().with_kv_unified(false);
    /// assert!(!params.kv_unified());
    /// ```
    #[must_use]
    pub fn with_kv_unified(mut self, kv_unified: bool) -> Self {
        self.context_params.kv_unified = kv_unified;
        self
    }

    /// Returns `true` when a unified KV buffer is enabled.
    #[must_use]
    pub fn kv_unified(&self) -> bool {
        self.context_params.kv_unified
    }

    /// Use a full-size sliding-window-attention (SWA) KV cache.
    ///
    /// Maps to `llama_context_params.swa_full`. When `false` and `n_seq_max > 1`,
    /// llama.cpp may use a smaller per-sequence SWA window for better performance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default().with_swa_full(true);
    /// assert!(params.swa_full());
    /// ```
    #[must_use]
    pub fn with_swa_full(mut self, swa_full: bool) -> Self {
        self.context_params.swa_full = swa_full;
        self
    }

    /// Returns `true` when full SWA cache is enabled.
    #[must_use]
    pub fn swa_full(&self) -> bool {
        self.context_params.swa_full
    }

    /// Offload eligible host tensor operations to the active device.
    ///
    /// Maps to `llama_context_params.op_offload`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default().with_op_offload(true);
    /// assert!(params.op_offload());
    /// ```
    #[must_use]
    pub fn with_op_offload(mut self, op_offload: bool) -> Self {
        self.context_params.op_offload = op_offload;
        self
    }

    /// Returns `true` when host tensor ops are offloaded to device.
    #[must_use]
    pub fn op_offload(&self) -> bool {
        self.context_params.op_offload
    }

    /// Pair this context with another for shared memory or cross-context results.
    ///
    /// Maps to `llama_context_params.ctx_other`. The paired context is returned
    /// by [`crate::context::LlamaContext::ctx_other`] after creation.
    ///
    /// `other` must remain alive until [`crate::model::LlamaModel::new_context`]
    /// returns.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let target = model.new_context(&backend, LlamaContextParams::default())?;
    /// let draft = model.new_context(
    ///     &backend,
    ///     LlamaContextParams::default().with_ctx_other(&target),
    /// )?;
    /// ```
    #[must_use]
    pub fn with_ctx_other(mut self, other: &crate::context::LlamaContext<'_>) -> Self {
        self.context_params.ctx_other = other.context.as_ptr();
        self
    }

    /// Set `YaRN` extrapolation mix factor.
    ///
    /// Maps to `llama_context_params.yarn_ext_factor`. Negative values use the
    /// model default. Only meaningful when [`super::RopeScalingType::Yarn`] is active.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default().with_yarn_ext_factor(1.0);
    /// assert_eq!(params.yarn_ext_factor(), 1.0);
    /// ```
    #[must_use]
    pub fn with_yarn_ext_factor(mut self, yarn_ext_factor: f32) -> Self {
        self.context_params.yarn_ext_factor = yarn_ext_factor;
        self
    }

    /// Get `YaRN` extrapolation mix factor (`yarn_ext_factor`).
    #[must_use]
    pub fn yarn_ext_factor(&self) -> f32 {
        self.context_params.yarn_ext_factor
    }

    /// Set `YaRN` magnitude scaling factor.
    ///
    /// Maps to `llama_context_params.yarn_attn_factor`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default().with_yarn_attn_factor(1.0);
    /// assert_eq!(params.yarn_attn_factor(), 1.0);
    /// ```
    #[must_use]
    pub fn with_yarn_attn_factor(mut self, yarn_attn_factor: f32) -> Self {
        self.context_params.yarn_attn_factor = yarn_attn_factor;
        self
    }

    /// Get `YaRN` magnitude scaling factor (`yarn_attn_factor`).
    #[must_use]
    pub fn yarn_attn_factor(&self) -> f32 {
        self.context_params.yarn_attn_factor
    }

    /// Set `YaRN` low correction dimension (`yarn_beta_fast`).
    ///
    /// Maps to `llama_context_params.yarn_beta_fast`.
    #[must_use]
    pub fn with_yarn_beta_fast(mut self, yarn_beta_fast: f32) -> Self {
        self.context_params.yarn_beta_fast = yarn_beta_fast;
        self
    }

    /// Get `YaRN` low correction dimension.
    #[must_use]
    pub fn yarn_beta_fast(&self) -> f32 {
        self.context_params.yarn_beta_fast
    }

    /// Set `YaRN` high correction dimension (`yarn_beta_slow`).
    ///
    /// Maps to `llama_context_params.yarn_beta_slow`.
    #[must_use]
    pub fn with_yarn_beta_slow(mut self, yarn_beta_slow: f32) -> Self {
        self.context_params.yarn_beta_slow = yarn_beta_slow;
        self
    }

    /// Get `YaRN` high correction dimension.
    #[must_use]
    pub fn yarn_beta_slow(&self) -> f32 {
        self.context_params.yarn_beta_slow
    }

    /// Set `YaRN` original context size.
    ///
    /// Maps to `llama_context_params.yarn_orig_ctx`. `0` uses the model default.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default().with_yarn_orig_ctx(8192);
    /// assert_eq!(params.yarn_orig_ctx(), 8192);
    /// ```
    #[must_use]
    pub fn with_yarn_orig_ctx(mut self, yarn_orig_ctx: u32) -> Self {
        self.context_params.yarn_orig_ctx = yarn_orig_ctx;
        self
    }

    /// Get `YaRN` original context size (`yarn_orig_ctx`).
    #[must_use]
    pub fn yarn_orig_ctx(&self) -> u32 {
        self.context_params.yarn_orig_ctx
    }

    /// Disable performance timing collection for this context.
    ///
    /// Maps to `llama_context_params.no_perf`. When `true`, calls such as
    /// [`crate::context::LlamaContext::timings`] return empty counters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// let params = LlamaContextParams::default().with_no_perf(true);
    /// assert!(params.no_perf());
    /// ```
    #[must_use]
    pub fn with_no_perf(mut self, no_perf: bool) -> Self {
        self.context_params.no_perf = no_perf;
        self
    }

    /// Returns `true` when perf timings are disabled for this context.
    #[must_use]
    pub fn no_perf(&self) -> bool {
        self.context_params.no_perf
    }

    /// Register an abort callback checked during `decode()` on CPU backends.
    ///
    /// Maps to `llama_context_params.abort_callback` / `abort_callback_data`.
    /// The callback is invoked periodically during long decodes; return a
    /// non-zero value to stop the current operation.
    ///
    /// `user_data` is passed through unchanged and must remain valid for the
    /// lifetime of any context created from these params.
    #[must_use]
    pub fn with_abort_callback(
        mut self,
        callback: llama_cpp_sys_4::ggml_abort_callback,
        user_data: *mut std::ffi::c_void,
    ) -> Self {
        self.context_params.abort_callback = callback;
        self.context_params.abort_callback_data = user_data;
        self
    }

    /// Assign per-sequence backend sampler chains.
    ///
    /// Maps to `llama_context_params.samplers` / `n_samplers`. Each
    /// [`LlamaSampler`] must be a sampler **chain** created with
    /// `llama_sampler_chain_init`. The samplers are kept alive inside these
    /// params until [`crate::model::LlamaModel::new_context`] returns.
    ///
    /// Pair sequence ids with the chains that should run when decoding those
    /// sequences on the backend.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use llama_cpp_4::context::params::LlamaContextParams;
    /// use llama_cpp_4::sampling::LlamaSampler;
    ///
    /// let chain = LlamaSampler::chain_default(&model)?;
    /// let params = LlamaContextParams::default()
    ///     .with_sampler_seq_configs([(0, chain)]);
    /// assert_eq!(params.n_sampler_seq_configs(), 1);
    /// ```
    #[must_use]
    pub fn with_sampler_seq_configs(
        mut self,
        configs: impl IntoIterator<Item = (i32, LlamaSampler)>,
    ) -> Self {
        self.owned_samplers.clear();
        self.sampler_configs.clear();

        for (seq_id, sampler) in configs {
            self.sampler_configs
                .push(llama_cpp_sys_4::llama_sampler_seq_config {
                    seq_id,
                    sampler: sampler.sampler.as_ptr(),
                });
            self.owned_samplers.push(sampler);
        }

        if self.sampler_configs.is_empty() {
            self.context_params.samplers = std::ptr::null_mut();
            self.context_params.n_samplers = 0;
        } else {
            self.context_params.samplers = self.sampler_configs.as_mut_ptr();
            self.context_params.n_samplers = self.sampler_configs.len();
        }

        self
    }

    /// Number of per-sequence sampler configs attached to these params.
    ///
    /// Returns `0` when no chains were set or after [`Clone`] (sampler chains
    /// are not duplicated).
    #[must_use]
    pub fn n_sampler_seq_configs(&self) -> usize {
        self.sampler_configs.len()
    }
}
