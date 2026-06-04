//! Safe wrapper around the C++ MTP draft session.
//!
//! [`MtpSession`] pairs a target [`LlamaContext`] with an MTP draft
//! [`LlamaContext`] (built with
//! [`crate::context::params::LlamaContextType::Mtp`]) and drives the
//! multi-token-prediction speculative-decoding loop introduced in upstream
//! llama.cpp [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673).
//!
//! The draft algorithm lives in upstream's
//! `common/speculative.cpp` (`common_speculative_impl_draft_mtp`). This module
//! wraps it through a stable C shim in `llama-cpp-sys-4/mtp_shim/`.
//!
//! # Upstream behaviour (llama.cpp #23269+)
//!
//! After [MTP clean-up #23269](https://github.com/ggml-org/llama.cpp/pull/23269):
//!
//! - Draft sampling uses `top_k = 10` inside upstream (not configurable from Rust).
//! - [`MtpSessionConfig::p_min`] filters low-confidence draft tokens (default `0.0`).
//! - Upstream CLI default for `n_max` is `3`; set [`MtpSessionConfig::n_draft_max`]
//!   explicitly — optimal values are model/quant dependent ([`MTP.md`] on GitHub).
//!
//! [`MTP.md`]: https://github.com/eugenehp/llama-cpp-rs/blob/main/MTP.md
//!
//! # Quick start
//!
//! ```ignore
//! use llama_cpp_4::context::params::{LlamaContextParams, LlamaContextType};
//! use llama_cpp_4::mtp::{MtpSession, MtpSessionConfig};
//!
//! let n_draft_max = 3;
//!
//! let target = model.new_context(&backend, LlamaContextParams::default())?;
//! let draft = model.new_context(
//!     &backend,
//!     LlamaContextParams::default()
//!         .with_ctx_type(LlamaContextType::Mtp)
//!         .with_n_rs_seq(n_draft_max.max(4)),
//! )?;
//!
//! let config = MtpSessionConfig::new(1, n_draft_max).with_p_min(0.0);
//! let mut session = MtpSession::new_with_config(&target, &draft, config)?;
//! ```
//!
//! # Speculative loop
//!
//! For each generation step, after decoding on the **target** context:
//!
//! ```ignore
//! // 1. Target prefill or verify decode (you build the batch)
//! target.decode(&mut batch)?;
//!
//! // 2. Tell MTP about the batch just decoded on the target
//! session.process(&batch)?;
//!
//! // 3. Ask for draft tokens starting from the last accepted token
//! let drafts = session.draft(0, n_past, last_token)?;
//!
//! // 4. Verify drafts on the target (compare logits / sample — your code)
//! let n_accepted: u16 = /* ... */;
//!
//! // 5. Sync draft recurrent state with what the target accepted
//! session.accept(0, n_accepted)?;
//! ```
//!
//! Call [`MtpSession::begin`] once per fresh generation if you want upstream
//! prompt tracking (optional for MTP). Call [`MtpSession::print_stats`] when
//! finished to log draft/accept counters via llama.cpp's log callback.
//!
//! A full runnable implementation is in `examples/mtp/`.
//!
//! # Embedding requirements
//!
//! | Method | MTP typical value | Meaning |
//! |---|---|---|
//! | [`MtpSession::need_embd_pre_norm`] | `true` | Next-n hidden states (upstream name) |
//! | [`MtpSession::need_embd`] | `false` | Post-norm / seq embeddings not used |
//!
//! Rust keeps `*_pre_norm` names; upstream C API uses `*_nextn` since the Jun 2026
//! llama.cpp bump. Session init configures extraction on both contexts automatically;
//! manual [`LlamaContext::set_embeddings_pre_norm`] is rarely needed.

use std::ptr::NonNull;

use crate::context::LlamaContext;
use crate::llama_batch::LlamaBatch;
use crate::token::LlamaToken;

/// Errors raised by the MTP draft session.
#[derive(Debug, thiserror::Error)]
pub enum MtpSessionError {
    /// Returned when `mtp_session_new` fails (typically: model lacks MTP heads,
    /// or one of the contexts is incompatible).
    #[error("failed to create MTP draft session — check that ctx_dft was built with LlamaContextType::Mtp and the model has MTP heads")]
    Init,

    /// `mtp_session_process` returned false.
    #[error("mtp_session_process failed (see llama.cpp logs)")]
    Process,

    /// Caller passed a sequence id outside `[0, n_seq)`.
    #[error("sequence id {seq_id} out of range (n_seq = {n_seq})")]
    BadSeqId {
        /// the offending seq id
        seq_id: i32,
        /// configured number of sequences
        n_seq: u32,
    },

    /// Invalid session configuration (e.g. `n_draft_max <= 0`).
    #[error("invalid MTP session config: {0}")]
    InvalidConfig(&'static str),
}

/// Parameters for [`MtpSession::new_with_config`].
///
/// Maps directly to upstream `common_params_speculative_draft`.
///
/// # Examples
///
/// ```ignore
/// // Defaults: n_min = 0, p_min = 0.0 (aligned with upstream #23269+)
/// let cfg = MtpSessionConfig::new(1, 3);
///
/// // Stricter drafts: skip tokens below 10% draft-model probability
/// let cfg = MtpSessionConfig::new(1, 1).with_p_min(0.10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MtpSessionConfig {
    /// Number of concurrent sequences (usually `1`).
    pub n_seq: u32,
    /// Maximum tokens drafted per [`MtpSession::draft`] call (`n_max` upstream).
    pub n_draft_max: i32,
    /// Minimum draft tokens to propose (`n_min` upstream, default `0`).
    pub n_min: i32,
    /// Greedy probability floor; drafts below this are dropped (`p_min` upstream, default `0.0`).
    pub p_min: f32,
}

impl MtpSessionConfig {
    /// Build config with upstream-aligned defaults for `n_min` (`0`) and `p_min` (`0.0`).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let cfg = MtpSessionConfig::new(1, 3); // one sequence, up to 3 draft tokens
    /// ```
    #[must_use]
    pub fn new(n_seq: u32, n_draft_max: i32) -> Self {
        Self {
            n_seq,
            n_draft_max,
            n_min: 0,
            p_min: 0.0,
        }
    }

    /// Set minimum draft tokens (`n_min` upstream).
    #[must_use]
    pub fn with_n_min(mut self, n_min: i32) -> Self {
        self.n_min = n_min;
        self
    }

    /// Set draft probability floor (`p_min` upstream).
    ///
    /// Draft tokens whose greedy probability falls below this value are dropped.
    /// Upstream default is `0.0` after #23269 (was `0.75` in older builds).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let cfg = MtpSessionConfig::new(1, 1).with_p_min(0.10);
    /// ```
    #[must_use]
    pub fn with_p_min(mut self, p_min: f32) -> Self {
        self.p_min = p_min;
        self
    }
}

/// Owned MTP draft session.
///
/// Drops the underlying `mtp_session *` (and the C++ `common_speculative *`
/// it holds) when freed.
///
/// # Lifetime contract (manual)
///
/// The session holds raw pointers to both the target and draft
/// [`LlamaContext`]s. **The caller must keep both contexts alive (i.e. not
/// drop them) for as long as the session exists.**
pub struct MtpSession {
    raw: NonNull<llama_cpp_sys_4::mtp_session>,
    config: MtpSessionConfig,
}

// SAFETY: the underlying C++ session owns its own state and is not tied to
// any TLS. Concurrent calls from multiple threads are NOT safe.
unsafe impl Send for MtpSession {}

impl MtpSession {
    /// Construct an MTP draft session with upstream defaults for `n_min` and
    /// `p_min`.
    ///
    /// Equivalent to `new_with_config(MtpSessionConfig::new(n_seq, n_draft_max))`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut session = MtpSession::new(&target, &draft, 1, 3)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`MtpSessionError::Init`] or [`MtpSessionError::InvalidConfig`].
    pub fn new(
        target: &LlamaContext<'_>,
        draft: &LlamaContext<'_>,
        n_seq: u32,
        n_draft_max: i32,
    ) -> Result<Self, MtpSessionError> {
        Self::new_with_config(target, draft, MtpSessionConfig::new(n_seq, n_draft_max))
    }

    /// Construct an MTP draft session with full speculative draft parameters.
    ///
    /// `target` must be a [`LlamaContextType::Default`](crate::context::params::LlamaContextType::Default) context.
    /// `draft` must be a [`LlamaContextType::Mtp`](crate::context::params::LlamaContextType::Mtp) context from the same model,
    /// with [`LlamaContextParams::with_n_rs_seq`](crate::context::params::LlamaContextParams::with_n_rs_seq)
    /// `>= config.n_draft_max`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let config = MtpSessionConfig::new(1, 1)
    ///     .with_p_min(0.0); // match upstream default after #23269
    /// let session = MtpSession::new_with_config(&target, &draft, config)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`MtpSessionError::Init`] or [`MtpSessionError::InvalidConfig`].
    pub fn new_with_config(
        target: &LlamaContext<'_>,
        draft: &LlamaContext<'_>,
        config: MtpSessionConfig,
    ) -> Result<Self, MtpSessionError> {
        if config.n_seq == 0 {
            return Err(MtpSessionError::InvalidConfig("n_seq must be > 0"));
        }
        if config.n_draft_max <= 0 {
            return Err(MtpSessionError::InvalidConfig("n_draft_max must be > 0"));
        }

        let c_config = llama_cpp_sys_4::mtp_session_config {
            n_seq: config.n_seq,
            n_draft_max: config.n_draft_max,
            n_min: config.n_min,
            p_min: config.p_min,
        };

        let raw = unsafe {
            llama_cpp_sys_4::mtp_session_new(
                target.context.as_ptr(),
                draft.context.as_ptr(),
                &raw const c_config,
            )
        };
        let raw = NonNull::new(raw).ok_or(MtpSessionError::Init)?;
        Ok(Self { raw, config })
    }

    /// Session configuration passed at construction.
    #[must_use]
    pub fn config(&self) -> MtpSessionConfig {
        self.config
    }

    /// True when the speculative backend needs post-norm embeddings on the
    /// target context (`llama_set_embeddings`).
    ///
    /// MTP returns **false**; use [`Self::need_embd_pre_norm`] for MTP.
    #[must_use]
    pub fn need_embd(&self) -> bool {
        unsafe { llama_cpp_sys_4::mtp_session_need_embd(self.raw.as_ptr()) }
    }

    /// True when the speculative backend needs pre-norm hidden states on the
    /// target context (`llama_set_embeddings_pre_norm`).
    ///
    /// MTP returns **true**. Upstream configures this on both contexts during
    /// session init; callers normally do not need to set it manually.
    #[must_use]
    pub fn need_embd_pre_norm(&self) -> bool {
        unsafe { llama_cpp_sys_4::mtp_session_need_embd_pre_norm(self.raw.as_ptr()) }
    }

    /// Configured maximum number of tokens drafted per [`draft`](Self::draft)
    /// call.
    #[must_use]
    pub fn n_draft_max(&self) -> i32 {
        self.config.n_draft_max
    }

    /// Configured minimum draft tokens (`n_min`).
    #[must_use]
    pub fn n_min(&self) -> i32 {
        self.config.n_min
    }

    /// Configured draft probability floor (`p_min`).
    #[must_use]
    pub fn p_min(&self) -> f32 {
        self.config.p_min
    }

    /// Configured number of sequences.
    #[must_use]
    pub fn n_seq(&self) -> u32 {
        self.config.n_seq
    }

    /// Log speculative-decoding statistics (draft/accept counts and timings) via
    /// llama.cpp `LOG_INF`. Install a log callback with [`crate::log_set`] to
    /// capture output.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // After your generation loop:
    /// session.print_stats();
    /// ```
    pub fn print_stats(&self) {
        unsafe { llama_cpp_sys_4::mtp_session_print_stats(self.raw.as_ptr()) }
    }

    /// Optional: call once at the start of a fresh generation with the
    /// prompt tokens that were just decoded into the target context.
    ///
    /// Upstream uses this for prompt tracking; MTP speculative loops often
    /// work without it if you call [`Self::process`] after every target decode.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// session.begin(0, &prompt_tokens)?;
    /// ```
    pub fn begin(&mut self, seq_id: i32, prompt: &[LlamaToken]) -> Result<(), MtpSessionError> {
        self.check_seq(seq_id)?;
        unsafe {
            llama_cpp_sys_4::mtp_session_begin(
                self.raw.as_ptr(),
                seq_id,
                prompt.as_ptr().cast(),
                prompt.len(),
            );
        }
        Ok(())
    }

    /// Hand the session a batch that was just decoded on the target context.
    ///
    /// Call this after every successful `target.decode(batch)` so upstream can
    /// sync draft recurrent state with the target KV cache.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// target.decode(&mut batch)?;
    /// session.process(&batch)?;
    /// ```
    pub fn process(&mut self, batch: &LlamaBatch) -> Result<(), MtpSessionError> {
        let ok =
            unsafe { llama_cpp_sys_4::mtp_session_process(self.raw.as_ptr(), &batch.llama_batch) };
        if ok {
            Ok(())
        } else {
            Err(MtpSessionError::Process)
        }
    }

    /// Generate up to [`n_draft_max`](Self::n_draft_max) speculative tokens.
    ///
    /// `n_past` is the number of tokens already in the target KV cache for
    /// `seq_id`. `id_last` is the last token accepted on the target (usually
    /// the token you just sampled).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let drafts = session.draft(0, n_past, last_token)?;
    /// for draft in &drafts {
    ///     // verify each draft against target logits ...
    /// }
    /// ```
    pub fn draft(
        &mut self,
        seq_id: i32,
        n_past: i32,
        id_last: LlamaToken,
    ) -> Result<Vec<LlamaToken>, MtpSessionError> {
        self.check_seq(seq_id)?;

        let cap = self.config.n_draft_max.max(0) as usize;
        let mut buf: Vec<i32> = vec![0; cap];
        let mut out_n: i32 = cap as i32;

        unsafe {
            llama_cpp_sys_4::mtp_session_draft(
                self.raw.as_ptr(),
                seq_id,
                n_past,
                id_last.0,
                buf.as_mut_ptr(),
                &mut out_n,
            );
        }

        let n = out_n.max(0) as usize;
        buf.truncate(n);
        Ok(buf.into_iter().map(LlamaToken).collect())
    }

    /// Inform the session how many draft tokens the target verifier accepted.
    ///
    /// Pass `0` when every draft was rejected. Upstream rolls back draft
    /// recurrent state accordingly.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// session.accept(0, n_accepted)?;
    /// ```
    pub fn accept(&mut self, seq_id: i32, n_accepted: u16) -> Result<(), MtpSessionError> {
        self.check_seq(seq_id)?;
        unsafe {
            llama_cpp_sys_4::mtp_session_accept(self.raw.as_ptr(), seq_id, n_accepted);
        }
        Ok(())
    }

    fn check_seq(&self, seq_id: i32) -> Result<(), MtpSessionError> {
        if seq_id < 0 || (seq_id as u32) >= self.config.n_seq {
            return Err(MtpSessionError::BadSeqId {
                seq_id,
                n_seq: self.config.n_seq,
            });
        }
        Ok(())
    }
}

impl Drop for MtpSession {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_4::mtp_session_free(self.raw.as_ptr()) }
    }
}

impl std::fmt::Debug for MtpSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpSession")
            .field("config", &self.config)
            .field("need_embd_pre_norm", &self.need_embd_pre_norm())
            .finish()
    }
}
