//! Safe wrapper around the C++ EAGLE-3 draft session.
//!
//! [`Eagle3Session`] drives **EAGLE-3** speculative decoding
//! (`COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3` in upstream llama.cpp). EAGLE-3
//! pairs a target model with a small, separately-trained **EAGLE-3 draft
//! model** that predicts the next tokens from hidden states extracted out of
//! the target model.
//!
//! The draft algorithm lives in upstream's `common/speculative.cpp`
//! (`common_speculative_impl_draft_eagle3`). This module wraps it through the
//! same stable C shim used for MTP (`llama-cpp-sys-4/mtp_shim/`); the two
//! techniques share an identical session lifecycle and differ only in how the
//! draft context is built.
//!
//! # EAGLE-3 vs MTP
//!
//! | | EAGLE-3 ([`Eagle3Session`]) | MTP ([`crate::mtp::MtpSession`]) |
//! |---|---|---|
//! | Draft weights | a **separate** EAGLE-3 draft model | the **same** model as the target |
//! | Draft context type | [`LlamaContextType::Default`](crate::context::params::LlamaContextType::Default) | [`LlamaContextType::Mtp`](crate::context::params::LlamaContextType::Mtp) |
//! | Requirement | draft model must expose 3 target-extract layers | target model must have MTP heads |
//!
//! # Setup
//!
//! ```ignore
//! use llama_cpp_4::context::params::LlamaContextParams;
//! use llama_cpp_4::eagle::{Eagle3Session, Eagle3SessionConfig};
//!
//! let n_draft_max = 3;
//!
//! // Target: the main model, a normal (default) context.
//! let target = main_model.new_context(&backend, LlamaContextParams::default())?;
//!
//! // Draft: a SEPARATE EAGLE-3 draft model, also a default context.
//! let draft = eagle3_model.new_context(&backend, LlamaContextParams::default())?;
//!
//! let config = Eagle3SessionConfig::new(1, n_draft_max);
//! let mut session = Eagle3Session::new_with_config(&target, &draft, config)?;
//! ```
//!
//! # Speculative loop
//!
//! Identical in shape to MTP: after each decode on the **target** context call
//! [`process`](Eagle3Session::process), then [`draft`](Eagle3Session::draft)
//! to get candidate tokens, verify them on the target, and report how many
//! were accepted with [`accept`](Eagle3Session::accept).
//!
//! ```ignore
//! target.decode(&mut batch)?;
//! session.process(&batch)?;
//! let drafts = session.draft(0, n_past, last_token)?;
//! // verify `drafts` against the target, count acceptances ...
//! session.accept(0, n_accepted)?;
//! ```
//!
//! # Hidden-state extraction
//!
//! EAGLE-3 needs the target model to expose internal hidden states. The
//! session configures the required extraction on both contexts at construction
//! time; [`need_embd`](Eagle3Session::need_embd) and
//! [`need_embd_pre_norm`](Eagle3Session::need_embd_pre_norm) report which kind
//! the active backend requested (rarely needed by callers).

use std::ptr::NonNull;

use crate::context::LlamaContext;
use crate::llama_batch::LlamaBatch;
use crate::token::LlamaToken;

/// Errors raised by the EAGLE-3 draft session.
#[derive(Debug, thiserror::Error)]
pub enum Eagle3SessionError {
    /// Returned when session init fails. The most common cause is that `draft`
    /// was not built from a valid EAGLE-3 draft model (upstream expects a draft
    /// model exposing exactly 3 target-extract layers), or that one of the
    /// contexts is incompatible.
    #[error("failed to create EAGLE-3 draft session — check that `draft` is a context over a valid EAGLE-3 draft model (3 extract layers) built from the same target")]
    Init,

    /// `process` returned false on the underlying speculative context.
    #[error("EAGLE-3 process failed (see llama.cpp logs)")]
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
    #[error("invalid EAGLE-3 session config: {0}")]
    InvalidConfig(&'static str),
}

/// Parameters for [`Eagle3Session::new_with_config`].
///
/// Maps directly to upstream `common_params_speculative_draft`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eagle3SessionConfig {
    /// Number of concurrent sequences (usually `1`).
    pub n_seq: u32,
    /// Maximum tokens drafted per [`Eagle3Session::draft`] call (`n_max` upstream).
    pub n_draft_max: i32,
    /// Minimum draft tokens to propose (`n_min` upstream, default `0`).
    pub n_min: i32,
    /// Greedy probability floor; drafts below this are dropped (`p_min` upstream, default `0.0`).
    pub p_min: f32,
}

impl Eagle3SessionConfig {
    /// Build a config with upstream-aligned defaults for `n_min` (`0`) and
    /// `p_min` (`0.0`).
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
    #[must_use]
    pub fn with_p_min(mut self, p_min: f32) -> Self {
        self.p_min = p_min;
        self
    }
}

/// Owned EAGLE-3 draft session.
///
/// Drops the underlying speculative context when freed.
///
/// # Lifetime contract (manual)
///
/// The session holds raw pointers to both the target and draft
/// [`LlamaContext`]s. **The caller must keep both contexts alive (i.e. not
/// drop them) for as long as the session exists.**
pub struct Eagle3Session {
    raw: NonNull<llama_cpp_sys_4::mtp_session>,
    config: Eagle3SessionConfig,
}

// SAFETY: the underlying C++ session owns its own state and is not tied to any
// TLS. Concurrent calls from multiple threads are NOT safe.
unsafe impl Send for Eagle3Session {}

impl Eagle3Session {
    /// Construct an EAGLE-3 draft session with upstream defaults for `n_min`
    /// and `p_min`.
    ///
    /// Equivalent to `new_with_config(target, draft, Eagle3SessionConfig::new(n_seq, n_draft_max))`.
    ///
    /// # Errors
    ///
    /// Returns [`Eagle3SessionError::Init`] or [`Eagle3SessionError::InvalidConfig`].
    pub fn new(
        target: &LlamaContext<'_>,
        draft: &LlamaContext<'_>,
        n_seq: u32,
        n_draft_max: i32,
    ) -> Result<Self, Eagle3SessionError> {
        Self::new_with_config(target, draft, Eagle3SessionConfig::new(n_seq, n_draft_max))
    }

    /// Construct an EAGLE-3 draft session with full speculative draft
    /// parameters.
    ///
    /// `target` must be a
    /// [`LlamaContextType::Default`](crate::context::params::LlamaContextType::Default)
    /// context over the main model. `draft` must be a `Default` context over a
    /// **separate EAGLE-3 draft model** trained against that target.
    ///
    /// # Errors
    ///
    /// Returns [`Eagle3SessionError::Init`] (e.g. the draft model is not a
    /// valid EAGLE-3 model) or [`Eagle3SessionError::InvalidConfig`].
    pub fn new_with_config(
        target: &LlamaContext<'_>,
        draft: &LlamaContext<'_>,
        config: Eagle3SessionConfig,
    ) -> Result<Self, Eagle3SessionError> {
        if config.n_seq == 0 {
            return Err(Eagle3SessionError::InvalidConfig("n_seq must be > 0"));
        }
        if config.n_draft_max <= 0 {
            return Err(Eagle3SessionError::InvalidConfig("n_draft_max must be > 0"));
        }

        let c_config = llama_cpp_sys_4::mtp_session_config {
            n_seq: config.n_seq,
            n_draft_max: config.n_draft_max,
            n_min: config.n_min,
            p_min: config.p_min,
            spec_type: llama_cpp_sys_4::MTP_SPEC_TYPE_EAGLE3 as i32,
        };

        let raw = unsafe {
            llama_cpp_sys_4::mtp_session_new(
                target.context.as_ptr(),
                draft.context.as_ptr(),
                &raw const c_config,
            )
        };
        let raw = NonNull::new(raw).ok_or(Eagle3SessionError::Init)?;
        Ok(Self { raw, config })
    }

    /// Session configuration passed at construction.
    #[must_use]
    pub fn config(&self) -> Eagle3SessionConfig {
        self.config
    }

    /// True when the speculative backend needs post-norm embeddings on the
    /// target context (`llama_set_embeddings`).
    #[must_use]
    pub fn need_embd(&self) -> bool {
        unsafe { llama_cpp_sys_4::mtp_session_need_embd(self.raw.as_ptr()) }
    }

    /// True when the speculative backend needs pre-norm hidden states on the
    /// target context (`llama_set_embeddings_pre_norm`).
    ///
    /// Configured automatically during session init; callers normally do not
    /// need to set it manually.
    #[must_use]
    pub fn need_embd_pre_norm(&self) -> bool {
        unsafe { llama_cpp_sys_4::mtp_session_need_embd_pre_norm(self.raw.as_ptr()) }
    }

    /// Configured maximum number of tokens drafted per [`draft`](Self::draft) call.
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

    /// Log speculative-decoding statistics (draft/accept counts and timings)
    /// via llama.cpp `LOG_INF`. Install a log callback with [`crate::log_set`]
    /// to capture output.
    pub fn print_stats(&self) {
        unsafe { llama_cpp_sys_4::mtp_session_print_stats(self.raw.as_ptr()) }
    }

    /// Optional: call once at the start of a fresh generation with the prompt
    /// tokens that were just decoded into the target context.
    ///
    /// # Errors
    ///
    /// Returns [`Eagle3SessionError::BadSeqId`] if `seq_id` is out of range.
    pub fn begin(&mut self, seq_id: i32, prompt: &[LlamaToken]) -> Result<(), Eagle3SessionError> {
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
    /// harvest the target hidden states EAGLE-3 drafts from.
    ///
    /// # Errors
    ///
    /// Returns [`Eagle3SessionError::Process`] if the underlying call fails.
    pub fn process(&mut self, batch: &LlamaBatch) -> Result<(), Eagle3SessionError> {
        let ok =
            unsafe { llama_cpp_sys_4::mtp_session_process(self.raw.as_ptr(), &batch.llama_batch) };
        if ok {
            Ok(())
        } else {
            Err(Eagle3SessionError::Process)
        }
    }

    /// Generate up to [`n_draft_max`](Self::n_draft_max) speculative tokens.
    ///
    /// `n_past` is the number of tokens already in the target KV cache for
    /// `seq_id`. `id_last` is the last token accepted on the target (usually
    /// the token you just sampled).
    ///
    /// # Errors
    ///
    /// Returns [`Eagle3SessionError::BadSeqId`] if `seq_id` is out of range.
    pub fn draft(
        &mut self,
        seq_id: i32,
        n_past: i32,
        id_last: LlamaToken,
    ) -> Result<Vec<LlamaToken>, Eagle3SessionError> {
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
    /// Pass `0` when every draft was rejected.
    ///
    /// # Errors
    ///
    /// Returns [`Eagle3SessionError::BadSeqId`] if `seq_id` is out of range.
    pub fn accept(&mut self, seq_id: i32, n_accepted: u16) -> Result<(), Eagle3SessionError> {
        self.check_seq(seq_id)?;
        unsafe {
            llama_cpp_sys_4::mtp_session_accept(self.raw.as_ptr(), seq_id, n_accepted);
        }
        Ok(())
    }

    fn check_seq(&self, seq_id: i32) -> Result<(), Eagle3SessionError> {
        if seq_id < 0 || (seq_id as u32) >= self.config.n_seq {
            return Err(Eagle3SessionError::BadSeqId {
                seq_id,
                n_seq: self.config.n_seq,
            });
        }
        Ok(())
    }
}

impl Drop for Eagle3Session {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_4::mtp_session_free(self.raw.as_ptr()) }
    }
}

impl std::fmt::Debug for Eagle3Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Eagle3Session")
            .field("config", &self.config)
            .finish()
    }
}
