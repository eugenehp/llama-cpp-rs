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
//! let mut target = model.new_context(&backend, LlamaContextParams::default())?;
//! let mut draft = model.new_context(
//!     &backend,
//!     LlamaContextParams::default()
//!         .with_ctx_type(LlamaContextType::Mtp)
//!         .with_n_rs_seq(n_draft_max.max(4)),
//! )?;
//!
//! let config = MtpSessionConfig::new(1, n_draft_max).with_p_min(0.0);
//! let mut session = MtpSession::new_with_config(&mut target, &mut draft, config)?;
//! ```
//!
//! # Speculative loop
//!
//! For each generation step, after decoding on the **target** context:
//!
//! ```ignore
//! // 1. Target prefill or verify decode (you build the batch)
//! session.decode_target_and_process(&mut batch)?;
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
//! # Multi-head `NextN` (Step3.5+)
//!
//! When [`crate::model::LlamaModel::n_layer_nextn`] returns a value greater than `1`, set the
//! draft context head before each [`MtpSession::draft`] call:
//!
//! ```ignore
//! for head in 0..model.n_layer_nextn() {
//!     draft.set_nextn_layer_offset(head);
//!     let drafts = session.draft(0, n_past, last_token)?;
//!     // verify on target ...
//! }
//! draft.set_nextn_layer_offset(0); // restore default
//! ```
//!

use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::context::params::LlamaContextType;
use crate::context::LlamaContext;
use crate::llama_batch::LlamaBatch;
use crate::speculative::MAX_SPECULATIVE_PROMPT_TOKENS;
use crate::speculative::{
    capture_state, restore_state, validate_config, validate_context_capacities,
    SpeculativeContextCapacity, SpeculativeStateError,
};
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

    /// Native prompt initialization failed or raised a contained exception.
    #[error("mtp_session_begin failed")]
    Begin,

    /// Native draft generation failed or raised a contained exception.
    #[error("mtp_session_draft failed")]
    Draft,

    /// Native proposal acceptance failed or raised a contained exception.
    #[error("mtp_session_accept failed")]
    Accept,

    /// Prompt storage exceeds the safe speculative-session bound.
    #[error("prompt has {size} tokens, exceeding the {maximum}-token bound")]
    PromptTooLong {
        /// Caller-supplied prompt-token count.
        size: usize,
        /// Inclusive safe prompt-token bound.
        maximum: usize,
    },

    /// The supplied contexts do not satisfy the native MTP contract.
    #[error("incompatible MTP contexts: {0}")]
    IncompatibleContexts(&'static str),

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

    /// The target context failed to decode.
    #[error("target decode failed: {0}")]
    Decode(#[from] crate::DecodeError),

    /// An operation requires all draft proposals to be completed first.
    #[error("sequence {seq_id} still has an unaccepted draft proposal")]
    ProposalPending {
        /// Sequence with a pending proposal.
        seq_id: i32,
    },

    /// `accept` was called without a preceding nonempty draft.
    #[error("sequence {seq_id} has no draft proposal to accept")]
    NoPendingProposal {
        /// Sequence without a pending proposal.
        seq_id: i32,
    },

    /// The accepted prefix exceeds the proposal length.
    #[error("accepted {accepted} tokens from a {proposed}-token proposal")]
    AcceptedTooMany {
        /// Accepted prefix length.
        accepted: u16,
        /// Exact proposal length.
        proposed: usize,
    },

    /// Exact speculative-state capture or restore failed.
    #[error(transparent)]
    State(#[from] SpeculativeStateError),
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
/// The session exclusively borrows both contexts for its Rust lifetime, so
/// neither can be moved, accessed mutably, or dropped while native code retains
/// their pointers. It is deliberately neither `Send` nor `Sync`.
pub struct MtpSession<'ctx, 'model> {
    raw: NonNull<llama_cpp_sys_4::mtp_session>,
    config: MtpSessionConfig,
    target: &'ctx mut LlamaContext<'model>,
    draft: &'ctx mut LlamaContext<'model>,
    pending_proposals: Vec<Option<usize>>,
    not_send_sync: PhantomData<Rc<()>>,
}

impl<'ctx, 'model> MtpSession<'ctx, 'model> {
    /// Construct an MTP draft session with upstream defaults for `n_min` and
    /// `p_min`.
    ///
    /// Equivalent to `new_with_config(MtpSessionConfig::new(n_seq, n_draft_max))`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut session = MtpSession::new(&mut target, &mut draft, 1, 3)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`MtpSessionError::Init`] or [`MtpSessionError::InvalidConfig`].
    pub fn new(
        target: &'ctx mut LlamaContext<'model>,
        draft: &'ctx mut LlamaContext<'model>,
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
    /// let session = MtpSession::new_with_config(&mut target, &mut draft, config)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`MtpSessionError::Init`] or [`MtpSessionError::InvalidConfig`].
    pub fn new_with_config(
        target: &'ctx mut LlamaContext<'model>,
        draft: &'ctx mut LlamaContext<'model>,
        config: MtpSessionConfig,
    ) -> Result<Self, MtpSessionError> {
        validate_config(config.n_seq, config.n_draft_max, config.n_min, config.p_min)
            .map_err(MtpSessionError::InvalidConfig)?;
        validate_contexts(target, draft, config)?;
        let sequence_slots = usize::try_from(config.n_seq)
            .map_err(|_| MtpSessionError::InvalidConfig("n_seq exceeds usize"))?;

        // `MTP_SPEC_TYPE_*` is `c_uint` under clang/gcc and `c_int` under MSVC;
        // `as i32` compiles on both. The allow covers the clang/gcc case.
        #[allow(clippy::cast_possible_wrap)]
        let c_config = llama_cpp_sys_4::mtp_session_config {
            n_seq: config.n_seq,
            n_draft_max: config.n_draft_max,
            n_min: config.n_min,
            p_min: config.p_min,
            spec_type: llama_cpp_sys_4::MTP_SPEC_TYPE_MTP as i32,
        };

        let raw = unsafe {
            llama_cpp_sys_4::mtp_session_new(
                target.context.as_ptr(),
                draft.context.as_ptr(),
                &raw const c_config,
            )
        };
        let raw = NonNull::new(raw).ok_or(MtpSessionError::Init)?;
        Ok(Self {
            raw,
            config,
            target,
            draft,
            pending_proposals: vec![None; sequence_slots],
            not_send_sync: PhantomData,
        })
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

    /// Returns shared access to the target context for reading logits,
    /// embeddings, and model metadata.
    #[must_use]
    pub fn target_context(&self) -> &LlamaContext<'model> {
        self.target
    }

    /// Returns exclusive access to the target context while this wrapper
    /// retains native pointer ownership.
    #[must_use]
    pub fn target_context_mut(&mut self) -> &mut LlamaContext<'model> {
        self.target
    }

    /// Returns shared access to the draft context for metadata inspection.
    #[must_use]
    pub fn draft_context(&self) -> &LlamaContext<'model> {
        self.draft
    }

    /// Returns exclusive access to the draft context while this wrapper
    /// retains native pointer ownership.
    #[must_use]
    pub fn draft_context_mut(&mut self) -> &mut LlamaContext<'model> {
        self.draft
    }

    /// Decodes on the target and immediately harvests the same batch into MTP.
    ///
    /// This is the causal target boundary required by the native draft
    /// implementation. A failed decode is never passed to `process`.
    ///
    /// # Errors
    ///
    /// Returns a target [`crate::DecodeError`] or native process failure.
    pub fn decode_target_and_process(
        &mut self,
        batch: &mut LlamaBatch,
    ) -> Result<(), MtpSessionError> {
        self.decode_target(batch)?;
        self.process(batch)
    }

    /// Decodes one batch on the exclusively held target context.
    ///
    /// Use [`Self::decode_target_and_process`] unless mechanics must run
    /// between target decode and draft-state harvesting. This method remains
    /// available while a draft proposal is pending because that is the target
    /// verification phase; proposal creation, begin, and state access retain
    /// their stricter lifecycle checks.
    ///
    /// # Errors
    ///
    /// Returns a target [`crate::DecodeError`].
    pub fn decode_target(&mut self, batch: &mut LlamaBatch) -> Result<(), MtpSessionError> {
        self.target.decode(batch)?;
        Ok(())
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
    ///
    /// # Errors
    ///
    /// Returns [`MtpSessionError::BadSeqId`] if `seq_id` is out of range.
    pub fn begin(&mut self, seq_id: i32, prompt: &[LlamaToken]) -> Result<(), MtpSessionError> {
        self.check_seq(seq_id)?;
        self.require_quiescent()?;
        if prompt.len() > MAX_SPECULATIVE_PROMPT_TOKENS {
            return Err(MtpSessionError::PromptTooLong {
                size: prompt.len(),
                maximum: MAX_SPECULATIVE_PROMPT_TOKENS,
            });
        }
        let ok = unsafe {
            llama_cpp_sys_4::mtp_session_begin(
                self.raw.as_ptr(),
                seq_id,
                prompt.as_ptr().cast(),
                prompt.len(),
            )
        };
        if !ok {
            return Err(MtpSessionError::Begin);
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
    ///
    /// # Errors
    ///
    /// Returns [`MtpSessionError::Process`] when upstream rejects the batch.
    pub fn process(&mut self, batch: &LlamaBatch) -> Result<(), MtpSessionError> {
        let ok = unsafe {
            llama_cpp_sys_4::mtp_session_process(self.raw.as_ptr(), &raw const batch.llama_batch)
        };
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
    ///
    /// # Errors
    ///
    /// Returns [`MtpSessionError::BadSeqId`] if `seq_id` is out of range.
    pub fn draft(
        &mut self,
        seq_id: i32,
        n_past: i32,
        id_last: LlamaToken,
    ) -> Result<Vec<LlamaToken>, MtpSessionError> {
        self.check_seq(seq_id)?;
        let sequence_index = self.sequence_index(seq_id)?;
        if self.pending_proposals[sequence_index].is_some() {
            return Err(MtpSessionError::ProposalPending { seq_id });
        }

        let cap = usize::try_from(self.config.n_draft_max.max(0)).unwrap_or(0);
        let mut buf: Vec<i32> = vec![0; cap];
        let mut out_n = i32::try_from(cap).unwrap_or(i32::MAX);

        let ok = unsafe {
            llama_cpp_sys_4::mtp_session_draft(
                self.raw.as_ptr(),
                seq_id,
                n_past,
                id_last.0,
                buf.as_mut_ptr(),
                &raw mut out_n,
            )
        };
        if !ok {
            return Err(MtpSessionError::Draft);
        }

        let n = usize::try_from(out_n.max(0)).unwrap_or(0);
        buf.truncate(n);
        if n > 0 {
            self.pending_proposals[sequence_index] = Some(n);
        }
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
    ///
    /// # Errors
    ///
    /// Returns [`MtpSessionError::BadSeqId`] if `seq_id` is out of range.
    pub fn accept(&mut self, seq_id: i32, n_accepted: u16) -> Result<(), MtpSessionError> {
        self.check_seq(seq_id)?;
        let sequence_index = self.sequence_index(seq_id)?;
        let proposed = self.pending_proposals[sequence_index]
            .ok_or(MtpSessionError::NoPendingProposal { seq_id })?;
        if usize::from(n_accepted) > proposed {
            return Err(MtpSessionError::AcceptedTooMany {
                accepted: n_accepted,
                proposed,
            });
        }
        let ok =
            unsafe { llama_cpp_sys_4::mtp_session_accept(self.raw.as_ptr(), seq_id, n_accepted) };
        if !ok {
            return Err(MtpSessionError::Accept);
        }
        self.pending_proposals[sequence_index] = None;
        Ok(())
    }

    /// Returns `true` when every draft proposal has been completed.
    #[must_use]
    pub fn is_quiescent(&self) -> bool {
        self.pending_proposals.iter().all(Option::is_none)
            && unsafe { llama_cpp_sys_4::mtp_session_is_quiescent(self.raw.as_ptr()) }
    }

    /// Captures versioned per-sequence speculative continuation state.
    ///
    /// Target and draft context bytes are separate and must be checkpointed at
    /// the same quiescent boundary.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid sequence, pending proposal, incomplete
    /// native support, or excessive state.
    pub fn speculative_state(&self, seq_id: i32) -> Result<Vec<u8>, MtpSessionError> {
        self.check_seq(seq_id)?;
        self.require_quiescent()?;
        Ok(capture_state(self.raw, seq_id)?)
    }

    /// Restores versioned per-sequence speculative continuation state.
    ///
    /// Restore the corresponding target and draft context bytes before calling
    /// this method.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid sequence, pending proposal, excessive
    /// input, or any version/configuration/state mismatch.
    pub fn restore_speculative_state(
        &mut self,
        seq_id: i32,
        state: &[u8],
    ) -> Result<(), MtpSessionError> {
        self.check_seq(seq_id)?;
        self.require_quiescent()?;
        restore_state(self.raw, seq_id, state)?;
        Ok(())
    }

    /// Removes a target-context KV range.
    ///
    /// # Errors
    ///
    /// Returns a conversion error when an identifier or position exceeds
    /// native `i32` bounds.
    pub fn clear_target_kv_cache_seq(
        &mut self,
        seq_id: Option<u32>,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<bool, crate::context::kv_cache::KvCacheConversionError> {
        self.target.clear_kv_cache_seq(seq_id, p0, p1)
    }

    /// Removes a draft-context KV range.
    ///
    /// # Errors
    ///
    /// Returns a conversion error when an identifier or position exceeds
    /// native `i32` bounds.
    pub fn clear_draft_kv_cache_seq(
        &mut self,
        seq_id: Option<u32>,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<bool, crate::context::kv_cache::KvCacheConversionError> {
        self.draft.clear_kv_cache_seq(seq_id, p0, p1)
    }

    /// Returns the target context's exact sequence-state byte count.
    pub fn target_state_seq_get_size_ext(&mut self, seq_id: i32, flags: u32) -> usize {
        self.target.state_seq_get_size_ext(seq_id, flags)
    }

    /// Copies target context sequence state with exact native flags.
    pub fn target_state_seq_get_data_ext(
        &mut self,
        dst: &mut [u8],
        seq_id: i32,
        flags: u32,
    ) -> usize {
        self.target.state_seq_get_data_ext(dst, seq_id, flags)
    }

    /// Restores target context sequence state with exact native flags.
    pub fn target_state_seq_set_data_ext(&mut self, src: &[u8], seq_id: i32, flags: u32) -> usize {
        self.target.state_seq_set_data_ext(src, seq_id, flags)
    }

    /// Returns the draft context's exact sequence-state byte count.
    pub fn draft_state_seq_get_size_ext(&mut self, seq_id: i32, flags: u32) -> usize {
        self.draft.state_seq_get_size_ext(seq_id, flags)
    }

    /// Copies draft context sequence state with exact native flags.
    pub fn draft_state_seq_get_data_ext(
        &mut self,
        dst: &mut [u8],
        seq_id: i32,
        flags: u32,
    ) -> usize {
        self.draft.state_seq_get_data_ext(dst, seq_id, flags)
    }

    /// Restores draft context sequence state with exact native flags.
    pub fn draft_state_seq_set_data_ext(&mut self, src: &[u8], seq_id: i32, flags: u32) -> usize {
        self.draft.state_seq_set_data_ext(src, seq_id, flags)
    }

    fn require_quiescent(&self) -> Result<(), MtpSessionError> {
        if let Some((index, _)) = self
            .pending_proposals
            .iter()
            .enumerate()
            .find(|(_, proposal)| proposal.is_some())
        {
            return Err(MtpSessionError::ProposalPending {
                seq_id: i32::try_from(index).unwrap_or(i32::MAX),
            });
        }
        if !unsafe { llama_cpp_sys_4::mtp_session_is_quiescent(self.raw.as_ptr()) } {
            return Err(MtpSessionError::State(SpeculativeStateError::NotQuiescent));
        }
        Ok(())
    }

    fn check_seq(&self, seq_id: i32) -> Result<(), MtpSessionError> {
        if seq_id < 0 || seq_id.cast_unsigned() >= self.config.n_seq {
            return Err(MtpSessionError::BadSeqId {
                seq_id,
                n_seq: self.config.n_seq,
            });
        }
        Ok(())
    }

    fn sequence_index(&self, seq_id: i32) -> Result<usize, MtpSessionError> {
        self.check_seq(seq_id)?;
        usize::try_from(seq_id)
            .map_err(|_| MtpSessionError::InvalidConfig("sequence id exceeds usize"))
    }
}

fn validate_contexts(
    target: &LlamaContext<'_>,
    draft: &LlamaContext<'_>,
    config: MtpSessionConfig,
) -> Result<(), MtpSessionError> {
    if target.context_type() != LlamaContextType::Default
        || draft.context_type() != LlamaContextType::Mtp
    {
        return Err(MtpSessionError::IncompatibleContexts(
            "target must be Default and draft must be Mtp",
        ));
    }
    if target.n_seq_max() < config.n_seq || draft.n_seq_max() != config.n_seq {
        return Err(MtpSessionError::IncompatibleContexts(
            "target sequence capacity is too small or draft capacity differs from n_seq",
        ));
    }
    let required_draft = u32::try_from(config.n_draft_max)
        .map_err(|_| MtpSessionError::InvalidConfig("n_draft_max exceeds u32"))?;
    validate_context_capacities(
        SpeculativeContextCapacity {
            batch: target.n_batch(),
            micro_batch: target.n_ubatch(),
            recurrent_slots: target.n_rs_seq(),
            recurrent_or_hybrid: target.model.is_recurrent() || target.model.is_hybrid(),
        },
        SpeculativeContextCapacity {
            batch: draft.n_batch(),
            micro_batch: draft.n_ubatch(),
            recurrent_slots: draft.n_rs_seq(),
            recurrent_or_hybrid: draft.model.is_recurrent() || draft.model.is_hybrid(),
        },
        required_draft,
    )
    .map_err(MtpSessionError::IncompatibleContexts)?;
    if draft.model.n_embd_out() != target.model.n_embd() {
        return Err(MtpSessionError::IncompatibleContexts(
            "draft output width differs from target hidden width",
        ));
    }
    Ok(())
}

impl Drop for MtpSession<'_, '_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_4::mtp_session_free(self.raw.as_ptr()) }
    }
}

impl std::fmt::Debug for MtpSession<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpSession")
            .field("config", &self.config)
            .field("need_embd_pre_norm", &self.need_embd_pre_norm())
            .finish_non_exhaustive()
    }
}
