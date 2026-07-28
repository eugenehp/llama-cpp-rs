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
//! let mut target = main_model.new_context(&backend, LlamaContextParams::default())?;
//!
//! // Draft: a SEPARATE EAGLE-3 draft model, also a default context.
//! let mut draft = eagle3_model.new_context(&backend, LlamaContextParams::default())?;
//!
//! let config = Eagle3SessionConfig::new(1, n_draft_max);
//! let mut session = Eagle3Session::new_with_config(&mut target, &mut draft, config)?;
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
//! session.decode_target_and_process(&mut batch)?;
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

    /// Native prompt initialization failed or raised a contained exception.
    #[error("EAGLE-3 begin failed")]
    Begin,

    /// Native draft generation failed or raised a contained exception.
    #[error("EAGLE-3 draft failed")]
    Draft,

    /// Native proposal acceptance failed or raised a contained exception.
    #[error("EAGLE-3 accept failed")]
    Accept,

    /// Prompt storage exceeds the safe speculative-session bound.
    #[error("prompt has {size} tokens, exceeding the {maximum}-token bound")]
    PromptTooLong {
        /// Caller-supplied prompt-token count.
        size: usize,
        /// Inclusive safe prompt-token bound.
        maximum: usize,
    },

    /// The supplied contexts do not satisfy the native EAGLE-3 contract.
    #[error("incompatible EAGLE-3 contexts: {0}")]
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
    #[error("invalid EAGLE-3 session config: {0}")]
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
/// Both contexts are exclusively borrowed for the session lifetime. The
/// wrapper retains no manually enforced lifetime and is neither `Send` nor
/// `Sync`.
pub struct Eagle3Session<'ctx, 'target_model, 'draft_model> {
    raw: NonNull<llama_cpp_sys_4::mtp_session>,
    config: Eagle3SessionConfig,
    target: &'ctx mut LlamaContext<'target_model>,
    draft: &'ctx mut LlamaContext<'draft_model>,
    pending_proposals: Vec<Option<usize>>,
    not_send_sync: PhantomData<Rc<()>>,
}

impl<'ctx, 'target_model, 'draft_model> Eagle3Session<'ctx, 'target_model, 'draft_model> {
    /// Construct an EAGLE-3 draft session with upstream defaults for `n_min`
    /// and `p_min`.
    ///
    /// Equivalent to `new_with_config(target, draft, Eagle3SessionConfig::new(n_seq, n_draft_max))`.
    ///
    /// # Errors
    ///
    /// Returns [`Eagle3SessionError::Init`] or [`Eagle3SessionError::InvalidConfig`].
    pub fn new(
        target: &'ctx mut LlamaContext<'target_model>,
        draft: &'ctx mut LlamaContext<'draft_model>,
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
        target: &'ctx mut LlamaContext<'target_model>,
        draft: &'ctx mut LlamaContext<'draft_model>,
        config: Eagle3SessionConfig,
    ) -> Result<Self, Eagle3SessionError> {
        validate_config(config.n_seq, config.n_draft_max, config.n_min, config.p_min)
            .map_err(Eagle3SessionError::InvalidConfig)?;
        validate_contexts(target, draft, config)?;
        let sequence_slots = usize::try_from(config.n_seq)
            .map_err(|_| Eagle3SessionError::InvalidConfig("n_seq exceeds usize"))?;

        // `MTP_SPEC_TYPE_*` is `c_uint` under clang/gcc and `c_int` under MSVC;
        // `as i32` compiles on both. The allow covers the clang/gcc case.
        #[allow(clippy::cast_possible_wrap)]
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

    /// Returns shared access to the target context for reading logits,
    /// embeddings, and model metadata.
    #[must_use]
    pub fn target_context(&self) -> &LlamaContext<'target_model> {
        self.target
    }

    /// Returns exclusive access to the target context while this wrapper
    /// retains native pointer ownership.
    #[must_use]
    pub fn target_context_mut(&mut self) -> &mut LlamaContext<'target_model> {
        self.target
    }

    /// Returns shared access to the draft context for metadata inspection.
    #[must_use]
    pub fn draft_context(&self) -> &LlamaContext<'draft_model> {
        self.draft
    }

    /// Returns exclusive access to the draft context while this wrapper
    /// retains native pointer ownership.
    #[must_use]
    pub fn draft_context_mut(&mut self) -> &mut LlamaContext<'draft_model> {
        self.draft
    }

    /// Decodes on the target and immediately harvests the same batch into
    /// EAGLE-3.
    ///
    /// # Errors
    ///
    /// Returns a target [`crate::DecodeError`] or native process failure.
    pub fn decode_target_and_process(
        &mut self,
        batch: &mut LlamaBatch,
    ) -> Result<(), Eagle3SessionError> {
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
    pub fn decode_target(&mut self, batch: &mut LlamaBatch) -> Result<(), Eagle3SessionError> {
        self.target.decode(batch)?;
        Ok(())
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
        self.require_quiescent()?;
        if prompt.len() > MAX_SPECULATIVE_PROMPT_TOKENS {
            return Err(Eagle3SessionError::PromptTooLong {
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
            return Err(Eagle3SessionError::Begin);
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
        let ok = unsafe {
            llama_cpp_sys_4::mtp_session_process(self.raw.as_ptr(), &raw const batch.llama_batch)
        };
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
        let sequence_index = self.sequence_index(seq_id)?;
        if self.pending_proposals[sequence_index].is_some() {
            return Err(Eagle3SessionError::ProposalPending { seq_id });
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
            return Err(Eagle3SessionError::Draft);
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
    /// Pass `0` when every draft was rejected.
    ///
    /// # Errors
    ///
    /// Returns [`Eagle3SessionError::BadSeqId`] if `seq_id` is out of range.
    pub fn accept(&mut self, seq_id: i32, n_accepted: u16) -> Result<(), Eagle3SessionError> {
        self.check_seq(seq_id)?;
        let sequence_index = self.sequence_index(seq_id)?;
        let proposed = self.pending_proposals[sequence_index]
            .ok_or(Eagle3SessionError::NoPendingProposal { seq_id })?;
        if usize::from(n_accepted) > proposed {
            return Err(Eagle3SessionError::AcceptedTooMany {
                accepted: n_accepted,
                proposed,
            });
        }
        let ok =
            unsafe { llama_cpp_sys_4::mtp_session_accept(self.raw.as_ptr(), seq_id, n_accepted) };
        if !ok {
            return Err(Eagle3SessionError::Accept);
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
    pub fn speculative_state(&self, seq_id: i32) -> Result<Vec<u8>, Eagle3SessionError> {
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
    ) -> Result<(), Eagle3SessionError> {
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

    fn require_quiescent(&self) -> Result<(), Eagle3SessionError> {
        if let Some((index, _)) = self
            .pending_proposals
            .iter()
            .enumerate()
            .find(|(_, proposal)| proposal.is_some())
        {
            return Err(Eagle3SessionError::ProposalPending {
                seq_id: i32::try_from(index).unwrap_or(i32::MAX),
            });
        }
        if !unsafe { llama_cpp_sys_4::mtp_session_is_quiescent(self.raw.as_ptr()) } {
            return Err(Eagle3SessionError::State(
                SpeculativeStateError::NotQuiescent,
            ));
        }
        Ok(())
    }

    fn check_seq(&self, seq_id: i32) -> Result<(), Eagle3SessionError> {
        if seq_id < 0 || seq_id.cast_unsigned() >= self.config.n_seq {
            return Err(Eagle3SessionError::BadSeqId {
                seq_id,
                n_seq: self.config.n_seq,
            });
        }
        Ok(())
    }

    fn sequence_index(&self, seq_id: i32) -> Result<usize, Eagle3SessionError> {
        self.check_seq(seq_id)?;
        usize::try_from(seq_id)
            .map_err(|_| Eagle3SessionError::InvalidConfig("sequence id exceeds usize"))
    }
}

fn validate_contexts(
    target: &LlamaContext<'_>,
    draft: &LlamaContext<'_>,
    config: Eagle3SessionConfig,
) -> Result<(), Eagle3SessionError> {
    if target.context_type() != LlamaContextType::Default
        || draft.context_type() != LlamaContextType::Default
    {
        return Err(Eagle3SessionError::IncompatibleContexts(
            "target and draft must both be Default contexts",
        ));
    }
    if target.n_seq_max() < config.n_seq || draft.n_seq_max() != config.n_seq {
        return Err(Eagle3SessionError::IncompatibleContexts(
            "target sequence capacity is too small or draft capacity differs from n_seq",
        ));
    }
    let required_draft = u32::try_from(config.n_draft_max)
        .map_err(|_| Eagle3SessionError::InvalidConfig("n_draft_max exceeds u32"))?;
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
    .map_err(Eagle3SessionError::IncompatibleContexts)?;
    let target_layers = target.model.n_layer();
    let target_architecture = target
        .model
        .meta_val_str("general.architecture", 64)
        .map_err(|_| {
            Eagle3SessionError::IncompatibleContexts(
                "target model architecture metadata is unavailable",
            )
        })?;
    if !valid_target_layer_ids(
        draft.model.target_layer_ids(),
        target_layers,
        target_architecture == "gpt-oss",
    ) {
        return Err(Eagle3SessionError::IncompatibleContexts(
            "draft must name exactly three supported target extraction sites",
        ));
    }
    Ok(())
}

fn valid_target_layer_ids(
    layer_ids: &[i32],
    target_layers: i32,
    terminal_nextn_site: bool,
) -> bool {
    target_layers > 0
        && layer_ids.len() == 3
        && layer_ids.iter().all(|&layer| {
            layer >= 0 && (layer < target_layers || (layer == target_layers && terminal_nextn_site))
        })
}

impl Drop for Eagle3Session<'_, '_, '_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_4::mtp_session_free(self.raw.as_ptr()) }
    }
}

impl std::fmt::Debug for Eagle3Session<'_, '_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Eagle3Session")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::valid_target_layer_ids;

    #[test]
    fn validates_transformer_and_terminal_nextn_sites() {
        assert!(valid_target_layer_ids(&[1, 4, 7], 8, false));
        assert!(!valid_target_layer_ids(&[1, 4], 8, false));
        assert!(!valid_target_layer_ids(&[1, -1, 7], 8, false));
        assert!(!valid_target_layer_ids(&[1, 4, 8], 8, false));
        assert!(valid_target_layer_ids(&[1, 4, 8], 8, true));
        assert!(!valid_target_layer_ids(&[1, 4, 9], 8, true));
    }
}
