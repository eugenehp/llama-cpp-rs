//! Safe wrapper around the C++ MTP draft session.
//!
//! [`MtpSession`] pairs a target [`LlamaContext`] with an MTP draft
//! [`LlamaContext`] (built with
//! [`crate::context::params::LlamaContextType::Mtp`]) and drives the
//! multi-token-prediction speculative-decoding loop introduced in upstream
//! llama.cpp [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673).
//!
//! The actual draft algorithm lives in upstream's
//! `common/speculative.cpp` (`common_speculative_state_draft_mtp`); this
//! module is a thin Rust safe wrapper around a small C++ shim in
//! `llama-cpp-sys-4/mtp_shim/` that re-exposes that C++ class with C linkage.
//!
//! # Usage outline
//!
//! ```ignore
//! // Build the target context (default) and the MTP draft context.
//! let target = model.new_context(&backend, LlamaContextParams::default())?;
//! let draft  = model.new_context(
//!     &backend,
//!     LlamaContextParams::default()
//!         .with_ctx_type(LlamaContextType::Mtp)
//!         .with_n_rs_seq(4),
//! )?;
//!
//! let mut sess = MtpSession::new(&target, &draft, 1, 3)?;
//!
//! // After every llama_decode on the target context, hand the batch to MTP:
//! sess.process(&target_batch)?;
//!
//! // Then ask for a draft starting from the last sampled token:
//! let drafts = sess.draft(0, n_past, last_token)?;
//!
//! // Verify against target, decide how many to accept, then:
//! sess.accept(0, n_accepted as u16)?;
//! ```

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
/// drop them) for as long as the session exists.** This contract is not
/// enforced by the borrow checker — the session does not hold Rust borrows of
/// the contexts, because both contexts must remain individually mutable
/// (you'll be calling `target.decode(...)` while the session exists, and the
/// session also mutates the draft context internally).
///
/// Dropping a context that the session still references is undefined
/// behaviour at the C++ level (use-after-free inside `common_speculative_*`).
pub struct MtpSession {
    raw: NonNull<llama_cpp_sys_4::mtp_session>,
    n_seq: u32,
    n_draft_max: i32,
}

// SAFETY: the underlying C++ session owns its own state and is not tied to
// any TLS. Concurrent calls from multiple threads are NOT safe (it mutates
// internal buffers without locking) — that's modelled by `&mut self` on the
// mutating methods.
unsafe impl Send for MtpSession {}

impl MtpSession {
    /// Construct an MTP draft session.
    ///
    /// `target` must be a `LlamaContextType::Default` context.
    /// `draft` must be a `LlamaContextType::Mtp` context built from the same
    /// model and configured with `with_n_rs_seq(>= n_draft_max)`.
    ///
    /// `n_seq` is the number of concurrent sequences (1 for a single
    /// conversation). `n_draft_max` caps the number of tokens drafted per
    /// round (commonly 3 for Qwen3.6 MTP).
    ///
    /// # Errors
    ///
    /// Returns [`MtpSessionError::Init`] if upstream rejects the
    /// configuration (e.g. the model has no MTP heads).
    pub fn new(
        target: &LlamaContext<'_>,
        draft: &LlamaContext<'_>,
        n_seq: u32,
        n_draft_max: i32,
    ) -> Result<Self, MtpSessionError> {
        let raw = unsafe {
            llama_cpp_sys_4::mtp_session_new(
                target.context.as_ptr(),
                draft.context.as_ptr(),
                n_seq,
                n_draft_max,
            )
        };
        let raw = NonNull::new(raw).ok_or(MtpSessionError::Init)?;
        Ok(Self {
            raw,
            n_seq,
            n_draft_max,
        })
    }

    /// True if MTP requires embeddings to be extractable from the target
    /// context. For MTP this is always true — exposed for symmetry with
    /// upstream's `common_speculative_need_embd`.
    #[must_use]
    pub fn need_embd(&self) -> bool {
        unsafe { llama_cpp_sys_4::mtp_session_need_embd(self.raw.as_ptr()) }
    }

    /// Configured maximum number of tokens drafted per [`draft`](Self::draft)
    /// call.
    #[must_use]
    pub fn n_draft_max(&self) -> i32 {
        self.n_draft_max
    }

    /// Configured number of sequences.
    #[must_use]
    pub fn n_seq(&self) -> u32 {
        self.n_seq
    }

    /// Optional: call once at the start of a fresh generation with the
    /// prompt tokens that were just decoded into the target context.
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
    /// MTP needs to see every target batch (prompt prefill + each
    /// verification step) to keep its per-sequence pre-norm-embedding
    /// carryover in sync.
    ///
    /// # Errors
    ///
    /// Returns [`MtpSessionError::Process`] if upstream rejects the batch
    /// (most often: the batch carries `embd` directly rather than tokens).
    pub fn process(&mut self, batch: &LlamaBatch) -> Result<(), MtpSessionError> {
        let ok =
            unsafe { llama_cpp_sys_4::mtp_session_process(self.raw.as_ptr(), &batch.llama_batch) };
        if ok {
            Ok(())
        } else {
            Err(MtpSessionError::Process)
        }
    }

    /// Generate up to [`n_draft_max`](Self::n_draft_max) speculative tokens
    /// for sequence `seq_id`, starting from `id_last` at position `n_past`.
    ///
    /// Returns an owned `Vec<LlamaToken>` of length `<= n_draft_max`.
    ///
    /// # Errors
    ///
    /// Returns [`MtpSessionError::BadSeqId`] if `seq_id` is outside the
    /// configured `n_seq` range.
    pub fn draft(
        &mut self,
        seq_id: i32,
        n_past: i32,
        id_last: LlamaToken,
    ) -> Result<Vec<LlamaToken>, MtpSessionError> {
        self.check_seq(seq_id)?;

        let cap = self.n_draft_max.max(0) as usize;
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

    /// Inform the session that `n_accepted` tokens from the last draft were
    /// accepted by the target verifier. This is required after every
    /// [`draft`](Self::draft) call to keep the draft context's recurrent
    /// state consistent.
    pub fn accept(&mut self, seq_id: i32, n_accepted: u16) -> Result<(), MtpSessionError> {
        self.check_seq(seq_id)?;
        unsafe {
            llama_cpp_sys_4::mtp_session_accept(self.raw.as_ptr(), seq_id, n_accepted);
        }
        Ok(())
    }

    fn check_seq(&self, seq_id: i32) -> Result<(), MtpSessionError> {
        if seq_id < 0 || (seq_id as u32) >= self.n_seq {
            return Err(MtpSessionError::BadSeqId {
                seq_id,
                n_seq: self.n_seq,
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
            .field("n_seq", &self.n_seq)
            .field("n_draft_max", &self.n_draft_max)
            .finish()
    }
}
