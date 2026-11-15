//! utilities for working with session files

use crate::context::LlamaContext;
use crate::token::LlamaToken;
use std::ffi::{CString, NulError};
use std::path::{Path, PathBuf};

/// Failed to save a Session file
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SaveSessionError {
    /// llama.cpp failed to save the session file
    #[error("Failed to save session file")]
    FailedToSave,

    /// null byte in string
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    /// failed to convert path to str
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
}

/// Failed to load a Session file
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LoadSessionError {
    /// llama.cpp failed to load the session file
    #[error("Failed to load session file")]
    FailedToLoad,

    /// null byte in string
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    /// failed to convert path to str
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),

    /// Insufficient max length
    #[error("max_length is not large enough to hold {n_out} (was {max_tokens})")]
    InsufficientMaxLength {
        /// The length of the session file
        n_out: usize,
        /// The maximum length
        max_tokens: usize,
    },
}

impl LlamaContext<'_> {
    /// Save the current session to a file.
    ///
    /// # Parameters
    ///
    /// * `path_session` - The file to save to.
    /// * `tokens` - The tokens to associate the session with. This should be a prefix of a sequence of tokens that the context has processed, so that the relevant KV caches are already filled.
    ///
    /// # Errors
    ///
    /// Fails if the path is not a valid utf8, is not a valid c string, or llama.cpp fails to save the session file.
    pub fn save_session_file(
        &self,
        path_session: impl AsRef<Path>,
        tokens: &[LlamaToken],
    ) -> Result<(), SaveSessionError> {
        let path = path_session.as_ref();
        let path = path
            .to_str()
            .ok_or_else(|| SaveSessionError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;

        // SAFETY: `self.context` is non-null and kept alive by this borrow; the path is a `CString` alive for the
        // call, and the token pointer and count come from the same slice.
        if unsafe {
            llama_cpp_sys_4::llama_state_save_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                tokens.as_ptr().cast::<llama_cpp_sys_4::llama_token>(),
                tokens.len(),
            )
        } {
            Ok(())
        } else {
            Err(SaveSessionError::FailedToSave)
        }
    }
    /// Load a session file into the current context.
    ///
    /// You still need to pass the returned tokens to the context for inference to work. What this function buys you is that the KV caches are already filled with the relevant data.
    ///
    /// # Parameters
    ///
    /// * `path_session` - The file to load from. It must be a session file from a compatible context, otherwise the function will error.
    /// * `max_tokens` - The maximum token length of the loaded session. If the session was saved with a longer length, the function will error.
    ///
    /// # Errors
    ///
    /// Fails if the path is not a valid utf8, is not a valid c string, or llama.cpp fails to load the session file. (e.g. the file does not exist, is not a session file, etc.)
    pub fn load_session_file(
        &mut self,
        path_session: impl AsRef<Path>,
        max_tokens: usize,
    ) -> Result<Vec<LlamaToken>, LoadSessionError> {
        let path = path_session.as_ref();
        let path = path
            .to_str()
            .ok_or(LoadSessionError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;
        let mut tokens: Vec<LlamaToken> = Vec::with_capacity(max_tokens);
        let mut n_out = 0;

        // SAFETY: cast is valid as LlamaToken is repr(transparent)
        let tokens_out = tokens.as_mut_ptr().cast::<llama_cpp_sys_4::llama_token>();

        // SAFETY: `self.context` is non-null and kept alive by this borrow; the path is a `CString` alive for the
        // call, and the output buffer's capacity is passed with its pointer.
        let load_session_success = unsafe {
            llama_cpp_sys_4::llama_state_load_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                tokens_out,
                max_tokens,
                &raw mut n_out,
            )
        };
        if load_session_success {
            if n_out > max_tokens {
                return Err(LoadSessionError::InsufficientMaxLength { n_out, max_tokens });
            }
            // SAFETY: we checked that n_out <= max_tokens and llama.cpp promises that n_out tokens will be written
            unsafe {
                tokens.set_len(n_out);
            }
            Ok(tokens)
        } else {
            Err(LoadSessionError::FailedToLoad)
        }
    }

    /// Returns the maximum size in bytes of the state (rng, logits, embedding
    /// and `kv_cache`) - will often be smaller after compacting tokens
    #[must_use]
    pub fn get_state_size(&self) -> usize {
        // SAFETY: a plain read through a live context.
        unsafe { llama_cpp_sys_4::llama_state_get_size(self.context.as_ptr()) }
    }

    /// Copy the state into `dest`, returning the number of bytes written.
    ///
    /// Size `dest` with [`get_state_size`](Self::get_state_size).
    ///
    /// This was an `unsafe fn` taking a bare `*mut u8` until `b12388` removed
    /// `llama_copy_state_data`. Its replacement takes the destination length,
    /// so the bound is checked by llama.cpp and this is now safe.
    pub fn copy_state_data(&self, dest: &mut [u8]) -> usize {
        // SAFETY: `self.context` is non-null and kept alive by this borrow, and
        // the destination pointer and its length are passed together, so
        // llama.cpp cannot write past the slice.
        unsafe {
            llama_cpp_sys_4::llama_state_get_data(
                self.context.as_ptr(),
                dest.as_mut_ptr(),
                dest.len(),
            )
        }
    }

    /// Restore the state from `src`, returning the number of bytes read.
    ///
    /// This was an `unsafe fn` until `b12388` removed `llama_set_state_data`:
    /// that entry point read as much as the serialized header claimed and
    /// never consulted `src.len()`, so a truncated or foreign buffer was read
    /// past its end. The replacement takes the length, so this is now safe.
    ///
    /// `src` must still come from [`copy_state_data`](Self::copy_state_data)
    /// on a context built from the same model with the same parameters —
    /// `LLAMA_SESSION_VERSION` guards the format across releases, not the
    /// shape within one — but a mismatch is a short read now, not undefined
    /// behaviour.
    pub fn set_state_data(&mut self, src: &[u8]) -> usize {
        // SAFETY: `self.context` is non-null and kept alive by this borrow, and
        // `src`'s pointer and length are passed together, so the read stays in
        // bounds.
        unsafe {
            llama_cpp_sys_4::llama_state_set_data(self.context.as_ptr(), src.as_ptr(), src.len())
        }
    }
}
