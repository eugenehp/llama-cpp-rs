//! Incremental, byte-exact detokenization.
//!
//! Byte-fallback tokenizers split a single UTF-8 codepoint across multiple
//! tokens (common for emoji, CJK, and accented text). Converting each token to
//! a [`String`] in isolation therefore yields invalid UTF-8 mid-sequence, and
//! the attribute-filtering conversions ([`LlamaModel::token_to_bytes`]) drop
//! control/byte pieces entirely.
//!
//! [`StreamDetokenizer`] solves this for token-by-token generation loops: it
//! accumulates the raw piece bytes from [`LlamaModel::token_to_raw_bytes`] and
//! only emits complete UTF-8, retaining any trailing partial sequence until a
//! later token completes it.
//!
//! ```no_run
//! use llama_cpp_4::model::{LlamaModel, Special};
//! use llama_cpp_4::token::detokenizer::StreamDetokenizer;
//! # fn demo(model: &LlamaModel, tokens: &[llama_cpp_4::token::LlamaToken]) {
//! let mut detok = StreamDetokenizer::new(model, Special::Plaintext);
//! let mut text = String::new();
//! for &token in tokens {
//!     text.push_str(&detok.push(token).unwrap());
//! }
//! text.push_str(&detok.finish().unwrap());
//! # }
//! ```

use std::str::Utf8Error;

use crate::model::{LlamaModel, Special};
use crate::token::LlamaToken;
use crate::TokenToStringError;

/// An error produced while streaming detokenization.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DetokenizeError {
    /// A token could not be converted to its raw piece bytes.
    #[error("failed to convert token to raw bytes: {0}")]
    TokenToBytes(#[from] TokenToStringError),
    /// The accumulated bytes contained a genuinely invalid UTF-8 sequence, as
    /// opposed to a merely incomplete trailing sequence (which is buffered).
    #[error("invalid utf-8 in detokenized output: {0}")]
    InvalidUtf8(Utf8Error),
    /// [`StreamDetokenizer::finish`] was called while an incomplete trailing
    /// UTF-8 sequence was still buffered. Carries the leftover bytes.
    #[error("stream ended with {} incomplete utf-8 byte(s)", .0.len())]
    IncompleteUtf8(Vec<u8>),
}

/// A stateful, incremental detokenizer built on
/// [`LlamaModel::token_to_raw_bytes`].
///
/// Feed tokens one at a time with [`push`](StreamDetokenizer::push); each call
/// returns the text that has become complete, buffering any partial multi-byte
/// character until it can be finished. Call
/// [`finish`](StreamDetokenizer::finish) at the end of a generation to flush and
/// validate any remaining bytes.
#[derive(Debug)]
pub struct StreamDetokenizer<'a> {
    model: &'a LlamaModel,
    special: Special,
    buffer: Vec<u8>,
}

impl<'a> StreamDetokenizer<'a> {
    /// Create a new streaming detokenizer over `model`.
    ///
    /// `special` is forwarded to [`LlamaModel::token_to_raw_bytes`] and controls
    /// whether special/control tokens are rendered ([`Special::Tokenize`]) or
    /// treated as plaintext ([`Special::Plaintext`]).
    #[must_use]
    pub fn new(model: &'a LlamaModel, special: Special) -> Self {
        Self {
            model,
            special,
            buffer: Vec::new(),
        }
    }

    /// Feed a single token, returning any text that is now complete.
    ///
    /// A trailing incomplete UTF-8 sequence is retained internally and emitted
    /// once a subsequent token completes it.
    ///
    /// # Errors
    ///
    /// - [`DetokenizeError::TokenToBytes`] if the token cannot be converted.
    /// - [`DetokenizeError::InvalidUtf8`] if the accumulated bytes contain a
    ///   genuinely malformed (not merely incomplete) UTF-8 sequence.
    pub fn push(&mut self, token: LlamaToken) -> Result<String, DetokenizeError> {
        let raw = self.model.token_to_raw_bytes(token, self.special)?;
        self.buffer.extend_from_slice(&raw);
        self.drain_complete()
    }

    /// Feed several tokens, returning the concatenation of all completed text.
    ///
    /// # Errors
    ///
    /// See [`push`](StreamDetokenizer::push).
    pub fn push_all(
        &mut self,
        tokens: impl IntoIterator<Item = LlamaToken>,
    ) -> Result<String, DetokenizeError> {
        let mut out = String::new();
        for token in tokens {
            out.push_str(&self.push(token)?);
        }
        Ok(out)
    }

    /// The bytes currently buffered awaiting completion of a multi-byte
    /// character. Empty when the stream is on a UTF-8 boundary.
    #[must_use]
    pub fn pending(&self) -> &[u8] {
        &self.buffer
    }

    /// Finish the stream, returning any remaining complete text.
    ///
    /// # Errors
    ///
    /// [`DetokenizeError::IncompleteUtf8`] if bytes remain that do not form a
    /// complete UTF-8 sequence (carrying the leftover bytes), or
    /// [`DetokenizeError::InvalidUtf8`] if the buffered bytes are malformed.
    pub fn finish(mut self) -> Result<String, DetokenizeError> {
        let text = self.drain_complete()?;
        if self.buffer.is_empty() {
            Ok(text)
        } else {
            Err(DetokenizeError::IncompleteUtf8(self.buffer))
        }
    }

    /// Move the longest valid UTF-8 prefix out of the buffer and return it,
    /// leaving any incomplete trailing sequence buffered.
    fn drain_complete(&mut self) -> Result<String, DetokenizeError> {
        take_valid_utf8(&mut self.buffer).map_err(DetokenizeError::InvalidUtf8)
    }
}

/// Drain the longest valid UTF-8 prefix from `buffer`, returning it as a
/// [`String`] and leaving any incomplete trailing sequence in place.
///
/// Returns [`Utf8Error`] only for genuinely malformed bytes; a merely
/// unfinished trailing multi-byte sequence is retained in `buffer` and reported
/// as an empty (or partial) success so the caller can wait for more input.
fn take_valid_utf8(buffer: &mut Vec<u8>) -> Result<String, Utf8Error> {
    let valid = match std::str::from_utf8(buffer) {
        Ok(_) => buffer.len(),
        // `error_len().is_some()` means the invalid bytes are not simply an
        // unfinished tail — they are genuinely malformed, so surface them.
        Err(e) if e.error_len().is_some() => return Err(e),
        Err(e) => e.valid_up_to(),
    };
    let complete: Vec<u8> = buffer.drain(..valid).collect();
    Ok(String::from_utf8(complete).expect("prefix up to valid_up_to is valid utf-8"))
}

#[cfg(test)]
mod tests {
    use super::{take_valid_utf8, DetokenizeError};

    #[test]
    fn complete_utf8_is_fully_drained() {
        let mut buf = "hello".as_bytes().to_vec();
        assert_eq!(take_valid_utf8(&mut buf).unwrap(), "hello");
        assert!(buf.is_empty());
    }

    #[test]
    fn incomplete_multibyte_is_retained() {
        // "€" is 0xE2 0x82 0xAC; feed only the first two bytes.
        let mut buf = vec![b'a', 0xE2, 0x82];
        assert_eq!(take_valid_utf8(&mut buf).unwrap(), "a");
        // The unfinished sequence stays buffered for the next chunk.
        assert_eq!(buf, vec![0xE2, 0x82]);

        // Completing it emits the character and empties the buffer.
        buf.push(0xAC);
        assert_eq!(take_valid_utf8(&mut buf).unwrap(), "€");
        assert!(buf.is_empty());
    }

    #[test]
    fn malformed_bytes_are_surfaced() {
        // 0xFF is never valid in UTF-8 and is not an unfinished tail.
        let mut buf = vec![b'a', 0xFF, b'b'];
        assert!(take_valid_utf8(&mut buf).is_err());
    }

    #[test]
    fn incomplete_utf8_reports_leftover_bytes() {
        let err = DetokenizeError::IncompleteUtf8(vec![0xE2, 0x82]);
        assert_eq!(err.to_string(), "stream ended with 2 incomplete utf-8 byte(s)");
    }
}
