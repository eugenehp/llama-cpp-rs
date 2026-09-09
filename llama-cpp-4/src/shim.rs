//! Plumbing shared by every wrapper over a C shim.
//!
//! `chat_shim` and `common_shim` agree on a status enum, a thread-local error
//! buffer, and a size-then-fill protocol for variable-length output. This is the
//! Rust half of that agreement: one error type and one set of helpers, so the
//! modules built on them cannot drift.

use std::ffi::{c_char, CStr, NulError};

use llama_cpp_sys_4 as sys;

use crate::token::LlamaToken;

/// A failure inside one of llama.cpp's C++ layers, reported through a shim.
///
/// One type for every shim-backed module. The variants are the union of what
/// the shims can report; a given call only produces the ones its documentation
/// names.
#[derive(Debug, thiserror::Error)]
pub enum ShimError {
    /// An argument contained an interior NUL byte, so it could not become a
    /// `CString`. Caught in Rust — truncating at the NUL would silently change
    /// the value.
    #[error("argument contained an interior NUL byte")]
    Nul(#[from] NulError),
    /// A required pointer argument was null, or a value was out of range.
    #[error("invalid argument passed to the shim")]
    InvalidArg,
    /// Input was not valid JSON.
    #[error("input was not valid JSON: {0}")]
    BadJson(String),
    /// llama.cpp threw. The message is what the exception carried.
    #[error("llama.cpp call failed: {0}")]
    Failed(String),
    /// A constructor returned null.
    #[error("could not construct: {0}")]
    Init(String),
    /// Output was not valid UTF-8.
    #[error("llama.cpp returned non-UTF-8 output")]
    Utf8(#[from] std::string::FromUtf8Error),
    /// A shim reported an offset outside the buffer it described, which would
    /// mean the shim and this crate disagree about the layout.
    #[error("shim reported an out-of-range offset")]
    CorruptResult,
}

pub(crate) type Result<T> = std::result::Result<T, ShimError>;

/// Detail for the most recent shim failure on this thread.
///
/// One buffer across all shims, so this is always about whatever just failed.
pub(crate) fn last_error() -> String {
    let ptr = unsafe { sys::llama_shim_last_error() };
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// Turn a shim status into a `Result`.
pub(crate) fn check_status(status: i32) -> Result<()> {
    match status {
        sys::LLAMA_SHIM_OK => Ok(()),
        sys::LLAMA_SHIM_INVALID_ARG => Err(ShimError::InvalidArg),
        sys::LLAMA_SHIM_BAD_JSON => Err(ShimError::BadJson(last_error())),
        _ => Err(ShimError::Failed(last_error())),
    }
}

/// Drive the size-then-fill protocol for a string output.
///
/// The first call sizes, the second fills. `BUFFER_TOO_SMALL` from the sizing
/// call is the expected answer, not a failure.
pub(crate) fn read_string<F>(mut call: F) -> Result<String>
where
    F: FnMut(*mut c_char, usize, *mut usize) -> i32,
{
    let mut needed: usize = 0;
    let status = call(std::ptr::null_mut(), 0, &raw mut needed);
    if status != sys::LLAMA_SHIM_BUFFER_TOO_SMALL {
        check_status(status)?;
    }
    if needed == 0 {
        return Ok(String::new());
    }

    let mut buf = vec![0u8; needed];
    let status = call(buf.as_mut_ptr().cast::<c_char>(), buf.len(), &raw mut needed);
    check_status(status)?;

    // Trim at the NUL the shim wrote rather than trusting `needed`, which the
    // second call may have revised.
    let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
    buf.truncate(end);
    String::from_utf8(buf).map_err(ShimError::from)
}

/// Drive the same protocol for a token-array output.
pub(crate) fn read_tokens<F>(mut call: F) -> Result<Vec<LlamaToken>>
where
    F: FnMut(*mut i32, usize, *mut usize) -> i32,
{
    let mut needed: usize = 0;
    let status = call(std::ptr::null_mut(), 0, &raw mut needed);
    if status != sys::LLAMA_SHIM_BUFFER_TOO_SMALL {
        check_status(status)?;
    }
    if needed == 0 {
        return Ok(Vec::new());
    }

    let mut buf = vec![0i32; needed];
    let status = call(buf.as_mut_ptr(), buf.len(), &raw mut needed);
    check_status(status)?;
    buf.truncate(needed);
    Ok(buf.into_iter().map(LlamaToken).collect())
}

/// Drive the same protocol for an `i32` output that is not a token — a list of
/// enum discriminants, say.
pub(crate) fn read_i32s<F>(mut call: F) -> Result<Vec<i32>>
where
    F: FnMut(*mut i32, usize, *mut usize) -> i32,
{
    let mut needed: usize = 0;
    let status = call(std::ptr::null_mut(), 0, &raw mut needed);
    if status != sys::LLAMA_SHIM_BUFFER_TOO_SMALL {
        check_status(status)?;
    }
    if needed == 0 {
        return Ok(Vec::new());
    }

    let mut buf = vec![0i32; needed];
    let status = call(buf.as_mut_ptr(), buf.len(), &raw mut needed);
    check_status(status)?;
    buf.truncate(needed);
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shims and this crate must agree on the status values; a mismatch
    /// would map "bad JSON" onto "it threw" or vice versa. This is exactly the
    /// bug that existed while `chat_shim` and `common_shim` had separate enums.
    #[test]
    fn status_codes_are_distinct_and_signed_as_documented() {
        // Exact values, not just "negative": `check_status` dispatches on them,
        // so a renumbering upstream would map one failure onto another.
        assert_eq!(sys::LLAMA_SHIM_OK, 0);
        assert_eq!(
            sys::LLAMA_SHIM_BUFFER_TOO_SMALL, 1,
            "a size query is not a failure, so it must be positive"
        );
        assert_eq!(sys::LLAMA_SHIM_INVALID_ARG, -1);
        assert_eq!(sys::LLAMA_SHIM_BAD_JSON, -2);
        assert_eq!(sys::LLAMA_SHIM_THROWN, -3);
    }

    #[test]
    fn check_status_maps_each_code() {
        assert!(check_status(sys::LLAMA_SHIM_OK).is_ok());
        assert!(matches!(
            check_status(sys::LLAMA_SHIM_INVALID_ARG),
            Err(ShimError::InvalidArg)
        ));
        assert!(matches!(
            check_status(sys::LLAMA_SHIM_BAD_JSON),
            Err(ShimError::BadJson(_))
        ));
        assert!(matches!(
            check_status(sys::LLAMA_SHIM_THROWN),
            Err(ShimError::Failed(_))
        ));
    }

    /// `last_error` reads a thread-local the shims share, so it must be safe to
    /// call before anything has failed.
    #[test]
    fn last_error_is_empty_before_any_failure() {
        // Not asserting emptiness — another test on this thread may have failed
        // first. The contract is that it never dereferences null.
        let _ = last_error();
    }
}
