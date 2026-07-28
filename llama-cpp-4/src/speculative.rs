//! Shared exact-state contracts for speculative draft sessions.

use std::ptr::NonNull;

/// Maximum speculative continuation bytes copied through the safe API.
pub const MAX_SPECULATIVE_STATE_BYTES: usize = 64 * 1024 * 1024;
/// Maximum concurrent sequence slots allocated by one speculative session.
pub const MAX_SPECULATIVE_SEQUENCES: u32 = 4_096;
/// Maximum tokens requested from one speculative draft boundary.
pub const MAX_SPECULATIVE_DRAFT_TOKENS: i32 = 4_096;
/// Maximum prompt tokens copied into one speculative session.
pub const MAX_SPECULATIVE_PROMPT_TOKENS: usize = 1_048_576;

/// Failure while capturing or restoring versioned speculative state.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SpeculativeStateError {
    /// A required native session or output pointer was null.
    #[error("speculative state operation received a null native value")]
    Null,
    /// The sequence id is outside the configured session range.
    #[error("speculative state sequence id is outside the configured range")]
    BadSequence,
    /// A draft proposal has not yet been completed with `accept`.
    #[error("speculative state is available only at a quiescent boundary")]
    NotQuiescent,
    /// Native state changed size between the size and copy calls.
    #[error("speculative state size changed from {expected} to {actual} bytes")]
    SizeChanged {
        /// Size returned by the initial query.
        expected: usize,
        /// Size required or written by the copy call.
        actual: usize,
    },
    /// Native reported that the supplied buffer was too small without a size.
    #[error("the speculative state buffer was too small")]
    BufferTooSmall,
    /// The active native implementation did not expose complete state.
    #[error("the active speculative implementation did not expose complete state")]
    Unavailable,
    /// The supplied state failed its exact version/configuration checks.
    #[error("speculative state is invalid for this session")]
    Invalid,
    /// Native state size arithmetic overflowed.
    #[error("speculative state size overflowed")]
    Overflow,
    /// A C++ exception was contained by the speculative-state shim.
    #[error("the native speculative-state operation raised an exception")]
    Exception,
    /// State exceeded the safe allocation/input bound.
    #[error("speculative state is {size} bytes, exceeding the {maximum}-byte bound")]
    Excessive {
        /// Requested or supplied byte count.
        size: usize,
        /// Inclusive safe bound.
        maximum: usize,
    },
    /// Native returned a status unknown to this binding revision.
    #[error("unknown speculative state status {0}")]
    Unknown(u32),
}

pub(crate) fn capture_state(
    raw: NonNull<llama_cpp_sys_4::mtp_session>,
    seq_id: i32,
) -> Result<Vec<u8>, SpeculativeStateError> {
    let mut size = 0_usize;
    // SAFETY: `raw` is owned by the calling safe session and `size` is a live
    // output for the duration of the synchronous call.
    let status =
        unsafe { llama_cpp_sys_4::mtp_session_state_size(raw.as_ptr(), seq_id, &raw mut size) };
    status_result(status)?;
    validate_size(size)?;

    let mut state = vec![0_u8; size];
    let mut written = 0_usize;
    // SAFETY: the vector has exactly `size` writable bytes and the session
    // remains exclusively borrowed by the caller.
    let status = unsafe {
        llama_cpp_sys_4::mtp_session_state_get(
            raw.as_ptr(),
            seq_id,
            state.as_mut_ptr(),
            state.len(),
            &raw mut written,
        )
    };
    if status == llama_cpp_sys_4::MTP_STATE_STATUS_BUFFER_SMALL {
        return Err(SpeculativeStateError::SizeChanged {
            expected: size,
            actual: written,
        });
    }
    status_result(status)?;
    if written != size {
        return Err(SpeculativeStateError::SizeChanged {
            expected: size,
            actual: written,
        });
    }
    Ok(state)
}

pub(crate) fn restore_state(
    raw: NonNull<llama_cpp_sys_4::mtp_session>,
    seq_id: i32,
    state: &[u8],
) -> Result<(), SpeculativeStateError> {
    validate_size(state.len())?;
    if state.is_empty() {
        return Err(SpeculativeStateError::Invalid);
    }
    // SAFETY: `state` remains live and immutable for the synchronous call; the
    // owning safe session provides exclusive native access.
    let status = unsafe {
        llama_cpp_sys_4::mtp_session_state_set(raw.as_ptr(), seq_id, state.as_ptr(), state.len())
    };
    status_result(status)
}

fn validate_size(size: usize) -> Result<(), SpeculativeStateError> {
    if size > MAX_SPECULATIVE_STATE_BYTES {
        return Err(SpeculativeStateError::Excessive {
            size,
            maximum: MAX_SPECULATIVE_STATE_BYTES,
        });
    }
    Ok(())
}

fn status_result(status: llama_cpp_sys_4::mtp_state_status) -> Result<(), SpeculativeStateError> {
    match status {
        llama_cpp_sys_4::MTP_STATE_STATUS_OK => Ok(()),
        llama_cpp_sys_4::MTP_STATE_STATUS_NULL => Err(SpeculativeStateError::Null),
        llama_cpp_sys_4::MTP_STATE_STATUS_BAD_SEQUENCE => Err(SpeculativeStateError::BadSequence),
        llama_cpp_sys_4::MTP_STATE_STATUS_NOT_QUIESCENT => Err(SpeculativeStateError::NotQuiescent),
        llama_cpp_sys_4::MTP_STATE_STATUS_BUFFER_SMALL => {
            Err(SpeculativeStateError::BufferTooSmall)
        }
        llama_cpp_sys_4::MTP_STATE_STATUS_UNAVAILABLE => Err(SpeculativeStateError::Unavailable),
        llama_cpp_sys_4::MTP_STATE_STATUS_INVALID => Err(SpeculativeStateError::Invalid),
        llama_cpp_sys_4::MTP_STATE_STATUS_OVERFLOW => Err(SpeculativeStateError::Overflow),
        llama_cpp_sys_4::MTP_STATE_STATUS_EXCEPTION => Err(SpeculativeStateError::Exception),
        unknown => Err(SpeculativeStateError::Unknown(unknown)),
    }
}

pub(crate) fn validate_config(
    n_seq: u32,
    n_draft_max: i32,
    n_min: i32,
    p_min: f32,
) -> Result<(), &'static str> {
    if n_seq == 0 || n_seq > MAX_SPECULATIVE_SEQUENCES {
        return Err("n_seq is outside the supported bound");
    }
    if !(1..=MAX_SPECULATIVE_DRAFT_TOKENS).contains(&n_draft_max) {
        return Err("n_draft_max is outside the supported bound");
    }
    if n_min < 0 || n_min > n_draft_max {
        return Err("n_min must be between zero and n_draft_max");
    }
    if !p_min.is_finite() || !(0.0..=1.0).contains(&p_min) {
        return Err("p_min must be finite and between zero and one");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_state_is_bounded_before_native_access() {
        assert!(validate_size(MAX_SPECULATIVE_STATE_BYTES).is_ok());
        assert_eq!(
            validate_size(MAX_SPECULATIVE_STATE_BYTES + 1),
            Err(SpeculativeStateError::Excessive {
                size: MAX_SPECULATIVE_STATE_BYTES + 1,
                maximum: MAX_SPECULATIVE_STATE_BYTES,
            })
        );
    }

    #[test]
    fn speculative_configuration_is_bounded_before_allocation() {
        assert!(validate_config(1, 4, 0, 0.0).is_ok());
        assert!(validate_config(0, 4, 0, 0.0).is_err());
        assert!(validate_config(MAX_SPECULATIVE_SEQUENCES + 1, 4, 0, 0.0).is_err());
        assert!(validate_config(1, MAX_SPECULATIVE_DRAFT_TOKENS + 1, 0, 0.0).is_err());
        assert!(validate_config(1, 4, 5, 0.0).is_err());
        assert!(validate_config(1, 4, 0, f32::NAN).is_err());
        assert!(validate_config(1, 4, 0, 1.1).is_err());
    }

    #[test]
    fn prompt_and_state_bounds_are_consistent() {
        let maximum_prompt_bytes = MAX_SPECULATIVE_PROMPT_TOKENS
            .checked_mul(std::mem::size_of::<i32>())
            .unwrap();
        assert!(maximum_prompt_bytes < MAX_SPECULATIVE_STATE_BYTES);
    }
}
