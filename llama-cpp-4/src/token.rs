//! Safe wrappers around `llama_token_data` and `llama_token_data_array`.

use std::fmt::Debug;
use std::fmt::Display;
use std::time::Instant; // Importing Instant for measuring elapsed time

pub mod data;

/// A safe wrapper for `llama_token`.
///
/// This struct wraps around a `llama_token` and implements various traits for safe usage, including
/// `Clone`, `Copy`, `Debug`, `Eq`, `PartialEq`, `Ord`, `PartialOrd`, and `Hash`. The `Display` trait
/// is also implemented to provide a simple way to format `LlamaToken` for printing.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaToken(pub llama_cpp_sys_4::llama_token);

impl Display for LlamaToken {
    /// Formats the `LlamaToken` for display by printing its inner value.
    ///
    /// This implementation allows you to easily print a `LlamaToken` by using `{}` in formatting macros.
    ///
    /// # Example
    ///
    /// ```
    /// # use llama_cpp_2::token::LlamaToken;
    /// let token = LlamaToken::new(42);
    /// println!("{}", token); // Prints: 42
    /// ```
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl LlamaToken {
    /// Creates a new `LlamaToken` from an `i32`.
    ///
    /// This constructor allows you to easily create a `LlamaToken` from a raw integer value representing
    /// the token's ID. This is useful when interacting with external systems that provide token IDs as integers.
    ///
    /// # Example
    ///
    /// ```
    /// # use llama_cpp_2::token::LlamaToken;
    /// let token = LlamaToken::new(0);
    /// assert_eq!(token, LlamaToken(0));
    /// ```
    ///
    /// # Parameters
    ///
    /// - `token_id`: The integer ID for the token.
    ///
    /// # Returns
    ///
    /// Returns a new instance of `LlamaToken` wrapping the provided `token_id`.
    #[must_use]
    pub fn new(token_id: i32) -> Self {
        Self(token_id)
    }
}

/// Converts a vector of `llama_token` to a vector of `LlamaToken` without memory allocation,
/// and consumes the original vector. This conversion is safe because `LlamaToken` is repr(transparent),
/// meaning it is just a wrapper around the raw `llama_token` type.
///
/// # Safety
///
/// This operation is safe because `LlamaToken` has a `repr(transparent)` attribute, ensuring that
/// the memory layout of `LlamaToken` is the same as that of the underlying `llama_token` type.
pub fn from_vec_token_sys(mut vec_sys: Vec<llama_cpp_sys_4::llama_token>) -> Vec<LlamaToken> {
    let ptr = vec_sys.as_mut_ptr() as *mut LlamaToken;
    unsafe { Vec::from_raw_parts(ptr, vec_sys.len(), vec_sys.capacity()) }
}

/// Converts a vector of `LlamaToken` to a vector of `llama_token` without memory allocation,
/// and consumes the original vector. This conversion is safe because `LlamaToken` is repr(transparent),
/// meaning it is just a wrapper around the raw `llama_token` type.
///
/// # Safety
///
/// This operation is safe because `LlamaToken` has a `repr(transparent)` attribute, ensuring that
/// the memory layout of `LlamaToken` is the same as that of the underlying `llama_token` type.
pub fn to_vec_token_sys(mut vec_llama: Vec<LlamaToken>) -> Vec<llama_cpp_sys_4::llama_token> {
    let ptr = vec_llama.as_mut_ptr() as *mut llama_cpp_sys_4::llama_token;
    unsafe { Vec::from_raw_parts(ptr, vec_llama.len(), vec_llama.capacity()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_new_llama_token() {
        let token = LlamaToken::new(42);
        assert_eq!(token, LlamaToken(42)); // Verify that the created token has the expected value
    }

    #[test]
    fn test_llama_token_display() {
        let token = LlamaToken::new(99);
        assert_eq!(format!("{}", token), "99"); // Verify that the token formats correctly
    }

    #[test]
    fn test_from_vec_token_sys() {
        // Test converting a vector of raw `llama_token` to a vector of `LlamaToken`
        let vec_sys: Vec<llama_cpp_sys_4::llama_token> = vec![1, 2, 3];
        let vec_llama = from_vec_token_sys(vec_sys);

        // Ensure that the conversion works correctly
        assert_eq!(vec_llama.len(), 3);
        assert_eq!(vec_llama[0], LlamaToken(1));
        assert_eq!(vec_llama[1], LlamaToken(2));
        assert_eq!(vec_llama[2], LlamaToken(3));
    }

    #[test]
    fn test_to_vec_token_sys() {
        // Test converting a vector of `LlamaToken` to a vector of raw `llama_token`
        let vec_llama = vec![LlamaToken(10), LlamaToken(20), LlamaToken(30)];
        let vec_sys = to_vec_token_sys(vec_llama);

        // Ensure that the conversion works correctly
        assert_eq!(vec_sys.len(), 3);
        assert_eq!(vec_sys[0], 10);
        assert_eq!(vec_sys[1], 20);
        assert_eq!(vec_sys[2], 30);
    }

    #[test]
    fn benchmark_to_vec_token_sys() {
        // Benchmark the speed of to_vec_token_sys by timing it
        let vec_llama: Vec<LlamaToken> = (0..100_000).map(LlamaToken::new).collect();

        let start = Instant::now();
        let _vec_sys = to_vec_token_sys(vec_llama);
        let duration = start.elapsed();

        println!(
            "Time taken to convert Vec<LlamaToken> to Vec<llama_token>: {:?}",
            duration
        );

        // Here we can assert that the conversion took a reasonable amount of time.
        // This threshold is arbitrary and can be adjusted according to expected performance.
        assert!(duration.as_micros() < 1_000); // Ensure it takes less than 1ms
    }
}
