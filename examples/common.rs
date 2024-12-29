//! Example demonstrating the usage of common sampler parameters and default LlamaSamplerParams
//!
//! This example shows how to retrieve and print both the default `common_sampler_params`
//! and the default `LlamaSamplerParams`. These parameters control the sampling behavior
//! during tasks like text generation. The `common_sampler_params` might be used for common
//! configurations across different sampling methods, while `LlamaSamplerParams` specifically
//! holds parameters for configuring a `LlamaSampler` instance.
//!
//! # Example Output
//! ```text
//! default sampler_params common_sampler_params {
//!     seed: 4294967295,
//!     n_prev: 64,
//!     n_probs: 0,
//!     min_keep: 0,
//!     top_k: 40,
//!     top_p: 0.95,
//!     min_p: 0.05,
//!     xtc_probability: 0.0,
//!     xtc_threshold: 0.1,
//!     tfs_z: 1.0,
//!     typ_p: 1.0,
//!     temp: 0.8,
//!     dynatemp_range: 0.0,
//!     dynatemp_exponent: 1.0,
//!     penalty_last_n: 64,
//!     penalty_repeat: 1.0,
//!     penalty_freq: 0.0,
//!     penalty_present: 0.0,
//!     dry_multiplier: 0.0,
//!     dry_base: 1.75,
//!     dry_allowed_length: 2,
//!     dry_penalty_last_n: -1,
//!     mirostat: 0,
//!     mirostat_tau: 5.0,
//!     mirostat_eta: 0.1,
//!     penalize_nl: false,
//!     ignore_eos: false,
//!     no_perf: false,
//!     dry_sequence_breakers: [
//!         "\n",
//!         ":",
//!         "\"",
//!         "*",
//!     ],
//!     samplers: [
//!         1,
//!         2,
//!         5,
//!         6,
//!         3,
//!         4,
//!         8,
//!         7,
//!     ],
//!     grammar: [],
//!     logit_bias: [],
//! }
//! common_sampler_params LlamaSamplerParams {
//!     top_k: 50,
//!     top_p: 0.9,
//!     temp: 0.8,
//!     seed: 1234,
//! }
//! ```
use llama_cpp_4::context::sampler::{LlamaSampler, LlamaSamplerParams};
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]

/// Demonstrates the usage of default common and LlamaSampler parameters.
///
/// This function first prints the default values for `common_sampler_params` (which are typical
/// settings for sampling tasks such as text generation). Then, it demonstrates how to create and
/// print default `LlamaSamplerParams` which are used to configure a `LlamaSampler` instance for
/// sampling tasks.
///
/// # Example Output
/// ```text
/// default sampler_params common_sampler_params {
///     seed: 4294967295,
///     n_prev: 64,
///     n_probs: 0,
///     min_keep: 0,
///     top_k: 40,
///     top_p: 0.95,
///     min_p: 0.05,
///     xtc_probability: 0.0,
///     xtc_threshold: 0.1,
///     tfs_z: 1.0,
///     typ_p: 1.0,
///     temp: 0.8,
///     dynatemp_range: 0.0,
///     dynatemp_exponent: 1.0,
///     penalty_last_n: 64,
///     penalty_repeat: 1.0,
///     penalty_freq: 0.0,
///     penalty_present: 0.0,
///     dry_multiplier: 0.0,
///     dry_base: 1.75,
///     dry_allowed_length: 2,
///     dry_penalty_last_n: -1,
///     mirostat: 0,
///     mirostat_tau: 5.0,
///     mirostat_eta: 0.1,
///     penalize_nl: false,
///     ignore_eos: false,
///     no_perf: false,
///     dry_sequence_breakers: [
///         "\n",
///         ":",
///         "\"",
///         "*",
///     ],
///     samplers: [
///         1,
///         2,
///         5,
///         6,
///         3,
///         4,
///         8,
///         7,
///     ],
///     grammar: [],
///     logit_bias: [],
/// }
/// common_sampler_params LlamaSamplerParams {
///     top_k: 50,
///     top_p: 0.9,
///     temp: 0.8,
///     seed: 1234,
/// }
/// ```
pub fn main() {
    // Retrieve and print the default common sampler parameters
    let params = llama_cpp_sys_4::common::common_sampler_params::default();
    println!("default sampler_params {:#?}", params);

    // Retrieve and print the default LlamaSamplerParams
    let params = LlamaSamplerParams::default();
    println!("common_sampler_params {:#?}", params);
}
