//! Sampler implementation for llama.cpp
//!
use std::{
    ffi::CString,
    fmt::{Debug, Formatter},
    ptr::NonNull,
};

use llama_cpp_sys_4::{
    common::common_sampler_params, llama_sampler_chain_add, llama_sampler_chain_default_params,
    llama_sampler_chain_init, llama_sampler_chain_params, llama_sampler_free,
    llama_sampler_init_dist, llama_sampler_init_dry, llama_sampler_init_min_p,
    llama_sampler_init_mirostat, llama_sampler_init_mirostat_v2, llama_sampler_init_penalties,
    llama_sampler_init_temp, llama_sampler_init_temp_ext, llama_sampler_init_top_k,
    llama_sampler_init_top_p, llama_sampler_init_typical, llama_sampler_init_xtc,
    llama_sampler_sample,
};

use crate::{model::LlamaModel, token::LlamaToken};

use super::LlamaContext;

#[cfg(target_os = "android")]
type CChar = u8;

#[cfg(not(target_os = "android"))]
type CChar = i8;

/// Safe wrapper around `llama_sampler`.
///
/// Original PR for the Sampler in llama.cpp
///
/// https://github.com/ggerganov/llama.cpp/pull/9294
#[allow(clippy::module_name_repetitions)]
pub struct LlamaSampler {
    pub(crate) sampler: NonNull<llama_cpp_sys_4::llama_sampler>,
}

impl Debug for LlamaSampler {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.debug_struct("LlamaSampler")
            .field("sampler", &self.sampler)
            .finish()
    }
}

impl Drop for LlamaSampler {
    fn drop(&mut self) {
        unsafe { llama_sampler_free(self.sampler.as_ptr()) }
    }
}

#[derive(Debug, Clone)]
#[allow(
    missing_docs,
    clippy::struct_excessive_bools,
    clippy::module_name_repetitions,
    dead_code
)]
pub struct LlamaSamplerParams {
    top_k: i32,
    top_p: f32,
    temp: f32,
    seed: u32,
}

impl LlamaSamplerParams {
    /// Set the seed of the context
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::sampler::LlamaSamplerParams;
    /// let params = LlamaSamplerParams::default();
    /// let params = params.with_seed(1234);
    /// assert_eq!(params.seed(), 1234);
    /// ```
    #[must_use]
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// Get the seed of the context
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llama_cpp_4::context::sampler::LlamaSamplerParams;
    /// let params = LlamaSamplerParams::default();
    ///     .with_seed(1234);
    /// assert_eq!(params.seed(), 1234);
    /// ```
    #[must_use]
    pub fn seed(&self) -> u32 {
        self.seed
    }
}

impl Default for LlamaSamplerParams {
    fn default() -> Self {
        Self {
            top_k: 50,
            top_p: 0.9,
            temp: 0.8,
            seed: 1234,
        }
    }
}

impl LlamaSampler {
    /// create new sampler with no_perf param
    /// * `no_perf` - whether to measure performance timings
    pub fn new(no_perf: Option<bool>) -> Self {
        let sparams = match no_perf {
            Some(no_perf) => llama_sampler_chain_params { no_perf: no_perf },
            None => unsafe { llama_sampler_chain_default_params() },
        };

        Self {
            sampler: NonNull::new(unsafe { llama_sampler_chain_init(sparams) }).unwrap(),
        }
    }

    /// sample next token
    pub fn sample(&self, ctx: &LlamaContext, idx: i32) -> LlamaToken {
        // println!("before sampler.sample");
        let token_id =
            unsafe { llama_sampler_sample(self.sampler.as_ptr(), ctx.context.as_ptr(), idx) };
        // println!("after sampler.sample");
        LlamaToken::new(token_id)
    }

    #[doc = " @details Top-K sampling described in academic paper \"The Curious Case of Neural Text Degeneration\" https://arxiv.org/abs/1904.09751"]
    pub fn with_top_k(&self, top_k: i32) -> &Self {
        unsafe {
            llama_sampler_chain_add(self.sampler.as_ptr(), llama_sampler_init_top_k(top_k));
        }

        self
    }

    #[doc = " @details Nucleus sampling described in academic paper \"The Curious Case of Neural Text Degeneration\" https://arxiv.org/abs/1904.09751"]
    pub fn with_top_p(&self, top_p: f32) -> &Self {
        unsafe {
            llama_sampler_chain_add(self.sampler.as_ptr(), llama_sampler_init_top_p(top_p, 1));
        }

        self
    }

    #[doc = " #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf"]
    pub fn with_temp(&self, temp: f32) -> &Self {
        unsafe {
            llama_sampler_chain_add(self.sampler.as_ptr(), llama_sampler_init_temp(temp));
        }

        self
    }

    #[doc = " @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772."]
    pub fn with_temp_ext(&self, temp: f32, delta: f32, exponent: f32) -> &Self {
        unsafe {
            llama_sampler_chain_add(
                self.sampler.as_ptr(),
                llama_sampler_init_temp_ext(temp, delta, exponent),
            );
        }

        self
    }

    #[doc = " @details Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841"]
    pub fn with_min_p(&self, p: f32, min_keep: usize) -> &Self {
        unsafe {
            llama_sampler_chain_add(self.sampler.as_ptr(), llama_sampler_init_min_p(p, min_keep));
        }

        self
    }

    #[doc = " @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666."]
    pub fn with_typical(&self, p: f32, min_keep: usize) -> &Self {
        unsafe {
            llama_sampler_chain_add(
                self.sampler.as_ptr(),
                llama_sampler_init_typical(p, min_keep),
            );
        }

        self
    }

    #[doc = " @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335"]
    pub fn with_xtc(&self, p: f32, t: f32, min_keep: usize, seed: u32) -> &Self {
        unsafe {
            llama_sampler_chain_add(
                self.sampler.as_ptr(),
                llama_sampler_init_xtc(p, t, min_keep, seed),
            );
        }

        self
    }

    #[doc = " @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.\n @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.\n @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.\n @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.\n @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.\n @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal."]
    pub fn with_mirostat(&self, n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32) -> &Self {
        unsafe {
            llama_sampler_chain_add(
                self.sampler.as_ptr(),
                llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m),
            );
        }

        self
    }

    #[doc = " @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.\n @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.\n @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.\n @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.\n @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal."]
    pub fn with_mirostat_v2(&self, seed: u32, tau: f32, eta: f32) -> &Self {
        unsafe {
            llama_sampler_chain_add(
                self.sampler.as_ptr(),
                llama_sampler_init_mirostat_v2(seed, tau, eta),
            );
        }

        self
    }

    /// init seed distribution
    pub fn with_seed(&self, seed: u32) -> &Self {
        unsafe {
            llama_sampler_chain_add(self.sampler.as_ptr(), llama_sampler_init_dist(seed));
        }

        self
    }

    /// @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
    pub fn dry(
        &self,
        model: &LlamaModel,
        multiplier: f32,
        base: f32,
        allowed_length: i32,
        penalty_last_n: i32,
        seq_breakers: impl IntoIterator<Item = impl AsRef<[u8]>>,
    ) -> Self {
        let seq_breakers: Vec<CString> = seq_breakers
            .into_iter()
            .map(|s| CString::new(s.as_ref()).unwrap())
            .collect();
        let mut seq_breaker_pointers: Vec<*const CChar> =
            seq_breakers.iter().map(|s| s.as_ptr()).collect();

        let sampler = unsafe {
            llama_sampler_init_dry(
                model.model.as_ptr(),
                multiplier,
                base,
                allowed_length,
                penalty_last_n,
                seq_breaker_pointers.as_mut_ptr(),
                seq_breaker_pointers.len(),
            )
        };

        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// init with penalties sampler
    pub fn with_penalties(
        &self,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) -> &Self {
        // TODO: should this be pulled from context instead?
        let penalty_last_n: i32 = 0;

        unsafe {
            llama_sampler_chain_add(
                self.sampler.as_ptr(),
                llama_sampler_init_penalties(
                    penalty_last_n,
                    penalty_repeat,
                    penalty_freq,
                    penalty_present,
                ),
            );
        }

        self
    }

    /// Creates a new instance of `LlamaSampler` with common sampling parameters.
    ///
    /// This function initializes a `LlamaSampler` using default values from `common_sampler_params`
    /// and configures it with common settings such as `top_k`, `top_p`, `temperature`, and `seed` values.
    ///
    /// # Returns
    /// A `LlamaSampler` instance configured with the common sampling parameters.
    ///
    /// # Example
    /// ```rust
    /// let sampler = LlamaSampler::common();
    /// ```
    pub fn common() -> Self {
        let sampler = LlamaSampler::new(None);
        let params = common_sampler_params::default();

        sampler
            .with_top_k(params.top_k)
            .with_top_p(params.top_p)
            .with_temp(params.temp)
            .with_seed(params.seed);

        sampler
    }
}

impl Default for LlamaSampler {
    /// Creates a default instance of `LlamaSampler` configured with `LlamaSamplerParams`.
    ///
    /// This function initializes a `LlamaSampler` with the default parameters defined in
    /// `LlamaSamplerParams`, which include common sampling settings such as `top_k`, `top_p`,
    /// `temperature`, and `seed`. These default parameters help set up a sampler with sensible
    /// starting values for typical use cases.
    ///
    /// # Returns
    /// A `LlamaSampler` instance, configured with default sampling parameters.
    ///
    /// # Example
    /// ```rust
    /// let default_sampler = LlamaSampler::default();
    /// ```
    fn default() -> Self {
        let sampler = LlamaSampler::new(None);
        let params = LlamaSamplerParams::default();

        sampler
            .with_top_k(params.top_k)
            .with_top_p(params.top_p)
            .with_temp(params.temp)
            .with_seed(params.seed);

        sampler
    }
}
