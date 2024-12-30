//! Safe wrapper around `llama_sampler`.

use std::borrow::Borrow;
use std::ffi::CString;
use std::fmt::{Debug, Formatter};
use std::ptr::NonNull;

use llama_cpp_sys_4::{common::*, *};

use crate::context::LlamaContext;
use crate::model::LlamaModel;
use crate::token::data_array::LlamaTokenDataArray;
use crate::token::LlamaToken;

#[cfg(target_os = "android")]
type CChar = u8;

#[cfg(not(target_os = "android"))]
type CChar = i8;

/// A safe wrapper around `llama_sampler`.
pub struct LlamaSampler {
    pub(crate) sampler: NonNull<llama_sampler>,
}

impl Debug for LlamaSampler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaSamplerChain").finish()
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
    /// create new sampler with default params
    pub fn new() -> Self {
        let sparams = unsafe { llama_sampler_chain_default_params() };

        Self {
            sampler: NonNull::new(unsafe { llama_sampler_chain_init(sparams) }).unwrap(),
        }
    }

    /// Sample and accept a token from the idx-th output of the last evaluation
    #[must_use]
    pub fn sample(&self, ctx: &LlamaContext, idx: i32) -> LlamaToken {
        let token =
            unsafe { llama_sampler_sample(self.sampler.as_ptr(), ctx.context.as_ptr(), idx) };

        LlamaToken(token)
    }

    /// Applies this sampler to a [`LlamaTokenDataArray`].
    pub fn apply(&mut self, data_array: &mut LlamaTokenDataArray) {
        data_array.apply_sampler(self);
    }

    /// Accepts a token from the sampler, possibly updating the internal state of certain samplers
    /// (e.g. grammar, repetition, etc.)
    pub fn accept(&mut self, token: LlamaToken) {
        unsafe { llama_sampler_accept(self.sampler.as_ptr(), token.0) }
    }

    /// Accepts several tokens from the sampler or context, possibly updating the internal state of
    /// certain samplers (e.g. grammar, repetition, etc.)
    pub fn accept_many(&mut self, tokens: impl IntoIterator<Item = impl Borrow<LlamaToken>>) {
        for token in tokens {
            unsafe { llama_sampler_accept(self.sampler.as_ptr(), token.borrow().0) }
        }
    }

    /// Accepts several tokens from the sampler or context, possibly updating the internal state of
    /// certain samplers (e.g. grammar, repetition, etc.)
    #[must_use]
    pub fn with_tokens(
        mut self,
        tokens: impl IntoIterator<Item = impl Borrow<LlamaToken>>,
    ) -> Self {
        self.accept_many(tokens);
        self
    }

    /// Combines a list of samplers into a single sampler that applies each component sampler one
    /// after another.
    ///
    /// If you are using a chain to select a token, the chain should always end with one of
    /// [`LlamaSampler::greedy`], [`LlamaSampler::dist`], [`LlamaSampler::mirostat`], and
    /// [`LlamaSampler::mirostat_v2`].
    #[must_use]
    pub fn chain(samplers: impl IntoIterator<Item = Self>, no_perf: bool) -> Self {
        unsafe {
            let mut params = llama_sampler_chain_default_params();
            params.no_perf = no_perf;
            let chain = llama_sampler_chain_init(params);

            for sampler in samplers {
                llama_sampler_chain_add(chain, sampler.sampler.as_ptr());

                // Do not call `llama_sampler_free` on the sampler, as the internal sampler is now
                // owned by the chain
                std::mem::forget(sampler);
            }

            Self {
                sampler: NonNull::new(chain).unwrap(),
            }
        }
    }

    /// Same as [`Self::chain`] with `no_perf = false`.
    ///
    /// # Example
    /// ```rust
    /// use llama_cpp_4::token::{
    ///    LlamaToken,
    ///    data::LlamaTokenData,
    ///    data_array::LlamaTokenDataArray
    /// };
    /// use llama_cpp_4::sampling::LlamaSampler;
    ///
    /// let mut data_array = LlamaTokenDataArray::new(vec![
    ///     LlamaTokenData::new(LlamaToken(0), 0., 0.),
    ///     LlamaTokenData::new(LlamaToken(1), 1., 0.),
    ///     LlamaTokenData::new(LlamaToken(2), 2., 0.),
    /// ], false);
    ///
    /// data_array.apply_sampler(&mut LlamaSampler::chain_simple([
    ///     LlamaSampler::temp(0.5),
    ///     LlamaSampler::greedy(),
    /// ]));
    ///
    /// assert_eq!(data_array.data[0].logit(), 0.);
    /// assert_eq!(data_array.data[1].logit(), 2.);
    /// assert_eq!(data_array.data[2].logit(), 4.);
    ///
    /// assert_eq!(data_array.data.len(), 3);
    /// assert_eq!(data_array.selected_token(), Some(LlamaToken(2)));
    /// ```
    #[must_use]
    pub fn chain_simple(samplers: impl IntoIterator<Item = Self>) -> Self {
        Self::chain(samplers, false)
    }

    /// Updates the logits l_i' = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original
    /// value, the rest are set to -inf
    ///
    /// # Example:
    /// ```rust
    /// use llama_cpp_4::token::{
    ///    LlamaToken,
    ///    data::LlamaTokenData,
    ///    data_array::LlamaTokenDataArray
    /// };
    /// use llama_cpp_4::sampling::LlamaSampler;
    ///
    /// let mut data_array = LlamaTokenDataArray::new(vec![
    ///     LlamaTokenData::new(LlamaToken(0), 0., 0.),
    ///     LlamaTokenData::new(LlamaToken(1), 1., 0.),
    ///     LlamaTokenData::new(LlamaToken(2), 2., 0.),
    /// ], false);
    ///
    /// data_array.apply_sampler(&mut LlamaSampler::temp(0.5));
    ///
    /// assert_eq!(data_array.data[0].logit(), 0.);
    /// assert_eq!(data_array.data[1].logit(), 2.);
    /// assert_eq!(data_array.data[2].logit(), 4.);
    /// ```
    #[must_use]
    pub fn temp(t: f32) -> Self {
        let sampler = unsafe { llama_sampler_init_temp(t) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Dynamic temperature implementation (a.k.a. entropy) described in the paper
    /// <https://arxiv.org/abs/2309.02772>.
    #[must_use]
    pub fn temp_ext(t: f32, delta: f32, exponent: f32) -> Self {
        let sampler = unsafe { llama_sampler_init_temp_ext(t, delta, exponent) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration"
    /// <https://arxiv.org/abs/1904.09751>
    ///
    /// # Example:
    /// ```rust
    /// use llama_cpp_4::token::{
    ///    LlamaToken,
    ///    data::LlamaTokenData,
    ///    data_array::LlamaTokenDataArray
    /// };
    /// use llama_cpp_4::sampling::LlamaSampler;
    ///
    /// let mut data_array = LlamaTokenDataArray::new(vec![
    ///     LlamaTokenData::new(LlamaToken(0), 0., 0.),
    ///     LlamaTokenData::new(LlamaToken(1), 1., 0.),
    ///     LlamaTokenData::new(LlamaToken(2), 2., 0.),
    ///     LlamaTokenData::new(LlamaToken(3), 3., 0.),
    /// ], false);
    ///
    /// data_array.apply_sampler(&mut LlamaSampler::top_k(2));
    ///
    /// assert_eq!(data_array.data.len(), 2);
    /// assert_eq!(data_array.data[0].id(), LlamaToken(3));
    /// assert_eq!(data_array.data[1].id(), LlamaToken(2));
    /// ```
    #[must_use]
    pub fn top_k(k: i32) -> Self {
        let sampler = unsafe { llama_sampler_init_top_k(k) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Locally Typical Sampling implementation described in the paper <https://arxiv.org/abs/2202.00666>.
    #[must_use]
    pub fn typical(p: f32, min_keep: usize) -> Self {
        let sampler = unsafe { llama_sampler_init_typical(p, min_keep) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration"
    /// <https://arxiv.org/abs/1904.09751>
    #[must_use]
    pub fn top_p(p: f32, min_keep: usize) -> Self {
        let sampler = unsafe { llama_sampler_init_top_p(p, min_keep) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Minimum P sampling as described in <https://github.com/ggerganov/llama.cpp/pull/3841>
    #[must_use]
    pub fn min_p(p: f32, min_keep: usize) -> Self {
        let sampler = unsafe { llama_sampler_init_min_p(p, min_keep) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// XTC sampler as described in <https://github.com/oobabooga/text-generation-webui/pull/6335>
    #[must_use]
    pub fn xtc(p: f32, t: f32, min_keep: usize, seed: u32) -> Self {
        let sampler = unsafe { llama_sampler_init_xtc(p, t, min_keep, seed) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Grammar sampler
    ///
    /// # Panics
    /// If either of ``grammar_str`` or ``grammar_root`` contain null bytes.
    #[must_use]
    pub fn grammar(model: &LlamaModel, grammar_str: &str, grammar_root: &str) -> Self {
        let grammar_str = CString::new(grammar_str).unwrap();
        let grammar_root = CString::new(grammar_root).unwrap();

        let sampler = unsafe {
            llama_sampler_init_grammar(
                model.model.as_ptr(),
                grammar_str.as_ptr(),
                grammar_root.as_ptr(),
            )
        };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// DRY sampler, designed by p-e-w, as described in:
    /// <https://github.com/oobabooga/text-generation-webui/pull/5677>, porting Koboldcpp
    /// implementation authored by pi6am: <https://github.com/LostRuins/koboldcpp/pull/982>
    ///
    /// # Panics
    /// If any string in ``seq_breakers`` contains null bytes.
    #[must_use]
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

    /// Penalizes tokens for being present in the context.
    ///
    /// Parameters:  
    /// - ``n_vocab``: [`LlamaModel::n_vocab`]
    /// - ``special_eos)id``: [`LlamaModel::token_eos`]
    /// - ``linefeed_id``: [`LlamaModel::token_nl`]
    /// - ``penalty_last_n``: last n tokens to penalize (0 = disable penalty, -1 = context size)
    ///// - ``penalty_repeat``: 1.0 = disabled
    ///// - ``penalty_freq``: 0.0 = disabled
    ///// - ``penalty_present``: 0.0 = disabled
    ///// - ``penalize_nl``: consider newlines as a repeatable token
    ///// - ``ignore_eos``: ignore the end-of-sequence token
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn penalties(
        n_vocab: i32,
        special_eos_id: f32,
        linefeed_id: f32,
        penalty_last_n: f32,
        // penalty_repeat: f32,
        // penalty_freq: f32,
        // penalty_present: f32,
        // penalize_nl: bool,
        // ignore_eos: bool,
    ) -> Self {
        let sampler = unsafe {
            llama_sampler_init_penalties(
                n_vocab,
                special_eos_id,
                linefeed_id,
                penalty_last_n,
                // penalty_repeat,
                // penalty_freq,
                // penalty_present,
                // penalize_nl,
                // ignore_eos,
            )
        };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Same as [`Self::penalties`], but with `n_vocab`, `special_eos_id`, and `linefeed_id`
    /// initialized from `model`, `penalize_nl = false`, and `ignore_eos = true`.
    ///
    /// Parameters:  
    /// - ``model``: The model's tokenizer to use to initialize the sampler.
    /// - ``penalty_last_n``: last n tokens to penalize (0 = disable penalty, -1 = context size)
    ///// - ``penalty_repeat``: 1.0 = disabled
    ///// - ``penalty_freq``: 0.0 = disabled
    ///// - ``penalty_present``: 0.0 = disabled
    #[must_use]
    pub fn penalties_simple(
        model: &LlamaModel,
        penalty_last_n: i32,
        // penalty_repeat: f32,
        // penalty_freq: f32,
        // penalty_present: f32,
    ) -> Self {
        Self::penalties(
            model.n_vocab(),
            model.token_eos().0 as f32,
            model.token_nl().0 as f32,
            penalty_last_n as f32,
            // penalty_repeat,
            // penalty_freq,
            // penalty_present,
            // false,
            // true,
        )
    }

    /// Mirostat 1.0 algorithm described in the paper <https://arxiv.org/abs/2007.14966>. Uses tokens instead of words.
    ///
    /// # Parameters:
    /// - ``n_vocab``: [`LlamaModel::n_vocab`]
    /// - ``seed``: Seed to initialize random generation with.
    /// - ``tau``: The target cross-entropy (or surprise) value you want to achieve for the
    ///     generated text. A higher value corresponds to more surprising or less predictable text,
    ///     while a lower value corresponds to less surprising or more predictable text.
    /// - ``eta``: The learning rate used to update `mu` based on the error between the target and
    ///     observed surprisal of the sampled word. A larger learning rate will cause `mu` to be
    ///     updated more quickly, while a smaller learning rate will result in slower updates.
    /// - ``m``: The number of tokens considered in the estimation of `s_hat`. This is an arbitrary
    ///     value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`.
    ///     In the paper, they use `m = 100`, but you can experiment with different values to see how
    ///     it affects the performance of the algorithm.
    #[must_use]
    pub fn mirostat(n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32) -> Self {
        let sampler = unsafe { llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Mirostat 2.0 algorithm described in the paper <https://arxiv.org/abs/2007.14966>. Uses tokens instead of words.
    ///
    /// # Parameters:
    /// - ``seed``: Seed to initialize random generation with.
    /// - ``tau``: The target cross-entropy (or surprise) value you want to achieve for the
    ///     generated text. A higher value corresponds to more surprising or less predictable text,
    ///     while a lower value corresponds to less surprising or more predictable text.
    /// - ``eta``: The learning rate used to update `mu` based on the error between the target and
    ///     observed surprisal of the sampled word. A larger learning rate will cause `mu` to be
    ///     updated more quickly, while a smaller learning rate will result in slower updates.
    #[must_use]
    pub fn mirostat_v2(seed: u32, tau: f32, eta: f32) -> Self {
        let sampler = unsafe { llama_sampler_init_mirostat_v2(seed, tau, eta) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Selects a token at random based on each token's probabilities
    #[must_use]
    pub fn dist(seed: u32) -> Self {
        let sampler = unsafe { llama_sampler_init_dist(seed) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Selects the most likely token
    ///
    /// # Example:
    /// ```rust
    /// use llama_cpp_4::token::{
    ///    LlamaToken,
    ///    data::LlamaTokenData,
    ///    data_array::LlamaTokenDataArray
    /// };
    /// use llama_cpp_4::sampling::LlamaSampler;
    ///
    /// let mut data_array = LlamaTokenDataArray::new(vec![
    ///     LlamaTokenData::new(LlamaToken(0), 0., 0.),
    ///     LlamaTokenData::new(LlamaToken(1), 1., 0.),
    /// ], false);
    ///
    /// data_array.apply_sampler(&mut LlamaSampler::greedy());
    ///
    /// assert_eq!(data_array.data.len(), 2);
    /// assert_eq!(data_array.selected_token(), Some(LlamaToken(1)));
    /// ```
    #[must_use]
    pub fn greedy() -> Self {
        let sampler = unsafe { llama_sampler_init_greedy() };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
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
        let params = common_sampler_params::default();

        let sampler = unsafe {
            let mut sparams = llama_sampler_chain_default_params();
            sparams.no_perf = false;

            let mut smpl = llama_sampler_chain_init(sparams);

            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params.top_k));
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params.top_p, 1));
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.temp));
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.seed));

            smpl
        };

        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }
}

impl Drop for LlamaSampler {
    fn drop(&mut self) {
        unsafe {
            llama_sampler_free(self.sampler.as_ptr());
        }
    }
}
