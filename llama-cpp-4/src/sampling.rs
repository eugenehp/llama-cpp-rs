//! Safe wrapper around `llama_sampler`.

use std::borrow::Borrow;
use std::ffi::{c_char, CString};
use std::fmt::{Debug, Formatter};
use std::ptr::NonNull;

use llama_cpp_sys_4::{
    common::common_sampler_params, llama_logit_bias, llama_sampler, llama_sampler_accept,
    llama_sampler_chain_add, llama_sampler_chain_default_params, llama_sampler_chain_init,
    llama_sampler_chain_n, llama_sampler_chain_remove, llama_sampler_clone, llama_sampler_free,
    llama_sampler_get_seed, llama_sampler_init_adaptive_p, llama_sampler_init_dist,
    llama_sampler_init_dry, llama_sampler_init_grammar, llama_sampler_init_grammar_lazy,
    llama_sampler_init_grammar_lazy_patterns, llama_sampler_init_greedy, llama_sampler_init_infill,
    llama_sampler_init_logit_bias, llama_sampler_init_min_p, llama_sampler_init_mirostat,
    llama_sampler_init_mirostat_v2, llama_sampler_init_penalties, llama_sampler_init_temp,
    llama_sampler_init_temp_ext, llama_sampler_init_top_k, llama_sampler_init_top_n_sigma,
    llama_sampler_init_top_p, llama_sampler_init_typical, llama_sampler_init_xtc,
    llama_sampler_name, llama_sampler_reset, llama_sampler_sample,
};

use crate::context::LlamaContext;
use crate::model::LlamaModel;
use crate::token::data_array::LlamaTokenDataArray;
use crate::token::LlamaToken;

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
    /// use llama_cpp_4::sampling::LlamaSamplerParams;
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
    /// use llama_cpp_4::sampling::LlamaSamplerParams;
    /// let params = LlamaSamplerParams::default()
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

impl Default for LlamaSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl LlamaSampler {
    /// Create new sampler with default params.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
    #[must_use]
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
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
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
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
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

    /// Updates the logits `l_i`' = `l_i/t`. When `t <= 0.0`, the maximum logit is kept at its original
    /// value, the rest are set to -inf.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
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
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
    #[must_use]
    pub fn temp_ext(t: f32, delta: f32, exponent: f32) -> Self {
        let sampler = unsafe { llama_sampler_init_temp_ext(t, delta, exponent) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration"
    /// <https://arxiv.org/abs/1904.09751>.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
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
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
    #[must_use]
    pub fn typical(p: f32, min_keep: usize) -> Self {
        let sampler = unsafe { llama_sampler_init_typical(p, min_keep) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration"
    /// <https://arxiv.org/abs/1904.09751>.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
    #[must_use]
    pub fn top_p(p: f32, min_keep: usize) -> Self {
        let sampler = unsafe { llama_sampler_init_top_p(p, min_keep) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Minimum P sampling as described in <https://github.com/ggerganov/llama.cpp/pull/3841>.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
    #[must_use]
    pub fn min_p(p: f32, min_keep: usize) -> Self {
        let sampler = unsafe { llama_sampler_init_min_p(p, min_keep) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// XTC sampler as described in <https://github.com/oobabooga/text-generation-webui/pull/6335>.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
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
    /// - If either of `grammar_str` or `grammar_root` contain null bytes.
    /// - If llama.cpp returns a null pointer.
    #[must_use]
    pub fn grammar(model: &LlamaModel, grammar_str: &str, grammar_root: &str) -> Self {
        let grammar_str = CString::new(grammar_str).unwrap();
        let grammar_root = CString::new(grammar_root).unwrap();

        let sampler = unsafe {
            llama_sampler_init_grammar(
                model.get_vocab().vocab.as_ref(),
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
    /// - If any string in `seq_breakers` contains null bytes.
    /// - If llama.cpp returns a null pointer.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn dry(
        &self,
        model: &LlamaModel,
        n_ctx_train: i32,
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
        // CString::as_ptr() returns *const c_char, which matches what the binding
        // expects on every platform (signed on macOS/x86 Linux, unsigned on musl ARM).
        let mut seq_breaker_pointers: Vec<*const c_char> =
            seq_breakers.iter().map(|s| s.as_ptr()).collect();

        let sampler = unsafe {
            llama_sampler_init_dry(
                model.get_vocab().vocab.as_ref(),
                n_ctx_train,
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
    /// - `penalty_last_n`: last n tokens to penalize (0 = disable penalty, -1 = context size)
    /// - `penalty_repeat`: repetition penalty (1.0 = disabled, >1.0 = penalize repeats)
    /// - `penalty_freq`: frequency penalty (0.0 = disabled)
    /// - `penalty_present`: presence penalty (0.0 = disabled)
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn penalties(
        penalty_last_n: i32,
        penalty_repeat: f32,
        penalty_freq: f32,
        penalty_present: f32,
    ) -> Self {
        let sampler = unsafe {
            llama_sampler_init_penalties(
                penalty_last_n,
                penalty_repeat,
                penalty_freq,
                penalty_present,
            )
        };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Same as [`Self::penalties`] with sensible defaults:
    /// `penalty_freq = 0.0` and `penalty_present = 0.0`.
    ///
    /// Parameters:
    /// - `penalty_last_n`: last n tokens to penalize (0 = disable, -1 = context size)
    /// - `penalty_repeat`: repetition penalty (1.0 = disabled)
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
    #[must_use]
    pub fn penalties_simple(penalty_last_n: i32, penalty_repeat: f32) -> Self {
        Self::penalties(
            #[allow(clippy::cast_precision_loss)]
            {
                penalty_last_n as i32
            },
            #[allow(clippy::cast_precision_loss)]
            {
                penalty_repeat as f32
            },
            #[allow(clippy::cast_precision_loss)]
            {
                0.0 as f32
            },
            #[allow(clippy::cast_precision_loss)]
            {
                0.0 as f32
            },
        )
    }

    /// Mirostat 1.0 algorithm described in the paper <https://arxiv.org/abs/2007.14966>. Uses tokens instead of words.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
    ///
    /// # Parameters:
    /// - `n_vocab`: [`LlamaModel::n_vocab`]
    /// - `seed`: Seed to initialize random generation with.
    /// - `tau`: The target cross-entropy (or surprise) value you want to achieve for the
    ///   generated text. A higher value corresponds to more surprising or less predictable text,
    ///   while a lower value corresponds to less surprising or more predictable text.
    /// - `eta`: The learning rate used to update `mu` based on the error between the target and
    ///   observed surprisal of the sampled word. A larger learning rate will cause `mu` to be
    ///   updated more quickly, while a smaller learning rate will result in slower updates.
    /// - `m`: The number of tokens considered in the estimation of `s_hat`. This is an arbitrary
    ///   value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`.
    ///   In the paper, they use `m = 100`, but you can experiment with different values to see how
    ///   it affects the performance of the algorithm.
    #[must_use]
    pub fn mirostat(n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32) -> Self {
        let sampler = unsafe { llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Mirostat 2.0 algorithm described in the paper <https://arxiv.org/abs/2007.14966>. Uses tokens instead of words.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
    ///
    /// # Parameters:
    /// - `seed`: Seed to initialize random generation with.
    /// - `tau`: The target cross-entropy (or surprise) value you want to achieve for the
    ///   generated text. A higher value corresponds to more surprising or less predictable text,
    ///   while a lower value corresponds to less surprising or more predictable text.
    /// - `eta`: The learning rate used to update `mu` based on the error between the target and
    ///   observed surprisal of the sampled word. A larger learning rate will cause `mu` to be
    ///   updated more quickly, while a smaller learning rate will result in slower updates.
    #[must_use]
    pub fn mirostat_v2(seed: u32, tau: f32, eta: f32) -> Self {
        let sampler = unsafe { llama_sampler_init_mirostat_v2(seed, tau, eta) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Selects a token at random based on each token's probabilities.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
    #[must_use]
    pub fn dist(seed: u32) -> Self {
        let sampler = unsafe { llama_sampler_init_dist(seed) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Selects the most likely token.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
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

    /// Top-N sigma sampling.
    ///
    /// Keeps tokens within N standard deviations of the maximum logit.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
    #[must_use]
    pub fn top_n_sigma(n: f32) -> Self {
        let sampler = unsafe { llama_sampler_init_top_n_sigma(n) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Adaptive P sampling.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
    ///
    /// # Parameters
    /// - `target`: Target probability.
    /// - `decay`: Decay rate.
    /// - `seed`: Random seed.
    #[must_use]
    pub fn adaptive_p(target: f32, decay: f32, seed: u32) -> Self {
        let sampler = unsafe { llama_sampler_init_adaptive_p(target, decay, seed) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Logit bias sampler.
    ///
    /// Applies additive bias to specific token logits before sampling.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
    ///
    /// # Parameters
    /// - `n_vocab`: Number of tokens in the vocabulary ([`LlamaModel::n_vocab`]).
    /// - `biases`: Slice of `(token_id, bias)` pairs.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    #[must_use]
    pub fn logit_bias(n_vocab: i32, biases: &[(LlamaToken, f32)]) -> Self {
        let logit_biases: Vec<llama_logit_bias> = biases
            .iter()
            .map(|(token, bias)| llama_logit_bias {
                token: token.0,
                bias: *bias,
            })
            .collect();

        let sampler = unsafe {
            llama_sampler_init_logit_bias(n_vocab, logit_biases.len() as i32, logit_biases.as_ptr())
        };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Infill sampler.
    ///
    /// Reorders token probabilities for fill-in-the-middle tasks.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
    #[must_use]
    pub fn infill(model: &LlamaModel) -> Self {
        let sampler = unsafe { llama_sampler_init_infill(model.get_vocab().vocab.as_ref()) };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Get the seed of the sampler.
    ///
    /// Returns `LLAMA_DEFAULT_SEED` if the sampler is not seeded.
    #[must_use]
    pub fn get_seed(&self) -> u32 {
        unsafe { llama_sampler_get_seed(self.sampler.as_ptr()) }
    }

    /// Get the name of the sampler.
    ///
    /// # Panics
    ///
    /// Panics if the name is not valid UTF-8.
    #[must_use]
    pub fn name(&self) -> String {
        let c_str = unsafe { llama_sampler_name(self.sampler.as_ptr()) };
        let c_str = unsafe { std::ffi::CStr::from_ptr(c_str) };
        c_str
            .to_str()
            .expect("sampler name is not valid UTF-8")
            .to_owned()
    }

    /// Reset the sampler state (e.g. grammar, repetition penalties).
    pub fn reset(&mut self) {
        unsafe { llama_sampler_reset(self.sampler.as_ptr()) }
    }

    /// Get the number of samplers in a chain.
    ///
    /// Returns 0 if this sampler is not a chain.
    #[must_use]
    pub fn chain_n(&self) -> i32 {
        unsafe { llama_sampler_chain_n(self.sampler.as_ptr()) }
    }

    /// Remove and return the sampler at position `i` from a chain.
    ///
    /// The returned sampler is owned by the caller and will be freed on drop.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of range or if llama.cpp returns a null pointer.
    #[must_use]
    pub fn chain_remove(&mut self, i: i32) -> Self {
        let sampler = unsafe { llama_sampler_chain_remove(self.sampler.as_ptr(), i) };
        Self {
            sampler: NonNull::new(sampler).expect("chain_remove returned null"),
        }
    }

    /// Grammar sampler with lazy activation.
    ///
    /// The grammar is only activated when one of the trigger words or trigger tokens is encountered.
    ///
    /// # Panics
    /// - If `grammar_str` or `grammar_root` contain null bytes.
    /// - If any trigger word contains null bytes.
    /// - If llama.cpp returns a null pointer.
    #[must_use]
    pub fn grammar_lazy(
        model: &LlamaModel,
        grammar_str: &str,
        grammar_root: &str,
        trigger_words: &[&str],
        trigger_tokens: &[LlamaToken],
    ) -> Self {
        let grammar_str = CString::new(grammar_str).unwrap();
        let grammar_root = CString::new(grammar_root).unwrap();
        let trigger_cstrings: Vec<CString> = trigger_words
            .iter()
            .map(|w| CString::new(*w).unwrap())
            .collect();
        let mut trigger_ptrs: Vec<*const c_char> =
            trigger_cstrings.iter().map(|s| s.as_ptr()).collect();

        let sampler = unsafe {
            llama_sampler_init_grammar_lazy(
                model.get_vocab().vocab.as_ref(),
                grammar_str.as_ptr(),
                grammar_root.as_ptr(),
                trigger_ptrs.as_mut_ptr(),
                trigger_ptrs.len(),
                trigger_tokens.as_ptr().cast(),
                trigger_tokens.len(),
            )
        };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Grammar sampler with lazy activation via regex patterns.
    ///
    /// The grammar is only activated when one of the trigger patterns or trigger tokens matches.
    ///
    /// # Panics
    /// - If `grammar_str` or `grammar_root` contain null bytes.
    /// - If any trigger pattern contains null bytes.
    /// - If llama.cpp returns a null pointer.
    #[must_use]
    pub fn grammar_lazy_patterns(
        model: &LlamaModel,
        grammar_str: &str,
        grammar_root: &str,
        trigger_patterns: &[&str],
        trigger_tokens: &[LlamaToken],
    ) -> Self {
        let grammar_str = CString::new(grammar_str).unwrap();
        let grammar_root = CString::new(grammar_root).unwrap();
        let pattern_cstrings: Vec<CString> = trigger_patterns
            .iter()
            .map(|w| CString::new(*w).unwrap())
            .collect();
        let mut pattern_ptrs: Vec<*const c_char> =
            pattern_cstrings.iter().map(|s| s.as_ptr()).collect();

        let sampler = unsafe {
            llama_sampler_init_grammar_lazy_patterns(
                model.get_vocab().vocab.as_ref(),
                grammar_str.as_ptr(),
                grammar_root.as_ptr(),
                pattern_ptrs.as_mut_ptr(),
                pattern_ptrs.len(),
                trigger_tokens.as_ptr().cast(),
                trigger_tokens.len(),
            )
        };
        Self {
            sampler: NonNull::new(sampler).unwrap(),
        }
    }

    /// Clone this sampler.
    ///
    /// Creates an independent copy of this sampler with the same state.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
    #[must_use]
    pub fn clone_sampler(&self) -> Self {
        let sampler = unsafe { llama_sampler_clone(self.sampler.as_ptr()) };
        Self {
            sampler: NonNull::new(sampler).expect("sampler_clone returned null"),
        }
    }

    /// Print sampler performance data.
    pub fn perf_print(&self) {
        unsafe { llama_cpp_sys_4::llama_perf_sampler_print(self.sampler.as_ptr()) }
    }

    /// Reset sampler performance counters.
    pub fn perf_reset(&mut self) {
        unsafe { llama_cpp_sys_4::llama_perf_sampler_reset(self.sampler.as_ptr()) }
    }

    /// Get sampler performance data.
    #[must_use]
    pub fn perf_data(&self) -> llama_cpp_sys_4::llama_perf_sampler_data {
        unsafe { llama_cpp_sys_4::llama_perf_sampler(self.sampler.as_ptr()) }
    }

    /// Get a non-owning reference to the `i`th sampler in a chain.
    ///
    /// # Safety
    ///
    /// The returned pointer is owned by the chain. Do not free it or use it
    /// after the chain is dropped or modified.
    #[must_use]
    pub unsafe fn chain_get_ptr(&self, i: i32) -> *mut llama_sampler {
        llama_cpp_sys_4::llama_sampler_chain_get(self.sampler.as_ptr(), i)
    }

    /// Create a sampler from a raw interface and context.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `iface` and `ctx` are valid and that the
    /// interface functions properly manage the context lifetime.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
    #[must_use]
    pub unsafe fn from_raw(
        iface: *mut llama_cpp_sys_4::llama_sampler_i,
        ctx: llama_cpp_sys_4::llama_sampler_context_t,
    ) -> Self {
        let sampler = llama_cpp_sys_4::llama_sampler_init(iface, ctx);
        Self {
            sampler: NonNull::new(sampler).expect("sampler_init returned null"),
        }
    }

    /// Creates a new instance of `LlamaSampler` with common sampling parameters.
    ///
    /// This function initializes a `LlamaSampler` using default values from `common_sampler_params`
    /// and configures it with common settings such as `top_k`, `top_p`, `temperature`, and `seed` values.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a null pointer.
    ///
    /// # Returns
    /// A `LlamaSampler` instance configured with the common sampling parameters.
    #[must_use]
    pub fn common() -> Self {
        let params = common_sampler_params::default();

        let sampler = unsafe {
            let mut sparams = llama_sampler_chain_default_params();
            sparams.no_perf = false;

            let smpl = llama_sampler_chain_init(sparams);

            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params.top_k));
            llama_sampler_chain_add(
                smpl,
                #[allow(clippy::cast_sign_loss)]
                llama_sampler_init_top_p(params.top_p, params.min_keep as usize),
            );
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.temp));
            #[allow(clippy::cast_sign_loss)]
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
