//! Manual wrapper for values in llama.cpp/common/common.h

use crate::LLAMA_DEFAULT_SEED;

pub const COMMON_SAMPLER_TYPE_NONE: common_sampler_type = 0;
pub const COMMON_SAMPLER_TYPE_DRY: common_sampler_type = 1;
pub const COMMON_SAMPLER_TYPE_TOP_K: common_sampler_type = 2;
pub const COMMON_SAMPLER_TYPE_TOP_P: common_sampler_type = 3;
pub const COMMON_SAMPLER_TYPE_MIN_P: common_sampler_type = 4;
pub const COMMON_SAMPLER_TYPE_TFS_Z: common_sampler_type = 5;
pub const COMMON_SAMPLER_TYPE_TYPICAL_P: common_sampler_type = 6;
pub const COMMON_SAMPLER_TYPE_TEMPERATURE: common_sampler_type = 7;
pub const COMMON_SAMPLER_TYPE_XTC: common_sampler_type = 8;
pub const COMMON_SAMPLER_TYPE_INFILL: common_sampler_type = 9;
pub type common_sampler_type = ::core::ffi::c_uint;

/// common sampler params
#[repr(C)]
#[derive(Debug, PartialEq)]
pub struct common_sampler_params {
    /// the seed used to initialize `llama_sampler`
    pub seed: u32,
    /// number of previous tokens to remember
    pub n_prev: i32,
    /// if greater than 0, output the probabilities of top `n_probs` tokens.
    pub n_probs: i32,
    /// 0 = disabled, otherwise samplers should return at least `min_keep` tokens
    pub min_keep: i32,
    /// <= 0 to use vocab size
    pub top_k: i32,
    /// 1.0 = disabled
    pub top_p: f32,
    /// 0.0 = disabled
    pub min_p: f32,
    /// 0.0 = disabled
    pub xtc_probability: f32,
    /// > 0.5 disables XTC
    pub xtc_threshold: f32,
    /// 1.0 = disabled
    pub tfs_z: f32,
    /// typical_p, 1.0 = disabled
    pub typ_p: f32,
    /// <= 0.0 to sample greedily, 0.0 to not output probabilities
    pub temp: f32,
    /// 0.0 = disabled
    pub dynatemp_range: f32,
    /// controls how entropy maps to temperature in dynamic temperature sampler
    pub dynatemp_exponent: f32,
    /// last n tokens to penalize (0 = disable penalty, -1 = context size)
    pub penalty_last_n: i32,
    /// 1.0 = disabled
    pub penalty_repeat: f32,
    /// 0.0 = disabled
    pub penalty_freq: f32,
    /// 0.0 = disabled
    pub penalty_present: f32,
    /// 0.0 = disabled;      DRY repetition penalty for tokens extending repetition:
    pub dry_multiplier: f32,
    /// 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length)
    pub dry_base: f32,
    /// tokens extending repetitions beyond this receive penalty
    pub dry_allowed_length: i32,
    /// how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)
    pub dry_penalty_last_n: i32,
    /// 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    pub mirostat: i32,
    /// target entropy
    pub mirostat_tau: f32,
    /// learning rate
    pub mirostat_eta: f32,
    /// consider newlines as a repeatable token
    pub penalize_nl: bool,
    pub ignore_eos: bool,
    /// disable performance metrics
    pub no_perf: bool,
    pub dry_sequence_breakers: Vec<String>,
    pub samplers: Vec<common_sampler_type>,
    pub grammar: Vec<String>,
    pub logit_bias: Vec<(i32, f64)>,
}

impl Default for common_sampler_params {
    fn default() -> Self {
        Self {
            seed: LLAMA_DEFAULT_SEED, // the seed used to initialize llama_sampler
            n_prev: 64,               // number of previous tokens to remember
            n_probs: 0, // if greater than 0, output the probabilities of top n_probs tokens.
            min_keep: 0, // 0 = disabled, otherwise samplers should return at least min_keep tokens
            top_k: 40,  // <= 0 to use vocab size
            top_p: 0.95, // 1.0 = disabled
            min_p: 0.05, // 0.0 = disabled
            xtc_probability: 0.00, // 0.0 = disabled
            xtc_threshold: 0.10, // > 0.5 disables XTC
            tfs_z: 1.00, // 1.0 = disabled
            typ_p: 1.00, // typical_p, 1.0 = disabled
            temp: 0.80, // <= 0.0 to sample greedily, 0.0 to not output probabilities
            dynatemp_range: 0.00, // 0.0 = disabled
            dynatemp_exponent: 1.00, // controls how entropy maps to temperature in dynamic temperature sampler
            penalty_last_n: 64, // last n tokens to penalize (0 = disable penalty, -1 = context size)
            penalty_repeat: 1.00, // 1.0 = disabled
            penalty_freq: 0.00, // 0.0 = disabled
            penalty_present: 0.00, // 0.0 = disabled
            dry_multiplier: 0.0, // 0.0 = disabled;      DRY repetition penalty for tokens extending repetition:
            dry_base: 1.75, // 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length)
            dry_allowed_length: 2, // tokens extending repetitions beyond this receive penalty
            dry_penalty_last_n: -1, // how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)
            mirostat: 0,            // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
            mirostat_tau: 5.00,     // target entropy
            mirostat_eta: 0.10,     // learning rate
            penalize_nl: false,     // consider newlines as a repeatable token
            ignore_eos: false,
            no_perf: false, // disable performance metrics

            dry_sequence_breakers: vec!["\n".into(), ":".into(), "\"".into(), "*".into()], // default sequence breakers for DRY

            samplers: vec![
                COMMON_SAMPLER_TYPE_DRY,
                COMMON_SAMPLER_TYPE_TOP_K,
                COMMON_SAMPLER_TYPE_TFS_Z,
                COMMON_SAMPLER_TYPE_TYPICAL_P,
                COMMON_SAMPLER_TYPE_TOP_P,
                COMMON_SAMPLER_TYPE_MIN_P,
                COMMON_SAMPLER_TYPE_XTC,
                COMMON_SAMPLER_TYPE_TEMPERATURE,
            ],

            grammar: vec![], // optional BNF-like grammar to constrain sampling

            logit_bias: vec![], // logit biases to apply
        }
    }
}

