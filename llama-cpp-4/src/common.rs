//! exposing common llama cpp structures like CommonParams
pub use llama_cpp_sys_4::common::*;

/// Struct containing common parameters for processing.
/// ## See more
/// https://github.com/ggerganov/llama.cpp/blob/master/common/common.h#L109
#[derive(Debug, Clone)]
pub struct CommonParams {
    /// New tokens to predict
    pub n_predict: i32,

    /// Context size
    pub n_ctx: i32,

    /// Logical batch size for prompt processing (must be >=32 to use BLAS)
    pub n_batch: i32,

    /// Physical batch size for prompt processing (must be >=32 to use BLAS)
    pub n_ubatch: i32,

    /// Number of tokens to keep from initial prompt
    pub n_keep: i32,

    /// Max number of chunks to process (-1 = unlimited)
    pub n_chunks: i32,

    /// Number of parallel sequences to decode
    pub n_parallel: i32,

    /// Number of sequences to decode
    pub n_sequences: i32,

    /// Group-attention factor
    pub grp_attn_n: i32,

    /// Group-attention width
    pub grp_attn_w: i32,

    /// Print token count every n tokens (-1 = disabled)
    pub n_print: i32,

    /// RoPE base frequency
    pub rope_freq_base: f32,

    /// RoPE frequency scaling factor
    pub rope_freq_scale: f32,

    /// YaRN extrapolation mix factor
    pub yarn_ext_factor: f32,

    /// YaRN magnitude scaling factor
    pub yarn_attn_factor: f32,

    /// YaRN low correction dim
    pub yarn_beta_fast: f32,

    /// YaRN high correction dim
    pub yarn_beta_slow: f32,

    /// YaRN original context length
    pub yarn_orig_ctx: i32,

    /// KV cache defragmentation threshold
    pub defrag_thold: f32,

    /// prompt for the model to consume
    pub prompt: String,
}

impl Default for CommonParams {
    fn default() -> Self {
        CommonParams {
            n_predict: -1,
            n_ctx: 4096,
            n_batch: 2048,
            n_ubatch: 512,
            n_keep: 0,
            n_chunks: -1,
            n_parallel: 1,
            n_sequences: 1,
            grp_attn_n: 1,
            grp_attn_w: 512,
            n_print: -1,
            rope_freq_base: 0.0,
            rope_freq_scale: 0.0,
            yarn_ext_factor: -1.0,
            yarn_attn_factor: 1.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_orig_ctx: 0,
            defrag_thold: 0.1,
            prompt: "".into(),
        }
    }
}
