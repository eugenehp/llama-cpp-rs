//! Manual wrapper for values in llama.cpp/common/common.h

use crate::{
    ggml_numa_strategy, llama_attention_type, llama_pooling_type, llama_rope_scaling_type,
    llama_split_mode, GGML_NUMA_STRATEGY_DISABLED, LLAMA_ATTENTION_TYPE_UNSPECIFIED,
    LLAMA_DEFAULT_SEED, LLAMA_POOLING_TYPE_UNSPECIFIED, LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
    LLAMA_SPLIT_MODE_LAYER,
};

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

#[repr(C)]
#[derive(Debug, PartialEq)]
pub struct common_params {
    /// new tokens to predict
    pub n_predict: i32,
    /// context size
    pub n_ctx: i32,
    /// logical batch size for prompt processing (must be >=32 to use BLAS)
    pub n_batch: i32,
    /// physical batch size for prompt processing (must be >=32 to use BLAS)
    pub n_ubatch: i32,
    /// number of tokens to keep from initial prompt
    pub n_keep: i32,
    /// number of tokens to draft during speculative decoding
    pub n_draft: i32,
    /// max number of chunks to process (-1 = unlimited)
    pub n_chunks: i32,
    /// number of parallel sequences to decode
    pub n_parallel: i32,
    /// number of sequences to decode
    pub n_sequences: i32,
    // speculative decoding split probability
    pub p_split: f32,
    /// number of layers to store in VRAM (-1 - use default)
    pub n_gpu_layers: i32,
    /// number of layers to store in VRAM for the draft model (-1 - use default)
    pub n_gpu_layers_draft: i32,
    /// the GPU that is used for scratch and small tensors
    pub main_gpu: i32,
    /// how split tensors should be distributed across GPUs
    // pub tensor_split: [f32; 128usize],
    /// group-attention factor
    pub grp_attn_n: i32,
    /// group-attention width
    pub grp_attn_w: i32,
    /// print token count every n tokens (-1 = disabled)
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
    // pub cpuparams: cpu_params,
    // pub cpuparams_batch: cpu_params,
    // pub draft_cpuparams: cpu_params,
    // pub draft_cpuparams_batch: cpu_params,
    // pub cb_eval: ggml_backend_sched_eval_callback,
    // pub cb_eval_user_data: *mut ::core::ffi::c_void,
    pub numa: ggml_numa_strategy,
    pub split_mode: llama_split_mode,
    pub rope_scaling_type: llama_rope_scaling_type,
    pub pooling_type: llama_pooling_type,
    pub attention_type: llama_attention_type,
    pub sparams: common_sampler_params,
    // pub model: std___1_string,
    // pub model_draft: std___1_string,
    // pub model_alias: std___1_string,
    // pub model_url: std___1_string,
    // pub hf_token: std___1_string,
    // pub hf_repo: std___1_string,
    // pub hf_file: std___1_string,
    // pub prompt: std___1_string,
    // pub prompt_file: std___1_string,
    // pub path_prompt_cache: std___1_string,
    // pub input_prefix: std___1_string,
    // pub input_suffix: std___1_string,
    // pub logdir: std___1_string,
    // pub lookup_cache_static: std___1_string,
    // pub lookup_cache_dynamic: std___1_string,
    // pub logits_file: std___1_string,
    // pub rpc_servers: std___1_string,
    // pub in_files: [u64; 3usize],
    // pub antiprompt: [u64; 3usize],
    // pub kv_overrides: [u64; 3usize],
    // pub lora_init_without_apply: bool,
    // pub lora_adapters: [u64; 3usize],
    // pub control_vectors: [u64; 3usize],
    // pub verbosity: i32,
    // pub control_vector_layer_start: i32,
    // pub control_vector_layer_end: i32,
    // pub ppl_stride: i32,
    // pub ppl_output_type: i32,
    // pub hellaswag: bool,
    // pub hellaswag_tasks: usize,
    // pub winogrande: bool,
    // pub winogrande_tasks: usize,
    // pub multiple_choice: bool,
    // pub multiple_choice_tasks: usize,
    // pub kl_divergence: bool,
    // pub usage: bool,
    // pub use_color: bool,
    // pub special: bool,
    // pub interactive: bool,
    // pub interactive_first: bool,
    // pub conversation: bool,
    // pub prompt_cache_all: bool,
    // pub prompt_cache_ro: bool,
    // pub escape: bool,
    // pub multiline_input: bool,
    // pub simple_io: bool,
    // pub cont_batching: bool,
    // pub flash_attn: bool,
    // pub no_perf: bool,
    // pub ctx_shift: bool,
    // pub input_prefix_bos: bool,
    // pub logits_all: bool,
    // pub use_mmap: bool,
    // pub use_mlock: bool,
    // pub verbose_prompt: bool,
    // pub display_prompt: bool,
    // pub dump_kv_cache: bool,
    // pub no_kv_offload: bool,
    // pub warmup: bool,
    // pub check_tensors: bool,
    // pub cache_type_k: std___1_string,
    // pub cache_type_v: std___1_string,
    // pub mmproj: std___1_string,
    // pub image: [u64; 3usize],
    // pub embedding: bool,
    // pub embd_normalize: i32,
    // pub embd_out: std___1_string,
    // pub embd_sep: std___1_string,
    // pub reranking: bool,
    // pub port: i32,
    // pub timeout_read: i32,
    // pub timeout_write: i32,
    // pub n_threads_http: i32,
    // pub n_cache_reuse: i32,
    // pub hostname: std___1_string,
    // pub public_path: std___1_string,
    // pub chat_template: std___1_string,
    // pub enable_chat_template: bool,
    // pub api_keys: [u64; 3usize],
    // pub ssl_file_key: std___1_string,
    // pub ssl_file_cert: std___1_string,
    // pub webui: bool,
    // pub endpoint_slots: bool,
    // pub endpoint_props: bool,
    // pub endpoint_metrics: bool,
    // pub log_json: bool,
    // pub slot_save_path: std___1_string,
    // pub slot_prompt_similarity: f32,
    // pub is_pp_shared: bool,
    // pub n_pp: [u64; 3usize],
    // pub n_tg: [u64; 3usize],
    // pub n_pl: [u64; 3usize],
    // pub context_files: [u64; 3usize],
    // pub chunk_size: i32,
    // pub chunk_separator: std___1_string,
    // pub n_junk: i32,
    // pub i_pos: i32,
    // pub out_file: std___1_string,
    // pub n_out_freq: i32,
    // pub n_save_freq: i32,
    // pub i_chunk: i32,
    // pub process_output: bool,
    // pub compute_ppl: bool,
    // pub n_pca_batch: ::core::ffi::c_int,
    // pub n_pca_iterations: ::core::ffi::c_int,
    // pub cvector_dimre_method: dimre_method,
    // pub cvector_outfile: std___1_string,
    // pub cvector_positive_file: std___1_string,
    // pub cvector_negative_file: std___1_string,
    // pub spm_infill: bool,
    // pub lora_outfile: std___1_string,
    // pub batched_bench_output_jsonl: bool,
}

impl Default for common_params {
    fn default() -> Self {
        Self {
            n_predict: -1,          // new tokens to predict
            n_ctx: 0,               // context size
            n_batch: 2048, // logical batch size for prompt processing (must be >=32 to use BLAS)
            n_ubatch: 512, // physical batch size for prompt processing (must be >=32 to use BLAS)
            n_keep: 0,     // number of tokens to keep from initial prompt
            n_draft: 5,    // number of tokens to draft during speculative decoding
            n_chunks: -1,  // max number of chunks to process (-1 = unlimited)
            n_parallel: 1, // number of parallel sequences to decode
            n_sequences: 1, // number of sequences to decode
            p_split: 0.1,  // speculative decoding split probability
            n_gpu_layers: -1, // number of layers to store in VRAM (-1 - use default)
            n_gpu_layers_draft: -1, // number of layers to store in VRAM for the draft model (-1 - use default)
            main_gpu: 0,            // the GPU that is used for scratch and small tensors
            // tensor_split[128]     :   {0}, // how split tensors should be distributed across GPUs
            grp_attn_n: 1,         // group-attention factor
            grp_attn_w: 512,       // group-attention width
            n_print: -1,           // print token count every n tokens (-1 = disabled)
            rope_freq_base: 0.0,   // RoPE base frequency
            rope_freq_scale: 0.0,  // RoPE frequency scaling factor
            yarn_ext_factor: -1.0, // YaRN extrapolation mix factor
            yarn_attn_factor: 1.0, // YaRN magnitude scaling factor
            yarn_beta_fast: 32.0,  // YaRN low correction dim
            yarn_beta_slow: 1.0,   // YaRN high correction dim
            yarn_orig_ctx: 0,      // YaRN original context length
            defrag_thold: -1.0,    // KV cache defragmentation threshold

            // struct cpu_params cpuparams;
            // struct cpu_params cpuparams_batch;
            // struct cpu_params draft_cpuparams;
            // struct cpu_params draft_cpuparams_batch;

            // ggml_backend_sched_eval_callback cb_eval = nullptr;
            // void * cb_eval_user_data                 = nullptr;
            numa: GGML_NUMA_STRATEGY_DISABLED,

            split_mode: LLAMA_SPLIT_MODE_LAYER, // how to split the model across GPUs
            rope_scaling_type: LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
            pooling_type: LLAMA_POOLING_TYPE_UNSPECIFIED, // pooling type for embeddings
            attention_type: LLAMA_ATTENTION_TYPE_UNSPECIFIED, // attention type for embeddings

            sparams: common_sampler_params::default(),
        }
    }
}
