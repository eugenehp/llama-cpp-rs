//! Unit tests for [`LlamaContextParams`] setters and getters.

use llama_cpp_4::context::params::{
    LlamaAttentionType, LlamaContextParams, LlamaFlashAttnType, ParamsCloneError, RopeScalingType,
};

#[test]
fn context_params_kv_and_flash_roundtrip() {
    let params = LlamaContextParams::default()
        .with_flash_attn_type(LlamaFlashAttnType::Enabled)
        .with_attention_type(LlamaAttentionType::NonCausal)
        .with_kv_unified(false)
        .with_swa_full(true)
        .with_op_offload(true)
        .with_n_outputs_max(128)
        .with_no_perf(true);

    assert_eq!(params.flash_attn_type(), LlamaFlashAttnType::Enabled);
    assert_eq!(params.attention_type(), LlamaAttentionType::NonCausal);
    assert!(!params.kv_unified());
    assert!(params.swa_full());
    assert!(params.op_offload());
    assert_eq!(params.n_outputs_max(), 128);
    assert!(params.no_perf());
}

#[test]
fn context_params_yarn_roundtrip() {
    let params = LlamaContextParams::default()
        .with_rope_scaling_type(RopeScalingType::Yarn)
        .with_yarn_ext_factor(1.0)
        .with_yarn_attn_factor(2.0)
        .with_yarn_beta_fast(32.0)
        .with_yarn_beta_slow(1.0)
        .with_yarn_orig_ctx(8192);

    assert_eq!(params.rope_scaling_type(), RopeScalingType::Yarn);
    assert_eq!(params.yarn_ext_factor(), 1.0);
    assert_eq!(params.yarn_attn_factor(), 2.0);
    assert_eq!(params.yarn_beta_fast(), 32.0);
    assert_eq!(params.yarn_beta_slow(), 1.0);
    assert_eq!(params.yarn_orig_ctx(), 8192);
}

#[test]
fn context_params_try_clone_without_samplers() {
    let params = LlamaContextParams::default().with_n_seq_max(4);
    let cloned = params.try_clone().expect("clone without samplers");
    assert_eq!(cloned.n_seq_max(), 4);
}

#[test]
fn context_params_clone_drops_samplers() {
    use llama_cpp_4::sampling::LlamaSampler;

    let sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let params = LlamaContextParams::default().with_sampler_seq_configs([(0, sampler)]);
    assert_eq!(params.n_sampler_seq_configs(), 1);
    assert_eq!(
        params.try_clone().unwrap_err(),
        ParamsCloneError::SamplerChains
    );
    let cloned = params.clone();
    assert_eq!(cloned.n_sampler_seq_configs(), 0);
}
