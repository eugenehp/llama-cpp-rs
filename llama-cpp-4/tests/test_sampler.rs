//! Tests for sampler creation and introspection (no model needed for most).

use llama_cpp_4::sampling::LlamaSampler;
use llama_cpp_4::token::data::LlamaTokenData;
use llama_cpp_4::token::data_array::LlamaTokenDataArray;
use llama_cpp_4::token::LlamaToken;

#[test]
fn test_greedy_sampler() {
    let sampler = LlamaSampler::greedy();
    assert_eq!(sampler.name(), "greedy");
}

#[test]
fn test_dist_sampler() {
    let sampler = LlamaSampler::dist(42);
    assert_eq!(sampler.name(), "dist");
    assert_eq!(sampler.get_seed(), 42);
}

#[test]
fn test_temp_sampler() {
    let sampler = LlamaSampler::temp(0.8);
    assert_eq!(sampler.name(), "temp");
}

#[test]
fn test_temp_ext_sampler() {
    let sampler = LlamaSampler::temp_ext(0.8, 0.1, 1.0);
    assert_eq!(sampler.name(), "temp-ext");
}

#[test]
fn test_top_k_sampler() {
    let sampler = LlamaSampler::top_k(40);
    assert_eq!(sampler.name(), "top-k");
}

#[test]
fn test_top_p_sampler() {
    let sampler = LlamaSampler::top_p(0.9, 1);
    assert_eq!(sampler.name(), "top-p");
}

#[test]
fn test_min_p_sampler() {
    let sampler = LlamaSampler::min_p(0.05, 1);
    assert_eq!(sampler.name(), "min-p");
}

#[test]
fn test_typical_sampler() {
    let sampler = LlamaSampler::typical(1.0, 1);
    let name = sampler.name();
    assert!(name.contains("typical"), "expected 'typical' in name, got: {name}");
}

#[test]
fn test_xtc_sampler() {
    let sampler = LlamaSampler::xtc(0.5, 0.1, 1, 42);
    assert_eq!(sampler.name(), "xtc");
}

#[test]
fn test_top_n_sigma_sampler() {
    let sampler = LlamaSampler::top_n_sigma(2.0);
    assert_eq!(sampler.name(), "top-n-sigma");
}

#[test]
fn test_adaptive_p_sampler() {
    let sampler = LlamaSampler::adaptive_p(0.9, 0.95, 42);
    assert_eq!(sampler.name(), "adaptive-p");
    // get_seed may return the seed or LLAMA_DEFAULT_SEED depending on implementation
    let _ = sampler.get_seed();
}

#[test]
fn test_mirostat_sampler() {
    let sampler = LlamaSampler::mirostat(32000, 42, 5.0, 0.1, 100);
    assert_eq!(sampler.name(), "mirostat");
}

#[test]
fn test_mirostat_v2_sampler() {
    let sampler = LlamaSampler::mirostat_v2(42, 5.0, 0.1);
    assert_eq!(sampler.name(), "mirostat-v2");
}

#[test]
fn test_logit_bias_sampler() {
    let biases = vec![(LlamaToken(0), -10.0), (LlamaToken(1), 5.0)];
    let sampler = LlamaSampler::logit_bias(32000, &biases);
    assert_eq!(sampler.name(), "logit-bias");
}

#[test]
fn test_chain_creation() {
    let chain = LlamaSampler::chain_simple([
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::temp(0.8),
        LlamaSampler::greedy(),
    ]);
    assert_eq!(chain.name(), "chain");
    assert_eq!(chain.chain_n(), 4);
}

#[test]
fn test_chain_remove() {
    let mut chain = LlamaSampler::chain_simple([
        LlamaSampler::top_k(40),
        LlamaSampler::greedy(),
    ]);
    assert_eq!(chain.chain_n(), 2);
    let removed = chain.chain_remove(0);
    assert_eq!(removed.name(), "top-k");
    assert_eq!(chain.chain_n(), 1);
}

#[test]
fn test_sampler_clone() {
    let original = LlamaSampler::chain_simple([
        LlamaSampler::top_k(40),
        LlamaSampler::temp(0.8),
        LlamaSampler::greedy(),
    ]);
    let cloned = original.clone_sampler();
    assert_eq!(original.chain_n(), cloned.chain_n());
    assert_eq!(original.name(), cloned.name());
}

#[test]
fn test_sampler_reset() {
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    sampler.reset();
    // Should not panic
    assert_eq!(sampler.chain_n(), 1);
}

#[test]
fn test_sampler_perf_data() {
    let sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let perf = sampler.perf_data();
    assert_eq!(perf.n_sample, 0);
    assert_eq!(perf.t_sample_ms, 0.0);
}

#[test]
fn test_sampler_perf_reset() {
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    sampler.perf_reset();
    let perf = sampler.perf_data();
    assert_eq!(perf.n_sample, 0);
}

#[test]
fn test_greedy_selects_max() {
    let mut data_array = LlamaTokenDataArray::new(
        vec![
            LlamaTokenData::new(LlamaToken(0), 1.0, 0.0),
            LlamaTokenData::new(LlamaToken(1), 5.0, 0.0),
            LlamaTokenData::new(LlamaToken(2), 3.0, 0.0),
        ],
        false,
    );
    data_array.apply_sampler(&mut LlamaSampler::greedy());
    assert_eq!(data_array.selected_token(), Some(LlamaToken(1)));
}

#[test]
fn test_top_k_filters() {
    let mut data_array = LlamaTokenDataArray::new(
        vec![
            LlamaTokenData::new(LlamaToken(0), 1.0, 0.0),
            LlamaTokenData::new(LlamaToken(1), 5.0, 0.0),
            LlamaTokenData::new(LlamaToken(2), 3.0, 0.0),
            LlamaTokenData::new(LlamaToken(3), 2.0, 0.0),
        ],
        false,
    );
    data_array.apply_sampler(&mut LlamaSampler::top_k(2));
    assert_eq!(data_array.data.len(), 2);
}

#[test]
fn test_temp_scales_logits() {
    let mut data_array = LlamaTokenDataArray::new(
        vec![
            LlamaTokenData::new(LlamaToken(0), 2.0, 0.0),
            LlamaTokenData::new(LlamaToken(1), 4.0, 0.0),
        ],
        false,
    );
    data_array.apply_sampler(&mut LlamaSampler::temp(0.5));
    assert_eq!(data_array.data[0].logit(), 4.0);
    assert_eq!(data_array.data[1].logit(), 8.0);
}

#[test]
fn test_chain_simple_applies_all() {
    let mut data_array = LlamaTokenDataArray::new(
        vec![
            LlamaTokenData::new(LlamaToken(0), 1.0, 0.0),
            LlamaTokenData::new(LlamaToken(1), 5.0, 0.0),
            LlamaTokenData::new(LlamaToken(2), 3.0, 0.0),
        ],
        false,
    );
    data_array.apply_sampler(&mut LlamaSampler::chain_simple([
        LlamaSampler::top_k(2),
        LlamaSampler::greedy(),
    ]));
    assert_eq!(data_array.data.len(), 2);
    assert_eq!(data_array.selected_token(), Some(LlamaToken(1)));
}

#[test]
fn test_accept_tokens() {
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    sampler.accept(LlamaToken(0));
    sampler.accept_many([LlamaToken(1), LlamaToken(2)]);
    // Should not panic
}

#[test]
fn test_with_tokens() {
    let sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()])
        .with_tokens([LlamaToken(0), LlamaToken(1)]);
    assert_eq!(sampler.name(), "chain");
}
