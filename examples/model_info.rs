//! # Model Info
//!
//! Comprehensive test of all model introspection, vocab, context, sampler, and system APIs.
//!
//! ```console
//! cargo run --example model_info -- path/to/model.gguf
//! ```
use llama_cpp_4::context::params::LlamaContextParams;
use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::model::params::LlamaModelParams;
use llama_cpp_4::model::{AddBos, LlamaModel};
use llama_cpp_4::sampling::LlamaSampler;
use llama_cpp_4::token::LlamaToken;

#[allow(clippy::too_many_lines)]
fn main() {
    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: model_info <path-to-gguf>");

    // === System Info ===
    println!("=== System Info ===");
    println!("{}", llama_cpp_4::print_system_info());
    println!("GPU offload       : {}", llama_cpp_4::supports_gpu_offload());
    println!("RPC support       : {}", llama_cpp_4::supports_rpc());
    println!("max parallel seq  : {}", llama_cpp_4::max_parallel_sequences());
    println!("max buft overrides: {}", llama_cpp_4::max_tensor_buft_overrides());
    println!("flash_attn name(0): {}", llama_cpp_4::flash_attn_type_name(0));
    println!("meta key str(0)   : {}", llama_cpp_4::model_meta_key_str(0));
    println!("quantize defaults : {:?}", llama_cpp_4::model_quantize_default_params().nthread);
    println!();

    let backend = LlamaBackend::init().unwrap();
    let params = LlamaModelParams::default();
    let model =
        LlamaModel::load_from_file(&backend, &model_path, &params).expect("unable to load model");

    // === Model Summary ===
    println!("=== Model Summary ===");
    println!("{model}");
    println!();

    // === Model Properties ===
    println!("=== Model Properties ===");
    println!("add_bos_token     : {}", model.add_bos_token());
    println!("add_eos_token     : {}", model.add_eos_token());
    println!("has_encoder       : {}", model.has_encoder());
    println!("has_decoder       : {}", model.has_decoder());
    println!("is_recurrent      : {}", model.is_recurrent());
    println!("is_hybrid         : {}", model.is_hybrid());
    println!("is_diffusion      : {}", model.is_diffusion());
    println!();

    // === Vocab (via LlamaVocab) ===
    println!("=== Vocab ===");
    let vocab = model.get_vocab();
    println!("n_tokens          : {}", vocab.n_tokens());
    println!("vocab_type        : {}", vocab.vocab_type());
    println!("bos               : {}", vocab.bos());
    println!("eos               : {}", vocab.eos());
    println!("eot               : {}", vocab.eot());
    println!("cls               : {}", vocab.cls());
    println!("sep               : {}", vocab.sep());
    println!("nl                : {}", vocab.nl());
    println!("pad               : {}", vocab.pad());
    println!("mask              : {}", vocab.mask());
    println!("fim_pre           : {}", vocab.fim_pre());
    println!("fim_suf           : {}", vocab.fim_suf());
    println!("fim_mid           : {}", vocab.fim_mid());
    println!("get_add_bos       : {}", vocab.get_add_bos());
    println!("get_add_eos       : {}", vocab.get_add_eos());
    println!("get_add_sep       : {}", vocab.get_add_sep());

    // Token info via vocab
    let bos = vocab.bos();
    println!("\nBOS via vocab:");
    println!("  is_control      : {}", vocab.is_control(bos));
    println!("  is_eog          : {}", vocab.is_eog(bos));
    println!("  score           : {}", vocab.get_score(bos));
    println!("  attr            : {}", vocab.get_attr(bos));
    match vocab.get_text(bos) {
        Ok(text) => println!("  text            : {:?}", text),
        Err(e) => println!("  text            : (error: {e})"),
    }
    println!();

    // === Token methods on model ===
    println!("=== Special Tokens (model) ===");
    println!("token_cls         : {}", model.token_cls());
    println!("token_eot         : {}", model.token_eot());
    println!("token_pad         : {}", model.token_pad());
    println!("token_sep         : {}", model.token_sep());
    println!("token_fim_pre     : {}", model.token_fim_pre());
    println!("token_fim_suf     : {}", model.token_fim_suf());
    println!("token_fim_mid     : {}", model.token_fim_mid());
    println!("token_fim_pad     : {}", model.token_fim_pad());
    println!("token_fim_rep     : {}", model.token_fim_rep());
    println!("token_fim_sep     : {}", model.token_fim_sep());
    println!();

    // === Detokenize ===
    println!("=== Detokenize ===");
    let test_str = "Hello, world!";
    let tokens = model.str_to_token(test_str, AddBos::Never).unwrap();
    println!("tokens: {:?}", tokens);
    match model.detokenize(&tokens, true, false) {
        Ok(s) => println!("round-trip        : {:?}", s),
        Err(e) => println!("error             : {e}"),
    }
    println!();

    // === Built-in Templates ===
    let templates = LlamaModel::chat_builtin_templates();
    println!("=== Built-in Templates ({}) ===", templates.len());
    println!("{:?}", &templates[..templates.len().min(5)]);
    println!();

    // === Context ===
    println!("=== Context ===");
    let ctx_params = LlamaContextParams::default();
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .expect("unable to create context");
    println!("n_ctx             : {}", ctx.n_ctx());
    println!("n_ctx_seq         : {}", ctx.n_ctx_seq());
    println!("n_seq_max         : {}", ctx.n_seq_max());
    println!("n_threads         : {}", ctx.n_threads());
    println!("n_threads_batch   : {}", ctx.n_threads_batch());
    println!("memory_can_shift  : {}", ctx.memory_can_shift());
    println!("memory_seq_pos_min: {}", ctx.memory_seq_pos_min(0));
    println!("state_get_size    : {} bytes", ctx.state_get_size());

    // Test set_n_threads
    ctx.set_n_threads(2, 2);
    println!("set_n_threads(2,2): ok (threads={})", ctx.n_threads());

    // Test set_causal_attn, set_embeddings, set_warmup
    ctx.set_causal_attn(true);
    println!("set_causal_attn   : ok");
    ctx.set_embeddings(false);
    println!("set_embeddings    : ok");
    ctx.set_warmup(false);
    println!("set_warmup        : ok");

    // Test synchronize
    ctx.synchronize();
    println!("synchronize       : ok");

    // Test perf
    ctx.perf_context_reset();
    println!("perf_context_reset: ok");

    // Memory breakdown
    ctx.memory_breakdown_print();

    // Detach threadpool (no-op if none attached)
    ctx.detach_threadpool();
    println!("detach_threadpool : ok");

    // Get model ptr
    let model_ptr = ctx.get_model_ptr();
    println!("get_model_ptr     : {:?}", model_ptr);

    // State seq operations
    let seq_size = ctx.state_seq_get_size(0);
    println!("state_seq_size(0) : {} bytes", seq_size);
    let seq_size_ext = ctx.state_seq_get_size_ext(0, 0);
    println!("state_seq_size_ext: {} bytes", seq_size_ext);
    println!();

    // === Sampler ===
    println!("=== Sampler ===");
    let greedy = LlamaSampler::greedy();
    println!("greedy name       : {}", greedy.name());
    println!("greedy seed       : {}", greedy.get_seed());

    let chain = LlamaSampler::chain_simple([
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::temp(0.8),
        LlamaSampler::dist(42),
    ]);
    println!("chain_n           : {}", chain.chain_n());

    // Clone
    let cloned = chain.clone_sampler();
    println!("clone chain_n     : {}", cloned.chain_n());

    // Perf
    let perf = chain.perf_data();
    println!("perf t_sample_ms  : {}", perf.t_sample_ms);
    println!("perf n_sample     : {}", perf.n_sample);

    // New samplers
    let _top_n = LlamaSampler::top_n_sigma(2.0);
    println!("top_n_sigma       : ok");
    let _adaptive = LlamaSampler::adaptive_p(0.9, 0.95, 42);
    println!("adaptive_p        : ok");
    let _logit_bias = LlamaSampler::logit_bias(
        model.n_vocab(),
        &[(LlamaToken(0), -10.0), (LlamaToken(1), 5.0)],
    );
    println!("logit_bias        : ok");
    let _infill = LlamaSampler::infill(&model);
    println!("infill            : ok");
    println!();

    // === Metadata ===
    println!("=== Metadata ({} entries) ===", model.meta_count());
    for key in ["general.name", "general.architecture"] {
        match model.meta_val_str(key, 256) {
            Ok(val) => println!("  {key} = {val}"),
            Err(e) => println!("  {key} = (error: {e})"),
        }
    }
}
