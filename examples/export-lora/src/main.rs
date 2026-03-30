//! # Export LoRA
//!
//! Load a base model with a LoRA adapter applied and verify it works.
//!
//! The C++ `llama-export-lora` tool performs a tensor-level merge using gguf APIs
//! to produce a standalone merged model file. This Rust version demonstrates loading
//! and applying a LoRA adapter, inspecting its metadata, and running inference with it.
//!
//! ## Usage
//!
//! ```console
//! cargo run -p export-lora -- -m base-model.gguf --lora adapter.gguf -p "Hello"
//! cargo run -p export-lora -- -m base-model.gguf --lora adapter.gguf --scale 0.5 --info
//! ```
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use anyhow::{Context, Result};
use clap::Parser;
use llama_cpp_4::context::params::LlamaContextParams;
use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::llama_batch::LlamaBatch;
use llama_cpp_4::model::params::LlamaModelParams;
use llama_cpp_4::model::{AddBos, LlamaModel, Special};
use llama_cpp_4::sampling::LlamaSampler;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;

#[derive(clap::Parser, Debug)]
#[command(about = "Load a model with LoRA adapter and run inference")]
struct Args {
    /// Path to the base GGUF model file
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Path to the LoRA adapter file
    #[arg(long)]
    lora: PathBuf,

    /// LoRA scale factor (default: 1.0)
    #[arg(long, default_value_t = 1.0)]
    scale: f32,

    /// Prompt for inference
    #[arg(short = 'p', long, default_value = "Hello")]
    prompt: String,

    /// Maximum tokens to generate
    #[arg(short = 'n', long, default_value_t = 64)]
    n_predict: i32,

    /// Only show adapter info, don't run inference
    #[arg(long)]
    info: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)
        .with_context(|| "failed to load base model")?;

    eprintln!("Base model: {model}");

    // Load LoRA adapter
    let mut adapter = model
        .lora_adapter_init(&args.lora)
        .with_context(|| format!("failed to load LoRA adapter: {}", args.lora.display()))?;

    // Show adapter info
    eprintln!("LoRA adapter: {}", args.lora.display());
    eprintln!("  scale: {}", args.scale);
    eprintln!("  invocation tokens: {}", adapter.n_invocation_tokens());

    let meta_count = adapter.meta_count();
    if meta_count > 0 {
        eprintln!("  metadata ({meta_count} entries):");
        if let Ok(entries) = adapter.metadata() {
            for (key, val) in &entries {
                let display = if val.len() > 80 {
                    format!("{}...", &val[..80])
                } else {
                    val.clone()
                };
                eprintln!("    {key} = {display}");
            }
        }
    }

    if args.info {
        return Ok(());
    }

    // Create context and apply adapter
    let ctx_params =
        LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(2048).unwrap()));
    let ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "failed to create context")?;

    ctx.lora_adapter_set(&mut adapter, args.scale)
        .with_context(|| "failed to apply LoRA adapter")?;
    eprintln!("LoRA adapter applied (scale={})", args.scale);

    // Run inference
    eprintln!("Generating with prompt: {:?}", args.prompt);
    eprintln!();

    let tokens_list = model.str_to_token(&args.prompt, AddBos::Always)?;
    let mut batch = LlamaBatch::new(2048, 1);
    let last_idx = tokens_list.len() as i32 - 1;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        batch.add(token, i, &[0], i == last_idx)?;
    }

    // Need mutable context for decode
    let mut ctx = ctx;
    ctx.decode(&mut batch)?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(0.8),
        LlamaSampler::dist(42),
    ]);

    let mut n_cur = batch.n_tokens();
    let n_len = n_cur + args.n_predict;
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    while n_cur < n_len {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let bytes = model.token_to_bytes(token, Special::Tokenize)?;
        let mut fragment = String::with_capacity(32);
        let _ = decoder.decode_to_string(&bytes, &mut fragment, false);
        print!("{fragment}");
        std::io::stdout().flush()?;

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch)?;
    }

    println!();
    eprintln!("\n--- Done ({} tokens generated) ---", n_cur - batch.n_tokens());

    Ok(())
}
