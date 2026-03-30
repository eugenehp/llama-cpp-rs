//! # Text-to-Speech (token generation)
//!
//! Generate audio tokens from text using a TTS model (e.g. OuteTTS, Kokoro).
//!
//! The C++ `llama-tts` tool includes a full audio codec decoder to produce WAV files.
//! This simplified Rust version generates the audio token sequence that a TTS model
//! produces, which can be decoded by an external audio codec.
//!
//! ## Usage
//!
//! ```console
//! cargo run -p tts -- -m tts-model.gguf -p "Hello, world!" -o tokens.txt
//! ```
//!
//! ## Notes
//!
//! TTS models produce special audio tokens (typically in a higher token range).
//! These tokens represent audio codebook entries that need to be decoded by a
//! compatible audio codec (e.g. Encodec, DAC) to produce actual audio.
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
#[command(about = "Generate audio tokens from text using a TTS model")]
struct Args {
    /// Path to the TTS GGUF model file
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Text to synthesize
    #[arg(short = 'p', long)]
    prompt: String,

    /// Output file for audio tokens
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

    /// Maximum tokens to generate
    #[arg(short = 'n', long, default_value_t = 2048)]
    n_predict: i32,

    /// Temperature
    #[arg(long, default_value_t = 0.7)]
    temp: f32,

    /// Top-K sampling
    #[arg(long, default_value_t = 50)]
    top_k: i32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)
        .with_context(|| "failed to load model")?;

    eprintln!("Model: {model}");
    eprintln!("Vocab: {} tokens", model.n_vocab());
    eprintln!("Input: {:?}", args.prompt);

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(4096).unwrap()));
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "failed to create context")?;

    // Tokenize input text
    let tokens = model.str_to_token(&args.prompt, AddBos::Always)?;
    eprintln!("Input tokens: {}", tokens.len());

    // Feed prompt
    let mut batch = LlamaBatch::new(4096, 1);
    let last_idx = tokens.len() as i32 - 1;
    for (i, token) in (0_i32..).zip(tokens.iter().copied()) {
        batch.add(token, i, &[0], i == last_idx)?;
    }
    ctx.decode(&mut batch)?;

    // Build sampler
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::top_k(args.top_k),
        LlamaSampler::temp(args.temp),
        LlamaSampler::dist(42),
    ]);

    // Generate
    let mut n_cur = batch.n_tokens();
    let n_len = n_cur + args.n_predict;
    let mut audio_tokens: Vec<i32> = Vec::new();
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    eprintln!("Generating...");

    while n_cur < n_len {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            eprintln!("[EOS]");
            break;
        }

        audio_tokens.push(token.0);

        // Show progress
        let bytes = model
            .token_to_bytes(token, Special::Tokenize)
            .unwrap_or_default();
        let mut fragment = String::with_capacity(32);
        let _ = decoder.decode_to_string(&bytes, &mut fragment, false);

        if !fragment.is_empty() && fragment.chars().all(|c| !c.is_control()) {
            eprint!("{fragment}");
        } else {
            eprint!("[{}]", token.0);
        }
        std::io::stderr().flush()?;

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch)?;
    }

    eprintln!();
    eprintln!("Generated {} tokens", audio_tokens.len());

    // Save tokens
    if let Some(output) = &args.output {
        let mut f = std::fs::File::create(output)?;
        for &t in &audio_tokens {
            writeln!(f, "{t}")?;
        }
        eprintln!("Tokens saved to {}", output.display());
    } else {
        // Print to stdout
        println!();
        println!("Audio tokens:");
        for chunk in audio_tokens.chunks(20) {
            let s: Vec<String> = chunk.iter().map(|t| t.to_string()).collect();
            println!("  {}", s.join(", "));
        }
    }

    Ok(())
}
