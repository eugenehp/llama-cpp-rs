//! # Structured Output
//!
//! Generate text that is constrained to a specific format using GBNF grammars.
//! This is how you force a model to produce valid JSON, XML, tool calls, etc.
//!
//! ## Usage
//!
//! ```console
//! # Single word answer
//! cargo run -p structured -- --format word -p "What is the largest planet?"
//!
//! # Lowercase text only
//! cargo run -p structured -- --format lowercase -p "Tell me about cats"
//!
//! # Programming identifier
//! cargo run -p structured -- --format identifier -p "Suggest a variable name for user age"
//!
//! # Comma-separated words
//! cargo run -p structured -- --format csv-words -p "List 5 colors"
//!
//! # Digits only
//! cargo run -p structured -- --format digits -p "What is 6 times 7?"
//!
//! # Custom inline grammar
//! cargo run -p structured -- --grammar 'root ::= [a-z]+' -p "Say a word"
//!
//! # Custom GBNF grammar from a file
//! cargo run -p structured -- --grammar-file my_grammar.gbnf -p "Generate something"
//!
//! # Use a local model
//! cargo run -p structured -- local path/to/model.gguf --format word -p "Hello"
//! ```
//!
//! ## Notes
//!
//! GBNF grammars constrain which tokens the model can generate at each step.
//! This is useful for structured output (JSON, XML, etc.) but requires grammars
//! compatible with your llama.cpp build. Some builds have issues with grammars
//! using quoted string literals — prefer character class patterns like `[a-z]+`.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use anyhow::{bail, Context, Result};
use clap::Parser;
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_4::context::params::LlamaContextParams;
use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::llama_batch::LlamaBatch;
use llama_cpp_4::model::params::LlamaModelParams;
use llama_cpp_4::model::LlamaModel;
use llama_cpp_4::model::{AddBos, Special};
use llama_cpp_4::sampling::LlamaSampler;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;

// ── CLI ──────────────────────────────────────────────────────

#[derive(clap::Parser, Debug)]
#[command(about = "Generate structured output using GBNF grammars")]
struct Args {
    /// Model source
    #[command(subcommand)]
    model: Option<Model>,

    /// The prompt
    #[arg(short = 'p', long)]
    prompt: Option<String>,

    /// Maximum number of tokens to generate
    #[arg(short = 'n', long, default_value_t = 256)]
    n_predict: i32,

    /// Built-in output format
    #[arg(long, value_enum)]
    format: Option<Format>,

    /// Inline GBNF grammar string (overrides --format)
    #[arg(long)]
    grammar: Option<String>,

    /// Path to a GBNF grammar file (overrides --format)
    #[arg(long)]
    grammar_file: Option<PathBuf>,

    /// Temperature (0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    temp: f32,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
enum Format {
    /// Single word (letters only)
    Word,
    /// Lowercase text (letters and spaces)
    Lowercase,
    /// Alphanumeric identifier
    Identifier,
    /// Simple comma-separated list of words
    CsvWords,
    /// Digits only (number)
    Digits,
}

#[derive(clap::Subcommand, Debug, Clone)]
enum Model {
    /// Use a local GGUF model file
    Local { path: PathBuf },
    /// Download from HuggingFace
    #[clap(name = "hf-model")]
    HuggingFace { repo: String, model: String },
}

impl Model {
    fn get_or_load(self) -> Result<PathBuf> {
        match self {
            Self::Local { path } => Ok(path),
            Self::HuggingFace { model, repo } => ApiBuilder::new()
                .with_progress(true)
                .build()
                .with_context(|| "unable to create huggingface api")?
                .model(repo)
                .get(&model)
                .with_context(|| "unable to download model"),
        }
    }
}

// ── Built-in GBNF grammars ──────────────────────────────────

fn grammar_for_format(format: Format) -> &'static str {
    match format {
        Format::Word => {
            // Single word (letters only, no spaces or punctuation)
            r"root ::= [A-Za-z]+"
        }
        Format::Lowercase => {
            // Lowercase letters and spaces only
            r"root ::= [a-z ]+"
        }
        Format::Identifier => {
            // Programming identifier: letter followed by alphanumerics/underscores
            r"root ::= [a-zA-Z_] [a-zA-Z0-9_]+"
        }
        Format::CsvWords => {
            // Comma-separated words (uses commas and spaces within char class)
            r"root ::= [a-zA-Z] [a-zA-Z, ]+"
        }
        Format::Digits => {
            // Digits only
            r"root ::= [0-9]+"
        }
    }
}

// ── Main ────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    // Resolve grammar: --grammar > --grammar-file > --format
    let grammar = if let Some(g) = &args.grammar {
        g.clone()
    } else if let Some(path) = &args.grammar_file {
        std::fs::read_to_string(path)
            .with_context(|| format!("failed to read grammar file: {}", path.display()))?
    } else if let Some(fmt) = args.format {
        grammar_for_format(fmt).to_string()
    } else {
        bail!(
            "Specify an output format with --format, --grammar, or --grammar-file.\n\
             Available formats: json, json-schema, xml, list, tool-call, yes-no"
        );
    };

    let prompt = args.prompt.unwrap_or_else(|| {
        match args.format {
            Some(Format::Word) => "What is the largest planet? Answer with one word:".to_string(),
            Some(Format::Lowercase) => "Describe the color blue in simple terms:".to_string(),
            Some(Format::Identifier) => "Suggest a variable name for a user's email address:".to_string(),
            Some(Format::CsvWords) => "List 5 colors:".to_string(),
            Some(Format::Digits) => "What is 6 times 7? Answer with just the number:".to_string(),
            None => "Hello!".to_string(),
        }
    });

    // Resolve model path
    let model_path = match args.model {
        Some(m) => m.get_or_load()?,
        None => {
            eprintln!("No model specified, downloading Qwen2.5-0.5B-Instruct (Q4_K_M)...");
            ApiBuilder::new()
                .with_progress(true)
                .build()?
                .model("Qwen/Qwen2.5-0.5B-Instruct-GGUF".to_string())
                .get("qwen2.5-0.5b-instruct-q4_k_m.gguf")?
        }
    };

    // ── Load model ──
    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .with_context(|| "unable to load model")?;

    eprintln!("Model: {model}");
    eprintln!("Grammar:\n{grammar}");
    eprintln!();

    // ── Create context ──
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(2048).unwrap()));
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create context")?;

    // ── Build the prompt ──
    // Try to use chat template if available, otherwise use raw prompt
    let full_prompt = match model.get_chat_template(2048) {
        Ok(_template) => {
            // Use a simple chatml-style prompt
            format!(
                "<|im_start|>system\nYou are a helpful assistant. Respond ONLY with the requested format, no explanation.<|im_end|>\n\
                 <|im_start|>user\n{prompt}<|im_end|>\n\
                 <|im_start|>assistant\n"
            )
        }
        Err(_) => prompt.clone(),
    };

    let tokens_list = model
        .str_to_token(&full_prompt, AddBos::Always)
        .with_context(|| "failed to tokenize")?;

    let n_prompt = tokens_list.len() as i32;
    let n_len = n_prompt + args.n_predict;

    // ── Feed prompt ──
    let mut batch = LlamaBatch::new(2048, 1);
    let last_index = tokens_list.len() as i32 - 1;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        batch.add(token, i, &[0], i == last_index)?;
    }
    ctx.decode(&mut batch).with_context(|| "decode failed")?;

    // ── Build sampler chain with grammar ──
    let mut samplers: Vec<LlamaSampler> = Vec::new();

    // Add grammar constraint
    samplers.push(LlamaSampler::grammar(&model, &grammar, "root"));

    // Add temperature / sampling
    if args.temp > 0.0 {
        samplers.push(LlamaSampler::temp(args.temp));
        samplers.push(LlamaSampler::dist(42));
    } else {
        samplers.push(LlamaSampler::greedy());
    }

    let mut sampler = LlamaSampler::chain_simple(samplers);

    // ── Generate ──
    let mut n_cur = batch.n_tokens();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();

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
        output.push_str(&fragment);

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;

        ctx.decode(&mut batch).with_context(|| "decode failed")?;
    }

    println!();
    eprintln!("\n--- Done ({} tokens generated) ---", n_cur - n_prompt);

    // Validate the output format if applicable
    if let Some(fmt) = args.format {
        validate_output(fmt, &output);
    } else {
        eprintln!("Output: {:?}", output.trim());
    }

    // Drop in correct order: sampler (holds vocab ref) before context before model
    drop(sampler);
    drop(batch);
    drop(ctx);
    drop(model);

    Ok(())
}

/// Quick validation of the generated output.
fn validate_output(format: Format, output: &str) {
    let trimmed = output.trim();
    match format {
        Format::Word => {
            if !trimmed.is_empty() && trimmed.chars().all(|c| c.is_ascii_alphabetic()) {
                eprintln!("✓ Valid word: {trimmed}");
            } else {
                eprintln!("✗ Not a valid word: {trimmed}");
            }
        }
        Format::Lowercase => {
            if !trimmed.is_empty() && trimmed.chars().all(|c| c.is_ascii_lowercase() || c == ' ') {
                eprintln!("✓ Valid lowercase text");
            } else {
                eprintln!("✗ Contains non-lowercase characters");
            }
        }
        Format::Identifier => {
            if !trimmed.is_empty()
                && trimmed
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                eprintln!("✓ Valid identifier: {trimmed}");
            } else {
                eprintln!("✗ Not a valid identifier: {trimmed}");
            }
        }
        Format::CsvWords => {
            let parts: Vec<&str> = trimmed.split(", ").collect();
            if parts.len() > 1 && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_alphabetic())) {
                eprintln!("✓ Valid CSV: {} words", parts.len());
            } else {
                eprintln!("✗ Not valid CSV words");
            }
        }
        Format::Digits => {
            if !trimmed.is_empty() && trimmed.chars().all(|c| c.is_ascii_digit()) {
                eprintln!("✓ Valid number: {trimmed}");
            } else {
                eprintln!("✗ Not a valid number: {trimmed}");
            }
        }
    }
}
