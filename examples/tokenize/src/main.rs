//! # Tokenize
//!
//! Tokenize a prompt and print each token with its ID and text representation.
//! Loads the model in vocab-only mode for fast startup.
//!
//! This is the Rust equivalent of llama.cpp's `llama-tokenize` tool.
//!
//! ## Usage
//!
//! ```console
//! cargo run -p tokenize -- -m model.gguf -p "Hello, world!"
//! cargo run -p tokenize -- -m model.gguf -p "Hello, world!" --ids
//! cargo run -p tokenize -- -m model.gguf -f input.txt --show-count
//! echo "Hello" | cargo run -p tokenize -- -m model.gguf --stdin
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::model::params::LlamaModelParams;
use llama_cpp_4::model::{AddBos, LlamaModel, Special};
use std::io::Read;
use std::path::PathBuf;
use std::pin::pin;

#[derive(clap::Parser, Debug)]
#[command(about = "Tokenize a prompt and display token IDs and text")]
struct Args {
    /// Path to the GGUF model file
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Prompt text to tokenize
    #[arg(short = 'p', long)]
    prompt: Option<String>,

    /// Read prompt from a file
    #[arg(short = 'f', long)]
    file: Option<PathBuf>,

    /// Read prompt from stdin
    #[arg(long)]
    stdin: bool,

    /// Print only token IDs in a Python-parseable format: [1, 2, 3]
    #[arg(long)]
    ids: bool,

    /// Do not add BOS token
    #[arg(long)]
    no_bos: bool,

    /// Print the total number of tokens
    #[arg(long)]
    show_count: bool,
}

#[allow(clippy::cast_possible_wrap)]
fn main() -> Result<()> {
    let args = Args::parse();

    // Read prompt from one of the sources
    let prompt = if let Some(ref p) = args.prompt {
        p.clone()
    } else if let Some(ref path) = args.file {
        std::fs::read_to_string(path)
            .with_context(|| format!("failed to read file: {}", path.display()))?
    } else if args.stdin {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .with_context(|| "failed to read stdin")?;
        buf
    } else {
        anyhow::bail!("specify one of: -p/--prompt, -f/--file, or --stdin");
    };

    // Load model in vocab-only mode for speed
    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default().with_vocab_only(true);
    let model_params = pin!(model_params);
    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)
        .with_context(|| format!("failed to load model: {}", args.model.display()))?;

    // Tokenize
    let add_bos = if args.no_bos {
        AddBos::Never
    } else if model.add_bos_token() {
        AddBos::Always
    } else {
        AddBos::Never
    };

    let tokens = model
        .str_to_token(&prompt, add_bos)
        .with_context(|| "tokenization failed")?;

    // Output
    if args.ids {
        // Python-parseable format
        print!("[");
        for (i, token) in tokens.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{}", token.0);
        }
        println!("]");
    } else {
        // Detailed format: ID -> 'text'
        for token in &tokens {
            let text = model
                .token_to_str(*token, Special::Tokenize)
                .unwrap_or_else(|_| format!("<token {}>", token.0));
            // Escape control chars for display
            let display: String = text
                .chars()
                .map(|c| {
                    if c.is_control() && c != ' ' {
                        format!("\\x{:02x}", c as u32)
                    } else {
                        c.to_string()
                    }
                })
                .collect();
            println!("{:6} -> '{}'", token.0, display);
        }
    }

    if args.show_count {
        println!("Total number of tokens: {}", tokens.len());
    }

    Ok(())
}
