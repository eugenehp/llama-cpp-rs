//! Example demonstrating how to load models that are split across multiple files.
//!
//! This example shows two modes:
//! 1. Loading a model that follows the standard split naming convention
//! 2. Loading a model with custom split file names

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use llama_cpp_4::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, LlamaModel, Special},
    token::data_array::LlamaTokenDataArray,
};
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Load a model using standard split naming convention
    Standard {
        /// Base path to the model (without split suffix)
        #[arg(short, long)]
        model_prefix: String,
        
        /// Number of splits
        #[arg(short, long)]
        num_splits: i32,
        
        /// Prompt to generate from
        #[arg(short, long, default_value = "Once upon a time")]
        prompt: String,
    },
    
    /// Load a model using custom split file paths
    Custom {
        /// Paths to all split files
        #[arg(short, long, num_args = 1..)]
        splits: Vec<PathBuf>,
        
        /// Prompt to generate from
        #[arg(short, long, default_value = "Once upon a time")]
        prompt: String,
    },
    
    /// Create split paths for a model
    CreatePaths {
        /// Base path prefix
        #[arg(short, long)]
        prefix: String,
        
        /// Number of splits to create paths for
        #[arg(short, long)]
        count: i32,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize the backend
    let _backend = LlamaBackend::init()?;
    
    match args.command {
        Commands::Standard { model_prefix, num_splits, prompt } => {
            load_standard_splits(&model_prefix, num_splits, &prompt)
        }
        Commands::Custom { splits, prompt } => {
            load_custom_splits(&splits, &prompt)
        }
        Commands::CreatePaths { prefix, count } => {
            create_split_paths(&prefix, count)
        }
    }
}

fn load_standard_splits(model_prefix: &str, num_splits: i32, prompt: &str) -> Result<()> {
    println!("Loading model with {} splits from prefix: {}", num_splits, model_prefix);
    
    // Generate the split file paths
    let mut split_paths = Vec::new();
    for i in 1..=num_splits {
        let path = LlamaModel::split_path(model_prefix, i, num_splits);
        println!("  Split {}: {}", i, path);
        split_paths.push(PathBuf::from(path));
    }
    
    // Check if all files exist
    for path in &split_paths {
        if !path.exists() {
            anyhow::bail!("Split file does not exist: {}", path.display());
        }
    }
    
    // Load the model
    println!("Loading model from splits...");
    let backend = LlamaBackend::init()?;
    let params = LlamaModelParams::default();
    let model = LlamaModel::load_from_splits(&backend, &split_paths, &params)
        .context("Failed to load model from splits")?;
    
    println!("Model loaded successfully!");
    
    // Generate some text
    generate_text(&model, prompt)?;
    
    Ok(())
}

fn load_custom_splits(splits: &[PathBuf], prompt: &str) -> Result<()> {
    println!("Loading model from {} custom split files:", splits.len());
    for (i, path) in splits.iter().enumerate() {
        println!("  Split {}: {}", i + 1, path.display());
        if !path.exists() {
            anyhow::bail!("Split file does not exist: {}", path.display());
        }
    }
    
    // Load the model
    println!("Loading model from splits...");
    let backend = LlamaBackend::init()?;
    let params = LlamaModelParams::default();
    let model = LlamaModel::load_from_splits(&backend, splits, &params)
        .context("Failed to load model from splits")?;
    
    println!("Model loaded successfully!");
    
    // Generate some text
    generate_text(&model, prompt)?;
    
    Ok(())
}

fn create_split_paths(prefix: &str, count: i32) -> Result<()> {
    println!("Creating split paths for prefix '{}' with {} splits:", prefix, count);
    println!();
    
    for i in 1..=count {
        let path = LlamaModel::split_path(prefix, i, count);
        println!("{}", path);
    }
    
    println!();
    println!("You can use these paths when splitting a large model file.");
    
    Ok(())
}

fn generate_text(model: &LlamaModel, prompt: &str) -> Result<()> {
    // Create context
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048));
    
    let mut ctx = model.new_context(&LlamaBackend::init()?, ctx_params)
        .context("Failed to create context")?;
    
    // Tokenize the prompt
    let tokens = model
        .str_to_token(prompt, llama_cpp_4::model::AddBos::Always)
        .context("Failed to tokenize prompt")?;
    
    println!("\nPrompt: {}", prompt);
    println!("Generating text...\n");
    
    // Create a batch
    let mut batch = LlamaBatch::new(512, 1);
    
    // Add tokens to batch
    for (i, token) in tokens.iter().enumerate() {
        let is_last = i == tokens.len() - 1;
        batch.add(*token, i as i32, &[0], is_last)?;
    }
    
    // Decode the batch
    ctx.decode(&mut batch)
        .context("Failed to decode batch")?;
    
    // Generate 128 tokens
    let mut n_cur = batch.n_tokens();
    let max_tokens = 128;
    
    print!("{}", prompt);
    std::io::stdout().flush()?;
    
    for _ in 0..max_tokens {
        // Sample next token
        let candidates = ctx.candidates();
        let mut candidates_p = LlamaTokenDataArray::from_iter(candidates, false);
        
        let new_token_id = candidates_p.sample_token_greedy();
        
        // Check for EOS
        if model.is_eog_token(new_token_id) {
            println!();
            break;
        }
        
        // Convert token to string and print
        let token_str = model
            .token_to_str(new_token_id, Special::Tokenize)
            .context("Failed to convert token to string")?;
        
        print!("{}", token_str);
        std::io::stdout().flush()?;
        
        // Prepare next batch
        batch.clear();
        batch.add(new_token_id, n_cur, &[0], true)?;
        
        n_cur += 1;
        
        // Decode the batch
        ctx.decode(&mut batch)
            .context("Failed to decode batch")?;
    }
    
    println!("\n\nText generation complete!");
    
    Ok(())
}