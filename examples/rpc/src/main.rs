//! Example of using RPC backend for distributed inference
//!
//! This example demonstrates how to use the RPC backend to distribute
//! inference across multiple machines.
//!
//! To run this example:
//! 1. Start an RPC server on a remote machine (or locally for testing)
//! 2. Run this client with the server's endpoint

use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_4::{
    context::params::LlamaContextParams,
    ggml::GgmlBackend,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    rpc::{RpcBackend, RpcServer},
    sampling::LlamaSampler,
};
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run as server mode
    #[arg(long)]
    server: bool,

    /// Endpoint for RPC (e.g., "127.0.0.1:50052")
    #[arg(short, long, default_value = "127.0.0.1:50052")]
    endpoint: String,

    /// Model to use (HuggingFace repo ID or local path)
    #[arg(short, long)]
    model: Option<String>,

    /// Prompt to generate text from (client mode only)
    #[arg(short, long, default_value = "Once upon a time")]
    prompt: String,

    /// Maximum number of tokens to generate
    #[arg(long, default_value = "128")]
    max_tokens: usize,

    /// Use CPU backend for server mode
    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize the llama backend
    let backend = LlamaBackend::init()?;

    if args.server {
        run_server(&args)
    } else {
        run_client(&args, &backend)
    }
}

fn run_server(args: &Args) -> Result<()> {
    println!("Starting RPC server on {}", args.endpoint);

    // Initialize a CPU backend
    let ggml_backend = GgmlBackend::cpu();
    let backend_ptr = std::ptr::NonNull::new(ggml_backend.as_ptr())
        .context("Failed to initialize CPU backend")?;

    // Start the RPC server
    let _server = RpcServer::start(
        backend_ptr,
        &args.endpoint,
        0, // Auto-detect free memory
        0, // Auto-detect total memory
    )?;

    // Keep the backend alive for the lifetime of the server
    std::mem::forget(ggml_backend);

    println!("RPC server listening on {}", args.endpoint);
    println!("Press Ctrl+C to stop the server");

    // Keep the server running
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn run_client(args: &Args, backend: &LlamaBackend) -> Result<()> {
    println!("Connecting to RPC server at {}", args.endpoint);

    // Initialize RPC backend
    let rpc_backend = RpcBackend::init(&args.endpoint)?;
    println!("Connected to RPC backend: {:?}", rpc_backend);

    // Query device memory
    match rpc_backend.get_device_memory() {
        Ok((free, total)) => {
            println!(
                "Remote device memory: {:.2} GB free / {:.2} GB total",
                free as f64 / 1_073_741_824.0,
                total as f64 / 1_073_741_824.0
            );
        }
        Err(e) => {
            println!("Could not query device memory: {}", e);
        }
    }

    // Load model
    let model_path = if let Some(model) = &args.model {
        if model.contains('/') && !model.contains('.') {
            // Looks like a HuggingFace repo ID
            download_model(model)?
        } else {
            // Local path
            PathBuf::from(model)
        }
    } else {
        // Default model
        download_model("TheBloke/Llama-2-7B-Chat-GGUF")?
    };

    println!("Loading model from {:?}", model_path);

    // Load the model
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(backend, &model_path, &model_params)
        .context("Failed to load model")?;

    // Create context
    let ctx_params =
        LlamaContextParams::default().with_n_ctx(NonZeroU32::new(2048));

    let mut ctx = model
        .new_context(backend, ctx_params)
        .context("Failed to create context")?;

    // Tokenize prompt
    let tokens_list = model
        .str_to_token(&args.prompt, AddBos::Always)
        .context("Failed to tokenize prompt")?;

    println!("Prompt: {}", args.prompt);
    println!("Generating {} tokens...\n", args.max_tokens);

    // Create batch
    let mut batch = LlamaBatch::new(512, 1);

    // Add tokens to batch
    let last_index = tokens_list.len() - 1;
    for (i, token) in tokens_list.iter().enumerate() {
        let is_last = i == last_index;
        batch.add(*token, i as i32, &[0], is_last)?;
    }

    // Decode the batch
    ctx.decode(&mut batch)
        .context("Failed to decode batch")?;

    // Set up sampler
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

    // Generate tokens
    let mut n_cur = batch.n_tokens();
    let mut n_decode = 0;

    print!("{}", args.prompt);
    std::io::stdout().flush()?;

    while n_decode < args.max_tokens {
        // Sample next token
        let new_token_id = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(new_token_id);

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
        n_decode += 1;

        // Decode the batch
        ctx.decode(&mut batch)
            .context("Failed to decode batch")?;
    }

    println!("\n\nGenerated {} tokens", n_decode);

    Ok(())
}

fn download_model(repo: &str) -> Result<PathBuf> {
    println!("Downloading model from HuggingFace: {}", repo);

    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .context("unable to create huggingface api")?;
    let repo = api.model(repo.to_string());

    // Try to find a GGUF file
    let files = repo.info()?;

    // Look for Q4_K_M quantization first, then any GGUF
    let gguf_file = files
        .siblings
        .iter()
        .find(|f| f.filename.contains("Q4_K_M") && f.filename.ends_with(".gguf"))
        .or_else(|| {
            files
                .siblings
                .iter()
                .find(|f| f.filename.ends_with(".gguf"))
        })
        .context("No GGUF file found in repository")?;

    println!("Downloading {}", gguf_file.filename);
    let path = repo.get(&gguf_file.filename)?;

    Ok(path)
}
