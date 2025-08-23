//! Example of using multimodal support with llama-cpp-rs
//! 
//! This example demonstrates how to process images alongside text using
//! vision models like LLaVA, Qwen2-VL, or Gemma3.

use anyhow::{Context, Result};
use clap::Parser;
use llama_cpp_4::context::{LlamaContext, LlamaContextParams};
use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::llama_batch::LlamaBatch;
use llama_cpp_4::model::{AddBos, LlamaModel, LlamaModelParams, Special};
use std::num::NonZeroU32;
use std::sync::Arc;

#[cfg(feature = "multimodal")]
use llama_cpp_4::multimodal::{
    Bitmap, InputChunks, InputText, MtmdContext, MtmdContextParams,
};

/// Command line arguments for the multimodal example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the main model file (GGUF format)
    #[arg(short, long)]
    model: String,
    
    /// Path to the multimodal projector file (mmproj GGUF)
    #[arg(short = 'p', long)]
    mmproj: String,
    
    /// Path to the image file to process
    #[arg(short, long)]
    image: String,
    
    /// Prompt text (use <__media__> as placeholder for image)
    #[arg(short = 't', long, default_value = "Describe this image in detail: <__media__>")]
    prompt: String,
    
    /// Number of threads to use
    #[arg(short, long, default_value_t = 4)]
    threads: i32,
    
    /// Number of tokens to predict
    #[arg(short, long, default_value_t = 256)]
    n_predict: i32,
    
    /// Context size
    #[arg(short = 'c', long, default_value_t = 2048)]
    ctx_size: u32,
    
    /// Number of GPU layers
    #[arg(short = 'g', long, default_value_t = 0)]
    n_gpu_layers: u32,
}

#[cfg(not(feature = "multimodal"))]
fn main() {
    eprintln!("This example requires the 'multimodal' feature to be enabled.");
    eprintln!("Please run with: cargo run --features multimodal");
    std::process::exit(1);
}

#[cfg(feature = "multimodal")]
fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize the backend
    let _backend = LlamaBackend::init()?;
    
    // Load the main model
    println!("Loading model from: {}", args.model);
    let model_params = LlamaModelParams::default()
        .with_n_gpu_layers(args.n_gpu_layers);
    
    let model = Arc::new(
        LlamaModel::load_from_file(&args.model, model_params)
            .context("Failed to load model")?
    );
    
    // Create context
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(args.ctx_size))
        .with_n_threads(args.threads)
        .with_n_threads_batch(args.threads);
    
    let mut context = model.new_context(ctx_params)
        .context("Failed to create context")?;
    
    // Initialize multimodal context
    println!("Loading multimodal projector from: {}", args.mmproj);
    let mtmd_params = MtmdContextParams {
        use_gpu: args.n_gpu_layers > 0,
        print_timings: true,
        n_threads: args.threads,
        verbosity: 0,
        media_marker: None,
    };
    
    let mtmd_context = MtmdContext::new_from_file(&args.mmproj, model.clone(), mtmd_params)
        .context("Failed to load multimodal projector")?;
    
    // Check capabilities
    if mtmd_context.supports_vision() {
        println!("✓ Vision support enabled");
    } else {
        return Err(anyhow::anyhow!("Model does not support vision input"));
    }
    
    if mtmd_context.supports_audio() {
        println!("✓ Audio support enabled");
        if let Some(bitrate) = mtmd_context.get_audio_bitrate() {
            println!("  Audio bitrate: {} Hz", bitrate);
        }
    }
    
    // Load and prepare image
    println!("Loading image from: {}", args.image);
    let img = image::open(&args.image)
        .context("Failed to load image")?;
    
    let rgb_img = img.to_rgb8();
    let (width, height) = (rgb_img.width(), rgb_img.height());
    let pixels = rgb_img.into_raw();
    
    println!("Image dimensions: {}x{}", width, height);
    
    // Create bitmap from image
    let bitmap = Bitmap::new_image(width, height, &pixels)
        .context("Failed to create bitmap")?;
    
    // Prepare input with media placeholder
    let input_text = InputText::new(args.prompt.clone());
    let bitmaps = vec![&bitmap];
    
    // Tokenize mixed input
    let mut chunks = InputChunks::new()
        .context("Failed to create input chunks")?;
    
    chunks.tokenize(&mtmd_context, input_text, &bitmaps)
        .context("Failed to tokenize input")?;
    
    println!("Tokenized into {} chunks:", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        println!("  Chunk {}: {:?}, {} tokens", 
                 i, chunk.chunk_type(), chunk.n_tokens());
    }
    
    // Create batch for processing
    let mut batch = LlamaBatch::new(512, 1);
    
    // Process each chunk
    let mut total_tokens = 0;
    for chunk in chunks.iter() {
        match chunk.chunk_type() {
            llama_cpp_4::multimodal::ChunkType::Text => {
                if let Some(tokens) = chunk.get_text_tokens() {
                    for token in tokens {
                        batch.add(token, total_tokens, &[0], false)?;
                        total_tokens += 1;
                    }
                }
            }
            llama_cpp_4::multimodal::ChunkType::Image | 
            llama_cpp_4::multimodal::ChunkType::Audio => {
                // TODO: Proper handling of image/audio embeddings
                // The actual implementation requires:
                // 1. Getting the image tokens from mtmd_input_chunk_get_tokens_image
                // 2. Processing them through the vision encoder
                // 3. Adding the embeddings to the batch
                // For now, this is a placeholder that just counts tokens
                println!("WARNING: Media chunk processing not fully implemented");
                println!("Processing media chunk with {} tokens", chunk.n_tokens());
                total_tokens += chunk.n_tokens() as i32;
            }
        }
    }
    
    // Set non-causal mask if needed
    if mtmd_context.decode_use_non_causal() {
        println!("Using non-causal attention mask");
    }
    
    if mtmd_context.decode_use_mrope() {
        println!("Using M-RoPE");
    }
    
    // Decode the batch
    batch.set_last_tokens_as_logits();
    context.decode(&mut batch)
        .context("Failed to decode batch")?;
    
    // Generate response
    println!("\nGenerating response...\n");
    println!("Input: {}\n", args.prompt);
    println!("Response:");
    
    let mut decoded_tokens = 0;
    
    while decoded_tokens < args.n_predict {
        // Sample next token
        let candidates = context.candidates();
        let token = context.sample_token_greedy(candidates);
        
        // Check for end of generation
        if token == model.token_eos() {
            break;
        }
        
        // Decode and print token
        let text = model.token_to_piece(token, Special::Tokenize)?;
        print!("{}", text);
        use std::io::{self, Write};
        io::stdout().flush()?;
        
        // Prepare for next iteration
        batch.clear();
        batch.add(token, total_tokens, &[0], true)?;
        context.decode(&mut batch)?;
        
        total_tokens += 1;
        decoded_tokens += 1;
    }
    
    println!("\n\nGeneration complete!");
    println!("Total tokens: {}", total_tokens);
    
    Ok(())
}