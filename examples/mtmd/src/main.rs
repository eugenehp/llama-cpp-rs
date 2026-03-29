#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::items_after_statements
)]
//! Multimodal (vision / audio) example using the `mtmd` feature.
//!
//! Usage:
//! ```sh
//! cargo run --features mtmd -p mtmd -- \
//!     --model       /path/to/model.gguf      \
//!     --mmproj      /path/to/mmproj.gguf     \
//!     --image       /path/to/image.jpg       \
//!     --prompt      "Describe this image."
//! ```
//!
//! You can also pass `--audio /path/to/audio.wav` instead of (or together
//! with) `--image`.  All media items are interleaved with the prompt using the
//! default marker returned by `MtmdContext::default_marker()`.

use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::Parser;
use llama_cpp_4::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, LlamaModel, Special},
    mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText},
    token::LlamaToken,
};

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to the main language model GGUF file.
    #[arg(short, long)]
    model: PathBuf,

    /// Path to the multimodal projector (mmproj) GGUF file.
    #[arg(long)]
    mmproj: PathBuf,

    /// One or more image files to embed in the prompt.
    #[arg(long, value_name = "FILE")]
    image: Vec<PathBuf>,

    /// One or more audio files to embed in the prompt.
    #[arg(long, value_name = "FILE")]
    audio: Vec<PathBuf>,

    /// Text prompt. Use `<__media__>` (or the string printed by --show-marker)
    /// as a placeholder for each media item, in order.
    #[arg(
        short,
        long,
        default_value = "Describe what you see or hear: <__media__>"
    )]
    prompt: String,

    /// Print the default media marker and exit.
    #[arg(long)]
    show_marker: bool,

    /// Maximum number of new tokens to generate.
    #[arg(long, default_value_t = 256)]
    n_predict: usize,

    /// Context size (tokens).
    #[arg(long, default_value_t = 4096)]
    n_ctx: u32,

    /// Batch size for decoding.
    #[arg(long, default_value_t = 512)]
    n_batch: u32,

    /// Number of threads for the encoder.
    #[arg(long, default_value_t = 4)]
    n_threads: i32,

    /// Do NOT offload the mmproj model to GPU.
    #[arg(long)]
    no_gpu: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    // ── Show marker shortcut ──────────────────────────────────────────────
    if args.show_marker {
        println!("{}", MtmdContext::default_marker());
        return Ok(());
    }

    // ── Initialise backend ────────────────────────────────────────────────
    let backend = LlamaBackend::init()?;

    // ── Load main model ───────────────────────────────────────────────────
    println!("Loading model: {}", args.model.display());
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)?;

    // ── Load multimodal projector ─────────────────────────────────────────
    println!("Loading mmproj: {}", args.mmproj.display());
    let ctx_params = MtmdContextParams::default()
        .use_gpu(!args.no_gpu)
        .n_threads(args.n_threads)
        .print_timings(false);

    let mtmd_ctx = MtmdContext::init_from_file(&args.mmproj, &model, ctx_params)
        .map_err(|e| anyhow::anyhow!("Failed to load mmproj: {e}"))?;

    println!(
        "  vision={}, audio={}",
        mtmd_ctx.supports_vision(),
        mtmd_ctx.supports_audio()
    );

    // ── Create LLM context ────────────────────────────────────────────────
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(args.n_ctx))
        .with_n_batch(args.n_batch);

    let mut lctx = model.new_context(&backend, ctx_params)?;

    // ── Load bitmaps ──────────────────────────────────────────────────────
    let mut bitmaps: Vec<MtmdBitmap> = Vec::new();

    for path in &args.image {
        println!("Loading image: {}", path.display());
        let bm = MtmdBitmap::from_file(&mtmd_ctx, path)
            .map_err(|e| anyhow::anyhow!("Failed to load image {}: {e}", path.display()))?;
        bitmaps.push(bm);
    }

    for path in &args.audio {
        println!("Loading audio: {}", path.display());
        let bm = MtmdBitmap::from_file(&mtmd_ctx, path)
            .map_err(|e| anyhow::anyhow!("Failed to load audio {}: {e}", path.display()))?;
        bitmaps.push(bm);
    }

    // ── Sanity check: marker count must match bitmap count ────────────────
    let marker = MtmdContext::default_marker();
    let marker_count = args.prompt.matches(marker).count();
    if marker_count != bitmaps.len() {
        bail!(
            "Prompt contains {} media marker(s) ({:?}) but {} media file(s) were provided.",
            marker_count,
            marker,
            bitmaps.len()
        );
    }

    // ── Tokenize ──────────────────────────────────────────────────────────
    println!("\nPrompt: {}", args.prompt);
    let input_text = MtmdInputText::new(&args.prompt, true, true);
    let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();

    let mut chunks = MtmdInputChunks::new();
    mtmd_ctx
        .tokenize(&input_text, &bitmap_refs, &mut chunks)
        .map_err(|e| anyhow::anyhow!("Tokenize failed: {e}"))?;

    println!(
        "Tokenized into {} chunk(s), {} total tokens.",
        chunks.len(),
        chunks.n_tokens()
    );

    // ── Evaluate all chunks ───────────────────────────────────────────────
    let mut n_past: i32 = 0;
    mtmd_ctx
        .eval_chunks(
            lctx.as_ptr(),
            &chunks,
            0,
            0,
            args.n_batch as i32,
            true,
            &mut n_past,
        )
        .map_err(|e| anyhow::anyhow!("Eval failed: {e}"))?;

    // ── Greedy decode loop ────────────────────────────────────────────────
    println!("\n=== Response ===");
    let eos = model.token_eos();
    let mut generated = 0usize;

    loop {
        if generated >= args.n_predict {
            break;
        }

        // Pick the token with the highest logit.
        let logits = lctx.get_logits();
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| LlamaToken::new(i as i32))
            .unwrap();

        if model.is_eog_token(next_token) || next_token == eos {
            break;
        }

        // Detokenize and print.
        let piece = model
            .token_to_str(next_token, Special::Tokenize)
            .unwrap_or_default();
        print!("{piece}");
        use std::io::Write;
        std::io::stdout().flush().ok();

        // Feed the token back.
        let mut batch = llama_cpp_4::llama_batch::LlamaBatch::new(1, 0);
        batch.add(next_token, n_past, &[0], true)?;
        lctx.decode(&mut batch)?;

        n_past += 1;
        generated += 1;
    }

    println!("\n\n=== Done ({generated} tokens generated) ===");
    Ok(())
}
