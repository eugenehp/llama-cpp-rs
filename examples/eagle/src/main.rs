//! EAGLE-3 speculative-decoding demo.
//!
//! Pairs a **target** model with a separate **EAGLE-3 draft** model and drives
//! the speculative-decode loop from Rust via
//! [`llama_cpp_4::eagle::Eagle3Session`], which wraps upstream's
//! `common_speculative_impl_draft_eagle3`
//! ([PR #18039](https://github.com/ggml-org/llama.cpp/pull/18039)).
//!
//! Unlike MTP (one model, special draft context type), EAGLE-3 needs **two
//! GGUFs**: the full target model and a small EAGLE-3 draft model trained for
//! it (one exposing 3 target-extract layers). There are no pre-converted
//! EAGLE-3 GGUFs published, so this example takes two **local** GGUF paths.
//! Run `scripts/setup-eagle3.sh` once to download + convert a pairing.
//!
//! # Modes
//!
//! - **Smoke test** (default): loads both models, builds both contexts, and
//!   creates the [`Eagle3Session`]. Session init validates that the draft is a
//!   real EAGLE-3 model (3 extract layers) compatible with the target — so a
//!   successful run already exercises the whole setup path.
//! - **Generation** (`--predict N`): runs prefill + draft/verify/accept and
//!   reports acceptance rate and tok/s.
//!
//! # Example
//!
//! ```sh
//! # after running scripts/fetch-eagle3.sh ./models/eagle3
//! cargo run --release -p eagle --features metal -- \
//!     --predict 64 --n-draft-max 8 --p-min 0.5 \
//!     ./models/eagle3/qwen3-8b.gguf ./models/eagle3/qwen3-8b-eagle3.gguf
//! ```
//!
//! The defaults for `--n-draft-max 8` and `--p-min 0.5` mirror the upstream
//! PR's `llama-server --spec-type draft-eagle3` invocation.
//!
//! # Verified
//!
//! This example was run end-to-end on an Apple M4 Pro (Metal) with
//! `Qwen/Qwen3-8B` (q8_0 target) and `RedHatAI/Qwen3-8B-speculator.eagle3`
//! (the draft), generating coherent text at ~53% draft acceptance.
//!
//! The draft/verify/accept loop manages the draft context's KV cache the same
//! way the `mtp` example does: `draft()` autoregressively advances the draft
//! KV, so it is rolled back to `n_past` before each `process()` (which
//! re-decodes those positions with the target's harvested features), and the
//! rejected suffix is trimmed afterwards. For a recurrent/hybrid target, pass
//! `--n-rs-seq >= n-draft-max` so those rollbacks succeed.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use llama_cpp_4::prelude::*;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(about = "EAGLE-3 speculative decoding (target + draft GGUF)")]
struct Args {
    /// Path to the target (verifier) model GGUF.
    target: PathBuf,

    /// Path to the EAGLE-3 draft model GGUF (converted with `--target-model-dir`).
    draft: PathBuf,

    /// Maximum number of tokens to draft per round (upstream `--spec-draft-n-max`).
    #[arg(long, default_value_t = 8)]
    n_draft_max: i32,

    /// Minimum draft-token probability; drafts below this are dropped
    /// (upstream `--spec-draft-p-min`).
    #[arg(long, default_value_t = 0.5)]
    p_min: f32,

    /// Recurrent-state snapshots per sequence. Leave 0 for standard (non-recurrent)
    /// targets like Qwen3/Llama; set `>= n-draft-max` for hybrid/recurrent targets.
    #[arg(long, default_value_t = 0)]
    n_rs_seq: u32,

    /// Context size.
    #[arg(short = 'c', long, default_value_t = NonZeroU32::new(2048).unwrap())]
    ctx_size: NonZeroU32,

    /// Number of layers to offload to the GPU (target model).
    #[arg(long, default_value_t = 1000)]
    n_gpu_layers: u32,

    /// If set, generate up to this many tokens through the EAGLE-3 draft loop.
    #[arg(long)]
    predict: Option<i32>,

    /// Prompt (only used when --predict is set).
    #[arg(long, default_value = "The capital of France is")]
    prompt: String,
}

fn build_ctx_params(args: &Args) -> LlamaContextParams {
    let mut params = LlamaContextParams::default()
        .with_n_ctx(Some(args.ctx_size))
        .with_ctx_type(LlamaContextType::Default);
    if args.n_rs_seq > 0 {
        params = params.with_n_rs_seq(args.n_rs_seq);
    }
    params
}

fn main() -> Result<()> {
    let args = Args::parse();
    let backend = LlamaBackend::init()?;

    // Both target and EAGLE-3 draft are loaded as ordinary models.
    let target_model_params = LlamaModelParams::default().with_n_gpu_layers(args.n_gpu_layers);
    let target_model = LlamaModel::load_from_file(&backend, &args.target, &target_model_params)
        .with_context(|| format!("failed to load target model from {}", args.target.display()))?;

    let draft_model_params = LlamaModelParams::default().with_n_gpu_layers(args.n_gpu_layers);
    let draft_model = LlamaModel::load_from_file(&backend, &args.draft, &draft_model_params)
        .with_context(|| format!("failed to load draft model from {}", args.draft.display()))?;

    let mut target_ctx = target_model.new_context(&backend, build_ctx_params(&args))?;
    let mut draft_ctx = draft_model.new_context(&backend, build_ctx_params(&args))?;

    println!(
        "target context: n_ctx={}  ({})",
        target_ctx.n_ctx(),
        args.target.display()
    );
    println!(
        "draft  context: n_ctx={}  ({})",
        draft_ctx.n_ctx(),
        args.draft.display()
    );

    let session_config = Eagle3SessionConfig::new(1, args.n_draft_max).with_p_min(args.p_min);
    let mut session = match Eagle3Session::new_with_config(&target_ctx, &draft_ctx, session_config)
    {
        Ok(s) => s,
        Err(e) => {
            println!("\nEAGLE-3 session could not be created: {e}");
            println!(
                "(Is `{}` a valid EAGLE-3 draft model for this target? It must be",
                args.draft.display()
            );
            println!(" converted with `convert_hf_to_gguf.py --target-model-dir <target>`.)");
            return Ok(());
        }
    };

    println!(
        "EAGLE-3 session: n_draft_max={}, p_min={}, need_embd={}, need_embd_pre_norm={}",
        session.n_draft_max(),
        session.p_min(),
        session.need_embd(),
        session.need_embd_pre_norm(),
    );

    let Some(n_predict) = args.predict else {
        println!();
        println!(
            "Both models ready and the EAGLE-3 session initialised. Pass --predict N to generate."
        );
        return Ok(());
    };

    run_speculative(
        &target_model,
        &mut target_ctx,
        &mut draft_ctx,
        &mut session,
        &args.prompt,
        n_predict,
    )
}

fn run_speculative(
    model: &LlamaModel,
    target_ctx: &mut LlamaContext<'_>,
    draft_ctx: &mut LlamaContext<'_>,
    session: &mut Eagle3Session,
    prompt: &str,
    n_predict: i32,
) -> Result<()> {
    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize prompt: {prompt}"))?;
    if tokens.is_empty() {
        return Err(anyhow!("prompt tokenised to zero tokens"));
    }

    // Prompt prefill: decode the whole prompt as a single batch on the target.
    let n_batch_max = target_ctx.n_batch() as usize;
    let prefill_capacity = tokens.len().max(n_batch_max);
    let mut batch = LlamaBatch::new(prefill_capacity, 1);
    let last_idx = tokens.len() - 1;
    for (i, tok) in tokens.iter().copied().enumerate() {
        batch.add(tok, i as i32, &[0], i == last_idx)?;
    }
    target_ctx.decode(&mut batch).context("prefill failed")?;
    // EAGLE-3 harvests the target's extracted hidden states from each decoded
    // batch; hand every target decode to the session.
    session
        .process(&batch)
        .context("EAGLE-3 process(prefill) failed")?;
    session.begin(0, &tokens)?;

    // Greedy sampling keeps draft/verify token-exact so acceptance reflects the
    // draft quality rather than sampling noise.
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut last_token = sampler.sample(target_ctx, batch.n_tokens() - 1);
    sampler.accept(last_token);

    let emit = |s: &str| {
        print!("{s}");
        let _ = std::io::stdout().flush();
    };
    emit(prompt);
    emit(&model.token_to_str(last_token, Special::Tokenize)?);

    let mut n_past = tokens.len() as i32;
    let mut n_generated: i32 = 1;
    let mut n_draft_calls: u64 = 0;
    let mut n_drafts_total: u64 = 0;
    let mut n_accepted_total: u64 = 0;

    let verify_cap = (session.n_draft_max() as usize + 1).max(n_batch_max);
    let mut verify = LlamaBatch::new(verify_cap, 1);

    let t_start = ggml_time_us();

    while n_generated < n_predict {
        if model.is_eog_token(last_token) {
            break;
        }

        let drafts = session.draft(0, n_past, last_token)?;
        n_draft_calls += 1;
        n_drafts_total += drafts.len() as u64;

        // Verify batch: [last_token, drafts...], all producing logits.
        verify.clear();
        verify.add(last_token, n_past, &[0], true)?;
        for (i, d) in drafts.iter().enumerate() {
            verify.add(*d, n_past + 1 + i as i32, &[0], true)?;
        }
        let n_verify = verify.n_tokens();

        target_ctx
            .decode(&mut verify)
            .context("verify decode failed")?;

        // draft() autoregressively advanced the draft context's KV at the
        // speculative positions (>= n_past). process(verify) re-decodes those
        // same positions with the target's harvested features, so roll the
        // draft KV back to n_past first to avoid a position collision
        // (mirrors the MTP example).
        draft_ctx
            .clear_kv_cache_seq(Some(0), Some(n_past as u32), None)
            .context("draft KV rollback before process failed")?;

        session
            .process(&verify)
            .context("EAGLE-3 process(verify) failed")?;

        // Longest accepted prefix: sample the target at each output position and
        // compare against the corresponding draft. Output 0 predicts draft[0].
        let mut n_accepted: usize = 0;
        let mut next_token = sampler.sample(target_ctx, 0);
        sampler.accept(next_token);
        for (i, draft) in drafts.iter().enumerate() {
            if next_token == *draft {
                n_accepted = i + 1;
                if i + 1 < n_verify as usize {
                    next_token = sampler.sample(target_ctx, (i + 1) as i32);
                    sampler.accept(next_token);
                }
            } else {
                break;
            }
        }
        n_accepted_total += n_accepted as u64;

        let new_n_past = n_past + 1 + n_accepted as i32;

        // Trim the rejected suffix from both KV caches so the next round starts
        // from a consistent state.
        if (n_accepted as i32) < drafts.len() as i32 {
            let ok = target_ctx
                .clear_kv_cache_seq(Some(0), Some(new_n_past as u32), None)
                .context("target KV rollback errored")?;
            if !ok {
                return Err(anyhow!(
                    "target context refused partial seq_rm at pos {new_n_past} — \
                     for recurrent/hybrid targets pass --n-rs-seq >= n-draft-max"
                ));
            }
            let _ = draft_ctx.clear_kv_cache_seq(Some(0), Some(new_n_past as u32), None);
        }

        // Only report acceptance when a draft was actually produced. EAGLE-3
        // can return an empty draft (e.g. before enough target features have
        // been harvested), and upstream only tracks per-seq accept state for a
        // sequence that drafted — calling accept() otherwise aborts on an
        // internal assert. (MTP always drafts, so its example skips this guard.)
        if !drafts.is_empty() {
            session.accept(0, n_accepted as u16)?;
        }

        for d in drafts.iter().take(n_accepted) {
            emit(&model.token_to_str(*d, Special::Tokenize)?);
        }
        emit(&model.token_to_str(next_token, Special::Tokenize)?);

        last_token = next_token;
        n_past = new_n_past;
        n_generated += (n_accepted as i32) + 1;
    }

    let t_end = ggml_time_us();
    let dur = Duration::from_micros((t_end - t_start) as u64);

    println!("\n");
    println!(
        "generated {} tokens in {:.2}s = {:.1} tok/s",
        n_generated,
        dur.as_secs_f32(),
        n_generated as f32 / dur.as_secs_f32()
    );
    let acceptance = if n_drafts_total == 0 {
        0.0
    } else {
        n_accepted_total as f32 / n_drafts_total as f32
    };
    println!(
        "EAGLE-3: {} draft calls, {} drafts proposed, {} accepted ({:.1}% acceptance)",
        n_draft_calls,
        n_drafts_total,
        n_accepted_total,
        100.0 * acceptance
    );
    session.print_stats();
    Ok(())
}
