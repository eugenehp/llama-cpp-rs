//! MTP (Multi-Token Prediction) speculative-decoding demo.
//!
//! Pairs a target context with an MTP draft context and drives the full
//! speculative-decode loop from Rust via [`llama_cpp_4::mtp::MtpSession`],
//! which is a safe wrapper around upstream's `common_speculative_*` MTP
//! implementation (PR #22673).
//!
//! Without `--predict`, the example just builds both contexts and reports
//! their configuration (useful as a connectivity smoke test on any MTP GGUF).
//! With `--predict N`, it generates up to `N` tokens through the draft loop
//! and reports how many drafts were accepted.
//!
//! # Verified invocation
//!
//! ```sh
//! cargo run --release -p mtp --features metal -- \
//!     --predict 64 --prompt "The capital of France is" \
//!     hf-model froggeric/Qwen3.6-27B-MTP-GGUF Qwen3.6-27B-IQ2_M-mtp.gguf
//! ```
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_4::context::params::{LlamaContextParams, LlamaContextType};
use llama_cpp_4::ggml_time_us;
use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::llama_batch::LlamaBatch;
use llama_cpp_4::model::params::LlamaModelParams;
use llama_cpp_4::model::{AddBos, LlamaModel, Special};
use llama_cpp_4::mtp::MtpSession;
use llama_cpp_4::sampling::LlamaSampler;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser, Debug)]
struct Args {
    /// How to locate the GGUF (local path or hf-hub repo/file).
    #[command(subcommand)]
    model: Model,

    /// Maximum number of tokens to draft per round. Upstream defaults to 3 for
    /// Qwen3.6 MTP and reports that as the sweet spot.
    #[arg(long, default_value_t = 3)]
    n_draft_max: i32,

    /// Number of recurrent-state snapshots per sequence (must be >= n_draft_max).
    #[arg(long, default_value_t = 4)]
    n_rs_seq: u32,

    /// Context size.
    #[arg(short = 'c', long, default_value_t = NonZeroU32::new(2048).unwrap())]
    ctx_size: NonZeroU32,

    /// If set, generate up to this many tokens through the MTP draft loop.
    #[arg(long)]
    predict: Option<i32>,

    /// Prompt (only used when --predict is set).
    #[arg(long, default_value = "The capital of France is")]
    prompt: String,
}

#[derive(clap::Subcommand, Debug, Clone)]
enum Model {
    Local {
        path: PathBuf,
    },
    #[clap(name = "hf-model")]
    HuggingFace {
        repo: String,
        file: String,
    },
}

impl Model {
    fn resolve(self) -> Result<PathBuf> {
        match self {
            Model::Local { path } => Ok(path),
            Model::HuggingFace { repo, file } => ApiBuilder::new()
                .with_progress(true)
                .build()
                .context("unable to create huggingface api")?
                .model(repo)
                .get(&file)
                .context("unable to download model"),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let model_path = args.model.resolve()?;

    let backend = LlamaBackend::init()?;

    let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .with_context(|| format!("failed to load model from {}", model_path.display()))?;

    let target_params = LlamaContextParams::default()
        .with_n_ctx(Some(args.ctx_size))
        .with_ctx_type(LlamaContextType::Default)
        // Required for hybrid/recurrent models (e.g. Qwen3.6) so that partial
        // KV rollback after rejected drafts succeeds. Without it, the
        // recurrent layers refuse seq_rm and the next verify batch fails the
        // M-RoPE monotonic-position check.
        .with_n_rs_seq(args.n_rs_seq);
    let draft_params = LlamaContextParams::default()
        .with_n_ctx(Some(args.ctx_size))
        .with_ctx_type(LlamaContextType::Mtp)
        .with_n_rs_seq(args.n_rs_seq);

    let mut target_ctx = model.new_context(&backend, target_params)?;
    let draft_ctx = match model.new_context(&backend, draft_params) {
        Ok(c) => c,
        Err(e) => {
            println!("MTP draft context could not be created: {e}");
            println!("(This GGUF likely lacks MTP heads. Try:");
            println!("   hf-model froggeric/Qwen3.6-27B-MTP-GGUF Qwen3.6-27B-IQ2_M-mtp.gguf)");
            return Ok(());
        }
    };

    println!(
        "target context: ctx_type={:?}, n_ctx={}",
        LlamaContextType::Default,
        target_ctx.n_ctx()
    );
    println!(
        "draft  context: ctx_type={:?}, n_ctx={}, n_rs_seq={}",
        LlamaContextType::Mtp,
        draft_ctx.n_ctx(),
        draft_ctx.n_rs_seq()
    );

    let mut draft_ctx = draft_ctx;
    let mut session = MtpSession::new(&target_ctx, &draft_ctx, 1, args.n_draft_max)?;
    println!(
        "MTP session: n_draft_max={}, need_embd={}",
        session.n_draft_max(),
        session.need_embd()
    );

    let Some(n_predict) = args.predict else {
        println!();
        println!("Both contexts ready. Pass --predict N to drive the draft loop.");
        return Ok(());
    };

    run_speculative(
        &model,
        &mut target_ctx,
        &mut draft_ctx,
        &mut session,
        &args.prompt,
        n_predict,
    )
}

fn run_speculative(
    model: &LlamaModel,
    target_ctx: &mut llama_cpp_4::context::LlamaContext<'_>,
    draft_ctx: &mut llama_cpp_4::context::LlamaContext<'_>,
    session: &mut MtpSession,
    prompt: &str,
    n_predict: i32,
) -> Result<()> {
    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize prompt: {prompt}"))?;
    if tokens.is_empty() {
        return Err(anyhow!("prompt tokenised to zero tokens"));
    }

    // Prompt prefill: decode the whole prompt as a single batch.
    let n_batch_max = target_ctx.n_batch() as usize;
    let prefill_capacity = tokens.len().max(n_batch_max);
    let mut batch = LlamaBatch::new(prefill_capacity, 1);
    // MtpSession::new configures the target context with pre-norm extraction
    // in unmasked mode (upstream PR #23198), so pre-norm rows are written for
    // every prompt token regardless of batch.logits. Only the final position
    // needs logits=true — that's what the first sample reads from.
    let last_idx = tokens.len() - 1;
    for (i, tok) in tokens.iter().copied().enumerate() {
        batch.add(tok, i as i32, &[0], i == last_idx)?;
    }
    target_ctx.decode(&mut batch).context("prefill failed")?;
    session
        .process(&batch)
        .context("MTP process(prefill) failed")?;
    session.begin(0, &tokens)?;

    // Sample the first token from the prefill.
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut last_token = sampler.sample(target_ctx, batch.n_tokens() - 1);
    sampler.accept(last_token);

    let mut output_text = String::new();
    output_text.push_str(prompt);
    let mut emit = |s: &str| {
        output_text.push_str(s);
        print!("{s}");
        let _ = std::io::stdout().flush();
    };
    emit(&model.token_to_str(last_token, Special::Tokenize)?);

    let mut n_past = tokens.len() as i32;
    let mut n_generated: i32 = 1;
    let mut n_draft_calls: u64 = 0;
    let mut n_drafts_total: u64 = 0;
    let mut n_accepted_total: u64 = 0;

    // Verification batch can hold last_token + up to n_draft_max drafts.
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

        // Build verify batch: [last_token, drafts...], all with output logits.
        verify.clear();
        verify.add(last_token, n_past, &[0], true)?;
        for (i, d) in drafts.iter().enumerate() {
            verify.add(*d, n_past + 1 + i as i32, &[0], true)?;
        }
        let n_verify = verify.n_tokens();

        // Roll back the draft context's KV to before draft()'s AR
        // pre-advancement. process(verify) is about to re-decode the same
        // positions on the draft side but with target's pre-norm h injected.
        // n_rs_seq on the draft context lets that recurrent-state rollback
        // succeed even though M-RoPE positions can't normally be re-written.
        draft_ctx
            .clear_kv_cache_seq(Some(0), Some(n_past as u32), None)
            .context("draft KV rollback failed")?;

        target_ctx
            .decode(&mut verify)
            .context("verify decode failed")?;
        session
            .process(&verify)
            .context("MTP process(verify) failed")?;

        // Sample target at each output position and find the longest matching
        // prefix of the drafts. Output index 0 corresponds to the logits
        // following last_token (i.e. predicts draft[0]).
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

        // last_token + n_accepted drafts are now committed (positions
        // [n_past, n_past + n_accepted]); next_token is the new generated
        // token but lives only as a sample — its KV entry will be created
        // when we use it as last_token in the next iteration.
        let new_n_past = n_past + 1 + n_accepted as i32;

        // Roll back the rejected suffix on BOTH contexts. After verify the
        // target and draft KVs both reach [0..n_past+drafts.len()]; keep only
        // up to position new_n_past - 1.
        if (n_accepted as i32) < drafts.len() as i32 {
            let ok = target_ctx
                .clear_kv_cache_seq(Some(0), Some(new_n_past as u32), None)
                .context("target KV rollback errored")?;
            if !ok {
                return Err(anyhow!(
                    "target context refused partial seq_rm at pos {new_n_past} — \
                     ensure with_n_rs_seq(>0) is set on the target context"
                ));
            }
            let ok = draft_ctx
                .clear_kv_cache_seq(Some(0), Some(new_n_past as u32), None)
                .context("draft KV rollback errored")?;
            if !ok {
                return Err(anyhow!(
                    "draft context refused partial seq_rm at pos {new_n_past}"
                ));
            }
        }

        // Tell MTP how many of its drafts were accepted (updates per-seq
        // pending-h carryover; recurrent state is rolled back via n_rs_seq).
        session.accept(0, n_accepted as u16)?;

        // Emit the accepted drafts plus the new sampled token.
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

    println!();
    println!();
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
        "MTP: {} draft calls, {} drafts proposed, {} accepted ({:.1}% acceptance)",
        n_draft_calls,
        n_drafts_total,
        n_accepted_total,
        100.0 * acceptance
    );
    Ok(())
}
