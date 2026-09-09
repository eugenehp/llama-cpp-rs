//! # Speculative decoding with no draft model
//!
//! Ordinary speculative decoding runs a second, smaller model to propose
//! tokens. This proposes them by **looking up where the text repeats**: find
//! the last place the recent tokens occurred, and guess that what followed then
//! will follow now.
//!
//! That costs a hash lookup instead of a forward pass, needs no extra weights
//! and no extra VRAM, and wins on exactly the workloads where text repeats —
//! code editing, RAG over a quoted document, JSON with recurring keys, chat
//! that restates the question. On prose that never repeats it drafts nothing
//! and degrades to ordinary decoding.
//!
//! ## The correctness guarantee
//!
//! Speculative decoding is *lossless*: a draft token is kept only if the target
//! model would have produced it anyway. With greedy sampling that means the
//! output must be **byte-identical** to plain greedy decoding. This example
//! runs both and compares them, so the demo is also a check — see
//! `--verify`.
//!
//! ## Usage
//!
//! ```console
//! # Repetitive prompt: drafting should land often
//! cargo run -p ngram -- --model model.gguf \
//!   -p "Repeat after me: the cat sat on the mat. The cat sat on the mat. The cat"
//!
//! # Verify the output matches plain greedy decoding
//! cargo run -p ngram -- --model model.gguf -p "..." --verify
//!
//! # Statistical cache instead of plain lookup, persisted between runs
//! cargo run -p ngram -- --model model.gguf -p "..." --strategy cache --cache-file ngrams.bin
//! ```

use anyhow::{bail, Context, Result};
use clap::Parser;
use llama_cpp_4::llama_batch::LlamaBatch;
use llama_cpp_4::model::{AddBos, Special};
use llama_cpp_4::prelude::*;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Instant;

#[derive(clap::Parser, Debug)]
#[command(about = "Speculative decoding with no draft model (n-gram lookup)")]
struct Args {
    /// Path to the model GGUF
    #[arg(long)]
    model: PathBuf,

    /// The prompt
    #[arg(short = 'p', long, default_value = "The cat sat on the mat. The cat")]
    prompt: String,

    /// Number of tokens to generate
    #[arg(short = 'n', long, default_value_t = 64)]
    n_predict: usize,

    /// Which lookup strategy to draft with
    #[arg(long, value_enum, default_value_t = Strategy::Simple)]
    strategy: Strategy,

    /// Maximum tokens to draft per round
    #[arg(long, default_value_t = 4)]
    n_draft: usize,

    /// Size of the n-gram used as a lookup key
    #[arg(long, default_value_t = 2)]
    ngram_size: u16,

    /// Persist the statistical cache here (only with `--strategy cache`)
    #[arg(long)]
    cache_file: Option<PathBuf>,

    /// Context size
    #[arg(long, default_value_t = 2048)]
    ctx_size: u32,

    /// Also decode without drafting and check the outputs match
    #[arg(long)]
    verify: bool,

    /// Layers to offload to the GPU
    #[arg(long, default_value_t = 0)]
    n_gpu_layers: u32,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum Strategy {
    /// Stateless: reuse the single most recent repeat.
    Simple,
    /// Statistical: learn a distribution over continuations, save/loadable.
    Cache,
    /// Adaptive: track how well its own drafts land and back off if they miss.
    Map,
}

/// What one run produced, so drafted and undrafted runs can be compared.
struct Run {
    tokens: Vec<LlamaToken>,
    text: String,
    elapsed_ms: u128,
    /// Draft tokens proposed across the whole run.
    drafted: usize,
    /// Draft tokens the target model agreed with.
    accepted: usize,
    /// Target-model forward passes.
    decodes: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let backend = LlamaBackend::init()?;

    let model_params = LlamaModelParams::default().with_n_gpu_layers(args.n_gpu_layers);
    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)
        .with_context(|| format!("failed to load model from {}", args.model.display()))?;

    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(args.ctx_size));

    let prompt_tokens = model
        .str_to_token(&args.prompt, AddBos::Always)
        .context("tokenizing the prompt")?;

    if prompt_tokens.len() + args.n_predict >= args.ctx_size as usize {
        bail!(
            "prompt ({} tokens) plus --n-predict ({}) exceeds --ctx-size ({})",
            prompt_tokens.len(),
            args.n_predict,
            args.ctx_size
        );
    }

    println!("prompt: {:?}", args.prompt);
    println!(
        "strategy: {:?}, n_draft={}, ngram_size={}\n",
        args.strategy, args.n_draft, args.ngram_size
    );

    let mut ctx = model.new_context(&backend, ctx_params.clone())?;
    let drafted_run = run(&model, &mut ctx, &prompt_tokens, &args, true)?;
    report("with drafting", &drafted_run);

    if args.verify {
        // Fresh context: the KV cache carries state from the previous run.
        let mut ctx = model.new_context(&backend, ctx_params)?;
        let plain_run = run(&model, &mut ctx, &prompt_tokens, &args, false)?;
        report("without drafting", &plain_run);

        // The guarantee: a draft token survives only if the target would have
        // produced it, so greedy output must match exactly.
        if drafted_run.tokens == plain_run.tokens {
            println!("\n✓ identical output — speculative decoding was lossless");
        } else {
            bail!(
                "outputs differ, which breaks the losslessness guarantee\n\
                 with drafting:    {:?}\n\
                 without drafting: {:?}",
                drafted_run.text,
                plain_run.text
            );
        }
    }

    Ok(())
}

/// Generate `n_predict` tokens, optionally drafting with n-gram lookup.
//
// One long function on purpose: it is a single decode loop threading `tokens`,
// `n_past`, `id`, the batch and the drafting state through every round.
// Splitting it would mean passing all of that between helpers, which obscures
// the position bookkeeping that is the tricky part.
#[allow(clippy::too_many_lines)]
fn run(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    prompt_tokens: &[LlamaToken],
    args: &Args,
    draft: bool,
) -> Result<Run> {
    // Greedy, so the comparison between drafted and undrafted runs is
    // deterministic. `CommonSampler` is used rather than a hand-built chain
    // because `sample_and_accept_n` — the acceptance half of speculative
    // decoding — lives on it.
    let mut params = CommonSamplerParams::new();
    let mut scalars = params.scalars();
    scalars.temp = 0.0;
    scalars.top_k = 1;
    params.set_scalars(&scalars);
    let mut sampler = CommonSampler::new(model, &mut params)?;

    // Drafting state. Only one of these is used, per `--strategy`.
    let mut cache = NgramCache::new();
    if let (Strategy::Cache, Some(path)) = (args.strategy, args.cache_file.as_ref()) {
        if path.exists() {
            cache = NgramCache::load(path.to_str().context("cache path is not UTF-8")?)
                .with_context(|| format!("loading {}", path.display()))?;
            println!("loaded {} n-grams from {}", cache.len(), path.display());
        }
    }
    let mut map = NgramMap::new(args.ngram_size, u16::try_from(args.n_draft)?, false, 1)?;

    let mut tokens: Vec<LlamaToken> = prompt_tokens.to_vec();
    let mut generated: Vec<LlamaToken> = Vec::new();
    let mut drafted = 0usize;
    let mut accepted = 0usize;
    let mut decodes = 0usize;

    let started = Instant::now();

    // ── prefill ─────────────────────────────────────────────────────────────
    let mut batch = LlamaBatch::new(args.ctx_size as usize, 1);
    for (i, token) in tokens.iter().enumerate() {
        batch.add(*token, i32::try_from(i)?, &[0], i == tokens.len() - 1)?;
    }
    ctx.decode(&mut batch).context("prefill")?;
    decodes += 1;

    if args.strategy == Strategy::Map {
        map.begin(&tokens)?;
    }

    // `n_past` counts tokens in the KV cache; `id` is the most recently
    // sampled token, which is *not* in it yet. Keeping that token out of the
    // cache is what lets each round decode it together with the draft.
    let mut n_past = i32::try_from(tokens.len())?;
    let mut id = sampler.sample(ctx, batch.n_tokens() - 1, false)?;
    sampler.accept(id, true);
    tokens.push(id);
    generated.push(id);

    while generated.len() < args.n_predict && !model.is_eog_token(id) {
        // ── draft ───────────────────────────────────────────────────────────
        // History excludes `id`, which the drafters take separately as the
        // just-sampled token.
        let history = &tokens[..tokens.len() - 1];

        let mut proposal: Vec<LlamaToken> = if draft {
            match args.strategy {
                Strategy::Simple => {
                    ngram_simple_draft(args.ngram_size, u16::try_from(args.n_draft)?, history, id)?
                }
                Strategy::Cache => ngram_cache_draft(
                    &tokens,
                    i32::try_from(args.n_draft)?,
                    1,
                    i32::from(args.ngram_size) + 2,
                    Some(&mut cache),
                    None,
                    None,
                )?,
                Strategy::Map => map.draft(history, id)?,
            }
        } else {
            Vec::new()
        };
        proposal.truncate(args.n_draft);

        // Never draft past the budget — an accepted token beyond it would be
        // generated only to be discarded.
        let room = args.n_predict - generated.len();
        proposal.truncate(room);
        drafted += proposal.len();

        // ── verify ──────────────────────────────────────────────────────────
        // The batch is `[id, draft...]` with logits on every position, because
        // `sample_and_accept_n` reads logits `0..=draft.len()`: index 0 is the
        // distribution after `id` (predicting draft[0]'s slot), index i+1 the
        // distribution after draft[i]. One forward pass covers the whole draft,
        // which is the entire saving.
        batch.clear();
        batch.add(id, n_past, &[0], true)?;
        for (i, token) in proposal.iter().enumerate() {
            batch.add(*token, n_past + 1 + i32::try_from(i)?, &[0], true)?;
        }
        ctx.decode(&mut batch).context("decoding the draft")?;
        decodes += 1;

        // Every accepted draft token, plus one freshly sampled at the first
        // divergence — so it is never empty and never longer than draft + 1.
        let accepted_now = sampler.sample_and_accept_n(ctx, &proposal, false)?;
        let n_accepted_draft = accepted_now.len().saturating_sub(1);
        accepted += n_accepted_draft;

        if args.strategy == Strategy::Map {
            map.accept(u16::try_from(n_accepted_draft).unwrap_or(u16::MAX));
        }

        // ── roll back rejected drafts ───────────────────────────────────────
        // The batch wrote `id` plus every draft token into the cache. Only the
        // matched prefix is real; the rest must go, or the next round decodes
        // on top of tokens that were never generated.
        //
        // After trimming, the cache holds positions `0 ..= n_past + len - 1`.
        // `accepted_now.last()` is the newly sampled token, which was never
        // decoded, so it stays out — it becomes the next round's `id`.
        let new_n_past = n_past + i32::try_from(accepted_now.len())?;
        let decoded_through = n_past + i32::try_from(proposal.len())?;
        if decoded_through >= new_n_past {
            ctx.clear_kv_cache_seq(Some(0), Some(u32::try_from(new_n_past)?), None)
                .context("rolling back rejected draft tokens")?;
        }

        for token in &accepted_now {
            tokens.push(*token);
            generated.push(*token);
            if generated.len() >= args.n_predict {
                break;
            }
        }
        n_past = new_n_past;
        id = *accepted_now.last().expect("never empty");

        if args.strategy == Strategy::Cache {
            let n_tokens = i32::try_from(tokens.len()).unwrap_or(i32::MAX);
            cache.update(1, i32::from(args.ngram_size) + 2, &tokens, n_tokens, false)?;
        }
    }

    let elapsed_ms = started.elapsed().as_millis();

    if let (Strategy::Cache, Some(path)) = (args.strategy, args.cache_file.as_ref()) {
        cache
            .save(path.to_str().context("cache path is not UTF-8")?)
            .with_context(|| format!("saving {}", path.display()))?;
        println!("saved {} n-grams to {}", cache.len(), path.display());
    }

    let mut text = String::new();
    for token in &generated {
        text.push_str(&model.token_to_str(*token, Special::Tokenize).unwrap_or_default());
    }

    Ok(Run {
        tokens: generated,
        text,
        elapsed_ms,
        drafted,
        accepted,
        decodes,
    })
}

fn report(label: &str, run: &Run) {
    println!("── {label} ──");
    println!("{}", run.text.trim_end());
    println!(
        "  {} tokens in {} ms, {} target decodes",
        run.tokens.len(),
        run.elapsed_ms,
        run.decodes
    );
    if run.drafted > 0 {
        // Acceptance rate is the number that matters: below roughly 30% the
        // wasted verification outweighs the saved forward passes.
        #[allow(clippy::cast_precision_loss)] // token counts are far below 2^52
        let rate = 100.0 * run.accepted as f64 / run.drafted as f64;
        println!(
            "  drafted {} tokens, accepted {} ({rate:.0}%)",
            run.drafted, run.accepted
        );
    } else {
        println!("  no tokens drafted");
    }
    println!();
}
