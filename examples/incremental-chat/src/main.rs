//! Chat example with incremental prefill — processes prompt tokens while the
//! user is still typing so that generation starts almost instantly when they
//! press Enter.
//!
//! # How it works
//!
//! 1. The system prompt is prefilled once at startup and kept across turns.
//! 2. As the user types, the current input is periodically tokenized (debounced).
//! 3. New tokens (beyond what is already in the KV cache) are decoded in
//!    small batches, **withholding the last 2 tokens** to avoid BPE churn.
//! 4. If the user deletes/edits text (including mid-line edits via arrow
//!    keys), the KV cache is trimmed from the divergence point.
//! 5. When the user presses Enter, the remaining tokens are flushed and
//!    generation begins immediately.
//! 6. Conversation history is kept in the KV cache across turns with
//!    sliding-window eviction when the context fills up.
//!
//! # Editing keys
//!
//! | Key | Action |
//! |-----|--------|
//! | **←/→** | Move cursor |
//! | **Home / Ctrl+A** | Jump to start of line |
//! | **End / Ctrl+E** | Jump to end of line |
//! | **Backspace** | Delete char before cursor |
//! | **Delete** | Delete char after cursor |
//! | **Ctrl+W** | Delete word before cursor |
//! | **Ctrl+U** | Clear entire line |
//! | **Ctrl+K** | Delete from cursor to end of line |
//! | **Alt+Enter** | Insert newline (multi-line input) |
//! | **Enter** | Submit |
//! | **Ctrl+C** | Cancel generation / quit (press twice) |
//! | **Ctrl+D** | Quit |
//!
//! ```console
//! cargo run --release -p incremental-chat -- local path/to/model.gguf
//! ```
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod line_editor;
mod prefill;

use std::ffi::CString;
use std::io::{self, Write};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use colored::Colorize;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal;
use hf_hub::api::sync::ApiBuilder;

use llama_cpp_4::context::params::LlamaContextParams;
use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::llama_batch::LlamaBatch;
use llama_cpp_4::model::params::kv_overrides::ParamOverrideValue;
use llama_cpp_4::model::params::LlamaModelParams;
use llama_cpp_4::model::LlamaModel;
use llama_cpp_4::model::{AddBos, LlamaChatMessage, Special};
use llama_cpp_4::quantize::GgmlType;
use llama_cpp_4::sampling::LlamaSampler;
use llama_cpp_4::token::LlamaToken;

use line_editor::LineEditor;
use prefill::IncrementalPrefill;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const BATCH_SIZE: usize = 512;
const CONTEXT_RESERVE: usize = 64;

#[derive(clap::Parser, Debug, Clone)]
#[command(about = "Chat with incremental prefill — processes tokens while you type")]
struct Args {
    #[command(subcommand)]
    model: Model,
    #[arg(long, default_value_t = 512)]
    n_len: i32,
    #[arg(long, default_value = "You are a helpful assistant. Be concise in your answers.")]
    system_prompt: String,
    #[arg(long, default_value_t = 2)]
    keep_turns: usize,
    #[arg(short = 'o', value_parser = parse_key_val)]
    key_value_overrides: Vec<(String, ParamOverrideValue)>,
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    #[clap(long)]
    disable_gpu: bool,
    #[arg(short = 't', long)]
    threads: Option<i32>,
    #[arg(long)]
    threads_batch: Option<i32>,
    #[arg(short = 'c', long)]
    ctx_size: Option<NonZeroU32>,
    #[arg(long, default_value_t = 150)]
    debounce_ms: u64,
    #[arg(long)]
    no_flash_attn: bool,

    /// KV cache quantization type (f16, q8_0, q5_0, q4_0)
    #[arg(long, default_value = "f16", value_parser = parse_kv_type)]
    kv_type: GgmlType,

    /// Disable TurboQuant attention rotation
    #[arg(long)]
    no_turbo_quant: bool,

    /// Path to cache the system-prompt session file (skips re-prefill on restart)
    #[arg(long)]
    session_cache: Option<PathBuf>,
}

fn parse_kv_type(s: &str) -> Result<GgmlType, String> {
    match s.to_lowercase().as_str() {
        "f16" => Ok(GgmlType::F16),
        "q8_0" | "q8" => Ok(GgmlType::Q8_0),
        "q5_0" | "q5" => Ok(GgmlType::Q5_0),
        "q5_1" => Ok(GgmlType::Q5_1),
        "q4_0" | "q4" => Ok(GgmlType::Q4_0),
        "q4_1" => Ok(GgmlType::Q4_1),
        _ => Err(format!("unknown KV type '{s}', expected: f16, q8_0, q5_0, q5_1, q4_0, q4_1")),
    }
}

fn parse_key_val(s: &str) -> Result<(String, ParamOverrideValue)> {
    let pos = s
        .find('=')
        .ok_or_else(|| anyhow!("invalid KEY=value: no `=` found in `{s}`"))?;
    let key = s[..pos].parse()?;
    let value: String = s[pos + 1..].parse()?;
    let value = i64::from_str(&value)
        .map(ParamOverrideValue::Int)
        .or_else(|_| f64::from_str(&value).map(ParamOverrideValue::Float))
        .or_else(|_| bool::from_str(&value).map(ParamOverrideValue::Bool))
        .map_err(|_| anyhow!("must be one of i64, f64, or bool"))?;
    Ok((key, value))
}

#[derive(clap::Subcommand, Debug, Clone)]
enum Model {
    Local { path: PathBuf },
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

// ---------------------------------------------------------------------------
// Conversation state
// ---------------------------------------------------------------------------

struct Conversation {
    messages: Vec<LlamaChatMessage>,
    prefix_tokens: Vec<LlamaToken>,
}

impl Conversation {
    fn new(sys_msg: LlamaChatMessage, prefix_tokens: Vec<LlamaToken>) -> Self {
        Self {
            messages: vec![sys_msg],
            prefix_tokens,
        }
    }

    fn prefix_len(&self) -> usize {
        self.prefix_tokens.len()
    }

    fn turn_count(&self) -> usize {
        (self.messages.len() - 1) / 2
    }

    fn push_turn(
        &mut self,
        model: &LlamaModel,
        user_msg: LlamaChatMessage,
        asst_msg: LlamaChatMessage,
        total_token_count: usize,
    ) -> Result<()> {
        self.messages.push(user_msg);
        self.messages.push(asst_msg);
        let formatted = model.apply_chat_template(None, &self.messages, false)?;
        self.prefix_tokens = model.str_to_token(&formatted, AddBos::Always)?;
        if self.prefix_tokens.len() != total_token_count {
            self.prefix_tokens.resize(
                total_token_count,
                *self.prefix_tokens.last().unwrap_or(&LlamaToken(0)),
            );
        }
        Ok(())
    }

    fn tokenize_with_user_message(
        &self,
        model: &LlamaModel,
        user_text: &str,
    ) -> Result<Vec<LlamaToken>> {
        let user_msg = LlamaChatMessage::new("user".into(), user_text.into())
            .context("invalid user message")?;
        let mut msgs = self.messages.clone();
        msgs.push(user_msg);
        let formatted = model.apply_chat_template(None, &msgs, true)?;
        model
            .str_to_token(&formatted, AddBos::Always)
            .context("tokenize current turn")
    }

    fn evict_oldest(&mut self, model: &LlamaModel, keep: usize) -> Result<Vec<LlamaToken>> {
        let n_turns = self.turn_count();
        let evict = n_turns.saturating_sub(keep);
        if evict > 0 {
            self.messages.drain(1..1 + evict * 2);
        }
        let formatted = model.apply_chat_template(None, &self.messages, false)?;
        self.prefix_tokens = model.str_to_token(&formatted, AddBos::Always)?;
        Ok(self.prefix_tokens.clone())
    }
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

enum InputMsg {
    TextChanged(String),
    Submit(String),
    Interrupt,
}

// ---------------------------------------------------------------------------
// Input thread with full line editor
// ---------------------------------------------------------------------------

fn input_thread(
    tx: mpsc::Sender<InputMsg>,
    debounce: Duration,
    phase: Arc<AtomicBool>,
) {
    let mut editor = LineEditor::new();
    let mut last_change = Instant::now();
    let mut pending_send = false;

    loop {
        let timeout = if pending_send {
            debounce.saturating_sub(last_change.elapsed())
        } else {
            Duration::from_millis(50)
        };

        if event::poll(timeout).unwrap_or(false) {
            if let Ok(Event::Key(key)) = event::read() {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                let generating = phase.load(Ordering::Relaxed);
                let has_alt = key.modifiers.contains(KeyModifiers::ALT);
                let has_shift = key.modifiers.contains(KeyModifiers::SHIFT);
                let has_ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

                match key.code {
                    KeyCode::Char('c') if has_ctrl => {
                        if tx.send(InputMsg::Interrupt).is_err() {
                            break;
                        }
                    }
                    KeyCode::Char('d') if has_ctrl => {
                        if editor.is_empty() {
                            let _ = tx.send(InputMsg::Interrupt);
                            break;
                        } else if !generating {
                            // Delete char at cursor (like bash)
                            if editor.delete() {
                                editor.redraw();
                                last_change = Instant::now();
                                pending_send = true;
                            }
                        }
                    }
                    _ if generating => {} // swallow during generation

                    // ── Cursor movement ────────────────────────────────
                    KeyCode::Left => {
                        if editor.move_left() {
                            editor.redraw();
                        }
                    }
                    KeyCode::Right => {
                        if editor.move_right() {
                            editor.redraw();
                        }
                    }
                    KeyCode::Home | KeyCode::Char('a') if has_ctrl || key.code == KeyCode::Home => {
                        editor.home();
                        editor.redraw();
                    }
                    KeyCode::End | KeyCode::Char('e') if has_ctrl || key.code == KeyCode::End => {
                        editor.end();
                        editor.redraw();
                    }

                    // ── Deletion ───────────────────────────────────────
                    KeyCode::Backspace => {
                        if editor.backspace() {
                            editor.redraw();
                            last_change = Instant::now();
                            pending_send = true;
                        }
                    }
                    KeyCode::Delete => {
                        if editor.delete() {
                            editor.redraw();
                            last_change = Instant::now();
                            pending_send = true;
                        }
                    }
                    KeyCode::Char('w') if has_ctrl => {
                        if editor.delete_word_back() {
                            editor.redraw();
                            last_change = Instant::now();
                            pending_send = true;
                        }
                    }
                    KeyCode::Char('u') if has_ctrl => {
                        if editor.clear_all() {
                            editor.redraw();
                            last_change = Instant::now();
                            pending_send = true;
                        }
                    }
                    KeyCode::Char('k') if has_ctrl => {
                        if editor.kill_to_end() {
                            editor.redraw();
                            last_change = Instant::now();
                            pending_send = true;
                        }
                    }

                    // ── Newline / Submit ────────────────────────────────
                    KeyCode::Enter if has_alt || has_shift => {
                        editor.insert_newline();
                        editor.redraw();
                        last_change = Instant::now();
                        pending_send = true;
                    }
                    KeyCode::Enter => {
                        print!("\r\n");
                        let _ = io::stdout().flush();
                        let _ = tx.send(InputMsg::Submit(editor.take_text()));
                        pending_send = false;
                    }

                    // ── Character input ────────────────────────────────
                    KeyCode::Char(c) if !has_ctrl && !has_alt => {
                        let at_end = editor.cursor_at_end();
                        editor.insert_char(c);
                        if at_end {
                            // Fast path: just print the char
                            print!("{c}");
                            let _ = io::stdout().flush();
                        } else {
                            // Mid-line insert: full redraw
                            editor.redraw();
                        }
                        last_change = Instant::now();
                        pending_send = true;
                    }
                    _ => {}
                }
            }
        }

        if pending_send && last_change.elapsed() >= debounce {
            let _ = tx.send(InputMsg::TextChanged(editor.text().to_string()));
            pending_send = false;
        }
    }
}

fn drain_latest(rx: &mpsc::Receiver<InputMsg>) -> Option<InputMsg> {
    let mut latest: Option<InputMsg> = None;
    while let Ok(msg) = rx.try_recv() {
        match &msg {
            InputMsg::Submit(_) | InputMsg::Interrupt => return Some(msg),
            InputMsg::TextChanged(_) => latest = Some(msg),
        }
    }
    latest
}

// ---------------------------------------------------------------------------
// Prefill helpers
// ---------------------------------------------------------------------------

fn prefill_tokens(
    ctx: &mut llama_cpp_4::context::LlamaContext<'_>,
    batch: &mut LlamaBatch,
    tokens: &[LlamaToken],
) -> Result<()> {
    let mut offset = 0usize;
    for chunk in tokens.chunks(BATCH_SIZE) {
        batch.clear();
        let is_final_chunk = offset + chunk.len() == tokens.len();
        for (i, &token) in chunk.iter().enumerate() {
            let pos = (offset + i) as i32;
            let is_last = is_final_chunk && i == chunk.len() - 1;
            batch.add(token, pos, &[0], is_last)?;
        }
        ctx.decode(batch).context("prefill decode")?;
        offset += chunk.len();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = terminal::disable_raw_mode();
        default_hook(info);
    }));

    terminal::enable_raw_mode().context("failed to enable raw mode")?;
    let result = run();
    let _ = terminal::disable_raw_mode();
    result
}

fn run() -> Result<()> {
    let Args {
        n_len,
        model,
        system_prompt,
        keep_turns,
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        disable_gpu,
        key_value_overrides,
        threads,
        threads_batch,
        ctx_size,
        debounce_ms,
        no_flash_attn,
        kv_type,
        no_turbo_quant,
        session_cache,
    } = Args::parse();

    let mut backend = LlamaBackend::init()?;
    backend.void_logs();

    let model_params = {
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        if !disable_gpu {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        } else {
            LlamaModelParams::default()
        }
        #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
        LlamaModelParams::default()
    };

    let mut model_params = pin!(model_params);
    for (k, v) in &key_value_overrides {
        let k = CString::new(k.as_bytes()).with_context(|| format!("invalid key: {k}"))?;
        model_params.as_mut().append_kv_override(k.as_c_str(), *v);
    }

    let model_path = model.get_or_load()?;
    let model =
        LlamaModel::load_from_file(&backend, model_path, &model_params).context("load model")?;

    let n_ctx_size = ctx_size.or_else(|| NonZeroU32::new(4096));
    let mut ctx_params = LlamaContextParams::default()
        .with_n_ctx(n_ctx_size)
        .with_flash_attention(!no_flash_attn)
        .with_cache_type_k(kv_type)
        .with_cache_type_v(kv_type)
        .with_attn_rot_disabled(no_turbo_quant);
    if let Some(t) = threads {
        ctx_params = ctx_params.with_n_threads(t);
    }
    if let Some(t) = threads_batch.or(threads) {
        ctx_params = ctx_params.with_n_threads_batch(t);
    }

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .context("create context")?;
    let n_ctx = ctx.n_ctx() as usize;

    // ── System prompt (with optional session caching) ──────────────────
    let sys_msg = LlamaChatMessage::new("system".into(), system_prompt.clone())
        .context("invalid system prompt")?;
    let sys_formatted = model.apply_chat_template(None, &[sys_msg.clone()], false)?;
    let sys_tokens = model.str_to_token(&sys_formatted, AddBos::Always)?;
    let mut batch = LlamaBatch::new(BATCH_SIZE, 1);

    let sys_prefill_start = Instant::now();
    let sys_loaded_from_cache = if let Some(ref cache_path) = session_cache {
        if cache_path.exists() {
            match ctx.load_session_file(cache_path, sys_tokens.len()) {
                Ok(cached_tokens) if cached_tokens == sys_tokens => {
                    eprintln!(
                        "{}",
                        format!("[system prompt loaded from cache: {}]", cache_path.display())
                            .bright_black()
                    );
                    true
                }
                _ => {
                    eprintln!(
                        "{}",
                        "[session cache stale/incompatible — re-prefilling]"
                            .bright_yellow()
                    );
                    false
                }
            }
        } else {
            false
        }
    } else {
        false
    };

    if !sys_loaded_from_cache {
        prefill_tokens(&mut ctx, &mut batch, &sys_tokens)?;

        // Save session cache for next startup
        if let Some(ref cache_path) = session_cache {
            match ctx.save_session_file(cache_path, &sys_tokens) {
                Ok(()) => {
                    eprintln!(
                        "{}",
                        format!("[system prompt cached to: {}]", cache_path.display())
                            .bright_black()
                    );
                }
                Err(e) => {
                    eprintln!(
                        "{}",
                        format!("[failed to save session cache: {e}]").bright_yellow()
                    );
                }
            }
        }
    }
    let sys_prefill_time = sys_prefill_start.elapsed();

    let mut conv = Conversation::new(sys_msg, sys_tokens.clone());

    let kv_label = format!("{kv_type:?}").to_lowercase();
    eprintln!(
        "{}",
        format!(
            "[system: {} tok ({:.1?}{}) | ctx: {} | kv: {} | turbo: {} | flash: {} | keeps {} turns]",
            sys_tokens.len(),
            sys_prefill_time,
            if sys_loaded_from_cache { " cached" } else { "" },
            n_ctx,
            kv_label,
            if no_turbo_quant { "off" } else { "on" },
            if no_flash_attn { "off" } else { "on" },
            keep_turns,
        )
        .bright_black()
    );

    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::common(), LlamaSampler::greedy()]);

    // ── Input thread ───────────────────────────────────────────────────
    let (tx, rx) = mpsc::channel::<InputMsg>();
    let debounce = Duration::from_millis(debounce_ms);
    let phase = Arc::new(AtomicBool::new(false));
    let phase_clone = Arc::clone(&phase);
    thread::spawn(move || input_thread(tx, debounce, phase_clone));

    println!(
        "{}",
        "←/→ to move cursor, edit anywhere. Alt+Enter=newline, Ctrl-C=cancel, Ctrl-D=quit"
            .bright_black()
    );

    let mut ctrl_c_count: u32;

    loop {
        print!("\r{} ", "user>".green());
        let _ = io::stdout().flush();

        let mut ip = IncrementalPrefill::new(conv.prefix_len(), BATCH_SIZE);

        phase.store(false, Ordering::Relaxed);
        ctrl_c_count = 0;

        // ── Input loop ─────────────────────────────────────────────────
        let user_text = 'input: loop {
            let msg = match rx.recv() {
                Ok(m) => m,
                Err(_) => break 'input None,
            };
            let msg = drain_latest(&rx).unwrap_or(msg);

            match msg {
                InputMsg::TextChanged(text) => {
                    ctrl_c_count = 0;
                    match conv.tokenize_with_user_message(&model, &text) {
                        Ok(turn_tokens) if turn_tokens.len() > conv.prefix_len() => {
                            let user_part = &turn_tokens[conv.prefix_len()..];
                            match ip.prefill_speculative(&mut ctx, &mut batch, user_part) {
                                Ok(n) if n > 0 => {
                                    let s = format!(
                                        " [pre {}/{}]",
                                        ip.len(),
                                        user_part.len()
                                    );
                                    // Save cursor, print status, restore cursor
                                    eprint!("\x1b[s\x1b[999C{}\x1b[u", s.bright_black());
                                    let _ = io::stderr().flush();
                                }
                                Err(e) => {
                                    eprintln!("\r\n{}", format!("[prefill err: {e}]").bright_red());
                                }
                                _ => {}
                            }
                        }
                        Err(e) => {
                            eprintln!("\r\n{}", format!("[tokenize err: {e}]").bright_red());
                        }
                        _ => {}
                    }
                }
                InputMsg::Submit(text) => break 'input Some(text),
                InputMsg::Interrupt => {
                    ctrl_c_count += 1;
                    if ctrl_c_count >= 2 {
                        break 'input None;
                    }
                    eprint!("\r\n{}", "(Ctrl-C again to quit)".bright_yellow());
                    print!("\r\n{} ", "user>".green());
                    let _ = io::stdout().flush();
                }
            }
        };

        let user_text = match user_text {
            Some(t) if !t.trim().is_empty() => t,
            Some(_) => continue,
            None => {
                print!("\r\n{}\r\n", "goodbye!".bright_black());
                return Ok(());
            }
        };

        // ── Context capacity ───────────────────────────────────────────
        let mut turn_tokens = conv.tokenize_with_user_message(&model, &user_text)?;
        let mut total_pos = turn_tokens.len();

        if total_pos + n_len as usize + CONTEXT_RESERVE > n_ctx {
            if conv.turn_count() > 0 {
                eprintln!(
                    "\r\n{}",
                    format!("[ctx {}/{} — evicting, keeping last {}]", total_pos, n_ctx, keep_turns)
                        .bright_yellow()
                );
                ctx.clear_kv_cache();
                let new_prefix = conv.evict_oldest(&model, keep_turns)?;
                prefill_tokens(&mut ctx, &mut batch, &new_prefix)?;
                ip = IncrementalPrefill::new(conv.prefix_len(), BATCH_SIZE);

                turn_tokens = conv.tokenize_with_user_message(&model, &user_text)?;
                total_pos = turn_tokens.len();
            }
            if total_pos + n_len as usize + CONTEXT_RESERVE > n_ctx {
                let max_user = n_ctx
                    .saturating_sub(conv.prefix_len())
                    .saturating_sub(n_len as usize)
                    .saturating_sub(CONTEXT_RESERVE);
                let user_part_len = total_pos - conv.prefix_len();
                eprintln!(
                    "\r\n{}",
                    format!("[msg too long ({user_part_len} tok) — truncating to {max_user}]")
                        .bright_red()
                );
                turn_tokens.truncate(conv.prefix_len() + max_user);
                total_pos = turn_tokens.len();
            }
        }

        // ── Final flush ────────────────────────────────────────────────
        let user_part = &turn_tokens[conv.prefix_len()..];
        let flushed = ip.flush(&mut ctx, &mut batch, user_part)?;

        eprintln!(
            "\r{}",
            format!(
                "[{} tok | pre: {} | flush: {} | hist: {} | ctx: {}/{}]",
                user_part.len(),
                user_part.len() - flushed,
                flushed,
                conv.turn_count(),
                total_pos,
                n_ctx,
            )
            .bright_black()
        );

        // ── Generation ─────────────────────────────────────────────────
        phase.store(true, Ordering::Relaxed);
        print!("{} ", "assistant>".red());
        let _ = io::stdout().flush();

        let gen_start = Instant::now();
        let mut n_cur = total_pos as i32;
        let mut n_gen = 0u32;
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut response = String::new();
        let mut cancelled = false;
        let mut ttft: Option<Duration> = None;

        let gen_limit = n_cur + n_len;
        while n_cur < gen_limit {
            if let Ok(InputMsg::Interrupt) = rx.try_recv() {
                cancelled = true;
                break;
            }

            let tok = sampler.sample(&ctx, batch.n_tokens() - 1);
            if ttft.is_none() {
                ttft = Some(gen_start.elapsed());
            }
            if model.is_eog_token(tok) {
                break;
            }

            let bytes = model.token_to_bytes(tok, Special::Tokenize)?;
            let mut s = String::with_capacity(32);
            let _ = decoder.decode_to_string(&bytes, &mut s, false);
            print!("{s}");
            let _ = io::stdout().flush();
            response.push_str(&s);

            batch.clear();
            batch.add(tok, n_cur, &[0], true)?;
            n_cur += 1;
            n_gen += 1;
            ctx.decode(&mut batch).context("decode failed")?;
        }

        let elapsed = gen_start.elapsed();
        if cancelled {
            print!("{}", " [cancelled]".bright_yellow());
        }
        print!("\r\n");

        let tps = if elapsed.as_secs_f64() > 0.0 {
            n_gen as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        eprintln!(
            "{}",
            format!(
                "[{n_gen} tok | {tps:.1} tok/s | ttft: {:.1?} | total: {elapsed:.1?}{}]",
                ttft.unwrap_or_default(),
                if cancelled { " | cancelled" } else { "" },
            )
            .bright_black()
        );
        print!("\r\n");
        let _ = io::stdout().flush();

        // ── Update history ─────────────────────────────────────────────
        let user_msg = LlamaChatMessage::new("user".into(), user_text).context("user msg")?;
        let asst_msg =
            LlamaChatMessage::new("assistant".into(), response).context("asst msg")?;
        conv.push_turn(&model, user_msg, asst_msg, n_cur as usize)?;
        sampler = LlamaSampler::chain_simple([LlamaSampler::common(), LlamaSampler::greedy()]);
    }
}
