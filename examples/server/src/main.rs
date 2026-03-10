//! OpenAI-compatible chat/completion/embedding server using llama.cpp.
//!
//! # Endpoints
//!
//! | Method | Path                    | Description                     |
//! |--------|-------------------------|---------------------------------|
//! | GET    | `/health`               | Liveness check                  |
//! | GET    | `/v1/models`            | List loaded model                |
//! | POST   | `/v1/chat/completions`  | Chat (streaming + non-streaming) |
//! | POST   | `/v1/completions`       | Raw text completion (streaming)  |
//! | POST   | `/v1/embeddings`        | Dense embeddings                 |
//!
//! # Usage
//!
//! ```console
//! # Local file
//! cargo run -p openai-server -- local path/to/model.gguf
//!
//! # Hugging Face (interactive quant picker)
//! cargo run -p openai-server -- hf-model unsloth/Qwen3.5-397B-A17B-GGUF
//!
//! # Hugging Face (pick quant by name, download all shards)
//! cargo run -p openai-server -- hf-model unsloth/Qwen3.5-397B-A17B-GGUF Q4_K_M
//!
//! # With GPU + auth key
//! cargo run -p openai-server --features metal -- \
//!     --n-gpu-layers 99 --api-key secret \
//!     hf-model bartowski/Llama-3.2-3B-Instruct-GGUF Q4_K_M
//! ```
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod tools;

use actix_multipart::Multipart;
use actix_web::{http::StatusCode, web, App, HttpRequest, HttpResponse, HttpServer};
use anyhow::Context as _;
use clap::Parser;
use futures_util::{stream, StreamExt as _};
use hf_hub::api::sync::{Api, ApiBuilder};
#[cfg(feature = "mtmd")]
use llama_cpp_4::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText,
};
use llama_cpp_4::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaChatMessage, LlamaModel, Special},
    sampling::LlamaSampler,
};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    num::NonZeroU32,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, RwLock, Semaphore};
use tools::{extract_tool_calls, inject_tools, normalise_messages, parse_tool_choice, parse_tools};
#[cfg(feature = "mtmd")]
use tools::{normalise_messages_multimodal, ImageSource};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "openai-server", about = "OpenAI-compatible llama.cpp server")]
struct Args {
    /// Host to listen on.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Number of layers to offload to GPU (0 = CPU only).
    #[arg(long, default_value_t = 0)]
    n_gpu_layers: u32,

    /// Context size override (default: use the model's trained context length).
    #[arg(short = 'c', long)]
    ctx_size: Option<NonZeroU32>,

    /// Require this bearer token on every request. Disabled when omitted.
    #[arg(long)]
    api_key: Option<String>,

    /// Maximum number of requests processed concurrently.
    /// llama.cpp contexts are not thread-safe so this effectively serialises
    /// inference while keeping HTTP connections responsive.
    #[arg(long, default_value_t = 1)]
    parallel: usize,

    /// Resolve (and download) the model, print its absolute local path to
    /// stdout, then exit without starting the server.
    /// Useful for scripts that need the cache path before launching.
    ///
    /// Example:
    ///   MODEL=$(cargo run -p openai-server -- --print-path \
    ///               hf-model bartowski/Llama-3.2-1B-Instruct-GGUF Q4_K_M)
    #[arg(long)]
    print_path: bool,

    // ── Multimodal (mtmd) ──────────────────────────────────────────────────
    /// Path to the multimodal projector (mmproj) GGUF file.
    /// Enables the `POST /v1/files` endpoint and image/audio inputs in chat
    /// completions.  Requires the `mtmd` Cargo feature.
    #[arg(long, value_name = "FILE")]
    mmproj: Option<PathBuf>,

    /// Number of threads used by the vision/audio encoder (default: 4).
    #[arg(long, default_value_t = 4)]
    mmproj_n_threads: i32,

    /// Do NOT offload the mmproj model to the GPU.
    #[arg(long)]
    no_mmproj_gpu: bool,

    #[command(subcommand)]
    model: ModelSource,
}

#[derive(clap::Subcommand, Debug)]
enum ModelSource {
    /// Load a model from a local file path.
    Local {
        /// Path to the GGUF model file.
        path: PathBuf,
    },
    /// Download a model from Hugging Face Hub (cached locally).
    ///
    /// If `<model>` is omitted the repo's GGUF files are listed and you are
    /// prompted to choose interactively (best quant auto-picked when stdin is
    /// not a terminal).  For sharded repos all shards are downloaded.
    #[clap(name = "hf-model")]
    HuggingFace {
        /// Repository id, e.g. `unsloth/Qwen3.5-397B-A17B-GGUF`.
        repo: String,
        /// Exact filename or quant directory name (e.g. `Q4_K_M`).
        /// Omit to pick interactively.
        model: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// HuggingFace model selection
// ---------------------------------------------------------------------------

const QUANT_PREFERENCE: &[&str] = &[
    "Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M", "Q5_K_S", "Q5_0", "Q3_K_M", "Q3_K_S", "Q8_0", "Q6_K",
    "Q2_K", "IQ4_XS", "IQ3_M",
];

#[derive(Debug)]
struct ModelGroup {
    label: String,
    files: Vec<String>,
}

impl ModelGroup {
    fn preference_score(&self) -> usize {
        QUANT_PREFERENCE
            .iter()
            .position(|q| self.label.to_uppercase().contains(q))
            .unwrap_or(usize::MAX)
    }
}

fn collect_groups(all_ggufs: Vec<String>) -> Vec<ModelGroup> {
    use std::collections::BTreeMap;
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in all_ggufs {
        let key = if let Some(slash) = path.find('/') {
            path[..slash].to_string()
        } else {
            let stem = path.trim_end_matches(".gguf");
            if let Some(of_pos) = stem.rfind("-of-") {
                let before_of = &stem[..of_pos];
                if let Some(dash) = before_of.rfind('-') {
                    let shard_num = &before_of[dash + 1..];
                    if shard_num.chars().all(|c| c.is_ascii_digit()) {
                        before_of[..dash].to_string()
                    } else {
                        stem.to_string()
                    }
                } else {
                    stem.to_string()
                }
            } else {
                stem.to_string()
            }
        };
        map.entry(key).or_default().push(path);
    }
    map.into_iter()
        .map(|(key, mut files)| {
            files.sort();
            let shard_info = if files.len() > 1 {
                format!("  [{} shards]", files.len())
            } else {
                String::new()
            };
            ModelGroup {
                label: format!("{key}{shard_info}"),
                files,
            }
        })
        .collect()
}

fn prompt_user(groups: &[ModelGroup]) -> anyhow::Result<usize> {
    use std::io::{self, IsTerminal as _, Write};
    eprintln!("\nAvailable models in repo:");
    for (i, g) in groups.iter().enumerate() {
        eprintln!("  {:>2})  {}", i + 1, g.label);
    }
    if !io::stdin().is_terminal() {
        let best = groups
            .iter()
            .enumerate()
            .min_by_key(|(_, g)| g.preference_score())
            .map(|(i, _)| i)
            .unwrap_or(0);
        eprintln!("\nNon-interactive — auto-selected: {}", groups[best].label);
        return Ok(best);
    }
    loop {
        eprint!("\nSelect a model [1–{}]: ", groups.len());
        io::stderr().flush().ok();
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        match line.trim().parse::<usize>() {
            Ok(n) if n >= 1 && n <= groups.len() => return Ok(n - 1),
            _ => eprintln!("  Enter a number between 1 and {}.", groups.len()),
        }
    }
}

fn resolve_hf(api: &Api, repo: &str, model: Option<String>) -> anyhow::Result<PathBuf> {
    let api_repo = api.model(repo.to_string());
    // Exact .gguf filename → download directly.
    if let Some(ref filename) = model {
        if filename.ends_with(".gguf") {
            return api_repo
                .get(filename)
                .with_context(|| format!("failed to download '{filename}' from '{repo}'"));
        }
    }
    let info = api_repo
        .info()
        .with_context(|| format!("failed to fetch repo info for '{repo}'"))?;
    let all_ggufs: Vec<String> = info
        .siblings
        .into_iter()
        .map(|s| s.rfilename)
        .filter(|n| n.ends_with(".gguf"))
        .collect();
    if all_ggufs.is_empty() {
        anyhow::bail!("no .gguf files found in repo '{repo}'");
    }
    let groups = collect_groups(all_ggufs);
    let chosen_idx = if let Some(filter) = model {
        let filter_up = filter.to_uppercase();
        groups
            .iter()
            .position(|g| {
                let label_key = g.label.split_whitespace().next().unwrap_or(&g.label);
                label_key.to_uppercase() == filter_up
                    || label_key.to_uppercase().contains(&filter_up)
            })
            .with_context(|| {
                let available: Vec<_> = groups
                    .iter()
                    .map(|g| {
                        g.label
                            .split_whitespace()
                            .next()
                            .unwrap_or(&g.label)
                            .to_string()
                    })
                    .collect();
                format!(
                    "no group matching '{filter}' in '{repo}'. Available: {}",
                    available.join(", ")
                )
            })?
    } else if groups.len() == 1 {
        eprintln!("Auto-selected: {}", groups[0].label);
        0
    } else {
        prompt_user(&groups)?
    };
    let group = &groups[chosen_idx];
    eprintln!("\nDownloading: {}", group.label);
    let mut first_path: Option<PathBuf> = None;
    for (i, file) in group.files.iter().enumerate() {
        if group.files.len() > 1 {
            eprintln!("  shard {}/{}: {file}", i + 1, group.files.len());
        }
        let path = api
            .model(repo.to_string())
            .get(file)
            .with_context(|| format!("failed to download shard '{file}'"))?;
        if first_path.is_none() {
            first_path = Some(path);
        }
    }
    first_path.ok_or_else(|| anyhow::anyhow!("no files downloaded"))
}

impl ModelSource {
    fn resolve(self) -> anyhow::Result<PathBuf> {
        match self {
            ModelSource::Local { path } => Ok(path),
            ModelSource::HuggingFace { repo, model } => {
                let api = ApiBuilder::new()
                    .with_progress(true)
                    .build()
                    .context("failed to build HF API client")?;
                resolve_hf(&api, &repo, model)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// mmproj auto-detection and download
// ---------------------------------------------------------------------------

/// Try to download the best mmproj GGUF from a HuggingFace repo.
///
/// Lists the repo's files, picks the best `mmproj-*.gguf` by the same
/// preference order as [`find_mmproj_in_dir`], downloads (or retrieves from
/// local cache) and returns its path.
#[cfg(feature = "mtmd")]
fn download_mmproj_from_hf(repo: &str) -> Option<PathBuf> {
    let api = match ApiBuilder::new().with_progress(true).build() {
        Ok(a) => a,
        Err(e) => {
            tracing::warn!("Could not build HF API client for mmproj lookup: {e}");
            return None;
        }
    };
    let api_repo = api.model(repo.to_string());

    let info = match api_repo.info() {
        Ok(i) => i,
        Err(e) => {
            tracing::warn!("Could not fetch repo info for '{repo}': {e}");
            return None;
        }
    };

    // Collect all mmproj GGUF filenames from the repo listing.
    let mut candidates: Vec<String> = info
        .siblings
        .into_iter()
        .map(|s| s.rfilename)
        .filter(|name| name.starts_with("mmproj") && name.ends_with(".gguf"))
        .collect();

    if candidates.is_empty() {
        tracing::warn!(
            "No mmproj-*.gguf files found in repo '{repo}'. \
             The repo may not include a vision projector."
        );
        return None;
    }

    // Sort by preference (same order as MMPROJ_PREFER).
    candidates.sort_by(|a, b| {
        let score = |name: &str| {
            MMPROJ_PREFER
                .iter()
                .position(|suf| name.ends_with(suf))
                .unwrap_or(MMPROJ_PREFER.len())
        };
        score(a).cmp(&score(b)).then_with(|| a.cmp(b))
    });

    let chosen = &candidates[0];
    tracing::info!(
        "Downloading mmproj '{chosen}' from '{repo}'{}…",
        if candidates.len() > 1 {
            format!(
                " ({} candidates; use --mmproj to override)",
                candidates.len()
            )
        } else {
            String::new()
        }
    );

    match api_repo.get(chosen) {
        Ok(path) => {
            tracing::info!("mmproj cached at: {}", path.display());
            Some(path)
        }
        Err(e) => {
            tracing::warn!("Failed to download '{chosen}' from '{repo}': {e}");
            None
        }
    }
}

/// Preference order when multiple mmproj files are found in the same directory.
/// Earlier entries win.
const _MMPROJ_PREFER: &[&str] = &[
    "-F16.gguf",
    "-f16.gguf",
    "-BF16.gguf",
    "-bf16.gguf",
    "-F32.gguf",
    "-f32.gguf",
];

/// Scan `dir` for files whose names start with `mmproj` and end with `.gguf`.
/// Returns the best match according to [`MMPROJ_PREFER`], or the first
/// alphabetically if none of the preferred suffixes match.
///
/// Skips the directory silently if it cannot be read.
#[cfg(feature = "mtmd")]
fn find_mmproj_in_dir(dir: &std::path::Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut candidates: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("gguf")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("mmproj"))
                    .unwrap_or(false)
        })
        .collect();

    if candidates.is_empty() {
        return None;
    }

    // Sort by preference, then alphabetically for a stable result.
    candidates.sort_by(|a, b| {
        let score = |p: &PathBuf| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            MMPROJ_PREFER
                .iter()
                .position(|suf| name.ends_with(suf))
                .unwrap_or(MMPROJ_PREFER.len())
        };
        score(a).cmp(&score(b)).then_with(|| a.cmp(b))
    });

    let chosen = candidates.remove(0);
    if !candidates.is_empty() {
        tracing::info!(
            "Auto-detected mmproj: {} ({} other candidate(s) in same dir; \
             use --mmproj to override)",
            chosen.display(),
            candidates.len()
        );
    } else {
        tracing::info!("Auto-detected mmproj: {}", chosen.display());
    }
    Some(chosen)
}

// ---------------------------------------------------------------------------
// File store
// ---------------------------------------------------------------------------

/// A file uploaded via `POST /v1/files`.
#[derive(Debug, Clone)]
struct FileEntry {
    id: String,
    filename: String,
    bytes: Vec<u8>,
    purpose: String,
    created_at: u64,
}

/// Generate a stable file ID by FNV-1a hashing the content + timestamp.
fn gen_file_id(data: &[u8]) -> String {
    let mut h: u64 = 0xcbf29ce484222325u64;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    for &b in &now_secs().to_le_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("file-{h:016x}")
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

struct AppState {
    backend: LlamaBackend,
    model: LlamaModel,
    chat_template: Option<String>,
    model_name: String,
    default_ctx_size: Option<NonZeroU32>,
    /// Limits the number of concurrent inference calls.
    inference_semaphore: Arc<Semaphore>,
    /// Optional bearer token that every request must present.
    api_key: Option<String>,
    /// In-memory store for files uploaded via `POST /v1/files`.
    file_store: Arc<RwLock<HashMap<String, FileEntry>>>,
    /// Multimodal context — `Some` when `--mmproj` is provided.
    #[cfg(feature = "mtmd")]
    mtmd_ctx: Option<MtmdContext>,
}

// ---------------------------------------------------------------------------
// HTTP error helpers
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct HttpError {
    status: StatusCode,
    r#type: &'static str,
    message: String,
}

fn bad_request(msg: impl Into<String>) -> HttpError {
    HttpError {
        status: StatusCode::BAD_REQUEST,
        r#type: "invalid_request_error",
        message: msg.into(),
    }
}

fn unauthorized(msg: impl Into<String>) -> HttpError {
    HttpError {
        status: StatusCode::UNAUTHORIZED,
        r#type: "authentication_error",
        message: msg.into(),
    }
}

fn internal_error(msg: impl Into<String>) -> HttpError {
    HttpError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        r#type: "server_error",
        message: msg.into(),
    }
}

fn error_response(err: HttpError) -> HttpResponse {
    let body = json!({
        "error": {
            "message": err.message,
            "type": err.r#type,
            "code": err.status.as_u16()
        }
    })
    .to_string();
    HttpResponse::build(err.status)
        .content_type("application/json")
        .body(body)
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

fn check_auth(req: &HttpRequest, state: &AppState) -> Option<HttpError> {
    let Some(ref expected) = state.api_key else {
        return None;
    };
    let auth = req
        .headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok());
    match auth {
        Some(v) if v == format!("Bearer {expected}") => None,
        _ => Some(unauthorized("invalid or missing API key")),
    }
}

// ---------------------------------------------------------------------------
// Request parsing
// ---------------------------------------------------------------------------

fn parse_stop_sequences(req: &Value) -> Result<Vec<String>, HttpError> {
    match req.get("stop") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|v| match v {
                Value::String(s) => Ok(s.clone()),
                _ => Err(bad_request("each element of 'stop' must be a string")),
            })
            .collect(),
        _ => Err(bad_request("'stop' must be a string or array of strings")),
    }
}

/// Convert `(role, content)` pairs into the `LlamaChatMessage` vec that
/// `apply_chat_template` expects.
fn to_chat_messages(pairs: Vec<(String, String)>) -> Result<Vec<LlamaChatMessage>, HttpError> {
    pairs
        .into_iter()
        .map(|(role, content)| {
            LlamaChatMessage::new(role.clone(), content)
                .map_err(|e| bad_request(format!("invalid message (role={role}): {e}")))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Core inference engine
// ---------------------------------------------------------------------------

/// All sampling / generation parameters extracted from a request.
#[allow(unused)]
struct InferenceParams {
    prompt: String,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    seed: u32,
    max_tokens: u32,
    stop_seqs: Vec<String>,
    /// Optional GBNF grammar string.
    grammar: Option<String>,
    /// Raw bytes for each media item (image or audio), in the order their
    /// markers appear in `prompt`.  Populated only when the `mtmd` feature is
    /// active and the request contains multimodal content.
    image_bytes: Vec<Vec<u8>>,
}

impl InferenceParams {
    fn from_request(req: &Value, prompt: String) -> Result<Self, HttpError> {
        let temperature = req
            .get("temperature")
            .and_then(Value::as_f64)
            .unwrap_or(1.0) as f32;
        if temperature < 0.0 {
            return Err(bad_request("'temperature' must be >= 0"));
        }
        let top_p = req.get("top_p").and_then(Value::as_f64).unwrap_or(1.0) as f32;
        if !(0.0 < top_p && top_p <= 1.0) {
            return Err(bad_request("'top_p' must be in (0, 1]"));
        }
        let top_k = req.get("top_k").and_then(Value::as_i64).unwrap_or(0) as i32;
        if top_k < 0 {
            return Err(bad_request("'top_k' must be >= 0"));
        }
        let seed = req.get("seed").and_then(Value::as_u64).unwrap_or(0) as u32;
        let max_tokens = req
            .get("max_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(1024) as u32;
        if max_tokens == 0 {
            return Err(bad_request("'max_tokens' must be > 0"));
        }
        let grammar = match req.get("grammar") {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Null) | None => None,
            _ => return Err(bad_request("'grammar' must be a GBNF string")),
        };
        let stop_seqs = parse_stop_sequences(req)?;
        Ok(InferenceParams {
            prompt,
            temperature,
            top_p,
            top_k,
            seed,
            max_tokens,
            stop_seqs,
            grammar,
            image_bytes: Vec::new(), // populated later by the multimodal path
        })
    }
}

/// Why the decode loop stopped.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

// ---------------------------------------------------------------------------
// Multimodal helpers  (compiled only when the `mtmd` feature is active)
// ---------------------------------------------------------------------------

/// Decode a `data:` URI or fetch an `http(s)://` URL, returning raw bytes.
#[cfg(feature = "mtmd")]
async fn fetch_url_bytes(url: &str) -> Result<Vec<u8>, HttpError> {
    tracing::info!("Fetching image: {}…", &url[..url.len().min(120)]);
    if let Some(rest) = url.strip_prefix("data:") {
        // data:[<mediatype>][;base64],<data>
        let comma = rest
            .find(',')
            .ok_or_else(|| bad_request("invalid data URI: missing ','"))?;
        let meta = &rest[..comma];
        let data = &rest[comma + 1..];
        if meta.ends_with(";base64") {
            use base64::Engine as _;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|e| bad_request(format!("base64 decode error: {e}")))?;
            tracing::info!("Decoded {} bytes from data URI", bytes.len());
            Ok(bytes)
        } else {
            // Plain text / URL-encoded — treat the raw bytes as the payload.
            Ok(data.as_bytes().to_vec())
        }
    } else if url.starts_with("http://") || url.starts_with("https://") {
        // Many CDNs (including Wikimedia) block requests that lack a
        // browser-like User-Agent and return an HTML error page instead of the
        // image.  stb_image then fails because it receives HTML, not JPEG/PNG.
        let client = reqwest::Client::builder()
            .user_agent(
                "Mozilla/5.0 (compatible; llama-cpp-rs; \
                 +https://github.com/utilityai/llama-cpp-rs)",
            )
            .build()
            .map_err(|e| internal_error(format!("reqwest client: {e}")))?;

        let resp = client
            .get(url)
            .send()
            .await
            .map_err(|e| bad_request(format!("failed to fetch image URL: {e}")))?;

        let status = resp.status();
        if !status.is_success() {
            return Err(bad_request(format!(
                "image URL returned HTTP {status}: {url}"
            )));
        }

        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("unknown")
            .to_owned();

        let bytes = resp
            .bytes()
            .await
            .map_err(|e| bad_request(format!("failed to read image response: {e}")))?;

        tracing::info!(
            "Downloaded {} bytes (content-type: {content_type}) from URL",
            bytes.len()
        );

        // Anything under 1 KB cannot be a real image — print the body so the
        // user can see what the server actually returned (redirect HTML, JSON
        // error, Cloudflare challenge, etc.).
        if bytes.len() < 1024 {
            let preview = std::str::from_utf8(&bytes).unwrap_or("(binary)");
            return Err(bad_request(format!(
                "image URL returned only {} bytes — not a valid image file. \
                 Response body: {preview:?}",
                bytes.len()
            )));
        }

        // Warn if the response looks like HTML rather than binary image data.
        // JPEG magic = 0xFF 0xD8; PNG = 0x89 0x50 0x4E; GIF = 0x47 0x49 0x46.
        if bytes.starts_with(b"<!") || bytes.starts_with(b"<h") || bytes.starts_with(b"<H") {
            return Err(bad_request(format!(
                "image URL returned HTML instead of an image. \
                 The server likely rejected the request (check the URL and any auth). \
                 First 200 bytes: {:?}",
                std::str::from_utf8(&bytes[..bytes.len().min(200)]).unwrap_or("(invalid utf-8)")
            )));
        }

        Ok(bytes.to_vec())
    } else {
        Err(bad_request(
            "unsupported image source: must start with 'data:', 'http://', or 'https://'",
        ))
    }
}

/// Resolve a list of [`ImageSource`] items to raw byte vectors.
/// `FileId` sources are looked up in the shared file store;
/// `Url` sources are decoded / fetched from the network.
#[cfg(feature = "mtmd")]
async fn resolve_image_sources(
    sources: Vec<ImageSource>,
    file_store: &RwLock<HashMap<String, FileEntry>>,
) -> Result<Vec<Vec<u8>>, HttpError> {
    let mut out = Vec::with_capacity(sources.len());
    for src in sources {
        let bytes = match src {
            ImageSource::Url(url) => fetch_url_bytes(&url).await?,
            ImageSource::FileId(id) => {
                let store = file_store.read().await;
                store.get(&id).map(|e| e.bytes.clone()).ok_or_else(|| {
                    bad_request(format!(
                        "file '{id}' not found — upload it first via POST /v1/files"
                    ))
                })?
            }
        };
        out.push(bytes);
    }
    Ok(out)
}

/// Multimodal inference: encode images with mtmd, then decode as normal.
///
/// Works like [`run_inference`] but uses `mtmd_tokenize` + `mtmd_helper_eval_chunks`
/// for the prefill step instead of a plain `llama_decode`.
#[cfg(feature = "mtmd")]
fn run_inference_multimodal<F>(
    state: &AppState,
    params: &InferenceParams,
    on_piece: F,
) -> Result<(u32, FinishReason), HttpError>
where
    F: FnMut(&str) -> bool,
{
    let mut on_piece = on_piece;
    let mtmd_ctx = state
        .mtmd_ctx
        .as_ref()
        .expect("run_inference_multimodal called without mtmd_ctx");

    // Vision models often embed 256–1024 tokens per image, so default to 8 K.
    const MM_DEFAULT_CTX: u32 = 8192;
    let n_ctx = state
        .default_ctx_size
        .map_or_else(
            || state.model.n_ctx_train().min(MM_DEFAULT_CTX),
            NonZeroU32::get,
        )
        .max(n_ctx_for_params(params));

    let n_batch = n_ctx.min(2048);

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_batch);

    let mut ctx = state
        .model
        .new_context(&state.backend, ctx_params)
        .map_err(|e| internal_error(format!("context init: {e}")))?;

    // ── Load bitmaps from raw bytes ───────────────────────────────────────────
    let bitmaps: Vec<MtmdBitmap> = params
        .image_bytes
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            MtmdBitmap::from_buf(mtmd_ctx, bytes)
                .map_err(|e| internal_error(format!("bitmap {i}: {e}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // ── Tokenize (text + image markers → chunks) ──────────────────────────────
    let input_text = MtmdInputText::new(&params.prompt, true, true);
    let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
    let mut chunks = MtmdInputChunks::new();
    mtmd_ctx
        .tokenize(&input_text, &bitmap_refs, &mut chunks)
        .map_err(|e| internal_error(format!("mtmd_tokenize: {e}")))?;

    tracing::info!(
        "run_inference_multimodal: {} image(s), prompt_len={}",
        params.image_bytes.len(),
        params.prompt.len()
    );

    // ── Evaluate all chunks (encodes images + decodes everything) ─────────────
    let mut n_past: i32 = 0;
    tracing::info!("Calling eval_chunks…");
    mtmd_ctx
        .eval_chunks(
            ctx.as_ptr(),
            &chunks,
            /* n_past */ 0,
            /* seq_id */ 0,
            n_batch as i32,
            /* logits_last */ true,
            &mut n_past,
        )
        .map_err(|e| internal_error(format!("mtmd_eval_chunks: {e}")))?;
    tracing::info!("eval_chunks done, n_past={n_past}");

    // ── Sampler chain ─────────────────────────────────────────────────────────
    let mut chain: Vec<LlamaSampler> = Vec::new();
    if let Some(gbnf) = &params.grammar {
        chain.push(LlamaSampler::grammar(&state.model, gbnf, "root"));
    }
    if params.temperature > 0.0 {
        if params.top_k > 0 {
            chain.push(LlamaSampler::top_k(params.top_k));
        }
        if params.top_p < 1.0 {
            chain.push(LlamaSampler::top_p(params.top_p, 1));
        }
        chain.push(LlamaSampler::temp(params.temperature));
        chain.push(LlamaSampler::dist(params.seed));
    } else {
        chain.push(LlamaSampler::greedy());
    }
    let sampler = LlamaSampler::chain_simple(chain);

    // ── Decode loop (identical structure to run_inference) ────────────────────
    let max_pos = n_past + params.max_tokens as i32;
    let mut completion_tokens: u32 = 0;
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut finish_reason = FinishReason::Stop;

    let max_stop_len = params.stop_seqs.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut window = String::new();
    let mut cancelled = false;

    let mut batch = LlamaBatch::new(1, 0);

    'decode: loop {
        if n_past >= max_pos {
            finish_reason = FinishReason::Length;
            break;
        }

        // -1 means "sample from the last position with logits computed".
        // After eval_chunks this is always correct, matching the mtmd-cli.cpp pattern.
        let token = sampler.sample(&ctx, -1);
        if state.model.is_eog_token(token) {
            break;
        }

        let bytes = state
            .model
            .token_to_bytes(token, Special::Plaintext)
            .map_err(|e| internal_error(format!("token_to_bytes: {e}")))?;
        let mut piece = String::with_capacity(8);
        let _ = decoder.decode_to_string(&bytes, &mut piece, false);
        completion_tokens += 1;

        window.push_str(&piece);

        for stop in &params.stop_seqs {
            if !stop.is_empty() && window.ends_with(stop.as_str()) {
                let emit_len = window.len() - stop.len();
                if emit_len > 0 {
                    let _ = on_piece(&window[..emit_len]);
                }
                break 'decode;
            }
        }

        if max_stop_len == 0 {
            if !on_piece(&window) {
                cancelled = true;
                break;
            }
            window.clear();
        } else {
            let keep = window.len().min(max_stop_len);
            let emit_len = window.len().saturating_sub(keep);
            if emit_len > 0 {
                if !on_piece(&window[..emit_len]) {
                    cancelled = true;
                    break;
                }
                let remaining = window[emit_len..].to_owned();
                window = remaining;
            }
        }

        batch.clear();
        batch
            .add(token, n_past, &[0], true)
            .map_err(|e| internal_error(format!("batch add: {e}")))?;
        n_past += 1;
        ctx.decode(&mut batch)
            .map_err(|e| internal_error(format!("decode: {e}")))?;
    }

    if !cancelled && !window.is_empty() {
        let _ = on_piece(&window);
    }

    Ok((completion_tokens, finish_reason))
}

/// Minimum context size needed to hold the prompt + generated tokens.
fn _n_ctx_for_params(params: &InferenceParams) -> u32 {
    // Rough upper bound: 4 chars per token on average.
    let prompt_est = (params.prompt.len() / 4 + 1) as u32;
    prompt_est + params.max_tokens
}

/// Run the full inference loop, calling `on_piece` for each decoded text
/// fragment.  `on_piece` returns `false` to stop early (e.g. cancelled
/// stream).  Returns `(completion_token_count, finish_reason)`.
fn run_inference<F>(
    state: &AppState,
    params: &InferenceParams,
    on_piece: F,
) -> Result<(u32, FinishReason), HttpError>
where
    F: FnMut(&str) -> bool,
{
    // ── Dispatch to multimodal path when images are present ───────────────────
    #[cfg(feature = "mtmd")]
    if !params.image_bytes.is_empty() {
        return if state.mtmd_ctx.is_some() {
            run_inference_multimodal(state, params, on_piece)
        } else {
            tracing::warn!(
                "Request contains {} image(s) but the server was started without --mmproj; \
                 images will be ignored and the prompt will be processed as text.",
                params.image_bytes.len()
            );
            // fall through to the text-only path below
            run_inference_text(state, params, on_piece)
        };
    }

    run_inference_text(state, params, on_piece)
}

fn run_inference_text<F>(
    state: &AppState,
    params: &InferenceParams,
    mut on_piece: F,
) -> Result<(u32, FinishReason), HttpError>
where
    F: FnMut(&str) -> bool,
{
    // ── Tokenise prompt ───────────────────────────────────────────────────────
    let tokens = state
        .model
        .str_to_token(&params.prompt, AddBos::Always)
        .map_err(|e| internal_error(format!("tokenisation failed: {e}")))?;

    let n_prompt = tokens.len() as u32;

    // When no explicit --ctx-size is set, default to the model's training
    // context but cap it at 4096.  n_ctx_train for modern models can be
    // 32 K–128 K tokens; allocating a full-size KV cache + compute buffer
    // for every request consumes tens of GB and reliably triggers OOM.
    // Users who need a larger window can set --ctx-size explicitly.
    const DEFAULT_MAX_CTX: u32 = 4096;
    let n_ctx = state
        .default_ctx_size
        .map_or_else(
            || state.model.n_ctx_train().min(DEFAULT_MAX_CTX),
            NonZeroU32::get,
        )
        .max(n_prompt + params.max_tokens);

    // n_batch controls the compute-buffer size inside llama.cpp.  Matching it
    // to n_ctx when n_ctx is large (e.g. 32 K) allocates a huge scratch
    // buffer even if the actual sequence is short.  Cap it independently.
    const DEFAULT_MAX_BATCH: u32 = 2048;
    let n_batch = n_ctx.min(DEFAULT_MAX_BATCH);

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_batch);

    let mut ctx = state
        .model
        .new_context(&state.backend, ctx_params)
        .map_err(|e| internal_error(format!("context init: {e}")))?;

    // ── Prefill ───────────────────────────────────────────────────────────────
    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    let last = tokens.len().saturating_sub(1) as i32;
    for (i, &tok) in tokens.iter().enumerate() {
        batch
            .add(tok, i as i32, &[0], i as i32 == last)
            .map_err(|e| internal_error(format!("batch add: {e}")))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| internal_error(format!("prefill: {e}")))?;

    // ── Sampler chain ─────────────────────────────────────────────────────────
    let mut chain: Vec<LlamaSampler> = Vec::new();
    if let Some(gbnf) = &params.grammar {
        chain.push(LlamaSampler::grammar(&state.model, gbnf, "root"));
    }
    if params.temperature > 0.0 {
        if params.top_k > 0 {
            chain.push(LlamaSampler::top_k(params.top_k));
        }
        if params.top_p < 1.0 {
            chain.push(LlamaSampler::top_p(params.top_p, 1));
        }
        chain.push(LlamaSampler::temp(params.temperature));
        chain.push(LlamaSampler::dist(params.seed));
    } else {
        chain.push(LlamaSampler::greedy());
    }
    let sampler = LlamaSampler::chain_simple(chain);

    // ── Decode loop ───────────────────────────────────────────────────────────
    let mut n_cur = batch.n_tokens();
    let max_pos = n_cur + params.max_tokens as i32;
    let mut completion_tokens: u32 = 0;
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut finish_reason = FinishReason::Stop;

    // For stop-sequence detection we keep a small rolling window of recently
    // generated (but not-yet-emitted) text.  Everything before the window has
    // already been forwarded to `on_piece`, so we never re-emit it when a stop
    // sequence is finally matched.
    let max_stop_len = params.stop_seqs.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut window = String::new();
    let mut cancelled = false;

    'decode: loop {
        if n_cur >= max_pos {
            finish_reason = FinishReason::Length;
            break;
        }

        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        if state.model.is_eog_token(token) {
            break;
        }

        let bytes = state
            .model
            .token_to_bytes(token, Special::Plaintext)
            .map_err(|e| internal_error(format!("token_to_bytes: {e}")))?;
        let mut piece = String::with_capacity(8);
        let _ = decoder.decode_to_string(&bytes, &mut piece, false);
        completion_tokens += 1;

        window.push_str(&piece);

        // Check stop sequences against the current window.
        for stop in &params.stop_seqs {
            if !stop.is_empty() && window.ends_with(stop.as_str()) {
                // Emit only the content that precedes the stop string.
                let emit_len = window.len() - stop.len();
                if emit_len > 0 {
                    let _ = on_piece(&window[..emit_len]);
                }
                break 'decode;
            }
        }

        // Emit the safe portion of the window – everything except the last
        // `max_stop_len` bytes, which might be a prefix of an upcoming stop
        // sequence and must stay buffered until we can rule that out.
        if max_stop_len == 0 {
            if !on_piece(&window) {
                cancelled = true;
                break;
            }
            window.clear();
        } else {
            let keep = window.len().min(max_stop_len);
            let emit_len = window.len().saturating_sub(keep);
            if emit_len > 0 {
                if !on_piece(&window[..emit_len]) {
                    cancelled = true;
                    break;
                }
                let remaining = window[emit_len..].to_owned();
                window = remaining;
            }
        }

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| internal_error(format!("batch add: {e}")))?;
        n_cur += 1;
        ctx.decode(&mut batch)
            .map_err(|e| internal_error(format!("decode: {e}")))?;
    }

    // Flush whatever remains in the window when the loop ended without
    // matching a stop sequence (EOG token, max_tokens, or caller cancel).
    if !cancelled && !window.is_empty() {
        let _ = on_piece(&window);
    }

    Ok((completion_tokens, finish_reason))
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------

fn sse_chunk(data: &Value) -> web::Bytes {
    web::Bytes::from(format!("data: {}\n\n", data))
}

fn sse_done() -> web::Bytes {
    web::Bytes::from("data: [DONE]\n\n")
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

// ---------------------------------------------------------------------------
// Chat completions  POST /v1/chat/completions
// ---------------------------------------------------------------------------

async fn chat_completions(
    req: HttpRequest,
    state: web::Data<AppState>,
    body: web::Bytes,
) -> HttpResponse {
    if let Some(err) = check_auth(&req, &state) {
        return error_response(err);
    }
    let text = match std::str::from_utf8(&body) {
        Ok(s) => s.to_owned(),
        Err(_) => return error_response(bad_request("body must be valid UTF-8")),
    };
    let parsed: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => return error_response(bad_request(format!("invalid JSON: {e}"))),
    };

    let streaming = parsed
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    // ── Early parameter validation ────────────────────────────────────────────
    // Validate sampling params *before* the expensive apply_chat_template call.
    // This guarantees that invalid values (e.g. temperature < 0) return the
    // correct 400 response rather than a 500 from a later failure.
    {
        let temperature = parsed
            .get("temperature")
            .and_then(Value::as_f64)
            .unwrap_or(1.0) as f32;
        if temperature < 0.0 {
            return error_response(bad_request("'temperature' must be >= 0"));
        }
        let top_p = parsed.get("top_p").and_then(Value::as_f64).unwrap_or(1.0) as f32;
        if !(0.0 < top_p && top_p <= 1.0) {
            return error_response(bad_request("'top_p' must be in (0, 1]"));
        }
        let top_k = parsed.get("top_k").and_then(Value::as_i64).unwrap_or(0) as i32;
        if top_k < 0 {
            return error_response(bad_request("'top_k' must be >= 0"));
        }
        if parsed.get("max_tokens").and_then(Value::as_u64) == Some(0) {
            return error_response(bad_request("'max_tokens' must be > 0"));
        }
        if matches!(
            parsed.get("grammar"),
            Some(v) if !v.is_string() && !v.is_null()
        ) {
            return error_response(bad_request("'grammar' must be a GBNF string"));
        }
    }

    // ── Parse tools ──────────────────────────────────────────────────────────
    let tool_defs = match parse_tools(&parsed) {
        Ok(t) => t,
        Err(e) => return error_response(e),
    };
    let tool_choice = match parse_tool_choice(&parsed) {
        Ok(c) => c,
        Err(e) => return error_response(e),
    };

    // ── Parse messages (with multimodal support when available) ─────────────
    // When `mtmd` is active and the server has an mmproj model, use the
    // multimodal normaliser: it replaces image_url / image_file parts with
    // the mtmd media marker and returns the sources for later resolution.
    // Otherwise fall back to the text-only normaliser (images are stripped).
    // Always run the multimodal parser so we can count image parts,
    // even when there is no mmproj — we use the count only for the warning.
    #[cfg(feature = "mtmd")]
    let (base_msg_pairs, image_sources) = {
        let marker = MtmdContext::default_marker();
        tracing::debug!("mtmd media marker: {:?}", marker);
        let (pairs, sources) = match normalise_messages_multimodal(&parsed, marker) {
            Ok(r) => r,
            Err(e) => return error_response(e),
        };
        if !sources.is_empty() {
            tracing::info!(
                n_images = sources.len(),
                mtmd_ctx_present = state.mtmd_ctx.is_some(),
                "Detected multimodal content in request"
            );
        }
        if !sources.is_empty() && state.mtmd_ctx.is_none() {
            tracing::warn!(
                n_images = sources.len(),
                "Request contains image(s) but the server was started without --mmproj. \
                 Images will be IGNORED and the prompt processed as plain text. \
                 Restart with `--mmproj <path-to-mmproj.gguf>` and a vision-capable model \
                 to enable multimodal inference."
            );
            // Fall back to the text-only normaliser so markers are not left in the prompt.
            match normalise_messages(&parsed) {
                Ok(text_pairs) => (text_pairs, vec![]),
                Err(e) => return error_response(e),
            }
        } else {
            (pairs, sources)
        }
    };

    #[cfg(not(feature = "mtmd"))]
    let base_msg_pairs = match normalise_messages(&parsed) {
        Ok(m) => m,
        Err(e) => return error_response(e),
    };

    // ── Build prompt from messages ───────────────────────────────────────────
    let prompt = {
        let mut msg_pairs = base_msg_pairs;

        // Inject tool definitions + usage instructions into the system message.
        inject_tools(&mut msg_pairs, &tool_defs, &tool_choice);

        let chat_msgs = match to_chat_messages(msg_pairs) {
            Ok(m) => m,
            Err(e) => return error_response(e),
        };

        let template_override = match parsed.get("chat_template") {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Null) | None => None,
            _ => return error_response(bad_request("'chat_template' must be a string")),
        };
        let template = template_override.or_else(|| state.chat_template.clone());
        match state.model.apply_chat_template(template, chat_msgs, true) {
            Ok(p) => p,
            Err(e) => return error_response(internal_error(format!("chat template: {e}"))),
        }
    };

    // ── Sampling params ───────────────────────────────────────────────────────
    let mut params = match InferenceParams::from_request(&parsed, prompt) {
        Ok(p) => p,
        Err(e) => return error_response(e),
    };

    // ── Resolve image sources → raw bytes (mtmd path only) ───────────────────
    #[cfg(feature = "mtmd")]
    if !image_sources.is_empty() {
        tracing::info!("Resolving {} image source(s)…", image_sources.len());
        match resolve_image_sources(image_sources, &state.file_store).await {
            Ok(bytes) => {
                tracing::info!(
                    "Images ready: {} image(s), sizes: {:?}",
                    bytes.len(),
                    bytes.iter().map(|b| b.len()).collect::<Vec<_>>()
                );
                params.image_bytes = bytes;
            }
            Err(e) => return error_response(e),
        }
    }

    // When tools are in play, give the model enough room to think and then
    // emit a complete tool call (thinking models like Qwen3.5 need extra
    // tokens for their <think>…</think> block before the <tool_call>).
    // Grammar-based forcing is intentionally NOT used here: GBNF grammars
    // conflict with models that use special tokens such as <tool_call>, and
    // they prevent thinking models from emitting their reasoning prefix.
    // The system-prompt injection (inject_tools) is sufficient for capable
    // models and avoids all of those compatibility issues.
    if !tool_defs.is_empty() && params.max_tokens < 1024 {
        params.max_tokens = 1024;
    }

    let model_name = parsed
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&state.model_name)
        .to_owned();
    let has_tools = !tool_defs.is_empty();
    let created = now_secs();
    let id = format!("chatcmpl-{created}");

    if streaming {
        run_chat_stream(state, params, id, model_name, created, has_tools).await
    } else {
        run_chat_blocking(state, params, id, model_name, created, has_tools).await
    }
}

async fn run_chat_blocking(
    state: web::Data<AppState>,
    params: InferenceParams,
    id: String,
    model_name: String,
    created: u64,
    has_tools: bool,
) -> HttpResponse {
    let permit = state.inference_semaphore.clone().acquire_owned().await;
    let state2 = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut raw = String::new();
        let outcome = run_inference(&state2, &params, |piece| {
            raw.push_str(piece);
            true
        });
        outcome.map(|(tokens, reason)| (raw, tokens, reason))
    })
    .await;

    match result {
        Ok(Ok((raw_output, completion_tokens, finish_reason))) => {
            let prompt_tokens = 0u32; // cheap approximation; full count needs a 2nd tokenise pass

            // Parse tool calls out of the raw output.
            let (content, tool_calls) = if has_tools {
                extract_tool_calls(&raw_output)
            } else {
                (raw_output, vec![])
            };

            let (final_finish, message) = if tool_calls.is_empty() {
                (
                    finish_reason.as_str(),
                    json!({ "role": "assistant", "content": content }),
                )
            } else {
                let calls_json: Vec<Value> = tool_calls.iter().map(|c| c.to_value()).collect();
                (
                    "tool_calls",
                    json!({
                        "role": "assistant",
                        "content": if content.is_empty() { Value::Null } else { Value::String(content) },
                        "tool_calls": calls_json
                    }),
                )
            };

            HttpResponse::Ok().content_type("application/json").body(
                json!({
                    "id": id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "message": message, "finish_reason": final_finish}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                })
                .to_string(),
            )
        }
        Ok(Err(e)) => error_response(e),
        Err(e) => error_response(internal_error(format!("inference task panicked: {e}"))),
    }
}

async fn run_chat_stream(
    state: web::Data<AppState>,
    params: InferenceParams,
    id: String,
    model_name: String,
    created: u64,
    has_tools: bool,
) -> HttpResponse {
    let (tx, rx) = mpsc::channel::<web::Bytes>(32);
    let id2 = id.clone();
    let model2 = model_name.clone();

    let permit = state.inference_semaphore.clone().acquire_owned().await;
    let state2 = state.clone();
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        const OBJ: &str = "chat.completion.chunk";

        // First chunk: role delta.
        let _ = tx.blocking_send(sse_chunk(&json!({
            "id": id2, "object": OBJ, "created": created, "model": model2,
            "choices": [{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]
        })));

        // Collect the whole output when tools are present so we can parse
        // tool calls before streaming; otherwise stream token-by-token.
        let mut finish_reason = FinishReason::Stop;

        if has_tools {
            // Buffered mode: collect, parse, then emit.
            let mut raw = String::new();
            if let Ok((_, fr)) = run_inference(&state2, &params, |piece| {
                raw.push_str(piece);
                true
            }) {
                finish_reason = fr;
            }

            let (content, tool_calls) = extract_tool_calls(&raw);

            if tool_calls.is_empty() {
                // No tool calls — stream content as a single delta.
                let _ = tx.blocking_send(sse_chunk(&json!({
                    "id": id2, "object": OBJ, "created": created, "model": model2,
                    "choices": [{"index":0,"delta":{"content":content},"finish_reason":null}]
                })));
                let _ = tx.blocking_send(sse_chunk(&json!({
                    "id": id2, "object": OBJ, "created": created, "model": model2,
                    "choices": [{"index":0,"delta":{},"finish_reason":finish_reason.as_str()}]
                })));
            } else {
                // Emit tool_calls delta.
                let calls_json: Vec<Value> = tool_calls.iter().map(|c| c.to_value()).collect();
                let content_val = if content.is_empty() {
                    Value::Null
                } else {
                    Value::String(content)
                };
                let _ = tx.blocking_send(sse_chunk(&json!({
                    "id": id2, "object": OBJ, "created": created, "model": model2,
                    "choices": [{"index":0,"delta":{"content":content_val,"tool_calls":calls_json},"finish_reason":null}]
                })));
                let _ = tx.blocking_send(sse_chunk(&json!({
                    "id": id2, "object": OBJ, "created": created, "model": model2,
                    "choices": [{"index":0,"delta":{},"finish_reason":"tool_calls"}]
                })));
            }
        } else {
            // Pure streaming: emit each token piece immediately.
            if let Ok((_, fr)) = run_inference(&state2, &params, |piece| {
                let ok = tx
                    .blocking_send(sse_chunk(&json!({
                        "id": id2, "object": OBJ, "created": created, "model": model2,
                        "choices": [{"index":0,"delta":{"content":piece},"finish_reason":null}]
                    })))
                    .is_ok();
                ok
            }) {
                finish_reason = fr;
            }
            let _ = tx.blocking_send(sse_chunk(&json!({
                "id": id2, "object": OBJ, "created": created, "model": model2,
                "choices": [{"index":0,"delta":{},"finish_reason":finish_reason.as_str()}]
            })));
        }

        let _ = tx.blocking_send(sse_done());
    });

    let body_stream = stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|chunk| (Ok::<_, actix_web::Error>(chunk), rx))
    });

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(body_stream)
}

// ---------------------------------------------------------------------------
// Raw completions  POST /v1/completions
// ---------------------------------------------------------------------------

async fn completions(
    req: HttpRequest,
    state: web::Data<AppState>,
    body: web::Bytes,
) -> HttpResponse {
    if let Some(err) = check_auth(&req, &state) {
        return error_response(err);
    }
    let text = match std::str::from_utf8(&body) {
        Ok(s) => s.to_owned(),
        Err(_) => return error_response(bad_request("body must be valid UTF-8")),
    };
    let parsed: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => return error_response(bad_request(format!("invalid JSON: {e}"))),
    };

    let prompt = match parsed.get("prompt") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => {
            // Array of strings → join (batch not yet supported, take first)
            match arr.first() {
                Some(Value::String(s)) => s.clone(),
                _ => return error_response(bad_request("'prompt' array must contain strings")),
            }
        }
        _ => return error_response(bad_request("'prompt' must be a string")),
    };

    let streaming = parsed
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let params = match InferenceParams::from_request(&parsed, prompt) {
        Ok(p) => p,
        Err(e) => return error_response(e),
    };

    let model_name = parsed
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&state.model_name)
        .to_owned();
    let created = now_secs();
    let id = format!("cmpl-{created}");

    if streaming {
        // Reuse chat stream logic with the "text_completion" object type
        // but emit `text` delta field instead of `content`.
        run_completion_stream(state, params, id, model_name, created).await
    } else {
        run_completion_blocking(state, params, id, model_name, created).await
    }
}

async fn run_completion_blocking(
    state: web::Data<AppState>,
    params: InferenceParams,
    id: String,
    model_name: String,
    created: u64,
) -> HttpResponse {
    let permit = state.inference_semaphore.clone().acquire_owned().await;
    let state2 = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut text = String::new();
        run_inference(&state2, &params, |piece| {
            text.push_str(piece);
            true
        })
        .map(|(tokens, reason)| (text, tokens, reason))
    })
    .await;

    match result {
        Ok(Ok((text, completion_tokens, finish_reason))) => {
            HttpResponse::Ok().content_type("application/json").body(
                json!({
                    "id": id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "text": text,
                        "finish_reason": finish_reason.as_str()
                    }],
                    "usage": {
                        "completion_tokens": completion_tokens
                    }
                })
                .to_string(),
            )
        }
        Ok(Err(e)) => error_response(e),
        Err(e) => error_response(internal_error(format!("inference task panicked: {e}"))),
    }
}

async fn run_completion_stream(
    state: web::Data<AppState>,
    params: InferenceParams,
    id: String,
    model_name: String,
    created: u64,
) -> HttpResponse {
    let (tx, rx) = mpsc::channel::<web::Bytes>(32);
    let id2 = id.clone();
    let model2 = model_name.clone();

    let permit = state.inference_semaphore.clone().acquire_owned().await;
    let state2 = state.clone();
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut finish_reason = FinishReason::Stop;
        let result = run_inference(&state2, &params, |piece| {
            let chunk = sse_chunk(&json!({
                "id": id2,
                "object": "text_completion",
                "created": created,
                "model": model2,
                "choices": [{"index": 0, "text": piece, "finish_reason": null}]
            }));
            tx.blocking_send(chunk).is_ok()
        });
        if let Ok((_, fr)) = result {
            finish_reason = fr;
        }
        let last = sse_chunk(&json!({
            "id": id2,
            "object": "text_completion",
            "created": created,
            "model": model2,
            "choices": [{"index": 0, "text": "", "finish_reason": finish_reason.as_str()}]
        }));
        let _ = tx.blocking_send(last);
        let _ = tx.blocking_send(sse_done());
    });

    let body_stream = stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|chunk| (Ok::<_, actix_web::Error>(chunk), rx))
    });

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(body_stream)
}

// ---------------------------------------------------------------------------
// Embeddings  POST /v1/embeddings
// ---------------------------------------------------------------------------

async fn embeddings(
    req: HttpRequest,
    state: web::Data<AppState>,
    body: web::Bytes,
) -> HttpResponse {
    if let Some(err) = check_auth(&req, &state) {
        return error_response(err);
    }
    let text = match std::str::from_utf8(&body) {
        Ok(s) => s.to_owned(),
        Err(_) => return error_response(bad_request("body must be valid UTF-8")),
    };
    let parsed: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => return error_response(bad_request(format!("invalid JSON: {e}"))),
    };

    // `input` may be a string or an array of strings.
    let inputs: Vec<String> = match parsed.get("input") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(arr)) => {
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                match v {
                    Value::String(s) => out.push(s.clone()),
                    _ => return error_response(bad_request("'input' array must contain strings")),
                }
            }
            out
        }
        _ => return error_response(bad_request("'input' must be a string or array of strings")),
    };

    let model_name = parsed
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&state.model_name)
        .to_owned();

    let permit = state.inference_semaphore.clone().acquire_owned().await;
    let state2 = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        // Return (vectors, total_prompt_tokens) together so `inputs` doesn't
        // need to be borrowed after the move.
        let total_tokens: u32 = inputs
            .iter()
            .filter_map(|s| state2.model.str_to_token(s, AddBos::Always).ok())
            .map(|t| t.len() as u32)
            .sum();
        embed_inputs(&state2, &inputs).map(|vecs| (vecs, total_tokens))
    })
    .await;

    match result {
        Ok(Ok((vectors, total_tokens))) => {
            let data: Vec<Value> = vectors
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    json!({
                        "object": "embedding",
                        "index": i,
                        "embedding": v
                    })
                })
                .collect();
            HttpResponse::Ok().content_type("application/json").body(
                json!({
                    "object": "list",
                    "model": model_name,
                    "data": data,
                    "usage": { "prompt_tokens": total_tokens, "total_tokens": total_tokens }
                })
                .to_string(),
            )
        }
        Ok(Err(e)) => error_response(e),
        Err(e) => error_response(internal_error(format!("embed task panicked: {e}"))),
    }
}

fn embed_inputs(state: &AppState, inputs: &[String]) -> Result<Vec<Vec<f32>>, HttpError> {
    let n_embd = state.model.n_embd() as usize;
    let mut results = Vec::with_capacity(inputs.len());

    for input in inputs {
        let tokens = state
            .model
            .str_to_token(input, AddBos::Always)
            .map_err(|e| internal_error(format!("tokenise: {e}")))?;

        let n_ctx = (tokens.len() as u32 + 16).max(64);
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx)
            .with_embeddings(true);

        let mut ctx = state
            .model
            .new_context(&state.backend, ctx_params)
            .map_err(|e| internal_error(format!("context init: {e}")))?;

        let mut batch = LlamaBatch::new(n_ctx as usize, 1);
        let last = tokens.len().saturating_sub(1) as i32;
        for (i, &tok) in tokens.iter().enumerate() {
            batch
                .add(tok, i as i32, &[0], i as i32 == last)
                .map_err(|e| internal_error(format!("batch add: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| internal_error(format!("decode: {e}")))?;

        // Try sequence-level pooled embedding first, fall back to last-token.
        let vec = if let Ok(emb) = ctx.embeddings_seq_ith(0) {
            emb.to_vec()
        } else if let Ok(emb) = ctx.embeddings_ith(last) {
            emb.to_vec()
        } else {
            vec![0.0f32; n_embd]
        };

        // L2-normalise.
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        results.push(vec.into_iter().map(|x| x / norm).collect());
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// File store handlers  POST/GET/DELETE /v1/files
// ---------------------------------------------------------------------------

/// `POST /v1/files`  — upload a file (multipart/form-data with `file` + `purpose`).
async fn upload_file(
    req: HttpRequest,
    state: web::Data<AppState>,
    mut payload: Multipart,
) -> HttpResponse {
    if let Some(err) = check_auth(&req, &state) {
        return error_response(err);
    }

    let mut file_bytes: Option<Vec<u8>> = None;
    let mut filename = "upload".to_owned();
    let mut purpose = "assistants".to_owned();

    while let Some(item) = payload.next().await {
        let mut field = match item {
            Ok(f) => f,
            Err(e) => return error_response(bad_request(format!("multipart error: {e}"))),
        };

        // Read metadata (returns a borrow; we convert to owned before streaming).
        let field_name = field
            .content_disposition()
            .and_then(|cd| cd.get_name())
            .unwrap_or("")
            .to_owned();
        let field_filename = field
            .content_disposition()
            .and_then(|cd| cd.get_filename())
            .map(str::to_owned);

        let mut data: Vec<u8> = Vec::new();
        while let Some(chunk) = field.next().await {
            match chunk {
                Ok(bytes) => data.extend_from_slice(&bytes),
                Err(e) => return error_response(internal_error(format!("chunk read error: {e}"))),
            }
        }

        match field_name.as_str() {
            "file" => {
                filename = field_filename.unwrap_or_else(|| "upload".to_owned());
                file_bytes = Some(data);
            }
            "purpose" => {
                purpose = String::from_utf8_lossy(&data).into_owned();
            }
            _ => {}
        }
    }

    let bytes = match file_bytes {
        Some(b) => b,
        None => {
            return error_response(bad_request(
                "'file' field is required (multipart/form-data)",
            ))
        }
    };

    let id = gen_file_id(&bytes);
    let size = bytes.len();
    let created_at = now_secs();

    state.file_store.write().await.insert(
        id.clone(),
        FileEntry {
            id: id.clone(),
            filename: filename.clone(),
            bytes,
            purpose: purpose.clone(),
            created_at,
        },
    );

    tracing::info!("Stored file {id} ({size} bytes, purpose={purpose})");

    HttpResponse::Ok().content_type("application/json").body(
        json!({
            "id": id,
            "object": "file",
            "bytes": size,
            "created_at": created_at,
            "filename": filename,
            "purpose": purpose,
            "status": "processed",
            "status_details": null
        })
        .to_string(),
    )
}

/// `GET /v1/files` — list all uploaded files.
async fn list_files(req: HttpRequest, state: web::Data<AppState>) -> HttpResponse {
    if let Some(err) = check_auth(&req, &state) {
        return error_response(err);
    }
    let store = state.file_store.read().await;
    let data: Vec<Value> = store
        .values()
        .map(|e| {
            json!({
                "id": e.id,
                "object": "file",
                "bytes": e.bytes.len(),
                "created_at": e.created_at,
                "filename": e.filename,
                "purpose": e.purpose,
            })
        })
        .collect();

    HttpResponse::Ok()
        .content_type("application/json")
        .body(json!({"object": "list", "data": data}).to_string())
}

/// `GET /v1/files/{file_id}` — retrieve file metadata.
async fn get_file(
    req: HttpRequest,
    state: web::Data<AppState>,
    path: web::Path<String>,
) -> HttpResponse {
    if let Some(err) = check_auth(&req, &state) {
        return error_response(err);
    }
    let id = path.into_inner();
    let store = state.file_store.read().await;
    match store.get(&id) {
        Some(e) => HttpResponse::Ok().content_type("application/json").body(
            json!({
                "id": e.id,
                "object": "file",
                "bytes": e.bytes.len(),
                "created_at": e.created_at,
                "filename": e.filename,
                "purpose": e.purpose,
            })
            .to_string(),
        ),
        None => error_response(HttpError {
            status: StatusCode::NOT_FOUND,
            r#type: "invalid_request_error",
            message: format!("No file with id '{id}'"),
        }),
    }
}

/// `GET /v1/files/{file_id}/content` — download raw file bytes.
async fn get_file_content(
    req: HttpRequest,
    state: web::Data<AppState>,
    path: web::Path<String>,
) -> HttpResponse {
    if let Some(err) = check_auth(&req, &state) {
        return error_response(err);
    }
    let id = path.into_inner();
    let store = state.file_store.read().await;
    match store.get(&id) {
        Some(e) => HttpResponse::Ok()
            .content_type("application/octet-stream")
            .body(e.bytes.clone()),
        None => error_response(HttpError {
            status: StatusCode::NOT_FOUND,
            r#type: "invalid_request_error",
            message: format!("No file with id '{id}'"),
        }),
    }
}

/// `DELETE /v1/files/{file_id}` — delete an uploaded file.
async fn delete_file(
    req: HttpRequest,
    state: web::Data<AppState>,
    path: web::Path<String>,
) -> HttpResponse {
    if let Some(err) = check_auth(&req, &state) {
        return error_response(err);
    }
    let id = path.into_inner();
    let removed = state.file_store.write().await.remove(&id).is_some();
    if removed {
        HttpResponse::Ok()
            .content_type("application/json")
            .body(json!({"id": id, "object": "file", "deleted": true}).to_string())
    } else {
        error_response(HttpError {
            status: StatusCode::NOT_FOUND,
            r#type: "invalid_request_error",
            message: format!("No file with id '{id}'"),
        })
    }
}

// ---------------------------------------------------------------------------
// Simple handlers
// ---------------------------------------------------------------------------

async fn list_models(req: HttpRequest, state: web::Data<AppState>) -> HttpResponse {
    if let Some(err) = check_auth(&req, &state) {
        return error_response(err);
    }
    let n_ctx = state
        .default_ctx_size
        .map_or(state.model.n_ctx_train(), NonZeroU32::get);
    HttpResponse::Ok().content_type("application/json").body(
        json!({
            "object": "list",
            "data": [{
                "id": state.model_name,
                "object": "model",
                "created": now_secs(),
                "owned_by": "llama.cpp",
                "context_length": n_ctx,
                "embedding_length": state.model.n_embd()
            }]
        })
        .to_string(),
    )
}

async fn health() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("application/json")
        .body(r#"{"status":"ok"}"#)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    // Capture the HF repo ID before `args.model` is consumed by `resolve()`.
    // Used later to auto-download the matching mmproj from the same repo.
    #[cfg(feature = "mtmd")]
    let hf_repo: Option<String> = match &args.model {
        ModelSource::HuggingFace { repo, .. } => Some(repo.clone()),
        ModelSource::Local { .. } => None,
    };

    let model_path = args
        .model
        .resolve()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;

    // --print-path: output the resolved path and exit immediately.
    if args.print_path {
        println!("{}", model_path.display());
        return Ok(());
    }

    let model_name = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("llama.cpp")
        .to_string();

    let backend = LlamaBackend::init()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    let mut model_params = LlamaModelParams::default();
    if args.n_gpu_layers > 0 {
        model_params = model_params.with_n_gpu_layers(args.n_gpu_layers);
    }

    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    let chat_template = model.get_chat_template(65536).ok();
    if chat_template.is_some() {
        tracing::info!("Loaded built-in chat template from model");
    } else {
        tracing::warn!("No built-in chat template — supply 'chat_template' per request");
    }

    let parallel = args.parallel.max(1);
    if args.api_key.is_some() {
        tracing::info!("API key authentication enabled");
    }

    // ── Multimodal projector (optional) ───────────────────────────────────────
    #[cfg(feature = "mtmd")]
    let mtmd_ctx: Option<MtmdContext> = {
        tracing::info!("Model resolved to: {}", model_path.display());
        let model_dir = model_path.parent().unwrap_or(std::path::Path::new("."));
        tracing::info!("Scanning for mmproj in: {}", model_dir.display());

        // Resolve the mmproj path:
        //  1. --mmproj given as an absolute/relative path → use as-is.
        //  2. --mmproj given as a bare filename (no directory component) →
        //     look for it next to the model file.
        //  3. --mmproj not given → scan the model's directory for any
        //     `mmproj-*.gguf` and pick the best one automatically.
        let mmproj_path: Option<PathBuf> = match &args.mmproj {
            Some(p)
                if p.components().count() == 1 && p.parent() == Some(std::path::Path::new("")) =>
            {
                // bare filename — resolve relative to model directory
                let candidate = model_path
                    .parent()
                    .map(|d| d.join(p))
                    .filter(|f| f.exists());
                if candidate.is_none() {
                    tracing::warn!(
                        "mmproj '{}' not found next to model ({}); skipping multimodal",
                        p.display(),
                        model_dir.display()
                    );
                }
                candidate
            }
            Some(p) => Some(p.clone()),
            None => {
                // 1. Scan the local cache directory (fast, no network).
                find_mmproj_in_dir(model_dir)
                    // 2. Fall back: download from the same HuggingFace repo.
                    .or_else(|| hf_repo.as_deref().and_then(download_mmproj_from_hf))
            }
        };

        if let Some(ref p) = mmproj_path {
            tracing::info!("Loading mmproj: {}", p.display());
            let ctx_params = MtmdContextParams::default()
                .use_gpu(!args.no_mmproj_gpu)
                .n_threads(args.mmproj_n_threads);
            match MtmdContext::init_from_file(p, &model, ctx_params) {
                Ok(ctx) => {
                    tracing::info!(
                        "  vision={} audio={}",
                        ctx.supports_vision(),
                        ctx.supports_audio()
                    );
                    Some(ctx)
                }
                Err(e) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("failed to load mmproj '{}': {e}", p.display()),
                    ))
                }
            }
        } else {
            None
        }
    };

    #[cfg(not(feature = "mtmd"))]
    if args.mmproj.is_some() {
        tracing::warn!(
            "--mmproj was provided but this binary was compiled without the `mtmd` feature. \
             Rebuild with `--features mtmd` to enable multimodal support."
        );
    }

    let state = web::Data::new(AppState {
        backend,
        model,
        chat_template,
        model_name,
        default_ctx_size: args.ctx_size,
        inference_semaphore: Arc::new(Semaphore::new(parallel)),
        api_key: args.api_key,
        file_store: Arc::new(RwLock::new(HashMap::new())),
        #[cfg(feature = "mtmd")]
        mtmd_ctx,
    });

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Listening on http://{addr}  (parallel={parallel})");
    tracing::info!("Endpoints:");
    tracing::info!("  GET    /health");
    tracing::info!("  GET    /v1/models");
    tracing::info!("  POST   /v1/chat/completions  (streaming supported)");
    tracing::info!("  POST   /v1/completions       (streaming supported)");
    tracing::info!("  POST   /v1/embeddings");
    tracing::info!("  POST   /v1/files             (upload image/audio for multimodal)");
    tracing::info!("  GET    /v1/files             (list uploaded files)");
    tracing::info!("  GET    /v1/files/{{id}}        (file metadata)");
    tracing::info!("  GET    /v1/files/{{id}}/content (download file)");
    tracing::info!("  DELETE /v1/files/{{id}}        (delete file)");

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .app_data(web::JsonConfig::default().error_handler(|err, _req| {
                let msg = format!("JSON parse error: {err}");
                actix_web::error::InternalError::from_response(
                    err,
                    error_response(bad_request(msg)),
                )
                .into()
            }))
            .route("/health", web::get().to(health))
            .route("/v1/models", web::get().to(list_models))
            .route("/v1/chat/completions", web::post().to(chat_completions))
            .route("/v1/completions", web::post().to(completions))
            .route("/v1/embeddings", web::post().to(embeddings))
            // File store
            .route("/v1/files", web::post().to(upload_file))
            .route("/v1/files", web::get().to(list_files))
            .route("/v1/files/{file_id}", web::get().to(get_file))
            .route(
                "/v1/files/{file_id}/content",
                web::get().to(get_file_content),
            )
            .route("/v1/files/{file_id}", web::delete().to(delete_file))
    })
    .bind(&addr)?
    .run()
    .await
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── collect_groups ───────────────────────────────────────────────────────

    #[test]
    fn single_plain_gguf() {
        let files = vec!["model.Q4_K_M.gguf".to_string()];
        let groups = collect_groups(files);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].files.len(), 1);
    }

    #[test]
    fn sharded_flat_files_grouped() {
        let files = vec![
            "model-Q4_K_M-00001-of-00003.gguf".to_string(),
            "model-Q4_K_M-00002-of-00003.gguf".to_string(),
            "model-Q4_K_M-00003-of-00003.gguf".to_string(),
        ];
        let groups = collect_groups(files);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].files.len(), 3);
        assert_eq!(groups[0].files[0], "model-Q4_K_M-00001-of-00003.gguf");
    }

    #[test]
    fn subdirectory_files_grouped_by_dir() {
        let files = vec![
            "Q4_K_M/model-00001-of-00006.gguf".to_string(),
            "Q4_K_M/model-00002-of-00006.gguf".to_string(),
            "Q3_K_M/model-00001-of-00005.gguf".to_string(),
            "Q3_K_M/model-00002-of-00005.gguf".to_string(),
        ];
        let groups = collect_groups(files);
        assert_eq!(groups.len(), 2);
        // BTreeMap orders alphabetically: Q3 before Q4
        assert_eq!(groups[0].label, "Q3_K_M  [2 shards]");
        assert_eq!(groups[1].label, "Q4_K_M  [2 shards]");
    }

    #[test]
    fn mixed_quants_each_get_own_group() {
        let files = vec![
            "llama-Q4_K_M.gguf".to_string(),
            "llama-Q8_0.gguf".to_string(),
        ];
        let groups = collect_groups(files);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn preference_score_orders_correctly() {
        let files = vec![
            "Q8_0/model.gguf".to_string(),
            "Q4_K_M/model.gguf".to_string(),
            "Q3_K_S/model.gguf".to_string(),
        ];
        let groups = collect_groups(files);
        let mut scores: Vec<_> = groups
            .iter()
            .map(|g| (g.preference_score(), &g.label))
            .collect();
        scores.sort();
        // Q4_K_M should have the lowest (best) score
        assert!(scores[0].1.contains("Q4_K_M"), "got {scores:?}");
    }
}
