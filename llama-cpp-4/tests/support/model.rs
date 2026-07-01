//! Locate and load GGUF files for integration tests.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::model::params::LlamaModelParams;
use llama_cpp_4::model::LlamaModel;
use llama_cpp_4::LLamaCppError;

/// Serialize tests that call `llama_decode` (not safe across parallel contexts).
pub static DECODE_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the decode lock, recovering from a poisoned mutex after a prior test panic.
pub fn decode_guard() -> std::sync::MutexGuard<'static, ()> {
    DECODE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// Default tiny model used in CI (`ggml-org/models` / llama.cpp server tests).
pub const DEFAULT_TEST_MODEL: &str = "stories260K.gguf";

/// Process-wide backend initialisation (llama.cpp allows one init per process).
pub fn backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| match LlamaBackend::init() {
        Ok(b) => b,
        Err(LLamaCppError::BackendAlreadyInitialized) => LlamaBackend {},
        Err(e) => panic!("backend init failed: {e}"),
    })
}

/// Resolved on-disk model file.
#[derive(Debug, Clone)]
pub struct ModelFixture {
    pub path: PathBuf,
    /// `true` when only tokenizer weights are present (no inference).
    pub vocab_only: bool,
}

/// Find a GGUF for tests. See module docs for search order.
pub fn find_test_model() -> Option<ModelFixture> {
    if let Ok(path) = std::env::var("LLAMA_TEST_MODEL") {
        let p = PathBuf::from(path);
        if p.is_file() {
            return Some(ModelFixture {
                path: p,
                vocab_only: false,
            });
        }
        eprintln!("LLAMA_TEST_MODEL is set but not a file: {}", p.display());
    }

    if let Some(path) = default_cached_test_model() {
        if path.is_file() {
            return Some(ModelFixture {
                path,
                vocab_only: false,
            });
        }
    }

    vocab_only_fixture()
}

fn default_cached_test_model() -> Option<PathBuf> {
    // `cargo test` runs with CWD = package dir (`llama-cpp-4/`).
    let path = PathBuf::from("../target/test-models").join(DEFAULT_TEST_MODEL);
    if path.is_file() {
        return Some(path);
    }
    // Workspace root when invoked from repo root.
    let path = PathBuf::from("target/test-models").join(DEFAULT_TEST_MODEL);
    path.is_file().then_some(path)
}

fn vocab_only_fixture() -> Option<ModelFixture> {
    let build_dir = PathBuf::from("target/debug/build");
    let entries = std::fs::read_dir(&build_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if name
            .to_str()
            .is_some_and(|n| n.starts_with("llama-cpp-sys-4-"))
        {
            let vocab_path = entry
                .path()
                .join("out/llama.cpp/models/ggml-vocab-llama-bpe.gguf");
            if vocab_path.is_file() {
                return Some(ModelFixture {
                    path: vocab_path,
                    vocab_only: true,
                });
            }
        }
    }
    None
}

/// Load a model for tests. Returns `(model, vocab_only)`.
pub fn load_model() -> Option<(LlamaModel, bool)> {
    let fixture = find_test_model()?;
    let mut params = LlamaModelParams::default();
    if fixture.vocab_only {
        params = params.with_vocab_only(true);
    }
    let params = std::pin::pin!(params);
    let model = LlamaModel::load_from_file(backend(), &fixture.path, &params).ok()?;
    Some((model, fixture.vocab_only))
}

/// Load a full (non-vocab-only) checkpoint, or `None` when unavailable.
pub fn load_full_model() -> Option<LlamaModel> {
    let (model, vocab_only) = load_model()?;
    if vocab_only {
        eprintln!(
            "SKIP: full model required (set LLAMA_TEST_MODEL or run scripts/fetch-test-model.sh)"
        );
        return None;
    }
    Some(model)
}

/// Path to the loaded test model, if any.
pub fn test_model_path() -> Option<PathBuf> {
    find_test_model().map(|f| f.path)
}

/// Skip helper — prints guidance when no model is available.
pub fn skip_no_model() {
    eprintln!("SKIP: no test model available");
    eprintln!("  export LLAMA_TEST_MODEL=/path/to/model.gguf");
    eprintln!("  ./scripts/fetch-test-model.sh");
}
