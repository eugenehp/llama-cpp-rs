//! Speculative decoding with no draft model.
//!
//! [`Eagle3Session`](crate::eagle::Eagle3Session) and
//! [`MtpSession`](crate::mtp::MtpSession) draft with a second model. These
//! draft by *looking up what came before*: find where the recent token history
//! repeats, and propose whatever followed it last time. That costs one hash
//! lookup per step instead of a forward pass, needs no extra weights and no
//! extra VRAM, and wins big on the workloads where text repeats — code
//! editing, RAG over a quoted document, JSON with recurring keys, chat that
//! restates the question.
//!
//! Three strategies, in increasing order of memory and payoff:
//!
//! | | Learns from | State |
//! |---|---|---|
//! | [`ngram_simple_draft`] | the current context only | none |
//! | [`NgramMap`] | the current context, adapting to how its drafts land | in memory |
//! | [`NgramCache`] | a corpus and/or past generations | in memory, saveable |
//!
//! All three return a *draft*: candidate tokens to verify against the target
//! model, typically via
//! [`CommonSampler::sample_and_accept_n`](crate::common_sampler::CommonSampler::sample_and_accept_n).
//! A wrong draft costs only the tokens it wasted.

use std::ffi::CString;
use std::ptr::NonNull;

use llama_cpp_sys_4 as sys;

use crate::shim::{check_status, last_error, read_tokens, ShimError};
use crate::token::LlamaToken;

/// Errors from the n-gram drafters.
pub type NgramError = ShimError;

type Result<T> = std::result::Result<T, NgramError>;

/// Draft by finding the most recent repeat of the trailing n-gram.
///
/// Looks for the last `size_ngram` tokens elsewhere in `tokens` and proposes
/// the `size_mgram` tokens that followed it. Entirely stateless — the whole
/// strategy is "this text repeated once, it may repeat again".
///
/// `tokens` is the history **excluding** `sampled`: upstream builds its search
/// pattern from the tail of `tokens` plus `sampled`, so passing a history that
/// already ends in `sampled` searches for the wrong thing.
///
/// Returns an empty draft when nothing matches, which is the common case for
/// prose and the reason this is nearly free. Also empty when the history is
/// shorter than `size_ngram + size_mgram + 1`, or when the only match sits at
/// position 0 — upstream treats index 0 as "no match".
///
/// # Errors
///
/// Returns [`NgramError::Failed`] if llama.cpp throws.
// The two parameter names mirror `common_ngram_simple_config`; renaming either
// to satisfy `similar_names` would obscure which upstream field it sets.
#[allow(clippy::similar_names)]
pub fn ngram_simple_draft(
    size_ngram: u16,
    size_mgram: u16,
    tokens: &[LlamaToken],
    sampled: LlamaToken,
) -> Result<Vec<LlamaToken>> {
    let raw: Vec<i32> = tokens.iter().map(|t| t.0).collect();
    read_tokens(|out, cap, len| unsafe {
        sys::common_shim_ngram_simple_draft(
            size_ngram,
            size_mgram,
            raw.as_ptr(),
            raw.len(),
            sampled.0,
            out,
            cap,
            len,
        )
    })
}

/// A statistical n-gram cache: which tokens tend to follow which n-grams.
///
/// Where [`ngram_simple_draft`] takes the single most recent repeat,
/// this accumulates a distribution over many observations and drafts the most
/// likely continuation. It can be persisted, so a cache built once from a
/// corpus — or grown across a user's sessions — keeps paying off.
///
/// llama.cpp consults up to three caches at once, in priority order:
///
/// * **context** — built from the current conversation, most specific;
/// * **dynamic** — built from this user's past generations;
/// * **static** — built offline from a large corpus, used to validate.
///
/// Wraps `common_ngram_cache`.
pub struct NgramCache {
    raw: NonNull<sys::common_shim_ngram_cache>,
}

impl std::fmt::Debug for NgramCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NgramCache").field("len", &self.len()).finish()
    }
}

// SAFETY: the handle owns an `unordered_map` with no shared state; mutating
// methods take `&mut self`.
unsafe impl Send for NgramCache {}

impl Drop for NgramCache {
    fn drop(&mut self) {
        unsafe { sys::common_shim_ngram_cache_free(self.raw.as_ptr()) }
    }
}

impl Default for NgramCache {
    fn default() -> Self {
        Self::new()
    }
}

impl NgramCache {
    /// An empty cache.
    ///
    /// # Panics
    ///
    /// Panics if the allocation fails.
    #[must_use]
    pub fn new() -> Self {
        let raw = unsafe { sys::common_shim_ngram_cache_init() };
        Self {
            raw: NonNull::new(raw).expect("common_shim_ngram_cache_init returned null"),
        }
    }

    /// Load a cache written by [`Self::save`].
    ///
    /// # Errors
    ///
    /// Returns [`NgramError::Failed`] if the file is missing or malformed, or
    /// [`NgramError::Nul`] for an interior NUL in `path`.
    pub fn load(path: &str) -> Result<Self> {
        let c_path = CString::new(path)?;
        let raw = unsafe { sys::common_shim_ngram_cache_load(c_path.as_ptr()) };
        NonNull::new(raw)
            .map(|raw| Self { raw })
            .ok_or_else(|| NgramError::Failed(last_error()))
    }

    /// Write this cache to disk.
    ///
    /// # Errors
    ///
    /// Returns [`NgramError::Failed`] if the file cannot be written, or
    /// [`NgramError::Nul`] for an interior NUL in `path`.
    pub fn save(&mut self, path: &str) -> Result<()> {
        let c_path = CString::new(path)?;
        let status =
            unsafe { sys::common_shim_ngram_cache_save(self.raw.as_ptr(), c_path.as_ptr()) };
        check_status(status)
    }

    /// Fold another cache's counts into this one.
    ///
    /// # Errors
    ///
    /// Returns [`NgramError::Failed`] if llama.cpp throws.
    pub fn merge(&mut self, other: &mut NgramCache) -> Result<()> {
        let status =
            unsafe { sys::common_shim_ngram_cache_merge(self.raw.as_ptr(), other.raw.as_ptr()) };
        check_status(status)
    }

    /// Number of distinct n-grams recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        unsafe { sys::common_shim_ngram_cache_size(self.raw.as_ptr()) }
    }

    /// Whether the cache has learned anything yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Learn from a token sequence.
    ///
    /// `nnew` is how many tokens were appended since the last call, so a live
    /// conversation only pays for its new tokens. Upstream requires `tokens` to
    /// be **append-only**: editing the middle invalidates the statistics and
    /// needs a rebuild from scratch.
    ///
    /// # Errors
    ///
    /// Returns [`NgramError::Failed`] if llama.cpp throws.
    pub fn update(
        &mut self,
        ngram_min: i32,
        ngram_max: i32,
        tokens: &[LlamaToken],
        nnew: i32,
        print_progress: bool,
    ) -> Result<()> {
        let raw: Vec<i32> = tokens.iter().map(|t| t.0).collect();
        let status = unsafe {
            sys::common_shim_ngram_cache_update(
                self.raw.as_ptr(),
                ngram_min,
                ngram_max,
                raw.as_ptr(),
                raw.len(),
                nnew,
                print_progress,
            )
        };
        check_status(status)
    }
}

/// Draft a continuation from up to three caches.
///
/// Any cache may be `None`. `tokens` must be non-empty — the last token seeds
/// the lookup — and the returned draft excludes it.
///
/// # Errors
///
/// Returns [`NgramError::InvalidArg`] for empty `tokens`, or
/// [`NgramError::Failed`] if llama.cpp throws.
pub fn ngram_cache_draft(
    tokens: &[LlamaToken],
    n_draft: i32,
    ngram_min: i32,
    ngram_max: i32,
    context: Option<&mut NgramCache>,
    dynamic: Option<&mut NgramCache>,
    statik: Option<&mut NgramCache>,
) -> Result<Vec<LlamaToken>> {
    if tokens.is_empty() {
        return Err(NgramError::InvalidArg);
    }
    let raw: Vec<i32> = tokens.iter().map(|t| t.0).collect();
    let ctx_ptr = context.map_or(std::ptr::null_mut(), |c| c.raw.as_ptr());
    let dyn_ptr = dynamic.map_or(std::ptr::null_mut(), |c| c.raw.as_ptr());
    let sta_ptr = statik.map_or(std::ptr::null_mut(), |c| c.raw.as_ptr());

    read_tokens(|out, cap, len| unsafe {
        sys::common_shim_ngram_cache_draft(
            raw.as_ptr(),
            raw.len(),
            n_draft,
            ngram_min,
            ngram_max,
            ctx_ptr,
            dyn_ptr,
            sta_ptr,
            out,
            cap,
            len,
        )
    })
}

/// An adaptive in-context n-gram drafter.
///
/// Like [`NgramCache`] it indexes repeated n-grams, but it also records how
/// many tokens each of its own drafts got accepted and uses that to decide
/// whether to draft again — so a context where lookup is not paying off stops
/// costing anything. Feed results back with [`Self::accept`].
///
/// State lives only in memory and only for this generation; call
/// [`Self::begin`] when starting a new one.
///
/// Wraps `common_ngram_map`.
pub struct NgramMap {
    raw: NonNull<sys::common_shim_ngram_map>,
}

impl std::fmt::Debug for NgramMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NgramMap").finish_non_exhaustive()
    }
}

// SAFETY: the handle owns a `common_ngram_map` with no shared state.
unsafe impl Send for NgramMap {}

impl Drop for NgramMap {
    fn drop(&mut self) {
        unsafe { sys::common_shim_ngram_map_free(self.raw.as_ptr()) }
    }
}

impl NgramMap {
    /// Build a map.
    ///
    /// * `size_key` — length of the n-grams used as lookup keys.
    /// * `size_value` — length of the continuations drafted.
    /// * `key_only` — index keys without tracking continuations, which is
    ///   cheaper but drafts nothing on its own.
    /// * `min_hits` — how many times a key must recur before it is trusted.
    ///
    /// # Errors
    ///
    /// Returns [`NgramError::Failed`] if llama.cpp cannot allocate the map —
    /// it reserves a 2^18-entry hash table up front.
    pub fn new(size_key: u16, size_value: u16, key_only: bool, min_hits: u16) -> Result<Self> {
        let raw =
            unsafe { sys::common_shim_ngram_map_init(size_key, size_value, key_only, min_hits) };
        NonNull::new(raw)
            .map(|raw| Self { raw })
            .ok_or_else(|| NgramError::Failed(last_error()))
    }

    /// Start a generation over `tokens` (the prompt).
    ///
    /// # Errors
    ///
    /// Returns [`NgramError::Failed`] if llama.cpp throws.
    pub fn begin(&mut self, tokens: &[LlamaToken]) -> Result<()> {
        let raw: Vec<i32> = tokens.iter().map(|t| t.0).collect();
        let status = unsafe {
            sys::common_shim_ngram_map_begin(self.raw.as_ptr(), raw.as_ptr(), raw.len())
        };
        check_status(status)
    }

    /// Draft a continuation, given everything generated so far and the token
    /// just sampled.
    ///
    /// Returns an empty draft when the map decides lookup is not worth it here.
    ///
    /// # Errors
    ///
    /// Returns [`NgramError::Failed`] if llama.cpp throws.
    pub fn draft(&mut self, tokens: &[LlamaToken], sampled: LlamaToken) -> Result<Vec<LlamaToken>> {
        let raw: Vec<i32> = tokens.iter().map(|t| t.0).collect();
        read_tokens(|out, cap, len| unsafe {
            sys::common_shim_ngram_map_draft(
                self.raw.as_ptr(),
                raw.as_ptr(),
                raw.len(),
                sampled.0,
                out,
                cap,
                len,
            )
        })
    }

    /// Report how many of the last draft's tokens the target model accepted.
    ///
    /// This is what makes the map adaptive; skipping it leaves it drafting
    /// blind.
    pub fn accept(&mut self, n_accepted: u16) {
        unsafe { sys::common_shim_ngram_map_accept(self.raw.as_ptr(), n_accepted) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(v: &[i32]) -> Vec<LlamaToken> {
        v.iter().copied().map(LlamaToken).collect()
    }

    /// The whole premise: text that repeats should be drafted from its earlier
    /// occurrence.
    ///
    /// `sampled` is deliberately *not* in `history` — upstream appends it to
    /// build the search pattern. The leading 9 keeps the earlier `1 2` off
    /// index 0, which upstream treats as "no match".
    #[test]
    fn simple_draft_predicts_a_repeat() {
        let history = toks(&[9, 1, 2, 3, 4, 5, 1]);
        let draft = ngram_simple_draft(2, 2, &history, LlamaToken(2)).unwrap();
        assert_eq!(
            draft,
            toks(&[3, 4]),
            "expected the continuation of the earlier `1 2`"
        );
    }

    /// Passing a history that already ends in `sampled` searches for
    /// `[sampled, sampled]`, which is the mistake the doc warns about.
    #[test]
    fn simple_draft_with_sampled_already_in_history_finds_nothing() {
        let history = toks(&[9, 1, 2, 3, 4, 5, 1, 2]);
        let draft = ngram_simple_draft(2, 2, &history, LlamaToken(2)).unwrap();
        assert!(draft.is_empty(), "got {draft:?}");
    }

    /// Upstream needs more than `size_ngram + size_mgram + 1` tokens before it
    /// will look at all.
    #[test]
    fn simple_draft_is_empty_below_the_length_floor() {
        let history = toks(&[1, 2, 1, 2, 1]);
        assert!(ngram_simple_draft(2, 2, &history, LlamaToken(2))
            .unwrap()
            .is_empty());
    }

    /// Non-repeating history must draft nothing rather than guess — a wrong
    /// draft costs a verification pass.
    #[test]
    fn simple_draft_is_empty_without_a_repeat() {
        let history = toks(&[1, 2, 3, 4, 5]);
        let draft = ngram_simple_draft(2, 2, &history, LlamaToken(5)).unwrap();
        assert!(draft.is_empty(), "expected no draft, got {draft:?}");
    }

    #[test]
    fn simple_draft_handles_empty_history() {
        assert!(ngram_simple_draft(2, 2, &[], LlamaToken(1)).unwrap().is_empty());
    }

    #[test]
    fn cache_starts_empty_and_learns() {
        let mut cache = NgramCache::new();
        assert!(cache.is_empty());

        let tokens = toks(&[1, 2, 3, 1, 2, 3, 1, 2, 3]);
        cache
            .update(1, 4, &tokens, i32::try_from(tokens.len()).unwrap(), false)
            .unwrap();
        assert!(!cache.is_empty(), "update recorded nothing");
    }

    /// A cache must survive a save/load round trip, since the point of it is
    /// being reusable across sessions.
    #[test]
    fn cache_round_trips_through_disk() {
        let dir = std::env::temp_dir();
        let path = dir.join("llama_cpp_rs_ngram_test.bin");
        let path_str = path.to_str().unwrap();

        let mut cache = NgramCache::new();
        let tokens = toks(&[7, 8, 9, 7, 8, 9, 7, 8, 9]);
        cache
            .update(1, 4, &tokens, i32::try_from(tokens.len()).unwrap(), false)
            .unwrap();
        let saved_len = cache.len();
        cache.save(path_str).unwrap();

        let loaded = NgramCache::load(path_str).unwrap();
        assert_eq!(loaded.len(), saved_len, "cache changed size across disk");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn cache_load_rejects_a_missing_file() {
        assert!(NgramCache::load("/definitely/not/a/cache.bin").is_err());
    }

    #[test]
    fn cache_rejects_interior_nul_in_path() {
        let mut cache = NgramCache::new();
        assert!(matches!(cache.save("a\0b"), Err(NgramError::Nul(_))));
        assert!(matches!(NgramCache::load("a\0b"), Err(NgramError::Nul(_))));
    }

    /// Merging must be additive, not replacing — two caches over the same text
    /// should not shrink the result.
    #[test]
    fn cache_merge_is_additive() {
        let tokens = toks(&[4, 5, 6, 4, 5, 6, 4, 5, 6]);
        let mut a = NgramCache::new();
        a.update(1, 4, &tokens, i32::try_from(tokens.len()).unwrap(), false).unwrap();
        let before = a.len();

        let mut b = NgramCache::new();
        b.update(1, 4, &tokens, i32::try_from(tokens.len()).unwrap(), false).unwrap();

        a.merge(&mut b).unwrap();
        assert!(a.len() >= before, "merge lost entries");
    }

    #[test]
    fn cache_draft_requires_tokens() {
        assert!(matches!(
            ngram_cache_draft(&[], 4, 1, 4, None, None, None),
            Err(NgramError::InvalidArg)
        ));
    }

    /// With no caches supplied there is nothing to draft from; it must return
    /// empty rather than dereference a null cache.
    #[test]
    fn cache_draft_with_no_caches_is_empty() {
        let tokens = toks(&[1, 2, 3]);
        let draft = ngram_cache_draft(&tokens, 4, 1, 4, None, None, None).unwrap();
        assert!(draft.is_empty(), "got {draft:?}");
    }

    #[test]
    fn cache_draft_predicts_a_learned_repeat() {
        let tokens = toks(&[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]);
        let mut cache = NgramCache::new();
        cache
            .update(1, 4, &tokens, i32::try_from(tokens.len()).unwrap(), false)
            .unwrap();

        let draft = ngram_cache_draft(&tokens, 4, 1, 4, Some(&mut cache), None, None).unwrap();
        assert!(
            draft.contains(&LlamaToken(3)),
            "expected 3 after 1,2; got {draft:?}"
        );
    }

    #[test]
    fn map_builds_and_drafts() {
        let mut map = NgramMap::new(2, 2, false, 1).expect("map");
        let tokens = toks(&[1, 2, 3, 4, 1, 2]);
        map.begin(&tokens).unwrap();
        // Whether it drafts depends on its heuristics; the contract under test
        // is that the call succeeds and reports acceptance without panicking.
        let _ = map.draft(&tokens, LlamaToken(2)).unwrap();
        map.accept(0);
    }

    #[test]
    fn map_begin_accepts_an_empty_prompt() {
        let mut map = NgramMap::new(2, 2, false, 1).expect("map");
        map.begin(&[]).unwrap();
    }
}
