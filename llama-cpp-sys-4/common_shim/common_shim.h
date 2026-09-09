#pragma once

// Stable C linkage over the rest of llama.cpp's `common/` layer.
//
// Companion to `chat_shim`, covering the parts that are not chat: the assembled
// sampler chain, the reasoning-budget sampler, n-gram (draft-model-free)
// speculative decoding, speculative-type introspection, logging controls, and
// model download/resolution.
//
// Every entry point catches C++ exceptions and reports a status, because
// unwinding across `extern "C"` into Rust is undefined behaviour. Status codes,
// the error buffer and the size-then-fill string protocol are shared with every
// other shim; see `shim_support.h`.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "shim_support.h"

#ifdef __cplusplus
extern "C" {
#endif

struct llama_model;
struct llama_context;
struct llama_vocab;
struct llama_sampler;

// ── reasoning budget ────────────────────────────────────────────────────────

// Mirrors `common_reasoning_budget_state`.
enum common_shim_rbudget_state {
    COMMON_SHIM_RBUDGET_IDLE         = 0,
    COMMON_SHIM_RBUDGET_COUNTING     = 1,
    COMMON_SHIM_RBUDGET_FORCING      = 2,
    COMMON_SHIM_RBUDGET_WAITING_UTF8 = 3,
    COMMON_SHIM_RBUDGET_DONE         = 4,
};

// Build a reasoning-budget sampler.
//
// Token sequences are passed flattened: `starts` holds every start sequence
// back to back, with `start_lens[i]` giving the length of sequence `i`. Same
// for `ends`. This keeps the ABI to plain arrays rather than arrays of
// pointers.
//
// Returns NULL on failure.
struct llama_sampler * common_shim_reasoning_budget_init(
        const struct llama_vocab * vocab,
        const int32_t * starts,    const size_t * start_lens, size_t n_starts,
        const int32_t * ends,      const size_t * end_lens,   size_t n_ends,
        const int32_t * forced,    size_t         n_forced,
        int32_t         budget,
        int32_t         initial_state);

// Current state, or a negative status if `smpl` is not a budget sampler.
int32_t common_shim_reasoning_budget_get_state(const struct llama_sampler * smpl);

// Force the transition into FORCING. Returns true if it occurred.
bool common_shim_reasoning_budget_force(struct llama_sampler * smpl);

// ── guarded raw-sampler operations ──────────────────────────────────────────
//
// `llama_sampler_accept` and `llama_sampler_sample` reach llama.cpp's grammar
// code, which **throws** when a grammar can no longer accept anything —
// `llama-grammar.cpp` raises "Unexpected empty grammar stack after accepting
// piece". That happens whenever a model's vocabulary cannot satisfy the
// grammar, e.g. a JSON schema against a vocabulary with no `{`. Calling the raw
// entry points from Rust turns that into a process abort, so route them here.

int32_t common_shim_sampler_accept_raw(struct llama_sampler * smpl, int32_t token);

// Returns the sampled token, or -1 with a negative `*out_status`.
int32_t common_shim_sampler_sample_raw(
        struct llama_sampler * smpl,
        struct llama_context * ctx,
        int32_t                idx,
        int32_t *              out_status);

// ── assembled sampler chain ─────────────────────────────────────────────────

// The POD half of `common_params_sampling`. The vector/string fields are set
// through the dedicated calls below, which keeps this struct stable across
// upstream bumps that add container fields.
struct common_shim_sampler_scalars {
    uint32_t seed;
    int32_t  n_prev;
    int32_t  n_probs;
    int32_t  min_keep;
    int32_t  top_k;
    float    top_p;
    float    min_p;
    float    xtc_probability;
    float    xtc_threshold;
    float    typ_p;
    float    temp;
    float    dynatemp_range;
    float    dynatemp_exponent;
    int32_t  penalty_last_n;
    float    penalty_repeat;
    float    penalty_freq;
    float    penalty_present;
    float    dry_multiplier;
    float    dry_base;
    int32_t  dry_allowed_length;
    int32_t  dry_penalty_last_n;
    float    adaptive_target;
    float    adaptive_decay;
    int32_t  mirostat;
    float    top_n_sigma;
    float    mirostat_tau;
    float    mirostat_eta;
    bool     ignore_eos;
    bool     no_perf;
    bool     timing_per_token;
    int32_t  reasoning_budget_tokens;
    bool     reasoning_control;
    bool     backend_sampling;
};

// Opaque `common_params_sampling`.
struct common_shim_sampler_params;

// Allocate with upstream's defaults. Returns NULL on failure.
struct common_shim_sampler_params * common_shim_sampler_params_init(void);
void common_shim_sampler_params_free(struct common_shim_sampler_params * params);

void common_shim_sampler_params_get_scalars(
        const struct common_shim_sampler_params * params,
        struct common_shim_sampler_scalars *      out);

void common_shim_sampler_params_set_scalars(
        struct common_shim_sampler_params *         params,
        const struct common_shim_sampler_scalars *  scalars);

// Mirrors `common_grammar_type`. The type decides whether the generation
// prompt is prefilled into the grammar: user grammars must not be.
enum common_shim_grammar_type {
    COMMON_SHIM_GRAMMAR_NONE          = 0,
    COMMON_SHIM_GRAMMAR_USER          = 1,
    COMMON_SHIM_GRAMMAR_OUTPUT_FORMAT = 2,
    COMMON_SHIM_GRAMMAR_TOOL_CALLS    = 3,
};

int32_t common_shim_sampler_params_set_grammar(
        struct common_shim_sampler_params * params,
        const char *                        grammar,
        int32_t                             grammar_type,
        bool                                lazy);

// `type` is a `common_grammar_trigger_type`: 0 token, 1 word, 2 pattern,
// 3 pattern_full.
int32_t common_shim_sampler_params_add_grammar_trigger(
        struct common_shim_sampler_params * params,
        int32_t                             type,
        const char *                        value,
        int32_t                             token);

int32_t common_shim_sampler_params_set_generation_prompt(
        struct common_shim_sampler_params * params,
        const char *                        generation_prompt);

int32_t common_shim_sampler_params_add_logit_bias(
        struct common_shim_sampler_params * params,
        int32_t                             token,
        float                               bias);

// Replace the sampler ordering. Values are `common_sampler_type`.
int32_t common_shim_sampler_params_set_samplers(
        struct common_shim_sampler_params * params,
        const int32_t *                     samplers,
        size_t                              n_samplers);

// Replace the DRY sequence breakers.
int32_t common_shim_sampler_params_set_dry_breakers(
        struct common_shim_sampler_params * params,
        const char * const *                breakers,
        size_t                              n_breakers);

// Reasoning-budget token sequences, flattened as in
// `common_shim_reasoning_budget_init`.
int32_t common_shim_sampler_params_set_reasoning_budget(
        struct common_shim_sampler_params * params,
        const int32_t * start,  size_t n_start,
        const int32_t * ends,   const size_t * end_lens, size_t n_ends,
        const int32_t * forced, size_t n_forced,
        const char *    message);

// Opaque `common_sampler`.
struct common_shim_sampler;

// Build the assembled chain. Returns NULL on failure.
struct common_shim_sampler * common_shim_sampler_init(
        const struct llama_model *          model,
        struct common_shim_sampler_params * params);

void common_shim_sampler_free(struct common_shim_sampler * smpl);
void common_shim_sampler_reset(struct common_shim_sampler * smpl);

// Returns NULL on failure.
struct common_shim_sampler * common_shim_sampler_clone(struct common_shim_sampler * smpl);

// Sample from logits at `idx`. Returns the token, or writes a negative status
// to `*out_status` and returns -1.
int32_t common_shim_sampler_sample(
        struct common_shim_sampler * smpl,
        struct llama_context *       ctx,
        int32_t                      idx,
        bool                         grammar_first,
        int32_t *                    out_status);

void common_shim_sampler_accept(
        struct common_shim_sampler * smpl,
        int32_t                      token,
        bool                         is_generated);

// Speculative acceptance: validate `draft` against the target model and return
// every accepted token plus the one resampled at the first divergence.
//
// Writes at most `out_cap` tokens and reports the true count in `*out_len`;
// `out_len` never exceeds `n_draft + 1`.
int32_t common_shim_sampler_sample_and_accept_n(
        struct common_shim_sampler * smpl,
        struct llama_context *       ctx,
        const int32_t *              draft,
        size_t                       n_draft,
        bool                         grammar_first,
        int32_t *                    out,
        size_t                       out_cap,
        size_t *                     out_len);

uint32_t common_shim_sampler_get_seed(const struct common_shim_sampler * smpl);
int32_t  common_shim_sampler_last(const struct common_shim_sampler * smpl);
bool     common_shim_sampler_reasoning_budget_force(struct common_shim_sampler * smpl);

// Borrow the underlying `llama_sampler` chain. Owned by `smpl`.
struct llama_sampler * common_shim_sampler_get(const struct common_shim_sampler * smpl);

// Human-readable description of the assembled chain.
int32_t common_shim_sampler_print(
        const struct common_shim_sampler * smpl,
        char *                             out_buf,
        size_t                             out_len,
        size_t *                           expected_len);

// The last `n` sampled tokens as a string.
int32_t common_shim_sampler_prev_str(
        struct common_shim_sampler * smpl,
        struct llama_context *       ctx,
        int32_t                      n,
        char *                       out_buf,
        size_t                       out_len,
        size_t *                     expected_len);

// Sampler-type name helpers.
int32_t common_shim_sampler_type_to_str(
        int32_t  sampler_type,
        char *   out_buf,
        size_t   out_len,
        size_t * expected_len);

// Parse names ("top_k", "temperature", …) into `common_sampler_type` values.
int32_t common_shim_sampler_types_from_names(
        const char * const * names,
        size_t               n_names,
        int32_t *            out,
        size_t               out_cap,
        size_t *             out_len);

// ── n-gram lookup decoding (no draft model) ─────────────────────────────────

// Self-lookup drafting: find the most recent occurrence of the trailing n-gram
// and draft what followed it.
//
// Writes at most `out_cap` tokens; the true count goes to `*out_len`.
int32_t common_shim_ngram_simple_draft(
        uint16_t        size_ngram,
        uint16_t        size_mgram,
        const int32_t * tokens,
        size_t          n_tokens,
        int32_t         sampled,
        int32_t *       out,
        size_t          out_cap,
        size_t *        out_len);

// Opaque `common_ngram_cache` — the disk-backed statistical variant.
struct common_shim_ngram_cache;

struct common_shim_ngram_cache * common_shim_ngram_cache_init(void);
void common_shim_ngram_cache_free(struct common_shim_ngram_cache * cache);

// Returns NULL on failure (missing or malformed file).
struct common_shim_ngram_cache * common_shim_ngram_cache_load(const char * path);
int32_t common_shim_ngram_cache_save(struct common_shim_ngram_cache * cache, const char * path);

// Fold the counts from `other` into `cache`.
int32_t common_shim_ngram_cache_merge(
        struct common_shim_ngram_cache * cache,
        struct common_shim_ngram_cache * other);

// Number of distinct n-grams recorded.
size_t common_shim_ngram_cache_size(const struct common_shim_ngram_cache * cache);

// Learn from `tokens`. `nnew` is how many tokens were appended since the last
// call; upstream requires `tokens` to only ever be appended to.
int32_t common_shim_ngram_cache_update(
        struct common_shim_ngram_cache * cache,
        int32_t                          ngram_min,
        int32_t                          ngram_max,
        const int32_t *                  tokens,
        size_t                           n_tokens,
        int32_t                          nnew,
        bool                             print_progress);

// Draft continuation tokens for `tokens`. Any cache may be NULL.
//
// `out` receives the draft, excluding the seed token upstream expects to find
// already present. Writes at most `out_cap`; the count goes to `*out_len`.
int32_t common_shim_ngram_cache_draft(
        const int32_t *                  tokens,
        size_t                           n_tokens,
        int32_t                          n_draft,
        int32_t                          ngram_min,
        int32_t                          ngram_max,
        struct common_shim_ngram_cache * nc_context,
        struct common_shim_ngram_cache * nc_dynamic,
        struct common_shim_ngram_cache * nc_static,
        int32_t *                        out,
        size_t                           out_cap,
        size_t *                         out_len);

// Opaque `common_ngram_map` — the adaptive in-context variant, which tracks
// how well its own drafts performed and adjusts.
struct common_shim_ngram_map;

struct common_shim_ngram_map * common_shim_ngram_map_init(
        uint16_t size_key,
        uint16_t size_value,
        bool     key_only,
        uint16_t min_hits);

void common_shim_ngram_map_free(struct common_shim_ngram_map * map);

// Start a generation over `tokens`.
int32_t common_shim_ngram_map_begin(
        struct common_shim_ngram_map * map,
        const int32_t *                tokens,
        size_t                         n_tokens);

int32_t common_shim_ngram_map_draft(
        struct common_shim_ngram_map * map,
        const int32_t *                tokens,
        size_t                         n_tokens,
        int32_t                        sampled,
        int32_t *                      out,
        size_t                         out_cap,
        size_t *                       out_len);

// Report how many of the last draft's tokens were accepted, so the map can
// adapt.
void common_shim_ngram_map_accept(struct common_shim_ngram_map * map, uint16_t n_accepted);

// ── speculative introspection ───────────────────────────────────────────────

// Which speculative types a draft GGUF supports, without loading it.
// Values are `common_speculative_type`.
int32_t common_shim_speculative_types_from_gguf(
        const char * path,
        int32_t *    out,
        size_t       out_cap,
        size_t *     out_len);

// Parse a type name ("mtp", "eagle3", …). Writes a `common_speculative_type`.
int32_t common_shim_speculative_type_from_name(const char * name, int32_t * out_type);

int32_t common_shim_speculative_type_to_str(
        int32_t  spec_type,
        char *   out_buf,
        size_t   out_len,
        size_t * expected_len);

// Every recognised type name, for help text and validation.
int32_t common_shim_speculative_all_types_str(
        char *   out_buf,
        size_t   out_len,
        size_t * expected_len);

// ── logging ─────────────────────────────────────────────────────────────────

// These drive llama.cpp's own logger, i.e. the output the library itself
// produces. They are documented upstream as not thread-safe; call them during
// setup.
void common_shim_log_set_verbosity(int32_t verbosity);
int32_t common_shim_log_get_verbosity(int32_t level);
void common_shim_log_set_timestamps(bool timestamps);
void common_shim_log_set_prefix(bool prefix);
void common_shim_log_set_colors(bool colors);
void common_shim_log_set_jsonl(bool jsonl);
// `path` may be NULL to stop writing to a file.
int32_t common_shim_log_set_file(const char * path);
void common_shim_log_pause(void);
void common_shim_log_resume(void);

// ── model download / resolution ─────────────────────────────────────────────

// Resolve a Hugging Face `repo[:tag]` (optionally a specific file) to a local
// path, downloading into llama.cpp's own cache if needed.
int32_t common_shim_download_resolve_path(
        const char * hf_repo_with_tag,
        const char * hf_file,
        char *       out_buf,
        size_t       out_len,
        size_t *     expected_len);

// Split `repo:tag` into its parts. Either output may be NULL.
int32_t common_shim_download_split_repo_tag(
        const char * hf_repo_with_tag,
        char *       out_repo, size_t repo_cap, size_t * repo_len,
        char *       out_tag,  size_t tag_cap,  size_t * tag_len);

// Delete a cached model. Returns 1 if something was removed, 0 if not.
int32_t common_shim_download_remove(const char * hf_repo_with_tag);

// Cached models as a JSON array of
// `{"name","path","size","modified"}` objects.
int32_t common_shim_list_cached_models(
        char *   out_buf,
        size_t   out_len,
        size_t * expected_len);

// Resolve a Docker model reference to a local path.
int32_t common_shim_docker_resolve_model(
        const char * docker,
        char *       out_buf,
        size_t       out_len,
        size_t *     expected_len);

#ifdef __cplusplus
}
#endif
