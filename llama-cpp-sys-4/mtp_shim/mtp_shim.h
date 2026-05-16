// Stable C entry points around upstream's `common_speculative` API
// (common/speculative.h), specialised for MTP — the multi-token-prediction
// speculative-decoding strategy added in llama.cpp PR #22673.
//
// Upstream exposes the draft loop only as C++ in `common/`. This shim
// re-exposes the bits we need with C linkage so Rust callers can bind to a
// stable surface that doesn't change shape every upstream refactor.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

struct mtp_session;

// Initialise an MTP draft session that pairs `ctx_tgt` (the target context,
// `LLAMA_CONTEXT_TYPE_DEFAULT`) with `ctx_dft` (the draft context, built with
// `LLAMA_CONTEXT_TYPE_MTP`). Both must be from the same MTP-capable model.
//
// `n_seq` is the number of concurrent sequences the session will track
// (usually 1 for a single conversation). `n_draft_max` caps the number of
// tokens drafted per round.
//
// Returns nullptr on failure (e.g. when the model lacks MTP heads).
struct mtp_session * mtp_session_new(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft,
        uint32_t n_seq,
        int32_t n_draft_max);

void mtp_session_free(struct mtp_session * s);

// Returns true if MTP requires embeddings to be extractable from the target
// context — callers should propagate this to `llama_set_embeddings(...)` /
// the pre-norm embeddings setter before any decode.
bool mtp_session_need_embd(const struct mtp_session * s);

// Optional: call once per fresh generation. `prompt` is the prompt-token array
// already decoded into the target context (used by ngram-style speculators;
// MTP currently uses it only for sanity assertions).
void mtp_session_begin(
        struct mtp_session * s,
        int32_t              seq_id,
        const llama_token *  prompt,
        size_t               n_prompt);

// Inform the session about a batch that was just decoded on the target
// context. MTP harvests the target's pre-norm hidden states from this batch
// to feed into the draft context on the next `mtp_session_draft` call.
//
// `batch` must be the exact same `llama_batch` that was passed to
// `llama_decode(ctx_tgt, batch)`.
bool mtp_session_process(
        struct mtp_session *      s,
        const struct llama_batch * batch);

// Generate up to `n_draft_max` draft tokens for sequence `seq_id`, starting
// from `id_last` at position `n_past`.
//
// On entry: `*out_n_tokens` is the capacity of `out_tokens` (must be at least
// `n_draft_max`).
// On return: `*out_n_tokens` is set to the number of tokens written, and
// `out_tokens[0..*out_n_tokens]` holds the draft.
void mtp_session_draft(
        struct mtp_session * s,
        int32_t              seq_id,
        llama_pos            n_past,
        llama_token          id_last,
        llama_token *        out_tokens,
        int32_t *            out_n_tokens);

// Inform the session that `n_accepted` of the last draft's tokens were
// accepted by the target verifier (and that the remainder were rejected).
// This updates per-sequence carryover state and rolls back the draft context's
// recurrent state past redundant pre-advancement.
void mtp_session_accept(
        struct mtp_session * s,
        int32_t              seq_id,
        uint16_t             n_accepted);

// Get/set the configured maximum draft length (mirrors
// `common_params_speculative_draft.n_max`).
int32_t mtp_session_n_max(const struct mtp_session * s);

#ifdef __cplusplus
}
#endif
