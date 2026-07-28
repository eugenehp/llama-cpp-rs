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

// Which draft-model speculative strategy the session drives. Both map to an
// upstream `common_speculative_type`; they share the identical session
// lifecycle (begin/process/draft/accept) but differ in how the draft context
// must be built (see `mtp_session_new`).
enum mtp_spec_type {
    // Multi-token prediction. `ctx_dft` is built from the *same* model as
    // `ctx_tgt`, with `LLAMA_CONTEXT_TYPE_MTP`. Value 0 keeps the original
    // (pre-EAGLE3) ABI: a zero-initialised config selects MTP.
    MTP_SPEC_TYPE_MTP    = 0,
    // EAGLE-3. `ctx_dft` is built from a *separate* EAGLE-3 draft model (one
    // exposing 3 target-extract layers), with `LLAMA_CONTEXT_TYPE_DEFAULT`.
    MTP_SPEC_TYPE_EAGLE3 = 1,
};

// Result of a versioned speculative-state operation.
enum mtp_state_status {
    MTP_STATE_STATUS_OK             = 0,
    MTP_STATE_STATUS_NULL           = 1,
    MTP_STATE_STATUS_BAD_SEQUENCE   = 2,
    MTP_STATE_STATUS_NOT_QUIESCENT  = 3,
    MTP_STATE_STATUS_BUFFER_SMALL   = 4,
    MTP_STATE_STATUS_UNAVAILABLE    = 5,
    MTP_STATE_STATUS_INVALID        = 6,
    MTP_STATE_STATUS_OVERFLOW       = 7,
    MTP_STATE_STATUS_EXCEPTION      = 8,
};

struct mtp_session_config {
    uint32_t n_seq;
    int32_t  n_draft_max;
    int32_t  n_min;
    float    p_min;
    // One of `enum mtp_spec_type`. 0 (= MTP_SPEC_TYPE_MTP) preserves the
    // original behaviour for existing callers.
    int32_t  spec_type;
};

// Initialise a draft session that pairs `ctx_tgt` (the target context,
// `LLAMA_CONTEXT_TYPE_DEFAULT`) with `ctx_dft` (the draft context). The draft
// context depends on `config->spec_type`:
//   - MTP_SPEC_TYPE_MTP:    `ctx_dft` from the *same* MTP-capable model, built
//                           with `LLAMA_CONTEXT_TYPE_MTP`.
//   - MTP_SPEC_TYPE_EAGLE3: `ctx_dft` from a *separate* EAGLE-3 draft model,
//                           built with `LLAMA_CONTEXT_TYPE_DEFAULT`.
//
// `config` must be non-null with `n_seq > 0` and `n_draft_max > 0`.
// `n_min` and `p_min` map to `common_params_speculative_draft` (upstream
// defaults: 0 and 0.0).
//
// Returns nullptr on failure (e.g. when the model lacks the required draft
// heads / extract layers, or `spec_type` is unknown).
struct mtp_session * mtp_session_new(
        struct llama_context *              ctx_tgt,
        struct llama_context *              ctx_dft,
        const struct mtp_session_config *   config);

void mtp_session_free(struct mtp_session * s);

// True when any speculative backend needs post-norm embeddings on the target
// context (`llama_set_embeddings`). MTP returns false.
bool mtp_session_need_embd(const struct mtp_session * s);

// True when any speculative backend needs pre-norm hidden states on the target
// context (`llama_set_embeddings_pre_norm`). MTP returns true.
bool mtp_session_need_embd_pre_norm(const struct mtp_session * s);

// Optional: call once per fresh generation. `prompt` is the prompt-token array
// already decoded into the target context (used by ngram-style speculators;
// MTP currently uses it only for sanity assertions).
bool mtp_session_begin(
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
        struct mtp_session *       s,
        const struct llama_batch * batch);

// Generate up to `n_draft_max` draft tokens for sequence `seq_id`, starting
// from `id_last` at position `n_past`.
//
// On entry: `*out_n_tokens` is the capacity of `out_tokens` (must be at least
// `n_draft_max`).
// On return: `*out_n_tokens` is set to the number of tokens written, and
// `out_tokens[0..*out_n_tokens]` holds the draft.
bool mtp_session_draft(
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
bool mtp_session_accept(
        struct mtp_session * s,
        int32_t              seq_id,
        uint16_t             n_accepted);

// True only when no sequence has an unaccepted draft proposal.
bool mtp_session_is_quiescent(const struct mtp_session * s);

// True when `seq_id` has a proposal produced by `mtp_session_draft` that has
// not yet been completed by `mtp_session_accept`.
bool mtp_session_has_pending_proposal(
        const struct mtp_session * s,
        int32_t                    seq_id);

// Return the exact size of the versioned per-sequence speculative state.
//
// State includes the prompt storage owned by this shim and the selected
// implementation's opaque continuation bytes. It can be captured/restored
// only while the complete session is quiescent. Target and draft
// `llama_context` state must be checkpointed separately.
enum mtp_state_status mtp_session_state_size(
        const struct mtp_session * s,
        int32_t                    seq_id,
        size_t *                   out_size);

// Copy the exact versioned state. On `BUFFER_SMALL`, `out_written` receives the
// required size and no bytes are written.
enum mtp_state_status mtp_session_state_get(
        const struct mtp_session * s,
        int32_t                    seq_id,
        uint8_t *                  out_data,
        size_t                     capacity,
        size_t *                   out_written);

// Restore exact versioned state. Configuration, strategy, sequence identity,
// lengths, and the implementation-specific state are all validated before the
// shim commits its prompt storage.
enum mtp_state_status mtp_session_state_set(
        struct mtp_session * s,
        int32_t              seq_id,
        const uint8_t *      data,
        size_t               size);

// Log speculative-decoding statistics via llama.cpp's LOG_INF (draft/accept
// counts and timings). Requires a log callback if you want to capture output.
void mtp_session_print_stats(const struct mtp_session * s);

// Configured maximum draft length (`common_params_speculative_draft.n_max`).
int32_t mtp_session_n_max(const struct mtp_session * s);

#ifdef __cplusplus
}
#endif
