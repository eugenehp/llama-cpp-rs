#include "mtp_shim.h"

#include "common.h"
#include "speculative.h"

#include <vector>

struct mtp_session {
    common_speculative_ptr spec;

    // Per-seq storage for the deprecated `prompt` pointer in
    // common_speculative_draft_params (kept alive across draft() calls).
    std::vector<llama_tokens> prompts;

    // Per-seq result buffer the draft() call writes into.
    std::vector<llama_tokens> results;

    uint32_t n_seq       = 0;
    int32_t  n_draft_max = 0;
    int32_t  n_min       = 0;
    float    p_min       = 0.0f;
};

extern "C" mtp_session * mtp_session_new(
        llama_context *             ctx_tgt,
        llama_context *             ctx_dft,
        const mtp_session_config * config) {
    if (ctx_tgt == nullptr || ctx_dft == nullptr || config == nullptr) {
        return nullptr;
    }
    if (config->n_seq == 0 || config->n_draft_max <= 0) {
        return nullptr;
    }

    common_params_speculative sparams;
    sparams.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    sparams.draft.ctx_tgt = ctx_tgt;
    sparams.draft.ctx_dft = ctx_dft;
    sparams.draft.n_max   = config->n_draft_max;
    sparams.draft.n_min   = config->n_min;
    sparams.draft.p_min   = config->p_min;

    common_speculative * raw = common_speculative_init(sparams, config->n_seq);
    if (raw == nullptr) {
        return nullptr;
    }

    auto * s = new mtp_session;
    s->spec.reset(raw);
    s->prompts.resize(config->n_seq);
    s->results.resize(config->n_seq);
    s->n_seq       = config->n_seq;
    s->n_draft_max = config->n_draft_max;
    s->n_min       = config->n_min;
    s->p_min       = config->p_min;
    return s;
}

extern "C" void mtp_session_free(mtp_session * s) {
    delete s;
}

extern "C" bool mtp_session_need_embd(const mtp_session * s) {
    if (s == nullptr) {
        return false;
    }
    return common_speculative_need_embd(s->spec.get());
}

extern "C" bool mtp_session_need_embd_pre_norm(const mtp_session * s) {
    if (s == nullptr) {
        return false;
    }
    return common_speculative_need_embd_pre_norm(s->spec.get());
}

extern "C" void mtp_session_begin(
        mtp_session *       s,
        int32_t             seq_id,
        const llama_token * prompt,
        size_t              n_prompt) {
    if (s == nullptr || seq_id < 0 || (uint32_t) seq_id >= s->n_seq) {
        return;
    }

    auto & p = s->prompts[seq_id];
    p.assign(prompt, prompt + n_prompt);
    common_speculative_begin(s->spec.get(), seq_id, p);
}

extern "C" bool mtp_session_process(
        mtp_session *       s,
        const llama_batch * batch) {
    if (s == nullptr || batch == nullptr) {
        return false;
    }
    return common_speculative_process(s->spec.get(), *batch);
}

extern "C" void mtp_session_draft(
        mtp_session * s,
        int32_t       seq_id,
        llama_pos     n_past,
        llama_token   id_last,
        llama_token * out_tokens,
        int32_t *     out_n_tokens) {
    if (s == nullptr || out_tokens == nullptr || out_n_tokens == nullptr) {
        if (out_n_tokens) *out_n_tokens = 0;
        return;
    }

    const int32_t cap = *out_n_tokens;
    *out_n_tokens = 0;

    if (seq_id < 0 || (uint32_t) seq_id >= s->n_seq) {
        return;
    }

    auto & dp = common_speculative_get_draft_params(s->spec.get(), seq_id);
    auto & result = s->results[seq_id];
    result.clear();

    dp.drafting = true;
    dp.n_max    = s->n_draft_max;
    dp.n_past   = n_past;
    dp.id_last  = id_last;
    dp.prompt   = &s->prompts[seq_id];
    dp.result   = &result;

    common_speculative_draft(s->spec.get());

    const int32_t n = (int32_t) result.size();
    const int32_t to_copy = n < cap ? n : cap;
    for (int32_t i = 0; i < to_copy; ++i) {
        out_tokens[i] = result[i];
    }
    *out_n_tokens = to_copy;
}

extern "C" void mtp_session_accept(
        mtp_session * s,
        int32_t       seq_id,
        uint16_t      n_accepted) {
    if (s == nullptr || seq_id < 0 || (uint32_t) seq_id >= s->n_seq) {
        return;
    }
    common_speculative_accept(s->spec.get(), seq_id, n_accepted);
}

extern "C" void mtp_session_print_stats(const mtp_session * s) {
    if (s == nullptr) {
        return;
    }
    common_speculative_print_stats(s->spec.get());
}

extern "C" int32_t mtp_session_n_max(const mtp_session * s) {
    return s ? s->n_draft_max : 0;
}
