#include "mtp_shim.h"

#include "common.h"
#include "speculative.h"

#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

struct mtp_session {
    common_speculative_ptr spec;

    // Per-seq storage for the deprecated `prompt` pointer in
    // common_speculative_draft_params (kept alive across draft() calls).
    std::vector<llama_tokens> prompts;

    // Per-seq result buffer the draft() call writes into.
    std::vector<llama_tokens> results;

    // True between a successful nonempty draft and its matching accept.
    std::vector<bool> pending_proposals;

    uint32_t n_seq       = 0;
    int32_t  n_draft_max = 0;
    int32_t  n_min       = 0;
    float    p_min       = 0.0f;
    int32_t  spec_type   = MTP_SPEC_TYPE_MTP;
};

namespace {

constexpr uint8_t MTP_STATE_MAGIC[8] = { 'L', 'L', 'S', 'P', 'E', 'C', '1', 0 };
constexpr uint32_t MTP_STATE_VERSION = 1;
constexpr size_t MTP_STATE_HEADER_SIZE = 8 + 9 * sizeof(uint32_t);
constexpr size_t MTP_MAX_PROMPT_TOKENS = 1U << 20;
constexpr size_t MTP_MAX_STATE_BYTES = 64U * 1024U * 1024U;

bool mtp_session_valid_seq(const mtp_session * s, int32_t seq_id) {
    return s != nullptr && seq_id >= 0 && (uint32_t) seq_id < s->n_seq;
}

bool mtp_session_quiescent(const mtp_session * s) {
    if (s == nullptr) {
        return false;
    }
    for (bool pending : s->pending_proposals) {
        if (pending) {
            return false;
        }
    }
    return true;
}

void append_u32(std::vector<uint8_t> & out, uint32_t value) {
    out.push_back((uint8_t) (value & 0xff));
    out.push_back((uint8_t) ((value >> 8) & 0xff));
    out.push_back((uint8_t) ((value >> 16) & 0xff));
    out.push_back((uint8_t) ((value >> 24) & 0xff));
}

bool read_u32(const uint8_t * data, size_t size, size_t & offset, uint32_t & value) {
    if (offset > size || size - offset < sizeof(uint32_t)) {
        return false;
    }
    value = (uint32_t) data[offset]
          | (uint32_t) data[offset + 1] << 8
          | (uint32_t) data[offset + 2] << 16
          | (uint32_t) data[offset + 3] << 24;
    offset += sizeof(uint32_t);
    return true;
}

mtp_state_status build_state(
        const mtp_session * s,
        int32_t             seq_id,
        std::vector<uint8_t> & out) {
    if (s == nullptr) {
        return MTP_STATE_STATUS_NULL;
    }
    if (!mtp_session_valid_seq(s, seq_id)) {
        return MTP_STATE_STATUS_BAD_SEQUENCE;
    }
    if (!mtp_session_quiescent(s)) {
        return MTP_STATE_STATUS_NOT_QUIESCENT;
    }

    std::vector<uint8_t> implementation;
    if (!common_speculative_get_state(s->spec.get(), seq_id, implementation)) {
        return MTP_STATE_STATUS_UNAVAILABLE;
    }
    const auto & prompt = s->prompts[seq_id];
    if (prompt.size() > std::numeric_limits<uint32_t>::max() ||
            implementation.size() > std::numeric_limits<uint32_t>::max()) {
        return MTP_STATE_STATUS_OVERFLOW;
    }
    const size_t prompt_bytes = prompt.size() * sizeof(uint32_t);
    if (prompt_bytes / sizeof(uint32_t) != prompt.size() ||
            MTP_STATE_HEADER_SIZE > std::numeric_limits<size_t>::max() - prompt_bytes ||
            MTP_STATE_HEADER_SIZE + prompt_bytes >
                    std::numeric_limits<size_t>::max() - implementation.size()) {
        return MTP_STATE_STATUS_OVERFLOW;
    }
    const size_t total_size = MTP_STATE_HEADER_SIZE + prompt_bytes + implementation.size();
    if (total_size > MTP_MAX_STATE_BYTES) {
        return MTP_STATE_STATUS_OVERFLOW;
    }

    out.clear();
    out.reserve(total_size);
    out.insert(out.end(), std::begin(MTP_STATE_MAGIC), std::end(MTP_STATE_MAGIC));
    append_u32(out, MTP_STATE_VERSION);
    append_u32(out, (uint32_t) s->spec_type);
    append_u32(out, s->n_seq);
    append_u32(out, (uint32_t) s->n_draft_max);
    append_u32(out, (uint32_t) s->n_min);
    uint32_t p_min_bits = 0;
    static_assert(sizeof(p_min_bits) == sizeof(s->p_min));
    std::memcpy(&p_min_bits, &s->p_min, sizeof(p_min_bits));
    append_u32(out, p_min_bits);
    append_u32(out, (uint32_t) seq_id);
    append_u32(out, (uint32_t) prompt.size());
    append_u32(out, (uint32_t) implementation.size());
    for (llama_token token : prompt) {
        append_u32(out, (uint32_t) token);
    }
    out.insert(out.end(), implementation.begin(), implementation.end());
    return MTP_STATE_STATUS_OK;
}

} // namespace

extern "C" mtp_session * mtp_session_new(
        llama_context *             ctx_tgt,
        llama_context *             ctx_dft,
        const mtp_session_config * config) {
    try {
        if (ctx_tgt == nullptr || ctx_dft == nullptr || config == nullptr) {
            return nullptr;
        }
        if (config->n_seq == 0 || config->n_draft_max <= 0) {
            return nullptr;
        }

        common_speculative_type spec_type;
        switch (config->spec_type) {
            case MTP_SPEC_TYPE_MTP:    spec_type = COMMON_SPECULATIVE_TYPE_DRAFT_MTP;    break;
            case MTP_SPEC_TYPE_EAGLE3: spec_type = COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3; break;
            default:                   return nullptr; // unknown spec_type
        }

        common_params_speculative sparams;
        sparams.types         = { spec_type };
        sparams.draft.ctx_tgt = ctx_tgt;
        sparams.draft.ctx_dft = ctx_dft;
        sparams.draft.n_max   = config->n_draft_max;
        sparams.draft.n_min   = config->n_min;
        sparams.draft.p_min   = config->p_min;

        common_speculative * raw = common_speculative_init(sparams, config->n_seq);
        if (raw == nullptr) {
            return nullptr;
        }

        common_speculative_ptr spec(raw);
        auto s = std::make_unique<mtp_session>();
        s->spec = std::move(spec);
        s->prompts.resize(config->n_seq);
        s->results.resize(config->n_seq);
        s->pending_proposals.resize(config->n_seq, false);
        s->n_seq       = config->n_seq;
        s->n_draft_max = config->n_draft_max;
        s->n_min       = config->n_min;
        s->p_min       = config->p_min;
        s->spec_type   = config->spec_type;
        return s.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void mtp_session_free(mtp_session * s) {
    delete s;
}

extern "C" bool mtp_session_need_embd(const mtp_session * s) {
    if (s == nullptr) {
        return false;
    }
    try {
        return common_speculative_need_embd(s->spec.get());
    } catch (...) {
        return false;
    }
}

extern "C" bool mtp_session_need_embd_pre_norm(const mtp_session * s) {
    if (s == nullptr) {
        return false;
    }
    try {
        return common_speculative_need_embd_nextn(s->spec.get());
    } catch (...) {
        return false;
    }
}

extern "C" bool mtp_session_begin(
        mtp_session *       s,
        int32_t             seq_id,
        const llama_token * prompt,
        size_t              n_prompt) {
    if (!mtp_session_valid_seq(s, seq_id) || s->pending_proposals[seq_id] ||
            n_prompt > MTP_MAX_PROMPT_TOKENS ||
            (n_prompt > 0 && prompt == nullptr)) {
        return false;
    }

    try {
        auto & p = s->prompts[seq_id];
        if (n_prompt == 0) {
            p.clear();
        } else {
            p.assign(prompt, prompt + n_prompt);
        }
        common_speculative_begin(s->spec.get(), seq_id, p);
        return true;
    } catch (...) {
        return false;
    }
}

extern "C" bool mtp_session_process(
        mtp_session *       s,
        const llama_batch * batch) {
    if (s == nullptr || batch == nullptr) {
        return false;
    }
    try {
        return common_speculative_process(s->spec.get(), *batch);
    } catch (...) {
        return false;
    }
}

extern "C" bool mtp_session_draft(
        mtp_session * s,
        int32_t       seq_id,
        llama_pos     n_past,
        llama_token   id_last,
        llama_token * out_tokens,
        int32_t *     out_n_tokens) {
    if (s == nullptr || out_tokens == nullptr || out_n_tokens == nullptr) {
        if (out_n_tokens) *out_n_tokens = 0;
        return false;
    }

    const int32_t cap = *out_n_tokens;
    *out_n_tokens = 0;

    if (seq_id < 0 || (uint32_t) seq_id >= s->n_seq) {
        return false;
    }
    if (s->pending_proposals[seq_id] || cap < s->n_draft_max) {
        return false;
    }

    try {
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
        s->pending_proposals[seq_id] = to_copy > 0;
        return true;
    } catch (...) {
        *out_n_tokens = 0;
        return false;
    }
}

extern "C" bool mtp_session_accept(
        mtp_session * s,
        int32_t       seq_id,
        uint16_t      n_accepted) {
    if (!mtp_session_valid_seq(s, seq_id) || !s->pending_proposals[seq_id] ||
            n_accepted > s->results[seq_id].size()) {
        return false;
    }
    try {
        common_speculative_accept(s->spec.get(), seq_id, n_accepted);
        s->pending_proposals[seq_id] = false;
        s->results[seq_id].clear();
        return true;
    } catch (...) {
        return false;
    }
}

extern "C" bool mtp_session_is_quiescent(const mtp_session * s) {
    return mtp_session_quiescent(s);
}

extern "C" bool mtp_session_has_pending_proposal(
        const mtp_session * s,
        int32_t             seq_id) {
    return mtp_session_valid_seq(s, seq_id) && s->pending_proposals[seq_id];
}

extern "C" mtp_state_status mtp_session_state_size(
        const mtp_session * s,
        int32_t             seq_id,
        size_t *            out_size) {
    if (out_size == nullptr) {
        return MTP_STATE_STATUS_NULL;
    }
    *out_size = 0;
    try {
        std::vector<uint8_t> state;
        const mtp_state_status status = build_state(s, seq_id, state);
        if (status == MTP_STATE_STATUS_OK) {
            *out_size = state.size();
        }
        return status;
    } catch (...) {
        return MTP_STATE_STATUS_EXCEPTION;
    }
}

extern "C" mtp_state_status mtp_session_state_get(
        const mtp_session * s,
        int32_t             seq_id,
        uint8_t *           out_data,
        size_t              capacity,
        size_t *            out_written) {
    if (out_written == nullptr) {
        return MTP_STATE_STATUS_NULL;
    }
    *out_written = 0;
    try {
        std::vector<uint8_t> state;
        const mtp_state_status status = build_state(s, seq_id, state);
        if (status != MTP_STATE_STATUS_OK) {
            return status;
        }
        *out_written = state.size();
        if (capacity < state.size()) {
            return MTP_STATE_STATUS_BUFFER_SMALL;
        }
        if (!state.empty() && out_data == nullptr) {
            return MTP_STATE_STATUS_NULL;
        }
        if (!state.empty()) {
            std::memcpy(out_data, state.data(), state.size());
        }
        return MTP_STATE_STATUS_OK;
    } catch (...) {
        return MTP_STATE_STATUS_EXCEPTION;
    }
}

extern "C" mtp_state_status mtp_session_state_set(
        mtp_session *  s,
        int32_t        seq_id,
        const uint8_t * data,
        size_t          size) {
    if (s == nullptr || (size > 0 && data == nullptr)) {
        return MTP_STATE_STATUS_NULL;
    }
    if (!mtp_session_valid_seq(s, seq_id)) {
        return MTP_STATE_STATUS_BAD_SEQUENCE;
    }
    if (!mtp_session_quiescent(s)) {
        return MTP_STATE_STATUS_NOT_QUIESCENT;
    }
    try {
        if (size < MTP_STATE_HEADER_SIZE ||
                std::memcmp(data, MTP_STATE_MAGIC, sizeof(MTP_STATE_MAGIC)) != 0) {
            return MTP_STATE_STATUS_INVALID;
        }

        size_t offset = sizeof(MTP_STATE_MAGIC);
        uint32_t version = 0;
        uint32_t state_spec_type = 0;
        uint32_t state_n_seq = 0;
        uint32_t state_n_draft_max = 0;
        uint32_t state_n_min = 0;
        uint32_t state_p_min_bits = 0;
        uint32_t state_seq_id = 0;
        uint32_t prompt_count = 0;
        uint32_t implementation_size = 0;
        if (!read_u32(data, size, offset, version) ||
                !read_u32(data, size, offset, state_spec_type) ||
                !read_u32(data, size, offset, state_n_seq) ||
                !read_u32(data, size, offset, state_n_draft_max) ||
                !read_u32(data, size, offset, state_n_min) ||
                !read_u32(data, size, offset, state_p_min_bits) ||
                !read_u32(data, size, offset, state_seq_id) ||
                !read_u32(data, size, offset, prompt_count) ||
                !read_u32(data, size, offset, implementation_size)) {
            return MTP_STATE_STATUS_INVALID;
        }
        uint32_t p_min_bits = 0;
        std::memcpy(&p_min_bits, &s->p_min, sizeof(p_min_bits));
        if (version != MTP_STATE_VERSION ||
                state_spec_type != (uint32_t) s->spec_type ||
                state_n_seq != s->n_seq ||
                state_n_draft_max != (uint32_t) s->n_draft_max ||
                state_n_min != (uint32_t) s->n_min ||
                state_p_min_bits != p_min_bits ||
                state_seq_id != (uint32_t) seq_id) {
            return MTP_STATE_STATUS_INVALID;
        }

        const size_t prompt_bytes = (size_t) prompt_count * sizeof(uint32_t);
        if (prompt_count != 0 && prompt_bytes / sizeof(uint32_t) != prompt_count ||
                offset > size || prompt_bytes > size - offset ||
                implementation_size != size - offset - prompt_bytes) {
            return MTP_STATE_STATUS_INVALID;
        }

        llama_tokens prompt;
        prompt.reserve(prompt_count);
        for (uint32_t i = 0; i < prompt_count; ++i) {
            uint32_t token = 0;
            if (!read_u32(data, size, offset, token)) {
                return MTP_STATE_STATUS_INVALID;
            }
            prompt.push_back((llama_token) token);
        }
        std::vector<uint8_t> implementation(
                data + offset,
                data + offset + implementation_size);
        std::vector<uint8_t> previous;
        if (!common_speculative_get_state(s->spec.get(), seq_id, previous)) {
            return MTP_STATE_STATUS_UNAVAILABLE;
        }
        if (!common_speculative_set_state(s->spec.get(), seq_id, implementation)) {
            return MTP_STATE_STATUS_INVALID;
        }
        std::vector<uint8_t> restored;
        if (!common_speculative_get_state(s->spec.get(), seq_id, restored) ||
                restored != implementation) {
            if (!common_speculative_set_state(s->spec.get(), seq_id, previous)) {
                return MTP_STATE_STATUS_EXCEPTION;
            }
            std::vector<uint8_t> rolled_back;
            if (!common_speculative_get_state(s->spec.get(), seq_id, rolled_back) ||
                    rolled_back != previous) {
                return MTP_STATE_STATUS_EXCEPTION;
            }
            return MTP_STATE_STATUS_INVALID;
        }

        s->prompts[seq_id].swap(prompt);
        s->results[seq_id].clear();
        return MTP_STATE_STATUS_OK;
    } catch (...) {
        return MTP_STATE_STATUS_EXCEPTION;
    }
}

extern "C" void mtp_session_print_stats(const mtp_session * s) {
    if (s == nullptr) {
        return;
    }
    try {
        common_speculative_print_stats(s->spec.get());
    } catch (...) {
        return;
    }
}

extern "C" int32_t mtp_session_n_max(const mtp_session * s) {
    return s ? s->n_draft_max : 0;
}
