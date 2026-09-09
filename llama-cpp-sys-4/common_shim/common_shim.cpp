#include "common_shim.h"

#include "common.h"
#include "download.h"
#include "json.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "reasoning-budget.h"
#include "sampling.h"
#include "speculative.h"

#include "shim_support_impl.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace {

// Status codes, the error buffer and the size-then-fill helpers are shared by
// every shim; see shim_support_impl.hpp.
using llama_shim::blank;
using llama_shim::clear_error;
using llama_shim::emit;
using llama_shim::emit_tokens;
using llama_shim::guard;
using llama_shim::set_error;
using llama_shim::str_or_empty;

// Rebuild the vector-of-sequences from the flattened (data, lengths) pair the
// C interface uses.
std::vector<llama_tokens> unflatten(const int32_t * data, const size_t * lens, size_t n) {
    std::vector<llama_tokens> out;
    if (!data || !lens) {
        return out;
    }
    out.reserve(n);
    size_t off = 0;
    for (size_t i = 0; i < n; i++) {
        out.emplace_back(data + off, data + off + lens[i]);
        off += lens[i];
    }
    return out;
}

std::vector<llama_token> to_tokens(const int32_t * data, size_t n) {
    if (!data || n == 0) {
        return {};
    }
    return std::vector<llama_token>(data, data + n);
}

}  // namespace

struct common_shim_sampler_params {
    common_params_sampling params;
};

struct common_shim_sampler {
    common_sampler * ptr = nullptr;
    ~common_shim_sampler() {
        if (ptr) {
            common_sampler_free(ptr);
        }
    }
};

struct common_shim_ngram_cache {
    common_ngram_cache cache;
};

struct common_shim_ngram_map {
    common_ngram_map map;
    common_shim_ngram_map(uint16_t k, uint16_t v, bool key_only, uint16_t min_hits)
        : map(k, v, key_only, min_hits) {}
};

// ── reasoning budget ────────────────────────────────────────────────────────

struct llama_sampler * common_shim_reasoning_budget_init(
        const struct llama_vocab * vocab,
        const int32_t * starts,    const size_t * start_lens, size_t n_starts,
        const int32_t * ends,      const size_t * end_lens,   size_t n_ends,
        const int32_t * forced,    size_t         n_forced,
        int32_t         budget,
        int32_t         initial_state) {
    clear_error();
    try {
        const auto start_seqs = unflatten(starts, start_lens, n_starts);
        const auto end_seqs   = unflatten(ends, end_lens, n_ends);
        const auto forced_seq = to_tokens(forced, n_forced);
        return common_reasoning_budget_init(
            vocab, start_seqs, end_seqs, forced_seq, budget,
            static_cast<common_reasoning_budget_state>(initial_state));
    } catch (const std::exception & e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown C++ exception");
        return nullptr;
    }
}

int32_t common_shim_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        return static_cast<int32_t>(common_reasoning_budget_get_state(smpl));
    });
}

bool common_shim_reasoning_budget_force(struct llama_sampler * smpl) {
    if (!smpl) {
        return false;
    }
    try {
        return common_reasoning_budget_force(smpl);
    } catch (...) {
        return false;
    }
}

// ── guarded raw-sampler operations ──────────────────────────────────────────

int32_t common_shim_sampler_accept_raw(struct llama_sampler * smpl, int32_t token) {
    if (!smpl) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        llama_sampler_accept(smpl, token);
        return LLAMA_SHIM_OK;
    });
}

int32_t common_shim_sampler_sample_raw(
        struct llama_sampler * smpl,
        struct llama_context * ctx,
        int32_t                idx,
        int32_t *              out_status) {
    if (!smpl || !ctx) {
        if (out_status) {
            *out_status = LLAMA_SHIM_INVALID_ARG;
        }
        return -1;
    }
    llama_token token = -1;
    const int32_t status = guard([&]() -> int32_t {
        token = llama_sampler_sample(smpl, ctx, idx);
        return LLAMA_SHIM_OK;
    });
    if (out_status) {
        *out_status = status;
    }
    return status == LLAMA_SHIM_OK ? token : -1;
}

// ── assembled sampler chain ─────────────────────────────────────────────────

struct common_shim_sampler_params * common_shim_sampler_params_init(void) {
    clear_error();
    try {
        return new common_shim_sampler_params{};
    } catch (...) {
        set_error("allocation failed");
        return nullptr;
    }
}

void common_shim_sampler_params_free(struct common_shim_sampler_params * params) {
    delete params;
}

void common_shim_sampler_params_get_scalars(
        const struct common_shim_sampler_params * params,
        struct common_shim_sampler_scalars *      out) {
    if (!params || !out) {
        return;
    }
    const auto & p = params->params;
    out->seed                    = p.seed;
    out->n_prev                  = p.n_prev;
    out->n_probs                 = p.n_probs;
    out->min_keep                = p.min_keep;
    out->top_k                   = p.top_k;
    out->top_p                   = p.top_p;
    out->min_p                   = p.min_p;
    out->xtc_probability         = p.xtc_probability;
    out->xtc_threshold           = p.xtc_threshold;
    out->typ_p                   = p.typ_p;
    out->temp                    = p.temp;
    out->dynatemp_range          = p.dynatemp_range;
    out->dynatemp_exponent       = p.dynatemp_exponent;
    out->penalty_last_n          = p.penalty_last_n;
    out->penalty_repeat          = p.penalty_repeat;
    out->penalty_freq            = p.penalty_freq;
    out->penalty_present         = p.penalty_present;
    out->dry_multiplier          = p.dry_multiplier;
    out->dry_base                = p.dry_base;
    out->dry_allowed_length      = p.dry_allowed_length;
    out->dry_penalty_last_n      = p.dry_penalty_last_n;
    out->adaptive_target         = p.adaptive_target;
    out->adaptive_decay          = p.adaptive_decay;
    out->mirostat                = p.mirostat;
    out->top_n_sigma             = p.top_n_sigma;
    out->mirostat_tau            = p.mirostat_tau;
    out->mirostat_eta            = p.mirostat_eta;
    out->ignore_eos              = p.ignore_eos;
    out->no_perf                 = p.no_perf;
    out->timing_per_token        = p.timing_per_token;
    out->reasoning_budget_tokens = p.reasoning_budget_tokens;
    out->reasoning_control       = p.reasoning_control;
    out->backend_sampling        = p.backend_sampling;
}

void common_shim_sampler_params_set_scalars(
        struct common_shim_sampler_params *        params,
        const struct common_shim_sampler_scalars * s) {
    if (!params || !s) {
        return;
    }
    auto & p = params->params;
    p.seed                    = s->seed;
    p.n_prev                  = s->n_prev;
    p.n_probs                 = s->n_probs;
    p.min_keep                = s->min_keep;
    p.top_k                   = s->top_k;
    p.top_p                   = s->top_p;
    p.min_p                   = s->min_p;
    p.xtc_probability         = s->xtc_probability;
    p.xtc_threshold           = s->xtc_threshold;
    p.typ_p                   = s->typ_p;
    p.temp                    = s->temp;
    p.dynatemp_range          = s->dynatemp_range;
    p.dynatemp_exponent       = s->dynatemp_exponent;
    p.penalty_last_n          = s->penalty_last_n;
    p.penalty_repeat          = s->penalty_repeat;
    p.penalty_freq            = s->penalty_freq;
    p.penalty_present         = s->penalty_present;
    p.dry_multiplier          = s->dry_multiplier;
    p.dry_base                = s->dry_base;
    p.dry_allowed_length      = s->dry_allowed_length;
    p.dry_penalty_last_n      = s->dry_penalty_last_n;
    p.adaptive_target         = s->adaptive_target;
    p.adaptive_decay          = s->adaptive_decay;
    p.mirostat                = s->mirostat;
    p.top_n_sigma             = s->top_n_sigma;
    p.mirostat_tau            = s->mirostat_tau;
    p.mirostat_eta            = s->mirostat_eta;
    p.ignore_eos              = s->ignore_eos;
    p.no_perf                 = s->no_perf;
    p.timing_per_token        = s->timing_per_token;
    p.reasoning_budget_tokens = s->reasoning_budget_tokens;
    p.reasoning_control       = s->reasoning_control;
    p.backend_sampling        = s->backend_sampling;
}

int32_t common_shim_sampler_params_set_grammar(
        struct common_shim_sampler_params * params,
        const char *                        grammar,
        int32_t                             grammar_type,
        bool                                lazy) {
    if (!params) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        params->params.grammar = { static_cast<common_grammar_type>(grammar_type),
                                   str_or_empty(grammar) };
        params->params.grammar_lazy = lazy;
        return LLAMA_SHIM_OK;
    });
}

int32_t common_shim_sampler_params_add_grammar_trigger(
        struct common_shim_sampler_params * params,
        int32_t                             type,
        const char *                        value,
        int32_t                             token) {
    if (!params) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        common_grammar_trigger trigger;
        trigger.type  = static_cast<common_grammar_trigger_type>(type);
        trigger.value = str_or_empty(value);
        trigger.token = token;
        params->params.grammar_triggers.push_back(std::move(trigger));
        return LLAMA_SHIM_OK;
    });
}

int32_t common_shim_sampler_params_set_generation_prompt(
        struct common_shim_sampler_params * params,
        const char *                        generation_prompt) {
    if (!params) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        params->params.generation_prompt = str_or_empty(generation_prompt);
        return LLAMA_SHIM_OK;
    });
}

int32_t common_shim_sampler_params_add_logit_bias(
        struct common_shim_sampler_params * params,
        int32_t                             token,
        float                               bias) {
    if (!params) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        params->params.logit_bias.push_back({ token, bias });
        return LLAMA_SHIM_OK;
    });
}

int32_t common_shim_sampler_params_set_samplers(
        struct common_shim_sampler_params * params,
        const int32_t *                     samplers,
        size_t                              n_samplers) {
    if (!params || (!samplers && n_samplers > 0)) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        std::vector<common_sampler_type> out;
        out.reserve(n_samplers);
        for (size_t i = 0; i < n_samplers; i++) {
            out.push_back(static_cast<common_sampler_type>(samplers[i]));
        }
        params->params.samplers = std::move(out);
        return LLAMA_SHIM_OK;
    });
}

int32_t common_shim_sampler_params_set_dry_breakers(
        struct common_shim_sampler_params * params,
        const char * const *                breakers,
        size_t                              n_breakers) {
    if (!params || (!breakers && n_breakers > 0)) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        std::vector<std::string> out;
        out.reserve(n_breakers);
        for (size_t i = 0; i < n_breakers; i++) {
            out.push_back(str_or_empty(breakers[i]));
        }
        params->params.dry_sequence_breakers = std::move(out);
        return LLAMA_SHIM_OK;
    });
}

int32_t common_shim_sampler_params_set_reasoning_budget(
        struct common_shim_sampler_params * params,
        const int32_t * start,  size_t n_start,
        const int32_t * ends,   const size_t * end_lens, size_t n_ends,
        const int32_t * forced, size_t n_forced,
        const char *    message) {
    if (!params) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        params->params.reasoning_budget_start   = to_tokens(start, n_start);
        params->params.reasoning_budget_end     = unflatten(ends, end_lens, n_ends);
        params->params.reasoning_budget_forced  = to_tokens(forced, n_forced);
        params->params.reasoning_budget_message = str_or_empty(message);
        return LLAMA_SHIM_OK;
    });
}

struct common_shim_sampler * common_shim_sampler_init(
        const struct llama_model *          model,
        struct common_shim_sampler_params * params) {
    clear_error();
    if (!model || !params) {
        set_error("null model or params");
        return nullptr;
    }
    try {
        auto * smpl = common_sampler_init(model, params->params);
        if (!smpl) {
            set_error("common_sampler_init returned null");
            return nullptr;
        }
        auto * handle = new common_shim_sampler{};
        handle->ptr   = smpl;
        return handle;
    } catch (const std::exception & e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown C++ exception");
        return nullptr;
    }
}

void common_shim_sampler_free(struct common_shim_sampler * smpl) {
    delete smpl;
}

void common_shim_sampler_reset(struct common_shim_sampler * smpl) {
    if (!smpl || !smpl->ptr) {
        return;
    }
    try {
        common_sampler_reset(smpl->ptr);
    } catch (...) {
        // reset cannot meaningfully fail; swallow rather than unwind into Rust
    }
}

struct common_shim_sampler * common_shim_sampler_clone(struct common_shim_sampler * smpl) {
    clear_error();
    if (!smpl || !smpl->ptr) {
        return nullptr;
    }
    try {
        auto * cloned = common_sampler_clone(smpl->ptr);
        if (!cloned) {
            return nullptr;
        }
        auto * handle = new common_shim_sampler{};
        handle->ptr   = cloned;
        return handle;
    } catch (const std::exception & e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown C++ exception");
        return nullptr;
    }
}

int32_t common_shim_sampler_sample(
        struct common_shim_sampler * smpl,
        struct llama_context *       ctx,
        int32_t                      idx,
        bool                         grammar_first,
        int32_t *                    out_status) {
    if (!smpl || !smpl->ptr || !ctx) {
        if (out_status) {
            *out_status = LLAMA_SHIM_INVALID_ARG;
        }
        return -1;
    }
    llama_token token = -1;
    const int32_t status = guard([&]() -> int32_t {
        token = common_sampler_sample(smpl->ptr, ctx, idx, grammar_first);
        return LLAMA_SHIM_OK;
    });
    if (out_status) {
        *out_status = status;
    }
    return status == LLAMA_SHIM_OK ? token : -1;
}

void common_shim_sampler_accept(
        struct common_shim_sampler * smpl,
        int32_t                      token,
        bool                         is_generated) {
    if (!smpl || !smpl->ptr) {
        return;
    }
    try {
        common_sampler_accept(smpl->ptr, token, is_generated);
    } catch (...) {
        // swallow rather than unwind into Rust
    }
}

int32_t common_shim_sampler_sample_and_accept_n(
        struct common_shim_sampler * smpl,
        struct llama_context *       ctx,
        const int32_t *              draft,
        size_t                       n_draft,
        bool                         grammar_first,
        int32_t *                    out,
        size_t                       out_cap,
        size_t *                     out_len) {
    if (!smpl || !smpl->ptr || !ctx) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        const llama_tokens draft_tokens = to_tokens(draft, n_draft);
        const auto accepted =
            common_sampler_sample_and_accept_n(smpl->ptr, ctx, draft_tokens, grammar_first);
        return emit_tokens(accepted, out, out_cap, out_len);
    });
}

uint32_t common_shim_sampler_get_seed(const struct common_shim_sampler * smpl) {
    if (!smpl || !smpl->ptr) {
        return 0;
    }
    try {
        return common_sampler_get_seed(smpl->ptr);
    } catch (...) {
        return 0;
    }
}

int32_t common_shim_sampler_last(const struct common_shim_sampler * smpl) {
    if (!smpl || !smpl->ptr) {
        return -1;
    }
    try {
        return common_sampler_last(smpl->ptr);
    } catch (...) {
        return -1;
    }
}

bool common_shim_sampler_reasoning_budget_force(struct common_shim_sampler * smpl) {
    if (!smpl || !smpl->ptr) {
        return false;
    }
    try {
        return common_sampler_reasoning_budget_force(smpl->ptr);
    } catch (...) {
        return false;
    }
}

struct llama_sampler * common_shim_sampler_get(const struct common_shim_sampler * smpl) {
    if (!smpl || !smpl->ptr) {
        return nullptr;
    }
    try {
        return common_sampler_get(smpl->ptr);
    } catch (...) {
        return nullptr;
    }
}

int32_t common_shim_sampler_print(
        const struct common_shim_sampler * smpl,
        char *                             out_buf,
        size_t                             out_len,
        size_t *                           expected_len) {
    if (!smpl || !smpl->ptr) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        return emit(common_sampler_print(smpl->ptr), out_buf, out_len, expected_len);
    });
}

int32_t common_shim_sampler_prev_str(
        struct common_shim_sampler * smpl,
        struct llama_context *       ctx,
        int32_t                      n,
        char *                       out_buf,
        size_t                       out_len,
        size_t *                     expected_len) {
    if (!smpl || !smpl->ptr || !ctx) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        return emit(common_sampler_prev_str(smpl->ptr, ctx, n), out_buf, out_len, expected_len);
    });
}

int32_t common_shim_sampler_type_to_str(
        int32_t  sampler_type,
        char *   out_buf,
        size_t   out_len,
        size_t * expected_len) {
    return guard([&]() -> int32_t {
        const auto s = common_sampler_type_to_str(static_cast<common_sampler_type>(sampler_type));
        return emit(s, out_buf, out_len, expected_len);
    });
}

int32_t common_shim_sampler_types_from_names(
        const char * const * names,
        size_t               n_names,
        int32_t *            out,
        size_t               out_cap,
        size_t *             out_len) {
    if (!names && n_names > 0) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        std::vector<std::string> name_strings;
        name_strings.reserve(n_names);
        for (size_t i = 0; i < n_names; i++) {
            name_strings.push_back(str_or_empty(names[i]));
        }
        const auto types = common_sampler_types_from_names(name_strings);
        if (out_len) {
            *out_len = types.size();
        }
        if (!out || out_cap < types.size()) {
            return LLAMA_SHIM_BUFFER_TOO_SMALL;
        }
        for (size_t i = 0; i < types.size(); i++) {
            out[i] = static_cast<int32_t>(types[i]);
        }
        return LLAMA_SHIM_OK;
    });
}

// ── n-gram lookup decoding ──────────────────────────────────────────────────

int32_t common_shim_ngram_simple_draft(
        uint16_t        size_ngram,
        uint16_t        size_mgram,
        const int32_t * tokens,
        size_t          n_tokens,
        int32_t         sampled,
        int32_t *       out,
        size_t          out_cap,
        size_t *        out_len) {
    return guard([&]() -> int32_t {
        common_ngram_simple_config config{ size_ngram, size_mgram };
        const auto draft = common_ngram_simple_draft(config, to_tokens(tokens, n_tokens), sampled);
        return emit_tokens(draft, out, out_cap, out_len);
    });
}

struct common_shim_ngram_cache * common_shim_ngram_cache_init(void) {
    clear_error();
    try {
        return new common_shim_ngram_cache{};
    } catch (...) {
        set_error("allocation failed");
        return nullptr;
    }
}

void common_shim_ngram_cache_free(struct common_shim_ngram_cache * cache) {
    delete cache;
}

struct common_shim_ngram_cache * common_shim_ngram_cache_load(const char * path) {
    clear_error();
    if (blank(path)) {
        set_error("empty path");
        return nullptr;
    }
    try {
        auto * handle  = new common_shim_ngram_cache{};
        handle->cache  = common_ngram_cache_load(path);
        return handle;
    } catch (const std::exception & e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown C++ exception");
        return nullptr;
    }
}

int32_t common_shim_ngram_cache_save(struct common_shim_ngram_cache * cache, const char * path) {
    if (!cache || blank(path)) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        common_ngram_cache_save(cache->cache, path);
        return LLAMA_SHIM_OK;
    });
}

int32_t common_shim_ngram_cache_merge(
        struct common_shim_ngram_cache * cache,
        struct common_shim_ngram_cache * other) {
    if (!cache || !other) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        common_ngram_cache_merge(cache->cache, other->cache);
        return LLAMA_SHIM_OK;
    });
}

size_t common_shim_ngram_cache_size(const struct common_shim_ngram_cache * cache) {
    return cache ? cache->cache.size() : 0;
}

int32_t common_shim_ngram_cache_update(
        struct common_shim_ngram_cache * cache,
        int32_t                          ngram_min,
        int32_t                          ngram_max,
        const int32_t *                  tokens,
        size_t                           n_tokens,
        int32_t                          nnew,
        bool                             print_progress) {
    if (!cache) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        // Upstream takes a mutable reference but only reads; copy so the
        // caller's slice is never aliased mutably from C++.
        auto inp = to_tokens(tokens, n_tokens);
        common_ngram_cache_update(cache->cache, ngram_min, ngram_max, inp, nnew, print_progress);
        return LLAMA_SHIM_OK;
    });
}

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
        size_t *                         out_len) {
    if (n_tokens == 0) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        auto inp = to_tokens(tokens, n_tokens);
        // Upstream expects `draft` to already hold the seed token and appends
        // to it; hand back only what it added.
        llama_tokens draft = { inp.back() };

        static common_ngram_cache empty_cache;
        common_ngram_cache & ctx_cache = nc_context ? nc_context->cache : empty_cache;
        common_ngram_cache & dyn_cache = nc_dynamic ? nc_dynamic->cache : empty_cache;
        common_ngram_cache & sta_cache = nc_static ? nc_static->cache : empty_cache;

        common_ngram_cache_draft(inp, draft, n_draft, ngram_min, ngram_max, ctx_cache, dyn_cache,
                                 sta_cache);

        if (!draft.empty()) {
            draft.erase(draft.begin());
        }
        return emit_tokens(draft, out, out_cap, out_len);
    });
}

struct common_shim_ngram_map * common_shim_ngram_map_init(
        uint16_t size_key,
        uint16_t size_value,
        bool     key_only,
        uint16_t min_hits) {
    clear_error();
    try {
        return new common_shim_ngram_map(size_key, size_value, key_only, min_hits);
    } catch (const std::exception & e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown C++ exception");
        return nullptr;
    }
}

void common_shim_ngram_map_free(struct common_shim_ngram_map * map) {
    delete map;
}

int32_t common_shim_ngram_map_begin(
        struct common_shim_ngram_map * map,
        const int32_t *                tokens,
        size_t                         n_tokens) {
    if (!map) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        const auto inp = to_tokens(tokens, n_tokens);
        common_ngram_map_begin(map->map, inp);
        return LLAMA_SHIM_OK;
    });
}

int32_t common_shim_ngram_map_draft(
        struct common_shim_ngram_map * map,
        const int32_t *                tokens,
        size_t                         n_tokens,
        int32_t                        sampled,
        int32_t *                      out,
        size_t                         out_cap,
        size_t *                       out_len) {
    if (!map) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        const auto inp = to_tokens(tokens, n_tokens);
        llama_tokens draft;
        common_ngram_map_draft(map->map, inp, sampled, draft);
        return emit_tokens(draft, out, out_cap, out_len);
    });
}

void common_shim_ngram_map_accept(struct common_shim_ngram_map * map, uint16_t n_accepted) {
    if (!map) {
        return;
    }
    try {
        common_ngram_map_accept(map->map, n_accepted);
    } catch (...) {
        // swallow rather than unwind into Rust
    }
}

// ── speculative introspection ───────────────────────────────────────────────

int32_t common_shim_speculative_types_from_gguf(
        const char * path,
        int32_t *    out,
        size_t       out_cap,
        size_t *     out_len) {
    if (blank(path)) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        const auto types = common_speculative_types_from_gguf(path);
        if (out_len) {
            *out_len = types.size();
        }
        if (!out || out_cap < types.size()) {
            return LLAMA_SHIM_BUFFER_TOO_SMALL;
        }
        for (size_t i = 0; i < types.size(); i++) {
            out[i] = static_cast<int32_t>(types[i]);
        }
        return LLAMA_SHIM_OK;
    });
}

int32_t common_shim_speculative_type_from_name(const char * name, int32_t * out_type) {
    if (blank(name) || !out_type) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        const auto type = common_speculative_type_from_name(name);
        // Upstream reports "not recognised" by returning the COUNT sentinel
        // rather than throwing, so translate it here — a caller that stored the
        // sentinel would silently select no strategy at all.
        if (type == COMMON_SPECULATIVE_TYPE_COUNT) {
            set_error(std::string("unknown speculative type: ") + name);
            return LLAMA_SHIM_INVALID_ARG;
        }
        *out_type = static_cast<int32_t>(type);
        return LLAMA_SHIM_OK;
    });
}

int32_t common_shim_speculative_type_to_str(
        int32_t  spec_type,
        char *   out_buf,
        size_t   out_len,
        size_t * expected_len) {
    return guard([&]() -> int32_t {
        const auto s =
            common_speculative_type_to_str(static_cast<common_speculative_type>(spec_type));
        return emit(s, out_buf, out_len, expected_len);
    });
}

int32_t common_shim_speculative_all_types_str(
        char *   out_buf,
        size_t   out_len,
        size_t * expected_len) {
    return guard([&]() -> int32_t {
        return emit(str_or_empty(common_speculative_all_types_str()), out_buf, out_len,
                    expected_len);
    });
}

// ── logging ─────────────────────────────────────────────────────────────────

void common_shim_log_set_verbosity(int32_t verbosity) {
    common_log_set_verbosity_thold(verbosity);
}

int32_t common_shim_log_get_verbosity(int32_t level) {
    return common_log_get_verbosity(static_cast<ggml_log_level>(level));
}

void common_shim_log_set_timestamps(bool timestamps) {
    common_log_set_timestamps(common_log_main(), timestamps);
}

void common_shim_log_set_prefix(bool prefix) {
    common_log_set_prefix(common_log_main(), prefix);
}

void common_shim_log_set_colors(bool colors) {
    common_log_set_colors(common_log_main(), colors ? LOG_COLORS_AUTO : LOG_COLORS_DISABLED);
}

void common_shim_log_set_jsonl(bool jsonl) {
    common_log_set_jsonl(common_log_main(), jsonl);
}

int32_t common_shim_log_set_file(const char * path) {
    return guard([&]() -> int32_t {
        common_log_set_file(common_log_main(), path);
        return LLAMA_SHIM_OK;
    });
}

void common_shim_log_pause(void) {
    common_log_pause(common_log_main());
}

void common_shim_log_resume(void) {
    common_log_resume(common_log_main());
}

// ── model download / resolution ─────────────────────────────────────────────

int32_t common_shim_download_resolve_path(
        const char * hf_repo_with_tag,
        const char * hf_file,
        char *       out_buf,
        size_t       out_len,
        size_t *     expected_len) {
    if (blank(hf_repo_with_tag)) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        const auto path = common_download_resolve_path(hf_repo_with_tag, str_or_empty(hf_file));
        return emit(path, out_buf, out_len, expected_len);
    });
}

int32_t common_shim_download_split_repo_tag(
        const char * hf_repo_with_tag,
        char *       out_repo, size_t repo_cap, size_t * repo_len,
        char *       out_tag,  size_t tag_cap,  size_t * tag_len) {
    if (blank(hf_repo_with_tag)) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        const auto [repo, tag] = common_download_split_repo_tag(hf_repo_with_tag);
        const int32_t r1 = emit(repo, out_repo, repo_cap, repo_len);
        const int32_t r2 = emit(tag, out_tag, tag_cap, tag_len);
        // Report "too small" if either side did not fit, so one retry sizes both.
        if (r1 != LLAMA_SHIM_OK) {
            return r1;
        }
        return r2;
    });
}

int32_t common_shim_download_remove(const char * hf_repo_with_tag) {
    if (blank(hf_repo_with_tag)) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        return common_download_remove(hf_repo_with_tag) ? 1 : 0;
    });
}

int32_t common_shim_list_cached_models(
        char *   out_buf,
        size_t   out_len,
        size_t * expected_len) {
    return guard([&]() -> int32_t {
        const auto models = common_list_cached_models();
        auto arr = common_json::array();
        for (const auto & m : models) {
            auto obj    = common_json::object();
            obj["repo"] = m.repo;
            obj["tag"]  = m.tag;
            obj["name"] = m.to_string();
            arr.push_back(obj);
        }
        return emit(arr.dump(), out_buf, out_len, expected_len);
    });
}

int32_t common_shim_docker_resolve_model(
        const char * docker,
        char *       out_buf,
        size_t       out_len,
        size_t *     expected_len) {
    if (blank(docker)) {
        return LLAMA_SHIM_INVALID_ARG;
    }
    return guard([&]() -> int32_t {
        return emit(common_docker_resolve_model(docker), out_buf, out_len, expected_len);
    });
}
