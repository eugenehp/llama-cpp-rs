#include "ext_shim.h"

#include "fit.h"
#include "ggml-backend.h"
#include "llama-ext.h"

#include <cstring>
#include <vector>

size_t llama_memory_breakdown_collect(
        const struct llama_context *          ctx,
        struct llama_memory_breakdown_entry * out,
        size_t                                max_out) {
    if (!ctx || !out || max_out == 0) {
        return 0;
    }

    const auto breakdown = llama_get_memory_breakdown(ctx);
    size_t     i         = 0;

    for (const auto & [buft, mb] : breakdown) {
        if (i >= max_out) {
            break;
        }

        const char * name = buft ? ggml_backend_buft_name(buft) : "";
        if (!name) {
            name = "";
        }

        std::strncpy(out[i].buft_name, name, sizeof(out[i].buft_name) - 1);
        out[i].buft_name[sizeof(out[i].buft_name) - 1] = '\0';
        out[i].model    = mb.model;
        out[i].context  = mb.context;
        out[i].compute  = mb.compute;
        ++i;
    }

    return i;
}

size_t common_device_memory_collect(
        const char *                           path_model,
        const struct llama_model_params *      mparams,
        const struct llama_context_params *      cparams,
        enum ggml_log_level                      log_level,
        struct common_device_memory_flat_entry * out,
        size_t                                   max_out,
        uint32_t *                               hp_ngl,
        uint32_t *                               hp_n_ctx_train,
        uint32_t *                               hp_n_expert) {
    if (!path_model || !mparams || !cparams || !out || max_out == 0) {
        return (size_t) -1;
    }

    try {
        std::vector<ggml_backend_dev_t> devs;
        uint32_t                          ngl = 0;
        uint32_t                          nct = 0;
        uint32_t                          nex = 0;

        const auto vec = common_get_device_memory_data(
                path_model, mparams, cparams, devs, ngl, nct, nex, log_level);

        if (hp_ngl) {
            *hp_ngl = ngl;
        }
        if (hp_n_ctx_train) {
            *hp_n_ctx_train = nct;
        }
        if (hp_n_expert) {
            *hp_n_expert = nex;
        }

        const size_t n = vec.size() < max_out ? vec.size() : max_out;
        for (size_t i = 0; i < n; ++i) {
            out[i].total   = vec[i].total;
            out[i].free    = vec[i].free;
            out[i].model   = vec[i].model;
            out[i].context = vec[i].context;
            out[i].compute = vec[i].compute;
        }
        return n;
    } catch (...) {
        return (size_t) -1;
    }
}
