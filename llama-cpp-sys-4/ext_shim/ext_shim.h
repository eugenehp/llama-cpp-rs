#pragma once

#include <stddef.h>
#include <stdint.h>

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

struct llama_memory_breakdown_entry {
    char    buft_name[128];
    size_t  model;
    size_t  context;
    size_t  compute;
};

// Flatten llama_get_memory_breakdown() into `out` (max `max_out` entries).
// Returns the number of entries written.
size_t llama_memory_breakdown_collect(
        const struct llama_context *          ctx,
        struct llama_memory_breakdown_entry * out,
        size_t                                max_out);

struct common_device_memory_flat_entry {
    int64_t total;
    int64_t free;
    size_t  model;
    size_t  context;
    size_t  compute;
};

// Flatten common_get_device_memory_data() into `out` (max `max_out` entries).
// Writes hyper-parameters to the optional out-pointers when non-null.
// Returns the number of entries written, or (size_t)-1 on error.
size_t common_device_memory_collect(
        const char *                           path_model,
        const struct llama_model_params *      mparams,
        const struct llama_context_params *      cparams,
        enum ggml_log_level                      log_level,
        struct common_device_memory_flat_entry * out,
        size_t                                   max_out,
        uint32_t *                               hp_ngl,
        uint32_t *                               hp_n_ctx_train,
        uint32_t *                               hp_n_expert);

#ifdef __cplusplus
}
#endif
