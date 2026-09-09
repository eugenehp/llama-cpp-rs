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

// Guarded wrappers over the `llama_quant_*` preview API.
//
// `llama_quant_model_from_metadata` and `llama_quant_init` throw on input they
// reject (an unknown architecture, a model they cannot quantize). Letting that
// unwind into Rust aborts the process with "Rust cannot catch foreign
// exceptions", so both are wrapped to return NULL instead.

struct llama_model * llama_quant_model_from_metadata_guarded(
        const struct llama_quant_model_desc * desc);

struct quantize_state_impl * llama_quant_init_guarded(
        const struct llama_model *                  model,
        const struct llama_model_quantize_params *  params);

// Returns 0 on success, non-zero if the underlying call threw.
int32_t llama_quant_compute_types_guarded(
        struct quantize_state_impl * qs,
        enum llama_ftype             ftype,
        struct ggml_tensor **        tensors,
        enum ggml_type *             result_types,
        size_t                       n_tensors);

// Returns 1 if quantizable, 0 if not, -1 if the underlying call threw.
int32_t llama_quant_tensor_allows_quantization_guarded(
        const struct quantize_state_impl * qs,
        const struct ggml_tensor *         tensor);

#ifdef __cplusplus
}
#endif
