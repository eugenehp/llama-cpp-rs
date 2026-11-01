# Changelog

## [0.10.0] - 2026-11-01

### Breaking

- **`SamplerInitError` gained a variant**, `TooManyDrySequenceBreakers`. The
  enum is not `#[non_exhaustive]` — matching the convention of this crate's
  other error types — so an exhaustive `match` on it needs a new arm.
- **`ChatParams::caps_json` values are no longer all booleans.** Upstream
  changed `common_chat_templates_get_caps` from
  `std::map<std::string, bool>` to a `common_json`, so the object can now
  carry richer types. It is still a JSON object; code that deserialized it as
  a map of bools needs widening.

### Added

- **`MtmdContext::capabilities`** → `MtmdCapabilities`, wrapping
  `mtmd_helper_get_capabilities`. Replaces the two single-purpose predicates
  upstream deprecated in `b12032` and reports six more besides — notably
  whether the model *generates* audio and whether it accepts a speaker
  reference for voice cloning, neither of which was previously expressible.
  It is keyed on the model rather than a context, matching upstream.
- **`LlamaSampler::try_dry`**, with `DRY_MAX_SEQ_BREAKERS` and
  `DRY_MAX_BREAKER_LEN`. llama.cpp keeps only the first 64 *distinct*
  sequence breakers, logs a warning and drops the rest — so a caller passing
  more gets a valid sampler that is quietly not the one they configured.
  `try_dry` refuses instead, counting breakers the way llama.cpp does
  (truncate to 40 bytes, drop empties, de-duplicate — in that order, so two
  breakers differing only past byte 40 are one). `dry` is unchanged.
- **`MtmdAudioGen::next_is_text` and `::logit_bias`** (`b12032`). Interleaved
  pipelines alternate between text and audio steps; `next_is_text` says which
  comes next, and `logit_bias` is what stops a text-only response switching to
  audio mid-sentence. Sampling without the bias can produce a token the
  pipeline cannot act on.
- **`MtmdBitmap::set_is_video`**, split out upstream in `b12032`. A
  video-capable model needs this *and* `set_mergeable` on every frame;
  `MtmdVideo::read_next` already sets both, so this is for frames you build
  yourself.
- **`GgmlType::F8E4M3` and `LlamaFtype::MostlyF8E4M3`** (`ggml_type` 43,
  `llama_ftype` 42), new upstream 8-bit float quantization.

### Fixed

- **`LlamaFtype::try_from` could not resolve the newest file types.** It
  resolves through `all()`, and a variant missing from that list is simply
  unconvertible — the raw ftype comes back as an error and a model quantized
  that way looks unreadable. `MostlyF8E4M3` is listed, and a new test walks
  every variant through the round trip so the next addition cannot repeat it.

### Changed

- **llama.cpp**: vendored submodule updated to `dde879bc4f` (tag `b12032`) from
  `184f653602` (`b11829`), 203 upstream commits spanning release `v0.8.0`.
  - **No state-format or RPC break.** `LLAMA_SESSION_VERSION`,
    `LLAMA_STATE_SEQ_VERSION` and the RPC wire protocol (`8.0.1`) are all
    unchanged, so context state written by 0.9.0 still loads and existing
    `rpc-server` peers still work.
  - `common_chat_templates_get_caps` now returns a `common_json` instead of a
    `std::map<std::string, bool>`. The shim no longer rebuilds the map into an
    object — the values are no longer all bools, which is why upstream changed
    it. `ChatParams::caps_json` is unaffected; it still yields a JSON object.
  - `mtmd_helper_gen_audio_inp` gained a `no_soundtrack` field. No build break:
    the request has been seeded from `mtmd_helper_gen_audio_inp_default()`
    since 0.9.0 precisely so that added fields are not breaking changes.
  - Patches `0003`–`0005` still apply in sequence; `common/speculative.h` saw
    only a comment change. None has been absorbed upstream.
- **Three more C++-by-value functions are blocklisted in `build.rs`.**
  `b12032` added architecture queries to `llama-ext.h` that traffic in C++
  types: `llama_arch_all` returns `std::vector<llm_arch>` (bindgen degraded it
  to `-> u8`), `llama_arch_kv_name` returns `std::string` by value, and
  `llama_arch_from_string` takes a `const std::string &`. All three are ABI
  mismatches, the same class as `llama_memory_get_buffers`. `llama_arch_name`
  and `llama_arch_supports_sm_tensor` take and return plain C types and are
  left alone.

- **The `openai-server` example no longer pulls in `zstd`.** `actix-web`'s
  default features include `compress-zstd`, which carried four unfixed
  advisories (RUSTSEC-2026-0391/0398/0403/0410) whose fixes need semver-major
  bumps that `actix-http`'s `zstd ^0.13` pin blocks. The server negotiates
  gzip and brotli and never used zstd, so the dependency is dropped rather
  than the advisory suppressed. Neither published crate was ever affected —
  `zstd` reached the workspace only through this example.

- **`main` would not build: `Cargo.lock` paired bindgen with a `syn` that
  `prettyplease` disagreed with.** bindgen 0.73 declares
  `prettyplease = ">=0.2.7, <0.4"` and `syn = ">=2, <4"` and lets cargo resolve
  them independently — prettyplease 0.2 is built on syn 2 and 0.3 on syn 3, so
  cargo is free to hand bindgen syn 2 alongside prettyplease 0.3, which cannot
  compile. Ordinary lockfile churn flips it, and it has flipped more than once.
  The failure surfaces inside bindgen's own source
  (`the trait bound syn::file::File: ParseQuote is not satisfied`) with nothing
  pointing at the lockfile, after roughly twenty minutes of CI.
  `scripts/check-bindgen-syn.sh` now compares the two majors straight out of
  `Cargo.lock` and says what to run; the `check` job calls it before anything
  compiles, so the same breakage costs seconds and names its own cause.

### Deprecated

- **`MtmdContext::supports_video` and `::model_can_chat`**, following upstream's
  deprecation of `mtmd_helper_support_video` and `mtmd_helper_model_can_chat`.
  Both still work; use `capabilities(model).video` and `.chat`.

## [0.9.0] - 2026-10-22

### Breaking

- **`LlamaFtype::MostlyQ1_0_G128` and `GgmlType::Q1_0_G128` are gone, and
  `GgmlType`'s numbering changed.** Both encoded the layout of this crate's own
  `0001-q1-quantization.patch`, which no longer exists: `Q1_0` landed upstream
  with different values, and upstream never had a `Q1_0_g128`. With the `q1`
  feature on, the crate disagreed with llama.cpp on every one of three
  `ggml_type` discriminants:

  | | upstream `ggml_type` | crate with `q1` (wrong) | now |
  |---|---|---|---|
  | `NVFP4` | 40 | 42 | 40 |
  | `Q1_0` | 41 | 40 | 41 |
  | `Q2_0` | 42 | — | 42 |

  A wrong discriminant here silently selects a different quantization. On the
  `llama_ftype` side `MostlyQ1_0 = 40` was already right; 41 was `Q1_0_g128` and
  is now `MostlyQ2_0`, matching upstream. `MostlyQ1_0`, `GgmlType::Q1_0` and
  `GgmlType::Q2_0` are all unconditional now — the feature gate was the bug.
  - This predates the bump (`Q2_0 = 41` was already at `b11557`), so 0.8.0 and
    earlier are affected whenever `q1` was enabled.
  - `tests/test_q1.rs` had `#![cfg(feature = "q1")]` on the whole file, so none
    of it ran under the default features — which is why the stale values went
    unnoticed. It is un-gated, the enum tests now run everywhere, and they pin
    against `llama.h` and `ggml.h` rather than against the old patch.
- **`ggml` and `rpc` are now default features** of `llama-cpp-4`
  (`default = ["openmp", "mtmd", "dynamic-link", "ggml", "rpc"]`). `ggml` is
  free — it appears nowhere in the sys build script and gates Rust API only.
  `rpc` adds a `GGML_RPC` backend build to every downstream build; turn it off
  with `default-features = false` if that matters. `mtmd` was already on.
- `q1` is retained as a no-op feature so `--features q1` keeps resolving.

### Fixed

- **The published `llama-cpp-sys-4` failed to configure with `metal` and `mtmd`
  together** — `mtmd` is a default feature, so this hit every Metal build that
  kept the defaults. `tools/CMakeLists.txt` adds `tools/tuning` under
  `GGML_METAL`, and the package `include` list did not ship it, so CMake
  stopped with "not an existing directory". Reported and diagnosed in
  [#346](https://github.com/eugenehp/llama-cpp-rs/pull/346) by @tmtiwari, whose
  macOS CI step — `cargo package --features metal,mtmd` — is taken with it; the
  Linux `package` job never sets `metal`, so this class of break stayed green
  everywhere.
- `tools/connect`, new in `b11829`, is packaged for the same reason: it is
  guarded by `LLAMA_CONNECT`, which the `LLAMA_*` passthrough made reachable.
  `LLAMA_CONNECT` is now refused by that passthrough — it makes cmake download
  a prebuilt third-party binary from a GitHub release mid-build, which is not a
  decision an environment variable should make.
- **`rustls` advisory in the published crate's build graph.** RUSTSEC-2026-0346
  (`Debug` impls for client session values expose TLS secrets) reached
  `llama-cpp-sys-4` through its `ureq` build dependency. Fixed by a lockfile
  update to `rustls` 0.23.46, along with RUSTSEC-2026-0382 (`rustls-webpki`
  accepting distrusted intermediates) and RUSTSEC-2026-0381 (`webpki-roots`
  trusting distrusted emSign intermediates). Both lockfiles updated.

### Added

- **`PerfContextData::n_reused`, `n_reallocs_same` and `n_reallocs_diff`**,
  wrapping upstream's graph-reuse counters — `n_reused` was never exposed, and
  the two realloc counters are new in this bump. Together they say whether a
  context is reusing its compute graph or rebuilding it every step, which is
  the first thing to check when decode throughput is worse than it should be.
- **`ChatParams::prompt_parts`** (`chat_shim_apply_result::prompt_parts_off`),
  carrying upstream's new per-part `is_input` flags, and
  **`ChatParams::tokenize_prompt`**, which is the reason they exist: it walks
  the parts, tokenizing template spans with special-token parsing on and user
  spans with it off, so a user who types `<|im_start|>assistant` gets the
  literal characters instead of the real control token. `add_bos` applies to
  the first non-empty span only. Prefer it over
  `model.str_to_token(&params.prompt, ..)` for anything taking untrusted input.
  See *Changed*.
- **`LlamaModel::str_to_token_with_special` and `str_to_token_into_with_special`.**
  `str_to_token` hardcoded `parse_special = true`, so until now the crate could
  not tokenize anything as plaintext at all — the defence above was
  inexpressible. The existing two keep their behaviour and now delegate.
- **`LlamaModel::token_embedding_rows`**, wrapping
  `llama_model_get_tok_embd_rows`. `token_embeddings()` copies the whole
  matrix, which is ~500 MB for a 128k-vocab 1024-dim model; this copies only
  the rows you name.
- **`LlamaModel::decision_labels`**, wrapping `llama_model_decision_label`.
  Decision models (`clef`, `nimble`, `pplx-decider`, the `d1` family) score a
  list of options instead of generating text; these are the names those scores
  belong to. Without them a caller has a vector of numbers and no way to say
  which option each is for. Empty for an ordinary model.
- **`LlamaModel::n_tokens_delay`**, wrapping `llama_model_n_tokens_delay` — how
  far a realtime model's output lags its audio input, which is what lines
  generated audio up against the input stream. `0` when not applicable.
- **`LlamaModel::set_attn_rot`**, wrapping `llama_model_set_attn_rot`
  (TurboQuant's KV-cache attention rotation). Upstream marks the setter as for
  testing, and it mutates the model every later context inherits from, so
  `LlamaContextParams::with_attn_rot_disabled` remains the right tool for
  ordinary use.
- **`LlamaContext::reserve_graph`** → `GraphReservation` / `ReserveGraphError`,
  wrapping `llama_graph_reserve`. Two uses: as warmup it pays the first
  decode's graph build and buffer allocation up front, so first-token latency
  stops being an outlier; as measurement the node count is what
  `n_reallocs_diff` counts changes in. It deliberately returns counts rather
  than the `ggml_cgraph *`, which upstream documents as valid only until the
  next `llama_graph_reserve` call.
- **`LlamaContext::memory_buffer_count`**, wrapping `llama_memory_get_buffers`
  through a new `ext_shim` entry point. Upstream derives it by running a state
  write, so it is `0` for an untouched cache and otherwise counts the backend
  buffers a save would touch.
- `GraphReservation`, `ReserveGraphError` and `PromptPart` are exported from
  the prelude.

### Changed

- **llama.cpp**: vendored submodule updated to `184f653602` (tag `b11829`) from
  `5c8854fd88` (`b11557`), 272 upstream commits spanning release `v0.7.0`.
  - **The chat prompt is now a list of parts, not a string**
    ([#30389](https://github.com/ggml-org/llama.cpp/pull/30389)). Each part
    carries `is_input`, marking the spans that came from user content — which
    must be tokenized *without* special-token parsing, or a user typing
    `<|im_start|>` forges a role. `ChatParams::prompt` keeps the flattened text
    so existing code is unaffected, and the flags are exposed alongside it
    rather than discarded.
  - `mtmd_helper_gen_audio_inp` gained a `stream` field. The request is now
    seeded from `mtmd_helper_gen_audio_inp_default()` and overridden, instead of
    being built as a struct literal — the same pattern `LlamaContextParams`
    uses, so the next field upstream adds is not a build break.
  - **No state-format or RPC break this time.** `LLAMA_SESSION_VERSION` and
    `LLAMA_STATE_SEQ_VERSION` are unchanged, so context state written by 0.8.0
    still loads, and the RPC wire protocol moved only `8.0.0` → `8.0.1` — the
    handshake ignores the patch component.
  - Patches `0003`–`0005` still apply, at offsets; `common/speculative.h` was
    untouched upstream. Neither has been absorbed: upstream still returns
    `void` from `common_speculative_set_state`, and the EAGLE3 abort sites are
    still there.
- **`llama_memory_get_buffers` is blocklisted in `build.rs`.** It returns a
  `std::vector<ggml_backend_buffer_t>` by value, which bindgen cannot express
  and silently degraded to `-> u8`; calling the generated binding would have
  been an ABI mismatch. `llama_memory_buffers_collect` in `ext_shim` replaces
  it with a count-or-fill C signature.
- **Patch `0005` now fail-closes a fourth EAGLE3 abort it had missed.** The
  constructor validates `target_layer_ids` — read from the *draft checkpoint's
  GGUF metadata* — against the target model's layer count, and called
  `GGML_ABORT` when they disagreed: a mismatched pair of models took the whole
  process down. It throws now, which `common_speculative_init` and the MTP shim
  already turn into a null return and a typed Rust error. Upstream's own sibling
  check in that same constructor throws too, so the abort was the odd one out.
- The build script no longer pretends to stage `0001-q1-quantization.patch`.
  The patch was deleted when Q1_0 landed upstream, and an `if source.exists()`
  guard turned the staging into a silent no-op, so the `q1` feature looked
  alive while doing nothing.

## [0.8.0] - 2026-10-11

### Breaking

- **`ChatParams::parse_with` is gone**, and with it the `parse_tool_calls` and
  `reasoning_in_content` arguments. llama.cpp
  [#30210](https://github.com/ggml-org/llama.cpp/pull/30210) builds the parser
  once in `common_chat_templates_apply` and no longer accepts either switch at
  parse time, so there is nothing left for them to change.
  - Replace `params.parse_with(text, is_partial, true, false)` with
    [`ChatParams::parse`].
  - For the `parse_tool_calls: false` / `reasoning_in_content: true` behaviour,
    set [`ChatApplyParams::with_force_pure_content`] when rendering — the
    decision has to be made before the parser is built. See *Added*.
  - [`ChatApplyParams::with_reasoning_format`] is unaffected; apply is where
    upstream has always read it.
- **Saved context state from 0.7.0 will not load.** Upstream bumped
  `LLAMA_SESSION_VERSION` 10 → 11 and `LLAMA_STATE_SEQ_VERSION` 3 → 4, and the
  loader requires an exact version match, so files written by `state_save_file`,
  `state_seq_save_file` or `save_session_file` under 0.7.0 are rejected by this
  build. Regenerate them. The MTP/EAGLE3 speculative blobs from
  `MtpSession::speculative_state` carry their own version and are unaffected.
- **The RPC wire protocol went 6 → 8**, so an `rpc`-feature client built here
  cannot talk to an `rpc-server` from 0.7.0 or earlier, or vice versa — the
  handshake rejects a mismatched major version. `-sm tensor` across RPC backends
  ([#26610](https://github.com/ggml-org/llama.cpp/pull/26610)) and a
  weights-only hash cache
  ([#28789](https://github.com/ggml-org/llama.cpp/pull/28789)) each bumped it.
  Upgrade both ends together.
- **New names in the prelude.** It now follows the rule that anything a
  prelude-exported API takes, returns or fails with is exported too, so `use
  llama_cpp_4::prelude::*` is enough to write a complete call including its
  `match` arms. That adds `BatchEntry`, `BatchExtError`, `CommonSamplerError`,
  `Eagle3SessionError`, `EmbeddingRows`, `FtypeNameError`, `LlamaBatchExt`,
  `LlamaDecisionOrder`, `LlamaLoraAdapter`, `LlamaLoraAdapterInitError`,
  `LlamaLoraAdapterRemoveError`, `LlamaLoraAdapterSetError`, `LlamaProcessType`,
  `LlamaTokenData`, `MROPE_SECTIONS`, `MtpSessionError`, `NgramError`,
  `ProcessError`, `QuantPreviewError`, `RuntimeError` and `ShimError`. A local
  item with one of those names will need disambiguating.
  - The gaps closed were real: `LlamaModel::lora_adapter_init` returned a type
    the prelude could not name, `LlamaContext::candidates_ith` yielded one, and
    four shim-backed modules' error types were unreachable — so a `match` on a
    failure meant reaching past the prelude.
  - Section comments inside the module now describe their contents; chat and
    sampler-chain exports were previously filed under *Speculative decoding*.

### Added

- **New `llama_batch_ext` module: entry-wise batches.** [`LlamaBatchExt`] wraps
  upstream's second batch API
  ([#24669](https://github.com/ggml-org/llama.cpp/pull/24669)), and
  [`LlamaContext::process`] runs one through the encoder or decoder
  (`llama_process`). Where [`LlamaBatch`] is six parallel arrays plus one flat
  embedding block for the whole batch, this is a list of entries that each own
  their token id, positions, input embedding and state embedding — which is the
  only way to express three things:
  - **embeddings on some entries but not others**, e.g. a vision encoder's
    output interleaved with text tokens;
  - **M-RoPE positions**, `[t, y, x, z]` per embedding entry instead of one
    scalar, via [`LlamaBatchExt::set_positions`] and [`MROPE_SECTIONS`];
  - **a state embedding** — the target model's hidden state that an MTP draft
    context reads ([`LlamaBatchExt::set_state_embedding`]).
  - Plus [`LlamaDecisionOrder`] for the joint-decision-head models
    ([#29831](https://github.com/ggml-org/llama.cpp/pull/29831)).
  - Four things the C API takes on trust are checked here instead. A
    [`BatchEntry`] handle is a distinct type, so it cannot be confused with the
    token ids, sequence ids and positions that are also `i32`, and it is
    validated against the batch — a handle kept across a
    [`LlamaBatchExt::clear`] is rejected rather than silently writing to
    a recycled entry. [`EmbeddingRows`] refuses a `n_rows × n_embd` that does
    not match the buffer it describes, which llama.cpp would otherwise read
    past. `llama_batch_ext_set_pos` reads one position for a token entry but
    `n_pos_per_embd` for an embedding entry, straight off the caller's pointer;
    [`LlamaBatchExt`] tracks which kind each entry is and rejects a slice of the
    wrong length instead of letting it read out of bounds. And a token id is
    range-checked in Rust, because `llama_batch_ext_add_token` appends the entry
    *before* it validates the id.
  - That last point is also why an entry can end up carrying no input at all:
    `llama_batch_ext_add_embd` appends before validating the block too, and
    upstream has no call to remove an entry. The wrapper mirrors the orphan
    rather than losing index alignment with upstream, reports it through
    [`LlamaBatchExt::has_incomplete_entry`], and
    [`LlamaContext::process`] refuses the batch with
    [`ProcessError::IncompleteEntry`] instead of submitting one with a hole in
    it. [`LlamaBatchExt::clear`] is the way out. The same guard covers an
    [`LlamaBatchExt::add_empty`] entry that was never filled in.
  - A batch captures its context's batch size, sequence count and memory
    module, so it is not transferable. [`LlamaContext::process`] rejects a batch
    built by any other context — including one that has since been dropped,
    which a raw pointer comparison would miss once the allocator reuses the
    address.
  - [`LlamaContext::process`] refuses to run when the context owns tensor
    transactions. Those are driven by hooks this repo patches into
    `llama_decode`, which `llama_process` does not pass through, and the hook
    signature is built around `llama_batch` so it cannot describe an entry-wise
    batch at all. Failing is better than running the batch with the
    transactions silently inert.
  - Note that a plain token batch has nothing to gain here: [`LlamaBatch`] with
    [`LlamaContext::decode`] is the same work with less ceremony, and it is the
    path the decode hooks cover.
- **`LlamaContextParams::with_moe_cache_size` / `moe_cache_size`**, wrapping
  upstream's new `llama_context_params::moe_cache_size`
  ([#29887](https://github.com/ggml-org/llama.cpp/pull/29887), spread across
  devices in [#30112](https://github.com/ggml-org/llama.cpp/pull/30112)). When a
  mixture-of-experts model does not fit in VRAM and some experts stay in host
  memory, this gives llama.cpp a device-side cache of that many bytes for the
  hottest of them instead of streaming every expert across the bus on every
  token.
  - A non-zero budget is a requirement, not a hint: llama.cpp refuses to build
    a context it cannot honour the budget on, so `new_context` fails with
    [`LlamaContextLoadError::NullReturn`] unless there is a GPU backend, the
    model is a mixture of experts, the split mode is not tensor/row, and the
    budget covers one token's experts for at least one host-resident layer.
    `0` (the default) is always accepted. The docs on the setter spell the
    conditions out, and a test pins the failure so they stay honest.
- **`ChatApplyParams::with_force_pure_content`**, wrapping upstream's new
  `common_chat_templates_inputs::force_pure_content`. It renders a
  content-only template, so reasoning and tool calls stay inline in `content`
  instead of being parsed out. This is where the two per-parse flags upstream
  dropped went — see *Breaking*.
- **`ProcessError`**, the error type for [`LlamaContext::process`].
  `llama_process` returns the same codes as `llama_decode`; the extra variants
  are the context-binding, tensor-transaction and incomplete-entry checks
  described above.
- **`LlamaContext::causal_attn`**, wrapping upstream's new
  `llama_get_causal_attn`
  ([#28876](https://github.com/ggml-org/llama.cpp/pull/28876)) — the read side
  of the [`LlamaContext::set_causal_attn`] this crate already
  had. The initial value comes from the model's default rather than anything
  the caller set, so reading it back beats tracking it.
- **Any of llama.cpp's `LLAMA_*` CMake options is now configurable**, not just
  the `LLAMA_OPENSSL` knob this started as. Upstream has dozens and gains more
  every release; mirroring each as a Cargo feature does not scale and could not
  carry a value anyway, so they are configured as data from three places —
  lowest precedence first:
  1. **`llama-cpp.toml`** (`[cmake]`, plus `[cmake.target."<triple>"]` for
     per-target overrides), found by walking up from the sys crate or from
     `OUT_DIR`, or named outright by `LLAMA_CONFIG`. For a machine-local or
     gitignored override.
  2. **`[package.metadata.llama-cpp-sys-4.cmake]`** in the consuming project's
     manifest — or `[workspace.metadata.…]`, since a workspace root is often a
     virtual manifest. For a setting that belongs to the project and should be
     checked in.
  3. **A `LLAMA_*` environment variable**, which always wins, so CI and one-off
     builds override anything.
  - Values are normalised to CMake's `OFF`/`ON` when they look boolean
    (`0`/`1`, `off`/`on`, `false`/`true`, `no`/`yes`, any case); anything else
    passes through verbatim, so string and path options work too.
  - Only `LLAMA_*` is configurable, never `GGML_*`: the backend options
    (`GGML_METAL`, `GGML_CUDA`, …) are driven by this crate's Cargo features,
    which also decide what the Rust side links against, and enabling a backend
    behind the feature system's back would configure a library the link step
    does not expect. A `GGML_*` key in a config file is reported and skipped,
    as is any value that is not a string, boolean or integer.
  - The build script's own knobs (`LLAMA_PATCH`, `LLAMA_PREBUILT_*`,
    `LLAMA_LIB_PROFILE`, `LLAMA_STATIC_CRT`, `LLAMA_NO_SCCACHE`,
    `LLAMA_BUILD_SHARED_LIBS`, `LLAMA_TEST_MODEL`) are not forwarded; they name
    tools and paths, or map to a differently-named define.
  - `LLAMA_OPENSSL=0`/`"OFF"` remains the motivating case: upstream defaults it
    `ON`, so any OpenSSL CMake can find gets linked into `common` — a second
    TLS stack that applications downloading models themselves never use but
    must still ship and patch. With it off, an `https://` request through
    `common/http.h` fails with an explicit error rather than falling back to
    plain HTTP.
  - Config files are tracked with `rerun-if-changed`, and each `LLAMA_*`
    variable that is set gets a `rerun-if-env-changed`. Exporting one for the
    *first* time cannot be detected — Cargo needs the name in advance — so run
    `cargo clean -p llama-cpp-sys-4` after that, or use a config file. See the
    `llama-cpp-sys-4` README.
- **`LlamaModel::lora_adapter_init_at_offset`**, for a `LoRA` adapter GGUF
  embedded at a byte offset inside a larger file — appended to an executable,
  packed into an archive, or stored alongside other blobs.
  [`lora_adapter_init`] opens the path itself and reads from byte zero, so it
  only works for an adapter that is its own file. Also
  `LlamaModel::lora_adapter_init_from_file_ptr` (`unsafe`), the thin wrapper
  over upstream's new `llama_adapter_lora_init_from_file_ptr`
  ([#28993](https://github.com/ggml-org/llama.cpp/pull/28993)) and the
  counterpart to the existing [`LlamaModel::load_from_file_ptr`]. Opening goes
  through ggml's own `ggml_fopen`, which already handles the UTF-8 → UTF-16
  conversion a bare `fopen` gets wrong for a non-ASCII path on Windows.
- **`FILE` is re-exported at the crate root**, so the `*_from_file_ptr`
  constructors can be called without the caller adding a direct dependency on
  `llama-cpp-sys-4`. They were previously unusable from outside this
  repository for that reason.
- **`ShimError` is re-exported at the crate root.** `ChatError`, `NgramError`,
  `CommonSamplerError` and `RuntimeError` are all aliases of it — the C shims
  share a status enum and a thread-local error buffer — but the type itself
  lived in a private module, so its variants had nowhere to be documented and
  every `[`ChatError::Failed`]`-style doc link resolved to a private item.

- **`LlamaSampler::try_grammar` and `try_grammar_lazy_patterns`**, with a
  `SamplerInitError`. The panicking constructors stay — for a grammar literal, or
  one from [`ChatParams::grammar`], a failure really is a bug — but they
  `CString::new(..).unwrap()` a *caller-supplied* string, and an interior NUL
  there is bad input, not a broken invariant. The same applies to llama.cpp
  returning null, which is what an unparseable GBNF grammar looks like. Both now
  have a fallible form, continuing the migration `try_sample`/`try_accept`
  started, and the error names which argument was at fault.
- **A declared MSRV.** Both published crates now set `rust-version = "1.88"`,
  and a new `msrv` CI job builds them on exactly that toolchain. Nothing
  checked it before — every other job runs `stable` — so the day someone used a
  newer language or std feature the floor would silently become a lie and
  downstream would get a syntax error instead of *"requires rustc X"*.
  - `1.88` comes from the dependency graph, not from this crate: the `ureq`
    build-dependency pulls `url` → `idna` → `icu_*`, which declare 1.88. The
    crate's own code only needs 1.82, for `&raw const` (60 uses). The comment
    above each `rust-version` records that, and the job reads the value out of
    the manifest rather than repeating it, so the two cannot drift.
  - Only the published crates are held to it. The examples depend on
    `actix-web`, `sysinfo` and `hf-hub`, whose floors are far higher.
- **`rust-toolchain.toml`**, pinning the development toolchain. `cargo fmt
  --check` and `cargo clippy -D warnings` are both CI gates and both move with
  the compiler — rustfmt changes its output between releases and clippy adds
  lints — so with CI on `stable` and contributors on whatever they have, a gate
  can go red from a toolchain bump nobody made. This is not the MSRV; the `msrv`
  job sets `RUSTUP_TOOLCHAIN` to override the file.

### Fixed

- **Neither published crate shipped its licence texts.** `license = "MIT OR
  Apache-2.0"` is a declaration; Cargo can only package files from inside the
  package directory, and `LICENSE-MIT`/`LICENSE-APACHE` existed only at the
  workspace root. Verified against the built `.crate`: no `LICENSE*` at all. Both
  crates now carry copies, and `llama-cpp-sys-4`'s `include` list names them —
  without that the list would have excluded them again.
- **`ggml-et` and `ggml-openvino` were the only two ggml backend directories
  missing from `llama-cpp-sys-4`'s `include` list** (16 of 18 were there), so the
  published crate did not contain them. Latent rather than live: both options
  default `OFF`, so cmake never descended into them — but `ggml/src/CMakeLists.txt`
  calls `ggml_add_backend()` for both unconditionally, so the day a feature turned
  one on, configure would fail on a published crate. The same trap as
  `ggml-version.h.in`. The package grew 1800 → 1974 files.
- **Six documentation examples were wrong, and `ignore` was hiding it.** Marking
  a doc example `ignore` means rustdoc never compiles it, so it rots silently.
  Converting the self-contained ones to `no_run` immediately surfaced:
  - `LlamaSampler::chain_default` — a function that has never existed
    (`chain_simple` is the real one), in the `with_sampler_seq_configs` example.
  - A type error in the `mtp` module's setup example: `MtpSessionConfig::new`
    takes `(u32, i32)` while `with_n_rs_seq` takes `u32`, so the single inferred
    `n_draft_max` the example used could not satisfy both.
  - Two `set_nextn_layer_offset` loops that told you to call
    `draft.set_nextn_layer_offset(..)` while an `MtpSession` was alive. Since
    0.6.0 the session borrows the draft context *exclusively*, so that cannot
    compile in real use; they now go through `session.draft_context_mut()`.
  - 6 examples converted, 128 → 132 compiled doctests. The 15 that remain
    `ignore` are one-to-five-line fragments referencing ambient bindings, or
    document a `pub(crate)` function — `ignore` is the honest marker for those,
    and scaffolding them would add more hidden setup than visible example.

### Changed

- **llama.cpp**: vendored submodule updated to `5c8854fd88` (tag `b11557`) from
  `22397c31a0` (`b10881`), 676 upstream commits spanning releases `v0.4.1`,
  `v0.5.0` and `v0.6.0`. Notable in this window:
  - **`llama_batch_ext`, a second batch API**
    ([#24669](https://github.com/ggml-org/llama.cpp/pull/24669)) — now wrapped,
    see *Added*. Speculative decoding, mtmd and the server moved onto it
    ([#29385](https://github.com/ggml-org/llama.cpp/pull/29385)), then the
    examples ([#29601](https://github.com/ggml-org/llama.cpp/pull/29601)), and
    it is how a batch can mix raw tokens with embeddings on architectures that
    allow it ([#29622](https://github.com/ggml-org/llama.cpp/pull/29622)).
    `llama_batch` and `llama_decode` are untouched, so [`LlamaBatch`] and
    everything built on it is unaffected.
  - **The chat API was refactored**
    ([#30210](https://github.com/ggml-org/llama.cpp/pull/30210)): the PEG
    parser arena lost its string serialization, `common_chat_parse` now takes a
    `common_chat_input` (text plus the token each byte came from), and the
    per-parse reasoning/tool-call switches are gone. See *Breaking* and the
    shim note below.
  - **MoE experts cached on device** — now wrapped, see *Added*.
  - **Probabilistic drafting for draft-simple and MTP**
    ([#27694](https://github.com/ggml-org/llama.cpp/pull/27694)), via new
    `common_speculative_draft_params::result_q`/`temp`/`seed` and
    `common_sampler_sample_and_accept_n_rejection`. Leaving `result_q` null is
    upstream's switch for the previous greedy verification, which is what the
    MTP shim does — so [`MtpSession`] and [`Eagle3Session`] draft exactly as
    before.
  - **Speculative decoding stops accepting draft tokens at EOG**
    ([#29638](https://github.com/ggml-org/llama.cpp/pull/29638)), and n-gram
    drafts are no longer rejected at `temp > 0` after truncation
    ([#29924](https://github.com/ggml-org/llama.cpp/pull/29924)).
  - **RPC wire protocol 6 → 8** — see *Breaking*.
  - Smaller additions to `llama.h`: `llama_get_causal_attn`
    ([#28876](https://github.com/ggml-org/llama.cpp/pull/28876)) — now wrapped,
    see *Added* — plus `llama_adapter_lora_init_from_file_ptr`
    ([#28993](https://github.com/ggml-org/llama.cpp/pull/28993)),
    `LLAMA_VOCAB_TYPE_PLAMO3`
    ([#30045](https://github.com/ggml-org/llama.cpp/pull/30045)) and
    `LLAMA_VOCAB_TYPE_TEST`, which reach the raw bindings only. The bindings
    only grow.
  - New architectures — K2 Horizon dense + MoVA, Ling 3.0 VL, Maple 20B-A1B
    ternary MoE, HrmText (DFM Mimir 1B), Prism Bonsai 2 27B, MiniCPM-V 4.7,
    embeddinggemma2, glm5-next, plus the `clef`/`nimble`/`pplx-decider`/`d1`
    decision models.
  - Patches `0003`–`0005` still apply, at offsets — `common/speculative.cpp`
    moved by 267 lines and `src/llama-context.cpp` by 491.
- Three of the four C++ shims needed signature work; none of it is visible from
  Rust except where *Breaking* says so.
  - `chat_shim` no longer hands Rust a serialized parser. The arena stays in
    C++ behind a new opaque `chat_shim_parser` handle, which `apply` allocates
    only on the call that fills the buffer — the size-query call of the
    two-call protocol would otherwise leak one. [`ChatParams`] holds it in an
    `Arc`, so cloning shares the arena rather than copying it, and parsing goes
    through `common_chat_peg_parse` to pass the arena by reference instead of
    deep-copying it per call.
  - `mtp_shim` converts the caller's `llama_batch` into the `common_batch`
    `common_speculative_process` now takes, and sets the renamed `pos0`
    ([#28715](https://github.com/ggml-org/llama.cpp/pull/28715)) where it set
    `n_past`. The conversion refuses an embeddings batch, which [`LlamaBatch`]
    cannot build, rather than guessing a row width `llama_batch` does not
    carry.
  - `common_shim` follows `common_log_set_jsonl` off the per-logger struct onto
    a global ([#28586](https://github.com/ggml-org/llama.cpp/pull/28586)).
- `llama-cpp-sys-4` allowlists `GGML_MROPE_SECTIONS` so the safe wrapper can
  size M-RoPE position slices from the constant rather than hardcoding `4`, and
  gains one build-dependency, `toml_edit` (no serde), to read the new config
  files.
- **`Cargo.lock` re-resolved.** The bump to `bindgen` 0.73.2 did not build: its
  dependency ranges (`prettyplease >=0.2.7, <0.4` and `syn >=2, <4`) are
  independent, so the lock was free to pair `prettyplease` 0.2 with `syn` 3 —
  which fails to compile, because `prettyplease` 0.2 re-exports `syn` 2's
  traits. A `cargo update` resolves `bindgen` onto `prettyplease` 0.3 + `syn` 3
  and builds; the cost is that it moves unrelated crates too.
  `examples/rpc` is its own workspace with its own lock and needed the same
  treatment; it still failed to build after the root lock was fixed.

- **`clippy::undocumented_unsafe_blocks` is being adopted as a ratchet.** The
  crate has ~1100 `unsafe` blocks and had 62 `SAFETY:` comments — about 6%, which
  for an FFI binding is the largest quality gap by volume, and nothing flagged it
  because the lint is in clippy's `restriction` group. Enabling it workspace-wide
  would mean a thousand warnings and, under CI's `-D warnings`, a red build.
  - Instead, a module whose `unsafe` blocks are *all* justified carries
    `#![deny(clippy::undocumented_unsafe_blocks)]`, which makes its coverage
    permanent. Done in this release: `context/tensor_transaction.rs` (which was
    already at 15/15 and is the reference for the standard), plus `cfile.rs`,
    `llama_backend.rs`, `llama_batch.rs` and `llama_batch_ext.rs` — 39 blocks
    now locked in.
  - The justifications name the precondition and why it holds at that call site.
    Writing them surfaced one genuinely unchecked invariant worth recording:
    `LlamaBatch::add` sizes nothing against the `n_seq_max` that
    `llama_batch_init` allocated each `seq_id` row with, so passing more
    sequence ids than that writes past the row. It is now called out in the
    code.
  - The workspace `Cargo.toml` documents the ratchet, lists what is done, and
    names the remaining modules worst-first (`model.rs`, `mtmd.rs`,
    `context.rs`, `ggml.rs`, `sampling.rs`).
- **`update-toml-version.yaml` bumps every manifest, at a chosen level.** It
  listed four manifests by hand (`llama-cpp-sys-4`, `llama-cpp-4`,
  `examples/simple`, `examples/embeddings`) out of the 25 that carry the
  version, and only ever incremented the patch — so firing it left 21 manifests
  behind at the old version, with `cargo package` then failing on a requirement
  that no longer resolved. It now discovers manifests, takes `patch`/`minor`/
  `major` (or an exact version) as a `workflow_dispatch` input, refreshes
  `examples/rpc`'s separate lockfile too, fails if any manifest still mentions
  the old version, and the PR body lists what a release still needs by hand.
- **Supply-chain and API-compatibility gates.** Two new CI jobs, and
  `deny.toml`:
  - `supply-chain` runs `cargo deny check` (advisories, licences, bans,
    sources). Nothing checked any of that before, and something was hiding:
    **RUSTSEC-2026-0258**, an unbounded empty-DATA-frame flaw in `h2 0.3.27`.
    It is reachable only through `examples/server` → `actix-web 4.15` →
    `actix-http` → `h2 "^0.3.27"`, and 0.3.27 is the newest 0.3.x, so it cannot
    be fixed from this repository — it needs actix-web on h2 0.4. Neither
    published crate depends on `h2` (`cargo tree -p llama-cpp-4 -i h2` finds
    nothing). `deny.toml` records it as a dated exception with that reasoning,
    and the job prints the ignore list on every run so an entry cannot quietly
    become permanent. The licence allow-list enumerates the 16 licences
    actually present, all permissive. This job builds no C++, so it is the
    fastest signal in the workflow.
  - `semver` runs `cargo semver-checks check-release` against the published
    crate. 0.8.0 ships deliberate breaking changes; nothing verified that a
    future *patch* release does not do the same by accident, which for a 0.x
    crate is exactly what semver forbids. Non-blocking, since the baseline is
    whatever is on crates.io.
  - `android/llama-jni` was the one manifest depending on `llama-cpp-4` by path
    with no version — inconsistent with the other 22, and invisible to the
    version-bump workflow. It now pins the version like the rest.
- **The macOS and BLAS backends are built on the pull-request gate.** Both were
  only ever built by `prebuilt-llama.yml`, which fires on a `v*` tag — so an
  upstream `ggml-metal` change broke the *release*, not the pull request that
  pulled it in. The `mac` job already pays for a macOS runner and now builds and
  lints `--features metal`; `blas` joins the Linux feature matrix, needing
  nothing but `libopenblas-dev`.
- Documentation: the last six rustdoc warnings are gone (redundant explicit link
  targets in `eagle.rs` and `mtp.rs`), so `cargo doc` is clean. Together with
  the `ShimError` re-export and the examples under *Fixed*, that is 22
  documentation defects fixed in this release.

## [0.7.0] - 2026-09-09

### Added

- **`QuantizeParams::max_buf_size` / `with_max_buf_size()`**, wrapping the new
  `llama_model_quantize_params::max_buf_size`
  ([#27795](https://github.com/ggml-org/llama.cpp/pull/27795)). It caps the
  bytes of tensor rows held in memory at once (`0` = upstream's 8 GiB default),
  so a model larger than available RAM can be quantized at the cost of more
  I/O. Seeded from `llama_model_quantize_default_params()` like every other
  field on the builder.
- **New `chat` module wrapping `common/chat.h`** — llama.cpp's own chat-template,
  tool-calling and output-parsing layer, previously unreachable from Rust
  because it traffics in `std::string`/`std::vector`/`common_json` and a PEG
  parser arena. Added `chat_shim`, a third C shim alongside `ext_shim` and
  `mtp_shim`.
  - `ChatTemplates::from_model()` / `apply()` render an OpenAI-shaped request
    into a prompt **plus its sampling constraints**: `ChatParams::grammar`,
    `grammar_lazy`, and `grammar_triggers`. The lazy path is the point — a
    grammar applied from token zero stops a thinking model ever opening its
    `<think>` block, whereas a lazy grammar stays dormant until a trigger such
    as `<tool_call>` fires.
  - `ChatParams::parse()` reads model output back into an OpenAI-shaped message
    with `content`, `reasoning_content` and `tool_calls`, using the parser that
    template produced rather than scraping for a fixed marker. `is_partial`
    tolerates a truncated tail for streaming.
  - `ToolChoice` (`Auto`/`Required`/`None`, plus `parse_oaicompat`),
    `ReasoningFormat`, `GrammarTrigger`, `ChatTemplates::caps_json()`,
    `format_name()`, `parse_messages_oaicompat()`, `parse_tools_oaicompat()`.
  - Note that at `b10881` upstream only emits a GBNF grammar for templates it
    *recognises* (the per-family parsers under `common/parsers/`); an
    unrecognised template falls through to the generic autoparser, which
    constrains via PEG instead and leaves `grammar` empty.
- **`GgmlContext::sized_for()`, `used_mem()`, `mem_size()`, `free_mem()`.**
  The ggml context takes a raw `mem_size` byte count with no guidance, and
  running out is not recoverable — see *Fixed*. `sized_for(n_tensors, n_graphs)`
  computes the pool from `tensor_overhead()`/`graph_overhead()` instead of
  guessing, and the accessors let a caller check headroom while building a
  graph rather than discovering the limit by hitting it.
- **New `ngram` example**: draft-model-free speculative decoding end to end.
  `--verify` runs the same prompt with and without drafting and asserts the
  outputs are byte-identical, which is speculative decoding's losslessness
  guarantee — so the demo doubles as a check. On a repetitive prompt with the
  tiny CI checkpoint it produces 32 tokens in 23 target forward passes instead
  of 32, at a 47% draft-acceptance rate.
- **`examples/structured` accepts a JSON Schema** (`--json-schema`,
  `--json-schema-file`), converted to GBNF by llama.cpp. It previously only
  took hand-written GBNF, which is the gap `json_schema_to_grammar` was added
  to close.
- **`examples/eagle` and `examples/mtp` probe the checkpoint before loading
  it**, via `speculative_types_from_gguf`. A model that does not advertise the
  needed strategy now fails immediately with a specific message instead of
  after a multi-gigabyte load. A checkpoint advertising nothing is not
  rejected — older conversions predate the metadata key.
- **`LlamaSampler::try_sample` / `try_accept`**, returning errors where
  `sample` / `accept` panic. See *Fixed*.
- **New `common_sampler` module** wrapping `common_sampler` — the sampler chain
  llama.cpp assembles for its own tools, as opposed to the individual
  `llama_sampler_*` primitives `LlamaSampler` exposes. It gets right several
  things that are easy to miss by hand: sampler ordering, grammar prefill,
  merging model-declared `suppress_tokens` into the logit bias, and creating the
  reasoning-budget sampler when a lazy grammar is active. `CommonSamplerParams`
  splits the ~35 POD knobs into a plain struct and the container-backed fields
  into methods, so an upstream bump that adds a field does not break the
  signature. Includes `CommonSampler::sample_and_accept_n`, the speculative
  acceptance path every caller previously had to re-implement around an
  `Eagle3Session`/`MtpSession`.
- **`ReasoningBudget`**, wrapping `common_reasoning_budget_*`. Caps how many
  tokens a model may spend inside a `<think>` block and force-closes it when the
  budget runs out. This is the piece llama.cpp pairs with a *lazy* grammar:
  the grammar leaves reasoning unconstrained by design, so something else has to
  bound it — without it a thinking model with a tool grammar can reason until
  the context ends.
- **New `ngram` module: speculative decoding with no draft model.** Three
  strategies, all drafting by looking up where the token history repeats —
  `ngram_simple_draft` (stateless), `NgramMap` (adapts to how its own drafts
  land), and `NgramCache` (statistical, save/loadable, consulted as
  context/dynamic/static tiers). Costs a hash lookup per step instead of a
  forward pass and needs no extra weights, which is why it wins on code editing,
  RAG over quoted text and repetitive JSON. Pairs with
  `CommonSampler::sample_and_accept_n` for verification.
- **New `runtime` module**:
  - `speculative_types_from_gguf()` reads which speculative strategies a draft
    checkpoint supports **without loading it** — a metadata read instead of a
    multi-gigabyte load followed by a failed session construction. Plus
    `SpeculativeType` name round-tripping.
  - `runtime::log` — llama.cpp's own logger: verbosity, timestamps, prefix,
    colours, JSONL, file output, pause/resume. Distinct from
    [`log_set`](crate::log_set), which installs a callback.
  - `runtime::download` — resolve Hugging Face `repo[:tag]` and Docker
    references through **llama.cpp's own cache**, so a model pulled by
    `llama-cli` is found here and vice versa. Plus `split_repo_tag`,
    `remove_cached` and `list_cached_json`.
- **mtmd**: `MtmdContext::tokenize_from_parts()` (explicit interleaving with
  per-part `parse_special`, so a media marker in user text is just text),
  `mmproj_caps()` (probe a projector's modalities without loading it),
  `MtmdContext::gen_audio_info()`, `MtmdContext::model_can_chat()`,
  `MtmdInputChunk::to_owned_chunk()`, and `MtmdLazyBitmap` — media produced on
  demand during tokenization, for video too large to materialise up front or
  that a stop sequence may never reach.
- **`ChatParams::grammar_sampler()`**, which builds the correctly-configured
  sampler for a render. Three things must line up and each is silently wrong on
  its own: lazy-vs-eager construction, trigger translation (literal triggers
  need regex-escaping, `pattern_full` needs anchoring, token triggers are not
  patterns at all — see `ChatParams::sampler_triggers`), and **generation-prompt
  prefill**. llama.cpp writes tool-call grammars to match
  `generation_prompt + output`, because that is what its parser later sees; the
  sampler only sees `output`, so without advancing the grammar past that prefix
  the model is forced to *re-emit* `<|im_start|>assistant` as generated text.
  Also adds `ChatParams::generation_prompt()`.
- **`chat::json_schema_to_grammar()`**, wrapping `json_schema_to_grammar` — the
  missing half of structured output. Feed the result to `LlamaSampler::grammar`
  and the model *cannot* emit anything the schema rejects. This is what
  `response_format: json_schema` needs; until now the `structured` example could
  only take hand-written GBNF.
- **`QuantPreview` and `QuantModelDesc`** (`quantize`), wrapping the
  `llama_quant_*` API: ask which tensors would be quantized and to which `ggml`
  type, without writing a file. The k-quant mixes keep attention and output
  tensors at higher precision, so the per-tensor answer differs from the ftype's
  nominal type — which is exactly what a size estimate or a `--dry-run` plan
  needs. `QuantModelDesc` builds a synthetic model from metadata, so no
  checkpoint is required. The tensor-facing methods need the `ggml` feature.
- **`LlamaModel::chat_template(name)`**, wrapping `llama_model_chat_template`.
  Reaches *named* templates — `Some("tool_use")` resolves
  `tokenizer.chat_template.tool_use` — which the existing `get_chat_template`
  could not, since it reads the default GGUF key directly.
- **`LlamaModel::ftype()`**, **`LlamaFtype::upstream_name()`** and
  **`LlamaFtype::default_ggml_type()`**, plus `TryFrom<llama_ftype> for
  LlamaFtype`. `upstream_name` asks llama.cpp (`"Q4_K - Medium"`) rather than
  this crate's filename-safe table (`"Q4_K_M"`), so it cannot drift as upstream
  adds types.
- **`LlamaModel::token_embeddings()`**, wrapping `llama_model_get_tok_embd` —
  the whole embedding matrix as `f32`, converted from whatever it is stored as.
- **`LlamaVocab::suppress_tokens()`**, wrapping
  `llama_vocab_get_suppress_tokens`: tokens the model declares must never be
  sampled (`tokenizer.ggml.suppress_tokens`). Feed to `LlamaSampler::logit_bias`.
- **`LlamaLoadMode::name()` / `from_name()`**, the string round-trip upstream's
  `--load-mode` flag uses.
- **mtmd: chunk persistence** — `MtmdInputChunk::save()`,
  `MtmdInputChunks::load_chunk()`, `MtmdInputChunk::to_placeholder()` and the
  owning `OwnedMtmdInputChunk`. Metadata only, which is what lets a restored KV
  cache line back up with the prompt that produced it; pairs with
  `state_seq_save_file`.
- **mtmd: `MtmdBatch`**, wrapping `mtmd_batch_init`/`add_chunk`/`encode` — runs
  the vision encoder once over a multi-image prompt or a run of video frames
  instead of once per image.
- **mtmd: `MtmdAudioGen`**, wrapping `mtmd_helper_gen_audio_*` — audio
  *generation*, a modality that was entirely unbound. Explicit `set_input` →
  `step_prompt` → `step_gen` → `output` loop, mirroring upstream's stateless
  design.
- **mtmd: `MtmdBitmap::set_mergeable()`**, needed when you build video frames
  yourself with `from_rgb`; frames from `MtmdVideo::read_next` already have it
  set by the helper.
- **`LlamaModelParams::with_lazy_mode()` / `lazy_mode()` and the `LlamaLazyMode`
  enum**, wrapping the new `llama_model_params::lazy_mode`
  ([#27794](https://github.com/ggml-org/llama.cpp/pull/27794)). `Off` always
  reads whole tensors up front, `Auto` (upstream's default) reads lazily only
  for arch-marked tensors above 4 GiB, and `On` does so for every marked tensor
  regardless of size. Only architectures that flag tensors `TENSOR_READ_LAZY`
  are affected — currently Gemma-4's per-layer token embedding and Qwen4Exp's
  PLE rows — and lazy reads require mmap, so llama.cpp warns and loads in full
  without it. The getter reports the *requested* mode: upstream resolves `Auto`
  to `Off` during load when a device lacks mmap support (iGPUs,
  [#28326](https://github.com/ggml-org/llama.cpp/pull/28326)) without writing
  that back.
- **`DFlashSession`, `Eagle3Session::new_dflash()` and `new_dflash_with_config()`
  are now unconditional.** They were behind the `dflash2` feature only because
  the C++ came from an unmerged PR; that PR is now upstream, so the gate is
  gone. Existing callers that enabled the feature keep working — they just no
  longer need it.

### Fixed

- **The published crate did not build.** `ggml/src/ggml-version.h.in` was
  missing from the package. `llama-cpp-sys-4`'s `include` list covers
  `ggml/src` by extension (`*.h`, `*.c`, `*.cpp`) rather than with a recursive
  glob, so when upstream started generating a header from a template
  ([#28364](https://github.com/ggml-org/llama.cpp/pull/28364), inside the
  `b10502`→`b10881` range) the input was silently dropped and CMake failed with
  `File .../ggml-version.h.in does not exist`. The workspace build was
  unaffected — it compiles the git checkout, which has the file — so only
  `cargo package` could reveal it. `ggml/src/*.in` and `LICENSE` are now
  shipped, and a **new CI job runs `cargo package` with its verification
  build**, which is the check that was missing: nothing previously compiled the
  artifact crates.io would receive.
- **A partial copy of the llama.cpp source tree could poison an `OUT_DIR`
  permanently.** `build.rs` copies the submodule into `OUT_DIR` with `cp -rf`
  (or `robocopy`) and then writes a version sentinel next to it; later builds
  skip the copy when that sentinel matches. But the copy's exit status was
  never checked, so a `cp` that failed part-way — which is what a concurrent
  `git checkout` of the submodule causes — was accepted, the sentinel written
  over it, and the truncated tree reused from then on. The symptom is a CMake
  error about a missing `LICENSE` or `ggml/src/ggml-version.h.in`, which says
  nothing about copying and is only cleared by `cargo clean`.

  The copy now (a) checks the exit status, allowing for `robocopy`'s 0–7
  success range, (b) stages into a temporary sibling and renames into place
  only on success, so the destination is either absent or complete, and
  (c) asserts that the `configure_file` inputs CMake hard-fails without
  actually arrived. A failure now names the real cause and leaves no sentinel,
  so the next build retries. `LICENSE` is deliberately not in that list —
  `cmake/license.cmake` only warns, and demanding it would reject the
  legitimately-filtered `cargo package` tree.

- **Two llama.cpp entry points throw C++ exceptions that would abort the
  process** if called directly from Rust — unwinding across `extern "C"` is
  undefined behaviour, and in practice yields `fatal runtime error: Rust cannot
  catch foreign exceptions`. Both are now reached through guarded wrappers:
  - `llama_quant_model_from_metadata` and `llama_quant_init` throw for input
    they reject (an unknown architecture, a model they cannot quantize); they
    are wrapped in `ext_shim` and return null instead.
  - `llama_load_mode_from_str` throws `std::invalid_argument` for an
    unrecognised string, so `LlamaLoadMode::from_name` deliberately does *not*
    call it — it compares against `llama_load_mode_name` output instead, which
    uses upstream's own strings and cannot throw.
  Everything in `chat_shim` is guarded the same way, since `common/chat.h`
  throws on malformed JSON, unparseable templates and unknown `tool_choice`.
- **Documented that exhausting a `GgmlContext` pool is unrecoverable, and
  gave callers the means to avoid it.** `ggml.c:1735` warns, then
  `GGML_ABORT`s under `#ifndef NDEBUG` and returns null otherwise — so a debug
  build of ggml kills the process and a release build returns null, which this
  crate's constructors turned into a bare `.expect()`. 22 functions now carry a
  `# Panics` section saying so, and [`GgmlContext::sized_for`] exists so the
  situation is avoidable rather than merely documented.
- **An unsatisfiable grammar aborted the process.** `llama_sampler_accept`
  reaches llama.cpp's grammar code, which *throws* `"Unexpected empty grammar
  stack after accepting piece"` when a model's vocabulary cannot satisfy the
  constraint — a JSON schema against a vocabulary with no `{`, for instance.
  `LlamaSampler::accept` and `sample` called it directly, so that C++ exception
  unwound into Rust and killed the process with `fatal runtime error: Rust
  cannot catch foreign exceptions`. Both now route through a guard: they panic
  with llama.cpp's own message, and `try_accept` / `try_sample` return it as an
  error instead. Found while wiring `--json-schema` into `examples/structured`,
  which reproduced it on the CI checkpoint.
- **`examples/server` silently overrode an explicit `max_tokens`.** With tools
  present it raised anything below 1024 to 1024 — but `parse_max_tokens`
  already *defaults* to 1024, so that could only ever fire on a value the
  caller had explicitly asked for, handing a request capped at 32 tokens 32x
  the cost. Removed; the default still gives thinking models room. This also
  cut the server integration suite from 30s to under a second.
- **`examples/server`'s test harness silently ran against stale binaries.** It
  leaks the server child by design (`std::mem::forget`), so a process from an
  earlier run stays bound to the test port; the next run's child then dies on
  bind while `/health` keeps answering from the *old* binary, and tests pass or
  fail against code no longer on disk. `start_server` now polls `try_wait()`
  and fails with an actionable message instead.

### Changed

- **CI now builds *and lints* `ggml`, `q1` and `--no-default-features`.** The
  matrix was `mtmd` and `rpc` only, so code behind any other feature was never
  compiled there — which is exactly how `src/ggml.rs` accumulated 47 clippy
  warnings unnoticed. The feature-combo jobs now run `clippy -D warnings` as
  well as `build`, and run the `ggml`/`q1` test files, which nothing else did.
  All 47 warnings are fixed.
- **Internal: the four C shims are deduplicated.** `build.rs` had four
  byte-identical 24-line `compile_*_shim` functions; they are now one
  `compile_shims()` building every shim into a single library. `chat_shim` and
  `common_shim` each carried their own copy of the same `guard`/`emit`/error
  boilerplate — now in `shim_support`, which also means **one** thread-local
  error buffer rather than two that could disagree about which failure was
  most recent. Their status enums are unified too: `CHAT_SHIM_BAD_JSON` was
  `-2` while `COMMON_SHIM_THROWN` was also `-2`, so a shared status mapping
  would have reported one as the other. On the Rust side, `ChatError` and
  `CommonSamplerError` are now aliases of one `ShimError`, over one set of
  `check_status` / `read_string` / `read_tokens` helpers.
- **`examples/server` now drives tool calling through `llama_cpp_4::chat`**
  instead of hand-rolling it. `tools.rs` drops from 526 lines of logic to 448,
  and the ~230 of those that injected a Hermes `<tools>` block into the system
  prompt and then scanned output for `<tool_call>` markers are gone entirely
  (the remaining growth is tests: 15 → 27). What changes behaviourally:
  - **Tool-call syntax is now per model family.** The old scraper only
    understood Hermes; Functionary (`>>>name`), DeepSeek and the rest silently
    produced no tool calls. llama.cpp picks the parser from the template.
  - **`tool_choice: "required"` is enforced by grammar, not by asking.** The
    old code explicitly gave up on grammar forcing — its comment noted GBNF
    "prevents thinking models from emitting their reasoning prefix", which is
    true of an *eager* grammar. Lazy grammars solve exactly that, so the
    constraint is now real rather than a prompt suggestion.
  - **`tool_choice: {"type":"function",…}`** narrows the tool list to that
    function and asks for `Required`, since `common_chat_tool_choice` cannot
    name one. Naming a function absent from `tools` is a 400 rather than a
    grammar referencing nothing.
  - **Request validation is llama.cpp's**: `tools` and `messages` go through
    `common_chat_tools_parse_oaicompat` / `common_chat_msgs_parse_oaicompat`,
    so anything the server accepts is something the template can render.
  - **Reasoning is split into `reasoning_content` for every chat request**, not
    only ones carrying tools — the old code parsed output only when tools were
    present.
  - `response_format.json_schema` is now honoured, through the same grammar
    path. A per-request `chat_template` still works, building a one-off
    template set.
  - The two end-to-end `tool_calling_*` integration tests are now `#[ignore]`d
    with a reason: the grammar for `required` is `<preamble>? tool-calls` with
    an *unbounded* preamble, so it guarantees a well-formed call eventually,
    not promptly — and the 260K-parameter CI checkpoint rambles until
    `max_tokens`. Six tests that the tiny model *can* verify replace them,
    including a deterministic guard against the generation-prompt prefill
    regression.
- **llama.cpp**: vendored submodule updated to `22397c31a0` (tag `b10881`) from
  `0adcc3bb5` (`b10502`), 379 upstream commits spanning releases `v0.2.0`,
  `v0.3.0` and `v0.4.0`. Notable in this window:
  - **DFlash2 merged upstream**
    ([#27342](https://github.com/ggml-org/llama.cpp/pull/27342) via
    [#27816](https://github.com/ggml-org/llama.cpp/pull/27816), landing in
    `b10658`) — see *Removed*.
    Follow-ups fuse the DFlash encoder into KV-cache injection
    ([#27310](https://github.com/ggml-org/llama.cpp/pull/27310)) and fix NVFP4
    scales for attention ([#28000](https://github.com/ggml-org/llama.cpp/pull/28000)).
  - **Lazy tensor loading** (`llama_lazy_mode`, `TENSOR_READ_LAZY`,
    [#27794](https://github.com/ggml-org/llama.cpp/pull/27794),
    [#27837](https://github.com/ggml-org/llama.cpp/pull/27837)): read rows of
    arch-marked tensors on demand instead of up front. Defaults to
    `LLAMA_LAZY_MODE_AUTO`, and is disabled on iGPUs
    ([#28326](https://github.com/ggml-org/llama.cpp/pull/28326)). Surfaced as
    [`LlamaModelParams::with_lazy_mode`] — see *Added*.
  - **mtmd video input**: webp via ffmpeg
    ([#27520](https://github.com/ggml-org/llama.cpp/pull/27520)), `--video-*`
    arguments ([#24318](https://github.com/ggml-org/llama.cpp/pull/24318)),
    and video IDs propagated to bitmaps
    ([#28601](https://github.com/ggml-org/llama.cpp/pull/28601)).
  - New architectures — Qwen3.8-Flash-Next, Tencent Hy 4, Spark2_5,
    NemotronHPuzzle, Kimi-K3 recurrent-state rollback, DSpark for Nemotron3.5,
    plus MTP for GLM-4.5-Air
    ([#26534](https://github.com/ggml-org/llama.cpp/pull/26534)).
  - `ggml_prec` gains `BF16`/`F16`/`Q8`/`Q4` levels; `ggml_mul_mat_set_prec`
    and `ggml_flash_attn_ext_set_prec` are deprecated in favour of
    `ggml_prec_set_acc()`. Nothing was removed, so the generated bindings only
    grow.
  - Patches `0003`–`0005` apply unchanged.
- **Saved context state from 0.6.1 will not load.** Upstream bumped
  `LLAMA_SESSION_VERSION` 9 → 10 and `LLAMA_STATE_SEQ_VERSION` 2 → 3, and the
  loader requires an exact version match, so files written by
  `state_save_file`, `state_seq_save_file` or `save_session_file` under 0.6.1
  are rejected by this build. Regenerate them.
- Three C signatures the crate calls grew a parameter; all three are threaded
  through with the upstream default, so behaviour is unchanged:
  - `common_fit_params` takes a `const common_fit_extra_model *` for a second
    model sharing the main model's devices
    ([#27496](https://github.com/ggml-org/llama.cpp/pull/27496)).
    [`fit_params`] passes `nullptr`; fitting a draft model alongside the target
    is not yet exposed.
  - `mtmd_helper_bitmap_init_from_file` / `_from_buf` take an
    `mtmd_helper_init_opt`, which carries the video-decode settings now that
    upstream can route webp through ffmpeg. [`MtmdBitmap::from_file`] and
    [`MtmdBitmap::from_buf`] pass `mtmd_helper_init_opt_default()` and are not
    yet parameterised by it; deliberate video input still goes through
    [`MtmdVideo`] with [`MtmdVideoParams`].
  - Many `mtmd` entry points took `const` pointers
    ([#28307](https://github.com/ggml-org/llama.cpp/pull/28307)). Rust's
    `*mut T` → `*const T` coercion absorbs this; no call site changed.

### Removed

- **BREAKING** — **the `dflash2` feature is retired** from both crates, along
  with `patches/0006-dflash2.patch`. It existed only to vendor unmerged PR
  [#27342](https://github.com/ggml-org/llama.cpp/pull/27342); `b10881` ships
  every DFlash2 GGUF KV key (`dflash.conv_group_size`, `dflash.selector_rank`,
  `dflash.selector_top_k`, `dflash.sample_from_anchor`), the convolution and
  candidate-selector graph, and the `common/speculative.cpp` integration — so
  the patch is redundant and no longer applies. A `Cargo.toml` naming
  `features = ["dflash2"]` now fails to resolve; delete the entry. The Rust API
  it gated is unaffected and is now always available (see *Added*).

## [0.6.1] - 2026-08-19

### Added

- **`dflash2` feature (opt-in, off by default): DFlash2 speculative decoding**,
  vendored from the **unmerged** upstream PR
  [#27342](https://github.com/ggml-org/llama.cpp/pull/27342) as
  `patches/0006-dflash2.patch`. Enabling it stages that patch after `0003`–`0005`
  (which also touch `common/speculative.cpp`) and unlocks the Rust entry points.
  Caveats, since this is pre-merge upstream code:
  - Only the C++ the library needs is vendored — `common/` and `src/` (13 of the
    PR's 20 files). The Python side (`gguf-py/`, `conversion/qwen.py`) is
    deliberately excluded because it is not part of the published crate, so this
    build can **run** a DFlash2 checkpoint but not **convert** one.
  - The patch adds GGUF KV keys and tensors under the existing `LLM_ARCH_DFLASH`
    architecture, so a `dflash2` build recognises checkpoints that stock
    llama.cpp releases do not.
  - Upstream may still change the PR. When it merges, drop the patch and retire
    the feature.
- **`Eagle3Session::new_dflash()` / `new_dflash_with_config()`** and the
  `DFlashSession` alias (both behind `dflash2`), plus `MTP_SPEC_TYPE_DFLASH` in
  the shim mapping to `COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH`. DFlash reuses the
  EAGLE-3 session type because the drafting protocol is identical through the
  shim — only construction differs — so this adds two constructors rather than
  duplicating a 768-line session. DFlash2 checkpoints are detected from GGUF
  metadata and need no distinct speculative type.
- Context validation split into a shared `validate_contexts_common` (context
  types, sequence and batch capacity) and the EAGLE-3-only requirement that the
  draft model name exactly three target-extraction sites — which DFlash drafts
  do not have, and which would otherwise reject every DFlash draft model.
- **`LlamaSampler::copy_state_from()`**, wrapping `llama_sampler_copy` (added
  upstream in `b10470`). Where `clone_sampler` allocates, this overwrites an
  existing sampler's state in place — the cheap way to rewind to a checkpoint in
  a loop. Upstream requires both samplers to be the same type and configuration;
  that is the caller's contract and is documented rather than checked.

### Changed

- **llama.cpp**: vendored submodule updated to `0adcc3bb5` (tag `b10502`) from
  `34af94cd9` (`b10470`), 32 upstream commits. This includes release `v0.1.2`
  (`1511ce3bc`) plus 17 later commits — notably RPC `use_count` population to
  enable backend fusion ([#27142](https://github.com/ggml-org/llama.cpp/pull/27142)),
  shared thread pools when `n_threads` differ
  ([#27138](https://github.com/ggml-org/llama.cpp/pull/27138)), per-layer weight
  eviction to cut quantization memory
  ([#22877](https://github.com/ggml-org/llama.cpp/pull/22877)), and mtmd fixes
  for DeepSeek-OCR and LFM2 tiling.
- No public API changed: `llama.h`, `common/speculative.h`, and `common/common.h`
  are byte-identical across the bump. The only header changes are the RPC
  protocol minor version (`5.0.0` → `5.1.0`), a new additive
  `mtmd_input_chunk_get_placeholder()`, and a comment noting `mtmd_helper`
  bitmap IDs are now SHA-256 rather than FNV. Patches `0003`–`0005` apply
  unchanged.

## [0.6.0] - 2026-08-17

### Changed

- **llama.cpp**: vendored submodule updated to `34af94cd9` (tag `b10470`, also
  tagged `v0.1.1` — upstream adopted [semantic versioning](https://github.com/ggml-org/llama.cpp/blob/master/docs/release.md)
  in this window and `v0.1.1` is its newest release) from `221f0f635` (b10235),
  pulling in 235 upstream commits — multi-output backend
  sampling ([#25532](https://github.com/ggml-org/llama.cpp/pull/25532)),
  speculative-type auto-detection from draft-GGUF metadata
  ([#26814](https://github.com/ggml-org/llama.cpp/pull/26814),
  [#27005](https://github.com/ggml-org/llama.cpp/pull/27005)), backend sampling
  for dflash + dspark ([#26958](https://github.com/ggml-org/llama.cpp/pull/26958)),
  the mtmd audio-generation API, and new model architectures (BailingMoE3,
  MiniMax, Granite-Switch, Muse Glimmer, GLM-4.7-Flash MTP).
- **BREAKING** — `LlamaSampler::penalties` / `LlamaSampler::penalties_simple` take
  a leading `n_vocab: i32` argument. Upstream moved `n_vocab` out of
  `llama_sampler_data` into the penalty sampler itself
  ([#26520](https://github.com/ggml-org/llama.cpp/pull/26520)), so
  `llama_sampler_init_penalties` now needs it explicitly. Pass
  [`LlamaModel::n_vocab`], as `mirostat` already required. `penalty_last_n = -1`
  ("context size") is gone upstream; only `0` disables the penalty.
- **BREAKING** — `LlamaSampler::dry` no longer takes `n_ctx_train`: upstream
  dropped the parameter from `llama_sampler_init_dry`.
- **BREAKING** — `common_sampler_params` (`llama-cpp-sys-4`) resynced with
  upstream `common_params_sampling`, which it hand-mirrors and had drifted far
  from. Removed `tfs_z` and `penalize_nl` (both deleted upstream); added
  `top_n_sigma`, `adaptive_target`, `adaptive_decay`, and `timing_per_token`;
  `dry_penalty_last_n` now defaults to `64` rather than `-1` (which is no longer
  a valid "context size" sentinel). The default `samplers` chain now matches
  upstream — it previously still listed the **removed TFS-Z sampler**.
  Correspondingly `COMMON_SAMPLER_TYPE_TFS_Z` is gone and
  `COMMON_SAMPLER_TYPE_PENALTIES`, `_TOP_N_SIGMA`, and `_ADAPTIVE_P` are added.
- **BREAKING** — `LlamaLoadMode` gained an `Auto` variant (`LLAMA_LOAD_MODE_AUTO`,
  `-1`) and is now `#[repr(i32)]` on every target rather than `#[repr(u32)]`,
  because the enum is signed everywhere once it carries a negative discriminant.
  `Auto` is llama.cpp's new default: it memory-maps unless one of the backend
  devices lacks mmap support. `LlamaModelParams::use_mmap()` reports `true` for
  `Auto`, so the default-parameters behaviour is unchanged.
- **Patch `0003` (exact speculative state) rebased**: upstream removed the
  `need_embd()` / `need_embd_nextn()` virtuals from `common_speculative_impl`
  ([#26904](https://github.com/ggml-org/llama.cpp/pull/26904)), which those hunks
  used as context. Patches `0004` and `0005` apply unchanged.
- `mtp_shim` no longer calls the deleted `common_speculative_need_embd` /
  `common_speculative_need_embd_nextn`. `MtpSession::need_embd` /
  `Eagle3Session::need_embd` still report `false` and `need_embd_pre_norm` still
  reports `true` only for MTP sessions, matching what upstream returned.

### Added

- **`LlamaContextParams::with_n_outputs_max_per_seq()` / `n_outputs_max_per_seq()`**
  (`llama-cpp-4`), wrapping the new `llama_context_params.n_outputs_max_per_seq`.
  Backend samplers are initialized for this many outputs per sequence, so
  multi-output backend sampling must raise it above its default of `1`.
- **`llama_version()`** (`llama-cpp-4`): the vendored llama.cpp version string.
  Now that upstream ships semver releases this is the direct way to report which
  upstream a binary carries, since the crate and llama.cpp versions move
  independently.
- **Tests for the APIs this release changes**, which previously had none:
  `LlamaLoadMode` (the `Auto` default, round-tripping every variant, and that the
  negative discriminant survives — the exact failure a `#[repr(u32)]` would
  reintroduce), `penalties` / `penalties_simple` construction and behaviour,
  `dry`, and `n_outputs_max_per_seq`.
- **CI job `feature-combos`**: PRs previously only ever built the default
  `openmp,mtmd,dynamic-link` set, so a break under any other feature stayed
  hidden until a release tag fired `prebuilt-llama.yml`. The new job builds
  static `mtmd` (no `dynamic-link`) and `rpc` — the two feature-gated bindgen
  paths. The `mtmd` bindgen collision fixed in this release was caught only
  because `mtmd` happens to be on by default; adding this job immediately
  surfaced that `--features rpc` had not compiled for some time (see below).
- `GGML_RPC_*` constants are now in the `rpc` bindgen allowlist, so
  `GGML_RPC_MAX_SERVERS` is available to validate device lists.

### Fixed

- **`mtmd` bindings no longer fail to compile**: upstream's new
  `namespace mtmd_helper` C++ RAII wrappers flatten to the same `mtmd_helper_*`
  names as the C API under bindgen without cxx-namespaces — `mtmd_helper::gen_audio`
  collided with the opaque C `struct mtmd_helper_gen_audio`. The `mtmd_helper::`
  namespace is now blocklisted; it was never usable from Rust.
- **Dynamically linked binaries now run when executed directly.** With the
  default `dynamic-link` feature, CMake stamped every llama/ggml dylib with an
  `@rpath/…` install name, which rustc recorded in the final binary — but Cargo
  never adds an `LC_RPATH`, so running a built binary yourself died with
  `Library not loaded: @rpath/libggml-base.0.dylib … no LC_RPATH's found`. This
  affected **every** binary built from this crate, not just the examples; it was
  masked because `cargo run` and `cargo test` set
  `DYLD_FALLBACK_LIBRARY_PATH` / `LD_LIBRARY_PATH` to the target directory.
  The build script now rewrites the dylib install names and their sibling
  references to `@loader_path/…`, which needs no rpath and works for downstream
  consumers automatically. ELF cannot be fixed the same way — the rpath must sit
  on the final executable, which a dependency's build script cannot inject — so
  Linux/BSD get `-Wl,-rpath,$ORIGIN` via a new [`.cargo/config.toml`], with the
  requirement documented for downstream users in the README. Windows was already
  fine. Note `RUSTFLAGS` *replaces* config rustflags, so the CI jobs that set it
  now repeat the rpath. The rewrite is applied on the prebuilt-archive path too,
  not just the CMake build — prebuilt libraries carry the same `@rpath/…` names.
  CI now executes a built binary directly, since `cargo run` / `cargo test`
  inject the library path and therefore can never catch this class of bug.

[`.cargo/config.toml`]: .cargo/config.toml

- **CI `Fmt` and `Clippy` steps now actually gate.** `Fmt` ran bare `cargo fmt`,
  which rewrites files inside the runner and always exits `0`, so formatting was
  never enforced and the tree had accumulated drift; it is now
  `cargo fmt --all -- --check`. `Clippy` ran bare `cargo clippy`, which skipped
  tests, benches, and examples and treated findings as non-fatal; it is now
  `cargo clippy --workspace --all-targets -- -D warnings`. The workspace was
  formatted and its clippy findings resolved to make both steps pass.
- **`examples/common.rs`** documented output regenerated from an actual run; it
  still advertised `tfs_z`, `penalize_nl`, and `dry_penalty_last_n: -1`.

- **The `rpc` feature compiles again.** `llama-cpp-4`'s `rpc` module had drifted
  out of sync with `ggml-rpc.h` and failed to build with five errors; nothing in
  CI ever compiled it, and `--features rpc` is not reachable from a default
  build, so it went unnoticed. Note `ggml-rpc.h` is **unchanged** in this
  llama.cpp bump — the breakage predates it. Upstream's RPC API is device-aware
  (one endpoint can expose several devices), so:
  - `RpcBackend::init` takes a `device: u32`, and `buffer_type` /
    `get_device_memory` use it. Added `RpcBackend::device()`, and `as_ptr()` is
    now public so the handle is actually usable.
  - `RpcServer` is replaced by a blocking `rpc::serve(endpoint, cache_dir,
    n_threads, devices)` matching `ggml_backend_rpc_start_server`. The old
    `RpcServer::start(backend, endpoint, free_mem, total_mem)` did not
    correspond to any current upstream signature, and returned a handle for a
    call that never returns.
  - `add_rpc_device` becomes `add_rpc_server`, returning a
    `ggml_backend_reg` — upstream renamed it and changed its return type from a
    device to a registration.
- **`examples/rpc` builds again**, and is now the working consumer that proves
  the API above. It was previously listed in neither `workspace.members` nor
  `workspace.exclude`, so cargo rejected it standalone ("current package
  believes it's in a workspace when it's not") and the workspace never compiled
  it — which is how it and the `rpc` module drifted unnoticed. It now declares
  its own `[workspace]`: making it a root member would unify the `rpc` feature
  into every workspace build, forcing all contributors to compile llama.cpp with
  RPC support. It gained `--device`, `--cache-dir`, and `--threads` flags, and
  enumerates devices from ggml's backend registry. Verified by actually running
  it: the server reports `device 0: CPU` and blocks serving.

### Documented

- **`LlamaSampler::name()`** now explains llama.cpp's `?`-prefix convention:
  handed parameters that make it a no-op (`temp(1.0)`, `penalties` with
  `penalty_last_n = 0`, …), llama.cpp silently substitutes an identity sampler
  named `"?temp"` / `"?penalties"`. Construction still succeeds, so the name is
  the only signal the sampler does nothing. Long-standing upstream behaviour
  (unchanged in this bump), but previously undocumented here.

## [0.5.1] - 2026-08-03

### Added

- **Android example** (`android/`): a Rust JNI `cdylib` (`llama-jni`) that exposes
  on-device text generation to a minimal Gradle/Kotlin app, plus an `aarch64`
  `smoke` binary sharing the same `generate()` core. Verified end-to-end on real
  arm64 — natively on Linux (via a native `linux/arm64` container) and as an
  Android NDK cross-build.
- **Prebuilt CI for arm64**: `prebuilt-llama.yml` now also builds Linux
  `aarch64-unknown-linux-gnu` (native arm runners) and Android
  `aarch64-linux-android` (NDK cross-compile) library artifacts.
- **Native arm64 CI** (`arm64-smoke.yml`): builds and runs the `llama-jni`
  `smoke` binary on a native arm64 Linux runner against the tiny test model,
  guarding the arm64 code path.

### Fixed

- **ARM64 / Android build** ([#306](https://github.com/eugenehp/llama-cpp-rs/issues/306)):
  `quantize.rs` used `.cast_signed()` (`i8`) where `c_char` is `u8` on ARM64 /
  Android, breaking compilation on those targets. It now uses a portable `as _`
  cast that infers the platform's `c_char` signedness.

### Changed

- Examples migrated from `hf-hub` 0.5.0 to **1.0.0** — a breaking API rewrite
  (`hf_hub::api::sync::{Api, ApiBuilder}` → `HFClientSync` behind the `blocking`
  feature; `.model("owner/name")` → `split_id()` + `.model(owner, name)`;
  `.get(file)` → `.download_file().filename(file).send()`).

## [0.5.0] - 2026-08-02

### Fixed

- **Integration tests no longer crash under parallel execution**: `fit_params` /
  `get_device_memory_data` install a process-global llama.cpp log callback that
  captures stack locals, so a concurrent model load on another test thread
  invoked a stale callback and segfaulted (SIGSEGV/SIGABRT). Every integration
  test now holds one process-wide lock (`llama_guard`) across its whole
  llama.cpp interaction — model load, context creation, and decode — making the
  suite safe under the default parallel test runner. Test-only change; no
  library API impact.

### Added

- **`LlamaModelParams::with_load_mtp()` / `load_mtp()`** (`llama-cpp-4`): opt into
  loading a model's MTP (multi-token prediction) layers, wrapping upstream's new
  `llama_model_params.load_mtp` field. This is the front door for DeepSeek V4
  MTP / DSpark speculative decoding
  ([#25784](https://github.com/ggml-org/llama.cpp/pull/25784)); once the MTP
  layers are loaded, the speculative state is driven through the existing
  `speculative` module.

### Changed

- **llama.cpp**: vendored submodule updated to `221f0f635` (tag `b10235`) from
  `15e755f30` (b10209), pulling in DeepSeek V4 MTP + DSpark
  ([#25784](https://github.com/ggml-org/llama.cpp/pull/25784)) and 25 other
  upstream commits. Native patches `0003`–`0005` (exact speculative state,
  decode-lifecycle hooks, fail-closed EAGLE-3) apply unchanged.
- Removed the Vulkan SPIRV-Headers patch (`0002`): its `VULKAN_SDK` prefix-path
  and `find_package(SPIRV-Headers CONFIG REQUIRED)` changes are now upstream in
  b10235, so the vendored patch was a no-op that could fail under
  `LLAMA_PATCH_ENGINE=cli --features vulkan`.

## [0.4.4] - 2026-07-31

### Fixed
- **Windows/MSVC build**: `LlamaLoadMode` and `SpeculativeStateError::Unknown`
  assumed the `llama_load_mode` / `mtp_state_status` C enums were `u32`, but
  bindgen types them as `i32` under MSVC — which broke the 0.4.3 build on
  Windows. Both now coerce with `as _` (matching `token_type`), so the crate
  builds on all targets again.

## [0.4.3] - 2026-07-31

### Added

- **Transactional tensor capture/write-back** (`llama-cpp-4`): the owned
  `TensorTransactions` / `LlamaContextParams::with_tensor_transactions` API —
  bounded, Rust-owned capture of decode-time graph nodes with transactional
  finite-`f32` write-back, decode lifecycle hooks, and a documented safe
  ownership boundary (`llama-cpp-4/TENSOR_TRANSACTIONS_SAFETY.md`). Handlers may
  be plain closures (`FnMut(TensorTransaction) -> Result<TensorWriteback, _>`);
  per-selector finiteness validation is configurable via `TensorFiniteValidation`
  (`Strict` / `OutputOnly` / `Trusted`).
- **Hardened speculative sessions** (`llama-cpp-4`): lifetime-bound,
  non-thread-transferable MTP and EAGLE-3 sessions with bounded
  prompt/proposal/state inputs, exact continuation-state capture/restore,
  checked topology/capacity construction, and contained C++ failures.
- **`LlamaLoadMode`** (`llama-cpp-4`): typed model load-mode enum with
  `LlamaModelParams::load_mode()` / `with_load_mode()`, wrapping upstream's new
  `load_mode` field ([#20834](https://github.com/ggml-org/llama.cpp/pull/20834)).
- Allocation-reusing tokenization and raw-token-piece sinks, plus a count-only
  tokenizer query for coordinator-owned bounded utility work.
- **`DecodeError::VetoedByDecodeHook`** (`llama-cpp-4`): a distinct error for a
  decode-lifecycle hook that vetoes a decode (native return `-4`).

### Breaking

- `MtpSession` and `Eagle3Session` now exclusively borrow mutable target and
  draft contexts, are neither `Send` nor `Sync`, and report lifecycle failures
  through typed `Result` values. Existing speculative loops must use the
  session's context accessors and checked decode helpers.
- `LlamaContextParams::with_tensor_capture` is now `unsafe` because it installs
  a pointer to caller-borrowed callback state without retaining that borrow.
  New code should use the safe, owned `with_tensor_transactions` API.

### Changed

- **llama.cpp**: vendored submodule updated to `15e755f30` (tag `b10209`) from
  `571d0d54` (b10068). Adapts `LlamaModelParams` to upstream's new `load_mode`
  enum (which replaced the `use_mmap` / `use_mlock` booleans); the existing
  `use_mmap()` / `use_mlock()` / `with_use_mlock()` API is unchanged.
- Forward-port the binding and native patches (`0003`–`0005`) to the vendored
  llama.cpp revision, including EAGLE-3 v3 target extraction for `gpt-oss`.
- Build `llama-common` for the speculative shim and verify the patched-source
  postcondition even when an OUT_DIR sentinel or shared CMake cache exists.
- Resolve shared runtime assets against Cargo's active target/profile directory
  even when `build.build-dir` places `OUT_DIR` elsewhere.
- Fall back from unverified prebuilt archives to a source build; explicit
  unverifiable `LLAMA_PREBUILT_DIR` inputs now fail closed.
- **ABI-robust decode-lifecycle hooks**: the native patch now appends
  `cb_decode_begin` / `cb_decode_end` to the end of `llama_context_params` so
  every upstream field keeps its original offset, and a link-time guard
  (`llama_cpp_rs_decode_hooks_abi_v1`) makes an unpatched or ABI-mismatched
  prebuilt `libllama` fail to link rather than silently misread the struct.

## [0.4.2] - 2026-07-12

### Added
- **`MtmdInputText::from_bytes`** (`llama-cpp-4`): construct a multimodal text
  prompt from raw bytes. Combined with upstream's new `text_len` field, prompts
  may now contain interior NUL bytes without being truncated.

### Changed
- **llama.cpp**: vendored submodule updated to `99f3dc3` (tag `b9982`) from
  `082b326f` (b9951).
- **`MtmdInputText`** (`llama-cpp-4`): now length-delimited instead of
  `CString`-backed, matching upstream `mtmd_input_text::text_len`
  ([#25548](https://github.com/ggml-org/llama.cpp/pull/25548)). `new` is now
  infallible (interior NUL bytes are permitted and preserved); `try_new` is kept
  for compatibility but never returns `Err`.

### Fixed
- **mtmd**: a NUL byte embedded in a multimodal prompt no longer silently
  truncates it (and drops later messages). The prompt length is now passed
  explicitly to `mtmd_tokenize`.

## [0.4.1] - 2026-07-10

### Added
- **Raw-byte detokenization** (`llama-cpp-4`): recover a token's exact
  `llama_token_to_piece` bytes, bypassing the token-attribute filtering in
  [`LlamaModel::token_to_bytes`] so control, byte-fallback, and other special
  pieces are preserved verbatim.
  - [`LlamaModel::token_to_raw_bytes`] — single token; auto-sizes the buffer to
    whatever llama.cpp requires (no more spurious `InsufficientBufferSpace` on
    long pieces).
  - [`LlamaModel::token_to_raw_bytes_with_size`] — explicit buffer control, with
    `lstrip` support.
  - [`LlamaModel::tokens_to_raw_bytes`] — lossless bulk conversion for a token slice.
- **Streaming detokenizer** (`llama-cpp-4`): [`token::detokenizer::StreamDetokenizer`],
  a stateful, UTF-8-aware decoder for token-by-token generation loops. It buffers
  partial multi-byte sequences split across byte-fallback tokens (emoji, CJK,
  accents) and emits only complete text; accompanied by [`DetokenizeError`]. Both
  are re-exported from `llama_cpp_4::prelude`.
- **Example**: `examples/detokenize.rs` demonstrating the single/bulk raw-byte
  APIs and streaming detokenization during generation.

### Changed
- **llama.cpp**: vendored submodule updated to `082b326f` (tag `b9951`), tracking
  daily upstream syncs from 2026-07-03 through 2026-07-10.

### Dependencies
- Bump `cc` from 1.2.65 to 1.2.66.

## [0.4.0] - 2026-07-01

### Removed (breaking)
- **`LlamaContextParams::with_flash_attention` / `flash_attention`** — use
  `with_flash_attn_type` / `flash_attn_type` with [`LlamaFlashAttnType`].
- **`LlamaContextParams::with_defrag_thold` / `defrag_thold`** — upstream removed
  the field from active use; leave the C default (`-1.0`, disabled).
- **`CommonParams::defrag_thold`**.
- **`LlamaSampler::grammar_lazy`** — use `grammar_lazy_patterns`.
- **`MtmdContext::audio_bitrate`** — use `audio_sample_rate`.
- **`llama_cpp_4::params_fit`** — use `llama_cpp_4::fit::fit_params`.
- **`llama_cpp_4::model_quantize_default_params`** — use `QuantizeParams::new`.

### Added
- **llama.cpp**: vendored submodule updated to `4fc4ec5` (tag `b9859`).
- **Context params** (`llama-cpp-4`): flash attention, attention type, `n_outputs_max`,
  `kv_unified`, `swa_full`, `op_offload`, `ctx_other`, YaRN fields, `no_perf`, abort
  callback, per-sequence sampler configs, and `LlamaPoolingType::Rank`.
- **Context** (`llama-cpp-4`): `memory_breakdown()`, layer input embeddings,
  `set_nextn_layer_offset()`, `ctx_other()`; [`TensorCapture`] for `cb_eval` hooks.
- **Model** (`llama-cpp-4`): `n_layer_nextn()`, `n_expert()`, `n_devices()`,
  `get_device()` / `LlamaBackendDevice`, `target_layer_ids()`, `devices()` iterator.
- **Fit** (`llama-cpp-4`): `fit::get_device_memory_data` for per-device memory estimates;
  `fit::fit_params` safe wrapper around `common_fit_params`.
- **Prelude** (`llama-cpp-4`): `llama_cpp_4::prelude` re-exports common inference types;
  expanded re-exports (`ParamOverrideValue`, `TensorTypeOverride`, `LlamaTokenDataArray`,
  `RpcServer`, …).
- **mtmd** (`llama-cpp-4`): `batch_max_tokens`, flash attention, progress callback.
- **sys** (`llama-cpp-sys-4`): `ext_shim` for structured memory breakdown and fit helpers.
- **Prebuilt download** (`llama-cpp-sys-4`): `--features prebuilt` downloads matching
  GitHub release tarballs into `target/llama-prebuilt-cache/` (or uses `LLAMA_PREBUILT_DIR`);
  falls back to local CMake when no asset exists. Script: `scripts/fetch-prebuilt.sh`.
- **Integration tests** (`llama-cpp-4`): GGUF end-to-end suite (`test_integration`) with
  `scripts/fetch-test-model.sh` and CI job.

### Changed
- **`LlamaContextParams`**: split into `params::{types,advanced}` submodules; added
  `try_clone()`.
- **READMEs**: updated to crate `0.4.0` and llama.cpp `4fc4ec5` (`b9859`); prelude-first
  quick-start, runnable rustdoc examples, and corrected API snippets.
- **Examples**: migrated to `llama_cpp_4::prelude`; chat example uses `apply_chat_template`.

### Fixed
- **`LlamaContextParams: Clone`**: manual impl clears sampler chains so `params.clone()`
  works in examples such as `incremental-chat`.

## [0.3.2] - 2026-06-20

### Changed
- **llama.cpp**: vendored submodule `94a220cd6` → `c57607016` (master, 2026-06-21;
  [PR #256](https://github.com/eugenehp/llama-cpp-rs/pull/256)). The public `llama.h`
  C API is unchanged; the `mtmd` helper API was reworked (see below).

### Added
- **EAGLE-3 speculative decoding** (`llama-cpp-4`): new [`eagle::Eagle3Session`]
  driving upstream `COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3`. Pairs a target context
  with a separate EAGLE-3 draft-model context. The `mtp_shim` is generalised with a
  `spec_type` selector shared by MTP and EAGLE-3 (`MtpSession` is unchanged).
- **`examples/eagle`** (+ README) and weight-fetch scripts: runnable EAGLE-3 demo plus
  `scripts/setup-eagle3.sh` (one command: bootstraps a convert env, then downloads +
  converts a target/draft pairing to GGUF) over `scripts/fetch-eagle3.sh` (download +
  convert). Defaults to the open Qwen3-8B + `RedHatAI/Qwen3-8B-speculator.eagle3`.
  Verified end-to-end on Apple M4 Pro (Metal): coherent generation at ~53% draft
  acceptance.
- **mtmd video input** (`llama-cpp-4`, `mtmd` feature): [`mtmd::MtmdVideo`] (+
  `MtmdVideoParams`, `MtmdVideoInfo`, `MtmdVideoItem`) wrapping the new
  `mtmd_helper_video_*` API for frame-by-frame decoding via ffmpeg. Plus
  `MtmdContext::supports_video()` and `MtmdContext::marker()` (per-context marker).

### Fixed
- **mtmd bindings** (`llama-cpp-4`): adapt to the reworked `mtmd` helper API —
  `mtmd_helper_bitmap_init_from_file`/`_from_buf` now take a `placeholder` flag and
  return a `mtmd_helper_bitmap_wrapper`; `mtmd_helper_decode_image_chunk` takes a
  post-decode callback. `MtmdBitmap::from_file`/`from_buf` also no longer leak the
  `video_ctx` returned for video inputs.

## [0.3.1] - 2026-06-04

### Changed
- **llama.cpp**: vendored submodule `b28a2f372` → `94a220cd6` (master, 2026-06-04;
  ~250 commits). Notable upstream: unified Gemma 4 FPE fix
  ([#24088](https://github.com/ggml-org/llama.cpp/pull/24088)), `LLAMA_BUILD_APP`
  unified binary, embedding API rename to **next-n**
  (`llama_set_embeddings_nextn`, `common_speculative_need_embd_nextn`).
- **Bindings** (`llama-cpp-4`): `LlamaContext` embedding getters/setters call upstream
  `llama_*_nextn` FFI; Rust method names stay `*_pre_norm` for compatibility.
- **`llama-cpp-sys-4` build** (`build.rs`): set `LLAMA_BUILD_APP=OFF` always; set
  `LLAMA_BUILD_COMMON=OFF` when the `mtmd` feature is disabled so the OUT_DIR CMake
  copy builds library targets only (fixes build failure after the submodule bump).
- **`mtp_shim`**: `mtp_session_need_embd_pre_norm` delegates to
  `common_speculative_need_embd_nextn`.

### Added
- **`openai-server`** ([`examples/server/README.md`](examples/server/README.md)):
  - `GET /v1/health` (alias of `/health`, both public when `--api-key` is set)
  - Legacy llama.cpp paths: `/chat/completions`, `/completions`, `/embeddings`
  - `POST /tokenize`, `POST /detokenize` (same JSON shape as upstream server)
  - `max_completion_tokens` accepted as an alias for `max_tokens` on chat/completion routes
  - Integration tests: `/v1/health`, tokenize/detokenize roundtrip, `max_completion_tokens`
- **Docs**: server endpoint tables and MTP next-n naming notes in root `README.md`,
  `llama-cpp-4/README.md`, `llama-cpp-sys-4/README.md`, and rustdoc on `mtp` / `context`.

### Fixed
- **`openai-server`**: compile against regenerated bindings after the llama.cpp bump
  (`llama_get_embeddings_nextn` / related symbols).

## [0.3.0] - 2026-05-19

### Changed
- **llama.cpp**: bumped vendored submodule to `b28a2f372` (includes MTP clean-up
  [#23269](https://github.com/ggml-org/llama.cpp/pull/23269)).
- **MTP draft API**: `MtpSession::new_with_config` and [`MtpSessionConfig`]
  expose `n_min` and `p_min`; upstream default `p_min` is now `0.0`.
- **CI**: Linux dynamic prebuilt collection now includes versioned `.so` files
  and symlinks.

### Added
- [`MtpSession::need_embd_pre_norm`], [`MtpSession::print_stats`],
  [`MtpSession::config`], [`MtpSession::n_min`], [`MtpSession::p_min`].
- `examples/mtp`: `--p-min` CLI flag; session stats printed after generation.

[`MtpSessionConfig`]: llama-cpp-4/src/mtp.rs
[`MtpSession::need_embd_pre_norm`]: llama-cpp-4/src/mtp.rs
[`MtpSession::print_stats`]: llama-cpp-4/src/mtp.rs
[`MtpSession::config`]: llama-cpp-4/src/mtp.rs
[`MtpSession::n_min`]: llama-cpp-4/src/mtp.rs
[`MtpSession::p_min`]: llama-cpp-4/src/mtp.rs

## [0.2.56] - 2026-05-16

### Changed
- **llama.cpp**: bumped vendored submodule to `64b38b561` (master, 2026-05-16),
  which now includes upstream MTP support (PR #22673,
  `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` / `LLAMA_CONTEXT_TYPE_MTP`).
- **Patch removed**: `llama-cpp-sys-4/patches/0002-mtp.patch` is gone — its
  functionality is now upstream. The `mtp` Cargo feature has been removed
  from both `llama-cpp-sys-4` and `llama-cpp-4`.

### Added
- New `LlamaContextType { Default, Mtp }` enum and
  `LlamaContextParams::with_ctx_type` / `ctx_type` wrapping upstream's
  `llama_context_type` (use `Mtp` to load a draft head as the MTP context for
  upstream's `--spec-type draft-mtp` speculative decoder).
- `LlamaContextParams::with_n_rs_seq` / `n_rs_seq` and
  `LlamaContext::n_rs_seq` are now always available (no feature gate).
- New `llama_cpp_4::mtp::MtpSession` — Rust-callable MTP speculative-decoding
  draft loop. Wraps a small C++ shim
  (`llama-cpp-sys-4/mtp_shim/mtp_shim.cpp`) that re-exports upstream's
  `common_speculative_*` MTP path with stable C linkage. Smoke-tested
  end-to-end on Qwen3.6-27B-IQ2_M with 94% draft acceptance.
- New `examples/mtp/` — without `--predict` configures contexts (smoke test);
  with `--predict N` drives the full draft loop via `MtpSession`.

### Removed (breaking)
- `mtp` Cargo feature on both crates.
- `LlamaContext::set_mtp` — upstream removed the `llama_set_mtp` C API; MTP is
  now configured via `ctx_type` on the context, not by post-hoc attachment.
- `LlamaModelParams::with_override_arch` / `override_arch` — the corresponding
  `override_arch` field on `llama_model_params` is gone upstream; MTP head
  architecture is detected automatically from the GGUF metadata.
- `llama_cpp_sys_4::llama_context_seq_rm` — the patched alias is gone; use
  `llama_get_memory` + `llama_memory_seq_rm` (which `clear_kv_cache_seq`
  already does internally).

### Migration
- Drop `features = ["mtp"]` from your `Cargo.toml`.
- Replace `set_mtp(Some(&draft_ctx))` with constructing the draft context from
  `LlamaContextParams::default().with_ctx_type(LlamaContextType::Mtp)`.
- Replace `with_override_arch(...)` calls with nothing — autodetected.
- `scripts/bench-mtp.sh` now passes `--spec-type draft-mtp` (was `mtp`).

## [0.2.43] - 2026-04-10

### Changed

- **Build System**: Changed default library type from dynamic to static
  - Default builds now produce static libraries (.a files)
  - Shared libraries are only built when the `dynamic-link` feature is explicitly enabled
  - Backend features (cuda, metal, blas, vulkan, etc.) no longer force shared library builds
  - The `dynamic-link` feature can be combined with any backend feature to produce shared libraries
  - Environment variable `LLAMA_BUILD_SHARED_LIBS` can override the default behavior

### Backward Compatibility

This change maintains backward compatibility for most use cases:
- Applications using default builds will now get static libraries instead of dynamic ones
- Applications explicitly using the `dynamic-link` feature will continue to work as before
- All backend features (cuda, metal, blas, etc.) continue to work as expected
- The `LLAMA_BUILD_SHARED_LIBS` environment variable provides an escape hatch for special requirements

### Migration Guide

If you were relying on the old behavior (dynamic libraries by default):

1. **Explicitly enable dynamic-link feature**:
   ```bash
   cargo build --features dynamic-link
   ```

2. **Or set the environment variable**:
   ```bash
   LLAMA_BUILD_SHARED_LIBS=1 cargo build
   ```

3. **For Cargo.toml**:
   ```toml
   [dependencies.llama-cpp-sys-4]
   version = "0.2.43"
   features = ["dynamic-link"]
   ```

### Benefits

- **Smaller distribution size**: Static libraries are self-contained
- **Easier deployment**: No need to manage separate .dylib/.so files
- **Better compatibility**: Static linking avoids library version conflicts
- **Explicit control**: Developers can choose the linking strategy that best fits their needs
