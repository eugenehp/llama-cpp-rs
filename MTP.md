# MTP Benchmark Notes

> **Note (2026-05-16):** numbers below predate the upstream merge of
> [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673). They were
> measured against the in-tree `0002-mtp.patch` (since removed). The new
> upstream CLI flag is `--spec-type draft-mtp` (was `--spec-type mtp`);
> `scripts/bench-mtp.sh` has been updated. Re-run if you need current numbers.

## Upstream changes (llama.cpp #23269, May 2026)

After [MTP clean-up #23269](https://github.com/ggml-org/llama.cpp/pull/23269)
landed in the vendored submodule:

- The internal MTP implementation was renamed to
  `common_speculative_impl_draft_mtp` (was `common_speculative_state_draft_mtp`).
- Draft sampling inside MTP now uses **`top_k = 10`** (was `1`) and applies
  **`p_min`** again; upstream default **`p_min = 0.0`** (was `0.75`).
- Upstream default **`spec-draft-n-max` / `n_max` is `3`** (was `16`). The Rust
  [`MtpSession::new`](llama-cpp-4/src/mtp.rs) `n_draft_max` argument overrides
  this explicitly.

No Rust API changes were required; behaviour may differ slightly from older
benchmarks. Re-run `scripts/bench-mtp.sh` after bumping llama.cpp if you rely
on published tok/s numbers.

Rust callers can tune draft behaviour via [`MtpSessionConfig`](llama-cpp-4/src/mtp.rs)
(`n_draft_max`, `p_min`, `n_min`) and inspect upstream stats with
[`MtpSession::print_stats`](llama-cpp-4/src/mtp.rs).

## Rust API

The safe wrapper is [`llama_cpp_4::mtp`](llama-cpp-4/src/mtp.rs). A runnable
end-to-end loop lives in [`examples/mtp/src/main.rs`](examples/mtp/src/main.rs).

### Context setup

Both contexts load the **same MTP-capable GGUF**. Only the draft context uses
`LlamaContextType::Mtp`:

```rust
use llama_cpp_4::context::params::{LlamaContextParams, LlamaContextType};

let n_draft_max = 1; // see benchmark notes below — often faster than 3

let target = model.new_context(&backend, LlamaContextParams::default())?;
let draft = model.new_context(
    &backend,
    LlamaContextParams::default()
        .with_ctx_type(LlamaContextType::Mtp)
        .with_n_rs_seq(n_draft_max.max(4)),
)?;
```

### Session config

```rust
use llama_cpp_4::mtp::{MtpSession, MtpSessionConfig};

// Shorthand: n_min=0, p_min=0.0 (upstream defaults after #23269)
let mut session = MtpSession::new(&target, &draft, 1, n_draft_max)?;

// Full control
let config = MtpSessionConfig::new(1, n_draft_max)
    .with_p_min(0.0)
    .with_n_min(0);
let mut session = MtpSession::new_with_config(&target, &draft, config)?;
```

| Field | Upstream name | Notes |
|---|---|---|
| `n_draft_max` | `n_max` / `--spec-draft-n-max` | Tune per model/quant |
| `p_min` | `p_min` | Default `0.0` since #23269 |
| `n_min` | `n_min` | Default `0` |
| `n_seq` | — | Usually `1` |

### Speculative loop

After each `target.decode(&batch)`:

```rust
session.process(&batch)?;
let drafts = session.draft(0, n_past, last_token)?;
// verify drafts on target, count n_accepted
session.accept(0, n_accepted)?;
```

When generation finishes:

```rust
session.print_stats(); // LOG_INF via llama.cpp log callback
```

### CLI examples

Smoke test (contexts only):

```bash
cargo run --release -p mtp --features metal -- \
    hf-model froggeric/Qwen3.6-27B-MTP-GGUF Qwen3.6-27B-IQ2_M-mtp.gguf
```

Generate with tuned draft depth:

```bash
cargo run --release -p mtp --features metal -- \
    --predict 64 \
    --n-draft-max 1 \
    --p-min 0.0 \
    --prompt "The capital of France is" \
    hf-model froggeric/Qwen3.6-27B-MTP-GGUF Qwen3.6-27B-IQ2_M-mtp.gguf
```

Compare against upstream CLI:

```bash
./scripts/bench-mtp.sh
```

## Q4_K_M Results (Apple M4 Pro, Metal)

Using:
- Baseline model: Qwen3.6-27B-Q4_K_M.gguf
- MTP model: Qwen3.6-27B-Q4_K_M-mtp.gguf

### Finding
MTP works, but it is sensitive to `--spec-draft-n-max`.

- With `--spec-draft-n-max 3`, MTP regressed versus baseline.
- With `--spec-draft-n-max 1`, MTP improved throughput versus baseline.

### Measured Throughput (n_predict=128)
- Baseline average: 12.04 tok/s
- MTP (`--spec-draft-n-max 1`) average: 12.79 tok/s
- Net gain: +0.75 tok/s (+6.2%)

### Practical Recommendation
For this Q4_K_M setup, prefer `--spec-draft-n-max 1` over `3`.
