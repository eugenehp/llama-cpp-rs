# MTP Benchmark Notes

> **Note (2026-05-16):** numbers below predate the upstream merge of
> [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673). They were
> measured against the in-tree `0002-mtp.patch` (since removed). The new
> upstream CLI flag is `--spec-type draft-mtp` (was `--spec-type mtp`);
> `scripts/bench-mtp.sh` has been updated. Re-run if you need current numbers.


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
