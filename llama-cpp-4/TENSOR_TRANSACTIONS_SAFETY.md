<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Tensor transaction safety contract

`TensorTransactions` is the safe Rust boundary around llama.cpp's
`ggml_backend_sched_eval_callback`. The implementation relies on the following
contract.

## Ownership and lifetime

- `LlamaContextParams::with_tensor_transactions` moves the complete callback
  state into a `Pin<Box<_>>` before installing its address in
  `cb_eval_user_data`.
- Creating a context moves that pinned box, without moving its allocation, into
  `LlamaContext`.
- `LlamaContext::drop` calls `llama_free` before Rust drops the pinned callback
  state. No callback can therefore observe freed Rust state.
- A transaction handler is owned by one context. Calls are synchronous and
  serialized by the native context. Exact begin/end hooks cover both direct
  `LlamaContext::decode` calls and llama.cpp decodes performed internally by a
  speculative implementation.

No Rust reference, slice, panic payload, or handler-owned pointer crosses the
native boundary.

## Native tensor access

llama.cpp invokes the data phase only after computing and synchronizing the
selected graph node. Before copying data, Rust verifies:

- the exact graph-node name selected by the caller;
- the exact `f32` or `i32` element type;
- a two-dimensional, nonempty, contiguous layout;
- the exact elements per row and an inclusive row bound;
- the checked total element and byte counts; and
- correspondence with the logical rows in the submitted decode batch.

The callback copies the complete tensor into Rust-owned storage. A mutable
`f32` transaction writes the complete owned copy back exactly once, only after
the handler returns successfully and every output value is finite. Errors and
unwinding cause no partial write-back.

Retained captures are staged per native decode. They enter the committed
capture set only after native success and complete selector-row coverage.
Several internal decodes may commit into that set before Rust drains it; a
failed decode discards only its staging area. Aggregate retained bytes remain
bounded across the complete undrained set.

## Failure containment

The FFI callback catches all Rust unwinding and stores a bounded typed failure.
After a data-phase failure it returns `true`: returning `false` there would
cause llama.cpp's scheduler to stop executing the remainder of that graph
split. Later ask phases return `false`, so native execution can finish without
re-entering Rust for more selected data.

`LlamaContext::decode` checks the stored failure after native execution returns
and reports `DecodeError::TensorCallback`. The caller must conservatively treat
that context as causally advanced; the binding does not roll back native
context state.

The first callback failure is sticky for the context lifetime. This prevents a
later internal speculative decode from clearing evidence that an earlier
callback failed before the outer Rust operation regained control.

## Profile binding

Graph-node names and layouts are implementation details of the pinned
llama.cpp revision. A caller must bind its exact selectors to a separately
validated model/backend profile. A selector mismatch fails closed after decode;
there is no prefix matching, type coercion, shape inference, or silent
fallback.

The older `TensorCapture` callback remains available only through an `unsafe`
constructor because it borrows external state without retaining a Rust
lifetime. New code should use `TensorTransactions`.
