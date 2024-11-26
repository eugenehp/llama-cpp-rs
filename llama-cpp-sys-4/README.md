# llama-cpp-sys

Raw bindings to llama.cpp with cuda support, including new Sampler API from llama-cpp.

See [llama-cpp-4](https://crates.io/crates/llama-cpp-4) for a safe API.

## OpenMPI support

```shell
brew install open-mpi
cargo build -F mpi
```