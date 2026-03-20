ARG CUDA_VERSION=12.3.1
ARG UBUNTU_VERSION=22.04
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as base-cuda

# Install requirements for rustup install + bindgen + llama-cpp-sys cmake build.
RUN DEBIAN_FRONTEND=noninteractive apt update -y && apt install -y curl llvm-dev libclang-dev clang pkg-config libssl-dev cmake mold
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH=/root/.cargo/bin:$PATH
ENV RUSTFLAGS="-C link-arg=-fuse-ld=mold"

COPY . .
RUN cargo build --bin simple --features cuda

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as base-cuda-runtime

COPY --from=base-cuda /target/debug/simple /usr/local/bin/simple

ENTRYPOINT ["/usr/local/bin/simple"]
