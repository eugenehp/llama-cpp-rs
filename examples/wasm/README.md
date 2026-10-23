# llama-cpp-4 in WebAssembly

The same Rust source that runs natively, running in a browser tab. Output is
byte-identical to the native build:

```
Once upon a time, there was a little girl named Lily. She loved to play
```

The result is **two static files** — `index.html` and `llama-wasm.js` — with no
server, no backend and no network requests at runtime. Drop them on GitHub
Pages, S3 or Netlify, or just open the HTML off disk.

## Hosted demo

`.github/workflows/wasm-demo.yml` builds this on every push to `main` and on
every `v*` tag, then:

- **deploys to GitHub Pages** — the live demo, with `stories260K.gguf` bundled
  so it runs without finding a model first
- **attaches `llama-wasm-demo.tar.gz` to the release** on a tag, with a
  `BUILD-INFO.txt` recording the commit, the vendored llama.cpp revision and
  the Emscripten version

It refuses to publish a build that cannot run: the smoke test decodes through
the harness first. GitHub Pages must be enabled for the repo with **Source:
GitHub Actions**; the workflow cannot turn that on itself.

CI also runs a `wasm` job on every PR (`llama-cpp-rs-check.yml`) that builds
both crates for wasm, runs inference, and asserts the generated bindings still
contain more than 500 functions — see *bindgen* below for why that check earns
its keep.

## Build

```bash
source /path/to/emsdk/emsdk_env.sh
rustup target add wasm32-unknown-emscripten

./build.sh                  # static site (default)
./build.sh node             # for testing against the real filesystem
```

Artifacts land in `target/wasm-dist/`.

## Verifying

Three checks, cheapest first. All three run in CI.

```bash
# 1. the wasm module, no DOM
node target/wasm-dist/harness.cjs model.gguf "Once upon a time" 16

# 2+3. the actual page, in a real Chrome over the DevTools Protocol
chrome --headless=new --remote-debugging-port=9222 \
       --user-data-dir=$(mktemp -d) --no-first-run about:blank &

node examples/wasm/browser-test.cjs \
  "file://$PWD/target/wasm-dist/index.html" \
  "$PWD/target/wasm-dist/stories260K.gguf" picker        # no server

(cd target/wasm-dist && python3 -m http.server 8000 &)
node examples/wasm/browser-test.cjs \
  "http://127.0.0.1:8000/index.html" "" sample           # as hosted
```

`browser-test.cjs` needs no dependencies — node 18+ has a global `WebSocket`,
which is all CDP requires. It drives the real file input and the real buttons,
so a broken callback fails there rather than in front of someone.

## How it works, and why these choices

**Target is `wasm32-unknown-emscripten`, not `wasm32-unknown-unknown`.**
llama.cpp needs libc, libc++, pthreads and a filesystem; only Emscripten
supplies them. That means Emscripten's JS glue rather than `wasm-bindgen`, and
an ordinary `main()` — which is why one source file serves both targets.

**The wasm is inlined into the `.js` (`-sSINGLE_FILE=1`).** It costs ~33% in
size, and buys a page that issues no second request. A separate `.wasm` has to
be served as `application/wasm` and fetched — neither of which holds over
`file://`, and not every static host sets the MIME type.

**The model comes from an `<input type="file">`, not the build.**
`--preload-file` would bake a model in and emit a `.data` file to fetch
alongside; instead the page writes the chosen file into MEMFS with
`FS.writeFile`. The model never leaves the machine, and the bundle stays model
agnostic.

**A factory, not a global `Module` (`-sMODULARIZE=1`).** The global form hinges
on `var Module = typeof Module !== "undefined" ? Module : {}` finding a
pre-assigned global, which silently fails under any module system because `var`
hoisting makes the name local — the runtime then starts with a default config
and your callbacks are never called. The factory returns a promise, so "runtime
is up" is awaited instead of guessed.

**bindgen drops every function for wasm triples.** Pointed at
`wasm32-*`, bindgen emits the constants, typedefs and structs — with correct
wasm32 layout — and *no* function declarations, with no error. The crate then
fails much later with hundreds of unresolved imports. `build.rs` parses with
`armv7-unknown-linux-gnueabihf` instead, the one common triple that agrees with
wasm32 on 32-bit pointers, 32-bit `long` and 8-byte-aligned `double` (`i686`
gets that last one wrong). Diffing every layout assertion over `wrapper.h`
leaves 12 differences out of 1709, all libc++/`__int128` internals that
llama.cpp's C API never exposes. The CI function-count assertion exists so this
cannot silently regress.

**Single-threaded, deliberately.** Emscripten's thread pool needs
`SharedArrayBuffer`, which browsers grant only to a cross-origin-isolated page,
which needs `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy`
response headers. A static host cannot send those, so `-pthread` is left off
rather than shipping a build that silently falls back. If you do control the
headers, add `-C link-arg=-pthread -C link-arg=-sPTHREAD_POOL_SIZE=4`.

## Limits worth knowing before you pick a model

- **~2 GB total.** This is a 32-bit wasm build, so weights, KV cache and
  runtime share one address space. llama.cpp defaults `LLAMA_WASM_MEM64=ON`
  (wasm64, which lifts this), but Rust has no stable wasm64 target, so
  `build.rs` forces it off — the two could not be linked otherwise.
- **MEMFS costs double.** The bytes written in sit in the same linear memory the
  weights load into. For a large model, mount the `File` via `WORKERFS` from a
  Worker, which reads it in place instead of copying.
- **No mmap.** There are no pages to map in a virtual filesystem, so the example
  passes `LlamaLoadMode::None`; `Auto` would try mmap wherever the backend
  claims support.
- **CPU + wasm SIMD.** `llama_print_system_info()` reports
  `WASM_SIMD = 1 | LLAMAFILE = 1 | REPACK = 1`. For GPU, build with
  `--features webgpu`, which additionally needs Dawn's `emdawnwebgpu` port.

## Features

Built with `default-features = false`: the defaults pull in `openmp`,
`dynamic-link` and `rpc`, none of which exist under Emscripten. `build.rs`
rejects those (and `metal`/`cuda`/`vulkan`/`hip`/`opencl`/`blas`/`prebuilt`) up
front with a message naming each one, rather than failing deep inside cmake.
