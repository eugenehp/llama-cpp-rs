# Android example (`llama-cpp-4` on arm64)

A complete, minimal Android example that runs on-device text generation through
`llama-cpp-4`, plus an arm64 smoke test that CI validates under QEMU.

```
android/
├── llama-jni/        # Rust cdylib: JNI bridge + `generate()` core + `smoke` bin
│   └── src/
│       ├── lib.rs        # generate() + Java_com_example_llama_LlamaBridge_generate
│       └── bin/smoke.rs  # aarch64 smoke test (run under qemu / natively)
└── app/              # Minimal Gradle/Kotlin app that loads libllama_jni.so
    └── src/main/java/com/example/llama/{LlamaBridge,MainActivity}.kt
```

The native library is **statically self-contained** (`llama-cpp-4` is pulled with
`default-features = false`: CPU-only, no `dynamic-link`/`mtmd`), so the app ships a
single `libllama_jni.so`.

## 1. Prerequisites

- Rust stable + the Android target:
  ```bash
  rustup target add aarch64-linux-android
  ```
- Android NDK (r26+). Note the path — you need **both** env vars below because
  `cargo-ndk` reads `ANDROID_NDK_HOME` while `llama-cpp-sys-4`'s `build.rs` reads
  `ANDROID_NDK`:
  ```bash
  export ANDROID_NDK_HOME=/path/to/android-sdk/ndk/<version>
  export ANDROID_NDK="$ANDROID_NDK_HOME"
  ```
- [`cargo-ndk`](https://github.com/bbqsrc/cargo-ndk): `cargo install cargo-ndk`
- Android SDK + Gradle (or Android Studio) to build the APK.

## 2. Build the native library (`libllama_jni.so`)

From the repository root, cross-compile the cdylib into the app's `jniLibs`:

```bash
cargo ndk -t arm64-v8a -o android/app/src/main/jniLibs \
    build -p llama-jni --release
```

This produces `android/app/src/main/jniLibs/arm64-v8a/libllama_jni.so`.

> **Why `cargo-ndk`?** Besides the per-target `CC`/`CXX`/`AR`/linker, it exports
> `BINDGEN_EXTRA_CLANG_ARGS` with the NDK sysroot so `bindgen` can resolve
> `stdio.h` and other system headers. A plain
> `cargo build --target aarch64-linux-android` fails at binding generation
> (`fatal error: 'stdio.h' file not found`) unless you set those args yourself.

## 3. Build & install the app

```bash
cd android
gradle wrapper            # first time only, or open the folder in Android Studio
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## 4. Provide a model and run

The app reads the GGUF from `filesDir/model.gguf`. Push any small GGUF (the
1 MB `stories260K.gguf` from `scripts/fetch-test-model.sh` is ideal for a smoke
run):

```bash
adb push stories260K.gguf /data/data/com.example.llama/files/model.gguf
```

Launch the app, type a prompt, tap **Generate**.

## Validating arm64 without a device (QEMU)

The same `generate()` code path is exercised on `aarch64-unknown-linux-gnu`
under `qemu-user` by the [`QEMU arm64 smoke`](../.github/workflows/qemu-arm64.yml)
workflow. To reproduce locally on an x86 Linux host:

```bash
sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu qemu-user
rustup target add aarch64-unknown-linux-gnu
export CC_aarch64-unknown-linux-gnu=aarch64-linux-gnu-gcc
export CXX_aarch64-unknown-linux-gnu=aarch64-linux-gnu-g++
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc

cargo build -p llama-jni --bin smoke --release --target aarch64-unknown-linux-gnu
LLAMA_TEST_MODEL=target/test-models/stories260K.gguf \
  qemu-aarch64 -L /usr/aarch64-linux-gnu \
  target/aarch64-unknown-linux-gnu/release/smoke "Once upon a time"
```

It runs natively too (handy on an Apple-Silicon / arm64 dev box):

```bash
cargo run -p llama-jni --bin smoke -- path/to/model.gguf "Once upon a time"
```
