//! Unit tests for [`LlamaModelParams`] setters and getters.
//!
//! These need neither a backend nor a model — `llama_model_default_params()` is
//! a pure call — so they run everywhere, including in the no-model CI job.

use llama_cpp_4::model::params::{LlamaLoadMode, LlamaModelParams};

/// llama.cpp `b10470` changed the default `load_mode` from `MMAP` to the new
/// `AUTO` (`-1`), which memory-maps unless a backend device lacks mmap support.
/// Pin the default so a future upstream bump that changes it again is loud
/// rather than silently altering how every model in the wild gets loaded.
#[test]
fn model_params_default_load_mode_is_auto() {
    let params = LlamaModelParams::default();
    assert_eq!(params.load_mode(), LlamaLoadMode::Auto);
}

/// `Auto` must report as mmap-capable: llama.cpp memory-maps under `Auto`
/// unless a device opts out. Reporting `false` here would silently flip the
/// meaning of `use_mmap()` for every caller using default parameters.
#[test]
fn model_params_default_reports_mmap_without_mlock() {
    let params = LlamaModelParams::default();
    assert!(params.use_mmap(), "Auto should report as mmap-capable");
    assert!(!params.use_mlock(), "Auto should not imply mlock");
}

#[test]
fn model_params_load_mode_roundtrips() {
    for mode in [
        LlamaLoadMode::Auto,
        LlamaLoadMode::None,
        LlamaLoadMode::Mmap,
        LlamaLoadMode::Mlock,
        LlamaLoadMode::MmapMlock,
        LlamaLoadMode::DirectIo,
    ] {
        let params = LlamaModelParams::default().with_load_mode(mode);
        assert_eq!(params.load_mode(), mode, "round-trip failed for {mode:?}");
    }
}

/// `Auto` is `-1`, so the enum is signed; a `#[repr(u32)]` would round-trip it
/// as `4294967295` and fall through `load_mode()`'s match to `None`.
#[test]
fn model_params_auto_survives_negative_discriminant() {
    let params = LlamaModelParams::default().with_load_mode(LlamaLoadMode::Auto);
    assert_eq!(params.load_mode(), LlamaLoadMode::Auto);
    assert_ne!(params.load_mode(), LlamaLoadMode::None);
}

#[test]
fn model_params_mmap_and_mlock_reporting() {
    let cases = [
        (LlamaLoadMode::None, false, false),
        (LlamaLoadMode::Mmap, true, false),
        (LlamaLoadMode::Mlock, false, true),
        (LlamaLoadMode::MmapMlock, true, true),
        (LlamaLoadMode::DirectIo, false, false),
    ];
    for (mode, want_mmap, want_mlock) in cases {
        let params = LlamaModelParams::default().with_load_mode(mode);
        assert_eq!(params.use_mmap(), want_mmap, "use_mmap for {mode:?}");
        assert_eq!(params.use_mlock(), want_mlock, "use_mlock for {mode:?}");
    }
}

#[test]
fn model_params_load_mtp_roundtrips() {
    assert!(!LlamaModelParams::default().load_mtp());
    assert!(LlamaModelParams::default().with_load_mtp(true).load_mtp());
}
