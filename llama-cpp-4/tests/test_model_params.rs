//! Unit tests for [`LlamaModelParams`] setters and getters.
//!
//! These need neither a backend nor a model — `llama_model_default_params()` is
//! a pure call — so they run everywhere, including in the no-model CI job.

use llama_cpp_4::model::params::{LlamaLazyMode, LlamaLoadMode, LlamaModelParams};

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

/// llama.cpp `b10881` added `lazy_mode`, defaulting to `AUTO`. Pin it: a future
/// upstream flip to `ON` would start reading marked tensors on demand for every
/// caller using default parameters, trading resident memory for I/O silently.
#[test]
fn model_params_default_lazy_mode_is_auto() {
    let params = LlamaModelParams::default();
    assert_eq!(params.lazy_mode(), LlamaLazyMode::Auto);
}

#[test]
fn model_params_lazy_mode_roundtrips() {
    for mode in [LlamaLazyMode::Off, LlamaLazyMode::Auto, LlamaLazyMode::On] {
        let params = LlamaModelParams::default().with_lazy_mode(mode);
        assert_eq!(params.lazy_mode(), mode, "round-trip failed for {mode:?}");
    }
}

/// `lazy_mode` and `load_mode` are separate upstream fields with unrelated
/// discriminants (one unsigned, one signed). Setting either must not disturb
/// the other.
#[test]
fn model_params_lazy_mode_is_independent_of_load_mode() {
    let params = LlamaModelParams::default()
        .with_load_mode(LlamaLoadMode::Mlock)
        .with_lazy_mode(LlamaLazyMode::On);
    assert_eq!(params.load_mode(), LlamaLoadMode::Mlock);
    assert_eq!(params.lazy_mode(), LlamaLazyMode::On);

    let params = LlamaModelParams::default()
        .with_lazy_mode(LlamaLazyMode::Off)
        .with_load_mode(LlamaLoadMode::DirectIo);
    assert_eq!(params.load_mode(), LlamaLoadMode::DirectIo);
    assert_eq!(params.lazy_mode(), LlamaLazyMode::Off);
}

/// `Off` is `0`, the same bit pattern a zeroed struct would have, so an
/// accidental fall-through in `lazy_mode()`'s match would report it as `Auto`.
#[test]
fn model_params_lazy_off_is_not_confused_with_auto() {
    let params = LlamaModelParams::default().with_lazy_mode(LlamaLazyMode::Off);
    assert_eq!(params.lazy_mode(), LlamaLazyMode::Off);
    assert_ne!(params.lazy_mode(), LlamaLazyMode::Auto);
}

/// `from_name` mirrors llama.cpp's `llama_load_mode_from_str` table by hand
/// (that function throws on unknown input, which cannot cross FFI safely), so
/// pin the round-trip: every mode's upstream name must parse back to itself.
#[test]
fn model_params_load_mode_name_roundtrips() {
    for mode in [
        LlamaLoadMode::Auto,
        LlamaLoadMode::None,
        LlamaLoadMode::Mmap,
        LlamaLoadMode::Mlock,
        LlamaLoadMode::MmapMlock,
        LlamaLoadMode::DirectIo,
    ] {
        let name = mode.name();
        assert!(!name.is_empty(), "{mode:?} has no upstream name");
        assert_eq!(
            LlamaLoadMode::from_name(name),
            Some(mode),
            "{mode:?} did not round-trip through {name:?}"
        );
    }
}

/// The exact spellings upstream's `--load-mode` flag accepts. If a bump renames
/// one, this fails rather than silently breaking every CLI that passes it.
#[test]
fn model_params_load_mode_names_match_upstream_spelling() {
    assert_eq!(LlamaLoadMode::Auto.name(), "auto");
    assert_eq!(LlamaLoadMode::None.name(), "none");
    assert_eq!(LlamaLoadMode::Mmap.name(), "mmap");
    assert_eq!(LlamaLoadMode::Mlock.name(), "mlock");
    assert_eq!(LlamaLoadMode::MmapMlock.name(), "mmap+mlock");
    assert_eq!(LlamaLoadMode::DirectIo.name(), "dio");
}

#[test]
fn model_params_load_mode_from_name_rejects_unknown() {
    assert_eq!(LlamaLoadMode::from_name("nonsense"), None);
    assert_eq!(LlamaLoadMode::from_name(""), None);
    // Case-sensitive, matching upstream's strcmp.
    assert_eq!(LlamaLoadMode::from_name("MMAP"), None);
}
