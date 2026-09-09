//! A safe wrapper around `llama_model_params`.

use crate::model::params::kv_overrides::KvOverrides;
use std::ffi::{c_char, CStr};
use std::fmt::{Debug, Formatter};
use std::pin::Pin;
use std::ptr::null;

pub mod kv_overrides;

/// Exact model-file loading strategy exposed by llama.cpp.
///
/// `llama_load_mode` is a signed enum on every target because of the negative
/// `LLAMA_LOAD_MODE_AUTO` discriminant, so each variant uses `as _` to coerce to
/// the `#[repr(i32)]` type (matching [`token_type`](crate::token_type)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum LlamaLoadMode {
    /// Pick the strategy from the backend devices' capabilities: memory-map when
    /// every device supports it, otherwise fall back to a plain read. This is
    /// llama.cpp's default.
    Auto = llama_cpp_sys_4::LLAMA_LOAD_MODE_AUTO as _,
    /// No memory mapping, locking, or direct I/O.
    None = llama_cpp_sys_4::LLAMA_LOAD_MODE_NONE as _,
    /// Memory-map model files when supported.
    Mmap = llama_cpp_sys_4::LLAMA_LOAD_MODE_MMAP as _,
    /// Read model files normally and lock loaded pages in memory.
    Mlock = llama_cpp_sys_4::LLAMA_LOAD_MODE_MLOCK as _,
    /// Memory-map model files and lock mapped pages in memory.
    MmapMlock = llama_cpp_sys_4::LLAMA_LOAD_MODE_MMAP_MLOCK as _,
    /// Use direct I/O when supported.
    DirectIo = llama_cpp_sys_4::LLAMA_LOAD_MODE_DIRECT_IO as _,
}

impl LlamaLoadMode {
    /// llama.cpp's own name for this mode: `"auto"`, `"none"`, `"mmap"`,
    /// `"mlock"`, `"mmap+mlock"` or `"dio"`.
    ///
    /// Wraps `llama_load_mode_name`, so the spelling always matches what
    /// upstream's logs print and what its `--load-mode` flag accepts.
    ///
    /// # Panics
    ///
    /// Panics if llama.cpp returns a non-UTF-8 name, which would mean the
    /// upstream table was corrupted.
    #[must_use]
    pub fn name(self) -> &'static str {
        let ptr = unsafe { llama_cpp_sys_4::llama_load_mode_name(self as _) };
        assert!(!ptr.is_null(), "llama_load_mode_name returned null");
        // SAFETY: upstream returns a pointer to a string literal, so `'static`
        // holds.
        unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .expect("llama_load_mode_name returned non-UTF-8")
    }

    /// Parse a mode from llama.cpp's own spelling — the inverse of
    /// [`Self::name`]. Returns `None` if `name` matches no mode.
    ///
    /// ```
    /// # use llama_cpp_4::model::params::LlamaLoadMode;
    /// assert_eq!(LlamaLoadMode::from_name("mmap+mlock"), Some(LlamaLoadMode::MmapMlock));
    /// assert_eq!(LlamaLoadMode::from_name("nonsense"), None);
    /// ```
    //
    // Deliberately *not* a call to `llama_load_mode_from_str`: that function
    // throws `std::invalid_argument` for an unrecognised string, and letting a
    // C++ exception unwind across the `extern "C"` boundary into Rust is
    // undefined behaviour. Comparing against `name()` uses upstream's own
    // strings, so this cannot drift from the C table it mirrors — the
    // round-trip test pins that.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [
            Self::Auto,
            Self::None,
            Self::Mmap,
            Self::Mlock,
            Self::MmapMlock,
            Self::DirectIo,
        ]
        .into_iter()
        .find(|mode| mode.name() == name)
    }
}

/// Whether tensors the architecture marks as lazy are read on demand rather
/// than up front.
///
/// Only tensors the model architecture flags carry this at all — today that is
/// Gemma-4's per-layer token embedding and `Qwen4Exp`'s PLE rows — so on every
/// other architecture the setting has no effect. Lazy reading always needs
/// mmap; without it llama.cpp warns and loads the tensor in full regardless.
///
/// `llama_lazy_mode` is an unsigned enum upstream (all discriminants are
/// non-negative), unlike the signed [`LlamaLoadMode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LlamaLazyMode {
    /// Never read lazily — always pull the whole tensor up front.
    Off = llama_cpp_sys_4::LLAMA_LAZY_MODE_OFF as _,
    /// Read lazily only for marked tensors larger than 4 GiB. llama.cpp's
    /// default, and downgraded to [`LlamaLazyMode::Off`] at load time if any
    /// backend device lacks mmap support (iGPUs, for instance).
    Auto = llama_cpp_sys_4::LLAMA_LAZY_MODE_AUTO as _,
    /// Read every marked tensor's rows on demand, whatever its size. Trades
    /// I/O for resident memory; the 4 GiB floor exists because the per-read
    /// overhead is not worth it on small tensors.
    On = llama_cpp_sys_4::LLAMA_LAZY_MODE_ON as _,
}

/// A safe wrapper around `llama_model_params`.
#[allow(clippy::module_name_repetitions)]
pub struct LlamaModelParams {
    pub(crate) params: llama_cpp_sys_4::llama_model_params,
    kv_overrides: Vec<llama_cpp_sys_4::llama_model_kv_override>,
}

impl Debug for LlamaModelParams {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaModelParams")
            .field("n_gpu_layers", &self.params.n_gpu_layers)
            .field("main_gpu", &self.params.main_gpu)
            .field("vocab_only", &self.params.vocab_only)
            .field("load_mode", &self.load_mode())
            .field("lazy_mode", &self.lazy_mode())
            .field("load_mtp", &self.load_mtp())
            .field("kv_overrides", &"vec of kv_overrides")
            .finish()
    }
}

impl LlamaModelParams {
    /// See [`KvOverrides`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use llama_cpp_4::model::params::LlamaModelParams;
    /// let params = Box::pin(LlamaModelParams::default());
    /// let kv_overrides = params.kv_overrides();
    /// let count = kv_overrides.into_iter().count();
    /// assert_eq!(count, 0);
    /// ```
    #[must_use]
    pub fn kv_overrides(&self) -> KvOverrides<'_> {
        KvOverrides::new(self)
    }

    /// Appends a key-value override to the model parameters. It must be pinned as this creates a self-referential struct.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::ffi::{CStr, CString};
    /// use std::pin::pin;
    /// # use llama_cpp_4::model::params::LlamaModelParams;
    /// # use llama_cpp_4::model::params::kv_overrides::ParamOverrideValue;
    /// let mut params = pin!(LlamaModelParams::default());
    /// let key = CString::new("key").expect("CString::new failed");
    /// params.as_mut().append_kv_override(&key, ParamOverrideValue::Int(50));
    ///
    /// let kv_overrides = params.kv_overrides().into_iter().collect::<Vec<_>>();
    /// assert_eq!(kv_overrides.len(), 1);
    ///
    /// let (k, v) = &kv_overrides[0];
    /// assert_eq!(v, &ParamOverrideValue::Int(50));
    ///
    /// assert_eq!(k.to_bytes(), b"key", "expected key to be 'key', was {:?}", k);
    /// ```
    #[allow(clippy::missing_panics_doc)] // panics are just to enforce internal invariants, not user errors
    pub fn append_kv_override(
        mut self: Pin<&mut Self>,
        key: &CStr,
        value: kv_overrides::ParamOverrideValue,
    ) {
        let kv_override = self
            .kv_overrides
            .get_mut(0)
            .expect("kv_overrides did not have a next allocated");

        assert_eq!(kv_override.key[0], 0, "last kv_override was not empty");

        // There should be some way to do this without iterating over everything.
        for (i, &c) in key.to_bytes_with_nul().iter().enumerate() {
            kv_override.key[i] = c_char::try_from(c).expect("invalid character in key");
        }

        kv_override.tag = value.tag();
        kv_override.__bindgen_anon_1 = value.value();

        // set to null pointer for panic safety (as push may move the vector, invalidating the pointer)
        self.params.kv_overrides = null();

        // push the next one to ensure we maintain the iterator invariant of ending with a 0
        self.kv_overrides
            .push(llama_cpp_sys_4::llama_model_kv_override {
                key: [0; 128],
                tag: 0,
                __bindgen_anon_1: llama_cpp_sys_4::llama_model_kv_override__bindgen_ty_1 {
                    val_i64: 0,
                },
            });

        // set the pointer to the (potentially) new vector
        self.params.kv_overrides = self.kv_overrides.as_ptr();

        eprintln!("saved ptr: {:?}", self.params.kv_overrides);
    }
}

impl LlamaModelParams {
    /// Get the number of layers to offload to the GPU.
    #[must_use]
    pub fn n_gpu_layers(&self) -> i32 {
        self.params.n_gpu_layers
    }

    /// The GPU that is used for scratch and small tensors
    #[must_use]
    pub fn main_gpu(&self) -> i32 {
        self.params.main_gpu
    }

    /// only load the vocabulary, no weights
    #[must_use]
    pub fn vocab_only(&self) -> bool {
        self.params.vocab_only
    }

    /// Returns the exact model-file loading strategy.
    #[must_use]
    pub fn load_mode(&self) -> LlamaLoadMode {
        match self.params.load_mode {
            llama_cpp_sys_4::LLAMA_LOAD_MODE_AUTO => LlamaLoadMode::Auto,
            llama_cpp_sys_4::LLAMA_LOAD_MODE_MMAP => LlamaLoadMode::Mmap,
            llama_cpp_sys_4::LLAMA_LOAD_MODE_MLOCK => LlamaLoadMode::Mlock,
            llama_cpp_sys_4::LLAMA_LOAD_MODE_MMAP_MLOCK => LlamaLoadMode::MmapMlock,
            llama_cpp_sys_4::LLAMA_LOAD_MODE_DIRECT_IO => LlamaLoadMode::DirectIo,
            _ => LlamaLoadMode::None,
        }
    }

    /// Returns whether arch-marked tensors are read on demand.
    ///
    /// This is the requested mode, not the effective one: llama.cpp resolves
    /// [`LlamaLazyMode::Auto`] down to [`LlamaLazyMode::Off`] during load when a
    /// device lacks mmap support, and that resolution is not written back here.
    #[must_use]
    pub fn lazy_mode(&self) -> LlamaLazyMode {
        match self.params.lazy_mode {
            llama_cpp_sys_4::LLAMA_LAZY_MODE_OFF => LlamaLazyMode::Off,
            llama_cpp_sys_4::LLAMA_LAZY_MODE_ON => LlamaLazyMode::On,
            _ => LlamaLazyMode::Auto,
        }
    }

    /// Whether the model's MTP (multi-token prediction) layers will be loaded.
    ///
    /// MTP layers drive multi-token-prediction speculative decoding for models
    /// that ship them (e.g. `DeepSeek V4`). Once loaded, the speculative state is
    /// captured and restored through [`crate::speculative`]. Defaults to `false`
    /// because most models carry no MTP weights.
    #[must_use]
    pub fn load_mtp(&self) -> bool {
        self.params.load_mtp
    }

    /// use mmap if possible
    ///
    /// [`LlamaLoadMode::Auto`] counts as "possible": llama.cpp memory-maps under
    /// `Auto` unless one of the backend devices lacks mmap support, which is only
    /// known once the model is loaded.
    #[must_use]
    pub fn use_mmap(&self) -> bool {
        matches!(
            self.load_mode(),
            LlamaLoadMode::Auto | LlamaLoadMode::Mmap | LlamaLoadMode::MmapMlock
        )
    }

    /// force system to keep model in RAM
    #[must_use]
    pub fn use_mlock(&self) -> bool {
        matches!(
            self.load_mode(),
            LlamaLoadMode::Mlock | LlamaLoadMode::MmapMlock
        )
    }

    /// sets the number of gpu layers to offload to the GPU.
    /// ```
    /// # use llama_cpp_4::model::params::LlamaModelParams;
    /// let params = LlamaModelParams::default();
    /// let params = params.with_n_gpu_layers(1);
    /// assert_eq!(params.n_gpu_layers(), 1);
    /// ```
    #[must_use]
    pub fn with_n_gpu_layers(mut self, n_gpu_layers: u32) -> Self {
        // The only way this conversion can fail is if u32 overflows the i32 - in which case we set
        // to MAX
        let n_gpu_layers = i32::try_from(n_gpu_layers).unwrap_or(i32::MAX);
        self.params.n_gpu_layers = n_gpu_layers;
        self
    }

    /// sets the main GPU
    #[must_use]
    pub fn with_main_gpu(mut self, main_gpu: i32) -> Self {
        self.params.main_gpu = main_gpu;
        self
    }

    /// sets `vocab_only`
    #[must_use]
    pub fn with_vocab_only(mut self, vocab_only: bool) -> Self {
        self.params.vocab_only = vocab_only;
        self
    }

    /// Sets the exact model-file loading strategy.
    #[must_use]
    pub fn with_load_mode(mut self, load_mode: LlamaLoadMode) -> Self {
        self.params.load_mode = load_mode as llama_cpp_sys_4::llama_load_mode;
        self
    }

    /// Sets whether arch-marked tensors are read on demand.
    ///
    /// Reach for [`LlamaLazyMode::On`] when a Gemma-4 or `Qwen4Exp` model's marked
    /// tensors will not fit in RAM and the extra I/O is the better trade;
    /// [`LlamaLazyMode::Off`] pins everything in memory up front. Corresponds to
    /// `llama_model_params.lazy_mode`, added upstream in llama.cpp PR #27794.
    ///
    /// ```
    /// # use llama_cpp_4::model::params::{LlamaLazyMode, LlamaModelParams};
    /// let params = LlamaModelParams::default().with_lazy_mode(LlamaLazyMode::On);
    /// assert_eq!(params.lazy_mode(), LlamaLazyMode::On);
    /// ```
    #[must_use]
    pub fn with_lazy_mode(mut self, lazy_mode: LlamaLazyMode) -> Self {
        self.params.lazy_mode = lazy_mode as llama_cpp_sys_4::llama_lazy_mode;
        self
    }

    /// Sets whether to load the model's MTP (multi-token prediction) layers.
    ///
    /// Enable this for models that ship MTP weights (e.g. `DeepSeek V4`) when you
    /// intend to use MTP-based speculative decoding, then drive the speculative
    /// state via [`crate::speculative`]. For models without MTP layers the flag
    /// has no effect. Corresponds to `llama_model_params.load_mtp`, added
    /// upstream in llama.cpp PR #25784 (`DeepSeek V4` MTP + `DSpark`).
    ///
    /// ```
    /// # use llama_cpp_4::model::params::LlamaModelParams;
    /// let params = LlamaModelParams::default().with_load_mtp(true);
    /// assert!(params.load_mtp());
    /// ```
    #[must_use]
    pub fn with_load_mtp(mut self, load_mtp: bool) -> Self {
        self.params.load_mtp = load_mtp;
        self
    }

    /// sets `use_mlock`
    #[must_use]
    pub fn with_use_mlock(mut self, use_mlock: bool) -> Self {
        let load_mode = match (self.use_mmap(), use_mlock) {
            (true, true) => LlamaLoadMode::MmapMlock,
            (true, false) => LlamaLoadMode::Mmap,
            (false, true) => LlamaLoadMode::Mlock,
            (false, false) => LlamaLoadMode::None,
        };
        self.params.load_mode = load_mode as llama_cpp_sys_4::llama_load_mode;
        self
    }
}

/// Default parameters for `LlamaModel`. (as defined in llama.cpp by `llama_model_default_params`)
/// ```
/// # use llama_cpp_4::model::params::LlamaModelParams;
/// let params = LlamaModelParams::default();
/// assert_eq!(params.n_gpu_layers(), -1, "n_gpu_layers should be -1 (all layers)");
/// assert_eq!(params.main_gpu(), 0, "main_gpu should be 0");
/// assert_eq!(params.vocab_only(), false, "vocab_only should be false");
/// assert_eq!(params.use_mmap(), true, "use_mmap should be true");
/// assert_eq!(params.use_mlock(), false, "use_mlock should be false");
/// ```
impl Default for LlamaModelParams {
    fn default() -> Self {
        let default_params = unsafe { llama_cpp_sys_4::llama_model_default_params() };
        LlamaModelParams {
            params: default_params,
            // push the next one to ensure we maintain the iterator invariant of ending with a 0
            kv_overrides: vec![llama_cpp_sys_4::llama_model_kv_override {
                key: [0; 128],
                tag: 0,
                __bindgen_anon_1: llama_cpp_sys_4::llama_model_kv_override__bindgen_ty_1 {
                    val_i64: 0,
                },
            }],
        }
    }
}
