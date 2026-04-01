//! Quantization types and parameters for converting models to lower-bit precisions.
//!
//! # Quick start
//!
//! ```no_run
//! use llama_cpp_4::quantize::{LlamaFtype, QuantizeParams};
//!
//! let params = QuantizeParams::new(LlamaFtype::MostlyQ4KM)
//!     .with_nthread(8)
//!     .with_quantize_output_tensor(true);
//!
//! llama_cpp_4::model_quantize("model-f16.gguf", "model-q4km.gguf", &params).unwrap();
//! ```
//!
//! # TurboQuant – attention rotation (PR #21038)
//!
//! llama.cpp applies a Hadamard rotation to Q/K/V tensors before writing them into the KV cache.
//! This significantly improves KV-cache quantization quality at near-zero cost, and is enabled by
//! default for every model whose head dimension is a power of two.  You can opt out per-context
//! with [`LlamaContextParams::with_attn_rot_disabled`] or globally with
//! [`set_attn_rot_disabled`].
//!
//! [`LlamaContextParams::with_attn_rot_disabled`]: crate::context::params::LlamaContextParams::with_attn_rot_disabled

use std::ffi::{CString, NulError};
use std::ptr::null;

// ─────────────────────────────────────────────────────────────────────────────
// LlamaFtype
// ─────────────────────────────────────────────────────────────────────────────

/// The quantization type used for the bulk of a model file (maps to `llama_ftype`).
///
/// Pass one of these variants to [`QuantizeParams::new`] to choose the target precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum LlamaFtype {
    /// All tensors stored as full F32 (very large, for reference only)
    AllF32 = 0,
    /// F16 – 14 GB @ 7B, +0.0020 ppl vs Mistral-7B
    MostlyF16 = 1,
    /// Q4_0 – 4.34 GB @ 8B, +0.4685 ppl
    MostlyQ4_0 = 2,
    /// Q4_1 – 4.78 GB @ 8B, +0.4511 ppl
    MostlyQ4_1 = 3,
    /// Q8_0 – 7.96 GB @ 8B, +0.0026 ppl
    MostlyQ8_0 = 7,
    /// Q5_0 – 5.21 GB @ 8B, +0.1316 ppl
    MostlyQ5_0 = 8,
    /// Q5_1 – 5.65 GB @ 8B, +0.1062 ppl
    MostlyQ5_1 = 9,
    /// Q2_K – 2.96 GB @ 8B, +3.5199 ppl
    MostlyQ2K = 10,
    /// Q3_K small – 3.41 GB @ 8B, +1.6321 ppl
    MostlyQ3KS = 11,
    /// Q3_K medium – 3.74 GB @ 8B, +0.6569 ppl
    MostlyQ3KM = 12,
    /// Q3_K large – 4.03 GB @ 8B, +0.5562 ppl
    MostlyQ3KL = 13,
    /// Q4_K small – 4.37 GB @ 8B, +0.2689 ppl
    MostlyQ4KS = 14,
    /// Q4_K medium – 4.58 GB @ 8B, +0.1754 ppl  *(recommended default)*
    MostlyQ4KM = 15,
    /// Q5_K small – 5.21 GB @ 8B, +0.1049 ppl
    MostlyQ5KS = 16,
    /// Q5_K medium – 5.33 GB @ 8B, +0.0569 ppl
    MostlyQ5KM = 17,
    /// Q6_K – 6.14 GB @ 8B, +0.0217 ppl
    MostlyQ6K = 18,
    /// IQ2_XXS – 2.06 bpw
    MostlyIQ2XXS = 19,
    /// IQ2_XS – 2.31 bpw
    MostlyIQ2XS = 20,
    /// Q2_K small
    MostlyQ2KS = 21,
    /// IQ3_XS – 3.3 bpw
    MostlyIQ3XS = 22,
    /// IQ3_XXS – 3.06 bpw
    MostlyIQ3XXS = 23,
    /// IQ1_S – 1.56 bpw (extremely small, high loss)
    MostlyIQ1S = 24,
    /// IQ4_NL – 4.50 bpw non-linear
    MostlyIQ4NL = 25,
    /// IQ3_S – 3.44 bpw
    MostlyIQ3S = 26,
    /// IQ3_M – 3.66 bpw
    MostlyIQ3M = 27,
    /// IQ2_S – 2.5 bpw
    MostlyIQ2S = 28,
    /// IQ2_M – 2.7 bpw
    MostlyIQ2M = 29,
    /// IQ4_XS – 4.25 bpw non-linear
    MostlyIQ4XS = 30,
    /// IQ1_M – 1.75 bpw
    MostlyIQ1M = 31,
    /// BF16 – 14 GB @ 7B, −0.0050 ppl vs Mistral-7B
    MostlyBF16 = 32,
    /// TQ1_0 – 1.69 bpw ternary
    MostlyTQ1_0 = 36,
    /// TQ2_0 – 2.06 bpw ternary
    MostlyTQ2_0 = 37,
    /// MXFP4 (MoE layers)
    MostlyMXFP4Moe = 38,
    /// NVFP4
    MostlyNVFP4 = 39,
}

impl LlamaFtype {
    /// Short name suitable for filenames (e.g. `"Q4_K_M"`).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::AllF32 => "F32",
            Self::MostlyF16 => "F16",
            Self::MostlyQ4_0 => "Q4_0",
            Self::MostlyQ4_1 => "Q4_1",
            Self::MostlyQ8_0 => "Q8_0",
            Self::MostlyQ5_0 => "Q5_0",
            Self::MostlyQ5_1 => "Q5_1",
            Self::MostlyQ2K => "Q2_K",
            Self::MostlyQ3KS => "Q3_K_S",
            Self::MostlyQ3KM => "Q3_K_M",
            Self::MostlyQ3KL => "Q3_K_L",
            Self::MostlyQ4KS => "Q4_K_S",
            Self::MostlyQ4KM => "Q4_K_M",
            Self::MostlyQ5KS => "Q5_K_S",
            Self::MostlyQ5KM => "Q5_K_M",
            Self::MostlyQ6K => "Q6_K",
            Self::MostlyIQ2XXS => "IQ2_XXS",
            Self::MostlyIQ2XS => "IQ2_XS",
            Self::MostlyQ2KS => "Q2_K_S",
            Self::MostlyIQ3XS => "IQ3_XS",
            Self::MostlyIQ3XXS => "IQ3_XXS",
            Self::MostlyIQ1S => "IQ1_S",
            Self::MostlyIQ4NL => "IQ4_NL",
            Self::MostlyIQ3S => "IQ3_S",
            Self::MostlyIQ3M => "IQ3_M",
            Self::MostlyIQ2S => "IQ2_S",
            Self::MostlyIQ2M => "IQ2_M",
            Self::MostlyIQ4XS => "IQ4_XS",
            Self::MostlyIQ1M => "IQ1_M",
            Self::MostlyBF16 => "BF16",
            Self::MostlyTQ1_0 => "TQ1_0",
            Self::MostlyTQ2_0 => "TQ2_0",
            Self::MostlyMXFP4Moe => "MXFP4_MOE",
            Self::MostlyNVFP4 => "NVFP4",
        }
    }

    /// Human-readable description with approximate size and PPL delta.
    #[must_use]
    pub fn description(self) -> &'static str {
        match self {
            Self::AllF32 => "26.00 GB @ 7B — full precision reference",
            Self::MostlyF16 => "14.00 GB @ 7B — +0.0020 ppl vs Mistral-7B",
            Self::MostlyBF16 => "14.00 GB @ 7B — -0.0050 ppl vs Mistral-7B",
            Self::MostlyQ8_0 => " 7.96 GB @ 8B — +0.0026 ppl",
            Self::MostlyQ6K => " 6.14 GB @ 8B — +0.0217 ppl",
            Self::MostlyQ5KM => " 5.33 GB @ 8B — +0.0569 ppl",
            Self::MostlyQ5KS => " 5.21 GB @ 8B — +0.1049 ppl",
            Self::MostlyQ5_1 => " 5.65 GB @ 8B — +0.1062 ppl",
            Self::MostlyQ5_0 => " 5.21 GB @ 8B — +0.1316 ppl",
            Self::MostlyQ4KM => " 4.58 GB @ 8B — +0.1754 ppl  [recommended]",
            Self::MostlyQ4KS => " 4.37 GB @ 8B — +0.2689 ppl",
            Self::MostlyQ4_1 => " 4.78 GB @ 8B — +0.4511 ppl",
            Self::MostlyQ4_0 => " 4.34 GB @ 8B — +0.4685 ppl",
            Self::MostlyQ3KL => " 4.03 GB @ 8B — +0.5562 ppl",
            Self::MostlyQ3KM => " 3.74 GB @ 8B — +0.6569 ppl",
            Self::MostlyQ3KS => " 3.41 GB @ 8B — +1.6321 ppl",
            Self::MostlyQ2KS => " 2.96 GB @ 8B — +3.1836 ppl",
            Self::MostlyQ2K => " 2.96 GB @ 8B — +3.5199 ppl",
            Self::MostlyIQ4XS => " 4.25 bpw non-linear",
            Self::MostlyIQ4NL => " 4.50 bpw non-linear",
            Self::MostlyIQ3S => " 3.44 bpw",
            Self::MostlyIQ3M => " 3.66 bpw",
            Self::MostlyIQ3XS => " 3.3 bpw",
            Self::MostlyIQ3XXS => " 3.06 bpw",
            Self::MostlyIQ2M => " 2.7 bpw",
            Self::MostlyIQ2S => " 2.5 bpw",
            Self::MostlyIQ2XS => " 2.31 bpw",
            Self::MostlyIQ2XXS => " 2.06 bpw",
            Self::MostlyIQ1M => " 1.75 bpw — extreme compression",
            Self::MostlyIQ1S => " 1.56 bpw — extreme compression",
            Self::MostlyTQ1_0 => " 1.69 bpw ternary",
            Self::MostlyTQ2_0 => " 2.06 bpw ternary",
            Self::MostlyMXFP4Moe => "MXFP4 MoE layers",
            Self::MostlyNVFP4 => "NVFP4",
        }
    }

    /// Look up a variant by its short name (case-insensitive).
    ///
    /// ```
    /// use llama_cpp_4::quantize::LlamaFtype;
    /// assert_eq!(LlamaFtype::from_name("Q4_K_M"), Some(LlamaFtype::MostlyQ4KM));
    /// assert_eq!(LlamaFtype::from_name("q4_k_m"), Some(LlamaFtype::MostlyQ4KM));
    /// assert_eq!(LlamaFtype::from_name("bogus"), None);
    /// ```
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        let upper = name.to_uppercase();
        match upper.as_str() {
            "F32" => Some(Self::AllF32),
            "F16" => Some(Self::MostlyF16),
            "BF16" => Some(Self::MostlyBF16),
            "Q4_0" => Some(Self::MostlyQ4_0),
            "Q4_1" => Some(Self::MostlyQ4_1),
            "Q8_0" => Some(Self::MostlyQ8_0),
            "Q5_0" => Some(Self::MostlyQ5_0),
            "Q5_1" => Some(Self::MostlyQ5_1),
            "Q2_K" => Some(Self::MostlyQ2K),
            "Q2_K_S" => Some(Self::MostlyQ2KS),
            "Q3_K_S" => Some(Self::MostlyQ3KS),
            "Q3_K_M" => Some(Self::MostlyQ3KM),
            "Q3_K_L" => Some(Self::MostlyQ3KL),
            "Q4_K_S" => Some(Self::MostlyQ4KS),
            "Q4_K_M" => Some(Self::MostlyQ4KM),
            "Q5_K_S" => Some(Self::MostlyQ5KS),
            "Q5_K_M" => Some(Self::MostlyQ5KM),
            "Q6_K" => Some(Self::MostlyQ6K),
            "IQ1_S" => Some(Self::MostlyIQ1S),
            "IQ1_M" => Some(Self::MostlyIQ1M),
            "IQ2_XXS" => Some(Self::MostlyIQ2XXS),
            "IQ2_XS" => Some(Self::MostlyIQ2XS),
            "IQ2_S" => Some(Self::MostlyIQ2S),
            "IQ2_M" => Some(Self::MostlyIQ2M),
            "IQ3_XXS" => Some(Self::MostlyIQ3XXS),
            "IQ3_XS" => Some(Self::MostlyIQ3XS),
            "IQ3_S" => Some(Self::MostlyIQ3S),
            "IQ3_M" => Some(Self::MostlyIQ3M),
            "IQ4_NL" => Some(Self::MostlyIQ4NL),
            "IQ4_XS" => Some(Self::MostlyIQ4XS),
            "TQ1_0" => Some(Self::MostlyTQ1_0),
            "TQ2_0" => Some(Self::MostlyTQ2_0),
            "MXFP4_MOE" => Some(Self::MostlyMXFP4Moe),
            "NVFP4" => Some(Self::MostlyNVFP4),
            _ => None,
        }
    }

    /// All available types, ordered roughly from largest to smallest.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::AllF32,
            Self::MostlyF16,
            Self::MostlyBF16,
            Self::MostlyQ8_0,
            Self::MostlyQ6K,
            Self::MostlyQ5KM,
            Self::MostlyQ5KS,
            Self::MostlyQ5_1,
            Self::MostlyQ5_0,
            Self::MostlyQ4KM,
            Self::MostlyQ4KS,
            Self::MostlyQ4_1,
            Self::MostlyQ4_0,
            Self::MostlyQ3KL,
            Self::MostlyQ3KM,
            Self::MostlyQ3KS,
            Self::MostlyQ2KS,
            Self::MostlyQ2K,
            Self::MostlyIQ4XS,
            Self::MostlyIQ4NL,
            Self::MostlyIQ3S,
            Self::MostlyIQ3M,
            Self::MostlyIQ3XS,
            Self::MostlyIQ3XXS,
            Self::MostlyIQ2M,
            Self::MostlyIQ2S,
            Self::MostlyIQ2XS,
            Self::MostlyIQ2XXS,
            Self::MostlyIQ1M,
            Self::MostlyIQ1S,
            Self::MostlyTQ1_0,
            Self::MostlyTQ2_0,
            Self::MostlyMXFP4Moe,
            Self::MostlyNVFP4,
        ]
    }
}

impl From<LlamaFtype> for llama_cpp_sys_4::llama_ftype {
    fn from(t: LlamaFtype) -> Self {
        t as llama_cpp_sys_4::llama_ftype
    }
}

impl std::fmt::Display for LlamaFtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GgmlType
// ─────────────────────────────────────────────────────────────────────────────

/// GGML tensor storage type (maps to `ggml_type`).
///
/// Used to set [`QuantizeParams::output_tensor_type`] and
/// [`QuantizeParams::token_embedding_type`], and for per-tensor type overrides
/// in [`TensorTypeOverride`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
}

impl From<GgmlType> for llama_cpp_sys_4::ggml_type {
    fn from(t: GgmlType) -> Self {
        t as llama_cpp_sys_4::ggml_type
    }
}

impl TryFrom<llama_cpp_sys_4::ggml_type> for GgmlType {
    type Error = llama_cpp_sys_4::ggml_type;
    fn try_from(v: llama_cpp_sys_4::ggml_type) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::IQ2XXS),
            17 => Ok(Self::IQ2XS),
            18 => Ok(Self::IQ3XXS),
            19 => Ok(Self::IQ1S),
            20 => Ok(Self::IQ4NL),
            21 => Ok(Self::IQ3S),
            22 => Ok(Self::IQ2S),
            23 => Ok(Self::IQ4XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1M),
            30 => Ok(Self::BF16),
            34 => Ok(Self::TQ1_0),
            35 => Ok(Self::TQ2_0),
            39 => Ok(Self::MXFP4),
            40 => Ok(Self::NVFP4),
            _ => Err(v),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ImatrixEntry / Imatrix
// ─────────────────────────────────────────────────────────────────────────────

/// A single per-tensor importance matrix entry, as loaded from a `.imatrix` file.
///
/// Each entry contains activation statistics for one model tensor collected from
/// a calibration dataset. When supplied to [`QuantizeParams::with_imatrix`] these
/// statistics guide the quantizer to allocate more precision to weights that
/// matter most.
#[derive(Debug, Clone)]
pub struct ImatrixEntry {
    name: CString,
    data: Vec<f32>,
}

impl ImatrixEntry {
    /// Create a new entry from a tensor name and its importance scores.
    ///
    /// # Errors
    ///
    /// Returns [`NulError`] if `name` contains an interior null byte.
    pub fn new(name: impl Into<Vec<u8>>, data: Vec<f32>) -> Result<Self, NulError> {
        Ok(Self {
            name: CString::new(name)?,
            data,
        })
    }

    /// Tensor name.
    #[must_use]
    pub fn name_str(&self) -> &str {
        self.name.to_str().unwrap_or("")
    }

    /// Number of importance values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the data slice is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// A collection of importance matrix entries (one per quantized tensor).
///
/// Build one by pushing [`ImatrixEntry`] values, then pass it to
/// [`QuantizeParams::with_imatrix`].
#[derive(Debug, Clone, Default)]
pub struct Imatrix {
    entries: Vec<ImatrixEntry>,
}

impl Imatrix {
    /// Create an empty imatrix.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry.
    pub fn push(&mut self, entry: ImatrixEntry) {
        self.entries.push(entry);
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no entries have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TensorTypeOverride
// ─────────────────────────────────────────────────────────────────────────────

/// Override the quantization type of every tensor whose name matches a glob `pattern`.
///
/// The pattern syntax is the same as used by the `--tensor-type` flag in
/// `llama-quantize`, e.g. `"attn.*"` or `"blk.0.*"`.
///
/// # Example
///
/// ```
/// use llama_cpp_4::quantize::{GgmlType, TensorTypeOverride};
///
/// // Keep the output projection in F16:
/// let ov = TensorTypeOverride::new("output", GgmlType::F16).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TensorTypeOverride {
    pattern: CString,
    ty: GgmlType,
}

impl TensorTypeOverride {
    /// Create a new override.
    ///
    /// # Errors
    ///
    /// Returns [`NulError`] if `pattern` contains an interior null byte.
    pub fn new(pattern: impl Into<Vec<u8>>, ty: GgmlType) -> Result<Self, NulError> {
        Ok(Self {
            pattern: CString::new(pattern)?,
            ty,
        })
    }

    /// The glob pattern that selects tensors.
    #[must_use]
    pub fn pattern_str(&self) -> &str {
        self.pattern.to_str().unwrap_or("")
    }

    /// The type to assign to matching tensors.
    #[must_use]
    pub fn ty(&self) -> GgmlType {
        self.ty
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KvOverrideValue / KvOverride
// ─────────────────────────────────────────────────────────────────────────────

/// A value in a GGUF key-value metadata override.
#[derive(Debug, Clone, PartialEq)]
pub enum KvOverrideValue {
    /// 64-bit integer
    Int(i64),
    /// 64-bit float
    Float(f64),
    /// Boolean
    Bool(bool),
    /// Fixed-length string (up to 127 bytes + NUL)
    Str([std::os::raw::c_char; 128]),
}

/// A single GGUF metadata key-value override.
///
/// These are written into the output file's metadata when quantizing.
#[derive(Debug, Clone)]
pub struct KvOverride {
    key: CString,
    /// The value for this override.
    pub value: KvOverrideValue,
}

impl KvOverride {
    /// Create a new override.
    ///
    /// # Errors
    ///
    /// Returns [`NulError`] if `key` contains an interior null byte.
    pub fn new(key: impl Into<Vec<u8>>, value: KvOverrideValue) -> Result<Self, NulError> {
        Ok(Self {
            key: CString::new(key)?,
            value,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QuantizeParams
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters for quantizing a model.
///
/// Create with [`QuantizeParams::new`] and chain `with_*` builder methods to
/// configure, then pass a reference to [`crate::model_quantize`].
///
/// # Example
///
/// ```no_run
/// use llama_cpp_4::quantize::{GgmlType, LlamaFtype, QuantizeParams, TensorTypeOverride};
///
/// let ov = TensorTypeOverride::new("output", GgmlType::F16).unwrap();
///
/// let params = QuantizeParams::new(LlamaFtype::MostlyQ4KM)
///     .with_nthread(8)
///     .with_allow_requantize(false)
///     .with_quantize_output_tensor(true)
///     .with_pure(false)
///     .with_tensor_type_override(ov);
///
/// llama_cpp_4::model_quantize("in.gguf", "out.gguf", &params).unwrap();
/// ```
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct QuantizeParams {
    /// Number of threads (0 = auto-detect).
    pub nthread: i32,
    /// Target quantization type.
    pub ftype: LlamaFtype,
    /// Force this storage type for the output/lm-head tensor (`None` = use ftype default).
    pub output_tensor_type: Option<GgmlType>,
    /// Force this storage type for the token-embedding tensor (`None` = use ftype default).
    pub token_embedding_type: Option<GgmlType>,
    /// Allow re-quantizing tensors that are already quantized.
    pub allow_requantize: bool,
    /// Quantize the output/lm-head weight tensor.
    pub quantize_output_tensor: bool,
    /// Copy all tensors without quantizing (ignores `ftype`).
    pub only_copy: bool,
    /// Quantize every tensor to the same type (no mixed k-quant strategy).
    pub pure: bool,
    /// Keep the same number of shards as the input (for split models).
    pub keep_split: bool,
    /// Estimate output size without writing anything to disk.
    pub dry_run: bool,

    imatrix: Vec<ImatrixEntry>,
    kv_overrides: Vec<KvOverride>,
    tt_overrides: Vec<TensorTypeOverride>,
    prune_layers: Vec<i32>,
}

impl QuantizeParams {
    /// Create a new params set targeting `ftype`.
    ///
    /// All other options are set to the same defaults as
    /// `llama_model_quantize_default_params()`.
    #[must_use]
    pub fn new(ftype: LlamaFtype) -> Self {
        // Read the C defaults so we match them exactly.
        let d = unsafe { llama_cpp_sys_4::llama_model_quantize_default_params() };
        Self {
            nthread: d.nthread,
            ftype,
            output_tensor_type: GgmlType::try_from(d.output_tensor_type).ok(),
            token_embedding_type: GgmlType::try_from(d.token_embedding_type).ok(),
            allow_requantize: d.allow_requantize,
            quantize_output_tensor: d.quantize_output_tensor,
            only_copy: d.only_copy,
            pure: d.pure_,
            keep_split: d.keep_split,
            dry_run: d.dry_run,
            imatrix: Vec::new(),
            kv_overrides: Vec::new(),
            tt_overrides: Vec::new(),
            prune_layers: Vec::new(),
        }
    }

    /// Set the number of quantization threads (`0` = auto).
    #[must_use]
    pub fn with_nthread(mut self, n: i32) -> Self {
        self.nthread = n;
        self
    }

    /// Override the output-tensor storage type.
    #[must_use]
    pub fn with_output_tensor_type(mut self, ty: GgmlType) -> Self {
        self.output_tensor_type = Some(ty);
        self
    }

    /// Override the token-embedding storage type.
    #[must_use]
    pub fn with_token_embedding_type(mut self, ty: GgmlType) -> Self {
        self.token_embedding_type = Some(ty);
        self
    }

    /// Allow (or disallow) re-quantizing already-quantized tensors.
    #[must_use]
    pub fn with_allow_requantize(mut self, v: bool) -> Self {
        self.allow_requantize = v;
        self
    }

    /// Quantize the output/lm-head weight (`true` by default).
    #[must_use]
    pub fn with_quantize_output_tensor(mut self, v: bool) -> Self {
        self.quantize_output_tensor = v;
        self
    }

    /// When `true`, only copy tensors verbatim (no quantization at all).
    #[must_use]
    pub fn with_only_copy(mut self, v: bool) -> Self {
        self.only_copy = v;
        self
    }

    /// When `true`, quantize all tensors to the same type (no mixed k-quant strategy).
    #[must_use]
    pub fn with_pure(mut self, v: bool) -> Self {
        self.pure = v;
        self
    }

    /// Preserve the number of shards when quantizing a split model.
    #[must_use]
    pub fn with_keep_split(mut self, v: bool) -> Self {
        self.keep_split = v;
        self
    }

    /// Only estimate the output size; do not write anything to disk.
    #[must_use]
    pub fn with_dry_run(mut self, v: bool) -> Self {
        self.dry_run = v;
        self
    }

    /// Supply importance matrix data to improve quantization quality.
    ///
    /// The imatrix is generated by the `imatrix` tool (or the `imatrix` example
    /// in this crate) and contains per-tensor activation statistics collected
    /// from a calibration dataset.
    #[must_use]
    pub fn with_imatrix(mut self, imatrix: Imatrix) -> Self {
        self.imatrix = imatrix.entries;
        self
    }

    /// Append a single imatrix entry.
    #[must_use]
    pub fn with_imatrix_entry(mut self, entry: ImatrixEntry) -> Self {
        self.imatrix.push(entry);
        self
    }

    /// Add (or replace) a GGUF metadata key-value pair in the output file.
    #[must_use]
    pub fn with_kv_override(mut self, kv: KvOverride) -> Self {
        self.kv_overrides.push(kv);
        self
    }

    /// Override the quantization type for tensors whose name matches `pattern`.
    ///
    /// Can be called multiple times; overrides are applied in order.
    #[must_use]
    pub fn with_tensor_type_override(mut self, ov: TensorTypeOverride) -> Self {
        self.tt_overrides.push(ov);
        self
    }

    /// Mark a layer index for pruning (removal) from the output model.
    #[must_use]
    pub fn with_pruned_layer(mut self, layer: i32) -> Self {
        self.prune_layers.push(layer);
        self
    }

    /// Mark multiple layer indices for pruning.
    #[must_use]
    pub fn with_pruned_layers(mut self, layers: impl IntoIterator<Item = i32>) -> Self {
        self.prune_layers.extend(layers);
        self
    }

    /// Build the raw C struct, together with the temporary backing storage
    /// that must outlive the struct.  Returns `(raw_params, _guards)`.
    ///
    /// This is `pub(crate)` so that `model_quantize` can call it safely while
    /// holding all the guards alive.
    pub(crate) fn to_raw(&self) -> RawQuantizeParamsGuard<'_> {
        // ── imatrix ─────────────────────────────────────────────────────────
        // Build a null-terminated array of llama_model_imatrix_data.
        // The `name` and `data` pointers point directly into our owned Vecs.
        let imatrix_c: Vec<llama_cpp_sys_4::llama_model_imatrix_data> = self
            .imatrix
            .iter()
            .map(|e| llama_cpp_sys_4::llama_model_imatrix_data {
                name: e.name.as_ptr(),
                data: e.data.as_ptr(),
                size: e.data.len(),
            })
            .chain(std::iter::once(llama_cpp_sys_4::llama_model_imatrix_data {
                name: null(),
                data: null(),
                size: 0,
            }))
            .collect();

        // ── kv_overrides ────────────────────────────────────────────────────
        // null-terminated by a sentinel with key[0] == 0
        let kv_c: Vec<llama_cpp_sys_4::llama_model_kv_override> = self
            .kv_overrides
            .iter()
            .map(|kv| {
                let mut raw = llama_cpp_sys_4::llama_model_kv_override {
                    key: [0; 128],
                    tag: 0,
                    __bindgen_anon_1: llama_cpp_sys_4::llama_model_kv_override__bindgen_ty_1 {
                        val_i64: 0,
                    },
                };
                // Copy key bytes (up to 127 chars + NUL).
                let bytes = kv.key.to_bytes_with_nul();
                let copy_len = bytes.len().min(128);
                for (dst, &src) in raw.key.iter_mut().zip(bytes[..copy_len].iter()) {
                    *dst = src as std::os::raw::c_char;
                }
                match &kv.value {
                    KvOverrideValue::Int(v) => {
                        raw.tag = llama_cpp_sys_4::LLAMA_KV_OVERRIDE_TYPE_INT;
                        raw.__bindgen_anon_1 =
                            llama_cpp_sys_4::llama_model_kv_override__bindgen_ty_1 {
                                val_i64: *v,
                            };
                    }
                    KvOverrideValue::Float(v) => {
                        raw.tag = llama_cpp_sys_4::LLAMA_KV_OVERRIDE_TYPE_FLOAT;
                        raw.__bindgen_anon_1 =
                            llama_cpp_sys_4::llama_model_kv_override__bindgen_ty_1 {
                                val_f64: *v,
                            };
                    }
                    KvOverrideValue::Bool(v) => {
                        raw.tag = llama_cpp_sys_4::LLAMA_KV_OVERRIDE_TYPE_BOOL;
                        raw.__bindgen_anon_1 =
                            llama_cpp_sys_4::llama_model_kv_override__bindgen_ty_1 {
                                val_bool: *v,
                            };
                    }
                    KvOverrideValue::Str(s) => {
                        raw.tag = llama_cpp_sys_4::LLAMA_KV_OVERRIDE_TYPE_STR;
                        raw.__bindgen_anon_1 =
                            llama_cpp_sys_4::llama_model_kv_override__bindgen_ty_1 {
                                val_str: *s,
                            };
                    }
                }
                raw
            })
            .chain(std::iter::once(llama_cpp_sys_4::llama_model_kv_override {
                key: [0; 128],
                tag: 0,
                __bindgen_anon_1: llama_cpp_sys_4::llama_model_kv_override__bindgen_ty_1 {
                    val_i64: 0,
                },
            }))
            .collect();

        // ── tt_overrides ────────────────────────────────────────────────────
        // null-terminated by { null, GGML_TYPE_COUNT }
        let tt_c: Vec<llama_cpp_sys_4::llama_model_tensor_override> = self
            .tt_overrides
            .iter()
            .map(|ov| llama_cpp_sys_4::llama_model_tensor_override {
                pattern: ov.pattern.as_ptr(),
                type_: ov.ty as llama_cpp_sys_4::ggml_type,
            })
            .chain(std::iter::once(
                llama_cpp_sys_4::llama_model_tensor_override {
                    pattern: null(),
                    type_: llama_cpp_sys_4::GGML_TYPE_COUNT,
                },
            ))
            .collect();

        // ── prune_layers ─────────────────────────────────────────────────────
        // -1-terminated
        let mut prune_c = self.prune_layers.clone();
        prune_c.push(-1);

        // ── assemble ────────────────────────────────────────────────────────
        let raw = llama_cpp_sys_4::llama_model_quantize_params {
            nthread: self.nthread,
            ftype: self.ftype as llama_cpp_sys_4::llama_ftype,
            output_tensor_type: self
                .output_tensor_type
                .map(|t| t as llama_cpp_sys_4::ggml_type)
                .unwrap_or(llama_cpp_sys_4::GGML_TYPE_COUNT),
            token_embedding_type: self
                .token_embedding_type
                .map(|t| t as llama_cpp_sys_4::ggml_type)
                .unwrap_or(llama_cpp_sys_4::GGML_TYPE_COUNT),
            allow_requantize: self.allow_requantize,
            quantize_output_tensor: self.quantize_output_tensor,
            only_copy: self.only_copy,
            pure_: self.pure,
            keep_split: self.keep_split,
            dry_run: self.dry_run,
            imatrix: if self.imatrix.is_empty() {
                null()
            } else {
                imatrix_c.as_ptr()
            },
            kv_overrides: if self.kv_overrides.is_empty() {
                null()
            } else {
                kv_c.as_ptr()
            },
            tt_overrides: if self.tt_overrides.is_empty() {
                null()
            } else {
                tt_c.as_ptr()
            },
            prune_layers: if self.prune_layers.is_empty() {
                null()
            } else {
                prune_c.as_ptr()
            },
        };

        RawQuantizeParamsGuard {
            raw,
            _imatrix_c: imatrix_c,
            _kv_c: kv_c,
            _tt_c: tt_c,
            _prune_c: prune_c,
            _marker: std::marker::PhantomData,
        }
    }
}

/// Temporary storage that keeps the C pointers inside a raw
/// `llama_model_quantize_params` valid.  Dropped after the quantize call.
pub(crate) struct RawQuantizeParamsGuard<'a> {
    pub(crate) raw: llama_cpp_sys_4::llama_model_quantize_params,
    _imatrix_c: Vec<llama_cpp_sys_4::llama_model_imatrix_data>,
    _kv_c: Vec<llama_cpp_sys_4::llama_model_kv_override>,
    _tt_c: Vec<llama_cpp_sys_4::llama_model_tensor_override>,
    _prune_c: Vec<i32>,
    // tie lifetime to the source QuantizeParams so the string/data
    // pointers inside imatrix_c and tt_c stay valid
    _marker: std::marker::PhantomData<&'a QuantizeParams>,
}

// ─────────────────────────────────────────────────────────────────────────────
// TurboQuant – attention rotation
// ─────────────────────────────────────────────────────────────────────────────

/// Control the TurboQuant attention-rotation feature globally.
///
/// When enabled (the default), llama.cpp applies a Hadamard rotation to Q/K/V
/// tensors before storing them in the KV cache.  This significantly improves
/// quantization quality of the KV cache with near-zero overhead, as described
/// in llama.cpp PR #21038.
///
/// This function sets or clears the `LLAMA_ATTN_ROT_DISABLE` environment
/// variable, which llama.cpp reads once when a context (and its KV cache) is
/// first created.  Call it **before** creating any [`LlamaContext`] on the
/// current process.
///
/// # Thread safety
///
/// Mutating environment variables while other threads may be reading them is
/// undefined behaviour.  Call this function before spawning any threads that
/// use llama contexts, or ensure no contexts are being created concurrently.
///
/// # Example
///
/// ```no_run
/// // Disable the rotation for benchmarking purposes:
/// llama_cpp_4::quantize::set_attn_rot_disabled(true);
///
/// // Re-enable (default behaviour):
/// llama_cpp_4::quantize::set_attn_rot_disabled(false);
/// ```
///
/// [`LlamaContext`]: crate::context::LlamaContext
pub fn set_attn_rot_disabled(disabled: bool) {
    if disabled {
        // SAFETY: single-threaded context required by the caller.
        #[allow(unused_unsafe)]
        unsafe {
            std::env::set_var("LLAMA_ATTN_ROT_DISABLE", "1");
        }
    } else {
        #[allow(unused_unsafe)]
        unsafe {
            std::env::remove_var("LLAMA_ATTN_ROT_DISABLE");
        }
    }
}

/// Returns `true` if TurboQuant attention rotation is currently disabled.
#[must_use]
pub fn attn_rot_disabled() -> bool {
    std::env::var("LLAMA_ATTN_ROT_DISABLE")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .map_or(false, |v| v != 0)
}
