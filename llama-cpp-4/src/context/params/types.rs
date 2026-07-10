/// A rusty wrapper around `llama_context_type`.
//
// Cast the sys constants to `u32` so the discriminants compile on both clang
// (where bindgen emits `c_uint`, making the cast a no-op) and MSVC (where it
// emits `c_int` and the cast is required). The `unnecessary_cast` allow covers
// the clang/gcc case where the source is already `u32`.
#[allow(clippy::unnecessary_cast)]
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LlamaContextType {
    /// Default context (standard inference).
    Default = llama_cpp_sys_4::LLAMA_CONTEXT_TYPE_DEFAULT as u32,
    /// Multi-token-prediction draft context, used as the draft side of
    /// speculative decoding. Pair with [`crate::mtp::MtpSession`].
    Mtp = llama_cpp_sys_4::LLAMA_CONTEXT_TYPE_MTP as u32,
}

impl From<llama_cpp_sys_4::llama_context_type> for LlamaContextType {
    fn from(value: llama_cpp_sys_4::llama_context_type) -> Self {
        if value == llama_cpp_sys_4::LLAMA_CONTEXT_TYPE_MTP {
            Self::Mtp
        } else {
            Self::Default
        }
    }
}

impl From<LlamaContextType> for llama_cpp_sys_4::llama_context_type {
    fn from(value: LlamaContextType) -> Self {
        value as u32 as Self
    }
}

/// Attention mask mode used when the context runs in embedding mode.
///
/// Maps to `llama_context_params.attention_type`. Use [`LlamaAttentionType::NonCausal`]
/// for encoder / bi-directional embedding models and [`LlamaAttentionType::Causal`]
/// for standard decoder models.
#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LlamaAttentionType {
    /// Unspecified — use model default.
    Unspecified = llama_cpp_sys_4::LLAMA_ATTENTION_TYPE_UNSPECIFIED,
    /// Causal attention (decoder-only).
    Causal = llama_cpp_sys_4::LLAMA_ATTENTION_TYPE_CAUSAL,
    /// Non-causal attention (e.g. embedding / encoder models).
    NonCausal = llama_cpp_sys_4::LLAMA_ATTENTION_TYPE_NON_CAUSAL,
}

impl From<i32> for LlamaAttentionType {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::Causal,
            1 => Self::NonCausal,
            _ => Self::Unspecified,
        }
    }
}

impl From<LlamaAttentionType> for i32 {
    fn from(value: LlamaAttentionType) -> Self {
        value as i32
    }
}

/// Flash-attention enablement policy for the context.
///
/// Maps to `llama_context_params.flash_attn_type`. [`LlamaFlashAttnType::Auto`]
/// lets llama.cpp pick based on model architecture and active backend.
#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LlamaFlashAttnType {
    /// Let llama.cpp decide based on model and backend.
    Auto = llama_cpp_sys_4::LLAMA_FLASH_ATTN_TYPE_AUTO,
    /// Force flash attention off.
    Disabled = llama_cpp_sys_4::LLAMA_FLASH_ATTN_TYPE_DISABLED,
    /// Force flash attention on.
    Enabled = llama_cpp_sys_4::LLAMA_FLASH_ATTN_TYPE_ENABLED,
}

impl From<i32> for LlamaFlashAttnType {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::Disabled,
            1 => Self::Enabled,
            _ => Self::Auto,
        }
    }
}

impl From<LlamaFlashAttnType> for i32 {
    fn from(value: LlamaFlashAttnType) -> Self {
        value as i32
    }
}

/// A rusty wrapper around `rope_scaling_type`.
#[repr(i8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RopeScalingType {
    /// The scaling type is unspecified
    Unspecified = -1,
    /// No scaling
    None = 0,
    /// Linear scaling
    Linear = 1,
    /// Yarn scaling
    Yarn = 2,
}

/// Create a `RopeScalingType` from a `c_int` - returns `RopeScalingType::ScalingUnspecified` if
/// the value is not recognized.
impl From<i32> for RopeScalingType {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::None,
            1 => Self::Linear,
            2 => Self::Yarn,
            _ => Self::Unspecified,
        }
    }
}

/// Create a `c_int` from a `RopeScalingType`.
impl From<RopeScalingType> for i32 {
    fn from(value: RopeScalingType) -> Self {
        match value {
            RopeScalingType::None => 0,
            RopeScalingType::Linear => 1,
            RopeScalingType::Yarn => 2,
            RopeScalingType::Unspecified => -1,
        }
    }
}

/// A rusty wrapper around `LLAMA_POOLING_TYPE`.
#[repr(i8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LlamaPoolingType {
    /// The pooling type is unspecified
    Unspecified = -1,
    /// No pooling    
    None = 0,
    /// Mean pooling
    Mean = 1,
    /// CLS pooling
    Cls = 2,
    /// Last pooling
    Last = 3,
    /// Rank pooling (used by reranking / cross-encoder models).
    Rank = 4,
}

/// Create a `LlamaPoolingType` from a `c_int` - returns `LlamaPoolingType::Unspecified` if
/// the value is not recognized.
impl From<i32> for LlamaPoolingType {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::None,
            1 => Self::Mean,
            2 => Self::Cls,
            3 => Self::Last,
            4 => Self::Rank,
            _ => Self::Unspecified,
        }
    }
}

/// Create a `c_int` from a `LlamaPoolingType`.
impl From<LlamaPoolingType> for i32 {
    fn from(value: LlamaPoolingType) -> Self {
        match value {
            LlamaPoolingType::None => 0,
            LlamaPoolingType::Mean => 1,
            LlamaPoolingType::Cls => 2,
            LlamaPoolingType::Last => 3,
            LlamaPoolingType::Rank => 4,
            LlamaPoolingType::Unspecified => -1,
        }
    }
}
