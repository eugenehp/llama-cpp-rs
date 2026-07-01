//! Memory estimation and parameter fitting from llama.cpp `common/fit`.
//!
//! - [`get_device_memory_data`] — project per-device memory for a parameter set
//!   without keeping a context alive.
//! - [`fit_params`] — adjust [`LlamaModelParams`] / [`LlamaContextParams`] to fit
//!   available device memory (upstream `common_fit_params`).
//!
//! # Example — memory estimate
//!
//! ```no_run
//! use llama_cpp_4::prelude::*;
//! use std::path::Path;
//!
//! fn main() {
//!     let _backend = LlamaBackend::init().unwrap();
//!     let report = get_device_memory_data(
//!         Path::new("model.gguf"),
//!         &LlamaModelParams::default().with_n_gpu_layers(99),
//!         &LlamaContextParams::default(),
//!         llama_cpp_sys_4::GGML_LOG_LEVEL_ERROR,
//!     )
//!     .unwrap();
//!
//!     println!("training ctx: {}", report.hyperparams.n_ctx_train);
//!     for (i, entry) in report.entries.iter().enumerate() {
//!         println!(
//!             "device {i}: {} bytes free / {} total (projected {})",
//!             entry.free,
//!             entry.total,
//!             entry.used(),
//!         );
//!     }
//! }
//! ```
//!
//! # Example — auto-fit parameters
//!
//! ```no_run
//! use llama_cpp_4::fit::{fit_params, FitParams};
//! use llama_cpp_4::prelude::*;
//! use std::path::Path;
//!
//! fn main() {
//!     let backend = LlamaBackend::init().unwrap();
//!     let result = fit_params(
//!         &backend,
//!         Path::new("model.gguf"),
//!         FitParams::default().with_n_ctx_min(512),
//!     )
//!     .unwrap();
//!
//!     use std::num::NonZeroU32;
//!
//!     println!("n_ctx: {}", result.context_params.n_ctx().map_or(0, NonZeroU32::get));
//!     println!("n_gpu_layers: {}", result.model_params.n_gpu_layers());
//! }
//! ```

use std::ffi::CString;
use std::path::Path;
use std::ptr::{null, null_mut};

use thiserror::Error;

use crate::context::params::LlamaContextParams;
use crate::llama_backend::LlamaBackend;
use crate::model::params::LlamaModelParams;
use crate::{max_devices, max_tensor_buft_overrides};

/// Per-device memory projection from [`get_device_memory_data`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceMemoryEntry {
    /// Total device memory in bytes.
    pub total: i64,
    /// Free device memory in bytes at query time.
    pub free: i64,
    /// Projected model weight bytes on this device.
    pub model: usize,
    /// Projected KV / recurrent cache bytes.
    pub context: usize,
    /// Projected temporary compute buffer bytes.
    pub compute: usize,
}

impl DeviceMemoryEntry {
    /// Sum of model, context, and compute bytes.
    #[must_use]
    pub fn used(&self) -> usize {
        self.model + self.context + self.compute
    }
}

/// Hyper-parameters discovered while estimating device memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryHyperParams {
    /// Model `n_gpu_layers` hyper-parameter used for the estimate.
    pub n_gpu_layers: u32,
    /// Model training context length.
    pub n_ctx_train: u32,
    /// Number of `MoE` experts (`0` when dense).
    pub n_expert: u32,
}

/// Result of [`get_device_memory_data`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceMemoryReport {
    /// Per-device memory breakdown (one entry per backend device).
    pub entries: Vec<DeviceMemoryEntry>,
    /// Hyper-parameters read from the checkpoint during estimation.
    pub hyperparams: DeviceMemoryHyperParams,
}

/// Errors from [`get_device_memory_data`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DeviceMemoryError {
    /// The model path could not be encoded as a C string.
    #[error("invalid model path")]
    InvalidPath,
    /// The underlying C++ helper failed (model missing, incompatible params, …).
    #[error("device memory query failed")]
    QueryFailed,
    /// More devices were reported than the internal buffer allows.
    #[error("device memory entry buffer overflow")]
    BufferOverflow,
}

/// Estimate per-device memory for a model path and parameter set.
///
/// This wraps `common_get_device_memory_data` through `ext_shim`. The model is
/// loaded with `no_alloc` and freed before returning; no context is kept alive.
///
/// # Errors
///
/// Returns [`DeviceMemoryError`] when the path is invalid or llama.cpp cannot
/// produce an estimate.
pub fn get_device_memory_data(
    path_model: &Path,
    mparams: &LlamaModelParams,
    cparams: &LlamaContextParams,
    log_level: llama_cpp_sys_4::ggml_log_level,
) -> Result<DeviceMemoryReport, DeviceMemoryError> {
    let path = CString::new(path_model.to_string_lossy().as_ref())
        .map_err(|_| DeviceMemoryError::InvalidPath)?;

    let mparams = mparams.params;
    let cparams = cparams.context_params;

    let mut capacity = 8usize;
    loop {
        let mut raw = vec![
            llama_cpp_sys_4::common_device_memory_flat_entry {
                total: 0,
                free: 0,
                model: 0,
                context: 0,
                compute: 0,
            };
            capacity
        ];
        let mut hp_ngl = 0u32;
        let mut hp_nct = 0u32;
        let mut hp_nex = 0u32;

        let n = unsafe {
            llama_cpp_sys_4::common_device_memory_collect(
                path.as_ptr(),
                &raw const mparams,
                &raw const cparams,
                log_level,
                raw.as_mut_ptr(),
                capacity,
                &raw mut hp_ngl,
                &raw mut hp_nct,
                &raw mut hp_nex,
            )
        };

        if n == usize::MAX {
            return Err(DeviceMemoryError::QueryFailed);
        }

        if n < capacity {
            let entries = raw
                .into_iter()
                .take(n)
                .map(|e| DeviceMemoryEntry {
                    total: e.total,
                    free: e.free,
                    model: e.model,
                    context: e.context,
                    compute: e.compute,
                })
                .collect();
            return Ok(DeviceMemoryReport {
                entries,
                hyperparams: DeviceMemoryHyperParams {
                    n_gpu_layers: hp_ngl,
                    n_ctx_train: hp_nct,
                    n_expert: hp_nex,
                },
            });
        }

        capacity = capacity.saturating_mul(2);
        if capacity > 256 {
            return Err(DeviceMemoryError::BufferOverflow);
        }
    }
}

const DEFAULT_MARGIN_BYTES: usize = 1024 * 1024 * 1024;

/// Input to [`fit_params`].
///
/// Defaults mirror upstream `common_params`: unset `n_ctx` (`0`) so context size
/// can be reduced, default model params so `n_gpu_layers` may be adjusted, and
/// 1 GiB per-device memory margins.
#[derive(Debug)]
pub struct FitParams {
    /// Starting model parameters. Only fields still at their defaults are modified.
    pub model_params: LlamaModelParams,
    /// Starting context parameters. Set `n_ctx` to `0` via
    /// [`LlamaContextParams::with_n_ctx`]`(None)` to let fitting pick a context size.
    pub context_params: LlamaContextParams,
    /// Minimum free memory to leave on each device, in bytes (one entry per device).
    pub margins: Vec<usize>,
    /// Minimum context size when fitting must reduce `n_ctx`.
    pub n_ctx_min: u32,
    /// Minimum log level printed during fitting.
    pub log_level: llama_cpp_sys_4::ggml_log_level,
}

impl Default for FitParams {
    fn default() -> Self {
        let nd = max_devices();
        Self {
            model_params: LlamaModelParams::default(),
            context_params: LlamaContextParams::default().with_n_ctx(None),
            margins: vec![DEFAULT_MARGIN_BYTES; nd],
            n_ctx_min: 4096,
            log_level: llama_cpp_sys_4::GGML_LOG_LEVEL_ERROR,
        }
    }
}

impl FitParams {
    /// Override starting model parameters.
    #[must_use]
    pub fn with_model_params(mut self, model_params: LlamaModelParams) -> Self {
        self.model_params = model_params;
        self
    }

    /// Override starting context parameters.
    #[must_use]
    pub fn with_context_params(mut self, context_params: LlamaContextParams) -> Self {
        self.context_params = context_params;
        self
    }

    /// Per-device memory margins in bytes (length must be at least [`max_devices`]`()`).
    #[must_use]
    pub fn with_margins(mut self, margins: Vec<usize>) -> Self {
        self.margins = margins;
        self
    }

    /// Minimum context size when fitting reduces memory by shrinking `n_ctx`.
    #[must_use]
    pub fn with_n_ctx_min(mut self, n_ctx_min: u32) -> Self {
        self.n_ctx_min = n_ctx_min;
        self
    }

    /// Minimum log level emitted while fitting.
    #[must_use]
    pub fn with_log_level(mut self, log_level: llama_cpp_sys_4::ggml_log_level) -> Self {
        self.log_level = log_level;
        self
    }
}

/// Fitted model/context parameters plus auxiliary buffers.
///
/// [`LlamaModelParams`] inside this struct may point at [`Self::tensor_split`] and
/// its internal tensor buffer-type override storage; keep the whole `FitParamsResult` alive while
/// loading a model with these parameters.
#[derive(Debug)]
pub struct FitParamsResult {
    /// Model parameters after fitting (`n_gpu_layers`, tensor split, …).
    pub model_params: LlamaModelParams,
    /// Context parameters after fitting (`n_ctx`, …).
    pub context_params: LlamaContextParams,
    /// Layer split ratios per device (writable buffer passed to llama.cpp).
    pub tensor_split: Vec<f32>,
    /// Tensor buffer-type overrides written by fitting (keeps pointers valid).
    #[allow(dead_code)]
    pub(crate) tensor_buft_overrides: Vec<llama_cpp_sys_4::llama_model_tensor_buft_override>,
    /// Per-device memory margins used during fitting (bytes).
    pub margins: Vec<usize>,
}

impl FitParamsResult {
    /// Tensor split values for active devices, trimming trailing zeros.
    #[must_use]
    pub fn active_tensor_split(&self) -> &[f32] {
        let mut nd = self.tensor_split.len();
        while nd > 1 && self.tensor_split[nd - 1] == 0.0 {
            nd -= 1;
        }
        &self.tensor_split[..nd]
    }
}

/// Errors from [`fit_params`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum FitParamsError {
    /// The model path could not be encoded as a C string.
    #[error("invalid model path")]
    InvalidPath,
    /// Fitting could not find allocations that fit device memory.
    #[error("could not fit parameters to available device memory")]
    CouldNotFit,
    /// A hard error occurred (e.g. model file missing).
    #[error("parameter fitting failed")]
    Failed,
}

/// Adjust model and context parameters to fit available device memory.
///
/// Wraps `common_fit_params`. Requires an initialized [`LlamaBackend`]. The model
/// is probed with `no_alloc` internally; nothing is kept loaded on return.
///
/// Only model fields still equal to [`LlamaModelParams::default`] are modified
/// (except `n_gpu_layers` on macOS where the default is `-1`). Context `n_ctx`
/// is adjusted only when it is `0` — use [`LlamaContextParams::with_n_ctx`] with `None`.
///
/// # Errors
///
/// Returns [`FitParamsError::InvalidPath`] for bad paths,
/// [`FitParamsError::CouldNotFit`] when no allocation fits, and
/// [`FitParamsError::Failed`] on hard errors (missing model, incompatible params, …).
pub fn fit_params(
    _backend: &LlamaBackend,
    path_model: &Path,
    options: FitParams,
) -> Result<FitParamsResult, FitParamsError> {
    let path = CString::new(path_model.to_string_lossy().as_ref())
        .map_err(|_| FitParamsError::InvalidPath)?;

    let nd = max_devices();
    let mut tensor_split = vec![0.0_f32; nd];

    let ntbo = max_tensor_buft_overrides();
    let mut tensor_buft_overrides = vec![
        llama_cpp_sys_4::llama_model_tensor_buft_override {
            pattern: null(),
            buft: null_mut(),
        };
        ntbo + 1
    ];

    let mut margins = options.margins;
    if margins.len() < nd {
        margins.resize(nd, DEFAULT_MARGIN_BYTES);
    }

    let mut model_params = options.model_params;
    model_params.params.tensor_split = tensor_split.as_mut_ptr();
    model_params.params.tensor_buft_overrides = tensor_buft_overrides.as_mut_ptr();

    let mut context_params = options.context_params;

    let status = unsafe {
        llama_cpp_sys_4::common_fit_params(
            path.as_ptr(),
            &raw mut model_params.params,
            &raw mut context_params.context_params,
            tensor_split.as_mut_ptr(),
            tensor_buft_overrides.as_mut_ptr(),
            margins.as_mut_ptr(),
            options.n_ctx_min,
            options.log_level,
        )
    };

    match status {
        llama_cpp_sys_4::COMMON_PARAMS_FIT_STATUS_SUCCESS => {
            model_params.params.tensor_split = tensor_split.as_mut_ptr();
            model_params.params.tensor_buft_overrides = tensor_buft_overrides.as_mut_ptr();
            Ok(FitParamsResult {
                model_params,
                context_params,
                tensor_split,
                tensor_buft_overrides,
                margins,
            })
        }
        llama_cpp_sys_4::COMMON_PARAMS_FIT_STATUS_FAILURE => Err(FitParamsError::CouldNotFit),
        _ => Err(FitParamsError::Failed),
    }
}
