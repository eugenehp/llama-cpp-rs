//! Representation of an initialized llama backend

use crate::LLamaCppError;
use llama_cpp_sys_4::ggml_log_level;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::SeqCst;

/// Representation of an initialized llama backend
/// This is required as a parameter for most llama functions as the backend must be initialized
/// before any llama functions are called. This type is proof of initialization.
#[derive(Eq, PartialEq, Debug)]
pub struct LlamaBackend {}

static LLAMA_BACKEND_INITIALIZED: AtomicBool = AtomicBool::new(false);

impl LlamaBackend {
    /// Mark the llama backend as initialized
    fn mark_init() -> crate::Result<()> {
        match LLAMA_BACKEND_INITIALIZED.compare_exchange(false, true, SeqCst, SeqCst) {
            Ok(_) => Ok(()),
            Err(_) => Err(LLamaCppError::BackendAlreadyInitialized),
        }
    }

    /// Initialize the llama backend (without numa).
    ///
    /// # Examples
    ///
    /// ```
    ///# use llama_cpp_4::llama_backend::LlamaBackend;
    ///# use llama_cpp_4::LLamaCppError;
    ///# use std::error::Error;
    ///
    ///# fn main() -> Result<(), Box<dyn Error>> {
    ///
    ///
    /// let backend = LlamaBackend::init()?;
    /// // the llama backend can only be initialized once
    /// assert_eq!(Err(LLamaCppError::BackendAlreadyInitialized), LlamaBackend::init());
    ///
    ///# Ok(())
    ///# }
    /// ```
    #[tracing::instrument(skip_all)]
    pub fn init() -> crate::Result<LlamaBackend> {
        Self::mark_init()?;
        unsafe { llama_cpp_sys_4::llama_backend_init() }
        Ok(LlamaBackend {})
    }

    /// Initialize the llama backend (with numa).
    /// ```
    ///# use llama_cpp_4::llama_backend::LlamaBackend;
    ///# use std::error::Error;
    ///# use llama_cpp_4::llama_backend::NumaStrategy;
    ///
    ///# fn main() -> Result<(), Box<dyn Error>> {
    ///
    /// let llama_backend = LlamaBackend::init_numa(NumaStrategy::MIRROR)?;
    ///
    ///# Ok(())
    ///# }
    /// ```
    #[tracing::instrument(skip_all)]
    pub fn init_numa(strategy: NumaStrategy) -> crate::Result<LlamaBackend> {
        Self::mark_init()?;
        unsafe {
            llama_cpp_sys_4::llama_numa_init(llama_cpp_sys_4::ggml_numa_strategy::from(strategy));
        }
        Ok(LlamaBackend {})
    }

    /// Change the output of llama.cpp's logging to be voided instead of pushed to `stderr`.
    pub fn void_logs(&mut self) {
        unsafe extern "C" fn void_log(
            _level: ggml_log_level,
            _text: *const ::std::os::raw::c_char,
            _user_data: *mut ::std::os::raw::c_void,
        ) {
        }

        unsafe {
            llama_cpp_sys_4::llama_log_set(Some(void_log), std::ptr::null_mut());
        }
    }
}

/// A rusty wrapper around `numa_strategy`.
///
/// ## Description
/// Represents different NUMA (Non-Uniform Memory Access) strategies for memory management
/// in multi-core or multi-processor systems.
///
/// ## See more
/// https://github.com/ggerganov/llama.cpp/blob/master/ggml/include/ggml-cpu.h#L25-L32
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum NumaStrategy {
    /// The NUMA strategy is disabled. No NUMA-aware optimizations are applied.
    /// Memory allocation will not consider NUMA node locality.
    DISABLED,

    /// Distribute memory across NUMA nodes. This strategy aims to balance memory usage
    /// across all available NUMA nodes, potentially improving load balancing and preventing
    /// memory hotspots on a single node. It may use round-robin or another method to
    /// distribute allocations.
    DISTRIBUTE,

    /// Isolate memory to specific NUMA nodes. Memory allocations will be restricted to
    /// specific NUMA nodes, potentially reducing contention and improving locality for
    /// processes or threads bound to a particular node.
    ISOLATE,

    /// Use `numactl` to manage memory and processor affinities. This strategy utilizes
    /// the `numactl` command or library to bind processes or memory allocations to specific
    /// NUMA nodes or CPUs, providing fine-grained control over memory placement.
    NUMACTL,

    /// Mirror memory across NUMA nodes. This strategy creates duplicate memory copies
    /// on multiple NUMA nodes, which can help with fault tolerance and redundancy,
    /// ensuring that each NUMA node has access to a copy of the memory.
    MIRROR,

    /// A placeholder representing the total number of strategies available.
    /// Typically used for iteration or determining the number of strategies in the enum.
    COUNT,
}

/// An invalid numa strategy was provided.
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub struct InvalidNumaStrategy(
    /// The invalid numa strategy that was provided.
    pub llama_cpp_sys_4::ggml_numa_strategy,
);

impl TryFrom<llama_cpp_sys_4::ggml_numa_strategy> for NumaStrategy {
    type Error = InvalidNumaStrategy;

    fn try_from(value: llama_cpp_sys_4::ggml_numa_strategy) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_sys_4::GGML_NUMA_STRATEGY_DISABLED => Ok(Self::DISABLED),
            llama_cpp_sys_4::GGML_NUMA_STRATEGY_DISTRIBUTE => Ok(Self::DISTRIBUTE),
            llama_cpp_sys_4::GGML_NUMA_STRATEGY_ISOLATE => Ok(Self::ISOLATE),
            llama_cpp_sys_4::GGML_NUMA_STRATEGY_NUMACTL => Ok(Self::NUMACTL),
            llama_cpp_sys_4::GGML_NUMA_STRATEGY_MIRROR => Ok(Self::MIRROR),
            llama_cpp_sys_4::GGML_NUMA_STRATEGY_COUNT => Ok(Self::COUNT),
            value => Err(InvalidNumaStrategy(value)),
        }
    }
}

impl From<NumaStrategy> for llama_cpp_sys_4::ggml_numa_strategy {
    fn from(value: NumaStrategy) -> Self {
        match value {
            NumaStrategy::DISABLED => llama_cpp_sys_4::GGML_NUMA_STRATEGY_DISABLED,
            NumaStrategy::DISTRIBUTE => llama_cpp_sys_4::GGML_NUMA_STRATEGY_DISTRIBUTE,
            NumaStrategy::ISOLATE => llama_cpp_sys_4::GGML_NUMA_STRATEGY_ISOLATE,
            NumaStrategy::NUMACTL => llama_cpp_sys_4::GGML_NUMA_STRATEGY_NUMACTL,
            NumaStrategy::MIRROR => llama_cpp_sys_4::GGML_NUMA_STRATEGY_MIRROR,
            NumaStrategy::COUNT => llama_cpp_sys_4::GGML_NUMA_STRATEGY_COUNT,
        }
    }
}

/// Drops the llama backend.
/// ```
///
///# use llama_cpp_4::llama_backend::LlamaBackend;
///# use std::error::Error;
///
///# fn main() -> Result<(), Box<dyn Error>> {
/// let backend = LlamaBackend::init()?;
/// drop(backend);
/// // can be initialized again after being dropped
/// let backend = LlamaBackend::init()?;
///# Ok(())
///# }
///
/// ```
impl Drop for LlamaBackend {
    fn drop(&mut self) {
        match LLAMA_BACKEND_INITIALIZED.compare_exchange(true, false, SeqCst, SeqCst) {
            Ok(_) => {}
            Err(_) => {
                unreachable!("This should not be reachable as the only ways to obtain a llama backend involve marking the backend as initialized.")
            }
        }
        unsafe { llama_cpp_sys_4::llama_backend_free() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numa_from_and_to() {
        let numas = [
            NumaStrategy::DISABLED,
            NumaStrategy::DISTRIBUTE,
            NumaStrategy::ISOLATE,
            NumaStrategy::NUMACTL,
            NumaStrategy::MIRROR,
            NumaStrategy::COUNT,
        ];

        for numa in &numas {
            let from = llama_cpp_sys_4::ggml_numa_strategy::from(*numa);
            let to = NumaStrategy::try_from(from).expect("Failed to convert from and to");
            assert_eq!(*numa, to);
        }
    }

    #[test]
    fn check_invalid_numa() {
        let invalid = 800;
        let invalid = NumaStrategy::try_from(invalid);
        assert_eq!(invalid, Err(InvalidNumaStrategy(invalid.unwrap_err().0)));
    }
}
