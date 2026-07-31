//! Bounded, owned tensor capture and transactional mutation during decode.
//!
//! A [`TensorTransactions`] value is moved into
//! [`LlamaContextParams::with_tensor_transactions`](crate::LlamaContextParams::with_tensor_transactions).
//! The resulting [`crate::LlamaContext`] owns the callback state for its complete
//! native lifetime. Matching tensors are synchronized by llama.cpp, copied into
//! Rust-owned storage, and optionally transformed. Mutable tensors are written
//! back exactly once only after the handler returns successfully and every
//! output value passes validation.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Maximum exact graph selectors attached to one context.
pub const MAX_TENSOR_SELECTORS: usize = 128;
/// Maximum UTF-8 bytes in one exact graph-node name.
pub const MAX_TENSOR_NAME_BYTES: usize = 256;
/// Maximum rows accepted from one selected tensor invocation.
pub const MAX_TENSOR_ROWS: usize = 4_096;
/// Maximum elements copied by one selected tensor invocation.
pub const MAX_TENSOR_ELEMENTS: usize = 16_777_216;
/// Maximum retained tensor bytes from one decode.
pub const MAX_RETAINED_TENSOR_BYTES: usize = MAX_TENSOR_ELEMENTS * size_of::<f32>();
/// Maximum retained callback failure bytes.
pub const MAX_TENSOR_FAILURE_BYTES: usize = 1_024;

/// Element representation required by an exact tensor selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TensorElementType {
    /// IEEE-754 single-precision values.
    F32,
    /// Signed 32-bit integer values.
    I32,
}

impl TensorElementType {
    const fn native(self) -> llama_cpp_sys_4::ggml_type {
        match self {
            Self::F32 => llama_cpp_sys_4::GGML_TYPE_F32,
            Self::I32 => llama_cpp_sys_4::GGML_TYPE_I32,
        }
    }
}

/// Native access granted to one selected tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorAccess {
    /// Copy and optionally retain the tensor without native mutation.
    ReadOnly,
    /// Run the handler over finite `f32` values and commit once on success.
    ReadWriteF32,
}

/// Native write-back decision returned by a transaction handler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorWriteback {
    /// Discard handler-side edits and leave the native tensor unchanged.
    Unchanged,
    /// Validate and commit the complete edited tensor exactly once.
    Commit,
}

/// Mapping between tensor rows and the submitted decode batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorRowMapping {
    /// Rows correspond in order to logical token entries in the decode batch.
    BatchTokens,
}

/// Exact bounded graph-node contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSelector {
    name: String,
    element_type: TensorElementType,
    row_elements: usize,
    maximum_rows: usize,
    access: TensorAccess,
    row_mapping: TensorRowMapping,
    retain: bool,
}

impl TensorSelector {
    /// Constructs an exact tensor selector.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty, NUL-containing, or excessive name,
    /// unusable dimensions, an excessive element bound, or mutable non-`f32`
    /// data.
    pub fn new(
        name: impl Into<String>,
        element_type: TensorElementType,
        row_elements: usize,
        maximum_rows: usize,
        access: TensorAccess,
        row_mapping: TensorRowMapping,
        retain: bool,
    ) -> Result<Self, TensorTransactionError> {
        let selector = Self {
            name: name.into(),
            element_type,
            row_elements,
            maximum_rows,
            access,
            row_mapping,
            retain,
        };
        selector.validate()?;
        Ok(selector)
    }

    /// Constructs one exact residual layer-output selector.
    ///
    /// llama.cpp names these graph nodes `l_out-{layer}` at the pinned
    /// implementation. The caller remains responsible for binding that naming
    /// profile to its backend revision.
    ///
    /// # Errors
    ///
    /// Returns an error for unusable dimensions or bounds.
    pub fn layer_output(
        layer: u32,
        row_elements: usize,
        maximum_rows: usize,
        access: TensorAccess,
        retain: bool,
    ) -> Result<Self, TensorTransactionError> {
        Self::new(
            format!("l_out-{layer}"),
            TensorElementType::F32,
            row_elements,
            maximum_rows,
            access,
            TensorRowMapping::BatchTokens,
            retain,
        )
    }

    /// Returns the exact graph-node name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the required element representation.
    #[must_use]
    pub const fn element_type(&self) -> TensorElementType {
        self.element_type
    }

    /// Returns the required elements per row.
    #[must_use]
    pub const fn row_elements(&self) -> usize {
        self.row_elements
    }

    /// Returns the inclusive row bound.
    #[must_use]
    pub const fn maximum_rows(&self) -> usize {
        self.maximum_rows
    }

    /// Returns native access granted to the handler.
    #[must_use]
    pub const fn access(&self) -> TensorAccess {
        self.access
    }

    /// Returns how native rows map to the submitted decode batch.
    #[must_use]
    pub const fn row_mapping(&self) -> TensorRowMapping {
        self.row_mapping
    }

    /// Returns whether the completed owned tensor is retained after decode.
    #[must_use]
    pub const fn retains_capture(&self) -> bool {
        self.retain
    }

    fn validate(&self) -> Result<(), TensorTransactionError> {
        if self.name.is_empty()
            || self.name.len() > MAX_TENSOR_NAME_BYTES
            || self.name.as_bytes().contains(&0)
        {
            return Err(TensorTransactionError::new(
                "tensor name must be bounded, nonempty UTF-8 without NUL",
            ));
        }
        let elements = self
            .row_elements
            .checked_mul(self.maximum_rows)
            .ok_or_else(|| TensorTransactionError::new("tensor element bound overflowed"))?;
        if self.row_elements == 0
            || self.maximum_rows == 0
            || self.maximum_rows > MAX_TENSOR_ROWS
            || elements > MAX_TENSOR_ELEMENTS
        {
            return Err(TensorTransactionError::new(
                "tensor row shape is outside the supported bound",
            ));
        }
        if self.access == TensorAccess::ReadWriteF32 && self.element_type != TensorElementType::F32
        {
            return Err(TensorTransactionError::new(
                "only f32 tensors support transactional write-back",
            ));
        }
        Ok(())
    }
}

/// Exact sequence and causal-position metadata for one tensor row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorBatchRow {
    /// Zero-based logical batch entry.
    pub batch_index: u32,
    /// Native causal position.
    pub position: i32,
    /// Exact sequence IDs attached to this entry.
    pub sequence_ids: Vec<i32>,
}

/// Validated tensor dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorShape {
    /// Elements in each logical row.
    pub row_elements: usize,
    /// Logical rows in this callback invocation.
    pub rows: usize,
    /// Total elements.
    pub elements: usize,
}

/// Typed Rust-owned tensor storage supplied to a callback.
pub enum TensorDataMut<'a> {
    /// Mutable finite `f32` storage.
    F32(&'a mut [f32]),
    /// Mutable copied `i32` storage. Read-only selectors never commit changes.
    I32(&'a mut [i32]),
}

impl fmt::Debug for TensorDataMut<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32(values) => formatter
                .debug_tuple("F32")
                .field(&format_args!("{} elements", values.len()))
                .finish(),
            Self::I32(values) => formatter
                .debug_tuple("I32")
                .field(&format_args!("{} elements", values.len()))
                .finish(),
        }
    }
}

/// One owned tensor transaction presented synchronously to Rust.
pub struct TensorTransaction<'a> {
    /// Exact graph-node name.
    pub name: &'a str,
    /// Validated shape.
    pub shape: TensorShape,
    /// Exact logical rows represented by this native tensor.
    pub rows: &'a [TensorBatchRow],
    /// Native access granted by the selector.
    pub access: TensorAccess,
    /// Typed Rust-owned values.
    pub data: TensorDataMut<'a>,
}

impl fmt::Debug for TensorTransaction<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TensorTransaction")
            .field("name", &self.name)
            .field("shape", &self.shape)
            .field("rows", &self.rows)
            .field("access", &self.access)
            .field("data", &self.data)
            .finish()
    }
}

/// Synchronous safe handler for selected tensor transactions.
pub trait TensorTransactionHandler: Send {
    /// Applies caller-defined mechanics to one Rust-owned tensor copy.
    ///
    /// For [`TensorAccess::ReadWriteF32`], successful finite output is written
    /// back exactly once after this method returns. Returning an error or
    /// unwinding causes no write-back.
    ///
    /// # Errors
    ///
    /// Returns an implementation-defined bounded failure.
    fn apply(
        &mut self,
        transaction: TensorTransaction<'_>,
    ) -> Result<TensorWriteback, TensorTransactionError>;
}

/// Error returned by a tensor transaction handler or selector validator.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct TensorTransactionError {
    message: String,
}

impl TensorTransactionError {
    /// Creates a bounded transaction error.
    pub fn new(message: impl Into<String>) -> Self {
        let mut message = message.into();
        if message.len() > MAX_TENSOR_FAILURE_BYTES {
            message.truncate(MAX_TENSOR_FAILURE_BYTES);
        }
        Self { message }
    }

    /// Returns the bounded failure message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Contained callback failure returned by [`crate::LlamaContext::decode`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("tensor callback failed{tensor_suffix}: {message}")]
pub struct TensorCallbackFailure {
    tensor: Option<String>,
    tensor_suffix: String,
    panicked: bool,
    message: String,
}

impl TensorCallbackFailure {
    fn new(tensor: Option<&str>, panicked: bool, message: impl Into<String>) -> Self {
        let mut message = message.into();
        if message.len() > MAX_TENSOR_FAILURE_BYTES {
            message.truncate(MAX_TENSOR_FAILURE_BYTES);
        }
        let tensor = tensor.map(ToOwned::to_owned);
        let tensor_suffix = tensor
            .as_deref()
            .map_or_else(String::new, |name| format!(" for {name}"));
        Self {
            tensor,
            tensor_suffix,
            panicked,
            message,
        }
    }

    /// Returns the exact tensor name when failure happened after selection.
    #[must_use]
    pub fn tensor(&self) -> Option<&str> {
        self.tensor.as_deref()
    }

    /// Returns whether Rust unwinding was contained.
    #[must_use]
    pub const fn panicked(&self) -> bool {
        self.panicked
    }

    /// Returns the bounded failure message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Complete retained typed tensor from one callback invocation.
#[derive(Clone, Debug, PartialEq)]
pub struct TransactionalTensorCapture {
    /// Exact graph-node name.
    pub name: String,
    /// Validated shape.
    pub shape: TensorShape,
    /// Exact logical batch rows.
    pub rows: Vec<TensorBatchRow>,
    /// Typed copied values after a successful handler invocation.
    pub data: CapturedTensorData,
}

/// Typed retained tensor storage.
#[derive(Clone, Debug, PartialEq)]
pub enum CapturedTensorData {
    /// Finite `f32` values.
    F32(Vec<f32>),
    /// Signed integer values.
    I32(Vec<i32>),
}

impl CapturedTensorData {
    /// Returns the retained element count.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::I32(values) => values.len(),
        }
    }

    /// Returns whether no elements are retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Owned, bounded callback program attached to one context.
pub struct TensorTransactions {
    selectors: Vec<TensorSelector>,
    handler: Option<Box<dyn TensorTransactionHandler>>,
    captures: Vec<TransactionalTensorCapture>,
    retained_bytes: usize,
    pending_captures: Vec<TransactionalTensorCapture>,
    pending_retained_bytes: usize,
    batch_rows: Vec<TensorBatchRow>,
    rows_seen: BTreeMap<String, usize>,
    failure: Option<TensorCallbackFailure>,
    decode_active: bool,
}

impl fmt::Debug for TensorTransactions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TensorTransactions")
            .field("selectors", &self.selectors)
            .field("has_handler", &self.handler.is_some())
            .field("captures", &self.captures.len())
            .field("retained_bytes", &self.retained_bytes)
            .field("pending_captures", &self.pending_captures.len())
            .field("pending_retained_bytes", &self.pending_retained_bytes)
            .field("failure", &self.failure)
            .field("decode_active", &self.decode_active)
            .finish_non_exhaustive()
    }
}

impl TensorTransactions {
    /// Constructs a read-only capture program.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, excessive, duplicate, unordered, mutable,
    /// or collectively over-bound selectors.
    pub fn capture(selectors: Vec<TensorSelector>) -> Result<Self, TensorTransactionError> {
        Self::build(selectors, None)
    }

    /// Constructs a capture/mutation program with one synchronous handler.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, excessive, duplicate, unordered, or
    /// collectively over-bound selectors.
    pub fn new(
        selectors: Vec<TensorSelector>,
        handler: impl TensorTransactionHandler + 'static,
    ) -> Result<Self, TensorTransactionError> {
        Self::build(selectors, Some(Box::new(handler)))
    }

    fn build(
        selectors: Vec<TensorSelector>,
        handler: Option<Box<dyn TensorTransactionHandler>>,
    ) -> Result<Self, TensorTransactionError> {
        if selectors.is_empty() || selectors.len() > MAX_TENSOR_SELECTORS {
            return Err(TensorTransactionError::new(
                "selector count is outside the supported bound",
            ));
        }
        let mut total_elements = 0_usize;
        let mut prior_name: Option<&str> = None;
        let mut needs_handler = false;
        for selector in &selectors {
            selector.validate()?;
            if prior_name.is_some_and(|prior| prior >= selector.name()) {
                return Err(TensorTransactionError::new(
                    "selectors must have unique canonically ordered names",
                ));
            }
            prior_name = Some(selector.name());
            needs_handler |= selector.access == TensorAccess::ReadWriteF32;
            total_elements = total_elements
                .checked_add(
                    selector
                        .row_elements
                        .checked_mul(selector.maximum_rows)
                        .ok_or_else(|| {
                            TensorTransactionError::new("selector element bound overflowed")
                        })?,
                )
                .ok_or_else(|| {
                    TensorTransactionError::new("total selector element bound overflowed")
                })?;
        }
        if total_elements > MAX_TENSOR_ELEMENTS {
            return Err(TensorTransactionError::new(
                "total selector element bound is excessive",
            ));
        }
        if needs_handler && handler.is_none() {
            return Err(TensorTransactionError::new(
                "mutable selectors require a transaction handler",
            ));
        }
        if !needs_handler && handler.is_some() {
            return Err(TensorTransactionError::new(
                "a transaction handler requires at least one mutable selector",
            ));
        }
        Ok(Self {
            selectors,
            handler,
            captures: Vec::new(),
            retained_bytes: 0,
            pending_captures: Vec::new(),
            pending_retained_bytes: 0,
            batch_rows: Vec::new(),
            rows_seen: BTreeMap::new(),
            failure: None,
            decode_active: false,
        })
    }

    /// Returns exact selector contracts.
    #[must_use]
    pub fn selectors(&self) -> &[TensorSelector] {
        &self.selectors
    }

    /// Returns successful retained tensors accumulated since the last drain.
    ///
    /// A native speculative operation may perform several internal decodes.
    /// Each decode commits its retained tensors only after its lifecycle and
    /// selector coverage complete successfully.
    #[must_use]
    pub fn captures(&self) -> &[TransactionalTensorCapture] {
        &self.captures
    }

    /// Removes retained tensors accumulated since the previous drain.
    pub fn take_captures(&mut self) -> Vec<TransactionalTensorCapture> {
        self.retained_bytes = 0;
        std::mem::take(&mut self.captures)
    }

    /// Returns the contained failure from the most recent decode.
    #[must_use]
    pub const fn failure(&self) -> Option<&TensorCallbackFailure> {
        self.failure.as_ref()
    }

    fn begin_decode_raw(
        &mut self,
        batch: &llama_cpp_sys_4::llama_batch,
    ) -> Result<(), TensorCallbackFailure> {
        if let Some(failure) = self.failure.clone() {
            return Err(failure);
        }
        if self.decode_active {
            return Err(TensorCallbackFailure::new(
                None,
                false,
                "tensor callback decode was already active",
            ));
        }
        self.pending_captures.clear();
        self.pending_retained_bytes = 0;
        self.rows_seen.clear();
        self.batch_rows = copy_batch_rows(batch)?;
        self.decode_active = true;
        Ok(())
    }

    pub(crate) fn finish_decode(
        &mut self,
        native_succeeded: bool,
    ) -> Result<(), TensorCallbackFailure> {
        self.decode_active = false;
        let expected_rows = self.batch_rows.len();
        self.batch_rows.clear();
        if let Some(failure) = self.failure.clone() {
            self.pending_captures.clear();
            self.pending_retained_bytes = 0;
            return Err(failure);
        }
        if !native_succeeded {
            self.pending_captures.clear();
            self.pending_retained_bytes = 0;
            return Ok(());
        }
        for selector in &self.selectors {
            let rows = self.rows_seen.get(selector.name()).copied().unwrap_or(0);
            let complete = match selector.row_mapping {
                TensorRowMapping::BatchTokens => rows == expected_rows,
            };
            if !complete {
                let failure = TensorCallbackFailure::new(
                    Some(selector.name()),
                    false,
                    format!(
                        "selected tensor covered {rows} rows but the decode submitted \
                         {expected_rows}"
                    ),
                );
                self.failure = Some(failure.clone());
                self.pending_captures.clear();
                self.pending_retained_bytes = 0;
                return Err(failure);
            }
        }
        self.retained_bytes = self
            .retained_bytes
            .checked_add(self.pending_retained_bytes)
            .ok_or_else(|| {
                TensorCallbackFailure::new(None, false, "committed retained byte count overflowed")
            })?;
        self.captures.append(&mut self.pending_captures);
        self.pending_retained_bytes = 0;
        Ok(())
    }

    fn selected(&self, name: &str) -> Option<usize> {
        self.selectors
            .binary_search_by(|selector| selector.name().cmp(name))
            .ok()
    }

    fn process(
        &mut self,
        tensor: *mut llama_cpp_sys_4::ggml_tensor,
        selector_index: usize,
    ) -> Result<(), TensorTransactionError> {
        let selector = self.selectors[selector_index].clone();
        let shape = validate_tensor(tensor, &selector)?;
        let start = match selector.row_mapping {
            TensorRowMapping::BatchTokens => {
                self.rows_seen.get(selector.name()).copied().unwrap_or(0)
            }
        };
        let end = start
            .checked_add(shape.rows)
            .ok_or_else(|| TensorTransactionError::new("tensor row mapping overflowed"))?;
        if end > self.batch_rows.len() {
            return Err(TensorTransactionError::new(
                "tensor rows exceed submitted decode batch",
            ));
        }
        let rows = self.batch_rows[start..end].to_vec();

        match selector.element_type {
            TensorElementType::F32 => {
                let mut values = vec![0.0_f32; shape.elements];
                copy_tensor_get(tensor, &mut values)?;
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(TensorTransactionError::new(
                        "selected f32 tensor contains a non-finite value",
                    ));
                }
                if selector.access == TensorAccess::ReadWriteF32 {
                    let original = selector.retain.then(|| values.clone());
                    let handler = self.handler.as_deref_mut().ok_or_else(|| {
                        TensorTransactionError::new("mutable tensor handler is unavailable")
                    })?;
                    let writeback = handler.apply(TensorTransaction {
                        name: selector.name(),
                        shape,
                        rows: &rows,
                        access: selector.access,
                        data: TensorDataMut::F32(&mut values),
                    })?;
                    match writeback {
                        TensorWriteback::Unchanged => {
                            if let Some(original) = original {
                                values = original;
                            }
                        }
                        TensorWriteback::Commit => {
                            if values.iter().any(|value| !value.is_finite()) {
                                return Err(TensorTransactionError::new(
                                    "transaction produced a non-finite f32 value",
                                ));
                            }
                            copy_tensor_set(tensor, &values)?;
                        }
                    }
                }
                if selector.retain {
                    self.retain(
                        selector.name.clone(),
                        shape,
                        rows,
                        CapturedTensorData::F32(values),
                    )?;
                }
            }
            TensorElementType::I32 => {
                let mut values = vec![0_i32; shape.elements];
                copy_tensor_get(tensor, &mut values)?;
                if selector.retain {
                    self.retain(
                        selector.name.clone(),
                        shape,
                        rows,
                        CapturedTensorData::I32(values),
                    )?;
                }
            }
        }
        self.rows_seen.insert(selector.name, end);
        Ok(())
    }

    fn retain(
        &mut self,
        name: String,
        shape: TensorShape,
        rows: Vec<TensorBatchRow>,
        data: CapturedTensorData,
    ) -> Result<(), TensorTransactionError> {
        let bytes = data
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| TensorTransactionError::new("retained byte count overflowed"))?;
        self.pending_retained_bytes = self
            .pending_retained_bytes
            .checked_add(bytes)
            .ok_or_else(|| TensorTransactionError::new("retained byte count overflowed"))?;
        let total_retained_bytes = self
            .retained_bytes
            .checked_add(self.pending_retained_bytes)
            .ok_or_else(|| TensorTransactionError::new("retained byte count overflowed"))?;
        if total_retained_bytes > MAX_RETAINED_TENSOR_BYTES {
            return Err(TensorTransactionError::new(
                "retained tensor bytes exceed the supported bound",
            ));
        }
        self.pending_captures.push(TransactionalTensorCapture {
            name,
            shape,
            rows,
            data,
        });
        Ok(())
    }

    fn record_failure(&mut self, tensor: Option<&str>, panicked: bool, message: impl Into<String>) {
        if self.failure.is_none() {
            self.failure = Some(TensorCallbackFailure::new(tensor, panicked, message));
        }
    }
}

fn validate_tensor(
    tensor: *mut llama_cpp_sys_4::ggml_tensor,
    selector: &TensorSelector,
) -> Result<TensorShape, TensorTransactionError> {
    if tensor.is_null() {
        return Err(TensorTransactionError::new(
            "native tensor pointer was null",
        ));
    }
    // SAFETY: llama.cpp supplies a live graph tensor for the synchronous
    // callback. No reference escapes this function.
    let tensor_ref = unsafe { &*tensor };
    if tensor_ref.type_ != selector.element_type.native() {
        return Err(TensorTransactionError::new(
            "native tensor element type does not match selector",
        ));
    }
    if tensor_ref.ne[2] != 1 || tensor_ref.ne[3] != 1 {
        return Err(TensorTransactionError::new(
            "selected tensor must be a two-dimensional row matrix",
        ));
    }
    let row_elements = usize::try_from(tensor_ref.ne[0])
        .map_err(|_| TensorTransactionError::new("native row width is negative or excessive"))?;
    let rows = usize::try_from(tensor_ref.ne[1])
        .map_err(|_| TensorTransactionError::new("native row count is negative or excessive"))?;
    let elements = row_elements
        .checked_mul(rows)
        .ok_or_else(|| TensorTransactionError::new("native tensor element count overflowed"))?;
    if row_elements != selector.row_elements
        || rows == 0
        || rows > selector.maximum_rows
        || elements > MAX_TENSOR_ELEMENTS
    {
        return Err(TensorTransactionError::new(
            "native tensor shape does not match selector",
        ));
    }
    // SAFETY: the pointer is live for this callback.
    if !unsafe { llama_cpp_sys_4::ggml_is_contiguous(tensor) } {
        return Err(TensorTransactionError::new(
            "selected tensor is not contiguous",
        ));
    }
    let expected_bytes = elements
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| TensorTransactionError::new("native tensor byte count overflowed"))?;
    // SAFETY: the pointer is live for this callback.
    if unsafe { llama_cpp_sys_4::ggml_nbytes(tensor) } != expected_bytes {
        return Err(TensorTransactionError::new(
            "native tensor byte size does not match selector",
        ));
    }
    Ok(TensorShape {
        row_elements,
        rows,
        elements,
    })
}

fn copy_tensor_get<T>(
    tensor: *mut llama_cpp_sys_4::ggml_tensor,
    values: &mut [T],
) -> Result<(), TensorTransactionError> {
    let bytes = size_of_val(values);
    if bytes == 0 {
        return Err(TensorTransactionError::new(
            "cannot copy an empty native tensor",
        ));
    }
    // SAFETY: `validate_tensor` proves the selected native tensor has exactly
    // this contiguous byte size. `values` is live and exclusively borrowed.
    unsafe {
        llama_cpp_sys_4::ggml_backend_tensor_get(
            tensor,
            values.as_mut_ptr().cast::<c_void>(),
            0,
            bytes,
        );
    }
    Ok(())
}

fn copy_tensor_set<T>(
    tensor: *mut llama_cpp_sys_4::ggml_tensor,
    values: &[T],
) -> Result<(), TensorTransactionError> {
    let bytes = size_of_val(values);
    if bytes == 0 {
        return Err(TensorTransactionError::new(
            "cannot write an empty native tensor",
        ));
    }
    // SAFETY: `validate_tensor` proves the selected native tensor has exactly
    // this contiguous byte size. llama.cpp synchronized this node before the
    // callback and later dependent nodes have not executed.
    unsafe {
        llama_cpp_sys_4::ggml_backend_tensor_set(
            tensor,
            values.as_ptr().cast::<c_void>(),
            0,
            bytes,
        );
    }
    Ok(())
}

fn copy_batch_rows(
    batch: &llama_cpp_sys_4::llama_batch,
) -> Result<Vec<TensorBatchRow>, TensorCallbackFailure> {
    let count = usize::try_from(batch.n_tokens).map_err(|_| {
        TensorCallbackFailure::new(None, false, "decode batch token count is negative")
    })?;
    if count == 0 || count > MAX_TENSOR_ROWS {
        return Err(TensorCallbackFailure::new(
            None,
            false,
            "decode batch token count is outside the callback bound",
        ));
    }
    if batch.pos.is_null() || batch.n_seq_id.is_null() || batch.seq_id.is_null() {
        return Err(TensorCallbackFailure::new(
            None,
            false,
            "decode batch metadata pointers are null",
        ));
    }
    let mut rows = Vec::with_capacity(count);
    for index in 0..count {
        // SAFETY: the native `llama_batch` contract provides arrays allocated
        // for at least `n_tokens` entries, and the begin hook synchronously
        // borrows the batch for this copy.
        let position = unsafe { *batch.pos.add(index) };
        // SAFETY: same allocation contract as `position`.
        let sequence_count = unsafe { *batch.n_seq_id.add(index) };
        let sequence_count = usize::try_from(sequence_count).map_err(|_| {
            TensorCallbackFailure::new(None, false, "decode batch sequence count is negative")
        })?;
        if sequence_count == 0 || sequence_count > MAX_TENSOR_ROWS {
            return Err(TensorCallbackFailure::new(
                None,
                false,
                "decode batch sequence count is outside the callback bound",
            ));
        }
        // SAFETY: the native `llama_batch` contract provides a sequence array
        // with exactly `n_seq_id[index]` entries.
        let sequence_ptr = unsafe { *batch.seq_id.add(index) };
        if sequence_ptr.is_null() {
            return Err(TensorCallbackFailure::new(
                None,
                false,
                "decode batch sequence pointer is null",
            ));
        }
        // SAFETY: validated count and live batch allocation above.
        let sequence_ids =
            unsafe { std::slice::from_raw_parts(sequence_ptr, sequence_count) }.to_vec();
        rows.push(TensorBatchRow {
            batch_index: u32::try_from(index)
                .map_err(|_| TensorCallbackFailure::new(None, false, "batch index exceeds u32"))?,
            position,
            sequence_ids,
        });
    }
    Ok(rows)
}

pub(crate) unsafe extern "C" fn tensor_transaction_decode_begin(
    batch: *const llama_cpp_sys_4::llama_batch,
    user_data: *mut c_void,
) -> bool {
    if batch.is_null() || user_data.is_null() {
        return false;
    }
    // SAFETY: context parameters retain the pinned transaction owner for every
    // native decode and llama.cpp supplies a live batch for this call.
    let state = unsafe { &mut *user_data.cast::<TensorTransactions>() };
    // SAFETY: null was rejected and the batch remains live synchronously.
    let batch = unsafe { &*batch };
    let result = catch_unwind(AssertUnwindSafe(|| state.begin_decode_raw(batch)));
    match result {
        Ok(Ok(())) => true,
        Ok(Err(error)) => {
            state.record_failure(None, false, error.to_string());
            false
        }
        Err(_) => {
            state.record_failure(None, true, "tensor decode-begin callback panicked");
            false
        }
    }
}

pub(crate) unsafe extern "C" fn tensor_transaction_decode_end(
    native_succeeded: bool,
    user_data: *mut c_void,
) -> bool {
    if user_data.is_null() {
        return false;
    }
    // SAFETY: context parameters retain the pinned transaction owner for every
    // native decode.
    let state = unsafe { &mut *user_data.cast::<TensorTransactions>() };
    let result = catch_unwind(AssertUnwindSafe(|| state.finish_decode(native_succeeded)));
    match result {
        Ok(Ok(())) => true,
        Ok(Err(error)) => {
            state.record_failure(error.tensor(), error.panicked(), error.message());
            false
        }
        Err(_) => {
            state.record_failure(None, true, "tensor decode-end callback panicked");
            false
        }
    }
}

pub(crate) unsafe extern "C" fn tensor_transaction_callback(
    tensor: *mut llama_cpp_sys_4::ggml_tensor,
    ask: bool,
    user_data: *mut c_void,
) -> bool {
    if tensor.is_null() || user_data.is_null() {
        return false;
    }
    // SAFETY: `LlamaContextParams::with_tensor_transactions` installs a pointer
    // to pinned state owned by the context for the complete native lifetime.
    let state = unsafe { &mut *user_data.cast::<TensorTransactions>() };
    if !state.decode_active {
        state.record_failure(
            None,
            false,
            "tensor evaluation callback ran outside a decode lifecycle",
        );
        return false;
    }
    if state.failure.is_some() {
        // Do not request more callbacks, allowing the scheduler to finish the
        // remaining graph without another Rust boundary.
        return false;
    }
    // SAFETY: graph tensor names are fixed NUL-terminated arrays.
    let name_bytes = unsafe { &(*tensor).name };
    let length = name_bytes
        .iter()
        .position(|value| *value == 0)
        .unwrap_or(name_bytes.len());
    let raw_name =
        // SAFETY: `c_char` and `u8` have identical byte width.
        unsafe { std::slice::from_raw_parts(name_bytes.as_ptr().cast::<u8>(), length) };
    let name = match std::str::from_utf8(raw_name) {
        Ok(name) => name,
        Err(error) => {
            state.record_failure(None, false, format!("tensor name is not UTF-8: {error}"));
            return false;
        }
    };
    let Some(selector_index) = state.selected(name) else {
        return false;
    };
    if ask {
        return true;
    }

    let result = catch_unwind(AssertUnwindSafe(|| state.process(tensor, selector_index)));
    match result {
        Ok(Ok(())) => true,
        Ok(Err(error)) => {
            state.record_failure(Some(name), false, error.to_string());
            true
        }
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map_or_else(
                    || {
                        payload
                            .downcast_ref::<String>()
                            .map_or("tensor handler panicked", String::as_str)
                    },
                    |message| *message,
                )
                .to_owned();
            state.record_failure(Some(name), true, message);
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AddOne;

    impl TensorTransactionHandler for AddOne {
        fn apply(
            &mut self,
            mut transaction: TensorTransaction<'_>,
        ) -> Result<TensorWriteback, TensorTransactionError> {
            let TensorDataMut::F32(values) = &mut transaction.data else {
                return Err(TensorTransactionError::new("expected f32"));
            };
            for value in values.iter_mut() {
                *value += 1.0;
            }
            Ok(TensorWriteback::Commit)
        }
    }

    #[test]
    fn selectors_are_bounded_and_canonical() {
        let selector = TensorSelector::layer_output(1, 4, 2, TensorAccess::ReadOnly, true).unwrap();
        assert_eq!(selector.name(), "l_out-1");
        assert!(TensorSelector::new(
            "bad\0name",
            TensorElementType::F32,
            4,
            2,
            TensorAccess::ReadOnly,
            TensorRowMapping::BatchTokens,
            true,
        )
        .is_err());
        assert!(TensorSelector::new(
            "integer",
            TensorElementType::I32,
            4,
            2,
            TensorAccess::ReadWriteF32,
            TensorRowMapping::BatchTokens,
            true,
        )
        .is_err());
    }

    #[test]
    fn transaction_sets_require_a_handler_and_ordered_names() {
        let mutable =
            TensorSelector::layer_output(1, 4, 2, TensorAccess::ReadWriteF32, false).unwrap();
        assert!(TensorTransactions::capture(vec![mutable.clone()]).is_err());
        assert!(TensorTransactions::new(vec![mutable], AddOne).is_ok());

        let later = TensorSelector::layer_output(2, 4, 2, TensorAccess::ReadOnly, true).unwrap();
        let earlier = TensorSelector::layer_output(1, 4, 2, TensorAccess::ReadOnly, true).unwrap();
        assert!(TensorTransactions::capture(vec![later, earlier]).is_err());
    }

    #[test]
    fn errors_and_failure_messages_are_bounded() {
        let error = TensorTransactionError::new("x".repeat(MAX_TENSOR_FAILURE_BYTES + 10));
        assert_eq!(error.message().len(), MAX_TENSOR_FAILURE_BYTES);
        let failure = TensorCallbackFailure::new(
            Some("l_out-1"),
            true,
            "y".repeat(MAX_TENSOR_FAILURE_BYTES + 10),
        );
        assert!(failure.panicked());
        assert_eq!(failure.message().len(), MAX_TENSOR_FAILURE_BYTES);
        assert_eq!(failure.tensor(), Some("l_out-1"));
    }

    #[test]
    fn successful_internal_decodes_accumulate_and_failed_staging_is_discarded() {
        let selector = TensorSelector::layer_output(1, 1, 1, TensorAccess::ReadOnly, true).unwrap();
        let mut transactions = TensorTransactions::capture(vec![selector]).unwrap();

        let stage = |transactions: &mut TensorTransactions, value: f32, succeeded: bool| {
            transactions.decode_active = true;
            transactions.batch_rows = vec![TensorBatchRow {
                batch_index: 0,
                position: 0,
                sequence_ids: vec![0],
            }];
            transactions.rows_seen.insert("l_out-1".to_owned(), 1);
            transactions
                .retain(
                    "l_out-1".to_owned(),
                    TensorShape {
                        row_elements: 1,
                        rows: 1,
                        elements: 1,
                    },
                    transactions.batch_rows.clone(),
                    CapturedTensorData::F32(vec![value]),
                )
                .unwrap();
            transactions.finish_decode(succeeded).unwrap();
        };

        stage(&mut transactions, 1.0, true);
        stage(&mut transactions, 2.0, true);
        stage(&mut transactions, 3.0, false);
        let captures = transactions.take_captures();
        assert_eq!(captures.len(), 2);
        assert!(matches!(
            captures[0].data,
            CapturedTensorData::F32(ref values) if values == &[1.0]
        ));
        assert!(matches!(
            captures[1].data,
            CapturedTensorData::F32(ref values) if values == &[2.0]
        ));
        assert!(transactions.captures().is_empty());
    }
}
