//! A safe wrapper around `llama_model`.
use std::ffi::CStr;
use std::ffi::CString;
use std::num::NonZeroU16;
use std::os::raw::c_int;
use std::path::Path;
use std::ptr::NonNull;

use crate::context::params::LlamaContextParams;
use crate::context::LlamaContext;
use crate::llama_backend::LlamaBackend;
use crate::model::params::LlamaModelParams;
use crate::token::from_vec_token_sys;
use crate::token::LlamaToken;
use crate::token_type::{LlamaTokenAttr, LlamaTokenAttrs};
use crate::{
    ApplyChatTemplateError, ChatTemplateError, LlamaContextLoadError, LlamaLoraAdapterInitError,
    LlamaModelLoadError, NewLlamaChatMessageError, StringToTokenError, TokenToStringError,
};

pub mod params;

/// A safe wrapper around `llama_model`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaModel {
    pub(crate) model: NonNull<llama_cpp_sys_4::llama_model>,
}

/// A safe wrapper around `llama_lora_adapter`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaLoraAdapter {
    pub(crate) lora_adapter: NonNull<llama_cpp_sys_4::llama_lora_adapter>,
}

/// A Safe wrapper around `llama_chat_message`
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct LlamaChatMessage {
    role: CString,
    content: CString,
}

impl LlamaChatMessage {
    /// Create a new `LlamaChatMessage`
    pub fn new(role: String, content: String) -> Result<Self, NewLlamaChatMessageError> {
        Ok(Self {
            role: CString::new(role)?,
            content: CString::new(content)?,
        })
    }
}

/// How to determine if we should prepend a bos token to tokens
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddBos {
    /// Add the beginning of stream token to the start of the string.
    Always,
    /// Do not add the beginning of stream token to the start of the string.
    Never,
}

/// How to determine if we should tokenize special tokens
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Special {
    /// Allow tokenizing special and/or control tokens which otherwise are not exposed and treated as plaintext. Does not insert a leading space.
    Tokenize,
    /// Treat special and/or control tokens as plaintext.
    Plaintext,
}

unsafe impl Send for LlamaModel {}

unsafe impl Sync for LlamaModel {}

impl LlamaModel {
    /// Get the number of tokens the model was trained on.
    ///
    /// This function returns the number of tokens that the model was trained on, represented as a `u32`.
    ///
    /// # Panics
    ///
    /// This function will panic if the number of tokens the model was trained on does not fit into a `u32`.
    /// This should be impossible on most platforms since llama.cpp returns a `c_int` (i32 on most platforms),
    /// which is almost certainly positive.
    #[must_use]
    pub fn n_ctx_train(&self) -> u32 {
        let n_ctx_train = unsafe { llama_cpp_sys_4::llama_n_ctx_train(self.model.as_ptr()) };
        u32::try_from(n_ctx_train).expect("n_ctx_train fits into an u32")
    }

    /// Get all tokens in the model.
    ///
    /// This function returns an iterator over all the tokens in the model. Each item in the iterator is a tuple
    /// containing a `LlamaToken` and its corresponding string representation (or an error if the conversion fails).
    ///
    /// # Parameters
    ///
    /// - `special`: The `Special` value that determines how special tokens (like BOS, EOS, etc.) are handled.
    pub fn tokens(
        &self,
        special: Special,
    ) -> impl Iterator<Item = (LlamaToken, Result<String, TokenToStringError>)> + '_ {
        (0..self.n_vocab())
            .map(LlamaToken::new)
            .map(move |llama_token| (llama_token, self.token_to_str(llama_token, special)))
    }

    /// Get the beginning of stream token.
    ///
    /// This function returns the token that represents the beginning of a stream (BOS token).
    #[must_use]
    pub fn token_bos(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_4::llama_token_bos(self.model.as_ptr()) };
        LlamaToken(token)
    }

    /// Get the end of stream token.
    ///
    /// This function returns the token that represents the end of a stream (EOS token).
    #[must_use]
    pub fn token_eos(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_4::llama_token_eos(self.model.as_ptr()) };
        LlamaToken(token)
    }

    /// Get the newline token.
    ///
    /// This function returns the token that represents a newline character.
    #[must_use]
    pub fn token_nl(&self) -> LlamaToken {
        let token = unsafe { llama_cpp_sys_4::llama_token_nl(self.model.as_ptr()) };
        LlamaToken(token)
    }

    /// Check if a token represents the end of generation (end of turn, end of sequence, etc.).
    ///
    /// This function returns `true` if the provided token signifies the end of generation or end of sequence,
    /// such as EOS or other special tokens.
    ///
    /// # Parameters
    ///
    /// - `token`: The `LlamaToken` to check.
    ///
    /// # Returns
    ///
    /// - `true` if the token is an end-of-generation token, otherwise `false`.
    #[must_use]
    pub fn is_eog_token(&self, token: LlamaToken) -> bool {
        unsafe { llama_cpp_sys_4::llama_token_is_eog(self.model.as_ptr(), token.0) }
    }

    /// Get the decoder start token.
    ///
    /// This function returns the token used to signal the start of decoding (i.e., the token used at the start
    /// of a sequence generation).
    #[must_use]
    pub fn decode_start_token(&self) -> LlamaToken {
        let token =
            unsafe { llama_cpp_sys_4::llama_model_decoder_start_token(self.model.as_ptr()) };
        LlamaToken(token)
    }

    /// Convert a single token to a string.
    ///
    /// This function converts a `LlamaToken` into its string representation.
    ///
    /// # Errors
    ///
    /// This function returns an error if the token cannot be converted to a string. For more details, refer to
    /// [`TokenToStringError`].
    ///
    /// # Parameters
    ///
    /// - `token`: The `LlamaToken` to convert.
    /// - `special`: The `Special` value used to handle special tokens.
    pub fn token_to_str(
        &self,
        token: LlamaToken,
        special: Special,
    ) -> Result<String, TokenToStringError> {
        self.token_to_str_with_size(token, 32, special)
    }

    /// Convert a single token to bytes.
    ///
    /// This function converts a `LlamaToken` into a byte representation.
    ///
    /// # Errors
    ///
    /// This function returns an error if the token cannot be converted to bytes. For more details, refer to
    /// [`TokenToStringError`].
    ///
    /// # Parameters
    ///
    /// - `token`: The `LlamaToken` to convert.
    /// - `special`: The `Special` value used to handle special tokens.
    pub fn token_to_bytes(
        &self,
        token: LlamaToken,
        special: Special,
    ) -> Result<Vec<u8>, TokenToStringError> {
        self.token_to_bytes_with_size(token, 32, special, None)
    }

    /// Convert a vector of tokens to a single string.
    ///
    /// This function takes a slice of `LlamaToken`s and converts them into a single string, concatenating their
    /// string representations.
    ///
    /// # Errors
    ///
    /// This function returns an error if any token cannot be converted to a string. For more details, refer to
    /// [`TokenToStringError`].
    ///
    /// # Parameters
    ///
    /// - `tokens`: A slice of `LlamaToken`s to convert.
    /// - `special`: The `Special` value used to handle special tokens.
    pub fn tokens_to_str(
        &self,
        tokens: &[LlamaToken],
        special: Special,
    ) -> Result<String, TokenToStringError> {
        let mut builder = String::with_capacity(tokens.len() * 4);
        for str in tokens
            .iter()
            .copied()
            .map(|t| self.token_to_str(t, special))
        {
            builder += &str?;
        }
        Ok(builder)
    }

    /// Convert a string to a vector of tokens.
    ///
    /// This function converts a string into a vector of `LlamaToken`s. The function will tokenize the string
    /// and return the corresponding tokens.
    ///
    /// # Errors
    ///
    /// - This function will return an error if the input string contains a null byte.
    ///
    /// # Panics
    ///
    /// - This function will panic if the number of tokens exceeds `usize::MAX`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_4::model::LlamaModel;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use std::path::Path;
    /// use llama_cpp_4::model::AddBos;
    /// let backend = llama_cpp_4::llama_backend::LlamaBackend::init()?;
    /// let model = LlamaModel::load_from_file(&backend, Path::new("path/to/model"), &Default::default())?;
    /// let tokens = model.str_to_token("Hello, World!", AddBos::Always)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn str_to_token(
        &self,
        str: &str,
        add_bos: AddBos,
    ) -> Result<Vec<LlamaToken>, StringToTokenError> {
        let add_bos = match add_bos {
            AddBos::Always => true,
            AddBos::Never => false,
        };

        let tokens_estimation = std::cmp::max(8, (str.len() / 2) + usize::from(add_bos));
        let mut buffer = Vec::with_capacity(tokens_estimation);

        let c_string = CString::new(str)?;
        let buffer_capacity =
            c_int::try_from(buffer.capacity()).expect("buffer capacity should fit into a c_int");

        let size = unsafe {
            llama_cpp_sys_4::llama_tokenize(
                self.model.as_ptr(),
                c_string.as_ptr(),
                c_int::try_from(c_string.as_bytes().len())?,
                buffer.as_mut_ptr(),
                buffer_capacity,
                add_bos,
                true,
            )
        };

        // if we fail the first time we can resize the vector to the correct size and try again. This should never fail.
        // as a result - size is guaranteed to be positive here.
        let size = if size.is_negative() {
            buffer.reserve_exact(usize::try_from(-size).expect("usize's are larger "));
            unsafe {
                llama_cpp_sys_4::llama_tokenize(
                    self.model.as_ptr(),
                    c_string.as_ptr(),
                    c_int::try_from(c_string.as_bytes().len())?,
                    buffer.as_mut_ptr(),
                    -size,
                    add_bos,
                    true,
                )
            }
        } else {
            size
        };

        let size = usize::try_from(size).expect("size is positive and usize ");

        // Safety: `size` < `capacity` and llama-cpp has initialized elements up to `size`
        unsafe { buffer.set_len(size) }

        // Convert to `LlamaToken` without memory copy
        let tokens = from_vec_token_sys(buffer);
        Ok(tokens)
    }

    /// Convert a string to a vector of tokens (fast version).
    ///
    /// This function quickly converts a string into a vector of `LlamaToken`s. The `fast` version of this function
    /// assumes that the input string is well-formed and tries to tokenize it as efficiently as possible, with minimal
    /// error checking beyond essential boundary conditions.
    ///
    /// # Errors
    ///
    /// - This function will return an error if the input `str` contains a null byte (i.e., an invalid UTF-8 sequence).
    ///
    /// # Panics
    ///
    /// - This function will panic if the number of `LlamaToken`s generated from the input string exceeds `usize::MAX`.
    ///
    /// # Parameters
    ///
    /// - `str`: The input string to convert to tokens.
    /// - `add_bos`: Determines whether to add the beginning-of-sequence (BOS) token. Use `AddBos::Always` to always add it,
    ///   or `AddBos::Never` to omit it.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_4::model::LlamaModel;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use std::path::Path;
    /// use llama_cpp_4::model::AddBos;
    /// let backend = llama_cpp_4::llama_backend::LlamaBackend::init()?;
    /// let model = LlamaModel::load_from_file(&backend, Path::new("path/to/model"), &Default::default())?;
    /// let tokens = model.str_to_token_fast("Hello, World!", AddBos::Always)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn str_to_token_fast(
        &self,
        str: &str,
        add_bos: AddBos,
    ) -> Result<Vec<LlamaToken>, StringToTokenError> {
        let add_bos = match add_bos {
            AddBos::Always => true,
            AddBos::Never => false,
        };

        // Estimate the initial buffer size based on the string length.
        // Each token typically takes ~2 bytes, so we estimate and reserve some extra space.
        let tokens_estimation = std::cmp::max(8, (str.len() / 2) + usize::from(add_bos));
        let mut buffer = Vec::with_capacity(tokens_estimation);

        // Convert the string to a C-style string (a null-terminated byte array).
        let c_string = CString::new(str)?;

        // Convert the buffer capacity to a c_int (required by llama-cpp)
        let buffer_capacity =
            c_int::try_from(buffer.capacity()).expect("buffer capacity should fit into a c_int");

        // Tokenize the string using llama-cpp.
        let size = unsafe {
            llama_cpp_sys_4::llama_tokenize(
                self.model.as_ptr(),
                c_string.as_ptr(),
                c_int::try_from(c_string.as_bytes().len())?,
                buffer.as_mut_ptr(),
                buffer_capacity,
                add_bos,
                true,
            )
        };

        // If the initial size is negative (buffer wasn't large enough), resize the vector and try again.
        let size = if size.is_negative() {
            buffer.reserve_exact(usize::try_from(-size).expect("usize's are larger "));
            unsafe {
                llama_cpp_sys_4::llama_tokenize(
                    self.model.as_ptr(),
                    c_string.as_ptr(),
                    c_int::try_from(c_string.as_bytes().len())?,
                    buffer.as_mut_ptr(),
                    -size,
                    add_bos,
                    true,
                )
            }
        } else {
            size
        };

        // Convert the size to usize and ensure it's valid.
        let size = usize::try_from(size).expect("size is positive and usize ");

        // Safety: Since `size` is less than the buffer capacity and llama-cpp has already initialized elements up to `size`,
        // we are safe to resize the buffer to the correct length.
        unsafe { buffer.set_len(size) }

        // Convert the raw tokens into `LlamaToken` instances.
        Ok(buffer.into_iter().map(LlamaToken).collect())
    }

    /// Get the type of a token.
    ///
    /// This function retrieves the attributes associated with a given token. The attributes are typically used to
    /// understand whether the token represents a special type of token (e.g., beginning-of-sequence (BOS), end-of-sequence (EOS),
    /// control tokens, etc.).
    ///
    /// # Panics
    ///
    /// - This function will panic if the token type is unknown or cannot be converted to a valid `LlamaTokenAttrs`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_4::model::{LlamaModel, LlamaToken};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = LlamaModel::load_from_file("path/to/model")?;
    /// let token = LlamaToken(42);
    /// let token_attrs = model.token_attr(token);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn token_attr(&self, LlamaToken(id): LlamaToken) -> LlamaTokenAttrs {
        let token_type = unsafe { llama_cpp_sys_4::llama_token_get_attr(self.model.as_ptr(), id) };
        LlamaTokenAttrs::try_from(token_type).expect("token type is valid")
    }

    /// Convert a token to a string with a specified buffer size.
    ///
    /// This function allows you to convert a token into a string, with the ability to specify a buffer size for the operation.
    /// It is generally recommended to use `LlamaModel::token_to_str` instead, as 8 bytes is typically sufficient for most tokens,
    /// and the extra buffer size doesn't usually matter.
    ///
    /// # Errors
    ///
    /// - If the token type is unknown, an error will be returned.
    /// - If the resultant token exceeds the provided `buffer_size`, an error will occur.
    /// - If the token string returned by `llama-cpp` is not valid UTF-8, it will return an error.
    ///
    /// # Panics
    ///
    /// - This function will panic if the `buffer_size` does not fit into a `c_int`.
    /// - It will also panic if the size returned from `llama-cpp` does not fit into a `usize`, which should typically never happen.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_4::model::{LlamaModel, LlamaToken};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = LlamaModel::load_from_file("path/to/model")?;
    /// let token = LlamaToken(42);
    /// let token_string = model.token_to_str_with_size(token, 32, Special::Plaintext)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn token_to_str_with_size(
        &self,
        token: LlamaToken,
        buffer_size: usize,
        special: Special,
    ) -> Result<String, TokenToStringError> {
        let bytes = self.token_to_bytes_with_size(token, buffer_size, special, None)?;
        Ok(String::from_utf8(bytes)?)
    }

    /// Convert a token to bytes with a specified buffer size.
    ///
    /// Similar to `token_to_str_with_size`, but instead of returning a string, this function converts the token into raw bytes.
    /// The buffer size parameter allows you to specify the maximum size to which the token can be converted. This is generally
    /// used when you need the raw byte representation of a token for further processing.
    ///
    /// # Errors
    ///
    /// - If the token type is unknown, an error will be returned.
    /// - If the resultant token exceeds the provided `buffer_size`, an error will occur.
    ///
    /// # Panics
    ///
    /// - This function will panic if `buffer_size` cannot fit into a `c_int`.
    /// - It will also panic if the size returned from `llama-cpp` cannot be converted to `usize` (which should not happen).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_4::model::{LlamaModel, LlamaToken};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = LlamaModel::load_from_file("path/to/model")?;
    /// let token = LlamaToken(42);
    /// let token_bytes = model.token_to_bytes_with_size(token, 32, Special::Plaintext, None)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn token_to_bytes_with_size(
        &self,
        token: LlamaToken,
        buffer_size: usize,
        special: Special,
        lstrip: Option<NonZeroU16>,
    ) -> Result<Vec<u8>, TokenToStringError> {
        if token == self.token_nl() {
            return Ok(String::from("\n").into_bytes());
        }

        // unsure what to do with this in the face of the 'special' arg + attr changes
        let attrs = self.token_attr(token);
        if attrs.contains(LlamaTokenAttr::Control)
            && (token == self.token_bos() || token == self.token_eos())
        {
            return Ok(Vec::new());
        } else if attrs.is_empty()
            || attrs
                .intersects(LlamaTokenAttr::Unknown | LlamaTokenAttr::Byte | LlamaTokenAttr::Unused)
        {
            return Ok(Vec::new());
        }

        let special = match special {
            Special::Tokenize => true,
            Special::Plaintext => false,
        };

        let string = CString::new(vec![b'*'; buffer_size]).expect("no null");
        let len = string.as_bytes().len();
        let len = c_int::try_from(len).expect("length fits into c_int");
        let buf = string.into_raw();
        let lstrip = lstrip.map_or(0, |it| i32::from(it.get()));
        let size = unsafe {
            llama_cpp_sys_4::llama_token_to_piece(
                self.model.as_ptr(),
                token.0,
                buf,
                len,
                lstrip,
                special,
            )
        };

        match size {
            0 => Err(TokenToStringError::UnknownTokenType),
            i if i.is_negative() => Err(TokenToStringError::InsufficientBufferSpace(i)),
            size => {
                let string = unsafe { CString::from_raw(buf) };
                let mut bytes = string.into_bytes();
                let len = usize::try_from(size).expect("size is positive and fits into usize");
                bytes.truncate(len);
                Ok(bytes)
            }
        }
    }

    /// The number of tokens the model was trained on.
    ///
    /// This function returns the number of tokens the model was trained on. It is returned as a `c_int` for maximum
    /// compatibility with the underlying llama-cpp library, though it can typically be cast to an `i32` without issue.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_4::model::LlamaModel;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = LlamaModel::load_from_file("path/to/model")?;
    /// let n_vocab = model.n_vocab();
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn n_vocab(&self) -> i32 {
        unsafe { llama_cpp_sys_4::llama_n_vocab(self.model.as_ptr()) }
    }

    /// The type of vocab the model was trained on.
    ///
    /// This function returns the type of vocabulary used by the model, such as whether it is based on byte-pair encoding (BPE),
    /// word-level tokens, or another tokenization scheme.
    ///
    /// # Panics
    ///
    /// - This function will panic if `llama-cpp` emits a vocab type that is not recognized or is invalid for this library.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_4::model::LlamaModel;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = LlamaModel::load_from_file("path/to/model")?;
    /// let vocab_type = model.vocab_type();
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn vocab_type(&self) -> VocabType {
        let vocab_type = unsafe { llama_cpp_sys_4::llama_vocab_type(self.model.as_ptr()) };
        VocabType::try_from(vocab_type).expect("invalid vocab type")
    }

    /// Returns the number of embedding dimensions for the model.
    ///
    /// This function retrieves the number of embeddings (or embedding dimensions) used by the model. It is typically
    /// used for analyzing model architecture and setting up context parameters or other model configuration aspects.
    ///
    /// # Panics
    ///
    /// - This function may panic if the underlying `llama-cpp` library returns an invalid embedding dimension value.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_4::model::LlamaModel;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = LlamaModel::load_from_file("path/to/model")?;
    /// let n_embd = model.n_embd();
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn n_embd(&self) -> c_int {
        unsafe { llama_cpp_sys_4::llama_n_embd(self.model.as_ptr()) }
    }

    /// Get the chat template from the model.
    ///
    /// This function retrieves the chat template associated with the model. The chat template defines the structure
    /// of conversational inputs and outputs, often useful in chat-based applications or interaction contexts.
    ///
    /// # Errors
    ///
    /// - If the model does not have a chat template, it will return an error.
    /// - If the chat template is not a valid `CString`, it will return an error.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_4::model::LlamaModel;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = LlamaModel::load_from_file("path/to/model")?;
    /// let chat_template = model.get_chat_template(1024)?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::missing_panics_doc)] // We statically know this will not panic as long as the buffer size is sufficient
    pub fn get_chat_template(&self, buf_size: usize) -> Result<String, ChatTemplateError> {
        // longest known template is about 1200 bytes from llama.cpp
        let chat_temp = CString::new(vec![b'*'; buf_size]).expect("no null");
        let chat_ptr = chat_temp.into_raw();
        let chat_name = CString::new("tokenizer.chat_template").expect("no null bytes");

        let ret = unsafe {
            llama_cpp_sys_4::llama_model_meta_val_str(
                self.model.as_ptr(),
                chat_name.as_ptr(),
                chat_ptr,
                buf_size,
            )
        };

        if ret < 0 {
            return Err(ChatTemplateError::MissingTemplate(ret));
        }

        let template_c = unsafe { CString::from_raw(chat_ptr) };
        let template = template_c.to_str()?;

        let ret: usize = ret.try_into().unwrap();
        if template.len() < ret {
            return Err(ChatTemplateError::BuffSizeError(ret + 1));
        }

        Ok(template.to_owned())
    }

    /// Loads a model from a file.
    ///
    /// This function loads a model from a specified file path and returns the corresponding `LlamaModel` instance.
    ///
    /// # Errors
    ///
    /// - If the path cannot be converted to a string or if the model file does not exist, it will return an error.
    /// - If the model cannot be loaded (e.g., due to an invalid or corrupted model file), it will return a `LlamaModelLoadError`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_4::model::LlamaModel;
    /// use std::path::Path;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = LlamaModel::load_from_file("path/to/model", &LlamaModelParams::default())?;
    /// # Ok(())
    /// # }
    /// ```
    #[tracing::instrument(skip_all, fields(params))]
    pub fn load_from_file(
        _: &LlamaBackend,
        path: impl AsRef<Path>,
        params: &LlamaModelParams,
    ) -> Result<Self, LlamaModelLoadError> {
        let path = path.as_ref();
        debug_assert!(Path::new(path).exists(), "{path:?} does not exist");
        let path = path
            .to_str()
            .ok_or(LlamaModelLoadError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;
        let llama_model =
            unsafe { llama_cpp_sys_4::llama_load_model_from_file(cstr.as_ptr(), params.params) };

        let model = NonNull::new(llama_model).ok_or(LlamaModelLoadError::NullResult)?;

        tracing::debug!(?path, "Loaded model");
        Ok(LlamaModel { model })
    }

    /// Initializes a lora adapter from a file.
    ///
    /// This function initializes a Lora adapter, which is a model extension used to adapt or fine-tune the existing model
    /// to a specific domain or task. The adapter file is typically in the form of a binary or serialized file that can be applied
    /// to the model for improved performance on specialized tasks.
    ///
    /// # Errors
    ///
    /// - If the adapter file path cannot be converted to a string or if the adapter cannot be initialized, it will return an error.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_4::model::{LlamaModel, LlamaLoraAdapter};
    /// use std::path::Path;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = LlamaModel::load_from_file("path/to/model", &LlamaModelParams::default())?;
    /// let adapter = model.lora_adapter_init("path/to/lora/adapter")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn lora_adapter_init(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<LlamaLoraAdapter, LlamaLoraAdapterInitError> {
        let path = path.as_ref();
        debug_assert!(Path::new(path).exists(), "{path:?} does not exist");

        let path = path
            .to_str()
            .ok_or(LlamaLoraAdapterInitError::PathToStrError(
                path.to_path_buf(),
            ))?;

        let cstr = CString::new(path)?;
        let adapter =
            unsafe { llama_cpp_sys_4::llama_lora_adapter_init(self.model.as_ptr(), cstr.as_ptr()) };

        let adapter = NonNull::new(adapter).ok_or(LlamaLoraAdapterInitError::NullResult)?;

        tracing::debug!(?path, "Initialized lora adapter");
        Ok(LlamaLoraAdapter {
            lora_adapter: adapter,
        })
    }

    /// Create a new context from this model.
    ///
    /// This function creates a new context for the model, which is used to manage and perform computations for inference,
    /// including token generation, embeddings, and other tasks that the model can perform. The context allows fine-grained
    /// control over model parameters for a specific task.
    ///
    /// # Errors
    ///
    /// - There are various potential failures such as invalid parameters or a failure to allocate the context. See [`LlamaContextLoadError`]
    ///   for more detailed error descriptions.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_4::model::{LlamaModel, LlamaContext};
    /// use llama_cpp_4::LlamaContextParams;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = LlamaModel::load_from_file("path/to/model", &LlamaModelParams::default())?;
    /// let context = model.new_context(&LlamaBackend::init()?, LlamaContextParams::default())?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::needless_pass_by_value)]
    pub fn new_context(
        &self,
        _: &LlamaBackend,
        params: LlamaContextParams,
    ) -> Result<LlamaContext, LlamaContextLoadError> {
        let context_params = params.context_params;
        let context = unsafe {
            llama_cpp_sys_4::llama_new_context_with_model(self.model.as_ptr(), context_params)
        };
        let context = NonNull::new(context).ok_or(LlamaContextLoadError::NullReturn)?;

        Ok(LlamaContext::new(self, context, params.embeddings()))
    }

    /// Apply the model's chat template to a sequence of messages.
    ///
    /// This function applies the model's chat template to the provided chat messages, formatting them accordingly. The chat
    /// template determines the structure or style of conversation between the system and user, such as token formatting,
    /// role separation, and more. The template can be customized by providing an optional template string, or if `None`
    /// is provided, the default template used by `llama.cpp` will be applied.
    ///
    /// For more information on supported templates, visit:
    /// https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
    ///
    /// # Arguments
    ///
    /// - `tmpl`: An optional custom template string. If `None`, the default template will be used.
    /// - `chat`: A vector of `LlamaChatMessage` instances, which represent the conversation between the system and user.
    /// - `add_ass`: A boolean flag indicating whether additional system-specific instructions (like "assistant") should be included.
    ///
    /// # Errors
    ///
    /// There are several possible points of failure when applying the chat template:
    /// - Insufficient buffer size to hold the formatted chat (this will return `ApplyChatTemplateError::BuffSizeError`).
    /// - If the template or messages cannot be processed properly, various errors from `ApplyChatTemplateError` may occur.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_4::model::{LlamaModel, LlamaChatMessage};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model = LlamaModel::load_from_file("path/to/model", &LlamaModelParams::default())?;
    /// let chat = vec![
    ///     LlamaChatMessage::new("user", "Hello!"),
    ///     LlamaChatMessage::new("assistant", "Hi! How can I assist you today?"),
    /// ];
    /// let formatted_chat = model.apply_chat_template(None, chat, true)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Notes
    ///
    /// The provided buffer is twice the length of the messages by default, which is recommended by the `llama.cpp` documentation.
    #[tracing::instrument(skip_all)]
    pub fn apply_chat_template(
        &self,
        tmpl: Option<String>,
        chat: Vec<LlamaChatMessage>,
        add_ass: bool,
    ) -> Result<String, ApplyChatTemplateError> {
        // Buffer is twice the length of messages per their recommendation
        let message_length = chat.iter().fold(0, |acc, c| {
            acc + c.role.to_bytes().len() + c.content.to_bytes().len()
        });
        let mut buff = vec![0; message_length * 4];

        // Build our llama_cpp_sys_4 chat messages
        let chat: Vec<llama_cpp_sys_4::llama_chat_message> = chat
            .iter()
            .map(|c| llama_cpp_sys_4::llama_chat_message {
                role: c.role.as_ptr(),
                content: c.content.as_ptr(),
            })
            .collect();

        // Set the tmpl pointer
        let tmpl = tmpl.map(CString::new);
        let tmpl_ptr = match &tmpl {
            Some(str) => str.as_ref().map_err(Clone::clone)?.as_ptr(),
            None => std::ptr::null(),
        };

        let formatted_chat = unsafe {
            let res = llama_cpp_sys_4::llama_chat_apply_template(
                self.model.as_ptr(),
                tmpl_ptr,
                chat.as_ptr(),
                chat.len(),
                add_ass,
                buff.as_mut_ptr(),
                buff.len() as i32,
            );
            // A buffer twice the size should be sufficient for all models, if this is not the case for a new model, we can increase it
            // The error message informs the user to contact a maintainer
            if res > buff.len() as i32 {
                return Err(ApplyChatTemplateError::BuffSizeError);
            }
            Ok::<String, ApplyChatTemplateError>(
                CStr::from_ptr(buff.as_mut_ptr())
                    .to_string_lossy()
                    .to_string(),
            )
        }?;
        Ok(formatted_chat)
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_4::llama_free_model(self.model.as_ptr()) }
    }
}

/// Defines the possible types of vocabulary used by the model.
///
/// The model may use different types of vocabulary depending on the tokenization method chosen during training.
/// This enum represents these types, specifically `BPE` (Byte Pair Encoding) and `SPM` (SentencePiece).
///
/// # Variants
///
/// - `BPE`: Byte Pair Encoding, a common tokenization method used in NLP tasks.
/// - `SPM`: SentencePiece, another popular tokenization method for NLP models.
///
/// # Example
///
/// ```no_run
/// use llama_cpp_4::model::VocabType;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let vocab_type = VocabType::BPE;
/// match vocab_type {
///     VocabType::BPE => println!("The model uses Byte Pair Encoding (BPE)"),
///     VocabType::SPM => println!("The model uses SentencePiece (SPM)"),
/// }
/// # Ok(())
/// # }
/// ```
#[repr(u32)]
#[derive(Debug, Eq, Copy, Clone, PartialEq)]
pub enum VocabType {
    /// Byte Pair Encoding
    BPE = llama_cpp_sys_4::LLAMA_VOCAB_TYPE_BPE as _,
    /// Sentence Piece Tokenizer
    SPM = llama_cpp_sys_4::LLAMA_VOCAB_TYPE_SPM as _,
}

/// Error that occurs when trying to convert a `llama_vocab_type` to a `VocabType`.
///
/// This error is raised when the integer value returned by the system does not correspond to a known vocabulary type.
///
/// # Variants
///
/// - `UnknownValue`: The error is raised when the value is not a valid `llama_vocab_type`. The invalid value is returned with the error.
///
/// # Example
///
/// ```no_run
/// use llama_cpp_4::model::LlamaTokenTypeFromIntError;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let invalid_value = 999; // Not a valid vocabulary type
/// let error = LlamaTokenTypeFromIntError::UnknownValue(invalid_value);
/// println!("Error: {}", error);
/// # Ok(())
/// # }
/// ```
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum LlamaTokenTypeFromIntError {
    /// The value is not a valid `llama_token_type`. Contains the int value that was invalid.
    #[error("Unknown Value {0}")]
    UnknownValue(llama_cpp_sys_4::llama_vocab_type),
}

impl TryFrom<llama_cpp_sys_4::llama_vocab_type> for VocabType {
    type Error = LlamaTokenTypeFromIntError;

    fn try_from(value: llama_cpp_sys_4::llama_vocab_type) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_sys_4::LLAMA_VOCAB_TYPE_BPE => Ok(VocabType::BPE),
            llama_cpp_sys_4::LLAMA_VOCAB_TYPE_SPM => Ok(VocabType::SPM),
            unknown => Err(LlamaTokenTypeFromIntError::UnknownValue(unknown)),
        }
    }
}
