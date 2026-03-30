//! A safe wrapper around `llama_model`.
use std::ffi::CStr;
use std::ffi::CString;
use std::fmt;
use std::num::NonZeroU16;
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::ptr::NonNull;

use llama_cpp_sys_4::{
    llama_adapter_lora, llama_adapter_lora_init, llama_add_bos_token, llama_add_eos_token,
    llama_chat_apply_template, llama_chat_builtin_templates, llama_chat_message,
    llama_detokenize, llama_free_model, llama_load_model_from_file, llama_model,
    llama_model_cls_label, llama_model_decoder_start_token, llama_model_desc,
    llama_model_get_vocab, llama_model_has_decoder, llama_model_has_encoder,
    llama_model_is_diffusion, llama_model_is_hybrid, llama_model_is_recurrent,
    llama_model_load_from_splits, llama_model_meta_count, llama_model_meta_key_by_index,
    llama_model_meta_val_str, llama_model_meta_val_str_by_index, llama_model_n_cls_out,
    llama_model_n_embd_inp, llama_model_n_embd_out, llama_model_n_head_kv, llama_model_n_params,
    llama_model_n_swa, llama_model_rope_freq_scale_train, llama_model_rope_type,
    llama_model_save_to_file, llama_model_size, llama_n_ctx_train, llama_n_embd, llama_n_head,
    llama_n_layer, llama_n_vocab, llama_new_context_with_model, llama_split_path,
    llama_split_prefix, llama_token_bos, llama_token_cls, llama_token_eos, llama_token_eot,
    llama_token_fim_mid, llama_token_fim_pad, llama_token_fim_pre, llama_token_fim_rep,
    llama_token_fim_sep, llama_token_fim_suf, llama_token_get_attr, llama_token_get_score,
    llama_token_get_text, llama_token_is_control, llama_token_is_eog, llama_token_nl,
    llama_token_pad, llama_token_sep, llama_token_to_piece, llama_tokenize, llama_vocab,
    llama_vocab_type, LLAMA_VOCAB_TYPE_BPE, LLAMA_VOCAB_TYPE_SPM,
};

use crate::context::params::LlamaContextParams;
use crate::context::LlamaContext;
use crate::llama_backend::LlamaBackend;
use crate::model::params::LlamaModelParams;
use crate::token::LlamaToken;
use crate::token_type::{LlamaTokenAttr, LlamaTokenAttrs};
use crate::{
    ApplyChatTemplateError, ChatTemplateError, LlamaContextLoadError, LlamaLoraAdapterInitError,
    LlamaModelLoadError, NewLlamaChatMessageError, StringFromModelError, StringToTokenError,
    TokenToStringError,
};

pub mod params;

/// A safe wrapper around `llama_model`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaModel {
    pub(crate) model: NonNull<llama_model>,
}

/// A safe wrapper around `llama_vocab`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaVocab {
    pub(crate) vocab: NonNull<llama_vocab>,
}

impl LlamaVocab {
    /// Get the number of tokens in the vocabulary.
    #[must_use]
    pub fn n_tokens(&self) -> i32 {
        unsafe { llama_cpp_sys_4::llama_vocab_n_tokens(self.vocab.as_ref()) }
    }

    /// Get the vocabulary type.
    #[must_use]
    pub fn vocab_type(&self) -> u32 {
        unsafe { llama_cpp_sys_4::llama_vocab_type(self.vocab.as_ref()).try_into().unwrap() }
    }

    /// Get the BOS token.
    #[must_use]
    pub fn bos(&self) -> LlamaToken {
        LlamaToken(unsafe { llama_cpp_sys_4::llama_vocab_bos(self.vocab.as_ref()) })
    }

    /// Get the EOS token.
    #[must_use]
    pub fn eos(&self) -> LlamaToken {
        LlamaToken(unsafe { llama_cpp_sys_4::llama_vocab_eos(self.vocab.as_ref()) })
    }

    /// Get the EOT (end of turn) token.
    #[must_use]
    pub fn eot(&self) -> LlamaToken {
        LlamaToken(unsafe { llama_cpp_sys_4::llama_vocab_eot(self.vocab.as_ref()) })
    }

    /// Get the CLS (classification) token.
    #[must_use]
    pub fn cls(&self) -> LlamaToken {
        LlamaToken(unsafe { llama_cpp_sys_4::llama_vocab_cls(self.vocab.as_ref()) })
    }

    /// Get the SEP (separator) token.
    #[must_use]
    pub fn sep(&self) -> LlamaToken {
        LlamaToken(unsafe { llama_cpp_sys_4::llama_vocab_sep(self.vocab.as_ref()) })
    }

    /// Get the NL (newline) token.
    #[must_use]
    pub fn nl(&self) -> LlamaToken {
        LlamaToken(unsafe { llama_cpp_sys_4::llama_vocab_nl(self.vocab.as_ref()) })
    }

    /// Get the PAD (padding) token.
    #[must_use]
    pub fn pad(&self) -> LlamaToken {
        LlamaToken(unsafe { llama_cpp_sys_4::llama_vocab_pad(self.vocab.as_ref()) })
    }

    /// Get the FIM prefix token.
    #[must_use]
    pub fn fim_pre(&self) -> LlamaToken {
        LlamaToken(unsafe { llama_cpp_sys_4::llama_vocab_fim_pre(self.vocab.as_ref()) })
    }

    /// Get the FIM suffix token.
    #[must_use]
    pub fn fim_suf(&self) -> LlamaToken {
        LlamaToken(unsafe { llama_cpp_sys_4::llama_vocab_fim_suf(self.vocab.as_ref()) })
    }

    /// Get the FIM middle token.
    #[must_use]
    pub fn fim_mid(&self) -> LlamaToken {
        LlamaToken(unsafe { llama_cpp_sys_4::llama_vocab_fim_mid(self.vocab.as_ref()) })
    }

    /// Get the FIM padding token.
    #[must_use]
    pub fn fim_pad(&self) -> LlamaToken {
        LlamaToken(unsafe { llama_cpp_sys_4::llama_vocab_fim_pad(self.vocab.as_ref()) })
    }

    /// Get the FIM repository token.
    #[must_use]
    pub fn fim_rep(&self) -> LlamaToken {
        LlamaToken(unsafe { llama_cpp_sys_4::llama_vocab_fim_rep(self.vocab.as_ref()) })
    }

    /// Get the FIM separator token.
    #[must_use]
    pub fn fim_sep(&self) -> LlamaToken {
        LlamaToken(unsafe { llama_cpp_sys_4::llama_vocab_fim_sep(self.vocab.as_ref()) })
    }

    /// Check whether BOS should be added.
    #[must_use]
    pub fn get_add_bos(&self) -> bool {
        unsafe { llama_cpp_sys_4::llama_vocab_get_add_bos(self.vocab.as_ref()) }
    }

    /// Check whether EOS should be added.
    #[must_use]
    pub fn get_add_eos(&self) -> bool {
        unsafe { llama_cpp_sys_4::llama_vocab_get_add_eos(self.vocab.as_ref()) }
    }

    /// Check whether SEP should be added.
    #[must_use]
    pub fn get_add_sep(&self) -> bool {
        unsafe { llama_cpp_sys_4::llama_vocab_get_add_sep(self.vocab.as_ref()) }
    }

    /// Get the text representation of a token.
    ///
    /// # Errors
    ///
    /// Returns an error if the text pointer is null or not valid UTF-8.
    pub fn get_text(&self, token: LlamaToken) -> Result<&str, StringFromModelError> {
        let ptr = unsafe { llama_cpp_sys_4::llama_vocab_get_text(self.vocab.as_ref(), token.0) };
        if ptr.is_null() {
            return Err(StringFromModelError::ReturnedError(-1));
        }
        let cstr = unsafe { CStr::from_ptr(ptr) };
        cstr.to_str().map_err(StringFromModelError::Utf8Error)
    }

    /// Get the score of a token.
    #[must_use]
    pub fn get_score(&self, token: LlamaToken) -> f32 {
        unsafe { llama_cpp_sys_4::llama_vocab_get_score(self.vocab.as_ref(), token.0) }
    }

    /// Get the attributes of a token.
    #[must_use]
    pub fn get_attr(&self, token: LlamaToken) -> u32 {
        unsafe { llama_cpp_sys_4::llama_vocab_get_attr(self.vocab.as_ref(), token.0).try_into().unwrap() }
    }

    /// Check if a token is a control token.
    #[must_use]
    pub fn is_control(&self, token: LlamaToken) -> bool {
        unsafe { llama_cpp_sys_4::llama_vocab_is_control(self.vocab.as_ref(), token.0) }
    }

    /// Check if a token is an end-of-generation token.
    #[must_use]
    pub fn is_eog(&self, token: LlamaToken) -> bool {
        unsafe { llama_cpp_sys_4::llama_vocab_is_eog(self.vocab.as_ref(), token.0) }
    }

    /// Get the token mask value for the vocabulary.
    #[must_use]
    pub fn mask(&self) -> LlamaToken {
        LlamaToken(unsafe { llama_cpp_sys_4::llama_vocab_mask(self.vocab.as_ref()) })
    }
}

/// A safe wrapper around `llama_adapter_lora`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaLoraAdapter {
    pub(crate) lora_adapter: NonNull<llama_adapter_lora>,
}

impl LlamaLoraAdapter {
    /// Get the number of metadata key-value pairs in the adapter.
    #[must_use]
    pub fn meta_count(&self) -> i32 {
        unsafe { llama_cpp_sys_4::llama_adapter_meta_count(self.lora_adapter.as_ptr()) }
    }

    /// Get a metadata key by index.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of range or the key is not valid UTF-8.
    #[allow(clippy::cast_sign_loss)]
    pub fn meta_key_by_index(
        &self,
        index: i32,
        buf_size: usize,
    ) -> Result<String, StringFromModelError> {
        let mut buf = vec![0u8; buf_size];
        let ret = unsafe {
            llama_cpp_sys_4::llama_adapter_meta_key_by_index(
                self.lora_adapter.as_ptr(),
                index,
                buf.as_mut_ptr().cast::<c_char>(),
                buf_size,
            )
        };
        if ret < 0 {
            return Err(StringFromModelError::ReturnedError(ret));
        }
        let len = ret as usize;
        let s = std::str::from_utf8(&buf[..len]).map_err(StringFromModelError::Utf8Error)?;
        Ok(s.to_owned())
    }

    /// Get a metadata value by key name.
    ///
    /// # Errors
    ///
    /// Returns an error if the key is not found or the value is not valid UTF-8.
    #[allow(clippy::cast_sign_loss)]
    pub fn meta_val_str(
        &self,
        key: &str,
        buf_size: usize,
    ) -> Result<String, StringFromModelError> {
        let c_key =
            CString::new(key).map_err(|_| StringFromModelError::ReturnedError(-1))?;
        let mut buf = vec![0u8; buf_size];
        let ret = unsafe {
            llama_cpp_sys_4::llama_adapter_meta_val_str(
                self.lora_adapter.as_ptr(),
                c_key.as_ptr(),
                buf.as_mut_ptr().cast::<c_char>(),
                buf_size,
            )
        };
        if ret < 0 {
            return Err(StringFromModelError::ReturnedError(ret));
        }
        let len = ret as usize;
        let s = std::str::from_utf8(&buf[..len]).map_err(StringFromModelError::Utf8Error)?;
        Ok(s.to_owned())
    }

    /// Get a metadata value by index.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of range or the value is not valid UTF-8.
    #[allow(clippy::cast_sign_loss)]
    pub fn meta_val_str_by_index(
        &self,
        index: i32,
        buf_size: usize,
    ) -> Result<String, StringFromModelError> {
        let mut buf = vec![0u8; buf_size];
        let ret = unsafe {
            llama_cpp_sys_4::llama_adapter_meta_val_str_by_index(
                self.lora_adapter.as_ptr(),
                index,
                buf.as_mut_ptr().cast::<c_char>(),
                buf_size,
            )
        };
        if ret < 0 {
            return Err(StringFromModelError::ReturnedError(ret));
        }
        let len = ret as usize;
        let s = std::str::from_utf8(&buf[..len]).map_err(StringFromModelError::Utf8Error)?;
        Ok(s.to_owned())
    }

    /// Get all metadata as a list of `(key, value)` pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if any key or value cannot be read or is not valid UTF-8.
    #[allow(clippy::cast_sign_loss)]
    pub fn metadata(&self) -> Result<Vec<(String, String)>, StringFromModelError> {
        let count = self.meta_count();
        let mut result = Vec::with_capacity(count as usize);
        for i in 0..count {
            let key = self.meta_key_by_index(i, 256)?;
            let val = self.meta_val_str_by_index(i, 4096)?;
            result.push((key, val));
        }
        Ok(result)
    }

    /// Get the number of invocation tokens for this adapter.
    #[must_use]
    pub fn n_invocation_tokens(&self) -> u64 {
        unsafe {
            llama_cpp_sys_4::llama_adapter_get_alora_n_invocation_tokens(
                self.lora_adapter.as_ptr(),
            )
        }
    }

    /// Get the invocation tokens for this adapter.
    ///
    /// Returns an empty slice if there are no invocation tokens.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn invocation_tokens(&self) -> &[LlamaToken] {
        let n = self.n_invocation_tokens() as usize;
        if n == 0 {
            return &[];
        }
        let ptr = unsafe {
            llama_cpp_sys_4::llama_adapter_get_alora_invocation_tokens(
                self.lora_adapter.as_ptr(),
            )
        };
        if ptr.is_null() {
            return &[];
        }
        // LlamaToken is repr(transparent) over llama_token (i32), so this cast is safe
        unsafe { std::slice::from_raw_parts(ptr.cast::<LlamaToken>(), n) }
    }
}

impl Drop for LlamaLoraAdapter {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys_4::llama_adapter_lora_free(self.lora_adapter.as_ptr());
        }
    }
}

/// A Safe wrapper around `llama_chat_message`
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct LlamaChatMessage {
    role: CString,
    content: CString,
}

impl LlamaChatMessage {
    /// Create a new `LlamaChatMessage`.
    ///
    /// # Errors
    ///
    /// Returns [`NewLlamaChatMessageError`] if the role or content contains a null byte.
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
    /// Retrieves the vocabulary associated with the current Llama model.
    ///
    /// This method fetches the vocabulary from the underlying model using an unsafe
    /// FFI call. The returned `LlamaVocab` struct contains a non-null pointer to
    /// the vocabulary data, which is wrapped in a `NonNull` for safety.
    ///
    /// # Safety
    /// This method uses an unsafe block to call a C function (`llama_model_get_vocab`),
    /// which is assumed to return a valid pointer to the vocabulary. The caller should
    /// ensure that the model object is properly initialized and valid before calling
    /// this method, as dereferencing invalid pointers can lead to undefined behavior.
    ///
    /// # Returns
    /// A `LlamaVocab` struct containing the vocabulary of the model.
    ///
    /// # Panics
    ///
    /// Panics if the underlying C function returns a null pointer.
    ///
    /// # Example
    /// ```rust,ignore
    /// let vocab = model.get_vocab();
    /// ```
    #[must_use]
    pub fn get_vocab(&self) -> LlamaVocab {
        let llama_vocab = unsafe { llama_model_get_vocab(self.model.as_ptr()) }.cast_mut();

        LlamaVocab {
            vocab: NonNull::new(llama_vocab).unwrap(),
        }
    }
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
        let n_ctx_train = unsafe { llama_n_ctx_train(self.model.as_ptr()) };
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
        let token = unsafe { llama_token_bos(self.get_vocab().vocab.as_ref()) };
        LlamaToken(token)
    }

    /// Get the end of stream token.
    ///
    /// This function returns the token that represents the end of a stream (EOS token).
    #[must_use]
    pub fn token_eos(&self) -> LlamaToken {
        let token = unsafe { llama_token_eos(self.get_vocab().vocab.as_ref()) };
        LlamaToken(token)
    }

    /// Get the newline token.
    ///
    /// This function returns the token that represents a newline character.
    #[must_use]
    pub fn token_nl(&self) -> LlamaToken {
        let token = unsafe { llama_token_nl(self.get_vocab().vocab.as_ref()) };
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
        unsafe { llama_token_is_eog(self.get_vocab().vocab.as_ref(), token.0) }
    }

    /// Get the classification token.
    #[must_use]
    pub fn token_cls(&self) -> LlamaToken {
        let token = unsafe { llama_token_cls(self.get_vocab().vocab.as_ref()) };
        LlamaToken(token)
    }

    /// Get the end-of-turn token.
    #[must_use]
    pub fn token_eot(&self) -> LlamaToken {
        let token = unsafe { llama_token_eot(self.get_vocab().vocab.as_ref()) };
        LlamaToken(token)
    }

    /// Get the padding token.
    #[must_use]
    pub fn token_pad(&self) -> LlamaToken {
        let token = unsafe { llama_token_pad(self.get_vocab().vocab.as_ref()) };
        LlamaToken(token)
    }

    /// Get the separator token.
    #[must_use]
    pub fn token_sep(&self) -> LlamaToken {
        let token = unsafe { llama_token_sep(self.get_vocab().vocab.as_ref()) };
        LlamaToken(token)
    }

    /// Get the fill-in-the-middle prefix token.
    #[must_use]
    pub fn token_fim_pre(&self) -> LlamaToken {
        let token = unsafe { llama_token_fim_pre(self.get_vocab().vocab.as_ref()) };
        LlamaToken(token)
    }

    /// Get the fill-in-the-middle suffix token.
    #[must_use]
    pub fn token_fim_suf(&self) -> LlamaToken {
        let token = unsafe { llama_token_fim_suf(self.get_vocab().vocab.as_ref()) };
        LlamaToken(token)
    }

    /// Get the fill-in-the-middle middle token.
    #[must_use]
    pub fn token_fim_mid(&self) -> LlamaToken {
        let token = unsafe { llama_token_fim_mid(self.get_vocab().vocab.as_ref()) };
        LlamaToken(token)
    }

    /// Get the fill-in-the-middle padding token.
    #[must_use]
    pub fn token_fim_pad(&self) -> LlamaToken {
        let token = unsafe { llama_token_fim_pad(self.get_vocab().vocab.as_ref()) };
        LlamaToken(token)
    }

    /// Get the fill-in-the-middle repository token.
    #[must_use]
    pub fn token_fim_rep(&self) -> LlamaToken {
        let token = unsafe { llama_token_fim_rep(self.get_vocab().vocab.as_ref()) };
        LlamaToken(token)
    }

    /// Get the fill-in-the-middle separator token.
    #[must_use]
    pub fn token_fim_sep(&self) -> LlamaToken {
        let token = unsafe { llama_token_fim_sep(self.get_vocab().vocab.as_ref()) };
        LlamaToken(token)
    }

    /// Check if a token is a control token.
    #[must_use]
    pub fn token_is_control(&self, token: LlamaToken) -> bool {
        unsafe { llama_token_is_control(self.get_vocab().vocab.as_ref(), token.0) }
    }

    /// Get the score of a token.
    #[must_use]
    pub fn token_get_score(&self, token: LlamaToken) -> f32 {
        unsafe { llama_token_get_score(self.get_vocab().vocab.as_ref(), token.0) }
    }

    /// Get the raw text of a token.
    ///
    /// # Errors
    ///
    /// Returns an error if the token text is null or not valid UTF-8.
    pub fn token_get_text(&self, token: LlamaToken) -> Result<&str, StringFromModelError> {
        let ptr =
            unsafe { llama_token_get_text(self.get_vocab().vocab.as_ref(), token.0) };
        if ptr.is_null() {
            return Err(StringFromModelError::ReturnedError(-1));
        }
        let cstr = unsafe { CStr::from_ptr(ptr) };
        cstr.to_str()
            .map_err(StringFromModelError::Utf8Error)
    }

    /// Check if a BOS token should be added when tokenizing.
    #[must_use]
    pub fn add_bos_token(&self) -> bool {
        unsafe { llama_add_bos_token(self.get_vocab().vocab.as_ref()) }
    }

    /// Check if an EOS token should be added when tokenizing.
    #[must_use]
    pub fn add_eos_token(&self) -> bool {
        unsafe { llama_add_eos_token(self.get_vocab().vocab.as_ref()) }
    }

    /// Get the decoder start token.
    ///
    /// This function returns the token used to signal the start of decoding (i.e., the token used at the start
    /// of a sequence generation).
    #[must_use]
    pub fn decode_start_token(&self) -> LlamaToken {
        let token = unsafe { llama_model_decoder_start_token(self.model.as_ptr()) };
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
            llama_tokenize(
                self.get_vocab().vocab.as_ref(),
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
                llama_tokenize(
                    self.get_vocab().vocab.as_ref(),
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
        let token_type = unsafe { llama_token_get_attr(self.get_vocab().vocab.as_ref(), id) };
        LlamaTokenAttrs::try_from(token_type).expect("token type is valid")
    }

    /// Detokenize a slice of tokens into a string.
    ///
    /// This is the inverse of [`str_to_token`](Self::str_to_token).
    ///
    /// # Parameters
    ///
    /// - `tokens`: The tokens to detokenize.
    /// - `remove_special`: If `true`, special tokens are removed from the output.
    /// - `unparse_special`: If `true`, special tokens are rendered as their text representation.
    ///
    /// # Errors
    ///
    /// Returns an error if the detokenized text is not valid UTF-8.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    pub fn detokenize(
        &self,
        tokens: &[LlamaToken],
        remove_special: bool,
        unparse_special: bool,
    ) -> Result<String, StringFromModelError> {
        // First call with empty buffer to get required size
        let n_tokens = tokens.len() as i32;
        let token_ptr = tokens.as_ptr().cast::<llama_cpp_sys_4::llama_token>();
        let needed = unsafe {
            llama_detokenize(
                self.get_vocab().vocab.as_ref(),
                token_ptr,
                n_tokens,
                std::ptr::null_mut(),
                0,
                remove_special,
                unparse_special,
            )
        };
        // llama_detokenize returns negative required size when buffer is too small
        let buf_size = if needed < 0 { (-needed) as usize } else { needed as usize };
        let mut buf = vec![0u8; buf_size];
        let ret = unsafe {
            llama_detokenize(
                self.get_vocab().vocab.as_ref(),
                token_ptr,
                n_tokens,
                buf.as_mut_ptr().cast::<c_char>(),
                buf_size as i32,
                remove_special,
                unparse_special,
            )
        };
        if ret < 0 {
            return Err(StringFromModelError::ReturnedError(ret));
        }
        let len = ret as usize;
        let s = std::str::from_utf8(&buf[..len]).map_err(StringFromModelError::Utf8Error)?;
        Ok(s.to_owned())
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
    /// Generally you should use [`LlamaModel::token_to_bytes`] instead as 8 bytes is enough for most words and
    /// the extra bytes do not really matter.
    ///
    /// # Errors
    ///
    /// - if the token type is unknown
    /// - the resultant token is larger than `buffer_size`.
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
        if (attrs.contains(LlamaTokenAttr::Control)
            && (token == self.token_bos() || token == self.token_eos()))
            || attrs.is_empty()
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
            llama_token_to_piece(
                self.get_vocab().vocab.as_ref(),
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
        unsafe { llama_n_vocab(self.get_vocab().vocab.as_ref()) }
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
        let vocab_type = unsafe { llama_vocab_type(self.get_vocab().vocab.as_ref()) };
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
        unsafe { llama_n_embd(self.model.as_ptr()) }
    }

    /// Get the number of transformer layers in the model.
    #[must_use]
    pub fn n_layer(&self) -> c_int {
        unsafe { llama_n_layer(self.model.as_ptr()) }
    }

    /// Get the number of attention heads in the model.
    #[must_use]
    pub fn n_head(&self) -> c_int {
        unsafe { llama_n_head(self.model.as_ptr()) }
    }

    /// Get the number of key-value attention heads in the model.
    #[must_use]
    pub fn n_head_kv(&self) -> c_int {
        unsafe { llama_model_n_head_kv(self.model.as_ptr()) }
    }

    /// Get the input embedding size of the model.
    #[must_use]
    pub fn n_embd_inp(&self) -> c_int {
        unsafe { llama_model_n_embd_inp(self.model.as_ptr()) }
    }

    /// Get the output embedding size of the model.
    #[must_use]
    pub fn n_embd_out(&self) -> c_int {
        unsafe { llama_model_n_embd_out(self.model.as_ptr()) }
    }

    /// Get the sliding window attention size of the model.
    /// Returns 0 if the model does not use sliding window attention.
    #[must_use]
    pub fn n_swa(&self) -> c_int {
        unsafe { llama_model_n_swa(self.model.as_ptr()) }
    }

    /// Get the `RoPE` type used by the model.
    #[must_use]
    pub fn rope_type(&self) -> i32 {
        unsafe { llama_model_rope_type(self.model.as_ptr()) }
    }

    /// Get the `RoPE` frequency scale used during training.
    #[must_use]
    pub fn rope_freq_scale_train(&self) -> f32 {
        unsafe { llama_model_rope_freq_scale_train(self.model.as_ptr()) }
    }

    /// Get the model size in bytes.
    #[must_use]
    pub fn model_size(&self) -> u64 {
        unsafe { llama_model_size(self.model.as_ptr()) }
    }

    /// Get the number of parameters in the model.
    #[must_use]
    pub fn n_params(&self) -> u64 {
        unsafe { llama_model_n_params(self.model.as_ptr()) }
    }

    /// Get the number of classification outputs.
    #[must_use]
    pub fn n_cls_out(&self) -> u32 {
        unsafe { llama_model_n_cls_out(self.model.as_ptr()) }
    }

    /// Get the classification label for the given index.
    ///
    /// # Errors
    ///
    /// Returns an error if the label is null or not valid UTF-8.
    pub fn cls_label(&self, index: u32) -> Result<&str, StringFromModelError> {
        let ptr = unsafe { llama_model_cls_label(self.model.as_ptr(), index) };
        if ptr.is_null() {
            return Err(StringFromModelError::ReturnedError(-1));
        }
        let cstr = unsafe { CStr::from_ptr(ptr) };
        cstr.to_str().map_err(StringFromModelError::Utf8Error)
    }

    /// Get the number of metadata key-value pairs.
    #[must_use]
    pub fn meta_count(&self) -> c_int {
        unsafe { llama_model_meta_count(self.model.as_ptr()) }
    }

    /// Get a model description string.
    ///
    /// The `buf_size` parameter specifies the maximum buffer size for the description.
    /// A default of 256 bytes is usually sufficient.
    ///
    /// # Errors
    ///
    /// Returns an error if the description could not be retrieved or is not valid UTF-8.
    #[allow(clippy::cast_sign_loss)]
    pub fn desc(&self, buf_size: usize) -> Result<String, StringFromModelError> {
        let mut buf = vec![0u8; buf_size];
        let ret = unsafe {
            llama_model_desc(
                self.model.as_ptr(),
                buf.as_mut_ptr().cast::<c_char>(),
                buf_size,
            )
        };
        if ret < 0 {
            return Err(StringFromModelError::ReturnedError(ret));
        }
        let len = ret as usize;
        let s = std::str::from_utf8(&buf[..len])
            .map_err(StringFromModelError::Utf8Error)?;
        Ok(s.to_owned())
    }

    /// Get a metadata key by index.
    ///
    /// The `buf_size` parameter specifies the maximum buffer size for the key.
    /// A default of 256 bytes is usually sufficient.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of range or the key is not valid UTF-8.
    #[allow(clippy::cast_sign_loss)]
    pub fn meta_key_by_index(&self, index: i32, buf_size: usize) -> Result<String, StringFromModelError> {
        let mut buf = vec![0u8; buf_size];
        let ret = unsafe {
            llama_model_meta_key_by_index(
                self.model.as_ptr(),
                index,
                buf.as_mut_ptr().cast::<c_char>(),
                buf_size,
            )
        };
        if ret < 0 {
            return Err(StringFromModelError::ReturnedError(ret));
        }
        let len = ret as usize;
        let s = std::str::from_utf8(&buf[..len])
            .map_err(StringFromModelError::Utf8Error)?;
        Ok(s.to_owned())
    }

    /// Get a metadata value string by index.
    ///
    /// The `buf_size` parameter specifies the maximum buffer size for the value.
    /// Values can be large (e.g. chat templates, token lists), so 4096+ may be needed.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of range or the value is not valid UTF-8.
    #[allow(clippy::cast_sign_loss)]
    pub fn meta_val_str_by_index(&self, index: i32, buf_size: usize) -> Result<String, StringFromModelError> {
        let mut buf = vec![0u8; buf_size];
        let ret = unsafe {
            llama_model_meta_val_str_by_index(
                self.model.as_ptr(),
                index,
                buf.as_mut_ptr().cast::<c_char>(),
                buf_size,
            )
        };
        if ret < 0 {
            return Err(StringFromModelError::ReturnedError(ret));
        }
        let len = ret as usize;
        let s = std::str::from_utf8(&buf[..len])
            .map_err(StringFromModelError::Utf8Error)?;
        Ok(s.to_owned())
    }

    /// Get a metadata value by key name.
    ///
    /// This is more convenient than iterating metadata by index when you know the key.
    /// The `buf_size` parameter specifies the maximum buffer size for the value.
    ///
    /// # Errors
    ///
    /// Returns an error if the key is not found, contains a null byte, or the value is not valid UTF-8.
    #[allow(clippy::cast_sign_loss)]
    pub fn meta_val_str(&self, key: &str, buf_size: usize) -> Result<String, StringFromModelError> {
        let c_key = CString::new(key)
            .map_err(|_| StringFromModelError::ReturnedError(-1))?;
        let mut buf = vec![0u8; buf_size];
        let ret = unsafe {
            llama_model_meta_val_str(
                self.model.as_ptr(),
                c_key.as_ptr(),
                buf.as_mut_ptr().cast::<c_char>(),
                buf_size,
            )
        };
        if ret < 0 {
            return Err(StringFromModelError::ReturnedError(ret));
        }
        let len = ret as usize;
        let s = std::str::from_utf8(&buf[..len])
            .map_err(StringFromModelError::Utf8Error)?;
        Ok(s.to_owned())
    }

    /// Get all metadata as a list of `(key, value)` pairs.
    ///
    /// This is a convenience method that iterates over all metadata entries.
    /// Keys use a buffer of 256 bytes and values use 4096 bytes.
    /// For values that may be larger (e.g. token lists), use
    /// [`meta_val_str_by_index`](Self::meta_val_str_by_index) directly with a larger buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if any key or value cannot be read or is not valid UTF-8.
    #[allow(clippy::cast_sign_loss)]
    pub fn metadata(&self) -> Result<Vec<(String, String)>, StringFromModelError> {
        let count = self.meta_count();
        let mut result = Vec::with_capacity(count as usize);
        for i in 0..count {
            let key = self.meta_key_by_index(i, 256)?;
            let val = self.meta_val_str_by_index(i, 4096)?;
            result.push((key, val));
        }
        Ok(result)
    }

    /// Check if the model has an encoder.
    #[must_use]
    pub fn has_encoder(&self) -> bool {
        unsafe { llama_model_has_encoder(self.model.as_ptr()) }
    }

    /// Check if the model has a decoder.
    #[must_use]
    pub fn has_decoder(&self) -> bool {
        unsafe { llama_model_has_decoder(self.model.as_ptr()) }
    }

    /// Check if the model is recurrent (e.g. Mamba, RWKV).
    #[must_use]
    pub fn is_recurrent(&self) -> bool {
        unsafe { llama_model_is_recurrent(self.model.as_ptr()) }
    }

    /// Check if the model is a hybrid model.
    #[must_use]
    pub fn is_hybrid(&self) -> bool {
        unsafe { llama_model_is_hybrid(self.model.as_ptr()) }
    }

    /// Check if the model is a diffusion model.
    #[must_use]
    pub fn is_diffusion(&self) -> bool {
        unsafe { llama_model_is_diffusion(self.model.as_ptr()) }
    }

    /// Get chat template from model.
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
            llama_model_meta_val_str(self.model.as_ptr(), chat_name.as_ptr(), chat_ptr, buf_size)
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
        debug_assert!(
            Path::new(path).exists(),
            "{} does not exist",
            path.display()
        );
        let path = path
            .to_str()
            .ok_or(LlamaModelLoadError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;
        let llama_model = unsafe { llama_load_model_from_file(cstr.as_ptr(), params.params) };

        let model = NonNull::new(llama_model).ok_or(LlamaModelLoadError::NullResult)?;

        tracing::debug!(?path, "Loaded model");
        Ok(LlamaModel { model })
    }

    /// Load a model from multiple split files.
    ///
    /// This function loads a model that has been split across multiple files. This is useful for
    /// very large models that exceed filesystem limitations or need to be distributed across
    /// multiple storage devices.
    ///
    /// # Arguments
    ///
    /// * `paths` - A slice of paths to the split model files
    /// * `params` - The model parameters
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any of the paths cannot be converted to a C string
    /// - The model fails to load from the splits
    /// - Any path doesn't exist or isn't accessible
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_4::model::{LlamaModel, params::LlamaModelParams};
    /// use llama_cpp_4::llama_backend::LlamaBackend;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let backend = LlamaBackend::init()?;
    /// let params = LlamaModelParams::default();
    ///
    /// let paths = vec![
    ///     "model-00001-of-00003.gguf",
    ///     "model-00002-of-00003.gguf",
    ///     "model-00003-of-00003.gguf",
    /// ];
    ///
    /// let model = LlamaModel::load_from_splits(&backend, &paths, &params)?;
    /// # Ok(())
    /// # }
    /// ```
    #[tracing::instrument(skip_all)]
    pub fn load_from_splits(
        _: &LlamaBackend,
        paths: &[impl AsRef<Path>],
        params: &LlamaModelParams,
    ) -> Result<Self, LlamaModelLoadError> {
        // Convert paths to C strings
        let c_strings: Vec<CString> = paths
            .iter()
            .map(|p| {
                let path = p.as_ref();
                debug_assert!(path.exists(), "{} does not exist", path.display());
                let path_str = path
                    .to_str()
                    .ok_or(LlamaModelLoadError::PathToStrError(path.to_path_buf()))?;
                CString::new(path_str).map_err(LlamaModelLoadError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Create array of pointers to C strings
        let c_ptrs: Vec<*const c_char> = c_strings.iter().map(|s| s.as_ptr()).collect();

        // Load the model from splits
        let llama_model = unsafe {
            llama_model_load_from_splits(c_ptrs.as_ptr().cast_mut(), c_ptrs.len(), params.params)
        };

        let model = NonNull::new(llama_model).ok_or(LlamaModelLoadError::NullResult)?;

        tracing::debug!("Loaded model from {} splits", paths.len());
        Ok(LlamaModel { model })
    }

    /// Load a model from a `FILE` pointer.
    ///
    /// # Safety
    ///
    /// The `file` pointer must be a valid, open `FILE*`.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded.
    pub unsafe fn load_from_file_ptr(
        file: *mut llama_cpp_sys_4::FILE,
        params: &LlamaModelParams,
    ) -> Result<Self, LlamaModelLoadError> {
        let model = llama_cpp_sys_4::llama_model_load_from_file_ptr(file, params.params);
        let model = NonNull::new(model).ok_or(LlamaModelLoadError::NullResult)?;
        Ok(LlamaModel { model })
    }

    /// Initialize a model from user-provided data.
    ///
    /// # Safety
    ///
    /// The metadata, callback, and user data must be valid.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be initialized.
    pub unsafe fn init_from_user(
        metadata: *mut llama_cpp_sys_4::gguf_context,
        set_tensor_data: llama_cpp_sys_4::llama_model_set_tensor_data_t,
        set_tensor_data_ud: *mut std::ffi::c_void,
        params: &LlamaModelParams,
    ) -> Result<Self, LlamaModelLoadError> {
        let model = llama_cpp_sys_4::llama_model_init_from_user(
            metadata,
            set_tensor_data,
            set_tensor_data_ud,
            params.params,
        );
        let model = NonNull::new(model).ok_or(LlamaModelLoadError::NullResult)?;
        Ok(LlamaModel { model })
    }

    /// Save the model to a file.
    ///
    /// # Panics
    ///
    /// Panics if the path contains null bytes.
    pub fn save_to_file(&self, path: impl AsRef<Path>) {
        let path = path.as_ref();
        let path_str = path.to_str().expect("path is not valid UTF-8");
        let c_path = CString::new(path_str).expect("path contains null bytes");
        unsafe {
            llama_model_save_to_file(self.model.as_ptr(), c_path.as_ptr());
        }
    }

    /// Get the list of built-in chat templates.
    ///
    /// Returns the names of all chat templates that are built into llama.cpp.
    ///
    /// # Panics
    ///
    /// Panics if any template name is not valid UTF-8.
    #[allow(clippy::cast_sign_loss)]
    #[must_use]
    pub fn chat_builtin_templates() -> Vec<String> {
        // First call to get count
        let count = unsafe { llama_chat_builtin_templates(std::ptr::null_mut(), 0) };
        if count <= 0 {
            return Vec::new();
        }
        let count = count as usize;
        let mut ptrs: Vec<*const c_char> = vec![std::ptr::null(); count];
        unsafe {
            llama_chat_builtin_templates(ptrs.as_mut_ptr(), count);
        }
        ptrs.iter()
            .map(|&p| {
                let cstr = unsafe { CStr::from_ptr(p) };
                cstr.to_str()
                    .expect("template name is not valid UTF-8")
                    .to_owned()
            })
            .collect()
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
        debug_assert!(
            Path::new(path).exists(),
            "{} does not exist",
            path.display()
        );

        let path = path
            .to_str()
            .ok_or(LlamaLoraAdapterInitError::PathToStrError(
                path.to_path_buf(),
            ))?;

        let cstr = CString::new(path)?;
        let adapter = unsafe { llama_adapter_lora_init(self.model.as_ptr(), cstr.as_ptr()) };

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
    ) -> Result<LlamaContext<'_>, LlamaContextLoadError> {
        let context_params = params.context_params;
        let context = unsafe { llama_new_context_with_model(self.model.as_ptr(), context_params) };
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
    /// <https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template>
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
    /// # Panics
    ///
    /// Panics if the buffer length exceeds `i32::MAX`.
    #[tracing::instrument(skip_all)]
    pub fn apply_chat_template(
        &self,
        tmpl: Option<&str>,
        chat: &[LlamaChatMessage],
        add_ass: bool,
    ) -> Result<String, ApplyChatTemplateError> {
        // Compute raw message byte total from the original LlamaChatMessage vec
        // *before* we shadow `chat` with the sys-type vec below.
        let message_length = chat.iter().fold(0usize, |acc, c| {
            acc + c.role.to_bytes().len() + c.content.to_bytes().len()
        });

        // Build our llama_cpp_sys chat messages (raw pointers into CStrings).
        let chat_sys: Vec<llama_chat_message> = chat
            .iter()
            .map(|c| llama_chat_message {
                role: c.role.as_ptr(),
                content: c.content.as_ptr(),
            })
            .collect();

        // Set the tmpl pointer.
        let tmpl_cstring = tmpl.map(CString::new).transpose()?;
        let tmpl_ptr = tmpl_cstring
            .as_ref()
            .map_or(std::ptr::null(), |s| s.as_ptr());

        // `message_length * 4` is far too small for models whose built-in chat
        // template adds a long default system prompt (e.g. Qwen3.5 prepends
        // ~80+ chars of markup even for a one-word user message).  Start with
        // at least 4 KiB so short inputs like "hi" always have room.
        //
        // `llama_chat_apply_template` returns the number of bytes it *actually*
        // needed when the buffer was too small, so we retry exactly once with
        // that precise size rather than giving up immediately.
        let mut buf_size = message_length.saturating_mul(4).max(4096);

        for _ in 0..2 {
            // Use u8 so that as_mut_ptr()/as_ptr() match the binding (*mut u8 / *const u8).
            let mut buff = vec![0u8; buf_size];
            let res = unsafe {
                llama_chat_apply_template(
                    tmpl_ptr,
                    chat_sys.as_ptr(),
                    chat_sys.len(),
                    add_ass,
                    buff.as_mut_ptr().cast(),
                    i32::try_from(buff.len()).expect("buffer length fits in i32"),
                )
            };

            if res < 0 {
                return Err(ApplyChatTemplateError::BuffSizeError);
            }

            #[allow(clippy::cast_sign_loss)]
            let needed = res as usize;
            if needed > buf_size {
                // Buffer was too small — retry with the exact size llama.cpp reported.
                buf_size = needed + 1; // +1 for null terminator
                continue;
            }

            // SAFETY: llama_chat_apply_template wrote a NUL-terminated string
            // into `buff`; `needed` bytes were used.
            let formatted = unsafe {
                CStr::from_ptr(buff.as_ptr().cast())
                    .to_string_lossy()
                    .into_owned()
            };
            return Ok(formatted);
        }

        Err(ApplyChatTemplateError::BuffSizeError)
    }

    /// Build a split GGUF file path for a specific chunk.
    ///
    /// This utility function creates the standardized filename for a split model chunk
    /// following the pattern: `{prefix}-{split_no:05d}-of-{split_count:05d}.gguf`
    ///
    /// # Arguments
    ///
    /// * `path_prefix` - The base path and filename prefix
    /// * `split_no` - The split number (1-indexed)
    /// * `split_count` - The total number of splits
    ///
    /// # Returns
    ///
    /// Returns the formatted split path as a String
    ///
    /// # Example
    ///
    /// ```
    /// use llama_cpp_4::model::LlamaModel;
    ///
    /// let path = LlamaModel::split_path("/models/llama", 2, 4);
    /// assert_eq!(path, "/models/llama-00002-of-00004.gguf");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the path prefix contains a null byte.
    #[must_use]
    pub fn split_path(path_prefix: &str, split_no: i32, split_count: i32) -> String {
        let mut buffer = vec![0u8; 1024];
        let len = unsafe {
            llama_split_path(
                buffer.as_mut_ptr().cast::<c_char>(),
                buffer.len(),
                CString::new(path_prefix).unwrap().as_ptr(),
                split_no,
                split_count,
            )
        };

        let len = usize::try_from(len).expect("split_path length fits in usize");
        buffer.truncate(len);
        String::from_utf8(buffer).unwrap_or_default()
    }

    /// Extract the path prefix from a split filename.
    ///
    /// This function extracts the base path prefix from a split model filename,
    /// but only if the `split_no` and `split_count` match the pattern in the filename.
    ///
    /// # Arguments
    ///
    /// * `split_path` - The full path to the split file
    /// * `split_no` - The expected split number
    /// * `split_count` - The expected total number of splits
    ///
    /// # Returns
    ///
    /// Returns the path prefix if the pattern matches, or None if it doesn't
    ///
    /// # Example
    ///
    /// ```
    /// use llama_cpp_4::model::LlamaModel;
    ///
    /// let prefix = LlamaModel::split_prefix("/models/llama-00002-of-00004.gguf", 2, 4);
    /// assert_eq!(prefix, Some("/models/llama".to_string()));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the split path contains a null byte.
    #[must_use]
    pub fn split_prefix(split_path: &str, split_no: i32, split_count: i32) -> Option<String> {
        let mut buffer = vec![0u8; 1024];
        let len = unsafe {
            llama_split_prefix(
                buffer.as_mut_ptr().cast::<c_char>(),
                buffer.len(),
                CString::new(split_path).unwrap().as_ptr(),
                split_no,
                split_count,
            )
        };

        if len > 0 {
            let len = usize::try_from(len).expect("split_prefix length fits in usize");
            buffer.truncate(len);
            String::from_utf8(buffer).ok()
        } else {
            None
        }
    }
}

#[allow(clippy::cast_precision_loss)]
impl fmt::Display for LlamaModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let desc = self.desc(256).unwrap_or_else(|_| "unknown".to_string());
        write!(
            f,
            "{desc} | {layers}L {heads}H {embd}E | {params} params | {size:.1} MiB",
            layers = self.n_layer(),
            heads = self.n_head(),
            embd = self.n_embd(),
            params = self.n_params(),
            size = self.model_size() as f64 / (1024.0 * 1024.0),
        )
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe { llama_free_model(self.model.as_ptr()) }
    }
}

/// Defines the possible types of vocabulary used by the model.
///
/// The model may use different types of vocabulary depending on the tokenization method chosen during training.
/// This enum represents these types, specifically `BPE` (Byte Pair Encoding) and `SPM` (`SentencePiece`).
///
/// # Variants
///
/// - `BPE`: Byte Pair Encoding, a common tokenization method used in NLP tasks.
/// - `SPM`: `SentencePiece`, another popular tokenization method for NLP models.
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
    BPE = LLAMA_VOCAB_TYPE_BPE as _,
    /// Sentence Piece Tokenizer
    SPM = LLAMA_VOCAB_TYPE_SPM as _,
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
    UnknownValue(llama_vocab_type),
}

impl TryFrom<llama_vocab_type> for VocabType {
    type Error = LlamaTokenTypeFromIntError;

    fn try_from(value: llama_vocab_type) -> Result<Self, Self::Error> {
        match value {
            LLAMA_VOCAB_TYPE_BPE => Ok(VocabType::BPE),
            LLAMA_VOCAB_TYPE_SPM => Ok(VocabType::SPM),
            unknown => Err(LlamaTokenTypeFromIntError::UnknownValue(unknown)),
        }
    }
}
