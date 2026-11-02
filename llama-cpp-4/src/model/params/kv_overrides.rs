//! Key-value overrides for a model.

use crate::model::params::LlamaModelParams;
use std::ffi::{CStr, CString};
use std::fmt::Debug;

/// An override value for a model parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamOverrideValue {
    /// A string value
    Bool(bool),
    /// A float value
    Float(f64),
    /// A integer value
    Int(i64),
    /// A string value
    Str([std::os::raw::c_char; 128]),
}

impl ParamOverrideValue {
    pub(crate) fn tag(&self) -> llama_cpp_sys_4::llama_model_kv_override_type {
        match self {
            ParamOverrideValue::Bool(_) => llama_cpp_sys_4::LLAMA_KV_OVERRIDE_TYPE_BOOL,
            ParamOverrideValue::Float(_) => llama_cpp_sys_4::LLAMA_KV_OVERRIDE_TYPE_FLOAT,
            ParamOverrideValue::Int(_) => llama_cpp_sys_4::LLAMA_KV_OVERRIDE_TYPE_INT,
            ParamOverrideValue::Str(_) => llama_cpp_sys_4::LLAMA_KV_OVERRIDE_TYPE_STR,
        }
    }

    pub(crate) fn value(&self) -> llama_cpp_sys_4::llama_model_kv_override__bindgen_ty_1 {
        match self {
            ParamOverrideValue::Bool(value) => {
                llama_cpp_sys_4::llama_model_kv_override__bindgen_ty_1 { val_bool: *value }
            }
            ParamOverrideValue::Float(value) => {
                llama_cpp_sys_4::llama_model_kv_override__bindgen_ty_1 { val_f64: *value }
            }
            ParamOverrideValue::Int(value) => {
                llama_cpp_sys_4::llama_model_kv_override__bindgen_ty_1 { val_i64: *value }
            }
            ParamOverrideValue::Str(c_string) => {
                llama_cpp_sys_4::llama_model_kv_override__bindgen_ty_1 { val_str: *c_string }
            }
        }
    }
}

impl From<&llama_cpp_sys_4::llama_model_kv_override> for ParamOverrideValue {
    fn from(
        llama_cpp_sys_4::llama_model_kv_override {
            key: _,
            tag,
            __bindgen_anon_1,
        }: &llama_cpp_sys_4::llama_model_kv_override,
    ) -> Self {
        // SAFETY (all four reads below): `llama_model_kv_override` is a tagged
        // union — `tag` says which arm of `__bindgen_anon_1` is live, and each
        // match arm reads exactly the field its tag names. Every variant is a
        // plain scalar or a fixed `c_char` array, so none carries a pointer
        // that could dangle.
        match *tag {
            llama_cpp_sys_4::LLAMA_KV_OVERRIDE_TYPE_INT => {
                // SAFETY: the tag matched above says this is the live union field.
                ParamOverrideValue::Int(unsafe { __bindgen_anon_1.val_i64 })
            }
            llama_cpp_sys_4::LLAMA_KV_OVERRIDE_TYPE_FLOAT => {
                // SAFETY: the tag matched above says this is the live union field.
                ParamOverrideValue::Float(unsafe { __bindgen_anon_1.val_f64 })
            }
            llama_cpp_sys_4::LLAMA_KV_OVERRIDE_TYPE_BOOL => {
                // SAFETY: the tag matched above says this is the live union field.
                ParamOverrideValue::Bool(unsafe { __bindgen_anon_1.val_bool })
            }
            llama_cpp_sys_4::LLAMA_KV_OVERRIDE_TYPE_STR => {
                // SAFETY: the tag matched above says this is the live union field.
                ParamOverrideValue::Str(unsafe { __bindgen_anon_1.val_str })
            }
            _ => unreachable!("Unknown tag of {tag}"),
        }
    }
}

/// A struct implementing [`IntoIterator`] over the key-value overrides for a model.
#[derive(Debug)]
pub struct KvOverrides<'a> {
    model_params: &'a LlamaModelParams,
}

impl KvOverrides<'_> {
    pub(super) fn new(model_params: &LlamaModelParams) -> KvOverrides<'_> {
        KvOverrides { model_params }
    }
}

impl<'a> IntoIterator for KvOverrides<'a> {
    // I'm fairly certain this could be written returning by reference, but I'm not sure how to do it safely. I do not
    // expect this to be a performance bottleneck so the copy should be fine. (let me know if it's not fine!)
    type Item = (CString, ParamOverrideValue);
    type IntoIter = KvOverrideValueIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        KvOverrideValueIterator {
            model_params: self.model_params,
            current: 0,
        }
    }
}

/// An iterator over the key-value overrides for a model.
#[derive(Debug)]
pub struct KvOverrideValueIterator<'a> {
    model_params: &'a LlamaModelParams,
    current: usize,
}

impl Iterator for KvOverrideValueIterator<'_> {
    type Item = (CString, ParamOverrideValue);

    fn next(&mut self) -> Option<Self::Item> {
        let overrides = self.model_params.params.kv_overrides;
        if overrides.is_null() {
            return None;
        }

        // SAFETY: llama.cpp seems to guarantee that the last element contains an empty key or is valid. We've checked
        // the prev one in the last iteration, the next one should be valid or 0 (and thus safe to deref)
        let current = unsafe { *overrides.add(self.current) };

        if current.key[0] == 0 {
            return None;
        }

        let value = ParamOverrideValue::from(&current);

        // SAFETY: `key` is a fixed-size `c_char` array inside the override
        // struct, and the non-empty check above means it holds at least one
        // byte; llama.cpp always NUL-terminates it. `to_owned` copies out
        // before the borrow ends.
        let key = unsafe { CStr::from_ptr(current.key.as_ptr()).to_owned() };

        self.current += 1;
        Some((key, value))
    }
}
