//! Verify that [`llama_cpp_4::prelude`] re-exports compile and name-check.

use llama_cpp_4::prelude::*;

#[test]
fn prelude_core_types_are_in_scope() {
    fn assert_send<T: Send>() {}
    assert_send::<LlamaBackend>();
    assert_send::<LlamaModel>();
    // Context parameters may own callback and sampler state whose native
    // pointers are deliberately confined to the constructing thread.
    let _ = LlamaContextParams::default();
    assert_send::<TensorCapture>();
}

#[test]
fn prelude_device_types_are_in_scope() {
    let _: LlamaBackendDeviceType = LlamaBackendDeviceType::Cpu;
}

#[test]
fn prelude_helper_types_are_in_scope() {
    let _ = ParamOverrideValue::Bool(true);
    let _: TensorTypeOverride = TensorTypeOverride::new("output", GgmlType::F16).unwrap();
    let _ = LlamaTokenDataArray::new(vec![], false);
    let _ = ParamsCloneError::SamplerChains;
    let _ = FitParamsError::InvalidPath;
    let _ = BatchAddError::InsufficientSpace(0);
}

#[test]
fn crate_root_reexports_match_prelude() {
    let _: fn() -> llama_cpp_4::Result<LlamaBackend> = || LlamaBackend::init();
}
