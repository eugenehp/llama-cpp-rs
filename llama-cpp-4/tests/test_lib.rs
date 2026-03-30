//! Tests for top-level library functions that don't require a model.

#[test]
fn test_print_system_info() {
    let info = llama_cpp_4::print_system_info();
    assert!(!info.is_empty(), "system info should not be empty");
    // Should contain CPU feature flags
    assert!(info.contains('='), "system info should contain feature flags");
}

#[test]
fn test_supports_gpu_offload() {
    // Just ensure it doesn't panic
    let _ = llama_cpp_4::supports_gpu_offload();
}

#[test]
fn test_supports_rpc() {
    let _ = llama_cpp_4::supports_rpc();
}

#[test]
fn test_max_devices() {
    let devices = llama_cpp_4::max_devices();
    assert!(devices > 0, "max_devices should be > 0");
}

#[test]
fn test_max_parallel_sequences() {
    let max_seq = llama_cpp_4::max_parallel_sequences();
    assert!(max_seq > 0);
}

#[test]
fn test_max_tensor_buft_overrides() {
    let max = llama_cpp_4::max_tensor_buft_overrides();
    assert!(max > 0);
}

#[test]
fn test_mmap_supported() {
    // Just ensure it doesn't panic
    let _ = llama_cpp_4::mmap_supported();
}

#[test]
fn test_mlock_supported() {
    let _ = llama_cpp_4::mlock_supported();
}

#[test]
fn test_llama_supports_mlock() {
    let _ = llama_cpp_4::llama_supports_mlock();
}

#[test]
fn test_flash_attn_type_name() {
    let name = llama_cpp_4::flash_attn_type_name(0);
    assert!(!name.is_empty());
}

#[test]
fn test_model_meta_key_str() {
    let key = llama_cpp_4::model_meta_key_str(0);
    assert!(!key.is_empty());
    assert!(key.contains("sampling"), "key 0 should be a sampling key, got: {key}");
}

#[test]
fn test_model_quantize_default_params() {
    let params = llama_cpp_4::model_quantize_default_params();
    // nthread=0 means auto
    assert!(params.nthread >= 0);
}

#[test]
fn test_llama_time_us() {
    let t1 = llama_cpp_4::llama_time_us();
    let t2 = llama_cpp_4::llama_time_us();
    assert!(t2 >= t1, "time should be monotonic");
}

#[test]
fn test_ggml_time_us() {
    let t1 = llama_cpp_4::ggml_time_us();
    let t2 = llama_cpp_4::ggml_time_us();
    assert!(t2 >= t1);
}
