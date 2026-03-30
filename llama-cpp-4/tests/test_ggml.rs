#![cfg(feature = "ggml")]
//! Tests for the ggml graph computation API.

use llama_cpp_4::ggml::*;

#[test]
fn test_context_creation() {
    let ctx = GgmlContext::new(1024 * 1024, true);
    assert!(!ctx.as_ptr().is_null());
}

#[test]
fn test_tensor_creation_1d() {
    let ctx = GgmlContext::new(1024 * 1024, true);
    let t = ctx.new_tensor_1d(llama_cpp_sys_4::GGML_TYPE_F32, 8);
    assert_eq!(t.nelements(), 8);
    assert_eq!(t.ne(), [8, 1, 1, 1]);
}

#[test]
fn test_tensor_creation_2d() {
    let ctx = GgmlContext::new(1024 * 1024, true);
    let t = ctx.new_tensor_2d(llama_cpp_sys_4::GGML_TYPE_F32, 4, 3);
    assert_eq!(t.nelements(), 12);
    assert_eq!(t.ne()[0], 4);
    assert_eq!(t.ne()[1], 3);
}

#[test]
fn test_tensor_name() {
    let ctx = GgmlContext::new(1024 * 1024, true);
    let t = ctx.new_tensor_1d(llama_cpp_sys_4::GGML_TYPE_F32, 4);
    t.set_name("my_tensor");
    assert_eq!(t.name(), "my_tensor");
}

#[test]
fn test_dup_tensor() {
    let ctx = GgmlContext::new(1024 * 1024, true);
    let t = ctx.new_tensor_2d(llama_cpp_sys_4::GGML_TYPE_F32, 4, 3);
    let dup = ctx.dup_tensor(&t);
    assert_eq!(dup.ne(), t.ne());
    assert_eq!(dup.typ(), t.typ());
}

#[test]
fn test_type_name() {
    let name = type_name(llama_cpp_sys_4::GGML_TYPE_F32);
    assert_eq!(name, "f32");
    let name = type_name(llama_cpp_sys_4::GGML_TYPE_F16);
    assert_eq!(name, "f16");
}

#[test]
fn test_is_quantized() {
    assert!(!is_quantized(llama_cpp_sys_4::GGML_TYPE_F32));
    assert!(!is_quantized(llama_cpp_sys_4::GGML_TYPE_F16));
    assert!(is_quantized(llama_cpp_sys_4::GGML_TYPE_Q4_0));
    assert!(is_quantized(llama_cpp_sys_4::GGML_TYPE_Q8_0));
}

#[test]
fn test_tensor_overhead() {
    let overhead = tensor_overhead();
    assert!(overhead > 0);
}

#[test]
fn test_graph_overhead() {
    let overhead = graph_overhead();
    assert!(overhead > 0);
}

#[test]
fn test_backend_cpu() {
    let backend = GgmlBackend::cpu();
    assert!(!backend.as_ptr().is_null());
}

#[test]
fn test_allocr_creation() {
    let backend = GgmlBackend::cpu();
    let _alloc = GgmlAllocr::new(&backend);
}

#[test]
fn test_graph_add_two_vectors() {
    // Create context for graph building (no_alloc = true)
    let n = 4_i64;
    let n_tensors = 3; // a, b, sum
    let mem_size = tensor_overhead() * n_tensors + graph_overhead() + 1024;
    let ctx = GgmlContext::new(mem_size, true);

    // Create input tensors
    let a = ctx.new_tensor_1d(llama_cpp_sys_4::GGML_TYPE_F32, n);
    a.set_name("a");
    let b = ctx.new_tensor_1d(llama_cpp_sys_4::GGML_TYPE_F32, n);
    b.set_name("b");

    // Build graph: sum = a + b
    let sum = ctx.add(&a, &b);
    let mut graph = ctx.new_graph();
    graph.build_forward(&sum);

    // Allocate with backend
    let backend = GgmlBackend::cpu();
    let alloc = GgmlAllocr::new(&backend);
    assert!(alloc.alloc_graph(&mut graph));

    // Set input data
    let a_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let b_data: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
    unsafe {
        tensor_set(&a, bytemuck_cast(&a_data));
        tensor_set(&b, bytemuck_cast(&b_data));
    }

    // Compute
    backend.graph_compute(&mut graph);

    // Read result
    let result_tensor = graph.node(-1);
    let mut result_data = [0.0_f32; 4];
    unsafe {
        tensor_get(&result_tensor, bytemuck_cast_mut(&mut result_data));
    }

    assert_eq!(result_data, [11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn test_graph_scale() {
    let n = 3_i64;
    let mem_size = tensor_overhead() * 2 + graph_overhead() + 1024;
    let ctx = GgmlContext::new(mem_size, true);

    let a = ctx.new_tensor_1d(llama_cpp_sys_4::GGML_TYPE_F32, n);
    let scaled = ctx.scale(&a, 2.5);
    let mut graph = ctx.new_graph();
    graph.build_forward(&scaled);

    let backend = GgmlBackend::cpu();
    let alloc = GgmlAllocr::new(&backend);
    alloc.alloc_graph(&mut graph);

    let a_data: [f32; 3] = [1.0, 2.0, 4.0];
    unsafe { tensor_set(&a, bytemuck_cast(&a_data)) };

    backend.graph_compute(&mut graph);

    let mut result = [0.0_f32; 3];
    unsafe { tensor_get(&graph.node(-1), bytemuck_cast_mut(&mut result)) };
    assert_eq!(result, [2.5, 5.0, 10.0]);
}

#[test]
fn test_graph_matmul() {
    // 2x2 @ 2x2
    let mem_size = tensor_overhead() * 3 + graph_overhead() + 1024;
    let ctx = GgmlContext::new(mem_size, true);

    let a = ctx.new_tensor_2d(llama_cpp_sys_4::GGML_TYPE_F32, 2, 2);
    let b = ctx.new_tensor_2d(llama_cpp_sys_4::GGML_TYPE_F32, 2, 2);
    let c = ctx.mul_mat(&a, &b);
    let mut graph = ctx.new_graph();
    graph.build_forward(&c);

    let backend = GgmlBackend::cpu();
    let alloc = GgmlAllocr::new(&backend);
    alloc.alloc_graph(&mut graph);

    // identity matrix
    let a_data: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    let b_data: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
    unsafe {
        tensor_set(&a, bytemuck_cast(&a_data));
        tensor_set(&b, bytemuck_cast(&b_data));
    }

    backend.graph_compute(&mut graph);

    let mut result = [0.0_f32; 4];
    unsafe { tensor_get(&graph.node(-1), bytemuck_cast_mut(&mut result)) };

    // identity @ B = B
    assert_eq!(result, [5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_tensor_iteration() {
    let mem_size = tensor_overhead() * 10 + 1024;
    let ctx = GgmlContext::new(mem_size, true);
    ctx.new_tensor_1d(llama_cpp_sys_4::GGML_TYPE_F32, 4).set_name("t1");
    ctx.new_tensor_1d(llama_cpp_sys_4::GGML_TYPE_F32, 8).set_name("t2");
    ctx.new_tensor_2d(llama_cpp_sys_4::GGML_TYPE_F16, 2, 3).set_name("t3");

    let mut names = Vec::new();
    let mut t = ctx.first_tensor();
    while let Some(tensor) = t {
        names.push(tensor.name().to_string());
        t = ctx.next_tensor(&tensor);
    }
    assert_eq!(names, vec!["t1", "t2", "t3"]);
}

// Helper to cast &[f32] to &[u8] without pulling in bytemuck
fn bytemuck_cast(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), data.len() * 4) }
}

fn bytemuck_cast_mut(data: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), data.len() * 4) }
}
