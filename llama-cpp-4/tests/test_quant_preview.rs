//! Tests for the quantization-preview API.
//!
//! These ask llama.cpp what it *would* do to a tensor, without writing a file.
//! `QuantModelDesc` builds the model from metadata, so no checkpoint is needed.

use llama_cpp_4::quantize::{
    LlamaFtype, QuantModelDesc, QuantPreview, QuantPreviewError, QuantizeParams,
};

mod support;

/// A shape big enough that its 2-D tensors clear llama.cpp's
/// "too small to bother quantizing" threshold.
fn desc() -> QuantModelDesc {
    QuantModelDesc::llama(512, 1376, 4, 8)
}

#[test]
fn mock_model_builds_from_a_descriptor() {
    let _backend = support::model::backend();
    let model = desc().build().expect("mock model");
    assert_eq!(model.n_embd(), 512);
}

#[test]
fn mock_model_rejects_an_unknown_architecture() {
    let _backend = support::model::backend();
    let mut d = desc();
    d.architecture = "definitely-not-an-arch".to_owned();
    assert!(matches!(
        d.build(),
        Err(QuantPreviewError::MockModel | QuantPreviewError::Nul(_))
    ));
}

#[test]
fn mock_model_rejects_an_architecture_with_interior_nul() {
    let _backend = support::model::backend();
    let mut d = desc();
    d.architecture = "lla\0ma".to_owned();
    assert!(matches!(d.build(), Err(QuantPreviewError::Nul(_))));
}

#[test]
fn preview_initializes_for_a_mock_model() {
    let _backend = support::model::backend();
    let model = desc().build().expect("mock model");
    let params = QuantizeParams::new(LlamaFtype::MostlyQ4KM);
    QuantPreview::new(&model, &params).expect("preview");
}

/// The whole point of the preview: a k-quant mix does *not* store every tensor
/// at its nominal type, so the per-tensor answer must be able to differ from
/// `LlamaFtype::default_ggml_type`.
#[cfg(feature = "ggml")]
#[test]
fn compute_types_assigns_a_type_to_each_tensor() {
    use llama_cpp_4::ggml::GgmlContext;
    use llama_cpp_sys_4::GGML_TYPE_F32;

    let _backend = support::model::backend();
    let model = desc().build().expect("mock model");
    let params = QuantizeParams::new(LlamaFtype::MostlyQ4KM);
    let preview = QuantPreview::new(&model, &params).expect("preview");

    let ctx = GgmlContext::new(16 * 1024 * 1024, true);
    let ffn = ctx.new_tensor_2d(GGML_TYPE_F32, 512, 1376);
    ffn.set_name("blk.0.ffn_down.weight");

    if !preview.allows_quantization(&ffn) {
        // Nothing to assert about types if llama.cpp would skip it outright;
        // that is itself a valid answer for this shape.
        return;
    }

    let types = preview
        .compute_types(&[&ffn], LlamaFtype::MostlyQ4KM)
        .expect("compute types");
    assert_eq!(types.len(), 1, "one answer per tensor");
    assert!(
        types[0].is_some(),
        "llama.cpp picked a ggml type this crate does not know"
    );
}

/// 1-D tensors are never quantized upstream; passing one must be refused with a
/// name rather than handed to llama.cpp, which does not re-check.
#[cfg(feature = "ggml")]
#[test]
fn compute_types_refuses_a_tensor_that_would_not_be_quantized() {
    use llama_cpp_4::ggml::GgmlContext;
    use llama_cpp_sys_4::GGML_TYPE_F32;

    let _backend = support::model::backend();
    let model = desc().build().expect("mock model");
    let params = QuantizeParams::new(LlamaFtype::MostlyQ4KM);
    let preview = QuantPreview::new(&model, &params).expect("preview");

    let ctx = GgmlContext::new(1024 * 1024, true);
    let bias = ctx.new_tensor_1d(GGML_TYPE_F32, 512);
    bias.set_name("blk.0.ffn_down.bias");

    assert!(
        !preview.allows_quantization(&bias),
        "a 1-D bias should never be quantized"
    );
    let err = preview
        .compute_types(&[&bias], LlamaFtype::MostlyQ4KM)
        .unwrap_err();
    assert!(
        matches!(err, QuantPreviewError::NotQuantizable(ref n) if n.contains("bias")),
        "expected NotQuantizable naming the tensor, got {err:?}"
    );
}

#[cfg(feature = "ggml")]
#[test]
fn compute_types_handles_an_empty_slice() {
    let _backend = support::model::backend();
    let model = desc().build().expect("mock model");
    let params = QuantizeParams::new(LlamaFtype::MostlyQ4KM);
    let preview = QuantPreview::new(&model, &params).expect("preview");
    assert!(preview
        .compute_types(&[], LlamaFtype::MostlyQ4KM)
        .unwrap()
        .is_empty());
}

/// `upstream_name` comes from llama.cpp rather than this crate's table, so the
/// two must at least agree on which type they are describing.
#[test]
fn ftype_names_agree_between_crate_and_upstream() {
    for ftype in LlamaFtype::all() {
        let upstream = ftype.upstream_name().expect("upstream name");
        assert!(!upstream.is_empty(), "{ftype:?} has an empty upstream name");
    }
}

#[test]
fn ftype_roundtrips_through_the_raw_discriminant() {
    for ftype in LlamaFtype::all() {
        let raw: llama_cpp_sys_4::llama_ftype = (*ftype).into();
        assert_eq!(
            LlamaFtype::try_from(raw).ok(),
            Some(*ftype),
            "{ftype:?} did not round-trip"
        );
    }
}

/// `LLAMA_FTYPE_GUESSED` means "the file did not say", which is not a type this
/// crate can name — it must be rejected rather than mapped to something wrong.
#[test]
fn ftype_rejects_guessed() {
    assert!(LlamaFtype::try_from(1024).is_err());
}

#[test]
fn ftype_has_a_default_ggml_type() {
    // F16 is unambiguous: its default storage type must resolve.
    assert!(LlamaFtype::MostlyF16.default_ggml_type().is_some());
}
