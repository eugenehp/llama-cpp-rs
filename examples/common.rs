//! example of common sampler params usage
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]

/// example of common params
pub fn main() {
 let params = llama_cpp_sys_4::common::common_sampler_params::default();
 println!("common_sampler_params {:?}",params);
}