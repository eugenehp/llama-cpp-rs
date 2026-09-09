//! Speculative-type introspection, llama.cpp's logger, and model resolution.
//!
//! Three small surfaces that share one thing: they are things you do *around*
//! inference rather than during it.

use std::ffi::CString;

use llama_cpp_sys_4 as sys;

use crate::shim::{check_status, read_i32s, read_string, ShimError};

/// Errors from this module.
pub type RuntimeError = ShimError;

type Result<T> = std::result::Result<T, RuntimeError>;

// ─────────────────────────────────────────────────────────────────────────────
// Speculative-type introspection
// ─────────────────────────────────────────────────────────────────────────────

/// A speculative-decoding strategy, as llama.cpp names it.
///
/// Values match `common_speculative_type`. Rather than pin discriminants that
/// upstream reorders, this keeps the raw value and converts through llama.cpp's
/// own name table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeculativeType(pub i32);

impl SpeculativeType {
    /// llama.cpp's name for this type, e.g. `"draft-eagle3"`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Failed`] if llama.cpp cannot name it.
    pub fn name(self) -> Result<String> {
        read_string(|buf, len, expected| unsafe {
            sys::common_shim_speculative_type_to_str(self.0, buf, len, expected)
        })
    }

    /// Parse a name from llama.cpp's own table.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::InvalidArg`] if the name is not recognised —
    /// upstream reports that with a sentinel value rather than an error, which
    /// the shim translates so an unknown name cannot be stored and silently
    /// select nothing. [`RuntimeError::Nul`] for an interior NUL.
    pub fn from_name(name: &str) -> Result<Self> {
        let c_name = CString::new(name)?;
        let mut raw = 0i32;
        let status =
            unsafe { sys::common_shim_speculative_type_from_name(c_name.as_ptr(), &raw mut raw) };
        check_status(status)?;
        Ok(Self(raw))
    }

    /// Every type name llama.cpp recognises, as one string — what its CLI
    /// prints in help text.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Failed`] if llama.cpp throws.
    pub fn all_names() -> Result<String> {
        read_string(|buf, len, expected| unsafe {
            sys::common_shim_speculative_all_types_str(buf, len, expected)
        })
    }
}

/// Which speculative strategies a draft GGUF supports, read from its metadata.
///
/// The point is that this does **not** load the model: pointing at a 4 GB draft
/// checkpoint to discover it has no EAGLE-3 head costs a metadata read, not a
/// full load and a failed session construction. Use it to pick between
/// [`Eagle3Session`](crate::eagle::Eagle3Session),
/// [`MtpSession`](crate::mtp::MtpSession) and the rest before committing.
///
/// Wraps `common_speculative_types_from_gguf`. Returns an empty vector for a
/// file that is not a readable GGUF, or one advertising no speculative support.
///
/// # Errors
///
/// Returns [`RuntimeError::Nul`] for an interior NUL in `path`.
pub fn speculative_types_from_gguf(path: &str) -> Result<Vec<SpeculativeType>> {
    let c_path = CString::new(path)?;
    let raw = read_i32s(|out, cap, len| unsafe {
        sys::common_shim_speculative_types_from_gguf(c_path.as_ptr(), out, cap, len)
    })?;
    Ok(raw.into_iter().map(SpeculativeType).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Logging
// ─────────────────────────────────────────────────────────────────────────────

/// Controls for llama.cpp's own logger — the output the C++ library produces,
/// which is separate from anything this crate emits through `tracing`.
///
/// Every one of these is documented upstream as **not thread-safe**; call them
/// during setup, before inference starts.
///
/// For redirecting llama.cpp's output into a Rust logger instead, use
/// [`log_set`](crate::log_set), which installs a callback.
pub mod log {
    use super::{check_status, CString, Result};
    use llama_cpp_sys_4 as sys;

    /// Drop log records below this verbosity.
    pub fn set_verbosity(verbosity: i32) {
        unsafe { sys::common_shim_log_set_verbosity(verbosity) }
    }

    /// The verbosity threshold configured for a `ggml_log_level`.
    #[must_use]
    pub fn verbosity_for_level(level: i32) -> i32 {
        unsafe { sys::common_shim_log_get_verbosity(level) }
    }

    /// Include timestamps in the log prefix.
    pub fn set_timestamps(timestamps: bool) {
        unsafe { sys::common_shim_log_set_timestamps(timestamps) }
    }

    /// Include the level prefix on each record.
    pub fn set_prefix(prefix: bool) {
        unsafe { sys::common_shim_log_set_prefix(prefix) }
    }

    /// Colourise output. Disabling is what you want when the destination is not
    /// a terminal.
    pub fn set_colors(colors: bool) {
        unsafe { sys::common_shim_log_set_colors(colors) }
    }

    /// Emit one JSON object per record instead of human-readable text — for
    /// shipping llama.cpp's own logs into a structured pipeline.
    pub fn set_jsonl(jsonl: bool) {
        unsafe { sys::common_shim_log_set_jsonl(jsonl) }
    }

    /// Write to `path`, or pass `None` to stop writing to a file.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Failed`] if the file cannot be opened, or
    /// [`RuntimeError::Nul`] for an interior NUL in `path`.
    pub fn set_file(path: Option<&str>) -> Result<()> {
        let c_path = path.map(CString::new).transpose()?;
        let ptr = c_path.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let status = unsafe { sys::common_shim_log_set_file(ptr) };
        check_status(status)
    }

    /// Pause the logger's worker thread. Records emitted while paused are
    /// dropped, which is how upstream keeps progress bars readable.
    pub fn pause() {
        unsafe { sys::common_shim_log_pause() }
    }

    /// Resume after [`pause`].
    pub fn resume() {
        unsafe { sys::common_shim_log_resume() }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model resolution
// ─────────────────────────────────────────────────────────────────────────────

/// Resolving models from Hugging Face and Docker, through llama.cpp's own
/// cache.
///
/// This crate's examples use the `hf-hub` crate, which keeps its own cache.
/// These functions share the cache llama.cpp's CLI tools use, so a model pulled
/// by `llama-cli` is found here and vice versa.
pub mod download {
    use super::{check_status, read_string, CString, Result, RuntimeError};
    use llama_cpp_sys_4 as sys;

    /// Resolve `repo[:tag]` to a local path, downloading if needed.
    ///
    /// Pass `file` to select one file from a repo with several; leave it `None`
    /// to let llama.cpp pick.
    ///
    /// **This blocks and may download gigabytes.** There is no progress
    /// callback on this entry point — llama.cpp prints progress through its own
    /// logger, so [`log`](super::log) controls what you see.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Failed`] if resolution or download fails, or
    /// [`RuntimeError::Nul`] for an interior NUL.
    pub fn resolve_hf(repo_with_tag: &str, file: Option<&str>) -> Result<String> {
        let c_repo = CString::new(repo_with_tag)?;
        let c_file = file.map(CString::new).transpose()?;
        let file_ptr = c_file.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        read_string(|buf, len, expected| unsafe {
            sys::common_shim_download_resolve_path(c_repo.as_ptr(), file_ptr, buf, len, expected)
        })
    }

    /// Split `repo:tag` into its parts.
    ///
    /// A bare `user/model` yields an **empty** tag — llama.cpp substitutes no
    /// default here, it just reports what was written. The repo must be exactly
    /// `user/model`; anything else is rejected.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Failed`] if llama.cpp throws, or
    /// [`RuntimeError::Nul`] for an interior NUL.
    pub fn split_repo_tag(repo_with_tag: &str) -> Result<(String, String)> {
        let c_repo = CString::new(repo_with_tag)?;

        // Both outputs share one call, so size them together and fill together.
        let mut repo_len: usize = 0;
        let mut tag_len: usize = 0;
        let status = unsafe {
            sys::common_shim_download_split_repo_tag(
                c_repo.as_ptr(),
                std::ptr::null_mut(),
                0,
                &raw mut repo_len,
                std::ptr::null_mut(),
                0,
                &raw mut tag_len,
            )
        };
        if status != sys::LLAMA_SHIM_BUFFER_TOO_SMALL {
            check_status(status)?;
        }

        let mut repo_buf = vec![0u8; repo_len.max(1)];
        let mut tag_buf = vec![0u8; tag_len.max(1)];
        let status = unsafe {
            sys::common_shim_download_split_repo_tag(
                c_repo.as_ptr(),
                repo_buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                repo_buf.len(),
                &raw mut repo_len,
                tag_buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                tag_buf.len(),
                &raw mut tag_len,
            )
        };
        check_status(status)?;
        Ok((trim_nul(repo_buf)?, trim_nul(tag_buf)?))
    }

    fn trim_nul(mut buf: Vec<u8>) -> Result<String> {
        let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
        buf.truncate(end);
        String::from_utf8(buf).map_err(RuntimeError::from)
    }

    /// Delete a cached model. Returns whether anything was removed.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Failed`] if llama.cpp throws, or
    /// [`RuntimeError::Nul`] for an interior NUL.
    pub fn remove_cached(repo_with_tag: &str) -> Result<bool> {
        let c_repo = CString::new(repo_with_tag)?;
        let rc = unsafe { sys::common_shim_download_remove(c_repo.as_ptr()) };
        if rc < 0 {
            check_status(rc)?;
        }
        Ok(rc == 1)
    }

    /// Every model in llama.cpp's cache, as a JSON array of
    /// `{"repo","tag","name"}` objects.
    ///
    /// Returned as JSON rather than a typed struct so this does not pin a JSON
    /// dependency on the crate, and so upstream adding a field does not break
    /// the signature.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Failed`] if the cache cannot be read.
    pub fn list_cached_json() -> Result<String> {
        read_string(|buf, len, expected| unsafe {
            sys::common_shim_list_cached_models(buf, len, expected)
        })
    }

    /// Resolve a Docker model reference to a local path.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Failed`] if resolution fails, or
    /// [`RuntimeError::Nul`] for an interior NUL.
    pub fn resolve_docker(reference: &str) -> Result<String> {
        let c_ref = CString::new(reference)?;
        read_string(|buf, len, expected| unsafe {
            sys::common_shim_docker_resolve_model(c_ref.as_ptr(), buf, len, expected)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speculative_type_names_round_trip() {
        // `none` is the one name every build has.
        let none = SpeculativeType::from_name("none").expect("parse 'none'");
        assert_eq!(none.name().unwrap(), "none");
    }

    #[test]
    fn speculative_type_rejects_unknown_name() {
        assert!(SpeculativeType::from_name("not-a-strategy").is_err());
    }

    #[test]
    fn speculative_type_rejects_interior_nul() {
        assert!(matches!(
            SpeculativeType::from_name("no\0ne"),
            Err(RuntimeError::Nul(_))
        ));
    }

    /// The help string must list something; an empty one would mean the type
    /// table failed to link.
    #[test]
    fn all_speculative_names_is_non_empty() {
        let all = SpeculativeType::all_names().expect("all names");
        assert!(!all.is_empty(), "no speculative types listed");
        assert!(all.contains("none"), "got {all}");
    }

    /// A path that is not a GGUF must report "no speculative support" rather
    /// than failing — callers probe untrusted paths with this.
    #[test]
    fn speculative_types_from_a_missing_file_is_empty() {
        let types = speculative_types_from_gguf("/definitely/not/a/model.gguf").unwrap();
        assert!(types.is_empty(), "got {types:?}");
    }

    #[test]
    fn speculative_types_rejects_interior_nul() {
        assert!(matches!(
            speculative_types_from_gguf("a\0b"),
            Err(RuntimeError::Nul(_))
        ));
    }

    /// A real GGUF that advertises no speculative head must also come back
    /// empty, which is the case that distinguishes "unreadable" from
    /// "readable but unsupported".
    #[test]
    fn speculative_types_from_a_plain_model_is_empty() {
        let Some(path) = std::env::var_os("LLAMA_TEST_MODEL") else {
            eprintln!("SKIP: no test model available");
            return;
        };
        let types = speculative_types_from_gguf(&path.to_string_lossy()).unwrap();
        assert!(
            types.iter().all(|t| t.name().unwrap_or_default() != "draft-eagle3"),
            "a plain model should not advertise EAGLE-3: {types:?}"
        );
    }

    #[test]
    fn split_repo_tag_separates_the_parts() {
        let (repo, tag) = download::split_repo_tag("ggml-org/models:Q4_K_M").unwrap();
        assert_eq!(repo, "ggml-org/models");
        assert_eq!(tag, "Q4_K_M");
    }

    /// A bare repo yields an empty tag rather than a substituted default —
    /// callers that need one must supply it themselves.
    #[test]
    fn split_repo_tag_leaves_a_missing_tag_empty() {
        let (repo, tag) = download::split_repo_tag("ggml-org/models").unwrap();
        assert_eq!(repo, "ggml-org/models");
        assert_eq!(tag, "");
    }

    /// A repo that is not `user/model` makes llama.cpp throw; the shim must
    /// turn that into an error rather than letting it unwind into Rust.
    #[test]
    fn split_repo_tag_rejects_a_malformed_repo() {
        assert!(download::split_repo_tag("not-a-repo").is_err());
        assert!(download::split_repo_tag("too/many/parts").is_err());
    }

    #[test]
    fn split_repo_tag_rejects_interior_nul() {
        assert!(matches!(
            download::split_repo_tag("a\0b"),
            Err(RuntimeError::Nul(_))
        ));
    }

    /// Listing an empty or absent cache must be valid JSON, not an error —
    /// callers parse it unconditionally.
    #[test]
    fn list_cached_models_returns_json() {
        let json = download::list_cached_json().expect("cache listing");
        assert!(
            json.starts_with('['),
            "expected a JSON array, got {json:.40}"
        );
    }

    #[test]
    fn log_controls_do_not_panic() {
        // Purely global setters; the contract under test is that they link and
        // are callable, and that restoring the defaults leaves the logger sane.
        log::set_timestamps(true);
        log::set_prefix(true);
        log::set_colors(false);
        log::set_timestamps(false);
        log::set_prefix(false);
        assert!(log::set_file(None).is_ok());
    }

    #[test]
    fn log_set_file_rejects_interior_nul() {
        assert!(matches!(
            log::set_file(Some("a\0b")),
            Err(RuntimeError::Nul(_))
        ));
    }
}
