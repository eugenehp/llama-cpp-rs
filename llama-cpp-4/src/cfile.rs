//! A minimal owned C `FILE *`.
//!
//! A couple of llama.cpp entry points read from an already-open handle at its
//! current position rather than from a path — that is the only way to load a
//! GGUF that is *embedded* in a larger file. Reaching them needs a `FILE *`,
//! which Rust's `std::fs` cannot produce.
//!
//! Opening goes through ggml's own `ggml_fopen`, which is already linked into
//! every build of this crate and does the UTF-8 → UTF-16 conversion that a bare
//! `fopen` gets wrong for a non-ASCII path on Windows. Only closing and seeking
//! have to be declared here.

// Every `unsafe` block below is justified in a `SAFETY:` comment; keep it that
// way. See the note on the lint in the workspace `Cargo.toml`.
#![deny(clippy::undocumented_unsafe_blocks)]

use std::ffi::{c_char, c_int, CString};
use std::io;
use std::path::Path;
use std::ptr::NonNull;

#[cfg(not(windows))]
use std::ffi::c_long;

use llama_cpp_sys_4::FILE;

/// `SEEK_SET` — offsets are from the start of the file.
const SEEK_SET: c_int = 0;

extern "C" {
    fn fclose(file: *mut FILE) -> c_int;

    // `fseek` takes a `long`, which is 32-bit on Windows, so use the CRT's
    // 64-bit variant there. On a 32-bit Unix target `long` is 32-bit too and
    // offsets are capped at 2 GiB — `seek_to` reports that rather than
    // truncating. `fseeko` would lift the cap but takes `off_t`, whose width
    // depends on `_FILE_OFFSET_BITS` in the C build, and guessing it wrong is
    // an ABI mismatch.
    #[cfg(not(windows))]
    fn fseek(file: *mut FILE, offset: c_long, whence: c_int) -> c_int;
    #[cfg(windows)]
    fn _fseeki64(file: *mut FILE, offset: i64, whence: c_int) -> c_int;
}

/// An owned read-only C `FILE *`, closed on drop.
#[derive(Debug)]
pub(crate) struct CFile {
    raw: NonNull<FILE>,
}

impl CFile {
    /// Open `path` for reading in binary mode.
    ///
    /// # Errors
    ///
    /// Returns the OS error if the path contains an interior NUL or cannot be
    /// opened.
    pub(crate) fn open_read(path: &Path) -> io::Result<Self> {
        // ggml_fopen takes UTF-8 and widens it itself on Windows, so a lossy
        // conversion here would corrupt a non-UTF-8 path rather than fail. Ask
        // for UTF-8 and report anything else as invalid.
        let path_str = path.to_str().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("path {} is not valid UTF-8", path.display()),
            )
        })?;
        let c_path = CString::new(path_str).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("path {} contains an interior NUL: {e}", path.display()),
            )
        })?;

        // SAFETY: both arguments are NUL-terminated C strings that outlive the
        // call — `c_path` is a live `CString` and the mode is a literal.
        // `ggml_fopen` only reads them, and returns either null or a handle this
        // type then owns.
        let raw = unsafe {
            llama_cpp_sys_4::ggml_fopen(c_path.as_ptr(), c"rb".as_ptr().cast::<c_char>())
        };
        NonNull::new(raw)
            .map(|raw| Self { raw })
            .ok_or_else(io::Error::last_os_error)
    }

    /// Position the handle `offset` bytes from the start of the file.
    ///
    /// # Errors
    ///
    /// Returns the OS error if the seek fails, or
    /// [`io::ErrorKind::InvalidInput`] if `offset` does not fit the platform's
    /// seek offset — only reachable on a 32-bit target.
    pub(crate) fn seek_to(&mut self, offset: u64) -> io::Result<()> {
        #[cfg(windows)]
        let rc = {
            let offset = i64::try_from(offset).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("offset {offset} does not fit this target's file offset type"),
                )
            })?;
            // SAFETY: `self.raw` is a live handle for as long as `self` is —
            // only `Drop` closes it — and `SEEK_SET` is a valid `whence`.
            unsafe { _fseeki64(self.raw.as_ptr(), offset, SEEK_SET) }
        };
        #[cfg(not(windows))]
        let rc = {
            let offset = c_long::try_from(offset).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("offset {offset} does not fit this target's file offset type"),
                )
            })?;
            // SAFETY: as above. `offset` was narrowed to `c_long` by the
            // `try_from` that precedes it, which is what makes the declared
            // `c_long` parameter type the right ABI on every target here.
            unsafe { fseek(self.raw.as_ptr(), offset, SEEK_SET) }
        };

        if rc == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut FILE {
        self.raw.as_ptr()
    }
}

impl Drop for CFile {
    fn drop(&mut self) {
        // SAFETY: `self.raw` came from `ggml_fopen` in `open_read` and is
        // closed exactly once, here. `CFile` is neither `Copy` nor `Clone`, so
        // there is no second owner, and nothing hands the raw handle out except
        // `as_ptr`, which borrows.
        unsafe { fclose(self.raw.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_read_reports_a_missing_file() {
        let err = CFile::open_read(Path::new("definitely/not/a/real/path.gguf")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn open_read_rejects_an_interior_nul() {
        // `Path` happily holds one; `ggml_fopen` would see a truncated name.
        let err = CFile::open_read(Path::new("bad\0path.gguf")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn open_and_seek_an_existing_file() {
        // Cargo runs tests with the package directory as CWD.
        let mut file = CFile::open_read(Path::new("Cargo.toml")).expect("open Cargo.toml");
        file.seek_to(4).expect("seek within the file");
        // Seeking past the end is legal for a read handle; the next read fails.
        file.seek_to(u64::from(u32::MAX))
            .expect("seek past the end");
    }
}
