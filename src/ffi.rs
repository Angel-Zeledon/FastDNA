// src/ffi.rs
//! PyO3 bindings. The entire FFI surface lives here and is deliberately small:
//! everything that can be expressed in pure Python lives in `python/fastdna/`
//! instead, because each function crossing this boundary must be compiled and
//! tested on five platforms.
//!
//! The module itself is gated in `lib.rs` via `#[cfg(feature = "python")]` on
//! the `pub mod ffi;` declaration, so no inner `#![cfg(...)]` is needed here
//! -- adding one produces a `duplicated_attributes` clippy warning.

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
