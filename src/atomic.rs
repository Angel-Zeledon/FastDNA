// src/atomic.rs

//! Write-then-rename file replacement, plus the same-file check the CLI's
//! input-overwrite guard is built on.
//!
//! Every exporter used to `File::create` its destination directly, which
//! truncates the previous good file the instant the write starts: a disk-full
//! error, a mid-export crash, or Ctrl+C left a corrupt half-written table at
//! the destination -- exactly what a downstream Parquet reader then chokes
//! on. `AtomicFile` writes to a sibling temp file in the same directory (so
//! the final rename is a same-volume move, never a copy) and only replaces
//! the destination on `commit`; dropping it uncommitted removes the temp
//! file and leaves the destination untouched.

use std::fs::{self, File};
use std::path::{Path, PathBuf};

use crate::error::{FastDnaError, Result};

/// Sibling temp path for `dest`: `counts.parquet` becomes
/// `counts.parquet.tmp-<pid>` in the same directory. The pid suffix keeps two
/// concurrent processes exporting to the same destination from clobbering
/// each other's in-progress temp file.
fn temp_path_for(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(format!(".tmp-{}", std::process::id()));
    dest.with_file_name(name)
}

/// A pending atomic replacement of `dest`. Obtain one with
/// [`AtomicFile::create`], write to the returned [`File`], drop the handle,
/// then call [`commit`](AtomicFile::commit). Dropping this without
/// committing abandons the write: the temp file is removed and `dest` keeps
/// whatever it held before.
pub struct AtomicFile {
    temp: PathBuf,
    dest: PathBuf,
    committed: bool,
}

impl AtomicFile {
    /// Creates the temp file next to `dest` and returns it together with the
    /// pending-replacement guard. Errors name `dest`, not the internal temp
    /// path: the destination is the path the caller asked for and the only
    /// one an error message can meaningfully point at.
    pub fn create(dest: &Path) -> Result<(File, AtomicFile)> {
        let temp = temp_path_for(dest);
        let file = File::create(&temp).map_err(|e| FastDnaError::Io {
            path: dest.to_path_buf(),
            source: e,
        })?;
        Ok((
            file,
            AtomicFile {
                temp,
                dest: dest.to_path_buf(),
                committed: false,
            },
        ))
    }

    /// The path being written under the hood. Callers that need to hand a
    /// path (rather than a `File`) to a writer can use this and still get
    /// the atomic-replace behavior on `commit`.
    pub fn temp_path(&self) -> &Path {
        &self.temp
    }

    /// Renames the temp file over the destination. The caller must have
    /// dropped (and thereby flushed/closed) every handle to the temp file
    /// first -- on Windows a rename with an open handle fails.
    pub fn commit(mut self) -> Result<()> {
        // std's rename replaces an existing destination on both Unix
        // (rename(2)) and Windows (MOVEFILE_REPLACE_EXISTING).
        match fs::rename(&self.temp, &self.dest) {
            Ok(()) => {
                self.committed = true;
                Ok(())
            }
            Err(e) => Err(FastDnaError::Io {
                path: self.dest.clone(),
                source: e,
            }),
        }
    }
}

impl Drop for AtomicFile {
    fn drop(&mut self) {
        if !self.committed {
            // Best-effort cleanup of an abandoned write; the destination was
            // never touched, so there is nothing else to undo.
            let _ = fs::remove_file(&self.temp);
        }
    }
}

/// Probes that `dest` can be created at all -- parent directory exists, path
/// is writable -- by creating and immediately abandoning its temp file.
/// Run this for every output path *before* a long counting run, so a typo'd
/// output directory fails in milliseconds instead of after hours of work.
pub fn preflight_writable(dest: &Path) -> Result<()> {
    let (_file, _pending) = AtomicFile::create(dest)?;
    Ok(())
}

/// Whether two paths refer to the same file, resolving relative components
/// and symlinks, and comparing case-insensitively on Windows. A path that
/// does not exist yet (a planned output) is resolved through its parent
/// directory; if neither interpretation resolves, the answer is `false` --
/// the guard built on this must never block a run over an unresolvable
/// path, only over a provable collision.
pub fn same_file(a: &Path, b: &Path) -> bool {
    let (Some(a), Some(b)) = (normalize(a), normalize(b)) else {
        return false;
    };
    if cfg!(windows) {
        a.as_os_str().to_string_lossy().to_lowercase()
            == b.as_os_str().to_string_lossy().to_lowercase()
    } else {
        a == b
    }
}

fn normalize(path: &Path) -> Option<PathBuf> {
    if let Ok(canonical) = fs::canonicalize(path) {
        return Some(canonical);
    }
    // The path itself does not exist (e.g. an output file to be created):
    // canonicalize its parent and re-attach the file name.
    let file_name = path.file_name()?;
    let parent = path.parent()?;
    let parent = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    fs::canonicalize(parent).ok().map(|p| p.join(file_name))
}
