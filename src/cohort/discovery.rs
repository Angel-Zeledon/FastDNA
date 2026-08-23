// src/cohort/discovery.rs
//! Sample discovery: given a directory of FASTQ files, figure out which
//! files belong to which patient sample, pairing forward/reverse reads
//! (`_R1`/`_R2`, `_1`/`_2`) into a single sample rather than counting them
//! as two.
//!
//! Getting this wrong is silent and catastrophic: a directory of 500
//! paired patients that gets treated as 1000 half-samples still produces a
//! matrix, still trains a model, and nothing downstream can tell the
//! difference. So pairing is its own tested unit, landing before anything
//! that consumes its output.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{FastDnaError, Result};

/// All the files that make up one patient sample, plus anything worth
/// telling the caller about how they were grouped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleFiles {
    pub sample_id: String,
    /// Ordered so matrix row construction from this sample is
    /// reproducible: R1 file(s) first, then R2, then any single-end
    /// file(s), each group sorted lexicographically within itself.
    pub files: Vec<PathBuf>,
    /// Empty when there is nothing to report. Non-empty when a paired-end
    /// suffix (`_R1`/`_R2`/`_1`/`_2`) was found without its mate -- the
    /// sample is still processed (as single-end), but silently falling
    /// back would make a half-loaded cohort look like a successful one.
    pub orphan_warning: String,
}

/// Which half of a read pair a suffix identifies, or neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Forward,
    Reverse,
}

/// Recognized FASTQ extensions, longest (most specific) first so the
/// compressed double extension is matched before the plain one.
const EXTENSIONS: [&str; 4] = [".fastq.gz", ".fq.gz", ".fastq", ".fq"];

/// Recognized pair suffixes, paired with the read they identify. Checked
/// as an exact trailing match on the stem (the filename with its FASTQ
/// extension already removed) -- never a `contains`, which would also
/// match a sample id that merely has this text in its middle, such as
/// `pat_R1_001`.
const PAIR_SUFFIXES: [(&str, Role); 8] = [
    ("_R1", Role::Forward),
    ("_R2", Role::Reverse),
    (".R1", Role::Forward),
    (".R2", Role::Reverse),
    ("_1", Role::Forward),
    ("_2", Role::Reverse),
    (".1", Role::Forward),
    (".2", Role::Reverse),
];

/// Strips a recognized FASTQ extension from a filename, returning the
/// remaining stem. Returns `None` for anything else (a `.txt`, a
/// `README`, a directory name) so callers can skip it silently.
fn strip_fastq_extension(filename: &str) -> Option<&str> {
    for ext in EXTENSIONS {
        if let Some(stem) = filename.strip_suffix(ext) {
            if !stem.is_empty() {
                return Some(stem);
            }
        }
    }
    None
}

/// Splits a stem into its sample id and, if the stem ends in a
/// recognized pair suffix, which read of the pair it is. A stem with no
/// recognized trailing suffix is single-end and keeps its id unchanged.
fn split_sample_id(stem: &str) -> (&str, Option<Role>) {
    for (suffix, role) in PAIR_SUFFIXES {
        if let Some(base) = stem.strip_suffix(suffix) {
            if !base.is_empty() {
                return (base, Some(role));
            }
        }
    }
    (stem, None)
}

/// Per-sample accumulator while walking the directory. Kept separate from
/// `SampleFiles` because the final `files` ordering and `orphan_warning`
/// text are only decided once every file has been seen.
#[derive(Default)]
struct SampleGroup {
    forward: Vec<PathBuf>,
    reverse: Vec<PathBuf>,
    single: Vec<PathBuf>,
}

/// Discovers FASTQ samples in `dir`, pairing `_R1`/`_R2` (and `_1`/`_2`,
/// with either `_` or `.` as the separator) files into a single sample so
/// forward and reverse reads of the same physical fragments are never
/// counted as two patients.
///
/// Non-FASTQ files and subdirectories are ignored. Samples are returned
/// sorted lexicographically by `sample_id`, and each sample's `files` are
/// ordered deterministically, so matrix row order is reproducible across
/// runs and platforms regardless of the OS's directory iteration order.
pub fn discover_samples(dir: &Path) -> Result<Vec<SampleFiles>> {
    let entries =
        fs::read_dir(dir).map_err(|e| FastDnaError::Io { path: dir.to_path_buf(), source: e })?;

    let mut groups: BTreeMap<String, SampleGroup> = BTreeMap::new();

    for entry in entries {
        let entry =
            entry.map_err(|e| FastDnaError::Io { path: dir.to_path_buf(), source: e })?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }
        let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(stem) = strip_fastq_extension(filename) else {
            continue;
        };
        let (sample_id, role) = split_sample_id(stem);
        if sample_id.is_empty() {
            continue;
        }

        let group = groups.entry(sample_id.to_string()).or_default();
        match role {
            Some(Role::Forward) => group.forward.push(path),
            Some(Role::Reverse) => group.reverse.push(path),
            None => group.single.push(path),
        }
    }

    if groups.is_empty() {
        return Err(FastDnaError::NoSamplesFound { dir: dir.to_path_buf() });
    }

    let mut samples = Vec::with_capacity(groups.len());
    for (sample_id, mut group) in groups {
        group.forward.sort();
        group.reverse.sort();
        group.single.sort();

        let orphan_warning = if !group.forward.is_empty() && group.reverse.is_empty() {
            format!(
                "sample '{sample_id}': found R1 file(s) {:?} with no matching R2 mate; \
                 processing as single-end",
                group.forward
            )
        } else if !group.reverse.is_empty() && group.forward.is_empty() {
            format!(
                "sample '{sample_id}': found R2 file(s) {:?} with no matching R1 mate; \
                 processing as single-end",
                group.reverse
            )
        } else {
            String::new()
        };

        let mut files = Vec::with_capacity(group.forward.len() + group.reverse.len() + group.single.len());
        files.extend(group.forward);
        files.extend(group.reverse);
        files.extend(group.single);

        samples.push(SampleFiles { sample_id, files, orphan_warning });
    }

    Ok(samples)
}

#[cfg(test)]
// Matches the established pattern in `progress.rs`/`preview.rs`: `unwrap`/
// `expect` are denied under `src/` because a production code path must
// never panic on a caller's bad input, but test assertions are not that
// code path, and spelling every assertion as a `match` would obscure what
// each test checks.
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn fixture(names: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("failed to create temp dir for fixture");
        for name in names {
            std::fs::write(dir.path().join(name), b"").expect("failed to write fixture file");
        }
        dir
    }

    #[test]
    fn pairs_r1_and_r2_into_one_sample() {
        let d = fixture(&["pat_001_R1.fastq.gz", "pat_001_R2.fastq.gz"]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].sample_id, "pat_001");
        assert_eq!(s[0].files.len(), 2);
    }

    #[test]
    fn unsuffixed_file_is_a_single_end_sample() {
        let d = fixture(&["pat_001.fastq.gz"]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].files.len(), 1);
    }

    #[test]
    fn orphan_r1_is_processed_but_recorded() {
        let d = fixture(&["pat_001_R1.fastq.gz"]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1);
        assert!(!s[0].orphan_warning.is_empty(), "an orphan must not be silent");
    }

    #[test]
    fn sample_order_is_lexicographic_and_reproducible() {
        let d = fixture(&["pat_010.fastq", "pat_002.fastq", "pat_001.fastq"]);
        let ids: Vec<_> = discover_samples(d.path())
            .expect("valid")
            .into_iter()
            .map(|s| s.sample_id)
            .collect();
        assert_eq!(ids, vec!["pat_001", "pat_002", "pat_010"]);
    }

    #[test]
    fn empty_directory_is_an_error_not_an_empty_cohort() {
        let d = fixture(&[]);
        assert!(matches!(discover_samples(d.path()), Err(FastDnaError::NoSamplesFound { .. })));
    }

    #[test]
    fn dots_and_dashes_in_names_do_not_confuse_pairing() {
        let d = fixture(&["p-1.a_R1.fastq.gz", "p-1.a_R2.fastq.gz"]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1, "got {:?}", s);
    }

    #[test]
    fn non_fastq_files_are_ignored_not_errors() {
        let d = fixture(&["pat_001.fastq.gz", "notes.txt", "README"]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].sample_id, "pat_001");
        assert_eq!(s[0].files.len(), 1);
    }

    #[test]
    fn r1_r2_and_1_2_conventions_coexist_in_the_same_directory() {
        let d = fixture(&[
            "pat_a_R1.fastq.gz",
            "pat_a_R2.fastq.gz",
            "pat_b_1.fastq.gz",
            "pat_b_2.fastq.gz",
        ]);
        let mut samples = discover_samples(d.path()).expect("valid");
        samples.sort_by(|a, b| a.sample_id.cmp(&b.sample_id));
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].sample_id, "pat_a");
        assert_eq!(samples[0].files.len(), 2);
        assert!(samples[0].orphan_warning.is_empty());
        assert_eq!(samples[1].sample_id, "pat_b");
        assert_eq!(samples[1].files.len(), 2);
        assert!(samples[1].orphan_warning.is_empty());
    }

    #[test]
    fn dot_separator_pairs_r1_and_r2() {
        let d = fixture(&["pat_001.R1.fastq.gz", "pat_001.R2.fastq.gz"]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].sample_id, "pat_001");
        assert_eq!(s[0].files.len(), 2);
        assert!(s[0].orphan_warning.is_empty());
    }

    #[test]
    fn suffix_like_text_inside_the_sample_id_does_not_confuse_pairing() {
        // A naive `contains("_R1")` would wrongly treat this as an R1 file
        // of sample "pat"; only a trailing suffix counts.
        let d = fixture(&["pat_R1_001.fastq"]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].sample_id, "pat_R1_001");
        assert_eq!(s[0].files.len(), 1);
        assert!(s[0].orphan_warning.is_empty(), "not a pair suffix, so no orphan to report");
    }
}
