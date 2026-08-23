// src/cohort/discovery.rs
//! Sample discovery: given a directory of FASTQ files, figure out which
//! files belong to which patient sample, pairing forward/reverse reads
//! (`_R1`/`_R2`, `_1`/`_2`, and Illumina's `_R1_001`/`_R2_001` demultiplexed
//! form) into a single sample rather than counting them as two.
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
/// `genome_R123` (contains the substring `_R1` inside `_R123`, but is not
/// suffixed by it).
///
/// This list alone does not cover Illumina's own demultiplexing output
/// (`sample_S1_L001_R1_001.fastq.gz`): that stem ends in `_001`, not
/// `_R1`, so none of these suffixes match it. See `illumina_pair_role`,
/// which is checked first in `split_sample_id` specifically for that shape.
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

/// Recognizes Illumina's demultiplexed FASTQ naming convention --
/// `<prefix>_R1_<digits>` / `<prefix>_R2_<digits>`, e.g.
/// `sample_S1_L001_R1_001` -- and splits it into the sample id prefix and
/// the pair role it names.
///
/// This is the single most common real-world FASTQ filename shape, and the
/// generic trailing-suffix rules in `PAIR_SUFFIXES` cannot see it: the stem
/// does not *end* in `_R1`/`_R2`, it ends in a numeric run (`_001`, the
/// lane/set index Illumina's software appends) added after the pair
/// marker. Without a dedicated rule, `_R1_001` and `_R2_001` files each
/// fall through as unsuffixed, so they become two single-end samples with
/// *different* ids and, because neither carries a recognized role, no
/// orphan warning fires either -- a paired cohort silently becomes twice
/// as many half-samples.
///
/// Deliberately narrow: the run after `_R1_`/`_R2_` must be non-empty and
/// entirely ASCII digits, and there must be a non-empty prefix before it.
/// `pat_R1_extra` (a non-numeric trailing part) does not match, and is left
/// to the generic rules below, which also do not match it (it does not end
/// in `_R1`), so it correctly stays single-end sample `pat_R1_extra`.
fn illumina_pair_role(stem: &str) -> Option<(&str, Role)> {
    for (marker, role) in [("_R1_", Role::Forward), ("_R2_", Role::Reverse)] {
        if let Some(idx) = stem.rfind(marker) {
            let prefix = &stem[..idx];
            let trailing = &stem[idx + marker.len()..];
            if !prefix.is_empty()
                && !trailing.is_empty()
                && trailing.bytes().all(|b| b.is_ascii_digit())
            {
                return Some((prefix, role));
            }
        }
    }
    None
}

/// Splits a stem into its sample id and, if the stem carries a recognized
/// pair marker, which read of the pair it is. Tries the Illumina-specific
/// form first (see `illumina_pair_role`), then falls back to the generic
/// trailing-suffix rules. A stem matching neither is single-end and keeps
/// its id unchanged.
fn split_sample_id(stem: &str) -> (&str, Option<Role>) {
    if let Some((base, role)) = illumina_pair_role(stem) {
        return (base, Some(role));
    }

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
        assert!(s[0].files[0].ends_with("pat_001_R1.fastq.gz"), "R1 must be files[0]: {:?}", s[0].files);
        assert!(s[0].files[1].ends_with("pat_001_R2.fastq.gz"), "R2 must be files[1]: {:?}", s[0].files);
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
        assert!(s[0].files[0].ends_with("p-1.a_R1.fastq.gz"), "R1 must be files[0]: {:?}", s[0].files);
        assert!(s[0].files[1].ends_with("p-1.a_R2.fastq.gz"), "R2 must be files[1]: {:?}", s[0].files);
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
        assert!(samples[0].files[0].ends_with("pat_a_R1.fastq.gz"), "R1 must be files[0]: {:?}", samples[0].files);
        assert!(samples[0].files[1].ends_with("pat_a_R2.fastq.gz"), "R2 must be files[1]: {:?}", samples[0].files);
        assert_eq!(samples[1].sample_id, "pat_b");
        assert_eq!(samples[1].files.len(), 2);
        assert!(samples[1].orphan_warning.is_empty());
        assert!(samples[1].files[0].ends_with("pat_b_1.fastq.gz"), "_1 must be files[0]: {:?}", samples[1].files);
        assert!(samples[1].files[1].ends_with("pat_b_2.fastq.gz"), "_2 must be files[1]: {:?}", samples[1].files);
    }

    #[test]
    fn dot_separator_pairs_r1_and_r2() {
        let d = fixture(&["pat_001.R1.fastq.gz", "pat_001.R2.fastq.gz"]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].sample_id, "pat_001");
        assert_eq!(s[0].files.len(), 2);
        assert!(s[0].orphan_warning.is_empty());
        assert!(s[0].files[0].ends_with("pat_001.R1.fastq.gz"), "R1 must be files[0]: {:?}", s[0].files);
        assert!(s[0].files[1].ends_with("pat_001.R2.fastq.gz"), "R2 must be files[1]: {:?}", s[0].files);
    }

    /// The doc comment on `SampleFiles::files` promises a specific,
    /// reproducible order within a sample -- R1 file(s), then R2, then
    /// any single-end file(s) -- because matrix row order downstream
    /// depends on it. Every other pairing test above only asserted
    /// `files.len()`, so a regression that shuffled the vector (e.g. an
    /// accidental `HashSet`, or swapping the `extend` order in
    /// `discover_samples`) would go green everywhere. This test
    /// exercises all three groups at once, and in a sample that mixes a
    /// paired R1/R2 with an unsuffixed file for the same sample id (an
    /// unsuffixed file and its `_R1`-suffixed sibling do resolve to the
    /// same `sample_id`), so the full three-way order is checked, not
    /// just the two-group case.
    #[test]
    fn files_within_a_sample_are_ordered_r1_then_r2_then_single_end() {
        let d = fixture(&["multi_R1.fastq", "multi_R2.fastq", "multi.fastq"]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].sample_id, "multi");
        assert_eq!(s[0].files.len(), 3);
        assert!(s[0].orphan_warning.is_empty(), "both R1 and R2 are present, so no orphan warning is expected");
        assert!(s[0].files[0].ends_with("multi_R1.fastq"), "R1 must be files[0]: {:?}", s[0].files);
        assert!(s[0].files[1].ends_with("multi_R2.fastq"), "R2 must be files[1]: {:?}", s[0].files);
        assert!(s[0].files[2].ends_with("multi.fastq"), "the unsuffixed file must be files[2], last: {:?}", s[0].files);
    }

    #[test]
    fn suffix_like_text_inside_the_sample_id_does_not_confuse_pairing() {
        // A naive `contains("_R1")` would wrongly treat this as an R1 file
        // of sample "pat"; only a trailing suffix (or the dedicated
        // Illumina `_R1_<digits>` form, which this is not: "R123" has no
        // separator after the digit) counts.
        let d = fixture(&["genome_R123.fastq"]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].sample_id, "genome_R123");
        assert_eq!(s[0].files.len(), 1);
        assert!(s[0].orphan_warning.is_empty(), "not a pair suffix, so no orphan to report");
    }

    // `pat_R1_001.fastq` used to be this suite's example of "suffix-like
    // text inside the sample id must not be mistaken for a pair suffix",
    // asserting it was single-end sample `pat_R1_001`. That was the
    // brief's own mistake, not a real requirement: `_R1_001` is exactly
    // Illumina's real demultiplexed naming convention, and treating it as
    // an opaque sample id is what silently turns every real Illumina pair
    // into two half-samples with no orphan warning (see the module-level
    // Illumina tests below). The claim is updated here, deliberately, to
    // match the corrected behaviour: `pat_R1_001` alone is now sample
    // `pat`, role R1, reported as an orphan because its `_R2_001` mate is
    // missing.
    #[test]
    fn pat_r1_001_alone_is_now_an_orphan_r1_of_sample_pat() {
        let d = fixture(&["pat_R1_001.fastq"]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].sample_id, "pat");
        assert_eq!(s[0].files.len(), 1);
        assert!(!s[0].orphan_warning.is_empty(), "an R1 with no R2 mate must not be silent");
    }

    #[test]
    fn illumina_r1_r2_001_form_pairs_into_one_sample() {
        let d = fixture(&[
            "sample_S1_L001_R1_001.fastq.gz",
            "sample_S1_L001_R2_001.fastq.gz",
        ]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].sample_id, "sample_S1_L001");
        assert_eq!(s[0].files.len(), 2);
        assert!(s[0].orphan_warning.is_empty());
        assert!(
            s[0].files[0].to_string_lossy().contains("_R1_"),
            "R1 must sort first: {:?}",
            s[0].files
        );
        assert!(
            s[0].files[1].to_string_lossy().contains("_R2_"),
            "R2 must sort second: {:?}",
            s[0].files
        );
    }

    #[test]
    fn illumina_form_with_two_lanes_pairs_each_lane_independently() {
        let d = fixture(&[
            "sample_S1_L001_R1_001.fastq.gz",
            "sample_S1_L001_R2_001.fastq.gz",
            "sample_S1_L002_R1_001.fastq.gz",
            "sample_S1_L002_R2_001.fastq.gz",
        ]);
        let samples = discover_samples(d.path()).expect("valid");
        assert_eq!(samples.len(), 2, "got {:?}", samples);
        assert_eq!(samples[0].sample_id, "sample_S1_L001");
        assert_eq!(samples[0].files.len(), 2);
        assert!(samples[0].orphan_warning.is_empty());
        assert_eq!(samples[1].sample_id, "sample_S1_L002");
        assert_eq!(samples[1].files.len(), 2);
        assert!(samples[1].orphan_warning.is_empty());
    }

    #[test]
    fn illumina_form_r1_with_no_r2_mate_produces_an_orphan_warning() {
        let d = fixture(&["sample_S1_L001_R1_001.fastq.gz"]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].sample_id, "sample_S1_L001");
        assert!(!s[0].orphan_warning.is_empty(), "an orphan must not be silent");
    }

    #[test]
    fn illumina_like_but_non_numeric_trailing_part_is_still_single_end() {
        // `_R1_extra` is not Illumina's `_R1_<digits>` form (the trailing
        // part is not numeric), and it does not end in the generic `_R1`
        // suffix either (it ends in `_extra`), so it must stay an opaque,
        // unsuffixed single-end sample id.
        let d = fixture(&["pat_R1_extra.fastq"]);
        let s = discover_samples(d.path()).expect("valid");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].sample_id, "pat_R1_extra");
        assert_eq!(s[0].files.len(), 1);
        assert!(s[0].orphan_warning.is_empty(), "not a pair marker, so no orphan to report");
    }
}
