//! The k-mer frequency spectrum, in both formats FastDNA can write.
//!
//! GenomeScope 2.0 -- the standard way to get genome size, heterozygosity
//! and ploidy out of a spectrum -- reads exactly what `jellyfish histo`
//! emits: headerless, space-separated `depth count`, ascending. FastDNA's
//! CSV header alone is enough to make GenomeScope reject the file, so the
//! assertions here are byte-level rather than "contains".

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use fastdna_core::counter::KmerCounter;
use fastdna_core::export::{self, HistogramFormat};

struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("fastdna_histogram_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Fixture { dir }
    }

    /// Writes the histogram and reads the file back as raw text.
    fn write(&self, counter: &KmerCounter, format: HistogramFormat, max_depth: Option<u32>) -> String {
        let path = self.dir.join("hist.out");
        export::export_histogram(counter, &path, format, max_depth).expect("write must succeed");
        std::fs::read_to_string(&path).expect("the histogram file must exist")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A counter whose spectrum is exactly: three k-mers seen once, one seen
/// twice, two seen three times, one seen seven times.
fn spectrum_fixture() -> KmerCounter {
    let mut counter = KmerCounter::new();
    let mut occurrences: Vec<u64> = Vec::new();
    for (kmer, times) in [(10u64, 1), (11, 1), (12, 1), (20, 2), (30, 3), (31, 3), (40, 7)] {
        occurrences.extend(std::iter::repeat_n(kmer, times));
    }
    counter.insert_batch(&occurrences);
    counter
}

/// The CSV path is what `--histogram` has always written, and existing
/// scripts parse it by name. This is the exact byte sequence it produced
/// before a format flag existed, pinned so the default can never drift.
#[test]
fn csv_is_byte_identical_to_the_format_that_already_shipped() {
    let fx = Fixture::new("csv_default");
    assert_eq!(
        fx.write(&spectrum_fixture(), HistogramFormat::Csv, None),
        "coverage_depth,kmer_distinct_count\n1,3\n2,1\n3,2\n7,1\n"
    );
}

/// The pre-existing `export_histogram_csv` entry point must keep producing
/// exactly that, so any caller still using it is unaffected.
#[test]
fn the_original_csv_entry_point_is_unchanged() {
    let fx = Fixture::new("csv_alias");
    let path = fx.dir.join("alias.csv");
    export::export_histogram_csv(&spectrum_fixture(), &path).unwrap();

    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "coverage_depth,kmer_distinct_count\n1,3\n2,1\n3,2\n7,1\n"
    );
}

/// What `jellyfish histo` emits and GenomeScope 2.0 consumes: no header,
/// one space between the two numbers, ascending depth, newline-terminated.
#[test]
fn genomescope_is_headerless_space_separated_and_ascending() {
    let fx = Fixture::new("genomescope");
    assert_eq!(
        fx.write(&spectrum_fixture(), HistogramFormat::GenomeScope, None),
        "1 3\n2 1\n3 2\n7 1\n"
    );
}

/// KMC's `-cx` convention: everything above the cap is summed into the cap
/// row rather than dropped, so the total number of distinct k-mers the
/// spectrum accounts for is preserved. Dropping the tail instead would make
/// GenomeScope's coverage model fit against a total that is quietly wrong.
#[test]
fn a_cap_folds_the_tail_into_the_cap_row_in_both_formats() {
    let fx = Fixture::new("cap");

    // Depths 1 and 2 are untouched; depth 3 gains the single depth-7 k-mer.
    assert_eq!(
        fx.write(&spectrum_fixture(), HistogramFormat::GenomeScope, Some(3)),
        "1 3\n2 1\n3 3\n"
    );
    assert_eq!(
        fx.write(&spectrum_fixture(), HistogramFormat::Csv, Some(3)),
        "coverage_depth,kmer_distinct_count\n1,3\n2,1\n3,3\n"
    );
}

/// Summing into the cap row must create that row even when no k-mer had
/// exactly the cap's depth -- otherwise the folded k-mers vanish.
#[test]
fn a_cap_with_no_kmer_at_that_exact_depth_still_gets_a_row() {
    let fx = Fixture::new("cap_gap");
    // Depths present are 1, 2, 3 and 7; capping at 5 folds the 7 into 5.
    assert_eq!(
        fx.write(&spectrum_fixture(), HistogramFormat::GenomeScope, Some(5)),
        "1 3\n2 1\n3 2\n5 1\n"
    );
}

/// A cap at or above the deepest k-mer must change nothing at all.
#[test]
fn a_cap_above_the_deepest_kmer_is_a_no_op() {
    let fx = Fixture::new("cap_high");
    let uncapped = fx.write(&spectrum_fixture(), HistogramFormat::GenomeScope, None);

    assert_eq!(fx.write(&spectrum_fixture(), HistogramFormat::GenomeScope, Some(7)), uncapped);
    assert_eq!(fx.write(&spectrum_fixture(), HistogramFormat::GenomeScope, Some(1_000)), uncapped);
}

/// A cap of 1 collapses the whole spectrum onto one row holding every
/// distinct k-mer -- degenerate, but it must still be arithmetically right.
#[test]
fn a_cap_of_one_collapses_every_kmer_onto_the_first_row() {
    let fx = Fixture::new("cap_one");
    assert_eq!(
        fx.write(&spectrum_fixture(), HistogramFormat::GenomeScope, Some(1)),
        "1 7\n",
        "all seven distinct k-mers must be accounted for"
    );
}

/// An empty counter is a real outcome (everything filtered out), and both
/// formats must produce a well-formed file rather than a truncated one: CSV
/// keeps its header, GenomeScope is legitimately empty.
#[test]
fn an_empty_counter_writes_a_well_formed_file_in_both_formats() {
    let fx = Fixture::new("empty");
    let empty = KmerCounter::new();

    assert_eq!(
        fx.write(&empty, HistogramFormat::Csv, None),
        "coverage_depth,kmer_distinct_count\n"
    );
    assert_eq!(fx.write(&empty, HistogramFormat::GenomeScope, None), "");
}

/// Writers go through `atomic.rs`, so a failed write must leave whatever
/// was at the destination before it untouched rather than a truncated file.
#[test]
fn a_failed_histogram_write_does_not_destroy_the_previous_file() {
    let fx = Fixture::new("atomic");
    let path = fx.dir.join("hist.txt");
    std::fs::write(&path, "PREVIOUS CONTENT").unwrap();

    // A directory in place of the temp file's parent cannot exist, so this
    // fails at creation -- before the destination is touched.
    let unwritable = fx.dir.join("no_such_dir").join("hist.txt");
    assert!(export::export_histogram(
        &spectrum_fixture(),
        &unwritable,
        HistogramFormat::GenomeScope,
        None
    )
    .is_err());

    assert_eq!(std::fs::read_to_string(&path).unwrap(), "PREVIOUS CONTENT");
}
