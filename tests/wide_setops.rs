//! The set operations across both table widths, checked against each other.
//!
//! `count --engine narrow` and `count --engine wide` at the same `k <= 32`
//! produce two tables holding **the same k-mers** in two different key
//! encodings (`kmer_u64` and the 16-byte `kmer_bits`). That is what makes a
//! real differential test possible here: run `union`/`intersect`/`diff` over
//! each width and require the two results to be identical k-mer for k-mer,
//! decoded back to bases. A merge that mis-orders the big-endian byte key,
//! or an exporter that writes the wrong footer, shows up as a difference --
//! and the narrow side of the comparison is the one `scripts/validation/`
//! measures exactly equal to KMC3.
//!
//! The set-algebra identity (`|A ∪ B| = |A| + |B| - |A ∩ B|`) is checked
//! too, because it is an answer this file knows without trusting either
//! width.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use fastdna_core::kmer;
use fastdna_core::ktab::KmerTable;
use fastdna_core::wide_kmer;
use fastdna_core::wide_ktab::WideKmerTable;

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(name: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = format!("{}_{}", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed));
        let dir = std::env::temp_dir().join(format!("fastdna_wide_setops_{name}_{unique}"));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Self(dir)
    }

    fn path(&self, file: &str) -> PathBuf {
        self.0.join(file)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run_fastdna(args: &[&str]) -> String {
    let bin = env!("CARGO_BIN_EXE_fastdna");
    let output = Command::new(bin).args(args).output().expect("run fastdna binary");
    assert!(
        output.status.success(),
        "fastdna {args:?} exited with {:?}\nstdout: {}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Two overlapping halves of one deterministic sequence, so the two samples
/// genuinely share some k-mers and genuinely differ in others -- an
/// intersection of everything or of nothing would make every assertion
/// below pass for the wrong reason.
fn write_samples(dir: &ScratchDir) -> (PathBuf, PathBuf) {
    // A deterministic pseudo-random genome, not a repeated motif. A
    // repeated unit shorter than k makes every window a function of
    // position-mod-unit, so two different regions of it hold the *same*
    // k-mers -- which silently turns the difference into the empty set and
    // the intersection into everything, and every assertion below would
    // then pass for the wrong reason. (That is exactly what the first
    // version of this fixture did.)
    let genome: String = {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        (0..2_000)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                b"ACGT"[(state >> 33) as usize % 4] as char
            })
            .collect()
    };

    let write = |name: &str, from: usize, to: usize| -> PathBuf {
        let path = dir.path(name);
        let mut file = std::fs::File::create(&path).expect("create fastq");
        // Fixed strides rather than random offsets: this test asserts exact
        // equality between two runs, so the input must not vary at all.
        let mut start = from;
        let mut index = 0;
        while start + 120 <= to {
            let read = &genome[start..start + 120];
            writeln!(file, "@r{index}\n{read}\n+\n{}", "I".repeat(read.len())).expect("write");
            start += 37;
            index += 1;
        }
        path
    };

    // A covers [0, 1200), B covers [800, 2000): a shared middle of 400
    // bases, and roughly 800 private to each side. Partial overlap is what
    // makes intersection and difference both non-empty and both proper
    // subsets, which is what the assertions actually rest on.
    (write("a.fastq", 0, 1_200), write("b.fastq", 800, 2_000))
}

/// Counts `reads` at `k` with `engine`, returning the table path.
fn count(dir: &ScratchDir, reads: &Path, k: usize, engine: &str, name: &str) -> PathBuf {
    let out = dir.path(name);
    run_fastdna(&[
        "count",
        "--input", reads.to_str().unwrap(),
        "-k", &k.to_string(),
        "--engine", engine,
        "-o", out.to_str().unwrap(),
        // Out of the current directory: `--qc` defaults to
        // `qc_report.json` relative to it and is always written.
        "--qc", dir.path(&format!("{name}.qc.json")).to_str().unwrap(),
    ]);
    out
}

/// `{decoded k-mer: count}` for a table of either width -- the one form the
/// two encodings share, and therefore the only basis for comparing them.
fn decoded(path: &Path) -> BTreeMap<String, u32> {
    match fastdna_core::ktab::table_key(path).expect("read the table's footer") {
        fastdna_core::ktab::TableKey::Narrow => {
            let table = KmerTable::open(path).expect("open narrow table");
            let k = table.k();
            table
                .iter()
                .expect("iterate narrow table")
                .map(|row| {
                    let (key, count) = row.expect("read row");
                    (kmer::decode_kmer(key, k), count)
                })
                .collect()
        }
        fastdna_core::ktab::TableKey::Wide => {
            let table = WideKmerTable::open(path).expect("open wide table");
            let k = table.k();
            table
                .iter()
                .expect("iterate wide table")
                .map(|row| {
                    let (key, count) = row.expect("read row");
                    (wide_kmer::decode_kmer(key, k), count)
                })
                .collect()
        }
    }
}

/// Runs one set operation over both widths and asserts the two results are
/// the same map. Returns that map, so a caller can also check its size.
fn both_widths_agree(dir: &ScratchDir, op: &str, label: &str) -> BTreeMap<String, u32> {
    let (a_reads, b_reads) = write_samples(dir);

    let mut results = Vec::new();
    for engine in ["narrow", "wide"] {
        let a = count(dir, &a_reads, 31, engine, &format!("a_{engine}.parquet"));
        let b = count(dir, &b_reads, 31, engine, &format!("b_{engine}.parquet"));
        let out = dir.path(&format!("{label}_{engine}.parquet"));

        match op {
            "diff" => run_fastdna(&[
                "diff",
                "--input", a.to_str().unwrap(),
                "--subtract", b.to_str().unwrap(),
                "--output", out.to_str().unwrap(),
            ]),
            _ => run_fastdna(&[
                op,
                "--input", a.to_str().unwrap(), b.to_str().unwrap(),
                "--output", out.to_str().unwrap(),
            ]),
        };
        results.push(decoded(&out));
    }

    assert_eq!(results[0], results[1], "{op}: the two widths disagree");
    assert!(!results[0].is_empty(), "{op} produced nothing; the fixture proves nothing");
    results.pop().unwrap()
}

#[test]
fn union_agrees_across_both_widths() {
    let dir = ScratchDir::new("union");
    both_widths_agree(&dir, "union", "u");
}

#[test]
fn intersect_agrees_across_both_widths() {
    let dir = ScratchDir::new("intersect");
    both_widths_agree(&dir, "intersect", "i");
}

#[test]
fn diff_agrees_across_both_widths() {
    let dir = ScratchDir::new("diff");
    both_widths_agree(&dir, "diff", "d");
}

#[test]
fn the_set_algebra_identity_holds_on_wide_tables() {
    // |A ∪ B| = |A| + |B| - |A ∩ B|, and |A \ B| = |A| - |A ∩ B|. Neither
    // is something this crate computed: they are what the words mean.
    let dir = ScratchDir::new("identity");
    let (a_reads, b_reads) = write_samples(&dir);

    let a = count(&dir, &a_reads, 41, "auto", "a.parquet");
    let b = count(&dir, &b_reads, 41, "auto", "b.parquet");
    assert_eq!(
        fastdna_core::ktab::table_key(&a).unwrap(),
        fastdna_core::ktab::TableKey::Wide,
        "k=41 must produce a wide table, or this test checks the wrong thing"
    );

    let u = dir.path("u.parquet");
    let i = dir.path("i.parquet");
    let d = dir.path("d.parquet");
    run_fastdna(&["union", "--input", a.to_str().unwrap(), b.to_str().unwrap(), "--output", u.to_str().unwrap()]);
    run_fastdna(&["intersect", "--input", a.to_str().unwrap(), b.to_str().unwrap(), "--output", i.to_str().unwrap()]);
    run_fastdna(&["diff", "--input", a.to_str().unwrap(), "--subtract", b.to_str().unwrap(), "--output", d.to_str().unwrap()]);

    let (na, nb) = (decoded(&a).len(), decoded(&b).len());
    let (nu, ni, nd) = (decoded(&u).len(), decoded(&i).len(), decoded(&d).len());

    assert!(ni > 0 && ni < na, "the fixture must overlap partially, not fully or not at all");
    assert_eq!(nu, na + nb - ni, "|A ∪ B| != |A| + |B| - |A ∩ B|");
    assert_eq!(nd, na - ni, "|A \\ B| != |A| - |A ∩ B|");
}

#[test]
fn union_with_sum_adds_the_counts_on_a_wide_table() {
    // A table unioned with itself under `--combine sum` must double every
    // count and change no key. Checkable without knowing any count.
    let dir = ScratchDir::new("sum");
    let (a_reads, _) = write_samples(&dir);
    let a = count(&dir, &a_reads, 41, "auto", "a.parquet");

    let out = dir.path("u.parquet");
    run_fastdna(&[
        "union",
        "--input", a.to_str().unwrap(), a.to_str().unwrap(),
        "--combine", "sum",
        "--output", out.to_str().unwrap(),
    ]);

    let before = decoded(&a);
    let after = decoded(&out);
    assert_eq!(before.keys().collect::<Vec<_>>(), after.keys().collect::<Vec<_>>());
    for (key, count) in &before {
        assert_eq!(after[key], count * 2, "self-union with sum must double {key}");
    }
}

#[test]
fn mixing_a_narrow_and_a_wide_table_is_rejected_by_name() {
    // Two widths never share a k, so a mixed set is always a mistake. It is
    // caught up front and named, rather than surfacing as the wide reader's
    // "missing a kmer_bits column", which would say nothing about what the
    // user actually did.
    let dir = ScratchDir::new("mixed");
    let (a_reads, b_reads) = write_samples(&dir);
    let narrow = count(&dir, &a_reads, 21, "auto", "narrow.parquet");
    let wide = count(&dir, &b_reads, 41, "auto", "wide.parquet");

    let bin = env!("CARGO_BIN_EXE_fastdna");
    let output = Command::new(bin)
        .args([
            "union",
            "--input", narrow.to_str().unwrap(), wide.to_str().unwrap(),
            "--output", dir.path("u.parquet").to_str().unwrap(),
        ])
        .output()
        .expect("run fastdna");
    assert!(!output.status.success(), "mixing widths must fail");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("narrow"), "the error should name both widths:\n{stderr}");
    assert!(stderr.contains("wide"), "the error should name both widths:\n{stderr}");
}

/// The similarity table, computed over both widths of the same k-mers.
///
/// `pairwise_similarity` never looks at a key -- every number it reports
/// comes from `MergedRow`'s per-table counts -- so the two widths must
/// agree to the digit, including the three floating-point ratios. Anything
/// else would mean the merge itself paired rows differently, which is the
/// failure this is here to catch.
#[test]
fn similarity_agrees_across_both_widths() {
    let dir = ScratchDir::new("similarity");
    let (a_reads, b_reads) = write_samples(&dir);

    let mut tables = Vec::new();
    for engine in ["narrow", "wide"] {
        let a = count(&dir, &a_reads, 31, engine, &format!("sa_{engine}.parquet"));
        let b = count(&dir, &b_reads, 31, engine, &format!("sb_{engine}.parquet"));
        let out = dir.path(&format!("sim_{engine}.csv"));
        run_fastdna(&[
            "similarity",
            "--input", a.to_str().unwrap(), b.to_str().unwrap(),
            "--output", out.to_str().unwrap(),
        ]);

        // Every column but the two path labels, which necessarily differ:
        // the two runs read different files.
        let text = std::fs::read_to_string(&out).expect("read similarity output");
        let numbers: Vec<String> = text
            .lines()
            .skip(1)
            .map(|line| line.splitn(3, ',').nth(2).unwrap_or_default().to_string())
            .collect();
        assert_eq!(numbers.len(), 1, "two inputs make exactly one pair:\n{text}");
        tables.push(numbers);
    }

    assert_eq!(tables[0], tables[1], "the two widths report different similarity");

    // And the numbers have to be a real comparison, not two empty tables
    // agreeing that they share nothing.
    let fields: Vec<&str> = tables[0][0].split(',').collect();
    let shared: u64 = fields[0].parse().expect("shared is a number");
    let jaccard: f64 = fields[3].parse().expect("jaccard is a number");
    assert!(shared > 0, "the fixture must share k-mers: {:?}", tables[0]);
    assert!(jaccard > 0.0 && jaccard < 1.0, "partial overlap expected, got {jaccard}");
}

/// A wide table reaches `similarity` at all -- the k>32 path, not just the
/// forced-wide one the differential test above uses.
#[test]
fn similarity_takes_a_genuinely_wide_table() {
    let dir = ScratchDir::new("similarity_wide");
    let (a_reads, b_reads) = write_samples(&dir);
    let a = count(&dir, &a_reads, 41, "auto", "a.parquet");
    let b = count(&dir, &b_reads, 41, "auto", "b.parquet");

    let out = dir.path("sim.csv");
    run_fastdna(&[
        "similarity",
        "--input", a.to_str().unwrap(), b.to_str().unwrap(),
        "--output", out.to_str().unwrap(),
    ]);

    let text = std::fs::read_to_string(&out).expect("read similarity output");
    let row = text.lines().nth(1).expect("one data row");
    let fields: Vec<&str> = row.split(',').collect();

    // shared + only_a is |A|, and shared + only_b is |B| -- checked against
    // the tables' own row counts, which came from a different code path
    // (the Parquet footer) than the merge that produced these.
    let shared: usize = fields[2].parse().unwrap();
    let only_a: usize = fields[3].parse().unwrap();
    let only_b: usize = fields[4].parse().unwrap();
    assert_eq!(shared + only_a, decoded(&a).len(), "shared + only_a != |A|");
    assert_eq!(shared + only_b, decoded(&b).len(), "shared + only_b != |B|");
    assert!(shared > 0, "the fixture must share k-mers at k=41");
}
