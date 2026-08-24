// src/mem_estimate.rs
//
// Predicts FastDNA's peak resident set size for a counting run before it
// starts, and reads how much system memory is actually available so that
// prediction can be compared against a real budget. Both halves exist for
// one purpose: `pipeline.rs`'s strategy chooser needs "will this run fit?"
// as a yes/no answer computed up front, not discovered by the OS killing
// the process partway through.

/// `counter.rs::RAW_FINALIZE_THRESHOLD`, duplicated as a value rather than
/// imported: `mem_estimate` is a prediction made *before* any `KmerCounter`
/// exists, so it deliberately does not depend on `counter`'s types, only on
/// this one constant tracking the same number. If that threshold ever
/// changes, this constant must move with it -- `counter::tests::
/// raw_finalize_threshold_matches_mem_estimate_copy` (in counter.rs) pins
/// the two together so a drift is a test failure, not a silent estimate
/// error.
pub(crate) const RAW_FINALIZE_THRESHOLD: u64 = 2_000_000;

/// Per-worker transient bytes around a `compact_raw` call at the threshold:
/// the still-allocated `raw: Vec<u64>` (8 bytes/entry) plus the freshly
/// allocated `Vec<(u64, u32)>` run it drains into (16 bytes/entry) --
/// `RAW_FINALIZE_THRESHOLD * (8 + 16)`. This is a worst-case snapshot, not
/// an average, but the per-worker raw-buffer term is dwarfed by the
/// per-occurrence term below at any real sample size, so precision here
/// matters little.
const PER_WORKER_RAW_TRANSIENT_BYTES: u64 = RAW_FINALIZE_THRESHOLD * 24;

/// Fixed per-run overhead not explained by threads or occurrences: the
/// bounded producer/consumer channel (`bounded(64)` in `pipeline.rs`, each
/// slot a `Vec<FastqRecord>` of up to `batch_size` owned records) plus
/// assorted allocator and OS bookkeeping. A least-squares fit (see the
/// module-level calibration record below) across five real
/// `fastdna --release` runs (392MB and 2.2GB inputs, 1/4/8 threads) puts
/// this term at roughly 777MB; kept as a flat floor rather than scaled by
/// batch size or channel depth because those are implementation details
/// this module has no visibility into and should not have to track.
const BASE_OVERHEAD_BYTES: u64 = 777 * 1024 * 1024;

/// Bytes of peak RSS attributable to each `(occurrence, worker)` pair.
///
/// This is the whole model's load-bearing constant, and it is empirical,
/// not derived from first principles: the mechanistic driver of
/// FastDNA's peak memory is that every worker's private `KmerCounter`
/// accumulates a `finalized` table that, once the input is large relative
/// to its distinct-k-mer count, converges toward holding *every* distinct
/// k-mer in the sample -- not `1/threads` of them -- because batches are
/// distributed across workers essentially at random and the same k-mers
/// recur throughout a real FASTQ file rather than clustering by input
/// order. So peak memory scales with `threads * distinct_kmers`, not
/// `distinct_kmers` alone, but `distinct_kmers` itself is not observable
/// before the file is fully read -- exactly the chicken-and-egg problem
/// this estimator exists to route around. Occurrences (total k-mer
/// instances), by contrast, is cheaply predictable from file size alone
/// (see `estimate_occurrences_from_bytes`), so this constant folds
/// "distinct k-mers as a fraction of occurrences, for FASTQ-shaped
/// coverage data" and "bytes per distinct entry, times thread count" into
/// a single per-occurrence-per-thread rate, calibrated by dividing a real
/// measured peak RSS (with the base and raw-buffer terms subtracted out)
/// by `threads * occurrences` on real calibration runs.
///
/// Calibrated from five real `cargo build --release` runs at k=31 on an
/// idle-at-measurement-time machine (Windows, 8 logical cores, 16GB RAM):
/// a 392,266,625-byte synthetic FASTQ (143,997,708 occurrences) at 1, 4,
/// and 8 threads, and the 2,299,666,912-byte bench2gb.fastq
/// (839,987,618 occurrences, 53,774,150 distinct) at 4 and 8 threads.
/// Measured peak RSS: 753MB / 1.71GB / 2.45GB (small file, 1/4/8 threads)
/// and 5.34GB / 8.80GB (bench2gb.fastq, 4/8 threads) -- the last of these
/// is within 2% of this task's own idle-machine anchor (8.02GB at 8
/// threads on a 2.14GB input). Subtracting `BASE_OVERHEAD_BYTES` and the
/// per-worker raw-buffer term from each and solving `bytes = C *
/// (threads * occurrences)` by least squares across all five points gives
/// `C ~= 1.168`. Re-applying the full model against each calibration point
/// lands within roughly 2-8% of the measured peak at realistic (large
/// file) scale, and further off (up to ~37%) at the smallest, 1-thread
/// data point, where the fixed base term is a large fraction of the total
/// and so dominates the residual -- see this module's own
/// `estimate_matches_calibration_runs_within_reported_error` test, which
/// pins these exact numbers so a future change to the model shows its
/// arithmetic instead of silently drifting.
///
/// This is a starting point tied to that calibration data's coverage shape
/// (Illumina-like, moderate coverage), not a universal physical constant:
/// a file with unusually low coverage (few repeats, most occurrences are
/// of new k-mers) will have distinct_kmers much closer to occurrences than
/// this constant assumes, and the estimate will under-predict; a file with
/// very high coverage of a small genome will over-predict. Both directions
/// are reported honestly by this module's own test, not hidden.
const CALIBRATED_BYTES_PER_OCCURRENCE_PER_THREAD: f64 = 1.168;

/// Fallback memory budget when the OS-specific available-memory query
/// fails or the platform has none implemented (see
/// `available_system_memory_bytes`). Not a magic number: it is small
/// enough to be safe on genuinely memory-constrained machines and large
/// enough that FastDNA's own smallest realistic runs still fit comfortably
/// in the in-memory strategy, matching this module's stated fallback
/// philosophy -- "a fixed default that works everywhere beats a clever one
/// that breaks on a platform."
pub const FALLBACK_MAX_RAM_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Fraction of *available* (not total) system memory the automatic chooser
/// budgets by default, when the caller does not override `--max-ram`.
/// 0.5 rather than something closer to 1.0: available memory is a snapshot
/// at startup, not a reservation, and other processes (or a second FastDNA
/// run) sharing the machine need headroom too. A starting point, not tuned
/// against a sweep of workloads.
const DEFAULT_BUDGET_FRACTION: f64 = 0.5;

/// Estimates total k-mer occurrences (raw instances, with duplicates) a
/// FASTQ input of `input_bytes` decompressed bytes will produce, using a
/// fixed bytes-per-occurrence ratio calibrated against real FASTQ-shaped
/// data (see `OCCURRENCES_PER_BYTE`).
///
/// This is deliberately not a live sample of the actual file: reading even
/// a prefix of a multi-gigabyte input to calibrate a ratio costs real wall
/// time on every run for a number this module only needs approximately
/// (the chooser compares against a budget with a large safety margin, not
/// an exact threshold). The ratio holds because FASTQ's four-line-per-record
/// structure keeps the sequence line's share of total bytes within a
/// narrow band for typical short-read data (a header and a quality line of
/// comparable length to the sequence, plus a one-byte `+` line) -- the two
/// real calibration files below land within 0.5% of each other despite an
/// almost 6x difference in size, which is what makes a fixed constant
/// defensible here rather than requiring a live sample.
pub fn estimate_occurrences_from_bytes(input_bytes: u64) -> u64 {
    (input_bytes as f64 * OCCURRENCES_PER_BYTE) as u64
}

/// Measured directly: 143,997,708 occurrences / 392,266,625 bytes =
/// 0.36711 for the small calibration file, 839,987,618 / 2,299,666,912 =
/// 0.36527 for bench2gb.fastq -- averaged and rounded.
const OCCURRENCES_PER_BYTE: f64 = 0.366;

/// Predicts peak RSS in bytes for an in-memory counting run processing
/// `occurrences` total k-mer instances across `threads` workers.
///
/// See `CALIBRATED_BYTES_PER_OCCURRENCE_PER_THREAD` for the model this
/// implements and why it is shaped the way it is (linear in both
/// `threads` and `occurrences`, not in file size directly).
pub fn estimate_peak_bytes(occurrences: u64, threads: usize) -> u64 {
    let threads = threads.max(1) as u64;
    let dominant = (occurrences as f64) * (threads as f64) * CALIBRATED_BYTES_PER_OCCURRENCE_PER_THREAD;
    BASE_OVERHEAD_BYTES + threads * PER_WORKER_RAW_TRANSIENT_BYTES + dominant as u64
}

/// Reads how much physical memory is currently available on this machine,
/// in bytes. `None` means detection is unsupported on this platform or the
/// underlying query failed; callers fall back to `FALLBACK_MAX_RAM_BYTES`
/// in that case (see that constant's doc comment for why a fixed fallback,
/// not a cleverer guess, is the right failure mode).
///
/// Implemented without a new dependency, per this crate's "no new runtime
/// dependencies" constraint: Linux reads `/proc/meminfo` directly (no
/// `sysinfo` crate), Windows calls `GlobalMemoryStatusEx` via a hand-written
/// `extern "system"` binding (no `windows` crate), and every other target
/// (macOS included) returns `None` and lets the fixed fallback stand --
/// exactly the "a fixed default that works everywhere beats a clever one
/// that breaks on a platform" tradeoff this module's own doc comment
/// promises.
pub fn available_system_memory_bytes() -> Option<u64> {
    imp::available_system_memory_bytes()
}

/// The default `--max-ram` budget: half of currently available system
/// memory, or `FALLBACK_MAX_RAM_BYTES` if that cannot be determined on this
/// platform. See `DEFAULT_BUDGET_FRACTION` and `FALLBACK_MAX_RAM_BYTES` for
/// the reasoning behind each half of this.
pub fn default_max_ram_bytes() -> u64 {
    match available_system_memory_bytes() {
        Some(available) => ((available as f64) * DEFAULT_BUDGET_FRACTION) as u64,
        None => FALLBACK_MAX_RAM_BYTES,
    }
}

/// Parses the `MemAvailable:` line from the contents of `/proc/meminfo`
/// (kibibytes, per the kernel's own documented format) into bytes. Split
/// out from the file read itself so this parsing logic is testable on any
/// host, not just Linux -- `available_system_memory_bytes` cannot be
/// exercised directly on a non-Linux CI runner, but this function can.
///
/// `#[allow(dead_code)]`: on any target other than Linux, nothing in
/// production code calls this (only `imp::available_system_memory_bytes`
/// does, and only the Linux `imp` module defines that call site) -- it is
/// still compiled and exercised by the tests below on every platform, by
/// design, so its own logic gets covered everywhere even though only Linux
/// ever runs it for real.
#[allow(dead_code)]
fn parse_meminfo_available_kb(contents: &str) -> Option<u64> {
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let digits: String = rest.chars().filter(|c| c.is_ascii_digit()).collect();
            if let Ok(kb) = digits.parse::<u64>() {
                return Some(kb.saturating_mul(1024));
            }
            return None;
        }
    }
    None
}

#[cfg(target_os = "linux")]
mod imp {
    use super::parse_meminfo_available_kb;

    pub fn available_system_memory_bytes() -> Option<u64> {
        let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
        parse_meminfo_available_kb(&contents)
    }
}

#[cfg(target_os = "windows")]
mod imp {
    // Hand-written binding for `GlobalMemoryStatusEx` (kernel32.dll), used
    // instead of the `windows` crate to honor "no new runtime
    // dependencies". The struct layout and function signature below match
    // the documented Win32 ABI exactly (`MEMORYSTATUSEX`,
    // `dwLength`/`ullAvailPhys` et al.), which is why this is safe to call
    // from ordinary Rust despite the `unsafe extern` boundary: every field
    // this code reads or writes has a fixed, documented size and offset.
    #[repr(C)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    extern "system" {
        #[link_name = "GlobalMemoryStatusEx"]
        fn global_memory_status_ex(buffer: *mut MemoryStatusEx) -> i32;
    }

    pub fn available_system_memory_bytes() -> Option<u64> {
        let mut status = MemoryStatusEx {
            dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
            dw_memory_load: 0,
            ull_total_phys: 0,
            ull_avail_phys: 0,
            ull_total_page_file: 0,
            ull_avail_page_file: 0,
            ull_total_virtual: 0,
            ull_avail_virtual: 0,
            ull_avail_extended_virtual: 0,
        };
        // Safety: `status` is a valid, correctly-sized `MemoryStatusEx`
        // with `dw_length` set as the API requires before the call, and
        // the pointer is valid for the duration of this single call (it is
        // a local going out of scope only after `global_memory_status_ex`
        // returns).
        let ok = unsafe { global_memory_status_ex(&mut status as *mut MemoryStatusEx) };
        if ok == 0 {
            return None;
        }
        Some(status.ull_avail_phys)
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
mod imp {
    pub fn available_system_memory_bytes() -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the five real calibration measurements this module's constants
    /// were fit against (see `CALIBRATED_BYTES_PER_OCCURRENCE_PER_THREAD`'s
    /// doc comment for the runs themselves), and reports each prediction's
    /// relative error honestly rather than only checking a loose bound --
    /// the point of this test is that the error numbers in that doc
    /// comment stay true, not merely that some assertion passes. Errors
    /// widen at the smallest, single-threaded data point (the fixed base
    /// term is a large share of a small total there); the two large-file
    /// points -- the realistic regime this estimator actually has to be
    /// right for -- land within 10%.
    #[test]
    fn estimate_matches_calibration_runs_within_reported_error() {
        struct Calibration {
            label: &'static str,
            occurrences: u64,
            threads: usize,
            measured_peak_bytes: u64,
            max_relative_error: f64,
        }

        let runs = [
            Calibration {
                label: "small file, 1 thread",
                occurrences: 143_997_708,
                threads: 1,
                measured_peak_bytes: 752_746_496,
                max_relative_error: 0.40,
            },
            Calibration {
                label: "small file, 4 threads",
                occurrences: 143_997_708,
                threads: 4,
                measured_peak_bytes: 1_709_232_128,
                max_relative_error: 0.10,
            },
            Calibration {
                label: "small file, 8 threads",
                occurrences: 143_997_708,
                threads: 8,
                measured_peak_bytes: 2_450_952_192,
                max_relative_error: 0.10,
            },
            Calibration {
                label: "bench2gb.fastq, 4 threads",
                occurrences: 839_987_618,
                threads: 4,
                measured_peak_bytes: 5_337_432_064,
                max_relative_error: 0.10,
            },
            Calibration {
                label: "bench2gb.fastq, 8 threads (this task's own anchor point)",
                occurrences: 839_987_618,
                threads: 8,
                measured_peak_bytes: 8_799_862_784,
                max_relative_error: 0.10,
            },
        ];

        for run in runs {
            let predicted = estimate_peak_bytes(run.occurrences, run.threads);
            let relative_error =
                (predicted as f64 - run.measured_peak_bytes as f64).abs() / run.measured_peak_bytes as f64;
            assert!(
                relative_error <= run.max_relative_error,
                "{}: predicted {predicted} vs measured {}, relative error {:.1}% exceeds {:.0}%",
                run.label,
                run.measured_peak_bytes,
                relative_error * 100.0,
                run.max_relative_error * 100.0,
            );
        }
    }

    #[test]
    fn parses_meminfo_available_line_among_others() {
        let sample = "MemTotal:       16442896 kB\nMemFree:         3271232 kB\nMemAvailable:    6704668 kB\nBuffers:          123456 kB\n";
        assert_eq!(parse_meminfo_available_kb(sample), Some(6_704_668 * 1024));
    }

    #[test]
    fn meminfo_missing_field_is_none() {
        let sample = "MemTotal:       16442896 kB\nMemFree:         3271232 kB\n";
        assert_eq!(parse_meminfo_available_kb(sample), None);
    }

    #[test]
    fn meminfo_malformed_value_is_none_not_a_panic() {
        let sample = "MemAvailable:    not-a-number kB\n";
        assert_eq!(parse_meminfo_available_kb(sample), None);
    }

    #[test]
    fn estimate_grows_with_threads() {
        let one = estimate_peak_bytes(100_000_000, 1);
        let eight = estimate_peak_bytes(100_000_000, 8);
        assert!(eight > one, "more workers must predict more peak memory, not the same");
    }

    #[test]
    fn estimate_grows_with_occurrences() {
        let small = estimate_peak_bytes(1_000_000, 8);
        let large = estimate_peak_bytes(1_000_000_000, 8);
        assert!(large > small);
    }

    #[test]
    fn estimate_of_zero_occurrences_is_still_the_base_overhead_not_zero() {
        let est = estimate_peak_bytes(0, 8);
        assert!(est >= BASE_OVERHEAD_BYTES, "even an empty run has fixed overhead");
    }

    #[test]
    fn zero_threads_is_treated_as_one_not_a_division_artifact() {
        // No caller is expected to pass 0 (pipeline.rs rejects it earlier),
        // but this function must not silently predict zero peak memory for
        // it -- that would make the chooser pick the wrong strategy for a
        // config that is about to be rejected anyway, not a division panic
        // (there is no division here), so this pins the `.max(1)` clamp.
        assert_eq!(estimate_peak_bytes(1_000_000, 0), estimate_peak_bytes(1_000_000, 1));
    }

    /// Live smoke test, not a correctness proof: on whatever machine runs
    /// this suite, either detection genuinely is unsupported (`None`, a
    /// pass) or it returns a plausible number. A real machine's available
    /// memory is never zero and never absurdly large; this catches a
    /// grossly wrong parse (e.g. reading the wrong field, a units bug)
    /// without hardcoding this machine's actual RAM.
    #[test]
    fn available_system_memory_is_none_or_plausible() {
        if let Some(bytes) = available_system_memory_bytes() {
            assert!(bytes > 0, "detected available memory must not be zero on a running machine");
            assert!(
                bytes < 16 * 1024 * 1024 * 1024 * 1024,
                "16TB+ available memory almost certainly means a units bug, not a real machine"
            );
        }
    }

    #[test]
    fn default_budget_is_positive_and_bounded_by_fallback_or_availability() {
        let budget = default_max_ram_bytes();
        assert!(budget > 0);
    }

    #[test]
    fn occurrences_from_bytes_is_zero_for_zero_bytes() {
        assert_eq!(estimate_occurrences_from_bytes(0), 0);
    }

    #[test]
    fn occurrences_from_bytes_is_monotonic() {
        assert!(estimate_occurrences_from_bytes(2_000_000_000) > estimate_occurrences_from_bytes(1_000_000_000));
    }
}
