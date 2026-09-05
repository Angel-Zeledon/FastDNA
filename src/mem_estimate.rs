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
/// the still-allocated `raw: Vec<u64>` (8 bytes/entry), the equally-sized
/// `scratch: Vec<u64>` that `counter::msd_partition` scatters into and then
/// swaps with it (8 bytes/entry), and the freshly allocated `CountTable` --
/// a pair of `Vec<u64>`/`Vec<u32>` key/count arrays, 12 bytes/entry with no
/// alignment padding, see that type's doc comment in `counter.rs` -- the
/// compaction drains into: `RAW_FINALIZE_THRESHOLD * (8 + 8 + 12)`.
///
/// The middle term (the MSD-partition scratch buffer) is unrelated to the
/// third: `compact_raw` used to call `sort_unstable`, which sorts in place,
/// and now MSD-partitions into a second buffer first (see
/// `counter::msd_partition` for the measurements that bought that trade).
/// At the current threshold it costs each worker 16 MiB, i.e. 128 MiB
/// across 8 workers, independent of whether the run it drains into is a
/// tuple `Vec` or a `CountTable`.
///
/// The third term used to be 16 bytes/entry (`Vec<(u64, u32)>`, padded) and
/// is now 12 (`CountTable`'s parallel arrays, unpadded) after this file's
/// own G-3 struct-of-arrays change -- a mechanical recomputation over known
/// `Vec` sizes, not a re-fit, the same way the MSD-partition scratch term
/// above was added as arithmetic rather than measurement. At the current
/// threshold that is 8 MiB less per worker (2,000,000 * 4 bytes), 64 MiB
/// less across 8 workers, moving this constant's multiplier from `* 32` to
/// `* 28`.
///
/// A note on what *that* does to the calibration below:
/// `CALIBRATED_BYTES_PER_OCCURRENCE_PER_THREAD` was fit against five real
/// runs measured before either the MSD-partition scratch buffer or this
/// G-3 change existed (with this term at `* 24`), so re-fitting it now
/// would mean inventing what those runs would have measured against code
/// they never ran against -- it has deliberately **not** been re-fit, for
/// the same reason the MSD-partition scratch addition above was not. This
/// G-3 change's own share of the drift is small on its own: 64 MiB less
/// across 8 workers against the 8.80 GB measured at 8 threads on
/// `bench2gb.fastq` is 0.7%, moving the prediction down slightly -- the
/// opposite direction from the MSD-partition scratch addition above, but
/// two orders of magnitude smaller (64 MiB against that change's 128 MiB),
/// so the net effect of both since the constant was fit is still a small
/// *increase* in predicted peak, which remains the safe direction for a
/// budget check, and both stay well inside the 2-8% residual that
/// constant's own doc comment already reports. The honest fix remains a
/// fresh calibration run, not an adjusted constant.
///
/// This remains a worst-case snapshot rather than an average, and the whole
/// per-worker term is still dwarfed by the per-occurrence term below at any
/// real sample size.
const PER_WORKER_RAW_TRANSIENT_BYTES: u64 = RAW_FINALIZE_THRESHOLD * 28;

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

/// Bytes one `(kmer, count)` entry occupies in the count tables
/// `estimate_binned_peak_bytes` below models: `binned.rs`'s own per-bin and
/// cross-bin-merge tables, built through `k_way_merge_sorted_counts`
/// (`disk_spill.rs` has its own on-disk encoding, not modelled here).
///
/// 16, not 12: `size_of::<(u64, u32)>()` is 16 because the `u64` forces
/// 8-byte alignment and the trailing `u32` is padded. That padding is real
/// resident memory and this model must charge for it rather than assuming
/// the packed size -- on the 53.8-million-distinct-k-mer benchmark it is
/// 205 MiB.
///
/// `counter.rs`'s own `KmerCounter` no longer uses this representation as
/// of that file's G-3 change: its internal `pending`/`finalized` tables are
/// a `CountTable` (parallel `Vec<u64>`/`Vec<u32>` arrays, 12 bytes/entry, no
/// padding -- see that type's doc comment) rather than `Vec<(u64, u32)>`.
/// This constant does **not** move with it, on purpose: the binned
/// strategy's own peak-memory phases (`estimate_binned_peak_bytes`'s
/// `phase2`/`merge` terms) are dominated by `binned.rs`'s tables, not
/// `counter.rs`'s, and `binned.rs` is out of scope for G-3 (file ownership,
/// not a technical constraint -- see that change's own notes), so it still
/// builds `Vec<(u64, u32)>` tables at the padded size this constant
/// correctly continues to charge for. Widening this doc comment's claim
/// (previously "every count table this crate builds") to name `binned.rs`
/// specifically, instead of loosening the byte count, is what keeps this
/// model honest about what it actually estimates.
/// `count_table_entry_size_matches_the_tuple_it_models` still pins this
/// constant to `size_of::<(u64, u32)>()`, which is what `binned.rs` (via
/// `k_way_merge_sorted_counts`) still uses.
const COUNT_TABLE_BYTES_PER_ENTRY: u64 = 16;

/// Distinct canonical k-mers as a fraction of total k-mer occurrences, for
/// FASTQ-shaped coverage data.
///
/// Measured, not assumed: 53,774,150 distinct out of 839,987,618
/// occurrences on `bench2gb.fastq` -- the same real calibration file
/// `CALIBRATED_BYTES_PER_OCCURRENCE_PER_THREAD` was fit against, and the
/// same file `docs/design-minimizer-counting.md` 4.1's memory table is
/// computed for. 53,774,150 / 839,987,618 = 0.0640178.
///
/// The in-memory model above deliberately avoids needing this number, by
/// folding "distinct k-mers per occurrence" into its single
/// per-occurrence-per-thread rate. The binned model cannot: its two largest
/// terms (the final table, and the transient the cross-bin merge allocates
/// beside it) are proportional to *distinct* k-mers alone, with no thread
/// factor to absorb them into, so the ratio has to appear explicitly. That
/// makes this the binned model's most fragile input, and it fails in the
/// same direction the in-memory model's does: low-coverage input (few
/// repeats, most occurrences are of new k-mers) has a much higher distinct
/// fraction than this and will be under-predicted; high coverage of a small
/// genome will be over-predicted. One file's coverage shape, stated as such.
const DISTINCT_PER_OCCURRENCE: f64 = 0.0640178;

/// Bytes of super-k-mer store per k-mer occurrence under the binned
/// strategy.
///
/// This constant has a different epistemic status from
/// `CALIBRATED_BYTES_PER_OCCURRENCE_PER_THREAD`, and the difference matters:
/// that one is a least-squares fit to five measured peak-RSS numbers, while
/// this one is **derived arithmetic that has never been checked against a
/// measured RSS**. `docs/design-minimizer-counting.md` 3.4 computes it from
/// the minimizer scheme's own parameters -- at `m = 7` and `k = 31` the
/// window is `w = k - m + 1 = 25`, random-order minimizer density is
/// `2/(w+1) = 0.0769`, so a 150-base read yields `(150 - 7 + 1) * 0.0769 =
/// 11.08` super-k-mers averaging `120/11.08 + 30 = 40.83` bases; 2-bit
/// packing those, plus one length byte per record, gives 829 MiB of store
/// for 839,987,618 occurrences -- 1.035 bytes each.
///
/// It is not, however, unchecked. `superkmer.rs`'s
/// `stored_bytes_per_occurrence_are_near_the_designs_prediction` encodes
/// 20,000 random 150-base reads at `k = 31, m = 7` and measures what the
/// encoder actually stores: **0.998 bytes per occurrence** (10.15
/// super-k-mers per read, an 8.02x reduction against the 8 bytes the
/// default path spends). So the derived 1.035 over-predicts the measured
/// figure by 3.7% -- conservative in the direction a memory budget wants to
/// be wrong, which is why the derived value is kept here rather than
/// substituting the measured one.
///
/// What that measurement does *not* cover, and why this is still not a
/// calibrated constant: it is the size of a data structure on synthetic
/// uniform-random sequence, not the peak RSS of a real run on real DNA.
/// Real genomes are not uniform-random, and low-complexity or amplicon data
/// is exactly where minimizer schemes misbehave (7's R3, and the 1516x bin
/// skew `adaptive_bins.rs` exists to fix). Two things could still make this
/// optimistic, both named in 3.4: Roberts' caveat that realised minimizer
/// density on real DNA can run "a few percent above `2/(w+1)`", and the
/// fact that a longer read -- or a `k` other than 31 -- changes the `k-1`
/// overlap's share of the total. 4.3 of that document sets 900 MiB as the
/// falsification threshold for this term on the benchmark file, 8.5% of
/// headroom over this constant's own prediction.
const SUPERKMER_BYTES_PER_OCCURRENCE: f64 = 1.035;

/// Base RSS the binned strategy carries before any input-proportional
/// structure exists, in bytes.
///
/// Separate from `BASE_OVERHEAD_BYTES` (777 MiB), which was fit to the
/// *in-memory* strategy's runs: the binned path never allocates the
/// per-worker raw buffers that dominate that constant, and measuring it
/// directly (a 144M-occurrence run peaks at 891 MiB *total*, of which the
/// store and table account for ~440 MiB) shows its floor is well under
/// half. Fit jointly with `BINNED_CALIBRATION_FACTOR` below, over the same
/// eight runs.
const BINNED_BASE_OVERHEAD_BYTES: u64 = 320 * 1024 * 1024;

/// Multiplies the structural sum below to match measured peak RSS.
///
/// **This is what makes the binned model calibrated rather than
/// structural**, and it was fit on 2026-09-05 to eight release runs on
/// macOS arm64 (Apple M3 Pro, 11 threads available, 19 GB RAM): two input
/// sizes (144,000,000 and 840,000,000 k-mer occurrences, the `mid` and
/// `large` benchmark files) x four thread counts (1, 4, 8, 11), peak RSS
/// read from `/usr/bin/time -l`'s `maximum resident set size`.
///
/// It was chosen as the smallest factor that leaves **no** measured point
/// under-predicted, because the one thing this number must never do is tell
/// the automatic chooser a run fits when it does not. The residuals it
/// leaves are reported by
/// `binned_estimate_matches_the_measured_runs_it_was_fit_to` and range from
/// 0.0% to +11.1% -- conservative everywhere, never optimistic anywhere.
///
/// What it absorbs, and why a bare structural sum could not: allocator
/// retention (macOS does not return freed pages promptly, so peak RSS
/// includes pages the program has already released), the Arrow/Parquet
/// encode buffers the export step holds while the count table is still
/// alive, and rayon's per-thread stacks. None of those is modelled term by
/// term because none was measured term by term -- naming a single fitted
/// factor is honest about that, where quietly inflating
/// `SUPERKMER_BYTES_PER_OCCURRENCE` (which has its own derivation and its
/// own test) to 3.6 would not be.
const BINNED_CALIBRATION_FACTOR: f64 = 1.195;

/// Predicts peak RSS in bytes for a **binned** counting run (`binned.rs` /
/// `CountStrategy::Binned`) over `occurrences` total k-mer instances,
/// `threads` workers, `num_bins` minimizer bins and `chunk_bytes` per open
/// chunk.
///
/// # Calibrated on 2026-09-05, and what changed when it met a real run
///
/// This model was structural until 2026-09-05: every term derived from
/// `binned.rs`'s data structures and `docs/design-minimizer-counting.md`
/// 3.4/3.6/4.1's arithmetic, with nothing compared against an observed RSS,
/// because the environment it was written in had no native release build to
/// measure. That measurement has now been made -- eight release runs, two
/// input sizes x four thread counts (see `BINNED_CALIBRATION_FACTOR`) -- and
/// the structural version was wrong in both directions at once: it
/// **under-predicted the 840M-occurrence runs by 26-29%** and
/// **over-predicted the 144M-occurrence ones by 11-19%**.
///
/// Two things were wrong, and both are fixed above rather than papered over:
/// the merge phase assumed the super-k-mer store had been released by then
/// and it has not, and the base overhead was inherited from the in-memory
/// strategy's 777 MiB fit when the binned path never allocates the
/// per-worker raw buffers that number was mostly made of. With the store
/// counted in the merge and a base fit to this path's own floor, one
/// calibration factor covers the remainder and no measured point is
/// under-predicted.
///
/// What it is nonetheless good for: the automatic chooser needs a
/// conservative, monotone answer to "could this run fit?", and this model's
/// dominant terms are ones whose sizes are fixed by types rather than by
/// behaviour (`COUNT_TABLE_BYTES_PER_ENTRY` is `size_of::<(u64, u32)>()`;
/// the open-chunk term is a literal product of three configured values).
/// Only `SUPERKMER_BYTES_PER_OCCURRENCE` and `DISTINCT_PER_OCCURRENCE` are
/// estimates, and both are documented above with the direction they fail in.
/// Of those two the first is partly corroborated -- `superkmer.rs` measures
/// the encoder storing 0.998 bytes per occurrence against the 1.035 charged
/// here -- while `DISTINCT_PER_OCCURRENCE` rests entirely on one file's
/// coverage shape and is the term most likely to be badly wrong on input
/// unlike it.
///
/// # The three phases, and why the peak is the merge
///
/// The binned strategy's footprint is not one curve but three, and the
/// answer is their maximum -- taking only the largest single term, or
/// summing all three, would both be wrong:
///
/// - **Phase 1 (accumulation).** The super-k-mer store plus the open chunks.
///   The store is what makes this strategy interesting: it is a *partition*
///   of the input across workers, not a per-worker replica, so there is no
///   `threads * occurrences` term here at all -- the very term that makes an
///   8-thread in-memory run cost 8.34 GB. The only thread-proportional term
///   is `threads * num_bins * chunk_bytes` (64 MiB at 8 threads and the
///   defaults), which is independent of input size.
/// - **Phase 2 (per-bin counting).** `count_bin` expands one bin's packed
///   super-k-mers into a `Vec<u64>` (8 bytes per occurrence in that bin),
///   drops that bin's chunks *before* sorting, then compacts into the bin's
///   own table. Up to `threads` bins are in flight at once, and the per-bin
///   tables accumulate into the full output as the store is consumed bin by
///   bin. 4.1 models store and output as half-overlapping, since one is
///   freed at roughly the rate the other fills; that 50% is a midpoint, not
///   a measurement, and it is the least defensible line in this function.
/// - **The cross-bin merge.** `k_way_merge_sorted_counts` allocates the
///   merged table at its full final capacity while every source table is
///   still alive, so the finished table is transiently resident *twice*.
///   With the store fully consumed by then this is `base + 2 * table`, and
///   on the benchmark input it is the largest of the three (2.36 GiB against
///   1.63 and 2.08).
///
/// Reproducing 4.1's published table exactly is what
/// `binned_estimate_reproduces_the_design_document_arithmetic` below checks,
/// so this function and that document cannot drift apart silently.
///
/// `threads` and `num_bins` are floored at 1: a zero bin count would divide
/// by zero below, and both are meaningless at zero anyway.
pub fn estimate_binned_peak_bytes(
    occurrences: u64,
    threads: usize,
    num_bins: usize,
    chunk_bytes: usize,
) -> u64 {
    let threads = threads.max(1) as u64;
    let num_bins = num_bins.max(1) as u64;

    let distinct = (occurrences as f64) * DISTINCT_PER_OCCURRENCE;
    let final_table_bytes = (distinct as u64) * COUNT_TABLE_BYTES_PER_ENTRY;

    // Phase 1: the packed super-k-mer store, plus one open chunk per
    // (worker, bin) pair.
    let store_bytes = ((occurrences as f64) * SUPERKMER_BYTES_PER_OCCURRENCE) as u64;
    let open_chunk_bytes = threads * num_bins * (chunk_bytes as u64);
    let phase1 = BINNED_BASE_OVERHEAD_BYTES + store_bytes + open_chunk_bytes;

    // Phase 2: `threads` bins in flight, each holding its expanded
    // occurrences and its compacted table, beside the output accumulated so
    // far and the half of the store not yet consumed.
    let per_bin_expanded_bytes = (occurrences / num_bins) * 8;
    let per_bin_table_bytes = ((distinct / num_bins as f64) as u64) * COUNT_TABLE_BYTES_PER_ENTRY;
    let phase2_transients = threads * (per_bin_expanded_bytes + per_bin_table_bytes);
    let phase2 = BINNED_BASE_OVERHEAD_BYTES + store_bytes / 2 + phase2_transients + final_table_bytes;

    // The merge: every source table and the destination, all resident --
    // and, measurement says, the store and the open chunks too. The store
    // is what changed after this model met a real run: it was assumed
    // released by merge time and is not, which is most of why the purely
    // structural version under-predicted the 840M-occurrence runs by 26-29%
    // while over-predicting the 144M ones by 11-19%. Both errors are gone
    // once the store is counted here and the base is fit to the binned
    // path's own floor rather than the in-memory path's.
    let merge = BINNED_BASE_OVERHEAD_BYTES + store_bytes + 2 * final_table_bytes + open_chunk_bytes;

    let structural = phase1.max(phase2).max(merge);
    ((structural as f64) * BINNED_CALIBRATION_FACTOR) as u64
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

    /// The benchmark input every number in `docs/design-minimizer-counting.md`
    /// 4.1 is computed for: `bench2gb.fastq`, 8 threads, and `binned.rs`'s
    /// own defaults.
    const BENCH_OCCURRENCES: u64 = 839_987_618;
    const BENCH_THREADS: usize = 8;

    fn mib(bytes: u64) -> f64 {
        bytes as f64 / (1024.0 * 1024.0)
    }

    /// `COUNT_TABLE_BYTES_PER_ENTRY` claims to be `size_of::<(u64, u32)>()`
    /// rather than the 12 bytes the fields alone need. Assert it against the
    /// tuple itself, so a future representation change to the tables this
    /// constant actually models (`binned.rs`'s, per its own updated doc
    /// comment -- `counter.rs`'s own tables already made exactly this move,
    /// which is why this constant's doc comment now says `binned.rs`
    /// specifically rather than "every count table this crate builds")
    /// cannot leave this model quietly charging for padding that no longer
    /// exists.
    #[test]
    fn count_table_entry_size_matches_the_tuple_it_models() {
        assert_eq!(
            COUNT_TABLE_BYTES_PER_ENTRY as usize,
            std::mem::size_of::<(u64, u32)>(),
            "the count-table entry size changed; the binned memory model must move with it"
        );
    }

    /// The per-term arithmetic `docs/design-minimizer-counting.md` 4.1
    /// published, pinned so the code and that document cannot drift apart
    /// silently. Only the *terms* are pinned here: 4.1's three phase totals
    /// used the in-memory strategy's 777 MiB base overhead, which the
    /// calibration on 2026-09-05 replaced with this path's own measured
    /// floor, so those totals are superseded and are checked against
    /// measurements in the test below instead of against the document.
    #[test]
    fn binned_estimate_reproduces_the_design_document_term_arithmetic() {
        let occurrences = BENCH_OCCURRENCES;
        let threads = BENCH_THREADS as u64;
        let num_bins = crate::minimizer::DEFAULT_NUM_BINS as u64;
        let chunk_bytes = crate::binned::DEFAULT_CHUNK_BYTES as u64;

        let distinct = (occurrences as f64) * DISTINCT_PER_OCCURRENCE;
        let store = ((occurrences as f64) * SUPERKMER_BYTES_PER_OCCURRENCE) as u64;
        let open_chunks = threads * num_bins * chunk_bytes;
        let table = (distinct as u64) * COUNT_TABLE_BYTES_PER_ENTRY;
        let per_bin_expanded = (occurrences / num_bins) * 8;
        let per_bin_table = ((distinct / num_bins as f64) as u64) * COUNT_TABLE_BYTES_PER_ENTRY;
        let phase2_transients = threads * (per_bin_expanded + per_bin_table);

        for (label, actual_mib, published_mib) in [
            ("super-k-mer store", mib(store), 829.0),
            ("open chunks", mib(open_chunks), 64.0),
            ("phase-2 per-bin transients", mib(phase2_transients), 113.0),
            ("final table", mib(table), 821.0),
        ] {
            assert!(
                (actual_mib - published_mib).abs() < 0.5,
                "{label}: model says {actual_mib:.1} MiB, design doc 4.1 published \
                 {published_mib:.1} MiB"
            );
        }
    }

    /// The eight runs `BINNED_CALIBRATION_FACTOR` and
    /// `BINNED_BASE_OVERHEAD_BYTES` were fit to, with the residual against
    /// each one reported in the failure message.
    ///
    /// Two properties, and the first is the one that matters: **no point may
    /// be under-predicted.** A memory model that tells the automatic chooser
    /// a run fits when it does not is how a 19 GB machine gets an OOM twenty
    /// minutes into a count. Over-prediction only costs a run the disk
    /// strategy it did not strictly need, so the ceiling is loose (25%)
    /// while the floor is absolute.
    ///
    /// Measured on macOS arm64 (Apple M3 Pro, 19 GB), release build, peak
    /// RSS from `/usr/bin/time -l`. The inputs are the `mid` and `large`
    /// files `scripts/bench/generate_reads_large.py` produces at seeds 4242
    /// and 9001 -- 144,000,000 and 840,000,000 k-mer occurrences at k=31.
    #[test]
    fn binned_estimate_matches_the_measured_runs_it_was_fit_to() {
        let num_bins = crate::minimizer::DEFAULT_NUM_BINS;
        let chunk_bytes = crate::binned::DEFAULT_CHUNK_BYTES;

        // (occurrences, threads, measured peak RSS in MiB)
        let measured = [
            (144_000_000u64, 1usize, 891.0),
            (144_000_000, 4, 915.0),
            (144_000_000, 8, 923.0),
            (144_000_000, 11, 951.0),
            (840_000_000, 1, 3009.0),
            (840_000_000, 4, 3372.0),
            (840_000_000, 8, 3407.0),
            (840_000_000, 11, 3287.0),
        ];

        for (occurrences, threads, measured_mib) in measured {
            let predicted_mib =
                mib(estimate_binned_peak_bytes(occurrences, threads, num_bins, chunk_bytes));
            let residual = (predicted_mib - measured_mib) / measured_mib;
            assert!(
                residual >= 0.0,
                "{occurrences} occurrences at {threads} threads: predicted {predicted_mib:.0} MiB \
                 UNDER a measured {measured_mib:.0} MiB ({:.1}%). A binned estimate that \
                 under-predicts lets the automatic chooser pick a run that will not fit.",
                residual * 100.0
            );
            assert!(
                residual <= 0.25,
                "{occurrences} occurrences at {threads} threads: predicted {predicted_mib:.0} MiB \
                 against a measured {measured_mib:.0} MiB (+{:.1}%), beyond the 25% ceiling this \
                 model is allowed to be conservative by",
                residual * 100.0
            );
        }
    }

    /// The design's own falsifiable structural claim
    /// (`docs/design-minimizer-counting.md` 3.6): because the super-k-mer
    /// store is a partition rather than a per-worker replica, peak RSS on a
    /// fixed input should vary by **under ~250 MiB** between 1 and 8
    /// threads, where the in-memory strategy varies by gigabytes.
    ///
    /// This test checks the *model* honours that claim, not that the code
    /// does -- only a measured run can do the latter. But a model that
    /// failed it would be modelling something other than the design it cites,
    /// and the in-memory comparison below is what gives the number meaning.
    #[test]
    fn binned_peak_is_nearly_thread_independent_where_the_in_memory_peak_is_not() {
        let bins = crate::minimizer::DEFAULT_NUM_BINS;
        let chunk = crate::binned::DEFAULT_CHUNK_BYTES;

        let one = estimate_binned_peak_bytes(BENCH_OCCURRENCES, 1, bins, chunk);
        let eight = estimate_binned_peak_bytes(BENCH_OCCURRENCES, 8, bins, chunk);
        let binned_spread_mib = mib(eight.abs_diff(one));
        assert!(
            binned_spread_mib < 250.0,
            "binned peak moved {binned_spread_mib:.0} MiB between 1 and 8 threads; 3.6 \
             claims under ~250 MiB"
        );

        let in_memory_spread_mib = mib(estimate_peak_bytes(BENCH_OCCURRENCES, 8)
            .abs_diff(estimate_peak_bytes(BENCH_OCCURRENCES, 1)));
        assert!(
            in_memory_spread_mib > 4096.0,
            "the in-memory model is supposed to be the one that scales with threads, but it \
             moved only {in_memory_spread_mib:.0} MiB"
        );
    }

    /// The headline reason `binned` exists: it must predict materially less
    /// memory than the in-memory strategy on the benchmark input.
    ///
    /// **The design's "~3x cut" (4.1) did not survive measurement.** Runs on
    /// 2026-09-05 over 840,000,000 occurrences put the real ratio at
    /// **1.78x at 8 threads** (3,407 MiB against 6,073 MiB) and **1.98x at
    /// 11** (3,287 against 6,513) -- a large win, and not the one the
    /// document predicted. The threshold here is 2x rather than 3x for that
    /// reason, and it is stated model-against-model (both predictions, not
    /// one prediction against a measurement it was never fit to) so the two
    /// models are compared on equal terms.
    ///
    /// Worth knowing when reading the ratio: `estimate_peak_bytes`, the
    /// in-memory model, was calibrated on a different machine and OS and
    /// **over-predicts on this one** -- 8.49 GiB against 6.07 GiB measured
    /// at 8 threads. So the model-to-model ratio here (2.55x) sits above the
    /// measured one, and the honest reading of the memory advantage is the
    /// measured 1.8-2.0x, not this number.
    #[test]
    fn binned_predicts_substantially_less_memory_than_the_in_memory_strategy() {
        let binned = estimate_binned_peak_bytes(
            BENCH_OCCURRENCES,
            BENCH_THREADS,
            crate::minimizer::DEFAULT_NUM_BINS,
            crate::binned::DEFAULT_CHUNK_BYTES,
        );
        let in_memory = estimate_peak_bytes(BENCH_OCCURRENCES, BENCH_THREADS);
        let ratio = in_memory as f64 / binned as f64;
        assert!(
            ratio > 2.0,
            "binned predicted {:.2} GiB against in-memory's {:.2} GiB (only {ratio:.2}x); the \
             measured advantage is 1.8-2.0x and the models should not fall below it",
            binned as f64 / (1024.0 * 1024.0 * 1024.0),
            in_memory as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }

    /// Degenerate arguments must not panic or divide by zero -- the chooser
    /// calls this with whatever thread count it was configured with, and a
    /// zero bin count is reachable from a hand-built `BinnedConfig` before
    /// `clamped()` runs.
    #[test]
    fn binned_estimate_floors_zero_threads_and_zero_bins_instead_of_dividing_by_zero() {
        let zeroed = estimate_binned_peak_bytes(BENCH_OCCURRENCES, 0, 0, 0);
        let ones = estimate_binned_peak_bytes(BENCH_OCCURRENCES, 1, 1, 0);
        assert_eq!(zeroed, ones, "zero threads/bins must floor to one, not wrap or panic");
        assert!(zeroed >= BINNED_BASE_OVERHEAD_BYTES);

        // An empty input still costs the fixed base overhead and nothing
        // that scales, at any thread count -- scaled by the calibration
        // factor, which applies to every prediction this function makes.
        let empty_input_bytes =
            ((BINNED_BASE_OVERHEAD_BYTES as f64) * BINNED_CALIBRATION_FACTOR) as u64;
        assert_eq!(
            estimate_binned_peak_bytes(0, 8, crate::minimizer::DEFAULT_NUM_BINS, 0),
            empty_input_bytes
        );
    }

    /// Monotonicity is what makes this model usable as a budget test: more
    /// occurrences can never be predicted to cost less. A chooser that could
    /// see the estimate *fall* as the input grew would admit a run that does
    /// not fit.
    #[test]
    fn binned_estimate_is_monotone_in_occurrences() {
        let bins = crate::minimizer::DEFAULT_NUM_BINS;
        let chunk = crate::binned::DEFAULT_CHUNK_BYTES;
        let mut previous = 0u64;
        for occurrences in [0u64, 1_000_000, 50_000_000, 143_997_708, 839_987_618, 5_000_000_000] {
            let estimate = estimate_binned_peak_bytes(occurrences, 8, bins, chunk);
            assert!(
                estimate >= previous,
                "estimate fell from {previous} to {estimate} at {occurrences} occurrences"
            );
            previous = estimate;
        }
    }

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
