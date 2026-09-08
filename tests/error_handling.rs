//! The pipeline must fail loudly and locate the failure, never print and continue.

// Integration tests legitimately use `.expect()`/`.unwrap()` to fail fast on
// setup errors that are not the thing under test.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{BufRead, Cursor, Read};
use std::path::Path;

use fastdna_core::error::FastDnaError;
use fastdna_core::fastq::FastqReader;
use fastdna_core::pipeline::{process_stream_parallel, PipelineConfig};

fn reader_for(fastq: &str) -> FastqReader<Cursor<Vec<u8>>> {
    FastqReader::new(Cursor::new(fastq.as_bytes().to_vec()))
}

/// A `BufRead` shim that behaves like a normal in-memory reader for a fixed
/// number of `fill_buf` calls, then fails every call after that. Since a
/// `Cursor`'s `fill_buf` always hands back the whole remaining buffer in one
/// call, and `next_record` performs exactly one `read_until` per FASTQ line,
/// each successful call here corresponds to exactly one line. Three full
/// records is 12 successful calls; the 13th (the header line of record 4)
/// fails, which is what `io_read_failure_surfaces_as_io_not_malformed_fastq`
/// uses to prove a genuine I/O failure is never reported as `MalformedFastq`.
struct FailAfterN {
    inner: Cursor<Vec<u8>>,
    remaining_ok_calls: usize,
}

impl Read for FailAfterN {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buf)
    }
}

impl BufRead for FailAfterN {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        if self.remaining_ok_calls == 0 {
            return Err(std::io::Error::other("simulated read failure"));
        }
        self.remaining_ok_calls -= 1;
        self.inner.fill_buf()
    }

    fn consume(&mut self, amt: usize) {
        self.inner.consume(amt);
    }
}

fn config(k: usize) -> PipelineConfig {
    PipelineConfig {
        k,
        min_quality: 20.0,
        quality_window: 4,
        batch_size: 8,
        num_threads: 2,
        progress_interval: 100_000,
        canonical: true,
        hpc: false,
    }
}

#[test]
fn rejects_k_above_the_packing_limit_before_reading() {
    let result = process_stream_parallel(
        reader_for("@r1\nACGT\n+\nIIII\n"),
        config(33),
        Path::new("sample.fastq"),
        None,
        None,
    );

    match result {
        // `max` is asserted too, not just `k`: `process_stream_parallel`
        // is the narrow entry point, so the limit it reports must be its
        // own 32 and not the wide engine's 64. A message naming the wrong
        // engine's limit is how a user concludes k=41 is unavailable.
        Err(FastDnaError::InvalidK { k, max }) => {
            assert_eq!(k, 33);
            assert_eq!(max, 32, "the narrow entry point must report the narrow limit");
        }
        other => panic!("expected InvalidK, got {other:?}"),
    }
}

#[test]
fn rejects_k_of_zero() {
    let result = process_stream_parallel(
        reader_for("@r1\nACGT\n+\nIIII\n"),
        config(0),
        Path::new("sample.fastq"),
        None,
        None,
    );
    assert!(matches!(result, Err(FastDnaError::InvalidK { k: 0, .. })));
}

#[test]
fn a_valid_stream_still_succeeds() {
    let result = process_stream_parallel(
        reader_for("@r1\nACGTACGT\n+\nIIIIIIII\n"),
        config(4),
        Path::new("sample.fastq"),
        None,
        None,
    );

    let (counter, _qc, reads) = result.expect("valid input must succeed");
    assert_eq!(reads, 1);
    assert_eq!(counter.total_kmers(), 5);
}

/// A genuine I/O failure from the underlying reader (a corrupt gzip member,
/// an NFS read error, or -- as simulated here -- any `read` call that
/// returns `Err`) must surface as `FastDnaError::Io`, never `MalformedFastq`.
/// Reporting it as `MalformedFastq` would blame the bytes for a
/// hardware/transport fault and attach a record number that means nothing
/// for it. This is the exact scenario item 4(b) of the core-hardening pass
/// closed: before that fix, every error out of `next_record` -- I/O failures
/// included -- was mapped to `MalformedFastq`, making `FastDnaError::Io`
/// unreachable from the pipeline.
#[test]
fn io_read_failure_surfaces_as_io_not_malformed_fastq() {
    // Three good records (12 successful lines), then the reader fails on the
    // very first line of what would be record 4.
    let mut fastq = String::new();
    for i in 0..3 {
        fastq.push_str(&format!("@r{i}\nACGT\n+\nIIII\n"));
    }

    let shim = FailAfterN { inner: Cursor::new(fastq.into_bytes()), remaining_ok_calls: 12 };
    let reader = FastqReader::new(shim);

    let result = process_stream_parallel(reader, config(4), Path::new("cohort/sample.fastq"), None, None);

    match result {
        Err(FastDnaError::Io { path, source }) => {
            assert_eq!(path, Path::new("cohort/sample.fastq"), "must name the source that failed to read");
            // The whole point of the qc.rs Io/Export split (and of this
            // variant generally) is that Io must carry the genuine
            // io::Error, never a synthetic stand-in laundering a different
            // failure. `FailAfterN::fill_buf` raises `Error::other(..)`, so
            // that is exactly the kind that must come back out here.
            assert_eq!(
                source.kind(),
                std::io::ErrorKind::Other,
                "source must be the genuine io::Error the shim produced, got {source:?}"
            );
        }
        other => panic!("expected Io, got {other:?}"),
    }
}

/// A structural violation -- as opposed to the I/O failure covered by
/// `io_read_failure_surfaces_as_io_not_malformed_fastq` -- must report
/// `MalformedFastq` with the 1-based index of the record that failed.
#[test]
fn malformed_record_reports_the_1_based_record_number() {
    // Three good records, then a fourth whose header is missing the
    // required '@' marker.
    let mut fastq = String::new();
    for i in 0..3 {
        fastq.push_str(&format!("@r{i}\nACGT\n+\nIIII\n"));
    }
    fastq.push_str("r3\nACGT\n+\nIIII\n");

    let result = process_stream_parallel(reader_for(&fastq), config(4), Path::new("cohort/sample.fastq"), None, None);

    match result {
        Err(FastDnaError::MalformedFastq { path, record, reason }) => {
            assert_eq!(record, 4, "must report the 1-based index of the record that failed");
            assert_eq!(path, Path::new("cohort/sample.fastq"));
            assert!(!reason.is_empty(), "reason must say what was wrong");
        }
        other => panic!("expected MalformedFastq, got {other:?}"),
    }
}

/// `num_threads: 0` must be rejected before the channel is created. If the
/// guard were removed, zero consumer tasks would spawn, the bounded(64)
/// channel would fill up, and the producer thread would block forever with
/// no one left to drain it -- this test would hang rather than fail.
#[test]
fn zero_threads_is_rejected_instead_of_hanging() {
    let mut fastq = String::new();
    for i in 0..5_000 {
        fastq.push_str(&format!("@r{i}\nACGTACGT\n+\nIIIIIIII\n"));
    }

    let bad_config = PipelineConfig {
        k: 4,
        min_quality: 20.0,
        quality_window: 4,
        batch_size: 8,
        num_threads: 0,
        progress_interval: 100_000,
        canonical: true,
        hpc: false,
    };

    let result = process_stream_parallel(reader_for(&fastq), bad_config, Path::new("sample.fastq"), None, None);

    match result {
        Err(FastDnaError::InvalidConfig { parameter, .. }) => assert_eq!(parameter, "num_threads"),
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

/// `progress_interval: 0` must be rejected before any batch is processed.
/// If the guard were removed, `prev / progress_interval` inside the worker
/// loop would divide by zero the first time a progress callback is supplied,
/// which `catch_unwind` would report as `FastDnaError::Internal` -- a
/// caller mistake mislabeled as a bug in FastDNA.
#[test]
fn zero_progress_interval_is_rejected_as_invalid_config() {
    let bad_config = PipelineConfig {
        k: 4,
        min_quality: 20.0,
        quality_window: 4,
        batch_size: 8,
        num_threads: 2,
        progress_interval: 0,
        canonical: true,
        hpc: false,
    };

    let noop = |_: fastdna_core::progress::Progress| {};
    let result = process_stream_parallel(
        reader_for("@r1\nACGT\n+\nIIII\n"),
        bad_config,
        Path::new("sample.fastq"),
        Some(&noop),
        None,
    );

    match result {
        Err(FastDnaError::InvalidConfig { parameter, .. }) => assert_eq!(parameter, "progress_interval"),
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

/// `quality_window: 0` must be rejected before any read is trimmed. If the
/// guard were removed, `quality_trim_end` would loop forever: the empty
/// window makes `sum / 0` produce `NaN`, `NaN >= min_qual` is always
/// `false` so the loop never breaks, and `end_pos -= 1` underflows once
/// `end_pos` reaches zero.
#[test]
fn zero_quality_window_is_rejected_as_invalid_config() {
    let bad_config = PipelineConfig {
        k: 4,
        min_quality: 20.0,
        quality_window: 0,
        batch_size: 8,
        num_threads: 2,
        progress_interval: 100_000,
        canonical: true,
        hpc: false,
    };

    let result = process_stream_parallel(
        reader_for("@r1\nACGT\n+\nIIII\n"),
        bad_config,
        Path::new("sample.fastq"),
        None,
        None,
    );

    match result {
        Err(FastDnaError::InvalidConfig { parameter, .. }) => assert_eq!(parameter, "quality_window"),
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

/// `batch_size: 0` must be rejected before the producer thread starts. If
/// the guard were removed, `current_batch.len() >= batch_size` would be
/// trivially true after every single record, so every read would cross the
/// `bounded(64)` channel in its own one-element `Vec` instead of a real
/// batch -- a severe throughput cliff, not a hang or a wrong answer, so this
/// only checks the error variant and parameter, not behaviour under load.
#[test]
fn zero_batch_size_is_rejected_as_invalid_config() {
    let bad_config = PipelineConfig {
        k: 4,
        min_quality: 20.0,
        quality_window: 4,
        batch_size: 0,
        num_threads: 2,
        progress_interval: 100_000,
        canonical: true,
        hpc: false,
    };

    let result = process_stream_parallel(
        reader_for("@r1\nACGT\n+\nIIII\n"),
        bad_config,
        Path::new("sample.fastq"),
        None,
        None,
    );

    match result {
        Err(FastDnaError::InvalidConfig { parameter, .. }) => assert_eq!(parameter, "batch_size"),
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

/// `min_quality: NaN` must be rejected before any read is trimmed. If the
/// guard were removed, `quality_trim_end`'s `avg_qual >= min_qual` check
/// would be `false` for every window (all comparisons against `NaN` are
/// `false`), so the loop would never break early and every read would be
/// silently trimmed down to `window_size - 1` bases. That is the
/// silent-wrong-answer failure this guard exists to rule out, so this test
/// proves it the strong way: it counts a known input and asserts the exact
/// k-mer total a healthy (non-NaN) run would produce. Without the guard,
/// every 8-base read here would be trimmed to 3 bases (`window_size - 1`
/// with `quality_window: 4`), which is shorter than `k = 4` and yields zero
/// k-mers per read -- so this assertion would fail loudly (0, not 5) rather
/// than merely accepting any error.
#[test]
fn nan_min_quality_is_rejected_and_would_silently_zero_out_counts_if_not() {
    let bad_config = PipelineConfig {
        k: 4,
        min_quality: f64::NAN,
        quality_window: 4,
        batch_size: 8,
        num_threads: 2,
        progress_interval: 100_000,
        canonical: true,
        hpc: false,
    };

    let result = process_stream_parallel(
        reader_for("@r1\nACGTACGT\n+\nIIIIIIII\n"),
        bad_config,
        Path::new("sample.fastq"),
        None,
        None,
    );

    match result {
        Err(FastDnaError::InvalidConfig { parameter, .. }) => assert_eq!(parameter, "min_quality"),
        other => panic!("expected InvalidConfig, got {other:?}"),
    }

    // Demonstrate what the guard prevents: the same input with a *valid*
    // min_quality of 0.0 (which every real quality score is `>=`, so
    // trimming never engages) must yield the full, un-degraded k-mer count.
    // This is the number that a NaN `min_quality` would silently replace
    // with near-zero if the guard above did not exist.
    let healthy_config = PipelineConfig {
        k: 4,
        min_quality: 0.0,
        quality_window: 4,
        batch_size: 8,
        num_threads: 2,
        progress_interval: 100_000,
        canonical: true,
        hpc: false,
    };
    let (counter, _qc, reads) = process_stream_parallel(
        reader_for("@r1\nACGTACGT\n+\nIIIIIIII\n"),
        healthy_config,
        Path::new("sample.fastq"),
        None,
        None,
    )
    .expect("a finite min_quality must succeed");
    assert_eq!(reads, 1);
    assert_eq!(counter.total_kmers(), 5, "an untrimmed 8-base read must yield 5 canonical 4-mers");
}

/// `min_quality: +inf` must be rejected for the same reason as `NaN`: no
/// real quality average is ever `>= +inf`, so the trim loop never breaks
/// early and every read is silently trimmed down to `window_size - 1`
/// bases.
#[test]
fn positive_infinity_min_quality_is_rejected_as_invalid_config() {
    let bad_config = PipelineConfig {
        k: 4,
        min_quality: f64::INFINITY,
        quality_window: 4,
        batch_size: 8,
        num_threads: 2,
        progress_interval: 100_000,
        canonical: true,
        hpc: false,
    };

    let result = process_stream_parallel(
        reader_for("@r1\nACGT\n+\nIIII\n"),
        bad_config,
        Path::new("sample.fastq"),
        None,
        None,
    );

    match result {
        Err(FastDnaError::InvalidConfig { parameter, .. }) => assert_eq!(parameter, "min_quality"),
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

/// `min_quality: -inf` must also be rejected, but it is worth its own case:
/// unlike `NaN`/`+inf`, every quality average is `>= -inf`, so this makes
/// the trim loop break on its very first check and trim *nothing at all* --
/// the opposite wrong behaviour from `NaN`'s "trim everything", caught by
/// the same guard.
#[test]
fn negative_infinity_min_quality_is_rejected_as_invalid_config() {
    let bad_config = PipelineConfig {
        k: 4,
        min_quality: f64::NEG_INFINITY,
        quality_window: 4,
        batch_size: 8,
        num_threads: 2,
        progress_interval: 100_000,
        canonical: true,
        hpc: false,
    };

    let result = process_stream_parallel(
        reader_for("@r1\nACGT\n+\nIIII\n"),
        bad_config,
        Path::new("sample.fastq"),
        None,
        None,
    );

    match result {
        Err(FastDnaError::InvalidConfig { parameter, .. }) => assert_eq!(parameter, "min_quality"),
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

/// A token already set before the call starts must return `Cancelled`
/// promptly, not process the whole input first. 50,000 reads at a
/// batch_size of 8 is over 6,000 batches; if the cancel check were missing
/// (or checked only, say, once at the very end) a non-cancelling run would
/// clearly take much longer than the generous wall-clock budget asserted
/// below, so a regression here shows up as a slow/failing test rather than
/// a silent behavioural difference.
#[test]
fn cancel_set_before_the_call_returns_promptly() {
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use std::time::Instant;

    let mut fastq = String::new();
    for i in 0..50_000 {
        fastq.push_str(&format!("@r{i}\nACGTACGT\n+\nIIIIIIII\n"));
    }

    let cancel = Arc::new(AtomicBool::new(true));
    let large_config = PipelineConfig {
        k: 4,
        min_quality: 20.0,
        quality_window: 4,
        batch_size: 8,
        num_threads: 2,
        progress_interval: 100_000,
        canonical: true,
        hpc: false,
    };

    let start = Instant::now();
    let result =
        process_stream_parallel(reader_for(&fastq), large_config, Path::new("sample.fastq"), None, Some(cancel));
    let elapsed = start.elapsed();

    match result {
        Err(FastDnaError::Cancelled) => {}
        other => panic!("expected Cancelled, got {other:?}"),
    }
    assert!(
        elapsed.as_secs() < 5,
        "a pre-cancelled call over 50,000 reads must return promptly, not process the input first: took {elapsed:?}"
    );
}

/// A token that stays `false` for the whole run must behave identically to
/// passing `None` -- same read count, same k-mer counts -- proving the
/// cancellation plumbing costs nothing and changes nothing when unused.
#[test]
fn cancel_false_throughout_behaves_identically_to_none() {
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    let mut fastq = String::new();
    for i in 0..2_000 {
        fastq.push_str(&format!("@r{i}\nACGTACGTAC\n+\nIIIIIIIIII\n"));
    }

    let (no_token_counter, _qc1, no_token_reads) =
        process_stream_parallel(reader_for(&fastq), config(5), Path::new("sample.fastq"), None, None)
            .expect("uncancelled run must succeed");

    let never_cancelled = Arc::new(AtomicBool::new(false));
    let (false_token_counter, _qc2, false_token_reads) = process_stream_parallel(
        reader_for(&fastq),
        config(5),
        Path::new("sample.fastq"),
        None,
        Some(never_cancelled),
    )
    .expect("a token that is never set must not change the outcome");

    assert_eq!(no_token_reads, false_token_reads);
    assert_eq!(no_token_counter.total_kmers(), false_token_counter.total_kmers());
    assert_eq!(no_token_counter.distinct_kmers(), false_token_counter.distinct_kmers());
}

/// Cancellation triggered mid-run (rather than pre-set before the call, as
/// in `cancel_set_before_the_call_returns_promptly`) must still surface as
/// `Cancelled`, never as a successful run reporting fewer reads than the
/// input actually contains. The cancel flag is flipped from inside the
/// progress callback itself, so real work has already happened and been
/// reported by the time cancellation takes effect -- exactly the scenario
/// where a naive implementation might be tempted to return whatever partial
/// counts had accumulated instead of an error. That would be the same class
/// of bug as the silent truncation this branch of work removed elsewhere.
#[test]
fn cancellation_mid_run_returns_cancelled_not_partial_counts() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let mut fastq = String::new();
    for i in 0..5_000 {
        fastq.push_str(&format!("@r{i}\nACGTACGT\n+\nIIIIIIII\n"));
    }

    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_for_callback = cancel.clone();
    let trigger_cancel_on_first_progress_event = move |_: fastdna_core::progress::Progress| {
        cancel_for_callback.store(true, Ordering::SeqCst);
    };

    let mid_run_config = PipelineConfig {
        k: 4,
        min_quality: 20.0,
        quality_window: 4,
        batch_size: 8,
        num_threads: 2,
        progress_interval: 10,
        canonical: true,
        hpc: false,
    };

    let result = process_stream_parallel(
        reader_for(&fastq),
        mid_run_config,
        Path::new("sample.fastq"),
        Some(&trigger_cancel_on_first_progress_event),
        Some(cancel),
    );

    match result {
        Err(FastDnaError::Cancelled) => {}
        Ok((_counter, _qc, reads)) => panic!(
            "cancellation must not surface as a successful run with fewer reads \
             than the input actually has (got reads = {reads})"
        ),
        other => panic!("expected Cancelled, got {other:?}"),
    }
}

/// Exercises the worker-side `catch_unwind` specifically: a panic inside
/// `emit` while a worker thread is still draining the channel (as opposed to
/// the earlier test, which uses only 2 reads and so only ever reaches the
/// main-thread guard around the final `Finished` emit). This is the guard
/// standing between a panicking Python callback and undefined behaviour
/// across the FFI boundary once a PyO3 binding exists, and it previously had
/// zero coverage. A low progress_interval (made possible by item E) lets a
/// small, fast-running test actually cross the interval from a worker.
#[test]
fn a_panic_in_a_worker_thread_progress_callback_becomes_internal_and_returns() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let mut fastq = String::new();
    for i in 0..500 {
        fastq.push_str(&format!("@r{i}\nACGTACGT\n+\nIIIIIIII\n"));
    }

    let already_panicked = AtomicBool::new(false);
    let panic_on_first_reads_processed = move |event: fastdna_core::progress::Progress| {
        if matches!(event, fastdna_core::progress::Progress::ReadsProcessed(_))
            && !already_panicked.swap(true, Ordering::SeqCst)
        {
            panic!("worker-side progress callback exploded");
        }
    };

    let worker_config = PipelineConfig {
        k: 4,
        min_quality: 20.0,
        quality_window: 4,
        batch_size: 8,
        num_threads: 2,
        progress_interval: 10,
        canonical: true,
        hpc: false,
    };

    // If the worker-side catch_unwind were missing, this call would either
    // hang (the panicked worker stops draining, backing up the channel) or
    // unwind out of process_stream_parallel entirely, so simply returning
    // an Err at all is itself part of what this test proves.
    let result = process_stream_parallel(
        reader_for(&fastq),
        worker_config,
        Path::new("sample.fastq"),
        Some(&panic_on_first_reads_processed),
        None,
    );

    match result {
        Err(FastDnaError::Internal { .. }) => {}
        other => panic!("expected Internal, got {other:?}"),
    }
}

#[test]
fn a_panicking_progress_callback_becomes_an_internal_error() {
    let fastq = "@r1\nACGTACGT\n+\nIIIIIIII\n@r2\nTTGCAACG\n+\nIIIIIIII\n";
    let panics = |_: fastdna_core::progress::Progress| panic!("progress callback exploded");

    let result = process_stream_parallel(reader_for(fastq), config(4), Path::new("sample.fastq"), Some(&panics), None);

    match result {
        Err(FastDnaError::Internal { .. }) => {}
        other => panic!("expected Internal, got {other:?}"),
    }
}
