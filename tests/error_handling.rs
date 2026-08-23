//! The pipeline must fail loudly and locate the failure, never print and continue.

// Integration tests legitimately use `.expect()`/`.unwrap()` to fail fast on
// setup errors that are not the thing under test.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{BufRead, Cursor, Read};
use std::path::Path;

use fastdna::error::FastDnaError;
use fastdna::fastq::FastqReader;
use fastdna::pipeline::{process_stream_parallel, PipelineConfig};

fn reader_for(fastq: &str) -> FastqReader<Cursor<Vec<u8>>> {
    FastqReader::new(Cursor::new(fastq.as_bytes().to_vec()))
}

/// A `BufRead` shim that behaves like a normal in-memory reader for a fixed
/// number of `fill_buf` calls, then fails every call after that. Since a
/// `Cursor`'s `fill_buf` always hands back the whole remaining buffer in one
/// call, and `next_record` performs exactly one `read_until` per FASTQ line,
/// each successful call here corresponds to exactly one line. Three full
/// records is 12 successful calls; the 13th (the header line of record 4)
/// fails, which is what pins the 1-based record arithmetic in
/// `malformed_record_reports_the_1_based_record_number`.
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
        Err(FastDnaError::InvalidK { k }) => assert_eq!(k, 33),
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
    assert!(matches!(result, Err(FastDnaError::InvalidK { k: 0 })));
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

#[test]
fn malformed_record_reports_the_1_based_record_number() {
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
        Err(FastDnaError::MalformedFastq { path, record, .. }) => {
            assert_eq!(record, 4, "must report the 1-based index of the record that failed");
            assert_eq!(path, Path::new("cohort/sample.fastq"));
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
    };

    let result = process_stream_parallel(reader_for(&fastq), bad_config, Path::new("sample.fastq"), None, None);

    match result {
        Err(FastDnaError::InvalidConfig { parameter, .. }) => assert_eq!(parameter, "num_threads"),
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
    let trigger_cancel_on_first_progress_event = move |_: fastdna::progress::Progress| {
        cancel_for_callback.store(true, Ordering::SeqCst);
    };

    let mid_run_config = PipelineConfig {
        k: 4,
        min_quality: 20.0,
        quality_window: 4,
        batch_size: 8,
        num_threads: 2,
        progress_interval: 10,
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
    let panic_on_first_reads_processed = move |event: fastdna::progress::Progress| {
        if matches!(event, fastdna::progress::Progress::ReadsProcessed(_))
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
    let panics = |_: fastdna::progress::Progress| panic!("progress callback exploded");

    let result = process_stream_parallel(reader_for(fastq), config(4), Path::new("sample.fastq"), Some(&panics), None);

    match result {
        Err(FastDnaError::Internal { .. }) => {}
        other => panic!("expected Internal, got {other:?}"),
    }
}
