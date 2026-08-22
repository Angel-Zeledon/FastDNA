// src/pipeline.rs

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use crossbeam_channel::{bounded, Receiver, Sender};
use rayon::prelude::*;

use crate::counter::KmerCounter;
use crate::error::{FastDnaError, Result};
use crate::fastq::{FastqReader, FastqRecord};
use crate::kmer;
use crate::progress::{Progress, ProgressFn, PROGRESS_INTERVAL};
use crate::qc::QcSummary;

pub struct PipelineConfig {
    pub k: usize,
    pub min_quality: f64,
    pub quality_window: usize,
    pub batch_size: usize,
    pub num_threads: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            k: 31,
            min_quality: 20.0,
            quality_window: 4,
            batch_size: 8192,
            num_threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
        }
    }
}

type RecordBatch = Vec<FastqRecord>;

/// Streams a FASTQ source and returns its canonical k-mer counts.
///
/// `source` names the input for error messages only; in-memory callers pass
/// `Path::new("<memory>")`. `progress` is optional; `None` means silence.
pub fn process_stream_parallel<R: BufRead + Send + 'static>(
    reader: FastqReader<R>,
    config: PipelineConfig,
    source: &Path,
    progress: ProgressFn<'_>,
) -> Result<(KmerCounter, QcSummary, u64)> {
    if config.k == 0 || config.k > 32 {
        return Err(FastDnaError::InvalidK { k: config.k });
    }

    let (sender, receiver): (Sender<RecordBatch>, Receiver<RecordBatch>) = bounded(64);

    let batch_size = config.batch_size;
    let k = config.k;
    let min_qual = config.min_quality;
    let qual_win = config.quality_window;
    let source_owned: PathBuf = source.to_path_buf();

    // 1. Producer thread. Returns the read count, or the record it choked on.
    let reader_handle = thread::spawn(move || -> Result<u64> {
        let mut reader = reader;
        let mut current_batch = Vec::with_capacity(batch_size);
        let mut total_reads: u64 = 0;

        loop {
            match reader.next_record() {
                Ok(Some(record)) => {
                    current_batch.push(record);
                    total_reads += 1;

                    if current_batch.len() >= batch_size {
                        let batch_to_send =
                            std::mem::replace(&mut current_batch, Vec::with_capacity(batch_size));
                        if sender.send(batch_to_send).is_err() {
                            break;
                        }
                    }
                }
                Ok(None) => break,
                Err(err) => {
                    return Err(FastDnaError::MalformedFastq {
                        path: source_owned,
                        record: total_reads + 1,
                        reason: err.to_string(),
                    });
                }
            }
        }

        if !current_batch.is_empty() {
            let _ = sender.send(current_batch);
        }

        Ok(total_reads)
    });

    // Shared across workers so `ReadsProcessed` is genuinely cumulative.
    // A per-worker counter would report roughly reads/num_threads and jump
    // around non-monotonically. This cannot live in the producer thread
    // instead: `thread::spawn` demands `'static` and `ProgressFn<'a>` is a
    // borrow, so the callback can only be used from the rayon closures, which
    // borrow rather than move.
    let reads_seen = AtomicU64::new(0);

    // 2. Parallel consumer pool. Each worker owns private state, so the hot
    //    path has no locks and no shared hashmap.
    let results: Vec<(KmerCounter, QcSummary)> = (0..config.num_threads)
        .into_par_iter()
        .map(|_| {
            let mut local_counter = KmerCounter::with_capacity(131_072);
            let mut local_qc = QcSummary::default();

            while let Ok(mut batch) = receiver.recv() {
                for record in &mut batch {
                    local_qc.observe_record(record);
                    record.quality_trim_end(min_qual, qual_win);

                    // Canonical k-mers: 2-bit packed, O(1) rolling window, and
                    // ambiguous bases ('N') reset the window rather than
                    // producing corrupt k-mers.
                    let canon_kmers = kmer::extract_canonical_kmers(&record.seq, k);
                    local_counter.insert_batch(&canon_kmers);

                    if let Some(emit) = progress {
                        let seen = reads_seen.fetch_add(1, Ordering::Relaxed) + 1;
                        if seen % PROGRESS_INTERVAL == 0 {
                            emit(Progress::ReadsProcessed(seen));
                        }
                    }
                }
            }

            (local_counter, local_qc)
        })
        .collect();

    let total_reads = reader_handle
        .join()
        .map_err(|_| FastDnaError::Internal { detail: "FASTQ reader thread panicked".to_string() })??;

    // 3. Map-reduce combine phase.
    let (master_counter, mut master_qc) = results.into_par_iter().reduce(
        || (KmerCounter::new(), QcSummary::default()),
        |(mut acc_cnt, mut acc_qc), (local_cnt, local_qc)| {
            acc_cnt.merge(local_cnt);
            acc_qc.merge(&local_qc);
            (acc_cnt, acc_qc)
        },
    );

    master_qc.finalize();

    if let Some(emit) = progress {
        emit(Progress::Finished { reads: total_reads });
    }

    Ok((master_counter, master_qc, total_reads))
}
