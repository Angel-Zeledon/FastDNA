// src/pipeline.rs

use std::io::BufRead;
use std::thread;
use crossbeam_channel::{bounded, Receiver, Sender};
use rayon::prelude::*;

use crate::counter::KmerCounter;
use crate::fastq::{FastqReader, FastqRecord};
use crate::kmer;
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

pub fn process_stream_parallel<R: BufRead + Send + 'static>(
    reader: FastqReader<R>,
    config: PipelineConfig,
) -> (KmerCounter, QcSummary, u64) {
    let (sender, receiver): (Sender<RecordBatch>, Receiver<RecordBatch>) = bounded(64);

    let batch_size = config.batch_size;
    let k = config.k;
    let min_qual = config.min_quality;
    let qual_win = config.quality_window;

    // 1. Producer Thread
    let reader_handle = thread::spawn(move || {
        let mut reader = reader;
        let mut current_batch = Vec::with_capacity(batch_size);
        let mut total_reads: u64 = 0;

        loop {
            match reader.next_record() {
                Ok(Some(record)) => {
                    current_batch.push(record);
                    total_reads += 1;

                    if current_batch.len() >= batch_size {
                        let batch_to_send = std::mem::replace(&mut current_batch, Vec::with_capacity(batch_size));
                        if sender.send(batch_to_send).is_err() {
                            break;
                        }
                    }
                }
                Ok(None) => break, // EOF reached
                Err(err) => {
                    eprintln!("\n[FASTQ Stream Error at record #{}] {}", total_reads + 1, err);
                    break;
                }
            }
        }

        if !current_batch.is_empty() {
            let _ = sender.send(current_batch);
        }

        total_reads
    });

    // 2. Parallel Consumer Worker Pool
    let results: Vec<(KmerCounter, QcSummary)> = (0..config.num_threads)
        .into_par_iter()
        .map(|_| {
            let mut local_counter = KmerCounter::with_capacity(131_072);
            let mut local_qc = QcSummary::default();

            while let Ok(mut batch) = receiver.recv() {
                for record in &mut batch {
                    local_qc.observe_record(record);
                    record.quality_trim_end(min_qual, qual_win);
                    
                    // ==========================================
                    // FEATURE 1: Lógica Canónica (Reverso Complementario)
                    // ==========================================
                    let mut canon_kmers = Vec::new();
                    
                    // Verificamos que la lectura sea al menos tan grande como K
                    if record.seq.len() >= k {
                        // Deslizamos la "ventana" de tamaño K sobre toda la secuencia de ADN
                        for i in 0..=record.seq.len() - k {
                            let window = &record.seq[i..i + k];
                            // Extraemos la versión biológica correcta (la menor alfabéticamente)
                            let canonico = crate::bio::canonical_kmer(window);
                            canon_kmers.push(canonico);
                        }
                    }
                    
                    // Insertamos el lote de k-mers canónicos puros en la memoria
                    local_counter.insert_batch(&canon_kmers);
                }
            }

            (local_counter, local_qc)
        })
        .collect();

    let total_reads = reader_handle.join().unwrap_or(0);

    // 3. Map-Reduce Combine Phase
    let (master_counter, mut master_qc) = results
        .into_par_iter()
        .reduce(
            || (KmerCounter::new(), QcSummary::default()),
            |(mut acc_cnt, mut acc_qc), (local_cnt, local_qc)| {
                acc_cnt.merge(local_cnt);
                acc_qc.merge(&local_qc);
                (acc_cnt, acc_qc)
            },
        );

    master_qc.finalize();
    (master_counter, master_qc, total_reads)
}