// src/wasm.rs

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use crate::fastq::FastqReader;
#[cfg(feature = "wasm")]
use crate::qc::QcSummary;
#[cfg(feature = "wasm")]
use crate::counter::KmerCounter;
#[cfg(feature = "wasm")]
use crate::kmer;
#[cfg(feature = "wasm")]
use std::io::Cursor;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn analyze_fastq_wasm(fastq_text: &str, k: usize) -> Result<JsValue, JsValue> {
    let mut reader = FastqReader::new(Cursor::new(fastq_text.as_bytes()));
    let mut counter = KmerCounter::with_capacity(4096);
    let mut qc = QcSummary::default();
    let mut records_read: u64 = 0;

    loop {
        match reader.next_record() {
            Ok(Some(mut record)) => {
                records_read += 1;
                qc.observe_record(&record);
                record.quality_trim_end(20.0, 4);
                let kmers = kmer::extract_canonical_kmers(&record.seq, k);
                counter.insert_batch(&kmers);
            }
            Ok(None) => break,
            // Any structural violation -- a missing '@'/'+' marker, a
            // sequence/quality length mismatch, or a file that ends
            // mid-record -- must end the analysis with a real error rather
            // than silently truncating the count returned to the browser.
            // There is no `path` to attach here (the input is an in-memory
            // string, not a file), so the record number and reason are all
            // the context available.
            Err(e) => {
                return Err(JsValue::from_str(&format!(
                    "malformed FASTQ at record {}: {e}",
                    records_read + 1
                )));
            }
        }
    }

    qc.finalize();

    #[derive(serde::Serialize)]
    struct WasmOutput {
        qc: QcSummary,
        distinct_kmers: usize,
        total_kmers: u64,
        top_5_kmers: Vec<(String, u32)>,
    }

    let top_5 = counter
        .top_kmers(5)
        .into_iter()
        .map(|(km, count)| (kmer::decode_kmer(km, k), count))
        .collect();

    let output = WasmOutput {
        qc,
        distinct_kmers: counter.distinct_kmers(),
        total_kmers: counter.total_kmers(),
        top_5_kmers: top_5,
    };

    serde_wasm_bindgen::to_value(&output).map_err(|e| JsValue::from_str(&e.to_string()))
}
