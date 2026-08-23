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

    while let Ok(Some(mut record)) = reader.next_record() {
        qc.observe_record(&record);
        record.quality_trim_end(20.0, 4);
        let kmers = kmer::extract_canonical_kmers(&record.seq, k);
        counter.insert_batch(&kmers);
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
