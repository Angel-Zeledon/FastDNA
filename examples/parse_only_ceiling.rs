//! How much of a counting run is the single-threaded FASTQ producer?
//!
//! `docs/design-minimizer-counting.md` step 0 gated the whole super-k-mer
//! design on this number: if the producer alone accounted for more than
//! ~36% of a run, the right fix was parallel parsing rather than a better
//! counting strategy. It measured 4.7% then, and the gate passed.
//!
//! That fraction is not a constant. The counting half got 2.4x faster when
//! the binned strategy became the default (`docs/BENCHMARKS.md`), and the
//! producer did not change at all, so its share of the total has grown by
//! roughly the same factor. This tool re-measures it directly rather than
//! inferring it: parse every record and discard it, with no channel, no
//! workers and no k-mer extraction -- a hard upper bound on what any
//! counting strategy could ever save.
//!
//! ```text
//! cargo run --release --example parse_only_ceiling -- FILE
//! ```
use std::time::Instant;

use fastdna_core::fastq::{FastqReader, FastqRecord};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: parse_only_ceiling FILE")?;
    let bytes = std::fs::metadata(&path)?.len();

    let mut reader = FastqReader::from_path(&path)?;
    // Ids are the one field a counting run never reads; leaving them on
    // would measure a cost the pipeline does not pay.
    reader.set_keep_ids(false);

    let mut record = FastqRecord::default();
    let mut records: u64 = 0;
    let mut bases: u64 = 0;
    let started = Instant::now();
    while reader.next_record_into(&mut record)? {
        records += 1;
        bases += record.seq.len() as u64;
    }
    let elapsed = started.elapsed().as_secs_f64();

    println!("{path}");
    println!("  {records} records, {bases} bases, {:.2} GB", bytes as f64 / 1e9);
    println!("  parse only: {elapsed:.2} s  ({:.0} MB/s)", bytes as f64 / 1e6 / elapsed);
    Ok(())
}
