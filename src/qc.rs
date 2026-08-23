// src/qc.rs

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use crate::error::{FastDnaError, Result};
use crate::fastq::FastqRecord;

/// Quality Control (QC) summary report for sequencing health.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct QcSummary {
    pub total_reads: u64,
    pub total_bases: u64,
    pub q20_bases: u64,
    pub q30_bases: u64,
    pub gc_bases: u64,
    pub gc_content_pct: f64,
    pub q20_pct: f64,
    pub q30_pct: f64,
}

impl QcSummary {
    pub fn observe_record(&mut self, record: &FastqRecord) {
        self.total_reads += 1;
        self.total_bases += record.seq.len() as u64;

        for &b in &record.seq {
            if b == b'G' || b == b'g' || b == b'C' || b == b'c' {
                self.gc_bases += 1;
            }
        }

        for &q in &record.qual {
            let score = FastqRecord::phred_score(q);
            if score >= 20 { self.q20_bases += 1; }
            if score >= 30 { self.q30_bases += 1; }
        }
    }

    pub fn merge(&mut self, other: &QcSummary) {
        self.total_reads += other.total_reads;
        self.total_bases += other.total_bases;
        self.q20_bases += other.q20_bases;
        self.q30_bases += other.q30_bases;
        self.gc_bases += other.gc_bases;
    }

    pub fn finalize(&mut self) {
        if self.total_bases > 0 {
            let total = self.total_bases as f64;
            self.gc_content_pct = (self.gc_bases as f64 / total) * 100.0;
            self.q20_pct = (self.q20_bases as f64 / total) * 100.0;
            self.q30_pct = (self.q30_bases as f64 / total) * 100.0;
        }
    }

    pub fn export_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let to_err = |e: std::io::Error| FastDnaError::Io { path: path.to_path_buf(), source: e };

        let file = File::create(path).map_err(to_err)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, self).map_err(|e| FastDnaError::Io {
            path: path.to_path_buf(),
            source: std::io::Error::other(e),
        })?;
        writer.flush().map_err(to_err)?;
        Ok(())
    }
}
