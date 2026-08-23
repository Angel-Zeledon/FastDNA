// src/fastq.rs

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use flate2::read::GzDecoder;

/// Represents a single FASTQ sequencing record.
pub struct FastqRecord {
    pub id: Vec<u8>,
    pub seq: Vec<u8>,
    pub qual: Vec<u8>,
}

impl FastqRecord {
    #[inline(always)]
    pub fn phred_score(qual_byte: u8) -> u8 {
        qual_byte.saturating_sub(33)
    }

    pub fn quality_trim_end(&mut self, min_qual: f64, window_size: usize) {
        if self.seq.len() < window_size {
            return;
        }

        let mut end_pos = self.seq.len();
        let qual_slice = &self.qual;

        while end_pos >= window_size {
            let start = end_pos - window_size;
            let window = &qual_slice[start..end_pos];

            let sum: u64 = window.iter().map(|&q| Self::phred_score(q) as u64).sum();
            let avg_qual = sum as f64 / window_size as f64;

            if avg_qual >= min_qual {
                break;
            }
            end_pos -= 1;
        }

        self.seq.truncate(end_pos);
        self.qual.truncate(end_pos);
    }
}

/// Streaming FASTQ reader supporting both plain-text and gzip files.
pub struct FastqReader<R: BufRead> {
    reader: R,
    line_buf: Vec<u8>,
}

impl FastqReader<Box<dyn BufRead + Send>> {
    pub fn from_path<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref)?;
        let is_gzipped = path_ref.extension().and_then(|s| s.to_str()) == Some("gz");

        let reader: Box<dyn BufRead + Send> = if is_gzipped {
            Box::new(BufReader::with_capacity(128 * 1024, GzDecoder::new(file)))
        } else {
            Box::new(BufReader::with_capacity(128 * 1024, file))
        };

        Ok(FastqReader {
            reader,
            line_buf: Vec::with_capacity(512),
        })
    }
}

impl<R: BufRead> FastqReader<R> {
    pub fn new(reader: R) -> Self {
        FastqReader {
            reader,
            line_buf: Vec::with_capacity(512),
        }
    }

    pub fn next_record(&mut self) -> io::Result<Option<FastqRecord>> {
        self.line_buf.clear();
        if self.reader.read_until(b'\n', &mut self.line_buf)? == 0 {
            return Ok(None);
        }
        if self.line_buf.ends_with(b"\n") { self.line_buf.pop(); }
        if self.line_buf.ends_with(b"\r") { self.line_buf.pop(); }
        let id = self.line_buf.clone();

        self.line_buf.clear();
        self.reader.read_until(b'\n', &mut self.line_buf)?;
        if self.line_buf.ends_with(b"\n") { self.line_buf.pop(); }
        if self.line_buf.ends_with(b"\r") { self.line_buf.pop(); }
        let seq = self.line_buf.clone();

        self.line_buf.clear();
        self.reader.read_until(b'\n', &mut self.line_buf)?;

        self.line_buf.clear();
        self.reader.read_until(b'\n', &mut self.line_buf)?;
        if self.line_buf.ends_with(b"\n") { self.line_buf.pop(); }
        if self.line_buf.ends_with(b"\r") { self.line_buf.pop(); }
        let qual = self.line_buf.clone();

        Ok(Some(FastqRecord { id, seq, qual }))
    }
}
