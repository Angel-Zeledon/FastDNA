// src/metagenomics.rs
//! Kraken2-style metagenomic read classification: a canonical-k-mer ->
//! lowest-common-ancestor database, and a per-read classifier over it.
//!
//! The algorithm is the one described in Wood, Lu & Langmead, "Improved
//! metagenomic analysis with Kraken 2", *Genome Biology* 20:257 (2019):
//! every k-mer of a reference set is mapped to the lowest common ancestor
//! (LCA) of every taxon it occurs in, a read is looked up k-mer by k-mer,
//! and the read is assigned by scoring every root-to-leaf path through the
//! taxa it hit. See `KmerDatabase::classify_sequence` for the exact rule
//! and `KmerDatabase::build` for the LCA construction.
//!
//! # What this is not: no parity claim with Kraken 2
//!
//! Kraken 2 is a mature, heavily validated, heavily optimized tool and this
//! module has not been benchmarked against it -- not for speed, not for
//! sensitivity, not for precision. Nothing here should be read as a claim
//! of equivalence. Concretely, the following parts of Kraken 2 are **not**
//! implemented, each with what its absence costs:
//!
//! - **Minimizers.** Kraken 2 stores one database entry per distinct
//!   *minimizer* (default: the smallest 31-mer inside each 35-mer window),
//!   not per distinct k-mer. This module stores one entry per distinct
//!   canonical k-mer. That is several times more entries for the same
//!   reference set -- the single largest reason the scale ceiling below is
//!   where it is.
//! - **Spaced seeds.** Kraken 2 masks 7 positions of each minimizer before
//!   hashing, so a single substitution does not necessarily destroy the
//!   match. Exact k-mers have no such tolerance: one SNP kills up to `k`
//!   consecutive k-mers, so this module is less sensitive on strains that
//!   diverge from the reference than Kraken 2 is.
//! - **A compact hash table.** Kraken 2 stores a truncated key hash plus a
//!   taxon id packed into a single 32-bit cell. This module stores exact
//!   64-bit keys (see "Memory cost" below).
//! - **Multithreaded classification.** `classify_source` is a single
//!   streaming loop. Building a database is single-threaded too.
//! - **Paired-end handling.** Kraken 2 classifies a read pair as one unit,
//!   joining the mates with a spacer so no k-mer spans the junction. Here,
//!   mates are two independent reads.
//! - **Kraken's report format.** [`KmerDatabase::abundance_batch`] reports
//!   direct assignments, not the clade-cumulative percentages of
//!   `kraken2-report`, and there is no `--report-minimizer-data`
//!   equivalent.
//! - **Bracken-style abundance correction** -- see `abundance_batch`.
//!
//! # Memory cost, measured, and the scale it implies
//!
//! The lookup table is two parallel arrays, `Vec<u64>` of canonical k-mers
//! and `Vec<u32>` of taxon ids, so the resident cost is exactly
//! [`BYTES_PER_KMER`] = **12 bytes per distinct k-mer** with no per-entry
//! overhead at all (a `Vec<(u64, u32)>` would be 16, since the tuple is
//! padded to its 8-byte alignment; that padding is the whole reason for the
//! parallel-array layout). `KmerDatabase::memory_bytes` reports the live
//! figure and `resident_memory_is_twelve_bytes_per_kmer` pins it.
//!
//! Building costs more than holding: construction accumulates into an
//! `FxHashMap<u64, u32>`, whose entry is a padded 16-byte pair plus one
//! control byte per bucket, in a table that grows to the next power of two
//! at 87.5% load -- so **17 to 34 bytes per distinct k-mer at peak**,
//! averaging around 25, plus the 12-byte sorted table built from it before
//! the map is dropped. Budget roughly **3x** the final size to build one.
//!
//! What that means in practice, at k=31:
//!
//! | Reference set | Distinct k-mers | Resident | To build |
//! |---|---|---|---|
//! | 1 bacterial genome (~4 Mbp) | ~4 M | ~48 MB | ~150 MB |
//! | 20 bacterial genomes | ~80 M | ~1.0 GB | ~3 GB |
//! | 150 bacterial genomes | ~600 M | ~7.2 GB | ~21 GB |
//! | RefSeq complete bacteria (~10^11 k-mers) | ~10^11 | **~1.2 TB** | -- |
//!
//! **So: this does not scale to RefSeq, and you must not plan around it
//! doing so.** On an ordinary 16 GB workstation the practical ceiling is
//! somewhere around 150-200 million distinct k-mers -- roughly 40 to 50
//! bacterial genomes -- and that is the *build* limit, which is the binding
//! one. Kraken 2's standard database covers all of RefSeq bacteria,
//! archaea and viruses in ~8.7 GB precisely because of the minimizer and
//! compact-hash choices listed above, neither of which this module makes.
//! Use this for a targeted panel (a pathogen set, a mock community, a
//! host-plus-contaminants screen), not for an open-world survey.
//!
//! The path forward is already open, and is why the on-disk format is a
//! *sorted* fixed-record table (see `KmerDatabase::save`): the on-disk
//! entry layout is byte-identical to the in-memory one, so a disk-backed
//! lookup is a change of accessor -- `mmap` the entry region and binary
//! search it in place -- rather than a change of format. That is not
//! implemented here.
//!
//! # Relationship to `python/fastdna/taxonomy.py`
//!
//! That module answers a different question with a different instrument:
//! it compares whole-sample MinHash sketches (containment, Jaccard,
//! `gather`) to rank which *references* best explain a *sample*. It is
//! approximate by construction, needs no taxonomy tree, and never looks at
//! an individual read. This module assigns *each read* to a *node of a
//! taxonomy*, exactly, from a database that must be built first. They
//! complement each other: sketch-based `gather` is the cheap "who is
//! probably in here" pass, this is the per-read assignment that a
//! composition table or a host-removal step needs.

use std::collections::hash_map::Entry;
use std::path::Path;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use crate::error::{FastDnaError, Result};

/// The taxon id reported for a read that could not be classified.
///
/// Zero, following Kraken's own convention, and reserved: the taxonomy
/// parser rejects a row that tries to define it (see
/// `tax_id_zero_is_reserved_and_rejected_by_row`), because otherwise an
/// unclassified read and a read assigned to that taxon would be
/// indistinguishable in the output table. It doubles as the "no parent"
/// sentinel above the root, which is what makes the confidence-promotion
/// walk in `classify_sequence` terminate instead of spinning on NCBI's
/// self-parented root node.
pub const UNCLASSIFIED_TAX_ID: u32 = 0;

/// Resident bytes per distinct k-mer in a built [`KmerDatabase`]: 8 for the
/// canonical k-mer plus 4 for its taxon id, with no per-entry overhead.
/// See this module's "Memory cost" section for what that implies and for
/// the (higher) peak cost of building one.
pub const BYTES_PER_KMER: usize = 12;

/// One node of the taxonomy tree.
///
/// `parent_tax_id` is [`UNCLASSIFIED_TAX_ID`] for the root and only for the
/// root -- the input format lets the root name itself as its own parent
/// (NCBI's convention), and the parser normalizes that away so that no
/// ancestor walk anywhere in this module needs a special case for it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Taxon {
    pub tax_id: u32,
    pub parent_tax_id: u32,
    pub rank: String,
    pub name: String,
}

/// A validated taxonomy tree: every node has a parent that exists, exactly
/// one node is the root, and no node is its own ancestor.
///
/// Those three properties are established once, by [`TaxonomyFile::parse`]
/// or by [`Taxonomy::from_taxa`], and then assumed by every walk in this
/// module. That is deliberate: `is_ancestor_or_self` runs once per pair of
/// hit taxa per read, and a cycle check on every one of those walks would
/// be paid millions of times to catch a problem that can only be introduced
/// at the file boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Taxonomy {
    /// Sorted by `tax_id`, so that serializing a taxonomy is deterministic
    /// and `save`/`load` round-trips byte for byte.
    taxa: Vec<Taxon>,
    root: u32,
    /// `tax_id -> index into taxa`. Rebuilt on deserialization rather than
    /// stored, since it is derivable and storing it would let a
    /// hand-edited file disagree with itself.
    #[serde(skip)]
    index: FxHashMap<u32, usize>,
}

impl Taxonomy {
    /// Builds and validates a taxonomy from nodes that already carry
    /// normalized parents (root's parent is [`UNCLASSIFIED_TAX_ID`]).
    ///
    /// `row_of` maps a `tax_id` back to the 1-based input line that defined
    /// it, so that a structural failure discovered here -- which is only
    /// discoverable *after* every row has been read -- can still be blamed
    /// on the row that caused it. A caller with no file behind it (a test,
    /// a programmatic build) passes an empty map and gets the tax id alone.
    fn from_taxa(taxa: Vec<Taxon>, row_of: &FxHashMap<u32, usize>) -> std::result::Result<Self, String> {
        let mut taxa = taxa;
        taxa.sort_unstable_by_key(|t| t.tax_id);

        let mut index: FxHashMap<u32, usize> = FxHashMap::default();
        for (i, taxon) in taxa.iter().enumerate() {
            index.insert(taxon.tax_id, i);
        }

        let roots: Vec<u32> = taxa
            .iter()
            .filter(|t| t.parent_tax_id == UNCLASSIFIED_TAX_ID)
            .map(|t| t.tax_id)
            .collect();
        let root = match roots.as_slice() {
            [only] => *only,
            [] => {
                return Err(
                    "no root: exactly one row must have parent_tax_id 0 or a parent_tax_id equal \
                     to its own tax_id"
                        .to_string(),
                )
            }
            [first, second, ..] => {
                let (a, b) = if row_of.get(first) <= row_of.get(second) {
                    (*first, *second)
                } else {
                    (*second, *first)
                };
                return Err(format!(
                    "{}: tax_id {b} is a second root; row {} already made tax_id {a} the root, \
                     and a taxonomy must have exactly one",
                    describe_row(row_of.get(&b).copied(), b),
                    row_of.get(&a).map(|r| r.to_string()).unwrap_or_else(|| "?".to_string()),
                ));
            }
        };

        // Every parent must exist, and every node must reach the root. The
        // two failures are checked together in one walk per node because
        // they are the same walk: a node whose walk neither reaches the
        // root nor hits a missing parent is, by elimination, in a cycle.
        for taxon in &taxa {
            let mut seen: FxHashSet<u32> = FxHashSet::default();
            let mut current = taxon.tax_id;
            loop {
                if !seen.insert(current) {
                    return Err(format!(
                        "{}: tax_id {} is its own ancestor -- the parent chain from it forms a \
                         cycle through tax_id {current} instead of reaching the root",
                        describe_row(row_of.get(&taxon.tax_id).copied(), taxon.tax_id),
                        taxon.tax_id,
                    ));
                }
                let Some(&i) = index.get(&current) else {
                    // `current` is always a *parent* here: the first
                    // iteration looks up `taxon.tax_id`, which is in the
                    // index by construction. Whether it is the row's own
                    // parent or one further up decides which of these two
                    // wordings actually helps the reader.
                    let where_ = describe_row(row_of.get(&taxon.tax_id).copied(), taxon.tax_id);
                    return Err(if current == taxon.parent_tax_id {
                        format!(
                            "{where_}: parent_tax_id {current} of tax_id {} is not defined by any row",
                            taxon.tax_id
                        )
                    } else {
                        format!(
                            "{where_}: the parent chain of tax_id {} reaches tax_id {current}, \
                             which is not defined by any row",
                            taxon.tax_id
                        )
                    });
                };
                let parent = taxa[i].parent_tax_id;
                if parent == UNCLASSIFIED_TAX_ID {
                    break;
                }
                current = parent;
            }
        }

        Ok(Taxonomy { taxa, root, index })
    }

    /// The single root node's id.
    pub fn root(&self) -> u32 {
        self.root
    }

    /// How many taxa the tree holds.
    pub fn len(&self) -> usize {
        self.taxa.len()
    }

    pub fn is_empty(&self) -> bool {
        self.taxa.is_empty()
    }

    /// Every taxon, ascending by `tax_id`.
    pub fn taxa(&self) -> &[Taxon] {
        &self.taxa
    }

    fn get(&self, tax_id: u32) -> Option<&Taxon> {
        self.index.get(&tax_id).map(|&i| &self.taxa[i])
    }

    pub fn contains(&self, tax_id: u32) -> bool {
        self.index.contains_key(&tax_id)
    }

    pub fn name_of(&self, tax_id: u32) -> Option<&str> {
        self.get(tax_id).map(|t| t.name.as_str())
    }

    pub fn rank_of(&self, tax_id: u32) -> Option<&str> {
        self.get(tax_id).map(|t| t.rank.as_str())
    }

    /// The parent of `tax_id`, or [`UNCLASSIFIED_TAX_ID`] if it is the
    /// root. `None` only when `tax_id` is not in the tree at all -- the two
    /// are different answers and callers (notably the promotion walk in
    /// `classify_sequence`) depend on telling them apart.
    pub fn parent_of(&self, tax_id: u32) -> Option<u32> {
        self.get(tax_id).map(|t| t.parent_tax_id)
    }

    /// The root-to-node path, root first. Empty if `tax_id` is unknown.
    pub fn lineage(&self, tax_id: u32) -> Vec<u32> {
        let mut path = Vec::new();
        let mut current = tax_id;
        while current != UNCLASSIFIED_TAX_ID {
            let Some(taxon) = self.get(current) else {
                // Only reachable for an id that is not in the tree, since
                // validation ruled out dangling parents.
                return Vec::new();
            };
            path.push(current);
            current = taxon.parent_tax_id;
        }
        path.reverse();
        path
    }

    /// Whether `ancestor` is on the root-to-`descendant` path, inclusive of
    /// `descendant` itself. Walks up from `descendant` rather than down
    /// from `ancestor`: a node knows its parent, and a downward walk would
    /// need a child index that exists only to answer this.
    pub fn is_ancestor_or_self(&self, ancestor: u32, descendant: u32) -> bool {
        if ancestor == UNCLASSIFIED_TAX_ID || descendant == UNCLASSIFIED_TAX_ID {
            return false;
        }
        let mut current = descendant;
        while current != UNCLASSIFIED_TAX_ID {
            if current == ancestor {
                return true;
            }
            match self.get(current) {
                Some(taxon) => current = taxon.parent_tax_id,
                None => return false,
            }
        }
        false
    }

    /// The lowest common ancestor of `a` and `b` -- the deepest node that is
    /// an ancestor-or-self of both.
    ///
    /// This is the operation the whole database rests on: a k-mer occurring
    /// in two references must map to the taxon that covers both, not to
    /// whichever of them happened to be read first. Getting it wrong does
    /// not produce a visible error, it produces a systematic bias toward
    /// the first reference in the file, which is why `build` funnels every
    /// collision through this one function.
    ///
    /// An unknown id on either side yields the other side (and
    /// [`UNCLASSIFIED_TAX_ID`] if both are unknown): folding LCA over a set
    /// of taxa must not have an unknown member silently pull the answer to
    /// the root.
    pub fn lca(&self, a: u32, b: u32) -> u32 {
        if a == b {
            return a;
        }
        let path_a = self.lineage(a);
        let path_b = self.lineage(b);
        if path_a.is_empty() {
            return b;
        }
        if path_b.is_empty() {
            return a;
        }
        let mut best = UNCLASSIFIED_TAX_ID;
        for (x, y) in path_a.iter().zip(path_b.iter()) {
            if x != y {
                break;
            }
            best = *x;
        }
        best
    }
}

/// Renders the "row N" prefix of a validation message, degrading to the tax
/// id alone when the caller had no file (and therefore no row) behind it.
fn describe_row(row: Option<usize>, tax_id: u32) -> String {
    match row {
        Some(row) => format!("row {row}"),
        None => format!("tax_id {tax_id}"),
    }
}

/// The required columns of a taxonomy TSV, in the order they are listed in
/// this module's format documentation. Order in the *file* is free -- the
/// parser resolves every column by header name.
const REQUIRED_COLUMNS: [&str; 4] = ["tax_id", "parent_tax_id", "rank", "name"];

/// The optional fifth column, holding the reference sequence ids that
/// belong to a taxon.
const SEQUENCE_IDS_COLUMN: &str = "sequence_ids";

/// A parsed taxonomy TSV: the tree, plus the reference-sequence-to-taxon
/// mapping carried in the same file.
///
/// # File format
///
/// A tab-separated file with a header line. Required columns, in any order,
/// resolved by name; any further column is ignored:
///
/// | column | meaning |
/// |---|---|
/// | `tax_id` | this taxon's id. A positive integer; 0 is reserved. |
/// | `parent_tax_id` | the parent's `tax_id`. The root names either `0` or itself. |
/// | `rank` | free text, e.g. `species`, `genus`, `no rank`. Reported, never interpreted. |
/// | `name` | free text, the display name. |
/// | `sequence_ids` | *optional*: the reference sequence ids assigned to this taxon, separated by `;` or `,`. Empty for internal nodes. |
///
/// Blank lines and lines beginning with `#` are skipped, but still count
/// toward the line numbers that error messages quote, so "row 6" is the
/// sixth line of the file as an editor shows it.
///
/// Written with `<TAB>` for the separators, since a literal tab would be
/// indistinguishable from alignment spaces once rendered:
///
/// ```text
/// tax_id<TAB>parent_tax_id<TAB>rank<TAB>name<TAB>sequence_ids
/// 1<TAB>1<TAB>no rank<TAB>root
/// 561<TAB>1<TAB>genus<TAB>Escherichia
/// 562<TAB>561<TAB>species<TAB>Escherichia coli<TAB>NC_000913.3;U00096.3
/// ```
///
/// A reference sequence id is the first whitespace-delimited token of a
/// FASTA header, without the `>` -- so `>NC_000913.3 Escherichia coli str.
/// K-12` is `NC_000913.3`. That is the same convention every aligner and
/// `samtools faidx` uses, so an existing `.fai` or a `grep '^>'` gives the
/// exact list to paste in.
///
/// # Relationship to NCBI's `nodes.dmp` / `names.dmp`
///
/// Those files are **not** accepted directly, and the reason is that they
/// are three files, not one, in a format that is not TSV: `nodes.dmp` is
/// `\t|\t`-delimited with the rank in field 3 and no name at all,
/// `names.dmp` carries several names per taxon distinguished by a
/// name-class field (only `scientific name` is usually wanted), and the
/// sequence-to-taxon mapping is a third file entirely
/// (`nucl_gb.accession2taxid`, or Kraken's own `seqid2taxid.map`). Reading
/// all three would mean owning the join, the name-class selection rule, and
/// the accession-versioning rules -- and NCBI's full tree is ~2.5 million
/// nodes, far more than a database this module can hold k-mers for anyway.
///
/// The join is the user's, and it is small: for the taxa you actually have
/// references for, take `tax_id`, `parent tax_id` and `rank` from
/// `nodes.dmp` fields 1-3, the `scientific name` row from `names.dmp`, and
/// your `seqid2taxid.map` lines, and emit one TSV row per taxon. Include
/// every ancestor up to the root -- a parent that is not in the file is a
/// hard error (by design; see `a_parent_that_is_not_in_the_taxonomy_is_rejected_by_row`),
/// not a silently pruned branch.
#[derive(Debug, Clone)]
pub struct TaxonomyFile {
    pub taxonomy: Taxonomy,
    /// Reference sequence id -> the taxon it belongs to.
    pub sequence_to_tax_id: FxHashMap<String, u32>,
}

impl TaxonomyFile {
    /// Reads and validates a taxonomy TSV from disk.
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| FastDnaError::Io { path: path.to_path_buf(), source: e })?;
        Self::parse(&text, path)
    }

    /// Parses and validates taxonomy TSV text. `path` names the file in
    /// error messages only, so this is directly testable against a string
    /// literal without touching the filesystem (same split as
    /// `sketch::GenomeSketch::from_reader`).
    ///
    /// Every failure is a [`FastDnaError::Load`] whose reason names the
    /// 1-based line that caused it. Nothing is ever skipped: a row this
    /// parser cannot make sense of stops the load rather than quietly
    /// shrinking the tree, because a taxonomy that is missing the branch
    /// you cared about still classifies reads -- to the wrong node, with
    /// full confidence, and no error anywhere.
    pub fn parse(text: &str, path: &Path) -> Result<Self> {
        let load_err = |reason: String| FastDnaError::Load { path: path.to_path_buf(), reason };

        let mut lines = text
            .lines()
            .enumerate()
            .map(|(i, line)| (i + 1, line))
            .filter(|(_, line)| !line.trim().is_empty() && !line.trim_start().starts_with('#'));

        let Some((_, header_line)) = lines.next() else {
            return Err(load_err(
                "the file has no header line; expected a tab-separated header naming at least \
                 tax_id, parent_tax_id, rank and name"
                    .to_string(),
            ));
        };

        let header: Vec<&str> = header_line.split('\t').map(|c| c.trim()).collect();
        let column_of = |wanted: &str| header.iter().position(|c| *c == wanted);

        let missing: Vec<&str> =
            REQUIRED_COLUMNS.iter().copied().filter(|c| column_of(c).is_none()).collect();
        if !missing.is_empty() {
            return Err(load_err(format!(
                "the header is missing required column(s): {}. Found: {}",
                missing.join(", "),
                header.join(", ")
            )));
        }
        // Checked immediately above, so these are present; `ok_or` rather
        // than `unwrap` only because `unwrap` is denied crate-wide.
        let idx = |wanted: &str| -> Result<usize> {
            column_of(wanted).ok_or_else(|| load_err(format!("the header is missing column {wanted}")))
        };
        let (col_tax, col_parent, col_rank, col_name) =
            (idx("tax_id")?, idx("parent_tax_id")?, idx("rank")?, idx("name")?);
        let col_seqs = column_of(SEQUENCE_IDS_COLUMN);

        let n_columns = header.len();
        let mut taxa: Vec<Taxon> = Vec::new();
        let mut row_of: FxHashMap<u32, usize> = FxHashMap::default();
        let mut sequence_to_tax_id: FxHashMap<String, u32> = FxHashMap::default();
        let mut sequence_row: FxHashMap<String, usize> = FxHashMap::default();

        for (row, line) in lines {
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < n_columns {
                return Err(load_err(format!(
                    "row {row}: expected {n_columns} tab-separated fields to match the header, \
                     found {}",
                    fields.len()
                )));
            }

            let parse_id = |value: &str, column: &str| -> Result<u32> {
                value.trim().parse::<u32>().map_err(|_| {
                    load_err(format!(
                        "row {row}: {column} {:?} is not a non-negative integer",
                        value.trim()
                    ))
                })
            };

            let tax_id = parse_id(fields[col_tax], "tax_id")?;
            if tax_id == UNCLASSIFIED_TAX_ID {
                return Err(load_err(format!(
                    "row {row}: tax_id 0 is reserved for unclassified reads and cannot name a taxon"
                )));
            }
            if let Some(previous) = row_of.get(&tax_id) {
                return Err(load_err(format!(
                    "row {row}: tax_id {tax_id} is already defined on row {previous}"
                )));
            }

            let raw_parent = parse_id(fields[col_parent], "parent_tax_id")?;
            // NCBI's root is its own parent; normalize that to the
            // no-parent sentinel so no ancestor walk needs a special case.
            let parent_tax_id = if raw_parent == tax_id { UNCLASSIFIED_TAX_ID } else { raw_parent };

            row_of.insert(tax_id, row);
            taxa.push(Taxon {
                tax_id,
                parent_tax_id,
                rank: fields[col_rank].trim().to_string(),
                name: fields[col_name].trim().to_string(),
            });

            if let Some(col) = col_seqs {
                for sequence_id in split_sequence_ids(fields[col]) {
                    match sequence_to_tax_id.entry(sequence_id.to_string()) {
                        Entry::Occupied(existing) => {
                            let previous = sequence_row.get(sequence_id).copied().unwrap_or(0);
                            return Err(load_err(format!(
                                "row {row}: sequence id {sequence_id:?} is already assigned to \
                                 tax_id {} on row {previous}",
                                existing.get()
                            )));
                        }
                        Entry::Vacant(slot) => {
                            slot.insert(tax_id);
                            sequence_row.insert(sequence_id.to_string(), row);
                        }
                    }
                }
            }
        }

        if taxa.is_empty() {
            return Err(load_err(
                "the file has a header but no taxon rows; a taxonomy needs at least a root"
                    .to_string(),
            ));
        }

        let taxonomy = Taxonomy::from_taxa(taxa, &row_of).map_err(load_err)?;
        Ok(TaxonomyFile { taxonomy, sequence_to_tax_id })
    }

    /// The 1-based rows sequence ids came from, for build-time errors that
    /// need to point back at the taxonomy file.
    pub fn sequence_ids(&self) -> impl Iterator<Item = (&str, u32)> {
        self.sequence_to_tax_id.iter().map(|(id, tax)| (id.as_str(), *tax))
    }
}

/// Splits a `sequence_ids` cell. Both `;` and `,` are accepted because
/// accession lists are written both ways in the wild and guessing wrong
/// would not fail loudly -- it would produce one long non-existent id,
/// whose only symptom is a "sequence has no tax_id" error pointing at the
/// reference rather than at the real problem here.
fn split_sequence_ids(cell: &str) -> impl Iterator<Item = &str> {
    cell.split([';', ',']).map(|s| s.trim()).filter(|s| !s.is_empty())
}

#[cfg(test)]
// Same rationale as the other in-module test blocks: unwrap/expect denial is
// about production paths, not test assertions.
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The three-taxon toy tree every classification test in this module and
    /// in `tests/metagenomics.rs` is anchored on:
    ///
    /// ```text
    ///   1 root
    ///   └── 10 genus  Toyella
    ///       ├── 100 species  Toyella alpha
    ///       └── 200 species  Toyella beta
    /// ```
    const TOY_TAXONOMY: &str = "\
tax_id\tparent_tax_id\trank\tname\tsequence_ids
1\t1\tno rank\troot\t
10\t1\tgenus\tToyella\t
100\t10\tspecies\tToyella alpha\tspecies_a
200\t10\tspecies\tToyella beta\tspecies_b
";

    fn toy() -> TaxonomyFile {
        TaxonomyFile::parse(TOY_TAXONOMY, std::path::Path::new("<toy>")).expect("valid toy taxonomy")
    }

    #[test]
    fn a_well_formed_taxonomy_parses_every_row() {
        let parsed = toy();
        assert_eq!(parsed.taxonomy.len(), 4);
        assert_eq!(parsed.taxonomy.name_of(100), Some("Toyella alpha"));
        assert_eq!(parsed.taxonomy.rank_of(10), Some("genus"));
        assert_eq!(parsed.taxonomy.root(), 1);
        // The root's parent is normalized to the reserved 0, so walking up
        // from any node terminates at a value that means "no taxon" rather
        // than looping on the root's self-reference.
        assert_eq!(parsed.taxonomy.parent_of(1), Some(UNCLASSIFIED_TAX_ID));
        assert_eq!(parsed.sequence_to_tax_id.get("species_a"), Some(&100));
        assert_eq!(parsed.sequence_to_tax_id.get("species_b"), Some(&200));
    }

    #[test]
    fn lca_of_two_sibling_species_is_their_genus() {
        let t = toy().taxonomy;
        assert_eq!(t.lca(100, 200), 10);
        assert_eq!(t.lca(200, 100), 10, "lca must be symmetric");
    }

    #[test]
    fn lca_of_a_node_with_its_own_ancestor_is_the_ancestor() {
        let t = toy().taxonomy;
        assert_eq!(t.lca(100, 10), 10);
        assert_eq!(t.lca(10, 100), 10);
        assert_eq!(t.lca(100, 100), 100);
    }

    #[test]
    fn ancestry_is_directional() {
        let t = toy().taxonomy;
        assert!(t.is_ancestor_or_self(10, 100), "the genus is an ancestor of the species");
        assert!(!t.is_ancestor_or_self(100, 10), "the species is not an ancestor of its genus");
        assert!(t.is_ancestor_or_self(100, 100), "every node is its own ancestor-or-self");
        assert!(!t.is_ancestor_or_self(100, 200), "siblings are unrelated");
    }

    // -- validation: each failure must name the offending row -------------

    /// Asserts that `text` is rejected and that the message mentions every
    /// one of `must_mention` -- the row number first of all, since "which
    /// line do I go fix" is the whole point of these errors.
    fn reject(text: &str, must_mention: &[&str]) -> String {
        match TaxonomyFile::parse(text, std::path::Path::new("bad_taxonomy.tsv")) {
            Ok(_) => panic!("expected this taxonomy to be rejected:\n{text}"),
            Err(FastDnaError::Load { path, reason }) => {
                assert_eq!(path, std::path::PathBuf::from("bad_taxonomy.tsv"));
                for needle in must_mention {
                    assert!(reason.contains(needle), "message must mention {needle:?}: {reason}");
                }
                reason
            }
            Err(other) => panic!("expected Load, got {other:?}"),
        }
    }

    #[test]
    fn a_parent_that_is_not_in_the_taxonomy_is_rejected_by_row() {
        reject(
            "tax_id\tparent_tax_id\trank\tname\n1\t1\tno rank\troot\n100\t99\tspecies\tOrphan\n",
            &["row 3", "99", "100"],
        );
    }

    #[test]
    fn a_cycle_in_the_tree_is_rejected_by_row() {
        // 100 -> 200 -> 100: no row reaches the root.
        let reason = reject(
            "tax_id\tparent_tax_id\trank\tname\n1\t1\tno rank\troot\n100\t200\tspecies\tA\n200\t100\tspecies\tB\n",
            &["100"],
        );
        assert!(reason.contains("cycle") || reason.contains("ancestor"), "reason: {reason}");
    }

    #[test]
    fn a_duplicate_tax_id_is_rejected_by_row() {
        reject(
            "tax_id\tparent_tax_id\trank\tname\n1\t1\tno rank\troot\n100\t1\tspecies\tA\n100\t1\tspecies\tB\n",
            &["row 4", "100", "row 3"],
        );
    }

    #[test]
    fn a_duplicate_sequence_id_is_rejected_by_row() {
        reject(
            "tax_id\tparent_tax_id\trank\tname\tsequence_ids\n1\t1\tno rank\troot\t\n100\t1\tspecies\tA\tseq1\n200\t1\tspecies\tB\tseq1\n",
            &["row 4", "seq1", "row 3"],
        );
    }

    /// Tax id 0 is how a classified-nothing read is reported (Kraken's own
    /// convention), so a taxonomy that also uses it would make an
    /// unclassified read indistinguishable from a classified one.
    #[test]
    fn tax_id_zero_is_reserved_and_rejected_by_row() {
        reject(
            "tax_id\tparent_tax_id\trank\tname\n1\t1\tno rank\troot\n0\t1\tspecies\tA\n",
            &["row 3", "reserved"],
        );
    }

    #[test]
    fn a_taxonomy_with_no_root_is_rejected() {
        // Every row has a parent inside the file, but none is its own.
        reject(
            "tax_id\tparent_tax_id\trank\tname\n1\t10\tno rank\ta\n10\t1\tno rank\tb\n",
            &["root"],
        );
    }

    #[test]
    fn a_second_root_is_rejected_by_row() {
        reject(
            "tax_id\tparent_tax_id\trank\tname\n1\t1\tno rank\troot\n2\t2\tno rank\tother root\n",
            &["row 3", "root"],
        );
    }

    #[test]
    fn a_missing_required_column_is_rejected_by_name() {
        reject("tax_id\tparent_tax_id\trank\n1\t1\tno rank\n", &["name"]);
    }

    #[test]
    fn a_non_numeric_tax_id_is_rejected_by_row() {
        reject(
            "tax_id\tparent_tax_id\trank\tname\n1\t1\tno rank\troot\nabc\t1\tspecies\tA\n",
            &["row 3", "abc"],
        );
    }

    #[test]
    fn a_short_row_is_rejected_by_row() {
        reject("tax_id\tparent_tax_id\trank\tname\n1\t1\tno rank\n", &["row 2"]);
    }

    #[test]
    fn an_empty_taxonomy_file_is_rejected() {
        reject("", &["header"]);
    }

    /// Comment and blank lines are skipped, but they still advance the line
    /// counter -- an error must point at the line a text editor shows.
    #[test]
    fn comments_and_blank_lines_are_skipped_without_shifting_row_numbers() {
        reject(
            "# a comment\ntax_id\tparent_tax_id\trank\tname\n\n1\t1\tno rank\troot\n\n100\t99\tspecies\tOrphan\n",
            &["row 6"],
        );
    }
}
