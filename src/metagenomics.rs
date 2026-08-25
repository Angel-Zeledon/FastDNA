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
use std::io::Write;
use std::path::Path;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use crate::error::{FastDnaError, Result};
use crate::fastq::{FastqReadError, RecordSource};

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
    /// Each taxon's full root-to-self lineage, parallel to `taxa`, computed
    /// once here rather than walked per query.
    ///
    /// This is the single most-repeated computation in the whole module.
    /// Classifying one read scores every root-to-leaf path through the taxa
    /// it hit, which is `H^2` ancestor tests for `H` distinct hit taxa. A
    /// walk-the-parents implementation costs one hash lookup per step, so
    /// `H^2 * D` hash lookups per read at tree depth `D`. Precomputed, the
    /// same test is a scan of a contiguous slice of at most `D` `u32`s with
    /// no hashing at all. At a typical `H = 3`, `D = 8`, that is 72 hash
    /// lookups saved per read -- half a billion over a 7-million-read file.
    #[serde(skip)]
    lineages: Vec<Vec<u32>>,
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

        // Safe to walk unguarded now: the loop above proved every chain
        // reaches the root without revisiting a node.
        let lineages = taxa
            .iter()
            .map(|taxon| {
                let mut path = Vec::new();
                let mut current = taxon.tax_id;
                while current != UNCLASSIFIED_TAX_ID {
                    path.push(current);
                    current = match index.get(&current) {
                        Some(&i) => taxa[i].parent_tax_id,
                        None => UNCLASSIFIED_TAX_ID,
                    };
                }
                path.reverse();
                path
            })
            .collect();

        Ok(Taxonomy { taxa, root, index, lineages })
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

    /// The root-to-node path, root first and `tax_id` itself last. Empty if
    /// `tax_id` is unknown.
    ///
    /// A borrowed slice of the table `from_taxa` built once, not a fresh
    /// walk: see the `lineages` field for the per-read operation count that
    /// motivates it.
    pub fn lineage(&self, tax_id: u32) -> &[u32] {
        match self.index.get(&tax_id) {
            Some(&i) => &self.lineages[i],
            None => &[],
        }
    }

    /// Whether `ancestor` is on the root-to-`descendant` path, inclusive of
    /// `descendant` itself.
    ///
    /// One contiguous scan of at most `depth(descendant)` `u32`s, with no
    /// hashing after the single index lookup. The alternative -- walking
    /// parent pointers -- costs one hash lookup per step of that same
    /// depth, and this runs `H^2` times per read (see `lineages`).
    pub fn is_ancestor_or_self(&self, ancestor: u32, descendant: u32) -> bool {
        if ancestor == UNCLASSIFIED_TAX_ID {
            return false;
        }
        self.lineage(descendant).contains(&ancestor)
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
        // Both are borrowed slices of the precomputed table, so an LCA
        // costs a zip over at most `depth` u32 comparisons and allocates
        // nothing -- it is folded over every tie in `classify_sequence`
        // and over every k-mer collision during `build`.
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
    /// The 1-based line each sequence id came from, so that a *build*-time
    /// failure -- a sequence named here that the reference never contains --
    /// can still point at the row to fix rather than only at the id.
    sequence_row: FxHashMap<String, usize>,
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
        Ok(TaxonomyFile { taxonomy, sequence_to_tax_id, sequence_row })
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

/// The reference sequence id for a FASTA/FASTQ header: the first
/// whitespace-delimited token, without the leading `>` or `@`. Shared by
/// database construction and by per-read reporting, so a reference and a
/// read are named by exactly the same rule.
fn record_id_of(header: &[u8]) -> String {
    let text = String::from_utf8_lossy(header);
    let trimmed = text.trim_start_matches(['>', '@']);
    trimmed.split_whitespace().next().unwrap_or("").to_string()
}

/// `kmer::extract_canonical_kmers`, writing into a caller-owned buffer
/// instead of returning a fresh `Vec`.
///
/// This exists purely to remove one allocation and one free per sequence.
/// Both the builder and the classifier call it once per record, so on a
/// 7-million-read file it is 14 million heap operations that simply do not
/// happen; the buffer instead reaches the longest read's length within the
/// first few reads and never grows again.
///
/// TODO: delete this and call `kmer::extract_canonical_kmers_into` once
/// that lands in `src/kmer.rs` -- it is being added concurrently in the
/// main tree, and duplicating it here rather than adding a second copy to
/// `kmer.rs` is what keeps that merge trivial.
/// `the_local_kmer_extractor_matches_the_shared_one` pins the two together
/// so this copy cannot drift in the meantime.
fn extract_canonical_kmers_into(seq: &[u8], k: usize, out: &mut Vec<u64>) {
    out.clear();
    if seq.len() < k || k == 0 || k > 32 {
        return;
    }

    let mask = if k == 32 { u64::MAX } else { (1u64 << (2 * k)) - 1 };
    let mut current_kmer: u64 = 0;
    let mut valid_len = 0;

    for &base in seq {
        if let Some(bits) = crate::kmer::base_to_bits(base) {
            current_kmer = ((current_kmer << 2) | bits) & mask;
            valid_len += 1;
            if valid_len >= k {
                out.push(crate::kmer::canonical_kmer_u64(current_kmer, k));
            }
        } else {
            // An ambiguous base resets the window rather than producing a
            // corrupt k-mer -- the same rule the shared extractor applies.
            current_kmer = 0;
            valid_len = 0;
        }
    }
}

// -- the database ---------------------------------------------------------

/// The first 8 bytes of a saved database. Version is carried separately so
/// a future format change is a clear "this file is version 2, I understand
/// version 1" rather than an unrecognizable magic.
const DB_MAGIC: &[u8; 8] = b"FDNAKDB\x00";

/// The on-disk format version this build writes and is willing to read.
const DB_FORMAT_VERSION: u32 = 1;

/// Bytes before the taxonomy JSON: magic(8) + version(4) + k(4) +
/// n_entries(8) + taxonomy_len(8).
const DB_HEADER_BYTES: usize = 32;

/// A canonical-k-mer -> taxon lookup table plus the taxonomy it refers to.
///
/// # Representation, and why it is a sorted table rather than a hash map
///
/// Two parallel arrays -- `Vec<u64>` of canonical k-mers in ascending
/// order, `Vec<u32>` of the taxon each maps to -- searched by
/// `binary_search`. Counted against the obvious alternative, an
/// `FxHashMap<u64, u32>`:
///
/// - **Building is strictly less work.** The sorted table is produced by
///   pushing one `(k-mer, taxon)` pair per k-mer occurrence into one
///   contiguous buffer, sorting it once, and merging equal runs through
///   [`Taxonomy::lca`] in a single linear pass. The hash map does all of
///   that *plus* a hash computation and a probe per occurrence, plus a
///   full rehash of everything inserted so far each time it doubles -- and
///   then still has to be sorted, because the entries have to come out in
///   some order to be written to disk. There is no work the map saves.
/// - **Density.** 12 bytes per entry, exactly, with no slack: see
///   [`BYTES_PER_KMER`]. `hashbrown` runs at up to 87.5% load and doubles
///   at that point, so it averages nearer 22 bytes for the same data (a
///   16-byte padded pair plus a control byte, over a table that is between
///   1.14x and 2.29x the entry count). At this module's documented ceiling
///   of ~100M k-mers that is 1.2 GB against 2.2 GB -- and memory, not
///   speed, is what bounds how large a reference set this module can take.
/// - **The disk path.** The on-disk entry layout is byte-identical to the
///   in-memory one, so an `mmap`-backed lookup is a change of accessor and
///   not of format. A hash table has no such property.
///
/// What that costs: a lookup is `ceil(log2(n))` comparisons -- 27 at 100M
/// entries -- against roughly one probe for a hash map. That is the trade
/// being made deliberately, in favour of fitting a bigger database in RAM.
#[derive(Debug, Clone)]
pub struct KmerDatabase {
    k: usize,
    taxonomy: Taxonomy,
    /// Ascending, deduplicated. `lookup`'s binary search assumes both, and
    /// `load` re-establishes them at the file boundary.
    kmers: Vec<u64>,
    /// Parallel to `kmers`.
    tax_ids: Vec<u32>,
}

impl KmerDatabase {
    /// Builds a database from a reference FASTA (or FASTQ) file and a
    /// taxonomy TSV. See [`TaxonomyFile`] for the taxonomy format.
    pub fn build<P: AsRef<Path>, Q: AsRef<Path>>(reference: P, taxonomy: Q, k: usize) -> Result<Self> {
        let reference = reference.as_ref();
        let taxonomy_path = taxonomy.as_ref();
        let taxonomy_file = TaxonomyFile::read(taxonomy_path)?;
        let source = crate::fastq::MultiSourceReader::from_paths(vec![reference]);
        Self::build_from_source(source, reference, taxonomy_file, taxonomy_path, k)
    }

    /// The construction logic itself, over any [`RecordSource`] -- kept
    /// separate from `build` so it is testable against an in-memory buffer
    /// without touching the filesystem (the same split
    /// `sketch::GenomeSketch::from_reader` uses).
    ///
    /// Every k-mer is mapped to the lowest common ancestor of every taxon
    /// whose reference sequences contain it. That is the property the
    /// classifier's correctness rests on, and it is why a collision is
    /// resolved through [`Taxonomy::lca`] rather than by keeping the first
    /// or the last writer: keeping either would assign every ambiguous
    /// k-mer to whichever reference happened to appear first in the file,
    /// which produces no error and no warning -- just a systematic bias
    /// toward that organism in every sample ever classified against the
    /// database.
    pub fn build_from_source<S: RecordSource>(
        mut source: S,
        reference_path: &Path,
        taxonomy_file: TaxonomyFile,
        taxonomy_path: &Path,
        k: usize,
    ) -> Result<Self> {
        // Checked before a single record is read: an out-of-range k makes
        // the extractor return nothing for every sequence, so without this
        // the whole reference set would "build" into an empty database and
        // classify every read as unclassified, in silence.
        if k == 0 || k > 32 {
            return Err(FastDnaError::InvalidK { k });
        }
        source.validate()?;

        let TaxonomyFile { taxonomy, sequence_to_tax_id, sequence_row } = taxonomy_file;

        // One pair per k-mer *occurrence*, in one contiguous buffer. See
        // the type-level doc comment for why this beats a hash map on
        // operation count; see the module docstring's "Memory cost"
        // section for the peak this implies.
        let mut pairs: Vec<(u64, u32)> = Vec::new();
        let mut kmer_buf: Vec<u64> = Vec::new();
        let mut seen: FxHashSet<String> = FxHashSet::default();
        let mut record_no: u64 = 0;

        loop {
            match source.next_record() {
                Ok(Some(record)) => {
                    record_no += 1;
                    let id = record_id_of(&record.id);
                    let Some(&tax_id) = sequence_to_tax_id.get(id.as_str()) else {
                        return Err(FastDnaError::Load {
                            path: reference_path.to_path_buf(),
                            reason: format!(
                                "reference sequence {id:?} (record {record_no}) has no tax_id: add \
                                 it to a {SEQUENCE_IDS_COLUMN} cell in {}",
                                taxonomy_path.display()
                            ),
                        });
                    };
                    seen.insert(id);

                    extract_canonical_kmers_into(&record.seq, k, &mut kmer_buf);
                    pairs.reserve(kmer_buf.len());
                    for &kmer in &kmer_buf {
                        pairs.push((kmer, tax_id));
                    }
                }
                Ok(None) => break,
                Err(FastqReadError::Io(source_err)) => {
                    let (path, _) = failing_location(&source, reference_path, record_no);
                    return Err(FastDnaError::Io { path, source: source_err });
                }
                Err(FastqReadError::Malformed(reason)) => {
                    let (path, record) = failing_location(&source, reference_path, record_no);
                    return Err(FastDnaError::MalformedFastq { path, record, reason });
                }
            }
        }

        // The mirror of the check above: a taxonomy naming a sequence the
        // reference does not contain is a typo or a mismatched pair of
        // files, and skipping it silently means that taxon contributes no
        // k-mers at all -- so reads from it are never classified to it, and
        // nothing anywhere says why.
        let mut missing: Vec<(usize, &String)> = sequence_to_tax_id
            .keys()
            .filter(|id| !seen.contains(*id))
            .map(|id| (sequence_row.get(id).copied().unwrap_or(0), id))
            .collect();
        // Sorted so that a file with several mistakes always reports the
        // same (earliest) one, rather than whichever the hash map's
        // iteration order happened to yield.
        missing.sort_unstable();
        if let Some((row, id)) = missing.first() {
            return Err(FastDnaError::Load {
                path: taxonomy_path.to_path_buf(),
                reason: format!(
                    "row {row}: sequence id {id:?} does not appear in {}; the taxonomy and the \
                     reference must name the same sequences",
                    reference_path.display()
                ),
            });
        }

        Ok(Self::from_pairs(pairs, taxonomy, k))
    }

    /// Sorts `pairs` and folds every run of equal k-mers into one entry via
    /// [`Taxonomy::lca`].
    ///
    /// The distinct count is taken in its own linear pass before the two
    /// output arrays are allocated, so each is allocated exactly once at
    /// exactly the right size. Letting them grow instead would cost
    /// `log2(n)` reallocations and copies each, and would leave up to
    /// twice the needed capacity resident in the database that is about to
    /// be held for the rest of the process's life.
    fn from_pairs(mut pairs: Vec<(u64, u32)>, taxonomy: Taxonomy, k: usize) -> Self {
        pairs.sort_unstable();

        let distinct = pairs
            .windows(2)
            .filter(|w| w[0].0 != w[1].0)
            .count()
            + usize::from(!pairs.is_empty());

        let mut kmers: Vec<u64> = Vec::with_capacity(distinct);
        let mut tax_ids: Vec<u32> = Vec::with_capacity(distinct);

        let mut i = 0;
        while i < pairs.len() {
            let kmer = pairs[i].0;
            let mut tax_id = pairs[i].1;
            let mut j = i + 1;
            while j < pairs.len() && pairs[j].0 == kmer {
                tax_id = taxonomy.lca(tax_id, pairs[j].1);
                j += 1;
            }
            kmers.push(kmer);
            tax_ids.push(tax_id);
            i = j;
        }

        KmerDatabase { k, taxonomy, kmers, tax_ids }
    }

    /// The taxon a canonical k-mer maps to, or `None` if the database has
    /// never seen it.
    #[inline]
    pub fn lookup(&self, kmer: u64) -> Option<u32> {
        self.kmers.binary_search(&kmer).ok().map(|i| self.tax_ids[i])
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// The number of distinct canonical k-mers in the table.
    pub fn len(&self) -> usize {
        self.kmers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kmers.is_empty()
    }

    pub fn taxonomy(&self) -> &Taxonomy {
        &self.taxonomy
    }

    /// The two parallel arrays, for tests and for callers that want to walk
    /// the table directly rather than through `lookup`.
    pub fn entries(&self) -> (&[u64], &[u32]) {
        (&self.kmers, &self.tax_ids)
    }

    /// Bytes held by the lookup table alone: exactly
    /// `len() * `[`BYTES_PER_KMER`]. This is the number the module
    /// docstring's scale table is derived from.
    pub fn table_memory_bytes(&self) -> usize {
        self.kmers.len() * std::mem::size_of::<u64>() + self.tax_ids.len() * std::mem::size_of::<u32>()
    }

    /// Bytes held by the whole database: the table plus the taxonomy.
    ///
    /// The taxonomy term is small and roughly constant (a few hundred bytes
    /// per taxon), so for anything but a toy database the table dominates
    /// completely -- which is exactly why the scale table quotes the
    /// per-k-mer figure and nothing else.
    pub fn memory_bytes(&self) -> usize {
        let taxonomy: usize = self
            .taxonomy
            .taxa
            .iter()
            .map(|t| std::mem::size_of::<Taxon>() + t.name.len() + t.rank.len())
            .sum::<usize>()
            + self
                .taxonomy
                .lineages
                .iter()
                .map(|l| l.len() * std::mem::size_of::<u32>())
                .sum::<usize>()
            + self.taxonomy.index.len() * (std::mem::size_of::<u32>() + std::mem::size_of::<usize>());
        self.table_memory_bytes() + taxonomy
    }

    /// Writes the database to `path`.
    ///
    /// # Format
    ///
    /// ```text
    /// offset  bytes  content
    /// 0       8      magic, b"FDNAKDB\0"
    /// 8       4      format version, u32 little-endian
    /// 12      4      k, u32 little-endian
    /// 16      8      entry count n, u64 little-endian
    /// 24      8      taxonomy JSON length L, u64 little-endian
    /// 32      L      taxonomy, JSON
    /// 32+L    12n    entries: (k-mer u64 LE, tax_id u32 LE), ascending
    /// ```
    ///
    /// The entry region is the in-memory table verbatim, which is the whole
    /// point of choosing it: it makes the file a sorted, fixed-record
    /// array that a future `mmap`-backed lookup can binary search in place
    /// without parsing anything, and it makes truncation *exactly*
    /// detectable -- a file whose length is not `32 + L + 12n` is corrupt,
    /// with no judgement call involved. A variable-length encoding would
    /// give up both.
    ///
    /// The taxonomy is JSON rather than a bespoke binary block because it
    /// is a few thousand rows of text at most (the table is the part that
    /// gets large), and a human being able to read it out of the file with
    /// `head -c` is worth more than the bytes it saves.
    ///
    /// Written through [`AtomicFile`](crate::atomic::AtomicFile) like every
    /// other writer in this crate: a disk-full error or a Ctrl-C partway
    /// through must leave the previous good database in place rather than a
    /// truncated file that `load` would then have to reject.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let io_err = |e: std::io::Error| FastDnaError::Io { path: path.to_path_buf(), source: e };

        let taxonomy_json = serde_json::to_vec(&self.taxonomy).map_err(|e| FastDnaError::Export {
            path: path.to_path_buf(),
            reason: format!("could not serialize the taxonomy: {e}"),
        })?;

        let (file, pending) = crate::atomic::AtomicFile::create(path)?;
        let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);

        writer.write_all(DB_MAGIC).map_err(io_err)?;
        writer.write_all(&DB_FORMAT_VERSION.to_le_bytes()).map_err(io_err)?;
        writer.write_all(&(self.k as u32).to_le_bytes()).map_err(io_err)?;
        writer.write_all(&(self.kmers.len() as u64).to_le_bytes()).map_err(io_err)?;
        writer.write_all(&(taxonomy_json.len() as u64).to_le_bytes()).map_err(io_err)?;
        writer.write_all(&taxonomy_json).map_err(io_err)?;

        // One 12-byte record at a time into a 1 MB buffered writer: the
        // alternative, building the whole 12n-byte image in memory first,
        // would double the database's footprint at the exact moment it is
        // already fully resident.
        let mut record = [0u8; BYTES_PER_KMER];
        for (&kmer, &tax_id) in self.kmers.iter().zip(self.tax_ids.iter()) {
            record[..8].copy_from_slice(&kmer.to_le_bytes());
            record[8..].copy_from_slice(&tax_id.to_le_bytes());
            writer.write_all(&record).map_err(io_err)?;
        }

        writer.flush().map_err(io_err)?;
        drop(writer);
        pending.commit()
    }

    /// Reads a database written by [`save`](Self::save).
    ///
    /// Every invariant `lookup` silently assumes is re-established here, at
    /// the file boundary, and nowhere else: the table is ascending and
    /// deduplicated, `k` is in range, and every taxon id in the table
    /// exists in the embedded taxonomy. Checking them per lookup would mean
    /// paying for them once per k-mer of every read forever, to catch a
    /// problem that can only be introduced by a corrupt or hand-edited
    /// file. The failure mode being ruled out is not a crash -- an unsorted
    /// table makes `binary_search` return confident nonsense.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let load_err = |reason: String| FastDnaError::Load { path: path.to_path_buf(), reason };

        let bytes = std::fs::read(path)
            .map_err(|e| FastDnaError::Io { path: path.to_path_buf(), source: e })?;

        if bytes.len() < DB_HEADER_BYTES {
            return Err(load_err(format!(
                "the file is {} bytes, shorter than the {DB_HEADER_BYTES}-byte header a FastDNA \
                 k-mer database starts with",
                bytes.len()
            )));
        }
        if &bytes[..8] != DB_MAGIC {
            return Err(load_err(
                "the file does not start with the FastDNA k-mer database magic bytes; it was \
                 written by something else, or it is not a database at all"
                    .to_string(),
            ));
        }

        let version = read_u32(&bytes[8..12]);
        if version != DB_FORMAT_VERSION {
            return Err(load_err(format!(
                "the file is format version {version}, and this build of FastDNA reads version \
                 {DB_FORMAT_VERSION}"
            )));
        }

        let k = read_u32(&bytes[12..16]) as usize;
        if k == 0 || k > 32 {
            return Err(load_err(format!(
                "k is {k}, outside the 1..=32 range 2-bit packing supports"
            )));
        }

        let n_entries = read_u64(&bytes[16..24]) as usize;
        let taxonomy_len = read_u64(&bytes[24..32]) as usize;

        // The one length check that makes truncation unambiguous. Both
        // additions are done in `u128` because `n_entries` and
        // `taxonomy_len` come straight off disk: a corrupt file can name a
        // count that overflows `usize` on the multiply, and an overflow
        // there would wrap to a small number that then *passes* the check.
        let expected = DB_HEADER_BYTES as u128
            + taxonomy_len as u128
            + (n_entries as u128) * (BYTES_PER_KMER as u128);
        if bytes.len() as u128 != expected {
            return Err(load_err(format!(
                "the file is {} bytes but its header describes {expected} ({DB_HEADER_BYTES} \
                 header + {taxonomy_len} taxonomy + {n_entries} entries x {BYTES_PER_KMER}); it is \
                 truncated or corrupt",
                bytes.len()
            )));
        }

        let taxonomy_end = DB_HEADER_BYTES + taxonomy_len;
        let taxonomy: Taxonomy = serde_json::from_slice(&bytes[DB_HEADER_BYTES..taxonomy_end])
            .map_err(|e| load_err(format!("the embedded taxonomy is not readable: {e}")))?;
        // `index` and `lineages` are `#[serde(skip)]`, so what came back is
        // a shell. Re-running the constructor rebuilds both *and* re-runs
        // every structural check, which is the point: a hand-edited file
        // must not be able to smuggle in a cycle that every later ancestor
        // walk would then loop on.
        let taxonomy = Taxonomy::from_taxa(taxonomy.taxa, &FxHashMap::default())
            .map_err(|reason| load_err(format!("the embedded taxonomy is invalid: {reason}")))?;

        let mut kmers: Vec<u64> = Vec::with_capacity(n_entries);
        let mut tax_ids: Vec<u32> = Vec::with_capacity(n_entries);
        let mut previous: Option<u64> = None;
        for entry in bytes[taxonomy_end..].chunks_exact(BYTES_PER_KMER) {
            let kmer = read_u64(&entry[..8]);
            let tax_id = read_u32(&entry[8..]);
            if let Some(previous) = previous {
                if kmer <= previous {
                    return Err(load_err(format!(
                        "entry {} is k-mer {kmer} after k-mer {previous}: the table is not in \
                         strictly ascending order, which is what lookup's binary search assumes",
                        kmers.len()
                    )));
                }
            }
            if !taxonomy.contains(tax_id) {
                return Err(load_err(format!(
                    "entry {} maps to tax_id {tax_id}, which the embedded taxonomy does not define",
                    kmers.len()
                )));
            }
            previous = Some(kmer);
            kmers.push(kmer);
            tax_ids.push(tax_id);
        }

        Ok(KmerDatabase { k, taxonomy, kmers, tax_ids })
    }
}

/// Where to blame a reader-side failure: the file a multi-file source was
/// actually reading and that file's own record number, or the caller's
/// path plus the running count when the source has no path of its own.
/// The same rule `pipeline.rs` applies, applied here so an error from a
/// reference file is located the same way an error from a read file is.
fn failing_location<S: RecordSource>(
    source: &S,
    fallback_path: &Path,
    records_so_far: u64,
) -> (std::path::PathBuf, u64) {
    match source.current_source() {
        Some((path, records_in_file)) => (path, records_in_file + 1),
        None => (fallback_path.to_path_buf(), records_so_far + 1),
    }
}

fn read_u32(bytes: &[u8]) -> u32 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&bytes[..4]);
    u32::from_le_bytes(buf)
}

fn read_u64(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    u64::from_le_bytes(buf)
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

    // -- database construction --------------------------------------------

    /// `k` for the toy reference. Small enough that every k-mer set below
    /// can be listed by hand, large enough that the three blocks do not
    /// collide by accident -- which `the_toy_reference_blocks_share_no_kmers`
    /// proves rather than assumes.
    const TOY_K: usize = 11;

    /// Present in both species, so every k-mer wholly inside it must map to
    /// the genus, not to either species.
    const SHARED: &str = "GATTACAGATTACAGGCC";
    const ONLY_A: &str = "TTGCACCGTAAGCTATCG";
    const ONLY_B: &str = "ACGCGTTAACCGGATCAT";

    fn toy_reference() -> String {
        format!(">species_a a description\n{ONLY_A}{SHARED}\n>species_b\n{ONLY_B}{SHARED}\n")
    }

    fn kmers_of(seq: &str) -> Vec<u64> {
        crate::kmer::extract_canonical_kmers(seq.as_bytes(), TOY_K)
    }

    fn source_over(text: &str) -> crate::fastq::FastqReader<std::io::Cursor<Vec<u8>>> {
        crate::fastq::FastqReader::new(std::io::Cursor::new(text.as_bytes().to_vec()))
    }

    fn build_toy_db() -> KmerDatabase {
        KmerDatabase::build_from_source(
            source_over(&toy_reference()),
            std::path::Path::new("<toy reference>"),
            toy(),
            std::path::Path::new("<toy>"),
            TOY_K,
        )
        .expect("the toy reference and taxonomy agree")
    }

    /// The local buffer-reusing extractor must agree with the shared one on
    /// every input, or the classifier and the rest of the crate would be
    /// counting different k-mers. Pins the duplication until
    /// `kmer::extract_canonical_kmers_into` lands and this copy goes away.
    #[test]
    fn the_local_kmer_extractor_matches_the_shared_one() {
        let cases: [&[u8]; 8] = [
            b"",
            b"ACG",
            b"ACGTACGTACGTACGT",
            b"ACGTNACGTNACGTACGTAC",
            b"NNNNNNNNNNNN",
            b"acgtacgtacgtacgt",
            b"ACGTACGTACGTACGTNN",
            b"TTTTTTTTTTTTTTTTTTTT",
        ];
        let mut buffer = Vec::new();
        for seq in cases {
            for k in [1usize, 4, 11, 31, 32] {
                extract_canonical_kmers_into(seq, k, &mut buffer);
                assert_eq!(
                    buffer,
                    crate::kmer::extract_canonical_kmers(seq, k),
                    "mismatch at k={k} on {:?}",
                    String::from_utf8_lossy(seq)
                );
            }
        }
    }

    /// The premise of every assertion below: the unique blocks really are
    /// unique. Without this, "the shared block maps to the genus" could
    /// pass for the wrong reason.
    #[test]
    fn the_toy_reference_blocks_share_no_kmers() {
        let a: FxHashSet<u64> = kmers_of(ONLY_A).into_iter().collect();
        let b: FxHashSet<u64> = kmers_of(ONLY_B).into_iter().collect();
        let shared: FxHashSet<u64> = kmers_of(SHARED).into_iter().collect();
        assert!(a.is_disjoint(&b));
        assert!(a.is_disjoint(&shared));
        assert!(b.is_disjoint(&shared));
        assert!(!shared.is_empty());
    }

    #[test]
    fn a_kmer_unique_to_one_species_maps_to_that_species() {
        let db = build_toy_db();
        for kmer in kmers_of(ONLY_A) {
            assert_eq!(db.lookup(kmer), Some(100), "a k-mer only in species_a must map to it");
        }
        for kmer in kmers_of(ONLY_B) {
            assert_eq!(db.lookup(kmer), Some(200));
        }
    }

    /// The property the whole database exists for: a k-mer occurring in two
    /// species maps to the taxon covering both, not to whichever reference
    /// the builder happened to read first.
    #[test]
    fn a_kmer_shared_by_two_species_maps_to_their_common_ancestor() {
        let db = build_toy_db();
        for kmer in kmers_of(SHARED) {
            assert_eq!(db.lookup(kmer), Some(10), "a k-mer in both species must map to the genus");
        }
    }

    /// Order-independence, stated as a test rather than trusted: reversing
    /// the two reference records must not change a single assignment.
    #[test]
    fn the_lca_assignment_does_not_depend_on_reference_order() {
        let reversed = format!(">species_b\n{ONLY_B}{SHARED}\n>species_a\n{ONLY_A}{SHARED}\n");
        let flipped = KmerDatabase::build_from_source(
            source_over(&reversed),
            std::path::Path::new("<toy reference>"),
            toy(),
            std::path::Path::new("<toy>"),
            TOY_K,
        )
        .expect("valid");
        assert_eq!(build_toy_db().entries(), flipped.entries());
    }

    #[test]
    fn a_reference_sequence_with_no_tax_id_is_rejected_by_name() {
        let reference = format!(">species_a\n{ONLY_A}\n>mystery_contig\n{ONLY_B}\n");
        match KmerDatabase::build_from_source(
            source_over(&reference),
            std::path::Path::new("ref.fasta"),
            toy(),
            std::path::Path::new("taxonomy.tsv"),
            TOY_K,
        ) {
            Err(FastDnaError::Load { reason, .. }) => {
                assert!(reason.contains("mystery_contig"), "reason: {reason}");
                assert!(reason.contains("taxonomy.tsv"), "reason must say where to add it: {reason}");
            }
            other => panic!("expected Load, got {other:?}"),
        }
    }

    /// The mirror case: a taxonomy naming a sequence the reference does not
    /// contain is a typo or a mismatched pair of files, and quietly ignoring
    /// it means a taxon silently contributes no k-mers at all.
    #[test]
    fn a_taxonomy_sequence_id_missing_from_the_reference_is_rejected_by_row() {
        let reference = format!(">species_a\n{ONLY_A}\n");
        match KmerDatabase::build_from_source(
            source_over(&reference),
            std::path::Path::new("ref.fasta"),
            toy(),
            std::path::Path::new("taxonomy.tsv"),
            TOY_K,
        ) {
            Err(FastDnaError::Load { reason, .. }) => {
                assert!(reason.contains("species_b"), "reason: {reason}");
                assert!(reason.contains("row 5"), "reason must name the taxonomy row: {reason}");
                assert!(reason.contains("ref.fasta"), "reason: {reason}");
            }
            other => panic!("expected Load, got {other:?}"),
        }
    }

    #[test]
    fn an_out_of_range_k_is_rejected_before_any_reference_is_read() {
        for k in [0usize, 33, 64] {
            match KmerDatabase::build_from_source(
                source_over(&toy_reference()),
                std::path::Path::new("ref.fasta"),
                toy(),
                std::path::Path::new("taxonomy.tsv"),
                k,
            ) {
                Err(FastDnaError::InvalidK { k: reported }) => assert_eq!(reported, k),
                other => panic!("expected InvalidK for k={k}, got {other:?}"),
            }
        }
    }

    /// The number the module docstring's scale table is derived from. If
    /// the representation ever changes, this fails and the docstring has to
    /// be corrected with it.
    #[test]
    fn resident_memory_is_twelve_bytes_per_kmer() {
        let db = build_toy_db();
        assert_eq!(BYTES_PER_KMER, 12);
        assert_eq!(db.table_memory_bytes(), db.len() * BYTES_PER_KMER);
        assert!(db.memory_bytes() > db.table_memory_bytes(), "the taxonomy costs something too");
    }

    // -- persistence ------------------------------------------------------

    /// A throwaway directory under the system temp dir, removed on drop --
    /// the same fixture shape `fastq.rs`'s own tests use.
    struct Fixture {
        dir: std::path::PathBuf,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("fastdna_metagenomics_rs_{name}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Fixture { dir }
        }

        fn path(&self, name: &str) -> std::path::PathBuf {
            self.dir.join(name)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn save_then_load_round_trips_to_an_identical_database() {
        let fx = Fixture::new("round_trip");
        let path = fx.path("toy.fdb");
        let original = build_toy_db();
        original.save(&path).unwrap();

        let loaded = KmerDatabase::load(&path).unwrap();
        assert_eq!(loaded.k(), original.k());
        assert_eq!(loaded.entries(), original.entries());
        assert_eq!(loaded.taxonomy(), original.taxonomy());

        // And saving the loaded copy reproduces the same bytes, which is
        // what makes the format worth calling deterministic.
        let second = fx.path("toy_again.fdb");
        loaded.save(&second).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), std::fs::read(&second).unwrap());
    }

    #[test]
    fn a_truncated_database_file_is_a_load_error_not_a_panic() {
        let fx = Fixture::new("truncated");
        let path = fx.path("toy.fdb");
        build_toy_db().save(&path).unwrap();

        let full = std::fs::read(&path).unwrap();
        for keep in [0, 8, 20, full.len() - 1, full.len() - 6] {
            std::fs::write(&path, &full[..keep]).unwrap();
            match KmerDatabase::load(&path) {
                Err(FastDnaError::Load { .. }) => {}
                other => panic!("expected Load for a {keep}-byte file, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_foreign_file_is_rejected_by_its_magic_bytes() {
        let fx = Fixture::new("foreign");
        let path = fx.path("not_a_db.fdb");
        std::fs::write(&path, b"this is a text file, not a k-mer database at all").unwrap();
        match KmerDatabase::load(&path) {
            Err(FastDnaError::Load { reason, .. }) => {
                assert!(reason.contains("FastDNA"), "reason: {reason}");
            }
            other => panic!("expected Load, got {other:?}"),
        }
    }

    /// A hand-edited or bit-rotted table whose entries are out of order
    /// would make `lookup`'s binary search return quiet nonsense, so the
    /// ordering invariant is checked at the file boundary -- once -- exactly
    /// as `sketch.rs` checks its own sorted-hashes invariant.
    #[test]
    fn an_unsorted_entry_table_is_rejected_on_load() {
        let fx = Fixture::new("unsorted");
        let path = fx.path("toy.fdb");
        let db = build_toy_db();
        db.save(&path).unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        let table_start = bytes.len() - db.len() * BYTES_PER_KMER;
        // Swap the first two entries, breaking ascending order.
        for i in 0..BYTES_PER_KMER {
            bytes.swap(table_start + i, table_start + BYTES_PER_KMER + i);
        }
        std::fs::write(&path, &bytes).unwrap();

        match KmerDatabase::load(&path) {
            Err(FastDnaError::Load { reason, .. }) => {
                assert!(reason.contains("ascending"), "reason: {reason}");
            }
            other => panic!("expected Load, got {other:?}"),
        }
    }

    #[test]
    fn an_entry_pointing_at_an_unknown_taxon_is_rejected_on_load() {
        let fx = Fixture::new("dangling_taxon");
        let path = fx.path("toy.fdb");
        let db = build_toy_db();
        db.save(&path).unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        let first_tax = bytes.len() - db.len() * BYTES_PER_KMER + 8;
        bytes[first_tax..first_tax + 4].copy_from_slice(&9_999u32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        match KmerDatabase::load(&path) {
            Err(FastDnaError::Load { reason, .. }) => {
                assert!(reason.contains("9999"), "reason: {reason}");
            }
            other => panic!("expected Load, got {other:?}"),
        }
    }

    #[test]
    fn load_of_a_missing_file_is_an_io_error_not_a_load_error() {
        let fx = Fixture::new("missing");
        match KmerDatabase::load(fx.path("nope.fdb")) {
            Err(FastDnaError::Io { .. }) => {}
            other => panic!("expected Io, got {other:?}"),
        }
    }
}
