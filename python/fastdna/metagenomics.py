"""fastdna.metagenomics -- Kraken2-style per-read metagenomic
classification against a k-mer -> lowest-common-ancestor database.

Build a database from a reference FASTA plus a taxonomy TSV, then assign
every read of a sample to a node of that taxonomy:

    from fastdna.metagenomics import build_database

    db = build_database("panel.fasta", "panel_taxonomy.tsv", k=31,
                        output="panel.fdb")
    calls = db.classify("sample.fastq.gz", confidence_threshold=0.1)
    print(db.abundance(calls).to_pandas())

The algorithm is the one from Wood, Lu & Langmead, "Improved metagenomic
analysis with Kraken 2", *Genome Biology* 20:257 (2019). This module is a
thin surface over `src/metagenomics.rs`, which holds every algorithmic
decision and documents each of them; the notes below are the ones a user
of the Python API has to know before trusting a result.

**Scale: this does not hold RefSeq, and it will not tell you so by
failing gracefully -- it will exhaust memory.** The lookup table is two
parallel arrays, costing
**12 bytes per distinct canonical k-mer** exactly, with no per-entry
overhead (measured: 12.0004 bytes/k-mer over a 1.1-million-k-mer
database, the excess being 467 bytes of taxonomy for 5 taxa). Building it
buffers one 16-byte pair per k-mer *occurrence* before merging into that
table, so the peak is
`16 * occurrences + 12 * distinct` -- around 28 bytes per k-mer, a
little over twice the finished size, for a panel of distinct organisms.
Budget 3x. Concretely, at k=31:

===========================  ===============  ==========  =========
Reference set                Distinct k-mers  Resident    To build
===========================  ===============  ==========  =========
1 bacterial genome (~4 Mbp)  ~4 M             ~48 MB      ~110 MB
20 bacterial genomes         ~80 M            ~1.0 GB     ~2.2 GB
150 bacterial genomes        ~600 M           ~7.2 GB     ~17 GB
RefSeq complete bacteria     ~10^11           **~1.2 TB** --
===========================  ===============  ==========  =========

On an ordinary 16 GB workstation the practical ceiling is 150-200
million distinct k-mers, roughly 40-50 bacterial genomes, and it is the
*build* that binds. Check `db.memory_bytes` and `db.n_kmers` on a small
reference set before scaling one up. Use this for a targeted panel -- a
pathogen set, a mock community, a host-plus-contaminants screen -- not
for an open-world survey.

**No parity claim with Kraken 2.** Kraken 2 is mature, fast and widely
validated; this has not been benchmarked against it for speed,
sensitivity or precision, and nothing here should be read as claiming
equivalence. Kraken 2's standard database covers all of RefSeq bacteria,
archaea and viruses in ~8.7 GB because it stores *minimizers* in a
compact hash table with spaced seeds; this module stores every distinct
canonical k-mer with exact 64-bit keys, which is the main reason the
table above says what it says. The spaced seeds also cost real
sensitivity: an exact k-mer match has no tolerance for a substitution,
so a strain that has drifted from the reference is matched less well
here than by Kraken 2. Classification is single-threaded, read pairs are
treated as two independent reads, and there is no `kraken2-report`
equivalent. See the Rust module docstring for the full list.

**Relationship to `fastdna.taxonomy`.** That module is a complement, not
an overlap: it compares whole-sample MinHash sketches (`classify`,
`gather`, `containment`) to rank which *references* best explain a
*sample*, approximately, with no taxonomy tree and no database build.
This module assigns *each read* to a *node of a taxonomy*, exactly, from
a database that must be built first. The two names collide -- both have
a `classify` -- but they answer different questions at different costs.
Reach for `fastdna.taxonomy.gather` to find out cheaply what is probably
in a sample; reach for this when you need per-read assignments, a
composition table, or a host-removal decision per read.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Union

import pyarrow as pa

from . import _core

__all__ = ["build_database", "KmerDatabase"]

# A path accepted anywhere in this module: a `str`, or anything implementing
# `os.PathLike` (e.g. `pathlib.Path`) -- every such parameter is converted
# with `str(path)` before use, matching `fastdna/__init__.py`'s own
# `_PathLike` convention.
_PathLike = Union[str, os.PathLike]


def build_database(
    reference: _PathLike, taxonomy: _PathLike, *, k: int = 31, output: Optional[_PathLike] = None
) -> "KmerDatabase":
    """Builds a k-mer -> lowest-common-ancestor database and returns it as
    a :class:`KmerDatabase`.

    `reference` is a FASTA (or FASTQ, gzipped or not) file of reference
    sequences. `taxonomy` is a tab-separated file carrying both the
    taxonomy tree and the sequence-to-taxon mapping, with a header naming
    at least `tax_id`, `parent_tax_id`, `rank` and `name`, plus an
    optional `sequence_ids` column listing the reference sequence ids
    belonging to each taxon, separated by `;` or `,`::

        tax_id  parent_tax_id  rank     name              sequence_ids
        1       1              no rank  root
        561     1              genus    Escherichia
        562     561            species  Escherichia coli  NC_000913.3

    A sequence id is the first whitespace-delimited token of a FASTA
    header without the `>`, the same convention `samtools faidx` uses.
    The root names either `0` or itself as its parent, following NCBI.

    NCBI's `nodes.dmp`/`names.dmp` are **not** accepted directly -- they
    are three files in a non-TSV format, and the join (scientific-name
    selection, accession-to-taxid mapping) is the user's. See the Rust
    module docstring for exactly which fields to take from which file.

    Every structural problem raises `ValueError` naming the offending
    1-based line rather than being skipped: a missing parent, a cycle, a
    duplicate `tax_id`, a sequence id assigned twice, `tax_id` 0
    (reserved for unclassified reads), a missing column, or no root. A
    reference sequence with no taxon, and a taxonomy sequence id absent
    from the reference, are errors too. None of these degrade to a
    warning, because a taxonomy quietly missing the branch you cared
    about still classifies every read -- to the wrong node, at full
    confidence, with nothing anywhere saying so.

    `k` must be in 1..=32 (the 2-bit packing limit) or `ValueError` is
    raised. `output`, if given, saves the database to that path; the
    database is returned either way, so building and classifying in one
    session never pays a save and a reload.
    """
    return KmerDatabase(
        _core.build_database(
            reference=str(reference),
            taxonomy=str(taxonomy),
            k=k,
            output=None if output is None else str(output),
        )
    )


class KmerDatabase:
    """A built k-mer -> taxon database, from :func:`build_database` or
    :meth:`load`.

    `.n_kmers` is the number of distinct canonical k-mers held and
    `.memory_bytes` what they cost -- exactly 12 bytes each plus a small
    taxonomy term. Both are exposed so the scale limits in this module's
    docstring can be checked against a real reference set rather than
    discovered by running out of memory.
    """

    def __init__(self, raw: "_core.KmerDatabase") -> None:
        self._raw = raw

    @classmethod
    def load(cls, path: _PathLike) -> "KmerDatabase":
        """Loads a database written by :meth:`save` or by
        `build_database(..., output=...)`.

        A truncated, corrupt or foreign file raises `ValueError` saying
        so. That check is not a formality: the table is searched by
        binary search, and an out-of-order table does not crash it -- it
        makes it return confident nonsense.
        """
        return cls(_core.KmerDatabase.load(str(path)))

    def save(self, path: _PathLike) -> None:
        """Persists the database, replacing `path` atomically."""
        self._raw.save(str(path))

    @property
    def k(self) -> int:
        return self._raw.k

    @property
    def n_kmers(self) -> int:
        """The number of distinct canonical k-mers in the lookup table."""
        return self._raw.n_kmers

    @property
    def memory_bytes(self) -> int:
        """Resident bytes: `12 * n_kmers` plus the taxonomy."""
        return self._raw.memory_bytes

    def classify(self, reads: _PathLike, *, confidence_threshold: float = 0.0) -> pa.Table:
        """Classifies every read of a FASTA/FASTQ(.gz) file, returning a
        `pyarrow.Table` with one row per read, in input order:

        ====================  =========  ====================================
        column                type       meaning
        ====================  =========  ====================================
        `read_id`             utf8       header up to the first whitespace
        `tax_id`              uint32     assigned taxon; **0 = unclassified**
        `confidence`          float64    see below
        `n_kmers`             uint32     k-mers the read yielded
        `n_classified_kmers`  uint32     of those, how many were in the db
        ====================  =========  ====================================

        `confidence` is Kraken 2's own definition: the fraction of *all*
        the read's k-mers that fall in the clade rooted at the assigned
        taxon. The denominator is every k-mer, not every k-mer that
        matched something -- a read of 20 k-mers where 8 hit one species
        and 12 hit nothing scores 0.4, not 1.0. Scoring it over hit
        k-mers would report near-total confidence for precisely the reads
        that mostly matched nothing, which is the population the score
        exists to flag.

        `confidence_threshold` is Kraken 2's semantics as well: a call
        must be backed by `ceil(threshold * n_kmers)` k-mers inside its
        clade, and while it is not, it moves up to its parent, whose
        clade can only be larger. If it passes the root without ever
        meeting the bar the read is unclassified (`tax_id` 0). Raising
        the threshold therefore makes calls less specific or drops them;
        it never moves a call to a different branch. Must be in
        [0.0, 1.0] or `ValueError` is raised.

        Reads shorter than `k`, reads whose every window contains an
        ambiguous base, and reads matching nothing all appear as
        unclassified rows -- they are never silently dropped, so the row
        count always equals the read count. `n_classified_kmers` keeps
        "matched nothing" distinguishable from "matched plenty, but not
        confidently enough". Windows containing an ambiguous base yield
        no k-mer and count toward neither the numerator nor the
        denominator, as in Kraken 2.

        The whole result is materialized: roughly 48 bytes plus the read
        name per read, so a 100-million-read file needs about 5 GB. Split
        such inputs into chunks.
        """
        batch = self._raw.classify(str(reads), confidence_threshold)
        return pa.Table.from_batches([batch])

    def abundance(self, classification: Union[pa.Table, pa.RecordBatch, Sequence[int]]) -> pa.Table:
        """Aggregates a table from :meth:`classify` into a composition
        report: `tax_id`, `name`, `rank`, `reads`, `relative_abundance`,
        most reads first with ties broken by `tax_id` so two runs over the
        same data give identical tables.

        Unclassified reads appear as their own row (`tax_id` 0, named
        `unclassified`) and are in the denominator, so
        `relative_abundance` sums to exactly 1 and the unclassified
        fraction stays visible instead of being renormalized away.

        **This is read abundance, not organism abundance.** It is not
        corrected for genome length, and that correction is not optional
        for the question people usually mean by "abundance": a 6 Mbp
        organism and a 1.5 Mbp organism present in equal cell numbers
        yield roughly four reads to one, so this report overstates the
        larger genome fourfold as a share of the *community* -- and
        comparing a bacterium against a virus is off by a factor of a
        thousand. Correcting it is what Bracken exists to do (Lu,
        Breitwieser, Thielen & Salzberg, "Bracken: estimating species
        abundance in metagenomics data", *PeerJ Computer Science* 3:e104,
        2017). **FastDNA does not implement it.** Read these numbers as
        "share of reads", never as "share of organisms".

        Counts are per assigned taxon and are not cumulative down the
        clade: a read called at the genus is counted at the genus alone,
        not also under each of its species. Kraken's own report format
        gives clade-cumulative totals as well; this does not.

        `classification` may be the `pyarrow.Table` :meth:`classify`
        returned, or any sequence of taxon ids.
        """
        if isinstance(classification, (pa.Table, pa.RecordBatch)):
            tax_ids = classification.column("tax_id").to_pylist()
        else:
            tax_ids = [int(tax_id) for tax_id in classification]
        return pa.Table.from_batches([self._raw.abundance(tax_ids)])

    def __repr__(self) -> str:
        return f"KmerDatabase(k={self.k}, n_kmers={self.n_kmers:,}, memory_bytes={self.memory_bytes:,})"

    def _repr_html_(self):
        """Rich display for Jupyter/IPython: the scale figures plus the
        one caveat a newcomer looking at a bare `KmerDatabase` in a cell
        has not read the module docstring for.

        No optional dependency involved (unlike `KmerCounts._repr_html_`):
        a database has no table worth previewing, just a few scalars.
        """
        return (
            "<div>"
            f"<p><b>KmerDatabase</b> &mdash; k={self.k}, "
            f"n_kmers={self.n_kmers:,}, memory={self.memory_bytes / 1e6:,.1f} MB</p>"
            "<p style='color: #666; font-size: 0.9em;'>"
            "A canonical k-mer &rarr; lowest-common-ancestor table for Kraken2-style "
            "per-read classification. Costs 12 bytes per distinct k-mer and roughly "
            "3&times; that to build, so it suits a targeted reference panel rather than "
            "an open-world survey &mdash; it does not scale to RefSeq."
            "</p>"
            "</div>"
        )
