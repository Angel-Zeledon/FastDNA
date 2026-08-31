"""fastdna.equivalence -- collapsing k-mer features that share an identical
presence/absence profile across one cohort matrix into equivalence classes.

## Why this exists

Consecutive/overlapping k-mers in the same genomic region are very often
present in exactly the same samples of a cohort: if a 31-mer is present in a
genome, its neighbor overlapping it by 30 bases usually is too. Feeding a
classifier (`fastdna.rules.SetCoveringClassifier`, or any scikit-learn
estimator) thousands of k-mer columns that are perfectly, or near-perfectly,
correlated inflates the feature space for no informational gain, and can
distort which single "representative" k-mer a greedy or linear model happens
to pick -- hurting interpretability even though the underlying signal is the
same. The real-world analogue is DBGWAS's compacted de Bruijn graph unitigs
(Jaillard, Lima, Tournoud, Mahe, van Belkum, Lacroix & Jacob, "A fast and
agnostic method for bacterial genome-wide association studies: bridging the
gap between k-mers and genetic events", *PLOS Genetics* 14(11):e1007758,
2018), which exist for exactly this reason: to decorrelate overlapping
k-mers before association testing. (The title above is the one actually
printed on the published article, PLOS Genetics 14(11):e1007758, November
2018 -- verified against the journal record rather than copied from memory,
since an earlier draft/preprint of the same work circulated under a
different working title, "Representing Genetic Determinants in Bacterial
GWAS with Compressed Vocabulary of Variable-length k-mers".)

**This module is deliberately NOT that.** True unitigs come from de Bruijn
graph adjacency -- does k-mer X's k-1 suffix match k-mer Y's k-1 prefix in
the assembly graph -- i.e. sequence-level adjacency, checked once per genome
assembly, independent of any particular cohort or phenotype. This module
instead groups k-mers by **identical presence/absence pattern across the
samples of one specific cohort matrix**: a purely statistical proxy that
needs no graph, no adjacency queries, and no new Rust infrastructure. It is
cheaper and simpler, but it is a genuinely different, weaker guarantee:

- Two adjacent k-mers that are truly part of the same unitig can still end
  up in *different* equivalence classes here, if a single sample happens to
  have a sequencing gap, an assembly break, or an error that removes one of
  the pair but not the other -- adjacency in the genome does not guarantee
  identical presence/absence across every sample of a noisy real cohort.
- Conversely, two k-mers from opposite ends of the genome that happen to be
  perfectly co-present in *this* cohort -- by real biology (two genes always
  co-inherited, e.g. on the same mobile element or under the same selective
  sweep) or by simple chance in a small cohort -- collapse together here
  exactly as if they were physically adjacent. Nothing in this module can
  tell those two cases apart; only graph adjacency could.

A future graph-based unitig feature -- tracked separately as a deferred item
in `docs/superpowers/plans/2026-08-24-completeness-phase.md` ("Unitigs
(compacted de Bruijn graph)"), which needs a k-mer adjacency/query layer
that does not exist in this codebase yet -- is therefore a different,
complementary capability, not a superset or a subset of this one. Prefer
that (once it exists) when biological unitig identity matters; use this
module when the goal is simply to stop feeding a classifier thousands of
columns that carry no independent information *in this particular cohort*.

## Scope of this first version

**Exact identity only.** Two k-mer columns belong to the same class iff
they have the *exactly* identical set of nonzero (present) sample rows --
not a correlation threshold, not "near-identical". A thresholded/
near-identical variant (e.g. Jaccard similarity above some cutoff) is a
plausible future extension, but is deliberately not built here: this
project's established discipline is not to ship an unvalidated statistical
knob nobody has asked for and nobody has picked a default for.

## Semantics worth stating explicitly

- **Presence/absence, not exact counts.** Two columns with the same nonzero
  pattern but different stored values (sample 3 has depth 40 for k-mer A and
  depth 12 for k-mer B, but both are simply "present") are the same class.
  Sequencing depth varies for reasons that have nothing to do with whether a
  k-mer is truly present (library prep, coverage unevenness, PCR bias), so
  count-level differences must never block collapsing two otherwise-identical
  columns. Because members of a class can therefore disagree on their raw
  counts, and there is no principled single number to keep (sum? mean? one
  arbitrary member's own count?), the *reduced* matrix this function returns
  stores plain 0/1 presence, not counts -- which also means it is directly
  usable by `fastdna.rules.SetCoveringClassifier.fit()` without a separate
  binarization step.
- **An all-zero column** (a k-mer present in no sample -- should not
  normally occur in `cohort_presence_matrix`'s own output, but nothing here
  assumes it cannot) is handled by the same rule as every other column,
  with no special case: its presence set is the empty set, and two empty
  sets are, definitionally, identical. So a lone all-zero column becomes its
  own singleton class, and two or more all-zero columns collapse into one
  class together, exactly as any other pair of matching columns would.
- **Hash collisions never cause a wrong merge.** Candidate columns are first
  grouped by a cheap, deliberately non-cryptographic hash of their presence
  pattern (see "Efficiency" below); every candidate group is then checked
  for *true* equality before anything is actually merged, splitting a
  candidate group back apart if it turns out to contain columns that only
  collided in the hash. `python/tests/test_equivalence.py::
  test_hash_collision_does_not_merge_distinct_classes` constructs two
  genuinely different presence sets that are guaranteed, by the documented
  hash formula, to land in the same candidate bucket, and asserts they come
  out as two classes, not one.
- **Representative choice is deterministic.** Each class's representative is
  the lexicographically smallest k-mer sequence among its members (the same
  byte-wise ordering `fastdna.gwas.cohort_presence_matrix` already uses for
  its own column order, and for the same reason: it is stable, hand
  -checkable from the member list alone, and does not depend on dict/set
  iteration order, hash-bucket order, or original column position). Final
  classes -- and therefore the reduced matrix's column order -- are sorted
  by that same representative, ascending, so the whole result is
  reproducible run to run on the same input.
- **`members` is complete.** Every input k-mer appears in exactly one
  class's member list in the returned Arrow table, never zero and never
  more than one -- this is what keeps this package's "every feature is a
  literal DNA sequence you can BLAST" property intact after collapsing:
  nothing is silently dropped, only grouped.

## Efficiency, argued by operation count

The obvious implementation -- a Python `for` loop over every column,
extracting and comparing its full nonzero-row-index slice against every
other column's -- is O(n_kmers^2) in the number of columns, on top of
O(n_kmers) Python-level loop overhead; with cohort screens routinely in the
10^6-10^7 k-mer range (see `fastdna.gwas`'s own docstrings), that is not
viable. This implementation instead does, in order:

1. `matrix.tocsc(copy=True)` (never mutates the caller's matrix),
   `eliminate_zeros()`, `sort_indices()`: converting to CSC makes each
   column's nonzero row indices a contiguous slice
   `indices[indptr[j]:indptr[j+1]]`; `eliminate_zeros()` drops any
   explicitly-stored zero so "has a stored entry" and "is present" mean the
   same thing; `sort_indices()` makes that slice a canonical, directly
   byte-comparable fingerprint of the column's presence set. All three are
   single vectorized scipy/C passes, O(nnz) total, not O(n_kmers) Python
   -level work.
2. A **cheap, vectorized, per-column candidate hash**: for every stored
   entry (row `r`, column `j`), contribute `r + 1` to a running sum for
   column `j`, via `numpy.add.at` over the whole `(row, column)` index
   arrays at once -- one C-level pass over the `nnz` stored entries, not a
   Python loop over columns. That sum is then combined with the column's
   own length into a single key, `hash = sum * (n_samples + 1) + length`,
   and `pyarrow.compute.dictionary_encode` (the same primitive
   `fastdna.gwas.cohort_presence_matrix` and `fastdna.sklearn.
   KmerVectorizer` already use for exactly this "group by value in one hash
   pass" purpose) assigns every column a candidate-bucket id in one more
   vectorized pass. This hash is deliberately weak -- a plain additive sum
   is trivial to collide on purpose (see the module docstring above) -- and
   that is fine precisely *because* step 3 below never trusts it alone.
3. **True-equality verification, but only for candidate buckets with more
   than one member.** A column whose candidate bucket contains only itself
   needs no further check: no other column can be indistinguishable from a
   never-collided candidate id. Only for the (expected to be rare) buckets
   with two or more candidate members is each member's canonical byte
   fingerprint (`indices_slice.tobytes()`, from the already-sorted CSC
   built in step 1) computed and compared; splitting a bucket into its
   genuinely-distinct sub-groups when they disagree. The total cost of this
   step is bounded by the total size of every multi-member bucket, which is
   at most the full `nnz` (when everything happens to land in one giant
   bucket, e.g. every column is identical) and, for the common case of a
   cohort with many genuinely distinct columns, touches only the columns
   that actually needed the check.
4. **Assembling the reduced matrix without a per-class Python loop.** Since
   every member of a class shares, by construction, one identical presence
   pattern, the union of their presence rows -- which is what the reduced
   column must represent -- equals any single member's own rows. So instead
   of picking a "representative column" per class and re-slicing it, every
   *original* stored entry is remapped straight to its final class id (one
   more vectorized `numpy` fancy-index) and handed to
   `scipy.sparse.coo_matrix`; `coo_matrix.sum_duplicates()` then merges the
   repeated `(row, class)` entries that a multi-member class produces (one
   duplicate per extra member sharing that row) into a single stored entry,
   after which every stored value is reset to `1` (the duplicates summed to
   the member count, not to a meaningful count -- see "presence, not
   counts" above). This is O(nnz) end to end, with no Python-level loop over
   classes or columns at all.
5. **Naming and ordering classes.** The lexicographically-smallest-member
   rule for `representative`, and the ascending-by-representative rule for
   final class order, are both computed with `pyarrow.compute` (a `group_by
   ("class").aggregate([("kmer_sequence", "min")])` plus a `sort_indices`
   over the resulting representatives) rather than a Python loop over
   classes -- consistent with how `fastdna.gwas.cohort_presence_matrix`
   already derives its own lexicographic column order.

Net complexity: O(nnz) for every step that must look at the matrix's actual
content, plus O(n_kmers log n_kmers) for the sorts, and no step whose cost
scales with `n_kmers^2` or that runs Python bytecode once per column across
the full column count.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from scipy import sparse

from fastdna import _core

__all__ = ["EquivalenceClasses", "collapse_equivalence_classes"]


class EquivalenceClasses(NamedTuple):
    """The result of :func:`collapse_equivalence_classes`.

    Attributes
    ----------
    matrix : scipy.sparse.csr_matrix of uint8
        The reduced cohort matrix, shape `(n_samples, n_classes)`. Values
        are `0`/`1` presence, not counts -- see the module docstring's
        "presence, not counts" note for why. Column `j` corresponds to
        `representative[j]`.
    representative : list of str
        One k-mer sequence per output column, the lexicographically
        smallest member of that class (see the module docstring). Sorted
        ascending, which is also the column order of `matrix`.
    members : pyarrow.Table
        Columns `class_id` (int64, `0..len(representative)-1`, matching
        `matrix`'s columns) and `kmer_sequence` (string) -- every input
        k-mer appears in exactly one row. Sorted by `class_id` ascending,
        then `kmer_sequence` ascending within a class (so a class's
        representative is always the first row of its group).
    """

    matrix: object  # scipy.sparse.csr_matrix
    representative: list
    members: object  # pyarrow.Table


def collapse_equivalence_classes(matrix: sparse.spmatrix, kmer_sequences: Sequence[str]) -> EquivalenceClasses:
    """Collapses columns of `matrix` that share an identical presence/
    absence profile into one equivalence class each.

    See the module docstring for the full design rationale (why this
    exists, how it differs from true de-Bruijn-graph unitigs, and the exact
    complexity argument for the approach used here).

    Parameters
    ----------
    matrix : scipy.sparse matrix, shape (n_samples, n_kmers)
        As returned by `fastdna.gwas.cohort_presence_matrix` or
        `fastdna.sklearn.KmerVectorizer.transform`. Any scipy sparse format
        is accepted (it is converted to CSC internally); a dense
        `numpy.ndarray` or plain nested list is rejected rather than
        silently densified/wrapped, because the whole point of this
        function is to stay usable at the 10^6-10^7-column scale
        `fastdna.gwas`'s own docstrings describe, where a dense copy would
        not fit in memory in the first place. Wrap a genuinely dense matrix
        yourself first (`scipy.sparse.csr_matrix(dense_array)`) if you truly
        need to run this on one. The input is never mutated.
    kmer_sequences : sequence of str, length n_kmers
        Column labels, aligned with `matrix`'s columns -- as returned
        alongside it by `cohort_presence_matrix` or
        `KmerVectorizer.get_feature_names_out()`.

    Returns
    -------
    EquivalenceClasses

    Raises
    ------
    TypeError
        If `matrix` is not a scipy sparse matrix.
    ValueError
        If `matrix` has zero rows or zero columns, or if `len(kmer_sequences)`
        does not equal `matrix.shape[1]`.
    """
    if not sparse.issparse(matrix):
        raise TypeError(
            f"collapse_equivalence_classes() expects a scipy.sparse matrix (any format -- "
            f"it is converted to CSC internally), got {type(matrix).__name__!r}. This function "
            "is built to stay usable on cohort matrices with millions of k-mer columns "
            "(see fastdna.gwas.cohort_presence_matrix), where a dense copy would not fit "
            "in memory -- wrap a genuinely dense array yourself first with "
            "scipy.sparse.csr_matrix(dense_array) if you really need to run this on one."
        )

    n_samples, n_kmers = matrix.shape
    if n_samples == 0 or n_kmers == 0:
        raise _core.InvalidConfigError(
            f"matrix is empty: shape {matrix.shape} has no "
            f"{'samples' if n_samples == 0 else 'k-mers'}. collapse_equivalence_classes() "
            "needs at least one sample and one k-mer column."
        )

    kmer_sequences = list(kmer_sequences)
    if len(kmer_sequences) != n_kmers:
        raise _core.InvalidConfigError(
            f"kmer_sequences has {len(kmer_sequences)} entries but matrix has {n_kmers} "
            "columns. A silent mismatch here would label every class with the WRONG "
            "k-mer, so this is refused rather than truncated -- pass exactly the "
            "kmer_sequences that came back alongside this matrix (e.g. from "
            "cohort_presence_matrix() or KmerVectorizer.get_feature_names_out())."
        )

    # Step 1: canonicalize. copy=True guarantees the caller's own matrix is
    # never touched by the in-place eliminate_zeros()/sort_indices() below,
    # regardless of what format it started in.
    csc = matrix.tocsc(copy=True)
    csc.eliminate_zeros()
    csc.sort_indices()

    indptr = csc.indptr
    row_of_entry = csc.indices  # row index of every stored entry, ascending within each column
    col_length = np.diff(indptr).astype(np.int64)  # nnz per column

    # Step 2: a cheap, vectorized, per-column candidate hash. `row_of_entry`
    # already tells us the row of every stored entry; `col_of_entry` (its
    # column) is derived the same way `fastdna.sklearn`'s own vectorized
    # passes derive a row-to-sample mapping from a per-sample row count --
    # one numpy repeat, not a Python loop.
    col_of_entry = np.repeat(np.arange(n_kmers, dtype=np.int64), col_length)
    weights = row_of_entry.astype(np.uint64) + np.uint64(1)
    col_hash = np.zeros(n_kmers, dtype=np.uint64)
    # np.add.at (not a plain fancy-index +=) is required here: col_of_entry
    # repeats whenever a column has more than one stored entry, and only
    # np.add.at accumulates repeated target indices correctly in one call.
    np.add.at(col_hash, col_of_entry, weights)
    combined_key = col_hash * np.uint64(n_samples + 1) + col_length.astype(np.uint64)

    codes = pc.dictionary_encode(pa.array(combined_key))
    candidate_code = np.asarray(codes.indices).astype(np.int64, copy=False)
    n_candidates = len(codes.dictionary)
    bucket_size = np.bincount(candidate_code, minlength=n_candidates)

    # Step 3: verify true equality, but only inside candidate buckets with
    # more than one member -- a column whose candidate id is unique to it
    # needs no further check.
    provisional = candidate_code.copy()
    multi_mask = bucket_size[candidate_code] > 1
    multi_cols = np.flatnonzero(multi_mask)
    next_new_id = n_candidates
    if multi_cols.size:
        order = multi_cols[np.argsort(candidate_code[multi_cols], kind="stable")]
        boundaries = np.flatnonzero(np.diff(candidate_code[order])) + 1
        for run in np.split(order, boundaries):
            signature_of = {}
            for col in run.tolist():
                start, end = indptr[col], indptr[col + 1]
                signature_of.setdefault(row_of_entry[start:end].tobytes(), []).append(col)
            if len(signature_of) > 1:
                # A genuine hash collision: this candidate bucket actually
                # holds two or more distinct presence patterns. Keep the
                # first sub-group under the existing candidate id and give
                # every other sub-group a fresh one, so they end up in
                # different final classes.
                for cols in list(signature_of.values())[1:]:
                    provisional[cols] = next_new_id
                    next_new_id += 1

    # Step 5 (naming/ordering) -- computed before step 4 (matrix assembly)
    # since the matrix needs the final, densely-numbered, representative
    # -ordered class ids.
    raw_ids, class_of_column = np.unique(provisional, return_inverse=True)
    n_classes = raw_ids.size
    class_of_column = class_of_column.astype(np.int64, copy=False)

    sequence_array = pa.array(kmer_sequences, type=pa.string())
    grouped = pa.table(
        {"raw_class": pa.array(class_of_column, type=pa.int64()), "kmer_sequence": sequence_array}
    ).group_by("raw_class").aggregate([("kmer_sequence", "min")])
    grouped_raw_class = np.asarray(grouped.column("raw_class"))
    grouped_representative = grouped.column("kmer_sequence_min")

    lexicographic_order = np.asarray(pc.sort_indices(grouped_representative))
    sorted_raw_class = grouped_raw_class[lexicographic_order]
    representative = grouped_representative.take(pa.array(lexicographic_order)).to_pylist()

    final_class_of_raw = np.empty(n_classes, dtype=np.int64)
    final_class_of_raw[sorted_raw_class] = np.arange(n_classes)
    final_class_of_column = final_class_of_raw[class_of_column]

    # Step 4: assemble the reduced matrix. Every member of a class shares,
    # by construction, one identical presence pattern, so remapping every
    # original stored entry straight to its final class id and letting
    # sum_duplicates() merge the (row, class) pairs that a multi-member
    # class produces reconstructs exactly that shared pattern -- no
    # per-class Python loop, no "pick one representative column" step.
    new_col_of_entry = final_class_of_column[col_of_entry]
    presence = np.ones(row_of_entry.size, dtype=np.uint8)
    reduced = sparse.coo_matrix(
        (presence, (row_of_entry, new_col_of_entry)), shape=(n_samples, n_classes)
    )
    reduced.sum_duplicates()
    reduced.data[:] = 1  # collapse the duplicate-count sums back to plain presence
    reduced = reduced.tocsr()

    members = pa.table(
        {
            "class_id": pa.array(final_class_of_column, type=pa.int64()),
            "kmer_sequence": sequence_array,
        }
    ).sort_by([("class_id", "ascending"), ("kmer_sequence", "ascending")])

    return EquivalenceClasses(matrix=reduced, representative=representative, members=members)
