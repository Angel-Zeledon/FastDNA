"""fastdna.multiomics -- joining k-mer features with other omics/clinical
data by sample ID.

Real studies are rarely genomics-only: clinical metadata, other omics
layers (transcriptomics, proteomics, metabolomics), and k-mer-derived
genomic features all need to line up correctly by sample before a model can
use them together. The hard part is not the join mechanics -- pandas/polars
already do joins -- it's getting *sample ID matching and mismatch handling*
right and explicit: what happens when a sample exists in one layer but not
another, when IDs need light normalization to match across layers, and when
a caller needs to know exactly which samples got dropped and why.

**Status: frozen.** Per `docs/audit/PLAN.md` §2 ("Qué se poda"), this
module is frozen: stable, not accepting new features, and a candidate for
extraction into a separate `fastdna-contrib` package in a future release.
Freezing is not deleting -- see that section for the full reasoning behind
the boundary.

This module is deliberately independent of `fastdna.sklearn.KmerVectorizer`
(built by a parallel agent, not merged when this module was written): the
join utilities here work over *any* per-sample tabular feature
representation -- a `{sample_id: pyarrow.Table}` dict of single-sample
`fastdna.count()` results (what `kmer_feature_table` below builds from),
or a single wide table already keyed by sample ID (what a
`KmerVectorizer`-style matrix, or any other omics layer, looks like) -- so
they work today, and keep working once `KmerVectorizer` lands.

Three pieces:

- `kmer_feature_table`: a simple, self-contained wide-table builder that
  turns a set of FASTQ files into one k-mer feature table keyed by
  `sample_id`. Not a general-purpose ML vectorizer (that's
  `KmerVectorizer`'s job) -- it exists so this module has a concrete, real,
  testable genomic feature table to join against other omics data.
- `join_omics_layers`: joins several such tables (or arbitrary other omics
  tables) on a shared sample ID column, returning both the combined table
  and an explicit report of what happened to every sample ID.
- `normalize_sample_ids`: a small, conservative, rule-based string
  normalizer for the common "sample_001" vs "Sample-1" vs "sample1"
  cosmetic-mismatch problem -- not a fuzzy matcher, and not meant to be one.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

import pyarrow as pa
import pyarrow.compute as pc

from . import _column_as_array
from . import _core
from . import count as _count

if TYPE_CHECKING:
    import pandas

__all__ = ["kmer_feature_table", "join_omics_layers", "normalize_sample_ids", "JoinReport"]


def _sample_id_from_filename(path):
    """The "filename" convention for `id_from`: the file's stem, e.g.
    `Path("data/sample_001.fastq.gz").stem` -- `pathlib.Path.stem` only
    strips the *last* suffix, so a `.fastq.gz` file's stem is
    `sample_001.fastq`, not `sample_001`. FASTQ files very commonly carry a
    double extension, and a stray `.fastq` left in every sample_id would
    silently break joins against a clinical sheet that used the bare name --
    exactly the kind of cosmetic mismatch this module exists to avoid
    creating in the first place. Any other trailing `.gz`/`.fastq`/`.fq`
    suffixes are stripped the same way.
    """
    p = Path(path)
    name = p.name
    for suffix in (".gz", ".fastq", ".fq"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name


_ID_FROM_STRATEGIES = {
    "filename": _sample_id_from_filename,
}


def kmer_feature_table(
    sample_paths: Union[Dict[str, Union[str, os.PathLike]], Iterable[Union[str, os.PathLike]]],
    *,
    k: int = 31,
    min_count: int = 1,
    top_features: Optional[int] = None,
    id_from: str = "filename",
) -> pa.Table:
    """Runs `fastdna.count()` once per sample and assembles the results
    into a single wide `pyarrow.Table`: one row per `sample_id`, one column
    per k-mer, plus a `sample_id` column.

    `sample_paths` is either a `dict` mapping `{sample_id: fastq_path}`
    (explicit IDs, used as-is), or a list/iterable of FASTQ paths, in which
    case each path's `sample_id` is derived per `id_from`. Only
    `id_from="filename"` (the default, and currently the only supported
    value) is implemented: the file's name with a trailing `.fastq`/`.fq`/
    `.gz` suffix stripped (see `_sample_id_from_filename`), e.g.
    `"data/sample_001.fastq.gz"` -> `"sample_001"`. Passing any other
    `id_from` value raises `ValueError` rather than silently falling back,
    since a caller relying on a convention this module does not actually
    implement is a real bug, not a preference to be silently ignored.

    The k-mer vocabulary (the set of columns emitted, beyond `sample_id`)
    is the union of every k-mer seen across all samples, unless
    `top_features` is set, in which case only the `top_features` k-mers
    with the highest *total* count summed across all samples are kept
    (ties broken by the k-mer's own sequence, for a deterministic column
    order). This is a much simpler selection rule than `KmerVectorizer`'s
    own vocabulary selection (see the design note at the top of this
    module) -- appropriate for this module's own scope of "produce one
    concrete, joinable feature table", not for general-purpose ML feature
    selection.

    A k-mer absent from a given sample's own counts becomes `0` in that
    sample's row -- ordinary sparse-count semantics, not a missing value:
    the k-mer simply was not observed in that sample, which is different
    from (and much more common than) "we don't know its count".

    Column order: `sample_id` first, then k-mer columns sorted by k-mer
    sequence, for a deterministic, reproducible table across runs.
    """
    # Mirrors `KmerVectorizer.fit()`'s own validation: a negative value
    # would silently slice from the *end* of the ranking (dropping the top
    # k-mers instead of keeping them), and 0 would silently produce a
    # feature-less table.
    if top_features is not None and (
        not isinstance(top_features, int) or isinstance(top_features, bool) or top_features <= 0
    ):
        raise _core.InvalidConfigError(f"top_features must be a positive int or None, got {top_features!r}")

    if isinstance(sample_paths, dict):
        id_to_path = dict(sample_paths)
    else:
        if id_from not in _ID_FROM_STRATEGIES:
            raise _core.InvalidConfigError(
                f"id_from={id_from!r} is not supported; only {sorted(_ID_FROM_STRATEGIES)} "
                "are implemented -- pass an explicit {sample_id: path} dict instead if you "
                "need a different convention."
            )
        derive = _ID_FROM_STRATEGIES[id_from]
        id_to_path = {}
        for path in sample_paths:
            sample_id = derive(path)
            if sample_id in id_to_path:
                raise _core.InvalidConfigError(
                    f"two input paths both derive sample_id {sample_id!r} under "
                    f"id_from={id_from!r} ({id_to_path[sample_id]!r} and {path!r}); "
                    "pass an explicit {sample_id: path} dict to disambiguate."
                )
            id_to_path[sample_id] = path

    # Everything below this point is a columnar operation over Arrow arrays,
    # not a Python loop over cells. That is not a stylistic preference: this
    # function is O(samples x vocabulary) by construction -- a 50-sample
    # cohort with a 10,000-k-mer vocabulary is a 500,000-cell table -- and
    # the previous implementation touched every one of those cells at least
    # three times from Python. It built a `{kmer: frequency}` dict per
    # sample (two `to_pylist()` calls and one dict construction each), then
    # accumulated cross-sample totals by iterating every (sample, k-mer)
    # pair one at a time, then materialised the output with one
    # `dict.get()` per cell. Arrow does each of those three passes in C++
    # instead, leaving Python with `samples` kernel invocations and one
    # transpose rather than several million interpreter steps.
    #
    # The output is unchanged, column for column and value for value: the
    # k-mer alphabet is ASCII `ACGT`, so Arrow's bytewise string ordering
    # and Python's `sorted()` agree exactly, and `to_pylist()` on the
    # `uint32` frequency column yields the same Python ints the dict held.
    sample_ids = list(id_to_path)
    if not sample_ids:
        # No samples: there is nothing to concatenate or group, and
        # `pa.concat_arrays([])` cannot infer a type from an empty list.
        # Returns exactly what the row-by-row version returned here.
        return pa.table({"sample_id": sample_ids})

    per_sample_sequences = []
    per_sample_frequencies = []
    for sample_id in sample_ids:
        # `with_sequence=True`: this function's output columns *are* k-mer
        # sequences, so the decoded column is not skippable overhead here
        # the way it is for the ML-facing paths that stay in `u64` space.
        table = _count(id_to_path[sample_id], k=k, min_count=min_count, with_sequence=True).table
        # `_column_as_array` flattens the `ChunkedArray` a `Table.column()`
        # hands back, so `pa.concat_arrays` below sees one contiguous
        # `Array` per sample regardless of how many chunks it arrived in.
        per_sample_sequences.append(_column_as_array(table.column("kmer_sequence")))
        per_sample_frequencies.append(_column_as_array(table.column("frequency")))

    # Total count per k-mer across all samples -- used both to pick
    # `top_features` (when set) and, either way, to fix a deterministic
    # column order. One hash-aggregate over the stacked per-sample columns
    # replaces the nested dict accumulation.
    stacked = pa.table(
        {
            "kmer": pa.concat_arrays(per_sample_sequences),
            "total": pa.concat_arrays(per_sample_frequencies),
        }
    )
    totals = stacked.group_by("kmer").aggregate([("total", "sum")])
    kmers = totals.column("kmer")

    if top_features is not None:
        # `(-count, kmer)` in the old Python sort key, expressed as Arrow's
        # own two-key sort: highest total first, ties broken by the k-mer's
        # own sequence. The kept slice is then re-sorted by sequence alone,
        # because the emitted column order is always alphabetical -- the
        # ranking only decides *which* k-mers survive, never where they sit.
        ranked = pc.sort_indices(
            totals, sort_keys=[("total_sum", "descending"), ("kmer", "ascending")]
        )
        kept = kmers.take(ranked[:top_features])
        vocabulary = kept.take(pc.sort_indices(kept)).to_pylist()
    else:
        vocabulary = kmers.take(pc.sort_indices(kmers)).to_pylist()

    vocabulary_array = pa.array(vocabulary, type=pa.string())
    rows = []
    for sequences, frequencies in zip(per_sample_sequences, per_sample_frequencies):
        # One vectorized hash lookup per *sample* in place of one Python
        # dict lookup per cell. `index_in` resolves each vocabulary entry to
        # its position in this sample's own k-mer list (null when the sample
        # never saw it), `take` gathers the frequencies at those positions,
        # and the nulls become 0 -- "not observed in this sample", which is
        # the sparse-count semantics documented above, not a missing value.
        positions = pc.index_in(vocabulary_array, value_set=sequences)
        aligned = pc.take(frequencies, positions)
        rows.append(pc.fill_null(aligned, 0).to_pylist())

    # `rows` is one list per sample; the table wants one column per k-mer.
    # `zip(*rows)` is that transpose, done once in C rather than by
    # re-indexing every sample once per k-mer column.
    columns = {"sample_id": sample_ids}
    columns.update(zip(vocabulary, map(list, zip(*rows))))

    return pa.table(columns)


def normalize_sample_ids(ids: Iterable[Any], *, strategy: str = "lower_strip_punct") -> List[str]:
    """Normalizes a list/Series of sample-ID strings so that IDs differing
    only cosmetically (case, punctuation, whitespace) can be made to match
    across layers before a join.

    Exactly one strategy is implemented, `"lower_strip_punct"` (the
    default): for each ID, lowercase it, strip leading/trailing whitespace,
    then replace every run of one or more characters that is not an ASCII
    letter or digit with a single `_`, and finally strip any leading or
    trailing `_` left over from that replacement. Note that `"Sample-1"`
    and `"sample_001"` do NOT normalize to the same string under this rule
    (`"Sample-1"` -> `"sample_1"`, `"sample_001"` -> `"sample_001"`): this
    function fixes *punctuation/case/whitespace* mismatches, not numeric
    zero-padding differences, and does not try to. Concretely:

        "Sample-1"    -> "sample_1"
        " sample_1 "  -> "sample_1"
        "SAMPLE 1"    -> "sample_1"
        "sample--1"   -> "sample_1"
        "sample.1"    -> "sample_1"

    What this deliberately does NOT do (by design, not by omission): it
    does not strip or add zero-padding (`"sample_1"` and `"sample_001"`
    stay distinct), it does not reorder tokens, it does not guess at
    abbreviations or synonyms, and it never merges two IDs that were
    genuinely different before normalization into a false match beyond
    the simple case/punctuation/whitespace rule above. This is a
    conservative, rule-based transform, not fuzzy matching: a wrong
    automatic match silently joining the wrong two samples' data together
    is a much worse failure mode than a join that drops an unmatched
    sample and reports it (see `join_omics_layers`'s `JoinReport`) -- so
    when normalization is not enough to make two IDs match, this function
    leaves them unmatched rather than guessing further.

    Edge case, stated rather than hidden: an ID consisting entirely of
    whitespace and/or punctuation (`"  "`, `"---"`) normalizes to the empty
    string, and several such IDs therefore all collide on `""`. That is the
    documented rule applied consistently, not a special case -- but it does
    mean a column of blank IDs will appear to "match" each other, so check
    for empty results before joining on them if blank IDs are possible in
    your data.

    Raises `ValueError` for any `strategy` other than
    `"lower_strip_punct"`.

    Accepts a `list`, a `pandas.Series`, or a `polars.Series` (or anything
    else iterable of strings); always returns a plain `list[str]` in the
    same order as the input, regardless of input type, since a normalized
    ID list is typically about to be assigned back onto some table's
    column rather than used as a Series in its own right.
    """
    if strategy != "lower_strip_punct":
        raise _core.InvalidConfigError(
            f"strategy={strategy!r} is not supported; only 'lower_strip_punct' is implemented."
        )

    result = []
    for raw in ids:
        s = str(raw).strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = s.strip("_")
        result.append(s)
    return result


@dataclass
class JoinReport:
    """The explicit account of what `join_omics_layers` did, so a caller
    never has to guess why they ended up with fewer rows than expected.

    - `layer_sample_ids`: `{layer_name: set of sample_id values present in
      that layer's input table}`, exactly as they appeared in the `on`
      column before joining (no normalization is applied by
      `join_omics_layers` itself -- see its docstring).
    - `kept_sample_ids`: the sample IDs present in the final combined
      table, in row order.
    - `dropped_sample_ids`: `{sample_id: sorted list of layer names that
      sample_id was MISSING from}` -- only samples actually dropped from
      the final result appear here. Under `how="inner"`, this is every
      sample_id not present in every layer; under `how="outer"`, this is
      always empty (nothing is dropped). A per-layer `how` dict (see
      `join_omics_layers`) produces the drop set implied by that
      configuration.
    - `row_count`: `len(kept_sample_ids)`, i.e. the final table's row
      count -- redundant with `len(kept_sample_ids)` but kept as a field
      in its own right so a caller reading only `report.row_count` does
      not have to know that equivalence holds.
    """

    layer_sample_ids: dict = field(default_factory=dict)
    kept_sample_ids: list = field(default_factory=list)
    dropped_sample_ids: dict = field(default_factory=dict)
    row_count: int = 0


def _missing_pandas():
    """`join_omics_layers` returns a `pandas.DataFrame` by design (see its
    own docstring's return-type note): most other-omics data already lives
    in pandas in practice, and downstream ML code overwhelmingly expects a
    DataFrame here. That makes pandas a de-facto requirement of this one
    join path even though it is not a runtime dependency of the `fastdna`
    package as a whole (`pyproject.toml`'s `project.dependencies` stays
    `["pyarrow>=14"]`). A bare `import pandas` deep inside this module would
    surface as a raw `ModuleNotFoundError` from somewhere the caller never
    typed; this names the package and the install command instead, matching
    `fastdna.embed`'s `_missing_dependency` convention.
    """
    return ImportError(
        "fastdna.multiomics.join_omics_layers() requires the optional "
        "'pandas' package, which is not installed. Install it with: "
        "pip install pandas"
    )


def _as_pandas(table):
    """Coerces a pandas.DataFrame, polars.DataFrame, or pyarrow.Table into
    a pandas.DataFrame (a copy, so later mutation -- e.g. normalizing the
    `on` column -- never touches the caller's own object).
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise _missing_pandas() from e

    if isinstance(table, pd.DataFrame):
        return table.copy()
    if isinstance(table, pa.Table):
        return table.to_pandas()
    # polars.DataFrame, without a hard import dependency on polars.
    if hasattr(table, "to_pandas"):
        return table.to_pandas()
    raise TypeError(
        f"unsupported table type {type(table)!r}; expected pandas.DataFrame, "
        "polars.DataFrame, or pyarrow.Table"
    )


def _fill_missing(df, on, column_owner_ids):
    """Fills values the `outer` merge left as NaN for rows a given layer
    did not contribute -- and *only* those -- "appropriately per column
    dtype":

    - numeric columns (any pandas dtype `is_numeric_dtype` accepts, i.e.
      every int/float/bool dtype) are filled with `0` -- matching this
      module's own sparse-count convention for k-mer columns (`0` means
      "not observed in this sample", not "unknown"), and a reasonable
      default for other numeric omics measurements in the same spirit
      (absence reads as "no signal" rather than a guessed average).
    - non-numeric columns (strings, categoricals, objects) are filled with
      the literal string `"missing"`, so a missing clinical/categorical
      value is visibly flagged rather than silently rendered as an empty
      string or coerced into some other category's value.

    `column_owner_ids` is `{column_name: set of sample_id values the one
    layer that column came from actually carried}` (built by the caller
    from the pre-merge, pre-rename-collision-safe per-layer frames -- see
    `join_omics_layers`). A cell is only a candidate for filling when its
    row's `on` value is NOT in that column's owning layer -- i.e. the row
    genuinely did not come from that layer, so pandas' outer merge is what
    produced the NaN there. A NaN in a row that *is* one of the owning
    layer's own sample IDs is the caller's own data (a viral load nobody
    recorded, a status nobody filled in) and is left untouched: this
    function cannot tell caller-supplied NaN from merge-introduced NaN by
    looking at the value alone, so it uses row membership instead, computed
    before the merge ever ran.

    The `on` column itself is never touched (every row has a real sample_id
    by construction of the outer join).

    One pandas artifact worth knowing about rather than being surprised by:
    an integer column that acquired any missing cell during the outer merge
    is promoted to float64 by pandas *before* this fill runs, so its filled
    value reads as `0.0` rather than `0`. The value is right either way; the
    dtype is pandas' own doing, not a choice made here, and is left alone
    rather than cast back -- a blind cast to int would corrupt any genuinely
    float-valued omics column that happened to be missing a cell.
    """
    import pandas as pd

    for column in df.columns:
        if column == on:
            continue
        owner_ids = column_owner_ids.get(column)
        if owner_ids is None:
            continue
        # Only rows whose sample_id is absent from the owning layer are
        # candidates: a NaN there was introduced by the merge, not carried
        # in from the caller's own data.
        merge_introduced = ~df[on].isin(owner_ids)
        if not merge_introduced.any():
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            fill_value = 0
        else:
            fill_value = "missing"
        df.loc[merge_introduced, column] = df.loc[merge_introduced, column].fillna(fill_value)
    return df


def join_omics_layers(
    layers: Dict[str, Any],  # each value: pandas.DataFrame, polars.DataFrame, or pyarrow.Table
    *,
    on: str = "sample_id",
    how: Union[str, Dict[str, str]] = "inner",
) -> Tuple["pandas.DataFrame", JoinReport]:
    """Joins several omics layers into a single `pandas.DataFrame`, on the
    `on` column (default `"sample_id"`) that each layer's table must carry
    alongside its own feature columns.

    `layers` is a `dict` `{layer_name: table}`, where each `table` is a
    `pandas.DataFrame`, `polars.DataFrame`, or `pyarrow.Table` (e.g.
    `{"kmers": kmer_feature_table(...), "clinical": my_clinical_df,
    "transcriptomics": my_expression_df}`). Every table is coerced to
    pandas internally (see the return-type note below) before joining.

    **This function requires pandas**, unlike the rest of `fastdna`, which
    stays importable with only `pyarrow` installed: its return type is a
    `pandas.DataFrame` (see the return-type note below), and every input
    -- including a plain `pyarrow.Table` -- is coerced through pandas on
    the way in. If `pandas` is not installed, calling this function raises
    a clear `ImportError` naming the package and `pip install pandas`,
    rather than a raw `ModuleNotFoundError` from deep inside this module;
    `import fastdna.multiomics` itself, and `kmer_feature_table` /
    `normalize_sample_ids`, do not require pandas at all.

    A feature column name appearing in more than one layer (anything other
    than `on`) is renamed to `<column>_<layer_name>` in *every* layer that
    carries it, so no layer's values can silently overwrite another's and
    every such column says which layer it came from. Columns unique to a
    single layer keep their original names untouched. Three layers each
    with a `value` column therefore yield `value_a`, `value_b`, `value_c`
    -- not pandas' own `value_a`, `value_b`, `value`, which leaves the
    last one ambiguous.

    No ID normalization is performed here -- if two layers' `on` columns
    use cosmetically different conventions for the same samples, call
    `normalize_sample_ids` on each layer's `on` column yourself first (see
    this module's end-to-end usage). Silently normalizing inside a generic
    join function would risk merging two IDs that were not actually meant
    to match; that decision belongs to the caller, made explicitly, not
    guessed at here.

    Every layer's `on` column must be free of duplicate values: a layer
    carrying the same sample_id twice would silently fan out into multiple
    rows once `DataFrame.merge` runs (a cartesian product if both sides of
    a merge have the same ID duplicated), quietly giving one sample extra
    weight with nothing in `JoinReport` to explain why the row count grew.
    `ValueError` is raised naming the layer and the duplicated ID(s)
    instead; resolve duplicates (rename, aggregate, or drop) before
    calling.

    `how`:
    - `"inner"` (default): keep only sample IDs present in *every* layer --
      the safe default for training a model that needs every modality
      present for every sample.
    - `"outer"`: keep every sample ID present in *any* layer. A cell whose
      row's sample_id is genuinely absent from the layer that column came
      from -- i.e. the row this `outer` merge added -- is filled per
      `_fill_missing`'s documented, dtype-appropriate rule (`0` for numeric
      columns, the string `"missing"` for everything else) rather than
      left as pandas' own `NaN`, since a `NaN` k-mer count would
      misrepresent "not observed" (which is `0`) as "unknown" (which `NaN`
      actually means). This fill is scoped to merge-introduced cells only:
      a `NaN` the caller's own input already had for a sample the owning
      layer *did* contribute (an unrecorded lab measurement, a blank
      clinical field) is left as `NaN` -- `_fill_missing` cannot tell
      caller-supplied "unknown" from merge-introduced "absent" by value
      alone, so it decides by row membership, computed before the merge
      runs, not by blanket-filling every NaN the merged frame happens to
      contain.
    - a `dict` `{layer_name: "inner" | "outer"}`: per-layer control -- a
      layer marked `"inner"` must contribute every sample_id in the final
      result (any sample_id missing from it is dropped from the result
      entirely, same as a plain `how="inner"` layer would be), while a
      layer marked `"outer"` may be missing some sample_ids (filled per
      the rule above) without affecting which rows survive. Every layer in
      `layers` must have an entry in this dict, or `ValueError` is raised
      -- silently defaulting an unlisted layer's behavior would be exactly
      the kind of unexplained row-count surprise this function's report
      exists to prevent.

    Returns `(combined_df, join_report)`:
    - `combined_df`: a `pandas.DataFrame` (chosen deliberately for this
      function's return type -- most other-omics data already lives in
      pandas in practice, and downstream ML code overwhelmingly expects a
      DataFrame here, even though `kmer_feature_table` above returns a
      `pyarrow.Table` and individual layers may be pandas, polars, or
      pyarrow on the way in).
    - `join_report`: a `JoinReport` (see its own docstring) recording,
      per layer, exactly which sample IDs were present, which final
      sample IDs were kept, which were dropped and from which layer(s)
      they were missing, and the final row count -- so a caller never has
      to guess why they ended up with fewer rows than expected.
    """
    if not layers:
        raise _core.InvalidConfigError("join_omics_layers requires at least one layer")

    if isinstance(how, dict):
        missing = set(layers) - set(how)
        if missing:
            raise _core.InvalidConfigError(
                f"how dict is missing an entry for layer(s) {sorted(missing)}; every layer in "
                "`layers` must have an explicit 'inner' or 'outer' entry in a per-layer `how`."
            )
        extra = set(how) - set(layers)
        if extra:
            raise _core.InvalidConfigError(f"how dict names layer(s) not present in `layers`: {sorted(extra)}")
        per_layer_how = dict(how)
    elif how in ("inner", "outer"):
        per_layer_how = {name: how for name in layers}
    else:
        raise _core.InvalidConfigError(f"how must be 'inner', 'outer', or a per-layer dict, got {how!r}")

    frames = {}
    layer_sample_ids = {}
    for name, table in layers.items():
        df = _as_pandas(table)
        if on not in df.columns:
            raise _core.InvalidConfigError(f"layer {name!r} has no {on!r} column (columns: {list(df.columns)})")
        # `layer_sample_ids` (below) is built with `set(...)`, which
        # silently collapses duplicates, while the sequential
        # `DataFrame.merge` a few lines down fans them out -- two sample
        # IDs with one duplicated becomes 3+ rows, and duplicates on both
        # sides of a merge give a full cartesian product. `JoinReport`
        # exists so a caller never has to guess why a join changed their
        # row count; it already handles *fewer* rows (via
        # `dropped_sample_ids`) but had nothing to say about *more*. Rather
        # than silently multiply a sample's weight in a cohort study, this
        # is refused outright -- a caller with genuinely duplicated
        # sample_ids (e.g. technical replicates) must resolve that
        # explicitly (rename, aggregate, or pick one) before joining.
        duplicate_ids = sorted(df.loc[df[on].duplicated(keep=False), on].unique().tolist())
        if duplicate_ids:
            raise _core.InvalidConfigError(
                f"layer {name!r} has duplicate {on!r} value(s) {duplicate_ids}; "
                "join_omics_layers requires unique sample IDs per layer, since a "
                "duplicate silently fans out into multiple rows during the merge. "
                "Resolve duplicates (rename, aggregate, or drop) before joining."
            )
        frames[name] = df
        layer_sample_ids[name] = set(df[on].tolist())

    names = list(layers)

    # Determine the final surviving sample_id set from the per-layer how:
    # start from the union of every layer's ids, then intersect down with
    # each layer marked "inner".
    all_ids = set()
    for ids in layer_sample_ids.values():
        all_ids |= ids
    kept_ids = set(all_ids)
    for name in names:
        if per_layer_how[name] == "inner":
            kept_ids &= layer_sample_ids[name]

    # Disambiguate feature columns that appear in more than one layer
    # BEFORE merging, rather than leaning on pandas' `suffixes=`. Relying
    # on pandas here is subtly wrong for more than two layers: `suffixes`
    # applies per pairwise merge, so in a sequential fold the third and
    # later layers' colliding columns come out unsuffixed (`value_a`,
    # `value_b`, `value`) and the caller cannot tell which layer the bare
    # one came from. Renaming upfront makes every collided column carry
    # its own layer's name, consistently, no matter how many layers there
    # are -- which is the whole point of a join utility that reports what
    # it did. Columns unique to one layer keep their original names.
    seen = {}
    for name in names:
        for column in frames[name].columns:
            if column != on:
                seen[column] = seen.get(column, 0) + 1
    collided = {column for column, n in seen.items() if n > 1}
    if collided:
        for name in names:
            renames = {c: f"{c}_{name}" for c in frames[name].columns if c in collided}
            if renames:
                frames[name] = frames[name].rename(columns=renames)

    # Every column (other than `on`) now belongs to exactly one layer's
    # frame -- collided names were just renamed to make that true. Record
    # which sample IDs that owning layer actually carried, so
    # `_fill_missing` can tell "this NaN is here because the merge
    # introduced it" (row's sample_id not in the owner's set) from "this
    # NaN was already in the caller's own data for a row the owning layer
    # genuinely contributed" (row's sample_id IS in the owner's set) --
    # only the former should ever be filled. See `_fill_missing`'s
    # docstring and this function's `how="outer"` bullet above.
    column_owner_ids = {}
    for name in names:
        for column in frames[name].columns:
            if column != on:
                column_owner_ids[column] = layer_sample_ids[name]

    # Sequentially outer-merge every layer, then restrict to the surviving
    # sample_id set and fill per dtype.
    combined = frames[names[0]]
    for name in names[1:]:
        combined = combined.merge(frames[name], on=on, how="outer")

    combined = combined[combined[on].isin(kept_ids)].reset_index(drop=True)

    combined = _fill_missing(combined, on, column_owner_ids)

    # Preserve a stable, deterministic row order: sorted by the join key.
    combined = combined.sort_values(on, kind="stable").reset_index(drop=True)

    kept_sample_ids = combined[on].tolist()
    dropped = {}
    for sample_id in sorted(all_ids - set(kept_sample_ids)):
        missing_from = sorted(name for name in names if sample_id not in layer_sample_ids[name])
        dropped[sample_id] = missing_from

    report = JoinReport(
        layer_sample_ids=layer_sample_ids,
        kept_sample_ids=kept_sample_ids,
        dropped_sample_ids=dropped,
        row_count=len(kept_sample_ids),
    )

    return combined, report
