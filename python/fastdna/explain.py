"""fastdna.explain -- which of your model's top features can you believe?

## The problem this answers

`fastdna.audit()` (see `docs/audit/audit-api.md`) answers "how much of my
score is real" by measuring the gap between random and lineage-blocked
cross-validation. That is half the question a reviewer actually has. The
other half is the one named directly by the paper `fastdna.cv` already
cites -- James, Williamson, Tino & Wheeler, *Whole-Genome Phenotype
Prediction with Machine Learning: Open Problems in Bacterial Genomics*
(arXiv 2502.07749):

    "attempts to extract any meaning from the predictive models are found
    to be corrupted by falsely identified 'causal' features."

A model can have an honest score and still point at the wrong k-mers.
`explain()` is the check for that: for each of a fitted model's top
features, it asks three separate questions and reports all three, rather
than collapsing them into one number.

## The three checks, and why each is necessary and none is sufficient alone

1. **Identifiability** (`fastdna.equivalence`). If a k-mer's presence/
   absence pattern across the cohort is identical to 846 other k-mers',
   none of the 847 is more "causal" than the rest -- they are one
   equivalence class, most often because they physically overlap in the
   genome (adjacent k-mers share almost all their sequence). Reporting one
   of them as *the* variant is exactly the false causal attribution the
   paper above describes. The class size is the honest answer.
2. **Lineage attribution** (`fastdna.cv.lineage_groups`). A feature
   restricted to a handful of lineages, out of many, is a candidate lineage
   marker rather than a phenotype marker -- the same population-structure
   confound `fastdna.audit()` measures at the level of the whole model,
   here applied to one feature.
3. **Within-lineage association**. A feature that is both prevalent across
   *many* lineages and whose presence still correlates with the phenotype
   *inside* each lineage separately survived the one confound stratification
   can control for. This uses the Cochran-Mantel-Haenszel test for
   stratified 2x2 tables (implemented by hand below -- `scipy.stats` alone
   is sufficient for it, so no new dependency is introduced for one
   closed-form statistic).

None of the three alone is enough: a feature can be identifiable (unique
pattern) and lineage-restricted (a marker, not a cause); or genuinely
prevalent across lineages but still fail the stratified test on this
cohort's numbers; or belong to a large equivalence class where its
particular member happened to survive the other two checks by chance.
`explain()` reports all three per feature and lets the verdict follow from
stating them plainly, rather than folding them into a single opaque score.

## Deliberately not attempted here

This does not compute a corrected p-value across genome-wide tests (that
is `fastdna.gwas.prefilter_association`'s job, with its own explicit
"unadjusted screen" warning), does not fit a mixed model, and does not
claim biological function -- only whether the *statistical* evidence for
"this feature marks the phenotype rather than the lineage" survives the
checks a naive read of `model.coef_`/`get_feature_names_out()` skips.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa

from fastdna import _PathLike
from . import _core

__all__ = ["explain", "ExplainReport", "FeatureExplanation"]

_ALPHA = 0.05
# A feature present in at most this fraction of the cohort's distinct
# lineages (and at least `_MIN_LINEAGES_FOR_RESTRICTION` total lineages
# exist) is flagged as lineage-restricted. A starting point, not a fitted
# threshold -- stated here so it is visible and overridable by reading the
# `n_lineages_present`/`n_lineages_total` fields directly rather than
# trusting the label alone.
_LINEAGE_RESTRICTED_FRACTION = 0.25
_MIN_LINEAGES_FOR_RESTRICTION = 4

#: How far apart an equivalence class's members may sit and still count as
#: one locus rather than a scattered, co-inherited set.
#:
#: Adjacent k-mers off a single variant tile a window not much wider than a
#: read: overlapping 21-mers spanning a 200 bp insert land within ~200 bp of
#: each other. A class whose members are thousands of bases apart is not one
#: variant seen many times; it is many variants that travel together because
#: they sit in the same clone. Measured on a constructed pair: 20
#: overlapping k-mers spanned 19 bp, 20 scattered ones 4,669 bp -- the two
#: regimes are separated by orders of magnitude, not by a fine cut, which is
#: why a round number serves and a fitted threshold would be false
#: precision.
_LOCALISED_SPAN_BP = 500

#: Members of a class actually located before deciding its extent. Each
#: lookup scans the reference, and a class can run to thousands of k-mers;
#: the span of a block is established by a sample of it long before the last
#: member is checked.
_SPATIAL_SAMPLE_CAP = 40


@dataclass(frozen=True)
class FeatureExplanation:
    """The three checks' results for one feature, plus the importance that
    put it in the report in the first place. See the module docstring for
    what each field means and why no single one is sufficient alone.
    """

    kmer: str
    kmer_u64: int
    importance: float

    equivalence_class_size: int
    identifiable: bool  # equivalence_class_size == 1

    n_lineages_present: int
    n_lineages_total: int
    lineage_restricted: bool

    association_p_value: Optional[float]  # None if not computable (see explain())
    survives_stratification: Optional[bool]  # None mirrors association_p_value

    annotation: Optional[str]  # best-effort gene/feature name, if given a reference

    verdict: str

    # --- spatial extent of the equivalence class (needs `annotation=`) ----
    #
    # `None` when no reference/annotation was supplied, since both are read
    # off real coordinates rather than inferred.
    #
    # Check #1 reports how MANY k-mers share this pattern. These report
    # WHERE they are, which is what separates two cases that check #1 scores
    # identically and that mean opposite things:
    #
    #   * a class whose members physically overlap -- adjacent k-mers off one
    #     locus -- spans tens of bases and touches one gene. That is a
    #     localised causal hypothesis, and it is the case this module's own
    #     docstring assumes ("most often because they physically overlap").
    #   * a class scattered across the genome shares its pattern because its
    #     members are co-inherited within a clone, not because they are one
    #     variant. That is a lineage signature wearing check #1's clothing.
    #
    # Measured on a constructed pair: 20 overlapping k-mers span 19 bp across
    # 1 gene; 20 scattered ones span 4,669 bp across 2. Reporting only the
    # class size collapses that distinction.
    #
    # arXiv 2502.07749 sec. 9.1 ("Spatial dependencies") asks for exactly
    # this -- encoding physical position so that "variants that are
    # physically proximal ... may be more likely to interact epistatically"
    # can constrain interpretation. Reference-free k-mer methods normally
    # cannot: they do not know where a k-mer lands. This one can, because
    # `fastdna.annotate` already resolves a k-mer to coordinates.
    class_span_bp: Optional[int] = None
    class_n_genes: Optional[int] = None
    class_localised: Optional[bool] = None  # confined to a single annotated feature

    def __repr__(self) -> str:
        return (
            f"FeatureExplanation(kmer={self.kmer!r}, importance={self.importance:.4g}, "
            f"verdict={self.verdict!r})"
        )


@dataclass(frozen=True)
class ExplainReport:
    """The result of :func:`explain`: one :class:`FeatureExplanation` per
    requested top feature, ranked by `|importance|` descending (the order
    `top_n` selected them in).
    """

    features: Tuple[FeatureExplanation, ...]
    n_lineages: int
    lineage_threshold: float

    @property
    def n_credible(self) -> int:
        return sum(1 for f in self.features if f.verdict == "credible candidate")

    def to_markdown(self) -> str:
        lines = [
            f"# FastDNA feature explanation -- {len(self.features)} top features",
            "",
            f"{self.n_lineages} lineages detected (threshold {self.lineage_threshold:g}). "
            f"{self.n_credible} of {len(self.features)} survive as credible candidates.",
            "",
            "| kmer | importance | equivalence class | lineages | p-value | verdict |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for f in self.features:
            p = f"{f.association_p_value:.3g}" if f.association_p_value is not None else "n/a"
            lines.append(
                f"| `{f.kmer}` | {f.importance:.4g} | {f.equivalence_class_size} | "
                f"{f.n_lineages_present}/{f.n_lineages_total} | {p} | {f.verdict} |"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ExplainReport({len(self.features)} features, {self.n_credible} credible, "
            f"{self.n_lineages} lineages)"
        )

    def __str__(self) -> str:
        return self.to_markdown()

    def _repr_html_(self):
        try:
            import pandas as pd
        except ImportError:
            return f"<pre>{self.to_markdown()}</pre>"

        df = pd.DataFrame(
            {
                "kmer": [f.kmer for f in self.features],
                "importance": [f.importance for f in self.features],
                "equivalence_class_size": [f.equivalence_class_size for f in self.features],
                "n_lineages_present": [f.n_lineages_present for f in self.features],
                "n_lineages_total": [f.n_lineages_total for f in self.features],
                "association_p_value": [f.association_p_value for f in self.features],
                "annotation": [f.annotation for f in self.features],
                "verdict": [f.verdict for f in self.features],
            }
        )
        header = (
            f"<p><b>ExplainReport</b> &mdash; {len(self.features)} features, "
            f"{self.n_credible} credible, {self.n_lineages} lineages</p>"
        )
        return header + df.to_html(index=False)


def _presence_matrix(vectorizer, paths):
    """The `(len(paths), len(vectorizer.vocabulary_))` presence (0/1)
    matrix for `paths`, regardless of `vectorizer.representation`.

    A fresh, already-"fitted" clone forces `representation="presence"`
    without disturbing the caller's own fitted vectorizer or requiring a
    second `representation` argument on `explain()` itself: `equivalence`/
    lineage attribution both need presence/absence specifically (not
    counts, not CLR), and reusing the original vectorizer's own
    `representation` here would give the wrong quantity if it were
    anything else. `check_is_fitted` only requires `vocabulary_` to be
    set, so copying it (and the other already-decided attributes) onto an
    unfitted clone is sufficient to call `.transform()` on it.

    `counts` is forced to `None` on the clone regardless of what the
    original `vectorizer` had, even though `sklearn.base.clone` would
    otherwise deep-copy it: `explain()`'s own `paths` parameter is
    documented to always be real FASTQ paths (lineage detection has to
    read the actual files), never the `sample_id` strings a `counts=`
    artifact expects `X` to be. Carrying the artifact through here would
    make `_count_cohort` try to look up file paths as sample_ids and fail.
    """
    from sklearn.base import clone

    presence_vec = clone(vectorizer)
    presence_vec.representation = "presence"
    presence_vec.counts = None
    presence_vec.vocabulary_ = vectorizer.vocabulary_
    presence_vec.n_features_in_ = vectorizer.n_features_in_
    presence_vec._feature_sequences_ = vectorizer._feature_sequences_
    return presence_vec.transform(paths)


def _cochran_mantel_haenszel(x, y, strata):
    """The Cochran-Mantel-Haenszel chi-square test for association between
    two binary variables `x`/`y`, controlling for `strata` (here, lineage).

    Implemented directly rather than via `statsmodels` (not a dependency of
    this package) because the statistic is a short, standard closed form:
    for each stratum `i` with 2x2 counts

                     y=1   y=0
              x=1     a_i   b_i
              x=0     c_i   d_i

    the CMH chi-square is `(sum(a_i) - sum(E[a_i]))^2 / sum(Var[a_i])` with
    `E[a_i] = (a_i+b_i)(a_i+c_i)/n_i` and
    `Var[a_i] = (a_i+b_i)(c_i+d_i)(a_i+c_i)(b_i+d_i) / (n_i^2 (n_i-1))`,
    1 degree of freedom. Strata with `n_i <= 1`, or with no variation in
    `x` or `y` (so `Var[a_i]` is 0), contribute nothing and are skipped --
    a stratum where every sample shares one lineage AND one phenotype value
    carries no information about whether `x` and `y` are associated *within*
    strata, which is exactly the quantity this test asks about.

    Returns `None` if fewer than two informative strata remain (not enough
    data to say anything), rather than a p-value computed from an unstable
    denominator.
    """
    from scipy.stats import chi2 as chi2_dist

    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    strata = np.asarray(strata)

    sum_a = 0.0
    sum_e = 0.0
    sum_var = 0.0
    informative_strata = 0

    for stratum in np.unique(strata):
        mask = strata == stratum
        xs, ys = x[mask], y[mask]
        n = xs.size
        if n <= 1:
            continue
        a = int(np.sum((xs == 1) & (ys == 1)))
        b = int(np.sum((xs == 1) & (ys == 0)))
        c = int(np.sum((xs == 0) & (ys == 1)))
        d = int(np.sum((xs == 0) & (ys == 0)))
        row1, row0 = a + b, c + d
        col1, col0 = a + c, b + d
        if row1 == 0 or row0 == 0 or col1 == 0 or col0 == 0:
            continue  # no variation in x or y within this stratum

        e = row1 * col1 / n
        var = (row1 * row0 * col1 * col0) / (n * n * (n - 1))
        if var <= 0:
            continue

        sum_a += a
        sum_e += e
        sum_var += var
        informative_strata += 1

    if informative_strata < 2 or sum_var <= 0:
        return None

    # Continuity-corrected CMH chi-square, 1 df.
    chi2_stat = (abs(sum_a - sum_e) - 0.5) ** 2 / sum_var
    return float(chi2_dist.sf(chi2_stat, df=1))


def explain(
    vectorizer: Any,  # fastdna.sklearn.KmerVectorizer; soft dependency, not imported here
    importances: Union[Sequence[float], np.ndarray],
    paths: Sequence[_PathLike],
    phenotype: Optional[Union[Sequence[int], np.ndarray]] = None,
    *,
    top_n: int = 20,
    k: int = 21,
    lineage_threshold: Optional[float] = None,
    annotation: Optional[Any] = None,  # fastdna.annotate.Annotation; soft dependency, not imported here
) -> ExplainReport:
    """Explains a fitted model's top features: for each, whether its
    presence/absence pattern is unique in the cohort (identifiability),
    how many lineages it appears in (lineage attribution), and whether it
    still associates with `phenotype` inside each lineage separately
    (stratified association). See the module docstring for why all three
    are reported rather than folded into one score.

    Parameters
    ----------
    vectorizer : fastdna.sklearn.KmerVectorizer
        Already fitted (`.vocabulary_` set). Its `representation` setting
        does not matter here -- see `_presence_matrix`.
    importances : array-like, length `len(vectorizer.vocabulary_)`
        A fitted model's per-feature importance, aligned with
        `vectorizer.get_feature_names_out()` -- e.g. `model.coef_[0]` for
        a linear model, or `model.feature_importances_` for a tree/boosting
        one. Ranked by absolute value, descending; `top_n` of them are
        explained.
    paths : sequence of str or pathlib.Path
        The real FASTQ(.gz) paths of the cohort `vectorizer` was fitted
        (or transformed) on -- **actual files**, not `sample_id` strings
        from a `fastdna.CohortCounts` artifact, even if `vectorizer.counts`
        was used to fit it: lineage detection sketches the files directly
        (`fastdna.cv.lineage_groups`), which needs real paths to read.
    phenotype : array-like of 0/1, length `len(paths)`, or None
        Aligned with `paths`. Required for the within-lineage association
        check; without it, `association_p_value`/`survives_stratification`
        are `None` for every feature and the verdict falls back to
        identifiability and lineage attribution alone.
    top_n : int, default 20
        How many top features (by `|importance|`) to explain.
    k : int, default 21
        The k-mer length forwarded to `fastdna.cv.lineage_groups` for the
        lineage-attribution and within-lineage association checks. This is
        independent of whatever `k` the caller's `vectorizer` was fitted
        with -- `explain()` sketches `paths` itself to detect lineages, it
        does not read `vectorizer`'s own k. The default (21) matches
        `fastdna.cv.lineage_groups`'s own default.
    lineage_threshold : float or None, default None
        Forwarded to `fastdna.cv.lineage_groups`; `None` uses that
        function's own default (0.01).
    annotation : fastdna.annotate.Annotation or None, default None
        If given, each feature's sequence is looked up against it
        (`fastdna.annotate.locate_kmer`, exact match only) and the first
        hit's gene/feature name is reported.

    Returns
    -------
    ExplainReport
    """
    from . import cv as _cv
    from .equivalence import collapse_equivalence_classes

    importances = np.asarray(importances, dtype=np.float64)
    if importances.shape[0] != len(vectorizer.vocabulary_):
        raise _core.InvalidConfigError(
            f"importances has {importances.shape[0]} entries but the vectorizer's "
            f"vocabulary has {len(vectorizer.vocabulary_)}. Pass the importances aligned "
            "with vectorizer.get_feature_names_out(), e.g. model.coef_[0]."
        )

    paths = [str(p) for p in paths]
    if len(paths) < 2:
        raise _core.InvalidConfigError(f"explain() needs at least 2 samples to detect lineages, got {len(paths)}")

    if phenotype is not None:
        phenotype = np.asarray(phenotype)
        if phenotype.shape[0] != len(paths):
            raise _core.InvalidConfigError(
                f"phenotype has {phenotype.shape[0]} entries but paths has {len(paths)}. "
                "Pass one phenotype value per path, in the same order."
            )
        distinct = set(np.unique(phenotype).tolist())
        if not distinct <= {0, 1}:
            raise _core.InvalidConfigError(
                f"explain()'s within-lineage association check only supports a binary "
                f"0/1 phenotype; got values {sorted(distinct)}. Pass phenotype=None to skip "
                "that check for a continuous or multi-class outcome."
            )

    ranked = np.argsort(-np.abs(importances))[: min(top_n, len(importances))]

    presence = _presence_matrix(vectorizer, paths).tocsc()
    equivalence = collapse_equivalence_classes(presence, list(vectorizer._feature_sequences_))
    # `sequence -> equivalence class size`, resolved once for the whole
    # vocabulary rather than once per top feature.
    class_sizes = np.zeros(len(vectorizer._feature_sequences_), dtype=np.int64)
    # class_id -> the sequences in it, for the spatial extent of a class.
    # Built from the same table `members_by_sequence` inverts, so no extra
    # pass over the cohort.
    class_members_by_id = {}
    for _seq, _cid in zip(
        equivalence.members.column("kmer_sequence").to_pylist(),
        equivalence.members.column("class_id").to_pylist(),
    ):
        class_members_by_id.setdefault(_cid, []).append(_seq)

    members_by_sequence = dict(
        zip(
            equivalence.members.column("kmer_sequence").to_pylist(),
            equivalence.members.column("class_id").to_pylist(),
        )
    )
    class_member_counts = np.bincount(
        equivalence.members.column("class_id").to_pylist(),
        minlength=len(equivalence.representative),
    )
    for i, seq in enumerate(vectorizer._feature_sequences_):
        class_sizes[i] = class_member_counts[members_by_sequence[seq]]

    labels = _cv.lineage_groups(
        paths,
        k=k,
        distance_threshold=lineage_threshold if lineage_threshold is not None else 0.01,
    )
    n_lineages_total = int(np.unique(labels).size)

    loaded_annotation = annotation

    features = []
    for column in ranked.tolist():
        kmer_bits = int(vectorizer.vocabulary_[column])
        sequence = vectorizer._feature_sequences_[column]
        importance = float(importances[column])

        col_presence = np.asarray(presence[:, column].todense()).ravel().astype(np.int64)
        present_rows = np.flatnonzero(col_presence)
        lineages_present = np.unique(labels[present_rows]) if present_rows.size else np.array([])
        n_present = int(lineages_present.size)

        eq_size = int(class_sizes[column])

        association_p = None
        survives = None
        if phenotype is not None:
            association_p = _cochran_mantel_haenszel(col_presence, phenotype, labels)
            if association_p is not None:
                survives = association_p < _ALPHA

        lineage_restricted = (
            n_lineages_total >= _MIN_LINEAGES_FOR_RESTRICTION
            and n_present <= max(1, int(n_lineages_total * _LINEAGE_RESTRICTED_FRACTION))
        )

        gene_name = None
        class_span_bp = class_n_genes = class_localised = None
        if loaded_annotation is not None:
            from .annotate import locate_kmer

            hits = locate_kmer(loaded_annotation, sequence)
            if hits:
                gene_name = hits[0].gene_name or hits[0].feature_type

            # Where the REST of this k-mer's equivalence class sits. Check #1
            # says how many share the pattern; this says whether they are one
            # locus or scattered, which is the difference between a localised
            # causal hypothesis and a clone signature (see
            # `FeatureExplanation.class_span_bp`).
            class_id = members_by_sequence.get(sequence)
            if class_id is not None:
                siblings = class_members_by_id.get(class_id, ())
                # Capped: a class can run to thousands of k-mers and each
                # lookup scans the reference. The span of a large block is
                # already established by a sample of it.
                positions, genes = [], set()
                for sibling in siblings[:_SPATIAL_SAMPLE_CAP]:
                    sibling_hits = locate_kmer(loaded_annotation, sibling)
                    if not sibling_hits:
                        continue
                    positions.append(sibling_hits[0].start)
                    if sibling_hits[0].gene_name:
                        genes.add(sibling_hits[0].gene_name)
                if positions:
                    class_span_bp = int(max(positions) - min(positions))
                    class_n_genes = len(genes)
                    # One annotated feature and a span no wider than the
                    # k-mers themselves could tile: overlapping members off a
                    # single locus.
                    class_localised = class_n_genes <= 1 and class_span_bp <= _LOCALISED_SPAN_BP

        if eq_size > 1:
            verdict = f"non-identifiable (linked block of {eq_size} k-mers)"
            # The spatial half of the answer, when it is available: the same
            # block size means opposite things depending on this.
            if class_localised is True:
                verdict += f", localised to {gene_name or 'one feature'}"
            elif class_localised is False:
                verdict += f", scattered over {class_span_bp} bp"
        elif lineage_restricted:
            verdict = "lineage marker"
        elif survives is True:
            verdict = "credible candidate"
        elif survives is False:
            verdict = "no evidence within lineages"
        else:
            verdict = "insufficient data" if phenotype is not None else "identifiable, not lineage-restricted"

        features.append(
            FeatureExplanation(
                kmer=sequence,
                kmer_u64=kmer_bits,
                importance=importance,
                equivalence_class_size=eq_size,
                identifiable=eq_size == 1,
                n_lineages_present=n_present,
                n_lineages_total=n_lineages_total,
                lineage_restricted=lineage_restricted,
                association_p_value=association_p,
                survives_stratification=survives,
                annotation=gene_name,
                verdict=verdict,
                class_span_bp=class_span_bp,
                class_n_genes=class_n_genes,
                class_localised=class_localised,
            )
        )

    return ExplainReport(
        features=tuple(features),
        n_lineages=n_lineages_total,
        lineage_threshold=lineage_threshold if lineage_threshold is not None else 0.01,
    )
