"""fastdna.workflow -- one entry point for the canonical cohort question:
"I have FASTQ files and a phenotype, tell me what predicts it."

## Why this exists

`fastdna.gwas`, `fastdna.equivalence`, `fastdna.cv`, `fastdna.rules`,
`fastdna.evaluation` and `fastdna.annotate` each solve one well-scoped part
of a k-mer association study, and each is independently useful and
independently documented -- that separation is deliberate (see
`docs/philosophy-narrow-not-broad.md`: complete within FastDNA's own
territory, not a monolith). But composing all six by hand for the ordinary
case means learning six APIs, getting the sample-id/row-order bookkeeping
right across every call (see `gwas._resolve_cohort`'s own warnings about
exactly this class of silent bug), and remembering which stage's warning
means what. `AssociationWorkflow` is that composition, done once, correctly,
with every intermediate result kept and every stage individually skippable
or replaceable.

This module is a **convenience layer, not a new capability**. It calls the
same public functions a hand-written script would, in the same order, with
the same validation and the same warnings -- nothing here relaxes
`gwas.prefilter_association`'s `ScreeningOnlyWarning`, `rules.
SetCoveringClassifier.predict_proba`'s hard-decision caveat, or `evaluation.
calibration_report`'s `UncalibratedScoresWarning`. A caller who outgrows the
convenience layer drops straight back to the underlying modules; every
parameter they accept remains reachable here through this module's own
`*_kwargs` dicts or explicit overrides.

**One deliberate exception.** When `cv=` replaces the default
`cv.LineageKFold` with a splitter that is not an instance of it, `run()`
emits `NaiveCrossValidationWarning` and records which splitter actually
produced `cv_predictions` in that table's own Arrow schema metadata. No
single one of the six composed modules can make this particular check by
itself: `sklearn.model_selection.cross_val_predict` has no notion of
lineage at all, and `cv.LineageKFold` only knows about itself, not about
whatever a caller substituted in its place. A hand-written script stitching
the same six modules together would therefore not get this warning for
free -- it is the one validation rule this convenience layer adds rather
than merely forwards, and it exists because population-structure-inflated
CV is precisely the risk the rest of this project is built around (see
`fastdna.cv`'s own module docstring, and `fastdna.audit`, which quantifies
the resulting gap directly for a given estimator and cohort). A caller who
overrides `cv=` still gets exactly the number they asked for -- it is never
withheld -- just never silently alone.

## The default pipeline

    from fastdna.workflow import AssociationWorkflow

    workflow = AssociationWorkflow(fastq_paths, phenotype, n_splits=5)
    result = workflow.run()

    print(result.classifier.explain())
    print(result.precision_recall.average_precision)
    workflow.plot_significance()
    workflow.plot_population_structure()

`run()` executes, in order:

1. `gwas.cohort_presence_matrix` -- one sparse `samples x k-mers` matrix.
2. `equivalence.collapse_equivalence_classes` (skippable via
   `collapse_equivalence=False`) -- decorrelates columns that are identical
   across this cohort, the DBGWAS-unitig-flavored dimensionality reduction
   `equivalence.py`'s own docstring describes.
3. `cv.lineage_groups` (skippable by passing precomputed `groups=`) --
   Mash-distance-derived lineage labels, so the evaluation below is not
   leaked by population structure.
4. A classifier -- `rules.SetCoveringClassifier()` by default, or any
   unfitted scikit-learn-compatible estimator passed as `classifier=` --
   fitted once on the full cohort (for interpretation: `.explain()`,
   `.rules_`) and evaluated honestly via `sklearn.model_selection.
   cross_val_predict` under a `cv.LineageKFold` built from the lineage
   groups (skippable via `cv=False`, or replaceable with any other
   scikit-learn splitter via `cv=<splitter>` -- doing so emits
   `NaiveCrossValidationWarning`, see "One deliberate exception" above).
5. `evaluation.precision_recall_report` and `evaluation.calibration_report`
   over those held-out predictions (only computed alongside step 4).
6. `gwas.prefilter_association` (skippable via `screen=False`) -- a fast,
   unadjusted screen for `plot_significance()` to visualize. Its
   `ScreeningOnlyWarning` is not suppressed.
7. `annotate.load_annotation` + `annotate.annotate_rule` for every rule of
   the fitted classifier (only when both `reference_fasta` and
   `annotation_path` are given, and only meaningful for a classifier that
   exposes `.rules_` -- the default `SetCoveringClassifier` does).
8. `cv.permutation_importance_pvalues` (only when `n_permutations` is
   given -- it costs `n_permutations + 1` model refits, so it is opt-in,
   not automatic). This stage needs a classifier exposing `coef_` or
   `feature_importances_` (see that function's own docstring); the default
   `SetCoveringClassifier` exposes neither, and the resulting `ValueError`
   from `permutation_importance_pvalues` is not caught here -- it says
   exactly what it needs.

`workflow.plot_significance()`/`workflow.plot_population_structure()` wrap
`fastdna.plotting` over the results `run()` already computed; both need
`matplotlib` (and the latter `scipy`), imported lazily by `plotting.py`
itself, and both raise a clear `ImportError` naming the package if it is
missing rather than failing deep inside a plotting call.

## What this module does not do

It does not add any statistics or plotting style beyond what its six
composed modules already have -- see each of their own module docstrings
for the honesty caveats that still apply unchanged here (unadjusted
screening, hard-decision `predict_proba`, Mash-derived rather than
phylogeny-derived kinship, and so on). The one validation rule it does add
beyond forwarding theirs -- `NaiveCrossValidationWarning` when `cv=`
overrides the default `cv.LineageKFold` -- is described above, under "One
deliberate exception", precisely because it is the only one. It also does not pick a
phenotype-encoding, threshold, or "best" model for you: `classifier=` and
every `*_kwargs` dict stay fully overridable, and the result is every
intermediate artifact, not a single verdict.
"""

from __future__ import annotations

import inspect
import os
import warnings
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Sequence, Union

import numpy as np
import pyarrow as pa
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

from . import _core
from .annotate import annotate_rule, load_annotation
from .cv import LineageKFold, lineage_groups, permutation_importance_pvalues
from .equivalence import EquivalenceClasses, collapse_equivalence_classes
from .evaluation import CalibrationReport, PrecisionRecallReport, calibration_report, precision_recall_report
from .gwas import _resolve_cohort, cohort_presence_matrix, kinship_matrix, prefilter_association
from .rules import SetCoveringClassifier

if TYPE_CHECKING:
    # matplotlib is a soft dependency, imported lazily by fastdna.plotting itself;
    # this import only runs for static type checkers.
    import matplotlib.axes

__all__ = ["AssociationWorkflow", "AssociationResult", "NaiveCrossValidationWarning"]


class NaiveCrossValidationWarning(UserWarning):
    """Raised by `AssociationWorkflow.run()` when `cv=` replaces the
    default `cv.LineageKFold` with a splitter that is not an instance of
    it, so this run's `cv_predictions`/`precision_recall`/`calibration`
    were **not** evaluated under this project's leakage-safe default.

    This is not a claim that the resulting score is wrong -- only that it
    is unprotected. `fastdna.cv`'s own module docstring lays out exactly
    why that matters for a clonal cohort: a random split scatters
    near-copies of the same lineage across the train/test boundary, so a
    held-out fold is not really held out, and the model can score highly
    by recognizing the lineage rather than the phenotype. `fastdna.audit`
    exists specifically to quantify that gap for a given estimator and
    cohort (its own `score_random` is, deliberately, exactly this same
    kind of naive score) -- run it alongside this workflow before treating
    the number this warning is attached to as a performance estimate on
    genomes this cohort did not contain.

    A distinct category (rather than a bare `UserWarning`), matching
    `gwas.ScreeningOnlyWarning`'s own reasoning, so a caller who has
    deliberately chosen a different splitter -- to reproduce a known
    random-CV baseline, say, or because an independent check already ruled
    out lineage structure in this cohort -- can silence *this* warning
    specifically (`warnings.filterwarnings("ignore",
    category=NaiveCrossValidationWarning)`) without silencing every other
    warning FastDNA might legitimately need to raise.
    """


class AssociationResult(NamedTuple):
    """Everything `AssociationWorkflow.run()` computed. Every field is a
    plain, independently usable artifact from one of the composed modules
    -- see `AssociationWorkflow`'s docstring for which stage produced it and
    how to reproduce it by hand.

    Attributes
    ----------
    matrix : scipy.sparse.csr_matrix
        The cohort feature matrix actually used for fitting/screening --
        post-equivalence-collapse if `collapse_equivalence=True` (the
        default), the raw `cohort_presence_matrix` output otherwise.
    sample_ids : list of str
        Row labels of `matrix`, in cohort order.
    kmer_sequences : list of str
        Column labels of `matrix` -- equivalence-class representatives if
        collapsed, raw k-mers otherwise.
    equivalence : fastdna.equivalence.EquivalenceClasses or None
        `None` when `collapse_equivalence=False`.
    groups : numpy.ndarray
        Lineage labels, one per sample, aligned to `sample_ids`.
    classifier : fitted estimator
        The classifier fitted on the full cohort (for interpretation --
        `.explain()`, `.rules_` if it is a `SetCoveringClassifier`). Not
        the same fitted object used for any one cross-validation fold.
    cv_predictions : pyarrow.Table or None
        Columns `sample_id`, `y_true`, `cv_score` -- one row per sample,
        `cv_score` its held-out `predict_proba` positive-class score from
        `sklearn.model_selection.cross_val_predict`. `None` when `cv=False`.
        Its Arrow schema metadata always records `fastdna.
        cv_lineage_blocked` (`"true"`/`"false"`) and `fastdna.cv_splitter`
        (the splitter's own `repr()`), plus `fastdna.leakage_risk` when it
        is `"false"` -- so whether this table came from the leakage-safe
        default survives a save-to-Parquet-and-reload, the same
        survives-serialization convention `gwas.prefilter_association` uses
        for its own screening-only caveat. See `AssociationWorkflow`'s `cv`
        parameter and `NaiveCrossValidationWarning` for what sets it to
        `"false"`.
    precision_recall : fastdna.evaluation.PrecisionRecallReport or None
        Computed from `cv_predictions`; `None` when `cv=False`.
    calibration : fastdna.evaluation.CalibrationReport or None
        Computed from `cv_predictions`; `None` when `cv=False`.
    screening : pyarrow.Table or None
        `gwas.prefilter_association`'s output; `None` when `screen=False`.
    importance : pyarrow.Table or None
        `cv.permutation_importance_pvalues`'s output; `None` unless
        `n_permutations` was given.
    annotations : pyarrow.Table or None
        `annotate.annotate_rule`'s output, concatenated across every rule
        of `classifier`; `None` unless both `reference_fasta` and
        `annotation_path` were given and `classifier` learned at least one
        rule.
    """

    matrix: object
    sample_ids: list
    kmer_sequences: list
    equivalence: Optional[EquivalenceClasses]
    groups: np.ndarray
    classifier: object
    cv_predictions: Optional[pa.Table]
    precision_recall: Optional[PrecisionRecallReport]
    calibration: Optional[CalibrationReport]
    screening: Optional[pa.Table]
    importance: Optional[pa.Table]
    annotations: Optional[pa.Table]

    def to_report(
        self,
        path: Union[str, os.PathLike],
        *,
        calibration_interval: Optional[tuple[float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,  # forwarded to fastdna.report.to_report(), e.g. title=
    ) -> None:
        """`fastdna.report.to_report()`, populated from this result's own
        `classifier` and `calibration` (`None` when `run()` was called with
        `cv=False` -- the report then shows "no calibration report was
        provided" rather than raising), plus `n_samples`/
        `n_features_evaluated` filled in from `sample_ids`/`kmer_sequences`.

        `calibration_interval` is not one of `run()`'s own outputs (this
        workflow never calls `fastdna.calibration.calibrate` itself -- see
        that module's docstring for why a `CalibratedEstimator` cannot
        stand in for `classifier=` here, since it wraps an already-fitted
        estimator rather than being one itself). Pass the `(p0, p1)` result
        of a `CalibratedEstimator.predict_interval()` call you made
        separately, if you have one, to include it in the figure.

        `metadata`, if given, is merged over (not replacing) the two
        auto-derived entries above -- pass `metadata={"n_samples": ...}` to
        override one specifically. `**kwargs` (e.g. `title=`) are forwarded
        to `fastdna.report.to_report()` unchanged.
        """
        from .report import to_report as _to_report

        resolved_metadata = {
            "n_samples": len(self.sample_ids),
            "n_features_evaluated": len(self.kmer_sequences),
        }
        resolved_metadata.update(metadata or {})
        return _to_report(
            path,
            classifier=self.classifier,
            calibration=self.calibration,
            calibration_interval=calibration_interval,
            metadata=resolved_metadata,
            **kwargs,
        )


def _align_phenotype(phenotype, sample_ids):
    """`phenotype` as a `numpy.ndarray` in `sample_ids` order.

    Accepts a `{sample_id: value}` mapping (order-independent, matched by
    id -- the safe form whenever `paths` itself was a mapping with ids that
    do not equal file-name order) or a plain array/list already aligned to
    `sample_ids` positionally. A mismatch in either form is refused rather
    than silently truncated or reordered: a misaligned phenotype attaches
    every downstream test, rule and score to the WRONG sample, invisibly.
    """
    if isinstance(phenotype, Mapping):
        missing = [s for s in sample_ids if s not in phenotype]
        extra = [s for s in phenotype if s not in sample_ids]
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing a value for {missing}")
            if extra:
                parts.append(f"has unexpected id(s) {extra}")
            raise _core.InvalidConfigError(
                f"phenotype mapping does not match the cohort's sample ids: "
                f"{' and '.join(parts)}. Every sample id cohort_presence_matrix() derived "
                f"({sample_ids}) needs exactly one phenotype value, and vice versa."
            )
        return np.asarray([phenotype[s] for s in sample_ids])

    arr = np.asarray(phenotype)
    if arr.ndim != 1 or len(arr) != len(sample_ids):
        shape = arr.shape if arr.ndim != 1 else (len(arr),)
        raise _core.InvalidConfigError(
            f"phenotype has shape {shape} but the cohort has {len(sample_ids)} samples. Pass "
            "phenotype as a {sample_id: value} mapping to avoid depending on row order, or a "
            "1-D array/list with exactly one entry per sample, in the same order as `paths` "
            "(the same order cohort_presence_matrix() returns as sample_ids)."
        )
    return arr


def _fit_accepts_feature_names(estimator):
    """Whether `estimator.fit()` takes a `feature_names` keyword -- true for
    `SetCoveringClassifier` and any duck-typed lookalike, false for a plain
    scikit-learn estimator. Used to decide whether the final full-cohort fit
    can be given real k-mer names for `.explain()`/`.rules_`/annotation, or
    must be called with the ordinary two-argument `fit(X, y)`.
    """
    try:
        parameters = inspect.signature(type(estimator).fit).parameters
    except (TypeError, ValueError):
        return False
    return "feature_names" in parameters


def _new_classifier(classifier):
    """A fresh, unfitted clone of `classifier`, or a fresh
    `SetCoveringClassifier()` when none was given -- the one default this
    module picks, since it is the classifier every other composed stage
    (`.explain()`, `export_rules_fasta()`, `annotate_rule()`) is built to
    read from.
    """
    return clone(classifier) if classifier is not None else SetCoveringClassifier()


def _cv_provenance_metadata(cv_splitter: Any) -> Dict[str, str]:
    """Arrow schema metadata for `AssociationResult.cv_predictions`
    recording whether `cv_splitter` is this project's leakage-safe default
    -- so the answer survives being saved to Parquet and read back
    separately from `NaiveCrossValidationWarning`, the same
    "the caveat lives in the data, not only in a transient warning"
    convention `gwas.prefilter_association`'s own screening-only metadata
    already established (see its `_SCREENING_METADATA`).

    Only `cv_splitter`'s *type* is inspected, matching the check `run()`
    itself makes before emitting `NaiveCrossValidationWarning`: an
    explicitly-constructed `cv.LineageKFold` (custom `k=`, `sketch_size=`,
    or a precomputed `groups=`) is still recorded as lineage-blocked, since
    it still carries the same "never splits a lineage across train/test"
    guarantee, whatever its parameters.
    """
    is_lineage_blocked = isinstance(cv_splitter, LineageKFold)
    metadata = {
        "fastdna.cv_lineage_blocked": "true" if is_lineage_blocked else "false",
        "fastdna.cv_splitter": repr(cv_splitter),
    }
    if not is_lineage_blocked:
        metadata["fastdna.leakage_risk"] = (
            "this cv_score was not evaluated under cv.LineageKFold -- a random split on a clonal "
            "cohort can let the model score highly by recognizing lineage rather than phenotype "
            "(see fastdna.cv's own module docstring). Compare against fastdna.audit()'s "
            "score_lineage/gap for the same cohort and estimator before treating this as a "
            "performance estimate on genomes this cohort did not contain."
        )
    return metadata


class AssociationWorkflow:
    """Orchestrates a full k-mer-cohort-to-phenotype association study from
    FASTQ files, chaining `fastdna.gwas`, `fastdna.equivalence`,
    `fastdna.cv`, a scikit-learn-compatible classifier, `fastdna.evaluation`,
    `fastdna.gwas.prefilter_association`, `fastdna.annotate` and (via
    `plot_*` methods) `fastdna.plotting`. See the module docstring for the
    exact pipeline and its honesty guarantees.

    Parameters
    ----------
    paths : iterable of path, or mapping of str to path
        As in `gwas.cohort_presence_matrix`: an iterable derives sample ids
        from file names; a `{sample_id: path}` mapping names them
        explicitly. At least 2 samples, no duplicate ids or files.
    phenotype : array-like or mapping of str to value
        One label per sample. A `{sample_id: value}` mapping is matched by
        id (order-independent); a plain array/list must already be aligned
        to `paths`' order. See `_align_phenotype`.
    collapse_equivalence : bool, default True
        Runs `equivalence.collapse_equivalence_classes` on the cohort
        matrix before anything downstream sees it. `False` keeps the raw,
        uncollapsed k-mer columns.
    matrix_kwargs : dict, optional
        Forwarded to `gwas.cohort_presence_matrix` (e.g. `k`, `min_count`,
        `min_samples`, `max_kmers`).
    groups : array-like, optional
        Precomputed lineage labels, one per sample, aligned to `paths`'
        order. `None` (the default) derives them with `cv.lineage_groups`.
    lineage_kwargs : dict, optional
        Forwarded to `cv.lineage_groups` when `groups` is not given (e.g.
        `k`, `sketch_size`, `distance_threshold`).
    n_splits : int, default 5
        Folds for the default `cv.LineageKFold` evaluation splitter. Must
        not exceed the number of distinct lineages found -- `LineageKFold`
        raises a specific, actionable error if it does (see its own
        docstring), and this is not caught or relaxed here.
    cv : scikit-learn cross-validation splitter, False, or None
        `None` (the default) builds `cv.LineageKFold(n_splits=n_splits,
        groups=<the resolved groups>)`. Pass any other scikit-learn
        splitter to use it instead (the resolved `groups` are still
        computed, e.g. for the `n_permutations` stage, but this workflow's
        own leakage-safe splitter is then not what evaluates the model).
        Doing so emits `NaiveCrossValidationWarning`, not suppressed, and
        marks `AssociationResult.cv_predictions` accordingly in that
        table's own Arrow schema metadata -- see that warning's and that
        field's own docstrings for exactly what each records. Passing an
        actual `cv.LineageKFold` instance here (e.g. one built with a
        custom `k=`/`sketch_size=`/`distance_threshold=`, or from a
        precomputed `groups=` of your own) is a non-default but still
        leakage-safe choice and triggers neither -- only the splitter's
        *type* is checked, never its parameters.
        Pass `False` to skip cross-validated evaluation entirely --
        `cv_predictions`, `precision_recall` and `calibration` all come
        back `None`, and the classifier is still fitted on the full cohort.
    classifier : unfitted scikit-learn-compatible estimator, optional
        `None` (the default) uses `rules.SetCoveringClassifier()`. Cloned
        (never mutated) for the full-cohort fit and, when `cv` is not
        `False`, for `sklearn.model_selection.cross_val_predict`, which
        clones it again per fold.
    screen : bool, default True
        Runs `gwas.prefilter_association` (its `ScreeningOnlyWarning` is
        not suppressed) so `plot_significance()` has something to draw.
    screen_kwargs : dict, optional
        Forwarded to `gwas.prefilter_association` (e.g. `test`, `top_n`).
    n_permutations : int, optional
        `None` (the default) skips `cv.permutation_importance_pvalues`
        entirely -- it costs `n_permutations + 1` model refits. Set it to
        run that stage; it needs a classifier exposing `coef_` or
        `feature_importances_`, which the default `SetCoveringClassifier`
        does not (its own docstring explains why a hard-coverage rule set
        has no such quantity) -- passing `n_permutations` with the default
        classifier surfaces `permutation_importance_pvalues`'s own
        `ValueError` unchanged, not a workflow-specific one.
    permutation_random_state : int, optional
        Forwarded to `cv.permutation_importance_pvalues` as `random_state`.
    reference_fasta, annotation_path : path, optional
        Must be given together or not at all. When given, every rule of
        the fitted classifier (via its `.rules_`, as
        `rules.SetCoveringClassifier` exposes) is located in the reference
        with `annotate.load_annotation`/`annotate.annotate_rule`. Given
        without a classifier that exposes `.rules_`, this raises
        `ValueError` naming the classifier -- gene-annotation lookup has no
        meaning for a model with no rule literals to locate.
    annotate_kwargs : dict, optional
        Forwarded to `annotate.annotate_rule` (e.g. `max_mismatches`).

    Attributes
    ----------
    result_ : AssociationResult or None
        The last `run()` call's result, cached so `plot_significance()`/
        `plot_population_structure()` need no argument. `None` before the
        first `run()`.
    """

    def __init__(
        self,
        paths: Union[Iterable[Union[str, os.PathLike]], Mapping[str, Union[str, os.PathLike]]],
        phenotype: Union[Mapping[str, Any], Sequence[Any], np.ndarray],
        *,
        collapse_equivalence: bool = True,
        matrix_kwargs: Optional[Dict[str, Any]] = None,
        groups: Optional[Union[Sequence[Any], np.ndarray]] = None,
        lineage_kwargs: Optional[Dict[str, Any]] = None,
        n_splits: int = 5,
        cv: Any = None,  # None (default), False (skip CV), or a scikit-learn-compatible splitter instance
        classifier: Any = None,  # unfitted scikit-learn-compatible estimator; sklearn estimator types have no precise common annotation
        screen: bool = True,
        screen_kwargs: Optional[Dict[str, Any]] = None,
        n_permutations: Optional[int] = None,
        permutation_random_state: Optional[int] = None,
        reference_fasta: Optional[Union[str, os.PathLike]] = None,
        annotation_path: Optional[Union[str, os.PathLike]] = None,
        annotate_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if (reference_fasta is None) != (annotation_path is None):
            raise _core.InvalidConfigError(
                "reference_fasta and annotation_path must be given together, or not at all -- "
                f"got reference_fasta={reference_fasta!r}, annotation_path={annotation_path!r}. "
                "Gene-annotation lookup needs both the reference sequence and its annotation."
            )

        self.paths = paths
        self.phenotype = phenotype
        self.collapse_equivalence = collapse_equivalence
        self.matrix_kwargs = dict(matrix_kwargs) if matrix_kwargs else {}
        self.groups = groups
        self.lineage_kwargs = dict(lineage_kwargs) if lineage_kwargs else {}
        self.n_splits = n_splits
        self.cv = cv
        self.classifier = classifier
        self.screen = screen
        self.screen_kwargs = dict(screen_kwargs) if screen_kwargs else {}
        self.n_permutations = n_permutations
        self.permutation_random_state = permutation_random_state
        self.reference_fasta = reference_fasta
        self.annotation_path = annotation_path
        self.annotate_kwargs = dict(annotate_kwargs) if annotate_kwargs else {}

        self.result_ = None
        self._cohort_mapping = None
        self._distance_matrix = None
        self._kinship_sample_ids = None

    def run(self) -> AssociationResult:
        """Executes the full pipeline (see the module docstring) and
        returns the `AssociationResult`, also cached as `self.result_`.
        """
        sample_ids, path_strings = _resolve_cohort(self.paths, caller="AssociationWorkflow")
        cohort_mapping = dict(zip(sample_ids, path_strings))
        self._cohort_mapping = cohort_mapping

        matrix, matrix_sample_ids, kmer_sequences = cohort_presence_matrix(cohort_mapping, **self.matrix_kwargs)
        assert matrix_sample_ids == sample_ids  # guaranteed by construction: same mapping, same order

        y = _align_phenotype(self.phenotype, sample_ids)

        equivalence = None
        if self.collapse_equivalence:
            equivalence = collapse_equivalence_classes(matrix, kmer_sequences)
            matrix, kmer_sequences = equivalence.matrix, equivalence.representative
        else:
            # cohort_presence_matrix() stores raw per-sample depth, not
            # presence -- every stored entry is already strictly positive
            # (see gwas.py), so setting every stored value to 1 recovers
            # plain 0/1 presence without touching the sparsity pattern.
            # Needed because every downstream stage (SetCoveringClassifier
            # above all -- see its own validation) treats this matrix as a
            # binary presence matrix, matching equivalence.py's own
            # "presence, not counts" convention for its reduced output.
            matrix = matrix.tocsr(copy=True)
            matrix.data = np.ones_like(matrix.data, dtype=np.uint8)

        if self.groups is not None:
            groups = np.asarray(self.groups)
            if len(groups) != len(sample_ids):
                raise _core.InvalidConfigError(
                    f"groups has {len(groups)} entries but the cohort has {len(sample_ids)} samples "
                    f"({sample_ids}). Pass one lineage label per sample, in the same order as `paths`, "
                    "or omit groups= to derive them automatically with cv.lineage_groups()."
                )
        else:
            groups = lineage_groups(path_strings, **self.lineage_kwargs)

        fitted_classifier = _new_classifier(self.classifier)
        if _fit_accepts_feature_names(fitted_classifier):
            fitted_classifier.fit(matrix, y, feature_names=kmer_sequences)
        else:
            fitted_classifier.fit(matrix, y)

        cv_predictions = precision_recall = calibration = None
        if self.cv is not False:
            cv_splitter = self.cv if self.cv is not None else LineageKFold(n_splits=self.n_splits, groups=groups)
            if not isinstance(cv_splitter, LineageKFold):
                # See the module docstring's "One deliberate exception": no
                # one of the six composed modules can make this check by
                # itself, since none of them ever sees both "the
                # leakage-safe default this workflow would have built" and
                # "the splitter actually passed" at once.
                warnings.warn(
                    f"cv={cv_splitter!r} replaces this workflow's default cv.LineageKFold, so this "
                    "run's cv_predictions/precision_recall/calibration were NOT evaluated under a "
                    "leakage-safe split. fastdna.cv's own module docstring explains why that matters: "
                    "a random split on a clonal cohort -- every real bacterial or viral one -- "
                    "scatters near-copies of the same lineage across the train/test boundary, so the "
                    "held-out score this produces can reflect the model recognizing lineage rather "
                    "than phenotype, not a performance estimate on genomes this cohort did not "
                    "contain. This result is not withheld -- only reported without the corroboration "
                    "this workflow gives it by default, and the same caveat is recorded in "
                    "cv_predictions' own Arrow schema metadata so it survives being saved and "
                    "reloaded. Compare it against fastdna.audit()'s score_lineage/gap for the "
                    "same cohort and estimator, or drop cv= (or pass cv=None) to use cv.LineageKFold "
                    "instead.",
                    NaiveCrossValidationWarning,
                    stacklevel=2,
                )
            cv_estimator = _new_classifier(self.classifier)
            proba = cross_val_predict(cv_estimator, matrix, y, cv=cv_splitter, method="predict_proba")
            cv_score = proba[:, 1]
            cv_predictions = pa.table(
                {"sample_id": sample_ids, "y_true": list(y), "cv_score": cv_score.tolist()}
            ).replace_schema_metadata(_cv_provenance_metadata(cv_splitter))
            precision_recall = precision_recall_report(y, cv_score)
            calibration = calibration_report(y, cv_score)

        screening = None
        if self.screen:
            screening = prefilter_association(matrix, y, kmer_sequences, **self.screen_kwargs)

        importance = None
        if self.n_permutations is not None:
            importance = permutation_importance_pvalues(
                _new_classifier(self.classifier),
                matrix,
                y,
                kmer_sequences,
                n_permutations=self.n_permutations,
                random_state=self.permutation_random_state,
                groups=groups,
            )

        annotations = None
        if self.reference_fasta is not None:
            if not hasattr(fitted_classifier, "rules_"):
                raise _core.InvalidConfigError(
                    "reference_fasta/annotation_path were given, but the fitted classifier "
                    f"({type(fitted_classifier).__name__}) exposes no `.rules_` to annotate -- "
                    "gene-annotation lookup is only meaningful for a rule-based model like "
                    "rules.SetCoveringClassifier (the default classifier=). Omit "
                    "reference_fasta/annotation_path when using a different classifier."
                )
            rules = fitted_classifier.rules_
            if rules:
                annotation = load_annotation(self.reference_fasta, self.annotation_path)
                annotations = pa.concat_tables(
                    [annotate_rule(rule, annotation, **self.annotate_kwargs) for rule in rules]
                )

        result = AssociationResult(
            matrix=matrix,
            sample_ids=sample_ids,
            kmer_sequences=kmer_sequences,
            equivalence=equivalence,
            groups=groups,
            classifier=fitted_classifier,
            cv_predictions=cv_predictions,
            precision_recall=precision_recall,
            calibration=calibration,
            screening=screening,
            importance=importance,
            annotations=annotations,
        )
        self.result_ = result
        return result

    def plot_significance(self, **kwargs: Any) -> "matplotlib.axes.Axes":
        """`fastdna.plotting.plot_significance` over this workflow's own
        `result_.screening`. Needs `run()` (with `screen=True`, the
        default) to have been called first. `**kwargs` are forwarded to
        `plotting.plot_significance` (e.g. `x`, `positions`, `ax`).
        """
        if self.result_ is None:
            raise _core.InvalidConfigError("plot_significance() needs run() to have been called first.")
        if self.result_.screening is None:
            raise _core.InvalidConfigError(
                "plot_significance() needs a screening table, but this workflow was run with "
                "screen=False, so result_.screening is None. Re-run with screen=True (the "
                "default) to produce one."
            )
        from . import plotting

        return plotting.plot_significance(self.result_.screening, **kwargs)

    def plot_population_structure(self, **kwargs: Any) -> "matplotlib.axes.Axes":
        """`fastdna.plotting.plot_population_structure` over this cohort's
        Mash-distance matrix, colored by the lineage groups `run()` already
        resolved. Needs `run()` to have been called first.

        The distance matrix itself is not one of `run()`'s own
        intermediate results (`cv.lineage_groups` computes one internally
        but does not return it): this method obtains it with a second,
        independent call to `gwas.kinship_matrix` -- a second sketch pass
        over the cohort -- the first time it is called, and caches the
        result for later calls. `**kwargs` are forwarded to
        `plotting.plot_population_structure` (e.g. `kind`, `ax`); passing
        `groups=` here overrides the cached lineage groups.
        """
        if self.result_ is None:
            raise _core.InvalidConfigError("plot_population_structure() needs run() to have been called first.")

        if self._distance_matrix is None:
            similarity, kinship_sample_ids = kinship_matrix(self._cohort_mapping)
            self._distance_matrix = 1.0 - similarity
            self._kinship_sample_ids = kinship_sample_ids

        kwargs.setdefault("groups", self.result_.groups)
        from . import plotting

        return plotting.plot_population_structure(self._distance_matrix, self._kinship_sample_ids, **kwargs)
