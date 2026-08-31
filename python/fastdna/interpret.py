"""fastdna.interpret -- feature importances back to real, literal DNA.

FastDNA's `sklearn.KmerVectorizer` (docs/ml-genomics-roadmap.md, item 1)
turns a list of FASTQ paths into a numeric feature matrix whose columns are
k-mers, in the order `get_feature_names_out()` reports them. Any model
trained on that matrix -- a linear model's `.coef_`, a tree ensemble's
`.feature_importances_`, or a reduced SHAP-values array -- assigns one
importance value per column. This module's whole job is closing that loop
back to biology: given an importances array and the matching feature-name
list, hand back the k-mers that mattered most, as literal sequence strings
a biologist can paste into BLAST -- not an opaque embedding dimension.

**Contract note (read this before wiring this module up to the real
`KmerVectorizer`):** this module was built in an isolated worktree in
parallel with the agent building `python/fastdna/sklearn.py`'s
`KmerVectorizer`, which had not landed yet at dispatch time (see
docs/ml-genomics-roadmap.md, item 6). It is written and tested against the
*documented* contract for that class:

    - `.fit(X)` learns `self.vocabulary_` from `X` alone.
    - `.transform(X)` returns a `scipy.sparse.csr_matrix` of shape
      `(len(X), len(self.vocabulary_))`.
    - `.get_feature_names_out()` returns the vocabulary as decoded k-mer
      sequence strings, in the same column order `.transform()` produces.

This module makes no import of `fastdna.sklearn` and has no hard dependency
on it: any `(importances, feature_names)` pair of matching length works,
regardless of what produced them. That also means nothing here has been
exercised against the *actual* shipped `KmerVectorizer` -- only against a
hand-written stub matching its documented shape (see
`python/tests/test_interpret.py`). Once both pieces are merged, add a real
integration test that fits an actual `KmerVectorizer`, trains a small model
on its `.transform()` output, and runs `top_features`/
`export_top_features_fasta` on the result, to confirm the real class's
`get_feature_names_out()` ordering matches what this module assumes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, Union

import pyarrow as pa

from fastdna import _PathLike
from . import _core

# Deliberately no top-level `numpy` import: like scikit-learn/scipy/UMAP/
# Biopython elsewhere in this package's convention, anything not needed by
# every caller of this module stays out of its hard import cost.
# `top_features`/`export_top_features_fasta` -- the two core deliverables --
# work on plain Python lists/tuples and array-likes that support `len()`
# and indexing (including numpy arrays, without numpy itself needing to be
# installed just to call them). Only `explain_with_shap` needs numpy, and
# it imports it lazily, alongside `shap` itself, inside that one function.
# `numpy` is imported here only under TYPE_CHECKING, purely for annotating
# the "array-like" parameters below without adding a runtime import.
if TYPE_CHECKING:
    import numpy as np

__all__ = ["top_features", "export_top_features_fasta", "explain_with_shap"]


def _validate_lengths(importances, feature_names):
    importances = list(importances)
    feature_names = list(feature_names)
    if any(hasattr(x, "__len__") for x in importances):
        raise _core.InvalidConfigError(
            f"importances must be a 1-D array-like (one value per feature), "
            f"got what looks like a 2-D/nested structure instead. If this "
            f"came from LogisticRegression.coef_ on a multi-class problem, "
            f"index a single class's row first, e.g. `model.coef_[0]`."
        )
    importances = [float(x) for x in importances]
    if len(importances) != len(feature_names):
        raise _core.InvalidConfigError(
            f"importances and feature_names must have the same length -- "
            f"got {len(importances)} importances but {len(feature_names)} "
            f"feature names. A silent mismatch here would map an importance "
            f"value to the WRONG k-mer, so this is refused rather than "
            f"truncated or zip()-ed short. Make sure feature_names is "
            f"exactly `vectorizer.get_feature_names_out()` for the same "
            f"vectorizer whose transform() output the model was trained on."
        )
    return importances, feature_names


def top_features(
    importances: Union[Sequence[float], "np.ndarray"],
    feature_names: Sequence[str],
    *,
    n: int = 20,
    ascending: bool = False,
) -> pa.Table:
    """The `n` most important `(kmer_sequence, importance)` pairs, as a
    `pyarrow.Table` with columns `rank` (1-based, 1 = most important),
    `kmer`, and `importance` -- consistent with this package's convention
    of returning `pyarrow.Table` from table-shaped results (`compare_all`).

    `importances` is any 1-D array-like of one value per feature -- e.g. a
    fitted `sklearn.linear_model.LogisticRegression`'s `.coef_[0]`, a tree
    model's `.feature_importances_`, or a SHAP values array already reduced
    to one value per feature (e.g. via mean absolute SHAP across samples).
    `feature_names` is the matching list of k-mer sequence strings, same
    order and same length -- typically
    `KmerVectorizer.get_feature_names_out()`'s output. `ValueError` is
    raised if the lengths don't match (see `_validate_lengths`): silently
    mapping importance `i` to the wrong k-mer would be a worse failure mode
    than a loud error, since nothing about a shape mismatch is visible in
    the *output* -- you'd just get confidently wrong biology.

    What "most important" means here depends on `ascending`, and the two
    modes answer genuinely different questions:

    - `ascending=False` (the default): rank by **magnitude**
      (`abs(importance)`), descending. This is the right default for
      *signed* importances such as linear-model coefficients or SHAP
      values, where a large negative coefficient is just as informative
      as a large positive one -- it says a k-mer strongly pushes the
      prediction toward the *other* class, not that it doesn't matter.
      For an always-non-negative array (e.g. `feature_importances_` from a
      tree ensemble), magnitude and literal value coincide, so this is
      simply "the largest values first".
    - `ascending=True`: rank by **literal value**, ascending -- i.e. the
      smallest (most negative, for signed arrays) values first, *not* by
      magnitude. Use this when you specifically want the features driving
      the prediction toward the negative/lower class, as opposed to "the
      features that matter most regardless of direction".

    Ties are broken by original feature order (stable sort), so results
    are deterministic across calls on the same input.
    """
    importances, feature_names = _validate_lengths(importances, feature_names)
    n = min(n, len(importances))

    indices = range(len(importances))
    # The sort key is a plain lookup into an already-computed list rather
    # than a Python-level lambda, so ranking F features runs F C-level
    # `list.__getitem__` calls instead of F interpreted frames that each
    # re-index and re-`abs()`. Python's sort is stable and sees exactly the
    # same key values in the same order either way, so ties still keep
    # their original relative order and the ranking is unchanged.
    if ascending:
        # Literal value, ascending.
        order = sorted(indices, key=importances.__getitem__)
    else:
        # Magnitude, descending.
        magnitudes = [-abs(value) for value in importances]
        order = sorted(indices, key=magnitudes.__getitem__)

    top_idx = order[:n]
    return pa.table(
        {
            "rank": list(range(1, n + 1)),
            "kmer": [feature_names[i] for i in top_idx],
            "importance": [importances[i] for i in top_idx],
        }
    )


def export_top_features_fasta(
    importances: Union[Sequence[float], "np.ndarray"],
    feature_names: Sequence[str],
    path: _PathLike,
    *,
    n: int = 20,
) -> pa.Table:
    """Writes the top `n` important k-mers (by the same "most important"
    rule `top_features` uses, i.e. magnitude-ranked by default) as a FASTA
    file at `path`, ready to paste directly into NCBI BLAST or a local
    `blastn` run.

    Each record is exactly two lines: a header starting with `>`, formatted
    `>rank{rank}_importance{importance:.3f}` (1-based rank, importance to
    3 decimal places -- e.g. `>rank1_importance0.842` for the single most
    important feature, or `>rank3_importance-0.417` for a negative
    coefficient), followed by one line holding the literal k-mer sequence.
    No line-wrapping is applied -- k-mers are short (tens of bases at
    most), so standard FASTA's 60/70/80-column wrapping convention (meant
    for whole chromosomes) would only add noise here.

    Returns the `pyarrow.Table` `top_features` itself computed, in case the
    caller wants it too without a second call.
    """
    table = top_features(importances, feature_names, n=n)
    ranks = table.column("rank").to_pylist()
    kmers = table.column("kmer").to_pylist()
    values = table.column("importance").to_pylist()

    # newline="\n" pins LF line endings (FASTA, like FASTQ, is conventionally
    # an LF format) instead of letting Windows text mode translate to CRLF.
    with open(path, "w", newline="\n") as f:
        for rank, kmer, importance in zip(ranks, kmers, values):
            f.write(f">rank{rank}_importance{importance:.3f}\n{kmer}\n")

    return table


def explain_with_shap(
    model: Any,  # any fitted estimator supported by shap.Explainer
    X_transformed: Any,  # dense array or scipy.sparse matrix, e.g. KmerVectorizer.transform() output
    feature_names: Sequence[str],
    *,
    n: int = 20,
) -> pa.Table:
    """Convenience wrapper around the optional `shap` package: computes
    SHAP values for `model` over already-vectorized data `X_transformed`
    (e.g. `KmerVectorizer.transform()`'s output), reduces them to one
    value per feature via mean absolute SHAP across samples, and returns
    the same kind of top-N table `top_features` does -- so a caller does
    not have to hand-write the "SHAP values -> top-N features" glue
    themselves.

    `shap` is a genuinely optional dependency of this module (not of the
    `fastdna` package as a whole): it is imported here, inside this
    function, and nowhere else in this file, so importing
    `fastdna.interpret` itself never requires `shap` to be installed --
    only calling this one function does. A missing `shap` raises a clear,
    actionable `ImportError` naming the package and the install command,
    rather than whatever raw error would otherwise surface from deep
    inside a failed `import shap`.

    Uses `shap.Explainer`, the library's current model-agnostic entry
    point (it dispatches to a fast tree/linear explainer when `model`'s
    type supports one, and falls back to a general-purpose explainer
    otherwise), rather than pinning to one specific explainer class.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            "explain_with_shap() requires the optional 'shap' package, "
            "which is not installed. Install it with: pip install shap"
        ) from e

    import numpy as np  # shap itself always pulls numpy in, so this is free

    explainer = shap.Explainer(model, X_transformed)
    shap_values = explainer(X_transformed)

    values = shap_values.values
    # Multi-class output: shap gives (n_samples, n_features, n_classes).
    # Per-class SHAP values for a given (sample, feature) sum to zero across
    # classes (predicted probabilities and base values both sum to 1), so
    # averaging over the class axis BEFORE taking the absolute value cancels
    # real per-class contributions down to floating-point noise. Absolute
    # value must come first; only then can the class axis be averaged away
    # along with the sample axis. Binary/regression output is already
    # (n_samples, n_features), so a plain mean(axis=0) is correct for it.
    if values.ndim == 3:
        mean_abs = np.abs(values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(values).mean(axis=0)
    return top_features(mean_abs.tolist(), feature_names, n=n)
