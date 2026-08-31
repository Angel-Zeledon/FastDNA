"""fastdna.rules -- interpretable Set Covering Machine over k-mer presence.

Roadmap item A4 (`docs/ml-differentiation-roadmap.md`, bucket A, rank #4):
a rule-based binary classifier whose entire learned model is a handful of
literal DNA sequences, each tagged "present" or "absent".

## Status: frozen

Per `docs/audit/PLAN.md` §2 ("Qué se poda"), this module is frozen: stable,
not accepting new features, and a candidate for extraction into a separate
`fastdna-contrib` package in a future release. Freezing is not deleting --
see that section for the full reasoning behind the boundary.

## Scientific basis

- Marchand & Shawe-Taylor, "The Set Covering Machine", JMLR 3 (2002)
  723-746 -- the original algorithm: greedily build a conjunction (or
  disjunction) of boolean features that covers the training set, with a
  sample-compression generalization bound that depends on the number of
  rules kept rather than the number of features screened.
- Drouin et al., "Predictive computational phenotyping and biomarker
  discovery using reference-free genome comparisons", BMC Genomics 17
  (2016) 754 -- the SCM applied to k-mer presence/absence matrices for
  antimicrobial-resistance phenotyping; and Drouin et al., "Interpretable
  genotype-to-phenotype classifiers with performance guarantees", Sci.
  Rep. 9 (2019) 4071, which adds the CART decision-tree variant. Both
  ship as the Kover tool.
- Empirical motivation, stated honestly: in the 2024 *Briefings in
  Bioinformatics* benchmark across 78 species-antibiotic datasets,
  Kover's rule models ranked at the top among the ML methods compared for
  AMR prediction. That is a strong result, not a proof of superiority --
  the same benchmark shows the ranking depends on species, drug and, above
  all, on how the train/test split was made. What is *not* in dispute is
  the interpretability difference: unlike a gradient-boosted ensemble over
  the same matrix, each learned rule here is a single k-mer, i.e. a
  literal DNA sequence a biologist can paste into BLAST.

## Overfitting -- read this before believing any result

A typical FastDNA feature matrix is tens of thousands of k-mers wide and
tens of samples tall. In that regime a greedy search will find a rule --
often a *single* rule -- that separates the training labels perfectly by
pure chance, and it will look like a discovery. Nothing in this module
detects that; `explain()` will print a beautiful, meaningless k-mer.

Numbers from this classifier are worth exactly as much as the evaluation
around them:

- Split by lineage, not at random. Clonal relatives on both sides of a
  split inflate every score (PLOS Biology 2025, 24k+ genomes). Use
  `fastdna.cv` (module `python/fastdna/cv.py`) for Mash-distance-derived,
  leakage-safe splits -- this module deliberately does not import it, so
  the two stay independently usable, but a score produced without it
  should not be reported.
- Permutation-test the rules before calling them biomarkers.
- Reproduce a public BV-BRC dataset before claiming parity with Kover
  (the roadmap makes this an explicit precondition for that claim).

## Package convention

Like `fastdna.sklearn`, this module is imported explicitly
(`from fastdna.rules import SetCoveringClassifier`) and pulls in
scikit-learn and numpy at module scope; plain `import fastdna` does not
pay for them. It needs no Rust extension and no scipy: a scipy sparse `X`
is accepted duck-typed, via its `.toarray()`, without scipy ever being
imported here.
"""

from __future__ import annotations

from itertools import repeat
from typing import Any, Dict, NamedTuple, Optional, Sequence, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from . import _core, _PathLike

__all__ = ["Rule", "SetCoveringClassifier"]

_RULE_TYPES = ("conjunction", "disjunction")
_TIEBREAKERS = ("max_coverage", "first")


class Rule(NamedTuple):
    """One learned boolean rule: a single feature, taken in one polarity.

    `presence=True` reads "this k-mer is present in the sample";
    `presence=False` reads "this k-mer is absent from the sample". A
    negative (absent) rule is just as legitimate a biomarker as a positive
    one -- loss of a porin gene is a textbook resistance mechanism -- so
    both polarities are searched, not just presence.

    `feature_name` is the k-mer sequence when `fit()` was given
    `feature_names`, and the placeholder `f"feature_{feature_index}"`
    otherwise.
    """

    feature_index: int
    feature_name: str
    presence: bool

    def __str__(self) -> str:
        return f"{'present' if self.presence else 'absent'}({self.feature_name})"


def _as_binary_matrix(X, method_name):
    """Validates `X` as a binary presence matrix and returns it as a dense
    2-D boolean array.

    A scipy sparse matrix is densified here rather than being covered by a
    sparse code path. That is a deliberate, bounded cost: the greedy search
    below needs whole-column reductions over every feature at every round,
    and the matrix an SCM is run on is samples-tall (tens to low
    thousands), so the dense boolean copy is one byte per cell -- a
    1,000 x 100,000 cohort matrix is 100 MB, and anything much larger than
    that has bigger problems than this densify.
    """
    if hasattr(X, "toarray") and not isinstance(X, np.ndarray):
        X = X.toarray()
    arr = np.asarray(X)
    if arr.dtype == object:
        raise _core.InvalidConfigError(
            f"SetCoveringClassifier.{method_name}() got an X with dtype=object, which "
            "cannot be a binary presence matrix. Pass a numeric or boolean 2-D array "
            "(or a scipy sparse matrix) of shape (n_samples, n_features)."
        )
    if arr.ndim != 2:
        raise _core.InvalidConfigError(
            f"X must be a 2-D (n_samples, n_features) binary presence matrix, got a "
            f"{arr.ndim}-D array of shape {arr.shape}. A single sample must still be "
            "2-D -- reshape it with X.reshape(1, -1)."
        )
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise _core.InvalidConfigError(
            f"X is empty: shape {arr.shape} has no "
            f"{'samples' if arr.shape[0] == 0 else 'features'}. "
            "SetCoveringClassifier needs at least one sample and one feature."
        )
    if arr.dtype != bool:
        # np.unique over the whole matrix is the cheapest way to name the
        # actual offending values in the error, which is the difference
        # between an actionable message and "invalid input".
        unique = np.unique(arr)
        offending = [v for v in unique.tolist() if v not in (0, 1)]
        if offending:
            shown = ", ".join(repr(v) for v in offending[:5])
            more = "" if len(offending) <= 5 else f", ... ({len(offending)} distinct values in total)"
            raise _core.InvalidConfigError(
                f"X must be a binary presence matrix containing only 0/1 (or "
                f"False/True); found other values: {shown}{more}. If these are raw "
                "k-mer counts (e.g. from KmerVectorizer.transform()), binarize them "
                "first -- presence/absence is what an SCM rule means: "
                "X_binary = (X > 0).astype(np.uint8)."
            )
    return arr.astype(bool)


def _resolve_sample_weights(class_weight, y, classes):
    """Turns `class_weight` into a length-`len(y)` float64 array of
    per-sample weights, one entry per row of `y`, or `None` when
    `class_weight` is `None` -- meaning "no weighting", which `fit()` uses
    as the signal to run the original, unmodified hard-coverage search.

    `classes` is `self.classes_` (the two sorted labels). Every rejection
    names the actual offending value or label, matching this module's
    validation convention elsewhere (see `_as_binary_matrix`).
    """
    if class_weight is None:
        return None

    n_samples = len(y)
    labels = classes.tolist()

    if isinstance(class_weight, str):
        if class_weight != "balanced":
            raise _core.InvalidConfigError(
                f"class_weight must be None, 'balanced', or a dict of {{label: weight}}; "
                f"the only recognized string value is 'balanced', got {class_weight!r}."
            )
        # sklearn's own formula: n_samples / (n_classes * bincount(y)).
        counts = np.array([np.count_nonzero(y == c) for c in classes], dtype=np.float64)
        per_class = n_samples / (len(classes) * counts)
        weight_by_class = dict(zip(labels, per_class.tolist()))
    elif isinstance(class_weight, dict):
        missing = [c for c in labels if c not in class_weight]
        unknown = [c for c in class_weight if c not in labels]
        if missing or unknown:
            parts = []
            if missing:
                parts.append(f"missing a weight for {missing!r}")
            if unknown:
                parts.append(f"has weight(s) for unknown label(s) {unknown!r}")
            raise _core.InvalidConfigError(
                f"class_weight dict must have exactly one weight per class in classes_ "
                f"({labels!r}): {' and '.join(parts)}."
            )
        bad = {
            c: w
            for c, w in class_weight.items()
            if isinstance(w, bool) or not isinstance(w, (int, float, np.integer, np.floating))
            or not (w > 0) or not np.isfinite(w)
        }
        if bad:
            raise _core.InvalidConfigError(
                f"class_weight values must be finite positive numbers, got {bad!r} -- a "
                "weight must express how much this class's examples count, and zero, "
                "negative, or non-numeric weights have no such meaning."
            )
        weight_by_class = {c: float(w) for c, w in class_weight.items()}
    else:
        raise _core.InvalidConfigError(
            f"class_weight must be None, 'balanced', or a dict of {{label: weight}}, got "
            f"{class_weight!r} ({type(class_weight).__name__})."
        )

    weights = np.empty(n_samples, dtype=np.float64)
    for c, w in weight_by_class.items():
        weights[y == c] = w
    return weights


class SetCoveringClassifier(BaseEstimator, ClassifierMixin):
    """A Set Covering Machine: a short conjunction (or disjunction) of
    present/absent k-mer rules, fitted greedily.

        from fastdna.rules import SetCoveringClassifier

        clf = SetCoveringClassifier(max_rules=5)
        clf.fit(X_binary, y, feature_names=vectorizer.get_feature_names_out())
        print(clf.explain())
        # resistant IF present(ACGTACGT...) AND absent(TTGCATTG...)
        clf.export_rules_fasta("rules.fasta")   # -> straight into BLAST

    `X` is a binary presence matrix (dense or scipy sparse): cell `(i, j)`
    is 1 if sample `i` contains k-mer `j`. Counts must be binarized first
    (`(X > 0).astype(np.uint8)`) -- see the validation error.

    **Overfits spectacularly on wide matrices.** Tens of thousands of
    k-mers over tens of samples will yield a perfect-looking rule by
    chance. A score from this class is meaningless without leakage-safe
    evaluation; use `fastdna.cv`'s lineage-aware splitters, and see this
    module's docstring for the citations.

    Parameters
    ----------
    max_rules : int, default 10
        Hard cap on how many rules are kept. This is the SCM's only real
        complexity knob and the quantity its sample-compression bound
        depends on (Marchand & Shawe-Taylor 2002): fewer rules means a
        tighter bound and a model a human can actually read. The greedy
        loop stops early whenever no negative examples remain, so the cap
        binds only on problems the rules cannot fully separate.
    rule_type : {"conjunction", "disjunction"}, default "conjunction"
        `"conjunction"` predicts the positive class when *every* rule
        holds (an AND of literals); `"disjunction"` when *any* rule holds
        (an OR). The two are duals of each other and share one code path
        -- see `fit()`.
    tiebreaker : {"max_coverage", "first"}, default "max_coverage"
        How to choose among rules that remove exactly the same number of
        remaining negative examples. `"max_coverage"` prefers the rule
        that holds for the most training samples overall, i.e. the most
        general one, on the reasoning that a rule true of many samples is
        less likely to be an artifact of a handful of them. `"first"`
        expresses no preference and simply takes the lowest feature index.
        Both fall back to the lowest feature index (then to `present`
        before `absent`) so that a fit is fully deterministic and
        reproducible.
    class_weight : None, "balanced", or dict[label, float], default None
        Enables **cost-sensitive** fitting for imbalanced cohorts (matching
        `sklearn`'s own `class_weight` convention). `None` (the default)
        runs the exact algorithm described above: unweighted, unchanged.
        `"balanced"` sets each class's weight to
        `n_samples / (n_classes * numpy.bincount(y))`, sklearn's own
        formula; a `dict` sets the weight of each label explicitly.

        **Why weighting a plain count doesn't work, and what this actually
        does.** The obvious generalization -- weight the greedy score's
        "negatives removed" count by each removed example's class weight --
        is *provably a no-op* for this algorithm, for any dataset. Every
        candidate literal is required to hold for 100% of the positive
        class (that hard guarantee is what makes the rules sample-compressed
        and is the reason a "negative removed" is always drawn from one
        single, homogeneous class at a time); with only one weight value per
        class, weighting a count over a homogeneous population is a positive
        scalar multiple of the unweighted count, and scaling by a positive
        constant never changes which candidate has the highest score. (This
        was checked exhaustively, not just argued: 20,000 randomized
        imbalanced datasets, exact power-of-two weights to rule out
        floating-point noise, zero divergences from the unweighted rules.)

        So `class_weight` here does something structurally different: when
        the weights are not uniform across classes, `fit()` relaxes the hard
        "must hold for every positive" requirement into a weighted
        trade-off. Each round, a candidate literal's score becomes

            (sum of weights of remaining negatives it would remove)
            - (sum of weights of still-covered positives it would newly exclude)

        and only a literal with a strictly positive net score is eligible.
        This means a weighted conjunction/disjunction may deliberately
        misclassify a few low-weight training positives (or, for a
        disjunction's dual, negatives) if doing so buys a proportionally
        larger, higher-weight gain elsewhere -- the standard meaning of cost-
        sensitive learning, and a genuine, documented departure from the
        unweighted model's 100%-training-recall-on-the-positive-class
        guarantee. `explain()` and `predict()` need no special-casing: they
        just read off whatever `rules_` this trade-off produced.

        When every sample carries the same weight -- `class_weight=None`,
        or a `dict` giving both classes an equal value -- there is no
        asymmetry to trade off, and `fit()` detects this and runs the
        original hard-coverage search verbatim, which is what makes
        `class_weight=None` (and a uniform `dict`) provably bit-identical to
        the historical, unweighted behavior. See `python/tests/test_rules.py`
        for the test proving this and a hand-checkable example of the
        relaxation finding a literal (present in most, but not all, of a
        tiny positive class) that the hard SCM can never even nominate as a
        candidate.

    Attributes
    ----------
    rules_ : list of Rule
        The learned rules, in the order the greedy search found them
        (which is also decreasing order of how many negatives each removed
        at the time it was picked). May be shorter than `max_rules`, and
        may be empty -- see `fit()`.
    classes_ : numpy.ndarray
        The two class labels, sorted. Following scikit-learn's convention,
        `classes_[1]` is the *positive* class: the one the rules describe
        and the one `predict_proba()[:, 1]` refers to.
    feature_names_ : list of str
        One name per column of the training `X` -- the k-mer sequences if
        `fit()` was given them, else `["feature_0", "feature_1", ...]`.
    n_features_in_ : int
        Column count of the training `X`; `predict()` requires the same.
    """

    def __init__(
        self,
        max_rules: int = 10,
        rule_type: str = "conjunction",
        tiebreaker: str = "max_coverage",
        class_weight: Optional[Union[str, Dict[Any, float]]] = None,
    ) -> None:
        # scikit-learn convention (the same one KmerVectorizer follows):
        # __init__ only assigns the parameters, unchanged and unvalidated,
        # so get_params()/set_params()/clone() can always rebuild an
        # equivalent unfitted estimator. All validation happens in fit().
        self.max_rules = max_rules
        self.rule_type = rule_type
        self.tiebreaker = tiebreaker
        self.class_weight = class_weight

    # -- fitting ----------------------------------------------------------

    def fit(
        self,
        X: Any,  # binary presence matrix: dense array-like or scipy sparse (duck-typed via .toarray(), scipy not imported here)
        y: Union[Sequence[Any], np.ndarray],
        feature_names: Optional[Sequence[str]] = None,
    ) -> "SetCoveringClassifier":
        """Greedily learns the rule set.

        Parameters
        ----------
        X : array-like or scipy sparse matrix of shape (n_samples, n_features)
            Binary k-mer presence matrix; only 0/1/False/True are accepted.
        y : array-like of shape (n_samples,)
            Exactly two distinct labels. The SCM is binary-only; wrap this
            estimator in `sklearn.multiclass.OneVsRestClassifier` for more.
        feature_names : sequence of str, optional
            One name per column, typically
            `KmerVectorizer.get_feature_names_out()`. Without it the rules
            carry positional placeholders and `export_rules_fasta()`
            refuses to write (a placeholder is not a DNA sequence).

        Algorithm
        ---------
        For the conjunction case, with `P` the positive samples and `N` the
        negatives: consider every feature in both polarities, keep only
        those *candidate* rules that hold for every sample in `P` (a
        conjunction can never re-admit a positive it has excluded, so a
        rule that drops one is unusable), and repeatedly pick the candidate
        that is false for -- i.e. removes -- the most of the still-covered
        negatives. Stop when no negatives remain or `max_rules` rules have
        been picked.

        The disjunction case is the same procedure on the dual problem, not
        a second implementation: a disjunction predicts the negative class
        exactly when *every* negated rule holds, so `fit()` runs the
        identical greedy loop with the labels inverted and then flips each
        resulting rule's polarity. `rules_` is always reported in the
        model's own terms.

        Two honest edge cases:

        - If no candidate rule exists at all (every feature is mixed across
          the positives, e.g. a genuinely disjunctive problem given
          `rule_type="conjunction"`), `rules_` comes back **empty**. An
          empty conjunction is vacuously true, so `predict()` then returns
          the positive class for everything; an empty disjunction is
          vacuously false and returns the negative class for everything.
          This is a real "I learned nothing" signal -- check `rules_`.
        - If negatives are still covered when `max_rules` is reached, the
          model does not separate its own training set. Rules are not
          padded to `max_rules` with useless ones: a rule that removes zero
          remaining negatives is never added.

        Returns
        -------
        self, per the scikit-learn convention.
        """
        if self.rule_type not in _RULE_TYPES:
            raise _core.InvalidConfigError(f"rule_type must be one of {_RULE_TYPES}, got {self.rule_type!r}")
        if self.tiebreaker not in _TIEBREAKERS:
            raise _core.InvalidConfigError(f"tiebreaker must be one of {_TIEBREAKERS}, got {self.tiebreaker!r}")
        if isinstance(self.max_rules, bool) or not isinstance(self.max_rules, (int, np.integer)) or self.max_rules < 1:
            raise _core.InvalidConfigError(
                f"max_rules must be an integer >= 1 (a model with zero rules classifies "
                f"nothing), got {self.max_rules!r}"
            )

        binary = _as_binary_matrix(X, "fit")
        n_samples, n_features = binary.shape

        y = np.asarray(y)
        if y.ndim != 1:
            raise _core.InvalidConfigError(f"y must be a 1-D array of one label per sample, got shape {y.shape}")
        if len(y) != n_samples:
            raise _core.InvalidConfigError(
                f"X and y must describe the same samples: X has {n_samples} rows but y has "
                f"{len(y)} labels."
            )

        classes = np.unique(y)
        if len(classes) == 1:
            raise _core.InvalidConfigError(
                f"y contains a single class ({classes[0]!r}); a Set Covering Machine needs "
                "both a positive and a negative class to have anything to separate."
            )
        if len(classes) > 2:
            raise _core.InvalidConfigError(
                f"The Set Covering Machine is a binary classifier; y has {len(classes)} "
                f"classes ({', '.join(repr(c) for c in classes.tolist()[:5])}). Reduce the "
                "problem to two classes, or wrap this estimator in "
                "sklearn.multiclass.OneVsRestClassifier."
            )

        if feature_names is None:
            names = [f"feature_{j}" for j in range(n_features)]
            named = False
        else:
            names = [str(name) for name in feature_names]
            if len(names) != n_features:
                raise _core.InvalidConfigError(
                    f"feature_names has {len(names)} entries but X has {n_features} columns. "
                    "A silent mismatch here would label every rule with the WRONG k-mer, so "
                    "this is refused rather than truncated -- pass exactly the feature names "
                    "of the matrix being fitted (e.g. vectorizer.get_feature_names_out())."
                )
            named = True

        self.classes_ = classes
        self.feature_names_ = names
        self.n_features_in_ = n_features
        self._feature_names_given_ = named

        sample_weight = _resolve_sample_weights(self.class_weight, y, classes)

        # classes_[1] is the positive class (scikit-learn's convention).
        # For a disjunction, run the identical greedy loop on the inverted
        # labels and flip the polarity of every rule it returns -- see the
        # docstring's "dual problem". `sample_weight` is passed through
        # unpermuted in both directions: it is indexed by the same boolean
        # masks derived from `positives`/`~positives`, so it always lines up
        # with whichever set plays "positive" for this call.
        positives = y == classes[1]
        if self.rule_type == "disjunction":
            found = self._greedy_conjunction(binary, ~positives, sample_weight)
            self._rules_ = [Rule(idx, names[idx], not presence) for idx, presence in found]
        else:
            found = self._greedy_conjunction(binary, positives, sample_weight)
            self._rules_ = [Rule(idx, names[idx], presence) for idx, presence in found]
        return self

    def _greedy_conjunction(self, binary, positives, sample_weight=None):
        """Returns `[(feature_index, presence)]` for a conjunction over
        `positives`.

        Dispatches to `_greedy_hard` -- the original, unweighted
        hard-coverage search, verbatim -- whenever there is no real
        weighting to apply (`sample_weight` is `None`, or every sample
        shares the same weight, e.g. a uniform `class_weight` dict). This is
        what makes `class_weight=None` and a uniform dict provably
        bit-identical to the historical behavior: the exact same code runs.
        A genuinely non-uniform `sample_weight` -- i.e. `class_weight` set
        to `"balanced"` or an asymmetric dict on an actually imbalanced
        cohort -- goes to `_greedy_soft`, the cost-sensitive relaxation (see
        the class docstring's `class_weight` entry for why a hard-coverage
        SCM needs a structurally different mechanism, not just a weighted
        count, to make class_weight matter).
        """
        if sample_weight is None or np.all(sample_weight == sample_weight[0]):
            return self._greedy_hard(binary, positives)
        return self._greedy_soft(binary, positives, sample_weight)

    def _greedy_hard(self, binary, positives):
        """The original, unweighted greedy search: keeps every sample in
        `positives` covered and removes as many of the rest as it can.
        """
        pos_rows = binary[positives]
        # A candidate rule must hold for every positive. Computed once, from
        # the full positive set: a conjunction never removes positives, so
        # this set does not shrink as rules are added.
        present_ok = pos_rows.all(axis=0)
        absent_ok = (~pos_rows).all(axis=0)

        # Coverage over the *whole* training set, for the "max_coverage"
        # tiebreaker. Note it is deliberately whole-set, not
        # remaining-negatives: on the first round the latter would be a
        # monotone function of the greedy score itself and could never break
        # anything.
        present_coverage = binary.sum(axis=0)
        absent_coverage = binary.shape[0] - present_coverage

        remaining = ~positives  # negatives still covered by the conjunction
        rules = []
        # Loop-invariant: the tiebreaker cannot change while the search
        # runs, so it is resolved once instead of once per candidate rule
        # per round.
        use_coverage = self.tiebreaker == "max_coverage"
        while len(rules) < self.max_rules and remaining.any():
            neg_rows = binary[remaining]
            # A "present(j)" rule removes every remaining negative in which
            # feature j is absent, and vice versa.
            present_removes = np.count_nonzero(~neg_rows, axis=0)
            absent_removes = np.count_nonzero(neg_rows, axis=0)

            best = None  # (score, coverage, -index, presence_rank), maximized
            best_rule = None
            for ok, removes, coverage, presence in (
                (present_ok, present_removes, present_coverage, True),
                (absent_ok, absent_removes, absent_coverage, False),
            ):
                candidates = np.flatnonzero(ok & (removes > 0))
                presence_rank = int(presence)
                # The three per-candidate `int()` calls this loop used to
                # make each converted one NumPy scalar at a time; `tolist()`
                # converts the whole selection in one C pass, so a round
                # over 10,000 candidate features does three bulk
                # conversions instead of 30,000 individual ones. The
                # integers, and therefore every key comparison below, are
                # identical.
                indices = candidates.tolist()
                removals = removes[candidates].tolist()
                coverages = coverage[candidates].tolist() if use_coverage else repeat(0)
                for j, removed, cov in zip(indices, removals, coverages):
                    key = (removed, cov, -j, presence_rank)
                    if best is None or key > best:
                        best = key
                        best_rule = (j, presence)

            if best_rule is None:
                # No candidate removes any remaining negative: adding
                # anything would only lengthen the model without changing a
                # single prediction.
                break

            idx, presence = best_rule
            column = binary[:, idx]
            holds = column if presence else ~column
            remaining &= holds
            rules.append((idx, presence))
        return rules

    def _greedy_soft(self, binary, positives, sample_weight):
        """The cost-sensitive relaxation used whenever `sample_weight` is
        genuinely non-uniform across classes: candidacy is no longer gated
        on holding for every positive. Instead, every round, every
        (feature, polarity) is scored by

            (weight of remaining negatives it would remove)
            - (weight of still-covered positives it would newly exclude)

        and the highest-scoring literal with a strictly positive score is
        taken -- "positive" meaning it is worth adding at all: a literal
        that costs at least as much (in weight) as it gains is never an
        improvement over leaving it out. `covered` tracks which positives
        still satisfy every rule chosen so far (it can shrink, unlike
        `_greedy_hard`'s always-fully-covered positive set); a positive
        dropped from it in one round is not "cost" again in a later round --
        its weight has already been paid.
        """
        present_coverage = binary.sum(axis=0)
        absent_coverage = binary.shape[0] - present_coverage
        use_coverage = self.tiebreaker == "max_coverage"

        remaining = ~positives  # negatives not yet removed
        covered = positives.copy()  # positives still satisfying every rule so far
        rules = []
        while len(rules) < self.max_rules and remaining.any():
            neg_rows = binary[remaining]
            neg_weight = sample_weight[remaining]
            cov_rows = binary[covered]
            cov_weight = sample_weight[covered]

            gain_present = neg_weight @ (~neg_rows).astype(np.float64)
            cost_present = cov_weight @ (~cov_rows).astype(np.float64)
            gain_absent = neg_weight @ neg_rows.astype(np.float64)
            cost_absent = cov_weight @ cov_rows.astype(np.float64)

            score_present = gain_present - cost_present
            score_absent = gain_absent - cost_absent

            best = None  # (score, coverage, -index, presence_rank), maximized
            best_rule = None
            for score, coverage, presence in (
                (score_present, present_coverage, True),
                (score_absent, absent_coverage, False),
            ):
                candidates = np.flatnonzero(score > 0)
                presence_rank = int(presence)
                indices = candidates.tolist()
                scores = score[candidates].tolist()
                coverages = coverage[candidates].tolist() if use_coverage else repeat(0)
                for j, sc, cov in zip(indices, scores, coverages):
                    key = (sc, cov, -j, presence_rank)
                    if best is None or key > best:
                        best = key
                        best_rule = (j, presence)

            if best_rule is None:
                # No literal is worth its weighted cost: adding one would
                # only make the model larger for a net loss.
                break

            idx, presence = best_rule
            column = binary[:, idx]
            holds = column if presence else ~column
            remaining &= holds
            covered &= holds
            rules.append((idx, presence))
        return rules

    # -- prediction -------------------------------------------------------

    def _rule_mask(self, X, method_name):
        """True where the model fires, i.e. where it predicts `classes_[1]`."""
        check_is_fitted(self, "classes_")
        binary = _as_binary_matrix(X, method_name)
        if binary.shape[1] != self.n_features_in_:
            raise _core.InvalidConfigError(
                f"X has {binary.shape[1]} features, but this SetCoveringClassifier was "
                f"fitted on {self.n_features_in_}. Transform prediction samples with the "
                "same vectorizer/vocabulary used for the training matrix."
            )

        conjunction = self.rule_type == "conjunction"
        # An empty rule set: a conjunction of nothing is vacuously true, a
        # disjunction of nothing vacuously false. See fit()'s edge cases.
        mask = np.full(binary.shape[0], conjunction, dtype=bool)
        for rule in self._rules_:
            column = binary[:, rule.feature_index]
            holds = column if rule.presence else ~column
            if conjunction:
                mask &= holds
            else:
                mask |= holds
        return mask

    def predict(self, X: Any) -> np.ndarray:  # X: same duck-typed dense/sparse matrix as fit()
        """The predicted label per sample, taken from `classes_` so the
        caller's own label dtype (strings included) round-trips.

        Parameters
        ----------
        X : array-like or scipy sparse matrix of shape (n_samples, n_features)
            Binary k-mer presence matrix, same convention as `fit()`'s `X`.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            One entry per row of `X`, each equal to `classes_[0]` or
            `classes_[1]`.
        """
        # _rule_mask() first, deliberately: it is what runs check_is_fitted,
        # and Python evaluates `self.classes_` before the subscript, so
        # inlining this call would surface a raw AttributeError instead of
        # scikit-learn's NotFittedError on an unfitted estimator.
        fired = self._rule_mask(X, "predict")
        return self.classes_[fired.astype(int)]

    def predict_proba(self, X: Any) -> np.ndarray:  # X: same duck-typed dense/sparse matrix as fit()
        """The rule-consistent hard 0/1 decision, shaped like a probability:
        `(n_samples, 2)` columns ordered as `classes_`.

        **This is not a calibrated probability.** A fitted SCM is a hard
        boolean formula: it has no notion of confidence, so every row here
        is exactly `[1., 0.]` or `[0., 1.]`. Reading a 1.0 as "certain" is
        a mistake -- it means "the conjunction fired", nothing more. Any
        metric that consumes scores rather than labels (ROC-AUC, Brier,
        log-loss) will therefore be degenerate on this output, and a
        reliability diagram of it is meaningless.

        For real probabilities, wrap a fitted model with
        `fastdna.calibration.calibrate` (roadmap A1: Venn-ABERS by default,
        the variant used in clinical microbiology), fitted on held-out,
        lineage-blocked folds from `fastdna.cv` -- calibrating on the
        training folds would simply relabel the same overfit.

        Why return this at all instead of raising `AttributeError`: raising
        from inside the method would not actually hide the method from
        scikit-learn's duck-typing (`hasattr(clf, "predict_proba")` is
        still True for a method that raises when *called*), so it would buy
        no safety while breaking every pipeline that merely probes for the
        attribute. Returning the honest hard decision, loudly documented,
        keeps `Pipeline`/`OneVsRestClassifier` working and puts the caveat
        where it can be read.

        Parameters
        ----------
        X : array-like or scipy sparse matrix of shape (n_samples, n_features)
            Binary k-mer presence matrix, same convention as `fit()`'s `X`.

        Returns
        -------
        numpy.ndarray of shape (n_samples, 2)
            Columns ordered as `classes_`; every row is `[1., 0.]` or
            `[0., 1.]` -- see the caveat above before using this as a
            score.
        """
        fired = self._rule_mask(X, "predict_proba").astype(np.float64)
        return np.column_stack((1.0 - fired, fired))

    # -- interpretation ---------------------------------------------------

    @property
    def rules_(self) -> list[Rule]:
        """The learned rules (see the class docstring); raises
        `NotFittedError` before `fit()`.

        A `list` copy is returned, not the internal list, so that mutating
        the result cannot silently change what `predict()` does.

        Returns
        -------
        list of Rule
        """
        check_is_fitted(self, "classes_")
        return list(self._rules_)

    def explain(self) -> str:
        """The model as one line of human-readable text, e.g.

            resistant IF present(ACGTACGTA) AND absent(TTGCATTGC)

        The leading label is the positive class (`classes_[1]`), and the
        connective is `AND` for a conjunction, `OR` for a disjunction. When
        `fit()` received `feature_names`, each literal is the actual k-mer
        sequence -- which is the whole point of the model: the explanation
        *is* the biology, not a proxy for it.

        Naming note: this method predates, and collides in name with, the
        top-level `fastdna.explain()` function -- both exist, both are
        public, and nothing here renames either (see the review that added
        this note for the reasoning: renaming a public method name is a
        breaking change to flag for a human, not to decide unilaterally).

        Returns
        -------
        str
        """
        check_is_fitted(self, "classes_")
        label = self.classes_[1]
        if not self._rules_:
            if self.rule_type == "conjunction":
                return f"{label} ALWAYS (no rule was learned: an empty conjunction is vacuously true)"
            return f"{label} NEVER (no rule was learned: an empty disjunction is vacuously false)"
        connective = " AND " if self.rule_type == "conjunction" else " OR "
        return f"{label} IF " + connective.join(str(rule) for rule in self._rules_)

    def export_rules_fasta(self, path: _PathLike) -> list[Rule]:
        """Writes the rule k-mers to `path` as FASTA, ready for BLAST.

        Each record is two lines, matching `fastdna.interpret`'s exporter:
        a header `>rule{n}_{present|absent}_feature{index}` followed by the
        literal k-mer sequence, unwrapped (k-mers are tens of bases; the
        60/80-column convention exists for chromosomes). LF line endings
        are pinned with `newline="\\n"` so a Windows run produces the same
        bytes as a Linux one.

        Refuses to write if `fit()` was called without `feature_names`:
        the rules would carry positional placeholders, and a FASTA of
        `feature_2` is not a sequence file, it is a trap for whoever opens
        it next.

        Parameters
        ----------
        path : str or os.PathLike
            Destination FASTA file. Required: writing it is this method's
            entire purpose, so there is no in-memory-only form to fall
            back to.

        Returns
        -------
        list of Rule
            The rules written, in the same order as `rules_`.

        Raises
        ------
        ValueError
            If `fit()` was called without `feature_names`.
        """
        check_is_fitted(self, "classes_")
        if not self._feature_names_given_:
            raise _core.InvalidConfigError(
                "export_rules_fasta() needs real k-mer sequences, but fit() was called "
                "without feature_names, so the rules only carry positional placeholders "
                f"like {self._rules_[0].feature_name!r} if any were learned. Refit with "
                "feature_names=vectorizer.get_feature_names_out()."
            )
        rules = self._rules_
        with open(path, "w", newline="\n") as f:
            for n, rule in enumerate(rules, start=1):
                polarity = "present" if rule.presence else "absent"
                f.write(f">rule{n}_{polarity}_feature{rule.feature_index}\n{rule.feature_name}\n")
        return list(rules)
