"""fastdna.rules -- interpretable Set Covering Machine over k-mer presence.

Roadmap item A4 (`docs/ml-differentiation-roadmap.md`, bucket A, rank #4):
a rule-based binary classifier whose entire learned model is a handful of
literal DNA sequences, each tagged "present" or "absent".

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

from typing import NamedTuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

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

    def __str__(self):
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
        raise ValueError(
            f"SetCoveringClassifier.{method_name}() got an X with dtype=object, which "
            "cannot be a binary presence matrix. Pass a numeric or boolean 2-D array "
            "(or a scipy sparse matrix) of shape (n_samples, n_features)."
        )
    if arr.ndim != 2:
        raise ValueError(
            f"X must be a 2-D (n_samples, n_features) binary presence matrix, got a "
            f"{arr.ndim}-D array of shape {arr.shape}. A single sample must still be "
            "2-D -- reshape it with X.reshape(1, -1)."
        )
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(
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
            raise ValueError(
                f"X must be a binary presence matrix containing only 0/1 (or "
                f"False/True); found other values: {shown}{more}. If these are raw "
                "k-mer counts (e.g. from KmerVectorizer.transform()), binarize them "
                "first -- presence/absence is what an SCM rule means: "
                "X_binary = (X > 0).astype(np.uint8)."
            )
    return arr.astype(bool)


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

    def __init__(self, max_rules=10, rule_type="conjunction", tiebreaker="max_coverage"):
        # scikit-learn convention (the same one KmerVectorizer follows):
        # __init__ only assigns the parameters, unchanged and unvalidated,
        # so get_params()/set_params()/clone() can always rebuild an
        # equivalent unfitted estimator. All validation happens in fit().
        self.max_rules = max_rules
        self.rule_type = rule_type
        self.tiebreaker = tiebreaker

    # -- fitting ----------------------------------------------------------

    def fit(self, X, y, feature_names=None):
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
            raise ValueError(f"rule_type must be one of {_RULE_TYPES}, got {self.rule_type!r}")
        if self.tiebreaker not in _TIEBREAKERS:
            raise ValueError(f"tiebreaker must be one of {_TIEBREAKERS}, got {self.tiebreaker!r}")
        if isinstance(self.max_rules, bool) or not isinstance(self.max_rules, (int, np.integer)) or self.max_rules < 1:
            raise ValueError(
                f"max_rules must be an integer >= 1 (a model with zero rules classifies "
                f"nothing), got {self.max_rules!r}"
            )

        binary = _as_binary_matrix(X, "fit")
        n_samples, n_features = binary.shape

        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be a 1-D array of one label per sample, got shape {y.shape}")
        if len(y) != n_samples:
            raise ValueError(
                f"X and y must describe the same samples: X has {n_samples} rows but y has "
                f"{len(y)} labels."
            )

        classes = np.unique(y)
        if len(classes) == 1:
            raise ValueError(
                f"y contains a single class ({classes[0]!r}); a Set Covering Machine needs "
                "both a positive and a negative class to have anything to separate."
            )
        if len(classes) > 2:
            raise ValueError(
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
                raise ValueError(
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

        # classes_[1] is the positive class (scikit-learn's convention).
        # For a disjunction, run the identical greedy loop on the inverted
        # labels and flip the polarity of every rule it returns -- see the
        # docstring's "dual problem".
        positives = y == classes[1]
        if self.rule_type == "disjunction":
            found = self._greedy_conjunction(binary, ~positives)
            self._rules_ = [Rule(idx, names[idx], not presence) for idx, presence in found]
        else:
            found = self._greedy_conjunction(binary, positives)
            self._rules_ = [Rule(idx, names[idx], presence) for idx, presence in found]
        return self

    def _greedy_conjunction(self, binary, positives):
        """The single shared code path: returns `[(feature_index, presence)]`
        for a conjunction that keeps every sample in `positives` and removes
        as many of the rest as it can.
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
                for j in candidates:
                    cov = int(coverage[j]) if self.tiebreaker == "max_coverage" else 0
                    key = (int(removes[j]), cov, -int(j), int(presence))
                    if best is None or key > best:
                        best = key
                        best_rule = (int(j), presence)

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

    # -- prediction -------------------------------------------------------

    def _rule_mask(self, X, method_name):
        """True where the model fires, i.e. where it predicts `classes_[1]`."""
        check_is_fitted(self, "classes_")
        binary = _as_binary_matrix(X, method_name)
        if binary.shape[1] != self.n_features_in_:
            raise ValueError(
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

    def predict(self, X):
        """The predicted label per sample, taken from `classes_` so the
        caller's own label dtype (strings included) round-trips.
        """
        # _rule_mask() first, deliberately: it is what runs check_is_fitted,
        # and Python evaluates `self.classes_` before the subscript, so
        # inlining this call would surface a raw AttributeError instead of
        # scikit-learn's NotFittedError on an unfitted estimator.
        fired = self._rule_mask(X, "predict")
        return self.classes_[fired.astype(int)]

    def predict_proba(self, X):
        """The rule-consistent hard 0/1 decision, shaped like a probability:
        `(n_samples, 2)` columns ordered as `classes_`.

        **This is not a calibrated probability.** A fitted SCM is a hard
        boolean formula: it has no notion of confidence, so every row here
        is exactly `[1., 0.]` or `[0., 1.]`. Reading a 1.0 as "certain" is
        a mistake -- it means "the conjunction fired", nothing more. Any
        metric that consumes scores rather than labels (ROC-AUC, Brier,
        log-loss) will therefore be degenerate on this output, and a
        reliability diagram of it is meaningless.

        For real probabilities, wrap a fitted model in the calibration
        helper planned as `fastdna.cv.calibrate` (roadmap A1: Venn-ABERS,
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
        """
        fired = self._rule_mask(X, "predict_proba").astype(np.float64)
        return np.column_stack((1.0 - fired, fired))

    # -- interpretation ---------------------------------------------------

    @property
    def rules_(self):
        """The learned rules (see the class docstring); raises
        `NotFittedError` before `fit()`.

        A `list` copy is returned, not the internal list, so that mutating
        the result cannot silently change what `predict()` does.
        """
        check_is_fitted(self, "classes_")
        return list(self._rules_)

    def explain(self):
        """The model as one line of human-readable text, e.g.

            resistant IF present(ACGTACGTA) AND absent(TTGCATTGC)

        The leading label is the positive class (`classes_[1]`), and the
        connective is `AND` for a conjunction, `OR` for a disjunction. When
        `fit()` received `feature_names`, each literal is the actual k-mer
        sequence -- which is the whole point of the model: the explanation
        *is* the biology, not a proxy for it.
        """
        check_is_fitted(self, "classes_")
        label = self.classes_[1]
        if not self._rules_:
            if self.rule_type == "conjunction":
                return f"{label} ALWAYS (no rule was learned: an empty conjunction is vacuously true)"
            return f"{label} NEVER (no rule was learned: an empty disjunction is vacuously false)"
        connective = " AND " if self.rule_type == "conjunction" else " OR "
        return f"{label} IF " + connective.join(str(rule) for rule in self._rules_)

    def export_rules_fasta(self, path):
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

        Returns the list of `Rule`s written.
        """
        check_is_fitted(self, "classes_")
        if not self._feature_names_given_:
            raise ValueError(
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
