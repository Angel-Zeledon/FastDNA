"""Uncertainty-sampling utilities over generic ranked classification output.

Turns "a batch of classification results with confidence scores" into "a
prioritized queue of what a human should look at first" -- the standard
active-learning workflow: when a k-mer-based classifier doesn't confidently
match a query against any known class/reference, that low-confidence result
is exactly the signal that a query needs expert review and, once labeled, is
a strong candidate for the reference set/training data.

Deliberately generic: this module does not import or assume the existence
of `fastdna.taxonomy` or `fastdna.sklearn`. It accepts two plain,
documented input shapes instead, so it works with output from either (or
anything else shaped the same way):

- **Ranked-tuples shape**: for one query, a list of `(label, score)` tuples
  ordered best-first, e.g. ``[("E. coli", 0.82), ("Shigella", 0.79), ...]``.
  This is the shape ``fastdna.taxonomy.classify()`` is documented to
  return (see docs/ml-genomics-roadmap.md item 2). Scores need not sum to
  1 -- they may be raw containment/Jaccard values.
- **Probability-vector shape**: for one query, a 1D array/list of
  per-class probabilities plus the matching list of class labels, e.g.
  straight out of a scikit-learn classifier's ``predict_proba()`` for one
  sample: ``(probs, labels)`` with ``len(probs) == len(labels)``.

`uncertainty_score` takes either shape directly (it distinguishes them
structurally, see its docstring) and computes one of two standard
active-learning uncertainty measures (Settles, "Active Learning
Literature Survey", 2009):

- ``"margin"``: 1 - (top-1 score - top-2 score), so a small gap between the
  best and second-best candidate (the classifier is nearly torn) yields a
  value close to 1 (highly uncertain), and a wide gap yields a value close
  to 0 (confident).
- ``"entropy"``: Shannon entropy (base 2, so a uniform distribution over
  ``k`` classes tops out at ``log2(k)``) of the full score distribution
  over *all* candidates, not just the top two, after normalizing it to a
  probability-like distribution (see `_normalize` below).

Convention used consistently by both methods: **larger return value means
more uncertain.**
"""

import math

__all__ = [
    "uncertainty_score",
    "prioritize_for_review",
    "suggest_reference_additions",
]

_VALID_METHODS = ("margin", "entropy")


def _is_valid_candidate_pair(entry):
    """True if `entry` looks like one `(label, score)` ranked-result
    candidate: a length-2 sequence whose second element is numeric."""
    return (
        isinstance(entry, (list, tuple))
        and len(entry) == 2
        and isinstance(entry[1], (int, float))
        and not isinstance(entry[1], bool)
    )


def _is_ranked_tuples(result):
    """True if `result` looks like the ranked-tuples shape: a sequence
    where *every* element is a `(label, score)` candidate pair (see
    `_is_valid_candidate_pair`), rather than the `(probs, labels)`
    probability-vector shape.

    This correctly distinguishes the two shapes for any number of ranked
    candidates, and for probability vectors of any length != 2 (the
    top-level `(probs, labels)` pair itself is never a valid `(label,
    score)` candidate unless it happens to have exactly 2 classes *and*
    numeric class labels -- documented known ambiguity below). A ranked
    list with string labels is unambiguous regardless of its length,
    including the length-2 case, because `labels` in the probability-vector
    shape is checked too: it only "looks like" a candidate pair if its own
    second element is itself numeric, which fails for ordinary (string)
    class labels.

    Known limitation: a 2-class probability vector paired with *numeric*
    class labels (e.g. `([0.6, 0.4], [0, 1])`) is structurally
    indistinguishable from a 2-entry ranked-tuples list and will be
    misdetected as ranked-tuples. Use string (or otherwise non-numeric)
    class labels, or pre-convert to the ranked-tuples shape yourself, to
    avoid this corner case.
    """
    if not isinstance(result, (list, tuple)):
        return False
    return all(_is_valid_candidate_pair(entry) for entry in result)


def _scores_from_ranked_tuples(ranked_results):
    if len(ranked_results) == 0:
        raise ValueError(
            "uncertainty_score: ranked_results is empty -- need at least "
            "one (label, score) candidate to compute an uncertainty score."
        )
    scores = []
    for i, entry in enumerate(ranked_results):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(
                f"uncertainty_score: entry {i!r} of ranked_results is not "
                f"a (label, score) pair: {entry!r}"
            )
        _, score = entry
        scores.append(float(score))
    return scores


def _scores_from_probability_vector(probs, labels):
    if len(probs) == 0 or len(labels) == 0:
        raise ValueError(
            "uncertainty_score: probability vector and label list must be "
            "non-empty."
        )
    if len(probs) != len(labels):
        raise ValueError(
            f"uncertainty_score: probs has {len(probs)} entries but labels "
            f"has {len(labels)} -- they must be the same length and "
            "correspond position-by-position."
        )
    return [float(p) for p in probs]


def _extract_scores(ranked_results_or_probs):
    """Returns a flat list of scores for one query, from either supported
    input shape. See module docstring / `uncertainty_score` for the shapes.
    """
    if _is_ranked_tuples(ranked_results_or_probs):
        return _scores_from_ranked_tuples(ranked_results_or_probs)

    if len(ranked_results_or_probs) != 2:
        raise ValueError(
            "uncertainty_score: probability-vector input must be a "
            "(probs, labels) pair of equal-length sequences."
        )
    probs, labels = ranked_results_or_probs
    return _scores_from_probability_vector(probs, labels)


def _normalize(scores):
    """Normalizes a list of scores into a probability-like distribution
    (non-negative, sums to 1), for use by the entropy method.

    Ranked-tuple scores may be raw containment/Jaccard values that are
    already non-negative but don't sum to 1 (e.g. `[0.4, 0.3, 0.1]` from
    independent containment checks against different references). In that
    common case, simple sum-normalization (dividing by the total) is used:
    it preserves the *relative* weighting of the original scores exactly,
    which is what entropy should measure, and is the natural choice when
    the input is already a set of non-negative "how well does this match"
    scores rather than log-odds or otherwise signed values.

    If any score is negative (e.g. raw classifier decision-function
    margins, or anything else that isn't already a similarity-like
    non-negative score), sum-normalization does not produce a valid
    probability distribution, so a softmax is used instead -- it maps any
    real-valued scores onto a valid, non-negative, sum-to-1 distribution
    while preserving relative order.
    """
    if any(s < 0 for s in scores):
        max_score = max(scores)
        exps = [math.exp(s - max_score) for s in scores]
        total = sum(exps)
        return [e / total for e in exps]

    total = sum(scores)
    if total == 0:
        # All-zero scores: treat as uniform (maximum uncertainty), the only
        # non-arbitrary choice when there's no signal to weight by at all.
        n = len(scores)
        return [1.0 / n] * n
    return [s / total for s in scores]


def _margin_uncertainty(scores):
    ordered = sorted(scores, reverse=True)
    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    margin = top1 - top2
    return 1.0 - margin


def _entropy_uncertainty(scores):
    probs = _normalize(scores)
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return entropy


def uncertainty_score(ranked_results_or_probs, *, method="margin"):
    """Computes a single uncertainty score for ONE query's classification
    result. Larger return value = more uncertain (consistent convention
    across both methods).

    `ranked_results_or_probs` accepts either of two shapes, distinguished
    structurally:

    - Ranked-tuples shape: a list of `(label, score)` pairs, best-first,
      e.g. ``[("E. coli", 0.82), ("Shigella", 0.79), ("Salmonella", 0.10)]``.
      This is the shape ``fastdna.taxonomy.classify()`` is documented to
      return. Scores need not sum to 1.
    - Probability-vector shape: a `(probs, labels)` pair of equal-length
      sequences, e.g. ``([0.82, 0.79, 0.10], ["E. coli", "Shigella",
      "Salmonella"])`` -- the shape you'd build from a scikit-learn
      classifier's ``predict_proba()`` output for one sample plus
      ``classifier.classes_``.

    `method`:

    - ``"margin"`` (default): ``1 - (top1_score - top2_score)``. A small
      gap between the best and second-best candidate means the classifier
      is nearly torn between them -- high uncertainty, so this returns a
      value close to 1. A wide gap (confident top pick) returns a value
      close to 0. Only the top two scores matter, by construction (Settles
      2009, "smallest margin" query strategy).
    - ``"entropy"``: Shannon entropy (base 2) of the full score
      distribution over *all* provided candidates, after normalizing to a
      probability-like distribution (see `_normalize`). A sharply-peaked
      distribution (confident) has low entropy; a spread-out/near-uniform
      distribution (uncertain across many candidates) has high entropy.
      Ranges from 0 (fully certain, all mass on one candidate) up to
      ``log2(n)`` for ``n`` candidates (fully uniform).

    Raises `ValueError` for malformed input: empty results, mismatched
    `probs`/`labels` lengths, or entries that aren't `(label, score)`
    pairs.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"uncertainty_score: unknown method {method!r}, expected one "
            f"of {_VALID_METHODS}"
        )

    scores = _extract_scores(ranked_results_or_probs)

    if method == "margin":
        return _margin_uncertainty(scores)
    return _entropy_uncertainty(scores)


def _iter_batch(batch_of_results):
    """Normalizes `batch_of_results` (a list of `(query_id, result)` pairs,
    or a dict `{query_id: result}`) into an iterable of `(query_id,
    result)` pairs."""
    if isinstance(batch_of_results, dict):
        return list(batch_of_results.items())
    return list(batch_of_results)


def prioritize_for_review(batch_of_results, *, method="margin", top_n=None):
    """Turns a batch of per-query classification results into a review
    queue, most-uncertain-first.

    `batch_of_results`: either

    - a list of `(query_id, ranked_results_or_probs)` tuples, or
    - a dict `{query_id: ranked_results_or_probs}`,

    where `ranked_results_or_probs` is one query's result in either shape
    accepted by `uncertainty_score`. `query_id` is any caller-supplied
    hashable identifier (e.g. a sample/read name) used only to label the
    output -- it is not interpreted.

    `method`: passed through to `uncertainty_score` for every entry (so all
    entries in one batch are scored the same way -- comparing "margin" on
    one entry against "entropy" on another would not be meaningful).

    `top_n`: if given, truncates the returned queue to the `top_n` most
    uncertain entries; if `None` (default), the full queue is returned.

    Returns a list of `(query_id, uncertainty_score, original_result)`
    tuples, sorted by `uncertainty_score` descending (most uncertain --
    i.e. the entries most worth a human/expert's attention -- first). Ties
    are broken by the input order (stable sort), so results are
    reproducible.

    Raises `ValueError` if `batch_of_results` is empty, or if any entry is
    malformed (propagated from `uncertainty_score`).
    """
    entries = _iter_batch(batch_of_results)
    if len(entries) == 0:
        raise ValueError(
            "prioritize_for_review: batch_of_results is empty -- nothing "
            "to prioritize."
        )

    scored = [
        (query_id, uncertainty_score(result, method=method), result)
        for query_id, result in entries
    ]
    scored.sort(key=lambda entry: entry[1], reverse=True)

    if top_n is not None:
        scored = scored[:top_n]

    return scored


def suggest_reference_additions(prioritized_queue, *, uncertainty_threshold):
    """Convenience filter over `prioritize_for_review`'s output: returns
    just the entries at or above `uncertainty_threshold`.

    These are the queries a k-mer classifier (whether
    `fastdna.taxonomy.classify()` or a `KmerVectorizer` + sklearn
    classifier) could not confidently place against any known
    class/reference. In a genomics lab workflow that is the concrete
    signal to (1) route the sample to an expert for a real label, and
    (2), once labeled, consider adding it to the reference database/
    training set -- rather than silently accepting a low-confidence guess
    or silently discarding the sample, which is the exact failure mode
    active learning exists to avoid.

    `prioritized_queue`: the output of `prioritize_for_review` -- a list of
    `(query_id, uncertainty_score, original_result)` tuples.

    `uncertainty_threshold`: entries with `uncertainty_score >=` this value
    are returned. Since both `uncertainty_score` methods use the "larger =
    more uncertain" convention, higher thresholds are stricter (fewer,
    more-uncertain entries returned). The right threshold is
    workflow/method-dependent (e.g. an entropy score is bounded by
    ``log2(n)`` for ``n`` candidates, while a margin score is always in
    ``[0, 1]``) -- callers should pick it based on their own data rather
    than a value hardcoded here.

    Returns a list in the same `(query_id, uncertainty_score,
    original_result)` shape as `prioritized_queue`, preserving the
    most-uncertain-first order, containing only the entries that cleared
    the threshold.
    """
    return [
        entry
        for entry in prioritized_queue
        if entry[1] >= uncertainty_threshold
    ]
