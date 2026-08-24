"""Tests for `fastdna.sklearn.KmerVectorizer` -- the scikit-learn-compatible
transformer that turns FASTQ file paths into a k-mer count feature matrix
(design doc §9.6).

The tests in this module fall into two groups: ordinary correctness tests
(shape, dtype, capping, decoded feature names), and one test whose entire
purpose is to *fail* if the vocabulary-leakage guarantee this class exists
for were ever broken -- see `test_unseen_sample_never_influences_vocabulary`.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

np = pytest.importorskip("numpy")

import scipy.sparse
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import fastdna
from fastdna.sklearn import KmerVectorizer


def write_fastq(tmp_path: pathlib.Path, name: str, reads: list[str]) -> pathlib.Path:
    p = tmp_path / name
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


# ---------------------------------------------------------------------------
# Basic fit/transform correctness
# ---------------------------------------------------------------------------


def test_fit_sets_vocabulary_and_n_features_in(tmp_path):
    # Same hand-checkable reads as test_api.py's EXPECTED_CANONICAL_COUNTS:
    # at k=5, "ACGTACGTAC" x3 yields exactly two distinct canonical 5-mers
    # (ACGTA, CGTAC), each occurring 9 times -- so an unbounded vocabulary
    # fit on a single such sample has exactly 2 entries.
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTAC"] * 3)

    vec = KmerVectorizer(k=5, top_features=None)
    fitted = vec.fit([str(path)])

    assert fitted is vec, "fit() must return self"
    assert len(vec.vocabulary_) == 2
    assert vec.n_features_in_ == 2
    assert vec.vocabulary_.dtype == np.uint64


def test_transform_shape_and_values_match_manual_counts(tmp_path):
    a = write_fastq(tmp_path, "a.fastq", ["ACGTACGTAC"] * 3)
    b = write_fastq(tmp_path, "b.fastq", ["ACGTACGTAC"] * 6)

    vec = KmerVectorizer(k=5, top_features=None)
    vec.fit([str(a), str(b)])
    matrix = vec.transform([str(a), str(b)])

    assert isinstance(matrix, scipy.sparse.csr_matrix)
    assert matrix.shape == (2, 2)  # 2 samples x 2 vocabulary k-mers

    dense = matrix.toarray()
    # Both canonical 5-mers occur 9x in `a` (3 reads) and 18x in `b` (6
    # reads) -- the same 1:9-per-read ratio test_api.py's
    # EXPECTED_CANONICAL_COUNTS hand-checks for a single sample.
    assert sorted(dense[0].tolist()) == [9.0, 9.0]
    assert sorted(dense[1].tolist()) == [18.0, 18.0]


def test_transform_ignores_kmers_outside_vocabulary(tmp_path):
    # Fit on a sample whose only content is A/C/G/T-cycling reads, so the
    # vocabulary contains only the two k-mers from test_fit_transform's own
    # reads above. Transform a *different* sample built entirely from a
    # disjoint alphabet (poly-G), whose k-mers cannot possibly be in that
    # vocabulary.
    train = write_fastq(tmp_path, "train.fastq", ["ACGTACGTAC"] * 3)
    unseen_alphabet = write_fastq(tmp_path, "other.fastq", ["GGGGGGGGGG"] * 3)

    vec = KmerVectorizer(k=5, top_features=None).fit([str(train)])
    matrix = vec.transform([str(unseen_alphabet)])

    assert matrix.shape == (1, 2)
    # None of the vocabulary k-mers (from the ACGT-cycling sample) appear
    # in a poly-G sample -- every column must come back zero rather than
    # raising.
    assert matrix.toarray().sum() == 0


def test_transform_before_fit_raises_not_fitted_error(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTAC"] * 3)
    vec = KmerVectorizer(k=5)

    with pytest.raises(NotFittedError):
        vec.transform([str(path)])

    with pytest.raises(NotFittedError):
        vec.get_feature_names_out()


def test_fit_on_empty_x_raises_valueerror():
    with pytest.raises(ValueError):
        KmerVectorizer(k=5).fit([])


def test_fit_on_a_bare_string_raises_typeerror_not_per_character_errors(tmp_path):
    # A single path passed as a bare string would iterate character by
    # character, producing a baffling per-character FileNotFoundError deep
    # inside the counting loop -- it must be a clear TypeError up front.
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTAC"] * 3)

    with pytest.raises(TypeError, match="list of paths"):
        KmerVectorizer(k=5).fit(str(path))


def test_transform_on_a_bare_string_raises_typeerror(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTAC"] * 3)
    vec = KmerVectorizer(k=5, top_features=None).fit([str(path)])

    with pytest.raises(TypeError, match="list of paths"):
        vec.transform(str(path))


def test_fit_on_duplicate_paths_raises_valueerror_naming_the_path(tmp_path):
    # The same file listed twice would double-count its k-mers' prevalence
    # (the vocabulary ranking's primary criterion) -- refuse it loudly.
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTAC"] * 3)

    with pytest.raises(ValueError) as exc_info:
        KmerVectorizer(k=5).fit([str(path), str(path)])

    # The message names the offending path (repr'd, so Windows backslashes
    # appear escaped).
    assert repr(str(path)) in str(exc_info.value)


def test_fit_transform_matches_separate_fit_then_transform(tmp_path):
    a = write_fastq(tmp_path, "a.fastq", ["ACGTACGTAC"] * 3)
    b = write_fastq(tmp_path, "b.fastq", ["AAACCCGGGT"] * 4)

    vec1 = KmerVectorizer(k=5, top_features=None)
    combined = vec1.fit_transform([str(a), str(b)])

    vec2 = KmerVectorizer(k=5, top_features=None)
    separate = vec2.fit([str(a), str(b)]).transform([str(a), str(b)])

    assert (combined != separate).nnz == 0
    assert list(vec1.vocabulary_) == list(vec2.vocabulary_)


# ---------------------------------------------------------------------------
# get_feature_names_out()
# ---------------------------------------------------------------------------


def test_get_feature_names_out_returns_decoded_sequences(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTAC"] * 3)
    k = 5

    vec = KmerVectorizer(k=k, top_features=None).fit([str(path)])
    names = vec.get_feature_names_out()

    assert len(names) == len(vec.vocabulary_)
    assert set(names) == {"ACGTA", "CGTAC"}
    for name in names:
        assert isinstance(name, str)
        assert len(name) == k
        assert set(name) <= set("ACGT")


# ---------------------------------------------------------------------------
# The leakage guarantee -- the entire point of this class
# ---------------------------------------------------------------------------


def _assert_pairwise_disjoint(k, samples):
    """Fails loudly if any two of `samples` share a canonical k-mer.

    The leakage tests below are only meaningful when each sample brings
    k-mers no other sample has -- otherwise "the vocabulary did not change"
    is trivially true and proves nothing. Reverse-complement collapsing
    makes that premise easy to break by accident (poly-G and poly-C are
    the same canonical k-mer), so this asserts it instead of assuming it.
    """
    kmer_sets = {}
    for name, path in samples.items():
        table = fastdna.count(str(path), k=k).table
        kmer_sets[name] = set(table.column("kmer_u64").to_pylist())
        assert kmer_sets[name], f"sample {name} produced no k-mers at k={k}"

    names = list(kmer_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            left, right = names[i], names[j]
            overlap = kmer_sets[left] & kmer_sets[right]
            assert not overlap, (
                f"samples {left} and {right} share canonical k-mers {overlap} -- "
                "these motifs cannot distinguish leakage from coincidence"
            )


def test_unseen_sample_never_influences_vocabulary(tmp_path):
    """Fits on two samples (A, B), then transforms a third (C) that was
    never part of fit(). C is built from a motif that shares no k-mers
    with A or B, so if C's content ever leaked into vocabulary selection
    -- e.g. a bug that had transform() mutate self.vocabulary_, or a bug
    that accidentally re-derived the vocabulary from whatever was last
    passed to any method -- this test catches it directly: it asserts
    vocabulary_ is bit-for-bit identical before and after the transform()
    call on the unseen sample, AND that a *separate* vectorizer fit on all
    three samples (A, B, C together) produces a vocabulary that genuinely
    differs, containing k-mers only C could have contributed. That second
    assertion rules out the test passing vacuously (e.g. because C's
    k-mers wouldn't have been selected anyway).
    """
    k = 6
    # These three motifs yield pairwise-DISJOINT canonical k-mer sets, which
    # is what makes C's contribution detectable at all. "Different letters"
    # is NOT sufficient for that: FastDNA counts *canonical* k-mers (the
    # lexicographic minimum of a k-mer and its reverse complement), so a
    # poly-G sample and a poly-C sample collapse onto the exact same
    # canonical 6-mer CCCCCC -- as do poly-A and poly-T. An earlier version
    # of this test used poly-G for C and silently proved nothing, because C
    # contributed no k-mer B had not already contributed. The
    # _assert_pairwise_disjoint() guard below makes that failure mode loud
    # rather than silent for anyone editing these motifs later.
    a = write_fastq(tmp_path, "a.fastq", ["AAAAAAAAAAAAAAAAAAAA"] * 5)
    b = write_fastq(tmp_path, "b.fastq", ["CCCCCCCCCCCCCCCCCCCC"] * 5)
    c = write_fastq(tmp_path, "c.fastq", ["ACACACACACACACACACAC"] * 5)
    _assert_pairwise_disjoint(k, {"a": a, "b": b, "c": c})

    vec = KmerVectorizer(k=k, top_features=None)
    vec.fit([str(a), str(b)])
    vocab_before = set(int(x) for x in vec.vocabulary_.tolist())
    assert vocab_before, "expected a/b to contribute at least one k-mer"

    # transform() on a sample never shown to fit() -- must not touch
    # vocabulary_ at all.
    vec.transform([str(a), str(b), str(c)])
    vocab_after = set(int(x) for x in vec.vocabulary_.tolist())
    assert vocab_after == vocab_before, "transform() must never mutate vocabulary_"

    # Prove C actually had something to leak: a vectorizer that DOES see C
    # during fit() ends up with a different, larger vocabulary containing
    # k-mers unique to C -- and those k-mers are provably absent from the
    # vocabulary that only ever saw A and B.
    vec_with_c = KmerVectorizer(k=k, top_features=None).fit([str(a), str(b), str(c)])
    vocab_with_c = set(int(x) for x in vec_with_c.vocabulary_.tolist())

    assert vocab_with_c != vocab_before
    c_only_kmers = vocab_with_c - vocab_before
    assert c_only_kmers, "expected c to contribute k-mers absent from a/b"
    assert c_only_kmers.isdisjoint(vocab_before)


def test_transform_on_training_samples_is_reproducible_regardless_of_fit_scope(tmp_path):
    """A second angle on the same guarantee: transforming sample A through
    a vectorizer fit on {A, B} vs one fit on {A, B, C} must NOT change how
    A's own features (the ones common to both vocabularies) are encoded --
    proving the projection is a stable function of the vocabulary alone,
    not of whatever else happened to be in the training set.
    """
    k = 5
    a = write_fastq(tmp_path, "a.fastq", ["ACGTACGTAC"] * 4)
    b = write_fastq(tmp_path, "b.fastq", ["ACGTACGTAC"] * 2)
    c = write_fastq(tmp_path, "c.fastq", ["TTTTTGGGGG"] * 3)

    vec_ab = KmerVectorizer(k=k, top_features=None).fit([str(a), str(b)])
    vec_abc = KmerVectorizer(k=k, top_features=None).fit([str(a), str(b), str(c)])

    names_ab = list(vec_ab.get_feature_names_out())
    names_abc = list(vec_abc.get_feature_names_out())
    shared = set(names_ab) & set(names_abc)
    assert shared, "expected a/b's k-mers to be present in both vocabularies"

    row_ab = dict(zip(names_ab, vec_ab.transform([str(a)]).toarray()[0]))
    row_abc = dict(zip(names_abc, vec_abc.transform([str(a)]).toarray()[0]))
    for name in shared:
        assert row_ab[name] == row_abc[name]


# ---------------------------------------------------------------------------
# top_features capping
# ---------------------------------------------------------------------------


def test_top_features_caps_vocabulary_size(tmp_path):
    # Many distinct 6-mers so there is something real to cap.
    reads = ["ACGTACGCATGCATGCACGTAGCTAGCTGACT"] * 5
    path = write_fastq(tmp_path, "a.fastq", reads)

    uncapped = KmerVectorizer(k=6, top_features=None).fit([str(path)])
    assert len(uncapped.vocabulary_) > 3

    capped = KmerVectorizer(k=6, top_features=3).fit([str(path)])
    assert len(capped.vocabulary_) == 3
    assert capped.n_features_in_ == 3

    # The capped vocabulary must be the top 3 of the uncapped one, by the
    # documented prevalence/total_freq/kmer_u64 ranking -- not an arbitrary
    # subset.
    assert list(capped.vocabulary_) == list(uncapped.vocabulary_[:3])


def test_top_features_cap_larger_than_available_kmers_is_a_noop(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTAC"] * 3)

    vec = KmerVectorizer(k=5, top_features=1_000_000).fit([str(path)])
    assert len(vec.vocabulary_) == 2  # only 2 distinct canonical 5-mers exist


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_non_positive_top_features_raises_valueerror(tmp_path, bad_value):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTAC"] * 3)
    with pytest.raises(ValueError):
        KmerVectorizer(k=5, top_features=bad_value).fit([str(path)])


# ---------------------------------------------------------------------------
# Integration: a real sklearn Pipeline
# ---------------------------------------------------------------------------


def _class0_reads() -> list[str]:
    # AT-rich motif repeated with light variation across reads.
    return ["AAATTTAAATTTAAATTTAA", "AAATTTAAATTTAAATTTAC", "AAATTTAAATTTAAATTTAG"] * 4


def _class1_reads() -> list[str]:
    # GC-rich motif, disjoint enough from class 0's to be trivially
    # separable by a linear classifier over k-mer counts.
    return ["GGGCCCGGGCCCGGGCCCGG", "GGGCCCGGGCCCGGGCCCGC", "GGGCCCGGGCCCGGGCCCGA"] * 4


def test_pipeline_fit_predict_end_to_end(tmp_path):
    """Builds a real two-class dataset, wires KmerVectorizer into a
    sklearn.pipeline.Pipeline ahead of LogisticRegression, fits and
    predicts, and checks the whole thing runs without error and returns
    predictions of the right shape. This is the test that proves the
    *point* of the feature: it must work as an ordinary Pipeline step, not
    just in isolation.
    """
    paths: list[str] = []
    labels: list[int] = []
    for i in range(4):
        p = write_fastq(tmp_path, f"class0_{i}.fastq", _class0_reads())
        paths.append(str(p))
        labels.append(0)
    for i in range(4):
        p = write_fastq(tmp_path, f"class1_{i}.fastq", _class1_reads())
        paths.append(str(p))
        labels.append(1)

    pipe = Pipeline(
        [
            ("kmers", KmerVectorizer(k=6, min_count=1, top_features=200)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(paths, labels)
    predictions = pipe.predict(paths)

    assert predictions.shape == (len(paths),)
    assert set(predictions.tolist()) <= {0, 1}
    # The two classes are built from disjoint motifs at k=6, so a linear
    # classifier over their k-mer counts should recover the (trivially
    # separable) training labels perfectly.
    assert predictions.tolist() == labels


def test_pipeline_cross_val_score_never_leaks_across_folds(tmp_path):
    """The scenario the whole class exists for: cross_val_score() calls
    fit() on each training fold and transform() on the corresponding test
    fold. This test does not assert on the score itself (that would be a
    weak, indirect check) -- it asserts the pipeline runs end to end under
    cross-validation without error and returns one score per fold, which
    is only possible if KmerVectorizer behaves correctly as a Pipeline
    step across repeated fit/transform cycles with different fold
    boundaries each time.
    """
    from sklearn.model_selection import cross_val_score

    paths: list[str] = []
    labels: list[int] = []
    for i in range(5):
        p = write_fastq(tmp_path, f"class0_{i}.fastq", _class0_reads())
        paths.append(str(p))
        labels.append(0)
    for i in range(5):
        p = write_fastq(tmp_path, f"class1_{i}.fastq", _class1_reads())
        paths.append(str(p))
        labels.append(1)

    pipe = Pipeline(
        [
            ("kmers", KmerVectorizer(k=6, min_count=1, top_features=200)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    scores = cross_val_score(pipe, paths, labels, cv=5)

    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)


# ---------------------------------------------------------------------------
# get_params/set_params -- required for clone()-based tools like GridSearchCV
# ---------------------------------------------------------------------------


def test_get_params_round_trips():
    vec = KmerVectorizer(k=21, min_count=3, top_features=500, threads=2)
    params = vec.get_params()

    assert params == {"k": 21, "min_count": 3, "top_features": 500, "threads": 2}

    from sklearn.base import clone

    cloned = clone(vec)
    assert cloned.get_params() == params
    assert not hasattr(cloned, "vocabulary_"), "clone() must yield an unfitted estimator"


# ---------------------------------------------------------------------------
# Audit additions (2026-08-24) -- coverage the original suite was missing.
#
# Everything below this line was added during a post-merge review. None of
# it changes `KmerVectorizer`; it pins behaviour the original tests either
# asserted too weakly to fail, or did not assert at all.
# ---------------------------------------------------------------------------


def test_transform_columns_line_up_positionally_with_get_feature_names_out(tmp_path):
    """`get_feature_names_out()[j]` must name column `j` of
    `transform()`'s output. Every downstream consumer of this class relies
    on that -- `fastdna.interpret.top_features(model.coef_[0],
    vec.get_feature_names_out())` maps importance `j` onto name `j` and
    would report confidently wrong biology if the two orders diverged.

    The original suite only asserted `set(names) == {...}` (see
    `test_get_feature_names_out_returns_decoded_sequences`), which is
    insensitive to any permutation of the columns. This asserts the
    correspondence *positionally*, against counts computed independently
    via `fastdna.count()` rather than via the vectorizer itself.
    """
    k = 6
    reads = ["ACGTACGCATGCATGCACGTAGCTAGCTGACT"] * 3
    a = write_fastq(tmp_path, "a.fastq", reads)
    b = write_fastq(tmp_path, "b.fastq", ["TTTTGGGGCCCCAAAATTTTGGGGCCCCAAAA"] * 2)

    vec = KmerVectorizer(k=k, top_features=None).fit([str(a), str(b)])
    names = list(vec.get_feature_names_out())
    dense = vec.transform([str(a), str(b)]).toarray()

    assert len(names) == dense.shape[1] == len(vec.vocabulary_)

    for sample_index, path in enumerate([a, b]):
        table = fastdna.count(str(path), k=k).table
        truth = dict(
            zip(table.column("kmer_sequence").to_pylist(), table.column("frequency").to_pylist())
        )
        for column_index, name in enumerate(names):
            assert dense[sample_index, column_index] == truth.get(name, 0), (
                f"column {column_index} of transform() does not hold the count of "
                f"get_feature_names_out()[{column_index}] == {name!r} for {path.name}"
            )


def test_transform_with_zero_vocabulary_overlap_returns_a_true_all_zero_row(tmp_path):
    """A sample sharing no k-mer at all with the fitted vocabulary must
    produce a full-width all-zero row -- not an error, not a narrower
    matrix, and not a matrix with explicitly-stored zeros that would make
    `nnz` lie to a downstream sparse consumer.
    """
    train = write_fastq(tmp_path, "train.fastq", ["ACGTACGTAC"] * 3)
    disjoint = write_fastq(tmp_path, "disjoint.fastq", ["GGGGGGGGGG"] * 3)
    _assert_pairwise_disjoint(5, {"train": train, "disjoint": disjoint})

    vec = KmerVectorizer(k=5, top_features=None).fit([str(train)])
    matrix = vec.transform([str(disjoint)])

    assert isinstance(matrix, scipy.sparse.csr_matrix)
    assert matrix.shape == (1, len(vec.vocabulary_))
    assert matrix.dtype == np.float64
    assert matrix.nnz == 0, "an all-zero row must contain no stored entries"
    assert matrix.toarray().tolist() == [[0.0] * len(vec.vocabulary_)]


def test_leakage_guarantee_check_actually_catches_a_leaky_transform(tmp_path):
    """Independent mutation test of the leakage guarantee.

    `test_unseen_sample_never_influences_vocabulary` asserts that
    `transform()` leaves `vocabulary_` untouched. That assertion is only
    worth anything if it would *fail* on an implementation that broke the
    guarantee. Here a deliberately-leaky subclass extends the vocabulary
    inside `transform()` -- precisely the bug the guarantee exists to
    forbid -- and this test asserts the same check the real test performs
    rejects it.

    Without this, "the leakage test passes" is unfalsifiable: it would
    also pass against a `transform()` that did nothing at all.
    """

    class LeakyVectorizer(KmerVectorizer):
        def transform(self, X):
            # The bug: let the transformed (possibly held-out) samples add
            # their own k-mers to the vocabulary.
            for path in [str(p) for p in X]:
                table = fastdna.count(
                    path, k=self.k, min_count=self.min_count, threads=self.threads
                ).table
                for kmer, seq in zip(
                    table.column("kmer_u64").to_pylist(),
                    table.column("kmer_sequence").to_pylist(),
                ):
                    if kmer not in self._vocab_index_:
                        self._vocab_index_[kmer] = len(self._vocab_index_)
                        self._feature_sequences_.append(seq)
                        self.vocabulary_ = np.append(self.vocabulary_, np.uint64(kmer))
            return super().transform(X)

    k = 6
    a = write_fastq(tmp_path, "a.fastq", ["AAAAAAAAAAAAAAAAAAAA"] * 5)
    b = write_fastq(tmp_path, "b.fastq", ["CCCCCCCCCCCCCCCCCCCC"] * 5)
    c = write_fastq(tmp_path, "c.fastq", ["ACACACACACACACACACAC"] * 5)
    _assert_pairwise_disjoint(k, {"a": a, "b": b, "c": c})

    # Control: the real class passes the guarantee check.
    honest = KmerVectorizer(k=k, top_features=None).fit([str(a), str(b)])
    before = set(int(x) for x in honest.vocabulary_.tolist())
    honest.transform([str(c)])
    assert set(int(x) for x in honest.vocabulary_.tolist()) == before

    # Mutant: the same check must reject it.
    leaky = LeakyVectorizer(k=k, top_features=None).fit([str(a), str(b)])
    before_leaky = set(int(x) for x in leaky.vocabulary_.tolist())
    assert before_leaky == before, "the mutant must start from the same fitted vocabulary"
    leaky.transform([str(c)])
    after_leaky = set(int(x) for x in leaky.vocabulary_.tolist())

    assert after_leaky != before_leaky, (
        "the mutation-test subclass did not actually leak -- this test proves "
        "nothing unless the leaky transform() genuinely widens vocabulary_"
    )
    assert after_leaky - before_leaky, "expected c's k-mers to have leaked in"


def test_cross_validation_vocabulary_excludes_kmers_unique_to_the_test_fold(tmp_path):
    """The assertion `test_pipeline_cross_val_score_never_leaks_across_folds`
    is named for but does not make.

    That test asserts only `len(scores) == 5` and `0.0 <= s <= 1.0` -- both
    trivially true of *any* accuracy array, including one produced by a
    vectorizer that pooled every fold's k-mers before splitting. It would
    not fail if leakage were reintroduced.

    This test walks the same `KFold` splits scikit-learn would, fits a
    fresh vectorizer on each training fold, and asserts directly that no
    k-mer contributed *only* by that split's held-out samples ever appears
    in the fitted vocabulary. Each sample carries a private marker motif so
    "unique to the test fold" is a real, non-empty set at every split --
    asserted, not assumed.
    """
    from sklearn.model_selection import KFold

    k = 6
    # A shared backbone every sample has, plus a per-sample marker motif no
    # other sample carries, so each fold's held-out samples own k-mers the
    # training fold provably never sees.
    # Chosen so that, at k=6 and after canonical (reverse-complement)
    # collapsing, each marker contributes 15 k-mers no other marker and no
    # backbone read supplies. Simple patterns like ACACAC.../GTGTGT... are
    # NOT usable here: they are reverse complements of one another and
    # collapse onto the same canonical k-mers, which would make the
    # "unique to the test fold" set empty and the test vacuous.
    markers = [
        "GCTAAAGACAATTACATAAC",
        "ATACACGTCAGCACGAAACT",
        "TGTTGGCCCAGTGTGAATCG",
        "CTTAAGGGTTAAGTAAGTGT",
        "CTGTGTCCACCCCATCGGAC",
        "TTGACAGGTCACGCAGAGGC",
    ]
    paths = []
    for i, marker in enumerate(markers):
        p = write_fastq(tmp_path, f"s{i}.fastq", ["ACGTACGTACGTACGTACGT"] * 3 + [marker] * 3)
        paths.append(str(p))

    def kmers_of(path):
        return set(fastdna.count(path, k=k).table.column("kmer_u64").to_pylist())

    per_sample = {p: kmers_of(p) for p in paths}

    checked_splits = 0
    for train_index, test_index in KFold(n_splits=3, shuffle=True, random_state=0).split(paths):
        train_paths = [paths[i] for i in train_index]
        test_paths = [paths[i] for i in test_index]

        train_kmers = set().union(*(per_sample[p] for p in train_paths))
        test_only = set().union(*(per_sample[p] for p in test_paths)) - train_kmers
        assert test_only, (
            "this split's held-out samples contribute no k-mer the training fold "
            "lacks -- the leakage assertion below would be vacuous"
        )

        vec = KmerVectorizer(k=k, top_features=None).fit(train_paths)
        fitted_vocab = set(int(x) for x in vec.vocabulary_.tolist())

        assert fitted_vocab & test_only == set(), (
            "vocabulary fitted on the training fold contains k-mers only the "
            "held-out fold could have supplied -- feature selection leaked"
        )
        # And the vocabulary must be exactly the training fold's k-mers,
        # not merely a subset avoiding the marked ones.
        assert fitted_vocab == train_kmers

        # transform() on the held-out fold must not change that.
        vec.transform(test_paths)
        assert set(int(x) for x in vec.vocabulary_.tolist()) == fitted_vocab
        checked_splits += 1

    assert checked_splits == 3
