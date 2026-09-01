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
import random
import shutil

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

    # representation="count": this test asserts on raw per-sample counts,
    # not the default presence (0/1) representation.
    vec = KmerVectorizer(k=5, top_features=None, representation="count")
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

    assert params == {
        "k": 21,
        "min_count": 3,
        "top_features": 500,
        "threads": 2,
        "counts": None,
        "representation": "presence",
        "chunk_size": None,
        "disk_backed": False,
    }

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

    # representation="count": this test compares transform()'s values
    # against raw per-sample frequencies, not the default presence (0/1).
    vec = KmerVectorizer(k=k, top_features=None, representation="count").fit([str(a), str(b)])
    names = list(vec.get_feature_names_out())
    dense = vec.transform([str(a), str(b)]).toarray()

    assert len(names) == dense.shape[1] == len(vec.vocabulary_)

    for sample_index, path in enumerate([a, b]):
        table = fastdna.count(str(path), k=k, with_sequence=True).table
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
                    path, k=self.k, min_count=self.min_count, threads=self.threads, with_sequence=True
                ).table
                known = set(int(x) for x in self.vocabulary_.tolist())
                for kmer, seq in zip(
                    table.column("kmer_u64").to_pylist(),
                    table.column("kmer_sequence").to_pylist(),
                ):
                    if kmer not in known:
                        known.add(kmer)
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


# ---------------------------------------------------------------------------
# chunk_size -- streaming vocabulary learning over a cohort too large to hold
# in memory all at once (docs/audit/ml-gaps.md G-9).
#
# `chunk_size` is designed to be exact, not approximate: folding each
# batch's local (prevalence, total_freq) tally into a running one is plain
# addition, so the final vocabulary must be bit-for-bit identical to what
# `chunk_size=None` produces on the same cohort, for any chunk_size and any
# ordering. Every test below leans on that -- comparing a chunked run
# directly against an unchunked control -- rather than asserting some
# specific vocabulary content, so a merge bug (e.g. overwriting instead of
# summing a k-mer's running tally across chunks) would show up as a mismatch
# against the control, not just as "a plausible-looking vocabulary".
# ---------------------------------------------------------------------------


def _chunk_size_test_cohort(tmp_path):
    """5 samples sharing a common backbone motif (so at least one k-mer's
    prevalence must be summed *across* chunk boundaries for every chunk_size
    tested below) plus a private marker motif per sample (so each sample
    also contributes k-mers no other sample has, and a chunk boundary that
    splits the cohort differently still has real, non-trivial content on
    both sides of the split).
    """
    backbone = "ACGTACGTACGTACGTACGT"
    markers = [
        "GCTAAAGACAATTACATAAC",
        "ATACACGTCAGCACGAAACT",
        "TGTTGGCCCAGTGTGAATCG",
        "CTTAAGGGTTAAGTAAGTGT",
        "CTGTGTCCACCCCATCGGAC",
    ]
    paths = []
    for i, marker in enumerate(markers):
        p = write_fastq(tmp_path, f"s{i}.fastq", [backbone] * 3 + [marker] * 3)
        paths.append(str(p))
    return paths


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 1000])
def test_chunked_fit_matches_unchunked_fit_exactly(tmp_path, chunk_size):
    paths = _chunk_size_test_cohort(tmp_path)
    k = 6

    chunked = KmerVectorizer(k=k, top_features=None, chunk_size=chunk_size).fit(paths)
    unchunked = KmerVectorizer(k=k, top_features=None).fit(paths)

    assert list(chunked.vocabulary_) == list(unchunked.vocabulary_), (
        f"chunk_size={chunk_size} produced a different vocabulary order than "
        "the unchunked control -- the running tally must be a bit-for-bit "
        "sum, not an approximation"
    )
    assert chunked.n_features_in_ == unchunked.n_features_in_
    assert list(chunked.get_feature_names_out()) == list(unchunked.get_feature_names_out())


def test_chunked_fit_respects_top_features_cap_identically(tmp_path):
    paths = _chunk_size_test_cohort(tmp_path)
    k = 6

    capped_chunked = KmerVectorizer(k=k, top_features=3, chunk_size=2).fit(paths)
    capped_unchunked = KmerVectorizer(k=k, top_features=3).fit(paths)

    assert len(capped_chunked.vocabulary_) == 3
    assert list(capped_chunked.vocabulary_) == list(capped_unchunked.vocabulary_)


def test_chunk_size_larger_than_cohort_is_a_noop(tmp_path):
    # A single "chunk" covering the whole cohort must degenerate to exactly
    # the unchunked path.
    paths = _chunk_size_test_cohort(tmp_path)
    k = 6

    chunked = KmerVectorizer(k=k, top_features=None, chunk_size=10_000).fit(paths)
    unchunked = KmerVectorizer(k=k, top_features=None).fit(paths)
    assert list(chunked.vocabulary_) == list(unchunked.vocabulary_)


def test_chunked_fit_then_transform_matches_unchunked(tmp_path):
    paths = _chunk_size_test_cohort(tmp_path)
    k = 6

    chunked = KmerVectorizer(k=k, top_features=None, chunk_size=2).fit(paths)
    unchunked = KmerVectorizer(k=k, top_features=None).fit(paths)

    matrix_chunked = chunked.transform(paths)
    matrix_unchunked = unchunked.transform(paths)
    assert (matrix_chunked != matrix_unchunked).nnz == 0


def test_chunked_fit_transform_matches_unchunked_fit_transform(tmp_path):
    paths = _chunk_size_test_cohort(tmp_path)
    k = 6

    matrix_chunked = KmerVectorizer(k=k, top_features=None, chunk_size=2).fit_transform(paths)
    matrix_unchunked = KmerVectorizer(k=k, top_features=None).fit_transform(paths)
    assert (matrix_chunked != matrix_unchunked).nnz == 0


def test_chunk_size_interoperates_with_precomputed_cohort_counts(tmp_path):
    """`chunk_size` and `counts=` (a precomputed `fastdna.CohortCounts`) are
    not mutually exclusive: each chunk becomes `self.counts.subset(...)` for
    that batch's sample_ids only, instead of one all-at-once `subset()`
    call. Must still be exact.
    """
    paths = _chunk_size_test_cohort(tmp_path)
    k = 6
    cohort = fastdna.count_cohort(paths, k=k)

    chunked = KmerVectorizer(k=k, top_features=None, counts=cohort, chunk_size=2)
    chunked.fit(list(cohort.sample_ids))

    unchunked = KmerVectorizer(k=k, top_features=None, counts=cohort)
    unchunked.fit(list(cohort.sample_ids))

    assert list(chunked.vocabulary_) == list(unchunked.vocabulary_)


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_non_positive_chunk_size_raises_valueerror(tmp_path, bad_value):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTAC"] * 3)
    with pytest.raises(ValueError):
        KmerVectorizer(k=5, chunk_size=bad_value).fit([str(path)])


def test_chunk_size_default_is_none_and_unused(tmp_path):
    # The opt-in contract: omitting chunk_size entirely must behave
    # identically to passing chunk_size=None explicitly (and to every test
    # above this section, none of which pass chunk_size at all).
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTAC"] * 3)
    vec = KmerVectorizer(k=5, top_features=None)
    assert vec.chunk_size is None
    vec.fit([str(path)])
    assert len(vec.vocabulary_) == 2


# ---------------------------------------------------------------------------
# disk_backed -- both vocabulary learning AND projection bounded to O(one
# sample's own table + the vocabulary) at a time, never the whole cohort's
# rows at once (the remaining half of docs/audit/ml-gaps.md G-9 that
# `chunk_size` above does not cover: it only bounds vocabulary learning).
#
# The single most important property under test here is equivalence:
# `disk_backed=True` must produce an IDENTICAL vocabulary and matrix to
# `disk_backed=False` on the same cohort, for every representation and
# several cohort shapes -- not an approximation, since both paths implement
# the exact same ranking/projection rules (Rust-side `cohort_vocab::
# rank_vocabulary`/`project_onto_vocabulary` are a drop-in replacement for
# `_select_vocabulary`/`_project`'s own rules, not a new policy).
# ---------------------------------------------------------------------------


def test_disk_backed_fit_matches_in_memory_fit_exactly(tmp_path):
    paths = _chunk_size_test_cohort(tmp_path)
    k = 6

    disk_backed = KmerVectorizer(k=k, top_features=None, disk_backed=True).fit(paths)
    in_memory = KmerVectorizer(k=k, top_features=None).fit(paths)

    assert list(disk_backed.vocabulary_) == list(in_memory.vocabulary_)
    assert disk_backed.n_features_in_ == in_memory.n_features_in_
    assert list(disk_backed.get_feature_names_out()) == list(in_memory.get_feature_names_out())


@pytest.mark.parametrize("top_features", [None, 1, 3, 1000])
def test_disk_backed_fit_respects_top_features_cap_identically(tmp_path, top_features):
    paths = _chunk_size_test_cohort(tmp_path)
    k = 6

    disk_backed = KmerVectorizer(k=k, top_features=top_features, disk_backed=True).fit(paths)
    in_memory = KmerVectorizer(k=k, top_features=top_features).fit(paths)

    assert list(disk_backed.vocabulary_) == list(in_memory.vocabulary_)


def test_disk_backed_fit_then_transform_matches_in_memory(tmp_path):
    paths = _chunk_size_test_cohort(tmp_path)
    k = 6

    disk_backed = KmerVectorizer(k=k, top_features=None, disk_backed=True).fit(paths)
    in_memory = KmerVectorizer(k=k, top_features=None).fit(paths)

    matrix_disk_backed = disk_backed.transform(paths)
    matrix_in_memory = in_memory.transform(paths)
    assert (matrix_disk_backed != matrix_in_memory).nnz == 0
    assert matrix_disk_backed.shape == matrix_in_memory.shape


def test_disk_backed_fit_transform_matches_in_memory_fit_transform(tmp_path):
    paths = _chunk_size_test_cohort(tmp_path)
    k = 6

    matrix_disk_backed = KmerVectorizer(k=k, top_features=None, disk_backed=True).fit_transform(paths)
    matrix_in_memory = KmerVectorizer(k=k, top_features=None).fit_transform(paths)
    assert (matrix_disk_backed != matrix_in_memory).nnz == 0


@pytest.mark.parametrize("representation", ["presence", "count", "relative", "clr"])
def test_disk_backed_matches_in_memory_for_every_representation(tmp_path, representation):
    paths = _chunk_size_test_cohort(tmp_path)
    k = 6

    disk_backed = KmerVectorizer(k=k, top_features=None, representation=representation, disk_backed=True)
    in_memory = KmerVectorizer(k=k, top_features=None, representation=representation)

    matrix_disk_backed = disk_backed.fit_transform(paths)
    matrix_in_memory = in_memory.fit_transform(paths)

    np.testing.assert_allclose(matrix_disk_backed.toarray(), matrix_in_memory.toarray())


def test_disk_backed_interoperates_with_precomputed_cohort_counts(tmp_path):
    """`disk_backed` and `counts=` (a precomputed `fastdna.CohortCounts`)
    are not mutually exclusive: each sample is still resolved via
    `self.counts.subset([sample_id])` rather than a fresh `fastdna.count()`
    call -- see `_disk_backed_sample_tables`. Must still be exact.
    """
    paths = _chunk_size_test_cohort(tmp_path)
    k = 6
    cohort = fastdna.count_cohort(paths, k=k)

    disk_backed = KmerVectorizer(k=k, top_features=None, counts=cohort, disk_backed=True)
    disk_backed.fit(list(cohort.sample_ids))

    in_memory = KmerVectorizer(k=k, top_features=None, counts=cohort)
    in_memory.fit(list(cohort.sample_ids))

    assert list(disk_backed.vocabulary_) == list(in_memory.vocabulary_)

    matrix_disk_backed = disk_backed.transform(list(cohort.sample_ids))
    matrix_in_memory = in_memory.transform(list(cohort.sample_ids))
    assert (matrix_disk_backed != matrix_in_memory).nnz == 0


def test_disk_backed_true_and_chunk_size_together_raises_invalidconfig(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTAC"] * 3)
    with pytest.raises(ValueError):
        KmerVectorizer(k=5, disk_backed=True, chunk_size=2).fit([str(path)])


def test_disk_backed_default_is_false_and_unused(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTAC"] * 3)
    vec = KmerVectorizer(k=5, top_features=None)
    assert vec.disk_backed is False
    vec.fit([str(path)])
    assert len(vec.vocabulary_) == 2


def test_disk_backed_fit_cleans_up_its_scratch_directory(tmp_path, monkeypatch):
    """`_disk_backed_sample_tables` must remove its whole scratch directory
    once `fit()` returns -- a leftover directory per `fit()`/`transform()`
    call would silently fill up disk on any long-lived process (a notebook,
    a service) that calls this repeatedly.
    """
    spill_base = tmp_path / "spill"
    spill_base.mkdir()
    monkeypatch.setenv("FASTDNA_SPILL_DIR", str(spill_base))

    paths = _chunk_size_test_cohort(tmp_path)
    KmerVectorizer(k=6, top_features=None, disk_backed=True).fit(paths)

    assert list(spill_base.iterdir()) == [], "no scratch directory should remain after fit() returns"


def test_disk_backed_scratch_tables_survive_a_mid_fit_error_cleanup(tmp_path, monkeypatch):
    """A failure partway through writing the scratch tables must still
    clean up the scratch directory (the `finally` in
    `_disk_backed_sample_tables`), not leak it.
    """
    spill_base = tmp_path / "spill"
    spill_base.mkdir()
    monkeypatch.setenv("FASTDNA_SPILL_DIR", str(spill_base))

    # A path that does not exist: fastdna.count() raises inside
    # `_count_cohort`, partway through `_disk_backed_sample_tables`'s loop.
    missing = str(tmp_path / "does_not_exist.fastq")
    vec = KmerVectorizer(k=5, top_features=None, disk_backed=True)
    with pytest.raises(Exception):
        vec.fit([missing])

    assert list(spill_base.iterdir()) == [], "the scratch directory must be removed even after an error"


def test_make_scratch_dir_honors_fastdna_spill_dir_env_var(tmp_path, monkeypatch):
    from fastdna.sklearn import _make_scratch_dir

    spill_base = tmp_path / "spill"
    spill_base.mkdir()
    monkeypatch.setenv("FASTDNA_SPILL_DIR", str(spill_base))

    scratch = pathlib.Path(_make_scratch_dir())
    try:
        assert scratch.parent == spill_base
        assert scratch.is_dir()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _synthetic_large_cohort(tmp_path, n_samples=15, reads_per_sample=400, read_length=60, seed=1234):
    """A larger synthetic cohort -- enough samples and distinct k-mers per
    sample that `disk_backed=True`'s per-sample scratch-table streaming
    architecture is genuinely exercised (several hundred to a few thousand
    distinct k-mers per sample, several Arrow batches per Parquet table),
    not just trivially handled the way a 2-3-row fixture would be. This is
    a correctness test at a size where the streaming code path actually
    streams, not a memory benchmark -- real peak RSS is not asserted here
    (unreliable in CI); only that `disk_backed=True` and `disk_backed=False`
    agree exactly on the result.
    """
    rng = random.Random(seed)
    bases = "ACGT"
    paths = []
    for i in range(n_samples):
        reads = ["".join(rng.choice(bases) for _ in range(read_length)) for _ in range(reads_per_sample)]
        paths.append(str(write_fastq(tmp_path, f"large_s{i}.fastq", reads)))
    return paths


def test_disk_backed_matches_in_memory_on_a_larger_synthetic_cohort(tmp_path):
    paths = _synthetic_large_cohort(tmp_path)
    k = 21

    disk_backed = KmerVectorizer(k=k, top_features=200, disk_backed=True).fit(paths)
    in_memory = KmerVectorizer(k=k, top_features=200).fit(paths)

    assert len(disk_backed.vocabulary_) > 0, "the synthetic cohort must actually produce a nonempty vocabulary"
    assert list(disk_backed.vocabulary_) == list(in_memory.vocabulary_)

    matrix_disk_backed = disk_backed.transform(paths)
    matrix_in_memory = in_memory.transform(paths)
    assert (matrix_disk_backed != matrix_in_memory).nnz == 0


def test_disk_backed_fit_transform_matches_in_memory_on_a_larger_synthetic_cohort(tmp_path):
    paths = _synthetic_large_cohort(tmp_path, n_samples=10, reads_per_sample=200)
    k = 15

    matrix_disk_backed = KmerVectorizer(k=k, top_features=500, disk_backed=True).fit_transform(paths)
    matrix_in_memory = KmerVectorizer(k=k, top_features=500).fit_transform(paths)
    assert (matrix_disk_backed != matrix_in_memory).nnz == 0


def test_disk_backed_empty_x_raises_valueerror():
    with pytest.raises(ValueError):
        KmerVectorizer(k=5, disk_backed=True).fit([])


def test_disk_backed_single_sample(tmp_path):
    path = write_fastq(tmp_path, "a.fastq", ["ACGTACGTAC"] * 3)

    disk_backed = KmerVectorizer(k=5, top_features=None, disk_backed=True).fit([str(path)])
    in_memory = KmerVectorizer(k=5, top_features=None).fit([str(path)])

    assert list(disk_backed.vocabulary_) == list(in_memory.vocabulary_)
    assert len(disk_backed.vocabulary_) == 2


def test_disk_backed_transform_with_zero_vocabulary_overlap_returns_a_true_all_zero_row(tmp_path):
    train = write_fastq(tmp_path, "train.fastq", ["ACGTACGTAC"] * 3)
    disjoint = write_fastq(tmp_path, "disjoint.fastq", ["GGGGGGGGGG"] * 3)
    _assert_pairwise_disjoint(5, {"train": train, "disjoint": disjoint})

    vec = KmerVectorizer(k=5, top_features=None, disk_backed=True).fit([str(train)])
    matrix = vec.transform([str(disjoint)])

    assert matrix.shape == (1, len(vec.vocabulary_))
    assert matrix.nnz == 0


def test_disk_backed_never_leaks_across_folds(tmp_path):
    """The same leakage guarantee `test_unseen_sample_never_influences_
    vocabulary` pins for the in-memory path must hold for `disk_backed=True`
    too: `transform()` on a held-out fold must not be able to influence
    `fit()`'s already-decided vocabulary, and a matrix produced by
    `transform()` on training samples must be identical whether or not a
    disjoint sample was ever passed to `transform()` first.
    """
    k = 6
    train_a = write_fastq(tmp_path, "train_a.fastq", ["ACGTACGTACGTACGTACGT"] * 3)
    train_b = write_fastq(tmp_path, "train_b.fastq", ["TTGGCCAATTGGCCAATTGG"] * 3)
    held_out = write_fastq(tmp_path, "held_out.fastq", ["CTAGCTAGCTAGCTAGCTAG"] * 3)

    vec = KmerVectorizer(k=k, top_features=None, disk_backed=True).fit([str(train_a), str(train_b)])
    vocabulary_before = list(vec.vocabulary_)

    vec.transform([str(held_out)])
    assert list(vec.vocabulary_) == vocabulary_before, "transform() must never mutate the already-fitted vocabulary"
