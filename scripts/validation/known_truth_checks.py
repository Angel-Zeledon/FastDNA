"""Checks modules whose claim can be tested against a truth we construct.

Some claims have an external reference to check against -- KMC3 for
counting, Merqury for QV, Mash for distances -- and those live in the
neighbouring scripts. The modules here have no such reference, but their
claims are still checkable, because the right answer can be *built*:

  * `calibration` claims to turn miscalibrated scores into genuine
    probabilities. Generate data where P(y=1|x) is known analytically, feed
    a deliberately miscalibrated model, and measure the distance to the true
    probability before and after.
  * `anomaly` claims to find the sample that does not belong. Build a
    homogeneous cohort, inject one organism from elsewhere, and see whether
    exactly that one is flagged.
  * `genomic_model` claims to warn when asked to predict outside its
    training domain. Fit on one organism, predict on another.
  * `validate_generated` claims to judge whether sequences are
    compositionally plausible. The sharpest test is the **null**: sequences
    drawn from the reference's own distribution must not be called
    deviated.

The last of those found a real defect. Before the coverage guard added
alongside this script, `validate_generated` called identical distributions
"very deviated" at k=6, 11 and 21 -- a false positive that fired on any
input at high k. It is included here as a permanent null control rather
than trusted to stay fixed.

None of these needs network access or Docker; they are slow only because
they count k-mers. Exits non-zero if any check fails.

Usage:

    python scripts/validation/known_truth_checks.py
    python scripts/validation/known_truth_checks.py --only calibration anomaly
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile
import warnings
from typing import Callable, Dict, List, Tuple


def _write_fastq(directory: pathlib.Path, name: str, sequence: str, read_len: int = 120) -> str:
    reads = [
        sequence[i : i + read_len]
        for i in range(0, len(sequence) - read_len, read_len // 2)
    ]
    path = directory / name
    path.write_text(
        "".join(f"@{name}_{i}\n{r}\n+\n{'I' * len(r)}\n" for i, r in enumerate(reads))
    )
    return str(path)


def check_calibration() -> Tuple[bool, str]:
    """Does calibration recover the TRUE probability, not merely a
    better-looking curve?

    The existing suite checks that calibration closes the gap
    `evaluation.calibration_report` complains about, plus structural
    properties. Neither compares against the probability that actually
    generated the labels, which is what the module claims to recover and is
    knowable here by construction.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from fastdna.calibration import calibrate

    rng = np.random.default_rng(7)
    n = 3000
    X = rng.normal(size=(n, 3))
    logit = 1.8 * X[:, 0] - 1.2 * X[:, 1] + 0.5 * X[:, 2]
    p_true = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p_true).astype(int)
    train, calib, test = slice(0, 1000), slice(1000, 2000), slice(2000, 3000)

    class Miscalibrated:
        """Ranks correctly, but its probabilities are squashed toward 0.5 --
        the exact failure calibration exists to repair."""

        def __init__(self):
            self.inner = LogisticRegression(max_iter=1000)

        def fit(self, X, y):
            self.inner.fit(X, y)
            self.classes_ = self.inner.classes_
            return self

        def predict_proba(self, X):
            p = self.inner.predict_proba(X)[:, 1]
            p = 0.5 + (p - 0.5) * 0.25
            return np.column_stack([1 - p, p])

    model = Miscalibrated().fit(X[train], y[train])
    raw = model.predict_proba(X[test])[:, 1]
    calibrated = calibrate(model, X[calib], y[calib]).predict_proba(X[test])[:, 1]
    truth = p_true[test]

    before = float(np.abs(raw - truth).mean())
    after = float(np.abs(calibrated - truth).mean())
    ok = after < before * 0.5
    return ok, (f"error vs TRUE probability {before:.4f} -> {after:.4f} "
                f"({1 - after / before:.0%} closer)")


def check_anomaly() -> Tuple[bool, str]:
    """Is the injected contaminant the sample that gets flagged, and only it?"""
    import numpy as np

    from fastdna.anomaly import flag_cohort

    rng = np.random.default_rng(3)
    bases = np.array(list("ACGT"))
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        root = "".join(rng.choice(bases, size=4000))
        paths = []
        for i in range(12):
            seq = list(root)
            for pos in rng.choice(len(seq), size=20, replace=False):
                seq[pos] = str(rng.choice(bases))
            paths.append(_write_fastq(directory, f"normal_{i}.fastq", "".join(seq)))
        # A different organism entirely: shares no ancestry with the rest.
        paths.append(_write_fastq(directory, "INTRUDER.fastq",
                                  "".join(rng.choice(bases, size=4000))))

        table = flag_cohort(paths, k=21, sketch_size=500)

    names = [pathlib.Path(p).name for p in table.column("sample").to_pylist()]
    flags = table.column("is_outlier").to_pylist()
    intruder_flagged = flags[names.index("INTRUDER.fastq")]
    false_positives = sum(flags) - int(intruder_flagged)
    ok = bool(intruder_flagged) and false_positives == 0
    return ok, f"intruder flagged={bool(intruder_flagged)}, false positives={false_positives}"


def check_genomic_model_domain() -> Tuple[bool, str]:
    """Silence inside the training domain, a warning outside it. Both halves
    matter: a check that always warns is as useless as one that never does."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from fastdna.genomic_model import GenomicModel, OutOfDistributionWarning
    from fastdna.sklearn import KmerVectorizer

    rng = np.random.default_rng(11)
    bases = np.array(list("ACGT"))
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        root = "".join(rng.choice(bases, size=4000))
        train = []
        for i in range(16):
            seq = list(root)
            for pos in rng.choice(len(seq), size=20, replace=False):
                seq[pos] = str(rng.choice(bases))
            train.append(_write_fastq(directory, f"A_{i}.fastq", "".join(seq)))
        y = np.array([i % 2 for i in range(16)])

        seq = list(root)
        for pos in rng.choice(len(seq), size=20, replace=False):
            seq[pos] = str(rng.choice(bases))
        in_domain = _write_fastq(directory, "A_new.fastq", "".join(seq))
        out_of_domain = _write_fastq(directory, "B_alien.fastq",
                                     "".join(rng.choice(bases, size=4000)))

        model = GenomicModel.fit(
            train, y,
            vectorizer=KmerVectorizer(k=21, top_features=200),
            estimator=LogisticRegression(max_iter=1000),
            k=21, sketch_size=500,
        )
        results = {}
        for label, path in (("in", in_domain), ("out", out_of_domain)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.predict([path])
                results[label] = any(
                    issubclass(w.category, OutOfDistributionWarning) for w in caught
                )

    ok = results["out"] and not results["in"]
    return ok, f"in-domain warned={results['in']}, out-of-domain warned={results['out']}"


def check_generated_null() -> Tuple[bool, str]:
    """THE NULL. Sequences from the reference's own distribution must never
    be called deviated. This found a real defect: before the coverage guard,
    identical distributions were called "very deviated" at k=6, 11 and 21.
    """
    import numpy as np

    from fastdna.validate_generated import validate_generated

    rng = np.random.default_rng(5)
    bases = np.array(list("ACGT"))
    pool = [
        "".join(rng.choice(bases, p=[0.3, 0.2, 0.2, 0.3], size=20_000))
        for _ in range(16)
    ]
    table = validate_generated(
        pool[:8], pool[8:], k=(3, 6, 11, 21), check_containment=False
    ).composition
    verdicts = {
        table.column("k")[i].as_py(): table.column("verdict")[i].as_py()
        for i in range(table.num_rows)
    }
    ok = "very deviated" not in verdicts.values() and "deviated" not in verdicts.values()
    return ok, f"null verdicts: {verdicts}"


def check_active_learning_ordering() -> Tuple[bool, str]:
    """Uncertainty must fall as the classes separate. Four queries whose
    separation is known by construction, from a dead tie to near-certainty."""
    from fastdna.active_learning import uncertainty_score

    cases = [[0.50, 0.50], [0.52, 0.48], [0.90, 0.10], [0.99, 0.01]]
    scores = [uncertainty_score((case, ["A", "B"])) for case in cases]
    monotone = all(scores[i] > scores[i + 1] for i in range(len(scores) - 1))
    return monotone, (
        "tie=%.3f -> near-certain=%.3f, monotone=%s" % (scores[0], scores[-1], monotone)
    )


def check_spectrum_valley() -> Tuple[bool, str]:
    """`suggest_min_count` claims to find the valley between the error peak
    and the coverage peak. Build spectra whose valley position is known by
    construction (a decaying error component plus a normal coverage peak at
    a chosen depth) and check it lands there."""
    import numpy as np

    from fastdna.spectrum import suggest_min_count

    hits = []
    for coverage in (10, 20, 30, 50, 80):
        depths = np.arange(1, coverage * 3)
        errors = 6_000_000 * np.exp(-depths / 0.8)
        peak = 2_000_000 * np.exp(-((depths - coverage) ** 2) / (2 * max(2.0, coverage / 4) ** 2))
        spectrum = {int(d): int(v) for d, v in zip(depths, (errors + peak).astype(int)) if v > 0}

        observed_peak = max((d for d in spectrum if d > 3), key=lambda d: spectrum[d])
        true_valley = min((d for d in spectrum if d < observed_peak), key=lambda d: spectrum[d])
        hits.append(suggest_min_count(spectrum) == true_valley)

    ok = all(hits)
    return ok, f"exact valley found in {sum(hits)}/{len(hits)} constructed spectra"


def check_embed_preserves_structure() -> Tuple[bool, str]:
    """An embedding that does not keep known lineages together is not
    preserving the structure it exists to show. Silhouette against the true
    lineage labels makes that checkable rather than eyeballed."""
    import numpy as np
    from sklearn.metrics import silhouette_score

    from fastdna.embed import embed_cohort

    rng = np.random.default_rng(21)
    bases = np.array(list("ACGT"))
    with tempfile.TemporaryDirectory() as tmp:
        directory = pathlib.Path(tmp)
        paths, truth = [], []
        for lineage in range(4):
            root = "".join(rng.choice(bases, size=4000))
            for member in range(6):
                seq = list(root)
                for pos in rng.choice(len(seq), size=25, replace=False):
                    seq[pos] = str(rng.choice(bases))
                paths.append(_write_fastq(directory, f"L{lineage}_{member}.fastq", "".join(seq)))
                truth.append(lineage)

        scores = {}
        for method in ("pcoa", "umap"):
            table = embed_cohort(paths, method=method, k=21, sketch_size=500)
            coords = np.column_stack(
                [np.asarray(table.column(c)) for c in table.column_names if c in ("x", "y")]
            )
            scores[method] = float(silhouette_score(coords, np.asarray(truth)))

    ok = all(v > 0.5 for v in scores.values())
    return ok, "silhouette vs true lineages: " + ", ".join(
        f"{m}={v:+.3f}" for m, v in scores.items()
    )


def check_equivalence_collapse() -> Tuple[bool, str]:
    """Columns with identical presence profiles are one equivalence class.
    Built from three independent features duplicated four times each, so the
    right answer is three."""
    import numpy as np
    import scipy.sparse as sp

    from fastdna.equivalence import collapse_equivalence_classes

    base = (np.random.default_rng(0).random((30, 3)) > 0.5).astype(np.uint8)
    matrix = np.hstack([np.repeat(base[:, [i]], 4, axis=1) for i in range(3)])
    names = [f"f{i}c{j}" for i in range(3) for j in range(4)]

    result = collapse_equivalence_classes(sp.csr_matrix(matrix), names)
    n_classes = len(result.representative)
    ok = n_classes == 3
    return ok, f"12 columns (3 features x 4 copies) -> {n_classes} classes"


def check_mic_essential_agreement() -> Tuple[bool, str]:
    """Essential agreement is a CLSI convention: a prediction within one
    two-fold dilution of the true MIC counts as agreeing. That gives an
    external, published definition to check against rather than an internal
    one."""
    import numpy as np

    from fastdna.mic import log2_mic, mic_regression_report

    truth = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32])
    exact = mic_regression_report(truth, truth).essential_agreement
    one = mic_regression_report(truth, truth * 2).essential_agreement
    two = mic_regression_report(truth, truth * 4).essential_agreement
    log2_ok = list(np.asarray(log2_mic(np.array([0.25, 1, 4, 16])))) == [-2.0, 0.0, 2.0, 4.0]

    ok = exact == 1.0 and one == 1.0 and two == 0.0 and log2_ok
    return ok, (f"EA exact={exact}, 1 dilution={one}, 2 dilutions={two}; "
                f"log2_mic exact={log2_ok}")


def check_multiomics_joins_by_id() -> Tuple[bool, str]:
    """Layers arrive in different orders with different sample sets. Joining
    by position instead of by id would silently pair the wrong rows, which is
    the failure this check exists to exclude."""
    pandas = __import__("pandas")

    from fastdna.multiomics import join_omics_layers

    kmers = pandas.DataFrame({"sample_id": ["S1", "S2", "S3"], "kmer": [1.0, 2.0, 3.0]})
    rna = pandas.DataFrame({"sample_id": ["S3", "S1", "S9"], "rna": [30.0, 10.0, 90.0]})

    frame, report = join_omics_layers({"kmers": kmers, "rna": rna})
    indexed = frame.set_index("sample_id")
    ok = (
        len(frame) == 2
        and indexed.loc["S1", "rna"] == 10.0   # would be 30.0 if joined by position
        and indexed.loc["S3", "rna"] == 30.0
        and set(report.dropped_sample_ids) == {"S2", "S9"}
    )
    return ok, f"kept {report.kept_sample_ids}, dropped {sorted(report.dropped_sample_ids)}"


def check_interop_biopython() -> Tuple[bool, str]:
    """Real Biopython SeqRecords, not a hand-written stand-in: the claim is
    interoperability with that specific library, so the check has to import
    it."""
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    from fastdna.interop import count_from_sequences

    sequence = "ACGTACGTTGCAACGTACGTACGTACGT" * 5
    records = [SeqRecord(Seq(sequence), id=f"r{i}") for i in range(3)]

    from_records = count_from_sequences(records, k=11)
    from_strings = count_from_sequences([sequence] * 3, k=11)

    ok = from_records.distinct_kmers == from_strings.distinct_kmers
    return ok, (f"SeqRecord -> {from_records.distinct_kmers} distinct, "
                f"plain strings -> {from_strings.distinct_kmers}")


CHECKS: Dict[str, Callable[[], Tuple[bool, str]]] = {
    "active_learning": check_active_learning_ordering,
    "anomaly": check_anomaly,
    "calibration": check_calibration,
    "embed": check_embed_preserves_structure,
    "equivalence": check_equivalence_collapse,
    "genomic_model": check_genomic_model_domain,
    "interop": check_interop_biopython,
    "mic": check_mic_essential_agreement,
    "multiomics": check_multiomics_joins_by_id,
    "spectrum": check_spectrum_valley,
    "validate_generated": check_generated_null,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=sorted(CHECKS), default=None)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    selected = args.only or sorted(CHECKS)

    failures: List[str] = []
    for name in selected:
        ok, detail = CHECKS[name]()
        print(f"  {'OK  ' if ok else 'FAIL'}  {name:<20} {detail}")
        if not ok:
            failures.append(name)

    if failures:
        print(f"\nFAIL: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\nOK: {len(selected)} module claims hold against constructed truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
