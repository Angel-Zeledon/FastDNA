"""Tests for `fastdna.report` -- bundling a classifier's rules, a
calibration curve and run metadata into one self-contained HTML file.

No browser is used: "valid enough" is checked with the standard-library
`html.parser.HTMLParser`, which raises on malformed markup, plus substring
assertions that the expected sections/data actually made it into the
document. The image is checked as a `data:image/png;base64,` URI without
decoding the PNG itself -- decoding would need Pillow, an extra dependency
this module does not otherwise need.
"""
from __future__ import annotations

from html.parser import HTMLParser

import pytest

pytest.importorskip("matplotlib")
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pa = pytest.importorskip("pyarrow")

from fastdna.evaluation import calibration_report
from fastdna.report import to_report
from fastdna.rules import Rule, SetCoveringClassifier


class _StrictHTMLParser(HTMLParser):
    """Raises on the specific malformed-markup shapes this module could
    plausibly produce (an unescaped '<'/'>'/'&' from a caller-supplied
    string leaking into the template): a bare `HTMLParser` already raises
    on outright broken tag syntax, so parsing without an exception is a
    real (if partial) validity check.
    """


def _assert_parses_as_html(text):
    parser = _StrictHTMLParser()
    parser.feed(text)
    parser.close()


@pytest.fixture
def fitted_classifier():
    X = np.array(
        [
            [1, 0],
            [1, 1],
            [0, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )
    y = np.array(["resistant", "resistant", "susceptible", "susceptible"])
    clf = SetCoveringClassifier()
    clf.fit(X, y, feature_names=["ACGTACGTACGT", "TTGCATTGCATT"])
    return clf


@pytest.fixture
def a_calibration_report():
    rng = np.random.default_rng(20260825)
    y_true = rng.integers(0, 2, size=60)
    # Genuinely varied probabilities (not the SCM's own degenerate 0/1
    # output) so the reliability diagram this produces is a real curve to
    # embed, not a two-point degenerate one -- test_calibration.py already
    # covers the degenerate-input warning path; this module's own tests
    # only need *some* real CalibrationReport to bundle.
    y_prob = np.clip(y_true * 0.6 + rng.normal(0.2, 0.15, size=60), 0.0, 1.0)
    return calibration_report(y_true, y_prob)


def test_to_report_writes_a_file_that_parses_as_html(tmp_path, fitted_classifier, a_calibration_report):
    out = tmp_path / "report.html"
    to_report(out, classifier=fitted_classifier, calibration=a_calibration_report)

    assert out.exists()
    text = out.read_text(encoding="utf-8")
    _assert_parses_as_html(text)
    assert text.startswith("<!doctype html>")


def test_report_contains_the_classifiers_rule_text(tmp_path, fitted_classifier, a_calibration_report):
    out = tmp_path / "report.html"
    to_report(out, classifier=fitted_classifier, calibration=a_calibration_report)

    text = out.read_text(encoding="utf-8")
    assert fitted_classifier.explain() in text
    # every learned rule's k-mer literal appears in the rendered rule table
    for rule in fitted_classifier.rules_:
        assert rule.feature_name in text


def test_report_embeds_a_self_contained_calibration_image(tmp_path, fitted_classifier, a_calibration_report):
    out = tmp_path / "report.html"
    to_report(out, classifier=fitted_classifier, calibration=a_calibration_report)

    text = out.read_text(encoding="utf-8")
    assert "data:image/png;base64," in text
    # no external asset reference of any kind -- offline-safe by construction
    assert "http://" not in text
    assert "https://" not in text
    assert "cdn." not in text.lower()
    assert f"{a_calibration_report.brier_score:.3f}" in text


def test_report_includes_venn_abers_interval_when_given(tmp_path, fitted_classifier, a_calibration_report):
    out_with = tmp_path / "with_interval.html"
    out_without = tmp_path / "without_interval.html"
    p0 = np.array([0.1, 0.3, 0.5])
    p1 = np.array([0.2, 0.5, 0.7])

    to_report(out_with, classifier=fitted_classifier, calibration=a_calibration_report, calibration_interval=(p0, p1))
    to_report(out_without, classifier=fitted_classifier, calibration=a_calibration_report)

    text_with = out_with.read_text(encoding="utf-8")
    text_without = out_without.read_text(encoding="utf-8")
    _assert_parses_as_html(text_with)
    # both embed a chart, but the one with an interval is a materially
    # different (larger) image -- a cheap proxy for "the error bars were
    # actually drawn" without decoding the PNG.
    assert "data:image/png;base64," in text_with
    assert len(text_with) != len(text_without)


def test_report_contains_run_metadata_and_an_auto_timestamp(tmp_path, fitted_classifier, a_calibration_report):
    out = tmp_path / "report.html"
    to_report(
        out,
        classifier=fitted_classifier,
        calibration=a_calibration_report,
        metadata={"n_samples": 42, "n_features_evaluated": 7},
    )

    text = out.read_text(encoding="utf-8")
    assert "42" in text
    assert "n_samples" in text
    assert "generated_at" in text  # auto-added even though metadata didn't include one


def test_report_with_no_calibration_data_says_so_instead_of_drawing_nothing(tmp_path, fitted_classifier):
    out = tmp_path / "report.html"
    to_report(out, classifier=fitted_classifier)

    text = out.read_text(encoding="utf-8")
    _assert_parses_as_html(text)
    assert "No calibration report or interval was provided" in text
    assert "data:image/png;base64," not in text


def test_report_with_no_classifier_says_so_instead_of_raising(tmp_path, a_calibration_report):
    out = tmp_path / "report.html"
    to_report(out, calibration=a_calibration_report)

    text = out.read_text(encoding="utf-8")
    _assert_parses_as_html(text)
    assert "No classifier was provided" in text


def test_report_escapes_untrusted_text_rather_than_injecting_it(tmp_path, a_calibration_report):
    # a classifier whose explain() contains characters that must be
    # HTML-escaped, not interpolated raw -- e.g. a feature name pulled
    # from an untrusted FASTA header.
    class _Fake:
        rules_ = [Rule(0, "<script>evil()</script>", True)]

        def explain(self):
            return "resistant IF present(<script>evil()</script>)"

    out = tmp_path / "report.html"
    to_report(out, classifier=_Fake(), calibration=a_calibration_report)

    text = out.read_text(encoding="utf-8")
    _assert_parses_as_html(text)
    assert "<script>evil()</script>" not in text
    assert "&lt;script&gt;" in text
