"""fastdna.report -- bundling a classifier's rules, a calibration curve and
run metadata into one self-contained, offline-safe HTML file.

## Why this exists

`fastdna.rules.SetCoveringClassifier.explain()`, `fastdna.evaluation.
calibration_report()` and `fastdna.calibration.calibrate()` each produce a
genuine, checkable piece of a single run's story, but each lives as a
Python object -- a string, a `CalibrationReport` NamedTuple, a `(p0, p1)`
interval -- with no single artifact to hand to someone who was not the one
who ran the pipeline. This module is that artifact: one HTML file bundling
(1) the rules a fitted classifier learned, in human-readable form, (2) a
calibration reliability diagram (with the Venn-ABERS multiprobability
interval, when one is given), and (3) descriptive run metadata (sample and
feature counts, when the report was generated -- **not** a speed
benchmark: this project's own discipline against timing this machine (see
`docs/BENCHMARKS.md`) applies here too, so no wall-clock or throughput
number is ever written into this template).

## What "self-contained" means here

The calibration figure is rendered with matplotlib to a PNG in memory and
embedded as a base64 `data:` URI directly in the HTML -- no external image
file, no CDN-hosted chart library, no network call of any kind. This is
deliberate: a clinical or laboratory environment generating this report may
be air-gapped, and a report that silently breaks (a blank chart, a 404)
the moment it is opened without internet access is worse than one that
never tried to be interactive. The tradeoff is a static image instead of a
zoomable/hoverable chart; that tradeoff is the point.

## What this module does not do

It does not compute anything -- no new statistic, no new plot style beyond
what `fastdna.evaluation.calibration_report`'s own reliability-diagram
convention already defines. It is a packaging layer over objects the
caller already has, the same role `fastdna.workflow.AssociationWorkflow`
plays one level up (and `AssociationResult.to_report()` is exactly this
module called with that result's own fields). It does not pick a "good
enough" calibration or rule count and present it as a verdict: the caveat
banner in the generated page says as much, and repeats which modules'
docstrings to read before treating a run's numbers as more than what they
are.

## Package convention

Imported explicitly (`from fastdna.report import to_report`); matplotlib
is imported lazily inside the function that renders the calibration
figure, matching `fastdna.plotting`'s convention -- neither is a runtime
dependency of `fastdna`.
"""

from __future__ import annotations

import base64
import datetime
import html
import io
import os
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

if TYPE_CHECKING:
    # `fastdna.evaluation` is not imported at module load time (this module
    # does not otherwise need it); this import only runs for static type
    # checkers, so `to_report`'s annotation can name the real class without
    # requiring an extra runtime import, matching `fastdna/__init__.py`'s own
    # `TYPE_CHECKING`-guarded `pandas`/`polars` imports.
    from .evaluation import CalibrationReport

__all__ = ["to_report"]


def _missing_dependency(package):
    return ImportError(
        f"fastdna.report requires the '{package}' package, which is not installed. "
        f"Install it with `pip install {package}` and try again."
    )


def _matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise _missing_dependency("matplotlib") from None
    return plt


def _render_calibration_figure(calibration, calibration_interval):
    """A base64-encoded PNG (no `data:` prefix) of the reliability diagram
    -- the diagonal reference line, `calibration.curve`'s observed points
    (when `calibration` is given), and the Venn-ABERS `[p0, p1]` interval
    as error bars around each point estimate `p1 / (1 - p0 + p1)` (when
    `calibration_interval` is given). Either argument may be `None`
    independently: a caller with only a `CalibratedEstimator.
    predict_interval()` result and no `CalibrationReport` (or vice versa)
    still gets a useful figure.
    """
    plt = _matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1, label="perfect calibration")

    title = "Calibration"
    if calibration is not None:
        curve = calibration.curve
        mean_predicted = curve.column("mean_predicted_probability").to_pylist()
        fraction_positive = curve.column("fraction_of_positives").to_pylist()
        ax.plot(mean_predicted, fraction_positive, marker="o", color="C0", label="observed")
        title = f"Calibration (Brier score = {calibration.brier_score:.3f})"

    if calibration_interval is not None:
        import numpy as np

        p0, p1 = calibration_interval
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        denom = 1.0 - p0 + p1
        point = np.divide(p1, denom, out=p1.copy(), where=denom > 0)
        order = np.argsort(point)
        ax.errorbar(
            point[order],
            point[order],
            yerr=[point[order] - p0[order], p1[order] - point[order]],
            fmt="none",
            ecolor="C1",
            elinewidth=1,
            capsize=3,
            label="Venn-ABERS interval [p0, p1]",
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction of positives")
    ax.set_title(title)
    ax.legend(fontsize="small", loc="upper left")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _rules_html(classifier):
    if classifier is None:
        return "<p>No classifier was provided.</p>"

    pieces = []
    explain = getattr(classifier, "explain", None)
    if callable(explain):
        try:
            pieces.append(f"<p class='explain'>{html.escape(str(explain()))}</p>")
        except Exception:  # pragma: no cover -- defensive: an unfitted/foreign estimator
            pass

    rules = getattr(classifier, "rules_", None)
    if rules:
        rows = "".join(
            "<tr><td>{n}</td><td>{polarity}</td><td><code>{kmer}</code></td></tr>".format(
                n=i + 1,
                polarity="present" if rule.presence else "absent",
                kmer=html.escape(str(rule.feature_name)),
            )
            for i, rule in enumerate(rules)
        )
        pieces.append(
            "<table class='rules'><thead><tr><th>#</th><th>polarity</th>"
            f"<th>k-mer</th></tr></thead><tbody>{rows}</tbody></table>"
        )

    if not pieces:
        pieces.append(
            f"<p>{html.escape(type(classifier).__name__)} exposes no <code>explain()</code> "
            "or <code>rules_</code> to display.</p>"
        )
    return "".join(pieces)


def _metadata_html(metadata):
    if not metadata:
        return "<p>No run metadata was provided.</p>"
    rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in metadata.items()
    )
    return f"<table class='metadata'><tbody>{rows}</tbody></table>"


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; max-width: 820px;
       margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.4; }}
h1 {{ font-size: 1.4rem; }}
h2 {{ font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: .25rem; }}
table {{ border-collapse: collapse; width: 100%; margin-top: .5rem; }}
th, td {{ text-align: left; padding: .3rem .6rem; border-bottom: 1px solid #eee; font-size: .9rem; }}
code {{ font-family: ui-monospace, Consolas, monospace; font-size: .85rem; }}
.explain {{ font-size: 1.05rem; background: #f6f6f6; padding: .6rem .8rem; border-radius: 6px; }}
img {{ max-width: 100%; }}
.caveat {{ color: #7a5b00; background: #fff8e1; padding: .5rem .8rem; border-radius: 6px;
           font-size: .85rem; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="caveat">Generated by <code>fastdna.report</code> -- this is a descriptive summary of
one run, not a validated diagnostic verdict. Read the docstrings of
<code>fastdna.rules</code>, <code>fastdna.evaluation</code> and
<code>fastdna.calibration</code> (overfitting risk, screening-vs-inference,
calibration-set independence) before treating anything below as more than
what it is.</p>

<h2>Run metadata</h2>
{metadata_html}

<h2>Rules</h2>
{rules_html}

<h2>Calibration</h2>
{calibration_html}
</body>
</html>
"""


def to_report(
    path: Union[str, os.PathLike],
    *,
    classifier: Optional[Any] = None,  # duck-typed fitted estimator: any object exposing explain()/rules_
    calibration: Optional["CalibrationReport"] = None,
    calibration_interval: Optional[tuple[Any, Any]] = None,  # (p0, p1), each array-like -- see docstring
    metadata: Optional[Mapping[str, Any]] = None,
    title: str = "FastDNA run report",
) -> None:
    """Writes a single self-contained HTML file at `path` bundling a
    classifier's rules, a calibration reliability diagram and descriptive
    run metadata.

    Parameters
    ----------
    path : path-like
        Output file path. Written as UTF-8 text with LF line endings.
    classifier : fitted estimator, optional
        Rendered via its `explain()` (if callable) and `rules_` (if
        present) -- e.g. a fitted `fastdna.rules.SetCoveringClassifier`.
        `None` (the default) renders a plain "no classifier was provided"
        note instead of raising, since a caller may want a report with
        only a calibration figure.
    calibration : fastdna.evaluation.CalibrationReport, optional
        Drawn as the reliability diagram's observed points, with its
        Brier score in the section title.
    calibration_interval : tuple of (p0, p1), optional
        The Venn-ABERS multiprobability interval from
        `fastdna.calibration.CalibratedEstimator.predict_interval()`,
        drawn as error bars around each point's estimate
        `p1 / (1 - p0 + p1)`. Independent of `calibration` -- either, both,
        or neither may be given; when both are `None` the calibration
        section notes that no calibration data was provided rather than
        drawing an empty axes.
    metadata : dict, optional
        Arbitrary `{label: value}` pairs rendered as a table -- e.g.
        sample/feature counts. A `"generated_at"` key is always added
        automatically (this call's own wall-clock timestamp, in ISO 8601)
        unless `metadata` already provides one; nothing here reports
        execution duration or throughput -- see the module docstring for
        why.
    title : str, default "FastDNA run report"
        The page's `<title>` and top-level heading.

    Returns
    -------
    None
    """
    metadata = dict(metadata) if metadata else {}
    metadata.setdefault("generated_at", datetime.datetime.now().isoformat(timespec="seconds"))

    if calibration is not None or calibration_interval is not None:
        image_b64 = _render_calibration_figure(calibration, calibration_interval)
        calibration_html = f"<img alt='calibration reliability diagram' src='data:image/png;base64,{image_b64}'>"
        if calibration is not None:
            # The Brier score is also drawn into the figure's own title, but
            # that text lives inside the embedded PNG's pixels -- invisible
            # to anyone (or anything) reading the page as text rather than
            # looking at the image. Repeating it here as real HTML text
            # keeps the number accessible without decoding the image.
            calibration_html += f"<p>Brier score: {calibration.brier_score:.3f} (lower is better; 0.0 is a perfect predictor).</p>"
    else:
        calibration_html = "<p>No calibration report or interval was provided.</p>"

    document = _TEMPLATE.format(
        title=html.escape(title),
        metadata_html=_metadata_html(metadata),
        rules_html=_rules_html(classifier),
        calibration_html=calibration_html,
    )

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(document)
