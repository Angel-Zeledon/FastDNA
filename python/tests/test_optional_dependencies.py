"""Dependency-posture tests: `import fastdna` and the core API must work
with *none* of the optional packages installed.

Every module added in the recent batch (`sklearn`, `interpret`, `anomaly`,
`embed`, `multiomics`, `interop`, `cv`) declares its heavy dependencies as
test-extra / opt-in rather than runtime. `pyproject.toml`'s
`project.dependencies` is exactly `["pyarrow>=14"]`, so a plain
`pip install fastdna` gets pyarrow and nothing else -- which is also what
`.github/workflows/wheels.yml` installs before running this suite
(`pip install dist/*.whl pytest`, no `[test]` extra).

Those claims are made in six separate module docstrings and, until this
file, were never actually exercised: every other test module in this
directory `importorskip`s its dependency, so a regression that made
`import fastdna` require numpy would skip rather than fail.

Each test runs in a **subprocess** with a `sys.meta_path` blocker installed
ahead of the real finders. A blocker installed in-process would leak into
every later test in the same session (import caches, already-bound module
references), so isolation is bought with a process boundary rather than
with monkeypatch teardown.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# The packages a bare `pip install fastdna` does NOT bring in. pyarrow is
# deliberately absent from this list: it is the one genuine runtime
# dependency.
_OPTIONAL_PACKAGES = (
    "sklearn",
    "scipy",
    "numpy",
    "pandas",
    "polars",
    "duckdb",
    "shap",
    "umap",
    "Bio",
    "numba",
    "llvmlite",
    "tqdm",
)

_PYTHON_SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_FASTQ = _REPO_ROOT / "test.fastq"


_BLOCKER_PREAMBLE = """
import sys

_BLOCKED = {blocked!r}


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in _BLOCKED:
            # ModuleNotFoundError, not the bare ImportError superclass: a
            # genuinely-not-installed package is exactly what Python's own
            # import machinery raises, and it is also what pytest 9.1+'s
            # `pytest.importorskip` catches by default (`exc_type` there
            # defaults to `ModuleNotFoundError`, not `ImportError` -- see
            # pytest's own changelog for 9.1). Raising the broader
            # `ImportError` here would make every `importorskip("sklearn")`
            # etc. in this test suite fail to skip and instead let the
            # blocked ImportError propagate as a collection error -- the
            # opposite of what this blocker exists to simulate.
            raise ModuleNotFoundError("No module named " + repr(root) + " (blocked)")
        return None


sys.meta_path.insert(0, _Blocker())
for _name in list(sys.modules):
    if _name.split(".")[0] in _BLOCKED:
        del sys.modules[_name]

sys.path.insert(0, {source_root!r})
"""


def _run_without(blocked, body):
    """Runs `body` in a subprocess where importing any package in `blocked`
    raises ImportError. `body` must print a single JSON object on its last
    line; that object is returned.
    """
    script = _BLOCKER_PREAMBLE.format(
        blocked=sorted(blocked), source_root=_PYTHON_SOURCE_ROOT
    ) + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"subprocess failed (exit {proc.returncode}).\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    last_line = [line for line in proc.stdout.splitlines() if line.strip()][-1]
    return json.loads(last_line)


def test_import_fastdna_works_with_no_optional_packages_installed():
    """The headline claim: `import fastdna` must not require numpy,
    pandas, scikit-learn, scipy, polars, duckdb, shap, umap-learn or
    Biopython. Only pyarrow.
    """
    result = _run_without(
        _OPTIONAL_PACKAGES,
        """
        import json
        import fastdna

        print(json.dumps({"version": fastdna.__version__}))
        """,
    )
    assert isinstance(result["version"], str)
    assert result["version"]


@pytest.mark.skipif(not _SAMPLE_FASTQ.exists(), reason="repo sample FASTQ not present")
def test_core_api_still_functions_with_no_optional_packages_installed():
    """Importing is not enough -- the core API (`count`, its Arrow table,
    `sketch`, `compare_all`, `peek`, and the notebook `_repr_html_`
    fallback) must actually *work* with only pyarrow available. In
    particular `KmerCounts._repr_html_` claims to fall back to a hand-built
    HTML table when pandas is missing; nothing exercised that in an
    environment where pandas is genuinely absent (the existing
    `test_repr.py` fakes it with monkeypatch).
    """
    result = _run_without(
        _OPTIONAL_PACKAGES,
        f"""
        import json
        import fastdna

        path = {str(_SAMPLE_FASTQ)!r}
        counts = fastdna.count(path, k=5)
        sketch = fastdna.sketch(path, k=5, sketch_size=50)
        pairs = fastdna.compare_all([path, path], k=5, sketch_size=50)

        print(json.dumps({{
            "columns": counts.table.column_names,
            "distinct": counts.distinct_kmers,
            "total": counts.total_kmers,
            "html_has_table": "<table>" in counts._repr_html_(),
            "sketch_html_has_sketch": "Sketch" in sketch._repr_html_(),
            "self_jaccard": sketch.jaccard(sketch),
            "compare_all_rows": pairs.num_rows,
            "peek_ok": fastdna.peek(path) is not None,
        }}))
        """,
    )
    assert result["columns"] == ["kmer_u64", "kmer_sequence", "frequency"]
    assert result["distinct"] > 0
    assert result["total"] > 0
    assert result["html_has_table"], "_repr_html_ must fall back to hand-built HTML without pandas"
    assert result["sketch_html_has_sketch"]
    assert result["self_jaccard"] == pytest.approx(1.0)
    assert result["compare_all_rows"] == 1
    assert result["peek_ok"]


def test_pure_python_submodules_import_with_no_optional_packages():
    """The submodules that genuinely have no third-party dependency beyond
    pyarrow must import in a bare environment. `embed`, `anomaly` and
    `sklearn` are deliberately excluded: they hard-import numpy (and
    scipy/scikit-learn) at module scope, which is a documented, opt-in
    cost -- see `test_module_level_numpy_dependency_is_confined_to_three_modules`.
    """
    result = _run_without(
        _OPTIONAL_PACKAGES,
        """
        import json
        import importlib

        outcome = {}
        for name in ("fastdna.interop", "fastdna.interpret", "fastdna.multiomics",
                     "fastdna.spectrum", "fastdna.taxonomy", "fastdna.active_learning",
                     "fastdna.assembly_qc"):
            try:
                importlib.import_module(name)
                outcome[name] = "ok"
            except Exception as exc:
                outcome[name] = type(exc).__name__ + ": " + str(exc)
        print(json.dumps(outcome))
        """,
    )
    for name, status in result.items():
        assert status == "ok", f"{name} failed to import without optional packages: {status}"


def test_module_level_numpy_dependency_is_confined_to_the_numpy_native_modules():
    """Characterisation test, not an endorsement: `fastdna.embed`,
    `fastdna.anomaly`, `fastdna.sklearn` and `fastdna.cv` hard-import numpy
    at module scope, so `from fastdna.embed import embed_cohort` raises on a
    bare install even though `embed.py`'s docstring advertises that its
    dependencies are lazily imported.

    All four are modules whose *entire* surface is numeric -- there is no
    useful call in any of them that does not already produce or consume a
    numpy array -- so deferring the import would buy an importable module
    with no callable function. The heavier dependencies each of them layers
    on top (scikit-learn, scipy, umap-learn) *are* lazily imported, and the
    tests below check that those raise actionable errors naming the missing
    package.

    Pinned here so that if anyone *fixes* this (by deferring numpy) or
    *worsens* it (by adding numpy to a module that does not need it), this
    test says so rather than the fact going unnoticed.
    """
    result = _run_without(
        _OPTIONAL_PACKAGES,
        """
        import json
        import importlib

        failing = []
        for name in ("fastdna.embed", "fastdna.anomaly", "fastdna.sklearn", "fastdna.cv"):
            try:
                importlib.import_module(name)
            except ImportError:
                failing.append(name)
        print(json.dumps(failing))
        """,
    )
    assert result == ["fastdna.embed", "fastdna.anomaly", "fastdna.sklearn", "fastdna.cv"]


def test_embed_names_its_missing_optional_package_actionably():
    """`embed_cohort`'s ImportError must name the pip package and the
    install command, not surface a raw `ModuleNotFoundError` from inside
    sklearn/umap. This is the standard the other modules are measured
    against below.
    """
    result = _run_without(
        ("umap", "sklearn", "shap", "numba", "llvmlite"),
        """
        import json
        import numpy as np
        from fastdna.embed import _reduce

        messages = {}
        for method in ("umap", "tsne", "pcoa"):
            try:
                _reduce(np.zeros((3, 3)), method, 2, {})
            except ImportError as exc:
                messages[method] = str(exc)
        print(json.dumps(messages))
        """,
    )
    assert "umap-learn" in result["umap"] and "pip install" in result["umap"]
    assert "scikit-learn" in result["tsne"] and "pip install" in result["tsne"]
    assert "scikit-learn" in result["pcoa"] and "pip install" in result["pcoa"]


def test_cv_names_its_missing_optional_packages_actionably():
    """`fastdna.cv` layers scipy (single-linkage clustering) and
    scikit-learn (GroupKFold, clone) on top of numpy. Both are imported
    lazily inside the functions that need them, so each must name its own
    package and install command rather than surfacing a raw
    `ModuleNotFoundError` from inside scipy/sklearn.

    Neither call below reads a FASTQ file: `lineage_groups` reaches its
    scipy import before any sketching, and `LineageKFold(groups=...)` never
    sketches at all -- so this stays a pure dependency-posture test.
    """
    result = _run_without(
        ("scipy", "sklearn", "shap", "umap", "numba", "llvmlite"),
        """
        import json
        from fastdna import cv

        messages = {}
        try:
            cv.lineage_groups(["a.fastq", "b.fastq"])
        except ImportError as exc:
            messages["scipy"] = str(exc)
        try:
            list(cv.LineageKFold(n_splits=2, groups=[0, 0, 1, 1]).split([0, 1, 2, 3]))
        except ImportError as exc:
            messages["sklearn"] = str(exc)
        print(json.dumps(messages))
        """,
    )
    assert "scipy" in result["scipy"] and "pip install" in result["scipy"]
    assert "scikit-learn" in result["sklearn"] and "pip install" in result["sklearn"]


def test_anomaly_names_scikit_learn_actionably_for_the_isolation_forest_path():
    """`fastdna.anomaly`'s default `method="robust_zscore"` needs nothing
    beyond numpy; only `flag_cohort(..., method="isolation_forest")` reaches
    for scikit-learn, and it must say so by name. The import is attempted
    before any sketching, so the nonexistent paths below never get read.
    """
    result = _run_without(
        ("sklearn", "shap", "umap", "numba", "llvmlite"),
        """
        import json
        from fastdna.anomaly import _isolation_forest

        try:
            _isolation_forest({})
            message = None
        except ImportError as exc:
            message = str(exc)
        print(json.dumps({"message": message}))
        """,
    )
    assert result["message"] is not None
    assert "scikit-learn" in result["message"] and "pip install" in result["message"]


def test_interpret_names_shap_actionably_when_it_is_missing():
    result = _run_without(
        ("shap", "numba", "llvmlite"),
        """
        import json
        from fastdna.interpret import explain_with_shap

        try:
            explain_with_shap(None, None, [])
            message = None
        except ImportError as exc:
            message = str(exc)
        print(json.dumps({"message": message}))
        """,
    )
    assert result["message"] is not None
    assert "shap" in result["message"]
    assert "pip install" in result["message"]


def test_the_test_suite_itself_collects_in_the_environment_ci_builds():
    """EXPECTED TO FAIL -- pins a reported defect.

    `.github/workflows/wheels.yml` installs exactly
    `pip install dist/*.whl pytest` and then runs `pytest python/tests -v`.
    That environment has pyarrow (a declared runtime dependency) and
    nothing else -- no `[test]` extra, so no numpy, pandas, scipy,
    scikit-learn, polars, duckdb, shap, umap-learn or Biopython.

    Five test modules import an optional package at module scope without
    guarding it, so pytest raises during *collection* rather than skipping:

      - `test_anomaly.py:12`     `import numpy as np` (the
        `importorskip("sklearn")` on line 15 comes after it, and never
        covers numpy)
      - `test_sklearn.py:15`     `import numpy as np` (same ordering; the
        importorskips are on lines 18-19)
      - `test_embed.py:10`       `import numpy as np`, unguarded
      - `test_interpret.py:36`   `import numpy` / `import scipy.sparse`,
        unguarded
      - `test_multiomics.py:10`  `import pandas as pd`, unguarded

    pytest treats collection errors as fatal (`Interrupted: N errors during
    collection`), so the wheel job aborts before running a single test.
    The "202 passing tests" figure is reachable only in a development
    environment that installed `.[test]`, which CI never does.

    The fix is one of: move each `import numpy`/`pandas`/`scipy` below a
    matching `pytest.importorskip`, or add the packages to whatever CI
    installs. Both are outside this audit's remit.
    """
    script = _BLOCKER_PREAMBLE.format(
        blocked=sorted(_OPTIONAL_PACKAGES), source_root=_PYTHON_SOURCE_ROOT
    ) + textwrap.dedent(
        f"""
        import json
        import pytest

        code = pytest.main(
            ["--collect-only", "-q", "--no-header", "-p", "no:cacheprovider",
             {str(Path(__file__).parent)!r}]
        )
        print(json.dumps({{"exit_code": int(code)}}))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    errored = [
        line for line in proc.stdout.splitlines() if line.startswith("ERROR python")
        or line.startswith("ERROR ") and line.endswith(".py")
    ]
    assert "errors during collection" not in proc.stdout, (
        "pytest aborted during collection in the bare-wheel environment CI builds.\n"
        "Modules that failed to import: "
        + ", ".join(errored)
        + "\n--- tail of collection output ---\n"
        + "\n".join(proc.stdout.splitlines()[-25:])
    )
