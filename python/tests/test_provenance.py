"""Tests for `fastdna.provenance`: the reproducible provenance block.

Three claims carry this module, and each has tests that would fail loudly
if it stopped being true:

1. **Determinism.** Two captures of the same run must produce the same
   block apart from the timestamp -- otherwise "compare the provenance of
   the published run against yours" is not a usable instruction. Covered
   from several angles below (parameter key order, sets, opaque objects
   whose default `repr` embeds a memory address).
2. **The input digest's guarantee is exactly what the docstring says it
   is** -- no more (`test_sampled_digest_is_blind_between_its_windows`
   deliberately demonstrates the documented blind spot rather than hiding
   it) and no less (`test_full_digest_equals_sha256sum` pins that the
   `"sha256"` mode is checkable with a standard command-line tool).
3. **It stays importable and usable in a bare environment.** Nothing in
   this module may need numpy, scipy, scikit-learn or pandas -- verified
   in a subprocess with those genuinely blocked, not by inspection.

Deliberately no module-scope `import numpy`/`pandas`/`sklearn`: see
`test_optional_dependencies.py::
test_the_test_suite_itself_collects_in_the_environment_ci_builds`, which
fails if any test module here needs an optional package merely to be
collected. The one test that uses numpy calls `pytest.importorskip` inside
its own body.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fastdna import provenance
from fastdna.provenance import InputFile, Provenance, capture

_PYTHON_SOURCE_ROOT = str(Path(__file__).resolve().parents[1])

_ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _write_bytes(tmp_path, name, payload):
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


# ---------------------------------------------------------------------------
# Shape: what a block actually contains.
# ---------------------------------------------------------------------------


def test_capture_with_no_arguments_records_the_environment():
    prov = capture()

    assert isinstance(prov, Provenance)
    assert isinstance(prov.fastdna_version, str) and prov.fastdna_version
    assert prov.python_version.count(".") >= 1
    assert prov.python_implementation
    assert prov.platform
    assert prov.machine
    assert prov.cpu_count is None or prov.cpu_count >= 1
    assert prov.inputs == ()
    assert dict(prov.parameters) == {}


def test_the_four_tracked_distributions_are_always_reported_present_or_not():
    """The version gathering must be defensive: every tracked distribution
    gets a key, whether or not it is installed. A test that asserted
    `scipy is not None` would be asserting something about the machine that
    happens to be running it, not about this module.
    """
    versions = dict(capture().package_versions)

    assert set(versions) == {"numpy", "pyarrow", "scipy", "scikit-learn"}
    for name, version in versions.items():
        assert version is None or (isinstance(version, str) and version), (name, version)
    # pyarrow is the one genuine runtime dependency of this package
    # (pyproject.toml: `dependencies = ["pyarrow>=14"]`), so it is the only
    # entry it is safe to require anywhere the suite runs at all.
    assert versions["pyarrow"] is not None


def test_an_uninstallable_distribution_is_reported_as_none_not_raised(monkeypatch):
    """The absent-package path, exercised without depending on which
    packages the running machine happens to lack.
    """
    monkeypatch.setattr(
        provenance,
        "_TRACKED_PACKAGES",
        (("pyarrow", "pyarrow"), ("no_such_module_xyzzy", "no-such-distribution-xyzzy")),
    )
    versions = dict(capture().package_versions)

    assert versions["no-such-distribution-xyzzy"] is None
    assert versions["pyarrow"] is not None


def test_only_the_allowlisted_environment_variables_are_recorded(monkeypatch):
    """A block meant to be pasted into a paper must never carry a dump of
    `os.environ` -- that is how credentials get published.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("FASTDNA_TOTALLY_SECRET_TOKEN", "hunter2")

    environment = dict(capture().environment)

    assert environment["OMP_NUM_THREADS"] == "3"
    assert "FASTDNA_TOTALLY_SECRET_TOKEN" not in environment
    assert set(environment) == set(provenance._TRACKED_ENVIRONMENT)
    assert "hunter2" not in capture().to_json()


def test_an_unset_environment_variable_is_none_not_absent(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    assert dict(capture().environment)["OMP_NUM_THREADS"] is None


def test_timestamp_is_iso_8601_utc_to_the_second():
    import datetime

    prov = capture()

    assert _ISO_UTC.match(prov.generated_utc), prov.generated_utc
    parsed = datetime.datetime.strptime(prov.generated_utc, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    assert abs((now - parsed).total_seconds()) < 300


# ---------------------------------------------------------------------------
# Determinism -- the property the whole module rests on.
# ---------------------------------------------------------------------------


def test_two_captures_of_the_same_run_differ_only_in_the_timestamp(tmp_path):
    path = _write_bytes(tmp_path, "sample.fastq", b"@r0\nACGT\n+\nIIII\n")
    parameters = {"k": 31, "n_splits": 5, "random_state": 0, "scoring": "roc_auc"}

    first = capture(inputs=[path], parameters=parameters)
    second = capture(inputs=[path], parameters=parameters)

    assert first.matches(second)

    left, right = first.to_dict(), second.to_dict()
    differing = {key for key in left if left[key] != right.get(key)}
    assert differing <= {"generated_utc"}


def test_parameter_key_order_does_not_change_the_block():
    assert capture(parameters={"k": 31, "seed": 0}).matches(capture(parameters={"seed": 0, "k": 31}))


def test_a_set_valued_parameter_serializes_in_a_stable_order():
    """Set iteration order depends on `PYTHONHASHSEED` for str members, so
    a set passed as a parameter would otherwise be a silent source of
    non-determinism between processes.
    """
    prov = capture(parameters={"metrics": {"roc_auc", "accuracy", "f1"}})

    assert prov.parameters["metrics"] == sorted(["roc_auc", "accuracy", "f1"], key=repr)
    assert prov.matches(capture(parameters={"metrics": {"f1", "accuracy", "roc_auc"}}))


def test_an_opaque_parameter_object_does_not_leak_its_memory_address():
    """`repr()` of an object relying on CPython's default `object.__repr__`
    embeds a memory address, which differs between two otherwise-identical
    runs. Recorded as a stable `<module.QualName>` instead.
    """

    class Opaque:
        pass

    first = capture(parameters={"vectorizer": Opaque()})
    second = capture(parameters={"vectorizer": Opaque()})

    recorded = first.parameters["vectorizer"]
    assert "0x" not in recorded
    assert recorded.endswith("Opaque>")
    assert first.matches(second)


def test_a_parameter_with_an_informative_repr_keeps_it():
    """The opaque-object rule must not swallow a `repr` that is exactly
    what belongs in a methods section (a scikit-learn estimator's is its
    constructor call).
    """

    class Estimator:
        def __repr__(self):
            return "Estimator(alpha=0.5)"

    assert capture(parameters={"est": Estimator()}).parameters["est"] == "Estimator(alpha=0.5)"


def test_matches_is_false_when_a_parameter_differs():
    assert not capture(parameters={"k": 31}).matches(capture(parameters={"k": 21}))


def test_matches_rejects_a_non_provenance_argument():
    with pytest.raises(TypeError, match="two Provenance blocks"):
        capture().matches({"fastdna_version": "0.2.0"})


def test_markdown_is_identical_between_two_captures_apart_from_the_timestamp(tmp_path):
    path = _write_bytes(tmp_path, "a.fastq", b"@r0\nACGT\n+\nIIII\n")

    def without_the_timestamp_line(text):
        return [line for line in text.splitlines() if not line.startswith("Generated ")]

    first = capture(inputs=[path], parameters={"k": 31}).to_markdown()
    second = capture(inputs=[path], parameters={"k": 31}).to_markdown()

    assert without_the_timestamp_line(first) == without_the_timestamp_line(second)


# ---------------------------------------------------------------------------
# Input identity: exactly the guarantee the docstring claims, no more.
# ---------------------------------------------------------------------------


def test_full_digest_equals_sha256sum(tmp_path):
    """`input_identity="sha256"` must be plain, unframed SHA-256, so a
    reviewer with no FastDNA installed can check it with `sha256sum`.
    """
    payload = b"@r0\n" + b"ACGT" * 500 + b"\n+\n" + b"I" * 2000 + b"\n"
    path = _write_bytes(tmp_path, "big.fastq", payload)

    entry = capture(inputs=[path], input_identity="sha256").inputs[0]

    assert entry.digest_method == "sha256"
    assert entry.digest == hashlib.sha256(payload).hexdigest()
    assert entry.size_bytes == len(payload)


def test_a_file_smaller_than_the_windows_is_hashed_whole_even_in_sampled_mode(tmp_path):
    """Reading four 1 MiB windows out of a 20-byte file reads the whole
    file anyway; reporting it as a weaker "sampled" identity would give up
    a guarantee for nothing. Each InputFile therefore states its own
    method, and a block can honestly mix the two.
    """
    payload = b"@r0\nACGT\n+\nIIII\n"
    path = _write_bytes(tmp_path, "tiny.fastq", payload)

    entry = capture(inputs=[path]).inputs[0]

    assert entry.digest_method == "sha256"
    assert entry.digest == hashlib.sha256(payload).hexdigest()


def test_sampled_digest_is_framed_so_it_can_never_be_read_as_a_full_sha256(tmp_path):
    payload = b"A" * 1000
    path = _write_bytes(tmp_path, "reads.fastq", payload)

    entry = capture(inputs=[path], chunk_bytes=16, sampled_chunks=2).inputs[0]

    assert entry.digest_method == "sha256-sampled"
    assert entry.digest != hashlib.sha256(payload).hexdigest()


def test_sampled_digest_detects_a_change_inside_a_window(tmp_path):
    """Offsets 0 and 984 with `chunk_bytes=16`, so bytes 0-15 and 984-999
    are read. A change at byte 5 is inside the first window.
    """
    original = bytearray(b"A" * 1000)
    edited = bytearray(original)
    edited[5:6] = b"C"

    a = capture(inputs=[_write_bytes(tmp_path, "a", bytes(original))], chunk_bytes=16, sampled_chunks=2)
    b = capture(inputs=[_write_bytes(tmp_path, "b", bytes(edited))], chunk_bytes=16, sampled_chunks=2)

    assert a.inputs[0].digest != b.inputs[0].digest


def test_sampled_digest_is_blind_between_its_windows(tmp_path):
    """THE DOCUMENTED LIMITATION, pinned rather than hidden.

    A byte changed at offset 500 falls between the two sampled windows
    (0-15 and 984-999), so the sampled digest cannot see it. This is
    exactly what `fastdna.provenance`'s docstring says -- a change
    *detector*, not a commitment to the file's content -- and the rendered
    block repeats it in prose next to every `sha256-sampled` identity. If
    this test ever starts failing because the scheme was quietly
    strengthened, the docstring's claim about the blind spot has to change
    with it; if it fails because the scheme was weakened, worse.

    `input_identity="sha256"` is the mode with no blind spot, asserted
    here alongside so the difference between the two is visible in one
    place.
    """
    original = bytearray(b"A" * 1000)
    edited = bytearray(original)
    edited[500:501] = b"C"

    path_a = _write_bytes(tmp_path, "a", bytes(original))
    path_b = _write_bytes(tmp_path, "b", bytes(edited))

    sampled_a = capture(inputs=[path_a], chunk_bytes=16, sampled_chunks=2).inputs[0]
    sampled_b = capture(inputs=[path_b], chunk_bytes=16, sampled_chunks=2).inputs[0]
    assert sampled_a.digest == sampled_b.digest, "the documented blind spot"

    full_a = capture(inputs=[path_a], input_identity="sha256").inputs[0]
    full_b = capture(inputs=[path_b], input_identity="sha256").inputs[0]
    assert full_a.digest != full_b.digest, "the full-content mode must see it"


def test_sampled_digest_detects_truncation_and_appending(tmp_path):
    """The first and last windows are anchored at the file's two ends, and
    the size goes into the framing header -- so the two ways a FASTQ most
    often changes by accident are both caught.
    """
    base = b"A" * 1000
    reference = capture(inputs=[_write_bytes(tmp_path, "ref", base)], chunk_bytes=16, sampled_chunks=2)
    truncated = capture(inputs=[_write_bytes(tmp_path, "cut", base[:900])], chunk_bytes=16, sampled_chunks=2)
    appended = capture(inputs=[_write_bytes(tmp_path, "app", base + b"T" * 50)], chunk_bytes=16, sampled_chunks=2)

    digests = {reference.inputs[0].digest, truncated.inputs[0].digest, appended.inputs[0].digest}
    assert len(digests) == 3


def test_changing_the_window_settings_changes_every_digest(tmp_path):
    """The window geometry is part of the hashed framing header, so two
    blocks taken with different settings cannot produce equal digests that
    would imply a match neither one supports.
    """
    path = _write_bytes(tmp_path, "reads.fastq", b"A" * 1000)

    two = capture(inputs=[path], chunk_bytes=16, sampled_chunks=2).inputs[0].digest
    three = capture(inputs=[path], chunk_bytes=16, sampled_chunks=3).inputs[0].digest
    wider = capture(inputs=[path], chunk_bytes=32, sampled_chunks=2).inputs[0].digest

    assert len({two, three, wider}) == 3


def test_identity_is_content_addressed_not_path_addressed(tmp_path):
    """The point of a digest over a path: two copies of the same bytes at
    different paths have the same identity, and two different files at
    similar paths do not.
    """
    payload = b"@r0\nACGTACGT\n+\nIIIIIIII\n"
    same_a = capture(inputs=[_write_bytes(tmp_path, "cohort_01.fastq", payload)]).inputs[0]
    same_b = capture(inputs=[_write_bytes(tmp_path, "cohort_02.fastq", payload)]).inputs[0]
    other = capture(inputs=[_write_bytes(tmp_path, "cohort_03.fastq", payload + b"@r1\nTTTT\n+\nIIII\n")]).inputs[0]

    assert same_a.digest == same_b.digest
    assert same_a.path != same_b.path
    assert other.digest != same_a.digest


def test_mtime_is_reported_but_is_not_part_of_the_digest(tmp_path):
    """Including mtime in the identity would make the *same* data hash
    differently after a `touch` or a copy to another machine -- the
    opposite of what a content-addressed identity is for. It is reported
    as advisory context only.
    """
    path = _write_bytes(tmp_path, "reads.fastq", b"@r0\nACGT\n+\nIIII\n")
    before = capture(inputs=[path]).inputs[0]

    os.utime(path, (0, 0))
    after = capture(inputs=[path]).inputs[0]

    assert after.digest == before.digest
    assert after.modified_utc != before.modified_utc
    assert _ISO_UTC.match(after.modified_utc), after.modified_utc


def test_an_empty_file_is_recorded_without_special_casing(tmp_path):
    entry = capture(inputs=[_write_bytes(tmp_path, "empty.fastq", b"")]).inputs[0]

    assert entry.size_bytes == 0
    assert entry.digest == hashlib.sha256(b"").hexdigest()
    assert entry.digest_method == "sha256"


def test_input_order_is_preserved(tmp_path):
    paths = [_write_bytes(tmp_path, "s{}.fastq".format(i), b"ACGT" * (i + 1)) for i in range(4)]

    recorded = [entry.path for entry in capture(inputs=reversed(paths)).inputs]

    assert recorded == list(reversed(paths))


def test_the_recorded_path_is_the_one_that_was_passed(tmp_path):
    """Not resolved to an absolute path: that would bake one machine's
    directory layout (often including a username) into a document meant to
    be published.
    """
    path = tmp_path / "reads.fastq"
    path.write_bytes(b"ACGT")

    assert capture(inputs=[path]).inputs[0].path == os.fspath(path)


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------


def test_a_missing_input_raises_rather_than_recording_a_placeholder(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        capture(inputs=[str(tmp_path / "not-here.fastq")])


def test_a_directory_input_is_rejected_with_an_actionable_message(tmp_path):
    with pytest.raises(ValueError, match="not a regular file"):
        capture(inputs=[str(tmp_path)])


def test_an_unknown_input_identity_names_both_accepted_modes():
    with pytest.raises(ValueError, match="sampled"):
        capture(input_identity="md5")


@pytest.mark.parametrize(
    "kwargs",
    [{"chunk_bytes": 0}, {"chunk_bytes": -1}, {"sampled_chunks": 0}, {"chunk_bytes": True}],
)
def test_window_settings_must_be_positive_ints(kwargs):
    with pytest.raises(ValueError):
        capture(**kwargs)


def test_a_single_window_is_allowed_and_does_not_divide_by_zero(tmp_path):
    path = _write_bytes(tmp_path, "reads.fastq", b"A" * 1000)

    entry = capture(inputs=[path], chunk_bytes=16, sampled_chunks=1).inputs[0]

    assert entry.digest_method == "sha256-sampled"


# ---------------------------------------------------------------------------
# Immutability: a block is evidence.
# ---------------------------------------------------------------------------


def test_the_block_cannot_be_rebound_after_capture():
    prov = capture(parameters={"k": 31})

    with pytest.raises(dataclasses.FrozenInstanceError):
        prov.fastdna_version = "99.0.0"
    with pytest.raises(dataclasses.FrozenInstanceError):
        prov.parameters = {}
    with pytest.raises(dataclasses.FrozenInstanceError):
        prov.inputs = ()


def test_the_mapping_fields_cannot_be_mutated_in_place():
    """`frozen=True` only stops rebinding a field; a plain dict field would
    still be editable in place, which for evidence is the same hole with an
    extra step.
    """
    prov = capture(parameters={"k": 31})

    for mapping in (prov.parameters, prov.environment, prov.package_versions, prov.fastdna_build):
        with pytest.raises(TypeError):
            mapping["injected"] = "value"


# ---------------------------------------------------------------------------
# Serialization and rendering.
# ---------------------------------------------------------------------------


def test_to_dict_is_json_serializable_and_carries_a_schema_version(tmp_path):
    path = _write_bytes(tmp_path, "reads.fastq", b"@r0\nACGT\n+\nIIII\n")

    prov = capture(inputs=[path], parameters={"k": 31, "estimator": None, "thresholds": [0.1, 0.5]})
    restored = json.loads(prov.to_json())

    assert restored["provenance_schema"] == provenance.SCHEMA_VERSION
    assert restored["parameters"]["thresholds"] == [0.1, 0.5]
    assert restored["inputs"][0]["digest_method"] == "sha256"
    assert restored["generated_utc"] == prov.generated_utc
    # Compact form must also be valid JSON, for a one-line sidecar record.
    assert json.loads(prov.to_json(indent=None)) == restored


def test_to_dict_keys_are_sorted_so_stored_json_is_byte_stable():
    prov = capture(parameters={"z": 1, "a": 2, "m": 3})
    stored = prov.to_dict()

    assert list(stored["parameters"]) == ["a", "m", "z"]
    assert list(stored["package_versions"]) == sorted(stored["package_versions"])
    assert list(stored["environment"]) == sorted(stored["environment"])


def test_a_numpy_scalar_parameter_is_reduced_to_a_plain_number():
    np = pytest.importorskip("numpy")

    prov = capture(parameters={"gap": np.float64(0.23), "folds": np.int64(5)})

    # `type(...) is`, not `isinstance`: numpy.float64 *subclasses* float, so
    # isinstance would pass on an unconverted numpy scalar and this test
    # would not be testing the conversion at all.
    assert type(prov.parameters["gap"]) is float
    assert type(prov.parameters["folds"]) is int
    assert prov.parameters["gap"] == pytest.approx(0.23)
    json.dumps(prov.to_dict())  # must not raise


def test_markdown_reports_every_recorded_fact(tmp_path):
    path = _write_bytes(tmp_path, "reads.fastq", b"@r0\nACGT\n+\nIIII\n")
    prov = capture(inputs=[path], parameters={"k": 31})
    text = prov.to_markdown()

    assert text.startswith("# Provenance")
    assert prov.generated_utc in text
    assert prov.fastdna_version in text
    assert prov.inputs[0].digest in text
    assert "| k | 31 |" in text
    assert "pyarrow" in text
    assert str(prov) == text


def test_markdown_states_what_a_sampled_digest_does_not_prove(tmp_path):
    path = _write_bytes(tmp_path, "reads.fastq", b"A" * 1000)

    text = capture(inputs=[path], chunk_bytes=16, sampled_chunks=2).to_markdown()

    assert "sha256-sampled" in text
    assert "do NOT detect" in text
    assert "forgeable" in text


def test_markdown_does_not_carry_the_sampling_caveat_when_nothing_was_sampled(tmp_path):
    path = _write_bytes(tmp_path, "reads.fastq", b"A" * 1000)

    text = capture(inputs=[path], input_identity="sha256").to_markdown()

    assert "sha256-sampled" not in text
    assert "cryptographic confidence" in text


def test_markdown_always_states_the_general_limits_even_with_no_inputs():
    text = capture().to_markdown()

    assert "records what ran, not that the result is correct" in text
    assert "recorded as they were passed" in text


def test_a_pipe_in_a_value_cannot_break_the_markdown_table(monkeypatch):
    monkeypatch.setenv("FASTDNA_STRATEGY", "in|memory")

    row = [line for line in capture().to_markdown().splitlines() if line.startswith("| FASTDNA_STRATEGY ")]

    assert row == ["| FASTDNA_STRATEGY | in\\|memory |"]


def test_a_newline_in_a_value_cannot_end_the_markdown_table_early():
    text = capture(parameters={"estimator": "Pipeline(steps=[\n  ('clf', LR())])"}).to_markdown()

    rows = [line for line in text.splitlines() if line.startswith("| estimator ")]
    assert len(rows) == 1
    assert rows[0].endswith("|")


def test_repr_is_a_one_liner(tmp_path):
    prov = capture(inputs=[_write_bytes(tmp_path, "r.fastq", b"ACGT")], parameters={"k": 31})

    text = repr(prov)
    assert text.startswith("Provenance(") and "\n" not in text
    assert "n_inputs=1" in text and "n_parameters=1" in text
    assert prov.generated_utc in text


def test_repr_html_is_the_escaped_markdown_block():
    """A notebook rendering must not let a recorded value inject markup --
    a parameter is arbitrary caller-supplied text.
    """
    prov = capture(parameters={"note": "<script>alert(1)</script>"})

    rendered = prov._repr_html_()

    assert rendered.startswith("<pre>") and rendered.endswith("</pre>")
    assert "&lt;script&gt;" in rendered
    assert "<script>" not in rendered


def test_input_file_repr_truncates_the_digest(tmp_path):
    entry = capture(inputs=[_write_bytes(tmp_path, "r.fastq", b"ACGT")]).inputs[0]

    assert isinstance(entry, InputFile)
    assert entry.digest[:12] in repr(entry)
    assert entry.digest not in repr(entry)


# ---------------------------------------------------------------------------
# Composition with what the package already has.
# ---------------------------------------------------------------------------


def test_to_metadata_is_flat_strings_ready_for_fastdna_report(tmp_path):
    """The anti-duplication claim: this module renders no HTML of its own,
    it feeds `fastdna.report.to_report`, whose `metadata` table renders
    `str(key)`/`str(value)` pairs and would otherwise show a stringified
    nested dict.
    """
    from fastdna.report import to_report

    source = _write_bytes(tmp_path, "reads.fastq", b"@r0\nACGT\n+\nIIII\n")
    prov = capture(inputs=[source], parameters={"k": 31})
    metadata = prov.to_metadata()

    assert all(isinstance(key, str) and isinstance(value, str) for key, value in metadata.items())
    assert metadata["fastdna version"] == prov.fastdna_version
    assert metadata["param: k"] == "31"

    out = tmp_path / "run.html"
    to_report(out, metadata=metadata)
    document = out.read_text(encoding="utf-8")

    assert prov.fastdna_version in document
    assert prov.inputs[0].digest in document


def test_to_metadata_spells_out_absence_rather_than_printing_none(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr(
        provenance, "_TRACKED_PACKAGES", (("no_such_module_xyzzy", "no-such-distribution-xyzzy"),)
    )

    metadata = capture().to_metadata()

    assert metadata["no-such-distribution-xyzzy"] == "not installed"
    assert metadata["env: OMP_NUM_THREADS"] == "unset"


# ---------------------------------------------------------------------------
# Dependency posture.
# ---------------------------------------------------------------------------


def test_importable_and_usable_with_numpy_scipy_sklearn_and_pandas_blocked(tmp_path):
    """`fastdna.provenance` is standard library only, and the optional
    packages whose versions it reports are queried through
    `importlib.metadata` -- never imported. Verified in a subprocess with
    them genuinely blocked rather than by reading the imports.
    """
    blocked = ["numpy", "scipy", "sklearn", "pandas", "polars", "matplotlib"]
    source = _write_bytes(tmp_path, "reads.fastq", b"@r0\nACGT\n+\nIIII\n")
    script = textwrap.dedent(
        """
        import sys

        _BLOCKED = {blocked!r}


        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in _BLOCKED:
                    raise ModuleNotFoundError("No module named " + repr(name) + " (blocked)")
                return None


        sys.meta_path.insert(0, _Blocker())
        sys.path.insert(0, {source_root!r})

        import json
        from fastdna.provenance import capture

        prov = capture(inputs=[{source!r}], parameters={{"k": 31}})
        print(json.dumps({{
            "loaded": sorted(m for m in sys.modules if m.split(".")[0] in _BLOCKED),
            "fastdna_modules": sorted(m for m in sys.modules if m.startswith("fastdna.")),
            "versions": dict(prov.package_versions),
            "digest_method": prov.inputs[0].digest_method,
            "markdown_ok": prov.to_markdown().startswith("# Provenance"),
            "json_ok": bool(json.loads(prov.to_json())),
        }}))
        """
    ).format(blocked=blocked, source_root=_PYTHON_SOURCE_ROOT, source=source)

    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, "--- stdout ---\n{}\n--- stderr ---\n{}".format(proc.stdout, proc.stderr)
    result = json.loads([line for line in proc.stdout.splitlines() if line.strip()][-1])

    assert result["loaded"] == [], "provenance imported a package it only reports the version of"
    assert set(result["versions"]) == {"numpy", "pyarrow", "scipy", "scikit-learn"}
    assert result["digest_method"] == "sha256"
    assert result["markdown_ok"]
    assert result["json_ok"]
    # Decoupling, checked rather than asserted in prose: `capture()` must not
    # drag in `fastdna.audit` or `fastdna.report`. That is what lets
    # `audit()` grow a `provenance` field later, and lets `to_report()` stay
    # the one HTML renderer, without either module importing the other.
    assert "fastdna.audit" not in result["fastdna_modules"]
    assert "fastdna.report" not in result["fastdna_modules"]
