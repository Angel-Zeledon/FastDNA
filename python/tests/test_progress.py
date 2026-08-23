"""Direct, deterministic tests of `fastdna._progress._MonotonicAdapter`.

`test_api.py::test_progress_counts_are_monotonic_for_the_consumer` drives a
real (sub-millisecond) `count()` call and asserts the observed events came
out sorted -- a useful smoke test, but out-of-order delivery from the Rust
core may simply never occur in that short a window, so that test alone
passes even with the adapter's serialization logic deleted entirely. These
tests instead feed `_MonotonicAdapter` known-unordered input directly and
assert on exactly what the sink receives, so a regression in the
serialization logic itself fails immediately, with no dependence on
scheduling luck.
"""

from fastdna._progress import _MonotonicAdapter


def test_monotonic_adapter_drops_out_of_order_counts():
    seen = []
    adapter = _MonotonicAdapter(seen.append)

    for event in [100, 300, 200, 400]:
        adapter(event)

    assert seen == [100, 300, 400]


def test_monotonic_adapter_passes_through_non_int_events_unconditionally():
    # Dict-shaped events (e.g. `{"event": "finished", ...}`) are not subject
    # to the monotonic-count check at all -- they must always reach the
    # sink, regardless of what integer events preceded them.
    seen = []
    adapter = _MonotonicAdapter(seen.append)

    adapter(500)
    adapter(100)  # dropped: lower than the highest ReadsProcessed seen
    finished = {"event": "finished", "reads": 500}
    adapter(finished)

    assert seen == [500, finished]
