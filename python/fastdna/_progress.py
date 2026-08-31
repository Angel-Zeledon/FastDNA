"""Progress adapter for `fastdna.count`.

The core's callback contract (docs/superpowers/specs/2026-08-22-fastdna-
python-design.md, "Callback contract") requires any Python-side consumer to
handle two things the Rust core does not guarantee on its own:

- The callback is invoked *concurrently and re-entrantly* from several
  rayon worker threads at once.
- `ReadsProcessed` values can arrive **out of order**: the underlying
  counter is atomic, but delivery to the callback is not serialized
  relative to it -- a worker can cross 200,000 reads, be preempted before
  calling the callback, and have another worker's 300,000 delivered first.

`tqdm` is not thread-safe and would render a bar that jumps backwards if fed
events directly. This module is the single place that serializes events --
with a lock, dropping any count lower than the highest seen -- before they
reach `tqdm` or a user-supplied callback. It applies unconditionally,
whether `progress=True` or a plain callable, because the out-of-order
problem is a property of the core's delivery, not of what the caller does
with the events.
"""

import threading
from typing import Any, Callable, Optional, Union

__all__ = ["make_progress_adapter"]


def make_progress_adapter(
    progress: Optional[Union[bool, Callable[[Any], None]]],
) -> Optional[Callable[[Any], None]]:
    """Builds the callable passed to `_core.count`'s `progress` parameter.

    - `progress is None` -> returns `None`; `_core.count` stays silent.
    - `progress is True` -> drives a `tqdm` bar if `tqdm` is importable.
      `tqdm` is a soft dependency: `import fastdna` must work without it,
      so this degrades to silence (returns `None`) rather than raising when
      it is missing.
    - a callable -> wrapped with the same serialization `tqdm` gets, so a
      user-supplied callback never has to reimplement thread-safety itself.
    """
    if progress is None:
        return None

    if progress is True:
        sink = _tqdm_sink()
        if sink is None:
            return None
        return _MonotonicAdapter(sink)

    if callable(progress):
        return _MonotonicAdapter(progress)

    raise TypeError(f"progress must be None, True, or a callable, got {progress!r}")


class _MonotonicAdapter:
    """Serializes progress events and drops out-of-order `ReadsProcessed` counts.

    Holds a lock because the Rust core invokes the wrapped callback
    concurrently from multiple worker threads; without it, a non-thread-safe
    consumer (`tqdm`, a list's `.append` racing itself) would see
    interleaved calls, and `ReadsProcessed` integers could be delivered out
    of the order the underlying counter actually reached them in.
    """

    def __init__(self, sink):
        self._sink = sink
        self._lock = threading.Lock()
        self._highest = -1

    def __call__(self, event):
        with self._lock:
            if isinstance(event, int):
                if event < self._highest:
                    # A lower count than one already delivered: another
                    # worker's later fetch_add was seen first. Drop it
                    # rather than let a consumer like tqdm render a bar
                    # that jumps backwards.
                    return
                self._highest = event
            self._sink(event)


def _tqdm_sink():
    """Builds a sink that drives a `tqdm` bar, or `None` if `tqdm` is not
    installed. `tqdm` itself is never imported at module load time -- only
    here, on first use with `progress=True` -- so `import fastdna` does not
    require it.
    """
    try:
        import tqdm
    except ImportError:
        return None

    bar = tqdm.tqdm(unit=" reads")
    state = {"last": 0}

    def sink(event):
        if isinstance(event, int):
            bar.update(event - state["last"])
            state["last"] = event
        elif isinstance(event, dict) and event.get("event") == "finished":
            bar.close()

    return sink
