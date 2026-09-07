"""Test-collection helper, not part of the `fastdna` package itself.

`fastdna/__init__.py` unconditionally does `from . import _core` (the
compiled Rust extension) at import time, so `import fastdna` -- and
therefore `import fastdna.<anything>` -- normally requires `maturin
develop` to have been run first.

`python/fastdna/spectrum.py` is pure Python over a plain
`{depth: count}` mapping and has no dependency on the Rust extension at
all -- it should be importable and testable without ever building it. The
only thing standing in the way is `fastdna/__init__.py`'s own
unconditional import.

If the real `fastdna._core` extension is already built and importable,
this is a complete no-op (the `try` succeeds and nothing is stubbed) --
every other test module continues to exercise the real Rust extension
exactly as before. Only when it is genuinely missing does this register a
minimal stand-in with just enough surface (`__version__`, read at
`fastdna/__init__.py` module scope) for `fastdna/__init__.py` to finish
importing; it does not stub any of the actual Rust-backed functions
(`count`, `sketch`, etc.), so tests that call into those still fail
honestly if exercised without the real extension built.
"""

import sys
import types

try:
    import fastdna._core  # noqa: F401
except ImportError:
    _stub = types.ModuleType("fastdna._core")
    _stub.__version__ = "0.0.0-dev-no-rust-extension-built"
    sys.modules.setdefault("fastdna._core", _stub)
