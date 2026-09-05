# Installation

!!! danger "There is no published package yet"

    FastDNA has never been released. As of this page:

    - **PyPI** — no project named `fastdna`. `pip install fastdna` fails with
      `ERROR: No matching distribution found for fastdna`.
    - **crates.io** — no crate named `fastdna`.
    - **Bioconda** — no package. A recipe skeleton exists at
      [`recipe/meta.yaml`](https://github.com/Angel-Zeledon/FastDNA/blob/master/recipe/meta.yaml),
      but it has never been built with `conda build` or submitted, and its
      source URL and sha256 are placeholders.
    - The repository has **no git tags**, and the wheel-building workflow
      (`.github/workflows/wheels.yml`) states in its own header that it
      "intentionally has no publish step".

    Both names are currently unregistered by anyone. Until a release lands,
    the build-from-source path below is the only way to install FastDNA, and
    it is fully supported.

## From source (works today)

Two things can be built, independently: the **Python extension module**, and
the **`fastdna` command-line binary**. Most users want the first.

### The Python package

```bash
git clone https://github.com/Angel-Zeledon/FastDNA.git
cd FastDNA
pip install maturin
maturin develop --release --features python
```

That compiles the Rust core into `fastdna._core` and installs the `fastdna`
package into the active environment — the same extension module a wheel
would deliver.

!!! tip "`--release` is not optional in practice"

    A debug build of the extension measured **4–5x slower** in this
    project's own benchmarking. Use `--release` unless you are actively
    debugging the Rust core.

`--features python` is required, and is the reason `maturin` is used rather
than a bare `cargo build`: the `python` feature enables PyO3 with
`extension-module`, which tells the crate not to link `libpython`. That is
correct for the `cdylib` maturin produces and *fatal* for the `[[bin]]`
target on Linux and macOS, where a plain `cargo build --features python`
fails with undefined `Py_*` symbols. maturin builds only `--lib` and never
turns the feature on for the binary.

To produce an installable wheel for the machine you are on, rather than
installing into the current environment:

```bash
maturin build --release --features python
```

**Requirements:** a Rust toolchain (install via [rustup](https://rustup.rs);
`Cargo.toml` does not pin a minimum version, so current stable is the
assumption) and CPython 3.8 or newer. The only runtime dependency is
`pyarrow >= 14`, which pip installs automatically.

### The command-line binary

The CLI is a separate Cargo target and does **not** come with the Python
package:

```bash
cargo build --release
./target/release/fastdna --input sample.fastq.gz --output counts.parquet -k 31
```

Note the deliberate absence of `--features python` here — see above.

## Optional extras

The package declares two extras in `pyproject.toml`. Neither is needed to
count k-mers, and `import fastdna` pulls in nothing beyond `pyarrow`.

| Extra | Command | What it is for |
|---|---|---|
| `test` | `pip install ".[test]"` | Running `pytest python/tests`. Brings in numpy, polars, duckdb, tqdm and pytest. |
| `docs` | `pip install ".[docs]"` | Building this site with `mkdocs build`. |

!!! note "Installing an extra also builds the package"

    Both commands above install `fastdna` itself, which means compiling the
    Rust core — so both need a Rust toolchain today, exactly as the
    from-source instructions do.

    The site is the exception that does not: mkdocstrings reads
    `python/fastdna/` **statically**, so `mkdocs build` needs neither a
    compiled `fastdna._core` nor any scientific dependency. To build the docs
    without building the crate, install just the three tools the `docs` extra
    names:

    ```bash
    pip install "mkdocs>=1.6" "mkdocs-material>=9.5" "mkdocstrings[python]>=0.25"
    mkdocs build --strict
    ```

    That is what `.github/workflows/docs.yml` does, which is why the docs job
    needs no Rust toolchain and no `maturin` step.

`numpy` is the only optional package any module reaches for, and it is
imported lazily inside the functions that need it, so `import fastdna`
works without it.

## Checking that it worked

```python
import fastdna

print(fastdna.__version__)
# 0.1.0
print(fastdna.build_info())
# {'version': '0.1.0', 'max_k': 32, 'avx2': True}
```

[`build_info()`](api/counting.md#fastdna.build_info) reports the installed
version, the maximum supported `k`, and whether AVX2 is live on *this* CPU —
which is what makes "it's slow on my machine" diagnosable remotely.

## Once a release exists

When the first release is published, installation becomes:

```bash
pip install fastdna
```

with no Rust toolchain, no compiler and no build step: the crate compiles
against [PyO3's `abi3` stable ABI](https://pyo3.rs) (`abi3-py38`), so a
single wheel per platform covers CPython 3.8 through 3.13+. CI already
builds and tests those wheels for five platforms — manylinux x86_64,
manylinux aarch64 (cross-compiled; built but not test-executed, since the
runner is x86_64), macOS x86_64, macOS arm64 and Windows x86_64 — and fails
the build if a produced wheel filename does not carry an `abi3` tag. The
wheels are tagged `cp38-abi3` and `requires-python = ">=3.8"`, though the CI
test matrix runs on Python 3.9+, so 3.8 support is declared but not
exercised.

In other words: the packaging is built and tested on every push. What is
missing is the publish step, not the wheels.
