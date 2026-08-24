# FastDNA bioconda recipe (skeleton, unverified)

`meta.yaml` in this directory is a **hand-written skeleton**, following
[bioconda's standard recipe format](https://bioconda.github.io/contributor/index.html)
for a Rust-plus-PyO3/maturin Python package, matching what `pyproject.toml`
at the repository root already declares (`maturin` build backend,
`pyarrow>=14` runtime dependency, `python-source = "python"`,
`module-name = "fastdna._core"`).

**It has not been built.** No `conda build` / `conda-mambabuild` was run
against it, and it has not been linted with bioconda's own `bioconda-utils
lint` tooling. Treat it as a well-informed starting point for a real
submission, not as evidence the package builds cleanly on conda.

## What still needs to happen before this could actually build

- **License**: this repository currently declares no license anywhere (no
  `LICENSE` file, no `license` field in `Cargo.toml` or `pyproject.toml`).
  Bioconda requires both an explicit `about.license` and a real
  `about.license_file` pointing at a file that exists in the source tree.
  `meta.yaml` uses `MIT` as a placeholder and says so in a comment --
  replace it with whatever license the project actually adopts, and add
  the corresponding `LICENSE` file.
- **`source.url` / `source.sha256`**: both are placeholders
  (`REPLACE_WITH_ORG`, an all-zero checksum). Bioconda recipes source from
  a released tarball (PyPI sdist or a GitHub release tag), not a git
  checkout -- once a real release exists, download it and compute the real
  `sha256sum`.
- **Rust dependency vendoring**: bioconda's Rust recipes typically need
  either network access disabled during `conda build` (Cargo needs
  `Cargo.lock` present and network access, or pre-vendored dependencies,
  since conda's build sandbox blocks network by default) or a
  `cargo vendor` step added to `build.script`. This skeleton does not
  attempt to solve that -- it is exactly the kind of thing bioconda's own
  CI would catch and that a real submission needs to work through.
- **A `test.commands` fixture**: the current `test:` section only checks
  that `import fastdna` and `fastdna.build_info()` work. A stronger test
  (recommended before submitting) would package a tiny FASTQ fixture and
  run `fastdna.count()` against it, asserting a known distinct/total count
  -- the same kind of check `python/tests/test_api.py` already does in
  this repository.

## How to actually submit this to bioconda

Bioconda recipes live in the community-maintained
[`bioconda/bioconda-recipes`](https://github.com/bioconda/bioconda-recipes)
repository, not in this project's own repository -- `recipe/meta.yaml`
here is a convenience copy to keep in sync by hand, not something that
gets picked up by bioconda automatically. The one manual step that
actually gets this in front of bioconda's maintainers:

1. Fork `bioconda/bioconda-recipes`.
2. Copy this file to `recipes/fastdna/meta.yaml` in that fork (note the
   path: bioconda nests every package one level deeper, under `recipes/`,
   plural, than this repository's own `recipe/`, singular).
3. Fill in the placeholders above (license, source URL/checksum, and
   ideally the stronger test).
4. Open a pull request against `bioconda/bioconda-recipes`. Their CI
   (`bioconda-utils`) lints and builds the recipe for real -- that is the
   actual verification this skeleton could not perform locally.
5. Address any review feedback from the bioconda maintainers.

This is not something this contribution attempted to do -- no PR has been
opened, and none should be implied by this file's existence.
