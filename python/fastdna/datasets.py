"""fastdna.datasets -- ready-made, labeled genomic cohorts for real-data work.

## Why this module exists

Two real-data reproductions were done by hand the same night this module
was written: a bacterial AMR cohort (E. coli / ciprofloxacin, from BV-BRC,
`scratch/amr_repro/select_and_download.py`) and a viral resistance cohort
(HIV-1 protease / nelfinavir, from Stanford HIVDB,
`scratch/hiv_repro/build_hiv_repro.py`). Both needed a bespoke script to
download, filter, and cache real public data with real phenotype labels
before `fastdna.count()` or `fastdna.audit()` could touch a single sample.
This module turns that one-off work into two reusable functions,
:func:`load_amr` and :func:`load_hiv_resistance`, so a real labeled cohort
is one function call away -- the cheapest way to let a new user try
FastDNA's honest-ML story (`fastdna.audit`, `fastdna.cv`, `fastdna.mic`) on
real data in minutes, instead of writing a download script first.

## Caching

Every download lands under a cache directory, resolved in this order:

1. The `cache_dir` argument to :func:`load_amr` / :func:`load_hiv_resistance`,
   if given.
2. The `FASTDNA_DATA_CACHE` environment variable, if set -- the same
   "one environment variable per concern, with a built-in default" pattern
   this package already uses for `FASTDNA_SPILL_DIR` (see `disk_spill.rs`),
   rather than a new dependency on `platformdirs` or similar for a single
   directory this package fully controls the layout of.
3. `~/.fastdna/datasets` otherwise -- a plain dotfile under the user's
   home directory, portable across the platforms this package already
   supports without pulling in an OS-specific-cache-directory library.

Caching is real and idempotent: before any network request, both loaders
check whether the target file already exists on disk and is non-empty, and
skip the download entirely if so. A second call with the same arguments
touches the network only for whatever it has not already cached (e.g. a
larger `n_samples` that needs additional genomes/sequences). Every write
into the cache goes through a temp-file-then-`os.replace` step (see
`_write_cache_file`), so a process killed mid-download never leaves a
truncated file that a later run would mistake for a complete one -- the
same write-to-temp-then-rename discipline `src/atomic.rs` uses on the Rust
side, reimplemented here in Python since nothing in this module crosses
the FFI boundary.

## Design decisions worth knowing before using this module

- **Species support (`load_amr`).** The upstream metadata repository
  (BarquistLab/AMR_prediction) publishes one metadata TSV per species.
  E. coli, Klebsiella pneumoniae, Salmonella enterica, and Streptococcus
  pneumoniae all share the same relevant column names (`genome_id`,
  `genome_name`, `antibiotic`, `resistant_phenotype`, `MIC`,
  `MIC_criterion`, `ST`), confirmed by fetching each file's header before
  writing this module -- so all four are supported. Staphylococcus aureus
  is also published there but its file's header has no `MIC`/
  `MIC_criterion` columns at all (confirmed the same way), which breaks
  this loader's contract that `AmrCohort.mic` is either a real value or an
  explicit `nan` -- it is deliberately not supported rather than silently
  returning all-`nan` MIC for that species alone.
- **Back-translation stays private to this module.** `load_hiv_resistance`
  needs to hand `fastdna.count()` a DNA string for a sequence HIVDB only
  publishes as amino-acid genotype calls, so each residue is mapped to one
  fixed, arbitrary human-preferred codon (`_CODON_TABLE`, transcribed
  verbatim from `scratch/hiv_repro/build_hiv_repro.py`). This is a lossless
  round-trip convention, not a claim about HIV-1's real codon usage.
  `fastdna.translate` is a general, Rust-backed, NCBI-genetic-code-table
  translator with real biological semantics; promoting an arbitrary
  reverse-translation table into it would misrepresent this narrow trick as
  a validated tool. It stays a private helper here instead.
- **Not re-exported from `fastdna.__init__`.** Per that module's own
  comment on the `audit`/`explain` re-export: flagship, load-bearing
  functions get a top-level re-export; heavier or narrower modules
  (`fastdna.sklearn`, `fastdna.rules`, and now `fastdna.datasets`) stay
  reachable via an explicit `from fastdna.datasets import load_amr`. These
  two functions are onboarding/convenience entry points, not a
  differentiator-level capability, and neither is on the hot path of
  anything else in this package -- there is no strong reason to pay the
  lazy-reexport machinery's complexity (see that same comment) for them.

`numpy` is imported at module scope here, matching `fastdna.cv`/
`fastdna.rules`/`fastdna.audit`'s own convention for explicit-import-only
modules: this module is never imported by `fastdna/__init__.py`, so a bare
`import fastdna` never pays for it (verified by
`python/tests/test_optional_dependencies.py`, which does not touch this
module). Only `import fastdna.datasets` requires numpy to be installed.
Downloading uses `urllib.request` (stdlib), matching
`scratch/amr_repro/select_and_download.py`'s own choice -- this package's
only prior download code -- rather than adding `requests` as a dependency
for two functions.
"""
from __future__ import annotations

import csv
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import _PathLike

__all__ = ["load_amr", "load_hiv_resistance", "AmrCohort", "HivResistanceCohort"]

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------

#: Overrides the default cache location for every loader in this module,
#: the same "one env var per concern" family as `FASTDNA_SPILL_DIR` /
#: `FASTDNA_MAX_RAM_BYTES` (see the module docstring's "Caching" section).
_CACHE_DIR_ENV_VAR = "FASTDNA_DATA_CACHE"


def _resolve_cache_dir(cache_dir: Optional[_PathLike]) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    override = os.environ.get(_CACHE_DIR_ENV_VAR)
    if override:
        return Path(override)
    return Path.home() / ".fastdna" / "datasets"


# ---------------------------------------------------------------------------
# Download / cache primitives -- shared by both loaders
# ---------------------------------------------------------------------------


def _download_bytes(
    url: str,
    *,
    max_attempts: int = 4,
    timeout: float = 60.0,
    validate: Optional[Callable[[bytes], bool]] = None,
) -> bytes:
    """Downloads `url` with retry/backoff, returning the response body.

    Retries up to `max_attempts` times with a linearly increasing delay
    (`2 * attempt` seconds) on a transient network error, a timeout, or a
    response that fails `validate` (e.g. an HTML error page returned in
    place of the expected FASTA/TSV body). BV-BRC and Stanford HIVDB are
    public, unauthenticated services with no rate limiting observed across
    150 sequential requests during this module's own manual reproduction,
    but neither guarantees 100% availability on every single request.

    This is the one seam every download in this module funnels through --
    tests monkeypatch this function directly to serve local fixture bytes
    instead of touching the network.

    Raises
    ------
    RuntimeError
        Every attempt failed; the message names `url` and the last error.
    """
    import urllib.error
    import urllib.request

    headers = {"User-Agent": "fastdna-datasets/1.0"}
    last_error: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
            if validate is not None and not validate(data):
                last_error = RuntimeError(
                    f"response body failed validation (first 60 bytes: {data[:60]!r})"
                )
            else:
                return data
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if attempt < max_attempts:
            time.sleep(2 * attempt)
    raise RuntimeError(
        f"failed to download {url} after {max_attempts} attempts: {last_error}"
    ) from last_error


def _write_cache_file(dest: Path, data: bytes) -> None:
    """Writes `data` to `dest` atomically: a temp file in the same
    directory is written and fsynced, then renamed into place with
    `os.replace` (atomic on both POSIX and Windows within one filesystem).
    A process killed mid-write never leaves a truncated file behind that a
    later `dest.exists() and dest.stat().st_size > 0` cache check would
    mistake for a complete entry.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), prefix=dest.name + ".", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(data)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_name, str(dest))
    except BaseException:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


def _fetch_cached(
    url: str,
    dest: Path,
    *,
    validate: Optional[Callable[[bytes], bool]] = None,
    max_attempts: int = 4,
    timeout: float = 60.0,
) -> Path:
    """Returns `dest`, downloading `url` into it first iff `dest` does not
    already exist as a non-empty file. Real idempotent caching: a second
    call with the same `dest` never touches the network.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    data = _download_bytes(url, max_attempts=max_attempts, timeout=timeout, validate=validate)
    _write_cache_file(dest, data)
    return dest


# ---------------------------------------------------------------------------
# Stratified sampling -- shared by both loaders
# ---------------------------------------------------------------------------


def _stratified_sample(records, is_positive: Callable, n_samples: Optional[int], random_state, *, caller: str):
    """A reproducible subsample of `n_samples` records from `records`,
    split by `is_positive` into two classes whose selected proportions
    match the full pool's own proportions as closely as integer rounding
    allows (the deficit, if a class's exact quota exceeds its own pool
    size, is made up from the other class -- always possible here since
    `n_samples <= len(records)` is checked below).

    `n_samples=None` returns every record, shuffled once rather than left
    in file order -- an unshuffled cohort would hand a downstream model a
    phenotype-sorted file, which trivially "predicts" the label from row
    position alone.

    Both classes' own shuffle order and the final interleave are fixed
    functions of `random_state` (via `random.Random(random_state)`), so
    two calls with identical arguments return the identical selection in
    the identical order.

    Raises
    ------
    ValueError
        `n_samples` is not a positive int or None, or exceeds the number
        of records available.
    """
    positive = [r for r in records if is_positive(r)]
    negative = [r for r in records if not is_positive(r)]
    total = len(positive) + len(negative)

    rng = random.Random(random_state)
    rng.shuffle(positive)
    rng.shuffle(negative)

    if n_samples is None:
        selected = positive + negative
        rng.shuffle(selected)
        return selected

    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"{caller}: n_samples must be a positive int or None, got {n_samples!r}")
    if n_samples > total:
        raise ValueError(
            f"{caller}: n_samples={n_samples} exceeds the {total} samples available after "
            f"filtering ({len(positive)} positive, {len(negative)} negative class). Pass "
            f"n_samples<={total}, or None to use all of them."
        )

    quota_positive = round(n_samples * len(positive) / total) if total else 0
    quota_positive = max(0, min(quota_positive, len(positive), n_samples))
    quota_negative = n_samples - quota_positive
    if quota_negative > len(negative):
        deficit = quota_negative - len(negative)
        quota_negative = len(negative)
        quota_positive = min(quota_positive + deficit, len(positive))

    selected = positive[:quota_positive] + negative[:quota_negative]
    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# AMR cohort (BV-BRC / BarquistLab AMR_prediction metadata)
# ---------------------------------------------------------------------------

_AMR_METADATA_BASE_URL = "https://raw.githubusercontent.com/BarquistLab/AMR_prediction/main/metadata/"

#: species -> metadata TSV filename. Every one of these was confirmed (by
#: fetching its real header) to share `genome_id`, `genome_name`,
#: `antibiotic`, `resistant_phenotype`, `MIC`, `MIC_criterion`, and `ST`
#: column names with the E. coli file this module was originally written
#: against -- see the module docstring's "Design decisions" section for why
#: Staphylococcus aureus (also published in the same repository) is not
#: included here.
_AMR_SPECIES_FILES = {
    "Escherichia coli": "metadata_extend_E_coli_checkm.tsv",
    "Klebsiella pneumoniae": "metadata_Klebsiella_pneumoniae_checkm.tsv",
    "Salmonella enterica": "metadata_Salmonella_enterica_checkm.tsv",
    "Streptococcus pneumoniae": "metadata_Streptococcus_pneumoniae_checkm.tsv",
}

#: BV-BRC's public REST endpoint for one genome's assembled FASTA. No
#: authentication, keyed only by `genome_id` -- not species-specific in the
#: URL at all, which is why genome download is supported uniformly across
#: every species in `_AMR_SPECIES_FILES` even though the manual
#: reproduction that preceded this module only exercised it against
#: E. coli genome ids (a Klebsiella pneumoniae genome id was spot-checked
#: against this same endpoint while writing this module and returned a
#: real multi-contig assembly, matching the E. coli behavior exactly).
_BVBRC_GENOME_SEQUENCE_URL = (
    "https://www.bv-brc.org/api/genome_sequence/?eq(genome_id,{genome_id})&http_accept=application/dna+fasta"
)

#: Extracts the first embedded number from a BV-BRC `MIC` field such as
#: `'32'`, `'>32'`, `'<=0.5'`, `'>=4'`, or a combination-drug value like
#: `'8/4'` (ampicillin/sulbactam) -- the first number of which is the
#: primary agent's concentration. The `<`/`<=`/`>`/`>=` operator itself is
#: discarded: this is a best-effort magnitude for `AmrCohort.mic`, not a
#: censored-regression-ready value, and `'>32'` and `'32'` both parse to
#: `32.0`. `fastdna.mic.log2_mic` will reject a resulting 0 or negative
#: value the same way it rejects any other non-positive MIC.
_MIC_NUMBER_RE = re.compile(r"(\d+\.?\d*|\.\d+)")


def _parse_mic(raw: Optional[str]) -> float:
    if not raw:
        return float("nan")
    match = _MIC_NUMBER_RE.search(raw)
    if not match:
        return float("nan")
    return float(match.group(1))


def _slug(species: str) -> str:
    return species.strip().lower().replace(" ", "_")


@dataclass(frozen=True)
class AmrCohort:
    """A labeled AMR cohort returned by :func:`load_amr`.

    `frozen=True`: this is a data artifact meant to be handed to
    `fastdna.count`/`fastdna.sklearn.KmerVectorizer`/`fastdna.mic` as-is,
    the same immutability convention `fastdna.cohort_counts.CohortCounts`
    uses for the same reason.

    Attributes
    ----------
    species : str
        The species requested, verbatim (a key of the species this module
        supports -- see the module docstring).
    antibiotic : str
        The antibiotic requested, verbatim.
    paths : tuple of str
        Local paths to each selected genome's assembled FASTA, cached under
        the resolved cache directory. Ready for `fastdna.count()` /
        `fastdna.sklearn.KmerVectorizer` directly.
    phenotype : numpy.ndarray of int64, shape (n,)
        `1` for resistant, `0` for susceptible, aligned with `paths`.
        Intermediate and any other non-binary phenotype call is dropped
        before this array is built (see :func:`load_amr`).
    mic : numpy.ndarray of float64, shape (n,)
        The continuous MIC value from the same metadata row, or `nan`
        where the `MIC` column is empty or not a parseable number (see
        `_parse_mic`) -- the regression-task counterpart of `phenotype`,
        for `fastdna.mic.MicRegressor`.
    sample_ids : tuple of str
        Each sample's BV-BRC `genome_id`, aligned with `paths`.
    genome_name : tuple of str
        Each sample's BV-BRC `genome_name`, aligned with `paths`.
    sequence_type : tuple of (str or None)
        Each sample's multilocus sequence type (`ST` column), or `None`
        where absent.
    """

    species: str
    antibiotic: str
    paths: tuple[str, ...]
    phenotype: np.ndarray
    mic: np.ndarray
    sample_ids: tuple[str, ...]
    genome_name: tuple[str, ...]
    sequence_type: tuple[Optional[str], ...]

    def __len__(self) -> int:
        return len(self.sample_ids)


def _load_amr_candidates(metadata_path: Path, antibiotic: str):
    """One record per `genome_id` with a resistant/susceptible phenotype
    call for `antibiotic`, deduplicated across rows; a genome called
    resistant in one row and susceptible in another (`conflicted`) is
    dropped rather than arbitrarily picked -- the same logic
    `scratch/amr_repro/select_and_download.py::load_candidates` uses.

    Returns `(by_genome, conflicted, antibiotics_seen)`: `by_genome` maps
    `genome_id -> {genome_id, genome_name, phenotype, ST, MIC}`;
    `conflicted` is the set of genome ids dropped for disagreeing rows;
    `antibiotics_seen` is every distinct value the file's own `antibiotic`
    column takes, used to build a helpful error message for an unknown
    `antibiotic` without a second pass over the file.
    """
    by_genome: dict = {}
    conflicted: set = set()
    antibiotics_seen: set = set()
    with open(metadata_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row_antibiotic = row.get("antibiotic") or ""
            antibiotics_seen.add(row_antibiotic)
            if row_antibiotic != antibiotic:
                continue
            phenotype = (row.get("resistant_phenotype") or "").strip().lower()
            if phenotype not in ("resistant", "susceptible"):
                continue
            genome_id = row["genome_id"]
            if genome_id in by_genome and by_genome[genome_id]["phenotype"] != phenotype:
                conflicted.add(genome_id)
                continue
            by_genome[genome_id] = {
                "genome_id": genome_id,
                "genome_name": row.get("genome_name") or "",
                "phenotype": phenotype,
                "ST": (row.get("ST") or None),
                "MIC": row.get("MIC") or "",
            }
    for genome_id in conflicted:
        by_genome.pop(genome_id, None)
    return by_genome, conflicted, antibiotics_seen


def load_amr(
    species: str,
    antibiotic: str,
    *,
    n_samples: Optional[int] = None,
    random_state: int = 0,
    cache_dir: Optional[_PathLike] = None,
) -> AmrCohort:
    """Loads a real, labeled antimicrobial-resistance cohort from BV-BRC.

    Downloads (or reuses a cached copy of) the BarquistLab/AMR_prediction
    metadata TSV for `species`, filters to `antibiotic` with a binary
    resistant/susceptible phenotype call (`intermediate` and any other
    non-binary or conflicting call is dropped -- see
    `_load_amr_candidates`), optionally stratified-subsamples to
    `n_samples`, and downloads each selected genome's assembled FASTA from
    BV-BRC's REST API into the cache. See the module docstring's "Caching"
    section for where files land and "Design decisions" for which species
    are supported and why.

    Parameters
    ----------
    species : str
        One of the species this module supports -- see the module
        docstring. An unsupported value raises `ValueError` listing what
        is available.
    antibiotic : str
        An antibiotic present in that species' metadata file (BV-BRC's own
        naming, e.g. `"ciprofloxacin"`). An unrecognized value raises
        `ValueError` listing every antibiotic actually present in the
        file.
    n_samples : int, optional
        `None` (the default) keeps every genome with a valid
        resistant/susceptible call for `antibiotic`. Otherwise, a
        stratified random sample of exactly `n_samples` genomes whose
        resistant/susceptible proportions match the full filtered pool's
        own, as closely as integer rounding allows. Raises `ValueError` if
        larger than the pool.
    random_state : int, default 0
        Seeds the stratified sample; the same `species`/`antibiotic`/
        `n_samples`/`random_state` always returns the same genomes in the
        same order.
    cache_dir : str or os.PathLike, optional
        Where downloads are cached. See the module docstring's "Caching"
        section for the default and the `FASTDNA_DATA_CACHE` override.

    Returns
    -------
    AmrCohort

    Raises
    ------
    ValueError
        `species` is not supported, `antibiotic` is not present in that
        species' metadata, no genome has a resistant/susceptible call for
        it, or `n_samples` is invalid.
    RuntimeError
        A download failed after retries (see `_download_bytes`).
    """
    if species not in _AMR_SPECIES_FILES:
        available = ", ".join(sorted(_AMR_SPECIES_FILES))
        raise ValueError(
            f"load_amr(): unknown species {species!r}. Available species (BV-BRC "
            f"AMR_prediction metadata with a matching schema, including MIC/MIC_criterion "
            f"columns): {available}. Staphylococcus aureus is also published in that "
            "repository, but its metadata file has no MIC/MIC_criterion columns, so it is "
            "not supported by this loader."
        )

    cache_root = _resolve_cache_dir(cache_dir)
    species_dir = cache_root / "amr" / _slug(species)
    metadata_filename = _AMR_SPECIES_FILES[species]
    metadata_path = species_dir / metadata_filename
    _fetch_cached(
        _AMR_METADATA_BASE_URL + metadata_filename,
        metadata_path,
        validate=lambda data: data.startswith(b"genome_id\t"),
    )

    by_genome, conflicted, antibiotics_seen = _load_amr_candidates(metadata_path, antibiotic)
    if antibiotic not in antibiotics_seen:
        available = ", ".join(sorted(a for a in antibiotics_seen if a))
        raise ValueError(
            f"load_amr(): unknown antibiotic {antibiotic!r} for species {species!r}. "
            f"Available antibiotics in this metadata file: {available}"
        )
    if not by_genome:
        raise ValueError(
            f"load_amr(): no genome has a resistant/susceptible phenotype call for "
            f"{antibiotic!r} in {species!r} ('intermediate' and any other non-binary call "
            f"is dropped; {len(conflicted)} genome(s) with a resistant call in one row and "
            "a susceptible call in another were also dropped)."
        )

    selected = _stratified_sample(
        list(by_genome.values()),
        lambda record: record["phenotype"] == "resistant",
        n_samples,
        random_state,
        caller="load_amr()",
    )

    genomes_dir = species_dir / "genomes"
    paths = []
    for record in selected:
        dest = genomes_dir / f"{record['genome_id']}.fna"
        url = _BVBRC_GENOME_SEQUENCE_URL.format(genome_id=record["genome_id"])
        _fetch_cached(url, dest, validate=lambda data: data.startswith(b">"))
        paths.append(str(dest))

    phenotype = np.array(
        [1 if record["phenotype"] == "resistant" else 0 for record in selected], dtype=np.int64
    )
    mic = np.array([_parse_mic(record["MIC"]) for record in selected], dtype=np.float64)

    return AmrCohort(
        species=species,
        antibiotic=antibiotic,
        paths=tuple(paths),
        phenotype=phenotype,
        mic=mic,
        sample_ids=tuple(record["genome_id"] for record in selected),
        genome_name=tuple(record["genome_name"] for record in selected),
        sequence_type=tuple(record["ST"] for record in selected),
    )


# ---------------------------------------------------------------------------
# HIV-1 protease drug-resistance cohort (Stanford HIVDB)
# ---------------------------------------------------------------------------

_HIVDB_PI_DATASET_URL = "https://hivdb.stanford.edu/download/GenoPhenoDatasets/PI_DataSet.txt"

#: The 8 protease-inhibitor fold-resistance columns in PI_DataSet.txt, in
#: the file's own column order.
_HIV_DRUG_COLUMNS = ("FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV")

#: Documented, defensible default fold-resistance cutoffs, per drug --
#: only drugs with a literature-backed default are listed. NFV's 4.0-fold
#: cutoff is the PhenoSense/Monogram Biosciences clinical cutoff for
#: nelfinavir, used by the same Rhee et al. 2006 (PNAS) methodology that
#: built this exact dataset's genotype/phenotype pairing (see
#: `scratch/hiv_repro/build_hiv_repro.py`, which this default is
#: transcribed from verbatim). A drug not listed here requires an explicit
#: `fold_cutoff` from the caller rather than a guessed default.
_HIV_FOLD_CUTOFFS = {"NFV": 4.0}

#: HIV-1 HXB2 protease (PR) reference amino-acid sequence, 99 residues --
#: the wild-type comparator HIVDB itself used to build this dataset's own
#: `CompMutList` column. Transcribed verbatim from
#: `scratch/hiv_repro/build_hiv_repro.py`, including its position-37
#: correction: position 37 is `S` in the literal HXB2 clone but `N` in the
#: subtype-B consensus HIVDB actually compares genotypes against for PI
#: resistance calling -- confirmed there empirically (reconstructing all
#: 895 clean rows of this dataset against an S-at-37 reference produced
#: exactly one recurring `CompMutList` disagreement, always at position 37;
#: N-at-37 makes every one of those rows agree). `_verify_hxb2_reference`
#: (called once, at import time, below) checks this string against five of
#: `CompMutList`'s own reported wild-type residues before it is trusted.
_HXB2_PR = (
    "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKA"
    "IGTVLVGPTPVNIIGRNLLTQIGCTLNF"
)
_HXB2_PR = _HXB2_PR[:36] + "N" + _HXB2_PR[37:]
if len(_HXB2_PR) != 99:
    raise AssertionError(f"fastdna.datasets: _HXB2_PR must be 99 residues, got {len(_HXB2_PR)}")

_AA_LETTERS = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: Arbitrary-but-consistent back-translation table: one representative
#: human-preferred codon per amino acid, transcribed verbatim from
#: `scratch/hiv_repro/build_hiv_repro.py`. Not a claim about HIV-1's own
#: codon usage -- see the module docstring's "Design decisions" section for
#: why this stays private to this module instead of living in
#: `fastdna.translate`.
_CODON_TABLE = {
    "A": "GCC", "R": "CGC", "N": "AAC", "D": "GAC", "C": "TGC",
    "Q": "CAG", "E": "GAG", "G": "GGC", "H": "CAC", "I": "ATC",
    "L": "CTG", "K": "AAG", "M": "ATG", "F": "TTC", "P": "CCC",
    "S": "AGC", "T": "ACC", "W": "TGG", "Y": "TAC", "V": "GTG",
}
if set(_CODON_TABLE) != _AA_LETTERS:
    raise AssertionError("fastdna.datasets: _CODON_TABLE must cover exactly the 20 standard amino acids")


def _verify_hxb2_reference() -> None:
    """Static sanity check for `_HXB2_PR`, reused verbatim from
    `scratch/hiv_repro/build_hiv_repro.py`'s own `verify_reference()`: the
    reference must read as the wild-type residue PI_DataSet.txt's row 1
    `CompMutList` itself reports for its five listed mutations (D30N,
    M46I, R57G, L63P, N88D). Pure string check against the hardcoded
    reference above -- no file I/O -- so it costs nothing and is called
    once, at import time, below.
    """
    expected = {30: "D", 46: "M", 57: "R", 63: "L", 88: "N"}
    for position, wild_type in expected.items():
        actual = _HXB2_PR[position - 1]
        if actual != wild_type:
            raise AssertionError(
                f"fastdna.datasets: HXB2 protease reference sanity check failed at position "
                f"{position}: expected wild-type {wild_type!r} (from CompMutList D30N, M46I, "
                f"R57G, L63P, N88D), got {actual!r}. _HXB2_PR was edited incorrectly."
            )


@dataclass(frozen=True)
class HivResistanceCohort:
    """A labeled HIV-1 protease drug-resistance cohort returned by
    :func:`load_hiv_resistance`.

    `frozen=True`, matching `AmrCohort` and
    `fastdna.cohort_counts.CohortCounts` -- see `AmrCohort`'s own
    docstring for the rationale.

    Attributes
    ----------
    drug : str
        The PI drug requested, verbatim (one of `_HIV_DRUG_COLUMNS`).
    fold_cutoff : float
        The fold-resistance cutoff actually used (either the caller's own,
        or the documented per-drug default).
    paths : tuple of str
        Local paths to each selected sample's back-translated 297 bp DNA
        FASTA, cached under the resolved cache directory. Ready for
        `fastdna.count()`/`fastdna.translate`/a
        `ProteinKmerVectorizer`-shaped pipeline.
    phenotype : numpy.ndarray of int64, shape (n,)
        `1` if `fold_resistance > fold_cutoff`, else `0`, aligned with
        `paths`.
    fold_resistance : numpy.ndarray of float64, shape (n,)
        The continuous fold-resistance value HIVDB reported for `drug`,
        aligned with `paths` -- the regression-task counterpart of
        `phenotype`.
    sample_ids : tuple of str
        Each sample's HIVDB `SeqID`, aligned with `paths`.
    """

    drug: str
    fold_cutoff: float
    paths: tuple[str, ...]
    phenotype: np.ndarray
    fold_resistance: np.ndarray
    sample_ids: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.sample_ids)


def _iter_hiv_rows(path: Path, drug: str):
    """Yields `(seq_id, fold_value, positions)` for every PI_DataSet.txt
    row with a numeric `drug` value and no mixture/missing/ambiguous P
    column. `positions` is a list of 99 single-letter strings, each `-`
    (matches the HXB2 reference) or a standard amino acid -- the same
    "clean row" definition `scratch/hiv_repro/build_hiv_repro.py::
    parse_dataset` uses: a mixture (e.g. `"RK"`), `.` (unsequenced), `X`
    (ambiguous) or `*` (a stop/indel artifact) at any of the 99 positions
    drops the whole row, rather than resolved by picking one letter, so
    every reconstructed sequence is the single, unambiguous sequence HIVDB
    itself reported for that isolate.
    """
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        try:
            p_indices = [header.index(f"P{i}") for i in range(1, 100)]
        except ValueError as exc:
            raise ValueError(f"PI_DataSet.txt header is missing an expected P-column: {exc}") from exc
        drug_index = header.index(drug)
        seqid_index = header.index("SeqID")

        for row in reader:
            raw_fold = row[drug_index]
            if not raw_fold or raw_fold == "NA":
                continue
            try:
                fold = float(raw_fold)
            except ValueError:
                continue
            positions = [row[i] for i in p_indices]
            if not all(v == "-" or (len(v) == 1 and v in _AA_LETTERS) for v in positions):
                continue
            yield row[seqid_index], fold, positions


def _reconstruct_protein(positions) -> str:
    return "".join(_HXB2_PR[i] if value == "-" else value for i, value in enumerate(positions))


def _back_translate(protein: str) -> str:
    return "".join(_CODON_TABLE[aa] for aa in protein)


def load_hiv_resistance(
    drug: str,
    *,
    n_samples: Optional[int] = None,
    fold_cutoff: Optional[float] = None,
    random_state: int = 0,
    cache_dir: Optional[_PathLike] = None,
) -> HivResistanceCohort:
    """Loads a real, labeled HIV-1 protease drug-resistance cohort from
    the Stanford HIV Drug Resistance Database.

    Downloads (or reuses a cached copy of) `PI_DataSet.txt`, drops rows
    with an ambiguous/mixture/unsequenced call at any of the 99 protease
    positions or no numeric fold-resistance value for `drug` (see
    `_iter_hiv_rows`), binarizes at `fold_cutoff`, optionally
    stratified-subsamples to `n_samples`, reconstructs each selected
    sample's 99-residue protease sequence against the HXB2-derived
    reference (`_HXB2_PR`, verified at import time by
    `_verify_hxb2_reference`), and writes each one's back-translated
    297 bp DNA sequence as its own FASTA file into the cache -- the same
    pipeline `scratch/hiv_repro/build_hiv_repro.py` used by hand, with the
    back-translation table transcribed verbatim from it (see the module
    docstring's "Design decisions").

    Parameters
    ----------
    drug : str
        One of the 8 PI drug columns in `PI_DataSet.txt`
        (`_HIV_DRUG_COLUMNS`: `FPV`, `ATV`, `IDV`, `LPV`, `NFV`, `SQV`,
        `TPV`, `DRV`). An unrecognized value raises `ValueError` listing
        them.
    n_samples : int, optional
        `None` (the default) keeps every clean, `drug`-tested row.
        Otherwise, a stratified random sample of exactly `n_samples`
        sequences whose resistant/susceptible proportions match the full
        filtered pool's own. Raises `ValueError` if larger than the pool.
    fold_cutoff : float, optional
        The fold-resistance value above which a sample is labeled
        resistant. `None` (the default) uses the documented per-drug
        default in `_HIV_FOLD_CUTOFFS` if one exists for `drug` (currently
        only `NFV`, at `4.0`); for any other drug, `None` raises
        `ValueError` rather than guessing.
    random_state : int, default 0
        Seeds the stratified sample; the same `drug`/`n_samples`/
        `fold_cutoff`/`random_state` always returns the same sequences in
        the same order.
    cache_dir : str or os.PathLike, optional
        Where downloads are cached. See the module docstring's "Caching"
        section for the default and the `FASTDNA_DATA_CACHE` override.

    Returns
    -------
    HivResistanceCohort

    Raises
    ------
    ValueError
        `drug` is not one of the 8 PI columns, `fold_cutoff` is `None` and
        `drug` has no documented default, `fold_cutoff` is not a positive
        number, no clean row is available for `drug`, or `n_samples` is
        invalid.
    RuntimeError
        A download failed after retries (see `_download_bytes`).
    """
    if drug not in _HIV_DRUG_COLUMNS:
        available = ", ".join(_HIV_DRUG_COLUMNS)
        raise ValueError(
            f"load_hiv_resistance(): unknown drug {drug!r}. Available PI drug columns in "
            f"PI_DataSet.txt: {available}"
        )

    if fold_cutoff is None:
        fold_cutoff = _HIV_FOLD_CUTOFFS.get(drug)
        if fold_cutoff is None:
            documented = ", ".join(sorted(_HIV_FOLD_CUTOFFS))
            raise ValueError(
                f"load_hiv_resistance(): no documented fold-resistance cutoff for drug "
                f"{drug!r} (only {documented} has one built in, from the PhenoSense/Monogram "
                "clinical cutoff Rhee et al. 2006 PNAS used to build this dataset). Pass "
                "fold_cutoff explicitly for this drug."
            )
    elif isinstance(fold_cutoff, bool) or not isinstance(fold_cutoff, (int, float)) or fold_cutoff <= 0:
        raise ValueError(f"load_hiv_resistance(): fold_cutoff must be a positive number, got {fold_cutoff!r}")

    cache_root = _resolve_cache_dir(cache_dir)
    hiv_dir = cache_root / "hiv"
    dataset_path = hiv_dir / "PI_DataSet.txt"
    _fetch_cached(
        _HIVDB_PI_DATASET_URL,
        dataset_path,
        validate=lambda data: data.startswith(b"SeqID\t"),
    )

    records = [
        {
            "seq_id": seq_id,
            "fold": fold,
            "label": 1 if fold > fold_cutoff else 0,
            "protein": _reconstruct_protein(positions),
        }
        for seq_id, fold, positions in _iter_hiv_rows(dataset_path, drug)
    ]
    if not records:
        raise ValueError(
            f"load_hiv_resistance(): no clean, {drug}-tested row found in PI_DataSet.txt "
            "(every row either had no numeric fold-resistance value for this drug, or a "
            "mixture/missing/ambiguous amino acid at one of the 99 protease positions)."
        )

    selected = _stratified_sample(
        records,
        lambda record: record["label"] == 1,
        n_samples,
        random_state,
        caller="load_hiv_resistance()",
    )

    samples_dir = hiv_dir / "samples"
    paths = []
    for record in selected:
        dest = samples_dir / f"{record['seq_id']}.fasta"
        if not (dest.exists() and dest.stat().st_size > 0):
            dna = _back_translate(record["protein"])
            _write_cache_file(dest, f">{record['seq_id']}\n{dna}\n".encode("ascii"))
        paths.append(str(dest))

    phenotype = np.array([record["label"] for record in selected], dtype=np.int64)
    fold_resistance = np.array([record["fold"] for record in selected], dtype=np.float64)

    return HivResistanceCohort(
        drug=drug,
        fold_cutoff=float(fold_cutoff),
        paths=tuple(paths),
        phenotype=phenotype,
        fold_resistance=fold_resistance,
        sample_ids=tuple(record["seq_id"] for record in selected),
    )


_verify_hxb2_reference()
