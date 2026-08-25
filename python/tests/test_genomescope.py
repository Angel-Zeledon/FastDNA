"""Tests for fastdna.genomescope -- GenomeScope-style genome profiling from
a k-mer frequency spectrum.

Most of these tests fit *synthetic* spectra generated from the same family
of models `profile_genome` fits (a negative-binomial mixture), because that
is the only way to know the true genome size / heterozygosity / repeat
fraction an answer should be compared against: no real sequencing run comes
with those numbers attached. The generator is deliberately not identical to
the fitted model -- it adds an exponential error component the model does
not include as a fitted term, truncates the tail, and perturbs every bin --
so the recovery test is not a pure algebraic identity.

The end-to-end test is the opposite: real `fastdna.count()` over a real
(tiny) FASTQ file, asserting only types and sanity bounds. A 5 kb synthetic
"genome" is not a genome, and no tolerance on its estimated size would mean
anything.
"""
from __future__ import annotations

import pathlib
import random
import time

import pytest

pytest.importorskip("scipy")
np = pytest.importorskip("numpy")

from scipy.stats import nbinom  # noqa: E402

from fastdna.genomescope import GenomeProfile, plot_spectrum_fit, profile_genome  # noqa: E402

# --------------------------------------------------------------------------
# Synthetic spectrum with known ground truth
# --------------------------------------------------------------------------

K = 21
TRUE_COVERAGE = 25.0  # haploid coverage: the homozygous peak sits at 2x this
TRUE_HETEROZYGOSITY = 0.01  # fraction of bases heterozygous
TRUE_UNIQUE_KMERS = 2_000_000  # haploid, non-repetitive genome positions
DISPERSION = 20.0  # negative-binomial size parameter per haploid copy
COPY3_KMERS = 100_000  # distinct k-mers present at total copy number 3
COPY4_KMERS = 50_000  # distinct k-mers present at total copy number 4
MAX_SYNTHETIC_DEPTH = 120

# Error component: distinct k-mers created by sequencing errors, decaying
# exponentially from depth 1. Not part of the fitted model -- the module has
# to survive it, and account for it in `error_rate`, without fitting it.
ERROR_SCALE = 4_000_000.0
ERROR_DECAY = 0.35


def _amplitudes():
    """The four component amplitudes (expected number of *distinct* k-mers
    at total copy number 1, 2, 3, 4) implied by the constants above.

    A heterozygous site makes both haplotypes' k-mers unique, so each
    heterozygous haploid position contributes *two* distinct k-mers at 1x
    haploid coverage; a homozygous position contributes one at 2x. The
    probability that a k-mer window spans no heterozygous site is
    `(1 - r)**k`.
    """
    homozygous_fraction = (1.0 - TRUE_HETEROZYGOSITY) ** K
    a1 = 2.0 * TRUE_UNIQUE_KMERS * (1.0 - homozygous_fraction)
    a2 = TRUE_UNIQUE_KMERS * homozygous_fraction
    return a1, a2, float(COPY3_KMERS), float(COPY4_KMERS)


def synthetic_spectrum(noise=0.02, seed=20260824):
    """A full `{depth: distinct k-mers}` spectrum built from the constants
    above: four negative-binomial components at 1x/2x/3x/4x haploid
    coverage plus an exponential error tail, every bin perturbed by `noise`
    (relative, seeded) so the fit is not handed back exactly what it
    assumes.

    Every depth in `1..MAX_SYNTHETIC_DEPTH` is present (zeros included) --
    a missing depth means "no k-mers there" to the valley detector, and
    dropping zero bins would change the shape it sees.
    """
    rng = np.random.default_rng(seed)
    depths = np.arange(1, MAX_SYNTHETIC_DEPTH + 1)
    p = DISPERSION / (DISPERSION + TRUE_COVERAGE)

    counts = np.zeros(depths.shape, dtype=float)
    for amplitude, copies in zip(_amplitudes(), (1, 2, 3, 4)):
        counts += amplitude * nbinom.pmf(depths, copies * DISPERSION, p)
    counts += ERROR_SCALE * ERROR_DECAY**depths

    counts *= 1.0 + noise * rng.standard_normal(counts.shape)
    counts = np.clip(counts, 0.0, None)
    return {int(d): int(round(c)) for d, c in zip(depths, counts)}


def expected_truth():
    """Ground truth for the synthetic spectrum, derived from the same
    constants rather than hard-coded.

    Genome size is in *haploid* k-mer positions: a k-mer at total copy
    number `i` occupies `i / 2` haploid positions, so the whole genome is
    `sum(i * a_i) / 2`. This is the same quantity
    `total non-error k-mer mass / (ploidy * haploid coverage)` recovers,
    since the mass of component `i` is `a_i * i * coverage`.

    The error rate inverts the generator: the error component's k-mer
    *mass* (sum of depth x k-mers) as a fraction of all k-mer mass is the
    fraction of k-mers containing at least one wrong base, which is
    `1 - (1-e)**k` for a per-base rate `e`.
    """
    a1, a2, a3, a4 = _amplitudes()
    weighted = 1 * a1 + 2 * a2 + 3 * a3 + 4 * a4
    depths = np.arange(1, MAX_SYNTHETIC_DEPTH + 1)
    error_mass = float((depths * ERROR_SCALE * ERROR_DECAY**depths).sum())
    genomic_mass = TRUE_COVERAGE * weighted
    error_kmer_fraction = error_mass / (error_mass + genomic_mass)
    return {
        "genome_size": weighted / 2.0,
        "unique_size": a1 / 2.0 + a2,
        "repeat_fraction": (3 * a3 + 4 * a4) / weighted,
        "error_rate": 1.0 - (1.0 - error_kmer_fraction) ** (1.0 / K),
    }


# --------------------------------------------------------------------------
# Parameter recovery
# --------------------------------------------------------------------------


def test_recovers_known_parameters_from_a_synthetic_spectrum():
    profile = profile_genome(synthetic_spectrum(), k=K)
    truth = expected_truth()

    assert profile.converged, profile.model_fit["convergence_reason"]

    # Coverage is the best-determined parameter in the fit: it is fixed by
    # *where* the peaks are, and the peaks are where the data is. 2% of 25x
    # is half a depth bin -- tight enough that a het-vs-hom peak
    # misassignment (which would land at 12.5x or 50x, the failure that
    # actually matters) cannot hide inside it, loose enough to absorb the
    # error tail leaking into the fitted range's left edge.
    assert profile.haploid_coverage == pytest.approx(TRUE_COVERAGE, rel=0.02)

    # Genome size and unique size follow from the observed k-mer mass
    # divided by the recovered coverage, so they carry coverage's error
    # plus the genomic mass that falls outside the fitted range entirely
    # (below the cutoff, above the deepest generated bin). Both are
    # one-sided, so the recovered size is expected to be slightly low; 5%
    # bounds that bias without admitting a qualitatively wrong answer.
    assert profile.genome_size_bp == pytest.approx(truth["genome_size"], rel=0.05)
    assert profile.unique_size_bp == pytest.approx(truth["unique_size"], rel=0.05)

    # Heterozygosity is the most fragile output: it comes from the *ratio*
    # of the 1x and 2x amplitudes, and the 1x (heterozygous) component is
    # the one that sits closest to the error tail, so a little leakage there
    # moves it much more than it moves coverage or genome size. It is also
    # a k-th root, which compresses amplitude error but never removes it.
    # 15% relative on a 1% rate means "0.0085-0.0115": the same rate to the
    # precision anyone quotes a heterozygosity at.
    assert profile.heterozygosity == pytest.approx(TRUE_HETEROZYGOSITY, rel=0.15)

    # Repeat fraction is a fraction, so an absolute tolerance is the honest
    # one: 0.02 on a true 0.11 keeps "roughly a tenth of this genome is
    # repetitive" while allowing the 3x/4x components (the smallest, most
    # tail-sensitive ones) to trade a little amplitude between themselves.
    assert profile.repeat_fraction == pytest.approx(truth["repeat_fraction"], abs=0.02)

    # The error rate is a per-base rate inferred from the k-mer mass below
    # the error/signal cutoff, and that split is decided by a single integer
    # depth: it is exact only if no error k-mer sits above the cutoff and no
    # genuine k-mer below it. Both happen, so 25% relative (0.001-0.0017
    # around a true ~0.14%) is the right band -- it pins the order of
    # magnitude, which is the claim a k-mer-spectrum error rate supports.
    assert profile.error_rate == pytest.approx(truth["error_rate"], rel=0.25)

    assert profile.k == K
    assert profile.model_fit["rmse"] >= 0.0
    assert profile.model_fit["optimizer_success"] is True


def test_reports_the_fitted_parameters_for_inspection():
    profile = profile_genome(synthetic_spectrum(), k=K)

    fit = profile.model_fit
    assert fit["haploid_coverage"] == pytest.approx(profile.haploid_coverage)
    assert len(fit["amplitudes"]) == len(fit["copy_numbers"]) == 4
    assert fit["fit_min_depth"] >= 1
    assert fit["max_coverage"] > fit["fit_min_depth"]
    assert fit["relative_rmse"] >= 0.0
    assert isinstance(fit["optimizer_message"], str)


# --------------------------------------------------------------------------
# Honesty: refuse, don't guess
# --------------------------------------------------------------------------


def test_a_spectrum_with_no_coverage_peak_is_refused():
    # Pure error decay: a shallow run where the genome's own k-mers never
    # rise above the error component. There is no peak to fit, and a
    # confidently-wrong genome size would be worse than no answer.
    spectrum = {1: 100_000, 2: 30_000, 3: 9_000, 4: 2_700, 5: 800, 6: 240, 7: 70, 8: 20}

    with pytest.raises(ValueError) as excinfo:
        profile_genome(spectrum, k=K)

    message = str(excinfo.value)
    assert "coverage peak" in message
    assert "10x" in message


def test_polyploid_is_refused_rather_than_faked():
    with pytest.raises(NotImplementedError) as excinfo:
        profile_genome(synthetic_spectrum(), k=K, ploidy=3)

    message = str(excinfo.value)
    assert "diploid" in message
    assert "A2" in message  # points at the roadmap item that would add it


def test_a_spectrum_dict_without_k_is_refused():
    # k is not recoverable from a bare spectrum, and both heterozygosity and
    # the error rate are k-dependent -- guessing one would silently produce
    # wrong biology.
    with pytest.raises(ValueError) as excinfo:
        profile_genome(synthetic_spectrum())

    assert "k" in str(excinfo.value)


def test_an_unfittable_spectrum_reports_converged_false_rather_than_raising():
    # An adversarial shape: a normal-looking error peak and valley (so the
    # coverage-peak check passes), followed by a sawtooth that alternates
    # between 100 and 400,000 distinct k-mers. No mixture of negative
    # binomials can follow that, so the fitted curve misses the data by
    # roughly its own magnitude. The result must still be a GenomeProfile
    # -- with `converged=False` and the reason recorded -- so a caller can
    # see the fit failed instead of getting an exception (or, worse, a
    # confident number).
    spectrum = {1: 50_000, 2: 10_000, 3: 2_000, 4: 400, 5: 100}
    for depth in range(6, 41):
        spectrum[depth] = 400_000 if depth % 2 == 0 else 100

    profile = profile_genome(spectrum, k=K)

    assert isinstance(profile, GenomeProfile)
    assert profile.converged is False
    assert profile.model_fit["relative_rmse"] > 0.5
    assert isinstance(profile.model_fit["convergence_reason"], str)
    assert profile.model_fit["convergence_reason"]


# --------------------------------------------------------------------------
# The max_coverage cap
# --------------------------------------------------------------------------


def test_a_single_enormous_depth_repeat_kmer_does_not_dominate_the_fit():
    # One k-mer at depth 10,000,000 (an adapter dimer, a rDNA array, a
    # contaminant): without a cap on the fitted range this both materializes
    # a 10-million-bin array and drags every fitted parameter toward it.
    baseline = profile_genome(synthetic_spectrum(), k=K)

    spiked = dict(synthetic_spectrum())
    spiked[10_000_000] = 5

    started = time.perf_counter()
    profile = profile_genome(spiked, k=K)
    elapsed = time.perf_counter() - started

    # Structural: the fitted range must stay a small multiple of the
    # coverage peak, not follow the outlier.
    assert profile.model_fit["max_coverage"] < 10_000
    assert profile.model_fit["kmers_above_max_coverage"] == 5 * 10_000_000

    # ... and the outlier must not move the answer. Tighter than the
    # recovery test's tolerances, because this compares two runs of the same
    # code over the same spectrum: the only difference between them is the
    # one spiked bin, so anything beyond rounding here means the outlier
    # reached the fit.
    assert profile.haploid_coverage == pytest.approx(baseline.haploid_coverage, rel=0.01)
    assert profile.genome_size_bp == pytest.approx(baseline.genome_size_bp, rel=0.01)

    # Generous, but a 10-million-bin fit is minutes or an OOM, not seconds.
    assert elapsed < 20.0


def test_max_coverage_can_be_set_explicitly():
    profile = profile_genome(synthetic_spectrum(), k=K, max_coverage=80)
    assert profile.model_fit["max_coverage"] == 80


# --------------------------------------------------------------------------
# End to end over a real FASTQ
# --------------------------------------------------------------------------


def write_fastq(tmp_path: pathlib.Path, name: str, reads: list[str]) -> pathlib.Path:
    p = tmp_path / name
    p.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    return p


def simulated_reads(rng: random.Random, genome: str, n_reads: int, read_length: int, error_rate: float):
    """Reads sampled uniformly from `genome` with per-base substitution
    errors -- the errors matter: an error-free read set has no error peak,
    hence no valley, hence nothing for the profiler to separate signal from.
    """
    reads = []
    for _ in range(n_reads):
        start = rng.randrange(0, len(genome) - read_length)
        read = list(genome[start : start + read_length])
        for i in range(read_length):
            if rng.random() < error_rate:
                read[i] = rng.choice([b for b in "ACGT" if b != read[i]])
        reads.append("".join(read))
    return reads


def test_profiles_a_real_fastq_end_to_end(tmp_path):
    rng = random.Random(9001)
    genome = "".join(rng.choice("ACGT") for _ in range(5_000))
    reads = simulated_reads(rng, genome, n_reads=2_000, read_length=100, error_rate=0.005)
    path = write_fastq(tmp_path, "sample.fastq", reads)

    profile = profile_genome(str(path), k=K)

    # Types and sanity bounds. A 5 kb random string sequenced from one
    # haplotype is not a diploid genome -- it has no heterozygosity and no
    # repeats, so most of the model's assumptions are violated and most of
    # its outputs are not meaningful here. What this test checks is that the
    # whole path (FASTQ -> fastdna.count -> spectrum -> valley -> fit ->
    # profile) runs and returns well-typed, in-range values.
    assert isinstance(profile, GenomeProfile)
    assert profile.k == K
    assert isinstance(profile.genome_size_bp, int) and profile.genome_size_bp > 0
    assert isinstance(profile.unique_size_bp, int) and profile.unique_size_bp >= 0
    assert profile.haploid_coverage > 0.0
    assert 0.0 <= profile.heterozygosity <= 1.0
    assert 0.0 <= profile.repeat_fraction <= 1.0
    assert 0.0 <= profile.error_rate <= 1.0
    assert isinstance(profile.converged, bool)
    assert "rmse" in profile.model_fit

    # The one quantitative claim this fixture *can* support: 2,000 reads
    # over a known 5,000 bp template is deep enough (~32x in k-mer space)
    # for the coverage peak to be real, so the genome size must land within
    # a factor of two of 5,000 rather than anywhere at all. It is asserted
    # as a factor of two, not a percentage, deliberately: with no
    # heterozygous shoulder the model cannot tell "coverage c, homozygous"
    # from "coverage c/2, fully heterozygous" from the data, so a factor of
    # two is exactly the resolution the input supports -- and a wider miss
    # would mean the peak itself was found wrong.
    assert 2_500 <= profile.genome_size_bp <= 10_000


def test_accepts_a_kmer_counts_object_and_uses_its_own_k(tmp_path):
    import fastdna

    rng = random.Random(4242)
    genome = "".join(rng.choice("ACGT") for _ in range(5_000))
    reads = simulated_reads(rng, genome, n_reads=2_000, read_length=100, error_rate=0.005)
    path = write_fastq(tmp_path, "sample.fastq", reads)

    counts = fastdna.count(str(path), k=17)
    profile = profile_genome(counts)

    assert profile.k == 17


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def test_plot_spectrum_fit_draws_onto_an_axes():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spectrum = synthetic_spectrum()
    profile = profile_genome(spectrum, k=K)

    _, ax = plt.subplots()
    returned = plot_spectrum_fit(profile, spectrum, ax=ax)

    assert returned is ax
    assert ax.lines  # the fitted curve and its components
    assert ax.get_legend() is not None
    plt.close("all")
