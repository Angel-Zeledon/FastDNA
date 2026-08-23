"""Local-minimum ("valley") detection over a k-mer frequency spectrum.

A sequenced sample's frequency spectrum has two peaks: a large one at
frequency 1-2 (machine errors) and another at the sample's true coverage
depth, with a valley between them. The correct `min_count` threshold sits
in that valley, and it differs per sample -- there is no universal 5. This
module picks it from the data instead of asking the user to eyeball a plot.

Kept in pure Python per the design doc (§9.2/§9.7): the Rust core already
computes the histogram (`KmerCounter::generate_histogram`, exposed as
`KmerCounts.spectrum()`); turning it into a threshold is array arithmetic,
not FFI-boundary work.
"""

#: Used when the spectrum carries too little signal to locate a valley (see
#: `suggest_min_count`). Not a recommendation in its own right -- callers
#: relying on it routinely are exactly the "eyeball a universal 5" problem
#: this module exists to avoid.
DEFAULT_MIN_COUNT = 2


def suggest_min_count(spectrum, default=DEFAULT_MIN_COUNT):
    """Finds the local minimum between the error peak and the coverage peak.

    `spectrum` maps depth (an int >= 1) to the number of distinct k-mers
    observed at that depth -- e.g. `KmerCounts.spectrum()`, or any dict
    shaped like it, such as `{1: 50_000, 2: 30_000, 3: 5_000, 4: 1_000,
    5: 800, 6: 4_000, 7: 9_000, ...}`.

    Walks depths in ascending order and follows the initial descent (the
    tail of the error peak) until the trend reverses -- the first point
    where the next depth's count is no longer lower than the current one.
    That reversal point is the valley floor, and its depth is the
    suggested `min_count`.

    Falls back to `default` when the spectrum does not carry enough signal
    to find a valley: fewer than 3 distinct depths, a spectrum that never
    stops decreasing (no coverage peak visible in the sample), or one that
    never starts decreasing (no error peak to walk past). Guessing a
    threshold from a spectrum shaped like that would be worse than a
    documented fallback.
    """
    if not spectrum:
        return default

    depths = sorted(spectrum)
    if len(depths) < 3:
        return default

    counts = [spectrum[d] for d in depths]

    i = 0
    while i + 1 < len(counts) and counts[i + 1] < counts[i]:
        i += 1

    if i == 0 or i == len(counts) - 1:
        # The trend never reversed (or reversed immediately): no distinct
        # valley to report.
        return default

    return depths[i]
