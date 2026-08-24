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

#: How far above the candidate valley floor the spectrum must climb before
#: that climb is accepted as "the coverage peak", rather than dismissed as
#: noise on the error peak's descending tail. A single-count uptick (one
#: more distinct k-mer at depth d+1 than at depth d) is routine noise in any
#: real spectrum and must not by itself end the search for the floor; a
#: genuine coverage peak, by contrast, is typically many times taller than
#: the valley between it and the error peak. 2x is a conservative line
#: between those two shapes -- low enough to accept a real (if modest)
#: coverage peak, high enough that ordinary single- or few-count noise on
#: the descent can never be mistaken for one.
RISE_FACTOR = 2.0


def suggest_min_count(spectrum, default=DEFAULT_MIN_COUNT):
    """Finds the valley floor between the error peak and the coverage peak.

    `spectrum` maps depth (an int >= 1) to the number of distinct k-mers
    observed at that depth -- e.g. `KmerCounts.spectrum()`, or any dict
    shaped like it, such as `{1: 50_000, 2: 30_000, 3: 5_000, 4: 1_000,
    5: 800, 6: 4_000, 7: 9_000, ...}`.

    Walks depths in ascending order, tracking the lowest count seen so far
    (the current valley-floor candidate). A depth is only accepted as the
    end of the valley -- i.e. the point where the *coverage* peak begins --
    once the spectrum climbs to at least `RISE_FACTOR` times that
    candidate's count. This means a single noisy uptick partway down the
    error peak's tail (`{..., 40: 500, 41: 501, 42: 400, ...}`) does not end
    the search: the walk keeps looking for a lower floor, because 501 is not
    a *substantial* rise over 500. Only a climb that actually looks like a
    second peak stops it. Depths between the lowest and highest key that
    are absent from `spectrum` are treated as zero distinct k-mers (not
    skipped), so a single sample with no k-mers at some depth cannot distort
    the shape by making its neighbors look adjacent.

    Returns the valley **floor**'s own depth, not one above it: since
    `min_count` is an inclusive lower bound, passing this return value
    straight to `count(..., min_count=...)` *keeps* the k-mers at the
    valley floor rather than discarding them as part of the error peak --
    deliberately erring towards keeping ambiguous, boundary-depth k-mers
    rather than discarding data that might belong to the real sample.

    Falls back to `default` whenever the spectrum's shape is not
    unambiguous: fewer than 3 distinct depths present, a walk that never
    finds a lower floor before the data runs out (no error peak to walk
    past), or one that never climbs back up by `RISE_FACTOR` from whatever
    floor it does find (no coverage peak visible in the sample, even if the
    tail is not perfectly monotonic). A wrong threshold here is worse than
    no threshold, so every case this function is not confident about returns
    the documented default instead of a guess.
    """
    if not spectrum:
        return default

    depths_present = sorted(spectrum)
    if len(depths_present) < 3:
        return default

    # Walk the *observed* depths only, never the contiguous range -- a
    # single adapter-dimer k-mer at depth 10^7+ would otherwise force a
    # multi-gigabyte materialized list. A depth with zero distinct k-mers is
    # still routine in a small sample and must count as a real (zero-height)
    # point in the shape, not be skipped so its neighbors look adjacent --
    # so a gap between consecutive observed depths is handled explicitly
    # below, exactly as the contiguous walk would have: the gap's first
    # missing depth (count 0) becomes a new floor (it is lower than any
    # positive floor), and the very next point -- present or missing --
    # trivially clears `0 * RISE_FACTOR`, ending the walk there.
    min_depth = depths_present[0]

    floor_depth = min_depth
    floor_count = spectrum[min_depth]
    valley_depth = None
    prev_depth = min_depth
    for d in depths_present[1:]:
        if d > prev_depth + 1:
            # A gap: depth `prev_depth + 1` exists in the shape with zero
            # distinct k-mers. If the current floor is already 0 (an
            # explicit zero entry earlier), the gap's zero is not lower and
            # itself clears the (zero) rise threshold, ending the walk at
            # that existing floor; otherwise the gap's first missing depth
            # is the new, lowest-possible floor, and the next point ends
            # the walk there.
            valley_depth = floor_depth if floor_count == 0 else prev_depth + 1
            break
        count = spectrum[d]
        if count < floor_count:
            # A new, lower candidate floor -- keep walking down the error
            # peak's tail.
            floor_depth, floor_count = d, count
        elif count >= floor_count * RISE_FACTOR:
            # A substantial climb from the current floor: accept it as the
            # start of the coverage peak, and the current floor as the
            # valley.
            valley_depth = floor_depth
            break
        # Otherwise: not lower, but not a substantial rise either -- this is
        # noise (e.g. a single-count uptick), not the end of the valley.
        # Keep walking without updating the floor.
        prev_depth = d

    if valley_depth is None:
        # Never found a climb big enough to call a coverage peak.
        return default
    if valley_depth == min_depth:
        # The very first depth in range was already the lowest, and the
        # next depth alone cleared the rise threshold: there was no actual
        # descent to walk down, i.e. no error peak to walk past.
        return default

    return valley_depth
